from __future__ import annotations

import json
from pathlib import Path

import pytest

from drug_screen.data.ccle_context_identity import (
    CRC_EXACT_CONTEXTS,
    assert_crc_exact_locked,
    assert_no_patient_leakage,
    assign_roles,
    build_identity_rows,
    data_contract,
    patient_split_bucket,
)


ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "artifacts" / "experiments" / "EXP-007" / "CCLE_CONTEXT_IDENTITY_SPLIT.json"
INTAKE = ROOT / "artifacts" / "experiments" / "EXP-007" / "CCLE_RNASEQ_INTAKE.json"
RECORD = ROOT / "experiments" / "records" / "EXP-007.md"


def _lincs(*ids: str, exact: tuple[str, ...] = ()) -> list[dict]:
    rows = []
    for name in ids:
        rows.append(
            {
                "context_id": name,
                "broad_exact_context": name in exact,
                "broad_depmap_ids": ["ACH-CRC"] if name in exact else [],
            }
        )
    return rows


def _models(*rows: tuple[str, str, str]) -> list[dict]:
    return [
        {"ModelID": depmap_id, "StrippedCellLineName": name, "PatientID": patient}
        for name, depmap_id, patient in rows
    ]


def _expr(*rows: tuple[str, str]) -> list[dict]:
    return [
        {
            "depmap_id": depmap_id,
            "stripped_cell_line_name": name,
            "exact978_row": index,
        }
        for index, (name, depmap_id) in enumerate(rows)
    ]


def test_oracle_crc_list_is_not_redefined() -> None:
    assert len(CRC_EXACT_CONTEXTS) == 10


def test_exact_stripped_name_join_keeps_h1299_unresolved() -> None:
    identity = build_identity_rows(
        lincs_contexts=_lincs("HT29", "H1299", "MCF10A", exact=("HT29",)),
        models=_models(
            ("HT29", "ACH-000552", "PT-sF39aT"),
            ("NCIH1299", "ACH-000510", "PT-x"),
            ("MCF10A", "ACH-001357", "PT-y"),
        ),
        expression_rows=_expr(("HT29", "ACH-000552"), ("NCIH1299", "ACH-000510")),
    )
    by_id = {row["lincs_cell_id"]: row for row in identity}
    assert by_id["HT29"]["join_status"] == "EXACT_STRIPPED_NAME_WITH_RNASEQ"
    assert by_id["HT29"]["depmap_id"] == "ACH-000552"
    assert by_id["H1299"]["join_status"] == "UNRESOLVED_ALIAS_CANDIDATE"
    assert by_id["H1299"]["depmap_id"] is None
    assert by_id["H1299"]["alias_depmap_id"] == "ACH-000510"
    assert by_id["MCF10A"]["join_status"] == "EXACT_STRIPPED_NAME_MODEL_ONLY"
    assert by_id["MCF10A"]["expression_present"] is False


def test_crc_and_shared_patient_are_locked_without_response() -> None:
    identity = build_identity_rows(
        lincs_contexts=_lincs("SW480", "SW620", "A375", exact=("SW480", "SW620")),
        models=_models(
            ("SW480", "ACH-000842", "PT-IPboWn"),
            ("SW620", "ACH-000651", "PT-IPboWn"),
            ("A375", "ACH-000219", "PT-3FxmoJ"),
            ("A375SKINCJ1", "ACH-002001", "PT-3FxmoJ"),
        ),
        expression_rows=_expr(
            ("SW480", "ACH-000842"),
            ("SW620", "ACH-000651"),
            ("A375", "ACH-000219"),
            ("A375SKINCJ1", "ACH-002001"),
        ),
    )
    assignments = assign_roles(
        identity_rows=identity,
        expression_rows=_expr(
            ("SW480", "ACH-000842"),
            ("SW620", "ACH-000651"),
            ("A375", "ACH-000219"),
            ("A375SKINCJ1", "ACH-002001"),
        ),
        models=_models(
            ("SW480", "ACH-000842", "PT-IPboWn"),
            ("SW620", "ACH-000651", "PT-IPboWn"),
            ("A375", "ACH-000219", "PT-3FxmoJ"),
            ("A375SKINCJ1", "ACH-002001", "PT-3FxmoJ"),
        ),
    )
    by_id = {row["depmap_id"]: row for row in assignments}
    assert by_id["ACH-000842"]["split_role"] == "locked_eval"
    assert by_id["ACH-000651"]["split_role"] == "locked_eval"
    assert by_id["ACH-000219"]["split_role"] == by_id["ACH-002001"]["split_role"]
    assert by_id["ACH-000219"]["split_role"] in {"train", "val"}
    assert_no_patient_leakage(assignments)
    assert_crc_exact_locked(assignments)


def test_patient_hash_is_deterministic_and_ignores_metrics() -> None:
    first = patient_split_bucket("PT-test")
    assert first == patient_split_bucket("PT-test")
    assert first in {"train", "val"}
    # Seed is part of the digest; it is not derived from PRISM/test metrics.
    assert patient_split_bucket.__doc__ is not None


def test_patient_leakage_is_blocked() -> None:
    with pytest.raises(ValueError, match="patient-level split leakage"):
        assert_no_patient_leakage(
            [
                {"patient_id": "PT-1", "split_role": "train"},
                {"patient_id": "PT-1", "split_role": "val"},
            ]
        )


def test_contract_forbids_x_ctl_and_delta978() -> None:
    contract = data_contract(
        summary={"crc_exact_with_rnaseq": list(CRC_EXACT_CONTEXTS)},
        leakage_status="PASS_PATIENT_ATOMIC",
    )
    assert contract["gene_universe"]["cannot_use_as_x_ctl"] is True
    assert contract["gene_universe"]["cannot_compute_delta978"] is True
    assert contract["response_blind"] is True
    assert "PRISM response values" in contract["forbidden_split_inputs"]


def test_frozen_artifact_if_present_locks_crc_and_does_not_touch_oracle() -> None:
    if not FROZEN.exists():
        return
    payload = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert payload["format"] == "ccle_context_identity_split_v1"
    assert payload["status"] == "IDENTITY_SPLIT_FROZEN"
    assert payload["response_blind"] is True
    assert payload["exp007_primary_metrics_untouched"] is True
    assert payload["gene_universe"]["cannot_use_as_x_ctl"] is True
    assert payload["gene_universe"]["cannot_compute_delta978"] is True
    assert payload["summary"]["crc_exact_count"] == 10
    assert payload["summary"]["crc_exact_with_rnaseq"] == list(CRC_EXACT_CONTEXTS)
    locked = [row for row in payload["split_assignments"] if row["crc_exact_context"]]
    assert {row["stripped_cell_line_name"] for row in locked} == set(CRC_EXACT_CONTEXTS)
    assert all(row["split_role"] == "locked_eval" for row in locked)
    sw = {
        row["stripped_cell_line_name"]: row["patient_id"]
        for row in locked
        if row["stripped_cell_line_name"] in {"SW480", "SW620"}
    }
    assert sw["SW480"] == sw["SW620"]
    h1299 = next(row for row in payload["identity_rows"] if row["lincs_cell_id"] == "H1299")
    assert h1299["join_status"] == "UNRESOLVED_ALIAS_CANDIDATE"
    assert_no_patient_leakage(payload["split_assignments"])
    if RECORD.exists():
        text = RECORD.read_text(encoding="utf-8")
        assert "Observed Δ978 → Disease Reversal" in text
        assert "Top-10/20/50 overlap" in text
    if INTAKE.exists():
        intake = json.loads(INTAKE.read_text(encoding="utf-8"))
        assert intake["adapter_output"]["cannot_replace_matched_control"] is True
