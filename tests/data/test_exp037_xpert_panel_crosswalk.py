from __future__ import annotations

import io
import json

import pytest

import scripts.data.audit_exp037_xpert_panel_crosswalk as audit
from scripts.data.audit_exp037_xpert_panel_crosswalk import (
    _record,
    build_crosswalk,
    build_input_chain,
    selected_xpert_rows,
)


def _panel_row(smiles: str, drug_index: int) -> dict[str, object]:
    canonical, inchi_key = audit._rdkit_smiles_identity(smiles)
    return {
        "drug_index": drug_index,
        "drug_id": inchi_key,
        "canonical_isomeric_smiles": canonical,
    }


def _sidecar_row(panel_row: dict[str, object], pert_id: str, pert_idx: int) -> dict[str, object]:
    return {
        "pert_id": pert_id,
        "pert_idx": pert_idx,
        "inchi_key": panel_row["drug_id"],
        "canonical_smiles": panel_row["canonical_isomeric_smiles"],
        "unimol": True,
        "hg": True,
    }


def test_binary_csv_parser_does_not_decode_or_retain_other_columns(tmp_path):
    path = tmp_path / "source.csv"
    path.write_bytes(
        b'pert_id,unused,inchi_key,drug_node_idx\r\n'
        b'p1,"invalid-utf8-\xff, with comma",KEY-ONE,7\r\n'
        b'p2,"quoted ""text""\nand newline",KEY-TWO,8\r\n'
    )

    rows, count = selected_xpert_rows(path, {"KEY-ONE", "KEY-TWO"})

    assert count == 2
    assert rows == [
        {"pert_id": "p1", "inchi_key": "KEY-ONE", "drug_node_idx": "7"},
        {"pert_id": "p2", "inchi_key": "KEY-TWO", "drug_node_idx": "8"},
    ]


def test_binary_csv_parser_rejects_unterminated_quote():
    stream = io.BytesIO(b'"unterminated')
    with pytest.raises(ValueError, match="unterminated"):
        _record(stream, selected=None)


def test_csv_summary_distinguishes_duplicate_rows_from_conflicting_mappings():
    panel = [
        {"drug_index": 0, "drug_id": "KEY-A"},
        {"drug_index": 1, "drug_id": "KEY-B"},
        {"drug_index": 2, "drug_id": "KEY-C"},
    ]
    source = [
        {"pert_id": "P1", "inchi_key": "KEY-A", "drug_node_idx": "7"},
        {"pert_id": "P1", "inchi_key": "KEY-A", "drug_node_idx": "7"},
        {"pert_id": "P2", "inchi_key": "KEY-B", "drug_node_idx": "8"},
    ]

    summary = build_crosswalk(panel, source)

    assert summary["status"] == "BLOCKED_PARTIAL_CANDIDATE"
    assert summary["panel_keys_with_candidate_rows"] == 2
    assert summary["panel_keys_with_one_distinct_identity_node_tuple"] == 2
    assert summary["missing_panel_keys"] == 1
    assert summary["panel_keys_with_repeated_physical_rows"] == 1
    assert summary["repeated_identical_candidate_rows"] == 1
    assert summary["panel_keys_with_multiple_distinct_identity_node_tuples"] == 0


def test_csv_summary_fails_closed_on_conflicting_identity_node_tuples():
    panel = [{"drug_index": 0, "drug_id": "KEY-A"}]
    source = [
        {"pert_id": "P1", "inchi_key": "KEY-A", "drug_node_idx": "7"},
        {"pert_id": "P2", "inchi_key": "KEY-A", "drug_node_idx": "8"},
    ]

    summary = build_crosswalk(panel, source)

    assert summary["panel_keys_with_multiple_distinct_identity_node_tuples"] == 1
    assert summary["status"] == "BLOCKED_PARTIAL_CANDIDATE"


def test_registered_sidecar_closes_complete_chain_and_emits_counts_only():
    panel = [_panel_row("CCO", 0), _panel_row("CCN", 1)]
    sidecar_rows = [
        _sidecar_row(panel[0], "P-ONE", 9001),
        _sidecar_row(panel[1], "P-TWO", 9002),
    ]
    csv_rows = [
        {"pert_id": "P-ONE", "inchi_key": panel[0]["drug_id"], "drug_node_idx": "9001"},
        {"pert_id": "P-ONE", "inchi_key": panel[0]["drug_id"], "drug_node_idx": "9001"},
        {"pert_id": "P-TWO", "inchi_key": panel[1]["drug_id"], "drug_node_idx": "9002"},
    ]
    sidecar = {
        "feature_index_key": audit.EXPECTED_FEATURE_INDEX_KEY,
        "drugs": sidecar_rows,
    }

    summary = build_input_chain(panel, csv_rows, sidecar)

    assert summary["status"] == "IDENTITY_CHAIN_VERIFIED_FOR_PANEL"
    assert summary["target_smiles_to_full_inchikey_valid"] == 2
    assert summary["panel_smiles_canonical_match_to_sidecar"] == 2
    assert summary["panel_rows_with_csv_node_and_sidecar_index_agreement"] == 2
    assert summary["panel_rows_with_complete_identity_to_model_index_chain"] == 2
    serialized = json.dumps(summary)
    for identity in [str(panel[0]["drug_id"]), str(panel[1]["drug_id"]), "P-ONE", "P-TWO", "9001", "9002"]:
        assert identity not in serialized


def test_registered_sidecar_keeps_missing_panel_drug_blocked():
    panel = [_panel_row("CCO", 0), _panel_row("CCN", 1)]
    sidecar = {
        "feature_index_key": audit.EXPECTED_FEATURE_INDEX_KEY,
        "drugs": [_sidecar_row(panel[0], "P-ONE", 9001)],
    }
    csv_rows = [
        {"pert_id": "P-ONE", "inchi_key": panel[0]["drug_id"], "drug_node_idx": "9001"},
    ]

    summary = build_input_chain(panel, csv_rows, sidecar)

    assert summary["registered_panel_inchikeys_found"] == 1
    assert summary["panel_rows_with_complete_identity_to_model_index_chain"] == 1
    assert summary["panel_rows_missing_from_registered_xpert_sidecar"] == 1
    assert summary["status"] == "DATA_BLOCKED_PARTIAL_IDENTITY_CHAIN"


def test_sidecar_index_collision_with_other_structure_blocks_row_mapping():
    panel = [_panel_row("CCO", 0)]
    other = _panel_row("CCC", 1)
    sidecar = {
        "feature_index_key": audit.EXPECTED_FEATURE_INDEX_KEY,
        "drugs": [
            _sidecar_row(panel[0], "P-ONE", 9001),
            _sidecar_row(other, "P-OTHER", 9001),
        ],
    }
    csv_rows = [
        {"pert_id": "P-ONE", "inchi_key": panel[0]["drug_id"], "drug_node_idx": "9001"},
    ]

    summary = build_input_chain(panel, csv_rows, sidecar)

    assert summary["panel_rows_with_csv_node_and_sidecar_index_agreement"] == 1
    assert summary["panel_feature_indexes_unique_across_sidecar"] == 0
    assert summary["panel_rows_with_complete_identity_to_model_index_chain"] == 0
    assert summary["status"] == "DATA_BLOCKED_PARTIAL_IDENTITY_CHAIN"


def test_main_writes_a_failure_receipt_when_preflight_fails(tmp_path, monkeypatch):
    output = tmp_path / "audit"
    csv_path = tmp_path / "must_not_open.csv"

    def fail_preflight():
        raise ValueError("preflight blocked")

    monkeypatch.setattr(audit, "OUTPUT", output)
    monkeypatch.setattr(audit, "XPERT_CSV", csv_path)
    monkeypatch.setattr(audit, "_preflight", fail_preflight)

    with pytest.raises(ValueError, match="preflight blocked"):
        audit.main()

    assert not csv_path.exists()
    assert (output / "receipt.json").exists()
