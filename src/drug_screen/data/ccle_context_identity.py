"""Response-blind CCLE/DepMap 24Q2 identity table and split contract.

This module freezes *who* a CCLE RNA-seq row is and *which split role* it may
occupy.  It does not read PRISM, GDSC, disease-reversal, or test performance.
The 978 log2(TPM+1) vector is a basal context only: it is not LINCS X_ctl and
cannot be used to compute Δ978.
"""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from typing import Any, Iterable, Mapping

from drug_screen.evaluation.full_observed_oracle import CRC_EXACT_CONTEXTS


FORMAT = "ccle_context_identity_split_v1"
JOIN_KEY = "Model.StrippedCellLineName.upper() == LINCS context_id.upper()"
SPLIT_UNIT = "PatientID"
SAMPLE_UNIT = "depmap_id"
SPLIT_SEED = 20260814
ORDERED_GENE_IDS_SHA256 = "b4e2fca877c5cfdcc1c712ad0fd67e97a88b6f7566b013e4bab065f699ebb623"
KNOWN_UNRESOLVED_ALIASES = {
    "H1299": ("NCIH1299", "H1299 is the LINCS cell_id; DepMap stripped name is NCIH1299"),
}

FORBIDDEN_SPLIT_INPUTS = (
    "PRISM response values",
    "GDSC response values",
    "test-set ranking or lift",
    "Oracle Spearman / Top-K / NDCG",
    "predicted-to-oracle gap",
    "disease-reversal scores",
)


def _upper(value: object) -> str:
    return str(value).strip().upper()


def _valid(value: object) -> bool:
    text = str(value).strip() if value is not None else ""
    return text not in {"", "NAN", "NONE", "NULL"}


def patient_split_bucket(patient_id: str, *, seed: int = SPLIT_SEED) -> str:
    """Deterministic 80/20 train/val bucket.  CRC locked eval is applied first."""
    digest = sha256(f"{seed}|{patient_id}".encode("utf-8")).hexdigest()
    return "val" if int(digest[:8], 16) % 10 == 0 else "train"


def _index_models(models: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_stripped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in models:
        name = _upper(row.get("StrippedCellLineName") or row.get("stripped_cell_line_name") or "")
        if not name:
            continue
        by_stripped[name].append(dict(row))
    return by_stripped


def _index_expression(mapping_rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_stripped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mapping_rows:
        name = _upper(row.get("stripped_cell_line_name") or "")
        if not name:
            continue
        by_stripped[name].append(dict(row))
    return by_stripped


def _model_field(row: Mapping[str, Any], *names: str) -> object:
    for name in names:
        if name in row and _valid(row[name]):
            return row[name]
    return None


def build_identity_rows(
    *,
    lincs_contexts: Iterable[Mapping[str, Any]],
    models: Iterable[Mapping[str, Any]],
    expression_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """One row per LINCS context, joined only by exact stripped-name identity."""
    model_index = _index_models(models)
    expr_index = _index_expression(expression_rows)
    rows: list[dict[str, Any]] = []
    for context in lincs_contexts:
        cell_id = str(context["context_id"])
        stripped = _upper(cell_id)
        model_hits = model_index.get(stripped, [])
        expr_hits = expr_index.get(stripped, [])
        alias = KNOWN_UNRESOLVED_ALIASES.get(cell_id)
        alias_hits = model_index.get(alias[0], []) if alias else []
        if len(model_hits) > 1 or len(expr_hits) > 1:
            status = "AMBIGUOUS_STRIPPED_NAME"
        elif expr_hits:
            status = "EXACT_STRIPPED_NAME_WITH_RNASEQ"
        elif model_hits:
            status = "EXACT_STRIPPED_NAME_MODEL_ONLY"
        elif alias_hits:
            status = "UNRESOLVED_ALIAS_CANDIDATE"
        else:
            status = "NO_DEPMAP_MATCH"
        model = model_hits[0] if len(model_hits) == 1 else None
        expr = expr_hits[0] if len(expr_hits) == 1 else None
        depmap_id = None
        if expr is not None:
            depmap_id = expr.get("depmap_id")
        elif model is not None:
            depmap_id = _model_field(model, "ModelID", "depmap_id")
        registered_ids = [str(item) for item in (context.get("broad_depmap_ids") or [])]
        rows.append(
            {
                "lincs_cell_id": cell_id,
                "stripped_cell_line_name": stripped if model or expr else stripped,
                "depmap_id": str(depmap_id) if depmap_id else None,
                "patient_id": str(_model_field(model, "PatientID", "patient_id")) if model else None,
                "oncotree_lineage": (
                    str(_model_field(model, "OncotreeLineage", "oncotree_lineage")) if model else None
                ),
                "primary_or_metastasis": (
                    str(_model_field(model, "PrimaryOrMetastasis", "primary_or_metastasis"))
                    if model
                    else None
                ),
                "expression_present": expr is not None,
                "exact978_row": None if expr is None else expr.get("exact978_row"),
                "crc_exact_context": cell_id in CRC_EXACT_CONTEXTS,
                "broad_exact_context": bool(context.get("broad_exact_context")),
                "registered_broad_depmap_ids": registered_ids,
                "depmap_id_matches_registry": (
                    depmap_id is not None and str(depmap_id) in registered_ids if registered_ids else None
                ),
                "join_status": status,
                "alias_note": None if alias is None else alias[1],
                "alias_depmap_id": (
                    str(_model_field(alias_hits[0], "ModelID", "depmap_id"))
                    if alias and len(alias_hits) == 1
                    else None
                ),
            }
        )
    return sorted(rows, key=lambda row: row["lincs_cell_id"])


def assign_roles(
    *,
    identity_rows: Iterable[Mapping[str, Any]],
    expression_rows: Iterable[Mapping[str, Any]],
    models: Iterable[Mapping[str, Any]],
    seed: int = SPLIT_SEED,
) -> list[dict[str, Any]]:
    """Patient-atomic roles.  CRC exact lines are locked eval and never hashed."""
    model_by_id = {
        str(_model_field(row, "ModelID", "depmap_id")): dict(row)
        for row in models
        if _model_field(row, "ModelID", "depmap_id")
    }
    identity_by_depmap = {
        row["depmap_id"]: dict(row) for row in identity_rows if row.get("depmap_id")
    }
    locked_patients = {
        row["patient_id"]
        for row in identity_rows
        if row.get("crc_exact_context") and _valid(row.get("patient_id"))
    }
    assignments: list[dict[str, Any]] = []
    for expr in expression_rows:
        depmap_id = str(expr["depmap_id"])
        model = model_by_id.get(depmap_id, {})
        patient_id = str(_model_field(model, "PatientID", "patient_id") or f"UNGROUPED:{depmap_id}")
        identity = identity_by_depmap.get(depmap_id)
        if patient_id in locked_patients or (identity and identity.get("crc_exact_context")):
            role = "locked_eval"
            reason = "crc_exact_patient_atomic_lock"
        else:
            role = patient_split_bucket(patient_id, seed=seed)
            reason = "deterministic_patient_hash_no_response"
        assignments.append(
            {
                "depmap_id": depmap_id,
                "stripped_cell_line_name": str(
                    expr.get("stripped_cell_line_name")
                    or _model_field(model, "StrippedCellLineName", "stripped_cell_line_name")
                    or ""
                ).upper(),
                "patient_id": patient_id,
                "lincs_cell_id": None if identity is None else identity["lincs_cell_id"],
                "crc_exact_context": bool(identity and identity.get("crc_exact_context")),
                "lincs_overlap": identity is not None and identity.get("join_status")
                == "EXACT_STRIPPED_NAME_WITH_RNASEQ",
                "split_role": role,
                "split_reason": reason,
                "exact978_row": expr.get("exact978_row"),
            }
        )
    return sorted(assignments, key=lambda row: (row["split_role"], row["depmap_id"]))


def assert_no_patient_leakage(assignments: Iterable[Mapping[str, Any]]) -> None:
    roles_by_patient: dict[str, set[str]] = defaultdict(set)
    for row in assignments:
        roles_by_patient[str(row["patient_id"])].add(str(row["split_role"]))
    leaks = {patient: sorted(roles) for patient, roles in roles_by_patient.items() if len(roles) > 1}
    if leaks:
        raise ValueError(f"patient-level split leakage: {leaks}")


def assert_crc_exact_locked(assignments: Iterable[Mapping[str, Any]]) -> None:
    unlocked = [
        row["stripped_cell_line_name"]
        for row in assignments
        if row.get("crc_exact_context") and row.get("split_role") != "locked_eval"
    ]
    if unlocked:
        raise ValueError(f"CRC exact contexts not locked: {unlocked}")


def assert_all_crc_exact_present(assignments: Iterable[Mapping[str, Any]]) -> None:
    seen = {
        row["stripped_cell_line_name"]
        for row in assignments
        if row.get("crc_exact_context")
    }
    missing = [name for name in CRC_EXACT_CONTEXTS if name not in seen]
    if missing:
        raise ValueError(f"CRC exact contexts missing from split assignments: {missing}")
    assert_crc_exact_locked(assignments)


def summarize(identity_rows: list[dict[str, Any]], assignments: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = defaultdict(int)
    for row in identity_rows:
        by_status[row["join_status"]] += 1
    by_role: dict[str, int] = defaultdict(int)
    patients_by_role: dict[str, set[str]] = defaultdict(set)
    for row in assignments:
        by_role[row["split_role"]] += 1
        patients_by_role[row["split_role"]].add(row["patient_id"])
    crc_locked = sorted(
        row["lincs_cell_id"] for row in identity_rows if row.get("crc_exact_context") and row.get("expression_present")
    )
    return {
        "lincs_contexts": len(identity_rows),
        "join_status_counts": dict(sorted(by_status.items())),
        "expression_overlap": by_status.get("EXACT_STRIPPED_NAME_WITH_RNASEQ", 0),
        "model_only": by_status.get("EXACT_STRIPPED_NAME_MODEL_ONLY", 0),
        "unresolved_alias": by_status.get("UNRESOLVED_ALIAS_CANDIDATE", 0),
        "crc_exact_with_rnaseq": crc_locked,
        "crc_exact_count": len(crc_locked),
        "assignment_count": len(assignments),
        "lines_by_role": dict(sorted(by_role.items())),
        "patients_by_role": {role: len(patients) for role, patients in sorted(patients_by_role.items())},
    }


def data_contract(*, summary: Mapping[str, Any], leakage_status: str) -> dict[str, Any]:
    return {
        "source": "DepMap Public 24Q2 + GSE92742 CONTEXT_REGISTRY",
        "version": "24Q2 / identity_split_v1",
        "sample_unit": SAMPLE_UNIT,
        "split_unit": SPLIT_UNIT,
        "gene_universe": {
            "ordered_gene_ids_sha256": ORDERED_GENE_IDS_SHA256,
            "units": "official log2(TPM+1)",
            "cannot_replace_matched_control": True,
            "cannot_use_as_x_ctl": True,
            "cannot_compute_delta978": True,
        },
        "label_definition": "none; identity and split only; no PRISM/GDSC/reversal labels",
        "leakage_status": leakage_status,
        "join_key": JOIN_KEY,
        "crc_exact_required": list(CRC_EXACT_CONTEXTS),
        "crc_exact_present": summary.get("crc_exact_with_rnaseq"),
        "forbidden_split_inputs": list(FORBIDDEN_SPLIT_INPUTS),
        "response_blind": True,
    }
