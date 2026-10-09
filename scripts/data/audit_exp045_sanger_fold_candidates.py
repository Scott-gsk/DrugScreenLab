"""Bind three approved Sanger aliases to DepMap 24Q2 rows and fold identities.

The receipt is an identity-candidate ledger only. It does not bind source
checkpoint training, model selection, test exposure, or mask execution.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "artifacts/experiments/EXP-045/20260925_sanger_fold_candidate_binding_v1/receipt.json"
BRIDGE = ROOT / "artifacts/experiments/EXP-045/20260925_sanger_source_alias_bridge_v1/receipt.json"
MANIFEST = ROOT / "artifacts/experiments/EXP-045/identity_review_20260923_v3b/identity_manifest.json"
REGISTRY = ROOT / "data/registry/datasets.json"
MODEL_CSV = ROOT / "data/raw/depmap/24q2/Model.csv"
RNA_CSV = ROOT / "data/raw/depmap/24q2/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
CRISPR_CSV = ROOT / "data/raw/depmap/24q2/CRISPRGeneEffect.csv"

SANGER_IDS = ("SIDM00117", "SIDM00427", "SIDM01117")
EXPECTED_BRIDGE_SHA256 = "8be00c61d9486ffa36fda4d77d8061e7133cb5acb3b69ed3befc78804ec6eb8c"
EXPECTED_MANIFEST_SHA256 = "c44e50edd88e93eb70cbe96de56327818d061c309ca4778c46d6ea550715edfc"
EXPECTED_INPUTS = {
    MODEL_CSV: (
        "depmap_public_24q2_rnaseq_raw_v1",
        "a4cac376131b41aa10b60a075b11c80264bfa860a5509d22cde259c5e85867f8",
    ),
    RNA_CSV: (
        "depmap_public_24q2_rnaseq_raw_v1",
        "39eff342bbbbe0a40208545e93a69cc85a945899592f9d683f87e3b2bb121670",
    ),
    CRISPR_CSV: (
        "depmap_public_24q2_crispr_gene_effect_raw_v1",
        "d155149181308f8e16b6a4677cde3e5af1ddef68ada1947486ae8c0bcb231452",
    ),
}
MODEL_IDENTITY_COLUMNS = (
    "ModelID",
    "PatientID",
    "CellLineName",
    "CCLEName",
    "HCMIID",
    "WTSIMasterCellID",
    "SangerModelID",
    "COSMICID",
)
EXPECTED_CANDIDATES = {
    "SIDM00117": ["ACH-000837", "ACH-002172"],
    "SIDM00427": [],
    "SIDM01117": ["ACH-003071"],
}
EXPECTED_BRIDGE_AXIS_MATCHES = {
    "SIDM00117": {"RNA_axis": ["ACH-000837"], "CRISPR_axis": ["ACH-000837"]},
    "SIDM00427": {"RNA_axis": [], "CRISPR_axis": []},
    "SIDM01117": {"RNA_axis": [], "CRISPR_axis": []},
}


class AuditStop(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, failure_code: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise AuditStop(failure_code) from None


def registry_entry(registry: Any, dataset_id: str, filename: str) -> dict[str, Any]:
    if not isinstance(registry, list):
        raise AuditStop("DATA_REGISTRY_SCHEMA_UNKNOWN")
    matches = [
        entry for entry in registry
        if isinstance(entry, dict) and entry.get("id") == dataset_id
    ]
    if len(matches) != 1:
        raise AuditStop("DATA_REGISTRY_ENTRY_NOT_UNIQUE")
    files = matches[0].get("files")
    if not isinstance(files, dict) or not isinstance(files.get(filename), dict):
        raise AuditStop("DATA_REGISTRY_FILE_ENTRY_MISSING")
    return files[filename]


def verify_provenance() -> dict[str, Any]:
    bridge_sha = sha256_file(BRIDGE)
    if bridge_sha != EXPECTED_BRIDGE_SHA256:
        raise AuditStop("OFFICIAL_ALIAS_BRIDGE_SHA256_MISMATCH")
    manifest_sha = sha256_file(MANIFEST)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise AuditStop("FROZEN_IDENTITY_MANIFEST_SHA256_MISMATCH")
    registry = load_json(REGISTRY, "DATA_REGISTRY_UNREADABLE")
    input_hashes: dict[str, str] = {}
    for path, (dataset_id, expected_hash) in EXPECTED_INPUTS.items():
        try:
            entry = registry_entry(registry, dataset_id, path.name)
            actual_hash = sha256_file(path)
            if path.stat().st_size != int(entry["bytes"]):
                raise AuditStop("REGISTERED_INPUT_SIZE_MISMATCH")
            if actual_hash != expected_hash or actual_hash != str(entry["sha256"]):
                raise AuditStop("REGISTERED_INPUT_CHECKSUM_MISMATCH")
            input_hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = actual_hash
        except AuditStop:
            raise
        except (OSError, KeyError, TypeError, ValueError):
            raise AuditStop("REGISTERED_INPUT_UNREADABLE_OR_SCHEMA_UNKNOWN") from None
    return {
        "official_alias_bridge_sha256": bridge_sha,
        "v3b_identity_manifest_sha256": manifest_sha,
        "depmap_24q2_input_sha256": input_hashes,
    }


def read_axis(path: Path) -> tuple[set[str], Counter[str]]:
    try:
        frame = pd.read_csv(path, usecols=[0], dtype=str, keep_default_na=False, low_memory=False)
    except Exception:
        raise AuditStop("INPUT_AXIS_SCHEMA_UNKNOWN_OR_UNREADABLE") from None
    if frame.shape[1] != 1:
        raise AuditStop("INPUT_AXIS_SCHEMA_UNKNOWN")
    values = [str(value).strip() for value in frame.iloc[:, 0] if str(value).strip()]
    return set(values), Counter(values)


def read_model_identity_rows() -> list[dict[str, str]]:
    try:
        frame = pd.read_csv(
            MODEL_CSV,
            usecols=list(MODEL_IDENTITY_COLUMNS),
            dtype=str,
            keep_default_na=False,
            low_memory=False,
        )
    except Exception:
        raise AuditStop("MODEL_IDENTITY_SCHEMA_UNKNOWN_OR_UNREADABLE") from None
    if set(frame.columns) != set(MODEL_IDENTITY_COLUMNS):
        raise AuditStop("MODEL_IDENTITY_SCHEMA_UNKNOWN")
    return [
        {column: str(row[column]).strip() for column in MODEL_IDENTITY_COLUMNS}
        for row in frame.to_dict(orient="records")
    ]


def read_bridge_candidates() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    bridge = load_json(BRIDGE, "OFFICIAL_ALIAS_BRIDGE_UNREADABLE")
    if not isinstance(bridge, dict) or bridge.get("status") != "IDENTITY_AXIS_AUDIT_COMPLETE":
        raise AuditStop("OFFICIAL_ALIAS_BRIDGE_SCHEMA_UNKNOWN")
    scope = bridge.get("scope")
    results = bridge.get("results")
    if not isinstance(scope, dict) or scope.get("endpoint") != "/models/{id}":
        raise AuditStop("OFFICIAL_ALIAS_BRIDGE_SCOPE_UNKNOWN")
    if not isinstance(results, dict) or not isinstance(results.get("per_key"), list):
        raise AuditStop("OFFICIAL_ALIAS_BRIDGE_RESULTS_SCHEMA_UNKNOWN")
    rows = [row for row in results["per_key"] if isinstance(row, dict)]
    by_id = {row.get("registered_Sanger_ID"): row for row in rows}
    if len(rows) != 3 or len(by_id) != 3 or set(by_id) != set(SANGER_IDS):
        raise AuditStop("OFFICIAL_ALIAS_BRIDGE_KEY_SET_MISMATCH")
    for sanger_id in SANGER_IDS:
        row = by_id[sanger_id]
        if row.get("http_status") != 200 or row.get("model_objects_returned") != 1:
            raise AuditStop("OFFICIAL_ALIAS_BRIDGE_KEY_NOT_RESOLVED")
        candidates = row.get("DepMap_24Q2_exact_candidate_ModelIDs")
        if not isinstance(candidates, list) or sorted(candidates) != EXPECTED_CANDIDATES[sanger_id]:
            raise AuditStop("OFFICIAL_ALIAS_BRIDGE_CANDIDATE_SCHEMA_OR_VALUE_MISMATCH")
        if row.get("CMP_primary_ModelID") != sanger_id:
            raise AuditStop("OFFICIAL_ALIAS_BRIDGE_PRIMARY_ID_MISMATCH")
        for axis_name in ("RNA_axis", "CRISPR_axis"):
            axis_record = row.get(axis_name)
            if not isinstance(axis_record, dict) or axis_record.get("matched_keys") != EXPECTED_BRIDGE_AXIS_MATCHES[sanger_id][axis_name]:
                raise AuditStop("OFFICIAL_ALIAS_BRIDGE_AXIS_SCHEMA_OR_VALUE_MISMATCH")
    return by_id, bridge


def read_fold_positions() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    manifest = load_json(MANIFEST, "FROZEN_IDENTITY_MANIFEST_UNREADABLE")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("source_folds"), list):
        raise AuditStop("FROZEN_IDENTITY_MANIFEST_SCHEMA_UNKNOWN")
    folds = manifest["source_folds"]
    if len(folds) != 5:
        raise AuditStop("FROZEN_IDENTITY_OUTER_FOLD_COUNT_MISMATCH")
    positions: dict[str, list[dict[str, Any]]] = {key: [] for key in SANGER_IDS}
    seen_outer: set[int] = set()
    expected_inner_ids: set[int] | None = None
    for outer in folds:
        if not isinstance(outer, dict):
            raise AuditStop("FROZEN_IDENTITY_OUTER_FOLD_SCHEMA_UNKNOWN")
        outer_fold = outer.get("fold")
        outer_ids = outer.get("unresolved_sanger_ids")
        inner_splits = outer.get("inner")
        if not isinstance(outer_fold, int) or not isinstance(outer_ids, list) or not isinstance(inner_splits, list):
            raise AuditStop("FROZEN_IDENTITY_OUTER_FOLD_SCHEMA_UNKNOWN")
        if outer_fold in seen_outer:
            raise AuditStop("FROZEN_IDENTITY_OUTER_FOLD_DUPLICATE")
        seen_outer.add(outer_fold)
        if any(not isinstance(value, str) for value in outer_ids):
            raise AuditStop("FROZEN_IDENTITY_OUTER_ID_LIST_SCHEMA_UNKNOWN")
        by_key: dict[str, dict[str, Any]] = {
            key: {
                "outer_fold": outer_fold,
                "outer_registered_member": key in outer_ids,
                "inner_splits": [],
            }
            for key in SANGER_IDS
        }
        seen_inner: set[int] = set()
        for inner in inner_splits:
            if not isinstance(inner, dict):
                raise AuditStop("FROZEN_IDENTITY_INNER_SPLIT_SCHEMA_UNKNOWN")
            inner_fold = inner.get("inner_fold")
            inner_ids = inner.get("unresolved_sanger_ids")
            if not isinstance(inner_fold, int) or not isinstance(inner_ids, list):
                raise AuditStop("FROZEN_IDENTITY_INNER_SPLIT_SCHEMA_UNKNOWN")
            if inner_fold in seen_inner or any(not isinstance(value, str) for value in inner_ids):
                raise AuditStop("FROZEN_IDENTITY_INNER_ID_LIST_SCHEMA_UNKNOWN")
            seen_inner.add(inner_fold)
            for key in SANGER_IDS:
                by_key[key]["inner_splits"].append({
                    "inner_fold": inner_fold,
                    "registered_member": key in inner_ids,
                })
        if len(inner_splits) != 3:
            raise AuditStop("FROZEN_IDENTITY_INNER_SPLITS_MISSING")
        if expected_inner_ids is None:
            expected_inner_ids = set(seen_inner)
        elif set(seen_inner) != expected_inner_ids:
            raise AuditStop("FROZEN_IDENTITY_INNER_FOLD_IDS_MISMATCH")
        for key in SANGER_IDS:
            positions[key].append(by_key[key])
    if seen_outer != set(range(5)):
        raise AuditStop("FROZEN_IDENTITY_OUTER_FOLD_IDS_MISMATCH")
    for key in SANGER_IDS:
        positions[key].sort(key=lambda item: item["outer_fold"])
    return positions, manifest


def main() -> int:
    if RECEIPT.exists():
        print("OUTPUT_EXISTS_NO_OVERWRITE")
        return 2
    receipt: dict[str, Any] = {
        "receipt_id": "EXP-045-Sanger-fold-candidate-binding-v1",
        "status": "NOT_STARTED",
        "scope": {
            "registered_Sanger_IDs": list(SANGER_IDS),
            "read_identity_fields": {
                "DepMap_Model_csv": list(MODEL_IDENTITY_COLUMNS),
                "RNA_CSV": "first column only",
                "CRISPR_CSV": "first column only",
                "official_bridge_receipt": [
                    "registered_Sanger_ID", "http_status", "model_objects_returned",
                    "CMP_primary_ModelID", "DepMap_24Q2_exact_candidate_ModelIDs",
                    "RNA_axis.matched_keys", "CRISPR_axis.matched_keys",
                ],
                "v3b_identity_manifest": [
                    "source_folds[*].fold", "source_folds[*].unresolved_sanger_ids",
                    "source_folds[*].inner[*].inner_fold",
                    "source_folds[*].inner[*].unresolved_sanger_ids",
                ],
            },
            "forbidden_values_read": [
                "expression", "dependency_or_drug_sensitivity_labels", "predictions",
                "checkpoint_tensors_or_values", "scores",
            ],
            "API_requests": 0,
            "training_or_tests_run": False,
            "GPU_used": False,
            "formal_training_support_file": False,
            "formal_training_GO": False,
        },
        "provenance": {},
        "per_object": [],
    }
    try:
        if len(set(SANGER_IDS)) != 3:
            raise AuditStop("REGISTERED_KEY_SET_INVALID")
        receipt["provenance"] = verify_provenance()
        bridge_rows, _ = read_bridge_candidates()
        fold_positions, _ = read_fold_positions()
        model_rows = read_model_identity_rows()
        rna_keys, rna_frequency = read_axis(RNA_CSV)
        crispr_keys, crispr_frequency = read_axis(CRISPR_CSV)

        actual_axis_results: dict[str, tuple[set[str], Counter[str]]] = {
            "RNA_axis": (rna_keys, rna_frequency),
            "CRISPR_axis": (crispr_keys, crispr_frequency),
        }
        for sanger_id in SANGER_IDS:
            bridge_row = bridge_rows[sanger_id]
            candidates = sorted(bridge_row["DepMap_24Q2_exact_candidate_ModelIDs"])
            present_model_ids = sorted({row["ModelID"] for row in model_rows if row["ModelID"] in candidates})
            if present_model_ids != candidates:
                raise AuditStop("BRIDGED_CANDIDATE_NOT_PRESENT_IN_MODEL_CSV")

            per_object: dict[str, Any] = {
                "Sanger_ID": sanger_id,
                "official_CMP": {
                    "http_status": bridge_row["http_status"],
                    "model_objects_returned": bridge_row["model_objects_returned"],
                    "CMP_primary_ModelID": bridge_row["CMP_primary_ModelID"],
                    "Model_csv_exact_candidate_ModelIDs": candidates,
                    "unique_candidate_ModelID_count": len(candidates) if len(candidates) == 1 else 0,
                    "ambiguous_candidate_ModelID_count": len(candidates) if len(candidates) > 1 else 0,
                },
                "current_input_axes": {},
                "fold_identity_registration": fold_positions[sanger_id],
                "source_checkpoint_training_selection_test_exposure": "UNKNOWN_UNBOUND",
                "potential_checkpoint_or_pretraining_exposure_isolation_ModelIDs": candidates,
                "unresolved_Sanger_ID_for_potential_exposure_isolation": sanger_id if not candidates else None,
            }
            for axis_name, (keys, frequencies) in actual_axis_results.items():
                matches = [candidate for candidate in candidates if candidate in keys]
                per_object["current_input_axes"][axis_name] = {
                    "hit": bool(matches),
                    "matched_current_ModelIDs": matches,
                    "matched_unique_key_count": len(matches),
                    "matched_row_count": sum(frequencies[value] for value in matches),
                    "candidate_misses": [candidate for candidate in candidates if candidate not in keys],
                    "assessment": (
                        "NO_DEPMAP_MODELID_CANDIDATE"
                        if not candidates
                        else "MAPPED_CURRENT_AXIS_ROW"
                        if matches
                        else "CANDIDATES_PRESENT_BUT_NO_CURRENT_AXIS_ROW"
                    ),
                }
            receipt["per_object"].append(per_object)

        receipt["status"] = "IDENTITY_CANDIDATE_BINDING_COMPLETE"
        receipt["summary"] = {
            "outer_folds_registered": 5,
            "inner_split_memberships_evaluated": sum(
                len(fold["inner_splits"])
                for per_object in receipt["per_object"]
                for fold in per_object["fold_identity_registration"]
            ),
            "distinct_outer_inner_slots": sum(
                len(fold["inner_splits"])
                for fold in receipt["per_object"][0]["fold_identity_registration"]
            ),
            "SIDM00117_current_RNA_and_CRISPR_ModelID": "ACH-000837",
            "SIDM00117_ambiguous_Model_csv_candidate_retained_for_potential_exposure_isolation": "ACH-002172",
            "SIDM01117_current_axis_status": "ACH-003071 candidate absent from RNA and CRISPR row keys",
            "SIDM00427_current_axis_status": "no DepMap Model.csv candidate; remains unresolved",
            "all_source_checkpoint_exposures": "UNKNOWN_UNBOUND",
        }
    except AuditStop as exc:
        receipt["status"] = "STOPPED_WITH_FIXED_FAILURE_CODE"
        receipt["failure_code"] = exc.code

    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "failure_code": receipt.get("failure_code"),
        "receipt": str(RECEIPT.relative_to(ROOT)).replace("\\", "/"),
        "summary": receipt.get("summary"),
        "per_object": receipt.get("per_object", []),
    }, indent=2, ensure_ascii=False))
    return 0 if receipt["status"] == "IDENTITY_CANDIDATE_BINDING_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
