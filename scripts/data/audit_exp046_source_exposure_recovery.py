"""Audit EXP-046 fold membership from approved identity sidecars only.

The audit writes counts and file hashes. It never writes identity values and
does not load labels, expression arrays, predictions, run summaries, or model
weights. Literal identifier equality is reported only as a namespace check;
it is not treated as a biological overlap result without an authority map.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


OUTPUT_REL = Path(
    "artifacts/experiments/EXP-046/20260924_source_exposure_recovery_v1/receipt.json"
)
PATIENT_AXIS_REL = Path(
    "data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1/patient_axis.json"
)
SPLIT_REL = Path("data/splits/dual_route_splits_20260922_v1.json")
STATIC_AUDIT_REL = Path(
    "artifacts/experiments/EXP-046/20260924_source_exposure_static_audit/receipt.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ids(value: Any, field: str) -> set[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Expected a list of string identity keys in {field}")
    if len(value) != len(set(value)):
        raise ValueError(f"Duplicate identity keys in {field}")
    return set(value)


def build_receipt(repo_root: Path) -> dict[str, Any]:
    axis_path = repo_root / PATIENT_AXIS_REL
    split_path = repo_root / SPLIT_REL
    prior_receipt_path = repo_root / STATIC_AUDIT_REL

    for path in (axis_path, split_path, prior_receipt_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    target_axis_raw = _read_json(axis_path)
    if not isinstance(target_axis_raw, list):
        raise ValueError("Target patient axis must be a JSON list")
    target_axis = _ids(target_axis_raw, "patient_axis")

    splits = _read_json(split_path)
    if (
        not isinstance(splits, dict)
        or splits.get("status") != "SPLIT_FROZEN_LABEL_COVERAGE_PENDING"
        or splits.get("protocol") != "dual_route_20260922_identity_only_v1"
    ):
        raise ValueError("Expected the approved identity-only frozen split manifest")
    target_folds = splits.get("target_folds")
    source_folds = splits.get("source_folds")
    if not isinstance(target_folds, list) or not isinstance(source_folds, list):
        raise ValueError("Frozen target/source fold lists are missing")
    if len(target_folds) != 5 or len(source_folds) != 5:
        raise ValueError("Expected five target and source folds")

    target_test_union: set[str] = set()
    source_test_union: set[str] = set()
    source_universe: set[str] | None = None
    folds: list[dict[str, Any]] = []
    source_ids_by_fold: dict[int, set[str]] = {}
    target_ids_by_fold: dict[int, set[str]] = {}

    for expected_fold, (target, source) in enumerate(zip(target_folds, source_folds)):
        fold_id = target.get("fold")
        if fold_id != expected_fold or source.get("fold") != expected_fold:
            raise ValueError("Target/source folds must have matching ordered IDs 0..4")

        target_train = _ids(target.get("outer_train_ids"), f"target fold {fold_id} train")
        target_test = _ids(target.get("outer_test_ids"), f"target fold {fold_id} test")
        source_train = _ids(source.get("outer_train_ids"), f"source fold {fold_id} train")
        source_test = _ids(source.get("outer_test_ids"), f"source fold {fold_id} test")

        if target_train & target_test:
            raise ValueError(f"Target train/test overlap in fold {fold_id}")
        if target_train != target_axis - target_test:
            raise ValueError(f"Target training IDs are not the axis complement in fold {fold_id}")
        if source_train & source_test:
            raise ValueError(f"Source train/test overlap in fold {fold_id}")
        if target_test_union & target_test:
            raise ValueError("Target outer test folds are not disjoint")
        target_test_union.update(target_test)

        fold_source_universe = source_train | source_test
        if source_universe is None:
            source_universe = fold_source_universe
        elif source_universe != fold_source_universe:
            raise ValueError("Source folds do not share one source-cell-line universe")
        if source_test_union & source_test:
            raise ValueError("Source outer test folds are not disjoint")
        source_test_union.update(source_test)
        if source_train != fold_source_universe - source_test:
            raise ValueError(f"Source training IDs are not the complement in fold {fold_id}")

        target_ids_by_fold[fold_id] = target_test
        source_ids_by_fold[fold_id] = source_train
        folds.append(
            {
                "fold": fold_id,
                "target_train_patients": len(target_train),
                "target_test_patients": len(target_test),
                "target_train_samples": len(target.get("outer_train_samples", [])),
                "target_test_samples": len(target.get("outer_test_samples", [])),
                "source_train_cell_lines": len(source_train),
                "source_test_cell_lines": len(source_test),
                "literal_target_test_vs_source_train_key_matches": len(
                    target_test & source_train
                ),
                "literal_key_match_interpretation": (
                    "namespace diagnostic only; not an authority-mapped biological overlap"
                ),
                "executed_source_mask": "not_verifiable_without_identity_crosswalk_and_fit_receipt",
            }
        )

    if target_test_union != target_axis:
        raise ValueError("Patient axis does not equal the frozen target outer-test universe")
    if source_universe is None:
        raise ValueError("Source fold universe is empty")
    if source_test_union != source_universe:
        raise ValueError("Source outer-test folds do not partition the source universe")

    prior = _read_json(prior_receipt_path)
    prior_evidence = {
        item.get("path"): item
        for item in prior.get("evidence", [])
        if isinstance(item, dict)
    }
    checkpoint_path = "data/artifacts/EXP-011/source_cv/fold0/M2_fold0.pt"
    source_index_path = "data/artifacts/EXP-011/source_cv/fold0/fold0_source_indices.json"
    checkpoint = prior_evidence.get(checkpoint_path)
    source_index = prior_evidence.get(source_index_path)
    if not checkpoint or not source_index:
        raise ValueError("Prior static audit lacks expected source checkpoint/index evidence")

    return {
        "experiment_id": "EXP-046",
        "audit_id": "20260924_source_exposure_recovery_v1",
        "scope": "frozen fold membership integrity and literal key comparison only",
        "status": "FOLD_MEMBERSHIP_VERIFIED_SOURCE_EXPOSURE_UNRESOLVED_NO_GO",
        "inputs": {
            "target_patient_axis": {
                "path": PATIENT_AXIS_REL.as_posix(),
                "sha256": _sha256(axis_path),
                "unique_patient_keys": len(target_axis),
            },
            "frozen_dual_route_split": {
                "path": SPLIT_REL.as_posix(),
                "sha256": _sha256(split_path),
                "manifest_status": splits.get("status"),
                "target_fold_count": len(target_folds),
                "source_fold_count": len(source_folds),
                "target_patient_universe": len(target_test_union),
                "source_cell_line_universe": len(source_universe),
            },
            "prior_static_source_exposure_audit": {
                "path": STATIC_AUDIT_REL.as_posix(),
                "sha256": _sha256(prior_receipt_path),
                "status": prior.get("result", {}).get("status"),
            },
        },
        "folds": folds,
        "checkpoint_lineage": {
            "known_legacy_checkpoint_path": checkpoint_path,
            "known_legacy_checkpoint_sha256": checkpoint.get("sha256"),
            "known_source_index_path": source_index_path,
            "known_source_index_sha256": source_index.get("sha256"),
            "known_source_index_counts": source_index.get("aggregate_counts_only"),
            "per_target_fold_slice_seed_mask_binding": "missing",
            "checkpoint_bytes_loaded": False,
        },
        "gates": {
            "target_patient_axis_matches_frozen_outer_test_universe": "verified",
            "source_train_test_membership_and_common_universe": "verified",
            "literal_identifier_key_matches": "counted_but_not_authoritative",
            "versioned_patient_to_source_cell_line_alias_map": "missing",
            "per_fit_source_identity_inputs_and_executed_mask_log": "missing",
            "per_fold_slice_seed_checkpoint_lineage": "missing",
            "matched_source_opportunity_receipts_across_five_arms": "missing",
            "source_exposure_zero_overlap_claim": "not_supported",
        },
        "minimum_missing_evidence": [
            {
                "needed": "versioned authority map from target patient/sample keys to source cell-line aliases",
                "known_related_manifest_path": "data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1/manifest.json",
                "known_related_manifest_sha256": "c6f9c3349b0a8598280ba099381b781c68ca5599f6f3b7e749090a0e46460b72",
                "reason": "the approved source input manifest does not contain the alias bridge",
            },
            {
                "needed": "one per-target-fold fit receipt binding source IDs, applied mask hash, and checkpoint hash",
                "reason": "frozen membership does not prove the actual source inputs or mask execution",
            },
            {
                "needed": "per-slice and initialization checkpoint lineage with matched source-opportunity counts for P-A/P-C/P-R/P-T/P-N",
                "reason": "the known fold0 legacy checkpoint is not bound to EXP-046 target folds or masks",
            },
        ],
        "access_limits": {
            "labels_read": False,
            "expression_or_perturbation_values_read": False,
            "predictions_or_metrics_read": False,
            "patient_identity_values_written": False,
            "checkpoint_weights_loaded": False,
            "training_or_cuda": False,
        },
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    receipt = build_receipt(repo_root)
    output_path = repo_root / OUTPUT_REL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_path.replace(output_path)
    print(json.dumps({"status": receipt["status"], "receipt": OUTPUT_REL.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
