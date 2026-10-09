#!/usr/bin/env python
"""Root-only readback of the partial EXP-067 fold-0 cost-gate artifact.

This deliberately cannot mark EXP-067 scientifically complete: it verifies the
single-fold partial artifact and the registered reason to stop at the CPU gate.
It reads only the authorized CTRP TRAIN+DEV-derived fold-0 outputs and their
frozen metadata receipt; it does not load PDO, outer/locked, PANACEA, or CDS-B
numeric data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


RUN_ID = "20261008_exp067_nested_response_benefit_v3"
EXPECTED_METADATA_RECEIPT = "dc4b649d960a326916bb53c1734c0204f67bd877e4cce9467f35a255779633c9"
ARMS = (
    "pretreatment_rna_plus_structure",
    "pretreatment_rna_plus_structure_plus_crossfitted_predicted_response",
    "equal_capacity_direct_rff",
    "object_shuffled_crossfitted_response",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual: Any, expected: Any, tolerance: float = 1e-9) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    return bool(np.isclose(float(actual), float(expected), rtol=tolerance, atol=tolerance))


def recompute(frame: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    object_rows: list[dict[str, Any]] = []
    for object_id, positions in frame.groupby("source_object_id", sort=True).indices.items():
        ix = np.asarray(positions, dtype=np.int64)
        local = frame.iloc[ix]
        y = local["aac_published"].to_numpy(dtype=np.float64)
        row: dict[str, Any] = {"source_object_id": str(object_id), "n_pairs": int(len(ix))}
        for arm in ARMS:
            pred = local[f"prediction__{arm}"].to_numpy(dtype=np.float64)
            row[f"{arm}__rmse"] = float(np.sqrt(np.mean(np.square(pred - y))))
            within_cell_rhos: list[float] = []
            centered_direction: list[float] = []
            for _, cell_positions in local.groupby("cellid", sort=True).indices.items():
                cell_ix = np.asarray(cell_positions, dtype=np.int64)
                cell_y = y[cell_ix]
                cell_pred = pred[cell_ix]
                if len(cell_ix) >= 2 and np.ptp(cell_y) > 0 and np.ptp(cell_pred) > 0:
                    rho = spearmanr(cell_y, cell_pred).statistic
                    if np.isfinite(rho):
                        within_cell_rhos.append(float(rho))
                if len(cell_ix) >= 2:
                    centered_y = cell_y - np.mean(cell_y)
                    centered_pred = cell_pred - np.mean(cell_pred)
                    centered_direction.append(float(np.mean(np.sign(centered_y) == np.sign(centered_pred))))
            row[f"{arm}__within_cell_drug_spearman"] = (
                float(np.mean(within_cell_rhos)) if within_cell_rhos else None
            )
            row[f"{arm}__spearman_cell_denominator"] = len(within_cell_rhos)
            row[f"{arm}__within_cell_centered_directional_agreement"] = (
                float(np.mean(centered_direction)) if centered_direction else None
            )
        object_rows.append(row)

    objects = pd.DataFrame(object_rows)
    totals: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        rho = pd.to_numeric(objects[f"{arm}__within_cell_drug_spearman"], errors="coerce")
        direction = pd.to_numeric(objects[f"{arm}__within_cell_centered_directional_agreement"], errors="coerce")
        totals[arm] = {
            "pair_count": int(objects["n_pairs"].sum()),
            "patient_count": int(len(objects)),
            "patient_macro_rmse": float(objects[f"{arm}__rmse"].mean()),
            "patient_macro_rmse_denominator": int(objects[f"{arm}__rmse"].notna().sum()),
            "patient_macro_within_cell_drug_spearman": float(rho.mean()) if rho.notna().any() else None,
            "patient_spearman_denominator": int(rho.notna().sum()),
            "patient_macro_within_cell_centered_directional_agreement": (
                float(direction.mean()) if direction.notna().any() else None
            ),
        }
    return totals, objects


def verify(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "run_summary.json"
    fold_path = run_dir / "fold_0_summary.json"
    predictions_path = run_dir / "fold_0_pair_predictions.tsv"
    audit_path = run_dir / "pair_domain_audit.json"
    runner_readback_path = run_dir / "run_readback.json"
    gate_path = run_dir / "first_outer_fold_cost_gate.json"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fold_summary = json.loads(fold_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    runner_readback = json.loads(runner_readback_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))

    require(summary.get("run_id") == RUN_ID and gate.get("run_id") == RUN_ID, "run ID mismatch")
    require(summary.get("experiment_id") == "EXP-067", "wrong experiment")
    require(summary.get("metadata_gate_receipt_sha256") == EXPECTED_METADATA_RECEIPT,
            "frozen metadata receipt mismatch")
    require(summary.get("status") == "COST_GATE_FAILED", "run is not the registered cost-gate stop")
    require(summary.get("scientific_comparison_valid") is False, "partial run cannot be labeled a valid full comparison")
    require(summary.get("outer_folds_completed") == 1 and summary.get("outer_folds_required") == 5,
            "expected exactly the first of five outer folds")
    require(gate.get("status") == "FAIL" and gate.get("completed_outer_folds") == 1
            and gate.get("required_outer_folds") == 5, "cost-gate status/denominator mismatch")
    require(runner_readback.get("run_id") == RUN_ID, "runner readback run ID mismatch")
    require(runner_readback.get("status") == "PASS_ROOT_INPUT_RUNNER_READBACK",
            "runner input/readback status mismatch")

    require(sha256(summary_path) == runner_readback.get("run_summary_sha256"),
            "run summary hash differs from the runner readback")
    expected_prediction_sha = runner_readback.get("fold_prediction_sha256", {}).get("fold_0")
    require(expected_prediction_sha == sha256(predictions_path), "fold-0 prediction hash mismatch")
    require(fold_summary.get("fold", {}).get("fold_id") == 0, "fold summary is not fold 0")

    frame = pd.read_csv(predictions_path, sep="\t", dtype={
        "pair_id": "string", "source_object_id": "string", "cellid": "string", "full_inchikey": "string",
    })
    required = {"pair_id", "source_object_id", "cellid", "full_inchikey", "fold_id", "aac_published"}
    required |= {f"prediction__{arm}" for arm in ARMS}
    require(required.issubset(frame.columns), f"missing prediction columns: {sorted(required-set(frame.columns))}")
    require(len(frame) == 816, "fold-0 held-out pair count mismatch")
    require(frame["pair_id"].notna().all() and not frame["pair_id"].duplicated().any(),
            "pair identities are missing or duplicated")
    require(frame[["source_object_id", "cellid", "full_inchikey"]].notna().all().all(),
            "a key field is missing")
    require(set(frame["fold_id"].astype(int).unique()) == {0}, "predictions include another outer fold")
    require(frame["source_object_id"].nunique() == 7 and frame["full_inchikey"].nunique() == 138,
            "fold-0 object or exact-drug-key denominator mismatch")
    require((frame["pair_id"].astype(str) == frame["cellid"].astype(str) + "|" +
             frame["full_inchikey"].astype(str)).all(), "cell/drug pair key does not match its components")
    numeric_columns = ["aac_published", *[f"prediction__{arm}" for arm in ARMS]]
    numeric = frame[numeric_columns].to_numpy(dtype=np.float64)
    require(np.isfinite(numeric).all(), "non-finite label or prediction in the fixed fold denominator")

    expected_audit = {
        "metadata_pair_count": 4187,
        "finite_aac_row_count_before_cell_drug_aggregation": 4312,
        "duplicate_finite_aac_rows_averaged": 347,
        "metadata_pairs_without_finite_aac": 222,
        "included_finite_aac_pairs": 3965,
        "unique_finite_cell_drug_label_count": 3965,
    }
    require(audit == expected_audit and summary.get("pair_domain_audit") == audit,
            "CTRP pair-domain audit mismatch")

    fold = fold_summary["fold"]
    nested = fold.get("nested_response_receipt", {})
    train_ids = set(map(str, nested.get("outer_train_patient_ids", [])))
    predict_ids = set(map(str, nested.get("outer_predict_patient_ids", [])))
    require(nested.get("outer_fold_id") == 0 and nested.get("target_gene_count") == 978,
            "nested response fold or gene-axis denominator mismatch")
    require(len(train_ids) == 27 and len(predict_ids) == 7 and not (train_ids & predict_ids)
            and len(train_ids | predict_ids) == 34, "outer PatientID isolation/denominator mismatch")
    require(set(frame["source_object_id"].astype(str)) == predict_ids,
            "fold-0 predictions do not match the held-out PatientID set")
    inner = nested.get("inner_fold_provenance", [])
    require({int(item.get("inner_validation_fold_id", -1)) for item in inner} == {1, 2, 3, 4},
            "nested TRAIN response OOF is missing a fixed inner fold")
    covered: set[str] = set()
    for item in inner:
        inner_predict = set(map(str, item.get("prediction_patient_ids", [])))
        inner_train = set(map(str, item.get("training_patient_ids", [])))
        require(inner_predict and inner_train.isdisjoint(inner_predict | predict_ids),
                "nested PatientID response-fit leakage")
        require(inner_train | inner_predict == train_ids, "nested fold does not partition outer TRAIN")
        require(not (covered & inner_predict), "nested OOF object was predicted more than once")
        covered |= inner_predict
    require(covered == train_ids, "nested response OOF does not cover outer TRAIN exactly once")
    outer_model = nested.get("outer_model_provenance", {})
    require(set(map(str, outer_model.get("training_patient_ids", []))) == train_ids
            and set(map(str, outer_model.get("prediction_patient_ids", []))) == predict_ids,
            "outer response model PatientID provenance mismatch")

    recomputed, object_rows = recompute(frame)
    stored_metrics = fold_summary.get("metrics", {})
    for arm in ARMS:
        for metric, expected in recomputed[arm].items():
            require(close(stored_metrics.get(arm, {}).get(metric), expected),
                    f"fold-0 metric mismatch: {arm}/{metric}")

    object_metrics_path = run_dir / "fold_0_object_metrics.tsv"
    drug_metrics_path = run_dir / "fold_0_drug_metrics.tsv"
    require(fold_summary.get("object_metrics_sha256") == sha256(object_metrics_path),
            "fold-0 object metrics hash mismatch")
    require(fold_summary.get("drug_metrics_sha256") == sha256(drug_metrics_path),
            "fold-0 drug metrics hash mismatch")
    stored_objects = pd.read_csv(object_metrics_path, sep="\t", dtype={"source_object_id": "string"})
    require(len(stored_objects) == len(object_rows), "fold-0 object metric row count mismatch")
    stored_objects = stored_objects.sort_values("source_object_id").reset_index(drop=True)
    object_rows = object_rows.sort_values("source_object_id").reset_index(drop=True)
    for arm in ARMS:
        for suffix in ("rmse", "within_cell_drug_spearman", "spearman_cell_denominator",
                       "within_cell_centered_directional_agreement"):
            col = f"{arm}__{suffix}"
            require(col in stored_objects.columns and col in object_rows.columns,
                    f"object metric column missing: {col}")
            left = pd.to_numeric(stored_objects[col], errors="coerce").to_numpy(dtype=np.float64)
            right = pd.to_numeric(object_rows[col], errors="coerce").to_numpy(dtype=np.float64)
            require(np.allclose(left, right, rtol=1e-9, atol=1e-9, equal_nan=True),
                    f"independent object metric mismatch: {col}")

    package_cap = float(gate["package_cpu_cap_seconds"])
    shared_cap = float(gate["shared_AB_cpu_cap_seconds"])
    a_b_pre = float(gate["a_b_cpu_spent_before_seconds"])
    b_pre = float(gate["b_package_cpu_spent_before_seconds"])
    projected = float(gate["projected_total_package_cpu_seconds_including_preflight_and_margin"])
    projected_remainder = float(gate["projected_remaining_shared_AB_seconds"])
    model_closeout = float(gate["projected_model_and_closeout_cpu_seconds_with_20pct_margin"])
    require(close(projected, b_pre + model_closeout), "projected B package cost arithmetic mismatch")
    require(close(projected_remainder, shared_cap - a_b_pre - projected),
            "projected shared A+B remaining cost arithmetic mismatch")
    require(close(float(gate["measured_fold_cpu_seconds"]), 1123.9739088),
            "fold CPU measurement differs from frozen cost-gate record")
    require(projected > package_cap and projected_remainder < 0,
            "registered cost gate should fail both package and shared CPU budgets")
    require(close(float(gate["shared_loader_cpu_seconds"]), 1.2107959), "loader CPU measurement mismatch")
    require(float(gate["fold_denominator"]["training_pairs"]) == 3149
            and float(gate["fold_denominator"]["test_pairs"]) == 816
            and int(gate["fold_denominator"]["training_objects"]) == 27
            and int(gate["fold_denominator"]["test_objects"]) == 7,
            "first-fold train/test denominator mismatch")

    result = {
        "schema": "exp067_cost_gate_fold0_root_readback_v1",
        "experiment_id": "EXP-067",
        "run_id": RUN_ID,
        "root_verification_status": "PASS_PARTIAL_FOLD_AND_COST_GATE_ONLY",
        "scientific_comparison_valid": False,
        "completed_outer_folds": 1,
        "required_outer_folds": 5,
        "metadata_gate_receipt_sha256": EXPECTED_METADATA_RECEIPT,
        "denominators": {
            "frozen_source_conditions": 2830,
            "frozen_source_objects": 34,
            "full_chemical_keys": 138,
            "genes": 978,
            "finite_ct_p_pairs": 3965,
            "fold0_train_pairs": 3149,
            "fold0_train_objects": 27,
            "fold0_heldout_pairs": 816,
            "fold0_heldout_objects": 7,
            "fold0_heldout_drugs": 138,
        },
        "fold0_metrics_recomputed": recomputed,
        "cost_gate": {
            "b_preflight_seconds": b_pre,
            "measured_fold_cpu_seconds": float(gate["measured_fold_cpu_seconds"]),
            "projected_package_cpu_seconds_including_20pct_margin": projected,
            "package_cap_seconds": package_cap,
            "projected_excess_over_package_cap_seconds": projected - package_cap,
            "projected_remaining_shared_ab_seconds": projected_remainder,
            "shared_precharge_seconds": a_b_pre,
            "shared_cap_seconds": shared_cap,
        },
        "hashes": {
            "run_summary_sha256": sha256(summary_path),
            "fold0_predictions_sha256": sha256(predictions_path),
            "fold0_summary_sha256": sha256(fold_path),
            "fold0_object_metrics_sha256": sha256(object_metrics_path),
            "fold0_drug_metrics_sha256": sha256(drug_metrics_path),
            "cost_gate_sha256": sha256(gate_path),
            "pair_domain_audit_sha256": sha256(audit_path),
        },
        "read_scope": {
            "ctpr_numeric": "authorized TRAIN+DEV-derived fold0 label/prediction outputs only",
            "pdo": "not read",
            "outer_locked": "not read",
            "panacea_cds_b": "not read",
        },
        "interpretation": "One complete outer fold was independently checked. The pre-registered CPU cost gate failed; this receipt is not a full five-fold scientific comparison or acceptance.",
        "verified_at_local": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    result = verify(run_dir)
    receipt = args.receipt.resolve() if args.receipt else run_dir / "root_cost_gate_readback.json"
    if receipt.exists():
        raise FileExistsError(f"refusing to overwrite immutable receipt: {receipt}")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "receipt": receipt.as_posix(),
        "receipt_sha256": sha256(receipt),
        "root_verification_status": result["root_verification_status"],
        "scientific_comparison_valid": result["scientific_comparison_valid"],
        "fold0_metrics_recomputed": result["fold0_metrics_recomputed"],
        "cost_gate": result["cost_gate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
