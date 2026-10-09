"""Independent numerical readback for a completed EXP-056 fixed-split run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/modeling"))
import exp051_contract_fit_smoke as fit  # noqa: E402
import run_exp052_transigen_lr_stability as source_reader  # noqa: E402

METHODS = (
    "per_drug_mean", "state_rank4", "state_rank8", "no_state_rank4", "no_state_rank8",
    "shuffled_state_rank4", "shuffled_state_rank8",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_close(actual: float, expected: float, label: str, tol: float = 1e-9) -> float:
    error = abs(float(actual) - float(expected))
    if error > tol:
        raise ValueError(f"{label} mismatch: actual={actual}, expected={expected}, abs_error={error}")
    return error


def independent_metrics(keys: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray,
                        train_var: np.ndarray) -> dict[str, Any]:
    gene_mask = np.isfinite(train_var) & (train_var > 1e-12)
    if not gene_mask.any():
        raise ValueError("no TRAIN variance genes available")
    squared = np.square(prediction.astype(np.float64) - truth.astype(np.float64))
    normalized = squared[:, gene_mask] / train_var[gene_mask][None, :]
    row_nmse = normalized.mean(axis=1)
    row_mae = np.abs(prediction.astype(np.float64) - truth.astype(np.float64)).mean(axis=1)
    row_dir = (np.sign(prediction) == np.sign(truth)).mean(axis=1)
    row_corr = np.full(len(keys), np.nan, dtype=np.float64)
    for i in range(len(keys)):
        y = truth[i].astype(np.float64)
        p = prediction[i].astype(np.float64)
        if np.std(y) > 0 and np.std(p) > 0:
            row_corr[i] = float(np.corrcoef(y, p)[0, 1])
    by_object: dict[str, dict[str, Any]] = {}
    obj_ids = keys.source_object_id.astype(str).to_numpy()
    for obj in sorted(set(obj_ids)):
        ix = np.flatnonzero(obj_ids == obj)
        corr = row_corr[ix][np.isfinite(row_corr[ix])]
        y_abs, p_abs = np.abs(truth[ix]).mean(), np.abs(prediction[ix]).mean()
        by_object[obj] = {
            "nmse": float(row_nmse[ix].mean()),
            "mae": float(row_mae[ix].mean()),
            "condition_gene_correlation": float(corr.mean()) if len(corr) else None,
            "direction_agreement": float(row_dir[ix].mean()),
            "amplitude_ratio": float(p_abs / y_abs) if y_abs > 0 else None,
            "conditions": int(len(ix)),
            "condition_correlations_defined": int(len(corr)),
        }
    metrics = ("nmse", "mae", "condition_gene_correlation", "direction_agreement", "amplitude_ratio")
    summary: dict[str, Any] = {"object_scores": by_object}
    for metric in metrics:
        vals = [by_object[obj][metric] for obj in by_object if by_object[obj][metric] is not None]
        summary[f"object_macro_{metric}"] = float(np.mean(vals)) if vals else None
    summary.update({"objects": len(by_object), "conditions": len(keys),
                    "condition_gene_correlations_defined": int(np.isfinite(row_corr).sum()),
                    "genes_in_nmse": int(gene_mask.sum()),
                    "zero_or_invalid_variance_genes_excluded": int((~gene_mask).sum())})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--write-receipt", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    out = ROOT / "artifacts/experiments/EXP-056" / args.run_id
    if not out.is_dir():
        raise FileNotFoundError(f"EXP-056 run not found: {out}")
    recovery_result_path = out / "serialization_recovery_001/recovery_result.json"
    result_path = recovery_result_path if recovery_result_path.is_file() else out / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    cfg = __import__("yaml").safe_load((ROOT / "configs/experiments/EXP-056-low-rank-state-drug-interaction.yaml").read_text(encoding="utf-8"))
    if result.get("experiment_id") != "EXP-056" or result.get("run_id") != args.run_id:
        raise ValueError("run identity does not match the registered EXP-056 result")
    source_id = cfg["data"]["source_run_id"]
    fit.configure_source_run(source_id)
    src = fit.SOURCE_RUN_DIR
    keys_all = pd.read_csv(src / "response_blind_conditions.tsv", sep="\t", dtype=str, keep_default_na=False)
    train_keys = keys_all.loc[keys_all.split_role.eq("train")].reset_index(drop=True)
    dev_keys = keys_all.loc[keys_all.split_role.eq("inner_dev")].reset_index(drop=True)
    axis_map = pd.read_csv(src / "gene_axis_mapping.tsv", sep="\t", dtype=str, keep_default_na=False)
    genes = axis_map.h5ad_var_name.astype(str).to_numpy(dtype=str)
    if (len(train_keys), train_keys.source_object_id.nunique(), len(dev_keys),
            dev_keys.source_object_id.nunique(), keys_all.full_inchikey.nunique(), len(genes)) != (419, 110, 93, 24, 5, 978):
        raise ValueError("frozen source denominator changed")
    if set(train_keys.source_object_id) & set(dev_keys.source_object_id):
        raise ValueError("TRAIN and inner-dev source object overlap detected")
    values_train = source_reader.read_role_expression(train_keys, genes)
    values_dev = source_reader.read_role_expression(dev_keys, genes)
    train_truth = values_train["delta"].astype(np.float64)
    dev_truth = values_dev["delta"].astype(np.float64)
    train_var = np.var(train_truth, axis=0, ddof=0)

    prediction_path = out / result["predictions"]["file"]
    if sha256(prediction_path) != result["predictions"]["sha256"]:
        raise ValueError("prediction archive does not match result.json hash")
    with np.load(prediction_path, allow_pickle=False) as archive:
        saved = {name: archive[name].copy() for name in archive.files}
    expected_axes = {
        "train_condition_key": train_keys.full_condition_key.astype(str).to_numpy(),
        "inner_dev_condition_key": dev_keys.full_condition_key.astype(str).to_numpy(),
        "train_source_object_id": train_keys.source_object_id.astype(str).to_numpy(),
        "inner_dev_source_object_id": dev_keys.source_object_id.astype(str).to_numpy(),
        "train_full_inchikey": train_keys.full_inchikey.astype(str).to_numpy(),
        "inner_dev_full_inchikey": dev_keys.full_inchikey.astype(str).to_numpy(),
        "gene_axis": genes.astype(str),
    }
    axis_errors: dict[str, bool] = {}
    for name, exp in expected_axes.items():
        axis_errors[name] = bool(np.array_equal(saved[name].astype(str), exp))
        if not axis_errors[name]:
            raise ValueError(f"saved identity axis differs from frozen source: {name}")
    if not np.allclose(saved["train_target_delta"], train_truth, atol=1e-6, rtol=0):
        raise ValueError("saved TRAIN target differs from frozen H5AD rows")
    if not np.allclose(saved["inner_dev_target_delta"], dev_truth, atol=1e-6, rtol=0):
        raise ValueError("saved inner-dev target differs from frozen H5AD rows")

    result_metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    rebuilt: dict[str, dict[str, Any]] = {}
    metric_error = 0.0
    for method in METHODS:
        train_pred = saved[f"{method}_train"].astype(np.float64)
        dev_pred = saved[f"{method}_inner_dev"].astype(np.float64)
        if (train_pred.shape != (419, 978) or dev_pred.shape != (93, 978)
                or not np.isfinite(train_pred).all() or not np.isfinite(dev_pred).all()):
            raise ValueError(f"prediction dimensions or finite checks failed for {method}")
        train_metrics = independent_metrics(train_keys, train_truth, train_pred, train_var)
        dev_metrics = independent_metrics(dev_keys, dev_truth, dev_pred, train_var)
        rebuilt[method] = {"train_in_sample": train_metrics, "inner_dev": dev_metrics}
        for role_name, recomputed in (("train_in_sample", train_metrics), ("inner_dev", dev_metrics)):
            saved_summary = result_metrics[method][role_name]
            for k in ("object_macro_nmse", "object_macro_mae", "object_macro_condition_gene_correlation",
                      "object_macro_direction_agreement", "object_macro_amplitude_ratio"):
                a, b = saved_summary[k], recomputed[k]
                if a is None and b is None:
                    continue
                metric_error = max(metric_error, assert_close(a, b, f"{method}/{role_name}/{k}", tol=5e-7))
            saved_objects = saved_summary["object_scores"]
            if set(saved_objects) != set(recomputed["object_scores"]):
                raise ValueError(f"object metric denominator differs for {method}/{role_name}")
            for obj in recomputed["object_scores"]:
                for k in ("nmse", "mae", "direction_agreement", "conditions"):
                    metric_error = max(metric_error, assert_close(
                        saved_objects[obj][k], recomputed["object_scores"][obj][k],
                        f"{method}/{role_name}/{obj}/{k}", tol=5e-7))

    decision = json.loads((out / "decision.json").read_text(encoding="utf-8"))
    dev_nmse = {m: rebuilt[m]["inner_dev"]["object_macro_nmse"] for m in METHODS}
    selected = min(("state_rank4", "state_rank8"), key=lambda m: (dev_nmse[m], m))
    controls = ["per_drug_mean", "no_state_rank4", "no_state_rank8", "shuffled_state_rank4", "shuffled_state_rank8"]
    strongest = min(controls, key=lambda m: (dev_nmse[m], m))
    object_ids = sorted(rebuilt[selected]["inner_dev"]["object_scores"])
    diff = np.asarray([rebuilt[strongest]["inner_dev"]["object_scores"][o]["nmse"]
                       - rebuilt[selected]["inner_dev"]["object_scores"][o]["nmse"]
                       for o in object_ids])
    seed, reps = int(cfg["metrics"]["bootstrap_seed"]), int(cfg["metrics"]["bootstrap_replicates"])
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, len(object_ids), size=(reps, len(object_ids)))
    ci = np.quantile(diff[boot_idx].mean(axis=1), [0.025, 0.975]).tolist()
    rel = ((dev_nmse[strongest] - dev_nmse[selected]) / dev_nmse[strongest]
           if dev_nmse[strongest] > 0 else math.nan)
    wins = int(np.sum(diff > 0))
    amp = rebuilt[selected]["inner_dev"]["object_macro_amplitude_ratio"]
    gate = bool(rel >= 0.05 and ci[0] > 0 and wins >= 15 and amp is not None and 0.5 <= amp <= 2.0)
    if (selected != decision["selected_interaction_rank"] or strongest != decision["strongest_control"]
            or abs(rel - decision["relative_nmse_reduction_vs_strongest_control"]) > 1e-8
            or wins != decision["objects_improved_of_24"] or gate != decision["development_gate_pass"]
            or not np.allclose(ci, decision["paired_object_bootstrap_95pct_ci"], atol=1e-8, rtol=0)):
        raise ValueError("independent rank selection / paired bootstrap / development-gate readback differs")

    mapping = json.loads((out / "randomization_mappings.json").read_text(encoding="utf-8"))
    mapping_checks = {}
    for role, expected_keys in (("train", set(train_keys.source_object_id.astype(str))),
                                ("inner_dev", set(dev_keys.source_object_id.astype(str)))):
        m = mapping[role]
        mapping_checks[role] = bool(set(m) == expected_keys and set(m.values()) == expected_keys
                                    and all(k != v for k, v in m.items()))
        if not mapping_checks[role]:
            raise ValueError(f"{role} shuffle is not a within-role derangement")
    if mapping.get("checks", {}).get("label_used_for_mapping") is not False:
        raise ValueError("shuffle map receipt does not declare label blindness")

    manifest_checks: dict[str, dict[str, bool]] = {}
    root_manifest = pd.read_csv(out / "sha256_manifest.tsv", sep="\t", dtype=str)
    manifest_checks["initial_run"] = {str(row.file): sha256(out / str(row.file)) == str(row.sha256)
                                      for row in root_manifest.itertuples(index=False)}
    if recovery_result_path.is_file():
        recovery_dir = out / "serialization_recovery_001"
        rec_manifest = pd.read_csv(recovery_dir / "sha256_manifest.tsv", sep="\t", dtype=str)
        manifest_checks["serialization_recovery"] = {
            str(row.file): sha256(recovery_dir / str(row.file)) == str(row.sha256)
            for row in rec_manifest.itertuples(index=False)}
    if not all(checks and all(checks.values()) for checks in manifest_checks.values()):
        raise ValueError("one or more run or recovery manifest SHA-256 checks failed")
    receipt = {
        "status": "PASS_EXP056_INDEPENDENT_NUMERICAL_READBACK",
        "run_id": args.run_id,
        "source_run_id": source_id,
        "verified_at_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds").replace("+00:00", "Z"),
        "checks": {
            "denominators": {"train_conditions": len(train_keys), "train_objects": int(train_keys.source_object_id.nunique()),
                             "inner_dev_conditions": len(dev_keys), "inner_dev_objects": int(dev_keys.source_object_id.nunique()),
                             "train_inner_dev_object_overlap": 0, "full_inchikeys": int(keys_all.full_inchikey.nunique()),
                             "genes": len(genes)},
            "identity_axis_match": axis_errors,
            "saved_truth_max_abs_error": float(max(np.max(np.abs(saved["train_target_delta"].astype(np.float64) - train_truth)),
                                                     np.max(np.abs(saved["inner_dev_target_delta"].astype(np.float64) - dev_truth)))),
            "all_predictions_finite_and_correct_shape": True,
            "recomputed_metric_max_abs_error": metric_error,
            "selected_rank": selected, "strongest_control": strongest,
            "recomputed_relative_reduction": float(rel),
            "recomputed_paired_ci": [float(x) for x in ci],
            "recomputed_improved_objects": f"{wins}/24",
            "recomputed_gate_pass": gate,
            "derangements_reconstructed_and_valid": mapping_checks,
            "manifest_sha256_checks": manifest_checks,
            "pdo_numeric_sensitivity_opened": False,
            "outer_locked_response_opened": False,
            "external_response_matrix_opened": False,
        },
        "resource": {"gpu_hours": 0.0, "cpu_seconds": time.monotonic() - started},
        "claim_boundary": "readback verifies saved fixed-split source-development comparison only; not independent biological validation",
    }
    if args.write_receipt:
        target = out / "root_independent_readback.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2,
                                  allow_nan=False) + "\n", encoding="utf-8")
        tmp.replace(target)
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
