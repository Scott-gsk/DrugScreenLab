#!/usr/bin/env python3
"""Root-style independent readback for the fixed EXP-065 transform."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from verify_exp064_state_neighbor_oof import metrics as exp064_metrics

ROOT = Path(__file__).resolve().parents[2]
BASE_ARMS = ("zero_change", "per_drug_mean", "nearest3_state", "random3_same_drug")
ARMS = BASE_ARMS + ("nearest3_count_shrunk", "random3_count_shrunk")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def check(cfg_path: Path, run_id: str) -> dict[str, Any]:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("experiment_id") != "EXP-065" or cfg.get("run_id") != run_id:
        raise ValueError("EXP-065 config/run identity mismatch")
    source = ROOT / cfg["source"]["attempt_dir"]
    if sha256(source / "root_independent_readback.json") != cfg["source"]["source_receipt_sha256"]:
        raise ValueError("EXP-064 PASS source receipt hash mismatch")
    if sha256(source / "keyed_oof_predictions.npz") != cfg["source"]["source_prediction_sha256"]:
        raise ValueError("EXP-064 frozen OOF prediction hash mismatch")
    if sha256(source / "fold_assignment.tsv") != cfg["source"]["fold_assignment_sha256"]:
        raise ValueError("EXP-064 TRAIN fold metadata hash mismatch")
    with np.load(source / "keyed_oof_predictions.npz", allow_pickle=False) as z:
        original = {k: z[k].copy() for k in z.files}
    fold = pd.read_csv(source / "fold_assignment.tsv", sep="\t", dtype=str, keep_default_na=False)
    fold["fold_id"] = pd.to_numeric(fold["fold_id"], errors="raise").astype(np.int16)
    for col in ("full_condition_key", "source_object_id", "full_inchikey"):
        fold[col] = fold[col].astype(str).str.strip()
    manifest = pd.read_csv(source / "neighbor_selection.tsv", sep="\t", dtype=str, keep_default_na=False)
    manifest["fold_id"] = pd.to_numeric(manifest["fold_id"], errors="raise").astype(np.int16)
    manifest["candidate_train_objects"] = pd.to_numeric(manifest["candidate_train_objects"], errors="raise").astype(np.int32)
    keys = pd.Index(original["full_condition_key"].astype(str))
    if set(keys) != set(fold.full_condition_key) or len(keys) != 1291 or keys.has_duplicates:
        raise ValueError("source OOF / frozen TRAIN metadata key denominator mismatch")
    aligned = fold.set_index("full_condition_key").loc[keys].reset_index()
    for field in ("source_object_id", "full_inchikey", "fold_id"):
        if not np.array_equal(aligned[field].astype(str).to_numpy(), original[field].astype(str)):
            raise ValueError(f"root metadata identity mismatch: {field}")
    if aligned.source_object_id.nunique() != 93 or aligned.full_inchikey.nunique() != 20:
        raise ValueError("source denominator is not 93 objects / 20 full keys")
    n = np.empty(len(keys), dtype=np.int16)
    count_rows = []
    for f in range(5):
        train = fold.loc[fold.fold_id.ne(f)]
        count = train.groupby("full_inchikey", sort=True).source_object_id.nunique().to_dict()
        idx = np.flatnonzero(original["fold_id"] == f)
        for i in idx:
            n[i] = int(count[str(original["full_inchikey"][i])])
        for drug, n_objects in sorted(count.items()):
            held = fold.loc[fold.fold_id.eq(f) & fold.full_inchikey.eq(str(drug))]
            train_ids = sorted(train.loc[train.full_inchikey.eq(str(drug)), "source_object_id"].unique())
            count_rows.append({"fold_id": int(f), "full_inchikey": str(drug),
                               "heldout_conditions": int(len(held)),
                               "heldout_objects": int(held.source_object_id.nunique()),
                               "N_train_same_drug_objects": int(n_objects),
                               "lambda": 3.0 / (int(n_objects) + 3.0),
                               "train_object_ids_sha256": hashlib.sha256("\n".join(train_ids).encode("utf-8")).hexdigest()})
    m = manifest.set_index("full_condition_key").loc[keys].reset_index()
    for field, expected in (("test_source_object_id", original["source_object_id"].astype(str)),
                            ("full_inchikey", original["full_inchikey"].astype(str)),
                            ("fold_id", original["fold_id"].astype(np.int16))):
        if not np.array_equal(m[field].astype(str).to_numpy(), expected.astype(str)):
            raise ValueError(f"root independent neighbor manifest identity mismatch: {field}")
    if not np.array_equal(m.candidate_train_objects.to_numpy(dtype=np.int32), n.astype(np.int32)):
        raise ValueError("independent TRAIN count differs from frozen nearest-neighbor manifest")
    lam = 3.0 / (n.astype(np.float64) + 3.0)
    mean = original["pred_per_drug_mean"].astype(np.float64)
    near = original["pred_nearest3_state"].astype(np.float64)
    rand = original["pred_random3_same_drug"].astype(np.float64)
    expected = {
        "nearest3_count_shrunk": mean + lam[:, None] * (near - mean),
        "random3_count_shrunk": mean + lam[:, None] * (rand - mean),
    }
    out_dir = ROOT / cfg["output"]["root"] / cfg["output"]["attempt_id"]
    with np.load(out_dir / cfg["output"]["keyed_predictions"], allow_pickle=False) as z:
        saved = {k: z[k].copy() for k in z.files}
    for key in ("full_condition_key", "source_object_id", "full_inchikey", "source_row", "fold_id", "y_true", "common_denominator", "fold_gene_variance"):
        if not np.array_equal(saved[key], original[key]):
            raise ValueError(f"saved EXP-065 changed frozen EXP-064 field: {key}")
    if not np.array_equal(saved["N_train_same_drug_objects"], n) or not np.allclose(saved["lambda"], lam, atol=0, rtol=0):
        raise ValueError("saved N/lambda does not follow fixed TRAIN-only count formula")
    pred_errors = {}
    for method in BASE_ARMS:
        if not np.array_equal(saved[f"pred_{method}"], original[f"pred_{method}"]):
            raise ValueError(f"saved original arm changed: {method}")
    for method, exp in expected.items():
        err = float(np.max(np.abs(saved[f"pred_{method}"].astype(np.float64) - exp)))
        pred_errors[method] = err
        if not np.allclose(saved[f"pred_{method}"], exp, atol=1e-12, rtol=1e-12):
            raise ValueError(f"independent fixed formula mismatch for {method}")

    keys_df = pd.DataFrame({"full_condition_key": original["full_condition_key"].astype(str),
                            "source_object_id": original["source_object_id"].astype(str),
                            "full_inchikey": original["full_inchikey"].astype(str), "fold_id": original["fold_id"].astype(np.int16)})
    variance = original["fold_gene_variance"].astype(np.float64)
    truth = original["y_true"].astype(np.float64)
    common = original["common_denominator"].astype(bool)
    arms = {m: original[f"pred_{m}"] for m in BASE_ARMS}
    arms.update({m: expected[m] for m in expected})
    summaries, obj_names = {}, {}
    obj_tables = []
    for method in ARMS:
        summary, obj = exp064_metrics(keys_df, truth, arms[method], variance, common)
        summaries[method] = summary
        obj_names[method] = obj.set_index("source_object_id").nmse
        obj = obj.rename(columns={c: f"{method}_{c}" for c in obj.columns if c != "source_object_id"})
        obj_tables.append((method, obj.set_index("source_object_id")))
    obj_table = obj_tables[0][1].reset_index()
    for _, frame in obj_tables[1:]:
        obj_table = obj_table.merge(frame.reset_index(), on="source_object_id", how="outer", validate="one_to_one")
    obj_to_drug = keys_df.groupby("source_object_id", sort=True).full_inchikey.agg(
        lambda x: ";".join(sorted(set(x.astype(str)))))
    obj_table = obj_table.merge(obj_to_drug.rename("full_inchikeys"), left_on="source_object_id", right_index=True,
                                validate="one_to_one")
    primary_delta = float(summaries["nearest3_count_shrunk"]["object_macro_nmse"] - summaries["per_drug_mean"]["object_macro_nmse"])
    random_delta = float(summaries["random3_count_shrunk"]["object_macro_nmse"] - summaries["per_drug_mean"]["object_macro_nmse"])

    # Independently reconstruct row contributions, then condition-average within
    # each object and give all 93 held-out objects equal weight.
    fold_ids = keys_df.fold_id.to_numpy(dtype=int)
    var_rows = variance[fold_ids]
    usable = np.isfinite(var_rows) & (var_rows > 1e-12)
    mu = arms["per_drug_mean"].astype(np.float64)
    err_mu = mu - truth
    row_components = {}
    for label, source_pred in (("nearest", near), ("random", rand)):
        # The frozen equation subtracts twice the covariance, so define
        # d = drug_mean - three-object prediction.
        d = mu - source_pred.astype(np.float64)
        lam_rows = lam[:, None]
        residual = np.divide(lam_rows ** 2 * d ** 2, var_rows, out=np.zeros_like(d), where=usable)
        covariance = np.divide(2.0 * lam_rows * d * err_mu, var_rows, out=np.zeros_like(d), where=usable)
        row_components[label] = {"residual": residual.mean(axis=1), "covariance": covariance.mean(axis=1),
                                 "delta": (residual - covariance).mean(axis=1)}
    decomp_rows = []
    for obj, group in keys_df.groupby("source_object_id", sort=True):
        idx = group.index.to_numpy(dtype=np.int64)
        if len(np.unique(fold_ids[idx])) != 1:
            raise ValueError("root decomposition found held-out object in multiple folds")
        row = {"fold_id": int(fold_ids[idx[0]]), "source_object_id": str(obj),
               "heldout_conditions": int(len(idx)),
               "full_inchikeys": ";".join(sorted(set(keys_df.full_inchikey.iloc[idx].astype(str))))}
        for label in ("nearest", "random"):
            for component in ("residual", "covariance", "delta"):
                key = f"{label}_{component}_component" if component != "delta" else f"{label}_formula_delta_nmse"
                row[key] = float(row_components[label][component][idx].mean())
        decomp_rows.append(row)
    decomp = pd.DataFrame(decomp_rows)
    decomp_summary = {
        "nearest": {"nearest_residual_variance_component_object_macro": float(decomp.nearest_residual_component.mean()),
                    "nearest_twice_covariance_component_object_macro": float(decomp.nearest_covariance_component.mean()),
                    "nearest_formula_delta_nmse_object_macro": float(decomp.nearest_formula_delta_nmse.mean())},
        "random": {"random_residual_variance_component_object_macro": float(decomp.random_residual_component.mean()),
                   "random_twice_covariance_component_object_macro": float(decomp.random_covariance_component.mean()),
                   "random_formula_delta_nmse_object_macro": float(decomp.random_formula_delta_nmse.mean())},
        "weighting": "first mean conditions within object, then equal weight each heldout object; preserve per-fold TRAIN gene variance",
        "object_denominator": int(decomp.source_object_id.nunique()),
    }
    if abs(decomp_summary["nearest"]["nearest_formula_delta_nmse_object_macro"] - primary_delta) > 1e-6:
        raise ValueError("nearest shrinkage decomposition does not equal observed object-macro NMSE change")
    if abs(decomp_summary["random"]["random_formula_delta_nmse_object_macro"] - random_delta) > 1e-6:
        raise ValueError("random shrinkage decomposition does not equal observed object-macro NMSE change")

    metrics_path = out_dir / cfg["output"]["metrics"]
    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for method in ARMS:
        for metric in summaries[method]:
            x = summaries[method].get(metric)
            y = saved_metrics["metrics_by_method"][method].get(metric)
            if isinstance(x, (int, np.integer)):
                if int(x) != int(y):
                    raise ValueError(f"metrics independent denominator mismatch {method}.{metric}")
            elif x is None or y is None:
                if x is not None or y is not None:
                    raise ValueError(f"metrics null mismatch {method}.{metric}")
            elif abs(float(x) - float(y)) > 1e-9:
                raise ValueError(f"metrics independent readback mismatch {method}.{metric}")

    # Independently verify all required tabular deliverables, not only headline scores.
    def compare_table(path: Path, expected_frame: pd.DataFrame, label: str, atol: float = 1e-9) -> None:
        actual = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        expected = expected_frame.copy().reset_index(drop=True)
        actual = actual.reset_index(drop=True)
        if list(actual.columns) != list(expected.columns) or len(actual) != len(expected):
            raise ValueError(f"{label} shape/schema mismatch")
        for col in expected.columns:
            if pd.api.types.is_numeric_dtype(expected[col]):
                raw = actual[col].astype(str)
                av_series = pd.to_numeric(raw, errors="coerce")
                malformed = av_series.isna() & raw.ne("")
                if malformed.any():
                    raise ValueError(f"{label}.{col} contains malformed numeric values")
                av = av_series.to_numpy(dtype=np.float64)
                ev = expected[col].to_numpy(dtype=np.float64)
                if not np.allclose(av, ev, atol=atol, rtol=1e-9, equal_nan=True):
                    raise ValueError(f"{label}.{col} numeric readback mismatch")
            elif not np.array_equal(actual[col].astype(str).to_numpy(), expected[col].astype(str).to_numpy()):
                raise ValueError(f"{label}.{col} identity/readback mismatch")

    count_expected = pd.DataFrame(count_rows)
    compare_table(out_dir / cfg["output"]["count_table"], count_expected, "count_table", atol=1e-12)
    compare_table(out_dir / cfg["output"]["object_metrics"], obj_table, "object_metrics")
    per_drug_rows = []
    for drug in sorted(keys_df.full_inchikey.unique()):
        mask = keys_df.full_inchikey.eq(drug).to_numpy()
        subset = keys_df.loc[mask].reset_index(drop=True)
        for method in ARMS:
            summary, _ = exp064_metrics(subset, truth[mask], arms[method][mask], variance, common[mask])
            per_drug_rows.append({"full_inchikey": drug, "method": method, **summary})
    per_drug_expected = pd.DataFrame(per_drug_rows)
    compare_table(out_dir / cfg["output"]["per_drug_metrics"], per_drug_expected, "per_drug_metrics")
    paired_rows = []
    bootstrap_expected = {}
    rng = np.random.default_rng(int(cfg["metrics"]["paired_object_bootstrap_seed"]))
    reps = int(cfg["metrics"]["paired_object_bootstrap_replicates"])
    for contrast, (base, candidate) in {
        "nearest_shrunk_vs_per_drug_mean": ("per_drug_mean", "nearest3_count_shrunk"),
        "nearest_shrunk_vs_random_shrunk": ("random3_count_shrunk", "nearest3_count_shrunk"),
        "nearest_original_vs_per_drug_mean": ("per_drug_mean", "nearest3_state"),
        "random_original_vs_per_drug_mean": ("per_drug_mean", "random3_same_drug"),
    }.items():
        pair = obj_table[["source_object_id", f"{base}_nmse"]].rename(columns={f"{base}_nmse": "baseline_nmse"}).merge(
            obj_table[["source_object_id", f"{candidate}_nmse"]].rename(columns={f"{candidate}_nmse": "candidate_nmse"}),
            on="source_object_id", validate="one_to_one")
        pair["improvement_baseline_minus_candidate"] = pair.baseline_nmse - pair.candidate_nmse
        pair["contrast"] = contrast
        paired_rows.append(pair)
        d = pair.improvement_baseline_minus_candidate.to_numpy(dtype=np.float64)
        samples = rng.integers(0, len(d), size=(reps, len(d)))
        bm = d[samples].mean(axis=1)
        bootstrap_expected[contrast] = {"baseline": base, "candidate": candidate, "object_denominator": int(len(d)),
            "mean_paired_improvement_baseline_minus_candidate": float(np.mean(d)),
            "ci95": [float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))],
            "objects_candidate_better": int(np.sum(d > 0)), "objects_candidate_worse": int(np.sum(d < 0)),
            "bootstrap_replicates": reps, "bootstrap_seed": int(cfg["metrics"]["paired_object_bootstrap_seed"])}
    compare_table(out_dir / cfg["output"]["paired_object_differences"], pd.concat(paired_rows, ignore_index=True), "paired_object_differences")
    if saved_metrics["paired_object_bootstrap"] != bootstrap_expected:
        raise ValueError("paired object bootstrap readback mismatch")
    compare_table(out_dir / cfg["output"]["decomposition_table"], decomp, "decomposition_table")
    saved_decomp = json.loads((out_dir / cfg["output"]["normalized_error_decomposition"]).read_text(encoding="utf-8"))
    if saved_decomp != saved_metrics["error_decomposition"]:
        raise ValueError("metrics/decomposition JSON mismatch")
    for arm in ("nearest", "random"):
        for k, v in decomp_summary[arm].items():
            if abs(float(saved_decomp[arm][k]) - float(v)) > 1e-10:
                raise ValueError("normalized decomposition JSON readback mismatch")
    if saved_decomp.get("weighting") != decomp_summary["weighting"] or saved_decomp.get("object_denominator") != 93:
        raise ValueError("normalized decomposition weighting/denominator mismatch")
    manifest_saved = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest_saved.get("status") != "COMPLETED_ROOT_READBACK_PENDING" or not manifest_saved.get("full_folds_completed"):
        raise ValueError("runner manifest incomplete or not in expected pending-readback state")
    runner_charge = float(manifest_saved["conservative_charge_including_precharge_seconds"])
    if runner_charge > min(float(cfg["resources"]["cpu_package_cap_seconds"]),
                           float(cfg["resources"]["inherited_exp055_cpu_remaining_start_seconds"])):
        raise ValueError(f"runner CPU charge exceeds cap: {runner_charge:.3f}s")
    if saved_metrics["denominators"] != {"conditions": 1291, "objects": 93, "full_inchikeys": 20,
                                         "genes": 978, "common_conditions": 1291, "common_objects": 93,
                                         "missing_conditions": 0}:
        raise ValueError("EXP-065 common denominator mismatch")
    gain = ((float(summaries["per_drug_mean"]["object_macro_nmse"])
             - float(summaries["nearest3_count_shrunk"]["object_macro_nmse"]))
            / float(summaries["per_drug_mean"]["object_macro_nmse"]))
    if abs(gain - float(saved_metrics["fixed_gate"]["nearest_shrunk_relative_nmse_improvement_vs_mean"])) > 1e-12:
        raise ValueError("relative gain in fixed decision gate does not reproduce")
    gate_pass = (gain >= 0.05
                 and float(summaries["nearest3_count_shrunk"]["object_macro_nmse"]) < float(summaries["random3_count_shrunk"]["object_macro_nmse"]))
    if bool(saved_metrics["fixed_gate"]["gate_passed"]) != gate_pass:
        raise ValueError("predeclared close/retain gate mismatch")
    return {"status": "PASS_ROOT_INDEPENDENT_FIXED_COUNT_SHRINKAGE_READBACK",
            "run_id": run_id, "source_root_receipt_sha256": cfg["source"]["source_receipt_sha256"],
            "source_prediction_sha256": cfg["source"]["source_prediction_sha256"],
            "fold_assignment_sha256": cfg["source"]["fold_assignment_sha256"],
            "output_prediction_sha256": sha256(out_dir / cfg["output"]["keyed_predictions"]),
            "output_metrics_sha256": sha256(metrics_path), "denominators": saved_metrics["denominators"],
            "independently_counted_train_object_range": [int(n.min()), int(n.max())],
            "max_prediction_roundtrip_error": pred_errors,
            "nearest_shrunk_nmse": summaries["nearest3_count_shrunk"]["object_macro_nmse"],
            "per_drug_mean_nmse": summaries["per_drug_mean"]["object_macro_nmse"],
            "random_shrunk_nmse": summaries["random3_count_shrunk"]["object_macro_nmse"],
            "nearest_shrunk_minus_mean_nmse": primary_delta,
            "random_shrunk_minus_mean_nmse": random_delta,
            "decomposition": decomp_summary,
            "paired_object_bootstrap": bootstrap_expected,
            "paired_comparison_rows": int(sum(x.shape[0] for x in paired_rows)),
            "gate_passed": gate_pass,
            "numeric_access": {"saved_exp064_train_oof_predictions": True, "original_fold_variances": True,
                               "train_metadata_only": True, "source_h5ad_numeric_values": False,
                               "dev_inner_dev_external_pdo_outer_locked_panacea_cdsdb": False},
            "gpu_hours": 0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_dir = ROOT / cfg["output"]["root"] / cfg["output"]["attempt_id"]
    t0, c0 = time.monotonic(), time.process_time()
    try:
        receipt = check(args.config, args.run_id)
        receipt["elapsed_wall_seconds"] = time.monotonic() - t0
        receipt["process_cpu_seconds"] = time.process_time() - c0
        receipt["conservative_precharge_seconds"] = float(cfg["resources"]["package_precharge_seconds"])
        manifest_saved = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
        receipt["conservative_total_package_charge_seconds"] = (
            float(manifest_saved["conservative_charge_including_precharge_seconds"]) + receipt["process_cpu_seconds"])
        receipt["runner_process_cpu_seconds"] = float(manifest_saved["process_cpu_seconds"])
        if receipt["conservative_total_package_charge_seconds"] > min(
                float(cfg["resources"]["cpu_package_cap_seconds"]),
                float(cfg["resources"]["inherited_exp055_cpu_remaining_start_seconds"])):
            raise TimeoutError("whole package CPU charge including root readback exceeded a fixed cap")
        path = out_dir / cfg["output"]["root_readback"]
        if path.exists():
            path = out_dir / "root_independent_readback_recovery_v1.json"
        write_json(path, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {"status": "FAILED_ROOT_READBACK", "run_id": args.run_id,
                   "elapsed_wall_seconds": time.monotonic() - t0, "process_cpu_seconds": time.process_time() - c0,
                   "conservative_precharge_seconds": float(cfg["resources"]["package_precharge_seconds"]),
                   "error_type": type(exc).__name__, "error": str(exc), "gpu_hours": 0,
                   "forbidden_numeric_values_read": False}
        path = out_dir / "root_independent_readback_failure_v1.json"
        if path.exists():
            path = out_dir / "root_independent_readback_failure_v2.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(path, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
