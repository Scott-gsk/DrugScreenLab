"""Descriptive per-drug decomposition of the frozen EXP-053 TRAIN OOF archive.

This is post-hoc stratification only: it does not fit a model, change folds,
open non-TRAIN data, or select a new candidate.
"""
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

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("zero_change", "per_drug_mean", "state_residual", "shuffled_state_control")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    cfg_path = (ROOT / args.config).resolve()
    cfg: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("experiment_id") != "EXP-054" or args.run_id != cfg.get("run_id"):
        raise ValueError("config EXP/run ID mismatch")

    source_dir = ROOT / cfg["source_run_dir"]
    recovery_dir = ROOT / cfg["recovery_run_dir"]
    output_dir = ROOT / cfg["output_root"] / args.run_id
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    source_npz = source_dir / "train_object_oof_predictions.npz"
    recovery_json = recovery_dir / "result.json"
    root_receipt = recovery_dir / "root_independent_readback.json"
    provenance = recovery_dir / "recovery_provenance.json"
    for required in (source_npz, recovery_json, root_receipt, provenance):
        if not required.is_file():
            raise FileNotFoundError(required)

    result = json.loads(recovery_json.read_text(encoding="utf-8"))
    receipt = json.loads(root_receipt.read_text(encoding="utf-8"))
    source_provenance = json.loads(provenance.read_text(encoding="utf-8"))
    if (receipt.get("status") != "PASS_ROOT_INDEPENDENT_RECOVERY_READBACK"
            or receipt.get("checks", {}).get("recovered_without_refit") is not True
            or result.get("status") != "RECOVERED_FROM_SAVED_TRAIN_OOF_AFTER_RESULT_SERIALIZATION_FAILURE"):
        raise ValueError("EXP-053 recovery is not root-verified or is not recovery-only")
    if result.get("denominator", {}).get("inner_dev_numeric_values_read") is not False:
        raise ValueError("source result does not preserve the TRAIN-only boundary")

    frozen_key_table = ROOT / cfg["frozen_train_key_table"]
    gene_axis_table = ROOT / cfg["gene_axis_table"]
    keys = pd.read_csv(frozen_key_table, sep="\t", dtype=str, keep_default_na=False)
    keys = keys.loc[keys["split_role"].eq("train")].reset_index(drop=True)
    gene_map = pd.read_csv(gene_axis_table, sep="\t", dtype=str, keep_default_na=False)
    expected_genes = gene_map["h5ad_var_name"].astype(str).to_numpy(dtype=str)
    with np.load(source_npz, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}

    required_arrays = {"condition_key", "source_object_id", "full_inchikey", "fold_id", "gene_axis",
                       "truth_delta", *METHODS}
    if set(arrays) != required_arrays:
        raise ValueError(f"OOF archive arrays differ: {sorted(set(arrays) ^ required_arrays)}")
    if len(keys) != 419 or not keys["full_condition_key"].is_unique:
        raise ValueError("frozen TRAIN condition denominator is not 419 unique full keys")
    for name, column in (("condition_key", "full_condition_key"), ("source_object_id", "source_object_id"),
                         ("full_inchikey", "full_inchikey")):
        if not np.array_equal(arrays[name].astype(str), keys[column].astype(str).to_numpy(dtype=str)):
            raise ValueError(f"OOF {name} axis differs from frozen TRAIN key table")
    if not np.array_equal(arrays["gene_axis"].astype(str), expected_genes):
        raise ValueError("OOF ordered gene axis differs from frozen 978-gene mapping")
    y = arrays["truth_delta"].astype(np.float64)
    objects = arrays["source_object_id"].astype(str)
    drugs = arrays["full_inchikey"].astype(str)
    folds = arrays["fold_id"].astype(int)
    if y.shape != (419, 978) or len(set(drugs)) != 5 or not np.isfinite(y).all():
        raise ValueError("unexpected TRAIN OOF response shape, drug count, or non-finite truth")
    if any(arrays[m].shape != y.shape or not np.isfinite(arrays[m]).all() for m in METHODS):
        raise ValueError("invalid prediction shape or non-finite saved prediction")
    object_fold: dict[str, int] = {}
    for obj, fold in zip(objects, folds):
        if obj in object_fold and object_fold[obj] != int(fold):
            raise ValueError("a source object appears in more than one held-out fold")
        object_fold[obj] = int(fold)
    if len(object_fold) != 110 or set(object_fold.values()) != set(range(5)):
        raise ValueError("source-object/fold denominator differs from frozen EXP-053")

    # Match EXP-053's denominator: normalize each held-out row using fold-train
    # gene variance, then aggregate conditions within object, then objects within drug.
    per_object_rows: list[dict[str, Any]] = []
    prediction = {name: arrays[name].astype(np.float64) for name in METHODS}
    for fold in range(5):
        train_idx = np.flatnonzero(folds != fold)
        test_idx = np.flatnonzero(folds == fold)
        gene_var = np.var(y[train_idx], axis=0, ddof=0)
        use = np.isfinite(gene_var) & (gene_var > 1e-12)
        if not use.any():
            raise ValueError(f"fold {fold} has no variable TRAIN response genes")
        for key in sorted(set(drugs[test_idx])):
            key_idx = test_idx[drugs[test_idx] == key]
            for obj in sorted(set(objects[key_idx])):
                idx = key_idx[objects[key_idx] == obj]
                for method in METHODS:
                    pred = prediction[method][idx]
                    truth = y[idx]
                    err = pred - truth
                    nmse = float((np.square(err[:, use]) / gene_var[use][None, :]).mean(axis=1).mean())
                    mae = float(np.abs(err).mean())
                    corr_values = []
                    for row_y, row_p in zip(truth, pred):
                        if np.std(row_y) > 0 and np.std(row_p) > 0:
                            corr_values.append(float(np.corrcoef(row_y, row_p)[0, 1]))
                    truth_amp = float(np.abs(truth).mean())
                    pred_amp = float(np.abs(pred).mean())
                    per_object_rows.append({
                        "full_inchikey": key,
                        "source_object_id": obj,
                        "heldout_fold": fold,
                        "method": method,
                        "conditions": int(len(idx)),
                        "genes_in_fold_train_nmse": int(use.sum()),
                        "nmse": nmse,
                        "mae": mae,
                        "gene_correlation": finite_or_none(float(np.mean(corr_values)) if corr_values else float("nan")),
                        "direction_agreement": float((np.sign(truth) == np.sign(pred)).mean()),
                        "amplitude_ratio": finite_or_none(pred_amp / truth_amp if truth_amp > 0 else float("nan")),
                    })
    per_object = pd.DataFrame(per_object_rows)
    summaries: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    keys_sorted = sorted(set(drugs))
    rng_base = int(cfg["analysis"]["bootstrap_seed"])
    boot_n = int(cfg["analysis"]["bootstrap_replicates"])
    for key_index, key in enumerate(keys_sorted):
        group = per_object.loc[per_object["full_inchikey"].eq(key)]
        for method in METHODS:
            rows = group.loc[group["method"].eq(method)]
            summaries.append({
                "full_inchikey": key,
                "method": method,
                "conditions": int(rows["conditions"].sum()),
                "source_objects": int(rows["source_object_id"].nunique()),
                "object_macro_nmse": float(rows["nmse"].mean()),
                "object_macro_mae": float(rows["mae"].mean()),
                "object_macro_gene_correlation": finite_or_none(float(rows["gene_correlation"].mean())),
                "object_macro_direction_agreement": float(rows["direction_agreement"].mean()),
                "object_macro_amplitude_ratio": finite_or_none(float(rows["amplitude_ratio"].mean())),
            })
        by_obj = group.pivot(index="source_object_id", columns="method", values="nmse")
        for reference, candidate in (("per_drug_mean", "state_residual"),
                                     ("shuffled_state_control", "state_residual")):
            paired_rows = by_obj[[reference, candidate]].dropna()
            if len(paired_rows) < 2:
                raise ValueError(f"too few paired objects for {key}/{reference}/{candidate}")
            diff = (paired_rows[reference] - paired_rows[candidate]).to_numpy(dtype=np.float64)
            rng = np.random.default_rng(rng_base + key_index * 10 + (0 if reference == "per_drug_mean" else 1))
            sample_idx = rng.integers(0, len(diff), size=(boot_n, len(diff)))
            samples = diff[sample_idx].mean(axis=1)
            paired.append({
                "full_inchikey": key,
                "reference": reference,
                "candidate": candidate,
                "paired_source_objects": int(len(diff)),
                "reference_minus_candidate_nmse": float(diff.mean()),
                "bootstrap_95pct_ci_low": float(np.quantile(samples, 0.025)),
                "bootstrap_95pct_ci_high": float(np.quantile(samples, 0.975)),
                "bootstrap_replicates": boot_n,
                "seed": rng_base + key_index * 10 + (0 if reference == "per_drug_mean" else 1),
                "interpretation": "descriptive interval within this fixed TRAIN OOF drug stratum; not a drug-population test",
            })

    output_dir.mkdir(parents=True)
    per_object_path = output_dir / "per_object_drug_method_metrics.tsv"
    summary_path = output_dir / "per_drug_method_summary.tsv"
    paired_path = output_dir / "per_drug_paired_nmse_differences.tsv"
    per_object.to_csv(per_object_path, sep="\t", index=False, na_rep="")
    pd.DataFrame(summaries).to_csv(summary_path, sep="\t", index=False, na_rep="")
    pd.DataFrame(paired).to_csv(paired_path, sep="\t", index=False, na_rep="")
    elapsed = time.monotonic() - started
    input_paths = (source_npz, recovery_json, root_receipt, provenance, frozen_key_table, gene_axis_table, cfg_path)
    input_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in input_paths}
    key_results = []
    summary_df = pd.DataFrame(summaries)
    paired_df = pd.DataFrame(paired)
    for key in keys_sorted:
        sub = summary_df.loc[summary_df.full_inchikey.eq(key)].set_index("method")
        pair = paired_df.loc[(paired_df.full_inchikey.eq(key)) &
                             (paired_df.reference.eq("per_drug_mean"))].iloc[0]
        key_results.append({
            "full_inchikey": key,
            "source_objects": int(sub.loc["per_drug_mean", "source_objects"]),
            "conditions": int(sub.loc["per_drug_mean", "conditions"]),
            "per_drug_mean_nmse": float(sub.loc["per_drug_mean", "object_macro_nmse"]),
            "state_residual_nmse": float(sub.loc["state_residual", "object_macro_nmse"]),
            "shuffled_state_nmse": float(sub.loc["shuffled_state_control", "object_macro_nmse"]),
            "state_minus_per_drug_mean_nmse": float(
                sub.loc["state_residual", "object_macro_nmse"] - sub.loc["per_drug_mean", "object_macro_nmse"]),
            "per_drug_mean_minus_state_nmse_ci": [float(pair.bootstrap_95pct_ci_low),
                                                   float(pair.bootstrap_95pct_ci_high)],
        })
    improved = sum(x["state_minus_per_drug_mean_nmse"] < 0 for x in key_results)
    result = {
        "status": "COMPLETED_POSTHOC_TRAIN_OOF_PER_DRUG_DECOMPOSITION",
        "experiment_id": "EXP-054",
        "run_id": args.run_id,
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "claim_boundary": "descriptive stratification of the fixed EXP-053 TRAIN-only source-object OOF predictions; no new fit and no PDO/outer/locked inference",
        "method": "Within each fixed InChIKey and held-out source object, aggregate OOF condition errors using the original fold-training gene variance; then average objects equally. Paired intervals resample source objects within drug.",
        "denominator": {"train_conditions": 419, "source_objects": 110, "full_inchikeys": 5,
                        "genes": 978, "drug_strata": 5, "folds": 5,
                        "inner_dev_values_read": False, "pdo_outer_locked_values_read": False},
        "state_residual_lower_nmse_than_per_drug_mean_drug_strata": improved,
        "drug_strata": key_results,
        "interpretation": "Exploratory heterogeneity only. Per-drug intervals do not establish generalization to new drugs and are not multiplicity-adjusted tests. The overall fixed TRAIN object-OOF result remains primary.",
        "resources": {"cpu_wall_seconds": elapsed, "gpu_hours": 0.0},
        "input_sha256": input_hashes,
        "outputs": {},
    }
    for p in (per_object_path, summary_path, paired_path):
        result["outputs"][p.name] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    manifest = output_dir / "sha256_manifest.tsv"
    pd.DataFrame([{"file": p.name, "sha256": sha256(p), "bytes": p.stat().st_size}
                  for p in (per_object_path, summary_path, paired_path, result_path)]).to_csv(
                      manifest, sep="\t", index=False)
    print(json.dumps({"status": result["status"], "output_dir": str(output_dir),
                      "cpu_wall_seconds": elapsed, "gpu_hours": 0.0,
                      "state_helped_strata": improved, "drug_strata": len(keys_sorted)},
                     ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
