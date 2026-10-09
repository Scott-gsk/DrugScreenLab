#!/usr/bin/env python3
"""EXP-065: one fixed count-shrinkage transform of sealed EXP-064 TRAIN OOF output."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
CONTRASTS = {
    "nearest_shrunk_vs_per_drug_mean": ("per_drug_mean", "nearest3_count_shrunk"),
    "nearest_shrunk_vs_random_shrunk": ("random3_count_shrunk", "nearest3_count_shrunk"),
    "nearest_original_vs_per_drug_mean": ("per_drug_mean", "nearest3_state"),
    "random_original_vs_per_drug_mean": ("per_drug_mean", "random3_same_drug"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def arrays_roundtrip_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return False
    if left.dtype.kind in "fc" or right.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def read_config(path: Path, run_id: str) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg.get("experiment_id") != "EXP-065" or cfg.get("run_id") != run_id:
        raise ValueError("config run identity mismatch")
    if float(cfg["resources"]["gpu_hours_cap"]) != 0:
        raise ValueError("EXP-065 is CPU-only")
    source = ROOT / cfg["source"]["attempt_dir"]
    if sha256(source / "root_independent_readback.json") != cfg["source"]["source_receipt_sha256"]:
        raise ValueError("EXP-064 root readback receipt SHA mismatch")
    receipt = json.loads((source / "root_independent_readback.json").read_text(encoding="utf-8"))
    if "PASS" not in str(receipt.get("status", "")):
        raise ValueError("EXP-064 source root receipt is not PASS")
    for name, field in (("keyed_oof_predictions.npz", "source_prediction_sha256"),
                        ("fold_assignment.tsv", "fold_assignment_sha256"),
                        ("fold_preprocessing.npz", "fold_preprocessing_sha256")):
        if sha256(source / name) != cfg["source"][field]:
            raise ValueError(f"EXP-064 immutable source artifact SHA mismatch: {name}")
    return cfg


def load_source(cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source = ROOT / cfg["source"]["attempt_dir"]
    with np.load(source / "keyed_oof_predictions.npz", allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    metadata = pd.read_csv(source / "fold_assignment.tsv", sep="\t", dtype=str, keep_default_na=False)
    neighbors = pd.read_csv(source / "neighbor_selection.tsv", sep="\t", dtype=str, keep_default_na=False)
    for col in ("full_condition_key", "source_object_id", "full_inchikey"):
        metadata[col] = metadata[col].astype(str).str.strip()
    metadata["fold_id"] = pd.to_numeric(metadata["fold_id"], errors="raise").astype(np.int16)
    for col in ("full_condition_key", "source_object_id", "full_inchikey"):
        neighbors[col if col != "source_object_id" else "test_source_object_id"] = neighbors[
            col if col != "source_object_id" else "test_source_object_id"].astype(str).str.strip()
    neighbors["fold_id"] = pd.to_numeric(neighbors["fold_id"], errors="raise").astype(np.int16)
    neighbors["candidate_train_objects"] = pd.to_numeric(neighbors["candidate_train_objects"], errors="raise").astype(np.int32)
    check_metadata(cfg, arrays, metadata, neighbors)
    source_metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    return arrays, metadata, neighbors, source_metrics


def check_metadata(cfg: dict[str, Any], a: dict[str, np.ndarray], meta: pd.DataFrame,
                   neighbors: pd.DataFrame) -> None:
    required = {"full_condition_key", "source_object_id", "full_inchikey", "fold_id"}
    if not required.issubset(meta.columns) or len(meta) != int(cfg["source"]["train_domain"][0]):
        raise ValueError("frozen TRAIN fold metadata schema/count mismatch")
    if meta.full_condition_key.duplicated().any():
        raise ValueError("duplicate condition in frozen TRAIN fold metadata")
    order = pd.Index(meta.full_condition_key)
    pred_keys = pd.Index(a["full_condition_key"].astype(str))
    if len(pred_keys) != len(order) or set(pred_keys) != set(order):
        raise ValueError("condition-key denominator differs from sealed EXP-064 OOF")
    aligned = meta.set_index("full_condition_key").loc[pred_keys].reset_index()
    for field, array_name in (("source_object_id", "source_object_id"), ("full_inchikey", "full_inchikey"),
                              ("fold_id", "fold_id")):
        if not np.array_equal(aligned[field].astype(str).to_numpy(), a[array_name].astype(str)):
            raise ValueError(f"TRAIN metadata vs OOF key mismatch: {field}")
    if int(aligned.source_object_id.nunique()) != int(cfg["source"]["train_domain"][1]):
        raise ValueError("TRAIN object denominator mismatch")
    if int(aligned.full_inchikey.nunique()) != int(cfg["source"]["train_domain"][2]):
        raise ValueError("TRAIN full-InChIKey denominator mismatch")
    if a["y_true"].shape != (len(meta), int(cfg["source"]["train_domain"][3])):
        raise ValueError("TRAIN response axis mismatch")
    if not np.all(a["common_denominator"] == 1):
        raise ValueError("EXP-064 common denominator is not complete")
    folds = sorted(int(x) for x in np.unique(a["fold_id"]))
    if folds != [0, 1, 2, 3, 4] or a["fold_gene_variance"].shape != (5, int(cfg["source"]["train_domain"][3])):
        raise ValueError("frozen fold/variance schema mismatch")
    if neighbors.full_condition_key.duplicated().any() or set(neighbors.full_condition_key) != set(pred_keys):
        raise ValueError("neighbor manifest condition keys mismatch")
    neighbors = neighbors.set_index("full_condition_key").loc[pred_keys].reset_index()
    for field, array_name in (("test_source_object_id", "source_object_id"), ("full_inchikey", "full_inchikey"),
                              ("fold_id", "fold_id")):
        if not np.array_equal(neighbors[field].astype(str).to_numpy(), a[array_name].astype(str)):
            raise ValueError(f"neighbor manifest vs OOF key mismatch: {field}")
    if int(aligned["source_object_id"].nunique()) != 93:
        raise ValueError("unexpected TRAIN object count")
    if not np.isfinite(a["fold_gene_variance"]).all() or np.any(a["fold_gene_variance"] <= 1e-12):
        raise ValueError("invalid inherited fold TRAIN variance")
    for arm in BASE_ARMS:
        pred = a[f"pred_{arm}"]
        if pred.shape != a["y_true"].shape or not np.isfinite(pred).all():
            raise ValueError(f"nonfinite or mis-shaped sealed OOF arm: {arm}")
    if int(a["common_denominator"].sum()) != len(meta):
        raise ValueError("common denominator changed")


def count_and_lambda(cfg: dict[str, Any], a: dict[str, np.ndarray], meta: pd.DataFrame,
                     neighbors: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    fold = a["fold_id"].astype(np.int16)
    drug = a["full_inchikey"].astype(str)
    n_by_fold_drug: dict[tuple[int, str], int] = {}
    for f in range(5):
        tr = meta.loc[meta.fold_id.ne(f)]
        counts = tr.groupby("full_inchikey", sort=True).source_object_id.nunique()
        for key, n in counts.items():
            n_by_fold_drug[(f, str(key))] = int(n)
    n_vec = np.asarray([n_by_fold_drug[(int(f), str(k))] for f, k in zip(fold, drug, strict=True)], dtype=np.int16)
    lambda_vec = float(cfg["shrinkage"]["neighbor_count"]) / (n_vec.astype(np.float64) + float(cfg["shrinkage"]["neighbor_count"]))
    neighbor_aligned = neighbors.set_index("full_condition_key").loc[a["full_condition_key"].astype(str)]
    manifest_n = neighbor_aligned["candidate_train_objects"].to_numpy(dtype=np.int32)
    if not np.array_equal(n_vec.astype(np.int32), manifest_n):
        raise ValueError("count from frozen TRAIN fold metadata differs from EXP-064 neighbor candidate count")
    if np.any(n_vec < int(cfg["shrinkage"]["neighbor_count"])):
        raise ValueError("frozen fold/drug TRAIN count insufficient for a three-object comparison")
    rows = []
    for (f, k), group in meta.assign(fold_id=fold, full_inchikey=drug).groupby(["fold_id", "full_inchikey"], sort=True):
        idx = group.index.to_numpy(dtype=np.int64)
        n = n_by_fold_drug[(int(f), str(k))]
        lam = float(cfg["shrinkage"]["neighbor_count"]) / (n + float(cfg["shrinkage"]["neighbor_count"]))
        rows.append({"fold_id": int(f), "full_inchikey": str(k), "heldout_conditions": int(len(idx)),
                     "heldout_objects": int(group.source_object_id.nunique()), "N_train_same_drug_objects": n,
                     "lambda": lam, "train_object_ids_sha256": hashlib.sha256("\n".join(sorted(
                         meta.loc[meta.fold_id.ne(int(f)) & meta.full_inchikey.eq(str(k)), "source_object_id"].unique()
                     )).encode("utf-8")).hexdigest()})
    return n_vec, lambda_vec, pd.DataFrame(rows)


def make_arms(a: dict[str, np.ndarray], n: np.ndarray, lam: np.ndarray) -> dict[str, np.ndarray]:
    mean = a["pred_per_drug_mean"].astype(np.float64)
    arms = {name: a[f"pred_{name}"].copy() for name in BASE_ARMS}
    arms["nearest3_count_shrunk"] = (mean + lam[:, None] * (a["pred_nearest3_state"].astype(np.float64) - mean)).astype(np.float64)
    arms["random3_count_shrunk"] = (mean + lam[:, None] * (a["pred_random3_same_drug"].astype(np.float64) - mean)).astype(np.float64)
    if not np.all(np.isfinite(arms["nearest3_count_shrunk"])) or not np.all(np.isfinite(arms["random3_count_shrunk"])):
        raise FloatingPointError("nonfinite fixed-shrinkage prediction")
    return arms


def score_all(a: dict[str, np.ndarray], arms: dict[str, np.ndarray],
              cfg: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    keys = pd.DataFrame({"full_condition_key": a["full_condition_key"].astype(str),
                         "source_object_id": a["source_object_id"].astype(str),
                         "full_inchikey": a["full_inchikey"].astype(str),
                         "fold_id": a["fold_id"].astype(np.int16)})
    truth = a["y_true"].astype(np.float64)
    variance = a["fold_gene_variance"].astype(np.float64)
    common = a["common_denominator"].astype(bool)
    summaries: dict[str, Any] = {}
    object_tables = []
    for method in ARMS:
        summary, obj = exp064_metrics(keys, truth, arms[method], variance, common)
        summaries[method] = summary
        obj = obj.rename(columns={c: f"{method}_{c}" for c in obj.columns if c != "source_object_id"})
        object_tables.append(obj)
    obj_table = object_tables[0]
    for frame in object_tables[1:]:
        obj_table = obj_table.merge(frame, on="source_object_id", how="outer", validate="one_to_one")
    obj_to_drug = keys.groupby("source_object_id", sort=True).full_inchikey.agg(lambda x: ";".join(sorted(set(x)))
    ).rename("full_inchikeys")
    obj_table = obj_table.merge(obj_to_drug, left_on="source_object_id", right_index=True, validate="one_to_one")
    per_drug_rows = []
    for drug in sorted(keys.full_inchikey.unique()):
        mask = keys.full_inchikey.eq(drug).to_numpy()
        subset = keys.loc[mask].reset_index(drop=True)
        for method in ARMS:
            sm, _ = exp064_metrics(subset, truth[mask], arms[method][mask], variance, common[mask])
            per_drug_rows.append({"full_inchikey": drug, "method": method, **sm})
    per_drug = pd.DataFrame(per_drug_rows)

    pairs = []
    bootstrap: dict[str, Any] = {}
    rng = np.random.default_rng(int(cfg["metrics"]["paired_object_bootstrap_seed"]))
    reps = int(cfg["metrics"]["paired_object_bootstrap_replicates"])
    for contrast, (base, candidate) in CONTRASTS.items():
        lhs = obj_table[["source_object_id", f"{base}_nmse"]].rename(columns={f"{base}_nmse": "baseline_nmse"})
        rhs = obj_table[["source_object_id", f"{candidate}_nmse"]].rename(columns={f"{candidate}_nmse": "candidate_nmse"})
        pair = lhs.merge(rhs, on="source_object_id", validate="one_to_one")
        pair["improvement_baseline_minus_candidate"] = pair.baseline_nmse - pair.candidate_nmse
        pair["contrast"] = contrast
        pairs.append(pair)
        d = pair.improvement_baseline_minus_candidate.to_numpy(dtype=np.float64)
        samples = rng.integers(0, len(d), size=(reps, len(d)))
        means = d[samples].mean(axis=1)
        bootstrap[contrast] = {
            "baseline": base, "candidate": candidate, "object_denominator": int(len(d)),
            "mean_paired_improvement_baseline_minus_candidate": float(np.mean(d)),
            "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            # d is baseline minus candidate, so positive values favor the candidate.
            "objects_candidate_better": int(np.sum(d > 0)),
            "objects_candidate_worse": int(np.sum(d < 0)),
            "bootstrap_replicates": reps,
            "bootstrap_seed": int(cfg["metrics"]["paired_object_bootstrap_seed"]),
        }
    paired = pd.concat(pairs, ignore_index=True)
    decomposition, decomp_rows = normalized_error_decomposition(keys, truth, arms, variance, n_vec=a["N_train_same_drug_objects"],
                                                                 lambda_vec=a["lambda"])
    primary = summaries["nearest3_count_shrunk"]["object_macro_nmse"]
    mean_nmse = summaries["per_drug_mean"]["object_macro_nmse"]
    random_nmse = summaries["random3_count_shrunk"]["object_macro_nmse"]
    gain = (mean_nmse - primary) / mean_nmse if mean_nmse is not None and mean_nmse != 0 else None
    gate = bool(gain is not None and gain >= 0.05 and primary < random_nmse)
    bundle = {
        "experiment_id": "EXP-065", "protocol_id": cfg["protocol_id"], "run_id": cfg["run_id"],
        "status": "COMPLETED_ROOT_READBACK_PENDING",
        "scope": "post-hoc fixed-source TRAIN OOF mechanism check; not independent validation",
        "denominators": {"conditions": int(len(keys)), "objects": int(keys.source_object_id.nunique()),
                         "full_inchikeys": int(keys.full_inchikey.nunique()), "genes": int(truth.shape[1]),
                         "common_conditions": int(common.sum()), "common_objects": int(keys.loc[common, "source_object_id"].nunique()),
                         "missing_conditions": int((~common).sum())},
        "metrics_by_method": summaries, "paired_object_bootstrap": bootstrap,
        "fixed_gate": {"nearest_shrunk_relative_nmse_improvement_vs_mean": gain,
                       "nearest_shrunk_nmse": primary, "per_drug_mean_nmse": mean_nmse,
                       "random_shrunk_nmse": random_nmse, "relative_gain_threshold": 0.05,
                       "lower_error_than_random_shrunk": bool(primary < random_nmse),
                       "gate_passed": gate,
                       "decision": "retain_as_future_confirmation_candidate_only" if gate else "close_fixed_shrinkage_candidate"},
        "error_decomposition": decomposition,
        "information_access": {"EXP064_saved_TRAIN_OOF_numeric_predictions": True,
                                "EXP064_fold_train_gene_variances": True,
                                "TRAIN_fold_object_and_drug_metadata_only": True,
                                "source_H5AD_numeric_values_read": False,
                                "DEV_inner_dev_external_PDO_outer_locked_PANACEA_CDSDB_numeric_values_read": False},
    }
    return bundle, obj_table, per_drug, paired, {"decomposition": decomposition, "per_fold_drug": decomp_rows}


def normalized_error_decomposition(keys: pd.DataFrame, y: np.ndarray, arms: dict[str, np.ndarray],
                                   fold_var: np.ndarray, n_vec: np.ndarray, lambda_vec: np.ndarray
                                   ) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    fold_ids = keys.fold_id.to_numpy(dtype=int)
    mu = arms["per_drug_mean"].astype(np.float64)
    # Compute row-level contributions first. This preserves the same condition
    # weighting within each held-out object as the authoritative EXP-064 metric,
    # even when an object has multiple drug conditions and different lambdas.
    d_by_arm = {
        # Sign follows the preregistered minus-covariance identity:
        # d = drug_mean - three-object prediction.
        "nearest": mu - arms["nearest3_state"].astype(np.float64),
        "random": mu - arms["random3_same_drug"].astype(np.float64),
    }
    err_mu = mu - y.astype(np.float64)
    row_values: dict[str, dict[str, np.ndarray]] = {}
    for label, d in d_by_arm.items():
        var_rows = fold_var[fold_ids]
        usable = np.isfinite(var_rows) & (var_rows > 1e-12)
        lam_rows = lambda_vec[:, None]
        residual = np.divide(lam_rows ** 2 * d ** 2, var_rows,
                             out=np.zeros_like(d, dtype=np.float64), where=usable)
        covariance = np.divide(2.0 * lam_rows * d * err_mu, var_rows,
                               out=np.zeros_like(d, dtype=np.float64), where=usable)
        delta = residual - covariance
        row_values[label] = {"residual": np.mean(residual, axis=1),
                             "covariance": np.mean(covariance, axis=1),
                             "delta": np.mean(delta, axis=1)}
    for obj, group in keys.groupby("source_object_id", sort=True):
        idx = group.index.to_numpy(dtype=np.int64)
        if len(np.unique(fold_ids[idx])) != 1:
            raise ValueError("source object split across folds")
        row = {"fold_id": int(fold_ids[idx[0]]), "source_object_id": str(obj),
               "heldout_conditions": int(len(idx)),
               "full_inchikeys": ";".join(sorted(set(keys.full_inchikey.iloc[idx].astype(str))))}
        for label in ("nearest", "random"):
            for component, values in row_values[label].items():
                row[f"{label}_{component}_component" if component != "delta" else f"{label}_formula_delta_nmse"] = float(np.mean(values[idx]))
        rows.append(row)
    frame = pd.DataFrame(rows)
    global_summary = {
        "nearest": {
            "nearest_residual_variance_component_object_macro": float(frame.nearest_residual_component.mean()),
            "nearest_twice_covariance_component_object_macro": float(frame.nearest_covariance_component.mean()),
            "nearest_formula_delta_nmse_object_macro": float(frame.nearest_formula_delta_nmse.mean()),
        },
        "random": {
            "random_residual_variance_component_object_macro": float(frame.random_residual_component.mean()),
            "random_twice_covariance_component_object_macro": float(frame.random_covariance_component.mean()),
            "random_formula_delta_nmse_object_macro": float(frame.random_formula_delta_nmse.mean()),
        },
        "weighting": "first mean conditions within object, then equal weight each heldout object; preserve per-fold TRAIN gene variance",
        "object_denominator": int(frame.source_object_id.nunique()),
    }
    return global_summary, frame


def keyed_for_rows(a: dict[str, np.ndarray], arms: dict[str, np.ndarray], n: np.ndarray,
                   lam: np.ndarray, idx: np.ndarray) -> dict[str, np.ndarray]:
    out = {k: a[k][idx] for k in ("full_condition_key", "source_object_id", "full_inchikey", "source_row", "fold_id",
                                    "y_true", "common_denominator")}
    out["N_train_same_drug_objects"] = n[idx]
    out["lambda"] = lam[idx].astype(np.float64)
    out["fold_gene_variance"] = a["fold_gene_variance"].copy()
    for arm in ARMS:
        out[f"pred_{arm}"] = arms[arm][idx]
    return out


def run(cfg_path: Path, run_id: str) -> int:
    wall0, cpu0 = time.monotonic(), time.process_time()
    attempt: Path | None = None
    cfg: dict[str, Any] | None = None
    try:
        cfg = read_config(cfg_path, run_id)
        out_root = ROOT / cfg["output"]["root"]
        out_root.mkdir(parents=True, exist_ok=True)
        attempt = out_root / cfg["output"]["attempt_id"]
        attempt.mkdir(parents=True, exist_ok=False)
        source_load_start = time.monotonic()
        a, meta, neighbors, source_metrics = load_source(cfg)
        n_vec, lambda_vec, count_table = count_and_lambda(cfg, a, meta, neighbors)
        input_load_seconds = time.monotonic() - source_load_start
        arms = make_arms(a, n_vec, lambda_vec)
        fold_sizes = meta.groupby("fold_id").source_object_id.nunique().to_dict()
        fold_condition_counts = meta.groupby("fold_id").size().to_dict()
        fold0 = np.flatnonzero(a["fold_id"] == 0)
        fold0_start = time.monotonic()
        fold0_payload = keyed_for_rows(a, arms, n_vec, lambda_vec, fold0)
        fold0_path = attempt / cfg["output"]["first_fold_predictions"]
        np.savez_compressed(fold0_path, **fold0_payload)
        with np.load(fold0_path, allow_pickle=False) as saved:
            if not np.array_equal(saved["full_condition_key"], fold0_payload["full_condition_key"]):
                raise IOError("fold0 key roundtrip failed")
            for key in fold0_payload:
                if not arrays_roundtrip_equal(saved[key], fold0_payload[key]):
                    raise IOError(f"fold0 artifact roundtrip failed: {key}")
        fold0_elapsed = time.monotonic() - fold0_start
        if not np.all(fold0_payload["N_train_same_drug_objects"] >= 3):
            raise ValueError("fold0 includes fewer than three same-drug TRAIN objects")
        precharge = float(cfg["resources"]["package_precharge_seconds"])
        largest_fold = max(int(v) for v in fold_sizes.values())
        fold0_conditions = max(1, int(fold_condition_counts[0]))
        largest_condition_fold = max(int(v) for v in fold_condition_counts.values())
        projected_other_fold = max(2.0, fold0_elapsed * 1.5 * largest_condition_fold / fold0_conditions)
        remaining_fold_cost = 4.0 * projected_other_fold
        unmargined = precharge + input_load_seconds + fold0_elapsed + remaining_fold_cost + float(
            cfg["resources"]["root_readback_reserve_seconds"]) + float(cfg["resources"]["aggregation_closeout_reserve_seconds"])
        with_margin = unmargined * (1 + float(cfg["resources"]["full_schedule_margin_fraction"]))
        inherited = float(cfg["resources"]["inherited_exp055_cpu_remaining_start_seconds"])
        cap = float(cfg["resources"]["cpu_package_cap_seconds"])
        gate = {"status": "PASS_CONTINUE_FULL_FOLDS" if with_margin <= min(cap, inherited) else "CLOSED_COST_GATE_AFTER_FOLD0",
                "fold0_complete_formula_and_keyed_roundtrip_seconds": fold0_elapsed,
                "input_load_and_metadata_validation_seconds": input_load_seconds,
                "fold_object_counts": {str(k): int(v) for k, v in fold_sizes.items()},
                "fold_condition_counts": {str(k): int(v) for k, v in fold_condition_counts.items()},
                "projected_remaining_fold_seconds_each": projected_other_fold,
                "projected_remaining_folds_seconds": remaining_fold_cost,
                "precharge_seconds": precharge,
                "root_readback_reserve_seconds": cfg["resources"]["root_readback_reserve_seconds"],
                "aggregation_closeout_reserve_seconds": cfg["resources"]["aggregation_closeout_reserve_seconds"],
                "projected_unmargined_seconds": unmargined, "projected_with_20pct_margin_seconds": with_margin,
                "package_cpu_cap_seconds": cap, "inherited_exp055_cpu_remaining_seconds": inherited,
                "gate_limit_seconds": min(cap, inherited), "gpu_hours": 0,
                "formula": "lambda=3/(N+3); N is distinct same-full-InChIKey TRAIN objects outside heldout fold"}
        write_json(attempt / cfg["output"]["cost_gate"], gate)
        write_json(attempt / "run_manifest.json", {"experiment_id": "EXP-065", "run_id": run_id,
                    "attempt_id": cfg["output"]["attempt_id"], "status": gate["status"],
                    "source_root_readback_sha256": cfg["source"]["source_receipt_sha256"],
                    "source_prediction_sha256": cfg["source"]["source_prediction_sha256"],
                    "train_fold_metadata_sha256": cfg["source"]["fold_assignment_sha256"],
                    "input_load_seconds": input_load_seconds, "fold0_complete_seconds": fold0_elapsed,
                    "projected_total_with_margin_seconds": with_margin, "gpu_hours": 0,
                    "forbidden_numeric_values_read": False})
        if gate["status"] != "PASS_CONTINUE_FULL_FOLDS":
            return 2

        # The complete keyed OOF predictions are a frozen input; only the fixed transform is applied.
        all_idx = np.arange(len(a["fold_id"]))
        payload = keyed_for_rows(a, arms, n_vec, lambda_vec, all_idx)
        pred_path = attempt / cfg["output"]["keyed_predictions"]
        np.savez_compressed(pred_path, **payload)
        with np.load(pred_path, allow_pickle=False) as saved:
            for key in payload:
                if not arrays_roundtrip_equal(saved[key], payload[key]):
                    raise IOError(f"full keyed artifact roundtrip failed: {key}")
        for fold in range(5):
            fold_idx = np.flatnonzero(a["fold_id"] == fold)
            fold_payload = keyed_for_rows(a, arms, n_vec, lambda_vec, fold_idx)
            with np.load(pred_path, allow_pickle=False) as saved:
                if not np.array_equal(saved["full_condition_key"][fold_idx], fold_payload["full_condition_key"]):
                    raise IOError(f"full keyed fold order mismatch at fold {fold}")
        keys = pd.DataFrame({"full_condition_key": a["full_condition_key"].astype(str),
                             "source_object_id": a["source_object_id"].astype(str),
                             "full_inchikey": a["full_inchikey"].astype(str), "fold_id": a["fold_id"].astype(np.int16)})
        bundle, obj_table, per_drug, paired, decomp = score_all({**a, "N_train_same_drug_objects": n_vec, "lambda": lambda_vec}, arms, cfg)
        bundle["status"] = "COMPLETED_ROOT_READBACK_PENDING"
        bundle["cost_gate"] = gate
        bundle["source_metrics_sha256"] = sha256(ROOT / cfg["source"]["attempt_dir"] / "metrics.json")
        bundle["source_keyed_prediction_sha256"] = cfg["source"]["source_prediction_sha256"]
        bundle["source_fold_assignment_sha256"] = cfg["source"]["fold_assignment_sha256"]
        bundle["source_fold_preprocessing_sha256"] = cfg["source"]["fold_preprocessing_sha256"]
        bundle["count_by_fold_drug_rows"] = len(count_table)
        bundle["source_oof_metrics_summary"] = {m: source_metrics["methods"][m] for m in BASE_ARMS}
        write_json(attempt / cfg["output"]["metrics"], bundle)
        obj_table.to_csv(attempt / cfg["output"]["object_metrics"], sep="\t", index=False)
        per_drug.to_csv(attempt / cfg["output"]["per_drug_metrics"], sep="\t", index=False)
        paired.to_csv(attempt / cfg["output"]["paired_object_differences"], sep="\t", index=False)
        count_table.to_csv(attempt / cfg["output"]["count_table"], sep="\t", index=False)
        decomp["per_fold_drug"].to_csv(attempt / cfg["output"]["decomposition_table"], sep="\t", index=False)
        write_json(attempt / cfg["output"]["normalized_error_decomposition"], decomp["decomposition"])
        used = float(cfg["resources"]["package_precharge_seconds"]) + time.process_time() - cpu0
        if used > min(cap, inherited):
            raise TimeoutError(f"actual conservative CPU charge exceeded package/inherited cap: {used:.3f}s")
        write_json(attempt / "run_manifest.json", {"experiment_id": "EXP-065", "protocol_id": cfg["protocol_id"],
                    "run_id": run_id, "attempt_id": cfg["output"]["attempt_id"],
                    "status": "COMPLETED_ROOT_READBACK_PENDING", "full_folds_completed": True,
                    "elapsed_wall_seconds": time.monotonic() - wall0, "process_cpu_seconds": time.process_time() - cpu0,
                    "conservative_charge_including_precharge_seconds": used, "gpu_hours": 0,
                    "source_root_readback_sha256": cfg["source"]["source_receipt_sha256"],
                    "source_prediction_sha256": cfg["source"]["source_prediction_sha256"],
                    "source_fold_assignment_sha256": cfg["source"]["fold_assignment_sha256"],
                    "source_fold_preprocessing_sha256": cfg["source"]["fold_preprocessing_sha256"],
                    "predictions_sha256": sha256(pred_path), "metrics_sha256": sha256(attempt / cfg["output"]["metrics"]),
                    "count_table_sha256": sha256(attempt / cfg["output"]["count_table"]),
                    "forbidden_numeric_values_read": False})
        return 0
    except Exception as exc:
        if attempt is not None:
            receipt = {"status": "FAILED_CLOSED", "run_id": run_id,
                       "elapsed_wall_seconds": time.monotonic() - wall0,
                       "process_cpu_seconds": time.process_time() - cpu0,
                       "conservative_precharge_seconds": float(cfg["resources"]["package_precharge_seconds"]) if cfg else None,
                       "error_type": type(exc).__name__, "error": str(exc),
                       "forbidden_numeric_values_read": False, "gpu_hours": 0}
            write_json(attempt / "attempt_close_receipt.json", receipt)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    return run(args.config, args.run_id)


if __name__ == "__main__":
    raise SystemExit(main())
