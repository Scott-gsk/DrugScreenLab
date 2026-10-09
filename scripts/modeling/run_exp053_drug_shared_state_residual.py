"""Train-source object-OOF diagnostic for a shared drug effect plus state residual."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/modeling"))
import exp051_contract_fit_smoke as fit  # noqa: E402
import run_exp052_transigen_lr_stability as source_reader  # noqa: E402

METHODS = ("zero_change", "per_drug_mean", "state_residual", "shuffled_state_control")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                              allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    return fit.sha256_file(path)


def finite_mean_or_none(values: list[float] | np.ndarray) -> float | None:
    """Summarize defined metrics while preserving all-undefined values as null."""
    arr = np.asarray(values, dtype=np.float64)
    defined = np.isfinite(arr)
    return float(arr[defined].mean()) if defined.any() else None


def source_inputs(source_run_id: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    fit.configure_source_run(source_run_id)
    src = fit.SOURCE_RUN_DIR
    required = ("metadata_freeze_receipt.json", "gene_axis_receipt.json", "value_contract_receipt.json",
                "root_values_readback_receipt.json", "response_blind_conditions.tsv", "gene_axis_mapping.tsv",
                "frozen_train_inner_values.npz")
    for name in required:
        if not (src / name).is_file():
            raise FileNotFoundError(f"frozen EXP-051 input is missing: {name}")
    axis = json.loads((src / "gene_axis_receipt.json").read_text(encoding="utf-8"))
    root_receipt = json.loads((src / "root_values_readback_receipt.json").read_text(encoding="utf-8"))
    value_receipt = json.loads((src / "value_contract_receipt.json").read_text(encoding="utf-8"))
    if axis.get("status") != "PASS_EXACT_978_AXIS" or root_receipt.get("checks", {}).get("result") != "PASS":
        raise ValueError("frozen source axis or root value readback is not PASS")
    archive = src / "frozen_train_inner_values.npz"
    expected_archive_sha = value_receipt.get("files_sha256", {}).get("frozen_train_inner_values_npz")
    if (not expected_archive_sha or sha256(archive) != expected_archive_sha
            or root_receipt.get("input_sha256") != expected_archive_sha):
        raise ValueError("frozen source response archive bytes are not bound to both value and root receipts")
    axis_map = pd.read_csv(src / "gene_axis_mapping.tsv", sep="\t", dtype=str, keep_default_na=False)
    if (len(axis_map) != 978 or axis_map.axis_index.astype(int).tolist() != list(range(978))
            or axis_map.h5ad_var_name.nunique() != 978):
        raise ValueError("frozen gene mapping is not a unique ordered 978-gene axis")
    axis_sha = hashlib.sha256(("\n".join(axis_map.h5ad_var_annotation.astype(str)) + "\n").encode()).hexdigest()
    if axis_sha != axis.get("mapping_sha256"):
        raise ValueError("frozen gene annotation sequence differs from axis receipt")
    key_path = src / "response_blind_conditions.tsv"
    keys_all = pd.read_csv(key_path, sep="\t", dtype=str, keep_default_na=False)
    required_cols = {"split_role", "source_object_id", "source_row", "obs_name", "pert_id",
                     "full_inchikey", "full_condition_key"}
    if not required_cols.issubset(keys_all.columns):
        raise ValueError(f"frozen key table missing required columns: {sorted(required_cols - set(keys_all.columns))}")
    keys = keys_all.loc[keys_all.split_role.eq("train")].reset_index(drop=True)
    if (len(keys) != 419 or keys.source_object_id.nunique() != 110
            or keys.full_inchikey.nunique() != 5 or not keys.full_condition_key.is_unique):
        raise ValueError("frozen TRAIN denominators or complete condition keys changed")
    genes = axis_map.h5ad_var_name.astype(str).to_numpy(dtype=str)
    # This helper reads only the selected TRAIN H5AD rows; it never opens NPZ numeric arrays.
    values = source_reader.read_role_expression(keys, genes)
    state_rows = values["X_ctl"].astype(np.float64)
    delta = values["delta"].astype(np.float64)
    if state_rows.shape != (419, 978) or delta.shape != (419, 978):
        raise ValueError("TRAIN-only expression matrices do not match the frozen 419x978 contract")
    if float(values["max_abs_decomposition_error"]) > 1e-6:
        raise ValueError("treated-control decomposition exceeds the registered tolerance")
    input_receipt = {
        "status": "PASS_TRAIN_ONLY_H5AD_ROW_READ",
        "source_run_id": source_run_id,
        "source_key_table_sha256": sha256(key_path),
        "source_value_archive_sha256": expected_archive_sha,
        "source_axis_receipt_sha256": sha256(src / "gene_axis_receipt.json"),
        "train_conditions": 419, "train_objects": 110, "full_inchikeys": 5, "genes": 978,
        "train_numeric_role_read": True, "inner_dev_numeric_response_read": False,
        "outer_locked_pdo_opened": False,
        "max_abs_treated_control_delta_error": float(values["max_abs_decomposition_error"]),
    }
    return keys, genes, state_rows, delta, input_receipt


def split_objects(keys: pd.DataFrame, folds: int, seed: int) -> tuple[dict[str, int], list[list[str]]]:
    objects = np.asarray(sorted(keys.source_object_id.astype(str).unique().tolist()), dtype=str)
    rng = np.random.default_rng(seed)
    ordered = objects[rng.permutation(len(objects))]
    fold_objects = [ordered[i::folds].tolist() for i in range(folds)]
    mapping = {obj: fold for fold, members in enumerate(fold_objects) for obj in members}
    if len(mapping) != len(objects) or any(not fold_objects[i] for i in range(folds)):
        raise ValueError("invalid deterministic object folds")
    return mapping, fold_objects


def object_states(keys: pd.DataFrame, state_rows: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    states: dict[str, np.ndarray] = {}
    variation: dict[str, float] = {}
    ids = keys.source_object_id.astype(str).to_numpy()
    for obj in sorted(set(ids)):
        rows = np.flatnonzero(ids == obj)
        block = state_rows[rows]
        states[obj] = block.mean(axis=0, dtype=np.float64)
        variation[obj] = float(np.mean(np.square(block - states[obj][None, :])))
    return states, variation


def per_drug_prediction(train_keys: pd.DataFrame, train_y: np.ndarray,
                        eval_keys: pd.DataFrame) -> np.ndarray:
    means: dict[str, np.ndarray] = {}
    drugs = train_keys.full_inchikey.astype(str).to_numpy()
    for drug in sorted(set(drugs)):
        means[drug] = train_y[drugs == drug].mean(axis=0, dtype=np.float64)
    eval_drugs = eval_keys.full_inchikey.astype(str).to_numpy()
    missing = sorted(set(eval_drugs) - set(means))
    if missing:
        raise ValueError(f"held-out fold contains drugs without training support: {missing}")
    return np.stack([means[drug] for drug in eval_drugs]).astype(np.float64)


def derangement(ids: list[str], rng: np.random.Generator) -> dict[str, str]:
    unique = sorted(set(ids))
    if len(unique) < 2:
        raise ValueError("state-shuffle control requires at least two objects in each partition")
    perm = np.asarray(unique, dtype=str)[rng.permutation(len(unique))]
    return {str(perm[i]): str(perm[(i + 1) % len(perm)]) for i in range(len(perm))}


def fit_state_residual(train_keys: pd.DataFrame, train_y: np.ndarray,
                       eval_keys: pd.DataFrame, drug_base_train: np.ndarray,
                       drug_base_eval: np.ndarray, states: dict[str, np.ndarray],
                       state_map: dict[str, str] | None, alpha: float) -> tuple[np.ndarray, dict[str, Any]]:
    train_ids = train_keys.source_object_id.astype(str).to_numpy()
    eval_ids = eval_keys.source_object_id.astype(str).to_numpy()
    train_state_ids = [state_map[obj] if state_map else obj for obj in train_ids]
    eval_state_ids = [state_map[obj] if state_map else obj for obj in eval_ids]
    train_x0 = np.stack([states[obj] for obj in train_state_ids]).astype(np.float64)
    eval_x0 = np.stack([states[obj] for obj in eval_state_ids]).astype(np.float64)
    mean = train_x0.mean(axis=0, dtype=np.float64)
    scale = train_x0.std(axis=0, dtype=np.float64)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    x_train = (train_x0 - mean) / scale
    x_eval = (eval_x0 - mean) / scale
    residual = train_y - drug_base_train
    counts = pd.Series(train_ids).value_counts().to_dict()
    sample_weight = np.asarray([1.0 / counts[obj] for obj in train_ids], dtype=np.float64)
    sample_weight *= len(sample_weight) / sample_weight.sum()
    model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
    model.fit(x_train, residual, sample_weight=sample_weight)
    residual_hat = model.predict(x_eval)
    prediction = drug_base_eval + residual_hat
    if not np.isfinite(prediction).all():
        raise FloatingPointError("state-residual ridge generated a non-finite prediction")
    return prediction, {"ridge_alpha": alpha, "fit_objects": int(len(set(train_ids))),
                       "fit_conditions": int(len(train_ids)), "feature_genes": int(x_train.shape[1]),
                       "scaler_fit_objects": int(len(set(train_state_ids))),
                       "training_object_total_weight_equalized": True,
                       "state_source": "object-mean paired X_ctl; fold-train standardization"}


def object_metrics(keys: pd.DataFrame, truth: np.ndarray, pred: np.ndarray,
                   gene_variance: np.ndarray) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    use = np.isfinite(gene_variance) & (gene_variance > 1e-12)
    if not use.any():
        raise ValueError("no nonzero-variance TRAIN genes remain in fold")
    err = pred.astype(np.float64) - truth.astype(np.float64)
    normalized = np.square(err[:, use]) / gene_variance[use][None, :]
    row_nmse = normalized.mean(axis=1)
    row_mae = np.abs(err).mean(axis=1)
    row_corr = np.full(len(truth), np.nan, dtype=np.float64)
    for i in range(len(truth)):
        if np.std(truth[i]) > 0 and np.std(pred[i]) > 0:
            row_corr[i] = float(np.corrcoef(truth[i], pred[i])[0, 1])
    row_direction = (np.sign(truth) == np.sign(pred)).mean(axis=1)
    ids = keys.source_object_id.astype(str).to_numpy()
    out: dict[str, dict[str, float]] = {}
    for obj in sorted(set(ids)):
        idx = np.flatnonzero(ids == obj)
        finite_corr = row_corr[idx][np.isfinite(row_corr[idx])]
        truth_amp = float(np.mean(np.abs(truth[idx])))
        pred_amp = float(np.mean(np.abs(pred[idx])))
        out[obj] = {"nmse": float(row_nmse[idx].mean()), "mae": float(row_mae[idx].mean()),
                    "gene_correlation": float(finite_corr.mean()) if len(finite_corr) else math.nan,
                    "direction_agreement": float(row_direction[idx].mean()),
                    "amplitude_ratio": pred_amp / truth_amp if truth_amp > 0 else math.nan,
                    "conditions": int(len(idx))}
    summary = {"object_macro_nmse": float(np.mean([x["nmse"] for x in out.values()])),
               "object_macro_mae": float(np.mean([x["mae"] for x in out.values()])),
               "object_macro_gene_correlation": finite_mean_or_none([x["gene_correlation"] for x in out.values()]),
               "object_macro_direction_agreement": float(np.mean([x["direction_agreement"] for x in out.values()])),
               "object_macro_amplitude_ratio": finite_mean_or_none([x["amplitude_ratio"] for x in out.values()]),
               "objects": len(out), "conditions": int(len(keys)), "genes_in_nmse": int(use.sum()),
               "zero_variance_genes_excluded": int((~use).sum())}
    return out, summary


def paired_bootstrap(object_scores: dict[str, dict[str, float]], reference: str, candidate: str,
                     reps: int, seed: int) -> dict[str, Any]:
    ids = sorted(set(object_scores))
    diff = np.asarray([object_scores[obj][reference] - object_scores[obj][candidate] for obj in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(ids), size=(reps, len(ids)))
    boot = diff[indexes].mean(axis=1)
    return {"reference": reference, "candidate": candidate, "objects": len(ids),
            "reference_minus_candidate_nmse": float(diff.mean()),
            "bootstrap_95pct_ci": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "positive_favors_candidate": True,
            "unit": "source object; fixed TRAIN-only object-OOF development diagnostic"}


def finalize_saved_oof(source_run_id: str, output_run_id: str, cfg_path: Path) -> dict[str, Any]:
    """Recover summary/receipts from saved OOF arrays without fitting or creating folds."""
    started = time.monotonic()
    source = ROOT / "artifacts/experiments/EXP-053" / source_run_id
    out = ROOT / "artifacts/experiments/EXP-053" / output_run_id
    if not source.is_dir() or out.exists():
        raise FileNotFoundError("source run must exist and recovery output must be new")
    required_files = ("failure_receipt.json", "per_object_method_metrics.tsv", "fold_method_metrics.tsv",
                     "fold_fit_and_permutation_receipt.tsv", "train_object_oof_predictions.npz",
                     "paired_object_bootstrap.json")
    if any(not (source / name).is_file() for name in required_files):
        raise FileNotFoundError("source run is missing one or more pre-serialization OOF outputs")
    failure_path = source / "failure_receipt.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    if (failure.get("status") != "FAILED_CLOSED" or failure.get("error_type") != "ValueError"
            or "Out of range float values are not JSON compliant" not in failure.get("error", "")):
        raise ValueError("source run did not close solely at non-finite summary JSON serialization")
    if (source / "result.json").exists() or (source / "sha256_manifest.tsv").exists():
        raise FileExistsError("source run already has finalized outputs; it must remain immutable")

    source_hashes = {name: {"sha256": sha256(source / name), "bytes": (source / name).stat().st_size}
                     for name in required_files}
    source_runner_sha = None
    card_text = (ROOT / "experiments/records/EXP-053.md").read_text(encoding="utf-8")
    match = re.search(r"实际加载runner SHA-256：`([0-9a-f]{64})`", card_text)
    if not match:
        raise ValueError("task card does not preserve the source runner SHA-256 at launch")
    source_runner_sha = match.group(1)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", output_run_id) or output_run_id == source_run_id:
        raise ValueError("recovery output run id is invalid or aliases the source run")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("experiment_id") != "EXP-053" or cfg.get("run_id") != source_run_id:
        raise ValueError("recovery config does not identify the immutable source run")
    if cfg["resources"].get("gpu_hours") != 0 or cfg["cv"].get("folds") != 5:
        raise ValueError("recovery config differs from the frozen CPU-only five-fold protocol")
    keys, genes, state_rows, delta, input_receipt = source_inputs(cfg["data"]["source_run_id"])
    with np.load(source / "train_object_oof_predictions.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    expected_arrays = {"condition_key", "source_object_id", "full_inchikey", "fold_id", "gene_axis",
                       "truth_delta", *METHODS}
    if set(arrays) != expected_arrays:
        raise ValueError("source OOF NPZ member set differs from the fixed prediction contract")
    expected_keys = keys.full_condition_key.astype(str).to_numpy()
    expected_objects = keys.source_object_id.astype(str).to_numpy()
    expected_drugs = keys.full_inchikey.astype(str).to_numpy()
    if (not np.array_equal(arrays["condition_key"].astype(str), expected_keys)
            or not np.array_equal(arrays["source_object_id"].astype(str), expected_objects)
            or not np.array_equal(arrays["full_inchikey"].astype(str), expected_drugs)
            or not np.array_equal(arrays["gene_axis"].astype(str), genes.astype(str))):
        raise ValueError("source OOF keys, objects, full drugs, or ordered genes differ from frozen TRAIN contract")
    if arrays["condition_key"].astype(str).shape != (419,) or not pd.Index(expected_keys).is_unique:
        raise ValueError("source TRAIN condition keys are not unique and complete")
    if (arrays["truth_delta"].shape != (419, 978) or arrays["fold_id"].shape != (419,)
            or any(arrays[name].shape != (419, 978) for name in METHODS)
            or any(not np.isfinite(arrays[name]).all() for name in ("truth_delta", *METHODS))):
        raise FloatingPointError("source OOF arrays have wrong shape or a non-finite target/prediction")
    truth_error = float(np.max(np.abs(arrays["truth_delta"].astype(np.float64) - delta)))
    if truth_error > 1e-6:
        raise ValueError(f"stored OOF truth differs from the TRAIN-only H5AD rows by {truth_error}")

    object_to_fold: dict[str, int] = {}
    for obj, fold in zip(expected_objects, arrays["fold_id"].astype(int)):
        if obj in object_to_fold and object_to_fold[obj] != fold:
            raise ValueError("one source object occurs in multiple held-out folds")
        object_to_fold[obj] = int(fold)
    if len(object_to_fold) != 110 or set(object_to_fold.values()) != set(range(5)):
        raise ValueError("source OOF folds do not cover 110 objects exactly once across five folds")
    expected_fold_map, fold_objects = split_objects(keys, int(cfg["cv"]["folds"]), int(cfg["cv"]["seed"]))
    if object_to_fold != expected_fold_map:
        raise ValueError("source OOF fold IDs differ from the registered label-blind deterministic split")

    fold_fit = pd.read_csv(source / "fold_fit_and_permutation_receipt.tsv", sep="\t", dtype=str,
                           keep_default_na=False)
    if len(fold_fit) != 5:
        raise ValueError("source fit/permutation receipt must contain exactly five folds")
    all_drugs = set(expected_drugs)
    for row in fold_fit.itertuples(index=False):
        fold = int(row.fold)
        train_idx = np.flatnonzero(arrays["fold_id"].astype(int) != fold)
        test_idx = np.flatnonzero(arrays["fold_id"].astype(int) == fold)
        train_objects = set(expected_objects[train_idx])
        test_objects = set(expected_objects[test_idx])
        if (int(row.training_drug_keys) != 5 or set(expected_drugs[train_idx]) != all_drugs
                or set(row.test_source_objects.split(";")) != test_objects):
            raise ValueError(f"fold {fold} receipt has changed object or full-drug support")
        state_model = ast.literal_eval(row.training_state_model)
        shuffled_model = ast.literal_eval(row.shuffled_state_model)
        for model in (state_model, shuffled_model):
            if (model.get("ridge_alpha") != float(cfg["models"]["state_residual"]["ridge_alpha"])
                    or model.get("fit_objects") != len(train_objects)
                    or model.get("fit_conditions") != int(len(train_idx))
                    or model.get("feature_genes") != 978 or model.get("scaler_fit_objects") != len(train_objects)
                    or model.get("training_object_total_weight_equalized") is not True
                    or model.get("state_source") != "object-mean paired X_ctl; fold-train standardization"):
                raise ValueError(f"fold {fold} training-only state preprocessing receipt differs")

    # Recreate the registered object-block derangements for an auditable identity/seed receipt only.
    shuffle_rng = np.random.default_rng(int(cfg["models"]["shuffled_state_capacity_control"]["seed"]))
    shuffle_receipts = []
    for fold in range(5):
        train_idx = np.flatnonzero(arrays["fold_id"].astype(int) != fold)
        test_idx = np.flatnonzero(arrays["fold_id"].astype(int) == fold)
        train_ids = sorted(set(expected_objects[train_idx]))
        test_ids = sorted(set(expected_objects[test_idx]))
        train_map = derangement(train_ids, shuffle_rng)
        test_map = derangement(test_ids, shuffle_rng)
        if (any(k == v for k, v in train_map.items()) or any(k == v for k, v in test_map.items())
                or set(train_map) != set(train_ids) or set(train_map.values()) != set(train_ids)
                or set(test_map) != set(test_ids) or set(test_map.values()) != set(test_ids)):
            raise ValueError(f"fold {fold} object-block shuffle is not a within-partition derangement")
        mapping_json = json.dumps({"train": train_map, "test": test_map}, sort_keys=True,
                                  separators=(",", ":"), ensure_ascii=False)
        shuffle_receipts.append({"fold": fold, "seed": int(cfg["models"]["shuffled_state_capacity_control"]["seed"]),
                                 "train_objects": len(train_ids), "test_objects": len(test_ids),
                                 "mapping_sha256": hashlib.sha256(mapping_json.encode()).hexdigest(),
                                 "mapping_reconstructed_from_registered_seed": True,
                                 "model_refit_during_recovery": False})

    fold_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    object_scores: dict[str, dict[str, float]] = {}
    validity: dict[str, dict[str, int]] = {name: {"objects_total": 110, "objects_gene_correlation_defined": 0,
                                                  "objects_amplitude_ratio_defined": 0,
                                                  "conditions_gene_correlation_defined": 0,
                                                  "conditions_total": 0} for name in METHODS}
    for fold in range(5):
        test_idx = np.flatnonzero(arrays["fold_id"].astype(int) == fold)
        train_idx = np.flatnonzero(arrays["fold_id"].astype(int) != fold)
        var = np.var(delta[train_idx].astype(np.float64), axis=0, ddof=0)
        test_keys = keys.iloc[test_idx].reset_index(drop=True)
        rows_by_method: dict[str, dict[str, dict[str, float]]] = {}
        for method in METHODS:
            details, summary = object_metrics(test_keys, delta[test_idx], arrays[method][test_idx], var)
            rows_by_method[method] = details
            fold_rows.append({"fold": fold, "method": method,
                              "train_conditions": len(train_idx), "test_conditions": len(test_idx),
                              "train_objects": int(len(set(expected_objects[train_idx]))),
                              "test_objects": int(len(set(expected_objects[test_idx]))),
                              "genes_in_nmse": summary["genes_in_nmse"],
                              "object_macro_nmse": summary["object_macro_nmse"],
                              "object_macro_mae": summary["object_macro_mae"],
                              "object_macro_gene_correlation": summary["object_macro_gene_correlation"],
                              "object_macro_direction_agreement": summary["object_macro_direction_agreement"],
                              "object_macro_amplitude_ratio": summary["object_macro_amplitude_ratio"]})
        for obj in fold_objects[fold]:
            scores = {method: rows_by_method[method][obj]["nmse"] for method in METHODS}
            object_scores[obj] = scores
            for method in METHODS:
                row = {"source_object_id": obj, "fold": fold, "method": method,
                       **rows_by_method[method][obj]}
                object_rows.append(row)
                validity[method]["conditions_total"] += int(row["conditions"])
                validity[method]["conditions_gene_correlation_defined"] += int(
                    sum(1 for idx in np.flatnonzero(expected_objects[test_idx] == obj)
                        if np.std(delta[test_idx][idx]) > 0 and np.std(arrays[method][test_idx][idx]) > 0))
                validity[method]["objects_gene_correlation_defined"] += int(
                    np.isfinite(row["gene_correlation"]))
                validity[method]["objects_amplitude_ratio_defined"] += int(
                    np.isfinite(row["amplitude_ratio"]))

    detail_frame = pd.DataFrame(object_rows)
    summaries = {}
    for method in METHODS:
        rows = detail_frame.loc[detail_frame.method.eq(method)]
        summaries[method] = {
            "object_macro_nmse": float(rows.nmse.mean()),
            "object_macro_mae": float(rows.mae.mean()),
            "object_macro_gene_correlation": finite_mean_or_none(rows.gene_correlation.to_numpy(dtype=np.float64)),
            "object_macro_direction_agreement": float(rows.direction_agreement.mean()),
            "object_macro_amplitude_ratio": finite_mean_or_none(rows.amplitude_ratio.to_numpy(dtype=np.float64)),
            "objects": int(rows.source_object_id.nunique()), "conditions": 419, "genes": 978,
        }
    reps, boot_seed = int(cfg["metrics"]["bootstrap_replicates"]), int(cfg["metrics"]["bootstrap_seed"])
    paired = [paired_bootstrap(object_scores, a, b, reps, boot_seed + i)
              for i, (a, b) in enumerate((
                  ("per_drug_mean", "state_residual"),
                  ("per_drug_mean", "shuffled_state_control"),
                  ("shuffled_state_control", "state_residual")))]
    saved_pairs = json.loads((source / "paired_object_bootstrap.json").read_text(encoding="utf-8"))["pairs"]
    if len(saved_pairs) != len(paired):
        raise ValueError("source bootstrap pair count differs from the frozen protocol")
    bootstrap_errors = []
    for saved, recomputed in zip(saved_pairs, paired):
        err = max(abs(float(saved["reference_minus_candidate_nmse"]) - recomputed["reference_minus_candidate_nmse"]),
                  *[abs(float(a) - float(b)) for a, b in zip(saved["bootstrap_95pct_ci"],
                                                             recomputed["bootstrap_95pct_ci"])])
        bootstrap_errors.append(err)
    if max(bootstrap_errors, default=0.0) > 2e-5:
        raise ValueError(f"saved bootstrap differs from recovered fixed predictions: {max(bootstrap_errors)}")

    # Compare recovered table contents with independent recomputation from the saved arrays.
    saved_obj = pd.read_csv(source / "per_object_method_metrics.tsv", sep="\t", dtype={"source_object_id": str})
    if len(saved_obj) != len(detail_frame):
        raise ValueError("saved per-object metric row count differs from the recovered complete denominator")
    joined = detail_frame.merge(saved_obj, on=["source_object_id", "fold", "method"], suffixes=("_r", "_s"),
                                validate="one_to_one")
    for metric in ("nmse", "mae", "gene_correlation", "direction_agreement", "amplitude_ratio", "conditions"):
        left, right = pd.to_numeric(joined[f"{metric}_r"], errors="coerce"), pd.to_numeric(
            joined[f"{metric}_s"], errors="coerce")
        if metric != "conditions" and not np.allclose(left, right, rtol=0, atol=2e-5, equal_nan=True):
            raise ValueError(f"saved per-object {metric} differs from recovery recomputation")
        if metric == "conditions" and not np.array_equal(left.astype(int), right.astype(int)):
            raise ValueError("saved per-object condition denominators differ")
    saved_fold = pd.read_csv(source / "fold_method_metrics.tsv", sep="\t")
    recomputed_fold = pd.DataFrame(fold_rows)
    merged_fold = recomputed_fold.merge(saved_fold, on=["fold", "method"], suffixes=("_r", "_s"), validate="one_to_one")
    for metric in ("object_macro_nmse", "object_macro_mae", "object_macro_direction_agreement",
                   "object_macro_amplitude_ratio", "object_macro_gene_correlation"):
        left = pd.to_numeric(merged_fold[f"{metric}_r"], errors="coerce")
        right = pd.to_numeric(merged_fold[f"{metric}_s"], errors="coerce")
        if not np.allclose(left, right, rtol=0, atol=2e-5, equal_nan=True):
            raise ValueError(f"saved fold {metric} differs from recovery recomputation")

    provenance = {
        "status": "RECOVERED_POSTPROCESSING_ONLY_NO_REFIT",
        "experiment_id": "EXP-053", "source_run_id": source_run_id, "output_run_id": output_run_id,
        "source_runner_sha256_at_launch": source_runner_sha,
        "current_runner_sha256": sha256(Path(__file__).resolve()),
        "config_sha256": sha256(cfg_path), "source_files": source_hashes,
        "source_failure_status": failure.get("status"), "source_failure_type": failure.get("error_type"),
        "source_failure_message": failure.get("error"),
        "recover_command": "run_exp053_drug_shared_state_residual.py --config configs/experiments/EXP-053-drug-shared-state-residual.yaml --finalize-from-run 20261007_exp053_drug_shared_state_residual_v1 --output-run-id 20261007_exp053_drug_shared_state_residual_recovery_v1",
        "fitting_performed": False, "new_fold_assignment_created": False,
        "gpu_hours": 0.0, "source_run_cpu_wall_upper_bound_minutes": 65.0,
        "source_run_cpu_wall_upper_bound_basis": "launch recorded at 02:00 +08; active at 02:49 +08; process absent by 03:04 +08; conservative bound 65 minutes, below registered 240-minute wall cap; aggregate CPU core-seconds were not captured at exit",
        "recovery_cpu_seconds": time.monotonic() - started,
        "recovered_at_utc": fit.utc_now(), "claim_boundary": "fixed 419 TRAIN condition, 110 source-object OOF development comparison only",
    }
    result = {
        "experiment_id": "EXP-053", "run_id": output_run_id,
        "status": "RECOVERED_FROM_SAVED_TRAIN_OOF_AFTER_RESULT_SERIALIZATION_FAILURE",
        "created_at_utc": fit.utc_now(), "source_run_id": source_run_id,
        "source_input_receipt": input_receipt,
        "denominator": {"train_conditions": 419, "train_objects": 110, "full_inchikeys": 5, "genes": 978,
                        "objects_per_fold": [len(x) for x in fold_objects],
                        "conditions_per_fold": [int(np.sum(arrays["fold_id"].astype(int) == f)) for f in range(5)],
                        "all_rows_oof_once": True, "inner_dev_numeric_values_read": False,
                        "object_overlap_train_test_by_fold": 0},
        "methods": summaries, "paired_object_bootstrap": {"pairs": paired,
            "claim_boundary": "TRAIN-only source-object OOF; not PDO/patient generalization"},
        "undefined_metric_denominators": validity,
        "shuffle_mapping_receipts": shuffle_receipts,
        "fold_method_metrics_source_path": str(source / "fold_method_metrics.tsv"),
        "per_object_method_metrics_source_path": str(source / "per_object_method_metrics.tsv"),
        "prediction_archive_source_path": str(source / "train_object_oof_predictions.npz"),
        "recovery_provenance": provenance,
        "selection": {"ridge_alpha": float(cfg["models"]["state_residual"]["ridge_alpha"]),
                      "parameter_search": False,
                      "feature_standardization": "fold-training source-object states only",
                      "object_weighting": "equal total weight per train object",
                      "state_summary": "mean paired X_ctl across each source object's TRAIN conditions",
                      "fold_assignment": "existing deterministic label-blind source-object split; preserved from source NPZ"},
        "forbidden_roles_opened": [],
        "forbidden_numeric_values_read": {"inner_dev": False, "pdo_sensitivity": False, "outer": False,
                                           "locked": False},
        "resource": {"gpu_hours": 0.0, "recovery_cpu_seconds": provenance["recovery_cpu_seconds"],
                     "source_run_wall_upper_bound_minutes": 65.0,
                     "cpu_cap_minutes": int(cfg["resources"]["cpu_cap_minutes"]),
                     "global_settled_gpu_hours": 1.2241092560986186,
                     "global_reserved_gpu_hours": 0.0},
    }
    # Assert the immutable failed source artifacts did not change during recovery.
    for name, expected in source_hashes.items():
        if sha256(source / name) != expected["sha256"]:
            raise RuntimeError(f"source failure-run artifact changed during recovery: {name}")
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "result.json", result)
    atomic_json(out / "recovery_provenance.json", provenance)
    sha_rows = [(p.name, sha256(p), p.stat().st_size) for p in sorted(out.iterdir()) if p.is_file()]
    pd.DataFrame(sha_rows, columns=["file", "sha256", "bytes"]).to_csv(
        out / "sha256_manifest.tsv", sep="\t", index=False)
    return {"status": result["status"], "run_id": output_run_id, "source_run_id": source_run_id,
            "methods": summaries, "paired": paired, "undefined_metrics": validity,
            "source_failure_preserved": True, "fitting_performed": False,
            "gpu_hours": 0.0, "output": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--finalize-from-run")
    parser.add_argument("--output-run-id")
    args = parser.parse_args()
    if args.finalize_from_run:
        if not args.output_run_id:
            raise ValueError("--output-run-id is required with --finalize-from-run")
        cfg_path = (ROOT / args.config).resolve()
        recovered = finalize_saved_oof(args.finalize_from_run, args.output_run_id, cfg_path)
        print(json.dumps(recovered, ensure_ascii=False, allow_nan=False), flush=True)
        return
    if not args.run_id or args.output_run_id:
        raise ValueError("normal fitting requires --run-id and does not accept --output-run-id")
    started = time.monotonic()
    cfg_path = (ROOT / args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("experiment_id") != "EXP-053" or cfg.get("run_id") != args.run_id:
        raise ValueError("config experiment/run identifiers do not match")
    if cfg["resources"].get("gpu_hours") != 0 or cfg["cv"].get("folds") != 5:
        raise ValueError("EXP-053 resource or fold protocol differs from the frozen CPU-only design")
    out = ROOT / cfg["reporting"]["output_root"]
    out.mkdir(parents=True, exist_ok=False)
    try:
        source_id = cfg["data"]["source_run_id"]
        keys, genes, state_rows, delta, input_receipt = source_inputs(source_id)
        state_by_object, state_variation = object_states(keys, state_rows)
        object_fold, fold_objects = split_objects(keys, int(cfg["cv"]["folds"]), int(cfg["cv"]["seed"]))
        fold_col = keys.source_object_id.astype(str).map(object_fold)
        if fold_col.isna().any():
            raise ValueError("one or more TRAIN rows lack a frozen source-object fold")
        fold_ids = fold_col.to_numpy(dtype=np.int16)
        all_drugs = sorted(keys.full_inchikey.astype(str).unique().tolist())
        if len(all_drugs) != 5:
            raise ValueError("expected the five frozen full-InChIKey drugs")
        pred = {name: np.full_like(delta, np.nan, dtype=np.float64)
                for name in ("zero_change", "per_drug_mean", "state_residual", "shuffled_state_control")}
        fold_rows: list[dict[str, Any]] = []
        fit_rows: list[dict[str, Any]] = []
        rng_shuffle = np.random.default_rng(int(cfg["models"]["shuffled_state_capacity_control"]["seed"]))
        alpha = float(cfg["models"]["state_residual"]["ridge_alpha"])
        for fold in range(int(cfg["cv"]["folds"])):
            train_idx = np.flatnonzero(fold_ids != fold)
            test_idx = np.flatnonzero(fold_ids == fold)
            train_keys, test_keys = keys.iloc[train_idx].reset_index(drop=True), keys.iloc[test_idx].reset_index(drop=True)
            train_y, test_y = delta[train_idx], delta[test_idx]
            train_drugs = set(train_keys.full_inchikey.astype(str))
            test_drugs = set(test_keys.full_inchikey.astype(str))
            if train_drugs != set(all_drugs) or not test_drugs:
                raise ValueError(f"fold {fold} does not preserve all five TRAIN drug keys")
            base_train = per_drug_prediction(train_keys, train_y, train_keys)
            base_test = per_drug_prediction(train_keys, train_y, test_keys)
            pred["zero_change"][test_idx] = 0.0
            pred["per_drug_mean"][test_idx] = base_test
            state_pred, method = fit_state_residual(train_keys, train_y, test_keys,
                                                    base_train, base_test, state_by_object, None, alpha)
            train_obj = sorted(train_keys.source_object_id.astype(str).unique().tolist())
            test_obj = sorted(test_keys.source_object_id.astype(str).unique().tolist())
            train_map = derangement(train_obj, rng_shuffle)
            test_map = derangement(test_obj, rng_shuffle)
            shuffle_map = {**train_map, **test_map}
            shuffle_pred, shuffle_method = fit_state_residual(train_keys, train_y, test_keys,
                                                               base_train, base_test, state_by_object,
                                                               shuffle_map, alpha)
            pred["state_residual"][test_idx] = state_pred
            pred["shuffled_state_control"][test_idx] = shuffle_pred
            var = np.var(train_y.astype(np.float64), axis=0, ddof=0)
            for method_name in pred:
                _, summ = object_metrics(test_keys, test_y,
                                         pred[method_name][test_idx], var)
                fold_rows.append({"fold": fold, "method": method_name,
                                  "train_conditions": len(train_idx), "test_conditions": len(test_idx),
                                  "train_objects": int(train_keys.source_object_id.nunique()),
                                  "test_objects": int(test_keys.source_object_id.nunique()),
                                  "genes_in_nmse": summ["genes_in_nmse"],
                                  "object_macro_nmse": summ["object_macro_nmse"],
                                  "object_macro_mae": summ["object_macro_mae"],
                                  "object_macro_gene_correlation": summ["object_macro_gene_correlation"],
                                  "object_macro_direction_agreement": summ["object_macro_direction_agreement"],
                                  "object_macro_amplitude_ratio": summ["object_macro_amplitude_ratio"]})
            fit_rows.append({"fold": fold, "test_source_objects": ";".join(fold_objects[fold]),
                             "training_drug_keys": len(train_drugs),
                             "training_state_model": method, "shuffled_state_model": shuffle_method,
                             "alpha_fixed_no_search": alpha})
        if any(not np.isfinite(value).all() for value in pred.values()):
            raise FloatingPointError("some TRAIN source condition did not receive a finite OOF prediction")
        object_scores: dict[str, dict[str, float]] = {}
        all_object_rows: list[dict[str, Any]] = []
        for fold in range(int(cfg["cv"]["folds"])):
            test_idx = np.flatnonzero(fold_ids == fold)
            test_keys = keys.iloc[test_idx].reset_index(drop=True)
            test_y = delta[test_idx]
            train_y = delta[fold_ids != fold]
            var = np.var(train_y.astype(np.float64), axis=0, ddof=0)
            by_method: dict[str, dict[str, dict[str, float]]] = {}
            for method_name, values in pred.items():
                by_method[method_name], _ = object_metrics(test_keys, test_y, values[test_idx], var)
            for obj in fold_objects[fold]:
                scores = {name: by_method[name][obj]["nmse"] for name in pred}
                object_scores[obj] = scores
                for method_name in pred:
                    all_object_rows.append({"source_object_id": obj, "fold": fold,
                                            "method": method_name, **by_method[method_name][obj]})
        summaries = {}
        for method_name in pred:
            vals = [row for row in all_object_rows if row["method"] == method_name]
            summaries[method_name] = {
                "object_macro_nmse": float(np.mean([x["nmse"] for x in vals])),
                "object_macro_mae": float(np.mean([x["mae"] for x in vals])),
                "object_macro_gene_correlation": finite_mean_or_none([x["gene_correlation"] for x in vals]),
                "object_macro_direction_agreement": float(np.mean([x["direction_agreement"] for x in vals])),
                "object_macro_amplitude_ratio": finite_mean_or_none([x["amplitude_ratio"] for x in vals]),
                "objects": len(vals), "conditions": int(len(keys)), "genes": 978,
            }
        reps = int(cfg["metrics"]["bootstrap_replicates"])
        seed = int(cfg["metrics"]["bootstrap_seed"])
        paired = [paired_bootstrap(object_scores, a, b, reps, seed + i)
                  for i, (a, b) in enumerate((
                      ("per_drug_mean", "state_residual"),
                      ("per_drug_mean", "shuffled_state_control"),
                      ("shuffled_state_control", "state_residual")))]
        obj_frame = pd.DataFrame(all_object_rows).sort_values(["source_object_id", "method"])
        obj_frame.to_csv(out / "per_object_method_metrics.tsv", sep="\t", index=False)
        pd.DataFrame(fold_rows).to_csv(out / "fold_method_metrics.tsv", sep="\t", index=False)
        pd.DataFrame(fit_rows).to_csv(out / "fold_fit_and_permutation_receipt.tsv", sep="\t", index=False)
        pred_path = out / "train_object_oof_predictions.npz"
        np.savez_compressed(pred_path,
                            condition_key=keys.full_condition_key.astype(str).to_numpy(dtype=str),
                            source_object_id=keys.source_object_id.astype(str).to_numpy(dtype=str),
                            full_inchikey=keys.full_inchikey.astype(str).to_numpy(dtype=str),
                            fold_id=fold_ids.astype(np.int16), gene_axis=genes.astype(str),
                            truth_delta=delta.astype(np.float32),
                            zero_change=pred["zero_change"].astype(np.float32),
                            per_drug_mean=pred["per_drug_mean"].astype(np.float32),
                            state_residual=pred["state_residual"].astype(np.float32),
                            shuffled_state_control=pred["shuffled_state_control"].astype(np.float32))
        bootstrap_payload = {"pairs": paired, "claim_boundary": "TRAIN-only source-object OOF; not PDO/patient generalization"}
        atomic_json(out / "paired_object_bootstrap.json", bootstrap_payload)
        elapsed = time.monotonic() - started
        result = {
            "experiment_id": "EXP-053", "run_id": args.run_id, "status": "COMPLETED_TRAIN_ONLY_OBJECT_OOF_DEVELOPMENT",
            "created_at_utc": fit.utc_now(), "config_sha256": sha256(cfg_path),
            "runner_sha256": sha256(Path(__file__).resolve()), "source_input_receipt": input_receipt,
            "denominator": {"train_conditions": len(keys), "train_objects": int(keys.source_object_id.nunique()),
                            "full_inchikeys": int(keys.full_inchikey.nunique()), "genes": len(genes),
                            "objects_per_fold": [len(x) for x in fold_objects],
                            "conditions_per_fold": [int(np.sum(fold_ids == f)) for f in range(5)],
                            "all_rows_oof_once": bool(np.all(np.isfinite(pred["state_residual"]))),
                            "inner_dev_numeric_values_read": False, "object_overlap_train_test_by_fold": 0},
            "methods": summaries, "paired_object_bootstrap": bootstrap_payload,
            "selection": {"ridge_alpha": alpha, "parameter_search": False,
                          "feature_standardization": "fold-training source-object states only",
                          "object_weighting": "equal total weight per train object",
                          "state_summary": "mean paired X_ctl across each source object's TRAIN conditions",
                          "fold_assignment": "label-blind deterministic source-object shuffle then round-robin"},
            "state_within_object_mean_squared_variation": {
                "median": float(np.median(list(state_variation.values()))),
                "max": float(np.max(list(state_variation.values()))),
                "interpretation": "diagnostic of paired-control variability; state input is their per-object mean"},
            "forbidden_roles_opened": [], "forbidden_numeric_values_read": {"inner_dev": False,
                "pdo_sensitivity": False, "outer": False, "locked": False},
            "resource": {"gpu_hours": 0.0, "cpu_seconds": elapsed,
                         "cpu_cap_minutes": int(cfg["resources"]["cpu_cap_minutes"])}
        }
        atomic_json(out / "result.json", result)
        sha_rows = []
        for p in sorted(out.iterdir()):
            if p.is_file() and p.name != "sha256_manifest.tsv":
                sha_rows.append((p.name, sha256(p), p.stat().st_size))
        pd.DataFrame(sha_rows, columns=["file", "sha256", "bytes"]).to_csv(
            out / "sha256_manifest.tsv", sep="\t", index=False)
        print(json.dumps({"status": result["status"], "run_id": args.run_id,
                          "denominator": result["denominator"], "methods": summaries,
                          "paired": paired, "gpu_hours": 0, "output": str(out)},
                         ensure_ascii=False), flush=True)
        if elapsed > int(cfg["resources"]["cpu_cap_minutes"]) * 60:
            raise TimeoutError("EXP-053 run exceeded its CPU cap")
    except Exception as exc:
        atomic_json(out / "failure_receipt.json", {"status": "FAILED_CLOSED", "error_type": type(exc).__name__,
                    "error": str(exc), "gpu_hours": 0, "run_id": args.run_id,
                    "traceback_role_boundary": "TRAIN-only input routine; no GPU authorized"})
        raise


if __name__ == "__main__":
    main()
