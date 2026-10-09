"""Post-hoc, TRAIN-only error decomposition for frozen XPert attempt009 outputs.

Only the six TRAIN fields named in ``read_train_fields`` are indexed from the
attempt009 prediction archive. Inner-dev, PDO, outer, locked, and checkpoint
arrays are never opened by this entry point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_RUN_ID = "20261007_exp055_representative_models_v1"
DEFAULT_SOURCE_ATTEMPT = "fit_attempt_009"
OUTPUT_RUN_ID = "20261007_exp055_xpert_train_error_components_v1"
sys.path.insert(0, str(ROOT / "scripts/evaluation"))
from verify_exp051_low_rank_baselines import metrics  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _finite_or_none(value: float | int | np.floating[Any]) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    result = float(np.corrcoef(a, b)[0, 1])
    return result if math.isfinite(result) else None


def _row_stats(truth: np.ndarray, prediction: np.ndarray, variance: np.ndarray) -> dict[str, float | None]:
    y = np.asarray(truth, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    error = p - y
    use = np.isfinite(variance) & (variance > 1e-12)
    if y.shape != p.shape or y.ndim != 1 or not use.any():
        raise ValueError("row metric input axes differ or lack a positive TRAIN variance denominator")
    nonzero = np.abs(y) > 1e-12
    return {
        "nmse": _finite_or_none(np.mean(error[use] ** 2 / variance[use])),
        "mae": _finite_or_none(np.mean(np.abs(error))),
        "rmse": _finite_or_none(math.sqrt(float(np.mean(error ** 2)))),
        "correlation": _correlation(y, p),
        "direction_agreement": _finite_or_none(np.mean(np.sign(y[nonzero]) == np.sign(p[nonzero]))) if nonzero.any() else None,
        "true_mean_abs": _finite_or_none(np.mean(np.abs(y))),
        "predicted_mean_abs": _finite_or_none(np.mean(np.abs(p))),
        "amplitude_ratio": _finite_or_none(np.mean(np.abs(p)) / np.mean(np.abs(y))) if np.mean(np.abs(y)) > 0 else None,
    }


def _weighted_residual_metrics(truth: np.ndarray, prediction: np.ndarray,
                               row_weights: np.ndarray) -> dict[str, float | None]:
    """Summarize gene residuals with equal source-object weight."""
    y = np.asarray(truth, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    w = np.asarray(row_weights, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 2 or len(w) != len(y) or np.any(w <= 0):
        raise ValueError("weighted residual axes/weights are invalid")
    w = w / w.sum()
    expanded = w[:, None] / y.shape[1]
    error = p - y
    target_mean = float(np.sum(expanded * y))
    pred_mean = float(np.sum(expanded * p))
    target_centered = y - target_mean
    pred_centered = p - pred_mean
    target_var = float(np.sum(expanded * target_centered ** 2))
    pred_var = float(np.sum(expanded * pred_centered ** 2))
    mse = float(np.sum(expanded * error ** 2))
    mae = float(np.sum(expanded * np.abs(error)))
    covariance = float(np.sum(expanded * target_centered * pred_centered))
    corr = covariance / math.sqrt(target_var * pred_var) if target_var > 0 and pred_var > 0 else None
    nonzero = np.abs(y) > 1e-12
    direction = (float(np.sum(expanded * nonzero * (np.sign(y) == np.sign(p)))
                       / np.sum(expanded * nonzero)) if np.any(nonzero) else None)
    return {
        "target_residual_rms": _finite_or_none(math.sqrt(max(target_var, 0.0))),
        "predicted_residual_rms": _finite_or_none(math.sqrt(max(pred_var, 0.0))),
        "residual_amplitude_ratio": _finite_or_none(math.sqrt(pred_var / target_var)) if target_var > 0 else None,
        "residual_rmse": _finite_or_none(math.sqrt(mse)),
        "residual_mae": _finite_or_none(mae),
        "residual_nmse_vs_zero_state": _finite_or_none(mse / target_var) if target_var > 0 else None,
        "residual_correlation": _finite_or_none(corr) if corr is not None else None,
        "residual_direction_agreement": _finite_or_none(direction) if direction is not None else None,
    }


def read_train_fields(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    required = (
        "train_condition_key", "train_source_object_id", "train_full_inchikey",
        "genes", "train_target_delta", "XPert_best_inner_dev_train_delta",
    )
    # Never index inner-dev arrays; only the frozen TRAIN keys, IDs, drug IDs,
    # gene axis, target, and selected-checkpoint TRAIN prediction are read.
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise KeyError(f"attempt009 archive is missing allowed TRAIN fields: {missing}")
        keys = pd.DataFrame({
            "full_condition_key": archive["train_condition_key"].astype(str, copy=True),
            "source_object_id": archive["train_source_object_id"].astype(str, copy=True),
            "full_inchikey": archive["train_full_inchikey"].astype(str, copy=True),
        })
        genes = archive["genes"].astype(str, copy=True)
        truth = archive["train_target_delta"].astype(np.float32, copy=True)
        prediction = archive["XPert_best_inner_dev_train_delta"].astype(np.float32, copy=True)
    if truth.shape != (419, 978) or prediction.shape != truth.shape:
        raise ValueError(f"selected TRAIN target/prediction shape changed: {truth.shape}, {prediction.shape}")
    if len(keys) != 419 or keys.full_condition_key.nunique() != 419:
        raise ValueError("TRAIN condition keys are not exactly 419 unique keys")
    if keys.source_object_id.nunique() != 110:
        raise ValueError("TRAIN source object denominator is not 110")
    if keys.full_inchikey.nunique() != 5 or any(len(key) != 27 or key.count("-") != 2
                                               for key in keys.full_inchikey.astype(str).unique()):
        raise ValueError("TRAIN drug axis is not exactly five complete InChIKeys")
    if len(genes) != 978 or len(set(genes.tolist())) != 978:
        raise ValueError("TRAIN gene axis is not exactly 978 unique genes")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise FloatingPointError("TRAIN selected target/prediction contains non-finite values")
    return keys, truth, prediction, genes


def build_baseline(keys: pd.DataFrame, truth: np.ndarray) -> np.ndarray:
    key_values = keys.full_inchikey.astype(str).to_numpy()
    means = {
        drug: truth[key_values == drug].mean(axis=0, dtype=np.float64).astype(np.float32)
        for drug in sorted(np.unique(key_values))
    }
    return np.stack([means[drug] for drug in key_values], axis=0)


def build_object_drug_groups(keys: pd.DataFrame, truth: np.ndarray,
                             prediction: np.ndarray) -> list[dict[str, Any]]:
    table = keys[["source_object_id", "full_inchikey"]].astype(str).copy()
    groups: list[dict[str, Any]] = []
    for (object_id, drug), positions in table.groupby(["source_object_id", "full_inchikey"], sort=True).indices.items():
        ix = np.asarray(positions, dtype=np.int64)
        groups.append({
            "source_object_id": str(object_id),
            "full_inchikey": str(drug),
            "condition_count": int(len(ix)),
            "truth": truth[ix].astype(np.float64).mean(axis=0),
            "prediction": prediction[ix].astype(np.float64).mean(axis=0),
        })
    return groups


def centered_group_matrices(groups: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    group_objects = [row["source_object_id"] for row in groups]
    group_drugs = [row["full_inchikey"] for row in groups]
    truth_means: dict[str, np.ndarray] = {}
    prediction_means: dict[str, np.ndarray] = {}
    for drug in sorted(set(group_drugs)):
        selected = [row for row in groups if row["full_inchikey"] == drug]
        truth_means[drug] = np.stack([row["truth"] for row in selected]).mean(axis=0)
        prediction_means[drug] = np.stack([row["prediction"] for row in selected]).mean(axis=0)
    truth_resid = np.stack([row["truth"] - truth_means[row["full_inchikey"]] for row in groups])
    prediction_resid = np.stack([row["prediction"] - prediction_means[row["full_inchikey"]] for row in groups])
    scaled_resid = prediction_resid.copy()
    objects = sorted(set(group_objects))
    drugs_per_object = {obj: len({drug for got_obj, drug in zip(group_objects, group_drugs) if got_obj == obj})
                        for obj in objects}
    weights = np.asarray([1.0 / drugs_per_object[obj] for obj in group_objects], dtype=np.float64)
    return truth_resid, prediction_resid, weights, group_objects, group_drugs


def source_object_equal_condition_weights(keys: pd.DataFrame) -> np.ndarray:
    """Weight original condition rows equally within object, then objects equally."""
    object_ids = keys["source_object_id"].astype(str)
    condition_counts = object_ids.value_counts(sort=False).to_dict()
    n_objects = int(object_ids.nunique())
    if n_objects < 1 or not condition_counts:
        raise ValueError("cannot form source-object condition weights from an empty key table")
    return np.asarray([1.0 / (n_objects * int(condition_counts[obj])) for obj in object_ids],
                      dtype=np.float64)


def fit_nonnegative_global_scale(
    truth: np.ndarray,
    prediction: np.ndarray,
    row_weights: np.ndarray,
    train_variance: np.ndarray,
) -> tuple[float, bool, float, float]:
    """Fit the nonnegative scalar for the exact object-macro TRAIN NMSE.

    This operates on original condition rows, before any object/drug averaging.
    ``train_variance`` is the same TRAIN-fitted gene variance used by
    ``metrics()``. The common per-row gene-count divisor cancels in alpha.
    Returns (alpha, identifiable, numerator, denominator). If the prediction
    has zero variance-weighted norm, alpha is not identifiable and the feasible
    unscaled value alpha=1 is retained.
    """
    y = np.asarray(truth, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    w = np.asarray(row_weights, dtype=np.float64)
    v = np.asarray(train_variance, dtype=np.float64)
    if y.ndim != 2 or p.shape != y.shape or v.shape != (y.shape[1],) or w.shape != (y.shape[0],):
        raise ValueError("NMSE scale fit requires aligned original conditions, weights, and gene variances")
    use = np.isfinite(v) & (v > 1e-12)
    if not use.any() or not np.isfinite(y).all() or not np.isfinite(p).all() or not np.isfinite(w).all():
        raise ValueError("NMSE scale fit requires finite arrays and at least one positive-variance gene")
    if np.any(w <= 0) or not np.isclose(w.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("condition weights must be positive and sum to one")
    weighted_gene = w[:, None] / v[None, use]
    numerator = float(np.sum(weighted_gene * p[:, use] * y[:, use], dtype=np.float64))
    denominator = float(np.sum(weighted_gene * p[:, use] ** 2, dtype=np.float64))
    if denominator <= 0.0:
        return 1.0, False, numerator, denominator
    return max(0.0, numerator / denominator), True, numerator, denominator


def fit_nonnegative_global_mse_scale(
    truth: np.ndarray,
    prediction: np.ndarray,
    row_weights: np.ndarray,
) -> tuple[float, bool, float, float]:
    """Fit a separate object-macro raw-MSE scalar on original condition rows."""
    y = np.asarray(truth, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    w = np.asarray(row_weights, dtype=np.float64)
    if y.ndim != 2 or p.shape != y.shape or w.shape != (y.shape[0],):
        raise ValueError("MSE scale fit requires aligned original conditions and weights")
    if not np.isfinite(y).all() or not np.isfinite(p).all() or not np.isfinite(w).all():
        raise ValueError("MSE scale fit requires finite arrays")
    if np.any(w <= 0) or not np.isclose(w.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("condition weights must be positive and sum to one")
    numerator = float(np.sum(w[:, None] * p * y, dtype=np.float64))
    denominator = float(np.sum(w[:, None] * p ** 2, dtype=np.float64))
    if denominator <= 0.0:
        return 1.0, False, numerator, denominator
    return max(0.0, numerator / denominator), True, numerator, denominator


def fit_legacy_object_drug_mse_scale(groups: list[dict[str, Any]], weights: np.ndarray) -> float:
    """Recompute v2's object-by-drug-averaged raw-MSE scale for audit only."""
    truth = np.stack([row["truth"] for row in groups]).astype(np.float64)
    prediction = np.stack([row["prediction"] for row in groups]).astype(np.float64)
    row_weight = np.asarray(weights, dtype=np.float64)
    numerator = float(np.sum(row_weight[:, None] * prediction * truth, dtype=np.float64))
    denominator = float(np.sum(row_weight[:, None] * prediction ** 2, dtype=np.float64))
    return max(0.0, numerator / denominator) if denominator > 0.0 else 1.0


def weighted_object_macro_nmse_float64(
    truth: np.ndarray, prediction: np.ndarray, keys: pd.DataFrame, variance: np.ndarray
) -> float:
    """Evaluate the exact row-then-object weighting in float64 for scale checks."""
    y = np.asarray(truth, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    v = np.asarray(variance, dtype=np.float64)
    use = np.isfinite(v) & (v > 1e-12)
    if y.shape != p.shape or y.ndim != 2 or not use.any():
        raise ValueError("float64 NMSE check has invalid axes or denominator")
    row_nmse = np.mean((p[:, use] - y[:, use]) ** 2 / v[None, use], axis=1, dtype=np.float64)
    per_condition = pd.DataFrame({"object": keys.source_object_id.astype(str).to_numpy(), "nmse": row_nmse})
    return float(per_condition.groupby("object", sort=True).nmse.mean().mean())


def weighted_object_macro_mse_float64(
    truth: np.ndarray, prediction: np.ndarray, keys: pd.DataFrame
) -> float:
    """Object-macro raw MSE using original conditions and the full gene axis."""
    y = np.asarray(truth, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 2:
        raise ValueError("float64 MSE check has invalid axes")
    row_mse = np.mean((p - y) ** 2, axis=1, dtype=np.float64)
    per_condition = pd.DataFrame({"object": keys.source_object_id.astype(str).to_numpy(), "mse": row_mse})
    return float(per_condition.groupby("object", sort=True).mse.mean().mean())


def build_rows(keys: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray,
               baseline: np.ndarray, scaled: np.ndarray, variance: np.ndarray,
               centered_variance: np.ndarray,
               true_centered: np.ndarray, pred_centered: np.ndarray,
               scaled_centered: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    row_data: list[dict[str, Any]] = []
    for i in range(len(keys)):
        raw = _row_stats(truth[i], prediction[i], variance)
        base = _row_stats(truth[i], baseline[i], variance)
        scaled_metric = _row_stats(truth[i], scaled[i], variance)
        raw_state = _row_stats(true_centered[i], pred_centered[i], centered_variance)
        scaled_state = _row_stats(true_centered[i], scaled_centered[i], centered_variance)
        row_data.append({
            "full_condition_key": str(keys.full_condition_key.iloc[i]),
            "source_object_id": str(keys.source_object_id.iloc[i]),
            "full_inchikey": str(keys.full_inchikey.iloc[i]),
            "xpert_nmse": raw["nmse"], "xpert_mae": raw["mae"], "xpert_rmse": raw["rmse"],
            "xpert_gene_correlation": raw["correlation"], "xpert_direction_agreement": raw["direction_agreement"],
            "xpert_mean_absolute_prediction": raw["predicted_mean_abs"],
            "target_mean_absolute": raw["true_mean_abs"], "xpert_amplitude_ratio": raw["amplitude_ratio"],
            "per_drug_mean_nmse": base["nmse"], "per_drug_mean_mae": base["mae"],
            "nmse_difference_vs_per_drug_mean": raw["nmse"] - base["nmse"] if raw["nmse"] is not None and base["nmse"] is not None else None,
            "relative_nmse_reduction_vs_per_drug_mean": (1.0 - raw["nmse"] / base["nmse"]
                if raw["nmse"] is not None and base["nmse"] not in (None, 0.0) else None),
            "globally_scaled_nmse": scaled_metric["nmse"], "globally_scaled_mae": scaled_metric["mae"],
            "target_drug_centered_mean_absolute": raw_state["true_mean_abs"],
            "predicted_drug_centered_mean_absolute": raw_state["predicted_mean_abs"],
            "drug_centered_residual_nmse_vs_train_centered_variance": raw_state["nmse"],
            "drug_centered_residual_mae": raw_state["mae"],
            "drug_centered_residual_rmse": raw_state["rmse"],
            "drug_centered_residual_correlation": raw_state["correlation"],
            "drug_centered_residual_direction_agreement": raw_state["direction_agreement"],
            "scaled_drug_centered_residual_nmse_vs_train_centered_variance": scaled_state["nmse"],
            "scaled_drug_centered_residual_mae": scaled_state["mae"],
            "scaled_drug_centered_residual_rmse": scaled_state["rmse"],
            "scaled_drug_centered_residual_correlation": scaled_state["correlation"],
        })
    per_key = pd.DataFrame(row_data)

    object_rows: list[dict[str, Any]] = []
    object_drug_table = keys[["source_object_id", "full_inchikey"]].astype(str).copy()
    object_drug_residuals: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for (object_id, drug), positions in object_drug_table.groupby(
            ["source_object_id", "full_inchikey"], sort=True).indices.items():
        ix = np.asarray(positions, dtype=np.int64)
        object_drug_residuals[(str(object_id), str(drug))] = (
            true_centered[ix].astype(np.float64).mean(axis=0),
            pred_centered[ix].astype(np.float64).mean(axis=0),
            scaled_centered[ix].astype(np.float64).mean(axis=0),
        )
    for object_id, frame in per_key.groupby("source_object_id", sort=True):
        obj = str(object_id)
        local = frame
        obj_groups = [value for (got_obj, _), value in object_drug_residuals.items() if got_obj == obj]
        obj_y = np.stack([value[0] for value in obj_groups])
        obj_p = np.stack([value[1] for value in obj_groups])
        obj_scaled = np.stack([value[2] for value in obj_groups])
        obj_state = _weighted_residual_metrics(obj_y, obj_p, np.ones(len(obj_groups), dtype=np.float64))
        scaled_obj_state = _weighted_residual_metrics(obj_y, obj_scaled, np.ones(len(obj_groups), dtype=np.float64))
        object_rows.append({
            "source_object_id": obj,
            "conditions": int(len(local)),
            "drug_count": int(local.full_inchikey.nunique()),
            "xpert_object_macro_nmse": float(local.xpert_nmse.mean()),
            "per_drug_mean_object_macro_nmse": float(local.per_drug_mean_nmse.mean()),
            "nmse_difference_vs_per_drug_mean": float(local.nmse_difference_vs_per_drug_mean.mean()),
            "relative_nmse_reduction_vs_per_drug_mean": float(local.relative_nmse_reduction_vs_per_drug_mean.mean()),
            "xpert_object_macro_mae": float(local.xpert_mae.mean()),
            "per_drug_mean_object_macro_mae": float(local.per_drug_mean_mae.mean()),
            "xpert_object_macro_rmse": float(local.xpert_rmse.mean()),
            "xpert_object_macro_gene_correlation": _finite_or_none(local.xpert_gene_correlation.mean()),
            "xpert_object_macro_direction_agreement": _finite_or_none(local.xpert_direction_agreement.mean()),
            "target_mean_absolute": float(local.target_mean_absolute.mean()),
            "xpert_mean_absolute_prediction": float(local.xpert_mean_absolute_prediction.mean()),
            "drug_centered_state_residual_correlation": obj_state["residual_correlation"],
            "drug_centered_state_residual_nmse_vs_zero": obj_state["residual_nmse_vs_zero_state"],
            "globally_scaled_state_residual_nmse_vs_zero": scaled_obj_state["residual_nmse_vs_zero_state"],
        })
    per_object = pd.DataFrame(object_rows)
    return per_key, per_object


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=OUTPUT_RUN_ID)
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--source-attempt", default=DEFAULT_SOURCE_ATTEMPT)
    parser.add_argument("--scope", choices=("train",), default="train")
    args = parser.parse_args()
    if args.source_run_id != DEFAULT_SOURCE_RUN_ID or args.source_attempt != DEFAULT_SOURCE_ATTEMPT:
        raise ValueError("this correction is frozen to the approved EXP-055 attempt009 TRAIN prediction")
    output = ROOT / "artifacts/experiments/EXP-055" / args.run_id
    if output.exists():
        raise FileExistsError(f"refusing to reopen or overwrite analysis run: {output}")

    started_wall = time.perf_counter()
    started_utc = utc_now()
    attempt_dir = ROOT / "artifacts/experiments/EXP-055" / args.source_run_id / args.source_attempt
    attempt_manifest_path = attempt_dir / "fit_smoke_manifest.json"
    predictions_path = attempt_dir / "full_keyed_predictions.npz"
    manifest_path = attempt_manifest_path
    for path in (attempt_manifest_path, predictions_path):
        if not path.is_file():
            raise FileNotFoundError(f"required attempt009 source is missing: {path}")
    source_hashes = {"attempt_manifest": sha256(manifest_path), "keyed_prediction_archive": sha256(predictions_path)}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("experiment_id") != "EXP-055" or manifest.get("fit_attempt") != "fit_attempt_009"
            or manifest.get("phase") != "EXP055_XPERT_FRESH_ORIGINAL_FULL_2500_OR_PATIENCE50"):
        raise ValueError("manifest does not bind the closed original XPert attempt009")

    keys, truth, prediction, genes = read_train_fields(predictions_path)
    variance = np.var(truth.astype(np.float64), axis=0, ddof=0)
    if not np.any(np.isfinite(variance) & (variance > 1e-12)):
        raise ValueError("TRAIN variance has no positive genes for the frozen metric normalizer")
    baseline = build_baseline(keys, truth)
    groups = build_object_drug_groups(keys, truth, prediction)
    true_resid_groups, pred_resid_groups, group_weights, group_objects, group_drugs = centered_group_matrices(groups)
    condition_weights = source_object_equal_condition_weights(keys)
    alpha, alpha_identifiable, alpha_numerator, alpha_denominator = fit_nonnegative_global_scale(
        truth, prediction, condition_weights, variance
    )
    alpha_mse, alpha_mse_identifiable, alpha_mse_numerator, alpha_mse_denominator = fit_nonnegative_global_mse_scale(
        truth, prediction, condition_weights
    )
    legacy_v2_alpha = fit_legacy_object_drug_mse_scale(groups, group_weights)
    scaled_prediction = alpha * prediction.astype(np.float64)
    mse_scaled_prediction = alpha_mse * prediction.astype(np.float64)
    legacy_v2_scaled_prediction = legacy_v2_alpha * prediction.astype(np.float64)
    nmse_alpha1_float64 = weighted_object_macro_nmse_float64(truth, prediction, keys, variance)
    nmse_alpha_float64 = weighted_object_macro_nmse_float64(truth, scaled_prediction, keys, variance)
    nmse_monotonic_tolerance = max(1e-12, abs(nmse_alpha1_float64) * 1e-12)
    if nmse_alpha_float64 > nmse_alpha1_float64 + nmse_monotonic_tolerance:
        raise AssertionError(
            "float64 constrained NMSE optimum exceeds alpha=1 within the preregistered numerical tolerance: "
            f"{nmse_alpha_float64} > {nmse_alpha1_float64} + {nmse_monotonic_tolerance}"
        )
    mse_alpha1_float64 = weighted_object_macro_mse_float64(truth, prediction, keys)
    mse_alpha_float64 = weighted_object_macro_mse_float64(truth, mse_scaled_prediction, keys)
    legacy_v2_metrics = metrics(keys, truth, legacy_v2_scaled_prediction, variance)
    pred_means = {
        drug: np.stack([row["prediction"] for row in groups if row["full_inchikey"] == drug]).mean(axis=0)
        for drug in sorted(set(group_drugs))
    }
    truth_means = {
        drug: np.stack([row["truth"] for row in groups if row["full_inchikey"] == drug]).mean(axis=0)
        for drug in sorted(set(group_drugs))
    }
    key_drugs = keys.full_inchikey.astype(str).to_numpy()
    true_centered = np.stack([truth[i].astype(np.float64) - truth_means[key_drugs[i]] for i in range(len(keys))])
    pred_centered = np.stack([prediction[i].astype(np.float64) - pred_means[key_drugs[i]] for i in range(len(keys))])
    scaled_centered = np.stack([scaled_prediction[i].astype(np.float64) - alpha * pred_means[key_drugs[i]]
                                for i in range(len(keys))])
    scaled_resid_groups = alpha * pred_resid_groups
    object_drug_counts = {obj: group_objects.count(obj) for obj in set(group_objects)}
    object_equal_group_weights = np.asarray([1.0 / object_drug_counts[obj] for obj in group_objects], dtype=np.float64)
    centered_state_raw = _weighted_residual_metrics(true_resid_groups, pred_resid_groups,
                                                    object_equal_group_weights)
    centered_state_scaled = _weighted_residual_metrics(true_resid_groups, scaled_resid_groups,
                                                       object_equal_group_weights)
    centered_variance = np.var(true_resid_groups.astype(np.float64), axis=0, ddof=0)

    train_metrics = metrics(keys, truth, prediction, variance)
    baseline_metrics = metrics(keys, truth, baseline, variance)
    scaled_metrics = metrics(keys, truth, scaled_prediction, variance)
    raw_relative = 1.0 - float(train_metrics["object_macro_nmse"]) / float(baseline_metrics["object_macro_nmse"])
    scaled_relative = 1.0 - float(scaled_metrics["object_macro_nmse"]) / float(baseline_metrics["object_macro_nmse"])

    per_key, per_object = build_rows(keys, truth, prediction, baseline, scaled_prediction, variance,
                                        centered_variance,
                                        true_centered, pred_centered, scaled_centered)
    per_drug_rows: list[dict[str, Any]] = []
    for drug in sorted(set(key_drugs)):
        group_ix = [i for i, row in enumerate(groups) if row["full_inchikey"] == drug]
        condition_mask = key_drugs == drug
        drug_raw = metrics(keys.loc[condition_mask].reset_index(drop=True), truth[condition_mask],
                           prediction[condition_mask], variance)
        drug_baseline = metrics(keys.loc[condition_mask].reset_index(drop=True), truth[condition_mask],
                                baseline[condition_mask], variance)
        y_mean = truth_means[drug]
        p_mean = pred_means[drug]
        y_norm = float(np.linalg.norm(y_mean))
        p_norm = float(np.linalg.norm(p_mean))
        cosine = float(np.dot(y_mean, p_mean) / (y_norm * p_norm)) if y_norm > 0 and p_norm > 0 else None
        nonzero = np.abs(y_mean) > 1e-12
        y_resid = true_resid_groups[group_ix]
        p_resid = pred_resid_groups[group_ix]
        state = _weighted_residual_metrics(y_resid, p_resid, np.ones(len(group_ix), dtype=np.float64))
        scaled_state = _weighted_residual_metrics(y_resid, alpha * p_resid,
                                                  np.ones(len(group_ix), dtype=np.float64))
        per_drug_rows.append({
            "full_inchikey": drug,
            "conditions": int(condition_mask.sum()),
            "objects": int(keys.loc[condition_mask, "source_object_id"].nunique()),
            "xpert_object_macro_nmse": drug_raw["object_macro_nmse"],
            "per_drug_mean_baseline_object_macro_nmse": drug_baseline["object_macro_nmse"],
            "nmse_difference_vs_per_drug_mean": (float(drug_raw["object_macro_nmse"]) - float(drug_baseline["object_macro_nmse"])),
            "relative_nmse_reduction_vs_per_drug_mean": (1.0 - float(drug_raw["object_macro_nmse"]) / float(drug_baseline["object_macro_nmse"])),
            "xpert_object_macro_mae": drug_raw["object_macro_mae"],
            "per_drug_mean_baseline_object_macro_mae": drug_baseline["object_macro_mae"],
            "xpert_object_macro_gene_correlation": drug_raw["object_macro_gene_correlation"],
            "xpert_object_macro_direction_agreement": drug_raw["object_macro_direction_agreement"],
            "object_equal_true_fixed_mean_rms": _finite_or_none(math.sqrt(float(np.mean(y_mean ** 2)))),
            "object_equal_xpert_fixed_mean_rms": _finite_or_none(math.sqrt(float(np.mean(p_mean ** 2)))),
            "object_equal_fixed_mean_amplitude_ratio": _finite_or_none(math.sqrt(float(np.mean(p_mean ** 2) / np.mean(y_mean ** 2))) if np.mean(y_mean ** 2) > 0 else None),
            "object_equal_fixed_mean_profile_correlation": _correlation(y_mean, p_mean),
            "object_equal_fixed_mean_profile_cosine": _finite_or_none(cosine) if cosine is not None else None,
            "object_equal_fixed_mean_direction_agreement": _finite_or_none(np.mean(np.sign(y_mean[nonzero]) == np.sign(p_mean[nonzero]))) if nonzero.any() else None,
            "object_equal_fixed_mean_error_rms": _finite_or_none(math.sqrt(float(np.mean((p_mean - y_mean) ** 2)))),
            "object_equal_fixed_mean_error_mae": _finite_or_none(float(np.mean(np.abs(p_mean - y_mean)))),
            "drug_centered_state_residual_nmse_vs_zero": state["residual_nmse_vs_zero_state"],
            "drug_centered_state_residual_rmse": state["residual_rmse"],
            "drug_centered_state_residual_correlation": state["residual_correlation"],
            "drug_centered_state_residual_amplitude_ratio": state["residual_amplitude_ratio"],
            "scaled_state_residual_nmse_vs_zero": scaled_state["residual_nmse_vs_zero_state"],
            "scaled_state_residual_rmse": scaled_state["residual_rmse"],
            "scaled_state_residual_correlation": scaled_state["residual_correlation"],
            "scaled_state_residual_amplitude_ratio": scaled_state["residual_amplitude_ratio"],
        })
    per_drug = pd.DataFrame(per_drug_rows)

    best = per_object.sort_values("relative_nmse_reduction_vs_per_drug_mean", ascending=False).head(5)
    worst = per_object.sort_values("relative_nmse_reduction_vs_per_drug_mean", ascending=True).head(5)
    counterexamples = {
        "objects_with_largest_relative_nmse_improvement_vs_per_drug_mean": best.to_dict(orient="records"),
        "objects_with_largest_relative_nmse_worsening_vs_per_drug_mean": worst.to_dict(orient="records"),
    }
    summary = {
        "status": "TRAIN_ONLY_POSTHOC_ERROR_COMPONENTS_COMPLETE",
        "experiment_id": "EXP-055",
        "run_id": args.run_id,
        "source_attempt": f"{args.source_run_id}/{args.source_attempt}",
        "created_at_utc": started_utc,
        "finished_at_utc": utc_now(),
        "interpretation_boundary": "Post-hoc diagnosis of the selected checkpoint's precomputed TRAIN prediction only. The scale coefficient is fitted and evaluated on the same TRAIN objects, so it is an optimistic same-TRAIN upper bound on scalar-only correction; it is not a candidate score, a validation result, or evidence of generalization.",
        "selection_and_data_boundary": {
            "prediction_field": "XPert_best_inner_dev_train_delta",
            "reselection_or_recalibration_performed": False,
            "selected_checkpoint_weights_loaded": False,
            "inner_dev_arrays_indexed_or_read": False,
            "pdo_outer_locked_inputs_opened": False,
            "allowed_arrays_indexed": ["train_condition_key", "train_source_object_id", "train_full_inchikey", "genes", "train_target_delta", "XPert_best_inner_dev_train_delta"],
            "forbidden_roles_opened": [],
        },
        "source_sha256": source_hashes,
        "denominator": {"train_conditions": int(len(keys)), "train_objects": int(keys.source_object_id.nunique()),
                        "full_inchikey_count": int(keys.full_inchikey.nunique()), "genes": int(len(genes)),
                        "condition_key_unique_count": int(keys.full_condition_key.nunique()),
                        "object_drug_groups": int(len(groups)),
                        "gene_variance_zero_count": int(np.count_nonzero(variance <= 1e-12))},
        "metric_definition": {
            "shared_metrics": "Reused metrics() from verify_exp051_low_rank_baselines.py: per-condition NMSE/MAE/correlation/direction, averaged within source_object_id, then averaged equally over 110 objects; gene variance uses only the 419 TRAIN target rows.",
            "per_drug_mean_baseline": "Each drug's 978-gene mean is fitted from TRAIN rows only and applied to those same TRAIN rows; this is a descriptive same-TRAIN baseline.",
            "state_residual": "For each drug, subtract the object-equal mean of object-by-drug TRAIN averages from both observed and predicted profiles; summarize object-by-drug residual vectors with objects equally weighted.",
            "nmse_global_scale": "The nonnegative scalar minimizes the exact existing object-macro NMSE on original TRAIN condition rows: each condition weight is 1/(110 * that object's TRAIN condition count), and each gene error is divided by the frozen 419-row TRAIN variance. Repeated conditions are not averaged before fitting.",
            "mse_global_scale": "A distinct nonnegative scalar minimizes object-macro unnormalized MSE on the same original TRAIN condition rows and object-equal condition weights, without dividing by gene variance. It is not the NMSE-optimal scalar.",
            "v2_correction": "The prior v2 coefficient was fitted on object-by-drug-averaged profiles and raw MSE. v3 retains it only as an audit comparator; corrected coefficients use original condition rows.",
        },
        "selected_xpert_train_metrics": train_metrics,
        "train_fitted_per_drug_mean_metrics": baseline_metrics,
        "scaled_selected_xpert_same_train_metrics": scaled_metrics,
        "relative_nmse_reduction_vs_per_drug_mean": {
            "unscaled_selected_checkpoint": raw_relative,
            "same_train_optimal_nonnegative_scale": scaled_relative,
            "scaled_vs_unscaled_nmse_reduction": 1.0 - float(scaled_metrics["object_macro_nmse"]) / float(train_metrics["object_macro_nmse"]),
            "existing_train_fit_gate_minimum_reduction": 0.10,
            "unscaled_passes_10_percent_gate": raw_relative >= 0.10,
            "scaled_same_train_passes_10_percent_gate": scaled_relative >= 0.10,
        },
        "same_train_scale_correction": {
            "alpha_nmse_condition_level": alpha,
            "alpha_nmse_identifiable": alpha_identifiable,
            "alpha_nmse_numerator": alpha_numerator,
            "alpha_nmse_denominator": alpha_denominator,
            "alpha_raw_mse_condition_level": alpha_mse,
            "alpha_raw_mse_identifiable": alpha_mse_identifiable,
            "alpha_raw_mse_numerator": alpha_mse_numerator,
            "alpha_raw_mse_denominator": alpha_mse_denominator,
            "legacy_v2_alpha_object_drug_raw_mse": legacy_v2_alpha,
            "object_equal_condition_weight_min": float(condition_weights.min()),
            "object_equal_condition_weight_max": float(condition_weights.max()),
            "float64_object_macro_nmse_alpha_1": nmse_alpha1_float64,
            "float64_object_macro_nmse_alpha_nmse": nmse_alpha_float64,
            "float64_nmse_monotonic_tolerance": nmse_monotonic_tolerance,
            "float64_object_macro_raw_mse_alpha_1": mse_alpha1_float64,
            "float64_object_macro_raw_mse_alpha_mse": mse_alpha_float64,
            "legacy_v2_alpha_object_macro_nmse": legacy_v2_metrics["object_macro_nmse"],
            "centered_residual_summary_using_nmse_alpha": {
                "before_scale": centered_state_raw,
                "after_scale": centered_state_scaled,
                "note": "The corrected full-response NMSE alpha is applied here for descriptive decomposition; it was not independently optimized on the centered residuals.",
            },
            "fit_condition_count": int(len(keys)),
            "fit_source_object_count": int(keys.source_object_id.nunique()),
            "fit_full_inchikey_count": int(keys.full_inchikey.nunique()),
            "fit_gene_count": int(len(genes)),
            "scope_warning": "Both coefficients are fitted and scored on the same 419 TRAIN conditions and 110 objects. They are post-hoc TRAIN oracles only, not candidate scores, validation results, or evidence of generalization.",
        },
        "per_drug_results": per_drug_rows,
        "object_counterexamples": counterexamples,
        "answer_to_error_component_question": {
            "amplitude_shrink_observed": float(train_metrics["object_macro_amplitude_ratio"]) < 1.0,
            "unscaled_nmse_reduction_vs_train_mean": raw_relative,
            "scale_only_same_train_nmse_reduction": scaled_relative,
            "drug_centered_state_residual_correlation_before_scale": centered_state_raw["residual_correlation"],
            "drug_centered_state_residual_nmse_before_scale": centered_state_raw["residual_nmse_vs_zero_state"],
            "drug_centered_state_residual_correlation_after_scale": centered_state_scaled["residual_correlation"],
            "drug_centered_state_residual_nmse_after_scale": centered_state_scaled["residual_nmse_vs_zero_state"],
            "interpretation": "Compare the same-TRAIN scaled result against the existing 10% fit gate and the drug-centered residual correlation/NMSE. A scalar can correct magnitude but cannot change the centered residual correlation or its direction pattern; any observed scalar gain remains optimistic TRAIN-only evidence.",
        },
        "crossfit_proposal_status": "NOT_RUN_AND_NOT_AUTHORIZED_IN_THIS_PACKAGE; five-fold scaling of the same already-trained TRAIN objects would not establish end-to-end OOF or unseen-object generalization.",
        "cpu_accounting": {"elapsed_script_wall_seconds": time.perf_counter() - started_wall,
                           "gpu_hours": 0.0},
    }

    output.mkdir(parents=True, exist_ok=False)
    summary_path = output / "train_error_components.json"
    summary = _json_safe(summary)
    write_json(summary_path, summary)
    per_key.to_csv(output / "per_key_metrics.tsv", sep="\t", index=False, na_rep="NA", float_format="%.10g")
    per_object.to_csv(output / "per_object_metrics.tsv", sep="\t", index=False, na_rep="NA", float_format="%.10g")
    per_drug.to_csv(output / "per_drug_metrics.tsv", sep="\t", index=False, na_rep="NA", float_format="%.10g")
    # A self-contained, pickle-free archive with only TRAIN inputs and derived
    # residuals; no inner-dev or other role arrays are copied or opened.
    residual_archive = output / "train_centered_residuals.npz"
    sorted_drugs = sorted(set(group_drugs))
    object_drug_rows = list(zip(group_objects, group_drugs))
    np.savez_compressed(
        residual_archive,
        train_condition_key=np.asarray(keys.full_condition_key.astype(str).tolist(), dtype="U"),
        train_source_object_id=np.asarray(keys.source_object_id.astype(str).tolist(), dtype="U"),
        train_full_inchikey=np.asarray(keys.full_inchikey.astype(str).tolist(), dtype="U"),
        genes=genes,
        train_target_delta=truth,
        XPert_best_inner_dev_train_delta=prediction,
        train_true_drug_centered_delta=true_centered.astype(np.float32),
        train_predicted_drug_centered_delta=pred_centered.astype(np.float32),
        train_globally_scaled_prediction_delta=scaled_prediction,
        train_scaled_drug_centered_prediction_delta=scaled_centered.astype(np.float32),
        full_inchikey_axis=np.asarray(sorted_drugs, dtype="U"),
        object_equal_true_fixed_drug_means=np.stack([truth_means[drug] for drug in sorted_drugs]).astype(np.float32),
        object_equal_xpert_fixed_drug_means=np.stack([pred_means[drug] for drug in sorted_drugs]).astype(np.float32),
        object_drug_source_object_id=np.asarray([obj for obj, _ in object_drug_rows], dtype="U"),
        object_drug_full_inchikey=np.asarray([drug for _, drug in object_drug_rows], dtype="U"),
        object_drug_true_state_residual=true_resid_groups.astype(np.float32),
        object_drug_xpert_state_residual=pred_resid_groups.astype(np.float32),
        object_drug_xpert_scaled_state_residual=scaled_resid_groups.astype(np.float32),
    )
    files = [summary_path, output / "per_key_metrics.tsv", output / "per_object_metrics.tsv",
             output / "per_drug_metrics.tsv", residual_archive]
    summary["cpu_accounting"]["elapsed_script_wall_seconds"] = time.perf_counter() - started_wall
    summary["finished_at_utc"] = utc_now()
    write_json(summary_path, summary)
    manifest_out = {
        "run_id": args.run_id,
        "created_at_utc": started_utc,
        "command": "PYTHONPATH=src conda run --no-capture-output -n drugscreening-gpu python scripts/evaluation/analyze_exp055_xpert_train_error_components.py",
        "python_environment": "WSL2 Conda drugscreening-gpu",
        "gpu_hours": 0.0,
        "source_files_sha256": source_hashes,
        "output_files_sha256": {path.name: sha256(path) for path in files},
        "script_sha256": sha256(Path(__file__).resolve()),
        "attempt009_access_contract": "Only manifest plus six named TRAIN NPZ fields were accessed; all inner-dev arrays and all other data roles were not indexed/read.",
    }
    write_json(output / "manifest.json", manifest_out)
    print(json.dumps({"output_dir": str(output), "status": summary["status"],
                      "train_metrics": train_metrics, "baseline_metrics": baseline_metrics,
                      "same_train_scale": alpha, "scaled_metrics": scaled_metrics,
                      "centered_state_raw": centered_state_raw,
                      "centered_state_scaled": centered_state_scaled,
                      "per_drug": per_drug_rows,
                      "cpu_accounting": summary["cpu_accounting"]}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
