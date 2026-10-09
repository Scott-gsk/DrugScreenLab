"""CPU-only reduced-rank baselines for the frozen EXP-051 TRAIN/inner-dev split."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.modeling import exp051_contract_fit_smoke as fit


def _standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, dtype=np.float64)
    scale = train.std(axis=0, dtype=np.float64)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    return ((train - mean) / scale).astype(np.float64), ((other - mean) / scale).astype(np.float64)


def _fit_rrr(x_train: np.ndarray, y_train: np.ndarray, x_dev: np.ndarray,
             alpha: float, variance_fraction: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Ridge on train-fitted target PCs; coefficient rank is the frozen train-only PCA rank."""
    y_mean = y_train.mean(axis=0, dtype=np.float64)
    y_sd = y_train.std(axis=0, dtype=np.float64)
    y_sd[~np.isfinite(y_sd) | (y_sd <= 1e-12)] = 1.0
    y_scaled = (y_train - y_mean) / y_sd
    pca = PCA(svd_solver="full")
    train_scores = pca.fit_transform(y_scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    rank = int(np.searchsorted(cumulative, variance_fraction, side="left") + 1)
    rank = max(1, min(rank, train_scores.shape[1], x_train.shape[0] - 1))
    scaler_train, scaler_dev = _standardize(x_train, x_dev)
    model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
    model.fit(scaler_train, train_scores[:, :rank])
    score_dev = model.predict(scaler_dev)
    prediction = pca.mean_ + score_dev @ pca.components_[:rank]
    prediction = prediction * y_sd + y_mean
    if not np.isfinite(prediction).all():
        raise FloatingPointError("reduced-rank baseline produced non-finite predictions")
    return prediction.astype(np.float32), {
        "rank": rank, "train_target_pca_cumulative_variance_at_rank": float(cumulative[rank - 1]),
        "train_target_pca_full_rank": int(train_scores.shape[1]),
        "ridge_alpha": float(alpha), "input_standardization": "mean and SD fit on TRAIN only",
        "target_scaling": "per-gene mean and SD fit on TRAIN only; PCA rank is minimum rank reaching the frozen 95% TRAIN variance fraction",
    }


def _object_nmse(keys: pd.DataFrame, truth: np.ndarray, pred: np.ndarray,
                 gene_variance: np.ndarray) -> dict[str, float]:
    use_gene = np.isfinite(gene_variance) & (gene_variance > 1e-12)
    per = np.mean(np.square(pred.astype(np.float64) - truth.astype(np.float64))[:, use_gene]
                  / gene_variance[use_gene], axis=1)
    frame = pd.DataFrame({"object": keys.source_object_id.astype(str), "value": per})
    return {str(k): float(v) for k, v in frame.groupby("object", sort=True).value.mean().items()}


def _paired_bootstrap(reference: dict[str, float], candidate: dict[str, float],
                      replicates: int, seed: int) -> dict[str, Any]:
    ids = sorted(set(reference) & set(candidate))
    if len(ids) != len(reference) or len(ids) != len(candidate) or not ids:
        raise ValueError("paired object denominator differs between low-rank and drug-mean arms")
    diff = np.asarray([reference[key] - candidate[key] for key in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(ids), size=(replicates, len(ids)))
    boot = diff[indices].mean(axis=1)
    return {"paired_objects": len(ids), "reference_minus_candidate_nmse_mean": float(diff.mean()),
            "bootstrap_95pct_ci": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "positive_favors_candidate": True, "unit": "inner_dev source object",
            "claim_boundary": "fixed source-development split only; not patient/PDO population inference"}


def main() -> None:
    started = time.monotonic()
    cfg_path = ROOT / "configs/experiments/EXP-051-low-rank-baselines.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("experiment_id") != "EXP-051":
        raise ValueError("low-rank config must bind to EXP-051")
    fit.configure_source_run(cfg["source_run_id"])
    src = fit.SOURCE_RUN_DIR
    output = ROOT / "artifacts/experiments/EXP-051" / cfg["run_id"]
    output.mkdir(parents=True, exist_ok=False)
    receipts = {"freeze": src / "metadata_freeze_receipt.json", "axis": src / "gene_axis_receipt.json",
                "values": src / "value_contract_receipt.json", "root_readback": src / "root_values_readback_receipt.json",
                "feature": src / "transigen_ecfp4_feature_audit_receipt.json"}
    for name, path in receipts.items():
        if not path.is_file():
            raise FileNotFoundError(f"frozen {name} receipt is missing: {path}")
    freeze = json.loads(receipts["freeze"].read_text(encoding="utf-8"))
    axis = json.loads(receipts["axis"].read_text(encoding="utf-8"))
    value_receipt = json.loads(receipts["values"].read_text(encoding="utf-8"))
    root_readback = json.loads(receipts["root_readback"].read_text(encoding="utf-8"))
    feature_receipt = json.loads(receipts["feature"].read_text(encoding="utf-8"))
    values_path = src / "frozen_train_inner_values.npz"
    if (axis.get("status") != "PASS_EXACT_978_AXIS"
            or root_readback.get("checks", {}).get("result") != "PASS"
            or feature_receipt.get("stage") != "CPU_ONLY_ECFP4_FULL_KEY_ALIAS_REGRESSION_PASS"
            or fit.sha256_file(values_path) != value_receipt.get("files_sha256", {}).get("frozen_train_inner_values_npz")
            or fit.sha256_file(receipts["feature"].parent / "transigen_ecfp4_features.npz")
            != feature_receipt.get("identity_feature_evidence", {}).get("output_npz_sha256")):
        raise RuntimeError("frozen value, exact-axis, independent-readback, or ECFP4 identity receipt failed")
    keys_path = src / "response_blind_conditions.tsv"
    keys = pd.read_csv(keys_path, sep="\t", dtype=str)
    with np.load(values_path, allow_pickle=True) as archive:
        values = {name: archive[name].copy() for name in
                  ("source_object_id", "split_role", "full_inchikey", "genes", "X_ctl", "delta")}
    for column in ("source_object_id", "split_role", "full_inchikey"):
        if not np.array_equal(keys[column].astype(str).to_numpy(), values[column].astype(str)):
            raise ValueError(f"frozen value/key mapping differs for {column}")
    train = keys.split_role.eq("train").to_numpy()
    dev = keys.split_role.eq("inner_dev").to_numpy()
    train_keys, dev_keys = keys.loc[train].reset_index(drop=True), keys.loc[dev].reset_index(drop=True)
    if (int(train.sum()), int(dev.sum()), train_keys.source_object_id.nunique(),
            dev_keys.source_object_id.nunique()) != (419, 93, 110, 24):
        raise ValueError("frozen TRAIN/inner-dev denominator differs from 419/93 keys and 110/24 objects")
    if set(train_keys.source_object_id) & set(dev_keys.source_object_id):
        raise ValueError("source object leaks between TRAIN and inner-development")
    genes = values["genes"].astype(str)
    if len(genes) != 978 or len(set(genes.tolist())) != 978:
        raise ValueError("fixed 978-gene axis is invalid")
    delta = values["delta"].astype(np.float32, copy=False)
    X_ctl = values["X_ctl"].astype(np.float32, copy=False)
    gene_variance = np.var(delta[train].astype(np.float64), axis=0, ddof=0)
    feature_path = receipts["feature"].parent / "transigen_ecfp4_features.npz"
    with np.load(feature_path, allow_pickle=False) as archive:
        feature_keys = archive["full_inchikey"].astype(str)
        feature_matrix = archive["features"].astype(np.float32, copy=True)
    if feature_matrix.shape != (5, 2048) or len(set(feature_keys.tolist())) != 5:
        raise ValueError("audited structure matrix must be exactly 5 full-key ECFP4 rows by 2048 bits")
    feature_by_key = {key: feature_matrix[i] for i, key in enumerate(feature_keys)}
    row_keys = keys.full_inchikey.astype(str).to_numpy()
    missing = sorted(set(row_keys) - set(feature_by_key))
    if missing:
        raise KeyError(f"frozen InChIKey lacks audited structure feature: {missing}")
    structure = np.stack([feature_by_key[key] for key in row_keys]).astype(np.float64)
    x_state_train, x_state_dev = X_ctl[train].astype(np.float64), X_ctl[dev].astype(np.float64)
    x_struct_train, x_struct_dev = structure[train], structure[dev]
    y_train, y_dev = delta[train].astype(np.float64), delta[dev].astype(np.float64)
    alpha = float(cfg["models"]["structural_only"]["ridge_alpha"])
    fraction = float(cfg["models"]["structural_only"]["target_variance_fraction"])
    predictions: dict[str, np.ndarray] = {}
    methods: dict[str, Any] = {}
    for name, xt, xd, model_cfg in (
        ("structural_only", x_struct_train, x_struct_dev, cfg["models"]["structural_only"]),
        ("state_plus_structure", np.concatenate([x_state_train, x_struct_train], axis=1),
         np.concatenate([x_state_dev, x_struct_dev], axis=1), cfg["models"]["state_plus_structure"])):
        pred, method = _fit_rrr(xt, y_train, xd, float(model_cfg["ridge_alpha"]),
                                float(model_cfg["target_variance_fraction"]))
        predictions[name] = pred
        methods[name] = method
    baseline = fit.data_baselines(keys, delta, gene_variance, train, dev, output)
    train_positions = np.flatnonzero(train)
    dev_truth = delta[dev]
    metrics: dict[str, Any] = {}
    # The baseline helper stores its own NPZ; read the fixed per-drug predictions by row index.
    baseline_npz = np.load(output / "baseline_predictions.npz", allow_pickle=False)
    try:
        base_dev_keys = baseline_npz["inner_dev_condition_key"].astype(str)
        if not np.array_equal(base_dev_keys, dev_keys.full_condition_key.astype(str).to_numpy()):
            raise ValueError("baseline prediction keys do not match the fixed inner-development key order")
        per_drug = baseline_npz["per_drug_mean_delta_inner_dev"].astype(np.float32, copy=True)
    finally:
        baseline_npz.close()
    objects_ref = _object_nmse(dev_keys, dev_truth, per_drug, gene_variance)
    train_predictions: dict[str, np.ndarray] = {}
    for name, pred in predictions.items():
        # Refit the same TRAIN-only estimator to generate the TRAIN metric without opening any held-out role.
        xt_train = x_struct_train if name == "structural_only" else np.concatenate([x_state_train, x_struct_train], axis=1)
        xt_dev = x_struct_dev if name == "structural_only" else np.concatenate([x_state_dev, x_struct_dev], axis=1)
        train_pred, train_method = _fit_rrr(xt_train, y_train, xt_train, alpha, fraction)
        train_predictions[name] = train_pred
        metrics[name] = {"train": fit.keyed_metrics(train_keys, delta[train], train_pred, gene_variance),
                         "inner_dev": fit.keyed_metrics(dev_keys, dev_truth, pred, gene_variance),
                         "paired_vs_per_drug_mean": _paired_bootstrap(
                             objects_ref, _object_nmse(dev_keys, dev_truth, pred, gene_variance),
                             int(cfg["metrics"]["bootstrap_replicates"]), int(cfg["metrics"]["seed"])),
                         "estimator": methods[name], "train_prediction_estimator": train_method}
    pred_path = output / "low_rank_predictions.npz"
    np.savez_compressed(pred_path,
                        train_condition_key=train_keys.full_condition_key.astype(str).to_numpy(),
                        inner_dev_condition_key=dev_keys.full_condition_key.astype(str).to_numpy(),
                        train_source_object_id=train_keys.source_object_id.astype(str).to_numpy(),
                        inner_dev_source_object_id=dev_keys.source_object_id.astype(str).to_numpy(),
                        train_full_inchikey=train_keys.full_inchikey.astype(str).to_numpy(),
                        inner_dev_full_inchikey=dev_keys.full_inchikey.astype(str).to_numpy(),
                        genes=genes,
                        train_target_delta=delta[train], inner_dev_target_delta=dev_truth,
                        structural_only_train= train_predictions["structural_only"],
                        state_plus_structure_train=train_predictions["state_plus_structure"],
                        structural_only_inner_dev=predictions["structural_only"],
                        state_plus_structure_inner_dev=predictions["state_plus_structure"])
    result = {
        "experiment_id": "EXP-051", "run_id": cfg["run_id"], "status": "COMPLETED_CPU_ONLY_LOW_RANK_BASELINES",
        "created_at_utc": fit.utc_now(), "config_sha256": fit.sha256_file(cfg_path),
        "runner_sha256": fit.sha256_file(Path(__file__).resolve()),
        "source_freeze_receipts": {name: {"path": str(path), "sha256": fit.sha256_file(path)}
                                   for name, path in receipts.items()},
        "source_files": {"condition_keys_sha256": fit.sha256_file(keys_path),
                         "values_npz_sha256": fit.sha256_file(values_path),
                         "ecfp4_npz_sha256": fit.sha256_file(feature_path)},
        "denominator": {"train_conditions": int(train.sum()), "train_objects": int(train_keys.source_object_id.nunique()),
                        "inner_dev_conditions": int(dev.sum()), "inner_dev_objects": int(dev_keys.source_object_id.nunique()),
                        "object_overlap": 0, "full_inchikeys": int(keys.full_inchikey.nunique()),
                        "genes": int(len(genes)), "normalizer_fit_on": "TRAIN only"},
        "methods": metrics,
        "predictions": {"path": pred_path.name, "sha256": fit.sha256_file(pred_path),
                        "full_keyed": True, "allow_pickle": False},
        "fixed_baseline": {"per_drug_mean_inner_dev": baseline["metrics"]["per_drug_mean_delta"]["inner_dev"],
                           "metrics_path": "baseline_metrics.json",
                           "metrics_sha256": fit.sha256_file(output / "baseline_metrics.json"),
                           "prediction_sha256": fit.sha256_file(output / "baseline_predictions.npz")},
        "forbidden_data_opened": {"PDO_sensitivity": False, "outer_or_locked_response": False,
                                   "external_expression": False},
        "resource": {"gpu_hours": 0, "cpu_seconds": time.monotonic() - started,
                     "cpu_cap_minutes": int(cfg["resources"]["cpu_cap_minutes"])}}
    fit.atomic_json(output / "result.json", result)
    print(json.dumps({"status": result["status"], "run_id": result["run_id"],
                      "denominator": result["denominator"],
                      "inner_dev": {name: value["inner_dev"] for name, value in metrics.items()},
                      "paired_vs_per_drug_mean": {name: value["paired_vs_per_drug_mean"] for name, value in metrics.items()},
                      "gpu_hours": 0, "output": str(output)}, ensure_ascii=False), flush=True)
    if result["resource"]["cpu_seconds"] > int(cfg["resources"]["cpu_cap_minutes"]) * 60:
        raise TimeoutError("low-rank baseline run exceeded its frozen CPU cap")


if __name__ == "__main__":
    main()
