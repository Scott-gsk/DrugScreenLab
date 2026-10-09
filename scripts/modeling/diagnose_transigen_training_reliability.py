#!/usr/bin/env python3
"""EXP-069 TRAIN-only TranSiGen frozen-fit readback and 32-condition fit.

The legacy phase reads only EXP-059's TRAIN endpoint predictions/checkpoints and
the already-authorized 419 TRAIN source rows. The fit32 phase uses the fixed
hash-selected 32-condition TRAIN subset. No inner-dev numeric archive is opened.
Run only in WSL2 Conda ``drugscreening-gpu`` with CUDA hidden.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20261009_transigen_train_reliability_v1"
OUT_ROOT = ROOT / "artifacts/experiments/EXP-069/transigen" / RUN_ID
EXP070_RUN_ID = "20261009_transigen_fixed_updates_v1"
EXP070_ROOT = ROOT / "artifacts/experiments/EXP-070" / EXP070_RUN_ID
EXP070_CONFIG = "configs/experiments/EXP-070-fixed-update-diagnostic.yaml"
EXP070_REFERENCE_ROOT = OUT_ROOT / "train32_attempt_002"
EXP070_REFERENCE_INITIAL = EXP070_REFERENCE_ROOT / "epoch_000_initial_checkpoint.pt"
EXP070_REFERENCE_300 = EXP070_REFERENCE_ROOT / "epoch_300_checkpoint.pt"
EXP070_REFERENCE_300_ENDPOINT = EXP070_REFERENCE_ROOT / "epoch_300" / "train_keyed_predictions.npz"
EXP071_RUN_ID = "20261009_transigen_source_object_dev_v1"
EXP071_CONFIG = "configs/experiments/EXP-071-source-object-development.yaml"
EXP071_CHECKPOINT = ROOT / "artifacts/experiments/EXP-070/20261009_transigen_fixed_updates_v1/fixed2100_attempt_001/epoch_2100_checkpoint.pt"
OLD_RUN = ROOT / "artifacts/experiments/EXP-059/20261008_exp059_transigen_lr1e4_bindingfix_v2/fit_attempt_001"
OLD_SOURCE_RUN = ROOT / "artifacts/experiments/EXP-051/20261006_exp051_contract_fit_smoke_v1"
TRAIN32_DIR_NAME = "train32_attempt_001"
SEED = 20261006
LOSS_NAMES = ("raw_total_loss", "mse_x1", "mse_x2", "mse_pert", "kld_x1", "kld_x2", "kld_pert")
METRIC_NAMES = ("object_macro_nmse", "object_macro_mae", "object_macro_gene_correlation",
                "object_macro_direction_agreement", "object_macro_amplitude_ratio")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_number(value: Any) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return json_number(value)
    if isinstance(value, float):
        return json_number(value)
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {path}")
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(json_safe(value), ensure_ascii=False, separators=(",", ":")) + "\n")


def require_runtime() -> None:
    if sys.platform != "linux" or "microsoft" not in platform.release().lower():
        raise RuntimeError("EXP-069 Python must run inside WSL2 Linux")
    if os.environ.get("CONDA_DEFAULT_ENV") != "drugscreening-gpu":
        raise RuntimeError("EXP-069 Python must use the existing drugscreening-gpu Conda environment")
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise RuntimeError("CPU-only EXP-069 requires CUDA_VISIBLE_DEVICES to be empty")


def load_runner() -> Any:
    path = ROOT / "scripts/modeling/exp051_contract_fit_smoke.py"
    spec = importlib.util.spec_from_file_location("exp051_transigen_reliability_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import the existing EXP-051/055 implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_train_inputs(fit: Any) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    gate = fit._reuse_root_verified_exp055_train_gate()
    keys, values, feature_by_key, evidence = fit._load_exp055_trainpath_train_inputs_reusing_gate(
        gate, OLD_SOURCE_RUN / "response_blind_conditions.tsv", OLD_SOURCE_RUN)
    train_keys = keys.reset_index(drop=True)
    if (len(train_keys) != 419 or train_keys.full_condition_key.astype(str).duplicated().any()
            or train_keys.source_object_id.astype(str).nunique() != 110
            or train_keys.full_inchikey.astype(str).nunique() != 5
            or values["X"].shape != (419, 978) or values["X_ctl"].shape != (419, 978)
            or values["delta"].shape != (419, 978)):
        raise ValueError("reused source input contract did not yield exactly 419 keyed TRAIN rows x 978 genes")
    if not np.array_equal(values["delta"], values["X"] - values["X_ctl"]):
        raise ValueError("TRAIN delta does not equal the original source treatment minus paired control")
    if set(train_keys.full_inchikey.astype(str)) != set(feature_by_key):
        raise ValueError("the five TRAIN chemical identities do not match the audited ECFP4 axis")
    features = np.stack([feature_by_key[str(key)] for key in train_keys.full_inchikey]).astype(np.float32)
    if features.shape != (419, 2048) or not np.isfinite(features).all():
        raise ValueError("keyed TRAIN ECFP4 features are not a finite 419 x 2048 matrix")
    return train_keys, values, feature_by_key, {"source_gate": gate, "train_input_evidence": evidence,
                                                "feature_matrix_sha256": hashlib.sha256(features.tobytes()).hexdigest()}


def array_stats(array: np.ndarray) -> dict[str, Any]:
    value = np.asarray(array, dtype=np.float64)
    return {"shape": list(value.shape), "finite_n": int(np.isfinite(value).sum()),
            "total_n": int(value.size), "min": float(np.min(value)), "max": float(np.max(value)),
            "mean": float(np.mean(value)), "sd_population": float(np.std(value, ddof=0)),
            "mean_absolute": float(np.mean(np.abs(value))),
            "q01": float(np.quantile(value, 0.01)), "q50": float(np.quantile(value, 0.5)),
            "q99": float(np.quantile(value, 0.99))}


def metrics(keys: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray,
            gene_variance: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    variance = np.asarray(gene_variance, dtype=np.float64)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 978:
        raise ValueError("TRAIN target and prediction must share the frozen 978-gene axis")
    use_gene = np.isfinite(variance) & (variance > 1e-12)
    if not use_gene.any():
        raise ValueError("TRAIN-only variance denominator contains no positive-variance gene")
    error = prediction - truth
    row_nmse = np.mean(error[:, use_gene] ** 2 / variance[use_gene][None, :], axis=1)
    row_mae = np.mean(np.abs(error), axis=1)
    row_corr: list[float | None] = []
    row_direction: list[float | None] = []
    for actual, predicted in zip(truth, prediction):
        if np.ptp(actual) <= 0 or np.ptp(predicted) <= 0:
            row_corr.append(None)
        else:
            row_corr.append(json_number(np.corrcoef(actual, predicted)[0, 1]))
        nonzero = np.abs(actual) > 1e-12
        row_direction.append(float(np.mean(np.sign(actual[nonzero]) == np.sign(predicted[nonzero])))
                             if nonzero.any() else None)
    rows = pd.DataFrame({"source_object_id": keys.source_object_id.astype(str).to_numpy(),
                         "full_inchikey": keys.full_inchikey.astype(str).to_numpy(),
                         "nmse": row_nmse, "mae": row_mae,
                         "correlation": row_corr, "direction": row_direction})
    per_object = rows.groupby("source_object_id", sort=True, as_index=False).agg(
        nmse=("nmse", "mean"), mae=("mae", "mean"),
        correlation=("correlation", "mean"), direction=("direction", "mean"))
    amp = []
    for object_id, positions in rows.groupby("source_object_id", sort=True).indices.items():
        ix = np.asarray(positions, dtype=np.int64)
        observed = float(np.mean(np.abs(truth[ix])))
        predicted = float(np.mean(np.abs(prediction[ix])))
        amp.append({"source_object_id": str(object_id), "observed_mean_abs": observed,
                    "predicted_mean_abs": predicted,
                    "ratio": json_number(predicted / observed) if observed > 0 else None})
    amp_values = [entry["ratio"] for entry in amp if entry["ratio"] is not None]
    return {"n_conditions": int(len(rows)), "n_objects": int(rows.source_object_id.nunique()),
            "n_full_inchikeys": int(rows.full_inchikey.nunique()),
            "n_genes_in_nmse_denominator": int(use_gene.sum()),
            "n_zero_variance_genes_excluded": int((~use_gene).sum()),
            "object_macro_nmse": json_number(per_object.nmse.mean()),
            "object_macro_mae": json_number(per_object.mae.mean()),
            "object_macro_gene_correlation": json_number(per_object.correlation.mean()),
            "object_macro_direction_agreement": json_number(per_object.direction.mean()),
            "object_macro_amplitude_ratio": json_number(np.mean(amp_values)) if amp_values else None,
            "object_amplitude": amp,
            "aggregation": "condition-level metrics averaged within object, then objects equally weighted",
            "normalizer": "population variance per gene from all 419 TRAIN delta rows; zero-variance genes excluded and counted"}


def baselines(keys: pd.DataFrame, truth: np.ndarray, gene_variance: np.ndarray,
              fit_scope: str) -> dict[str, Any]:
    truth32 = np.asarray(truth, dtype=np.float32)
    global_mean = truth32.mean(axis=0, dtype=np.float64).astype(np.float32)
    predictions = {"zero_change": np.zeros_like(truth32),
                   "global_mean": np.tile(global_mean, (len(truth32), 1))}
    drug_keys = keys.full_inchikey.astype(str).to_numpy()
    per_drug = np.empty_like(truth32)
    for full_key in sorted(set(drug_keys)):
        positions = np.flatnonzero(drug_keys == full_key)
        per_drug[positions] = truth32[positions].mean(axis=0, dtype=np.float64).astype(np.float32)
    predictions["per_drug_mean"] = per_drug
    return {"fit_scope": fit_scope, "metrics": {
        name: metrics(keys, truth32, pred, gene_variance)
        for name, pred in predictions.items()},
        "predictions": predictions}


def baseline_payload(keys: pd.DataFrame, truth: np.ndarray,
                     gene_variance: np.ndarray, fit_scope: str) -> dict[str, Any]:
    result = baselines(keys, truth, gene_variance, fit_scope)
    result["predictions"] = None
    return result


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {path}")
    temp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


def tensor_state_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8")); digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes()); digest.update(value.tobytes())
    return digest.hexdigest()


def bn_summary(model: Any) -> dict[str, Any]:
    import torch
    buffers: dict[str, Any] = {}
    finite = True
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            item = {}
            for field in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(module, field, None)
                if value is not None:
                    array = value.detach().cpu().numpy()
                    if np.issubdtype(array.dtype, np.floating):
                        finite &= bool(np.isfinite(array).all())
                    item[field] = {"shape": list(array.shape), "min": float(np.min(array)),
                                   "max": float(np.max(array)), "mean": float(np.mean(array))}
                    if field == "running_var":
                        item[field]["nonpositive_n"] = int(np.sum(array <= 0))
            buffers[name] = item
    return {"module_count": len(buffers), "all_finite": finite, "buffers": buffers}


def load_model(fit: Any, torch: Any, state_dict: dict[str, Any] | None = None) -> Any:
    model, _ = fit._load_transigen(torch, torch.device("cpu"), SEED, features_dim=2048, seed_cuda=False)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model


def checkpoint_payload(path: Path, torch: Any) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True)


def nested_exact_equal(torch: Any, left: Any, right: Any) -> bool:
    """Compare optimizer/model payload trees without lossy serialization."""
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right)
                and left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape)
                and bool(torch.equal(left, right)))
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys()
                and all(nested_exact_equal(torch, left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))
                and len(left) == len(right)
                and all(nested_exact_equal(torch, a, b) for a, b in zip(left, right)))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and left.dtype == right.dtype and left.shape == right.shape
                and bool(np.array_equal(left, right)))
    return bool(left == right)


def verify_exp070_anchor(torch: Any, attempt: Path, epoch300_reference: Path,
                         endpoint_reference: Path, reference_curve: Path,
                         new_curve: Path, endpoint_report: dict[str, Any]) -> dict[str, Any]:
    """Require exact model, optimizer, curve, and keyed prediction replay at update 300."""
    new_checkpoint_path = attempt / "epoch_300_checkpoint.pt"
    new_checkpoint = checkpoint_payload(new_checkpoint_path, torch)
    old_checkpoint = checkpoint_payload(epoch300_reference, torch)
    new_state_sha = tensor_state_sha256(new_checkpoint["state_dict"])
    old_state_sha = tensor_state_sha256(old_checkpoint["state_dict"])
    model_state_exact = new_state_sha == old_state_sha
    optimizer_exact = nested_exact_equal(torch, new_checkpoint["optimizer_state_dict"],
                                         old_checkpoint["optimizer_state_dict"])
    new_prediction_path = attempt / "epoch_300" / "train_keyed_predictions.npz"
    with np.load(new_prediction_path, allow_pickle=False) as new, \
            np.load(endpoint_reference, allow_pickle=False) as old:
        prediction_exact = np.array_equal(new["train_eval_mode_prediction_delta"],
                                          old["train_eval_mode_prediction_delta"])
        identity_exact = all(np.array_equal(new[field], old[field]) for field in (
            "train_condition_key", "train_source_object_id", "train_full_inchikey", "train_pert_id", "genes",
            "train_target_delta", "train_control_input", "train_treated_target"))
    new_curve_rows = [json.loads(line) for line in new_curve.read_text(encoding="utf-8").splitlines()
                      if line.strip()]
    old_curve_rows = [json.loads(line) for line in reference_curve.read_text(encoding="utf-8").splitlines()
                      if line.strip()]
    if len(new_curve_rows) < 300 or len(old_curve_rows) < 300:
        raise ValueError("anchor comparison requires the full first 300 training updates")
    curve_loss_exact = new_curve_rows[299]["loss_raw_sum"] == old_curve_rows[299]["loss_raw_sum"]
    checks = {"model_state_sha256_exact": model_state_exact,
              "optimizer_state_exact": optimizer_exact,
              "epoch300_keyed_prediction_exact": bool(prediction_exact),
              "keys_inputs_targets_gene_axis_exact": bool(identity_exact),
              "epoch300_seven_raw_loss_values_exact": bool(curve_loss_exact),
              "epoch300_author_eval_finite": bool(endpoint_report.get("eval_monitor_finite", False)),
              "new_model_state_sha256": new_state_sha, "reference_model_state_sha256": old_state_sha,
              "new_optimizer_checkpoint_sha256": sha256_file(new_checkpoint_path),
              "reference_optimizer_checkpoint_sha256": sha256_file(epoch300_reference),
              "new_endpoint_prediction_sha256": sha256_file(new_prediction_path),
              "reference_endpoint_prediction_sha256": sha256_file(endpoint_reference)}
    checks["all_anchor_checks_pass"] = all(checks[name] for name in (
        "model_state_sha256_exact", "optimizer_state_exact", "epoch300_keyed_prediction_exact",
        "keys_inputs_targets_gene_axis_exact", "epoch300_seven_raw_loss_values_exact",
        "epoch300_author_eval_finite"))
    return checks


def eval_mode_predictions(fit: Any, torch: Any, model: Any, x_ctl: np.ndarray, x: np.ndarray,
                          features: np.ndarray, epoch: int) -> tuple[np.ndarray, dict[str, Any]]:
    # EXP-059's CPU fit loop uses zero-based epoch indexing and seed
    # ``SEED + 2 * epoch_index + 1``. Its fixed epoch-N endpoint therefore
    # binds to ``SEED + 2*N - 1``.
    seed = SEED if int(epoch) == 0 else SEED + 2 * int(epoch) - 1
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        return fit._transigen_predict_and_losses(torch, model, x_ctl, x, features, 64,
                                                  torch.device("cpu"), seed, seed_cuda=False)
    finally:
        random.setstate(python_rng); np.random.set_state(numpy_rng); torch.set_rng_state(torch_rng)


def eval_mode_monitor(fit: Any, torch: Any, model: Any, x_ctl: np.ndarray, x: np.ndarray,
                      features: np.ndarray, epoch: int) -> tuple[np.ndarray, dict[str, Any], dict[str, Any] | None]:
    """Author eval monitor that records nonfinite values as missing and never mutates model state."""
    seed = SEED if int(epoch) == 0 else SEED + 2 * int(epoch) - 1
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    eval_mode_monitor._last_nonfinite = None  # type: ignore[attr-defined]
    before = tensor_state_sha256({name: value.detach().cpu().clone()
                                  for name, value in model.state_dict().items()})
    try:
        prediction, losses = fit._transigen_predict_and_losses(
            torch, model, x_ctl, x, features, 64, torch.device("cpu"), seed, seed_cuda=False,
            allow_nonfinite_auxiliary=True, allow_nonfinite_main=True,
            nonfinite_main_delta_callback=_save_nonfinite_callback)
        finite = bool(np.isfinite(prediction).all())
        diagnostic = getattr(eval_mode_monitor, "_last_nonfinite", None)
        if not finite and diagnostic is None:
            diagnostic = {"reason": "nonfinite_prediction_without_callback",
                          "finite_n": int(np.isfinite(prediction).sum()),
                          "total_n": int(prediction.size), "evaluation_split": "TRAIN_ONLY"}
        after = tensor_state_sha256({name: value.detach().cpu().clone()
                                     for name, value in model.state_dict().items()})
        if before != after:
            raise RuntimeError("author eval monitor mutated model parameters or BatchNorm state")
        if diagnostic is not None:
            diagnostic = {**diagnostic, "eval_seed": seed,
                          "predictions_finite": finite, "losses": losses,
                          "model_and_batchnorm_state_unchanged": True}
        return prediction, losses, diagnostic
    finally:
        eval_mode_monitor._last_nonfinite = None  # type: ignore[attr-defined]
        random.setstate(python_rng); np.random.set_state(numpy_rng); torch.set_rng_state(torch_rng)


def _save_nonfinite_callback(item: dict[str, Any]) -> None:
    eval_mode_monitor._last_nonfinite = dict(item)  # type: ignore[attr-defined]


def train_mode_shadow(fit: Any, torch: Any, model: Any, x_ctl: np.ndarray, x: np.ndarray,
                      features: np.ndarray, epoch: int) -> tuple[np.ndarray, dict[str, Any]]:
    state_before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    state_sha_before = tensor_state_sha256(state_before)
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    seed = SEED + 500_000 + int(epoch)
    try:
        random.seed(seed); np.random.seed(seed % (2**32)); torch.random.default_generator.manual_seed(seed)
        model.train()
        with torch.no_grad():
            x1 = torch.as_tensor(x_ctl, dtype=torch.float32)
            x2 = torch.as_tensor(x, dtype=torch.float32)
            feat = torch.as_tensor(features, dtype=torch.float32)
            _, _, predicted_treated, _ = fit._transigen_forward_batch(
                torch, model, (x1, x2, feat), torch.device("cpu"))
            prediction = (predicted_treated - x1).cpu().numpy().astype(np.float32)
        changed_bn = tensor_state_sha256(model.state_dict()) != state_sha_before
        model.load_state_dict(state_before, strict=True)
        restored = tensor_state_sha256(model.state_dict()) == state_sha_before
        model.eval()
        return prediction, {"seed": seed, "model_mode": "train", "dropout_enabled": True,
                            "batchnorm_uses_current_subset_batch": True,
                            "batchnorm_would_update_without_restore": changed_bn,
                            "model_and_batchnorm_state_restored_after_shadow": restored}
    finally:
        random.setstate(python_rng); np.random.set_state(numpy_rng); torch.set_rng_state(torch_rng)


def metric_rows(keys: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray,
                variance: np.ndarray, label: str) -> list[dict[str, Any]]:
    output = []
    overall = metrics(keys, truth, prediction, variance)
    output.append({"scope": label, "group_type": "all", "group": "ALL", **{
        key: overall[key] for key in METRIC_NAMES}})
    for group_type, column in (("drug", "full_inchikey"), ("object", "source_object_id")):
        groups = sorted(keys[column].astype(str).unique())
        for group in groups:
            mask = keys[column].astype(str).to_numpy() == group
            row = metrics(keys.loc[mask].reset_index(drop=True), truth[mask], prediction[mask], variance)
            output.append({"scope": label, "group_type": group_type, "group": group,
                           "n_conditions": row["n_conditions"], "n_objects": row["n_objects"],
                           **{key: row[key] for key in METRIC_NAMES}})
    return output


def verify_endpoint_archive(path: Path, expected_keys: pd.DataFrame,
                            expected_genes: np.ndarray, expected_delta: np.ndarray) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        required = {"train_condition_key", "train_source_object_id", "train_full_inchikey",
                    "genes", "train_target_delta", "train_delta"}
        if not required.issubset(saved.files):
            raise ValueError(f"legacy endpoint lacks required keyed TRAIN fields: {sorted(required - set(saved.files))}")
        condition_key = saved["train_condition_key"].astype(str)
        objects = saved["train_source_object_id"].astype(str)
        drugs = saved["train_full_inchikey"].astype(str)
        genes = saved["genes"].astype(str)
        target = saved["train_target_delta"].astype(np.float32)
        pred = saved["train_delta"].astype(np.float32)
    expected_condition = expected_keys.full_condition_key.astype(str).to_numpy()
    expected_objects = expected_keys.source_object_id.astype(str).to_numpy()
    expected_drugs = expected_keys.full_inchikey.astype(str).to_numpy()
    correspondence = {"condition_keys_exact_order": bool(np.array_equal(condition_key, expected_condition)),
                      "objects_exact_order": bool(np.array_equal(objects, expected_objects)),
                      "full_inchikeys_exact_order": bool(np.array_equal(drugs, expected_drugs)),
                      "genes_exact_order": bool(np.array_equal(genes, expected_genes)),
                      "target_equals_reloaded_train_delta": bool(np.array_equal(target, expected_delta)),
                      "shape_target": list(target.shape), "shape_prediction": list(pred.shape),
                      "all_prediction_values_finite": bool(np.isfinite(pred).all()),
                      "prediction_sha256": hashlib.sha256(np.ascontiguousarray(pred).tobytes()).hexdigest()}
    if (not all(correspondence[name] for name in ("condition_keys_exact_order", "objects_exact_order",
              "full_inchikeys_exact_order", "genes_exact_order", "target_equals_reloaded_train_delta"))
            or target.shape != (419, 978) or pred.shape != (419, 978)):
        raise ValueError(f"legacy TRAIN prediction/key/target correspondence failed for {path}: {correspondence}")
    return {"prediction": pred, "correspondence": correspondence}


def read_curve(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    epochs = [int(row.get("epoch", -1)) for row in rows]
    if len(rows) != 300 or epochs != list(range(1, 301)):
        raise ValueError("EXP-059 TRAIN curve must contain exactly one record for each epoch 1..300")
    if any("inner_dev_numeric_rows_loaded" in row and row["inner_dev_numeric_rows_loaded"] is not False
           for row in rows):
        raise ValueError("legacy curve explicitly records inner-dev numeric values opened during training")
    if any(any(token in key.lower() for token in ("inner_dev", "outer", "locked", "pdo"))
           for row in rows for key in row):
        raise ValueError("legacy training curve unexpectedly contains a forbidden-role result field")
    return rows


def curve_loss_summary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    compact = []
    for row in rows:
        losses = row.get("train_loss_terms_mean_per_condition", {})
        if set(LOSS_NAMES) - set(losses):
            raise ValueError(f"TRAIN epoch {row['epoch']} lacks one of the seven recorded loss terms")
        values = {name: float(losses[name]) for name in LOSS_NAMES}
        expected_total = values["mse_x1"] + values["mse_x2"] + values["mse_pert"] + 0.1 * (
            values["kld_x1"] + values["kld_x2"] + values["kld_pert"])
        compact.append({"epoch": int(row["epoch"]), **values,
                        "reconstructed_raw_total_loss": expected_total,
                        "raw_total_formula_abs_error": abs(expected_total - values["raw_total_loss"]),
                        "train_nmse": row.get("train_metrics", {}).get("object_macro_nmse"),
                        "train_mae": row.get("train_metrics", {}).get("object_macro_mae"),
                        "train_amplitude_ratio": row.get("train_metrics", {}).get("object_macro_amplitude_ratio"),
                        "gradient_norm_mean": row.get("gradient_norm_mean"),
                        "gradient_norm_max": row.get("gradient_norm_max"),
                        "batchnorm_finite": row.get("batchnorm_finite"),
                        "parameters_finite": row.get("parameters_finite"),
                        "optimizer_finite": row.get("optimizer_finite"),
                        "train_evaluation_main_delta_finite": row.get("train_evaluation_main_delta_finite")})
    selected_epochs = (1, 10, 50, 100, 200, 300)
    selected = [row for row in compact if row["epoch"] in selected_epochs]
    all_formula_errors = [row["raw_total_formula_abs_error"] for row in compact]
    summary = {"records": len(rows), "epochs_with_finite_gradient": sum(
                   int(row.get("gradient_finite", False)) for row in rows),
               "epochs_with_finite_parameters": sum(int(row.get("parameters_finite", False)) for row in rows),
               "epochs_with_finite_optimizer": sum(int(row.get("optimizer_finite", False)) for row in rows),
               "epochs_with_finite_batchnorm": sum(int(row.get("batchnorm_finite", False)) for row in rows),
               "all_recorded_raw_total_losses_reconstruct_from_seven_terms": max(all_formula_errors) < 1e-5,
               "maximum_raw_total_formula_abs_error": max(all_formula_errors),
               "selected_epochs": selected}
    return compact, summary


def select_train32(train_keys: pd.DataFrame) -> pd.DataFrame:
    rows = train_keys.copy()
    rows["full_condition_key"] = rows.full_condition_key.astype(str)
    rows["full_inchikey"] = rows.full_inchikey.astype(str)
    rows["source_object_id"] = rows.source_object_id.astype(str)
    rows["selection_sha256"] = rows.full_condition_key.map(
        lambda key: hashlib.sha256(f"{SEED}|{key}".encode("utf-8")).hexdigest())
    selected: list[pd.DataFrame] = []
    sorted_drugs = sorted(rows.full_inchikey.unique())
    if len(sorted_drugs) != 5:
        raise ValueError("fixed shared subset contract requires the five TRAIN full-InChIKey values")
    for drug_index, drug in enumerate(sorted_drugs):
        group = rows.loc[rows.full_inchikey.eq(drug)].sort_values(
            ["selection_sha256", "full_condition_key"], kind="mergesort")
        take = 7 if drug_index < 2 else 6
        if len(group) < take:
            raise ValueError(f"drug {drug} has only {len(group)} eligible TRAIN rows; {take} required")
        selected.append(group.head(take))
    result = pd.concat(selected, ignore_index=True).sort_values(
        "full_condition_key", kind="mergesort").reset_index(drop=True)
    result.insert(0, "subset_rank", np.arange(1, len(result) + 1, dtype=np.int64))
    if len(result) != 32 or result.full_condition_key.duplicated().any():
        raise ValueError("deterministic EXP-069 TRAIN subset did not produce 32 unique condition keys")
    if result.full_inchikey.value_counts().to_dict() != {drug: (7 if i < 2 else 6)
            for i, drug in enumerate(sorted_drugs)}:
        raise ValueError("fixed 32-key set does not satisfy the frozen 7/7/6/6/6 drug counts")
    return result


def source_summary(fit: Any, values: dict[str, np.ndarray], features: np.ndarray) -> dict[str, Any]:
    model_path = fit.TRANSIGEN_ROOT / "model.py"
    return {"runner": {"path": str(Path(fit.__file__).resolve()),
                        "sha256": sha256_file(Path(fit.__file__).resolve())},
            "upstream_model": {"path": str(model_path), "sha256": sha256_file(model_path)},
            "train_input": {"h5ad_sha256": fit._reuse_root_verified_exp055_train_gate()["source_h5ad_sha256"],
                            "train_conditions": 419, "objects": 110, "drugs": 5, "genes": 978,
                            "control_X_ctl": array_stats(values["X_ctl"]),
                            "treated_X": array_stats(values["X"]), "delta_X_minus_X_ctl": array_stats(values["delta"]),
                            "ecfp4": {"shape": list(features.shape), "finite": bool(np.isfinite(features).all()),
                                      "bit_sum_mean": float(features.sum(axis=1).mean()),
                                      "sha256": hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest()}},
            "input_output_contract": {
                "x1": "the frozen source H5AD paired pre-treatment control, X_ctl, unchanged on its ordered 978-gene axis",
                "x2": "the frozen source H5AD post-treatment expression X, unchanged on the same axis",
                "drug_feature": "full-InChIKey joined audited ECFP4 Morgan radius=2, nBits=2048",
                "target_delta": "X - X_ctl in the source-provided processed scale",
                "model_output": "predicted treated expression x2_pred; reported prediction delta is x2_pred - x1",
                "loss_delta_target": "(x2_pred - x1) compared with (x2 - x1) using summed squared error",
                "extra_scaling_clipping_or_inverse_transform": "none in the frozen source contract",
                "claim_boundary": "current project input contract using upstream TranSiGen architecture; not a full reproduction of the authors' original training dataset"}}


def run_legacy_audit(out_dir: Path) -> dict[str, Any]:
    require_runtime()
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError("CUDA became visible in a CPU-only diagnostic")
    fit = load_runner()
    train_keys, values, feature_by_key, evidence = load_train_inputs(fit)
    features = np.stack([feature_by_key[str(key)] for key in train_keys.full_inchikey]).astype(np.float32)
    variance = np.var(values["delta"].astype(np.float64), axis=0, ddof=0)
    target = values["delta"].astype(np.float32, copy=False)
    selected32 = select_train32(train_keys)
    subset_path = out_dir / "shared_train32_keys.tsv"
    if subset_path.exists():
        prior_keys = pd.read_csv(subset_path, sep="\t", dtype=str)
        if prior_keys.full_condition_key.astype(str).tolist() != selected32.full_condition_key.astype(str).tolist():
            raise ValueError("existing frozen TRAIN32 key list differs from the deterministic selection rule")
    else:
        selected32.to_csv(subset_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)

    metrics_table: list[dict[str, Any]] = []
    endpoint_reports: dict[str, Any] = {}
    for epoch in (100, 300):
        pred_path = OLD_RUN / f"TranSiGen_epoch_{epoch}_train_keyed_predictions.npz"
        checkpoint_path = OLD_RUN / f"TranSiGen_epoch_{epoch}.pt"
        if not pred_path.is_file():
            raise FileNotFoundError(f"EXP-059 epoch-{epoch} TRAIN prediction is missing")
        archive = verify_endpoint_archive(pred_path, train_keys, values["genes"], target)
        prediction = archive["prediction"]
        endpoint_baselines = baselines(train_keys, target, variance, "the same 419 TRAIN targets")
        stored_metrics = metrics(train_keys, target, prediction, variance)
        for name, baseline_pred in endpoint_baselines["predictions"].items():
            metrics_table.extend(metric_rows(train_keys, target, baseline_pred, variance,
                                             f"epoch{epoch}_{name}"))
        metrics_table.extend(metric_rows(train_keys, target, prediction, variance, f"epoch{epoch}_model"))
        endpoint_reports[str(epoch)] = {
            "prediction_path": str(pred_path.relative_to(ROOT)), "prediction_sha256": sha256_file(pred_path),
            "checkpoint_path": str(checkpoint_path.relative_to(ROOT)) if checkpoint_path.is_file() else None,
            "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path.is_file() else None,
            "input_key_target_correspondence": archive["correspondence"],
            "model_metrics": stored_metrics,
            "baseline_metrics": endpoint_baselines["metrics"]}

    metrics_path = out_dir / "legacy_train_endpoint_metrics.tsv"
    pd.DataFrame(metrics_table).to_csv(metrics_path, sep="\t", index=False)
    old_metrics = {"status": "PASS_LEGACY_TRAIN_PREDICTION_KEYED_RECOMPUTATION",
        "source_run": "EXP-059 fixed 300-epoch candidate; TRAIN-only endpoints",
        "metrics_path": str(metrics_path.relative_to(ROOT)), "metrics_sha256": sha256_file(metrics_path),
        "input_key_target_correspondence": {epoch: endpoint_reports[epoch]["input_key_target_correspondence"]
                                             for epoch in endpoint_reports},
        "train_denominator": {"conditions": 419, "objects": 110, "full_inchikeys": 5, "genes": 978,
            "variance_fit_on": "all 419 TRAIN target deltas", "variance_ddof": 0,
            "positive_gene_n": int(np.sum(variance > 1e-12)),
            "zero_or_tiny_gene_n": int(np.sum(variance <= 1e-12))},
        "endpoints": endpoint_reports,
        "claim_boundary": "frozen TRAIN fit diagnostics; do not interpret as breast-cell generalization"}
    atomic_json(out_dir / "old_prediction_metrics.json", old_metrics)

    # Checkpoint replay is a separate verification. It cannot block the direct,
    # already-keyed prediction metrics above.
    replay_report: dict[str, Any] = {}
    for epoch in (100, 300):
        checkpoint_path = OLD_RUN / f"TranSiGen_epoch_{epoch}.pt"
        try:
            payload = checkpoint_payload(checkpoint_path, torch)
            model = load_model(fit, torch, payload.get("state_dict"))
            if payload.get("epoch") != epoch or not bn_summary(model)["all_finite"]:
                raise ValueError("checkpoint metadata or BatchNorm finite-state check failed")
            with np.load(OLD_RUN / f"TranSiGen_epoch_{epoch}_train_keyed_predictions.npz", allow_pickle=False) as saved:
                stored_prediction = saved["train_delta"].astype(np.float32)
            recomputed, eval_losses = eval_mode_predictions(fit, torch, model, values["X_ctl"], values["X"],
                                                            features, epoch)
            max_diff = float(np.max(np.abs(recomputed.astype(np.float64) - stored_prediction.astype(np.float64))))
            replay_report[str(epoch)] = {"status": "PASS_EXACT" if np.allclose(recomputed, stored_prediction, rtol=0, atol=1e-6)
                                              else "REPLAY_DIFFERS_FROM_STORED_PREDICTION",
                "epoch": int(payload["epoch"]), "checkpoint_sha256": sha256_file(checkpoint_path),
                "eval_seed": SEED + 2 * epoch - 1, "max_absolute_prediction_difference": max_diff,
                "author_eval_seven_loss_terms_per_condition": eval_losses,
                "batchnorm_buffers": bn_summary(model)}
        except Exception as exc:
            replay_report[str(epoch)] = {"status": "REPLAY_UNVERIFIED", "exception_type": type(exc).__name__,
                                         "exception_message": str(exc)}
    atomic_json(out_dir / "checkpoint_replay_readback.json", {
        "status": "COMPLETED_SEPARATE_CHECKPOINT_REPLAY",
        "frozen_train_metrics_file": "old_prediction_metrics.json",
        "endpoints": replay_report,
        "data_roles": {"TRAIN_numeric_rows_only": True, "inner_dev_new_values_read": False,
                       "outer_locked_pdo_breast_response_opened": False}})

    curve_rows = read_curve(OLD_RUN / "TranSiGen_training_curve.jsonl")
    compact_curve, curve_summary = curve_loss_summary(curve_rows)

    pd.DataFrame(compact_curve).to_csv(out_dir / "legacy_train_curve_300.tsv", sep="\t", index=False)
    result = {"status": "PASS_LEGACY_EXP059_TRAIN_ONLY_READBACK",
              "run_id": RUN_ID, "source_run": "EXP-059 epoch100/300; TRAIN only",
              "run_directory": str(out_dir), "created_at_unix": time.time(),
              "runtime": {"python": sys.executable, "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
                          "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                          "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
                          "gpu_hours": 0.0},
              "input_evidence": evidence, "source_contract": source_summary(fit, values, features),
              "train_denominator": {"conditions": 419, "objects": 110, "full_inchikeys": 5, "genes": 978,
                                    "condition_key_sha256": sha256_text("\n".join(train_keys.full_condition_key.astype(str)) + "\n"),
                                    "gene_variance": {"fit_scope": "all 419 TRAIN deltas", "ddof": 0,
                                                      "positive_gene_n": int(np.sum(variance > 1e-12)),
                                                      "zero_or_tiny_gene_n": int(np.sum(variance <= 1e-12))}},
              "loss_contract": {"names": list(LOSS_NAMES), "formula": "mse_x1 + mse_x2 + mse_pert + 0.1*(kld_x1 + kld_x2 + kld_pert)",
                                "mse_reduction": "sum over batch and genes", "gradient_clipping": False,
                                "curve_diagnostic": curve_summary},
              "endpoints": endpoint_reports,
              "train32_keys": {"path": str(subset_path.relative_to(ROOT)), "sha256": sha256_file(subset_path),
                               "n": len(selected32), "by_drug": selected32.full_inchikey.value_counts().sort_index().to_dict(),
                               "rule": "per full InChIKey: six lowest SHA256('20261006|'+full_condition_key), plus one for the lexicographically first two drugs; final order by full_condition_key"},
              "data_roles": {"numeric_rows_opened": "419 source TRAIN rows only",
                             "inner_dev_numeric_values_read_by_this_run": False,
                             "outer_locked_pdo_or_breast_threeway_response_opened": False},
              "claim_boundary": "source-side fit diagnostic only; no breast-cell generalization conclusion"}
    atomic_json(out_dir / "legacy_train_readback.json", result)
    return result


def finite_optimizer(torch: Any, optimizer: Any) -> tuple[bool, str | None]:
    for state_index, state in enumerate(optimizer.state.values()):
        for name, value in state.items():
            if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
                return False, f"optimizer_state_{state_index}.{name}"
    return True, None


def state_checkpoint(torch: Any, path: Path, model: Any, optimizer: Any, epoch: int,
                     keys: pd.DataFrame, train32_sha: str, role: str,
                     attempt_id: str = TRAIN32_DIR_NAME,
                     experiment_id: str = "EXP-069", run_id: str = RUN_ID) -> str:
    payload = {"model": "upstream original TranSiGen architecture",
               "experiment_id": experiment_id, "run_id": run_id, "attempt_id": attempt_id,
               "epoch": epoch, "seed": SEED, "state_dict": {k: v.detach().cpu().clone()
                   for k, v in model.state_dict().items()},
               "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
               "training_contract": {"objective": "raw_sum", "learning_rate": 1e-3,
                   "weight_decay": 1e-5, "batch_size": 64, "gradient_clipping": False,
                   "training_rows": len(keys), "train32_key_sha256": train32_sha},
               "checkpoint_role": role}
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)
    return sha256_file(path)


def _tensor_finite_summary(torch: Any, value: Any) -> dict[str, Any]:
    if not torch.is_tensor(value):
        return {"type": type(value).__name__}
    array = value.detach().cpu().numpy()
    finite = np.isfinite(array)
    good = array[finite]
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "finite_n": int(finite.sum()), "nonfinite_n": int(array.size - finite.sum()),
            "min_finite": float(good.min()) if good.size else None,
            "max_finite": float(good.max()) if good.size else None,
            "mean_finite": float(good.mean()) if good.size else None,
            "max_abs_finite": float(np.max(np.abs(good))) if good.size else None}


def _nested_tensor_summaries(torch: Any, value: Any) -> Any:
    if torch.is_tensor(value):
        return _tensor_finite_summary(torch, value)
    if isinstance(value, (tuple, list)):
        return [_nested_tensor_summaries(torch, item) for item in value]
    if isinstance(value, dict):
        return {str(key): _nested_tensor_summaries(torch, item) for key, item in value.items()}
    return {"type": type(value).__name__}


def _trace_model_forward(torch: Any, model: Any, x1: Any, features: Any, seed: int,
                         mode: str, x2: Any | None = None) -> dict[str, Any]:
    """One frozen-state author forward, tracing module and reparameterization values."""
    random.seed(seed); np.random.seed(seed % (2**32)); torch.random.default_generator.manual_seed(seed)
    model.train(mode == "train")
    module_trace: list[dict[str, Any]] = []
    sampler_trace: list[dict[str, Any]] = []
    hooks = []
    for name, module in model.named_modules():
        if name and not any(module.children()):
            def hook(_module: Any, _inputs: Any, output: Any, module_name: str = name) -> None:
                module_trace.append({"module": module_name,
                                     "output": _nested_tensor_summaries(torch, output)})
            hooks.append(module.register_forward_hook(hook))

    def traced_sample(mu: Any, logvar: Any) -> Any:
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(model.dev)
        z = eps.mul_(std).add_(mu)
        sampler_trace.append({"call_index": len(sampler_trace) + 1,
                              "mu": _tensor_finite_summary(torch, mu),
                              "logvar": _tensor_finite_summary(torch, logvar),
                              "exp_half_logvar_std": _tensor_finite_summary(torch, std),
                              "eps": _tensor_finite_summary(torch, eps),
                              "z": _tensor_finite_summary(torch, z)})
        return z

    model.sample_latent = traced_sample
    try:
        with torch.set_grad_enabled(mode == "train"):
            output = model.forward(x1, features)
            loss_values = None
            gradient_summary = None
            if x2 is not None:
                z2, mu2, logvar2 = model.encode_x2(x2)
                x2_rec = model.decode_x2(z2)
                loss_terms = model.loss(x1, output[0], output[1], output[2], x2, x2_rec,
                                        mu2, logvar2, output[3], output[4], output[5])
                loss_values = [float(term.detach().cpu()) for term in loss_terms]
                if mode == "train":
                    loss_terms[0].backward()
                    grad_elements = 0; grad_finite_elements = 0; grad_max_abs = 0.0
                    grad_norm_sq = 0.0; first_bad_gradient = None
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None:
                            grad = parameter.grad.detach()
                            finite = torch.isfinite(grad)
                            grad_elements += int(grad.numel()); grad_finite_elements += int(finite.sum())
                            if finite.any():
                                finite_grad = grad[finite]
                                grad_max_abs = max(grad_max_abs, float(finite_grad.abs().max()))
                                grad_norm_sq += float(torch.sum(finite_grad.double() ** 2))
                            if not finite.all() and first_bad_gradient is None:
                                first_bad_gradient = name
                    gradient_summary = {"elements": grad_elements,
                        "finite_elements": grad_finite_elements,
                        "nonfinite_elements": grad_elements-grad_finite_elements,
                        "max_abs_finite": grad_max_abs, "finite_l2_norm": math.sqrt(grad_norm_sq),
                        "first_nonfinite_parameter": first_bad_gradient}
        output_names = ("x1_reconstruction", "mu1", "logvar1", "x2_pred",
                        "mu_pred", "logvar_pred", "z2_pred")
        outputs = {name: _tensor_finite_summary(torch, tensor)
                   for name, tensor in zip(output_names, output)}
        first_bad_module = next((row for row in module_trace
                                 if _contains_nonfinite(row["output"])), None)
        first_bad_sample = next((row for row in sampler_trace
                                 if any(row[field]["nonfinite_n"]
                                        for field in ("mu", "logvar", "exp_half_logvar_std", "eps", "z"))), None)
        report = {"mode": mode, "seed": seed, "module_trace": module_trace,
                "sampler_trace": sampler_trace, "outputs": outputs,
                "first_nonfinite_module_output": first_bad_module,
                "first_nonfinite_reparameterization": first_bad_sample,
                "main_delta": _tensor_finite_summary(torch, output[3] - x1)}
        if loss_values is not None:
            report["seven_loss_terms"] = dict(zip(LOSS_NAMES, loss_values))
            report["loss_terms_all_finite"] = all(math.isfinite(value) for value in loss_values)
        if gradient_summary is not None:
            report["gradient_summary"] = gradient_summary
        return report
    finally:
        for hook_handle in hooks:
            hook_handle.remove()


def _contains_nonfinite(summary: Any) -> bool:
    if isinstance(summary, dict):
        if "nonfinite_n" in summary and int(summary["nonfinite_n"]) > 0:
            return True
        return any(_contains_nonfinite(value) for value in summary.values())
    if isinstance(summary, list):
        return any(_contains_nonfinite(value) for value in summary)
    return False


def _batchnorm_state_summary(torch: Any, model: Any) -> dict[str, Any]:
    items = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            item = {}
            for field in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(module, field, None)
                if value is not None:
                    array = value.detach().cpu().numpy()
                    finite = np.isfinite(array)
                    good = array[finite]
                    item[field] = {"shape": list(array.shape), "finite_n": int(finite.sum()),
                                   "nonfinite_n": int(array.size - finite.sum()),
                                   "min_finite": float(good.min()) if good.size else None,
                                   "max_finite": float(good.max()) if good.size else None,
                                   "mean_finite": float(good.mean()) if good.size else None,
                                   "num_batches_tracked": int(array.item()) if field == "num_batches_tracked" else None}
            items[name] = item
    return {"module_count": len(items), "all_finite": not _contains_nonfinite(items), "modules": items}


def run_initial_failure_diagnostic(out_dir: Path) -> dict[str, Any]:
    """Diagnose the already saved epoch-0 nonfinite without optimizer updates."""
    require_runtime()
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError("initial failure trace must remain CPU-only")
    fit = load_runner()
    train_keys_all, values, feature_by_key, evidence = load_train_inputs(fit)
    keys = select_train32(train_keys_all).reset_index(drop=True)
    key_path = out_dir / "shared_train32_keys.tsv"
    frozen = pd.read_csv(key_path, sep="\t", dtype=str)
    if frozen.full_condition_key.astype(str).tolist() != keys.full_condition_key.astype(str).tolist():
        raise ValueError("initial-failure trace keys differ from the frozen shared TRAIN32 list")
    lookup = {str(key): i for i, key in enumerate(train_keys_all.full_condition_key.astype(str))}
    positions = np.asarray([lookup[str(key)] for key in keys.full_condition_key], dtype=np.int64)
    x1_np = values["X_ctl"][positions].astype(np.float32, copy=True)
    x2_np = values["X"][positions].astype(np.float32, copy=True)
    delta_np = values["delta"][positions].astype(np.float32, copy=True)
    features_np = np.stack([feature_by_key[str(key)] for key in keys.full_inchikey]).astype(np.float32)
    initial_path = out_dir / TRAIN32_DIR_NAME / "epoch_000_initial_checkpoint.pt"
    checkpoint = checkpoint_payload(initial_path, torch)
    base, _ = fit._load_transigen(torch, torch.device("cpu"), SEED,
                                  features_dim=2048, seed_cuda=False)
    base.load_state_dict(checkpoint["state_dict"], strict=True)
    x1 = torch.as_tensor(x1_np, dtype=torch.float32)
    x2 = torch.as_tensor(x2_np, dtype=torch.float32)
    feat = torch.as_tensor(features_np, dtype=torch.float32)
    input_summary = {"X_ctl": array_stats(x1_np), "X": array_stats(x2_np),
                     "delta": array_stats(delta_np), "ECFP4": array_stats(features_np),
                     "input_all_finite": bool(np.isfinite(x1_np).all() and np.isfinite(x2_np).all()
                                               and np.isfinite(delta_np).all() and np.isfinite(features_np).all()),
                     "feature_unique_values": int(np.unique(features_np).size)}
    eval_model = load_model(fit, torch, checkpoint["state_dict"])
    eval_bn_before = _batchnorm_state_summary(torch, eval_model)
    eval_trace = _trace_model_forward(torch, eval_model, x1, feat, SEED, "eval")
    eval_bn_after = _batchnorm_state_summary(torch, eval_model)
    train_model = load_model(fit, torch, checkpoint["state_dict"])
    train_bn_before = _batchnorm_state_summary(torch, train_model)
    train_model.train()
    random.seed(SEED); np.random.seed(SEED % (2**32)); torch.random.default_generator.manual_seed(SEED)
    train_trace = _trace_model_forward(torch, train_model, x1, feat, SEED, "train", x2=x2)
    train_trace["batchnorm_before"] = train_bn_before
    train_trace["batchnorm_after"] = _batchnorm_state_summary(torch, train_model)
    report = {"status": "INITIAL_MAIN_DELTA_NONFINITE_DIAGNOSED_NO_OPTIMIZER_UPDATE",
              "run_id": RUN_ID, "attempt_id": TRAIN32_DIR_NAME,
              "input_receipt_path": str((out_dir / TRAIN32_DIR_NAME / "input_receipt.json").relative_to(ROOT)),
              "input_receipt_sha256": sha256_file(out_dir / TRAIN32_DIR_NAME / "input_receipt.json"),
              "key_path": str(key_path.relative_to(ROOT)), "key_file_sha256": sha256_file(key_path),
              "condition_key_sha256": sha256_text("\n".join(keys.full_condition_key.astype(str)) + "\n"),
              "checkpoint_path": str(initial_path.relative_to(ROOT)), "checkpoint_sha256": sha256_file(initial_path),
              "checkpoint_role": checkpoint.get("checkpoint_role"),
              "checkpoint_epoch": checkpoint.get("epoch"), "optimizer_updates_before_trace": 0,
              "inner_dev_new_values_read": False, "outer_locked_pdo_breast_response_opened": False,
              "input_summary": input_summary,
              "eval_mode": {"batchnorm_before": eval_bn_before, "batchnorm_after": eval_bn_after,
                            "batchnorm_unchanged": eval_bn_before == eval_bn_after, **eval_trace},
              "train_mode_shadow": train_trace,
              "interpretation": "Eval-mode nonfinite is diagnosed on the frozen fresh initialization. Train-mode is a one-batch shadow on an independent copy with no optimizer update; its BatchNorm changes are not reused.",
              "original_fit_exception": {"type": "FloatingPointError",
                  "message": "non-finite TranSiGen main-delta prediction in TRAIN evaluation batch [0:32] (31296/31296 nonfinite)"}}
    anomaly_path = out_dir / TRAIN32_DIR_NAME / "initial_prediction_failure_diagnostic.json"
    atomic_json(anomaly_path, report)
    outcome_path = out_dir / TRAIN32_DIR_NAME / "run_outcome.json"
    outcome = {"status": "STOPPED_EPOCH0_EVAL_MAIN_PREDICTION_NONFINITE_BEFORE_OPTIMIZER_UPDATE",
               "optimizer_updates_completed": 0, "initial_checkpoint_path": str(initial_path.relative_to(ROOT)),
               "initial_checkpoint_sha256": sha256_file(initial_path),
               "first_anomaly_path": str(anomaly_path.relative_to(ROOT)),
               "first_anomaly_sha256": sha256_file(anomaly_path),
               "initial_fit_exception": report["original_fit_exception"],
               "requested_epochs": 300, "actual_batch_sizes": [32], "gpu_hours": 0.0,
               "inner_dev_new_values_read": False, "outer_locked_pdo_breast_response_opened": False}
    atomic_json(outcome_path, outcome)
    return {"outcome": outcome, "diagnostic": report}


def prediction_endpoints(fit: Any, torch: Any, model: Any, keys: pd.DataFrame,
                         x_ctl: np.ndarray, x: np.ndarray, features: np.ndarray,
                         truth: np.ndarray, genes: np.ndarray, variance: np.ndarray, epoch: int,
                         attempt: Path, fit_baseline: dict[str, Any],
                         experiment_id: str = "EXP-069", run_id: str = RUN_ID) -> dict[str, Any]:
    state_before_monitor = {name: tensor.detach().cpu().clone()
                            for name, tensor in model.state_dict().items()}
    eval_pred, eval_losses, eval_diagnostic = eval_mode_monitor(
        fit, torch, model, x_ctl, x, features, epoch)
    state_after_monitor_sha = tensor_state_sha256(model.state_dict())
    if tensor_state_sha256(state_before_monitor) != state_after_monitor_sha:
        raise RuntimeError("author eval monitor changed parameter or BatchNorm state")
    train_pred, train_mode = train_mode_shadow(fit, torch, model, x_ctl, x, features, epoch)
    model.eval()
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    state_sha_before_save = tensor_state_sha256(state)
    if not train_mode["model_and_batchnorm_state_restored_after_shadow"] or tensor_state_sha256(model.state_dict()) != state_sha_before_save:
        raise RuntimeError("train-mode shadow diagnostic changed the frozen training checkpoint state")
    eval_is_finite = bool(np.isfinite(eval_pred).all())
    eval_metric = metrics(keys, truth, eval_pred, variance) if eval_is_finite else None
    train_metric = metrics(keys, truth, train_pred, variance)
    endpoint_dir = attempt / f"epoch_{epoch:03d}"
    endpoint_dir.mkdir(parents=True, exist_ok=False)
    pred_path = endpoint_dir / "train_keyed_predictions.npz"
    atomic_npz(pred_path, train_condition_key=keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
               train_source_object_id=keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
               train_full_inchikey=keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
               train_pert_id=keys.pert_id.astype(str).to_numpy(dtype=np.str_),
               genes=np.asarray(genes, dtype=np.str_),
               train_target_delta=truth, train_eval_mode_prediction_delta=eval_pred,
               train_mode_shadow_prediction_delta=train_pred, train_control_input=x_ctl,
               train_treated_target=x)
    if eval_diagnostic is not None:
        atomic_json(endpoint_dir / "author_eval_monitor_diagnostic.json", {
            "status": "NONFINITE_EVAL_MONITOR_RECORDED_AS_NULL", "epoch": epoch,
            "eval_metric": None, "diagnostic": eval_diagnostic,
            "model_state_before_and_after_eval_sha256": state_after_monitor_sha,
            "train_mode_shadow_used_for_scoring": False, "inner_dev_new_values_read": False})
    base_metrics = fit_baseline["metrics"]
    strongest_name = min(("zero_change", "global_mean", "per_drug_mean"),
                         key=lambda name: base_metrics[name]["object_macro_nmse"])
    strongest_nmse = float(base_metrics[strongest_name]["object_macro_nmse"])
    model_nmse = eval_metric["object_macro_nmse"] if eval_metric is not None else None
    improvement = ((strongest_nmse - float(model_nmse)) / strongest_nmse
                   if strongest_nmse > 0 and model_nmse is not None else None)
    amplitude_ratio = eval_metric["object_macro_amplitude_ratio"] if eval_metric is not None else None
    gate = {"strongest_baseline": strongest_name, "strongest_baseline_nmse": strongest_nmse,
            "model_nmse": model_nmse,
            "absolute_nmse_reduction": (strongest_nmse - float(model_nmse)) if model_nmse is not None else None,
            "relative_error_reduction": improvement,
            "amplitude_ratio": amplitude_ratio,
            "passes_10pct_and_amplitude_gate": bool(improvement is not None and improvement >= 0.10
                and amplitude_ratio is not None and 0.5 <= amplitude_ratio <= 2.0),
            "status": "NOT_SCOREABLE_EVAL_PREDICTION_NONFINITE" if not eval_is_finite else "SCORED",
            "scope": "training-subset diagnostic only"}
    ckpt = endpoint_dir / "checkpoint.pt"
    torch.save({"model": "upstream original TranSiGen architecture",
                "experiment_id": experiment_id, "run_id": run_id, "epoch": epoch,
                "seed": SEED, "state_dict": state,
                "optimizer_state_dict": None,
                "role": f"fixed_train_only_epoch_{epoch}"}, ckpt)
    return {"experiment_id": experiment_id, "run_id": run_id,
            "epoch": epoch, "eval_seed": SEED if epoch == 0 else SEED + 2 * epoch - 1,
            "eval_mode_metrics": eval_metric, "eval_monitor_finite": eval_is_finite,
            "eval_monitor_diagnostic": eval_diagnostic,
            "train_mode_shadow_metrics": train_metric,
            "train_mode_shadow_protocol": train_mode,
            "author_eval_seven_loss_terms_per_condition": eval_losses,
            "state_sha256_before_and_after_mode_diagnostic": state_sha_before_save,
            "checkpoint_path": str(ckpt.relative_to(ROOT)), "checkpoint_sha256": sha256_file(ckpt),
            "prediction_path": str(pred_path.relative_to(ROOT)), "prediction_sha256": sha256_file(pred_path),
            "fit_gate": gate}


def prepare_exp070_fixed_update_run(torch: Any) -> dict[str, Any]:
    """Read-only contract preflight, then freeze the EXP-070 run's 32-key copy."""
    if EXP070_ROOT.exists():
        raise FileExistsError(f"refusing to reuse an EXP-070 run directory: {EXP070_ROOT}")
    config_path = ROOT / EXP070_CONFIG
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config_text = config_path.read_text(encoding="utf-8")
    required_config_lines = (
        "experiment_id: EXP-070", "seed: 20261006", "device: cpu", "gpu_hours_cap: 0",
        "train_conditions: 32", "genes: 978", "fixed_epochs: 2100", "fixed_optimizer_updates: 2100",
        "batch_size: 64", "learning_rate: 0.001", "weight_decay: 0.00001",
        "loss: original_seven_raw_sum_no_clip", "diagnostic_epochs: [300, 700, 1400]",
        "final_scoring_epoch: 2100", "early_stopping: false")
    missing = [line for line in required_config_lines if line not in config_text]
    if missing:
        raise ValueError(f"EXP-070 config disagrees with the registered fixed-update contract: {missing}")
    reference_result_path = EXP070_REFERENCE_ROOT / "result.json"
    reference_manifest_path = EXP070_REFERENCE_ROOT / "run_manifest.json"
    reference_key_path = OUT_ROOT / "shared_train32_keys.tsv"
    for required in (reference_result_path, reference_manifest_path, reference_key_path,
                     EXP070_REFERENCE_INITIAL, EXP070_REFERENCE_300,
                     EXP070_REFERENCE_300_ENDPOINT, EXP070_REFERENCE_ROOT / "training_curve.jsonl"):
        if not required.is_file():
            raise FileNotFoundError(f"required frozen EXP-069 anchor asset is missing: {required}")
    reference_result = json.loads(reference_result_path.read_text(encoding="utf-8"))
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    if (reference_result.get("status") != "COMPLETED_FIXED_300_EPOCH_TRAIN32_DIAGNOSTIC"
            or reference_result.get("optimizer_updates_completed") != 300
            or reference_result.get("endpoint_reports", {}).get("300", {}).get("eval_monitor_finite") is not True
            or reference_manifest.get("seed") != SEED
            or reference_manifest.get("batch_size") != 64
            or reference_manifest.get("learning_rate") != 1e-3
            or reference_manifest.get("weight_decay") != 1e-5
            or reference_manifest.get("optimizer_updates_completed") != 300):
        raise ValueError("EXP-069 does not contain the expected finite 300-update anchor")
    fit = load_runner()
    source_keys, values, _, evidence = load_train_inputs(fit)
    selected = select_train32(source_keys).reset_index(drop=True)
    frozen_keys = pd.read_csv(reference_key_path, sep="\t", dtype=str)
    if selected.full_condition_key.astype(str).tolist() != frozen_keys.full_condition_key.astype(str).tolist():
        raise ValueError("EXP-069 frozen TRAIN32 keys no longer match the deterministic selection")
    init_ref = checkpoint_payload(EXP070_REFERENCE_INITIAL, torch)
    anchor_ref = checkpoint_payload(EXP070_REFERENCE_300, torch)
    initial_state_sha = tensor_state_sha256(init_ref["state_dict"])
    expected_train_key_sha = sha256_text("\n".join(frozen_keys.full_condition_key.astype(str)) + "\n")
    endpoint = np.load(EXP070_REFERENCE_300_ENDPOINT, allow_pickle=False)
    try:
        endpoint_keys_match = np.array_equal(endpoint["train_condition_key"].astype(str),
                                             frozen_keys.full_condition_key.astype(str).to_numpy())
        endpoint_genes_match = np.array_equal(endpoint["genes"].astype(str), values["genes"].astype(str))
        endpoint_target_shape = endpoint["train_target_delta"].shape == (32, 978)
    finally:
        endpoint.close()
    if not (endpoint_keys_match and endpoint_genes_match and endpoint_target_shape):
        raise ValueError("EXP-069 epoch300 anchor prediction keys/genes/target shape do not match TRAIN32")
    anchor_training_contract = anchor_ref.get("training_contract", {})
    if (anchor_ref.get("epoch") != 300 or anchor_ref.get("seed") != SEED
            or anchor_training_contract.get("learning_rate") != 1e-3
            or anchor_training_contract.get("weight_decay") != 1e-5
            or anchor_training_contract.get("batch_size") != 64
            or anchor_training_contract.get("train32_key_sha256") != expected_train_key_sha):
        raise ValueError("EXP-069 epoch300 checkpoint metadata disagrees with the EXP-070 anchor contract")
    preflight = {"status": "PASS_EXP070_FIXED_UPDATE_PREFLIGHT",
        "experiment_id": "EXP-070", "run_id": EXP070_RUN_ID, "attempt_id": "fixed2100_attempt_001",
        "config_path": EXP070_CONFIG, "config_sha256": sha256_file(config_path),
        "seed": SEED, "fixed_epochs": 2100, "fixed_optimizer_updates": 2100,
        "checkpoint_epochs": [300, 700, 1400, 2100], "batch_size": 64,
        "observed_batch_sizes": [32], "learning_rate": 1e-3, "weight_decay": 1e-5,
        "loss": "original seven raw-sum losses; no clipping", "device": "cpu",
        "source_train_input_evidence": evidence["train_input_evidence"],
        "source_train_conditions": len(source_keys), "fit_conditions": 32,
        "objects": int(selected.source_object_id.nunique()), "drugs": int(selected.full_inchikey.nunique()),
        "genes": int(values["delta"].shape[1]), "train32_keys_path": str(reference_key_path.relative_to(ROOT)),
        "train32_key_sha256": expected_train_key_sha, "initial_state_sha256": initial_state_sha,
        "initial_checkpoint_matches_epoch300_initialization": None,
        "reference_exp069_run_id": RUN_ID, "reference_attempt_id": "train32_attempt_002",
        "reference_300_optimizer_updates": 300,
        "reference_300_eval_finite": True,
        "reference_300_nmse": reference_result["endpoint_reports"]["300"]["eval_mode_metrics"]["object_macro_nmse"],
        "per_drug_mean_baseline_nmse": reference_result["baseline_metrics"]["per_drug_mean"]["object_macro_nmse"],
        "reference_300_checkpoint_sha256": sha256_file(EXP070_REFERENCE_300),
        "reference_300_model_state_sha256": tensor_state_sha256(anchor_ref["state_dict"]),
        "reference_300_endpoint_sha256": sha256_file(EXP070_REFERENCE_300_ENDPOINT),
        "anchor_reproduction_will_be_checked_after_update_300": True,
        "inner_dev_new_values_read": False, "outer_locked_pdo_breast_response_opened": False,
        "gpu_hours": 0.0}
    EXP070_ROOT.parent.mkdir(parents=True, exist_ok=True)
    EXP070_ROOT.mkdir(parents=False, exist_ok=False)
    (EXP070_ROOT / "shared_train32_keys.tsv").write_bytes(reference_key_path.read_bytes())
    preflight["exp070_key_copy_sha256"] = sha256_file(EXP070_ROOT / "shared_train32_keys.tsv")
    atomic_json(EXP070_ROOT / "preflight.json", preflight)
    return preflight


def run_train32(out_dir: Path, attempt_id: str = TRAIN32_DIR_NAME,
                fixed_epochs: int = 300, checkpoint_epochs: tuple[int, ...] = (100, 300),
                experiment_id: str = "EXP-069", run_id: str = RUN_ID,
                config_path: str = "configs/experiments/EXP-069-training-reliability-diagnostics.yaml",
                initial_reference_path: Path | None = None,
                anchor_reference_path: Path | None = None,
                anchor_endpoint_reference: Path | None = None,
                anchor_curve_reference: Path | None = None) -> dict[str, Any]:
    require_runtime()
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError("CUDA became visible in a CPU-only fit")
    fit = load_runner()
    train_keys_all, values, feature_by_key, evidence = load_train_inputs(fit)
    selected = select_train32(train_keys_all)
    subset_path = out_dir / "shared_train32_keys.tsv"
    if not subset_path.is_file():
        raise FileNotFoundError("run the TRAIN-only legacy audit first to freeze the shared 32-key list")
    prior_keys = pd.read_csv(subset_path, sep="\t", dtype=str)
    if prior_keys.full_condition_key.astype(str).tolist() != selected.full_condition_key.astype(str).tolist():
        raise ValueError("shared TRAIN32 key list changed between the readback and fit phases")
    attempt = out_dir / attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    lookup = {str(key): index for index, key in enumerate(train_keys_all.full_condition_key.astype(str))}
    positions = np.asarray([lookup[str(key)] for key in selected.full_condition_key], dtype=np.int64)
    keys = train_keys_all.iloc[positions].reset_index(drop=True)
    x_ctl = values["X_ctl"][positions].astype(np.float32, copy=True)
    x = values["X"][positions].astype(np.float32, copy=True)
    truth = values["delta"][positions].astype(np.float32, copy=True)
    features = np.stack([feature_by_key[str(key)] for key in keys.full_inchikey]).astype(np.float32)
    variance = np.var(values["delta"].astype(np.float64), axis=0, ddof=0)
    train32_sha = sha256_text("\n".join(keys.full_condition_key.astype(str)) + "\n")
    if (truth.shape != (32, 978) or x_ctl.shape != truth.shape or x.shape != truth.shape
            or features.shape != (32, 2048) or not np.isfinite(truth).all()
            or not np.isfinite(x_ctl).all() or not np.isfinite(x).all() or not np.isfinite(features).all()):
        raise ValueError("TRAIN32 selected input arrays do not satisfy finite 32x978 and 32x2048 contracts")
    baselines_fitted_on_subset = baselines(keys, truth, variance, "only the 32 selected TRAIN labels")
    atomic_json(attempt / "train32_baselines.json", baseline_payload(keys, truth, variance,
                                                                      "only the 32 selected TRAIN labels"))
    atomic_json(attempt / "input_receipt.json", {
        "status": "PASS_EXP069_TRANSIGEN_TRAIN32_INPUTS",
        "run_id": RUN_ID, "attempt_id": attempt_id,
        "source_input": evidence["train_input_evidence"],
        "subset_key_path": str(subset_path.relative_to(ROOT)),
        "subset_key_sha256": sha256_file(subset_path), "subset_condition_sha256": train32_sha,
        "condition_keys": keys.full_condition_key.astype(str).tolist(),
        "per_drug_conditions": keys.groupby("full_inchikey").size().to_dict(),
        "per_object_conditions": keys.groupby("source_object_id").size().to_dict(),
        "genes": 978, "feature_dimension": 2048,
        "x1": "original frozen X_ctl rows by condition key",
        "x2": "original frozen X rows by condition key",
        "target": "x2 - x1",
        "feature_identity_join": "audited ECFP4 vector by complete InChIKey",
        "full_train_variance_used_only_as_nmse_denominator": {"fit_rows": 419, "ddof": 0,
            "positive_genes": int(np.sum(variance > 1e-12))},
        "subset_baselines_fit_rows": 32,
        "inner_dev_new_values_read": False, "outer_locked_pdo_breast_response_opened": False,
        "gpu_hours": 0.0})

    random.seed(SEED); np.random.seed(SEED % (2**32)); torch.manual_seed(SEED)
    model, _ = fit._load_transigen(torch, torch.device("cpu"), SEED, features_dim=2048, seed_cuda=False)
    initial_state_sha = tensor_state_sha256(model.state_dict())
    first_attempt_initial = initial_reference_path or (out_dir / TRAIN32_DIR_NAME / "epoch_000_initial_checkpoint.pt")
    if attempt_id != TRAIN32_DIR_NAME:
        parent_initial = checkpoint_payload(first_attempt_initial, torch)
        expected_initial_sha = tensor_state_sha256(parent_initial["state_dict"])
        if initial_state_sha != expected_initial_sha:
            raise RuntimeError("fresh attempt initial state differs byte-for-byte from the sealed attempt001 state")
    else:
        expected_initial_sha = initial_state_sha
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loader_gen = torch.Generator().manual_seed(SEED)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_ctl), torch.from_numpy(x),
                                             torch.from_numpy(features), torch.arange(32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False,
                                         num_workers=0, generator=loader_gen)
    observed_batch_sizes = [int(batch[0].shape[0]) for batch in loader]
    loader_gen.manual_seed(SEED)
    if observed_batch_sizes != [32]:
        raise ValueError("fixed TRAIN32 fit must use a single full 32-condition TRAIN batch each epoch")
    command = ("CUDA_VISIBLE_DEVICES='' PYTHONPATH=src /home/dell/miniconda3/bin/conda run "
               "--no-capture-output -n drugscreening-gpu python "
               "scripts/modeling/diagnose_transigen_training_reliability.py --phase fixed-updates")
    manifest = {"status": f"{experiment_id}_CPU_FIXED_UPDATE_RUNNING",
                "experiment_id": experiment_id, "run_id": run_id, "attempt_id": attempt_id,
                "started_at_unix": time.time(), "pid": os.getpid(), "command": command,
                "python": sys.executable, "torch": torch.__version__,
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "exp051_runner_sha256": sha256_file(Path(fit.__file__).resolve()),
                "upstream_model_sha256": sha256_file(fit.TRANSIGEN_ROOT / "model.py"),
                "device": "cpu", "cuda_visible_devices": "", "gpu_hours": 0.0,
                "seed": SEED, "epochs": fixed_epochs, "fixed_optimizer_updates": fixed_epochs,
                "checkpoint_epochs": [0, *checkpoint_epochs], "learning_rate": 1e-3,
                "weight_decay": 1e-5, "batch_size": 64, "actual_batch_sizes": [32],
                "objective": "original seven-loss raw-sum", "gradient_clipping": False,
                "model_selection": "none; fixed update-count endpoints only", "train32_key_sha256": train32_sha,
                "fresh_initial_state_sha256": initial_state_sha,
                "sealed_attempt001_initial_state_sha256": expected_initial_sha,
                "matches_attempt001_fresh_state": initial_state_sha == expected_initial_sha,
                "config_path": config_path,
                "config_sha256": sha256_file(ROOT / config_path),
                "inner_dev_new_values_read": False}
    atomic_json(attempt / "run_manifest.json", manifest)
    started = time.perf_counter()
    progress_path = attempt / "progress.json"
    curve_path = attempt / "training_curve.jsonl"
    endpoint_reports: dict[str, Any] = {}
    epoch0_checkpoint_sha = state_checkpoint(torch, attempt / "epoch_000_initial_checkpoint.pt",
        model, optimizer, 0, keys, train32_sha, "fresh_initialization_before_training", attempt_id,
        experiment_id, run_id)
    endpoint_reports["0"] = prediction_endpoints(fit, torch, model, keys, x_ctl, x, features, truth,
                                                  values["genes"], variance, 0, attempt,
                                                  baselines_fitted_on_subset, experiment_id, run_id)
    atomic_json(progress_path, {"phase": "TRAIN32_CPU_STARTED", "pid": os.getpid(),
                                "completed_epochs": 0, "target_epochs": fixed_epochs,
                                "elapsed_seconds": time.perf_counter() - started,
                                "gpu_hours": 0.0, "inner_dev_new_values_read": False})
    last_finite_model = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    last_finite_optimizer = copy.deepcopy(optimizer.state_dict())
    last_finite_epoch = 0
    failure: dict[str, Any] | None = None
    try:
        for epoch in range(1, fixed_epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            model.zero_grad(set_to_none=True)
            loss_values: list[float] = []
            gradient_max = 0.0
            epoch_update_count = 0
            for batch_no, batch in enumerate(loader, start=1):
                epoch_update_count += 1
                before_rng = torch.get_rng_state()
                try:
                    x1, x2, predicted_treated, loss_terms = fit._transigen_forward_batch(
                        torch, model, batch[:3], torch.device("cpu"))
                    vals = [float(term.detach().cpu()) for term in loss_terms]
                    if len(vals) != 7 or not all(math.isfinite(value) for value in vals):
                        bad = [LOSS_NAMES[i] for i, value in enumerate(vals) if not math.isfinite(value)]
                        raise FloatingPointError(f"nonfinite raw loss terms: {bad}")
                    if not torch.isfinite((predicted_treated - x1)).all():
                        raise FloatingPointError("nonfinite main TRAIN delta prediction before update")
                    loss_terms[0].backward()
                    gradient_norm_sq = 0.0
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None:
                            if not torch.isfinite(parameter.grad).all():
                                raise FloatingPointError(f"nonfinite gradient in {name}")
                            gradient_norm_sq += float(torch.sum(parameter.grad.detach().double() ** 2))
                    gradient_norm = math.sqrt(gradient_norm_sq)
                    gradient_max = max(gradient_max, gradient_norm)
                    if not math.isfinite(gradient_norm):
                        raise FloatingPointError("nonfinite global gradient norm")
                    optimizer.step()
                    if epoch_update_count != 1:
                        raise RuntimeError("TRAIN32 contract requires exactly one optimizer update per epoch")
                    params_finite = all(torch.isfinite(parameter).all() for parameter in model.parameters())
                    optimizer_finite, optimizer_bad = finite_optimizer(torch, optimizer)
                    bn_after = bn_summary(model)
                    if not params_finite:
                        raise FloatingPointError("nonfinite model parameter after optimizer update")
                    if not optimizer_finite:
                        raise FloatingPointError(f"nonfinite Adam state: {optimizer_bad}")
                    if not bn_after["all_finite"]:
                        raise FloatingPointError("nonfinite BatchNorm running state after update")
                    loss_values = vals
                    append_jsonl(attempt / "training_batches.jsonl", {
                        "epoch": epoch, "batch_index": batch_no, "condition_keys": keys.full_condition_key.astype(str).tolist(),
                        "raw_loss_terms": dict(zip(LOSS_NAMES, vals)), "gradient_global_l2": gradient_norm,
                        "gradient_elements_finite": True, "parameters_finite": True,
                        "optimizer_state_finite": True, "batchnorm_state_finite": True,
                        "batchnorm_num_batches_tracked": bn_after["buffers"].get("encoder_x1.1", {}).get("num_batches_tracked"),
                        "pre_update_rng_state_sha256": hashlib.sha256(before_rng.numpy().tobytes()).hexdigest()})
                except BaseException as exc:
                    failure = {"status": "FAILED_CLOSED_FIRST_NONFINITE_OR_TRAIN_EXCEPTION",
                               "epoch": epoch, "batch_index": batch_no,
                               "exception_type": type(exc).__name__, "exception_message": str(exc),
                               "last_finite_epoch": last_finite_epoch,
                               "last_finite_model_state_sha256": tensor_state_sha256(last_finite_model),
                               "inner_dev_new_values_read": False, "gpu_hours": 0.0}
                    model.load_state_dict(last_finite_model, strict=True)
                    optimizer.load_state_dict(last_finite_optimizer)
                    failure_checkpoint = attempt / "last_finite_checkpoint.pt"
                    torch.save({"epoch": last_finite_epoch, "state_dict": last_finite_model,
                                "optimizer_state_dict": last_finite_optimizer,
                                "seed": SEED, "role": "last_finite_completed_epoch_before_first_failure"},
                               failure_checkpoint)
                    atomic_json(attempt / "first_failure.json", failure)
                    raise
            last_finite_model = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            last_finite_optimizer = copy.deepcopy(optimizer.state_dict())
            last_finite_epoch = epoch
            if epoch_update_count != 1:
                raise RuntimeError(f"epoch {epoch} processed {epoch_update_count} updates, expected one")
            epoch_record = {"epoch": epoch, "optimizer_updates_this_epoch": epoch_update_count,
                            "objective": "raw_sum", "learning_rate": 1e-3,
                            "weight_decay": 1e-5, "batch_sizes": [32],
                            "loss_raw_sum": dict(zip(LOSS_NAMES, loss_values)),
                            "loss_per_condition": {name: value / 32 for name, value in zip(LOSS_NAMES, loss_values)},
                            "gradient_global_l2": gradient_max, "all_gradient_elements_finite": True,
                            "all_parameters_finite": True, "adam_state_finite": True,
                            "batchnorm_state": bn_summary(model),
                            "elapsed_epoch_seconds": time.perf_counter() - epoch_start,
                            "training_conditions": 32, "inner_dev_new_values_read": False}
            append_jsonl(curve_path, epoch_record)
            if epoch in checkpoint_epochs:
                checkpoint_path = attempt / f"epoch_{epoch:03d}_checkpoint.pt"
                checkpoint_sha = state_checkpoint(torch, checkpoint_path, model, optimizer, epoch, keys,
                                                   train32_sha, f"fixed_train_only_epoch_{epoch}", attempt_id,
                                                   experiment_id, run_id)
                endpoint_reports[str(epoch)] = prediction_endpoints(fit, torch, model, keys, x_ctl, x,
                    features, truth, values["genes"], variance, epoch, attempt,
                    baselines_fitted_on_subset, experiment_id, run_id)
                endpoint_reports[str(epoch)]["checkpoint_path"] = str(checkpoint_path.relative_to(ROOT))
                endpoint_reports[str(epoch)]["checkpoint_sha256"] = checkpoint_sha
                if epoch == 300 and anchor_reference_path is not None:
                    if anchor_endpoint_reference is None or anchor_curve_reference is None:
                        raise ValueError("EXP-070 epoch300 anchor requires checkpoint, prediction, and curve references")
                    anchor = verify_exp070_anchor(torch, attempt, anchor_reference_path,
                        anchor_endpoint_reference, anchor_curve_reference, curve_path,
                        endpoint_reports[str(epoch)])
                    anchor_path = attempt / "anchor_reproduction.json"
                    atomic_json(anchor_path, anchor)
                    endpoint_reports[str(epoch)]["anchor_reproduction"] = anchor
                    endpoint_reports[str(epoch)]["anchor_reproduction_path"] = str(anchor_path.relative_to(ROOT))
                    if not anchor["all_anchor_checks_pass"]:
                        failure = {"status": "FAILED_CLOSED_300_UPDATE_ANCHOR_MISMATCH",
                                   "epoch": 300, "last_finite_epoch": last_finite_epoch,
                                   "anchor_reproduction": anchor, "gpu_hours": 0.0,
                                   "inner_dev_new_values_read": False}
                        atomic_json(attempt / "first_failure.json", failure)
                        raise RuntimeError("EXP-070 failed the frozen 300-update exact-reproduction anchor")
            if epoch == 1 or epoch % 10 == 0:
                atomic_json(progress_path.with_name(f"progress_epoch_{epoch:03d}.json"), {
                    "phase": "TRAIN32_CPU_RUNNING", "pid": os.getpid(), "completed_epochs": epoch,
                    "target_epochs": fixed_epochs, "elapsed_seconds": time.perf_counter() - started,
                    "loss": dict(zip(LOSS_NAMES, loss_values)), "gpu_hours": 0.0,
                    "inner_dev_new_values_read": False})
        final_endpoint_key = str(fixed_epochs)
        status = (f"COMPLETED_FIXED_{fixed_epochs}_UPDATES_FINAL_EVAL_NONFINITE"
                  if not endpoint_reports.get(final_endpoint_key, {}).get("eval_monitor_finite", False)
                  else f"COMPLETED_FIXED_{fixed_epochs}_UPDATES_TRAIN32_DIAGNOSTIC")
    except BaseException as exc:
        if failure is None:
            failure = {"status": "FAILED_CLOSED_TRAIN_EXCEPTION", "epoch": last_finite_epoch + 1,
                       "exception_type": type(exc).__name__, "exception_message": str(exc),
                       "last_finite_epoch": last_finite_epoch, "gpu_hours": 0.0,
                       "inner_dev_new_values_read": False}
            atomic_json(attempt / "first_failure.json", failure)
        status = str(failure["status"])
    elapsed = time.perf_counter() - started
    report = {"status": status, "experiment_id": experiment_id, "run_id": run_id, "attempt_id": attempt_id,
              "seed": SEED, "train32_key_sha256": train32_sha,
              "elapsed_cpu_wall_seconds": elapsed, "cpu_process_seconds": time.process_time(),
              "gpu_hours": 0.0, "fixed_epochs": fixed_epochs, "checkpoint_epochs": [0, *checkpoint_epochs],
              "epochs_completed": last_finite_epoch, "optimizer_updates_completed": last_finite_epoch,
              "optimizer_updates_per_epoch": 1,
              "endpoint_reports": endpoint_reports,
              "baseline_metrics": baselines_fitted_on_subset["metrics"],
              "loss_contract": {"names": list(LOSS_NAMES),
                  "formula": "mse_x1 + mse_x2 + mse_pert + 0.1*(kld_x1 + kld_x2 + kld_pert)",
                  "raw_sum": True, "gradient_clipping": False},
              "failure": failure,
              "data_roles": {"training_numeric_rows": 32, "full_train_numeric_rows_for_nmse_variance_only": 419,
                             "inner_dev_new_values_read": False,
                             "outer_locked_pdo_breast_response_opened": False},
              "claim_boundary": ("fixed-update TRAIN-only sufficiency diagnostic; not an official representative-model comparison "
                                  "and says nothing about unseen cells" if experiment_id == "EXP-070"
                                  else "fixed small TRAIN subset fit diagnostic; says nothing about unseen cells")}
    atomic_json(attempt / "result.json", report)
    return report


def readback_exp070_fixed_updates(result: dict[str, Any]) -> dict[str, Any]:
    """Reopen the fixed-update run from disk and recompute TRAIN-only endpoint metrics."""
    fit = load_runner()
    train_keys, values, _, _ = load_train_inputs(fit)
    selected = select_train32(train_keys).reset_index(drop=True)
    frozen = pd.read_csv(EXP070_ROOT / "shared_train32_keys.tsv", sep="\t", dtype=str)
    for field in ("full_condition_key", "source_object_id", "full_inchikey", "pert_id"):
        if selected[field].astype(str).tolist() != frozen[field].astype(str).tolist():
            raise ValueError(f"EXP-070 readback frozen key mismatch: {field}")
    lookup = {str(key): i for i, key in enumerate(train_keys.full_condition_key.astype(str))}
    positions = np.asarray([lookup[str(key)] for key in frozen.full_condition_key], dtype=np.int64)
    truth = values["delta"][positions].astype(np.float32, copy=True)
    x_ctl = values["X_ctl"][positions].astype(np.float32, copy=True)
    x = values["X"][positions].astype(np.float32, copy=True)
    variance = np.var(values["delta"].astype(np.float64), axis=0, ddof=0)
    attempt = EXP070_ROOT / "fixed2100_attempt_001"
    endpoint_checks: dict[str, Any] = {}
    missing_endpoints: list[int] = []
    for epoch in (300, 700, 1400, 2100):
        pred_path = attempt / f"epoch_{epoch:03d}" / "train_keyed_predictions.npz"
        checkpoint_path = attempt / f"epoch_{epoch:03d}_checkpoint.pt"
        if not pred_path.is_file() or not checkpoint_path.is_file():
            missing_endpoints.append(epoch)
            continue
        with np.load(pred_path, allow_pickle=False) as saved:
            pred = saved["train_eval_mode_prediction_delta"].astype(np.float32, copy=True)
            keys_match = np.array_equal(saved["train_condition_key"].astype(str),
                                         frozen.full_condition_key.astype(str).to_numpy())
            objects_match = np.array_equal(saved["train_source_object_id"].astype(str),
                                            frozen.source_object_id.astype(str).to_numpy())
            drugs_match = np.array_equal(saved["train_full_inchikey"].astype(str),
                                          frozen.full_inchikey.astype(str).to_numpy())
            pert_match = np.array_equal(saved["train_pert_id"].astype(str),
                                         frozen.pert_id.astype(str).to_numpy())
            genes_match = np.array_equal(saved["genes"].astype(str), values["genes"].astype(str))
            target_match = np.array_equal(saved["train_target_delta"], truth)
            control_match = np.array_equal(saved["train_control_input"], x_ctl)
            treated_match = np.array_equal(saved["train_treated_target"], x)
        finite = bool(np.isfinite(pred).all())
        report_endpoint = result.get("endpoint_reports", {}).get(str(epoch), {})
        recomputed = metrics(frozen, truth, pred, variance) if finite else None
        stored_metrics = report_endpoint.get("eval_mode_metrics")
        if finite:
            metrics_match = bool(isinstance(stored_metrics, dict) and all(
                np.isclose(float(recomputed[name]), float(stored_metrics[name]), rtol=1e-10, atol=1e-10)
                for name in METRIC_NAMES))
            monitor_matches = report_endpoint.get("eval_monitor_finite") is True
        else:
            metrics_match = stored_metrics is None
            monitor_matches = report_endpoint.get("eval_monitor_finite") is False
        checkpoint = checkpoint_payload(checkpoint_path, __import__("torch"))
        checkpoint_contract = checkpoint.get("training_contract", {})
        endpoint_checks[str(epoch)] = {
            "prediction_shape": list(pred.shape), "prediction_finite": finite,
            "keys_objects_drugs_pert_id_and_gene_axis_match": all((keys_match, objects_match, drugs_match,
                pert_match, genes_match)), "targets_controls_and_treated_inputs_match": all(
                (target_match, control_match, treated_match)), "metrics_recomputed_from_saved_prediction_match": metrics_match,
            "monitor_finiteness_record_matches_saved_values": bool(monitor_matches),
            "recomputed_metrics": recomputed,
            "checkpoint_epoch_matches_path": checkpoint.get("epoch") == epoch,
            "checkpoint_seed_matches": checkpoint.get("seed") == SEED,
            "checkpoint_contract_matches": (checkpoint_contract.get("objective") == "raw_sum"
                and checkpoint_contract.get("learning_rate") == 1e-3
                and checkpoint_contract.get("weight_decay") == 1e-5
                and checkpoint_contract.get("batch_size") == 64
                and checkpoint_contract.get("train32_key_sha256") == result.get("train32_key_sha256")),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "prediction_sha256": sha256_file(pred_path)}
    curve_path = attempt / "training_curve.jsonl"
    batch_path = attempt / "training_batches.jsonl"
    curve = [json.loads(line) for line in curve_path.read_text(encoding="utf-8").splitlines() if line.strip()] if curve_path.is_file() else []
    batches = [json.loads(line) for line in batch_path.read_text(encoding="utf-8").splitlines() if line.strip()] if batch_path.is_file() else []
    curve_epochs = [int(row.get("epoch", -1)) for row in curve]
    batch_epochs = [int(row.get("epoch", -1)) for row in batches]
    loss_formula_errors = []
    all_train_state_finite = True
    for row in curve:
        terms = row.get("loss_raw_sum", {})
        if set(LOSS_NAMES) - set(terms):
            all_train_state_finite = False
            continue
        term_values = [float(terms[name]) for name in LOSS_NAMES]
        expected = term_values[1] + term_values[2] + term_values[3] + 0.1 * sum(term_values[4:7])
        raw_error = abs(expected - term_values[0])
        loss_formula_errors.append(raw_error)
        all_train_state_finite &= (all(math.isfinite(value) for value in term_values)
            and row.get("all_gradient_elements_finite") is True
            and row.get("all_parameters_finite") is True
            and row.get("adam_state_finite") is True
            and row.get("batchnorm_state", {}).get("all_finite") is True)
    baseline_nmse = result.get("baseline_metrics", {}).get("per_drug_mean", {}).get("object_macro_nmse")
    final_metrics = endpoint_checks.get("2100", {}).get("recomputed_metrics")
    final_finite = endpoint_checks.get("2100", {}).get("prediction_finite") is True
    improvement = ((float(baseline_nmse) - float(final_metrics["object_macro_nmse"])) / float(baseline_nmse)
                   if final_finite and baseline_nmse and final_metrics else None)
    amplitude = final_metrics.get("object_macro_amplitude_ratio") if final_metrics else None
    fit_gate = bool(final_finite and improvement is not None and improvement >= 0.10
                    and amplitude is not None and 0.5 <= float(amplitude) <= 2.0)
    anchor = json.loads((attempt / "anchor_reproduction.json").read_text(encoding="utf-8")) \
        if (attempt / "anchor_reproduction.json").is_file() else None
    endpoint_assets_ok = (not missing_endpoints and all(
        all(check.get(key) is True for key in (
            "keys_objects_drugs_pert_id_and_gene_axis_match", "targets_controls_and_treated_inputs_match",
            "metrics_recomputed_from_saved_prediction_match", "checkpoint_epoch_matches_path",
            "monitor_finiteness_record_matches_saved_values", "checkpoint_seed_matches",
            "checkpoint_contract_matches"))
        for check in endpoint_checks.values()))
    training_complete = (result.get("optimizer_updates_completed") == 2100
                         and curve_epochs == list(range(1, 2101))
                         and batch_epochs == list(range(1, 2101)))
    technical_pass = bool(training_complete and all_train_state_finite and endpoint_assets_ok
                          and final_finite and anchor is not None
                          and anchor.get("all_anchor_checks_pass") is True)
    return {"status": "PASS_EXP070_READBACK" if technical_pass else "EXP070_READBACK_INCOMPLETE_OR_FAILED",
        "experiment_id": "EXP-070", "run_id": EXP070_RUN_ID,
        "attempt_id": "fixed2100_attempt_001", "source_scope": "32 frozen TRAIN labels; all 419 TRAIN deltas used only for NMSE variance denominator",
        "rows": 32, "objects": int(selected.source_object_id.nunique()),
        "drugs": int(selected.full_inchikey.nunique()), "genes": 978,
        "train32_key_sha256": sha256_file(EXP070_ROOT / "shared_train32_keys.tsv"),
        "seed": SEED, "fixed_epochs": 2100, "optimizer_updates_completed": result.get("optimizer_updates_completed"),
        "training_curve_records": len(curve), "training_batch_records": len(batches),
        "training_curve_epochs_exact_1_through_2100": curve_epochs == list(range(1, 2101)),
        "exactly_one_batch_update_per_epoch": batch_epochs == list(range(1, 2101)),
        "all_training_losses_gradients_parameters_adam_and_batchnorm_finite": all_train_state_finite,
        "max_seven_loss_formula_abs_residual": max(loss_formula_errors) if loss_formula_errors else None,
        "missing_diagnostic_endpoints": missing_endpoints, "endpoint_readback": endpoint_checks,
        "300_update_exact_anchor": anchor, "per_drug_mean_baseline_nmse": baseline_nmse,
        "final_2100_nmse": final_metrics.get("object_macro_nmse") if final_metrics else None,
        "final_2100_absolute_nmse_reduction": (float(baseline_nmse) - float(final_metrics["object_macro_nmse"])
            if final_metrics and baseline_nmse is not None else None),
        "final_2100_relative_error_reduction": improvement,
        "final_2100_amplitude_ratio": amplitude, "final_2100_eval_all_finite": final_finite,
        "fit_gate_pass": fit_gate,
        "fit_gate_rule": "at least 10% NMSE reduction vs 32-label per-drug mean, amplitude ratio 0.5–2.0, and all-finite final author eval",
        "technical_artifact_checks_pass": technical_pass,
        "inner_dev_new_values_read": False, "outer_locked_pdo_breast_response_opened": False,
        "gpu_hours": 0.0}


def run_exp070_fixed_updates() -> dict[str, Any]:
    require_runtime()
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError("EXP-070 fixed-update fit is CPU-only")
    preflight = prepare_exp070_fixed_update_run(torch)
    result = run_train32(EXP070_ROOT, attempt_id="fixed2100_attempt_001", fixed_epochs=2100,
        checkpoint_epochs=(300, 700, 1400, 2100), experiment_id="EXP-070", run_id=EXP070_RUN_ID,
        config_path=EXP070_CONFIG, initial_reference_path=EXP070_REFERENCE_INITIAL,
        anchor_reference_path=EXP070_REFERENCE_300, anchor_endpoint_reference=EXP070_REFERENCE_300_ENDPOINT,
        anchor_curve_reference=EXP070_REFERENCE_ROOT / "training_curve.jsonl")
    readback = readback_exp070_fixed_updates(result)
    atomic_json(EXP070_ROOT / "readback.json", readback)
    completion = {"status": result["status"], "readback_status": readback["status"],
        "experiment_id": "EXP-070", "run_id": EXP070_RUN_ID,
        "attempt_id": "fixed2100_attempt_001", "fixed_updates": 2100,
        "elapsed_cpu_wall_seconds": result.get("elapsed_cpu_wall_seconds"),
        "cpu_process_seconds": result.get("cpu_process_seconds"), "gpu_hours": 0.0,
        "config_path": EXP070_CONFIG, "config_sha256": preflight["config_sha256"],
        "train32_key_sha256": preflight["train32_key_sha256"],
        "result_sha256": sha256_file(EXP070_ROOT / "fixed2100_attempt_001" / "result.json"),
        "readback_sha256": sha256_file(EXP070_ROOT / "readback.json"),
        "inner_dev_new_values_read": False, "outer_locked_pdo_breast_response_opened": False}
    atomic_json(EXP070_ROOT / "completion.json", completion)
    return {"preflight": preflight, "result": result, "readback": readback, "completion": completion}


def run_sampling_decomposition(out_dir: Path, attempt_id: str = "train32_attempt_002",
                              posthoc_id: str = "posthoc_native_sampling_v1") -> dict[str, Any]:
    """Eight pre-registered author-eval draws from one frozen epoch-300 TRAIN checkpoint."""
    require_runtime()
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError("posthoc sampling decomposition must remain CPU-only")
    started = time.perf_counter()
    attempt = out_dir / attempt_id
    posthoc = out_dir / posthoc_id
    posthoc.mkdir(parents=True, exist_ok=False)
    fit = load_runner()
    train_keys_all, values, feature_by_key, evidence = load_train_inputs(fit)
    keys = select_train32(train_keys_all).reset_index(drop=True)
    key_path = out_dir / "shared_train32_keys.tsv"
    frozen_keys = pd.read_csv(key_path, sep="\t", dtype=str)
    if frozen_keys.full_condition_key.astype(str).tolist() != keys.full_condition_key.astype(str).tolist():
        raise ValueError("posthoc sampling keys differ from the frozen shared TRAIN32 list")
    lookup = {str(key): index for index, key in enumerate(train_keys_all.full_condition_key.astype(str))}
    positions = np.asarray([lookup[str(key)] for key in keys.full_condition_key], dtype=np.int64)
    x_ctl = values["X_ctl"][positions].astype(np.float32, copy=True)
    x = values["X"][positions].astype(np.float32, copy=True)
    truth = values["delta"][positions].astype(np.float32, copy=True)
    features = np.stack([feature_by_key[str(key)] for key in keys.full_inchikey]).astype(np.float32)
    variance = np.var(values["delta"].astype(np.float64), axis=0, ddof=0)
    if (x_ctl.shape != (32, 978) or x.shape != (32, 978) or truth.shape != (32, 978)
            or features.shape != (32, 2048) or not all(np.isfinite(a).all() for a in (x_ctl, x, truth, features))):
        raise ValueError("posthoc TRAIN inputs failed the 32x978 / 32x2048 finite contract")

    endpoint_file = attempt / "epoch_300" / "train_keyed_predictions.npz"
    with np.load(endpoint_file, allow_pickle=False) as saved:
        for field, expected in (("train_condition_key", keys.full_condition_key.astype(str).to_numpy(dtype=np.str_)),
                                ("train_source_object_id", keys.source_object_id.astype(str).to_numpy(dtype=np.str_)),
                                ("train_full_inchikey", keys.full_inchikey.astype(str).to_numpy(dtype=np.str_)),
                                ("train_pert_id", keys.pert_id.astype(str).to_numpy(dtype=np.str_)),
                                ("genes", np.asarray(values["genes"], dtype=np.str_))):
            if not np.array_equal(saved[field], expected):
                raise ValueError(f"frozen epoch300 TRAIN endpoint has a key/axis mismatch: {field}")
        if not (np.array_equal(saved["train_control_input"], x_ctl)
                and np.array_equal(saved["train_treated_target"], x)
                and np.array_equal(saved["train_target_delta"], truth)):
            raise ValueError("frozen epoch300 endpoint inputs/targets differ from the reloaded keyed TRAIN values")
        frozen_epoch300_prediction = saved["train_eval_mode_prediction_delta"].copy()
    result_path = attempt / "result.json"
    fit_result = json.loads(result_path.read_text(encoding="utf-8"))
    epoch300 = fit_result["endpoint_reports"]["300"]
    base_seed = int(epoch300["eval_seed"])
    seeds = list(range(base_seed, base_seed + 8))
    # The result path records the epoch300 checkpoint relative to the repository root.
    checkpoint_path = ROOT / epoch300["checkpoint_path"]
    checkpoint = checkpoint_payload(checkpoint_path, torch)
    model, _ = fit._load_transigen(torch, torch.device("cpu"), SEED, features_dim=2048, seed_cuda=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    state_before_sha = tensor_state_sha256(model.state_dict())
    bn_before = bn_summary(model)
    if not bn_before["all_finite"]:
        raise FloatingPointError("frozen epoch300 checkpoint has nonfinite BatchNorm state")
    protocol = {"status": "PRESPECIFIED_READ_ONLY_EVAL_SAMPLING_DECOMPOSITION",
                "experiment_id": "EXP-069", "run_id": RUN_ID, "attempt_id": attempt_id,
                "posthoc_id": posthoc_id, "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "checkpoint_epoch": checkpoint.get("epoch"), "model_state_sha256": state_before_sha,
                "base_epoch300_eval_seed": base_seed, "seeds": seeds,
                "seed_rule": "the frozen epoch300 evaluation seed plus offsets 0 through 7; no seed selection",
                "condition_count": 32, "object_count": int(keys.source_object_id.nunique()),
                "drug_count": int(keys.full_inchikey.nunique()), "gene_count": 978,
                "condition_key_sha256": sha256_text("\n".join(keys.full_condition_key.astype(str)) + "\n"),
                "key_file_sha256": sha256_file(key_path),
                "epoch300_prediction_sha256": sha256_file(endpoint_file),
                "x1": "frozen paired control X_ctl; unscaled",
                "x2": "frozen treated expression X; same 978-gene axis",
                "true_delta": "X - X_ctl from the frozen source TRAIN rows",
                "predicted_delta": "x2_pred - x1 from unchanged upstream TranSiGen sampler",
                "drug_features": "audited ECFP4 joined by complete InChIKey",
                "nmse_variance_fit_scope": "all 419 TRAIN deltas, population variance, denominator only",
                "eval_mode": "author model.eval(); frozen BatchNorm buffers; original stochastic reparameterization",
                "per_draw_checkpoint_and_batchnorm_mutation_allowed": False,
                "average_output_role": "error decomposition only; not a candidate, score, or replacement for epoch300主评分",
                "seed_selection": "none; all eight prespecified draws retained",
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "config_path": "configs/experiments/EXP-069-training-reliability-diagnostics.yaml",
                "config_sha256": sha256_file(ROOT / "configs/experiments/EXP-069-training-reliability-diagnostics.yaml"),
                "inner_dev_new_values_read": False, "outer_locked_pdo_breast_response_opened": False,
                "gpu_hours": 0.0}
    atomic_json(posthoc / "sampling_protocol.json", protocol)

    x1_t = torch.as_tensor(x_ctl, dtype=torch.float32)
    x2_t = torch.as_tensor(x, dtype=torch.float32)
    feat_t = torch.as_tensor(features, dtype=torch.float32)
    predictions: list[np.ndarray] = []
    predicted_treated: list[np.ndarray] = []
    loss_records: list[dict[str, Any]] = []
    metric_records: list[dict[str, Any]] = []
    identity_records: list[dict[str, Any]] = []
    state_checks: list[dict[str, Any]] = []
    original_forward = model.forward
    capture: list[np.ndarray] = []

    def capture_forward(x1_batch: Any, features_batch: Any) -> Any:
        output = original_forward(x1_batch, features_batch)
        capture.append(output[3].detach().cpu().numpy().astype(np.float32, copy=True))
        return output

    model.forward = capture_forward
    draw_sha: list[str] = []
    try:
        for draw_index, seed in enumerate(seeds):
            capture.clear()
            before_state = tensor_state_sha256(model.state_dict())
            before_bn = bn_summary(model)
            python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
            try:
                pred, losses = fit._transigen_predict_and_losses(
                    torch, model, x_ctl, x, features, 64, torch.device("cpu"), seed, seed_cuda=False)
            finally:
                random.setstate(python_rng); np.random.set_state(numpy_rng); torch.set_rng_state(torch_rng)
            if len(capture) != 1:
                raise RuntimeError(f"seed {seed} produced {len(capture)} forward calls, expected exactly one")
            treated_pred = capture[0]
            if not np.array_equal(pred, treated_pred - x_ctl):
                raise ValueError(f"seed {seed}: helper delta differs from captured x2_pred - x1")
            if not np.isfinite(pred).all() or not np.isfinite(treated_pred).all():
                raise FloatingPointError(f"seed {seed} produced nonfinite author-eval prediction")
            after_state = tensor_state_sha256(model.state_dict())
            after_bn = bn_summary(model)
            if before_state != after_state or after_state != state_before_sha or before_bn != after_bn:
                raise RuntimeError(f"seed {seed} mutated the frozen model or BatchNorm state")
            metric = metrics(keys, truth, pred, variance)
            metric_records.append({"draw": draw_index, "seed": seed,
                "object_macro_nmse": metric["object_macro_nmse"],
                "object_macro_mae": metric["object_macro_mae"],
                "object_macro_gene_correlation": metric["object_macro_gene_correlation"],
                "object_macro_direction_agreement": metric["object_macro_direction_agreement"],
                "object_macro_amplitude_ratio": metric["object_macro_amplitude_ratio"],
                "prediction_mean_abs_all_conditions_genes": float(np.mean(np.abs(pred.astype(np.float64)))),
                "prediction_rms_all_conditions_genes": float(np.sqrt(np.mean(pred.astype(np.float64) ** 2))),
                "prediction_finite_elements": int(pred.size), "state_unchanged": True,
                "batchnorm_finite": after_bn["all_finite"]})
            loss_records.append({"draw": draw_index, "seed": seed, **{
                name: float(value) for name, value in losses.items()}})
            delta_resid = pred.astype(np.float64) - truth.astype(np.float64)
            expression_resid = treated_pred.astype(np.float64) - x.astype(np.float64)
            identity_error = delta_resid - expression_resid
            identity_records.append({"draw": draw_index, "seed": seed,
                "max_abs_identity_difference": float(np.max(np.abs(identity_error))),
                "rmse_identity_difference": float(np.sqrt(np.mean(identity_error ** 2))),
                "elements_over_1e-5": int(np.sum(np.abs(identity_error) > 1e-5)),
                "tolerance": 1e-5, "identity_pass": bool(np.allclose(delta_resid, expression_resid,
                                                                      atol=1e-5, rtol=1e-6))})
            predictions.append(pred.copy())
            predicted_treated.append(treated_pred.copy())
            state_checks.append({"draw": draw_index, "seed": seed,
                "state_before_sha256": before_state, "state_after_sha256": after_state,
                "batchnorm_before_finite": before_bn["all_finite"],
                "batchnorm_after_finite": after_bn["all_finite"],
                "batchnorm_state_unchanged": before_bn == after_bn})
            draw_path = posthoc / f"draw_{draw_index:02d}_seed_{seed}.npz"
            atomic_npz(draw_path,
                draw=np.asarray(draw_index, dtype=np.int16), sampling_seed=np.asarray(seed, dtype=np.int64),
                train_condition_key=keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
                train_source_object_id=keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
                train_full_inchikey=keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
                train_pert_id=keys.pert_id.astype(str).to_numpy(dtype=np.str_),
                genes=np.asarray(values["genes"], dtype=np.str_), train_target_delta=truth,
                train_control_input=x_ctl, train_treated_target=x,
                train_eval_mode_prediction_delta=pred, train_eval_mode_predicted_treated=treated_pred)
            draw_sha.append(sha256_file(draw_path))
    finally:
        model.forward = original_forward

    pred_stack = np.stack(predictions, axis=0).astype(np.float64)
    treated_stack = np.stack(predicted_treated, axis=0).astype(np.float64)
    y = truth.astype(np.float64)
    variance_p = np.var(pred_stack, axis=0, ddof=0)
    mean_p = np.mean(pred_stack, axis=0)
    mean_error_sq = np.mean((pred_stack - y[None, :, :]) ** 2, axis=0)
    bias_sq = (mean_p - y) ** 2
    mse_identity_residual = mean_error_sq - variance_p - bias_sq
    gene_use = variance > 1e-12
    gene_variance = variance[gene_use].astype(np.float64)
    if not gene_use.any():
        raise ValueError("419 TRAIN variance normalizer contains no positive-variance genes")

    def object_macro(matrix: np.ndarray) -> float:
        row_values = np.mean(matrix, axis=1)
        temp = pd.DataFrame({"object": keys.source_object_id.astype(str).to_numpy(), "value": row_values})
        return float(temp.groupby("object", sort=True).value.mean().mean())

    raw_error = mean_error_sq
    raw_sampling = variance_p
    raw_bias = bias_sq
    nmse_error = mean_error_sq[:, gene_use] / gene_variance[None, :]
    nmse_sampling = variance_p[:, gene_use] / gene_variance[None, :]
    nmse_bias = bias_sq[:, gene_use] / gene_variance[None, :]
    per_draw_nmse = np.asarray([record["object_macro_nmse"] for record in metric_records], dtype=np.float64)
    raw_expected = float(np.mean(raw_error))
    raw_sampling_mean = float(np.mean(raw_sampling))
    raw_bias_mean = float(np.mean(raw_bias))
    raw_object_macro = object_macro(raw_error)
    raw_object_sampling = object_macro(raw_sampling)
    raw_object_bias = object_macro(raw_bias)
    nmse_expected = float(np.mean(per_draw_nmse))
    nmse_object_macro = object_macro(nmse_error)
    nmse_sampling_macro = object_macro(nmse_sampling)
    nmse_bias_macro = object_macro(nmse_bias)
    decomposition = {
        "status": "PASS_EXACT_STOCHASTIC_ERROR_DECOMPOSITION",
        "identity": "E_s[(prediction_s - target)^2] = Var_s(prediction_s) + (E_s[prediction_s] - target)^2",
        "seed_count": int(len(seeds)), "seeds": seeds, "ddof": 0,
        "average_output_role": "diagnostic decomposition only; not an evaluated candidate or replacement score",
        "raw_mse_pooled_all_conditions_genes": {
            "expected_per_draw_error": raw_expected, "sampling_variance_component": raw_sampling_mean,
            "average_output_bias_component": raw_bias_mean,
            "sum_components": raw_sampling_mean + raw_bias_mean,
            "identity_abs_residual": abs(raw_expected - raw_sampling_mean - raw_bias_mean),
            "sampling_variance_share": raw_sampling_mean / raw_expected if raw_expected > 0 else None,
            "average_output_bias_share": raw_bias_mean / raw_expected if raw_expected > 0 else None},
        "raw_mse_object_macro": {
            "expected_per_draw_error": raw_object_macro, "sampling_variance_component": raw_object_sampling,
            "average_output_bias_component": raw_object_bias,
            "sum_components": raw_object_sampling + raw_object_bias,
            "identity_abs_residual": abs(raw_object_macro - raw_object_sampling - raw_object_bias),
            "sampling_variance_share": raw_object_sampling / raw_object_macro if raw_object_macro > 0 else None,
            "average_output_bias_share": raw_object_bias / raw_object_macro if raw_object_macro > 0 else None},
        "nmse_object_macro": {
            "mean_of_eight_per_draw_scores": nmse_expected,
            "sampling_variance_component": nmse_sampling_macro,
            "average_output_bias_component": nmse_bias_macro,
            "sum_components": nmse_sampling_macro + nmse_bias_macro,
            "identity_abs_residual": abs(nmse_expected - nmse_sampling_macro - nmse_bias_macro),
            "sampling_variance_share": nmse_sampling_macro / nmse_expected if nmse_expected > 0 else None,
            "average_output_bias_share": nmse_bias_macro / nmse_expected if nmse_expected > 0 else None,
            "relation_to_frozen_per_drug_mean_baseline": {
                "baseline_nmse_32_train_labels": fit_result["baseline_metrics"]["per_drug_mean"]["object_macro_nmse"],
                "mean_draw_nmse_is_diagnostic_only": True}},
        "cellwise_raw_mse_identity_max_abs_residual": float(np.max(np.abs(mse_identity_residual))),
        "cellwise_raw_mse_identity_rmse_residual": float(np.sqrt(np.mean(mse_identity_residual ** 2))),
        "all_delta_vs_expression_residual_identities_pass": all(row["identity_pass"] for row in identity_records),
        "delta_vs_expression_residual_identity_draws": identity_records,
        "draw0_exact_replay_of_frozen_epoch300_prediction": bool(np.array_equal(predictions[0], frozen_epoch300_prediction)),
        "draw0_max_abs_difference_vs_frozen_epoch300_prediction": float(np.max(np.abs(predictions[0] - frozen_epoch300_prediction))),
        "all_eight_seed_draws_kept_without_selection": True,
        "nmse_normalizer": "all 419 TRAIN delta rows; population gene variance; 978 positive-variance genes",
        "object_aggregation": "condition-level gene-mean errors averaged within each source object, then objects equally weighted"}

    loss_frame = pd.DataFrame(loss_records)
    for name in LOSS_NAMES:
        if not np.isfinite(loss_frame[name].to_numpy(dtype=float)).all():
            raise FloatingPointError(f"nonfinite loss {name} across frozen epoch300 sampling draws")
    loss_summary = {name: {"mean_per_condition_across_8_draws": float(loss_frame[name].mean()),
                           "sd_across_8_draws": float(loss_frame[name].std(ddof=0)),
                           "min": float(loss_frame[name].min()), "max": float(loss_frame[name].max())}
                    for name in LOSS_NAMES}
    curve = [json.loads(line) for line in (attempt / "training_curve.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    train_epoch1, train_epoch100, train_epoch300 = curve[0], curve[99], curve[299]
    loss_context = {"training_curve_epochs": {
        str(epoch): {"raw_sum": row["loss_raw_sum"], "per_condition": row["loss_per_condition"],
                    "optimizer_updates": row["optimizer_updates_this_epoch"]}
        for epoch, row in ((1, train_epoch1), (100, train_epoch100), (300, train_epoch300))},
        "author_eval_epoch300_eight_draw_loss_mean_per_condition": loss_summary,
        "objective_formula": "mse_x1 + mse_x2 + mse_pert + 0.1*(kld_x1+kld_x2+kld_pert)",
        "interpretation_boundary": "the raw objective includes control reconstruction, treated-expression reconstruction, target-delta prediction, and three KL regularizers; total-loss reduction alone does not establish improvement over the 32-label per-drug mean baseline"}

    all_draw_path = posthoc / "all_eight_train_keyed_predictions.npz"
    atomic_npz(all_draw_path, draw_index=np.arange(8, dtype=np.int16), sampling_seed=np.asarray(seeds, dtype=np.int64),
        train_condition_key=keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
        train_source_object_id=keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
        train_full_inchikey=keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
        train_pert_id=keys.pert_id.astype(str).to_numpy(dtype=np.str_), genes=np.asarray(values["genes"], dtype=np.str_),
        train_target_delta=truth, train_control_input=x_ctl, train_treated_target=x,
        train_eval_mode_prediction_delta=pred_stack.astype(np.float32),
        train_eval_mode_predicted_treated=treated_stack.astype(np.float32))
    pd.DataFrame(metric_records).to_csv(posthoc / "per_draw_metrics.tsv", sep="\t", index=False)
    loss_frame.to_csv(posthoc / "loss_draws.tsv", sep="\t", index=False)
    atomic_json(posthoc / "error_decomposition.json", decomposition)
    atomic_json(posthoc / "loss_context.json", loss_context)
    state_unchanged = all(row["state_before_sha256"] == row["state_after_sha256"] == state_before_sha
                          and row["batchnorm_state_unchanged"] for row in state_checks)
    final_manifest = {"status": "COMPLETED_READ_ONLY_EIGHT_DRAW_TRAIN_DECOMPOSITION",
        "run_id": RUN_ID, "attempt_id": attempt_id, "posthoc_id": posthoc_id,
        "started_command": "python scripts/modeling/diagnose_transigen_training_reliability.py --phase sampling-decomposition --attempt-id train32_attempt_002 --posthoc-id posthoc_native_sampling_v1",
        "python": sys.executable, "torch": torch.__version__, "device": "cpu",
        "cuda_visible_devices": "", "gpu_hours": 0.0,
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "seed_count": len(seeds), "seeds": seeds, "all_draw_predictions_finite": True,
        "model_and_batchnorm_state_unchanged_every_draw": state_unchanged,
        "draw0_exact_replay": decomposition["draw0_exact_replay_of_frozen_epoch300_prediction"],
        "train32_key_sha256": protocol["condition_key_sha256"],
        "inner_dev_new_values_read": False, "outer_locked_pdo_breast_response_opened": False,
        "elapsed_cpu_wall_seconds": time.perf_counter() - started,
        "draw_file_sha256": draw_sha,
        "outputs": {"all_eight_train_keyed_predictions.npz": sha256_file(all_draw_path),
                    "per_draw_metrics.tsv": sha256_file(posthoc / "per_draw_metrics.tsv"),
                    "loss_draws.tsv": sha256_file(posthoc / "loss_draws.tsv"),
                    "error_decomposition.json": sha256_file(posthoc / "error_decomposition.json"),
                    "loss_context.json": sha256_file(posthoc / "loss_context.json")},
        "claim_boundary": "frozen epoch300 model's author-eval stochastic variability on the same 32 TRAIN rows; no held-out accuracy claim"}
    atomic_json(posthoc / "run_manifest.json", final_manifest)
    return {"protocol": protocol, "per_draw_metrics": metric_records,
            "decomposition": decomposition, "loss_context": loss_context,
            "state_checks": state_checks, "run_manifest": final_manifest}


def _object_metric_table(keys: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray,
                         gene_variance: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    """Score conditions first, then give each source object equal weight."""
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    variance = np.asarray(gene_variance, dtype=np.float64)
    use_gene = np.isfinite(variance) & (variance > 1e-12)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 978:
        raise ValueError("source-object evaluation requires matching 978-gene targets and predictions")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise FloatingPointError("source-object evaluation inputs or predictions contain nonfinite values")
    if not use_gene.any():
        raise ValueError("TRAIN-only primary variance has no positive-variance genes")
    error = prediction - truth
    row_nmse = np.mean(error[:, use_gene] ** 2 / variance[use_gene][None, :], axis=1)
    row_mae = np.mean(np.abs(error), axis=1)
    row_corr: list[float | None] = []
    row_direction: list[float | None] = []
    row_direction_n: list[int] = []
    for actual, predicted in zip(truth, prediction):
        row_corr.append(float(np.corrcoef(actual, predicted)[0, 1])
                         if np.ptp(actual) > 0 and np.ptp(predicted) > 0 else None)
        nonzero = np.abs(actual) > 1e-12
        row_direction_n.append(int(nonzero.sum()))
        row_direction.append(float(np.mean(np.sign(actual[nonzero]) == np.sign(predicted[nonzero])))
                             if nonzero.any() else None)
    rows = pd.DataFrame({
        "source_object_id": keys.source_object_id.astype(str).to_numpy(),
        "full_inchikey": keys.full_inchikey.astype(str).to_numpy(),
        "condition_nmse": row_nmse, "condition_mae": row_mae,
        "condition_gene_correlation": row_corr,
        "condition_direction_agreement": row_direction,
        "condition_nonzero_target_gene_count": row_direction_n,
    })
    object_rows: list[dict[str, Any]] = []
    for object_id, frame in rows.groupby("source_object_id", sort=True):
        row_ix = np.flatnonzero(rows.source_object_id.to_numpy() == object_id)
        actual_abs = float(np.mean(np.abs(truth[row_ix])))
        predicted_abs = float(np.mean(np.abs(prediction[row_ix])))
        corr = frame.condition_gene_correlation.dropna()
        direction = frame.condition_direction_agreement.dropna()
        object_rows.append({
            "source_object_id": str(object_id),
            "full_inchikey": str(frame.full_inchikey.iloc[0]),
            "condition_count": int(len(frame)),
            "nmse": float(frame.condition_nmse.mean()),
            "mae": float(frame.condition_mae.mean()),
            "gene_correlation": float(corr.mean()) if len(corr) else None,
            "correlation_valid_condition_count": int(len(corr)),
            "direction_agreement": float(direction.mean()) if len(direction) else None,
            "direction_valid_condition_count": int(len(direction)),
            "direction_nonzero_target_gene_count": int(frame.condition_nonzero_target_gene_count.sum()),
            "observed_mean_absolute_delta": actual_abs,
            "predicted_mean_absolute_delta": predicted_abs,
            "amplitude_ratio": predicted_abs / actual_abs if actual_abs > 0 else None,
        })
    objects = pd.DataFrame(object_rows)
    amp = objects.amplitude_ratio.dropna()
    corr = objects.gene_correlation.dropna()
    direction = objects.direction_agreement.dropna()
    summary = {
        "n_conditions": int(len(keys)), "n_objects": int(len(objects)),
        "n_drugs": int(keys.full_inchikey.astype(str).nunique()),
        "n_genes_total": int(truth.shape[1]), "n_genes_in_nmse_denominator": int(use_gene.sum()),
        "n_zero_variance_genes_excluded": int((~use_gene).sum()),
        "n_nmse_valid_conditions": int(np.isfinite(row_nmse).sum()),
        "n_mae_valid_conditions": int(np.isfinite(row_mae).sum()),
        "n_direction_valid_conditions": int(pd.Series(row_direction).notna().sum()),
        "n_correlation_valid_conditions": int(pd.Series(row_corr).notna().sum()),
        "n_amplitude_valid_objects": int(len(amp)),
        "object_macro_nmse": float(objects.nmse.mean()),
        "object_macro_mae": float(objects.mae.mean()),
        "object_macro_gene_correlation": float(corr.mean()) if len(corr) else None,
        "object_macro_direction_agreement": float(direction.mean()) if len(direction) else None,
        "object_macro_amplitude_ratio": float(amp.mean()) if len(amp) else None,
        "aggregation": "per-condition metrics, then conditions equally averaged within each source object, then source objects equally averaged",
        "normalizer": "population variance per gene estimated only from the specified TRAIN fit rows; genes with variance <= 1e-12 excluded",
    }
    return summary, objects


def _bootstrap_object_nmse(model_objects: pd.DataFrame, baseline_objects: dict[str, pd.DataFrame],
                           seed: int, replicates: int) -> dict[str, Any]:
    object_ids = model_objects.source_object_id.astype(str).tolist()
    for name, frame in baseline_objects.items():
        if frame.source_object_id.astype(str).tolist() != object_ids:
            raise ValueError(f"paired bootstrap object order mismatch for {name}")
    model_nmse = model_objects.nmse.to_numpy(dtype=np.float64)
    baseline_names = list(baseline_objects)
    baseline_nmse = np.stack([baseline_objects[name].nmse.to_numpy(dtype=np.float64)
                              for name in baseline_names])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(object_ids), size=(replicates, len(object_ids)), dtype=np.int16)
    model_draws = model_nmse[indices].mean(axis=1)
    baseline_draws = baseline_nmse[:, indices].mean(axis=2)
    baseline_gains = {name: baseline_draws[i] - model_draws
                      for i, name in enumerate(baseline_names)}
    best_idx = np.argmin(baseline_draws, axis=0)
    strongest_gain = baseline_draws[best_idx, np.arange(replicates)] - model_draws

    def ci(values: np.ndarray) -> dict[str, float]:
        return {"mean": float(values.mean()), "ci95_percentile_lower": float(np.quantile(values, 0.025)),
                "ci95_percentile_upper": float(np.quantile(values, 0.975))}

    return {
        "replicates": replicates, "seed": seed, "resampling_unit": "source_object_id",
        "paired": True, "object_count": len(object_ids),
        "fixed_baseline_absolute_nmse_improvement": {name: ci(values) for name, values in baseline_gains.items()},
        "best_fixed_baseline_within_each_resample": ci(strongest_gain),
        "strongest_baseline_selection_counts": {
            name: int(np.sum(best_idx == i)) for i, name in enumerate(baseline_names)},
        "draws": {"object_indices": indices, "model_object_macro_nmse": model_draws,
                  "baseline_object_macro_nmse": baseline_draws,
                  "absolute_improvement_best_baseline_minus_model": strongest_gain,
                  "best_baseline_index": best_idx.astype(np.int8)},
        "object_ids": np.asarray(object_ids, dtype=np.str_),
        "baseline_names": np.asarray(baseline_names, dtype=np.str_),
    }


def run_source_object_development(config_relative_path: str) -> dict[str, Any]:
    """Evaluate the fixed EXP-070 model on source-TRAIN objects excluded from its 32 fit rows."""
    require_runtime()
    import torch
    import yaml
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    torch.set_num_threads(1)
    if torch.cuda.is_available() or torch.cuda.is_initialized() or torch.get_num_threads() != 1:
        raise RuntimeError("EXP-071 is CPU-only and must use one PyTorch thread")
    config_path = ROOT / config_relative_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != "EXP-071" or config.get("run_id") != EXP071_RUN_ID:
        raise ValueError("EXP-071 config identity or run_id differs from the frozen protocol")
    if (config.get("model", {}).get("checkpoint_epoch") != 2100
            or config.get("model", {}).get("author_evaluation_seed") != SEED
            or config.get("model", {}).get("model_training") is not False
            or config.get("resources", {}).get("gpu_hours") != 0
            or config.get("resources", {}).get("cpu_threads") != 1):
        raise ValueError("EXP-071 checkpoint, seed, or CPU resource contract changed")
    output_root = ROOT / str(config["outputs"]["root"])
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=False, exist_ok=False)
    started = time.perf_counter()

    try:
        fit = load_runner()
        train_keys, values, feature_by_key, evidence = load_train_inputs(fit)
        frozen_key_path = ROOT / str(config["inputs"]["frozen_fit_keys"])
        frozen_fit = pd.read_csv(frozen_key_path, sep="\t", dtype=str, keep_default_na=False)
        if (len(frozen_fit) != 32 or frozen_fit.full_condition_key.astype(str).duplicated().any()
                or frozen_fit.full_condition_key.astype(str).tolist() !=
                select_train32(train_keys).full_condition_key.astype(str).tolist()):
            raise ValueError("EXP-071's 32 frozen fit keys do not match the previously verified TRAIN32 selection")
        key_index = {str(key): index for index, key in enumerate(train_keys.full_condition_key.astype(str))}
        if len(key_index) != 419:
            raise ValueError("source TRAIN condition keys are not unique")
        fit_positions = np.asarray([key_index[str(key)] for key in frozen_fit.full_condition_key], dtype=np.int64)
        fit_keys = train_keys.iloc[fit_positions].reset_index(drop=True)
        if not np.array_equal(fit_keys.full_condition_key.astype(str), frozen_fit.full_condition_key.astype(str)):
            raise ValueError("frozen fit keys are not aligned to the 419-row source TRAIN contract")
        fit_objects = set(fit_keys.source_object_id.astype(str))
        dev_mask = ~train_keys.source_object_id.astype(str).isin(fit_objects).to_numpy()
        dev_positions = np.flatnonzero(dev_mask)
        dev_keys = train_keys.iloc[dev_positions].reset_index(drop=True)
        if (len(dev_positions) != 316 or dev_keys.source_object_id.astype(str).nunique() != 85
                or dev_keys.full_inchikey.astype(str).nunique() != 5
                or set(dev_keys.source_object_id.astype(str)) & fit_objects
                or dev_keys.full_condition_key.astype(str).duplicated().any()):
            raise ValueError("frozen source-object exclusion did not yield 316 conditions / 85 objects / 5 drugs")
        excluded_rows = train_keys.loc[~dev_mask]
        same_object_nonfit = excluded_rows.loc[~excluded_rows.full_condition_key.isin(
            frozen_fit.full_condition_key.astype(str))]
        if len(excluded_rows) != 103 or len(same_object_nonfit) != 71:
            raise ValueError("excluded fit-object rows differ from the frozen 32 + 71 = 103 condition contract")
        train_delta = values["delta"][fit_positions].astype(np.float32, copy=True)
        fit_ctl = values["X_ctl"][fit_positions].astype(np.float32, copy=True)
        dev_ctl = values["X_ctl"][dev_positions].astype(np.float32, copy=True)
        dev_truth = values["delta"][dev_positions].astype(np.float32, copy=True)
        fit_features = np.stack([feature_by_key[str(k)] for k in fit_keys.full_inchikey]).astype(np.float32)
        dev_features = np.stack([feature_by_key[str(k)] for k in dev_keys.full_inchikey]).astype(np.float32)
        if (train_delta.shape != (32, 978) or fit_ctl.shape != (32, 978)
                or dev_ctl.shape != (316, 978) or dev_truth.shape != (316, 978)
                or fit_features.shape != (32, 2048) or dev_features.shape != (316, 2048)
                or not all(np.isfinite(x).all() for x in
                           (train_delta, fit_ctl, dev_ctl, dev_truth, fit_features, dev_features))):
            raise ValueError("source-object evaluation failed the frozen finite 32/316 x 978/2048 input contract")
        if not np.array_equal(dev_truth, values["X"][dev_positions] - dev_ctl):
            raise ValueError("evaluation targets differ from paired treated expression minus control expression")

        genes = values["genes"].astype(str)
        gene_map_path = OLD_SOURCE_RUN / "gene_axis_mapping.tsv"
        gene_map = pd.read_csv(gene_map_path, sep="\t", dtype=str, keep_default_na=False)
        if (len(gene_map) != 978 or gene_map.axis_index.astype(int).tolist() != list(range(978))
                or gene_map.h5ad_var_name.astype(str).tolist() != genes.tolist()):
            raise ValueError("verified source gene-axis mapping no longer aligns with 978 model columns")
        gene_ids = gene_map.h5ad_var_annotation.astype(str).to_numpy()
        variance32 = np.var(train_delta.astype(np.float64), axis=0, ddof=0)
        variance419 = np.var(values["delta"].astype(np.float64), axis=0, ddof=0)
        primary_gene_use = variance32 > 1e-12
        zero_variance_gene_ids = gene_ids[~primary_gene_use]

        baseline_predictions: dict[str, np.ndarray] = {
            "zero_change": np.zeros((316, 978), dtype=np.float32),
            "global_mean": np.tile(train_delta.mean(axis=0, dtype=np.float64).astype(np.float32), (316, 1)),
        }
        per_drug_mean: dict[str, np.ndarray] = {}
        fit_drugs = fit_keys.full_inchikey.astype(str).to_numpy()
        for drug in sorted(set(fit_drugs)):
            per_drug_mean[drug] = train_delta[fit_drugs == drug].mean(axis=0, dtype=np.float64).astype(np.float32)
        if set(dev_keys.full_inchikey.astype(str)) - set(per_drug_mean):
            raise ValueError("a held-out evaluation drug lacks a training-only per-drug baseline")
        baseline_predictions["per_drug_mean"] = np.stack(
            [per_drug_mean[str(drug)] for drug in dev_keys.full_inchikey], axis=0).astype(np.float32)

        ridge_x_train = np.concatenate([fit_ctl, fit_features], axis=1).astype(np.float64)
        ridge_x_dev = np.concatenate([dev_ctl, dev_features], axis=1).astype(np.float64)
        scaler = StandardScaler()
        ridge_x_train_scaled = scaler.fit_transform(ridge_x_train)
        ridge_x_dev_scaled = scaler.transform(ridge_x_dev)
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(ridge_x_train_scaled, train_delta.astype(np.float64))
        baseline_predictions["ridge_control_ecfp"] = ridge.predict(ridge_x_dev_scaled).astype(np.float32)

        checkpoint = checkpoint_payload(EXP071_CHECKPOINT, torch)
        if checkpoint.get("epoch") != 2100:
            raise ValueError("fixed EXP-070 checkpoint is not the registered epoch-2100 state")
        model, _ = fit._load_transigen(torch, torch.device("cpu"), SEED, features_dim=2048, seed_cuda=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        checkpoint_state_sha = tensor_state_sha256(model.state_dict())
        bn_before = bn_summary(model)
        if not bn_before["all_finite"]:
            raise FloatingPointError("fixed epoch2100 checkpoint has nonfinite BatchNorm buffers")
        original_forward = model.forward
        captured_treated: list[np.ndarray] = []

        def capture_forward(x1_batch: Any, feature_batch: Any) -> Any:
            output = original_forward(x1_batch, feature_batch)
            captured_treated.append(output[3].detach().cpu().numpy().astype(np.float32, copy=True))
            return output

        model.forward = capture_forward
        try:
            transigen_prediction, eval_losses = eval_mode_predictions(
                fit, torch, model, dev_ctl, values["X"][dev_positions].astype(np.float32, copy=True),
                dev_features, epoch=0)
        finally:
            model.forward = original_forward
        if len(captured_treated) != math.ceil(316 / 64):
            raise RuntimeError(f"author eval made {len(captured_treated)} forward batches, expected 5")
        predicted_treated = np.concatenate(captured_treated, axis=0)
        if predicted_treated.shape != (316, 978) or transigen_prediction.shape != (316, 978):
            raise ValueError("fixed checkpoint did not return a complete 316 x 978 evaluation prediction")
        identity_difference = transigen_prediction - (predicted_treated - dev_ctl)
        if not np.allclose(transigen_prediction, predicted_treated - dev_ctl, atol=1e-6, rtol=1e-6):
            raise ValueError("prediction delta identity failed: expected x2_pred - x1")
        if not np.isfinite(transigen_prediction).all() or not np.isfinite(predicted_treated).all():
            raise FloatingPointError("fixed epoch2100 author-eval prediction contains nonfinite values")
        state_after_sha = tensor_state_sha256(model.state_dict())
        bn_after = bn_summary(model)
        if state_after_sha != checkpoint_state_sha or bn_after != bn_before:
            raise RuntimeError("author evaluation mutated fixed checkpoint or BatchNorm state")
        predictions = {"transigen_fixed2100": transigen_prediction.astype(np.float32), **baseline_predictions}

        # In-sample readback is diagnostic only; it does not select or tune the checkpoint.
        fit_model_predictions, fit_losses = eval_mode_predictions(
            fit, torch, model, fit_ctl, values["X"][fit_positions].astype(np.float32, copy=True),
            fit_features, epoch=0)
        if fit_model_predictions.shape != (32, 978) or not np.isfinite(fit_model_predictions).all():
            raise FloatingPointError("fixed epoch2100 in-sample author-eval readback is nonfinite")
        fit_predictions = {
            "transigen_fixed2100": fit_model_predictions.astype(np.float32),
            "zero_change": np.zeros((32, 978), dtype=np.float32),
            "global_mean": np.tile(train_delta.mean(axis=0, dtype=np.float64).astype(np.float32), (32, 1)),
            "per_drug_mean": np.stack([per_drug_mean[str(drug)] for drug in fit_drugs], axis=0).astype(np.float32),
            "ridge_control_ecfp": ridge.predict(scaler.transform(
                np.concatenate([fit_ctl, fit_features], axis=1).astype(np.float64))).astype(np.float32),
        }

        atomic_json(output_root / "run_protocol.json", {
            "experiment_id": "EXP-071", "run_id": EXP071_RUN_ID,
            "protocol_status": "FROZEN_SOURCE_TRAIN_INTERNAL_DEVELOPMENT",
            "config_path": config_relative_path, "config_sha256": sha256_file(config_path),
            "runner_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "source_runner_sha256": sha256_file(Path(fit.__file__).resolve()),
            "checkpoint_path": str(EXP071_CHECKPOINT.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(EXP071_CHECKPOINT),
            "checkpoint_epoch": 2100, "checkpoint_state_sha256": checkpoint_state_sha,
            "author_evaluation_seed": SEED, "author_evaluation_batch_size": 64,
            "fit_key_path": str(frozen_key_path.relative_to(ROOT)),
            "fit_key_sha256": sha256_file(frozen_key_path),
            "fit_condition_key_sha256": sha256_text("\n".join(fit_keys.full_condition_key.astype(str)) + "\n"),
            "fit_conditions": 32, "fit_objects": len(fit_objects),
            "development_conditions": len(dev_keys), "development_objects": int(dev_keys.source_object_id.nunique()),
            "development_drugs": int(dev_keys.full_inchikey.nunique()), "genes": 978,
            "excluded_fit_object_conditions": len(excluded_rows),
            "excluded_fit_conditions": len(frozen_fit), "excluded_same_object_nonfit_conditions": len(same_object_nonfit),
            "fit_development_object_overlap": 0,
            "input_contract": "existing root-verified TRAIN-only source inputs: paired X_ctl, X and X-X_ctl; audited full-InChIKey ECFP4",
            "input_evidence": {
                "source_h5ad_sha256": evidence["source_gate"].get("source_h5ad_sha256"),
                "train_condition_keys_sha256": evidence["train_input_evidence"].get("train_condition_keys_sha256"),
                "train_delta_sha256": evidence["train_input_evidence"].get("train_delta_sha256"),
                "feature_matrix_sha256": evidence.get("feature_matrix_sha256"),
            },
            "development_role": "source-TRAIN internal object-held-out development; not external testing",
            "forbidden_numeric_roles_opened": {"historical_93_dev": False, "outer_locked": False,
                                                 "pdo": False, "breast_response": False},
            "new_training_or_finetuning": False, "seed_or_checkpoint_selection": False,
            "baselines_fit_scope": "32 fixed labels only",
            "ridge": {"input": "32-row X_ctl concatenated with audited ECFP4", "scaler": "StandardScaler fit on 32 rows only",
                      "alpha": 1.0, "fit_intercept": True, "search": False},
            "primary_gene_variance_fit_scope": "32 fixed training deltas only",
            "gene_variance_ddof": 0, "zero_variance_threshold": 1e-12,
            "sensitivity_gene_variance_fit_scope": "all 419 source TRAIN deltas only",
            "bootstrap": {"unit": "source_object_id", "paired": True, "replicates": 10000,
                          "seed": SEED, "interval": "percentile 95%"},
            "device": "cpu", "gpu_hours": 0.0, "torch_num_threads": 1,
        })
        pd.DataFrame({"full_condition_key": fit_keys.full_condition_key.astype(str),
                      "source_object_id": fit_keys.source_object_id.astype(str),
                      "full_inchikey": fit_keys.full_inchikey.astype(str),
                      "pert_id": fit_keys.pert_id.astype(str)}).to_csv(
            output_root / "train32_conditions.tsv", sep="\t", index=False, lineterminator="\n")
        pd.DataFrame({"full_condition_key": dev_keys.full_condition_key.astype(str),
                      "source_object_id": dev_keys.source_object_id.astype(str),
                      "full_inchikey": dev_keys.full_inchikey.astype(str),
                      "pert_id": dev_keys.pert_id.astype(str)}).to_csv(
            output_root / "development_conditions.tsv", sep="\t", index=False, lineterminator="\n")
        pd.DataFrame({"axis_index": np.arange(978, dtype=np.int64), "model_gene_axis_name": genes,
                      "gene_id": gene_ids, "variance_from_train32": variance32,
                      "variance_from_train419": variance419,
                      "included_in_primary_nmse": primary_gene_use}).to_csv(
            output_root / "primary_gene_variance.tsv", sep="\t", index=False, lineterminator="\n")
        pd.DataFrame({"zero_variance_gene_id": zero_variance_gene_ids}).to_csv(
            output_root / "zero_variance_gene_ids.tsv", sep="\t", index=False, lineterminator="\n")
        np.savez_compressed(output_root / "ridge_train32_parameters.npz",
                            scaler_mean=scaler.mean_.astype(np.float64), scaler_scale=scaler.scale_.astype(np.float64),
                            coefficients=ridge.coef_.astype(np.float64), intercept=ridge.intercept_.astype(np.float64))
        atomic_npz(output_root / "predictions_316_keyed.npz",
            full_condition_key=dev_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
            source_object_id=dev_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
            full_inchikey=dev_keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
            pert_id=dev_keys.pert_id.astype(str).to_numpy(dtype=np.str_),
            genes=genes.astype(np.str_), gene_ids=gene_ids.astype(np.str_),
            target_delta=dev_truth, control_input=dev_ctl, treated_target=values["X"][dev_positions],
            transigen_fixed2100_prediction_delta=predictions["transigen_fixed2100"],
            transigen_fixed2100_predicted_treated=predicted_treated,
            zero_change_prediction=predictions["zero_change"],
            global_mean_prediction=predictions["global_mean"],
            per_drug_mean_prediction=predictions["per_drug_mean"],
            ridge_control_ecfp_prediction=predictions["ridge_control_ecfp"])
        atomic_npz(output_root / "train32_readback_keyed.npz",
            full_condition_key=fit_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
            source_object_id=fit_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
            full_inchikey=fit_keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
            pert_id=fit_keys.pert_id.astype(str).to_numpy(dtype=np.str_),
            genes=genes.astype(np.str_), gene_ids=gene_ids.astype(np.str_), target_delta=train_delta,
            transigen_fixed2100_prediction_delta=fit_predictions["transigen_fixed2100"],
            zero_change_prediction=fit_predictions["zero_change"],
            global_mean_prediction=fit_predictions["global_mean"],
            per_drug_mean_prediction=fit_predictions["per_drug_mean"],
            ridge_control_ecfp_prediction=fit_predictions["ridge_control_ecfp"])

        # Metric calculation is deliberately performed from a fresh read of the keyed archive.
        with np.load(output_root / "predictions_316_keyed.npz", allow_pickle=False) as saved:
            if (not np.array_equal(saved["full_condition_key"], dev_keys.full_condition_key.astype(str).to_numpy())
                    or not np.array_equal(saved["source_object_id"], dev_keys.source_object_id.astype(str).to_numpy())
                    or not np.array_equal(saved["genes"], genes)
                    or not np.array_equal(saved["gene_ids"], gene_ids)
                    or not np.array_equal(saved["target_delta"], dev_truth)):
                raise ValueError("keyed prediction archive readback does not match frozen evaluation rows/axis/targets")
            saved_predictions = {
                "transigen_fixed2100": saved["transigen_fixed2100_prediction_delta"].copy(),
                "zero_change": saved["zero_change_prediction"].copy(),
                "global_mean": saved["global_mean_prediction"].copy(),
                "per_drug_mean": saved["per_drug_mean_prediction"].copy(),
                "ridge_control_ecfp": saved["ridge_control_ecfp_prediction"].copy(),
            }
        metric_records: list[dict[str, Any]] = []
        primary_objects: dict[str, pd.DataFrame] = {}
        sensitivity_objects: dict[str, pd.DataFrame] = {}
        metric_summaries: dict[str, dict[str, Any]] = {"primary_train32_variance": {}, "sensitivity_train419_variance": {}}
        per_object_rows: list[pd.DataFrame] = []
        for variance_name, variance in (("primary_train32_variance", variance32),
                                        ("sensitivity_train419_variance", variance419)):
            for method, prediction in saved_predictions.items():
                summary, objects = _object_metric_table(dev_keys, dev_truth, prediction, variance)
                metric_summaries[variance_name][method] = summary
                per_obj = objects.copy()
                per_obj.insert(0, "method", method)
                per_obj.insert(0, "variance_scope", variance_name)
                per_object_rows.append(per_obj)
                metric_records.append({"evaluation_scope": "316_source_train_conditions_85_objects",
                                       "variance_scope": variance_name, "method": method, **{
                    name: summary[name] for name in ("n_conditions", "n_objects", "n_drugs",
                        "n_genes_in_nmse_denominator", "n_zero_variance_genes_excluded",
                        "n_nmse_valid_conditions", "n_mae_valid_conditions", "n_direction_valid_conditions",
                        "n_correlation_valid_conditions", "n_amplitude_valid_objects", "object_macro_nmse",
                        "object_macro_mae", "object_macro_gene_correlation",
                        "object_macro_direction_agreement", "object_macro_amplitude_ratio")}})
                if variance_name == "primary_train32_variance":
                    primary_objects[method] = objects
                else:
                    sensitivity_objects[method] = objects
        pd.DataFrame(metric_records).to_csv(output_root / "per_method_metrics.tsv", sep="\t", index=False,
                                            lineterminator="\n")
        pd.concat(per_object_rows, ignore_index=True).to_csv(
            output_root / "per_object_metrics.tsv", sep="\t", index=False, lineterminator="\n")

        fit_metric_summaries: dict[str, dict[str, Any]] = {"primary_train32_variance": {}, "sensitivity_train419_variance": {}}
        fit_objects_by_method: dict[str, pd.DataFrame] = {}
        fit_metric_rows: list[dict[str, Any]] = []
        with np.load(output_root / "train32_readback_keyed.npz", allow_pickle=False) as saved_fit:
            if (not np.array_equal(saved_fit["full_condition_key"], fit_keys.full_condition_key.astype(str).to_numpy())
                    or not np.array_equal(saved_fit["target_delta"], train_delta)):
                raise ValueError("TRAIN32 readback keys or targets differ from frozen fitting labels")
            saved_fit_predictions = {
                "transigen_fixed2100": saved_fit["transigen_fixed2100_prediction_delta"].copy(),
                "zero_change": saved_fit["zero_change_prediction"].copy(),
                "global_mean": saved_fit["global_mean_prediction"].copy(),
                "per_drug_mean": saved_fit["per_drug_mean_prediction"].copy(),
                "ridge_control_ecfp": saved_fit["ridge_control_ecfp_prediction"].copy(),
            }
        for variance_name, variance in (("primary_train32_variance", variance32),
                                        ("sensitivity_train419_variance", variance419)):
            for method, prediction in saved_fit_predictions.items():
                summary, objects = _object_metric_table(fit_keys, train_delta, prediction, variance)
                fit_metric_summaries[variance_name][method] = summary
                if variance_name == "primary_train32_variance":
                    fit_objects_by_method[method] = objects
                fit_metric_rows.append({"evaluation_scope": "32_fixed_train_rows_in_sample_readback",
                                        "variance_scope": variance_name, "method": method, **{
                    name: summary[name] for name in ("n_conditions", "n_objects", "n_drugs",
                        "n_genes_in_nmse_denominator", "n_zero_variance_genes_excluded",
                        "n_nmse_valid_conditions", "n_mae_valid_conditions", "n_direction_valid_conditions",
                        "n_correlation_valid_conditions", "n_amplitude_valid_objects", "object_macro_nmse",
                        "object_macro_mae", "object_macro_gene_correlation",
                        "object_macro_direction_agreement", "object_macro_amplitude_ratio")}})
        pd.DataFrame(fit_metric_rows).to_csv(output_root / "train32_readback_metrics.tsv", sep="\t", index=False,
                                             lineterminator="\n")

        baseline_names = ("zero_change", "global_mean", "per_drug_mean", "ridge_control_ecfp")
        bootstrap = _bootstrap_object_nmse(
            primary_objects["transigen_fixed2100"], {name: primary_objects[name] for name in baseline_names},
            seed=SEED, replicates=int(config["metrics"]["paired_object_bootstrap_replicates"]))
        bootstrap_archive = {key: value for key, value in bootstrap["draws"].items()}
        bootstrap_archive["object_ids"] = bootstrap["object_ids"]
        bootstrap_archive["baseline_names"] = bootstrap["baseline_names"]
        atomic_npz(output_root / "paired_object_bootstrap.npz", **bootstrap_archive)
        strongest_baseline = min(baseline_names,
            key=lambda name: metric_summaries["primary_train32_variance"][name]["object_macro_nmse"])
        strongest_nmse = float(metric_summaries["primary_train32_variance"][strongest_baseline]["object_macro_nmse"])
        model_summary = metric_summaries["primary_train32_variance"]["transigen_fixed2100"]
        relative_reduction = 1.0 - float(model_summary["object_macro_nmse"]) / strongest_nmse if strongest_nmse > 0 else None
        absolute_gain = strongest_nmse - float(model_summary["object_macro_nmse"])
        bootstrap_best_ci = bootstrap["best_fixed_baseline_within_each_resample"]
        amp = model_summary["object_macro_amplitude_ratio"]
        pass_nmse = relative_reduction is not None and relative_reduction >= 0.10
        pass_amplitude = amp is not None and 0.5 <= float(amp) <= 2.0
        pass_ci = float(bootstrap_best_ci["ci95_percentile_lower"]) > 0
        gate_pass = bool(pass_nmse and pass_amplitude and pass_ci and np.isfinite(transigen_prediction).all())
        bootstrap_summary = {key: value for key, value in bootstrap.items() if key not in ("draws", "object_ids", "baseline_names")}
        result = {
            "status": "PASS_SOURCE_OBJECT_DEVELOPMENT_GATE" if gate_pass else "FAIL_SOURCE_OBJECT_DEVELOPMENT_GATE",
            "experiment_id": "EXP-071", "run_id": EXP071_RUN_ID,
            "development_role": "source-TRAIN internal object-held-out development; not independent external validation",
            "counts": {"source_train_conditions": 419, "source_train_objects": 110, "fit_conditions": 32,
                       "fit_objects": len(fit_objects), "fit_object_excluded_conditions": len(excluded_rows),
                       "excluded_fit_conditions": len(frozen_fit),
                       "excluded_same_object_nonfit_conditions": len(same_object_nonfit),
                       "development_conditions": 316, "development_objects": int(dev_keys.source_object_id.nunique()),
                       "development_drugs": int(dev_keys.full_inchikey.nunique()), "genes": 978,
                       "fit_development_object_overlap": 0},
            "checkpoint": {"path": str(EXP071_CHECKPOINT.relative_to(ROOT)),
                           "sha256": sha256_file(EXP071_CHECKPOINT), "epoch": 2100,
                           "state_sha256": checkpoint_state_sha, "author_eval_seed": SEED,
                           "author_eval_batch_size": 64, "training_or_finetuning": False,
                           "parameter_and_batchnorm_state_unchanged": state_after_sha == checkpoint_state_sha,
                           "batchnorm_all_finite": bn_after["all_finite"]},
            "prediction_readback": {"all_methods_finite": all(np.isfinite(x).all() for x in saved_predictions.values()),
                                    "transigen_finite_elements": int(np.isfinite(transigen_prediction).sum()),
                                    "transigen_total_elements": int(transigen_prediction.size),
                                    "delta_identity_max_abs_error": float(np.max(np.abs(identity_difference))),
                                    "fixed_checkpoint_author_eval_loss": {k: float(v) for k, v in eval_losses.items()},
                                    "train32_fixed_checkpoint_author_eval_loss": {k: float(v) for k, v in fit_losses.items()},
                                    "prediction_archive_reloaded_for_metrics": True},
            "primary_gene_variance": {"fit_scope": "32 fixed training labels only", "ddof": 0,
                                       "threshold": 1e-12, "included_gene_count": int(primary_gene_use.sum()),
                                       "excluded_zero_or_tiny_variance_gene_count": int((~primary_gene_use).sum()),
                                       "excluded_gene_ids_path": "zero_variance_gene_ids.tsv",
                                       "variance_table_path": "primary_gene_variance.tsv"},
            "evaluation_metrics": metric_summaries,
            "train32_in_sample_readback_metrics": fit_metric_summaries,
            "strongest_fixed_baseline": strongest_baseline,
            "strongest_fixed_baseline_nmse": strongest_nmse,
            "transigen_primary_nmse": float(model_summary["object_macro_nmse"]),
            "absolute_nmse_improvement": absolute_gain,
            "relative_nmse_reduction": relative_reduction,
            "paired_object_bootstrap": bootstrap_summary,
            "gate": {"required_relative_nmse_reduction": 0.10, "observed_relative_nmse_reduction": relative_reduction,
                     "passes_nmse_reduction": pass_nmse, "required_amplitude_ratio": [0.5, 2.0],
                     "observed_amplitude_ratio": amp, "passes_amplitude": pass_amplitude,
                     "required_paired_absolute_nmse_improvement_ci_lower": "> 0",
                     "observed_paired_absolute_nmse_improvement_ci_lower": bootstrap_best_ci["ci95_percentile_lower"],
                     "passes_bootstrap_ci": pass_ci, "final_eval_all_finite": bool(np.isfinite(transigen_prediction).all()),
                     "pass": gate_pass},
            "interpretation": ("A pass is only an internal source-TRAIN development signal and does not approve a breast panel or PDO." if gate_pass
                               else "The fixed checkpoint did not pass the preregistered source-object development gate; stop fixed-model panel extrapolation. This result does not identify the cause as lack of information."),
            "forbidden_numeric_roles_opened": {"historical_93_dev": False, "outer_locked": False,
                                                 "pdo": False, "breast_response": False},
            "resources": {"device": "cpu", "gpu_hours": 0.0, "torch_num_threads": 1,
                           "elapsed_wall_seconds": time.perf_counter() - started},
        }
        atomic_json(output_root / "result.json", result)
        atomic_json(output_root / "paired_object_bootstrap_summary.json", bootstrap_summary)
        files = sorted(p for p in output_root.iterdir() if p.is_file())
        final_manifest = {"status": result["status"], "experiment_id": "EXP-071", "run_id": EXP071_RUN_ID,
                          "config_sha256": sha256_file(config_path),
                          "runner_sha256": sha256_file(Path(__file__).resolve()),
                          "checkpoint_sha256": sha256_file(EXP071_CHECKPOINT),
                          "files_sha256": {str(p.name): sha256_file(p) for p in files},
                          "result_sha256": sha256_file(output_root / "result.json"),
                          "inner_dev_new_numeric_values_read": False,
                          "outer_locked_pdo_breast_response_opened": False}
        atomic_json(output_root / "run_manifest.json", final_manifest)
        return result
    except Exception as error:
        failure = {"status": "FAILED_DURING_SOURCE_OBJECT_DEVELOPMENT_EVALUATION",
                   "experiment_id": "EXP-071", "run_id": EXP071_RUN_ID,
                   "exception_type": type(error).__name__, "error": str(error),
                   "config_path": config_relative_path,
                   "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
                   "checkpoint_path": str(EXP071_CHECKPOINT.relative_to(ROOT)),
                   "checkpoint_sha256": sha256_file(EXP071_CHECKPOINT) if EXP071_CHECKPOINT.is_file() else None,
                   "forbidden_numeric_roles_opened": {"historical_93_dev": False, "outer_locked": False,
                                                        "pdo": False, "breast_response": False},
                   "elapsed_wall_seconds": time.perf_counter() - started}
        if not (output_root / "failure.json").exists():
            atomic_json(output_root / "failure.json", failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("audit", "train32", "trace-initial", "sampling-decomposition",
                                             "fixed-updates", "source-object-dev", "all"), required=True)
    parser.add_argument("--config", default=None, help="frozen experiment config for the source-object-dev phase")
    parser.add_argument("--attempt-id", default=TRAIN32_DIR_NAME,
                        help="unique immutable TRAIN32 attempt directory; use a new id for each fit")
    parser.add_argument("--posthoc-id", default="posthoc_native_sampling_v1",
                        help="unique read-only diagnostic output directory under the experiment run")
    args = parser.parse_args()
    if args.phase == "source-object-dev":
        if args.config is None:
            parser.error("--config is required for --phase source-object-dev")
        print(json.dumps(run_source_object_development(args.config), ensure_ascii=False), flush=True)
        return
    if not OUT_ROOT.exists():
        OUT_ROOT.mkdir(parents=True, exist_ok=False)
    if args.phase in {"audit", "all"}:
        print(json.dumps(run_legacy_audit(OUT_ROOT), ensure_ascii=False), flush=True)
    if args.phase in {"train32", "all"}:
        print(json.dumps(run_train32(OUT_ROOT, args.attempt_id), ensure_ascii=False), flush=True)
    if args.phase == "trace-initial":
        print(json.dumps(run_initial_failure_diagnostic(OUT_ROOT), ensure_ascii=False), flush=True)
    if args.phase == "sampling-decomposition":
        print(json.dumps(run_sampling_decomposition(OUT_ROOT, args.attempt_id, args.posthoc_id),
                         ensure_ascii=False), flush=True)
    if args.phase == "fixed-updates":
        print(json.dumps(run_exp070_fixed_updates(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
