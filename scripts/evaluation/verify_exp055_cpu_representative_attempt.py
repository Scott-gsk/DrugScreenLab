#!/usr/bin/env python3
"""Root-only readback for the immutable EXP-055 CPU representative attempt.

Purpose: independently verify TRAIN identity/axis, saved epoch-1 TRAIN arrays,
the last finite checkpoint, the epoch-9 nonfinite close, cost-gate arithmetic,
resource settlement, and the TRAIN-only baseline without opening inner-dev,
PDO, outer, or locked numeric values.

Reuse barrier: the existing v12 readback is bound to another run's schemas and
an epoch-1 cost-stop result. This attempt passed its cost gate, continued, then
failed during epoch-9 TRAIN evaluation; its receipt and checkpoint schema differ.
Retention: keep as the root readback entrypoint for this immutable attempt.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20261007_exp055_transigen_cpu_representative_v1"
RUN_DIR = ROOT / "artifacts/experiments/EXP-055" / RUN_ID
ATTEMPT_DIR = RUN_DIR / "fit_attempt_001"
SOURCE_DIR = ROOT / "artifacts/experiments/EXP-051/20261006_exp051_contract_fit_smoke_v1"
RUNNER = ROOT / "scripts/modeling/exp051_contract_fit_smoke.py"
CONFIG = ROOT / "configs/experiments/EXP-055-transigen-cpu-representative.yaml"
OUTPUT = RUN_DIR / "root_independent_readback_v2.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def close(actual: Any, expected: Any, *, rtol: float = 1e-7, atol: float = 1e-10) -> bool:
    try:
        a, b = float(actual), float(expected)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def same_optional_number(actual: Any, expected: Any) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    return close(actual, expected)


def metric_summary(keys: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    variance = np.var(truth.astype(np.float64), axis=0, ddof=0)
    use = np.isfinite(variance) & (variance > 1e-12)
    y, p = truth.astype(np.float64), prediction.astype(np.float64)
    err = p - y
    row_nmse = np.mean(np.square(err[:, use]) / variance[use][None, :], axis=1)
    row_mae = np.mean(np.abs(err), axis=1)
    corr: list[float | None] = []
    direction: list[float | None] = []
    for yy, pp in zip(y, p):
        corr.append(None if np.ptp(yy) <= 0 or np.ptp(pp) <= 0 else float(np.corrcoef(yy, pp)[0, 1]))
        nz = np.abs(yy) > 1e-12
        direction.append(None if not nz.any() else float(np.mean(np.sign(yy[nz]) == np.sign(pp[nz]))))
    frame = keys[["source_object_id", "full_inchikey"]].copy().reset_index(drop=True)
    frame["nmse"] = row_nmse
    frame["mae"] = row_mae
    frame["corr"] = corr
    frame["direction"] = direction
    obj = frame.groupby("source_object_id", sort=True).agg(
        nmse=("nmse", "mean"), mae=("mae", "mean"), corr=("corr", "mean"), direction=("direction", "mean"))
    ratios = []
    for _, positions in frame.groupby("source_object_id", sort=True).indices.items():
        idx = np.asarray(positions, dtype=np.int64)
        denom = float(np.mean(np.abs(y[idx])))
        if denom > 0:
            ratios.append(float(np.mean(np.abs(p[idx])) / denom))
    return {
        "n_conditions": int(len(frame)),
        "n_source_objects": int(frame.source_object_id.nunique()),
        "n_full_inchikeys": int(frame.full_inchikey.nunique()),
        "n_genes": int(truth.shape[1]),
        "n_genes_in_nmse_denominator": int(use.sum()),
        "n_zero_variance_genes_excluded": int((~use).sum()),
        "object_macro_nmse": float(obj.nmse.mean()),
        "object_macro_mae": float(obj.mae.mean()),
        "object_macro_gene_correlation": None if obj["corr"].isna().all() else float(obj["corr"].mean()),
        "object_macro_direction_agreement": None if obj.direction.isna().all() else float(obj.direction.mean()),
        "object_macro_amplitude_ratio": None if not ratios else float(np.mean(ratios)),
        "n_amplitude_objects_with_nonzero_observed_mean_abs": int(len(ratios)),
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite root readback: {OUTPUT}")
    close_receipt = load_json(ATTEMPT_DIR / "attempt_close_receipt.json")
    cost = load_json(ATTEMPT_DIR / "cpu_cost_gate.json")
    manifest = load_json(ATTEMPT_DIR / "fit_smoke_manifest.json")
    input_receipt = load_json(ATTEMPT_DIR / "run_input_receipt.json")
    failure = load_json(ATTEMPT_DIR / "first_failure_context.json")
    progress = load_json(ATTEMPT_DIR / "attempt_progress.json")
    owner_baselines = load_json(ATTEMPT_DIR / "train_baseline_metrics.json")
    ledger = [json.loads(line) for line in (ATTEMPT_DIR / "resource_ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    prediction_path = ATTEMPT_DIR / "TranSiGen_epoch_1_train_keyed_predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as arrays:
        expected_fields = {"train_condition_key", "train_source_object_id", "train_full_inchikey", "genes", "train_target_delta", "train_delta"}
        if set(arrays.files) != expected_fields:
            raise ValueError(f"Unexpected prediction fields: {arrays.files}")
        keys = pd.DataFrame({
            "full_condition_key": arrays["train_condition_key"].astype(str),
            "source_object_id": arrays["train_source_object_id"].astype(str),
            "full_inchikey": arrays["train_full_inchikey"].astype(str),
        })
        genes = arrays["genes"].astype(str)
        truth = np.asarray(arrays["train_target_delta"], dtype=np.float32)
        prediction = np.asarray(arrays["train_delta"], dtype=np.float32)

    # Read only the response-blind TRAIN identity table and gene-axis metadata.
    metadata = pd.read_csv(SOURCE_DIR / "response_blind_conditions.tsv", sep="\t", dtype=str, keep_default_na=False)
    train_meta = metadata.loc[metadata["split_role"].str.lower().eq("train"), ["full_condition_key", "source_object_id", "full_inchikey"]].copy()
    merged = keys.merge(train_meta, on="full_condition_key", how="outer", suffixes=("_out", "_meta"), indicator=True, validate="one_to_one")
    identity_ok = bool(
        len(merged) == 419 and merged["_merge"].eq("both").all()
        and merged["source_object_id_out"].eq(merged["source_object_id_meta"]).all()
        and merged["full_inchikey_out"].eq(merged["full_inchikey_meta"]).all()
        and keys["full_condition_key"].nunique() == 419
    )
    with np.load(SOURCE_DIR / "frozen_train_inner_values.npz", allow_pickle=False) as frozen:
        # This archive member is identity metadata only; no X, X_ctl, or delta is accessed.
        expected_genes = frozen["genes"].astype(str)

    # Safe, read-only checkpoint validation. No optimizer step or model inference occurs.
    checkpoint_path = ATTEMPT_DIR / "TranSiGen_last_finite.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    optimizer_state = checkpoint["optimizer_state_dict"]["state"]
    tensor_checks: dict[str, Any] = {"model_tensors": 0, "optimizer_tensors": 0, "nonfinite_float_tensors": []}
    for name, tensor in state.items():
        if torch.is_tensor(tensor):
            tensor_checks["model_tensors"] += 1
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                tensor_checks["nonfinite_float_tensors"].append(f"model:{name}")
    for parameter_id, slots in optimizer_state.items():
        for slot_name, tensor in slots.items():
            if torch.is_tensor(tensor):
                tensor_checks["optimizer_tensors"] += 1
                if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                    tensor_checks["nonfinite_float_tensors"].append(f"optimizer:{parameter_id}:{slot_name}")

    y_sha = hashlib.sha256(truth.tobytes(order="C")).hexdigest()
    fail = failure["main_delta_failure"]
    fail_row = fail.get("batch_start")
    close_wall = float(close_receipt["cpu_elapsed_wall_seconds"])
    success_contract = float(close_receipt["measured_successful_contract_check_seconds"])
    failed_contract_allowance = float(close_receipt["prior_failed_no_update_contract_check_allowance_seconds"])
    elapsed_reconstructed = close_wall + success_contract + failed_contract_allowance

    base_pred = np.zeros_like(truth)
    for full_key in sorted(keys.full_inchikey.unique()):
        mask = keys.full_inchikey.to_numpy() == full_key
        base_pred[mask] = truth[mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    zero_pred = np.zeros_like(truth)
    train_keys = keys.rename(columns={"full_condition_key": "condition_key"})
    recomputed_baseline = metric_summary(train_keys, truth, base_pred)
    recomputed_zero = metric_summary(train_keys, truth, zero_pred)
    owner_baseline = owner_baselines["metrics"]["per_drug_mean_delta"]
    owner_zero = owner_baselines["metrics"]["zero_change"]
    metric_fields = ("object_macro_nmse", "object_macro_mae", "object_macro_gene_correlation",
                     "object_macro_direction_agreement", "object_macro_amplitude_ratio")
    baseline_checks = {name: same_optional_number(recomputed_baseline[name], owner_baseline[name]) for name in metric_fields}
    zero_checks = {name: same_optional_number(recomputed_zero[name], owner_zero[name]) for name in metric_fields}

    checks = {
        "run_attempt_close_identity": close_receipt.get("run_id") == RUN_ID and close_receipt.get("attempt_id") == "fit_attempt_001" and close_receipt.get("status") == "FAILED_CLOSED_CPU_ATTEMPT",
        "runner_and_config_hashes_match_manifest_and_close": sha256(RUNNER) == manifest.get("runner_sha256") == close_receipt.get("runner_sha256") and sha256(CONFIG) == manifest.get("config_sha256") == close_receipt.get("config_sha256"),
        "train_condition_object_drug_keys_match_frozen_response_blind_metadata": identity_ok,
        "gene_axis_matches_frozen_metadata": len(genes) == 978 and len(np.unique(genes)) == 978 and np.array_equal(genes, expected_genes),
        "target_array_is_exact_registered_train_delta": truth.shape == (419, 978) and np.isfinite(truth).all() and y_sha == input_receipt["train_identity"]["train_delta_sha256"],
        "epoch1_predictions_are_finite_over_full_train_domain": prediction.shape == (419, 978) and np.isfinite(prediction).all() and prediction.size == 409782,
        "checkpoint_is_last_finite_epoch8_and_all_float_states_finite": checkpoint.get("epoch") == 8 and checkpoint.get("checkpoint_role") == "last_finite_completed_TRAIN_epoch" and tensor_checks["model_tensors"] == 51 and tensor_checks["optimizer_tensors"] == 108 and not tensor_checks["nonfinite_float_tensors"] and sha256(checkpoint_path) == close_receipt.get("last_finite_checkpoint_sha256"),
        "epoch9_failure_is_full_train_eval_nonfinite_not_scored": failure.get("epoch") == 9 and failure.get("evaluation") == "TRAIN_ONLY" and fail.get("batch_start") == 192 and fail.get("batch_stop") == 256 and fail.get("nonfinite_n") == 978 and fail.get("numel") == 62592 and failure.get("main_delta_denominator") == 409782 and failure.get("first_nonfinite_layer") is None,
        "cost_gate_passed_fixed300_estimate_but_numerical_failure_closed_run": cost.get("status") == "PASS_FIXED300_WITHIN_CPU_CAP" and cost.get("projected_fixed300_with_20pct_margin_seconds", math.inf) <= cost.get("cpu_cap_seconds", 0) and cost.get("automatic_continuation") is True and close_receipt.get("status") == "FAILED_CLOSED_CPU_ATTEMPT",
        "total_cpu_charge_reconstructs_from_component_allowances": close_receipt.get("cpu_total_cap_seconds") == 7200.0 and close(elapsed_reconstructed, close_receipt.get("cpu_elapsed_total_with_contract_checks_and_allowance_seconds"), atol=1e-6),
        "no_inner_dev_or_forbidden_numeric_values_opened": manifest["data_roles"].get("inner_dev_numeric_rows_loaded_before_checkpoint_freeze") is False and cost.get("inner_dev_numeric_rows_loaded") is False and close_receipt.get("inner_dev_numeric_rows_loaded") is False and close_receipt.get("pdo_outer_locked_values_opened") is False and input_receipt.get("inner_dev_numeric_rows_loaded") is False and input_receipt.get("pdo_outer_locked_values_opened") is False,
        "owner_per_drug_and_zero_train_baselines_independently_recomputed": all(baseline_checks.values()) and all(zero_checks.values()),
        "resource_receipts_are_cpu_only_and_closed": close_receipt.get("gpu_hours") == 0.0 and close_receipt.get("cuda_visible_devices") == "" and progress.get("gpu_hours") == 0.0 and any(x.get("event") == "CPU_ATTEMPT_FAILED_CLOSED" for x in ledger),
    }

    receipt = {
        "status": "ROOT_INDEPENDENT_CPU_ATTEMPT_READBACK_PASS" if all(checks.values()) else "ROOT_INDEPENDENT_CPU_ATTEMPT_READBACK_FAIL",
        "run_id": RUN_ID,
        "attempt_id": "fit_attempt_001",
        "checks": checks,
        "independent_train_domain": {"conditions": int(len(keys)), "objects": int(keys.source_object_id.nunique()),
                                     "full_inchikeys": int(keys.full_inchikey.nunique()), "genes": int(len(genes)),
                                     "target_elements": int(truth.size), "prediction_elements": int(prediction.size),
                                     "epoch9_failure_batch_start": fail_row, "epoch9_nonfinite": int(fail["nonfinite_n"]),
                                     "epoch9_failure_denominator": int(fail["numel"]), "inner_dev_numeric_rows_opened": 0,
                                     "pdo_outer_locked_values_opened": False},
        "independent_checkpoint": {"epoch": int(checkpoint["epoch"]), "role": checkpoint["checkpoint_role"],
                                   "model_tensor_count": tensor_checks["model_tensors"], "optimizer_tensor_count": tensor_checks["optimizer_tensors"],
                                   "nonfinite_float_tensor_count": len(tensor_checks["nonfinite_float_tensors"]),
                                   "sha256": sha256(checkpoint_path)},
        "independent_train_baselines": {"per_drug_mean": recomputed_baseline, "zero_change": recomputed_zero,
                                        "per_drug_match": baseline_checks, "zero_match": zero_checks},
        "cost_and_resources": {"fixed300_projection_with_margin_seconds": cost["projected_fixed300_with_20pct_margin_seconds"],
                               "cpu_cap_seconds": close_receipt["cpu_total_cap_seconds"],
                               "cpu_elapsed_including_contract_failures_seconds": close_receipt["cpu_elapsed_total_with_contract_checks_and_allowance_seconds"],
                               "gpu_hours": close_receipt["gpu_hours"]},
        "input_hashes": {"runner": sha256(RUNNER), "config": sha256(CONFIG), "contract_check": sha256(RUN_DIR / "cpu_entry_contract_check.json"),
                         "manifest": sha256(ATTEMPT_DIR / "fit_smoke_manifest.json"), "input_receipt": sha256(ATTEMPT_DIR / "run_input_receipt.json"),
                         "cost_gate": sha256(ATTEMPT_DIR / "cpu_cost_gate.json"), "close_receipt": sha256(ATTEMPT_DIR / "attempt_close_receipt.json"),
                         "failure_context": sha256(ATTEMPT_DIR / "first_failure_context.json"), "predictions": sha256(prediction_path),
                         "last_finite_checkpoint": sha256(checkpoint_path)},
        "interpretation": "The fixed-300 CPU cost gate passed at 26.93 minutes including the registered 20% margin. The same attempt then failed closed at epoch-9 TRAIN evaluation: 978 of 62,592 values in batch [192:256] were nonfinite. Epoch-8 model/optimizer checkpoint and epoch-1 keyed TRAIN arrays read back safely. No fixed-300 checkpoint or inner-dev score exists; this is an optimization/numerical stability failure, not an evaluable transfer score.",
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
