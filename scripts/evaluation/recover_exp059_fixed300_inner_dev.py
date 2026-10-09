#!/usr/bin/env python3
"""Recover the frozen EXP-059 epoch-300 inner-dev evaluation without retraining."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.modeling import exp051_contract_fit_smoke as runner


RUN_ID = "20261008_exp059_transigen_lr1e4_bindingfix_v2"
ATTEMPT_ID = "fit_attempt_001"
RUN = ROOT / "artifacts/experiments/EXP-059" / RUN_ID
ATTEMPT = RUN / ATTEMPT_ID
OUT = RUN / "postcheckpoint_inner_dev_evaluation_recovery_v1"
CONFIG = ROOT / "configs/experiments/EXP-059-transigen-lr-stability.yaml"
EXPECTED_ATTEMPT_RUNNER_SHA = "06898e2fb566dcd90680ec12753a4173aa21e728fd1a284cc190807ac6435f9c"
EXPECTED_CONFIG_SHA = "02a817f6f523d1edcfa1068ee6a940e1a1c9a307733629b19c6d2d1b99505e4b"
CPU_TOTAL_CAP_SECONDS = 7200.0
RECOVERY_PRECHECK_CPU_CHARGE_SECONDS = 33.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def frozen_endpoint_context(*, allow_output_exists: bool = False) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("EXP-059 post-checkpoint recovery is CPU-only")
    required = (ATTEMPT / "fit_smoke_manifest.json", ATTEMPT / "attempt_close_receipt.json",
        ATTEMPT / "checkpoint_freeze_receipt.json", ATTEMPT / "epoch_300_endpoint_receipt.json",
        ATTEMPT / "cpu_cost_gate.json", ATTEMPT / "TranSiGen_final_epoch_300.pt",
        ATTEMPT / "TranSiGen_epoch_300_train_keyed_predictions.npz",
        ATTEMPT / "TranSiGen_training_curve.jsonl", ATTEMPT / "train_baseline_metrics.json",
        ATTEMPT / "run_input_receipt.json")
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("fixed-300 endpoint evidence is incomplete")
    manifest = read_json(ATTEMPT / "fit_smoke_manifest.json")
    close = read_json(ATTEMPT / "attempt_close_receipt.json")
    freeze = read_json(ATTEMPT / "checkpoint_freeze_receipt.json")
    endpoint = read_json(ATTEMPT / "epoch_300_endpoint_receipt.json")
    cost = read_json(ATTEMPT / "cpu_cost_gate.json")
    input_receipt = read_json(ATTEMPT / "run_input_receipt.json")
    checkpoint = ATTEMPT / "TranSiGen_final_epoch_300.pt"
    checkpoint_sha = sha256(checkpoint)
    if (manifest.get("run_id") != RUN_ID or manifest.get("fit_attempt") != ATTEMPT_ID
            or manifest.get("runner_sha256") != EXPECTED_ATTEMPT_RUNNER_SHA
            or manifest.get("config_sha256") != EXPECTED_CONFIG_SHA
            or close.get("status") != "FAILED_CLOSED_CPU_ATTEMPT"
            or close.get("exception_type") != "IndexError"
            or close.get("last_completed_finite_epoch") != 300
            or freeze.get("status") != "PASS_FROZEN_EPOCH_300_CHECKPOINT"
            or freeze.get("checkpoint_sha256") != checkpoint_sha
            or endpoint.get("prediction_archive_written") is not True
            or endpoint.get("train_main_delta_finite") is not True
            or endpoint.get("train_main_delta_denominator") != 419 * 978
            or cost.get("status") != "PASS_FIXED300_WITHIN_CPU_CAP"
            or input_receipt.get("status") != "PASS_TRAIN_ONLY_FROZEN_INPUTS"):
        raise RuntimeError("the saved EXP-059 fixed-300 endpoint does not match its immutable receipts")
    if (ATTEMPT / "transigen_cpu_full_fit_result.json").exists():
        raise FileExistsError("a full-fit result already exists; this recovery may not replace it")
    if OUT.exists() and not allow_output_exists:
        raise FileExistsError("post-checkpoint evaluation recovery is immutable and already exists")
    cumulative = (float(close["cpu_elapsed_total_with_contract_checks_and_allowance_seconds"])
        + RECOVERY_PRECHECK_CPU_CHARGE_SECONDS)
    if cumulative >= CPU_TOTAL_CAP_SECONDS:
        raise TimeoutError("EXP-059 CPU package has no remaining allowance for endpoint scoring")
    runner.configure_experiment("EXP-059")
    runner.configure_run(RUN_ID)
    cfg, config_sha = runner.load_config(CONFIG, "EXP-059")
    if config_sha != EXPECTED_CONFIG_SHA or cfg.get("run_id") != RUN_ID:
        raise RuntimeError("recovery config differs from the immutable formal attempt")
    return {"manifest": manifest, "close": close, "freeze": freeze, "endpoint": endpoint,
        "cost": cost, "input_receipt": input_receipt, "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha, "cumulative_cpu_before_recovery_seconds": cumulative,
        "recovery_precheck_cpu_charge_seconds": RECOVERY_PRECHECK_CPU_CHARGE_SECONDS,
        "remaining_cpu_before_recovery_seconds": CPU_TOTAL_CAP_SECONDS - cumulative,
        "config_sha256": config_sha}


def load_train_dev_inputs() -> tuple[dict, object, object, dict, dict, dict]:
    # This function is called only after frozen_endpoint_context() verifies the epoch-300 receipt.
    verified_gate = runner._reuse_root_verified_exp055_train_gate()
    source_run = runner.SOURCE_RUN_DIR
    train_keys, train_values, feature_by_key, train_evidence = runner._load_exp055_trainpath_train_inputs_reusing_gate(
        verified_gate, source_run / "response_blind_conditions.tsv", source_run)
    dev_keys, dev_values, dev_evidence = runner._load_exp055_inner_dev_after_checkpoint_freeze(
        train_keys, train_values, source_run)
    if len(train_keys) != 419 or len(dev_keys) != 93 or dev_evidence.get("object_overlap") != 0:
        raise ValueError("recovered fixed split differs from the registered 419/93 object-disjoint split")
    return verified_gate, train_keys, dev_keys, train_values, dev_values, {
        "feature_by_key": feature_by_key, "train_evidence": train_evidence, "dev_evidence": dev_evidence}


def read_train_prediction(train_keys, train_values) -> tuple[dict[str, np.ndarray], dict]:
    path = ATTEMPT / "TranSiGen_epoch_300_train_keyed_predictions.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    expected_keys = train_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_)
    expected_objects = train_keys.source_object_id.astype(str).to_numpy(dtype=np.str_)
    expected_drugs = train_keys.full_inchikey.astype(str).to_numpy(dtype=np.str_)
    checks = {
        "condition_keys_exact": np.array_equal(arrays["train_condition_key"], expected_keys),
        "object_ids_exact": np.array_equal(arrays["train_source_object_id"], expected_objects),
        "full_inchikeys_exact": np.array_equal(arrays["train_full_inchikey"], expected_drugs),
        "gene_axis_exact": np.array_equal(arrays["genes"].astype(str), train_values["genes"].astype(str)),
        "targets_exact": np.array_equal(arrays["train_target_delta"], train_values["delta"]),
        "predictions_finite": bool(np.isfinite(arrays["train_delta"]).all()),
        "prediction_shape": arrays["train_delta"].shape == (419, 978),
    }
    if not all(checks.values()):
        raise RuntimeError(f"saved fixed-300 TRAIN predictions failed keyed readback: {checks}")
    return arrays, {"status": "PASS_TRAIN_PREDICTION_KEYED_READBACK",
        "path": str(path.relative_to(ROOT)), "sha256": sha256(path), "checks": checks}


def original_attempt_hashes() -> dict[str, str]:
    names = ("fit_smoke_manifest.json", "attempt_close_receipt.json", "checkpoint_freeze_receipt.json",
        "epoch_100_endpoint_receipt.json", "epoch_300_endpoint_receipt.json", "cpu_cost_gate.json",
        "run_input_receipt.json", "train_baseline_metrics.json", "TranSiGen_final_epoch_300.pt",
        "TranSiGen_last_finite.pt", "TranSiGen_epoch_300_train_keyed_predictions.npz",
        "TranSiGen_training_curve.jsonl", "TranSiGen_training_batches.jsonl")
    return {name: sha256(ATTEMPT / name) for name in names}


def run_evaluation() -> dict:
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    context = frozen_endpoint_context()
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError("post-checkpoint EXP-059 recovery unexpectedly initialized CUDA")
    _, train_keys, dev_keys, train_values, dev_values, input_evidence = load_train_dev_inputs()
    feature_by_key = input_evidence["feature_by_key"]
    train_y = train_values["delta"].astype(np.float32, copy=False)
    dev_y = dev_values["delta"].astype(np.float32, copy=False)
    gene_variance = np.var(train_y.astype(np.float64), axis=0, ddof=0)
    device = torch.device("cpu")
    model, _ = runner._load_transigen(torch, device, runner.SEED,
        features_dim=2048, seed_cuda=False)
    payload = torch.load(context["checkpoint"], map_location="cpu", weights_only=True)
    if (payload.get("epoch") != 300
            or payload.get("checkpoint_role") != "frozen_fixed_epoch_300_main_endpoint"):
        raise RuntimeError("saved checkpoint role or epoch is not the fixed main endpoint")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    dev_features = runner._full_key_features(dev_keys, np.arange(len(dev_keys), dtype=np.int64), feature_by_key)
    dev_failures: list[dict] = []
    hooks, layer_state = runner._trainpath_attach_eval_failure_hooks(torch, model)
    try:
        dev_prediction, dev_losses = runner._transigen_predict_and_losses(
            torch, model, dev_values["X_ctl"], dev_values["X"], dev_features, 64, device,
            runner.SEED + 602, seed_cuda=False, allow_nonfinite_auxiliary=True,
            nonfinite_main_delta_callback=dev_failures.append)
    finally:
        for hook in hooks:
            hook.remove()
    dev_prediction = np.asarray(dev_prediction, dtype=np.float32)
    if dev_prediction.shape != (93, 978) or not np.isfinite(dev_prediction).all():
        raise FloatingPointError("fixed-300 inner-dev main delta is non-finite; no score is valid")
    train_arrays, train_readback = read_train_prediction(train_keys, train_values)
    train_metric = runner.keyed_metrics(train_keys, train_y, train_arrays["train_delta"], gene_variance)
    dev_metric = runner.keyed_metrics(dev_keys, dev_y, dev_prediction, gene_variance)
    baselines = runner._exp055_train_and_dev_baselines(train_keys, train_y, dev_keys, dev_y, gene_variance)
    model_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    model_sha = hashlib.sha256(b"".join(
        tensor.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()
        for _, tensor in sorted(model_state.items()))).hexdigest()
    pred_arrays = dict(train_arrays)
    pred_arrays.update(runner._exp055_cpu_prediction_arrays(
        dev_keys, dev_values["genes"], dev_y, dev_prediction, "inner_dev"))
    OUT.mkdir(parents=False, exist_ok=False)
    prediction_path = OUT / "TranSiGen_fixed300_recovered_keyed_predictions.npz"
    runner.atomic_npz(prediction_path, **pred_arrays)
    prediction_readback = runner._exp055_cpu_prediction_readback(prediction_path, pred_arrays)
    curve_rows = [json.loads(line) for line in (ATTEMPT / "TranSiGen_training_curve.jsonl").read_text(
        encoding="utf-8").splitlines()]
    epoch100 = next((row for row in curve_rows if row.get("epoch") == 100), None)
    epoch300 = next((row for row in curve_rows if row.get("epoch") == 300), None)
    result = {"status": "COMPLETED_FIXED300_TRAINED_MODEL_AND_POSTCHECKPOINT_INNER_DEV_EVALUATION_RECOVERED",
        "experiment_id": "EXP-059", "run_id": RUN_ID, "attempt_id": ATTEMPT_ID,
        "original_attempt_status": "FAILED_CLOSED_CPU_ATTEMPT_POSTCHECKPOINT_EVALUATION_IMPLEMENTATION_ERROR",
        "scientific_model_result_valid": True, "evaluation_recovered_without_retraining": True,
        "checkpoint_path": str(context["checkpoint"].relative_to(ROOT)),
        "checkpoint_sha256": context["checkpoint_sha256"],
        "checkpoint_role": payload["checkpoint_role"], "checkpoint_epoch": 300,
        "checkpoint_state_tensor_tree_sha256": model_sha,
        "original_attempt_artifact_hashes": original_attempt_hashes(),
        "runner_sha256_that_trained_checkpoint": context["manifest"]["runner_sha256"],
        "runner_sha256_used_for_read_only_evaluation": runner.sha256_file(Path(runner.__file__).resolve()),
        "config_sha256": context["config_sha256"], "device": "cpu", "gpu_hours": 0.0,
        "training_contract": {"initialization": "fresh author initialization", "seed": runner.SEED,
            "train_conditions": 419, "train_objects": 110, "inner_dev_conditions": 93,
            "inner_dev_objects": 24, "object_overlap": 0, "full_inchikeys": 5, "genes": 978,
            "objective": "raw_sum", "gradient_clipping": False, "learning_rate": 0.0001,
            "weight_decay": 1e-5, "batch_size": 64, "shuffle": True,
            "normal_batchnorm_updates": True, "epochs_completed": 300,
            "optimizer_updates": 2100, "selected_epoch": None, "model_selection": "none"},
        "epoch_100_diagnostic_train_metrics": epoch100.get("train_metrics") if epoch100 else None,
        "epoch_300_train_metrics": epoch300.get("train_metrics") if epoch300 else train_metric,
        "metrics_by_role": {"TRAIN": {"model": train_metric,
            "zero_change_baseline": baselines["metrics"]["zero_change"]["train"],
            "per_drug_mean_delta_baseline": baselines["metrics"]["per_drug_mean_delta"]["train"]},
            "inner_dev": {"model": dev_metric,
                "zero_change_baseline": baselines["metrics"]["zero_change"]["inner_dev"],
                "per_drug_mean_delta_baseline": baselines["metrics"]["per_drug_mean_delta"]["inner_dev"]}},
        "train_prediction_readback": train_readback,
        "inner_dev_input_evidence": input_evidence["dev_evidence"],
        "train_input_evidence_sha256": hashlib.sha256(json.dumps(
            input_evidence["train_evidence"], sort_keys=True, default=str).encode()).hexdigest(),
        "dev_prediction_auxiliary_losses": dev_losses,
        "dev_evaluation_failure_hooks": dev_failures,
        "first_nonfinite_layer": layer_state.get("first_nonfinite_layer"),
        "prediction_path": prediction_path.name,
        "prediction_readback": prediction_readback,
        "selection_rule": "epoch 300 fixed endpoint; epoch 100 diagnostic only; no best-epoch selection",
        "data_access": {"inner_dev_values_loaded_after_epoch_300_checkpoint_freeze": True,
            "inner_dev_used_for_training_or_model_selection": False,
            "pdo_outer_locked_values_opened": False, "panacea_numeric_values_opened": False,
            "train_values_only_for_model_fit_and_baselines": True},
        "parent_checkpoint_loaded_into_candidate": False,
        "original_attempt_closed_receipt_field_correction": {
            "field": "inner_dev_numeric_rows_loaded", "stored_value": False,
            "corrected_interpretation": True,
            "evidence": "The registered loader was called only after epoch-300 checkpoint verification; it opened the frozen 512-row TRAIN/inner-dev NPZ and then raised while applying the 512-row TRAIN mask to the 978-element gene axis.",
            "original_receipt_immutable": True},
        "cpu_elapsed_wall_seconds": time.perf_counter() - started_wall,
        "cpu_process_seconds": time.process_time() - started_cpu,
        "cumulative_cpu_before_recovery_seconds": context["cumulative_cpu_before_recovery_seconds"],
        "cpu_total_cap_seconds": CPU_TOTAL_CAP_SECONDS,
        "gpu_hours": 0.0}
    result_path = OUT / "recovered_fixed300_evaluation.json"
    runner.atomic_json(result_path, result)
    return result


def root_readback() -> dict:
    started = time.perf_counter()
    context = frozen_endpoint_context(allow_output_exists=True)
    result_path = OUT / "recovered_fixed300_evaluation.json"
    prediction_path = OUT / "TranSiGen_fixed300_recovered_keyed_predictions.npz"
    if not result_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError("recovered fixed-300 evaluation artifacts are incomplete")
    result = read_json(result_path)
    attempt_hashes_before = result.get("original_attempt_artifact_hashes", {})
    attempt_hashes_after = original_attempt_hashes()
    _, train_keys, dev_keys, train_values, dev_values, _ = load_train_dev_inputs()
    with np.load(prediction_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    expected = {
        "train_condition_key": train_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
        "train_source_object_id": train_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
        "train_full_inchikey": train_keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
        "train_target_delta": train_values["delta"],
        "inner_dev_condition_key": dev_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
        "inner_dev_source_object_id": dev_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
        "inner_dev_full_inchikey": dev_keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
        "inner_dev_target_delta": dev_values["delta"],
    }
    checks = {f"{name}_exact": name in arrays and np.array_equal(arrays[name], expected[name])
              for name in expected}
    checks.update({"train_prediction_shape": arrays.get("train_delta", np.empty(0)).shape == (419, 978),
        "inner_dev_prediction_shape": arrays.get("inner_dev_delta", np.empty(0)).shape == (93, 978),
        "numeric_arrays_finite": all(np.isfinite(arrays[name]).all()
            for name in ("train_target_delta", "train_delta", "inner_dev_target_delta", "inner_dev_delta")),
        "epoch300_checkpoint_unchanged": sha256(context["checkpoint"]) == result.get("checkpoint_sha256"),
        "original_attempt_still_closed": context["close"].get("status") == "FAILED_CLOSED_CPU_ATTEMPT",
        "endpoint_status_and_score_valid": result.get("scientific_model_result_valid") is True,
        "inner_dev_access_after_freeze": result.get("data_access", {}).get(
            "inner_dev_values_loaded_after_epoch_300_checkpoint_freeze") is True,
        "no_forbidden_values": result.get("data_access", {}).get("pdo_outer_locked_values_opened") is False
            and result.get("data_access", {}).get("panacea_numeric_values_opened") is False,
        "retraining_not_performed": result.get("evaluation_recovered_without_retraining") is True,
        "original_attempt_hashes_unchanged": attempt_hashes_before == attempt_hashes_after,
        "gpu_zero": result.get("gpu_hours") == 0.0})
    recomputed = {}
    for split, keys, y, pred in (
            ("TRAIN", train_keys, train_values["delta"], arrays["train_delta"]),
            ("inner_dev", dev_keys, dev_values["delta"], arrays["inner_dev_delta"])):
        variance = np.var(train_values["delta"].astype(np.float64), axis=0, ddof=0)
        recomputed[split] = runner.keyed_metrics(keys, y, pred, variance)
        recorded = result["metrics_by_role"][split]["model"]
        checks[f"{split}_metrics_recomputed_within_tolerance"] = metrics_equal(
            recomputed[split], recorded, atol=2e-7)
    if not all(checks.values()):
        raise RuntimeError(f"root independent readback failed: {checks}")
    output = OUT / "root_independent_readback.json"
    if output.exists():
        raise FileExistsError("root recovery readback is immutable and already exists")
    payload = {"status": "PASS_ROOT_EXP059_FIXED300_POSTCHECKPOINT_EVALUATION_READBACK",
        "run_id": RUN_ID, "attempt_id": ATTEMPT_ID,
        "recovered_evaluation_sha256": sha256(result_path),
        "prediction_sha256": sha256(prediction_path),
        "frozen_epoch300_checkpoint_sha256": sha256(context["checkpoint"]),
        "checks": checks, "recomputed_model_metrics": recomputed,
        "train_conditions": 419, "train_objects": 110, "inner_dev_conditions": 93,
        "inner_dev_objects": 24, "train_dev_object_overlap": 0, "full_inchikeys": 5,
        "genes": 978, "inner_dev_loaded_after_endpoint_freeze": True,
        "no_retraining": True, "gpu_hours": 0.0,
        "root_readback_wall_seconds": time.perf_counter() - started,
        "original_attempt_close_receipt_sha256": sha256(ATTEMPT / "attempt_close_receipt.json"),
        "original_attempt_mutated": False}
    runner.atomic_json(output, payload)
    return payload


def metrics_equal(left, right, *, atol: float) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(metrics_equal(left[key], right[key], atol=atol) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(metrics_equal(a, b, atol=atol) for a, b in zip(left, right))
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isfinite(left) and np.isfinite(right) and abs(float(left) - float(right)) <= atol)
    return left == right


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["evaluate", "root-readback"], required=True)
    args = parser.parse_args()
    output = run_evaluation() if args.phase == "evaluate" else root_readback()
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
