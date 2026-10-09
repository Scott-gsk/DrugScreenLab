#!/usr/bin/env python3
"""One bounded TRAIN-only replay of the immutable EXP-055 nonfinite loss batch."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/modeling/exp051_contract_fit_smoke.py"
RUN_ID = "20261007_exp059_transigen_lr1e4_fixed300_v1"
SOURCE_RUN_ID = "20261006_exp051_contract_fit_smoke_v1"
PARENT_CHECKPOINT = (ROOT / "artifacts/experiments/EXP-055/"
    "20261007_exp055_transigen_cpu_norm_recovery_v1/fit_attempt_001/TranSiGen_last_finite.pt")
PARENT_FAILURE_CONTEXT = PARENT_CHECKPOINT.with_name("first_failure_context.json")
EXPECTED_PARENT_SHA = "5555e4f0aa3a85c34d075c9211a494862ccb35b65c766b559919ab7f60bc2dfb"
EXPECTED_CONFIG = "EXP-059-transigen-lr-stability.yaml"
MAX_WALL_SECONDS = 600.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(nested_tuple(item) for item in value)
    return value


def tensor_summary(torch: Any, value: Any) -> dict[str, Any]:
    tensor = value.detach()
    finite = torch.isfinite(tensor)
    finite_values = tensor[finite]
    result: dict[str, Any] = {
        "shape": list(tensor.shape), "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()), "finite_n": int(finite.sum().item()),
        "nonfinite_n": int((~finite).sum().item()),
    }
    if finite_values.numel():
        result.update({"finite_min": float(finite_values.min().item()),
            "finite_max": float(finite_values.max().item()),
            "finite_max_abs": float(finite_values.abs().max().item())})
    if not bool(finite.all().item()):
        result["first_nonfinite_index"] = torch.nonzero(~finite, as_tuple=False)[0].cpu().tolist()
    return result


def build_rng_state(torch: Any, raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "python": nested_tuple(raw["python"]),
        "numpy": raw["numpy"],
        "torch_cpu": torch.as_tensor(raw["torch_cpu"], dtype=torch.uint8),
        "dataloader_generator": torch.as_tensor(raw["dataloader_generator"], dtype=torch.uint8),
    }


def first_nonfinite_indices(torch: Any, tensor: Any, limit: int = 8) -> list[list[int]]:
    bad = torch.nonzero(~torch.isfinite(tensor), as_tuple=False)
    return bad[:limit].cpu().tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if (args.run_id != RUN_ID or cfg_path.name != EXPECTED_CONFIG
            or config.get("experiment_id") != "EXP-059"
            or config.get("run_id") != RUN_ID
            or config.get("authorization", {}).get("user_scope") != "EXP059-LR-STABILITY-2303"
            or config.get("authorization", {}).get("gpu_hours_cap") != 0
            or config.get("authorization", {}).get("training_device_policy") != "cpu_only_zero_gpu"):
        raise ValueError("mechanism replay is bound to the single registered EXP-059 CPU package")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("EXP-059 mechanism replay requires CUDA_VISIBLE_DEVICES empty")
    if not PARENT_CHECKPOINT.is_file() or sha256(PARENT_CHECKPOINT) != EXPECTED_PARENT_SHA:
        raise RuntimeError("immutable epoch-211 parent checkpoint is missing or changed")
    if not PARENT_FAILURE_CONTEXT.is_file():
        raise FileNotFoundError("the immutable original epoch-212 batch-3 failure context is missing")
    if not RUNNER.is_file():
        raise FileNotFoundError(RUNNER)
    source_attempt = ROOT / "artifacts/experiments/EXP-055/20261007_exp055_transigen_cpu_norm_recovery_v1/fit_attempt_001"
    close = json.loads((source_attempt / "attempt_close_receipt.json").read_text(encoding="utf-8"))
    context = json.loads(PARENT_FAILURE_CONTEXT.read_text(encoding="utf-8"))
    if (close.get("status") != "FAILED_CLOSED_CPU_ATTEMPT"
            or not str(close.get("exception_message", "")).endswith("epoch 212, batch 3")
            or context.get("epoch") != 212 or context.get("batch_index") != 3
            or context.get("status") != "FIRST_NONFINITE_TRAINING_LOSS"):
        raise RuntimeError("parent failure identity differs from the registered epoch-212/batch-3 event")

    out_dir = ROOT / "artifacts/experiments/EXP-059" / RUN_ID / "mechanism_replay_corrected_001"
    if out_dir.exists():
        raise FileExistsError(f"immutable mechanism replay output already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def budget_check() -> None:
        if time.perf_counter() - started_wall > MAX_WALL_SECONDS:
            raise TimeoutError("single mechanism replay exceeded the 600-second wall cap")

    spec = importlib.util.spec_from_file_location("exp059_shared_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the existing registered TranSiGen runner")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.configure_experiment("EXP-059")
    runner.configure_run(RUN_ID)
    runner.configure_source_run(SOURCE_RUN_ID)

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.set_num_threads(1)
    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized during the CPU-only mechanism replay")
    device = torch.device("cpu")

    gate = runner._reuse_root_verified_exp055_train_gate()
    train_keys, values, feature_by_key, evidence = runner._load_exp055_trainpath_train_inputs_reusing_gate(
        gate, runner.SOURCE_RUN_DIR / "response_blind_conditions.tsv", runner.SOURCE_RUN_DIR)
    budget_check()
    requested_keys = [str(key) for key in context.get("condition_keys", [])]
    if len(requested_keys) != 64 or len(set(requested_keys)) != 64:
        raise ValueError("original failure context does not specify one unique 64-key batch")
    key_to_position = {str(key): index for index, key in enumerate(train_keys.full_condition_key.astype(str))}
    if any(key not in key_to_position for key in requested_keys):
        raise ValueError("one or more original failure-batch keys are absent from the frozen TRAIN table")
    positions = np.asarray([key_to_position[key] for key in requested_keys], dtype=np.int64)
    selected_objects = train_keys.iloc[positions].source_object_id.astype(str).tolist()
    if selected_objects != [str(x) for x in context.get("source_object_ids", [])]:
        raise ValueError("original batch object identities no longer match its exact keys")
    feature_matrix = np.stack([feature_by_key[key] for key in
        train_keys.full_inchikey.astype(str)]).astype(np.float32, copy=False)
    data = TensorDataset(torch.as_tensor(values["X_ctl"], dtype=torch.float32),
        torch.as_tensor(values["X"], dtype=torch.float32),
        torch.as_tensor(feature_matrix, dtype=torch.float32),
        torch.arange(len(train_keys), dtype=torch.int64))
    budget_check()

    payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    if payload.get("epoch") != 211 or payload.get("learning_rate") != 1e-3:
        raise RuntimeError("parent payload is not the immutable original LR=1e-3 epoch-211 checkpoint")
    model, upstream_module = runner._load_transigen(torch, device, runner.SEED,
        features_dim=2048, seed_cuda=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    loader_generator = torch.Generator().manual_seed(runner.SEED)
    restored_rng = runner._exp055_cpu_restore_rng_state(torch, loader_generator, payload["rng_state"])
    model_state_exact = runner._torch_tree_exact(torch, model.state_dict(), payload["state_dict"])
    optimizer_state_exact = runner._torch_tree_exact(torch, optimizer.state_dict(), payload["optimizer_state_dict"])
    if not all(restored_rng.values()) or not model_state_exact or not optimizer_state_exact:
        raise RuntimeError(f"exact replay restoration failed: rng={restored_rng}, model={model_state_exact}, optimizer={optimizer_state_exact}")
    checkpoint_sha_before = sha256(PARENT_CHECKPOINT)
    bn_before, _ = runner._trainpath_bn_snapshot(torch, model, "parent_epoch211_before_shadow")

    # Replay epoch 212 batches 1-2 on this in-memory copy, then inspect the original batch 3.
    loader = DataLoader(data, batch_size=64, shuffle=True, drop_last=False,
                        num_workers=0, generator=loader_generator)
    model.train()
    preceding_batches: list[dict[str, Any]] = []
    target_batch = None
    target_rng = None
    for batch_index, batch in enumerate(loader, start=1):
        budget_check()
        row_positions = batch[3].cpu().numpy().astype(np.int64, copy=False)
        batch_keys = train_keys.iloc[row_positions].full_condition_key.astype(str).tolist()
        rng_before_batch = runner._exp055_cpu_rng_state_json(torch, loader_generator)
        if batch_index == 3:
            target_batch = batch
            target_rng = rng_before_batch
            if batch_keys != requested_keys:
                raise RuntimeError("reconstructed epoch-212 batch-3 key order differs from immutable failure context")
            break
        optimizer.zero_grad(set_to_none=True)
        x1_pre, x2_pre, feature_pre = (batch[0].to(device), batch[1].to(device), batch[2].to(device))
        _, _, _, pre_losses = runner._transigen_forward_batch(
            torch, model, (x1_pre, x2_pre, feature_pre), device)
        pre_loss_values = [float(loss.detach().cpu().item()) for loss in pre_losses]
        if len(pre_loss_values) != 7 or not all(math.isfinite(value) for value in pre_loss_values):
            raise FloatingPointError(f"nonfinite loss during shadow reconstruction batch {batch_index}")
        pre_losses[0].backward()
        grad_finite, grad_norm, first_bad_gradient, grad_norm_finite = runner._lean_monitor_gradient_summary(torch, model)
        if not grad_finite:
            raise FloatingPointError(f"nonfinite gradient during shadow reconstruction batch {batch_index}")
        optimizer.step()
        optimizer_bad = runner._first_nonfinite_optimizer_state(torch, optimizer)
        params_finite, first_bad_parameter = runner._lean_monitor_finite_named_tensors(
            torch, [(name, value) for name, value in model.state_dict().items()])
        bn_finite, first_bad_bn = runner._lean_monitor_finite_named_tensors(
            torch, runner._lean_monitor_batchnorm_named_tensors(model))
        if optimizer_bad is not None or not params_finite or not bn_finite:
            raise FloatingPointError(f"nonfinite state during shadow reconstruction batch {batch_index}")
        preceding_batches.append({"batch_index": batch_index, "condition_keys": batch_keys,
            "loss_names": ["raw_total_loss", "mse_x1", "mse_x2", "mse_pert", "kld_x1", "kld_x2", "kld_pert"],
            "loss_values": pre_loss_values, "gradient_elements_finite": grad_finite,
            "gradient_norm": runner.safe_number(grad_norm), "gradient_norm_finite": grad_norm_finite,
            "first_nonfinite_gradient": first_bad_gradient, "optimizer_finite": optimizer_bad is None,
            "parameters_finite": params_finite, "first_nonfinite_parameter": first_bad_parameter,
            "batchnorm_finite": bn_finite, "first_nonfinite_batchnorm": first_bad_bn})
    if target_batch is None or target_rng is None:
        raise RuntimeError("could not reconstruct original epoch-212 batch 3")
    recorded_rng_json = json.dumps(context["rng_before_batch"], sort_keys=True)
    replayed_rng_json = json.dumps(target_rng, sort_keys=True)
    rng_matches_failed_batch = recorded_rng_json == replayed_rng_json
    if len(preceding_batches) != 2:
        raise RuntimeError("epoch-212 replay did not perform exactly its first two shadow updates")

    # Inspect the target forward and loss decomposition once on the shadow state.
    model.train()
    optimizer.zero_grad(set_to_none=True)
    x1, x2, features = target_batch[0].to(device), target_batch[1].to(device), target_batch[2].to(device)
    x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred = model.forward(x1, features)
    z2, mu2, logvar2 = model.encode_x2(x2)
    x2_rec = model.decode_x2(z2)
    losses = model.loss(x1, x1_rec, mu1, logvar1, x2, x2_rec, mu2, logvar2,
                        x2_pred, mu_pred, logvar_pred)
    loss_names = ["raw_total_loss", "mse_x1", "mse_x2", "mse_pert", "kld_x1", "kld_x2", "kld_pert"]
    loss_values = [float(value.detach().cpu().item()) for value in losses]

    delta_mu = mu_pred - mu2
    squared_delta = delta_mu.square()
    exp_logvar_pred = torch.exp(logvar_pred)
    denominator = torch.exp(logvar2)
    numerator = squared_delta + exp_logvar_pred
    ratio = numerator / denominator
    logvar_difference = logvar_pred - logvar2
    bracket = 1.0 + logvar_difference - ratio
    kld_elements = -0.5 * bracket
    decomposition = {
        "formula": "-0.5 * sum(1 + (logvar_pred-logvar2) - ((mu_pred-mu2)^2 + exp(logvar_pred))/exp(logvar2))",
        "mu_pred": tensor_summary(torch, mu_pred), "logvar_pred": tensor_summary(torch, logvar_pred),
        "mu2": tensor_summary(torch, mu2), "logvar2": tensor_summary(torch, logvar2),
        "squared_mu_residual": tensor_summary(torch, squared_delta),
        "exp_logvar_pred": tensor_summary(torch, exp_logvar_pred),
        "numerator": tensor_summary(torch, numerator), "exp_logvar2_denominator": tensor_summary(torch, denominator),
        "denominator_zero_n": int((denominator == 0).sum().item()),
        "denominator_subnormal_n": int(((denominator > 0) & (denominator < torch.finfo(denominator.dtype).tiny)).sum().item()),
        "ratio": tensor_summary(torch, ratio), "logvar_difference": tensor_summary(torch, logvar_difference),
        "bracket": tensor_summary(torch, bracket), "kld_element": tensor_summary(torch, kld_elements),
        "ratio_nonfinite_cause_counts": {
            "numerator_nonfinite_n": int((~torch.isfinite(numerator)).sum().item()),
            "denominator_nonfinite_n": int((~torch.isfinite(denominator)).sum().item()),
            "zero_denominator_n": int((denominator == 0).sum().item()),
            "inf_over_inf_n": int((torch.isinf(numerator) & torch.isinf(denominator)).sum().item()),
            "finite_operands_ratio_nonfinite_n": int((torch.isfinite(numerator) & torch.isfinite(denominator)
                & ~torch.isfinite(ratio)).sum().item()),
        },
        "first_nonfinite_indices": {
            "logvar_pred_exp": first_nonfinite_indices(torch, exp_logvar_pred),
            "denominator": first_nonfinite_indices(torch, denominator),
            "numerator": first_nonfinite_indices(torch, numerator),
            "ratio": first_nonfinite_indices(torch, ratio),
            "kld_element": first_nonfinite_indices(torch, kld_elements),
        },
    }
    bn_after_forward, _ = runner._trainpath_bn_snapshot(torch, model, "shadow_after_forward")

    gradient_report: dict[str, Any] = {"backward_attempted": False, "optimizer_step_applied": False}
    if all(math.isfinite(value) for value in loss_values):
        gradient_report["backward_attempted"] = True
        losses[0].backward()
        finite, norm, first_bad, norm_finite = runner._lean_monitor_gradient_summary(torch, model)
        gradient_report.update({"gradient_elements_finite": finite, "gradient_norm": runner.safe_number(norm),
                                "gradient_norm_finite": norm_finite, "first_nonfinite_gradient": first_bad})
        if finite:
            optimizer.step()
            gradient_report["optimizer_step_applied"] = True
            gradient_report["post_step_optimizer_first_bad"] = runner._first_nonfinite_optimizer_state(torch, optimizer)
    else:
        gradient_report["reason"] = "nonfinite forward loss; backward/optimizer update skipped to preserve safe shadow semantics"

    # Canonical upstream losses must reproduce the recorded failed batch within float tolerance.
    recorded_losses = context.get("loss_values", [])
    comparable = len(recorded_losses) == len(loss_values) and len(loss_values) == 7
    loss_comparison = []
    if comparable:
        for name, observed, recorded in zip(loss_names, loss_values, recorded_losses):
            rec = float(recorded) if recorded is not None else math.nan
            both_finite = math.isfinite(observed) and math.isfinite(rec)
            delta = abs(observed - rec) if both_finite else None
            loss_comparison.append({"name": name, "replay": runner.safe_number(observed),
                "recorded": runner.safe_number(rec), "abs_difference": runner.safe_number(delta) if delta is not None else None,
                "match_1e-5": bool(delta is not None and delta <= 1e-5 * max(1.0, abs(rec)))})

    bn_final, _ = runner._trainpath_bn_snapshot(torch, model, "shadow_final")
    params_finite, first_bad_parameter = runner._lean_monitor_finite_named_tensors(
        torch, [(name, value) for name, value in model.state_dict().items()])
    optimizer_bad = runner._first_nonfinite_optimizer_state(torch, optimizer)
    checkpoint_sha_after = sha256(PARENT_CHECKPOINT)
    elapsed_wall = time.perf_counter() - started_wall
    elapsed_cpu = time.process_time() - started_cpu
    budget_check()
    result = {
        "status": "COMPLETED_ONE_BOUNDED_TRAIN_ONLY_SHADOW_REPLAY",
        "experiment_id": "EXP-059", "run_id": RUN_ID, "phase": "mechanism_replay",
        "started_at_utc": started_utc, "closed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runner_sha256": sha256(RUNNER), "diagnostic_script_sha256": sha256(Path(__file__).resolve()),
        "config_sha256": sha256(cfg_path), "source_checkpoint": str(PARENT_CHECKPOINT.relative_to(ROOT)),
        "source_checkpoint_sha256_before": checkpoint_sha_before,
        "source_checkpoint_sha256_after": checkpoint_sha_after,
        "parent_checkpoint_immutable": checkpoint_sha_before == checkpoint_sha_after == EXPECTED_PARENT_SHA,
        "parent_epoch": int(payload["epoch"]), "parent_learning_rate": payload.get("learning_rate"),
        "restored_model_exact": bool(model_state_exact), "restored_optimizer_exact": bool(optimizer_state_exact),
        "restored_checkpoint_rng_exact": restored_rng, "original_failure_epoch": 212, "original_failure_batch_index": 3,
        "preceding_shadow_batches": preceding_batches,
        "condition_keys_match_original_context": requested_keys == [str(x) for x in context["condition_keys"]],
        "rng_before_batch3_matches_failure_context": rng_matches_failed_batch,
        "condition_count": int(len(requested_keys)), "source_objects": int(len(set(selected_objects))),
        "condition_keys": requested_keys, "source_object_ids": selected_objects,
        "loss_names": loss_names, "loss_values": [runner.safe_number(value) for value in loss_values],
        "original_loss_values": context.get("loss_values"), "canonical_loss_readback": loss_comparison,
        "kld_pert_decomposition": decomposition, "batchnorm_before": bn_before,
        "batchnorm_after_forward": bn_after_forward, "batchnorm_final": bn_final,
        "shadow_backward_and_update": gradient_report,
        "shadow_model_parameters_finite": bool(params_finite), "first_bad_shadow_parameter": first_bad_parameter,
        "shadow_optimizer_first_bad": optimizer_bad,
        "data_gate": evidence, "train_conditions_read": 419, "inner_dev_numeric_rows_loaded": False,
        "pdo_outer_locked_panacea_numeric_values_loaded": False, "cuda_initialized": False, "gpu_hours": 0.0,
        "cpu_wall_seconds": elapsed_wall, "cpu_process_seconds": elapsed_cpu,
        "cpu_wall_cap_seconds": MAX_WALL_SECONDS,
        "interpretation": "single corrected diagnostic replay of the raw kld_pert numerical path; it does not establish that learning rate alone caused the training divergence",
    }
    result_path = out_dir / "mechanism_replay_result.json"
    runner.atomic_json(result_path, result)
    print(json.dumps({"status": result["status"], "cpu_wall_seconds": elapsed_wall,
        "cpu_process_seconds": elapsed_cpu, "loss_values": result["loss_values"],
        "kld_summary": {key: decomposition[key] for key in
            ("exp_logvar_pred", "exp_logvar2_denominator", "ratio", "kld_element")},
        "cause_counts": decomposition["ratio_nonfinite_cause_counts"],
        "loss_match": loss_comparison, "checkpoint_unchanged": result["parent_checkpoint_immutable"],
        "result_path": str(result_path), "result_sha256": sha256(result_path)},
        ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
