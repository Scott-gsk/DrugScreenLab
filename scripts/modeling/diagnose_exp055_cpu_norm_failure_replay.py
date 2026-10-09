#!/usr/bin/env python3
"""Bounded TRAIN-only replay of the immutable EXP-055 epoch-211 gradient check.

This diagnostic resumes a copy of the epoch-210 checkpoint for four real TRAIN
batches, records actual gradient elements separately from diagnostic norms, and
never writes to the source checkpoint. All Python execution is WSL2 Conda only.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20261007_exp055_cpu_norm_shadow_v2"
ATTEMPT_ID = "diag_attempt_001"
SOURCE_RUN = "20261007_exp055_transigen_cpu_resume_v1"
SOURCE_ATTEMPT = "fit_attempt_001"
SOURCE_CHECKPOINT_REL = Path("artifacts/experiments/EXP-055") / SOURCE_RUN / SOURCE_ATTEMPT / "TranSiGen_last_finite.pt"
SOURCE_CHECKPOINT_SHA256 = "9708047c6679ccc073bf18dee1ff63900a07052c3139a267ed4d0a7b2c2e3326"
FAILURE_CONTEXT_REL = Path("artifacts/experiments/EXP-055") / SOURCE_RUN / SOURCE_ATTEMPT / "first_failure_context.json"
CONFIG_REL = Path("configs/experiments/EXP-055-transigen-cpu-norm-recovery.yaml")
RUNNER_REL = Path("scripts/modeling/exp051_contract_fit_smoke.py")
CAP_SECONDS = 600.0


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("exp051_contract_fit_smoke_norm_replay", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load the registered TranSiGen runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tensor_summary(torch: Any, name: str, value: Any) -> dict[str, Any]:
    tensor = value.detach().cpu()
    flat = tensor.reshape(-1)
    finite_mask = torch.isfinite(flat)
    finite = flat[finite_mask]
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite_n": int(finite_mask.sum().item()),
        "denominator": int(flat.numel()),
        "nonfinite_n": int(flat.numel() - finite_mask.sum().item()),
        "finite_min": float(finite.min().item()) if finite.numel() else None,
        "finite_max": float(finite.max().item()) if finite.numel() else None,
        "finite_max_abs": float(finite.abs().max().item()) if finite.numel() else None,
        "first_nonfinite_positions": torch.nonzero(~finite_mask, as_tuple=False)[:12].reshape(-1).tolist(),
    }


def state_summary(torch: Any, mapping: dict[str, Any]) -> dict[str, Any]:
    rows = []
    total = finite = 0
    for name, value in mapping.items():
        if not torch.is_tensor(value) or not value.is_floating_point():
            continue
        summary = tensor_summary(torch, name, value)
        rows.append(summary)
        total += summary["denominator"]
        finite += summary["finite_n"]
    return {"tensor_count": len(rows), "finite_n": finite, "denominator": total,
            "nonfinite_n": total - finite, "tensors": rows}


def optimizer_tensor_summary(torch: Any, optimizer: Any) -> dict[str, Any]:
    rows = []
    total = finite = 0
    for parameter_index, state in enumerate(optimizer.state.values()):
        for name, value in state.items():
            if not torch.is_tensor(value) or not value.is_floating_point():
                continue
            summary = tensor_summary(torch, f"parameter_{parameter_index}.{name}", value)
            rows.append(summary)
            total += summary["denominator"]
            finite += summary["finite_n"]
    return {"tensor_count": len(rows), "finite_n": finite, "denominator": total,
            "nonfinite_n": total - finite, "tensors": rows}


def old_float32_norm(torch: Any, named_grads: list[tuple[str, Any]]) -> dict[str, Any]:
    # Reproduce the pre-fix diagnostic accumulation exactly in float32.
    squared = torch.tensor(0.0, dtype=torch.float32)
    for _, grad in named_grads:
        squared = squared + torch.sum(grad.detach().float() * grad.detach().float(), dtype=torch.float32)
    norm = torch.sqrt(squared)
    squared_value = float(squared.item())
    norm_value = float(norm.item())
    return {"squared_sum": squared_value if math.isfinite(squared_value) else None,
            "norm": norm_value if math.isfinite(norm_value) else None,
            "overflow_or_nonfinite": not math.isfinite(norm_value),
            "finite": bool(torch.isfinite(norm).item()),
            "accumulator_dtype": str(squared.dtype)}


def gradient_elements(torch: Any, runner: Any, model: Any) -> dict[str, Any]:
    named = [(name, parameter.grad.detach()) for name, parameter in model.named_parameters()
             if parameter.grad is not None]
    n_total = n_finite = 0
    max_abs = 0.0
    first_bad = None
    rows = []
    for name, grad in named:
        flat = grad.reshape(-1)
        finite_mask = torch.isfinite(flat)
        finite_n = int(finite_mask.sum().item())
        n_total += int(flat.numel())
        n_finite += finite_n
        finite_values = flat[finite_mask]
        local_max = float(finite_values.abs().max().item()) if finite_values.numel() else None
        if local_max is not None:
            max_abs = max(max_abs, local_max)
        bad_n = int(flat.numel()) - finite_n
        positions = torch.nonzero(~finite_mask, as_tuple=False)[:12].reshape(-1).tolist()
        if bad_n and first_bad is None:
            first_bad = {"parameter": name, "nonfinite_n": bad_n,
                         "positions": positions, "shape": list(grad.shape)}
        rows.append({"parameter": name, "finite_n": finite_n,
                     "denominator": int(flat.numel()), "nonfinite_n": bad_n,
                     "finite_max_abs": local_max, "first_nonfinite_positions": positions})
    return {"gradient_tensor_count": len(named), "finite_n": n_finite,
            "denominator": n_total, "nonfinite_n": n_total - n_finite,
            "finite_max_abs": max_abs if n_finite else None,
            "first_nonfinite_parameter": first_bad,
            "old_float32_norm": old_float32_norm(torch, named),
            "safe_float64_norm": runner._float64_global_l2_norm(
                torch, [grad for _, grad in named]),
            "per_parameter": rows}


def bn_snapshot(torch: Any, model: Any) -> dict[str, Any]:
    layers = []
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            entry = {"module": module_name, "num_batches_tracked": int(module.num_batches_tracked.item())}
            for name in ("running_mean", "running_var"):
                value = getattr(module, name)
                entry[name] = tensor_summary(torch, f"{module_name}.{name}", value)
            layers.append(entry)
    return {"layer_count": len(layers), "layers": layers}


def rng_fingerprints(torch: Any, generator: Any) -> dict[str, str]:
    import hashlib
    py_bytes = repr(__import__("random").getstate()).encode("utf-8")
    np_state = np.random.get_state()
    np_bytes = np_state[1].tobytes() + str(np_state[0:1] + np_state[2:]).encode("utf-8")
    return {"python_sha256": hashlib.sha256(py_bytes).hexdigest(),
            "numpy_sha256": hashlib.sha256(np_bytes).hexdigest(),
            "torch_cpu_sha256": hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
            "dataloader_sha256": hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()}


def main() -> int:
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("norm shadow requires CUDA_VISIBLE_DEVICES empty")
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    torch.set_num_threads(1)
    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError("CPU norm shadow must not initialize CUDA")

    source_checkpoint = ROOT / SOURCE_CHECKPOINT_REL
    failure_context_path = ROOT / FAILURE_CONTEXT_REL
    runner_path = ROOT / RUNNER_REL
    config_path = ROOT / CONFIG_REL
    if not config_path.is_file():
        raise FileNotFoundError(f"registered config does not exist: {config_path}")
    if sha256(source_checkpoint) != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("epoch-210 source checkpoint SHA mismatch")
    source_sha_before = sha256(source_checkpoint)
    failure = json.loads(failure_context_path.read_text(encoding="utf-8"))
    if (failure.get("status") != "FIRST_NONFINITE_TRAIN_GRADIENT"
            or failure.get("epoch") != 211 or failure.get("batch_index") != 4
            or failure.get("last_finite_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256):
        raise RuntimeError("source failure context is not the registered epoch211/batch4 event")

    output_dir = ROOT / "artifacts/experiments/EXP-055" / RUN_ID / ATTEMPT_ID
    if output_dir.exists():
        raise FileExistsError(f"norm shadow output directory is immutable if already present: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    runner = load_module(runner_path)
    runner.configure_experiment("EXP-055")
    # Config identity is bound to the distinct formal recovery run; the shadow's
    # result directory remains separate and is managed by this script.
    runner.configure_run("20261007_exp055_transigen_cpu_norm_recovery_v1")
    runner.configure_source_run("20261006_exp051_contract_fit_smoke_v1", "EXP-051")
    cfg, config_sha = runner.load_config(config_path, "EXP-055")
    verified_gate = runner._reuse_root_verified_exp055_train_gate()
    train_keys, values, feature_by_key, data_evidence = runner._load_exp055_trainpath_train_inputs_reusing_gate(
        verified_gate, runner.SOURCE_RUN_DIR / "response_blind_conditions.tsv", runner.SOURCE_RUN_DIR)
    if (len(train_keys) != 419 or values["X"].shape != (419, 978)
            or values["X_ctl"].shape != (419, 978)
            or train_keys.source_object_id.astype(str).nunique() != 110
            or train_keys.full_inchikey.astype(str).nunique() != 5):
        raise RuntimeError("TRAIN-only data contract mismatch in norm shadow")
    feature_matrix = np.stack([feature_by_key[key] for key in train_keys.full_inchikey.astype(str)]).astype(np.float32)
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    expected_keys = {"state_dict", "optimizer_state_dict", "rng_state", "epoch", "seed",
                     "objective", "gradient_clipping", "learning_rate", "weight_decay", "checkpoint_role"}
    if (payload.get("epoch") != 210 or payload.get("seed") != runner.SEED
            or payload.get("objective") != "raw_sum" or payload.get("gradient_clipping") is not False
            or payload.get("learning_rate") != 1e-3 or payload.get("weight_decay") != 1e-5
            or payload.get("checkpoint_role") != "last_finite_completed_TRAIN_epoch"
            or not expected_keys.issubset(payload)):
        raise RuntimeError("source checkpoint protocol metadata differs")

    generator = torch.Generator().manual_seed(runner.SEED)
    dataset = TensorDataset(torch.as_tensor(values["X_ctl"], dtype=torch.float32),
                            torch.as_tensor(values["X"], dtype=torch.float32),
                            torch.as_tensor(feature_matrix, dtype=torch.float32),
                            torch.arange(419, dtype=torch.int64))
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False,
                        num_workers=0, generator=generator)
    model, _ = runner._load_transigen(torch, torch.device("cpu"), runner.SEED,
                                      features_dim=2048, seed_cuda=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    restore = runner._exp055_cpu_restore_rng_state(torch, generator, payload["rng_state"])
    if not all(restore.values()):
        raise RuntimeError("could not exactly restore checkpoint RNG")
    if sha256(source_checkpoint) != source_sha_before:
        raise RuntimeError("source checkpoint changed while loading")
    rng_before_shadow = rng_fingerprints(torch, generator)

    loader_iter = iter(loader)
    batch_records = []
    target_batch_keys = None
    target_delta_summary = None
    target_forward_trace = None
    target_grad_record = None
    target_losses = None
    all_updates_state_finite = True
    original_forward = model.forward
    original_sample_latent = model.sample_latent
    module_trace: list[dict[str, Any]] = []
    hooks = []
    active_batch = {"value": 0}
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        def hook(_module: Any, inputs: Any, output: Any, module_name: str = module_name) -> None:
            if active_batch["value"] == 4:
                outputs = output if isinstance(output, (tuple, list)) else (output,)
                for output_index, item in enumerate(outputs):
                    if torch.is_tensor(item):
                        module_trace.append(tensor_summary(torch, f"{module_name}.output_{output_index}", item))
        hooks.append(module.register_forward_hook(hook))

    latent_trace: list[dict[str, Any]] = []
    def sample_latent_trace(mu: Any, logvar: Any) -> Any:
        z = original_sample_latent(mu, logvar)
        if active_batch["value"] == 4:
            std = torch.exp(0.5 * logvar)
            latent_trace.extend([tensor_summary(torch, "latent.mu", mu),
                                 tensor_summary(torch, "latent.logvar", logvar),
                                 tensor_summary(torch, "latent.std_exp_half_logvar", std),
                                 tensor_summary(torch, "latent.sample", z)])
        return z
    model.sample_latent = sample_latent_trace

    try:
        model.train()
        for batch_index in range(1, 5):
            if time.perf_counter() - start_wall > CAP_SECONDS:
                raise TimeoutError("CPU norm shadow exceeded the 600-second hard timebox")
            active_batch["value"] = batch_index
            rows = next(loader_iter)
            x1, x2, features, row_positions = [value.to("cpu") for value in rows]
            positions = row_positions.numpy().astype(int)
            keys = train_keys.iloc[positions].full_condition_key.astype(str).tolist()
            optimizer.zero_grad(set_to_none=True)
            if batch_index == 4:
                before_param = {name: value.detach().clone() for name, value in model.named_parameters()}
                before_bn = {name: value.detach().clone() for name, value in model.named_buffers()}
                before_param_summary = state_summary(torch, before_param)
                before_opt_summary = optimizer_tensor_summary(torch, optimizer)
                before_bn_summary = state_summary(torch, before_bn)
                before_bn_layers = bn_snapshot(torch, model)
            x1_forward, x2_forward, prediction, losses = runner._transigen_forward_batch(
                torch, model, (x1, x2, features), torch.device("cpu"))
            delta = prediction - x1_forward
            loss_names = ["raw_total_loss", "mse_x1", "mse_x2", "mse_pert",
                          "kld_x1", "kld_x2", "kld_pert"]
            loss_values = [float(value.detach().cpu().item()) for value in losses]
            if len(losses) != 7 or not np.isfinite(loss_values).all():
                raise FloatingPointError(f"losses are nonfinite at shadow TRAIN batch {batch_index}")
            delta_info = tensor_summary(torch, "main_delta", delta)
            if delta_info["nonfinite_n"]:
                raise FloatingPointError(f"main delta is nonfinite at shadow TRAIN batch {batch_index}")
            losses[0].backward()
            grad_info = gradient_elements(torch, runner, model)
            if grad_info["nonfinite_n"]:
                all_updates_state_finite = False
            optimizer.step()
            param_after = {name: value.detach() for name, value in model.named_parameters()}
            buffers_after = {name: value.detach() for name, value in model.named_buffers()}
            param_summary = state_summary(torch, param_after)
            optimizer_summary = optimizer_tensor_summary(torch, optimizer)
            bn_summary = state_summary(torch, buffers_after)
            state_finite = (param_summary["nonfinite_n"] == 0
                            and optimizer_summary["nonfinite_n"] == 0
                            and bn_summary["nonfinite_n"] == 0)
            all_updates_state_finite = all_updates_state_finite and state_finite
            record = {"epoch": 211, "batch_index": batch_index,
                "condition_keys": keys,
                "source_object_ids": train_keys.iloc[positions].source_object_id.astype(str).tolist(),
                "full_inchikeys": train_keys.iloc[positions].full_inchikey.astype(str).tolist(),
                "losses": dict(zip(loss_names, loss_values)),
                "main_delta": delta_info,
                "gradient": grad_info,
                "post_update": {"parameters": param_summary, "optimizer": optimizer_summary,
                                "batchnorm_buffers": bn_summary,
                                "batchnorm_layers": bn_snapshot(torch, model),
                                "all_finite": state_finite},
                "optimizer_update_performed": True}
            batch_records.append(record)
            if batch_index == 4:
                target_batch_keys = keys
                target_delta_summary = delta_info
                target_grad_record = grad_info
                target_losses = dict(zip(loss_names, loss_values))
                parameter_diff = {}
                for name, before in before_param.items():
                    after = param_after[name]
                    diff = after.to(torch.float64) - before.to(torch.float64)
                    parameter_diff[name] = tensor_summary(torch, f"delta.{name}", diff)
                target_forward_trace = {"module_outputs": module_trace, "latent_outputs": latent_trace}
                record["batch4_before_update"] = {"parameters": before_param_summary,
                    "optimizer_state": before_opt_summary, "batchnorm_buffers": before_bn_summary}
                record["batch4_bn_layers_before_update"] = before_bn_layers
                record["batch4_parameter_differences"] = parameter_diff
                record["batch4_bn_before_update"] = bn_snapshot(torch, model)
    finally:
        model.sample_latent = original_sample_latent
        for hook_handle in hooks:
            hook_handle.remove()

    same_keys = target_batch_keys == failure.get("condition_keys")
    recorded_fail_losses = failure.get("losses", {})
    loss_matches = {name: (name in recorded_fail_losses
        and math.isclose(value, float(recorded_fail_losses[name]), rel_tol=1e-6, abs_tol=1e-4))
        for name, value in (target_losses or {}).items()}
    checkpoint_sha_after = sha256(source_checkpoint)
    first_bad_layer = next((row["name"].rsplit(".output_", 1)[0]
        for row in (target_forward_trace or {}).get("module_outputs", []) if row.get("nonfinite_n", 0)), None)
    elapsed_wall = time.perf_counter() - start_wall
    elapsed_cpu = time.process_time() - start_cpu
    status = "PASS_EXACT_TRAIN_BATCH4_ACTUAL_GRADIENT_FINITE_MONITOR_NORM_OVERFLOW" if (
        same_keys and all(loss_matches.values()) and all_updates_state_finite
        and target_grad_record and target_grad_record["nonfinite_n"] == 0
        and not target_grad_record["old_float32_norm"]["finite"]
        and target_grad_record["safe_float64_norm"] is not None
        and math.isfinite(target_grad_record["safe_float64_norm"])
        and checkpoint_sha_after == SOURCE_CHECKPOINT_SHA256 and elapsed_wall <= CAP_SECONDS
        and first_bad_layer is None) else "FAIL_SHADOW_GATE_NOT_CLEARED"
    result = {"status": status, "run_id": RUN_ID, "attempt_id": ATTEMPT_ID,
        "started_at_epoch": 211, "last_source_checkpoint_epoch": 210,
        "source_checkpoint_path": str(SOURCE_CHECKPOINT_REL),
        "source_checkpoint_sha256_before": source_sha_before,
        "source_checkpoint_sha256_after": checkpoint_sha_after,
        "source_checkpoint_immutable": checkpoint_sha_after == source_sha_before,
        "source_checkpoint_protocol": {"seed": runner.SEED, "objective": "raw_sum",
            "gradient_clipping": False, "learning_rate": 1e-3, "weight_decay": 1e-5,
            "batch_size": 64, "shuffle": True, "train_conditions": 419,
            "train_objects": 110, "full_inchikeys": 5, "genes": 978},
        "data_evidence": data_evidence,
        "failure_context_batch_identity_exact_match": same_keys,
        "failure_context_batch_loss_close_match": loss_matches,
        "replay_target_loss_values": target_losses,
        "target_batch4_gradient": target_grad_record,
        "target_batch4_main_delta": target_delta_summary,
        "target_batch4_first_nonfinite_forward_layer": first_bad_layer,
        "target_batch4_forward_trace": target_forward_trace,
        "batches": batch_records,
        "all_replayed_post_update_states_finite": all_updates_state_finite,
        "optimizer_updates": len(batch_records),
        "inner_dev_numeric_rows_loaded": False,
        "pdo_outer_locked_numeric_values_opened": False,
        "cuda_used": False, "gpu_hours": 0.0,
        "cpu_wall_seconds": elapsed_wall, "cpu_process_seconds": elapsed_cpu,
        "cpu_cap_seconds": CAP_SECONDS,
        "rng_fingerprints_before_shadow": rng_before_shadow,
        "rng_fingerprints_after_shadow": rng_fingerprints(torch, generator),
        "shadow_rng_state_reused_for_formal_fit": False,
        "runner_sha256": sha256(runner_path), "config_sha256": sha256(config_path),
        "diagnostic_logic": "Replay exact DataLoader/RNG stream and four TRAIN optimizer updates from epoch210; compare actual gradient element scan to legacy float32 squared norm and safe float64 norm."}
    result_path = output_dir / "norm_shadow_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    close = {"status": status, "closed_at_epoch": 211, "closed_after_batch": 4,
             "run_id": RUN_ID, "attempt_id": ATTEMPT_ID, "runner_sha256": result["runner_sha256"],
             "config_sha256": result["config_sha256"], "result_sha256": sha256(result_path),
             "source_checkpoint_sha256_before": source_sha_before,
             "source_checkpoint_sha256_after": checkpoint_sha_after,
             "cpu_wall_seconds": elapsed_wall, "cpu_process_seconds": elapsed_cpu,
             "gpu_hours": 0.0}
    (output_dir / "norm_shadow_close_receipt.json").write_text(
        json.dumps(close, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": status, "result_path": str(result_path),
                      "result_sha256": sha256(result_path), "cpu_wall_seconds": elapsed_wall,
                      "cpu_process_seconds": elapsed_cpu, "gpu_hours": 0.0}, indent=2), flush=True)
    return 0 if status.startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
