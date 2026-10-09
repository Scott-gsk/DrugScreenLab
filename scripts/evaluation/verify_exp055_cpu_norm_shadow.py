#!/usr/bin/env python3
"""Independent root readback for the bounded EXP-055 CPU gradient-norm shadow."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SHADOW = ROOT / "artifacts/experiments/EXP-055/20261007_exp055_cpu_norm_shadow_v2/diag_attempt_001"
SOURCE = ROOT / "artifacts/experiments/EXP-055/20261007_exp055_transigen_cpu_resume_v1/fit_attempt_001"
CHECKPOINT_SHA = "9708047c6679ccc073bf18dee1ff63900a07052c3139a267ed4d0a7b2c2e3326"
CANONICAL_LOSS_NAMES = ["raw_total_loss", "mse_x1", "mse_x2", "mse_pert",
                        "kld_x1", "kld_x2", "kld_pert"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_tensor_counts(value: Any, torch: Any) -> tuple[int, int, int]:
    tensor_n = finite_n = element_n = 0
    if torch.is_tensor(value):
        if value.is_floating_point():
            tensor_n += 1
            n = int(value.numel())
            element_n += n
            finite_n += int(torch.isfinite(value).sum().item())
        return tensor_n, finite_n, element_n
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (list, tuple)):
        children = value
    else:
        return 0, 0, 0
    for child in children:
        t, f, e = finite_tensor_counts(child, torch)
        tensor_n += t
        finite_n += f
        element_n += e
    return tensor_n, finite_n, element_n


def main() -> int:
    start = time.perf_counter()
    import torch
    torch.set_num_threads(1)
    result_path = SHADOW / "norm_shadow_result.json"
    close_path = SHADOW / "norm_shadow_close_receipt.json"
    failure_path = SOURCE / "first_failure_context.json"
    checkpoint_path = SOURCE / "TranSiGen_last_finite.pt"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    close = json.loads(close_path.read_text(encoding="utf-8"))
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    checkpoint_sha_before = sha256(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_sha_after = sha256(checkpoint_path)

    batch_rows = result.get("batches", [])
    batch_numbers = [row.get("batch_index") for row in batch_rows]
    batch4 = next((row for row in batch_rows if row.get("batch_index") == 4), {})
    raw_losses = list(batch4.get("losses", {}).values())
    mapped_losses = dict(zip(CANONICAL_LOSS_NAMES, raw_losses))
    loss_matches = {name: (name in failure.get("losses", {})
        and math.isclose(float(mapped_losses.get(name, math.nan)),
                         float(failure["losses"][name]), rel_tol=1e-6, abs_tol=1e-4))
        for name in CANONICAL_LOSS_NAMES}
    gradient = batch4.get("gradient", {})
    old_norm = gradient.get("old_float32_norm", {})
    state_ok = all(bool(row.get("post_update", {}).get("all_finite")) for row in batch_rows)
    gradient_ok = all(row.get("gradient", {}).get("nonfinite_n") == 0
                      and row.get("gradient", {}).get("finite_n") == row.get("gradient", {}).get("denominator")
                      for row in batch_rows)
    model_counts = finite_tensor_counts(payload.get("state_dict", {}), torch)
    optimizer_counts = finite_tensor_counts(payload.get("optimizer_state_dict", {}), torch)
    checks = {
        "shadow_closed_attempt_status_preserved": close.get("status") == "FAIL_SHADOW_GATE_NOT_CLEARED",
        "exact_parent_checkpoint_sha_before_after": checkpoint_sha_before == CHECKPOINT_SHA == checkpoint_sha_after,
        "checkpoint_epoch_seed_and_protocol": payload.get("epoch") == 210 and payload.get("seed") == 20261006
            and payload.get("objective") == "raw_sum" and payload.get("gradient_clipping") is False
            and payload.get("learning_rate") == 1e-3 and payload.get("weight_decay") == 1e-5,
        "parent_model_floating_tensors_finite": model_counts[1] == model_counts[2] and model_counts[0] > 0,
        "parent_optimizer_floating_tensors_finite": optimizer_counts[1] == optimizer_counts[2] and optimizer_counts[0] > 0,
        "four_real_updates_recorded": batch_numbers == [1, 2, 3, 4] and result.get("optimizer_updates") == 4,
        "batch4_condition_keys_match_failed_context": batch4.get("condition_keys") == failure.get("condition_keys"),
        "batch4_canonical_losses_match_failed_context": all(loss_matches.values()),
        "all_replayed_grad_elements_finite": gradient_ok,
        "all_replayed_post_update_model_optimizer_bn_finite": state_ok,
        "legacy_float32_global_norm_overflowed": old_norm.get("overflow_or_nonfinite") is True
            and old_norm.get("finite") is False and old_norm.get("norm") is None,
        "safe_float64_global_norm_finite": math.isfinite(float(gradient.get("safe_float64_norm", math.nan))),
        "shadow_is_within_600_seconds": float(result.get("cpu_wall_seconds", math.inf)) <= 600.0,
        "inner_dev_pdo_outer_locked_unopened": result.get("inner_dev_numeric_rows_loaded") is False
            and result.get("pdo_outer_locked_numeric_values_opened") is False,
        "cuda_not_used": result.get("cuda_used") is False and float(result.get("gpu_hours", math.inf)) == 0.0,
    }
    result_mapping_defect = {
        "executor_reported_status": result.get("status"),
        "reason": "executor assigned the seven forward outputs to the wrong diagnostic names; raw ordered values were preserved",
        "raw_ordered_values_reinterpreted_using_frozen_runner_loss_order": CANONICAL_LOSS_NAMES,
        "canonical_batch4_loss_values": mapped_losses,
        "canonical_loss_matches": loss_matches,
    }
    status = "PASS_ROOT_NORM_OVERFLOW_REPLAY_GATE" if all(checks.values()) else "FAIL_ROOT_NORM_OVERFLOW_REPLAY_GATE"
    receipt = {
        "status": status,
        "run_id": "20261007_exp055_cpu_norm_shadow_v2",
        "attempt_id": "diag_attempt_001",
        "executed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "executor_result_sha256": sha256(result_path),
        "executor_close_receipt_sha256": sha256(close_path),
        "failure_context_sha256": sha256(failure_path),
        "source_checkpoint_sha256_before": checkpoint_sha_before,
        "source_checkpoint_sha256_after": checkpoint_sha_after,
        "source_checkpoint_epoch": payload.get("epoch"),
        "parent_model_tensor_count_finite_denominator": model_counts,
        "parent_optimizer_tensor_count_finite_denominator": optimizer_counts,
        "batch4_gradient_finite_n": gradient.get("finite_n"),
        "batch4_gradient_denominator": gradient.get("denominator"),
        "batch4_gradient_nonfinite_n": gradient.get("nonfinite_n"),
        "batch4_gradient_finite_max_abs": gradient.get("finite_max_abs"),
        "legacy_float32_norm": old_norm,
        "safe_float64_norm": gradient.get("safe_float64_norm"),
        "replayed_post_update_states_finite": state_ok,
        "diagnostic_metadata_correction": result_mapping_defect,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "shadow_model_optimizer_rng_reused_for_formal_fit": False,
        "root_cpu_wall_seconds": time.perf_counter() - start,
        "gpu_hours": 0.0,
    }
    out = SHADOW / "root_independent_readback.json"
    if out.exists():
        raise FileExistsError(f"root receipt is immutable: {out}")
    out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": status, "receipt_path": str(out), "receipt_sha256": sha256(out),
                      "failed_checks": receipt["failed_checks"],
                      "root_cpu_wall_seconds": receipt["root_cpu_wall_seconds"],
                      "gpu_hours": 0.0}, ensure_ascii=False, indent=2))
    return 0 if status.startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
