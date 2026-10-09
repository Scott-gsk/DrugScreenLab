#!/usr/bin/env python3
"""Fresh CPU-only, runner/config/source-bound gate for the EXP-055 norm repair."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20261007_exp055_transigen_cpu_norm_recovery_v1"
ATTEMPT_ID = "fit_attempt_001"
RUNNER = ROOT / "scripts/modeling/exp051_contract_fit_smoke.py"
CONFIG = ROOT / "configs/experiments/EXP-055-transigen-cpu-norm-recovery.yaml"
RUN_DIR = ROOT / "artifacts/experiments/EXP-055" / RUN_ID
PREFLIGHT_FAILURE = RUN_DIR / "cpu_norm_recovery_preflight_attempt_001_failure.json"
SOURCE_CHECKPOINT = (ROOT / "artifacts/experiments/EXP-055/20261007_exp055_transigen_cpu_resume_v1/"
                     "fit_attempt_001/TranSiGen_last_finite.pt")
SOURCE_CLOSE = SOURCE_CHECKPOINT.parent / "attempt_close_receipt.json"
SOURCE_CURVE = SOURCE_CHECKPOINT.parent / "TranSiGen_training_curve.jsonl"
SHADOW_ROOT = (ROOT / "artifacts/experiments/EXP-055/20261007_exp055_cpu_norm_shadow_v2/"
               "diag_attempt_001/root_independent_readback.json")
SHADOW_RESULT = SHADOW_ROOT.with_name("norm_shadow_result.json")
SHADOW_CLOSE = SHADOW_ROOT.with_name("norm_shadow_close_receipt.json")
SOURCE_EPOCH100 = SOURCE_CHECKPOINT.parent / "TranSiGen_epoch_100.pt"
SOURCE_EPOCH100_SHA = "2aaf5691a46a820cf74613f98ea840acd2ceaa15642d925219c8759e4b92753a"
SOURCE_SHA = "9708047c6679ccc073bf18dee1ff63900a07052c3139a267ed4d0a7b2c2e3326"
SHADOW_ROOT_SHA = "196ae8ec9b6440fd5ea829c84fed2fa283e348ef6eee57f4cd564fa8af413245"
CPU_CAP = 7200.0


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> int:
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty")
    if not all(path.is_file() for path in (RUNNER, CONFIG, SOURCE_CHECKPOINT, SOURCE_EPOCH100, SOURCE_CLOSE,
                                           SOURCE_CURVE, SHADOW_ROOT, SHADOW_RESULT, SHADOW_CLOSE)):
        raise FileNotFoundError("one or more registered inputs for the norm-recovery gate are absent")
    if sha(SOURCE_CHECKPOINT) != SOURCE_SHA or sha(SHADOW_ROOT) != SHADOW_ROOT_SHA:
        raise RuntimeError("immutable source checkpoint or root shadow receipt SHA mismatch")
    if sha(SOURCE_EPOCH100) != SOURCE_EPOCH100_SHA:
        raise RuntimeError("linked epoch-100 checkpoint from the same resume lineage changed")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if (config.get("run_id") != RUN_ID or config.get("resource", {}).get("run_id") != RUN_ID
            or config.get("authorization", {}).get("user_scope") != "CPU-NORM-RECOVERY-2203"
            or config["authorization"].get("training_device_policy") != "cpu_only_zero_gpu"
            or float(config["authorization"].get("cpu_total_cap_minutes", -1)) != 120.0
            or float(config["authorization"].get("gpu_hours_cap", -1)) != 0.0
            or config["model"].get("resume_from_absolute_epoch") != 210
            or config["model"].get("max_epochs") != 300):
        raise RuntimeError("norm-recovery configuration no longer matches the approved fixed-300 CPU protocol")
    source_close = json.loads(SOURCE_CLOSE.read_text(encoding="utf-8"))
    shadow_root = json.loads(SHADOW_ROOT.read_text(encoding="utf-8"))
    shadow_result = json.loads(SHADOW_RESULT.read_text(encoding="utf-8"))
    shadow_close = json.loads(SHADOW_CLOSE.read_text(encoding="utf-8"))
    required_checks = ("exact_parent_checkpoint_sha_before_after", "four_real_updates_recorded",
        "batch4_condition_keys_match_failed_context", "batch4_canonical_losses_match_failed_context",
        "all_replayed_grad_elements_finite", "all_replayed_post_update_model_optimizer_bn_finite",
        "legacy_float32_global_norm_overflowed", "safe_float64_global_norm_finite",
        "inner_dev_pdo_outer_locked_unopened", "cuda_not_used")
    if (source_close.get("status") != "FAILED_CLOSED_CPU_ATTEMPT"
            or source_close.get("last_completed_finite_epoch") != 210
            or source_close.get("last_finite_checkpoint_sha256") != SOURCE_SHA
            or source_close.get("inner_dev_numeric_rows_loaded") is not False
            or source_close.get("pdo_outer_locked_values_opened") is not False
            or shadow_root.get("status") != "PASS_ROOT_NORM_OVERFLOW_REPLAY_GATE"
            or shadow_root.get("source_checkpoint_sha256_before") != SOURCE_SHA
            or shadow_root.get("source_checkpoint_sha256_after") != SOURCE_SHA
            or not all(shadow_root.get("checks", {}).get(key) is True for key in required_checks)
            or shadow_result.get("source_checkpoint_sha256_before") != SOURCE_SHA
            or shadow_result.get("source_checkpoint_sha256_after") != SOURCE_SHA
            or shadow_close.get("gpu_hours") != 0.0):
        raise RuntimeError("source attempt or shadow replay does not satisfy the approved resume gate")
    if sha(SOURCE_CHECKPOINT) != SOURCE_SHA:
        raise RuntimeError("source checkpoint changed during metadata checks")
    run_attempt_dir = RUN_DIR / ATTEMPT_ID
    if run_attempt_dir.exists():
        raise FileExistsError("formal attempt directory already exists and is immutable")
    if RUN_DIR.exists():
        unexpected = [path.name for path in RUN_DIR.iterdir()
                      if path.name not in {PREFLIGHT_FAILURE.name}]
        if unexpected:
            raise FileExistsError(f"norm-recovery run root contains unexpected files: {unexpected}")
        if PREFLIGHT_FAILURE.is_file():
            failure = json.loads(PREFLIGHT_FAILURE.read_text(encoding="utf-8"))
            if failure.get("status") != "FAILED_CLOSED_PREFLIGHT_FIELD_NAME_MISMATCH_BEFORE_TRAIN_DATA_LOAD":
                raise RuntimeError("unrecognized prior preflight failure; refusing to overwrite or retry")

    curve = [json.loads(line) for line in SOURCE_CURVE.read_text(encoding="utf-8").splitlines() if line.strip()]
    prior_rows = [row for row in curve if 9 <= int(row.get("epoch", 0)) <= 210]
    if len(prior_rows) != 202:
        raise RuntimeError("source CPU resume curve lacks the complete 202-epoch timing window")
    prior_attempt_wall = float(source_close["cpu_elapsed_wall_seconds"])
    epoch_wall_observed = prior_attempt_wall / len(prior_rows)
    per_epoch_projection = max(5.0, epoch_wall_observed * 1.5)
    remaining_epochs = 300 - 210
    cost_components = {
        "remaining_90_train_evaluation_checkpoint_epochs_seconds": remaining_epochs * per_epoch_projection,
        "fixed_endpoint_and_extra_checkpoint_io_seconds": 120.0,
        "final_93_row_inner_dev_evaluation_and_prediction_roundtrip_seconds": 120.0,
        "root_independent_final_readback_and_closeout_seconds": 180.0,
        "preflight_launch_and_startup_allowance_seconds": 60.0,
    }
    schedule_before_margin = sum(cost_components.values())
    remaining_with_margin = math.ceil(schedule_before_margin * 1.20)
    prior_charged = float(config["authorization"]["prior_cpu_charged_seconds"])
    preflight_process = max(1.0, time.process_time() - start_cpu + 1.0)
    preflight_wall = max(1.0, time.perf_counter() - start_wall + 1.0)
    root_readback_allowance = 30.0
    cumulative_projected = prior_charged + preflight_process + preflight_wall + root_readback_allowance + remaining_with_margin
    runner_sha, config_sha = sha(RUNNER), sha(CONFIG)
    payload = {
        "status": "PASS_CPU_NORM_RECOVERY_PREFLIGHT" if cumulative_projected <= CPU_CAP else "FAIL_CPU_COST_CAP",
        "run_id": RUN_ID, "attempt_id": ATTEMPT_ID,
        "authorization_id": "CPU-NORM-RECOVERY-2203",
        "runner_sha256": runner_sha, "config_sha256": config_sha,
        "preflight_script_sha256": sha(Path(__file__).resolve()),
        "prior_failed_preflight_receipt_sha256": sha(PREFLIGHT_FAILURE) if PREFLIGHT_FAILURE.is_file() else None,
        "source_checkpoint_path": str(SOURCE_CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "source_checkpoint_sha256": SOURCE_SHA, "resume_from_epoch": 210, "fixed_endpoint_epoch": 300,
        "linked_epoch100_checkpoint_path": str(SOURCE_EPOCH100.relative_to(ROOT)).replace("\\", "/"),
        "linked_epoch100_checkpoint_sha256": SOURCE_EPOCH100_SHA,
        "source_close_receipt_sha256": sha(SOURCE_CLOSE),
        "source_curve_sha256": sha(SOURCE_CURVE),
        "source_completed_epoch_count": len(prior_rows),
        "source_attempt_cpu_wall_seconds": prior_attempt_wall,
        "observed_wall_seconds_per_completed_epoch": epoch_wall_observed,
        "projected_wall_seconds_per_remaining_epoch": per_epoch_projection,
        "remaining_epochs": remaining_epochs,
        "cost_components_seconds_before_margin": cost_components,
        "remaining_schedule_before_margin_seconds": schedule_before_margin,
        "remaining_schedule_with_margin_seconds": remaining_with_margin,
        "margin_fraction": 0.20, "root_preflight_readback_allowance_seconds": root_readback_allowance,
        "prior_cpu_charged_seconds": prior_charged,
        "preflight_cpu_process_seconds": preflight_process,
        "preflight_cpu_wall_seconds": preflight_wall,
        "cumulative_projected_cpu_with_20pct_margin_seconds": cumulative_projected,
        "cpu_total_cap_seconds": CPU_CAP,
        "root_shadow_readback_sha256": SHADOW_ROOT_SHA,
        "shadow_result_sha256": sha(SHADOW_RESULT),
        "shadow_close_sha256": sha(SHADOW_CLOSE),
        "shadow_update_count": 4,
        "gradient_elements_finite": shadow_root["batch4_gradient_finite_n"],
        "gradient_elements_denominator": shadow_root["batch4_gradient_denominator"],
        "legacy_float32_norm_finite": False,
        "safe_float64_norm": shadow_root["safe_float64_norm"],
        "inner_dev_numeric_rows_read": False,
        "pdo_outer_locked_numeric_values_read": False,
        "gpu_hours": 0.0,
    }
    receipt = RUN_DIR / "cpu_norm_recovery_preflight_receipt.json"
    write_json(receipt, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS_CPU_NORM_RECOVERY_PREFLIGHT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
