#!/usr/bin/env python3
"""Independent metadata-only root review of the EXP-055 CPU recovery preflight."""
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
PREFLIGHT = ROOT / "artifacts/experiments/EXP-055" / RUN_ID / "cpu_norm_recovery_preflight_receipt.json"
PREFLIGHT_FAILURE = PREFLIGHT.with_name("cpu_norm_recovery_preflight_attempt_001_failure.json")
ROOT_RECEIPT = PREFLIGHT.with_name("root_cpu_norm_recovery_preflight_readback.json")
SOURCE_CHECKPOINT = (ROOT / "artifacts/experiments/EXP-055/20261007_exp055_transigen_cpu_resume_v1/"
                     "fit_attempt_001/TranSiGen_last_finite.pt")
SOURCE_CLOSE = SOURCE_CHECKPOINT.parent / "attempt_close_receipt.json"
SOURCE_CURVE = SOURCE_CHECKPOINT.parent / "TranSiGen_training_curve.jsonl"
SOURCE_EPOCH100 = SOURCE_CHECKPOINT.parent / "TranSiGen_epoch_100.pt"
SOURCE_EPOCH100_SHA = "2aaf5691a46a820cf74613f98ea840acd2ceaa15642d925219c8759e4b92753a"
SHADOW_ROOT = (ROOT / "artifacts/experiments/EXP-055/20261007_exp055_cpu_norm_shadow_v2/"
               "diag_attempt_001/root_independent_readback.json")
SHADOW_RESULT = SHADOW_ROOT.with_name("norm_shadow_result.json")
SHADOW_CLOSE = SHADOW_ROOT.with_name("norm_shadow_close_receipt.json")
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
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> int:
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("root preflight review must run with CUDA disabled")
    inputs = (RUNNER, CONFIG, PREFLIGHT, SOURCE_CHECKPOINT, SOURCE_EPOCH100, SOURCE_CLOSE, SOURCE_CURVE,
              SHADOW_ROOT, SHADOW_RESULT, SHADOW_CLOSE)
    if not all(path.is_file() for path in inputs):
        raise FileNotFoundError("missing registered norm-recovery preflight input")
    if ROOT_RECEIPT.exists():
        raise FileExistsError("root preflight receipt exists; preserve it and use no overwrite")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    gate = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    source_close = json.loads(SOURCE_CLOSE.read_text(encoding="utf-8"))
    shadow = json.loads(SHADOW_ROOT.read_text(encoding="utf-8"))
    shadow_result = json.loads(SHADOW_RESULT.read_text(encoding="utf-8"))
    shadow_close = json.loads(SHADOW_CLOSE.read_text(encoding="utf-8"))
    curve_rows = [json.loads(line) for line in SOURCE_CURVE.read_text(encoding="utf-8").splitlines() if line.strip()]
    prior_rows = [row for row in curve_rows if 9 <= int(row.get("epoch", 0)) <= 210]
    avg_epoch_seconds = float(source_close.get("cpu_elapsed_wall_seconds", math.inf)) / len(prior_rows) if prior_rows else math.inf
    projected_per_epoch = max(5.0, avg_epoch_seconds * 1.5)
    components = {
        "remaining_90_train_evaluation_checkpoint_epochs_seconds": 90 * projected_per_epoch,
        "fixed_endpoint_and_extra_checkpoint_io_seconds": 120.0,
        "final_93_row_inner_dev_evaluation_and_prediction_roundtrip_seconds": 120.0,
        "root_independent_final_readback_and_closeout_seconds": 180.0,
        "preflight_launch_and_startup_allowance_seconds": 60.0,
    }
    schedule_before_margin = sum(components.values())
    schedule_with_margin = math.ceil(schedule_before_margin * 1.2)
    prior_cpu = float(cfg.get("authorization", {}).get("prior_cpu_charged_seconds", math.inf))
    required_checks = ("exact_parent_checkpoint_sha_before_after", "four_real_updates_recorded",
        "batch4_condition_keys_match_failed_context", "batch4_canonical_losses_match_failed_context",
        "all_replayed_grad_elements_finite", "all_replayed_post_update_model_optimizer_bn_finite",
        "legacy_float32_global_norm_overflowed", "safe_float64_global_norm_finite",
        "inner_dev_pdo_outer_locked_unopened", "cuda_not_used")
    root_shadow_ok = (shadow.get("status") == "PASS_ROOT_NORM_OVERFLOW_REPLAY_GATE"
        and shadow.get("source_checkpoint_sha256_before") == SOURCE_SHA
        and shadow.get("source_checkpoint_sha256_after") == SOURCE_SHA
        and all(shadow.get("checks", {}).get(key) is True for key in required_checks)
        and int(shadow.get("batch4_gradient_finite_n", -1)) == int(shadow.get("batch4_gradient_denominator", -2))
        and int(shadow.get("batch4_gradient_nonfinite_n", -1)) == 0
        and math.isfinite(float(shadow.get("safe_float64_norm", math.nan))))
    checks = {
        "fixed_run_and_attempt": gate.get("run_id") == RUN_ID and gate.get("attempt_id") == ATTEMPT_ID,
        "authorized_scope_and_cpu_caps": gate.get("authorization_id") == "CPU-NORM-RECOVERY-2203"
            and float(cfg.get("authorization", {}).get("cpu_total_cap_minutes", -1)) == 120.0
            and float(cfg.get("authorization", {}).get("gpu_hours_cap", -1)) == 0.0,
        "runner_and_config_freshly_bound": gate.get("runner_sha256") == sha(RUNNER)
            and gate.get("config_sha256") == sha(CONFIG),
        "fixed_epoch210_source_checkpoint_immutable": gate.get("source_checkpoint_sha256") == SOURCE_SHA
            and sha(SOURCE_CHECKPOINT) == SOURCE_SHA
            and gate.get("linked_epoch100_checkpoint_sha256") == SOURCE_EPOCH100_SHA
            and sha(SOURCE_EPOCH100) == SOURCE_EPOCH100_SHA
            and source_close.get("last_completed_finite_epoch") == 210
            and source_close.get("last_finite_checkpoint_sha256") == SOURCE_SHA,
        "root_shadow_gate_and_cost_result_inputs": gate.get("root_shadow_readback_sha256") == SHADOW_ROOT_SHA
            and sha(SHADOW_ROOT) == SHADOW_ROOT_SHA
            and gate.get("shadow_result_sha256") == sha(SHADOW_RESULT)
            and gate.get("shadow_close_sha256") == sha(SHADOW_CLOSE)
            and root_shadow_ok and shadow_result.get("source_checkpoint_sha256_after") == SOURCE_SHA
            and shadow_close.get("gpu_hours") == 0.0,
        "source_timing_window_complete": len(prior_rows) == 202 and math.isclose(avg_epoch_seconds,
            float(gate.get("observed_wall_seconds_per_completed_epoch", math.nan)), rel_tol=0, abs_tol=1e-6),
        "cost_projection_independently_recomputed": gate.get("cost_components_seconds_before_margin") == components
            and gate.get("remaining_schedule_before_margin_seconds") == schedule_before_margin
            and gate.get("remaining_schedule_with_margin_seconds") == schedule_with_margin,
        "prior_failed_preflight_preserved": gate.get("prior_failed_preflight_receipt_sha256")
            == (sha(PREFLIGHT_FAILURE) if PREFLIGHT_FAILURE.is_file() else None),
        "full_cpu_envelope_with_margin_below_cap": prior_cpu + float(gate.get("preflight_cpu_process_seconds", math.inf))
            + float(gate.get("preflight_cpu_wall_seconds", math.inf))
            + float(gate.get("root_preflight_readback_allowance_seconds", math.inf))
            + schedule_with_margin <= CPU_CAP
            and gate.get("cpu_total_cap_seconds") == CPU_CAP
            and gate.get("cumulative_projected_cpu_with_20pct_margin_seconds") <= CPU_CAP,
        "all_data_roles_and_gpu_limits_preserved": gate.get("inner_dev_numeric_rows_read") is False
            and gate.get("pdo_outer_locked_numeric_values_read") is False and gate.get("gpu_hours") == 0.0,
        "no_existing_attempt_or_overwrite": not (PREFLIGHT.parent / ATTEMPT_ID).exists()
            and not any(p.name not in {PREFLIGHT.name, PREFLIGHT_FAILURE.name} for p in PREFLIGHT.parent.iterdir())
            and (not PREFLIGHT_FAILURE.is_file() or json.loads(PREFLIGHT_FAILURE.read_text(encoding="utf-8")).get("status")
                 == "FAILED_CLOSED_PREFLIGHT_FIELD_NAME_MISMATCH_BEFORE_TRAIN_DATA_LOAD"),
    }
    cpu_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    payload = {
        "status": "PASS_ROOT_CPU_NORM_RECOVERY_PREFLIGHT" if all(checks.values()) and gate.get("status") == "PASS_CPU_NORM_RECOVERY_PREFLIGHT"
            else "FAIL_ROOT_CPU_NORM_RECOVERY_PREFLIGHT",
        "run_id": RUN_ID, "attempt_id": ATTEMPT_ID,
        "preflight_receipt_sha256": sha(PREFLIGHT),
        "prior_failed_preflight_receipt_sha256": sha(PREFLIGHT_FAILURE) if PREFLIGHT_FAILURE.is_file() else None,
        "runner_sha256": sha(RUNNER), "config_sha256": sha(CONFIG),
        "source_checkpoint_sha256": sha(SOURCE_CHECKPOINT),
        "linked_epoch100_checkpoint_sha256": sha(SOURCE_EPOCH100),
        "root_shadow_readback_sha256": sha(SHADOW_ROOT),
        "shadow_result_sha256": sha(SHADOW_RESULT), "shadow_close_sha256": sha(SHADOW_CLOSE),
        "source_attempt_avg_wall_seconds_per_epoch": avg_epoch_seconds,
        "projected_remaining_schedule_with_margin_seconds": schedule_with_margin,
        "prior_cpu_charged_seconds": prior_cpu,
        "preflight_cpu_process_seconds": gate.get("preflight_cpu_process_seconds"),
        "preflight_cpu_wall_seconds": gate.get("preflight_cpu_wall_seconds"),
        "cpu_process_seconds": cpu_seconds, "cpu_wall_seconds": wall_seconds,
        "additional_preflight_startup_allowance_seconds": 30.0,
        "cumulative_projected_cpu_with_20pct_margin_seconds": prior_cpu
            + float(gate.get("preflight_cpu_process_seconds", math.inf))
            + float(gate.get("preflight_cpu_wall_seconds", math.inf))
            + cpu_seconds + 30.0 + schedule_with_margin,
        "cpu_total_cap_seconds": CPU_CAP, "gpu_hours": 0.0,
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "source_train_only": True, "inner_dev_numeric_rows_read": False,
        "pdo_outer_locked_numeric_values_read": False,
    }
    write_json(ROOT_RECEIPT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS_ROOT_CPU_NORM_RECOVERY_PREFLIGHT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
