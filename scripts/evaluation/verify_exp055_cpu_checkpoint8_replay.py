#!/usr/bin/env python3
"""Independent readback and CPU preflight for the EXP-055 checkpoint-8 resume.

Purpose: validate the immutable, read-only replay of TRAIN[192:256], its exact
checkpoint/RNG restoration, and the cumulative CPU cost gate before a new fit.
Reuse barrier: the existing EXP-055 verifier is bound to the closed epoch-9
failure output and cannot validate latent-layer replay or create a resume gate.
Retention: preserve as the root verification entrypoint for this replay run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = ROOT / "artifacts/experiments/EXP-055"
REPLAY_RUN_ID = "20261007_exp055_transigen_checkpoint8_replay_v1"
RESUME_RUN_ID = "20261007_exp055_transigen_cpu_resume_v1"
REPLAY_ATTEMPT = EXP_ROOT / REPLAY_RUN_ID / "fit_attempt_001"
RESUME_RUN = EXP_ROOT / RESUME_RUN_ID
SOURCE_RUN_ID = "20261007_exp055_transigen_cpu_representative_v1"
SOURCE_ATTEMPT = EXP_ROOT / SOURCE_RUN_ID / "fit_attempt_001"
SOURCE_CHECKPOINT = SOURCE_ATTEMPT / "TranSiGen_last_finite.pt"
SOURCE_CHECKPOINT_SHA256 = "266fc19656503362a94349e1cdb16e033c3fc755d78b20fe58010225302a9805"
REPLAY_RUNNER_SHA256 = "6d07a1051f81ddab286f5e2cfc14572c905f95067c9043f151e2d801f1e47e33"
SOURCE_KEYS = ROOT / "artifacts/experiments/EXP-051/20261006_exp051_contract_fit_smoke_v1/response_blind_conditions.tsv"
RUNNER = ROOT / "scripts/modeling/exp051_contract_fit_smoke.py"
CONFIG = ROOT / "configs/experiments/EXP-055-transigen-cpu-resume.yaml"
OLD_COST_GATE = SOURCE_ATTEMPT / "cpu_cost_gate.json"
ROOT_RECEIPT = REPLAY_ATTEMPT / "root_independent_readback_v2.json"
PREFLIGHT_RECEIPT = RESUME_RUN / "cpu_resume_preflight_receipt.json"
PRIOR_FAILED_ROOT_PREFLIGHT_WALL_ALLOWANCE_SECONDS = 10.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def tensor_tree_counts_and_finiteness(value: Any) -> tuple[int, int, list[str]]:
    tensor_count = 0
    floating_count = 0
    nonfinite: list[str] = []

    def visit(node: Any, label: str) -> None:
        nonlocal tensor_count, floating_count
        if torch.is_tensor(node):
            tensor_count += 1
            if node.is_floating_point() or node.is_complex():
                floating_count += 1
                if not bool(torch.isfinite(node).all()):
                    nonfinite.append(label)
        elif isinstance(node, dict):
            for key, child in node.items():
                visit(child, f"{label}.{key}")
        elif isinstance(node, (list, tuple)):
            for index, child in enumerate(node):
                visit(child, f"{label}[{index}]")

    visit(value, "checkpoint")
    return tensor_count, floating_count, nonfinite


def independent_rng_restore_check(saved: dict[str, Any]) -> dict[str, bool]:
    """Restore and compare RNGs without calling the training runner's helper."""
    if set(saved) != {"python", "numpy", "torch_cpu", "dataloader_generator"}:
        raise ValueError("checkpoint RNG stream set changed")
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1)
    generator_before = gen.get_state().clone()
    try:
        random.setstate(tuple(saved["python"]))
        ns = saved["numpy"]
        np.random.set_state((str(ns["bit_generator"]), np.asarray(ns["keys"], dtype=np.uint32),
                             int(ns["position"]), int(ns["has_gauss"]), float(ns["cached_gaussian"])))
        torch.set_rng_state(saved["torch_cpu"].cpu())
        gen.set_state(saved["dataloader_generator"].cpu())
        actual_np = np.random.get_state()
        actual_py = random.getstate()
        return {
            "python": actual_py == tuple(saved["python"]),
            "numpy": (actual_np[0] == ns["bit_generator"]
                     and np.array_equal(actual_np[1], np.asarray(ns["keys"], dtype=np.uint32))
                     and actual_np[2] == int(ns["position"])
                     and actual_np[3] == int(ns["has_gauss"])
                     and actual_np[4] == float(ns["cached_gaussian"])),
            "torch_cpu": torch.equal(torch.get_rng_state(), saved["torch_cpu"].cpu()),
            "dataloader_generator": torch.equal(gen.get_state(), saved["dataloader_generator"].cpu()),
        }
    finally:
        random.setstate(py_before)
        np.random.set_state(np_before)
        torch.set_rng_state(torch_before)
        gen.set_state(generator_before)


def main() -> int:
    started = time.process_time()
    if ROOT_RECEIPT.exists() or PREFLIGHT_RECEIPT.exists():
        raise FileExistsError("root readback/preflight already exists; this immutable replay cannot be repeated")
    if RESUME_RUN.exists() and any(RESUME_RUN.iterdir()):
        raise FileExistsError(f"resume run directory is not empty: {RESUME_RUN}")
    replay = read_json(REPLAY_ATTEMPT / "checkpoint8_replay_result.json")
    closed = read_json(REPLAY_ATTEMPT / "checkpoint8_replay_close_receipt.json")
    prior_failure = read_json(SOURCE_ATTEMPT / "first_failure_context.json")
    source_payload = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=True)
    checkpoint_hash_before = sha256(SOURCE_CHECKPOINT)
    train_meta = pd.read_csv(SOURCE_KEYS, sep="\t", dtype=str, keep_default_na=False)
    train_meta = train_meta.loc[train_meta.split_role.eq("train"),
        ["full_condition_key", "source_object_id", "full_inchikey"]].reset_index(drop=True)
    eval_batch = replay["eval_replay"]
    train_batch = replay["train_forward_only_replay"]
    prior_failure_batch = prior_failure.get("main_delta_failure", {})
    expected_batch = train_meta.iloc[192:256]
    expected = {
        "full_condition_key": expected_batch.full_condition_key.astype(str).tolist(),
        "source_object_id": expected_batch.source_object_id.astype(str).tolist(),
        "full_inchikey": expected_batch.full_inchikey.astype(str).tolist(),
    }
    eval_actual = {"full_condition_key": eval_batch.get("batch_condition_keys"),
                   "source_object_id": eval_batch.get("batch_source_object_ids"),
                   "full_inchikey": eval_batch.get("batch_full_inchikeys")}
    train_actual = {"full_condition_key": train_batch.get("batch_condition_keys"),
                    "source_object_id": train_batch.get("batch_source_object_ids"),
                    "full_inchikey": train_batch.get("batch_full_inchikeys")}

    model_tensor_n, model_float_n, model_nonfinite = tensor_tree_counts_and_finiteness(source_payload["state_dict"])
    optimizer_tensor_n, optimizer_float_n, optimizer_nonfinite = tensor_tree_counts_and_finiteness(
        source_payload["optimizer_state_dict"]["state"])
    rng_check = independent_rng_restore_check(source_payload["rng_state"])
    delta_summary = eval_batch.get("main_delta") or {}
    eval_nf = int(delta_summary.get("nonfinite_n", -1))
    eval_numel = int(delta_summary.get("denominator", -1))
    train_forward = train_batch.get("training_forward") or {}
    std_summaries = [record.get("exp_half_logvar_std") for record in train_batch.get("latent_records", [])]
    first_bad_stage = (eval_batch.get("first_nonfinite_layer")
                       or train_forward.get("first_nonfinite_layer")
                       or train_batch.get("first_nonfinite_layer"))
    checkpoint_hash_after = sha256(SOURCE_CHECKPOINT)
    old_cost = read_json(OLD_COST_GATE)
    cpu_charge_before = float(replay.get("cpu_wall_seconds", math.inf))
    prior_closed_cpu_seconds = 92.741
    replay_cpu_cap = 600.0
    projected_schedule = float(old_cost.get("projected_fixed300_with_20pct_margin_seconds", math.inf))
    cpu_cap_total = 7200.0
    current_preflight_process_seconds = time.process_time() - started
    cumulative_projection = (prior_closed_cpu_seconds + cpu_charge_before
        + PRIOR_FAILED_ROOT_PREFLIGHT_WALL_ALLOWANCE_SECONDS
        + current_preflight_process_seconds + projected_schedule)

    checks = {
        "run_attempt_and_cpu_only_replay_closed": (
            replay.get("status") == "PASS_READ_ONLY_CHECKPOINT8_TRAIN_BATCH_REPLAY"
            and replay.get("run_id") == "20261007_exp055_transigen_checkpoint8_replay_v1"
            and replay.get("attempt_id") == "fit_attempt_001"
            and replay.get("cuda_used") is False and replay.get("gpu_hours") == 0.0
            and closed.get("status") == replay.get("status") and closed.get("gpu_hours") == 0.0),
        "runner_config_and_source_checkpoint_hashes_match": (
            replay.get("runner_sha256") == REPLAY_RUNNER_SHA256
            and replay.get("config_sha256") == sha256(CONFIG)
            and replay.get("source_checkpoint_sha256_before") == SOURCE_CHECKPOINT_SHA256
            and replay.get("source_checkpoint_sha256_after") == SOURCE_CHECKPOINT_SHA256
            and checkpoint_hash_before == checkpoint_hash_after == SOURCE_CHECKPOINT_SHA256),
        "frozen_epoch8_model_optimizer_states_are_finite": (
            source_payload.get("epoch") == 8
            and source_payload.get("checkpoint_role") == "last_finite_completed_TRAIN_epoch"
            and model_tensor_n == 51 and optimizer_tensor_n == 108
            and not model_nonfinite and not optimizer_nonfinite),
        "root_independent_rng_restore_exact": all(rng_check.values()),
        "training_eval_batch_exact_identity_rows_192_255": (
            len(expected_batch) == 64 and all(eval_actual[k] == expected[k] for k in expected)
            and all(train_actual[k] == expected[k] for k in expected)
            and prior_failure_batch.get("batch_start") == 192
            and prior_failure_batch.get("batch_stop") == 256
            and len(set(expected["full_condition_key"])) == 64),
        "domain_and_forbidden_access_contract": (
            len(train_meta) == 419 and train_meta.source_object_id.nunique() == 110
            and train_meta.full_inchikey.nunique() == 5
            and replay.get("train_conditions") == 419 and replay.get("train_objects") == 110
            and replay.get("full_inchikeys") == 5 and replay.get("genes") == 978
            and replay.get("inner_dev_numeric_rows_loaded") is False
            and replay.get("pdo_outer_locked_values_opened") is False),
        "replay_eval_full_denominator_accounting": (
            eval_numel == 62592 and 0 <= eval_nf <= eval_numel
            and int(delta_summary.get("finite_n", -1)) + eval_nf == eval_numel),
        "training_replay_is_forward_only_with_bn_rng_state_restored": (
            train_forward.get("optimizer_update_count") == 0
            and train_forward.get("bn_exact_rollback_pass") is True
            and train_batch.get("model_state_unchanged_or_rollback_verified") is True
            and replay.get("optimizer_updates") == 0
            and replay.get("source_checkpoint_modified") is False),
        "latent_and_layer_diagnostics_present": (
            len(eval_batch.get("latent_records", [])) == 3
            and len(train_batch.get("latent_records", [])) == 3
            and all(all(name in record for name in ("mu", "logvar", "exp_half_logvar_std", "epsilon", "sample"))
                    for record in eval_batch.get("latent_records", []) + train_batch.get("latent_records", []))
            and all(isinstance(x, dict) and "nonfinite_n" in x for x in std_summaries)),
        "training_mode_replay_main_and_losses_finite": (
            int(train_forward.get("main_delta", {}).get("nonfinite_n", -1)) == 0
            and int(train_forward.get("main_delta", {}).get("denominator", -1)) == 62592
            and len(train_forward.get("losses", [])) == 7
            and all(int(item.get("nonfinite_n", -1)) == 0 and int(item.get("denominator", -1)) == 1
                    for item in train_forward.get("losses", []))),
        "cpu_time_and_resume_schedule_fit_cap": (
            float(replay.get("cpu_wall_seconds", math.inf)) <= replay_cpu_cap
            and math.isfinite(projected_schedule) and projected_schedule > 0
            and cumulative_projection <= cpu_cap_total
            and old_cost.get("status") == "PASS_FIXED300_WITHIN_CPU_CAP"),
    }
    root_status = "PASS_ROOT_READ_ONLY_REPLAY_ARTIFACT_AND_STATE" if all(checks.values()) else "FAIL_ROOT_READ_ONLY_REPLAY_ARTIFACT_AND_STATE"
    root_receipt = {
        "status": root_status,
        "run_id": replay.get("run_id"), "attempt_id": replay.get("attempt_id"),
        "checks": checks,
        "replay_result_sha256": sha256(REPLAY_ATTEMPT / "checkpoint8_replay_result.json"),
        "close_receipt_sha256": sha256(REPLAY_ATTEMPT / "checkpoint8_replay_close_receipt.json"),
        "replay_runner_sha256": replay.get("runner_sha256"),
        "current_resume_runner_sha256": sha256(RUNNER), "config_sha256": sha256(CONFIG),
        "source_checkpoint_sha256": checkpoint_hash_after,
        "source_checkpoint_state": {"epoch": source_payload.get("epoch"),
            "model_tensor_count": model_tensor_n, "model_floating_tensor_count": model_float_n,
            "model_nonfinite_tensor_count": len(model_nonfinite),
            "optimizer_state_tensor_count": optimizer_tensor_n,
            "optimizer_floating_tensor_count": optimizer_float_n,
            "optimizer_nonfinite_tensor_count": len(optimizer_nonfinite), "rng_restore": rng_check},
        "identity_domain": {"train_conditions": len(train_meta),
            "train_objects": int(train_meta.source_object_id.nunique()),
            "full_inchikeys": int(train_meta.full_inchikey.nunique()), "genes": 978,
            "replayed_batch_rows": [192, 256], "replayed_batch_conditions": len(expected_batch),
            "inner_dev_numeric_rows_opened": 0, "pdo_outer_locked_values_opened": False},
        "diagnosis": {"eval_first_nonfinite_layer": eval_batch.get("first_nonfinite_layer"),
            "first_nonfinite_stage": first_bad_stage,
            "eval_first_nonfinite_layer": eval_batch.get("first_nonfinite_layer"),
            "train_forward_first_nonfinite_layer": train_forward.get("first_nonfinite_layer"),
            "train_forward_latent_exp_half_logvar_std": std_summaries,
            "eval_main_delta_nonfinite": eval_nf, "eval_main_delta_denominator": eval_numel,
            "eval_main_delta_finite_n": int(delta_summary.get("finite_n", -1)),
            "eval_main_delta_finite_max_abs": delta_summary.get("finite_max_abs"),
            "prior_failure_eval_nonfinite_n": prior_failure_batch.get("nonfinite_n"),
            "prior_failure_eval_denominator": prior_failure_batch.get("numel"),
            "optimizer_updates": 0, "bn_rollback_exact": train_forward.get("bn_exact_rollback_pass")},
        "cpu_resources": {"replay_wall_seconds": replay.get("cpu_wall_seconds"),
            "replay_process_seconds": replay.get("cpu_process_seconds"),
            "prior_failed_root_readback_wall_allowance_seconds": PRIOR_FAILED_ROOT_PREFLIGHT_WALL_ALLOWANCE_SECONDS,
            "cpu_cap_seconds": cpu_cap_total, "gpu_hours": 0.0},
        "read_scope": "checkpoint tensor states, checkpoint replay JSON, and response-blind TRAIN identity fields only; no expression target arrays or inner-dev/PDO/outer/locked numeric values",
    }

    elapsed_process = time.process_time() - started
    preflight_status = "PASS_CPU_EXACT_RESUME_PREFLIGHT" if all(checks.values()) else "FAIL_CPU_EXACT_RESUME_PREFLIGHT"
    preflight = {"status": preflight_status,
        "run_id": RESUME_RUN_ID, "attempt_id": "fit_attempt_001",
        "runner_sha256": sha256(RUNNER), "replay_runner_sha256": REPLAY_RUNNER_SHA256,
        "config_sha256": sha256(CONFIG),
        "source_checkpoint_sha256": checkpoint_hash_after,
        "replay_result_sha256": root_receipt["replay_result_sha256"],
        "root_readback_sha256": None,
        "original_fixed300_cost_gate_path": str(OLD_COST_GATE.relative_to(ROOT)).replace("\\", "/"),
        "original_fixed300_cost_gate_sha256": sha256(OLD_COST_GATE),
        "original_fixed300_projected_seconds_with_margin": projected_schedule,
        "prior_closed_cpu_charged_seconds": prior_closed_cpu_seconds,
        "checkpoint_replay_cpu_wall_seconds": cpu_charge_before,
        "prior_failed_root_readback_wall_allowance_seconds": PRIOR_FAILED_ROOT_PREFLIGHT_WALL_ALLOWANCE_SECONDS,
        "root_preflight_cpu_process_seconds": elapsed_process,
        "cpu_process_seconds": elapsed_process,
        "cumulative_cpu_projection_seconds": (prior_closed_cpu_seconds + cpu_charge_before
            + PRIOR_FAILED_ROOT_PREFLIGHT_WALL_ALLOWANCE_SECONDS + elapsed_process + projected_schedule),
        "cpu_total_cap_seconds": cpu_cap_total,
        "cpu_remaining_seconds_before_resume": (cpu_cap_total - prior_closed_cpu_seconds - cpu_charge_before
            - PRIOR_FAILED_ROOT_PREFLIGHT_WALL_ALLOWANCE_SECONDS - elapsed_process),
        "gpu_hours": 0.0, "cuda_used": False,
        "authorized_schedule": {"resume_epoch": 8, "first_epoch": 9, "fixed_diagnostic_epoch": 100,
            "fixed_endpoint_epoch": 300, "inner_dev_access": "after frozen epoch 300 only"},
        "checks": checks,
    }
    if all(checks.values()):
        # Write root receipt first; the preflight binds its final bytes.
        write_json_new(ROOT_RECEIPT, root_receipt)
        preflight["root_readback_sha256"] = sha256(ROOT_RECEIPT)
        write_json_new(PREFLIGHT_RECEIPT, preflight)
    else:
        # Preserve a failed root readback, but do not create a preflight that could authorize training.
        write_json_new(ROOT_RECEIPT, root_receipt)
    print(json.dumps({"root_readback": root_receipt, "cpu_resume_preflight": preflight},
                     indent=2, sort_keys=True, allow_nan=False))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
