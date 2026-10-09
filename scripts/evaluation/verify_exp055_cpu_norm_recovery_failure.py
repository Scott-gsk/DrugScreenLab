#!/usr/bin/env python3
"""Root-only independent verification of the immutable EXP-055 norm-recovery failure.

Purpose: verify the authorized CPU continuation's true non-finite training-loss
stop, last finite epoch-211 checkpoint, TRAIN identity, resource settlement, and
forbidden-data boundaries without reading inner-dev or any locked/PDO values.

Reuse barrier: the earlier EXP-055 verifiers are bound to different run IDs,
checkpoint epochs, and failure schemas. This verifier is specific to the
immutable CPU norm-recovery attempt and refuses to overwrite its receipt.
Retention: preserve beside the attempt as the root independent readback.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = ROOT / "artifacts/experiments/EXP-055"
RUN_ID = "20261007_exp055_transigen_cpu_norm_recovery_v1"
ATTEMPT_ID = "fit_attempt_001"
RUN_DIR = EXP_ROOT / RUN_ID
ATTEMPT_DIR = RUN_DIR / ATTEMPT_ID
OUTPUT = RUN_DIR / "root_independent_failure_readback.json"
RUNNER = ROOT / "scripts/modeling/exp051_contract_fit_smoke.py"
CONFIG = ROOT / "configs/experiments/EXP-055-transigen-cpu-norm-recovery.yaml"
SOURCE_RUN_ID = "20261007_exp055_transigen_cpu_resume_v1"
SOURCE_ATTEMPT = EXP_ROOT / SOURCE_RUN_ID / ATTEMPT_ID
SOURCE_CHECKPOINT = SOURCE_ATTEMPT / "TranSiGen_last_finite.pt"
TRAIN_METADATA = (ROOT / "artifacts/experiments/EXP-051/20261006_exp051_contract_fit_smoke_v1"
                  / "response_blind_conditions.tsv")
PRIOR_FAILED_PREFLIGHT = RUN_DIR / "cpu_norm_recovery_preflight_attempt_001_failure.json"
PREVIOUS_SHADOW_ROOT_SHA256 = "196ae8ec9b6440fd5ea829c84fed2fa283e348ef6eee57f4cd564fa8af413245"
PREVIOUS_SOURCE_CHECKPOINT_SHA256 = "9708047c6679ccc073bf18dee1ff63900a07052c3139a267ed4d0a7b2c2e3326"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return value


def write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable root receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def scan_tensors(value: Any) -> dict[str, Any]:
    counts = {"tensor_n": 0, "floating_tensor_n": 0, "nonfinite": []}

    def visit(node: Any, label: str) -> None:
        if torch.is_tensor(node):
            counts["tensor_n"] += 1
            if node.is_floating_point() or node.is_complex():
                counts["floating_tensor_n"] += 1
                if not bool(torch.isfinite(node).all()):
                    counts["nonfinite"].append(label)
        elif isinstance(node, dict):
            for key, child in node.items():
                visit(child, f"{label}.{key}")
        elif isinstance(node, (list, tuple)):
            for index, child in enumerate(node):
                visit(child, f"{label}[{index}]")

    visit(value, "checkpoint")
    return counts


def no_matching_fit_process() -> tuple[bool, list[str]]:
    result = subprocess.run(["ps", "-eo", "pid=,args="], check=True, text=True,
                            capture_output=True)
    matching = [line.strip() for line in result.stdout.splitlines()
                if "exp051_contract_fit_smoke.py" in line
                and RUN_ID in line and "--phase fit" in line]
    return not matching, matching


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"Immutable root readback already exists: {OUTPUT}")

    close = read_json(ATTEMPT_DIR / "attempt_close_receipt.json")
    progress = read_json(ATTEMPT_DIR / "attempt_progress.json")
    failure = read_json(ATTEMPT_DIR / "first_failure_context.json")
    manifest = read_json(ATTEMPT_DIR / "fit_smoke_manifest.json")
    inputs = read_json(ATTEMPT_DIR / "run_input_receipt.json")
    gate = read_json(ATTEMPT_DIR / "cpu_cost_gate.json")
    ledger_lines = [json.loads(line) for line in
                    (ATTEMPT_DIR / "resource_ledger.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()]
    checkpoint_path = ATTEMPT_DIR / "TranSiGen_last_finite.pt"
    checkpoint_sha = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_counts = scan_tensors(checkpoint.get("state_dict", {}))
    optimizer_counts = scan_tensors(checkpoint.get("optimizer_state_dict", {}).get("state", {}))
    bn_counts = scan_tensors({key: value for key, value in checkpoint.get("state_dict", {}).items()
                              if re.search(r"running_(mean|var)|num_batches_tracked", key)})

    metadata = pd.read_csv(TRAIN_METADATA, sep="\t", dtype=str, keep_default_na=False)
    train = metadata.loc[metadata["split_role"].str.lower().eq("train"),
                         ["full_condition_key", "source_object_id", "full_inchikey"]].copy()
    key_to_identity = train.set_index("full_condition_key").to_dict(orient="index")
    batch_keys = failure.get("condition_keys", [])
    batch_objects = failure.get("source_object_ids", [])
    batch_inchikeys = [key.split("|")[1] if len(key.split("|")) > 1 else "" for key in batch_keys]
    batch_identity_ok = (
        len(batch_keys) == 64 and len(set(batch_keys)) == 64 and len(batch_objects) == 64
        and len(batch_inchikeys) == 64
        and all(key in key_to_identity for key in batch_keys)
        and all(key_to_identity[key]["source_object_id"] == obj
                and key_to_identity[key]["full_inchikey"] == ik
                for key, obj, ik in zip(batch_keys, batch_objects, batch_inchikeys))
    )

    loss_names = failure.get("loss_names", [])
    loss_values = failure.get("loss_values", [])
    loss_map = dict(zip(loss_names, loss_values)) if len(loss_names) == len(loss_values) else {}
    finite_components = {name: value for name, value in loss_map.items()
                         if isinstance(value, (int, float)) and math.isfinite(float(value))}
    failed_components = [name for name, value in loss_map.items() if value is None]
    preflight = read_json(RUN_DIR / "cpu_norm_recovery_preflight_receipt.json")
    preflight_root = read_json(RUN_DIR / "root_cpu_norm_recovery_preflight_readback.json")
    previous_shadow_path = (EXP_ROOT / "20261007_exp055_cpu_norm_shadow_v2/diag_attempt_001"
                            / "root_independent_readback.json")
    previous_shadow = read_json(previous_shadow_path)
    previous_shadow_sha = sha256(previous_shadow_path)
    source_sha = sha256(SOURCE_CHECKPOINT)

    process_absent, matching_processes = no_matching_fit_process()
    training_curve = [json.loads(line) for line in
                      (ATTEMPT_DIR / "TranSiGen_training_curve.jsonl").read_text(encoding="utf-8").splitlines()
                      if line.strip()]
    last_curve = training_curve[-1] if training_curve else {}
    failed_batch_records = [json.loads(line) for line in
                            (ATTEMPT_DIR / "TranSiGen_training_batches.jsonl").read_text(encoding="utf-8").splitlines()
                            if line.strip()]
    epoch212_batch_records = [row for row in failed_batch_records if row.get("epoch") == 212]
    epoch212_completed_batch_indices = sorted(int(row["batch_index"]) for row in epoch212_batch_records)

    cpu_total = float(close["cpu_elapsed_total_with_contract_checks_and_allowance_seconds"])
    expected_cpu_total = (float(close["cpu_elapsed_wall_seconds"])
                          + float(close["measured_successful_contract_check_seconds"])
                          + float(gate["preflight_wall_seconds"])
                          + float(gate["root_preflight_readback_process_seconds"])
                          + float(gate["preflight_startup_allowance_seconds"])
                          + float(gate["prior_cpu_charged_seconds"]))
    checks = {
        "immutable_attempt_closed_on_true_nonfinite_training_loss": (
            close.get("run_id") == RUN_ID and close.get("attempt_id") == ATTEMPT_ID
            and close.get("status") == "FAILED_CLOSED_CPU_ATTEMPT"
            and close.get("exception_type") == "FloatingPointError"
            and close.get("exception_message") == "TranSiGen non-finite training loss at epoch 212, batch 3"
            and progress.get("phase") == "CLOSED_FAILED"
            and progress.get("exception_type") == close.get("exception_type")
            and progress.get("exception_message") == close.get("exception_message")
        ),
        "runner_config_and_source_checkpoint_chain_match": (
            sha256(RUNNER) == close.get("runner_sha256") == manifest.get("runner_sha256")
            and sha256(CONFIG) == close.get("config_sha256") == manifest.get("config_sha256")
            and source_sha == PREVIOUS_SOURCE_CHECKPOINT_SHA256
            and gate.get("source_checkpoint_sha256") == source_sha
            and manifest.get("resume_from_checkpoint_sha256") == source_sha
        ),
        "fresh_preflight_and_root_readback_pass_and_prior_shadow_is_preserved": (
            preflight.get("status") == "PASS_CPU_NORM_RECOVERY_PREFLIGHT"
            and preflight.get("run_id") == RUN_ID and preflight.get("attempt_id") == ATTEMPT_ID
            and sha256(RUN_DIR / "cpu_norm_recovery_preflight_receipt.json")
                == manifest.get("cpu_contract_check_sha256")
            and preflight_root.get("status") == "PASS_ROOT_CPU_NORM_RECOVERY_PREFLIGHT"
            and previous_shadow_sha == PREVIOUS_SHADOW_ROOT_SHA256
            and previous_shadow.get("status") == "PASS_ROOT_NORM_OVERFLOW_REPLAY_GATE"
            and PRIOR_FAILED_PREFLIGHT.exists()
        ),
        "train_domain_and_failed_batch_identity_match_frozen_rows": (
            len(train) == 419 and train["source_object_id"].nunique() == 110
            and train["full_inchikey"].nunique() == 5 and batch_identity_ok
            and failure.get("epoch") == 212 and failure.get("batch_index") == 3
        ),
        "loss_failure_component_and_batch_denominator_are_explicit": (
            failure.get("first_nonfinite_component") == "raw_total_loss"
            and loss_names == ["raw_total_loss", "mse_x1", "mse_x2", "mse_pert", "kld_x1", "kld_x2", "kld_pert"]
            and len(loss_values) == 7 and loss_map.get("raw_total_loss") is None
            and loss_map.get("kld_pert") is None
            and len(finite_components) == 5 and set(finite_components) ==
                {"mse_x1", "mse_x2", "mse_pert", "kld_x1", "kld_x2"}
            and failed_components == ["raw_total_loss", "kld_pert"]
            and failure.get("main_delta_denominator") == 62592
            and failure.get("main_delta_finite_n") == 62592
            and epoch212_completed_batch_indices == [1, 2]
            and all(row.get("gradient_elements_finite") is True
                    and row.get("gradient_norm_finite") is True
                    and row.get("parameters_finite") is True
                    and row.get("optimizer_finite") is True
                    and row.get("batchnorm_finite") is True
                    and all(isinstance(value, (int, float)) and math.isfinite(float(value))
                            for value in (row.get("losses") or {}).values())
                    for row in epoch212_batch_records)
        ),
        "epoch211_checkpoint_is_last_finite_and_weights_only_state_finite": (
            checkpoint.get("epoch") == 211
            and checkpoint.get("checkpoint_role") == "last_finite_completed_TRAIN_epoch"
            and checkpoint_sha == close.get("last_finite_checkpoint_sha256")
            and close.get("last_completed_finite_epoch") == 211
            and not model_counts["nonfinite"] and not optimizer_counts["nonfinite"]
            and model_counts["tensor_n"] > 0 and optimizer_counts["tensor_n"] > 0
            and model_counts["floating_tensor_n"] > 0 and optimizer_counts["floating_tensor_n"] > 0
            and isinstance(checkpoint.get("rng_state"), dict)
        ),
        "train_identity_and_forbidden_data_roles_remain_separated": (
            inputs.get("status") == "PASS_TRAIN_ONLY_FROZEN_INPUTS"
            and inputs.get("train_conditions") == 419 and inputs.get("train_objects") == 110
            and inputs.get("full_inchikeys") == 5 and inputs.get("genes") == 978
            and inputs.get("inner_dev_numeric_rows_loaded") is False
            and inputs.get("pdo_outer_locked_values_opened") is False
            and close.get("inner_dev_numeric_rows_loaded") is False
            and close.get("pdo_outer_locked_values_opened") is False
            and manifest.get("data_roles", {}).get("inner_dev_numeric_rows_loaded_before_checkpoint_freeze") is False
            and manifest.get("data_roles", {}).get("pdo_outer_locked_values_opened") is False
        ),
        "cpu_budget_gpu_zero_and_no_fit_process": (
            close.get("cpu_total_cap_seconds") == 7200.0
            and cpu_total <= 7200.0 and math.isclose(cpu_total, expected_cpu_total, rel_tol=0, abs_tol=1e-6)
            and close.get("gpu_hours") == 0.0 and gate.get("gpu_hours") == 0.0
            and all(entry.get("gpu_hours", 0.0) == 0.0 for entry in ledger_lines)
            and close.get("cuda_visible_devices") == "" and process_absent
        ),
        "no_fixed300_or_innerdev_performance_score_was_produced": (
            not (ATTEMPT_DIR / "TranSiGen_epoch_300_train_keyed_predictions.npz").exists()
            and not (ATTEMPT_DIR / "TranSiGen_inner_dev_predictions.npz").exists()
            and progress.get("completed_epochs") == 211
        ),
    }
    receipt = {
        "status": "PASS_ROOT_INDEPENDENT_CPU_NORM_RECOVERY_FAILURE_READBACK" if all(checks.values())
                  else "FAIL_ROOT_INDEPENDENT_CPU_NORM_RECOVERY_FAILURE_READBACK",
        "run_id": RUN_ID,
        "attempt_id": ATTEMPT_ID,
        "close_status": close.get("status"),
        "failure": {
            "exception": close.get("exception_message"),
            "epoch": failure.get("epoch"),
            "batch_index": failure.get("batch_index"),
            "first_nonfinite_component": failure.get("first_nonfinite_component"),
            "serialized_null_loss_components": failed_components,
            "finite_loss_components": finite_components,
            "main_delta_finite_n": failure.get("main_delta_finite_n"),
            "main_delta_denominator": failure.get("main_delta_denominator"),
            "failed_batch_key_count": len(batch_keys),
            "failed_batch_unique_key_count": len(set(batch_keys)),
            "failed_batch_matches_frozen_train_identity": batch_identity_ok,
        },
        "last_finite_checkpoint": {
            "epoch": checkpoint.get("epoch"),
            "sha256": checkpoint_sha,
            "model_tensor_counts": model_counts,
            "optimizer_tensor_counts": optimizer_counts,
            "batchnorm_tensor_counts": bn_counts,
        },
        "frozen_domain": {"train_conditions": len(train), "train_objects": int(train.source_object_id.nunique()),
                          "full_inchikeys": int(train.full_inchikey.nunique()), "genes": inputs.get("genes")},
        "resources": {"cpu_seconds_cumulative": cpu_total, "cpu_cap_seconds": 7200.0,
                      "formal_attempt_wall_seconds": close.get("cpu_elapsed_wall_seconds"),
                      "formal_attempt_process_seconds": close.get("cpu_process_seconds"),
                      "gpu_hours": 0.0, "fit_process_absent": process_absent,
                      "matching_processes": matching_processes},
        "scope": {"inner_dev_numeric_rows_loaded": False, "pdo_outer_locked_values_opened": False,
                  "fixed300_endpoint": False, "valid_generalization_score": False},
        "checks": checks,
    }
    if not all(checks.values()):
        print(json.dumps({"status": receipt["status"], "checks": checks,
                          "checkpoint_role": checkpoint.get("checkpoint_role"),
                          "model_tensor_counts": model_counts,
                          "optimizer_tensor_counts": optimizer_counts,
                          "epoch212_completed_batch_indices": epoch212_completed_batch_indices,
                          "preflight_root_status": preflight_root.get("status"),
                          "previous_shadow_status": previous_shadow.get("status")},
                         indent=2, sort_keys=True, allow_nan=False))
        return 1
    write_new(OUTPUT, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
