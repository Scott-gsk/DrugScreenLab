#!/usr/bin/env python3
"""Build and independently read the fresh CPU-only preflight for the sole EXP-059 fit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20261008_exp059_transigen_lr1e4_bindingfix_v2"
RUN = ROOT / "artifacts/experiments/EXP-059" / RUN_ID
CONFIG = ROOT / "configs/experiments/EXP-059-transigen-lr-stability.yaml"
RUNNER = ROOT / "scripts/modeling/exp051_contract_fit_smoke.py"
PRIOR_FAILED_RUN_ID = "20261007_exp059_transigen_lr1e4_fixed300_v1"
PREVIOUS_BINDINGFIX_RUN_ID = "20261008_exp059_transigen_lr1e4_bindingfix_v1"
PREVIOUS_BINDINGFIX_FAILURE_SHA = "6320001d316e761b1035aea151bc7622e4046204503116881fae3234d51e6418"
PREVIOUS_BINDINGFIX_ROOT_SHA = "2a42d809abd77797d2475960958f7606cd8c6defc4bd4072c8e0cda78e88391b"
PRE_REPAIR_RUNNER_SHA = "c15bf8bd04265c772369e4affde0b6c669e70abd87675854e497a5fb855ffb03"
PRE_REPAIR_CONFIG_SHA = "4642c48ad2a8d459fd826ef7842018258df140aad861f193d906dc338beb8b89"
PRIOR_CPU = 1582.999912906989
MECHANISM_CLOSE_SHA = "154309323e746a505683dd18da7cfa4e9c1c526ee4832a6b988e3d3f68780e43"
MECHANISM_ROOT_SHA = "a59d5e4eeb29fc172f7dd7abc98f806d3dcdaaf85b0952dd19cd16c764cd093b"
TOTAL_CPU_CAP = 7200.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def resolve_parent_evidence(cfg: dict) -> tuple[Path, str]:
    authorization = cfg.get("authorization", {})
    relative = authorization.get("mechanism_replay_checkpoint")
    expected_sha = authorization.get("mechanism_replay_checkpoint_sha256")
    if not isinstance(relative, str) or not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("EXP-059 config lacks a valid parent evidence path and SHA-256")
    parent = (ROOT / relative).resolve()
    allowed = (ROOT / "artifacts/experiments/EXP-055").resolve()
    if allowed not in parent.parents or not parent.is_file():
        raise FileNotFoundError("parent evidence must be an existing EXP-055 artifact")
    actual = sha256(parent)
    if actual != expected_sha:
        raise RuntimeError(f"config-bound parent evidence hash mismatch: {expected_sha} != {actual}")
    return parent, actual


def verify_preserved_provenance() -> dict:
    root = RUN / "repair_provenance/pre_fix"
    manifest_path = root / "provenance.json"
    runner_copy = root / "exp051_contract_fit_smoke.py"
    config_copy = root / "EXP-059-transigen-lr-stability.yaml"
    source_failure = ROOT / "artifacts/experiments/EXP-059" / PRIOR_FAILED_RUN_ID / "fit_attempt_001/fit_entry_failure.json"
    bindingfix_run = ROOT / "artifacts/experiments/EXP-059" / PREVIOUS_BINDINGFIX_RUN_ID
    bindingfix_failure = bindingfix_run / "formal_entry_failure_before_attempt.json"
    bindingfix_root = bindingfix_run / "root_independent_readback_pre_attempt_failure.json"
    bindingfix_root_preflight = bindingfix_run / "root_entry_preflight_readback.json"
    if not all(path.is_file() for path in (
            manifest_path, runner_copy, config_copy, source_failure, bindingfix_failure,
            bindingfix_root, bindingfix_root_preflight)):
        raise FileNotFoundError("pre-repair provenance or prior immutable failure receipt is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    checks = {
        "runner_snapshot_sha_matches_manifest": sha256(runner_copy) == manifest.get("source_runner_sha256"),
        "config_snapshot_sha_matches_manifest": sha256(config_copy) == manifest.get("source_config_sha256"),
        "pre_v2_snapshot_source_run_matches_manifest": manifest.get("source_run_id") == PREVIOUS_BINDINGFIX_RUN_ID,
        "pre_v2_runner_identity_matches_v1": manifest.get("source_runner_sha256") == PRE_REPAIR_RUNNER_SHA,
        "pre_v2_config_identity_matches_v1": manifest.get("source_config_sha256") == PRE_REPAIR_CONFIG_SHA,
        "prior_failure_receipt_still_present": source_failure.is_file(),
        "bindingfix_v1_failure_hash_matches": sha256(bindingfix_failure) == PREVIOUS_BINDINGFIX_FAILURE_SHA,
        "bindingfix_v1_root_hash_matches": sha256(bindingfix_root) == PREVIOUS_BINDINGFIX_ROOT_SHA,
        "bindingfix_v1_attempt_was_never_created": not (bindingfix_run / "fit_attempt_001").exists(),
        "bindingfix_v1_root_preflight_passed": read_json(bindingfix_root_preflight).get("status") ==
            "PASS_ROOT_EXP059_PREFLIGHT_READBACK",
    }
    if not all(checks.values()):
        raise RuntimeError(f"pre-repair provenance verification failed: {checks}")
    return {"status": "PASS_PRE_REPAIR_PROVENANCE_PRESERVED", "checks": checks,
            "runner_sha256": manifest["source_runner_sha256"],
            "config_sha256": manifest["source_config_sha256"],
            "previous_bindingfix_failure_sha256": sha256(bindingfix_failure),
            "previous_bindingfix_root_readback_sha256": sha256(bindingfix_root)}


def assert_no_fit_process() -> None:
    result = subprocess.run(["ps", "-eo", "args="], check=True, capture_output=True, text=True)
    needle = f"--run-id {RUN_ID}"
    for command in result.stdout.splitlines():
        if "exp051_contract_fit_smoke.py" in command and "--phase fit" in command and needle in command:
            raise RuntimeError(f"an EXP-059 fit process is already active: {command[:300]}")


def validate_static() -> tuple[dict, str, str]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty")
    if not all(path.is_file() for path in (CONFIG, RUNNER,
            RUN / "cpu_entry_contract_check.json", RUN / "cpu_contract_checkpoint.pt")):
        raise FileNotFoundError("EXP-059 preflight source or current-run contract check is missing")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if (cfg.get("experiment_id") != "EXP-059" or cfg.get("run_id") != RUN_ID
            or cfg.get("resource", {}).get("cpu_total_cap_seconds") != TOTAL_CPU_CAP
            or cfg.get("resource", {}).get("gpu_hours_cap") != 0
            or cfg.get("model", {}).get("learning_rate") != 0.0001
            or cfg.get("authorization", {}).get("prior_cpu_charged_seconds") != PRIOR_CPU
            or cfg.get("resource", {}).get("cumulative_prior_cpu_seconds") != PRIOR_CPU):
        raise ValueError("registered EXP-059 configuration differs from the single authorized CPU candidate")
    if (cfg.get("authorization", {}).get("previous_bindingfix_run_id") != PREVIOUS_BINDINGFIX_RUN_ID
            or cfg.get("authorization", {}).get("previous_bindingfix_failure_sha256") != PREVIOUS_BINDINGFIX_FAILURE_SHA):
        raise ValueError("current EXP-059 run does not bind the immutable pre-attempt failure it supersedes")
    parent_path, parent_sha = resolve_parent_evidence(cfg)
    evidence_run_id = cfg.get("authorization", {}).get("mechanism_replay_evidence_run_id")
    if evidence_run_id != PRIOR_FAILED_RUN_ID:
        raise ValueError("mechanism evidence must remain tied to the immutable prior EXP-059 diagnostic run")
    mechanism = ROOT / "artifacts/experiments/EXP-059" / evidence_run_id / "mechanism_replay_corrected_001"
    close_path = mechanism / "replay_close_receipt.json"
    root_path = mechanism / "root_independent_readback.json"
    if not all(path.is_file() for path in (close_path, root_path)):
        raise FileNotFoundError("immutable mechanism replay evidence is missing")
    provenance = verify_preserved_provenance()
    close, root = read_json(close_path), read_json(root_path)
    if (sha256(close_path) != MECHANISM_CLOSE_SHA
            or close.get("source_checkpoint_sha256_after") != parent_sha
            or close.get("source_checkpoint_sha256_before") != parent_sha
            or close.get("cuda_initialized") is not False
            or close.get("inner_dev_numeric_rows_loaded") is not False
            or close.get("pdo_outer_locked_panacea_numeric_values_loaded") is not False
            or sha256(root_path) != MECHANISM_ROOT_SHA
            or root.get("status") != "PASS_ROOT_ARTIFACT_AND_CHECKPOINT_READBACK"
            or root.get("parent_checkpoint_sha256") != parent_sha
            or root.get("corrected_replay_close_receipt_sha256") != MECHANISM_CLOSE_SHA):
        raise RuntimeError("bounded mechanism replay or root readback does not match its immutable evidence")
    if RUN.exists() and (RUN / "fit_attempt_001").exists():
        raise FileExistsError("EXP-059 fresh run already has fit_attempt_001; do not overwrite or retry")
    assert_no_fit_process()
    cfg["_resolved_parent_evidence_path"] = str(parent_path)
    cfg["_resolved_parent_evidence_sha256"] = parent_sha
    cfg["_verified_pre_repair_provenance"] = provenance
    cfg["_mechanism_evidence_root"] = str(mechanism)
    return cfg, sha256(CONFIG), sha256(RUNNER)


def contract_readback() -> dict:
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    cfg, config_sha, runner_sha = validate_static()
    parent_path, parent_sha = resolve_parent_evidence(cfg)
    mechanism = Path(cfg["_mechanism_evidence_root"])
    receipt_path = RUN / "cpu_entry_contract_check.json"
    checkpoint_path = RUN / "cpu_contract_checkpoint.pt"
    receipt = read_json(receipt_path)
    if (receipt.get("status") != "PASS_CPU_ENTRY_SHAPE_SAVE_AND_FAILURE_PROPAGATION_NO_TRAIN_UPDATE"
            or receipt.get("experiment_id") != "EXP-059"
            or receipt.get("run_id") != RUN_ID or receipt.get("attempt_id") != "fit_attempt_001"
            or receipt.get("runner_sha256") != runner_sha or receipt.get("config_sha256") != config_sha
            or receipt.get("cuda_visible_devices") != "" or receipt.get("torch_cuda_initialized") is not False
            or receipt.get("input_gate", {}).get("inner_dev_numeric_rows_loaded") is not False
            or receipt.get("input_gate", {}).get("pdo_outer_locked_values_opened") is not False
            or receipt.get("optimizer_updates") != 0
            or receipt.get("model_state_unchanged") is not True
            or receipt.get("initialization_mode") != "fresh_from_registered_seed"
            or receipt.get("parent_checkpoint_loaded_into_candidate") is not False
            or receipt.get("parent_checkpoint_evidence_path") != str(parent_path.relative_to(ROOT))
            or receipt.get("parent_checkpoint_evidence_sha256") != parent_sha
            or receipt.get("checkpoint_weights_only_readback") != "PASS_EXACT"
            or receipt.get("injected_write_failure_propagated") is not True):
        raise RuntimeError("CPU model/save contract receipt does not pass its registered no-update checks")
    import torch
    torch.set_num_threads(1)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (saved.get("status") != "CPU_CONTRACT_TEST_ONLY_NO_TRAIN_UPDATE"
            or saved.get("optimizer_updates") != 0 or saved.get("epoch") != 0
            or saved.get("device") != "cpu"):
        raise RuntimeError("safe weights-only CPU contract checkpoint fields differ")
    parent_sha_after = sha256(parent_path)
    if parent_sha_after != parent_sha:
        raise RuntimeError("config-bound parent evidence changed during the no-load CPU contract check")
    elapsed_wall = time.perf_counter() - started_wall
    elapsed_cpu = time.process_time() - started_cpu
    output = RUN / "root_cpu_contract_check_readback.json"
    if output.exists():
        raise FileExistsError(f"immutable root contract receipt already exists: {output}")
    receipt_out = {"status": "PASS_ROOT_EXP059_CPU_CONTRACT_READBACK",
        "experiment_id": "EXP-059", "run_id": RUN_ID, "attempt_id": "fit_attempt_001",
        "runner_sha256": runner_sha, "config_sha256": config_sha,
        "cpu_contract_check_sha256": sha256(receipt_path),
        "cpu_contract_checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_loaded_weights_only": True, "optimizer_updates": 0,
        "parent_checkpoint_path": str(parent_path.relative_to(ROOT)),
        "parent_checkpoint_sha256_before_after": [parent_sha, parent_sha_after],
        "parent_checkpoint_loaded_into_candidate": False,
        "mechanism_close_sha256": sha256(mechanism / "replay_close_receipt.json"),
        "mechanism_root_readback_sha256": sha256(mechanism / "root_independent_readback.json"),
        "inner_dev_numeric_rows_loaded": False, "pdo_outer_locked_values_opened": False,
        "cuda_visible_devices": "", "cuda_initialized": False, "gpu_hours": 0.0,
        "cpu_process_seconds": elapsed_cpu, "cpu_wall_seconds": elapsed_wall,
        "conservative_cpu_charge_seconds": 60.0}
    output.write_text(json.dumps(receipt_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt_out


def build_preflight() -> dict:
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    cfg, config_sha, runner_sha = validate_static()
    parent_path, parent_sha = resolve_parent_evidence(cfg)
    mechanism = Path(cfg["_mechanism_evidence_root"])
    contract_path = RUN / "cpu_entry_contract_check.json"
    checkpoint_path = RUN / "cpu_contract_checkpoint.pt"
    root_contract_path = RUN / "root_cpu_contract_check_readback.json"
    if not root_contract_path.is_file():
        raise FileNotFoundError("root independent contract-check readback is required")
    contract, root_contract = read_json(contract_path), read_json(root_contract_path)
    if (root_contract.get("status") != "PASS_ROOT_EXP059_CPU_CONTRACT_READBACK"
            or root_contract.get("cpu_contract_check_sha256") != sha256(contract_path)
            or root_contract.get("cpu_contract_checkpoint_sha256") != sha256(checkpoint_path)
            or root_contract.get("runner_sha256") != runner_sha
            or root_contract.get("config_sha256") != config_sha
            or root_contract.get("parent_checkpoint_sha256_before_after") != [parent_sha, parent_sha]
            or root_contract.get("parent_checkpoint_path") != str(parent_path.relative_to(ROOT))
            or root_contract.get("parent_checkpoint_loaded_into_candidate") is not False):
        raise RuntimeError("root readback of the no-update CPU contract check is stale or invalid")
    if (RUN / "fit_attempt_001").exists() or (RUN / "exp059_fresh_preflight.json").exists():
        raise FileExistsError("formal fit attempt or fresh preflight already exists; do not reopen")
    prior_cpu = float(cfg["authorization"]["prior_cpu_charged_seconds"])
    contract_wall = float(contract["contract_check_cpu_wall_seconds"])
    root_contract_charge = max(60.0, float(root_contract["cpu_wall_seconds"]))
    elapsed_preflight_cpu = time.process_time() - started_cpu
    elapsed_preflight_wall = time.perf_counter() - started_wall
    preflight_close_allowance = 10.0
    root_preflight_charge = 60.0
    fit_startup_allowance = 30.0
    preflight_attempt_and_repair_allowance = 10.0
    cumulative_before_fit = (prior_cpu + contract_wall + root_contract_charge
        + max(elapsed_preflight_cpu, elapsed_preflight_wall) + preflight_close_allowance
        + root_preflight_charge + fit_startup_allowance + preflight_attempt_and_repair_allowance)
    if cumulative_before_fit >= TOTAL_CPU_CAP:
        raise TimeoutError("EXP-059 total CPU envelope is exhausted before formal fitting")
    output = RUN / "exp059_fresh_preflight.json"
    if output.exists():
        raise FileExistsError(f"immutable fresh preflight already exists: {output}")
    payload = {"status": "PASS_EXP059_FRESH_PREFLIGHT",
        "experiment_id": "EXP-059", "run_id": RUN_ID, "attempt_id": "fit_attempt_001",
        "runner_sha256": runner_sha, "config_sha256": config_sha,
        "preflight_script_sha256": sha256(Path(__file__).resolve()),
        "cpu_contract_check_sha256": sha256(contract_path),
        "cpu_contract_checkpoint_sha256": sha256(checkpoint_path),
        "cpu_contract_root_readback_sha256": sha256(root_contract_path),
        "mechanism_replay_close_sha256": sha256(mechanism / "replay_close_receipt.json"),
        "mechanism_replay_root_readback_sha256": sha256(mechanism / "root_independent_readback.json"),
        "mechanism_replay_evidence_run_id": cfg["authorization"]["mechanism_replay_evidence_run_id"],
        "parent_checkpoint_path": str(parent_path.relative_to(ROOT)),
        "parent_checkpoint_sha256": parent_sha,
        "parent_checkpoint_loaded_into_candidate": False,
        "prior_cpu_charged_seconds": prior_cpu,
        "cpu_charge_components_seconds": {"previous_package_and_replay_charges": prior_cpu,
            "no_update_contract_check_actual_wall": contract_wall,
            "root_contract_check_readback_conservative_charge": root_contract_charge,
            "fresh_preflight_actual_cpu_or_wall": max(elapsed_preflight_cpu, elapsed_preflight_wall),
            "preflight_close_allowance": preflight_close_allowance,
            "root_fresh_preflight_readback_allowance": root_preflight_charge,
            "formal_fit_process_startup_allowance": fit_startup_allowance,
            "preflight_attempt_and_repair_allowance": preflight_attempt_and_repair_allowance},
        "cumulative_cpu_before_fit_seconds": cumulative_before_fit,
        "remaining_cpu_budget_before_fit_seconds": TOTAL_CPU_CAP-cumulative_before_fit,
        "cpu_total_cap_seconds": TOTAL_CPU_CAP, "gpu_hours_cap": 0.0,
        "device": "cpu", "cuda_visible_devices": "", "cuda_used": False,
        "optimizer_updates": 0, "inner_dev_numeric_rows_loaded": False,
        "pdo_outer_locked_values_opened": False,
        "formal_training_started": False,
        "pre_repair_provenance": cfg["_verified_pre_repair_provenance"],
        "training_contract": {"fresh_init": True, "source_run_id": "20261006_exp051_contract_fit_smoke_v1",
            "train_conditions": 419, "train_objects": 110, "full_inchikeys": 5, "genes": 978,
            "learning_rate": 0.0001, "fixed_epochs": 300, "diagnostic_epoch": 100}}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def root_preflight_readback() -> dict:
    cfg, config_sha, runner_sha = validate_static()
    parent_path, parent_sha = resolve_parent_evidence(cfg)
    preflight_path = RUN / "exp059_fresh_preflight.json"
    contract_path = RUN / "cpu_entry_contract_check.json"
    checkpoint_path = RUN / "cpu_contract_checkpoint.pt"
    root_contract_path = RUN / "root_cpu_contract_check_readback.json"
    if not all(path.is_file() for path in (preflight_path, contract_path, checkpoint_path, root_contract_path)):
        raise FileNotFoundError("fresh preflight or no-update contract evidence missing")
    p, c, rc = read_json(preflight_path), read_json(contract_path), read_json(root_contract_path)
    checks = {
        "preflight_status": p.get("status") == "PASS_EXP059_FRESH_PREFLIGHT",
        "runner_and_config_bound": p.get("runner_sha256") == runner_sha and p.get("config_sha256") == config_sha,
        "contract_receipt_bound": p.get("cpu_contract_check_sha256") == sha256(contract_path),
        "contract_checkpoint_bound": p.get("cpu_contract_checkpoint_sha256") == sha256(checkpoint_path),
        "root_contract_readback_pass": rc.get("status") == "PASS_ROOT_EXP059_CPU_CONTRACT_READBACK",
        "mechanism_close_and_root_bound": (p.get("mechanism_replay_close_sha256") == MECHANISM_CLOSE_SHA
            and p.get("mechanism_replay_root_readback_sha256") == MECHANISM_ROOT_SHA),
        "config_bound_parent_checkpoint_immutable": (
            p.get("parent_checkpoint_path") == str(parent_path.relative_to(ROOT))
            and p.get("parent_checkpoint_sha256") == parent_sha and sha256(parent_path) == parent_sha),
        "parent_evidence_not_loaded_into_fresh_candidate": (
            p.get("parent_checkpoint_loaded_into_candidate") is False
            and c.get("parent_checkpoint_loaded_into_candidate") is False
            and c.get("initialization_mode") == "fresh_from_registered_seed"),
        "cost_and_device_within_cap": (p.get("cumulative_cpu_before_fit_seconds", TOTAL_CPU_CAP) < TOTAL_CPU_CAP
            and p.get("remaining_cpu_budget_before_fit_seconds", -1) > 0
            and p.get("gpu_hours_cap") == 0 and p.get("cuda_used") is False),
        "no_forbidden_values_loaded": (p.get("inner_dev_numeric_rows_loaded") is False
            and p.get("pdo_outer_locked_values_opened") is False),
        "no_optimizer_updates": p.get("optimizer_updates") == 0 and c.get("optimizer_updates") == 0,
        "fit_attempt_not_created": not (RUN / "fit_attempt_001").exists()}
    if not all(checks.values()):
        raise RuntimeError(f"root fresh preflight readback failed: {checks}")
    output = RUN / "root_entry_preflight_readback.json"
    if output.exists():
        raise FileExistsError(f"immutable root fresh preflight readback already exists: {output}")
    payload = {"status": "PASS_ROOT_EXP059_PREFLIGHT_READBACK",
        "experiment_id": "EXP-059", "run_id": RUN_ID, "attempt_id": "fit_attempt_001",
        "preflight_sha256": sha256(preflight_path), "runner_sha256": runner_sha,
        "preflight_script_sha256": p.get("preflight_script_sha256"),
        "config_sha256": config_sha, "cpu_contract_check_sha256": sha256(contract_path),
        "cpu_contract_root_readback_sha256": sha256(root_contract_path),
        "cpu_contract_checkpoint_sha256": sha256(checkpoint_path),
        "parent_checkpoint_path": str(parent_path.relative_to(ROOT)),
        "parent_checkpoint_sha256": parent_sha,
        "parent_checkpoint_loaded_into_candidate": False, "checks": checks,
        "cumulative_cpu_before_fit_seconds": p["cumulative_cpu_before_fit_seconds"],
        "remaining_cpu_budget_before_fit_seconds": p["remaining_cpu_budget_before_fit_seconds"],
        "inner_dev_numeric_rows_loaded": False, "pdo_outer_locked_values_opened": False,
        "cuda_used": False, "gpu_hours": 0.0}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["contract-readback", "build-preflight", "root-preflight-readback"], required=True)
    args = parser.parse_args()
    if args.phase == "contract-readback":
        result = contract_readback()
    elif args.phase == "build-preflight":
        result = build_preflight()
    else:
        result = root_preflight_readback()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
