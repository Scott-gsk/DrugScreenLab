#!/usr/bin/env python3
"""EXP-051 response-blind source selection, input-contract audit, and fit smoke.

All Python execution for this entry point is restricted to WSL2 Conda
``drugscreening-gpu``.  The metadata phase never opens AnnData X or X_ctl.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ID = "20261006_exp051_contract_fit_smoke_v1"
DEFAULT_SOURCE_RUN_ID = DEFAULT_RUN_ID
DEFAULT_EXPERIMENT_ID = "EXP-051"
EXP055_TRAINPATH_ENTRY_SMOKE_RUN_ID = "20261007_exp055_transigen_entry_smoke_v1"
EXP055_TRAINPATH_ENTRY_DIAG_RUN_ID = "20261007_exp055_transigen_entry_smoke_diag_v2"
EXP055_TRAINPATH_BATCH_REPLAY_RUN_ID = "20261007_exp055_transigen_batch_eval_replay_v1"
EXP055_TRAINPATH_LATENT_REPLAY_RUN_ID = "20261007_exp055_transigen_latent_replay_v1"
EXP055_TRAINPATH_RECOVERY_RUN_ID = "20261007_exp055_transigen_trainpath_recovery_v12"
EXP055_TRANSIGEN_LEAN_COST_RUN_ID = "20261007_exp055_transigen_lean_monitor_cost_cal_v1"
EXP055_TRANSIGEN_FIXED300_LEAN_RUN_ID = "20261007_exp055_transigen_fixed300_lean_monitor_v1"
EXP055_TRANSIGEN_CPU_REPRESENTATIVE_RUN_ID = "20261007_exp055_transigen_cpu_representative_v1"
EXP055_TRANSIGEN_CPU_REPRESENTATIVE_CONFIG = "EXP-055-transigen-cpu-representative.yaml"
EXP055_TRANSIGEN_CPU_REPLAY_RUN_ID = "20261007_exp055_transigen_checkpoint8_replay_v1"
EXP055_TRANSIGEN_CPU_REPLAY_RUNNER_SHA256 = (
    "6d07a1051f81ddab286f5e2cfc14572c905f95067c9043f151e2d801f1e47e33")
EXP055_TRANSIGEN_CPU_RESUME_RUN_ID = "20261007_exp055_transigen_cpu_resume_v1"
EXP055_TRANSIGEN_CPU_RESUME_CONFIG = "EXP-055-transigen-cpu-resume.yaml"
EXP055_TRANSIGEN_CPU_NORM_RECOVERY_RUN_ID = "20261007_exp055_transigen_cpu_norm_recovery_v1"
EXP055_TRANSIGEN_CPU_NORM_RECOVERY_CONFIG = "EXP-055-transigen-cpu-norm-recovery.yaml"
EXP059_TRANSIGEN_LR_STABILITY_RUN_ID = "20261008_exp059_transigen_lr1e4_bindingfix_v2"
EXP059_TRANSIGEN_LR_STABILITY_CONFIG = "EXP-059-transigen-lr-stability.yaml"
EXP059_CPU_RUN_ROOT_ALLOWED_ENTRIES = frozenset({
    "cpu_entry_contract_check.json", "cpu_contract_checkpoint.pt",
    "root_cpu_contract_check_readback.json", "exp059_fresh_preflight.json",
    "root_entry_preflight_readback.json", "repair_provenance",
})
EXP055_TRANSIGEN_CPU_NORM_SHADOW_RUN_ID = "20261007_exp055_cpu_norm_shadow_v1"
EXP055_TRANSIGEN_CPU_NORM_SHADOW_SHA256 = "9708047c6679ccc073bf18dee1ff63900a07052c3139a267ed4d0a7b2c2e3326"
EXP055_TRANSIGEN_CPU_NORM_SOURCE_CHECKPOINT = (
    ROOT / "artifacts/experiments/EXP-055/20261007_exp055_transigen_cpu_resume_v1/"
    "fit_attempt_001/TranSiGen_last_finite.pt")
EXP055_TRANSIGEN_CPU_NORM_SOURCE_CHECKPOINT_SHA256 = (
    "9708047c6679ccc073bf18dee1ff63900a07052c3139a267ed4d0a7b2c2e3326")
EXP055_TRANSIGEN_CPU_NORM_EPOCH100_CHECKPOINT = (
    ROOT / "artifacts/experiments/EXP-055/20261007_exp055_transigen_cpu_resume_v1/"
    "fit_attempt_001/TranSiGen_epoch_100.pt")
EXP055_TRANSIGEN_CPU_NORM_EPOCH100_CHECKPOINT_SHA256 = (
    "2aaf5691a46a820cf74613f98ea840acd2ceaa15642d925219c8759e4b92753a")
EXP055_TRANSIGEN_CPU_NORM_ROOT_SHADOW_READBACK = (
    ROOT / "artifacts/experiments/EXP-055/20261007_exp055_cpu_norm_shadow_v2/"
    "diag_attempt_001/root_independent_readback.json")
EXP055_TRANSIGEN_CPU_NORM_ROOT_SHADOW_READBACK_SHA256 = (
    "196ae8ec9b6440fd5ea829c84fed2fa283e348ef6eee57f4cd564fa8af413245")
EXP055_TRANSIGEN_CPU_PRIOR_FAILED_ROOT_PREFLIGHT_WALL_ALLOWANCE_SECONDS = 10.0
EXP055_TRANSIGEN_CPU_RESUME_SOURCE_CHECKPOINT = (
    ROOT / "artifacts/experiments/EXP-055/20261007_exp055_transigen_cpu_representative_v1/"
    "fit_attempt_001/TranSiGen_last_finite.pt")
EXP055_TRANSIGEN_CPU_RESUME_SOURCE_CHECKPOINT_SHA256 = (
    "266fc19656503362a94349e1cdb16e033c3fc755d78b20fe58010225302a9805")
EXP055_TRANSIGEN_COST_SOURCE_RUN_ID = EXP055_TRAINPATH_RECOVERY_RUN_ID
EXP055_TRANSIGEN_COST_SOURCE_ATTEMPT = "fit_attempt_001"
EXP055_TRANSIGEN_COST_SOURCE_CHECKPOINT = "TranSiGen_epoch_1_cost_gate_checkpoint.pt"
EXP055_TRANSIGEN_COST_SOURCE_CHECKPOINT_SHA256 = "30ec5444badc936779e681a64753cbc6f62966bdfd95d02467e4eb6e550f2c69"
EXP055_TRAINPATH_PRELAUNCH_FAILURE_RUN_IDS = {
    "20261007_exp055_transigen_trainpath_recovery_v7",
    "20261007_exp055_transigen_trainpath_recovery_v8",
    EXP055_TRAINPATH_RECOVERY_RUN_ID,
}
EXP055_TRAINPATH_MONITORED_300_RUN_ID = "20261007_exp055_transigen_monitored_fixed300_v1"
EXP055_DIAG_V2_CHECKPOINT_SHA256 = "ba9f056b32b8ce56acc0be5020887a8ccd7e0e0c1b3e2b2de180f00bcd11d17c"
EXPERIMENT_ID = DEFAULT_EXPERIMENT_ID
EXPERIMENT_ROOT = ROOT / "artifacts/experiments" / EXPERIMENT_ID
RUN_ID = DEFAULT_RUN_ID
RUN_DIR = EXPERIMENT_ROOT / RUN_ID
SOURCE_RUN_ID = DEFAULT_SOURCE_RUN_ID
SOURCE_EXPERIMENT_ID = "EXP-051"
SOURCE_EXPERIMENT_ROOT = ROOT / "artifacts/experiments" / SOURCE_EXPERIMENT_ID
SOURCE_RUN_DIR = SOURCE_EXPERIMENT_ROOT / SOURCE_RUN_ID
H5AD = Path("/mnt/d/Code/XPert/processed_data/l1000_sdst_78453.h5ad")
DRUG_MAP = Path("/mnt/d/Code/XPert/processed_data/l1000_sdst_drugs_info_8432.csv")
GPU_UUID = "GPU-9c366c2e-bfa5-c7b3-b16a-1117c40e6871"
FULL_KEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
SEED = 20261006
XPERT_ROOT = Path("/mnt/d/Code/XPert")
TRANSIGEN_ROOT = Path("/mnt/d/Code/TranSiGen/参考代码/TranSiGen/src")


def configure_experiment(experiment_id: str, output_root: str | Path | None = None) -> None:
    """Select a registered experiment artifact root contained in the workspace."""
    global EXPERIMENT_ID, EXPERIMENT_ROOT
    if not re.fullmatch(r"EXP-[0-9]{3}", str(experiment_id)):
        raise ValueError(f"invalid experiment id: {experiment_id!r}")
    EXPERIMENT_ID = str(experiment_id)
    if output_root is None:
        resolved = ROOT / "artifacts/experiments" / EXPERIMENT_ID
    else:
        candidate = Path(output_root)
        resolved = candidate if candidate.is_absolute() else ROOT / candidate
    resolved = resolved.resolve()
    allowed_parent = (ROOT / "artifacts/experiments").resolve()
    if allowed_parent not in resolved.parents or resolved.name != EXPERIMENT_ID:
        raise ValueError("experiment output root must be an EXP-### directory under artifacts/experiments")
    EXPERIMENT_ROOT = resolved


def configure_run(run_id: str) -> None:
    """Select a run directory without allowing path traversal."""
    global EXPERIMENT_ROOT, RUN_ID, RUN_DIR
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", str(run_id)):
        raise ValueError(f"invalid {EXPERIMENT_ID} run_id: {run_id!r}")
    allowed_parent = (ROOT / "artifacts/experiments").resolve()
    if allowed_parent not in EXPERIMENT_ROOT.resolve().parents or EXPERIMENT_ROOT.name != EXPERIMENT_ID:
        EXPERIMENT_ROOT = allowed_parent / EXPERIMENT_ID
    RUN_ID = str(run_id)
    RUN_DIR = EXPERIMENT_ROOT / RUN_ID


def configure_source_run(run_id: str, experiment_id: str = "EXP-051") -> None:
    """Select the immutable run holding frozen fit inputs and their receipts."""
    global SOURCE_RUN_ID, SOURCE_EXPERIMENT_ID, SOURCE_EXPERIMENT_ROOT, SOURCE_RUN_DIR
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", str(run_id)):
        raise ValueError(f"invalid {experiment_id} source_run_id: {run_id!r}")
    if not re.fullmatch(r"EXP-[0-9]{3}", str(experiment_id)):
        raise ValueError(f"invalid source experiment id: {experiment_id!r}")
    SOURCE_RUN_ID = str(run_id)
    SOURCE_EXPERIMENT_ID = str(experiment_id)
    SOURCE_EXPERIMENT_ROOT = ROOT / "artifacts/experiments" / SOURCE_EXPERIMENT_ID
    SOURCE_RUN_DIR = SOURCE_EXPERIMENT_ROOT / SOURCE_RUN_ID


def validate_attempt_id(attempt_id: str) -> int:
    match = re.fullmatch(r"fit_attempt_([0-9]{3})", str(attempt_id))
    if match is None or int(match.group(1)) < 1:
        raise ValueError(f"attempt id must look like fit_attempt_001: {attempt_id!r}")
    return int(match.group(1))


def next_attempt_id(run_dir: Path) -> str:
    """Return the next never-used attempt id in a run directory."""
    existing = [validate_attempt_id(path.name)
                for path in run_dir.glob("fit_attempt_[0-9][0-9][0-9]")]
    return f"fit_attempt_{max(existing, default=0) + 1:03d}"


def new_attempt_path(run_dir: Path, attempt_id: str) -> Path:
    """Resolve only the next unused attempt; never reopen an existing attempt."""
    index = validate_attempt_id(attempt_id)
    existing: list[int] = []
    for path in run_dir.glob("fit_attempt_[0-9][0-9][0-9]"):
        existing.append(validate_attempt_id(path.name))
    next_index = max(existing, default=0) + 1
    if index != next_index:
        raise ValueError(f"requested {attempt_id}, but the next unused attempt is fit_attempt_{next_index:03d}")
    target = run_dir / attempt_id
    if target.exists():
        raise FileExistsError(f"refusing to reopen or overwrite existing attempt: {target}")
    return target


def _validated_zero_gpu_preflight_receipt(attempt_dir: Path, receipt_path: Path,
                                          run_id: str) -> dict[str, Any]:
    """Validate an immutable zero-GPU closure, including older receipt field names."""
    if not receipt_path.is_file():
        raise RuntimeError(f"zero-GPU attempt has no failure closure receipt: {attempt_dir}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    attempt_identity = receipt.get("attempt", receipt.get("attempt_id"))
    charge = receipt.get("charged_gpu_hours", receipt.get(
        "charged_gpu_hours_conservative_upper_bound", float("nan")))
    reserved = receipt.get("reserved_gpu_hours", float("nan"))
    if (receipt.get("run_id") != run_id or attempt_identity != attempt_dir.name
            or receipt.get("status") != "CPU_PREFLIGHT_FAILED_CLOSED_NO_GPU"
            or receipt.get("gpu_smoke_started", False) is not False
            or receipt.get("gpu_model_initialization_started", False) is not False
            or not math.isfinite(float(charge)) or float(charge) != 0.0
            or not math.isfinite(float(reserved)) or float(reserved) != 0.0):
        raise RuntimeError(f"invalid or nonzero pre-GPU failure closure: {attempt_dir}")
    evidence_path = receipt.get("failure_evidence_path")
    evidence_sha = receipt.get("failure_evidence_sha256")
    if evidence_path or evidence_sha:
        if not evidence_path or not evidence_sha:
            raise RuntimeError(f"incomplete attached failure evidence reference: {attempt_dir}")
        resolved_evidence = Path(str(evidence_path))
        if not resolved_evidence.is_absolute():
            resolved_evidence = ROOT / resolved_evidence
        if (not resolved_evidence.is_file()
                or sha256_file(resolved_evidence).lower() != str(evidence_sha).lower()):
            raise RuntimeError(f"attached CPU failure evidence hash mismatch: {attempt_dir}")
    return receipt


def _validate_exp055_smoke_attempt(run_dir: Path, attempt_id: str,
                                   source_run_id: str) -> None:
    """Permit at most two zero-GPU CPU-preflight retries, each proven closed."""
    expected_next = next_attempt_id(run_dir)
    if attempt_id != expected_next:
        raise ValueError(f"requested {attempt_id}, but the next unused attempt is {expected_next}")
    if attempt_id == "fit_attempt_001":
        return
    if attempt_id not in {"fit_attempt_002", "fit_attempt_003"}:
        raise ValueError("EXP-055 permits at most two smoke retries after the initial zero-GPU CPU-preflight failure")

    terminal_artifacts = ("resource_ledger.jsonl", "attempt_close_receipt.json",
                          "failure_close_receipt.json", "fit_smoke_result.json",
                          "fit_smoke_manifest.json")
    next_index = validate_attempt_id(attempt_id)
    for index in range(1, next_index):
        prior_id = f"fit_attempt_{index:03d}"
        prior_attempt = run_dir / prior_id
        receipt_path = prior_attempt / "pre_gpu_failure_receipt.json"
        started_path = prior_attempt / "pre_gpu_preflight_started.json"
        if not prior_attempt.is_dir() or not receipt_path.is_file() or not started_path.is_file():
            raise RuntimeError(f"EXP-055 retry requires {prior_id}'s CPU-only closure and start receipts")
        if any((prior_attempt / name).exists() for name in terminal_artifacts):
            raise RuntimeError(f"EXP-055 retry unavailable: {prior_id} has a GPU ledger or another terminal artifact")
        started = json.loads(started_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (started.get("run_id") != RUN_ID
                or started.get("attempt") != prior_id
                or started.get("source_run_id") != source_run_id
                or started.get("status") != "CPU_PREFLIGHT_STARTED"
                or receipt.get("run_id") != RUN_ID
                or receipt.get("attempt") != prior_id
                or receipt.get("source_run_id") != source_run_id
                or receipt.get("status") != "CPU_PREFLIGHT_FAILED_CLOSED_NO_GPU"
                or receipt.get("gpu_smoke_started") is not False
                or receipt.get("model_initialization_started") is not False
                or float(receipt.get("charged_gpu_hours", float("nan"))) != 0.0
                or float(receipt.get("reserved_gpu_hours", float("nan"))) != 0.0):
            raise RuntimeError(f"EXP-055 retry requires {prior_id} to be a zero-charge pre-GPU closure")


def resource_accounting(experiment_root: Path, run_id: str, package_cap_gpu_hours: float,
                        total_exp_cap_gpu_hours: float,
                        exp_settled_before_artifact_ledgers_gpu_hours: float,
                        package_settled_before_artifact_ledgers_gpu_hours: float = 0.0) -> dict[str, Any]:
    """Calculate package and EXP balances from every closed per-attempt ledger."""
    terminal_events = {
        "GPU_SMOKE_FAILED_CLOSED": "charged_gpu_hours_conservative_upper_bound",
        "GPU_ATTEMPT_FAILED_CLOSED": "charged_gpu_hours_conservative_upper_bound",
        "GPU_SMOKE_FINISHED": "actual_gpu_hours",
        "GPU_ATTEMPT_COMPLETED": "actual_gpu_hours",
        "GPU_ATTEMPT_CLOSED_AFTER_COST_GATE": "actual_gpu_hours",
    }
    package_settled = 0.0
    experiment_ledger_settled = 0.0
    attempts: list[dict[str, Any]] = []
    run_dirs = sorted(path for path in experiment_root.iterdir() if path.is_dir()) if experiment_root.exists() else []
    for run_dir in run_dirs:
        for attempt_dir in sorted(run_dir.glob("fit_attempt_[0-9][0-9][0-9]")):
            ledger_path = attempt_dir / "resource_ledger.jsonl"
            if not ledger_path.is_file():
                latent_close = attempt_dir / "latent_replay_close_receipt.json"
                latent_failure = attempt_dir / "latent_replay_failure_receipt.json"
                if latent_close.is_file() or latent_failure.is_file():
                    is_failure = not latent_close.is_file()
                    receipt_path = latent_failure if is_failure else latent_close
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    owner_path = _trainpath_trace_owner_path(attempt_dir)
                    trace_path = attempt_dir / "latent_replay_trace.jsonl"
                    result_path = attempt_dir / "latent_replay_result.json"
                    expected_status = ("FAILED_CLOSED_CPU_LATENT_REPLAY" if is_failure
                                       else "COMPLETED_READ_ONLY_CPU_LATENT_REPLAY")
                    if (run_dir.name != EXP055_TRAINPATH_LATENT_REPLAY_RUN_ID
                            or receipt.get("run_id") != run_dir.name
                            or receipt.get("attempt_id") != attempt_dir.name
                            or receipt.get("status") != expected_status
                            or receipt.get("device") != "cpu"
                            or float(receipt.get("gpu_hours_charged", float("nan"))) != 0.0
                            or not owner_path.is_file()
                            or receipt.get("owner_manifest_sha256") != sha256_file(owner_path)
                            or not trace_path.is_file()
                            or receipt.get("trace_sha256") != sha256_file(trace_path)):
                        raise RuntimeError(f"invalid zero-GPU TranSiGen latent replay closure: {attempt_dir}")
                    if not is_failure and (not result_path.is_file()
                            or receipt.get("result_sha256") != sha256_file(result_path)):
                        raise RuntimeError(f"latent replay result failed SHA readback: {result_path}")
                    attempts.append({"run_id": run_dir.name, "attempt": attempt_dir.name,
                        "close_event": "CPU_LATENT_REPLAY_FAILED_NO_GPU" if is_failure
                            else "CPU_LATENT_REPLAY_COMPLETED_NO_GPU",
                        "charged_gpu_hours": 0.0, "reserved_gpu_hours": 0.0,
                        "close_at_utc": receipt.get("closed_at_utc")})
                    continue
                replay_close = attempt_dir / "batch_replay_close_receipt.json"
                replay_failure = attempt_dir / "batch_replay_failure_receipt.json"
                if replay_close.is_file() or replay_failure.is_file():
                    is_failure = not replay_close.is_file()
                    receipt_path = replay_failure if is_failure else replay_close
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    owner_path = _trainpath_trace_owner_path(attempt_dir)
                    trace_path = attempt_dir / "TranSiGen_trainpath_layer_trace.jsonl"
                    result_path = attempt_dir / "batch_replay_result.json"
                    expected_status = ("FAILED_CLOSED_CPU_BATCH_REPLAY" if is_failure
                                       else "COMPLETED_READ_ONLY_CPU_BATCH_REPLAY")
                    if (run_dir.name != EXP055_TRAINPATH_BATCH_REPLAY_RUN_ID
                            or receipt.get("run_id") != run_dir.name
                            or receipt.get("attempt_id") != attempt_dir.name
                            or receipt.get("status") != expected_status
                            or receipt.get("device") != "cpu"
                            or float(receipt.get("gpu_hours_charged", float("nan"))) != 0.0
                            or not owner_path.is_file()
                            or receipt.get("owner_manifest_sha256") != sha256_file(owner_path)
                            or not trace_path.is_file()
                            or receipt.get("layer_trace_sha256") != sha256_file(trace_path)):
                        raise RuntimeError(f"invalid zero-GPU TranSiGen batch replay closure: {attempt_dir}")
                    if not is_failure and (not result_path.is_file()
                            or receipt.get("result_sha256") != sha256_file(result_path)):
                        raise RuntimeError(f"batch replay result failed SHA readback: {result_path}")
                    attempts.append({"run_id": run_dir.name, "attempt": attempt_dir.name,
                        "close_event": "CPU_BATCH_REPLAY_FAILED_NO_GPU" if is_failure
                            else "CPU_BATCH_REPLAY_COMPLETED_NO_GPU",
                        "charged_gpu_hours": 0.0, "reserved_gpu_hours": 0.0,
                        "close_at_utc": receipt.get("closed_at_utc")})
                    continue
                entry_smoke_close = attempt_dir / "entry_smoke_close_receipt.json"
                entry_smoke_failure = attempt_dir / "entry_smoke_failure_receipt.json"
                if entry_smoke_close.is_file() or entry_smoke_failure.is_file():
                    is_failure = not entry_smoke_close.is_file()
                    receipt_path = entry_smoke_failure if is_failure else entry_smoke_close
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    result_path = attempt_dir / ("entry_smoke_failure_receipt.json" if is_failure
                                                 else "entry_smoke_result.json")
                    owner_path = attempt_dir / "TranSiGen_trainpath_layer_trace_manifest.json"
                    expected_status = ("FAILED_CLOSED_CPU_ENTRY_SMOKE" if is_failure
                                       else "PASS_REAL_ORIGINAL_TRANSIGEN_CPU_ENTRY_SMOKE")
                    if (receipt.get("run_id") != run_dir.name
                            or receipt.get("attempt_id") != attempt_dir.name
                            or receipt.get("status") != expected_status
                            or receipt.get("device") != "cpu"
                            or float(receipt.get("gpu_hours_charged", float("nan"))) != 0.0
                            or not result_path.is_file() or not owner_path.is_file()
                            or (not is_failure and receipt.get("result_sha256") != sha256_file(result_path))
                            or receipt.get("owner_manifest_sha256") != sha256_file(owner_path)):
                        raise RuntimeError(f"invalid zero-GPU CPU entry-smoke closure: {attempt_dir}")
                    for path_key, hash_key in (
                            ("checkpoint_path", "checkpoint_sha256"),
                            ("checkpoint_readback_path", "checkpoint_readback_sha256"),
                            ("train_batch_trace_path", "train_batch_trace_sha256"),
                            ("first_nonfinite_delta_path", "first_nonfinite_delta_sha256")):
                        artifact_name, expected_hash = receipt.get(path_key), receipt.get(hash_key)
                        if artifact_name is None:
                            if expected_hash is not None:
                                raise RuntimeError(f"CPU entry-smoke has a hash without {path_key}: {attempt_dir}")
                            continue
                        artifact_path = attempt_dir / str(artifact_name)
                        if not artifact_path.is_file() or sha256_file(artifact_path) != expected_hash:
                            raise RuntimeError(f"CPU entry-smoke referenced artifact failed SHA readback: {artifact_path}")
                    if not is_failure:
                        result = json.loads(result_path.read_text(encoding="utf-8"))
                        if (result.get("status") != receipt.get("status")
                                or result.get("gpu_hours_charged") != 0.0
                                or result.get("inner_dev_numeric_rows_loaded") is not False
                                or result.get("pdo_outer_locked_numeric_values_opened") is not False):
                            raise RuntimeError(f"CPU entry-smoke result violates its zero-GPU/data-role receipt: {attempt_dir}")
                        if run_dir.name == EXP055_TRAINPATH_ENTRY_DIAG_RUN_ID:
                            readback_path = attempt_dir / str(result.get("checkpoint_readback_path"))
                            if (not readback_path.is_file()
                                    or sha256_file(readback_path) != result.get("checkpoint_readback_sha256")
                                    or json.loads(readback_path.read_text(encoding="utf-8")).get("status")
                                        != "PASS_EXACT_PRE_EVALUATION_READBACK"):
                                raise RuntimeError(f"diagnostic smoke checkpoint readback evidence failed: {readback_path}")
                    elif run_dir.name == EXP055_TRAINPATH_ENTRY_DIAG_RUN_ID:
                        if receipt.get("exception_type") == "FloatingPointError" and "main-delta" in str(
                                receipt.get("exception_message", "")):
                            if not (receipt.get("checkpoint_path")
                                    and receipt.get("checkpoint_readback_status")
                                        == "PASS_EXACT_PRE_EVALUATION_READBACK"
                                    and receipt.get("train_batch_trace_path")
                                    and receipt.get("first_nonfinite_delta_path")):
                                raise RuntimeError(f"diagnostic main-delta failure lacks required evidence: {attempt_dir}")
                    attempts.append({"run_id": run_dir.name, "attempt": attempt_dir.name,
                                     "close_event": ("CPU_ENTRY_SMOKE_FAILED_NO_GPU" if is_failure
                                                     else "CPU_ENTRY_SMOKE_COMPLETED_NO_GPU"),
                                     "charged_gpu_hours": 0.0, "reserved_gpu_hours": 0.0,
                                     "close_at_utc": receipt.get("closed_at_utc")})
                    continue
                if (run_dir.name == "20261007_exp055_transigen_trainpath_recovery_v6"
                        and attempt_dir.name == "fit_attempt_001"):
                    legacy_owner_path = attempt_dir / "TranSiGen_trainpath_layer_trace_manifest.json"
                    if legacy_owner_path.is_file():
                        legacy_owner = json.loads(legacy_owner_path.read_text(encoding="utf-8"))
                        if (legacy_owner.get("run_id") != run_dir.name
                                or legacy_owner.get("attempt_id") != attempt_dir.name
                                or legacy_owner.get("close_status") != "FAILED_CLOSED"
                                or legacy_owner.get("exception_type") != "NameError"
                                or legacy_owner.get("exception_message") != "name 'prior_close_sha' is not defined"
                                or legacy_owner.get("runner_sha256") != "e87c1cbcb4da951a94ef31bafca4debc72d65447e5450c85538b20fcdda0901a"
                                or legacy_owner.get("trace_sha256") is not None):
                            raise RuntimeError(f"immutable v6 zero-GPU failure manifest changed: {attempt_dir}")
                        attempts.append({"run_id": run_dir.name, "attempt": attempt_dir.name,
                                         "close_event": "PRE_MODEL_CPU_IMPLEMENTATION_FAILURE_NO_GPU",
                                         "charged_gpu_hours": 0.0, "reserved_gpu_hours": 0.0,
                                         "close_at_utc": legacy_owner.get("closed_at_utc")})
                        continue
                if run_dir.name in EXP055_TRAINPATH_PRELAUNCH_FAILURE_RUN_IDS:
                    owner_path = _trainpath_trace_owner_path(attempt_dir)
                    started_path = attempt_dir / "pre_gpu_preflight_started.json"
                    manifest_path = attempt_dir / "fit_smoke_manifest.json"
                    if owner_path.is_file() and started_path.is_file() and manifest_path.is_file():
                        owner = json.loads(owner_path.read_text(encoding="utf-8"))
                        started = json.loads(started_path.read_text(encoding="utf-8"))
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                        if (owner.get("run_id") == run_dir.name
                                and owner.get("attempt_id") == attempt_dir.name
                                and owner.get("status") == "CLOSED"
                                and owner.get("close_status") == "FAILED_CLOSED"
                                and owner.get("trace_sha256") is None
                                and not (attempt_dir / "TranSiGen_trainpath_layer_trace.jsonl").exists()
                                and started.get("run_id") == run_dir.name
                                and started.get("attempt") == attempt_dir.name
                                and started.get("gpu_started") is False
                                and float(started.get("reserved_gpu_hours", float("nan"))) == 0.0
                                and started.get("runner_sha256") == owner.get("runner_sha256")
                                and manifest.get("status") == "CPU_IDENTITY_AND_BUDGET_GATE_PASS_GPU_NOT_YET_STARTED"
                                and manifest.get("current_runner_sha256") == owner.get("runner_sha256")):
                            attempts.append({"run_id": run_dir.name, "attempt": attempt_dir.name,
                                "close_event": "PRE_MODEL_CPU_IMPLEMENTATION_FAILURE_NO_GPU",
                                "charged_gpu_hours": 0.0, "reserved_gpu_hours": 0.0,
                                "close_at_utc": owner.get("closed_at_utc"),
                                "exception_type": owner.get("exception_type"),
                                "exception_message": owner.get("exception_message"),
                                "runner_sha256": owner.get("runner_sha256")})
                            continue
                pre_gpu_receipt_path = attempt_dir / "pre_gpu_failure_receipt.json"
                pre_gpu_receipt = _validated_zero_gpu_preflight_receipt(
                    attempt_dir, pre_gpu_receipt_path, run_dir.name)
                attempts.append({"run_id": run_dir.name, "attempt": attempt_dir.name,
                                 "close_event": "CPU_PREFLIGHT_FAILED_CLOSED_NO_GPU",
                                 "charged_gpu_hours": 0.0, "reserved_gpu_hours": 0.0,
                                 "close_at_utc": pre_gpu_receipt.get("observed_close_time_utc")})
                continue
            entries = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
            starts = [entry for entry in entries if entry.get("event") in
                      {"GPU_SMOKE_STARTED", "GPU_ATTEMPT_STARTED"}]
            terminal = [entry for entry in entries if entry.get("event") in terminal_events]
            cpu_closures = [entry for entry in entries
                            if entry.get("event") == "CPU_PREFLIGHT_FAILED_CLOSED_NO_GPU"]
            if cpu_closures and not starts and not terminal:
                if len(cpu_closures) != 1:
                    raise RuntimeError(f"CPU-only attempt must have exactly one terminal closure: {attempt_dir}")
                pre_gpu_receipt = _validated_zero_gpu_preflight_receipt(
                    attempt_dir, attempt_dir / "pre_gpu_failure_receipt.json", run_dir.name)
                event = cpu_closures[0]
                event_charge = float(event.get("charged_gpu_hours_conservative_upper_bound", float("nan")))
                if (event.get("attempt") != attempt_dir.name or not math.isfinite(event_charge)
                        or event_charge != 0.0 or float(event.get("reserved_gpu_hours", float("nan"))) != 0.0):
                    raise RuntimeError(f"invalid CPU-only ledger closure: {attempt_dir}")
                attempts.append({"run_id": run_dir.name, "attempt": attempt_dir.name,
                                 "close_event": "CPU_PREFLIGHT_FAILED_CLOSED_NO_GPU",
                                 "charged_gpu_hours": 0.0, "reserved_gpu_hours": 0.0,
                                 "close_at_utc": event.get("at_utc") or pre_gpu_receipt.get("closed_at_utc")})
                continue
            if len(starts) != 1 or len(terminal) != 1:
                raise RuntimeError(f"attempt must have one start and one resource close event: {attempt_dir}")
            close = terminal[0]
            charge_key = terminal_events[str(close["event"])]
            if charge_key not in close:
                raise RuntimeError(f"resource close event lacks {charge_key}: {attempt_dir}")
            charge = float(close[charge_key])
            if not math.isfinite(charge) or charge < 0:
                raise ValueError(f"invalid settled GPU charge for {attempt_dir}: {charge}")
            reserved = float(starts[0].get("reserved_gpu_hours", float("nan")))
            if not math.isfinite(reserved) or reserved < 0:
                raise ValueError(f"attempt start has an invalid GPU reservation: {attempt_dir}")
            experiment_ledger_settled += charge
            if run_dir.name == run_id:
                package_settled += charge
            attempts.append({"run_id": run_dir.name, "attempt": attempt_dir.name,
                             "close_event": close["event"], "charged_gpu_hours": charge,
                             "reserved_gpu_hours": reserved,
                             "close_at_utc": close.get("at_utc")})
    package_settled_before = float(package_settled_before_artifact_ledgers_gpu_hours)
    if not math.isfinite(package_settled_before) or package_settled_before < 0:
        raise ValueError("package settled baseline must be finite and non-negative")
    package_artifact_settled = package_settled
    package_settled += package_settled_before
    package_remaining = float(package_cap_gpu_hours) - package_settled
    exp_settled = float(exp_settled_before_artifact_ledgers_gpu_hours) + experiment_ledger_settled
    exp_remaining = float(total_exp_cap_gpu_hours) - exp_settled
    if package_remaining < -1e-9:
        raise RuntimeError(f"settled run charges exceed its configured package cap by {-package_remaining:.6f} GPUh")
    if exp_remaining < -1e-9:
        raise RuntimeError(f"settled {EXPERIMENT_ID} charges exceed its configured cap by {-exp_remaining:.6f} GPUh")
    remaining = max(0.0, min(package_remaining, exp_remaining))
    return {"run_id": run_id, "package_cap_gpu_hours": float(package_cap_gpu_hours),
            "package_settled_gpu_hours": package_settled,
            "package_settled_before_artifact_ledgers_gpu_hours": package_settled_before,
            "package_artifact_ledger_settled_gpu_hours": package_artifact_settled,
            "package_remaining_gpu_hours": max(0.0, package_remaining),
            "total_exp_cap_gpu_hours": float(total_exp_cap_gpu_hours),
            "exp_settled_before_artifact_ledgers_gpu_hours": float(exp_settled_before_artifact_ledgers_gpu_hours),
            "exp_artifact_ledger_settled_gpu_hours": experiment_ledger_settled,
            "exp_settled_gpu_hours": exp_settled,
            "exp_remaining_gpu_hours": max(0.0, exp_remaining),
            "settled_gpu_hours": package_settled,
            "remaining_authorized_gpu_hours": remaining,
            "closed_attempts": attempts,
            "source": f"all {EXPERIMENT_ID} per-attempt append-only resource_ledger.jsonl terminal events plus frozen package baseline"}


def resource_limits(cfg: dict[str, Any]) -> dict[str, float]:
    """Read package, smoke, and EXP limits from the approved config schema."""
    authorization = cfg.get("authorization", {})
    resource = cfg.get("resource", {})
    package_cap = authorization.get("package_gpu_cap")
    if package_cap is None:
        package_cap = next((resource[key] for key in
                            ("comparison_package_gpu_cap_hours", "package_cap_gpu_hours",
                             "first_package_cap_gpu_hours") if key in resource), None)
    if package_cap is None:
        package_cap = authorization.get("run_gpu_cap")
    if package_cap is None:
        raise KeyError(f"{EXPERIMENT_ID} config must provide the authorized package GPU cap")
    smoke_cap = next((resource[key] for key in
                      ("paired_smoke_gpu_cap_hours", "first_paired_smoke_gpu_cap_hours",
                       "first_fit_smoke_cap_gpu_hours") if key in resource), None)
    if smoke_cap is None:
        smoke_cap = authorization.get("first_fit_smoke_gpu_cap", package_cap)
    exp_cap = authorization.get("exp_gpu_cap", resource.get("total_exp_cap_gpu_hours"))
    if exp_cap is None:
        raise KeyError(f"{EXPERIMENT_ID} config must provide the total EXP GPU cap")
    exp_settled_base = authorization.get(
        "exp_settled_before_artifact_ledgers_gpu_hours",
        authorization.get("exp_settled_gpu_hours_before_run",
                          authorization.get("exp_settled_gpu_hours_before_artifact_ledgers", 0.0)))
    package_settled_base = authorization.get("package_settled_before_artifact_ledgers_gpu_hours",
                     authorization.get("package_settled_gpu_hours_before_run",
                     authorization.get("package_settled_gpu_hours_before_artifact_ledgers", 0.0)))
    first_phase_cap = authorization.get("first_phase_gpu_cap", package_cap)
    values = {"package_cap_gpu_hours": float(package_cap),
              "paired_smoke_cap_gpu_hours": float(smoke_cap),
              "total_exp_cap_gpu_hours": float(exp_cap),
              "exp_settled_before_artifact_ledgers_gpu_hours": float(exp_settled_base),
              "package_settled_before_artifact_ledgers_gpu_hours": float(package_settled_base),
              "first_phase_gpu_cap_hours": float(first_phase_cap)}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError(f"invalid configured GPU authorization values: {values}")
    duplicated_limits = [
        ("package cap", package_cap, resource.get("package_gpu_cap_hours")),
        ("smoke cap", smoke_cap, resource.get("paired_smoke_cap_gpu_hours")),
        ("first fit smoke cap", authorization.get("first_fit_smoke_gpu_cap", smoke_cap),
         resource.get("paired_smoke_cap_gpu_hours")),
        ("EXP cap", exp_cap, resource.get("total_exp_cap_gpu_hours")),
        ("first phase cap", first_phase_cap, resource.get("first_phase_cap_gpu_hours")),
        ("settled package baseline", package_settled_base,
         resource.get("package_settled_before_artifact_ledgers_gpu_hours")),
        ("settled EXP baseline", exp_settled_base,
         resource.get("exp_settled_before_artifact_ledgers_gpu_hours")),
    ]
    for label, authorized, configured in duplicated_limits:
        if configured is not None and not math.isclose(float(authorized), float(configured), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"authorization/resource config disagrees on {label}: {authorized} != {configured}")
    return values


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_exp059_parent_evidence(cfg: dict[str, Any]) -> tuple[Path, str]:
    """Validate the config-bound EXP-055 checkpoint as provenance evidence only."""
    authorization = cfg.get("authorization", {})
    relative = authorization.get("mechanism_replay_checkpoint")
    expected_sha = authorization.get("mechanism_replay_checkpoint_sha256")
    if not isinstance(relative, str) or not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha)):
        raise ValueError("EXP-059 must bind a relative parent evidence path and lowercase SHA-256 in its config")
    candidate = (ROOT / relative).resolve()
    allowed_parent = (ROOT / "artifacts/experiments/EXP-055").resolve()
    if allowed_parent not in candidate.parents or not candidate.is_file():
        raise FileNotFoundError("EXP-059 parent evidence must resolve to an existing immutable EXP-055 artifact")
    actual_sha = sha256_file(candidate)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"EXP-059 config-bound parent evidence SHA mismatch: expected {expected_sha}, got {actual_sha}")
    return candidate, actual_sha


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write a NumPy archive through a same-directory temporary file, then replace atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def exp055_trainpath_recovery_prior_charge() -> tuple[float, dict[str, str]]:
    """Reconcile immutable v4/v5/v9/v11 GPU charges and the zero-GPU v6 failure."""
    roots = {
        "v4": ("20261007_exp055_transigen_trainpath_recovery_v4",
               "72bd511e4bea52fb167248328d3b98d0e1b731f06d3e2fe7350fe10d41a4cbfe",
               0.0013252952777777778, "4104bc2d68525c25cb00cbb90453b78562b8d665a37c05abf56720d301214e3f"),
        "v5": ("20261007_exp055_transigen_trainpath_recovery_v5", None,
               0.0016761897222222224, "71844cc452694a17a5c7a6efcb895cea0423bf5ba2adf2e6085d5a3724d864b4"),
    }
    total = 0.0
    close_hashes: dict[str, str] = {}
    for label, (run_id, expected_close_sha, expected_charge, expected_runner_sha) in roots.items():
        prior_attempt = ROOT / "artifacts/experiments/EXP-055" / run_id / "fit_attempt_001"
        ledger_path = prior_attempt / "resource_ledger.jsonl"
        close_path = prior_attempt / "failure_close_receipt.json"
        if not ledger_path.is_file() or not close_path.is_file():
            raise FileNotFoundError(f"the immutable {label} recovery attempt must have its ledger and failure receipt")
        close_sha = sha256_file(close_path)
        if expected_close_sha is not None and close_sha != expected_close_sha:
            raise RuntimeError(f"the {label} recovery failure receipt differs from its verified version")
        close = json.loads(close_path.read_text(encoding="utf-8"))
        entries = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        charge = float(close.get("charged_gpu_hours_conservative_upper_bound", -1.0))
        settled_through_attempt = sum(roots[prior_label][2] for prior_label in roots
                                      if prior_label == "v4" or label == "v5")
        expected_unused = 1.0 - settled_through_attempt
        if (len(entries) != 2 or entries[0].get("event") != "GPU_ATTEMPT_STARTED"
                or entries[1].get("event") != "GPU_ATTEMPT_FAILED_CLOSED"
                or close.get("run_id") != run_id or close.get("attempt") != "fit_attempt_001"
                or close.get("status") != "FAILED_CLOSED"
                or close.get("runner_sha256") != expected_runner_sha
                or entries[0].get("device_uuid") != GPU_UUID or entries[1].get("device_uuid") != GPU_UUID
                or entries[1].get("close_receipt") != "failure_close_receipt.json"
                or not math.isfinite(charge) or not math.isclose(charge, expected_charge, rel_tol=0.0, abs_tol=1e-15)
                or not math.isclose(charge, float(entries[1].get("charged_gpu_hours_conservative_upper_bound", -2.0)),
                                    rel_tol=0.0, abs_tol=1e-15)
                or not math.isclose(float(close.get("unused_reservation_released_gpu_hours", -1.0)),
                                    expected_unused, rel_tol=0.0, abs_tol=1e-12)):
            raise RuntimeError(f"the {label} recovery charge does not reconcile to its immutable A5000 closure")
        total += charge
        close_hashes[label] = close_sha
    if not math.isclose(total, 0.003001485, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError(f"v4/v5 recovery charge changed from the verified 0.003001485 GPUh: {total}")
    v6_attempt = (ROOT / "artifacts/experiments/EXP-055/"
                  "20261007_exp055_transigen_trainpath_recovery_v6/fit_attempt_001")
    v6_manifest_path = v6_attempt / "TranSiGen_trainpath_layer_trace_manifest.json"
    if not v6_manifest_path.is_file():
        raise FileNotFoundError("immutable v6 failed-closed attempt manifest is required for recovery accounting")
    v6_manifest = json.loads(v6_manifest_path.read_text(encoding="utf-8"))
    if (v6_manifest.get("run_id") != "20261007_exp055_transigen_trainpath_recovery_v6"
            or v6_manifest.get("attempt_id") != "fit_attempt_001"
            or v6_manifest.get("status") != "CLOSED"
            or v6_manifest.get("close_status") != "FAILED_CLOSED"
            or v6_manifest.get("exception_type") != "NameError"
            or v6_manifest.get("exception_message") != "name 'prior_close_sha' is not defined"
            or v6_manifest.get("runner_sha256") != "e87c1cbcb4da951a94ef31bafca4debc72d65447e5450c85538b20fcdda0901a"
            or v6_manifest.get("trace_sha256") is not None
            or (v6_attempt / "resource_ledger.jsonl").exists()
            or any((v6_attempt / name).exists() for name in (
                "TranSiGen_trainpath_layer_trace.jsonl", "TranSiGen_epoch_1_cost_gate_checkpoint.pt",
                "TranSiGen_epoch_100.pt", "TranSiGen_epoch_300.pt",
                "TranSiGen_trainpath_recovery_curve.jsonl"))):
        raise RuntimeError("the immutable v6 closure no longer proves a pre-GPU zero-charge failure")
    close_hashes["v6_zero_gpu_failure_manifest"] = sha256_file(v6_manifest_path)

    v9_attempt = (ROOT / "artifacts/experiments/EXP-055/"
                  "20261007_exp055_transigen_trainpath_recovery_v9/fit_attempt_001")
    v9_close_path = v9_attempt / "failure_close_receipt.json"
    v9_ledger_path = v9_attempt / "resource_ledger.jsonl"
    v9_owner_path = v9_attempt / "TranSiGen_trainpath_layer_trace_manifest.json"
    v9_trace_path = v9_attempt / "TranSiGen_trainpath_layer_trace.jsonl"
    if not all(path.is_file() for path in (v9_close_path, v9_ledger_path, v9_owner_path, v9_trace_path)):
        raise FileNotFoundError("the immutable v9 A5000 failure evidence is required for recovery accounting")
    v9_close_sha = sha256_file(v9_close_path)
    if v9_close_sha != "605d72f605f6be7d995ecba73e04f8c1c2a410629c7d7390088b0fd79bffb021":
        raise RuntimeError("the v9 recovery failure receipt differs from its root-verified immutable version")
    v9_close = json.loads(v9_close_path.read_text(encoding="utf-8"))
    v9_entries = [json.loads(line) for line in v9_ledger_path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    v9_owner = json.loads(v9_owner_path.read_text(encoding="utf-8"))
    v9_charge = 0.009245871944444446
    v9_prior_charge = 0.003001485
    v9_unused = 0.9877526430555555
    v9_expected_message = "_trainpath_parameter_snapshot() missing 1 required positional argument: 'attempt'"
    if (len(v9_entries) != 2
            or v9_entries[0].get("event") != "GPU_ATTEMPT_STARTED"
            or v9_entries[1].get("event") != "GPU_ATTEMPT_FAILED_CLOSED"
            or any(entry.get("device_uuid") != GPU_UUID for entry in v9_entries)
            or v9_entries[0].get("runner_sha256") != "6c0a60c95c0e6269205681e4536653a407d16c80ab6dba128c5ab18664497ff8"
            or v9_entries[0].get("reserved_gpu_hours") != 0.996998515
            or v9_entries[1].get("close_receipt") != "failure_close_receipt.json"
            or not math.isclose(float(v9_entries[1].get("charged_gpu_hours_conservative_upper_bound", -1.0)),
                                v9_charge, rel_tol=0.0, abs_tol=1e-15)
            or not math.isclose(float(v9_entries[1].get("prior_attempts_settled_gpu_hours", -1.0)),
                                v9_prior_charge, rel_tol=0.0, abs_tol=1e-15)
            or not math.isclose(float(v9_entries[1].get("unused_authorized_reservation_released_gpu_hours", -1.0)),
                                v9_unused, rel_tol=0.0, abs_tol=1e-12)
            or v9_close.get("run_id") != "20261007_exp055_transigen_trainpath_recovery_v9"
            or v9_close.get("attempt") != "fit_attempt_001"
            or v9_close.get("status") != "FAILED_CLOSED"
            or v9_close.get("exception_type") != "TypeError"
            or v9_close.get("exception_message") != v9_expected_message
            or v9_close.get("runner_sha256") != v9_entries[0].get("runner_sha256")
            or not math.isclose(float(v9_close.get("charged_gpu_hours_conservative_upper_bound", -1.0)),
                                v9_charge, rel_tol=0.0, abs_tol=1e-15)
            or not math.isclose(float(v9_close.get("prior_attempts_settled_gpu_hours", -1.0)),
                                v9_prior_charge, rel_tol=0.0, abs_tol=1e-15)
            or not math.isclose(float(v9_close.get("unused_reservation_released_gpu_hours", -1.0)),
                                v9_unused, rel_tol=0.0, abs_tol=1e-12)
            or v9_owner.get("run_id") != "20261007_exp055_transigen_trainpath_recovery_v9"
            or v9_owner.get("attempt_id") != "fit_attempt_001"
            or v9_owner.get("close_status") != "FAILED_CLOSED"
            or v9_owner.get("exception_type") != "TypeError"
            or v9_owner.get("exception_message") != v9_expected_message
            or v9_owner.get("runner_sha256") != v9_close.get("runner_sha256")
            or v9_owner.get("trace_sha256") != sha256_file(v9_trace_path)):
        raise RuntimeError("the immutable v9 recovery charge does not reconcile to its root-verified A5000 closure")
    total += v9_charge
    if not math.isclose(total, 0.012247356944444445, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError(f"v4/v5/v9 recovery charge changed from the verified 0.0122473569444 GPUh: {total}")
    close_hashes["v9_failure_close_receipt"] = v9_close_sha

    v11_attempt = (ROOT / "artifacts/experiments/EXP-055/"
                   "20261007_exp055_transigen_trainpath_recovery_v11/fit_attempt_001")
    v11_close_path = v11_attempt / "failure_close_receipt.json"
    v11_ledger_path = v11_attempt / "resource_ledger.jsonl"
    v11_owner_path = v11_attempt / "TranSiGen_trainpath_layer_trace_manifest.json"
    v11_trace_path = v11_attempt / "TranSiGen_trainpath_layer_trace.jsonl"
    if not all(path.is_file() for path in (v11_close_path, v11_ledger_path, v11_owner_path, v11_trace_path)):
        raise FileNotFoundError("the immutable v11 A5000 failure evidence is required for recovery accounting")
    v11_close_sha = sha256_file(v11_close_path)
    if v11_close_sha != "a333c0d260eb7fe13dae2449cb92c9ea874f0e6c6eec94e5622a42459deddc9f":
        raise RuntimeError("the v11 recovery failure receipt differs from its root-verified immutable version")
    v11_close = json.loads(v11_close_path.read_text(encoding="utf-8"))
    v11_entries = [json.loads(line) for line in v11_ledger_path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    v11_owner = json.loads(v11_owner_path.read_text(encoding="utf-8"))
    v11_charge = 0.009226228055555556
    v11_prior_charge = 0.012247356944444445
    v11_reserved = 0.9877526430555555
    v11_unused = 0.978526415
    if (len(v11_entries) != 2
            or v11_entries[0].get("event") != "GPU_ATTEMPT_STARTED"
            or v11_entries[1].get("event") != "GPU_ATTEMPT_FAILED_CLOSED"
            or any(entry.get("device_uuid") != GPU_UUID for entry in v11_entries)
            or v11_entries[0].get("runner_sha256") != "5ec41d8a7887617d763cf17eaba90a69b920fc6e98bbad1c2a67c9f870f73850"
            or not math.isclose(float(v11_entries[0].get("reserved_gpu_hours", -1.0)),
                                v11_reserved, rel_tol=0.0, abs_tol=1e-15)
            or v11_entries[1].get("close_receipt") != "failure_close_receipt.json"
            or not math.isclose(float(v11_entries[1].get("charged_gpu_hours_conservative_upper_bound", -1.0)),
                                v11_charge, rel_tol=0.0, abs_tol=1e-15)
            or not math.isclose(float(v11_entries[1].get("prior_attempts_settled_gpu_hours", -1.0)),
                                v11_prior_charge, rel_tol=0.0, abs_tol=1e-15)
            or not math.isclose(float(v11_entries[1].get("unused_authorized_reservation_released_gpu_hours", -1.0)),
                                v11_unused, rel_tol=0.0, abs_tol=1e-12)
            or v11_close.get("run_id") != "20261007_exp055_transigen_trainpath_recovery_v11"
            or v11_close.get("attempt") != "fit_attempt_001"
            or v11_close.get("status") != "FAILED_CLOSED"
            or v11_close.get("exception_type") != "KeyError"
            or v11_close.get("exception_message") != "'metrics'"
            or v11_close.get("runner_sha256") != v11_entries[0].get("runner_sha256")
            or not math.isclose(float(v11_close.get("reserved_gpu_hours", -1.0)),
                                v11_reserved, rel_tol=0.0, abs_tol=1e-15)
            or not math.isclose(float(v11_close.get("charged_gpu_hours_conservative_upper_bound", -1.0)),
                                v11_charge, rel_tol=0.0, abs_tol=1e-15)
            or not math.isclose(float(v11_close.get("prior_attempts_settled_gpu_hours", -1.0)),
                                v11_prior_charge, rel_tol=0.0, abs_tol=1e-15)
            or not math.isclose(float(v11_close.get("unused_reservation_released_gpu_hours", -1.0)),
                                v11_unused, rel_tol=0.0, abs_tol=1e-12)
            or v11_owner.get("run_id") != "20261007_exp055_transigen_trainpath_recovery_v11"
            or v11_owner.get("attempt_id") != "fit_attempt_001"
            or v11_owner.get("close_status") != "FAILED_CLOSED"
            or v11_owner.get("exception_type") != "KeyError"
            or v11_owner.get("exception_message") != "'metrics'"
            or v11_owner.get("runner_sha256") != v11_close.get("runner_sha256")
            or v11_owner.get("trace_sha256") != sha256_file(v11_trace_path)):
        raise RuntimeError("the immutable v11 recovery charge does not reconcile to its root-verified A5000 closure")
    total += v11_charge
    if not math.isclose(total, 0.021473585, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError(f"v4/v5/v9/v11 recovery charge changed from the verified 0.021473585 GPUh: {total}")
    close_hashes["v11_failure_close_receipt"] = v11_close_sha
    return total, close_hashes


def exp055_transigen_lean_monitor_settled_charge(
        *, include_cost_calibration: bool = False) -> tuple[float, dict[str, str]]:
    """Reconcile the closed v4/v5/v9/v11/v12 recovery costs and optional lean calibration."""
    prior, close_hashes = exp055_trainpath_recovery_prior_charge()
    v12_attempt = (ROOT / "artifacts/experiments/EXP-055" /
        f"{EXP055_TRANSIGEN_COST_SOURCE_RUN_ID}/{EXP055_TRANSIGEN_COST_SOURCE_ATTEMPT}")
    v12_close_path = v12_attempt / "attempt_close_receipt.json"
    v12_ledger_path = v12_attempt / "resource_ledger.jsonl"
    checkpoint_path = v12_attempt / EXP055_TRANSIGEN_COST_SOURCE_CHECKPOINT
    if not all(path.is_file() for path in (v12_close_path, v12_ledger_path, checkpoint_path)):
        raise FileNotFoundError("root-verified immutable v12 cost-gate closure is required")
    v12_close_sha = sha256_file(v12_close_path)
    v12_close = json.loads(v12_close_path.read_text(encoding="utf-8"))
    v12_entries = [json.loads(line) for line in v12_ledger_path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    v12_charge = 0.009391404533040866
    if (sha256_file(checkpoint_path) != EXP055_TRANSIGEN_COST_SOURCE_CHECKPOINT_SHA256
            or v12_close.get("status") != "CLOSED_AFTER_TRAIN_ONLY_TRACE_COST_GATE"
            or v12_close.get("runner_sha256") != "c4e7a7074dc074c424af49169eaf51a049584dfce871478fad9ea82a9113c0ac"
            or v12_close.get("config_sha256") != "40cafd30ec004b06782b12ea7ad6f33f6b28fc3f7fe3fb62c4d3ba880a2c14b2"
            or v12_close.get("epoch1_cost_gate_checkpoint_sha256") != EXP055_TRANSIGEN_COST_SOURCE_CHECKPOINT_SHA256
            or v12_close.get("inner_dev_numeric_rows_loaded") is not False
            or not math.isclose(float(v12_close.get("actual_gpu_hours", -1.0)), v12_charge,
                                rel_tol=0.0, abs_tol=1e-15)
            or len(v12_entries) != 2
            or [entry.get("event") for entry in v12_entries] != [
                "GPU_ATTEMPT_STARTED", "GPU_ATTEMPT_CLOSED_AFTER_COST_GATE"]
            or any(entry.get("device_uuid") != GPU_UUID for entry in v12_entries)
            or not math.isclose(float(v12_entries[1].get("actual_gpu_hours", -1.0)), v12_charge,
                                rel_tol=0.0, abs_tol=1e-15)
            or v12_entries[1].get("close_receipt") != "attempt_close_receipt.json"):
        raise RuntimeError("the immutable v12 calibration checkpoint and A5000 charge no longer reconcile")
    prior += v12_charge
    close_hashes["v12_cost_gate_close_receipt"] = v12_close_sha

    if include_cost_calibration:
        cal_attempt = (ROOT / "artifacts/experiments/EXP-055" /
            f"{EXP055_TRANSIGEN_LEAN_COST_RUN_ID}/fit_attempt_001")
        cal_close_path = cal_attempt / "attempt_close_receipt.json"
        cal_ledger_path = cal_attempt / "resource_ledger.jsonl"
        cal_gate_path = cal_attempt / "lean_monitor_cost_gate.json"
        if not all(path.is_file() for path in (cal_close_path, cal_ledger_path, cal_gate_path)):
            raise FileNotFoundError("closed lean-monitor calibration and cost gate are required before fixed300")
        cal_close_sha = sha256_file(cal_close_path)
        cal_close = json.loads(cal_close_path.read_text(encoding="utf-8"))
        cal_gate = json.loads(cal_gate_path.read_text(encoding="utf-8"))
        cal_entries = [json.loads(line) for line in cal_ledger_path.read_text(encoding="utf-8").splitlines()
                       if line.strip()]
        cal_charge = float(cal_close.get("actual_gpu_hours", -1.0))
        if (cal_close.get("run_id") != EXP055_TRANSIGEN_LEAN_COST_RUN_ID
                or cal_close.get("attempt_id") != "fit_attempt_001"
                or cal_close.get("status") != "CLOSED_AFTER_COST_CALIBRATION"
                or cal_gate.get("status") != "PASS_FIXED300_WITHIN_RECOVERY_CAP"
                or cal_close.get("runner_sha256") != sha256_file(Path(__file__).resolve())
                or not math.isfinite(cal_charge) or cal_charge <= 0.0 or cal_charge > 0.05
                or len(cal_entries) != 2
                or [entry.get("event") for entry in cal_entries] != [
                    "GPU_ATTEMPT_STARTED", "GPU_ATTEMPT_CLOSED_AFTER_COST_CALIBRATION"]
                or any(entry.get("device_uuid") != GPU_UUID for entry in cal_entries)
                or not math.isclose(float(cal_entries[1].get("actual_gpu_hours", -1.0)), cal_charge,
                                    rel_tol=0.0, abs_tol=1e-15)):
            raise RuntimeError("lean-monitor calibration closure is not a verified finite fixed300 cost pass")
        prior += cal_charge
        close_hashes["lean_cost_calibration_close_receipt"] = cal_close_sha
    return prior, close_hashes


def load_config(path: Path, experiment_id: str | None = None) -> tuple[dict[str, Any], str]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected_experiment_id = experiment_id or EXPERIMENT_ID
    if not isinstance(cfg, dict) or cfg.get("experiment_id") != expected_experiment_id:
        raise ValueError(f"configuration is not {expected_experiment_id}")
    config_run_id = cfg.get("run_id")
    resource_run_id = cfg.get("resource", {}).get("run_id")
    # This separately registered recovery run reuses the frozen EXP-055 recipe
    # instead of creating a copied config with duplicate model parameters.
    approved_exp055_recovery_run = (
        expected_experiment_id == "EXP-055"
        and RUN_ID in {EXP055_TRAINPATH_ENTRY_SMOKE_RUN_ID,
                       EXP055_TRAINPATH_ENTRY_DIAG_RUN_ID,
                       EXP055_TRAINPATH_BATCH_REPLAY_RUN_ID,
                       EXP055_TRAINPATH_LATENT_REPLAY_RUN_ID,
                       EXP055_TRAINPATH_MONITORED_300_RUN_ID,
                       EXP055_TRAINPATH_RECOVERY_RUN_ID,
                       EXP055_TRANSIGEN_LEAN_COST_RUN_ID,
                       EXP055_TRANSIGEN_FIXED300_LEAN_RUN_ID,
                       EXP055_TRANSIGEN_CPU_REPLAY_RUN_ID,
                       EXP055_TRANSIGEN_CPU_RESUME_RUN_ID}
        and config_run_id == "20261007_exp055_representative_models_v1"
        and resource_run_id == "20261007_exp055_representative_models_v1")
    approved_cpu_resume_config = (
        expected_experiment_id == "EXP-055"
        and ((RUN_ID in {EXP055_TRANSIGEN_CPU_REPLAY_RUN_ID,
                         EXP055_TRANSIGEN_CPU_RESUME_RUN_ID}
              and path.name == EXP055_TRANSIGEN_CPU_RESUME_CONFIG
              and config_run_id == EXP055_TRANSIGEN_CPU_RESUME_RUN_ID
              and resource_run_id == EXP055_TRANSIGEN_CPU_RESUME_RUN_ID)
             or (RUN_ID == EXP055_TRANSIGEN_CPU_NORM_RECOVERY_RUN_ID
                 and path.name == EXP055_TRANSIGEN_CPU_NORM_RECOVERY_CONFIG
                 and config_run_id == EXP055_TRANSIGEN_CPU_NORM_RECOVERY_RUN_ID
                 and resource_run_id == EXP055_TRANSIGEN_CPU_NORM_RECOVERY_RUN_ID)))
    approved_exp059_config = (
        expected_experiment_id == "EXP-059"
        and RUN_ID == EXP059_TRANSIGEN_LR_STABILITY_RUN_ID
        and path.name == EXP059_TRANSIGEN_LR_STABILITY_CONFIG
        and config_run_id == EXP059_TRANSIGEN_LR_STABILITY_RUN_ID
        and resource_run_id == EXP059_TRANSIGEN_LR_STABILITY_RUN_ID)
    if (not approved_exp055_recovery_run and not approved_cpu_resume_config
            and not approved_exp059_config
            and (config_run_id != RUN_ID or resource_run_id != RUN_ID)):
        raise ValueError(f"configuration run_id differs from the selected {expected_experiment_id} package")
    device_policy = cfg["authorization"]["training_device_policy"]
    cpu_representative_config = (
        ((expected_experiment_id == "EXP-055"
        and ((RUN_ID == EXP055_TRANSIGEN_CPU_REPRESENTATIVE_RUN_ID
              and path.name == EXP055_TRANSIGEN_CPU_REPRESENTATIVE_CONFIG)
             or approved_cpu_resume_config))
         or approved_exp059_config)
    )
    if cpu_representative_config:
        if (device_policy != "cpu_only_zero_gpu"
                or float(cfg["authorization"].get("cpu_total_cap_minutes", -1)) != 120.0
                or float(cfg["authorization"].get("gpu_hours_cap", -1)) != 0.0):
            raise ValueError("registered CPU representative run must retain its 120-minute/zero-GPU cap")
    elif device_policy != "approved_A5000_only":
        raise ValueError("training device policy changed")
    allowed_uuid = cfg["authorization"].get("allowed_device_uuid")
    if allowed_uuid is not None and allowed_uuid != GPU_UUID:
        raise ValueError("configured training UUID differs from the one authorized A5000")
    return cfg, sha256_file(path)


def deterministic_role(object_id: str) -> str:
    rank = int(sha256_text(f"EXP051|{SEED}|{object_id}")[:16], 16) / 16**16
    return "inner_dev" if rank < 0.2 else "train"


def metadata_phase(cfg: dict[str, Any], config_sha: str) -> dict[str, Any]:
    """Freeze condition keys/whole-object roles without touching either matrix."""
    if RUN_DIR.exists():
        raise FileExistsError(f"refusing to overwrite run artifacts: {RUN_DIR}")
    if not H5AD.is_file() or not DRUG_MAP.is_file():
        raise FileNotFoundError(f"required XPert H5AD or identity/feature table missing: {H5AD}, {DRUG_MAP}")

    import anndata as ad

    adata = ad.read_h5ad(H5AD, backed="r")
    try:
        shape = tuple(map(int, adata.shape))
        if shape != (78453, 978):
            raise ValueError(f"unexpected H5AD shape: {shape}")
        if "X_ctl" not in adata.obsm:
            raise ValueError("H5AD lacks paired obsm['X_ctl']")
        genes = list(map(str, adata.var_names))
        if len(genes) != 978 or len(set(genes)) != 978:
            raise ValueError("the H5AD var axis is not 978 unique names")
        obs = adata.obs.copy()
        obs_names = pd.Index(adata.obs_names.astype(str))
    finally:
        adata.file.close()

    required = ["sample_id", "pert_id", "pert_idx", "cell_iname", "cell_idx",
                "tissue_idx", "pert_dose", "pert_dose_unit", "pert_time",
                "pert_time_unit", "project_code", "build_name"]
    missing = [name for name in required if name not in obs.columns]
    if missing:
        raise ValueError(f"H5AD metadata lacks identity/condition columns: {missing}")
    if not obs_names.is_unique:
        raise ValueError("H5AD obs_names are not unique")
    obs = obs.reset_index(drop=True)
    obs["source_row"] = np.arange(len(obs), dtype=np.int64)
    obs["obs_name"] = obs_names.to_numpy(dtype=str)
    obs["source_object_id"] = obs["cell_iname"].astype("string").str.strip()
    obs["pert_id"] = obs["pert_id"].astype("string").str.strip()
    obs["sample_id"] = obs["sample_id"].astype("string").str.strip()
    if obs["source_object_id"].isna().any() or obs["source_object_id"].eq("").any():
        raise ValueError("missing cell-line identity in H5AD metadata")
    if (obs.groupby("source_object_id", dropna=False)["cell_idx"].nunique(dropna=False) > 1).any():
        raise ValueError("cell_idx maps one cell-line object to multiple source indices")
    if (obs.groupby("cell_idx", dropna=False)["source_object_id"].nunique(dropna=False) > 1).any():
        raise ValueError("cell_idx is not one-to-one with cell-line object IDs")

    drug_map = pd.read_csv(DRUG_MAP, usecols=["pert_id", "inchi_key", "drug_node_idx"],
                           dtype={"pert_id": "string", "inchi_key": "string"},
                           keep_default_na=False)
    drug_map["pert_id"] = drug_map["pert_id"].str.strip()
    drug_map["full_inchikey"] = drug_map["inchi_key"].str.strip().str.upper()
    drug_map["drug_node_idx"] = pd.to_numeric(drug_map["drug_node_idx"], errors="coerce")
    drug_map = drug_map.loc[drug_map["full_inchikey"].str.fullmatch(FULL_KEY, na=False)
                            & drug_map["drug_node_idx"].notna()].copy()
    if drug_map.duplicated(["pert_id"]).any():
        mapping_counts = drug_map.groupby("pert_id").agg(
            keys=("full_inchikey", "nunique"), nodes=("drug_node_idx", "nunique"))
        bad_ids = set(mapping_counts.index[(mapping_counts["keys"] != 1)
                                           | (mapping_counts["nodes"] != 1)])
        if bad_ids:
            raise ValueError(f"drug feature table has ambiguous pert_id mappings: {len(bad_ids)}")
        drug_map = drug_map.drop_duplicates("pert_id")
    mapped = obs.merge(drug_map[["pert_id", "full_inchikey", "drug_node_idx"]],
                       on="pert_id", how="left", validate="many_to_one")
    valid = mapped["full_inchikey"].str.fullmatch(FULL_KEY, na=False)
    mapped = mapped.loc[valid].copy()
    if mapped.empty:
        raise ValueError("no H5AD perturbations have a complete full-InChIKey identity")
    mapped["drug_node_idx"] = mapped["drug_node_idx"].astype(np.int64)
    mapped["pert_dose_num"] = pd.to_numeric(mapped["pert_dose"], errors="coerce")
    mapped["pert_time_num"] = pd.to_numeric(mapped["pert_time"], errors="coerce")
    if mapped[["pert_dose_num", "pert_time_num"]].isna().any().any():
        raise ValueError("mapped chemical conditions contain missing/non-numeric dose or time")

    # The key includes all source-side condition fields needed to distinguish rows;
    # it is built only from obs metadata and contains no expression/response values.
    key_fields = ["source_object_id", "full_inchikey", "pert_id", "pert_dose_num",
                  "pert_dose_unit", "pert_time_num", "pert_time_unit", "project_code",
                  "build_name", "sample_id", "obs_name"]
    mapped["full_condition_key"] = mapped[key_fields].astype("string").fillna("").agg("|".join, axis=1)
    mapped["split_role"] = mapped["source_object_id"].map(deterministic_role)

    # Exactly one row per whole source cell object x chemical identity. If the
    # object has several dose/time/source rows, dose, time, then full condition key
    # settle the selection, independent of X and X_ctl.
    one_per_pair = (mapped.sort_values(["full_inchikey", "source_object_id", "pert_dose_num",
                                        "pert_time_num", "full_condition_key"], kind="mergesort")
                    .drop_duplicates(["source_object_id", "full_inchikey"], keep="first"))
    cover = one_per_pair.groupby("full_inchikey", as_index=False).agg(
        source_coverage=("source_object_id", "nunique"),
        n_train_objects=("split_role", lambda x: int(x.eq("train").sum())),
        n_dev_objects=("split_role", lambda x: int(x.eq("inner_dev").sum())))
    cover = cover.sort_values(["source_coverage", "full_inchikey"],
                              ascending=[False, True], kind="mergesort")
    preferred = int(cfg["data"]["sampling"]["preferred_min_sources_per_drug"])
    fallback_min = int(cfg["data"]["sampling"]["fallback_min_sources_per_drug"])
    fallback_conditions = int(cfg["data"]["sampling"]["fallback_min_conditions_per_drug"])
    use_preferred = bool((cover["source_coverage"] >= preferred).any())
    minimum_coverage = preferred if use_preferred else fallback_min
    minimum_conditions = preferred if use_preferred else fallback_conditions
    eligible_keys = set(cover.loc[
        cover["source_coverage"].ge(minimum_coverage)
        & cover["n_train_objects"].gt(0) & cover["n_dev_objects"].gt(0)
        & cover["source_coverage"].ge(minimum_conditions), "full_inchikey"])
    cover = cover.loc[cover["full_inchikey"].isin(eligible_keys)].copy()
    cover = cover.sort_values(["source_coverage", "full_inchikey"], ascending=[False, True], kind="mergesort")
    max_conditions = int(cfg["data"]["upstream_training_asset"]["response_blind_selection_max_conditions"])
    selected_keys: list[str] = []
    selected_count = 0
    for key in cover["full_inchikey"].astype(str):
        group_n = int(one_per_pair["full_inchikey"].eq(key).sum())
        if selected_count + group_n > max_conditions:
            continue
        selected_keys.append(key)
        selected_count += group_n
    selected = one_per_pair.loc[one_per_pair["full_inchikey"].isin(selected_keys)].copy()
    drug_rank = {key: rank for rank, key in enumerate(selected_keys, 1)}
    selected["drug_selection_rank"] = selected["full_inchikey"].map(drug_rank).astype(int)
    selected = selected.sort_values(["drug_selection_rank", "split_role", "source_object_id",
                                     "full_condition_key"], kind="mergesort").reset_index(drop=True)
    selected["selection_rank"] = np.arange(1, len(selected) + 1, dtype=np.int64)
    if len(selected) > max_conditions or selected["full_inchikey"].nunique() < 2:
        raise ValueError(f"response-blind complete-group selection yielded {len(selected)} rows and "
                         f"{selected['full_inchikey'].nunique()} drugs; need 2+ drugs and <= {max_conditions}")
    if selected.duplicated(["source_object_id", "full_inchikey"]).any():
        raise AssertionError("selection duplicated an object x drug pair")
    train_ids = set(selected.loc[selected.split_role.eq("train"), "source_object_id"].astype(str))
    dev_ids = set(selected.loc[selected.split_role.eq("inner_dev"), "source_object_id"].astype(str))
    if train_ids & dev_ids or not train_ids or not dev_ids:
        raise AssertionError("whole-object train/inner-development split is invalid")
    if not selected.groupby("full_inchikey")["source_object_id"].apply(
            lambda values: values.is_unique).all():
        raise AssertionError("complete-drug selection has a duplicate source object")

    RUN_DIR.mkdir(parents=True, exist_ok=False)
    keep = ["selection_rank", "drug_selection_rank", "source_row", "obs_name", "sample_id",
            "source_object_id", "split_role", "full_inchikey", "pert_id", "drug_node_idx",
            "pert_dose_num", "pert_dose_unit", "pert_time_num", "pert_time_unit",
            "project_code", "build_name", "full_condition_key"]
    selected[keep].to_csv(RUN_DIR / "response_blind_conditions.tsv", sep="\t", index=False,
                          lineterminator="\n")
    coverage = selected.groupby(["full_inchikey", "drug_selection_rank"], as_index=False).agg(
        n_conditions=("source_row", "size"), n_sources=("source_object_id", "nunique"),
        n_train=("split_role", lambda x: int(x.eq("train").sum())),
        n_inner_dev=("split_role", lambda x: int(x.eq("inner_dev").sum())))
    coverage.to_csv(RUN_DIR / "drug_source_coverage.tsv", sep="\t", index=False, lineterminator="\n")

    result = {
        "experiment_id": "EXP-051", "run_id": RUN_ID, "stage": "RESPONSE_BLIND_SELECTION_FROZEN",
        "created_at_utc": utc_now(), "config_sha256": config_sha,
        "source_h5ad": str(H5AD), "source_drug_identity_table": str(DRUG_MAP),
        "source_shape": list(shape), "n_obs_metadata_rows": len(obs),
        "gene_axis": {"n": len(genes), "sha256": sha256_text("\n".join(genes) + "\n"),
                      "first_10": genes[:10]},
        "matrix_access": {"X_read": False, "X_ctl_read": False,
                          "claim": "metadata and full structure keys only; response values not opened"},
        "identity": {"source_object_definition": "H5AD cell_iname; all dose/time/project rows from one cell-line object stay together",
                     "drug_identity": "unique complete InChIKey from XPert l1000_sdst_drugs_info_8432.csv",
                     "full_condition_key_fields": key_fields,
                     "all_obs_names_unique": bool(obs_names.is_unique),
                     "n_source_objects_all": int(obs.source_object_id.nunique()),
                     "n_mapped_condition_rows": int(len(mapped)),
                     "n_unmapped_or_nonchemical_rows_excluded_by_identity_only": int(len(obs) - len(mapped)),
                     "cell_idx_one_to_one_with_cell_iname": True},
        "selection": {"seed": SEED, "group_split": "stable SHA256 rank by source_object_id; first 20% inner_dev, remainder train",
                      "preferred_min_sources_per_drug": preferred,
                      "fallback_used": not use_preferred,
                      "actual_min_sources_per_drug": minimum_coverage,
                      "per_object_drug_conditions": 1,
                      "dose_time_tie_break": "ascending numeric dose, ascending numeric time, then full_condition_key",
                      "drug_order": "source coverage descending, full InChIKey ascending",
                      "whole_drug_group_cap": max_conditions,
                      "conditions_selected": int(len(selected)),
                      "drugs_selected": int(selected.full_inchikey.nunique()),
                      "train_source_objects": len(train_ids), "inner_dev_source_objects": len(dev_ids),
                      "train_conditions": int(selected.split_role.eq("train").sum()),
                      "inner_dev_conditions": int(selected.split_role.eq("inner_dev").sum()),
                      "selected_condition_ids_sha256": sha256_text("\n".join(selected.full_condition_key) + "\n"),
                      "train_object_ids_sha256": sha256_text("\n".join(sorted(train_ids)) + "\n"),
                      "inner_dev_object_ids_sha256": sha256_text("\n".join(sorted(dev_ids)) + "\n")},
        "file_sha256": {"h5ad": sha256_file(H5AD), "drug_identity_table": sha256_file(DRUG_MAP),
                        "runner_at_freeze": sha256_file(Path(__file__).resolve()),
                        "condition_keys": sha256_file(RUN_DIR / "response_blind_conditions.tsv"),
                        "drug_coverage": sha256_file(RUN_DIR / "drug_source_coverage.tsv")},
    }
    atomic_json(RUN_DIR / "metadata_freeze_receipt.json", result)
    print(json.dumps({k: result[k] for k in ("run_id", "source_shape", "gene_axis", "identity", "selection")},
                     ensure_ascii=False, indent=2))
    return result


def values_phase(cfg: dict[str, Any], config_sha: str) -> dict[str, Any]:
    """Open only the already-frozen train/inner-dev rows and check X contracts."""
    freeze_path = RUN_DIR / "metadata_freeze_receipt.json"
    keys_path = RUN_DIR / "response_blind_conditions.tsv"
    if not freeze_path.is_file() or not keys_path.is_file():
        raise FileNotFoundError("run metadata freeze is required before opening any expression matrix")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("config_sha256") != config_sha:
        raise ValueError("frozen response-blind key set belongs to a different config")
    axis_path = RUN_DIR / "gene_axis_receipt.json"
    if not axis_path.is_file():
        raise RuntimeError("hard gate: verify the H5AD's 978-gene identity axis before opening X or X_ctl")
    axis_receipt = json.loads(axis_path.read_text(encoding="utf-8"))
    if axis_receipt.get("status") != "PASS_EXACT_978_AXIS" or axis_receipt.get("mapping_rows") != 978:
        raise RuntimeError("hard gate: the 978-gene identity mapping did not pass")
    if RUN_DIR.joinpath("frozen_train_inner_values.npz").exists():
        raise FileExistsError("train/inner values already materialized; refusing to overwrite")
    selected = pd.read_csv(keys_path, sep="\t", dtype={"source_object_id": str,
                                                          "full_inchikey": str,
                                                          "split_role": str})
    if len(selected) != int(freeze["selection"]["conditions_selected"]):
        raise ValueError("frozen key table row count changed")
    allowed_roles = {"train", "inner_dev"}
    if not set(selected["split_role"]).issubset(allowed_roles):
        raise ValueError("frozen key table contains a forbidden response role")
    train_ids = set(selected.loc[selected.split_role.eq("train"), "source_object_id"])
    dev_ids = set(selected.loc[selected.split_role.eq("inner_dev"), "source_object_id"])
    if train_ids & dev_ids:
        raise ValueError("source object identity leaks between train and inner development")

    import anndata as ad
    from scipy import sparse

    adata = ad.read_h5ad(H5AD, backed="r")
    try:
        genes = list(map(str, adata.var_names))
        var = adata.var.copy()
        rows = selected["source_row"].to_numpy(dtype=np.int64)
        order = np.argsort(rows, kind="mergesort")
        sorted_rows = rows[order]
        if len(np.unique(sorted_rows)) != len(sorted_rows):
            raise ValueError("frozen H5AD source rows are not unique")
        X_sorted = adata.X[sorted_rows, :]
        X_ctl_sorted = adata.obsm["X_ctl"][sorted_rows, :]
        # Restore the frozen selection_rank order after h5py's required sorted
        # fancy indexing. This is exact key-based reordering, never row-order join.
        if hasattr(X_sorted, "to_memory"):
            X_sorted = X_sorted.to_memory()
        if hasattr(X_ctl_sorted, "to_memory"):
            X_ctl_sorted = X_ctl_sorted.to_memory()
        if sparse.issparse(X_sorted):
            X_sorted = X_sorted.toarray()
        if sparse.issparse(X_ctl_sorted):
            X_ctl_sorted = X_ctl_sorted.toarray()
        X = np.empty((len(rows), 978), dtype=np.float32)
        X_ctl = np.empty_like(X)
        X[order] = X_sorted
        X_ctl[order] = X_ctl_sorted
    finally:
        adata.file.close()
    X = X.toarray() if sparse.issparse(X) else np.asarray(X)
    X_ctl = X_ctl.toarray() if sparse.issparse(X_ctl) else np.asarray(X_ctl)
    if X.shape != (len(selected), 978) or X_ctl.shape != X.shape:
        raise ValueError(f"selected X/X_ctl axis shape mismatch: {X.shape} vs {X_ctl.shape}")
    if not np.isfinite(X).all() or not np.isfinite(X_ctl).all():
        raise ValueError("selected training/inner-development X or X_ctl contains non-finite values")
    delta = X.astype(np.float32, copy=False) - X_ctl.astype(np.float32, copy=False)
    trt32 = X.astype(np.float32, copy=False)
    ctl32 = X_ctl.astype(np.float32, copy=False)
    round_trip = ctl32 + delta
    max_round_trip = float(np.max(np.abs(round_trip.astype(np.float64) - trt32.astype(np.float64))))
    if max_round_trip > 2e-6:
        raise ValueError(f"X != X_ctl + delta within float32 storage precision: max_abs={max_round_trip}")

    np.savez_compressed(RUN_DIR / "frozen_train_inner_values.npz",
                        selection_rank=selected.selection_rank.to_numpy(dtype=np.int64),
                        source_row=rows,
                        split_role=selected.split_role.astype(str).to_numpy(),
                        source_object_id=selected.source_object_id.astype(str).to_numpy(),
                        full_inchikey=selected.full_inchikey.astype(str).to_numpy(),
                        pert_id=selected.pert_id.astype(str).to_numpy(),
                        genes=np.asarray(genes, dtype=str),
                        X=trt32, X_ctl=ctl32, delta=delta)

    def stats(array: np.ndarray) -> dict[str, Any]:
        values = array.astype(np.float64, copy=False)
        return {"min": float(values.min()), "max": float(values.max()),
                "mean": float(values.mean()), "sd": float(values.std()),
                "q01": float(np.quantile(values, 0.01)),
                "q50": float(np.quantile(values, 0.50)),
                "q99": float(np.quantile(values, 0.99))}

    gene_annotation = {str(column): [str(x) for x in var[column].head(8).tolist()]
                       for column in var.columns}
    receipt = {
        "experiment_id": "EXP-051", "run_id": RUN_ID, "stage": "TRAIN_INNER_VALUE_CONTRACT_VERIFIED",
        "created_at_utc": utc_now(), "config_sha256": config_sha,
        "selection_receipt_sha256": sha256_file(freeze_path),
        "gene_axis_receipt_sha256": sha256_file(axis_path),
        "input": {"path": str(H5AD), "whole_file_sha256": freeze["file_sha256"]["h5ad"],
                  "H5AD_shape": [78453, 978], "selected_rows": len(rows),
                  "selected_source_rows_sha256": sha256_text("\n".join(map(str, rows)) + "\n")},
        "gene_axis": {"n": len(genes), "var_names_sha256": sha256_text("\n".join(genes) + "\n"),
                      "var_names_first_10": genes[:10], "var_annotations": gene_annotation,
                      "var_columns": list(map(str, var.columns)),
                      "numeric_var_names_are_gene_ids": False},
        "pair_contract": {"X_role": "source H5AD post-treatment measured layer",
                          "X_ctl_role": "paired pre-treatment control from obsm['X_ctl']",
                          "target": "delta = X - X_ctl in the source-provided replicate-collapsed z-score scale",
                          "all_values_preserved_without_rescaling": True,
                          "X_equals_X_ctl_plus_delta_max_abs_float32": max_round_trip,
                          "inverse_transform": "none available or applied; predictions remain in source H5AD z-score units"},
        "value_distribution": {"X": stats(trt32), "X_ctl": stats(ctl32), "delta": stats(delta),
                               "train_X": stats(trt32[selected.split_role.eq("train").to_numpy()]),
                               "inner_dev_X": stats(trt32[selected.split_role.eq("inner_dev").to_numpy()]),
                               "train_X_ctl": stats(ctl32[selected.split_role.eq("train").to_numpy()]),
                               "inner_dev_X_ctl": stats(ctl32[selected.split_role.eq("inner_dev").to_numpy()])},
        "XPert_upstream_range_check": {"official_config_n_bins": 128,
                                       "official_config_min_max": [0.0, 15.002],
                                       "source_H5AD_inferred_to_be_zscore": True,
                                       "fixed_range_covers_all_X_ctl": bool(ctl32.min() >= 0.0 and ctl32.max() <= 15.002),
                                       "warning": "The upstream fixed bin range is recorded for diagnosis. No target or input z-score, clipping, or rescaling is applied here."},
        "axis_gate": {"status": axis_receipt["status"],
                      "reference": axis_receipt["gene_identity_reference"],
                      "mapping_sha256": axis_receipt["mapping_sha256"],
                      "mapping_rows": axis_receipt["mapping_rows"]},
        "forbidden_opened": {"PANACEA_expression": False, "CDSDB_expression": False,
                             "PDO_sensitivity": False, "outer_or_locked_responses": False},
        "files_sha256": {"frozen_train_inner_values_npz": sha256_file(RUN_DIR / "frozen_train_inner_values.npz"),
                         "runner_at_value_read": sha256_file(Path(__file__).resolve())},
    }
    atomic_json(RUN_DIR / "value_contract_receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("stage", "input", "gene_axis", "pair_contract",
                                             "value_distribution", "XPert_upstream_range_check")},
                     ensure_ascii=False, indent=2))
    return receipt


def axis_phase(config_sha: str) -> dict[str, Any]:
    """Resolve the numeric AnnData var_names against the author's ordered 978 axis."""
    freeze_path = RUN_DIR / "metadata_freeze_receipt.json"
    if not freeze_path.is_file():
        raise FileNotFoundError("response-blind metadata/key freeze must precede the gene-axis audit")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("config_sha256") != config_sha or freeze.get("matrix_access", {}).get("X_read") is not False:
        raise ValueError("gene-axis audit is not bound to the frozen response-blind selection")
    receipt_path = RUN_DIR / "gene_axis_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("gene-axis receipt already exists; do not overwrite frozen axis evidence")

    import anndata as ad

    gene_info_path = Path("/mnt/d/Code/XPert/processed_data/l1000_gene_info_978.csv")
    if not gene_info_path.is_file():
        raise FileNotFoundError(f"author's 978-gene axis file is missing: {gene_info_path}")
    adata = ad.read_h5ad(H5AD, backed="r")
    try:
        h5_names = list(map(str, adata.var_names))
        var = adata.var.copy().reset_index(drop=False)
    finally:
        adata.file.close()
    expected = pd.read_csv(gene_info_path, dtype="string", keep_default_na=False)
    if len(h5_names) != 978 or len(expected) != 978:
        raise ValueError(f"expected 978 variables in both resources; H5AD={len(h5_names)} table={len(expected)}")
    if len(set(h5_names)) != 978:
        raise ValueError("H5AD var_names are not unique")

    # Build row-wise equality evidence across the H5AD's complete var annotations
    # and the author's explicit 978-gene map. Numeric var_names alone are never
    # treated as gene identifiers.
    candidates = [str(c) for c in expected.columns]
    pairs: list[dict[str, Any]] = []
    matched_column = None
    for column in candidates:
        if column not in var.columns:
            continue
        left = var[column].astype("string").fillna("").str.strip()
        right = expected[column].astype("string").fillna("").str.strip()
        if left.equals(right):
            matched_column = column
            break
    if matched_column is None:
        # If column names differ, detect a unique aligned metadata vector by
        # comparing every scalar text column, without looking at expression data.
        for left_col in map(str, var.columns):
            left = var[left_col].astype("string").fillna("").str.strip()
            if not left.ne("").all():
                continue
            for right_col in candidates:
                right = expected[right_col].astype("string").fillna("").str.strip()
                if left.equals(right):
                    matched_column = (left_col, right_col)
                    break
            if matched_column is not None:
                break

    if isinstance(matched_column, tuple):
        h5_column, source_column = matched_column
        h5_values = var[h5_column].astype("string").fillna("").str.strip()
        source_values = expected[source_column].astype("string").fillna("").str.strip()
        proof = "exact row-by-row H5AD var annotation to author's l1000_gene_info_978.csv"
    elif isinstance(matched_column, str):
        h5_column = source_column = matched_column
        h5_values = var[h5_column].astype("string").fillna("").str.strip()
        source_values = expected[source_column].astype("string").fillna("").str.strip()
        proof = "exact row-by-row shared-column match"
    else:
        h5_column = None
        source_column = None
        h5_values = pd.Series([""] * 978, dtype="string")
        source_values = pd.Series([""] * 978, dtype="string")
        proof = "no metadata identity mapping found"

    passed = (h5_column is not None and h5_values.equals(source_values)
              and h5_values.nunique() == 978)
    if passed:
        for i in range(978):
            pairs.append({"axis_index": i, "h5ad_var_name": h5_names[i],
                          "h5ad_var_annotation": str(h5_values.iloc[i]),
                          "l1000_gene_info_value": str(source_values.iloc[i])})
        pd.DataFrame(pairs).to_csv(RUN_DIR / "gene_axis_mapping.tsv", sep="\t", index=False,
                                  lineterminator="\n")

    result = {
        "experiment_id": "EXP-051", "run_id": RUN_ID,
        "status": "PASS_EXACT_978_AXIS" if passed else "BLOCKED_AXIS_IDENTITY_UNRESOLVED",
        "created_at_utc": utc_now(), "config_sha256": config_sha,
        "metadata_freeze_receipt_sha256": sha256_file(freeze_path),
        "H5AD": str(H5AD), "gene_identity_reference": str(gene_info_path),
        "H5AD_var_names_unique_but_not_gene_ids": bool(len(set(h5_names)) == 978 and h5_names[:10] == [str(i) for i in range(10)]),
        "H5AD_var_columns": list(map(str, var.columns)),
        "H5AD_var_first_8": {str(c): [str(x) for x in var[c].head(8).tolist()] for c in var.columns},
        "reference_columns": list(map(str, expected.columns)),
        "reference_first_8": {str(c): [str(x) for x in expected[c].head(8).tolist()] for c in expected.columns},
        "matched_columns": {"h5ad": h5_column, "reference": source_column},
        "mapping_rows": len(pairs) if passed else 0,
        "mapping_proof": proof,
        "mapping_sha256": sha256_text("\n".join(str(x["h5ad_var_annotation"]) for x in pairs) + "\n") if passed else None,
        "reference_file_sha256": sha256_file(gene_info_path),
        "X_read": False, "X_ctl_read": False,
        "prior_failed_command": {"command": "--phase values",
                                 "status": "FAILED_BEFORE_ANY_MATRIX_VALUE_RETURNED",
                                 "error": "HDF5 requires source row indices in increasing order; X indexing rejected unsorted frozen ranks",
                                 "gpu_hours": 0.0},
    }
    atomic_json(receipt_path, result)
    if passed:
        result["files_sha256"] = {"gene_axis_mapping.tsv": sha256_file(RUN_DIR / "gene_axis_mapping.tsv"),
                                  "runner_at_axis_audit": sha256_file(Path(__file__).resolve())}
        atomic_json(receipt_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def safe_number(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def keyed_metrics(keys: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray,
                  gene_variance: np.ndarray) -> dict[str, Any]:
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 978:
        raise ValueError("prediction/target arrays do not share the frozen 978-gene axis")
    use_gene = np.isfinite(gene_variance) & (gene_variance > 1e-12)
    if not use_gene.any():
        raise ValueError("training-only gene variance has no positive denominator")
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    target = truth.astype(np.float64)
    per_row_nmse = np.mean(np.square(error[:, use_gene]) / gene_variance[use_gene][None, :], axis=1)
    per_row_mae = np.mean(np.abs(error), axis=1)
    per_row_corr: list[float | None] = []
    per_row_direction: list[float | None] = []
    for yt, yp in zip(target, prediction.astype(np.float64)):
        if np.ptp(yt) <= 0 or np.ptp(yp) <= 0:
            per_row_corr.append(None)
        else:
            per_row_corr.append(safe_number(float(np.corrcoef(yt, yp)[0, 1])))
        nonzero = np.abs(yt) > 1e-12
        per_row_direction.append(safe_number(float(np.mean(np.sign(yt[nonzero]) == np.sign(yp[nonzero]))))
                                 if nonzero.any() else None)
    row = keys[["source_object_id", "full_inchikey"]].reset_index(drop=True).copy()
    if len(row) != len(truth):
        raise ValueError("keyed metric denominator differs from the frozen condition count")
    row["nmse"] = per_row_nmse
    row["mae"] = per_row_mae
    row["correlation"] = per_row_corr
    row["direction"] = per_row_direction
    per_object = row.groupby("source_object_id", sort=True, as_index=False).agg(
        nmse=("nmse", "mean"), mae=("mae", "mean"),
        correlation=("correlation", "mean"), direction=("direction", "mean"))
    amplitude_rows = []
    for object_id, positions in row.groupby("source_object_id", sort=True).indices.items():
        ix = np.asarray(positions, dtype=np.int64)
        observed = float(np.mean(np.abs(target[ix])))
        predicted = float(np.mean(np.abs(prediction[ix])))
        amplitude_rows.append({"source_object_id": str(object_id),
                               "ratio": safe_number(predicted / observed) if observed > 0 else None,
                               "observed_mean_abs": observed, "predicted_mean_abs": predicted})
    amp = [x["ratio"] for x in amplitude_rows if x["ratio"] is not None]
    return {
        "n_conditions": int(len(row)), "n_source_objects": int(row.source_object_id.nunique()),
        "n_full_inchikeys": int(row.full_inchikey.nunique()),
        "n_genes_in_nmse_denominator": int(use_gene.sum()),
        "n_zero_variance_genes_excluded": int((~use_gene).sum()),
        "object_macro_nmse": safe_number(float(per_object.nmse.mean())),
        "object_macro_mae": safe_number(float(per_object.mae.mean())),
        "object_macro_gene_correlation": safe_number(float(per_object.correlation.mean())),
        "object_macro_direction_agreement": safe_number(float(per_object.direction.mean())),
        "object_macro_amplitude_ratio": safe_number(float(np.mean(amp))) if amp else None,
        "n_amplitude_objects_with_nonzero_observed_mean_abs": int(len(amp)),
        "aggregation": "conditions averaged within source cell-line object, then objects equally weighted",
        "normalizer": "per-gene variance fitted on TRAIN delta rows only; zero-variance genes excluded and counted",
    }


def data_baselines(keys: pd.DataFrame, delta: np.ndarray, gene_variance: np.ndarray,
                   train_mask: np.ndarray, dev_mask: np.ndarray, out: Path) -> dict[str, Any]:
    train_keys = keys.loc[train_mask].reset_index(drop=True)
    dev_keys = keys.loc[dev_mask].reset_index(drop=True)
    train_y = delta[train_mask]
    dev_y = delta[dev_mask]
    global_mean = train_y.mean(axis=0, dtype=np.float64).astype(np.float32)
    by_key: dict[str, np.ndarray] = {}
    for full_key in sorted(train_keys.full_inchikey.astype(str).unique()):
        by_key[full_key] = train_y[
            train_keys.full_inchikey.astype(str).to_numpy() == full_key].mean(axis=0, dtype=np.float64).astype(np.float32)
    output: dict[str, Any] = {"fit_policy": "zero/global/per-drug means fit on TRAIN only",
                              "metrics": {}, "predictions": {}}
    for name in ("zero_change", "global_mean_delta", "per_drug_mean_delta"):
        pred_train = np.zeros_like(train_y) if name == "zero_change" else np.tile(global_mean, (len(train_y), 1))
        pred_dev = np.zeros_like(dev_y) if name == "zero_change" else np.tile(global_mean, (len(dev_y), 1))
        if name == "per_drug_mean_delta":
            pred_train = np.stack([by_key[str(k)] for k in train_keys.full_inchikey], axis=0)
            pred_dev = np.stack([by_key[str(k)] for k in dev_keys.full_inchikey], axis=0)
        output["metrics"][name] = {
            "train": keyed_metrics(train_keys, train_y, pred_train, gene_variance),
            "inner_dev": keyed_metrics(dev_keys, dev_y, pred_dev, gene_variance),
        }
        output["predictions"][name] = {"train": pred_train, "inner_dev": pred_dev}
    with np.load(SOURCE_RUN_DIR / "frozen_train_inner_values.npz", allow_pickle=False) as frozen:
        genes = frozen["genes"].copy()
    np.savez_compressed(out / "baseline_predictions.npz",
                        train_condition_key=np.asarray(train_keys.full_condition_key.astype(str), dtype=str),
                        inner_dev_condition_key=np.asarray(dev_keys.full_condition_key.astype(str), dtype=str),
                        train_target_delta=train_y, inner_dev_target_delta=dev_y,
                        genes=genes,
                        **{f"{name}_{role}": arr for name, by_role in output["predictions"].items()
                           for role, arr in by_role.items()})
    del output["predictions"]
    atomic_json(out / "baseline_metrics.json", output)
    return output


def _float64_global_l2_norm(torch: Any, tensors: list[Any]) -> float:
    """Accumulate diagnostic L2 norms in float64 without changing model math."""
    squared = 0.0
    for tensor in tensors:
        value = tensor.detach().to(dtype=torch.float64)
        if value.numel() == 0:
            continue
        squared += float(torch.sum(value * value, dtype=torch.float64).item())
        if not math.isfinite(squared):
            return math.inf
    return math.sqrt(squared)


def _grad_global_norm(torch: Any, model: Any) -> float:
    return _float64_global_l2_norm(torch, [param.grad for param in model.parameters()
                                           if param.grad is not None])


def _param_global_norm(torch: Any, model: Any) -> float:
    return _float64_global_l2_norm(torch, list(model.parameters()))


def _norm_difference(torch: Any, before: list[Any], model: Any) -> float:
    differences: list[Any] = []
    with torch.no_grad():
        for old, parameter in zip(before, model.parameters()):
            # Cast before subtraction so opposite large finite float32 values do not overflow.
            differences.append(parameter.detach().to(dtype=torch.float64)
                               - old.detach().to(dtype=torch.float64))
    return _float64_global_l2_norm(torch, differences)


def _source_functions(train_file: Path, utils_file: Path) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch

    tree = ast.parse(train_file.read_text(encoding="utf-8"))
    source_train = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                         and node.name == "train"), None)
    if source_train is None:
        raise RuntimeError("official XPert train_xpert.py has no top-level train function")
    # The upstream no-scaler branch never assigns this optional variable, but
    # reads it after the epoch loop to format the returned loss. Initialize it
    # to None in the adapter copy; this leaves its objective and updates intact.
    loop_index = next((i for i, node in enumerate(source_train.body)
                       if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                       and node.target.id == "data"), None)
    if loop_index is None:
        raise RuntimeError("official XPert train function has no batch loop")
    source_train.body.insert(loop_index, ast.Assign(
        targets=[ast.Name(id="cell_class_predict", ctx=ast.Store())],
        value=ast.Constant(value=None)))
    ast.fix_missing_locations(source_train)
    utils_tree = ast.parse(utils_file.read_text(encoding="utf-8"))
    needed = {"mse_loss_ls_sum", "pcc_loss_sum"}
    selected = [node for node in utils_tree.body if isinstance(node, ast.FunctionDef)
                and node.name in needed]
    if {node.name for node in selected} != needed:
        raise RuntimeError("official XPert utils.py lacks its MSE/Pearson loss functions")
    ns: dict[str, Any] = {"np": np, "torch": torch, "nn": torch.nn}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
                 str(utils_file), "exec"), ns)
    train_ns = dict(ns)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[source_train], type_ignores=[])),
                 str(train_file), "exec"), train_ns)
    return train_ns["train"], ns["mse_loss_ls_sum"], ns["pcc_loss_sum"], {
        "train_function_file": str(train_file), "train_function_line_start": int(source_train.lineno),
        "train_function_line_end": int(source_train.end_lineno),
        "train_function_source_sha256": sha256_file(train_file),
        "adapter_patch": "initialize optional cell_class_predict=None before upstream loop; objective and optimizer updates are unchanged",
        "loss_source_file": str(utils_file), "loss_functions_source_sha256": sha256_file(utils_file),
    }


def _training_modules(torch: Any) -> tuple[Any, Any, Any, Any]:
    sys.path.insert(0, str(XPERT_ROOT))
    spec = importlib.util.spec_from_file_location("exp051_original_xpert_model", XPERT_ROOT / "models/model_XPert.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the original XPertNet module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logger = type("SilentLogger", (), {"info": lambda self, message: None})()
    config = yaml.safe_load((XPERT_ROOT / "configs/config_l1000.yaml").read_text(encoding="utf-8"))
    config["model"]["ATTN"]["ppi_gene_vector_path"] = str(XPERT_ROOT / "processed_data/PPI_gene_vector_128d.npy")
    config["model"]["HG"]["drug_hg_pretrained_embed_path"] = str(XPERT_ROOT / "HG_data/saved_embedding/HG_drug_embeddings.npy")
    return module, config, logger, sys.path


def _new_xpert(torch: Any, module: Any, config: dict[str, Any], device: Any, seed: int) -> tuple[Any, Any]:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args = type("EXP051XPertArgs", (), {})()
    args.device = device
    args.dataset = "l1000_sdst"
    args.drug_feat = "unimol"
    args.pretrained_mode = "global"
    args.include_cell_idx = False
    args.mode = "train"
    args.output_attention = False
    args.use_gene_pos_emed = False
    args.wo_ppi = False
    args.wo_HG = False
    args.wo_atom = False
    args.wo_atom_HG = False
    args.wo_unimol = False
    args.pretrained_model = False
    core = module.XPertNet(args, config, device, type("SilentLogger", (), {"info": lambda self, message: None})())
    core.init_weights()
    class TrainView(torch.nn.Module):
        def __init__(self, wrapped: Any):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, batch: Any) -> tuple[Any, ...]:
            return self.wrapped(batch)[:6]
    return core.to(device), TrainView(core).to(device)


class ProbeOptimizer:
    def __init__(self, optimizer: Any, model: Any, torch: Any,
                 progress_check: Any | None = None):
        self.optimizer = optimizer
        self.model = model
        self.torch = torch
        self.gradient_norms: list[float] = []
        self.progress_check = progress_check

    def step(self) -> Any:
        if self.progress_check is not None:
            self.progress_check()
        value = _grad_global_norm(self.torch, self.model)
        if not math.isfinite(value):
            raise FloatingPointError("non-finite XPert gradient norm")
        self.gradient_norms.append(value)
        return self.optimizer.step()


def _dose_idx(value: float) -> int:
    edges = [(0.21, 0), (0.41, 1), (0.71, 2), (1.01, 3), (1.51, 4),
             (3.1, 5), (4.1, 6), (7.1, 7), (12.1, 8)]
    for upper, index in edges:
        if float(value) < upper:
            return index
    return 9


def _xpert_tensors(torch: Any, values: dict[str, np.ndarray], keys: pd.DataFrame,
                   positions: np.ndarray, unimol: np.ndarray, config: dict[str, Any],
                   *, training_targets: bool) -> tuple[Any, ...]:
    X = values["X"][positions]
    ctl = values["X_ctl"][positions]
    rows = keys.iloc[positions]
    drug_nodes = rows.drug_node_idx.to_numpy(dtype=np.int64)
    drug_feat = np.asarray(unimol[drug_nodes], dtype=np.float32)
    low, high = float(config["dataset"]["min_value"]), float(config["dataset"]["max_value"])
    n_bins = int(config["dataset"]["n_bins"])
    bins = np.quantile(np.asarray([low, high]), np.linspace(0.0, 1.0, n_bins - 1))
    # Import only the official deterministic digitizer used by MyDataset.
    source = XPERT_ROOT / "datasets/MyDataset.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next((item for item in tree.body if isinstance(item, ast.FunctionDef)
                 and item.name == "_digitize"), None)
    if node is None:
        raise RuntimeError("official XPert MyDataset.py lacks _digitize")
    ns = {"np": np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(source), "exec"), ns)
    digitize = ns["_digitize"]
    if training_targets:
        trt_input = X.astype(np.float32, copy=False)
        trt_bins = digitize(trt_input, bins, side="one").astype(np.int64)
    else:
        trt_input = np.zeros_like(X, dtype=np.float32)
        trt_bins = np.zeros_like(X, dtype=np.int64)
    ctl_bins = digitize(ctl.astype(np.float32, copy=False), bins, side="one").astype(np.int64)
    dose_idx = np.asarray([_dose_idx(float(x)) for x in rows.pert_dose_num], dtype=np.int64)
    time_map = {3.0: 0, 6.0: 1, 24.0: 2}
    time_idx = np.asarray([time_map[float(x)] for x in rows.pert_time_num], dtype=np.int64)
    zeros = np.zeros(len(positions), dtype=np.int64)
    return (
        torch.from_numpy(trt_input), torch.from_numpy(ctl.astype(np.float32, copy=False)),
        torch.from_numpy(trt_bins), torch.from_numpy(ctl_bins), torch.from_numpy(drug_feat),
        torch.from_numpy(dose_idx), torch.from_numpy(time_idx), torch.from_numpy(drug_nodes),
        torch.from_numpy(zeros.copy()), torch.from_numpy(zeros.copy()),
    )


def _xpert_predict(torch: Any, model: Any, values: dict[str, np.ndarray], keys: pd.DataFrame,
                   positions: np.ndarray, unimol: np.ndarray, config: dict[str, Any],
                   batch_size: int = 64,
                   progress_check: Any | None = None) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            if progress_check is not None:
                progress_check()
            idx = positions[start:start + batch_size]
            inputs = _xpert_tensors(torch, values, keys, idx, unimol, config, training_targets=False)
            prediction = model(inputs)[2]
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("non-finite XPert prediction")
            chunks.append(prediction.detach().cpu().numpy().astype(np.float32))
            if progress_check is not None:
                progress_check()
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 978), dtype=np.float32)


def _load_transigen(torch: Any, device: Any, seed: int, features_dim: int,
                    *, seed_cuda: bool = True) -> tuple[Any, Any]:
    if str(TRANSIGEN_ROOT) not in sys.path:
        sys.path.insert(0, str(TRANSIGEN_ROOT))
    model_path = TRANSIGEN_ROOT / "model.py"
    spec = importlib.util.spec_from_file_location("exp051_original_transigen_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load original TranSiGen model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    if seed_cuda:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        torch.random.default_generator.manual_seed(seed)
    model = module.TranSiGen(n_genes=978, n_latent=100, n_en_hidden=[1200], n_de_hidden=[800],
                             features_dim=features_dim, features_embed_dim=[400],
                             init_w=True, beta=0.1, device=device, dropout=0.1,
                             path_model=str(RUN_DIR) + os.sep, random_seed=seed)
    return model.to(device), module


def _transigen_forward_batch(torch: Any, model: Any, batch: Any,
                             device: Any, diagnostic_capture: dict[str, Any] | None = None
                             ) -> tuple[Any, Any, Any, tuple[Any, ...]]:
    """Return exactly (control, treated, treated_prediction, seven loss terms)."""
    x1, x2, features = (batch[0].to(device), batch[1].to(device), batch[2].to(device))
    x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred = model.forward(x1, features)
    z2, mu2, logvar2 = model.encode_x2(x2)
    x2_rec = model.decode_x2(z2)
    if diagnostic_capture is not None:
        diagnostic_capture.update({"mu_pred": mu_pred, "logvar_pred": logvar_pred,
                                   "mu2": mu2, "logvar2": logvar2,
                                   "z2_pred": z2_pred, "z2": z2})
    loss_tuple = model.loss(x1, x1_rec, mu1, logvar1, x2, x2_rec, mu2, logvar2,
                            x2_pred, mu_pred, logvar_pred)
    return x1, x2, x2_pred, loss_tuple


def _transigen_predict(torch: Any, model: Any, dataset: Any, positions: np.ndarray,
                       batch_size: int, device: Any) -> np.ndarray:
    from torch.utils.data import DataLoader, Subset
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in DataLoader(Subset(dataset, positions.tolist()), batch_size=batch_size,
                                shuffle=False, num_workers=0, drop_last=False):
            x1, _, prediction, _ = _transigen_forward_batch(torch, model, batch, device)
            delta_pred = (prediction - x1).detach().cpu().numpy().astype(np.float32)
            if not np.isfinite(delta_pred).all():
                raise FloatingPointError("non-finite TranSiGen delta prediction")
            chunks.append(delta_pred)
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 978), dtype=np.float32)


def _source_hashes() -> dict[str, str]:
    paths = {
        "xpert_config": XPERT_ROOT / "configs/config_l1000.yaml",
        "xpert_train": XPERT_ROOT / "train_xpert.py",
        "xpert_model": XPERT_ROOT / "models/model_XPert.py",
        "xpert_dataset": XPERT_ROOT / "datasets/MyDataset.py",
        "xpert_utils": XPERT_ROOT / "utils.py",
        "xpert_unimol": XPERT_ROOT / "processed_data/all_drugs_unimol_arr.npy",
        "transigen_train": TRANSIGEN_ROOT / "train_TranSiGen.py",
        "transigen_model": TRANSIGEN_ROOT / "model.py",
        "transigen_ecfp4_source": TRANSIGEN_ROOT / "run_original_repurposing.py",
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing original model source/feature assets: " + "; ".join(missing))
    return {name: sha256_file(path) for name, path in paths.items()}


def _device_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5:
            rows.append({"uuid": parts[0], "name": parts[1], "memory_used_mib": int(parts[2]),
                         "memory_total_mib": int(parts[3]), "utilization_percent": int(parts[4])})
    selected = [row for row in rows if row["uuid"] == GPU_UUID]
    if len(selected) != 1:
        raise RuntimeError(f"approved A5000 UUID not found in nvidia-smi: {GPU_UUID}")
    return {"gpus": rows, "selected_gpu": selected[0], "captured_at_utc": utc_now()}


def _assert_exp055_gpu_idle(allowed_pids: set[str] | None = None) -> dict[str, Any]:
    """Require the one authorized A5000 to have no active compute processes."""
    snapshot = _device_snapshot()
    result = subprocess.run(
        ["nvidia-smi", "--id", GPU_UUID,
         "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    processes = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[0]:
            processes.append({"pid": parts[0], "process_name": parts[1], "used_memory_mib": parts[2]})
    selected = snapshot["selected_gpu"]
    allowed = allowed_pids or set()
    other_processes = [entry for entry in processes if entry["pid"] not in allowed]
    if selected["memory_used_mib"] > 1024 or selected["utilization_percent"] > 20 or other_processes:
        raise RuntimeError(f"authorized A5000 is not idle: snapshot={selected}, active_compute_processes={processes}")
    return {"device_snapshot": snapshot, "active_compute_processes": processes}


def _build_transigen_ecfp4(keys: pd.DataFrame, feature_map_path: Path,
                           out_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build source-matched 2048-bit ECFP4, validating and grouping aliases by full key."""
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import AllChem

    table = pd.read_csv(feature_map_path, usecols=["pert_id", "inchi_key", "canonical_smiles",
                                                  "drug_node_idx"], dtype="string",
                         keep_default_na=False)
    table["full_inchikey"] = table["inchi_key"].str.strip().str.upper()
    table["pert_id"] = table["pert_id"].str.strip()
    table["canonical_smiles"] = table["canonical_smiles"].str.strip()
    table["drug_node_idx"] = pd.to_numeric(table["drug_node_idx"], errors="coerce").astype("Int64")
    selected = keys[["full_inchikey", "pert_id", "drug_node_idx"]].drop_duplicates()
    selected["full_inchikey"] = selected["full_inchikey"].astype(str).str.upper()
    selected["drug_node_idx"] = pd.to_numeric(selected["drug_node_idx"], errors="raise").astype("Int64")
    table = table.drop_duplicates(["full_inchikey", "pert_id", "canonical_smiles", "drug_node_idx"])
    joined = selected.merge(table[["full_inchikey", "pert_id", "canonical_smiles", "drug_node_idx"]],
                            on=["full_inchikey", "pert_id", "drug_node_idx"], how="left",
                            validate="many_to_many")
    if joined["canonical_smiles"].isna().any() or joined["canonical_smiles"].eq("").any():
        missing = sorted(joined.loc[joined["canonical_smiles"].isna() | joined["canonical_smiles"].eq(""),
                                    "full_inchikey"].astype(str).unique())
        raise ValueError(f"frozen full InChIKey has no canonical_smiles source mapping: {missing}")
    if joined["full_inchikey"].nunique() != 5:
        raise ValueError(f"expected five frozen full InChIKeys; found {joined.full_inchikey.nunique()}")

    output: dict[str, np.ndarray] = {}
    groups: list[dict[str, Any]] = []
    for full_key, group in joined.groupby("full_inchikey", sort=True):
        node_ids = sorted(pd.to_numeric(group["drug_node_idx"], errors="raise").astype(int).unique().tolist())
        if len(node_ids) != 1:
            raise ValueError(f"full InChIKey {full_key} maps to multiple XPert feature nodes: {node_ids}")
        alias_audit: list[dict[str, Any]] = []
        feature_by_hash: dict[str, np.ndarray] = {}
        for row in group.drop_duplicates(["pert_id", "canonical_smiles"]).sort_values(
                ["pert_id", "canonical_smiles"], kind="mergesort").itertuples(index=False):
            mol = Chem.MolFromSmiles(str(row.canonical_smiles))
            if mol is None:
                raise ValueError(f"RDKit cannot parse canonical_smiles for {full_key}/{row.pert_id}")
            computed_key = Chem.MolToInchiKey(mol)
            if computed_key != str(full_key):
                raise ValueError(f"canonical_smiles identity mismatch for {full_key}/{row.pert_id}: {computed_key}")
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            vector_u8 = np.zeros((2048,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fingerprint, vector_u8)
            feature_hash = hashlib.sha256(vector_u8.tobytes()).hexdigest()
            feature_by_hash[feature_hash] = vector_u8.astype(np.float32)
            alias_audit.append({"pert_id": str(row.pert_id), "canonical_smiles": str(row.canonical_smiles),
                                "computed_inchikey": computed_key, "inchikey_matches_frozen_full_key": True,
                                "ecfp4_2048_sha256": feature_hash, "on_bits": int(vector_u8.sum())})
        if len(feature_by_hash) != 1:
            raise ValueError(f"aliases for full InChIKey {full_key} yield {len(feature_by_hash)} distinct ECFP4 vectors")
        vector = next(iter(feature_by_hash.values()))
        if vector.shape != (2048,) or not np.isfinite(vector).all():
            raise ValueError(f"ECFP4 feature for {full_key} is not a finite 2048-vector")
        output[str(full_key)] = vector
        groups.append({"full_inchikey": str(full_key), "drug_node_idx": node_ids[0],
                       "n_pert_id_aliases": int(group["pert_id"].nunique()),
                       "n_canonical_smiles_aliases": len(alias_audit),
                       "n_unique_ecfp4_vectors": 1,
                       "ecfp4_2048_sha256": hashlib.sha256(vector.astype(np.uint8).tobytes()).hexdigest(),
                       "aliases": alias_audit})
    ordered_keys = sorted(output)
    feature_matrix = np.stack([output[key] for key in ordered_keys], axis=0)
    node_by_key = {item["full_inchikey"]: item["drug_node_idx"] for item in groups}
    np.savez_compressed(out_path, full_inchikey=np.asarray(ordered_keys, dtype=str),
                        drug_node_idx=np.asarray([node_by_key[key] for key in ordered_keys], dtype=np.int64),
                        features=feature_matrix)
    return output, {"source_drug_identity_table": str(feature_map_path),
                    "source_drug_identity_table_sha256": sha256_file(feature_map_path),
                    "generation_reference": str(TRANSIGEN_ROOT / "run_original_repurposing.py"),
                    "generation_reference_sha256": sha256_file(TRANSIGEN_ROOT / "run_original_repurposing.py"),
                    "generation_line_evidence": [165, 170],
                    "method": "RDKit AllChem.GetMorganFingerprintAsBitVect(radius=2, nBits=2048), author canonical_smiles, grouped by full InChIKey after per-alias identity and vector-hash checks",
                    "rdkit_version": rdBase.rdkitVersion, "full_key_groups": groups,
                    "n_full_inchikeys": len(output), "all_aliases_identity_checked": True,
                    "all_aliases_within_key_have_identical_ecfp4": True,
                    "output_npz": str(out_path), "output_npz_sha256": sha256_file(out_path)}


def feature_audit_phase() -> dict[str, Any]:
    """CPU-only identity and feature regression on the frozen response-blind drug keys."""
    key_path = RUN_DIR / "response_blind_conditions.tsv"
    if not key_path.is_file():
        raise FileNotFoundError("frozen response-blind key table is required")
    out_path = RUN_DIR / "transigen_ecfp4_features.npz"
    receipt_path = RUN_DIR / "transigen_ecfp4_feature_audit_receipt.json"
    if out_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite the CPU-side ECFP4 identity audit")
    keys = pd.read_csv(key_path, sep="\t", dtype=str)
    vectors, evidence = _build_transigen_ecfp4(keys, DRUG_MAP, out_path)
    if len(vectors) != 5 or any(value.shape != (2048,) for value in vectors.values()):
        raise AssertionError("CPU ECFP4 regression requires exactly five keyed 2048-vectors")
    receipt = {"experiment_id": "EXP-051", "run_id": RUN_ID,
               "stage": "CPU_ONLY_ECFP4_FULL_KEY_ALIAS_REGRESSION_PASS",
               "created_at_utc": utc_now(), "current_runner_sha256": sha256_file(Path(__file__).resolve()),
               "frozen_condition_table": str(key_path), "frozen_condition_table_sha256": sha256_file(key_path),
               "response_values_read": False, "H5AD_X_or_X_ctl_read": False,
               "PDO_outer_locked_or_external_expression_opened": False,
               "identity_feature_evidence": evidence}
    atomic_json(receipt_path, receipt)
    print(json.dumps({"stage": receipt["stage"], "n_full_inchikeys": evidence["n_full_inchikeys"],
                      "ecfp4_feature_npz_sha256": evidence["output_npz_sha256"],
                      "receipt": str(receipt_path)}, ensure_ascii=False, indent=2), flush=True)
    return receipt


def _transigen_predict_arrays(torch: Any, model: Any, X_ctl: np.ndarray,
                              X: np.ndarray, features: np.ndarray,
                              batch_size: int, device: Any) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X_ctl), batch_size):
            stop = min(start + batch_size, len(X_ctl))
            x1 = torch.as_tensor(X_ctl[start:stop], dtype=torch.float32, device=device)
            x2 = torch.as_tensor(X[start:stop], dtype=torch.float32, device=device)
            feat = torch.as_tensor(features[start:stop], dtype=torch.float32, device=device)
            _, _, _, x2_pred, _, _, _ = model.forward(x1, feat)
            main_delta = (x2_pred - x1).detach().cpu().numpy().astype(np.float32)
            if not np.isfinite(main_delta).all():
                raise FloatingPointError("non-finite TranSiGen main-delta prediction during evaluation")
            chunks.append(main_delta)
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 978), dtype=np.float32)


def _load_audited_ecfp4(artifact_dir: Path | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    artifact_dir = artifact_dir or SOURCE_RUN_DIR
    receipt_path = artifact_dir / "transigen_ecfp4_feature_audit_receipt.json"
    feature_path = artifact_dir / "transigen_ecfp4_features.npz"
    if not receipt_path.is_file() or not feature_path.is_file():
        raise FileNotFoundError("CPU full-key ECFP4 audit must pass before any GPU attempt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    evidence = receipt.get("identity_feature_evidence", {})
    if (receipt.get("run_id") != artifact_dir.name
            or receipt.get("stage") != "CPU_ONLY_ECFP4_FULL_KEY_ALIAS_REGRESSION_PASS"
            or receipt.get("response_values_read") is not False
            or evidence.get("all_aliases_identity_checked") is not True
            or evidence.get("all_aliases_within_key_have_identical_ecfp4") is not True
            or evidence.get("output_npz_sha256") != sha256_file(feature_path)):
        raise RuntimeError("CPU ECFP4 identity receipt/hash gate failed")
    with np.load(feature_path, allow_pickle=False) as data:
        keys = data["full_inchikey"].astype(str)
        matrix = data["features"].astype(np.float32, copy=True)
        nodes = data["drug_node_idx"].astype(np.int64, copy=True)
    if (len(keys) != 5 or len(set(keys.tolist())) != 5 or matrix.shape != (5, 2048)
            or len(nodes) != 5 or not np.isfinite(matrix).all()):
        raise RuntimeError("audited ECFP4 artifact must contain five finite 2048-bit full-key vectors")
    if set(keys.tolist()) != set(evidence_key["full_inchikey"] for evidence_key in evidence["full_key_groups"]):
        raise RuntimeError("ECFP4 NPZ key axis differs from its full-key alias audit")
    return {key: matrix[i] for i, key in enumerate(keys)}, {
        "feature_audit_receipt": str(receipt_path),
        "feature_audit_receipt_sha256": sha256_file(receipt_path),
        "feature_npz": str(feature_path),
        "feature_npz_sha256": sha256_file(feature_path),
        "n_full_inchikeys": len(keys), "feature_shape": list(matrix.shape),
        "source_aliases_verified": True,
    }


def _full_key_features(keys: pd.DataFrame, positions: np.ndarray,
                       feature_by_key: dict[str, np.ndarray]) -> np.ndarray:
    full_keys = keys.iloc[positions]["full_inchikey"].astype(str).to_numpy()
    missing = sorted(set(full_keys) - set(feature_by_key))
    if missing:
        raise KeyError(f"full InChIKey has no CPU-audited TranSiGen ECFP4 vector: {missing}")
    features = np.stack([feature_by_key[key] for key in full_keys]).astype(np.float32, copy=False)
    if features.shape != (len(positions), 2048) or not np.isfinite(features).all():
        raise ValueError("full-key ECFP4 feature rows are not finite 2048-vectors")
    return features


def _transigen_predict_and_losses(torch: Any, model: Any, X_ctl: np.ndarray,
                                  X: np.ndarray, features: np.ndarray,
                                  batch_size: int, device: Any, seed: int, *,
                                  seed_cuda: bool = True,
                                  allow_nonfinite_auxiliary: bool = False,
                                  allow_nonfinite_main: bool = False,
                                  progress_check: Any | None = None,
                                  nonfinite_main_delta_callback: Any | None = None
                                  ) -> tuple[np.ndarray, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    if seed_cuda:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        torch.random.default_generator.manual_seed(seed)
    model.eval()
    chunks: list[np.ndarray] = []
    loss_sums = np.zeros(7, dtype=np.float64)
    valid_counts = np.zeros(7, dtype=np.int64)
    nonfinite_counts = np.zeros(7, dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(X_ctl), batch_size):
            if progress_check is not None:
                progress_check()
            stop = min(start + batch_size, len(X_ctl))
            x1 = torch.as_tensor(X_ctl[start:stop], dtype=torch.float32, device=device)
            x2 = torch.as_tensor(X[start:stop], dtype=torch.float32, device=device)
            feat = torch.as_tensor(features[start:stop], dtype=torch.float32, device=device)
            x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, _ = model.forward(x1, feat)
            z2, mu2, logvar2 = model.encode_x2(x2)
            x2_rec = model.decode_x2(z2)
            loss_terms = model.loss(x1, x1_rec, mu1, logvar1, x2, x2_rec,
                                    mu2, logvar2, x2_pred, mu_pred, logvar_pred)
            main_delta_tensor = (x2_pred - x1).detach()
            finite_mask = torch.isfinite(main_delta_tensor)
            finite_n = int(finite_mask.sum().item())
            if finite_n != main_delta_tensor.numel():
                finite_values = main_delta_tensor[finite_mask]
                bad = torch.nonzero(~finite_mask, as_tuple=False)[:32].detach().cpu().tolist()
                diagnostic = {"batch_start": int(start), "batch_stop": int(stop),
                    "batch_rows": int(stop - start), "shape": list(main_delta_tensor.shape),
                    "numel": int(main_delta_tensor.numel()), "finite_n": finite_n,
                    "nonfinite_n": int(main_delta_tensor.numel() - finite_n),
                    "finite_max_abs": (float(finite_values.abs().max().cpu().item())
                                       if finite_values.numel() else None),
                    "nonfinite_row_gene_positions_first_32": bad,
                    "timestamp_utc": utc_now(), "evaluation_split": "TRAIN_ONLY",
                    "inner_dev_numeric_rows_loaded": False,
                    "pdo_outer_locked_numeric_values_opened": False}
                if nonfinite_main_delta_callback is not None:
                    nonfinite_main_delta_callback(diagnostic)
                if not allow_nonfinite_main:
                    raise FloatingPointError(
                        f"non-finite TranSiGen main-delta prediction in TRAIN evaluation batch "
                        f"[{start}:{stop}] ({diagnostic['nonfinite_n']}/{diagnostic['numel']} nonfinite)")
            main_delta = main_delta_tensor.cpu().numpy().astype(np.float32)
            chunks.append(main_delta)
            for index, term in enumerate(loss_terms):
                if bool(torch.isfinite(term).all()):
                    loss_sums[index] += float(term.detach().cpu())
                    valid_counts[index] += len(x1)
                else:
                    nonfinite_counts[index] += len(x1)
                    if not allow_nonfinite_auxiliary:
                        raise FloatingPointError("non-finite TranSiGen evaluation loss")
            if progress_check is not None:
                progress_check()
    prediction = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 978), dtype=np.float32)
    if len(prediction) != len(X_ctl):
        raise ValueError("TranSiGen prediction denominator does not equal its keyed sample count")
    names = ("raw_total_loss", "mse_x1", "mse_x2", "mse_pert", "kld_x1", "kld_x2", "kld_pert")
    report: dict[str, Any] = {}
    for index, name in enumerate(names):
        if not allow_nonfinite_auxiliary:
            report[name] = safe_number(float(loss_sums[index] / len(X_ctl)))
            continue
        if nonfinite_counts[index]:
            report[name] = {"value": None, "reason": "nonfinite_auxiliary_evaluation_loss",
                            "effective_n": int(valid_counts[index]),
                            "nonfinite_n": int(nonfinite_counts[index])}
        elif valid_counts[index] == len(X_ctl):
            report[name] = {"value": safe_number(float(loss_sums[index] / len(X_ctl))),
                            "reason": None, "effective_n": int(valid_counts[index]), "nonfinite_n": 0}
        else:
            report[name] = {"value": None, "reason": "no_finite_auxiliary_evaluation_loss",
                            "effective_n": int(valid_counts[index]), "nonfinite_n": int(nonfinite_counts[index])}
    return prediction, report


def _fit_gate(train_metrics: dict[str, Any], baseline_metrics: dict[str, Any]) -> dict[str, Any]:
    candidates = {}
    for name in ("zero_change", "per_drug_mean_delta"):
        value = baseline_metrics["metrics"][name]["train"].get("object_macro_nmse")
        if value is not None and math.isfinite(float(value)):
            candidates[name] = float(value)
    if not candidates:
        raise ValueError("XPert/TranSiGen fit gate has no finite zero/per-drug training baseline")
    strongest_name = min(candidates, key=candidates.get)
    strongest_nmse = candidates[strongest_name]
    model_nmse = train_metrics.get("object_macro_nmse")
    amplitude = train_metrics.get("object_macro_amplitude_ratio")
    absolute_difference = (strongest_nmse - float(model_nmse)) if model_nmse is not None else None
    relative_improvement = (absolute_difference / strongest_nmse
                            if strongest_nmse > 0 and absolute_difference is not None else None)
    amplitude_pass = amplitude is not None and 0.5 <= float(amplitude) <= 2.0
    if strongest_nmse == 0:
        status = "NOT_APPLICABLE_ZERO_BASELINE"
    elif relative_improvement is not None and relative_improvement >= 0.10 and amplitude_pass:
        status = "PASS"
    else:
        status = "FAIL_CURRENT_IMPLEMENTATION_FIT_GATE"
    return {"status": status, "strongest_train_baseline": strongest_name,
            "strongest_train_baseline_nmse": strongest_nmse,
            "model_train_nmse": model_nmse,
            "absolute_nmse_reduction": absolute_difference,
            "relative_nmse_reduction": relative_improvement,
            "relative_improvement_threshold": 0.10,
            "model_train_amplitude_ratio": amplitude,
            "amplitude_ratio_range": [0.5, 2.0],
            "amplitude_gate_pass": bool(amplitude_pass),
            "interpretation": ("relative improvement is undefined at a zero denominator; report absolute difference only"
                               if strongest_nmse == 0 else "training-side fit diagnostic only")}


def _wrap_trainpath_recovery_baselines_for_fit_gate(
        train_baselines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Adapt this TRAIN-only runner's flat baseline metrics to the shared fit-gate contract."""
    names = ("zero_change", "per_drug_mean_delta")
    if any(name not in train_baselines or not isinstance(train_baselines[name], dict) for name in names):
        raise ValueError("TRAIN-path recovery requires zero-change and per-drug baseline metrics")
    return {"metrics": {name: {"train": train_baselines[name]} for name in names}}


def _metric_from_baseline_readback(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize a previously root-verified metric row without refitting or rescoring it."""
    return {"object_macro_nmse": value.get("nmse", value.get("object_macro_nmse")),
            "object_macro_mae": value.get("mae", value.get("object_macro_mae")),
            "object_macro_gene_correlation": value.get("correlation", value.get("object_macro_gene_correlation")),
            "object_macro_direction_agreement": value.get("direction", value.get("object_macro_direction_agreement")),
            "object_macro_amplitude_ratio": value.get("amplitude_ratio", value.get("object_macro_amplitude_ratio"))}


def _reuse_exp051_baseline_readbacks(attempt: Path) -> dict[str, Any]:
    """Bind EXP-051 baseline predictions/metrics by their independent readback receipts."""
    source_root = ROOT / "artifacts/experiments/EXP-051"
    baseline_run = source_root / "20261006_exp051_representative_compare_v1"
    baseline_readback_path = baseline_run / "baseline_independent_readback_v2.json"
    baseline_prediction = baseline_run / "fit_attempt_002/baseline_predictions.npz"
    baseline_metric = baseline_run / "fit_attempt_002/baseline_metrics.json"
    key_table = SOURCE_RUN_DIR / "response_blind_conditions.tsv"
    low_rank_run = source_root / "20261007_exp051_low_rank_baselines_v1"
    low_rank_readback_path = low_rank_run / "root_independent_readback.json"
    low_rank_result_path = low_rank_run / "result.json"
    low_rank_prediction_path = low_rank_run / "low_rank_predictions.npz"
    required = (baseline_readback_path, baseline_prediction, baseline_metric, key_table,
                low_rank_readback_path, low_rank_result_path, low_rank_prediction_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required EXP-051 root-readback baseline assets are missing: {missing}")

    baseline = json.loads(baseline_readback_path.read_text(encoding="utf-8"))
    low_readback = json.loads(low_rank_readback_path.read_text(encoding="utf-8"))
    low_result = json.loads(low_rank_result_path.read_text(encoding="utf-8"))
    checks = baseline.get("readback", {})
    if (baseline.get("source_attempt") != "fit_attempt_002"
            or checks.get("train_conditions") != 419 or checks.get("train_objects") != 110
            or checks.get("inner_dev_conditions") != 93 or checks.get("inner_dev_objects") != 24
            or checks.get("train_inner_dev_object_overlap") != 0
            or checks.get("full_inchikeys") != 5 or checks.get("genes") != 978
            or checks.get("all_targets_and_predictions_finite") is not True
            or checks.get("generator_metrics_recomputed_match_abs_tol_1e_7") is not True):
        raise RuntimeError("EXP-051 baseline root readback does not certify the frozen split and finite predictions")
    if (sha256_file(baseline_prediction) != checks.get("baseline_predictions_sha256")
            or sha256_file(baseline_metric) != checks.get("baseline_metrics_sha256")
            or sha256_file(key_table) != checks.get("response_blind_key_table_sha256")):
        raise RuntimeError("EXP-051 mean-baseline prediction, metrics, or keyed table changed after root readback")
    if (low_readback.get("status") != "ROOT_INDEPENDENT_KEYED_READBACK_PASS"
            or low_readback.get("denominator", {}).get("train_conditions") != 419
            or low_readback.get("denominator", {}).get("inner_dev_conditions") != 93
            or low_readback.get("checks", {}).get("all_predictions_finite") not in (True, "PASS")
            or sha256_file(low_rank_result_path) != low_readback.get("run_result_sha256")
            or sha256_file(low_rank_prediction_path) != low_readback.get("prediction_sha256")
            or sha256_file(baseline_prediction) != low_readback.get("baseline_prediction_sha256")):
        raise RuntimeError("EXP-051 low-rank root readback or bound prediction hashes failed")

    metrics: dict[str, Any] = {"zero_change": {}, "global_mean_delta": {}, "per_drug_mean_delta": {},
                               "structural_only": {}, "state_plus_structure": {}}
    for method, split_name in (("train_object_macro_metrics", "train"),
                               ("inner_dev_object_macro_metrics", "inner_dev")):
        for name in ("zero_change", "global_mean_delta", "per_drug_mean_delta"):
            metrics[name][split_name] = _metric_from_baseline_readback(baseline[method][name])
        source_split = "train" if split_name == "train" else "inner_dev"
        for name in ("structural_only", "state_plus_structure"):
            metrics[name][split_name] = _metric_from_baseline_readback(
                low_result["methods"][name][source_split])
    result = {"experiment_id": "EXP-055", "source": "EXP-051 root-independent baseline readbacks",
              "data_access": "No H5AD matrix, PDO, outer or locked values read; saved authorized baseline predictions reused.",
              "metrics": metrics,
              "readback_refs": {
                  "mean_baseline_readback": {"path": str(baseline_readback_path),
                                              "sha256": sha256_file(baseline_readback_path)},
                  "mean_baseline_predictions": {"path": str(baseline_prediction),
                                                "sha256": sha256_file(baseline_prediction)},
                  "mean_baseline_metrics": {"path": str(baseline_metric),
                                            "sha256": sha256_file(baseline_metric)},
                  "low_rank_readback": {"path": str(low_rank_readback_path),
                                        "sha256": sha256_file(low_rank_readback_path)},
                  "low_rank_result": {"path": str(low_rank_result_path),
                                      "sha256": sha256_file(low_rank_result_path)},
                  "low_rank_predictions": {"path": str(low_rank_prediction_path),
                                           "sha256": sha256_file(low_rank_prediction_path)}}}
    atomic_json(attempt / "baseline_metrics.json", result)
    return result


def _train_full_models(torch: Any, attempt: Path, keys: pd.DataFrame,
                       values: dict[str, np.ndarray], train_mask: np.ndarray,
                       dev_mask: np.ndarray, train_keys: pd.DataFrame,
                       dev_keys: pd.DataFrame, gene_variance: np.ndarray,
                       baseline_metrics: dict[str, Any], unimol: np.ndarray,
                       xpert_module: Any, xpert_config: dict[str, Any],
                       feature_by_key: dict[str, np.ndarray], device: Any,
                       update_seconds: dict[str, float], device_budget_hours: float,
                       elapsed_gpu_origin: float) -> dict[str, Any]:
    """Complete fixed 100-epoch original-model fits after the cost gate passes."""
    from torch.utils.data import DataLoader, TensorDataset

    train_positions = np.flatnonzero(train_mask)
    dev_positions = np.flatnonzero(dev_mask)
    train_y, dev_y = values["delta"][train_mask], values["delta"][dev_mask]
    curves: dict[str, Path] = {
        "XPert": attempt / "XPert_training_curve.jsonl",
        "TranSiGen": attempt / "TranSiGen_training_curve.jsonl",
    }
    for path in curves.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite a training curve: {path}")

    # XPert uses the original source epoch function and config. The adapter
    # initialization above only repairs its unbound optional return variable.
    original_train, _, _, xpert_source_receipt = _source_functions(
        XPERT_ROOT / "train_xpert.py", XPERT_ROOT / "utils.py")
    epochs = 100
    batch_size_xpert = int(xpert_config["train"]["batch_size"])
    init_epoch = int(xpert_config["train"]["init_epoch"])
    if batch_size_xpert != 128 or init_epoch != 70:
        raise ValueError("XPert config must preserve frozen batch128 and 70-epoch schedule transition")
    xpert_model, train_view = _new_xpert(torch, xpert_module, xpert_config, device, SEED)
    xpert_opt_raw = torch.optim.Adam(
        xpert_model.parameters(), lr=float(xpert_config["train"]["train_lr"]),
        weight_decay=float(xpert_config["train"]["weight_decay"]))
    xpert_opt = ProbeOptimizer(xpert_opt_raw, xpert_model, torch)
    xpert_scheduler = torch.optim.lr_scheduler.LambdaLR(
        xpert_opt_raw, lr_lambda=lambda epoch: 1.0 if epoch < 40 else 0.5)
    xpert_tensors = _xpert_tensors(torch, values, keys, train_positions, unimol,
                                   xpert_config, training_targets=True)
    xpert_loader = DataLoader(TensorDataset(*xpert_tensors), batch_size=batch_size_xpert,
                              shuffle=True, drop_last=False, num_workers=0,
                              generator=torch.Generator().manual_seed(SEED))
    fit_started = time.perf_counter()
    xpert_best_nmse = math.inf
    xpert_best_epoch = -1
    xpert_best_state: dict[str, Any] | None = None
    xpert_best_train = xpert_best_dev = None
    xpert_last_loss = None
    xpert_epoch_seconds: list[float] = []
    for epoch in range(epochs):
        started = time.perf_counter()
        gradients_before = len(xpert_opt.gradient_norms)
        learning_rate = float(xpert_opt_raw.param_groups[0]["lr"])
        class Args:
            pretrained_model = False
        losses = original_train(train_view, xpert_opt, xpert_loader, Args(), xpert_config,
                                scaler=None, epoch=epoch)
        torch.cuda.synchronize()
        train_pred = _xpert_predict(torch, xpert_model, values, keys,
                                    train_positions, unimol, xpert_config, batch_size=64)
        dev_pred = _xpert_predict(torch, xpert_model, values, keys,
                                  dev_positions, unimol, xpert_config, batch_size=64)
        train_metric = keyed_metrics(train_keys, train_y, train_pred, gene_variance)
        dev_metric = keyed_metrics(dev_keys, dev_y, dev_pred, gene_variance)
        grad_slice = xpert_opt.gradient_norms[gradients_before:]
        if not grad_slice:
            raise RuntimeError(f"XPert epoch {epoch} completed no optimizer updates")
        xpert_last_loss = [safe_number(float(value)) for value in losses]
        duration = time.perf_counter() - started
        xpert_epoch_seconds.append(duration)
        row = {"epoch": epoch + 1, "epoch_index_zero_based": epoch,
               "schedule_stage": "per_sample_mean_root_mse" if epoch < init_epoch else "raw_weighted_sum",
               "learning_rate": learning_rate,
               "official_train_loss_terms": xpert_last_loss,
               "gradient_norm_mean": safe_number(float(np.mean(grad_slice))),
               "gradient_norm_max": safe_number(float(np.max(grad_slice))),
               "gradient_norm_n_updates": len(grad_slice),
               "train_metrics": train_metric, "inner_dev_metrics": dev_metric,
               "epoch_seconds": duration}
        append_jsonl(curves["XPert"], row)
        dev_nmse = dev_metric["object_macro_nmse"]
        if dev_nmse is not None and float(dev_nmse) < xpert_best_nmse:
            xpert_best_nmse = float(dev_nmse)
            xpert_best_epoch = epoch + 1
            xpert_best_state = {key: value.detach().cpu().clone()
                                for key, value in xpert_model.state_dict().items()}
            xpert_best_train, xpert_best_dev = train_pred.copy(), dev_pred.copy()
        xpert_scheduler.step()
        elapsed_all = time.perf_counter() - elapsed_gpu_origin
        if elapsed_all / 3600 >= device_budget_hours:
            raise TimeoutError("ledger-derived EXP-051 GPU attempt envelope reached during XPert training")
        atomic_json(attempt / "training_progress.json", {
            "model": "XPert", "completed_epochs": epoch + 1, "target_epochs": epochs,
            "best_inner_dev_epoch": xpert_best_epoch,
            "elapsed_gpu_hours_upper_bound": elapsed_all / 3600,
            "updated_at_utc": utc_now()})
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(json.dumps({"stage": "XPert_FULL_FIT_PROGRESS", "epoch": epoch + 1,
                              "train_nmse": train_metric["object_macro_nmse"],
                              "inner_dev_nmse": dev_metric["object_macro_nmse"],
                              "elapsed_gpu_hours": elapsed_all / 3600}, ensure_ascii=False), flush=True)

    if xpert_best_state is None or xpert_best_train is None or xpert_best_dev is None:
        raise RuntimeError("XPert full fit finished without a finite inner-dev checkpoint")
    xpert_final_state = {key: value.detach().cpu().clone()
                         for key, value in xpert_model.state_dict().items()}
    torch.save({"state_dict": xpert_best_state, "model": "upstream XPertNet",
                "epoch": xpert_best_epoch, "seed": SEED,
                "official_train_function_sha256": xpert_source_receipt["train_function_source_sha256"]},
               attempt / "XPert_best_inner_dev.pt")
    torch.save({"state_dict": xpert_final_state, "model": "upstream XPertNet",
                "epoch": epochs, "seed": SEED,
                "official_train_function_sha256": xpert_source_receipt["train_function_source_sha256"]},
               attempt / "XPert_final_epoch_100.pt")
    del xpert_model, train_view, xpert_opt, xpert_opt_raw, xpert_scheduler, xpert_loader, xpert_tensors
    torch.cuda.empty_cache()

    # TranSiGen update loop follows the source model.train_model contract:
    # Adam, raw sum objective, no clipping, batch64 shuffle, and scratch init.
    transigen_model, transigen_module = _load_transigen(torch, device, SEED, features_dim=2048)
    transigen_optimizer = torch.optim.Adam(transigen_model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_features = _full_key_features(keys, train_positions, feature_by_key)
    dev_features = _full_key_features(keys, dev_positions, feature_by_key)
    x1_train = torch.from_numpy(values["X_ctl"][train_mask].astype(np.float32, copy=False))
    x2_train = torch.from_numpy(values["X"][train_mask].astype(np.float32, copy=False))
    features_train = torch.from_numpy(train_features)
    transigen_train_loader = DataLoader(TensorDataset(x1_train, x2_train, features_train),
                                        batch_size=64, shuffle=True, drop_last=False,
                                        num_workers=0, generator=torch.Generator().manual_seed(SEED))
    loss_names = ("raw_total_loss", "mse_x1", "mse_x2", "mse_pert", "kld_x1", "kld_x2", "kld_pert")
    transigen_best_nmse = math.inf
    transigen_best_epoch = -1
    transigen_best_state: dict[str, Any] | None = None
    transigen_best_train = transigen_best_dev = None
    transigen_best_losses = None
    transigen_epoch_seconds: list[float] = []
    for epoch in range(epochs):
        started = time.perf_counter()
        transigen_model.train()
        epoch_loss_sums = np.zeros(7, dtype=np.float64)
        update_gradients: list[float] = []
        seen_training_samples = 0
        for batch in transigen_train_loader:
            batch = tuple(tensor.to(device) for tensor in batch)
            seen_training_samples += int(batch[0].shape[0])
            transigen_optimizer.zero_grad(set_to_none=True)
            _, _, _, loss_terms = _transigen_forward_batch(torch, transigen_model, batch, device)
            if not all(torch.isfinite(loss).all() for loss in loss_terms):
                raise FloatingPointError(f"TranSiGen non-finite training loss at epoch {epoch + 1}")
            loss_terms[0].backward()
            gradient_norm = _grad_global_norm(torch, transigen_model)
            if not math.isfinite(gradient_norm):
                raise FloatingPointError(f"TranSiGen non-finite gradient norm at epoch {epoch + 1}")
            update_gradients.append(gradient_norm)
            epoch_loss_sums += np.asarray([float(loss.detach().cpu()) for loss in loss_terms])
            transigen_optimizer.step()
        if seen_training_samples != len(train_positions):
            raise RuntimeError("TranSiGen full-fit loader denominator changed from 419 training keys")
        torch.cuda.synchronize()
        transigen_train_pred, transigen_train_losses = _transigen_predict_and_losses(
            torch, transigen_model, values["X_ctl"][train_mask], values["X"][train_mask],
            train_features, 64, device, SEED + epoch * 2 + 1)
        transigen_dev_pred, transigen_dev_losses = _transigen_predict_and_losses(
            torch, transigen_model, values["X_ctl"][dev_mask], values["X"][dev_mask],
            dev_features, 64, device, SEED + epoch * 2 + 2)
        train_metric = keyed_metrics(train_keys, train_y, transigen_train_pred, gene_variance)
        dev_metric = keyed_metrics(dev_keys, dev_y, transigen_dev_pred, gene_variance)
        # This equals the upstream epoch update objective, averaged per sample.
        objective_losses = {name: safe_number(float(total / len(train_positions)))
                            for name, total in zip(loss_names, epoch_loss_sums)}
        duration = time.perf_counter() - started
        transigen_epoch_seconds.append(duration)
        row = {"epoch": epoch + 1, "epoch_index_zero_based": epoch,
               "batch_size": 64, "learning_rate": 1e-3, "weight_decay": 1e-5,
               "shuffle": True, "gradient_clip": None,
               "raw_sum_objective_per_sample": objective_losses,
               "post_update_train_eval_loss_per_sample": transigen_train_losses,
               "inner_dev_loss_per_sample": transigen_dev_losses,
               "gradient_norm_mean": safe_number(float(np.mean(update_gradients))),
               "gradient_norm_max": safe_number(float(np.max(update_gradients))),
               "gradient_norm_n_updates": len(update_gradients),
               "train_metrics": train_metric, "inner_dev_metrics": dev_metric,
               "epoch_seconds": duration}
        append_jsonl(curves["TranSiGen"], row)
        dev_nmse = dev_metric["object_macro_nmse"]
        if dev_nmse is not None and float(dev_nmse) < transigen_best_nmse:
            transigen_best_nmse = float(dev_nmse)
            transigen_best_epoch = epoch + 1
            transigen_best_state = {key: value.detach().cpu().clone()
                                    for key, value in transigen_model.state_dict().items()}
            transigen_best_train, transigen_best_dev = transigen_train_pred.copy(), transigen_dev_pred.copy()
            transigen_best_losses = {"train": transigen_train_losses, "inner_dev": transigen_dev_losses}
        elapsed_all = time.perf_counter() - elapsed_gpu_origin
        if elapsed_all / 3600 >= device_budget_hours:
            raise TimeoutError("ledger-derived EXP-051 GPU attempt envelope reached during TranSiGen training")
        atomic_json(attempt / "training_progress.json", {
            "model": "TranSiGen", "completed_epochs": epoch + 1, "target_epochs": epochs,
            "best_inner_dev_epoch": transigen_best_epoch,
            "elapsed_gpu_hours_upper_bound": elapsed_all / 3600,
            "updated_at_utc": utc_now()})
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(json.dumps({"stage": "TranSiGen_FULL_FIT_PROGRESS", "epoch": epoch + 1,
                              "train_nmse": train_metric["object_macro_nmse"],
                              "inner_dev_nmse": dev_metric["object_macro_nmse"],
                              "elapsed_gpu_hours": elapsed_all / 3600}, ensure_ascii=False), flush=True)

    if transigen_best_state is None or transigen_best_train is None or transigen_best_dev is None:
        raise RuntimeError("TranSiGen full fit finished without a finite inner-dev checkpoint")
    transigen_final_state = {key: value.detach().cpu().clone()
                             for key, value in transigen_model.state_dict().items()}
    torch.save({"state_dict": transigen_best_state, "model": "upstream TranSiGen",
                "epoch": transigen_best_epoch, "seed": SEED,
                "training_contract": "source model.train_model raw summed loss, Adam lr=1e-3 wd=1e-5, no clipping"},
               attempt / "TranSiGen_best_inner_dev.pt")
    torch.save({"state_dict": transigen_final_state, "model": "upstream TranSiGen",
                "epoch": epochs, "seed": SEED,
                "training_contract": "source model.train_model raw summed loss, Adam lr=1e-3 wd=1e-5, no clipping"},
               attempt / "TranSiGen_final_epoch_100.pt")
    del transigen_module, transigen_model, transigen_optimizer, transigen_train_loader
    torch.cuda.empty_cache()

    np.savez_compressed(
        attempt / "full_keyed_predictions.npz",
        train_condition_key=train_keys.full_condition_key.astype(str).to_numpy(),
        inner_dev_condition_key=dev_keys.full_condition_key.astype(str).to_numpy(),
        train_source_object_id=train_keys.source_object_id.astype(str).to_numpy(),
        inner_dev_source_object_id=dev_keys.source_object_id.astype(str).to_numpy(),
        train_full_inchikey=train_keys.full_inchikey.astype(str).to_numpy(),
        inner_dev_full_inchikey=dev_keys.full_inchikey.astype(str).to_numpy(),
        genes=values["genes"].astype(str),
        train_target_delta=train_y, inner_dev_target_delta=dev_y,
        XPert_best_inner_dev_train_delta=xpert_best_train,
        XPert_best_inner_dev_dev_delta=xpert_best_dev,
        TranSiGen_best_inner_dev_train_delta=transigen_best_train,
        TranSiGen_best_inner_dev_dev_delta=transigen_best_dev)
    predictions_path = attempt / "full_keyed_predictions.npz"
    best_metrics = {
        "XPert": {"selected_epoch_by_inner_dev_nmse": xpert_best_epoch,
                  "train": keyed_metrics(train_keys, train_y, xpert_best_train, gene_variance),
                  "inner_dev": keyed_metrics(dev_keys, dev_y, xpert_best_dev, gene_variance),
                  "fit_gate": _fit_gate(keyed_metrics(train_keys, train_y, xpert_best_train, gene_variance), baseline_metrics),
                  "schedule": {"epochs_completed": epochs, "init_epoch_transition": init_epoch,
                               "batch_size": batch_size_xpert, "lr": float(xpert_config["train"]["train_lr"]),
                               "weight_decay": float(xpert_config["train"]["weight_decay"]),
                               "lr_scheduler": "1.0 through epoch 40, 0.5 thereafter"},
                  "checkpoint_best_sha256": sha256_file(attempt / "XPert_best_inner_dev.pt"),
                  "checkpoint_final_sha256": sha256_file(attempt / "XPert_final_epoch_100.pt"),
                  "mean_epoch_seconds": float(np.mean(xpert_epoch_seconds)),
                  "last_official_loss_terms": xpert_last_loss},
        "TranSiGen": {"selected_epoch_by_inner_dev_nmse": transigen_best_epoch,
                       "train": keyed_metrics(train_keys, train_y, transigen_best_train, gene_variance),
                       "inner_dev": keyed_metrics(dev_keys, dev_y, transigen_best_dev, gene_variance),
                       "fit_gate": _fit_gate(keyed_metrics(train_keys, train_y, transigen_best_train, gene_variance), baseline_metrics),
                       "selected_checkpoint_loss_per_sample": transigen_best_losses,
                       "schedule": {"epochs_completed": epochs, "batch_size": 64,
                                    "lr": 1e-3, "weight_decay": 1e-5,
                                    "shuffle": True, "gradient_clip": None,
                                    "initialization": "scratch; pretrain_shRNA disabled"},
                       "checkpoint_best_sha256": sha256_file(attempt / "TranSiGen_best_inner_dev.pt"),
                       "checkpoint_final_sha256": sha256_file(attempt / "TranSiGen_final_epoch_100.pt"),
                       "mean_epoch_seconds": float(np.mean(transigen_epoch_seconds))},
    }
    return {"status": "COMPLETED_100_EPOCHS_BOTH_MODELS",
            "selected_by": "minimum object-macro inner_dev NMSE across fixed 100 epochs; all rounds retained",
            "best_metrics": best_metrics, "predictions_path": predictions_path.name,
            "predictions_sha256": sha256_file(predictions_path),
            "gene_axis_sha256": sha256_text("\n".join(values["genes"].astype(str)) + "\n"),
            "curves": {name: {"path": path.name, "sha256": sha256_file(path)}
                       for name, path in curves.items()},
            "checkpoints": {
                "XPert_best": {"path": "XPert_best_inner_dev.pt", "sha256": sha256_file(attempt / "XPert_best_inner_dev.pt")},
                "XPert_final": {"path": "XPert_final_epoch_100.pt", "sha256": sha256_file(attempt / "XPert_final_epoch_100.pt")},
                "TranSiGen_best": {"path": "TranSiGen_best_inner_dev.pt", "sha256": sha256_file(attempt / "TranSiGen_best_inner_dev.pt")},
                "TranSiGen_final": {"path": "TranSiGen_final_epoch_100.pt", "sha256": sha256_file(attempt / "TranSiGen_final_epoch_100.pt")}},
            "source_contract": {"XPert_train_function": xpert_source_receipt,
                                "TranSiGen_update": "exact source model.train_model training math (source loss sum, Adam, shuffle, batch64, no clip); adapter retains keyed object-macro metrics and checkpoints"}}


def _smoke_exp055_transigen_first(torch: Any, attempt: Path, keys: pd.DataFrame,
                                  values: dict[str, np.ndarray], smoke_positions: np.ndarray,
                                  dev_positions: np.ndarray, smoke_keys: pd.DataFrame,
                                  dev_keys: pd.DataFrame, gene_variance: np.ndarray,
                                  feature_by_key: dict[str, np.ndarray], device: Any) -> dict[str, Any]:
    """Measure one authorized TranSiGen update before the XPert cost smoke."""
    tick = time.perf_counter()
    model, model_module = _load_transigen(torch, device, SEED, features_dim=2048)
    smoke_features = _full_key_features(smoke_keys, np.arange(len(smoke_keys)), feature_by_key)
    dev_features = _full_key_features(dev_keys, np.arange(len(dev_keys)), feature_by_key)
    x1 = torch.as_tensor(values["X_ctl"][smoke_positions], dtype=torch.float32, device=device)
    x2 = torch.as_tensor(values["X"][smoke_positions], dtype=torch.float32, device=device)
    features = torch.as_tensor(smoke_features, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    initialization_seconds = time.perf_counter() - tick
    model.train()
    optimizer.zero_grad(set_to_none=True)
    update_start = time.perf_counter()
    _, _, _, losses = _transigen_forward_batch(torch, model, (x1, x2, features), device)
    if not all(torch.isfinite(loss).all() for loss in losses):
        raise FloatingPointError("TranSiGen smoke raw-sum loss is non-finite before update")
    losses[0].backward()
    gradient_norm = _grad_global_norm(torch, model)
    if not math.isfinite(gradient_norm):
        raise FloatingPointError("TranSiGen smoke gradient norm is non-finite")
    optimizer.step()
    torch.cuda.synchronize()
    update_seconds = time.perf_counter() - update_start
    eval_start = time.perf_counter()
    train_prediction = _transigen_predict_arrays(
        torch, model, values["X_ctl"][smoke_positions], values["X"][smoke_positions],
        smoke_features, 64, device)
    dev_prediction = _transigen_predict_arrays(
        torch, model, values["X_ctl"][dev_positions], values["X"][dev_positions],
        dev_features, 64, device)
    torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - eval_start
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    checkpoint_path = attempt / "TranSiGen_smoke_one_update_state.pt"
    checkpoint_payload = {"state_dict": state, "model": "upstream TranSiGen", "updates": 1, "seed": SEED}
    if EXPERIMENT_ID == "EXP-055":
        checkpoint_payload.update({"objective": "raw_sum", "gradient_clipping": False})
    torch.save(checkpoint_payload, checkpoint_path)
    loss_terms = {name: safe_number(float(value.detach().cpu())) for name, value in zip(
        ("raw_total_loss", "mse_x1", "mse_x2", "mse_pert", "kld_x1", "kld_x2", "kld_pert"), losses)}
    result = {"train_prediction": train_prediction, "dev_prediction": dev_prediction,
              "initialization_seconds": initialization_seconds,
              "update_seconds": update_seconds, "eval_seconds": eval_seconds,
              "gradient_norm": gradient_norm, "loss_terms": loss_terms,
              "checkpoint_path": checkpoint_path.name,
              "metrics": {"train64_after": keyed_metrics(
                              smoke_keys, values["delta"][smoke_positions], train_prediction, gene_variance),
                          "inner_dev93_after": keyed_metrics(
                              dev_keys, values["delta"][dev_positions], dev_prediction, gene_variance),
                          "n_updates": 1, "n_training_keys": len(smoke_positions),
                          "prediction_max_abs": float(np.max(np.abs(dev_prediction))),
                          "loss_terms_order": ["total", "MSE x1", "MSE x2", "MSE perturbation",
                                               "KL x1", "KL x2", "KL perturbation"]}}
    del optimizer, model, model_module, x1, x2, features
    torch.cuda.empty_cache()
    return result


def _verify_exp055_xpert_cpu_schedule(attempt: Path) -> dict[str, Any]:
    """Verify the local XPert author config and schedule before reserving a GPU."""
    config_path = XPERT_ROOT / "configs/config_l1000.yaml"
    train_path = XPERT_ROOT / "train_xpert.py"
    utils_path = XPERT_ROOT / "utils.py"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train = config["train"]
    expected = {"batch_size": 128, "train_lr": 0.004, "weight_decay": 1e-5,
                "num_epochs": 2500, "init_epoch": 70, "patience": 50,
                "loss_weight": [0.2, 0.003, 0.2, 1]}
    observed = {key: train[key] for key in expected}
    if observed != expected:
        raise ValueError(f"local XPert config_l1000.yaml differs from EXP-055 official schedule: {observed}")
    source_text = train_path.read_text(encoding="utf-8")
    utils_text = utils_path.read_text(encoding="utf-8")
    schedule_checks = {
        "warmup_condition": "if epoch < init_epoch:",
        "warmup_backward": "batch_weighted_loss.backward()",
        "raw_stage_backward": "weighted_loss.backward()",
        "official_scheduler": "lr_lambda=lambda epoch: 1.0 if epoch < 40 else 0.5",
        "official_patience": "patience=config['train']['patience']",
        "official_selection_quantity": "stopper.step(val_loss4, model, epoch, optimizer)",
        "pearson_loss_sum": "def pcc_loss_sum(pred, target):",
        "checkpoint_restore": "stopper.load_checkpoint(model, optimizer)",
    }
    combined = source_text + "\n" + utils_text
    missing = [name for name, marker in schedule_checks.items() if marker not in combined]
    if missing:
        raise RuntimeError(f"upstream XPert source schedule markers changed: {missing}")
    receipt = {"status": "PASS_OFFICIAL_LOCAL_SCHEDULE", "config_path": str(config_path),
               "config_sha256": sha256_file(config_path), "train_source_path": str(train_path),
               "train_source_sha256": sha256_file(train_path), "utils_source_path": str(utils_path),
               "utils_source_sha256": sha256_file(utils_path), "schedule": observed,
               "source_checks": schedule_checks,
               "interpretation": {"epochs_0_to_69": "per-sample root-MSE warmup loss",
                                  "epoch_70_onward": "raw weighted sum loss",
                                  "learning_rate": "LambdaLR index <40 uses 1.0 and index >=40 uses 0.5; the 0.5 factor begins on training-loop epoch index 40 after 40 completed epochs",
                                  "selection_and_patience": "minimum val_loss4 (mean per-sample delta Pearson loss), patience 50, author checkpoint restore"}}
    atomic_json(attempt / "XPert_official_schedule_cpu_receipt.json", receipt)
    return receipt


def _continue_exp055_after_smoke(context: dict[str, Any]) -> dict[str, Any]:
    """Close the smoke, enforce the one-time 5h/10h gate, then run the next attempt."""
    cfg = context["cfg"]
    limits = context["limits"]
    smoke_attempt: Path = context["attempt"]
    smoke_ledger: Path = context["ledger"]
    results: dict[str, Any] = context["results"]
    smoke_actual_seconds = float(context["smoke_elapsed_seconds"])
    smoke_actual_hours = smoke_actual_seconds / 3600.0
    xpert_update_seconds = float(context["xpert_update_seconds"])
    xpert_eval_seconds = float(context["xpert_eval_seconds"])
    transigen_update_seconds = float(context["transigen_update_seconds"])
    transigen_eval_seconds = float(context["transigen_eval_seconds"])
    observed_initialization_seconds = (
        float(context["xpert_initialization_seconds"])
        + float(context["transigen_initialization_seconds"]))
    # Scale the 64-key update to four author batch128 updates and seven
    # TranSiGen batch64 updates. Add every-epoch train+inner-dev scoring,
    # double initialization allowance, 5% file/optimizer bookkeeping, then
    # the registered 20% margin. XPert is conservatively projected through
    # its full 2500-epoch maximum; official patience may stop it earlier.
    xpert_epoch_seconds = (math.ceil(419 / 128) * 2.0 * xpert_update_seconds
                           + xpert_eval_seconds * (512.0 / 157.0))
    transigen_epoch_seconds = (math.ceil(419 / 64) * transigen_update_seconds
                               + transigen_eval_seconds * (512.0 / 157.0))
    xpert_raw_seconds = 2500 * xpert_epoch_seconds
    transigen_raw_seconds = 300 * transigen_epoch_seconds
    initialization_full_seconds = observed_initialization_seconds * 2.0
    full_fit_raw_seconds = (xpert_raw_seconds + transigen_raw_seconds + initialization_full_seconds) * 1.05
    full_fit_with_margin_hours = full_fit_raw_seconds * 1.20 / 3600.0
    accounting: dict[str, Any] = context["accounting"]
    package_cumulative_projected = (float(accounting["package_settled_gpu_hours"])
                                    + smoke_actual_hours + full_fit_with_margin_hours)
    exp_cumulative_projected = (float(accounting["exp_settled_gpu_hours"])
                                + smoke_actual_hours + full_fit_with_margin_hours)
    first_phase_cap = limits["first_phase_gpu_cap_hours"]
    package_cap = limits["package_cap_gpu_hours"]
    exp_cap = limits["total_exp_cap_gpu_hours"]
    reallocation_receipts = list(RUN_DIR.glob("fit_attempt_*/one_time_package_reallocation_receipt.json"))
    reallocation_authorized = bool(cfg.get("authorization", {}).get("one_internal_reallocation_within_package", False))
    smoke_pass = smoke_actual_hours <= limits["paired_smoke_cap_gpu_hours"]
    package_pass = package_cumulative_projected <= package_cap
    exp_pass = exp_cumulative_projected <= exp_cap
    if not smoke_pass:
        decision = "STOP_SMOKE_EXCEEDED_0_25_GPUH"
    elif not package_pass or not exp_pass:
        decision = "STOP_PROJECTED_COST_EXCEEDS_AUTHORIZED_PACKAGE_OR_EXP_CAP"
    elif package_cumulative_projected <= first_phase_cap:
        decision = "PASS_WITHIN_INITIAL_5_GPUH_PHASE"
    elif reallocation_authorized and not reallocation_receipts:
        decision = "PASS_AFTER_ONE_TIME_INTERNAL_REALLOCATION_WITHIN_10_GPUH_PACKAGE"
    else:
        decision = "STOP_INITIAL_5_GPUH_EXCEEDED_AND_INTERNAL_REALLOCATION_UNAVAILABLE"
    proceed_formal = decision.startswith("PASS_")
    results["cost_gate"] = {
        "projection_method": "64-key measured updates scaled to 419 TRAIN/93 inner_dev: TranSiGen 7 batch64 updates/epoch for 300 epochs; XPert 4 batch128 updates/epoch at 2x measured batch64 update for the complete official 2500-epoch maximum; every-epoch train/dev evaluation, doubled initialization allowance, 5% bookkeeping reserve and 20% cost margin. XPert official patience=50 is applied during training but not used to lower the pre-run maximum-cost estimate.",
        "XPert_official_schedule": {"maximum_epochs": 2500, "warmup_epochs": 70,
                                   "patience": 50, "selection": "val_loss4"},
        "XPert_observed_update_seconds_batch64": xpert_update_seconds,
        "XPert_observed_train64_dev93_eval_seconds": xpert_eval_seconds,
        "XPert_projected_raw_seconds_per_epoch": xpert_epoch_seconds,
        "XPert_projected_2500_epoch_raw_gpu_hours": xpert_raw_seconds / 3600.0,
        "TranSiGen_schedule": {"maximum_epochs": 300, "batch_size": 64,
                               "raw_sum": True, "gradient_clipping": False},
        "TranSiGen_observed_update_seconds_batch64": transigen_update_seconds,
        "TranSiGen_observed_train64_dev93_eval_seconds": transigen_eval_seconds,
        "TranSiGen_projected_raw_seconds_per_epoch": transigen_epoch_seconds,
        "TranSiGen_projected_300_epoch_raw_gpu_hours": transigen_raw_seconds / 3600.0,
        "initialization_seconds_smoke_pair": observed_initialization_seconds,
        "initialization_allowance_full_pair_seconds": initialization_full_seconds,
        "file_optimizer_bookkeeping_factor": 1.05,
        "projected_full_fit_raw_gpu_hours": full_fit_raw_seconds / 3600.0,
        "projected_full_fit_gpu_hours_with_20_percent_margin": full_fit_with_margin_hours,
        "smoke_actual_gpu_hours": smoke_actual_hours,
        "smoke_absolute_cap_gpu_hours": limits["paired_smoke_cap_gpu_hours"],
        "smoke_within_cap": smoke_pass,
        "package_settled_before_artifact_ledgers_gpu_hours": limits["package_settled_before_artifact_ledgers_gpu_hours"],
        "package_artifact_ledgers_settled_before_smoke_gpu_hours": accounting["package_artifact_ledger_settled_gpu_hours"],
        "package_settled_before_smoke_gpu_hours": accounting["package_settled_gpu_hours"],
        "package_cumulative_projected_with_smoke_and_margin_gpu_hours": package_cumulative_projected,
        "package_first_phase_cap_gpu_hours": first_phase_cap,
        "package_total_cap_gpu_hours": package_cap,
        "exp_settled_before_smoke_gpu_hours": accounting["exp_settled_gpu_hours"],
        "exp_cumulative_projected_with_smoke_and_margin_gpu_hours": exp_cumulative_projected,
        "exp_total_cap_gpu_hours": exp_cap,
        "one_time_reallocation_authorized": reallocation_authorized,
        "one_time_reallocation_already_used": bool(reallocation_receipts),
        "decision": decision,
        "formal_fit_started": False,
    }
    results["baseline_metrics_path"] = "baseline_metrics.json"
    results["baseline_metrics_sha256"] = sha256_file(smoke_attempt / "baseline_metrics.json")
    results["baseline_train_denominator"] = {"conditions": 419, "objects": 110,
                                               "genes": 978, "normalizer_fit_split": "TRAIN only"}
    results["smoke_prediction_path"] = context["prediction_path"].name
    results["smoke_predictions_sha256"] = sha256_file(context["prediction_path"])
    results["smoke_checkpoints"] = {
        "XPert": {"path": "XPert_smoke_one_update_state.pt",
                  "sha256": sha256_file(smoke_attempt / "XPert_smoke_one_update_state.pt"), "updates": 1},
        "TranSiGen": {"path": "TranSiGen_smoke_one_update_state.pt",
                      "sha256": sha256_file(smoke_attempt / "TranSiGen_smoke_one_update_state.pt"), "updates": 1}}
    results["XPert_official_schedule_cpu_receipt"] = json.loads(
        (smoke_attempt / "XPert_official_schedule_cpu_receipt.json").read_text(encoding="utf-8"))
    results["cpu_audited_features"] = context["feature_artifact_receipt"]
    results["readback_receipt"] = context["readback_receipt"]
    results["resource"] = {"wsl_pid": context["process_id"], "device_uuid": GPU_UUID,
                           "smoke_gpu_hours_actual": smoke_actual_hours,
                           "smoke_attempt_reservation_gpu_hours": limits["paired_smoke_cap_gpu_hours"],
                           "device_snapshot_after_smoke": _device_snapshot()["selected_gpu"]}
    atomic_json(smoke_attempt / "fit_smoke_result.json", results)
    smoke_status = {"status": "SMOKE_COMPLETED_COST_GATE_PASS" if proceed_formal else decision,
                    "actual_gpu_hours": smoke_actual_hours,
                    "reserved_gpu_hours": limits["paired_smoke_cap_gpu_hours"],
                    "unused_reservation_released_gpu_hours": max(0.0, limits["paired_smoke_cap_gpu_hours"] - smoke_actual_hours),
                    "cost_gate": results["cost_gate"], "runner_sha256": context["manifest"]["current_runner_sha256"],
                    "config_sha256": context["config_sha256"], "process_id": context["process_id"],
                    "device_uuid": GPU_UUID}
    atomic_json(smoke_attempt / "attempt_close_receipt.json", smoke_status)
    append_jsonl(smoke_ledger, {"event": "GPU_ATTEMPT_COMPLETED" if proceed_formal else "GPU_ATTEMPT_CLOSED_AFTER_COST_GATE",
                                "at_utc": utc_now(), "wsl_pid": context["process_id"],
                                "device_uuid": GPU_UUID, "actual_gpu_hours": smoke_actual_hours,
                                "reserved_gpu_hours": limits["paired_smoke_cap_gpu_hours"],
                                "unused_authorized_reservation_released_gpu_hours": max(0.0, limits["paired_smoke_cap_gpu_hours"] - smoke_actual_hours),
                                "cumulative_package_projected_with_margin_gpu_hours": package_cumulative_projected,
                                "cumulative_exp_projected_with_margin_gpu_hours": exp_cumulative_projected,
                                "cost_gate_decision": decision,
                                "cost_gate_passed": proceed_formal,
                                "close_receipt": "attempt_close_receipt.json",
                                "gpu_snapshot_at_close": _device_snapshot()["selected_gpu"]})
    atomic_json(smoke_attempt / "attempt_progress.json", {
        "phase": "CLOSED_COST_GATE_PASS" if proceed_formal else "CLOSED_COST_GATE_STOP",
        "updated_at_utc": utc_now(), "wsl_pid": context["process_id"],
        "smoke_actual_gpu_hours": smoke_actual_hours, "cost_gate_decision": decision})
    print(json.dumps({"stage": "EXP055_SMOKE_COST_GATE", "decision": decision,
                      "smoke_actual_gpu_hours": smoke_actual_hours,
                      "projected_TranSiGen_300_epochs_gpu_hours": results["cost_gate"]["TranSiGen_projected_300_epoch_raw_gpu_hours"],
                      "projected_XPert_2500_epochs_gpu_hours": results["cost_gate"]["XPert_projected_2500_epoch_raw_gpu_hours"],
                      "projected_full_fit_with_margin_gpu_hours": full_fit_with_margin_hours,
                      "package_cumulative_projected_gpu_hours": package_cumulative_projected,
                      "attempt": str(smoke_attempt)}, ensure_ascii=False), flush=True)
    if not proceed_formal:
        return {"smoke": results, "close_receipt": smoke_status, "formal_fit_started": False}

    if decision.startswith("PASS_AFTER_ONE_TIME"):
        reallocation_receipt = {"status": "USED_ONCE_WITHIN_USER_AUTHORIZED_10_GPUH_PACKAGE",
                                "used_at_utc": utc_now(), "from_phase_cap_gpu_hours": first_phase_cap,
                                "to_package_cap_gpu_hours": package_cap,
                                "projected_cumulative_package_gpu_hours": package_cumulative_projected,
                                "smoke_actual_gpu_hours": smoke_actual_hours,
                                "projected_formal_fits_with_20_percent_margin_gpu_hours": full_fit_with_margin_hours,
                                "authorization_source": "EXP-055 user-approved one_internal_reallocation_within_package"}
        atomic_json(smoke_attempt / "one_time_package_reallocation_receipt.json", reallocation_receipt)

    formal_accounting = resource_accounting(
        EXPERIMENT_ROOT, RUN_ID, limits["package_cap_gpu_hours"], limits["total_exp_cap_gpu_hours"],
        limits["exp_settled_before_artifact_ledgers_gpu_hours"],
        limits["package_settled_before_artifact_ledgers_gpu_hours"])
    formal_reservation = min(full_fit_with_margin_hours,
                             float(formal_accounting["remaining_authorized_gpu_hours"]))
    if formal_reservation + 1e-9 < full_fit_with_margin_hours:
        raise RuntimeError("formal EXP-055 reservation fell below its projected cost plus 20% margin")
    formal_attempt_id = next_attempt_id(RUN_DIR)
    formal_attempt = new_attempt_path(RUN_DIR, formal_attempt_id)
    formal_attempt.mkdir(parents=False, exist_ok=False)
    process_id = int(context["process_id"])
    current_runner_sha = sha256_file(Path(__file__).resolve())
    if current_runner_sha != context["manifest"]["current_runner_sha256"]:
        raise RuntimeError("EXP-055 runner changed after the smoke reservation began")
    gpu_preflight = _assert_exp055_gpu_idle(allowed_pids={str(process_id)})
    atomic_json(formal_attempt / "pre_gpu_preflight_started.json", {
        "run_id": RUN_ID, "attempt": formal_attempt_id, "status": "CPU_PREFLIGHT_STARTED",
        "started_at_utc": utc_now(), "wsl_pid": process_id,
        "runner_sha256": current_runner_sha, "source_run_id": SOURCE_RUN_ID,
        "config_sha256": context["config_sha256"], "official_cost_gate": decision,
        "reserved_gpu_hours": formal_reservation})
    baseline_receipt = _reuse_exp051_baseline_readbacks(formal_attempt)
    shutil.copy2(smoke_attempt / "fit_smoke_64_keys.tsv", formal_attempt / "fit_smoke_64_keys.tsv")
    formal_manifest = dict(context["manifest"])
    formal_manifest.update({"fit_attempt": formal_attempt_id, "phase": "full_exp055_original_schedule",
                            "created_at_utc": utc_now(), "reservation_gpu_hours": formal_reservation,
                            "resource_accounting_before_attempt": formal_accounting,
                            "current_runner_sha256": current_runner_sha,
                            "device_snapshot_before_attempt": gpu_preflight["device_snapshot"],
                            "gpu_process_preflight": gpu_preflight,
                            "cost_gate_source": str(smoke_attempt / "fit_smoke_result.json"),
                            "cost_gate_decision": decision,
                            "training_schedule": {"TranSiGen": {"epochs": 300, "batch_size": 64,
                                                                    "objective": "raw_sum", "clip": False},
                                                  "XPert": {"maximum_epochs": 2500, "patience": 50,
                                                            "selection": "val_loss4",
                                                            "warmup_epochs": 70}}})
    atomic_json(formal_attempt / "fit_smoke_manifest.json", formal_manifest)
    formal_ledger = formal_attempt / "resource_ledger.jsonl"
    formal_start = time.perf_counter()
    append_jsonl(formal_ledger, {"event": "GPU_ATTEMPT_STARTED", "at_utc": utc_now(),
                                 "wsl_pid": process_id, "device_uuid": GPU_UUID,
                                 "reserved_gpu_hours": formal_reservation,
                                 "reservation_scope": "TranSiGen 300 epochs plus XPert official 2500/patience50 schedule with 20% margin",
                                 "cost_gate_decision": decision,
                                 "smoke_attempt_id": smoke_attempt.name,
                                 "smoke_actual_gpu_hours_already_settled": smoke_actual_hours,
                                 "package_settled_before_reservation_gpu_hours": formal_accounting["package_settled_gpu_hours"],
                                 "remaining_authorized_gpu_hours_at_start": formal_accounting["remaining_authorized_gpu_hours"],
                                 "device_snapshot_before_attempt": gpu_preflight["device_snapshot"]["selected_gpu"],
                                 "runner_sha256": current_runner_sha,
                                 "config_sha256": context["config_sha256"]})
    atomic_json(formal_attempt / "attempt_progress.json", {
        "phase": "FULL_FITS_STARTED", "updated_at_utc": utc_now(), "wsl_pid": process_id,
        "training_order": ["TranSiGen", "XPert"], "TranSiGen_max_epochs": 300,
        "XPert_max_epochs": 2500, "XPert_patience": 50})
    print(json.dumps({"stage": "EXP055_FORMAL_ATTEMPT_STARTED", "attempt": str(formal_attempt),
                      "reserved_gpu_hours": formal_reservation,
                      "package_settled_gpu_hours": formal_accounting["package_settled_gpu_hours"],
                      "device_uuid": GPU_UUID}, ensure_ascii=False), flush=True)
    try:
        full_fit = _train_exp055_full_models(
            context["torch"], formal_attempt, context["keys"], context["values"],
            context["train_mask"], context["dev_mask"], context["train_keys"], context["dev_keys"],
            context["gene_variance"], baseline_receipt, context["unimol"], context["xpert_module"],
            context["xpert_config"], context["feature_by_key"], context["device"],
            device_budget_hours=formal_reservation, elapsed_gpu_origin=formal_start)
        full_fit["cost_gate"] = results["cost_gate"]
        full_fit["baseline_metrics_path"] = "baseline_metrics.json"
        full_fit["baseline_metrics_sha256"] = sha256_file(formal_attempt / "baseline_metrics.json")
        full_fit["fit_gate_baseline_denominator"] = {"strongest_baseline_candidates": ["zero_change", "per_drug_mean_delta"],
                                                      "baseline_metrics_sha256": full_fit["baseline_metrics_sha256"],
                                                      "TRAIN_conditions": 419, "TRAIN_objects": 110,
                                                      "normalizer_fit_on": "TRAIN only"}
        atomic_json(formal_attempt / "full_fit_result.json", full_fit)
        full_elapsed_seconds = time.perf_counter() - formal_start
        full_actual_hours = full_elapsed_seconds / 3600.0
        if full_actual_hours > formal_reservation:
            raise TimeoutError("EXP-055 formal schedule exceeded its projected-plus-margin reservation")
        final_snapshot = _device_snapshot()["selected_gpu"]
        close = {"status": full_fit["status"],
                 "completed_epochs": {"TranSiGen": full_fit["best_metrics"]["TranSiGen"]["epochs_completed"],
                                      "XPert": full_fit["best_metrics"]["XPert"]["epochs_completed"]},
                 "cost_gate": results["cost_gate"], "actual_gpu_hours": full_actual_hours,
                 "reserved_gpu_hours": formal_reservation,
                 "unused_reservation_released_gpu_hours": formal_reservation - full_actual_hours,
                 "cumulative_package_charge_gpu_hours": formal_accounting["package_settled_gpu_hours"] + full_actual_hours,
                 "runner_sha256": current_runner_sha, "config_sha256": context["config_sha256"],
                 "process_id": process_id, "device_uuid": GPU_UUID,
                 "device_snapshot_at_close": final_snapshot,
                 "baseline_metrics_sha256": full_fit["baseline_metrics_sha256"],
                 "full_keyed_predictions_sha256": full_fit["predictions_sha256"]}
        atomic_json(formal_attempt / "attempt_close_receipt.json", close)
        append_jsonl(formal_ledger, {"event": "GPU_ATTEMPT_COMPLETED", "at_utc": utc_now(),
                                     "wsl_pid": process_id, "device_uuid": GPU_UUID,
                                     "actual_gpu_hours": full_actual_hours,
                                     "reserved_gpu_hours": formal_reservation,
                                     "unused_authorized_reservation_released_gpu_hours": formal_reservation - full_actual_hours,
                                     "cumulative_package_charge_gpu_hours": formal_accounting["package_settled_gpu_hours"] + full_actual_hours,
                                     "completed_epochs": close["completed_epochs"],
                                     "cost_gate_decision": decision,
                                     "full_keyed_predictions_sha256": full_fit["predictions_sha256"],
                                     "close_receipt": "attempt_close_receipt.json",
                                     "gpu_snapshot_at_close": final_snapshot})
        atomic_json(formal_attempt / "attempt_progress.json", {
            "phase": "CLOSED_COMPLETED", "updated_at_utc": utc_now(), "wsl_pid": process_id,
            "actual_gpu_hours": full_actual_hours, "completed_epochs": close["completed_epochs"]})
        print(json.dumps({"stage": "EXP055_FORMAL_ATTEMPT_COMPLETED", "attempt": str(formal_attempt),
                          "actual_gpu_hours": full_actual_hours, "completed_epochs": close["completed_epochs"],
                          "best_metrics": full_fit["best_metrics"]}, ensure_ascii=False), flush=True)
        return {"smoke": results, "full_fit": full_fit, "smoke_close_receipt": smoke_status,
                "formal_close_receipt": close}
    except BaseException as error:
        _close_unhandled_attempt(RUN_ID, formal_attempt_id, error)
        raise


def _xpert_official_selection_loss(torch: Any, pcc_loss_func: Any,
                                  prediction: np.ndarray, target: np.ndarray,
                                  device: Any) -> float:
    predicted = torch.as_tensor(prediction, dtype=torch.float32, device=device)
    observed = torch.as_tensor(target, dtype=torch.float32, device=device)
    value = float((pcc_loss_func(predicted, observed) / len(target)).detach().cpu())
    if not math.isfinite(value):
        raise FloatingPointError("XPert official inner-dev val_loss4 is non-finite")
    return value


def _verify_exp055_prediction_and_checkpoint_readback(
        torch: Any, attempt: Path, prediction_file: Path,
        expected_arrays: dict[str, np.ndarray], checkpoint_names: list[str]) -> dict[str, Any]:
    """Read saved predictions/checkpoints back and bind their exact keys and axes."""
    with np.load(prediction_file, allow_pickle=False) as saved:
        checks: dict[str, Any] = {"prediction_file_sha256": sha256_file(prediction_file),
                                  "prediction_fields": sorted(saved.files)}
        for name, expected in expected_arrays.items():
            if name not in saved.files:
                raise KeyError(f"EXP-055 prediction readback is missing {name}: {prediction_file}")
            actual = saved[name]
            if actual.shape != expected.shape or not np.array_equal(actual.astype(str), expected.astype(str)):
                raise ValueError(f"EXP-055 prediction readback mismatch for {name}")
            checks[name] = {"shape": list(actual.shape), "exact": True}
        prediction_fields = [name for name in saved.files if name.endswith("_delta") and name not in
                             {"train_target_delta", "inner_dev_target_delta"}]
        nonfinite_prediction_fields: dict[str, int] = {}
        for name in prediction_fields:
            array = saved[name]
            if array.shape[-1] != 978:
                raise ValueError(f"EXP-055 saved prediction gene axis differs from 978: {name}")
            nonfinite_prediction_fields[name] = int(array.size - np.isfinite(array).sum())
        checks["finite_prediction_fields"] = prediction_fields
        checks["nonfinite_prediction_counts"] = nonfinite_prediction_fields
    checkpoint_checks = {}
    for filename in checkpoint_names:
        path = attempt / filename
        if not path.is_file():
            raise FileNotFoundError(f"EXP-055 checkpoint readback is missing: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload.get("model_state_dict"))
        if not isinstance(state, dict) or not state:
            raise ValueError(f"EXP-055 checkpoint has no state_dict: {path}")
        nonfinite = [name for name, value in state.items()
                    if isinstance(value, torch.Tensor) and value.is_floating_point()
                    and not torch.isfinite(value).all()]
        if nonfinite:
            raise FloatingPointError(f"EXP-055 checkpoint contains non-finite state tensors: {path}, {nonfinite[:3]}")
        optimizer_state = payload.get("optimizer_state_dict", {})
        optimizer_tensor_rows: list[tuple[str, Any]] = []
        def collect_optimizer_tensors(value: Any, prefix: str = "optimizer") -> None:
            if isinstance(value, torch.Tensor):
                optimizer_tensor_rows.append((prefix, value))
            elif isinstance(value, dict):
                for key, item in value.items():
                    collect_optimizer_tensors(item, f"{prefix}.{key}")
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    collect_optimizer_tensors(item, f"{prefix}.{index}")
        collect_optimizer_tensors(optimizer_state)
        nonfinite_optimizer = [name for name, value in optimizer_tensor_rows
            if value.is_floating_point() and not bool(torch.isfinite(value).all().item())]
        if nonfinite_optimizer:
            raise FloatingPointError(f"EXP-055 checkpoint optimizer state is nonfinite: {path}, {nonfinite_optimizer[:3]}")
        checkpoint_checks[filename] = {"sha256": sha256_file(path),
                                       "state_tensor_count": sum(isinstance(v, torch.Tensor) for v in state.values()),
                                       "state_tensor_shapes": {name: list(value.shape) for name, value in state.items()
                                                                if isinstance(value, torch.Tensor)},
                                       "optimizer_state_tensor_count": len(optimizer_tensor_rows),
                                       "optimizer_state_all_finite": not bool(nonfinite_optimizer),
                                       "metadata": {name: value for name, value in payload.items()
                                                    if name not in {"state_dict", "optimizer_state_dict"}}}
    receipt = {"status": "PASS_KEY_AND_CHECKPOINT_READBACK",
               "prediction_file": prediction_file.name, "checks": checks,
               "checkpoints": checkpoint_checks}
    atomic_json(attempt / "prediction_checkpoint_readback_receipt.json", receipt)
    return receipt


def _cpu_copy_tree(torch: Any, value: Any) -> Any:
    """Clone a nested optimizer state to CPU so an official best checkpoint is immutable."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy_tree(torch, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy_tree(torch, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy_tree(torch, item) for item in value)
    return value


def _train_exp055_full_models(torch: Any, attempt: Path, keys: pd.DataFrame,
                              values: dict[str, np.ndarray], train_mask: np.ndarray,
                              dev_mask: np.ndarray, train_keys: pd.DataFrame,
                              dev_keys: pd.DataFrame, gene_variance: np.ndarray,
                              baseline_metrics: dict[str, Any], unimol: np.ndarray,
                              xpert_module: Any, xpert_config: dict[str, Any],
                              feature_by_key: dict[str, np.ndarray], device: Any,
                              device_budget_hours: float,
                              elapsed_gpu_origin: float) -> dict[str, Any]:
    """Run the full EXP-055 TranSiGen and author-scheduled XPert fits in order."""
    from torch.utils.data import DataLoader, TensorDataset

    train_positions = np.flatnonzero(train_mask)
    dev_positions = np.flatnonzero(dev_mask)
    train_y = values["delta"][train_mask]
    dev_y = values["delta"][dev_mask]
    transigen_curve = attempt / "TranSiGen_training_curve.jsonl"
    xpert_curve = attempt / "XPert_training_curve.jsonl"
    if transigen_curve.exists() or xpert_curve.exists():
        raise FileExistsError("refusing to overwrite an EXP-055 training curve")

    # TranSiGen runs first with the registered upstream raw-sum update contract.
    train_features = _full_key_features(keys, train_positions, feature_by_key)
    dev_features = _full_key_features(keys, dev_positions, feature_by_key)
    tensor_train = TensorDataset(
        torch.as_tensor(values["X_ctl"][train_mask], dtype=torch.float32),
        torch.as_tensor(values["X"][train_mask], dtype=torch.float32),
        torch.as_tensor(train_features, dtype=torch.float32))
    transigen_loader = DataLoader(
        tensor_train, batch_size=64, shuffle=True, drop_last=False, num_workers=0,
        generator=torch.Generator().manual_seed(SEED))
    transigen, transigen_module = _load_transigen(torch, device, SEED, features_dim=2048)
    transigen_optimizer = torch.optim.Adam(transigen.parameters(), lr=1e-3, weight_decay=1e-5)
    transigen_best_loss = math.inf
    transigen_best_epoch = -1
    transigen_best_state: dict[str, Any] | None = None
    transigen_best_train = transigen_best_dev = None
    transigen_epoch_seconds: list[float] = []
    transigen_snapshot_predictions: dict[str, np.ndarray] = {}
    transigen_loss_names = ("raw_total_loss", "mse_x1", "mse_x2", "mse_pert",
                            "kld_x1", "kld_x2", "kld_pert")
    for epoch in range(300):
        started = time.perf_counter()
        transigen.train()
        loss_sums = np.zeros(7, dtype=np.float64)
        gradient_norms: list[float] = []
        update_count = 0
        for x1, x2, features in transigen_loader:
            if len(x1) == 1:
                continue
            transigen.zero_grad(set_to_none=True)
            batch = (x1.to(device), x2.to(device), features.to(device))
            _, _, _, losses = _transigen_forward_batch(torch, transigen, batch, device)
            if not all(torch.isfinite(loss).all() for loss in losses):
                raise FloatingPointError(f"TranSiGen raw-sum training loss is non-finite at epoch {epoch + 1}")
            losses[0].backward()
            gradient_norm = _grad_global_norm(torch, transigen)
            if not math.isfinite(gradient_norm):
                raise FloatingPointError(f"TranSiGen gradient norm is non-finite at epoch {epoch + 1}")
            gradient_norms.append(gradient_norm)
            transigen_optimizer.step()
            loss_sums += np.asarray([float(value.detach().cpu()) for value in losses])
            update_count += 1
        if update_count == 0:
            raise RuntimeError(f"TranSiGen epoch {epoch + 1} made no optimizer updates")
        torch.cuda.synchronize()
        train_pred, train_eval_loss = _transigen_predict_and_losses(
            torch, transigen, values["X_ctl"][train_mask], values["X"][train_mask],
            train_features, 64, device, SEED + 2 * epoch + 1)
        dev_pred, dev_eval_loss = _transigen_predict_and_losses(
            torch, transigen, values["X_ctl"][dev_mask], values["X"][dev_mask],
            dev_features, 64, device, SEED + 2 * epoch + 2)
        train_metric = keyed_metrics(train_keys, train_y, train_pred, gene_variance)
        dev_metric = keyed_metrics(dev_keys, dev_y, dev_pred, gene_variance)
        duration = time.perf_counter() - started
        transigen_epoch_seconds.append(duration)
        row = {"epoch": epoch + 1, "learning_rate": 1e-3, "weight_decay": 1e-5,
               "objective": "raw_sum", "gradient_clipping": False,
               "train_batch_size": 64, "train_shuffle": True,
               "train_loss_terms_mean_per_condition": {
                   name: safe_number(float(value / len(train_positions)))
                   for name, value in zip(transigen_loss_names, loss_sums)},
               "train_eval_loss_terms": train_eval_loss, "inner_dev_eval_loss_terms": dev_eval_loss,
               "gradient_norm_mean": safe_number(float(np.mean(gradient_norms))),
               "gradient_norm_max": safe_number(float(np.max(gradient_norms))),
               "gradient_norm_n_updates": len(gradient_norms), "train_metrics": train_metric,
               "inner_dev_metrics": dev_metric, "epoch_seconds": duration}
        append_jsonl(transigen_curve, row)
        selection_value = dev_eval_loss["raw_total_loss"]
        if selection_value is None or not math.isfinite(float(selection_value)):
            raise FloatingPointError(f"TranSiGen inner-dev raw loss is undefined at epoch {epoch + 1}")
        if float(selection_value) < transigen_best_loss:
            transigen_best_loss = float(selection_value)
            transigen_best_epoch = epoch + 1
            transigen_best_state = {k: v.detach().cpu().clone()
                                    for k, v in transigen.state_dict().items()}
            transigen_best_train, transigen_best_dev = train_pred.copy(), dev_pred.copy()
        if epoch + 1 in (100, 300):
            name = f"TranSiGen_epoch_{epoch + 1}.pt"
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in transigen.state_dict().items()},
                        "model": "upstream TranSiGen", "epoch": epoch + 1,
                        "seed": SEED, "selection_metric": "inner_dev raw_total_loss"}, attempt / name)
            transigen_snapshot_predictions[f"epoch_{epoch + 1}_train_delta"] = train_pred.copy()
            transigen_snapshot_predictions[f"epoch_{epoch + 1}_inner_dev_delta"] = dev_pred.copy()
        elapsed = time.perf_counter() - elapsed_gpu_origin
        if elapsed / 3600 >= device_budget_hours:
            raise TimeoutError("EXP-055 GPU reservation reached during TranSiGen training")
        atomic_json(attempt / "training_progress.json", {
            "model": "TranSiGen", "completed_epochs": epoch + 1, "target_epochs": 300,
            "selected_inner_dev_loss_epoch": transigen_best_epoch,
            "elapsed_gpu_hours_upper_bound": elapsed / 3600, "updated_at_utc": utc_now()})
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 in (100, 300):
            print(json.dumps({"stage": "EXP055_TRANSGEN_PROGRESS", "epoch": epoch + 1,
                              "train_nmse": train_metric["object_macro_nmse"],
                              "inner_dev_nmse": dev_metric["object_macro_nmse"],
                              "elapsed_gpu_hours": elapsed / 3600}, ensure_ascii=False), flush=True)
    if transigen_best_state is None or transigen_best_train is None or transigen_best_dev is None:
        raise RuntimeError("TranSiGen completed without a finite inner-dev checkpoint")
    transigen_final_state = {k: v.detach().cpu().clone() for k, v in transigen.state_dict().items()}
    torch.save({"state_dict": transigen_best_state, "model": "upstream TranSiGen",
                "epoch": transigen_best_epoch, "seed": SEED,
                "selection_metric": "minimum inner_dev raw_total_loss"},
               attempt / "TranSiGen_best_inner_dev.pt")
    torch.save({"state_dict": transigen_final_state, "model": "upstream TranSiGen",
                "epoch": 300, "seed": SEED, "selection_metric": "final registered epoch"},
               attempt / "TranSiGen_final_epoch_300.pt")
    transigen_snapshot_predictions["best_inner_dev_train_delta"] = transigen_best_train
    transigen_snapshot_predictions["best_inner_dev_delta"] = transigen_best_dev
    transigen_snapshot_predictions["final_epoch_300_train_delta"] = train_pred.copy()
    transigen_snapshot_predictions["final_epoch_300_inner_dev_delta"] = dev_pred.copy()
    transigen_summary = {"epochs_completed": 300, "selected_epoch_by_inner_dev_raw_total_loss": transigen_best_epoch,
                         "selected_train": keyed_metrics(train_keys, train_y, transigen_best_train, gene_variance),
                         "selected_inner_dev": keyed_metrics(dev_keys, dev_y, transigen_best_dev, gene_variance),
                         "fit_gate": _fit_gate(keyed_metrics(train_keys, train_y, transigen_best_train, gene_variance),
                                               baseline_metrics),
                         "final_train": keyed_metrics(train_keys, train_y, train_pred, gene_variance),
                         "final_inner_dev": keyed_metrics(dev_keys, dev_y, dev_pred, gene_variance),
                         "checkpoint_epochs_saved": [100, 300],
                         "checkpoint_hashes": {name: sha256_file(attempt / name) for name in
                             ("TranSiGen_epoch_100.pt", "TranSiGen_epoch_300.pt",
                              "TranSiGen_best_inner_dev.pt", "TranSiGen_final_epoch_300.pt")},
                         "mean_epoch_seconds": float(np.mean(transigen_epoch_seconds)),
                         "gradient_clipping": False}
    del transigen, transigen_module, transigen_optimizer, transigen_loader, tensor_train
    torch.cuda.empty_cache()

    # XPert uses the author's complete L1000 schedule and exact val_loss4 patience rule.
    config_train = xpert_config["train"]
    epochs = int(config_train["num_epochs"])
    patience = int(config_train["patience"])
    batch_size = int(config_train["batch_size"])
    init_epoch = int(config_train["init_epoch"])
    learning_rate = float(config_train["train_lr"])
    weight_decay = float(config_train["weight_decay"])
    if (epochs != 2500 or patience != 50 or batch_size != 128 or init_epoch != 70
            or learning_rate != 0.004 or weight_decay != 1e-5
            or list(config_train["loss_weight"]) != [0.2, 0.003, 0.2, 1]):
        raise ValueError("XPert config_l1000.yaml no longer matches the registered official schedule")
    original_train, _, pcc_loss_func, xpert_source_receipt = _source_functions(
        XPERT_ROOT / "train_xpert.py", XPERT_ROOT / "utils.py")
    xpert_model, train_view = _new_xpert(torch, xpert_module, xpert_config, device, SEED)
    xpert_opt_raw = torch.optim.Adam(xpert_model.parameters(), lr=learning_rate,
                                     weight_decay=weight_decay)
    xpert_optimizer = ProbeOptimizer(xpert_opt_raw, xpert_model, torch)
    xpert_scheduler = torch.optim.lr_scheduler.LambdaLR(
        xpert_opt_raw, lr_lambda=lambda epoch: 1.0 if epoch < 40 else 0.5)
    xpert_tensors = _xpert_tensors(torch, values, keys, train_positions, unimol,
                                   xpert_config, training_targets=True)
    xpert_loader = DataLoader(TensorDataset(*xpert_tensors), batch_size=batch_size,
                              shuffle=True, drop_last=False, num_workers=0,
                              generator=torch.Generator().manual_seed(SEED))
    best_val_loss4 = math.inf
    best_epoch = -1
    epochs_since_best = 0
    best_state = None
    best_optimizer_state = None
    best_train_prediction = best_dev_prediction = None
    last_train_prediction = last_dev_prediction = None
    last_train_metrics = last_dev_metrics = None
    xpert_epoch_seconds: list[float] = []
    stopped_by_patience = False
    for epoch in range(epochs):
        started = time.perf_counter()
        gradient_start = len(xpert_optimizer.gradient_norms)
        learning_rate_current = float(xpert_opt_raw.param_groups[0]["lr"])
        class Args:
            pretrained_model = False
        train_loss_terms = original_train(train_view, xpert_optimizer, xpert_loader, Args(),
                                          xpert_config, scaler=None, epoch=epoch)
        if not all(math.isfinite(float(value)) for value in train_loss_terms):
            raise FloatingPointError(f"XPert official training loss is non-finite at epoch {epoch + 1}")
        torch.cuda.synchronize()
        last_train_prediction = _xpert_predict(torch, xpert_model, values, keys, train_positions,
                                               unimol, xpert_config, batch_size=64)
        last_dev_prediction = _xpert_predict(torch, xpert_model, values, keys, dev_positions,
                                             unimol, xpert_config, batch_size=64)
        train_metric = keyed_metrics(train_keys, train_y, last_train_prediction, gene_variance)
        dev_metric = keyed_metrics(dev_keys, dev_y, last_dev_prediction, gene_variance)
        val_loss4 = _xpert_official_selection_loss(torch, pcc_loss_func, last_dev_prediction,
                                                   dev_y, device)
        gradients = xpert_optimizer.gradient_norms[gradient_start:]
        if not gradients:
            raise RuntimeError(f"XPert epoch {epoch + 1} completed no optimizer updates")
        improved = val_loss4 < best_val_loss4
        if improved:
            best_val_loss4, best_epoch = val_loss4, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in xpert_model.state_dict().items()}
            best_optimizer_state = _cpu_copy_tree(torch, xpert_opt_raw.state_dict())
            best_train_prediction = last_train_prediction.copy()
            best_dev_prediction = last_dev_prediction.copy()
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        duration = time.perf_counter() - started
        xpert_epoch_seconds.append(duration)
        row = {"epoch": epoch + 1, "epoch_index_zero_based": epoch,
               "schedule_stage": "per_sample_mean_root_mse" if epoch < init_epoch else "raw_weighted_sum",
               "learning_rate": learning_rate_current,
               "official_train_loss_terms": [safe_number(float(value)) for value in train_loss_terms],
               "official_inner_dev_selection": "val_loss4_mean_delta_pcc_loss",
               "inner_dev_val_loss4": val_loss4,
               "gradient_norm_mean": safe_number(float(np.mean(gradients))),
               "gradient_norm_max": safe_number(float(np.max(gradients))),
               "gradient_norm_n_updates": len(gradients), "train_metrics": train_metric,
               "inner_dev_metrics": dev_metric, "epochs_since_best": epochs_since_best,
               "epoch_seconds": duration}
        append_jsonl(xpert_curve, row)
        last_train_metrics, last_dev_metrics = train_metric, dev_metric
        elapsed = time.perf_counter() - elapsed_gpu_origin
        if elapsed / 3600 >= device_budget_hours:
            raise TimeoutError("EXP-055 GPU reservation reached during XPert training")
        atomic_json(attempt / "training_progress.json", {
            "model": "XPert", "completed_epochs": epoch + 1, "target_epochs": epochs,
            "best_epoch_by_val_loss4": best_epoch, "epochs_since_best": epochs_since_best,
            "patience": patience, "elapsed_gpu_hours_upper_bound": elapsed / 3600,
            "updated_at_utc": utc_now()})
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(json.dumps({"stage": "EXP055_XPERT_PROGRESS", "epoch": epoch + 1,
                              "best_epoch": best_epoch, "val_loss4": val_loss4,
                              "train_nmse": train_metric["object_macro_nmse"],
                              "inner_dev_nmse": dev_metric["object_macro_nmse"],
                              "elapsed_gpu_hours": elapsed / 3600}, ensure_ascii=False), flush=True)
        if epochs_since_best >= patience:
            stopped_by_patience = True
            break
        xpert_scheduler.step()
    if (best_state is None or best_optimizer_state is None
            or best_train_prediction is None or best_dev_prediction is None
            or last_train_prediction is None or last_dev_prediction is None):
        raise RuntimeError("XPert official schedule completed without a finite selected checkpoint")
    completed_epochs = epoch + 1
    final_state = {k: v.detach().cpu().clone() for k, v in xpert_model.state_dict().items()}
    final_optimizer_state = _cpu_copy_tree(torch, xpert_opt_raw.state_dict())
    torch.save({"model_state_dict": best_state, "optimizer_state_dict": best_optimizer_state,
                "epoch": best_epoch - 1, "epoch_one_based": best_epoch,
                "selection_metric": "minimum inner_dev val_loss4",
                "official_train_function_sha256": xpert_source_receipt["train_function_source_sha256"]},
               attempt / "XPert_best_inner_dev.pt")
    final_path = f"XPert_final_stop_epoch_{completed_epochs}.pt"
    torch.save({"model_state_dict": final_state, "optimizer_state_dict": final_optimizer_state,
                "epoch": completed_epochs - 1, "epoch_one_based": completed_epochs,
                "official_train_function_sha256": xpert_source_receipt["train_function_source_sha256"]},
               attempt / final_path)
    official_best_checkpoint = torch.load(attempt / "XPert_best_inner_dev.pt",
                                          map_location="cpu", weights_only=False)
    xpert_model.load_state_dict(official_best_checkpoint["model_state_dict"])
    xpert_opt_raw.load_state_dict(official_best_checkpoint["optimizer_state_dict"])
    restored_best_train = _xpert_predict(torch, xpert_model, values, keys, train_positions,
                                         unimol, xpert_config, batch_size=64)
    restored_best_dev = _xpert_predict(torch, xpert_model, values, keys, dev_positions,
                                       unimol, xpert_config, batch_size=64)
    if (not np.allclose(restored_best_train, best_train_prediction, rtol=1e-5, atol=1e-6)
            or not np.allclose(restored_best_dev, best_dev_prediction, rtol=1e-5, atol=1e-6)):
        raise RuntimeError("XPert best author-style checkpoint restore changed keyed predictions")
    best_train_prediction, best_dev_prediction = restored_best_train, restored_best_dev

    prediction_path = attempt / "full_keyed_predictions.npz"
    np.savez_compressed(
        prediction_path,
        train_condition_key=train_keys.full_condition_key.astype(str).to_numpy(),
        inner_dev_condition_key=dev_keys.full_condition_key.astype(str).to_numpy(),
        train_source_object_id=train_keys.source_object_id.astype(str).to_numpy(),
        inner_dev_source_object_id=dev_keys.source_object_id.astype(str).to_numpy(),
        train_full_inchikey=train_keys.full_inchikey.astype(str).to_numpy(),
        inner_dev_full_inchikey=dev_keys.full_inchikey.astype(str).to_numpy(),
        genes=values["genes"].astype(str), train_target_delta=train_y,
        inner_dev_target_delta=dev_y,
        TranSiGen_selected_train_delta=transigen_best_train,
        TranSiGen_selected_inner_dev_delta=transigen_best_dev,
        XPert_selected_train_delta=best_train_prediction,
        XPert_selected_inner_dev_delta=best_dev_prediction,
        XPert_final_stop_train_delta=last_train_prediction,
        XPert_final_stop_inner_dev_delta=last_dev_prediction,
        **transigen_snapshot_predictions)
    np.savez_compressed(attempt / "transigen_checkpoint_keyed_predictions.npz",
                        train_condition_key=train_keys.full_condition_key.astype(str).to_numpy(),
                        inner_dev_condition_key=dev_keys.full_condition_key.astype(str).to_numpy(),
                        genes=values["genes"].astype(str), train_target_delta=train_y,
                        inner_dev_target_delta=dev_y, **transigen_snapshot_predictions)
    full_readback = _verify_exp055_prediction_and_checkpoint_readback(
        torch, attempt, prediction_path,
        {"train_condition_key": train_keys.full_condition_key.astype(str).to_numpy(),
         "inner_dev_condition_key": dev_keys.full_condition_key.astype(str).to_numpy(),
         "train_source_object_id": train_keys.source_object_id.astype(str).to_numpy(),
         "inner_dev_source_object_id": dev_keys.source_object_id.astype(str).to_numpy(),
         "train_full_inchikey": train_keys.full_inchikey.astype(str).to_numpy(),
         "inner_dev_full_inchikey": dev_keys.full_inchikey.astype(str).to_numpy(),
         "genes": values["genes"].astype(str), "train_target_delta": train_y,
         "inner_dev_target_delta": dev_y},
        ["TranSiGen_epoch_100.pt", "TranSiGen_epoch_300.pt", "TranSiGen_best_inner_dev.pt",
         "TranSiGen_final_epoch_300.pt", "XPert_best_inner_dev.pt", final_path])
    transigen_readback_path = attempt / "transigen_checkpoint_keyed_predictions.npz"
    with np.load(transigen_readback_path, allow_pickle=False) as saved:
        for name, expected in {"train_condition_key": train_keys.full_condition_key.astype(str).to_numpy(),
                               "inner_dev_condition_key": dev_keys.full_condition_key.astype(str).to_numpy(),
                               "genes": values["genes"].astype(str), "train_target_delta": train_y,
                               "inner_dev_target_delta": dev_y}.items():
            if name not in saved.files or not np.array_equal(saved[name].astype(str), expected.astype(str)):
                raise ValueError(f"TranSiGen checkpoint prediction key readback mismatch: {name}")
        if any(not np.isfinite(saved[name]).all() for name in saved.files if name.endswith("_delta")
               and name not in {"train_target_delta", "inner_dev_target_delta"}):
            raise FloatingPointError("TranSiGen checkpoint prediction readback has non-finite values")
    full_readback["TranSiGen_checkpoint_prediction_file"] = {
        "path": transigen_readback_path.name, "sha256": sha256_file(transigen_readback_path),
        "status": "PASS_KEY_AND_VALUE_READBACK"}
    atomic_json(attempt / "prediction_checkpoint_readback_receipt.json", full_readback)
    xpert_train_metric = keyed_metrics(train_keys, train_y, best_train_prediction, gene_variance)
    xpert_dev_metric = keyed_metrics(dev_keys, dev_y, best_dev_prediction, gene_variance)
    xpert_summary = {"max_epochs": epochs, "epochs_completed": completed_epochs,
                     "stopped_by_patience": stopped_by_patience,
                     "patience": patience, "patience_metric": "upstream val_loss4 (mean delta PCC loss)",
                     "selected_epoch_by_val_loss4": best_epoch,
                     "selected_val_loss4": best_val_loss4,
                     "best_checkpoint_restored_and_prediction_readback_passed": True,
                     "init_epoch_transition": init_epoch, "batch_size": batch_size,
                     "learning_rate": learning_rate, "weight_decay": weight_decay,
                     "scheduler": "LambdaLR index <40 uses 1.0 and index >=40 uses 0.5; the 0.5 factor begins on training-loop epoch index 40 after 40 completed epochs",
                     "train": xpert_train_metric, "inner_dev": xpert_dev_metric,
                     "final_stop_train": last_train_metrics, "final_stop_inner_dev": last_dev_metrics,
                     "fit_gate": _fit_gate(xpert_train_metric, baseline_metrics),
                     "checkpoint_hashes": {"best": sha256_file(attempt / "XPert_best_inner_dev.pt"),
                                            "final_stop": sha256_file(attempt / final_path)},
                     "mean_epoch_seconds": float(np.mean(xpert_epoch_seconds)),
                     "source_training_contract": xpert_source_receipt}
    metrics_readback = {}
    with np.load(prediction_path, allow_pickle=False) as saved:
        metric_specs = {
            "TranSiGen_best_train": ("train", "TranSiGen_selected_train_delta", transigen_summary["selected_train"]),
            "TranSiGen_best_inner_dev": ("dev", "TranSiGen_selected_inner_dev_delta", transigen_summary["selected_inner_dev"]),
            "TranSiGen_final_train": ("train", "final_epoch_300_train_delta", transigen_summary["final_train"]),
            "TranSiGen_final_inner_dev": ("dev", "final_epoch_300_inner_dev_delta", transigen_summary["final_inner_dev"]),
            "XPert_best_train": ("train", "XPert_selected_train_delta", xpert_summary["train"]),
            "XPert_best_inner_dev": ("dev", "XPert_selected_inner_dev_delta", xpert_summary["inner_dev"]),
            "XPert_final_stop_train": ("train", "XPert_final_stop_train_delta", xpert_summary["final_stop_train"]),
            "XPert_final_stop_inner_dev": ("dev", "XPert_final_stop_inner_dev_delta", xpert_summary["final_stop_inner_dev"]),
        }
        metric_fields = ("object_macro_nmse", "object_macro_mae", "object_macro_gene_correlation",
                         "object_macro_direction_agreement", "object_macro_amplitude_ratio")
        for label, (role, prediction_name, recorded) in metric_specs.items():
            truth, metric_keys = (train_y, train_keys) if role == "train" else (dev_y, dev_keys)
            recomputed = keyed_metrics(metric_keys, truth, saved[prediction_name], gene_variance)
            for field in metric_fields:
                expected_value, actual_value = recorded.get(field), recomputed.get(field)
                if expected_value is None or actual_value is None:
                    if expected_value is not actual_value:
                        raise ValueError(f"EXP-055 saved-prediction metric readback differs for {label}/{field}")
                elif not math.isclose(float(expected_value), float(actual_value), rel_tol=0.0, abs_tol=1e-7):
                    raise ValueError(f"EXP-055 saved-prediction metric readback differs for {label}/{field}")
            metrics_readback[label] = {"status": "PASS", "metrics": recomputed}
    readback_path = attempt / "prediction_checkpoint_readback_receipt.json"
    full_readback["metrics_recomputed_from_saved_predictions"] = metrics_readback
    atomic_json(readback_path, full_readback)
    del xpert_model, train_view, xpert_optimizer, xpert_opt_raw, xpert_scheduler, xpert_loader, xpert_tensors
    torch.cuda.empty_cache()

    return {"status": "COMPLETED_EXP055_REPRESENTATIVE_FITS",
            "training_order": ["TranSiGen", "XPert"],
            "best_metrics": {"TranSiGen": transigen_summary, "XPert": xpert_summary},
            "predictions_path": prediction_path.name,
            "predictions_sha256": sha256_file(prediction_path),
            "transigen_checkpoint_predictions_path": "transigen_checkpoint_keyed_predictions.npz",
            "transigen_checkpoint_predictions_sha256": sha256_file(attempt / "transigen_checkpoint_keyed_predictions.npz"),
            "prediction_checkpoint_readback_receipt_path": "prediction_checkpoint_readback_receipt.json",
            "prediction_checkpoint_readback_receipt_sha256": sha256_file(attempt / "prediction_checkpoint_readback_receipt.json"),
            "gene_axis_sha256": sha256_text("\n".join(values["genes"].astype(str)) + "\n"),
            "curves": {"TranSiGen": {"path": transigen_curve.name, "sha256": sha256_file(transigen_curve)},
                       "XPert": {"path": xpert_curve.name, "sha256": sha256_file(xpert_curve)}}}


def _train_xpert_only(torch: Any, attempt: Path, keys: pd.DataFrame,
                      values: dict[str, np.ndarray], train_mask: np.ndarray,
                      dev_mask: np.ndarray, train_keys: pd.DataFrame,
                      dev_keys: pd.DataFrame, gene_variance: np.ndarray,
                      baseline_metrics: dict[str, Any], unimol: np.ndarray,
                      xpert_module: Any, xpert_config: dict[str, Any],
                      device: Any, device_budget_hours: float,
                      elapsed_gpu_origin: float) -> dict[str, Any]:
    """Fit XPert alone; EXP-055 follows its verified official 2500/patience50 schedule."""
    from torch.utils.data import DataLoader, TensorDataset

    train_positions = np.flatnonzero(train_mask)
    dev_positions = np.flatnonzero(dev_mask)
    train_y, dev_y = values["delta"][train_mask], values["delta"][dev_mask]
    curve = attempt / "XPert_training_curve.jsonl"
    if curve.exists():
        raise FileExistsError(f"refusing to overwrite XPert curve: {curve}")
    original_train, _, pcc_loss_func, source_receipt = _source_functions(
        XPERT_ROOT / "train_xpert.py", XPERT_ROOT / "utils.py")
    config_train = xpert_config["train"]
    exp055 = EXPERIMENT_ID == "EXP-055"
    epochs = int(config_train["num_epochs"]) if exp055 else 100
    patience = int(config_train["patience"]) if exp055 else None
    batch_size = int(config_train["batch_size"])
    init_epoch = int(config_train["init_epoch"])
    if batch_size != 128 or init_epoch != 70:
        raise ValueError("XPert must preserve upstream batch128 and 70-epoch stage transition")
    if exp055 and (epochs != 2500 or patience != 50
                   or float(config_train["train_lr"]) != 0.004
                   or float(config_train["weight_decay"]) != 1e-5
                   or list(config_train["loss_weight"]) != [0.2, 0.003, 0.2, 1]):
        raise ValueError("EXP-055 XPert must retain the verified official config_l1000 schedule")
    model, train_view = _new_xpert(torch, xpert_module, xpert_config, device, SEED)
    opt_raw = torch.optim.Adam(model.parameters(), lr=float(xpert_config["train"]["train_lr"]),
                               weight_decay=float(xpert_config["train"]["weight_decay"]))
    opt = ProbeOptimizer(opt_raw, model, torch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt_raw, lr_lambda=lambda epoch: 1.0 if epoch < 40 else 0.5)
    tensors = _xpert_tensors(torch, values, keys, train_positions, unimol, xpert_config,
                             training_targets=True)
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True,
                        drop_last=False, num_workers=0,
                        generator=torch.Generator().manual_seed(SEED))
    best_nmse, best_val_loss4, best_epoch, best_state = math.inf, math.inf, -1, None
    best_optimizer_state = None
    best_train = best_dev = None
    epochs_since_best = 0
    stopped_by_patience = False
    epoch_seconds = []
    for epoch in range(epochs):
        started = time.perf_counter()
        grad_start = len(opt.gradient_norms)
        lr = float(opt_raw.param_groups[0]["lr"])
        class Args:
            pretrained_model = False
        losses = original_train(train_view, opt, loader, Args(), xpert_config,
                                scaler=None, epoch=epoch)
        if exp055 and not all(math.isfinite(float(value)) for value in losses):
            raise FloatingPointError(f"XPert official training loss is non-finite at epoch {epoch + 1}")
        torch.cuda.synchronize()
        train_pred = _xpert_predict(torch, model, values, keys, train_positions,
                                    unimol, xpert_config, batch_size=64)
        dev_pred = _xpert_predict(torch, model, values, keys, dev_positions,
                                  unimol, xpert_config, batch_size=64)
        train_metric = keyed_metrics(train_keys, train_y, train_pred, gene_variance)
        dev_metric = keyed_metrics(dev_keys, dev_y, dev_pred, gene_variance)
        val_loss4 = (_xpert_official_selection_loss(torch, pcc_loss_func, dev_pred, dev_y, device)
                     if exp055 else None)
        grad = opt.gradient_norms[grad_start:]
        if not grad:
            raise RuntimeError(f"XPert epoch {epoch + 1} completed no optimizer updates")
        duration = time.perf_counter() - started
        epoch_seconds.append(duration)
        row = {"epoch": epoch + 1, "epoch_index_zero_based": epoch,
               "schedule_stage": "per_sample_mean_root_mse" if epoch < init_epoch else "raw_weighted_sum",
               "learning_rate": lr, "official_train_loss_terms": [safe_number(float(x)) for x in losses],
               "official_inner_dev_selection": "val_loss4_mean_delta_pcc_loss" if exp055 else "object_macro_nmse",
               "inner_dev_val_loss4": val_loss4,
               "gradient_norm_mean": safe_number(float(np.mean(grad))),
               "gradient_norm_max": safe_number(float(np.max(grad))),
               "gradient_norm_n_updates": len(grad), "train_metrics": train_metric,
               "inner_dev_metrics": dev_metric, "epoch_seconds": duration}
        append_jsonl(curve, row)
        value = val_loss4 if exp055 else dev_metric.get("object_macro_nmse")
        current_best = best_val_loss4 if exp055 else best_nmse
        if value is not None and math.isfinite(float(value)) and float(value) < current_best:
            if exp055:
                best_val_loss4 = float(value)
                best_optimizer_state = _cpu_copy_tree(torch, opt_raw.state_dict())
            else:
                best_nmse = float(value)
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_train, best_dev = train_pred.copy(), dev_pred.copy()
            epochs_since_best = 0
            if exp055:
                _atomic_torch_checkpoint(torch, attempt / "XPert_best_inner_dev.pt", {
                    "model_state_dict": best_state,
                    "optimizer_state_dict": best_optimizer_state,
                    "epoch": best_epoch - 1, "epoch_one_based": best_epoch,
                    "selection_metric": "minimum inner_dev val_loss4",
                    "official_train_function_sha256": source_receipt["train_function_source_sha256"]})
        elif exp055:
            epochs_since_best += 1
        elapsed = time.perf_counter() - elapsed_gpu_origin
        if elapsed / 3600 >= device_budget_hours:
            raise TimeoutError("ledger-derived GPU reservation reached during XPert fit")
        atomic_json(attempt / "training_progress.json", {
            "model": "XPert", "completed_epochs": epoch + 1, "target_epochs": epochs,
            "best_inner_dev_epoch": best_epoch, "elapsed_gpu_hours_upper_bound": elapsed / 3600,
            "updated_at_utc": utc_now()})
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(json.dumps({"stage": "XPert_FULL_FIT_PROGRESS", "epoch": epoch + 1,
                              "train_nmse": train_metric["object_macro_nmse"],
                              "inner_dev_nmse": dev_metric["object_macro_nmse"],
                              "elapsed_gpu_hours": elapsed / 3600}, ensure_ascii=False), flush=True)
        if exp055 and epochs_since_best >= int(patience):
            stopped_by_patience = True
            break
        scheduler.step()
    if best_state is None or best_train is None or best_dev is None:
        raise RuntimeError("XPert fit finished without a finite inner-development checkpoint")
    completed_epochs = epoch + 1
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_payload = {"state_dict": best_state, "model": "upstream XPertNet", "epoch": best_epoch,
                    "seed": SEED, "official_train_function_sha256": source_receipt["train_function_source_sha256"]}
    if exp055:
        best_payload = {"model_state_dict": best_state,
                        "optimizer_state_dict": best_optimizer_state,
                        "epoch": best_epoch - 1, "epoch_one_based": best_epoch,
                        "selection_metric": "minimum inner_dev val_loss4",
                        "official_train_function_sha256": source_receipt["train_function_source_sha256"]}
    torch.save(best_payload, attempt / "XPert_best_inner_dev.pt")
    final_name = (f"XPert_final_stop_epoch_{completed_epochs}.pt" if exp055
                  else "XPert_final_epoch_100.pt")
    torch.save({"state_dict": final_state, "model": "upstream XPertNet", "epoch": completed_epochs,
                "seed": SEED, "official_train_function_sha256": source_receipt["train_function_source_sha256"]},
               attempt / final_name)
    if exp055:
        best_checkpoint = torch.load(attempt / "XPert_best_inner_dev.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        if best_optimizer_state is not None:
            opt_raw.load_state_dict(best_checkpoint["optimizer_state_dict"])
        restored_train = _xpert_predict(torch, model, values, keys, train_positions, unimol,
                                        xpert_config, batch_size=64)
        restored_dev = _xpert_predict(torch, model, values, keys, dev_positions, unimol,
                                      xpert_config, batch_size=64)
        if (not np.allclose(restored_train, best_train, rtol=1e-5, atol=1e-6)
                or not np.allclose(restored_dev, best_dev, rtol=1e-5, atol=1e-6)):
            raise RuntimeError("XPert official best-checkpoint restore changed keyed predictions")
        best_train, best_dev = restored_train, restored_dev
        np.savez_compressed(
            attempt / "full_keyed_predictions.npz",
            train_condition_key=np.asarray(train_keys.full_condition_key.astype(str).to_numpy(), dtype=np.str_),
            inner_dev_condition_key=np.asarray(dev_keys.full_condition_key.astype(str).to_numpy(), dtype=np.str_),
            train_source_object_id=np.asarray(train_keys.source_object_id.astype(str).to_numpy(), dtype=np.str_),
            inner_dev_source_object_id=np.asarray(dev_keys.source_object_id.astype(str).to_numpy(), dtype=np.str_),
            train_full_inchikey=np.asarray(train_keys.full_inchikey.astype(str).to_numpy(), dtype=np.str_),
            inner_dev_full_inchikey=np.asarray(dev_keys.full_inchikey.astype(str).to_numpy(), dtype=np.str_),
            genes=np.asarray(values["genes"].astype(str), dtype=np.str_), train_target_delta=train_y,
            inner_dev_target_delta=dev_y, XPert_best_inner_dev_train_delta=best_train,
            XPert_best_inner_dev_dev_delta=best_dev,
            XPert_final_stop_train_delta=train_pred.copy(), XPert_final_stop_inner_dev_delta=dev_pred.copy())
        _verify_exp055_prediction_and_checkpoint_readback(
            torch, attempt, attempt / "full_keyed_predictions.npz",
            {"train_condition_key": np.asarray(train_keys.full_condition_key.astype(str).to_numpy(), dtype=np.str_),
             "inner_dev_condition_key": np.asarray(dev_keys.full_condition_key.astype(str).to_numpy(), dtype=np.str_),
             "train_source_object_id": np.asarray(train_keys.source_object_id.astype(str).to_numpy(), dtype=np.str_),
             "inner_dev_source_object_id": np.asarray(dev_keys.source_object_id.astype(str).to_numpy(), dtype=np.str_),
             "train_full_inchikey": np.asarray(train_keys.full_inchikey.astype(str).to_numpy(), dtype=np.str_),
             "inner_dev_full_inchikey": np.asarray(dev_keys.full_inchikey.astype(str).to_numpy(), dtype=np.str_),
             "genes": np.asarray(values["genes"].astype(str), dtype=np.str_), "train_target_delta": train_y,
             "inner_dev_target_delta": dev_y},
            ["XPert_best_inner_dev.pt", final_name])
    if not exp055:
        np.savez_compressed(
            attempt / "full_keyed_predictions.npz",
            train_condition_key=np.asarray(train_keys.full_condition_key.astype(str).to_numpy(), dtype=np.str_),
            inner_dev_condition_key=np.asarray(dev_keys.full_condition_key.astype(str).to_numpy(), dtype=np.str_),
            train_source_object_id=np.asarray(train_keys.source_object_id.astype(str).to_numpy(), dtype=np.str_),
            inner_dev_source_object_id=np.asarray(dev_keys.source_object_id.astype(str).to_numpy(), dtype=np.str_),
            train_full_inchikey=np.asarray(train_keys.full_inchikey.astype(str).to_numpy(), dtype=np.str_),
            inner_dev_full_inchikey=np.asarray(dev_keys.full_inchikey.astype(str).to_numpy(), dtype=np.str_),
            genes=np.asarray(values["genes"].astype(str), dtype=np.str_), train_target_delta=train_y,
            inner_dev_target_delta=dev_y, XPert_best_inner_dev_train_delta=best_train,
            XPert_best_inner_dev_dev_delta=best_dev)
    del model, train_view, opt, opt_raw, scheduler, loader, tensors
    torch.cuda.empty_cache()
    train_metrics = keyed_metrics(train_keys, train_y, best_train, gene_variance)
    dev_metrics = keyed_metrics(dev_keys, dev_y, best_dev, gene_variance)
    return {"status": "COMPLETED_XPERT_OFFICIAL_SCHEDULE_ONLY" if exp055 else "COMPLETED_100_EPOCHS_XPERT_ONLY",
            "selected_by": ("minimum official inner_dev val_loss4 with patience=50; best checkpoint restored"
                            if exp055 else "minimum object-macro inner_dev NMSE across fixed 100 epochs; all rounds retained"),
            "best_metrics": {"XPert": {("selected_epoch_by_inner_dev_val_loss4" if exp055
                                          else "selected_epoch_by_inner_dev_nmse"): best_epoch,
                            "train": train_metrics, "inner_dev": dev_metrics,
                            "fit_gate": _fit_gate(train_metrics, baseline_metrics),
                            "max_epochs": epochs, "epochs_completed": completed_epochs,
                            "stopped_by_patience": stopped_by_patience,
                            "patience": patience, "selected_val_loss4": best_val_loss4 if exp055 else None,
                            "init_epoch_transition": init_epoch,
                            "batch_size": batch_size,
                            "selection_metric": "val_loss4" if exp055 else "object_macro_nmse",
                            "checkpoint_best_sha256": sha256_file(attempt / "XPert_best_inner_dev.pt"),
                            "checkpoint_final_sha256": sha256_file(attempt / final_name),
                            "mean_epoch_seconds": float(np.mean(epoch_seconds))}},
            "predictions_path": "full_keyed_predictions.npz",
            "predictions_sha256": sha256_file(attempt / "full_keyed_predictions.npz"),
            "gene_axis_sha256": sha256_text("\n".join(values["genes"].astype(str)) + "\n"),
            "curves": {"XPert": {"path": curve.name, "sha256": sha256_file(curve)}},
            "unselected_models_initialized_or_evaluated": []}


def _fit_phase_superseded_attempt_001_002(cfg: dict[str, Any], config_sha: str) -> dict[str, Any]:
    """Retained for audit context only; main never calls this pre-alias runner."""
    raise RuntimeError("superseded fit runner is disabled; use fit_phase with an explicit --attempt-id")
    stage_receipts = {
        "freeze": RUN_DIR / "metadata_freeze_receipt.json",
        "axis": RUN_DIR / "gene_axis_receipt.json",
        "values": RUN_DIR / "value_contract_receipt.json",
        "root_values_readback": RUN_DIR / "root_values_readback_receipt.json",
    }
    for label, path in stage_receipts.items():
        if not path.is_file():
            raise FileNotFoundError(f"required {label} gate receipt missing: {path}")
    freeze = json.loads(stage_receipts["freeze"].read_text(encoding="utf-8"))
    axis = json.loads(stage_receipts["axis"].read_text(encoding="utf-8"))
    values_receipt = json.loads(stage_receipts["values"].read_text(encoding="utf-8"))
    independent_values = json.loads(stage_receipts["root_values_readback"].read_text(encoding="utf-8"))
    if freeze.get("config_sha256") != config_sha or axis.get("status") != "PASS_EXACT_978_AXIS":
        raise RuntimeError("frozen key/config or 978-gene axis hard gate did not pass")
    if independent_values.get("checks", {}).get("result") != "PASS":
        raise RuntimeError("coordinator's independent frozen-value readback did not pass")
    if values_receipt.get("files_sha256", {}).get("frozen_train_inner_values_npz") != independent_values.get("input_sha256"):
        raise RuntimeError("frozen values differ from the independent readback receipt")

    existing_attempts = sorted(RUN_DIR.glob("fit_attempt_[0-9][0-9][0-9]"))
    attempt_index = len(existing_attempts) + 1
    attempt = RUN_DIR / f"fit_attempt_{attempt_index:03d}"
    command = ("cd /mnt/d/Code/DrugScreenLab && export CUDA_VISIBLE_DEVICES=" + GPU_UUID
               + " && export PYTHONPATH=src && /home/dell/miniconda3/bin/conda run --no-capture-output"
               " -n drugscreening-gpu python scripts/modeling/exp051_contract_fit_smoke.py --phase fit")
    keys = pd.read_csv(RUN_DIR / "response_blind_conditions.tsv", sep="\t", dtype=str)
    # This frozen file was written by our own values phase and independently
    # hash-verified by the coordinator. Its role and identity sidecars use
    # object dtype; load only this trusted artifact with pickle enabled.
    frozen_npz = np.load(RUN_DIR / "frozen_train_inner_values.npz", allow_pickle=True)
    values = {name: frozen_npz[name].copy() for name in
              ("selection_rank", "source_row", "split_role", "source_object_id", "full_inchikey",
               "pert_id", "genes", "X", "X_ctl", "delta")}
    frozen_npz.close()
    if len(keys) != values["X"].shape[0] or values["X"].shape != (512, 978):
        raise ValueError("the frozen model matrices do not match the expected 512x978 contract")
    if not np.array_equal(pd.to_numeric(keys["source_row"], errors="raise").to_numpy(dtype=np.int64),
                          values["source_row"].astype(np.int64)):
        raise ValueError("frozen source-row order differs between key table and numeric values")
    for column in ("source_object_id", "full_inchikey", "pert_id", "split_role"):
        if not np.array_equal(keys[column].astype(str).to_numpy(), values[column].astype(str)):
            raise ValueError(f"frozen key/value identity mismatch in {column}")
    train_mask = keys["split_role"].eq("train").to_numpy()
    dev_mask = keys["split_role"].eq("inner_dev").to_numpy()
    train_ids = set(keys.loc[train_mask, "source_object_id"])
    dev_ids = set(keys.loc[dev_mask, "source_object_id"])
    if train_ids & dev_ids or int(train_mask.sum()) != 419 or int(dev_mask.sum()) != 93:
        raise ValueError("frozen whole-object TRAIN/inner-dev split contract failed")

    # XPert's MyDataset indexes UniMol/HG features with H5AD obs.pert_idx;
    # bind that runtime field to the frozen identity table before training.
    import anndata as ad
    h5ad_meta = ad.read_h5ad(H5AD, backed="r")
    try:
        feature_meta = h5ad_meta.obs.iloc[values["source_row"].astype(np.int64)][["pert_id", "pert_idx"]].copy()
    finally:
        h5ad_meta.file.close()
    feature_meta = feature_meta.reset_index(drop=True)
    if not np.array_equal(feature_meta["pert_id"].astype(str).to_numpy(), keys["pert_id"].astype(str).to_numpy()):
        raise ValueError("source H5AD pert_id changed against the frozen condition table")
    feature_ids = pd.to_numeric(feature_meta["pert_idx"], errors="raise").to_numpy(dtype=np.int64)
    node_ids = pd.to_numeric(keys["drug_node_idx"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(feature_ids, node_ids):
        raise ValueError("XPert source pert_idx does not equal the frozen drug_node_idx feature axis")
    attempt.mkdir(parents=True, exist_ok=False)
    train_keys, dev_keys = keys.loc[train_mask].reset_index(drop=True), keys.loc[dev_mask].reset_index(drop=True)
    train_y, dev_y = values["delta"][train_mask], values["delta"][dev_mask]
    gene_variance = np.var(train_y.astype(np.float64), axis=0, ddof=0)
    baseline_receipt = data_baselines(keys, values["delta"], gene_variance, train_mask, dev_mask, attempt)
    baselines = baseline_receipt["metrics"]

    # Select 64 frozen TRAIN conditions, response-blind and evenly across all five drugs.
    train_positions = np.flatnonzero(train_mask)
    drug_keys = sorted(keys.loc[train_mask, "full_inchikey"].unique())
    if len(drug_keys) != 5:
        raise ValueError(f"expected the five frozen complete drug groups, got {len(drug_keys)}")
    chosen: list[int] = []
    per_drug_target = [13, 13, 13, 13, 12]
    for key, count in zip(drug_keys, per_drug_target):
        available = train_positions[keys.loc[train_mask, "full_inchikey"].to_numpy() == key]
        ordered = sorted(available, key=lambda pos: (str(keys.iloc[pos].source_object_id),
                                                      str(keys.iloc[pos].full_condition_key)))
        if len(ordered) < count:
            raise ValueError(f"64-key fit smoke cannot retain paired drug coverage for {key}")
        chosen.extend(ordered[:count])
    smoke_positions = np.asarray(chosen, dtype=np.int64)
    smoke_positions.sort()
    smoke_keys = keys.iloc[smoke_positions].reset_index(drop=True)
    if smoke_keys.full_inchikey.nunique() != 5 or len(smoke_keys) != 64:
        raise AssertionError("paired smoke lost a complete key or condition")
    smoke_keys.to_csv(attempt / "fit_smoke_64_keys.tsv", sep="\t", index=False, lineterminator="\n")
    xpert_schedule_receipt = (_verify_exp055_xpert_cpu_schedule(attempt)
                             if EXPERIMENT_ID == "EXP-055" else None)

    source_hashes = _source_hashes()
    process_id = os.getpid()
    snapshot = _device_snapshot()
    if snapshot["selected_gpu"]["memory_used_mib"] > 1024 or snapshot["selected_gpu"]["utilization_percent"] > 20:
        raise RuntimeError("approved A5000 is no longer idle enough for the bounded smoke")
    manifest = {
        "experiment_id": "EXP-051", "run_id": RUN_ID, "fit_attempt": "fit_attempt_001",
        "phase": "64_key_original_model_cost_smoke", "created_at_utc": utc_now(),
        "command": command, "wsl_pid": process_id, "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "frozen_receipts": {key: {"path": str(path), "sha256": sha256_file(path)}
                            for key, path in stage_receipts.items()},
        "frozen_files": {"conditions_tsv_sha256": sha256_file(RUN_DIR / "response_blind_conditions.tsv"),
                         "values_npz_sha256": sha256_file(RUN_DIR / "frozen_train_inner_values.npz")},
        "current_runner_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": config_sha, "source_and_feature_sha256": source_hashes,
        "device_snapshot": snapshot,
        "identity_units_boundary": "source H5AD processed layer; stored-value units remain project-level unknown. No rescaling, z-scoring, clipping, or inverse transform is performed in EXP-051.",
        "data_roles": {"TRAIN_conditions": 419, "TRAIN_objects": 110,
                       "inner_dev_conditions": 93, "inner_dev_objects": 24,
                       "object_overlap": 0, "smoke_conditions": 64,
                       "smoke_drugs": 5, "external_or_locked_expression_opened": False},
        "feature_contract": "XPert source pert_idx exactly matches the frozen drug_node_idx on all 512 frozen rows; original UniMol and ECFP4 features use that author-defined feature index; full InChIKey remains the condition identity.",
        "model_arms": {
            "XPert": "upstream XPertNet from random init; official train_xpert.train function body; config_l1000.yaml batch128/lr0.004/wd1e-5/init_epoch70/weights[0.2,0.003,0.2,1]; first-stage batch objective per source.",
            "TranSiGen": "upstream TranSiGen model from random init; source model.loss and raw-sum training update, Adam lr1e-3/wd1e-5; upstream pretrain_shRNA initialization deliberately disabled per approved from-scratch comparison."},
        "upstream_schedule_evidence": {
            "XPert_official": {"config": str(XPERT_ROOT / "configs/config_l1000.yaml"),
                               "train": str(XPERT_ROOT / "train_xpert.py"),
                               "train_epoch_switch": "epoch < config['train']['init_epoch'] uses per-sample mean-root MSE components; epoch >= init_epoch uses raw weighted sums",
                               "old_EXP050_runner_difference": "old runner did not call train_xpert.train; batch16, patience10, raw weighted loss from epoch1. These are implementation differences; they do not identify the cause of any prior outcome."},
            "TranSiGen_official": {"train": str(TRANSIGEN_ROOT / "train_TranSiGen.py"),
                                   "model": str(TRANSIGEN_ROOT / "model.py"),
                                   "old_EXP050_runner_difference": "old diagnostic arms used scratch/lr1e-4/100 epochs/fixed order; sum/no-clip and norm+clip were update-rule diagnostics, not official reproduction."}},
        "limits": ["This phase estimates full-fit compute; a single optimizer step is not a model-fit result.",
                   "The numeric units of the author-processed H5AD layer are not established here.",
                   "No causal attribution is made from implementation/schedule differences."],
    }
    atomic_json(attempt / "fit_smoke_manifest.json", manifest)
    append_jsonl(attempt / "resource_ledger.jsonl", {
        "event": "GPU_SMOKE_STARTED", "at_utc": utc_now(), "wsl_pid": process_id,
        "device_uuid": GPU_UUID,
        "reserved_gpu_hours": resource_limits(cfg)["paired_smoke_cap_gpu_hours"],
        "reservation_scope": "64-key paired original-model cost smoke and approved first fit gate",
        "pre_smoke_gpu_snapshot": snapshot["selected_gpu"]})

    import torch
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the approved WSL2 environment")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(json.dumps({"stage": "GPU_SMOKE_STARTED", "pid": process_id, "device": str(device),
                      "uuid": GPU_UUID, "run_dir": str(attempt)}, ensure_ascii=False), flush=True)
    start_gpu = time.perf_counter()
    results: dict[str, Any] = {"smoke_metrics": {}, "timings_seconds": {}}

    # Load only model features for the five frozen drug nodes.
    drug_nodes = sorted(keys["drug_node_idx"].astype(int).unique().tolist())
    if len(drug_nodes) != 5:
        raise ValueError("frozen identity table does not map exactly five drugs to five model nodes")
    unimol_asset = np.load(XPERT_ROOT / "processed_data/all_drugs_unimol_arr.npy", mmap_mode="r")
    if unimol_asset.ndim != 3 or unimol_asset.shape[1:] != (122, 514):
        raise ValueError(f"unexpected official XPert UniMol shape: {unimol_asset.shape}")
    if max(drug_nodes) >= unimol_asset.shape[0]:
        raise IndexError("frozen XPert drug node index lies outside UniMol feature array")
    unimol = unimol_asset
    del unimol_asset
    morgan, morgan_receipt = _build_transigen_ecfp4(
        keys, DRUG_MAP, attempt / "transigen_ecfp4_features.npz")
    results["feature_assets"] = {"unimol_shape": [8981, 122, 514], "unimol_dtype": "float64",
                                 "unimol_sha256": source_hashes["xpert_unimol"],
                                 "TranSiGen_ECFP4": morgan_receipt,
                                 "feature_drug_node_indices": drug_nodes}

    # XPert: call the exact upstream epoch trainer on a single 64-example loader.
    tick = time.perf_counter()
    xpert_mod, xpert_config, _, _ = _training_modules(torch)
    original_train, mse_func, pcc_func, source_train_receipt = _source_functions(
        XPERT_ROOT / "train_xpert.py", XPERT_ROOT / "utils.py")
    xpert_config["dataset"]["drug_unimol_path"] = str(XPERT_ROOT / "processed_data/all_drugs_unimol_arr.npy")
    xpert, train_view = _new_xpert(torch, xpert_mod, xpert_config, device, SEED)
    xpert_inputs = _xpert_tensors(torch, values, keys, smoke_positions, unimol, xpert_config,
                                   training_targets=True)
    class Args:
        pretrained_model = False
    opt_raw = torch.optim.Adam(xpert.parameters(), lr=0.004, weight_decay=1e-5)
    opt = ProbeOptimizer(opt_raw, xpert, torch)
    xpert_params_before = [parameter.detach().clone() for parameter in xpert.parameters()]
    xpert_before_train = _xpert_predict(torch, xpert, values, keys, smoke_positions, unimol, xpert_config)
    xpert_before_dev = _xpert_predict(torch, xpert, values, keys, np.flatnonzero(dev_mask), unimol, xpert_config)
    train_started = time.perf_counter()
    loss_components = original_train(train_view, opt, [xpert_inputs], Args(), xpert_config,
                                     scaler=None, epoch=0)
    torch.cuda.synchronize()
    xpert_update_seconds = time.perf_counter() - train_started
    xpert_after_train = _xpert_predict(torch, xpert, values, keys, smoke_positions, unimol, xpert_config)
    xpert_after_dev = _xpert_predict(torch, xpert, values, keys, np.flatnonzero(dev_mask), unimol, xpert_config)
    results["timings_seconds"]["XPert_feature_and_init"] = time.perf_counter() - tick
    results["timings_seconds"]["XPert_one_official_update"] = xpert_update_seconds
    results["XPert_source_train_function"] = source_train_receipt
    results["XPert_official_loss_summary"] = [safe_number(value) for value in loss_components]
    results["XPert_gradient_norm"] = safe_number(opt.gradient_norms[-1]) if opt.gradient_norms else None
    results["XPert_parameter_update_l2"] = safe_number(_norm_difference(
        torch, xpert_params_before, xpert))
    xpert_smoke_state = {key: value.detach().cpu() for key, value in xpert.state_dict().items()}
    del xpert_params_before
    results["smoke_metrics"]["XPert"] = {
        "train64_before": keyed_metrics(smoke_keys, values["delta"][smoke_positions],
                                         xpert_before_train, gene_variance),
        "train64_after": keyed_metrics(smoke_keys, values["delta"][smoke_positions],
                                        xpert_after_train, gene_variance),
        "inner_dev93_after": keyed_metrics(dev_keys, dev_y, xpert_after_dev, gene_variance),
        "n_updates": len(opt.gradient_norms), "n_training_keys": len(smoke_positions),
        "prediction_max_abs": float(np.max(np.abs(xpert_after_dev))),
        "loss_terms_order": ["official weighted total", "trt reconstruction RMSE", "control reconstruction RMSE",
                             "delta RMSE", "delta PCC loss per sample"],
    }
    torch.cuda.empty_cache()

    # TranSiGen: same 64 identities and paired values, official raw-sum objective/update.
    tick = time.perf_counter()
    transigen, transigen_module = _load_transigen(torch, device, SEED, features_dim=2048)
    features_smoke = np.stack([morgan[int(index)] for index in smoke_keys.drug_node_idx.astype(int)], axis=0)
    x1_smoke = values["X_ctl"][smoke_positions]
    x2_smoke = values["X"][smoke_positions]
    transigen.train()
    optimizer_t = torch.optim.Adam(transigen.parameters(), lr=1e-3, weight_decay=1e-5)
    x1_t = torch.as_tensor(x1_smoke, dtype=torch.float32, device=device)
    x2_t = torch.as_tensor(x2_smoke, dtype=torch.float32, device=device)
    feat_t = torch.as_tensor(features_smoke, dtype=torch.float32, device=device)
    update_started = time.perf_counter()
    optimizer_t.zero_grad(set_to_none=True)
    x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, _ = transigen.forward(x1_t, feat_t)
    z2, mu2, logvar2 = transigen.encode_x2(x2_t)
    x2_rec = transigen.decode_x2(z2)
    transigen_losses = transigen.loss(x1_t, x1_rec, mu1, logvar1, x2_t, x2_rec,
                                      mu2, logvar2, x2_pred, mu_pred, logvar_pred)
    transigen_losses[0].backward()
    transigen_grad = _grad_global_norm(torch, transigen)
    if not math.isfinite(transigen_grad):
        raise FloatingPointError("non-finite TranSiGen gradient norm")
    optimizer_t.step()
    torch.cuda.synchronize()
    transigen_update_seconds = time.perf_counter() - update_started
    dev_positions = np.flatnonzero(dev_mask)
    del xpert, train_view, xpert_mod
    torch.cuda.empty_cache()
    transigen_train_pred = _transigen_predict_arrays(torch, transigen, x1_smoke, x2_smoke,
                                                      features_smoke, 64, device)
    features_dev = np.stack([morgan[int(index)] for index in dev_keys.drug_node_idx.astype(int)], axis=0)
    transigen_dev_pred = _transigen_predict_arrays(torch, transigen, values["X_ctl"][dev_positions],
                                                    values["X"][dev_positions], features_dev, 64, device)
    results["timings_seconds"]["TranSiGen_feature_and_init"] = time.perf_counter() - tick
    results["timings_seconds"]["TranSiGen_one_official_update"] = transigen_update_seconds
    results["TranSiGen_loss_terms"] = {name: safe_number(float(value.detach().cpu())) for name, value in zip(
        ("raw_total_loss", "mse_x1", "mse_x2", "mse_pert", "kld_x1", "kld_x2", "kld_pert"), transigen_losses)}
    results["TranSiGen_gradient_norm"] = safe_number(transigen_grad)
    results["smoke_metrics"]["TranSiGen"] = {
        "train64_after": keyed_metrics(smoke_keys, values["delta"][smoke_positions],
                                        transigen_train_pred, gene_variance),
        "inner_dev93_after": keyed_metrics(dev_keys, dev_y, transigen_dev_pred, gene_variance),
        "n_updates": 1, "n_training_keys": len(smoke_positions),
        "prediction_max_abs": float(np.max(np.abs(transigen_dev_pred))),
        "loss_terms_order": ["total", "MSE x1", "MSE x2", "MSE perturbation", "KL x1", "KL x2", "KL perturbation"],
    }

    prediction_path = attempt / "smoke_predictions.npz"
    np.savez_compressed(prediction_path,
                        train64_condition_key=smoke_keys.full_condition_key.astype(str).to_numpy(),
                        inner_dev_condition_key=dev_keys.full_condition_key.astype(str).to_numpy(),
                        genes=values["genes"].astype(str),
                        train64_target_delta=values["delta"][smoke_positions],
                        inner_dev_target_delta=dev_y,
                        xpert_train64_before_delta=xpert_before_train,
                        xpert_train64_after_delta=xpert_after_train,
                        xpert_inner_dev_after_delta=xpert_after_dev,
                        transigen_train64_after_delta=transigen_train_pred,
                        transigen_inner_dev_after_delta=transigen_dev_pred)
    xpert_ckpt = attempt / "XPert_smoke_one_update_state.pt"
    transigen_ckpt = attempt / "TranSiGen_smoke_one_update_state.pt"
    torch.save({"state_dict": xpert_smoke_state, "model": "upstream XPertNet",
                "updates": 1, "seed": SEED,
                "official_train_function_sha256": source_train_receipt["train_function_source_sha256"]}, xpert_ckpt)
    # Smoke checkpoints are diagnostic-only; the code hash and update receipt
    # make clear that each contains a single update, not a fitted model.
    torch.save({"state_dict": transigen.state_dict(), "model": "upstream TranSiGen",
                "updates": 1, "seed": SEED}, transigen_ckpt)

    elapsed = time.perf_counter() - start_gpu
    # Costs include full per-epoch train and dev prediction/evaluation passes.
    # Use the observed optimizer step, then a conservative 1.5x allowance for
    # batch growth, source I/O and metric passes.
    xpert_step = max(xpert_update_seconds, 1e-4)
    transigen_step = max(transigen_update_seconds, 1e-4)
    xpert_steps_per_epoch = int(math.ceil(419 / 128))
    transigen_steps_per_epoch = int(math.ceil(419 / 64))
    xpert_forward_seconds = results["timings_seconds"]["XPert_one_official_update"]
    transigen_forward_seconds = results["timings_seconds"]["TranSiGen_one_official_update"]
    projected_xpert_hours = (100 * (xpert_steps_per_epoch * xpert_step * 1.5 + 10 * xpert_forward_seconds)) / 3600
    projected_transigen_hours = (100 * (transigen_steps_per_epoch * transigen_step + 10 * transigen_forward_seconds)) / 3600
    projected_full_hours = projected_xpert_hours + projected_transigen_hours
    projected_with_smoke_hours = projected_full_hours + elapsed / 3600
    legacy_limits = resource_limits(cfg)
    proceed_full = projected_with_smoke_hours <= legacy_limits["paired_smoke_cap_gpu_hours"]
    results["cost_gate"] = {
        "projection_method": "per observed 64-condition training update, scaled to official batch/update counts and 100 epochs; per-epoch train+inner-dev forward passes added; XPert update estimate uses 1.5x safety multiplier for batch128, IO, and metric overhead.",
        "projected_XPert_100_epoch_gpu_hours": projected_xpert_hours,
        "projected_TranSiGen_100_epoch_gpu_hours": projected_transigen_hours,
        "projected_combined_gpu_hours": projected_full_hours,
        "first_smoke_actual_gpu_hours": elapsed / 3600,
        "projected_smoke_plus_100_epoch_gpu_hours": projected_with_smoke_hours,
        "first_smoke_cap_gpu_hours": legacy_limits["paired_smoke_cap_gpu_hours"],
        "full_100_epoch_fit_within_first_fit_smoke_cap": proceed_full,
        "decision": "PASS_AND_CONTINUE_TO_COMPLETE_100_EPOCH_FITS" if proceed_full else "STOP_AFTER_COST_SMOKE; smoke plus projected full fits exceed first-fit budget",
        "meets_train_fit_gate": None,
        "warning": "one update per model cannot establish fit; full schedule remains required before any fit-gate decision."}
    results["resource"] = {"wsl_pid": process_id, "device": GPU_UUID,
                            "gpu_hours_actual": elapsed / 3600,
                            "gpu_snapshot_after": _device_snapshot()["selected_gpu"],
                            "reserved_gpu_hours": legacy_limits["paired_smoke_cap_gpu_hours"]}
    results["artifact_contract"] = {"baseline_metrics": "baseline_metrics.json",
                                    "baseline_predictions": "baseline_predictions.npz",
                                    "fit_smoke_keys": "fit_smoke_64_keys.tsv",
                                    "smoke_predictions": prediction_path.name,
                                    "smoke_predictions_sha256": sha256_file(prediction_path),
                                    "smoke_checkpoints": {"XPert": {"path": xpert_ckpt.name,
                                                                       "sha256": sha256_file(xpert_ckpt),
                                                                       "updates": 1, "diagnostic_only": True},
                                                           "TranSiGen": {"path": transigen_ckpt.name,
                                                                          "sha256": sha256_file(transigen_ckpt),
                                                                          "updates": 1, "diagnostic_only": True}},
                                    "full_model_predictions": "not_created_by_smoke"}
    atomic_json(attempt / "fit_smoke_result.json", results)
    append_jsonl(attempt / "resource_ledger.jsonl", {
        "event": "GPU_SMOKE_FINISHED", "at_utc": utc_now(), "wsl_pid": process_id,
        "device_uuid": GPU_UUID, "actual_gpu_hours": elapsed / 3600,
        "projected_combined_gpu_hours": projected_with_smoke_hours,
        "full_100_epoch_fit_within_first_fit_smoke_cap": proceed_full})
    print(json.dumps({"stage": "GPU_SMOKE_FINISHED", "pid": process_id,
                      "actual_gpu_hours": elapsed / 3600,
                      "projected_combined_gpu_hours": projected_with_smoke_hours,
                      "cost_gate": results["cost_gate"]["decision"],
                      "fit_attempt": str(attempt)}, ensure_ascii=False), flush=True)
    return results


def fit_phase(cfg: dict[str, Any], config_sha: str, attempt_id: str) -> dict[str, Any]:
    """Run one explicitly named attempt after reading the closed resource ledger."""
    limits = resource_limits(cfg)
    accounting = resource_accounting(
        EXPERIMENT_ROOT, RUN_ID,
        limits["package_cap_gpu_hours"], limits["total_exp_cap_gpu_hours"],
        limits["exp_settled_before_artifact_ledgers_gpu_hours"],
        limits["package_settled_before_artifact_ledgers_gpu_hours"])
    settled_before_gpu_hours = float(accounting["settled_gpu_hours"])
    reserved_gpu_hours = float(accounting["remaining_authorized_gpu_hours"])
    if reserved_gpu_hours <= 0:
        raise RuntimeError(f"no authorized {EXPERIMENT_ID} package or experiment GPU balance remains")
    if EXPERIMENT_ID == "EXP-055":
        _validate_exp055_smoke_attempt(RUN_DIR, attempt_id, SOURCE_RUN_ID)
        if reserved_gpu_hours < limits["paired_smoke_cap_gpu_hours"]:
            raise RuntimeError("settled package/EXP balance cannot cover the reserved EXP-055 64-key smoke")
        authorized_uuid = cfg.get("authorization", {}).get("allowed_device_uuid")
        if authorized_uuid != GPU_UUID:
            raise ValueError("EXP-055 config UUID differs from the runner's sole A5000 allowlist")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
            raise RuntimeError("EXP-055 must set CUDA_VISIBLE_DEVICES to the sole authorized A5000 UUID")
        reserved_gpu_hours = limits["paired_smoke_cap_gpu_hours"]
    attempt = new_attempt_path(RUN_DIR, attempt_id)
    stage_receipts = {
        "freeze": SOURCE_RUN_DIR / "metadata_freeze_receipt.json",
        "axis": SOURCE_RUN_DIR / "gene_axis_receipt.json",
        "values": SOURCE_RUN_DIR / "value_contract_receipt.json",
        "root_values_readback": SOURCE_RUN_DIR / "root_values_readback_receipt.json",
        "feature_audit": SOURCE_RUN_DIR / "transigen_ecfp4_feature_audit_receipt.json",
    }
    for label, path in stage_receipts.items():
        if not path.is_file():
            raise FileNotFoundError(f"required {label} gate receipt missing: {path}")
    freeze = json.loads(stage_receipts["freeze"].read_text(encoding="utf-8"))
    axis = json.loads(stage_receipts["axis"].read_text(encoding="utf-8"))
    values_receipt = json.loads(stage_receipts["values"].read_text(encoding="utf-8"))
    independent_values = json.loads(stage_receipts["root_values_readback"].read_text(encoding="utf-8"))
    source_config_sha = freeze.get("config_sha256")
    if (freeze.get("run_id") != SOURCE_RUN_ID or axis.get("run_id") != SOURCE_RUN_ID
            or values_receipt.get("run_id") != SOURCE_RUN_ID
            or not source_config_sha or axis.get("config_sha256") != source_config_sha
            or axis.get("metadata_freeze_receipt_sha256") != sha256_file(stage_receipts["freeze"])
            or axis.get("status") != "PASS_EXACT_978_AXIS"):
        raise RuntimeError("source-run frozen key/config or exact 978-gene axis hard gate did not pass")
    if independent_values.get("checks", {}).get("result") != "PASS":
        raise RuntimeError("coordinator's independent frozen-value readback did not pass")
    if values_receipt.get("files_sha256", {}).get("frozen_train_inner_values_npz") != independent_values.get("input_sha256"):
        raise RuntimeError("frozen train/inner values differ from their independent readback")
    feature_by_key, feature_artifact_receipt = _load_audited_ecfp4(SOURCE_RUN_DIR)

    keys = pd.read_csv(SOURCE_RUN_DIR / "response_blind_conditions.tsv", sep="\t", dtype=str)
    frozen_npz = np.load(SOURCE_RUN_DIR / "frozen_train_inner_values.npz", allow_pickle=True)
    values = {name: frozen_npz[name].copy() for name in
              ("selection_rank", "source_row", "split_role", "source_object_id", "full_inchikey",
               "pert_id", "genes", "X", "X_ctl", "delta")}
    frozen_npz.close()
    if len(keys) != 512 or values["X"].shape != (512, 978) or values["X_ctl"].shape != (512, 978):
        raise ValueError("fit attempt requires only the already-frozen 512x978 TRAIN/inner_dev values")
    if not np.array_equal(pd.to_numeric(keys["source_row"], errors="raise").to_numpy(dtype=np.int64),
                          values["source_row"].astype(np.int64)):
        raise ValueError("frozen source-row order differs between key table and response values")
    for column in ("source_object_id", "full_inchikey", "pert_id", "split_role"):
        if not np.array_equal(keys[column].astype(str).to_numpy(), values[column].astype(str)):
            raise ValueError(f"frozen key/value identity mismatch in {column}")
    train_mask = keys["split_role"].eq("train").to_numpy()
    dev_mask = keys["split_role"].eq("inner_dev").to_numpy()
    train_ids = set(keys.loc[train_mask, "source_object_id"])
    dev_ids = set(keys.loc[dev_mask, "source_object_id"])
    if (train_ids & dev_ids or int(train_mask.sum()) != 419 or int(dev_mask.sum()) != 93
            or len(train_ids) != 110 or len(dev_ids) != 24):
        raise ValueError("frozen whole-object 419/93 TRAIN/inner-dev contract failed")
    if set(keys.full_inchikey.astype(str).unique()) != set(feature_by_key):
        raise ValueError("five response-blind condition keys differ from the audited ECFP4 key axis")

    # Confirm the source feature index against the frozen identity table. This
    # reads only H5AD obs metadata, never X, X_ctl, PDO, outer, or locked values.
    import anndata as ad
    h5ad_meta = ad.read_h5ad(H5AD, backed="r")
    try:
        feature_meta = h5ad_meta.obs.iloc[values["source_row"].astype(np.int64)][["pert_id", "pert_idx"]].copy()
    finally:
        h5ad_meta.file.close()
    feature_meta = feature_meta.reset_index(drop=True)
    if not np.array_equal(feature_meta["pert_id"].astype(str).to_numpy(), keys["pert_id"].astype(str).to_numpy()):
        raise ValueError("source H5AD drug identity metadata changed against the frozen key table")
    feature_ids = pd.to_numeric(feature_meta["pert_idx"], errors="raise").to_numpy(dtype=np.int64)
    node_ids = pd.to_numeric(keys["drug_node_idx"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(feature_ids, node_ids):
        raise ValueError("XPert source pert_idx differs from frozen drug_node_idx on the 512 allowed keys")

    source_hashes = _source_hashes()
    process_id = os.getpid()
    if EXPERIMENT_ID == "EXP-055":
        gpu_preflight = _assert_exp055_gpu_idle()
        snapshot = gpu_preflight["device_snapshot"]
    else:
        snapshot = _device_snapshot()
        if snapshot["selected_gpu"]["memory_used_mib"] > 1024 or snapshot["selected_gpu"]["utilization_percent"] > 20:
            raise RuntimeError("approved A5000 is not idle enough for the final bounded attempt")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    attempt.mkdir(parents=False, exist_ok=False)
    atomic_json(attempt / "pre_gpu_preflight_started.json", {
        "run_id": RUN_ID, "attempt": attempt_id, "status": "CPU_PREFLIGHT_STARTED",
        "started_at_utc": utc_now(), "wsl_pid": process_id,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "source_run_id": SOURCE_RUN_ID, "config_sha256": config_sha})
    train_keys = keys.loc[train_mask].reset_index(drop=True)
    dev_keys = keys.loc[dev_mask].reset_index(drop=True)
    train_y, dev_y = values["delta"][train_mask], values["delta"][dev_mask]
    gene_variance = np.var(train_y.astype(np.float64), axis=0, ddof=0)
    if EXPERIMENT_ID == "EXP-055":
        baseline_receipt = _reuse_exp051_baseline_readbacks(attempt)
    else:
        baseline_receipt = data_baselines(keys, values["delta"], gene_variance, train_mask, dev_mask, attempt)

    # Freeze a response-blind, 64-key balanced timing sample. Formal EXP-055
    # fits use all 419 TRAIN keys and reserve all 93 for inner development.
    train_positions = np.flatnonzero(train_mask)
    drug_keys = sorted(keys.loc[train_mask, "full_inchikey"].astype(str).unique())
    if len(drug_keys) != 5:
        raise ValueError(f"expected five full frozen drug groups, found {len(drug_keys)}")
    chosen: list[int] = []
    for full_key, count in zip(drug_keys, (13, 13, 13, 13, 12)):
        available = train_positions[keys.loc[train_mask, "full_inchikey"].astype(str).to_numpy() == full_key]
        ordered = sorted(available, key=lambda pos: (str(keys.iloc[pos].source_object_id),
                                                      str(keys.iloc[pos].full_condition_key)))
        if len(ordered) < count:
            raise ValueError(f"64-key smoke cannot cover the frozen drug {full_key}")
        chosen.extend(ordered[:count])
    smoke_positions = np.asarray(sorted(chosen), dtype=np.int64)
    smoke_keys = keys.iloc[smoke_positions].reset_index(drop=True)
    if len(smoke_keys) != 64 or smoke_keys.full_inchikey.nunique() != 5:
        raise AssertionError("paired 64-key smoke lost one of the five full-key drugs")
    smoke_keys.to_csv(attempt / "fit_smoke_64_keys.tsv", sep="\t", index=False, lineterminator="\n")

    xpert_schedule_receipt = (
        _verify_exp055_xpert_cpu_schedule(attempt)
        if EXPERIMENT_ID == "EXP-055" else None
    )

    command = ("cd /mnt/d/Code/DrugScreenLab && CUDA_VISIBLE_DEVICES=" + GPU_UUID
               + " PYTHONPATH=src /home/dell/miniconda3/bin/conda run --no-capture-output"
               " -n drugscreening-gpu python scripts/modeling/exp051_contract_fit_smoke.py"
               " --experiment-id " + EXPERIMENT_ID
               + (" --fit-kind representative" if EXPERIMENT_ID == "EXP-055" else "")
               + (" --config configs/experiments/EXP-055-representative-models.yaml" if EXPERIMENT_ID == "EXP-055" else "")
               + " --run-id " + RUN_ID + " --source-run-id " + SOURCE_RUN_ID
               + " --phase fit --attempt-id " + attempt_id
               + (" --models both" if EXPERIMENT_ID == "EXP-055" else ""))
    manifest = {
        "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID, "fit_attempt": attempt_id,
        "phase": ("EXP-055 64-key smoke capped at 0.25 GPUh; next sequential attempt starts only after the official full-schedule cost gate passes"
                  if EXPERIMENT_ID == "EXP-055" else "64_key_original_model_cost_smoke_then_authorized_full_schedule_if_cost_gate_passes"),
        "created_at_utc": utc_now(), "command": command, "wsl_pid": process_id,
        "python_executable": sys.executable, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "reservation_gpu_hours": reserved_gpu_hours,
        "resource_accounting_before_attempt": accounting,
        "source_run_id": SOURCE_RUN_ID,
        "source_config_sha256": source_config_sha,
        "resource_limits": limits,
        "frozen_receipts": {name: {"path": str(path), "sha256": sha256_file(path)}
                            for name, path in stage_receipts.items()},
        "frozen_files": {"conditions_tsv_sha256": sha256_file(SOURCE_RUN_DIR / "response_blind_conditions.tsv"),
                         "values_npz_sha256": sha256_file(SOURCE_RUN_DIR / "frozen_train_inner_values.npz")},
        "current_runner_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": config_sha, "source_and_feature_sha256": source_hashes,
        "cpu_audited_ECFP4": feature_artifact_receipt,
        "device_snapshot_before_attempt": snapshot,
        "gpu_process_preflight": gpu_preflight if EXPERIMENT_ID == "EXP-055" else None,
        "XPert_official_schedule_cpu_receipt": xpert_schedule_receipt,
        "identity_units_boundary": "XPert author-processed H5AD; stored units remain project-level unknown; no z-scoring, clipping, rescaling, or inverse transform is performed.",
        "data_roles": {"TRAIN_conditions": 419, "TRAIN_objects": 110,
                       "inner_dev_conditions": 93, "inner_dev_objects": 24,
                       "object_overlap": 0, "timing_smoke_conditions": 64,
                       "timing_smoke_full_keys": len(drug_keys),
                       "PANACEA_or_CDSDB_expression_opened": False,
                       "PDO_outer_or_locked_read": False},
        "model_arms": {
            "XPert": "Original XPertNet scratch, official train function and config; schedule is set by fit kind.",
            "TranSiGen": "Original TranSiGen scratch with 2048-bit source ECFP4 keyed by full InChIKey; raw-sum Adam update, TRAIN-shuffled batch64, no clipping; pretrain_shRNA is disabled for this from-scratch comparison."},
        "evidence_boundary": ["Old EXP-050 XPert/TranSiGen runner mismatches are implementation differences, not established causes of prior outcomes.",
                              "Single-update timing smoke is not a fit result; complete curves and inner-dev selection are written only if the cost gate passes."]}
    atomic_json(attempt / "fit_smoke_manifest.json", manifest)
    ledger = attempt / "resource_ledger.jsonl"
    reservation_start = utc_now()
    append_jsonl(ledger, {"event": "GPU_ATTEMPT_STARTED" if EXPERIMENT_ID == "EXP-055" else "GPU_SMOKE_STARTED",
                          "at_utc": reservation_start,
                          "wsl_pid": process_id, "device_uuid": GPU_UUID,
                          "reserved_gpu_hours": reserved_gpu_hours,
                          "paired_smoke_cap_gpu_hours": limits["paired_smoke_cap_gpu_hours"],
                          "reservation_scope": ("EXP-055 attempt001 only: paired 64-key model timing smoke, absolute 0.25 GPUh ceiling"
                                                if EXPERIMENT_ID == "EXP-055" else "new-run 64-key paired smoke plus projected full fits within package and EXP authorization"),
                          "prior_attempts_settled_gpu_hours": settled_before_gpu_hours,
                          "remaining_authorized_gpu_hours_at_start": reserved_gpu_hours,
                          "pre_smoke_gpu_snapshot": snapshot["selected_gpu"],
                          "runner_sha256": manifest["current_runner_sha256"],
                          "config_sha256": config_sha})

    start_gpu = time.perf_counter()
    import torch
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the approved WSL2 Conda environment")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    atomic_json(attempt / "attempt_progress.json", {"phase": "GPU_READY_PRE_MODEL_INITIALIZATION",
                                                     "updated_at_utc": utc_now(), "wsl_pid": process_id})
    print(json.dumps({"stage": "GPU_SMOKE_STARTED" if EXPERIMENT_ID != "EXP-055" else "EXP055_SMOKE_STARTED",
                      "pid": process_id,
                      "device_uuid": GPU_UUID, "attempt": str(attempt)}, ensure_ascii=False), flush=True)
    results: dict[str, Any] = {"smoke_metrics": {}, "timings_seconds": {}}
    try:
        unimol_asset = np.load(XPERT_ROOT / "processed_data/all_drugs_unimol_arr.npy", mmap_mode="r")
        if unimol_asset.ndim != 3 or unimol_asset.shape[1:] != (122, 514):
            raise ValueError(f"unexpected XPert author UniMol array shape: {unimol_asset.shape}")
        if max(node_ids) >= unimol_asset.shape[0]:
            raise IndexError("frozen XPert drug feature node is out of bounds")
        unimol = unimol_asset
        del unimol_asset

        # EXP-055 measures TranSiGen first, while XPert's full official schedule
        # is checked against the author config before its timing update.
        import torch
        from torch.utils.data import TensorDataset
        dev_positions = np.flatnonzero(dev_mask)
        transigen_smoke: dict[str, Any] | None = None
        if EXPERIMENT_ID == "EXP-055":
            atomic_json(attempt / "attempt_progress.json", {"phase": "TranSiGen_INITIALIZING_FIRST",
                                                             "updated_at_utc": utc_now(), "wsl_pid": process_id})
            transigen_smoke = _smoke_exp055_transigen_first(
                torch, attempt, keys, values, smoke_positions, dev_positions,
                smoke_keys, dev_keys, gene_variance, feature_by_key, device)
            transigen_train_pred = transigen_smoke["train_prediction"]
            transigen_dev_pred = transigen_smoke["dev_prediction"]
            transigen_update_seconds = float(transigen_smoke["update_seconds"])
            transigen_eval_seconds = float(transigen_smoke["eval_seconds"])
            transigen_initialization_seconds = float(transigen_smoke["initialization_seconds"])
            results["timings_seconds"]["TranSiGen_one_source_update"] = transigen_update_seconds
            results["timings_seconds"]["TranSiGen_train64_and_dev93_eval"] = transigen_eval_seconds
            results["timings_seconds"]["TranSiGen_smoke_initialization_and_data"] = transigen_initialization_seconds
            results["TranSiGen_loss_terms"] = transigen_smoke["loss_terms"]
            results["TranSiGen_gradient_norm"] = safe_number(transigen_smoke["gradient_norm"])
            results["smoke_metrics"]["TranSiGen"] = transigen_smoke["metrics"]
        atomic_json(attempt / "attempt_progress.json", {"phase": "XPert_INITIALIZING",
                                                         "updated_at_utc": utc_now(), "wsl_pid": process_id})
        tick = time.perf_counter()
        xpert_module, xpert_config, _, _ = _training_modules(torch)
        xpert_config["dataset"]["drug_unimol_path"] = str(XPERT_ROOT / "processed_data/all_drugs_unimol_arr.npy")
        if (int(xpert_config["dataset"]["n_bins"]) != 128
                or float(xpert_config["dataset"]["min_value"]) != 0.0
                or float(xpert_config["dataset"]["max_value"]) != 15.002
                or int(xpert_config["train"]["init_epoch"]) != 70
                or int(xpert_config["train"]["batch_size"]) != 128
                or int(xpert_config["train"]["num_epochs"]) != 2500
                or int(xpert_config["train"]["patience"]) != 50
                or float(xpert_config["train"]["train_lr"]) != 0.004
                or float(xpert_config["train"]["weight_decay"]) != 1e-5
                or list(xpert_config["train"]["loss_weight"]) != [0.2, 0.003, 0.2, 1]):
            raise ValueError("upstream XPert official L1000 fixed-schedule settings differ from the frozen contract")
        original_train, _, _, xpert_source_receipt = _source_functions(
            XPERT_ROOT / "train_xpert.py", XPERT_ROOT / "utils.py")
        if EXPERIMENT_ID == "EXP-055":
            cpu_receipt = json.loads((attempt / "XPert_official_schedule_cpu_receipt.json").read_text(encoding="utf-8"))
            if (xpert_source_receipt["train_function_source_sha256"] != cpu_receipt["train_source_sha256"]
                    or xpert_source_receipt["loss_functions_source_sha256"] != cpu_receipt["utils_source_sha256"]):
                raise RuntimeError("XPert source files changed between CPU schedule verification and GPU smoke")
        xpert, train_view = _new_xpert(torch, xpert_module, xpert_config, device, SEED)
        xpert_inputs = _xpert_tensors(torch, values, keys, smoke_positions, unimol,
                                      xpert_config, training_targets=True)
        xpert_opt_raw = torch.optim.Adam(xpert.parameters(), lr=0.004, weight_decay=1e-5)
        xpert_opt = ProbeOptimizer(xpert_opt_raw, xpert, torch)
        xpert_before_params = [parameter.detach().clone() for parameter in xpert.parameters()]
        xpert_before_train = _xpert_predict(torch, xpert, values, keys, smoke_positions, unimol, xpert_config)
        xpert_before_dev = _xpert_predict(torch, xpert, values, keys, dev_positions, unimol, xpert_config)
        results["timings_seconds"]["XPert_smoke_initialization_and_preupdate_eval"] = time.perf_counter() - tick
        atomic_json(attempt / "attempt_progress.json", {"phase": "XPert_INITIALIZED_PRE_UPDATE",
                                                         "updated_at_utc": utc_now(), "wsl_pid": process_id})
        class SmokeArgs:
            pretrained_model = False
        step_started = time.perf_counter()
        smoke_xpert_loss = original_train(train_view, xpert_opt, [xpert_inputs], SmokeArgs(),
                                          xpert_config, scaler=None, epoch=0)
        torch.cuda.synchronize()
        xpert_update_seconds = time.perf_counter() - step_started
        atomic_json(attempt / "attempt_progress.json", {"phase": "XPert_SMOKE_UPDATE_COMPLETED",
                                                         "updated_at_utc": utc_now(), "wsl_pid": process_id,
                                                         "updates": len(xpert_opt.gradient_norms)})
        eval_started = time.perf_counter()
        xpert_after_train = _xpert_predict(torch, xpert, values, keys, smoke_positions, unimol, xpert_config)
        xpert_after_dev = _xpert_predict(torch, xpert, values, keys, dev_positions, unimol, xpert_config)
        torch.cuda.synchronize()
        xpert_eval_seconds = time.perf_counter() - eval_started
        xpert_smoke_state = {key: value.detach().cpu() for key, value in xpert.state_dict().items()}
        results["timings_seconds"]["XPert_one_official_update"] = xpert_update_seconds
        results["timings_seconds"]["XPert_train64_and_dev93_eval"] = xpert_eval_seconds
        results["XPert_source_train_function"] = xpert_source_receipt
        results["XPert_official_loss_summary"] = [safe_number(float(value)) for value in smoke_xpert_loss]
        results["XPert_gradient_norm"] = safe_number(xpert_opt.gradient_norms[-1]) if xpert_opt.gradient_norms else None
        results["XPert_parameter_update_l2"] = safe_number(_norm_difference(torch, xpert_before_params, xpert))
        results["smoke_metrics"]["XPert"] = {
            "train64_before": keyed_metrics(smoke_keys, values["delta"][smoke_positions], xpert_before_train, gene_variance),
            "train64_after": keyed_metrics(smoke_keys, values["delta"][smoke_positions], xpert_after_train, gene_variance),
            "inner_dev93_after": keyed_metrics(dev_keys, dev_y, xpert_after_dev, gene_variance),
            "n_updates": len(xpert_opt.gradient_norms), "n_training_keys": len(smoke_positions),
            "prediction_max_abs": float(np.max(np.abs(xpert_after_dev))),
            "loss_terms_order": ["official weighted total", "trt reconstruction RMSE", "control reconstruction RMSE",
                                 "delta RMSE", "delta PCC loss per sample"]}
        del xpert_before_params, xpert_inputs, xpert_opt, xpert_opt_raw, xpert, train_view
        torch.cuda.empty_cache()

        if EXPERIMENT_ID != "EXP-055":
            atomic_json(attempt / "attempt_progress.json", {"phase": "TranSiGen_INITIALIZING",
                                                             "updated_at_utc": utc_now(), "wsl_pid": process_id})
            transigen_smoke = _smoke_exp055_transigen_first(
                torch, attempt, keys, values, smoke_positions, dev_positions,
                smoke_keys, dev_keys, gene_variance, feature_by_key, device)
            transigen_train_pred = transigen_smoke["train_prediction"]
            transigen_dev_pred = transigen_smoke["dev_prediction"]
            transigen_update_seconds = float(transigen_smoke["update_seconds"])
            transigen_eval_seconds = float(transigen_smoke["eval_seconds"])
            transigen_initialization_seconds = float(transigen_smoke["initialization_seconds"])
            results["timings_seconds"]["TranSiGen_smoke_initialization_and_data"] = transigen_initialization_seconds
            results["timings_seconds"]["TranSiGen_one_source_update"] = transigen_update_seconds
            results["timings_seconds"]["TranSiGen_train64_and_dev93_eval"] = transigen_eval_seconds
            results["TranSiGen_loss_terms"] = transigen_smoke["loss_terms"]
            results["TranSiGen_gradient_norm"] = safe_number(transigen_smoke["gradient_norm"])
            results["smoke_metrics"]["TranSiGen"] = transigen_smoke["metrics"]
        prediction_path = attempt / "smoke_predictions.npz"
        np.savez_compressed(prediction_path,
                            train64_condition_key=smoke_keys.full_condition_key.astype(str).to_numpy(),
                            inner_dev_condition_key=dev_keys.full_condition_key.astype(str).to_numpy(),
                            genes=values["genes"].astype(str),
                            train64_target_delta=values["delta"][smoke_positions], inner_dev_target_delta=dev_y,
                            xpert_train64_before_delta=xpert_before_train, xpert_train64_after_delta=xpert_after_train,
                            xpert_inner_dev_after_delta=xpert_after_dev,
                            transigen_train64_after_delta=transigen_train_pred,
                            transigen_inner_dev_after_delta=transigen_dev_pred)
        torch.save({"state_dict": xpert_smoke_state, "model": "upstream XPertNet",
                    "updates": 1, "seed": SEED,
                    "official_train_function_sha256": xpert_source_receipt["train_function_source_sha256"]},
                   attempt / "XPert_smoke_one_update_state.pt")
        smoke_readback_receipt = _verify_exp055_prediction_and_checkpoint_readback(
            torch, attempt, prediction_path,
            {"train64_condition_key": smoke_keys.full_condition_key.astype(str).to_numpy(),
             "inner_dev_condition_key": dev_keys.full_condition_key.astype(str).to_numpy(),
             "genes": values["genes"].astype(str)},
            ["XPert_smoke_one_update_state.pt", "TranSiGen_smoke_one_update_state.pt"])
        torch.cuda.synchronize()
        smoke_elapsed_seconds = time.perf_counter() - start_gpu

        if EXPERIMENT_ID == "EXP-055":
            return _continue_exp055_after_smoke({
                "cfg": cfg, "config_sha256": config_sha, "limits": limits,
                "accounting": accounting, "attempt": attempt, "ledger": ledger,
                "smoke_elapsed_seconds": smoke_elapsed_seconds, "results": results,
                "prediction_path": prediction_path, "process_id": process_id,
                "manifest": manifest, "xpert_update_seconds": xpert_update_seconds,
                "xpert_eval_seconds": xpert_eval_seconds,
                "readback_receipt": smoke_readback_receipt,
                "xpert_initialization_seconds": results["timings_seconds"]["XPert_smoke_initialization_and_preupdate_eval"],
                "transigen_update_seconds": transigen_update_seconds,
                "transigen_eval_seconds": transigen_eval_seconds,
                "transigen_initialization_seconds": transigen_initialization_seconds,
                "torch": torch, "keys": keys, "values": values,
                "train_mask": train_mask, "dev_mask": dev_mask,
                "train_keys": train_keys, "dev_keys": dev_keys,
                "smoke_keys": smoke_keys,
                "gene_variance": gene_variance, "baseline_receipt": baseline_receipt,
                "unimol": unimol, "xpert_module": xpert_module,
                "xpert_config": xpert_config, "feature_by_key": feature_by_key,
                "feature_artifact_receipt": feature_artifact_receipt,
                "device": device})

        # Estimate the complete 419-key/93-key schedule from observed update and
        # prediction times. XPert batch128 is conservatively costed at 2x its
        # observed batch64 update; both arms get an additional 20% margin.
        xpert_full_epoch_raw = (math.ceil(419 / 128) * xpert_update_seconds * 2.0
                                + xpert_eval_seconds * (512.0 / 157.0))
        transigen_full_epoch_raw = (math.ceil(419 / 64) * transigen_update_seconds
                                    + transigen_eval_seconds * (512.0 / 157.0))
        model_initialization_seconds = (
            results["timings_seconds"]["XPert_smoke_initialization_and_preupdate_eval"]
            + results["timings_seconds"]["TranSiGen_smoke_initialization_and_data"])
        projected_full_raw_seconds = 100 * (xpert_full_epoch_raw + transigen_full_epoch_raw) + model_initialization_seconds
        projected_full_with_margin_hours = projected_full_raw_seconds * 1.20 / 3600
        projected_total_with_margin_hours = smoke_elapsed_seconds / 3600 + projected_full_with_margin_hours
        smoke_actual_gpu_hours = smoke_elapsed_seconds / 3600
        smoke_cap_pass = smoke_actual_gpu_hours <= limits["paired_smoke_cap_gpu_hours"]
        package_projection_pass = projected_total_with_margin_hours <= reserved_gpu_hours
        proceed_full = smoke_cap_pass and package_projection_pass
        results["cost_gate"] = {
            "projection_method": "Observed 64-key paired updates and 64-key+93-dev forward passes scaled to 419 TRAIN/93 inner_dev; XPert 4 batches/epoch with batch128 update costed at 2x batch64; TranSiGen 7 updates/epoch; 100 epochs; smoke initialization included; 20% margin over projected full-fit cost.",
            "XPert_observed_update_seconds_batch64": xpert_update_seconds,
            "XPert_observed_eval_seconds_train64_dev93": xpert_eval_seconds,
            "XPert_projected_raw_seconds_per_epoch": xpert_full_epoch_raw,
            "TranSiGen_observed_update_seconds_batch64": transigen_update_seconds,
            "TranSiGen_observed_eval_seconds_train64_dev93": transigen_eval_seconds,
            "TranSiGen_projected_raw_seconds_per_epoch": transigen_full_epoch_raw,
            "projected_initialization_seconds": model_initialization_seconds,
            "projected_full_fit_raw_gpu_hours": projected_full_raw_seconds / 3600,
            "projected_full_fit_gpu_hours_with_20_percent_margin": projected_full_with_margin_hours,
            "smoke_actual_gpu_hours": smoke_actual_gpu_hours,
            "paired_smoke_cap_gpu_hours": limits["paired_smoke_cap_gpu_hours"],
            "paired_smoke_within_cap": smoke_cap_pass,
            "smoke_plus_full_fit_with_margin_gpu_hours": projected_total_with_margin_hours,
            "remaining_attempt_reservation_gpu_hours": reserved_gpu_hours,
            "prior_attempts_settled_gpu_hours": settled_before_gpu_hours,
            "package_cap_gpu_hours": limits["package_cap_gpu_hours"],
            "total_exp_cap_gpu_hours": limits["total_exp_cap_gpu_hours"],
            "exp_settled_gpu_hours": accounting["exp_settled_gpu_hours"],
            "exp_remaining_gpu_hours": accounting["exp_remaining_gpu_hours"],
            "decision": "PASS_COST_GATE_CONTINUE_BOTH_100_EPOCH_FITS" if proceed_full else "STOP_AFTER_SMOKE_CLOSE_RESERVATION",
            "full_100_epoch_fit_within_authorized_attempt_reservation": proceed_full}
        results["baseline_metrics_path"] = "baseline_metrics.json"
        results["baseline_metrics_sha256"] = sha256_file(attempt / "baseline_metrics.json")
        results["baseline_train_denominator"] = {"conditions": 419, "objects": 110,
                                                  "genes": 978, "normalizer_fit_split": "TRAIN only"}
        results["smoke_prediction_path"] = prediction_path.name
        results["smoke_predictions_sha256"] = sha256_file(prediction_path)
        results["smoke_checkpoints"] = {
            "XPert": {"path": "XPert_smoke_one_update_state.pt", "sha256": sha256_file(attempt / "XPert_smoke_one_update_state.pt"), "updates": 1},
            "TranSiGen": {"path": "TranSiGen_smoke_one_update_state.pt", "sha256": sha256_file(attempt / "TranSiGen_smoke_one_update_state.pt"), "updates": 1}}
        results["cpu_audited_features"] = feature_artifact_receipt
        results["resource"] = {"wsl_pid": process_id, "device_uuid": GPU_UUID,
                                "smoke_gpu_hours_actual": smoke_elapsed_seconds / 3600,
                                "attempt_reservation_gpu_hours": reserved_gpu_hours,
                                "device_snapshot_after_smoke": _device_snapshot()["selected_gpu"]}
        atomic_json(attempt / "fit_smoke_result.json", results)
        if not proceed_full:
            final_snapshot = _device_snapshot()["selected_gpu"]
            append_jsonl(ledger, {"event": "GPU_ATTEMPT_CLOSED_AFTER_COST_GATE", "at_utc": utc_now(),
                                  "wsl_pid": process_id, "device_uuid": GPU_UUID,
                                  "actual_gpu_hours": smoke_elapsed_seconds / 3600,
                                  "reserved_gpu_hours": reserved_gpu_hours,
                                  "unused_authorized_reservation_released_gpu_hours": max(0.0, reserved_gpu_hours - smoke_elapsed_seconds / 3600),
                                  "cumulative_conservative_charge_including_prior_attempts_gpu_hours": settled_before_gpu_hours + smoke_elapsed_seconds / 3600,
                                  "cost_gate_passed": False, "gpu_snapshot_at_close": final_snapshot})
            atomic_json(attempt / "attempt_progress.json", {"phase": "CLOSED_COST_GATE_FAILED",
                                                             "updated_at_utc": utc_now(), "wsl_pid": process_id})
            print(json.dumps({"stage": "GPU_ATTEMPT_CLOSED_AFTER_COST_GATE",
                              "pid": process_id, "actual_gpu_hours": smoke_elapsed_seconds / 3600,
                              "projected_with_20pct_margin_gpu_hours": projected_total_with_margin_hours,
                              "reserved_gpu_hours": reserved_gpu_hours, "fit_started": False,
                              "attempt": str(attempt)}, ensure_ascii=False), flush=True)
            return results

        # Recreate both source models from scratch for the fixed full schedules.
        full_start = time.perf_counter()
        append_jsonl(ledger, {"event": "FULL_FIT_STARTED", "at_utc": utc_now(),
                              "wsl_pid": process_id, "device_uuid": GPU_UUID,
                              "models": ["XPert", "TranSiGen"], "epochs_each": 100,
                              "smoke_gpu_hours_already_charged_within_reservation": smoke_elapsed_seconds / 3600,
                              "projected_total_with_margin_gpu_hours": projected_total_with_margin_hours})
        atomic_json(attempt / "attempt_progress.json", {"phase": "FULL_FITS_STARTED",
                                                         "updated_at_utc": utc_now(), "wsl_pid": process_id,
                                                         "completed_smoke_updates": {"XPert": 1, "TranSiGen": 1}})
        full_fit = _train_full_models(
            torch, attempt, keys, values, train_mask, dev_mask, train_keys, dev_keys,
            gene_variance, baseline_receipt, unimol, xpert_module, xpert_config,
            feature_by_key, device,
            {"XPert": xpert_update_seconds, "TranSiGen": transigen_update_seconds},
            device_budget_hours=reserved_gpu_hours, elapsed_gpu_origin=start_gpu)
        full_fit["fit_gate_baseline_denominator"] = {
            "strongest_baseline_candidates": ["zero_change", "per_drug_mean_delta"],
            "baseline_metrics_sha256": results["baseline_metrics_sha256"],
            "TRAIN_conditions": 419, "TRAIN_objects": 110, "normalizer_fit_on": "TRAIN only"}
        atomic_json(attempt / "full_fit_result.json", full_fit)
        full_start_seconds = time.perf_counter() - full_start
        total_elapsed_seconds = time.perf_counter() - start_gpu
        if total_elapsed_seconds / 3600 > reserved_gpu_hours:
            raise TimeoutError("attempt actual wall resource exceeded its ledger-derived GPU reservation")
        final_snapshot = _device_snapshot()["selected_gpu"]
        result_status = {"status": "COMPLETED", "completed_epochs": {"XPert": 100, "TranSiGen": 100},
                         "cost_gate": results["cost_gate"],
                         "full_fit_summary": full_fit["best_metrics"],
                         "actual_gpu_hours": total_elapsed_seconds / 3600,
                         "reserved_gpu_hours": reserved_gpu_hours,
                         "unused_reservation_released_gpu_hours": reserved_gpu_hours - total_elapsed_seconds / 3600,
                         "cumulative_conservative_settled_gpu_hours": settled_before_gpu_hours + total_elapsed_seconds / 3600,
                         "runner_sha256": manifest["current_runner_sha256"], "config_sha256": config_sha,
                         "process_id": process_id, "device_uuid": GPU_UUID,
                         "device_snapshot_at_close": final_snapshot,
                         "baseline_metrics_sha256": results["baseline_metrics_sha256"],
                         "full_keyed_predictions_sha256": full_fit["predictions_sha256"]}
        atomic_json(attempt / "attempt_close_receipt.json", result_status)
        append_jsonl(ledger, {"event": "GPU_ATTEMPT_COMPLETED", "at_utc": utc_now(),
                              "wsl_pid": process_id, "device_uuid": GPU_UUID,
                              "actual_gpu_hours": total_elapsed_seconds / 3600,
                              "reserved_gpu_hours": reserved_gpu_hours,
                              "unused_authorized_reservation_released_gpu_hours": reserved_gpu_hours - total_elapsed_seconds / 3600,
                              "cumulative_conservative_charge_including_prior_attempts_gpu_hours": settled_before_gpu_hours + total_elapsed_seconds / 3600,
                              "cost_gate_projected_gpu_hours": projected_total_with_margin_hours,
                              "full_epochs_completed": {"XPert": 100, "TranSiGen": 100},
                              "full_fit_wall_seconds_after_smoke": full_start_seconds,
                              "gpu_snapshot_at_close": final_snapshot,
                              "close_receipt": "attempt_close_receipt.json"})
        atomic_json(attempt / "attempt_progress.json", {"phase": "CLOSED_COMPLETED",
                                                         "updated_at_utc": utc_now(), "wsl_pid": process_id,
                                                         "actual_gpu_hours": total_elapsed_seconds / 3600})
        print(json.dumps({"stage": "GPU_ATTEMPT_COMPLETED", "pid": process_id,
                          "actual_gpu_hours": total_elapsed_seconds / 3600,
                          "full_epochs": {"XPert": 100, "TranSiGen": 100},
                          "fit_attempt": str(attempt)}, ensure_ascii=False), flush=True)
        return {"smoke": results, "full_fit": full_fit, "close_receipt": result_status}
    finally:
        try:
            del unimol
            torch.cuda.empty_cache()
        except Exception:
            pass


def _run_exp055_cost_calibration(cfg: dict[str, Any], attempt: Path, ledger: Path,
                                process_id: int, reservation_hours: float,
                                accounting: dict[str, Any], torch: Any, model: Any,
                                train_view: Any, original_train: Any,
                                source_receipt: dict[str, Any], values: dict[str, np.ndarray],
                                keys: pd.DataFrame, train_mask: np.ndarray, dev_mask: np.ndarray,
                                train_keys: pd.DataFrame, dev_keys: pd.DataFrame,
                                gene_variance: np.ndarray, baseline: dict[str, Any],
                                unimol: np.ndarray, xpert_config: dict[str, Any], device: Any,
                                start_gpu: float, config_sha: str,
                                initialization_seconds: float) -> dict[str, Any]:
    """Measure one full 419-key XPert warmup and one raw-stage epoch, then close."""
    if EXPERIMENT_ID != "EXP-055" or attempt.name != "fit_attempt_006":
        raise ValueError("full-data cost calibration is reserved for EXP-055 fit_attempt_006")
    hard_cap_seconds = reservation_hours * 3600.0
    stop_before_seconds = hard_cap_seconds - 15.0

    def budget_check() -> None:
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_gpu
        if elapsed >= stop_before_seconds:
            raise TimeoutError(f"EXP-055 calibration reached its hard-stop guard at {elapsed:.1f}s / {hard_cap_seconds:.1f}s")

    train_positions = np.flatnonzero(train_mask)
    dev_positions = np.flatnonzero(dev_mask)
    train_tensors = _xpert_tensors(torch, values, keys, train_positions, unimol,
                                   xpert_config, training_targets=True)
    sizes = [int(x.shape[0]) for x in train_tensors]
    if any(size != 419 for size in sizes):
        raise ValueError(f"XPert TRAIN tensors are not full 419-key arrays: {sizes}")
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(*train_tensors)
    loader_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True,
                              generator=loader_generator, num_workers=0, drop_last=False)
    batch_sizes = [int(batch[0].shape[0]) for batch in train_loader]
    if batch_sizes != [128, 128, 128, 35]:
        raise ValueError(f"XPert calibration must use four complete TRAIN batches [128, 128, 128, 35], got {batch_sizes}")
    # Recreate after the batch-shape inspection so the measured epoch starts with a fresh seeded shuffle.
    loader_generator.manual_seed(SEED)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True,
                              generator=loader_generator, num_workers=0, drop_last=False)
    raw_optimizer = torch.optim.Adam(model.parameters(), lr=0.004, weight_decay=1e-5)
    optimizer = ProbeOptimizer(raw_optimizer, model, torch, progress_check=budget_check)
    class Args:
        pretrained_model = False

    stage_rows: list[dict[str, Any]] = []
    stage_predictions: dict[str, np.ndarray] = {}
    stage_seconds: dict[str, float] = {}
    for stage_name, official_epoch, learning_rate in (
            ("warmup", 0, 0.004), ("raw", 70, 0.002)):
        budget_check()
        for group in raw_optimizer.param_groups:
            group["lr"] = learning_rate
        before_parameters = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer.gradient_norms.clear()
        train_start = time.perf_counter()
        epoch_losses = original_train(train_view, optimizer, train_loader, Args(), xpert_config,
                                      scaler=None, epoch=official_epoch)
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - train_start
        budget_check()
        train_eval_start = time.perf_counter()
        train_prediction = _xpert_predict(torch, model, values, keys, train_positions, unimol,
                                          xpert_config, batch_size=128, progress_check=budget_check)
        torch.cuda.synchronize()
        train_eval_seconds = time.perf_counter() - train_eval_start
        dev_eval_start = time.perf_counter()
        dev_prediction = _xpert_predict(torch, model, values, keys, dev_positions, unimol,
                                        xpert_config, batch_size=128, progress_check=budget_check)
        torch.cuda.synchronize()
        dev_eval_seconds = time.perf_counter() - dev_eval_start
        budget_check()
        grad_values = list(optimizer.gradient_norms)
        params_finite = all(bool(torch.isfinite(parameter).all())
                            for parameter in model.parameters())
        if (len(grad_values) != 4 or not all(math.isfinite(float(value)) for value in epoch_losses)
                or not all(math.isfinite(float(value)) for value in grad_values)
                or not params_finite or not np.isfinite(train_prediction).all()
                or not np.isfinite(dev_prediction).all()):
            raise FloatingPointError(f"XPert {stage_name} calibration produced non-finite loss, gradient, parameter, or prediction")
        param_update_l2 = _norm_difference(torch, before_parameters, model)
        if not math.isfinite(param_update_l2):
            raise FloatingPointError(f"XPert {stage_name} parameter update is non-finite")
        torch.cuda.synchronize()
        total_stage_seconds = train_seconds + train_eval_seconds + dev_eval_seconds
        stage_seconds[stage_name] = total_stage_seconds
        train_metrics = keyed_metrics(train_keys, values["delta"][train_mask], train_prediction, gene_variance)
        dev_metrics = keyed_metrics(dev_keys, values["delta"][dev_mask], dev_prediction, gene_variance)
        row = {"stage": stage_name, "official_epoch_index": official_epoch,
               "official_loss_rule": "batch_weighted_loss" if official_epoch < 70 else "weighted_loss raw sum",
               "learning_rate": learning_rate, "batch_size": 128, "train_batch_sizes": batch_sizes,
               "train_conditions": 419, "train_objects": int(train_keys.source_object_id.nunique()),
               "loss_terms_official_return_order": ["weighted_total_per_condition", "trt_reconstruction_rmse",
                                                     "control_reconstruction_rmse", "delta_rmse", "delta_pearson_loss"],
               "loss_terms": [safe_number(float(value)) for value in epoch_losses],
               "gradient_norms_by_optimizer_step": [safe_number(float(value)) for value in grad_values],
               "gradient_norm_max": max(grad_values), "parameter_update_l2": param_update_l2,
               "train_seconds": train_seconds, "train_evaluation_seconds": train_eval_seconds,
               "inner_dev_evaluation_seconds": dev_eval_seconds,
               "stage_seconds_including_full_train_and_dev_evaluation": total_stage_seconds,
               "train_metrics": train_metrics, "inner_dev_metrics": dev_metrics,
               "train_prediction_max_abs": float(np.max(np.abs(train_prediction))),
               "inner_dev_prediction_max_abs": float(np.max(np.abs(dev_prediction))),
               "finite": True}
        stage_rows.append(row)
        stage_predictions[f"{stage_name}_train_delta"] = train_prediction
        stage_predictions[f"{stage_name}_inner_dev_delta"] = dev_prediction
        atomic_json(attempt / f"XPert_cost_calibration_{stage_name}_epoch.json", row)
        append_jsonl(attempt / "XPert_cost_calibration_curve.jsonl", row)
        atomic_json(attempt / "attempt_progress.json", {
            "phase": f"XPert_COST_CALIBRATION_{stage_name.upper()}_EPOCH_COMPLETED",
            "updated_at_utc": utc_now(), "wsl_pid": process_id,
            "elapsed_gpu_wall_seconds": time.perf_counter() - start_gpu,
            "official_epoch_index": official_epoch, "train_batch_sizes": batch_sizes})
        del before_parameters, train_prediction, dev_prediction
        torch.cuda.empty_cache()

    final_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    state_path = attempt / "XPert_cost_calibration_final_state.pt"
    torch.save({"state_dict": final_state, "model": "upstream XPertNet",
                "seed": SEED, "completed_official_epoch_indices": [0, 70],
                "official_train_function_sha256": source_receipt["train_function_source_sha256"],
                "config_sha256": sha256_file(XPERT_ROOT / "configs/config_l1000.yaml")}, state_path)
    state_reload = torch.load(state_path, map_location="cpu", weights_only=False)
    if (state_reload.get("model") != "upstream XPertNet"
            or set(state_reload["state_dict"]) != set(final_state)
            or any(not torch.equal(state_reload["state_dict"][key], final_state[key]) for key in final_state)):
        raise ValueError("XPert calibration checkpoint state readback mismatch")
    predictions_path = attempt / "XPert_cost_calibration_predictions.npz"
    expected_predictions = {
        "train_condition_key": train_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
        "inner_dev_condition_key": dev_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
        "train_source_object_id": train_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
        "inner_dev_source_object_id": dev_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
        "genes": values["genes"].astype(str), "train_target_delta": values["delta"][train_mask],
        "inner_dev_target_delta": values["delta"][dev_mask], **stage_predictions}
    np.savez_compressed(predictions_path, **expected_predictions)
    with np.load(predictions_path, allow_pickle=False) as stored:
        readback_ok = all(name in stored.files and np.array_equal(stored[name], expected)
                          for name, expected in expected_predictions.items())
        readback_keys = {"train_conditions": len(stored["train_condition_key"]),
                         "inner_dev_conditions": len(stored["inner_dev_condition_key"]),
                         "genes": len(stored["genes"])}
    if not readback_ok or readback_keys != {"train_conditions": 419, "inner_dev_conditions": 93, "genes": 978}:
        raise ValueError("XPert calibration keyed prediction NPZ readback failed")

    # EXP-051 attempt_007 supplies an observed original TranSiGen TRAIN epoch cost;
    # multiply it by 2.5 to cover full TRAIN+inner-dev scoring and overhead.
    transigen_attempt_dir = (ROOT / "artifacts/experiments/EXP-051"
        / "20261006_exp051_representative_compare_v1/fit_attempt_007")
    transigen_source_progress = transigen_attempt_dir / "attempt_close_receipt.json"
    if transigen_source_progress.is_file():
        transigen_evidence = json.loads(transigen_source_progress.read_text(encoding="utf-8"))
        prior_transigen_hours = float(transigen_evidence.get(
            "actual_gpu_hours_upper_bound", transigen_evidence.get("actual_gpu_hours", float("nan"))))
        prior_transigen_epochs = int(transigen_evidence.get(
            "completed_train_epochs", transigen_evidence.get("completed_epochs", 0)))
    else:
        transigen_source_progress = transigen_attempt_dir / "failure_close_receipt.json"
        if not transigen_source_progress.is_file():
            raise FileNotFoundError("prior TranSiGen attempt has neither attempt_close_receipt.json nor failure_close_receipt.json")
        transigen_evidence = json.loads(transigen_source_progress.read_text(encoding="utf-8"))
        prior_transigen_hours = float(transigen_evidence.get(
            "charged_gpu_hours_conservative_upper_bound", float("nan")))
        prior_transigen_epochs = int(transigen_evidence.get("last_progress", {}).get("completed_train_epochs", 0))
    if (not math.isfinite(prior_transigen_hours) or prior_transigen_hours <= 0
            or prior_transigen_epochs <= 0):
        raise ValueError(f"prior TranSiGen timing receipt lacks finite charged hours/completed epochs: {transigen_source_progress}")
    transigen_projected_with_margin = (prior_transigen_hours / prior_transigen_epochs * 2.5 * 300 * 1.05 * 1.20)
    xp_raw_seconds = 70 * stage_seconds["warmup"] + 2430 * stage_seconds["raw"]
    xp_projected_with_margin = xp_raw_seconds * 1.05 * 1.20 / 3600.0
    actual_before_close = time.perf_counter() - start_gpu
    if actual_before_close > hard_cap_seconds:
        raise TimeoutError(f"XPert cost calibration exceeded its {hard_cap_seconds:.1f}s GPU cap")
    calibration_hours = actual_before_close / 3600.0
    package_projected = (float(accounting["package_settled_gpu_hours"]) + calibration_hours
                         + xp_projected_with_margin + transigen_projected_with_margin)
    exp_projected = (float(accounting["exp_settled_gpu_hours"]) + calibration_hours
                     + xp_projected_with_margin + transigen_projected_with_margin)
    package_cap = float(resource_limits(cfg)["package_cap_gpu_hours"])
    exp_cap = float(resource_limits(cfg)["total_exp_cap_gpu_hours"])
    first_phase_cap = float(resource_limits(cfg)["first_phase_gpu_cap_hours"])
    total_projected_fits = package_projected <= package_cap and exp_projected <= exp_cap
    reallocation_used = any(path.is_file() for path in RUN_DIR.glob("fit_attempt_*/one_time_package_reallocation_receipt.json"))
    reallocation_authorized = bool(cfg.get("authorization", {}).get("one_internal_reallocation_within_package", False))
    reallocation_needed = package_projected > first_phase_cap and total_projected_fits
    if reallocation_needed and reallocation_authorized and not reallocation_used:
        atomic_json(attempt / "one_time_package_reallocation_receipt.json", {
            "status": "USED_ONCE_WITHIN_USER_AUTHORIZED_10_GPUH_PACKAGE",
            "reason": "Measured 2500-epoch XPert maximum plus TranSiGen 300-epoch projection exceeds 5h but remains within the existing 10h package",
            "created_at_utc": utc_now(), "run_id": RUN_ID, "attempt": attempt.name,
            "projected_package_total_gpu_hours": package_projected,
            "package_cap_gpu_hours": package_cap, "first_phase_cap_gpu_hours": first_phase_cap})
        reallocation_used = True
    gate = ("PASS_WITHIN_5_GPUH_FIRST_PHASE" if total_projected_fits and package_projected <= first_phase_cap
            else "PASS_WITHIN_10_GPUH_AFTER_ONE_INTERNAL_REALLOCATION" if total_projected_fits and reallocation_used
            else "STOP_XPERT_FORMAL_OVER_10_GPUH; TRANSiGEN MAY_PROCEED_SEPARATELY_IF_WITHIN_REMAINING_PACKAGE"
            if not total_projected_fits else "STOP_FIRST_PHASE_REALLOCATION_UNAVAILABLE")
    result = {
        "status": "COST_CALIBRATION_COMPLETED_NO_MODEL_SELECTION", "attempt": attempt.name,
        "run_id": RUN_ID, "source_run_id": SOURCE_RUN_ID, "created_at_utc": utc_now(),
        "device_uuid": GPU_UUID, "reservation_gpu_hours": reservation_hours,
        "actual_gpu_wall_seconds_including_initialization_and_artifact_write": actual_before_close,
        "actual_gpu_hours": calibration_hours, "hard_cap_gpu_hours": reservation_hours,
        "guard_stop_seconds": stop_before_seconds, "initialization_seconds_before_epoch_calibration": initialization_seconds,
        "official_schedule": {"maximum_epochs": 2500, "warmup_epochs": 70,
                               "raw_epochs": 2430, "batch_size": 128, "patience": 50,
                               "selection": "val_loss4; patience is excluded from cost reduction",
                               "calibration_epochs": [{"official_index": 0, "objective": "batch_weighted_loss", "learning_rate": 0.004},
                                                       {"official_index": 70, "objective": "weighted_loss raw sum", "learning_rate": 0.002}]},
        "train_batch_sizes": batch_sizes, "epoch_measurements": stage_rows,
        "XPert_2500_epoch_projected_raw_gpu_hours": xp_raw_seconds / 3600.0,
        "XPert_2500_epoch_projected_gpu_hours_with_20_percent_margin_and_5_percent_bookkeeping": xp_projected_with_margin,
        "TranSiGen_projection_source": str(transigen_source_progress),
        "TranSiGen_projection_source_sha256": sha256_file(transigen_source_progress),
        "TranSiGen_projection_formula": "EXP-051 attempt_007 observed seconds/train epoch * 2.5 for full TRAIN+inner-dev evaluation and overhead * 300 epochs * 1.05 bookkeeping * 1.20 margin",
        "TranSiGen_300_epoch_projected_gpu_hours_with_20_percent_margin": transigen_projected_with_margin,
        "settled_package_before_calibration_gpu_hours": accounting["package_settled_gpu_hours"],
        "settled_global_before_calibration_gpu_hours": accounting["exp_settled_gpu_hours"],
        "package_cumulative_projected_gpu_hours": package_projected,
        "package_cap_gpu_hours": package_cap, "global_cumulative_projected_gpu_hours": exp_projected,
        "global_cap_gpu_hours": exp_cap, "first_phase_cap_gpu_hours": first_phase_cap,
        "over_10_gpu_hour_package_by": max(0.0, package_projected - package_cap),
        "one_time_internal_reallocation_authorized": reallocation_authorized,
        "one_time_internal_reallocation_used": reallocation_used,
        "cost_gate_decision": gate,
        "baseline_metrics_sha256": sha256_file(attempt / "baseline_metrics.json"),
        "checkpoint_path": state_path.name, "checkpoint_sha256": sha256_file(state_path),
        "checkpoint_readback": "PASS_EXACT_STATE_DICT",
        "predictions_path": predictions_path.name, "predictions_sha256": sha256_file(predictions_path),
        "prediction_readback": {"status": "PASS_EXACT_KEYS_VALUES_AND_SHAPE", **readback_keys},
        "official_train_function_sha256": source_receipt["train_function_source_sha256"],
        "config_sha256": config_sha, "runner_sha256": sha256_file(Path(__file__).resolve()),
        "boundary": "This is a real full-TRAIN cost calibration, not a fitted model comparison. Metrics are recorded descriptively and do not choose a stage or checkpoint."}
    atomic_json(attempt / "cost_calibration_result.json", result)
    atomic_json(attempt / "cost_calibration_readback_receipt.json", {
        "status": "PASS", "checkpoint_sha256": sha256_file(state_path),
        "predictions_sha256": sha256_file(predictions_path), "prediction_shapes": readback_keys,
        "train_keys_sha256": sha256_text("\n".join(expected_predictions["train_condition_key"].tolist()) + "\n"),
        "inner_dev_keys_sha256": sha256_text("\n".join(expected_predictions["inner_dev_condition_key"].tolist()) + "\n")})
    actual_close_hours = (time.perf_counter() - start_gpu) / 3600.0
    if actual_close_hours > reservation_hours:
        raise TimeoutError("EXP-055 calibration close artifacts exceeded the 0.05 GPU-hour reservation")
    close = {"status": "COST_CALIBRATION_COMPLETED_NO_MODEL_SELECTION", "actual_gpu_hours": actual_close_hours,
             "reserved_gpu_hours": reservation_hours, "cost_gate_decision": gate,
             "package_cumulative_projected_gpu_hours": package_projected,
             "device_uuid": GPU_UUID, "wsl_pid": process_id,
             "checkpoint_sha256": sha256_file(state_path), "predictions_sha256": sha256_file(predictions_path)}
    atomic_json(attempt / "attempt_close_receipt.json", close)
    append_jsonl(ledger, {"event": "GPU_ATTEMPT_COMPLETED", "at_utc": utc_now(),
                          "wsl_pid": process_id, "device_uuid": GPU_UUID,
                          "actual_gpu_hours": actual_close_hours, "reserved_gpu_hours": reservation_hours,
                          "cost_gate_decision": gate, "cost_calibration_result": "cost_calibration_result.json",
                          "close_receipt": "attempt_close_receipt.json"})
    atomic_json(attempt / "attempt_progress.json", {"phase": "CLOSED_COST_CALIBRATION",
                                                      "updated_at_utc": utc_now(), "wsl_pid": process_id,
                                                      "actual_gpu_hours": actual_close_hours,
                                                      "cost_gate_decision": gate})
    print(json.dumps({"stage": "EXP055_COST_CALIBRATION_CLOSED", "attempt": attempt.name,
                      "actual_gpu_hours": actual_close_hours, "XPert_full_schedule_projection_gpu_hours": xp_projected_with_margin,
                      "TranSiGen_300_projection_gpu_hours": transigen_projected_with_margin,
                      "package_projected_gpu_hours": package_projected, "decision": gate}, ensure_ascii=False), flush=True)
    return result


def _atomic_torch_checkpoint(torch: Any, path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _torch_tree_exact(torch: Any, left: Any, right: Any) -> bool:
    """Compare checkpoint trees exactly, including optimizer tensor state."""
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            return False
        if (left.shape != right.shape or left.dtype != right.dtype
                or left.layout != right.layout):
            return False
        # Checkpoint payloads are often retained on CPU while readback maps to
        # the active accelerator.  torch.equal requires matching devices, so
        # normalize both operands before the exact value comparison.
        return torch.equal(left.detach().to(device="cpu"), right.detach().to(device="cpu"))
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(_torch_tree_exact(torch, left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_torch_tree_exact(torch, a, b) for a, b in zip(left, right)))
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _train_exp055_transigen_only(torch: Any, attempt: Path, keys: pd.DataFrame,
                                 values: dict[str, np.ndarray], train_mask: np.ndarray,
                                 dev_mask: np.ndarray, train_keys: pd.DataFrame,
                                 dev_keys: pd.DataFrame, gene_variance: np.ndarray,
                                 baseline_metrics: dict[str, Any], feature_by_key: dict[str, np.ndarray],
                                 device: Any, reservation_hours: float,
                                 elapsed_gpu_origin: float, initial_progress: Any) -> dict[str, Any]:
    """Run the registered fixed 300-epoch original TranSiGen route without selection."""
    from torch.utils.data import DataLoader, TensorDataset

    train_positions, dev_positions = np.flatnonzero(train_mask), np.flatnonzero(dev_mask)
    train_y, dev_y = values["delta"][train_mask], values["delta"][dev_mask]
    train_features = _full_key_features(keys, train_positions, feature_by_key)
    dev_features = _full_key_features(keys, dev_positions, feature_by_key)
    train_dataset = TensorDataset(
        torch.as_tensor(values["X_ctl"][train_mask], dtype=torch.float32),
        torch.as_tensor(values["X"][train_mask], dtype=torch.float32),
        torch.as_tensor(train_features, dtype=torch.float32))
    loader_generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False,
                        num_workers=0, generator=loader_generator)
    batch_sizes = [int(batch[0].shape[0]) for batch in loader]
    if batch_sizes != [64, 64, 64, 64, 64, 64, 35]:
        raise ValueError(f"TranSiGen full TRAIN loader must have six 64-row batches and a 35-row tail, got {batch_sizes}")
    loader_generator.manual_seed(SEED)
    loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False,
                        num_workers=0, generator=loader_generator)
    model, model_module = _load_transigen(torch, device, SEED, features_dim=2048)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    progress = dict(initial_progress)
    stop_before_seconds = reservation_hours * 3600.0 - 60.0

    def budget_check() -> None:
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - elapsed_gpu_origin
        progress["elapsed_gpu_hours_upper_bound"] = elapsed / 3600.0
        if elapsed >= stop_before_seconds:
            raise TimeoutError(f"TranSiGen reached the package budget guard at {elapsed:.1f}s")

    curve = attempt / "TranSiGen_training_curve.jsonl"
    if curve.exists():
        raise FileExistsError(f"refusing to overwrite TranSiGen training curve: {curve}")
    loss_names = ("raw_total_loss", "mse_x1", "mse_x2", "mse_pert",
                  "kld_x1", "kld_x2", "kld_pert")
    last_finite_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    _atomic_torch_checkpoint(torch, attempt / "TranSiGen_last_finite.pt", {
        "state_dict": last_finite_state, "model": "upstream TranSiGen", "epoch": 0,
        "seed": SEED, "status": "INITIAL_FINITE_CHECKPOINT"})
    last_train_prediction = last_dev_prediction = None
    last_train_metrics = last_dev_metrics = None
    epoch_seconds: list[float] = []

    def save_checkpoint(name: str, epoch: int, state: dict[str, Any]) -> None:
        _atomic_torch_checkpoint(torch, attempt / name, {
            "state_dict": {key: tensor.detach().cpu().clone() for key, tensor in state.items()},
            "model": "upstream TranSiGen", "epoch": epoch, "seed": SEED,
            "objective": "raw_sum", "gradient_clipping": False,
            "learning_rate": 1e-3, "weight_decay": 1e-5})

    try:
        for epoch in range(300):
            budget_check()
            epoch_start = time.perf_counter()
            before_parameters = [parameter.detach().clone() for parameter in model.parameters()]
            model.train()
            loss_sums = np.zeros(7, dtype=np.float64)
            gradient_norms: list[float] = []
            seen_conditions = 0
            for x1, x2, features in loader:
                budget_check()
                transigen_batch = (x1.to(device), x2.to(device), features.to(device))
                model.zero_grad(set_to_none=True)
                _, _, _, losses = _transigen_forward_batch(torch, model, transigen_batch, device)
                if not all(bool(torch.isfinite(loss).all()) for loss in losses):
                    raise FloatingPointError(f"TranSiGen raw-sum training loss is non-finite at epoch {epoch + 1}")
                losses[0].backward()
                gradient_norm = _grad_global_norm(torch, model)
                if not math.isfinite(gradient_norm):
                    raise FloatingPointError(f"TranSiGen gradient norm is non-finite at epoch {epoch + 1}")
                gradient_norms.append(gradient_norm)
                optimizer.step()
                loss_sums += np.asarray([float(value.detach().cpu()) for value in losses], dtype=np.float64)
                seen_conditions += int(len(x1))
                if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                    raise FloatingPointError(f"TranSiGen parameter became non-finite at epoch {epoch + 1}")
            if seen_conditions != 419 or len(gradient_norms) != 7:
                raise ValueError(f"TranSiGen epoch did not process all 419 keys in seven updates: n={seen_conditions}, updates={len(gradient_norms)}")
            torch.cuda.synchronize()
            train_prediction, train_eval_loss = _transigen_predict_and_losses(
                torch, model, values["X_ctl"][train_mask], values["X"][train_mask],
                train_features, 64, device, SEED + 2 * epoch + 1,
                allow_nonfinite_auxiliary=True, progress_check=budget_check)
            dev_prediction, dev_eval_loss = _transigen_predict_and_losses(
                torch, model, values["X_ctl"][dev_mask], values["X"][dev_mask],
                dev_features, 64, device, SEED + 2 * epoch + 2,
                allow_nonfinite_auxiliary=True, progress_check=budget_check)
            if not np.isfinite(train_prediction).all() or not np.isfinite(dev_prediction).all():
                raise FloatingPointError(f"TranSiGen main-delta prediction is non-finite at epoch {epoch + 1}")
            train_metric = keyed_metrics(train_keys, train_y, train_prediction, gene_variance)
            dev_metric = keyed_metrics(dev_keys, dev_y, dev_prediction, gene_variance)
            parameter_norm = _param_global_norm(torch, model)
            parameter_update_l2 = _norm_difference(torch, before_parameters, model)
            if not math.isfinite(parameter_norm) or not math.isfinite(parameter_update_l2):
                raise FloatingPointError(f"TranSiGen parameter diagnostics are non-finite at epoch {epoch + 1}")
            duration = time.perf_counter() - epoch_start
            epoch_seconds.append(duration)
            row = {"epoch": epoch + 1, "epoch_index_zero_based": epoch,
                   "objective": "raw_sum", "gradient_clipping": False,
                   "learning_rate": 1e-3, "weight_decay": 1e-5,
                   "batch_size": 64, "train_shuffle": True,
                   "train_conditions": seen_conditions, "train_batch_sizes": batch_sizes,
                   "train_loss_terms_mean_per_condition": {
                       name: safe_number(float(value / seen_conditions))
                       for name, value in zip(loss_names, loss_sums)},
                   "train_evaluation_auxiliary_losses": train_eval_loss,
                   "inner_dev_evaluation_auxiliary_losses": dev_eval_loss,
                   "gradient_norm_mean": safe_number(float(np.mean(gradient_norms))),
                   "gradient_norm_max": safe_number(float(np.max(gradient_norms))),
                   "gradient_norms_by_update": [safe_number(float(value)) for value in gradient_norms],
                   "gradient_norm_n_updates": len(gradient_norms),
                   "parameter_global_l2_norm": safe_number(parameter_norm),
                   "parameter_update_l2": safe_number(parameter_update_l2),
                   "train_metrics": train_metric, "inner_dev_metrics": dev_metric,
                   "train_prediction_max_abs": float(np.max(np.abs(train_prediction))),
                   "inner_dev_prediction_max_abs": float(np.max(np.abs(dev_prediction))),
                   "epoch_seconds": duration, "finite_training_and_main_delta": True}
            append_jsonl(curve, row)
            last_train_prediction, last_dev_prediction = train_prediction.copy(), dev_prediction.copy()
            last_train_metrics, last_dev_metrics = train_metric, dev_metric
            last_finite_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            save_checkpoint("TranSiGen_last_finite.pt", epoch + 1, last_finite_state)
            if epoch + 1 in (100, 300):
                name = f"TranSiGen_epoch_{epoch + 1}.pt"
                save_checkpoint(name, epoch + 1, last_finite_state)
                if epoch + 1 == 300:
                    save_checkpoint("TranSiGen_final_epoch_300.pt", epoch + 1, last_finite_state)
                snapshot_path = attempt / f"TranSiGen_epoch_{epoch + 1}_keyed_predictions.npz"
                np.savez_compressed(snapshot_path,
                    train_condition_key=train_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
                    inner_dev_condition_key=dev_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
                    train_source_object_id=train_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
                    inner_dev_source_object_id=dev_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
                    genes=values["genes"].astype(str), train_target_delta=train_y,
                    inner_dev_target_delta=dev_y, train_delta=train_prediction,
                    inner_dev_delta=dev_prediction)
                with np.load(snapshot_path, allow_pickle=False) as saved:
                    if (not np.array_equal(saved["train_condition_key"], train_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_))
                            or not np.array_equal(saved["inner_dev_condition_key"], dev_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_))
                            or not np.array_equal(saved["train_delta"], train_prediction)
                            or not np.array_equal(saved["inner_dev_delta"], dev_prediction)):
                        raise ValueError(f"TranSiGen epoch {epoch + 1} keyed prediction readback failed")
            elapsed = time.perf_counter() - elapsed_gpu_origin
            progress.update({"model": "TranSiGen", "completed_epochs": epoch + 1,
                "target_epochs": 300, "updated_at_utc": utc_now(),
                "elapsed_gpu_hours_upper_bound": elapsed / 3600.0,
                "last_train_metrics": train_metric, "last_inner_dev_metrics": dev_metric})
            atomic_json(attempt / "attempt_progress.json", progress)
            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 in (100, 300):
                print(json.dumps({"stage": "EXP055_TRANSGEN_PROGRESS", "epoch": epoch + 1,
                    "train_nmse": train_metric["object_macro_nmse"],
                    "inner_dev_nmse": dev_metric["object_macro_nmse"],
                    "elapsed_gpu_hours": elapsed / 3600.0}, ensure_ascii=False), flush=True)
            del before_parameters, train_prediction, dev_prediction
            torch.cuda.empty_cache()
    except BaseException as error:
        save_checkpoint("TranSiGen_last_finite.pt", int(progress.get("completed_epochs", 0)), last_finite_state)
        atomic_json(attempt / "model_error.json", {
            "model": "TranSiGen", "status": "STOPPED_ON_NONFINITE_OR_RUNTIME_ERROR",
            "exception_type": type(error).__name__, "exception_message": str(error),
            "last_completed_finite_epoch": int(progress.get("completed_epochs", 0)),
            "last_finite_checkpoint": "TranSiGen_last_finite.pt",
            "last_finite_checkpoint_sha256": sha256_file(attempt / "TranSiGen_last_finite.pt"),
            "updated_at_utc": utc_now()})
        raise

    if last_train_prediction is None or last_dev_prediction is None:
        raise RuntimeError("TranSiGen fixed schedule completed without keyed predictions")
    final_summary = {
        "status": "COMPLETED_TRANSIGEN_FIXED_300_EPOCH_SCHEDULE",
        "epochs_completed": 300, "selected_epoch": None,
        "selection": "none; all 300 epochs executed and checkpointed at epochs 100/300/final",
        "train_metrics": last_train_metrics, "inner_dev_metrics": last_dev_metrics,
        "fit_gate": _fit_gate(last_train_metrics, baseline_metrics),
        "checkpoint_paths": ["TranSiGen_epoch_100.pt", "TranSiGen_epoch_300.pt",
                             "TranSiGen_final_epoch_300.pt", "TranSiGen_last_finite.pt"],
        "checkpoint_sha256": {name: sha256_file(attempt / name) for name in
            ("TranSiGen_epoch_100.pt", "TranSiGen_epoch_300.pt",
             "TranSiGen_final_epoch_300.pt", "TranSiGen_last_finite.pt")},
        "mean_epoch_seconds": float(np.mean(epoch_seconds)),
        "main_delta_predictions_finite": True,
        "baseline_metrics_sha256": sha256_file(attempt / "baseline_metrics.json")}
    final_prediction_path = attempt / "TranSiGen_final_keyed_predictions.npz"
    np.savez_compressed(final_prediction_path,
        train_condition_key=train_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
        inner_dev_condition_key=dev_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
        train_source_object_id=train_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
        inner_dev_source_object_id=dev_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
        train_full_inchikey=train_keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
        inner_dev_full_inchikey=dev_keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
        genes=values["genes"].astype(str), train_target_delta=train_y,
        inner_dev_target_delta=dev_y, train_delta=last_train_prediction,
        inner_dev_delta=last_dev_prediction)
    with np.load(final_prediction_path, allow_pickle=False) as saved:
        expected = {"train_condition_key": train_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
                    "inner_dev_condition_key": dev_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
                    "train_source_object_id": train_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
                    "inner_dev_source_object_id": dev_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
                    "genes": values["genes"].astype(str), "train_target_delta": train_y,
                    "inner_dev_target_delta": dev_y, "train_delta": last_train_prediction,
                    "inner_dev_delta": last_dev_prediction}
        if any(name not in saved.files or not np.array_equal(saved[name], value)
               for name, value in expected.items()):
            raise ValueError("TranSiGen final keyed prediction readback failed")
    final_summary["predictions_path"] = final_prediction_path.name
    final_summary["predictions_sha256"] = sha256_file(final_prediction_path)
    final_summary["prediction_readback"] = "PASS_EXACT_KEYED_VALUES"
    atomic_json(attempt / "transigen_full_fit_result.json", final_summary)
    del model, model_module, optimizer, loader, train_dataset
    torch.cuda.empty_cache()
    return final_summary


def _load_exp055_trainpath_train_inputs(source_receipts: dict[str, Path],
                                       keys_path: Path, feature_root: Path
                                       ) -> tuple[pd.DataFrame, dict[str, np.ndarray],
                                                  dict[str, np.ndarray], dict[str, Any]]:
    """Load only the frozen TRAIN rows from source H5AD for the recovery attempt.

    This deliberately does not open ``frozen_train_inner_values.npz``: that archive
    contains numeric inner-development rows alongside TRAIN rows. The response-blind
    key table is safe to use for row selection and object/identity validation.
    """
    freeze_path = source_receipts["freeze"]
    axis_path = source_receipts["axis"]
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    axis = json.loads(axis_path.read_text(encoding="utf-8"))
    if (freeze.get("run_id") != SOURCE_RUN_ID
            or freeze.get("stage") != "RESPONSE_BLIND_SELECTION_FROZEN"
            or axis.get("run_id") != SOURCE_RUN_ID
            or axis.get("status") != "PASS_EXACT_978_AXIS"
            or axis.get("mapping_rows") != 978):
        raise RuntimeError("EXP-051 response-blind freeze or 978-gene axis receipt failed")
    if not keys_path.is_file() or sha256_file(keys_path) != freeze.get("file_sha256", {}).get("condition_keys"):
        raise RuntimeError("frozen response-blind condition table SHA differs from its freeze receipt")
    if not H5AD.is_file() or sha256_file(H5AD) != freeze.get("file_sha256", {}).get("h5ad"):
        raise RuntimeError("source H5AD SHA differs from the response-blind freeze receipt")
    mapping_path = SOURCE_RUN_DIR / "gene_axis_mapping.tsv"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"frozen 978-gene mapping is missing: {mapping_path}")
    mapping_file_sha256 = sha256_file(mapping_path)
    expected_mapping_file_sha256 = axis.get("files_sha256", {}).get("gene_axis_mapping.tsv")
    if not expected_mapping_file_sha256 or mapping_file_sha256 != expected_mapping_file_sha256:
        raise RuntimeError("gene-axis mapping file-byte SHA differs from files_sha256 receipt field")
    mapping_table = pd.read_csv(mapping_path, sep="\t", dtype=str, keep_default_na=False)
    required_mapping_columns = {"axis_index", "h5ad_var_name", "h5ad_var_annotation"}
    if len(mapping_table) != 978 or not required_mapping_columns.issubset(mapping_table.columns):
        raise ValueError("frozen gene-axis mapping lacks the 978 axis/name/annotation rows")
    mapping_table["axis_index"] = pd.to_numeric(
        mapping_table["axis_index"], errors="raise").astype(np.int64)
    if (mapping_table.axis_index.duplicated().any()
            or set(mapping_table.axis_index.tolist()) != set(range(978))):
        raise ValueError("frozen gene-axis mapping axis_index must be a unique 0..977 sequence")
    mapping_table = mapping_table.sort_values("axis_index", kind="mergesort").reset_index(drop=True)
    semantic_mapping_sha256 = sha256_text(
        "\n".join(mapping_table["h5ad_var_annotation"].astype(str).tolist()) + "\n")
    if semantic_mapping_sha256 != axis.get("mapping_sha256"):
        raise RuntimeError("ordered gene-axis annotation SHA differs from mapping_sha256 receipt field")

    keys = pd.read_csv(keys_path, sep="\t", dtype=str, keep_default_na=False)
    required = {"selection_rank", "source_row", "obs_name", "source_object_id",
                "full_inchikey", "split_role", "full_condition_key"}
    missing = sorted(required - set(keys.columns))
    if missing:
        raise ValueError(f"response-blind table lacks TRAIN recovery fields: {missing}")
    if (len(keys) != 512 or keys.full_condition_key.astype(str).duplicated().any()
            or keys.selection_rank.astype(str).duplicated().any()):
        raise ValueError("frozen 512 condition-key table is not unique")
    role_set = set(keys.split_role.astype(str))
    if role_set != {"train", "inner_dev"}:
        raise ValueError(f"unexpected response-blind split roles: {sorted(role_set)}")
    train_keys = keys.loc[keys.split_role.eq("train")].copy().reset_index(drop=True)
    dev_keys = keys.loc[keys.split_role.eq("inner_dev")]
    train_keys["source_row"] = pd.to_numeric(train_keys.source_row, errors="raise").astype(np.int64)
    if (len(train_keys) != 419 or len(dev_keys) != 93
            or train_keys.source_row.duplicated().any()
            or train_keys.source_object_id.nunique() != 110
            or train_keys.full_inchikey.nunique() != 5
            or dev_keys.source_object_id.nunique() != 24
            or set(train_keys.source_object_id) & set(dev_keys.source_object_id)):
        raise ValueError("frozen 419/93 TRAIN/inner-dev key-role/object contract failed")
    if not train_keys.full_inchikey.map(lambda value: bool(FULL_KEY.fullmatch(str(value)))).all():
        raise ValueError("TRAIN rows contain an incomplete chemical identity")

    import anndata as ad
    from scipy import sparse

    train_rows = train_keys.source_row.to_numpy(dtype=np.int64)
    order = np.argsort(train_rows, kind="mergesort")
    sorted_rows = train_rows[order]
    adata = ad.read_h5ad(H5AD, backed="r")
    try:
        genes = np.asarray(list(map(str, adata.var_names)), dtype=str)
        gene_sha = sha256_text("\n".join(genes.tolist()) + "\n")
        expected_var_names = mapping_table["h5ad_var_name"].astype(str).to_numpy(dtype=np.str_)
        expected_annotations = mapping_table["h5ad_var_annotation"].astype(str).to_numpy(dtype=np.str_)
        if adata.shape != (78453, 978) or len(genes) != 978:
            raise ValueError("source H5AD metadata shape differs from the frozen 78453x978 contract")
        if not np.array_equal(genes, expected_var_names):
            raise ValueError("source H5AD ordered var_names differ from gene_axis_mapping.tsv")
        if "gene_id" not in adata.var.columns:
            raise ValueError("source H5AD var metadata lacks gene_id for frozen axis binding")
        h5ad_gene_ids = adata.var["gene_id"].astype(str).to_numpy(dtype=np.str_)
        if not np.array_equal(h5ad_gene_ids, expected_annotations):
            raise ValueError("source H5AD var.gene_id differs from mapped h5ad_var_annotation")
        obs_names = np.asarray(adata.obs_names[sorted_rows].astype(str), dtype=str)
        if not np.array_equal(obs_names, train_keys.obs_name.astype(str).to_numpy()[order]):
            raise ValueError("TRAIN source_row-to-obs_name identity differs from frozen response-blind keys")
        if "cell_iname" not in adata.obs.columns:
            raise ValueError("source H5AD metadata lacks the registered cell_iname object identity")
        source_objects = np.asarray(adata.obs["cell_iname"].iloc[sorted_rows].astype(str), dtype=str)
        if not np.array_equal(source_objects, train_keys.source_object_id.astype(str).to_numpy()[order]):
            raise ValueError("TRAIN source_row-to-source_object_id identity differs from frozen keys")
        if "X_ctl" not in adata.obsm:
            raise ValueError("source H5AD lacks paired pre-treatment obsm['X_ctl']")
        X_sorted = adata.X[sorted_rows, :]
        X_ctl_sorted = adata.obsm["X_ctl"][sorted_rows, :]
        if hasattr(X_sorted, "to_memory"):
            X_sorted = X_sorted.to_memory()
        if hasattr(X_ctl_sorted, "to_memory"):
            X_ctl_sorted = X_ctl_sorted.to_memory()
        if sparse.issparse(X_sorted):
            X_sorted = X_sorted.toarray()
        if sparse.issparse(X_ctl_sorted):
            X_ctl_sorted = X_ctl_sorted.toarray()
        X_train = np.empty((len(train_rows), 978), dtype=np.float32)
        X_ctl_train = np.empty_like(X_train)
        X_train[order] = np.asarray(X_sorted, dtype=np.float32)
        X_ctl_train[order] = np.asarray(X_ctl_sorted, dtype=np.float32)
    finally:
        adata.file.close()
    if (X_train.shape != (419, 978) or X_ctl_train.shape != X_train.shape
            or not np.isfinite(X_train).all() or not np.isfinite(X_ctl_train).all()):
        raise ValueError("TRAIN-only source H5AD slice is nonfinite or has a wrong 419x978 shape")
    delta_train = X_train - X_ctl_train
    if not np.isfinite(delta_train).all():
        raise FloatingPointError("TRAIN-only X-X_ctl contains nonfinite response values")
    if not np.allclose(X_train, X_ctl_train + delta_train, rtol=0.0, atol=2e-6):
        raise ValueError("TRAIN-only X, X_ctl, and delta fail the float32 round-trip")
    feature_by_key, feature_receipt = _load_audited_ecfp4(feature_root)
    if set(train_keys.full_inchikey.astype(str)) - set(feature_by_key):
        raise ValueError("audited ECFP4 features do not cover the five frozen TRAIN keys")
    feature_matrix = np.stack([feature_by_key[key] for key in
                                train_keys.full_inchikey.astype(str)]).astype(np.float32, copy=False)
    if feature_matrix.shape != (419, 2048) or not np.isfinite(feature_matrix).all():
        raise ValueError("TRAIN ECFP4 matrix must be finite 419x2048")
    values = {"source_row": train_rows.copy(),
              "source_object_id": train_keys.source_object_id.astype(str).to_numpy(),
              "full_inchikey": train_keys.full_inchikey.astype(str).to_numpy(),
              "pert_id": train_keys.pert_id.astype(str).to_numpy(),
              "genes": genes, "X": X_train, "X_ctl": X_ctl_train, "delta": delta_train}
    evidence = {
        "source_run_id": SOURCE_RUN_ID,
        "metadata_freeze_receipt_sha256": sha256_file(freeze_path),
        "gene_axis_receipt_sha256": sha256_file(axis_path),
        "condition_table_sha256": sha256_file(keys_path),
        "source_h5ad_sha256": sha256_file(H5AD),
        "gene_axis_mapping_sha256": mapping_file_sha256,
        "gene_axis_mapping_semantic_sha256": semantic_mapping_sha256,
        "gene_axis_sha256": gene_sha,
        "source_h5ad_shape": [78453, 978],
        "var_names_match_mapping_axis_index": True,
        "gene_id_match_mapping_annotation": True,
        "train_source_rows_sha256": sha256_text("\n".join(map(str, train_rows.tolist())) + "\n"),
        "train_condition_keys_sha256": sha256_text("\n".join(train_keys.full_condition_key.astype(str).tolist()) + "\n"),
        "train_source_objects": 110, "train_conditions": 419,
        "train_full_inchikeys": sorted(train_keys.full_inchikey.astype(str).unique().tolist()),
        "inner_dev_condition_count_from_response_blind_metadata_only": int(len(dev_keys)),
        "inner_dev_numeric_rows_loaded": False,
        "frozen_train_inner_values_npz_opened": False,
        "matrix_rows_selected_from_h5ad": "TRAIN source_row values only",
        "train_X_finite": True, "train_X_ctl_finite": True, "train_delta_finite": True,
        "train_delta_roundtrip_max_abs": float(np.max(np.abs(
            (X_ctl_train + delta_train).astype(np.float64) - X_train.astype(np.float64)))),
        "feature_audit": feature_receipt,
        "train_feature_matrix_shape": list(feature_matrix.shape),
        "feature_matrix_sha256": hashlib.sha256(feature_matrix.tobytes()).hexdigest(),
    }
    return train_keys, values, {key: feature_matrix[index] for index, key in
                                enumerate(train_keys.full_inchikey.astype(str))}, evidence


def _reuse_root_verified_exp055_train_gate() -> dict[str, Any]:
    """Reuse the existing 21/21 root-verified TRAIN gate without reopening inner-dev values."""
    v5_root = ROOT / "artifacts/experiments/EXP-055/20261007_exp055_transigen_trainpath_recovery_v5"
    root_readback_path = v5_root / "root_independent_preflight_readback.json"
    executor_preflight_path = v5_root / "run_preflight_receipt.json"
    cpu_gate_path = v5_root / "fit_attempt_001/cpu_preflight_gate.json"
    for path in (root_readback_path, executor_preflight_path, cpu_gate_path):
        if not path.is_file():
            raise FileNotFoundError(f"root-verified v5 TRAIN gate evidence is missing: {path}")
    root_receipt = json.loads(root_readback_path.read_text(encoding="utf-8"))
    preflight = json.loads(executor_preflight_path.read_text(encoding="utf-8"))
    cpu_gate = json.loads(cpu_gate_path.read_text(encoding="utf-8"))
    checks = root_receipt.get("checks", {})
    if (root_receipt.get("status") != "PASS_ROOT_INDEPENDENT_TRAIN_KEY_AXIS_NUMERIC_READBACK_CPU_ONLY"
            or root_receipt.get("run_id") != v5_root.name
            or root_receipt.get("executor_preflight_sha256") != sha256_file(executor_preflight_path)
            or root_receipt.get("attempt_created") is not False
            or root_receipt.get("gpu_hours") != 0.0
            or len(checks) != 21 or not all(value is True for value in checks.values())
            or root_receipt.get("train_conditions") != 419
            or root_receipt.get("train_objects") != 110
            or root_receipt.get("train_full_inchikeys") != 5
            or root_receipt.get("gene_count") != 978
            or root_receipt.get("train_inner_object_overlap") != 0
            or root_receipt.get("inner_dev_numeric_rows_read") is not False
            or root_receipt.get("pdo_outer_locked_numeric_values_read") is not False
            or root_receipt.get("torch_or_cuda_used") is not False):
        raise RuntimeError("the existing v5 root-independent 21/21 TRAIN input gate is not reusable")
    if (cpu_gate.get("status") != "PASS_TRAIN_ONLY_IDENTITY_BUDGET_AND_NO_OVERWRITE_GATES"
            or cpu_gate.get("run_id") != v5_root.name
            or cpu_gate.get("train_conditions") != root_receipt.get("train_conditions")
            or cpu_gate.get("train_objects") != root_receipt.get("train_objects")
            or cpu_gate.get("genes") != root_receipt.get("gene_count")
            or cpu_gate.get("train_delta_sha256") != root_receipt.get("train_delta_sha256")
            or cpu_gate.get("train_feature_matrix_sha256") != root_receipt.get("train_feature_matrix_sha256")
            or preflight.get("run_id") != v5_root.name
            or preflight.get("data_roles", {}).get("inner_dev_numeric_rows_loaded") is not False
            or preflight.get("data_roles", {}).get("pdo_numeric_sensitivity_or_expression_opened") is not False
            or preflight.get("data_roles", {}).get("outer_or_locked_numeric_values_opened") is not False):
        raise RuntimeError("the v5 executor CPU gate and independent TRAIN readback disagree")
    return {"root_readback_path": str(root_readback_path),
            "root_readback_sha256": sha256_file(root_readback_path),
            "executor_preflight_path": str(executor_preflight_path),
            "executor_preflight_sha256": sha256_file(executor_preflight_path),
            "runner_sha256_from_verified_gate": root_receipt.get("runner_sha256"),
            "source_h5ad_sha256": preflight.get("source_train_only_data_identity_gate", {}).get("source_h5ad_sha256"),
            "gene_axis_sha256": preflight.get("source_train_only_data_identity_gate", {}).get("gene_axis_sha256"),
            "condition_table_sha256": preflight.get("source_train_only_data_identity_gate", {}).get(
                "condition_table_sha256"),
            "train_conditions": 419, "train_objects": 110, "inner_dev_conditions": 93,
            "inner_dev_objects": 24, "object_overlap": 0, "full_inchikeys": 5,
            "genes": 978, "train_delta_sha256": root_receipt["train_delta_sha256"],
            "train_condition_keys_sha256": cpu_gate.get("train_condition_keys_sha256"),
            "train_feature_matrix_sha256": root_receipt["train_feature_matrix_sha256"],
            "checks_reused": checks, "checks_reused_count": len(checks),
            "inner_dev_numeric_rows_loaded": False,
            "pdo_outer_locked_numeric_values_opened": False}


def _load_exp055_trainpath_train_inputs_reusing_gate(
        verified_gate: dict[str, Any], keys_path: Path, feature_root: Path
        ) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Read the already-verified 419 TRAIN rows without repeating the frozen 21 identity/axis checks."""
    import anndata as ad
    from scipy import sparse

    keys = pd.read_csv(keys_path, sep="\t", dtype=str, keep_default_na=False)
    train_keys = keys.loc[keys.split_role.eq("train")].copy().reset_index(drop=True)
    train_rows = pd.to_numeric(train_keys.source_row, errors="raise").to_numpy(dtype=np.int64)
    if len(train_keys) != 419 or len(np.unique(train_rows)) != 419:
        raise ValueError("reused frozen key table no longer yields the verified 419 TRAIN source rows")
    order = np.argsort(train_rows, kind="mergesort")
    sorted_rows = train_rows[order]
    adata = ad.read_h5ad(H5AD, backed="r")
    try:
        genes = np.asarray(list(map(str, adata.var_names)), dtype=str)
        if adata.shape != (78453, 978) or len(genes) != 978:
            raise ValueError("reused source H5AD no longer has the root-verified 78453x978 shape")
        x_sorted = adata.X[sorted_rows, :]
        x_ctl_sorted = adata.obsm["X_ctl"][sorted_rows, :]
        if hasattr(x_sorted, "to_memory"):
            x_sorted = x_sorted.to_memory()
        if hasattr(x_ctl_sorted, "to_memory"):
            x_ctl_sorted = x_ctl_sorted.to_memory()
        if sparse.issparse(x_sorted):
            x_sorted = x_sorted.toarray()
        if sparse.issparse(x_ctl_sorted):
            x_ctl_sorted = x_ctl_sorted.toarray()
        x = np.empty((419, 978), dtype=np.float32)
        x_ctl = np.empty_like(x)
        x[order] = np.asarray(x_sorted, dtype=np.float32)
        x_ctl[order] = np.asarray(x_ctl_sorted, dtype=np.float32)
    finally:
        adata.file.close()
    delta = x - x_ctl
    if not np.isfinite(x).all() or not np.isfinite(x_ctl).all() or not np.isfinite(delta).all():
        raise FloatingPointError("reused verified TRAIN numeric slice contains non-finite values")
    delta_sha = hashlib.sha256(delta.tobytes()).hexdigest()
    if delta_sha != verified_gate["train_delta_sha256"]:
        raise RuntimeError("TRAIN delta changed since the root-verified v5 input gate")
    train_key_sha = sha256_text("\n".join(train_keys.full_condition_key.astype(str).tolist()) + "\n")
    if train_key_sha != verified_gate.get("train_condition_keys_sha256", train_key_sha):
        raise RuntimeError("TRAIN condition keys changed since the root-verified v5 input gate")
    feature_by_key, feature_receipt = _load_audited_ecfp4(feature_root)
    feature_matrix = np.stack([feature_by_key[key] for key in
                               train_keys.full_inchikey.astype(str)]).astype(np.float32, copy=False)
    if hashlib.sha256(feature_matrix.tobytes()).hexdigest() != verified_gate["train_feature_matrix_sha256"]:
        raise RuntimeError("TRAIN ECFP4 features changed since the root-verified v5 input gate")
    values = {"source_row": train_rows.copy(),
              "source_object_id": train_keys.source_object_id.astype(str).to_numpy(),
              "full_inchikey": train_keys.full_inchikey.astype(str).to_numpy(),
              "pert_id": train_keys.pert_id.astype(str).to_numpy(),
              "genes": genes, "X": x, "X_ctl": x_ctl, "delta": delta}
    evidence = {"reused_root_verified_gate": verified_gate,
                "source_row_read_method": "response-blind frozen TRAIN source_row only",
                "train_source_rows_sha256": sha256_text(
                    "\n".join(map(str, train_rows.tolist())) + "\n"),
                "train_delta_sha256": delta_sha, "train_condition_keys_sha256": train_key_sha,
                "feature_audit": feature_receipt,
                "feature_matrix_sha256": hashlib.sha256(feature_matrix.tobytes()).hexdigest(),
                "train_conditions": 419, "train_objects": 110, "genes": 978,
                "full_inchikey_count": 5, "inner_dev_numeric_rows_loaded": False}
    return train_keys, values, {key: feature_matrix[index] for index, key in
                                enumerate(train_keys.full_inchikey.astype(str))}, evidence


def _load_exp055_inner_dev_after_checkpoint_freeze(
        train_keys: pd.DataFrame, train_values: dict[str, np.ndarray], source_run: Path
        ) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """Load the already-frozen 93 inner-dev rows only after the 300-epoch checkpoint is saved."""
    values_path = source_run / "frozen_train_inner_values.npz"
    values_receipt_path = source_run / "value_contract_receipt.json"
    readback_path = source_run / "root_values_readback_receipt.json"
    keys_path = source_run / "response_blind_conditions.tsv"
    for path in (values_path, values_receipt_path, readback_path, keys_path):
        if not path.is_file():
            raise FileNotFoundError(f"post-checkpoint EXP-055 development readback input missing: {path}")
    values_receipt = json.loads(values_receipt_path.read_text(encoding="utf-8"))
    readback = json.loads(readback_path.read_text(encoding="utf-8"))
    values_sha = sha256_file(values_path)
    expected_sha = values_receipt.get("files_sha256", {}).get("frozen_train_inner_values_npz")
    if (readback.get("checks", {}).get("result") != "PASS"
            or readback.get("run_id") != source_run.name
            or readback.get("input_sha256") != expected_sha or values_sha != expected_sha):
        raise RuntimeError("the previously root-verified frozen TRAIN/inner-dev values changed")
    keys = pd.read_csv(keys_path, sep="\t", dtype=str, keep_default_na=False)
    # This archive was created by the project's own frozen-value pipeline and is checksum-bound
    # by the root readback receipt; categorical key arrays in it are NumPy object arrays.
    with np.load(values_path, allow_pickle=True) as stored:
        values = {name: stored[name].copy() for name in
                  ("source_row", "split_role", "source_object_id", "full_inchikey", "pert_id",
                   "genes", "X", "X_ctl", "delta")}
    if (len(keys) != 512 or keys.full_condition_key.astype(str).duplicated().any()
            or values["delta"].shape != (512, 978)):
        raise ValueError("frozen 512-row key/value artifact has a wrong row or gene shape")
    for column in ("source_row", "split_role", "source_object_id", "full_inchikey", "pert_id"):
        if not np.array_equal(keys[column].astype(str).to_numpy(), values[column].astype(str)):
            raise ValueError(f"frozen post-checkpoint value/key alignment failed for {column}")
    if not np.array_equal(values["genes"].astype(str), train_values["genes"].astype(str)):
        raise ValueError("post-checkpoint frozen expression gene axis differs from the TRAIN inputs")
    train_mask = keys.split_role.eq("train").to_numpy()
    dev_mask = keys.split_role.eq("inner_dev").to_numpy()
    train_ids = set(keys.loc[train_mask, "source_object_id"].astype(str))
    dev_ids = set(keys.loc[dev_mask, "source_object_id"].astype(str))
    if (train_mask.sum() != 419 or dev_mask.sum() != 93 or len(train_ids) != 110
            or len(dev_ids) != 24 or train_ids & dev_ids):
        raise ValueError("frozen post-checkpoint TRAIN/inner-dev object denominator changed")
    _assert_exp055_train_rows_match_frozen_archive(values, train_values, train_mask)
    if not np.isfinite(values["delta"]).all() or not np.isfinite(values["X"]).all() or not np.isfinite(values["X_ctl"]).all():
        raise FloatingPointError("root-verified frozen TRAIN/inner-dev artifact contains non-finite expression values")
    metadata = {"status": "PASS_POSTCHECKPOINT_FROZEN_INNER_DEV_VALUE_READBACK",
                "values_npz": str(values_path), "values_npz_sha256": values_sha,
                "root_values_readback_receipt_sha256": sha256_file(readback_path),
                "value_contract_receipt_sha256": sha256_file(values_receipt_path),
                "train_conditions": int(train_mask.sum()), "train_objects": len(train_ids),
                "inner_dev_conditions": int(dev_mask.sum()), "inner_dev_objects": len(dev_ids),
                "object_overlap": 0, "full_inchikeys": int(keys.full_inchikey.nunique()),
                "genes": int(values["delta"].shape[1]),
                "inner_dev_values_loaded_after_checkpoint_epoch": 300,
                "inner_dev_used_for_training_or_selection": False,
                "pdo_outer_locked_values_opened": False}
    dev_keys = keys.loc[dev_mask].copy().reset_index(drop=True)
    dev_values = {name: values[name][dev_mask].copy() for name in
                  ("source_row", "source_object_id", "full_inchikey", "pert_id", "X", "X_ctl", "delta")}
    dev_values["genes"] = values["genes"].astype(str).copy()
    return dev_keys, dev_values, metadata


def _assert_exp055_train_rows_match_frozen_archive(
        frozen_values: dict[str, np.ndarray], train_values: dict[str, np.ndarray],
        train_mask: np.ndarray) -> None:
    """Compare TRAIN rows while keeping the 978-gene axis outside the row mask."""
    if not np.array_equal(frozen_values["genes"].astype(str), train_values["genes"].astype(str)):
        raise ValueError("post-checkpoint frozen gene axis differs from the fitted TRAIN input")
    for name in ("source_row", "source_object_id", "full_inchikey", "pert_id", "X", "X_ctl", "delta"):
        if not np.array_equal(frozen_values[name][train_mask], train_values[name]):
            raise ValueError(f"post-checkpoint frozen TRAIN values differ from fitted input for {name}")


def _exp055_train_and_dev_baselines(train_keys: pd.DataFrame, train_delta: np.ndarray,
                                    dev_keys: pd.DataFrame, dev_delta: np.ndarray,
                                    gene_variance: np.ndarray) -> dict[str, Any]:
    """Compute zero-change and TRAIN-fitted per-drug means for matched TRAIN/dev reporting."""
    means: dict[str, np.ndarray] = {}
    for key in sorted(train_keys.full_inchikey.astype(str).unique()):
        positions = np.flatnonzero(train_keys.full_inchikey.astype(str).to_numpy() == key)
        means[key] = train_delta[positions].mean(axis=0, dtype=np.float64).astype(np.float32)
    per_drug_train = np.stack([means[str(key)] for key in train_keys.full_inchikey.astype(str)])
    per_drug_dev = np.stack([means[str(key)] for key in dev_keys.full_inchikey.astype(str)])
    zero_train = np.zeros_like(train_delta, dtype=np.float32)
    zero_dev = np.zeros_like(dev_delta, dtype=np.float32)
    return {"fit_policy": "zero-change is fixed; per-drug mean delta is fitted on TRAIN only",
            "normalizer_fit_on": "TRAIN delta rows only",
            "metrics": {
                "zero_change": {"train": keyed_metrics(train_keys, train_delta, zero_train, gene_variance),
                                "inner_dev": keyed_metrics(dev_keys, dev_delta, zero_dev, gene_variance)},
                "per_drug_mean_delta": {"train": keyed_metrics(train_keys, train_delta, per_drug_train, gene_variance),
                                         "inner_dev": keyed_metrics(dev_keys, dev_delta, per_drug_dev, gene_variance)}}}


def _exp055_cpu_rng_state(torch: Any, loader_generator: Any) -> dict[str, Any]:
    """Return a weights_only-safe CPU RNG snapshot for a restart point."""
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"bit_generator": str(numpy_state[0]),
                      "keys": numpy_state[1].astype(np.uint32, copy=False).tolist(),
                      "position": int(numpy_state[2]),
                      "has_gauss": int(numpy_state[3]),
                      "cached_gaussian": float(numpy_state[4])},
            "torch_cpu": torch.get_rng_state().cpu(),
            "dataloader_generator": loader_generator.get_state().cpu()}


def _exp055_cpu_restore_rng_state(torch: Any, loader_generator: Any,
                                 state: dict[str, Any]) -> dict[str, bool]:
    """Restore every saved CPU RNG stream without touching model or optimizer state."""
    required = {"python", "numpy", "torch_cpu", "dataloader_generator"}
    if set(state) != required:
        raise ValueError(f"resume checkpoint RNG fields differ: {sorted(state)}")
    np_state = state["numpy"]
    if set(np_state) != {"bit_generator", "keys", "position", "has_gauss", "cached_gaussian"}:
        raise ValueError("resume checkpoint NumPy RNG fields differ")
    random.setstate(tuple(state["python"]))
    np.random.set_state((str(np_state["bit_generator"]),
        np.asarray(np_state["keys"], dtype=np.uint32), int(np_state["position"]),
        int(np_state["has_gauss"]), float(np_state["cached_gaussian"])))
    torch.set_rng_state(state["torch_cpu"].cpu())
    loader_generator.set_state(state["dataloader_generator"].cpu())
    restored = _exp055_cpu_rng_state(torch, loader_generator)
    expected = {"python": state["python"], "numpy": np_state,
                "torch_cpu": state["torch_cpu"].cpu(),
                "dataloader_generator": state["dataloader_generator"].cpu()}
    return {"python": restored["python"] == tuple(expected["python"]),
            "numpy": restored["numpy"] == expected["numpy"],
            "torch_cpu": _torch_tree_exact(torch, restored["torch_cpu"], expected["torch_cpu"]),
            "dataloader_generator": _torch_tree_exact(
                torch, restored["dataloader_generator"], expected["dataloader_generator"])}


def _exp055_cpu_rng_state_json(torch: Any, loader_generator: Any) -> dict[str, Any]:
    state = _exp055_cpu_rng_state(torch, loader_generator)
    return {"python": state["python"], "numpy": state["numpy"],
            "torch_cpu": state["torch_cpu"].tolist(),
            "dataloader_generator": state["dataloader_generator"].tolist()}


def _exp055_cpu_checkpoint_payload(torch: Any, model: Any, optimizer: Any,
                                   loader_generator: Any, *, epoch: int,
                                   checkpoint_role: str,
                                   train_metrics: dict[str, Any] | None) -> dict[str, Any]:
    learning_rate = (1e-4 if EXPERIMENT_ID == "EXP-059"
                     and RUN_ID == EXP059_TRANSIGEN_LR_STABILITY_RUN_ID else 1e-3)
    return {"state_dict": {key: tensor.detach().cpu().clone()
                            for key, tensor in model.state_dict().items()},
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "rng_state": _exp055_cpu_rng_state(torch, loader_generator),
            "model": "upstream TranSiGen", "epoch": int(epoch), "seed": int(SEED),
            "objective": "raw_sum", "gradient_clipping": False,
            "learning_rate": learning_rate, "weight_decay": 1e-5,
            "checkpoint_role": checkpoint_role, "train_metrics": train_metrics}


def _exp059_tensor_summary(torch: Any, tensor: Any) -> dict[str, Any]:
    detached = tensor.detach()
    finite = torch.isfinite(detached)
    finite_values = detached[finite]
    return {"shape": list(detached.shape), "finite_n": int(finite.sum().item()),
            "denominator": int(detached.numel()),
            "nonfinite_n": int(detached.numel() - finite.sum().item()),
            "finite_min": safe_number(float(finite_values.min().item())) if finite_values.numel() else None,
            "finite_max": safe_number(float(finite_values.max().item())) if finite_values.numel() else None,
            "finite_max_abs": safe_number(float(finite_values.abs().max().item())) if finite_values.numel() else None,
            "nonfinite_positions_first16": torch.nonzero(~finite, as_tuple=False)[:16].cpu().tolist()}


def _exp059_kld_pert_decomposition(torch: Any, capture: dict[str, Any]) -> dict[str, Any] | None:
    if not all(key in capture for key in ("mu_pred", "logvar_pred", "mu2", "logvar2")):
        return None
    mu_pred, logvar_pred, mu2, logvar2 = (
        capture["mu_pred"].detach(), capture["logvar_pred"].detach(),
        capture["mu2"].detach(), capture["logvar2"].detach())
    with torch.no_grad():
        terms = {
            "mu_difference": mu_pred - mu2,
            "logvar_difference": logvar_pred - logvar2,
            "exp_logvar_pred": logvar_pred.exp(),
            "exp_logvar2_denominator": logvar2.exp(),
        }
        terms["mu_difference_squared"] = terms["mu_difference"].pow(2)
        terms["numerator"] = terms["mu_difference_squared"] + terms["exp_logvar_pred"]
        terms["ratio"] = terms["numerator"] / terms["exp_logvar2_denominator"]
        terms["one_plus_logvar_difference"] = 1.0 + terms["logvar_difference"]
        terms["elementwise_kld_inside_sum"] = terms["one_plus_logvar_difference"] - terms["ratio"]
    return {name: _exp059_tensor_summary(torch, tensor) for name, tensor in terms.items()}


def _exp059_batchnorm_health(torch: Any, model: Any) -> dict[str, Any]:
    rows = []
    first_nonfinite = None
    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        for field in ("running_mean", "running_var"):
            tensor = getattr(module, field, None)
            if tensor is None:
                continue
            finite = bool(torch.isfinite(tensor).all().item())
            row = {"module": module_name, "field": field, "finite": finite,
                   "summary": _exp059_tensor_summary(torch, tensor)}
            rows.append(row)
            if not finite and first_nonfinite is None:
                first_nonfinite = {"module": module_name, "field": field,
                                   "summary": row["summary"]}
    return {"finite": first_nonfinite is None, "first_nonfinite": first_nonfinite,
            "buffers": rows}


def _exp055_cpu_save_checkpoint(torch: Any, path: Path, model: Any, optimizer: Any,
                                loader_generator: Any, *, epoch: int,
                                checkpoint_role: str,
                                train_metrics: dict[str, Any] | None,
                                readback: bool) -> tuple[dict[str, Any], float]:
    payload = _exp055_cpu_checkpoint_payload(torch, model, optimizer, loader_generator,
        epoch=epoch, checkpoint_role=checkpoint_role, train_metrics=train_metrics)
    start = time.perf_counter()
    _atomic_torch_checkpoint(torch, path, payload)
    if readback:
        stored = torch.load(path, map_location="cpu", weights_only=True)
        if not _torch_tree_exact(torch, payload, stored):
            raise RuntimeError(f"CPU checkpoint weight/optimizer/RNG readback differs: {path.name}")
    return payload, time.perf_counter() - start


def _exp055_cpu_prediction_arrays(keys: pd.DataFrame, genes: np.ndarray,
                                  truth: np.ndarray, prediction: np.ndarray,
                                  prefix: str = "train") -> dict[str, np.ndarray]:
    if truth.shape != prediction.shape or truth.shape != (len(keys), 978):
        raise ValueError(f"{prefix} prediction/target shape differs from keyed rows: {truth.shape}, {prediction.shape}")
    return {f"{prefix}_condition_key": keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
            f"{prefix}_source_object_id": keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
            f"{prefix}_full_inchikey": keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
            "genes": genes.astype(str), f"{prefix}_target_delta": truth,
            f"{prefix}_delta": prediction}


def _exp055_cpu_prediction_readback(path: Path, expected: dict[str, np.ndarray]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        missing = sorted(set(expected) - set(saved.files))
        mismatched = [name for name, value in expected.items()
                      if name in saved.files and not _np_arrays_equal_with_nan(saved[name], value)]
        if missing or mismatched:
            raise RuntimeError(f"keyed CPU prediction readback failed: missing={missing}, mismatched={mismatched}")
        shapes = {name: list(saved[name].shape) for name in expected}
        finite = {name: bool(np.isfinite(saved[name]).all())
                  for name in expected if saved[name].dtype.kind in "fc"}
    if not all(finite.values()):
        raise FloatingPointError(f"keyed CPU prediction archive contains non-finite values: {path.name}")
    return {"status": "PASS_EXACT_KEYED_VALUES_AND_FINITE_NUMERIC_ARRAYS",
            "archive": path.name, "sha256": sha256_file(path),
            "arrays": shapes, "finite_numeric_arrays": finite}


def run_exp055_transigen_cpu_contract_check(cfg: dict[str, Any], config_sha: str,
                                             attempt_id: str, device_name: str) -> dict[str, Any]:
    """Exercise the real CPU model/save boundary without a TRAIN optimizer update."""
    contract_wall_start = time.perf_counter()
    contract_process_start = time.process_time()
    is_exp059 = (EXPERIMENT_ID == "EXP-059"
                 and RUN_ID == EXP059_TRANSIGEN_LR_STABILITY_RUN_ID
                 and cfg.get("run_id") == RUN_ID)
    expected_run = EXP059_TRANSIGEN_LR_STABILITY_RUN_ID if is_exp059 else EXP055_TRANSIGEN_CPU_REPRESENTATIVE_RUN_ID
    if (EXPERIMENT_ID not in {"EXP-055", "EXP-059"} or RUN_ID != expected_run
            or SOURCE_RUN_ID != DEFAULT_SOURCE_RUN_ID or attempt_id != "fit_attempt_001"
            or device_name != "cpu"):
        raise ValueError("CPU contract check is bound to one registered EXP-055/059 CPU TranSiGen run")
    if (cfg.get("authorization", {}).get("training_device_policy") != "cpu_only_zero_gpu"
            or cfg.get("authorization", {}).get("allowed_device_uuid") != GPU_UUID
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""):
        raise RuntimeError("CPU contract check requires the registered CPU-only route and an empty CUDA_VISIBLE_DEVICES")
    receipt_path = RUN_DIR / "cpu_entry_contract_check.json"
    checkpoint_path = RUN_DIR / "cpu_contract_checkpoint.pt"
    allowed_run_files = ({receipt_path.name, checkpoint_path.name}
                         if not is_exp059 else EXP059_CPU_RUN_ROOT_ALLOWED_ENTRIES)
    if RUN_DIR.exists() and any(path.name not in allowed_run_files for path in RUN_DIR.iterdir()):
        raise FileExistsError("CPU representative run contains unexpected artifacts before its one contract check")
    if receipt_path.exists() or checkpoint_path.exists():
        raise FileExistsError("CPU representative contract check is immutable and may run only once")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError("CPU contract check unexpectedly sees or initialized CUDA")
    device = torch.device("cpu")
    parent_evidence_path, parent_evidence_sha = (
        resolve_exp059_parent_evidence(cfg) if is_exp059 else (None, None))
    verified_gate = _reuse_root_verified_exp055_train_gate()
    train_keys, values, feature_by_key, data_evidence = _load_exp055_trainpath_train_inputs_reusing_gate(
        verified_gate, SOURCE_RUN_DIR / "response_blind_conditions.tsv", SOURCE_RUN_DIR)
    positions = np.arange(min(4, len(train_keys)), dtype=np.int64)
    features = _full_key_features(train_keys, positions, feature_by_key)
    model, _ = _load_transigen(torch, device, SEED, features_dim=2048, seed_cuda=False)
    model.eval()
    before = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    with torch.no_grad():
        x1 = torch.as_tensor(values["X_ctl"][positions], dtype=torch.float32)
        x2 = torch.as_tensor(values["X"][positions], dtype=torch.float32)
        feat = torch.as_tensor(features, dtype=torch.float32)
        _, _, prediction, losses = _transigen_forward_batch(
            torch, model, (x1, x2, feat), device)
        delta = (prediction - x1).detach().cpu().numpy().astype(np.float32)
    if delta.shape != (len(positions), 978) or len(losses) != 7:
        raise ValueError("real TranSiGen CPU contract check did not return the required 978-gene interface shape")
    after = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    if not _torch_tree_exact(torch, before, after):
        raise RuntimeError("CPU contract forward unexpectedly changed model parameters or BatchNorm buffers")

    contract_payload = {"status": "CPU_CONTRACT_TEST_ONLY_NO_TRAIN_UPDATE",
        "state_dict": before, "epoch": 0, "seed": int(SEED),
        "device": "cpu", "optimizer_updates": 0}
    _atomic_torch_checkpoint(torch, checkpoint_path, contract_payload)
    checkpoint_readback = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not _torch_tree_exact(torch, contract_payload, checkpoint_readback):
        raise RuntimeError("CPU contract checkpoint save/readback did not preserve the original model state")

    injected_error = OSError("EXP-055 CPU contract check injected checkpoint write failure")
    original_torch_save = torch.save
    failure_propagated = False
    try:
        def fail_torch_save(*_args: Any, **_kwargs: Any) -> None:
            raise injected_error
        torch.save = fail_torch_save
        try:
            _atomic_torch_checkpoint(torch, RUN_DIR / "must_not_be_created.pt", contract_payload)
        except OSError as error:
            failure_propagated = error is injected_error
        else:
            raise RuntimeError("atomic checkpoint writer swallowed an injected write error")
    finally:
        torch.save = original_torch_save
    if not failure_propagated:
        raise RuntimeError("injected checkpoint I/O failure did not propagate unchanged")

    delta_finite_n = int(np.isfinite(delta).sum())
    loss_finite = [bool(torch.isfinite(loss).all()) for loss in losses]
    loss_values = [safe_number(float(loss.detach().cpu().item())) if finite else None
                   for loss, finite in zip(losses, loss_finite)]
    receipt = {"status": "PASS_CPU_ENTRY_SHAPE_SAVE_AND_FAILURE_PROPAGATION_NO_TRAIN_UPDATE",
        "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID, "attempt_id": attempt_id,
        "created_at_utc": utc_now(), "runner_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": config_sha, "source_run_id": SOURCE_RUN_ID,
        "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_initialized": bool(torch.cuda.is_initialized()),
        "approved_gpu_uuid_recorded_but_not_used": GPU_UUID,
        "input_gate": {"root_verified_gate": verified_gate,
                       "train_identity": data_evidence,
                       "inner_dev_numeric_rows_loaded": False,
                       "pdo_outer_locked_values_opened": False},
        "model": "upstream TranSiGen", "model_source_sha256": sha256_file(TRANSIGEN_ROOT / "model.py"),
        "probe_condition_keys": train_keys.iloc[positions].full_condition_key.astype(str).tolist(),
        "main_delta_shape": list(delta.shape), "main_delta_finite": bool(np.isfinite(delta).all()),
        "main_delta_finite_n": delta_finite_n, "main_delta_denominator": int(delta.size),
        "main_delta_nonfinite_n": int(delta.size - delta_finite_n),
        "loss_term_count": len(losses), "loss_terms_finite": loss_finite,
        "loss_term_values_or_null": loss_values,
        "model_state_unchanged": True, "optimizer_updates": 0,
        "initialization_mode": "fresh_from_registered_seed",
        "parent_checkpoint_loaded_into_candidate": False,
        "parent_checkpoint_evidence_path": str(parent_evidence_path.relative_to(ROOT)) if parent_evidence_path else None,
        "parent_checkpoint_evidence_sha256": parent_evidence_sha,
        "checkpoint_path": checkpoint_path.name, "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_weights_only_readback": "PASS_EXACT",
        "injected_write_failure_type": type(injected_error).__name__,
        "injected_write_failure_propagated": failure_propagated,
        "contract_check_cpu_wall_seconds": time.perf_counter() - contract_wall_start,
        "contract_check_cpu_process_seconds": time.process_time() - contract_process_start,
        "gpu_hours": 0.0}
    atomic_json(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path)}


def _exp055_resume_tensor_summary(torch: Any, name: str, value: Any) -> dict[str, Any]:
    if not torch.is_tensor(value):
        return {"name": name, "type": type(value).__name__}
    flat = value.detach().cpu().reshape(-1)
    mask = torch.isfinite(flat)
    finite = flat[mask]
    return {"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
        "finite_n": int(mask.sum().item()), "denominator": int(flat.numel()),
        "nonfinite_n": int(flat.numel() - mask.sum().item()),
        "finite_min": float(finite.min().item()) if finite.numel() else None,
        "finite_max": float(finite.max().item()) if finite.numel() else None,
        "finite_max_abs": float(finite.abs().max().item()) if finite.numel() else None,
        "first_nonfinite_positions": torch.nonzero(~mask, as_tuple=False)[:16].reshape(-1).tolist()}


def run_exp055_transigen_checkpoint8_replay_cpu(cfg: dict[str, Any], config_sha: str,
        attempt_id: str, device_name: str, checkpoint_path: Path) -> dict[str, Any]:
    """Read-only TRAIN replay of the frozen epoch-8 checkpoint and failed eval batch."""
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    if (EXPERIMENT_ID != "EXP-055" or RUN_ID != EXP055_TRANSIGEN_CPU_REPLAY_RUN_ID
            or SOURCE_RUN_ID != DEFAULT_SOURCE_RUN_ID or attempt_id != "fit_attempt_001"
            or device_name != "cpu" or checkpoint_path.resolve() != EXP055_TRANSIGEN_CPU_RESUME_SOURCE_CHECKPOINT.resolve()):
        raise ValueError("checkpoint-8 replay is bound to one immutable CPU run and source checkpoint")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("checkpoint replay requires CUDA_VISIBLE_DEVICES to be empty")
    if sha256_file(checkpoint_path) != EXP055_TRANSIGEN_CPU_RESUME_SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("source epoch-8 checkpoint SHA differs from the registered immutable value")
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise FileExistsError(f"checkpoint replay run directory is not empty: {RUN_DIR}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    attempt = new_attempt_path(RUN_DIR, attempt_id)
    attempt.mkdir(parents=True, exist_ok=False)
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    torch.set_num_threads(1)
    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError("checkpoint replay unexpectedly initialized CUDA")
    device = torch.device("cpu")
    verified_gate = _reuse_root_verified_exp055_train_gate()
    train_keys, values, feature_by_key, data_evidence = _load_exp055_trainpath_train_inputs_reusing_gate(
        verified_gate, SOURCE_RUN_DIR / "response_blind_conditions.tsv", SOURCE_RUN_DIR)
    if (len(train_keys) != 419 or values["X"].shape != (419, 978)
            or values["X_ctl"].shape != (419, 978)
            or train_keys.source_object_id.astype(str).nunique() != 110
            or train_keys.full_inchikey.astype(str).nunique() != 5):
        raise ValueError("read-only replay did not load exactly the frozen 419-by-978 TRAIN domain")
    features = np.stack([feature_by_key[key] for key in train_keys.full_inchikey.astype(str)]).astype(np.float32)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (payload.get("epoch") != 8 or payload.get("seed") != SEED
            or payload.get("objective") != "raw_sum" or payload.get("gradient_clipping") is not False
            or payload.get("learning_rate") != 1e-3 or payload.get("weight_decay") != 1e-5
            or payload.get("checkpoint_role") != "last_finite_completed_TRAIN_epoch"
            or set(payload.get("rng_state", {})) != {"python", "numpy", "torch_cpu", "dataloader_generator"}):
        raise RuntimeError("source checkpoint metadata or exact resume RNG contract differs")
    loader_generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(TensorDataset(torch.as_tensor(values["X_ctl"], dtype=torch.float32),
        torch.as_tensor(values["X"], dtype=torch.float32), torch.as_tensor(features, dtype=torch.float32),
        torch.arange(419, dtype=torch.int64)), batch_size=64, shuffle=True, drop_last=False,
        num_workers=0, generator=loader_generator)
    model, _ = _load_transigen(torch, device, SEED, features_dim=2048, seed_cuda=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    restore_check = _exp055_cpu_restore_rng_state(torch, loader_generator, payload["rng_state"])
    if not all(restore_check.values()) or not _torch_tree_exact(torch, model.state_dict(), payload["state_dict"]):
        raise RuntimeError("checkpoint model/RNG exact restore check failed")
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    model_before = copy.deepcopy(model.state_dict())
    checkpoint_sha_before = sha256_file(checkpoint_path)

    def run_mode(*, training: bool) -> dict[str, Any]:
        probe = copy.deepcopy(model) if training else model
        probe.eval()
        captures: list[dict[str, Any]] = []
        capture_active = {"value": False}
        first_bad_layer: list[str] = []
        handles = []
        for module_name, module in probe.named_modules():
            if not module_name:
                continue
            def hook(_module: Any, _inputs: Any, output: Any, module_name: str = module_name) -> None:
                if not capture_active["value"]:
                    return
                summary = _exp055_resume_tensor_summary(torch, module_name, output)
                if summary.get("nonfinite_n", 0) and not first_bad_layer:
                    first_bad_layer.append(module_name)
                captures.append(summary)
            handles.append(module.register_forward_hook(hook))
        original_sample = probe.sample_latent
        latent_records: list[dict[str, Any]] = []
        def capture_sample(mu: Any, logvar: Any) -> Any:
            std = logvar.mul(0.5).exp_()
            eps = torch.FloatTensor(std.size()).normal_().to(probe.dev)
            z = eps.mul_(std).add_(mu)
            if capture_active["value"]:
                latent_records.append({"mu": _exp055_resume_tensor_summary(torch, "mu", mu),
                    "logvar": _exp055_resume_tensor_summary(torch, "logvar", logvar),
                    "exp_half_logvar_std": _exp055_resume_tensor_summary(torch, "exp_half_logvar_std", std),
                    "epsilon": _exp055_resume_tensor_summary(torch, "epsilon", eps),
                    "sample": _exp055_resume_tensor_summary(torch, "sample", z)})
                if not first_bad_layer:
                    for key in ("exp_half_logvar_std", "sample"):
                        if next(item for item in latent_records[-1].values()
                                if isinstance(item, dict) and item.get("name") == key).get("nonfinite_n", 0):
                            first_bad_layer.append(f"sample_latent.{key}")
                            break
            return z
        probe.sample_latent = capture_sample
        x1_all = values["X_ctl"]
        x2_all = values["X"]
        replay_start = 0
        replay_stop = 256
        target_x1 = target_x2 = target_feat = None
        target_losses = None
        try:
            if not training:
                random.seed(SEED + 17); np.random.seed((SEED + 17) % (2 ** 32))
                torch.random.default_generator.manual_seed(SEED + 17)
            else:
                # The failed epoch's training pass had an independent dropout stream; use a
                # reproducible source-checkpoint state and restore all caller streams after.
                _exp055_cpu_restore_rng_state(torch, loader_generator, payload["rng_state"])
            with torch.no_grad():
                for start in range(replay_start, replay_stop, 64):
                    stop = start + 64
                    capture_active["value"] = start == 192
                    x1 = torch.as_tensor(x1_all[start:stop], dtype=torch.float32, device=device)
                    x2 = torch.as_tensor(x2_all[start:stop], dtype=torch.float32, device=device)
                    feat = torch.as_tensor(features[start:stop], dtype=torch.float32, device=device)
                    out = probe.forward(x1, feat)
                    x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred = out
                    z2, mu2, logvar2 = probe.encode_x2(x2)
                    x2_rec = probe.decode_x2(z2)
                    losses = probe.loss(x1, x1_rec, mu1, logvar1, x2, x2_rec,
                        mu2, logvar2, x2_pred, mu_pred, logvar_pred)
                    if start == 192:
                        target_x1, target_x2, target_feat = x1, x2, feat
                        target_losses = losses
                        captures.extend([
                            _exp055_resume_tensor_summary(torch, "forward.mu_pred", mu_pred),
                            _exp055_resume_tensor_summary(torch, "forward.logvar_pred", logvar_pred),
                            _exp055_resume_tensor_summary(torch, "forward.z2_pred_decoder_input", z2_pred),
                            _exp055_resume_tensor_summary(torch, "forward.x2_pred_decoder_output", x2_pred),
                            _exp055_resume_tensor_summary(torch, "forward.main_delta", x2_pred - x1),
                            _exp055_resume_tensor_summary(torch, "encode_x2.z", z2),
                            _exp055_resume_tensor_summary(torch, "encode_x2.mu", mu2),
                            _exp055_resume_tensor_summary(torch, "encode_x2.logvar", logvar2)])
            if target_x1 is None or target_losses is None:
                raise RuntimeError("exact failed TRAIN evaluation batch was not replayed")
            bn_before, _ = _trainpath_bn_snapshot(torch, probe, "replay_before_train_forward")
            bn_before_state = {name: tensor.detach().clone() for name, tensor in probe.named_buffers()}
            if training:
                captures.clear()
                latent_records.clear()
                first_bad_layer.clear()
                _exp055_cpu_restore_rng_state(torch, loader_generator, payload["rng_state"])
                probe.train()
                capture_active["value"] = True
                with torch.no_grad():
                    x1, x2, pred, losses = _transigen_forward_batch(
                        torch, probe, (target_x1, target_x2, target_feat), device)
                train_main = pred - x1
                training_result = {"main_delta": _exp055_resume_tensor_summary(torch, "train.main_delta", train_main),
                    "losses": [_exp055_resume_tensor_summary(torch, f"train.loss_{i}", value) for i, value in enumerate(losses)],
                    "loss_values": [safe_number(float(value.detach().cpu().item()))
                                    if bool(torch.isfinite(value).all()) else None for value in losses]}
                bn_after, _ = _trainpath_bn_snapshot(torch, probe, "replay_after_train_forward")
                # Restore the cloned training-mode BN buffers and prove rollback exactly.
                with torch.no_grad():
                    for name, tensor in probe.named_buffers():
                        tensor.copy_(bn_before_state[name])
                bn_rollback = _torch_tree_exact(torch, bn_before_state,
                    {name: tensor.detach() for name, tensor in probe.named_buffers()})
                training_result.update({"bn_before": bn_before, "bn_after": bn_after,
                    "bn_exact_rollback_pass": bn_rollback, "optimizer_update_count": 0,
                    "first_nonfinite_layer": first_bad_layer[0] if first_bad_layer else None})
            else:
                training_result = None
                bn_after = None
                bn_rollback = True
            return {"mode": "train_forward_only" if training else "eval",
                "batch_global_train_rows": [192, 256],
                "batch_condition_keys": train_keys.iloc[192:256].full_condition_key.astype(str).tolist(),
                "batch_source_object_ids": train_keys.iloc[192:256].source_object_id.astype(str).tolist(),
                "batch_full_inchikeys": train_keys.iloc[192:256].full_inchikey.astype(str).tolist(),
                "latent_records": latent_records, "module_outputs": captures,
                "first_nonfinite_layer": first_bad_layer[0] if first_bad_layer else None,
                "main_delta": next((item for item in captures if item.get("name") == "forward.main_delta"), None),
                "auxiliary_losses": [_exp055_resume_tensor_summary(torch, f"eval.loss_{i}", value)
                                      for i, value in enumerate(target_losses)],
                "training_forward": training_result,
                "model_state_unchanged_or_rollback_verified": _torch_tree_exact(
                    torch, model_before, probe.state_dict())}
        finally:
            probe.sample_latent = original_sample
            for handle in handles:
                handle.remove()

    caller_rng = _exp055_cpu_rng_state(torch, loader_generator)
    try:
        eval_result = run_mode(training=False)
        train_result = run_mode(training=True)
    finally:
        rng_restore = _exp055_cpu_restore_rng_state(torch, loader_generator, caller_rng)
    if not all(rng_restore.values()):
        raise RuntimeError("read-only replay did not restore caller RNG streams exactly")
    if not _torch_tree_exact(torch, model_before, model.state_dict()):
        raise RuntimeError("read-only replay changed original checkpoint model state")
    if not _torch_tree_exact(torch, optimizer_before, optimizer.state_dict()):
        raise RuntimeError("read-only replay changed original checkpoint optimizer state")
    checkpoint_sha_after = sha256_file(checkpoint_path)
    if checkpoint_sha_after != checkpoint_sha_before:
        raise RuntimeError("source epoch-8 checkpoint bytes changed during read-only replay")
    elapsed_wall, elapsed_cpu = time.perf_counter() - started_wall, time.process_time() - started_cpu
    def tensor_leaf_count(value: Any) -> int:
        if torch.is_tensor(value):
            return 1
        if isinstance(value, dict):
            return sum(tensor_leaf_count(child) for child in value.values())
        if isinstance(value, (list, tuple)):
            return sum(tensor_leaf_count(child) for child in value)
        return 0

    result = {"status": "PASS_READ_ONLY_CHECKPOINT8_TRAIN_BATCH_REPLAY",
        "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID, "attempt_id": attempt_id,
        "runner_sha256": sha256_file(Path(__file__).resolve()), "config_sha256": config_sha,
        "source_checkpoint_path": str(checkpoint_path), "source_checkpoint_sha256_before": checkpoint_sha_before,
        "source_checkpoint_sha256_after": checkpoint_sha_after,
        "source_checkpoint_epoch": 8, "source_checkpoint_tensor_count": len(payload["state_dict"]),
        "source_optimizer_state_tensor_count": tensor_leaf_count(payload["optimizer_state_dict"]),
        "checkpoint_exact_model_optimizer_rng_restore": restore_check,
        "train_identity": data_evidence, "train_conditions": 419, "train_objects": 110,
        "full_inchikeys": 5, "genes": 978, "target_batch": "TRAIN[192:256]",
        "eval_replay": eval_result, "train_forward_only_replay": train_result,
        "optimizer_updates": 0, "source_checkpoint_modified": False,
        "inner_dev_numeric_rows_loaded": False, "pdo_outer_locked_values_opened": False,
        "cuda_used": False, "gpu_hours": 0.0,
        "cpu_wall_seconds": elapsed_wall, "cpu_process_seconds": elapsed_cpu,
        "cpu_cap_seconds": 600.0}
    if elapsed_wall > 600.0:
        result["status"] = "FAILED_CPU_TIMEBOX_EXCEEDED"
    atomic_json(attempt / "checkpoint8_replay_result.json", result)
    atomic_json(attempt / "checkpoint8_replay_close_receipt.json", {
        "status": result["status"], "run_id": RUN_ID, "attempt_id": attempt_id,
        "closed_at_utc": utc_now(), "runner_sha256": result["runner_sha256"],
        "config_sha256": config_sha, "source_checkpoint_sha256": checkpoint_sha_after,
        "cpu_wall_seconds": elapsed_wall, "cpu_process_seconds": elapsed_cpu, "gpu_hours": 0.0})
    return {**result, "result_path": str(attempt / "checkpoint8_replay_result.json"),
        "result_sha256": sha256_file(attempt / "checkpoint8_replay_result.json")}


def fit_phase_transigen_cpu_representative(cfg: dict[str, Any], config_sha: str,
                                            attempt_id: str, device_name: str,
                                            resume_from_checkpoint: Path | None = None) -> dict[str, Any]:
    """Train upstream TranSiGen on CPU with a first-epoch cost gate and fixed 300 endpoint."""
    run_clock_start = time.perf_counter()
    process_clock_start = time.process_time()
    started_at = utc_now()
    is_norm_recovery = RUN_ID == EXP055_TRANSIGEN_CPU_NORM_RECOVERY_RUN_ID
    is_exp059 = (EXPERIMENT_ID == "EXP-059"
                 and RUN_ID == EXP059_TRANSIGEN_LR_STABILITY_RUN_ID
                 and cfg.get("run_id") == RUN_ID)
    is_resume = RUN_ID in {EXP055_TRANSIGEN_CPU_RESUME_RUN_ID,
                           EXP055_TRANSIGEN_CPU_NORM_RECOVERY_RUN_ID}
    resume_epoch = 210 if is_norm_recovery else 8
    expected_run_id = (EXP059_TRANSIGEN_LR_STABILITY_RUN_ID if is_exp059
        else EXP055_TRANSIGEN_CPU_NORM_RECOVERY_RUN_ID if is_norm_recovery
        else EXP055_TRANSIGEN_CPU_RESUME_RUN_ID if is_resume
        else EXP055_TRANSIGEN_CPU_REPRESENTATIVE_RUN_ID)
    expected_resume_checkpoint = (EXP055_TRANSIGEN_CPU_NORM_SOURCE_CHECKPOINT if is_norm_recovery
        else EXP055_TRANSIGEN_CPU_RESUME_SOURCE_CHECKPOINT)
    if (EXPERIMENT_ID not in {"EXP-055", "EXP-059"} or RUN_ID != expected_run_id
            or SOURCE_RUN_ID != DEFAULT_SOURCE_RUN_ID or attempt_id != "fit_attempt_001"
            or device_name != "cpu" or (resume_from_checkpoint is not None) != is_resume):
        raise ValueError("CPU representative/resume fit is bound to its immutable run/attempt and CPU device")
    authorization = cfg.get("authorization", {})
    learning_rate = 1e-4 if is_exp059 else 1e-3
    cpu_total_cap_seconds = float(authorization.get("cpu_total_cap_minutes", 0.0)) * 60.0
    if (authorization.get("training_device_policy") != "cpu_only_zero_gpu"
            or authorization.get("allowed_device_uuid") != GPU_UUID
            or authorization.get("gpu_hours_cap") != 0.0 or cpu_total_cap_seconds != 7200.0
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""):
        raise RuntimeError("CPU representative/resume fit requires its exact 120-minute, zero-GPU configuration")
    if is_exp059 and (cfg.get("model", {}).get("learning_rate") != learning_rate
            or float(authorization.get("prior_cpu_charged_seconds", -1.0)) <= 0):
        raise RuntimeError("EXP-059 must preserve its single 1e-4 candidate and cumulative CPU ledger")
    resume_preflight = None
    resume_cost_gate_data = None
    contract_path = None
    contract_check_seconds = 0.0
    prior_failed_contract_check_allowance_seconds = 0.0
    if is_exp059:
        parent_evidence_path, parent_evidence_sha = resolve_exp059_parent_evidence(cfg)
        contract_check_path = RUN_DIR / "cpu_entry_contract_check.json"
        preflight_path = RUN_DIR / "exp059_fresh_preflight.json"
        root_preflight_path = RUN_DIR / "root_entry_preflight_readback.json"
        if not all(path.is_file() for path in (contract_check_path, preflight_path, root_preflight_path)):
            raise FileNotFoundError("EXP-059 CPU contract check, fresh preflight, and root readback are required")
        contract_check = json.loads(contract_check_path.read_text(encoding="utf-8"))
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        root_preflight = json.loads(root_preflight_path.read_text(encoding="utf-8"))
        contract_check_sha = sha256_file(contract_check_path)
        contract_path = preflight_path
        contract_sha = sha256_file(preflight_path)
        runner_sha_now = sha256_file(Path(__file__).resolve())
        prior_cpu_charged_seconds = float(authorization.get("prior_cpu_charged_seconds", -1.0))
        if (contract_check.get("status") != "PASS_CPU_ENTRY_SHAPE_SAVE_AND_FAILURE_PROPAGATION_NO_TRAIN_UPDATE"
                or contract_check.get("experiment_id") != "EXP-059"
                or contract_check.get("run_id") != RUN_ID
                or contract_check.get("attempt_id") != attempt_id
                or contract_check.get("optimizer_updates") != 0
                or contract_check.get("runner_sha256") != runner_sha_now
                or contract_check.get("config_sha256") != config_sha
                or preflight.get("status") != "PASS_EXP059_FRESH_PREFLIGHT"
                or preflight.get("experiment_id") != "EXP-059"
                or preflight.get("run_id") != RUN_ID
                or preflight.get("attempt_id") != attempt_id
                or preflight.get("runner_sha256") != runner_sha_now
                or preflight.get("config_sha256") != config_sha
                or preflight.get("cpu_contract_check_sha256") != contract_check_sha
                or preflight.get("cpu_contract_root_readback_sha256") !=
                    sha256_file(RUN_DIR / "root_cpu_contract_check_readback.json")
                or preflight.get("mechanism_replay_close_sha256") !=
                    "154309323e746a505683dd18da7cfa4e9c1c526ee4832a6b988e3d3f68780e43"
                or preflight.get("mechanism_replay_root_readback_sha256") !=
                    "a59d5e4eeb29fc172f7dd7abc98f806d3dcdaaf85b0952dd19cd16c764cd093b"
                or preflight.get("parent_checkpoint_path") != str(parent_evidence_path.relative_to(ROOT))
                or preflight.get("parent_checkpoint_sha256") != parent_evidence_sha
                or root_preflight.get("status") != "PASS_ROOT_EXP059_PREFLIGHT_READBACK"
                or root_preflight.get("preflight_sha256") != contract_sha
                or root_preflight.get("runner_sha256") != runner_sha_now
                or root_preflight.get("config_sha256") != config_sha
                or root_preflight.get("cpu_contract_check_sha256") != contract_check_sha
                or root_preflight.get("cpu_contract_root_readback_sha256") !=
                    sha256_file(RUN_DIR / "root_cpu_contract_check_readback.json")
                or root_preflight.get("parent_checkpoint_path") != str(parent_evidence_path.relative_to(ROOT))
                or root_preflight.get("parent_checkpoint_sha256") != parent_evidence_sha
                or preflight.get("parent_checkpoint_loaded_into_candidate") is not False
                or contract_check.get("parent_checkpoint_loaded_into_candidate") is not False
                or preflight.get("optimizer_updates") != 0
                or preflight.get("cuda_used") is not False
                or preflight.get("cuda_visible_devices") != ""
                or preflight.get("inner_dev_numeric_rows_loaded") is not False
                or preflight.get("pdo_outer_locked_values_opened") is not False
                or not math.isclose(prior_cpu_charged_seconds,
                    float(preflight.get("prior_cpu_charged_seconds", math.nan)), rel_tol=0.0, abs_tol=1e-6)):
            raise RuntimeError("EXP-059 preflight is stale, incomplete, or differs from its immutable evidence")
        contract_check_seconds = 0.0
        prior_failed_contract_check_allowance_seconds = 0.0
        cumulative_before_resume = float(preflight.get("cumulative_cpu_before_fit_seconds", math.inf))
        if cumulative_before_resume >= cpu_total_cap_seconds:
            raise TimeoutError("EXP-059 cumulative CPU budget is already exhausted")
    if is_resume:
        if resume_from_checkpoint.resolve() != expected_resume_checkpoint.resolve():
            raise ValueError("resume checkpoint path differs from the registered immutable checkpoint source")
    if is_norm_recovery:
        shadow_root = EXP055_TRANSIGEN_CPU_NORM_ROOT_SHADOW_READBACK
        if (not shadow_root.is_file()
                or sha256_file(shadow_root) != EXP055_TRANSIGEN_CPU_NORM_ROOT_SHADOW_READBACK_SHA256):
            raise RuntimeError("root-verified norm-overflow shadow replay is missing or changed")
        shadow = json.loads(shadow_root.read_text(encoding="utf-8"))
        required_shadow_checks = {
            "exact_parent_checkpoint_sha_before_after", "four_real_updates_recorded",
            "batch4_condition_keys_match_failed_context", "batch4_canonical_losses_match_failed_context",
            "all_replayed_grad_elements_finite", "all_replayed_post_update_model_optimizer_bn_finite",
            "legacy_float32_global_norm_overflowed", "safe_float64_global_norm_finite",
            "inner_dev_pdo_outer_locked_unopened", "cuda_not_used"}
        if (shadow.get("status") != "PASS_ROOT_NORM_OVERFLOW_REPLAY_GATE"
                or shadow.get("source_checkpoint_sha256_before") != EXP055_TRANSIGEN_CPU_NORM_SOURCE_CHECKPOINT_SHA256
                or shadow.get("source_checkpoint_sha256_after") != EXP055_TRANSIGEN_CPU_NORM_SOURCE_CHECKPOINT_SHA256
                or not all(shadow.get("checks", {}).get(key) is True for key in required_shadow_checks)):
            raise RuntimeError("root norm-overflow replay gate does not satisfy the authorized recovery condition")
    if is_resume:
        replay_attempt = (ROOT / "artifacts/experiments/EXP-055" /
            f"{EXP055_TRANSIGEN_CPU_REPLAY_RUN_ID}/fit_attempt_001")
        preflight_path = RUN_DIR / ("cpu_norm_recovery_preflight_receipt.json" if is_norm_recovery
                                    else "cpu_resume_preflight_receipt.json")
        root_preflight_path = RUN_DIR / ("root_cpu_norm_recovery_preflight_readback.json" if is_norm_recovery
                                         else "root_cpu_resume_preflight_readback.json")
        if is_norm_recovery:
            replay_result_path = (ROOT / "artifacts/experiments/EXP-055/20261007_exp055_cpu_norm_shadow_v2/"
                                  "diag_attempt_001/norm_shadow_result.json")
            replay_close_path = replay_result_path.with_name("norm_shadow_close_receipt.json")
            if not (replay_result_path.is_file() and replay_close_path.is_file()
                    and preflight_path.is_file() and root_preflight_path.is_file()):
                raise FileNotFoundError("immutable norm-shadow result, close, and fresh root-reviewed preflight are required")
            replay_result = json.loads(replay_result_path.read_text(encoding="utf-8"))
            replay_close = json.loads(replay_close_path.read_text(encoding="utf-8"))
            root_preflight = json.loads(root_preflight_path.read_text(encoding="utf-8"))
            replay_result_sha = sha256_file(replay_result_path)
            replay_close_sha = sha256_file(replay_close_path)
            expected_shadow_sha = EXP055_TRANSIGEN_CPU_NORM_ROOT_SHADOW_READBACK_SHA256
            if (root_preflight.get("status") != "PASS_ROOT_CPU_NORM_RECOVERY_PREFLIGHT"
                    or root_preflight.get("preflight_receipt_sha256") != sha256_file(preflight_path)
                    or root_preflight.get("runner_sha256") != sha256_file(Path(__file__).resolve())
                    or root_preflight.get("config_sha256") != config_sha
                    or root_preflight.get("source_checkpoint_sha256") != EXP055_TRANSIGEN_CPU_NORM_SOURCE_CHECKPOINT_SHA256
                    or root_preflight.get("root_shadow_readback_sha256") != expected_shadow_sha
                    or root_preflight.get("shadow_result_sha256") != replay_result_sha
                    or root_preflight.get("shadow_close_sha256") != replay_close_sha
                    or root_preflight.get("prior_failed_preflight_receipt_sha256")
                        != (sha256_file(RUN_DIR / "cpu_norm_recovery_preflight_attempt_001_failure.json")
                            if (RUN_DIR / "cpu_norm_recovery_preflight_attempt_001_failure.json").is_file() else None)):
                raise RuntimeError("root CPU norm-recovery preflight receipt is stale or mismatched")
            resume_preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            runner_sha_now = sha256_file(Path(__file__).resolve())
            if (resume_preflight.get("status") != "PASS_CPU_NORM_RECOVERY_PREFLIGHT"
                    or resume_preflight.get("runner_sha256") != runner_sha_now
                    or resume_preflight.get("config_sha256") != config_sha
                    or resume_preflight.get("source_checkpoint_sha256") != EXP055_TRANSIGEN_CPU_NORM_SOURCE_CHECKPOINT_SHA256
                    or resume_preflight.get("root_shadow_readback_sha256") != expected_shadow_sha
                    or resume_preflight.get("prior_failed_preflight_receipt_sha256")
                        != (sha256_file(RUN_DIR / "cpu_norm_recovery_preflight_attempt_001_failure.json")
                            if (RUN_DIR / "cpu_norm_recovery_preflight_attempt_001_failure.json").is_file() else None)
                    or resume_preflight.get("fixed_endpoint_epoch") != 300
                    or resume_preflight.get("resume_from_epoch") != 210
                    or not math.isfinite(float(resume_preflight.get("cumulative_projected_cpu_with_20pct_margin_seconds", math.inf)))):
                raise RuntimeError("CPU norm-recovery preflight is stale, mismatched, or failed")
            prior_cpu_charged_seconds = float(authorization.get("prior_cpu_charged_seconds", -1.0))
            if prior_cpu_charged_seconds <= 0 or prior_cpu_charged_seconds >= cpu_total_cap_seconds:
                raise RuntimeError("cumulative CPU prior charge is invalid")
            root_preflight_cpu = float(root_preflight.get("cpu_process_seconds", math.inf))
            preflight_cpu = float(resume_preflight.get("preflight_cpu_process_seconds", math.inf))
            preflight_wall = float(resume_preflight.get("preflight_cpu_wall_seconds", math.inf))
            startup_allowance = float(root_preflight.get("additional_preflight_startup_allowance_seconds", math.inf))
            cumulative_before_resume = (prior_cpu_charged_seconds + root_preflight_cpu + preflight_cpu
                                       + preflight_wall + startup_allowance)
            projected_remaining = float(resume_preflight.get("remaining_schedule_with_margin_seconds", math.inf))
            projected_cumulative_total = cumulative_before_resume + projected_remaining
            if (not math.isfinite(projected_remaining) or projected_remaining <= 0
                    or projected_cumulative_total > cpu_total_cap_seconds):
                raise TimeoutError("measured CPU schedule projection plus 20% margin exceeds the fixed cumulative cap")
            resume_cost_gate_data = {"status": "PASS_CPU_NORM_RECOVERY_FIXED300_WITHIN_CAP",
                "run_id": RUN_ID, "attempt_id": attempt_id,
                "source_checkpoint_sha256": EXP055_TRANSIGEN_CPU_NORM_SOURCE_CHECKPOINT_SHA256,
                "source_checkpoint_epoch": 210,
                "shadow_root_readback_sha256": expected_shadow_sha,
                "fresh_preflight_sha256": sha256_file(preflight_path),
                "root_preflight_readback_sha256": sha256_file(root_preflight_path),
                "prior_cpu_charged_seconds": prior_cpu_charged_seconds,
                "cpu_preflight_process_seconds": preflight_cpu,
                "root_preflight_readback_process_seconds": root_preflight_cpu,
                "preflight_wall_seconds": preflight_wall,
                "preflight_startup_allowance_seconds": startup_allowance,
                "cumulative_cpu_charged_before_formal_attempt_seconds": cumulative_before_resume,
                "remaining_fixed_schedule_projected_with_20pct_margin_seconds": projected_remaining,
                "cumulative_projected_cpu_with_20pct_margin_seconds": projected_cumulative_total,
                "cpu_total_cap_seconds": cpu_total_cap_seconds,
                "gpu_hours": 0.0}
            contract_path = preflight_path
            contract_sha = sha256_file(preflight_path)
            contract_check_seconds = preflight_cpu
            prior_failed_contract_check_allowance_seconds = 0.0
        else:
            replay_result_path = replay_attempt / "checkpoint8_replay_result.json"
            root_replay_path = replay_attempt / "root_independent_readback_v2.json"
            preflight_path = RUN_DIR / "cpu_resume_preflight_receipt.json"
            if not (replay_result_path.is_file() and root_replay_path.is_file() and preflight_path.is_file()):
                raise FileNotFoundError("checkpoint-8 replay and root readback must pass before resume")
            replay_result = json.loads(replay_result_path.read_text(encoding="utf-8"))
            root_replay = json.loads(root_replay_path.read_text(encoding="utf-8"))
            resume_preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            runner_sha_now = sha256_file(Path(__file__).resolve())
            if (replay_result.get("status") != "PASS_READ_ONLY_CHECKPOINT8_TRAIN_BATCH_REPLAY"
                    or replay_result.get("runner_sha256") != EXP055_TRANSIGEN_CPU_REPLAY_RUNNER_SHA256
                    or replay_result.get("config_sha256") != config_sha
                    or replay_result.get("source_checkpoint_sha256_after") != EXP055_TRANSIGEN_CPU_RESUME_SOURCE_CHECKPOINT_SHA256
                    or root_replay.get("status") != "PASS_ROOT_READ_ONLY_REPLAY_ARTIFACT_AND_STATE"
                    or root_replay.get("replay_result_sha256") != sha256_file(replay_result_path)
                    or resume_preflight.get("status") != "PASS_CPU_EXACT_RESUME_PREFLIGHT"
                    or resume_preflight.get("runner_sha256") != runner_sha_now
                    or resume_preflight.get("config_sha256") != config_sha
                    or resume_preflight.get("source_checkpoint_sha256") != EXP055_TRANSIGEN_CPU_RESUME_SOURCE_CHECKPOINT_SHA256):
                raise RuntimeError("checkpoint replay/root readback preflight is stale, mismatched, or failed")
            prior_cpu_charged_seconds = float(authorization.get("prior_cpu_charged_seconds", -1.0))
            if not math.isclose(prior_cpu_charged_seconds, 92.741, rel_tol=0.0, abs_tol=0.001):
                raise RuntimeError("cumulative CPU package prior charge differs from the closed representative attempt")
            cumulative_before_resume = (prior_cpu_charged_seconds
                + float(replay_result.get("cpu_wall_seconds", math.inf))
                + EXP055_TRANSIGEN_CPU_PRIOR_FAILED_ROOT_PREFLIGHT_WALL_ALLOWANCE_SECONDS
                + float(resume_preflight.get("cpu_process_seconds", math.inf)))
            old_cost_gate_path = (ROOT / "artifacts/experiments/EXP-055/20261007_exp055_transigen_cpu_representative_v1/"
                "fit_attempt_001/cpu_cost_gate.json")
            old_cost_gate = json.loads(old_cost_gate_path.read_text(encoding="utf-8"))
            projected_full_schedule_seconds = float(old_cost_gate.get("projected_fixed300_with_20pct_margin_seconds", math.inf))
            if old_cost_gate.get("status") != "PASS_FIXED300_WITHIN_CPU_CAP" or not math.isfinite(projected_full_schedule_seconds):
                raise RuntimeError("the previously measured fixed-300 CPU schedule cost gate is unavailable")
            projected_cumulative_total = cumulative_before_resume + projected_full_schedule_seconds
            if projected_cumulative_total > cpu_total_cap_seconds:
                raise RuntimeError("cumulative CPU budget cannot cover checkpoint resume and fixed-300 schedule projection")
            resume_cost_gate_data = {"status": "PASS_CUMULATIVE_CPU_FIXED300_RESUME_WITHIN_CAP",
                "run_id": RUN_ID, "attempt_id": attempt_id,
                "source_checkpoint_sha256": EXP055_TRANSIGEN_CPU_RESUME_SOURCE_CHECKPOINT_SHA256,
                "original_fixed300_cost_gate_path": str(old_cost_gate_path),
                "original_fixed300_cost_gate_sha256": sha256_file(old_cost_gate_path),
                "original_projected_fixed300_with_20pct_margin_seconds": projected_full_schedule_seconds,
                "previous_closed_attempt_cpu_seconds": prior_cpu_charged_seconds,
                "checkpoint_replay_cpu_wall_seconds": float(replay_result.get("cpu_wall_seconds", math.inf)),
                "prior_failed_root_readback_wall_allowance_seconds":
                    EXP055_TRANSIGEN_CPU_PRIOR_FAILED_ROOT_PREFLIGHT_WALL_ALLOWANCE_SECONDS,
                "root_preflight_cpu_process_seconds": float(resume_preflight.get("cpu_process_seconds", math.inf)),
                "cumulative_cpu_charged_before_resume_seconds": cumulative_before_resume,
                "cumulative_projected_cpu_with_fixed300_seconds": projected_cumulative_total,
                "cpu_total_cap_seconds": cpu_total_cap_seconds,
                "remaining_cpu_budget_seconds_before_resume": cpu_total_cap_seconds-cumulative_before_resume,
                "margin_already_in_original_projection": True,
                "gpu_hours": 0.0}
            contract_sha = sha256_file(preflight_path)
            contract_path = preflight_path
            contract_check_seconds = float(resume_preflight.get("cpu_process_seconds", 0.0))
            prior_failed_contract_check_allowance_seconds = (
                float(replay_result.get("cpu_wall_seconds", 0.0))
                + EXP055_TRANSIGEN_CPU_PRIOR_FAILED_ROOT_PREFLIGHT_WALL_ALLOWANCE_SECONDS)
    elif not is_exp059:
        contract_path = RUN_DIR / "cpu_entry_contract_check.json"
        if not contract_path.is_file():
            raise FileNotFoundError("the no-update CPU model/save contract check must pass before formal fitting")
        contract_sha = sha256_file(contract_path)
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract_check_seconds = float(contract.get("contract_check_cpu_wall_seconds", 0.0))
        prior_failed_contract_check_allowance_seconds = 40.0
        prior_cpu_charged_seconds = float(authorization.get("prior_cpu_charged_seconds", 0.0))
        cumulative_before_resume = (prior_cpu_charged_seconds + contract_check_seconds
                                    + prior_failed_contract_check_allowance_seconds)
        contract_check_runner_sha = sha256_file(Path(__file__).resolve())
        if (contract.get("status") != "PASS_CPU_ENTRY_SHAPE_SAVE_AND_FAILURE_PROPAGATION_NO_TRAIN_UPDATE"
                or contract.get("runner_sha256") != contract_check_runner_sha
                or contract.get("config_sha256") != config_sha
                or contract.get("optimizer_updates") != 0
                or contract.get("injected_write_failure_propagated") is not True):
            raise RuntimeError("the no-update CPU model/save contract check is stale, mismatched, or failed")
    # Enforce the cumulative envelope rather than giving a resumed process a new 120 minutes.
    cpu_cap_seconds = cpu_total_cap_seconds - cumulative_before_resume
    if cpu_cap_seconds <= 0:
        raise TimeoutError("EXP-055 cumulative CPU package has no remaining runtime")
    runner_sha = sha256_file(Path(__file__).resolve())
    if is_exp059:
        allowed_run_files = EXP059_CPU_RUN_ROOT_ALLOWED_ENTRIES
    elif is_norm_recovery:
        allowed_run_files = {"cpu_norm_recovery_preflight_receipt.json",
                             "root_cpu_norm_recovery_preflight_readback.json",
                             "cpu_norm_recovery_preflight_attempt_001_failure.json"}
    elif is_resume:
        allowed_run_files = {"cpu_resume_preflight_receipt.json",
                             "root_cpu_resume_preflight_readback.json"}
    else:
        allowed_run_files = {"cpu_entry_contract_check.json", "cpu_contract_checkpoint.pt"}
    if RUN_DIR.exists() and any(path.name not in allowed_run_files for path in RUN_DIR.iterdir()):
        raise FileExistsError("CPU representative run has unexpected files before its one formal fit attempt")
    if (RUN_DIR / attempt_id).exists():
        raise FileExistsError("CPU representative attempt is immutable and may not be reopened")

    attempt = new_attempt_path(RUN_DIR, attempt_id)
    attempt.mkdir(parents=False, exist_ok=False)
    if is_resume:
        atomic_json(attempt / "cpu_cost_gate.json", resume_cost_gate_data)
    process_id = os.getpid()
    selected_config = (EXP059_TRANSIGEN_LR_STABILITY_CONFIG if is_exp059
        else EXP055_TRANSIGEN_CPU_NORM_RECOVERY_CONFIG if is_norm_recovery
        else EXP055_TRANSIGEN_CPU_RESUME_CONFIG if is_resume
        else EXP055_TRANSIGEN_CPU_REPRESENTATIVE_CONFIG)
    command = ("CUDA_VISIBLE_DEVICES='' PYTHONPATH=src /home/dell/miniconda3/bin/conda run "
        "--no-capture-output -n drugscreening-gpu python "
        f"scripts/modeling/exp051_contract_fit_smoke.py --config configs/experiments/{selected_config} "
        f"--experiment-id {EXPERIMENT_ID} --fit-kind representative --models transigen "
        f"--phase fit --device cpu --run-id {RUN_ID} --source-run-id {SOURCE_RUN_ID} --attempt-id {attempt_id}"
        + (f" --resume-from-checkpoint {resume_from_checkpoint}" if is_resume else ""))
    progress_path, ledger_path = attempt / "attempt_progress.json", attempt / "resource_ledger.jsonl"
    curve_path, batch_trace_path = attempt / "TranSiGen_training_curve.jsonl", attempt / "TranSiGen_training_batches.jsonl"
    completed_epochs = resume_epoch if is_resume else 0
    last_finite_checkpoint: Path | None = None
    model = optimizer = loader_generator = None
    failure_context: dict[str, Any] | None = None
    atomic_json(attempt / "fit_smoke_manifest.json", {
        "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID, "fit_attempt": attempt_id,
        "status": "CPU_FIT_STARTED", "started_at_utc": started_at,
        "wsl_pid": process_id, "runner_sha256": runner_sha, "config_sha256": config_sha,
        "command": command, "python_executable": sys.executable,
        "device": "cpu", "cuda_visible_devices": "", "gpu_hours_cap": 0.0,
        "approved_gpu_uuid_recorded_but_not_used": GPU_UUID,
        "cpu_total_cap_minutes": 120.0,
        "formal_attempt_wall_cap_seconds_after_cumulative_prior_charges": cpu_cap_seconds,
        "measured_successful_contract_check_seconds": contract_check_seconds,
        "prior_failed_no_update_contract_check_allowance_seconds": prior_failed_contract_check_allowance_seconds,
        "cpu_contract_check_path": str(contract_path), "cpu_contract_check_sha256": contract_sha,
        "resume_from_checkpoint": str(resume_from_checkpoint) if is_resume else None,
        "resume_from_checkpoint_sha256": sha256_file(resume_from_checkpoint) if is_resume else None,
        "resume_checkpoint_epoch": resume_epoch if is_resume else None,
        "cumulative_cpu_charged_before_attempt_seconds": cumulative_before_resume,
        "resume_preflight_receipt_sha256": sha256_file(contract_path) if is_resume else None,
        "root_resume_preflight_readback_sha256": sha256_file(RUN_DIR / "root_cpu_norm_recovery_preflight_readback.json")
            if is_norm_recovery else sha256_file(RUN_DIR / "root_cpu_resume_preflight_readback.json") if is_resume else None,
        "source_run_id": SOURCE_RUN_ID,
        "training_contract": {"architecture": "upstream original TranSiGen",
            "initialization": f"exact epoch-{resume_epoch} checkpoint model+Adam+BN+RNG continuation" if is_resume else "fresh author initialization from scratch", "seed": SEED,
            "train_conditions": 419, "train_objects": 110, "full_inchikey_count": 5,
            "genes": 978, "objective": "raw_sum", "gradient_clipping": False,
            "learning_rate": (1e-4 if is_exp059 else 1e-3), "weight_decay": 1e-5, "batch_size": 64,
            "batch_sizes": [64, 64, 64, 64, 64, 64, 35], "shuffle": True,
            "batchnorm": "normal TRAIN batches", "target_epochs": 300,
            "diagnostic_epoch": 100, "main_endpoint_epoch": 300,
            "inner_dev": "load only after epoch-300 checkpoint freeze; final contract evaluation only"},
        "device_comparability": "CPU/GPU float and RNG trajectories may differ; no same-trajectory claim",
        "data_roles": {"train_only_numeric_loaded_before_epoch_300": True,
            "inner_dev_numeric_rows_loaded_before_checkpoint_freeze": False,
            "pdo_outer_locked_values_opened": False},
        "resource_contract": {"cpu_total_wall_cap_seconds": cpu_total_cap_seconds,
            "formal_attempt_wall_cap_after_contract_checks_seconds": cpu_cap_seconds,
            "measured_contract_check_seconds": contract_check_seconds,
            "prior_failed_contract_check_allowance_seconds": prior_failed_contract_check_allowance_seconds,
            "gpu_hours_cap": 0.0, "gpu_uuid_used": None}})
    append_jsonl(ledger_path, {"event": "CPU_ATTEMPT_STARTED", "at_utc": started_at,
        "wsl_pid": process_id, "device": "cpu", "cuda_visible_devices": "",
        "approved_gpu_uuid_recorded_but_not_used": GPU_UUID, "gpu_hours": 0.0,
        "cpu_wall_cap_seconds": cpu_cap_seconds, "cpu_total_wall_cap_seconds": cpu_total_cap_seconds,
        "runner_sha256": runner_sha,
        "config_sha256": config_sha, "cpu_contract_check_sha256": contract_sha})
    starting_epoch = resume_epoch if is_resume else 0
    atomic_json(progress_path, {"phase": "CPU_NORM_RECOVERY_STARTED_TRAIN_INPUT_GATE" if is_norm_recovery
        else "CPU_RESUME_STARTED_TRAIN_INPUT_GATE" if is_resume else "CPU_STARTED_TRAIN_INPUT_GATE",
        "updated_at_utc": utc_now(), "wsl_pid": process_id, "completed_epochs": starting_epoch,
        "target_epochs": 300, "inner_dev_numeric_rows_loaded": False,
        "gpu_hours": 0.0, "elapsed_cpu_wall_seconds": time.perf_counter()-run_clock_start})

    def elapsed_wall() -> float:
        return time.perf_counter() - run_clock_start

    def budget_check() -> None:
        if elapsed_wall() >= cpu_cap_seconds:
            raise TimeoutError(f"EXP-055 CPU run reached its 120-minute cap at {elapsed_wall():.1f}s")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("CUDA_VISIBLE_DEVICES changed during the CPU-only attempt")

    def close_failed(error: BaseException) -> None:
        elapsed, process_elapsed = elapsed_wall(), time.process_time() - process_clock_start
        close = {"status": "FAILED_CLOSED_CPU_ATTEMPT", "run_id": RUN_ID,
            "attempt_id": attempt_id, "started_at_utc": started_at, "closed_at_utc": utc_now(),
            "wsl_pid": process_id, "runner_sha256": runner_sha, "config_sha256": config_sha,
            "exception_type": type(error).__name__, "exception_message": str(error),
            "last_completed_finite_epoch": completed_epochs,
            "last_finite_checkpoint": last_finite_checkpoint.name if last_finite_checkpoint and last_finite_checkpoint.is_file() else None,
            "last_finite_checkpoint_sha256": sha256_file(last_finite_checkpoint) if last_finite_checkpoint and last_finite_checkpoint.is_file() else None,
            "first_failure_context_path": "first_failure_context.json" if failure_context is not None else None,
            "cpu_elapsed_wall_seconds": elapsed, "cpu_process_seconds": process_elapsed,
            "cpu_cap_seconds": cpu_cap_seconds, "cpu_total_cap_seconds": cpu_total_cap_seconds,
            "cpu_elapsed_total_with_contract_checks_and_allowance_seconds":
                cumulative_before_resume + elapsed,
            "measured_successful_contract_check_seconds": contract_check_seconds,
            "prior_failed_no_update_contract_check_allowance_seconds": prior_failed_contract_check_allowance_seconds,
            "gpu_hours": 0.0,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "inner_dev_numeric_rows_loaded": False, "pdo_outer_locked_values_opened": False}
        if failure_context is not None and not (attempt / "first_failure_context.json").exists():
            atomic_json(attempt / "first_failure_context.json", failure_context)
        atomic_json(attempt / "attempt_close_receipt.json", close)
        append_jsonl(ledger_path, {"event": "CPU_ATTEMPT_FAILED_CLOSED", "at_utc": utc_now(), **close})
        atomic_json(progress_path, {"phase": "CLOSED_FAILED", "updated_at_utc": utc_now(),
            "wsl_pid": process_id, "completed_epochs": completed_epochs,
            "cpu_elapsed_wall_seconds": elapsed, "gpu_hours": 0.0,
            "exception_type": type(error).__name__, "exception_message": str(error)})

    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        torch.set_num_threads(1)
        if torch.cuda.is_available() or torch.cuda.is_initialized():
            raise RuntimeError("formal TranSiGen CPU run unexpectedly sees or initialized CUDA")
        device = torch.device("cpu")
        verified_gate = _reuse_root_verified_exp055_train_gate()
        train_keys, values, feature_by_key, data_evidence = _load_exp055_trainpath_train_inputs_reusing_gate(
            verified_gate, SOURCE_RUN_DIR / "response_blind_conditions.tsv", SOURCE_RUN_DIR)
        features = np.stack([feature_by_key[key] for key in
                             train_keys.full_inchikey.astype(str)]).astype(np.float32, copy=False)
        train_y = values["delta"].astype(np.float32, copy=False)
        if (len(train_keys) != 419 or train_keys.source_object_id.astype(str).nunique() != 110
                or train_keys.full_inchikey.astype(str).nunique() != 5
                or values["X"].shape != (419, 978) or values["X_ctl"].shape != (419, 978)
                or train_y.shape != (419, 978) or features.shape != (419, 2048)
                or not np.isfinite(train_y).all() or not np.isfinite(features).all()):
            raise ValueError("CPU TranSiGen input gate did not yield 419 TRAIN rows, 110 objects, 5 drugs, 978 genes")
        gene_variance = np.var(train_y.astype(np.float64), axis=0, ddof=0)
        zero_train = np.zeros_like(train_y, dtype=np.float32)
        per_drug_train = np.empty_like(train_y, dtype=np.float32)
        for drug_key in sorted(train_keys.full_inchikey.astype(str).unique()):
            pos = np.flatnonzero(train_keys.full_inchikey.astype(str).to_numpy() == drug_key)
            per_drug_train[pos] = train_y[pos].mean(axis=0, dtype=np.float64).astype(np.float32)
        baseline_train_metrics = {"zero_change": keyed_metrics(train_keys, train_y, zero_train, gene_variance),
            "per_drug_mean_delta": keyed_metrics(train_keys, train_y, per_drug_train, gene_variance)}
        atomic_json(attempt / "train_baseline_metrics.json", {
            "status": "PASS_TRAIN_ONLY_BASELINES", "run_id": RUN_ID,
            "train_conditions": 419, "train_objects": 110, "full_inchikeys": 5,
            "genes": 978, "metrics": baseline_train_metrics,
            "normalizer_fit_on": "419 TRAIN delta rows only"})
        setup_seconds = elapsed_wall()
        budget_check()

        dataset = TensorDataset(torch.as_tensor(values["X_ctl"], dtype=torch.float32),
            torch.as_tensor(values["X"], dtype=torch.float32), torch.as_tensor(features, dtype=torch.float32),
            torch.arange(419, dtype=torch.int64))
        loader_generator = torch.Generator().manual_seed(SEED)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False,
            num_workers=0, generator=loader_generator)
        batch_sizes = [int(batch[0].shape[0]) for batch in loader]
        loader_generator.manual_seed(SEED)
        if batch_sizes != [64, 64, 64, 64, 64, 64, 35]:
            raise ValueError(f"CPU TranSiGen loader batch sizes changed: {batch_sizes}")
        loss_names = ("raw_total_loss", "mse_x1", "mse_x2", "mse_pert",
                      "kld_x1", "kld_x2", "kld_pert")
        initialization_start = time.perf_counter()
        model, _ = _load_transigen(torch, device, SEED, features_dim=2048, seed_cuda=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        initialization_seconds = time.perf_counter() - initialization_start
        last_finite_checkpoint = attempt / "TranSiGen_last_finite.pt"
        if is_resume:
            source_checkpoint_sha_before = sha256_file(resume_from_checkpoint)
            source_payload = torch.load(resume_from_checkpoint, map_location="cpu", weights_only=True)
            expected_source_sha = (EXP055_TRANSIGEN_CPU_NORM_SOURCE_CHECKPOINT_SHA256 if is_norm_recovery
                                   else EXP055_TRANSIGEN_CPU_RESUME_SOURCE_CHECKPOINT_SHA256)
            if (source_checkpoint_sha_before != expected_source_sha
                    or source_payload.get("epoch") != resume_epoch):
                raise RuntimeError("resume checkpoint changed or is not the registered immutable source")
            model.load_state_dict(source_payload["state_dict"], strict=True)
            optimizer.load_state_dict(source_payload["optimizer_state_dict"])
            restore_check = _exp055_cpu_restore_rng_state(torch, loader_generator, source_payload["rng_state"])
            model_restored = _torch_tree_exact(torch, model.state_dict(), source_payload["state_dict"])
            optimizer_restored = _torch_tree_exact(torch, optimizer.state_dict(), source_payload["optimizer_state_dict"])
            params_finite, bad_parameter = _lean_monitor_finite_named_tensors(
                torch, [(name, tensor) for name, tensor in model.state_dict().items()])
            optimizer_bad = _first_nonfinite_optimizer_state(torch, optimizer)
            if (not all(restore_check.values()) or not model_restored or not optimizer_restored
                    or not params_finite or optimizer_bad is not None):
                raise RuntimeError(f"checkpoint resume state verification failed: rng={restore_check}, "
                    f"model={model_restored}, optimizer={optimizer_restored}, bad_parameter={bad_parameter}, "
                    f"bad_optimizer={optimizer_bad}")
            _, initial_checkpoint_seconds = _exp055_cpu_save_checkpoint(
                torch, last_finite_checkpoint, model, optimizer, loader_generator,
                epoch=resume_epoch, checkpoint_role=f"exact_resume_origin_epoch{resume_epoch}_model_optimizer_bn_rng",
                train_metrics=source_payload.get("train_metrics"), readback=True)
            if sha256_file(resume_from_checkpoint) != source_checkpoint_sha_before:
                raise RuntimeError("immutable source checkpoint bytes changed while restoring")
            completed_epochs = resume_epoch
            source_curve_path = (ROOT / "artifacts/experiments/EXP-055" /
                ("20261007_exp055_transigen_cpu_resume_v1" if is_norm_recovery else
                 "20261007_exp055_transigen_cpu_representative_v1") /
                "fit_attempt_001/TranSiGen_training_curve.jsonl")
            source_curve_rows = [json.loads(line) for line in source_curve_path.read_text(encoding="utf-8").splitlines()]
            prefix_rows = [row for row in source_curve_rows if 1 <= int(row.get("epoch", 0)) <= resume_epoch]
            if len(prefix_rows) != resume_epoch:
                raise RuntimeError(f"source attempt does not contain the exact first {resume_epoch} epoch history rows")
            for row in prefix_rows:
                append_jsonl(curve_path, {**row, "history_source_run_id":
                    (EXP055_TRANSIGEN_CPU_RESUME_RUN_ID if is_norm_recovery else EXP055_TRANSIGEN_CPU_REPRESENTATIVE_RUN_ID),
                    "history_source_attempt_id": "fit_attempt_001"})
        else:
            random.seed(SEED); np.random.seed(SEED % (2 ** 32)); torch.manual_seed(SEED)
            _, initial_checkpoint_seconds = _exp055_cpu_save_checkpoint(
                torch, last_finite_checkpoint, model, optimizer, loader_generator,
                epoch=0, checkpoint_role="fresh_author_initialization_before_first_update",
                train_metrics=None, readback=True)
        initial_checkpoint_sha = sha256_file(last_finite_checkpoint)
        initial_bn, _ = _trainpath_bn_snapshot(torch, model, f"resume_origin_epoch{resume_epoch}" if is_resume else "fresh_author_initialization")
        atomic_json(progress_path, {"phase": "CPU_READY_NORM_RECOVERY_RESUME" if is_norm_recovery
            else "CPU_READY_EXACT_EPOCH8_RESUME" if is_resume else "CPU_READY_FRESH_INITIALIZATION",
            "updated_at_utc": utc_now(), "wsl_pid": process_id, "completed_epochs": completed_epochs,
            "target_epochs": 300, "device": "cpu", "gpu_hours": 0.0,
            "data_setup_seconds": setup_seconds, "initialization_seconds": initialization_seconds,
            "initial_checkpoint_sha256": initial_checkpoint_sha,
            "source_checkpoint_sha256": sha256_file(resume_from_checkpoint) if is_resume else None,
            "next_epoch": completed_epochs + 1})
        atomic_json(attempt / "run_input_receipt.json", {
            "status": "PASS_TRAIN_ONLY_FROZEN_INPUTS", "run_id": RUN_ID,
            "runner_sha256": runner_sha, "config_sha256": config_sha,
            "source_gate": verified_gate, "train_identity": data_evidence,
            "train_conditions": int(len(train_keys)), "train_objects": int(train_keys.source_object_id.nunique()),
            "full_inchikeys": int(train_keys.full_inchikey.nunique()), "genes": 978,
            "train_condition_key_sha256": sha256_text("\n".join(train_keys.full_condition_key.astype(str)) + "\n"),
            "train_delta_sha256": hashlib.sha256(train_y.tobytes()).hexdigest(),
            "train_feature_matrix_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
            "inner_dev_numeric_rows_loaded": False, "pdo_outer_locked_values_opened": False,
            "cpu_contract_check_sha256": contract_sha,
            "resume_source_checkpoint_sha256": sha256_file(resume_from_checkpoint) if is_resume else None,
            "resumed_absolute_epoch": resume_epoch if is_resume else None,
            "exact_rng_restore_pass": restore_check if is_resume else None,
            "inner_dev_numeric_rows_loaded": False, "pdo_outer_locked_values_opened": False,
            "gpu_hours": 0.0})

        last_train_metrics: dict[str, Any] | None = (source_payload.get("train_metrics") if is_resume else None)
        last_train_prediction: np.ndarray | None = None
        for epoch in range(starting_epoch, 300):
            budget_check()
            epoch_index, epoch_start = epoch + 1, time.perf_counter()
            before_parameters = [parameter.detach().clone() for parameter in model.parameters()]
            model.train()
            loss_sums = np.zeros(7, dtype=np.float64)
            gradient_norms: list[float] = []
            seen_conditions = 0
            train_seconds_start = time.perf_counter()
            for batch_index, (x1_cpu, x2_cpu, feature_cpu, row_cpu) in enumerate(loader, start=1):
                budget_check()
                batch_start = time.perf_counter()
                row_positions = row_cpu.numpy().astype(np.int64, copy=False)
                x1, x2, batch_features = x1_cpu.to(device), x2_cpu.to(device), feature_cpu.to(device)
                rng_before_batch = _exp055_cpu_rng_state_json(torch, loader_generator)
                optimizer.zero_grad(set_to_none=True)
                exp059_capture: dict[str, Any] | None = {} if is_exp059 else None
                _, _, prediction, losses = _transigen_forward_batch(
                    torch, model, (x1, x2, batch_features), device,
                    diagnostic_capture=exp059_capture)
                values_loss = [float(loss.detach().cpu().item()) for loss in losses]
                main_delta = prediction - x1
                exp059_bn_after_forward = (_exp059_batchnorm_health(torch, model)
                                           if is_exp059 else None)
                if is_exp059 and exp059_bn_after_forward is not None and not exp059_bn_after_forward["finite"]:
                    failure_context = {"status": "FIRST_NONFINITE_BATCHNORM_AFTER_TRAIN_FORWARD",
                        "epoch": epoch_index, "batch_index": batch_index,
                        "condition_keys": train_keys.iloc[row_positions].full_condition_key.astype(str).tolist(),
                        "source_object_ids": train_keys.iloc[row_positions].source_object_id.astype(str).tolist(),
                        "batchnorm_health_after_forward": exp059_bn_after_forward,
                        "loss_names": list(loss_names), "loss_values": [safe_number(value) for value in values_loss],
                        "kld_pert_elementwise_decomposition": _exp059_kld_pert_decomposition(
                            torch, exp059_capture or {}),
                        "main_delta_finite_n": int(torch.isfinite(main_delta).sum().item()),
                        "main_delta_denominator": int(main_delta.numel()),
                        "rng_before_batch": rng_before_batch,
                        "last_finite_checkpoint": last_finite_checkpoint.name,
                        "last_finite_checkpoint_sha256": sha256_file(last_finite_checkpoint)}
                    raise FloatingPointError(
                        f"TranSiGen first non-finite BatchNorm buffer after forward at epoch {epoch_index}, batch {batch_index}")
                if len(losses) != 7 or not all(math.isfinite(value) for value in values_loss):
                    component = next((name for name, value in zip(loss_names, values_loss)
                                      if not math.isfinite(value)), "loss_term_count")
                    failure_context = {"status": "FIRST_NONFINITE_TRAINING_LOSS",
                        "epoch": epoch_index, "batch_index": batch_index,
                        "condition_keys": train_keys.iloc[row_positions].full_condition_key.astype(str).tolist(),
                        "source_object_ids": train_keys.iloc[row_positions].source_object_id.astype(str).tolist(),
                        "loss_names": list(loss_names), "loss_values": [safe_number(value) for value in values_loss],
                        "first_nonfinite_component": component, "main_delta_shape": list(main_delta.shape),
                        "main_delta_finite_n": int(torch.isfinite(main_delta).sum().item()),
                        "main_delta_denominator": int(main_delta.numel()), "rng_before_batch": rng_before_batch,
                        "batchnorm_health_after_forward": exp059_bn_after_forward,
                        "kld_pert_elementwise_decomposition": (_exp059_kld_pert_decomposition(
                            torch, exp059_capture or {}) if is_exp059 else None),
                        "last_finite_checkpoint": last_finite_checkpoint.name,
                        "last_finite_checkpoint_sha256": sha256_file(last_finite_checkpoint)}
                    raise FloatingPointError(f"TranSiGen non-finite training loss at epoch {epoch_index}, batch {batch_index}")
                if prediction.shape != x1.shape or not bool(torch.isfinite(main_delta).all().item()):
                    bad = ~torch.isfinite(main_delta)
                    failure_context = {"status": "FIRST_NONFINITE_TRAINING_MAIN_DELTA",
                        "epoch": epoch_index, "batch_index": batch_index,
                        "condition_keys": train_keys.iloc[row_positions].full_condition_key.astype(str).tolist(),
                        "source_object_ids": train_keys.iloc[row_positions].source_object_id.astype(str).tolist(),
                        "prediction_shape": list(prediction.shape), "main_delta_shape": list(main_delta.shape),
                        "main_delta_finite_n": int(torch.isfinite(main_delta).sum().item()),
                        "main_delta_denominator": int(main_delta.numel()),
                        "first_nonfinite_positions": torch.nonzero(bad, as_tuple=False)[:16].cpu().tolist(),
                        "rng_before_batch": rng_before_batch,
                        "last_finite_checkpoint": last_finite_checkpoint.name,
                        "last_finite_checkpoint_sha256": sha256_file(last_finite_checkpoint)}
                    raise FloatingPointError(f"TranSiGen non-finite training main delta at epoch {epoch_index}, batch {batch_index}")
                losses[0].backward()
                gradient_finite, gradient_norm, bad_gradient, gradient_norm_finite = _lean_monitor_gradient_summary(torch, model)
                if not gradient_finite:
                    failure_context = {"status": "FIRST_NONFINITE_TRAIN_GRADIENT_ELEMENT",
                        "epoch": epoch_index, "batch_index": batch_index,
                        "condition_keys": train_keys.iloc[row_positions].full_condition_key.astype(str).tolist(),
                        "source_object_ids": train_keys.iloc[row_positions].source_object_id.astype(str).tolist(),
                        "gradient_elements_finite": gradient_finite,
                        "gradient_norm_finite": gradient_norm_finite,
                        "gradient_norm": safe_number(gradient_norm), "first_nonfinite_parameter_gradient": bad_gradient,
                        "losses": dict(zip(loss_names, values_loss)), "rng_before_batch": rng_before_batch,
                        "last_finite_checkpoint": last_finite_checkpoint.name,
                        "last_finite_checkpoint_sha256": sha256_file(last_finite_checkpoint)}
                    raise FloatingPointError(f"TranSiGen non-finite gradient at epoch {epoch_index}, batch {batch_index}")
                optimizer.step()
                optimizer_bad = _first_nonfinite_optimizer_state(torch, optimizer)
                parameters_finite, bad_parameter = _lean_monitor_finite_named_tensors(
                    torch, [(name, parameter) for name, parameter in model.named_parameters()])
                bn_finite, bad_bn = _lean_monitor_finite_named_tensors(
                    torch, _lean_monitor_batchnorm_named_tensors(model))
                if optimizer_bad is not None or not parameters_finite or not bn_finite:
                    component = "optimizer_state" if optimizer_bad is not None else "parameter" if not parameters_finite else "batchnorm_state"
                    failure_context = {"status": "FIRST_NONFINITE_TRAIN_STATE", "epoch": epoch_index,
                        "batch_index": batch_index, "first_nonfinite_component": component,
                        "first_nonfinite_optimizer_state": optimizer_bad,
                        "first_nonfinite_parameter": bad_parameter, "first_nonfinite_batchnorm_buffer": bad_bn,
                        "condition_keys": train_keys.iloc[row_positions].full_condition_key.astype(str).tolist(),
                        "source_object_ids": train_keys.iloc[row_positions].source_object_id.astype(str).tolist(),
                        "losses": dict(zip(loss_names, values_loss)), "gradient_norm": gradient_norm,
                        "rng_before_batch": rng_before_batch,
                        "last_finite_checkpoint": last_finite_checkpoint.name,
                        "last_finite_checkpoint_sha256": sha256_file(last_finite_checkpoint)}
                    raise FloatingPointError(f"TranSiGen non-finite state after update at epoch {epoch_index}, batch {batch_index}")
                loss_sums += np.asarray(values_loss, dtype=np.float64)
                gradient_norms.append(gradient_norm if gradient_norm_finite else math.nan)
                seen_conditions += int(len(x1))
                append_jsonl(batch_trace_path, {"epoch": epoch_index, "batch_index": batch_index,
                    "conditions": int(len(x1)),
                    "condition_keys": train_keys.iloc[row_positions].full_condition_key.astype(str).tolist(),
                    "losses": dict(zip(loss_names, values_loss)),
                    "gradient_elements_finite": gradient_finite,
                    "gradient_norm_finite": gradient_norm_finite,
                    "gradient_norm": safe_number(gradient_norm),
                    "parameters_finite": parameters_finite, "optimizer_finite": optimizer_bad is None,
                    "batchnorm_finite": bn_finite, "batch_seconds": time.perf_counter()-batch_start,
                    "device": "cpu", "gpu_hours": 0.0})
                del x1, x2, batch_features, prediction, main_delta, losses
            train_seconds = time.perf_counter() - train_seconds_start
            if seen_conditions != 419 or len(gradient_norms) != 7:
                raise ValueError(f"TranSiGen epoch did not process exactly 419 rows in seven updates: {seen_conditions}")

            evaluation_start = time.perf_counter()
            eval_failures: list[dict[str, Any]] = []
            eval_hooks: list[Any] = []
            eval_layer_state: dict[str, Any] = {"first_nonfinite_layer": None}
            if is_resume or epoch_index in {1, 10}:
                eval_hooks, eval_layer_state = _trainpath_attach_eval_failure_hooks(torch, model)
            try:
                train_prediction, train_eval_losses = _transigen_predict_and_losses(
                    torch, model, values["X_ctl"], values["X"], features, 64, device,
                    SEED + 2 * epoch + 1, seed_cuda=False, allow_nonfinite_auxiliary=True,
                    allow_nonfinite_main=(is_resume or is_exp059),
                    progress_check=budget_check, nonfinite_main_delta_callback=eval_failures.append)
            except BaseException as error:
                if eval_failures:
                    failure_context = {"status": "FIRST_NONFINITE_TRAIN_EVALUATION_MAIN_DELTA",
                        "epoch": epoch_index, "evaluation": "TRAIN_ONLY",
                        "main_delta_failure": eval_failures[0],
                        "first_nonfinite_layer": eval_layer_state.get("first_nonfinite_layer"),
                        "train_condition_count": 419, "main_delta_denominator": 419 * 978,
                        "main_delta_nonfinite_n": eval_failures[0].get("nonfinite_n"),
                        "rng_before_evaluation": _exp055_cpu_rng_state_json(torch, loader_generator),
                        "last_finite_checkpoint": last_finite_checkpoint.name,
                        "last_finite_checkpoint_sha256": sha256_file(last_finite_checkpoint)}
                raise error
            finally:
                for handle in eval_hooks:
                    handle.remove()
            train_eval_seconds = time.perf_counter() - evaluation_start
            if train_prediction.shape != (419, 978):
                raise ValueError("full TRAIN evaluation did not preserve all 419 keyed conditions and 978 genes")
            main_eval_finite = bool(np.isfinite(train_prediction).all()) and not eval_failures
            if main_eval_finite:
                train_metric = keyed_metrics(train_keys, train_y, train_prediction, gene_variance)
            elif is_resume or is_exp059:
                failure = eval_failures[0] if eval_failures else {
                    "finite_n": int(np.isfinite(train_prediction).sum()),
                    "nonfinite_n": int(train_prediction.size - np.isfinite(train_prediction).sum()),
                    "numel": int(train_prediction.size), "batch_start": None, "batch_stop": None}
                train_metric = {"status": "NULL_NONFINITE_FULL_TRAIN_EVALUATION",
                    "object_macro_nmse": None, "object_macro_mae": None,
                    "object_macro_gene_correlation": None,
                    "object_macro_direction_agreement": None,
                    "object_macro_amplitude_ratio": None,
                    "effective_n": 0, "nonfinite_n": int(failure.get("nonfinite_n", 0)),
                    "denominator": 419 * 978, "first_nonfinite_layer": eval_layer_state.get("first_nonfinite_layer"),
                    "first_nonfinite_batch": [failure.get("batch_start"), failure.get("batch_stop")],
                    "metric_scope": "full 419x978 denominator; no finite-subset score reported"}
            else:
                train_metric = keyed_metrics(train_keys, train_y, train_prediction, gene_variance)
            eval_anomalies = {name: detail for name, detail in train_eval_losses.items()
                              if detail.get("reason") is not None} if train_eval_losses else {}
            if eval_anomalies and (is_resume or is_exp059 or epoch_index in {1, 10}):
                train_eval_losses["evaluation_anomaly_summary"] = {
                    "status": "AUXILIARY_EVALUATION_NONFINITE_RECORDED_NULL",
                    "first_nonfinite_component": next(iter(eval_anomalies)),
                    "first_nonfinite_layer": eval_layer_state.get("first_nonfinite_layer"),
                    "components": eval_anomalies, "condition_denominator": 419,
                    "main_delta_denominator": 419 * 978}
            parameter_norm = _param_global_norm(torch, model)
            update_l2 = _norm_difference(torch, before_parameters, model)
            if not math.isfinite(parameter_norm) or not math.isfinite(update_l2):
                failure_context = {"status": "NONFINITE_TRAIN_PARAMETER_DIAGNOSTIC",
                    "epoch": epoch_index, "parameter_global_l2_norm": safe_number(parameter_norm),
                    "parameter_update_l2": safe_number(update_l2),
                    "last_finite_checkpoint": last_finite_checkpoint.name,
                    "last_finite_checkpoint_sha256": sha256_file(last_finite_checkpoint)}
                raise FloatingPointError(f"TranSiGen non-finite parameter diagnostic at epoch {epoch_index}")
            bn_after, _ = _trainpath_bn_snapshot(torch, model, f"epoch_{epoch_index}_TRAIN_complete")
            if main_eval_finite:
                last_train_metrics, last_train_prediction = train_metric, train_prediction.copy()
            row = {"epoch": epoch_index, "epoch_index_zero_based": epoch,
                "objective": "raw_sum", "gradient_clipping": False,
                "learning_rate": learning_rate, "weight_decay": 1e-5, "batch_size": 64,
                "train_shuffle": True, "train_conditions": seen_conditions,
                "train_batch_sizes": batch_sizes,
                "train_loss_terms_mean_per_condition": {
                    name: safe_number(float(value / seen_conditions))
                    for name, value in zip(loss_names, loss_sums)},
                "train_evaluation_auxiliary_losses": train_eval_losses,
                "gradient_norm_mean": safe_number(float(np.mean(gradient_norms))),
                "gradient_norm_max": safe_number(float(np.max(gradient_norms))),
                "gradient_norm_n_updates": len(gradient_norms),
                "gradient_norm_finite_n_updates": sum(math.isfinite(value) for value in gradient_norms),
                "parameter_global_l2_norm": safe_number(parameter_norm),
                "parameter_update_l2": safe_number(update_l2), "batchnorm_after": bn_after,
                "train_metrics": train_metric,
                "train_prediction_max_abs": (float(np.max(np.abs(train_prediction)))
                    if main_eval_finite else None),
                "train_main_delta_finite": main_eval_finite,
                "train_main_delta_nonfinite_n": int(train_prediction.size - np.isfinite(train_prediction).sum()),
                "train_main_delta_denominator": int(train_prediction.size),
                "first_nonfinite_layer": eval_layer_state.get("first_nonfinite_layer"),
                "training_seconds": train_seconds, "train_evaluation_seconds": train_eval_seconds,
                "finite_training_and_main_delta": bool(main_eval_finite), "device": "cpu", "gpu_hours": 0.0}
            curve_write_start = time.perf_counter()
            append_jsonl(curve_path, row)
            curve_write_seconds = time.perf_counter() - curve_write_start
            _, last_checkpoint_seconds = _exp055_cpu_save_checkpoint(
                torch, last_finite_checkpoint, model, optimizer, loader_generator,
                epoch=epoch_index, checkpoint_role="last_finite_completed_TRAIN_epoch",
                train_metrics=train_metric, readback=(epoch_index in {1, 100, 300}))

            if epoch_index == 1 and not is_resume:
                endpoint_checkpoint = attempt / "TranSiGen_epoch_1_cost_gate_checkpoint.pt"
                _, endpoint_checkpoint_seconds = _exp055_cpu_save_checkpoint(
                    torch, endpoint_checkpoint, model, optimizer, loader_generator,
                    epoch=1, checkpoint_role="first_epoch_cost_gate_checkpoint_not_a_selected_model",
                    train_metrics=train_metric, readback=True)
                endpoint_prediction_readback = None
                prediction_io_seconds = 0.0
                if main_eval_finite:
                    endpoint_prediction = attempt / "TranSiGen_epoch_1_train_keyed_predictions.npz"
                    prediction_arrays = _exp055_cpu_prediction_arrays(train_keys, values["genes"], train_y,
                                                                       train_prediction, "train")
                    prediction_io_start = time.perf_counter()
                    atomic_npz(endpoint_prediction, **prediction_arrays)
                    endpoint_prediction_readback = _exp055_cpu_prediction_readback(endpoint_prediction, prediction_arrays)
                    prediction_io_seconds = time.perf_counter() - prediction_io_start
                progress_write_start = time.perf_counter()
                completed_epochs = 1
                atomic_json(progress_path, {"phase": "CPU_FIRST_FULL_EPOCH_COST_GATE_READY",
                    "updated_at_utc": utc_now(), "wsl_pid": process_id,
                    "completed_epochs": 1, "target_epochs": 300, "device": "cpu",
                    "gpu_hours": 0.0, "elapsed_cpu_wall_seconds": elapsed_wall(),
                    "train_metrics": train_metric,
                    "last_finite_checkpoint_sha256": sha256_file(last_finite_checkpoint),
                    "epoch_1_cost_gate_checkpoint_sha256": sha256_file(endpoint_checkpoint),
                    "inner_dev_numeric_rows_loaded": False})
                progress_write_seconds = time.perf_counter() - progress_write_start
                per_epoch_observed_seconds = (train_seconds + train_eval_seconds
                    + last_checkpoint_seconds + curve_write_seconds + progress_write_seconds)
                projected_endpoint_checkpoint_seconds = 3.0 * endpoint_checkpoint_seconds
                projected_endpoint_prediction_seconds = 2.0 * prediction_io_seconds
                projected_final_bundle_seconds = prediction_io_seconds * (512.0 / 419.0)
                projected_final_inner_dev_eval_seconds = max(30.0, train_eval_seconds * 0.5)
                projected_close_seconds = 60.0
                projected_without_margin = (contract_check_seconds + prior_failed_contract_check_allowance_seconds
                    + elapsed_wall() + 299.0 * per_epoch_observed_seconds
                    + projected_endpoint_checkpoint_seconds + projected_endpoint_prediction_seconds
                    + projected_final_bundle_seconds + projected_final_inner_dev_eval_seconds
                    + projected_close_seconds)
                projected_schedule_with_margin = projected_without_margin * 1.20
                projected_total_with_margin = (cumulative_before_resume + projected_schedule_with_margin
                    if is_exp059 else projected_schedule_with_margin)
                cost_gate = {"status": "PASS_FIXED300_WITHIN_CPU_CAP" if projected_total_with_margin <= cpu_total_cap_seconds
                    else "STOP_AFTER_EPOCH1_PROJECTED_FIXED300_EXCEEDS_CPU_CAP",
                    "run_id": RUN_ID, "attempt_id": attempt_id, "runner_sha256": runner_sha,
                    "config_sha256": config_sha, "cpu_contract_check_sha256": contract_sha,
                    "cpu_cap_seconds": cpu_total_cap_seconds,
                    "formal_attempt_wall_cap_seconds_after_contract_checks": cpu_cap_seconds,
                    "cumulative_cpu_charged_before_formal_attempt_seconds": cumulative_before_resume,
                    "measured_successful_contract_check_seconds": contract_check_seconds,
                    "prior_failed_no_update_contract_check_allowance_seconds": prior_failed_contract_check_allowance_seconds,
                    "gpu_hours": 0.0,
                    "actual_cpu_wall_seconds_at_gate": elapsed_wall(),
                    "actual_cpu_process_seconds_at_gate": time.process_time()-process_clock_start,
                    "first_epoch_segments_seconds": {"data_setup": setup_seconds,
                        "model_initialization": initialization_seconds,
                        "initial_epoch_zero_checkpoint_save_readback": initial_checkpoint_seconds,
                        "seven_batch_training": train_seconds,
                        "full_419_train_evaluation_and_metrics": train_eval_seconds,
                        "last_finite_checkpoint_save_readback": last_checkpoint_seconds,
                        "cost_gate_checkpoint_save_readback": endpoint_checkpoint_seconds,
                        "keyed_prediction_save_readback": prediction_io_seconds,
                        "curve_and_progress_io": curve_write_seconds + progress_write_seconds},
                    "projection_formula_seconds": {"successful_pre_run_contract_check_actual": contract_check_seconds,
                        "prior_failed_no_update_contract_checks_upper_allowance": prior_failed_contract_check_allowance_seconds,
                        "formal_setup_and_first_epoch_actual": elapsed_wall(),
                        "remaining_299_epoch_training_eval_last_finite_checkpoint_and_curve_progress":
                            299.0 * per_epoch_observed_seconds,
                        "fixed_100_300_final_endpoint_checkpoint_save_readbacks": projected_endpoint_checkpoint_seconds,
                        "fixed_100_and_300_keyed_prediction_io": projected_endpoint_prediction_seconds,
                        "final_419_train_plus_93_inner_dev_keyed_bundle_io": projected_final_bundle_seconds,
                        "post_freeze_93_inner_dev_evaluation_estimate": projected_final_inner_dev_eval_seconds,
                        "close_receipt_allowance": projected_close_seconds},
                    "projected_fixed300_schedule_without_margin_seconds": projected_without_margin,
                    "projected_fixed300_schedule_with_20pct_margin_seconds": projected_schedule_with_margin,
                    "projected_fixed300_with_20pct_margin_seconds": projected_total_with_margin,
                    "projected_fixed300_with_20pct_margin_minutes": projected_total_with_margin / 60.0,
                    "automatic_continuation": projected_total_with_margin <= cpu_total_cap_seconds,
                    "cost_projection_method": "actual first complete TRAIN epoch segment rates extrapolated over remaining 299 fixed epochs; extra 100/300 checkpoint and prediction I/O plus final frozen inner-dev evaluation, combined prediction write, 60-second close allowance, and 20% margin",
                    "fixed_epochs": 300, "diagnostic_epoch": 100, "main_endpoint_epoch": 300,
                    "train_conditions": 419, "train_objects": 110, "full_inchikeys": 5,
                    "genes": 978, "inner_dev_conditions_expected": 93,
                    "inner_dev_objects_expected": 24, "inner_dev_numeric_rows_loaded": False,
                    "train_main_delta_finite": bool(np.isfinite(train_prediction).all()),
                    "epoch_1_prediction_readback": endpoint_prediction_readback,
                    "last_finite_checkpoint_sha256": sha256_file(last_finite_checkpoint),
                    "cost_gate_checkpoint_sha256": sha256_file(endpoint_checkpoint),
                    "epoch_1_auxiliary_eval_anomalies": {
                        name: detail for name, detail in train_eval_losses.items()
                        if isinstance(detail, dict) and detail.get("reason") is not None}}
                atomic_json(attempt / "cpu_cost_gate.json", cost_gate)
                append_jsonl(ledger_path, {"event": "CPU_FIRST_EPOCH_COST_GATE", "at_utc": utc_now(),
                    "wsl_pid": process_id, "status": cost_gate["status"],
                    "actual_cpu_wall_seconds": elapsed_wall(),
                    "projected_fixed300_with_20pct_margin_seconds": projected_total_with_margin,
                    "cpu_cap_seconds": cpu_total_cap_seconds, "gpu_hours": 0.0,
                    "automatic_continuation": cost_gate["automatic_continuation"]})
                if not cost_gate["automatic_continuation"]:
                    summary = {"status": cost_gate["status"], "scientific_model_result_valid": False,
                        "epochs_completed": 1, "epochs_required": 300,
                        "train_metrics_epoch_1_diagnostic_only": train_metric,
                        "baseline_train_metrics": baseline_train_metrics,
                        "cost_gate_path": "cpu_cost_gate.json",
                        "cost_gate_sha256": sha256_file(attempt / "cpu_cost_gate.json"),
                        "inner_dev_numeric_rows_loaded": False,
                        "no_fixed300_comparison_score": True}
                    atomic_json(attempt / "cpu_attempt_result.json", summary)
                    close = {"status": "CLOSED_AFTER_EPOCH1_COST_GATE", "run_id": RUN_ID,
                        "attempt_id": attempt_id, "started_at_utc": started_at, "closed_at_utc": utc_now(),
                        "wsl_pid": process_id, "runner_sha256": runner_sha, "config_sha256": config_sha,
                        "completed_epochs": 1, "required_epochs": 300,
                        "cpu_elapsed_wall_seconds": elapsed_wall(),
                        "cpu_process_seconds": time.process_time()-process_clock_start,
                        "cpu_cap_seconds": cpu_cap_seconds, "cpu_total_cap_seconds": cpu_total_cap_seconds,
                        "cpu_elapsed_total_with_contract_checks_and_allowance_seconds": elapsed_wall()
                            + contract_check_seconds + prior_failed_contract_check_allowance_seconds,
                        "measured_successful_contract_check_seconds": contract_check_seconds,
                        "prior_failed_no_update_contract_check_allowance_seconds": prior_failed_contract_check_allowance_seconds,
                        "gpu_hours": 0.0,
                        "cost_gate_status": cost_gate["status"],
                        "inner_dev_numeric_rows_loaded": False,
                        "checkpoint_sha256": sha256_file(endpoint_checkpoint),
                        "prediction_sha256": endpoint_prediction_readback["sha256"]}
                    atomic_json(attempt / "attempt_close_receipt.json", close)
                    append_jsonl(ledger_path, {"event": "CPU_ATTEMPT_CLOSED_AFTER_COST_GATE",
                        "at_utc": utc_now(), "wsl_pid": process_id,
                        "cpu_elapsed_wall_seconds": elapsed_wall(), "gpu_hours": 0.0,
                        "completed_epochs": 1, "cost_gate_status": cost_gate["status"]})
                    atomic_json(progress_path, {"phase": "CLOSED_AFTER_EPOCH1_COST_GATE",
                        "updated_at_utc": utc_now(), "wsl_pid": process_id,
                        "completed_epochs": 1, "target_epochs": 300,
                        "cpu_elapsed_wall_seconds": elapsed_wall(), "gpu_hours": 0.0,
                        "scientific_model_result_valid": False})
                    return {"cpu_cost_gate": cost_gate, "attempt_result": summary}

            elif epoch_index in {100, 300}:
                endpoint_checkpoint = attempt / f"TranSiGen_epoch_{epoch_index}.pt"
                _, endpoint_seconds = _exp055_cpu_save_checkpoint(torch, endpoint_checkpoint,
                    model, optimizer, loader_generator, epoch=epoch_index,
                    checkpoint_role="fixed_epoch_100_diagnostic_only" if epoch_index == 100 else "fixed_epoch_300_main_endpoint",
                    train_metrics=train_metric, readback=True)
                endpoint_readback = None
                endpoint_prediction = None
                if main_eval_finite:
                    endpoint_prediction = attempt / f"TranSiGen_epoch_{epoch_index}_train_keyed_predictions.npz"
                    prediction_arrays = _exp055_cpu_prediction_arrays(train_keys, values["genes"], train_y,
                                                                       train_prediction, "train")
                    atomic_npz(endpoint_prediction, **prediction_arrays)
                    endpoint_readback = _exp055_cpu_prediction_readback(endpoint_prediction, prediction_arrays)
                atomic_json(attempt / f"epoch_{epoch_index}_endpoint_receipt.json", {
                    "checkpoint_seconds": endpoint_seconds, "prediction_readback": endpoint_readback,
                    "train_main_delta_finite": main_eval_finite,
                    "train_main_delta_nonfinite_n": int(train_prediction.size - np.isfinite(train_prediction).sum()),
                    "train_main_delta_denominator": int(train_prediction.size),
                    "first_nonfinite_layer": eval_layer_state.get("first_nonfinite_layer"),
                    "prediction_archive_written": endpoint_prediction is not None})
                if epoch_index == 300 and main_eval_finite:
                    final_checkpoint = attempt / "TranSiGen_final_epoch_300.pt"
                    _, final_checkpoint_seconds = _exp055_cpu_save_checkpoint(torch, final_checkpoint,
                        model, optimizer, loader_generator, epoch=300,
                        checkpoint_role="frozen_fixed_epoch_300_main_endpoint",
                        train_metrics=train_metric, readback=True)
                    atomic_json(attempt / "checkpoint_freeze_receipt.json", {
                        "status": "PASS_FROZEN_EPOCH_300_CHECKPOINT", "epoch": 300,
                        "checkpoint_path": final_checkpoint.name,
                        "checkpoint_sha256": sha256_file(final_checkpoint),
                        "epoch_300_checkpoint_sha256": sha256_file(endpoint_checkpoint),
                        "optimizer_updates": 300 * 7, "model_selection": "none",
                        "final_checkpoint_save_seconds": final_checkpoint_seconds})

            completed_epochs = epoch_index
            progress_start = time.perf_counter()
            atomic_json(progress_path, {"phase": "CPU_FIXED300_TRAINING",
                "updated_at_utc": utc_now(), "wsl_pid": process_id,
                "completed_epochs": completed_epochs, "target_epochs": 300,
                "elapsed_cpu_wall_seconds": elapsed_wall(), "gpu_hours": 0.0,
                "train_metrics": train_metric,
                "last_finite_checkpoint_sha256": sha256_file(last_finite_checkpoint),
                "inner_dev_numeric_rows_loaded": False})
            progress_write_seconds = time.perf_counter() - progress_start
            row["epoch_curve_write_seconds"] = curve_write_seconds
            row["epoch_progress_write_seconds"] = progress_write_seconds
            if epoch_index == 1 or epoch_index % 10 == 0 or epoch_index in {100, 300}:
                print(json.dumps({"stage": "EXP055_TRANSGEN_CPU_PROGRESS", "epoch": epoch_index,
                    "train_nmse": train_metric.get("object_macro_nmse"),
                    "elapsed_cpu_minutes": elapsed_wall()/60.0, "gpu_hours": 0.0}, ensure_ascii=False), flush=True)
            if (is_resume or is_exp059) and epoch_index in {100, 300} and not main_eval_finite:
                close_status = f"CLOSED_AT_FIXED_EPOCH{epoch_index}_NONFINITE_MAIN_DELTA"
                summary = {"status": close_status, "scientific_model_result_valid": False,
                    "epochs_completed": epoch_index, "epochs_required": 300,
                    "fixed_endpoint": epoch_index,
                    "train_metrics_at_endpoint": None,
                    "main_delta_nonfinite_n": int(train_prediction.size - np.isfinite(train_prediction).sum()),
                    "main_delta_denominator": int(train_prediction.size),
                    "first_nonfinite_layer": eval_layer_state.get("first_nonfinite_layer"),
                    "first_main_delta_failure": eval_failures[0] if eval_failures else None,
                    "inner_dev_numeric_rows_loaded": False,
                    "no_finite_subset_score_reported": True,
                    "no_valid_endpoint_score": True}
                atomic_json(attempt / "cpu_attempt_result.json", summary)
                close = {"status": close_status, "run_id": RUN_ID, "attempt_id": attempt_id,
                    "started_at_utc": started_at, "closed_at_utc": utc_now(), "wsl_pid": process_id,
                    "runner_sha256": runner_sha, "config_sha256": config_sha,
                    "completed_epochs": epoch_index, "required_epochs": 300,
                    "cpu_elapsed_wall_seconds": elapsed_wall(), "cpu_process_seconds": time.process_time()-process_clock_start,
                    "cpu_total_cap_seconds": cpu_total_cap_seconds,
                    "cumulative_cpu_elapsed_with_prior_seconds": cumulative_before_resume + elapsed_wall(),
                    "cpu_total_remaining_seconds": cpu_total_cap_seconds - cumulative_before_resume - elapsed_wall(),
                    "gpu_hours": 0.0, "cuda_visible_devices": "", "inner_dev_numeric_rows_loaded": False,
                    "pdo_outer_locked_values_opened": False, "checkpoint_sha256": sha256_file(last_finite_checkpoint),
                    "prediction_archive_written": False, "main_delta_nonfinite_n": summary["main_delta_nonfinite_n"],
                    "main_delta_denominator": summary["main_delta_denominator"]}
                atomic_json(attempt / "attempt_close_receipt.json", close)
                append_jsonl(ledger_path, {"event": "CPU_ATTEMPT_CLOSED_FIXED_ENDPOINT_NONFINITE",
                    "at_utc": utc_now(), "wsl_pid": process_id, "gpu_hours": 0.0, **close})
                atomic_json(progress_path, {"phase": close_status, "updated_at_utc": utc_now(),
                    "wsl_pid": process_id, "completed_epochs": epoch_index, "target_epochs": 300,
                    "cpu_elapsed_wall_seconds": elapsed_wall(), "gpu_hours": 0.0,
                    "scientific_model_result_valid": False})
                return {"attempt_result": summary, "close_receipt": close}
            del before_parameters

        if completed_epochs != 300 or last_train_prediction is None or model is None:
            raise RuntimeError("TranSiGen fixed 300-epoch CPU schedule did not reach its registered endpoint")
        frozen_path = attempt / "TranSiGen_final_epoch_300.pt"
        frozen_payload = torch.load(frozen_path, map_location="cpu", weights_only=True)
        if (frozen_payload.get("epoch") != 300
                or frozen_payload.get("checkpoint_role") != "frozen_fixed_epoch_300_main_endpoint"
                or not _torch_tree_exact(torch, model.state_dict(), frozen_payload["state_dict"])):
            raise RuntimeError("frozen epoch-300 checkpoint does not exactly match the final in-memory model")
        model.load_state_dict(frozen_payload["state_dict"], strict=True)
        model.eval()
        dev_keys, dev_values, dev_evidence = _load_exp055_inner_dev_after_checkpoint_freeze(
            train_keys, values, SOURCE_RUN_DIR)
        dev_features = _full_key_features(dev_keys, np.arange(len(dev_keys), dtype=np.int64), feature_by_key)
        dev_eval_failures: list[dict[str, Any]] = []
        dev_hooks, dev_layer_state = _trainpath_attach_eval_failure_hooks(torch, model)
        try:
            dev_prediction, dev_eval_losses = _transigen_predict_and_losses(
                torch, model, dev_values["X_ctl"], dev_values["X"], dev_features, 64, device,
                SEED + 602, seed_cuda=False, allow_nonfinite_auxiliary=True,
                progress_check=budget_check, nonfinite_main_delta_callback=dev_eval_failures.append)
        except BaseException as error:
            failure_context = {"status": "FAILED_FINAL_INNER_DEV_EVALUATION",
                "exception_type": type(error).__name__, "exception_message": str(error),
                "first_main_delta_failure": dev_eval_failures[0] if dev_eval_failures else None,
                "first_nonfinite_layer": dev_layer_state.get("first_nonfinite_layer"),
                "inner_dev_condition_denominator": 93 * 978, "inner_dev_conditions": 93,
                "inner_dev_objects": 24, "frozen_checkpoint": frozen_path.name,
                "frozen_checkpoint_sha256": sha256_file(frozen_path),
                "inner_dev_numeric_rows_loaded_after_checkpoint_freeze": True}
            raise error
        finally:
            for handle in dev_hooks:
                handle.remove()
        if dev_prediction.shape != (93, 978) or not np.isfinite(dev_prediction).all():
            failure_context = {"status": "FAILED_FINAL_INNER_DEV_MAIN_DELTA_NONFINITE",
                "finite_n": int(np.isfinite(dev_prediction).sum()), "denominator": int(dev_prediction.size),
                "first_nonfinite_layer": dev_layer_state.get("first_nonfinite_layer"),
                "checkpoint_sha256": sha256_file(frozen_path)}
            raise FloatingPointError("frozen TranSiGen epoch-300 inner-dev main delta is non-finite")
        dev_metric = keyed_metrics(dev_keys, dev_values["delta"], dev_prediction, gene_variance)
        matched_baselines = _exp055_train_and_dev_baselines(train_keys, train_y, dev_keys,
                                                            dev_values["delta"], gene_variance)
        final_train_fit_gate = _fit_gate(last_train_metrics, {"metrics": {
            name: {"train": metric} for name, metric in baseline_train_metrics.items()}})
        final_arrays = _exp055_cpu_prediction_arrays(train_keys, values["genes"], train_y,
                                                      last_train_prediction, "train")
        final_arrays.update(_exp055_cpu_prediction_arrays(dev_keys, values["genes"],
            dev_values["delta"], dev_prediction, "inner_dev"))
        final_prediction_path = attempt / "TranSiGen_final_keyed_predictions.npz"
        final_prediction_start = time.perf_counter()
        atomic_npz(final_prediction_path, **final_arrays)
        final_prediction_readback = _exp055_cpu_prediction_readback(final_prediction_path, final_arrays)
        final_prediction_io_seconds = time.perf_counter() - final_prediction_start
        metrics = {"TRAIN": {"model": last_train_metrics,
                    "zero_change_baseline": matched_baselines["metrics"]["zero_change"]["train"],
                    "per_drug_mean_delta_baseline": matched_baselines["metrics"]["per_drug_mean_delta"]["train"]},
            "inner_dev": {"model": dev_metric,
                    "zero_change_baseline": matched_baselines["metrics"]["zero_change"]["inner_dev"],
                    "per_drug_mean_delta_baseline": matched_baselines["metrics"]["per_drug_mean_delta"]["inner_dev"]}}
        curve_rows = [json.loads(line) for line in curve_path.read_text(encoding="utf-8").splitlines()]
        epoch100 = next((row for row in curve_rows if row.get("epoch") == 100), None)
        final_result = {"status": "COMPLETED_TRANSIGEN_FIXED_300_CPU_EPOCH_COMPARISON",
            "scientific_model_result_valid": True, "model": "upstream TranSiGen",
            "device": "cpu", "gpu_hours": 0.0,
            "device_comparability": "CPU/GPU RNG and floating-point trajectories differ; no same-trajectory reproduction claim",
            "epochs_completed": 300, "selected_epoch": None,
            "selection": "none; epoch 100 is diagnostic only and epoch 300 is the fixed main endpoint",
            "training_contract": {"initialization": (f"exact epoch-{resume_epoch} model+Adam+BN+RNG continuation"
                    if is_resume else "fresh author initialization"), "seed": SEED,
                "train_conditions": 419, "train_objects": 110, "full_inchikeys": 5,
                "genes": 978, "objective": "raw_sum", "gradient_clipping": False,
                "learning_rate": learning_rate, "weight_decay": 1e-5, "batch_size": 64,
                "shuffle": True, "normal_batchnorm_updates": True, "updates": 300 * 7},
            "metrics_by_role": metrics, "train_fit_gate": final_train_fit_gate,
            "epoch_100_train_metrics_diagnostic": epoch100.get("train_metrics") if epoch100 else None,
            "final_train_auxiliary_losses": curve_rows[-1].get("train_evaluation_auxiliary_losses"),
            "final_inner_dev_auxiliary_losses": dev_eval_losses,
            "prediction_path": final_prediction_path.name,
            "prediction_readback": final_prediction_readback,
            "post_checkpoint_inner_dev_input_readback": dev_evidence,
            "checkpoint_paths": (["source_epoch210_checkpoint_external", "source_epoch100_checkpoint_external",
                "TranSiGen_epoch_300.pt", "TranSiGen_final_epoch_300.pt", "TranSiGen_last_finite.pt"]
                if is_norm_recovery else ["source_epoch8_checkpoint_external", "TranSiGen_epoch_100.pt",
                "TranSiGen_epoch_300.pt", "TranSiGen_final_epoch_300.pt", "TranSiGen_last_finite.pt"]
                if is_resume else ["TranSiGen_epoch_1_cost_gate_checkpoint.pt", "TranSiGen_epoch_100.pt",
                "TranSiGen_epoch_300.pt", "TranSiGen_final_epoch_300.pt", "TranSiGen_last_finite.pt"]),
            "checkpoint_sha256": ({"source_epoch210_checkpoint_external": EXP055_TRANSIGEN_CPU_NORM_SOURCE_CHECKPOINT_SHA256,
                    "source_epoch100_checkpoint_external": sha256_file(EXP055_TRANSIGEN_CPU_NORM_EPOCH100_CHECKPOINT),
                    **{name: sha256_file(attempt / name) for name in
                    ("TranSiGen_epoch_300.pt", "TranSiGen_final_epoch_300.pt", "TranSiGen_last_finite.pt")}}
                if is_norm_recovery else {"source_epoch8_checkpoint_external": EXP055_TRANSIGEN_CPU_RESUME_SOURCE_CHECKPOINT_SHA256,
                    **{name: sha256_file(attempt / name) for name in
                    ("TranSiGen_epoch_100.pt", "TranSiGen_epoch_300.pt", "TranSiGen_final_epoch_300.pt", "TranSiGen_last_finite.pt")}}
                if is_resume else {name: sha256_file(attempt / name) for name in
                ("TranSiGen_epoch_1_cost_gate_checkpoint.pt", "TranSiGen_epoch_100.pt",
                 "TranSiGen_epoch_300.pt", "TranSiGen_final_epoch_300.pt", "TranSiGen_last_finite.pt")}),
            "epoch_curve_sha256": sha256_file(curve_path),
            "batch_trace_sha256": sha256_file(batch_trace_path),
            "input_receipt_sha256": sha256_file(attempt / "run_input_receipt.json"),
            "cost_gate_sha256": sha256_file(attempt / "cpu_cost_gate.json"),
            "final_prediction_save_readback_seconds": final_prediction_io_seconds,
            "inner_dev_numeric_rows_loaded_after_epoch_300_checkpoint_freeze": True,
            "inner_dev_used_for_training_debug_or_model_selection": False,
            "pdo_outer_locked_values_opened": False}
        atomic_json(attempt / "transigen_cpu_full_fit_result.json", final_result)
        elapsed, process_elapsed = elapsed_wall(), time.process_time() - process_clock_start
        close = {"status": "COMPLETED_CPU_FIXED300", "run_id": RUN_ID,
            "attempt_id": attempt_id, "started_at_utc": started_at, "closed_at_utc": utc_now(),
            "wsl_pid": process_id, "runner_sha256": runner_sha, "config_sha256": config_sha,
            "completed_epochs": 300, "cpu_elapsed_wall_seconds": elapsed,
            "cpu_process_seconds": process_elapsed, "cpu_cap_seconds": cpu_cap_seconds,
            "cpu_total_cap_seconds": cpu_total_cap_seconds,
            "cumulative_cpu_charged_before_attempt_seconds": cumulative_before_resume,
            "cumulative_cpu_elapsed_seconds": cumulative_before_resume + elapsed,
            "cpu_elapsed_total_with_contract_checks_and_allowance_seconds":
                cumulative_before_resume + elapsed,
            "measured_successful_contract_check_seconds": contract_check_seconds,
            "prior_failed_no_update_contract_check_allowance_seconds": prior_failed_contract_check_allowance_seconds,
            "cpu_cap_remaining_seconds": cpu_cap_seconds - elapsed, "gpu_hours": 0.0,
            "cumulative_cpu_remaining_seconds": cpu_total_cap_seconds-cumulative_before_resume-elapsed,
            "cuda_visible_devices": "", "approved_gpu_uuid_recorded_but_not_used": GPU_UUID,
            "prediction_sha256": sha256_file(final_prediction_path),
            "checkpoint_sha256": sha256_file(frozen_path),
            "inner_dev_numeric_rows_loaded_after_checkpoint_freeze": True,
            "pdo_outer_locked_values_opened": False}
        atomic_json(attempt / "attempt_close_receipt.json", close)
        append_jsonl(ledger_path, {"event": "CPU_ATTEMPT_COMPLETED", "at_utc": utc_now(),
            "wsl_pid": process_id, "cpu_elapsed_wall_seconds": elapsed,
            "cpu_process_seconds": process_elapsed, "cpu_cap_seconds": cpu_cap_seconds,
            "gpu_hours": 0.0, "completed_epochs": 300,
            "close_receipt": "attempt_close_receipt.json"})
        atomic_json(progress_path, {"phase": "CLOSED_COMPLETED_CPU_FIXED300",
            "updated_at_utc": utc_now(), "wsl_pid": process_id,
            "completed_epochs": 300, "cpu_elapsed_wall_seconds": elapsed,
            "cpu_process_seconds": process_elapsed, "gpu_hours": 0.0})
        return {"full_fit": final_result, "close_receipt": close}
    except BaseException as error:
        close_failed(error)
        raise


def preflight_exp055_trainpath_recovery(cfg: dict[str, Any], config_sha: str,
                                         attempt_id: str) -> dict[str, Any]:
    """CPU-only run-safety/resource gate; reuse the existing root-verified input gates."""
    if (EXPERIMENT_ID != "EXP-055" or RUN_ID != EXP055_TRANSIGEN_FIXED300_LEAN_RUN_ID
            or attempt_id != "fit_attempt_001" or SOURCE_RUN_ID != DEFAULT_SOURCE_RUN_ID):
        raise ValueError("lean TranSiGen fixed300 preflight is bound to its new immutable run fit_attempt_001")
    if cfg.get("authorization", {}).get("allowed_device_uuid") != GPU_UUID:
        raise ValueError("EXP-055 registered device UUID differs from the sole approved A5000")
    limits = resource_limits(cfg)
    if limits["total_exp_cap_gpu_hours"] != 48.0 or limits["package_cap_gpu_hours"] != 10.0:
        raise ValueError("frozen EXP-055/global caps differ from 10/48 GPUh")
    if RUN_DIR.exists():
        existing = sorted(path.name for path in RUN_DIR.iterdir())
        if existing:
            raise FileExistsError(f"EXP-055 recovery run directory must be empty before preflight: {existing}")
    attempt_path = RUN_DIR / attempt_id
    if attempt_path.exists():
        raise FileExistsError(f"refusing to reopen a closed or pre-existing recovery attempt: {attempt_path}")
    prior_charge, prior_close_hashes = exp055_transigen_lean_monitor_settled_charge(
        include_cost_calibration=True)
    verified_gate = _reuse_root_verified_exp055_train_gate()
    reservation = 1.0 - prior_charge
    accounting = resource_accounting(EXPERIMENT_ROOT, RUN_ID, 1.0,
        limits["total_exp_cap_gpu_hours"], limits["exp_settled_before_artifact_ledgers_gpu_hours"],
        package_settled_before_artifact_ledgers_gpu_hours=prior_charge)
    if (accounting["package_remaining_gpu_hours"] + 1e-12 < reservation
            or accounting["exp_remaining_gpu_hours"] + 1e-12 < reservation):
        raise RuntimeError("CPU preflight cannot establish the full one-GPU-hour attempt reservation")
    receipt_path = RUN_DIR / "run_preflight_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"refusing to replace the immutable CPU preflight: {receipt_path}")
    device_snapshot = _device_snapshot()
    runner_sha = sha256_file(Path(__file__).resolve())
    upstream_model = TRANSIGEN_ROOT / "model.py"
    if not upstream_model.is_file():
        raise FileNotFoundError(f"official TranSiGen model source is absent: {upstream_model}")
    receipt = {
        "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID, "attempt_id": attempt_id,
        "status": "PASS_CPU_TRAIN_IDENTITY_BUDGET_AND_DEVICE_GATE_GPU_NOT_STARTED",
        "created_at_utc": utc_now(), "runner_sha256": runner_sha,
        "config_path": str(ROOT / "configs/experiments/EXP-055-representative-models.yaml"),
        "config_sha256": config_sha,
        "official_transigen_model_sha256": sha256_file(upstream_model),
        "source_run_id": SOURCE_RUN_ID,
        "reused_root_verified_train_input_gate": verified_gate,
        "resource_accounting_before_attempt": accounting,
        "prior_recovery_attempts": {"run_ids": [
            "20261007_exp055_transigen_trainpath_recovery_v4",
            "20261007_exp055_transigen_trainpath_recovery_v5",
            "20261007_exp055_transigen_trainpath_recovery_v9",
            "20261007_exp055_transigen_trainpath_recovery_v11",
            EXP055_TRANSIGEN_COST_SOURCE_RUN_ID,
            EXP055_TRANSIGEN_LEAN_COST_RUN_ID],
            "charged_gpu_hours": prior_charge,
            "failure_close_receipt_sha256": prior_close_hashes,
            "remaining_same_package_cap_gpu_hours": reservation},
        "attempt_gpu_cap_hours": reservation,
        "device_snapshot_before_launch_check": device_snapshot,
        "gpu_compute_pid_query_deferred_to_single_fit_launch_gate": True,
        "training_contract": {"architecture": "upstream original TranSiGen",
            "objective": "raw-sum/no-clip", "learning_rate": 1e-3, "weight_decay": 1e-5,
            "batch_size": 64, "shuffle": True,
            "batchnorm": "ordinary TRAIN batches only", "full_schedule_epochs": 300,
            "fixed_epoch_100_diagnostic": True,
            "per_epoch_train_evaluation": True,
            "lean_monitoring": True,
            "checkpoint_policy": "atomic last-finite model+optimizer checkpoint each completed epoch",
            "cost_margin_fraction": 0.20},
        "data_roles": {"inner_dev_numeric_rows_loaded": False,
            "frozen_train_inner_values_npz_opened": False,
            "pdo_numeric_sensitivity_or_expression_opened": False,
            "outer_or_locked_numeric_values_opened": False,
            "torch_imported": False, "cuda_used": False, "gpu_started": False},
        "attempt_directory_exists": False,
        "run_directory_empty_before_preflight_output": True,
        "closed_prior_attempts_reopened": False}
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path)}


def preflight_exp055_transigen_lean_cost_calibration(
        cfg: dict[str, Any], config_sha: str, attempt_id: str) -> dict[str, Any]:
    """Bind a fresh one-epoch TranSiGen cost calibration without initializing a model or CUDA."""
    if (EXPERIMENT_ID != "EXP-055" or RUN_ID != EXP055_TRANSIGEN_LEAN_COST_RUN_ID
            or attempt_id != "fit_attempt_001"
            or SOURCE_RUN_ID != DEFAULT_SOURCE_RUN_ID):
        raise ValueError("lean TranSiGen cost preflight requires its registered immutable run/attempt and frozen input run")
    if cfg.get("authorization", {}).get("allowed_device_uuid") != GPU_UUID:
        raise ValueError("EXP-055 registered device UUID differs from the sole approved A5000")
    if not math.isclose(float(cfg.get("authorization", {}).get("package_gpu_cap", -1.0)),
                        10.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("the frozen EXP-055 package cap is not 10 GPUh")
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise FileExistsError(f"lean calibration run directory must be empty before preflight: {RUN_DIR}")
    attempt_path = RUN_DIR / attempt_id
    if attempt_path.exists():
        raise FileExistsError(f"refusing to reopen a prior calibration attempt: {attempt_path}")
    receipt_path = RUN_DIR / "run_preflight_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"refusing to replace immutable cost calibration preflight: {receipt_path}")

    verified_gate = _reuse_root_verified_exp055_train_gate()
    if (verified_gate.get("checks_reused_count") != 21
            or len(verified_gate.get("checks_reused", {})) != 21
            or not all(value is True for value in verified_gate.get("checks_reused", {}).values())):
        raise RuntimeError("the existing root-verified 21/21 TRAIN input gate is unavailable")
    prior_charge, prior_close_hashes = exp055_transigen_lean_monitor_settled_charge()
    package_cap = 1.0
    calibration_cap = 0.05
    remaining = package_cap - prior_charge
    if remaining + 1e-12 < calibration_cap:
        raise RuntimeError("settled recovery balance cannot reserve the 0.05 GPUh calibration cap")
    limits = resource_limits(cfg)
    accounting = resource_accounting(EXPERIMENT_ROOT, RUN_ID, package_cap,
        limits["total_exp_cap_gpu_hours"],
        limits["exp_settled_before_artifact_ledgers_gpu_hours"], prior_charge)
    if (accounting["package_remaining_gpu_hours"] + 1e-12 < remaining
            or accounting["exp_remaining_gpu_hours"] + 1e-12 < calibration_cap):
        raise RuntimeError("resource ledger cannot reserve this calibration inside the existing recovery envelope")

    source_attempt = (ROOT / "artifacts/experiments/EXP-055" /
        f"{EXP055_TRANSIGEN_COST_SOURCE_RUN_ID}/{EXP055_TRANSIGEN_COST_SOURCE_ATTEMPT}")
    source_checkpoint = source_attempt / EXP055_TRANSIGEN_COST_SOURCE_CHECKPOINT
    source_close_path = source_attempt / "attempt_close_receipt.json"
    source_receipt_path = source_attempt / "last_finite_checkpoint_receipt.json"
    if not all(path.is_file() for path in (source_checkpoint, source_close_path, source_receipt_path)):
        raise FileNotFoundError("immutable v12 epoch1 checkpoint, receipt, and cost closure are required")
    source_close = json.loads(source_close_path.read_text(encoding="utf-8"))
    source_checkpoint_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    if (sha256_file(source_checkpoint) != EXP055_TRANSIGEN_COST_SOURCE_CHECKPOINT_SHA256
            or source_close.get("status") != "CLOSED_AFTER_TRAIN_ONLY_TRACE_COST_GATE"
            or source_close.get("epoch1_cost_gate_checkpoint_sha256") != EXP055_TRANSIGEN_COST_SOURCE_CHECKPOINT_SHA256
            or source_checkpoint_receipt.get("epoch") != 1
            or source_close.get("inner_dev_numeric_rows_loaded") is not False):
        raise RuntimeError("the v12 source checkpoint is not the verified finite epoch1 TRAIN-only artifact")
    device_snapshot = _device_snapshot()
    if (device_snapshot["selected_gpu"]["uuid"] != GPU_UUID
            or device_snapshot["selected_gpu"]["memory_used_mib"] > 1024
            or device_snapshot["selected_gpu"]["utilization_percent"] > 20):
        raise RuntimeError("the selected A5000 is not idle enough for the approved calibration preflight")
    runner_sha = sha256_file(Path(__file__).resolve())
    receipt = {
        "experiment_id": "EXP-055", "run_id": RUN_ID, "attempt_id": attempt_id,
        "status": "PASS_CPU_TRAIN_GATE_AND_V12_CHECKPOINT_BOUND_GPU_NOT_STARTED",
        "created_at_utc": utc_now(), "runner_sha256": runner_sha,
        "config_path": str(ROOT / "configs/experiments/EXP-055-representative-models.yaml"),
        "config_sha256": config_sha, "training_input_source_run_id": SOURCE_RUN_ID,
        "calibration_checkpoint_source_run_id": EXP055_TRANSIGEN_COST_SOURCE_RUN_ID,
        "calibration_checkpoint_source_attempt": EXP055_TRANSIGEN_COST_SOURCE_ATTEMPT,
        "calibration_checkpoint_path": str(source_checkpoint),
        "calibration_checkpoint_sha256": sha256_file(source_checkpoint),
        "source_close_receipt_sha256": sha256_file(source_close_path),
        "source_checkpoint_receipt_sha256": sha256_file(source_receipt_path),
        "reused_root_verified_train_input_gate": verified_gate,
        "prior_settled_recovery_gpu_hours": prior_charge,
        "prior_recovery_close_receipt_sha256": prior_close_hashes,
        "recovery_package_cap_gpu_hours": package_cap,
        "recovery_package_remaining_before_calibration_gpu_hours": remaining,
        "calibration_gpu_cap_hours": calibration_cap,
        "resource_accounting_before_attempt": accounting,
        "device_snapshot_before_launch_check": device_snapshot,
        "gpu_compute_pid_query_deferred_to_root_launch_gate": True,
        "training_contract": {"architecture": "upstream original TranSiGen",
            "training_conditions": 419, "training_objects": 110, "full_inchikey_count": 5,
            "genes": 978, "objective": "raw-sum/no-clip", "learning_rate": 1e-3,
            "weight_decay": 1e-5, "batch_size": 64, "shuffle": True,
            "epoch_updates": 7, "source_checkpoint_is_calibration_only": True,
            "formal_run_fresh_initialization": True, "full_schedule_epochs": 300},
        "monitoring_contract": {"per_batch_module_hooks_on_normal_path": False,
            "per_epoch_loss_gradient_parameter_optimizer_bn_train_metric_cost_summary": True,
            "failure_context_contains_first_batch_and_layer_rng_and_model_state": True},
        "data_roles": {"TRAIN_conditions": 419, "TRAIN_objects": 110,
            "inner_dev_identity_rows": 93, "inner_dev_objects": 24, "object_overlap": 0,
            "inner_dev_numeric_rows_loaded": False, "PDO_outer_locked_numeric_values_opened": False,
            "torch_imported": False, "cuda_used": False, "gpu_started": False},
        "attempt_directory_exists": False, "run_directory_empty_before_preflight_output": True,
        "closed_attempts_reopened": False}
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    atomic_json(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path)}


def _lean_monitor_finite_named_tensors(torch: Any, named_tensors: list[tuple[str, Any]]) -> tuple[bool, str | None]:
    if not named_tensors:
        return True, None
    flags = torch.stack([torch.isfinite(tensor.detach()).all() for _, tensor in named_tensors])
    if bool(flags.all().item()):
        return True, None
    for (name, _), flag in zip(named_tensors, flags):
        if not bool(flag.item()):
            return False, name
    return False, "unknown"


def _lean_monitor_batchnorm_named_tensors(model: Any) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for module_name, module in model.named_modules():
        if isinstance(module, __import__("torch").nn.modules.batchnorm._BatchNorm):
            for name in ("running_mean", "running_var"):
                value = getattr(module, name, None)
                if value is not None:
                    rows.append((f"{module_name or '<root>'}.{name}", value))
    return rows


def _lean_monitor_gradient_summary(torch: Any, model: Any) -> tuple[bool, float, str | None, bool]:
    named = [(name, parameter.grad.detach()) for name, parameter in model.named_parameters()
             if parameter.grad is not None]
    elements_finite, bad_name = _lean_monitor_finite_named_tensors(torch, named)
    if not named:
        return elements_finite, 0.0, bad_name, True
    if not elements_finite:
        # Keep the failure class explicit: a non-finite element is distinct from
        # an unrepresentable diagnostic norm over otherwise finite elements.
        return False, math.nan, bad_name, False
    norm = _float64_global_l2_norm(torch, [value for _, value in named])
    return True, norm, None, math.isfinite(norm)


def _lean_monitor_eval_failure_replay(torch: Any, model: Any, optimizer: Any, attempt: Path, *,
                                      train_keys: pd.DataFrame, X_ctl: np.ndarray, X: np.ndarray,
                                      features: np.ndarray, device: Any, seed: int,
                                      first_failure: dict[str, Any],
                                      evaluation_role: str = "TRAIN") -> dict[str, Any]:
    """Use full module hooks only after a nonfinite frozen-state evaluation is observed."""
    if evaluation_role not in {"TRAIN", "inner_dev"}:
        raise ValueError(f"unsupported lean monitor evaluation role: {evaluation_role}")
    rng_before = _lean_monitor_rng_state(torch)
    bn_before = _lean_monitor_bn_buffers(torch, model)
    state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    mode_before = bool(model.training)
    state_path = attempt / "lean_monitor_eval_failure_state.pt"
    _atomic_torch_checkpoint(torch, state_path, {
        "model_state_dict": {name: value.detach().cpu().clone() for name, value in state_before.items()},
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "rng_before_eval": rng_before,
        "first_nonfinite_batch": first_failure,
        "evaluation_role": evaluation_role,
        "checkpoint_role": f"frozen_state_at_first_nonfinite_{evaluation_role}_evaluation"})
    current_rng = _lean_monitor_rng_state(torch)
    current_bn = _lean_monitor_bn_buffers(torch, model)
    handles, layer_state = _trainpath_attach_eval_failure_hooks(torch, model)
    replay_failures: list[dict[str, Any]] = []
    try:
        _lean_monitor_restore_rng(torch, rng_before)
        model.eval()
        _transigen_predict_and_losses(torch, model, X_ctl, X, features, 64, device, seed,
            allow_nonfinite_auxiliary=True, allow_nonfinite_main=True,
            nonfinite_main_delta_callback=replay_failures.append)
    finally:
        for handle in handles:
            handle.remove()
        _lean_monitor_restore_bn(torch, model, current_bn)
        _lean_monitor_restore_rng(torch, current_rng)
        if any(not torch.equal(value, model.state_dict()[name])
               for name, value in state_before.items()):
            model.load_state_dict(state_before, strict=True)
        model.train(mode_before)
    failure_batch = (replay_failures[0] if replay_failures else first_failure)
    start = int(failure_batch.get("batch_start", 0))
    stop = int(failure_batch.get("batch_stop", min(start + 64, len(train_keys))))
    receipt = {"status": "FIRST_NONFINITE_TRAIN_EVALUATION_REPLAYED",
        "first_failure_batch": failure_batch,
        "first_nonfinite_layer": layer_state.get("first_nonfinite_layer"),
        "condition_keys": train_keys.iloc[start:stop].full_condition_key.astype(str).tolist(),
        "source_object_ids": train_keys.iloc[start:stop].source_object_id.astype(str).tolist(),
        "full_inchikeys": train_keys.iloc[start:stop].full_inchikey.astype(str).tolist(),
        "evaluation_role": evaluation_role,
        "model_state_checkpoint": state_path.name,
        "model_state_checkpoint_sha256": sha256_file(state_path),
        "rng_state_saved": True, "batchnorm_state_restored_after_replay": True,
        "optimizer_updates_during_replay": 0,
        "inner_dev_numeric_rows_loaded": evaluation_role == "inner_dev"}
    atomic_json(attempt / "lean_monitor_eval_first_nonfinite.json", receipt)
    return receipt


def run_exp055_transigen_lean_cost_calibration(
        cfg: dict[str, Any], config_sha: str, attempt_id: str,
        reviewed_preflight_sha256: str) -> dict[str, Any]:
    """Run one TRAIN-only TranSiGen cost epoch from the immutable v12 state, with lean monitoring."""
    if (EXPERIMENT_ID != "EXP-055" or RUN_ID != EXP055_TRANSIGEN_LEAN_COST_RUN_ID
            or attempt_id != "fit_attempt_001" or SOURCE_RUN_ID != DEFAULT_SOURCE_RUN_ID):
        raise ValueError("lean calibration is bound to its registered run/attempt and frozen TRAIN input source")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise RuntimeError("lean cost calibration requires only the approved A5000 UUID")
    preflight_path = RUN_DIR / "run_preflight_receipt.json"
    if not reviewed_preflight_sha256 or not preflight_path.is_file():
        raise RuntimeError("calibration requires this run's immutable CPU preflight SHA")
    preflight_sha = sha256_file(preflight_path)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    runner_sha = sha256_file(Path(__file__).resolve())
    if (preflight_sha != reviewed_preflight_sha256
            or preflight.get("status") != "PASS_CPU_TRAIN_GATE_AND_V12_CHECKPOINT_BOUND_GPU_NOT_STARTED"
            or preflight.get("run_id") != RUN_ID or preflight.get("attempt_id") != attempt_id
            or preflight.get("runner_sha256") != runner_sha
            or preflight.get("config_sha256") != config_sha
            or preflight.get("data_roles", {}).get("inner_dev_numeric_rows_loaded") is not False):
        raise RuntimeError("calibration preflight is stale, mismatched, or failed")
    prior_charge, prior_hashes = exp055_transigen_lean_monitor_settled_charge()
    calibration_cap = 0.05
    if (not math.isclose(prior_charge, 0.03086498953304087, rel_tol=0.0, abs_tol=1e-14)
            or 1.0 - prior_charge + 1e-12 < calibration_cap):
        raise RuntimeError("recovery balance differs from the root-verified v12-inclusive calibration envelope")
    limits = resource_limits(cfg)
    accounting = resource_accounting(EXPERIMENT_ROOT, RUN_ID, 1.0,
        limits["total_exp_cap_gpu_hours"],
        limits["exp_settled_before_artifact_ledgers_gpu_hours"], prior_charge)
    if (accounting["package_remaining_gpu_hours"] + 1e-12 < 1.0 - prior_charge
            or accounting["exp_remaining_gpu_hours"] + 1e-12 < calibration_cap):
        raise RuntimeError("resource ledger cannot reserve the one-time lean calibration")
    if (RUN_DIR.iterdir() and {path.name for path in RUN_DIR.iterdir()} != {"run_preflight_receipt.json"}):
        raise FileExistsError("calibration run contains artifacts beyond its fresh preflight")
    attempt = new_attempt_path(RUN_DIR, attempt_id)
    attempt.mkdir(parents=False, exist_ok=False)
    process_id = os.getpid()
    checkpoint_path = (ROOT / "artifacts/experiments/EXP-055" /
        f"{EXP055_TRANSIGEN_COST_SOURCE_RUN_ID}/{EXP055_TRANSIGEN_COST_SOURCE_ATTEMPT}/"
        f"{EXP055_TRANSIGEN_COST_SOURCE_CHECKPOINT}")
    if sha256_file(checkpoint_path) != EXP055_TRANSIGEN_COST_SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("v12 source checkpoint changed after CPU preflight")
    trace_owner = _open_trainpath_trace_owner(attempt, RUN_ID, attempt_id, process_id, runner_sha)
    manifest = {"experiment_id": "EXP-055", "run_id": RUN_ID, "fit_attempt": attempt_id,
        "phase": "TRANSIGEN_LEAN_MONITOR_COST_CALIBRATION_ONE_FULL_TRAIN_EPOCH",
        "status": "GPU_CALIBRATION_STARTED", "created_at_utc": utc_now(),
        "wsl_pid": process_id, "runner_sha256": runner_sha, "config_sha256": config_sha,
        "bound_preflight_receipt_sha256": preflight_sha,
        "calibration_only_source_checkpoint": str(checkpoint_path),
        "calibration_only_source_checkpoint_sha256": EXP055_TRANSIGEN_COST_SOURCE_CHECKPOINT_SHA256,
        "formal_initialization": "fresh upstream initialization; calibration checkpoint never loaded",
        "training_contract": {"architecture": "upstream original TranSiGen",
            "objective": "raw-sum/no-clip", "learning_rate": 1e-3, "weight_decay": 1e-5,
            "batch_size": 64, "shuffle": True, "train_conditions": 419,
            "train_objects": 110, "full_inchikey_count": 5, "genes": 978},
        "monitoring_contract": {"per_batch_full_module_hooks": False,
            "per_batch_loss_gradient_parameter_optimizer_finite_and_loss_trace": True,
            "per_epoch_train_metrics_bn_summary_cost_and_atomic_readback": True,
            "full_hooks_only_after_anomaly": True},
        "data_roles": {"inner_dev_numeric_rows_loaded": False,
            "pdo_outer_locked_numeric_values_opened": False}}
    atomic_json(attempt / "fit_smoke_manifest.json", manifest)
    atomic_json(attempt / "attempt_progress.json", {"phase": "GPU_READY_PRE_CALIBRATION_MODEL_LOAD",
        "updated_at_utc": utc_now(), "wsl_pid": process_id, "completed_calibration_epochs": 0,
        "inner_dev_numeric_rows_loaded": False})
    ledger = attempt / "resource_ledger.jsonl"
    append_jsonl(ledger, {"event": "GPU_ATTEMPT_STARTED", "at_utc": utc_now(),
        "wsl_pid": process_id, "device_uuid": GPU_UUID, "selected_models": ["transigen"],
        "reserved_gpu_hours": calibration_cap, "attempt_gpu_cap_hours": calibration_cap,
        "recovery_package_cap_gpu_hours": 1.0,
        "prior_recovery_attempts_settled_gpu_hours": prior_charge,
        "prior_recovery_close_receipt_sha256": prior_hashes,
        "runner_sha256": runner_sha, "config_sha256": config_sha,
        "preflight_receipt_sha256": preflight_sha})

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in WSL2 Conda drugscreening-gpu")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(SEED)
    np.random.seed(SEED % (2 ** 32))
    random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    gpu_start = time.perf_counter()
    cap_seconds = calibration_cap * 3600.0
    def budget_check() -> None:
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - gpu_start
        atomic_json(attempt / "attempt_progress.json", {"phase": "LEAN_COST_CALIBRATION_RUNNING",
            "updated_at_utc": utc_now(), "wsl_pid": process_id,
            "elapsed_gpu_hours_upper_bound": elapsed / 3600.0,
            "reserved_gpu_hours": calibration_cap, "completed_calibration_epochs": 0,
            "inner_dev_numeric_rows_loaded": False})
        if elapsed >= cap_seconds - 15.0:
            raise TimeoutError("lean TranSiGen calibration reached its 0.05 GPUh hard stop")

    try:
        verified_gate = _reuse_root_verified_exp055_train_gate()
        train_keys, values, feature_by_key, data_evidence = _load_exp055_trainpath_train_inputs_reusing_gate(
            verified_gate, SOURCE_RUN_DIR / "response_blind_conditions.tsv", SOURCE_RUN_DIR)
        features = np.stack([feature_by_key[key] for key in
            train_keys.full_inchikey.astype(str)]).astype(np.float32, copy=False)
        if (len(train_keys) != 419 or features.shape != (419, 2048)
                or values["delta"].shape != (419, 978)):
            raise ValueError("lean calibration inputs do not match 419 TRAIN conditions and 978 genes")
        model_start = time.perf_counter()
        model, model_module = _load_transigen(torch, device, SEED, features_dim=2048)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        source_payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if (source_payload.get("epoch") != 1 or source_payload.get("objective") != "raw_sum"
                or source_payload.get("gradient_clipping") is not False
                or source_payload.get("fit_scope") != "TRAIN_ONLY_NO_INNER_DEV_VALUES"
                or not isinstance(source_payload.get("state_dict"), dict)
                or not isinstance(source_payload.get("optimizer_state_dict"), dict)):
            raise RuntimeError("v12 epoch1 source checkpoint metadata does not match calibration-only role")
        source_state = source_payload["state_dict"]
        if any(isinstance(value, torch.Tensor) and value.is_floating_point()
               and not bool(torch.isfinite(value).all().item()) for value in source_state.values()):
            raise FloatingPointError("v12 source checkpoint contains a nonfinite model tensor")
        model.load_state_dict(source_state, strict=True)
        optimizer.load_state_dict(source_payload["optimizer_state_dict"])
        if (not math.isclose(float(optimizer.param_groups[0]["lr"]), 1e-3,
                             rel_tol=0.0, abs_tol=1e-12)
                or not math.isclose(float(optimizer.param_groups[0]["weight_decay"]), 1e-5,
                                    rel_tol=0.0, abs_tol=1e-12)):
            raise RuntimeError("v12 source optimizer state differs from the fixed TranSiGen LR/weight decay")
        source_loaded_seconds = time.perf_counter() - model_start
        for name, value in model.state_dict().items():
            if not torch.equal(value.detach().cpu(), source_state[name].detach().cpu()):
                raise RuntimeError(f"v12 source checkpoint load readback differs at {name}")

        dataset = TensorDataset(torch.as_tensor(values["X_ctl"], dtype=torch.float32),
            torch.as_tensor(values["X"], dtype=torch.float32),
            torch.as_tensor(features, dtype=torch.float32), torch.arange(419, dtype=torch.int64))
        loader_generator = torch.Generator().manual_seed(SEED)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False,
            num_workers=0, generator=loader_generator)
        batch_sizes = [int(batch[0].shape[0]) for batch in loader]
        loader_generator.manual_seed(SEED)
        if batch_sizes != [64, 64, 64, 64, 64, 64, 35]:
            raise ValueError(f"lean calibration TRAIN loader changed: {batch_sizes}")
        curve_path = attempt / "lean_monitor_batch_trace.jsonl"
        curve_path.touch(exist_ok=False)
        loss_names = ("raw_total_loss", "mse_x1", "mse_x2", "mse_pert",
                      "kld_x1", "kld_x2", "kld_pert")
        loss_sums = np.zeros(7, dtype=np.float64)
        grad_norms: list[float] = []
        loss_rows: list[dict[str, float]] = []
        seen = 0
        bn_before, bn_prior = _trainpath_bn_snapshot(torch, model, "calibration_source_epoch1_checkpoint")
        train_start = time.perf_counter()
        model.train()
        for batch_index, (x1_cpu, x2_cpu, feat_cpu, row_cpu) in enumerate(loader, start=1):
            budget_check()
            row_positions = row_cpu.numpy().astype(np.int64, copy=False)
            rng_before = _lean_monitor_rng_state(torch)
            bn_buffers_before = _lean_monitor_bn_buffers(torch, model)
            x1, x2, feat = x1_cpu.to(device), x2_cpu.to(device), feat_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, losses = _trainpath_forward_losses(torch, model, x1, x2, feat)
            loss_values = [float(value.detach().cpu().item()) for value in losses]
            if len(losses) != 7 or not all(math.isfinite(value) for value in loss_values):
                detail = {"loss_names": list(loss_names), "loss_values": loss_values}
                failure = _lean_monitor_failure_context(torch, model, optimizer, attempt,
                    epoch=2, batch_index=batch_index, component="training_loss",
                    train_keys=train_keys, row_positions=row_positions, x1=x1, x2=x2, features=feat,
                    rng_before_batch=rng_before, bn_before_batch=bn_buffers_before,
                    last_finite_checkpoint=checkpoint_path, component_detail=detail)
                raise FloatingPointError(f"nonfinite raw-sum loss at calibration batch {batch_index}: {failure}")
            main_delta = prediction - x1
            if (prediction.shape != x1.shape or not bool(torch.isfinite(prediction).all().item())
                    or not bool(torch.isfinite(main_delta).all().item())):
                detail = {"prediction_shape": list(prediction.shape), "expected_shape": list(x1.shape),
                    "prediction_nonfinite_n": int((~torch.isfinite(prediction)).sum().item()),
                    "main_delta_nonfinite_n": int((~torch.isfinite(main_delta)).sum().item())}
                failure = _lean_monitor_failure_context(torch, model, optimizer, attempt,
                    epoch=2, batch_index=batch_index, component="training_main_delta",
                    train_keys=train_keys, row_positions=row_positions, x1=x1, x2=x2, features=feat,
                    rng_before_batch=rng_before, bn_before_batch=bn_buffers_before,
                    last_finite_checkpoint=checkpoint_path, component_detail=detail)
                raise FloatingPointError(f"nonfinite or malformed TRAIN main delta at calibration batch {batch_index}: {failure}")
            losses[0].backward()
            grad_finite, grad_norm, bad_gradient, grad_norm_finite = _lean_monitor_gradient_summary(torch, model)
            if not grad_finite:
                failure = _lean_monitor_failure_context(torch, model, optimizer, attempt,
                    epoch=2, batch_index=batch_index, component="gradient",
                    train_keys=train_keys, row_positions=row_positions, x1=x1, x2=x2, features=feat,
                    rng_before_batch=rng_before, bn_before_batch=bn_buffers_before,
                    last_finite_checkpoint=checkpoint_path,
                    component_detail={"gradient_elements_finite": grad_finite,
                        "gradient_norm_finite": grad_norm_finite,
                        "first_nonfinite_parameter_gradient": bad_gradient,
                        "gradient_norm": safe_number(grad_norm)})
                raise FloatingPointError(f"nonfinite TRAIN gradient at calibration batch {batch_index}: {failure}")
            optimizer.step()
            optimizer_bad = _first_nonfinite_optimizer_state(torch, optimizer)
            params_finite, bad_parameter = _lean_monitor_finite_named_tensors(
                torch, [(name, parameter) for name, parameter in model.named_parameters()])
            bn_finite, bad_bn = _lean_monitor_finite_named_tensors(
                torch, _lean_monitor_batchnorm_named_tensors(model))
            if optimizer_bad is not None or not params_finite or not bn_finite:
                component = ("optimizer_state" if optimizer_bad is not None else
                             "parameter" if not params_finite else "batchnorm_state")
                detail = {"optimizer_state": optimizer_bad,
                    "first_nonfinite_parameter": bad_parameter,
                    "first_nonfinite_batchnorm_buffer": bad_bn}
                failure = _lean_monitor_failure_context(torch, model, optimizer, attempt,
                    epoch=2, batch_index=batch_index, component=component,
                    train_keys=train_keys, row_positions=row_positions, x1=x1, x2=x2, features=feat,
                    rng_before_batch=rng_before, bn_before_batch=bn_buffers_before,
                    last_finite_checkpoint=checkpoint_path, replay_forward=False,
                    component_detail=detail)
                raise FloatingPointError(f"nonfinite model or optimizer state at calibration batch {batch_index}: {failure}")
            loss_sums += np.asarray(loss_values, dtype=np.float64)
            loss_rows.append(dict(zip(loss_names, loss_values)))
            grad_norms.append(grad_norm if grad_norm_finite else math.nan)
            seen += int(len(x1))
            append_jsonl(curve_path, {"epoch": 2, "batch_index": batch_index,
                "conditions": int(len(x1)), "condition_keys": train_keys.iloc[row_positions].full_condition_key.astype(str).tolist(),
                "losses": dict(zip(loss_names, loss_values)), "loss_finite": True,
                "gradient_elements_finite": grad_finite,
                "gradient_norm_finite": grad_norm_finite,
                "gradient_norm": safe_number(grad_norm),
                "gradient_finite": grad_finite,
                "parameters_finite": params_finite, "optimizer_finite": optimizer_bad is None,
                "batchnorm_finite": bn_finite, "batch_seconds": None,
                "inner_dev_numeric_rows_loaded": False})
            del x1, x2, feat, prediction, main_delta, losses
        if seen != 419 or len(grad_norms) != 7:
            raise ValueError(f"lean calibration must update all 419 TRAIN conditions in seven batches; got {seen}")
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - train_start
        bn_after, _ = _trainpath_bn_snapshot(torch, model, "calibration_epoch2_complete", bn_prior)
        params_finite, bad_parameter = _lean_monitor_finite_named_tensors(
            torch, [(name, parameter) for name, parameter in model.named_parameters()])
        optimizer_finite = _first_nonfinite_optimizer_state(torch, optimizer) is None
        if not params_finite or not optimizer_finite:
            raise FloatingPointError(f"nonfinite final calibration state: parameter={bad_parameter}")

        train_eval_start = time.perf_counter()
        eval_rng = _lean_monitor_rng_state(torch)
        eval_failures: list[dict[str, Any]] = []
        train_prediction, auxiliary_losses = _transigen_predict_and_losses(
            torch, model, values["X_ctl"], values["X"], features, 64, device, SEED + 2002,
            allow_nonfinite_auxiliary=True, allow_nonfinite_main=True,
            nonfinite_main_delta_callback=eval_failures.append)
        _lean_monitor_restore_rng(torch, eval_rng)
        eval_layer = None
        finite_eval = bool(np.isfinite(train_prediction).all())
        if not finite_eval:
            replay = _lean_monitor_eval_failure_replay(torch, model, optimizer, attempt,
                train_keys=train_keys, X_ctl=values["X_ctl"], X=values["X"], features=features,
                device=device, seed=SEED + 2002, first_failure=eval_failures[0])
            eval_layer = replay["first_nonfinite_layer"]
        eval_seconds = time.perf_counter() - train_eval_start
        eval_summary = _monitored_main_prediction_summary(train_prediction,
            expected_shape=(419, 978), first_layer=eval_layer, batch_failures=eval_failures)
        train_keys_ordered = train_keys.reset_index(drop=True)
        train_y = values["delta"]
        gene_variance = np.var(train_y.astype(np.float64), axis=0, ddof=0)
        baseline_group = train_keys_ordered.assign(_row=np.arange(419)).groupby(
            "full_inchikey", sort=True)["_row"].apply(list)
        per_drug_prediction = np.empty_like(train_y)
        for positions in baseline_group.tolist():
            pos = np.asarray(positions, dtype=np.int64)
            per_drug_prediction[pos] = train_y[pos].mean(axis=0, dtype=np.float64).astype(np.float32)
        baselines = {"zero_change": keyed_metrics(train_keys_ordered, train_y,
                       np.zeros_like(train_y), gene_variance),
                     "per_drug_mean_delta": keyed_metrics(train_keys_ordered, train_y,
                       per_drug_prediction, gene_variance)}
        if finite_eval:
            metrics = keyed_metrics(train_keys_ordered, train_y, train_prediction, gene_variance)
            metrics.update({key: eval_summary[key] for key in (
                "status", "main_delta_nonfinite_n", "main_delta_denominator", "first_nonfinite_layer")})
            fit_gate = _fit_gate(metrics, _wrap_trainpath_recovery_baselines_for_fit_gate(baselines))
        else:
            metrics = eval_summary
            fit_gate = {"status": "UNDEFINED_NONFINITE_MAIN_DELTA",
                "main_delta_nonfinite_n": eval_summary["main_delta_nonfinite_n"],
                "main_delta_denominator": eval_summary["main_delta_denominator"],
                "first_nonfinite_layer": eval_summary["first_nonfinite_layer"]}

        prediction_path = attempt / "TranSiGen_lean_cost_calibration_train_predictions.npz"
        prediction_io_start = time.perf_counter()
        atomic_npz(prediction_path,
            train_condition_key=train_keys_ordered.full_condition_key.astype(str).to_numpy(dtype=np.str_),
            train_source_object_id=train_keys_ordered.source_object_id.astype(str).to_numpy(dtype=np.str_),
            train_full_inchikey=train_keys_ordered.full_inchikey.astype(str).to_numpy(dtype=np.str_),
            genes=values["genes"].astype(str), train_target_delta=train_y,
            train_delta=train_prediction)
        with np.load(prediction_path, allow_pickle=False) as saved:
            prediction_readback_pass = (
                np.array_equal(saved["train_condition_key"], train_keys_ordered.full_condition_key.astype(str).to_numpy(dtype=np.str_))
                and np.array_equal(saved["train_source_object_id"], train_keys_ordered.source_object_id.astype(str).to_numpy(dtype=np.str_))
                and np.array_equal(saved["train_full_inchikey"], train_keys_ordered.full_inchikey.astype(str).to_numpy(dtype=np.str_))
                and np.array_equal(saved["genes"], values["genes"].astype(str))
                and _np_arrays_equal_with_nan(saved["train_target_delta"], train_y)
                and _np_arrays_equal_with_nan(saved["train_delta"], train_prediction))
        if not prediction_readback_pass:
            raise RuntimeError("lean calibration TRAIN prediction key/axis/value readback failed")
        prediction_io_seconds = time.perf_counter() - prediction_io_start

        checkpoint_payload = {"state_dict": {name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()},
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "model": "upstream TranSiGen", "epoch": 2, "seed": SEED,
            "objective": "raw_sum", "gradient_clipping": False,
            "learning_rate": 1e-3, "weight_decay": 1e-5,
            "fit_scope": "TRAIN_ONLY_NO_INNER_DEV_VALUES",
            "checkpoint_role": "cost_calibration_only_never_formal_initialization",
            "source_checkpoint_sha256": EXP055_TRANSIGEN_COST_SOURCE_CHECKPOINT_SHA256}
        checkpoint_out = attempt / "TranSiGen_lean_cost_calibration_epoch.pt"
        checkpoint_start = time.perf_counter()
        _atomic_torch_checkpoint(torch, checkpoint_out, checkpoint_payload)
        saved_checkpoint = torch.load(checkpoint_out, map_location=device, weights_only=False)
        checkpoint_readback_pass = (_torch_tree_exact(torch, checkpoint_payload["state_dict"],
                                                       saved_checkpoint["state_dict"])
            and _torch_tree_exact(torch, checkpoint_payload["optimizer_state_dict"],
                                   saved_checkpoint["optimizer_state_dict"]))
        if not checkpoint_readback_pass:
            raise RuntimeError("lean calibration checkpoint/model/optimizer readback failed")
        checkpoint_io_seconds = time.perf_counter() - checkpoint_start
        torch.cuda.synchronize()
        total_calibration_seconds = time.perf_counter() - gpu_start
        actual_gpu_hours = total_calibration_seconds / 3600.0
        projected_fresh_init_seconds = source_loaded_seconds
        projected_final_train_eval_seconds = eval_seconds
        projected_final_inner_dev_seconds = eval_seconds * (93.0 / 419.0)
        projected_endpoint_and_final_prediction_io = prediction_io_seconds * 3.0
        projected_endpoint_checkpoint_io = checkpoint_io_seconds * 3.0
        projected_json_close_seconds = 5.0
        projected_fixed300_without_margin = (projected_fresh_init_seconds
            + 300.0 * (train_seconds + eval_seconds + checkpoint_io_seconds)
            + projected_endpoint_checkpoint_io + projected_endpoint_and_final_prediction_io
            + projected_final_train_eval_seconds + projected_final_inner_dev_seconds
            + projected_json_close_seconds)
        projected_fixed300_with_margin = projected_fixed300_without_margin * 1.20 / 3600.0
        formal_remaining_after_calibration = 1.0 - prior_charge - actual_gpu_hours
        full_package_projection = prior_charge + actual_gpu_hours + projected_fixed300_with_margin
        schedule_finite = finite_eval and params_finite and optimizer_finite
        gate_status = ("PASS_FIXED300_WITHIN_RECOVERY_CAP"
            if schedule_finite and actual_gpu_hours <= calibration_cap
                and projected_fixed300_with_margin <= formal_remaining_after_calibration
            else "STOP_AFTER_CALIBRATION_FIXED300_EXCEEDS_RECOVERY_CAP"
                if projected_fixed300_with_margin > formal_remaining_after_calibration
                else "STOP_AFTER_CALIBRATION_NONFINITE_TRAIN_OR_EVALUATION")
        cost_gate = {"status": gate_status,
            "calibration_actual_gpu_hours": actual_gpu_hours,
            "calibration_gpu_cap_hours": calibration_cap,
            "prior_recovery_settled_gpu_hours": prior_charge,
            "formal_remaining_after_calibration_gpu_hours": formal_remaining_after_calibration,
            "projected_fresh_initialization_seconds": projected_fresh_init_seconds,
            "observed_calibration_model_load_seconds": source_loaded_seconds,
            "observed_full_train_epoch_seconds": train_seconds,
            "observed_full_train_evaluation_seconds": eval_seconds,
            "observed_keyed_prediction_save_readback_seconds": prediction_io_seconds,
            "observed_atomic_checkpoint_save_readback_seconds": checkpoint_io_seconds,
            "projected_fixed100_and_300_endpoint_checkpoint_io_seconds": projected_endpoint_checkpoint_io,
            "projected_fixed100_300_and_final_prediction_io_seconds": projected_endpoint_and_final_prediction_io,
            "projected_final_train_eval_seconds": projected_final_train_eval_seconds,
            "projected_final_93_inner_dev_eval_seconds": projected_final_inner_dev_seconds,
            "projected_close_seconds": projected_json_close_seconds,
            "projected_fixed300_without_margin_seconds": projected_fixed300_without_margin,
            "projected_fixed300_including_20pct_margin_gpu_hours": projected_fixed300_with_margin,
            "projected_prior_plus_calibration_plus_fixed300_gpu_hours": full_package_projection,
            "recovery_package_cap_gpu_hours": 1.0,
            "fixed_schedule_epochs": 300, "fixed_trajectory_checkpoint_epochs": [100, 300],
            "per_epoch_train_evaluation_included": True,
            "source_checkpoint_used_for_formal_initialization": False,
            "inner_dev_numeric_rows_loaded": False,
            "train_evaluation_main_delta_nonfinite_n": eval_summary["main_delta_nonfinite_n"],
            "train_evaluation_main_delta_denominator": eval_summary["main_delta_denominator"],
            "first_nonfinite_layer": eval_summary["first_nonfinite_layer"],
            "prediction_readback_pass": prediction_readback_pass,
            "checkpoint_readback_pass": checkpoint_readback_pass}
        atomic_json(attempt / "lean_monitor_cost_gate.json", cost_gate)
        epoch_result = {"status": "PASS_LEAN_MONITOR_COMPLETE_TRAIN_COST_EPOCH" if schedule_finite
                else "FAILED_LEAN_MONITOR_NONFINITE_TRAIN_EVALUATION",
            "run_id": RUN_ID, "attempt_id": attempt_id, "source_checkpoint_sha256":
                EXP055_TRANSIGEN_COST_SOURCE_CHECKPOINT_SHA256,
            "source_checkpoint_unchanged": sha256_file(checkpoint_path) == EXP055_TRANSIGEN_COST_SOURCE_CHECKPOINT_SHA256,
            "source_checkpoint_used_for_training_only": True,
            "formal_model_initialization_is_fresh": True,
            "train_conditions": seen, "train_batch_sizes": batch_sizes,
            "loss_names": list(loss_names),
            "loss_mean_per_condition": {name: safe_number(float(value / seen))
                for name, value in zip(loss_names, loss_sums)},
            "gradient_norm_mean": safe_number(float(np.mean(grad_norms))),
            "gradient_norm_max": safe_number(float(np.max(grad_norms))),
            "gradient_norm_finite_n_updates": sum(math.isfinite(value) for value in grad_norms),
            "parameters_finite": params_finite, "optimizer_finite": optimizer_finite,
            "batchnorm_before": bn_before, "batchnorm_after": bn_after,
            "train_metrics": metrics, "train_fit_gate": fit_gate,
            "train_evaluation_auxiliary_losses": auxiliary_losses,
            "train_evaluation": eval_summary,
            "train_prediction_path": prediction_path.name,
            "train_prediction_sha256": sha256_file(prediction_path),
            "prediction_readback_pass": prediction_readback_pass,
            "checkpoint_path": checkpoint_out.name,
            "checkpoint_sha256": sha256_file(checkpoint_out),
            "checkpoint_readback_pass": checkpoint_readback_pass,
            "model_load_seconds": source_loaded_seconds,
            "train_epoch_seconds": train_seconds, "train_eval_seconds": eval_seconds,
            "prediction_save_readback_seconds": prediction_io_seconds,
            "checkpoint_save_readback_seconds": checkpoint_io_seconds,
            "actual_gpu_hours": actual_gpu_hours,
            "data_identity_evidence": data_evidence,
            "inner_dev_numeric_rows_loaded": False,
            "pdo_outer_locked_numeric_values_opened": False}
        atomic_json(attempt / "lean_monitor_cost_calibration_result.json", epoch_result)
        trace_owner = _close_trainpath_trace_owner(attempt, RUN_ID, attempt_id,
            "CLOSED_AFTER_COST_CALIBRATION" if gate_status.startswith("PASS")
            else "CLOSED_AFTER_COST_CALIBRATION_STOP")
        close = {"status": "CLOSED_AFTER_COST_CALIBRATION", "run_id": RUN_ID,
            "attempt_id": attempt_id, "wsl_pid": process_id, "device_uuid": GPU_UUID,
            "runner_sha256": runner_sha, "config_sha256": config_sha,
            "preflight_receipt_sha256": preflight_sha,
            "actual_gpu_hours": actual_gpu_hours, "reserved_gpu_hours": calibration_cap,
            "released_unused_reservation_gpu_hours": calibration_cap - actual_gpu_hours,
            "prior_recovery_settled_gpu_hours": prior_charge,
            "cumulative_recovery_spend_gpu_hours": prior_charge + actual_gpu_hours,
            "cost_gate_status": gate_status,
            "cost_gate_sha256": sha256_file(attempt / "lean_monitor_cost_gate.json"),
            "result_sha256": sha256_file(attempt / "lean_monitor_cost_calibration_result.json"),
            "checkpoint_sha256": sha256_file(checkpoint_out),
            "prediction_sha256": sha256_file(prediction_path),
            "trace_owner_sha256": sha256_file(_trainpath_trace_owner_path(attempt)),
            "inner_dev_numeric_rows_loaded": False,
            "pdo_outer_locked_numeric_values_opened": False,
            "closed_at_utc": utc_now()}
        atomic_json(attempt / "attempt_close_receipt.json", close)
        append_jsonl(ledger, {"event": "GPU_ATTEMPT_CLOSED_AFTER_COST_CALIBRATION",
            "at_utc": utc_now(), "wsl_pid": process_id, "device_uuid": GPU_UUID,
            "selected_models": ["transigen"], "actual_gpu_hours": actual_gpu_hours,
            "reserved_gpu_hours": calibration_cap,
            "unused_reservation_released_gpu_hours": calibration_cap - actual_gpu_hours,
            "prior_recovery_attempts_settled_gpu_hours": prior_charge,
            "full_package_projection_gpu_hours": full_package_projection,
            "close_receipt": "attempt_close_receipt.json"})
        atomic_json(attempt / "attempt_progress.json", {"phase": close["status"],
            "updated_at_utc": utc_now(), "wsl_pid": process_id,
            "completed_calibration_epochs": 1, "actual_gpu_hours": actual_gpu_hours,
            "cost_gate_status": gate_status, "inner_dev_numeric_rows_loaded": False})
        return {"result": epoch_result, "cost_gate": cost_gate, "close_receipt": close}
    except BaseException:
        owner_path = _trainpath_trace_owner_path(attempt)
        if owner_path.is_file():
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            if owner.get("status") == "OPEN":
                _close_trainpath_trace_owner(attempt, RUN_ID, attempt_id,
                    "FAILED_CLOSED_CALIBRATION_EXCEPTION")
        raise


def _trainpath_tensor_leaves(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten tensor-valued module inputs/outputs without retaining activations."""
    if hasattr(value, "detach") and hasattr(value, "numel") and hasattr(value, "shape"):
        return [(prefix or "tensor", value)]
    leaves: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            leaves.extend(_trainpath_tensor_leaves(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            leaves.extend(_trainpath_tensor_leaves(item, f"{prefix}.{index}" if prefix else str(index)))
    return leaves


def _trainpath_record_tensor(torch: Any, attempt: Path, events: list[dict[str, Any]],
                             *, scope: str, batch_index: int | None,
                             module_name: str, io_role: str, tensor_name: str,
                             tensor: Any) -> None:
    detached = tensor.detach()
    numel = int(detached.numel())
    finite_n = int(torch.isfinite(detached).sum().item()) if numel else 0
    finite = finite_n == numel
    max_abs_tensor = (detached.abs().max() if numel else
                      torch.zeros((), device=detached.device, dtype=torch.float32))
    if not finite:
        finite_values = detached[torch.isfinite(detached)]
        finite_max = (float(finite_values.abs().max().item())
                      if finite_values.numel() else None)
        first_bad = {"status": "FIRST_NONFINITE_TRAINPATH_TENSOR",
            "scope": scope, "batch_index": batch_index, "module": module_name,
            "io_role": io_role, "tensor_name": tensor_name,
            "shape": list(detached.shape), "numel": numel, "finite_n": finite_n,
            "nonfinite_n": numel - finite_n, "finite_max_abs": finite_max,
            "timestamp_utc": utc_now(), "train_only": True,
            "inner_dev_numeric_rows_loaded": False,
            "pdo_outer_locked_numeric_values_opened": False}
        atomic_json(attempt / "trainpath_first_nonfinite.json", first_bad)
        raise FloatingPointError(
            f"first nonfinite TRAIN trace tensor at {scope}/{batch_index}/{module_name}/{io_role}/{tensor_name}")
    events.append({"scope": scope, "batch_index": batch_index, "module": module_name,
                   "io_role": io_role, "tensor_name": tensor_name, "shape": list(detached.shape),
                   "numel": numel, "finite_n": finite_n, "_max_abs_tensor": max_abs_tensor})


def _trainpath_trace_owner_path(attempt: Path) -> Path:
    return attempt / "TranSiGen_trainpath_layer_trace_manifest.json"


def _open_trainpath_trace_owner(attempt: Path, run_id: str, attempt_id: str,
                                process_id: int, runner_sha: str) -> dict[str, Any]:
    """Create the immutable run/attempt binding before this attempt emits any layer trace."""
    if (run_id != RUN_ID or attempt_id != attempt.name
            or attempt.resolve() != (RUN_DIR / attempt_id).resolve()):
        raise ValueError("layer-trace owner must bind to the current run and its exact attempt directory")
    if any(attempt.iterdir()):
        raise FileExistsError("new trainpath attempt directory must be empty before ownership is registered")
    owner_path = _trainpath_trace_owner_path(attempt)
    trace_path = attempt / "TranSiGen_trainpath_layer_trace.jsonl"
    if owner_path.exists() or trace_path.exists():
        raise FileExistsError("refusing to attach this attempt to existing trainpath trace artifacts")
    owner = {"status": "OPEN", "run_id": run_id, "attempt_id": attempt_id,
             "attempt_directory": str(attempt.resolve()), "wsl_pid": int(process_id),
             "runner_sha256": runner_sha, "trace_path": trace_path.name,
             "created_at_utc": utc_now(), "append_policy": "append-only under this run/attempt owner"}
    atomic_json(owner_path, owner)
    return owner


def _require_trainpath_trace_owner(attempt: Path, run_id: str, attempt_id: str) -> dict[str, Any]:
    owner_path = _trainpath_trace_owner_path(attempt)
    trace_path = attempt / "TranSiGen_trainpath_layer_trace.jsonl"
    if not owner_path.is_file():
        if trace_path.exists():
            raise FileExistsError("trainpath trace exists without a run/attempt owner; refusing stale-file append")
        raise FileNotFoundError("trainpath trace owner manifest is missing")
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    if (owner.get("run_id") != run_id or owner.get("attempt_id") != attempt_id
            or owner.get("attempt_directory") != str(attempt.resolve())
            or owner.get("trace_path") != trace_path.name):
        raise RuntimeError("trainpath trace owner does not match the current run/attempt directory")
    if owner.get("status") != "OPEN":
        raise RuntimeError("trainpath trace attempt is closed; refusing any further append")
    return owner


def _append_trainpath_trace_row(attempt: Path, run_id: str, attempt_id: str,
                                row: dict[str, Any]) -> None:
    _require_trainpath_trace_owner(attempt, run_id, attempt_id)
    append_jsonl(attempt / "TranSiGen_trainpath_layer_trace.jsonl",
                 {**row, "run_id": run_id, "attempt_id": attempt_id})


def _close_trainpath_trace_owner(attempt: Path, run_id: str, attempt_id: str,
                                 status: str) -> dict[str, Any]:
    owner = _require_trainpath_trace_owner(attempt, run_id, attempt_id)
    owner.update({"status": "CLOSED", "close_status": status, "closed_at_utc": utc_now(),
                  "trace_sha256": sha256_file(attempt / "TranSiGen_trainpath_layer_trace.jsonl")
                  if (attempt / "TranSiGen_trainpath_layer_trace.jsonl").is_file() else None})
    atomic_json(_trainpath_trace_owner_path(attempt), owner)
    return owner


def _trainpath_flush_layer_events(torch: Any, attempt: Path,
                                  events: list[dict[str, Any]]) -> None:
    if not events:
        return
    values = torch.stack([item["_max_abs_tensor"].reshape(()).to(dtype=torch.float64)
                          for item in events]).detach().cpu().tolist()
    target = attempt / "TranSiGen_trainpath_layer_trace.jsonl"
    for event, max_abs in zip(events, values):
        row = {key: value for key, value in event.items() if key != "_max_abs_tensor"}
        row["finite"] = row["finite_n"] == row["numel"]
        row["nonfinite_n"] = row["numel"] - row["finite_n"]
        row["max_abs"] = float(max_abs)
        _append_trainpath_trace_row(attempt, RUN_ID, attempt.name, row)
    events.clear()


def _trainpath_attach_layer_hooks(torch: Any, model: Any, attempt: Path,
                                  event_state: dict[str, Any]) -> list[Any]:
    handles: list[Any] = []
    for name, module in model.named_modules():
        module_name = name or "<root>"
        def pre_hook(_module: Any, inputs: Any, *, module_name: str = module_name) -> None:
            for tensor_name, tensor in _trainpath_tensor_leaves(inputs):
                _trainpath_record_tensor(torch, attempt, event_state["events"],
                    scope=event_state["scope"], batch_index=event_state["batch_index"],
                    module_name=module_name, io_role="input", tensor_name=tensor_name, tensor=tensor)
        def post_hook(_module: Any, _inputs: Any, output: Any,
                      *, module_name: str = module_name) -> None:
            for tensor_name, tensor in _trainpath_tensor_leaves(output):
                _trainpath_record_tensor(torch, attempt, event_state["events"],
                    scope=event_state["scope"], batch_index=event_state["batch_index"],
                    module_name=module_name, io_role="output", tensor_name=tensor_name, tensor=tensor)
        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))
    return handles


def _trainpath_attach_eval_failure_hooks(torch: Any, model: Any) -> tuple[list[Any], dict[str, Any]]:
    """Observe the first non-finite module boundary during eval without stopping or changing the model."""
    state: dict[str, Any] = {"first_nonfinite_layer": None}

    def inspect(module_name: str, io_role: str, tree: Any) -> None:
        if state["first_nonfinite_layer"] is not None:
            return
        for tensor_name, tensor in _trainpath_tensor_leaves(tree):
            if not bool(torch.isfinite(tensor).all().item()):
                finite = torch.isfinite(tensor)
                bad_positions = torch.nonzero(~finite, as_tuple=False)[:16].detach().cpu().tolist()
                state["first_nonfinite_layer"] = {
                    "module": module_name, "io_role": io_role, "tensor_name": tensor_name,
                    "shape": list(tensor.shape), "numel": int(tensor.numel()),
                    "finite_n": int(finite.sum().item()),
                    "nonfinite_n": int((~finite).sum().item()),
                    "first_nonfinite_positions": bad_positions,
                    "timestamp_utc": utc_now(), "evaluation_split": "TRAIN_ONLY"}
                return

    handles = []
    for name, module in model.named_modules():
        module_name = name or "<root>"
        handles.append(module.register_forward_pre_hook(
            lambda _module, inputs, n=module_name: inspect(n, "input", inputs)))
        handles.append(module.register_forward_hook(
            lambda _module, _inputs, output, n=module_name: inspect(n, "output", output)))
    return handles, state


def _lean_monitor_rng_state(torch: Any) -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _lean_monitor_restore_rng(torch: Any, state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _lean_monitor_bn_buffers(torch: Any, model: Any) -> dict[str, dict[str, Any]]:
    return {name: {buffer_name: buffer.detach().clone()
                   for buffer_name, buffer in module.named_buffers(recurse=False)}
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}


def _lean_monitor_restore_bn(torch: Any, model: Any,
                             saved: dict[str, dict[str, Any]]) -> None:
    with torch.no_grad():
        for name, module in model.named_modules():
            if name in saved:
                current = dict(module.named_buffers(recurse=False))
                for buffer_name, value in saved[name].items():
                    if buffer_name in current and value is not None:
                        current[buffer_name].copy_(value)


def _lean_monitor_failure_context(torch: Any, model: Any, optimizer: Any, attempt: Path, *,
                                  epoch: int, batch_index: int, component: str,
                                  train_keys: pd.DataFrame, row_positions: np.ndarray,
                                  x1: Any, x2: Any, features: Any,
                                  rng_before_batch: dict[str, Any],
                                  bn_before_batch: dict[str, dict[str, Any]],
                                  last_finite_checkpoint: Path | None,
                                  replay_forward: bool = True,
                                  component_detail: dict[str, Any] | None = None) -> dict[str, Any]:
    """Persist exact failure state; only abnormal batches pay for full module hooks."""
    failure_state_path = attempt / "lean_monitor_failure_state.pt"
    payload = {"model_state_dict": {name: value.detach().cpu().clone()
                                    for name, value in model.state_dict().items()},
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "rng_before_batch": rng_before_batch,
        "batch_inputs": {"x1": x1.detach().cpu(), "x2": x2.detach().cpu(),
                         "features": features.detach().cpu()},
        "epoch": epoch, "batch_index": batch_index,
        "component": component, "component_detail": component_detail,
        "checkpoint_role": "state_at_first_nonfinite_detection_not_a_selected_checkpoint"}
    _atomic_torch_checkpoint(torch, failure_state_path, payload)
    layer_state: dict[str, Any] = {"first_nonfinite_layer": None}
    trace_events: list[dict[str, Any]] = []
    if replay_forward:
        current_rng = _lean_monitor_rng_state(torch)
        current_bn = _lean_monitor_bn_buffers(torch, model)
        was_training = bool(model.training)
        event_state = {"events": trace_events, "scope": f"failure_replay_epoch{epoch}",
                       "batch_index": batch_index}
        handles = _trainpath_attach_layer_hooks(torch, model, attempt, event_state)
        replay_error: BaseException | None = None
        try:
            _lean_monitor_restore_bn(torch, model, bn_before_batch)
            _lean_monitor_restore_rng(torch, rng_before_batch)
            model.train()
            with torch.no_grad():
                _trainpath_forward_losses(torch, model, x1, x2, features)
        except FloatingPointError as error:
            replay_error = error
        finally:
            _trainpath_flush_layer_events(torch, attempt, trace_events)
            for handle in handles:
                handle.remove()
            _lean_monitor_restore_bn(torch, model, current_bn)
            _lean_monitor_restore_rng(torch, current_rng)
            model.train(was_training)
        first_bad = (json.loads((attempt / "trainpath_first_nonfinite.json").read_text(encoding="utf-8"))
                     if replay_error is not None and (attempt / "trainpath_first_nonfinite.json").is_file()
                     else None)
        if first_bad:
            layer_state["first_nonfinite_layer"] = {
                key: first_bad.get(key) for key in (
                    "module", "io_role", "tensor_name", "shape", "numel", "finite_n",
                    "nonfinite_n", "finite_max_abs", "batch_index", "scope")}
    batch_rows = train_keys.iloc[row_positions]
    receipt = {"status": "FIRST_NONFINITE_TRAINPATH_CAPTURED", "run_id": RUN_ID,
        "attempt_id": attempt.name, "epoch": epoch, "batch_index": batch_index,
        "component": component, "component_detail": component_detail,
        "first_nonfinite_layer": layer_state["first_nonfinite_layer"],
        "batch_rows": int(len(row_positions)),
        "condition_keys": batch_rows.full_condition_key.astype(str).tolist(),
        "source_object_ids": batch_rows.source_object_id.astype(str).tolist(),
        "full_inchikeys": batch_rows.full_inchikey.astype(str).tolist(),
        "rng_state_saved_in_checkpoint": True,
        "failure_state_checkpoint": failure_state_path.name,
        "failure_state_checkpoint_sha256": sha256_file(failure_state_path),
        "last_finite_checkpoint": str(last_finite_checkpoint) if last_finite_checkpoint else None,
        "last_finite_checkpoint_sha256": sha256_file(last_finite_checkpoint)
            if last_finite_checkpoint is not None and last_finite_checkpoint.is_file() else None,
        "layer_replay_performed": replay_forward,
        "timestamp_utc": utc_now(), "inner_dev_numeric_rows_loaded": False,
        "pdo_outer_locked_numeric_values_opened": False}
    atomic_json(attempt / "lean_monitor_first_nonfinite.json", receipt)
    if _trainpath_trace_owner_path(attempt).is_file():
        owner = json.loads(_trainpath_trace_owner_path(attempt).read_text(encoding="utf-8"))
        if owner.get("status") == "OPEN":
            _close_trainpath_trace_owner(attempt, RUN_ID, attempt.name,
                                         "FAILED_CLOSED_FIRST_NONFINITE_CAPTURED")
    return receipt


def _monitored_main_prediction_summary(prediction: np.ndarray, *, expected_shape: tuple[int, int],
                                       first_layer: dict[str, Any] | None,
                                       batch_failures: list[dict[str, Any]]) -> dict[str, Any]:
    if prediction.shape != expected_shape:
        raise ValueError(f"monitored TRAIN prediction shape changed: {prediction.shape} != {expected_shape}")
    finite = np.isfinite(prediction)
    nonfinite_n = int(prediction.size - finite.sum())
    if nonfinite_n:
        return {"status": "UNDEFINED_NONFINITE_MAIN_DELTA",
            "object_macro_nmse": None, "object_macro_amplitude_ratio": None,
            "main_delta_nonfinite_n": nonfinite_n, "main_delta_denominator": int(prediction.size),
            "first_nonfinite_layer": first_layer,
            "nonfinite_batches": batch_failures, "finite_subset_metrics_computed": False}
    return {"status": "FINITE_FULL_TRAIN_MAIN_DELTA",
        "main_delta_nonfinite_n": 0, "main_delta_denominator": int(prediction.size),
        "first_nonfinite_layer": None, "nonfinite_batches": [],
        "finite_subset_metrics_computed": False}


def _np_arrays_equal_with_nan(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype.kind != right.dtype.kind:
        return False
    if left.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _first_nonfinite_optimizer_state(torch: Any, optimizer: Any) -> dict[str, Any] | None:
    for parameter_index, state in enumerate(optimizer.state.values()):
        for name, value in state.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                finite_n = int(torch.isfinite(value).sum().item())
                if finite_n != value.numel():
                    finite_values = value[torch.isfinite(value)]
                    return {"parameter_state_index": parameter_index, "state_name": str(name),
                        "shape": list(value.shape), "numel": int(value.numel()),
                        "finite_n": finite_n, "nonfinite_n": int(value.numel()-finite_n),
                        "finite_max_abs": float(finite_values.abs().max().cpu().item())
                            if finite_values.numel() else None}
    return None


def _trainpath_bn_snapshot(torch: Any, model: Any, stage: str,
                           previous: dict[str, tuple[Any, Any, int]] | None = None
                           ) -> tuple[dict[str, Any], dict[str, tuple[Any, Any, int]]]:
    rows: list[dict[str, Any]] = []
    current: dict[str, tuple[Any, Any, int]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        mean = module.running_mean.detach().clone() if module.running_mean is not None else None
        var = module.running_var.detach().clone() if module.running_var is not None else None
        count = (int(module.num_batches_tracked.detach().cpu().item())
                 if module.num_batches_tracked is not None else None)
        item: dict[str, Any] = {"module": name or "<root>", "stage": stage,
                               "num_batches_tracked": count}
        for label, tensor in (("running_mean", mean), ("running_var", var)):
            if tensor is None:
                item[f"{label}_finite"] = None
                item[f"{label}_max_abs"] = None
                item[f"{label}_delta_l2"] = None
                continue
            finite = bool(torch.isfinite(tensor).all().item())
            item[f"{label}_finite"] = finite
            item[f"{label}_max_abs"] = (float(tensor.abs().max().cpu().item()) if tensor.numel() else 0.0)
            before = previous.get(name) if previous else None
            prior_tensor = before[0 if label == "running_mean" else 1] if before else None
            item[f"{label}_delta_l2"] = (
                float(torch.linalg.vector_norm(tensor - prior_tensor).cpu().item())
                if prior_tensor is not None else None)
            if not finite:
                raise FloatingPointError(f"nonfinite BatchNorm {label} in {name} at {stage}")
        rows.append(item)
        current[name] = (mean, var, count)
    return {"stage": stage, "batchnorm_layer_count": len(rows), "layers": rows}, current


def _trainpath_parameter_snapshot(torch: Any, model: Any, attempt: Path, *,
                                  stage: str, previous: dict[str, Any] | None = None
                                  ) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        value = parameter.detach()
        finite_n = int(torch.isfinite(value).sum().item())
        if finite_n != value.numel():
            receipt = {"status": "FIRST_NONFINITE_TRAIN_PARAMETER", "stage": stage,
                       "parameter": name, "shape": list(value.shape),
                       "numel": int(value.numel()), "finite_n": finite_n,
                       "timestamp_utc": utc_now(), "train_only": True,
                       "inner_dev_numeric_rows_loaded": False}
            atomic_json(attempt / "trainpath_first_nonfinite.json", receipt)
            raise FloatingPointError(f"first nonfinite TRAIN parameter at {stage}/{name}")
        row = {"name": name, "shape": list(value.shape), "numel": int(value.numel()),
               "finite": True, "finite_n": finite_n,
               "max_abs": float(value.abs().max().cpu().item()) if value.numel() else 0.0}
        before = previous.get(name) if previous else None
        if before is not None:
            delta = value - before
            delta_finite_n = int(torch.isfinite(delta).sum().item())
            if delta_finite_n != delta.numel():
                atomic_json(attempt / "trainpath_first_nonfinite.json", {
                    "status": "FIRST_NONFINITE_PARAMETER_DELTA", "stage": stage,
                    "parameter": name, "shape": list(delta.shape), "numel": int(delta.numel()),
                    "finite_n": delta_finite_n, "timestamp_utc": utc_now(),
                    "train_only": True, "inner_dev_numeric_rows_loaded": False})
                raise FloatingPointError(f"first nonfinite TRAIN parameter delta at {stage}/{name}")
            row["delta_l2"] = float(torch.linalg.vector_norm(delta).cpu().item())
            row["delta_max_abs"] = float(delta.abs().max().cpu().item()) if delta.numel() else 0.0
        rows.append(row)
        current[name] = value.clone()
    return {"stage": stage, "parameter_count": len(rows), "parameters": rows}, current


def _trainpath_gradient_snapshot(torch: Any, model: Any, attempt: Path,
                                 batch_index: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_squared = 0.0
    global_max = 0.0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            rows.append({"name": name, "has_gradient": False, "finite": None,
                         "max_abs": None, "l2": None})
            continue
        value = gradient.detach()
        finite_n = int(torch.isfinite(value).sum().item())
        if finite_n != value.numel():
            finite_values = value[torch.isfinite(value)]
            receipt = {"status": "FIRST_NONFINITE_TRAIN_GRADIENT", "batch_index": batch_index,
                       "parameter": name, "shape": list(value.shape),
                       "numel": int(value.numel()), "finite_n": finite_n,
                       "finite_max_abs": float(finite_values.abs().max().cpu().item())
                           if finite_values.numel() else None,
                       "timestamp_utc": utc_now(), "train_only": True,
                       "inner_dev_numeric_rows_loaded": False}
            atomic_json(attempt / "trainpath_first_nonfinite.json", receipt)
            raise FloatingPointError(f"first nonfinite TRAIN gradient at batch {batch_index}/{name}")
        max_abs = float(value.abs().max().cpu().item()) if value.numel() else 0.0
        l2 = float(torch.linalg.vector_norm(value).cpu().item()) if value.numel() else 0.0
        total_squared += l2 * l2
        global_max = max(global_max, max_abs)
        rows.append({"name": name, "has_gradient": True, "finite": True,
                     "finite_n": finite_n, "numel": int(value.numel()),
                     "max_abs": max_abs, "l2": l2})
    return {"batch_index": batch_index, "parameter_gradient_count": len(rows),
            "gradient_global_l2": math.sqrt(total_squared), "gradient_max_abs": global_max,
            "parameters": rows}


def _trainpath_forward_losses(torch: Any, model: Any, x1: Any, x2: Any,
                              features: Any) -> tuple[Any, tuple[Any, ...]]:
    x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, _ = model(x1, features)
    z2, mu2, logvar2 = model.encode_x2(x2)
    x2_rec = model.decode_x2(z2)
    losses = model.loss(x1, x1_rec, mu1, logvar1, x2, x2_rec,
                        mu2, logvar2, x2_pred, mu_pred, logvar_pred)
    return x2_pred, losses


def fit_phase_transigen_trainpath_recovery(cfg: dict[str, Any], config_sha: str,
                                           attempt_id: str,
                                           reviewed_preflight_sha256: str | None,
                                           *, same_owner_gpu_chain: bool = False) -> dict[str, Any]:
    """Run the original fresh-init 300-epoch schedule with lean normal-path monitoring."""
    if (EXPERIMENT_ID != "EXP-055" or RUN_ID != EXP055_TRANSIGEN_FIXED300_LEAN_RUN_ID
            or attempt_id != "fit_attempt_001" or SOURCE_RUN_ID != DEFAULT_SOURCE_RUN_ID):
        raise ValueError("TranSiGen fixed300 fit is bound to the new immutable lean-monitor run fit_attempt_001")
    if cfg.get("authorization", {}).get("allowed_device_uuid") != GPU_UUID:
        raise ValueError("EXP-055 registered device UUID no longer matches the sole A5000 allowlist")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise RuntimeError("TranSiGen trainpath recovery must use only the approved A5000 UUID")
    preflight_path = RUN_DIR / "run_preflight_receipt.json"
    if not preflight_path.is_file() or not reviewed_preflight_sha256:
        raise RuntimeError("GPU fit requires the immutable CPU preflight receipt SHA")
    preflight_sha = sha256_file(preflight_path)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (preflight_sha != reviewed_preflight_sha256
            or preflight.get("status") != "PASS_CPU_TRAIN_IDENTITY_BUDGET_AND_DEVICE_GATE_GPU_NOT_STARTED"
            or preflight.get("run_id") != RUN_ID or preflight.get("attempt_id") != attempt_id
            or preflight.get("runner_sha256") != sha256_file(Path(__file__).resolve())
            or preflight.get("config_sha256") != config_sha
            or preflight.get("data_roles", {}).get("inner_dev_numeric_rows_loaded") is not False):
        raise RuntimeError("CPU preflight receipt is stale, mismatched, or failed")
    package_cap = 1.0
    prior_charge, prior_close_hashes = exp055_transigen_lean_monitor_settled_charge(
        include_cost_calibration=True)
    reservation = package_cap - prior_charge
    accounting = resource_accounting(EXPERIMENT_ROOT, RUN_ID, package_cap,
        resource_limits(cfg)["total_exp_cap_gpu_hours"],
        resource_limits(cfg)["exp_settled_before_artifact_ledgers_gpu_hours"], prior_charge)
    if (accounting["package_remaining_gpu_hours"] + 1e-12 < reservation
            or accounting["exp_remaining_gpu_hours"] + 1e-12 < reservation):
        raise RuntimeError("the recovery package cannot reserve its remaining balance after the immutable v4 charge")
    attempt = new_attempt_path(RUN_DIR, attempt_id)
    attempt.mkdir(parents=False, exist_ok=False)
    _open_trainpath_trace_owner(attempt, RUN_ID, attempt_id, os.getpid(),
                                sha256_file(Path(__file__).resolve()))
    process_id = os.getpid()
    runner_sha = sha256_file(Path(__file__).resolve())
    atomic_json(attempt / "pre_gpu_preflight_started.json", {
        "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID, "attempt": attempt_id,
        "status": "CPU_PREFLIGHT_STARTED", "started_at_utc": utc_now(),
        "wsl_pid": process_id, "runner_sha256": runner_sha, "config_sha256": config_sha,
        "source_run_id": SOURCE_RUN_ID, "gpu_started": False,
        "reserved_gpu_hours": 0.0, "attempt_gpu_cap_hours": reservation,
        "prior_recovery_charge_gpu_hours": prior_charge,
        "prior_recovery_close_receipt_sha256": prior_close_hashes})
    keys_path = SOURCE_RUN_DIR / "response_blind_conditions.tsv"
    verified_gate = _reuse_root_verified_exp055_train_gate()
    train_keys, values, feature_by_train_key, data_evidence = _load_exp055_trainpath_train_inputs_reusing_gate(
        verified_gate, keys_path, SOURCE_RUN_DIR)
    source_receipts = {name: SOURCE_RUN_DIR / filename for name, filename in {
        "freeze": "metadata_freeze_receipt.json", "axis": "gene_axis_receipt.json"}.items()}
    for label, path in source_receipts.items():
        if not path.is_file():
            raise FileNotFoundError(f"frozen response-blind {label} receipt missing: {path}")
    feature_matrix = np.stack([feature_by_train_key[key] for key in
                                train_keys.full_inchikey.astype(str)]).astype(np.float32, copy=False)
    if feature_matrix.shape != (419, 2048):
        raise ValueError("TRAIN ECFP4 rows do not match the fixed 419-condition identity table")
    allowed_run_entries = {attempt_id, "run_preflight_receipt.json"}
    if RUN_DIR.exists() and any(path.name not in allowed_run_entries for path in RUN_DIR.iterdir()):
        raise FileExistsError("recovery run directory contains artifacts outside the exact registered attempt/preflight/readback files")
    limits = resource_limits(cfg)
    if (float(limits["total_exp_cap_gpu_hours"]) != 48.0
            or float(limits["package_cap_gpu_hours"]) != 10.0):
        raise ValueError("the registered EXP-055 parent cap differs from the frozen 48/10 GPUh limits")
    snapshot = (_device_snapshot() if same_owner_gpu_chain else _assert_exp055_gpu_idle())
    if snapshot.get("selected_gpu", {}).get("uuid") != GPU_UUID:
        raise RuntimeError("formal TranSiGen run is not bound to the one approved A5000")
    baseline_metrics = None
    gene_variance = np.var(values["delta"].astype(np.float64), axis=0, ddof=0)
    # TRAIN-only fit gate baseline; no cached all-role metrics are opened.
    train_group_means = train_keys.assign(_row=np.arange(len(train_keys))).groupby(
        "full_inchikey", sort=True)["_row"].apply(list)
    per_drug = np.empty_like(values["delta"])
    for positions in train_group_means.tolist():
        pos = np.asarray(positions, dtype=np.int64)
        per_drug[pos] = values["delta"][pos].mean(axis=0, dtype=np.float64).astype(np.float32)
    zero = np.zeros_like(values["delta"], dtype=np.float32)
    baseline_metrics = {
        "zero_change": keyed_metrics(train_keys, values["delta"], zero, gene_variance),
        "per_drug_mean_delta": keyed_metrics(train_keys, values["delta"], per_drug, gene_variance),
    }
    recovery_fit_gate_baselines = _wrap_trainpath_recovery_baselines_for_fit_gate(baseline_metrics)
    prior_timing_path = (ROOT / "artifacts/experiments/EXP-051/"
                         "20261006_exp051_representative_compare_v1/fit_attempt_007/"
                         "failure_close_receipt.json")
    if not prior_timing_path.is_file():
        prior_timing_path = (RUN_DIR.parent / "20261007_exp055_representative_models_v1/"
                             "fit_attempt_007/pre_gpu_failure_receipt.json")
    prior_timing_sha = sha256_file(prior_timing_path) if prior_timing_path.is_file() else None
    command = ("CUDA_VISIBLE_DEVICES=" + GPU_UUID + " PYTHONPATH=src "
        "/home/dell/miniconda3/bin/conda run --no-capture-output -n drugscreening-gpu python "
        "scripts/modeling/exp051_contract_fit_smoke.py --experiment-id EXP-055 "
        "--fit-kind representative --phase fit --models transigen "
        "--trainpath-recovery --config configs/experiments/EXP-055-representative-models.yaml "
         "--source-run-id " + SOURCE_RUN_ID + " --run-id " + RUN_ID + " --attempt-id " + attempt_id
         + " --reviewed-preflight-sha256 " + str(reviewed_preflight_sha256))
    manifest = {"experiment_id": EXPERIMENT_ID, "run_id": RUN_ID, "fit_attempt": attempt_id,
        "status": "CPU_IDENTITY_AND_BUDGET_GATE_PASS_GPU_NOT_YET_STARTED",
        "created_at_utc": utc_now(), "wsl_pid": process_id, "command": command,
        "python_executable": sys.executable, "torch_import_deferred_until_after_cpu_gate": True,
        "source_code_path": str(Path(__file__).resolve()), "runner_sha256": runner_sha,
        "current_runner_sha256": runner_sha,
        "config_path": str(ROOT / "configs/experiments/EXP-055-representative-models.yaml"),
        "config_sha256": config_sha, "source_run_id": SOURCE_RUN_ID,
        "source_receipts": {name: {"path": str(path), "sha256": sha256_file(path)}
                            for name, path in source_receipts.items()},
        "train_only_data_identity_gate": data_evidence,
        "train_baseline_metrics": baseline_metrics,
        "prior_attempt007_receipt_path": str(prior_timing_path) if prior_timing_path.is_file() else None,
        "prior_attempt007_receipt_sha256": prior_timing_sha,
        "bound_cpu_preflight_receipt_sha256": reviewed_preflight_sha256,
        "attempt_gpu_cap_hours": reservation, "reserved_gpu_hours": reservation,
        "prior_recovery_attempts": {"run_ids": [
            "20261007_exp055_transigen_trainpath_recovery_v4",
            "20261007_exp055_transigen_trainpath_recovery_v5",
            "20261007_exp055_transigen_trainpath_recovery_v9",
            "20261007_exp055_transigen_trainpath_recovery_v11",
            EXP055_TRANSIGEN_COST_SOURCE_RUN_ID,
            EXP055_TRANSIGEN_LEAN_COST_RUN_ID],
            "charged_gpu_hours": prior_charge, "failure_close_receipt_sha256": prior_close_hashes,
            "remaining_same_package_cap_gpu_hours": reservation},
        "resource_accounting_before_attempt": accounting,
        "resource_settled_package_gpu_hours": accounting["package_settled_gpu_hours"],
        "resource_settled_exp_gpu_hours": accounting["exp_settled_gpu_hours"],
        "device_preflight": snapshot,
        "training_contract": {"architecture": "upstream original TranSiGen",
            "train_conditions": 419, "train_objects": 110, "genes": 978,
            "full_inchikey_count": 5, "objective": "raw_sum", "gradient_clipping": False,
            "learning_rate": learning_rate, "weight_decay": 1e-5, "batch_size": 64,
            "batch_sizes": [64, 64, 64, 64, 64, 64, 35], "shuffle": True,
            "batchnorm": "ordinary TRAIN batches only; no calibration",
            "monitoring": "lean per-batch finite checks and compact epoch summaries; full module replay only after anomaly",
            "full_schedule_epochs": 300, "checkpoints_if_cost_gate_passes": [100, 300, "final"]},
        "data_roles": {"inner_dev_numeric_rows_loaded": False,
            "frozen_train_inner_values_npz_opened": False,
            "pdo_numeric_sensitivity_or_expression_opened": False,
            "outer_or_locked_numeric_values_opened": False,
            "inner_dev_used_for_model_selection": False}}
    atomic_json(attempt / "fit_smoke_manifest.json", manifest)
    atomic_json(attempt / "cpu_preflight_gate.json", {
        "status": "PASS_TRAIN_ONLY_IDENTITY_BUDGET_AND_NO_OVERWRITE_GATES",
        "at_utc": utc_now(), "runner_sha256": runner_sha, "config_sha256": config_sha,
        "run_id": RUN_ID, "attempt": attempt_id, "train_conditions": 419,
        "train_objects": 110, "full_inchikey_count": 5, "genes": 978,
        "train_source_rows_sha256": data_evidence["train_source_rows_sha256"],
        "train_condition_keys_sha256": data_evidence["train_condition_keys_sha256"],
        "train_delta_sha256": hashlib.sha256(values["delta"].tobytes()).hexdigest(),
        "train_feature_matrix_sha256": data_evidence["feature_matrix_sha256"],
        "settled_gpu_hours_before_attempt": accounting["exp_settled_gpu_hours"],
        "global_gpu_cap_hours": 48.0,
        "exp055_package_cap_hours": 10.0,
        "new_attempt_cap_hours": reservation,
        "prior_recovery_charge_gpu_hours": prior_charge,
        "attempt_directory_unique": True, "gpu_idle_and_allowed_uuid_pass": True,
        "inner_dev_numeric_rows_loaded": False, "pdo_outer_locked_values_opened": False})
    ledger = attempt / "resource_ledger.jsonl"
    append_jsonl(ledger, {"event": "GPU_ATTEMPT_STARTED", "at_utc": utc_now(),
        "wsl_pid": process_id, "device_uuid": GPU_UUID, "selected_models": ["transigen"],
        "reserved_gpu_hours": reservation, "reservation_scope": "one TRAIN-path trace plus cost-gated original 300-epoch TranSiGen schedule",
        "package_cap_gpu_hours": package_cap,
        "package_settled_before_attempt_gpu_hours": accounting["package_settled_gpu_hours"],
        "exp_settled_before_attempt_gpu_hours": accounting["exp_settled_gpu_hours"],
        "remaining_authorized_gpu_hours": accounting["remaining_authorized_gpu_hours"],
        "runner_sha256": runner_sha, "config_sha256": config_sha,
        "device_snapshot_before_attempt": snapshot["device_snapshot"]["selected_gpu"]})

    import torch
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in approved WSL2 Conda drugscreening-gpu")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    gpu_start = time.perf_counter()
    guard_seconds = reservation * 3600.0 - 60.0
    def budget_check() -> None:
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - gpu_start
        progress.update({"phase": "TRAINPATH_RECOVERY_RUNNING", "updated_at_utc": utc_now(),
            "wsl_pid": process_id, "elapsed_gpu_hours_upper_bound": elapsed / 3600.0,
            "attempt_cap_gpu_hours": reservation, "batch_index": progress.get("batch_index"),
            "inner_dev_numeric_rows_loaded": False})
        if elapsed >= guard_seconds:
            atomic_json(attempt / "attempt_progress.json", progress)
            raise TimeoutError(f"TranSiGen reached the 1.0 GPUh recovery guard at {elapsed:.1f}s")

    progress = {"phase": "GPU_READY_PRE_MODEL_INITIALIZATION", "updated_at_utc": utc_now(),
                "wsl_pid": process_id, "completed_epochs": 0, "target_epochs": 300}
    atomic_json(attempt / "attempt_progress.json", progress)
    model, model_module = _load_transigen(torch, device, SEED, features_dim=2048)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    train_features = feature_matrix
    train_y = values["delta"]
    train_positions = np.arange(419, dtype=np.int64)
    init_params, init_param_state = _trainpath_parameter_snapshot(
        torch, model, attempt, stage="model_initialization")
    atomic_json(attempt / "trainpath_initial_parameter_snapshot.json", init_params)
    initial_bn, previous_bn = _trainpath_bn_snapshot(torch, model, "model_initialization")
    initial_finite_checkpoint = attempt / "TranSiGen_last_finite.pt"
    if initial_finite_checkpoint.exists():
        raise FileExistsError("refusing to overwrite the initial last-finite checkpoint")
    _atomic_torch_checkpoint(torch, initial_finite_checkpoint, {
        "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "model": "upstream TranSiGen", "epoch": 0, "seed": SEED,
        "objective": "raw_sum", "gradient_clipping": False,
        "learning_rate": 1e-3, "weight_decay": 1e-5,
        "fit_scope": "TRAIN_ONLY_NO_INNER_DEV_VALUES",
        "checkpoint_role": "last_finite_state_before_any_optimizer_update"})
    atomic_json(attempt / "last_finite_checkpoint_receipt.json", {
        "status": "PASS_INITIAL_FINITE_CHECKPOINT", "checkpoint": initial_finite_checkpoint.name,
        "checkpoint_sha256": sha256_file(initial_finite_checkpoint), "epoch": 0,
        "optimizer_updates": 0, "created_at_utc": utc_now(),
        "inner_dev_numeric_rows_loaded": False})
    bn_trace = [initial_bn]
    events: list[dict[str, Any]] = []
    event_state = {"events": events, "scope": "initial_train_mode_no_grad_snapshot", "batch_index": None}
    handles: list[Any] = []
    try:
        model.train()
        # Diagnostic dry-forward uses one fixed TRAIN batch. Restore random streams and
        # BatchNorm buffers afterwards, so it records the normal TRAIN path without
        # inserting a calibration update into the actual optimizer schedule.
        first = slice(0, 64)
        x1_init = torch.as_tensor(values["X_ctl"][first], dtype=torch.float32, device=device)
        x2_init = torch.as_tensor(values["X"][first], dtype=torch.float32, device=device)
        feat_init = torch.as_tensor(train_features[first], dtype=torch.float32, device=device)
        rng_before_initial = _lean_monitor_rng_state(torch)
        rng_python, rng_numpy, rng_torch = (rng_before_initial["python"], rng_before_initial["numpy"],
                                            rng_before_initial["torch_cpu"])
        rng_cuda = rng_before_initial["torch_cuda"]
        bn_buffers = {name: {key: (buffer.detach().clone() if buffer is not None else None)
                             for key, buffer in module.named_buffers(recurse=False)}
                      for name, module in model.named_modules()
                      if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}
        with torch.no_grad():
            prediction_init, losses_init = _trainpath_forward_losses(
                torch, model, x1_init, x2_init, feat_init)
            delta_init = prediction_init - x1_init
            loss_init_values = [float(loss.detach().cpu().item()) for loss in losses_init]
            if len(losses_init) != 7 or not all(math.isfinite(value) for value in loss_init_values):
                _lean_monitor_failure_context(torch, model, optimizer, attempt,
                    epoch=0, batch_index=0, component="initial_training_loss",
                    train_keys=train_keys, row_positions=np.arange(64, dtype=np.int64),
                    x1=x1_init, x2=x2_init, features=feat_init,
                    rng_before_batch=rng_before_initial, bn_before_batch=bn_buffers,
                    last_finite_checkpoint=initial_finite_checkpoint,
                    component_detail={"loss_values": loss_init_values})
                atomic_json(attempt / "trainpath_first_nonfinite.json", {
                    "status": "FIRST_NONFINITE_INITIAL_TRAIN_LOSS", "scope": event_state["scope"],
                    "loss_values": loss_init_values, "timestamp_utc": utc_now(),
                    "train_only": True, "inner_dev_numeric_rows_loaded": False})
                raise FloatingPointError("nonfinite initial TRAIN loss or unexpected TranSiGen loss tuple")
            initial_main_delta_finite = bool(torch.isfinite(delta_init).all().item())
            if not initial_main_delta_finite:
                _lean_monitor_failure_context(torch, model, optimizer, attempt,
                    epoch=0, batch_index=0, component="initial_training_main_delta",
                    train_keys=train_keys, row_positions=np.arange(64, dtype=np.int64),
                    x1=x1_init, x2=x2_init, features=feat_init,
                    rng_before_batch=rng_before_initial, bn_before_batch=bn_buffers,
                    last_finite_checkpoint=initial_finite_checkpoint,
                    component_detail={"main_delta_shape": list(delta_init.shape),
                        "main_delta_nonfinite_n": int((~torch.isfinite(delta_init)).sum().item())})
        _trainpath_flush_layer_events(torch, attempt, events)
        initial_bn_after, transient_bn = _trainpath_bn_snapshot(
            torch, model, "initial_train_mode_no_grad_forward_after", previous_bn)
        bn_trace.append(initial_bn_after)
        # Roll back only this no-gradient diagnostic's BN/RNG side effects before the
        # first real optimizer epoch; no fitted state or calibration is retained.
        with torch.no_grad():
            for name, module in model.named_modules():
                if name in bn_buffers:
                    for key, buffer in module.named_buffers(recurse=False):
                        saved = bn_buffers[name].get(key)
                        if buffer is not None and saved is not None:
                            buffer.copy_(saved)
        random.setstate(rng_python)
        np.random.set_state(rng_numpy)
        torch.set_rng_state(rng_torch)
        torch.cuda.set_rng_state_all(rng_cuda)
    finally:
        for handle in handles:
            handle.remove()
    init_delta_summary, _ = _trainpath_parameter_snapshot(
        torch, model, attempt, stage="after_initial_dry_forward", previous=init_param_state)
    if any(float(row.get("delta_l2", 0.0)) != 0.0 for row in init_delta_summary["parameters"]):
        raise RuntimeError("no-gradient initialization snapshot changed a trainable parameter")
    if not initial_main_delta_finite:
        atomic_json(attempt / "trainpath_initial_forward_failure.json", {
            "status": "FIRST_NONFINITE_INITIAL_TRAIN_MAIN_DELTA",
            "created_at_utc": utc_now(), "scope": "initial_train_mode_no_grad_snapshot",
            "train_conditions": 64, "main_delta_shape": list(delta_init.shape),
            "main_delta_nonfinite_n": int((~torch.isfinite(delta_init)).sum().item()),
            "main_delta_denominator": int(delta_init.numel()),
            "layer_trace": "TranSiGen_trainpath_layer_trace.jsonl",
            "last_finite_checkpoint": initial_finite_checkpoint.name,
            "inner_dev_numeric_rows_loaded": False,
            "pdo_outer_locked_numeric_values_opened": False})
        raise FloatingPointError("nonfinite initial TRAIN main-delta output")
    initial_snapshot = {"status": "PASS_TRAIN_ONLY_INITIAL_FORWARD_FINITE",
        "created_at_utc": utc_now(), "mode": "model.train() with torch.no_grad()",
        "train_conditions_in_forward": 64, "loss_terms": loss_init_values,
        "main_delta_shape": list(delta_init.shape),
        "main_delta_finite": bool(torch.isfinite(delta_init).all().item()),
        "main_delta_max_abs": float(delta_init.abs().max().cpu().item()),
        "batchnorm_state": bn_trace, "dry_forward_batchnorm_state_rolled_back": True,
        "python_numpy_cpu_cuda_rng_restored": True,
        "parameters_unchanged": True, "inner_dev_numeric_rows_loaded": False,
        "pdo_outer_locked_values_opened": False}
    atomic_json(attempt / "trainpath_initial_forward_snapshot.json", initial_snapshot)
    del x1_init, x2_init, feat_init, prediction_init, losses_init, delta_init

    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(torch.as_tensor(values["X_ctl"], dtype=torch.float32),
        torch.as_tensor(values["X"], dtype=torch.float32),
        torch.as_tensor(train_features, dtype=torch.float32), torch.arange(419, dtype=torch.int64))
    loader_generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False,
        num_workers=0, generator=loader_generator)
    batch_sizes = [int(batch[0].shape[0]) for batch in loader]
    loader_generator.manual_seed(SEED)
    if batch_sizes != [64, 64, 64, 64, 64, 64, 35]:
        raise ValueError(f"TRAIN recovery loader batch contract changed: {batch_sizes}")
    curve_path = attempt / "TranSiGen_trainpath_recovery_curve.jsonl"
    if curve_path.exists():
        raise FileExistsError("refusing to overwrite the current attempt's TRAIN curve")
    _require_trainpath_trace_owner(attempt, RUN_ID, attempt_id)
    loss_names = ("raw_total_loss", "mse_x1", "mse_x2", "mse_pert",
                  "kld_x1", "kld_x2", "kld_pert")
    torch.cuda.synchronize()
    epoch_start = time.perf_counter()
    before_params = [parameter.detach().clone() for parameter in model.parameters()]
    gradient_trace: list[dict[str, Any]] = []
    parameter_delta_trace: list[dict[str, Any]] = []
    batch_losses: list[dict[str, Any]] = []
    seen = 0
    gradient_norms: list[float] = []
    batch_index = 0
    event_state = {"events": [], "scope": "train_epoch1", "batch_index": None}
    handles: list[Any] = []
    lean_batch_trace = attempt / "lean_monitor_batch_trace.jsonl"
    lean_batch_trace.touch(exist_ok=False)
    try:
        model.train()
        for x1_cpu, x2_cpu, feat_cpu, row_cpu in loader:
            batch_index += 1
            budget_check()
            batch_start = time.perf_counter()
            event_state["batch_index"] = batch_index
            rng_before = _lean_monitor_rng_state(torch)
            bn_buffers_before = _lean_monitor_bn_buffers(torch, model)
            row_positions = row_cpu.numpy().astype(np.int64, copy=False)
            x1, x2, features = x1_cpu.to(device), x2_cpu.to(device), feat_cpu.to(device)
            model.zero_grad(set_to_none=True)
            prediction, losses = _trainpath_forward_losses(torch, model, x1, x2, features)
            loss_values = [float(loss.detach().cpu().item()) for loss in losses]
            if len(losses) != 7 or not all(math.isfinite(value) for value in loss_values):
                failure = _lean_monitor_failure_context(torch, model, optimizer, attempt,
                    epoch=1, batch_index=batch_index, component="training_loss",
                    train_keys=train_keys, row_positions=row_positions, x1=x1, x2=x2,
                    features=features, rng_before_batch=rng_before,
                    bn_before_batch=bn_buffers_before, last_finite_checkpoint=initial_finite_checkpoint,
                    component_detail={"loss_names": list(loss_names), "loss_values": loss_values})
                raise FloatingPointError(f"nonfinite TRAIN loss at epoch 1 batch {batch_index}: {failure}")
            main_delta = prediction - x1
            if (prediction.shape != x1.shape or not bool(torch.isfinite(prediction).all().item())
                    or not bool(torch.isfinite(main_delta).all().item())):
                failure = _lean_monitor_failure_context(torch, model, optimizer, attempt,
                    epoch=1, batch_index=batch_index, component="training_main_delta",
                    train_keys=train_keys, row_positions=row_positions, x1=x1, x2=x2,
                    features=features, rng_before_batch=rng_before,
                    bn_before_batch=bn_buffers_before, last_finite_checkpoint=initial_finite_checkpoint,
                    component_detail={"prediction_shape": list(prediction.shape),
                        "expected_shape": list(x1.shape),
                        "prediction_nonfinite_n": int((~torch.isfinite(prediction)).sum().item()),
                        "main_delta_nonfinite_n": int((~torch.isfinite(main_delta)).sum().item())})
                raise FloatingPointError(f"nonfinite TRAIN main delta at epoch 1 batch {batch_index}: {failure}")
            losses[0].backward()
            grad_finite, grad_norm, bad_gradient, grad_norm_finite = _lean_monitor_gradient_summary(torch, model)
            if not grad_finite:
                failure = _lean_monitor_failure_context(torch, model, optimizer, attempt,
                    epoch=1, batch_index=batch_index, component="gradient",
                    train_keys=train_keys, row_positions=row_positions, x1=x1, x2=x2,
                    features=features, rng_before_batch=rng_before,
                    bn_before_batch=bn_buffers_before, last_finite_checkpoint=initial_finite_checkpoint,
                    component_detail={"gradient_elements_finite": grad_finite,
                        "gradient_norm_finite": grad_norm_finite,
                        "first_nonfinite_parameter_gradient": bad_gradient,
                        "gradient_norm": safe_number(grad_norm)})
                raise FloatingPointError(f"nonfinite TRAIN gradient at batch {batch_index}: {failure}")
            gradient_norms.append(grad_norm if grad_norm_finite else math.nan)
            optimizer.step()
            bad_optimizer_state = _first_nonfinite_optimizer_state(torch, optimizer)
            params_finite, bad_parameter = _lean_monitor_finite_named_tensors(
                torch, [(name, parameter) for name, parameter in model.named_parameters()])
            bn_finite, bad_bn = _lean_monitor_finite_named_tensors(
                torch, _lean_monitor_batchnorm_named_tensors(model))
            if bad_optimizer_state is not None or not params_finite or not bn_finite:
                component = ("optimizer_state" if bad_optimizer_state is not None else
                             "parameter" if not params_finite else "batchnorm_state")
                failure = _lean_monitor_failure_context(torch, model, optimizer, attempt,
                    epoch=1, batch_index=batch_index, component=component,
                    train_keys=train_keys, row_positions=row_positions, x1=x1, x2=x2,
                    features=features, rng_before_batch=rng_before,
                    bn_before_batch=bn_buffers_before, last_finite_checkpoint=initial_finite_checkpoint,
                    replay_forward=False,
                    component_detail={"optimizer_state": bad_optimizer_state,
                        "first_nonfinite_parameter": bad_parameter,
                        "first_nonfinite_batchnorm_buffer": bad_bn})
                raise FloatingPointError(f"nonfinite model/optimizer state after batch {batch_index}: {failure}")
            gradient_trace.append({"batch_index": batch_index,
                "gradient_global_l2": safe_number(grad_norm),
                "gradient_elements_finite": grad_finite,
                "gradient_norm_finite": grad_norm_finite,
                "gradient_finite": grad_finite})
            batch_losses.append({"batch_index": batch_index, "conditions": int(len(x1)),
                                 "losses": dict(zip(loss_names, loss_values)),
                                  "gradient_global_l2": safe_number(grad_norm)})
            seen += int(len(x1))
            append_jsonl(lean_batch_trace, {"epoch": 1, "batch_index": batch_index,
                "conditions": int(len(x1)), "losses": dict(zip(loss_names, loss_values)),
                "loss_finite": True, "gradient_norm": safe_number(grad_norm),
                "gradient_elements_finite": grad_finite,
                "gradient_norm_finite": grad_norm_finite, "gradient_finite": grad_finite,
                "parameters_finite": params_finite, "optimizer_finite": bad_optimizer_state is None,
                "batchnorm_finite": bn_finite, "batch_seconds": time.perf_counter() - batch_start,
                "inner_dev_numeric_rows_loaded": False})
            del x1, x2, features, prediction, main_delta, losses
        if seen != 419 or batch_index != 7:
            raise ValueError(f"TRAIN epoch must update all 419 conditions in seven batches; got {seen}/{batch_index}")
    finally:
        for handle in handles:
            handle.remove()
    torch.cuda.synchronize()
    epoch_seconds = time.perf_counter() - epoch_start
    bn_final, previous_bn = _trainpath_bn_snapshot(torch, model, "epoch1_complete", previous_bn)
    bn_trace.append(bn_final)
    final_param = {"parameters_finite": True, "parameter_global_l2_norm": _param_global_norm(torch, model),
                   "parameter_update_l2": _norm_difference(torch, before_params, model)}
    del before_params
    # Coherent post-epoch predictions and fit metrics use TRAIN rows only.
    eval_start = time.perf_counter()
    rng_state = _lean_monitor_rng_state(torch)
    eval_batch_failures: list[dict[str, Any]] = []
    eval_layer_state = {"first_nonfinite_layer": None}
    train_prediction, train_eval_losses = _transigen_predict_and_losses(
        torch, model, values["X_ctl"], values["X"], train_features, 64, device,
        SEED + 1, allow_nonfinite_auxiliary=True, allow_nonfinite_main=True,
        progress_check=budget_check,
        nonfinite_main_delta_callback=eval_batch_failures.append)
    _lean_monitor_restore_rng(torch, rng_state)
    train_eval_seconds = time.perf_counter() - eval_start
    eval_summary = _monitored_main_prediction_summary(train_prediction, expected_shape=(419, 978),
        first_layer=eval_layer_state["first_nonfinite_layer"], batch_failures=eval_batch_failures)
    if eval_summary["main_delta_nonfinite_n"] > 0:
        eval_failure = _lean_monitor_eval_failure_replay(torch, model, optimizer, attempt,
            train_keys=train_keys, X_ctl=values["X_ctl"], X=values["X"], features=train_features,
            device=device, seed=SEED + 1, first_failure=eval_batch_failures[0])
        eval_summary["first_nonfinite_layer"] = eval_failure["first_nonfinite_layer"]
    if eval_summary["main_delta_nonfinite_n"] == 0:
        train_metrics = keyed_metrics(train_keys, train_y, train_prediction, gene_variance)
        train_metrics.update({key: eval_summary[key] for key in (
            "status", "main_delta_nonfinite_n", "main_delta_denominator", "first_nonfinite_layer")})
        fit_gate = _fit_gate(train_metrics, recovery_fit_gate_baselines)
    else:
        train_metrics = eval_summary
        fit_gate = {"status": "UNDEFINED_NONFINITE_MAIN_DELTA",
            "reason": "full 419x978 TRAIN prediction contains nonfinite values; no NMSE or finite-subset metric",
            "main_delta_nonfinite_n": eval_summary["main_delta_nonfinite_n"],
            "main_delta_denominator": eval_summary["main_delta_denominator"],
            "first_nonfinite_layer": eval_summary["first_nonfinite_layer"]}
    prediction_path = attempt / "TranSiGen_epoch1_train_keyed_predictions.npz"
    prediction_save_started = time.perf_counter()
    atomic_npz(prediction_path,
        train_condition_key=train_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
        train_source_object_id=train_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
        train_full_inchikey=train_keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
        genes=values["genes"].astype(str), train_target_delta=train_y,
        train_delta=train_prediction)
    with np.load(prediction_path, allow_pickle=False) as saved:
        expected = {"train_condition_key": train_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
                    "train_source_object_id": train_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
                    "train_full_inchikey": train_keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
                    "genes": values["genes"].astype(str), "train_target_delta": train_y,
                    "train_delta": train_prediction}
        if any(name not in saved.files or not _np_arrays_equal_with_nan(saved[name], value)
               for name, value in expected.items()):
            raise ValueError("TRAIN-only recovery prediction write/readback failed")
    prediction_readback_seconds = time.perf_counter() - prediction_save_started
    atomic_json(attempt / "trainpath_epoch1_trace.json", {
        "status": "COMPLETED_EPOCH1_TRAIN_PATH_WITH_MONITORED_EVAL",
        "created_at_utc": utc_now(), "epoch": 1, "train_conditions": seen,
        "train_batch_sizes": batch_sizes, "loss_names": loss_names,
        "batch_loss_and_gradient_trace": batch_losses,
        "gradient_by_parameter": gradient_trace,
        "parameter_delta_by_batch_and_parameter": parameter_delta_trace,
        "parameter_delta_epoch": final_param,
        "batchnorm_state_trajectory": bn_trace,
        "train_layer_trace_jsonl": "TranSiGen_trainpath_layer_trace.jsonl",
        "train_prediction_metrics": train_metrics, "train_fit_gate": fit_gate,
        "train_prediction_evaluation": eval_summary,
        "train_evaluation_auxiliary_losses": train_eval_losses,
        "train_prediction_path": prediction_path.name,
        "train_prediction_sha256": sha256_file(prediction_path),
        "prediction_readback": "PASS_EXACT_KEYS_AXES_AND_VALUES_INCLUDING_NONFINITE_TOKENS",
        "train_epoch_seconds_with_lean_monitoring": epoch_seconds,
        "train_eval_seconds": train_eval_seconds,
        "prediction_save_and_readback_seconds": prediction_readback_seconds,
        "inner_dev_numeric_rows_loaded": False,
        "pdo_outer_locked_values_opened": False})
    elapsed = time.perf_counter() - gpu_start
    cost_checkpoint_started = time.perf_counter()
    checkpoint_payload = {
        "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "model": "upstream TranSiGen", "epoch": 1, "seed": SEED,
        "objective": "raw_sum", "gradient_clipping": False,
        "learning_rate": 1e-3, "weight_decay": 1e-5,
        "fit_scope": "TRAIN_ONLY_NO_INNER_DEV_VALUES",
        "purpose": "preserve the measured real epoch if the 300-epoch cost gate closes"}
    _atomic_torch_checkpoint(torch, attempt / "TranSiGen_epoch_1_cost_gate_checkpoint.pt", checkpoint_payload)
    _atomic_torch_checkpoint(torch, initial_finite_checkpoint, checkpoint_payload)
    atomic_json(attempt / "last_finite_checkpoint_receipt.json", {
        "status": "PASS_LAST_FINITE_COMPLETED_TRAIN_EPOCH_CHECKPOINT",
        "checkpoint": initial_finite_checkpoint.name,
        "checkpoint_sha256": sha256_file(initial_finite_checkpoint), "epoch": 1,
        "optimizer_updates": 7, "created_at_utc": utc_now(),
        "inner_dev_numeric_rows_loaded": False})
    checkpoint_save_seconds = time.perf_counter() - cost_checkpoint_started
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - gpu_start
    projected_final_dev_eval_seconds = train_eval_seconds * (512.0 / 419.0)
    projected_one_checkpoint_io_seconds = checkpoint_save_seconds / 2.0
    projected_prediction_save_seconds = prediction_readback_seconds * 4.5
    projected_checkpoint_io_seconds = checkpoint_save_seconds * 3.0
    projected_json_and_close_seconds = 5.0
    projected_without_margin_seconds = (
        elapsed + 299 * (epoch_seconds + train_eval_seconds + projected_one_checkpoint_io_seconds)
        + projected_final_dev_eval_seconds + projected_prediction_save_seconds
        + projected_checkpoint_io_seconds + projected_json_and_close_seconds)
    projected_with_margin = projected_without_margin_seconds * 1.20 / 3600.0
    cost_gate = {"status": "PASS_FULL_300_EPOCH_COST_GATE" if projected_with_margin <= reservation
                    else "STOP_AFTER_TRAIN_ONLY_DIAGNOSTIC_COST_OVER_CAP",
        "observed_epoch1_train_seconds_with_lean_monitoring": epoch_seconds,
        "observed_train_eval_seconds": train_eval_seconds,
        "observed_prediction_save_and_readback_seconds": prediction_readback_seconds,
        "observed_epoch1_checkpoint_atomic_save_seconds": checkpoint_save_seconds,
        "already_elapsed_gpu_hours": elapsed / 3600.0,
        "projected_remaining_epochs": 299,
        "projected_postcheckpoint_train_plus_inner_dev_eval_seconds": projected_final_dev_eval_seconds,
        "projected_endpoint_and_final_prediction_io_seconds": projected_prediction_save_seconds,
        "projected_checkpoint_write_and_read_seconds": projected_checkpoint_io_seconds,
        "projected_per_epoch_last_finite_checkpoint_seconds": projected_one_checkpoint_io_seconds,
        "projected_final_json_close_seconds": projected_json_and_close_seconds,
        "full_300_epoch_projection_including_20pct_margin_gpu_hours": projected_with_margin,
        "attempt_cap_gpu_hours": reservation,
        "estimate_assumes_each_remaining_epoch_not_faster_than_lean_epoch1_plus_eval_and_atomic_checkpoint": True,
        "per_epoch_train_evaluation": True,
        "fixed_epoch100_diagnostic_cost_included": True,
        "main_delta_evaluation_nonfinite": eval_summary["main_delta_nonfinite_n"] > 0,
        "main_delta_nonfinite_n": eval_summary["main_delta_nonfinite_n"],
        "main_delta_denominator": eval_summary["main_delta_denominator"],
        "first_nonfinite_layer": eval_summary["first_nonfinite_layer"],
        "inner_dev_numeric_rows_loaded": False}
    atomic_json(attempt / "trainpath_cost_gate.json", cost_gate)
    append_jsonl(curve_path, {"epoch": 1, "objective": "raw_sum", "gradient_clipping": False,
        "learning_rate": 1e-3, "weight_decay": 1e-5, "batch_size": 64, "train_shuffle": True,
        "train_conditions": seen, "train_batch_sizes": batch_sizes,
        "loss_terms_mean_per_condition": {name: safe_number(float(sum(
            item["losses"][name] * item["conditions"] for item in batch_losses) / seen))
            for name in loss_names},
        "gradient_norm_mean": safe_number(float(np.mean(gradient_norms))),
        "gradient_norm_max": safe_number(float(np.max(gradient_norms))),
        "gradient_norm_finite_n_updates": sum(math.isfinite(value) for value in gradient_norms),
        "parameter_update_l2": safe_number(float(math.sqrt(sum(
            float(item["delta_l2"]) ** 2 for item in final_param["parameters"])))),
        "parameter_global_l2_norm_after_epoch": _param_global_norm(torch, model),
        "train_metrics": train_metrics, "train_fit_gate": fit_gate,
        "train_evaluation_auxiliary_losses": train_eval_losses,
        "epoch_seconds_with_layer_trace": epoch_seconds,
        "train_eval_seconds": train_eval_seconds, "finite_training_and_main_delta": True})
    if projected_with_margin > reservation:
        elapsed = time.perf_counter() - gpu_start
        trace_owner = _close_trainpath_trace_owner(attempt, RUN_ID, attempt_id,
                                                   "CLOSED_AFTER_TRAIN_ONLY_TRACE_COST_GATE")
        close = {"status": "CLOSED_AFTER_TRAIN_ONLY_TRACE_COST_GATE",
            "completed_train_epochs": 1, "selected_epoch": None,
            "actual_gpu_hours": elapsed / 3600.0, "reserved_gpu_hours": reservation,
            "released_unused_reservation_gpu_hours": max(0.0, reservation - elapsed / 3600.0),
            "cost_gate": cost_gate, "trainpath_epoch1_trace_sha256": sha256_file(attempt / "trainpath_epoch1_trace.json"),
            "trainpath_trace_owner_sha256": sha256_file(_trainpath_trace_owner_path(attempt)),
            "epoch1_cost_gate_checkpoint_sha256": sha256_file(attempt / "TranSiGen_epoch_1_cost_gate_checkpoint.pt"),
            "train_prediction_sha256": sha256_file(prediction_path),
            "runner_sha256": runner_sha, "config_sha256": config_sha,
            "device_uuid": GPU_UUID, "inner_dev_numeric_rows_loaded": False}
        atomic_json(attempt / "attempt_close_receipt.json", close)
        append_jsonl(ledger, {"event": "GPU_ATTEMPT_CLOSED_AFTER_COST_GATE", "at_utc": utc_now(),
            "wsl_pid": process_id, "device_uuid": GPU_UUID, "selected_models": ["transigen"],
            "actual_gpu_hours": elapsed / 3600.0, "reserved_gpu_hours": reservation,
            "unused_reservation_released_gpu_hours": reservation - elapsed / 3600.0,
            "prior_attempts_settled_gpu_hours": accounting["exp_settled_gpu_hours"],
            "close_receipt": "attempt_close_receipt.json"})
        atomic_json(attempt / "attempt_progress.json", {"phase": close["status"],
            "updated_at_utc": utc_now(), "wsl_pid": process_id,
            "completed_epochs": 1, "actual_gpu_hours": elapsed / 3600.0})
        return {"status": close["status"], "close_receipt": close, "train_metrics": train_metrics}

    # Only after the fixed training-only trace and conservative budget gate pass do
    # we run the original 300-epoch raw-sum schedule. No inner-dev values are loaded.
    # Reuse the same seeded loader generator so epoch 2 receives the next ordinary
    # shuffled TRAIN permutation rather than repeating epoch 1's permutation.
    train_metrics_epoch1 = train_metrics
    full_loader = loader
    epoch_seconds_all = [epoch_seconds]
    for epoch in range(1, 300):
        epoch_no = epoch + 1
        budget_check()
        model.train()
        epoch_begin = time.perf_counter()
        loss_sums = np.zeros(7, dtype=np.float64)
        epoch_grads: list[float] = []
        conditions = 0
        lean_batch_trace = attempt / "lean_monitor_batch_trace.jsonl"
        for batch_number, (x1_cpu, x2_cpu, feat_cpu, row_cpu) in enumerate(full_loader, start=1):
            budget_check()
            batch_start = time.perf_counter()
            row_positions = row_cpu.numpy().astype(np.int64, copy=False)
            rng_before = _lean_monitor_rng_state(torch)
            bn_buffers_before = _lean_monitor_bn_buffers(torch, model)
            x1, x2, features = x1_cpu.to(device), x2_cpu.to(device), feat_cpu.to(device)
            model.zero_grad(set_to_none=True)
            prediction, losses = _trainpath_forward_losses(torch, model, x1, x2, features)
            loss_values = [float(value.detach().cpu().item()) for value in losses]
            if len(losses) != 7 or not all(math.isfinite(value) for value in loss_values):
                failure = _lean_monitor_failure_context(torch, model, optimizer, attempt,
                    epoch=epoch_no, batch_index=batch_number, component="training_loss",
                    train_keys=train_keys, row_positions=row_positions, x1=x1, x2=x2,
                    features=features, rng_before_batch=rng_before,
                    bn_before_batch=bn_buffers_before, last_finite_checkpoint=initial_finite_checkpoint,
                    component_detail={"loss_names": list(loss_names), "loss_values": loss_values})
                raise FloatingPointError(f"nonfinite TRAIN loss at epoch {epoch_no} batch {batch_number}: {failure}")
            main_delta = prediction - x1
            if (prediction.shape != x1.shape or not bool(torch.isfinite(prediction).all().item())
                    or not bool(torch.isfinite(main_delta).all().item())):
                failure = _lean_monitor_failure_context(torch, model, optimizer, attempt,
                    epoch=epoch_no, batch_index=batch_number, component="training_main_delta",
                    train_keys=train_keys, row_positions=row_positions, x1=x1, x2=x2,
                    features=features, rng_before_batch=rng_before,
                    bn_before_batch=bn_buffers_before, last_finite_checkpoint=initial_finite_checkpoint,
                    component_detail={"prediction_shape": list(prediction.shape),
                        "expected_shape": list(x1.shape),
                        "prediction_nonfinite_n": int((~torch.isfinite(prediction)).sum().item()),
                        "main_delta_nonfinite_n": int((~torch.isfinite(main_delta)).sum().item())})
                raise FloatingPointError(f"nonfinite TRAIN main delta at epoch {epoch_no} batch {batch_number}: {failure}")
            losses[0].backward()
            grad_finite, grad_norm, bad_gradient, grad_norm_finite = _lean_monitor_gradient_summary(torch, model)
            if not grad_finite:
                failure = _lean_monitor_failure_context(torch, model, optimizer, attempt,
                    epoch=epoch_no, batch_index=batch_number, component="gradient",
                    train_keys=train_keys, row_positions=row_positions, x1=x1, x2=x2,
                    features=features, rng_before_batch=rng_before,
                    bn_before_batch=bn_buffers_before, last_finite_checkpoint=initial_finite_checkpoint,
                    component_detail={"gradient_elements_finite": grad_finite,
                        "gradient_norm_finite": grad_norm_finite,
                        "first_nonfinite_parameter_gradient": bad_gradient,
                        "gradient_norm": safe_number(grad_norm)})
                raise FloatingPointError(f"nonfinite TRAIN gradient at epoch {epoch_no} batch {batch_number}: {failure}")
            epoch_grads.append(grad_norm if grad_norm_finite else math.nan)
            optimizer.step()
            bad_optimizer_state = _first_nonfinite_optimizer_state(torch, optimizer)
            params_finite, bad_parameter = _lean_monitor_finite_named_tensors(
                torch, [(name, parameter) for name, parameter in model.named_parameters()])
            bn_finite, bad_bn = _lean_monitor_finite_named_tensors(
                torch, _lean_monitor_batchnorm_named_tensors(model))
            if bad_optimizer_state is not None or not params_finite or not bn_finite:
                component = ("optimizer_state" if bad_optimizer_state is not None else
                             "parameter" if not params_finite else "batchnorm_state")
                failure = _lean_monitor_failure_context(torch, model, optimizer, attempt,
                    epoch=epoch_no, batch_index=batch_number, component=component,
                    train_keys=train_keys, row_positions=row_positions, x1=x1, x2=x2,
                    features=features, rng_before_batch=rng_before,
                    bn_before_batch=bn_buffers_before, last_finite_checkpoint=initial_finite_checkpoint,
                    replay_forward=False,
                    component_detail={"optimizer_state": bad_optimizer_state,
                        "first_nonfinite_parameter": bad_parameter,
                        "first_nonfinite_batchnorm_buffer": bad_bn})
                raise FloatingPointError(f"nonfinite model/optimizer state at epoch {epoch_no} batch {batch_number}: {failure}")
            loss_sums += np.asarray(loss_values, dtype=np.float64)
            conditions += int(len(x1))
            append_jsonl(lean_batch_trace, {"epoch": epoch_no, "batch_index": batch_number,
                "conditions": int(len(x1)),
                "condition_keys": train_keys.iloc[row_positions].full_condition_key.astype(str).tolist(),
                "losses": dict(zip(loss_names, loss_values)), "loss_finite": True,
                "gradient_norm": safe_number(grad_norm),
                "gradient_elements_finite": grad_finite,
                "gradient_norm_finite": grad_norm_finite, "gradient_finite": grad_finite,
                "parameters_finite": params_finite, "optimizer_finite": bad_optimizer_state is None,
                "batchnorm_finite": bn_finite, "batch_seconds": time.perf_counter() - batch_start,
                "inner_dev_numeric_rows_loaded": False})
            del x1, x2, features, prediction, main_delta, losses
        if conditions != 419 or len(epoch_grads) != 7:
            raise ValueError(f"TRAIN epoch {epoch_no} did not process the fixed 419/7 batches")
        parameters_finite, bad_parameter = _lean_monitor_finite_named_tensors(
            torch, [(name, parameter) for name, parameter in model.named_parameters()])
        bad_optimizer_state = _first_nonfinite_optimizer_state(torch, optimizer)
        batchnorm_finite, bad_batchnorm = _lean_monitor_finite_named_tensors(
            torch, _lean_monitor_batchnorm_named_tensors(model))
        if not parameters_finite or bad_optimizer_state is not None or not batchnorm_finite:
            raise FloatingPointError(
                f"nonfinite epoch {epoch_no} terminal state: parameter={bad_parameter}, "
                f"optimizer={bad_optimizer_state}, batchnorm={bad_batchnorm}")
        bn_epoch_summary, previous_bn = _trainpath_bn_snapshot(
            torch, model, f"epoch{epoch_no}_complete", previous_bn)
        bn_trace.append(bn_epoch_summary)
        # This checkpoint advances only after an entire finite TRAIN epoch.
        epoch_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        last_finite_payload = {"state_dict": epoch_state,
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "model": "upstream TranSiGen", "epoch": epoch_no, "seed": SEED,
            "objective": "raw_sum", "gradient_clipping": False,
            "learning_rate": 1e-3, "weight_decay": 1e-5,
            "fit_scope": "TRAIN_ONLY_NO_INNER_DEV_VALUES",
            "checkpoint_role": "last_finite_completed_train_epoch"}
        _atomic_torch_checkpoint(torch, initial_finite_checkpoint, last_finite_payload)
        atomic_json(attempt / "last_finite_checkpoint_receipt.json", {
            "status": "PASS_LAST_FINITE_COMPLETED_EPOCH_CHECKPOINT",
            "checkpoint": initial_finite_checkpoint.name,
            "checkpoint_sha256": sha256_file(initial_finite_checkpoint),
            "epoch": epoch_no, "optimizer_updates": epoch_no * 7,
            "created_at_utc": utc_now(), "inner_dev_numeric_rows_loaded": False})
        torch.cuda.synchronize()
        eval_begin = time.perf_counter()
        rng_state = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        eval_batch_failures: list[dict[str, Any]] = []
        train_prediction, train_eval_losses = _transigen_predict_and_losses(
            torch, model, values["X_ctl"], values["X"], train_features, 64, device,
            SEED + 2 * epoch_no, allow_nonfinite_auxiliary=True, allow_nonfinite_main=True,
            progress_check=budget_check,
            nonfinite_main_delta_callback=eval_batch_failures.append)
        random.setstate(rng_state[0]); np.random.set_state(rng_state[1])
        torch.set_rng_state(rng_state[2]); torch.cuda.set_rng_state_all(rng_state[3])
        eval_summary = _monitored_main_prediction_summary(train_prediction,
            expected_shape=(419, 978), first_layer=None,
            batch_failures=eval_batch_failures)
        if eval_summary["main_delta_nonfinite_n"] > 0:
            eval_failure = _lean_monitor_eval_failure_replay(torch, model, optimizer, attempt,
                train_keys=train_keys, X_ctl=values["X_ctl"], X=values["X"], features=train_features,
                device=device, seed=SEED + 2 * epoch_no, first_failure=eval_batch_failures[0])
            eval_summary["first_nonfinite_layer"] = eval_failure["first_nonfinite_layer"]
        if eval_summary["main_delta_nonfinite_n"] == 0:
            train_metrics = keyed_metrics(train_keys, train_y, train_prediction, gene_variance)
            train_metrics.update({key: eval_summary[key] for key in (
                "status", "main_delta_nonfinite_n", "main_delta_denominator", "first_nonfinite_layer")})
            fit_gate = _fit_gate(train_metrics, recovery_fit_gate_baselines)
        else:
            train_metrics = eval_summary
            fit_gate = {"status": "UNDEFINED_NONFINITE_MAIN_DELTA",
                "reason": "full 419x978 TRAIN prediction contains nonfinite values; no NMSE or finite-subset metric",
                "main_delta_nonfinite_n": eval_summary["main_delta_nonfinite_n"],
                "main_delta_denominator": eval_summary["main_delta_denominator"],
                "first_nonfinite_layer": eval_summary["first_nonfinite_layer"]}
        elapsed_epoch = time.perf_counter() - epoch_begin
        epoch_seconds_all.append(elapsed_epoch)
        row = {"epoch": epoch_no, "objective": "raw_sum", "gradient_clipping": False,
            "learning_rate": learning_rate, "weight_decay": 1e-5, "batch_size": 64,
            "train_shuffle": True, "train_conditions": conditions,
            "train_batch_sizes": batch_sizes,
            "train_loss_terms_mean_per_condition": {name: safe_number(float(value / conditions))
                for name, value in zip(loss_names, loss_sums)},
            "train_evaluation_auxiliary_losses": train_eval_losses,
            "train_prediction_evaluation": eval_summary,
            "gradient_norm_mean": safe_number(float(np.mean(epoch_grads))),
            "gradient_norm_max": safe_number(float(np.max(epoch_grads))),
            "gradient_norm_finite_n_updates": sum(math.isfinite(value) for value in epoch_grads),
            "gradient_finite": True,
            "parameters_finite": parameters_finite,
            "optimizer_finite": bad_optimizer_state is None,
            "batchnorm_finite": batchnorm_finite,
            "batchnorm_epoch_summary": bn_epoch_summary,
            "parameter_global_l2_norm": _param_global_norm(torch, model),
            "train_metrics": train_metrics, "train_fit_gate": fit_gate,
            "epoch_seconds_including_lean_trace_eval_and_last_finite_save": elapsed_epoch,
            "finite_training_updates": True,
            "train_evaluation_main_delta_finite": eval_summary["main_delta_nonfinite_n"] == 0,
            "inner_dev_numeric_rows_loaded": False}
        append_jsonl(curve_path, row)
        if epoch_no in (100, 300):
            checkpoint_payload = {**last_finite_payload, "epoch": epoch_no,
                "checkpoint_role": f"fixed_epoch_{epoch_no}_train_only_endpoint"}
            checkpoint = attempt / f"TranSiGen_epoch_{epoch_no}.pt"
            _atomic_torch_checkpoint(torch, checkpoint, checkpoint_payload)
            if epoch_no == 300:
                _atomic_torch_checkpoint(torch, attempt / "TranSiGen_final_epoch_300.pt", checkpoint_payload)
            endpoint = attempt / f"TranSiGen_epoch_{epoch_no}_train_keyed_predictions.npz"
            atomic_npz(endpoint,
                train_condition_key=train_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
                train_source_object_id=train_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
                train_full_inchikey=train_keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
                genes=values["genes"].astype(str), train_target_delta=train_y,
                train_delta=train_prediction)
            with np.load(endpoint, allow_pickle=False) as saved:
                if (not np.array_equal(saved["train_condition_key"], train_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_))
                        or not np.array_equal(saved["genes"], values["genes"].astype(str))
                        or not _np_arrays_equal_with_nan(saved["train_delta"], train_prediction)):
                    raise ValueError(f"TRAIN endpoint keyed readback failed at epoch {epoch_no}")
        progress.update({"phase": "TRAIN_ONLY_FULL_SCHEDULE", "updated_at_utc": utc_now(),
            "completed_epochs": epoch_no, "target_epochs": 300,
            "elapsed_gpu_hours_upper_bound": (time.perf_counter() - gpu_start) / 3600.0,
            "last_train_metrics": train_metrics, "inner_dev_numeric_rows_loaded": False})
        atomic_json(attempt / "attempt_progress.json", progress)
        if epoch_no in (1, 10, 100, 300) and eval_summary["main_delta_nonfinite_n"] > 0:
            atomic_json(attempt / f"trainpath_epoch{epoch_no}_diagnostic.json", {
                "status": "FIXED_EPOCH_MAIN_DELTA_UNDEFINED_NONFINITE",
                "epoch": epoch_no, "train_prediction_metrics": None,
                "nonfinite_count": eval_summary["main_delta_nonfinite_n"],
                "denominator": eval_summary["main_delta_denominator"],
                "first_nonfinite_layer": eval_summary["first_nonfinite_layer"],
                "auxiliary_loss_fields": train_eval_losses,
                "prediction_path": f"TranSiGen_epoch_{epoch_no}_train_keyed_predictions.npz"
                    if epoch_no in (100, 300) else None,
                "inner_dev_numeric_rows_loaded": False})
        if epoch_no == 100 and eval_summary["main_delta_nonfinite_n"] > 0:
            raise FloatingPointError(
                "fixed epoch-100 TRAIN evaluation main delta remains nonfinite; closing this recovery run")
        if epoch_no == 100 or epoch_no == 300 or epoch_no % 10 == 0:
            print(json.dumps({"stage": "EXP055_TRAINPATH_RECOVERY_PROGRESS",
                "epoch": epoch_no, "train_nmse": train_metrics.get("object_macro_nmse"),
                "train_main_delta_nonfinite_n": eval_summary["main_delta_nonfinite_n"],
                "train_main_delta_denominator": eval_summary["main_delta_denominator"],
                "first_nonfinite_layer": eval_summary["first_nonfinite_layer"],
                "elapsed_gpu_hours": progress["elapsed_gpu_hours_upper_bound"],
                "inner_dev_numeric_rows_loaded": False}, ensure_ascii=False), flush=True)
        if (time.perf_counter() - gpu_start) >= guard_seconds:
            raise TimeoutError("TranSiGen reached the 1.0 GPUh recovery guard")
    torch.cuda.synchronize()
    frozen_checkpoint = attempt / "TranSiGen_final_epoch_300.pt"
    checkpoint_sha_before_readback = sha256_file(frozen_checkpoint)
    atomic_json(attempt / "checkpoint_freeze_receipt.json", {
        "status": "FROZEN_EPOCH_300_BEFORE_INNER_DEV_NUMERIC_READ",
        "checkpoint": frozen_checkpoint.name, "checkpoint_sha256": checkpoint_sha_before_readback,
        "epoch": 300, "run_id": RUN_ID, "attempt_id": attempt_id,
        "training_scope": "419 TRAIN conditions / 110 objects only",
        "inner_dev_numeric_rows_loaded": False, "model_selection_on_inner_dev": False,
        "pdo_outer_locked_values_opened": False, "created_at_utc": utc_now()})
    checkpoint_payload = torch.load(frozen_checkpoint, map_location=device, weights_only=False)
    checkpoint_state = checkpoint_payload.get("state_dict")
    if (checkpoint_payload.get("epoch") != 300 or not isinstance(checkpoint_state, dict)
            or checkpoint_payload.get("fit_scope") != "TRAIN_ONLY_NO_INNER_DEV_VALUES"):
        raise RuntimeError("frozen epoch-300 checkpoint metadata does not match the registered fit")
    nonfinite_checkpoint = [name for name, value in checkpoint_state.items()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
        and not bool(torch.isfinite(value).all().item())]
    if nonfinite_checkpoint:
        raise FloatingPointError(f"frozen epoch-300 checkpoint contains non-finite state: {nonfinite_checkpoint[:3]}")
    model.load_state_dict(checkpoint_state, strict=True)
    checkpoint_state_readback = {name: torch.equal(value.detach().cpu(), checkpoint_state[name].detach().cpu())
                                 for name, value in model.state_dict().items()}
    if not checkpoint_state_readback or not all(checkpoint_state_readback.values()):
        raise RuntimeError("loaded epoch-300 checkpoint state differs from the saved frozen tensors")
    checkpoint_freeze_receipt = json.loads((attempt / "checkpoint_freeze_receipt.json").read_text(encoding="utf-8"))
    if sha256_file(frozen_checkpoint) != checkpoint_freeze_receipt["checkpoint_sha256"]:
        raise RuntimeError("epoch-300 checkpoint changed after the freeze receipt")

    dev_keys, dev_values, dev_readback = _load_exp055_inner_dev_after_checkpoint_freeze(
        train_keys, values, SOURCE_RUN_DIR)
    dev_features = _full_key_features(dev_keys, np.arange(93, dtype=np.int64), feature_by_train_key)
    dev_y = dev_values["delta"]
    rng_state = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    try:
        train_eval_failures: list[dict[str, Any]] = []
        train_prediction_final, train_aux_losses_final = _transigen_predict_and_losses(
            torch, model, values["X_ctl"], values["X"], train_features, 64, device,
            SEED + 3001, allow_nonfinite_auxiliary=True, allow_nonfinite_main=True,
            progress_check=budget_check,
            nonfinite_main_delta_callback=train_eval_failures.append)
        dev_eval_failures: list[dict[str, Any]] = []
        dev_prediction_final, dev_aux_losses_final = _transigen_predict_and_losses(
            torch, model, dev_values["X_ctl"], dev_values["X"], dev_features, 64, device,
            SEED + 3002, allow_nonfinite_auxiliary=True, allow_nonfinite_main=True,
            progress_check=budget_check,
            nonfinite_main_delta_callback=dev_eval_failures.append)
    finally:
        random.setstate(rng_state[0]); np.random.set_state(rng_state[1])
        torch.set_rng_state(rng_state[2]); torch.cuda.set_rng_state_all(rng_state[3])
    train_eval_layer = None
    if train_eval_failures:
        train_failure_replay = _lean_monitor_eval_failure_replay(torch, model, optimizer, attempt,
            train_keys=train_keys, X_ctl=values["X_ctl"], X=values["X"], features=train_features,
            device=device, seed=SEED + 3001, first_failure=train_eval_failures[0],
            evaluation_role="TRAIN")
        train_eval_layer = train_failure_replay["first_nonfinite_layer"]
    dev_eval_layer = None
    if dev_eval_failures:
        dev_failure_replay = _lean_monitor_eval_failure_replay(torch, model, optimizer, attempt,
            train_keys=dev_keys, X_ctl=dev_values["X_ctl"], X=dev_values["X"], features=dev_features,
            device=device, seed=SEED + 3002, first_failure=dev_eval_failures[0],
            evaluation_role="inner_dev")
        dev_eval_layer = dev_failure_replay["first_nonfinite_layer"]
    train_final_eval_summary = _monitored_main_prediction_summary(
        train_prediction_final, expected_shape=(419, 978),
        first_layer=train_eval_layer, batch_failures=train_eval_failures)
    dev_final_eval_summary = _monitored_main_prediction_summary(
        dev_prediction_final, expected_shape=(93, 978),
        first_layer=dev_eval_layer, batch_failures=dev_eval_failures)
    if train_final_eval_summary["main_delta_nonfinite_n"] == 0:
        train_metrics_final = keyed_metrics(train_keys, train_y, train_prediction_final, gene_variance)
        train_metrics_final.update({key: train_final_eval_summary[key] for key in (
            "status", "main_delta_nonfinite_n", "main_delta_denominator", "first_nonfinite_layer")})
    else:
        train_metrics_final = train_final_eval_summary
    if dev_final_eval_summary["main_delta_nonfinite_n"] == 0:
        dev_metrics_final = keyed_metrics(dev_keys, dev_y, dev_prediction_final, gene_variance)
        dev_metrics_final.update({key: dev_final_eval_summary[key] for key in (
            "status", "main_delta_nonfinite_n", "main_delta_denominator", "first_nonfinite_layer")})
    else:
        dev_metrics_final = dev_final_eval_summary
    baseline_final = _exp055_train_and_dev_baselines(
        train_keys, train_y, dev_keys, dev_y, gene_variance)
    atomic_json(attempt / "baseline_metrics.json", baseline_final)
    baseline_differences: dict[str, Any] = {}
    for baseline_name, by_role in baseline_final["metrics"].items():
        baseline_differences[baseline_name] = {}
        for role, model_metrics, baseline_metric in (
                ("train", train_metrics_final, by_role["train"]),
                ("inner_dev", dev_metrics_final, by_role["inner_dev"])):
            model_nmse = model_metrics["object_macro_nmse"]
            baseline_nmse = baseline_metric["object_macro_nmse"]
            baseline_differences[baseline_name][role] = {
                "model_minus_baseline_object_macro_nmse": safe_number(model_nmse - baseline_nmse)
                    if model_nmse is not None and baseline_nmse is not None else None,
                "relative_nmse_reduction_percent": safe_number(100.0 * (baseline_nmse - model_nmse) / baseline_nmse)
                    if model_nmse is not None and baseline_nmse not in (None, 0.0) else None,
                "model_minus_baseline_amplitude_ratio": safe_number(
                    model_metrics["object_macro_amplitude_ratio"] - baseline_metric["object_macro_amplitude_ratio"])
                    if model_metrics["object_macro_amplitude_ratio"] is not None
                    and baseline_metric["object_macro_amplitude_ratio"] is not None else None}

    final_prediction_path = attempt / "TranSiGen_epoch_300_frozen_keyed_predictions.npz"
    atomic_npz(final_prediction_path,
        train_condition_key=train_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
        inner_dev_condition_key=dev_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
        train_source_object_id=train_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
        inner_dev_source_object_id=dev_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
        train_full_inchikey=train_keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
        inner_dev_full_inchikey=dev_keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
        genes=values["genes"].astype(str), train_target_delta=train_y,
        inner_dev_target_delta=dev_y, train_delta=train_prediction_final,
        inner_dev_delta=dev_prediction_final)
    expected_prediction_arrays = {
        "train_condition_key": train_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
        "inner_dev_condition_key": dev_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
        "train_source_object_id": train_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
        "inner_dev_source_object_id": dev_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
        "train_full_inchikey": train_keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
        "inner_dev_full_inchikey": dev_keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
        "genes": values["genes"].astype(str), "train_target_delta": train_y,
        "inner_dev_target_delta": dev_y, "train_delta": train_prediction_final,
        "inner_dev_delta": dev_prediction_final}
    checkpoint_readback = _verify_exp055_prediction_and_checkpoint_readback(
        torch, attempt, final_prediction_path, expected_prediction_arrays,
        ["TranSiGen_epoch_100.pt", "TranSiGen_epoch_300.pt", "TranSiGen_final_epoch_300.pt"])
    with np.load(final_prediction_path, allow_pickle=False) as saved:
        if train_final_eval_summary["main_delta_nonfinite_n"] == 0:
            train_metric_readback = keyed_metrics(train_keys, saved["train_target_delta"],
                                                    saved["train_delta"], gene_variance)
            train_metric_exact = math.isclose(
                train_metric_readback["object_macro_nmse"], train_metrics_final["object_macro_nmse"],
                rel_tol=0.0, abs_tol=1e-12)
        else:
            train_metric_exact = None
        if dev_final_eval_summary["main_delta_nonfinite_n"] == 0:
            dev_metric_readback = keyed_metrics(dev_keys, saved["inner_dev_target_delta"],
                                                  saved["inner_dev_delta"], gene_variance)
            dev_metric_exact = math.isclose(
                dev_metric_readback["object_macro_nmse"], dev_metrics_final["object_macro_nmse"],
                rel_tol=0.0, abs_tol=1e-12)
        else:
            dev_metric_exact = None
    metric_readback = {
        "train_object_macro_nmse_exact": train_metric_exact,
        "inner_dev_object_macro_nmse_exact": dev_metric_exact,
        "train_metric_status": train_final_eval_summary["status"],
        "inner_dev_metric_status": dev_final_eval_summary["status"]}
    if any(value is False for value in (train_metric_exact, dev_metric_exact)):
        raise RuntimeError("frozen prediction artifact does not reproduce its summary metrics")
    prediction_nonfinite_counts = checkpoint_readback["checks"]["nonfinite_prediction_counts"]
    prediction_readback_status = ("PASS_FULL_KEY_PREDICTION_AND_CHECKPOINT_READBACK"
        if all(count == 0 for count in prediction_nonfinite_counts.values())
        else "PASS_FULL_KEY_READBACK_WITH_NONFINITE_MODEL_OUTPUTS_METRICS_UNDEFINED")
    atomic_json(attempt / "frozen_300_prediction_readback.json", {
        "status": prediction_readback_status,
        "prediction_file": final_prediction_path.name,
        "prediction_sha256": sha256_file(final_prediction_path),
        "checkpoint_readback": checkpoint_readback,
        "metric_readback": metric_readback,
        "inner_dev_readback": dev_readback,
        "train_evaluation": train_final_eval_summary,
        "inner_dev_evaluation": dev_final_eval_summary,
        "inner_dev_used_for_training_or_selection": False,
        "pdo_outer_locked_values_opened": False})
    elapsed = time.perf_counter() - gpu_start
    if elapsed / 3600.0 > reservation:
        raise TimeoutError("TranSiGen 300-epoch training and frozen evaluation exceeded its recovery-package cap")
    final_evaluation_finite = (train_final_eval_summary["main_delta_nonfinite_n"] == 0
        and dev_final_eval_summary["main_delta_nonfinite_n"] == 0)
    result_status = ("COMPLETED_TRANSIGEN_FIXED_300_EPOCH_TRAIN_AND_FROZEN_DEV_EVALUATION"
        if final_evaluation_finite
        else "COMPLETED_TRANSIGEN_FIXED_300_EPOCHS_WITH_UNDEFINED_NONFINITE_EVALUATION")
    final_fit_gate = (_fit_gate(train_metrics_final, baseline_final)
        if train_final_eval_summary["main_delta_nonfinite_n"] == 0
        else {"status": "UNDEFINED_NONFINITE_MAIN_DELTA",
              "reason": "full frozen TRAIN prediction is nonfinite; no NMSE or finite-subset gate"})
    result = {"status": result_status,
        "epochs_completed": 300, "selected_epoch": None,
        "selection": "fixed epoch 300 primary endpoint; epoch 100 retained as a predeclared trajectory only",
        "train_metrics_epoch1": train_metrics_epoch1,
        "train_metrics_epoch300_frozen_checkpoint": train_metrics_final,
        "inner_dev_metrics_epoch300_frozen_checkpoint": dev_metrics_final,
        "train_fit_gate_epoch300": final_fit_gate,
        "train_evaluation_epoch300": train_final_eval_summary,
        "inner_dev_evaluation_epoch300": dev_final_eval_summary,
        "baseline_metrics": baseline_final["metrics"],
        "baseline_differences": baseline_differences,
        "train_evaluation_auxiliary_losses_epoch300": train_aux_losses_final,
        "inner_dev_evaluation_auxiliary_losses_epoch300": dev_aux_losses_final,
        "cost_gate": cost_gate, "mean_train_epoch_seconds": float(np.mean(epoch_seconds_all)),
        "train_conditions": 419, "train_objects": 110,
        "inner_dev_conditions": 93, "inner_dev_objects": 24, "object_overlap": 0,
        "full_inchikey_count": 5, "genes": 978,
        "inner_dev_loaded_only_after_checkpoint_freeze": True,
        "inner_dev_used_for_training_or_selection": False,
        "pdo_outer_locked_values_opened": False,
        "checkpoint_freeze_receipt_sha256": sha256_file(attempt / "checkpoint_freeze_receipt.json"),
        "prediction_readback_receipt_sha256": sha256_file(attempt / "frozen_300_prediction_readback.json"),
        "predictions_path": final_prediction_path.name,
        "predictions_sha256": sha256_file(final_prediction_path),
        "checkpoint_sha256": {name: sha256_file(attempt / name) for name in
            ("TranSiGen_epoch_100.pt", "TranSiGen_epoch_300.pt", "TranSiGen_final_epoch_300.pt")}}
    atomic_json(attempt / "transigen_trainpath_recovery_result.json", result)
    trace_owner = _close_trainpath_trace_owner(attempt, RUN_ID, attempt_id, result_status)
    final_snapshot = _device_snapshot()["selected_gpu"]
    close = {"status": result["status"], "completed_train_epochs": 300,
        "selected_epoch": None, "actual_gpu_hours": elapsed / 3600.0,
        "reserved_gpu_hours": reservation,
        "unused_reservation_released_gpu_hours": reservation - elapsed / 3600.0,
        "cumulative_exp055_gpu_hours": accounting["exp_settled_gpu_hours"] + elapsed / 3600.0,
        "runner_sha256": runner_sha, "config_sha256": config_sha,
        "device_uuid": GPU_UUID, "device_snapshot_at_close": final_snapshot,
        "train_result_sha256": sha256_file(attempt / "transigen_trainpath_recovery_result.json"),
        "predictions_sha256": sha256_file(final_prediction_path),
        "trainpath_trace_owner_sha256": sha256_file(_trainpath_trace_owner_path(attempt)),
        "inner_dev_numeric_rows_loaded": True,
        "inner_dev_loaded_after_checkpoint_freeze": True,
        "inner_dev_used_for_training_or_selection": False}
    atomic_json(attempt / "attempt_close_receipt.json", close)
    append_jsonl(ledger, {"event": "GPU_ATTEMPT_COMPLETED", "at_utc": utc_now(),
        "wsl_pid": process_id, "device_uuid": GPU_UUID, "selected_models": ["transigen"],
        "actual_gpu_hours": elapsed / 3600.0, "reserved_gpu_hours": reservation,
        "unused_reservation_released_gpu_hours": reservation - elapsed / 3600.0,
        "cumulative_conservative_charge_including_prior_attempts_gpu_hours":
            accounting["exp_settled_gpu_hours"] + elapsed / 3600.0,
        "close_receipt": "attempt_close_receipt.json"})
    atomic_json(attempt / "attempt_progress.json", {"phase": "CLOSED_COMPLETED",
        "updated_at_utc": utc_now(), "wsl_pid": process_id, "completed_epochs": 300,
        "actual_gpu_hours": elapsed / 3600.0})
    del model, model_module, optimizer, loader, full_loader, dataset
    torch.cuda.empty_cache()
    return {"status": result["status"], "result": result, "close_receipt": close}


def run_exp055_trainpath_entry_smoke_cpu(cfg: dict[str, Any], config_sha: str,
                                        attempt_id: str) -> dict[str, Any]:
    """Exercise the real original TranSiGen TRAIN update/evaluation/save path on CPU only."""
    started_at = utc_now()
    start_clock = time.perf_counter()

    def enforce_cpu_time_cap(stage: str) -> None:
        elapsed = time.perf_counter() - start_clock
        if elapsed > 600.0:
            raise TimeoutError(f"10-minute CPU entry-smoke cap exceeded during {stage}: {elapsed:.3f}s")

    if (EXPERIMENT_ID != "EXP-055" or RUN_ID not in {
            EXP055_TRAINPATH_ENTRY_SMOKE_RUN_ID, EXP055_TRAINPATH_ENTRY_DIAG_RUN_ID}
            or SOURCE_RUN_ID != DEFAULT_SOURCE_RUN_ID or attempt_id != "fit_attempt_001"):
        raise ValueError("CPU entry smoke is bound to one registered immutable EXP-055 run")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1", None}:
        raise RuntimeError("CPU entry smoke requires CUDA_VISIBLE_DEVICES empty or hidden")
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise FileExistsError(f"CPU entry-smoke run directory must be empty: {RUN_DIR}")
    enforce_cpu_time_cap("initial guards")
    prior_charge, prior_close_hashes = exp055_trainpath_recovery_prior_charge()
    if not math.isclose(prior_charge, 0.003001485, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("CPU entry smoke could not reconcile the frozen recovery-package charge")
    verified_gate = _reuse_root_verified_exp055_train_gate()
    train_keys, values, feature_by_key, data_evidence = _load_exp055_trainpath_train_inputs_reusing_gate(
        verified_gate, SOURCE_RUN_DIR / "response_blind_conditions.tsv", SOURCE_RUN_DIR)
    enforce_cpu_time_cap("TRAIN input loading")
    features = np.stack([feature_by_key[key] for key in
                         train_keys.full_inchikey.astype(str)]).astype(np.float32, copy=False)
    if (features.shape != (419, 2048) or values["X"].shape != (419, 978)
            or values["X_ctl"].shape != (419, 978) or values["delta"].shape != (419, 978)
            or train_keys.source_object_id.nunique() != 110
            or train_keys.full_inchikey.nunique() != 5
            or not np.isfinite(features).all()):
        raise ValueError("CPU entry-smoke inputs differ from the frozen 419-condition / 110-object / 5-key / 978-gene contract")

    attempt = new_attempt_path(RUN_DIR, attempt_id)
    attempt.mkdir(parents=True, exist_ok=False)
    runner_sha = sha256_file(Path(__file__).resolve())
    process_id = os.getpid()
    checkpoint_path = attempt / "TranSiGen_cpu_entry_smoke_epoch_1.pt"
    checkpoint_readback_path = attempt / "entry_smoke_pre_eval_checkpoint_readback.json"
    first_nonfinite_path = attempt / "entry_smoke_first_nonfinite_eval_delta.json"
    checkpoint_readback: dict[str, Any] | None = None
    _open_trainpath_trace_owner(attempt, RUN_ID, attempt_id, process_id, runner_sha)
    batch_trace_path = attempt / "entry_smoke_train_batch_trace.jsonl"
    manifest = {"experiment_id": EXPERIMENT_ID, "run_id": RUN_ID, "attempt_id": attempt_id,
        "phase": "entry-smoke", "status": "CPU_REAL_TRAINPATH_RUNNING", "device": "cpu",
        "started_at_utc": started_at, "wsl_pid": process_id,
        "runner_sha256": runner_sha, "config_sha256": config_sha,
        "source_run_id": SOURCE_RUN_ID, "source_gate": verified_gate,
        "train_input_evidence": data_evidence,
        "model": "upstream original TranSiGen", "model_source_sha256": sha256_file(TRANSIGEN_ROOT / "model.py"),
        "fresh_initialization": True, "pretrained_weights_loaded": False,
        "training_contract": {"train_conditions": 419, "train_objects": 110,
            "full_inchikey_count": 5, "genes": 978, "objective": "raw_sum/no-clip",
            "learning_rate": 1e-3, "weight_decay": 1e-5, "batch_size": 64,
            "shuffle": True, "normal_batchnorm_training": True,
            "expected_batch_sizes": [64, 64, 64, 64, 64, 64, 35],
            "optimizer_updates_required": 7},
        "data_roles": {"train_numeric_rows_loaded": True,
            "inner_dev_numeric_rows_loaded": False,
            "pdo_numeric_sensitivity_or_expression_opened": False,
            "outer_or_locked_numeric_values_opened": False},
        "prior_recovery_charge_gpu_hours": prior_charge,
        "prior_recovery_close_sha256": prior_close_hashes,
        "gpu_hours_charged": 0.0, "attempt_directory": str(attempt.resolve()),
        "train_batch_trace_path": batch_trace_path.name,
        "command": " ".join(sys.argv)}
    atomic_json(attempt / "entry_smoke_manifest.json", manifest)
    atomic_json(attempt / "attempt_progress.json", {"phase": "CPU_INPUTS_READY",
        "updated_at_utc": utc_now(), "wsl_pid": process_id,
        "train_conditions": 419, "inner_dev_numeric_rows_loaded": False})

    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
        device = torch.device("cpu")
        if device.type != "cpu":
            raise RuntimeError("CPU entry smoke resolved a non-CPU execution device")
        cuda_initialized_before_model = bool(torch.cuda.is_initialized())
        model, _model_module = _load_transigen(torch, device, SEED, features_dim=2048,
                                               seed_cuda=False)
        initial_parameters, initial_parameter_state = _trainpath_parameter_snapshot(
            torch, model, attempt, stage="cpu_entry_smoke_fresh_initialization")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        dataset = TensorDataset(torch.as_tensor(values["X_ctl"], dtype=torch.float32),
            torch.as_tensor(values["X"], dtype=torch.float32),
            torch.as_tensor(features, dtype=torch.float32))
        loader_generator = torch.Generator(device="cpu").manual_seed(SEED)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False,
                            num_workers=0, generator=loader_generator)
        observed_batch_sizes = [int(batch[0].shape[0]) for batch in loader]
        loader_generator.manual_seed(SEED)
        if observed_batch_sizes != [64, 64, 64, 64, 64, 64, 35]:
            raise ValueError(f"CPU entry smoke TRAIN loader changed: {observed_batch_sizes}")

        loss_names = ("raw_total_loss", "mse_x1", "mse_x2", "mse_pert",
                      "kld_x1", "kld_x2", "kld_pert")
        batch_rows: list[dict[str, Any]] = []
        seen = 0
        for batch_index, (x1_cpu, x2_cpu, feat_cpu) in enumerate(loader, start=1):
            x1, x2, batch_features = x1_cpu.to(device), x2_cpu.to(device), feat_cpu.to(device)
            model.train()
            model.zero_grad(set_to_none=True)
            prediction, losses = _trainpath_forward_losses(
                torch, model, x1, x2, batch_features)
            loss_values = [float(loss.detach().cpu().item()) for loss in losses]
            if len(losses) != 7 or not all(math.isfinite(value) for value in loss_values):
                atomic_json(attempt / "trainpath_first_nonfinite.json", {
                    "status": "FIRST_NONFINITE_CPU_ENTRY_SMOKE_TRAIN_LOSS",
                    "batch_index": batch_index, "loss_names": loss_names,
                    "loss_values": loss_values, "timestamp_utc": utc_now(),
                    "train_only": True, "inner_dev_numeric_rows_loaded": False})
                raise FloatingPointError(f"nonfinite real TRAIN loss in CPU entry-smoke batch {batch_index}")
            main_delta = prediction - x1
            if not bool(torch.isfinite(main_delta).all().item()):
                atomic_json(attempt / "trainpath_first_nonfinite.json", {
                    "status": "FIRST_NONFINITE_CPU_ENTRY_SMOKE_MAIN_DELTA",
                    "batch_index": batch_index, "timestamp_utc": utc_now(),
                    "train_only": True, "inner_dev_numeric_rows_loaded": False})
                raise FloatingPointError(f"nonfinite original TranSiGen main delta in batch {batch_index}")
            parameters_before_update = {name: parameter.detach().clone()
                                        for name, parameter in model.named_parameters()}
            losses[0].backward()
            gradient_row = _trainpath_gradient_snapshot(torch, model, attempt, batch_index)
            gradient_norm = float(gradient_row["gradient_global_l2"])
            gradient_max_abs = float(gradient_row["gradient_max_abs"])
            if not (math.isfinite(gradient_norm) and math.isfinite(gradient_max_abs)):
                raise FloatingPointError(f"nonfinite real TRAIN gradient statistic in CPU entry-smoke batch {batch_index}")
            optimizer.step()
            post_parameters, _ = _trainpath_parameter_snapshot(torch, model, attempt,
                stage=f"cpu_entry_smoke_batch_{batch_index}_after_update",
                previous=parameters_before_update)
            if not all(bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters()):
                raise FloatingPointError(f"nonfinite parameter after CPU entry-smoke batch {batch_index}")
            batch_update_l2 = float(math.sqrt(sum(
                float(row.get("delta_l2", 0.0)) ** 2 for row in post_parameters["parameters"])))
            if not math.isfinite(batch_update_l2):
                raise FloatingPointError(f"nonfinite real TRAIN parameter update norm in batch {batch_index}")
            batch_row = {"run_id": RUN_ID, "attempt_id": attempt_id,
                "batch_index": batch_index, "conditions": int(len(x1)),
                "losses": dict(zip(loss_names, loss_values)),
                "gradient_global_l2": gradient_norm,
                "gradient_max_abs": gradient_max_abs,
                "gradient_parameter_count": gradient_row["parameter_gradient_count"],
                "parameter_update_l2": batch_update_l2,
                "parameter_max_abs_after_update": max(
                    (float(row["max_abs"]) for row in post_parameters["parameters"]), default=0.0),
                "timestamp_utc": utc_now()}
            append_jsonl(batch_trace_path, batch_row)
            batch_rows.append(batch_row)
            seen += int(len(x1))
            enforce_cpu_time_cap(f"training batch {batch_index}")
            atomic_json(attempt / "attempt_progress.json", {"phase": "CPU_TRAIN_UPDATES",
                "updated_at_utc": utc_now(), "wsl_pid": process_id,
                "completed_optimizer_updates": batch_index, "target_optimizer_updates": 7,
                "completed_train_conditions": seen, "target_train_conditions": 419,
                "elapsed_cpu_seconds": time.perf_counter() - start_clock,
                "inner_dev_numeric_rows_loaded": False})
            print(json.dumps({"stage": "EXP055_CPU_ENTRY_SMOKE_TRAIN_PROGRESS",
                "pid": process_id, "optimizer_update": batch_index, "updates_total": 7,
                "train_conditions_seen": seen, "batch_size": int(len(x1)),
                "elapsed_cpu_seconds": time.perf_counter() - start_clock}, ensure_ascii=False),
                flush=True)
            del x1, x2, batch_features, prediction, main_delta, losses
        if seen != 419 or len(batch_rows) != 7:
            raise ValueError(f"CPU entry smoke must execute 7 updates across 419 TRAIN rows: {seen}/{len(batch_rows)}")

        post_parameters, _ = _trainpath_parameter_snapshot(torch, model, attempt,
            stage="cpu_entry_smoke_epoch1_complete", previous=initial_parameter_state)
        parameter_update_l2 = float(math.sqrt(sum(
            float(row.get("delta_l2", 0.0)) ** 2 for row in post_parameters["parameters"])))
        if not math.isfinite(parameter_update_l2) or parameter_update_l2 <= 0.0:
            raise RuntimeError("seven real optimizer updates did not produce a finite nonzero parameter change")
        enforce_cpu_time_cap("pre-evaluation state capture")

        # Freeze and verify all learned state before the diagnostic evaluation. This makes
        # an evaluation-only failure recoverable and separates it from an update failure.
        batchnorm_buffers = {name: value.detach().cpu().clone()
                             for name, value in model.named_buffers()}
        checkpoint_payload = {"state_dict": {name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()},
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "batchnorm_buffers": batchnorm_buffers,
            "model": "upstream TranSiGen", "epoch": 1, "seed": SEED,
            "device": "cpu", "optimizer_updates": 7, "objective": "raw_sum",
            "gradient_clipping": False, "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "checkpoint_stage": "after_7_train_updates_before_full_train_evaluation",
            "purpose": "CPU numeric-path diagnosis only; no model selection"}
        _atomic_torch_checkpoint(torch, checkpoint_path, checkpoint_payload)
        loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_readback_exact = _torch_tree_exact(
            torch, checkpoint_payload["state_dict"], loaded_checkpoint.get("state_dict"))
        optimizer_readback_exact = _torch_tree_exact(
            torch, checkpoint_payload["optimizer_state_dict"],
            loaded_checkpoint.get("optimizer_state_dict"))
        batchnorm_readback_exact = _torch_tree_exact(
            torch, checkpoint_payload["batchnorm_buffers"],
            loaded_checkpoint.get("batchnorm_buffers"))
        checkpoint_metadata_exact = (
            loaded_checkpoint.get("epoch") == 1
            and loaded_checkpoint.get("optimizer_updates") == 7
            and loaded_checkpoint.get("checkpoint_stage")
                == "after_7_train_updates_before_full_train_evaluation")
        if not (state_readback_exact and optimizer_readback_exact
                and batchnorm_readback_exact and checkpoint_metadata_exact):
            raise ValueError("pre-evaluation checkpoint did not exactly read back model/optimizer/BN state")
        checkpoint_readback = {"status": "PASS_EXACT_PRE_EVALUATION_READBACK",
            "run_id": RUN_ID, "attempt_id": attempt_id, "wsl_pid": process_id,
            "checkpoint_path": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "state_dict_exact": state_readback_exact,
            "optimizer_state_dict_exact": optimizer_readback_exact,
            "batchnorm_buffers_exact": batchnorm_readback_exact,
            "metadata_exact": checkpoint_metadata_exact,
            "batchnorm_buffer_names": sorted(batchnorm_buffers),
            "checkpoint_stage": checkpoint_payload["checkpoint_stage"],
            "inner_dev_numeric_rows_loaded": False,
            "pdo_outer_locked_numeric_values_opened": False,
            "verified_at_utc": utc_now()}
        atomic_json(checkpoint_readback_path, checkpoint_readback)

        gene_variance = np.var(values["delta"].astype(np.float64), axis=0, ddof=0)
        model.eval()

        def persist_first_nonfinite_delta(diagnostic: dict[str, Any]) -> None:
            row_details = []
            for row_index, gene_index in diagnostic["nonfinite_row_gene_positions_first_32"]:
                global_row = int(diagnostic["batch_start"]) + int(row_index)
                key_row = train_keys.iloc[global_row]
                row_details.append({"evaluation_row_index": global_row,
                    "batch_local_row_index": int(row_index),
                    "gene_index": int(gene_index),
                    "condition_key": str(key_row["full_condition_key"]),
                    "source_object_id": str(key_row["source_object_id"]),
                    "full_inchikey": str(key_row["full_inchikey"])})
            atomic_json(first_nonfinite_path, {**diagnostic,
                "status": "FIRST_NONFINITE_TRAIN_MAIN_DELTA_CAPTURED",
                "run_id": RUN_ID, "attempt_id": attempt_id, "wsl_pid": process_id,
                "evaluation_batch_index_1_based": int(diagnostic["batch_start"]) // 64 + 1,
                "nonfinite_key_gene_details_first_32": row_details,
                "checkpoint_path": checkpoint_path.name,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "checkpoint_readback_path": checkpoint_readback_path.name,
                "checkpoint_readback_sha256": sha256_file(checkpoint_readback_path),
                "checkpoint_readback_status": checkpoint_readback["status"],
                "training_batch_trace_path": batch_trace_path.name,
                "training_batch_trace_sha256": sha256_file(batch_trace_path),
                "runner_sha256": runner_sha,
                "evaluation_role": "TRAIN_ONLY",
                "inner_dev_numeric_rows_loaded": False,
                "pdo_outer_locked_numeric_values_opened": False})

        def evaluation_progress_check() -> None:
            enforce_cpu_time_cap("full TRAIN evaluation")

        prediction, eval_losses = _transigen_predict_and_losses(torch, model,
            values["X_ctl"], values["X"], features, 64, device, SEED + 1,
            seed_cuda=False, allow_nonfinite_auxiliary=True,
            nonfinite_main_delta_callback=persist_first_nonfinite_delta,
            progress_check=evaluation_progress_check)
        enforce_cpu_time_cap("full TRAIN evaluation completion")
        if prediction.shape != (419, 978) or not np.isfinite(prediction).all():
            raise FloatingPointError("CPU entry-smoke TRAIN evaluation prediction failed finite 419x978 contract")
        train_metrics = keyed_metrics(train_keys, values["delta"], prediction, gene_variance)

        prediction_path = attempt / "TranSiGen_cpu_entry_smoke_train_keyed_predictions.npz"
        expected_arrays = {"train_condition_key": train_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
            "train_source_object_id": train_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
            "train_full_inchikey": train_keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
            "genes": values["genes"].astype(str), "train_target_delta": values["delta"],
            "train_delta": prediction}
        atomic_npz(prediction_path, **expected_arrays)
        with np.load(prediction_path, allow_pickle=False) as saved:
            if any(name not in saved.files or not np.array_equal(saved[name], expected)
                   for name, expected in expected_arrays.items()):
                raise ValueError("CPU entry-smoke keyed prediction archive failed exact identity/value readback")
        enforce_cpu_time_cap("final keyed prediction save and readback")

        cuda_initialized_after = bool(torch.cuda.is_initialized())
        if cuda_initialized_before_model or cuda_initialized_after:
            raise RuntimeError("CPU entry smoke unexpectedly initialized CUDA")
        elapsed_cpu_seconds = time.perf_counter() - start_clock
        result = {"status": "PASS_REAL_ORIGINAL_TRANSIGEN_CPU_ENTRY_SMOKE",
            "scientific_result": False, "device": "cpu", "wsl_pid": process_id,
            "started_at_utc": started_at, "closed_at_utc": utc_now(),
            "elapsed_cpu_seconds": elapsed_cpu_seconds, "gpu_hours_charged": 0.0,
            "model_source_sha256": sha256_file(TRANSIGEN_ROOT / "model.py"),
            "fresh_initialization": True, "pretrained_weights_loaded": False,
            "initial_parameter_snapshot": initial_parameters,
            "train_conditions": seen, "train_objects": 110, "full_inchikey_count": 5,
            "genes": 978, "batch_sizes": observed_batch_sizes,
            "real_optimizer_updates": len(batch_rows), "batch_training_trace": batch_rows,
            "parameter_delta_after_updates": post_parameters,
            "parameter_update_l2": parameter_update_l2,
            "train_eval_metrics": train_metrics, "train_eval_auxiliary_losses": eval_losses,
            "checkpoint_readback_path": checkpoint_readback_path.name,
            "checkpoint_readback_sha256": sha256_file(checkpoint_readback_path),
            "checkpoint_stage": checkpoint_readback["checkpoint_stage"],
            "checkpoint_path": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_readback": "PASS_EXACT_STATE_DICT_AND_EPOCH_METADATA",
            "keyed_prediction_path": prediction_path.name,
            "keyed_prediction_sha256": sha256_file(prediction_path),
            "keyed_prediction_readback": "PASS_EXACT_ALL_KEYS_TARGETS_PREDICTIONS_AND_978_GENE_AXIS",
            "inner_dev_numeric_rows_loaded": False,
            "pdo_outer_locked_numeric_values_opened": False,
            "cuda_initialized_before_or_after": False,
            "interpretation": "real executable-path verification only; no model selection or scientific performance claim"}
        atomic_json(attempt / "entry_smoke_result.json", result)
        owner = _close_trainpath_trace_owner(attempt, RUN_ID, attempt_id,
                                             "PASS_REAL_ORIGINAL_TRANSIGEN_CPU_ENTRY_SMOKE")
        close = {"status": result["status"], "run_id": RUN_ID, "attempt_id": attempt_id,
            "wsl_pid": process_id, "device": "cpu", "gpu_hours_charged": 0.0,
            "elapsed_cpu_seconds": elapsed_cpu_seconds,
            "runner_sha256": runner_sha,
            "result_sha256": sha256_file(attempt / "entry_smoke_result.json"),
            "checkpoint_sha256": result["checkpoint_sha256"],
            "prediction_sha256": result["keyed_prediction_sha256"],
            "owner_manifest_sha256": sha256_file(_trainpath_trace_owner_path(attempt)),
            "inner_dev_numeric_rows_loaded": False,
            "pdo_outer_locked_numeric_values_opened": False}
        atomic_json(attempt / "entry_smoke_close_receipt.json", close)
        atomic_json(attempt / "attempt_progress.json", {"phase": "CPU_ENTRY_SMOKE_CLOSED",
            "updated_at_utc": utc_now(), "wsl_pid": process_id,
            "completed_optimizer_updates": 7, "completed_train_conditions": 419,
            "elapsed_cpu_seconds": elapsed_cpu_seconds, "gpu_hours_charged": 0.0})
        print(json.dumps({"stage": result["status"], "run_id": RUN_ID,
            "attempt_id": attempt_id, "pid": process_id,
            "elapsed_cpu_seconds": elapsed_cpu_seconds,
            "checkpoint_sha256": result["checkpoint_sha256"],
            "prediction_sha256": result["keyed_prediction_sha256"],
            "entry_smoke_result_path": str(attempt / "entry_smoke_result.json")},
            ensure_ascii=False), flush=True)
        return {"result": result, "close_receipt": close}
    except BaseException as error:
        elapsed_cpu_seconds = time.perf_counter() - start_clock
        owner_path = _trainpath_trace_owner_path(attempt)
        if owner_path.is_file():
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            if owner.get("status") == "OPEN":
                owner.update({"status": "CLOSED", "close_status": "FAILED_CLOSED",
                    "exception_type": type(error).__name__, "exception_message": str(error),
                    "closed_at_utc": utc_now()})
                atomic_json(owner_path, owner)
        failure = {"status": "FAILED_CLOSED_CPU_ENTRY_SMOKE",
            "run_id": RUN_ID, "attempt_id": attempt_id, "wsl_pid": process_id,
            "device": "cpu", "started_at_utc": started_at,
            "closed_at_utc": utc_now(), "elapsed_cpu_seconds": elapsed_cpu_seconds,
            "gpu_hours_charged": 0.0, "runner_sha256": runner_sha,
            "exception_type": type(error).__name__, "exception_message": str(error),
            "owner_manifest_sha256": sha256_file(owner_path) if owner_path.is_file() else None,
            "checkpoint_path": (checkpoint_path.name if checkpoint_path.is_file() else None),
            "checkpoint_sha256": (sha256_file(checkpoint_path)
                                  if checkpoint_path.is_file() else None),
            "checkpoint_readback_path": (checkpoint_readback_path.name
                                          if checkpoint_readback_path.is_file() else None),
            "checkpoint_readback_sha256": (sha256_file(checkpoint_readback_path)
                                           if checkpoint_readback_path.is_file() else None),
            "checkpoint_readback_status": (checkpoint_readback.get("status")
                                           if checkpoint_readback else None),
            "train_batch_trace_path": (batch_trace_path.name if batch_trace_path.is_file() else None),
            "train_batch_trace_sha256": (sha256_file(batch_trace_path)
                                         if batch_trace_path.is_file() else None),
            "first_nonfinite_delta_path": (first_nonfinite_path.name
                                           if first_nonfinite_path.is_file() else None),
            "first_nonfinite_delta_sha256": (sha256_file(first_nonfinite_path)
                                             if first_nonfinite_path.is_file() else None),
            "inner_dev_numeric_rows_loaded": False,
            "pdo_outer_locked_numeric_values_opened": False}
        atomic_json(attempt / "entry_smoke_failure_receipt.json", failure)
        print(json.dumps({"stage": failure["status"], "attempt": str(attempt),
            "elapsed_cpu_seconds": elapsed_cpu_seconds,
            "gpu_hours_charged": 0.0,
            "failure": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
        raise


def _batch_replay_tensor_stats(torch: Any, tensor: Any) -> dict[str, Any]:
    detached = tensor.detach()
    finite_mask = torch.isfinite(detached)
    finite_values = detached[finite_mask]
    bad_positions = torch.nonzero(~finite_mask, as_tuple=False)[:16].detach().cpu().tolist()
    return {"shape": list(detached.shape), "dtype": str(detached.dtype),
        "numel": int(detached.numel()), "finite_n": int(finite_mask.sum().item()),
        "nonfinite_n": int((~finite_mask).sum().item()),
        "finite_max_abs": (float(finite_values.abs().max().cpu().item())
                           if finite_values.numel() else None),
        "nonfinite_positions_first_16": bad_positions}


def run_exp055_trainpath_batch_replay_cpu(cfg: dict[str, Any], config_sha: str,
                                          attempt_id: str) -> dict[str, Any]:
    """Replay the saved diag-v2 checkpoint on its first failing 64-row TRAIN evaluation batch."""
    started_at = utc_now()
    start_clock = time.perf_counter()
    run_id = EXP055_TRAINPATH_BATCH_REPLAY_RUN_ID
    checkpoint_path = (ROOT / "artifacts/experiments/EXP-055/"
        "20261007_exp055_transigen_entry_smoke_diag_v2/fit_attempt_001/"
        "TranSiGen_cpu_entry_smoke_epoch_1.pt")
    expected_checkpoint_sha = EXP055_DIAG_V2_CHECKPOINT_SHA256

    def enforce_cpu_cap(stage: str) -> None:
        elapsed = time.perf_counter() - start_clock
        if elapsed > 600.0:
            raise TimeoutError(f"10-minute batch replay CPU cap exceeded at {stage}: {elapsed:.3f}s")

    if (EXPERIMENT_ID != "EXP-055" or RUN_ID != run_id
            or SOURCE_RUN_ID != DEFAULT_SOURCE_RUN_ID or attempt_id != "fit_attempt_001"):
        raise ValueError("TranSiGen batch replay is bound to one new immutable EXP-055 run")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1", None}:
        raise RuntimeError("read-only batch replay requires CUDA_VISIBLE_DEVICES empty or hidden")
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise FileExistsError(f"batch replay run directory must be empty: {RUN_DIR}")
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != expected_checkpoint_sha:
        raise RuntimeError("the immutable diag-v2 checkpoint is missing or its pinned SHA changed")
    source_smoke_dir = checkpoint_path.parent
    source_readback_path = source_smoke_dir / "entry_smoke_pre_eval_checkpoint_readback.json"
    source_failure_path = source_smoke_dir / "entry_smoke_failure_receipt.json"
    source_delta_path = source_smoke_dir / "entry_smoke_first_nonfinite_eval_delta.json"
    for required in (source_readback_path, source_failure_path, source_delta_path):
        if not required.is_file():
            raise FileNotFoundError(f"required immutable diag-v2 evidence is missing: {required}")
    source_readback = json.loads(source_readback_path.read_text(encoding="utf-8"))
    source_failure = json.loads(source_failure_path.read_text(encoding="utf-8"))
    source_delta = json.loads(source_delta_path.read_text(encoding="utf-8"))
    if (source_readback.get("status") != "PASS_EXACT_PRE_EVALUATION_READBACK"
            or source_readback.get("checkpoint_sha256") != expected_checkpoint_sha
            or source_failure.get("run_id") != EXP055_TRAINPATH_ENTRY_DIAG_RUN_ID
            or source_delta.get("batch_start") != 128 or source_delta.get("batch_stop") != 192
            or source_delta.get("evaluation_split") != "TRAIN_ONLY"):
        raise RuntimeError("immutable diag-v2 failure/checkpoint metadata no longer describes TRAIN[128:192]")
    enforce_cpu_cap("checkpoint/evidence guards")

    prior_charge, prior_close_hashes = exp055_trainpath_recovery_prior_charge()
    verified_gate = _reuse_root_verified_exp055_train_gate()
    train_keys, values, feature_by_key, input_evidence = _load_exp055_trainpath_train_inputs_reusing_gate(
        verified_gate, SOURCE_RUN_DIR / "response_blind_conditions.tsv", SOURCE_RUN_DIR)
    enforce_cpu_cap("frozen TRAIN input load")
    if (values["X_ctl"].shape != (419, 978) or values["X"].shape != (419, 978)
            or values["delta"].shape != (419, 978) or len(train_keys) != 419
            or train_keys.source_object_id.nunique() != 110
            or train_keys.full_inchikey.nunique() != 5):
        raise ValueError("batch replay inputs changed from frozen 419 TRAIN / 110 objects / 5 keys / 978 genes")
    start, stop = 128, 192
    replay_keys = train_keys.iloc[start:stop].reset_index(drop=True)
    if len(replay_keys) != 64:
        raise ValueError("batch replay must bind exactly the failed TRAIN evaluation slice [128:192]")
    feature_matrix = np.stack([feature_by_key[key] for key in
                               train_keys.full_inchikey.astype(str)]).astype(np.float32, copy=False)
    if feature_matrix.shape != (419, 2048) or not np.isfinite(feature_matrix).all():
        raise ValueError("TRAIN feature matrix changed from the frozen 419 x 2048 finite contract")
    enforce_cpu_cap("TRAIN feature assembly")

    import torch

    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU batch replay must not initialize CUDA")
    enforce_cpu_cap("CPU Torch initialization")

    attempt = new_attempt_path(RUN_DIR, attempt_id)
    attempt.mkdir(parents=True, exist_ok=False)
    trace_path = attempt / "TranSiGen_trainpath_layer_trace.jsonl"
    runner_sha = sha256_file(Path(__file__).resolve())
    process_id = os.getpid()
    _open_trainpath_trace_owner(attempt, run_id, attempt_id, process_id, runner_sha)
    trace_path.touch()
    checkpoint_sha_before = sha256_file(checkpoint_path)
    manifest = {"experiment_id": EXPERIMENT_ID, "run_id": run_id,
        "attempt_id": attempt_id, "phase": "read_only_batch_layer_replay",
        "status": "CPU_BATCH_REPLAY_RUNNING", "device": "cpu",
        "wsl_pid": process_id, "started_at_utc": started_at,
        "runner_sha256": runner_sha, "config_sha256": config_sha,
        "source_run_id": SOURCE_RUN_ID, "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": checkpoint_sha_before,
        "source_checkpoint_readback_sha256": sha256_file(source_readback_path),
        "source_failure_receipt_sha256": sha256_file(source_failure_path),
        "source_first_nonfinite_delta_sha256": sha256_file(source_delta_path),
        "replay_contract": {"train_row_start_inclusive": start,
            "train_row_stop_exclusive": stop, "train_conditions": 64,
            "gene_count": 978, "features_per_key": 2048,
            "modes": ["eval", "train_forward_only"],
            "optimizer_updates": 0, "checkpoint_write": False,
            "train_mode_runs_on_deep_copy": True,
            "batchnorm_buffers_restored_and_exactly_checked": True,
            "python_numpy_torch_rng_restored": True},
        "train_keys": replay_keys[["full_condition_key", "source_object_id",
                                    "full_inchikey"]].astype(str).to_dict(orient="records"),
        "input_evidence": input_evidence,
        "prior_recovery_charge_gpu_hours": prior_charge,
        "prior_recovery_closures": prior_close_hashes,
        "data_roles": {"inner_dev_numeric_rows_loaded": False,
            "pdo_numeric_sensitivity_or_expression_opened": False,
            "outer_or_locked_numeric_values_opened": False},
        "gpu_hours_charged": 0.0, "command": " ".join(sys.argv)}
    atomic_json(attempt / "batch_replay_manifest.json", manifest)
    atomic_json(attempt / "attempt_progress.json", {"phase": "CPU_BATCH_REPLAY_STARTED",
        "updated_at_utc": utc_now(), "wsl_pid": process_id,
        "inner_dev_numeric_rows_loaded": False})

    original_rng = {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state()}
    mode_results: dict[str, dict[str, Any]] = {}
    first_nonfinite_by_mode: dict[str, dict[str, Any] | None] = {"eval": None, "train_forward_only": None}
    trace_counts = {"eval": 0, "train_forward_only": 0}
    try:
        model, _module = _load_transigen(torch, torch.device("cpu"), SEED,
                                         features_dim=2048, seed_cuda=False)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("epoch") != 1 or checkpoint.get("optimizer_updates") != 7:
            raise ValueError("diag-v2 checkpoint metadata differs from the seven-update frozen source")
        checkpoint_state = checkpoint.get("state_dict")
        if not isinstance(checkpoint_state, dict):
            raise ValueError("diag-v2 checkpoint lacks a state_dict")
        model.load_state_dict(checkpoint_state, strict=True)
        checkpoint_state_cpu = {name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()}
        if not _torch_tree_exact(torch, checkpoint_state_cpu, checkpoint_state):
            raise ValueError("model state loaded from the diag-v2 checkpoint failed exact tensor readback")
        x1 = torch.as_tensor(values["X_ctl"][start:stop], dtype=torch.float32)
        x2 = torch.as_tensor(values["X"][start:stop], dtype=torch.float32)
        features = torch.as_tensor(feature_matrix[start:stop], dtype=torch.float32)

        def replay_mode(mode_name: str, training: bool) -> None:
            enforce_cpu_cap(f"{mode_name} replay setup")
            replay_model = copy.deepcopy(model)
            replay_model.train(training)
            bn_before = {name: value.detach().clone()
                         for name, value in replay_model.named_buffers()}
            event_index = 0
            first_event: dict[str, Any] | None = None

            def observe(module_name: str, io_role: str, tree: Any) -> None:
                nonlocal event_index, first_event
                for tensor_name, tensor in _trainpath_tensor_leaves(tree):
                    if event_index % 128 == 0:
                        enforce_cpu_cap(f"{mode_name} layer replay trace")
                    stats = _batch_replay_tensor_stats(torch, tensor)
                    row = {"run_id": run_id, "attempt_id": attempt_id,
                        "scope": mode_name, "event_index": event_index,
                        "module": module_name, "io_role": io_role,
                        "tensor_name": tensor_name, "evaluation_batch_start": start,
                        "evaluation_batch_stop": stop, **stats,
                        "timestamp_utc": utc_now(), "train_only": True,
                        "inner_dev_numeric_rows_loaded": False,
                        "pdo_outer_locked_numeric_values_opened": False}
                    append_jsonl(trace_path, row)
                    trace_counts[mode_name] += 1
                    if stats["nonfinite_n"] and first_event is None:
                        first_event = {key: value for key, value in row.items()
                                       if key not in {"run_id", "attempt_id"}}
                    event_index += 1

            hooks = []
            for name, submodule in replay_model.named_modules():
                module_name = name or "<root>"
                hooks.append(submodule.register_forward_pre_hook(
                    lambda _m, inputs, n=module_name: observe(n, "input", inputs)))
                hooks.append(submodule.register_forward_hook(
                    lambda _m, _i, output, n=module_name: observe(n, "output", output)))
            mode_rng = {"python": random.getstate(), "numpy": np.random.get_state(),
                        "torch_cpu": torch.get_rng_state()}
            try:
                random.seed(SEED + 1)
                np.random.seed((SEED + 1) % (2**32))
                torch.random.default_generator.manual_seed(SEED + 1)
                with torch.no_grad():
                    x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, _ = replay_model(x1, features)
                    z2, mu2, logvar2 = replay_model.encode_x2(x2)
                    x2_rec = replay_model.decode_x2(z2)
                    losses = replay_model.loss(x1, x1_rec, mu1, logvar1,
                        x2, x2_rec, mu2, logvar2, x2_pred, mu_pred, logvar_pred)
                    main_delta = x2_pred - x1
                    observe("<TranSiGen.main_delta>", "output", main_delta)
                    loss_stats = {name: _batch_replay_tensor_stats(torch, loss)
                        for name, loss in zip(("raw_total_loss", "mse_x1", "mse_x2",
                            "mse_pert", "kld_x1", "kld_x2", "kld_pert"), losses)}
                    bn_after_forward = {name: value.detach().clone()
                                        for name, value in replay_model.named_buffers()}
                    changed_bn = [name for name in bn_before
                                  if not torch.equal(bn_before[name], bn_after_forward[name])]
                    if training:
                        with torch.no_grad():
                            for name, value in replay_model.named_buffers():
                                value.copy_(bn_before[name])
                    bn_restored = all(torch.equal(bn_before[name], value)
                                      for name, value in replay_model.named_buffers())
                    final_state = {name: value.detach().cpu().clone()
                                   for name, value in replay_model.state_dict().items()}
                    state_exact_after_replay = _torch_tree_exact(
                        torch, checkpoint_state_cpu, final_state)
                    mode_results[mode_name] = {
                        "mode": "train" if training else "eval",
                        "training_forward_only": training,
                        "optimizer_updates": 0,
                        "main_delta": _batch_replay_tensor_stats(torch, main_delta),
                        "main_delta_nonfinite": bool(not torch.isfinite(main_delta).all().item()),
                        "losses": loss_stats,
                        "first_nonfinite_layer_event": first_event,
                        "batchnorm_buffers_changed_during_forward": changed_bn,
                        "batchnorm_buffers_restored_exactly": bn_restored,
                        "complete_model_state_matches_source_checkpoint_after_replay": state_exact_after_replay,
                        "layer_tensor_event_count": trace_counts[mode_name]}
                    if first_event is not None:
                        first_nonfinite_by_mode[mode_name] = first_event
                    if not (bn_restored and state_exact_after_replay):
                        raise RuntimeError(f"{mode_name} batch replay changed model/BN state after restoration")
            finally:
                for hook in hooks:
                    hook.remove()
                random.setstate(mode_rng["python"])
                np.random.set_state(mode_rng["numpy"])
                torch.set_rng_state(mode_rng["torch_cpu"])

        replay_mode("eval", training=False)
        replay_mode("train_forward_only", training=True)
        enforce_cpu_cap("both mode replays")
        random.setstate(original_rng["python"])
        np.random.set_state(original_rng["numpy"])
        torch.set_rng_state(original_rng["torch_cpu"])
        if sha256_file(checkpoint_path) != checkpoint_sha_before:
            raise RuntimeError("read-only replay changed the source diag-v2 checkpoint bytes")
        result = {"status": "COMPLETED_READ_ONLY_CPU_BATCH_REPLAY",
            "scientific_result": False, "run_id": run_id,
            "attempt_id": attempt_id, "wsl_pid": process_id,
            "device": "cpu", "started_at_utc": started_at,
            "closed_at_utc": utc_now(), "elapsed_cpu_seconds": time.perf_counter()-start_clock,
            "gpu_hours_charged": 0.0, "source_checkpoint_sha256": checkpoint_sha_before,
            "source_checkpoint_unchanged": True,
            "train_batch": {"start": start, "stop": stop, "conditions": 64,
                "train_source_objects": int(replay_keys.source_object_id.nunique()),
                "full_inchikeys": sorted(replay_keys.full_inchikey.astype(str).unique().tolist()),
                "condition_keys_sha256": sha256_text("\n".join(
                    replay_keys.full_condition_key.astype(str).tolist()))},
            "mode_results": mode_results,
            "first_nonfinite_by_mode": first_nonfinite_by_mode,
            "layer_trace_path": trace_path.name,
            "layer_trace_sha256": sha256_file(trace_path),
            "train_layer_tensor_events": trace_counts["train_forward_only"],
            "eval_layer_tensor_events": trace_counts["eval"],
            "inner_dev_numeric_rows_loaded": False,
            "pdo_outer_locked_numeric_values_opened": False,
            "interpretation": "replay of saved seven-update checkpoint on the same TRAIN eval batch; no model update or score selection"}
        atomic_json(attempt / "batch_replay_result.json", result)
        owner = _close_trainpath_trace_owner(attempt, run_id, attempt_id,
            "COMPLETED_READ_ONLY_CPU_BATCH_REPLAY")
        close = {"status": result["status"], "run_id": run_id,
            "attempt_id": attempt_id, "wsl_pid": process_id,
            "device": "cpu", "gpu_hours_charged": 0.0,
            "elapsed_cpu_seconds": result["elapsed_cpu_seconds"],
            "runner_sha256": runner_sha,
            "result_sha256": sha256_file(attempt / "batch_replay_result.json"),
            "layer_trace_sha256": sha256_file(trace_path),
            "owner_manifest_sha256": sha256_file(_trainpath_trace_owner_path(attempt)),
            "closed_at_utc": utc_now()}
        atomic_json(attempt / "batch_replay_close_receipt.json", close)
        atomic_json(attempt / "attempt_progress.json", {"phase": "CPU_BATCH_REPLAY_CLOSED",
            "updated_at_utc": utc_now(), "wsl_pid": process_id,
            "elapsed_cpu_seconds": result["elapsed_cpu_seconds"], "gpu_hours_charged": 0.0})
        return {"result": result, "close_receipt": close}
    except BaseException as error:
        random.setstate(original_rng["python"])
        np.random.set_state(original_rng["numpy"])
        torch.set_rng_state(original_rng["torch_cpu"])
        owner_path = _trainpath_trace_owner_path(attempt)
        if owner_path.is_file():
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            if owner.get("status") == "OPEN":
                owner.update({"status": "CLOSED", "close_status": "FAILED_CLOSED_CPU_BATCH_REPLAY",
                    "exception_type": type(error).__name__, "exception_message": str(error),
                    "closed_at_utc": utc_now(),
                    "trace_sha256": sha256_file(trace_path) if trace_path.is_file() else None})
                atomic_json(owner_path, owner)
        failure = {"status": "FAILED_CLOSED_CPU_BATCH_REPLAY", "run_id": run_id,
            "attempt_id": attempt_id, "wsl_pid": process_id, "device": "cpu",
            "started_at_utc": started_at, "closed_at_utc": utc_now(),
            "elapsed_cpu_seconds": time.perf_counter()-start_clock,
            "gpu_hours_charged": 0.0, "runner_sha256": runner_sha,
            "exception_type": type(error).__name__, "exception_message": str(error),
            "layer_trace_path": trace_path.name if trace_path.exists() else None,
            "layer_trace_sha256": sha256_file(trace_path) if trace_path.is_file() else None,
            "owner_manifest_sha256": sha256_file(owner_path) if owner_path.is_file() else None,
            "inner_dev_numeric_rows_loaded": False,
            "pdo_outer_locked_numeric_values_opened": False}
        atomic_json(attempt / "batch_replay_failure_receipt.json", failure)
        raise


def _latent_replay_json_vector(torch: Any, tensor: Any) -> list[float | str]:
    values = tensor.detach().cpu().reshape(-1).tolist()
    result: list[float | str] = []
    for value in values:
        number = float(value)
        if math.isnan(number):
            result.append("NaN")
        elif math.isinf(number):
            result.append("Infinity" if number > 0 else "-Infinity")
        else:
            result.append(number)
    return result


def run_exp055_trainpath_latent_replay_cpu(cfg: dict[str, Any], config_sha: str,
                                           attempt_id: str) -> dict[str, Any]:
    """Capture the original forward's latent standard deviation/sample on the fixed failing TRAIN row."""
    started_at = utc_now()
    start_clock = time.perf_counter()
    run_id = EXP055_TRAINPATH_LATENT_REPLAY_RUN_ID
    checkpoint_path = (ROOT / "artifacts/experiments/EXP-055/"
        "20261007_exp055_transigen_entry_smoke_diag_v2/fit_attempt_001/"
        "TranSiGen_cpu_entry_smoke_epoch_1.pt")
    expected_checkpoint_sha = EXP055_DIAG_V2_CHECKPOINT_SHA256
    source_replay_attempt = (ROOT / "artifacts/experiments/EXP-055/"
        "20261007_exp055_transigen_batch_eval_replay_v1/fit_attempt_001")
    source_replay_path = source_replay_attempt / "batch_replay_result.json"
    source_trace_path = source_replay_attempt / "TranSiGen_trainpath_layer_trace.jsonl"

    def enforce_cpu_cap(stage: str) -> None:
        elapsed = time.perf_counter() - start_clock
        if elapsed > 600.0:
            raise TimeoutError(f"10-minute latent replay CPU cap exceeded at {stage}: {elapsed:.3f}s")

    if (EXPERIMENT_ID != "EXP-055" or RUN_ID != run_id
            or SOURCE_RUN_ID != DEFAULT_SOURCE_RUN_ID or attempt_id != "fit_attempt_001"):
        raise ValueError("TranSiGen latent replay is bound to one new immutable EXP-055 run")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1", None}:
        raise RuntimeError("read-only latent replay requires CUDA_VISIBLE_DEVICES empty or hidden")
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise FileExistsError(f"latent replay run directory must be empty: {RUN_DIR}")
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != expected_checkpoint_sha:
        raise RuntimeError("the immutable diag-v2 checkpoint is missing or its pinned SHA changed")
    if not source_replay_path.is_file() or not source_trace_path.is_file():
        raise FileNotFoundError("the already-closed same-batch replay evidence is missing")
    source_replay = json.loads(source_replay_path.read_text(encoding="utf-8"))
    source_close_path = source_replay_attempt / "batch_replay_close_receipt.json"
    if (source_replay.get("status") != "COMPLETED_READ_ONLY_CPU_BATCH_REPLAY"
            or source_replay.get("source_checkpoint_sha256") != expected_checkpoint_sha
            or source_replay.get("train_batch", {}).get("start") != 128
            or source_replay.get("train_batch", {}).get("stop") != 192
            or not source_close_path.is_file()):
        raise RuntimeError("prior immutable replay is not bound to the approved checkpoint and TRAIN[128:192]")
    source_close = json.loads(source_close_path.read_text(encoding="utf-8"))
    if (source_close.get("result_sha256") != sha256_file(source_replay_path)
            or source_close.get("layer_trace_sha256") != sha256_file(source_trace_path)):
        raise RuntimeError("prior immutable batch replay hashes do not read back")
    source_smoke_dir = checkpoint_path.parent
    source_readback_path = source_smoke_dir / "entry_smoke_pre_eval_checkpoint_readback.json"
    source_failure_path = source_smoke_dir / "entry_smoke_failure_receipt.json"
    source_delta_path = source_smoke_dir / "entry_smoke_first_nonfinite_eval_delta.json"
    for required in (source_readback_path, source_failure_path, source_delta_path):
        if not required.is_file():
            raise FileNotFoundError(f"required immutable diag-v2 evidence is missing: {required}")
    source_readback = json.loads(source_readback_path.read_text(encoding="utf-8"))
    if (source_readback.get("status") != "PASS_EXACT_PRE_EVALUATION_READBACK"
            or source_readback.get("checkpoint_sha256") != expected_checkpoint_sha):
        raise RuntimeError("diag-v2 checkpoint exact readback no longer validates")
    enforce_cpu_cap("checkpoint and prior replay gates")

    prior_charge, prior_close_hashes = exp055_trainpath_recovery_prior_charge()
    verified_gate = _reuse_root_verified_exp055_train_gate()
    train_keys, values, feature_by_key, input_evidence = _load_exp055_trainpath_train_inputs_reusing_gate(
        verified_gate, SOURCE_RUN_DIR / "response_blind_conditions.tsv", SOURCE_RUN_DIR)
    enforce_cpu_cap("frozen TRAIN input load")
    if (values["X_ctl"].shape != (419, 978) or values["X"].shape != (419, 978)
            or values["delta"].shape != (419, 978) or len(train_keys) != 419
            or train_keys.source_object_id.nunique() != 110
            or train_keys.full_inchikey.nunique() != 5):
        raise ValueError("latent replay inputs changed from frozen 419 TRAIN / 110 objects / 5 keys / 978 genes")
    start, stop, row_in_batch = 128, 192, 22
    replay_keys = train_keys.iloc[start:stop].reset_index(drop=True)
    if len(replay_keys) != 64:
        raise ValueError("latent replay must bind exactly TRAIN eval batch [128:192]")
    feature_matrix = np.stack([feature_by_key[key] for key in
                               train_keys.full_inchikey.astype(str)]).astype(np.float32, copy=False)
    if feature_matrix.shape != (419, 2048) or not np.isfinite(feature_matrix).all():
        raise ValueError("TRAIN feature matrix changed from the frozen 419 x 2048 finite contract")
    key_sha = sha256_text("\n".join(replay_keys.full_condition_key.astype(str).tolist()))
    if key_sha != source_replay.get("train_batch", {}).get("condition_keys_sha256"):
        raise RuntimeError("TRAIN[128:192] key ordering differs from the prior fixed-batch replay")
    enforce_cpu_cap("TRAIN feature assembly and batch identity")

    import torch

    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU latent replay must not initialize CUDA")

    attempt = new_attempt_path(RUN_DIR, attempt_id)
    attempt.mkdir(parents=True, exist_ok=False)
    trace_path = attempt / "latent_replay_trace.jsonl"
    runner_sha = sha256_file(Path(__file__).resolve())
    process_id = os.getpid()
    _open_trainpath_trace_owner(attempt, run_id, attempt_id, process_id, runner_sha)
    trace_path.touch()
    checkpoint_sha_before = sha256_file(checkpoint_path)
    manifest = {"experiment_id": EXPERIMENT_ID, "run_id": run_id,
        "attempt_id": attempt_id, "phase": "read_only_original_forward_latent_capture",
        "status": "CPU_LATENT_REPLAY_RUNNING", "device": "cpu", "wsl_pid": process_id,
        "started_at_utc": started_at, "runner_sha256": runner_sha, "config_sha256": config_sha,
        "source_run_id": SOURCE_RUN_ID, "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": checkpoint_sha_before,
        "source_checkpoint_readback_sha256": sha256_file(source_readback_path),
        "source_failure_receipt_sha256": sha256_file(source_failure_path),
        "source_first_nonfinite_delta_sha256": sha256_file(source_delta_path),
        "prior_batch_replay_result_sha256": sha256_file(source_replay_path),
        "prior_batch_replay_trace_sha256": sha256_file(source_trace_path),
        "prior_batch_replay_close_sha256": sha256_file(source_close_path),
        "replay_contract": {"train_row_start_inclusive": start,
            "train_row_stop_exclusive": stop, "conditions": 64, "gene_count": 978,
            "features_per_key": 2048, "direct_original_model_forward": True,
            "mode": "eval", "optimizer_updates": 0, "checkpoint_write": False,
            "row_with_first_nonfinite_output": row_in_batch,
            "row_global_train_index": start + row_in_batch,
            "rng_seed": SEED + 1,
            "historical_eval_rng_state_recorded_in_prior_manifest": False},
        "target_train_row": replay_keys.iloc[row_in_batch][["full_condition_key",
            "source_object_id", "full_inchikey"]].astype(str).to_dict(),
        "condition_keys_sha256": key_sha, "input_evidence": input_evidence,
        "prior_recovery_charge_gpu_hours": prior_charge,
        "prior_recovery_closures": prior_close_hashes,
        "data_roles": {"inner_dev_numeric_rows_loaded": False,
            "pdo_numeric_sensitivity_or_expression_opened": False,
            "outer_or_locked_numeric_values_opened": False},
        "gpu_hours_charged": 0.0, "command": " ".join(sys.argv)}
    atomic_json(attempt / "latent_replay_manifest.json", manifest)
    atomic_json(attempt / "attempt_progress.json", {"phase": "CPU_LATENT_REPLAY_STARTED",
        "updated_at_utc": utc_now(), "wsl_pid": process_id,
        "inner_dev_numeric_rows_loaded": False})

    original_rng = {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state()}
    try:
        model, _module = _load_transigen(torch, torch.device("cpu"), SEED,
                                         features_dim=2048, seed_cuda=False)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("epoch") != 1 or checkpoint.get("optimizer_updates") != 7:
            raise ValueError("diag-v2 checkpoint metadata differs from its seven-update source")
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("diag-v2 checkpoint lacks a state_dict")
        model.load_state_dict(state_dict, strict=True)
        before_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        before_bn = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
        x1 = torch.as_tensor(values["X_ctl"][start:stop], dtype=torch.float32)
        features = torch.as_tensor(feature_matrix[start:stop], dtype=torch.float32)
        replay_model = copy.deepcopy(model)
        replay_model.eval()
        original_sample_latent = replay_model.sample_latent
        latent_events: list[dict[str, Any]] = []
        sampled_tensors: list[Any] = []
        sample_call = 0

        def observe_sample_latent(mu: Any, logvar: Any) -> Any:
            nonlocal sample_call
            actual_sample = original_sample_latent(mu, logvar)
            std = logvar.mul(0.5).exp_()
            sample_call += 1
            event = {"sample_call": sample_call,
                "purpose": "predicted_perturbation_latent_z2_pred" if sample_call == 2
                    else "upstream_encode_x1_latent_z1",
                "mu_stats": _batch_replay_tensor_stats(torch, mu),
                "logvar_stats": _batch_replay_tensor_stats(torch, logvar),
                "std_exp_0_5_logvar_stats": _batch_replay_tensor_stats(torch, std),
                "sample_stats": _batch_replay_tensor_stats(torch, actual_sample)}
            if sample_call == 2:
                event["row22"] = {"mu": _latent_replay_json_vector(torch, mu[row_in_batch]),
                    "logvar": _latent_replay_json_vector(torch, logvar[row_in_batch]),
                    "std_exp_0_5_logvar": _latent_replay_json_vector(torch, std[row_in_batch]),
                    "sample_z2_pred": _latent_replay_json_vector(torch, actual_sample[row_in_batch])}
            append_jsonl(trace_path, {"run_id": run_id, "attempt_id": attempt_id,
                "event": "original_sample_latent_observed", "timestamp_utc": utc_now(), **event,
                "train_only": True, "inner_dev_numeric_rows_loaded": False,
                "pdo_outer_locked_numeric_values_opened": False})
            latent_events.append(event)
            sampled_tensors.append(actual_sample.detach().clone())
            return actual_sample

        replay_model.sample_latent = observe_sample_latent
        mode_rng = {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state()}
        try:
            random.seed(SEED + 1)
            np.random.seed((SEED + 1) % (2**32))
            torch.random.default_generator.manual_seed(SEED + 1)
            with torch.no_grad():
                forward_outputs = replay_model.forward(x1, features)
                if len(forward_outputs) != 7 or sample_call != 2:
                    raise RuntimeError("original forward did not produce the expected two latent sample calls")
                x2_pred, z2_pred = forward_outputs[3], forward_outputs[6]
                captured_sample = sampled_tensors[1]
                sample_match = ((z2_pred == captured_sample)
                    | (torch.isnan(z2_pred) & torch.isnan(captured_sample))).all().item()
                if not sample_match:
                    raise RuntimeError("original forward latent sample changed unexpectedly")
                decode = x2_pred
                main_delta = x2_pred - x1
                decode_event = {"event": "original_forward_decode_output", "timestamp_utc": utc_now(),
                    "row22": {"decode_x2_pred": _latent_replay_json_vector(torch, decode[row_in_batch]),
                              "main_delta": _latent_replay_json_vector(torch, main_delta[row_in_batch])},
                    "decode_stats": _batch_replay_tensor_stats(torch, decode),
                    "main_delta_stats": _batch_replay_tensor_stats(torch, main_delta),
                    "row22_decode_stats": _batch_replay_tensor_stats(torch, decode[row_in_batch]),
                    "row22_main_delta_stats": _batch_replay_tensor_stats(torch, main_delta[row_in_batch]),
                    "first_nonfinite_decode_module": "decoder_x2"}
                append_jsonl(trace_path, {"run_id": run_id, "attempt_id": attempt_id,
                    "train_only": True, "inner_dev_numeric_rows_loaded": False,
                    "pdo_outer_locked_numeric_values_opened": False, **decode_event})
                after_state = {name: value.detach().cpu().clone()
                               for name, value in replay_model.state_dict().items()}
                after_bn = {name: value.detach().cpu().clone()
                            for name, value in replay_model.named_buffers()}
                state_unchanged = _torch_tree_exact(torch, before_state, after_state)
                bn_unchanged = _torch_tree_exact(torch, before_bn, after_bn)
                if not (state_unchanged and bn_unchanged):
                    raise RuntimeError("eval latent replay modified checkpoint model or BN buffers")
        finally:
            random.setstate(mode_rng["python"])
            np.random.set_state(mode_rng["numpy"])
            torch.set_rng_state(mode_rng["torch_cpu"])
        random.setstate(original_rng["python"])
        np.random.set_state(original_rng["numpy"])
        torch.set_rng_state(original_rng["torch_cpu"])
        if sha256_file(checkpoint_path) != checkpoint_sha_before:
            raise RuntimeError("latent replay changed immutable source checkpoint bytes")
        enforce_cpu_cap("original forward and exact state readback")

        # Drop no rows and do not score a finite subset: this artifact is numerical diagnosis only.
        target_event = latent_events[1]
        std_stats = target_event["std_exp_0_5_logvar_stats"]
        result = {"status": "COMPLETED_READ_ONLY_CPU_LATENT_REPLAY",
            "scientific_result": False, "run_id": run_id, "attempt_id": attempt_id,
            "wsl_pid": process_id, "device": "cpu", "started_at_utc": started_at,
            "closed_at_utc": utc_now(), "elapsed_cpu_seconds": time.perf_counter()-start_clock,
            "gpu_hours_charged": 0.0, "source_checkpoint_sha256": checkpoint_sha_before,
            "source_checkpoint_unchanged": True, "model_state_unchanged_after_eval": True,
            "batchnorm_buffers_unchanged_after_eval": True, "optimizer_updates": 0,
            "train_batch": {"start": start, "stop": stop, "conditions": 64,
                "row22_global_train_index": start + row_in_batch,
                "row22_key": manifest["target_train_row"],
                "condition_keys_sha256": key_sha},
            "rng": {"seed_used": SEED + 1,
                "historical_eval_rng_state_was_not_saved": True,
                "same_noise_sample_as_prior_replay_proven": False,
                "note": "The prior manifest binds checkpoint and batch but did not store the eval RNG state; this run fixes seed 20261007 and captures its actual sample."},
            "sample_latent_calls": [{key: value for key, value in event.items() if key != "row22"}
                | ({"row22": event["row22"]} if "row22" in event else {})
                for event in latent_events],
            "predicted_latent_std_overflow": bool(std_stats["nonfinite_n"] > 0),
            "decoder_output": decode_event,
            "full_train_nmse_computed": False, "finite_subset_nmse_computed": False,
            "inner_dev_numeric_rows_loaded": False,
            "pdo_outer_locked_numeric_values_opened": False,
            "trace_path": trace_path.name, "trace_sha256": sha256_file(trace_path),
            "interpretation": "read-only replay of original model.forward on frozen TRAIN[128:192]; no parameter or optimizer update"}
        atomic_json(attempt / "latent_replay_result.json", result)
        owner = _close_trainpath_trace_owner(attempt, run_id, attempt_id,
            "COMPLETED_READ_ONLY_CPU_LATENT_REPLAY")
        close = {"status": result["status"], "run_id": run_id,
            "attempt_id": attempt_id, "wsl_pid": process_id, "device": "cpu",
            "gpu_hours_charged": 0.0, "elapsed_cpu_seconds": result["elapsed_cpu_seconds"],
            "runner_sha256": runner_sha,
            "result_sha256": sha256_file(attempt / "latent_replay_result.json"),
            "trace_sha256": sha256_file(trace_path),
            "owner_manifest_sha256": sha256_file(_trainpath_trace_owner_path(attempt)),
            "closed_at_utc": utc_now()}
        atomic_json(attempt / "latent_replay_close_receipt.json", close)
        atomic_json(attempt / "attempt_progress.json", {"phase": "CPU_LATENT_REPLAY_CLOSED",
            "updated_at_utc": utc_now(), "wsl_pid": process_id,
            "elapsed_cpu_seconds": result["elapsed_cpu_seconds"], "gpu_hours_charged": 0.0})
        return {"result": result, "close_receipt": close}
    except BaseException as error:
        random.setstate(original_rng["python"])
        np.random.set_state(original_rng["numpy"])
        torch.set_rng_state(original_rng["torch_cpu"])
        owner_path = _trainpath_trace_owner_path(attempt)
        if owner_path.is_file():
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            if owner.get("status") == "OPEN":
                owner.update({"status": "CLOSED", "close_status": "FAILED_CLOSED_CPU_LATENT_REPLAY",
                    "exception_type": type(error).__name__, "exception_message": str(error),
                    "closed_at_utc": utc_now(),
                    "trace_sha256": sha256_file(trace_path) if trace_path.is_file() else None})
                atomic_json(owner_path, owner)
        failure = {"status": "FAILED_CLOSED_CPU_LATENT_REPLAY", "run_id": run_id,
            "attempt_id": attempt_id, "wsl_pid": process_id, "device": "cpu",
            "started_at_utc": started_at, "closed_at_utc": utc_now(),
            "elapsed_cpu_seconds": time.perf_counter()-start_clock, "gpu_hours_charged": 0.0,
            "runner_sha256": runner_sha, "exception_type": type(error).__name__,
            "exception_message": str(error), "trace_sha256": sha256_file(trace_path)
                if trace_path.is_file() else None,
            "owner_manifest_sha256": sha256_file(owner_path) if owner_path.is_file() else None,
            "inner_dev_numeric_rows_loaded": False,
            "pdo_outer_locked_numeric_values_opened": False}
        atomic_json(attempt / "latent_replay_failure_receipt.json", failure)
        raise


def fit_phase_transigen_only(cfg: dict[str, Any], config_sha: str, attempt_id: str,
                             *, preflight_only: bool = False) -> dict[str, Any]:
    """Run only original TranSiGen on the frozen EXP-051 419/93 object-isolated rows."""
    if EXPERIMENT_ID != "EXP-055" or attempt_id != "fit_attempt_007":
        raise ValueError("EXP-055 original TranSiGen fit requires the new immutable fit_attempt_007 slot")
    closed_probe = RUN_DIR / "fit_attempt_007" / "pre_gpu_failure_receipt.json"
    if closed_probe.is_file():
        receipt = _validated_zero_gpu_preflight_receipt(
            RUN_DIR / "fit_attempt_007", closed_probe, RUN_ID)
        if receipt.get("failure", {}).get("exception_type") == "FloatingPointError":
            raise RuntimeError(
                "TranSiGen attempt007 CPU interface preflight is already closed at the main-delta finite gate; "
                "do not retry the probe or start its GPU route")
    limits = resource_limits(cfg)
    accounting = resource_accounting(EXPERIMENT_ROOT, RUN_ID, limits["package_cap_gpu_hours"],
        limits["total_exp_cap_gpu_hours"], limits["exp_settled_before_artifact_ledgers_gpu_hours"],
        limits["package_settled_before_artifact_ledgers_gpu_hours"])
    settled = float(accounting["package_settled_gpu_hours"])
    source_receipts = {name: SOURCE_RUN_DIR / filename for name, filename in {
        "freeze": "metadata_freeze_receipt.json", "axis": "gene_axis_receipt.json",
        "values": "value_contract_receipt.json", "readback": "root_values_readback_receipt.json"}.items()}
    for label, path in source_receipts.items():
        if not path.is_file():
            raise FileNotFoundError(f"frozen TranSiGen input receipt missing: {path}")
    freeze, axis, value_receipt, readback = [json.loads(source_receipts[name].read_text(encoding="utf-8"))
                                              for name in ("freeze", "axis", "values", "readback")]
    values_path = SOURCE_RUN_DIR / "frozen_train_inner_values.npz"
    keys_path = SOURCE_RUN_DIR / "response_blind_conditions.tsv"
    if (freeze.get("run_id") != SOURCE_RUN_ID or axis.get("run_id") != SOURCE_RUN_ID
            or value_receipt.get("run_id") != SOURCE_RUN_ID
            or axis.get("status") != "PASS_EXACT_978_AXIS"
            or readback.get("checks", {}).get("result") != "PASS"
            or sha256_file(values_path) != value_receipt.get("files_sha256", {}).get("frozen_train_inner_values_npz")):
        raise RuntimeError("source EXP-051 frozen identity/value readback contract failed for TranSiGen")
    keys = pd.read_csv(keys_path, sep="\t", dtype=str)
    if len(keys) != 512 or keys.full_condition_key.astype(str).duplicated().any():
        raise ValueError("TranSiGen frozen 512 condition keys are missing or nonunique")
    with np.load(values_path, allow_pickle=True) as stored:
        values = {name: stored[name].copy() for name in
            ("source_row", "source_object_id", "full_inchikey", "pert_id", "genes", "X", "X_ctl", "delta")}
    for col in ("source_object_id", "full_inchikey", "pert_id"):
        if not np.array_equal(keys[col].astype(str).to_numpy(), values[col].astype(str)):
            raise ValueError(f"TranSiGen frozen keys and response-free values differ for {col}")
    train_mask, dev_mask = keys.split_role.eq("train").to_numpy(), keys.split_role.eq("inner_dev").to_numpy()
    train_ids, dev_ids = set(keys.loc[train_mask, "source_object_id"]), set(keys.loc[dev_mask, "source_object_id"])
    if (train_mask.sum() != 419 or dev_mask.sum() != 93 or len(train_ids) != 110
            or len(dev_ids) != 24 or train_ids & dev_ids):
        raise ValueError("TranSiGen TRAIN/inner-dev object split contract failed")
    train_keys, dev_keys = keys.loc[train_mask].reset_index(drop=True), keys.loc[dev_mask].reset_index(drop=True)
    gene_variance = np.var(values["delta"][train_mask].astype(np.float64), axis=0, ddof=0)
    feature_by_key, feature_receipt = _load_audited_ecfp4(SOURCE_RUN_DIR)
    if set(keys.full_inchikey.astype(str)) - set(feature_by_key):
        raise ValueError("CPU-audited full-key ECFP4 features do not cover all frozen TranSiGen keys")
    import torch
    torch.set_num_threads(1)
    cpu_probe_start = time.perf_counter()
    cpu_probe_model, _ = _load_transigen(torch, torch.device("cpu"), SEED,
                                         features_dim=2048, seed_cuda=False)
    cpu_probe_model.eval()
    cpu_probe_positions = np.arange(min(4, len(keys)), dtype=np.int64)
    cpu_probe_features = _full_key_features(keys, cpu_probe_positions, feature_by_key)
    with torch.no_grad():
        cpu_probe_x1 = torch.as_tensor(values["X_ctl"][cpu_probe_positions], dtype=torch.float32)
        cpu_probe_x2 = torch.as_tensor(values["X"][cpu_probe_positions], dtype=torch.float32)
        cpu_probe_feat = torch.as_tensor(cpu_probe_features, dtype=torch.float32)
        _, _, cpu_probe_delta, cpu_probe_losses = _transigen_forward_batch(
            torch, cpu_probe_model, (cpu_probe_x1, cpu_probe_x2, cpu_probe_feat), torch.device("cpu"))
        cpu_probe_main = (cpu_probe_delta - cpu_probe_x1).cpu().numpy()
    if cpu_probe_main.shape != (len(cpu_probe_positions), 978):
        raise ValueError("TranSiGen CPU interface probe did not produce the expected 978-gene main delta shape")
    cpu_interface_receipt = {"status": "PASS_CPU_ORIGINAL_MODEL_SHAPES; NONFINITE_REVIEW_REQUIRED"
                                      if not np.isfinite(cpu_probe_main).all() else "PASS_CPU_ORIGINAL_MODEL_INTERFACE",
        "probe_conditions": int(len(cpu_probe_positions)), "main_delta_shape": list(cpu_probe_main.shape),
        "main_delta_finite": bool(np.isfinite(cpu_probe_main).all()),
        "main_delta_nonfinite_count": int((~np.isfinite(cpu_probe_main)).sum()),
        "main_delta_finite_max_abs": (float(np.max(np.abs(cpu_probe_main[np.isfinite(cpu_probe_main)])))
                                      if np.isfinite(cpu_probe_main).any() else None),
        "input_X_ctl_finite": bool(np.isfinite(values["X_ctl"][cpu_probe_positions]).all()),
        "input_X_finite": bool(np.isfinite(values["X"][cpu_probe_positions]).all()),
        "input_ECFP4_finite": bool(np.isfinite(cpu_probe_features).all()),
        "loss_term_count": len(cpu_probe_losses),
        "loss_terms_finite": [bool(torch.isfinite(term).all()) for term in cpu_probe_losses],
        "cpu_elapsed_seconds": time.perf_counter() - cpu_probe_start,
        "model": "upstream TranSiGen; no optimizer update and no GPU allocation"}
    del cpu_probe_model, cpu_probe_x1, cpu_probe_x2, cpu_probe_feat
    # Historical epoch timing is expanded to include this route's every-epoch TRAIN and inner-dev scoring.
    prior_timing_path = ROOT / "artifacts/experiments/EXP-051/20261006_exp051_representative_compare_v1/fit_attempt_007/attempt_close_receipt.json"
    prior_timing = json.loads(prior_timing_path.read_text(encoding="utf-8"))
    projection = (float(prior_timing["actual_gpu_hours_upper_bound"])
                  / int(prior_timing["completed_train_epochs"]) * 2.5 * 300 * 1.05 * 1.20)
    reservation = projection
    if reservation > accounting["remaining_authorized_gpu_hours"]:
        raise RuntimeError(f"TranSiGen 300-epoch projection {reservation:.4f}h exceeds remaining package/global cap")
    if preflight_only:
        return {"status": "PASS_CPU_TRANSGEN_INTERFACE_NO_ATTEMPT_OR_GPU",
            "run_id": RUN_ID, "attempt": attempt_id, "source_run_id": SOURCE_RUN_ID,
            "config_sha256": config_sha, "runner_sha256": sha256_file(Path(__file__).resolve()),
            "cpu_interface_preflight": cpu_interface_receipt,
            "frozen_condition_keys_sha256": sha256_file(keys_path),
            "frozen_values_sha256": sha256_file(values_path),
            "feature_audit_sha256": feature_receipt.get("feature_npz_sha256",
                feature_receipt.get("output_npz_sha256")),
            "train_conditions": int(train_mask.sum()), "inner_dev_conditions": int(dev_mask.sum()),
            "train_batches": [64, 64, 64, 64, 64, 64, 35],
            "projected_300_epoch_package_reservation_gpu_hours": reservation,
            "package_settled_gpu_hours": settled,
            "package_remaining_gpu_hours": accounting["remaining_authorized_gpu_hours"],
            "model_initialized_on_cpu_only_for_interface_probe": True,
            "optimizer_updates": 0, "cuda_used": False}
    if cfg.get("authorization", {}).get("allowed_device_uuid") != GPU_UUID:
        raise ValueError("EXP-055 registered device UUID no longer matches the sole A5000 allowlist")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise RuntimeError("TranSiGen must use only the approved A5000 UUID")
    attempt = new_attempt_path(RUN_DIR, attempt_id)
    attempt.mkdir(parents=False, exist_ok=False)
    process_id = os.getpid()
    runner_sha = sha256_file(Path(__file__).resolve())
    atomic_json(attempt / "pre_gpu_preflight_started.json", {
        "run_id": RUN_ID, "attempt": attempt_id, "status": "CPU_PREFLIGHT_STARTED",
        "started_at_utc": utc_now(), "wsl_pid": process_id, "runner_sha256": runner_sha,
        "config_sha256": config_sha, "source_run_id": SOURCE_RUN_ID,
        "selected_models": ["transigen"], "reservation_gpu_hours": reservation})
    try:
        baseline = _reuse_exp051_baseline_readbacks(attempt)
        snapshot = _assert_exp055_gpu_idle()
        source_hashes = _source_hashes()
        manifest = {"experiment_id": EXPERIMENT_ID, "run_id": RUN_ID, "fit_attempt": attempt_id,
            "phase": "EXP055_TRANSIGEN_ORIGINAL_FIXED_300_EPOCH_TRAINING", "created_at_utc": utc_now(),
            "wsl_pid": process_id, "command": ("CUDA_VISIBLE_DEVICES=" + GPU_UUID + " PYTHONPATH=src "
            "/home/dell/miniconda3/bin/conda run --no-capture-output -n drugscreening-gpu python "
            "scripts/modeling/exp051_contract_fit_smoke.py --phase fit --fit-kind representative "
            "--models transigen --run-id " + RUN_ID + " --source-run-id " + SOURCE_RUN_ID +
            " --attempt-id " + attempt_id),
            "python_executable": sys.executable, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_uuid": GPU_UUID, "reservation_gpu_hours": reservation,
            "resource_accounting_before_attempt": accounting, "settled_package_baseline_gpu_hours": settled,
            "config_sha256": config_sha, "current_runner_sha256": runner_sha,
            "source_receipts": {name: {"path": str(path), "sha256": sha256_file(path)}
                                for name, path in source_receipts.items()},
            "frozen_inputs": {"condition_table_sha256": sha256_file(keys_path),
                              "values_npz_sha256": sha256_file(values_path)},
            "feature_audit": feature_receipt, "CPU_interface_preflight": cpu_interface_receipt,
            "source_and_feature_sha256": source_hashes,
            "device_preflight": snapshot, "training_schedule": {"epochs": 300, "batch_size": 64,
                "last_batch_size": 35, "shuffle": True, "objective": "raw_sum", "gradient_clipping": False,
                "learning_rate": 1e-3, "weight_decay": 1e-5, "normal_train_batchnorm": True,
                "checkpoints": [100, 300, "final"]},
            "selected_models": ["transigen"], "XPert_initialized_or_evaluated": False,
            "data_roles": {"TRAIN_conditions": 419, "TRAIN_objects": 110,
                           "inner_dev_conditions": 93, "inner_dev_objects": 24, "object_overlap": 0,
                           "PDO_outer_locked_or_external_expression_opened": False}}
        atomic_json(attempt / "fit_smoke_manifest.json", manifest)
        ledger = attempt / "resource_ledger.jsonl"
        append_jsonl(ledger, {"event": "GPU_ATTEMPT_STARTED", "at_utc": utc_now(),
            "wsl_pid": process_id, "device_uuid": GPU_UUID, "selected_models": ["transigen"],
            "reserved_gpu_hours": reservation, "reservation_scope": "TranSiGen original 300 epochs including full TRAIN/dev evaluations and 20% margin",
            "package_settled_before_reservation_gpu_hours": settled,
            "remaining_authorized_gpu_hours": accounting["remaining_authorized_gpu_hours"],
            "runner_sha256": runner_sha, "config_sha256": config_sha,
            "device_snapshot_before_attempt": snapshot["device_snapshot"]["selected_gpu"]})
        import torch
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable in the approved WSL2 Conda environment")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        start_gpu = time.perf_counter()
        progress = {"phase": "TranSiGen_GPU_READY_PRE_MODEL_INITIALIZATION", "updated_at_utc": utc_now(),
                    "wsl_pid": process_id, "completed_epochs": 0, "target_epochs": 300}
        atomic_json(attempt / "attempt_progress.json", progress)
        result = _train_exp055_transigen_only(torch, attempt, keys, values, train_mask, dev_mask,
            train_keys, dev_keys, gene_variance, baseline, feature_by_key, device,
            reservation, start_gpu, progress)
        elapsed = time.perf_counter() - start_gpu
        if elapsed / 3600 > reservation:
            raise TimeoutError("TranSiGen fixed 300-epoch fit exceeded its reserved package balance")
        final_snapshot = _device_snapshot()["selected_gpu"]
        close = {"status": result["status"], "completed_epochs": 300,
                 "actual_gpu_hours": elapsed / 3600, "reserved_gpu_hours": reservation,
                 "unused_reservation_released_gpu_hours": reservation - elapsed / 3600,
                 "cumulative_package_charge_gpu_hours": settled + elapsed / 3600,
                 "runner_sha256": runner_sha, "config_sha256": config_sha,
                 "wsl_pid": process_id, "device_uuid": GPU_UUID,
                 "device_snapshot_at_close": final_snapshot,
                 "predictions_sha256": result["predictions_sha256"]}
        atomic_json(attempt / "attempt_close_receipt.json", close)
        append_jsonl(ledger, {"event": "GPU_ATTEMPT_COMPLETED", "at_utc": utc_now(),
            "wsl_pid": process_id, "device_uuid": GPU_UUID, "actual_gpu_hours": elapsed / 3600,
            "reserved_gpu_hours": reservation, "completed_epochs": 300,
            "unused_authorized_reservation_released_gpu_hours": reservation - elapsed / 3600,
            "predictions_sha256": result["predictions_sha256"],
            "close_receipt": "attempt_close_receipt.json", "gpu_snapshot_at_close": final_snapshot})
        atomic_json(attempt / "attempt_progress.json", {"phase": "CLOSED_COMPLETED", "updated_at_utc": utc_now(),
            "wsl_pid": process_id, "completed_epochs": 300, "actual_gpu_hours": elapsed / 3600})
        return {"full_fit": result, "close_receipt": close}
    except BaseException as error:
        _close_unhandled_attempt(RUN_ID, attempt_id, error)
        raise


def _exp055_attempt008_cpu_preflight(cfg: dict[str, Any], config_sha: str,
                                    *, include_bundle: bool = False,
                                    attempt_id: str = "fit_attempt_008") -> dict[str, Any]:
    """Prove the root cost receipt, frozen inputs, and original XPert schedule on CPU only."""
    if EXPERIMENT_ID != "EXP-055" or RUN_ID != "20261007_exp055_representative_models_v1":
        raise ValueError("the fresh XPert full-fit route is reserved for the registered EXP-055 representative-model run")
    if SOURCE_RUN_ID != "20261006_exp051_contract_fit_smoke_v1":
        raise ValueError("the fresh XPert full-fit route must read the frozen EXP-051 source run")
    if attempt_id not in {"fit_attempt_008", "fit_attempt_009"}:
        raise ValueError(f"the authorized fresh XPert attempts are 008 (closed) and 009: {attempt_id}")
    if next_attempt_id(RUN_DIR) != attempt_id or (RUN_DIR / attempt_id).exists():
        raise RuntimeError(f"{attempt_id} requires prior attempts closed and its own output path unused")

    limits = resource_limits(cfg)
    if (float(limits["package_cap_gpu_hours"]) != 10.0
            or float(limits["total_exp_cap_gpu_hours"]) != 48.0
            or cfg.get("authorization", {}).get("allowed_device_uuid") != GPU_UUID):
        raise ValueError("attempt008 package, global, or approved-device authorization changed")

    root_cost_path = RUN_DIR / "fit_attempt_006" / "root_independent_cost_calibration_readback.json"
    if not root_cost_path.is_file():
        raise FileNotFoundError(f"verified attempt006 root cost receipt is required before attempt008: {root_cost_path}")
    root_cost = json.loads(root_cost_path.read_text(encoding="utf-8"))
    expected_cost_status = "ROOT_INDEPENDENT_READBACK_PASS_COST_CALIBRATION_COMPLETED_RUNNER_POSTPROCESS_FAILED"
    if (root_cost.get("status") != expected_cost_status
            or root_cost.get("experiment_id") != "EXP-055"
            or root_cost.get("run_id") != RUN_ID
            or root_cost.get("source_run_id") != SOURCE_RUN_ID
            or root_cost.get("config_sha256") != config_sha
            or root_cost.get("within_both_caps") is not True
            or root_cost.get("train_conditions") != 419
            or root_cost.get("inner_dev_conditions") != 93
            or root_cost.get("genes") != 978
            or root_cost.get("train_objects") != 110
            or root_cost.get("inner_dev_objects") != 24
            or root_cost.get("object_overlap") != 0
            or root_cost.get("prediction_readback") != "PASS_EXACT_KEYS_TARGETS_AND_FINITE_SHAPES"):
        raise RuntimeError("attempt006 independent root readback does not certify the approved XPert full-run contract")

    calibration_dir = RUN_DIR / "fit_attempt_006"
    calibration_failure = calibration_dir / "failure_close_receipt.json"
    calibration_ledger = calibration_dir / "resource_ledger.jsonl"
    calibration_checkpoint = calibration_dir / "XPert_cost_calibration_final_state.pt"
    calibration_predictions = calibration_dir / "XPert_cost_calibration_predictions.npz"
    for path in (calibration_failure, calibration_ledger, calibration_checkpoint, calibration_predictions):
        if not path.is_file():
            raise FileNotFoundError(f"attempt006 referenced evidence is missing: {path}")
    if (sha256_file(calibration_failure) != root_cost.get("failure_close_receipt_sha256")
            or sha256_file(calibration_checkpoint) != root_cost.get("checkpoint_sha256")
            or sha256_file(calibration_predictions) != root_cost.get("prediction_npz_sha256")):
        raise RuntimeError("attempt006 cost receipt no longer matches its failure closure, checkpoint, or keyed predictions")
    failure_data = json.loads(calibration_failure.read_text(encoding="utf-8"))
    if (failure_data.get("attempt") != "fit_attempt_006"
            or failure_data.get("status") != "FAILED_CLOSED"
            or abs(float(failure_data.get("charged_gpu_hours_conservative_upper_bound", float("nan")))
                   - float(root_cost["actual_attempt_gpu_hours_conservative_upper_bound"])) > 1e-12):
        raise RuntimeError("attempt006 failure close charge differs from the root cost readback")

    failed_tran_dir = RUN_DIR / "fit_attempt_007"
    failed_tran_receipt_path = failed_tran_dir / "pre_gpu_failure_receipt.json"
    failed_tran_receipt = _validated_zero_gpu_preflight_receipt(
        failed_tran_dir, failed_tran_receipt_path, RUN_ID)
    staged_tran_evidence = RUN_DIR / "fit_attempt_007_cpu_preflight_failure_evidence.json"
    if (failed_tran_receipt.get("attempt_id") != "fit_attempt_007"
            or failed_tran_receipt.get("source_run_id") != SOURCE_RUN_ID
            or failed_tran_receipt.get("failure", {}).get("exception_type") != "FloatingPointError"
            or "main deltas" not in failed_tran_receipt.get("failure", {}).get("message", "")
            or not staged_tran_evidence.is_file()
            or sha256_file(staged_tran_evidence).lower()
                != str(failed_tran_receipt.get("failure_evidence_sha256", "")).lower()):
        raise RuntimeError("attempt007 TranSiGen CPU failure closure is missing or differs from its staged evidence")
    with (failed_tran_dir / "resource_ledger.jsonl").open("r", encoding="utf-8") as stream:
        tran_events = [json.loads(line) for line in stream if line.strip()]
    if (len(tran_events) != 1 or tran_events[0].get("event") != "CPU_PREFLIGHT_FAILED_CLOSED_NO_GPU"
            or tran_events[0].get("attempt") != "fit_attempt_007"
            or float(tran_events[0].get("charged_gpu_hours_conservative_upper_bound", float("nan"))) != 0.0):
        raise RuntimeError("attempt007 is not a single immutable zero-GPU CPU failure closure")

    if attempt_id == "fit_attempt_009":
        prior_dir = RUN_DIR / "fit_attempt_008"
        prior_start_path = prior_dir / "pre_gpu_preflight_started.json"
        prior_gate_path = prior_dir / "attempt008_cpu_cost_and_identity_gate.json"
        prior_reallocation_path = prior_dir / "one_time_package_reallocation_receipt.json"
        prior_start = json.loads(prior_start_path.read_text(encoding="utf-8"))
        prior_gate = json.loads(prior_gate_path.read_text(encoding="utf-8"))
        prior_close = _validated_zero_gpu_preflight_receipt(
            prior_dir, prior_dir / "pre_gpu_failure_receipt.json", RUN_ID)
        if (prior_start.get("attempt") != "fit_attempt_008"
                or prior_start.get("source_run_id") != SOURCE_RUN_ID
                or prior_start.get("runner_sha256") != prior_close.get("runner_sha256_at_failure")
                or prior_gate.get("status") != "PASS_CPU_COST_AND_IDENTITY_GATE"
                or prior_gate.get("can_start_if_explicitly_invoked") is not True
                or prior_close.get("failure", {}).get("exception_type") != "KeyError"
                or prior_close.get("failure", {}).get("message") != "'selected_gpu'"
                or prior_close.get("gpu_smoke_started") is not False
                or prior_close.get("gpu_model_initialization_started") is not False
                or not prior_reallocation_path.is_file()):
            raise RuntimeError("attempt008 is not a verified closed zero-GPU route failure with the authorized cost gate")
        prior_reallocation = json.loads(prior_reallocation_path.read_text(encoding="utf-8"))
        if (prior_reallocation.get("status") != "USED_ONCE_WITHIN_EXP055_AUTHORIZED_10_GPUH_PACKAGE"
                or prior_reallocation.get("attempt") != "fit_attempt_008"
                or float(prior_reallocation.get("package_cap_gpu_hours", float("nan"))) != 10.0):
            raise RuntimeError("attempt008 did not preserve the single authorized 5-to-10 GPUh package reallocation")
    elif (RUN_DIR / "fit_attempt_008").exists():
        raise RuntimeError("attempt008 exists; it must be verified and closed before a new XPert attempt")

    accounting = resource_accounting(
        EXPERIMENT_ROOT, RUN_ID,
        limits["package_cap_gpu_hours"], limits["total_exp_cap_gpu_hours"],
        limits["exp_settled_before_artifact_ledgers_gpu_hours"],
        limits["package_settled_before_artifact_ledgers_gpu_hours"])
    if (abs(float(accounting["package_settled_gpu_hours"])
            - float(root_cost["package_settled_after"])) > 1e-8
            or abs(float(accounting["exp_settled_gpu_hours"])
                   - float(root_cost["global_settled_after"])) > 1e-8):
        raise RuntimeError(
            "live attempt accounting differs from the root-verified attempt006 settlement; "
            f"package={accounting['package_settled_gpu_hours']:.12f}, "
            f"global={accounting['exp_settled_gpu_hours']:.12f}")

    xpert_projection = float(root_cost[
        "xpert_2500_epoch_projection_with_5pct_bookkeeping_20pct_margin_gpu_hours"])
    tran_reserve = float(root_cost[
        "transigen_300_epoch_projection_with_5pct_bookkeeping_20pct_margin_gpu_hours"])
    if (not math.isfinite(xpert_projection) or xpert_projection <= 0
            or not math.isfinite(tran_reserve) or tran_reserve <= 0):
        raise ValueError("root cost receipt contains a non-finite or non-positive official model projection")
    package_projected = float(accounting["package_settled_gpu_hours"]) + xpert_projection + tran_reserve
    global_projected = float(accounting["exp_settled_gpu_hours"]) + xpert_projection + tran_reserve
    if (abs(package_projected - float(root_cost["projected_exp055_total_gpu_hours"])) > 1e-6
            or abs(global_projected - float(root_cost["projected_global_total_gpu_hours"])) > 1e-6):
        raise RuntimeError("live XPert plus reserved TranSiGen projection differs from root cost receipt")
    package_cap = float(limits["package_cap_gpu_hours"])
    global_cap = float(limits["total_exp_cap_gpu_hours"])
    package_remaining_after_both = package_cap - package_projected
    global_remaining_after_both = global_cap - global_projected
    first_phase_cap = float(limits["first_phase_gpu_cap_hours"])
    reallocation_files = list(RUN_DIR.glob("fit_attempt_*/one_time_package_reallocation_receipt.json"))
    reallocation_authorized = bool(cfg.get("authorization", {}).get(
        "one_internal_reallocation_within_package", False))
    if reallocation_files and not reallocation_authorized:
        raise RuntimeError("an existing one-time internal reallocation receipt is not covered by the current EXP-055 authorization")
    reallocation_needed = float(accounting["package_settled_gpu_hours"]) + xpert_projection + tran_reserve > first_phase_cap
    reallocation_already_used = bool(reallocation_files)
    reallocation_available = reallocation_authorized and not reallocation_already_used
    reallocation_covered = (not reallocation_needed or reallocation_available or reallocation_already_used)
    cost_pass = (package_remaining_after_both >= -1e-9 and global_remaining_after_both >= -1e-9
                 and reallocation_covered)

    source_receipts = {name: SOURCE_RUN_DIR / filename for name, filename in {
        "freeze": "metadata_freeze_receipt.json", "axis": "gene_axis_receipt.json",
        "values": "value_contract_receipt.json", "readback": "root_values_readback_receipt.json"}.items()}
    for label, path in source_receipts.items():
        if not path.is_file():
            raise FileNotFoundError(f"frozen EXP-051 {label} receipt is missing: {path}")
    freeze, axis, value_receipt, source_readback = [
        json.loads(source_receipts[name].read_text(encoding="utf-8"))
        for name in ("freeze", "axis", "values", "readback")]
    keys_path = SOURCE_RUN_DIR / "response_blind_conditions.tsv"
    values_path = SOURCE_RUN_DIR / "frozen_train_inner_values.npz"
    if (freeze.get("run_id") != SOURCE_RUN_ID or axis.get("run_id") != SOURCE_RUN_ID
            or value_receipt.get("run_id") != SOURCE_RUN_ID
            or axis.get("status") != "PASS_EXACT_978_AXIS"
            or source_readback.get("checks", {}).get("result") != "PASS"
            or source_readback.get("input_sha256") != root_cost.get("source_values_sha256")
            or value_receipt.get("files_sha256", {}).get("frozen_train_inner_values_npz")
                != root_cost.get("source_values_sha256")
            or not keys_path.is_file() or not values_path.is_file()
            or sha256_file(keys_path) != root_cost.get("source_keys_sha256")
            or sha256_file(values_path) != root_cost.get("source_values_sha256")):
        raise RuntimeError("frozen EXP-051 source keys/values differ from their root-verified identities")

    keys = pd.read_csv(keys_path, sep="\t", dtype=str, keep_default_na=False)
    with np.load(values_path, allow_pickle=True) as stored:
        values = {name: stored[name].copy() for name in
                  ("source_row", "source_object_id", "full_inchikey", "pert_id", "split_role",
                   "genes", "X", "X_ctl", "delta")}
    if (len(keys) != 512 or keys["full_condition_key"].astype(str).duplicated().any()
            or values["X"].shape != (512, 978) or values["X_ctl"].shape != (512, 978)
            or values["delta"].shape != (512, 978)
            or len(values["genes"]) != 978 or len(set(values["genes"].astype(str))) != 978
            or not np.isfinite(values["X"]).all() or not np.isfinite(values["X_ctl"]).all()
            or not np.isfinite(values["delta"]).all()):
        raise ValueError("frozen XPert source must contain 512 unique keys, 978 genes, and finite authorized values")
    for column in ("source_object_id", "full_inchikey", "pert_id", "split_role"):
        if column not in keys or column not in values or not np.array_equal(
                keys[column].astype(str).to_numpy(), values[column].astype(str)):
            raise ValueError(f"frozen source key/value alignment failed for {column}")
    train_mask = keys["split_role"].eq("train").to_numpy()
    dev_mask = keys["split_role"].eq("inner_dev").to_numpy()
    train_ids = set(keys.loc[train_mask, "source_object_id"])
    dev_ids = set(keys.loc[dev_mask, "source_object_id"])
    if (train_mask.sum() != 419 or dev_mask.sum() != 93 or len(train_ids) != 110
            or len(dev_ids) != 24 or train_ids & dev_ids
            or keys.loc[train_mask, "full_inchikey"].nunique() != 5):
        raise ValueError("frozen whole-object 419/93 TRAIN/inner-dev assignment or five-key contract changed")
    train_keys = keys.loc[train_mask].reset_index(drop=True)
    dev_keys = keys.loc[dev_mask].reset_index(drop=True)
    gene_variance = np.var(values["delta"][train_mask].astype(np.float64), axis=0, ddof=0)

    with tempfile.TemporaryDirectory(prefix="exp055_attempt008_schedule_") as temp_dir:
        schedule = _verify_exp055_xpert_cpu_schedule(Path(temp_dir))
        xpert_config = yaml.safe_load((XPERT_ROOT / "configs/config_l1000.yaml").read_text(encoding="utf-8"))
    unimol_path = XPERT_ROOT / "processed_data/all_drugs_unimol_arr.npy"
    if not unimol_path.is_file():
        raise FileNotFoundError(f"official XPert UniMol asset is missing: {unimol_path}")
    unimol = np.load(unimol_path, mmap_mode="r")
    if unimol.ndim != 3 or unimol.shape[1:] != (122, 514):
        raise ValueError(f"official XPert UniMol feature shape changed: {unimol.shape}")
    node_ids = pd.to_numeric(keys["drug_node_idx"], errors="raise").to_numpy(dtype=np.int64)
    if node_ids.min() < 0 or node_ids.max() >= unimol.shape[0]:
        raise IndexError("frozen XPert node indices are outside the official UniMol feature array")
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    train_positions = np.flatnonzero(train_mask)
    cpu_tensors = _xpert_tensors(torch, values, keys, train_positions, unimol,
                                 xpert_config, training_targets=True)
    if len(cpu_tensors) != 10 or any(int(tensor.shape[0]) != 419 for tensor in cpu_tensors):
        raise ValueError("official XPert training adapter does not retain all 419 frozen TRAIN keys")
    cpu_loader = DataLoader(TensorDataset(*cpu_tensors), batch_size=128, shuffle=False,
                            drop_last=False, num_workers=0)
    batch_sizes = [int(batch[0].shape[0]) for batch in cpu_loader]
    if batch_sizes != [128, 128, 128, 35]:
        raise ValueError(f"official XPert TRAIN batch plan changed: {batch_sizes}")
    del cpu_loader, cpu_tensors
    with tempfile.TemporaryDirectory(prefix="exp055_attempt008_baseline_") as temp_dir:
        baseline = _reuse_exp051_baseline_readbacks(Path(temp_dir))

    gate = {
        "status": "PASS_CPU_COST_AND_IDENTITY_GATE" if cost_pass else "STOP_CPU_COST_GATE",
        "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID,
        "attempt": attempt_id, "source_run_id": SOURCE_RUN_ID,
        "config_sha256": config_sha, "runner_sha256": sha256_file(Path(__file__).resolve()),
        "root_cost_receipt": {"path": str(root_cost_path), "sha256": sha256_file(root_cost_path),
                              "status": root_cost["status"]},
        "attempt006_settled_charge_gpu_hours": float(root_cost["actual_attempt_gpu_hours_conservative_upper_bound"]),
        "settled_package_gpu_hours": float(accounting["package_settled_gpu_hours"]),
        "settled_global_gpu_hours": float(accounting["exp_settled_gpu_hours"]),
        "xpert_2500_epoch_projection_gpu_hours_including_eval_5pct_bookkeeping_20pct_margin": xpert_projection,
        "tran_300_epoch_projection_gpu_hours_preserved": tran_reserve,
        "projected_package_total_including_settled_and_both_models_gpu_hours": package_projected,
        "projected_global_total_including_settled_and_both_models_gpu_hours": global_projected,
        "package_cap_gpu_hours": package_cap, "package_remaining_after_xpert_and_tran_gpu_hours": package_remaining_after_both,
        "global_cap_gpu_hours": global_cap, "global_remaining_after_xpert_and_tran_gpu_hours": global_remaining_after_both,
        "first_phase_gpu_hours": first_phase_cap,
        "one_time_reallocation_needed": reallocation_needed,
        "one_time_reallocation_available": reallocation_available,
        "one_time_reallocation_already_used": reallocation_already_used,
        "can_start_if_explicitly_invoked": cost_pass,
        "decision": ("PASS_WITHIN_10_GPUH_AFTER_AUTHORIZED_ONE_TIME_REALLOCATION_ALREADY_USED"
                     if cost_pass and reallocation_needed and reallocation_already_used
                     else "PASS_WITHIN_10_GPUH_AFTER_AUTHORIZED_ONE_TIME_REALLOCATION"
                     if cost_pass and reallocation_needed else "PASS_WITHIN_FIRST_PHASE"
                     if cost_pass else "STOP_PACKAGE_GLOBAL_OR_INTERNAL_REALLOCATION_GATE"),
        "official_xpert_schedule": schedule,
        "train_conditions": 419, "train_objects": 110,
        "inner_dev_conditions": 93, "inner_dev_objects": 24,
        "genes": 978, "train_batches": batch_sizes,
        "frozen_condition_keys_sha256": sha256_file(keys_path),
        "frozen_values_sha256": sha256_file(values_path),
        "baseline_readbacks_verified": baseline["readback_refs"],
        "attempt007_tran_status": "CLOSED_ZERO_GPU_NO_RETRY",
        "attempt008_route_status": ("CLOSED_ZERO_GPU_KEYERROR_NO_REPLAY" if attempt_id == "fit_attempt_009"
                                    else "CURRENT_FRESH_XPERT_ROUTE"),
        "model_initialization_started": False, "cuda_initialized": False,
        "pdo_outer_locked_opened": False,
    }
    if include_bundle:
        gate["_training_bundle"] = {"keys": keys, "values": values,
                                    "train_mask": train_mask, "dev_mask": dev_mask,
                                    "train_keys": train_keys, "dev_keys": dev_keys,
                                    "gene_variance": gene_variance,
                                    "baseline_metrics": baseline,
                                    "unimol_path": unimol_path,
                                    "unimol": unimol,
                                    "official_schedule": schedule}
    return gate


def fit_phase_xpert_full_attempt008(cfg: dict[str, Any], config_sha: str,
                                    attempt_id: str, *, preflight_only: bool = False) -> dict[str, Any]:
    """Run fresh upstream XPert on all frozen TRAIN keys, after root-receipt cost gating."""
    if attempt_id not in {"fit_attempt_008", "fit_attempt_009"}:
        raise ValueError("the independent original XPert full-fit route accepts only the closed 008 or current 009 slot")
    gate = _exp055_attempt008_cpu_preflight(cfg, config_sha, include_bundle=not preflight_only,
                                            attempt_id=attempt_id)
    if preflight_only:
        return {key: value for key, value in gate.items() if key != "_training_bundle"}
    if not gate["can_start_if_explicitly_invoked"]:
        raise RuntimeError(
            f"{attempt_id} stopped on CPU before model/GPU initialization: "
            f"projected package={gate['projected_package_total_including_settled_and_both_models_gpu_hours']:.9f}/"
            f"{gate['package_cap_gpu_hours']:.3f} GPUh; remaining after XPert+Tran="
            f"{gate['package_remaining_after_xpert_and_tran_gpu_hours']:.9f} GPUh; "
            f"projected global={gate['projected_global_total_including_settled_and_both_models_gpu_hours']:.9f}/"
            f"{gate['global_cap_gpu_hours']:.3f} GPUh")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise RuntimeError(f"{attempt_id} requires CUDA_VISIBLE_DEVICES set only to the approved A5000 UUID")

    bundle = gate.pop("_training_bundle")
    snapshot = _assert_exp055_gpu_idle()
    reservation = float(gate["xpert_2500_epoch_projection_gpu_hours_including_eval_5pct_bookkeeping_20pct_margin"])
    if reservation + float(gate["tran_300_epoch_projection_gpu_hours_preserved"]) > float(
            gate["package_cap_gpu_hours"]) - float(gate["settled_package_gpu_hours"]) + 1e-9:
        raise RuntimeError(f"{attempt_id} XPert reservation plus the protected TranSiGen reserve no longer fits the package")
    source_hashes = _source_hashes()
    before_attempt = resource_accounting(
        EXPERIMENT_ROOT, RUN_ID, resource_limits(cfg)["package_cap_gpu_hours"],
        resource_limits(cfg)["total_exp_cap_gpu_hours"],
        resource_limits(cfg)["exp_settled_before_artifact_ledgers_gpu_hours"],
        resource_limits(cfg)["package_settled_before_artifact_ledgers_gpu_hours"])
    attempt = new_attempt_path(RUN_DIR, attempt_id)
    attempt.mkdir(parents=False, exist_ok=False)
    process_id = os.getpid()
    atomic_json(attempt / "pre_gpu_preflight_started.json", {
        "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID, "attempt": attempt_id,
        "selected_models": ["xpert"], "status": "CPU_PREFLIGHT_STARTED",
        "started_at_utc": utc_now(), "wsl_pid": process_id,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "source_run_id": SOURCE_RUN_ID, "config_sha256": config_sha,
        "root_cost_gate_sha256": gate["root_cost_receipt"]["sha256"]})
    ledger_path = attempt / "resource_ledger.jsonl"
    schedule_receipt = bundle["official_schedule"]
    baseline = bundle["baseline_metrics"]
    atomic_json(attempt / "XPert_official_schedule_cpu_receipt.json", schedule_receipt)
    atomic_json(attempt / "baseline_metrics.json", baseline)
    if gate["one_time_reallocation_needed"] and not gate["one_time_reallocation_already_used"]:
        atomic_json(attempt / "one_time_package_reallocation_receipt.json", {
            "status": "USED_ONCE_WITHIN_EXP055_AUTHORIZED_10_GPUH_PACKAGE",
            "reason": "Official XPert 2500-epoch maximum plus the retained TranSiGen reserve exceeds the initial 5 GPUh phase but fits the existing 10 GPUh package",
            "prior_package_settled_gpu_hours": gate["settled_package_gpu_hours"],
            "xpert_2500_epoch_projection_gpu_hours": gate["xpert_2500_epoch_projection_gpu_hours_including_eval_5pct_bookkeeping_20pct_margin"],
            "reserved_tran_300_epoch_gpu_hours": gate["tran_300_epoch_projection_gpu_hours_preserved"],
            "projected_package_total_gpu_hours": gate["projected_package_total_including_settled_and_both_models_gpu_hours"],
            "package_cap_gpu_hours": gate["package_cap_gpu_hours"],
            "created_at_utc": utc_now(), "run_id": RUN_ID, "attempt": attempt_id})
    manifest = {
        "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID, "fit_attempt": attempt_id,
        "selected_models": ["xpert"], "phase": "EXP055_XPERT_FRESH_ORIGINAL_FULL_2500_OR_PATIENCE50",
        "created_at_utc": utc_now(), "wsl_pid": process_id,
        "command": "CUDA_VISIBLE_DEVICES=" + GPU_UUID + " PYTHONPATH=src /home/dell/miniconda3/bin/conda run --no-capture-output -n drugscreening-gpu python scripts/modeling/exp051_contract_fit_smoke.py --phase fit --fit-kind representative --experiment-id EXP-055 --run-id " + RUN_ID + " --source-run-id " + SOURCE_RUN_ID + " --attempt-id " + attempt_id + " --models xpert",
        "python_executable": sys.executable, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_uuid": GPU_UUID, "reservation_gpu_hours": reservation,
        "resource_accounting_before_attempt": before_attempt,
        "root_independent_cost_gate": {key: value for key, value in gate.items()
                                       if key not in {"official_xpert_schedule", "baseline_readbacks_verified"}},
        "source_run_id": SOURCE_RUN_ID,
        "frozen_files": {"conditions_tsv_sha256": gate["frozen_condition_keys_sha256"],
                         "values_npz_sha256": gate["frozen_values_sha256"]},
        "current_runner_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": config_sha, "source_and_feature_sha256": source_hashes,
        "model_contract": "fresh upstream XPertNet initialization; all 419 TRAIN keys; batch128; author warmup/raw switch at epoch index 70; author scheduler; inner-dev val_loss4 with patience50; at most 2500 epochs; no prior calibration checkpoint reuse",
        "data_roles": {"TRAIN_conditions": 419, "TRAIN_objects": 110,
                       "inner_dev_conditions": 93, "inner_dev_objects": 24,
                       "object_overlap": 0, "timing_smoke_conditions": None,
                       "PDO_outer_locked_or_external_expression_opened": False},
        "XPert_official_schedule_cpu_receipt": schedule_receipt,
        "evidence_boundary": ["This is the separate official full-fit route; attempt006 calibration weights are not loaded.",
                              "TRAIN is used for fitting; inner_dev selects the author val_loss4 checkpoint.",
                              "TranSiGen attempt007 remains closed after its CPU main-delta finite-value failure.",
                              "XPert attempt008 remains closed after CPU manifest assembly failed; this attempt uses the corrected GPU snapshot field."]}
    atomic_json(attempt / "fit_smoke_manifest.json", manifest)
    atomic_json(attempt / f"{attempt_id}_cpu_cost_and_identity_gate.json", gate)
    append_jsonl(ledger_path, {"event": "GPU_ATTEMPT_STARTED", "at_utc": utc_now(),
                               "wsl_pid": process_id, "device_uuid": GPU_UUID,
                               "selected_models": ["xpert"], "reserved_gpu_hours": reservation,
                               "prior_attempts_settled_gpu_hours": gate["settled_package_gpu_hours"],
                               "protected_tran_reserve_gpu_hours": gate["tran_300_epoch_projection_gpu_hours_preserved"],
                               "projected_package_total_gpu_hours": gate["projected_package_total_including_settled_and_both_models_gpu_hours"],
                               "pre_attempt_gpu_snapshot": snapshot["device_snapshot"]["selected_gpu"],
                               "runner_sha256": manifest["current_runner_sha256"], "config_sha256": config_sha})
    try:
        import torch
        torch.set_num_threads(1)
        started_gpu = time.perf_counter()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable in the approved WSL2 Conda environment")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        xpert_module, xpert_config, _, _ = _training_modules(torch)
        xpert_config["dataset"]["drug_unimol_path"] = str(bundle["unimol_path"])
        if (int(xpert_config["train"]["num_epochs"]) != 2500
                or int(xpert_config["train"]["init_epoch"]) != 70
                or int(xpert_config["train"]["patience"]) != 50
                or int(xpert_config["train"]["batch_size"]) != 128
                or float(xpert_config["train"]["train_lr"]) != 0.004
                or float(xpert_config["train"]["weight_decay"]) != 1e-5
                or list(xpert_config["train"]["loss_weight"]) != [0.2, 0.003, 0.2, 1]):
            raise ValueError("loaded XPert training configuration no longer matches the CPU-gated official schedule")
        atomic_json(attempt / "attempt_progress.json", {"phase": "XPert_OFFICIAL_FULL_FIT_STARTED",
                                                         "updated_at_utc": utc_now(), "wsl_pid": process_id,
                                                         "schedule": "max 2500 epochs; val_loss4; patience 50",
                                                         "fresh_initialization": True})
        full = _train_xpert_only(
            torch, attempt, bundle["keys"], bundle["values"], bundle["train_mask"],
            bundle["dev_mask"], bundle["train_keys"], bundle["dev_keys"],
            bundle["gene_variance"], baseline, bundle["unimol"],
            xpert_module, xpert_config, device, reservation, started_gpu)
        full["fit_gate_baseline_denominator"] = {
            "strongest_baseline_candidates": ["zero_change", "per_drug_mean_delta"],
            "baseline_metrics_sha256": sha256_file(attempt / "baseline_metrics.json"),
            "TRAIN_conditions": 419, "TRAIN_objects": 110,
            "normalizer_fit_on": "TRAIN only"}
        atomic_json(attempt / "full_fit_result.json", full)
        torch.cuda.synchronize()
        elapsed_gpu_hours = (time.perf_counter() - started_gpu) / 3600.0
        if elapsed_gpu_hours > reservation:
            raise TimeoutError("XPert official full-fit exceeded its root-readback-derived GPU reservation")
        final_snapshot = _device_snapshot()["selected_gpu"]
        close = {"status": "COMPLETED_XPERT_OFFICIAL_SCHEDULE_ONLY",
                 "selected_models": ["xpert"],
                 "completed_epochs": full["best_metrics"]["XPert"]["epochs_completed"],
                 "selected_epoch_by_val_loss4": full["best_metrics"]["XPert"]["selected_epoch_by_inner_dev_val_loss4"],
                 "stopped_by_patience": full["best_metrics"]["XPert"]["stopped_by_patience"],
                 "actual_gpu_hours": elapsed_gpu_hours, "reserved_gpu_hours": reservation,
                 "unused_reservation_released_gpu_hours": reservation - elapsed_gpu_hours,
                 "projected_package_total_including_tran_reserve_gpu_hours": gate["projected_package_total_including_settled_and_both_models_gpu_hours"],
                 "device_uuid": GPU_UUID, "device_snapshot_at_close": final_snapshot,
                 "runner_sha256": manifest["current_runner_sha256"], "config_sha256": config_sha,
                 "process_id": process_id, "full_keyed_predictions_sha256": full["predictions_sha256"],
                 "best_checkpoint_sha256": full["best_metrics"]["XPert"]["checkpoint_best_sha256"],
                 "final_checkpoint_sha256": full["best_metrics"]["XPert"]["checkpoint_final_sha256"]}
        atomic_json(attempt / "attempt_close_receipt.json", close)
        append_jsonl(ledger_path, {"event": "GPU_ATTEMPT_COMPLETED", "at_utc": utc_now(),
                                   "wsl_pid": process_id, "device_uuid": GPU_UUID,
                                   "selected_models": ["xpert"], "actual_gpu_hours": elapsed_gpu_hours,
                                   "reserved_gpu_hours": reservation,
                                   "unused_reservation_released_gpu_hours": reservation - elapsed_gpu_hours,
                                   "close_receipt": "attempt_close_receipt.json"})
        atomic_json(attempt / "attempt_progress.json", {"phase": "CLOSED_COMPLETED",
                                                         "updated_at_utc": utc_now(), "wsl_pid": process_id,
                                                         "actual_gpu_hours": elapsed_gpu_hours,
                                                         "completed_epochs": close["completed_epochs"]})
        return {"cost_gate": {key: value for key, value in gate.items()
                               if key != "_training_bundle"},
                "full_fit": full, "close_receipt": close}
    except BaseException as error:
        _close_unhandled_attempt(RUN_ID, attempt_id, error)
        raise


def fit_phase_xpert_only(cfg: dict[str, Any], config_sha: str, attempt_id: str,
                         *, cost_calibration: bool = False) -> dict[str, Any]:
    """Run XPert alone; EXP-055 uses the original 2500-epoch/val_loss4 contract."""
    limits = resource_limits(cfg)
    accounting = resource_accounting(
        EXPERIMENT_ROOT, RUN_ID,
        limits["package_cap_gpu_hours"], limits["total_exp_cap_gpu_hours"],
        limits["exp_settled_before_artifact_ledgers_gpu_hours"],
        limits["package_settled_before_artifact_ledgers_gpu_hours"])
    settled = float(accounting["settled_gpu_hours"])
    reservation = float(accounting["remaining_authorized_gpu_hours"])
    if reservation <= 0:
        raise RuntimeError(f"no authorized {EXPERIMENT_ID} package or experiment GPU balance remains")
    if EXPERIMENT_ID == "EXP-055":
        expected_id = "fit_attempt_006" if cost_calibration else "fit_attempt_005"
        if attempt_id != expected_id:
            raise ValueError(f"EXP-055 {'cost calibration' if cost_calibration else 'XPert smoke'} requires {expected_id}; attempts 001-005 are immutable")
        if cost_calibration:
            calibration_cap = 0.05
            if reservation + 1e-9 < calibration_cap:
                raise RuntimeError("settled EXP-055 package balance cannot reserve the 0.05 GPU-hour calibration cap")
            reservation = calibration_cap
    attempt = new_attempt_path(RUN_DIR, attempt_id)
    receipts = {"freeze": SOURCE_RUN_DIR / "metadata_freeze_receipt.json",
                "axis": SOURCE_RUN_DIR / "gene_axis_receipt.json",
                "values": SOURCE_RUN_DIR / "value_contract_receipt.json",
                "root_values_readback": SOURCE_RUN_DIR / "root_values_readback_receipt.json"}
    for label, path in receipts.items():
        if not path.is_file():
            raise FileNotFoundError(f"required {label} gate receipt missing: {path}")
    freeze = json.loads(receipts["freeze"].read_text(encoding="utf-8"))
    axis = json.loads(receipts["axis"].read_text(encoding="utf-8"))
    value_receipt = json.loads(receipts["values"].read_text(encoding="utf-8"))
    readback = json.loads(receipts["root_values_readback"].read_text(encoding="utf-8"))
    source_config_sha = freeze.get("config_sha256")
    if (freeze.get("run_id") != SOURCE_RUN_ID or axis.get("run_id") != SOURCE_RUN_ID
            or value_receipt.get("run_id") != SOURCE_RUN_ID
            or not source_config_sha or axis.get("config_sha256") != source_config_sha
            or axis.get("metadata_freeze_receipt_sha256") != sha256_file(receipts["freeze"])
            or axis.get("status") != "PASS_EXACT_978_AXIS"
            or readback.get("checks", {}).get("result") != "PASS"
            or value_receipt.get("files_sha256", {}).get("frozen_train_inner_values_npz") != readback.get("input_sha256")):
        raise RuntimeError("frozen train/inner response and exact gene-axis gates failed")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    attempt.mkdir(parents=False, exist_ok=False)
    process_id = os.getpid()
    atomic_json(attempt / "pre_gpu_preflight_started.json", {
        "run_id": RUN_ID, "attempt": attempt_id, "selected_models": ["xpert"],
        "status": "CPU_PREFLIGHT_STARTED", "started_at_utc": utc_now(),
        "wsl_pid": process_id, "runner_sha256": sha256_file(Path(__file__).resolve()),
        "source_run_id": SOURCE_RUN_ID, "config_sha256": config_sha})
    keys_path = SOURCE_RUN_DIR / "response_blind_conditions.tsv"
    values_path = SOURCE_RUN_DIR / "frozen_train_inner_values.npz"
    keys = pd.read_csv(keys_path, sep="\t", dtype=str)
    # This internally generated, checksum-bound NPZ contains NumPy object arrays
    # for categorical key columns; only the frozen source artifact is opened here.
    if sha256_file(values_path) != value_receipt.get("files_sha256", {}).get("frozen_train_inner_values_npz"):
        raise ValueError("frozen response-value artifact hash differs from its authorized receipt")
    with np.load(values_path, allow_pickle=True) as frozen:
        values = {name: frozen[name].copy() for name in
                  ("selection_rank", "source_row", "split_role", "source_object_id", "full_inchikey",
                   "pert_id", "genes", "X", "X_ctl", "delta")}
    if len(keys) != 512 or values["X"].shape != (512, 978) or values["X_ctl"].shape != (512, 978):
        raise ValueError("XPert selected-model fit requires the frozen 512x978 values")
    for col in ("source_object_id", "full_inchikey", "pert_id", "split_role"):
        if not np.array_equal(keys[col].astype(str).to_numpy(), values[col].astype(str)):
            raise ValueError(f"frozen key/value mismatch for {col}")
    train_mask = keys["split_role"].eq("train").to_numpy()
    dev_mask = keys["split_role"].eq("inner_dev").to_numpy()
    train_ids = set(keys.loc[train_mask, "source_object_id"])
    dev_ids = set(keys.loc[dev_mask, "source_object_id"])
    if (train_mask.sum() != 419 or dev_mask.sum() != 93 or len(train_ids) != 110
            or len(dev_ids) != 24 or train_ids & dev_ids):
        raise ValueError("whole-object TRAIN/inner-development split contract failed")

    # XPert feature-node identity check uses only source metadata; no other model asset is opened.
    import anndata as ad
    meta = ad.read_h5ad(H5AD, backed="r")
    try:
        source_meta = meta.obs.iloc[values["source_row"].astype(np.int64)][["pert_id", "pert_idx"]].copy()
    finally:
        meta.file.close()
    source_meta = source_meta.reset_index(drop=True)
    if not np.array_equal(source_meta.pert_id.astype(str).to_numpy(), keys.pert_id.astype(str).to_numpy()):
        raise ValueError("source H5AD metadata drug identities differ from the frozen full-key table")
    source_nodes = pd.to_numeric(source_meta.pert_idx, errors="raise").to_numpy(dtype=np.int64)
    frozen_nodes = pd.to_numeric(keys.drug_node_idx, errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(source_nodes, frozen_nodes):
        raise ValueError("source XPert feature nodes differ from the frozen condition keys")

    train_keys = keys.loc[train_mask].reset_index(drop=True)
    dev_keys = keys.loc[dev_mask].reset_index(drop=True)
    train_y, dev_y = values["delta"][train_mask], values["delta"][dev_mask]
    gene_variance = np.var(train_y.astype(np.float64), axis=0, ddof=0)
    baseline = (_reuse_exp051_baseline_readbacks(attempt) if EXPERIMENT_ID == "EXP-055"
                else data_baselines(keys, values["delta"], gene_variance, train_mask, dev_mask, attempt))
    train_positions = np.flatnonzero(train_mask)
    drug_keys = sorted(keys.loc[train_mask, "full_inchikey"].astype(str).unique())
    if len(drug_keys) != 5:
        raise ValueError("XPert smoke requires the fixed five full-InChIKey groups")
    chosen = []
    for full_key, count in zip(drug_keys, (13, 13, 13, 13, 12)):
        available = train_positions[keys.loc[train_mask, "full_inchikey"].astype(str).to_numpy() == full_key]
        ordered = sorted(available, key=lambda pos: (str(keys.iloc[pos].source_object_id),
                                                      str(keys.iloc[pos].full_condition_key)))
        if len(ordered) < count:
            raise ValueError(f"fixed 64-row smoke lacks sufficient TRAIN keys for {full_key}")
        chosen.extend(ordered[:count])
    smoke_positions = np.asarray(sorted(chosen), dtype=np.int64)
    smoke_keys = keys.iloc[smoke_positions].reset_index(drop=True)
    smoke_keys.to_csv(attempt / "fit_smoke_64_keys.tsv", sep="\t", index=False, lineterminator="\n")
    xpert_schedule_receipt = (_verify_exp055_xpert_cpu_schedule(attempt)
                              if EXPERIMENT_ID == "EXP-055" else None)
    snapshot = _device_snapshot()
    if snapshot["selected_gpu"]["memory_used_mib"] > 1024 or snapshot["selected_gpu"]["utilization_percent"] > 20:
        raise RuntimeError("approved A5000 is not idle enough for selected XPert fit")
    source_hashes = _source_hashes()
    manifest = {
        "experiment_id": EXPERIMENT_ID, "run_id": RUN_ID, "fit_attempt": attempt_id,
        "selected_models": ["xpert"],
        "phase": ("EXP055_XPERT_FULL_TRAIN_COST_CALIBRATION_1_WARMUP_1_RAW_EPOCH"
                  if cost_calibration else "XPert_only_64_key_smoke_cost_gate_then_official_2500_patience50_schedule"
                  if EXPERIMENT_ID == "EXP-055" else "XPert_only_64_key_one_update_cost_gate_then_fixed100_fit"),
        "created_at_utc": utc_now(), "wsl_pid": process_id,
        "command": ("CUDA_VISIBLE_DEVICES=" + GPU_UUID + " PYTHONPATH=src "
                    "/home/dell/miniconda3/bin/conda run --no-capture-output -n drugscreening-gpu "
                    "python scripts/modeling/exp051_contract_fit_smoke.py --phase "
                    + ("cost-calibration" if cost_calibration else "fit") + " --models xpert "
                    "--run-id " + RUN_ID + " --source-run-id " + SOURCE_RUN_ID + " --attempt-id " + attempt_id),
        "python_executable": sys.executable, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_uuid": GPU_UUID, "reservation_gpu_hours": reservation,
        "resource_accounting_before_attempt": accounting, "source_run_id": SOURCE_RUN_ID,
        "source_config_sha256": source_config_sha,
        "resource_limits": limits,
        "frozen_receipts": {name: {"path": str(path), "sha256": sha256_file(path)}
                            for name, path in receipts.items()},
        "frozen_files": {"conditions_tsv_sha256": sha256_file(keys_path),
                         "values_npz_sha256": sha256_file(values_path)},
        "current_runner_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": config_sha, "source_and_feature_sha256": source_hashes,
        "model_contract": ("upstream XPertNet scratch, full official 2500-epoch maximum with val_loss4/patience50, batch128, source loss/scheduler and epoch70 transition; TranSiGen was not imported, initialized, evaluated, or feature-audited"
                           if EXPERIMENT_ID == "EXP-055" else "upstream XPertNet scratch, batch128, source loss/scheduler, first 70 epoch transition, 100 epochs; TranSiGen was not imported, initialized, evaluated, or feature-audited"),
        "data_roles": {"TRAIN_conditions": 419, "TRAIN_objects": 110,
                       "inner_dev_conditions": 93, "inner_dev_objects": 24,
                       "object_overlap": 0,
                       "timing_smoke_conditions": None if cost_calibration else 64,
                       "cost_calibration_conditions": 419 if cost_calibration else None,
                       "timing_smoke_full_keys": len(drug_keys),
                       "PDO_outer_locked_or_external_expression_opened": False},
        "XPert_official_schedule_cpu_receipt": xpert_schedule_receipt,
        "evidence_boundary": (["This phase measures full-TRAIN cost for one official warmup epoch and one official raw-stage epoch; it does not select a model or establish fit.",
                               "Only upstream XPertNet was initialized and evaluated."] if cost_calibration else
                              ["Smoke is only for cost and interface, not a model result.",
                               "Only the selected original model was initialized and evaluated."])}
    atomic_json(attempt / "fit_smoke_manifest.json", manifest)
    ledger = attempt / "resource_ledger.jsonl"
    append_jsonl(ledger, {"event": "GPU_ATTEMPT_STARTED" if cost_calibration else "GPU_SMOKE_STARTED", "at_utc": utc_now(),
                          "wsl_pid": process_id, "device_uuid": GPU_UUID,
                          "selected_models": ["xpert"], "reserved_gpu_hours": reservation,
                          "paired_smoke_cap_gpu_hours": (0.05 if cost_calibration else limits["paired_smoke_cap_gpu_hours"]),
                          "prior_attempts_settled_gpu_hours": settled,
                          "pre_smoke_gpu_snapshot": snapshot["selected_gpu"],
                          "runner_sha256": manifest["current_runner_sha256"], "config_sha256": config_sha})
    import torch
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in the approved WSL2 Conda environment")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    start_gpu = time.perf_counter()
    atomic_json(attempt / "attempt_progress.json", {"phase": "XPert_GPU_READY_PRE_MODEL_INITIALIZATION",
                                                     "updated_at_utc": utc_now(), "wsl_pid": process_id})
    try:
        unimol_path = XPERT_ROOT / "processed_data/all_drugs_unimol_arr.npy"
        unimol_asset = np.load(unimol_path, mmap_mode="r")
        if unimol_asset.ndim != 3 or unimol_asset.shape[1:] != (122, 514):
            raise ValueError(f"unexpected author UniMol shape: {unimol_asset.shape}")
        node_ids = pd.to_numeric(keys.drug_node_idx, errors="raise").to_numpy(dtype=np.int64)
        if node_ids.max() >= unimol_asset.shape[0]:
            raise IndexError("frozen XPert drug node exceeds source UniMol feature array")
        unimol = unimol_asset
        del unimol_asset
        tick = time.perf_counter()
        xpert_module, xpert_config, _, _ = _training_modules(torch)
        xpert_config["dataset"]["drug_unimol_path"] = str(unimol_path)
        if (int(xpert_config["dataset"]["n_bins"]) != 128
                or float(xpert_config["dataset"]["min_value"]) != 0.0
                or float(xpert_config["dataset"]["max_value"]) != 15.002
                or int(xpert_config["train"]["init_epoch"]) != 70
                or int(xpert_config["train"]["batch_size"]) != 128
                or (EXPERIMENT_ID == "EXP-055" and (
                    int(xpert_config["train"]["num_epochs"]) != 2500
                    or int(xpert_config["train"]["patience"]) != 50
                    or float(xpert_config["train"]["train_lr"]) != 0.004
                    or float(xpert_config["train"]["weight_decay"]) != 1e-5
                    or list(xpert_config["train"]["loss_weight"]) != [0.2, 0.003, 0.2, 1]))):
            raise ValueError("upstream XPert schedule or fixed-bin contract changed")
        original_train, _, _, source_receipt = _source_functions(
            XPERT_ROOT / "train_xpert.py", XPERT_ROOT / "utils.py")
        model, train_view = _new_xpert(torch, xpert_module, xpert_config, device, SEED)
        tensors = _xpert_tensors(torch, values, keys, smoke_positions, unimol,
                                 xpert_config, training_targets=True)
        if cost_calibration:
            _run_exp055_cost_calibration(
                cfg, attempt, ledger, process_id, reservation, accounting, torch, model, train_view,
                original_train, source_receipt, values, keys, train_mask, dev_mask,
                train_keys, dev_keys, gene_variance, baseline, unimol, xpert_config,
                device, start_gpu, config_sha, time.perf_counter() - tick)
            return {"attempt": str(attempt), "phase": "cost-calibration", "completed": True}
        opt_raw = torch.optim.Adam(model.parameters(), lr=0.004, weight_decay=1e-5)
        opt = ProbeOptimizer(opt_raw, model, torch)
        before_params = [p.detach().clone() for p in model.parameters()]
        before_train = _xpert_predict(torch, model, values, keys, smoke_positions, unimol, xpert_config)
        dev_positions = np.flatnonzero(dev_mask)
        before_dev = _xpert_predict(torch, model, values, keys, dev_positions, unimol, xpert_config)
        init_seconds = time.perf_counter() - tick
        class SmokeArgs:
            pretrained_model = False
        update_started = time.perf_counter()
        smoke_loss = original_train(train_view, opt, [tensors], SmokeArgs(), xpert_config,
                                    scaler=None, epoch=0)
        torch.cuda.synchronize()
        update_seconds = time.perf_counter() - update_started
        eval_started = time.perf_counter()
        after_train = _xpert_predict(torch, model, values, keys, smoke_positions, unimol, xpert_config)
        after_dev = _xpert_predict(torch, model, values, keys, dev_positions, unimol, xpert_config)
        torch.cuda.synchronize()
        if (not all(math.isfinite(float(value)) for value in smoke_loss)
                or not np.isfinite(before_train).all() or not np.isfinite(before_dev).all()
                or not np.isfinite(after_train).all() or not np.isfinite(after_dev).all()
                or not opt.gradient_norms or not math.isfinite(float(opt.gradient_norms[-1]))):
            raise FloatingPointError("XPert one-update smoke produced non-finite loss, gradient, or predictions")
        eval_seconds = time.perf_counter() - eval_started
        smoke_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        smoke_elapsed = time.perf_counter() - start_gpu
        if smoke_elapsed / 3600 > limits["paired_smoke_cap_gpu_hours"]:
            raise TimeoutError("XPert-only 64-key smoke exceeded its 0.25 GPU-hour cap")
        prediction_path = attempt / "smoke_predictions.npz"
        np.savez_compressed(prediction_path,
                            train64_condition_key=np.asarray(smoke_keys.full_condition_key.astype(str).to_numpy(), dtype=np.str_),
                            inner_dev_condition_key=np.asarray(dev_keys.full_condition_key.astype(str).to_numpy(), dtype=np.str_),
                            genes=np.asarray(values["genes"].astype(str), dtype=np.str_),
                            train64_target_delta=values["delta"][smoke_positions],
                            inner_dev_target_delta=dev_y, xpert_train64_before_delta=before_train,
                            xpert_train64_after_delta=after_train, xpert_inner_dev_after_delta=after_dev)
        with np.load(prediction_path, allow_pickle=False) as smoke_saved:
            expected_smoke = {
                "train64_condition_key": np.asarray(smoke_keys.full_condition_key.astype(str).to_numpy(), dtype=np.str_),
                "inner_dev_condition_key": np.asarray(dev_keys.full_condition_key.astype(str).to_numpy(), dtype=np.str_),
                "genes": np.asarray(values["genes"].astype(str), dtype=np.str_),
                "train64_target_delta": values["delta"][smoke_positions],
                "inner_dev_target_delta": dev_y,
                "xpert_train64_before_delta": before_train,
                "xpert_train64_after_delta": after_train,
                "xpert_inner_dev_after_delta": after_dev,
            }
            for name, expected in expected_smoke.items():
                if name not in smoke_saved.files or not np.array_equal(smoke_saved[name], expected):
                    raise ValueError(f"XPert smoke prediction readback mismatch: {name}")
        smoke_checkpoint_finite = all(
            (not isinstance(value, torch.Tensor) or not value.is_floating_point()
             or bool(torch.isfinite(value).all())) for value in smoke_state.values())
        if not smoke_checkpoint_finite:
            raise FloatingPointError("XPert one-update smoke checkpoint contains non-finite model parameters")
        torch.save({"state_dict": smoke_state, "model": "upstream XPertNet", "updates": len(opt.gradient_norms),
                    "seed": SEED, "official_train_function_sha256": source_receipt["train_function_source_sha256"]},
                   attempt / "XPert_smoke_one_update_state.pt")
        results = {"selected_models": ["xpert"], "smoke_metrics": {"XPert": {
                        "train64_before": keyed_metrics(smoke_keys, values["delta"][smoke_positions], before_train, gene_variance),
                        "train64_after": keyed_metrics(smoke_keys, values["delta"][smoke_positions], after_train, gene_variance),
                        "inner_dev93_after": keyed_metrics(dev_keys, dev_y, after_dev, gene_variance),
                        "loss_terms": [safe_number(float(x)) for x in smoke_loss],
                        "gradient_norm": safe_number(opt.gradient_norms[-1]) if opt.gradient_norms else None,
                        "parameter_update_l2": _norm_difference(torch, before_params, model),
                        "n_updates": len(opt.gradient_norms), "prediction_max_abs": float(np.max(np.abs(after_dev)))}}}
        del before_params, tensors, opt, opt_raw, model, train_view
        torch.cuda.empty_cache()
        raw_epoch = math.ceil(419 / 128) * update_seconds * 2.0 + eval_seconds * (512.0 / 157.0)
        init_plus = init_seconds
        planned_epochs = int(xpert_config["train"]["num_epochs"]) if EXPERIMENT_ID == "EXP-055" else 100
        projected_raw = planned_epochs * raw_epoch + init_plus
        projected_with_margin = projected_raw * 1.20 / 3600
        total_projected = smoke_elapsed / 3600 + projected_with_margin
        projected_package_total = float(accounting["package_settled_gpu_hours"]) + total_projected
        projected_global_total = float(accounting["exp_settled_gpu_hours"]) + total_projected
        within_package = projected_package_total <= limits["package_cap_gpu_hours"]
        within_exp = projected_global_total <= limits["total_exp_cap_gpu_hours"]
        first_phase_limit = float(limits.get("first_phase_gpu_cap_hours", limits["package_cap_gpu_hours"]))
        reallocation_receipts = list(RUN_DIR.glob("fit_attempt_*/one_time_package_reallocation_receipt.json"))
        projected_within_first_phase = projected_package_total <= first_phase_limit
        reallocation_available = (EXPERIMENT_ID == "EXP-055"
                                  and bool(cfg.get("authorization", {}).get("one_internal_reallocation_within_package", False))
                                  and not reallocation_receipts)
        proceed = within_package and within_exp and (projected_within_first_phase
                                                      or reallocation_available)
        if proceed and not projected_within_first_phase:
            atomic_json(attempt / "one_time_package_reallocation_receipt.json", {
                "status": "USED_ONCE_WITHIN_EXP055_AUTHORIZED_PACKAGE",
                "reason": "XPert-only full official 2500-epoch maximum cost projection exceeds the 5 GPUh first phase but remains within the original 10 GPUh package cap and 48 GPUh EXP cap",
                "prior_package_settled_gpu_hours": accounting["package_settled_gpu_hours"],
                "projected_package_total_gpu_hours": projected_package_total,
                "package_cap_gpu_hours": limits["package_cap_gpu_hours"],
                "prior_exp_settled_gpu_hours": accounting["exp_settled_gpu_hours"],
                "projected_exp_total_gpu_hours": projected_global_total,
                "exp_cap_gpu_hours": limits["total_exp_cap_gpu_hours"],
                "created_at_utc": utc_now(), "run_id": RUN_ID, "attempt": attempt_id})
        results["cost_gate"] = {
            "projection_method": f"XPert-only observed 64-key update and 64-train+93-inner-dev evaluation; batch128 update conservatively 2x batch64; maximum {planned_epochs} epochs and all per-epoch TRAIN/inner-dev evaluation, initialization allowance, plus20% margin. Official patience=50 is not used to reduce the pre-run projection.",
            "observed_update_seconds": update_seconds, "observed_eval_seconds": eval_seconds,
            "planned_max_epochs": planned_epochs,
            "projected_raw_seconds_per_epoch": raw_epoch, "projected_full_fit_gpu_hours_with_20_percent_margin": projected_with_margin,
            "smoke_gpu_hours": smoke_elapsed / 3600, "smoke_cap_gpu_hours": limits["paired_smoke_cap_gpu_hours"],
            "smoke_plus_full_fit_with_margin_gpu_hours": total_projected,
            "remaining_attempt_reservation_gpu_hours": reservation,
            "exp_settled_gpu_hours_before_this_attempt": settled,
            "package_cap_gpu_hours": limits["package_cap_gpu_hours"], "exp_cap_gpu_hours": limits["total_exp_cap_gpu_hours"],
            "projected_package_total_gpu_hours": projected_package_total,
            "projected_global_total_gpu_hours": projected_global_total,
            "first_phase_cap_gpu_hours": first_phase_limit,
            "one_time_reallocation_available": reallocation_available,
            "decision": ("PASS_COST_GATE_CONTINUE_XPERT_OFFICIAL_2500_PATIENCE50_SCHEDULE"
                         if proceed and EXPERIMENT_ID == "EXP-055"
                         else "PASS_COST_GATE_CONTINUE_XPERT_100_EPOCH_FIT" if proceed
                         else "STOP_AFTER_SMOKE_CLOSE_PACKAGE_OR_GLOBAL_COST_GATE")}
        results["source_train_function"] = source_receipt
        results["baseline_metrics_sha256"] = sha256_file(attempt / "baseline_metrics.json")
        results["smoke_predictions_sha256"] = sha256_file(prediction_path)
        results["model_initialization_and_preupdate_eval_seconds"] = init_seconds
        results["cpu_audited_features"] = {"XPert_UniMol_path": str(unimol_path), "loaded": True,
                                            "TranSiGen_ECFP4_loaded_or_audited": False}
        atomic_json(attempt / "fit_smoke_result.json", results)
        if not proceed:
            final_gpu = _device_snapshot()["selected_gpu"]
            append_jsonl(ledger, {"event": "GPU_ATTEMPT_CLOSED_AFTER_COST_GATE", "at_utc": utc_now(),
                                  "wsl_pid": process_id, "device_uuid": GPU_UUID,
                                  "selected_models": ["xpert"], "actual_gpu_hours": smoke_elapsed / 3600,
                                  "reserved_gpu_hours": reservation,
                                  "unused_authorized_reservation_released_gpu_hours": max(0.0, reservation - smoke_elapsed / 3600),
                                  "cost_gate_passed": False, "gpu_snapshot_at_close": final_gpu})
            atomic_json(attempt / "attempt_progress.json", {"phase": "CLOSED_COST_GATE_FAILED",
                                                             "updated_at_utc": utc_now(), "wsl_pid": process_id})
            return results
        append_jsonl(ledger, {"event": "FULL_FIT_STARTED", "at_utc": utc_now(), "wsl_pid": process_id,
                              "device_uuid": GPU_UUID, "selected_models": ["xpert"], "epochs_max": planned_epochs,
                              "smoke_gpu_hours_already_charged_within_reservation": smoke_elapsed / 3600,
                              "projected_total_with_margin_gpu_hours": total_projected})
        atomic_json(attempt / "attempt_progress.json", {"phase": "XPert_FULL_FIT_STARTED",
                                                         "updated_at_utc": utc_now(), "wsl_pid": process_id,
                                                         "completed_smoke_updates": {"XPert": results["smoke_metrics"]["XPert"]["n_updates"]},
                                                         "schedule": "official max 2500 epochs, val_loss4, patience 50" if EXPERIMENT_ID == "EXP-055" else "fixed 100 epoch legacy"})
        full = _train_xpert_only(torch, attempt, keys, values, train_mask, dev_mask,
                                 train_keys, dev_keys, gene_variance, baseline, unimol,
                                 xpert_module, xpert_config, device, reservation, start_gpu)
        full["fit_gate_baseline_denominator"] = {"strongest_baseline_candidates": ["zero_change", "per_drug_mean_delta"],
                                                 "baseline_metrics_sha256": results["baseline_metrics_sha256"],
                                                 "TRAIN_conditions": 419, "TRAIN_objects": 110,
                                                 "normalizer_fit_on": "TRAIN only"}
        atomic_json(attempt / "full_fit_result.json", full)
        elapsed = time.perf_counter() - start_gpu
        if elapsed / 3600 > reservation:
            raise TimeoutError("XPert selected-model attempt exceeded its reserved GPU balance")
        final_gpu = _device_snapshot()["selected_gpu"]
        close = {"status": "COMPLETED_XPERT_ONLY", "selected_models": ["xpert"],
                 "completed_epochs": {"XPert": full["best_metrics"]["XPert"]["epochs_completed"]},
                 "cost_gate": results["cost_gate"],
                 "full_fit_summary": full["best_metrics"], "actual_gpu_hours": elapsed / 3600,
                 "reserved_gpu_hours": reservation, "unused_reservation_released_gpu_hours": reservation - elapsed / 3600,
                 "cumulative_conservative_settled_gpu_hours": settled + elapsed / 3600,
                 "runner_sha256": manifest["current_runner_sha256"], "config_sha256": config_sha,
                 "process_id": process_id, "device_uuid": GPU_UUID,
                 "device_snapshot_at_close": final_gpu,
                 "full_keyed_predictions_sha256": full["predictions_sha256"]}
        atomic_json(attempt / "attempt_close_receipt.json", close)
        append_jsonl(ledger, {"event": "GPU_ATTEMPT_COMPLETED", "at_utc": utc_now(),
                              "wsl_pid": process_id, "device_uuid": GPU_UUID, "selected_models": ["xpert"],
                              "actual_gpu_hours": elapsed / 3600, "reserved_gpu_hours": reservation,
                              "unused_reservation_released_gpu_hours": reservation - elapsed / 3600,
                              "cumulative_conservative_charge_including_prior_attempts_gpu_hours": settled + elapsed / 3600,
                              "close_receipt": "attempt_close_receipt.json"})
        atomic_json(attempt / "attempt_progress.json", {"phase": "CLOSED_COMPLETED",
                                                         "updated_at_utc": utc_now(), "wsl_pid": process_id,
                                                         "actual_gpu_hours": elapsed / 3600})
        return {"smoke": results, "full_fit": full, "close_receipt": close}
    finally:
        try:
            del unimol
            torch.cuda.empty_cache()
        except Exception:
            pass


def _close_unhandled_attempt(run_id: str, attempt_id: str, error: BaseException) -> None:
    """Close only an attempt created by this exact process and runner version."""
    configure_run(run_id)
    validate_attempt_id(attempt_id)
    attempt = RUN_DIR / attempt_id
    trace_owner_path = _trainpath_trace_owner_path(attempt)
    if trace_owner_path.is_file():
        owner = json.loads(trace_owner_path.read_text(encoding="utf-8"))
        if (owner.get("run_id") == run_id and owner.get("attempt_id") == attempt_id
                and owner.get("wsl_pid") == os.getpid() and owner.get("status") == "OPEN"):
            owner.update({"status": "CLOSED", "close_status": "FAILED_CLOSED",
                          "exception_type": type(error).__name__,
                          "exception_message": str(error), "closed_at_utc": utc_now(),
                          "trace_sha256": sha256_file(attempt / "TranSiGen_trainpath_layer_trace.jsonl")
                              if (attempt / "TranSiGen_trainpath_layer_trace.jsonl").is_file() else None})
            atomic_json(trace_owner_path, owner)
    manifest_path = attempt / "fit_smoke_manifest.json"
    ledger_path = attempt / "resource_ledger.jsonl"
    if not manifest_path.is_file() and not ledger_path.is_file():
        preflight_path = attempt / "pre_gpu_preflight_started.json"
        close_path = attempt / "pre_gpu_failure_receipt.json"
        if not preflight_path.is_file() or close_path.exists():
            return
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if (preflight.get("run_id") != RUN_ID or preflight.get("attempt") != attempt_id
                or preflight.get("wsl_pid") != os.getpid()
                or preflight.get("runner_sha256") != sha256_file(Path(__file__).resolve())):
            return
        close = {"run_id": RUN_ID, "attempt": attempt_id,
                 "status": "CPU_PREFLIGHT_FAILED_CLOSED_NO_GPU",
                 "started_at_utc": preflight["started_at_utc"],
                 "observed_close_time_utc": utc_now(), "wsl_pid": os.getpid(),
                 "runner_sha256": preflight["runner_sha256"],
                 "config_sha256": preflight["config_sha256"],
                 "source_run_id": SOURCE_RUN_ID,
                 "charged_gpu_hours": 0.0, "gpu_smoke_started": False,
                 "reserved_gpu_hours": 0.0, "model_initialization_started": False,
                 "exception_type": type(error).__name__, "exception_message": str(error)}
        atomic_json(close_path, close)
        print(json.dumps({"stage": close["status"], "attempt": str(attempt),
                          "charged_gpu_hours": 0.0,
                          "failure": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)
        return
    if not manifest_path.is_file() or not ledger_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("run_id") != RUN_ID or manifest.get("fit_attempt") != attempt_id
            or manifest.get("wsl_pid") != os.getpid()
            or manifest.get("current_runner_sha256") != sha256_file(Path(__file__).resolve())
            or (attempt / "failure_close_receipt.json").exists()):
        return
    entries = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    start = next((entry for entry in entries if entry.get("event") in
                  {"GPU_SMOKE_STARTED", "GPU_ATTEMPT_STARTED"}), None)
    if start is None or any(entry.get("event") in {"GPU_ATTEMPT_COMPLETED", "GPU_ATTEMPT_CLOSED_AFTER_COST_GATE",
                                                     "GPU_ATTEMPT_FAILED_CLOSED", "GPU_SMOKE_FAILED_CLOSED"} for entry in entries):
        return
    now = datetime.now(timezone.utc)
    started_at = datetime.fromisoformat(str(start["at_utc"]).replace("Z", "+00:00"))
    seconds = max(0.0, (now - started_at).total_seconds())
    hours = seconds / 3600
    progress_path = attempt / "attempt_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
    prior_accounting = manifest.get("resource_accounting_before_attempt", {})
    prior_settled_hours = float(prior_accounting.get("settled_gpu_hours", 0.0))
    reservation_hours = float(start.get("reserved_gpu_hours", manifest.get("reservation_gpu_hours", 0.0)))
    close = {"run_id": RUN_ID, "attempt": attempt_id,
             "status": "FAILED_CLOSED",
             "runner_sha256": manifest.get("current_runner_sha256"),
             "started_at_utc": start["at_utc"], "observed_close_time_utc": utc_now(),
             "elapsed_wall_seconds_conservative_upper_bound": seconds,
             "charged_gpu_hours_conservative_upper_bound": hours,
             "prior_attempts_settled_gpu_hours": prior_settled_hours,
             "cumulative_conservative_settled_gpu_hours": prior_settled_hours + hours,
             "reserved_gpu_hours": reservation_hours,
             "unused_reservation_released_gpu_hours": max(0.0, reservation_hours - hours),
             "last_progress": progress, "exception_type": type(error).__name__,
             "exception_message": str(error)}
    atomic_json(attempt / "failure_close_receipt.json", close)
    try:
        snapshot = _device_snapshot()["selected_gpu"]
    except Exception as snapshot_error:
        snapshot = {"read_error": str(snapshot_error)}
    append_jsonl(ledger_path, {"event": "GPU_ATTEMPT_FAILED_CLOSED", "at_utc": utc_now(),
                               "device_uuid": GPU_UUID, "wsl_pid": manifest.get("wsl_pid"),
                               "charged_gpu_hours_conservative_upper_bound": hours,
                               "reserved_gpu_hours": reservation_hours,
                               "unused_authorized_reservation_released_gpu_hours": max(0.0, reservation_hours - hours),
                               "prior_attempts_settled_gpu_hours": prior_settled_hours,
                               "cumulative_conservative_settled_gpu_hours": prior_settled_hours + hours,
                               "last_progress": progress.get("phase"),
                               "close_receipt": "failure_close_receipt.json",
                               "gpu_snapshot_after_failure": snapshot})
    print(json.dumps({"stage": "GPU_ATTEMPT_FAILED_CLOSED", "attempt": str(attempt),
                      "charged_gpu_hours_conservative_upper_bound": hours,
                      "failure": f"{type(error).__name__}: {error}"}, ensure_ascii=False), flush=True)


def _close_all_unhandled_attempts(run_id: str, error: BaseException) -> None:
    """Close any started EXP attempt left open by a multi-attempt EXP-055 run."""
    run_dir = RUN_DIR if RUN_ID == run_id else EXPERIMENT_ROOT / run_id
    for attempt in sorted(run_dir.glob("fit_attempt_[0-9][0-9][0-9]"), reverse=True):
        _close_unhandled_attempt(run_id, attempt.name, error)


def _exp055_cost_calibration_cpu_preflight(cfg: dict[str, Any], config_sha: str) -> dict[str, Any]:
    """Verify attempt006 frozen inputs, official XPert schedule, and batch API without CUDA/model init."""
    if EXPERIMENT_ID != "EXP-055":
        raise ValueError("cost-calibration is only registered for EXP-055")
    if next_attempt_id(RUN_DIR) != "fit_attempt_006":
        raise RuntimeError(f"cost calibration requires next immutable slot fit_attempt_006, got {next_attempt_id(RUN_DIR)}")
    limits = resource_limits(cfg)
    accounting = resource_accounting(EXPERIMENT_ROOT, RUN_ID, limits["package_cap_gpu_hours"],
        limits["total_exp_cap_gpu_hours"], limits["exp_settled_before_artifact_ledgers_gpu_hours"],
        limits["package_settled_before_artifact_ledgers_gpu_hours"])
    if accounting["remaining_authorized_gpu_hours"] < 0.05:
        raise RuntimeError("package ledger cannot reserve 0.05 GPUh for cost calibration")
    source_root = SOURCE_RUN_DIR
    receipt_path = source_root / "value_contract_receipt.json"
    keys_path = source_root / "response_blind_conditions.tsv"
    values_path = source_root / "frozen_train_inner_values.npz"
    if not all(path.is_file() for path in (receipt_path, keys_path, values_path)):
        raise FileNotFoundError("EXP-051 frozen condition keys, values, or receipt are missing")
    value_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if sha256_file(values_path) != value_receipt.get("files_sha256", {}).get("frozen_train_inner_values_npz"):
        raise ValueError("frozen EXP-051 values SHA differs from its receipt")
    keys = pd.read_csv(keys_path, sep="\t", dtype=str)
    with np.load(values_path, allow_pickle=True) as source:
        values = {name: source[name].copy() for name in
                  ("source_row", "source_object_id", "full_inchikey", "pert_id", "genes", "X", "X_ctl", "delta")}
    train_mask, dev_mask = keys["split_role"].eq("train").to_numpy(), keys["split_role"].eq("inner_dev").to_numpy()
    if (len(keys) != 512 or values["X"].shape != (512, 978) or values["X_ctl"].shape != (512, 978)
            or values["delta"].shape != (512, 978) or train_mask.sum() != 419 or dev_mask.sum() != 93
            or set(keys.loc[train_mask, "source_object_id"]) & set(keys.loc[dev_mask, "source_object_id"])):
        raise ValueError("frozen EXP-051 key/value shapes or object-isolated split fail calibration contract")
    with tempfile.TemporaryDirectory(prefix="exp055_xpert_schedule_") as tmp_dir:
        schedule = _verify_exp055_xpert_cpu_schedule(Path(tmp_dir))
    import torch
    tensors = _xpert_tensors(torch, values, keys, np.flatnonzero(train_mask),
        np.load(XPERT_ROOT / "processed_data/all_drugs_unimol_arr.npy", mmap_mode="r"),
        yaml.safe_load((XPERT_ROOT / "configs/config_l1000.yaml").read_text(encoding="utf-8")),
        training_targets=True)
    if len(tensors) != 10 or any(int(tensor.shape[0]) != 419 for tensor in tensors):
        raise ValueError("XPert official 10-input training API does not accept all 419 frozen TRAIN keys")
    return {"status": "PASS_CPU_INTERFACE_NO_CUDA_OR_MODEL_INIT", "run_id": RUN_ID,
            "source_run_id": SOURCE_RUN_ID, "config_sha256": config_sha,
            "next_attempt": "fit_attempt_006", "package_settled_gpu_hours": accounting["package_settled_gpu_hours"],
            "package_remaining_gpu_hours": accounting["remaining_authorized_gpu_hours"],
            "reservation_gpu_hours": 0.05, "train_keys": int(train_mask.sum()),
            "inner_dev_keys": int(dev_mask.sum()), "genes": int(values["delta"].shape[1]),
            "XPert_input_fields": len(tensors), "official_schedule": schedule,
            "frozen_conditions_tsv_sha256": sha256_file(keys_path),
            "frozen_values_npz_sha256": sha256_file(values_path),
            "model_initialization_started": False, "cuda_used": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--experiment-id", default="EXP-051",
                        help="Registered experiment whose artifact root owns this run")
    parser.add_argument("--fit-kind", choices=["contract", "representative"], default="contract",
                        help="Fit contract; representative selects the registered original-model schedule")
    parser.add_argument("--output-root", type=Path,
                        help="Optional EXP-### output directory under artifacts/experiments")
    parser.add_argument("--run-id", help="Output run_id; required for fit and always selects the output directory")
    parser.add_argument("--source-run-id", help="Run containing the already-frozen response-blind fit inputs")
    parser.add_argument("--phase", choices=["metadata", "axis", "values", "feature_audit", "fit", "entry-smoke", "layer-replay", "latent-replay", "cost-calibration", "cpu-contract-check", "checkpoint-replay"], required=True)
    parser.add_argument("--attempt-id", help="Required for fit, entry-smoke, and cost-calibration")
    parser.add_argument("--models", choices=["xpert", "transigen", "both"], default="both",
                        help="Models initialized/evaluated by --phase fit; default preserves the paired smoke")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Run the CPU-only EXP-055 calibration interface check without reserving GPU")
    parser.add_argument("--trainpath-recovery", action="store_true",
                        help="Run the registered EXP-055 TranSiGen recovery path")
    parser.add_argument("--device", choices=["cpu"],
                        help="CPU device for registered entry-smoke or EXP-055 representative routes")
    parser.add_argument("--reviewed-preflight-sha256",
                        help="SHA-256 of this run's immutable CPU preflight receipt")
    parser.add_argument("--resume-from-checkpoint", type=Path,
                        help="Registered immutable EXP-055 CPU checkpoint to replay or resume")
    args = parser.parse_args()
    if args.fit_kind == "representative" and args.experiment_id not in {"EXP-055", "EXP-059"}:
        parser.error("--fit-kind representative is currently registered only for EXP-055/059")
    if args.experiment_id == "EXP-055" and args.fit_kind != "representative":
        parser.error("EXP-055 requires --fit-kind representative")
    if args.experiment_id == "EXP-059" and (args.fit_kind != "representative"
            or args.run_id != EXP059_TRANSIGEN_LR_STABILITY_RUN_ID):
        parser.error("EXP-059 requires its one registered representative run")
    if args.trainpath_recovery and (args.experiment_id != "EXP-055"
            or args.fit_kind != "representative" or args.models != "transigen"):
        parser.error("--trainpath-recovery requires the registered EXP-055 original TranSiGen model")
    if args.trainpath_recovery and args.phase == "fit" and (
            args.run_id != EXP055_TRAINPATH_RECOVERY_RUN_ID or args.attempt_id != "fit_attempt_001"):
        parser.error("recovery fit requires the new immutable run and fit_attempt_001")
    if args.trainpath_recovery and args.phase == "entry-smoke" and (
            args.run_id not in {EXP055_TRAINPATH_ENTRY_SMOKE_RUN_ID,
                                EXP055_TRAINPATH_ENTRY_DIAG_RUN_ID}
            or args.attempt_id != "fit_attempt_001" or args.device != "cpu"):
        parser.error("recovery entry-smoke requires an immutable registered run, fit_attempt_001, and --device cpu")
    if args.trainpath_recovery and args.phase == "layer-replay" and (
            args.run_id != EXP055_TRAINPATH_BATCH_REPLAY_RUN_ID
            or args.attempt_id != "fit_attempt_001" or args.device != "cpu"):
        parser.error("read-only layer replay requires its registered immutable run, fit_attempt_001, and --device cpu")
    if args.trainpath_recovery and args.phase == "latent-replay" and (
            args.run_id != EXP055_TRAINPATH_LATENT_REPLAY_RUN_ID
            or args.attempt_id != "fit_attempt_001" or args.device != "cpu"):
        parser.error("read-only latent replay requires its new immutable run, fit_attempt_001, and --device cpu")
    if args.phase == "cpu-contract-check" and (
            args.experiment_id not in {"EXP-055", "EXP-059"} or args.fit_kind != "representative"
            or args.models != "transigen"
            or args.run_id not in {EXP055_TRANSIGEN_CPU_REPRESENTATIVE_RUN_ID,
                                   EXP059_TRANSIGEN_LR_STABILITY_RUN_ID}
            or (args.experiment_id == "EXP-055" and args.run_id != EXP055_TRANSIGEN_CPU_REPRESENTATIVE_RUN_ID)
            or (args.experiment_id == "EXP-059" and args.run_id != EXP059_TRANSIGEN_LR_STABILITY_RUN_ID)
            or args.attempt_id != "fit_attempt_001" or args.device != "cpu"):
        parser.error("CPU contract check requires one registered EXP-055/059 TranSiGen run and --device cpu")
    if args.phase == "checkpoint-replay" and (
            args.experiment_id != "EXP-055" or args.fit_kind != "representative"
            or args.models != "transigen" or args.run_id != EXP055_TRANSIGEN_CPU_REPLAY_RUN_ID
            or args.attempt_id != "fit_attempt_001" or args.device != "cpu"
            or args.resume_from_checkpoint is None or args.trainpath_recovery):
        parser.error("checkpoint replay requires its registered immutable run, attempt, source checkpoint, and --device cpu")
    if args.phase == "fit" and args.run_id == EXP055_TRANSIGEN_CPU_RESUME_RUN_ID and (
            args.experiment_id != "EXP-055" or args.fit_kind != "representative"
            or args.models != "transigen" or args.attempt_id != "fit_attempt_001"
            or args.device != "cpu" or args.resume_from_checkpoint is None or args.trainpath_recovery):
        parser.error("CPU resume fit requires its registered immutable run/attempt and exact checkpoint path")
    if args.phase == "fit" and args.run_id == EXP055_TRANSIGEN_CPU_NORM_RECOVERY_RUN_ID and (
            args.experiment_id != "EXP-055" or args.fit_kind != "representative"
            or args.models != "transigen" or args.attempt_id != "fit_attempt_001"
            or args.device != "cpu" or args.resume_from_checkpoint is None or args.trainpath_recovery):
        parser.error("CPU norm-recovery fit requires its fresh preflight-bound immutable run and epoch-210 checkpoint")
    if args.device == "cpu" and args.phase == "fit" and (
            args.experiment_id not in {"EXP-055", "EXP-059"} or args.fit_kind != "representative"
            or args.models != "transigen" or args.run_id not in {
                EXP055_TRANSIGEN_CPU_REPRESENTATIVE_RUN_ID, EXP055_TRANSIGEN_CPU_RESUME_RUN_ID,
                EXP055_TRANSIGEN_CPU_NORM_RECOVERY_RUN_ID, EXP059_TRANSIGEN_LR_STABILITY_RUN_ID}
            or (args.experiment_id == "EXP-055" and args.run_id == EXP059_TRANSIGEN_LR_STABILITY_RUN_ID)
            or (args.experiment_id == "EXP-059" and args.run_id != EXP059_TRANSIGEN_LR_STABILITY_RUN_ID)
            or args.attempt_id != "fit_attempt_001" or args.trainpath_recovery):
        parser.error("--device cpu fit requires the registered immutable CPU TranSiGen run/attempt")
    if args.phase == "fit" and args.run_id == EXP055_TRANSIGEN_CPU_REPRESENTATIVE_RUN_ID and args.device != "cpu":
        parser.error("the registered CPU TranSiGen representative run requires --device cpu")
    if args.phase == "entry-smoke" and (not args.trainpath_recovery or args.device != "cpu"):
        parser.error("--phase entry-smoke is reserved for --trainpath-recovery --device cpu")
    if args.phase == "layer-replay" and (not args.trainpath_recovery or args.device != "cpu"):
        parser.error("--phase layer-replay is reserved for --trainpath-recovery --device cpu")
    if args.phase == "latent-replay" and (not args.trainpath_recovery or args.device != "cpu"):
        parser.error("--phase latent-replay is reserved for --trainpath-recovery --device cpu")
    if args.device is not None and args.phase not in {"entry-smoke", "layer-replay", "latent-replay", "fit", "cpu-contract-check", "checkpoint-replay"}:
        parser.error("--device is only valid for registered CPU entry or fit phases")
    if args.resume_from_checkpoint is not None and not (
            args.phase == "checkpoint-replay"
            or (args.phase == "fit" and args.run_id in {
                EXP055_TRANSIGEN_CPU_RESUME_RUN_ID, EXP055_TRANSIGEN_CPU_NORM_RECOVERY_RUN_ID})):
        parser.error("--resume-from-checkpoint is reserved for the registered checkpoint replay/resume package")
    if args.trainpath_recovery and args.preflight_only and args.reviewed_preflight_sha256:
        parser.error("CPU-only preflight cannot include a GPU launch receipt SHA")
    if args.trainpath_recovery and args.phase == "fit" and not args.preflight_only and not args.reviewed_preflight_sha256:
        parser.error("GPU recovery fit requires --reviewed-preflight-sha256 from this run's CPU gate")
    if args.phase == "entry-smoke" and (args.preflight_only or args.reviewed_preflight_sha256):
        parser.error("CPU entry-smoke runs directly and cannot consume or write a GPU preflight receipt")
    if args.reviewed_preflight_sha256 and not (args.trainpath_recovery
            or (args.phase == "cost-calibration" and args.experiment_id == "EXP-055"
                and args.models == "transigen")):
        parser.error("--reviewed-preflight-sha256 is reserved for registered EXP-055 recovery/calibration")
    config_path = args.config or (ROOT / "configs/experiments/EXP-055-representative-models.yaml"
                                  if args.experiment_id == "EXP-055"
                                  else ROOT / f"configs/experiments/{EXP059_TRANSIGEN_LR_STABILITY_CONFIG}"
                                  if args.experiment_id == "EXP-059"
                                  else ROOT / "configs/experiments/EXP-051.yaml")
    configure_experiment(args.experiment_id, args.output_root)
    if args.phase in {"fit", "entry-smoke", "layer-replay", "latent-replay", "cost-calibration", "cpu-contract-check", "checkpoint-replay"} and args.run_id is None:
        parser.error("--run-id is required for fit, entry-smoke, layer/latent replay, or cost-calibration")
    if args.phase in {"fit", "entry-smoke", "layer-replay", "latent-replay", "cost-calibration", "cpu-contract-check", "checkpoint-replay"} and args.attempt_id is None:
        parser.error("--attempt-id is required for fit, entry-smoke, layer/latent replay, or cost-calibration")
    if args.run_id is None:
        args.run_id = DEFAULT_RUN_ID
    configure_run(args.run_id)
    cfg, config_sha = load_config(config_path, args.experiment_id)
    if args.phase in {"fit", "entry-smoke", "layer-replay", "latent-replay", "cost-calibration", "cpu-contract-check", "checkpoint-replay"}:
        source_run_id = args.source_run_id or cfg.get("source_run_id") or DEFAULT_SOURCE_RUN_ID
        if (args.phase == "cost-calibration" and args.experiment_id == "EXP-055"
                and args.models == "transigen"):
            if (args.run_id != EXP055_TRANSIGEN_LEAN_COST_RUN_ID
                    or source_run_id != EXP055_TRANSIGEN_COST_SOURCE_RUN_ID):
                parser.error("TranSiGen cost calibration requires the registered run and v12 checkpoint source id")
            # The CLI source id identifies the calibration checkpoint. Training input
            # keys and values still come from the frozen EXP-051 response-blind source.
            configure_source_run(DEFAULT_SOURCE_RUN_ID, "EXP-051")
        else:
            configure_source_run(source_run_id, cfg.get("source_experiment_id", "EXP-051"))
    elif args.source_run_id is not None:
        parser.error("--source-run-id is only valid with fit, entry-smoke, or cost-calibration")
    else:
        configure_source_run(args.run_id, args.experiment_id)
    if args.phase not in {"fit", "entry-smoke", "layer-replay", "latent-replay", "cost-calibration", "cpu-contract-check", "checkpoint-replay"} and args.attempt_id is not None:
        parser.error("--attempt-id is only valid with fit, entry-smoke, layer/latent replay, or cost-calibration")
    if args.phase not in {"fit", "entry-smoke", "layer-replay", "latent-replay", "cost-calibration", "cpu-contract-check", "checkpoint-replay"} and args.models != "both":
        parser.error("--models is only valid for fit, entry-smoke, or read-only replay")
    if args.phase == "metadata":
        metadata_phase(cfg, config_sha)
    elif args.phase == "axis":
        axis_phase(config_sha)
    elif args.phase == "values":
        values_phase(cfg, config_sha)
    elif args.phase == "feature_audit":
        feature_audit_phase()
    elif args.phase == "entry-smoke":
        validate_attempt_id(args.attempt_id)
        try:
            output = run_exp055_trainpath_entry_smoke_cpu(cfg, config_sha, args.attempt_id)
            print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
        except BaseException as error:
            _close_all_unhandled_attempts(args.run_id, error)
            raise
    elif args.phase == "cpu-contract-check":
        validate_attempt_id(args.attempt_id)
        try:
            output = run_exp055_transigen_cpu_contract_check(
                cfg, config_sha, args.attempt_id, args.device or "")
            print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
        except BaseException as error:
            _close_all_unhandled_attempts(args.run_id, error)
            raise
    elif args.phase == "layer-replay":
        validate_attempt_id(args.attempt_id)
        try:
            output = run_exp055_trainpath_batch_replay_cpu(cfg, config_sha, args.attempt_id)
            print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
        except BaseException as error:
            _close_all_unhandled_attempts(args.run_id, error)
            raise
    elif args.phase == "latent-replay":
        validate_attempt_id(args.attempt_id)
        try:
            output = run_exp055_trainpath_latent_replay_cpu(cfg, config_sha, args.attempt_id)
            print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
        except BaseException as error:
            _close_all_unhandled_attempts(args.run_id, error)
            raise
    elif args.phase == "checkpoint-replay":
        validate_attempt_id(args.attempt_id)
        try:
            output = run_exp055_transigen_checkpoint8_replay_cpu(
                cfg, config_sha, args.attempt_id, args.device or "", args.resume_from_checkpoint)
            print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
        except BaseException as error:
            _close_all_unhandled_attempts(args.run_id, error)
            raise
    elif args.phase == "fit":
        validate_attempt_id(args.attempt_id)
        if args.run_id == EXP055_TRANSIGEN_CPU_REPRESENTATIVE_RUN_ID:
            try:
                output = fit_phase_transigen_cpu_representative(
                    cfg, config_sha, args.attempt_id, args.device or "")
                print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
            except BaseException as error:
                _close_all_unhandled_attempts(args.run_id, error)
                raise
            return
        if args.run_id == EXP055_TRANSIGEN_CPU_RESUME_RUN_ID:
            try:
                output = fit_phase_transigen_cpu_representative(
                    cfg, config_sha, args.attempt_id, args.device or "",
                    resume_from_checkpoint=args.resume_from_checkpoint)
                print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
            except BaseException as error:
                _close_all_unhandled_attempts(args.run_id, error)
                raise
            return
        if args.run_id == EXP055_TRANSIGEN_CPU_NORM_RECOVERY_RUN_ID:
            try:
                output = fit_phase_transigen_cpu_representative(
                    cfg, config_sha, args.attempt_id, args.device or "",
                    resume_from_checkpoint=args.resume_from_checkpoint)
                print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
            except BaseException as error:
                _close_all_unhandled_attempts(args.run_id, error)
                raise
            return
        if args.run_id == EXP059_TRANSIGEN_LR_STABILITY_RUN_ID:
            try:
                output = fit_phase_transigen_cpu_representative(
                    cfg, config_sha, args.attempt_id, args.device or "")
                print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
            except BaseException as error:
                _close_all_unhandled_attempts(args.run_id, error)
                raise
            return
        if (args.experiment_id == "EXP-055" and args.attempt_id == "fit_attempt_007"
                and args.models in {"transigen", "both"}):
            raise RuntimeError(
                "TranSiGen attempt007 is closed after its one CPU main-delta preflight failure; "
                "the probe and GPU fit cannot be retried")
        try:
            if args.trainpath_recovery:
                if args.preflight_only:
                    output = preflight_exp055_trainpath_recovery(cfg, config_sha, args.attempt_id)
                    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
                    return
                output = fit_phase_transigen_trainpath_recovery(
                    cfg, config_sha, args.attempt_id, args.reviewed_preflight_sha256)
                print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
            elif args.fit_kind == "representative":
                if args.experiment_id == "EXP-055" and args.models == "transigen":
                    output = fit_phase_transigen_only(cfg, config_sha, args.attempt_id,
                                                      preflight_only=args.preflight_only)
                    if args.preflight_only:
                        print(json.dumps(output, ensure_ascii=False, indent=2))
                        return
                elif args.experiment_id == "EXP-055" and args.models == "xpert":
                    if args.attempt_id in {"fit_attempt_008", "fit_attempt_009"}:
                        output = fit_phase_xpert_full_attempt008(
                            cfg, config_sha, args.attempt_id,
                            preflight_only=args.preflight_only)
                        if args.preflight_only:
                            print(json.dumps(output, ensure_ascii=False, indent=2))
                            return
                    else:
                        fit_phase_xpert_only(cfg, config_sha, args.attempt_id)
                else:
                    fit_phase(cfg, config_sha, args.attempt_id)
            elif args.models == "xpert":
                fit_phase_xpert_only(cfg, config_sha, args.attempt_id)
            elif args.models == "transigen":
                diagnostic_path = ROOT / "scripts/modeling/diagnose_exp051_transigen_train_state.py"
                spec = importlib.util.spec_from_file_location("exp051_transigen_train_state_diagnostic", diagnostic_path)
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"cannot load the bounded TranSiGen diagnostic entry: {diagnostic_path}")
                diagnostic = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(diagnostic)
                diagnostic.run_diagnostic(sys.modules[__name__], cfg, config_sha, args.attempt_id)
            else:
                fit_phase(cfg, config_sha, args.attempt_id)
        except BaseException as error:
            _close_all_unhandled_attempts(args.run_id, error)
            raise
    elif args.phase == "cost-calibration":
        validate_attempt_id(args.attempt_id)
        if args.experiment_id != "EXP-055" or args.fit_kind != "representative":
            parser.error("--phase cost-calibration requires EXP-055 --fit-kind representative")
        if args.models == "transigen":
            if args.run_id != EXP055_TRANSIGEN_LEAN_COST_RUN_ID or args.attempt_id != "fit_attempt_001":
                parser.error("TranSiGen cost calibration requires its unique registered run/attempt")
            if args.preflight_only:
                output = preflight_exp055_transigen_lean_cost_calibration(cfg, config_sha, args.attempt_id)
                print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
                return
            if not args.reviewed_preflight_sha256:
                parser.error("TranSiGen calibration requires --reviewed-preflight-sha256 after root review")
            try:
                if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
                    raise RuntimeError("EXP-055 calibration requires only the approved A5000 UUID")
                calibration = run_exp055_transigen_lean_cost_calibration(
                    cfg, config_sha, args.attempt_id, args.reviewed_preflight_sha256)
                print(json.dumps(calibration, ensure_ascii=False, indent=2), flush=True)
                if calibration.get("cost_gate", {}).get("status") != "PASS_FIXED300_WITHIN_RECOVERY_CAP":
                    return
                # The approved chain may continue only after the measured fixed-300
                # estimate passes; formal fitting creates a new run and fresh init.
                configure_run(EXP055_TRANSIGEN_FIXED300_LEAN_RUN_ID)
                configure_source_run(DEFAULT_SOURCE_RUN_ID, "EXP-051")
                formal_preflight = preflight_exp055_trainpath_recovery(
                    cfg, config_sha, "fit_attempt_001")
                formal_preflight_sha = str(formal_preflight["receipt_sha256"])
                print(json.dumps({"stage": "EXP055_LEAN_FIXED300_AUTOCHAIN_PREFLIGHT",
                    "run_id": RUN_ID, "preflight_receipt_path": formal_preflight["receipt_path"],
                    "preflight_receipt_sha256": formal_preflight_sha,
                    "same_owner_gpu_chain": True,
                    "calibration_cost_gate": calibration["cost_gate"]["status"]},
                    ensure_ascii=False, indent=2), flush=True)
                formal = fit_phase_transigen_trainpath_recovery(
                    cfg, config_sha, "fit_attempt_001", formal_preflight_sha,
                    same_owner_gpu_chain=True)
                print(json.dumps(formal, ensure_ascii=False, indent=2), flush=True)
            except BaseException as error:
                _close_all_unhandled_attempts(RUN_ID, error)
                raise
            return
        if args.models != "xpert":
            parser.error("--phase cost-calibration is registered only for EXP-055 XPert or lean TranSiGen")
        if args.preflight_only:
            print(json.dumps(_exp055_cost_calibration_cpu_preflight(cfg, config_sha), ensure_ascii=False, indent=2))
            return
        try:
            if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
                raise RuntimeError("EXP-055 calibration requires CUDA_VISIBLE_DEVICES set to the one approved A5000 UUID")
            fit_phase_xpert_only(cfg, config_sha, args.attempt_id, cost_calibration=True)
        except BaseException as error:
            _close_all_unhandled_attempts(args.run_id, error)
            raise
    else:
        raise ValueError(f"unsupported phase {args.phase}")


if __name__ == "__main__":
    main()
