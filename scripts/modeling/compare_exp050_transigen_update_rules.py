"""One authorized train-only TranSiGen optimizer-rule A/B on frozen 512 keys.

No validation, outer, locked, or PDO inputs are opened by this runner.  The
freeze phase binds the selected train values, selected ECFP rows, code, and the
exact key order before any CUDA work.  The run phase first smoke-tests one
64-condition step per arm, then executes one eight-batch pass per arm only when
the measured smoke-based total estimate (including a 20% margin) is <= 0.9 h.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20261001_transigen_update_rule_compare_v2"
OUT = ROOT / "artifacts/experiments/EXP-050" / RUN_ID
AUDIT_DIR = ROOT / "artifacts/experiments/EXP-050/20261001_transigen_official_contract_audit_v1"
PREPARED = ROOT / "artifacts/experiments/EXP-037/20260928_lincs_scale_16_121/prepared"
LEDGER = ROOT / "artifacts/experiments/EXP-050/gpu_ledger.jsonl"
LEDGER_LOCK = ROOT / "artifacts/experiments/EXP-050/gpu_ledger.lock"
OFFICIAL_SRC = Path("/mnt/d/Code/TranSiGen/参考代码/TranSiGen/src")
OFFICIAL_MODEL = OFFICIAL_SRC / "model.py"
AUDIT_SCRIPT = ROOT / "scripts/data/audit_exp050_transigen_official_contract.py"
FEATURE_BUILDER = ROOT / "artifacts/experiments/EXP-037/20260928_lincs_scale_16_121/wrapper_transigen121/code_snapshot/exp037_lincs_scale_probe.py"
UPSTREAM_ADAPTER = Path("/mnt/c/Users/dell/.codex/worktrees/bea4/DrugScreenLab/scripts/modeling/exp050_transigen_fit_diagnostic.py")
KEYS = AUDIT_DIR / "selected_train512_condition_keys.tsv"
VALUE_NPZ = AUDIT_DIR / "selected_train512_control_delta_treated.npz"
KEY_RECEIPT = AUDIT_DIR / "response_blind_key_freeze_receipt.json"
VALUE_RECEIPT = AUDIT_DIR / "train512_value_contract_receipt.json"
PREP_MANIFEST = PREPARED / "manifest.json"
FEATURE_MATRIX = PREPARED / "ecfp4.npy"
FEATURE_CACHE = OUT / "selected_train512_ecfp4.npy"
PROTOCOL = OUT / "protocol.json"

EXPECTED_LEDGER_START_GPU_HOURS = 0.5825016765391673
EXPECTED_OFFICIAL_MODEL_SHA256 = "a96706b11b29c8be3122640e9a05a54a1e2210039f3fd8a30b0f664d62cf7c0f"
PACKAGE_CAP_GPU_HOURS = 48.0
ATTEMPT_RESERVATION_GPU_HOURS = 1.0
SMOKE_ESTIMATE_GATE_GPU_HOURS = 0.9
ABSOLUTE_ATTEMPT_CAP_SECONDS = 3600.0
APPROVED_GPU_UUID = "GPU-9c366c2e-bfa5-c7b3-b16a-1117c40e6871"
ATTEMPT_ID = "EXP050-TRANSIGEN-UPDATE-RULE-COMPARE-20261001-002"

N_KEYS = 512
N_GENES = 978
N_FEATURES = 2048
BATCH_SIZE = 64
N_BATCHES = 8
N_ARMS = 2
N_FULL_STEPS = N_BATCHES * N_ARMS
MODEL_SEED = 20261003
STEP_SEED_PREFIX = 20262003
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BETA = 0.1
DROPOUT = 0.1
GRAD_CLIP_NORM = 5.0
LATENT_DIM = 100
ENCODER_HIDDEN = [1200]
DECODER_HIDDEN = [800]
FEATURE_EMBED_DIM = [400]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode("utf-8"))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _ledger_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(LEDGER.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"EXP-050 ledger JSON invalid at line {line_no}") from exc
        if row.get("experiment_id") != "EXP-050":
            raise RuntimeError(f"unexpected experiment ID in EXP-050 ledger line {line_no}")
        rows.append(row)
    return rows


def _ledger_state(rows: list[dict[str, Any]]) -> dict[str, Any]:
    starts: dict[str, dict[str, Any]] = {}
    closes: dict[str, dict[str, Any]] = {}
    for row in rows:
        event = row.get("event")
        attempt = str(row.get("attempt_id", ""))
        if event == "attempt_started":
            if not attempt or attempt in starts:
                raise RuntimeError("EXP-050 ledger has a missing or duplicate attempt start")
            starts[attempt] = row
        elif event in {"attempt_finished", "attempt_reconciled"}:
            if not attempt or attempt not in starts or attempt in closes:
                raise RuntimeError("EXP-050 ledger has an orphan or duplicate attempt close")
            charge = float(row.get("gpu_hours_charged", math.nan))
            if not math.isfinite(charge) or charge < 0:
                raise RuntimeError(f"EXP-050 ledger has invalid GPU charge for {attempt}")
            closes[attempt] = row
    open_ids = sorted(starts.keys() - closes.keys())
    reserve = sum(float(starts[key].get("reserved_gpu_hours", math.nan)) for key in open_ids)
    charged = sum(float(row["gpu_hours_charged"]) for row in closes.values())
    if not math.isfinite(reserve) or reserve < 0:
        raise RuntimeError("EXP-050 ledger has an invalid active reservation")
    return {"settled_gpu_hours": charged, "reserved_gpu_hours": reserve,
            "open_attempt_ids": open_ids, "attempt_ids": sorted(starts)}


def _append_event_locked(event: str, **fields: Any) -> dict[str, Any]:
    record = {"experiment_id": "EXP-050", "run_id": RUN_ID, "event": event,
              "timestamp_utc": utc_now(), **fields}
    data = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(LEDGER, os.O_WRONLY | os.O_APPEND)
    try:
        if os.write(fd, data) != len(data):
            raise OSError("short EXP-050 ledger append")
        os.fsync(fd)
    finally:
        os.close(fd)
    return record


def _ledger_snapshot() -> dict[str, Any]:
    if not LEDGER.is_file() or not LEDGER_LOCK.is_file():
        raise RuntimeError("EXP-050 GPU ledger or lock is missing")
    with LEDGER_LOCK.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            rows = _ledger_rows()
            state = _ledger_state(rows)
            return {**state, "ledger_sha256": sha256_file(LEDGER)}
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _reserve_attempt(protocol_sha: str, gpu_snapshot: dict[str, Any]) -> dict[str, Any]:
    with LEDGER_LOCK.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            rows = _ledger_rows()
            state = _ledger_state(rows)
            if abs(state["settled_gpu_hours"] - EXPECTED_LEDGER_START_GPU_HOURS) > 1e-9:
                raise RuntimeError(f"EXP-050 budget baseline changed: {state['settled_gpu_hours']:.12f}")
            if state["reserved_gpu_hours"] != 0 or state["open_attempt_ids"]:
                raise RuntimeError(f"another EXP-050 GPU attempt is active: {state['open_attempt_ids']}")
            if ATTEMPT_ID in state["attempt_ids"]:
                raise RuntimeError(f"attempt ID already exists; this run cannot be retried: {ATTEMPT_ID}")
            if state["settled_gpu_hours"] + ATTEMPT_RESERVATION_GPU_HOURS > PACKAGE_CAP_GPU_HOURS:
                raise RuntimeError("one-hour attempt reservation exceeds the approved 48 GPU-hour cap")
            start = _append_event_locked(
                "attempt_started", attempt_id=ATTEMPT_ID, operation="512-train-key TranSiGen optimizer update-rule A/B",
                gpu_uuid=APPROVED_GPU_UUID, gpu_name="NVIDIA RTX A5000", gpu_count=1,
                pid=os.getpid(), reserved_gpu_hours=ATTEMPT_RESERVATION_GPU_HOURS,
                estimated_gpu_hours=None, package_cap_gpu_hours=PACKAGE_CAP_GPU_HOURS,
                baseline_settled_gpu_hours=state["settled_gpu_hours"],
                baseline_reserved_gpu_hours=state["reserved_gpu_hours"],
                preflight_gpu_snapshot=gpu_snapshot, protocol_sha256=protocol_sha,
                no_retry=True,
            )
            return {"before": state, "start": start}
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _close_attempt(start_monotonic: float, status: str, error: str | None,
                   gate: dict[str, Any] | None = None) -> dict[str, Any]:
    elapsed = max(0.0, time.perf_counter() - start_monotonic)
    charged = elapsed / 3600.0
    fields = {"attempt_id": ATTEMPT_ID, "status": status, "error": error,
              "elapsed_seconds": elapsed, "gpu_hours_charged": charged,
              "reserved_gpu_hours": 0.0, "gpu_uuid": APPROVED_GPU_UUID,
              "gpu_count": 1, "cost_gate": gate, "no_retry": True}
    with LEDGER_LOCK.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            rows = _ledger_rows()
            state = _ledger_state(rows)
            if ATTEMPT_ID not in state["open_attempt_ids"]:
                raise RuntimeError("refusing to close a missing or already closed ledger attempt")
            closed = _append_event_locked("attempt_finished", **fields)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return closed


def _load_bound_inputs() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if not PROTOCOL.is_file():
        raise FileNotFoundError("protocol must be frozen before any GPU execution")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    frozen_digest = protocol.get("protocol_sha256")
    digest_input = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if canonical_hash(digest_input) != frozen_digest:
        raise RuntimeError("frozen protocol digest mismatch")
    code_expected = protocol["hashes"]["runner_sha256"]
    if sha256_file(Path(__file__).resolve()) != code_expected:
        raise RuntimeError("runner code changed after protocol freeze")
    if sha256_file(OFFICIAL_MODEL) != protocol["hashes"]["official_model_py_sha256"]:
        raise RuntimeError("official model.py changed after protocol freeze")
    if sha256_file(AUDIT_SCRIPT) != protocol["hashes"]["audit_script_sha256"]:
        raise RuntimeError("key/value audit script changed after protocol freeze")
    if sha256_file(FEATURE_BUILDER) != protocol["hashes"]["feature_builder_sha256"]:
        raise RuntimeError("feature builder code changed after protocol freeze")
    if sha256_file(KEYS) != protocol["hashes"]["selected_key_tsv_sha256"]:
        raise RuntimeError("frozen selected key file changed")
    if sha256_file(VALUE_NPZ) != protocol["hashes"]["selected_value_npz_sha256"]:
        raise RuntimeError("frozen 512-value archive changed")
    if sha256_file(FEATURE_CACHE) != protocol["hashes"]["selected_ecfp4_npy_sha256"]:
        raise RuntimeError("frozen selected ECFP feature cache changed")

    keys = pd.read_csv(KEYS, sep="\t", dtype={"condition_id": str, "cell_id": str, "pert_id": str})
    if len(keys) != N_KEYS or not np.array_equal(keys["selection_rank"].to_numpy(), np.arange(1, N_KEYS + 1)):
        raise RuntimeError("selected train key table no longer has its frozen 512-row order")
    with np.load(VALUE_NPZ, allow_pickle=False) as values:
        if not np.array_equal(values["condition_id"].astype(str), keys["condition_id"].astype(str).to_numpy()):
            raise RuntimeError("train values condition identities differ from frozen keys")
        if not np.array_equal(values["source_row"], keys["source_row"].to_numpy(dtype=np.int64)):
            raise RuntimeError("train values source-row identity differs from frozen keys")
        control = values["control"].astype(np.float32, copy=True)
        delta = values["delta"].astype(np.float32, copy=True)
        treated = values["treated"].astype(np.float32, copy=True)
        gene_id = values["gene_id"].astype(str)
    if control.shape != (N_KEYS, N_GENES) or delta.shape != control.shape or treated.shape != control.shape:
        raise RuntimeError("frozen train expression arrays have an unexpected shape")
    if not np.array_equal(treated, control + delta):
        raise RuntimeError("frozen train treated expression no longer equals control plus delta")
    if not np.isfinite(control).all() or not np.isfinite(delta).all() or not np.isfinite(treated).all():
        raise RuntimeError("frozen train values contain non-finite entries")
    ecfp = np.load(FEATURE_CACHE, allow_pickle=False)
    if ecfp.shape != (N_KEYS, N_FEATURES) or not np.isfinite(ecfp).all():
        raise RuntimeError(f"frozen selected ECFP array has invalid shape/values: {ecfp.shape}")
    if not np.isin(ecfp, [0, 1]).all():
        raise RuntimeError("selected ECFP4 features are not binary")
    if len(gene_id) != N_GENES:
        raise RuntimeError("frozen training gene axis is not exactly 978 genes")
    return keys, control, delta, ecfp.astype(np.float32, copy=False), {
        "treated": treated, "gene_id": gene_id, "protocol": protocol,
    }


def _gpu_inventory() -> dict[str, Any]:
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    devices = []
    for line in query:
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError(f"unrecognized nvidia-smi device row: {line}")
        devices.append({"uuid": fields[0], "name": fields[1],
                        "utilization_percent": int(fields[2]), "memory_used_mib": int(fields[3])})
    selected = next((item for item in devices if item["uuid"] == APPROVED_GPU_UUID), None)
    if selected is None or "A5000" not in selected["name"]:
        raise RuntimeError("approved RTX A5000 UUID is not available")
    if selected["utilization_percent"] > 5 or selected["memory_used_mib"] > 512:
        raise RuntimeError(f"approved RTX A5000 is occupied: {selected}")
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
         "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    if any(line.strip().startswith(APPROVED_GPU_UUID) for line in apps):
        raise RuntimeError("a compute application is already attached to the approved RTX A5000")
    ps = subprocess.check_output(["ps", "-eo", "pid=,ppid=,etimes=,args="], text=True)
    current_pid = os.getpid()
    process_rows: dict[int, tuple[int, str]] = {}
    for line in ps.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) != 4:
            continue
        try:
            pid, parent_pid = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        process_rows[pid] = (parent_pid, fields[3])
    if current_pid not in process_rows:
        raise RuntimeError(f"current Python PID is missing from process inventory: {current_pid}")
    # Exclude this runner and its launch chain (including `conda run` and the
    # WSL shell wrapper), while continuing to scan unrelated processes for
    # actual TranSiGen/XPert training commands.
    own_process_chain = {current_pid}
    ancestor_pid = current_pid
    while ancestor_pid in process_rows:
        parent_pid = process_rows[ancestor_pid][0]
        if parent_pid <= 1 or parent_pid in own_process_chain:
            break
        own_process_chain.add(parent_pid)
        ancestor_pid = parent_pid
    training_processes = []
    for pid, (parent_pid, command_line) in process_rows.items():
        if (pid not in own_process_chain
                and any(token in command_line for token in (
                    "compare_exp050_transigen_update_rules.py",
                    "exp050_transigen_fit_diagnostic.py",
                    "exp050_xpert_independent_response_fit.py",
                ))):
            training_processes.append(
                f"pid={pid} ppid={parent_pid} command={command_line}"
            )
    if training_processes:
        raise RuntimeError(f"another relevant training process is active: {training_processes}")
    return {"devices": devices, "selected_device": selected,
            "selected_compute_apps": apps, "other_relevant_training_processes": training_processes}


def _freeze() -> dict[str, Any]:
    if OUT.exists() or PROTOCOL.exists() or FEATURE_CACHE.exists():
        raise FileExistsError("run directory/protocol/selected-feature cache already exists; do not overwrite or resume a frozen run")
    key_receipt = json.loads(KEY_RECEIPT.read_text(encoding="utf-8"))
    value_receipt = json.loads(VALUE_RECEIPT.read_text(encoding="utf-8"))
    key_receipt_hash = sha256_file(KEY_RECEIPT)
    value_receipt_hash = sha256_file(VALUE_RECEIPT)
    if key_receipt.get("run_id") != "20261001_transigen_official_contract_audit_v1":
        raise RuntimeError("expected audit package key receipt is missing")
    if value_receipt.get("run_id") != key_receipt.get("run_id"):
        raise RuntimeError("key and value contract receipts do not belong to the same audit package")
    if key_receipt.get("selected_condition_count") != N_KEYS or value_receipt.get("selected_train_response_rows_read") != N_KEYS:
        raise RuntimeError("the existing train-only audit package does not prove exactly 512 selected rows")
    if value_receipt.get("treated_equals_control_plus_delta_max_abs_error") != 0.0 or not value_receipt.get("value_round_trip_exact"):
        raise RuntimeError("the frozen value contract receipt failed its identity/reconstruction gates")
    if value_receipt.get("validation_or_locked_response_arrays_opened") is not False:
        raise RuntimeError("audit value receipt does not confirm the validation/locked boundary")
    if key_receipt.get("selected_key_sha256") != sha256_file(KEYS):
        raise RuntimeError("audit key receipt does not match the frozen key table")
    keys = pd.read_csv(KEYS, sep="\t", dtype={"condition_id": str, "cell_id": str, "pert_id": str})
    if len(keys) != N_KEYS or keys["condition_id"].duplicated().any():
        raise RuntimeError("frozen train key table is not 512 unique condition identities")
    if list(keys["selection_rank"].astype(int)) != list(range(1, N_KEYS + 1)):
        raise RuntimeError("frozen train key ordering is invalid")

    with np.load(VALUE_NPZ, allow_pickle=False) as values:
        if not np.array_equal(values["condition_id"].astype(str), keys["condition_id"].astype(str).to_numpy()):
            raise RuntimeError("512 value archive identity order differs from the frozen key table")
        if not np.array_equal(values["source_row"], keys["source_row"].to_numpy(dtype=np.int64)):
            raise RuntimeError("512 value archive source rows differ from the frozen key table")
        if values["control"].shape != (N_KEYS, N_GENES) or values["delta"].shape != (N_KEYS, N_GENES):
            raise RuntimeError("512 value archive does not match the registered 978-gene contract")
        if not np.array_equal(values["treated"], values["control"] + values["delta"]):
            raise RuntimeError("treated != control + delta in the frozen value archive")
        gene_ids = values["gene_id"].astype(str)
        if len(gene_ids) != N_GENES or len(set(gene_ids.tolist())) != N_GENES:
            raise RuntimeError("512 value archive gene-axis keys are not 978 unique IDs")

    prep_manifest = json.loads(PREP_MANIFEST.read_text(encoding="utf-8"))
    registered_ecfp_sha = prep_manifest.get("files_sha256", {}).get("ecfp4.npy")
    if not registered_ecfp_sha:
        raise RuntimeError("prepared manifest lacks the registered ECFP4 whole-file hash")
    row_index = keys["prepared_row_index"].to_numpy(dtype=np.int64)
    if len(np.unique(row_index)) != N_KEYS or row_index.min() < 0:
        raise RuntimeError("selected feature row indices are duplicated or invalid")
    feature_map = np.load(FEATURE_MATRIX, mmap_mode="r", allow_pickle=False)
    if feature_map.ndim != 2 or feature_map.shape[1] != N_FEATURES or row_index.max() >= feature_map.shape[0]:
        raise RuntimeError(f"prepared ECFP4 array dimensions do not bind the frozen 512 rows: {feature_map.shape}")
    ecfp = np.asarray(feature_map[row_index], dtype=np.float32)
    if ecfp.shape != (N_KEYS, N_FEATURES) or not np.isfinite(ecfp).all() or not np.isin(ecfp, [0, 1]).all():
        raise RuntimeError("selected 512 ECFP4 rows have an invalid shape, values, or binary contract")
    import torch
    source_hash = sha256_file(OFFICIAL_MODEL)
    if source_hash != EXPECTED_OFFICIAL_MODEL_SHA256:
        raise RuntimeError(f"official model.py SHA-256 differs from the audited source: {source_hash}")
    initial_ledger = _ledger_snapshot()
    if abs(initial_ledger["settled_gpu_hours"] - EXPECTED_LEDGER_START_GPU_HOURS) > 1e-9:
        raise RuntimeError("current 48-hour package ledger differs from the supervisor-provided baseline")
    if initial_ledger["reserved_gpu_hours"] != 0 or initial_ledger["open_attempt_ids"]:
        raise RuntimeError("there is an active EXP-050 GPU reservation; protocol may not be executed")
    OUT.mkdir(parents=True, exist_ok=False)
    with FEATURE_CACHE.open("wb") as stream:
        np.save(stream, ecfp, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())

    audit_code_hash = sha256_file(AUDIT_SCRIPT)
    adapter_hash = sha256_file(UPSTREAM_ADAPTER) if UPSTREAM_ADAPTER.is_file() else None

    protocol: dict[str, Any] = {
        "experiment_id": "EXP-050",
        "run_id": RUN_ID,
        "status": "FROZEN_BEFORE_GPU_SMOKE",
        "created_at_utc": utc_now(),
        "authorization": "Manager-authorized one-time train-only update-rule comparison; no retry; <=1.0 total GPU hour including smoke; training proceeds only if smoke + projected full 16-step compare + 20% margin <=0.9 GPU hours.",
        "data_scope": {
            "only_condition_source": str(KEYS.relative_to(ROOT)).replace("\\", "/"),
            "only_response_source": str(VALUE_NPZ.relative_to(ROOT)).replace("\\", "/"),
            "only_feature_source": str(FEATURE_CACHE.relative_to(ROOT)).replace("\\", "/"),
            "condition_count": N_KEYS,
            "genes": N_GENES,
            "features": N_FEATURES,
            "validation_dev_outer_locked_or_pdo_inputs_opened": False,
            "selection_order": "selection_rank 1..512; fixed contiguous batches of 64; no shuffle",
        },
        "arms": {
            "current_adapter": {"objective": "official six-term sum / (actual_batch_size * 978)",
                                "gradient_clip_global_norm": GRAD_CLIP_NORM},
            "official_update_rule": {"objective": "official six-term raw sum",
                                      "gradient_clip_global_norm": None},
            "shared": "Same official model class, 512 train keys/targets, selected ECFP4 features, scratch state, batch order, per-step random stream, Adam settings, and model architecture; only objective scalar and gradient clipping differ.",
        },
        "model": {"official_model_source": str(OFFICIAL_MODEL), "genes": N_GENES,
                  "latent_dim": LATENT_DIM, "encoder_hidden": ENCODER_HIDDEN,
                  "decoder_hidden": DECODER_HIDDEN, "feature_dim": N_FEATURES,
                  "feature_embed_dim": FEATURE_EMBED_DIM, "beta": BETA,
                  "dropout": DROPOUT, "initialization": "shared scratch initialization; no pretrained weights"},
        "optimizer": {"name": "torch.optim.Adam", "learning_rate": LEARNING_RATE,
                      "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
                      "betas": [0.9, 0.999], "epsilon": 1e-8,
                      "precision": "float32; autocast disabled"},
        "randomness": {"model_seed": MODEL_SEED,
                       "step_seed_rule": "Reset Python, NumPy, CPU Torch and CUDA Torch RNG to STEP_SEED_PREFIX + batch_index immediately before each matched A/B update; smoke uses batch_index 0 and the full pass uses 0..7.",
                       "step_seed_prefix": STEP_SEED_PREFIX,
                       "deterministic_cudnn": True, "benchmark_cudnn": False},
        "gate": {"smoke_steps": "one first-batch step for each arm, starting from one common scratch state",
                 "full_steps": N_FULL_STEPS,
                  "projected_formula": "(measured reservation-to-smoke-complete seconds + 16 * slower measured one-step time + measured smoke non-step initialization overhead + 16 * mean measured JSONL-write time) * 1.20 / 3600",
                 "continue_if_gpu_hours_at_most": SMOKE_ESTIMATE_GATE_GPU_HOURS,
                 "absolute_attempt_cap_gpu_hours": 1.0,
                 "no_retry": True},
        "budget": {"package_cap_gpu_hours": PACKAGE_CAP_GPU_HOURS,
                   "settled_before_run_gpu_hours": initial_ledger["settled_gpu_hours"],
                   "reserved_before_run_gpu_hours": initial_ledger["reserved_gpu_hours"],
                   "attempt_reservation_gpu_hours": ATTEMPT_RESERVATION_GPU_HOURS,
                   "ledger_sha256_before_run": initial_ledger["ledger_sha256"]},
        "device": {"approved_gpu_uuid": APPROVED_GPU_UUID,
                   "selected_before_gpu_freeze": "NVIDIA RTX A5000; exact utilization and memory are rechecked immediately before reservation"},
        "hashes": {
            "selected_key_tsv_sha256": sha256_file(KEYS),
            "selected_key_receipt_sha256": key_receipt_hash,
            "selected_value_npz_sha256": sha256_file(VALUE_NPZ),
            "selected_value_receipt_sha256": value_receipt_hash,
            "selected_ecfp4_npy_sha256": sha256_file(FEATURE_CACHE),
            "prepared_ecfp4_registered_file_sha256": registered_ecfp_sha,
            "prepared_manifest_sha256": sha256_file(PREP_MANIFEST),
            "official_model_py_sha256": source_hash,
            "audit_script_sha256": audit_code_hash,
            "feature_builder_sha256": sha256_file(FEATURE_BUILDER),
            "adapter_script_sha256": adapter_hash,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "selected_gene_ids_sha256": sha256_bytes("\n".join(gene_ids.tolist()).encode("utf-8")),
            "initial_state_sha256": None,
        },
        "software": {"python": sys.version.split()[0], "numpy": np.__version__,
                      "pandas": pd.__version__, "torch": torch.__version__},
        "outputs": ["selected_train512_ecfp4.npy", "smoke_steps.jsonl",
                    "smoke_receipt.json", "training_steps.jsonl", "run_receipt.json",
                    "run_manifest.json"],
    }
    protocol["protocol_sha256"] = canonical_hash(protocol)
    atomic_json(PROTOCOL, protocol)
    freeze = {"run_id": RUN_ID, "status": protocol["status"],
              "protocol_sha256": protocol["protocol_sha256"],
              "selected_key_sha256": protocol["hashes"]["selected_key_tsv_sha256"],
              "selected_values_sha256": protocol["hashes"]["selected_value_npz_sha256"],
              "selected_features_sha256": protocol["hashes"]["selected_ecfp4_npy_sha256"],
              "runner_sha256": protocol["hashes"]["runner_sha256"],
              "official_model_py_sha256": source_hash,
              "ledger_settled_gpu_hours": initial_ledger["settled_gpu_hours"],
              "ledger_reserved_gpu_hours": initial_ledger["reserved_gpu_hours"],
              "protocol_file_sha256": sha256_file(PROTOCOL)}
    atomic_json(OUT / "freeze_receipt.json", freeze)
    return freeze


def _set_matched_rng(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_model_class() -> tuple[Any, Any]:
    sys.path.insert(0, str(OFFICIAL_SRC))
    spec = importlib.util.spec_from_file_location("transigen_update_compare_official", OFFICIAL_MODEL)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load official TranSiGen model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TranSiGen, module


def _state_digest(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _global_gradient_norm(torch: Any, model: Any) -> float:
    grads = [p.grad.detach().float() for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    total_sq = torch.zeros((), device=grads[0].device, dtype=torch.float64)
    for grad in grads:
        total_sq += grad.double().pow(2).sum()
    return float(torch.sqrt(total_sq).item())


def _parameter_update_norm(torch: Any, before: list[Any], model: Any) -> float:
    total_sq = torch.zeros((), device=before[0].device, dtype=torch.float64)
    with torch.no_grad():
        for old, parameter in zip(before, model.parameters()):
            difference = parameter.detach().double() - old.double()
            total_sq += difference.pow(2).sum()
    return float(torch.sqrt(total_sq).item())


def _new_model(torch: Any, model_class: Any, state: dict[str, Any], device: Any) -> Any:
    _set_matched_rng(torch, MODEL_SEED)
    model = model_class(N_GENES, LATENT_DIM, ENCODER_HIDDEN, DECODER_HIDDEN,
                        N_FEATURES, FEATURE_EMBED_DIM, device=device, init_w=True,
                        beta=BETA, dropout=DROPOUT, random_seed=MODEL_SEED)
    model.load_state_dict(state)
    model.to(device)
    return model


def _one_step(torch: Any, model: Any, optimizer: Any, batch: tuple[np.ndarray, np.ndarray, np.ndarray],
              arm: str, batch_index: int, seed: int, device: Any) -> dict[str, Any]:
    _set_matched_rng(torch, seed)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    x1_np, delta_np, feature_np = batch
    x1 = torch.as_tensor(x1_np, dtype=torch.float32, device=device)
    delta = torch.as_tensor(delta_np, dtype=torch.float32, device=device)
    features = torch.as_tensor(feature_np, dtype=torch.float32, device=device)
    x2 = x1 + delta
    model.train()
    optimizer.zero_grad(set_to_none=True)
    x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, _ = model(x1, features)
    z2, mu2, logvar2 = model.encode_x2(x2)
    x2_rec = model.decode_x2(z2)
    total, mse_x1, mse_x2, mse_pert, kld_x1, kld_x2, kld_pert = model.loss(
        x1, x1_rec, mu1, logvar1, x2, x2_rec, mu2, logvar2,
        x2_pred, mu_pred, logvar_pred,
    )
    components = {"mse_x1": float(mse_x1.detach().cpu()),
                  "mse_x2": float(mse_x2.detach().cpu()),
                  "mse_pert": float(mse_pert.detach().cpu()),
                  "kld_x1": float(kld_x1.detach().cpu()),
                  "kld_x2": float(kld_x2.detach().cpu()),
                  "kld_pert": float(kld_pert.detach().cpu())}
    if not all(math.isfinite(value) for value in [float(total.detach().cpu()), *components.values()]):
        raise FloatingPointError(f"non-finite official loss for {arm} batch {batch_index}")
    batch_n = int(x1.shape[0])
    scale = float(batch_n * N_GENES) if arm == "current_adapter" else 1.0
    objective = total / scale
    objective.backward()
    gradient_norm_preclip = _global_gradient_norm(torch, model)
    if not math.isfinite(gradient_norm_preclip):
        raise FloatingPointError(f"non-finite gradient norm for {arm} batch {batch_index}")
    clip_returned_norm: float | None = None
    if arm == "current_adapter":
        clip_returned_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRAD_CLIP_NORM, error_if_nonfinite=True
        ).detach().cpu())
    parameters = list(model.parameters())
    before = [parameter.detach().clone() for parameter in parameters]
    optimizer.step()
    torch.cuda.synchronize(device)
    update_norm = _parameter_update_norm(torch, before, model)
    if not math.isfinite(update_norm):
        raise FloatingPointError(f"non-finite parameter update norm for {arm} batch {batch_index}")
    elapsed = time.perf_counter() - started
    return {"arm": arm, "batch_index": int(batch_index), "batch_seed": int(seed),
            "batch_size": batch_n, "loss_total_raw_sum": float(total.detach().cpu()),
            **components, "objective_scale_divisor": scale,
            "gradient_norm_before_clip": gradient_norm_preclip,
            "gradient_norm_if_raw_objective": (gradient_norm_preclip * scale),
            "clip_returned_preclip_norm": clip_returned_norm,
            "parameter_update_l2_norm": update_norm,
            "step_wall_seconds": elapsed}


def _reserve_runtime_gpu(torch: Any, run_receipt: dict[str, Any]) -> tuple[dict[str, Any], float]:
    gpu_snapshot = _gpu_inventory()
    selected = gpu_snapshot["selected_device"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != APPROVED_GPU_UUID:
        raise RuntimeError("CUDA visibility was not fixed to the approved RTX A5000 before importing torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the frozen comparison requires exactly one visible CUDA device")
    torch.cuda.set_device(0)
    if "A5000" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"visible CUDA device differs from approved RTX A5000: {torch.cuda.get_device_name(0)}")
    reserved = _reserve_attempt(run_receipt["protocol_sha256"], gpu_snapshot)
    return {**reserved, "gpu_snapshot": gpu_snapshot}, time.perf_counter()


def _execute() -> dict[str, Any]:
    keys, control, delta, ecfp, extra = _load_bound_inputs()
    protocol = extra["protocol"]
    if OUT.joinpath("run_receipt.json").exists() or OUT.joinpath("smoke_receipt.json").exists():
        raise FileExistsError("run or smoke receipt already exists; no retry is authorized")
    # CUDA visibility must be fixed before importing torch or initializing its
    # runtime. The selected UUID is checked again immediately below.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = APPROVED_GPU_UUID
    torch = __import__("torch")
    gpu_context: dict[str, Any] | None = None
    attempt_started: float | None = None
    closed = False
    gate_receipt: dict[str, Any] | None = None
    error_text: str | None = None
    final_status = "FAILED_NO_RETRY"
    smoke_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    smoke_write_seconds_by_arm: dict[str, float] = {}
    train_write_seconds_by_arm_batch: dict[str, float] = {}
    try:
        gpu_context, attempt_started = _reserve_runtime_gpu(torch, {
            "protocol_sha256": protocol["protocol_sha256"]
        })
        signal.setitimer(signal.ITIMER_REAL, ABSOLUTE_ATTEMPT_CAP_SECONDS - 30.0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_num_threads(4)
        device = torch.device("cuda:0")
        model_class, _module = _load_model_class()
        _set_matched_rng(torch, MODEL_SEED)
        template = model_class(N_GENES, LATENT_DIM, ENCODER_HIDDEN, DECODER_HIDDEN,
                               N_FEATURES, FEATURE_EMBED_DIM, device=device, init_w=True,
                               beta=BETA, dropout=DROPOUT, random_seed=MODEL_SEED).to(device)
        initial_state = {name: tensor.detach().cpu().clone() for name, tensor in template.state_dict().items()}
        initial_state_sha = _state_digest(initial_state)
        del template
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

        smoke_models = {
            "current_adapter": _new_model(torch, model_class, initial_state, device),
            "official_update_rule": _new_model(torch, model_class, initial_state, device),
        }
        smoke_opts = {name: torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                              weight_decay=WEIGHT_DECAY)
                      for name, model in smoke_models.items()}
        first_batch = (control[:BATCH_SIZE], delta[:BATCH_SIZE], ecfp[:BATCH_SIZE])
        # Include model import/initialization and smoke I/O in the measured
        # reservation-to-gate wall time used by the continuation decision.
        smoke_stage_started = attempt_started
        for arm in ("current_adapter", "official_update_rule"):
            row = _one_step(torch, smoke_models[arm], smoke_opts[arm], first_batch,
                            arm, 0, STEP_SEED_PREFIX, device)
            row["stage"] = "cost_smoke"
            smoke_rows.append(row)
            write_started = time.perf_counter()
            _append_jsonl(OUT / "smoke_steps.jsonl", row)
            smoke_write_seconds_by_arm[arm] = time.perf_counter() - write_started
        torch.cuda.synchronize(device)
        smoke_stage_seconds = time.perf_counter() - smoke_stage_started
        smoke_step_seconds = [row["step_wall_seconds"] for row in smoke_rows]
        slow_step_seconds = max(smoke_step_seconds)
        mean_smoke_write_seconds = float(np.mean(list(smoke_write_seconds_by_arm.values())))
        nonstep_overhead_seconds = max(
            0.0, smoke_stage_seconds - sum(smoke_step_seconds)
            - sum(smoke_write_seconds_by_arm.values())
        )
        projected_full_seconds = ((N_FULL_STEPS * slow_step_seconds)
                                  + nonstep_overhead_seconds
                                  + (N_FULL_STEPS * mean_smoke_write_seconds))
        projected_with_margin_seconds = (smoke_stage_seconds + projected_full_seconds) * 1.20
        projected_gpu_hours = projected_with_margin_seconds / 3600.0
        loss_pair_diff = max(abs(smoke_rows[0][name] - smoke_rows[1][name])
                             for name in ["loss_total_raw_sum", "mse_x1", "mse_x2", "mse_pert",
                                          "kld_x1", "kld_x2", "kld_pert"])
        if loss_pair_diff > 1e-5:
            raise RuntimeError(f"matched smoke initial losses differ between arms: max_abs={loss_pair_diff}")
        gate_pass = projected_gpu_hours <= SMOKE_ESTIMATE_GATE_GPU_HOURS
        gate_receipt = {
            "status": "SMOKE_GATE_PASSED" if gate_pass else "STOPPED_AT_SMOKE_COST_GATE",
            "smoke_step_count": len(smoke_rows),
            "smoke_step_seconds_by_arm": {row["arm"]: row["step_wall_seconds"] for row in smoke_rows},
            "smoke_stage_wall_seconds": smoke_stage_seconds,
            "slower_smoke_step_wall_seconds": slow_step_seconds,
            "nonstep_init_and_io_overhead_wall_seconds": nonstep_overhead_seconds,
            "smoke_jsonl_write_seconds_by_arm": smoke_write_seconds_by_arm,
            "projected_full_jsonl_write_seconds": N_FULL_STEPS * mean_smoke_write_seconds,
            "projected_full_16_step_comparison_seconds": projected_full_seconds,
            "projected_smoke_plus_full_with_20pct_margin_gpu_hours": projected_gpu_hours,
            "continue_gate_gpu_hours": SMOKE_ESTIMATE_GATE_GPU_HOURS,
            "absolute_attempt_cap_gpu_hours": 1.0,
            "matched_initial_six_loss_max_abs_difference": loss_pair_diff,
            "gpu_snapshot_at_reservation": gpu_context["gpu_snapshot"],
            "initial_state_sha256": initial_state_sha,
        }
        atomic_json(OUT / "smoke_receipt.json", gate_receipt)
        if not gate_pass:
            final_status = "STOPPED_AT_SMOKE_COST_GATE"
            del smoke_opts, smoke_models
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            closed_row = _close_attempt(attempt_started, final_status, None, gate_receipt)
            closed = True
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            result = {"run_id": RUN_ID, "status": final_status,
                      "protocol_sha256": protocol["protocol_sha256"],
                      "smoke_receipt": gate_receipt, "ledger_close": closed_row,
                      "gpu_training_full_compare_started": False}
            atomic_json(OUT / "run_receipt.json", result)
            atomic_json(OUT / "run_manifest.json", result)
            return result

        if time.perf_counter() - attempt_started >= ABSOLUTE_ATTEMPT_CAP_SECONDS:
            raise TimeoutError("one-hour absolute GPU attempt cap reached after smoke")
        train_models = {
            "current_adapter": _new_model(torch, model_class, initial_state, device),
            "official_update_rule": _new_model(torch, model_class, initial_state, device),
        }
        train_opts = {name: torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                             weight_decay=WEIGHT_DECAY)
                      for name, model in train_models.items()}
        for arm in ("current_adapter", "official_update_rule"):
            for batch_index in range(N_BATCHES):
                if time.perf_counter() - attempt_started >= ABSOLUTE_ATTEMPT_CAP_SECONDS:
                    raise TimeoutError("one-hour absolute GPU attempt cap reached during comparison")
                start = batch_index * BATCH_SIZE
                end = start + BATCH_SIZE
                batch = (control[start:end], delta[start:end], ecfp[start:end])
                row = _one_step(torch, train_models[arm], train_opts[arm], batch,
                                arm, batch_index, STEP_SEED_PREFIX + batch_index, device)
                row["stage"] = "one_epoch_train_only"
                row["selection_rank_start"] = start + 1
                row["selection_rank_end"] = end
                train_rows.append(row)
                write_started = time.perf_counter()
                _append_jsonl(OUT / "training_steps.jsonl", row)
                train_write_seconds_by_arm_batch[f"{arm}:{batch_index}"] = time.perf_counter() - write_started
        if len(train_rows) != N_FULL_STEPS:
            raise RuntimeError(f"expected exactly {N_FULL_STEPS} train-only steps, wrote {len(train_rows)}")
        final_status = "COMPLETED_TRAIN512_UPDATE_RULE_A_B"
        del train_opts, train_models, smoke_opts, smoke_models
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        closed_row = _close_attempt(attempt_started, final_status, None, gate_receipt)
        closed = True
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        result = _summarize(protocol, gate_receipt, train_rows, initial_state_sha, closed_row)
        result["training_jsonl_write_seconds_by_arm_batch"] = train_write_seconds_by_arm_batch
        result["training_jsonl_write_seconds_total"] = float(sum(train_write_seconds_by_arm_batch.values()))
        atomic_json(OUT / "run_receipt.json", result)
        atomic_json(OUT / "run_manifest.json", result)
        return result
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        error_text = f"{type(exc).__name__}: {exc}"
        if attempt_started is not None and not closed:
            try:
                close_row = _close_attempt(attempt_started, "FAILED_NO_RETRY", error_text, gate_receipt)
                closed = True
            except BaseException as ledger_exc:
                error_text += f"; ledger_close_error={type(ledger_exc).__name__}: {ledger_exc}"
                close_row = None
        else:
            close_row = None
        result = {"run_id": RUN_ID, "status": "FAILED_NO_RETRY",
                  "protocol_sha256": protocol["protocol_sha256"],
                  "error": error_text, "smoke_steps_completed": smoke_rows,
                  "training_steps_completed": train_rows, "cost_gate": gate_receipt,
                  "ledger_close": close_row}
        atomic_json(OUT / "failure_receipt.json", result)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        if attempt_started is not None and not closed:
            # Preserve the full charged attempt if an unexpected control-flow path misses close.
            try:
                _close_attempt(attempt_started, "FAILED_NO_RETRY", error_text or "unexpected unclosed attempt", gate_receipt)
            except BaseException:
                pass


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _summarize(protocol: dict[str, Any], gate: dict[str, Any], rows: list[dict[str, Any]],
               init_sha: str, ledger_close: dict[str, Any]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    summaries: dict[str, Any] = {}
    compare_metrics = ["loss_total_raw_sum", "mse_x1", "mse_x2", "mse_pert",
                       "kld_x1", "kld_x2", "kld_pert", "gradient_norm_before_clip",
                       "gradient_norm_if_raw_objective", "parameter_update_l2_norm",
                       "step_wall_seconds"]
    for arm in ("current_adapter", "official_update_rule"):
        arm_frame = frame.loc[frame["arm"].eq(arm)].sort_values("batch_index")
        summaries[arm] = {
            "step_count": int(len(arm_frame)),
            "per_step": [{key: (int(row[key]) if key in {"batch_index", "selection_rank_start", "selection_rank_end"}
                                else float(row[key]) if key in compare_metrics else row[key])
                          for key in ["batch_index", "selection_rank_start", "selection_rank_end", *compare_metrics]
                          if key in row and pd.notna(row[key])}
                         for row in arm_frame.to_dict(orient="records")],
            "total_step_wall_seconds": float(arm_frame["step_wall_seconds"].sum()),
            "mean_parameter_update_l2_norm": float(arm_frame["parameter_update_l2_norm"].mean()),
            "last_raw_loss_sum": float(arm_frame.iloc[-1]["loss_total_raw_sum"]),
        }
    paired: list[dict[str, Any]] = []
    for batch_index in range(N_BATCHES):
        left = frame[(frame.arm == "current_adapter") & (frame.batch_index == batch_index)].iloc[0]
        right = frame[(frame.arm == "official_update_rule") & (frame.batch_index == batch_index)].iloc[0]
        paired.append({"batch_index": batch_index,
                       "selection_rank_start": batch_index * BATCH_SIZE + 1,
                       "selection_rank_end": (batch_index + 1) * BATCH_SIZE,
                       "raw_loss_sum_current": float(left.loss_total_raw_sum),
                       "raw_loss_sum_official_update": float(right.loss_total_raw_sum),
                       "preclip_gradient_norm_current": float(left.gradient_norm_before_clip),
                       "preclip_gradient_norm_official_update": float(right.gradient_norm_before_clip),
                       "parameter_update_norm_current": float(left.parameter_update_l2_norm),
                       "parameter_update_norm_official_update": float(right.parameter_update_l2_norm),
                       "current_over_official_update_norm_ratio": float(left.parameter_update_l2_norm / right.parameter_update_l2_norm) if right.parameter_update_l2_norm else None})
    return {
        "experiment_id": "EXP-050", "run_id": RUN_ID,
        "status": "COMPLETED_TRAIN512_UPDATE_RULE_A_B",
        "completed_at_utc": utc_now(), "protocol_sha256": protocol["protocol_sha256"],
        "input_key_sha256": protocol["hashes"]["selected_key_tsv_sha256"],
        "input_train_values_sha256": protocol["hashes"]["selected_value_npz_sha256"],
        "input_selected_ecfp4_sha256": protocol["hashes"]["selected_ecfp4_npy_sha256"],
        "initial_state_sha256": init_sha,
        "smoke_cost_gate": gate,
        "training_step_count": len(rows), "steps_per_arm": N_BATCHES,
        "arm_summaries": summaries, "paired_train_step_differences": paired,
        "ledger_close": ledger_close,
        "interpretation_boundary": "Train-side optimization trajectory for these 512 conditions only. No validation/generalization, efficacy, drug-screening performance, or official HDF scale compatibility claim.",
        "forbidden_inputs_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["freeze", "run"], required=True)
    args = parser.parse_args()
    result = _freeze() if args.mode == "freeze" else _execute()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
