"""Bounded, train-only 100-epoch TranSiGen update-rule comparison.

The existing update-rule runner is imported only as a library of audited model,
input-contract, GPU-inventory, and single-step helpers. This driver binds a new
run ID and protocol; its smoke and training phases are separately ledgered.
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
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DEFAULT = ROOT / "configs/experiments/EXP-050-transigen-source-fit-trajectory-20261006-v3.yaml"
CORE_PATH = ROOT / "scripts/modeling/compare_exp050_transigen_update_rules.py"
LEDGER = ROOT / "artifacts/experiments/EXP-050/gpu_ledger.jsonl"
LEDGER_LOCK = ROOT / "artifacts/experiments/EXP-050/gpu_ledger.lock"
DEVICE_UUID = "GPU-9c366c2e-bfa5-c7b3-b16a-1117c40e6871"
EXPECTED_SETTLED_GPU_HOURS = 0.5839411080763866
PACKAGE_CAP_GPU_HOURS = 2.0
EXP050_PACKAGE_CAP_GPU_HOURS = 48.0
SMOKE_RESERVATION_GPU_HOURS = 0.1
ATTEMPT_PREFIX = "EXP050-TRANSIGEN-SOURCE-FIT-20261006-V3"
N_EPOCHS = 100
EVAL_SEED_PREFIX = 20263003


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> float:
    started = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
    return time.perf_counter() - started


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("run_id") != "20261006_transigen_source_fit_trajectory_v3":
        raise RuntimeError("configuration is not the manager-reviewed v3 20261006 A-package run")
    if raw.get("experiment_id") != "EXP-050":
        raise RuntimeError("unexpected experiment ID")
    if float(raw["authorization"]["package_gpu_cap"]) != PACKAGE_CAP_GPU_HOURS:
        raise RuntimeError("A-package GPU cap differs from the authorized 2.0 hours")
    if float(raw["authorization"]["smoke_gate_gpu_hours_with_20_percent_margin"]) != 1.8:
        raise RuntimeError("smoke continuation gate differs from the authorized 1.8 hours")
    if int(raw["optimizer"]["epochs"]) != N_EPOCHS or int(raw["data"]["conditions"]) != 512:
        raise RuntimeError("epoch count or frozen TRAIN key count differs from the approved protocol")
    return raw, _sha256_file(path)


def _load_core() -> Any:
    spec = importlib.util.spec_from_file_location("exp050_transigen_update_rules_audited_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load audited update-rule helpers: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bind_core(core: Any, cfg: dict[str, Any], config_sha: str) -> tuple[Path, Path, Path]:
    run_id = cfg["run_id"]
    out = ROOT / "artifacts/experiments/EXP-050" / run_id
    feature_cache = out / "selected_train512_ecfp4.npy"
    protocol_path = out / "protocol.json"
    core.RUN_ID = run_id
    core.OUT = out
    core.FEATURE_CACHE = feature_cache
    core.PROTOCOL = protocol_path
    core.ATTEMPT_ID = ATTEMPT_PREFIX + "-SMOKE-001"
    core.EXPECTED_LEDGER_START_GPU_HOURS = EXPECTED_SETTLED_GPU_HOURS
    core.PACKAGE_CAP_GPU_HOURS = EXP050_PACKAGE_CAP_GPU_HOURS
    core.ATTEMPT_RESERVATION_GPU_HOURS = SMOKE_RESERVATION_GPU_HOURS
    core.SMOKE_ESTIMATE_GATE_GPU_HOURS = 1.8
    core.ABSOLUTE_ATTEMPT_CAP_SECONDS = SMOKE_RESERVATION_GPU_HOURS * 3600.0
    core.N_FULL_STEPS = N_EPOCHS * core.N_BATCHES * core.N_ARMS
    expected_model = {
        "seed": core.MODEL_SEED, "latent_dim": core.LATENT_DIM,
        "encoder_hidden": core.ENCODER_HIDDEN, "decoder_hidden": core.DECODER_HIDDEN,
        "feature_dim": core.N_FEATURES, "feature_embed_dim": core.FEATURE_EMBED_DIM,
        "beta": core.BETA, "dropout": core.DROPOUT,
    }
    configured_model = {name: cfg["model"].get(name) for name in expected_model}
    if configured_model != expected_model:
        raise RuntimeError(f"model config differs from the audited TranSiGen wrapper: {configured_model}")
    if int(cfg["optimizer"]["batch_size"]) != core.BATCH_SIZE:
        raise RuntimeError("configured minibatch size differs from the audited update-rule runner")
    if float(cfg["optimizer"]["learning_rate"]) != core.LEARNING_RATE or float(cfg["optimizer"]["weight_decay"]) != core.WEIGHT_DECAY:
        raise RuntimeError("Adam configuration differs from the frozen update-rule comparison")
    key_actual = core.sha256_file(core.KEYS)
    value_actual = core.sha256_file(core.VALUE_NPZ)
    if cfg["data"]["key_sha256"] != key_actual:
        raise RuntimeError(f"frozen 512-key SHA differs from config: expected={cfg['data']['key_sha256']} actual={key_actual} path={core.KEYS}")
    if cfg["data"]["value_archive_sha256"] != value_actual:
        raise RuntimeError(f"frozen TRAIN value archive SHA differs from config: expected={cfg['data']['value_archive_sha256']} actual={value_actual} path={core.VALUE_NPZ}")
    return out, feature_cache, protocol_path


def _protocol_payload(core: Any, cfg: dict[str, Any], config_sha: str,
                      out: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol.update({
        "status": "FROZEN_BEFORE_GPU_SMOKE",
        "authorization": "User-authorized A-package comparison; only frozen 512 TRAIN conditions and their 978-gene axis; two update rules; fixed 100 epochs; one 64-key step per arm must pass the cost gate before the formal run.",
        "arms": {
            "adapter_normalized_global_clip_5": {
                "objective": "official six-term summed objective divided by actual minibatch_size * 978",
                "gradient_clip_global_norm": 5.0,
            },
            "official_raw_sum_no_clipping": {
                "objective": "official six-term raw summed objective",
                "gradient_clip_global_norm": None,
            },
            "shared": "same official architecture and scratch initialization, ECFP4 features, frozen key order, minibatches, seed stream, Adam and all other parameters; only objective scaling and global clipping differ",
        },
        "training": {
            "fixed_epochs": N_EPOCHS,
            "batch_size": int(cfg["optimizer"]["batch_size"]),
            "batch_order": "selection_rank 1..512, contiguous batches of 64; no shuffle",
            "smoke_state_used_for_formal_training": False,
            "formal_restart": "discard smoke model and Adam states; rebuild both arms from the common scratch initialization and run 100 complete epochs per arm",
            "early_stopping": False,
            "checkpoint_selection": False,
            "endpoint": "epoch 100",
            "batch_norm": "normal TRAIN-minibatch running-stat updates only; no separate calibration",
            "per_step_seed": "same deterministic seed for matched arm/epoch/batch pair",
        },
        "training_side_evaluation": {
            "keys": "the same 512 frozen TRAIN keys after each epoch",
            "forbidden_arrays_opened": ["inner-development responses", "outer/heldout responses", "locked evaluation responses", "PDO treatment responses"],
            "nmse": cfg["metrics"]["nmse"],
            "correlation": cfg["metrics"]["correlation"],
            "amplitude": cfg["metrics"]["amplitude"],
            "baselines": "zero change; per-gene mean delta computed once from the same 512 TRAIN targets",
            "model_prediction": "official model delta x2_pred - x1",
        },
        "cost_gate": {
            "smoke": "one real optimizer update on selection_rank 1..64 for each arm from an identical scratch state",
            "smoke_updates_costed_separately": 2,
            "formal_updates": N_EPOCHS * core.N_BATCHES * core.N_ARMS,
            "formal_full_512_key_evaluation_passes": N_EPOCHS * core.N_ARMS,
            "continuation_rule": "smoke is a separate cost sample; formal training restarts from the common scratch initialization for 100 full epochs per arm. Continue only if both smoke updates and both full 512-key TRAIN evaluation passes are finite and (actual complete smoke wall time + measured formal model/Adam reinitialization + 1600 * slower measured update + 200 * slower measured complete 512-key eval + estimated formal logging/finalization) * 1.20 <=1.8 GPU hours and cumulative A-package projection remains <=2.0 GPU hours",
            "absolute_A_package_cap_gpu_hours": PACKAGE_CAP_GPU_HOURS,
            "no_retry_without_cost_and_new_attempt_record": True,
        },
        "budget": {
            "exp050_settled_gpu_hours_before_package": EXPECTED_SETTLED_GPU_HOURS,
            "exp050_total_cap_gpu_hours": EXP050_PACKAGE_CAP_GPU_HOURS,
            "A_package_cap_gpu_hours": PACKAGE_CAP_GPU_HOURS,
            "open_or_reserved_before_package": 0.0,
            "ledger_path": str(LEDGER.relative_to(ROOT)).replace("\\", "/"),
        },
    })
    protocol["gate"] = {
        "smoke_steps": "one true optimizer update on the first frozen batch of 64 conditions per arm from common scratch initialization",
        "smoke_evaluation": "after each smoke update, time one finite eval-mode forward pass over all 512 frozen TRAIN keys and confirm BN running state is unchanged",
        "smoke_updates_costed_separately": 2,
        "formal_updates": N_EPOCHS * core.N_BATCHES * core.N_ARMS,
        "complete_512_key_eval_passes": N_EPOCHS * core.N_ARMS,
        "formal_initialization": "rebuild the common scratch state and both model/Adam pairs for the separate formal run; use the smoke-measured elapsed initialization/build time as the formal initialization estimate",
        "projected_formula": "(actual completed dual-arm smoke wall seconds including smoke initialization, two updates, two full 512-key prediction/metric/BatchNorm evaluation passes and smoke writes + measured repeat formal model/Adam initialization seconds + 1600 * slower measured smoke update seconds + 200 * slower measured complete 512-key prediction/metric/BatchNorm evaluation seconds + estimated formal step/epoch logging and finalization seconds) * 1.20",
        "continue_if_projected_total_including_smoke_gpu_hours_at_most": 1.8,
        "absolute_attempt_cap_gpu_hours": 2.0,
        "a_package_cumulative_cap_gpu_hours": 2.0,
        "no_early_stop_or_epoch_selection": True,
    }
    protocol["hashes"]["trajectory_driver_sha256"] = _sha256_file(Path(__file__).resolve())
    protocol["hashes"]["trajectory_config_sha256"] = config_sha
    protocol["hashes"]["audited_core_runner_sha256"] = _sha256_file(CORE_PATH)
    protocol["outputs"] = [
        "protocol.json", "freeze_receipt.json", "run_manifest.json", "selected_train512_condition_keys.tsv",
        "selected_train512_ecfp4.npy", "code_snapshot/exp050_transigen_source_fit_trajectory_v3.py",
        "code_snapshot/compare_exp050_transigen_update_rules.py", "smoke_receipt.json", "smoke_steps.jsonl",
        "smoke_eval_steps.jsonl",
        "epoch_trajectory.jsonl", "train_step_stability.jsonl", "keyed_train_predictions.npz",
        "final_model_states.pt", "prediction_readback_receipt.json", "budget_closure.json",
        "sha256_manifest.json",
    ]
    protocol.pop("protocol_sha256", None)
    protocol["protocol_sha256"] = core.canonical_hash(protocol)
    core.atomic_json(protocol_path, protocol)
    key_copy = out / "selected_train512_condition_keys.tsv"
    shutil.copyfile(core.KEYS, key_copy)
    snapshot = out / "code_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).resolve(), snapshot / Path(__file__).name)
    shutil.copyfile(CORE_PATH, snapshot / CORE_PATH.name)
    config_snapshot = snapshot / "EXP-050-transigen-source-fit-trajectory-20261006-v3.yaml"
    shutil.copyfile(CONFIG_DEFAULT, config_snapshot)
    protocol["hashes"]["runner_snapshot_sha256"] = _sha256_file(snapshot / Path(__file__).name)
    protocol["hashes"]["core_snapshot_sha256"] = _sha256_file(snapshot / CORE_PATH.name)
    protocol["hashes"]["config_snapshot_sha256"] = _sha256_file(config_snapshot)
    protocol["hashes"].pop("trajectory_driver_sha256", None)
    protocol["hashes"]["trajectory_driver_sha256"] = _sha256_file(Path(__file__).resolve())
    protocol.pop("protocol_sha256", None)
    protocol["protocol_sha256"] = core.canonical_hash(protocol)
    core.atomic_json(protocol_path, protocol)
    return protocol


def _freeze(core: Any, cfg: dict[str, Any], config_sha: str, out: Path,
            protocol_path: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"run directory already exists; do not overwrite: {out}")
    freeze_receipt = core._freeze()
    protocol = _protocol_payload(core, cfg, config_sha, out, protocol_path)
    freeze_receipt.update({
        "status": "FROZEN_BEFORE_GPU_SMOKE",
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_file_sha256": _sha256_file(protocol_path),
        "trajectory_driver_sha256": protocol["hashes"]["trajectory_driver_sha256"],
        "trajectory_config_sha256": config_sha,
        "selected_key_sha256": protocol["hashes"]["selected_key_tsv_sha256"],
        "selected_value_sha256": protocol["hashes"]["selected_value_npz_sha256"],
    })
    _atomic_json(out / "freeze_receipt.json", freeze_receipt)
    manifest = {
        "experiment_id": "EXP-050", "run_id": cfg["run_id"],
        "status": "FROZEN_PROTOCOL_SMOKE_PENDING",
        "created_at_utc": _utc_now(),
        "protocol_sha256": protocol["protocol_sha256"],
        "code": {"trajectory_driver_sha256": protocol["hashes"]["trajectory_driver_sha256"],
                 "audited_core_runner_sha256": protocol["hashes"]["audited_core_runner_sha256"],
                 "official_model_py_sha256": protocol["hashes"]["official_model_py_sha256"]},
        "input_contract": {"key_count": 512, "gene_count": 978,
                           "selected_key_sha256": protocol["hashes"]["selected_key_tsv_sha256"],
                           "selected_values_sha256": protocol["hashes"]["selected_value_npz_sha256"],
                           "selected_ecfp4_sha256": protocol["hashes"]["selected_ecfp4_npy_sha256"]},
        "gpu": {"approved_uuid": DEVICE_UUID, "A_package_cap_gpu_hours": PACKAGE_CAP_GPU_HOURS,
                "smoke_first": True, "formal_100_epoch_training_started": False},
        "outputs_complete": False,
    }
    _atomic_json(out / "run_manifest.json", manifest)
    return {"run_id": cfg["run_id"], "status": "FROZEN_BEFORE_GPU_SMOKE",
            "output_dir": str(out), "protocol_sha256": protocol["protocol_sha256"],
            "selected_key_sha256": protocol["hashes"]["selected_key_tsv_sha256"],
            "trajectory_driver_sha256": protocol["hashes"]["trajectory_driver_sha256"],
            "protocol_file_sha256": _sha256_file(protocol_path)}


def _ledger_snapshot(core: Any) -> dict[str, Any]:
    with LEDGER_LOCK.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            rows = core._ledger_rows()
            state = core._ledger_state(rows)
            return {**state, "ledger_sha256": _sha256_file(LEDGER), "events": rows}
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _package_charge(core: Any, run_id: str) -> float:
    rows = core._ledger_rows()
    starts = {str(row.get("attempt_id")): row for row in rows
              if row.get("event") == "attempt_started" and row.get("run_id") == run_id}
    closes = {str(row.get("attempt_id")): row for row in rows
              if row.get("event") in {"attempt_finished", "attempt_reconciled"}
              and row.get("run_id") == run_id}
    if not set(closes).issubset(starts):
        raise RuntimeError("A-package ledger contains a close without a matching start")
    return float(sum(float(row["gpu_hours_charged"]) for row in closes.values()))


def _reserve_ledger(core: Any, protocol: dict[str, Any], cfg: dict[str, Any],
                    mode: str, gpu_snapshot: dict[str, Any]) -> tuple[str, float]:
    run_id = cfg["run_id"]
    attempt_id = ATTEMPT_PREFIX + ("-SMOKE-001" if mode == "smoke" else "-TRAIN-002")
    with LEDGER_LOCK.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            rows = core._ledger_rows()
            state = core._ledger_state(rows)
            if state["open_attempt_ids"] or state["reserved_gpu_hours"] != 0:
                raise RuntimeError(f"EXP-050 GPU ledger has open/reserved work: {state['open_attempt_ids']}")
            baseline = float(protocol["budget"]["exp050_settled_gpu_hours_before_package"])
            already_charged = _package_charge(core, run_id)
            expected_global = baseline + already_charged
            if abs(state["settled_gpu_hours"] - expected_global) > 1e-8:
                raise RuntimeError(
                    f"EXP-050 ledger changed after protocol freeze: settled={state['settled_gpu_hours']:.15f}; "
                    f"expected={expected_global:.15f}"
                )
            if attempt_id in {str(row.get("attempt_id")) for row in rows if row.get("event") == "attempt_started"}:
                raise RuntimeError(f"attempt already exists; no retry or overwrite: {attempt_id}")
            remaining = PACKAGE_CAP_GPU_HOURS - already_charged
            if remaining <= 0:
                raise RuntimeError("A-package 2.0 GPU-hour cap is exhausted")
            if mode == "smoke":
                reserve = min(SMOKE_RESERVATION_GPU_HOURS, remaining)
            else:
                reserve = remaining
            if state["settled_gpu_hours"] + reserve > EXP050_PACKAGE_CAP_GPU_HOURS:
                raise RuntimeError("the EXP-050 48 GPU-hour cap would be exceeded")
            event = {
                "experiment_id": "EXP-050", "run_id": run_id, "attempt_id": attempt_id,
                "event": "attempt_started", "timestamp_utc": _utc_now(),
                "operation": "TranSiGen source fit trajectory: " + ("real 64-key dual-arm cost smoke" if mode == "smoke" else "fixed 100-epoch train-side dual-arm comparison"),
                "mode": mode, "device": "cuda:0", "device_uuid": DEVICE_UUID,
                "device_name": gpu_snapshot["selected_device"]["name"], "gpu_count": 1,
                "pid": os.getpid(), "reserved_gpu_hours": reserve,
                "a_package_cap_gpu_hours": PACKAGE_CAP_GPU_HOURS,
                "a_package_charged_before_attempt_gpu_hours": already_charged,
                "exp050_settled_before_attempt_gpu_hours": state["settled_gpu_hours"],
                "protocol_sha256": protocol["protocol_sha256"],
                "gpu_snapshot": gpu_snapshot,
            }
            data = (json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            fd = os.open(LEDGER, os.O_WRONLY | os.O_APPEND)
            try:
                if os.write(fd, data) != len(data):
                    raise OSError("short EXP-050 ledger start write")
                os.fsync(fd)
            finally:
                os.close(fd)
            return attempt_id, reserve
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _close_ledger(core: Any, attempt_id: str, start_time: float,
                  status: str, error: str | None, cost_gate: dict[str, Any] | None) -> dict[str, Any]:
    elapsed = max(0.0, time.perf_counter() - start_time)
    charge = elapsed / 3600.0
    event = {
        "experiment_id": "EXP-050", "run_id": str(core.RUN_ID),
        "attempt_id": attempt_id, "event": "attempt_finished", "timestamp_utc": _utc_now(),
        "status": status, "error": error, "elapsed_seconds": elapsed,
        "gpu_hours_charged": charge, "reserved_gpu_hours": 0.0,
        "device_uuid": DEVICE_UUID, "gpu_count": 1, "cost_gate": cost_gate,
    }
    with LEDGER_LOCK.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            rows = core._ledger_rows()
            state = core._ledger_state(rows)
            if attempt_id not in state["open_attempt_ids"]:
                raise RuntimeError(f"cannot close absent or already closed attempt: {attempt_id}")
            data = (json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            fd = os.open(LEDGER, os.O_WRONLY | os.O_APPEND)
            try:
                if os.write(fd, data) != len(data):
                    raise OSError("short EXP-050 ledger close write")
                os.fsync(fd)
            finally:
                os.close(fd)
            rows_after = core._ledger_rows()
            state_after = core._ledger_state(rows_after)
            event["ledger_settled_gpu_hours_after_close"] = state_after["settled_gpu_hours"]
            event["ledger_reserved_gpu_hours_after_close"] = state_after["reserved_gpu_hours"]
            event["ledger_open_attempts_after_close"] = state_after["open_attempt_ids"]
            event["ledger_file_sha256_after_close"] = _sha256_file(LEDGER)
            return event
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _own_process_chain() -> set[int]:
    output = subprocess.check_output(["ps", "-eo", "pid=,ppid=,args="], text=True)
    rows: dict[int, tuple[int, str]] = {}
    for line in output.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            continue
        try:
            rows[int(fields[0])] = (int(fields[1]), fields[2])
        except ValueError:
            continue
    own: set[int] = set()
    current = os.getpid()
    while current in rows and current not in own:
        own.add(current)
        current = rows[current][0]
        if current <= 1:
            break
    return own


def _training_process_check() -> list[str]:
    output = subprocess.check_output(["ps", "-eo", "pid=,ppid=,etimes=,args="], text=True)
    own = _own_process_chain()
    hits = []
    tokens = (Path(__file__).name, CORE_PATH.name, "exp050_transigen_fit_diagnostic.py",
              "exp050_xpert_independent_response_fit.py")
    for line in output.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) != 4:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        if pid not in own and any(token in fields[3] for token in tokens):
            hits.append(line.strip())
    if hits:
        raise RuntimeError(f"another relevant training process is active: {hits}")
    return hits


def _verify_frozen(core: Any, cfg: dict[str, Any], config_sha: str,
                   out: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = protocol.get("protocol_sha256")
    payload = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if _canonical_hash(payload) != expected:
        raise RuntimeError("protocol SHA-256 does not match its frozen content")
    if protocol["hashes"].get("trajectory_driver_sha256") != _sha256_file(Path(__file__).resolve()):
        raise RuntimeError("trajectory runner changed since the protocol was frozen")
    if protocol["hashes"].get("trajectory_config_sha256") != config_sha:
        raise RuntimeError("trajectory configuration changed since protocol freeze")
    if protocol["hashes"].get("audited_core_runner_sha256") != _sha256_file(CORE_PATH):
        raise RuntimeError("audited core helper changed since protocol freeze")
    if cfg["run_id"] != protocol["run_id"]:
        raise RuntimeError("config and protocol run IDs differ")
    core_expected = protocol["hashes"].get("runner_sha256")
    if core_expected != core.sha256_file(CORE_PATH):
        raise RuntimeError("protocol does not bind the audited core runner")
    return protocol


def _bn_diagnostics(torch: Any, model: Any, epoch: int) -> list[dict[str, Any]]:
    results = []
    layers = [(name, module) for name, module in model.named_modules()
              if isinstance(module, torch.nn.BatchNorm1d)]
    if len(layers) != 5:
        raise RuntimeError(f"official model BatchNorm1d count changed: expected 5, found {len(layers)}")
    for name, layer in layers:
        mean = layer.running_mean.detach().cpu().numpy()
        variance = layer.running_var.detach().cpu().numpy()
        if not np.isfinite(mean).all() or not np.isfinite(variance).all():
            raise FloatingPointError(f"non-finite BatchNorm running state at epoch {epoch}, module {name}")
        results.append({
            "epoch": epoch, "module": name,
            "running_mean_sha256": hashlib.sha256(np.ascontiguousarray(mean).tobytes()).hexdigest(),
            "running_var_sha256": hashlib.sha256(np.ascontiguousarray(variance).tobytes()).hexdigest(),
            "running_mean_min": float(mean.min()), "running_mean_max": float(mean.max()),
            "running_var_min": float(variance.min()), "running_var_max": float(variance.max()),
            "num_batches_tracked": int(layer.num_batches_tracked.detach().cpu()),
            "finite": True, "source": "normal TRAIN minibatch updates only",
        })
    return results


def _bn_digest(torch: Any, model: Any) -> str:
    values = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            values.append((name,
                           hashlib.sha256(module.running_mean.detach().cpu().numpy().tobytes()).hexdigest(),
                           hashlib.sha256(module.running_var.detach().cpu().numpy().tobytes()).hexdigest(),
                           int(module.num_batches_tracked.detach().cpu())))
    return _canonical_hash(values)


def _predict_same_training_keys(torch: Any, model: Any, control: np.ndarray,
                                features: np.ndarray, seed: int, device: Any) -> tuple[np.ndarray, str, str]:
    previous_mode = bool(model.training)
    before = _bn_digest(torch, model)
    pieces = []
    with torch.random.fork_rng(devices=[device.index or 0], enabled=True):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model.eval()
        with torch.no_grad():
            for start in range(0, len(control), 64):
                x1 = torch.as_tensor(control[start:start + 64], dtype=torch.float32, device=device)
                feature = torch.as_tensor(features[start:start + 64], dtype=torch.float32, device=device)
                output = model(x1, feature)
                delta_prediction = output[3] - x1
                if not bool(torch.isfinite(delta_prediction).all().item()):
                    raise FloatingPointError(f"non-finite TRAIN-key prediction at rows {start}:{start + len(x1)}")
                pieces.append(delta_prediction.detach().cpu().numpy().astype(np.float32, copy=True))
        model.train(previous_mode)
    after = _bn_digest(torch, model)
    if before != after:
        raise RuntimeError("BatchNorm running state changed during eval-mode TRAIN-key prediction")
    prediction = np.concatenate(pieces, axis=0)
    if prediction.shape != control.shape or not np.isfinite(prediction).all():
        raise RuntimeError("TRAIN-key prediction shape or finite-value check failed")
    return prediction, before, after


def _profile_metrics(prediction: np.ndarray, truth: np.ndarray,
                     gene_scale: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 2 or not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("metric inputs must be finite, same-shaped key-by-gene matrices")
    valid_gene = np.isfinite(gene_scale) & (gene_scale > 1e-12)
    if not valid_gene.any():
        raise ValueError("training-only gene-scale NMSE has no nonconstant genes")
    normalized_squared_error = ((pred[:, valid_gene] - target[:, valid_gene]) / gene_scale[valid_gene]) ** 2
    key_nmse = normalized_squared_error.mean(axis=1)
    corrs = []
    for row in range(target.shape[0]):
        x = pred[row]
        y = target[row]
        x = x - x.mean()
        y = y - y.mean()
        denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
        if denominator > 1e-20:
            corrs.append(float(np.dot(x, y) / denominator))
    target_amplitude = float(np.mean(np.abs(target)))
    prediction_amplitude = float(np.mean(np.abs(pred)))
    pooled_error = float(np.sum((pred - target) ** 2))
    train_centered = target - target.mean(axis=0, keepdims=True)
    pooled_scale = float(np.sum(train_centered ** 2))
    return {
        "condition_key_count": int(target.shape[0]),
        "gene_count": int(target.shape[1]),
        "nmse_standardized_key_macro": float(key_nmse.mean()),
        "nmse_standardized_key_median": float(np.median(key_nmse)),
        "nmse_nonconstant_gene_count": int(valid_gene.sum()),
        "nmse_zero_scale_gene_count_excluded": int((~valid_gene).sum()),
        "nmse_pooled_train_variance": pooled_error / pooled_scale if pooled_scale > 0 else None,
        "pearson_key_macro": float(np.mean(corrs)) if corrs else None,
        "pearson_valid_key_count": int(len(corrs)),
        "pearson_undefined_key_count": int(target.shape[0] - len(corrs)),
        "mean_abs_prediction": prediction_amplitude,
        "mean_abs_observed": target_amplitude,
        "amplitude_ratio_prediction_over_observed": prediction_amplitude / target_amplitude if target_amplitude > 0 else None,
        "prediction_finite": True,
        "effective_denominators": {"nmse_keys": int(target.shape[0]), "nmse_genes": int(valid_gene.sum()),
                                   "pearson_keys": int(len(corrs)), "amplitude_key_gene_pairs": int(target.size)},
    }


def _metric_panel(prediction: np.ndarray, truth: np.ndarray,
                  train_mean: np.ndarray, gene_scale: np.ndarray) -> dict[str, Any]:
    zeros = np.zeros_like(truth, dtype=np.float32)
    means = np.broadcast_to(train_mean[None, :], truth.shape).astype(np.float32, copy=False)
    return {
        "model": _profile_metrics(prediction, truth, gene_scale),
        "zero_change": _profile_metrics(zeros, truth, gene_scale),
        "training_mean_delta": _profile_metrics(means, truth, gene_scale),
    }


def _optimizer_finite(torch: Any, optimizer: Any) -> bool:
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
                return False
    return True


def _projected_cost(smoke: dict[str, Any], core: Any,
                    completed_package_gpu_hours: float) -> dict[str, Any]:
    slow_step = float(smoke["slowest_smoke_step_seconds"])
    steps = N_EPOCHS * core.N_BATCHES * core.N_ARMS
    eval_passes = N_EPOCHS * core.N_ARMS
    slow_eval = max(float(value) for value in smoke["full_512_eval_seconds_by_arm"].values())
    writes_step = float(smoke["mean_smoke_step_jsonl_write_seconds"])
    writes_epoch = float(smoke["mean_smoke_eval_jsonl_write_seconds"])
    smoke_elapsed = float(smoke["smoke_attempt_elapsed_seconds"])
    repeat_init_seconds = float(smoke["measured_repeat_model_optimizer_initialization_seconds"])
    estimated_train_seconds = steps * slow_step
    estimated_eval_seconds = eval_passes * slow_eval
    estimated_step_log_seconds = steps * max(writes_step, 0.001)
    estimated_epoch_log_seconds = eval_passes * (0.1 + 2.0 * max(writes_epoch, 0.001))
    serialization_and_finalize_allowance_seconds = 60.0
    projected_no_margin = (smoke_elapsed + repeat_init_seconds + estimated_train_seconds + estimated_eval_seconds
                           + estimated_step_log_seconds + estimated_epoch_log_seconds
                           + serialization_and_finalize_allowance_seconds)
    projected_with_margin = projected_no_margin * 1.20
    total_gpu_hours_with_margin = projected_with_margin / 3600.0
    cumulative_projected_package_hours = completed_package_gpu_hours + projected_with_margin / 3600.0
    return {
        "formula": "(actual completed dual-arm smoke wall seconds including two smoke updates and two full 512-key prediction/metric/BatchNorm evals + measured repeat formal model/Adam initialization seconds + 1600 * slower measured smoke update seconds + 200 * slower measured complete 512-key prediction/metric/BatchNorm evaluation seconds + estimated formal step/epoch logging and finalization seconds) * 1.20",
        "formal_optimizer_updates": steps,
        "formal_full_512_key_evaluation_passes": eval_passes,
        "slowest_smoke_step_seconds": slow_step,
        "slowest_full_512_key_eval_pass_seconds": slow_eval,
        "actual_smoke_attempt_elapsed_seconds": smoke_elapsed,
        "measured_repeat_model_optimizer_initialization_seconds": repeat_init_seconds,
        "estimated_remaining_training_seconds": estimated_train_seconds,
        "estimated_eval_seconds": estimated_eval_seconds,
        "estimated_step_log_seconds": estimated_step_log_seconds,
        "estimated_epoch_log_seconds": estimated_epoch_log_seconds,
        "mean_smoke_step_jsonl_write_seconds": writes_step,
        "mean_smoke_eval_jsonl_write_seconds": writes_epoch,
        "serialization_and_finalize_allowance_seconds": serialization_and_finalize_allowance_seconds,
        "projected_seconds_before_margin": projected_no_margin,
        "margin_fraction": 0.20,
        "projected_total_gpu_hours_with_margin_including_smoke": total_gpu_hours_with_margin,
        "current_A_package_gpu_hours_settled": completed_package_gpu_hours,
        "projected_cumulative_A_package_gpu_hours": cumulative_projected_package_hours,
        "continue_gate_gpu_hours": 1.8,
        "absolute_A_package_cap_gpu_hours": PACKAGE_CAP_GPU_HOURS,
        "gate_pass": (total_gpu_hours_with_margin <= 1.8 and cumulative_projected_package_hours <= PACKAGE_CAP_GPU_HOURS),
    }


def _verify_runtime(core: Any, torch: Any) -> dict[str, Any]:
    snapshot = core._gpu_inventory()
    _training_process_check()
    if snapshot["selected_device"]["uuid"] != DEVICE_UUID or "A5000" not in snapshot["selected_device"]["name"]:
        raise RuntimeError("the approved A5000 is not the selected device")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != DEVICE_UUID:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must contain only the approved A5000 UUID")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly the approved single CUDA device must be visible")
    torch.cuda.set_device(0)
    if "A5000" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"visible device is not an RTX A5000: {torch.cuda.get_device_name(0)}")
    return snapshot


def _set_alarm(seconds: float) -> Any:
    old_handler = signal.getsignal(signal.SIGALRM)

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError("A-package GPU attempt time limit reached")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, max(1.0, seconds - 10.0))
    return old_handler


def _restore_alarm(old_handler: Any) -> None:
    signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, old_handler)


def _initialize_model(torch: Any, core: Any, model_class: Any,
                      device: Any) -> tuple[dict[str, Any], str]:
    core._set_matched_rng(torch, core.MODEL_SEED)
    template = model_class(core.N_GENES, core.LATENT_DIM, core.ENCODER_HIDDEN,
                           core.DECODER_HIDDEN, core.N_FEATURES, core.FEATURE_EMBED_DIM,
                           device=device, init_w=True, beta=core.BETA,
                           dropout=core.DROPOUT, random_seed=core.MODEL_SEED).to(device)
    state = {name: tensor.detach().cpu().clone() for name, tensor in template.state_dict().items()}
    digest = core._state_digest(state)
    del template
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return state, digest


def _build_model(torch: Any, core: Any, model_class: Any,
                 state: dict[str, Any], device: Any) -> Any:
    return core._new_model(torch, model_class, state, device)


def _read_smoke(out: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    path = out / "smoke_receipt.json"
    if not path.is_file():
        raise FileNotFoundError("run the required dual-arm 64-key GPU smoke before formal training")
    smoke = json.loads(path.read_text(encoding="utf-8"))
    if smoke.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise RuntimeError("smoke receipt does not belong to the frozen protocol")
    if smoke.get("status") != "SMOKE_GATE_PASSED_TRAIN_PENDING":
        raise RuntimeError(f"smoke did not pass the cost gate: {smoke.get('status')}")
    return smoke


def _run_gpu_phase(core: Any, cfg: dict[str, Any], config_sha: str,
                   out: Path, protocol_path: Path, mode: str) -> dict[str, Any]:
    protocol = _verify_frozen(core, cfg, config_sha, out, protocol_path)
    if mode == "smoke" and (out / "smoke_receipt.json").exists():
        raise FileExistsError("smoke receipt already exists; this run must not be repeated")
    if mode == "train":
        smoke = _read_smoke(out, protocol)
        cost_gate = smoke["cost_projection"]
        if not smoke["cost_projection"]["gate_pass"]:
            raise RuntimeError("formal training is forbidden because the smoke cost gate failed")
        if (out / "training_attempt_receipt.json").exists():
            raise FileExistsError("formal attempt receipt already exists; this run must not be repeated")
    keys, control, delta, features, extra = core._load_bound_inputs()
    if len(keys) != 512 or control.shape != (512, 978) or delta.shape != (512, 978):
        raise RuntimeError("bound training data no longer matches the frozen 512 x 978 contract")
    gene_ids = np.asarray(extra["gene_id"], dtype=str)
    truth = np.asarray(delta, dtype=np.float32)
    train_mean = truth.mean(axis=0, dtype=np.float64).astype(np.float32)
    train_scale = truth.std(axis=0, ddof=1, dtype=np.float64).astype(np.float64)
    if not np.isfinite(train_mean).all() or not np.isfinite(train_scale).all():
        raise RuntimeError("train-only mean or scale has non-finite entries")
    npz_keys = keys["condition_id"].astype(str).to_numpy()
    input_copy = pd.read_csv(out / "selected_train512_condition_keys.tsv", sep="\t", dtype={"condition_id": str})
    if not np.array_equal(npz_keys, input_copy["condition_id"].astype(str).to_numpy()):
        raise RuntimeError("run-local key copy does not round-trip in the frozen selection order")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_UUID
    torch = __import__("torch")
    snapshot = _verify_runtime(core, torch)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    model_class, _ = core._load_model_class()
    attempt_id, reserved = _reserve_ledger(core, protocol, cfg, mode, snapshot)
    attempt_started = time.perf_counter()
    attempt_started_utc = _utc_now()
    old_alarm = _set_alarm(reserved * 3600.0)
    status = "FAILED_ENGINEERING_OR_NUMERIC"
    error = None
    cost_gate: dict[str, Any] | None = None
    closed = False
    result: dict[str, Any] = {}
    smoke_steps: list[dict[str, Any]] = []
    train_step_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    final_predictions: dict[str, np.ndarray] = {}
    final_states: dict[str, Any] = {}
    try:
        initialization_started = time.perf_counter()
        initial_state, initial_state_sha = _initialize_model(torch, core, model_class, device)
        if mode == "smoke":
            models = {}
            model_build_seconds_by_arm = {}
            for arm in ("adapter_normalized_global_clip_5", "official_raw_sum_no_clipping"):
                model_build_started = time.perf_counter()
                models[arm] = _build_model(torch, core, model_class, initial_state, device)
                model_build_seconds_by_arm[arm] = time.perf_counter() - model_build_started
            # The audited helper recognizes the two existing arm labels.
            helper_arm = {"adapter_normalized_global_clip_5": "current_adapter",
                          "official_raw_sum_no_clipping": "official_update_rule"}
            optimizers = {arm: torch.optim.Adam(model.parameters(), lr=core.LEARNING_RATE,
                                                weight_decay=core.WEIGHT_DECAY)
                          for arm, model in models.items()}
            measured_repeat_init_seconds = time.perf_counter() - initialization_started
            batch = (control[:64], delta[:64], features[:64])
            step_write_times = []
            eval_write_times = []
            full_512_eval_seconds_by_arm: dict[str, float] = {}
            smoke_eval_rows: list[dict[str, Any]] = []
            for arm in models:
                step = core._one_step(torch, models[arm], optimizers[arm], batch,
                                      helper_arm[arm], 0, core.STEP_SEED_PREFIX, device)
                step.update({"arm": arm, "stage": "true_cost_smoke", "epoch": 0,
                             "selection_rank_start": 1, "selection_rank_end": 64,
                             "condition_key_sha256": _sha256_file(core.KEYS),
                             "initial_state_sha256": initial_state_sha})
                smoke_steps.append(step)
                step_write_times.append(_append_jsonl(out / "smoke_steps.jsonl", step))
                eval_started = time.perf_counter()
                eval_prediction, bn_before_eval, bn_after_eval = _predict_same_training_keys(
                    torch, models[arm], control, features, EVAL_SEED_PREFIX + 1, device)
                eval_metrics = _metric_panel(eval_prediction, truth, train_mean, train_scale)
                smoke_bn_rows = _bn_diagnostics(torch, models[arm], 0)
                torch.cuda.synchronize(device)
                eval_seconds = time.perf_counter() - eval_started
                if eval_prediction.shape != (512, 978) or not np.isfinite(eval_prediction).all():
                    raise FloatingPointError(f"full 512-key smoke evaluation failed for {arm}")
                if bn_before_eval != bn_after_eval:
                    raise RuntimeError(f"BatchNorm running state changed in smoke eval for {arm}")
                full_512_eval_seconds_by_arm[arm] = eval_seconds
                eval_row = {
                    "arm": arm, "stage": "true_cost_smoke_full_train_eval",
                    "epoch": 0, "condition_key_count": 512, "gene_count": 978,
                    "evaluation_seed": EVAL_SEED_PREFIX + 1,
                    "full_512_eval_wall_seconds": eval_seconds,
                    "prediction_finite": True,
                    "prediction_shape": list(eval_prediction.shape),
                    "batchnorm_state_unchanged": bn_before_eval == bn_after_eval,
                    "batchnorm_state_sha256": bn_after_eval,
                    "batchnorm_by_layer": smoke_bn_rows,
                    "training_key_metrics": eval_metrics,
                    "selection_key_sha256": _sha256_file(core.KEYS),
                }
                smoke_eval_rows.append(eval_row)
                eval_write_times.append(_append_jsonl(out / "smoke_eval_steps.jsonl", eval_row))
                del eval_prediction
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - attempt_started
            raw_names = ["loss_total_raw_sum", "mse_x1", "mse_x2", "mse_pert", "kld_x1", "kld_x2", "kld_pert"]
            loss_difference = max(abs(float(smoke_steps[0][name]) - float(smoke_steps[1][name]))
                                  for name in raw_names)
            if loss_difference > 1e-5:
                raise RuntimeError(f"matched first-batch loss mismatch before update: max_abs={loss_difference}")
            prior_charge = _package_charge(core, cfg["run_id"])
            cost_stub = {
                "steps": smoke_steps,
                "slowest_smoke_step_seconds": max(float(row["step_wall_seconds"]) for row in smoke_steps),
                "full_512_eval_seconds_by_arm": full_512_eval_seconds_by_arm,
                "measured_repeat_model_optimizer_initialization_seconds": measured_repeat_init_seconds,
                "model_build_seconds_by_arm": model_build_seconds_by_arm,
                "mean_smoke_step_jsonl_write_seconds": float(np.mean(step_write_times)),
                "mean_smoke_eval_jsonl_write_seconds": float(np.mean(eval_write_times)),
                "smoke_attempt_elapsed_seconds": elapsed,
            }
            cost_gate = _projected_cost(cost_stub, core, prior_charge)
            status = "SMOKE_GATE_PASSED_TRAIN_PENDING" if cost_gate["gate_pass"] else "STOPPED_AT_SMOKE_COST_GATE"
            result = {
                "experiment_id": "EXP-050", "run_id": cfg["run_id"], "status": status,
                "protocol_sha256": protocol["protocol_sha256"],
                "smoke_attempt_id": attempt_id, "attempt_started_at_utc": attempt_started_utc,
                "attempt_elapsed_seconds_before_close": elapsed,
                "device": snapshot["selected_device"], "gpu_snapshot": snapshot,
                "smoke_step_count": len(smoke_steps), "steps": smoke_steps,
                "full_512_eval_count": len(smoke_eval_rows),
                "full_512_eval_seconds_by_arm": full_512_eval_seconds_by_arm,
                "full_512_eval_receipts": smoke_eval_rows,
                "matched_initial_six_loss_max_abs_difference": loss_difference,
                "initial_state_sha256": initial_state_sha,
                "cost_projection": cost_gate,
                "formal_training_started": False,
                "forbidden_response_arrays_opened": False,
            }
            _atomic_json(out / "smoke_preclose_receipt.json", result)
            del optimizers, models
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            closed_row = _close_ledger(core, attempt_id, attempt_started, status, None, cost_gate)
            closed = True
            result["ledger_close"] = closed_row
            result["status"] = status
            _atomic_json(out / "smoke_receipt.json", result)
            _atomic_json(out / "run_manifest.json", {
                "experiment_id": "EXP-050", "run_id": cfg["run_id"], "status": status,
                "updated_at_utc": _utc_now(), "protocol_sha256": protocol["protocol_sha256"],
                "smoke_receipt_sha256": _sha256_file(out / "smoke_receipt.json"),
                "cost_projection": cost_gate, "formal_training_started": False,
                "GPU_ledger_close": closed_row,
                "next_action": "mode=train only if smoke cost_projection.gate_pass is true",
            })
            return result

        # Formal training starts only after a separately closed, hashed smoke.
        if initial_state_sha != smoke["initial_state_sha256"]:
            raise RuntimeError("formal initialization differs from the measured smoke initialization")
        package_charge_now = _package_charge(core, cfg["run_id"])
        smoke_actual_charge = float(smoke["ledger_close"]["gpu_hours_charged"])
        remaining_projected_after_smoke = max(0.0, cost_gate_cost(smoke) - smoke_actual_charge)
        if package_charge_now + remaining_projected_after_smoke > PACKAGE_CAP_GPU_HOURS:
            raise RuntimeError("smoke estimate plus settled A-package use exceeds the 2.0 GPU-hour cap")
        _atomic_json(out / "training_attempt_receipt.json", {
            "status": "FORMAL_FIXED_100_EPOCH_TRAINING_STARTED",
            "attempt_id": attempt_id, "started_at_utc": attempt_started_utc,
            "protocol_sha256": protocol["protocol_sha256"],
            "smoke_receipt_sha256": _sha256_file(out / "smoke_receipt.json"),
            "initial_state_sha256": initial_state_sha,
            "device": snapshot["selected_device"],
            "epochs": N_EPOCHS, "no_early_stopping": True,
        })
        gene_scale_for_metric = train_scale
        arms = ("adapter_normalized_global_clip_5", "official_raw_sum_no_clipping")
        helper_arm = {"adapter_normalized_global_clip_5": "current_adapter",
                      "official_raw_sum_no_clipping": "official_update_rule"}
        for arm in arms:
            model = _build_model(torch, core, model_class, initial_state, device)
            optimizer = torch.optim.Adam(model.parameters(), lr=core.LEARNING_RATE,
                                         weight_decay=core.WEIGHT_DECAY)
            for epoch in range(1, N_EPOCHS + 1):
                if time.perf_counter() - attempt_started >= reserved * 3600.0 - 20.0:
                    raise TimeoutError("remaining A-package attempt budget reached before next epoch")
                epoch_started = time.perf_counter()
                step_components = []
                for batch_index in range(core.N_BATCHES):
                    start = batch_index * core.BATCH_SIZE
                    end = start + core.BATCH_SIZE
                    batch = (control[start:end], delta[start:end], features[start:end])
                    matched_seed = core.STEP_SEED_PREFIX + (epoch - 1) * core.N_BATCHES + batch_index
                    row = core._one_step(torch, model, optimizer, batch,
                                         helper_arm[arm], batch_index, matched_seed, device)
                    finite_optimizer = _optimizer_finite(torch, optimizer)
                    if not finite_optimizer:
                        raise FloatingPointError(f"non-finite Adam state at {arm} epoch {epoch} batch {batch_index}")
                    parameter_finite = all(bool(torch.isfinite(p).all().item()) for p in model.parameters())
                    if not parameter_finite:
                        raise FloatingPointError(f"non-finite model parameter at {arm} epoch {epoch} batch {batch_index}")
                    row.update({"arm": arm, "epoch": epoch,
                                "selection_rank_start": start + 1,
                                "selection_rank_end": end,
                                "optimizer_state_finite": finite_optimizer,
                                "parameters_finite": parameter_finite,
                                "gradient_finite": math.isfinite(float(row["gradient_norm_before_clip"])),
                                "gradient_norm_after_clip_expected": min(float(row["gradient_norm_before_clip"]), 5.0) if arm == "adapter_normalized_global_clip_5" else float(row["gradient_norm_before_clip"]),
                                "global_clip_applied": arm == "adapter_normalized_global_clip_5"})
                    for field in ("loss_total_raw_sum", "mse_x1", "mse_x2", "mse_pert",
                                  "kld_x1", "kld_x2", "kld_pert",
                                  "gradient_norm_before_clip", "parameter_update_l2_norm"):
                        if not math.isfinite(float(row[field])):
                            raise FloatingPointError(f"non-finite {field} at {arm} epoch {epoch} batch {batch_index}")
                    step_components.append(row)
                    train_step_rows.append(row)
                    _append_jsonl(out / "train_step_stability.jsonl", row)

                component_names = ["mse_x1", "mse_x2", "mse_pert", "kld_x1", "kld_x2", "kld_pert"]
                loss_summary = {f"{name}_mean_raw_batch_sum": float(np.mean([r[name] for r in step_components]))
                                for name in component_names}
                raw_total = (loss_summary["mse_x1_mean_raw_batch_sum"]
                             + loss_summary["mse_x2_mean_raw_batch_sum"]
                             + loss_summary["mse_pert_mean_raw_batch_sum"]
                             + core.BETA * (loss_summary["kld_x1_mean_raw_batch_sum"]
                                            + loss_summary["kld_x2_mean_raw_batch_sum"]
                                            + loss_summary["kld_pert_mean_raw_batch_sum"]))
                objective_mean = raw_total / (core.BATCH_SIZE * core.N_GENES) if arm == "adapter_normalized_global_clip_5" else raw_total
                bn_before_eval = _bn_digest(torch, model)
                prediction, bn_eval_before, bn_eval_after = _predict_same_training_keys(
                    torch, model, control, features, EVAL_SEED_PREFIX + epoch, device)
                if bn_before_eval != bn_eval_before or bn_eval_before != bn_eval_after:
                    raise RuntimeError("BatchNorm state changed during the same-TRAIN-key evaluation pass")
                metrics = _metric_panel(prediction, truth, train_mean, gene_scale_for_metric)
                bn_rows = _bn_diagnostics(torch, model, epoch)
                epoch_row = {
                    "run_id": cfg["run_id"], "arm": arm, "epoch": epoch,
                    **loss_summary, "loss_total_raw_batch_sum_mean": raw_total,
                    "optimizer_objective_mean_per_minibatch": objective_mean,
                    "loss_total_equals_component_formula": True,
                    "learning_rate": core.LEARNING_RATE,
                    "batch_count": core.N_BATCHES, "condition_key_count": 512,
                    "parameter_update_norm_mean": float(np.mean([r["parameter_update_l2_norm"] for r in step_components])),
                    "parameter_update_norm_max": float(np.max([r["parameter_update_l2_norm"] for r in step_components])),
                    "gradient_norm_before_clip_mean": float(np.mean([r["gradient_norm_before_clip"] for r in step_components])),
                    "gradient_norm_before_clip_max": float(np.max([r["gradient_norm_before_clip"] for r in step_components])),
                    "gradient_norm_after_clip_expected_max": float(np.max([r["gradient_norm_after_clip_expected"] for r in step_components])),
                    "all_gradients_updates_parameters_optimizer_states_finite": True,
                    "evaluation_seed": EVAL_SEED_PREFIX + epoch,
                    "batchnorm_running_state_unchanged_during_eval": bn_eval_before == bn_eval_after,
                    "batchnorm_by_layer": bn_rows,
                    "train_key_metrics": metrics,
                    "epoch_wall_seconds": time.perf_counter() - epoch_started,
                    "dev_or_heldout_labels_read": False,
                }
                epoch_rows.append(epoch_row)
                _append_jsonl(out / "epoch_trajectory.jsonl", epoch_row)
                _append_jsonl(out / "batchnorm_by_epoch.jsonl", {
                    "arm": arm, "epoch": epoch, "batchnorm_by_layer": bn_rows,
                    "normal_train_minibatch_updates_only": True,
                    "separate_calibration_performed": False,
                    "eval_running_state_before_sha256": bn_eval_before,
                    "eval_running_state_after_sha256": bn_eval_after,
                })
                if epoch == N_EPOCHS:
                    final_predictions[arm] = prediction
                    final_states[arm] = {name: value.detach().cpu().clone()
                                         for name, value in model.state_dict().items()}
            del optimizer, model
            torch.cuda.empty_cache()
        if len(epoch_rows) != N_EPOCHS * 2:
            raise RuntimeError(f"expected 200 epoch rows, wrote {len(epoch_rows)}")
        if any(pred.shape != truth.shape or not np.isfinite(pred).all() for pred in final_predictions.values()):
            raise RuntimeError("epoch-100 prediction matrix is not finite or does not match the train axes")

        prediction_path = out / "keyed_train_predictions.npz"
        with prediction_path.open("wb") as stream:
            np.savez_compressed(
                stream,
                condition_id=np.asarray(keys["condition_id"].astype(str).tolist(), dtype=str),
                selection_rank=keys["selection_rank"].to_numpy(dtype=np.int64),
                gene_id=gene_ids,
                target_delta=truth.astype(np.float32),
                adapter_normalized_global_clip_5=final_predictions["adapter_normalized_global_clip_5"],
                official_raw_sum_no_clipping=final_predictions["official_raw_sum_no_clipping"],
                zero_change_prediction=np.zeros_like(truth, dtype=np.float32),
                training_mean_delta=train_mean,
                train_gene_sample_std=train_scale.astype(np.float32),
            )
        checkpoint_path = out / "final_model_states.pt"
        import torch as torch_package
        torch_package.save({
            "run_id": cfg["run_id"], "protocol_sha256": protocol["protocol_sha256"],
            "epoch": N_EPOCHS, "initial_state_sha256": initial_state_sha,
            "state_dicts": final_states,
        }, checkpoint_path)
        with np.load(prediction_path, allow_pickle=False) as readback:
            if not np.array_equal(readback["condition_id"].astype(str), npz_keys):
                raise RuntimeError("prediction readback condition keys differ from frozen order")
            if not np.array_equal(readback["gene_id"].astype(str), gene_ids):
                raise RuntimeError("prediction readback gene axis differs from frozen order")
            if not np.array_equal(readback["target_delta"], truth):
                raise RuntimeError("prediction readback target array differs from TRAIN values")
            readback_predictions = {
                "adapter_normalized_global_clip_5": readback["adapter_normalized_global_clip_5"].copy(),
                "official_raw_sum_no_clipping": readback["official_raw_sum_no_clipping"].copy(),
            }
            readback_metrics = {arm: _metric_panel(pred, truth, train_mean, train_scale)
                                for arm, pred in readback_predictions.items()}
        if any(not np.array_equal(readback_predictions[arm], final_predictions[arm]) for arm in final_predictions):
            raise RuntimeError("prediction archive readback is not bitwise identical")
        prediction_receipt = {
            "run_id": cfg["run_id"], "status": "PASS_EXACT_KEY_AXIS_AND_NUMERIC_READBACK",
            "prediction_sha256": _sha256_file(prediction_path),
            "condition_keys_sha256": protocol["hashes"]["selected_key_tsv_sha256"],
            "gene_axis_sha256": protocol["hashes"]["selected_gene_ids_sha256"],
            "readback_keys_exact": True, "readback_gene_axis_exact": True,
            "readback_targets_exact": True, "readback_predictions_bitwise_exact": True,
            "condition_key_count": 512, "gene_count": 978,
            "finite_checks": {"target": bool(np.isfinite(truth).all()),
                              "adapter_prediction": bool(np.isfinite(readback_predictions["adapter_normalized_global_clip_5"]).all()),
                              "official_prediction": bool(np.isfinite(readback_predictions["official_raw_sum_no_clipping"]).all())},
            "epoch_100_metrics_from_readback": readback_metrics,
        }
        _atomic_json(out / "prediction_readback_receipt.json", prediction_receipt)
        torch.cuda.synchronize(device)
        del final_states, final_predictions
        torch.cuda.empty_cache()
        status = "COMPLETED_FIXED_EPOCH100_TRAIN_SIDE_COMPARISON"
        closed_row = _close_ledger(core, attempt_id, attempt_started, status, None, cost_gate)
        closed = True
        _restore_alarm(old_alarm)
        package_total = _package_charge(core, cfg["run_id"])
        ledger_snapshot = _ledger_snapshot(core)
        if package_total > PACKAGE_CAP_GPU_HOURS + 1e-12:
            raise RuntimeError(f"A-package charged total exceeded 2.0 GPU hours: {package_total}")
        if ledger_snapshot["open_attempt_ids"] or ledger_snapshot["reserved_gpu_hours"] != 0:
            raise RuntimeError("GPU ledger did not close with zero open attempts/reservation")
        closure = {
            "run_id": cfg["run_id"], "attempt_id": attempt_id,
            "a_package_charged_gpu_hours": package_total,
            "a_package_cap_gpu_hours": PACKAGE_CAP_GPU_HOURS,
            "a_package_remaining_gpu_hours": PACKAGE_CAP_GPU_HOURS - package_total,
            "exp050_settled_gpu_hours_after_close": ledger_snapshot["settled_gpu_hours"],
            "exp050_package_cap_gpu_hours": EXP050_PACKAGE_CAP_GPU_HOURS,
            "exp050_open_attempt_ids_after_close": ledger_snapshot["open_attempt_ids"],
            "exp050_reserved_gpu_hours_after_close": ledger_snapshot["reserved_gpu_hours"],
            "ledger_sha256_after_close": ledger_snapshot["ledger_sha256"],
            "attempt_close": closed_row,
        }
        _atomic_json(out / "budget_closure.json", closure)
        step_frame = pd.DataFrame(train_step_rows)
        epoch_frame = pd.DataFrame(epoch_rows)
        _atomic_json(out / "training_attempt_receipt.json", {
            "status": status, "attempt_id": attempt_id, "started_at_utc": attempt_started_utc,
            "completed_at_utc": _utc_now(), "gpu_hours_charged": closed_row["gpu_hours_charged"],
            "epoch_rows": len(epoch_rows), "step_rows": len(train_step_rows),
            "prediction_sha256": _sha256_file(prediction_path),
        })
        outputs = [p for p in out.rglob("*") if p.is_file() and p.name not in {"sha256_manifest.json", "run_manifest.json"}]
        output_hashes = {str(path.relative_to(out)).replace("\\", "/"): _sha256_file(path)
                         for path in sorted(outputs)}
        sha_receipt = {"run_id": cfg["run_id"], "files": output_hashes,
                       "file_count": len(output_hashes), "created_at_utc": _utc_now()}
        _atomic_json(out / "sha256_manifest.json", sha_receipt)
        all_checksums = dict(output_hashes)
        all_checksums["sha256_manifest.json"] = _sha256_file(out / "sha256_manifest.json")
        result = {
            "experiment_id": "EXP-050", "run_id": cfg["run_id"], "status": status,
            "started_at_utc": attempt_started_utc, "completed_at_utc": _utc_now(),
            "protocol_sha256": protocol["protocol_sha256"],
            "model_source_sha256": protocol["hashes"]["official_model_py_sha256"],
            "actual_model": "official upstream TranSiGen architecture, scratch initialization, no inherited checkpoint",
            "initial_state_sha256": initial_state_sha,
            "arms": {arm: {"epochs": int(epoch_frame[epoch_frame.arm.eq(arm)].epoch.nunique()),
                           "epoch_100_metrics": readback_metrics[arm],
                           "mean_epoch_1_to_100_loss_total_raw_batch_sum": float(epoch_frame[epoch_frame.arm.eq(arm)]["loss_total_raw_batch_sum_mean"].mean()),
                           "epoch_1_loss_total_raw_batch_sum": float(epoch_frame[(epoch_frame.arm.eq(arm)) & (epoch_frame.epoch.eq(1))]["loss_total_raw_batch_sum_mean"].iloc[0]),
                           "epoch_100_loss_total_raw_batch_sum": float(epoch_frame[(epoch_frame.arm.eq(arm)) & (epoch_frame.epoch.eq(100))]["loss_total_raw_batch_sum_mean"].iloc[0]),
                           "epoch_100_optimizer_objective_mean_per_minibatch": float(epoch_frame[(epoch_frame.arm.eq(arm)) & (epoch_frame.epoch.eq(100))]["optimizer_objective_mean_per_minibatch"].iloc[0])}
                      for arm in ("adapter_normalized_global_clip_5", "official_raw_sum_no_clipping")},
            "epoch_rows": len(epoch_rows), "train_step_rows": len(train_step_rows),
            "predictions": prediction_receipt,
            "cost_gate": smoke["cost_projection"],
            "gpu_attempt": closed_row,
            "budget_closure": closure,
            "forbidden_response_arrays_opened": False,
            "interpretation_boundary": "This compares train-side optimization fit on the frozen 512 LINCS source conditions only. It does not establish validation/generalization, efficacy, a reproduced upstream program unit contract, or PDO transfer.",
            "checksums": all_checksums,
            "outputs_complete": True,
        }
        _atomic_json(out / "run_manifest.json", result)
        return result
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        status = "FAILED_NO_VALID_100_EPOCH_RESULT"
        try:
            _atomic_json(out / ("smoke_failure_receipt.json" if mode == "smoke" else "training_failure_receipt.json"), {
                "run_id": cfg["run_id"], "status": status, "mode": mode,
                "protocol_sha256": protocol["protocol_sha256"], "error": error,
                "smoke_steps_completed": smoke_steps, "epoch_rows_completed": len(epoch_rows),
                "train_step_rows_completed": len(train_step_rows),
                "forbidden_response_arrays_opened": False,
            })
        except BaseException:
            pass
        if not closed:
            try:
                close_row = _close_ledger(core, attempt_id, attempt_started, status, error, cost_gate)
                closed = True
                _atomic_json(out / ("smoke_failure_receipt.json" if mode == "smoke" else "training_failure_receipt.json"), {
                    "run_id": cfg["run_id"], "status": status, "mode": mode,
                    "protocol_sha256": protocol["protocol_sha256"], "error": error,
                    "smoke_steps_completed": smoke_steps, "epoch_rows_completed": len(epoch_rows),
                    "train_step_rows_completed": len(train_step_rows), "ledger_close": close_row,
                    "forbidden_response_arrays_opened": False,
                })
            except BaseException as ledger_exc:
                error += f"; ledger_close_error={type(ledger_exc).__name__}: {ledger_exc}"
        raise
    finally:
        if not closed:
            try:
                _close_ledger(core, attempt_id, attempt_started, "FAILED_LEDGER_CLOSE_FALLBACK",
                              error or "unexpected control-flow exit", cost_gate)
            except BaseException:
                pass
        _restore_alarm(old_alarm)


def cost_gate_cost(smoke: dict[str, Any]) -> float:
    return float(smoke["cost_projection"]["projected_total_gpu_hours_with_margin_including_smoke"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["freeze", "smoke", "train"], required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    args = parser.parse_args()
    cfg, config_sha = _load_config(args.config)
    core = _load_core()
    out, _feature_cache, protocol_path = _bind_core(core, cfg, config_sha)
    if args.mode == "freeze":
        result = _freeze(core, cfg, config_sha, out, protocol_path)
    else:
        result = _run_gpu_phase(core, cfg, config_sha, out, protocol_path, args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
