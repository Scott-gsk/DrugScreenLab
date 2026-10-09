"""Budget-gated, recoverable EXP-048 five-fold training queue.

One attempt covers one outer fold and is capped at 20 wall-clock GPU hours.
The same append-only EXP-037 shared ledger charges every failed attempt and
recovery. Completed fold predictions are verified and reused on resume. The
post-profile continuation gate persists Sol-approved cold-fold hard caps,
the shared-contingency reserve, and full-package calculation before every
formal outer-fold launch.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from functools import lru_cache
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

from shared_gpu_ledger import (
    allocation_from_events,
    budget_snapshot,
    ledger_transaction as _shared_ledger_transaction,
    package_budget_snapshot,
    read_events as _shared_read_events,
)
from exp048_fusion_fold import CANDIDATES as FUSION_CANDIDATES


ROOT = Path(__file__).resolve().parents[2]
QUEUE_ID = "EXP037-COND-V2-F34-INCREMENTAL-20260926"
TRANCHE = ROOT / "artifacts/experiments/EXP-037/20260926_conditional_v2_folds3_4_budget_amendment_v2"
LEDGER = TRANCHE / "queue_ledger.jsonl"
LOCK = TRANCHE / "queue.lock"
TRAIN_SCRIPT = ROOT / "scripts/modeling/exp048_fusion_fold.py"
CONFIG = ROOT / "configs/experiments/EXP-048.yaml"
MODEL = ROOT / "src/drug_screen/models/exp048_fusion.py"
CHECKPOINT = ROOT / "data/artifacts/EXP-011/source_cv/fold0/M2_fold0.pt"
OFFICIAL = ROOT / "data/external/exp010/pharmaformer/repo/pretrain.py"
INPUT_MANIFEST = ROOT / "data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1/manifest.json"
SPLIT = ROOT / "data/splits/dual_route_splits_20260922_v1.json"
SOURCE_RECEIPT = ROOT / "artifacts/experiments/EXP-048/20260926_nine_drug_mcf7_fusion_inputs_v1/receipt.json"
BINDING_RECEIPT = ROOT / "artifacts/experiments/EXP-048/20260926_target_feature_binding_v1/receipt.json"
SOURCE_VECTOR = ROOT / "artifacts/experiments/EXP-048/20260926_nine_drug_mcf7_fusion_inputs_v1/source_reference_perturbation.npy"
CPU_PREFLIGHT = ROOT / "artifacts/experiments/EXP-048/20260926_five_arm_cpu_preflight_v1/receipt.json"
POSTPROFILE_SOL_RULING = ROOT / "artifacts/experiments/EXP-048/20260926_exp048_postprofile_cost_gate_sol_ruling/decision.json"
POSTPROFILE_SOL_RULING_SHA256 = "6d6c4a50c65539c718e13085dd110cb0f53a0a3130c534823bfe14ce40df6acd"
CONTINUATION_COST_SOL_RULING = ROOT / "artifacts/experiments/EXP-048/20260927_postprofile_cost_gate_sol_ruling/decision.json"
CONTINUATION_COST_SOL_RULING_SHA256 = "35c00b89af169b39959bd602a8fd750a92f1fbde2b7da19651b950bd10fd5b69"
FOLD4_PRELAUNCH_SOL_RULING = ROOT / "artifacts/experiments/EXP-048/20260927_fold4_prelaunch_retry_budget_ruling/decision.json"
FOLD4_PRELAUNCH_SOL_RULING_SHA256 = "74a0f615d9281e2f25c96dd6a96c532aa9311ba7dd794565ea25a80d0010aa77"
RUN_ROOT = ROOT / "artifacts/experiments/EXP-048/20260926_fusion_training_v1"
ATTEMPT_ROOT = ROOT / "artifacts/experiments/EXP-048/20260926_fusion_training_v1/attempts"
COST_GATE_ROOT = ROOT / "artifacts/experiments/EXP-048"
EXPECTED = {
    "input_manifest": "c6f9c3349b0a8598280ba099381b781c68ca5599f6f3b7e749090a0e46460b72",
    "split": "9bfeeb4e0982421cc188a54e2a65f348709043c05529e641c7dd11ffc0635be8",
    "source_receipt": "7f70be0fdf1106038463455dd4d2bf25a01b61b140f2b78e9046c60eb24283e2",
    "binding_receipt": "4877ba04d628258033f42770728bce52c56bcf94a2c0335a20464bb80fae6caf",
    "cpu_preflight": "91d2ea62b9935b72b971404e4de303b65d8c5d9af139235cf211c59a740841b9",
}
ARMS = ["F-A", "F-B", "F-F", "F-C", "F-S"]
SEEDS = [20260922, 20260923, 20260924]
PACKAGE_ID = "EXP048_FIVE_ARM"
TRANCHE_CAP_HOURS = 40.0
RUN_CAP_HOURS = 20.0
TARGET_GPU_INDEX = 1
COLD_FOLD_FIRST_ATTEMPT_CAP_GPU_HOURS = 2.3
SINGLE_FULL_FOLD_RETRY_CAP_GPU_HOURS = 2.3
PROTECTED_SHARED_CONTINGENCY_GPU_HOURS = 2.2293433805
SOL_APPROVED_SHARED_HEADROOM_GPU_HOURS = 11.64904329790778


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@lru_cache(maxsize=8192)
def _cached_file_sha256(path_text: str, size: int, modified_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    return _cached_file_sha256(str(resolved), int(stat.st_size), int(stat.st_mtime_ns))


def trainer_manifest_hashes(hashes: dict[str, str]) -> dict[str, str]:
    """Translate queue-bound files into the trainer's immutable run-manifest keys."""
    return {
        "config": hashes["config"],
        "split": hashes["split"],
        "input_manifest": hashes["target_manifest"],
        "source_receipt": hashes["source_receipt"],
        "binding_receipt": hashes["binding_receipt"],
        "cpu_preflight": hashes["cpu_preflight"],
        "source_vector": sha256(SOURCE_VECTOR),
        "source_checkpoint": hashes["checkpoint"],
        "official_source": hashes["official_source"],
        "fusion_model": hashes["fusion_model"],
        "fold_runner": hashes["trainer"],
    }


def classified_non_substantive_failures(
    attempts_by_fold: dict[int, list[tuple[str, dict[str, Any], dict[str, Any] | None]]],
) -> tuple[list[dict[str, Any]], str | None]:
    """Honor the one Sol-reviewed fold-4 prelaunch classification without erasing its charge.

    The attempt remains in the append-only history and package/shared charges. It is
    omitted only from the set of folds requiring a training retry because it exited
    at the Conda environment guard before CUDA/model work and produced no model
    artifacts. Any evidence drift falls back to the ordinary failed-fold gate.
    """
    attempt_id = "exp048_fold4_attempt001"
    rows = [row for row in attempts_by_fold.get(4, []) if row[0] == attempt_id]
    if not rows:
        return [], None
    if len(rows) != 1:
        return [], "fold4_prelaunch_attempt001_has_duplicate_ledger_starts"

    _, start, close = rows[0]
    try:
        if close is None or start.get("fold") != 4 or close.get("fold") != 4:
            raise ValueError("attempt ledger record is open or bound to a different fold")
        if close.get("outcome") != "NONZERO_EXIT" or int(close.get("exit_code", -999)) != 1:
            raise ValueError("attempt001 no longer matches the reviewed nonzero exit")
        charge = float(close.get("gpu_hours_charged", -1.0))
        reviewed_charge = 0.0010509213852777873
        if abs(charge - reviewed_charge) > 1e-12:
            raise ValueError("attempt001 charged GPU hours differ from the Sol-reviewed amount")
        if sha256(FOLD4_PRELAUNCH_SOL_RULING) != FOLD4_PRELAUNCH_SOL_RULING_SHA256:
            raise ValueError("fold4 prelaunch Sol ruling hash mismatch")
        ruling = json.loads(FOLD4_PRELAUNCH_SOL_RULING.read_text(encoding="utf-8"))
        scope = ruling.get("scope", {})
        if (ruling.get("actual_dispatch") != "/root/exp048_fold4_prelaunch_retry_budget_ruling_0927"
                or ruling.get("actual_model") != "gpt-6-sol"
                or ruling.get("reasoning_effort") != "medium"
                or ruling.get("decision") != "CONDITIONAL_GO_WITHIN_EXISTING_CAP"
                or scope.get("failed_attempt_id") != attempt_id
                or scope.get("corrected_attempt_id") != "exp048_fold4_attempt002"
                or scope.get("failure_class") != "orchestration_environment_guard_before_cuda_or_model_execution"
                or float(scope.get("actual_gpu_hours_charged", -1.0)) != charge
                or float(scope.get("corrected_attempt_hard_cap_gpu_hours", -1.0)) != 2.3
                or scope.get("additional_budget_authorized") is not False
                or scope.get("further_fold4_training_retry_after_attempt002") is not False):
            raise ValueError("fold4 prelaunch Sol ruling does not authorize this exact classification")

        attempt_dir = ATTEMPT_ROOT / attempt_id
        log_path = attempt_dir / "training.log"
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if "RuntimeError: run this entrypoint only in WSL2 Conda drugscreening-gpu" not in log_text:
            raise ValueError("attempt001 log no longer proves the reviewed Conda environment-guard exit")
        attempt_files = {p.name for p in attempt_dir.rglob("*") if p.is_file()}
        if attempt_files != {"training.log"}:
            raise ValueError("attempt001 contains unexpected output files")
        fold4_files = {p.relative_to(RUN_ROOT / "fold_4").as_posix()
                       for p in (RUN_ROOT / "fold_4").rglob("*") if p.is_file()}
        if fold4_files - {"run_manifest.json"}:
            raise ValueError("fold4 contains model or prediction artifacts from attempt001")
    except Exception as exc:
        return [], f"fold4_prelaunch_attempt001_not_classified_non_substantive: {type(exc).__name__}: {exc}"

    return [{
        "attempt_id": attempt_id,
        "fold": 4,
        "outcome": "NONZERO_EXIT",
        "gpu_hours_charged": charge,
        "classification": "reviewed_prelaunch_environment_guard_no_cuda_or_model_artifacts",
        "sol_ruling_path": FOLD4_PRELAUNCH_SOL_RULING.relative_to(ROOT).as_posix(),
        "sol_ruling_sha256": FOLD4_PRELAUNCH_SOL_RULING_SHA256,
        "still_charged_to_package_and_shared_ledger": True,
    }], None


def prepare_fold_output_dir(run_dir: Path, fold: int,
                            trainer_hashes: dict[str, str]) -> None:
    """Write the exact trainer manifest, then pre-create every final arm directory."""
    manifest_path = run_dir / "run_manifest.json"
    expected = {
        "experiment_id": "EXP-048", "fold": fold, "target_label_fraction": 0.5,
        "candidate_grid": [asdict(item) for item in FUSION_CANDIDATES],
        "arms": list(ARMS), "seeds": list(SEEDS),
        "input_and_code_sha256": trainer_hashes,
    }
    prior = None
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(prior.get(key) != value for key, value in expected.items()):
            raise ValueError(f"existing EXP-048 run manifest differs from current contract: {manifest_path}")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"nonempty EXP-048 fold directory lacks run manifest: {run_dir}")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {**expected, "created_at_utc": utc_now(), "status": "IN_PROGRESS"}
        temporary = manifest_path.with_name(manifest_path.name + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        os.replace(temporary, manifest_path)
    for arm in ARMS:
        (run_dir / "units" / "final" / arm).mkdir(parents=True, exist_ok=True)


def ledger_transaction():
    """Use the root-maintained common file-lock implementation."""
    return _shared_ledger_transaction(LOCK)


def read_events(tx=None) -> list[dict[str, Any]]:
    return tx.read(LEDGER) if tx is not None else _shared_read_events(LEDGER, LOCK)


def append_event(event: str, *, tx=None, **fields: Any) -> dict[str, Any]:
    if tx is not None:
        return tx.append(LEDGER, event, **fields)
    with ledger_transaction() as transaction:
        return transaction.append(LEDGER, event, **fields)


def _pid_is_live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    proc_cmd = Path(f"/proc/{pid}/cmdline")
    if not proc_cmd.is_file():
        return False
    command = proc_cmd.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
    return bool(command.strip()) and ("exp048_fusion_fold.py" in command
                                      or "exp037_conditional_fold.py" in command
                                      or "exp036_pharmaformer_fold.py" in command)


def verify_live_reservations(events: list[dict[str, Any]], budget: dict[str, Any]) -> None:
    processes = {str(row.get("attempt_id")): row for row in events
                 if row.get("event") == "process_spawned"}
    for active in budget.get("active_attempts", []):
        attempt_id = str(active["attempt_id"])
        process = processes.get(attempt_id)
        if process is None or not _pid_is_live(int(process.get("pid", -1))):
            raise RuntimeError(f"unmatched GPU reservation has no verifiably live process; manual recovery required: {attempt_id}")


def verify_scope(events: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    opened = [row for row in events if row.get("event") == "queue_opened"]
    if (len(opened) != 1
            or float(opened[0].get("authorized_incremental_tranche_cap_gpu_hours", -1)) != TRANCHE_CAP_HOURS
            or opened[0].get("historical_exp037_actual_gpu_hours") != "UNKNOWN_UNRECOVERED"):
        raise RuntimeError("the existing user-approved 40-hour shared tranche record is absent or changed")
    reallocated = [row for row in events if row.get("event") == "queue_scope_reallocated"]
    if len(reallocated) != 1 or reallocated[0].get("budget_reset") is True:
        raise RuntimeError("the original shared-tranche scope reallocation is not uniquely recorded")
    state = budget_snapshot(events)
    verify_live_reservations(events, state)
    allocation = allocation_from_events(events)
    if state["available_gpu_hours"] < 0:
        raise RuntimeError("shared 40-hour tranche is over its cap")
    return state, allocation


def validate_authorization(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("EXP-048 GPU remains blocked until the actual Sol/user authorization receipt exists")
    auth = json.loads(path.read_text(encoding="utf-8"))
    if (auth.get("experiment_id") != "EXP-048"
            or auth.get("status") != "EXISTING_USER_APPROVAL_BOUND_TO_SOL_ALLOCATION"):
        raise RuntimeError("authorization must explicitly bind the complete five-fold EXP-048 package")
    if auth.get("user_authorized") is not True:
        raise RuntimeError("receipt must bind the existing user-approved 2026-09-26 EXP-048 direction")
    if (auth.get("authorization_source") != "current_dialogue_user_direction_2026-09-26"
            or auth.get("new_user_approval_requested") is not False):
        raise RuntimeError("receipt must cite the existing dialogue authorization without claiming a new approval")
    if auth.get("user_authorized_date") != "2026-09-26" or "nine-drug" not in str(auth.get("user_authorized_scope", "")).lower():
        raise RuntimeError("authorization must record the date and nine-drug scope of the current user direction")
    if auth.get("legacy_48h_usage") != "UNKNOWN_UNRECOVERED":
        raise RuntimeError("legacy 48-hour history must remain UNKNOWN_UNRECOVERED")
    if auth.get("budget_reset") is not False or auth.get("budget_split") is not False:
        raise RuntimeError("resetting or splitting the historic budget is forbidden")
    if float(auth.get("shared_tranche_cap_gpu_hours", -1)) != TRANCHE_CAP_HOURS:
        raise RuntimeError("authorization does not bind to the existing shared 40-hour tranche")
    if auth.get("outer_folds") != [0, 1, 2, 3, 4] or auth.get("arms") != ARMS or auth.get("seeds") != SEEDS:
        raise RuntimeError("authorization changes the frozen five-fold, five-arm, three-seed scope")
    allocation = float(auth.get("allocated_gpu_hours", 0.0))
    if not (0.0 < allocation <= TRANCHE_CAP_HOURS):
        raise RuntimeError("Sol must allocate a positive package cap within the existing tranche")
    decision = auth.get("sol_decision", {})
    decision_path = ROOT / str(decision.get("path", ""))
    if not decision_path.is_file() or decision.get("sha256") != sha256(decision_path):
        raise RuntimeError("authorization does not bind the actual Sol/medium ruling by hash")
    ruling = json.loads(decision_path.read_text(encoding="utf-8"))
    if (ruling.get("actual_model") != "gpt-6-sol"
            or ruling.get("reasoning_effort") != "medium"
            or ruling.get("decision") != "AUTHORIZE_BOUNDED_SHARED_TRANCHE_REALLOCATION_WITH_COST_GATES"):
        raise RuntimeError("actual Sol/medium ruling must authorize the existing shared-tranche allocation")
    ruling_allocation = float(ruling.get("allocations_from_current_available_gpu_hours", {})
                              .get("exp048_profile_and_fivefold_combined_max_gpu_hours", -1))
    profile_cap = float(ruling.get("allocations_from_current_available_gpu_hours", {})
                        .get("exp048_first_single_fold_five_arm_profile_max_gpu_hours", -1))
    if ruling_allocation != allocation or profile_cap != float(auth.get("first_fold_profile_cap_gpu_hours", -2)):
        raise RuntimeError("authorization allocation differs from the amount recorded by Sol")
    if ruling.get("shared_ledger", {}).get("legacy_exp037_48h_usage") != "UNKNOWN_UNRECOVERED":
        raise RuntimeError("Sol ruling must retain the unresolved legacy GPU usage as unknown")
    with ledger_transaction() as tx:
        events = read_events(tx)
    allocation_event = allocation_from_events(events)
    allocations = allocation_event.get("allocations_gpu_hours", {})
    if (float(allocation_event.get("authorized_total_gpu_hours", -1)) != TRANCHE_CAP_HOURS
            or float(allocations.get("EXP048_FIVE_ARM", -1)) != allocation
            or allocation_event.get("decision_path", allocation_event.get("sol_decision_path"))
            != str(decision_path.relative_to(ROOT).as_posix())
            or allocation_event.get("decision_sha256", allocation_event.get("sol_decision_sha256"))
            != decision.get("sha256")
            or allocation_event.get("historical_usage") != "UNKNOWN_UNRECOVERED"
            or allocation_event.get("budget_reset") is not False
            or allocation_event.get("budget_split") is not False):
        raise RuntimeError("authorization, Sol ruling, and shared allocation_set do not agree")
    event_binding = auth.get("shared_allocation_event")
    bound_fields = (
        "queue_id", "event", "timestamp_utc", "package_id", "authorized_total_gpu_hours",
        "baseline_spent_gpu_hours", "baseline_reserved_gpu_hours", "available_at_allocation_gpu_hours",
        "allocations_gpu_hours", "sol_decision_path", "sol_decision_sha256", "actual_model",
        "reasoning_effort", "historical_usage", "budget_reset", "budget_split",
    )
    if not isinstance(event_binding, dict) or any(
            event_binding.get(key) != allocation_event.get(key) for key in bound_fields):
        raise RuntimeError("authorization does not bind the exact current shared allocation_set event")
    gpu = auth.get("gpu", {})
    if (int(gpu.get("index", -1)) != TARGET_GPU_INDEX
            or "RTX A5000" not in str(gpu.get("name", ""))
            or not str(gpu.get("uuid", "")).startswith("GPU-")):
        raise RuntimeError("authorization must bind the currently idle second RTX A5000 by UUID")
    return auth


def gpu_preflight(expected_uuid: str) -> dict[str, Any]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    devices = []
    for row in csv.reader(query.stdout.splitlines(), skipinitialspace=True):
        if len(row) == 6:
            devices.append({"index": int(row[0]), "name": row[1].strip(), "uuid": row[2].strip(),
                            "memory_total_mib": int(row[3]), "memory_used_mib": int(row[4]),
                            "utilization_percent": int(row[5])})
    matches = [device for device in devices if device["index"] == TARGET_GPU_INDEX]
    if len(matches) != 1:
        raise RuntimeError("second A5000 device index is unavailable")
    device = matches[0]
    if ("RTX A5000" not in device["name"] or device["uuid"] != expected_uuid
            or device["memory_used_mib"] > 512 or device["utilization_percent"] > 5):
        raise RuntimeError(f"authorized second A5000 is not idle or UUID changed: {device}")
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True)
    known_uuids = {str(item["uuid"]) for item in devices}
    process_mapping: list[dict[str, Any]] = []
    for row in csv.reader(processes.stdout.splitlines(), skipinitialspace=True):
        if len(row) < 3 or row[0].strip() != expected_uuid:
            continue
        try:
            used_memory = float(row[2].strip())
        except ValueError:
            used_memory = None
        if used_memory is not None and used_memory > 0:
            raise RuntimeError("a compute process has allocated memory on the authorized second A5000")
        try:
            process_id = int(row[1].strip())
            os.kill(process_id, 0)
        except ProcessLookupError:
            process_mapping.append({"reported_gpu_uuid": expected_uuid, "pid": row[1].strip(),
                                    "process_alive": False, "used_memory_reported": row[2].strip(),
                                    "treatment": "stale_process_row_ignored"})
            continue
        except (PermissionError, ValueError) as exc:
            raise RuntimeError("could not verify the live compute-process PID under WSL2") from exc
        try:
            environment_path = Path(f"/proc/{process_id}/environ")
            if not environment_path.is_file():
                raise RuntimeError("live compute PID has no readable /proc environment for GPU attribution")
            values = environment_path.read_bytes().split(b"\0")
            environment = dict(item.split(b"=", 1) for item in values if b"=" in item)
            visible = environment.get(b"CUDA_VISIBLE_DEVICES", b"").decode("utf-8", errors="replace")
            tokens = [token.strip() for token in visible.split(",") if token.strip()]
            if (len(tokens) != 1 or not tokens[0].startswith("GPU-")
                    or tokens[0] not in known_uuids):
                raise RuntimeError("live compute PID has a missing, multi-device, or unresolvable CUDA_VISIBLE_DEVICES mask")
            if tokens[0] == expected_uuid:
                raise RuntimeError("a compute process already uses the authorized second A5000")
            process_mapping.append({"reported_gpu_uuid": expected_uuid, "pid": process_id,
                                    "process_alive": True, "used_memory_reported": row[2].strip(),
                                    "cuda_visible_devices": visible, "bound_gpu_uuid": tokens[0],
                                    "treatment": "excluded_nvidia_smi_cross_gpu_alias_by_live_proc_mask"})
        except (OSError, ValueError) as exc:
            raise RuntimeError("could not attribute the compute-process GPU mask under WSL2") from exc
    device["compute_process_mapping_evidence"] = process_mapping
    device["direct_gpu_snapshot_source"] = "nvidia-smi query-gpu index,name,uuid,memory.used,utilization.gpu"
    device["sampled_at_utc"] = utc_now()
    return device


def input_hashes() -> dict[str, str]:
    paths = {
        "queue": Path(__file__).resolve(), "trainer": TRAIN_SCRIPT, "config": CONFIG,
        "fusion_model": MODEL, "checkpoint": CHECKPOINT, "official_source": OFFICIAL,
        "target_manifest": INPUT_MANIFEST, "split": SPLIT, "source_receipt": SOURCE_RECEIPT,
        "binding_receipt": BINDING_RECEIPT, "cpu_preflight": CPU_PREFLIGHT,
        "postprofile_sol_ruling": POSTPROFILE_SOL_RULING,
    }
    if sha256(POSTPROFILE_SOL_RULING) != POSTPROFILE_SOL_RULING_SHA256:
        raise RuntimeError("the EXP-048 post-profile Sol ruling is missing or changed")
    if sha256(CONTINUATION_COST_SOL_RULING) != CONTINUATION_COST_SOL_RULING_SHA256:
        raise RuntimeError("the EXP-048 continuation cost Sol ruling is missing or changed")
    cost_ruling = json.loads(CONTINUATION_COST_SOL_RULING.read_text(encoding="utf-8"))
    hard_caps = cost_ruling.get("hard_caps", {})
    ruling_evidence = cost_ruling.get("evidence", {})
    protected_headroom = (
        float(ruling_evidence.get("shared_available_gpu_hours", -1.0))
        - float(ruling_evidence.get("other_curve_package_remaining_commitment_gpu_hours", -1.0))
        - PROTECTED_SHARED_CONTINGENCY_GPU_HOURS
    )
    if (cost_ruling.get("actual_model") != "gpt-6-sol"
            or cost_ruling.get("reasoning_effort") != "medium"
            or cost_ruling.get("decision") != "CONDITIONAL_GO"
            or float(hard_caps.get("first_attempt_each_fold_1_2_3_4_gpu_hours", -1.0))
            != COLD_FOLD_FIRST_ATTEMPT_CAP_GPU_HOURS
            or float(hard_caps.get("one_full_fold_retry_reserve_gpu_hours", -1.0))
            != SINGLE_FULL_FOLD_RETRY_CAP_GPU_HOURS
            or abs(float(hard_caps.get(
                "shared_available_after_other_curve_commitment_and_contingency_gpu_hours", -1.0
            )) - protected_headroom) > 1e-8
            or abs(protected_headroom - SOL_APPROVED_SHARED_HEADROOM_GPU_HOURS) > 1e-8):
        raise RuntimeError("the bound EXP-048 continuation cap does not match the actual Sol ruling")
    for name, expected in EXPECTED.items():
        path_key = {"input_manifest": "target_manifest", "split": "split",
                    "source_receipt": "source_receipt", "binding_receipt": "binding_receipt",
                    "cpu_preflight": "cpu_preflight"}[name]
        actual = sha256(paths[path_key])
        if actual != expected:
            raise RuntimeError(f"frozen EXP-048 {path_key} hash changed: {actual} != {expected}")
    return {name: sha256(path) for name, path in paths.items()}


def same_model_input_hashes(left: dict[str, str] | None,
                            right: dict[str, str] | None) -> bool:
    """Ignore only the queue wrapper hash when reusing model outputs.

    The queue changed after fold 0 solely to enforce a new resource ruling.
    Model/trainer/input hashes remain exact; the current queue source hash is
    separately bound into every post-profile cost-gate receipt and launch.
    """
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    left_model = {key: value for key, value in left.items() if key != "queue"}
    right_model = {key: value for key, value in right.items() if key != "queue"}
    return left_model == right_model


def complete_fold(fold: int, expected_hashes: dict[str, str]) -> bool:
    fold_dir = RUN_ROOT / f"fold_{fold}"
    summary_path, manifest_path = fold_dir / "fold_summary.json", fold_dir / "run_manifest.json"
    prediction_path = fold_dir / "predictions.parquet"
    if not all(path.is_file() for path in (summary_path, manifest_path, prediction_path)):
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (summary.get("experiment_id") != "EXP-048" or summary.get("fold") != fold
            or summary.get("phase") != "predictions"
            or summary.get("status") != "FOLD_PREDICTIONS_COMPLETE_PENDING_INDEPENDENT_VERIFICATION"
            or summary.get("predictions_sha256") != sha256(prediction_path)
            or manifest.get("fold") != fold
            or manifest.get("status") != summary.get("status")
            or manifest.get("input_and_code_sha256") != summary.get("input_and_code_sha256")):
        raise RuntimeError(f"fold {fold} has an invalid or partial prediction receipt")
    child_hashes = summary.get("input_and_code_sha256", {})
    expected_child_hashes = {
        "config": sha256(CONFIG), "split": sha256(SPLIT), "input_manifest": sha256(INPUT_MANIFEST),
        "source_receipt": sha256(SOURCE_RECEIPT), "binding_receipt": sha256(BINDING_RECEIPT),
        "cpu_preflight": sha256(CPU_PREFLIGHT), "source_checkpoint": sha256(CHECKPOINT),
        "official_source": sha256(OFFICIAL), "fusion_model": sha256(MODEL),
        "fold_runner": sha256(TRAIN_SCRIPT),
    }
    if any(child_hashes.get(name) != digest for name, digest in expected_child_hashes.items()):
        raise RuntimeError(f"fold {fold} input/code identity hashes differ from the frozen queue")
    selection_path = fold_dir / "inner_selection.json"
    if (not selection_path.is_file()
            or summary.get("inner_selection_sha256") != sha256(selection_path)
            or int(summary.get("n_inner_candidate_units", -1)) != len(ARMS) * 18 * 3
            or int(summary.get("n_final_fit_units", -1)) != len(ARMS) * len(SEEDS)):
        raise RuntimeError(f"fold {fold} lacks the complete bound inner-search/final-fit record")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (selection.get("status") != "INNER_SELECTION_COMPLETE" or selection.get("fold") != fold
            or selection.get("input_and_code_sha256") != child_hashes
            or len(selection.get("arms", [])) != len(ARMS)):
        raise RuntimeError(f"fold {fold} inner selection identity differs from its summary")
    selected = {str(row.get("arm")): row for row in selection["arms"]}
    final_rows = summary.get("final_fit_receipts", [])
    expected_slots = {(arm, seed) for arm in ARMS for seed in SEEDS}
    seen_slots: set[tuple[str, int]] = set()
    if len(final_rows) != len(expected_slots):
        raise RuntimeError(f"fold {fold} has an incomplete final-seed receipt list")
    for row in final_rows:
        arm, seed = str(row.get("arm")), int(row.get("seed", -1))
        slot = (arm, seed)
        if slot not in expected_slots or slot in seen_slots:
            raise RuntimeError(f"fold {fold} final-fit arm/seed identity is invalid")
        seen_slots.add(slot)
        if arm not in selected or row.get("candidate") != selected[arm].get("selected_candidate"):
            raise RuntimeError(f"fold {fold} final-fit candidate differs from inner selection")
        checkpoint_path = ROOT / str(row.get("checkpoint", ""))
        unit_path = fold_dir / "units" / "final" / arm / f"seed_{seed}.json"
        if (not checkpoint_path.is_file() or not unit_path.is_file()
                or row.get("checkpoint_sha256") != sha256(checkpoint_path)):
            raise RuntimeError(f"fold {fold} final-fit checkpoint/receipt is missing or changed")
        unit = json.loads(unit_path.read_text(encoding="utf-8"))
        binding = unit.get("unit_binding", {})
        if (unit.get("status") != "UNIT_COMPLETE" or unit.get("model_sha256") != sha256(checkpoint_path)
                or binding.get("arm") != arm or int(binding.get("seed", -1)) != seed
                or int(binding.get("fold", -1)) != fold
                or binding.get("candidate") != row.get("candidate")
                or binding.get("input_and_code_sha256") != child_hashes):
            raise RuntimeError(f"fold {fold} final-fit checkpoint is not resumably bound to this contract")
    if seen_slots != expected_slots:
        raise RuntimeError(f"fold {fold} final-fit seed set is incomplete")
    if (summary.get("outer_observed_labels_or_scores_in_output") is not False
            or summary.get("outer_test_labels_loaded_during_training") is not False
            or summary.get("outer_test_labels_scored") is not False):
        raise RuntimeError(f"fold {fold} does not attest outer-label isolation")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    outer_patients = set(str(value) for value in split["target_folds"][fold]["outer_test_ids"])
    binding = json.loads(BINDING_RECEIPT.read_text(encoding="utf-8"))
    key_mask_path = ROOT / str(binding["target_key_mask"]["path"])
    if not key_mask_path.is_file() or sha256(key_mask_path) != binding["target_key_mask"]["sha256"]:
        raise RuntimeError("fold verification cannot bind the frozen observed-key mask")
    import numpy as np
    import pandas as pd
    key_mask = pd.read_csv(key_mask_path)
    key_mask["patient_id"] = key_mask["patient_id"].astype(str)
    key_mask["drug_id"] = key_mask["drug_id"].astype(str)
    observed = key_mask.loc[key_mask["target_key_present"].astype(bool), ["patient_id", "drug_id"]]
    expected_pairs = set(map(tuple, observed.loc[observed["patient_id"].isin(outer_patients)]
                             .to_numpy(dtype=str)))
    prediction = pd.read_parquet(prediction_path)
    required = {"fold", "arm", "seed", "patient_id", "drug_id", "prediction"}
    if not required.issubset(prediction.columns):
        raise RuntimeError(f"fold {fold} prediction file lacks required keyed columns")
    prediction["patient_id"] = prediction["patient_id"].astype(str)
    prediction["drug_id"] = prediction["drug_id"].astype(str)
    key_columns = ["fold", "arm", "seed", "patient_id", "drug_id"]
    actual_pairs = set(map(tuple, prediction[["patient_id", "drug_id"]]
                           .drop_duplicates().to_numpy(dtype=str)))
    expected_rows = len(expected_pairs) * len(ARMS) * len(SEEDS)
    if (len(outer_patients) == 0 or len(expected_pairs) != len(outer_patients) * 7
            or prediction.duplicated(key_columns).any()
            or actual_pairs != expected_pairs
            or len(prediction) != expected_rows
            or set(prediction["arm"]) != set(ARMS)
            or set(prediction["seed"].astype(int)) != set(SEEDS)
            or set(prediction["patient_id"]) != outer_patients
            or not prediction["fold"].eq(fold).all()
            or not np.isfinite(pd.to_numeric(prediction["prediction"], errors="raise")).all()
            or int(summary.get("outer_test_patient_count", -1)) != len(outer_patients)
            or int(summary.get("outer_test_key_count", -1)) != len(expected_pairs)):
        raise RuntimeError(f"fold {fold} keyed predictions do not match the frozen outer patient/key/arm/seed contract")
    return True


def partial_fold_recovery_status(fold: int, expected_hashes: dict[str, str]) -> dict[str, Any]:
    """Check that an incomplete fold can resume under the exact current contract."""
    fold_dir = RUN_ROOT / f"fold_{fold}"
    if not fold_dir.exists():
        return {"fold": fold, "state": "NEW_OUTPUT_DIRECTORY"}
    if complete_fold(fold, expected_hashes):
        return {"fold": fold, "state": "VERIFIED_COMPLETE_REUSABLE"}
    manifest_path = fold_dir / "run_manifest.json"
    if not manifest_path.is_file():
        if any(fold_dir.iterdir()):
            raise RuntimeError(f"fold {fold} has unbound partial outputs; manual recovery is required")
        return {"fold": fold, "state": "EMPTY_OUTPUT_DIRECTORY"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("experiment_id") != "EXP-048" or manifest.get("fold") != fold
            or manifest.get("input_and_code_sha256", {}).get("fold_runner") != sha256(TRAIN_SCRIPT)
            or manifest.get("input_and_code_sha256", {}).get("fusion_model") != sha256(MODEL)
            or manifest.get("arms") != ARMS or manifest.get("seeds") != SEEDS):
        raise RuntimeError(f"fold {fold} partial outputs are bound to a different identity/protocol")
    return {"fold": fold, "state": "RESUMABLE_MATCHING_MANIFEST",
            "manifest_sha256": sha256(manifest_path)}


def _attempt_number(attempt_id: str) -> int:
    try:
        return int(attempt_id.rsplit("attempt", 1)[1])
    except (IndexError, ValueError):
        return 1


def _other_package_commitments(events: list[dict[str, Any]], allocation_event: dict[str, Any]) -> dict[str, Any]:
    """Protect every non-EXP048 allocation and the shared emergency reserve."""
    commitments: dict[str, Any] = {}
    for package_id, allocated in allocation_event["allocations_gpu_hours"].items():
        if package_id == PACKAGE_ID:
            continue
        snapshot = package_budget_snapshot(events, str(package_id))
        remaining = max(0.0, float(allocated)
                        - float(snapshot["settled_after_allocation_gpu_hours"])
                        - float(snapshot["reserved_gpu_hours"]))
        commitments[str(package_id)] = {
            "allocated_gpu_hours": float(allocated),
            "settled_gpu_hours": float(snapshot["settled_after_allocation_gpu_hours"]),
            "reserved_gpu_hours": float(snapshot["reserved_gpu_hours"]),
            "protected_remaining_gpu_hours": remaining,
        }
    return commitments


def write_cost_gate_receipt(report: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Persist each cost decision as a unique, hash-addressed artifact."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = COST_GATE_ROOT / f"{stamp}_postprofile_cost_gate"
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = output_dir / "receipt.json"
    report["artifact_path"] = receipt_path.relative_to(ROOT).as_posix()
    temporary_path = receipt_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_path, receipt_path)
    return report, sha256(receipt_path)


def estimate_remaining_package(events: list[dict[str, Any]], auth: dict[str, Any],
                               hashes: dict[str, str], *, allow_retry_fold: int | None = None,
                               gpu: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply the Sol-approved full-package envelope without reading outer performance."""
    starts = {str(row["attempt_id"]): row for row in events if row.get("event") == "attempt_started"}
    closes = {str(row["attempt_id"]): row for row in events
              if row.get("event") in {"attempt_finished", "attempt_reconciled"}}
    allocation_event = allocation_from_events(events)
    allocation_position = events.index(allocation_event)
    postallocation_attempt_ids = {
        str(row["attempt_id"]) for row in events[allocation_position + 1:]
        if row.get("event") == "attempt_started" and row.get("package_id") == PACKAGE_ID
    }
    package = package_budget_snapshot(events, PACKAGE_ID)
    shared = budget_snapshot(events)
    verify_live_reservations(events, shared)
    allocation = float(allocation_event["allocations_gpu_hours"][PACKAGE_ID])
    if allocation != float(auth["allocated_gpu_hours"]):
        raise RuntimeError("authorization and shared allocation event disagree at the post-profile gate")

    fold0_rows: list[dict[str, Any]] = []
    fold0_open: list[str] = []
    c0_known = 0.0
    c0_charges_valid = True
    for attempt_id, start in starts.items():
        if (attempt_id not in postallocation_attempt_ids
                or start.get("package_id") != PACKAGE_ID or int(start.get("fold", -1)) != 0):
            continue
        close = closes.get(attempt_id)
        if close is None:
            fold0_open.append(attempt_id)
            charge = None
        else:
            try:
                charge = float(close.get("gpu_hours_charged", math.nan))
            except (TypeError, ValueError):
                charge = math.nan
            if not math.isfinite(charge) or charge < 0:
                c0_charges_valid = False
                charge = None
            else:
                c0_known += charge
        fold0_rows.append({
            "attempt_id": attempt_id,
            "started_utc": start.get("start_utc", start.get("timestamp_utc")),
            "ended_utc": None if close is None else close.get("ended_utc", close.get("timestamp_utc")),
            "gpu_uuid": start.get("gpu_uuid"),
            "gpu_count": start.get("gpu_count"),
            "reserved_gpu_hours_at_start": start.get("reserved_gpu_hours"),
            "outcome": None if close is None else close.get("outcome"),
            "exit_code": None if close is None else close.get("exit_code"),
            "gpu_hours_charged": charge,
            "input_and_code_hashes": start.get("input_and_code_hashes"),
            "fold_summary_sha256": None if close is None else close.get("fold_summary_sha256"),
            "predictions_sha256": None if close is None else close.get("predictions_sha256"),
            "closed": close is not None,
        })

    fold0_verified = False
    fold0_verification_error: str | None = None
    try:
        fold0_verified = complete_fold(0, hashes)
    except Exception as exc:
        fold0_verification_error = f"{type(exc).__name__}: {exc}"
    fold0_dir = RUN_ROOT / "fold_0"
    fold0_summary_path = fold0_dir / "fold_summary.json"
    fold0_manifest_path = fold0_dir / "run_manifest.json"
    fold0_predictions_path = fold0_dir / "predictions.parquet"
    fold0_selection_path = fold0_dir / "inner_selection.json"
    output_hashes = {
        "fold_summary_sha256": sha256(fold0_summary_path) if fold0_summary_path.is_file() else None,
        "run_manifest_sha256": sha256(fold0_manifest_path) if fold0_manifest_path.is_file() else None,
        "inner_selection_sha256": sha256(fold0_selection_path) if fold0_selection_path.is_file() else None,
        "predictions_sha256": sha256(fold0_predictions_path) if fold0_predictions_path.is_file() else None,
    }
    expected_gpu_uuid = str(auth["gpu"]["uuid"])
    reusable_successes = []
    if fold0_verified:
        for row in fold0_rows:
            attempt_id = str(row["attempt_id"])
            start, close = starts[attempt_id], closes.get(attempt_id)
            if (close is None or close.get("outcome") != "EXIT_0"
                    or row.get("gpu_hours_charged") is None
                    or row["gpu_hours_charged"] <= 0
                    or start.get("gpu_uuid") != expected_gpu_uuid
                    or int(start.get("gpu_count", -1)) != 1
                    or start.get("phase") != "predictions"
                    or start.get("arms") != ARMS or start.get("seeds") != SEEDS
                    or int(start.get("candidate_count_per_arm", -1)) != 18
                    or int(start.get("inner_folds", -1)) != 3
                    or start.get("outer_test_labels_loaded_during_training") is not False
                    or not same_model_input_hashes(start.get("input_and_code_hashes"), hashes)
                    or close.get("fold_summary_sha256") != output_hashes["fold_summary_sha256"]
                    or close.get("predictions_sha256") != output_hashes["predictions_sha256"]):
                continue
            reusable_successes.append({
                "attempt_id": attempt_id,
                "ended_utc": row["ended_utc"],
                "gpu_hours_charged": row["gpu_hours_charged"],
                "gpu_uuid": row["gpu_uuid"],
                "fold_summary_sha256": row["fold_summary_sha256"],
                "predictions_sha256": row["predictions_sha256"],
            })
    reusable_successes.sort(key=lambda row: str(row.get("ended_utc", "")))
    selected_profile = reusable_successes[-1] if reusable_successes else None
    t0 = None if selected_profile is None else float(selected_profile["gpu_hours_charged"])
    c0 = None if fold0_open or not c0_charges_valid else c0_known
    profile_cap = float(auth["first_fold_profile_cap_gpu_hours"])
    c0_valid = c0 is not None and 0.0 < c0 <= profile_cap + 1e-8
    t0_valid = t0 is not None and t0 > 0.0 and c0 is not None and t0 <= c0 + 1e-8
    fold0_reusable = bool(fold0_verified and selected_profile is not None and c0_valid and t0_valid)
    # Sol ruled that attempt003 is a cached-resume fragment and cannot set a
    # cold-start cap. These are hard resource caps, not time estimates.
    per_fold_cap = None if not t0_valid else COLD_FOLD_FIRST_ATTEMPT_CAP_GPU_HOURS
    retry_cap = None if not t0_valid else SINGLE_FULL_FOLD_RETRY_CAP_GPU_HOURS

    completion: dict[int, bool] = {}
    verification_errors: dict[str, str] = {}
    for fold in range(5):
        try:
            completion[fold] = complete_fold(fold, hashes)
        except Exception as exc:
            completion[fold] = False
            verification_errors[str(fold)] = f"{type(exc).__name__}: {exc}"
    complete_folds = [fold for fold in range(5) if completion[fold]]
    incomplete_folds = [fold for fold in range(5) if not completion[fold]]

    attempts_by_fold: dict[int, list[tuple[str, dict[str, Any], dict[str, Any] | None]]] = {}
    for attempt_id, start in starts.items():
        if attempt_id not in postallocation_attempt_ids or start.get("package_id") != PACKAGE_ID:
            continue
        fold = int(start.get("fold", -1))
        if fold not in range(5):
            continue
        attempts_by_fold.setdefault(fold, []).append((attempt_id, start, closes.get(attempt_id)))
    non_substantive_failures, non_substantive_error = classified_non_substantive_failures(
        attempts_by_fold
    )
    non_substantive_attempt_ids = {row["attempt_id"] for row in non_substantive_failures}
    active_attempts = [
        {"attempt_id": attempt_id, "fold": fold, "reserved_gpu_hours": start.get("reserved_gpu_hours"),
         "gpu_uuid": start.get("gpu_uuid"), "pid": next((row.get("pid") for row in events
                if row.get("event") == "process_spawned" and row.get("attempt_id") == attempt_id), None)}
        for fold, rows in attempts_by_fold.items() for attempt_id, start, close in rows if close is None
    ]
    failed_incomplete_folds = sorted({
        fold for fold in incomplete_folds if fold > 0
        for attempt_id, _, close in attempts_by_fold.get(fold, [])
        if attempt_id not in non_substantive_attempt_ids
        and close is not None and close.get("outcome") != "EXIT_0"
    })
    retry_starts = []
    for fold, rows in attempts_by_fold.items():
        if fold == 0:
            continue
        for attempt_id, start, _ in rows:
            if (start.get("attempt_kind") == "full_fold_retry" or _attempt_number(attempt_id) > 1):
                retry_starts.append({"attempt_id": attempt_id, "fold": fold,
                                     "started_utc": start.get("start_utc", start.get("timestamp_utc")),
                                     "closed": closes.get(attempt_id) is not None})
    retry_used = bool(retry_starts)
    retry_count_valid = len(retry_starts) <= 1

    formula_first_need = None
    if c0 is not None and per_fold_cap is not None and retry_cap is not None:
        formula_first_need = c0 + 4.0 * per_fold_cap + retry_cap
    first_package_pass = formula_first_need is not None and formula_first_need <= allocation + 1e-8

    fold_caps: dict[str, float] = {}
    retry_reserve_for_plan = 0.0
    retry_planned_fold: int | None = None
    unrecoverable_failed_folds: list[int] = []
    plan_reasons: list[str] = []
    if non_substantive_error:
        plan_reasons.append(non_substantive_error)
    if not fold0_reusable:
        plan_reasons.append("fold0_incomplete_unverified_or_not_reusable")
    if fold0_open:
        plan_reasons.append("fold0_attempt_unmatched_or_unresolved")
    if c0 is None:
        plan_reasons.append("C0_incomplete_or_invalid")
    elif not c0_valid:
        plan_reasons.append("C0_not_within_approved_2_gpu_hour_profile_cap")
    if not t0_valid:
        plan_reasons.append("fold0_successful_reuse_evidence_missing")
    if formula_first_need is not None and not first_package_pass:
        plan_reasons.append("C0_plus_four_cold_fold_caps_plus_retry_exceeds_14_5_gpu_hour_package_cap")
    if per_fold_cap is not None and per_fold_cap > RUN_CAP_HOURS + 1e-8:
        plan_reasons.append("per_fold_hard_cap_exceeds_20_hour_single_attempt_cap")
    if not retry_count_valid:
        plan_reasons.append("single_retry_reserve_was_started_more_than_once")

    if fold0_reusable and t0_valid and retry_count_valid:
        for fold in incomplete_folds:
            if fold == 0:
                continue
            if any(item[0] == fold for item in active_attempts):
                continue  # the full active reservation is already deducted by both ledgers
            if fold in failed_incomplete_folds:
                if retry_used:
                    plan_reasons.append(f"fold_{fold}_failed_and_the_single_retry_reserve_is_already_used")
                    unrecoverable_failed_folds.append(fold)
                    continue
                fold_caps[str(fold)] = float(retry_cap)
                retry_planned_fold = fold
            else:
                fold_caps[str(fold)] = float(per_fold_cap)
        if retry_planned_fold is not None:
            if len(failed_incomplete_folds) != 1:
                plan_reasons.append("more_than_one_incomplete_failed_fold_cannot_fit_one_retry_reserve")
                unrecoverable_failed_folds.extend(
                    fold for fold in failed_incomplete_folds if fold not in unrecoverable_failed_folds
                )
            elif allow_retry_fold != retry_planned_fold:
                plan_reasons.append("failed_fold_requires_explicit_retry_flag_after_cause_review")
            elif retry_starts:
                plan_reasons.append("retry_reserve_is_no_longer_available")
            else:
                # This one fixed hard-cap slot is assigned to the reviewed failed fold.
                retry_reserve_for_plan = 0.0
        elif retry_used:
            retry_reserve_for_plan = 0.0
        else:
            retry_reserve_for_plan = float(retry_cap)

    for fold in range(5):
        if str(fold) in verification_errors:
            plan_reasons.append(f"fold_{fold}_verification_failed")
    if active_attempts:
        plan_reasons.append("an_EXP048_attempt_is_still_reserved_or_unresolved")
    if allow_retry_fold is not None and retry_planned_fold != allow_retry_fold:
        plan_reasons.append("explicit_retry_flag_does_not_match_the_single_failed_incomplete_fold")

    future_demand = (None if unrecoverable_failed_folds
                     else sum(fold_caps.values()) + retry_reserve_for_plan)
    projected_package_total = (None if future_demand is None else
                               float(package["settled_after_allocation_gpu_hours"])
                               + float(package["reserved_gpu_hours"]) + future_demand)
    within_package = (False if future_demand is None else
                      future_demand <= float(package["remaining_gpu_hours"]) + 1e-8)
    other_commitments = _other_package_commitments(events, allocation_event)
    protected_other = sum(float(item["protected_remaining_gpu_hours"])
                          for item in other_commitments.values())
    emergency = other_commitments.get("SHARED_EMERGENCY", {})
    emergency_untouched = (
        abs(float(emergency.get("allocated_gpu_hours", 0.0))
            - PROTECTED_SHARED_CONTINGENCY_GPU_HOURS) <= 1e-8
        and float(emergency.get("settled_gpu_hours", 0.0)) <= 1e-8
        and float(emergency.get("reserved_gpu_hours", 0.0)) <= 1e-8
    )
    if not emergency_untouched:
        plan_reasons.append("shared_emergency_contingency_is_missing_or_already_committed")
    live_headroom_after_other = float(shared["available_gpu_hours"]) - protected_other
    # SHARED_EMERGENCY already includes the 2.229343 GPUh contingency. Apply
    # it once, and cap available headroom at Sol's lower limit so the original
    # 0.149 GPUh slack remains protected.
    shared_headroom_for_exp048 = min(
        live_headroom_after_other,
        SOL_APPROVED_SHARED_HEADROOM_GPU_HOURS,
    )
    within_shared = (False if future_demand is None
                     else future_demand <= shared_headroom_for_exp048 + 1e-8)
    if not within_package:
        plan_reasons.append("current_package_remaining_does_not_cover_unfinished_fold_caps_and_retry")
    if not within_shared:
        plan_reasons.append("shared_remaining_after_other_package_commitments_does_not_cover_EXP048")

    gate = "PASS" if not plan_reasons else "STOP"
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "experiment_id": "EXP-048",
        "package_id": PACKAGE_ID,
        "gate": gate,
        "gate_reasons": plan_reasons,
        "decision_basis": "Sol-approved fixed cold-fold hard caps and protected shared contingency; no outer-test metrics or performance read",
        "sol_decision": {
            "path": CONTINUATION_COST_SOL_RULING.relative_to(ROOT).as_posix(),
            "sha256": CONTINUATION_COST_SOL_RULING_SHA256,
            "decision": "CONDITIONAL_GO",
            "actual_model": "gpt-6-sol",
            "reasoning_effort": "medium",
        },
        "queue_executor_sha256": sha256(Path(__file__).resolve()),
        "fold0_profile": {
            "attempts": fold0_rows,
            "C0_all_fold0_attempts_actual_gpu_hours": c0,
            "C0_known_closed_attempt_charges_gpu_hours": c0_known,
            "C0_is_complete": c0 is not None,
            "C0_approved_cap_gpu_hours": profile_cap,
            "C0_within_approved_cap": c0_valid,
            "t0_verified_reusable_success_gpu_hours": t0,
            "t0_source_attempt": selected_profile,
            "t0_success_candidates": reusable_successes,
            "fold0_complete_verification_pass": fold0_verified,
            "fold0_verification_error": fold0_verification_error,
            "fold0_reuse_eligible": fold0_reusable,
            "output_hashes": output_hashes,
            "open_attempt_ids": fold0_open,
        },
        "cost_formula": {
            "first_continuation_formula": "C0 + 4*2.3 + 2.3 <= 14.5 GPUh, and preserve shared contingency",
            "cold_start_duration_estimate_gpu_hours": None,
            "cold_start_duration_estimate_status": "NOT_RELIABLY_ESTIMABLE_FROM_ONE_CACHED_RESUME_PROFILE",
            "estimated_gpu_hours_per_remaining_first_attempt": None,
            "enforced_hard_stop_gpu_hours_per_remaining_first_attempt": per_fold_cap,
            "one_fold_retry_reserved_gpu_hours": retry_cap if not retry_used else 0.0,
            "retry_slot_assigned_to_failed_fold": retry_planned_fold,
            "retry_slot_remaining_unallocated_gpu_hours": retry_reserve_for_plan,
            "unrecoverable_failed_folds": unrecoverable_failed_folds,
            "retry_reserve_used_or_started": retry_used,
            "retry_attempts": retry_starts,
            "non_substantive_failed_attempts": non_substantive_failures,
            "retry_cap_for_explicit_recovery_gpu_hours": retry_cap,
            "retry_planned_for_fold": retry_planned_fold,
            "retry_flag_present_for_this_gate": allow_retry_fold is not None,
            "explicitly_requested_retry_fold": allow_retry_fold,
            "per_fold_hard_stop_caps_gpu_hours": fold_caps,
            "uncompleted_nonactive_folds": sorted(int(fold) for fold in fold_caps),
            "active_folds_covered_by_existing_reservation": active_attempts,
            "complete_and_reusable_folds": complete_folds,
            "incomplete_folds": incomplete_folds,
            "remaining_unreserved_contract_demand_gpu_hours": future_demand,
            "first_gate_projected_total_gpu_hours": formula_first_need,
            "first_gate_within_14_5_gpu_hour_cap": first_package_pass,
            "current_projected_package_total_gpu_hours": projected_package_total,
            "current_within_package_cap": within_package,
            "current_within_shared_tranche_after_other_commitments": within_shared,
            "protected_shared_contingency_gpu_hours": PROTECTED_SHARED_CONTINGENCY_GPU_HOURS,
            "single_attempt_general_cap_gpu_hours": RUN_CAP_HOURS,
            "every_enforced_attempt_cap_within_general_cap": all(
                float(cap) <= RUN_CAP_HOURS + 1e-8 for cap in fold_caps.values()
            ) and (per_fold_cap is None or per_fold_cap <= RUN_CAP_HOURS + 1e-8),
        },
        "package_budget": {
            "allocation_gpu_hours": float(package["allocation_gpu_hours"]),
            "settled_gpu_hours": float(package["settled_after_allocation_gpu_hours"]),
            "reserved_gpu_hours": float(package["reserved_gpu_hours"]),
            "remaining_gpu_hours_after_settled_and_reservations": float(package["remaining_gpu_hours"]),
            "projected_total_after_full_unfinished_contract_gpu_hours": projected_package_total,
        },
        "shared_budget": {
            "cap_gpu_hours": TRANCHE_CAP_HOURS,
            "allocation_set": {
                "timestamp_utc": allocation_event.get("timestamp_utc"),
                "authorized_total_gpu_hours": allocation_event.get("authorized_total_gpu_hours"),
                "allocations_gpu_hours": allocation_event.get("allocations_gpu_hours"),
                "sol_decision_path": allocation_event.get("sol_decision_path"),
                "sol_decision_sha256": allocation_event.get("sol_decision_sha256"),
                "legacy_usage": allocation_event.get("historical_usage"),
            },
            "settled_gpu_hours": float(shared["settled_gpu_hours"]),
            "reserved_gpu_hours": float(shared["reserved_gpu_hours"]),
            "remaining_gpu_hours_after_all_active_reservations": float(shared["available_gpu_hours"]),
            "other_package_commitments": other_commitments,
            "protected_other_package_commitments_gpu_hours": protected_other,
            "live_headroom_after_other_package_commitments_before_sol_ceiling_gpu_hours": live_headroom_after_other,
            "headroom_for_EXP048_after_protecting_other_packages_gpu_hours": shared_headroom_for_exp048,
            "protected_shared_contingency_gpu_hours": PROTECTED_SHARED_CONTINGENCY_GPU_HOURS,
            "shared_emergency_contingency_untouched": emergency_untouched,
            "headroom_after_other_packages_and_contingency_gpu_hours": shared_headroom_for_exp048,
            "sol_approved_shared_headroom_ceiling_gpu_hours": SOL_APPROVED_SHARED_HEADROOM_GPU_HOURS,
            "remaining_EXP048_contract_need_plus_other_commitments_gpu_hours": (
                None if future_demand is None else future_demand + protected_other),
            "sol_approved_shared_headroom_ceiling_gpu_hours": SOL_APPROVED_SHARED_HEADROOM_GPU_HOURS,
            "shared_emergency_contingency_untouched": emergency_untouched,
            "within_shared_40_gpu_hour_cap": within_shared,
            "legacy_EXP037_48h_usage": "UNKNOWN_UNRECOVERED",
        },
        "verification_errors_by_fold": verification_errors,
        "non_substantive_failed_attempts": non_substantive_failures,
        "uncertainty": {
            "cold_start_duration_estimate": "not reliable from the single cached-resume fold-0 success; 2.3 GPUh is a hard stop, not an expected duration",
            "single_fold_limit": "one fold does not establish future-fold runtime distribution or guarantee completion",
            "stopping_rule": "a fold that fails or reaches its hard stop is charged at actual elapsed single-GPU hours; stop at that fold; no automatic retry",
            "retry_rule": "at most one 2.3 GPUh full-fold retry, only after a verified correctable failure, explicit recovery invocation, and a new full-contract review",
        },
        "outer_metrics_read": False,
        "outer_metrics_written": False,
        "input_and_code_sha256": hashes,
        "gpu_preflight": gpu,
    }
    report, receipt_sha256 = write_cost_gate_receipt(report)
    report["artifact_sha256"] = receipt_sha256
    return report


def next_attempt_id(events: list[dict[str, Any]], fold: int) -> tuple[str, bool]:
    attempts = [row for row in events if row.get("event") == "attempt_started"
                and "EXP-048" in row.get("experiment_ids", []) and int(row.get("fold", -1)) == fold]
    failed_previously = False
    if attempts:
        finished = {str(row["attempt_id"]): row for row in events
                    if row.get("event") in {"attempt_finished", "attempt_reconciled"}}
        latest = sorted(attempts, key=lambda row: int(str(row["attempt_id"]).rsplit("attempt", 1)[1]))[-1]
        latest_end = finished.get(str(latest["attempt_id"]))
        if latest_end is None:
            raise RuntimeError(f"fold {fold} has an unmatched reserved attempt; manual recovery is required")
        failed_previously = latest_end.get("outcome") != "EXIT_0"
    number = len(attempts) + 1
    return f"exp048_fold{fold}_attempt{number:03d}", failed_previously


def run_attempt(fold: int, attempt_id: str, auth_path: Path, auth: dict[str, Any],
                device: dict[str, Any], *, hashes: dict[str, str],
                profile_fold0: bool, attempt_cap_gpu_hours: float | None = None,
                required_remaining_gpu_hours: float | None = None,
                attempt_kind: str = "full_fold",
                cost_gate: dict[str, Any] | None = None,
                retry_reserve_gpu_hours: float = 0.0) -> tuple[int, float, dict[str, Any]]:
    fold_dir = RUN_ROOT / f"fold_{fold}"
    attempt_dir = ATTEMPT_ROOT / attempt_id
    log_path = attempt_dir / "training.log"
    rel = lambda path: path.relative_to(ROOT).as_posix()
    auth_sha = sha256(auth_path)
    command = [sys.executable, str(TRAIN_SCRIPT), "--fold", str(fold), "--output-dir", str(fold_dir),
               "--phase", "predictions", "--device", "cuda:0", "--resume"]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["CUDA_VISIBLE_DEVICES"] = str(auth["gpu"]["uuid"])
    env["EXP048_APPROVED_QUEUE_ATTEMPT_ID"] = attempt_id
    env["EXP048_BUDGET_AUTHORIZATION_SHA256"] = auth_sha
    gate_binding_base: dict[str, Any] | None = None
    if fold > 0:
        if cost_gate is None or cost_gate.get("gate") != "PASS":
            raise RuntimeError("a passing Sol-bound persisted cost gate is required before every formal fold start")
        receipt_path = ROOT / str(cost_gate.get("artifact_path", ""))
        if (not receipt_path.is_file()
                or sha256(receipt_path) != cost_gate.get("artifact_sha256")):
            raise RuntimeError("post-profile cost-gate receipt is missing or its hash changed before launch")
        persisted_gate = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (persisted_gate.get("gate") != "PASS"
                or persisted_gate.get("sol_decision", {}).get("sha256") != CONTINUATION_COST_SOL_RULING_SHA256
                or persisted_gate.get("queue_executor_sha256") != sha256(Path(__file__).resolve())
                or persisted_gate.get("input_and_code_sha256") != hashes):
            raise RuntimeError("persisted post-profile cost gate is not bound to current code and Sol ruling")
        persisted_cap = persisted_gate.get("cost_formula", {}).get(
            "per_fold_hard_stop_caps_gpu_hours", {}).get(str(fold))
        if persisted_cap is None or abs(float(persisted_cap) - float(attempt_cap_gpu_hours or 0.0)) > 1e-8:
            raise RuntimeError("attempt hard-stop does not match the persisted cost gate for this fold")
        persisted_need = persisted_gate.get("cost_formula", {}).get(
            "remaining_unreserved_contract_demand_gpu_hours")
        if (persisted_need is None or required_remaining_gpu_hours is None
                or abs(float(persisted_need) - float(required_remaining_gpu_hours)) > 1e-8):
            raise RuntimeError("live full-contract reservation request differs from the persisted cost gate")
        gate_binding_base = {"path": str(cost_gate["artifact_path"]),
                             "sha256": str(cost_gate["artifact_sha256"]),
                             "sol_decision_sha256": CONTINUATION_COST_SOL_RULING_SHA256,
                             "queue_executor_sha256": sha256(Path(__file__).resolve()),
                             "remaining_contract_demand_gpu_hours": required_remaining_gpu_hours}
    outcome, exit_code, pid = "SPAWN_FAILED", 127, None
    start_mono: float | None = None
    cap = 0.0
    attempt_dir.mkdir(parents=True, exist_ok=False)
    # Check, reserve, start the child, and record its PID in one short shared
    # ledger transaction. Release the lock while the fold runs.
    with log_path.open("wb") as log:
        with ledger_transaction() as tx:
            events = read_events(tx)
            state, allocation_event = verify_scope(events)
            remaining_shared = float(state["available_gpu_hours"])
            verify_live_reservations(events, state)
            allocation = float(allocation_event["allocations_gpu_hours"]["EXP048_FIVE_ARM"])
            package_id = "EXP048_FIVE_ARM"
            package_state = package_budget_snapshot(events, package_id)
            spent_package = package_state["settled_after_allocation_gpu_hours"]
            reserved_package = package_state["reserved_gpu_hours"]
            remaining_package = package_state["remaining_gpu_hours"]
            other_commitments = _other_package_commitments(events, allocation_event)
            protected_other = sum(float(item["protected_remaining_gpu_hours"])
                                  for item in other_commitments.values())
            emergency_commitment = other_commitments.get("SHARED_EMERGENCY", {})
            emergency_untouched = (
                abs(float(emergency_commitment.get("allocated_gpu_hours", 0.0))
                    - PROTECTED_SHARED_CONTINGENCY_GPU_HOURS) <= 1e-8
                and float(emergency_commitment.get("settled_gpu_hours", 0.0)) <= 1e-8
                and float(emergency_commitment.get("reserved_gpu_hours", 0.0)) <= 1e-8
            )
            if fold == 0 and profile_fold0:
                starts = {str(row["attempt_id"]): row for row in events if row.get("event") == "attempt_started"}
                closed = {str(row["attempt_id"]): row for row in events
                          if row.get("event") in {"attempt_finished", "attempt_reconciled"}}
                fold0_spent = sum(
                    float(closed[item]["gpu_hours_charged"])
                    for item, started in starts.items()
                    if item in closed and started.get("package_id") == package_id
                    and int(started.get("fold", -1)) == 0)
                fold0_reserved = sum(
                    float(started.get("reserved_gpu_hours", started.get("planned_attempt_cap_gpu_hours", 0.0)))
                    for item, started in starts.items()
                    if item not in closed and started.get("package_id") == package_id
                    and int(started.get("fold", -1)) == 0)
                profile_remaining = float(auth["first_fold_profile_cap_gpu_hours"]) - fold0_spent - fold0_reserved
                cap = min(RUN_CAP_HOURS, remaining_shared, remaining_package, profile_remaining)
            else:
                cap = min(RUN_CAP_HOURS, remaining_shared, remaining_package)
            if required_remaining_gpu_hours is not None and (
                    remaining_package + 1e-8 < required_remaining_gpu_hours
                    or not emergency_untouched
                    or min(remaining_shared - protected_other,
                           SOL_APPROVED_SHARED_HEADROOM_GPU_HOURS) + 1e-8
                    < required_remaining_gpu_hours):
                log.close()
                log_path.unlink(missing_ok=True)
                attempt_dir.rmdir()
                raise RuntimeError("live shared/package headroom after other package commitments no longer covers the full remaining-fold contract")
            if (required_remaining_gpu_hours is None
                    and remaining_shared + 1e-8 < cap + protected_other):
                log.close()
                log_path.unlink(missing_ok=True)
                attempt_dir.rmdir()
                raise RuntimeError("live shared headroom after other package commitments cannot reserve this attempt")
            if attempt_cap_gpu_hours is not None:
                if cap + 1e-8 < attempt_cap_gpu_hours:
                    log.close()
                    log_path.unlink(missing_ok=True)
                    attempt_dir.rmdir()
                    raise RuntimeError("live headroom is below the estimated cap for this complete fold")
                cap = min(cap, attempt_cap_gpu_hours)
            if cap <= 0:
                log.close()
                log_path.unlink(missing_ok=True)
                attempt_dir.rmdir()
                raise RuntimeError("no GPU-hour headroom remains in shared, package, or first-fold profile cap")
            gate_binding: dict[str, Any] | None = None
            if fold > 0 and gate_binding_base is not None:
                gate_binding = {**gate_binding_base,
                                "live_other_package_commitments_gpu_hours": protected_other}
            seconds_cap = cap * 3600.0
            start_mono = time.monotonic()
            start_utc = utc_now()
            append_event("attempt_started", tx=tx, attempt_id=attempt_id, package_id=package_id,
                         experiment_ids=["EXP-048"], run_id="20260926_fusion_training_v1",
                         fold=fold, fraction=0.5, phase="predictions", arms=ARMS, seeds=SEEDS,
                         candidate_count_per_arm=18, inner_folds=3,
                         attempt_kind=("fold0_profile" if profile_fold0 else attempt_kind),
                         retry_reserve_gpu_hours=float(retry_reserve_gpu_hours),
                         postprofile_cost_gate=gate_binding,
                         planned_attempt_cap_gpu_hours=cap, reserved_gpu_hours=cap,
                         package_allocation_gpu_hours=allocation, gpu_count=1,
                         gpu_index=device["index"], gpu_uuid=device["uuid"], gpu_name=device["name"],
                         conda_env="drugscreening-gpu", python=sys.executable, device="cuda:0",
                         seconds_cap=seconds_cap, start_utc=start_utc, start_monotonic_seconds=start_mono,
                         input_and_code_hashes=hashes, authorization_path=rel(auth_path), authorization_sha256=auth_sha,
                         output_dir=rel(fold_dir), attempt_dir=rel(attempt_dir), log_path=rel(log_path), command=command,
                         preflight=device, cumulative_gpu_hours_before=state["settled_gpu_hours"],
                         committed_gpu_hours_before=(state["settled_gpu_hours"]
                                                     + state["reserved_gpu_hours"]),
                         remaining_shared_gpu_hours_before=remaining_shared,
                         exp048_package_gpu_hours_before=spent_package + reserved_package,
                         remaining_exp048_package_gpu_hours_before=remaining_package,
                         historical_exp037_48h_usage="UNKNOWN_UNRECOVERED",
                         outer_test_predictions_written=True, outer_test_labels_loaded_during_training=False,
                         outer_test_labels_scored=False)
            try:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                pid = child.pid
                append_event("process_spawned", tx=tx, attempt_id=attempt_id, package_id=package_id,
                             pid=pid, process_start_utc=utc_now(),
                             start_monotonic_seconds=time.monotonic(), command=command)
            except Exception:
                # The reserved start is intentionally left open if the child
                # could not be durably registered. Manual recovery must use
                # evidence; an empty start is never closed or charged as zero.
                raise
        try:
            soft_deadline = start_mono + max(1.0, seconds_cap - 90.0)
            while child.poll() is None:
                wait_seconds = min(300.0, max(0.2, soft_deadline - time.monotonic()))
                try:
                    child.wait(timeout=wait_seconds)
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - start_mono
                    with ledger_transaction() as tx:
                        append_event("progress", tx=tx, attempt_id=attempt_id, package_id="EXP048_FIVE_ARM",
                                     pid=pid, fold=fold, elapsed_seconds=elapsed,
                                     gpu_hours_so_far=elapsed / 3600.0, heartbeat_utc=utc_now())
                if child.poll() is None and time.monotonic() >= soft_deadline:
                    outcome = "ATTEMPT_CAP_STOP"
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=15)
                    break
            exit_code = int(child.returncode)
            if outcome != "ATTEMPT_CAP_STOP":
                outcome = "EXIT_0" if exit_code == 0 else "NONZERO_EXIT"
        except KeyboardInterrupt:
            outcome = "INTERRUPTED"
            if pid is not None:
                try:
                    os.killpg(pid, signal.SIGTERM)
                    time.sleep(2)
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            exit_code = 130
        except Exception as exc:
            log.write((f"\nQUEUE_ERROR {type(exc).__name__}: {exc}\n").encode())
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=15)
                except ProcessLookupError:
                    pass
                if child.poll() is None:
                    # Preserve the reservation if the supervisor cannot prove
                    # that the GPU child has stopped.
                    raise RuntimeError("could not confirm the EXP-048 child has stopped") from exc
            outcome, exit_code = "SUPERVISOR_FAILURE", 127
    elapsed = time.monotonic() - start_mono
    if elapsed > seconds_cap + 1.0:
        outcome = "CAP_INVARIANT_BREACH_REQUIRES_REVIEW"
    charged = elapsed / 3600.0
    if outcome == "EXIT_0" and not complete_fold(fold, hashes):
        outcome, exit_code = "EXIT_0_BUT_FOLD_RECEIPT_INVALID", 1
    fold_summary = fold_dir / "fold_summary.json"
    with ledger_transaction() as tx:
        append_event("attempt_finished", tx=tx, attempt_id=attempt_id, package_id="EXP048_FIVE_ARM",
                     experiment_ids=["EXP-048"], run_id="20260926_fusion_training_v1",
                     fold=fold, fraction=0.5, phase="predictions", pid=pid,
                     attempt_kind=("fold0_profile" if profile_fold0 else attempt_kind),
                     outcome=outcome, exit_code=exit_code, ended_utc=utc_now(),
                     elapsed_seconds=elapsed, gpu_count=1, gpu_index=device["index"],
                     gpu_uuid=device["uuid"], reserved_gpu_hours=0.0, gpu_hours_charged=charged,
                     output_dir=rel(fold_dir), attempt_dir=rel(attempt_dir), log_path=rel(log_path),
                     historical_exp037_48h_usage="UNKNOWN_UNRECOVERED",
                     fold_summary_sha256=sha256(fold_summary) if fold_summary.is_file() else None,
                     predictions_sha256=sha256(fold_dir / "predictions.parquet")
                     if (fold_dir / "predictions.parquet").is_file() else None)
        final_events = read_events(tx)
        final_state = budget_snapshot(final_events)
        final_package_state = package_budget_snapshot(final_events, "EXP048_FIVE_ARM")
    receipt = {"attempt_id": attempt_id, "fold": fold, "outcome": outcome, "exit_code": exit_code,
               "elapsed_seconds": elapsed, "gpu_hours_charged": charged,
               "shared_tranche_committed_gpu_hours": final_state["settled_gpu_hours"] + final_state["reserved_gpu_hours"],
               "shared_tranche_remaining_gpu_hours": final_state["available_gpu_hours"],
               "exp048_package_committed_gpu_hours": final_package_state["settled_after_allocation_gpu_hours"]
               + final_package_state["reserved_gpu_hours"],
               "exp048_package_allocation_gpu_hours": float(auth["allocated_gpu_hours"]),
               "log_path": rel(log_path), "fold_dir": rel(fold_dir),
               "fold_summary": rel(fold_summary) if fold_summary.is_file() else None}
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return exit_code, charged, receipt


def run_queue(args: argparse.Namespace) -> int:
    auth_path = args.authorization_json.resolve()
    auth = validate_authorization(auth_path)
    with ledger_transaction() as tx:
        events = read_events(tx)
        state, allocation_event = verify_scope(events)
        verify_live_reservations(events, state)
        package_state = package_budget_snapshot(events, PACKAGE_ID)
    allocation = float(allocation_event["allocations_gpu_hours"]["EXP048_FIVE_ARM"])
    if allocation != float(auth["allocated_gpu_hours"]):
        raise RuntimeError("authorization and shared allocation event disagree")
    hashes = input_hashes()
    device = gpu_preflight(str(auth["gpu"]["uuid"]))
    if args.folds is None:
        requested_folds = [0]
    else:
        requested_folds = list(args.folds)
    if len(set(requested_folds)) != len(requested_folds) or not requested_folds:
        raise ValueError("fold selection must be nonempty and contain no duplicates")
    for fold in requested_folds:
        if fold not in range(5):
            raise ValueError(f"invalid outer fold: {fold}")

    fold0_attempts = [row for row in events if row.get("event") == "attempt_started"
                      and row.get("package_id") == "EXP048_FIVE_ARM" and int(row.get("fold", -1)) == 0]
    if args.continue_after_profile:
        if not fold0_attempts:
            raise RuntimeError("the <=2h fold-0 cost profile must run before full-five-fold continuation")
        if not complete_fold(0, hashes) and 0 not in requested_folds:
            raise RuntimeError("fold 0 is incomplete; include it to recover before running later folds")
    elif any(fold != 0 for fold in requested_folds):
        raise RuntimeError("later folds require --continue-after-profile after fold-0 cost review")

    if args.estimate_after_profile:
        if not args.continue_after_profile:
            raise ValueError("cost estimation requires --continue-after-profile to confirm the profile phase")
        estimate = estimate_remaining_package(events, auth, hashes, gpu=device)
        print(json.dumps({"status": "EXP048_COMPLETION_COST_GATE_" + estimate["gate"],
                          "receipt_path": estimate["artifact_path"],
                          "receipt_sha256": estimate["artifact_sha256"],
                          "gate_reasons": estimate["gate_reasons"],
                          "package_budget": estimate["package_budget"],
                          "shared_budget": estimate["shared_budget"],
                          "cost_formula": estimate["cost_formula"]},
                         ensure_ascii=False, indent=2), flush=True)
        return 0 if estimate["gate"] == "PASS" else 2

    if args.preflight_only:
        recovery = []
        for fold in requested_folds:
            recovery.append(partial_fold_recovery_status(fold, hashes))
        attempts = []
        for fold in requested_folds:
            fold_events = [row for row in events if row.get("event") == "attempt_started"
                           and row.get("package_id") == PACKAGE_ID
                           and int(row.get("fold", -1)) == fold]
            _, failed_before = next_attempt_id(events, fold)
            attempts.append({"fold": fold, "attempt_count": len(fold_events),
                             "resume_after_failure_required": failed_before})
        report = {
            "status": "PREFLIGHT_PASS_NO_GPU_TRAINING_STARTED",
            "requested_folds": requested_folds,
            "profile_cap_gpu_hours": auth["first_fold_profile_cap_gpu_hours"],
            "package_budget": package_state,
            "shared_settled_gpu_hours": state["settled_gpu_hours"],
            "shared_reserved_gpu_hours": state["reserved_gpu_hours"],
            "shared_available_gpu_hours": state["available_gpu_hours"],
            "gpu": device,
            "fold_recovery": recovery,
            "prior_attempts": attempts,
            "input_and_code_sha256": hashes,
            "outer_metrics_printed": False,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0

    completed = []
    for fold in requested_folds:
        if complete_fold(fold, hashes):
            completed.append(fold)
            print(json.dumps({"fold": fold, "status": "REUSED_VERIFIED_COMPLETE_PREDICTIONS"}), flush=True)
    for fold in requested_folds:
        if fold in completed:
            continue
        with ledger_transaction():
            events = read_events()
        attempt_id, failed_previously = next_attempt_id(events, fold)
        if fold == 0 and _attempt_number(attempt_id) > 3:
            raise RuntimeError("the Sol-authorized fold-0 profile/recovery attempts are exhausted; do not retry without a new decision")
        attempt_cap, required_remaining = None, None
        cost_gate = None
        attempt_kind = "fold0_profile" if fold == 0 else "full_fold"
        retry_reserve_gpu_hours = 0.0
        if fold > 0:
            with ledger_transaction() as tx:
                latest_events = read_events(tx)
                latest_state, _ = verify_scope(latest_events)
                verify_live_reservations(latest_events, latest_state)
            estimate = estimate_remaining_package(
                latest_events, auth, hashes,
                allow_retry_fold=fold if failed_previously and args.resume_after_failure else None,
                gpu=device,
            )
            if estimate["gate"] != "PASS":
                print(json.dumps({"status": "STOP_BEFORE_FURTHER_GPU_START",
                                  "receipt_path": estimate["artifact_path"],
                                  "receipt_sha256": estimate["artifact_sha256"],
                                  "gate_reasons": estimate["gate_reasons"]},
                                 ensure_ascii=False, indent=2), flush=True)
                return 2
            cap_value = estimate["cost_formula"]["per_fold_hard_stop_caps_gpu_hours"].get(str(fold))
            if cap_value is None:
                raise RuntimeError(f"passing cost-gate receipt has no hard-stop allocation for fold {fold}")
            attempt_cap = float(cap_value)
            required_remaining = float(estimate["cost_formula"]["remaining_unreserved_contract_demand_gpu_hours"])
            attempt_kind = "full_fold_retry" if estimate["cost_formula"]["retry_planned_for_fold"] == fold else "full_fold"
            retry_reserve_gpu_hours = (float(estimate["cost_formula"]["retry_cap_for_explicit_recovery_gpu_hours"])
                                       if attempt_kind == "full_fold_retry" else 0.0)
            cost_gate = estimate
        profile_fold0 = fold == 0
        prepare_fold_output_dir(RUN_ROOT / f"fold_{fold}", fold,
                                trainer_manifest_hashes(hashes))
        code, _, _ = run_attempt(fold, attempt_id, auth_path, auth, device,
                                 hashes=hashes, profile_fold0=profile_fold0,
                                 attempt_cap_gpu_hours=attempt_cap,
                                 required_remaining_gpu_hours=required_remaining,
                                 attempt_kind=attempt_kind, cost_gate=cost_gate,
                                 retry_reserve_gpu_hours=retry_reserve_gpu_hours)
        if code != 0:
            return code
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization-json", type=Path, required=True,
                        help="receipt binding the 2026-09-26 user authorization and actual Sol/medium ruling")
    parser.add_argument("--folds", type=int, nargs="+", choices=range(5), default=None,
                        help="folds to run; default runs the <=2 GPU-hour fold-0 cost profile only")
    parser.add_argument("--continue-after-profile", action="store_true",
                        help="continue within the Sol-approved 14.5h package after reviewing fold-0 cost")
    parser.add_argument("--resume-after-failure", action="store_true",
                        help="explicitly resume a charged failed/interrupted fold; no budget is reset")
    parser.add_argument("--preflight-only", action="store_true",
                        help="validate authorization, ledger, inputs, GPU UUID, and fold recovery without starting training")
    parser.add_argument("--estimate-after-profile", action="store_true",
                        help="write a Sol-bound cold-fold-cap full-package cost-gate receipt; never starts training")
    args = parser.parse_args()
    sys.exit(run_queue(args))


if __name__ == "__main__":
    main()
