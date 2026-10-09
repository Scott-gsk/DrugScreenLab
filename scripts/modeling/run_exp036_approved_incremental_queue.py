"""Run frozen EXP-036/041 patient folds under the shared approved 40-GPU-hour tranche."""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from shared_gpu_ledger import (
    allocation_from_events,
    append_event as shared_append_event,
    budget_snapshot,
    ledger_transaction,
    package_budget_snapshot,
    read_events as shared_read_events,
    release_ledger_transaction,
    total_budget_snapshot,
)


ROOT = Path(__file__).resolve().parents[2]
QUEUE_ID = "EXP037-COND-V2-F34-INCREMENTAL-20260926"
TRANCHE = ROOT / "artifacts/experiments/EXP-037/20260926_conditional_v2_folds3_4_budget_amendment_v2"
LEDGER = TRANCHE / "queue_ledger.jsonl"
LOCK = TRANCHE / "queue.lock"
REALLOCATION = TRANCHE / "priority_reallocation_20260926.json"
SOL_DECISION = ROOT / "artifacts/experiments/EXP-036/20260926_transfer_queue_sol_ruling/decision.json"
SHARED_ALLOCATION_DECISION = ROOT / "artifacts/experiments/EXP-048/20260926_shared_tranche_reallocation_sol_ruling/decision.json"
SHARED_ALLOCATION_REL = "artifacts/experiments/EXP-048/20260926_shared_tranche_reallocation_sol_ruling/decision.json"
RUN_ROOTS = {
    0.25: ROOT / "artifacts/experiments/EXP-036/20260926_pharmaformer_transfer_25pct",
    0.5: ROOT / "artifacts/experiments/EXP-036/20260926_pharmaformer_transfer_50pct",
    1.0: ROOT / "artifacts/experiments/EXP-036/20260926_pharmaformer_transfer_100pct",
}
OLD_RUN_ROOT = ROOT / "artifacts/experiments/EXP-036/20260923_pharmaformer_50pct_v2"
INPUT_DIR = ROOT / "data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1"
SPLIT = ROOT / "data/splits/dual_route_splits_20260922_v1.json"
BASELINE_DIR = ROOT / "artifacts/experiments/EXP-036/20260922_baselines_v1"
CHECKPOINT = ROOT / "data/artifacts/EXP-011/source_cv/fold0/M2_fold0.pt"
OFFICIAL = ROOT / "data/external/exp010/pharmaformer/repo/pretrain.py"
TRAIN_SCRIPT = ROOT / "scripts/modeling/exp036_pharmaformer_fold.py"
MODEL_SCRIPT = ROOT / "src/drug_screen/models/pharmaformer_target.py"
VERIFY_SCRIPT = ROOT / "scripts/evaluation/verify_exp036_pharmaformer_fold.py"
AGGREGATE_SCRIPT = ROOT / "scripts/evaluation/aggregate_exp036_pharmaformer.py"
RUNNER_REL = "scripts/modeling/run_exp036_approved_incremental_queue.py"
LEDGER_HELPER_REL = "scripts/modeling/shared_gpu_ledger.py"
OLD_PROVENANCE = OLD_RUN_ROOT / "run_provenance.json"
EXPECTED_SPLIT_SHA256 = "9bfeeb4e0982421cc188a54e2a65f348709043c05529e641c7dd11ffc0635be8"
EXPECTED_MANIFEST_SHA256 = "c6f9c3349b0a8598280ba099381b781c68ca5599f6f3b7e749090a0e46460b72"
EXPECTED_CHECKPOINT_SHA256 = "1dbe98a2f52c4affdcee091cf011f3e52e29b51992c9bb3d57a60b08fa464de6"
EXPECTED_TRAIN_SHA256 = "b00b422b1acf7aef2f3f89f95fa076cdb26aab5730aed5cde7f2ea16604bd8bf"
EXPECTED_MODEL_SHA256 = "9a8a7e501501e5446b8f5ad5b1d46a8ea0a8121a0650e8d2d26eec8533505ca0"
APPROVED_GPU_UUID = "GPU-9c366c2e-bfa5-c7b3-b16a-1117c40e6871"
PACKAGE_CAP_HOURS = 40.0
RUN_CAP_HOURS = 20.0
CURVE_FRACTIONS = (0.25, 1.0)
CURVE_TOTAL_CAP_HOURS = 12.2  # new remaining first-pass allocation; prior 26h was a ceiling
CURVE_ATTEMPT_CAP_HOURS = {0.25: 0.8, 1.0: 1.8}
CURVE_ATTEMPT_LIMIT_PER_FOLD = 2  # one initial attempt plus one retry
SEEDS = (20260922, 20260923, 20260924)
ARMS = ("random_full", "source_head", "source_full", "source_adapter")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def run_root_for(fraction: float) -> Path:
    try:
        return RUN_ROOTS[fraction]
    except KeyError as exc:
        raise ValueError(f"unsupported frozen label fraction: {fraction}") from exc


def fraction_tag(fraction: float) -> str:
    return {0.25: "025", 0.5: "050", 1.0: "100"}[fraction]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_events() -> list[dict]:
    return shared_read_events(LEDGER, LOCK)


def append_event(event: str, **fields: object) -> None:
    shared_append_event(LEDGER, LOCK, event, **fields)


def charge_state(events: list[dict]) -> tuple[float, dict[str, dict], dict[str, dict]]:
    starts = {str(row["attempt_id"]): row for row in events
              if row.get("event") == "attempt_started"}
    finishes = {str(row["attempt_id"]): row for row in events
                if row.get("event") in {"attempt_finished", "attempt_reconciled"}}
    if len(finishes) != sum(row.get("event") in {"attempt_finished", "attempt_reconciled"}
                            for row in events):
        raise RuntimeError("duplicate attempt close; manual reconciliation required")
    snapshot = budget_snapshot(events)
    spent = float(snapshot["settled_gpu_hours"])
    if not math.isfinite(spent) or spent < 0:
        raise RuntimeError("shared ledger has an invalid charge")
    return spent, starts, finishes


def curve_spend(events: list[dict]) -> float:
    return float(package_budget_snapshot(events, "EXP036041_CURVES")
                 ["settled_after_allocation_gpu_hours"])


def remaining_curve_reserve(events: list[dict]) -> float:
    _, starts, finishes = charge_state(events)
    promoted = {(float(row["fraction"]), int(row["fold"])) for row in events
                if row.get("event") == "fold_promoted" and row.get("fraction") is not None
                and row.get("run_id") in {run_root_for(fraction).name for fraction in CURVE_FRACTIONS}}
    reserve = 0.0
    for fraction in CURVE_FRACTIONS:
        run_id = run_root_for(fraction).name
        for fold in range(5):
            if (fraction, fold) in promoted:
                continue
            matching = [(attempt_id, row) for attempt_id, row in starts.items()
                        if row.get("run_id") == run_id and int(row.get("fold", -1)) == fold]
            attempts = len(matching)
            if attempts >= CURVE_ATTEMPT_LIMIT_PER_FOLD:
                raise RuntimeError(
                    f"fold {fold} at {fraction:g} fraction exhausted its retry reserve without promotion")
            if attempts == 0:
                reserve += CURVE_ATTEMPT_CAP_HOURS[fraction]
            elif not any(attempt_id not in finishes for attempt_id, _ in matching):
                last_id, _ = matching[-1]
                last = finishes[last_id]
                if last.get("outcome") != "EXIT_0":
                    reserve += CURVE_ATTEMPT_CAP_HOURS[fraction]
    return reserve


def assert_frozen_inputs() -> dict[str, str]:
    paths = {
        "data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1/manifest.json": INPUT_DIR / "manifest.json",
        "data/splits/dual_route_splits_20260922_v1.json": SPLIT,
        "artifacts/experiments/EXP-036/20260922_baselines_v1/run_summary.json": BASELINE_DIR / "run_summary.json",
        "artifacts/experiments/EXP-036/20260922_baselines_v1/unit_scores.parquet": BASELINE_DIR / "unit_scores.parquet",
        SHARED_ALLOCATION_REL: SHARED_ALLOCATION_DECISION,
        "data/artifacts/EXP-011/source_cv/fold0/M2_fold0.pt": CHECKPOINT,
        "data/external/exp010/pharmaformer/repo/pretrain.py": OFFICIAL,
        "scripts/modeling/exp036_pharmaformer_fold.py": TRAIN_SCRIPT,
        "src/drug_screen/models/pharmaformer_target.py": MODEL_SCRIPT,
        "scripts/evaluation/verify_exp036_pharmaformer_fold.py": VERIFY_SCRIPT,
        "scripts/evaluation/aggregate_exp036_pharmaformer.py": AGGREGATE_SCRIPT,
        LEDGER_HELPER_REL: ROOT / LEDGER_HELPER_REL,
        RUNNER_REL: Path(__file__).resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError("missing registered input/code: " + ", ".join(missing))
    actual = {name: sha256(path) for name, path in paths.items()}
    fixed = {
        "data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1/manifest.json": EXPECTED_MANIFEST_SHA256,
        "data/splits/dual_route_splits_20260922_v1.json": EXPECTED_SPLIT_SHA256,
        "data/artifacts/EXP-011/source_cv/fold0/M2_fold0.pt": EXPECTED_CHECKPOINT_SHA256,
        "scripts/modeling/exp036_pharmaformer_fold.py": EXPECTED_TRAIN_SHA256,
        "src/drug_screen/models/pharmaformer_target.py": EXPECTED_MODEL_SHA256,
    }
    changed = {name: {"expected": expected, "actual": actual[name]}
               for name, expected in fixed.items() if actual[name] != expected}
    if changed:
        raise RuntimeError(f"frozen EXP-036/041 input/code hash changed: {changed}")
    return actual


def verify_authority(events: list[dict], *, install_reallocation: bool = False) -> tuple[float, dict]:
    opened = [row for row in events if row.get("event") == "queue_opened"]
    if (len(opened) != 1 or opened[0].get("authorized_incremental_tranche_cap_gpu_hours") != PACKAGE_CAP_HOURS
            or opened[0].get("historical_exp037_actual_gpu_hours") != "UNKNOWN_UNRECOVERED"):
        raise RuntimeError("the original 40-GPU-hour tranche record is missing or changed")
    proposal = json.loads((TRANCHE / "proposal.json").read_text(encoding="utf-8"))
    approval = json.loads((TRANCHE / "user_approval.json").read_text(encoding="utf-8"))
    if (proposal.get("proposal_id") != QUEUE_ID
            or proposal.get("status") != "USER_APPROVED_INCREMENTAL_TRANCHE_READY_TO_RUN"
            or sha256(TRANCHE / "proposal.json") != approval.get("proposal_sha256")
            or float(approval.get("approved_incremental_tranche_gpu_hours", 0)) != PACKAGE_CAP_HOURS
            or approval.get("historical_exp037_usage") != "UNKNOWN_UNRECOVERED"
            or approval.get("budget_reset_or_split") is not False):
        raise RuntimeError("the prior user-approved 40-hour tranche binding is invalid")
    amendment = json.loads(REALLOCATION.read_text(encoding="utf-8"))
    decision = json.loads(SOL_DECISION.read_text(encoding="utf-8"))
    if (amendment.get("status") != "USER_AUTHORIZED_PRIORITY_REALLOCATION"
            or amendment.get("existing_queue_id") != QUEUE_ID
            or amendment.get("budget_reset") is not False
            or amendment.get("new_package_created") is not False
            or amendment.get("split_to_bypass_cap") is not False
            or decision.get("status") != "QUEUE_DECISION_FROZEN"
            or decision.get("actual_model_dispatch", {}).get("model") != "gpt-6-sol"
            or decision.get("actual_model_dispatch", {}).get("reasoning_effort") != "medium"):
        raise RuntimeError("user priority amendment or actual Sol/medium decision is missing")
    reallocated = [row for row in events if row.get("event") == "queue_scope_reallocated"]
    spend, starts, _ = charge_state(events)
    if not reallocated and install_reallocation:
        if (abs(spend - amendment["previous_charged_gpu_hours"]) > 1e-9
                or starts.keys() != {str(row["attempt_id"]) for row in events
                                     if row.get("event") == "attempt_finished"}):
            raise RuntimeError("current ledger differs from the approved reallocation starting balance")
        append_event("queue_scope_reallocated",
                     from_scope="EXP-037 folds 3-4 conditional_v2 and state_ablated continuation",
                     to_scope="EXP-036/041 50-percent target-label PharmaFormer transfer folds 2-4; later registered learning curves and approved candidate/fusion work",
                     user_priority_reallocation_path=str(REALLOCATION.relative_to(ROOT)),
                     user_priority_reallocation_sha256=sha256(REALLOCATION),
                     sol_decision_path=str(SOL_DECISION.relative_to(ROOT)),
                     sol_decision_sha256=sha256(SOL_DECISION),
                     charged_before_reallocation_gpu_hours=spend,
                     authorized_tranche_cap_gpu_hours=PACKAGE_CAP_HOURS,
                     historical_exp037_usage="UNKNOWN_UNRECOVERED")
        events = read_events()
        reallocated = [row for row in events if row.get("event") == "queue_scope_reallocated"]
    if (len(reallocated) != 1
            or reallocated[0].get("user_priority_reallocation_sha256") != sha256(REALLOCATION)
            or reallocated[0].get("sol_decision_sha256") != sha256(SOL_DECISION)):
        raise RuntimeError("the shared ledger does not contain the exact approved scope reallocation")
    allocation = allocation_from_events(events)
    shared_decision = json.loads(SHARED_ALLOCATION_DECISION.read_text(encoding="utf-8"))
    expected_allocations = {
        "EXP036041_CURVES": 12.2,
        "EXP048_FIVE_ARM": 14.5,
        "EXP046_PATHWAY": 0.0,
        "SHARED_EMERGENCY": 2.22934338049611,
    }
    if (allocation.get("sol_decision_path") != str(SHARED_ALLOCATION_DECISION.relative_to(ROOT))
            or allocation.get("sol_decision_sha256") != sha256(SHARED_ALLOCATION_DECISION)
            or shared_decision.get("actual_model") != "gpt-6-sol"
            or shared_decision.get("reasoning_effort") != "medium"
            or allocation.get("allocations_gpu_hours") != expected_allocations
            or float(allocation.get("baseline_spent_gpu_hours", -1))
            != float(shared_decision["shared_ledger"]["settled_cumulative_gpu_hours"])):
        raise RuntimeError("the Sol/medium shared allocation event is missing or differs from its receipt")
    spend, _, _ = charge_state(events)
    if spend > PACKAGE_CAP_HOURS + 1e-9:
        raise RuntimeError("shared incremental 40-hour tranche cap would be exceeded")
    return spend, decision


def gpu_preflight() -> dict:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    devices = []
    for row in csv.reader(query.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 5:
            continue
        uuid, name, total, used, utilization = [part.strip() for part in row]
        if uuid == APPROVED_GPU_UUID:
            devices.append({"uuid": uuid, "name": name, "total_mib": int(total),
                            "used_mib": int(used), "utilization_percent": int(utilization)})
    if len(devices) != 1 or "A5000" not in devices[0]["name"]:
        raise RuntimeError("the frozen RTX A5000 UUID is not visible")
    device = devices[0]
    if device["used_mib"] > 512 or device["utilization_percent"] > 5:
        raise RuntimeError(f"the authorized A5000 is already allocated or busy: {device}")
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
                           "--format=csv,noheader"], check=True, capture_output=True, text=True)
    if any(row.strip().startswith(APPROVED_GPU_UUID) for row in apps.stdout.splitlines() if row.strip()):
        raise RuntimeError("a compute process already holds the authorized A5000")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=APPROVED_GPU_UUID)
    probe = ("import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; "
             "n=torch.cuda.get_device_name(0); assert 'A5000' in n, n; "
             "p=torch.cuda.get_device_properties(0); print(n, p.total_memory)")
    result = subprocess.run([sys.executable, "-c", probe], env=env,
                            check=True, capture_output=True, text=True)
    return {**device, "torch_cuda_probe": result.stdout.strip()}


def runtime_gate() -> None:
    if sys.platform != "linux" or not os.environ.get("WSL_DISTRO_NAME"):
        raise RuntimeError("formal training must run in WSL2")
    if os.environ.get("CONDA_DEFAULT_ENV") != "drugscreening-gpu":
        raise RuntimeError("active Conda environment must be drugscreening-gpu")
    if not sys.executable.endswith("/envs/drugscreening-gpu/bin/python"):
        raise RuntimeError(f"unexpected training Python: {sys.executable}")
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi is unavailable")
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise RuntimeError("less than 20 GiB free on the artifact volume")


def no_duplicate_training() -> None:
    own = os.getpid()
    ancestors = {own}
    current = own
    while current > 1:
        try:
            fields = Path(f"/proc/{current}/stat").read_text(encoding="utf-8").split()
            current = int(fields[3])
        except (OSError, ValueError, IndexError):
            break
        ancestors.add(current)
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in ancestors:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if ("scripts/modeling/exp036_pharmaformer_fold.py" in command
                or RUNNER_REL in command) and str(own) not in command:
            matches.append({"pid": int(entry.name), "command": command.strip()})
    if matches:
        raise RuntimeError(f"duplicate EXP-036/041 training/queue process found: {matches}")


def run_provenance(code_hashes: dict[str, str], input_hashes: dict[str, str],
                   fraction: float) -> dict:
    reused_folds = [0, 1] if fraction == 0.5 else []
    new_folds = [2, 3, 4] if fraction == 0.5 else list(range(5))
    return {
        "experiment_id": "EXP-036/041",
        "status": "RUN_PROVENANCE_CAPTURED",
        "captured_at_utc": utc_now(),
        "parent_run_id": ("20260923_pharmaformer_50pct_v2" if fraction == 0.5
                          else "20260926_pharmaformer_transfer_50pct"),
        "run_id": run_root_for(fraction).name,
        "label_fraction": fraction,
        "reused_verified_folds": reused_folds,
        "new_folds": new_folds,
        "queue_id": QUEUE_ID,
        "code_sha256": code_hashes,
        "input_sha256": input_hashes,
        "environment": {
            "platform": "WSL2",
            "conda_env": "drugscreening-gpu",
            "python": sys.version,
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "cuda_device_uuid": APPROVED_GPU_UUID,
            "single_run_cap_gpu_hours": RUN_CAP_HOURS,
            "incremental_tranche_cap_gpu_hours": PACKAGE_CAP_HOURS,
            "historical_exp037_48h_usage": "UNKNOWN_UNRECOVERED"
        },
        "curve_budget_policy": {
            "fractions": list(CURVE_FRACTIONS),
            "package_cap_gpu_hours_in_shared_tranche": CURVE_TOTAL_CAP_HOURS,
            "per_attempt_cap_gpu_hours": {str(key): value
                                           for key, value in CURVE_ATTEMPT_CAP_HOURS.items()},
            "maximum_attempts_per_fold": CURVE_ATTEMPT_LIMIT_PER_FOLD,
            "retries_per_fold": CURVE_ATTEMPT_LIMIT_PER_FOLD - 1,
            "charge_failures_and_retries": True,
            "reuse_verified_50pct_folds": [0, 1, 2, 3, 4],
            "single_attempt_global_cap_gpu_hours": RUN_CAP_HOURS
        }
    }


def prepare(fraction: float) -> dict:
    events = read_events()
    actual_spend, decision = verify_authority(events, install_reallocation=True)
    total_state = total_budget_snapshot(events, PACKAGE_CAP_HOURS)
    remaining = float(total_state["available_gpu_hours"])
    curve_reserve = remaining_curve_reserve(events)
    curve_package_state = package_budget_snapshot(events, "EXP036041_CURVES")
    curve_budget_remaining = float(curve_package_state["remaining_gpu_hours"])
    if curve_reserve > min(remaining, curve_budget_remaining) + 1e-9:
        raise RuntimeError(
            f"complete curve/retry reserve ({curve_reserve:.3f}h) exceeds available budget "
            f"(tranche {remaining:.3f}h, curve cap {curve_budget_remaining:.3f}h)")
    if fraction == 1.0 and any((0.25, fold) not in {
            (float(row.get("fraction", -1)), int(row.get("fold", -1))) for row in events
            if row.get("event") == "fold_promoted"} for fold in range(5)):
        raise RuntimeError("the approved priority requires all five 25-percent folds to be promoted first")
    run_root = run_root_for(fraction)
    code_input_hashes = assert_frozen_inputs()
    code_hashes = {key: code_input_hashes[key] for key in code_input_hashes if key.startswith("scripts/") or key.startswith("src/")}
    input_hashes = {key: code_input_hashes[key] for key in code_input_hashes
                    if key.startswith("data/") or key.startswith("artifacts/")}
    run_root.mkdir(parents=True, exist_ok=True)
    provenance_path = run_root / "run_provenance.json"
    provenance = run_provenance(code_hashes, input_hashes, fraction)
    provenance["budget_before_prepare"] = {
        "shared_tranche_spent_gpu_hours": actual_spend,
        "shared_tranche_remaining_gpu_hours": remaining,
        "historical_48h_usage": "UNKNOWN_UNRECOVERED",
        "curve_package_cap_gpu_hours": CURVE_TOTAL_CAP_HOURS,
        "curve_package_actual_spent_gpu_hours": curve_spend(events),
        "remaining_curve_and_retry_reserve_gpu_hours": curve_reserve
    }
    if provenance_path.exists():
        old = json.loads(provenance_path.read_text(encoding="utf-8"))
        if old.get("code_sha256") != code_hashes or old.get("input_sha256") != input_hashes:
            events = read_events()
            route1_started = [row for row in events if row.get("event") == "attempt_started"
                              and row.get("run_id") == run_root.name]
            old_codes = old.get("code_sha256", {})
            old_inputs = old.get("input_sha256", {})
            changed_code = {key for key in set(old_codes) | set(code_hashes)
                            if old_codes.get(key) != code_hashes.get(key)}
            changed_inputs = {key for key in set(old_inputs) | set(input_hashes)
                              if old_inputs.get(key) != input_hashes.get(key)}
            if route1_started:
                allowed_orchestration_changes = {RUNNER_REL, LEDGER_HELPER_REL}
                allowed_new_budget_inputs = {SHARED_ALLOCATION_REL}
                if (not changed_code
                        or not changed_code.issubset(allowed_orchestration_changes)
                        or changed_inputs != allowed_new_budget_inputs
                        or old_inputs.get(SHARED_ALLOCATION_REL) is not None):
                    raise RuntimeError(
                        "post-fold provenance amendment may add only the single shared-budget ruling and change only queue/ledger orchestration files")
                backup = provenance_path.with_name("run_provenance_pre_parallel_ledger_v1.json")
                if backup.exists():
                    raise RuntimeError("the one allowed parallel-ledger provenance archive already exists")
                shutil.copy2(provenance_path, backup)
                provenance["orchestration_amendment"] = {
                    "reason": "bounded shared-ledger reservations enable the user-authorized second package in parallel",
                    "previous_provenance_path": str(backup.relative_to(ROOT)),
                    "previous_provenance_sha256": sha256(backup),
                    "changed_orchestration_files": sorted(changed_code),
                    "shared_allocation_decision_path": str(SHARED_ALLOCATION_DECISION.relative_to(ROOT)),
                    "shared_allocation_decision_sha256": sha256(SHARED_ALLOCATION_DECISION),
                    "new_budget_authority_input": SHARED_ALLOCATION_REL,
                    "new_budget_authority_input_sha256": input_hashes[SHARED_ALLOCATION_REL],
                    "model_training_code_and_scientific_inputs_unchanged": True,
                }
                provenance["provenance_revision"] = "POST_FOLD0_SHARED_LEDGER_CONTINUATION_V2"
                temp = provenance_path.with_suffix(".json.tmp")
                temp.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                os.replace(temp, provenance_path)
                append_event("run_provenance_amended_after_fold_boundary",
                             run_id=run_root.name, fraction=fraction, completed_fold=0,
                             prior_provenance_path=str(backup.relative_to(ROOT)),
                             prior_provenance_sha256=sha256(backup),
                             current_provenance_sha256=sha256(provenance_path),
                             changed_orchestration_files=sorted(changed_code),
                             sol_decision_path=str(SHARED_ALLOCATION_DECISION.relative_to(ROOT)),
                             sol_decision_sha256=sha256(SHARED_ALLOCATION_DECISION),
                             training_code_and_input_semantics_unchanged=True)
            else:
                backup = provenance_path.with_name("run_provenance_prelaunch_v1.json")
                if not backup.exists():
                    os.replace(provenance_path, backup)
                provenance["supersedes_prelaunch_provenance"] = str(backup.relative_to(ROOT))
                temp = provenance_path.with_suffix(".json.tmp")
                temp.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                os.replace(temp, provenance_path)
                append_event("run_provenance_refreshed_before_first_training",
                             prior_provenance=str(backup.relative_to(ROOT)),
                             prior_provenance_sha256=sha256(backup),
                             current_provenance_sha256=sha256(provenance_path),
                             reason="prelaunch process-ancestor false positive fixed before any model child started")
    else:
        temp = provenance_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, provenance_path)
    if fraction == 0.5:
        for fold in (0, 1):
            target = run_root / f"fold_{fold}"
            source = OLD_RUN_ROOT / f"fold_{fold}"
            if not (source / "verification.json").is_file():
                raise RuntimeError(f"verified reusable outer fold {fold} is missing")
            if target.exists() or target.is_symlink():
                if not target.exists() or target.resolve() != source.resolve():
                    raise RuntimeError(f"reused fold target already exists with another identity: {target}")
            else:
                target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=True)
    return {"status": "PREPARED", "run_root": str(run_root.relative_to(ROOT)),
            "fraction": fraction,
            "reused_folds": [0, 1] if fraction == 0.5 else [],
            "new_folds": [2, 3, 4] if fraction == 0.5 else list(range(5)),
            "charged_gpu_hours": actual_spend,
            "remaining_gpu_hour_cap": remaining,
            "curve_package_cap_gpu_hours": CURVE_TOTAL_CAP_HOURS,
            "curve_package_remaining_gpu_hours": curve_budget_remaining,
            "remaining_curve_and_retry_reserve_gpu_hours": curve_reserve,
            "reserve_fits_current_shared_tranche": curve_reserve <= remaining + 1e-9,
            "code_and_input_sha256": code_input_hashes,
            "sol_priority": decision["decision"]}


def attempt_directory(fold: int, attempt_id: str) -> Path:
    return TRANCHE / "attempts" / f"{attempt_id}"


def start_fold(fold: int, fraction: float) -> int:
    allowed_folds = (2, 3, 4) if fraction == 0.5 else tuple(range(5))
    if fold not in allowed_folds:
        raise ValueError(f"fold {fold} is outside the authorized {fraction:g} label-fraction queue")
    run_root = run_root_for(fraction)
    events = read_events()
    spent, decision = verify_authority(events)
    total_state = total_budget_snapshot(events, PACKAGE_CAP_HOURS)
    remaining = float(total_state["available_gpu_hours"])
    attempt_cap = CURVE_ATTEMPT_CAP_HOURS[fraction] if fraction in CURVE_FRACTIONS else RUN_CAP_HOURS
    if fraction in CURVE_FRACTIONS:
        curve_package_state = package_budget_snapshot(events, "EXP036041_CURVES")
        curve_remaining = float(curve_package_state["remaining_gpu_hours"])
        needed_reserve = remaining_curve_reserve(events)
        if needed_reserve > min(curve_remaining, remaining) + 1e-9:
            raise RuntimeError(
                f"remaining curve/retry reserve ({needed_reserve:.3f}h) no longer fits "
                f"(curve cap {curve_remaining:.3f}h, shared tranche {remaining:.3f}h)")
        if fraction == 1.0 and any((0.25, item) not in {
                (float(row.get("fraction", -1)), int(row.get("fold", -1))) for row in events
                if row.get("event") == "fold_promoted"} for item in range(5)):
            raise RuntimeError("the approved priority requires all five 25-percent folds to be promoted first")
    elif remaining < 0.25:
        raise RuntimeError(f"less than 0.25 GPUh remains in the shared tranche: {remaining}")
    canonical = run_root / f"fold_{fold}"
    if canonical.exists() or canonical.is_symlink():
        raise RuntimeError(f"fold {fold} already has a canonical output")
    # The append-only tranche ledger also records route-two folds with the same
    # numeric fold IDs. Scope promotion checks to this route-one output path.
    canonical_relative = canonical.relative_to(ROOT).as_posix()
    matching_promotion = any(
        row.get("event") == "fold_promoted"
        and int(row.get("fold", -1)) == fold
        and Path(str(row.get("canonical_dir", "")).replace("\\", "/")).as_posix().endswith(canonical_relative)
        for row in events
    )
    if matching_promotion:
        raise RuntimeError(f"fold {fold} was already promoted for this transfer run")
    try:
        runtime_gate()
        no_duplicate_training()
        assert_frozen_inputs()
        device = gpu_preflight()
    except Exception as exc:
        append_event("gpu_preflight_blocked", fold=fold, gpu_process_spawned=False,
                     gpu_hours_charged=0.0, reason=f"{type(exc).__name__}: {exc}",
                      cumulative_incremental_charge_gpu_hours=spent,
                      remaining_incremental_tranche_gpu_hours=PACKAGE_CAP_HOURS - spent,
                      historical_exp037_48h_usage="UNKNOWN_UNRECOVERED")
        raise
    prior = [row for row in events if row.get("event") == "attempt_started"
             and row.get("run_id") == run_root.name and int(row.get("fold", -1)) == fold]
    if fraction in CURVE_FRACTIONS and len(prior) >= CURVE_ATTEMPT_LIMIT_PER_FOLD:
        raise RuntimeError(f"fold {fold} at {fraction:g} fraction used its one allowed retry")
    attempt_id = f"exp036_f{fraction_tag(fraction)}_fold{fold}_attempt{len(prior) + 1:03d}"
    outdir = attempt_directory(fold, attempt_id) / "result"
    log_path = attempt_directory(fold, attempt_id) / "training.log"
    attempt_directory(fold, attempt_id).mkdir(parents=True, exist_ok=False)
    if outdir.exists():
        raise RuntimeError(f"attempt output already exists: {outdir}")
    input_hashes = assert_frozen_inputs()
    frozen = json.loads((run_root / "run_provenance.json").read_text(encoding="utf-8"))
    current_codes = {key: value for key, value in input_hashes.items()
                     if key.startswith("scripts/") or key.startswith("src/")}
    current_inputs = {key: value for key, value in input_hashes.items()
                      if key.startswith("data/") or key.startswith("artifacts/")}
    if frozen.get("code_sha256") != current_codes or frozen.get("input_sha256") != current_inputs:
        raise RuntimeError("current code or input hashes differ from this fraction's frozen run provenance")
    command = [sys.executable, str(TRAIN_SCRIPT), "--input-dir", str(INPUT_DIR),
               "--split", str(SPLIT), "--source-checkpoint", str(CHECKPOINT),
               "--official-source", str(OFFICIAL), "--fold", str(fold),
               "--fraction", str(fraction), "--output-dir", str(outdir),
               "--device", "cuda:0", "--batch-size", "16"]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES=APPROVED_GPU_UUID)
    seconds_cap = min(RUN_CAP_HOURS, attempt_cap, remaining) * 3600.0
    start_mono = time.monotonic()
    start_utc = utc_now()
    append_event("attempt_started", attempt_id=attempt_id, experiment_ids=["EXP-036", "EXP-041"],
                 package_id="EXP036041_CURVES",
                 run_id=run_root.name, fold=fold, fraction=fraction,
                 planned_attempt_cap_gpu_hours=attempt_cap, reserved_gpu_hours=attempt_cap,
                 arms=list(ARMS), seeds=list(SEEDS),
                 gpu_count=1, gpu_uuid=device["uuid"], gpu_name=device["name"],
                 python=sys.executable, conda_env="drugscreening-gpu", device="cuda:0",
                 seconds_cap=seconds_cap, start_utc=start_utc, start_monotonic_seconds=start_mono,
                 input_and_code_hashes=input_hashes, output_dir=str(outdir.relative_to(ROOT)),
                 log_path=str(log_path.relative_to(ROOT)), command=command,
                 preflight=device, cumulative_gpu_hours_before=spent,
                 remaining_gpu_hours_before=remaining,
                 curve_package_cap_gpu_hours=(CURVE_TOTAL_CAP_HOURS
                                              if fraction in CURVE_FRACTIONS else None),
                 curve_package_spent_before_gpu_hours=(curve_spend(events)
                                                       if fraction in CURVE_FRACTIONS else None),
                 historical_exp037_48h_usage="UNKNOWN_UNRECOVERED")
    outcome = "SPAWN_FAILED"
    exit_code = 127
    pid = None
    with log_path.open("wb") as log:
        try:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            pid = child.pid
            append_event("process_spawned", attempt_id=attempt_id, pid=pid,
                         child_process_start_utc=utc_now(), start_monotonic_seconds=time.monotonic())
            release_ledger_transaction(LOCK)
            soft_deadline = start_mono + seconds_cap - 90.0
            while child.poll() is None:
                wait = min(300.0, max(0.2, soft_deadline - time.monotonic()))
                try:
                    child.wait(timeout=wait)
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - start_mono
                    live_events = read_events()
                    live_total = total_budget_snapshot(live_events, PACKAGE_CAP_HOURS)
                    live_curve = package_budget_snapshot(live_events, "EXP036041_CURVES")
                    append_event("progress", attempt_id=attempt_id, pid=pid,
                                 elapsed_seconds=elapsed, gpu_hours_so_far=elapsed / 3600.0,
                                 shared_available_gpu_hours=live_total["available_gpu_hours"],
                                 remaining_gpu_hours=live_total["available_gpu_hours"],
                                 remaining_curve_cap_gpu_hours=live_curve["remaining_gpu_hours"])
                if child.poll() is None and time.monotonic() >= soft_deadline:
                    outcome = "RUN_CAP_STOP"
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=15)
                    break
            exit_code = int(child.returncode)
            if outcome != "RUN_CAP_STOP":
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
            outcome = "SPAWN_OR_SUPERVISOR_FAILURE"
            log.write((f"\nSUPERVISOR_ERROR {type(exc).__name__}: {exc}\n").encode())
            exit_code = 127
    elapsed = time.monotonic() - start_mono
    if elapsed > seconds_cap + 1.0:
        outcome = "CAP_INVARIANT_BREACH_REQUIRES_REVIEW"
    charged = elapsed / 3600.0
    with ledger_transaction(LOCK) as transaction:
        current_events = transaction.read(LEDGER)
        current_total = total_budget_snapshot(current_events, PACKAGE_CAP_HOURS)
        current_package = package_budget_snapshot(current_events, "EXP036041_CURVES")
        current_start = next(row for row in current_events
                             if row.get("event") == "attempt_started"
                             and row.get("attempt_id") == attempt_id)
        own_reserve = float(current_start.get("reserved_gpu_hours",
                                               current_start["planned_attempt_cap_gpu_hours"]))
        remaining_shared_after = (float(current_total["available_gpu_hours"])
                                  + own_reserve - charged)
        remaining_package_after = (float(current_package["remaining_gpu_hours"])
                                   + own_reserve - charged)
        settled_after = float(current_total["settled_gpu_hours"]) + charged
        transaction.append(
            LEDGER, "attempt_finished", attempt_id=attempt_id,
            experiment_ids=["EXP-036", "EXP-041"], package_id="EXP036041_CURVES",
            run_id=run_root.name, fold=fold, fraction=fraction,
            pid=pid, outcome=outcome, exit_code=exit_code,
            ended_utc=utc_now(), elapsed_seconds=elapsed, gpu_count=1,
            gpu_uuid=device["uuid"], gpu_hours_charged=charged,
            incremental_tranche_cumulative_gpu_hours=settled_after,
            remaining_incremental_tranche_gpu_hours=remaining_shared_after,
            remaining_curve_allocation_gpu_hours=remaining_package_after,
            output_dir=str(outdir.relative_to(ROOT)), log_path=str(log_path.relative_to(ROOT)),
            historical_exp037_48h_usage="UNKNOWN_UNRECOVERED")
    print(json.dumps({"attempt_id": attempt_id, "run_id": run_root.name,
                      "fold": fold, "fraction": fraction, "pid": pid, "exit_code": exit_code,
                      "outcome": outcome, "elapsed_seconds": elapsed,
                      "gpu_hours_charged": charged,
                      "cumulative_incremental_gpu_hours": settled_after,
                      "remaining_incremental_gpu_hours": remaining_shared_after,
                      "remaining_curve_allocation_gpu_hours": remaining_package_after,
                      "log": str(log_path.relative_to(ROOT)),
                      "result_dir": str(outdir.relative_to(ROOT))}, ensure_ascii=False))
    return exit_code


def promote_fold(fold: int, fraction: float, attempt_id: str) -> dict:
    events = read_events()
    _, starts, finishes = charge_state(events)
    if attempt_id not in starts or attempt_id not in finishes:
        raise RuntimeError("promotion requires a recorded, completed GPU attempt")
    if (int(starts[attempt_id].get("fold", -1)) != fold
            or float(starts[attempt_id].get("fraction", -1)) != fraction
            or starts[attempt_id].get("run_id") != run_root_for(fraction).name
            or finishes[attempt_id].get("outcome") != "EXIT_0"):
        raise RuntimeError("promotion attempt/fold/exit status mismatch")
    attempt = attempt_directory(fold, attempt_id)
    result = attempt / "result"
    verification = json.loads((result / "verification.json").read_text(encoding="utf-8"))
    summary = json.loads((result / "fold_summary.json").read_text(encoding="utf-8"))
    if (verification.get("status") != "PASS" or int(verification.get("fold", -1)) != fold
            or summary.get("status") != "FOLD_COMPLETE_PENDING_INDEPENDENT_REVIEW"
            or int(summary.get("fold", -1)) != fold
            or float(summary.get("fraction", -1)) != fraction):
        raise RuntimeError("independent fold verification or frozen summary failed")
    run_root = run_root_for(fraction)
    canonical = run_root / f"fold_{fold}"
    if canonical.exists() or canonical.is_symlink():
        raise RuntimeError(f"canonical fold path already exists: {canonical}")
    digest = hashlib.sha256()
    for path in sorted(item for item in result.rglob("*") if item.is_file()):
        digest.update(path.relative_to(result).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    tree_hash = digest.hexdigest()
    os.replace(result, canonical)
    append_event("fold_promoted", attempt_id=attempt_id, fold=fold, fraction=fraction,
                 run_id=run_root.name,
                 canonical_dir=str(canonical.relative_to(ROOT)),
                 output_tree_sha256=tree_hash,
                 verification_sha256=sha256(canonical / "verification.json"),
                 summary_sha256=sha256(canonical / "fold_summary.json"))
    return {"status": "PROMOTED", "fold": fold, "fraction": fraction,
            "canonical_dir": str(canonical.relative_to(ROOT)),
            "output_tree_sha256": tree_hash}


def record_prelaunch_block(reason: str) -> dict:
    events = read_events()
    spent, _ = verify_authority(events)
    append_event("gpu_preflight_blocked", gpu_process_spawned=False,
                 gpu_hours_charged=0.0, reason=reason,
                 cumulative_incremental_charge_gpu_hours=spent,
                 remaining_incremental_tranche_gpu_hours=PACKAGE_CAP_HOURS - spent,
                 historical_exp037_48h_usage="UNKNOWN_UNRECOVERED")
    return {"status": "RECORDED_NO_GPU_CHILD_STARTED", "gpu_hours_charged": 0.0,
            "cumulative_incremental_gpu_hours": spent,
            "reason": reason}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run-fold", type=int, choices=range(5))
    modes.add_argument("--promote-fold", type=int, choices=range(5))
    modes.add_argument("--record-prelaunch-block", action="store_true")
    parser.add_argument("--fraction", type=float, choices=(0.25, 0.5, 1.0), default=0.5)
    parser.add_argument("--attempt-id")
    parser.add_argument("--reason")
    args = parser.parse_args()
    if args.run_fold is not None:
        with ledger_transaction(LOCK):
            if args.attempt_id:
                raise ValueError("attempt IDs are assigned automatically by the shared ledger")
            exit_code = start_fold(args.run_fold, args.fraction)
        sys.exit(exit_code)
    with ledger_transaction(LOCK):
        if args.prepare:
            result = prepare(args.fraction)
        elif args.record_prelaunch_block:
            if not args.reason:
                raise ValueError("--record-prelaunch-block requires --reason")
            result = record_prelaunch_block(args.reason)
        else:
            if not args.attempt_id:
                raise ValueError("--promote-fold requires --attempt-id from the completed ledger event")
            result = promote_fold(args.promote_fold, args.fraction, args.attempt_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
