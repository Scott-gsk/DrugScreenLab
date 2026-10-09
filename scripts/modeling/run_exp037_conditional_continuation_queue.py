"""User-authorized incremental single-GPU runner for EXP-037 conditional folds 3-4."""

from __future__ import annotations

import csv
import fcntl
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


ROOT = Path(__file__).resolve().parents[2]
QUEUE_ID = "EXP037-COND-V2-F34-INCREMENTAL-20260926"
PACKAGE_DIR = ROOT / "artifacts/experiments/EXP-037/20260926_conditional_v2_folds3_4_budget_amendment_v2"
PROPOSAL_PATH = PACKAGE_DIR / "proposal.json"
APPROVAL_PATH = PACKAGE_DIR / "user_approval.json"
LEDGER_PATH = PACKAGE_DIR / "queue_ledger.jsonl"
LOCK_PATH = PACKAGE_DIR / "queue.lock"
ATTEMPT_ROOT = PACKAGE_DIR / "attempts"
CANONICAL_ROOT = ROOT / "artifacts/experiments/EXP-037/20260922_conditional_v2"
PREDECESSOR_PROPOSAL_PATH = ROOT / "artifacts/experiments/EXP-037/20260925_conditional_v2_folds3_4_budget_proposal_v1/proposal.json"

MAX_QUEUE_CAP_HOURS = 40.0
MINIMUM_COMPLETE_COMPARISON_HOURS_PER_FOLD = 12.0
RUN_CAP_HOURS = 20.0
WATCHDOG_RESERVE_SECONDS = 180
FOLDS = (3, 4)
ARMS = ("conditional", "state_ablated")
SEEDS = (20260922, 20260923, 20260924)
CONDITION_DIR = ROOT / "data/processed/lincs/GSE92742/continuous_conditions_v2"
INPUT_DIR = ROOT / "data/processed/lincs/GSE92742/continuous_inputs_v2"
CONTRACT_DIR = ROOT / "data/processed/lincs/GSE92742/continuous_contract_v2"
CACHE_PATH = ROOT / "data/processed/lincs/GSE92742/exact978_cache_v1/exact978_cache.npy"
SPLIT_PATH = ROOT / "data/splits/lincs_cell_splits_20260922_v2.json"
REFERENCE_PROVENANCE_PATH = (
    CANONICAL_ROOT / "fold_2_attempt1_provenance.json"
)
REFERENCE_PROVENANCE_SHA256 = "b564944eec3d21540bcea9a87cc9dc25db453caa3ad9e0d1b9a67862d78ef678"
REFERENCE_CODE_HASHES = {
    "scripts/modeling/exp037_conditional_fold.py": "949847b727d8ca23b970cf5c018b0357bcd8e59a34f876ad491a44f4033c4106",
    "src/drug_screen/models/conditional_perturbation.py": "d4863de7b26e1dbbc603932c889b8241e7ecbf61ae4bce87804afa4044cd8511",
    "scripts/evaluation/verify_exp037_conditional_fold.py": "03c94f89d536764ce711d08231a603db0e7e8283f5e8b5876c53aa7e491794da",
    "src/drug_screen/evaluation/perturbation.py": "617a11c7419a51e2eb9bf914e2f2e63991eeed7c6005194beb8b47045d8b0376",
}
REFERENCE_INPUT_HASHES = {
    "data/splits/lincs_cell_splits_20260922_v2.json": "e474cede3d7474ad87c5685850726e9a8db813eabbd775e552fd09899df753d7",
    "data/processed/lincs/GSE92742/continuous_conditions_v2/manifest.json": "0947fdf55d4f016928258d9a372c699efe12bdbdf6cf404d46253bcdd6f3e68e",
    "data/processed/lincs/GSE92742/continuous_inputs_v2/manifest.json": "da8d66da9e087f2ee57c3a8eca3d1d973fdc195f005a220a8e3eb404df7b3337",
    "data/processed/lincs/GSE92742/continuous_contract_v2/verification.json": "a8c73f648c9596e4a55f848c7d237c93fbdc33c3c55f56acd3e9a622f95062c3",
    "data/processed/lincs/GSE92742/exact978_cache_v1/exact978_cache.npy": "04b8bb746a61ba4992e49566315327023783ec1c0448da2a9e263e0881281733",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def child_pids(supervisor_pid: int) -> list[int]:
    children_file = Path(f"/proc/{supervisor_pid}/task/{supervisor_pid}/children")
    if not children_file.is_file():
        return []
    try:
        return [int(value) for value in children_file.read_text(encoding="utf-8").split()]
    except (OSError, ValueError):
        return []


def append_event(event_type: str, **fields: object) -> None:
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    record = {"queue_id": QUEUE_ID, "event": event_type, "timestamp_utc": utc_now(), **fields}
    with LEDGER_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_events() -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    events = []
    for line_number, line in enumerate(LEDGER_PATH.read_text(encoding="utf-8").splitlines(), 1):
        record = json.loads(line)
        if record.get("queue_id") != QUEUE_ID:
            raise RuntimeError(f"ledger queue id mismatch at line {line_number}")
        events.append(record)
    return events


def verified_approved_tranche() -> tuple[dict, dict, float]:
    proposal = json.loads(PROPOSAL_PATH.read_text(encoding="utf-8"))
    if not APPROVAL_PATH.is_file():
        raise RuntimeError(f"no GPU start: user approval record is missing: {APPROVAL_PATH}")
    approval = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
    scope = proposal.get("scientific_scope", {})
    resources = proposal.get("resources", {})
    accounting = proposal.get("accounting", {})
    frozen_reference = proposal.get("frozen_reference_hashes", {})
    predecessor = proposal.get("predecessor_proposal", {})
    execution_runner = proposal.get("execution_runner", {})
    if (scope.get("missing_outer_folds") != list(FOLDS)
            or scope.get("reuse_verified_outer_folds") != [0, 1, 2]
            or scope.get("arms_per_missing_fold") != list(ARMS)
            or scope.get("final_seeds_per_arm") != list(SEEDS)
            or scope.get("training_command_settings_reused_from_verified_fold_2")
            != ["--batch-size 8192", "--load-to-memory"]
            or scope.get("protocol_reference") != str(REFERENCE_PROVENANCE_PATH.relative_to(ROOT))
            or scope.get("protocol_reference_sha256") != REFERENCE_PROVENANCE_SHA256):
        raise RuntimeError("proposal scientific scope differs from the frozen folds/arms/seeds/command")
    if (resources.get("single_run_cap_gpu_hours") != RUN_CAP_HOURS
            or resources.get("approved_incremental_tranche_cap_gpu_hours") != MAX_QUEUE_CAP_HOURS
            or resources.get("historical_exp037_usage") != "UNKNOWN_UNRECOVERED"
            or resources.get("historical_total_claimed") is not False
            or resources.get("budget_reset") is not False
            or resources.get("new_experiment_package_created") is not False
            or resources.get("split_package") is not False):
        raise RuntimeError("proposal does not match the explicit 20-hour/run, 40-hour incremental authorization")
    if accounting.get("minimum_remaining_gpu_hours_per_missing_fold") != MINIMUM_COMPLETE_COMPARISON_HOURS_PER_FOLD:
        raise RuntimeError("proposal resume headroom differs from the conservative remaining-fold gate")
    expected_predecessor = str(PREDECESSOR_PROPOSAL_PATH.relative_to(ROOT))
    if (predecessor.get("path") != expected_predecessor
            or not PREDECESSOR_PROPOSAL_PATH.is_file()
            or predecessor.get("sha256") != sha256(PREDECESSOR_PROPOSAL_PATH)):
        raise RuntimeError("the prior in-cap proposal snapshot is missing or changed")
    runner_relative = str(Path(__file__).resolve().relative_to(ROOT).as_posix())
    if (execution_runner.get("path") != runner_relative
            or execution_runner.get("sha256") != sha256(Path(__file__).resolve())):
        raise RuntimeError("continuation runner differs from the code version bound in the approved proposal")
    if (frozen_reference.get("code") != REFERENCE_CODE_HASHES
            or frozen_reference.get("inputs") != REFERENCE_INPUT_HASHES):
        raise RuntimeError("proposal source/input hashes do not match the verified fold-2 reference")
    if (proposal.get("proposal_id") != QUEUE_ID
            or proposal.get("status") != "USER_APPROVED_INCREMENTAL_TRANCHE_READY_TO_RUN"):
        raise RuntimeError("proposal identity or status is invalid")
    approved_max = float(approval.get("approved_incremental_tranche_gpu_hours", 0))
    approved_at = approval.get("approval_recorded_at_utc", "")
    try:
        approved_at_parsed = datetime.fromisoformat(str(approved_at).replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError("approval record must include a UTC recording timestamp") from exc
    if approved_at_parsed.tzinfo is None or approved_at_parsed.utcoffset().total_seconds() != 0:
        raise RuntimeError("approval recording timestamp must be UTC")
    if (approval.get("status") != "USER_APPROVED_INCREMENTAL_CONTINUATION_TRANCHE"
            or approval.get("proposal_id") != QUEUE_ID
            or approval.get("proposal_sha256") != sha256(PROPOSAL_PATH)
            or approval.get("approved_folds") != list(FOLDS)
            or approval.get("approved_arms") != list(ARMS)
            or approval.get("approved_seeds_per_arm") != list(SEEDS)
            or approval.get("single_run_cap_gpu_hours") != RUN_CAP_HOURS
            or approval.get("historical_exp037_usage") != "UNKNOWN_UNRECOVERED"
            or approval.get("historical_total_claimed") is not False
            or approval.get("budget_reset_or_split") is not False
            or approval.get("approval_text") != "批准新增40 GPU小时，启动续跑"
            or not math.isfinite(approved_max)
            or abs(approved_max - MAX_QUEUE_CAP_HOURS) > 1e-9):
        raise RuntimeError("user approval does not bind this exact 40-GPU-hour amendment and frozen scope")
    return proposal, approval, approved_max


def assert_no_duplicate_exp037_process() -> None:
    ancestors = {os.getpid()}
    current = os.getpid()
    while current > 1:
        stat_path = Path(f"/proc/{current}/stat")
        try:
            fields = stat_path.read_text(encoding="utf-8").split()
            current = int(fields[3])
        except (OSError, ValueError, IndexError):
            break
        ancestors.add(current)
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in ancestors:
            continue
        try:
            comm = (entry / "comm").read_text(encoding="utf-8").strip()
            argv = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if comm.startswith("python") and (
                "scripts/modeling/run_exp037_conditional_continuation_queue.py" in argv
                or "scripts/modeling/exp037_conditional_fold.py" in argv):
            matches.append({"pid": int(entry.name), "command": argv.strip()})
    if matches:
        raise RuntimeError(f"another EXP-037 queue/training process is active; refusing duplicate: {matches}")


def assert_runtime(promoted_folds: set[int]) -> None:
    if sys.platform != "linux" or not os.environ.get("WSL_DISTRO_NAME"):
        raise RuntimeError("formal training is permitted only from WSL2")
    if os.environ.get("CONDA_DEFAULT_ENV") != "drugscreening-gpu":
        raise RuntimeError("active Conda environment must be drugscreening-gpu")
    if not sys.executable.endswith("/envs/drugscreening-gpu/bin/python"):
        raise RuntimeError(f"unexpected Python executable: {sys.executable}")
    if shutil.which("timeout") is None or shutil.which("nvidia-smi") is None:
        raise RuntimeError("GNU timeout and nvidia-smi are required for bounded GPU execution")
    required = (CONDITION_DIR / "manifest.json", INPUT_DIR / "manifest.json",
                CONTRACT_DIR / "verification.json", SPLIT_PATH, CACHE_PATH,
                ROOT / "scripts/modeling/exp037_conditional_fold.py",
                ROOT / "src/drug_screen/models/conditional_perturbation.py",
                ROOT / "src/drug_screen/evaluation/perturbation.py",
                ROOT / "scripts/evaluation/verify_exp037_conditional_fold.py",
                REFERENCE_PROVENANCE_PATH)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("missing frozen input/code files: " + ", ".join(missing))
    for fold in FOLDS:
        canonical = CANONICAL_ROOT / f"fold_{fold}"
        if canonical.exists() and fold not in promoted_folds:
            raise RuntimeError(f"canonical fold_{fold} exists without a ledger promotion; refusing overwrite")
        if fold in promoted_folds and not canonical.is_dir():
            raise RuntimeError(f"ledger records fold_{fold} promotion but its canonical directory is missing")
    disk_free = shutil.disk_usage(PACKAGE_DIR.parent).free
    if disk_free < 20 * 1024**3:
        raise RuntimeError(f"less than 20 GiB free on output volume: {disk_free} bytes")


def select_idle_a5000() -> dict:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    rows = list(csv.reader(query.stdout.splitlines(), skipinitialspace=True))
    devices = []
    for row in rows:
        if len(row) != 5:
            continue
        uuid, name, memory_total, memory_used, utilization = [cell.strip() for cell in row]
        if "A5000" in name and int(memory_total) >= 22000:
            devices.append({"uuid": uuid, "name": name,
                            "memory_total_mib": int(memory_total),
                            "memory_used_mib": int(memory_used),
                            "utilization_percent": int(utilization)})
    if not devices:
        raise RuntimeError("no visible NVIDIA RTX A5000 24 GB device")
    apps_result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
         "--format=csv,noheader"], check=True, capture_output=True, text=True)
    busy_uuids = set()
    for row in csv.reader(apps_result.stdout.splitlines(), skipinitialspace=True):
        if row:
            busy_uuids.add(row[0].strip())
    idle = [device for device in devices if device["uuid"] not in busy_uuids]
    if not idle:
        raise RuntimeError("all visible RTX A5000 devices already have compute processes")
    return min(idle, key=lambda device: (device["utilization_percent"], device["memory_used_mib"], device["uuid"]))


def verify_cuda(uuid: str) -> None:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid)
    probe = "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==1; n=torch.cuda.get_device_name(0); assert 'A5000' in n, n; print(n)"
    result = subprocess.run([sys.executable, "-c", probe], env=env,
                            check=True, capture_output=True, text=True)
    if "A5000" not in result.stdout:
        raise RuntimeError(f"CUDA probe did not identify the selected A5000: {result.stdout.strip()}")


def frozen_hashes() -> dict[str, str]:
    if sha256(REFERENCE_PROVENANCE_PATH) != REFERENCE_PROVENANCE_SHA256:
        raise RuntimeError("fold-2 source provenance hash changed; refusing a non-comparable continuation")
    files = {rel: ROOT / rel for rel in (*REFERENCE_CODE_HASHES, *REFERENCE_INPUT_HASHES)}
    actual = {rel: sha256(path) for rel, path in files.items()}
    expected = {**REFERENCE_CODE_HASHES, **REFERENCE_INPUT_HASHES}
    mismatches = {rel: {"expected": expected[rel], "actual": actual[rel]}
                  for rel in expected if actual[rel] != expected[rel]}
    if mismatches:
        raise RuntimeError(f"fold-2 frozen source/input hashes changed: {mismatches}")
    return {"fold2_reference_provenance": REFERENCE_PROVENANCE_SHA256,
            "scripts/modeling/run_exp037_conditional_continuation_queue.py": sha256(Path(__file__).resolve()),
            **actual}


def output_tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()
                       and p.name != "verification_recovery.json"):
        relative = path.relative_to(directory).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def attempt_state(events: list[dict]) -> tuple[float, set[int], int]:
    starts = {str(event["attempt_id"]): event for event in events if event["event"] == "attempt_started"}
    finishes = {str(event["attempt_id"]): event for event in events if event["event"] == "attempt_finished"}
    unresolved = sorted(set(starts) - set(finishes))
    if unresolved:
        raise RuntimeError("unmatched attempt starts require manual host reconciliation: " + ", ".join(unresolved))
    spent = sum(float(event["gpu_hours_charged"]) for event in finishes.values())
    promoted = {int(event["fold"]) for event in events if event["event"] == "fold_promoted"}
    return spent, promoted, len(starts)


def recover_promotions(events: list[dict]) -> set[int]:
    promoted = {int(event["fold"]) for event in events if event["event"] == "fold_promoted"}
    for fold in FOLDS:
        canonical = CANONICAL_ROOT / f"fold_{fold}"
        if fold in promoted:
            continue
        intents = [event for event in events if event["event"] == "fold_promotion_intent"
                   and int(event["fold"]) == fold]
        if not intents:
            if canonical.exists():
                raise RuntimeError(f"canonical fold_{fold} exists without a durable promotion intent")
            continue
        if len(intents) != 1:
            raise RuntimeError(f"fold_{fold} has multiple promotion intents; manual reconciliation required")
        intent = intents[0]
        attempt_id = str(intent["attempt_id"])
        source = Path(str(intent["source_dir"]))
        expected_digest = str(intent["output_tree_sha256"])
        if Path(str(intent["canonical_dir"])) != canonical:
            raise RuntimeError(f"fold_{fold} promotion intent points to another canonical directory")
        finished = [event for event in events if event["event"] == "attempt_finished"
                    and str(event["attempt_id"]) == attempt_id
                    and event.get("outcome") == "EXIT_0" and int(event["fold"]) == fold]
        if len(finished) != 1:
            raise RuntimeError(f"fold_{fold} promotion intent lacks one successful matching attempt")
        result_dir = canonical if canonical.exists() else source
        if not result_dir.is_dir() or output_tree_sha256(result_dir) != expected_digest:
            raise RuntimeError(f"fold_{fold} result tree does not match its durable promotion intent")
        summary = json.loads((result_dir / "fold_summary.json").read_text(encoding="utf-8"))
        if (int(summary.get("fold", -1)) != fold
                or summary.get("status") != "FOLD_COMPLETE_PENDING_CROSS_FOLD_REVIEW"):
            raise RuntimeError(f"orphan fold_{fold} has an invalid summary")
        recovery_receipt = result_dir / "verification_recovery.json"
        verify_env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        verify_cmd = [sys.executable, "scripts/evaluation/verify_exp037_conditional_fold.py",
                      "--condition-dir", str(CONDITION_DIR), "--split", str(SPLIT_PATH),
                      "--fold-dir", str(result_dir), "--output", str(recovery_receipt)]
        subprocess.run(verify_cmd, cwd=ROOT, env=verify_env, check=True,
                       capture_output=True, text=True)
        verification = json.loads(recovery_receipt.read_text(encoding="utf-8"))
        if verification.get("status") != "PASS" or int(verification.get("fold", -1)) != fold:
            raise RuntimeError(f"orphan canonical fold_{fold} failed recovery verification")
        if verification.get("status") != "PASS" or int(verification.get("fold", -1)) != fold:
            raise RuntimeError(f"orphan fold_{fold} failed recovery verification")
        if not canonical.exists():
            canonical.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, canonical)
        append_event("fold_promoted", fold=fold, attempt_id=attempt_id,
                     canonical_dir=str(canonical), verification=verification,
                     reconciled_after_atomic_move=True,
                     output_tree_sha256=expected_digest,
                     recovery_receipt=str(recovery_receipt))
        promoted.add(fold)
        events = read_events()
    return promoted


def run_attempt(fold: int, attempt_number: int, seconds_cap: float,
                gpu: dict, input_hashes: dict[str, str]) -> bool:
    attempt_id = f"fold{fold}_attempt{attempt_number:03d}"
    attempt_dir = ATTEMPT_ROOT / attempt_id
    if attempt_dir.exists():
        raise RuntimeError(f"attempt directory already exists: {attempt_dir}")
    attempt_dir.mkdir(parents=True)
    log_path = attempt_dir / "training.log"
    command = [
        sys.executable, "scripts/modeling/exp037_conditional_fold.py",
        "--condition-dir", str(CONDITION_DIR), "--input-dir", str(INPUT_DIR),
        "--contract-dir", str(CONTRACT_DIR), "--cache", str(CACHE_PATH),
        "--split", str(SPLIT_PATH), "--fold", str(fold),
        "--output-dir", str(attempt_dir / "result"), "--device", "cuda:0",
        "--batch-size", "8192", "--load-to-memory",
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"])
    start_utc = utc_now()
    start_mono = time.monotonic()
    prior_spend, _, _ = attempt_state(read_events())
    hard_timeout_seconds = max(1, int(seconds_cap) - WATCHDOG_RESERVE_SECONDS)
    watchdog_command = ["timeout", "--signal=TERM", "--kill-after=5s",
                        f"{hard_timeout_seconds}s", *command]
    append_event("attempt_started", attempt_id=attempt_id, fold=fold,
                 attempt_number=attempt_number, pid_pending=True,
                 gpu_uuid=gpu["uuid"], gpu_name=gpu["name"], gpu_count=1,
                 start_utc=start_utc, seconds_cap=seconds_cap,
                 watchdog_deadline_seconds=hard_timeout_seconds,
                 historical_exp037_usage="UNKNOWN_UNRECOVERED",
                 command=watchdog_command, input_hashes=input_hashes,
                 attempt_dir=str(attempt_dir), log_path=str(log_path))
    timed_out = False
    last_heartbeat = start_mono
    process = None
    try:
        with log_path.open("wb") as log_stream:
            process = subprocess.Popen(watchdog_command, cwd=ROOT, env=env, stdout=log_stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            training_pids = child_pids(process.pid)
            append_event("process_spawned", attempt_id=attempt_id,
                         watchdog_pid=process.pid, training_pids=training_pids,
                         child_process_start_utc=utc_now(),
                         child_process_start_monotonic_seconds=time.monotonic())
            while process.poll() is None:
                now = time.monotonic()
                elapsed = now - start_mono
                if elapsed >= hard_timeout_seconds + 10.0:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired as exc:
                        append_event("attempt_unreconciled", attempt_id=attempt_id,
                                     fold=fold, pid=process.pid,
                                     elapsed_seconds=time.monotonic() - start_mono,
                                     reason="external watchdog did not exit after bounded kill")
                        raise RuntimeError("training watchdog remains active; reconcile before any restart") from exc
                    break
                if now - last_heartbeat >= 300:
                    append_event("progress", attempt_id=attempt_id, pid=process.pid,
                                 training_pids=child_pids(process.pid),
                                 elapsed_seconds=elapsed,
                                 gpu_hours_so_far=elapsed / 3600.0)
                    last_heartbeat = now
                time.sleep(min(10.0, max(0.2, seconds_cap - elapsed)))
            exit_code = process.returncode
    except BaseException:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                append_event("attempt_unreconciled", attempt_id=attempt_id,
                             fold=fold, pid=process.pid,
                             elapsed_seconds=time.monotonic() - start_mono,
                             reason="bounded termination could not confirm process exit")
                raise
        elapsed = 0.0 if process is None else time.monotonic() - start_mono
        append_event("attempt_finished", attempt_id=attempt_id, fold=fold,
                     outcome="INTERRUPTED_EXCEPTION", exit_code=None,
                     ended_utc=utc_now(), elapsed_seconds=elapsed,
                     gpu_hours_charged=elapsed / 3600.0,
                     incremental_tranche_cumulative_gpu_hours=prior_spend + elapsed / 3600.0)
        raise
    elapsed = time.monotonic() - start_mono
    timed_out = timed_out or exit_code == 124
    success = exit_code == 0 and not timed_out
    append_event("attempt_finished", attempt_id=attempt_id, fold=fold,
                 outcome="TIME_LIMIT" if timed_out else ("EXIT_0" if success else "NONZERO_EXIT"),
                 exit_code=exit_code, ended_utc=utc_now(), elapsed_seconds=elapsed,
                 gpu_hours_charged=elapsed / 3600.0,
                 incremental_tranche_cumulative_gpu_hours=prior_spend + elapsed / 3600.0)
    if not success:
        append_event("queue_paused_after_failure", attempt_id=attempt_id, fold=fold,
                     remaining_attempts_require_explicit_invocation=True)
        return False

    result_dir = attempt_dir / "result"
    verification_path = result_dir / "verification_continuation.json"
    verify_env = dict(env, CUDA_VISIBLE_DEVICES="")
    verify_cmd = [sys.executable, "scripts/evaluation/verify_exp037_conditional_fold.py",
                  "--condition-dir", str(CONDITION_DIR), "--split", str(SPLIT_PATH),
                  "--fold-dir", str(result_dir), "--output", str(verification_path)]
    try:
        verify_result = subprocess.run(verify_cmd, cwd=ROOT, env=verify_env,
                                       check=True, capture_output=True, text=True)
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        if verification.get("status") != "PASS" or int(verification.get("fold", -1)) != fold:
            raise RuntimeError(f"fold {fold} independent verifier did not PASS")
    except Exception as exc:
        append_event("fold_verification_failed", fold=fold, attempt_id=attempt_id,
                     error=repr(exc), result_dir=str(result_dir))
        raise
    canonical = CANONICAL_ROOT / f"fold_{fold}"
    if canonical.exists():
        raise RuntimeError(f"canonical fold_{fold} appeared during run; refusing overwrite")
    result_tree_hash = output_tree_sha256(result_dir)
    append_event("fold_promotion_intent", fold=fold, attempt_id=attempt_id,
                 source_dir=str(result_dir), canonical_dir=str(canonical),
                 output_tree_sha256=result_tree_hash,
                 verification=verification)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    os.replace(result_dir, canonical)
    append_event("fold_promoted", fold=fold, attempt_id=attempt_id,
                 canonical_dir=str(canonical), verification=verification,
                 output_tree_sha256=result_tree_hash,
                 verifier_stdout=verify_result.stdout[-2000:])
    return True


def main() -> int:
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another EXP-037 add-on queue runner holds the lock") from exc
        proposal, approval, cap_hours = verified_approved_tranche()
        assert_no_duplicate_exp037_process()
        events = read_events()
        if not events:
            append_event("queue_opened", historical_exp037_actual_gpu_hours="UNKNOWN_UNRECOVERED",
                         historical_exp037_total_claimed=False,
                         authorized_incremental_tranche_cap_gpu_hours=cap_hours,
                         incremental_budget_origin="first attempt's child-process Popen; no preflight time charged",
                         predecessor_proposal_path=str(PREDECESSOR_PROPOSAL_PATH.relative_to(ROOT)),
                         predecessor_proposal_sha256=sha256(PREDECESSOR_PROPOSAL_PATH),
                         user_approval_sha256=sha256(APPROVAL_PATH),
                         user_approved_tranche_gpu_hours=approval["approved_incremental_tranche_gpu_hours"],
                         proposal_sha256=sha256(PROPOSAL_PATH),
                         scope={"folds": list(FOLDS), "arms": list(ARMS), "seeds": list(SEEDS)})
            events = read_events()
        spent, promoted, _ = attempt_state(events)
        opened = next(event for event in events if event["event"] == "queue_opened")
        if (opened.get("historical_exp037_actual_gpu_hours") != "UNKNOWN_UNRECOVERED"
                or opened.get("historical_exp037_total_claimed") is not False):
            raise RuntimeError("queue ledger must preserve unknown historical usage without a cumulative-total claim")
        if opened.get("predecessor_proposal_sha256") != sha256(PREDECESSOR_PROPOSAL_PATH):
            raise RuntimeError("predecessor proposal changed after this queue was opened")
        if opened.get("user_approval_sha256") != sha256(APPROVAL_PATH):
            raise RuntimeError("user approval changed after this queue was opened")
        if opened.get("proposal_sha256") != sha256(PROPOSAL_PATH):
            raise RuntimeError("proposal changed after this queue was opened")
        promoted = recover_promotions(events)
        spent, promoted, _ = attempt_state(read_events())
        assert_runtime(promoted)
        if spent > cap_hours + 1e-6:
            raise RuntimeError(f"add-on queue already exceeds its approved cap: {spent:.6f} GPUh")
        frozen = frozen_hashes()
        previous_frozen = [event for event in events if event["event"] == "frozen_inputs"]
        if len(previous_frozen) > 1:
            raise RuntimeError("queue ledger contains duplicate frozen-input records")
        if previous_frozen and previous_frozen[0].get("hashes") != frozen:
            raise RuntimeError("frozen input/code hashes differ from the queue-open record")
        if not previous_frozen:
            append_event("frozen_inputs", hashes=frozen)
            events = read_events()
        if promoted == set(FOLDS):
            return 0
        for fold in FOLDS:
            if fold in promoted:
                continue
            outstanding = sum(1 for pending in FOLDS if pending not in promoted)
            remaining = cap_hours - spent
            minimum_complete_comparison_hours = (
                MINIMUM_COMPLETE_COMPARISON_HOURS_PER_FOLD * outstanding
            )
            if remaining + 1e-6 < minimum_complete_comparison_hours:
                append_event(
                    "queue_paused_insufficient_completion_headroom",
                    remaining_gpu_hours=remaining,
                    minimum_required_gpu_hours=minimum_complete_comparison_hours,
                    missing_folds=[f for f in FOLDS if f not in promoted],
                    minimum_per_fold_gpu_hours=MINIMUM_COMPLETE_COMPARISON_HOURS_PER_FOLD,
                    reason="do not restart a partial comparison with less than the conservative 12-hour-per-fold scenario",
                )
                return 2
            seconds_cap = min(RUN_CAP_HOURS * 3600.0, remaining * 3600.0 / outstanding)
            if seconds_cap < 1800:
                append_event("queue_paused_budget_exhausted", remaining_gpu_hours=remaining,
                             missing_folds=[f for f in FOLDS if f not in promoted])
                return 2
            gpu = select_idle_a5000()
            verify_cuda(gpu["uuid"])
            append_event("gpu_preflight_pass", fold=fold, gpu=gpu,
                         python=sys.executable, conda_env=os.environ.get("CONDA_DEFAULT_ENV"),
                         disk_free_bytes=shutil.disk_usage(PACKAGE_DIR.parent).free)
            prior_starts = [event for event in events
                            if event["event"] == "attempt_started" and int(event["fold"]) == fold]
            attempt_number = len(prior_starts) + 1
            if not run_attempt(fold, attempt_number, seconds_cap, gpu, frozen):
                return 3
            events = read_events()
            spent, promoted, _ = attempt_state(events)
        append_event("queue_completed_folds_3_4", incremental_tranche_gpu_hours_charged=spent,
                     historical_exp037_actual_gpu_hours="UNKNOWN_UNRECOVERED",
                     historical_exp037_total_claimed=False,
                     approved_incremental_tranche_cap_gpu_hours=cap_hours,
                     canonical_folds=[3, 4])
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
