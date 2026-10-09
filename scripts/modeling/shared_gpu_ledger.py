"""Small locked JSONL helpers for the shared 40-GPU-hour experiment tranche."""

from __future__ import annotations

import fcntl
import argparse
import hashlib
import json
import math
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


QUEUE_ID = "EXP037-COND-V2-F34-INCREMENTAL-20260926"
CLOSING_EVENTS = {"attempt_finished", "attempt_reconciled"}
_ACTIVE_TRANSACTIONS: dict[str, dict] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@contextmanager
def ledger_transaction(lock_path: Path) -> Iterator["LedgerTransaction"]:
    """Hold the common lock only for one read/check/write transaction."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    key = str(lock_path.resolve())
    active = _ACTIVE_TRANSACTIONS.get(key)
    if active is not None:
        active["depth"] += 1
        try:
            yield active["transaction"]
        finally:
            active["depth"] -= 1
        return
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        active = {"stream": stream, "transaction": LedgerTransaction(),
                  "depth": 1, "locked": True}
        _ACTIVE_TRANSACTIONS[key] = active
        try:
            yield active["transaction"]
        finally:
            if _ACTIVE_TRANSACTIONS.get(key) is active:
                _ACTIVE_TRANSACTIONS.pop(key, None)
                if active["locked"]:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def release_ledger_transaction(lock_path: Path) -> None:
    """Release the current process's lock after a durable process_spawned event."""
    key = str(lock_path.resolve())
    active = _ACTIVE_TRANSACTIONS.pop(key, None)
    if active is not None and active["locked"]:
        fcntl.flock(active["stream"].fileno(), fcntl.LOCK_UN)
        active["locked"] = False


class LedgerTransaction:
    """Operations that must be called while ``ledger_transaction`` is held."""

    @staticmethod
    def read(ledger_path: Path) -> list[dict]:
        if not ledger_path.exists():
            return []
        rows: list[dict] = []
        for number, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("queue_id") != QUEUE_ID:
                raise RuntimeError(f"shared ledger queue id mismatch at line {number}")
            rows.append(row)
        return rows

    @staticmethod
    def append(ledger_path: Path, event: str, **fields: object) -> dict:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"queue_id": QUEUE_ID, "event": event, "timestamp_utc": utc_now(), **fields}
        encoded = (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        descriptor = os.open(ledger_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
        try:
            written = os.write(descriptor, encoded)
            if written != len(encoded):
                raise OSError(f"short JSONL append: {written}/{len(encoded)} bytes")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return row


def read_events(ledger_path: Path, lock_path: Path) -> list[dict]:
    with ledger_transaction(lock_path) as transaction:
        return transaction.read(ledger_path)


def append_event(ledger_path: Path, lock_path: Path, event: str, **fields: object) -> dict:
    with ledger_transaction(lock_path) as transaction:
        return transaction.append(ledger_path, event, **fields)


def package_for_attempt(start: dict) -> str:
    """Resolve old ledger starts without ``package_id`` without changing history."""
    package = start.get("package_id")
    if package:
        return str(package)
    run_id = str(start.get("run_id", ""))
    if run_id in {"20260926_pharmaformer_transfer_25pct", "20260926_pharmaformer_transfer_100pct"}:
        return "EXP036041_CURVES"
    if "EXP-048" in start.get("experiment_ids", []):
        return "EXP048_FIVE_ARM"
    return "LEGACY_UNASSIGNED"


def budget_snapshot(events: list[dict]) -> dict:
    """Return settled charges and full caps held by attempts without a close event.

    An unmatched start is never treated as zero. Its complete reserved cap remains
    unavailable until an attempt_finished or evidence-backed attempt_reconciled row.
    """
    starts: dict[str, dict] = {}
    closes: dict[str, dict] = {}
    for row in events:
        event = row.get("event")
        if event == "attempt_started":
            attempt_id = str(row.get("attempt_id", ""))
            if not attempt_id or attempt_id in starts:
                raise RuntimeError("missing or duplicate attempt_started ID")
            starts[attempt_id] = row
        elif event in CLOSING_EVENTS:
            attempt_id = str(row.get("attempt_id", ""))
            if not attempt_id or attempt_id not in starts or attempt_id in closes:
                raise RuntimeError("orphan or duplicate attempt close; manual reconciliation required")
            closes[attempt_id] = row

    settled_by_package: dict[str, float] = {}
    reserved_by_package: dict[str, float] = {}
    charged_total = 0.0
    reserved_total = 0.0
    active: list[dict] = []
    for attempt_id, start in starts.items():
        package = package_for_attempt(start)
        if attempt_id in closes:
            charge = float(closes[attempt_id].get("gpu_hours_charged", math.nan))
            if not math.isfinite(charge) or charge < 0:
                raise RuntimeError(f"invalid charge for attempt {attempt_id}")
            charged_total += charge
            settled_by_package[package] = settled_by_package.get(package, 0.0) + charge
        else:
            reserve = float(start.get("reserved_gpu_hours",
                                      start.get("planned_attempt_cap_gpu_hours", math.nan)))
            if not math.isfinite(reserve) or reserve <= 0:
                raise RuntimeError(f"missing/invalid GPU reservation for unresolved {attempt_id}")
            reserved_total += reserve
            reserved_by_package[package] = reserved_by_package.get(package, 0.0) + reserve
            active.append({"attempt_id": attempt_id, "package_id": package,
                           "reserved_gpu_hours": reserve,
                           "pid": start.get("pid")})
    return {
        "settled_gpu_hours": charged_total,
        "reserved_gpu_hours": reserved_total,
        "available_gpu_hours": 40.0 - charged_total - reserved_total,
        "settled_by_package": settled_by_package,
        "reserved_by_package": reserved_by_package,
        "active_attempts": active,
    }


def allocation_from_events(events: list[dict]) -> dict:
    allocations = [row for row in events if row.get("event") == "allocation_set"]
    if len(allocations) != 1:
        raise RuntimeError("exactly one Sol-backed allocation_set is required")
    row = allocations[0]
    if (float(row.get("authorized_total_gpu_hours", 0)) != 40.0
            or row.get("budget_reset") is not False
            or row.get("budget_split") is not False
            or row.get("historical_usage") != "UNKNOWN_UNRECOVERED"):
        raise RuntimeError("shared allocation does not preserve the approved budget boundary")
    mapping = row.get("allocations_gpu_hours")
    required = {"EXP036041_CURVES", "EXP048_FIVE_ARM", "EXP046_PATHWAY",
                "SHARED_EMERGENCY"}
    if not isinstance(mapping, dict) or set(mapping) != required:
        raise RuntimeError("allocation_set package keys do not match the approved packages")
    values = {key: float(value) for key, value in mapping.items()}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise RuntimeError("invalid shared package allocation")
    if abs(sum(values.values()) - (40.0 - float(row["baseline_spent_gpu_hours"])
                                   - float(row.get("baseline_reserved_gpu_hours", 0.0)))) > 1e-8:
        raise RuntimeError("package allocations do not sum to the true remaining tranche")
    return row


def total_budget_snapshot(events: list[dict], total_cap_gpu_hours: float = 40.0) -> dict:
    snapshot = budget_snapshot(events)
    snapshot["available_gpu_hours"] = (float(total_cap_gpu_hours)
                                       - snapshot["settled_gpu_hours"]
                                       - snapshot["reserved_gpu_hours"])
    return snapshot


def package_budget_snapshot(events: list[dict], package_id: str) -> dict:
    allocation = allocation_from_events(events)
    allocation_position = events.index(allocation)
    started_after: set[str] = set()
    for row in events[allocation_position + 1:]:
        if row.get("event") == "attempt_started" and package_for_attempt(row) == package_id:
            started_after.add(str(row["attempt_id"]))
    starts = {str(row["attempt_id"]): row for row in events if row.get("event") == "attempt_started"}
    closes = {str(row["attempt_id"]): row for row in events if row.get("event") in CLOSING_EVENTS}
    spent = sum(float(closes[item]["gpu_hours_charged"]) for item in started_after if item in closes)
    reserved = sum(float(starts[item].get("reserved_gpu_hours",
                                        starts[item].get("planned_attempt_cap_gpu_hours")))
                   for item in started_after if item not in closes)
    allocation_value = float(allocation["allocations_gpu_hours"][package_id])
    return {"package_id": package_id, "allocation_gpu_hours": allocation_value,
            "settled_after_allocation_gpu_hours": spent,
            "reserved_gpu_hours": reserved,
            "remaining_gpu_hours": allocation_value - spent - reserved,
            "allocation_sha256": allocation.get("sol_decision_sha256")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_sol_allocation(root: Path, decision_path: Path) -> dict:
    ledger_path = root / "artifacts/experiments/EXP-037/20260926_conditional_v2_folds3_4_budget_amendment_v2/queue_ledger.jsonl"
    lock_path = ledger_path.with_name("queue.lock")
    relative_decision = decision_path.relative_to(root).as_posix()
    decision_hash = sha256(decision_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (decision.get("actual_model") != "gpt-6-sol"
            or decision.get("reasoning_effort") != "medium"
            or decision.get("decision") != "AUTHORIZE_BOUNDED_SHARED_TRANCHE_REALLOCATION_WITH_COST_GATES"):
        raise RuntimeError("the Sol/medium shared budget decision is not the expected ruling")
    shared = decision["shared_ledger"]
    planned = decision["allocations_from_current_available_gpu_hours"]
    allocations = {
        "EXP036041_CURVES": float(planned["exp036_041_remaining_25pct_100pct_first_pass_max_gpu_hours"]),
        "EXP048_FIVE_ARM": float(planned["exp048_profile_and_fivefold_combined_max_gpu_hours"]),
        "EXP046_PATHWAY": 0.0,
        "SHARED_EMERGENCY": float(planned["unallocated_same_pool_contingency_gpu_hours"]),
    }
    with ledger_transaction(lock_path) as transaction:
        events = transaction.read(ledger_path)
        if any(row.get("event") == "allocation_set" for row in events):
            raise RuntimeError("shared allocation_set already exists; do not overwrite or reset it")
        snapshot = budget_snapshot(events)
        if (abs(snapshot["settled_gpu_hours"] - float(shared["settled_cumulative_gpu_hours"])) > 1e-8
                or abs(snapshot["reserved_gpu_hours"]) > 1e-8
                or abs(snapshot["available_gpu_hours"] - float(shared["available_gpu_hours"])) > 1e-8):
            raise RuntimeError("live ledger does not match the Sol ruling baseline")
        row = transaction.append(
            ledger_path,
            "allocation_set",
            package_id="SHARED_40H_TRANCHE",
            authorized_total_gpu_hours=40.0,
            baseline_spent_gpu_hours=snapshot["settled_gpu_hours"],
            baseline_reserved_gpu_hours=snapshot["reserved_gpu_hours"],
            available_at_allocation_gpu_hours=snapshot["available_gpu_hours"],
            allocations_gpu_hours=allocations,
            sol_decision_path=relative_decision,
            sol_decision_sha256=decision_hash,
            actual_model=decision["actual_model"],
            reasoning_effort=decision["reasoning_effort"],
            historical_usage="UNKNOWN_UNRECOVERED",
            budget_reset=False,
            budget_split=False,
        )
    return {"status": "ALLOCATION_SET", "event": row,
            "ledger_balance": budget_snapshot(read_events(ledger_path, lock_path))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--install-sol-allocation", type=Path)
    args = parser.parse_args()
    if args.install_sol_allocation is None:
        parser.error("--install-sol-allocation is required")
    print(json.dumps(install_sol_allocation(args.root.resolve(),
                                            args.install_sol_allocation.resolve()),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
