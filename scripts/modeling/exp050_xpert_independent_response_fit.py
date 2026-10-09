"""One fresh XPert fit for EXP-050 independent-object transcript-response transfer.

The default mode is a response-blind identity/key preflight.  Training and
response scoring require a separately reviewed protocol digest and the
manager's explicit final GO flag.  No legacy split JSON or checkpoint is used.
"""
from __future__ import annotations

import argparse
import fcntl
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch

import exp037_xpert_fold as adapter


RUN_ID = "20261001_xpert_independent_object_fit_v1"
GATE_ID = "20261001_xpert_independent_object_gate_v1"
MASK_DIR = Path("artifacts/experiments/EXP-037/20260928_lincs_scale_16_121/prepared")
RUN_REL = Path("artifacts/experiments/EXP-050") / RUN_ID
GATE_REL = Path("artifacts/experiments/EXP-050") / GATE_ID
MODEL_CSV_REL = Path("data/raw/depmap/24q2/Model.csv")
PERT_INFO_REL = Path("data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_pert_info.txt.gz")
FULL_KEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
COND_COLUMNS = ["condition_id", "source_row", "cell_id", "pert_id", "pert_dose", "pert_time"]
MODEL_COLUMNS = ["ModelID", "PatientID", "CellLineName", "StrippedCellLineName"]
PARAM_SOURCE = Path("artifacts/experiments/EXP-037/20260928_lincs_scale_16_121/xpert_121/protocol.json")
PARAM_SOURCE_SHA256 = "45b4f773636524f980854e598a988d539d92c159026fb4d06bcb9300aa4fdf72"
FROZEN_PARAMS = {
    "seed": 20260922,
    "batch_size": 16,
    "max_epochs": 100,
    "patience": 10,
    "learning_rate": 0.004,
    "weight_decay": 0.00001,
    "loss_weights": [0.2, 0.003, 0.2, 1.0],
}
ATTEMPT_CAP_GPU_HOURS = 10.0
EXPECTED_FIT_GPU_HOURS = 8.152780639016001
GPU_LEDGER_REL = Path("artifacts/experiments/EXP-050/gpu_ledger.jsonl")
GPU_LEDGER_LOCK_REL = Path("artifacts/experiments/EXP-050/gpu_ledger.lock")


class FitContractError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


def write_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    frame.to_csv(temp, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL,
                 lineterminator="\n")
    os.replace(temp, path)


def parse_ids(raw: Any, *, label: str) -> list[str]:
    if raw is None or pd.isna(raw):
        return []
    values = [str(part).strip() for part in str(raw).split(";") if str(part).strip()]
    if len(values) != len(set(values)):
        raise FitContractError(f"duplicate condition ID within {label}")
    return values


def hash_ids(values: set[str] | list[str]) -> str:
    return sha256_text("\n".join(sorted(map(str, values))) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FitContractError(f"expected JSON object: {path}")
    return value


def _read_attempt_receipts(run_dir: Path) -> list[dict[str, Any]]:
    attempts_dir = run_dir / "attempts"
    if not attempts_dir.exists():
        return []
    receipts: list[dict[str, Any]] = []
    for attempt_dir in sorted(attempts_dir.glob("attempt_[0-9][0-9][0-9]")):
        receipt_path = attempt_dir / "attempt_receipt.json"
        if not receipt_path.is_file():
            raise FitContractError(f"attempt directory lacks a receipt; reconcile before continuing: {attempt_dir}")
        receipt = read_json(receipt_path)
        if receipt.get("status") == "RUNNING":
            raise FitContractError(f"prior XPert attempt is still open; inspect its process and receipt: {attempt_dir}")
        if receipt.get("status") not in {"FAILED", "COMPLETED_ONE_FRESH_XPERT_FIT"}:
            raise FitContractError(f"unknown prior attempt status; reconcile before continuing: {attempt_dir}")
        charge = receipt.get("charged_gpu_hours")
        if not isinstance(charge, (int, float)) or not math.isfinite(float(charge)) or float(charge) < 0:
            raise FitContractError(f"prior XPert attempt has no valid settled GPU charge: {attempt_dir}")
        receipts.append(receipt)
    expected_ids = [f"attempt_{index:03d}" for index in range(1, len(receipts) + 1)]
    if [str(value.get("attempt_id")) for value in receipts] != expected_ids:
        raise FitContractError("XPert attempt sequence is incomplete or out of order")
    if sum(value.get("status") == "COMPLETED_ONE_FRESH_XPERT_FIT" for value in receipts) > 0:
        raise FitContractError("a completed XPert fit already exists; do not start another")
    if sum(float(value["charged_gpu_hours"]) for value in receipts) > ATTEMPT_CAP_GPU_HOURS:
        raise FitContractError("prior XPert engineering attempts exhausted the cumulative 10-GPU-hour subcap")
    if any(not value.get("error") and value.get("status") == "FAILED" for value in receipts):
        raise FitContractError("a failed XPert attempt lacks a recorded cause; reconcile before any retry")
    return receipts


def _read_gpu_ledger(root: Path) -> list[dict[str, Any]]:
    path = root / GPU_LEDGER_REL
    if not path.is_file():
        raise FitContractError(f"EXP-050 GPU ledger is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FitContractError(f"EXP-050 GPU ledger is invalid at line {line_no}") from exc
        if row.get("experiment_id") != "EXP-050":
            raise FitContractError(f"unexpected experiment in EXP-050 GPU ledger at line {line_no}")
        rows.append(row)
    return rows


def _ledger_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    starts: dict[str, dict[str, Any]] = {}
    closes: dict[str, dict[str, Any]] = {}
    for row in rows:
        event = row.get("event")
        attempt_id = str(row.get("attempt_id", ""))
        if event == "attempt_started":
            if not attempt_id or attempt_id in starts:
                raise FitContractError("EXP-050 GPU ledger has a missing or duplicate attempt start")
            starts[attempt_id] = row
        elif event in {"attempt_finished", "attempt_reconciled"}:
            if not attempt_id or attempt_id not in starts or attempt_id in closes:
                raise FitContractError("EXP-050 GPU ledger has an orphan or duplicate attempt close")
            charge = float(row.get("gpu_hours_charged", math.nan))
            if not math.isfinite(charge) or charge < 0:
                raise FitContractError(f"EXP-050 GPU ledger has an invalid charge: {attempt_id}")
            closes[attempt_id] = row
    settled = sum(float(row["gpu_hours_charged"]) for row in closes.values())
    reserved = sum(float(starts[key].get("reserved_gpu_hours", math.nan))
                   for key in starts.keys() - closes.keys())
    fit_starts = {key: row for key, row in starts.items() if row.get("run_id") == RUN_ID}
    fit_settled = sum(float(closes[key]["gpu_hours_charged"])
                      for key in fit_starts.keys() & closes.keys())
    fit_reserved = sum(float(fit_starts[key].get("reserved_gpu_hours", math.nan))
                       for key in fit_starts.keys() - closes.keys())
    if not all(math.isfinite(value) and value >= 0 for value in (settled, reserved, fit_settled, fit_reserved)):
        raise FitContractError("EXP-050 GPU ledger has a missing or invalid reservation")
    return {"settled_gpu_hours": settled, "reserved_gpu_hours": reserved,
            "fit_settled_gpu_hours": fit_settled, "fit_reserved_gpu_hours": fit_reserved,
            "open_attempt_ids": sorted(starts.keys() - closes.keys()),
            "fit_attempt_count": len(fit_starts)}


def _reserve_gpu_attempt(root: Path, attempt_id: str, device_name: str,
                         pid: int) -> tuple[float, dict[str, Any]]:
    ledger_path = root / GPU_LEDGER_REL
    lock_path = root / GPU_LEDGER_LOCK_REL
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows = _read_gpu_ledger(root)
        usage = _ledger_usage(rows)
        if usage["open_attempt_ids"]:
            raise FitContractError(f"unclosed EXP-050 GPU attempt reservation: {usage['open_attempt_ids']}")
        if usage["settled_gpu_hours"] + ATTEMPT_CAP_GPU_HOURS > 48.0:
            raise FitContractError("reserving this XPert attempt would exceed the 48-GPU-hour EXP-050 cap")
        remaining_fit = ATTEMPT_CAP_GPU_HOURS - usage["fit_settled_gpu_hours"]
        if remaining_fit + 1e-12 < EXPECTED_FIT_GPU_HOURS:
            raise FitContractError(
                f"remaining XPert subcap {remaining_fit:.6f} GPUh is below the frozen full-fit estimate "
                f"{EXPECTED_FIT_GPU_HOURS:.6f} GPUh; preserve the remaining balance")
        if attempt_id in {str(row.get("attempt_id")) for row in rows}:
            raise FitContractError(f"GPU ledger already contains attempt ID {attempt_id}")
        reserved = min(ATTEMPT_CAP_GPU_HOURS, remaining_fit, 48.0 - usage["settled_gpu_hours"])
        if reserved + 1e-12 < EXPECTED_FIT_GPU_HOURS:
            raise FitContractError("available EXP-050 balance is below the frozen XPert fit estimate")
        start = {"experiment_id": "EXP-050", "run_id": RUN_ID, "attempt_id": attempt_id,
                 "event": "attempt_started", "operation": "fixed XPert independent-object response fit",
                 "device": "cuda", "device_name": device_name, "pid": pid,
                 "reserved_gpu_hours": reserved,
                 "estimated_gpu_hours": EXPECTED_FIT_GPU_HOURS,
                 "cumulative_subtask_cap_gpu_hours": ATTEMPT_CAP_GPU_HOURS,
                 "cumulative_package_cap_gpu_hours": 48.0,
                 "timestamp_utc": utc_now()}
        encoded = (json.dumps(start, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        descriptor = os.open(ledger_path, os.O_WRONLY | os.O_APPEND)
        try:
            if os.write(descriptor, encoded) != len(encoded):
                raise OSError("short EXP-050 GPU ledger append")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return reserved, usage


def _finish_gpu_attempt(root: Path, attempt_id: str, charged: float,
                        status: str, error: str | None) -> None:
    ledger_path = root / GPU_LEDGER_REL
    lock_path = root / GPU_LEDGER_LOCK_REL
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows = _read_gpu_ledger(root)
        usage = _ledger_usage(rows)
        if attempt_id not in {value for value in usage["open_attempt_ids"]}:
            raise FitContractError(f"cannot close absent/open GPU reservation: {attempt_id}")
        close = {"experiment_id": "EXP-050", "run_id": RUN_ID, "attempt_id": attempt_id,
                 "event": "attempt_finished", "gpu_hours_charged": charged,
                 "status": status, "error": error, "timestamp_utc": utc_now()}
        encoded = (json.dumps(close, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
        descriptor = os.open(ledger_path, os.O_WRONLY | os.O_APPEND)
        try:
            if os.write(descriptor, encoded) != len(encoded):
                raise OSError("short EXP-050 GPU ledger close append")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write_run_manifest(run_dir: Path, preflight_status: str | None = None) -> dict[str, Any]:
    receipts = _read_attempt_receipts(run_dir)
    manifest = {"experiment_id": "EXP-050", "run_id": RUN_ID,
                "status": receipts[-1]["status"] if receipts else (preflight_status or "PREFLIGHT_ONLY"),
                "attempt_count": len(receipts),
                "settled_subtask_gpu_hours": sum(float(row["charged_gpu_hours"]) for row in receipts),
                "cumulative_subtask_cap_gpu_hours": ATTEMPT_CAP_GPU_HOURS,
                "attempts": [{"attempt_id": row["attempt_id"], "status": row["status"],
                              "charged_gpu_hours": row["charged_gpu_hours"],
                              "code_sha256": row.get("code_sha256"),
                              "preflight_protocol_sha256": row.get("preflight_protocol_sha256"),
                              "attempt_dir": f"attempts/{row['attempt_id']}"}
                             for row in receipts],
                "preflight_protocol_sha256": (sha256_file(run_dir / "training_protocol_snapshot.json")
                                              if (run_dir / "training_protocol_snapshot.json").is_file() else None),
                "updated_at_utc": utc_now()}
    atomic_json(run_dir / "run_manifest.json", manifest)
    return manifest


def _assert_retry_snapshot_compatible(previous: dict[str, Any], current: dict[str, Any]) -> None:
    for field in ("experiment_id", "run_id", "model", "gate", "identity_key_validation",
                  "drug_feature_validation", "data_axes", "access_boundary"):
        if previous.get(field) != current.get(field):
            raise FitContractError(f"engineering retry changed frozen preflight field: {field}")
    previous_parameters = dict(previous.get("parameters", {}))
    current_parameters = dict(current.get("parameters", {}))
    for mutable in ("correction_note_before_retry",):
        previous_parameters.pop(mutable, None)
        current_parameters.pop(mutable, None)
    if previous_parameters != current_parameters:
        raise FitContractError("engineering retry changed frozen XPert parameters or selection protocol")
    previous_hashes = dict(previous.get("source_hashes", {}))
    current_hashes = dict(current.get("source_hashes", {}))
    previous_hashes.pop("fit_script", None)
    current_hashes.pop("fit_script", None)
    if previous_hashes != current_hashes:
        raise FitContractError("engineering retry changed a frozen data/model/config input hash")


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


@dataclass
class Scope:
    fit_train: pd.DataFrame
    fit_dev: pd.DataFrame
    heldout: pd.DataFrame
    train_key_ids: dict[str, set[str]]
    dev_key_ids: dict[str, set[str]]
    heldout_key_ids: dict[tuple[str, str], set[str]]
    train_keys: set[str]
    dev_keys: set[str]
    common_keys: set[str]
    all_source_objects: set[str]


def _rows_by_key(frame: pd.DataFrame, role: str) -> dict[str, set[str]]:
    selected = frame.loc[frame["split_role"].astype(str).eq(role)]
    result: dict[str, set[str]] = {}
    for row in selected.itertuples(index=False):
        key = str(row.full_inchikey).strip().upper()
        if not FULL_KEY.fullmatch(key):
            raise FitContractError(f"invalid complete InChIKey in gate support: {key}")
        ids = set(parse_ids(row.condition_ids_after_exclusion,
                            label=f"{role}/{key}"))
        if str(row.support_status) == "SUPPORTED" and not ids:
            raise FitContractError(f"supported gate key has no condition IDs: {key}/{role}")
        if ids:
            if key in result:
                raise FitContractError(f"duplicate gate drug/split row: {key}/{role}")
            result[key] = ids
    return result


def collect_scope(root: Path, gate_dir: Path) -> tuple[Scope, dict[str, Any]]:
    snapshot_path = gate_dir / "protocol_snapshot.json"
    support_path = gate_dir / "train_dev_drug_support.tsv"
    common_path = gate_dir / "common_object_exposure_support.tsv"
    manifest_path = gate_dir / "run_manifest.json"
    snapshot = read_json(snapshot_path)
    manifest = read_json(manifest_path)
    if snapshot.get("status") != "GO_TO_MANAGER_FOR_NEXT_FROZEN_ATTEMPT":
        raise FitContractError("gate snapshot is not approved for manager review")
    scope_block = snapshot.get("training_scope_freeze", {})
    common_keys_list = scope_block.get("train_inner_dev_common_supported_full_inchikeys")
    if not isinstance(common_keys_list, list):
        raise FitContractError("gate snapshot lacks frozen train/dev common drug keys")
    common_keys = {str(value).strip().upper() for value in common_keys_list}
    if len(common_keys) != len(common_keys_list) or len(common_keys) != 78:
        raise FitContractError("gate common supported key list is not exactly 78 unique keys")
    if any(not FULL_KEY.fullmatch(key) for key in common_keys):
        raise FitContractError("gate common set contains an incomplete InChIKey")

    expected_hashes = manifest.get("output_sha256", {})
    for name, path in (("protocol_snapshot.json", snapshot_path),
                       ("train_dev_drug_support.tsv", support_path),
                       ("common_object_exposure_support.tsv", common_path)):
        expected = expected_hashes.get(name)
        if not expected or sha256_file(path) != expected:
            raise FitContractError(f"gate output checksum mismatch or missing: {name}")
    dev_record = gate_dir / "attempt_001_deviation_record.json"
    if dev_record.is_file():
        expected = expected_hashes.get("attempt_001_deviation_record.json")
        if expected and sha256_file(dev_record) != expected:
            raise FitContractError("gate deviation receipt checksum mismatch")

    support = pd.read_csv(support_path, sep="\t", dtype="string", keep_default_na=False)
    required = {"full_inchikey", "lincs_pert_ids", "split_role",
                "n_independent_conditions_after_exclusion", "condition_ids_after_exclusion",
                "support_status"}
    if required - set(support.columns):
        raise FitContractError("gate drug support table is missing required columns")
    train_ids = _rows_by_key(support, "train")
    dev_ids = _rows_by_key(support, "inner_dev")
    expected_train_keys = int(snapshot.get("summary", {}).get("target_drugs_with_train_support", -1))
    expected_dev_keys = int(snapshot.get("summary", {}).get("target_drugs_with_inner_dev_support", -1))
    if (len(train_ids) != expected_train_keys or len(dev_ids) != expected_dev_keys
            or train_ids.keys() & dev_ids.keys() != common_keys):
        raise FitContractError("gate train/dev drug support does not reproduce the frozen 111/82/78 sets")
    # This table is scoped to drugs observed in the direct-sensitivity cohorts. It
    # defines the 78-key selection intersection, but it is not the full independent
    # source-object training whitelist (9,889 conditions) or full dev cohort (2,402).
    dev_common = {key: dev_ids[key] for key in sorted(common_keys)}

    common = pd.read_csv(common_path, sep="\t", dtype="string", keep_default_na=False)
    required_common = {"source_origin_id", "full_inchikey", "condition_ids",
                       "excluded_from_xpert_fit_and_selection",
                       "n_exact_10uM_24h_lincs_conditions"}
    if required_common - set(common.columns):
        raise FitContractError("gate source-object support table is missing required columns")
    source_objects = set(common["source_origin_id"].astype(str).str.strip()) - {""}
    target_map: dict[tuple[str, str], set[str]] = {}
    for row in common.itertuples(index=False):
        object_id = str(row.source_origin_id).strip()
        key = str(row.full_inchikey).strip().upper()
        if not object_id or not FULL_KEY.fullmatch(key):
            raise FitContractError("gate held-out support has an invalid object or complete drug key")
        if not _truth(row.excluded_from_xpert_fit_and_selection):
            raise FitContractError("gate source-cohort row is not marked excluded")
        ids = set(parse_ids(row.condition_ids, label=f"heldout/{object_id}/{key}"))
        count = int(row.n_exact_10uM_24h_lincs_conditions)
        if count != len(ids):
            raise FitContractError(f"held-out condition count disagrees with listed keys: {object_id}/{key}")
        if ids:
            pair = (object_id, key)
            if pair in target_map:
                raise FitContractError(f"duplicate held-out object/key row: {pair}")
            target_map[pair] = ids
    expected_pairs = int(snapshot.get("summary", {}).get(
        "direct_pair_rows_with_exact_10uM_24h_LINCS_support", -1))
    if len(target_map) != expected_pairs or expected_pairs != 462:
        raise FitContractError("gate held-out supported pair count differs from 462")
    target_objects = {obj for obj, _ in target_map}
    expected_source_objects = int(snapshot.get("summary", {}).get("source_patient_ids_excluded", -1))
    if not target_objects or len(source_objects) != expected_source_objects or expected_source_objects != 34:
        raise FitContractError("gate held-out pair objects or direct-source exclusion cohort differ from the frozen gate")

    train_frame = pd.DataFrame([
        {"condition_id": cid, "split_role": "train", "full_inchikey": key}
        for key, ids in train_ids.items() for cid in sorted(ids)
    ])
    dev_frame = pd.DataFrame([
        {"condition_id": cid, "split_role": "inner_dev", "full_inchikey": key}
        for key, ids in dev_common.items() for cid in sorted(ids)
    ])
    heldout_frame = pd.DataFrame([
        {"condition_id": cid, "split_role": "heldout_source",
         "source_object_id": obj, "full_inchikey": key}
        for (obj, key), ids in target_map.items() for cid in sorted(ids)
    ])
    train_set = set(train_frame["condition_id"])
    dev_set = set(dev_frame["condition_id"])
    target_set = set(heldout_frame["condition_id"])
    if (len(train_set) != len(train_frame) or len(dev_set) != len(dev_frame)
            or len(target_set) != len(heldout_frame)
            or train_set & dev_set or train_set & target_set or dev_set & target_set):
        raise FitContractError("gate condition_id whitelists overlap or contain duplicates")
    scope = Scope(train_frame, dev_frame, heldout_frame, train_ids, dev_common,
                  target_map, set(train_ids), common_keys, common_keys, source_objects)
    train_support_ids = set().union(*train_ids.values()) if train_ids else set()
    dev_support_ids = set().union(*dev_ids.values()) if dev_ids else set()
    dev_common_ids = set().union(*dev_common.values()) if dev_common else set()
    facts = {
        "gate_snapshot_sha256": sha256_file(snapshot_path),
        "gate_manifest_sha256": sha256_file(manifest_path),
        "train_dev_drug_support_sha256": sha256_file(support_path),
        "common_object_exposure_support_sha256": sha256_file(common_path),
        "common_key_sha256": hash_ids(common_keys),
        "direct_target_train_condition_id_sha256": hash_ids(train_set),
        "direct_target_common_dev_condition_id_sha256": hash_ids(dev_set),
        "heldout_condition_id_sha256": hash_ids(target_set),
        "n_direct_target_train_keys": len(train_ids),
        "n_direct_target_dev_keys": len(dev_ids),
        "n_common_train_dev_keys": len(common_keys),
        "n_train_conditions": int(snapshot["summary"]["remaining_exact_10uM_24h_train_conditions"]),
        "n_inner_dev_all_supported_conditions": int(snapshot["summary"]["remaining_exact_10uM_24h_inner_dev_conditions"]),
        "n_inner_dev_common_key_conditions": len(dev_common_ids),
        "n_train_direct_target_key_conditions": len(train_support_ids),
        "n_inner_dev_direct_target_key_conditions": len(dev_support_ids),
        "n_heldout_source_pairs": len(target_map),
        "n_heldout_source_conditions": len(target_set),
        "n_heldout_source_objects_with_exposure": len(target_objects),
        "n_direct_source_objects_excluded": len(source_objects),
        "direct_source_object_id_sha256": hash_ids(source_objects),
    }
    return scope, facts


def read_index_members(path: Path) -> set[int]:
    values = np.load(path, allow_pickle=False)
    if values.ndim != 1:
        raise FitContractError(f"membership mask must be one-dimensional: {path.name}")
    if values.dtype.kind == "b":
        return set(map(int, np.flatnonzero(values)))
    if values.dtype.kind not in "iu":
        raise FitContractError(f"membership mask must be boolean or integer indices: {path.name}")
    members = set(map(int, values.tolist()))
    if len(members) != len(values) or any(value < 0 for value in members):
        raise FitContractError(f"membership mask has duplicates or negative positions: {path.name}")
    return members


def read_whitelisted_conditions(path: Path, condition_ids: set[str]) -> pd.DataFrame:
    schema = pq.read_schema(path)
    missing = set(COND_COLUMNS) - set(schema.names)
    if missing:
        raise FitContractError(f"prepared condition metadata lacks {sorted(missing)}")
    field_type = schema.field("condition_id").type
    if pa.types.is_integer(field_type):
        values: list[Any] = [int(value) for value in condition_ids]
    elif pa.types.is_string(field_type) or pa.types.is_large_string(field_type):
        values = sorted(condition_ids)
    else:
        raise FitContractError(f"unsupported condition_id parquet type: {field_type}")
    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(columns=COND_COLUMNS,
                             filter=ds.field("condition_id").isin(values))
    frame = table.to_pandas()
    frame["condition_id"] = frame["condition_id"].astype(str).str.strip()
    frame["source_row"] = pd.to_numeric(frame["source_row"], errors="raise").astype(np.int64)
    frame["cell_id"] = frame["cell_id"].astype(str).str.strip()
    frame["pert_id"] = frame["pert_id"].astype(str).str.strip()
    frame["pert_dose"] = pd.to_numeric(frame["pert_dose"], errors="raise")
    frame["pert_time"] = pd.to_numeric(frame["pert_time"], errors="raise")
    if frame["condition_id"].duplicated().any() or set(frame["condition_id"]) != condition_ids:
        raise FitContractError("filtered condition read did not exactly match the gate whitelist")
    if frame["source_row"].duplicated().any() or frame["cell_id"].eq("").any():
        raise FitContractError("whitelisted conditions have duplicate source rows or empty cell IDs")
    return frame


def build_identity_map(root: Path, scope: Scope, facts: dict[str, Any],
                       output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    gate = root / GATE_REL
    support = pd.read_csv(gate / "train_dev_drug_support.tsv", sep="\t",
                          dtype="string", keep_default_na=False)
    common = pd.read_csv(gate / "common_object_exposure_support.tsv", sep="\t",
                         dtype="string", keep_default_na=False)
    expected_key_for_id: dict[str, str] = {}
    expected_object_for_id: dict[str, str] = {}
    registered_fit_key_for_id: dict[str, str] = {}
    registered_fit_role_for_id: dict[str, str] = {}
    for row in support.itertuples(index=False):
        role = str(row.split_role)
        key = str(row.full_inchikey).strip().upper()
        allowed_role = (role == "train" and str(row.support_status) == "SUPPORTED") or (
            role == "inner_dev" and key in scope.common_keys and str(row.support_status) == "SUPPORTED")
        if not allowed_role:
            continue
        for condition_id in parse_ids(row.condition_ids_after_exclusion,
                                      label=f"{role}/{key}"):
            if condition_id in registered_fit_key_for_id:
                raise FitContractError(f"condition ID assigned to multiple direct-target keys: {condition_id}")
            registered_fit_key_for_id[condition_id] = key
            registered_fit_role_for_id[condition_id] = "train" if role == "train" else "inner_dev"
    for row in common.itertuples(index=False):
        object_id = str(row.source_origin_id).strip()
        key = str(row.full_inchikey).strip().upper()
        if not object_id or not _truth(row.excluded_from_xpert_fit_and_selection):
            continue
        for condition_id in parse_ids(row.condition_ids, label=f"heldout/{object_id}/{key}"):
            if condition_id in expected_object_for_id:
                raise FitContractError(f"held-out condition ID maps to multiple source objects: {condition_id}")
            expected_key_for_id[condition_id] = key
            expected_object_for_id[condition_id] = object_id
    prepared = root / MASK_DIR / "conditions.parquet"
    schema = pq.read_schema(prepared)
    if set(COND_COLUMNS) - set(schema.names):
        raise FitContractError("prepared conditions lack the registered response-blind projection")
    conditions = pq.read_table(prepared, columns=COND_COLUMNS).to_pandas()
    conditions["condition_id"] = conditions["condition_id"].astype(str).str.strip()
    conditions["source_row"] = pd.to_numeric(conditions["source_row"], errors="raise").astype(np.int64)
    conditions["cell_id"] = conditions["cell_id"].astype(str).str.strip()
    conditions["pert_id"] = conditions["pert_id"].astype(str).str.strip()
    conditions["pert_dose"] = pd.to_numeric(conditions["pert_dose"], errors="raise")
    conditions["pert_time"] = pd.to_numeric(conditions["pert_time"], errors="raise")
    if (conditions["condition_id"].eq("").any() or conditions["condition_id"].duplicated().any()
            or conditions["source_row"].duplicated().any() or conditions["cell_id"].eq("").any()):
        raise FitContractError("prepared response-blind condition IDs/rows are missing or duplicated")
    train_members = read_index_members(root / MASK_DIR / "train121.npy")
    dev_members = read_index_members(root / MASK_DIR / "validation.npy")
    n_conditions = len(conditions)
    expected_members = set(range(n_conditions))
    if (train_members & dev_members or train_members | dev_members != expected_members
            or len(train_members) != 32094 or len(dev_members) != 4954):
        raise FitContractError("registered train/inner-dev membership does not cover the frozen condition rows")
    roles = np.full(n_conditions, "", dtype=object)
    roles[list(train_members)] = "train"
    roles[list(dev_members)] = "inner_dev"
    conditions["split_role"] = roles

    model_path = root / MODEL_CSV_REL
    model = pd.read_csv(model_path, usecols=MODEL_COLUMNS, dtype="string", keep_default_na=False)
    for column in MODEL_COLUMNS:
        model[column] = model[column].astype(str).str.strip()
    stripped_map: dict[str, set[str]] = {}
    cellname_map: dict[str, set[str]] = {}
    for row in model.itertuples(index=False):
        if not row.PatientID:
            continue
        if row.StrippedCellLineName:
            stripped_map.setdefault(row.StrippedCellLineName.casefold(), set()).add(row.PatientID)
        if row.CellLineName:
            cellname_map.setdefault(row.CellLineName.casefold(), set()).add(row.PatientID)
    object_for_cell: dict[str, str | None] = {}
    for cell in conditions["cell_id"].astype(str).unique():
        candidates = stripped_map.get(cell.casefold()) or cellname_map.get(cell.casefold()) or set()
        object_for_cell[cell] = next(iter(candidates)) if len(candidates) == 1 else None

    conditions["source_object_id"] = conditions["cell_id"].map(object_for_cell)
    target_mask = conditions["source_object_id"].isin(scope.all_source_objects)
    exact_mask = (np.isclose(conditions["pert_dose"].to_numpy(dtype=float), 10.0, rtol=0.0, atol=1e-12)
                  & np.isclose(conditions["pert_time"].to_numpy(dtype=float), 24.0, rtol=0.0, atol=1e-12))
    conditions["exact_exposure"] = exact_mask
    fit_candidates = conditions.loc[
        conditions["split_role"].isin(["train", "inner_dev"])
        & conditions["exact_exposure"]
        & conditions["source_object_id"].notna()
        & ~target_mask
    ].copy()
    heldout_ids = set(expected_key_for_id)
    heldout_pert_ids = set(conditions.loc[conditions["condition_id"].isin(heldout_ids), "pert_id"].astype(str))
    pert_ids_needed = set(fit_candidates["pert_id"].astype(str)) | heldout_pert_ids
    pert_path = root / PERT_INFO_REL
    pert_info = pd.read_csv(pert_path, sep="\t", compression="gzip",
                            usecols=["pert_id", "inchi_key"], dtype="string",
                            keep_default_na=False)
    pert_info["pert_id"] = pert_info["pert_id"].str.strip()
    pert_info["inchi_key"] = pert_info["inchi_key"].str.strip().str.upper()
    pert_info = pert_info.loc[pert_info["pert_id"].isin(pert_ids_needed)]
    pert_keys = pert_info.groupby("pert_id")["inchi_key"].agg(lambda values: set(values)).to_dict()

    key_by_pert = {pert: next(iter(keys)) for pert, keys in pert_keys.items()
                   if len(keys) == 1 and FULL_KEY.fullmatch(next(iter(keys)))}
    fit_candidates["full_inchikey"] = fit_candidates["pert_id"].map(key_by_pert)
    if fit_candidates["full_inchikey"].isna().any():
        n_missing = int(fit_candidates["full_inchikey"].isna().sum())
        raise FitContractError(f"{n_missing} gate-eligible fit/dev conditions lack a unique full InChIKey")
    if fit_candidates["full_inchikey"].astype(str).map(lambda value: not FULL_KEY.fullmatch(value)).any():
        raise FitContractError("a gate-eligible fit/dev condition has an incomplete InChIKey")
    fit_candidates["condition_id"] = fit_candidates["condition_id"].astype(str)
    for cid, key in registered_fit_key_for_id.items():
        hits = fit_candidates.loc[fit_candidates["condition_id"].eq(cid)]
        if len(hits) != 1 or str(hits.iloc[0]["full_inchikey"]).upper() != key:
            raise FitContractError(f"direct-target gate support does not match full fit whitelist: {cid}")
        if hits.iloc[0]["split_role"] != registered_fit_role_for_id[cid]:
            raise FitContractError(f"direct-target gate support has the wrong registered split role: {cid}")

    expected_train_count = facts["n_train_conditions"]
    expected_dev_count = facts["n_inner_dev_all_supported_conditions"]
    if (int(fit_candidates["split_role"].eq("train").sum()) != expected_train_count
            or int(fit_candidates["split_role"].eq("inner_dev").sum()) != expected_dev_count):
        raise FitContractError("response-blind full fit whitelist does not reproduce the frozen 9,889/2,402 counts")
    dev_common = fit_candidates.loc[
        fit_candidates["split_role"].eq("inner_dev")
        & fit_candidates["full_inchikey"].isin(scope.common_keys)
    ].copy()
    if (int(fit_candidates.loc[fit_candidates["split_role"].eq("train"), "source_object_id"].nunique()) != 61
            or int(fit_candidates.loc[fit_candidates["split_role"].eq("inner_dev"), "source_object_id"].nunique()) != 6
            or not scope.common_keys.issubset(set(fit_candidates.loc[
                fit_candidates["split_role"].eq("train"), "full_inchikey"]))
            or not scope.common_keys.issubset(set(dev_common["full_inchikey"]))):
        raise FitContractError("full fit whitelist lost an independent object or a frozen common drug key")

    mapped_parts = [fit_candidates.loc[fit_candidates["split_role"].eq("train")], dev_common]
    for cid, key in expected_key_for_id.items():
        hits = conditions.loc[conditions["condition_id"].astype(str).eq(cid)]
        if len(hits) != 1:
            raise FitContractError(f"held-out gate condition ID is missing or duplicated: {cid}")
        source = hits.iloc[0]
        if (not source["exact_exposure"] or object_for_cell[str(source.cell_id)] != expected_object_for_id[cid]
                or len(pert_keys.get(str(source.pert_id), set())) != 1
                or key not in pert_keys[str(source.pert_id)]):
            raise FitContractError(f"held-out source condition does not match its gate identity/key: {cid}")
        mapped_parts.append(pd.DataFrame([{
            "condition_id": cid, "source_row": int(source.source_row), "split_role": "heldout_source",
            "source_object_id": expected_object_for_id[cid], "full_inchikey": key,
            "cell_id": str(source.cell_id), "pert_id": str(source.pert_id),
            "pert_dose": float(source.pert_dose), "pert_time": float(source.pert_time),
        }]))
    mapped = pd.concat(mapped_parts, ignore_index=True)
    if mapped["source_row"].min() < 0:
        raise FitContractError("whitelisted source_row is negative")
    if mapped["source_row"].duplicated().any():
        raise FitContractError("whitelisted condition rows map to duplicate response-matrix positions")
    fit = mapped.loc[mapped["split_role"].isin(["train", "inner_dev"])]
    heldout = mapped.loc[mapped["split_role"].eq("heldout_source")]
    train_objects = set(fit.loc[fit.split_role.eq("train"), "source_object_id"])
    dev_objects = set(fit.loc[fit.split_role.eq("inner_dev"), "source_object_id"])
    heldout_objects = set(heldout["source_object_id"])
    if len(train_objects) != 61 or len(dev_objects) != 6 or train_objects & dev_objects:
        raise FitContractError("mapped train/dev PatientID counts or disjointness are invalid")
    if train_objects & scope.all_source_objects or dev_objects & scope.all_source_objects:
        raise FitContractError("one direct-sensitivity source object entered train/dev whitelist")
    if heldout_objects != {obj for obj, _ in scope.heldout_key_ids}:
        raise FitContractError("held-out mapped object set differs from gate's supported source object set")

    fields = ["condition_id", "source_row", "split_role", "source_object_id", "full_inchikey",
              "cell_id", "pert_id", "pert_dose", "pert_time"]
    output = mapped[fields].sort_values(["split_role", "source_object_id", "full_inchikey", "condition_id"])
    map_path = output_dir / "fit_condition_identity_key_map.tsv"
    write_tsv(map_path, output)

    denominators = []
    for (role, key), group in mapped.groupby(["split_role", "full_inchikey"], sort=True):
        denominators.append({"split_role": role, "full_inchikey": key,
                             "n_conditions": int(group.condition_id.nunique()),
                             "n_source_objects": int(group.source_object_id.nunique())})
    denom_path = output_dir / "drug_support_denominators.tsv"
    write_tsv(denom_path, pd.DataFrame(denominators))
    map_receipt = {
        "status": "PASS_RESPONSE_BLIND_KEY_VALIDATION",
        "mask_membership_source": ["train121.npy", "validation.npy"],
        "n_whitelisted_conditions": int(len(mapped)),
        "n_fit_train_conditions": int(mapped.split_role.eq("train").sum()),
        "n_fit_inner_dev_common_key_conditions": int(mapped.split_role.eq("inner_dev").sum()),
        "n_heldout_source_conditions": int(mapped.split_role.eq("heldout_source").sum()),
        "n_train_patient_ids": len(train_objects), "n_inner_dev_patient_ids": len(dev_objects),
        "n_train_dev_patient_id_overlap": 0,
        "n_direct_source_patient_ids_excluded_from_fit": len(scope.all_source_objects),
        "n_heldout_patient_ids_with_exact_exposure": len(heldout_objects),
        "train_patient_ids_disjoint_from_direct_source": True,
        "inner_dev_patient_ids_disjoint_from_direct_source": True,
        "condition_identity_key_map_sha256": sha256_file(map_path),
        "drug_support_denominators_sha256": sha256_file(denom_path),
        "source_row_sha256": hash_ids(set(map(str, mapped.source_row))),
        "condition_id_whitelist_sha256": hash_ids(set(mapped.condition_id.astype(str))),
        "response_delta_values_read": False,
        "control_expression_values_read": False,
        "sensitivity_values_read": False,
        "outer_or_locked_eval_fields_read": False,
    }
    return mapped, map_receipt


def validate_drug_features(root: Path, mapped: pd.DataFrame) -> dict[str, Any]:
    xpert = root / "data/external/xpert_source"
    feature_path = xpert / "processed_data/l1000_sdst_drugs_info_8432.csv"
    feature_rows = pd.read_csv(feature_path, usecols=["pert_id", "inchi_key", "drug_node_idx"],
                               dtype={"pert_id": "string", "inchi_key": "string"},
                               keep_default_na=False)
    feature_rows["pert_id"] = feature_rows["pert_id"].str.strip()
    feature_rows["inchi_key"] = feature_rows["inchi_key"].str.strip().str.upper()
    allowed = mapped[["pert_id", "full_inchikey", "split_role"]].drop_duplicates()
    feature_rows = feature_rows.loc[feature_rows["pert_id"].isin(set(allowed["pert_id"]))].copy()
    if feature_rows["pert_id"].duplicated().any():
        duplicate_map = feature_rows.groupby("pert_id").agg(
            n_key=("inchi_key", "nunique"), n_node=("drug_node_idx", "nunique"))
        if (duplicate_map["n_key"] > 1).any() or (duplicate_map["n_node"] > 1).any():
            raise FitContractError("XPert source table maps a needed pert_id to multiple drug features")
        feature_rows = feature_rows.drop_duplicates("pert_id")
    merged = allowed.merge(feature_rows, on="pert_id", how="left", validate="many_to_one")
    merged["drug_node_idx"] = pd.to_numeric(merged["drug_node_idx"], errors="coerce")
    exact = merged["inchi_key"].eq(merged["full_inchikey"])
    train_dev = merged["split_role"].isin(["train", "inner_dev"])
    if not exact.loc[train_dev].all() or merged.loc[train_dev, "drug_node_idx"].isna().any():
        missing = merged.loc[train_dev & ~exact, "full_inchikey"].nunique()
        raise FitContractError(f"XPert exact structure feature missing for fit/dev conditions; keys={missing}")
    if not exact.loc[~train_dev].all() or merged.loc[~train_dev, "drug_node_idx"].isna().any():
        bad_target = merged.loc[~train_dev & ~exact, "full_inchikey"].nunique()
        raise FitContractError(f"XPert exact structure feature missing for held-out source keys; keys={bad_target}")
    if merged.duplicated(["pert_id", "split_role"]).any():
        # Same pert_id may occur under multiple role/key rows only if its structure is identical.
        grouped = merged.groupby("pert_id").agg(n_key=("full_inchikey", "nunique"),
                                                 n_node=("drug_node_idx", "nunique"))
        if (grouped["n_key"] > 1).any() or (grouped["n_node"] > 1).any():
            raise FitContractError("XPert feature mapping is inconsistent across gate keys")
    node_max = int(pd.to_numeric(feature_rows["drug_node_idx"], errors="raise").max())
    unimol_path = xpert / "processed_data/all_drugs_unimol_arr.npy"
    hg_path = xpert / "HG_data/saved_embedding/HG_drug_embeddings.npy"
    ppi_path = xpert / "processed_data/PPI_gene_vector_128d.npy"
    unimol_shape = adapter.read_npy_shape(unimol_path)
    hg_shape = adapter.read_npy_shape(hg_path)
    ppi_shape = adapter.read_npy_shape(ppi_path)
    if (len(unimol_shape) != 3 or unimol_shape[1:] != (122, 514)
            or node_max >= unimol_shape[0] or len(hg_shape) != 2
            or hg_shape[0] != unimol_shape[0] or ppi_shape != (978, 128)):
        raise FitContractError("XPert feature-axis shapes do not satisfy the official 978/122 contract")
    node_ids = sorted(set(pd.to_numeric(merged["drug_node_idx"], errors="raise").astype(int)))
    unimol = np.load(unimol_path, mmap_mode="r", allow_pickle=False)
    drug_hg = np.load(hg_path, mmap_mode="r", allow_pickle=False)
    selected_unimol = np.asarray(unimol[node_ids], dtype=np.float32)
    selected_hg = np.asarray(drug_hg[node_ids], dtype=np.float32)
    if not np.isfinite(selected_unimol).all() or not np.isfinite(selected_hg).all():
        raise FitContractError("selected exact XPert drug features contain non-finite values")
    ppi = np.load(ppi_path, mmap_mode="r", allow_pickle=False)
    if not np.isfinite(np.asarray(ppi, dtype=np.float32)).all():
        raise FitContractError("official XPert PPI gene vectors contain non-finite values")
    return {
        "n_gate_pert_id_role_mappings": int(len(merged)),
        "n_unique_gate_pert_ids": int(merged.pert_id.nunique()),
        "n_exact_model_feature_drug_nodes": len(node_ids),
        "n_unmapped_fit_or_dev_keys": 0,
        "n_unmapped_heldout_keys": 0,
        "unimol_shape": list(unimol_shape), "drug_hg_shape": list(hg_shape),
        "ppi_shape": list(ppi_shape),
        "selected_unimol_rows_sha256": hashlib.sha256(selected_unimol.tobytes()).hexdigest(),
        "selected_drug_hg_rows_sha256": hashlib.sha256(selected_hg.tobytes()).hexdigest(),
        "ppi_values_sha256": hashlib.sha256(np.asarray(ppi, dtype=np.float32).tobytes()).hexdigest(),
        "source_drug_table_sha256": sha256_file(feature_path),
        "unimol_asset_sha256": sha256_file(unimol_path),
        "drug_hg_asset_sha256": sha256_file(hg_path),
        "ppi_asset_sha256": sha256_file(ppi_path),
    }


def run_preflight(root: Path, correction_note: str | None = None) -> dict[str, Any]:
    run_dir = root / RUN_REL
    gate_dir = root / GATE_REL
    run_dir.mkdir(parents=True, exist_ok=True)
    prior_attempts = _read_attempt_receipts(run_dir)
    ledger_usage = _ledger_usage(_read_gpu_ledger(root))
    prior_charge = sum(float(value["charged_gpu_hours"]) for value in prior_attempts)
    if (ledger_usage["open_attempt_ids"]
            or ledger_usage["fit_attempt_count"] != len(prior_attempts)
            or abs(ledger_usage["fit_settled_gpu_hours"] - prior_charge) > 1e-9):
        raise FitContractError("XPert attempt receipts and append-only budget ledger need reconciliation")
    if prior_attempts:
        if not correction_note or len(correction_note.strip()) < 24:
            raise FitContractError("a documented engineering correction is required before another preflight")
        if prior_attempts[-1].get("status") != "FAILED":
            raise FitContractError("only a failed fit can be followed by an engineering retry")
    deviation_path = run_dir / "preflight_deviation_receipt.json"
    deviation = read_json(deviation_path)
    if deviation.get("run_id") != RUN_ID or deviation.get("source_path") != PARAM_SOURCE.as_posix():
        raise FitContractError("the D3 deviation receipt is missing or bound to another source")
    if deviation.get("source_sha256") != PARAM_SOURCE_SHA256:
        raise FitContractError("the D3 receipt protocol hash does not match the parameter source")
    scope, gate_facts = collect_scope(root, gate_dir)
    mapped, identity_facts = build_identity_map(root, scope, gate_facts, run_dir)
    if prior_attempts:
        previous = prior_attempts[-1]
        if identity_facts["condition_identity_key_map_sha256"] != previous.get("condition_identity_key_map_sha256"):
            raise FitContractError("engineering retry changed the frozen condition/object/drug-key whitelist")
        if identity_facts["drug_support_denominators_sha256"] != previous.get("drug_support_denominators_sha256"):
            raise FitContractError("engineering retry changed the frozen train/dev/heldout denominators")
    drug_facts = validate_drug_features(root, mapped)

    cost_path = gate_dir / "cost_preflight.json"
    cost_preflight = read_json(cost_path)
    if (float(cost_preflight.get("conservative_estimated_gpu_hours", -1)) != EXPECTED_FIT_GPU_HOURS
            or float(cost_preflight.get("registered_next_xpert_attempt_cap_gpu_hours", -1)) != ATTEMPT_CAP_GPU_HOURS):
        raise FitContractError("frozen XPert cost estimate/subcap differs from the approved gate")

    paths = adapter.ProjectPaths.from_root(root)
    config = adapter.load_model_config(paths)
    gene_path = paths.contract_dir / "genes.json"
    gene_map_path = paths.audit_dir / "source_to_target_gene_indices.json"
    gene_ids = json.loads(gene_path.read_text(encoding="utf-8"))
    source_gene_indices = json.loads(gene_map_path.read_text(encoding="utf-8"))
    gene_map = adapter.validate_gene_map(gene_ids, source_gene_indices)
    config["_source_root"] = str(paths.xpert_dir)
    condition_manifest = read_json(paths.condition_dir / "manifest.json")
    input_manifest = read_json(paths.input_dir / "manifest.json")
    delta_shape = adapter.read_npy_shape(paths.condition_dir / "condition_delta.npy")
    control_shape = adapter.read_npy_shape(paths.input_dir / "condition_control.npy")
    if (delta_shape != control_shape or len(delta_shape) != 2 or delta_shape[1] != 978
            or delta_shape[0] != int(condition_manifest.get("n_conditions", -1))):
        raise FitContractError("response/control array headers do not match registered matrix axes")
    max_row = int(mapped.source_row.max())
    if max_row >= delta_shape[0]:
        raise FitContractError("a gated source_row lies beyond the registered matrix axis")

    script_path = Path(__file__).resolve()
    source_files = {
        "fit_script": script_path,
        "exp037_xpert_adapter": root / "scripts/modeling/exp037_xpert_fold.py",
        "official_model": paths.xpert_dir / "models/model_XPert.py",
        "official_model_utils": paths.xpert_dir / "models/model_utils.py",
        "official_dataset": paths.xpert_dir / "datasets/MyDataset.py",
        "official_loss_utils": paths.xpert_dir / "utils.py",
        "official_model_config": paths.xpert_dir / "configs/config_l1000.yaml",
        "gene_axis": gene_path,
        "source_to_target_gene_map": gene_map_path,
        "train_membership": root / MASK_DIR / "train121.npy",
        "inner_dev_membership": root / MASK_DIR / "validation.npy",
        "gate_cost_preflight": cost_path,
        "experiment_protocol_config": root / "configs/experiments/EXP-050.yaml",
    }
    source_hashes = {name: {"path": path.relative_to(root).as_posix(),
                            "sha256": sha256_file(path)} for name, path in source_files.items()}
    param_source_hash = sha256_file(root / PARAM_SOURCE)
    if param_source_hash != PARAM_SOURCE_SHA256:
        raise FitContractError("frozen parameter source protocol hash changed")
    params = json.loads((root / PARAM_SOURCE).read_text(encoding="utf-8"))
    selected_params = {key: params[key] for key in FROZEN_PARAMS}
    if selected_params != FROZEN_PARAMS:
        raise FitContractError("registered XPert fixed parameters changed from the frozen values")
    dev_receipt = {
        "access_deviation_receipt_sha256": sha256_file(deviation_path),
        "parameter_source_path": PARAM_SOURCE.as_posix(),
        "parameter_source_sha256": param_source_hash,
        "selected_parameter_fields": selected_params,
        "prior_protocol_wallclock_hours": params.get("wallclock_hours"),
        "current_attempt_cap_gpu_hours": ATTEMPT_CAP_GPU_HOURS,
        "prior_row_indices_used": False,
        "prior_gene_map_used": False,
        "prior_input_contract_used": False,
    }
    snapshot = {
        "schema_version": 1,
        "experiment_id": "EXP-050",
        "run_id": RUN_ID,
        "retry": {"prior_attempt_id": prior_attempts[-1]["attempt_id"] if prior_attempts else None,
                   "reason": correction_note},
        "status": "PREFLIGHT_PASS_WAITING_MANAGER_GO",
        "created_at_utc": utc_now(),
        "model": {
            "name": "official XPertNet",
            "initialization": "fresh XPertNet.init_weights(); no legacy checkpoint or optimizer state",
            "prediction_inputs": "control expression bins + exact full-InChIKey XPert drug features + dose/time; treated-expression fields are zero placeholders at prediction",
            "response_axis_gene_count": 978,
            "target_gene_map_sha256": sha256_file(gene_map_path),
            "official_config_sha256": source_hashes["official_model_config"]["sha256"],
        },
        "parameters": {
            "source_protocol": dev_receipt,
            "frozen": {**selected_params, "attempt_cap_gpu_hours": ATTEMPT_CAP_GPU_HOURS},
            "optimizer": "Adam",
            "early_stop_metric": "inner-dev object-macro NMSE over exactly the 78 full-InChIKey train/dev intersection; lower is better; earliest minimum selected",
            "training_loss_rows": "all 9,889 independent train conditions with exact exposure and complete full-InChIKey identity; direct-source target-key support is a subset",
            "training_dev_score_rows": "train fit diagnostic and inner-dev selection use only common 78 keys",
            "cumulative_xpert_fit_cap_gpu_hours": ATTEMPT_CAP_GPU_HOURS,
            "estimated_complete_attempt_gpu_hours": EXPECTED_FIT_GPU_HOURS,
            "necessary_engineering_retry_allowed": True,
            "retry_gate": "each attempt has a separate immutable directory and ledger entry; a failed attempt must be diagnosed and a correction note documented before a necessary engineering retry; no blind retry, model search, architecture/loss/hyperparameter change, or extra candidate; all charges count toward the cumulative 10 GPU-hour subcap and 48 GPU-hour package cap",
            "correction_note_before_retry": correction_note,
        },
        "gate": gate_facts,
        "identity_key_validation": identity_facts,
        "drug_feature_validation": drug_facts,
        "data_axes": {
            "condition_manifest_sha256": sha256_file(paths.condition_dir / "manifest.json"),
            "condition_delta_sha256_declared_only": condition_manifest.get("files_sha256", {}).get("condition_delta.npy"),
            "input_manifest_sha256": sha256_file(paths.input_dir / "manifest.json"),
            "condition_control_sha256_declared_only": input_manifest.get("files_sha256", {}).get("condition_control.npy"),
            "matrix_shape": list(delta_shape),
            "matrix_shape_read_from_header_only": True,
            "response_delta_or_control_values_read_in_preflight": False,
        },
        "source_hashes": source_hashes,
        "gpu": {"device": "PENDING_MANAGER_DEVICE_CHECK", "actual_attempt_charge_gpu_hours": None},
        "access_boundary": {
            "response_blind_condition_metadata_projection_used_to_materialize_whitelist": True,
            "response_blind_condition_metadata_columns": COND_COLUMNS,
            "train_inner_dev_membership_source": [
                (root / MASK_DIR / "train121.npy").relative_to(root).as_posix(),
                (root / MASK_DIR / "validation.npy").relative_to(root).as_posix(),
            ],
            "split_assignments_or_locked_eval_metadata_read": False,
            "prior_protocol_row_indices_or_gene_map_used": False,
            "direct_pharmacology_numeric_labels_read": False,
            "pdo_labels_read": False,
        },
        "input_validation_outputs": {
            "fit_condition_identity_key_map": "fit_condition_identity_key_map.tsv",
            "drug_support_denominators": "drug_support_denominators.tsv",
        },
        "code_sha256": sha256_file(script_path),
    }
    if prior_attempts:
        prior_snapshot_path = (run_dir / "attempts" / prior_attempts[-1]["attempt_id"]
                               / "training_protocol_snapshot.json")
        if not prior_snapshot_path.is_file():
            raise FitContractError("prior attempt lacks its immutable training protocol snapshot")
        _assert_retry_snapshot_compatible(read_json(prior_snapshot_path), snapshot)
    snapshot_path = run_dir / "training_protocol_snapshot.json"
    atomic_json(snapshot_path, snapshot)
    preflight = {
        "status": snapshot["status"],
        "training_protocol_snapshot": snapshot_path.relative_to(root).as_posix(),
        "training_protocol_snapshot_sha256": sha256_file(snapshot_path),
        "fit_condition_identity_key_map_sha256": identity_facts["condition_identity_key_map_sha256"],
        "drug_support_denominators_sha256": identity_facts["drug_support_denominators_sha256"],
        "train_conditions": gate_facts["n_train_conditions"],
        "inner_dev_scoring_conditions": gate_facts["n_inner_dev_common_key_conditions"],
        "common_drug_keys": gate_facts["n_common_train_dev_keys"],
        "heldout_source_pairs": gate_facts["n_heldout_source_pairs"],
        "heldout_source_objects": gate_facts["n_heldout_source_objects_with_exposure"],
        "response_values_read": False,
    }
    atomic_json(run_dir / "preflight_receipt.json", preflight)
    _write_run_manifest(run_dir, preflight["status"])
    return preflight


@dataclass
class MatrixRows:
    mapped: pd.DataFrame
    delta: np.ndarray
    control: np.ndarray
    gene_indices: np.ndarray
    unimol: np.ndarray
    drug_hg: np.ndarray
    drug_nodes: dict[str, int]
    config: dict[str, Any]
    source_root: Path
    device: torch.device
    digitize: Any
    dose_bin: Any
    bins: np.ndarray
    dose_time_map: dict[float, int]

    def prediction_inputs(self, frame: pd.DataFrame) -> tuple[torch.Tensor, ...]:
        rows = frame["source_row"].to_numpy(dtype=np.int64)
        ctl_target_order = np.asarray(self.control[rows], dtype=np.float32)
        if not np.isfinite(ctl_target_order).all():
            raise FitContractError("non-finite control expression in allowed prediction rows")
        ctl = ctl_target_order[:, self.gene_indices]
        n = len(frame)
        zeros = np.zeros_like(ctl, dtype=np.float32)
        ctl_bins = self.digitize(ctl, self.bins, side="one").astype(np.int64)
        node_indices = np.asarray([self.drug_nodes[str(pid)] for pid in frame.pert_id], dtype=np.int64)
        dose = np.asarray([self.dose_bin(float(value)) for value in frame.pert_dose], dtype=np.int64)
        times = np.asarray([self.dose_time_map[float(value)] for value in frame.pert_time], dtype=np.int64)
        to_tensor = lambda value, dtype=None: torch.as_tensor(value, dtype=dtype, device=self.device)
        return (
            to_tensor(zeros, torch.float32), to_tensor(ctl, torch.float32),
            to_tensor(zeros, torch.long), to_tensor(ctl_bins, torch.long),
            to_tensor(np.asarray(self.unimol[node_indices], dtype=np.float32), torch.float32),
            to_tensor(dose, torch.long), to_tensor(times, torch.long),
            to_tensor(node_indices, torch.long), to_tensor(np.zeros(n), torch.long),
            to_tensor(np.zeros(n), torch.long),
        )

    def training_batch(self, frame: pd.DataFrame) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        rows = frame["source_row"].to_numpy(dtype=np.int64)
        ctl_target_order = np.asarray(self.control[rows], dtype=np.float32)
        delta_target_order = np.asarray(self.delta[rows], dtype=np.float32)
        if not np.isfinite(ctl_target_order).all() or not np.isfinite(delta_target_order).all():
            raise FitContractError("non-finite input/response in approved train or inner-dev rows")
        ctl = ctl_target_order[:, self.gene_indices]
        delta = delta_target_order[:, self.gene_indices]
        trt = ctl + delta
        trt_bins = self.digitize(trt, self.bins, side="one").astype(np.int64)
        ctl_bins = self.digitize(ctl, self.bins, side="one").astype(np.int64)
        node_indices = np.asarray([self.drug_nodes[str(pid)] for pid in frame.pert_id], dtype=np.int64)
        dose = np.asarray([self.dose_bin(float(value)) for value in frame.pert_dose], dtype=np.int64)
        times = np.asarray([self.dose_time_map[float(value)] for value in frame.pert_time], dtype=np.int64)
        to_tensor = lambda value, dtype=None: torch.as_tensor(value, dtype=dtype, device=self.device)
        inputs = (
            to_tensor(trt, torch.float32), to_tensor(ctl, torch.float32),
            to_tensor(trt_bins, torch.long), to_tensor(ctl_bins, torch.long),
            to_tensor(np.asarray(self.unimol[node_indices], dtype=np.float32), torch.float32),
            to_tensor(dose, torch.long), to_tensor(times, torch.long),
            to_tensor(node_indices, torch.long), to_tensor(np.zeros(len(frame)), torch.long),
            to_tensor(np.zeros(len(frame)), torch.long),
        )
        return inputs, to_tensor(delta, torch.float32)

    def target_delta(self, frame: pd.DataFrame) -> np.ndarray:
        rows = frame["source_row"].to_numpy(dtype=np.int64)
        target = np.asarray(self.delta[rows], dtype=np.float32)
        return target[:, np.argsort(self.gene_indices)]


def _load_runtime_rows(root: Path, mapped: pd.DataFrame, device: torch.device) -> MatrixRows:
    paths = adapter.ProjectPaths.from_root(root)
    config = adapter.load_model_config(paths)
    config["_source_root"] = str(paths.xpert_dir)
    gene_ids = json.loads((paths.contract_dir / "genes.json").read_text(encoding="utf-8"))
    gene_map = json.loads((paths.audit_dir / "source_to_target_gene_indices.json").read_text(encoding="utf-8"))
    gene_indices = adapter.validate_gene_map(gene_ids, gene_map)
    source_drugs_path = paths.xpert_dir / "processed_data/l1000_sdst_drugs_info_8432.csv"
    source_drugs = pd.read_csv(source_drugs_path, usecols=["pert_id", "inchi_key", "drug_node_idx"],
                               dtype={"pert_id": "string", "inchi_key": "string"}, keep_default_na=False)
    source_drugs["pert_id"] = source_drugs["pert_id"].str.strip()
    source_drugs["inchi_key"] = source_drugs["inchi_key"].str.strip().str.upper()
    needed = set(mapped.pert_id.astype(str))
    source_drugs = source_drugs.loc[source_drugs.pert_id.isin(needed)]
    if source_drugs.pert_id.duplicated().any():
        grouped = source_drugs.groupby("pert_id").agg(n_key=("inchi_key", "nunique"),
                                                       n_node=("drug_node_idx", "nunique"))
        if (grouped.n_key > 1).any() or (grouped.n_node > 1).any():
            raise FitContractError("selected XPert drug feature map is not one-to-one")
        source_drugs = source_drugs.drop_duplicates("pert_id")
    expected = mapped.drop_duplicates(["pert_id", "full_inchikey"])
    check = expected.merge(source_drugs, on="pert_id", how="left", validate="one_to_one")
    if check["drug_node_idx"].isna().any() or not check["full_inchikey"].eq(
            check["inchi_key"].astype(str).str.upper()).all():
        raise FitContractError("selected exact-key XPert features changed after preflight")
    drug_nodes = dict(zip(check.pert_id.astype(str), pd.to_numeric(check.drug_node_idx).astype(int)))
    delta = np.load(paths.condition_dir / "condition_delta.npy", mmap_mode="r", allow_pickle=False)
    control = np.load(paths.input_dir / "condition_control.npy", mmap_mode="r", allow_pickle=False)
    unimol = np.load(paths.xpert_dir / "processed_data/all_drugs_unimol_arr.npy", mmap_mode="r", allow_pickle=False)
    drug_hg = np.load(paths.xpert_dir / "HG_data/saved_embedding/HG_drug_embeddings.npy", mmap_mode="r", allow_pickle=False)
    from pathlib import Path as _Path
    digitize = adapter.load_upstream_functions(paths.xpert_dir / "datasets/MyDataset.py", ("_digitize",))["_digitize"]
    dose_bin = adapter.load_upstream_functions(paths.xpert_dir / "datasets/MyDataset.py",
                                               ("assign_pert_dose_corrected",))["assign_pert_dose_corrected"]
    low = float(config["dataset"]["min_value"])
    high = float(config["dataset"]["max_value"])
    n_bins = int(config["dataset"]["n_bins"])
    bins = np.quantile(np.asarray([low, high]), np.linspace(0.0, 1.0, n_bins - 1))
    return MatrixRows(mapped, delta, control, gene_indices, unimol, drug_hg,
                      drug_nodes, config, paths.xpert_dir, device, digitize, dose_bin,
                      bins, {6.0: 1, 24.0: 2})


def _predict(model: torch.nn.Module, rows: MatrixRows, frame: pd.DataFrame,
             batch_size: int) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[start:start + batch_size]
            outputs = model(rows.prediction_inputs(batch))
            prediction = outputs[2]
            if not torch.isfinite(prediction).all():
                raise FitContractError("non-finite XPert response prediction")
            parts.append(prediction.detach().cpu().numpy())
    if not parts:
        return np.empty((0, 978), dtype=np.float32)
    source_order = np.concatenate(parts).astype(np.float32, copy=False)
    return source_order[:, np.argsort(rows.gene_indices)]


def _scale_from_train(rows: MatrixRows, train: pd.DataFrame, batch_size: int) -> np.ndarray:
    n = 0
    sums = np.zeros(978, dtype=np.float64)
    squares = np.zeros(978, dtype=np.float64)
    for start in range(0, len(train), batch_size * 16):
        block = train.iloc[start:start + batch_size * 16]
        values = rows.target_delta(block).astype(np.float64)
        if not np.isfinite(values).all():
            raise FitContractError("non-finite train-only scale values")
        sums += values.sum(axis=0)
        squares += np.square(values).sum(axis=0)
        n += len(values)
    scale = np.sqrt(np.maximum(squares / n - np.square(sums / n), 0.0))
    scale = np.maximum(scale, 1e-6).astype(np.float32)
    return scale


def _metrics(frame: pd.DataFrame, pred: np.ndarray, truth: np.ndarray, scale: np.ndarray,
             drug_mean: dict[str, np.ndarray] | None = None) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if len(frame) != len(pred) or len(frame) != len(truth):
        raise FitContractError("prediction/truth/condition-key row counts differ")
    scaled_true = truth / scale[None, :]
    scaled_pred = pred / scale[None, :]
    zero = np.zeros_like(truth)
    mean_pred = np.stack([drug_mean[key] for key in frame.full_inchikey], axis=0) if drug_mean else None
    row = frame[["condition_id", "source_object_id", "full_inchikey", "split_role"]].copy().reset_index(drop=True)
    row["model_nmse"] = np.square(scaled_pred - scaled_true).mean(axis=1)
    row["zero_nmse"] = np.square(scaled_true).mean(axis=1)
    row["model_raw_mse"] = np.square(pred - truth).mean(axis=1)
    row["zero_raw_mse"] = np.square(truth).mean(axis=1)
    if mean_pred is not None:
        row["train_drug_mean_nmse"] = np.square((mean_pred - truth) / scale[None, :]).mean(axis=1)
        row["train_drug_mean_raw_mse"] = np.square(mean_pred - truth).mean(axis=1)
    else:
        row["train_drug_mean_nmse"] = np.nan
        row["train_drug_mean_raw_mse"] = np.nan
    by_key = row.groupby(["source_object_id", "full_inchikey"], as_index=False).agg(
        n_conditions=("condition_id", "nunique"), model_nmse=("model_nmse", "mean"),
        zero_nmse=("zero_nmse", "mean"), model_raw_mse=("model_raw_mse", "mean"),
        zero_raw_mse=("zero_raw_mse", "mean"),
        train_drug_mean_nmse=("train_drug_mean_nmse", "mean"),
        train_drug_mean_raw_mse=("train_drug_mean_raw_mse", "mean"))
    by_object = by_key.groupby("source_object_id", as_index=False).agg(
        n_drugs=("full_inchikey", "nunique"), n_conditions=("n_conditions", "sum"),
        model_nmse=("model_nmse", "mean"), zero_nmse=("zero_nmse", "mean"),
        model_raw_mse=("model_raw_mse", "mean"), zero_raw_mse=("zero_raw_mse", "mean"),
        train_drug_mean_nmse=("train_drug_mean_nmse", "mean"),
        train_drug_mean_raw_mse=("train_drug_mean_raw_mse", "mean"))
    result = {
        "n_conditions": int(len(row)), "n_source_objects": int(row.source_object_id.nunique()),
        "n_full_inchikeys": int(row.full_inchikey.nunique()),
        "object_macro_model_nmse": float(by_object.model_nmse.mean()),
        "object_macro_zero_nmse": float(by_object.zero_nmse.mean()),
        "object_macro_train_drug_mean_nmse": (float(by_object.train_drug_mean_nmse.mean())
                                                if by_object.train_drug_mean_nmse.notna().all() else None),
        "object_macro_model_raw_mse": float(by_object.model_raw_mse.mean()),
        "object_macro_zero_raw_mse": float(by_object.zero_raw_mse.mean()),
        "object_macro_train_drug_mean_raw_mse": (float(by_object.train_drug_mean_raw_mse.mean())
                                                  if by_object.train_drug_mean_raw_mse.notna().all() else None),
        "metric_direction": "lower is better",
        "aggregation": "condition mean within source_object_id x full InChIKey; equal key weight within object; equal object weight overall",
    }
    return result, row, by_object


def _mean_by_key(rows: MatrixRows, train: pd.DataFrame, batch_size: int) -> dict[str, np.ndarray]:
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for start in range(0, len(train), batch_size * 16):
        block = train.iloc[start:start + batch_size * 16]
        values = rows.target_delta(block)
        for key in sorted(set(block.full_inchikey.astype(str))):
            positions = np.flatnonzero(block.full_inchikey.astype(str).to_numpy() == key)
            sums.setdefault(key, np.zeros(978, dtype=np.float64))
            sums[key] += values[positions].sum(axis=0, dtype=np.float64)
            counts[key] = counts.get(key, 0) + len(positions)
    return {key: (value / counts[key]).astype(np.float32) for key, value in sums.items()}


def _predict_curve_metrics(model: torch.nn.Module, rows: MatrixRows, frame: pd.DataFrame,
                           scale: np.ndarray, batch_size: int) -> tuple[float, np.ndarray, pd.DataFrame]:
    prediction = _predict(model, rows, frame, batch_size)
    truth = rows.target_delta(frame)
    score, row_metrics, _ = _metrics(frame, prediction, truth, scale)
    return float(score["object_macro_model_nmse"]), prediction, row_metrics


def _train_once(root: Path, expected_snapshot_hash: str, device_name_required: str,
                attempt_id: str, correction_note: str | None) -> dict[str, Any]:
    run_dir = root / RUN_REL
    snapshot_path = run_dir / "training_protocol_snapshot.json"
    snapshot = read_json(snapshot_path)
    if sha256_file(snapshot_path) != expected_snapshot_hash:
        raise FitContractError("training protocol snapshot differs from the reviewed digest")
    if snapshot.get("status") != "PREFLIGHT_PASS_WAITING_MANAGER_GO":
        raise FitContractError("training preflight has not passed")
    code_hash = sha256_file(Path(__file__).resolve())
    if code_hash != snapshot.get("code_sha256"):
        raise FitContractError("training code changed after the reviewed preflight snapshot")
    mapped_path = run_dir / "fit_condition_identity_key_map.tsv"
    denominator_path = run_dir / "drug_support_denominators.tsv"
    if sha256_file(mapped_path) != snapshot["identity_key_validation"]["condition_identity_key_map_sha256"]:
        raise FitContractError("frozen condition identity/key map changed")
    if sha256_file(denominator_path) != snapshot["identity_key_validation"]["drug_support_denominators_sha256"]:
        raise FitContractError("frozen train/dev/heldout denominators changed")
    receipts = _read_attempt_receipts(run_dir)
    expected_attempt_id = f"attempt_{len(receipts) + 1:03d}"
    if attempt_id != expected_attempt_id or not re.fullmatch(r"attempt_[0-9]{3}", attempt_id):
        raise FitContractError(f"next XPert attempt must be {expected_attempt_id}")
    if receipts:
        if not correction_note or len(correction_note.strip()) < 24:
            raise FitContractError("a diagnosed cause and scoped correction note is required before an engineering retry")
        if receipts[-1].get("preflight_protocol_sha256") == expected_snapshot_hash:
            raise FitContractError("a retry requires a refreshed, reviewed protocol/code snapshot")
        if snapshot.get("parameters", {}).get("correction_note_before_retry") != correction_note:
            raise FitContractError("retry correction note differs from the reviewed preflight snapshot")
    elif correction_note:
        raise FitContractError("do not record a retry correction on the first attempt")
    attempt_dir = run_dir / "attempts" / attempt_id
    if attempt_dir.exists():
        raise FitContractError(f"attempt output directory already exists; inspect it before continuing: {attempt_dir}")
    ledger_preview = _ledger_usage(_read_gpu_ledger(root))
    receipts_charged = sum(float(row["charged_gpu_hours"]) for row in receipts)
    if (ledger_preview["open_attempt_ids"]
            or abs(receipts_charged - ledger_preview["fit_settled_gpu_hours"]) > 1e-9
            or len(receipts) != ledger_preview["fit_attempt_count"]):
        raise FitContractError("XPert attempt receipts disagree with the append-only GPU ledger")

    mapped = pd.read_csv(mapped_path, sep="\t", dtype={
        "condition_id": str, "source_object_id": str, "full_inchikey": str, "cell_id": str,
        "pert_id": str})
    required_map = {"condition_id", "source_row", "split_role", "source_object_id", "full_inchikey",
                    "cell_id", "pert_id", "pert_dose", "pert_time"}
    if required_map - set(mapped.columns):
        raise FitContractError(f"frozen identity/key map lacks training inputs: {sorted(required_map-set(mapped.columns))}")
    train = mapped.loc[mapped.split_role.eq("train")].reset_index(drop=True)
    dev = mapped.loc[mapped.split_role.eq("inner_dev")].reset_index(drop=True)
    heldout = mapped.loc[mapped.split_role.eq("heldout_source")].reset_index(drop=True)
    device = torch.device(device_name_required)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise FitContractError("approved XPert attempt requires the verified CUDA device")
    if torch.cuda.get_device_name(device).lower().find("a5000") < 0:
        raise FitContractError(f"expected A5000, found {torch.cuda.get_device_name(device)}")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes / total_bytes < 0.85:
        raise FitContractError("A5000 is not sufficiently idle for the XPert attempt reservation")

    device_name = torch.cuda.get_device_name(device)
    reserved, usage_before = _reserve_gpu_attempt(root, attempt_id, device_name, os.getpid())

    attempt_started = time.perf_counter()
    attempt = {"experiment_id": "EXP-050", "run_id": RUN_ID, "attempt_id": attempt_id,
               "attempt_number": len(receipts) + 1, "status": "RUNNING",
               "started_at_utc": utc_now(), "device": device_name,
               "reserved_gpu_hours": reserved, "estimated_gpu_hours": EXPECTED_FIT_GPU_HOURS,
               "prior_subtask_settled_gpu_hours": receipts_charged,
               "cumulative_subtask_cap_gpu_hours": ATTEMPT_CAP_GPU_HOURS,
               "prior_package_settled_gpu_hours": usage_before["settled_gpu_hours"],
               "cumulative_package_cap_gpu_hours": 48.0,
               "charged_gpu_hours": None, "max_epochs": FROZEN_PARAMS["max_epochs"],
               "correction_note": correction_note,
               "preflight_protocol_sha256": expected_snapshot_hash,
               "condition_identity_key_map_sha256": snapshot["identity_key_validation"]["condition_identity_key_map_sha256"],
               "drug_support_denominators_sha256": snapshot["identity_key_validation"]["drug_support_denominators_sha256"],
               "code_sha256": code_hash}
    result: dict[str, Any] = {"status": "FAILED", "run_id": RUN_ID, "attempt_id": attempt_id}
    current_epoch: int | None = None
    current_batch: int | None = None
    try:
        attempt_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(snapshot_path, attempt_dir / "training_protocol_snapshot.json")
        shutil.copy2(mapped_path, attempt_dir / "fit_condition_identity_key_map.tsv")
        shutil.copy2(denominator_path, attempt_dir / "drug_support_denominators.tsv")
        atomic_json(attempt_dir / "attempt_receipt.json", attempt)
        rows = _load_runtime_rows(root, mapped, device)
        config = rows.config
        model, mse_loss, pcc_loss = adapter.load_xpert_model(
            adapter.ProjectPaths.from_root(root), config, device=device,
            seed=FROZEN_PARAMS["seed"])
        optimizer = torch.optim.Adam(model.parameters(), lr=FROZEN_PARAMS["learning_rate"],
                                     weight_decay=FROZEN_PARAMS["weight_decay"])
        scale = _scale_from_train(rows, train, FROZEN_PARAMS["batch_size"])
        gate_snapshot = read_json(root / GATE_REL / "protocol_snapshot.json")
        common_keys = set(gate_snapshot["training_scope_freeze"][
            "train_inner_dev_common_supported_full_inchikeys"])
        train_common = train.loc[train.full_inchikey.isin(common_keys)].reset_index(drop=True)
        dev_common = dev.loc[dev.full_inchikey.isin(common_keys)].reset_index(drop=True)
        if (len(train) != 9889 or len(train_common) == 0 or len(dev_common) == 0
                or set(dev_common.full_inchikey) != common_keys or len(common_keys) != 78):
            raise FitContractError("train/dev key restriction changed after protocol freeze")
        best_score = math.inf
        best_epoch = 0
        stale = 0
        curve_path = attempt_dir / "training_curve.jsonl"
        def enforce_attempt_cap(stage: str) -> None:
            elapsed_now = time.perf_counter() - attempt_started
            if elapsed_now >= reserved * 3600.0 - 120.0:
                raise FitContractError(f"cumulative GPU-hour budget safety stop reached during {stage}")

        with curve_path.open("w", encoding="utf-8") as curve:
            for epoch in range(FROZEN_PARAMS["max_epochs"]):
                current_epoch, current_batch = epoch + 1, None
                model.train()
                rng = np.random.default_rng(FROZEN_PARAMS["seed"] + epoch)
                order = rng.permutation(len(train))
                total_loss, n_train = 0.0, 0
                for start in range(0, len(order), FROZEN_PARAMS["batch_size"]):
                    enforce_attempt_cap("training")
                    current_batch = start // FROZEN_PARAMS["batch_size"] + 1
                    batch_frame = train.iloc[order[start:start + FROZEN_PARAMS["batch_size"]]]
                    inputs, target = rows.training_batch(batch_frame)
                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = adapter.build_official_loss(
                        model(inputs), tuple(FROZEN_PARAMS["loss_weights"]), mse_loss, pcc_loss)
                    if not torch.isfinite(loss):
                        raise FitContractError("non-finite official XPert loss; preserve diagnostics before any retry")
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all()
                           for p in model.parameters()):
                        raise FitContractError("non-finite XPert gradient; preserve diagnostics before any retry")
                    optimizer.step()
                    total_loss += float(loss.detach())
                    n_train += len(batch_frame)
                enforce_attempt_cap("train-side curve scoring")
                train_score, _, _ = _predict_curve_metrics(
                    model, rows, train_common, scale, FROZEN_PARAMS["batch_size"])
                enforce_attempt_cap("inner-dev curve scoring")
                dev_score, _, _ = _predict_curve_metrics(
                    model, rows, dev_common, scale, FROZEN_PARAMS["batch_size"])
                improved = dev_score < best_score
                if improved:
                    best_score, best_epoch, stale = dev_score, epoch + 1, 0
                    torch.save({"model": model.state_dict(), "epoch": best_epoch,
                                "validation_nmse": best_score}, attempt_dir / "best.tmp")
                    os.replace(attempt_dir / "best.tmp", attempt_dir / "best.pt")
                else:
                    stale += 1
                item = {"epoch": epoch + 1, "official_loss_per_condition": total_loss / max(1, n_train),
                        "train_common_78_object_macro_nmse": train_score,
                        "inner_dev_78_object_macro_nmse": dev_score,
                        "best_epoch": best_epoch, "stale_epochs": stale,
                        "learning_rate": FROZEN_PARAMS["learning_rate"],
                        "elapsed_seconds": time.perf_counter() - attempt_started}
                curve.write(json.dumps(item, sort_keys=True) + "\n")
                curve.flush()
                os.fsync(curve.fileno())
                print(json.dumps(item, sort_keys=True), flush=True)
                if stale >= FROZEN_PARAMS["patience"]:
                    break
            if best_epoch == 0:
                raise FitContractError("no finite inner-dev selection epoch")

        enforce_attempt_cap("frozen checkpoint reload and held-out prediction")
        best_path = attempt_dir / "best.pt"
        best_sha = sha256_file(best_path)
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        # Freeze and persist all held-out predictions before loading their delta labels.
        heldout_prediction = _predict(model, rows, heldout, FROZEN_PARAMS["batch_size"])
        np.save(attempt_dir / "heldout_source_prediction_target_order.npy", heldout_prediction)
        with (attempt_dir / "heldout_source_prediction_target_order.npy").open("rb+") as stream:
            os.fsync(stream.fileno())
        heldout_prediction_sha = sha256_file(attempt_dir / "heldout_source_prediction_target_order.npy")
        heldout_keys = heldout[["condition_id", "source_row", "source_object_id", "full_inchikey"]].copy()
        write_tsv(attempt_dir / "heldout_source_prediction_keys.tsv", heldout_keys)
        enforce_attempt_cap("held-out response access and final scoring")
        # This first target-delta access is after the frozen checkpoint and predictions are durable.
        heldout_truth = rows.target_delta(heldout)
        np.save(attempt_dir / "heldout_source_true_delta_target_order.npy", heldout_truth)
        train_mean = _mean_by_key(rows, train, FROZEN_PARAMS["batch_size"])
        common_target = heldout.loc[heldout.full_inchikey.isin(common_keys)].reset_index(drop=True)
        common_idx = heldout.index[heldout.full_inchikey.isin(common_keys)].to_numpy()
        if len(common_target):
            common_truth = heldout_truth[common_idx]
            common_pred = heldout_prediction[common_idx]
            common_mean = {key: train_mean[key] for key in common_keys if key in train_mean}
            if set(common_target.full_inchikey) - set(common_mean):
                raise FitContractError("train drug mean missing for a frozen common key")
            common_metrics, common_per_row, common_per_object = _metrics(
                common_target, common_pred, common_truth, scale, common_mean)
            write_tsv(attempt_dir / "heldout_common78_per_condition_metrics.tsv", common_per_row)
            write_tsv(attempt_dir / "heldout_common78_per_object_metrics.tsv", common_per_object)
        else:
            common_metrics = {"status": "NO_HELDOUT_CONDITIONS_ON_COMMON_78_KEYS"}

        train_pred = _predict(model, rows, train_common, FROZEN_PARAMS["batch_size"])
        train_truth = rows.target_delta(train_common)
        dev_pred = _predict(model, rows, dev_common, FROZEN_PARAMS["batch_size"])
        dev_truth = rows.target_delta(dev_common)
        dev_mean = _mean_by_key(rows, train, FROZEN_PARAMS["batch_size"])
        train_score, train_rows_metrics, train_object_metrics = _metrics(
            train_common, train_pred, train_truth, scale, dev_mean)
        dev_score_final, dev_rows_metrics, dev_object_metrics = _metrics(
            dev_common, dev_pred, dev_truth, scale, dev_mean)
        write_tsv(attempt_dir / "train_common78_per_condition_metrics.tsv", train_rows_metrics)
        write_tsv(attempt_dir / "train_common78_per_object_metrics.tsv", train_object_metrics)
        write_tsv(attempt_dir / "inner_dev_common78_per_condition_metrics.tsv", dev_rows_metrics)
        write_tsv(attempt_dir / "inner_dev_common78_per_object_metrics.tsv", dev_object_metrics)
        np.save(attempt_dir / "train_common78_prediction_target_order.npy", train_pred)
        np.save(attempt_dir / "inner_dev_common78_prediction_target_order.npy", dev_pred)
        np.save(attempt_dir / "training_gene_scale_target_order.npy", scale)
        all_target_metrics, all_target_rows, all_target_objects = _metrics(
            heldout, heldout_prediction, heldout_truth, scale,
            {key: train_mean[key] for key in set(heldout.full_inchikey) if key in train_mean}
            if set(heldout.full_inchikey).issubset(train_mean) else None)
        write_tsv(attempt_dir / "heldout_source_per_condition_metrics.tsv", all_target_rows)
        write_tsv(attempt_dir / "heldout_source_per_object_metrics.tsv", all_target_objects)
        result = {
            "status": "COMPLETED_ONE_FRESH_XPERT_FIT",
            "run_id": RUN_ID,
            "attempt_id": attempt_id,
            "selected_epoch": best_epoch,
            "n_epochs_completed": sum(1 for _ in curve_path.open(encoding="utf-8")),
            "selected_inner_dev_object_macro_nmse": best_score,
            "train_common78_metrics": train_score,
            "inner_dev_common78_metrics_at_best": dev_score_final,
            "heldout_source_common78_metrics": common_metrics,
            "heldout_source_all_supported_metrics": all_target_metrics,
            "heldout_prediction_sha256_before_target_delta_read": heldout_prediction_sha,
            "best_checkpoint_sha256": best_sha,
            "n_train_conditions": int(len(train)),
            "n_inner_dev_common78_conditions": int(len(dev_common)),
            "n_heldout_source_conditions": int(len(heldout)),
            "n_heldout_source_objects": int(heldout.source_object_id.nunique()),
            "target_delta_read_after_checkpoint_frozen": True,
            "direct_sensitivity_values_read": False,
            "pdo_treated_labels_read": False,
            "outer_or_locked_eval_data_read": False,
        }
        atomic_json(attempt_dir / "fit_result.json", result)
    except Exception as exc:
        result.update({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)})
        atomic_json(attempt_dir / "failure_diagnostic.json", {
            "status": "FAILED", "attempt_id": attempt_id,
            "error_type": type(exc).__name__, "error": str(exc),
            "epoch": current_epoch, "batch": current_batch,
            "elapsed_seconds_at_failure": time.perf_counter() - attempt_started,
            "retry_policy": "diagnose this failure first; any necessary engineering retry requires a scoped correction note and remaining cumulative budget",
        })
        raise
    finally:
        final_status = result.get("status", "FAILED")
        error = result.get("error")
        output_hashes = {path.relative_to(attempt_dir).as_posix(): sha256_file(path)
                         for path in sorted(attempt_dir.rglob("*"))
                         if path.is_file() and path.name not in {"attempt_receipt.json", "attempt_manifest.json"}}
        elapsed = time.perf_counter() - attempt_started
        charged = elapsed / 3600.0
        attempt.update({"status": final_status, "finished_at_utc": utc_now(),
                        "elapsed_seconds": elapsed, "charged_gpu_hours": charged,
                        "within_reserved_gpu_hours": charged <= reserved,
                        "within_cumulative_subtask_cap_gpu_hours": receipts_charged + charged <= ATTEMPT_CAP_GPU_HOURS,
                        "within_cumulative_package_cap_gpu_hours": usage_before["settled_gpu_hours"] + charged <= 48.0,
                        "error": error, "output_sha256": output_hashes})
        atomic_json(attempt_dir / "attempt_receipt.json", attempt)
        _finish_gpu_attempt(root, attempt_id, charged, final_status, error)
        atomic_json(attempt_dir / "attempt_manifest.json", {
            "run_id": RUN_ID, "attempt_id": attempt_id, "status": final_status,
            "preflight_protocol_sha256": expected_snapshot_hash,
            "code_sha256": code_hash, "correction_note": correction_note,
            "charged_gpu_hours": charged, "cumulative_subtask_gpu_hours": receipts_charged + charged,
            "cumulative_package_gpu_hours_lower_bound": usage_before["settled_gpu_hours"] + charged,
            "output_sha256": output_hashes,
        })
        _write_run_manifest(run_dir)
        if charged > reserved or receipts_charged + charged > ATTEMPT_CAP_GPU_HOURS:
            raise FitContractError("actual XPert attempt exceeded its reserved or cumulative GPU-hour budget")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "train-score"), default="preflight")
    parser.add_argument("--expected-protocol-sha256")
    parser.add_argument("--attempt-id", default="attempt_001")
    parser.add_argument("--correction-note",
                        help="required before a diagnosed engineering retry; recorded in the protocol and attempt receipts")
    parser.add_argument("--manager-go", action="store_true",
                        help="required only after the manager's final script/input/budget review")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    if args.mode == "preflight":
        result = run_preflight(root, args.correction_note)
        print(json.dumps(result, sort_keys=True), flush=True)
        return
    if not args.manager_go:
        raise FitContractError("train-score requires the manager's final GO")
    if not args.expected_protocol_sha256:
        raise FitContractError("train-score requires the reviewed frozen protocol snapshot SHA-256")
    result = _train_once(root, args.expected_protocol_sha256, args.device,
                         args.attempt_id, args.correction_note)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__,
                          "error": str(exc)}, sort_keys=True), file=sys.stderr, flush=True)
        raise
