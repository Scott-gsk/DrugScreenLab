"""Safely read back the closed EXP-055 lean-monitor checkpoint on CPU.

This recovery entry never reopens or modifies the closed attempt.  It verifies
only the persisted TRAIN prediction archive and checkpoint, then records a new
readback run with independent TRAIN metrics and evidence-bounded timing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ATTEMPT = ROOT / "artifacts/experiments/EXP-055/20261007_exp055_transigen_lean_monitor_cost_cal_v1/fit_attempt_001"
OUTPUT_RUN_ID = "20261007_exp055_lean_monitor_calibration_readback_v1"
sys.path.insert(0, str(ROOT / "scripts/modeling"))
from exp051_contract_fit_smoke import _torch_tree_exact  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts/evaluation"))
from verify_exp051_low_rank_baselines import metrics  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


class _FakeTensor:
    """Small test double to exercise cross-device normalization without a GPU."""

    def __init__(self, value: list[float], device: str, dtype: str = "float32") -> None:
        self.value = tuple(value)
        self.device = device
        self.dtype = dtype
        self.layout = "strided"
        self.shape = (len(value),)
        self.normalized = False

    def detach(self) -> "_FakeTensor":
        return self

    def to(self, *, device: str) -> "_FakeTensor":
        output = _FakeTensor(list(self.value), device=device, dtype=self.dtype)
        output.normalized = device == "cpu"
        return output


def comparator_unit_checks() -> dict[str, bool]:
    fake_torch = SimpleNamespace(
        Tensor=_FakeTensor,
        equal=lambda left, right: (left.device == right.device
                                   and left.normalized and right.normalized
                                   and left.value == right.value),
    )
    cpu = _FakeTensor([1.25, -2.5], "cpu")
    simulated_cuda = _FakeTensor([1.25, -2.5], "cuda:0")
    changed = _FakeTensor([1.25, -2.499], "cuda:0")
    dtype_changed = _FakeTensor([1.25, -2.5], "cuda:0", dtype="float64")
    checks = {
        "same_values_across_simulated_devices_pass": _torch_tree_exact(fake_torch, cpu, simulated_cuda),
        "changed_value_across_simulated_devices_rejected": not _torch_tree_exact(fake_torch, cpu, changed),
        "dtype_mismatch_rejected": not _torch_tree_exact(fake_torch, cpu, dtype_changed),
    }
    if not all(checks.values()):
        raise AssertionError(f"checkpoint comparator targeted check failed: {checks}")
    return checks


def count_tensors(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return 1
    if isinstance(value, dict):
        return sum(count_tensors(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return sum(count_tensors(child) for child in value)
    return 0


def read_train_predictions(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    # The attempt001 archive contains TRAIN fields only.  Keep pickle disabled
    # and explicitly index no field other than the six TRAIN axes/arrays below.
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "train_condition_key", "train_source_object_id", "train_full_inchikey",
            "genes", "train_target_delta", "train_delta",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise KeyError(f"TRAIN prediction archive is missing required fields: {missing}")
        keys = pd.DataFrame({
            "full_condition_key": archive["train_condition_key"].astype(str, copy=True),
            "source_object_id": archive["train_source_object_id"].astype(str, copy=True),
            "full_inchikey": archive["train_full_inchikey"].astype(str, copy=True),
        })
        genes = archive["genes"].astype(str, copy=True)
        truth = archive["train_target_delta"].astype(np.float32, copy=True)
        prediction = archive["train_delta"].astype(np.float32, copy=True)
        observed_names = sorted(required)
    if truth.shape != (419, 978) or prediction.shape != truth.shape:
        raise ValueError(f"TRAIN prediction/target shape changed: {truth.shape}, {prediction.shape}")
    if len(keys) != 419 or keys.full_condition_key.nunique() != 419:
        raise ValueError("TRAIN condition keys are not 419 unique rows")
    if keys.source_object_id.nunique() != 110:
        raise ValueError("TRAIN object denominator is not 110")
    full_keys = set(keys.full_inchikey.astype(str))
    if len(full_keys) != 5 or any(not re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", key) for key in full_keys):
        raise ValueError("TRAIN must contain five complete full InChIKeys")
    if len(genes) != 978 or len(set(genes.tolist())) != 978:
        raise ValueError("TRAIN gene axis is not exactly 978 unique genes")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise FloatingPointError("TRAIN target or persisted prediction contains a non-finite value")
    return keys, truth, prediction, genes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=OUTPUT_RUN_ID)
    args = parser.parse_args()
    output = ROOT / "artifacts/experiments/EXP-055" / args.run_id
    if output.exists():
        raise FileExistsError(f"refusing to reopen or overwrite readback run: {output}")

    started_wall = time.perf_counter()
    started_utc = utc_now()
    comparator_checks = comparator_unit_checks()
    required_paths = {
        "manifest": SOURCE_ATTEMPT / "fit_smoke_manifest.json",
        "close_receipt": SOURCE_ATTEMPT / "failure_close_receipt.json",
        "progress": SOURCE_ATTEMPT / "attempt_progress.json",
        "trace": SOURCE_ATTEMPT / "lean_monitor_batch_trace.jsonl",
        "resource_ledger": SOURCE_ATTEMPT / "resource_ledger.jsonl",
        "checkpoint": SOURCE_ATTEMPT / "TranSiGen_lean_cost_calibration_epoch.pt",
        "predictions": SOURCE_ATTEMPT / "TranSiGen_lean_cost_calibration_train_predictions.npz",
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("closed-attempt evidence is missing: " + ", ".join(missing))
    output.mkdir(parents=True, exist_ok=False)

    source_hashes = {name: sha256(path) for name, path in required_paths.items()}
    manifest = json.loads(required_paths["manifest"].read_text(encoding="utf-8"))
    close = json.loads(required_paths["close_receipt"].read_text(encoding="utf-8"))
    progress = json.loads(required_paths["progress"].read_text(encoding="utf-8"))
    if (manifest.get("experiment_id") != "EXP-055" or manifest.get("fit_attempt") != "fit_attempt_001"
            or manifest.get("run_id") != "20261007_exp055_transigen_lean_monitor_cost_cal_v1"):
        raise ValueError("manifest does not bind the immutable lean-monitor calibration attempt001")
    if close.get("status") != "FAILED_CLOSED" or close.get("attempt_id") != "fit_attempt_001":
        raise ValueError("attempt001 close receipt is not the expected immutable FAILED_CLOSED record")
    if close.get("keyed_prediction_sha256") != source_hashes["predictions"]:
        raise ValueError("persisted TRAIN prediction checksum differs from the close receipt")
    if close.get("calibration_checkpoint_sha256") != source_hashes["checkpoint"]:
        raise ValueError("persisted checkpoint checksum differs from the close receipt")
    if close.get("inner_dev_numeric_rows_loaded") is not False or close.get("pdo_outer_locked_numeric_values_opened") is not False:
        raise ValueError("closed-attempt receipt does not preserve the required data boundary")

    stage_started = time.perf_counter()
    # Safe CPU loading is required: this does not use unrestricted pickle.
    checkpoint = torch.load(required_paths["checkpoint"], map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise TypeError("safe checkpoint readback lacks a model state_dict")
    if checkpoint.get("checkpoint_role") != "cost_calibration_only_never_formal_initialization":
        raise ValueError("checkpoint is not explicitly labeled as calibration-only")
    state_tensors = count_tensors(checkpoint["state_dict"])
    optimizer_tensors = count_tensors(checkpoint.get("optimizer_state_dict", {}))
    if (state_tensors != 51 or optimizer_tensors != 108
            or int(checkpoint.get("epoch", -1)) != 2):
        raise ValueError("checkpoint epoch/state tensor counts differ from the root safe-readback contract")
    cpu_serialization_exact = False
    temporary_checkpoint_sha = None
    with tempfile.TemporaryDirectory(prefix="checkpoint_readback_", dir=output) as temp_dir:
        temp_path = Path(temp_dir) / "cpu_roundtrip.pt"
        torch.save(checkpoint, temp_path)
        temporary_checkpoint_sha = sha256(temp_path)
        loaded = torch.load(temp_path, map_location="cpu", weights_only=True)
        cpu_serialization_exact = _torch_tree_exact(torch, checkpoint, loaded)
        if not cpu_serialization_exact:
            raise AssertionError("CPU save/load changed one or more checkpoint tree values")
    checkpoint_seconds = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    keys, truth, prediction, genes = read_train_predictions(required_paths["predictions"])
    variance = np.var(truth.astype(np.float64), axis=0, ddof=0)
    by_drug_mean = {
        drug: truth[keys.full_inchikey.astype(str).to_numpy() == drug].mean(axis=0, dtype=np.float64).astype(np.float32)
        for drug in sorted(keys.full_inchikey.astype(str).unique())
    }
    baseline_prediction = np.stack([by_drug_mean[drug] for drug in keys.full_inchikey.astype(str)], axis=0)
    prediction_metrics = metrics(keys, truth, prediction, variance)
    baseline_metrics = metrics(keys, truth, baseline_prediction, variance)
    per_drug = {}
    for drug in sorted(by_drug_mean):
        mask = keys.full_inchikey.astype(str).to_numpy() == drug
        per_drug[drug] = {
            "conditions": int(mask.sum()),
            "objects": int(keys.loc[mask, "source_object_id"].nunique()),
            "train_metric": metrics(keys.loc[mask].reset_index(drop=True), truth[mask], prediction[mask], variance),
            "per_drug_mean_baseline_metric": metrics(keys.loc[mask].reset_index(drop=True), truth[mask],
                                                       baseline_prediction[mask], variance),
        }
    train_seconds = time.perf_counter() - stage_started
    total_seconds = time.perf_counter() - started_wall

    # Existing attempt timestamps/trace do not partition initialization,
    # training, evaluation, and I/O.  Preserve them as lost; never divide the
    # cumulative elapsed field or multiply a one-epoch total into a 300-epoch
    # lower-bound estimate.
    trace_rows = [json.loads(line) for line in required_paths["trace"].read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    batch_time_values = [row.get("batch_seconds") for row in trace_rows]
    timing_evidence = {
        "attempt_wall_upper_bound_seconds": close.get("elapsed_wall_seconds_conservative_upper_bound"),
        "progress_cumulative_gpu_hours_upper_bound": progress.get("elapsed_gpu_hours_upper_bound"),
        "trace_rows": len(trace_rows),
        "trace_batch_seconds_present": sum(value is not None for value in batch_time_values),
        "fresh_initialization_seconds": None,
        "train_update_seconds": None,
        "full_train_evaluation_seconds": None,
        "prediction_io_seconds": None,
        "checkpoint_io_seconds": None,
        "final_inner_dev_93_seconds": None,
        "fixed_epoch_100_or_300_save_seconds": None,
        "missing_components": ["fresh init", "per-epoch TRAIN update", "per-epoch TRAIN evaluation",
                               "prediction/checkpoint I/O split", "fixed 100/300 save", "final 93-row inner-dev evaluation"],
        "cumulative_elapsed_was_split_or_multiplied": False,
        "full_fixed_300_cost_gate_evaluable": False,
        "gpu_restarted": False,
    }

    summary = {
        "status": "CPU_CHECKPOINT_READBACK_AND_TRAIN_RECOMPUTATION_PASS_COST_GATE_UNEVALUABLE",
        "experiment_id": "EXP-055",
        "run_id": args.run_id,
        "source_attempt": "20261007_exp055_transigen_lean_monitor_cost_cal_v1/fit_attempt_001",
        "created_at_utc": started_utc,
        "finished_at_utc": utc_now(),
        "interpretation": "Persisted epoch-2 calibration checkpoint and one-epoch TRAIN outputs were safely read back and independently recomputed. This is not a fit, formal comparison, or transfer result; the failed attempt's lost in-memory payload is not claimed to match the checkpoint elementwise.",
        "source_sha256": source_hashes,
        "denominator": {"train_conditions": int(len(keys)), "train_objects": int(keys.source_object_id.nunique()),
                        "full_inchikey_count": int(keys.full_inchikey.nunique()), "genes": int(len(genes)),
                        "unique_condition_keys": int(keys.full_condition_key.nunique())},
        "checkpoint_readback": {
            "weights_only": True,
            "map_location": "cpu",
            "epoch": int(checkpoint["epoch"]),
            "checkpoint_role": checkpoint["checkpoint_role"],
            "model_state_tensor_count": state_tensors,
            "optimizer_state_tensor_count": optimizer_tensors,
            "cpu_save_load_readback_exact": cpu_serialization_exact,
            "temporary_roundtrip_sha256": temporary_checkpoint_sha,
            "pre_failure_in_memory_payload_available": False,
            "old_memory_payload_elementwise_match_claimed": False,
        },
        "comparator_targeted_checks": comparator_checks,
        "train_metrics_recomputed": prediction_metrics,
        "train_per_drug_mean_baseline_recomputed_on_train": baseline_metrics,
        "relative_train_nmse_reduction_vs_per_drug_mean": (
            1.0 - prediction_metrics["object_macro_nmse"] / baseline_metrics["object_macro_nmse"]
            if prediction_metrics["object_macro_nmse"] is not None and baseline_metrics["object_macro_nmse"] else None),
        "per_drug_train_metrics": per_drug,
        "timing_evidence": timing_evidence,
        "cpu_readback_seconds": {"checkpoint_safe_load_and_cpu_roundtrip": checkpoint_seconds,
                                 "train_npz_identity_and_metrics": train_seconds,
                                 "total_script_wall_seconds": total_seconds,
                                 "gpu_hours": 0.0},
        "forbidden_roles_opened": [],
        "budget_decision": {
            "recovery_gpu_hours_remaining_before_this_readback": 0.964968343800,
            "can_prove_full_300_schedule_plus_20_percent_with_existing_evidence": False,
            "decision": "NO_GPU_RESTART",
            "reason": "Independent stage timings for fresh init, per-epoch training/evaluation, 100/300 saves, final inner-dev evaluation, and close are absent. The one cumulative elapsed record cannot be decomposed or multiplied into a minimum-cost claim."
        },
    }
    summary_path = output / "summary.json"
    write_json(summary_path, summary)
    manifest_out = {
        "run_id": args.run_id,
        "created_at_utc": started_utc,
        "command": "PYTHONPATH=src conda run --no-capture-output -n drugscreening-gpu python scripts/evaluation/restore_exp055_lean_monitor_calibration.py",
        "python_environment": "WSL2 Conda drugscreening-gpu",
        "gpu_hours": 0.0,
        "source_files_sha256": source_hashes,
        "output_files_sha256": {"summary.json": sha256(summary_path)},
        "runner_source_sha256": sha256(Path(__file__).resolve()),
        "comparator_source_sha256": sha256(ROOT / "scripts/modeling/exp051_contract_fit_smoke.py"),
    }
    write_json(output / "manifest.json", manifest_out)
    print(json.dumps({"output_dir": str(output), "status": summary["status"],
                      "train_metrics": prediction_metrics, "cost_gate": summary["budget_decision"],
                      "cpu_seconds": summary["cpu_readback_seconds"]}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
