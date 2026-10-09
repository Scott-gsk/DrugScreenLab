"""Real-data numerical gate for EXP-039 cached inner-fold validation.

This gate uses one deterministic, frozen representative model per real inner
fold. It does not train, select candidates, or access outer-test response rows.
Its receipt requires review before the formal cache path can be unlocked.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from drug_screen.evaluation.reversal_validation import ReversalValidationCache
from drug_screen.evaluation.reversal import reversal_score
from drug_screen.models.conditional_perturbation import SharedEffectInteraction
from scripts.modeling.exp039_b5_fold import (
    SEEDS,
    load_exp037_helpers,
    predict_delta,
    predict_delta_inputs_only,
    reversal_validation_metric,
    verify_validation_cache_gate,
)


ROOT = Path(__file__).resolve().parents[2]
CONDITION_DIR = ROOT / "data/processed/lincs/GSE92742/continuous_conditions_v2"
INPUT_DIR = ROOT / "data/processed/lincs/GSE92742/continuous_inputs_v2"
CONTRACT_DIR = ROOT / "data/processed/lincs/GSE92742/continuous_contract_v2"
CACHE_PATH = ROOT / "data/processed/lincs/GSE92742/exact978_cache_v1/exact978_cache.npy"
SPLIT_PATH = ROOT / "data/splits/lincs_cell_splits_20260922_v2.json"
SIGNATURE_DIR = ROOT / "data/processed/dual_route/EXP-038/gse74602_crc_signature_v2"
AUDIT_PATH = ROOT / "artifacts/experiments/EXP-039/20260923_b5_validation_audit/panel_audit.json"
RECEIPT_PATH = ROOT / "artifacts/experiments/EXP-039/20260923_b5_validation_cache_equivalence_v3/receipt.json"
PRIOR_FOLD0_PROFILE = ROOT / "artifacts/experiments/EXP-039/20260923_b5_rank_v1/diagnostics/performance_profile_full_validation_v1.json"
BATCH_SIZE = 8192
REPRESENTATIVE_RANK = 16


class _AllowedRows:
    """Raise if a guarded condition array is indexed outside outer-train cells."""

    def __init__(self, array, allowed_rows: np.ndarray, label: str):
        self._array = array
        self._allowed = np.asarray(allowed_rows, dtype=bool)
        self._label = label
        self.shape = array.shape
        self.outer_test_reads = 0

    def __getitem__(self, index):
        row_index = index[0] if isinstance(index, tuple) else index
        if isinstance(row_index, slice):
            selected = np.arange(self.shape[0], dtype=np.int64)[row_index]
        else:
            selected_array = np.asarray(row_index)
            if selected_array.dtype == bool:
                selected = np.flatnonzero(selected_array)
            else:
                selected = selected_array.astype(np.int64, copy=False).reshape(-1)
        if selected.size and (np.any(selected < 0) or np.any(selected >= len(self._allowed))):
            raise IndexError(f"{self._label} row index is out of range")
        if selected.size and not np.all(self._allowed[selected]):
            self.outer_test_reads += int(np.count_nonzero(~self._allowed[selected]))
            raise AssertionError(f"attempted to read {self._label} outside outer-train rows")
        return self._array[index]


class _GpuSampler:
    def __init__(self, gpu_index: int, process_id: int):
        self.gpu_index = gpu_index
        self.process_id = process_id
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _query() -> list[dict]:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True)
        rows = []
        for line in result.stdout.splitlines():
            index, name, utilization, memory = [part.strip() for part in line.split(",", 3)]
            rows.append({"index": int(index), "name": name,
                         "utilization_pct": int(utilization), "memory_mib": int(memory)})
        return rows

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rows = self._query()
                target = next((row for row in rows if row["index"] == self.gpu_index), None)
                if target is not None:
                    self.samples.append({"utc": datetime.now(timezone.utc).isoformat(),
                                         **target})
            except Exception as exc:  # captured in receipt; it does not alter model outputs
                self.samples.append({"utc": datetime.now(timezone.utc).isoformat(),
                                     "sampling_error": f"{type(exc).__name__}: {exc}"})
            self._stop.wait(1.0)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_receipt_once(receipt: dict) -> None:
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RECEIPT_PATH.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _code_paths() -> list[Path]:
    return [
        ROOT / "scripts/modeling/exp039_b5_fold.py",
        ROOT / "scripts/modeling/exp037_conditional_fold.py",
        ROOT / "src/drug_screen/evaluation/reversal_validation.py",
        ROOT / "src/drug_screen/evaluation/reversal.py",
        ROOT / "src/drug_screen/models/conditional_perturbation.py",
        ROOT / "tests/evaluation/test_reversal_validation_cache.py",
        ROOT / "tests/modeling/test_exp039_b5_validation_cache_equivalence.py",
    ]


def _input_paths() -> list[Path]:
    return [
        CONDITION_DIR / "manifest.json",
        CONDITION_DIR / "conditions.parquet",
        INPUT_DIR / "manifest.json",
        INPUT_DIR / "drugs.parquet",
        CONTRACT_DIR / "verification.json",
        CONTRACT_DIR / "genes.json",
        CONTRACT_DIR / "controls.parquet",
        CACHE_PATH.parent / "asset_manifest.json",
        SPLIT_PATH,
        SIGNATURE_DIR / "manifest.json",
        SIGNATURE_DIR / "up_genes.json",
        SIGNATURE_DIR / "down_genes.json",
        AUDIT_PATH,
        PRIOR_FOLD0_PROFILE,
    ]


def _gpu_rows() -> list[dict]:
    return _GpuSampler._query()


def _open_fold_data(helpers, base, allowed_rows: np.ndarray,
                    genes: list[str], up: list[str], down: list[str]):
    """Open exposure values only for this outer fold's training cells."""
    condition_ids = base.keys.loc[allowed_rows, "condition_id"].to_numpy()
    exposure_frame = pd.read_parquet(
        INPUT_DIR / "condition_exposure.parquet",
        filters=[("condition_id", "in", condition_ids.tolist())])
    if not exposure_frame["condition_id"].is_unique:
        raise ValueError("filtered exposure condition IDs are not unique")
    aligned = exposure_frame.set_index("condition_id").reindex(condition_ids)
    if aligned.index.hasnans or aligned[["log10_dose_um", "time_hours"]].isna().any().any():
        raise ValueError("filtered outer-train exposure axis is incomplete")
    exposure = np.empty((len(base.keys), 2), dtype=np.float32)
    exposure[allowed_rows] = aligned[["log10_dose_um", "time_hours"]].to_numpy(np.float32)
    data = SimpleNamespace(
        keys=base.keys,
        delta=_AllowedRows(base.delta, allowed_rows, "response delta"),
        state=_AllowedRows(base.state, allowed_rows, "cell-state input"),
        exposure=_AllowedRows(exposure, allowed_rows, "exposure input"),
        fingerprint=base.fingerprint,
        drug_index=base.drug_index,
        _b5_genes=genes,
        _b5_up=up,
        _b5_down=down,
    )
    data.tensors = MethodType(helpers.Arrays.tensors, data)
    return data


def _choose_idle_gpu() -> tuple[int, dict]:
    rows = [row for row in _gpu_rows() if "RTX A5000" in row["name"]]
    idle = [row for row in rows if row["utilization_pct"] == 0]
    if not idle:
        raise RuntimeError("no idle RTX A5000 is available for the cache gate")
    selected = min(idle, key=lambda row: (row["memory_mib"], row["index"]))
    return selected["index"], selected


def _restricted_noise_table(outer_train_cells: set[str]) -> pd.DataFrame:
    """Build noise rows only from outer-train control cache positions."""
    controls = pd.read_parquet(
        CONTRACT_DIR / "controls.parquet",
        filters=[("cell_id", "in", sorted(outer_train_cells))])
    cache = np.load(CACHE_PATH, mmap_mode="r")
    rows = []
    for group, frame in controls.groupby("control_group_id", sort=False):
        values = np.asarray(cache[frame["cache_row"].to_numpy(dtype=np.int64)], dtype=np.float32)
        rows.append({"control_group_id": int(group), "cell_id": frame["cell_id"].iloc[0],
                     "std": values.std(axis=0, dtype=np.float64).astype(np.float32)})
    return pd.DataFrame(rows)


def _assert_frame_exact(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(actual, expected, check_exact=True, check_dtype=True)


def _representative_model(device: torch.device) -> tuple[SharedEffectInteraction, str]:
    """Create the same frozen rank-16 initialization for each inner fold."""
    seed = SEEDS[0]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SharedEffectInteraction(rank=REPRESENTATIVE_RANK).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        tensor = value.detach().cpu().contiguous().numpy()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.tobytes())
    return model, digest.hexdigest()


def _measurement_payload(keys: pd.DataFrame, observed: np.ndarray,
                         genes: list[str], up: list[str], down: list[str]) -> list[dict]:
    """Rebuild fixed cell-drug group identity and measured scores for comparison."""
    groups = keys[["cell_id", "pert_id"]].reset_index(drop=True).groupby(
        ["cell_id", "pert_id"], sort=True).indices
    rows = []
    for (cell_id, drug_id), positions in groups.items():
        positions = np.asarray(positions, dtype=np.int64)
        measured = reversal_score(
            observed[positions].mean(axis=0, dtype=np.float64), genes, up, down)
        rows.append({
            "cell_id": str(cell_id),
            "pert_id": str(drug_id),
            "positions": positions.tolist(),
            "n_conditions": int(len(positions)),
            "measured_reversal_hex": float(measured).hex(),
        })
    return rows


def _cached_measurement_payload(cache: ReversalValidationCache) -> list[dict]:
    return [{
        "cell_id": group.cell_id,
        "pert_id": group.drug_id,
        "positions": group.positions.tolist(),
        "n_conditions": int(len(group.positions)),
        "measured_reversal_hex": float(group.measured_reversal).hex(),
    } for group in cache.drug_groups]


def _score_rows(frame: pd.DataFrame) -> list[dict]:
    return [{
        "cell_id": str(row.cell_id),
        "spearman": float(row.spearman) if np.isfinite(row.spearman) else None,
        "n_drugs": int(row.n_drugs),
        "reason": row.reason,
        "n_conditions": int(row.n_conditions),
    } for row in frame.itertuples(index=False)]


def _real_gate(started_utc: str) -> dict:
    missing = [str(path) for path in _code_paths() + _input_paths() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cache gate inputs or code are missing: {missing}")

    device_index, gpu_before = _choose_idle_gpu()
    device = torch.device(f"cuda:{device_index}")
    if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
        raise RuntimeError("the selected idle RTX A5000 is not visible to PyTorch")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    stage_seconds = {
        "arrays_open_and_axis_load": 0.0,
        "outer_train_noise_cache_read": 0.0,
        "inner_panel_build": 0.0,
        "train_side_preprocess": 0.0,
        "fixed_validation_cache_build": 0.0,
        "representative_model_initialization": 0.0,
        "reference_group_measurement_rebuild": 0.0,
        "legacy_prediction_including_unused_label_transfer": 0.0,
        "input_only_prediction": 0.0,
        "legacy_validation_scoring": 0.0,
        "cached_validation_scoring": 0.0,
    }
    panel_records = []
    failures = []
    max_prediction_error = 0.0
    n_prediction_equal = 0
    n_cell_frames_equal = 0
    n_score_equal = 0
    n_group_payloads_equal = 0
    n_group_rows_compared = 0
    model_state_hashes = []

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    genes = json.loads((CONTRACT_DIR / "genes.json").read_text(encoding="utf-8"))
    up = json.loads((SIGNATURE_DIR / "up_genes.json").read_text(encoding="utf-8"))
    down = json.loads((SIGNATURE_DIR / "down_genes.json").read_text(encoding="utf-8"))
    conditions_manifest = json.loads((CONDITION_DIR / "manifest.json").read_text(encoding="utf-8"))
    inputs_manifest = json.loads((INPUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    cache_asset_manifest = json.loads((CACHE_PATH.parent / "asset_manifest.json").read_text(encoding="utf-8"))
    contract_verification = json.loads((CONTRACT_DIR / "verification.json").read_text(encoding="utf-8"))
    if len(genes) != 978 or min(len(up), len(down)) < 20:
        raise ValueError("the frozen gene or disease-signature axis is invalid")
    if (contract_verification.get("status") != "PASS"
            or cache_asset_manifest.get("cache_file") != CACHE_PATH.name
            or cache_asset_manifest.get("cache_sha256") != conditions_manifest.get("source_cache_sha256")
            or cache_asset_manifest.get("cache_sha256") != inputs_manifest.get("source", {}).get("cache_sha256")):
        raise ValueError("the fixed noise-cache asset binding or verification differs")

    helpers = load_exp037_helpers()
    t0 = time.perf_counter()
    all_keys = pd.read_parquet(CONDITION_DIR / "conditions.parquet")
    drugs = pd.read_parquet(INPUT_DIR / "drugs.parquet")
    drug_position = dict(zip(drugs["pert_id"], drugs["drug_index"]))
    base = SimpleNamespace(
        keys=all_keys,
        delta=np.load(CONDITION_DIR / "condition_delta.npy", mmap_mode="r"),
        state=np.load(INPUT_DIR / "condition_control.npy", mmap_mode="r"),
        fingerprint=np.load(INPUT_DIR / "drug_morgan2048.npy", mmap_mode="r"),
        drug_index=np.asarray([drug_position[drug] for drug in all_keys["pert_id"]],
                              dtype=np.int32),
    )
    if (base.delta.shape != base.state.shape or base.delta.shape[0] != len(all_keys)
            or base.fingerprint.shape[0] != len(drugs)):
        raise ValueError("condition arrays and key axes differ")
    stage_seconds["arrays_open_and_axis_load"] = time.perf_counter() - t0
    outer_train_masks = []
    for fold in split["folds"]:
        mask = all_keys["cell_id"].isin(fold["outer_train_ids"]).to_numpy()
        outer_train_masks.append(mask)

    gpu_assigned_start = time.perf_counter()
    sampler = _GpuSampler(device_index, os.getpid())
    sampler.start()
    gate_start = time.perf_counter()
    total_inner = 0
    try:
        for outer_fold, allowed_rows in zip(split["folds"], outer_train_masks):
            outer_train_cells = set(outer_fold["outer_train_ids"])
            t0 = time.perf_counter()
            noise = _restricted_noise_table(outer_train_cells)
            stage_seconds["outer_train_noise_cache_read"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            data = _open_fold_data(helpers, base, allowed_rows, genes, up, down)
            stage_seconds["arrays_open_and_axis_load"] += time.perf_counter() - t0
            for inner in outer_fold["label_fractions"][0]["inner_folds"]:
                total_inner += 1
                if not set(inner["train_ids"]).issubset(outer_train_cells):
                    raise AssertionError("inner training cells escaped the outer-training cells")
                if not set(inner["validation_ids"]).issubset(outer_train_cells):
                    raise AssertionError("inner validation cells escaped the outer-training cells")
                t0 = time.perf_counter()
                train_index = np.flatnonzero(all_keys["cell_id"].isin(inner["train_ids"]).to_numpy())
                validation_all = np.flatnonzero(
                    all_keys["cell_id"].isin(inner["validation_ids"]).to_numpy())
                seen_drugs = set(all_keys.iloc[train_index]["pert_id"])
                validation_seen = validation_all[
                    all_keys.iloc[validation_all]["pert_id"].isin(seen_drugs).to_numpy()]
                fixed_mask = np.isclose(data.exposure[validation_seen, 0], 1.0,
                                        rtol=0, atol=1e-8)
                fixed_mask &= np.isclose(data.exposure[validation_seen, 1], 24.0,
                                         rtol=0, atol=1e-8)
                validation_panel = validation_seen[fixed_mask]
                validation_cells = set(inner["validation_ids"])
                counts = (all_keys.iloc[validation_panel].groupby("cell_id")["pert_id"].nunique()
                          .reindex(sorted(validation_cells), fill_value=0))
                eligible_cells = set(counts[counts >= 5].index)
                validation_panel = validation_panel[
                    all_keys.iloc[validation_panel]["cell_id"].isin(eligible_cells).to_numpy()]
                stage_seconds["inner_panel_build"] += time.perf_counter() - t0

                audit_outer = next(row for row in audit["folds"]
                                   if int(row["fold"]) == int(outer_fold["fold"]))
                audit_inner = next(row for row in audit_outer["inner_folds"]
                                   if int(row["inner_fold"]) == int(inner["inner_fold"]))
                if len(eligible_cells) != audit_inner[
                        "n_validation_cells_with_at_least_five_seen_drugs"]:
                    failures.append({"fold": int(outer_fold["fold"]),
                                     "inner_fold": int(inner["inner_fold"]),
                                     "check": "frozen_audit_cell_coverage"})
                if len(eligible_cells) < 3:
                    failures.append({"fold": int(outer_fold["fold"]),
                                     "inner_fold": int(inner["inner_fold"]),
                                     "check": "at_least_three_eligible_cells"})
                if not np.all(allowed_rows[train_index]) or not np.all(allowed_rows[validation_panel]):
                    raise AssertionError("outer-test rows entered an inner panel")

                t0 = time.perf_counter()
                prep = helpers.fit_preprocess(data, train_index, noise)
                stage_seconds["train_side_preprocess"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                validation_keys = all_keys.iloc[validation_panel][["cell_id", "pert_id"]].reset_index(drop=True)
                validation_observed = np.asarray(data.delta[validation_panel], dtype=np.float32)
                cache = ReversalValidationCache(validation_keys, validation_observed,
                                                genes, up, down)
                stage_seconds["fixed_validation_cache_build"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                reference_measurements = _measurement_payload(
                    validation_keys, validation_observed, genes, up, down)
                cached_measurements = _cached_measurement_payload(cache)
                group_payload_equal = cached_measurements == reference_measurements
                stage_seconds["reference_group_measurement_rebuild"] += time.perf_counter() - t0
                n_group_rows_compared += len(reference_measurements)
                if not group_payload_equal:
                    failures.append({"fold": int(outer_fold["fold"]),
                                     "inner_fold": int(inner["inner_fold"]),
                                     "check": "fixed_measured_cell_drug_groups_exact"})
                else:
                    n_group_payloads_equal += 1

                record = {
                    "fold": int(outer_fold["fold"]),
                    "inner_fold": int(inner["inner_fold"]),
                    "n_train_conditions": len(train_index),
                    "n_validation_conditions": len(validation_panel),
                    "n_validation_cells": len(eligible_cells),
                    "n_validation_drugs": int(all_keys.iloc[validation_panel]["pert_id"].nunique()),
                    "eligible_validation_cells": sorted(eligible_cells),
                    "validation_unit": "cell_id; this panel contains cell-line groups, not patient samples",
                    "n_cell_drug_groups": len(cache.drug_groups),
                    "fixed_measured_group_payload_exact": group_payload_equal,
                    "fixed_measured_group_payload_sha256": hashlib.sha256(json.dumps(
                        cached_measurements, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")).hexdigest(),
                    "fixed_panel_condition_ids_sha256": hashlib.sha256(
                        all_keys.iloc[validation_panel]["condition_id"].to_numpy(np.int64).tobytes()
                    ).hexdigest(),
                }
                panel_records.append(record)

                t0 = time.perf_counter()
                model, model_state_sha256 = _representative_model(device)
                torch.cuda.synchronize(device)
                stage_seconds["representative_model_initialization"] += time.perf_counter() - t0
                model_state_hashes.append(model_state_sha256)

                t0 = time.perf_counter()
                legacy_prediction = predict_delta(
                    model, data, validation_panel, prep, device, BATCH_SIZE)
                torch.cuda.synchronize(device)
                stage_seconds["legacy_prediction_including_unused_label_transfer"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                input_prediction = predict_delta_inputs_only(
                    model, data, validation_panel, prep, device, BATCH_SIZE)
                torch.cuda.synchronize(device)
                stage_seconds["input_only_prediction"] += time.perf_counter() - t0

                prediction_equal = np.array_equal(legacy_prediction, input_prediction)
                prediction_error = (float(np.max(np.abs(legacy_prediction - input_prediction)))
                                    if legacy_prediction.size else 0.0)
                max_prediction_error = max(max_prediction_error, prediction_error)
                if not prediction_equal:
                    failures.append({"fold": int(outer_fold["fold"]),
                                     "inner_fold": int(inner["inner_fold"]),
                                     "check": "legacy_vs_input_only_prediction"})
                else:
                    n_prediction_equal += 1

                t0 = time.perf_counter()
                legacy_score, legacy_cells = reversal_validation_metric(
                    data, validation_panel, legacy_prediction, genes, up, down)
                stage_seconds["legacy_validation_scoring"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                cached_score, cached_cells = cache.score(input_prediction)
                stage_seconds["cached_validation_scoring"] += time.perf_counter() - t0

                try:
                    _assert_frame_exact(cached_cells, legacy_cells)
                    frame_equal = True
                except AssertionError as exc:
                    frame_equal = False
                    failures.append({"fold": int(outer_fold["fold"]),
                                     "inner_fold": int(inner["inner_fold"]),
                                     "check": "per_cell_score_frame_exact",
                                     "detail": str(exc)[:500]})
                if frame_equal:
                    n_cell_frames_equal += 1
                score_equal = (legacy_score == cached_score
                               or (np.isnan(legacy_score) and np.isnan(cached_score)))
                if not score_equal:
                    failures.append({"fold": int(outer_fold["fold"]),
                                     "inner_fold": int(inner["inner_fold"]),
                                     "check": "equal_cell_validation_score_exact"})
                else:
                    n_score_equal += 1
                record.update({
                    "representative_model_rank": REPRESENTATIVE_RANK,
                    "representative_model_state_sha256": model_state_sha256,
                    "prediction_keys_exact": validation_keys.equals(
                        all_keys.iloc[validation_panel][["cell_id", "pert_id"]].reset_index(drop=True)),
                    "prediction_exact": prediction_equal,
                    "max_prediction_abs_error": prediction_error,
                    "per_cell_score_frames_exact": frame_equal,
                    "legacy_cell_score_vector": _score_rows(legacy_cells),
                    "cached_cell_score_vector": _score_rows(cached_cells),
                    "legacy_equal_cell_validation_score": (
                        float(legacy_score) if np.isfinite(legacy_score) else None),
                    "cached_equal_cell_validation_score": (
                        float(cached_score) if np.isfinite(cached_score) else None),
                    "equal_cell_validation_score_exact": score_equal,
                })
                if record["prediction_keys_exact"] is not True:
                    failures.append({"fold": int(outer_fold["fold"]),
                                     "inner_fold": int(inner["inner_fold"]),
                                     "check": "validation_prediction_key_order_exact"})
                del model
        torch.cuda.synchronize(device)
    finally:
        sampler.stop()
    assigned_gpu_wall = time.perf_counter() - gpu_assigned_start

    if total_inner != 15:
        failures.append({"check": "exactly_15_inner_folds", "observed": total_inner})
    if n_prediction_equal != total_inner:
        failures.append({"check": "all_representative_predictions_exact",
                         "observed": n_prediction_equal, "expected": total_inner})
    if n_cell_frames_equal != total_inner:
        failures.append({"check": "all_per_cell_score_frames_exact",
                         "observed": n_cell_frames_equal, "expected": total_inner})
    if n_score_equal != total_inner:
        failures.append({"check": "all_equal_cell_validation_scores_exact",
                         "observed": n_score_equal, "expected": total_inner})
    if n_group_payloads_equal != total_inner:
        failures.append({"check": "all_fixed_measured_group_payloads_exact",
                         "observed": n_group_payloads_equal, "expected": total_inner})
    if len(set(model_state_hashes)) != 1:
        failures.append({"check": "same_deterministic_frozen_model_each_inner_fold",
                         "distinct_state_hashes": sorted(set(model_state_hashes))})
    if any("sampling_error" in row for row in sampler.samples):
        failures.append({"check": "gpu_occupancy_sampling"})

    input_hashes = {_relative(path): _sha256(path) for path in _input_paths()}
    code_hashes = {_relative(path): _sha256(path) for path in _code_paths()}
    elapsed = time.perf_counter() - gate_start
    sampler_rows = [row for row in sampler.samples if "utilization_pct" in row]
    target_util = [row["utilization_pct"] for row in sampler_rows]
    target_memory = [row["memory_mib"] for row in sampler_rows]
    status = ("B5_VALIDATION_CACHE_NUMERICAL_EQUIVALENCE_PASS_REVIEW_REQUIRED"
              if not failures else "B5_VALIDATION_CACHE_EQUIVALENCE_GATE_FAIL")
    config = {
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "seed": SEEDS[0],
        "optimizer_steps": 0,
        "model_state": "same deterministic rank-16 seeded initialization; frozen; untrained",
        "representative_model_state_sha256": sorted(set(model_state_hashes)),
        "condition_dir": _relative(CONDITION_DIR),
        "input_dir": _relative(INPUT_DIR),
        "contract_dir": _relative(CONTRACT_DIR),
        "noise_cache_path": _relative(CACHE_PATH),
        "noise_cache_sha256_from_asset_manifest": cache_asset_manifest["cache_sha256"],
        "split_path": _relative(SPLIT_PATH),
        "signature_dir": _relative(SIGNATURE_DIR),
        "validation_audit_path": _relative(AUDIT_PATH),
        "prior_fold0_one_update_profile_path": _relative(PRIOR_FOLD0_PROFILE),
        "prior_fold0_one_update_profile_sha256": _sha256(PRIOR_FOLD0_PROFILE),
        "panel_rule": "fixed 24h, log10 dose 1.0, inner-train-seen drugs, >=5 unique drugs per eligible validation cell",
        "minimum_seen_drugs_per_cell": 5,
        "fixed_time_hours": 24.0,
        "fixed_log10_dose_um": 1.0,
    }
    config_sha256 = hashlib.sha256(json.dumps(
        config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    return {
        "experiment_id": "EXP-039",
        "status": status,
        "started_at_utc": started_utc,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "real_15_inner_fold_engineering_equivalence_gate_only",
        "all_equivalence_checks_passed": False,
        "all_numerical_equivalence_checks_passed": not failures,
        "formal_unlock_eligible": False,
        "n_inner_folds": total_inner,
        "optimizer_steps_performed": 0,
        "training_performed": False,
        "formal_fold_training_performed": False,
        "formal_candidate_selection_performed": False,
        "candidate_selection": "NOT_ASSESSED_NO_FROZEN_WEIGHTS",
        "outer_test_response_rows_read": False,
        "outer_test_condition_rows_indexed": 0,
        "outer_test_target_values_accessed": False,
        "outer_train_rows_read_guard": "delta, state, and exposure arrays raise if indexed outside each outer-training fold",
        "validation_aggregation_unit": "eligible cell_id groups; no patient-level observations occur in this EXP-039 panel",
        "cuda_device": {"index": device_index, "name": torch.cuda.get_device_name(device),
                        "memory_before_mib": gpu_before["memory_mib"],
                        "utilization_before_pct": gpu_before["utilization_pct"]},
        "gpu_monitor": {
            "sample_interval_seconds": 1.0,
            "n_samples": len(sampler_rows),
            "utilization_mean_pct": float(np.mean(target_util)) if target_util else None,
            "utilization_max_pct": max(target_util) if target_util else None,
            "memory_max_mib": max(target_memory) if target_memory else None,
            "samples": sampler.samples,
            "assigned_single_gpu_wall_seconds_upper_bound": assigned_gpu_wall,
            "assigned_single_gpu_hours_upper_bound": assigned_gpu_wall / 3600,
        },
        "gate_wall_seconds": elapsed,
        "gpu_assigned_wall_seconds": assigned_gpu_wall,
        "stage_seconds": stage_seconds,
        "equality_counts": {
            "predictions_exact": n_prediction_equal,
            "per_cell_score_frames_exact": n_cell_frames_equal,
            "validation_scores_exact": n_score_equal,
            "fixed_measured_group_payloads_exact": n_group_payloads_equal,
            "fixed_measured_cell_drug_groups_compared": n_group_rows_compared,
            "max_prediction_abs_error": max_prediction_error,
        },
        "candidate_selection_assessment": "NOT_ASSESSED_NO_FROZEN_WEIGHTS",
        "prior_fold0_one_update_evidence": {
            "path": _relative(PRIOR_FOLD0_PROFILE),
            "sha256": _sha256(PRIOR_FOLD0_PROFILE),
            "scope": "single real fold, one-update engineering microbenchmark; not formal candidate selection",
            "candidate_score_and_choice_equivalence": "see profile_result.equivalence; used only as limited prior numerical evidence",
        },
        "panel_records": panel_records,
        "failures": failures,
        "config": config,
        "config_sha256": config_sha256,
        "input_sha256": input_hashes,
        "declared_asset_sha256": {
            "condition_manifest_files": conditions_manifest["files_sha256"],
            "input_manifest_files": inputs_manifest["files_sha256"],
            "source_cache_sha256": conditions_manifest.get("source_cache_sha256"),
            "exact978_noise_cache_sha256": cache_asset_manifest["cache_sha256"],
            "contract_controls_sha256": _sha256(CONTRACT_DIR / "controls.parquet"),
        },
        "code_sha256": code_hashes,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        },
    }


def test_failed_equivalence_receipt_cannot_enable_cached_validation(tmp_path):
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps({
        "experiment_id": "EXP-039",
        "status": "B5_VALIDATION_CACHE_EQUIVALENCE_GATE_FAIL",
        "n_inner_folds": 15,
        "all_equivalence_checks_passed": False,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt is absent, incomplete, or failed"):
        verify_validation_cache_gate(receipt_path, code_files=[], input_files=[])


def test_real_15_inner_fold_cache_equivalence_gate():
    """Run all 15 current real inner panels and write a success/failure receipt."""
    if os.environ.get("EXP039_RUN_REAL_CACHE_GATE") != "1":
        pytest.skip("set EXP039_RUN_REAL_CACHE_GATE=1 to run the real-data 15-fold gate")
    if RECEIPT_PATH.exists():
        raise FileExistsError(f"cache-equivalence receipt already exists: {RECEIPT_PATH}")
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        receipt = _real_gate(started_utc)
    except Exception as exc:
        receipt = {
            "experiment_id": "EXP-039",
            "status": "B5_VALIDATION_CACHE_EQUIVALENCE_GATE_FAIL",
            "all_equivalence_checks_passed": False,
            "n_inner_folds": 0,
            "started_at_utc": started_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
            "formal_fold_training_performed": False,
            "outer_test_response_rows_read": False,
            "input_sha256": {
                _relative(path): _sha256(path) for path in _input_paths() if path.is_file()
            },
            "code_sha256": {
                _relative(path): _sha256(path) for path in _code_paths() if path.is_file()
            },
        }
        _write_receipt_once(receipt)
        raise

    _write_receipt_once(receipt)
    assert receipt["status"] == "B5_VALIDATION_CACHE_NUMERICAL_EQUIVALENCE_PASS_REVIEW_REQUIRED", receipt["failures"]
    assert receipt["all_numerical_equivalence_checks_passed"] is True
    assert receipt["formal_unlock_eligible"] is False
    assert receipt["candidate_selection"] == "NOT_ASSESSED_NO_FROZEN_WEIGHTS"
