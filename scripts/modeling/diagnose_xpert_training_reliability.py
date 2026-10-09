#!/usr/bin/env python3
"""Read back prior XPert TRAIN evidence and run the frozen EXP-069 CPU fit.

This script is intentionally separate from the shared EXP-051 runner.  The
readback phase indexes only TRAIN fields from old keyed archives and reads only
TRAIN source rows from the author H5AD.  The fit phase uses one deterministic
32-condition subset and the upstream XPert model, tensor adapter, and training
function.  It never opens inner-dev numeric values.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import importlib
import json
import math
import os
import random
import re
import signal
import subprocess
import sys
import time
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN = ROOT / "artifacts/experiments/EXP-051/20261006_exp051_contract_fit_smoke_v1"
SOURCE_KEYS = SOURCE_RUN / "response_blind_conditions.tsv"
EXP055 = ROOT / "artifacts/experiments/EXP-055/20261007_exp055_representative_models_v1/fit_attempt_009"
EXP063 = ROOT / "artifacts/experiments/EXP-063/coverage20_official_v1/fit_attempt_003"
DEFAULT_RUN_ID = "20261009_xpert_train_reliability_v1"
RUN_ROOT = ROOT / "artifacts/experiments/EXP-069/xpert"
SEED = 20261006
EXPECTED_XPERT_ROOT = Path("/mnt/d/Code/XPert")
AUTHOR_LOW = 0.0
AUTHOR_HIGH = 15.002
AUTHOR_BINS = 128
LOSS_WEIGHTS = [0.2, 0.003, 0.2, 1.0]
GENE_COUNT = 978
RECOVERY_SCHEMA_VERSION = 1
RECOVERY_STATE_SLOTS = ("XPert_training_state_a.pt", "XPert_training_state_b.pt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any], *, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if not overwrite and path.exists():
        tmp.unlink(missing_ok=True)
        raise FileExistsError(path)
    os.replace(tmp, path)


def atomic_torch_save(torch: Any, path: Path, payload: dict[str, Any]) -> None:
    """Write a complete torch payload through a same-directory atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def canonical_json_sha(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def output_root(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run_id may contain only letters, digits, dot, underscore, or hyphen")
    return RUN_ROOT / run_id


def read_frozen_train_keys() -> pd.DataFrame:
    if not SOURCE_KEYS.is_file():
        raise FileNotFoundError(SOURCE_KEYS)
    keys = pd.read_csv(SOURCE_KEYS, sep="\t", dtype=str, keep_default_na=False)
    required = {"split_role", "full_condition_key", "source_object_id", "full_inchikey",
                "pert_id", "source_row", "drug_node_idx", "pert_dose_num", "pert_time_num"}
    missing = sorted(required - set(keys.columns))
    if missing:
        raise ValueError(f"frozen condition table lacks required columns: {missing}")
    train = keys.loc[keys.split_role.eq("train")].copy().reset_index(drop=True)
    if len(train) != 419 or train.full_condition_key.nunique() != 419:
        raise ValueError("the frozen EXP-051 TRAIN condition-key contract changed")
    if train.full_inchikey.nunique() != 5 or train.source_object_id.nunique() != 110:
        raise ValueError("the frozen EXP-051 TRAIN object/drug contract changed")
    train["source_row"] = pd.to_numeric(train.source_row, errors="raise").astype(np.int64)
    train["drug_node_idx"] = pd.to_numeric(train.drug_node_idx, errors="raise").astype(np.int64)
    train["pert_dose_num"] = pd.to_numeric(train.pert_dose_num, errors="raise").astype(float)
    train["pert_time_num"] = pd.to_numeric(train.pert_time_num, errors="raise").astype(float)
    return train


def load_train_values_from_author_h5ad(keys: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Read requested source rows only; never materialize an inner-dev matrix."""
    if str(EXPECTED_XPERT_ROOT) not in sys.path:
        sys.path.insert(0, str(EXPECTED_XPERT_ROOT))
    import anndata as ad
    from scipy import sparse

    h5ad = EXPECTED_XPERT_ROOT / "processed_data/l1000_sdst_78453.h5ad"
    if not h5ad.is_file():
        raise FileNotFoundError(h5ad)
    rows = keys.source_row.to_numpy(dtype=np.int64)
    order = np.argsort(rows, kind="mergesort")
    sorted_rows = rows[order]
    if len(np.unique(sorted_rows)) != len(rows):
        raise ValueError("TRAIN source_row values are not unique")
    adata = ad.read_h5ad(h5ad, backed="r")
    try:
        if tuple(map(int, adata.shape)) != (78453, GENE_COUNT) or "X_ctl" not in adata.obsm:
            raise ValueError("author H5AD shape or paired-control slot changed")
        genes = list(map(str, adata.var_names))
        if len(genes) != GENE_COUNT or len(set(genes)) != GENE_COUNT:
            raise ValueError("author H5AD gene axis is not 978 unique names")
        selected_obs = adata.obs.iloc[sorted_rows][["pert_id", "pert_idx"]].copy()
        if not np.array_equal(selected_obs.pert_id.astype(str).to_numpy(),
                              keys.pert_id.astype(str).to_numpy()[order]):
            raise ValueError("TRAIN condition identities do not bind to author H5AD source rows")
        source_nodes = pd.to_numeric(selected_obs.pert_idx, errors="raise").to_numpy(dtype=np.int64)
        expected_nodes = keys.drug_node_idx.to_numpy(dtype=np.int64)[order]
        if not np.array_equal(source_nodes, expected_nodes):
            raise ValueError("TRAIN pert_idx does not bind to the frozen official feature row axis")
        x_sorted = adata.X[sorted_rows, :]
        ctl_sorted = adata.obsm["X_ctl"][sorted_rows, :]
        if hasattr(x_sorted, "to_memory"):
            x_sorted = x_sorted.to_memory()
        if hasattr(ctl_sorted, "to_memory"):
            ctl_sorted = ctl_sorted.to_memory()
        if sparse.issparse(x_sorted):
            x_sorted = x_sorted.toarray()
        if sparse.issparse(ctl_sorted):
            ctl_sorted = ctl_sorted.toarray()
    finally:
        adata.file.close()
    x = np.empty((len(rows), GENE_COUNT), dtype=np.float32)
    ctl = np.empty_like(x)
    x[order] = np.asarray(x_sorted, dtype=np.float32)
    ctl[order] = np.asarray(ctl_sorted, dtype=np.float32)
    delta = x - ctl
    if not (np.isfinite(x).all() and np.isfinite(ctl).all() and np.isfinite(delta).all()):
        raise FloatingPointError("non-finite value in authorized TRAIN X, X_ctl, or delta")
    source_meta = {
        "path": str(h5ad), "sha256": sha256(h5ad), "h5ad_shape": [78453, GENE_COUNT],
        "selected_role": "TRAIN only", "selected_rows": int(len(rows)),
        "selected_source_rows_sha256": canonical_sha("\n".join(map(str, rows)) + "\n"),
        "gene_axis_sha256": canonical_sha("\n".join(genes) + "\n"),
        "source_units": "source-provided replicate-collapsed z-score scale per EXP-051 frozen value contract; no rescaling applied",
    }
    return x, ctl, delta, genes, source_meta


def metrics(keys: pd.DataFrame, truth: np.ndarray, pred: np.ndarray,
            full_train_variance: np.ndarray) -> dict[str, Any]:
    if truth.shape != pred.shape or truth.ndim != 2 or truth.shape[1] != GENE_COUNT:
        raise ValueError("truth/prediction do not share the frozen 978-gene axis")
    use = np.isfinite(full_train_variance) & (full_train_variance > 1e-12)
    if not use.any():
        raise ValueError("no positive gene variance from full authorized TRAIN")
    yt = truth.astype(np.float64, copy=False)
    yp = pred.astype(np.float64, copy=False)
    per_row = np.mean((yp[:, use] - yt[:, use]) ** 2 / full_train_variance[use][None, :], axis=1)
    per_mae = np.mean(np.abs(yp - yt), axis=1)
    correlations: list[float | None] = []
    directions: list[float | None] = []
    for y, p in zip(yt, yp):
        correlations.append(float(np.corrcoef(y, p)[0, 1]) if np.ptp(y) > 0 and np.ptp(p) > 0 else None)
        nz = np.abs(y) > 1e-12
        directions.append(float(np.mean(np.sign(y[nz]) == np.sign(p[nz]))) if nz.any() else None)
    frame = keys[["source_object_id", "full_inchikey"]].reset_index(drop=True).copy()
    if len(frame) != len(truth):
        raise ValueError("metric keys do not align with target rows")
    frame["nmse"] = per_row
    frame["mae"] = per_mae
    frame["correlation"] = correlations
    frame["direction"] = directions
    object_metrics = frame.groupby("source_object_id", sort=True, as_index=False).agg(
        nmse=("nmse", "mean"), mae=("mae", "mean"),
        correlation=("correlation", "mean"), direction=("direction", "mean"))
    amp: list[float] = []
    for _, indexes in frame.groupby("source_object_id", sort=True).indices.items():
        ix = np.asarray(indexes, dtype=np.int64)
        observed = float(np.mean(np.abs(yt[ix])))
        if observed > 0:
            amp.append(float(np.mean(np.abs(yp[ix])) / observed))
    corr = object_metrics.correlation.dropna()
    direction = object_metrics.direction.dropna()
    return {
        "n_conditions": int(len(frame)), "n_source_objects": int(frame.source_object_id.nunique()),
        "n_full_inchikeys": int(frame.full_inchikey.nunique()),
        "n_genes_in_nmse_denominator": int(use.sum()), "n_zero_variance_genes_excluded": int((~use).sum()),
        "object_macro_nmse": float(object_metrics.nmse.mean()),
        "object_macro_mae": float(object_metrics.mae.mean()),
        "object_macro_gene_correlation": float(corr.mean()) if len(corr) else None,
        "object_macro_direction_agreement": float(direction.mean()) if len(direction) else None,
        "object_macro_amplitude_ratio": float(np.mean(amp)) if amp else None,
        "n_amplitude_objects_with_nonzero_observed_mean_abs": int(len(amp)),
    }


def baseline_predictions(keys: pd.DataFrame, truth: np.ndarray) -> dict[str, np.ndarray]:
    if len(keys) != len(truth):
        raise ValueError("baseline TRAIN keys and labels differ in length")
    zero = np.zeros_like(truth, dtype=np.float32)
    global_mean = np.broadcast_to(np.mean(truth, axis=0, keepdims=True), truth.shape).astype(np.float32).copy()
    per_drug = np.empty_like(truth, dtype=np.float32)
    for drug in sorted(keys.full_inchikey.astype(str).unique()):
        ix = np.flatnonzero(keys.full_inchikey.astype(str).to_numpy() == drug)
        per_drug[ix] = np.mean(truth[ix], axis=0, keepdims=True)
    return {"zero_change": zero, "global_mean_delta": global_mean, "per_drug_mean_delta": per_drug}


def _capture_json_value(line: str, field: str) -> Any:
    """Decode one top-level JSON field without parsing neighboring DEV metrics."""
    marker = f'"{field}"'
    start = line.find(marker)
    if start < 0:
        raise KeyError(field)
    colon = line.find(":", start + len(marker))
    value_start = colon + 1
    while value_start < len(line) and line[value_start].isspace():
        value_start += 1
    first = line[value_start]
    if first in "[{":
        opener, closer = first, "]" if first == "[" else "}"
        depth, in_string, escaped = 0, False, False
        for i in range(value_start, len(line)):
            c = line[i]
            if in_string:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_string = False
            else:
                if c == '"':
                    in_string = True
                elif c == opener:
                    depth += 1
                elif c == closer:
                    depth -= 1
                    if depth == 0:
                        return json.loads(line[value_start:i + 1])
        raise ValueError(f"unterminated JSON value for {field}")
    if first == '"':
        end = value_start + 1
        escaped = False
        while end < len(line):
            if line[end] == '"' and not escaped:
                return json.loads(line[value_start:end + 1])
            escaped = line[end] == "\\" and not escaped
            if line[end] != "\\":
                escaped = False
            end += 1
        raise ValueError(f"unterminated string value for {field}")
    end = value_start
    while end < len(line) and line[end] not in ",}":
        end += 1
    return json.loads(line[value_start:end].strip())


def read_train_curve(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for raw in stream:
            line = raw.strip()
            if not line:
                continue
            rows.append({
                "epoch": _capture_json_value(line, "epoch"),
                "schedule_stage": _capture_json_value(line, "schedule_stage"),
                "learning_rate": _capture_json_value(line, "learning_rate"),
                "official_train_loss_terms": _capture_json_value(line, "official_train_loss_terms"),
                "gradient_norm_mean": _capture_json_value(line, "gradient_norm_mean"),
                "gradient_norm_max": _capture_json_value(line, "gradient_norm_max"),
                "train_metrics": _capture_json_value(line, "train_metrics"),
            })
    if not rows:
        raise ValueError(f"empty training curve: {path}")
    return rows


def input_scale_summary(ctl: np.ndarray, train_x: np.ndarray) -> dict[str, Any]:
    bins = np.quantile(np.asarray([AUTHOR_LOW, AUTHOR_HIGH]), np.linspace(0.0, 1.0, AUTHOR_BINS - 1))
    digitized = np.digitize(ctl.astype(np.float32, copy=False), bins, right=False)
    total = int(ctl.size)
    low = int(np.count_nonzero(ctl < AUTHOR_LOW))
    high = int(np.count_nonzero(ctl > AUTHOR_HIGH))
    return {
        "author_digitized_model_input": "paired control X_ctl only; trt expression is returned as a loss target and is not used to form the cell embedding",
        "source_expression_scale": "frozen source-provided z-score scale; no rescaling applied",
        "official_fixed_bin_range": [AUTHOR_LOW, AUTHOR_HIGH], "official_n_bins": AUTHOR_BINS,
        "train_X_ctl": {"min": float(np.min(ctl)), "max": float(np.max(ctl)),
                        "mean": float(np.mean(ctl)), "sd": float(np.std(ctl)),
                        "fraction_below_bin_range": low / total,
                        "fraction_above_bin_range": high / total,
                        "count_below_bin_range": low, "count_above_bin_range": high,
                        "unique_digitized_ids": int(np.unique(digitized).size),
                        "digitized_id_min": int(digitized.min()), "digitized_id_max": int(digitized.max())},
        "train_X_target_only_distribution": {"min": float(np.min(train_x)), "max": float(np.max(train_x)),
                                             "mean": float(np.mean(train_x)), "sd": float(np.std(train_x))},
        "interpretation": "Out-of-range fractions quantify how much control input is assigned to edge digitization bins; they do not by themselves prove a model defect.",
    }


def verify_feature_axes(keys: pd.DataFrame, *, xpert_root: Path) -> dict[str, Any]:
    unimol_path = xpert_root / "processed_data/all_drugs_unimol_arr.npy"
    hg_path = xpert_root / "HG_data/saved_embedding/HG_drug_embeddings.npy"
    ppi_path = xpert_root / "processed_data/PPI_gene_vector_128d.npy"
    unimol = np.load(unimol_path, mmap_mode="r", allow_pickle=False)
    hg = np.load(hg_path, mmap_mode="r", allow_pickle=True)
    ppi = np.load(ppi_path, mmap_mode="r", allow_pickle=False)
    nodes = keys.drug_node_idx.to_numpy(dtype=np.int64)
    if unimol.ndim != 3 or unimol.shape[1:] != (122, 514):
        raise ValueError(f"official UniMol atom axis changed: {unimol.shape}")
    if hg.shape[0] <= int(nodes.max()) or ppi.shape[0] != GENE_COUNT:
        raise ValueError("official HG/PPi feature axes do not cover the frozen TRAIN keys")
    selected_unimol = np.asarray(unimol[nodes], dtype=np.float32)
    selected_hg = np.asarray(hg[nodes])
    if not np.isfinite(selected_unimol).all() or not np.isfinite(selected_hg).all() or not np.isfinite(ppi).all():
        raise FloatingPointError("non-finite value in official XPert feature rows")
    return {
        "drug_feature": "official UniMol atomic features plus official HG drug embedding",
        "feature_node_axis": "frozen pert_idx -> source H5AD pert_idx -> same row in official UniMol and HG arrays",
        "n_unique_train_drug_nodes": int(len(np.unique(nodes))),
        "selected_drug_nodes": sorted(map(int, np.unique(nodes))),
        "unimol_shape": list(unimol.shape), "unimol_selected_shape": list(selected_unimol.shape),
        "hg_shape": list(hg.shape), "hg_selected_shape": list(selected_hg.shape),
        "ppi_shape": list(ppi.shape), "all_selected_feature_values_finite": True,
        "files_sha256": {"unimol": sha256(unimol_path), "hg": sha256(hg_path), "ppi": sha256(ppi_path)},
    }


def _read_npz_train(path: Path, prediction_field: str,
                    additional_train_fields: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    """Read only separately stored TRAIN fields; no inner-dev NPZ member is indexed."""
    with np.load(path, allow_pickle=False) as saved:
        required = ["train_condition_key", "train_source_object_id", "train_full_inchikey", "genes",
                    "train_target_delta", prediction_field, *additional_train_fields]
        missing = [name for name in required if name not in saved.files]
        if missing:
            raise KeyError(f"TRAIN-only prediction archive lacks fields {missing}: {path}")
        return {name: saved[name].copy() for name in required}


def _safe_checkpoint_summary(path: Path) -> dict[str, Any]:
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload.get("model_state_dict")) if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not state:
        raise ValueError(f"checkpoint has no model state: {path}")
    tensors = [value for value in state.values() if isinstance(value, torch.Tensor)]
    nonfinite_names = [name for name, value in state.items()
                       if isinstance(value, torch.Tensor) and value.is_floating_point()
                       and not bool(torch.isfinite(value).all())]
    if nonfinite_names:
        raise FloatingPointError(f"non-finite checkpoint state: {path}: {nonfinite_names[:3]}")
    return {"path": str(path), "sha256": sha256(path), "weights_only_load_pass": True,
            "tensor_count": len(tensors), "parameter_elements": int(sum(t.numel() for t in tensors)),
            "nonfinite_float_tensors": len(nonfinite_names),
            "saved_epoch_zero_based": payload.get("epoch"),
            "saved_epoch_one_based": payload.get("epoch_one_based", payload.get("epoch"))}


def old_assets_readback(run_id: str) -> dict[str, Any]:
    out = output_root(run_id)
    out.mkdir(parents=True, exist_ok=False)
    keys = read_frozen_train_keys()
    subset = select_train32(keys)
    shared_keys_path = ROOT / "artifacts/experiments/EXP-069/transigen/20261009_transigen_train_reliability_v1/shared_train32_keys.tsv"
    if not shared_keys_path.is_file():
        raise FileNotFoundError("the shared EXP-069 TRAIN32 key file from the TranSiGen owner is missing")
    shared_keys = pd.read_csv(shared_keys_path, sep="\t", dtype=str, keep_default_na=False)
    shared_columns = ["full_condition_key", "source_object_id", "full_inchikey", "source_row"]
    for column in ("source_row",):
        if column in subset:
            subset[column] = subset[column].astype(str)
    if not all(column in shared_keys for column in shared_columns):
        raise ValueError("the shared TRAIN32 key file lacks identity columns")
    if not np.array_equal(subset[shared_columns].astype(str).to_numpy(),
                          shared_keys[shared_columns].astype(str).to_numpy()):
        raise ValueError("XPert TRAIN32 selection differs from the shared TranSiGen-owner TRAIN32 key list")
    subset["selection_sha256"] = subset.full_condition_key.astype(str).map(
        lambda key: hashlib.sha256(f"{SEED}|{key}".encode("utf-8")).hexdigest())
    subset.to_csv(out / "train32_keys.tsv", sep="\t", index=False, lineterminator="\n")
    x, ctl, truth, genes, source_meta = load_train_values_from_author_h5ad(keys)
    gene_variance = np.var(truth.astype(np.float64), axis=0, ddof=0)
    old055 = _read_npz_train(EXP055 / "full_keyed_predictions.npz", "XPert_best_inner_dev_train_delta",
                             ("XPert_final_stop_train_delta",))
    if not np.array_equal(old055["train_condition_key"].astype(str), keys.full_condition_key.astype(str).to_numpy()):
        raise ValueError("EXP-055 old TRAIN prediction keys differ from the frozen TRAIN table")
    if not np.array_equal(old055["train_source_object_id"].astype(str), keys.source_object_id.astype(str).to_numpy()):
        raise ValueError("EXP-055 old TRAIN object identities differ from the frozen TRAIN table")
    if not np.array_equal(old055["train_full_inchikey"].astype(str), keys.full_inchikey.astype(str).to_numpy()):
        raise ValueError("EXP-055 old TRAIN compound identities differ from the frozen TRAIN table")
    if not np.array_equal(old055["genes"].astype(str), np.asarray(genes, dtype=str)):
        raise ValueError("EXP-055 old TRAIN gene axis differs from the author H5AD axis")
    target_delta_diff = float(np.max(np.abs(old055["train_target_delta"].astype(np.float64) - truth.astype(np.float64))))
    if target_delta_diff > 2e-6:
        raise ValueError(f"EXP-055 saved TRAIN target differs from source H5AD TRAIN rows: {target_delta_diff}")
    best = old055["XPert_best_inner_dev_train_delta"].astype(np.float32)
    final = old055["XPert_final_stop_train_delta"].astype(np.float32)
    old_baselines = baseline_predictions(keys, truth)
    best_metrics = metrics(keys, truth, best, gene_variance)
    final_metrics = metrics(keys, truth, final, gene_variance)
    baseline_metrics = {name: metrics(keys, truth, pred, gene_variance)
                        for name, pred in old_baselines.items()}
    best_base = min(baseline_metrics["zero_change"]["object_macro_nmse"],
                    baseline_metrics["per_drug_mean_delta"]["object_macro_nmse"])
    train_curve = read_train_curve(EXP055 / "XPert_training_curve.jsonl")
    if len(train_curve) != 189 or [int(row["epoch"]) for row in train_curve] != list(range(1, 190)):
        raise ValueError("EXP-055 immutable curve is not the expected contiguous 189-epoch record")
    curve_rows = []
    for row in train_curve:
        losses = row["official_train_loss_terms"]
        if not isinstance(losses, list) or len(losses) != 5:
            raise ValueError("XPert source-reported training loss tuple changed")
        m = row["train_metrics"]
        curve_rows.append({
            "epoch": int(row["epoch"]), "stage": str(row["schedule_stage"]),
            "learning_rate": float(row["learning_rate"]),
            "author_loss_total_per_source_sample": losses[0],
            "author_treatment_head_rmse": losses[1], "author_control_head_rmse": losses[2],
            "author_delta_head_rmse": losses[3], "author_delta_pearson_loss": losses[4],
            "train_object_macro_nmse": m["object_macro_nmse"],
            "train_amplitude_ratio": m["object_macro_amplitude_ratio"],
            "train_object_macro_mae": m["object_macro_mae"],
            "train_direction_agreement": m["object_macro_direction_agreement"],
            "train_gene_correlation": m["object_macro_gene_correlation"],
            "gradient_norm_mean": row["gradient_norm_mean"], "gradient_norm_max": row["gradient_norm_max"],
        })
    pd.DataFrame(curve_rows).to_csv(out / "exp055_train_curve_readback.tsv", sep="\t", index=False,
                                    lineterminator="\n")
    minimum_curve = min(curve_rows, key=lambda row: float(row["train_object_macro_nmse"]))
    exp055_checkpoint = {
        "selected_by_official_rule": _safe_checkpoint_summary(EXP055 / "XPert_best_inner_dev.pt"),
        "last_stop": _safe_checkpoint_summary(EXP055 / "XPert_final_stop_epoch_189.pt"),
        "official_root_readback_sha256": sha256(EXP055 / "root_independent_attempt009_readback.json"),
    }

    old063_pred_path = EXP063 / "epoch001_keyed_predictions.npz"
    old063 = _read_npz_train(old063_pred_path, "XPert_epoch001_train_delta")
    if len(old063["train_target_delta"]) != 1291 or old063["train_target_delta"].shape[1] != GENE_COUNT:
        raise ValueError("EXP-063 TRAIN epoch-1 archive has changed shape")
    if (len(np.unique(old063["train_condition_key"].astype(str))) != 1291
            or len(np.unique(old063["train_source_object_id"].astype(str))) != 93
            or len(np.unique(old063["train_full_inchikey"].astype(str))) != 20):
        raise ValueError("EXP-063 TRAIN epoch-1 archive violates its frozen key/object/drug counts")
    t063 = old063["train_target_delta"].astype(np.float32)
    p063 = old063["XPert_epoch001_train_delta"].astype(np.float32)
    v063 = np.var(t063.astype(np.float64), axis=0, ddof=0)
    k063 = pd.DataFrame({"source_object_id": old063["train_source_object_id"].astype(str),
                         "full_inchikey": old063["train_full_inchikey"].astype(str)})
    m063_pred = metrics(k063, t063, p063, v063)
    b063 = baseline_predictions(k063, t063)
    m063_base = {name: metrics(k063, t063, pred, v063) for name, pred in b063.items()}
    exp063_checkpoint = {
        "epoch_one": _safe_checkpoint_summary(EXP063 / "XPert_epoch001_checkpoint.pt"),
        "initialization": _safe_checkpoint_summary(EXP063 / "initial_checkpoint.pt"),
        "root_readback_sha256": sha256(EXP063 / "root_independent_cost_gate_readback.json"),
    }
    source_scale = input_scale_summary(ctl, x)
    feature_axes = verify_feature_axes(keys, xpert_root=EXPECTED_XPERT_ROOT)
    selected_epoch = 139
    fit_gate_threshold = 0.10
    selected_reduction = 1.0 - best_metrics["object_macro_nmse"] / best_base
    maximum_train_reduction = 1.0 - float(minimum_curve["train_object_macro_nmse"]) / best_base
    report: dict[str, Any] = {
        "status": "OLD_TRAIN_READBACK_COMPLETE",
        "run_id": run_id, "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": {"new_inner_dev_numeric_values_read": False, "outer_locked_pdo_values_read": False,
                  "EXP063_train_only_fields_indexed": True, "GPU_hours": 0.0},
        "files": {
            "script": {"path": "scripts/modeling/diagnose_xpert_training_reliability.py",
                       "sha256": sha256(Path(__file__).resolve())},
            "frozen_train_keys": {"path": str(SOURCE_KEYS.relative_to(ROOT)), "sha256": sha256(SOURCE_KEYS)},
            "shared_train32_keys": {"path": str(shared_keys_path.relative_to(ROOT)),
                                    "sha256": sha256(shared_keys_path),
                                    "xp_train32_selection_identical": True},
            "frozen_train32_keys": {"path": str((out / "train32_keys.tsv").relative_to(ROOT)),
                                    "sha256": sha256(out / "train32_keys.tsv")},
            "EXP055_archive": {"path": str((EXP055 / "full_keyed_predictions.npz").relative_to(ROOT)),
                               "sha256": sha256(EXP055 / "full_keyed_predictions.npz")},
            "EXP055_curve": {"path": str((EXP055 / "XPert_training_curve.jsonl").relative_to(ROOT)),
                             "sha256": sha256(EXP055 / "XPert_training_curve.jsonl")},
            "EXP063_archive": {"path": str(old063_pred_path.relative_to(ROOT)), "sha256": sha256(old063_pred_path)},
            "EXP051_frozen_source_receipt": {"path": str((SOURCE_RUN / "value_contract_receipt.json").relative_to(ROOT)),
                                             "sha256": sha256(SOURCE_RUN / "value_contract_receipt.json")},
        },
        "data_binding": {"EXP055": {**source_meta, "conditions": int(len(keys)),
                                     "objects": int(keys.source_object_id.nunique()),
                                     "full_inchikeys": int(keys.full_inchikey.nunique()),
                                     "genes": int(len(genes)),
                                     "max_abs_saved_target_delta_difference": target_delta_diff,
                                     "full_condition_keys_unique": True},
                          "EXP063": {"train_conditions": int(len(t063)), "train_objects": 93,
                                     "full_inchikeys": 20, "genes": GENE_COUNT,
                                     "prediction_field": "XPert_epoch001_train_delta"}},
        "official_feature_axis": feature_axes,
        "input_scale": source_scale,
        "output_and_loss_semantics": {
            "author_outputs": ["trt_output", "ctl_output", "deg_output"],
            "project_delta_output": "deg_output directly predicts X-X_ctl; author inference reconstructs treated expression as ctl_raw_data + deg_output. It is not trt_output minus ctl_output.",
            "author_training_targets": {"trt_output": "X", "ctl_output": "X_ctl",
                                        "deg_output": "X-X_ctl"},
            "author_four_objectives": ["treated expression MSE", "control expression MSE",
                                        "delta MSE", "delta Pearson loss"],
            "loss_weights": LOSS_WEIGHTS,
            "schedule": {"batch_size": 128, "learning_rate": 0.004, "weight_decay": 1e-5,
                         "epochs_zero_to_69": "per-sample root-MSE warmup objective",
                         "epoch_40_onward_learning_rate_factor": 0.5,
                         "epoch_70_onward": "raw weighted sum objective",
                         "full_run_checkpoint_rule": "minimum inner-dev delta Pearson loss, patience 50; not reused in EXP-069 fixed endpoint"},
            "inference_control_input": "X_ctl digitized by the upstream fixed 0..15.002 range; treated raw inputs/bins are zeroed for model inference, preventing direct use of the treated target as an input.",
        },
        "EXP055_old_train": {
            "completed_epochs": len(curve_rows), "official_selected_epoch": selected_epoch,
            "selected_checkpoint": exp055_checkpoint["selected_by_official_rule"],
            "selected_train_metrics_recomputed": best_metrics,
            "last_stop_epoch": 189, "last_stop_checkpoint": exp055_checkpoint["last_stop"],
            "last_stop_train_metrics_recomputed": final_metrics,
            "train_baselines_recomputed": baseline_metrics,
            "registered_fit_gate": {"baseline_nmse": best_base,
                                    "selected_epoch_relative_reduction": selected_reduction,
                                    "minimum_reduction_required": fit_gate_threshold,
                                    "selected_epoch_passes": bool(selected_reduction >= fit_gate_threshold
                                                                   and 0.5 <= best_metrics["object_macro_amplitude_ratio"] <= 2.0),
                                    "best_train_only_epoch": minimum_curve,
                                    "best_train_only_relative_reduction": maximum_train_reduction,
                                    "best_train_only_epoch_met_10_percent_gate": bool(maximum_train_reduction >= fit_gate_threshold),
                                    "best_train_only_epoch_changes_no_selection": True},
            "checkpoint_hashes": exp055_checkpoint,
            "curve_train_only": {"rows": len(curve_rows), "first_epoch": curve_rows[0],
                                 "epoch_70": curve_rows[69], "epoch_100": curve_rows[99],
                                 "epoch_139": curve_rows[138], "epoch_189": curve_rows[-1],
                                 "minimum_train_nmse": minimum_curve},
        },
        "EXP063_epoch001_train_only": {"checkpoint_hashes": exp063_checkpoint,
                                       "train_metrics_recomputed": m063_pred,
                                       "train_baselines_recomputed": m063_base,
                                       "interpretation": "one-epoch 20-drug cost-gate observation; not a full fit and not directly comparable to EXP-055 because its TRAIN domain differs"},
        "interpretation": {
            "fact": "EXP-055 selected TRAIN NMSE and its TRAIN-only best-epoch NMSE both fail the registered 10% improvement gate; the selected output magnitude meets the lower amplitude boundary by a small margin.",
            "inference": "The trajectory and finite checkpoint indicate the 5-drug model learned some TRAIN signal, but the evidence does not isolate whether limited fit is caused by data, optimization, or objective/input scaling.",
            "unknown": "The fixed 0..15.002 input bins are inherited from the author source contract; the observed TRAIN edge-bin rate is reported here, while no alternative scale is tested.",
        },
    }
    atomic_json(out / "old_train_readback.json", report, overwrite=False)
    return report


def select_train32(keys: pd.DataFrame) -> pd.DataFrame:
    drugs = sorted(keys.full_inchikey.astype(str).unique())
    if len(drugs) != 5:
        raise ValueError("EXP-069 common 32-key protocol requires the fixed five-drug TRAIN domain")
    selected = []
    for rank, drug in enumerate(drugs):
        take = 7 if rank < 2 else 6
        rows = keys.loc[keys.full_inchikey.astype(str).eq(drug)].copy()
        rows["_rank_hash"] = rows.full_condition_key.astype(str).map(
            lambda key: hashlib.sha256(f"{SEED}|{key}".encode("utf-8")).hexdigest())
        rows = rows.sort_values(["_rank_hash", "full_condition_key"], kind="mergesort").head(take)
        if len(rows) != take:
            raise ValueError(f"not enough TRAIN conditions for drug {drug}: {len(rows)} < {take}")
        selected.append(rows)
    result = pd.concat(selected, ignore_index=True).drop(columns="_rank_hash")
    result = result.sort_values("full_condition_key", kind="mergesort").reset_index(drop=True)
    if len(result) != 32 or result.full_condition_key.nunique() != 32:
        raise ValueError("the frozen EXP-069 TRAIN subset does not contain 32 unique conditions")
    return result


def write_old_prediction_summary(run_id: str) -> dict[str, Any]:
    out = output_root(run_id)
    source_path = out / "old_train_readback.json"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    report = json.loads(source_path.read_text(encoding="utf-8"))
    old = report["EXP055_old_train"]
    selected = old["selected_train_metrics_recomputed"]
    stop = old["last_stop_train_metrics_recomputed"]
    gate = old["registered_fit_gate"]
    best = gate["best_train_only_epoch"]
    baseline = old["train_baselines_recomputed"]
    strongest_baseline_name, strongest_baseline_metrics = min(
        ((name, baseline[name]) for name in ("zero_change", "per_drug_mean_delta")),
        key=lambda item: item[1]["object_macro_nmse"])
    exp063 = report["EXP063_epoch001_train_only"]
    selected_nmse = float(selected["object_macro_nmse"])
    strongest_fit_baseline = float(strongest_baseline_metrics["object_macro_nmse"])
    min_reduction = float(gate["minimum_reduction_required"])
    selected_amp = float(selected["object_macro_amplitude_ratio"])
    amp_margin = selected_amp - 0.5
    summary = {
        "status": "OLD_TRAIN_METRIC_SUMMARY_COMPLETE",
        "run_id": run_id,
        "summary_script_sha256": sha256(Path(__file__).resolve()),
        "summary_command": f"python scripts/modeling/diagnose_xpert_training_reliability.py --phase old-summary --run-id {run_id}",
        "source_readback": {"path": str(source_path.relative_to(ROOT)), "sha256": sha256(source_path)},
        "files": {"train32_key_list": {"path": "train32_keys.tsv", "sha256": sha256(out / "train32_keys.tsv")},
                  "curve": {"path": "exp055_train_curve_readback.tsv", "sha256": sha256(out / "exp055_train_curve_readback.tsv")}},
        "EXP055_author_selected_checkpoint_train_only": {
            "checkpoint_epoch": old["official_selected_epoch"],
            "checkpoint_sha256": old["selected_checkpoint"]["sha256"],
            "conditions_objects_drugs_genes": [419, 110, 5, GENE_COUNT],
            "metrics": selected,
            "strongest_registered_baseline": {"name": strongest_baseline_name,
                                              "object_macro_nmse": strongest_fit_baseline},
            "absolute_nmse_difference_vs_baseline": selected_nmse - strongest_fit_baseline,
            "relative_nmse_reduction_percent": 100.0 * float(gate["selected_epoch_relative_reduction"]),
            "minimum_required_reduction_percent": 100.0 * min_reduction,
            "nmse_gate_margin": selected_nmse - strongest_fit_baseline * (1.0 - min_reduction),
            "amplitude_gate_margin_above_0_5": amp_margin,
            "fit_gate_passed": bool(gate["selected_epoch_passes"]),
            "baseline_metrics": baseline,
        },
        "EXP055_best_train_only_epoch_diagnostic": {
            "epoch": best["epoch"], "metrics": {
                "train_object_macro_nmse": best["train_object_macro_nmse"],
                "train_object_macro_amplitude_ratio": best["train_amplitude_ratio"],
                "train_object_macro_mae": best["train_object_macro_mae"],
                "train_direction_agreement": best["train_direction_agreement"],
                "train_gene_correlation": best["train_gene_correlation"]},
            "relative_reduction_percent": 100.0 * float(gate["best_train_only_relative_reduction"]),
            "fit_gate_passed": bool(gate["best_train_only_epoch_met_10_percent_gate"]),
            "official_checkpoint_unchanged": True,
        },
        "EXP055_last_stop_checkpoint_train_only": {
            "epoch": old["last_stop_epoch"], "checkpoint_sha256": old["last_stop_checkpoint"]["sha256"],
            "metrics": stop,
        },
        "EXP063_one_epoch_train_only_cost_gate": {
            "train_conditions_objects_drugs_genes": [1291, 93, 20, GENE_COUNT],
            "metrics": exp063["train_metrics_recomputed"],
            "baselines": exp063["train_baselines_recomputed"],
            "checkpoint_sha256": exp063["checkpoint_hashes"]["epoch_one"]["sha256"],
            "scope": "epoch-1 cost measurement only; distinct 20-drug domain, not compared directly with EXP-055",
        },
        "input_scale": report["input_scale"],
        "author_output_and_objective": report["output_and_loss_semantics"],
        "feature_axis": report["official_feature_axis"],
        "evidence_boundary": {"new_inner_dev_numeric_values_read": False,
                              "outer_locked_pdo_numeric_values_read": False,
                              "GPU_hours": 0.0},
        "conclusion": {
            "fact": f"The officially selected EXP-055 checkpoint lowered object-macro TRAIN NMSE by {100 * gate['selected_epoch_relative_reduction']:.2f}% versus the per-drug mean, below the preregistered 10% fit gate. Its amplitude ratio is {selected_amp:.3f}, only {amp_margin:.3f} above the lower allowed bound.",
            "fact_best_train_only": f"The best TRAIN-only curve epoch {best['epoch']} lowered NMSE by {100 * gate['best_train_only_relative_reduction']:.2f}%, still below 10%; this epoch was not used to change official checkpoint selection.",
            "inference": "Finite checkpoints, predictions, and a gradual TRAIN metric improvement support that this run learned some source TRAIN signal. The existing evidence does not identify one cause for the remaining fit gap.",
            "unknown": "The fixed author input-bin range and the measured out-of-range frequency are diagnostic evidence only; no alternative input scaling has been tested.",
        },
    }
    atomic_json(out / "old_prediction_metrics.json", summary, overwrite=False)
    return summary


def guarded_optimizer(torch: Any, optimizer: Any, model: Any, guard_state: dict[str, Any]) -> Any:
    class Guard:
        def __init__(self) -> None:
            self.param_groups = optimizer.param_groups
            self.state = optimizer.state
            self._optimizer = optimizer

        def step(self) -> Any:
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    guard_state.update({"first_nonfinite_component": "gradient", "first_nonfinite_tensor": name})
                    raise FloatingPointError(f"non-finite gradient tensor: {name}")
            result = optimizer.step()
            for name, parameter in model.named_parameters():
                if not bool(torch.isfinite(parameter).all()):
                    guard_state.update({"first_nonfinite_component": "parameter", "first_nonfinite_tensor": name})
                    raise FloatingPointError(f"non-finite model parameter after optimizer update: {name}")
            for name, state in optimizer.state.items():
                for state_name, value in state.items():
                    if isinstance(value, torch.Tensor) and value.is_floating_point() and not bool(torch.isfinite(value).all()):
                        guard_state.update({"first_nonfinite_component": "optimizer_state",
                                            "first_nonfinite_tensor": str(name), "state_name": state_name})
                        raise FloatingPointError(f"non-finite Adam state: parameter={name}, state={state_name}")
            return result

    return Guard()


class TrainingInterrupted(Exception):
    """A signal requested a clean stop at the last durable epoch boundary."""


def _active_training_process(pid: int, run_id: str) -> bool:
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    try:
        command = proc.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
    except OSError:
        return True
    return ("diagnose_xpert_training_reliability.py" in command and "--phase fit32" in command
            and "--worker" in command)


def acquire_training_lock(out: Path, run_id: str) -> tuple[Path, dict[str, Any] | None]:
    lock_path = RUN_ROOT / ".active_xpert_training.json"
    stale = None
    if lock_path.exists():
        try:
            previous = json.loads(lock_path.read_text(encoding="utf-8"))
            previous_pid = int(previous["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"training lock exists but cannot be audited safely: {lock_path}") from exc
        if _active_training_process(previous_pid, run_id):
            raise RuntimeError(f"an XPert fit process for {run_id} is still active (pid={previous_pid})")
        stale = previous
        archived = RUN_ROOT / f"active_xpert_training.stale.{time.time_ns()}.json"
        os.replace(lock_path, archived)
        stale["archived_as"] = archived.name
    lock_payload = {"run_id": run_id, "pid": os.getpid(),
                    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "command": sys.argv}
    fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(lock_payload, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return lock_path, stale


def release_training_lock(path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("cannot verify ownership of the XPert training lock") from exc
    if int(payload.get("pid", -1)) != os.getpid():
        raise RuntimeError("refusing to remove an XPert training lock owned by another process")
    path.unlink()


def capture_rng_state(torch: Any, loader_generator: Any) -> dict[str, Any]:
    return {"python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state().clone(),
            "loader_generator_state": loader_generator.get_state().clone()}


def restore_rng_state(torch: Any, loader_generator: Any, state: dict[str, Any]) -> None:
    required = {"python_random_state", "numpy_random_state", "torch_cpu_rng_state", "loader_generator_state"}
    if not required.issubset(state):
        raise ValueError(f"recovery checkpoint lacks RNG state: {sorted(required - set(state))}")
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"].cpu())
    loader_generator.set_state(state["loader_generator_state"].cpu())


def commit_latest_training_state(torch: Any, out: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Commit one of two full-state slots, then atomically move the latest pointer."""
    pointer_path = out / "latest_training_state.json"
    prior_slot = None
    if pointer_path.exists():
        prior = json.loads(pointer_path.read_text(encoding="utf-8"))
        prior_slot = prior.get("slot")
    slot = "b" if prior_slot == "a" else "a"
    filename = RECOVERY_STATE_SLOTS[1 if slot == "b" else 0]
    checkpoint_path = out / filename
    atomic_torch_save(torch, checkpoint_path, payload)
    pointer = {"schema_version": RECOVERY_SCHEMA_VERSION, "slot": slot,
               "checkpoint": filename, "sha256": sha256(checkpoint_path),
               "run_id": payload["run_id"], "completed_epochs": payload["completed_epochs"],
               "optimizer_steps": payload["optimizer_steps"], "key_manifest_sha256": payload["key_manifest_sha256"],
               "run_contract_sha256": payload["run_contract_sha256"],
               "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_json(pointer_path, pointer)
    return pointer


def load_latest_training_state(torch: Any, out: Path, *, run_id: str,
                               key_sha: str, contract_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    pointer_path = out / "latest_training_state.json"
    if not pointer_path.is_file():
        raise FileNotFoundError("no committed full-state XPert checkpoint is available for resume")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("schema_version") != RECOVERY_SCHEMA_VERSION:
        raise ValueError("unsupported XPert recovery checkpoint schema")
    filename = pointer.get("checkpoint")
    if filename not in RECOVERY_STATE_SLOTS:
        raise ValueError("latest pointer names an unapproved checkpoint path")
    checkpoint_path = out / filename
    if not checkpoint_path.is_file() or sha256(checkpoint_path) != pointer.get("sha256"):
        raise ValueError("latest full-state checkpoint is missing or fails its SHA256 binding")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (state.get("schema_version") != RECOVERY_SCHEMA_VERSION or state.get("run_id") != run_id
            or state.get("key_manifest_sha256") != key_sha
            or state.get("run_contract_sha256") != contract_sha
            or int(state.get("completed_epochs", -1)) != int(pointer.get("completed_epochs", -2))
            or int(state.get("optimizer_steps", -1)) != int(pointer.get("optimizer_steps", -2))):
        raise ValueError("full-state checkpoint metadata does not match the latest pointer or frozen run contract")
    if not isinstance(state.get("full_condition_keys"), list) or not state["full_condition_keys"]:
        raise ValueError("full-state checkpoint is missing its frozen condition-key list")
    if not isinstance(state.get("training_config"), dict) or not state.get("gene_axis_sha256"):
        raise ValueError("full-state checkpoint is missing its training or gene-axis contract")
    if not 0 <= int(state["completed_epochs"]) <= 2500:
        raise ValueError("full-state checkpoint epoch is outside the frozen endpoint")
    return state, pointer


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def reconcile_training_curve(out: Path, completed_epochs: int,
                             last_curve_record: dict[str, Any] | None) -> dict[str, Any]:
    curve_path = out / "XPert_train32_curve.jsonl"
    if not curve_path.exists():
        if completed_epochs:
            raise FileNotFoundError("training curve is missing despite committed optimizer updates")
        curve_path.touch(exist_ok=False)
        return {"complete_rows": 0, "appended_committed_row": False, "partial_tail_preserved": False}
    raw = curve_path.read_bytes()
    partial_preserved = False
    if raw and not raw.endswith(b"\n"):
        tail_start = raw.rfind(b"\n") + 1
        tail = raw[tail_start:]
        backup = out / f"XPert_train32_curve.partial_tail_epoch_{completed_epochs:04d}.bin"
        if backup.exists():
            raise FileExistsError(f"curve recovery backup already exists: {backup}")
        _atomic_write_bytes(backup, tail)
        _atomic_write_bytes(curve_path, raw[:tail_start])
        raw = raw[:tail_start]
        partial_preserved = True
    lines = raw.splitlines()
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"training curve has invalid complete JSON at row {index}") from exc
        if int(row.get("epoch", -1)) != index:
            raise ValueError("training curve epoch sequence is not contiguous from epoch 1")
        rows.append(row)
    if len(rows) > completed_epochs:
        raise ValueError("training curve is ahead of the latest committed optimizer state")
    appended = False
    if len(rows) == completed_epochs - 1:
        if not isinstance(last_curve_record, dict) or int(last_curve_record.get("epoch", -1)) != completed_epochs:
            raise ValueError("latest full-state checkpoint cannot repair the missing last curve row")
        line = json.dumps(last_curve_record, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
        with curve_path.open("ab") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        rows.append(last_curve_record)
        appended = True
    elif len(rows) == completed_epochs:
        if completed_epochs and canonical_json_sha(rows[-1]) != canonical_json_sha(last_curve_record):
            raise ValueError("last training curve row differs from the committed full-state checkpoint")
    elif len(rows) != completed_epochs:
        raise ValueError("training curve length cannot be reconciled with the latest checkpoint")
    return {"complete_rows": len(rows), "appended_committed_row": appended,
            "partial_tail_preserved": partial_preserved}


def make_recovery_payload(torch: Any, *, run_id: str, model: Any, optimizer: Any, scheduler: Any,
                          loader: Any, completed_epochs: int, optimizer_steps: int,
                          key_sha: str, contract_sha: str, last_curve_record: dict[str, Any] | None,
                          last_prediction: np.ndarray, full_condition_keys: list[str],
                          gene_axis_sha256: str, training_config: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": RECOVERY_SCHEMA_VERSION, "run_id": run_id,
            "completed_epochs": int(completed_epochs), "optimizer_steps": int(optimizer_steps),
            "key_manifest_sha256": key_sha, "run_contract_sha256": contract_sha,
            "full_condition_keys": [str(key) for key in full_condition_keys],
            "gene_axis_sha256": gene_axis_sha256, "training_config": training_config,
            "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": capture_rng_state(torch, loader.generator),
            "last_curve_record": last_curve_record,
            "last_prediction": torch.as_tensor(last_prediction, dtype=torch.float32).cpu().clone()}


def _atomic_text(path: Path, content: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
        if not content.endswith("\n"):
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def prepare_fit_run(run_id: str) -> Path:
    """Prepare a new immutable run directory by reusing the verified TRAIN readback."""
    out = output_root(run_id)
    if not out.exists():
        out.mkdir(parents=True, exist_ok=False)
    required = ("old_train_readback.json", "old_prediction_metrics.json", "exp055_train_curve_readback.tsv",
                "train32_keys.tsv")
    source = output_root("20261009_xpert_train_reliability_v3")
    has_fit_state = any((out / name).exists() for name in (
        "fit_preflight_started.json", "fit_started.json", "fit_terminal_receipt.json", "fit_failure.json",
        "latest_training_state.json"))
    for name in required:
        destination = out / name
        if destination.exists():
            continue
        if has_fit_state:
            raise FileNotFoundError(f"active run is missing required preserved input {destination}")
        original = source / name
        if not original.is_file():
            raise FileNotFoundError(f"verified v3 TRAIN artifact required for reuse is missing: {original}")
        shutil.copy2(original, destination)
    readback = json.loads((out / "old_train_readback.json").read_text(encoding="utf-8"))
    if readback.get("status") != "OLD_TRAIN_READBACK_COMPLETE":
        raise ValueError("copied prior TRAIN readback is not complete")
    recovery_path = out / "recovery_note.json"
    if not recovery_path.exists():
        v3_curve = source / "XPert_train32_curve.jsonl"
        rows = [json.loads(line) for line in v3_curve.read_text(encoding="utf-8").splitlines() if line]
        if not rows:
            raise ValueError("v3 has no persisted XPert training curve to account for")
        progress_path = source / "fit_progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
        started = json.loads((source / "fit_started.json").read_text(encoding="utf-8"))
        curve_mtime_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(v3_curve.stat().st_mtime))
        started_epoch = calendar.timegm(time.strptime(started["started_at_utc"], "%Y-%m-%dT%H:%M:%SZ"))
        wall_to_last_curve_write = max(0.0, v3_curve.stat().st_mtime - started_epoch)
        epoch_wall_sum = float(sum(float(row.get("epoch_wall_seconds", 0.0)) for row in rows))
        saved_checkpoints = {}
        for name in ("XPert_epoch000_initial.pt", "XPert_epoch070_after_warmup.pt", "XPert_epoch100.pt"):
            path = source / name
            if path.is_file():
                saved_checkpoints[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        old_cpu_seconds = progress.get("elapsed_process_cpu_seconds")
        note = {
            "prior_run_id": source.name, "prior_run_started_at_utc": started.get("started_at_utc"),
            "prior_pid": started.get("pid"), "current_process_check": "caller verified PID 734 absent from ps and pgrep",
            "curve_path": str(v3_curve.relative_to(ROOT)), "curve_rows": len(rows),
            "completed_optimizer_updates": int(rows[-1]["epoch"]),
            "last_curve_zero_based_epoch": int(rows[-1]["zero_based_epoch"]),
            "last_curve_file_mtime_utc": curve_mtime_utc,
            "wall_seconds_from_start_to_last_curve_write": wall_to_last_curve_write,
            "recorded_epoch_wall_seconds_sum": epoch_wall_sum,
            "last_saved_progress": progress, "recorded_cpu_hours_at_progress":
                float(old_cpu_seconds) / 3600.0 if old_cpu_seconds is not None else None,
            "cpu_time_after_last_saved_progress": "UNKNOWN_NOT_CAPTURED",
            "stop_reason": "UNKNOWN_NO_TERMINAL_OR_FAILURE_RECEIPT",
            "manual_cpu_time_cap": "ruled out by current task evidence",
            "prior_checkpoint_files": saved_checkpoints,
            "prior_checkpoint_contract": "model_state_dict only; no Adam, scheduler, RNG or dataloader-generator state; exact continuation unsupported",
            "prior_terminal_receipt_exists": (source / "fit_terminal_receipt.json").exists(),
            "prior_failure_receipt_exists": (source / "fit_failure.json").exists(),
            "new_inner_dev_numeric_values_read": False,
            "v3_directory_modified": False,
        }
        atomic_json(recovery_path, note, overwrite=False)
        hours_at_progress = note["recorded_cpu_hours_at_progress"]
        elapsed = note["wall_seconds_from_start_to_last_curve_write"]
        md = (
            "# XPert v3 中断与 v4 恢复记录\n\n"
            f"旧运行 `{source.name}` 共持久化 {len(rows)} 条曲线，完成 {int(rows[-1]['epoch'])} 次更新；"
            f"最后一条为 zero-based epoch {int(rows[-1]['zero_based_epoch'])}，文件时间 {curve_mtime_utc}。\n\n"
            f"启动至最后曲线写入约 {elapsed / 3600.0:.2f} 小时；曲线记录的各轮训练时间合计 "
            f"{epoch_wall_sum / 3600.0:.2f} 小时。最近一份累计 CPU 计时对应第 "
            f"{progress.get('completed_epochs', '未知')} 轮，已记录 "
            f"{hours_at_progress:.2f} CPU 小时；之后的准确 CPU 时间未保存。\n\n"
            "目录没有终点回执或失败回执。PID 734 已由当前任务负责人用 ps/pgrep 核实不存在；"
            "没有证据确定进程为何停止。人工 CPU 秒上限已由当前任务证据排除。\n\n"
            "v3 检查点只保存模型权重和轮次，没有 Adam、学习率调度器、随机数生成器或数据加载器随机状态，"
            "所以 v3 无法精确续训。v3 目录保持原样；v4 按相同 32 键、初始化种子、作者模型与 2500 轮固定终点从头训练。\n"
        )
        _atomic_text(out / "recovery_note.md", md)
    return out


def assert_no_other_xpert_fit_processes() -> None:
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        raise RuntimeError("cannot inspect Linux process table before starting the XPert fit")
    found = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = entry.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        except OSError:
            continue
        if ("diagnose_xpert_training_reliability.py" in command and "--phase fit32" in command
                and "--worker" in command):
            found.append({"pid": int(entry.name), "command": command})
    if found:
        raise RuntimeError(f"another XPert fit process is already active: {found}")


def launch_detached_fit(run_id: str, *, resume: bool = False) -> dict[str, Any]:
    target = output_root(run_id)
    if not resume and target.exists():
        source = output_root("20261009_xpert_train_reliability_v3")
        prepared_only_names = {"old_train_readback.json", "old_prediction_metrics.json",
                               "exp055_train_curve_readback.tsv", "train32_keys.tsv",
                               "recovery_note.json", "recovery_note.md"}
        existing_names = {path.name for path in target.iterdir()}
        if (not prepared_only_names.issubset(existing_names)
                or existing_names != prepared_only_names
                or any((target / name).exists() for name in ("fit_preflight_started.json", "fit_started.json",
                                                               "latest_training_state.json", "XPert_train32.stdout.log"))):
            raise FileExistsError(f"new fit requires an unused or preparation-only run id: {run_id}")
        for name in ("old_train_readback.json", "old_prediction_metrics.json",
                     "exp055_train_curve_readback.tsv", "train32_keys.tsv"):
            if sha256(target / name) != sha256(source / name):
                raise ValueError(f"preparation-only run artifact differs from the verified source: {name}")
    assert_no_other_xpert_fit_processes()
    out = prepare_fit_run(run_id)
    if not resume and any((out / name).exists() for name in (
            "fit_preflight_started.json", "fit_started.json", "fit_terminal_receipt.json", "fit_failure.json")):
        raise FileExistsError("the requested unique run id already contains fit state")
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log_path = out / "XPert_train32.stdout.log"
    command = [sys.executable, str(Path(__file__).resolve()), "--phase", "fit32", "--run-id", run_id, "--worker"]
    if resume:
        command.append("--resume")
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   close_fds=True)
    time.sleep(0.5)
    proc_cmdline = Path(f"/proc/{process.pid}/cmdline")
    try:
        worker_command = proc_cmdline.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
    except OSError:
        worker_command = ""
    worker_confirmed = "diagnose_xpert_training_reliability.py" in worker_command and run_id in worker_command and "--worker" in worker_command
    if process.poll() is None and worker_confirmed:
        launch_status = "DETACHED_WORKER_CONFIRMED_RUNNING"
    elif process.poll() is not None:
        launch_status = "WORKER_EXITED_DURING_LAUNCH"
    else:
        launch_status = "WORKER_LAUNCHED_BUT_PID_NOT_CONFIRMED"
    receipt = {"status": launch_status,
               "run_id": run_id, "worker_pid": process.pid, "launcher_pid": os.getpid(),
               "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "command": command, "cuda_visible_devices": "", "thread_environment": {
                   key: env[key] for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
               "stdout_log": str(log_path.relative_to(ROOT)), "resume": resume,
               "worker_command_line": worker_command, "worker_pid_confirmed": worker_confirmed,
               "return_code_after_500ms": process.poll()}
    atomic_json(out / "detached_launcher_receipt.json", receipt, overwrite=resume)
    return receipt


def _ensure_epoch_artifacts(torch: Any, out: Path, model: Any, prediction: np.ndarray,
                            key_rows: pd.DataFrame, genes: list[str], truth: np.ndarray,
                            *, epoch: int, optimizer_steps: int) -> None:
    checkpoint_names = {0: "XPert_epoch000_initial.pt", 70: "XPert_epoch070_after_warmup.pt",
                        100: "XPert_epoch100.pt", 2500: "XPert_epoch2500.pt"}
    if epoch not in checkpoint_names:
        return
    checkpoint = out / checkpoint_names[epoch]
    pred_path = out / f"XPert_epoch{epoch:03d}_train32_predictions.npz"
    if not checkpoint.exists():
        _save_checkpoint(torch, checkpoint, model, epoch=epoch, optimizer_steps=optimizer_steps)
    if not pred_path.exists():
        atomic_npz(pred_path,
                   full_condition_key=key_rows.full_condition_key.astype(str).to_numpy(dtype=np.str_),
                   source_object_id=key_rows.source_object_id.astype(str).to_numpy(dtype=np.str_),
                   full_inchikey=key_rows.full_inchikey.astype(str).to_numpy(dtype=np.str_),
                   genes=np.asarray(genes, dtype=np.str_), target_delta=truth,
                   prediction_delta=prediction)


def _assert_tree_equal(torch: Any, left: Any, right: Any, path: str = "root") -> None:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
            raise AssertionError(f"recovery mismatch at {path}")
    elif isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or not np.array_equal(left, right):
            raise AssertionError(f"recovery mismatch at {path}")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError(f"recovery mismatch at {path} keys")
        for key in left:
            _assert_tree_equal(torch, left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError(f"recovery mismatch at {path} length/type")
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_tree_equal(torch, a, b, f"{path}[{index}]")
    elif left != right:
        raise AssertionError(f"recovery mismatch at {path}: {left!r} != {right!r}")


def verify_recovery_protocol(torch: Any) -> dict[str, Any]:
    """One-time synthetic proof of exact optimizer/RNG restore and atomic pointer behavior."""
    with tempfile.TemporaryDirectory(prefix="exp069_xpert_recovery_") as temp:
        out = Path(temp)
        torch.manual_seed(SEED)
        random.seed(SEED)
        np.random.seed(SEED % (2**32))
        generator = torch.Generator().manual_seed(SEED)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.004, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 if epoch < 40 else 0.5)
        loader = type("SyntheticLoader", (), {"generator": generator})()

        def update(active_model: Any, active_optimizer: Any, active_scheduler: Any,
                   active_generator: Any) -> tuple[float, float, float, torch.Tensor]:
            batch_order = torch.randperm(8, generator=active_generator)
            x = torch.randn(8, 3)
            y = torch.randn(8, 2)
            active_optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(active_model(x[batch_order]), y[batch_order])
            loss.backward()
            active_optimizer.step()
            active_scheduler.step()
            return random.random(), float(np.random.random()), float(torch.rand(())), batch_order

        update(model, optimizer, scheduler, generator)
        row = {"epoch": 1, "synthetic_score": 0.25}
        state = make_recovery_payload(torch, run_id="synthetic", model=model, optimizer=optimizer,
                                      scheduler=scheduler, loader=loader, completed_epochs=1,
                                      optimizer_steps=1, key_sha="key-contract", contract_sha="run-contract",
                                      last_curve_record=row, last_prediction=np.ones((2, 3), dtype=np.float32),
                                      full_condition_keys=["synthetic-condition"], gene_axis_sha256="synthetic-genes",
                                      training_config={"epochs": 3, "learning_rate": 0.004})
        pointer = commit_latest_training_state(torch, out, state)
        expected_random = update(model, optimizer, scheduler, generator)
        expected_model = {key: value.detach().clone() for key, value in model.state_dict().items()}
        expected_optimizer = optimizer.state_dict()
        expected_scheduler = scheduler.state_dict()

        restored_model = torch.nn.Linear(3, 2)
        restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.004, weight_decay=1e-5)
        restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer,
                                                               lr_lambda=lambda epoch: 1.0 if epoch < 40 else 0.5)
        restored_loader = type("SyntheticLoader", (), {"generator": torch.Generator()})()
        loaded, _ = load_latest_training_state(torch, out, run_id="synthetic", key_sha="key-contract",
                                               contract_sha="run-contract")
        restored_model.load_state_dict(loaded["model_state_dict"], strict=True)
        restored_optimizer.load_state_dict(loaded["optimizer_state_dict"])
        restored_scheduler.load_state_dict(loaded["scheduler_state_dict"])
        restore_rng_state(torch, restored_loader.generator, loaded["rng_state"])
        observed_random = update(restored_model, restored_optimizer, restored_scheduler,
                                 restored_loader.generator)
        _assert_tree_equal(torch, expected_model, restored_model.state_dict(), "model")
        _assert_tree_equal(torch, expected_optimizer, restored_optimizer.state_dict(), "optimizer")
        _assert_tree_equal(torch, expected_scheduler, restored_scheduler.state_dict(), "scheduler")
        _assert_tree_equal(torch, expected_random, observed_random, "rng_and_batch_order")

        # A complete but unpointed alternate slot simulates a stop before pointer commit.
        abandoned = dict(state)
        abandoned["completed_epochs"] = 2
        abandoned["optimizer_steps"] = 2
        atomic_torch_save(torch, out / RECOVERY_STATE_SLOTS[1], abandoned)
        stable_state, stable_pointer = load_latest_training_state(torch, out, run_id="synthetic",
                                                                  key_sha="key-contract", contract_sha="run-contract")
        if stable_pointer["sha256"] != pointer["sha256"] or stable_state["completed_epochs"] != 1:
            raise AssertionError("latest pointer did not preserve the prior complete checkpoint")

        curve = out / "XPert_train32_curve.jsonl"
        curve.write_text(json.dumps(row, sort_keys=True) + "\n{\"partial\":", encoding="utf-8")
        recovery = reconcile_training_curve(out, 1, row)
        if not recovery["partial_tail_preserved"] or not recovery["complete_rows"] == 1:
            raise AssertionError("curve reconciliation did not preserve and repair a partial tail")
        missing_curve = out / "missing" / "XPert_train32_curve.jsonl"
        missing_curve.parent.mkdir()
        missing_curve.touch()
        _atomic_write_bytes(missing_curve, b"")
        missing_receipt = reconcile_training_curve(missing_curve.parent, 1, row)
        if not missing_receipt["appended_committed_row"]:
            raise AssertionError("curve reconciliation did not restore the checkpointed final row")
    return {"status": "RECOVERY_PROTOCOL_VERIFIED", "exact_model_adam_scheduler_rng_restore": True,
            "unpointed_checkpoint_slot_ignored": True, "curve_partial_tail_preserved": True,
            "checkpointed_curve_row_restored": True}


def install_cpu_attention_layout_adapter(model_utils: Any) -> dict[str, Any]:
    """Make upstream SDPA output safe for its immediate head-flattening view().

    On CPU with dropout enabled, torch SDPA can return a transposed, strided
    attention tensor that upstream XPert immediately flattens with ``view``.
    Materializing the same values contiguously preserves the tensor's shape,
    values, and gradients while making that reshape valid.
    """
    original = model_utils._attention
    if getattr(original, "_exp069_cpu_layout_adapter", False):
        raise RuntimeError("EXP-069 CPU attention layout adapter was already installed")
    counts = {"calls": 0, "cpu_outputs_materialized": 0}

    def cpu_layout_safe_attention(query: Any, key: Any, value: Any, dropout_p: float) -> Any:
        result = original(query, key, value, dropout_p)
        counts["calls"] += 1
        if result.device.type == "cpu" and not result.is_contiguous():
            result = result.contiguous()
            counts["cpu_outputs_materialized"] += 1
        return result

    cpu_layout_safe_attention._exp069_cpu_layout_adapter = True
    model_utils._attention = cpu_layout_safe_attention
    return {"patch": "CPU-only materialize non-contiguous _attention output before upstream view()",
            "numerical_change": False, "changes_attention_formula_or_dropout": False,
            "original_attention_source": str(Path(model_utils.__file__).resolve()),
            "original_attention_source_sha256": sha256(Path(model_utils.__file__).resolve()),
            "runtime_counts": counts}


def author_losses(torch: Any, model: Any, batch: tuple[Any, ...], source_losses: tuple[Any, Any, Any, dict[str, Any]],
                  config: dict[str, Any]) -> dict[str, float]:
    _, mse_sum, pcc_sum, _ = source_losses
    model.eval()
    with torch.no_grad():
        trt_out, ctl_out, delta_out, trt_target, ctl_target, *_ = model(batch)
        delta_target = trt_target - ctl_target
        l1 = mse_sum(trt_out, trt_target)
        l2 = mse_sum(ctl_out, ctl_target)
        l3 = mse_sum(delta_out, delta_target)
        l4 = pcc_sum(delta_out, delta_target)
        raw = sum(weight * loss for weight, loss in zip(config["train"]["loss_weight"], (l1, l2, l3, l4)))
        count = int(len(delta_target))
        warmup = (config["train"]["loss_weight"][0] * torch.sqrt(l1 / count)
                  + config["train"]["loss_weight"][1] * torch.sqrt(l2 / count)
                  + config["train"]["loss_weight"][2] * torch.sqrt(l3 / count)
                  + config["train"]["loss_weight"][3] * (l4 / count))
    values = {"treatment_mse_sum": l1, "control_mse_sum": l2, "delta_mse_sum": l3,
              "delta_pearson_loss_sum": l4, "raw_weighted_sum": raw, "warmup_weighted_per_sample": warmup,
              "treatment_rmse_per_sample": torch.sqrt(l1 / count),
              "control_rmse_per_sample": torch.sqrt(l2 / count),
              "delta_rmse_per_sample": torch.sqrt(l3 / count),
              "delta_output_mean": torch.mean(delta_out), "delta_output_sd": torch.std(delta_out),
              "treatment_output_mean": torch.mean(trt_out), "control_output_mean": torch.mean(ctl_out)}
    result = {key: float(value.detach().cpu()) for key, value in values.items()}
    for key, value in result.items():
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite author diagnostic loss/output: {key}")
    return result


def _save_checkpoint(torch: Any, path: Path, model: Any, *, epoch: int, optimizer_steps: int) -> None:
    _save_checkpoint_state(torch, path, model.state_dict(), epoch=epoch, optimizer_steps=optimizer_steps)


def _save_checkpoint_state(torch: Any, path: Path, state_dict: dict[str, Any], *, epoch: int,
                           optimizer_steps: int, checkpoint_note: str = "fixed EXP-069 TRAIN-only diagnostic checkpoint; no inner-dev model selection") -> None:
    payload = {"model_state_dict": {key: value.detach().cpu().clone() for key, value in state_dict.items()},
               "epoch_one_based": int(epoch), "seed": SEED, "optimizer_steps": int(optimizer_steps),
               "interpretation": checkpoint_note}
    atomic_torch_save(torch, path, payload)


def _predict_delta(torch: Any, model: Any, tensors: tuple[Any, ...], *, training_targets: bool = False) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        output = model(tensors)
        pred = output[2].detach().cpu().numpy().astype(np.float32)
    if pred.shape != (len(tensors[0]), GENE_COUNT) or not np.isfinite(pred).all():
        raise FloatingPointError("non-finite or wrong-shaped author deg_output")
    return pred


def fit_train32(run_id: str, *, resume: bool = False) -> dict[str, Any]:
    out = prepare_fit_run(run_id)
    if (out / "fit_terminal_receipt.json").exists() or (out / "fit_failure.json").exists():
        raise FileExistsError("this EXP-069 XPert run is already closed and cannot be resumed")
    lock_path, stale_lock = acquire_training_lock(out, run_id)
    try:
        return _fit_train32_locked(run_id, resume=resume, stale_lock=stale_lock)
    finally:
        release_training_lock(lock_path)


def _fit_train32_locked(run_id: str, *, resume: bool, stale_lock: dict[str, Any] | None) -> dict[str, Any]:
    out = output_root(run_id)
    if not out.is_dir() or not (out / "old_train_readback.json").is_file():
        raise FileNotFoundError("verified prior TRAIN readback is required before the fixed 32-key fit")
    if not resume:
        if any((out / name).exists() for name in ("fit_preflight_started.json", "fit_started.json",
                                                   "fit_terminal_receipt.json", "fit_failure.json",
                                                   "latest_training_state.json")):
            raise FileExistsError("this unique EXP-069 XPert fit run has already started or closed")
    elif not all((out / name).is_file() for name in ("fit_preflight_started.json", "fit_started.json",
                                                      "latest_training_state.json")):
        raise FileNotFoundError("safe resume requires the original manifest and a committed full-state checkpoint")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in {"", "-1", "None"}:
        raise RuntimeError("CPU-only XPert diagnostic requires CUDA_VISIBLE_DEVICES empty")
    if not resume:
        preflight = {"status": "CPU_FIT_PREFLIGHT_STARTED", "run_id": run_id,
                     "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "pid": os.getpid(), "device": "cpu", "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                     "GPU_hours": 0.0, "seed": SEED, "target_epochs": 2500,
                     "protocol": "EXP-069 frozen 32-key TRAIN-only XPert diagnostic; no inner-dev reads or selection"}
        atomic_json(out / "fit_preflight_started.json", preflight, overwrite=False)
    elif (out / "fit_interrupted.json").exists():
        interrupted = json.loads((out / "fit_interrupted.json").read_text(encoding="utf-8"))
        if not interrupted.get("resumable", False):
            raise RuntimeError("the prior XPert interruption was not marked safe to resume")
    readback_path = out / "old_train_readback.json"
    readback_provenance = json.loads(readback_path.read_text(encoding="utf-8"))
    if readback_provenance.get("status") != "OLD_TRAIN_READBACK_COMPLETE":
        raise ValueError("reused TRAIN-only historical readback is incomplete")
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before CPU-only fit")
    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED % (2**32))
    import exp051_contract_fit_smoke as helper

    keys_all = read_frozen_train_keys()
    subset = select_train32(keys_all)
    subset["selection_sha256"] = subset.full_condition_key.astype(str).map(
        lambda key: hashlib.sha256(f"{SEED}|{key}".encode("utf-8")).hexdigest())
    frozen_all = keys_all.reset_index(drop=True)
    x_all, ctl_all, delta_all, genes, source_meta = load_train_values_from_author_h5ad(frozen_all)
    pos = {key: idx for idx, key in enumerate(frozen_all.full_condition_key.astype(str))}
    positions = np.asarray([pos[key] for key in subset.full_condition_key.astype(str)], dtype=np.int64)
    x = x_all[positions]
    ctl = ctl_all[positions]
    truth = delta_all[positions]
    variance = np.var(delta_all.astype(np.float64), axis=0, ddof=0)
    if not np.array_equal(subset.full_condition_key.astype(str).to_numpy(),
                          frozen_all.full_condition_key.astype(str).to_numpy()[positions]):
        raise ValueError("selected TRAIN32 keys do not map exactly into the frozen full-TRAIN data")
    baseline = baseline_predictions(subset, truth)
    baseline_metrics = {name: metrics(subset, truth, pred, variance) for name, pred in baseline.items()}
    strongest_name, strongest_metrics = min(baseline_metrics.items(),
                                             key=lambda item: item[1]["object_macro_nmse"])
    if not (out / "train32_keys.tsv").exists():
        tmp = out / "train32_keys.tsv.tmp"
        subset.to_csv(tmp, sep="\t", index=False, lineterminator="\n")
        os.replace(tmp, out / "train32_keys.tsv")
    elif sha256(out / "train32_keys.tsv") != canonical_sha(subset.to_csv(sep="\t", index=False, lineterminator="\n")):
        raise ValueError("existing frozen TRAIN32 key manifest differs from deterministic key selection")

    source_meta_keys = {"xpert_train": helper.XPERT_ROOT / "train_xpert.py",
                        "xpert_model": helper.XPERT_ROOT / "models/model_XPert.py",
                        "xpert_utils": helper.XPERT_ROOT / "utils.py",
                        "xpert_dataset": helper.XPERT_ROOT / "datasets/MyDataset.py",
                        "xpert_config": helper.XPERT_ROOT / "configs/config_l1000.yaml",
                        "unimol": helper.XPERT_ROOT / "processed_data/all_drugs_unimol_arr.npy",
                        "hg": helper.XPERT_ROOT / "HG_data/saved_embedding/HG_drug_embeddings.npy",
                        "ppi": helper.XPERT_ROOT / "processed_data/PPI_gene_vector_128d.npy"}
    feature_info = verify_feature_axes(subset, xpert_root=helper.XPERT_ROOT)
    xpert_module, config, _, _ = helper._training_modules(torch)
    model_utils = importlib.import_module("models.model_utils")
    cpu_attention_adapter = install_cpu_attention_layout_adapter(model_utils)
    train_fn, mse_sum, pcc_sum, source_receipt = helper._source_functions(
        helper.XPERT_ROOT / "train_xpert.py", helper.XPERT_ROOT / "utils.py")
    tr_cfg = config["train"]
    if (int(tr_cfg["num_epochs"]) != 2500 or int(tr_cfg["batch_size"]) != 128
            or int(tr_cfg["init_epoch"]) != 70 or int(tr_cfg["patience"]) != 50
            or float(tr_cfg["train_lr"]) != 0.004 or float(tr_cfg["weight_decay"]) != 1e-5
            or list(tr_cfg["loss_weight"]) != LOSS_WEIGHTS):
        raise ValueError("upstream XPert author schedule differs from the frozen EXP-069 protocol")
    config["model"]["ATTN"]["ppi_gene_vector_path"] = str(helper.XPERT_ROOT / "processed_data/PPI_gene_vector_128d.npy")
    config["model"]["HG"]["drug_hg_pretrained_embed_path"] = str(helper.XPERT_ROOT / "HG_data/saved_embedding/HG_drug_embeddings.npy")
    values = {"X": x, "X_ctl": ctl, "delta": truth, "genes": np.asarray(genes, dtype=str)}
    unimol = np.load(helper.XPERT_ROOT / "processed_data/all_drugs_unimol_arr.npy", mmap_mode="r", allow_pickle=False)
    tensors = helper._xpert_tensors(torch, values, subset.reset_index(drop=True),
                                    np.arange(len(subset), dtype=np.int64), unimol, config,
                                    training_targets=True)
    tensor_dataset = torch.utils.data.TensorDataset(*tensors)
    loader_generator = torch.Generator().manual_seed(SEED)
    loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=128, shuffle=True,
                                         drop_last=False, num_workers=0,
                                         generator=loader_generator)
    if len(loader) != 1:
        raise ValueError(f"frozen 32-condition fit must use exactly one batch, observed {len(loader)}")
    model, train_view = helper._new_xpert(torch, xpert_module, config, torch.device("cpu"), SEED)
    if torch.cuda.is_initialized() or next(model.parameters()).device.type != "cpu":
        raise RuntimeError("XPert initialized off CPU or initialized CUDA in the frozen CPU-only run")
    optimizer_raw = torch.optim.Adam(model.parameters(), lr=0.004, weight_decay=1e-5)
    optimizer_steps = 0
    guard_state: dict[str, Any] = {}
    optimizer = guarded_optimizer(torch, optimizer_raw, model, guard_state)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_raw, lr_lambda=lambda epoch: 1.0 if epoch < 40 else 0.5)
    source_losses = (train_fn, mse_sum, pcc_sum, source_receipt)
    predict_tensors = helper._xpert_tensors(torch, values, subset.reset_index(drop=True),
                                            np.arange(len(subset), dtype=np.int64), unimol,
                                            config, training_targets=False)
    previous_manifest = (json.loads((out / "fit_started.json").read_text(encoding="utf-8"))
                         if resume else None)
    if resume:
        initial_prediction = None
        initial_metrics = previous_manifest["initial_epoch0_project_delta_metrics"]
        initial_author_losses = previous_manifest["initial_epoch0_author_loss"]
    else:
        initial_prediction = _predict_delta(torch, model, predict_tensors)
        initial_metrics = metrics(subset, truth, initial_prediction, variance)
        initial_author_losses = author_losses(torch, model, tensors, source_losses, config)
    loss_namespace = train_fn.__globals__
    original_mse, original_pcc = loss_namespace["mse_loss_ls_sum"], loss_namespace["pcc_loss_sum"]
    observed_losses: dict[str, float] = {}

    def checked_loss(name: str, function: Any) -> Any:
        def wrapped(pred: Any, target: Any) -> Any:
            value = function(pred, target)
            numeric = float(value.detach().cpu())
            observed_losses[name] = numeric
            if not math.isfinite(numeric):
                guard_state.update({"first_nonfinite_component": "loss", "loss_name": name})
                raise FloatingPointError(f"non-finite author loss: {name}")
            return value
        return wrapped

    mse_call = {"n": 0}
    def wrapped_mse(pred: Any, target: Any) -> Any:
        mse_call["n"] += 1
        name = {1: "treated_expression_mse", 2: "control_expression_mse", 3: "delta_mse"}.get(mse_call["n"], f"mse_call_{mse_call['n']}")
        value = original_mse(pred, target)
        numeric = float(value.detach().cpu())
        observed_losses[name] = numeric
        if not math.isfinite(numeric):
            guard_state.update({"first_nonfinite_component": "loss", "loss_name": name})
            raise FloatingPointError(f"non-finite author loss: {name}")
        return value
    def wrapped_pcc(pred: Any, target: Any) -> Any:
        value = original_pcc(pred, target)
        numeric = float(value.detach().cpu())
        observed_losses["delta_pearson_loss"] = numeric
        if not math.isfinite(numeric):
            guard_state.update({"first_nonfinite_component": "loss", "loss_name": "delta_pearson_loss"})
            raise FloatingPointError("non-finite author loss: delta_pearson_loss")
        return value
    loss_namespace["mse_loss_ls_sum"] = wrapped_mse
    loss_namespace["pcc_loss_sum"] = wrapped_pcc

    manifest = {
        "status": "CPU_FIT_STARTED", "run_id": run_id,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(), "device": "cpu", "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "GPU_hours": 0.0, "seed": SEED,
        "subset_rule": "five sorted full InChIKeys: 6 conditions per drug plus one extra for each first two; within drug sort SHA256('20261006|'+full_condition_key), then final full_condition_key sort",
        "input_contract": {"conditions": 32, "source_objects": int(subset.source_object_id.nunique()),
                           "full_inchikeys": int(subset.full_inchikey.nunique()), "genes": GENE_COUNT,
                           "all_32_keys_within_authorized_419_TRAIN": True,
                           "source_h5ad": source_meta,
                           "truth": "X-X_ctl in source-provided units", "new_inner_dev_numeric_values_read": False},
        "old_train_readback_reuse": {"copied_into_run_path": str(readback_path.relative_to(ROOT)),
                                      "source_run_id": readback_provenance.get("run_id"),
                                      "sha256": sha256(readback_path),
                                      "original_source_path": f"artifacts/experiments/EXP-069/xpert/{readback_provenance.get('run_id')}/old_train_readback.json"},
        "baseline_metrics_train32": baseline_metrics,
        "initial_epoch0_project_delta_metrics": initial_metrics,
        "initial_epoch0_author_loss": initial_author_losses,
        "strongest_baseline": {"name": strongest_name, **strongest_metrics},
        "NMSE_variance_source": "full 419-condition authorized TRAIN only; ddof=0",
        "official_features": feature_info,
        "official_schedule": {"architecture": "upstream XPertNet fresh initialization",
                              "epochs": 2500, "batch_size": 128, "learning_rate": 0.004,
                              "weight_decay": 1e-5, "loss_weights": LOSS_WEIGHTS,
                              "warmup_epoch_indexes": "0-69", "raw_loss_from_epoch_index": 70,
                              "lr_scheduler": "1.0 through epoch index 39; 0.5 from index 40",
                              "validation_early_stopping": False,
                              "CPU_attention_layout_adapter": {key: value for key, value in cpu_attention_adapter.items()
                                                                if key != "runtime_counts"},
                              "loss_source": source_receipt,
                              "interpretation": "TRAIN-only fixed endpoint diagnostic; not EXP-055/063 official selected model"},
        "source_sha256": {name: sha256(path) for name, path in source_meta_keys.items()},
        "runner_sha256": sha256(Path(__file__).resolve()),
        "runner_adapter_sha256": sha256(helper.Path(helper.__file__).resolve()),
        "cpu_threads": torch.get_num_threads(), "cpu_interop_threads": torch.get_num_interop_threads(),
    }
    key_manifest_sha = sha256(out / "train32_keys.tsv")
    contract_fields = {"run_id": run_id, "seed": SEED, "target_epochs": 2500,
                       "key_manifest_sha256": key_manifest_sha,
                       "input_contract": manifest["input_contract"],
                       "old_train_readback_sha256": manifest["old_train_readback_reuse"]["sha256"],
                       "official_features": manifest["official_features"],
                       "official_schedule": manifest["official_schedule"],
                       "source_sha256": manifest["source_sha256"],
                       "runner_sha256": manifest["runner_sha256"],
                       "runner_adapter_sha256": manifest["runner_adapter_sha256"],
                       "cpu_threads": manifest["cpu_threads"],
                       "cpu_interop_threads": manifest["cpu_interop_threads"]}
    contract_sha = canonical_json_sha(contract_fields)
    manifest["run_contract_sha256"] = contract_sha
    key_rows = subset[["full_condition_key", "source_object_id", "full_inchikey", "pert_id",
                       "source_row", "drug_node_idx", "pert_dose_num", "pert_time_num"]].copy()
    if not resume:
        if initial_prediction is None:
            raise RuntimeError("fresh XPert initialization did not produce its epoch-0 prediction")
        _save_checkpoint(torch, out / "XPert_epoch000_initial.pt", model, epoch=0, optimizer_steps=0)
        atomic_npz(out / "XPert_epoch000_train32_predictions.npz",
                   full_condition_key=key_rows.full_condition_key.astype(str).to_numpy(dtype=np.str_),
                   source_object_id=key_rows.source_object_id.astype(str).to_numpy(dtype=np.str_),
                   full_inchikey=key_rows.full_inchikey.astype(str).to_numpy(dtype=np.str_),
                   genes=np.asarray(genes, dtype=np.str_), target_delta=truth,
                   prediction_delta=initial_prediction)
        state = make_recovery_payload(torch, run_id=run_id, model=model, optimizer=optimizer_raw,
                                      scheduler=scheduler, loader=loader, completed_epochs=0,
                                      optimizer_steps=0, key_sha=key_manifest_sha, contract_sha=contract_sha,
                                      last_curve_record=None, last_prediction=initial_prediction,
                                      full_condition_keys=key_rows.full_condition_key.astype(str).tolist(),
                                      gene_axis_sha256=source_meta["gene_axis_sha256"],
                                      training_config=manifest["official_schedule"])
        initial_pointer = commit_latest_training_state(torch, out, state)
        manifest["initial_full_state_checkpoint"] = initial_pointer
        atomic_json(out / "fit_started.json", manifest, overwrite=False)
        curve_path = out / "XPert_train32_curve.jsonl"
        curve_path.touch(exist_ok=False)
        completed_epochs, optimizer_steps = 0, 0
        last_curve_record = None
        last_finite_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        last_finite_prediction = initial_prediction.copy()
        last_finite_epoch = 0
        curve_recovery = {"complete_rows": 0, "appended_committed_row": False,
                          "partial_tail_preserved": False}
        progress_row = {"status": "STARTED_FROM_FRESH_INITIALIZATION", "completed_epochs": 0,
                        "target_epochs": 2500, "pid": os.getpid(),
                        "latest_state_sha256": initial_pointer["sha256"],
                        "run_contract_sha256": contract_sha,
                        "elapsed_wall_seconds": 0.0, "elapsed_process_cpu_seconds": 0.0}
        atomic_json(out / "fit_progress.json", progress_row)
        print(json.dumps({"stage": "XPert_EXP069_CPU_TRAIN32_STARTED", **progress_row}, ensure_ascii=False),
              flush=True)
    else:
        if previous_manifest is None or previous_manifest.get("run_contract_sha256") != contract_sha:
            raise ValueError("current inputs, source code, feature axes, or training protocol differ from the original run")
        state, pointer = load_latest_training_state(torch, out, run_id=run_id,
                                                   key_sha=key_manifest_sha, contract_sha=contract_sha)
        if state["full_condition_keys"] != key_rows.full_condition_key.astype(str).tolist():
            raise ValueError("checkpoint condition keys differ from the current frozen 32-key manifest")
        if state["gene_axis_sha256"] != source_meta["gene_axis_sha256"]:
            raise ValueError("checkpoint gene axis differs from the current frozen source axis")
        if canonical_json_sha(state["training_config"]) != canonical_json_sha(manifest["official_schedule"]):
            raise ValueError("checkpoint training schedule differs from the current frozen XPert protocol")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer_raw.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        optimizer_steps = int(state["optimizer_steps"])
        completed_epochs = int(state["completed_epochs"])
        if optimizer_steps != completed_epochs:
            raise ValueError("the saved optimizer update count does not equal the committed epoch count")
        restore_rng_state(torch, loader_generator, state["rng_state"])
        last_curve_record = state.get("last_curve_record")
        last_finite_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        last_finite_prediction = state["last_prediction"].detach().cpu().numpy().astype(np.float32, copy=True)
        last_finite_epoch = completed_epochs
        curve_recovery = reconcile_training_curve(out, completed_epochs, last_curve_record)
        for milestone in (0, 70, 100):
            if milestone < completed_epochs:
                checkpoint_name = {0: "XPert_epoch000_initial.pt", 70: "XPert_epoch070_after_warmup.pt",
                                   100: "XPert_epoch100.pt"}[milestone]
                prediction_name = f"XPert_epoch{milestone:03d}_train32_predictions.npz"
                if not (out / checkpoint_name).is_file() or not (out / prediction_name).is_file():
                    raise FileNotFoundError(f"required completed milestone artifact is missing: {checkpoint_name}")
        _ensure_epoch_artifacts(torch, out, model, last_finite_prediction, key_rows, genes, truth,
                                epoch=completed_epochs, optimizer_steps=optimizer_steps)
        resume_records = out / "fit_resume_history.jsonl"
        resume_row = {"resume_number": 1 + (len(resume_records.read_text(encoding="utf-8").splitlines())
                                               if resume_records.exists() else 0),
                      "resumed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "pid": os.getpid(), "completed_epochs": completed_epochs,
                      "latest_state_sha256": pointer["sha256"], "curve_recovery": curve_recovery,
                      "stale_lock": stale_lock}
        with resume_records.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(resume_row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        progress_row = {"status": "RESUMED_FROM_FULL_STATE", "completed_epochs": completed_epochs,
                        "target_epochs": 2500, "pid": os.getpid(),
                        "latest_state_sha256": pointer["sha256"], "curve_recovery": curve_recovery,
                        "elapsed_wall_seconds": 0.0, "elapsed_process_cpu_seconds": 0.0}
        if last_curve_record is not None:
            progress_row.update({"train_object_macro_nmse": last_curve_record["project_delta_metrics"]["object_macro_nmse"],
                                 "train_amplitude_ratio": last_curve_record["project_delta_metrics"]["object_macro_amplitude_ratio"]})
        atomic_json(out / "fit_progress.json", progress_row)
        print(json.dumps({"stage": "XPert_EXP069_CPU_TRAIN32_RESUMED", **progress_row}, ensure_ascii=False),
              flush=True)

    start_wall, start_cpu = time.perf_counter(), time.process_time()
    terminal_status = "COMPLETED_2500_EPOCH_TRAIN_ONLY_DIAGNOSTIC"
    durable_completed_epochs = completed_epochs
    active_epoch = completed_epochs + 1
    signal_handlers = {}
    def _request_stop(signum: int, _frame: Any) -> None:
        raise TrainingInterrupted(f"received signal {signum}; latest full-state epoch is {durable_completed_epochs}")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, _request_stop)
    try:
        with (out / "XPert_train32_curve.jsonl").open("a", encoding="utf-8", newline="\n") as curve:
            for epoch in range(completed_epochs, 2500):
                active_epoch = epoch + 1
                mse_call["n"] = 0
                observed_losses.clear()
                lr = float(optimizer_raw.param_groups[0]["lr"])
                fit_started = time.perf_counter()
                before_pred = None
                losses = train_fn(train_view, optimizer, loader,
                                  type("AuthorArgs", (), {"pretrained_model": False})(),
                                  config, scaler=None, epoch=epoch)
                optimizer_steps += 1
                if not all(math.isfinite(float(value)) for value in losses):
                    guard_state.update({"first_nonfinite_component": "author_returned_loss_tuple",
                                        "loss_values": [float(value) if math.isfinite(float(value)) else None for value in losses]})
                    raise FloatingPointError("non-finite author-returned loss tuple")
                model.eval()
                predict_tensors = helper._xpert_tensors(torch, values, subset.reset_index(drop=True),
                                                        np.arange(len(subset), dtype=np.int64), unimol,
                                                        config, training_targets=False)
                pred = _predict_delta(torch, model, predict_tensors)
                metric = metrics(subset, truth, pred, variance)
                actual_author_losses = author_losses(torch, model, tensors, source_losses, config)
                scheduler.step()
                row = {
                    "epoch": epoch + 1, "zero_based_epoch": epoch,
                    "schedule_stage": "per_sample_root_mse_warmup" if epoch < 70 else "raw_weighted_sum",
                    "learning_rate_before_epoch": lr,
                    "author_train_returned_losses": [float(value) for value in losses],
                    "author_train_returned_loss_components": {"weighted_loss_per_source_sample": float(losses[0]),
                                                               "treatment_head_rmse": float(losses[1]),
                                                               "control_head_rmse": float(losses[2]),
                                                               "delta_head_rmse": float(losses[3]),
                                                               "delta_pearson_loss": float(losses[4])},
                    "author_loss_recomputed_at_checkpoint": actual_author_losses,
                    "CPU_attention_layout_adapter_runtime_counts": dict(cpu_attention_adapter["runtime_counts"]),
                    "project_delta_metrics": metric,
                    "gradient_tensors_finite_before_step": True,
                    "parameters_and_adam_state_finite_after_step": True,
                    "epoch_wall_seconds": time.perf_counter() - fit_started,
                    "train_predictions_finite": True,
                }
                state = make_recovery_payload(torch, run_id=run_id, model=model, optimizer=optimizer_raw,
                                              scheduler=scheduler, loader=loader, completed_epochs=epoch + 1,
                                              optimizer_steps=optimizer_steps, key_sha=key_manifest_sha,
                                              contract_sha=contract_sha, last_curve_record=row,
                                              last_prediction=pred,
                                              full_condition_keys=key_rows.full_condition_key.astype(str).tolist(),
                                              gene_axis_sha256=source_meta["gene_axis_sha256"],
                                              training_config=manifest["official_schedule"])
                pointer = commit_latest_training_state(torch, out, state)
                durable_completed_epochs = epoch + 1
                completed_epochs = durable_completed_epochs
                last_finite_epoch = durable_completed_epochs
                last_finite_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                last_finite_prediction = pred.copy()
                curve.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                curve.flush()
                os.fsync(curve.fileno())
                _ensure_epoch_artifacts(torch, out, model, pred, key_rows, genes, truth,
                                        epoch=epoch + 1, optimizer_steps=optimizer_steps)
                progress = {"completed_epochs": epoch + 1, "target_epochs": 2500,
                            "train_object_macro_nmse": metric["object_macro_nmse"],
                            "train_amplitude_ratio": metric["object_macro_amplitude_ratio"],
                            "elapsed_wall_seconds": time.perf_counter() - start_wall,
                            "elapsed_process_cpu_seconds": time.process_time() - start_cpu,
                            "pid": os.getpid(), "latest_state_sha256": pointer["sha256"],
                            "run_contract_sha256": contract_sha}
                atomic_json(out / "fit_progress.json", progress)
                if (epoch + 1) % 10 == 0 or epoch + 1 in {70, 100, 2500}:
                    print(json.dumps({"stage": "XPert_EXP069_CPU_TRAIN32_PROGRESS", **progress}, ensure_ascii=False),
                          flush=True)
    except BaseException as exc:
        if isinstance(exc, (TrainingInterrupted, KeyboardInterrupt)):
            try:
                pointer_now = json.loads((out / "latest_training_state.json").read_text(encoding="utf-8"))
                checkpoint_now = out / pointer_now["checkpoint"]
                if (pointer_now.get("run_id") == run_id and pointer_now.get("run_contract_sha256") == contract_sha
                        and checkpoint_now.is_file() and sha256(checkpoint_now) == pointer_now.get("sha256")):
                    durable_completed_epochs = int(pointer_now["completed_epochs"])
            except (OSError, KeyError, ValueError, json.JSONDecodeError):
                pass
        safe_boundary = (isinstance(exc, (TrainingInterrupted, KeyboardInterrupt))
                         or durable_completed_epochs == active_epoch or durable_completed_epochs == 2500)
        if safe_boundary:
            interrupted = {"status": "INTERRUPTED_RESUMABLE_FROM_LATEST_FULL_STATE",
                           "run_id": run_id, "pid": os.getpid(), "resumable": True,
                           "interrupted_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "active_epoch": active_epoch, "completed_epochs": durable_completed_epochs,
                           "optimizer_steps": optimizer_steps if durable_completed_epochs == active_epoch else durable_completed_epochs,
                           "rollback_uncommitted_epoch": durable_completed_epochs < active_epoch,
                           "error_type": type(exc).__name__, "error": str(exc),
                           "elapsed_wall_seconds": time.perf_counter() - start_wall,
                           "elapsed_process_cpu_seconds": time.process_time() - start_cpu,
                           "GPU_hours": 0.0}
            atomic_json(out / "fit_interrupted.json", interrupted)
            with (out / "fit_interruptions.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(interrupted, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        else:
            failure_kind = "NONFINITE_OR_INVALID_NUMERIC_OUTPUT" if isinstance(exc, FloatingPointError) else "IMPLEMENTATION_EXCEPTION"
            terminal_status = f"FAILED_CLOSED_{failure_kind}"
            last_finite_checkpoint = out / f"XPert_last_finite_epoch{last_finite_epoch:04d}_on_failure.pt"
            _save_checkpoint_state(torch, last_finite_checkpoint, last_finite_state,
                                   epoch=last_finite_epoch, optimizer_steps=durable_completed_epochs,
                                   checkpoint_note="last durable finite model weights before closed failure; fixed TRAIN-only diagnostic")
            last_finite_prediction_path = out / f"XPert_last_finite_epoch{last_finite_epoch:04d}_train32_predictions_on_failure.npz"
            atomic_npz(last_finite_prediction_path,
                       full_condition_key=key_rows.full_condition_key.astype(str).to_numpy(dtype=np.str_),
                       source_object_id=key_rows.source_object_id.astype(str).to_numpy(dtype=np.str_),
                       full_inchikey=key_rows.full_inchikey.astype(str).to_numpy(dtype=np.str_),
                       genes=np.asarray(genes, dtype=np.str_), target_delta=truth,
                       prediction_delta=last_finite_prediction)
            failure = {"status": terminal_status, "run_id": run_id, "pid": os.getpid(),
                       "failed_epoch": active_epoch, "last_finite_epoch": last_finite_epoch,
                       "failure_kind": failure_kind,
                       "last_finite_checkpoint": {"path": str(last_finite_checkpoint.relative_to(ROOT)),
                                                   "sha256": sha256(last_finite_checkpoint),
                                                   "bytes": last_finite_checkpoint.stat().st_size},
                       "last_finite_predictions": {"path": str(last_finite_prediction_path.relative_to(ROOT)),
                                                   "sha256": sha256(last_finite_prediction_path)},
                       "error_type": type(exc).__name__, "error": str(exc), "first_failure": guard_state,
                       "completed_epochs": durable_completed_epochs,
                       "elapsed_wall_seconds": time.perf_counter() - start_wall,
                       "elapsed_process_cpu_seconds": time.process_time() - start_cpu,
                       "GPU_hours": 0.0, "new_inner_dev_numeric_values_read": False}
            atomic_json(out / "fit_failure.json", failure, overwrite=False)
        raise
    finally:
        for sig, previous_handler in signal_handlers.items():
            signal.signal(sig, previous_handler)
    final = {
        "status": terminal_status, "run_id": run_id, "pid": os.getpid(),
        "completed_epochs": completed_epochs, "seed": SEED, "device": "cpu", "GPU_hours": 0.0,
        "elapsed_wall_seconds": time.perf_counter() - start_wall,
        "elapsed_process_cpu_seconds": time.process_time() - start_cpu,
        "latest_training_state": json.loads((out / "latest_training_state.json").read_text(encoding="utf-8")),
        "run_contract_sha256": contract_sha, "recovery_schema_version": RECOVERY_SCHEMA_VERSION,
        "resume_history": str((out / "fit_resume_history.jsonl").relative_to(ROOT))
            if (out / "fit_resume_history.jsonl").exists() else None,
        "training_curve": {"path": str((out / "XPert_train32_curve.jsonl").relative_to(ROOT)),
                           "sha256": sha256(out / "XPert_train32_curve.jsonl")},
        "checkpoints": {name: {"sha256": sha256(out / name), "bytes": (out / name).stat().st_size}
                        for name in ["XPert_epoch000_initial.pt", "XPert_epoch070_after_warmup.pt",
                                     "XPert_epoch100.pt", "XPert_epoch2500.pt"]},
        "predictions": {path.name: sha256(path) for path in out.glob("XPert_epoch*_train32_predictions.npz")},
        "train32_key_sha256": sha256(out / "train32_keys.tsv"),
        "strongest_baseline": strongest_name,
        "baseline_metrics_train32": baseline_metrics,
        "endpoint": json.loads((out / "XPert_train32_curve.jsonl").read_text(encoding="utf-8").splitlines()[-1])[
            "project_delta_metrics"],
        "CPU_attention_layout_adapter": {key: value for key, value in cpu_attention_adapter.items()
                                          if key != "runtime_counts"},
        "CPU_attention_layout_adapter_runtime_counts": cpu_attention_adapter["runtime_counts"],
        "new_inner_dev_numeric_values_read": False,
    }
    endpoint_nmse = final["endpoint"]["object_macro_nmse"]
    baseline_nmse = strongest_metrics["object_macro_nmse"]
    reduction = 1.0 - endpoint_nmse / baseline_nmse
    amplitude = final["endpoint"]["object_macro_amplitude_ratio"]
    final["fit_gate"] = {"strongest_baseline_nmse": baseline_nmse,
                         "relative_nmse_reduction": reduction,
                         "minimum_required_reduction": 0.10,
                         "amplitude_ratio": amplitude, "amplitude_range": [0.5, 2.0],
                         "passed": bool(reduction >= 0.10 and amplitude is not None and 0.5 <= amplitude <= 2.0)}
    atomic_json(out / "fit_terminal_receipt.json", final, overwrite=False)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["old-readback", "old-summary", "fit32", "verify-recovery"], required=True)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--resume", action="store_true", help="restore the last committed full training state")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.phase == "verify-recovery":
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError("recovery verification must not initialize CUDA")
        result = verify_recovery_protocol(torch)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    elif args.phase == "old-readback":
        try:
            report = old_assets_readback(args.run_id)
            summary = write_old_prediction_summary(args.run_id)
        except BaseException as exc:
            out = output_root(args.run_id)
            if out.is_dir() and not (out / "old_readback_failure.json").exists():
                atomic_json(out / "old_readback_failure.json", {
                    "status": "FAILED_CLOSED_OLD_TRAIN_READBACK",
                    "run_id": args.run_id, "error_type": type(exc).__name__, "error": str(exc),
                    "GPU_hours": 0.0, "new_inner_dev_numeric_values_read": False,
                }, overwrite=False)
            raise
        print(json.dumps({"status": report["status"], "run_id": args.run_id,
                          "output": str((output_root(args.run_id) / "old_train_readback.json").relative_to(ROOT)),
                          "EXP055_selected_train_nmse": report["EXP055_old_train"]["selected_train_metrics_recomputed"]["object_macro_nmse"],
                          "EXP055_best_train_nmse": report["EXP055_old_train"]["registered_fit_gate"]["best_train_only_epoch"]["train_object_macro_nmse"],
                          "EXP063_train_epoch1_nmse": report["EXP063_epoch001_train_only"]["train_metrics_recomputed"]["object_macro_nmse"],
                          "old_prediction_metrics": str((output_root(args.run_id) / "old_prediction_metrics.json").relative_to(ROOT)),
                          "EXP055_X_ctl_out_of_range_fraction": report["input_scale"]["train_X_ctl"]["fraction_below_bin_range"] + report["input_scale"]["train_X_ctl"]["fraction_above_bin_range"]},
                         ensure_ascii=False, indent=2), flush=True)
    elif args.phase == "old-summary":
        summary = write_old_prediction_summary(args.run_id)
        print(json.dumps({"status": summary["status"], "run_id": args.run_id,
                          "output": str((output_root(args.run_id) / "old_prediction_metrics.json").relative_to(ROOT)),
                          "selected_EXP055_TRAIN_NMSE": summary["EXP055_author_selected_checkpoint_train_only"]["metrics"]["object_macro_nmse"],
                          "selected_EXP055_relative_reduction_percent": summary["EXP055_author_selected_checkpoint_train_only"]["relative_nmse_reduction_percent"],
                          "best_TRAIN_curve_epoch": summary["EXP055_best_train_only_epoch_diagnostic"]["epoch"],
                          "best_TRAIN_curve_NMSE": summary["EXP055_best_train_only_epoch_diagnostic"]["metrics"]["train_object_macro_nmse"],
                          "EXP063_epoch1_TRAIN_NMSE": summary["EXP063_one_epoch_train_only_cost_gate"]["metrics"]["object_macro_nmse"],
                          "new_inner_dev_numeric_values_read": False}, ensure_ascii=False, indent=2), flush=True)
    else:
        try:
            if args.worker:
                receipt = fit_train32(args.run_id, resume=args.resume)
            else:
                receipt = launch_detached_fit(args.run_id, resume=args.resume)
        except BaseException as exc:
            out = output_root(args.run_id)
            interruption = None
            if (out / "fit_interrupted.json").is_file():
                try:
                    interruption = json.loads((out / "fit_interrupted.json").read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    interruption = None
            interrupted_this_process = interruption is not None and interruption.get("pid") == os.getpid()
            if (args.worker and args.resume and not interrupted_this_process
                    and not (out / "fit_failure.json").exists()):
                atomic_json(out / f"fit_resume_blocked_pid_{os.getpid()}.json", {
                    "status": "RESUME_BLOCKED_NO_MODEL_UPDATE", "run_id": args.run_id,
                    "pid": os.getpid(), "error_type": type(exc).__name__, "error": str(exc),
                    "GPU_hours": 0.0}, overwrite=False)
            elif (args.worker and not args.resume and not interrupted_this_process
                  and (out / "fit_preflight_started.json").is_file()
                  and not (out / "fit_failure.json").exists()
                  and not (out / "fit_terminal_receipt.json").exists()):
                pointer_path = out / "latest_training_state.json"
                if pointer_path.is_file():
                    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                    safe_error = {"status": "WORKER_STOPPED_AT_DURABLE_FULL_STATE",
                                  "run_id": args.run_id, "pid": os.getpid(), "resumable": True,
                                  "completed_epochs": pointer.get("completed_epochs"),
                                  "latest_state_sha256": pointer.get("sha256"),
                                  "error_type": type(exc).__name__, "error": str(exc)}
                    atomic_json(out / "fit_interrupted.json", safe_error)
                else:
                    failure_kind = "NONFINITE_OR_INVALID_NUMERIC_OUTPUT" if isinstance(exc, FloatingPointError) else "IMPLEMENTATION_EXCEPTION"
                    atomic_json(out / "fit_failure.json", {
                        "status": f"FAILED_CLOSED_{failure_kind}", "failure_kind": failure_kind,
                        "run_id": args.run_id, "pid": os.getpid(), "error_type": type(exc).__name__,
                        "error": str(exc), "GPU_hours": 0.0, "new_inner_dev_numeric_values_read": False,
                        "started_at_utc": json.loads((out / "fit_preflight_started.json").read_text(encoding="utf-8"))["started_at_utc"],
                        "last_finite_checkpoint_available": (out / "XPert_epoch000_initial.pt").is_file(),
                    }, overwrite=False)
            raise
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
