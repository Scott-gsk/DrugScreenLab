"""Execute frozen EXP-002 baselines directly from immutable Level-3 GCTX.

This runner deliberately keeps the complete exact-978 matrix in memory rather
than writing a multi-gigabyte JSONL materialization.  It follows the v2 data
contract and emits only a compact, reproducible evaluation artifact.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data"))

import audit_exp002_strict_split as audit  # noqa: E402
from drug_screen.data import exp002  # noqa: E402
from drug_screen.evaluation.protocol import compute_vector_metrics  # noqa: E402
from drug_screen.modeling.exp002 import (  # noqa: E402
    GENE_COUNT,
    METRICS,
    ORDERED_GENE_IDS_SHA256,
    load_exp002_config,
)


def _key(frame: pd.DataFrame) -> pd.Series:
    return frame.loc[:, audit.MATCH_KEY].astype(str).agg("\x1f".join, axis=1)


def _bootstrap(values: np.ndarray, *, seed: int, resamples: int) -> dict[str, float | int]:
    """Vectorized non-parametric bootstrap over already independent groups."""
    generator = np.random.default_rng(seed)
    count = len(values)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 64):
        stop = min(start + 64, resamples)
        draw = generator.integers(0, count, size=(stop - start, count))
        means[start:stop] = values[draw].mean(axis=1)
    means.sort()
    return {
        "point_estimate": float(values.mean()),
        "low": float(means[round(0.025 * (resamples - 1))]),
        "high": float(means[round(0.975 * (resamples - 1))]),
        "seed": seed,
        "resamples": resamples,
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    groups: np.ndarray,
    *,
    higher_is_better: bool,
    seed: int,
    resamples: int,
) -> dict[str, float | int]:
    unique, inverse = np.unique(groups, return_inverse=True)
    group_values = np.empty(len(unique), dtype=np.float64)
    signed = candidate - baseline if higher_is_better else baseline - candidate
    for index in range(len(unique)):
        group_values[index] = signed[inverse == index].mean()
    return _bootstrap(group_values, seed=seed, resamples=resamples)


def _metrics(
    observed: np.ndarray, predicted: np.ndarray, gene_ids: list[str]
) -> dict[str, float | None]:
    try:
        metric = compute_vector_metrics(
            observed.tolist(),
            predicted.tolist(),
            observed_gene_ids=gene_ids,
            predicted_gene_ids=gene_ids,
            expected_gene_ids=gene_ids,
        )
        return {name: float(getattr(metric, name)) for name in METRICS}
    except ValueError as error:
        if not any(
            message in str(error)
            for message in ("correlation undefined", "direction accuracy has no eligible")
        ):
            raise
        residual = observed - predicted
        eligible = observed != 0.0
        return {
            "pearson": None,
            "spearman": None,
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "mae": float(np.mean(np.abs(residual))),
            "direction_accuracy": float(
                np.mean((observed[eligible] > 0) == (predicted[eligible] > 0))
            )
            if eligible.any()
            else None,
        }


def _read_exact978(
    hdf5: Path,
    landmark_indices: np.ndarray,
    cache_path: Path,
    cache_metadata_path: Path,
    expected_cache_metadata: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    if cache_path.is_file() and cache_metadata_path.is_file():
        metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
        cached = np.load(cache_path, mmap_mode="r")
        if (
            cached.shape == tuple(expected_cache_metadata["cache_shape"])
            and cached.dtype == np.float32
            and all(metadata.get(key) == value for key, value in expected_cache_metadata.items())
            and metadata.get("cache_sha256") == _file_sha256(cache_path)
        ):
            print(f"reuse exact-978 cache: {cache_path}", flush=True)
            return cached, metadata
    with h5py.File(hdf5, "r") as handle:
        matrix = handle["0/DATA/0/matrix"]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result = np.lib.format.open_memmap(
            cache_path, mode="w+", dtype=np.float32, shape=(matrix.shape[0], GENE_COUNT)
        )
        for start in range(0, matrix.shape[0], 50_000):
            stop = min(start + 50_000, matrix.shape[0])
            result[start:stop] = matrix[start:stop, :][:, landmark_indices]
            print(f"read exact-978 rows {start}:{stop}", flush=True)
        result.flush()
    metadata = {
        **expected_cache_metadata,
        "cache_sha256": _file_sha256(cache_path),
        "cache_bytes": cache_path.stat().st_size,
    }
    cache_metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result, metadata


def _prepare(
    root: Path, cache_path: Path, contract_path: Path
) -> tuple[
    pd.DataFrame, pd.DataFrame, list[str], np.ndarray, dict[str, object], dict[str, object]
]:
    raw = root / "raw" / "lincs" / "GSE92742"
    interim = root / "interim" / "lincs" / "GSE92742"
    instances = pd.read_csv(raw / audit.METADATA_FILES[3], sep="\t", low_memory=False)
    perturbagens = pd.read_csv(raw / audit.METADATA_FILES[1], sep="\t", usecols=["pert_id", "inchi_key"])
    cells = pd.read_csv(raw / audit.METADATA_FILES[2], sep="\t", usecols=["cell_id", "base_cell_id"])
    genes = pd.read_csv(raw / audit.METADATA_FILES[0], sep="\t", low_memory=False)
    treatments = audit._prepare_treatments(instances, perturbagens, cells)
    vehicles = instances.loc[instances["pert_type"].eq("ctl_vehicle"), list(audit.MATCH_KEY) + ["inst_id"]].copy()
    matched = treatments.merge(
        vehicles.loc[:, audit.MATCH_KEY].drop_duplicates(), on=list(audit.MATCH_KEY), how="inner", validate="many_to_one"
    ).reset_index(drop=True)
    hdf5 = interim / audit.LEVEL3_HDF5
    with h5py.File(hdf5, "r") as handle:
        row_ids = audit._decode(handle["0/META/ROW/id"][:])
        column_ids = audit._decode(handle["0/META/COL/id"][:])
    flags = dict(zip(genes["pr_gene_id"].astype(str), genes["pr_is_lm"].astype(int)))
    landmark_indices = np.asarray([index for index, gene in enumerate(row_ids) if flags[gene] == 1], dtype=np.int64)
    gene_ids = [row_ids[index] for index in landmark_indices]
    if len(gene_ids) != GENE_COUNT or audit._stream_digest(gene_ids) != ORDERED_GENE_IDS_SHA256:
        raise ValueError("frozen exact-978 order mismatch")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if audit._stream_digest(row_ids) != contract["gene_universe"]["all_gctx_gene_ids_sha256"]:
        raise ValueError("actual HDF5 row identity digest differs from frozen contract")
    if audit._stream_digest(column_ids) != contract["gene_universe"]["gctx_instance_order_sha256"]:
        raise ValueError("actual HDF5 column identity digest differs from frozen contract")
    runtime_manifests = {}
    for mode, entity_column in (("cold_drug", "drug_id"), ("cold_context", "context_id")):
        observed = audit._audit_mode(matched, vehicles, entity_column, mode)
        expected = contract["split"][mode]["manifest_digests"]
        if observed["manifest_digests"] != expected:
            raise ValueError(f"{mode} runtime split manifest digest differs from frozen contract")
        runtime_manifests[mode] = observed["manifest_digests"]
    index_by_instance = {value: index for index, value in enumerate(column_ids)}
    matched["_row"] = matched["inst_id"].astype(str).map(index_by_instance)
    vehicles["_row"] = vehicles["inst_id"].astype(str).map(index_by_instance)
    if matched["_row"].isna().any() or vehicles["_row"].isna().any():
        raise ValueError("GCTX instance identity mismatch")
    expected_cache_metadata = {
        "cache_format": "npy_float32_exact978_v1",
        "source_contract_sha256": _file_sha256(contract_path),
        "source_level3_sha512": contract["provenance"]["level3_compressed_sha512"],
        "ordered_gene_ids_sha256": ORDERED_GENE_IDS_SHA256,
        "source_matrix_shape": [int(len(column_ids)), int(len(row_ids))],
        "cache_shape": [int(len(column_ids)), GENE_COUNT],
        "runner_sha256": _file_sha256(Path(__file__).resolve()),
    }
    values, cache_metadata = _read_exact978(
        hdf5,
        landmark_indices,
        cache_path,
        cache_path.with_suffix(".metadata.json"),
        expected_cache_metadata,
    )
    runtime_verification = {
        "actual_hdf5_path": str(hdf5),
        "actual_hdf5_row_ids_sha256": audit._stream_digest(row_ids),
        "actual_hdf5_column_ids_sha256": audit._stream_digest(column_ids),
        "runtime_split_manifest_digests": runtime_manifests,
    }
    return matched, vehicles, gene_ids, values, cache_metadata, runtime_verification


def _evaluate_mode(
    matched: pd.DataFrame,
    vehicles: pd.DataFrame,
    values: np.ndarray,
    gene_ids: list[str],
    mode: str,
    config: dict[str, object],
) -> dict[str, object]:
    entity_column = "drug_id" if mode == "cold_drug" else "context_id"
    frame = matched.copy()
    frame["split"] = frame[entity_column].map(lambda entity: exp002.deterministic_split(f"{mode}:{entity}"))
    frame["_key"] = _key(frame)
    vehicle_frame = vehicles.copy()
    vehicle_frame["_key"] = _key(vehicle_frame)
    controls_by_key = {key: group["inst_id"].astype(str).tolist() for key, group in vehicle_frame.groupby("_key", sort=False)}
    control_rows = dict(zip(vehicle_frame["inst_id"].astype(str), vehicle_frame["_row"].astype(int)))
    means: dict[tuple[str, str], np.ndarray] = {}
    for key, group in frame.groupby("_key", sort=False):
        active = group["split"].unique().tolist()
        allocation = exp002.deterministic_vehicle_partition(key.split("\x1f"), controls_by_key[key], active)
        for split in active:
            rows = [control_rows[control] for control, owner in allocation.items() if owner == split]
            means[(key, split)] = values[rows].mean(axis=0, dtype=np.float64).astype(np.float32)
    deltas = values[frame["_row"].to_numpy(dtype=np.int64)].copy()
    for (key, split), positions in frame.groupby(["_key", "split"], sort=False).indices.items():
        deltas[positions] -= means[(key, split)]
    train_positions = np.flatnonzero(frame["split"].eq("train").to_numpy())
    test_positions = np.flatnonzero(frame["split"].eq("test").to_numpy())
    global_mean = deltas[train_positions].mean(axis=0, dtype=np.float64).astype(np.float32)
    context_mean: dict[str, np.ndarray] = {}
    for context, positions in frame.iloc[train_positions].groupby("context_id", sort=False).indices.items():
        context_mean[context] = deltas[train_positions[np.fromiter(positions, dtype=np.int64)]].mean(axis=0, dtype=np.float64).astype(np.float32)
    group_rows: list[dict[str, object]] = []
    test_frame = frame.iloc[test_positions].reset_index(drop=True)
    test_deltas = deltas[test_positions]
    for (drug, context), positions in test_frame.groupby(["drug_id", "context_id"], sort=False).indices.items():
        observed = test_deltas[np.fromiter(positions, dtype=np.int64)].mean(axis=0)
        baseline = global_mean
        candidate = context_mean.get(context, global_mean)
        group_rows.append({
            "drug_id": drug, "context_id": context, "fallback": context not in context_mean,
            "global_train_mean": _metrics(observed, baseline, gene_ids),
            "context_train_mean": _metrics(observed, candidate, gene_ids),
        })
    resamples = int(config["evaluation"]["bootstrap_resamples"])
    summaries = []
    for seed in config["reproducibility"]["seeds"]:
        for metric in METRICS:
            candidate = np.asarray([row["context_train_mean"][metric] for row in group_rows if row["context_train_mean"][metric] is not None], dtype=np.float64)
            baseline = np.asarray([row["global_train_mean"][metric] for row in group_rows if row["global_train_mean"][metric] is not None], dtype=np.float64)
            paired = [row for row in group_rows if row["context_train_mean"][metric] is not None and row["global_train_mean"][metric] is not None]
            group_axis = np.asarray([row["drug_id"] if mode == "cold_drug" else row["context_id"] for row in paired])
            paired_candidate = np.asarray([row["context_train_mean"][metric] for row in paired], dtype=np.float64)
            paired_baseline = np.asarray([row["global_train_mean"][metric] for row in paired], dtype=np.float64)
            summaries.append({
                "seed": seed, "metric": metric,
                "global_train_mean": _bootstrap(baseline, seed=int(config["evaluation"]["bootstrap_seed"]) + seed, resamples=resamples),
                "context_train_mean": _bootstrap(candidate, seed=int(config["evaluation"]["bootstrap_seed"]) + seed, resamples=resamples),
                "context_minus_global": _group_bootstrap(paired_candidate, paired_baseline, group_axis, higher_is_better=metric in {"pearson", "spearman", "direction_accuracy"}, seed=int(config["evaluation"]["bootstrap_seed"]) + seed, resamples=resamples),
            })
    return {
        "eligible_treatment_count": int(len(test_positions)),
        "eligible_group_count": len(group_rows),
        "context_fallback_count": int(sum(row["fallback"] for row in group_rows)),
        "summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    config = load_exp002_config(args.config)
    matched, vehicles, gene_ids, values, cache_metadata, runtime_verification = _prepare(
        args.root, args.cache, args.contract
    )
    result = {
        "experiment_id": "EXP-002", "status": "EVALUATION_COMPLETE",
        "execution": "full Level-3 exact-978 in-memory streaming; no raw data modified",
        "config_sha256": sha256(args.config.read_bytes()).hexdigest(),
        "provenance": {
            "runner_sha256": _file_sha256(Path(__file__).resolve()),
            "source_contract_path": str(args.contract),
            "source_contract_sha256": _file_sha256(args.contract),
            "cache_path": str(args.cache),
            "cache_metadata": cache_metadata,
            "runtime_verification": runtime_verification,
            "command": [
                "scripts/modeling/run_exp002_streaming.py", "--root", str(args.root),
                "--config", str(args.config), "--cache", str(args.cache),
                "--contract", str(args.contract), "--output", str(args.output),
            ],
            "python": sys.version,
            "platform": platform.platform(),
        },
        "results": {mode: _evaluate_mode(matched, vehicles, values, gene_ids, mode, config) for mode in ("cold_drug", "cold_context")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
