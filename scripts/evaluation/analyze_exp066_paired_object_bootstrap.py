#!/usr/bin/env python
"""Add the preregistered paired PatientID bootstrap to frozen EXP-066 OOF outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


ARMS = (
    "zero_change",
    "per_drug_training_mean",
    "rank16_rrr",
    "no_state_equal_capacity_rank16_rrr",
    "fold_local_object_state_shuffle_rank16_rrr",
)
COMPARATORS = (
    "per_drug_training_mean",
    "no_state_equal_capacity_rank16_rrr",
    "fold_local_object_state_shuffle_rank16_rrr",
    "zero_change",
)
SEED = 660067
ITERATIONS = 2000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_macro(values: np.ndarray, objects: np.ndarray) -> dict[str, float]:
    frame = pd.DataFrame({"object": objects.astype(str), "value": np.asarray(values, dtype=np.float64)})
    if frame["object"].isna().any() or not np.isfinite(frame["value"]).all():
        raise ValueError("object aggregation requires keyed objects and finite condition metrics")
    return frame.groupby("object", sort=True)["value"].mean().astype(float).to_dict()


def _paired_bootstrap(reference: np.ndarray, candidate: np.ndarray,
                      draws: np.ndarray) -> dict[str, object]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.ndim != 1 or reference.shape != candidate.shape or draws.ndim != 2:
        raise ValueError("paired object values or bootstrap draw matrix have inconsistent shapes")
    if draws.shape[1] != len(reference) or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("paired bootstrap requires finite values on one common object denominator")
    sampled_reference = reference[draws]
    sampled_candidate = candidate[draws]
    absolute = (sampled_reference - sampled_candidate).mean(axis=1)
    denominators = sampled_reference.mean(axis=1)
    relative = absolute / denominators
    mean_absolute = float(reference.mean() - candidate.mean())
    mean_relative = float(mean_absolute / reference.mean()) if reference.mean() > 0 else None
    return {
        "objects": int(len(reference)),
        "reference_object_macro_nmse": float(reference.mean()),
        "candidate_object_macro_nmse": float(candidate.mean()),
        "mean_absolute_nmse_reduction_reference_minus_candidate": mean_absolute,
        "ci95_absolute_nmse_reduction": [float(np.quantile(absolute, .025)), float(np.quantile(absolute, .975))],
        "mean_relative_nmse_reduction": mean_relative,
        "ci95_relative_nmse_reduction": [float(np.quantile(relative, .025)), float(np.quantile(relative, .975))],
    }


def analyze(run_dir: Path) -> dict[str, object]:
    run_dir = run_dir.resolve()
    summary_path = run_dir / "run_summary.json"
    run_readback_path = run_dir / "run_readback.json"
    root_readback_path = run_dir / "root_independent_readback.json"
    if not all(path.is_file() for path in (summary_path, run_readback_path, root_readback_path)):
        raise FileNotFoundError("A summary, runner receipt, and root PASS receipt are required")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runner_receipt = json.loads(run_readback_path.read_text(encoding="utf-8"))
    root_receipt = json.loads(root_readback_path.read_text(encoding="utf-8"))
    if (summary.get("experiment_id") != "EXP-066" or summary.get("status") != "COMPLETE"
            or int(summary.get("fold_count_completed", -1)) != 5
            or runner_receipt.get("run_summary_sha256") != _sha256(summary_path)
            or root_receipt.get("status") != "PASS_EXP066_READBACK"
            or root_receipt.get("run_summary_sha256") != _sha256(summary_path)):
        raise ValueError("EXP-066 fixed five-fold run has not passed both readback gates")

    condition_ids: list[str] = []
    objects: list[str] = []
    fold_ids: list[int] = []
    condition_values: dict[str, dict[str, list[float]]] = {arm: {"": []} for arm in ARMS}
    condition_values = {arm: {"nmse": []} for arm in ARMS}
    condition_objects: list[str] = []
    condition_fold_ids: list[int] = []
    prediction_hashes: dict[str, str] = {}
    for fold in range(5):
        info = summary.get("files", {}).get(f"fold_{fold}_predictions")
        if not isinstance(info, Mapping):
            raise ValueError(f"run summary lacks fold {fold} prediction provenance")
        prediction_path = Path(str(info["path"]))
        if not prediction_path.is_absolute():
            prediction_path = run_dir / prediction_path
        if not prediction_path.is_file() or _sha256(prediction_path) != info.get("sha256"):
            raise ValueError(f"fold {fold} prediction file is absent or differs from the summary checksum")
        prediction_hashes[f"fold_{fold}"] = _sha256(prediction_path)
        with np.load(prediction_path, allow_pickle=False) as payload:
            ids = payload["condition_id"].astype(str)
            object_ids = payload["source_object_id"].astype(str)
            assigned_fold = payload["fold_id"].astype(int)
            y_true = payload["y_true"].astype(np.float64)
            train_variance = payload["train_gene_variance"].astype(np.float64)
            if (y_true.shape != (len(ids), 978) or len(object_ids) != len(ids)
                    or not np.all(assigned_fold == fold) or not np.isfinite(y_true).all()
                    or train_variance.shape != (978,) or not np.isfinite(train_variance).all()):
                raise ValueError(f"fold {fold} has invalid keys, labels, or TRAIN variance")
            valid_genes = train_variance > float(summary["variance_epsilon"])
            if not valid_genes.any():
                raise ValueError(f"fold {fold} has no evaluable TRAIN-variance genes")
            for arm in ARMS:
                pred = payload[f"pred__{arm}"].astype(np.float64)
                if pred.shape != y_true.shape or not np.isfinite(pred).all():
                    raise ValueError(f"fold {fold}/{arm} prediction is invalid")
                err = pred - y_true
                row_nmse = (np.square(err[:, valid_genes]) / train_variance[valid_genes]).mean(axis=1)
                condition_values[arm]["nmse"].extend(row_nmse.tolist())
            condition_ids.extend(ids.tolist())
            condition_objects.extend(object_ids.tolist())
            condition_fold_ids.extend(assigned_fold.tolist())

    if len(set(condition_ids)) != len(condition_ids) or len(condition_ids) != int(summary["condition_count"]):
        raise ValueError("OOF conditions are duplicated or differ from the frozen denominator")
    frame = pd.DataFrame({
        "condition_id": condition_ids,
        "source_object_id": condition_objects,
        "fold_id": condition_fold_ids,
    })
    if frame.groupby("source_object_id")["fold_id"].nunique().max() != 1:
        raise ValueError("a PatientID appears in more than one held-out fold")
    object_metrics: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        object_metrics[arm] = _object_macro(
            np.asarray(condition_values[arm]["nmse"], dtype=np.float64),
            frame["source_object_id"].to_numpy(dtype=str),
        )
        recorded = summary["out_of_fold_metrics"][arm]["object_macro"]["normalized_mse"]["object_macro_mean"]
        if not np.isclose(np.mean(list(object_metrics[arm].values())), float(recorded), rtol=1e-9, atol=1e-10):
            raise ValueError(f"root OOF object NMSE differs from frozen summary for {arm}")
    object_ids_sorted = sorted(object_metrics["rank16_rrr"])
    if any(set(object_metrics[arm]) != set(object_ids_sorted) for arm in ARMS):
        raise ValueError("all A comparison arms must share the same PatientID denominator")
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(object_ids_sorted), size=(ITERATIONS, len(object_ids_sorted)))
    intervals = {}
    candidate = np.asarray([object_metrics["rank16_rrr"][obj] for obj in object_ids_sorted])
    for comparator in COMPARATORS:
        reference = np.asarray([object_metrics[comparator][obj] for obj in object_ids_sorted])
        intervals[f"rank16_rrr_vs_{comparator}"] = _paired_bootstrap(reference, candidate, draws)

    metrics_rows = []
    for object_id in object_ids_sorted:
        row: dict[str, object] = {"source_object_id": object_id}
        for arm in ARMS:
            row[f"{arm}__nmse"] = object_metrics[arm][object_id]
        metrics_rows.append(row)
    metrics_path = run_dir / "root_paired_object_nmse_v1.tsv"
    receipt_path = run_dir / "root_paired_object_bootstrap_v1.json"
    if metrics_path.exists() or receipt_path.exists():
        raise FileExistsError("Refusing to overwrite an existing EXP-066 paired-analysis supplement")
    pd.DataFrame(metrics_rows).to_csv(metrics_path, sep="\t", index=False, lineterminator="\n")
    receipt = {
        "schema": "exp066_paired_object_bootstrap_v1",
        "status": "PASS_ROOT_PAIRED_OBJECT_ANALYSIS",
        "run_id": summary["run_id"],
        "run_summary_sha256": _sha256(summary_path),
        "root_readback_sha256": _sha256(root_readback_path),
        "prediction_artifact_sha256": prediction_hashes,
        "condition_count": len(condition_ids),
        "object_count": len(object_ids_sorted),
        "target_gene_count": 978,
        "normalized_metric": "condition NMSE divides squared error by each held-out fold's TRAIN gene variance; mean genes, then conditions within PatientID, then equal PatientID weight",
        "bootstrap": {
            "seed": SEED,
            "iterations": ITERATIONS,
            "sampling_unit": "PatientID/source object",
            "interval": "paired percentile 95 percent bootstrap; descriptive only for the 34 fixed source objects",
        },
        "paired_intervals": intervals,
        "object_metrics_path": str(metrics_path.relative_to(run_dir)),
        "object_metrics_sha256": _sha256(metrics_path),
        "source_values_read": ["frozen EXP-066 five-fold TRAIN/OOF keyed prediction artifacts only"],
        "forbidden_numeric_roles_read": [],
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
