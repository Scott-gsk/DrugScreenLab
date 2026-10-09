"""Independent readback verifier for EXP-066 keyed response predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ARMS = (
    "zero_change",
    "per_drug_training_mean",
    "rank16_rrr",
    "no_state_equal_capacity_rank16_rrr",
    "fold_local_object_state_shuffle_rank16_rrr",
)
METRICS = (
    "normalized_mse", "raw_mse", "spearman", "directional_agreement",
    "nonzero_directional_agreement",
)
EPSILON = 1e-8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric_arrays(y_true: np.ndarray, y_pred: np.ndarray,
                   train_variance: np.ndarray) -> dict[str, np.ndarray]:
    if y_true.shape != y_pred.shape or y_true.ndim != 2 or y_true.shape[1] != 978:
        raise ValueError("prediction artifact has an invalid response matrix shape")
    if train_variance.shape != (978,) or not np.isfinite(train_variance).all() or (train_variance < 0).any():
        raise ValueError("prediction artifact has an invalid fold TRAIN variance vector")
    valid_genes = train_variance > EPSILON
    valid_prediction = np.isfinite(y_pred).all(axis=1)
    n = len(y_true)
    normalized = np.full(n, np.nan, dtype=np.float64)
    raw = np.full(n, np.nan, dtype=np.float64)
    spearman = np.full(n, np.nan, dtype=np.float64)
    direction = np.full(n, np.nan, dtype=np.float64)
    nonzero_direction = np.full(n, np.nan, dtype=np.float64)
    if valid_prediction.any():
        error = y_pred[valid_prediction] - y_true[valid_prediction]
        raw[valid_prediction] = np.square(error).mean(axis=1)
        if valid_genes.any():
            normalized[valid_prediction] = (
                np.square(error[:, valid_genes]) / train_variance[valid_genes]
            ).mean(axis=1)
    for row in np.flatnonzero(valid_prediction):
        if np.ptp(y_true[row]) > 0 and np.ptp(y_pred[row]) > 0:
            spearman[row] = float(spearmanr(y_true[row], y_pred[row]).statistic)
        direction[row] = float(np.mean(np.sign(y_true[row]) == np.sign(y_pred[row])))
        nonzero = (y_true[row] != 0) & (y_pred[row] != 0)
        if nonzero.any():
            nonzero_direction[row] = float(
                np.mean(np.sign(y_true[row, nonzero]) == np.sign(y_pred[row, nonzero]))
            )
    return {
        "normalized_mse": normalized,
        "raw_mse": raw,
        "spearman": spearman,
        "directional_agreement": direction,
        "nonzero_directional_agreement": nonzero_direction,
        "valid_target_gene_count": np.full(n, int(valid_genes.sum()), dtype=np.int64),
        "excluded_target_gene_count": np.full(n, int((~valid_genes).sum()), dtype=np.int64),
    }


def _object_macro(values: np.ndarray, objects: np.ndarray) -> dict[str, Any]:
    frame = pd.DataFrame({"source_object_id": objects.astype(str), "value": values})
    finite = frame.loc[np.isfinite(frame["value"])]
    total_objects = int(frame["source_object_id"].nunique())
    if finite.empty:
        return {
            "object_macro_mean": None,
            "total_objects": total_objects,
            "valid_objects": 0,
            "undefined_objects": total_objects,
            "valid_conditions": 0,
            "undefined_conditions": int(len(frame)),
        }
    per_object = finite.groupby("source_object_id", sort=True)["value"].mean()
    return {
        "object_macro_mean": float(per_object.mean()),
        "total_objects": total_objects,
        "valid_objects": int(len(per_object)),
        "undefined_objects": int(total_objects - len(per_object)),
        "valid_conditions": int(len(finite)),
        "undefined_conditions": int(len(frame) - len(finite)),
    }


def _aggregate(metrics_by_arm: Mapping[str, Mapping[str, np.ndarray]],
               objects: np.ndarray, keys: np.ndarray,
               fold_local: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm, metrics in metrics_by_arm.items():
        per_drug: dict[str, Any] = {}
        frame = pd.DataFrame({"source_object_id": objects.astype(str), "full_inchikey": keys.astype(str)})
        for metric in METRICS:
            frame[metric] = metrics[metric]
        for key, group in frame.groupby("full_inchikey", sort=True):
            total_objects = int(group["source_object_id"].nunique())
            drug_summary: dict[str, Any] = {"conditions": int(len(group)), "objects": total_objects}
            for metric in METRICS:
                finite = group.loc[np.isfinite(group[metric])]
                by_object = finite.groupby("source_object_id", sort=True)[metric].mean()
                drug_summary[metric] = {
                    "object_macro_mean": float(by_object.mean()) if len(by_object) else None,
                    "total_objects": total_objects,
                    "valid_objects": int(len(by_object)),
                    "undefined_objects": int(total_objects - len(by_object)),
                    "valid_conditions": int(len(finite)),
                    "undefined_conditions": int(len(group) - len(finite)),
                }
            per_drug[str(key)] = drug_summary
        if fold_local:
            denominators = {
                "conditions": int(len(objects)),
                "objects": int(pd.Series(objects.astype(str)).nunique()),
                "prediction_valid_conditions": int(np.isfinite(metrics["raw_mse"]).sum()),
                "prediction_undefined_conditions": int(np.isnan(metrics["raw_mse"]).sum()),
                "spearman_valid_conditions": int(np.isfinite(metrics["spearman"]).sum()),
                "spearman_undefined_conditions": int(np.isnan(metrics["spearman"]).sum()),
                "direction_valid_conditions": int(np.isfinite(metrics["directional_agreement"]).sum()),
                "direction_undefined_conditions": int(np.isnan(metrics["directional_agreement"]).sum()),
                "nonzero_direction_valid_conditions": int(np.isfinite(metrics["nonzero_directional_agreement"]).sum()),
                "nonzero_direction_undefined_conditions": int(np.isnan(metrics["nonzero_directional_agreement"]).sum()),
                "valid_target_genes_per_condition": int(metrics["valid_target_gene_count"][0]),
                "excluded_near_zero_variance_genes": int(metrics["excluded_target_gene_count"][0]),
            }
        else:
            denominators = {
                "conditions": int(len(objects)),
                "objects": int(pd.Series(objects.astype(str)).nunique()),
                "prediction_valid_conditions": int(np.isfinite(metrics["raw_mse"]).sum()),
                "prediction_undefined_conditions": int(np.isnan(metrics["raw_mse"]).sum()),
                "spearman_valid_conditions": int(np.isfinite(metrics["spearman"]).sum()),
                "spearman_undefined_conditions": int(np.isnan(metrics["spearman"]).sum()),
                "direction_valid_conditions": int(np.isfinite(metrics["directional_agreement"]).sum()),
                "direction_undefined_conditions": int(np.isnan(metrics["directional_agreement"]).sum()),
                "nonzero_direction_valid_conditions": int(np.isfinite(metrics["nonzero_directional_agreement"]).sum()),
                "nonzero_direction_undefined_conditions": int(np.isnan(metrics["nonzero_directional_agreement"]).sum()),
                "valid_target_genes_per_condition_min": int(np.min(metrics["valid_target_gene_count"])),
                "valid_target_genes_per_condition_max": int(np.max(metrics["valid_target_gene_count"])),
                "valid_target_gene_condition_pairs": int(np.sum(metrics["valid_target_gene_count"])),
                "excluded_near_zero_variance_genes_per_condition_min": int(np.min(metrics["excluded_target_gene_count"])),
                "excluded_near_zero_variance_genes_per_condition_max": int(np.max(metrics["excluded_target_gene_count"])),
                "excluded_gene_condition_pairs": int(np.sum(metrics["excluded_target_gene_count"])),
            }
        result[arm] = {
            "object_macro": {metric: _object_macro(metrics[metric], objects) for metric in METRICS},
            "per_drug": per_drug,
            "denominators": denominators,
        }
    return result


def _assert_close(actual: Any, expected: Any, path: str = "root") -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"summary structure mismatch at {path}")
        for key in expected:
            _assert_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"summary list mismatch at {path}")
        for index, (a, e) in enumerate(zip(actual, expected)):
            _assert_close(a, e, f"{path}[{index}]")
        return
    if expected is None or isinstance(expected, str) or isinstance(expected, bool):
        if actual != expected:
            raise ValueError(f"summary value mismatch at {path}: {actual!r} != {expected!r}")
        return
    if isinstance(expected, (int, np.integer)) and not isinstance(expected, bool):
        if actual != int(expected):
            raise ValueError(f"summary integer mismatch at {path}: {actual!r} != {int(expected)!r}")
        return
    if not np.isclose(float(actual), float(expected), rtol=1e-9, atol=1e-10, equal_nan=True):
        raise ValueError(f"summary number mismatch at {path}: {actual!r} != {expected!r}")


def verify_run(run_dir: str | Path) -> dict[str, Any]:
    """Verify stored keyed predictions and recompute heldout metrics."""
    run = Path(run_dir)
    summary_path = run / "run_summary.json"
    readback_path = run / "run_readback.json"
    if not summary_path.is_file() or not readback_path.is_file():
        raise FileNotFoundError("EXP-066 run summary and readback receipt are required")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    readback = json.loads(readback_path.read_text(encoding="utf-8"))
    if summary.get("experiment_id") != "EXP-066" or summary.get("arms") != list(ARMS):
        raise ValueError("run summary does not match the frozen EXP-066 arm set")
    if readback.get("run_id") != summary.get("run_id"):
        raise ValueError("run readback receipt belongs to another run")
    if readback.get("run_summary_sha256") != _sha256(summary_path):
        raise ValueError("run summary checksum differs from its readback receipt")

    fold_count = int(summary.get("fold_count_completed", -1))
    if fold_count < 1 or fold_count > 5 or fold_count != len(summary.get("folds", [])):
        raise ValueError("run summary has an invalid completed fold count")
    fold_details: list[dict[str, Mapping[str, np.ndarray]]] = []
    all_objects: list[str] = []
    all_keys: list[str] = []
    all_condition_ids: list[str] = []
    all_metrics: dict[str, dict[str, list[np.ndarray]]] = {
        arm: {metric: [] for metric in (*METRICS, "valid_target_gene_count", "excluded_target_gene_count")}
        for arm in ARMS
    }
    fold_file_hashes: dict[str, str] = {}
    target_axis: np.ndarray | None = None
    target_axis_hash: str | None = None

    for fold in range(fold_count):
        key = f"fold_{fold}_predictions"
        file_info = summary.get("files", {}).get(key)
        if not isinstance(file_info, Mapping):
            raise ValueError(f"run summary lacks prediction file receipt for fold {fold}")
        path = Path(str(file_info["path"]))
        if not path.is_absolute():
            path = run / path
        if not path.is_file() or _sha256(path) != file_info.get("sha256"):
            raise ValueError(f"fold {fold} prediction artifact is absent or has a checksum mismatch")
        fold_file_hashes[key] = str(file_info["sha256"])
        with np.load(path, allow_pickle=False) as payload:
            condition_ids = payload["condition_id"].astype(str)
            objects = payload["source_object_id"].astype(str)
            keys = payload["full_inchikey"].astype(str)
            fold_ids = payload["fold_id"].astype(int)
            y_true = payload["y_true"].astype(np.float64)
            variance = payload["train_gene_variance"].astype(np.float64)
            genes = payload["target_gene_ids"].astype(str)
            axes = payload["target_axis_hash"].astype(str)
            if (len(set(condition_ids.tolist())) != len(condition_ids)
                    or not np.all(fold_ids == fold)
                    or len(condition_ids) != len(objects)
                    or len(condition_ids) != len(keys)
                    or y_true.shape != (len(condition_ids), 978)
                    or not np.isfinite(y_true).all()):
                raise ValueError(f"fold {fold} prediction artifact has invalid condition keys or response shape")
            if len(genes) != 978 or len(set(genes.tolist())) != 978:
                raise ValueError(f"fold {fold} prediction artifact has a malformed 978-gene axis")
            if len(set(axes.tolist())) != 1:
                raise ValueError(f"fold {fold} artifact mixes target axis hashes")
            if target_axis is None:
                target_axis = genes
                target_axis_hash = str(axes[0])
            elif not np.array_equal(target_axis, genes) or target_axis_hash != str(axes[0]):
                raise ValueError("target-gene axis changed across folds")

            metrics_for_fold: dict[str, Mapping[str, np.ndarray]] = {}
            for arm in ARMS:
                prediction = payload[f"pred__{arm}"].astype(np.float64)
                valid = payload[f"prediction_valid__{arm}"].astype(bool)
                if prediction.shape != y_true.shape or not np.array_equal(valid, np.isfinite(prediction).all(axis=1)):
                    raise ValueError(f"fold {fold} has invalid prediction shape/mask for {arm}")
                expected_metrics = _metric_arrays(y_true, prediction, variance)
                for metric, values in expected_metrics.items():
                    stored = payload[f"metric__{arm}__{metric}"]
                    if not np.array_equal(stored, values, equal_nan=True):
                        raise ValueError(f"fold {fold} stored metric differs after recomputation: {arm}/{metric}")
                    all_metrics[arm][metric].append(values)
                metrics_for_fold[arm] = expected_metrics
            expected_summary = _aggregate(metrics_for_fold, objects, keys, fold_local=True)
            _assert_close(summary["folds"][fold]["heldout_metrics"], expected_summary, f"folds[{fold}].heldout_metrics")
            fold_details.append(metrics_for_fold)
            all_objects.extend(objects.tolist())
            all_keys.extend(keys.tolist())
            all_condition_ids.extend(condition_ids.tolist())

    if len(set(all_condition_ids)) != len(all_condition_ids):
        raise ValueError("one condition_id appears in more than one heldout fold")
    if len(fold_file_hashes) != int(readback.get("prediction_artifact_count", -1)):
        raise ValueError("run readback artifact count differs from verified prediction files")
    if readback.get("prediction_artifact_sha256") != fold_file_hashes:
        raise ValueError("run readback prediction checksums differ from fold files")

    complete = fold_count == 5 and bool(summary.get("scientific_comparison_valid"))
    if complete:
        combined: dict[str, dict[str, np.ndarray]] = {}
        for arm in ARMS:
            combined[arm] = {
                metric: np.concatenate(all_metrics[arm][metric])
                for metric in (*METRICS, "valid_target_gene_count", "excluded_target_gene_count")
            }
        expected_oof = _aggregate(combined, np.asarray(all_objects), np.asarray(all_keys))
        _assert_close(summary.get("out_of_fold_metrics"), expected_oof, "out_of_fold_metrics")
    elif summary.get("out_of_fold_metrics") is not None:
        raise ValueError("partial cost-gate run must not claim complete out-of-fold metrics")

    return {
        "status": "PASS_EXP066_READBACK" if complete else "PASS_EXP066_PARTIAL_COST_GATE_READBACK",
        "run_id": str(summary["run_id"]),
        "verified_folds": int(fold_count),
        "verified_conditions": int(len(all_condition_ids)),
        "target_gene_count": int(len(target_axis) if target_axis is not None else 0),
        "target_axis_hash": target_axis_hash,
        "prediction_artifacts": fold_file_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = verify_run(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
