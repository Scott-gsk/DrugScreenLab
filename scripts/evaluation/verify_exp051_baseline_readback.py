#!/usr/bin/env python3
"""Safely recompute EXP-051 train-fitted mean-baseline metrics from a closed run.

This root-side readback consumes only the attempt's safe NPZ predictions and the
frozen response-blind condition table. It never opens the source H5AD matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METRIC_FIELDS = {
    "object_macro_nmse": "nmse",
    "object_macro_mae": "mae",
    "object_macro_gene_correlation": "correlation",
    "object_macro_direction_agreement": "direction",
    "object_macro_amplitude_ratio": "amplitude_ratio",
}
BASELINES = ("zero_change", "global_mean_delta", "per_drug_mean_delta")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_by_object(
    truth: np.ndarray,
    prediction: np.ndarray,
    objects: np.ndarray,
    variance: np.ndarray,
    object_ids: np.ndarray,
) -> tuple[dict[str, float | None], np.ndarray]:
    use_gene = np.isfinite(variance) & (variance > 1e-12)
    error = prediction - truth
    row_nmse = np.mean(np.square(error[:, use_gene]) / variance[use_gene][None, :], axis=1)
    row_mae = np.mean(np.abs(error), axis=1)
    row_correlation: list[float | None] = []
    row_direction: list[float | None] = []
    for observed, predicted in zip(truth, prediction):
        row_correlation.append(
            float(np.corrcoef(observed, predicted)[0, 1])
            if np.ptp(observed) > 0 and np.ptp(predicted) > 0
            else None
        )
        nonzero = np.abs(observed) > 1e-12
        row_direction.append(
            float(np.mean(np.sign(observed[nonzero]) == np.sign(predicted[nonzero])))
            if nonzero.any()
            else None
        )

    object_metric_rows: dict[str, list[float]] = {
        "nmse": [], "mae": [], "correlation": [], "direction": []
    }
    amplitude_ratios: list[float] = []
    per_object_nmse: list[float] = []
    for object_id in object_ids:
        rows = objects == object_id
        for name, values in (
            ("nmse", row_nmse),
            ("mae", row_mae),
            ("correlation", np.asarray([np.nan if x is None else x for x in row_correlation])),
            ("direction", np.asarray([np.nan if x is None else x for x in row_direction])),
        ):
            finite_values = values[rows][np.isfinite(values[rows])]
            value = float(finite_values.mean()) if len(finite_values) else float("nan")
            if np.isfinite(value):
                object_metric_rows[name].append(value)
        per_object_nmse.append(float(row_nmse[rows].mean()))
        observed_abs = float(np.mean(np.abs(truth[rows])))
        predicted_abs = float(np.mean(np.abs(prediction[rows])))
        if observed_abs > 0:
            amplitude_ratios.append(predicted_abs / observed_abs)

    metrics = {
        name: (float(np.mean(values)) if values else None)
        for name, values in object_metric_rows.items()
    }
    metrics["amplitude_ratio"] = float(np.mean(amplitude_ratios)) if amplitude_ratios else None
    return metrics, np.asarray(per_object_nmse, dtype=np.float64)


def assert_metric_match(actual: Any, observed: float | None, label: str) -> float:
    if actual is None or observed is None:
        if actual is not None or observed is not None:
            raise AssertionError(f"undefined metric mismatch: {label}: {actual} vs {observed}")
        return 0.0
    difference = abs(float(actual) - observed)
    # Stored baseline prediction matrices are float32; recomputing on their
    # serialized values can differ from the original float64 accumulator by a
    # few 1e-8, so the readback gate uses a tight 1e-7 absolute tolerance.
    if difference > 1e-7:
        raise AssertionError(f"metric mismatch: {label}: {actual} vs {observed}")
    return difference


def build_readback(run_dir: Path, source_run_dir: Path, attempt_id: str) -> dict[str, Any]:
    attempt = run_dir / attempt_id
    prediction_path = attempt / "baseline_predictions.npz"
    metrics_path = attempt / "baseline_metrics.json"
    key_table_path = source_run_dir / "response_blind_conditions.tsv"
    output_path = run_dir / "baseline_independent_readback_v2.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    # `allow_pickle=False` is intentional: object arrays are never trusted/read.
    with np.load(prediction_path, allow_pickle=False) as archive:
        train_keys = archive["train_condition_key"].astype(str)
        dev_keys = archive["inner_dev_condition_key"].astype(str)
        train_truth = archive["train_target_delta"].astype(np.float64)
        dev_truth = archive["inner_dev_target_delta"].astype(np.float64)
        genes = archive["genes"].astype(str)
        predictions = {
            name: {
                "train": archive[f"{name}_train"].astype(np.float64),
                "inner_dev": archive[f"{name}_inner_dev"].astype(np.float64),
            }
            for name in BASELINES
        }

    metadata = pd.read_csv(key_table_path, sep="\t", dtype=str, keep_default_na=False)
    stored = json.loads(metrics_path.read_text(encoding="utf-8"))
    if train_truth.shape != (419, 978) or dev_truth.shape != (93, 978):
        raise ValueError("frozen 419/93 row and 978-gene shape contract failed")
    if len(genes) != 978 or not np.isfinite(train_truth).all() or not np.isfinite(dev_truth).all():
        raise ValueError("gene axis or truth values failed finite/length checks")
    if len(set(train_keys)) != 419 or len(set(dev_keys)) != 93:
        raise ValueError("condition keys are not unique within the frozen splits")
    if set(train_keys) & set(dev_keys):
        raise ValueError("TRAIN/inner_dev condition-key overlap detected")
    if not metadata["full_condition_key"].is_unique:
        raise ValueError("frozen metadata condition keys are not unique")
    train_metadata_keys = set(
        metadata.loc[metadata["split_role"] == "train", "full_condition_key"]
    )
    dev_metadata_keys = set(
        metadata.loc[metadata["split_role"] == "inner_dev", "full_condition_key"]
    )
    if set(train_keys) != train_metadata_keys or set(dev_keys) != dev_metadata_keys:
        raise ValueError("NPZ condition keys differ from frozen response-blind split metadata")

    key_to_object = metadata.set_index("full_condition_key")["source_object_id"].to_dict()
    train_objects = np.asarray([key_to_object[key] for key in train_keys], dtype=str)
    dev_objects = np.asarray([key_to_object[key] for key in dev_keys], dtype=str)
    if set(train_objects) & set(dev_objects):
        raise ValueError("source-object overlap detected between TRAIN and inner_dev")
    if len(set(metadata.loc[metadata["split_role"].isin(["train", "inner_dev"]), "full_inchikey"])) != 5:
        raise ValueError("the frozen five-full-InChIKey domain changed")
    if not np.isfinite(train_truth).all() or not np.isfinite(dev_truth).all():
        raise ValueError("non-finite targets found")
    if any(
        pred[role].shape != (419, 978) or not np.isfinite(pred[role]).all()
        for pred in predictions.values() for role in ("train",)
    ) or any(
        pred["inner_dev"].shape != dev_truth.shape or not np.isfinite(pred["inner_dev"]).all()
        for pred in predictions.values()
    ):
        raise ValueError("baseline predictions are non-finite or have the wrong shape")

    train_variance = np.var(train_truth, axis=0, ddof=0)
    use_gene = np.isfinite(train_variance) & (train_variance > 1e-12)
    if int(use_gene.sum()) != 978:
        raise ValueError("TRAIN-fitted gene-variance denominator changed")
    object_ids_train = np.asarray(sorted(set(train_objects)), dtype=str)
    object_ids_dev = np.asarray(sorted(set(dev_objects)), dtype=str)
    if len(object_ids_train) != 110 or len(object_ids_dev) != 24:
        raise ValueError("inner_dev must contain the frozen 24 source objects")

    recomputed: dict[str, dict[str, dict[str, float | None]]] = {}
    per_object_nmse: dict[str, dict[str, np.ndarray]] = {}
    metric_abs_differences: list[float] = []
    for name, prediction_by_role in predictions.items():
        recomputed[name] = {}
        per_object_nmse[name] = {}
        for role, truth, objects, object_ids in (
            ("train", train_truth, train_objects, object_ids_train),
            ("inner_dev", dev_truth, dev_objects, object_ids_dev),
        ):
            recomputed[name][role], per_object_nmse[name][role] = metric_by_object(
                truth, prediction_by_role[role], objects, train_variance, object_ids
            )
            for stored_field, result_field in METRIC_FIELDS.items():
                metric_abs_differences.append(assert_metric_match(
                    stored["metrics"][name][role][stored_field],
                    recomputed[name][role][result_field],
                    f"{name}.{role}.{stored_field}"
                ))

    rng = np.random.default_rng(51051)
    paired: dict[str, dict[str, Any]] = {}
    for reference in ("zero_change", "global_mean_delta"):
        # Positive means the TRAIN-fitted per-drug mean has lower NMSE.
        difference = (
            per_object_nmse[reference]["inner_dev"]
            - per_object_nmse["per_drug_mean_delta"]["inner_dev"]
        )
        resample_index = rng.integers(0, len(object_ids_dev), size=(20_000, len(object_ids_dev)))
        bootstrap_mean = difference[resample_index].mean(axis=1)
        paired[reference] = {
            "reference_minus_per_drug_object_macro_nmse": float(difference.mean()),
            "per_drug_relative_improvement_percent_vs_reference": float(
                100.0 * difference.mean() / per_object_nmse[reference]["inner_dev"].mean()
            ),
            "paired_object_bootstrap_95pct_percentile_interval": [
                float(np.quantile(bootstrap_mean, 0.025)),
                float(np.quantile(bootstrap_mean, 0.975)),
            ],
            "n_paired_inner_dev_objects": int(len(object_ids_dev)),
            "bootstrap_replicates": 20_000,
            "seed": 51051,
        }

    return {
        "run_id": run_dir.name,
        "source_attempt": attempt_id,
        "purpose": "Root-only safe readback and paired object-level comparison of already generated train-fitted baselines after the TranSiGen smoke closed.",
        "data_access": "Only the attempt's safe NPZ predictions and frozen response-blind condition metadata were read. NPZ was opened with allow_pickle=False; the source H5AD matrix, PDO, outer, and locked labels were not read.",
        "readback": {
            "safe_npz_allow_pickle_false": True,
            "train_conditions": int(len(train_keys)),
            "train_objects": int(len(set(train_objects))),
            "inner_dev_conditions": int(len(dev_keys)),
            "inner_dev_objects": int(len(object_ids)),
            "train_inner_dev_object_overlap": 0,
            "train_inner_dev_full_condition_key_overlap": 0,
            "full_inchikeys": 5,
            "genes": int(len(genes)),
            "all_targets_and_predictions_finite": True,
            "generator_metrics_recomputed_match_abs_tol_1e_7": True,
            "generator_metrics_max_abs_difference": max(metric_abs_differences),
            "baseline_predictions_sha256": sha256_file(prediction_path),
            "baseline_metrics_sha256": sha256_file(metrics_path),
            "response_blind_key_table_sha256": sha256_file(key_table_path),
            "readback_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "train_object_macro_metrics": {name: values["train"] for name, values in recomputed.items()},
        "inner_dev_object_macro_metrics": {name: values["inner_dev"] for name, values in recomputed.items()},
        "paired_object_bootstrap": paired,
        "interpretation_boundary": (
            "Development baseline comparison only. Intervals resample 24 inner-development source objects, "
            "not drugs, patients, PDO models, or an independent domain; no original-model fit result exists."
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=root / "artifacts/experiments/EXP-051/20261006_exp051_representative_compare_v1",
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        default=root / "artifacts/experiments/EXP-051/20261006_exp051_contract_fit_smoke_v1",
    )
    parser.add_argument("--attempt-id", default="fit_attempt_002")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    source_run_dir = args.source_run_dir.resolve()
    payload = build_readback(run_dir, source_run_dir, args.attempt_id)
    output_path = run_dir / "baseline_independent_readback_v2.json"
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
