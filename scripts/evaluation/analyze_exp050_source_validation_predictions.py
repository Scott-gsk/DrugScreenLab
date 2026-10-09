"""Score frozen EXP-037 response predictions against permitted inner-validation labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/experiments/EXP-037/20260928_lincs_scale_16_121"
PREPARED = RUN / "prepared"
OUTPUT_ID = "20260930_source_validation_model_vs_baselines_v1"
TRANSIGEN_REPLAY = ROOT / "artifacts/experiments/EXP-050/20260930_transigen_validation_replay_v2"
XPERT_REPLAY = ROOT / "artifacts/experiments/EXP-050/20260930_xpert_validation_replay_v4"
ATOL = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def score_conditions(
    observed: np.ndarray,
    predicted: np.ndarray,
    gene_scale: np.ndarray,
) -> dict[str, np.ndarray]:
    obs = np.asarray(observed, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    if (obs.shape != pred.shape or obs.ndim != 2 or obs.shape[1] != 978
            or not np.isfinite(obs).all() or not np.isfinite(pred).all()):
        raise ValueError("observed and predicted source response matrices violate the finite 978-gene contract")

    nmse = np.square((pred - obs) / gene_scale).mean(axis=1)
    obs_centered = obs - obs.mean(axis=1, keepdims=True)
    pred_centered = pred - pred.mean(axis=1, keepdims=True)
    pearson_denominator = np.linalg.norm(obs_centered, axis=1) * np.linalg.norm(pred_centered, axis=1)
    pearson = np.divide(
        np.einsum("ij,ij->i", obs_centered, pred_centered),
        pearson_denominator,
        out=np.full(len(obs), np.nan, dtype=np.float64),
        where=pearson_denominator > 0,
    )

    obs_rank = rankdata(obs, axis=1, method="average")
    pred_rank = rankdata(pred, axis=1, method="average")
    obs_rank -= obs_rank.mean(axis=1, keepdims=True)
    pred_rank -= pred_rank.mean(axis=1, keepdims=True)
    rank_denominator = np.linalg.norm(obs_rank, axis=1) * np.linalg.norm(pred_rank, axis=1)
    spearman = np.divide(
        np.einsum("ij,ij->i", obs_rank, pred_rank),
        rank_denominator,
        out=np.full(len(obs), np.nan, dtype=np.float64),
        where=rank_denominator > 0,
    )

    both_nonzero = (obs != 0) & (pred != 0)
    same_sign = np.signbit(obs) == np.signbit(pred)
    pair_count = both_nonzero.sum(axis=1)
    sign_agreement = np.divide(
        (both_nonzero & same_sign).sum(axis=1),
        pair_count,
        out=np.full(len(obs), np.nan, dtype=np.float64),
        where=pair_count > 0,
    )
    return {
        "normalized_mse": nmse,
        "pearson": pearson,
        "spearman": spearman,
        "sign_agreement": sign_agreement,
        "nonzero_pair_count": pair_count,
    }


def run(output_dir: Path) -> dict[str, Any]:
    if sys.platform != "linux" or os.environ.get("CONDA_DEFAULT_ENV") != "drugscreening-gpu":
        raise RuntimeError("EXP-050 analyses require WSL2 Conda environment drugscreening-gpu")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite source-evaluation output: {output_dir}")

    manifest_path = PREPARED / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "PREPARED_TRAIN_VALIDATION_ONLY"
            or manifest.get("n_train121") != 32094
            or manifest.get("n_validation") != 4954
            or manifest.get("official_test_responses_read") is not False):
        raise ValueError("prepared inputs are not the frozen inner train/validation-only tier")

    keys_path = PREPARED / "conditions.parquet"
    train_path = PREPARED / "train121.npy"
    validation_path = PREPARED / "validation.npy"
    delta_path = PREPARED / "delta.npy"
    keys = pd.read_parquet(keys_path)
    train_positions = np.load(train_path, allow_pickle=False)
    validation_positions = np.load(validation_path, allow_pickle=False)
    delta = np.load(delta_path, mmap_mode="r", allow_pickle=False)
    if (delta.shape != (len(keys), 978)
            or len(train_positions) != 32094 or len(validation_positions) != 4954
            or len(train_positions) + len(validation_positions) != len(keys)
            or len(np.unique(train_positions)) != len(train_positions)
            or len(np.unique(validation_positions)) != len(validation_positions)
            or np.intersect1d(train_positions, validation_positions).size
            or train_positions.min() < 0 or validation_positions.min() < 0
            or train_positions.max() >= len(keys) or validation_positions.max() >= len(keys)):
        raise ValueError("frozen train/validation row axes do not match the train-only source matrix")
    if not set(("cell_id", "pert_id", "condition_id")).issubset(keys.columns):
        raise ValueError("prepared response keys are missing cell, drug, or condition identity")

    train_keys = keys.iloc[train_positions].reset_index(drop=True)
    validation_keys = keys.iloc[validation_positions].reset_index(drop=True)
    if (not np.allclose(pd.to_numeric(train_keys.pert_dose).to_numpy(dtype=np.float64), 10.0)
            or not np.allclose(pd.to_numeric(validation_keys.pert_dose).to_numpy(dtype=np.float64), 10.0)
            or not np.allclose(pd.to_numeric(train_keys.pert_time).to_numpy(dtype=np.float64), 24.0)
            or not np.allclose(pd.to_numeric(validation_keys.pert_time).to_numpy(dtype=np.float64), 24.0)):
        raise ValueError("source train/validation rows are outside the frozen 10 uM x 24 h scope")
    train_values = np.asarray(delta[train_positions], dtype=np.float32)
    observed = np.asarray(delta[validation_positions], dtype=np.float32)
    if (not np.isfinite(train_values).all() or not np.isfinite(observed).all()
            or train_keys.cell_id.nunique() != 121
            or validation_keys.cell_id.nunique() != 8
            or set(train_keys.cell_id.astype(str)) & set(validation_keys.cell_id.astype(str))):
        raise ValueError("source label finiteness or cell-disjoint inner split contract failed")

    transigen_receipt_path = TRANSIGEN_REPLAY / "summary.json"
    xpert_receipt_path = XPERT_REPLAY / "summary.json"
    transigen_receipt = json.loads(transigen_receipt_path.read_text(encoding="utf-8"))
    xpert_receipt = json.loads(xpert_receipt_path.read_text(encoding="utf-8"))
    for name, receipt in (("TranSiGen", transigen_receipt), ("XPert", xpert_receipt)):
        validation_not_read = receipt.get(
            "validation_response_values_read", receipt.get("validation_labels_read")
        )
        outer_test_not_read = receipt.get(
            "outer_test_response_values_read", receipt.get("outer_test_labels_read")
        )
        if (receipt.get("experiment_id") != "EXP-050" or receipt.get("status") != "PASS"
                or receipt.get("replay_match") is not True
                or receipt.get("max_absolute_difference") != 0.0
                or validation_not_read is not False
                or outer_test_not_read is not False
                or receipt.get("target_treated_responses_read") is not False):
            raise ValueError(f"{name} frozen replay receipt is not an exact validation-only PASS")

    transigen_training = json.loads((RUN / "transigen_121/summary.json").read_text(encoding="utf-8"))
    xpert_training = json.loads((RUN / "xpert_121/summary.json").read_text(encoding="utf-8"))
    selected_epochs = {
        "TranSiGen": int(transigen_training["selected_epoch"]),
        "XPert": int(xpert_training["selected_epoch"]),
    }

    prediction_paths = {
        "TranSiGen": RUN / "transigen_121/transigen_predictions.npy",
        "XPert": RUN / "xpert_121/validation_prediction.npy",
    }
    predictions: dict[str, np.ndarray] = {}
    for name, path in prediction_paths.items():
        receipt = transigen_receipt if name == "TranSiGen" else xpert_receipt
        declared = receipt["input_hashes"]["saved_validation_prediction"]
        if path.stat().st_size != declared["bytes"] or sha256(path) != declared["sha256"]:
            raise ValueError(f"{name} saved validation predictions differ from exact replay receipt")
        prediction = np.load(path, mmap_mode="r", allow_pickle=False)
        replayed_path = ROOT / receipt["replayed_prediction"]["path"]
        replayed_prediction = np.load(replayed_path, mmap_mode="r", allow_pickle=False)
        if prediction.shape != observed.shape or not np.isfinite(prediction).all():
            raise ValueError(f"{name} source validation prediction axis/value contract failed")
        if replayed_prediction.shape != prediction.shape or not np.array_equal(prediction, replayed_prediction):
            raise ValueError(f"{name} current saved prediction differs from its previously exact replay output")
        predictions[name] = np.asarray(prediction, dtype=np.float32)

    train_scale = train_values.std(axis=0, dtype=np.float64).astype(np.float64)
    replaced_scale_count = int((train_scale <= 1e-8).sum())
    train_scale[train_scale <= 1e-8] = 1.0
    if not np.isfinite(train_scale).all() or (train_scale <= 0).any():
        raise ValueError("training-side gene scales are invalid")

    train_drug_means: dict[str, np.ndarray] = {}
    train_drug = train_keys.pert_id.astype(str).to_numpy()
    for drug in pd.unique(train_drug):
        drug_rows = np.flatnonzero(train_drug == drug)
        train_drug_means[drug] = train_values[drug_rows].mean(axis=0, dtype=np.float64)
    validation_drug = validation_keys.pert_id.astype(str).to_numpy()
    absent_drugs = sorted(set(validation_drug) - set(train_drug_means))
    if absent_drugs:
        raise ValueError(f"validation drugs without a training-only mean baseline: {absent_drugs[:10]}")

    validation_mean = np.stack([train_drug_means[drug] for drug in validation_drug]).astype(np.float32)
    zeros = np.zeros_like(observed, dtype=np.float32)
    arms = {
        "TranSiGen": predictions["TranSiGen"],
        "XPert": predictions["XPert"],
        "inner_training_per_drug_mean": validation_mean,
        "zero_change": zeros,
    }
    cell_ids = validation_keys.cell_id.astype(str).to_numpy()
    drug_ids = validation_keys.pert_id.astype(str).to_numpy()
    condition_ids = validation_keys.condition_id.astype(str).to_numpy()
    condition_frames = []
    for arm, prediction in arms.items():
        metrics = score_conditions(observed, prediction, train_scale)
        frame = pd.DataFrame({
            "cell_id": cell_ids,
            "pert_id": drug_ids,
            "condition_id": condition_ids,
            "arm": arm,
            "normalized_mse": metrics["normalized_mse"],
            "pearson": metrics["pearson"],
            "spearman": metrics["spearman"],
            "sign_agreement": metrics["sign_agreement"],
            "nonzero_pair_count": metrics["nonzero_pair_count"],
        })
        condition_frames.append(frame)
    condition_scores = pd.concat(condition_frames, ignore_index=True)

    cell_rows = []
    for (arm, cell), frame in condition_scores.groupby(["arm", "cell_id"], sort=True):
        per_drug = frame.groupby("pert_id", sort=True).normalized_mse.mean()
        cell_row: dict[str, Any] = {
            "arm": str(arm),
            "cell_id": str(cell),
            "normalized_mse": float(per_drug.mean()),
            "n_drugs": int(per_drug.size),
            "n_conditions": int(len(frame)),
            "median_condition_pearson": json_safe_number(float(frame.pearson.median())),
            "valid_pearson_conditions": int(frame.pearson.notna().sum()),
            "median_condition_spearman": json_safe_number(float(frame.spearman.median())),
            "valid_spearman_conditions": int(frame.spearman.notna().sum()),
            "pooled_nonzero_sign_agreement": None,
            "nonzero_gene_response_pairs": int(frame.nonzero_pair_count.sum()),
        }
        denominator = cell_row["nonzero_gene_response_pairs"]
        if denominator:
            # The sign numerator is recomputed against the same finite, nonzero pairs.
            obs_cell = observed[cell_ids == str(cell)]
            pred_cell = prediction_for_arm(arm, predictions, validation_mean, zeros)[cell_ids == str(cell)]
            mask = (obs_cell != 0) & (pred_cell != 0)
            cell_row["pooled_nonzero_sign_agreement"] = float(
                ((np.signbit(obs_cell) == np.signbit(pred_cell)) & mask).sum() / mask.sum()
            )
        cell_rows.append(cell_row)

    cell_scores = pd.DataFrame(cell_rows)
    arm_summary: dict[str, Any] = {}
    for arm in arms:
        frame = cell_scores.loc[cell_scores.arm.eq(arm)].sort_values("cell_id")
        arm_summary[arm] = {
            "n_validation_cells": int(frame.cell_id.nunique()),
            "mean_equal_cell_normalized_mse": float(frame.normalized_mse.mean()),
            "median_cell_normalized_mse": float(frame.normalized_mse.median()),
            "cell_normalized_mse_range": [float(frame.normalized_mse.min()), float(frame.normalized_mse.max())],
            "median_across_cells_of_cell_median_condition_pearson": json_safe_number(
                float(frame.median_condition_pearson.median())
            ),
            "median_across_cells_of_cell_median_condition_spearman": json_safe_number(
                float(frame.median_condition_spearman.median())
            ),
            "median_across_cells_of_pooled_nonzero_sign_agreement": json_safe_number(
                float(frame.pooled_nonzero_sign_agreement.median())
            ),
            "undefined_correlation_cells": {
                "pearson": int(frame.median_condition_pearson.isna().sum()),
                "spearman": int(frame.median_condition_spearman.isna().sum()),
            },
            "undefined_sign_cells": int(frame.pooled_nonzero_sign_agreement.isna().sum()),
            "mean_nonzero_gene_pair_denominator_per_cell": float(frame.nonzero_gene_response_pairs.mean()),
            "cell_scores": [
                {key: json_safe_number(float(value)) if isinstance(value, (np.floating, float))
                 else (int(value) if isinstance(value, (np.integer, int)) else value)
                 for key, value in row.items()}
                for row in frame.to_dict(orient="records")
            ],
        }

    paired_comparisons = {}
    by_cell = cell_scores.pivot(index="cell_id", columns="arm", values="normalized_mse")
    for model in ("TranSiGen", "XPert"):
        paired_comparisons[model] = {}
        for baseline in ("inner_training_per_drug_mean", "zero_change"):
            gain = by_cell[baseline] - by_cell[model]
            base_mean = float(by_cell[baseline].mean())
            model_mean = float(by_cell[model].mean())
            paired_comparisons[model][baseline] = {
                "gain_in_equal_cell_normalized_mse_baseline_minus_model": float(gain.mean()),
                "cells_where_model_error_is_lower": int((gain > 0).sum()),
                "n_validation_cells": int(len(gain)),
                "mean_relative_error_reduction_percent": (
                    float((base_mean - model_mean) / base_mean * 100.0) if base_mean > 0 else None
                ),
                "per_cell_gain": {str(cell): float(value) for cell, value in gain.items()},
                "inferential_test_performed": False,
            }

    input_hashes = {
        "prepared_manifest": sha256(manifest_path),
        "conditions_parquet": sha256(keys_path),
        "train_rows": sha256(train_path),
        "validation_rows": sha256(validation_path),
        "transigen_replay_receipt": sha256(transigen_receipt_path),
        "xpert_replay_receipt": sha256(xpert_receipt_path),
        "transigen_saved_validation_prediction": sha256(prediction_paths["TranSiGen"]),
        "xpert_saved_validation_prediction": sha256(prediction_paths["XPert"]),
        "configuration": sha256(ROOT / "configs/experiments/EXP-050.yaml"),
    }
    label_slice_digest = hashlib.sha256()
    label_slice_digest.update(np.ascontiguousarray(train_values).tobytes())
    label_slice_digest.update(np.ascontiguousarray(observed).tobytes())
    input_hashes["authorized_train_and_validation_response_slices"] = label_slice_digest.hexdigest()

    result = {
        "experiment_id": "EXP-050",
        "run_id": OUTPUT_ID,
        "status": "PASS",
        "environment": "WSL2 Conda drugscreening-gpu",
        "compute": "CPU-only evaluation; 0 GPU hours",
        "source_partition": {
            "training_conditions": int(len(train_positions)),
            "training_cells": int(train_keys.cell_id.nunique()),
            "validation_conditions": int(len(validation_positions)),
            "validation_cells": int(validation_keys.cell_id.nunique()),
            "training_rows_sha256": input_hashes["train_rows"],
            "validation_rows_sha256": input_hashes["validation_rows"],
            "validation_drugs": int(validation_keys.pert_id.nunique()),
            "condition_axis_order_matches_frozen_validation_positions": True,
        },
        "scoring_contract": {
            "primary_metric": "training-SD-normalized MSE; average genes within condition, conditions within drug, drugs equally within cell, then equal average of validation cells; lower is better",
            "gene_scale": "standard deviation of measured inner-training response by gene; scales <=1e-8 replaced by 1",
            "replaced_near_zero_gene_scales": replaced_scale_count,
            "secondary_metrics": "median condition-wise gene Pearson/Spearman and pooled nonzero sign agreement, first summarized within each cell",
            "undefined_handling": "constant predictions or zero nonzero-pair denominator remain null with counts/reason; no zero substitution",
            "baselines": "zero change; within-training perturbation mean per drug, computed from 32,094 inner-training rows only",
        },
        "selection_bias_and_interpretation": {
            "selected_epochs": selected_epochs,
            "both_checkpoints_selected_using_same_inner_validation_labels": True,
            "validation_response_values_read": True,
            "validation_response_values_used_for_scoring": True,
            "outer_test_response_values_read": False,
            "target_treated_response_values_read": False,
            "model_or_checkpoint_selection_after_scoring": False,
            "independent_generalization_claim": False,
            "interpretation": "Development-set descriptive comparison only. Both early-stopping epochs were selected using these same inner-validation responses, so reported model scores are selection-biased and are not an independent estimate of generalization.",
        },
        "arms": arm_summary,
        "paired_cell_descriptive_comparisons": paired_comparisons,
        "input_hashes": input_hashes,
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "source_delta_manifest_sha256_declared_not_recomputed": manifest["files_sha256"]["delta.npy"],
    }

    output_dir.mkdir(parents=True)
    cell_scores_path = output_dir / "cell_scores.tsv"
    cell_scores.to_csv(cell_scores_path, sep="\t", index=False, na_rep="")
    condition_scores_path = output_dir / "condition_scores.tsv"
    condition_scores.to_csv(condition_scores_path, sep="\t", index=False, na_rep="")
    result["output_files"] = {
        "cell_scores": {"path": cell_scores_path.relative_to(ROOT).as_posix(),
                        "sha256": sha256(cell_scores_path), "rows": int(len(cell_scores))},
        "condition_scores": {"path": condition_scores_path.relative_to(ROOT).as_posix(),
                             "sha256": sha256(condition_scores_path),
                             "rows": int(len(condition_scores))},
    }
    summary_path = output_dir / "summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")
    os.replace(temporary, summary_path)
    return result


def prediction_for_arm(
    arm: str,
    predictions: dict[str, np.ndarray],
    training_mean: np.ndarray,
    zeros: np.ndarray,
) -> np.ndarray:
    return {
        "TranSiGen": predictions["TranSiGen"],
        "XPert": predictions["XPert"],
        "inner_training_per_drug_mean": training_mean,
        "zero_change": zeros,
    }[arm]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(
        f"artifacts/experiments/EXP-050/{OUTPUT_ID}"))
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    print(json.dumps(run(output_dir), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
