"""连续扰动预测的训练侧尺度和细胞/药物等权评价。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GeneScale:
    mean: np.ndarray
    scale: np.ndarray


def fit_gene_scale(training_delta: np.ndarray, *, minimum: float = 1e-8) -> GeneScale:
    values = np.asarray(training_delta, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("training_delta must be a finite [conditions, genes] matrix")
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    invalid = ~np.isfinite(scale) | (scale <= minimum)
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} genes have undefined training scale")
    return GeneScale(mean=mean, scale=scale)


def condition_normalized_mse(observed: np.ndarray, predicted: np.ndarray,
                             scale: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if observed.shape != predicted.shape or observed.ndim != 2:
        raise ValueError("observed and predicted must share [conditions, genes] shape")
    if scale.shape != (observed.shape[1],) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("scale must be finite positive with one value per gene")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("observed/predicted contain non-finite values")
    return np.mean(np.square((predicted - observed) / scale), axis=1)


def equal_weight_cell_score(keys: pd.DataFrame, condition_scores: np.ndarray) -> tuple[pd.DataFrame, float]:
    """Average conditions within cell/drug, then drugs within cell, then cells."""
    required = {"condition_id", "cell_id", "pert_id"}
    if required - set(keys.columns):
        raise ValueError(f"condition keys lack {sorted(required - set(keys.columns))}")
    scores = np.asarray(condition_scores, dtype=np.float64)
    if len(keys) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("condition scores must be finite and align to keys")
    frame = keys[["condition_id", "cell_id", "pert_id"]].copy()
    frame["normalized_mse"] = scores
    drug = frame.groupby(["cell_id", "pert_id"], sort=True, as_index=False).agg(
        normalized_mse=("normalized_mse", "mean"), n_conditions=("condition_id", "size"))
    cell = drug.groupby("cell_id", sort=True, as_index=False).agg(
        normalized_mse=("normalized_mse", "mean"), n_drugs=("pert_id", "size"),
        n_conditions=("n_conditions", "sum"))
    return cell, float(cell["normalized_mse"].mean())
