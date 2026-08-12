"""Transparent Phase-1 reversal and continuous PRISM ranking metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def average_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left_rank = average_rank(np.asarray(left, dtype=float))
    right_rank = average_rank(np.asarray(right, dtype=float))
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def reversal_score(disease_signature: np.ndarray, predicted_delta: np.ndarray) -> float:
    """Return larger-is-better anti-correlation with the disease signature."""
    value = spearman(np.asarray(disease_signature, dtype=float), np.asarray(predicted_delta, dtype=float))
    if value is None:
        raise ValueError("reversal score is undefined for constant vectors")
    return -value


def line_metrics(frame: pd.DataFrame, score_column: str, *, minimum_candidates: int = 3) -> dict[str, object]:
    eligible = frame.dropna(subset=[score_column, "sensitivity_score"]).copy()
    if len(eligible) < minimum_candidates:
        return {
            "candidate_count": int(len(eligible)),
            "eligible": False,
            "reason": "fewer_than_minimum_candidates",
        }
    predicted = eligible[score_column].to_numpy(float)
    response = eligible["sensitivity_score"].to_numpy(float)
    order_pred = np.argsort(-predicted, kind="mergesort")
    order_true = np.argsort(-response, kind="mergesort")
    top_k = min(2, len(eligible))
    top_pred = set(eligible.iloc[order_pred[:top_k]]["pert_id"])
    top_true = set(eligible.iloc[order_true[:top_k]]["pert_id"])
    return {
        "candidate_count": int(len(eligible)),
        "eligible": True,
        "spearman": spearman(predicted, response),
        "top2_overlap_count": int(len(top_pred & top_true)),
        "top2_overlap_rate": float(len(top_pred & top_true) / top_k),
        "predicted_top2": sorted(top_pred),
        "response_top2": sorted(top_true),
    }
