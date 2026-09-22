"""Core ranking and regression metrics (方案第三章主要指标).

Implements:
- NDCG@k (任务 B 主指标默认 k=5)
- Spearman ρ (任务 A 逐药 / 任务 B 补充)
- Top-k 重合率 (默认 k=5)
- MAE / RMSE (同研究终点内数值预测)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


def _as_1d(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """DCG with binary-log discount; relevances already in ranked order."""
    k = min(k, relevances.size)
    if k <= 0:
        return 0.0
    rel = relevances[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=float))
    return float(np.sum(rel * discounts))


def ndcg_at_k(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    k: int = 5,
) -> float:
    """Normalized Discounted Cumulative Gain at ``k``.

    ``y_true`` is graded relevance (higher = more sensitive / better).
    ``y_score`` is model score used to induce the predicted ranking
    (higher score ranked first). Both arrays are aligned by position
    (same candidate set for one patient / grouping unit).

    Untested drugs should be filtered out by the caller before calling
    (方案: 未测药不计为错误).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    truth = _as_1d(y_true, "y_true")
    score = _as_1d(y_score, "y_score")
    if truth.shape != score.shape:
        raise ValueError("y_true and y_score must have the same shape")

    order = np.argsort(-score, kind="mergesort")
    dcg = _dcg_at_k(truth[order], k)

    ideal_order = np.argsort(-truth, kind="mergesort")
    idcg = _dcg_at_k(truth[ideal_order], k)
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def spearman_rho(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Spearman rank correlation (average ranks for ties).

    Returns NaN when either side has zero variance in ranks (undefined).
    """
    truth = _as_1d(y_true, "y_true")
    pred = _as_1d(y_pred, "y_pred")
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if truth.size < 2:
        return float("nan")

    def _rank(a: np.ndarray) -> np.ndarray:
        sorter = np.argsort(a, kind="mergesort")
        ranks = np.empty(a.size, dtype=float)
        ranks[sorter] = np.arange(1, a.size + 1, dtype=float)
        # average ties
        _, start_idx, counts = np.unique(a, return_index=True, return_counts=True)
        # unique on unsorted a: recompute via sorted values
        sorted_a = a[sorter]
        i = 0
        while i < sorted_a.size:
            j = i + 1
            while j < sorted_a.size and sorted_a[j] == sorted_a[i]:
                j += 1
            if j - i > 1:
                avg = (i + 1 + j) / 2.0
                ranks[sorter[i:j]] = avg
            i = j
        return ranks

    rt = _rank(truth)
    rp = _rank(pred)
    rt_c = rt - rt.mean()
    rp_c = rp - rp.mean()
    denom = float(np.sqrt(np.sum(rt_c**2) * np.sum(rp_c**2)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(rt_c * rp_c) / denom)


def top_k_overlap(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    k: int = 5,
    item_ids: Sequence[object] | None = None,
) -> float:
    """Top-k 重合率 = |Top-k(true) ∩ Top-k(score)| / k.

    When ``item_ids`` is None, positional indices are used as item identity.
    Ties broken by stable mergesort on scores / truth (higher first).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    truth = _as_1d(y_true, "y_true")
    score = _as_1d(y_score, "y_score")
    if truth.shape != score.shape:
        raise ValueError("y_true and y_score must have the same shape")

    n = truth.size
    if item_ids is None:
        ids = np.arange(n)
    else:
        if len(item_ids) != n:
            raise ValueError("item_ids length must match y_true")
        ids = np.asarray(item_ids, dtype=object)

    eff_k = min(k, n)
    true_top = set(ids[np.argsort(-truth, kind="mergesort")[:eff_k]].tolist())
    pred_top = set(ids[np.argsort(-score, kind="mergesort")[:eff_k]].tolist())
    return float(len(true_top & pred_top) / float(k))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error (same study endpoint only)."""
    truth = _as_1d(y_true, "y_true")
    pred = _as_1d(y_pred, "y_pred")
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(np.abs(truth - pred)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error (same study endpoint only)."""
    truth = _as_1d(y_true, "y_true")
    pred = _as_1d(y_pred, "y_pred")
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.sqrt(np.mean((truth - pred) ** 2)))
