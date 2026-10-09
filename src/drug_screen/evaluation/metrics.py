"""Core ranking and regression metrics (方案第三章主要指标).

Implements:
- NDCG@k (任务 B 主指标默认 k=5)
- Spearman ρ (任务 A 逐药 / 任务 B 补充)
- Top-k 重合率 (默认 k=5)
- MAE / RMSE (同研究终点内数值预测)
- masked_spearman / masked_pearson（W3-09，2026-09-22 审计修复）：
  正确并列秩（``scipy.stats.rankdata`` method="average"）+ 显式缺失
  掩码（pairwise complete）
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import rankdata

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
    输入须为有限值（含 NaN 时抛错）；缺失掩码（pairwise complete）版本
    见 ``masked_spearman``。
    """
    truth = _as_1d(y_true, "y_true")
    pred = _as_1d(y_pred, "y_pred")
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if truth.size < 2:
        return float("nan")

    rt = rankdata(truth, method="average")
    rp = rankdata(pred, method="average")
    rt_c = rt - rt.mean()
    rp_c = rp - rp.mean()
    denom = float(np.sqrt(np.sum(rt_c**2) * np.sum(rp_c**2)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(rt_c * rp_c) / denom)


def _pairwise_valid(
    a: ArrayLike, b: ArrayLike, mask: ArrayLike | None, name_a: str, name_b: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """成对完整（pairwise complete）有效位置：isfinite(a) & isfinite(b)
    & (mask 若给定)。返回 (a[valid], b[valid], n_valid)。"""
    arr_a = np.asarray(a, dtype=float).reshape(-1)
    arr_b = np.asarray(b, dtype=float).reshape(-1)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"{name_a} and {name_b} must have the same shape")
    if arr_a.size == 0:
        raise ValueError("inputs must be non-empty")
    valid = np.isfinite(arr_a) & np.isfinite(arr_b)
    if mask is not None:
        m = np.asarray(mask, dtype=bool).reshape(-1)
        if m.shape != arr_a.shape:
            raise ValueError("mask must have the same shape as the inputs")
        valid = valid & m
    return arr_a[valid], arr_b[valid], int(valid.sum())


def masked_spearman(
    a: ArrayLike,
    b: ArrayLike,
    mask: ArrayLike | None = None,
) -> tuple[float, int]:
    """显式缺失掩码的 Spearman 相关（W3-09）。

    有效位置 = isfinite(a) & isfinite(b) & (mask 若给定)——pairwise
    complete 缺失处理 + 正确并列秩（``rankdata`` method="average"）。

    Returns
    -------
    (rho, n_used)：有效数 < 2 或任一侧秩零方差（常数）→ (NaN, n_used)。
    """
    x, y, n_used = _pairwise_valid(a, b, mask, "a", "b")
    if n_used < 2:
        return float("nan"), n_used
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    rx_c = rx - rx.mean()
    ry_c = ry - ry.mean()
    denom = float(np.sqrt(np.sum(rx_c**2) * np.sum(ry_c**2)))
    if denom == 0.0:
        return float("nan"), n_used
    return float(np.sum(rx_c * ry_c) / denom), n_used


def masked_pearson(
    a: ArrayLike,
    b: ArrayLike,
    mask: ArrayLike | None = None,
) -> tuple[float, int]:
    """显式缺失掩码的 Pearson 相关（W3-09：pairwise complete）。

    Returns
    -------
    (rho, n_used)：有效数 < 2 或任一侧零方差 → (NaN, n_used)。
    """
    x, y, n_used = _pairwise_valid(a, b, mask, "a", "b")
    if n_used < 2:
        return float("nan"), n_used
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = float(np.sqrt(np.sum(x_c**2) * np.sum(y_c**2)))
    if denom == 0.0:
        return float("nan"), n_used
    return float(np.sum(x_c * y_c) / denom), n_used


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
