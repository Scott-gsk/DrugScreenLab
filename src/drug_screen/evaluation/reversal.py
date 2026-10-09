"""固定疾病签名的双向扰动逆转读出。

输入为同一基因轴上的连续处理减对照表达变化；输出是方案 6.5 节定义的
无权重秩富集分数，不表示临床疗效或表达恢复幅度。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _validate_axis(
    delta: Sequence[float], genes: Sequence[str]
) -> tuple[np.ndarray, tuple[str, ...]]:
    values = np.asarray(delta, dtype=float)
    names = tuple(str(g) for g in genes)
    if values.ndim != 1 or len(values) != len(names) or len(names) < 2:
        raise ValueError("delta and genes must be aligned one-dimensional axes of length >= 2")
    if not np.isfinite(values).all():
        raise ValueError("delta contains missing or non-finite values")
    if any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("gene names must be non-empty and unique")
    return values, names


def _enrichment(values: np.ndarray, membership: np.ndarray) -> float:
    """按变化值降序，在每个同值组末尾计算带符号的前缀差。"""
    n_hit = int(membership.sum())
    n = len(values)
    if not 0 < n_hit < n:
        raise ValueError("signature set must be non-empty and smaller than output axis")
    order = np.argsort(-values, kind="stable")
    ranked = values[order]
    hit = membership[order]
    ends = np.r_[np.flatnonzero(ranked[1:] != ranked[:-1]) + 1, n]
    cumulative_hit = np.cumsum(hit, dtype=int)
    seen_hit = cumulative_hit[ends - 1]
    diff = seen_hit / n_hit - (ends - seen_hit) / (n - n_hit)
    high = float(np.max(diff))
    low = float(np.min(diff))
    if np.isclose(high, -low, rtol=0, atol=1e-12):
        return 0.0
    return high if high > -low else low


def reversal_score(
    delta: Sequence[float],
    genes: Sequence[str],
    disease_up: Sequence[str],
    disease_down: Sequence[str],
) -> float:
    """计算 ``[ES(疾病下调)-ES(疾病上调)]/2``，范围 [-1, 1]。

    基因轴必须显式给出、无重复且与 ``delta`` 等长。签名基因缺失、
    上下调集合重叠、空集合或覆盖整个输出轴时拒绝评分，避免自动改变
    预先固定的疾病签名或评价面板。并列变化作为一组处理；全零输出为 0。
    """
    values, names = _validate_axis(delta, genes)
    up, down = set(disease_up), set(disease_down)
    if up & down:
        raise ValueError("disease_up and disease_down overlap")
    axis = set(names)
    if (up | down) - axis:
        raise ValueError("signature contains genes absent from the output axis")
    if not 0 < len(up) < len(names) or not 0 < len(down) < len(names):
        raise ValueError("each signature direction must be non-empty and smaller than axis")
    if np.ptp(values) == 0:
        return 0.0
    up_mask = np.fromiter((g in up for g in names), dtype=bool)
    down_mask = np.fromiter((g in down for g in names), dtype=bool)
    return float((_enrichment(values, down_mask) - _enrichment(values, up_mask)) / 2)
