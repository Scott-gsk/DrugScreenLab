"""EXP-026 数据清洗消融所需的训练折拟合归一化原语。

三个版本对应方案 ``第二波研究方案`` EXP-026 消融表的归一化维度：
1. ``ZScoreScaler``      逐基因 z-score（训练折拟合均值/标准差）；
2. ``QuantileNormalizer`` 分位数归一化（训练折构建参考分布，测试值经
   训练折经验分位数插值映射到参考分布，越界截断）；
3. ``RankNormalizer``     秩归一化（训练折经验 CDF，输出 (0,1) 开区间）。

三者共享 ``fit`` / ``transform`` / ``fit_transform`` 接口；所有统计量只在
训练折上估计，测试折仅变换（患者隔离协议的防泄漏要求）。
"""

from __future__ import annotations

import numpy as np

__all__ = ["ZScoreScaler", "QuantileNormalizer", "RankNormalizer"]


class ZScoreScaler:
    """逐基因 z-score，训练折拟合。"""

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ZScoreScaler":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + self.eps
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("fit before transform")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class QuantileNormalizer:
    """分位数归一化（训练折参考分布对齐）。

    训练折内：每基因列排序，取各排序位置跨基因均值得参考分布 ``reference_``。
    变换（训练或测试）：值 → 该基因训练折经验分位位置（线性插值，越界截断）
    → 参考分布上同一分位的值。等价于经典 quantile normalization 的插值实现，
    且对样本数不同的测试折同样适用。
    """

    def fit(self, X: np.ndarray) -> "QuantileNormalizer":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (samples x genes)")
        sorted_cols = np.sort(X, axis=0)
        self.reference_ = sorted_cols.mean(axis=1)
        self.sorted_train_ = sorted_cols
        n = X.shape[0]
        self.probs_ = (np.arange(n) + 0.5) / n
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "sorted_train_"):
            raise RuntimeError("fit before transform")
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.sorted_train_.shape[1]:
            raise ValueError("gene dimension mismatch")
        out = np.empty_like(X)
        for g in range(X.shape[1]):
            q = np.interp(X[:, g], self.sorted_train_[:, g], self.probs_)
            out[:, g] = np.interp(q, self.probs_, self.reference_)
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class RankNormalizer:
    """秩归一化（训练折经验 CDF）。

    变换值 = (searchsorted(train_col, x, 'right') + 0.5) / (n_train + 1)，
    输出严格落在 (0, 1)；列内顺序保持单调。
    """

    def fit(self, X: np.ndarray) -> "RankNormalizer":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (samples x genes)")
        self.sorted_train_ = np.sort(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "sorted_train_"):
            raise RuntimeError("fit before transform")
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.sorted_train_.shape[1]:
            raise ValueError("gene dimension mismatch")
        n = self.sorted_train_.shape[0]
        out = np.empty_like(X)
        for g in range(X.shape[1]):
            pos = np.searchsorted(self.sorted_train_[:, g], X[:, g], side="right")
            out[:, g] = (pos + 0.5) / (n + 1.0)
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
