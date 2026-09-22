"""Permutation / substitute contrast hooks (个体信息贡献, 方案第三/八章).

Hooks only — callers supply the model ``predict_fn`` and expression matrix.
Does not run heavy training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

PredictFn = Callable[[np.ndarray], np.ndarray]
MetricFn = Callable[[np.ndarray, np.ndarray], float]


class ExpressionBatch(Protocol):
    """Minimal view of an expression matrix (n_samples, n_features)."""

    def __array__(self) -> np.ndarray: ...


@dataclass(frozen=True)
class PermutationContrastResult:
    """Contrast between original metric and disrupted-expression metric."""

    metric_name: str
    baseline_score: float
    disrupted_score: float
    delta: float  # baseline - disrupted; >0 means expression helped
    n_repeats: int
    disrupted_scores: tuple[float, ...]


def _as_matrix(x: ExpressionBatch | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError("expression matrix must be 2-D (n_samples, n_features)")
    return arr


@dataclass
class ExpressionPermutationContrast:
    """同癌种（或同队列）表达整行置换对照.

    Shuffle sample rows of the expression block so that drug / label
    alignment is preserved but individual expression identity is broken.
    """

    predict_fn: PredictFn
    metric_fn: MetricFn
    metric_name: str = "metric"
    n_repeats: int = 100
    random_state: int | None = 0

    def run(
        self,
        expression: ExpressionBatch | np.ndarray,
        y_true: Sequence[float],
        *,
        baseline_pred: Sequence[float] | None = None,
    ) -> PermutationContrastResult:
        expr = _as_matrix(expression)
        truth = np.asarray(y_true, dtype=float).reshape(-1)
        if expr.shape[0] != truth.size:
            raise ValueError("expression rows must match y_true length")

        if baseline_pred is None:
            baseline_pred_arr = np.asarray(self.predict_fn(expr), dtype=float).reshape(-1)
        else:
            baseline_pred_arr = np.asarray(baseline_pred, dtype=float).reshape(-1)
        if baseline_pred_arr.size != truth.size:
            raise ValueError("baseline_pred length mismatch")

        baseline = float(self.metric_fn(truth, baseline_pred_arr))
        rng = np.random.default_rng(self.random_state)
        disrupted: list[float] = []
        for _ in range(self.n_repeats):
            perm = rng.permutation(expr.shape[0])
            pred = np.asarray(self.predict_fn(expr[perm]), dtype=float).reshape(-1)
            disrupted.append(float(self.metric_fn(truth, pred)))

        mean_disrupted = float(np.mean(disrupted)) if disrupted else float("nan")
        return PermutationContrastResult(
            metric_name=self.metric_name,
            baseline_score=baseline,
            disrupted_score=mean_disrupted,
            delta=baseline - mean_disrupted,
            n_repeats=len(disrupted),
            disrupted_scores=tuple(disrupted),
        )


@dataclass
class MeanExpressionSubstituteContrast:
    """训练均值表达替代对照.

    Replace every sample's expression with the training-set mean vector
    (broadcast), then re-score. Large drop vs baseline supports
    individualized signal (方案第八章: 表达置换无影响 → 不足以支持个体化).
    """

    predict_fn: PredictFn
    metric_fn: MetricFn
    metric_name: str = "metric"
    train_expression: ExpressionBatch | np.ndarray | None = None

    def run(
        self,
        expression: ExpressionBatch | np.ndarray,
        y_true: Sequence[float],
        *,
        baseline_pred: Sequence[float] | None = None,
    ) -> PermutationContrastResult:
        expr = _as_matrix(expression)
        truth = np.asarray(y_true, dtype=float).reshape(-1)
        if expr.shape[0] != truth.size:
            raise ValueError("expression rows must match y_true length")

        if baseline_pred is None:
            baseline_pred_arr = np.asarray(self.predict_fn(expr), dtype=float).reshape(-1)
        else:
            baseline_pred_arr = np.asarray(baseline_pred, dtype=float).reshape(-1)

        baseline = float(self.metric_fn(truth, baseline_pred_arr))

        source = (
            _as_matrix(self.train_expression)
            if self.train_expression is not None
            else expr
        )
        mean_vec = source.mean(axis=0, keepdims=True)
        substituted = np.repeat(mean_vec, expr.shape[0], axis=0)
        pred = np.asarray(self.predict_fn(substituted), dtype=float).reshape(-1)
        disrupted = float(self.metric_fn(truth, pred))

        return PermutationContrastResult(
            metric_name=self.metric_name,
            baseline_score=baseline,
            disrupted_score=disrupted,
            delta=baseline - disrupted,
            n_repeats=1,
            disrupted_scores=(disrupted,),
        )
