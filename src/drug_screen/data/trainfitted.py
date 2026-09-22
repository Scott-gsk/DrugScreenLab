"""Train-only-fitted preprocessing wrapper (work-package B, task 4).

Contract (scheme B5 data-side):
- Every statistic that touches a test / validation / audit row must be fitted
  on **training fold rows only**, then reused verbatim for any other row.
- ``standardize``: per-gene z-score with mean/std estimated on finite training
  values only (missing values never contribute to the estimate).
- ``impute``: a missing gene value is filled with its **training mean**, which
  makes its standardized value exactly ``z=0`` (the training mean).
- ``transform`` must never mutate fitted state and must never read statistics
  out of the rows it transforms (test rows cannot change the transform).

This module deliberately generalises :mod:`drug_screen.data.normalization`
and :mod:`drug_screen.transfer.preprocessing` by carrying an explicit
``fit_folds`` ledger so downstream audits can prove which fold set produced a
fitted transformer.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional

import numpy as np

__all__ = ["FitOnTrainTransformer"]


class FitOnTrainTransformer:
    """Standardize + missing-gene imputation fitted on train rows only.

    Parameters
    ----------
    eps:
        Floor for a gene standard deviation. A gene with zero (or all-missing)
        variance receives ``std=1.0`` so its transformation is well defined.

    Attributes
    ----------
    mean_ / std_:
        Per-gene statistics fitted on ``fit`` rows only.
    fit_folds:
        Frozen set of fold labels passed at ``fit`` time; empty when ``folds``
        was omitted.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.fit_folds: frozenset[str] = frozenset()

    def fit(
        self,
        X: object,
        *,
        folds: Optional[Iterable[object]] = None,
    ) -> "FitOnTrainTransformer":
        values = self._as_matrix(X)
        if values.shape[0] == 0:
            raise ValueError("fit requires at least one training row")
        finite = np.isfinite(values)
        n_present = finite.sum(axis=0)
        sums = np.where(finite, values, 0.0).sum(axis=0)
        mean = np.where(n_present > 0, sums / np.maximum(n_present, 1), 0.0)
        centered = np.where(finite, values - mean, 0.0)
        # Population std (matches the project's ZScoreScaler convention).
        variance = (centered * centered).sum(axis=0) / np.maximum(n_present, 1)
        std = np.sqrt(variance) + self.eps
        # Genes with no finite training value collapse to identity: missing and
        # present values both map to z=0.
        std = np.where(n_present > 0, std, 1.0)
        self.mean_ = mean
        self.std_ = std
        self.fit_folds = frozenset(str(f) for f in folds) if folds is not None else frozenset()
        return self

    def transform(self, X: object) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("fit before transform")
        values = self._as_matrix(X)
        if values.shape[1] != len(self.mean_):
            raise ValueError(
                f"gene dimension mismatch: got {values.shape[1]}, fitted {len(self.mean_)}"
            )
        imputed = np.where(np.isfinite(values), values, self.mean_)
        return (imputed - self.mean_) / self.std_

    def fit_transform(
        self,
        X: object,
        *,
        folds: Optional[Iterable[object]] = None,
    ) -> np.ndarray:
        return self.fit(X, folds=folds).transform(X)

    def _as_matrix(self, X: object) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("expression must be a 2-D array (samples x genes)")
        if values.shape[1] == 0:
            raise ValueError("expression must have at least one gene column")
        if np.isinf(values).any():
            raise ValueError("expression cannot contain +/- infinity")
        return values
