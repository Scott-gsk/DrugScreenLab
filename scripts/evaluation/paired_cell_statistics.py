"""Paired uncertainty summaries for fixed cell-level evaluation metrics."""

from __future__ import annotations

import numpy as np


BOOTSTRAPS = 2_000
SIGN_FLIPS = 10_000


def paired_cell_statistics(difference: np.ndarray, *, seed: int) -> dict:
    """Bootstrap cell-level paired gains and test them with finite sign flips."""
    values = np.asarray(difference, dtype=np.float64)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("paired cell gains must contain at least three finite values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    bootstrap = values[indices].mean(axis=1)
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(SIGN_FLIPS, len(values)))
    observed = abs(float(values.mean()))
    permuted = np.abs((signs * values).mean(axis=1))
    extreme = int(np.sum(permuted >= observed))
    return {
        "estimate": float(values.mean()),
        "ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "p_two_sided_finite_corrected": float((extreme + 1) / (SIGN_FLIPS + 1)),
        "n_cells": int(len(values)),
        "bootstrap_replicates": BOOTSTRAPS,
        "sign_flip_replicates": SIGN_FLIPS,
        "unit": "held-out cell",
    }
