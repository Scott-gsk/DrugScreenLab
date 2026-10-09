"""Hand-check the TRAIN-only scale against the registered object-macro metric."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/evaluation"))
from analyze_exp055_xpert_train_error_components import (  # noqa: E402
    fit_nonnegative_global_mse_scale,
    fit_nonnegative_global_scale,
    source_object_equal_condition_weights,
    weighted_object_macro_mse_float64,
    weighted_object_macro_nmse_float64,
)


def test_object_equal_condition_weights_and_distinct_scale_objectives() -> None:
    # Object A has two conditions and object B has one; gene variances differ.
    # Hand calculation gives alpha_NMSE=(0.25 + 0.0625)/(0.3125 + 0.0625 + 0.5)
    # = 5/14, while raw-MSE alpha=0.5/(0.5 + 0.25 + 0.5)=0.4.
    keys = pd.DataFrame({
        "full_condition_key": ["a1", "a2", "b1"],
        "source_object_id": ["A", "A", "B"],
        "full_inchikey": ["drug", "drug", "drug"],
    })
    truth = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    prediction = np.asarray([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    variance = np.asarray([1.0, 4.0], dtype=np.float64)

    weights = source_object_equal_condition_weights(keys)
    np.testing.assert_allclose(weights, np.asarray([0.25, 0.25, 0.5]), rtol=0.0, atol=1e-15)
    alpha_nmse, nmse_identifiable, nmse_num, nmse_den = fit_nonnegative_global_scale(
        truth, prediction, weights, variance
    )
    alpha_mse, mse_identifiable, mse_num, mse_den = fit_nonnegative_global_mse_scale(
        truth, prediction, weights
    )
    assert nmse_identifiable and mse_identifiable
    np.testing.assert_allclose([nmse_num, nmse_den, alpha_nmse], [0.3125, 0.875, 5.0 / 14.0],
                               rtol=0.0, atol=1e-14)
    np.testing.assert_allclose([mse_num, mse_den, alpha_mse], [0.5, 1.25, 0.4],
                               rtol=0.0, atol=1e-14)
    assert alpha_nmse != alpha_mse

    nmse_at_one = weighted_object_macro_nmse_float64(truth, prediction, keys, variance)
    nmse_at_optimum = weighted_object_macro_nmse_float64(truth, alpha_nmse * prediction, keys, variance)
    mse_at_one = weighted_object_macro_mse_float64(truth, prediction, keys)
    mse_at_optimum = weighted_object_macro_mse_float64(truth, alpha_mse * prediction, keys)
    assert nmse_at_optimum <= nmse_at_one + 1e-12
    assert mse_at_optimum <= mse_at_one + 1e-12


def test_zero_prediction_norm_retains_feasible_alpha_one_as_nonidentifiable() -> None:
    keys = pd.DataFrame({
        "full_condition_key": ["a1", "b1"],
        "source_object_id": ["A", "B"],
        "full_inchikey": ["drug", "drug"],
    })
    truth = np.asarray([[1.0, -1.0], [2.0, -2.0]], dtype=np.float64)
    prediction = np.zeros_like(truth)
    weights = source_object_equal_condition_weights(keys)
    alpha, identifiable, numerator, denominator = fit_nonnegative_global_scale(
        truth, prediction, weights, np.asarray([1.0, 4.0])
    )
    assert alpha == 1.0
    assert not identifiable
    assert numerator == 0.0 and denominator == 0.0
