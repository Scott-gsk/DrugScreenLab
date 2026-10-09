import pandas as pd
import pytest

from scripts.evaluation.verify_exp044_zero_target_label_prism_pharmaformer import (
    compare_keyed_predictions,
)


def frame(rows):
    return pd.DataFrame(rows, columns=[
        "cell_line_id", "drug_id", "predicted_source_pharmaformer",
        "predicted_source_train_mean",
    ])


def test_keyed_recomputation_is_independent_of_row_order():
    expected = frame([
        ("cell-a", "drug-a", 0.1, 0.4),
        ("cell-a", "drug-b", 0.2, 0.5),
        ("cell-b", "drug-a", 0.3, 0.4),
    ])
    actual = expected.iloc[[2, 0, 1]].reset_index(drop=True)

    result = compare_keyed_predictions(actual, expected)

    assert result["n_prediction_keys"] == 3
    assert result["n_exact_model_predictions"] == 3
    assert result["max_abs_model_prediction_error"] == 0.0
    assert result["source_mean_values_exact"] is True


def test_keyed_recomputation_rejects_duplicate_or_missing_keys():
    expected = frame([
        ("cell-a", "drug-a", 0.1, 0.4),
        ("cell-a", "drug-b", 0.2, 0.5),
    ])

    with pytest.raises(ValueError, match="duplicate"):
        compare_keyed_predictions(pd.concat([expected, expected.iloc[[0]]]), expected)
    with pytest.raises(ValueError, match="identical keyed coverage"):
        compare_keyed_predictions(expected.iloc[[0]], expected)


def test_keyed_recomputation_rejects_prediction_difference_outside_tolerance():
    expected = frame([("cell-a", "drug-a", 0.1, 0.4)])
    actual = frame([("cell-a", "drug-a", 0.10001, 0.4)])

    with pytest.raises(ValueError, match="model predictions exceed"):
        compare_keyed_predictions(actual, expected)
