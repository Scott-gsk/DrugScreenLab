from __future__ import annotations

import numpy as np

from scripts.modeling.run_exp009_full_mvp import (
    execution_scope,
    load_morgan_feature_table,
    metrics,
    select_feature_rows,
    select_partition_positions,
)


def test_full_runner_marks_none_limits_as_full_and_debug_limits_as_debug():
    assert execution_scope(None, None) == "FULL"
    assert execution_scope(64, 64) == "DEBUG"


def test_partition_selector_preserves_official_partition_and_honors_optional_debug_limit():
    labels = np.asarray(["train", "test", "train", "valid", "test"])

    selected = select_partition_positions(labels, "train", 1)

    np.testing.assert_array_equal(selected, np.asarray([0]))


def test_full_runner_loads_morgan_features_once_and_selects_batch_rows_in_order(tmp_path):
    artifact = tmp_path / "features.npz"
    np.savez_compressed(
        artifact,
        pert_id=np.asarray(["drug-b", "drug-a"]),
        soft_target_probabilities=np.asarray([[0.2, 0.3], [0.4, 0.5]], dtype=np.float32),
        feature_valid=np.asarray([True, True]),
    )

    table, audit = load_morgan_feature_table(artifact, expected_target_dim=2)
    values = select_feature_rows(table, ["drug-a", "drug-b"])

    np.testing.assert_array_equal(values, np.asarray([[0.4, 0.5], [0.2, 0.3]], dtype=np.float32))
    assert audit["rows"] == 2


def test_metrics_report_row_macro_spearman_flat_statistics_and_finite_rate():
    truth = np.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
    prediction = np.asarray([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)

    result = metrics(truth, prediction)

    assert result["rows"] == 2
    assert result["genes"] == 2
    assert result["row_macro_spearman"] == 0.0
    assert result["finite_rate"] == 1.0
    assert result["mse"] == 0.5
