from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.modeling.exp044_zero_label_prism_pharmaformer import (
    build_prediction_frame,
    source_means,
    validate_frozen_prediction_axes,
)


def test_source_means_use_only_frozen_fold_training_ids_and_requested_panel():
    source = pd.DataFrame({
        "sample_id": ["train_a", "train_a", "train_b", "train_b", "outside", "train_a"],
        "drug_id": ["drug_a", "drug_b", "drug_a", "drug_b", "drug_a", "other"],
        "response": [0.2, 0.8, 0.6, 0.4, 99.0, -99.0],
    })

    means, count = source_means(source, {"train_a", "train_b"}, ["drug_a", "drug_b"])

    assert count == 4
    assert means.to_dict() == {"drug_a": pytest.approx(0.4), "drug_b": pytest.approx(0.6)}


def test_prediction_frame_is_complete_keyed_and_contains_no_observed_response():
    means = pd.Series({"drug_a": 0.3, "drug_b": 0.7})

    frame = build_prediction_frame(
        ["cell_1", "cell_2"], ["drug_a", "drug_b"],
        np.asarray([0.1, 0.2, 0.8, 0.9]), means,
    )

    assert len(frame) == 4
    assert not frame.duplicated(["cell_line_id", "drug_id"]).any()
    assert set(frame.columns) == {
        "cell_line_id", "drug_id", "predicted_source_pharmaformer", "predicted_source_train_mean",
    }
    assert frame["predicted_source_train_mean"].tolist() == [0.3, 0.7, 0.3, 0.7]


def test_prediction_frame_rejects_incomplete_grid():
    with pytest.raises(ValueError, match="does not cover"):
        build_prediction_frame(["cell_1"], ["drug_a", "drug_b"], np.asarray([0.1]),
                               pd.Series({"drug_a": 0.3, "drug_b": 0.7}))


def test_frozen_prediction_axes_accept_revised_352_cell_cohort():
    cell_ids = [f"cell_{index:03d}" for index in range(352)]
    drug_ids = [f"drug_{index:02d}" for index in range(13)]

    validate_frozen_prediction_axes(cell_ids, drug_ids)


def test_frozen_prediction_axes_reject_old_353_cell_cohort():
    cell_ids = [f"cell_{index:03d}" for index in range(353)]
    drug_ids = [f"drug_{index:02d}" for index in range(13)]

    with pytest.raises(ValueError, match="352 unique cells and 13 unique drugs"):
        validate_frozen_prediction_axes(cell_ids, drug_ids)
