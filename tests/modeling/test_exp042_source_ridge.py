import numpy as np
import pandas as pd

from scripts.modeling.exp042_source_ridge import (
    fit_expression_transform,
    fit_predict_ridge,
    macro_cell_scores,
    transform_expression,
    _validate_partitions,
)


def test_expression_transform_is_fitted_only_on_training_rows():
    expression = np.array([[1.0, np.nan], [3.0, 5.0], [999.0, 100.0]])
    center, scale = fit_expression_transform(expression, np.array([0, 1]))
    transformed = transform_expression(expression, center, scale)
    np.testing.assert_allclose(center, [2.0, 5.0])
    np.testing.assert_allclose(scale, [1.0, 1.0])
    np.testing.assert_allclose(transformed[:2, 0], [-1.0, 1.0])
    assert transformed[2, 0] > 900


def test_cell_macro_spearman_reports_undefined_constant_rows():
    frame = pd.DataFrame({
        "sample_id": ["a", "a", "a", "b", "b", "b"],
        "drug_id": ["d1", "d2", "d3", "d1", "d2", "d3"],
        "observed": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "predicted": [3.0, 2.0, 1.0, 1.0, 1.0, 1.0],
    })
    scores = macro_cell_scores(frame).set_index("sample_id")
    assert scores.loc["a", "spearman"] == -1.0
    assert np.isnan(scores.loc["b", "spearman"])
    assert scores.loc["b", "reason"] == "CONSTANT_PREDICTED"


def test_partition_validator_requires_inner_validation_to_cover_outer_train():
    identities = [f"c{i}" for i in range(10)]
    folds = []
    for outer_index in range(5):
        test = identities[outer_index * 2 : outer_index * 2 + 2]
        train = [identity for identity in identities if identity not in test]
        inner = []
        for inner_index in range(3):
            validation = train[inner_index::3]
            inner_train = [identity for identity in train if identity not in validation]
            inner.append({"inner_fold": inner_index, "train_ids": inner_train,
                          "validation_ids": validation})
        folds.append({"fold": outer_index, "outer_train_ids": train,
                      "outer_test_ids": test,
                      "label_fractions": [{"fraction": 1.0, "inner_folds": inner}]})
    pairs = pd.DataFrame({"sample_id": identities, "drug_id": "d1", "response": 0.1})
    _validate_partitions({"source_folds": folds}, set(identities), pairs)

    folds[0]["label_fractions"][0]["inner_folds"][0]["validation_ids"] = ["c0"]
    try:
        _validate_partitions({"source_folds": folds}, set(identities), pairs)
    except ValueError as error:
        assert "inner split is invalid" in str(error)
    else:
        raise AssertionError("inner identity leakage must fail closed")


def test_per_drug_ridge_uses_explicit_sample_identity_alignment():
    expression = np.array([
        [0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0],
    ], dtype=np.float32)
    train = pd.DataFrame({
        "sample_id": ["a", "b", "c", "a", "b", "c"],
        "drug_id": ["d1", "d1", "d1", "d2", "d2", "d2"],
        "response": [0.0, 0.1, 0.2, 0.2, 0.1, 0.0],
    })
    evaluation = pd.DataFrame({
        "sample_id": ["d", "e"], "drug_id": ["d1", "d2"], "response": [0.3, 0.1],
    })
    positions = {identity: index for index, identity in enumerate("abcde")}
    result = fit_predict_ridge(train, evaluation, expression, positions, alpha=1.0)
    assert result["sample_id"].tolist() == ["d", "e"]
    assert np.isfinite(result["predicted"].to_numpy()).all()
