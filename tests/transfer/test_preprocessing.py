import numpy as np
import pytest

from drug_screen.transfer.preprocessing import ExpressionPreprocessor


def test_fit_uses_only_train_medians_and_preserves_frozen_feature_order():
    train = np.array([[1.0, 10.0, np.nan], [3.0, 20.0, np.nan], [5.0, 30.0, np.nan]])
    processor = ExpressionPreprocessor.fit(train, ["A", "B", "ALL_MISSING"], min_gene_coverage=0.5, scale=False)
    assert processor.gene_ids == ("B", "A")  # B has the greater training variance
    transformed, report = processor.transform_with_report(np.array([[np.nan, 999.0]]), ["A", "B"])
    assert transformed.tolist() == [[999.0, 3.0]]
    assert report.n_missing_frozen_genes == 0
    assert report.n_imputed_values == 1


def test_transform_aligns_missing_genes_without_inventing_observed_zeros():
    processor = ExpressionPreprocessor.fit(
        np.array([[1.0, 4.0], [3.0, 8.0]]), ["A", "B"], scale=False
    )
    values, report = processor.transform_with_report(np.array([[9.0]]), ["A"])
    assert values.shape == (1, 2)
    assert values[0, processor.gene_ids.index("B")] == 6.0
    assert report.n_missing_frozen_genes == 1
    assert report.observed_fraction == 0.5


def test_preprocessor_round_trips_compact_npz_state(tmp_path):
    processor = ExpressionPreprocessor.fit(np.array([[1.0], [3.0]]), ["A"], scale=True)
    path = tmp_path / "expression_state.npz"
    processor.save(path)
    restored = ExpressionPreprocessor.load(path)
    np.testing.assert_allclose(restored.transform(np.array([[2.0]]), ["A"]), [[0.0]])


@pytest.mark.parametrize("values", [np.array([[np.inf]]), np.array([[-np.inf]])])
def test_fit_rejects_infinite_expression(values):
    with pytest.raises(ValueError, match="infinity"):
        ExpressionPreprocessor.fit(values, ["A"])


@pytest.mark.parametrize("gene_ids", [[None], ["  "]])
def test_fit_rejects_missing_or_blank_gene_ids(gene_ids):
    with pytest.raises(ValueError, match="non-empty strings"):
        ExpressionPreprocessor.fit(np.array([[1.0]]), gene_ids)


def test_deserialization_rejects_inconsistent_or_nonfinite_frozen_statistics():
    state = {
        "gene_ids": ["A", "B"], "medians": [1.0], "means": [1.0, 2.0],
        "scales": [1.0, 1.0], "min_gene_coverage": 0.8, "scaled": True,
    }
    with pytest.raises(ValueError, match="statistics"):
        ExpressionPreprocessor.from_dict(state)
    state["medians"] = [1.0, np.nan]
    with pytest.raises(ValueError, match="finite"):
        ExpressionPreprocessor.from_dict(state)
