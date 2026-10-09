import numpy as np
import pandas as pd
import pytest

from drug_screen.evaluation.perturbation import (
    condition_normalized_mse, equal_weight_cell_score, fit_gene_scale,
)


def test_train_scale_and_hand_calculated_nmse():
    train = np.array([[0.0, 2.0], [2.0, 6.0]])
    scale = fit_gene_scale(train)
    assert np.allclose(scale.mean, [1.0, 4.0])
    assert np.allclose(scale.scale, [1.0, 2.0])
    score = condition_normalized_mse(np.array([[1.0, 4.0]]), np.array([[2.0, 2.0]]), scale.scale)
    assert score[0] == pytest.approx(1.0)


def test_equal_weighting_prevents_replicated_drug_and_cell_dominance():
    keys = pd.DataFrame({
        "condition_id": range(5),
        "cell_id": ["A", "A", "A", "A", "B"],
        "pert_id": ["d1", "d1", "d1", "d2", "d3"],
    })
    cells, total = equal_weight_cell_score(keys, np.array([0.0, 0.0, 0.0, 2.0, 4.0]))
    # A: mean(drug d1=0, drug d2=2)=1; B=4; cells equal => 2.5.
    assert dict(zip(cells.cell_id, cells.normalized_mse)) == {"A": 1.0, "B": 4.0}
    assert total == pytest.approx(2.5)


def test_constant_training_gene_is_explicitly_invalid():
    with pytest.raises(ValueError, match="undefined training scale"):
        fit_gene_scale(np.ones((3, 2)))
