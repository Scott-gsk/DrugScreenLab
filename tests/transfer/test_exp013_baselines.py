import numpy as np

from drug_screen.transfer.baselines import fit_b0_drug_means, fit_b1_per_drug_ridge, predict_b0, predict_b1


def test_b0_returns_train_drug_means():
    model = fit_b0_drug_means(["a", "a", "b"], [2.0, 4.0, 10.0])
    assert model.drug_means["a"] == 3.0
    assert predict_b0(model, ["a", "b", "c"]) == [3.0, 10.0, model.global_mean]


def test_b1_recovers_a_linear_expression_effect():
    expression = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [4.0, 1.0, 0.0],
            [5.0, 1.0, 0.0],
            [6.0, 1.0, 0.0],
            [7.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    sample_indices = list(range(8))
    drug_ids = ["d"] * 8
    responses = [float(index) for index in range(8)]
    model = fit_b1_per_drug_ridge(
        expression,
        sample_indices,
        drug_ids,
        responses,
        n_hvg=2,
        alpha=1e-6,
        min_train=8,
    )
    predicted = predict_b1(model, expression, sample_indices, drug_ids)
    errors = [abs(left - right) for left, right in zip(predicted, responses)]
    assert max(errors) < 0.05
