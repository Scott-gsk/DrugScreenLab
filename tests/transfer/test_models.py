import numpy as np
import pytest

from drug_screen.transfer.models import DualEncoderPairModel, RidgePairModel


def _data(n=48):
    rng = np.random.default_rng(3)
    sample, drug = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
    labels = 1.5 * sample[:, 0] - .5 * drug[:, 1] + 2 * sample[:, 0] * drug[:, 0]
    return sample, drug, labels, np.repeat(["o1", "o2", "o3", "o4"], n // 4)


def test_ridge_learns_interaction_and_explanation_reconstructs_prediction(tmp_path):
    sample, drug, labels, _ = _data()
    model = RidgePairModel(alpha=.01, interaction_features=2).fit(sample, drug, labels)
    prediction = model.predict(sample, drug)
    assert np.corrcoef(labels, prediction)[0, 1] > .98
    explanation = model.explain(sample, drug)
    assert np.allclose(explanation["prediction"], prediction)
    saved = tmp_path / "ridge.pkl"; model.save(saved)
    assert np.allclose(RidgePairModel.load(saved).predict(sample, drug), prediction)


def test_dual_encoder_pair_mask_save_load_and_integrated_gradient_completeness(tmp_path):
    sample, drug, labels, pdos = _data()
    left, right = DualEncoderPairModel.ranking_pairs([2, 1, 1, 0], ["x", "x", "x", "y"])
    assert list(zip(left, right)) == [(0, 1), (0, 2)]
    assert all({index, other} <= {0, 1, 2} for index, other in zip(left, right))
    model = DualEncoderPairModel(hidden_dim=8, learning_rate=.02, ranking_weight=.1, seed=7, epochs=250).fit(sample, drug, labels, pdo_ids=pdos)
    prediction = model.predict(sample, drug)
    assert np.corrcoef(labels, prediction)[0, 1] > .85
    explanation = model.explain(sample[:3], drug[:3], steps=128)
    assert explanation["sample_attributions"].shape == (3, 3)
    assert explanation["drug_attributions"].shape == (3, 3)
    assert np.allclose(explanation["reference"] + explanation["sample_attributions"].sum(1) + explanation["drug_attributions"].sum(1), explanation["prediction"], atol=.08)
    saved = tmp_path / "dual.pt"; model.save(saved)
    assert np.allclose(DualEncoderPairModel.load(saved).predict(sample, drug), prediction)


def test_validation_does_not_replace_train_label_standardization():
    sample, drug, labels, pdos = _data()
    model = DualEncoderPairModel(epochs=2, seed=1).fit(sample, drug, labels, pdo_ids=pdos, validation_data=(sample[:8], drug[:8], labels[:8] + 100, pdos[:8]))
    assert model.label_mean == pytest.approx(labels.mean())


def test_loaded_source_model_can_be_fine_tuned_with_target_train_scaler(tmp_path):
    sample, drug, labels, pdos = _data()
    source = DualEncoderPairModel(epochs=2, seed=1).fit(sample, drug, labels, pdo_ids=pdos)
    saved = tmp_path / "source.pt"; source.save(saved)
    target_labels = labels + 20
    loaded = DualEncoderPairModel.load(saved)
    before = loaded.predict(sample, drug)
    # A zero learning rate isolates the calibration transform from optimization.
    loaded.config["learning_rate"] = 0.0
    tuned = loaded.fit(sample, drug, target_labels, pdo_ids=pdos, reset_label_scaler=True)
    assert tuned.label_mean == pytest.approx(target_labels.mean())
    assert np.allclose(tuned.predict(sample, drug), before)


@pytest.mark.parametrize("factory", [lambda: RidgePairModel().fit([[1]], [[1]], [1], sample_weight=[0]), lambda: DualEncoderPairModel.ranking_pairs([1], ["x", "y"])])
def test_input_validation(factory):
    with pytest.raises(ValueError): factory()
