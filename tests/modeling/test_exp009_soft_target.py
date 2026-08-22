from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from drug_screen.foundation.soft_target import (
    MechanismResidual,
    SoftTargetHead,
    count_trainable_parameters,
)


def test_soft_target_head_returns_bounded_probability_and_confidence():
    model = SoftTargetHead(structure_dim=8, target_dim=5)

    probabilities, confidence = model(torch.randn(3, 8))

    assert probabilities.shape == (3, 5)
    assert confidence.shape == (3, 1)
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert torch.all((confidence >= 0.0) & (confidence <= 1.0))


def test_mechanism_residual_zero_gate_is_strict_baseline_equivalence():
    model = MechanismResidual(feature_dim=7, hidden_dim=4, n_genes=6)
    features = torch.randn(3, 7)
    baseline = torch.randn(3, 6)

    output = model(features, baseline)

    assert model.raw_gamma.item() == 0.0
    torch.testing.assert_close(output, baseline, rtol=0.0, atol=1e-6)


def test_mechanism_residual_gate_magnitude_is_bounded_at_point_zero_five():
    model = MechanismResidual(feature_dim=1, hidden_dim=1, n_genes=1)
    features = torch.ones(1, 1)
    baseline = torch.zeros(1, 1)
    with torch.no_grad():
        model.hidden[0].weight.fill_(1.0)
        model.hidden[0].bias.zero_()
        model.output.weight.fill_(1.0)
        model.raw_gamma.fill_(1e6)

    residual = model.output(model.hidden(features))
    gamma = (model(features, baseline) - baseline) / residual

    assert torch.all(torch.abs(gamma) <= 0.05)


def test_mechanism_residual_accepts_default_385_feature_width():
    model = MechanismResidual()

    output = model(torch.randn(2, 385), torch.randn(2, 978))

    assert output.shape == (2, 978)


@pytest.mark.parametrize("features", [torch.randn(385), torch.randn(2, 3, 385)])
def test_mechanism_residual_rejects_non_two_dimensional_features(features):
    model = MechanismResidual()

    with pytest.raises(ValueError, match="rank-2"):
        model(features, torch.randn(2, 978))


def test_mechanism_residual_rejects_wrong_feature_width():
    model = MechanismResidual()

    with pytest.raises(ValueError, match="feature width"):
        model(torch.randn(2, 384), torch.randn(2, 978))


def test_mechanism_residual_output_layer_is_zero_initialized():
    model = MechanismResidual(feature_dim=7, hidden_dim=4, n_genes=6)

    assert torch.count_nonzero(model.output.weight).item() == 0
    assert model.output.bias is not None
    assert torch.count_nonzero(model.output.bias).item() == 0


def test_count_trainable_parameters_excludes_frozen_parameters():
    model = SoftTargetHead(structure_dim=4, target_dim=3)
    expected = sum(parameter.numel() for parameter in model.parameters())
    model.target_logits.bias.requires_grad_(False)

    assert count_trainable_parameters(model) == expected - model.target_logits.bias.numel()
