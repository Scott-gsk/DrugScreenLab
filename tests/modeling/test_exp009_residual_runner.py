from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from drug_screen.foundation.soft_target import (
    build_exp009_residual_wrapper,
    count_trainable_parameters,
)
from scripts.modeling.run_exp009_full_mvp import metrics


class _OfficialStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 6)

    def forward(self, data):
        baseline = self.backbone(data)
        return baseline, baseline + 1.0, baseline


def test_metrics_reports_exact_row_macro_spearman_for_identical_delta978_rows():
    truth = np.array([[1.0, 3.0, 2.0], [4.0, 6.0, 5.0]], dtype=np.float32)

    result = metrics(truth, truth.copy())

    assert result["rows"] == 2
    assert result["genes"] == 3
    assert result["row_macro_spearman"] == pytest.approx(1.0)


def test_exp009_wrapper_freezes_every_official_parameter_and_preserves_baseline_at_init():
    official = _OfficialStub()
    wrapper = build_exp009_residual_wrapper(
        official,
        variant="C",
        structure_dim=4,
        soft_target_dim=64,
        hidden_dim=5,
        n_genes=6,
    )
    data = torch.randn(3, 4)
    soft_targets = torch.randn(3, 64)

    result = wrapper(data, structure_features=data, soft_targets=soft_targets)

    assert all(not parameter.requires_grad for parameter in official.parameters())
    assert wrapper.official_parameters_frozen is True
    torch.testing.assert_close(result[2], official(data)[2], rtol=0.0, atol=0.0)
    assert wrapper.residual.raw_gamma.item() == 0.0
    assert torch.count_nonzero(wrapper.residual.output.weight).item() == 0


def test_exp009_structure_only_and_soft_target_residuals_have_equal_trainable_parameters():
    structure = build_exp009_residual_wrapper(
        _OfficialStub(), variant="B", structure_dim=4, soft_target_dim=64, hidden_dim=5, n_genes=6
    )
    soft_target = build_exp009_residual_wrapper(
        _OfficialStub(), variant="C", structure_dim=4, soft_target_dim=64, hidden_dim=5, n_genes=6
    )

    assert count_trainable_parameters(structure) == count_trainable_parameters(soft_target)
    assert not any(parameter.requires_grad for parameter in structure.official.parameters())
    assert not any(parameter.requires_grad for parameter in soft_target.official.parameters())


def test_zero_gate_and_zero_output_layer_produce_no_residual_gradients_before_activation():
    wrapper = build_exp009_residual_wrapper(
        _OfficialStub(), variant="C", structure_dim=4, soft_target_dim=64, hidden_dim=5, n_genes=6
    )
    prediction = wrapper(
        torch.randn(3, 4), structure_features=torch.randn(3, 4), soft_targets=torch.randn(3, 64)
    )[2]
    prediction.square().mean().backward()

    audit = wrapper.gradient_audit()

    assert audit["residual_gradient_norms"] == {
        name: 0.0 for name, _ in wrapper.residual.named_parameters()
    }


def test_nonzero_gate_after_equivalence_yields_output_layer_gradient():
    wrapper = build_exp009_residual_wrapper(
        _OfficialStub(), variant="C", structure_dim=4, soft_target_dim=64, hidden_dim=5, n_genes=6
    )
    with torch.no_grad():
        wrapper.residual.raw_gamma.fill_(1e-3)
    prediction = wrapper(
        torch.randn(3, 4), structure_features=torch.randn(3, 4), soft_targets=torch.randn(3, 64)
    )[2]
    prediction.square().mean().backward()

    audit = wrapper.gradient_audit()

    assert audit["residual_gradient_norms"]["output.weight"] > 0.0
    assert audit["residual_gradient_norms"]["output.bias"] > 0.0


def test_exp009_wrapper_gradient_audit_excludes_official_backbone():
    official = _OfficialStub()
    wrapper = build_exp009_residual_wrapper(
        official,
        variant="B",
        structure_dim=4,
        soft_target_dim=64,
        hidden_dim=5,
        n_genes=6,
    )
    prediction = wrapper(torch.randn(3, 4), structure_features=torch.randn(3, 4))[2]
    prediction.square().mean().backward()

    audit = wrapper.gradient_audit()

    assert audit["official_trainable_parameter_count"] == 0
    assert audit["official_parameters_with_grad"] == []
    assert audit["residual_trainable_parameter_count"] == count_trainable_parameters(wrapper.residual)
