import torch

from drug_screen.models.conditional_perturbation import SharedEffectInteraction


def test_forward_shapes_and_state_ablation_uses_same_parameters():
    torch.manual_seed(1)
    model = SharedEffectInteraction(n_genes=5, drug_features=7, hidden=8, rank=3)
    state = torch.randn(4, 5)
    drug = torch.randn(4, 7)
    exposure = torch.randn(4, 2)
    conditional = model(state, drug, exposure, use_state=True)
    ablated = model(state, drug, exposure, use_state=False)
    assert conditional.shape == ablated.shape == (4, 5)
    assert not torch.allclose(conditional, ablated)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_state_ablation_prediction_does_not_depend_on_state_values():
    model = SharedEffectInteraction(n_genes=3, drug_features=4, hidden=6, rank=2)
    drug, exposure = torch.randn(2, 4), torch.randn(2, 2)
    first = model(torch.zeros(2, 3), drug, exposure, use_state=False)
    second = model(torch.randn(2, 3) * 100, drug, exposure, use_state=False)
    assert torch.equal(first, second)
