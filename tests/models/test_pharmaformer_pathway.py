import pytest
import torch
from torch import nn

from drug_screen.models.pharmaformer_pathway import (
    PharmaFormerPathwayOutput,
    PharmaFormerWithPathwayBranch,
)


class TinyBackbone(nn.Module):
    def __init__(self, gene_features=5, drug_features=4, drug_hidden=3):
        super().__init__()
        self.feature_extractor = nn.Module()
        self.feature_extractor.gene_fc1 = nn.Linear(gene_features, 6)
        self.feature_extractor.smiles_fc = nn.Linear(drug_features, drug_hidden)
        self.output = nn.Linear(6 + drug_hidden, 1)

    def forward(self, gene_expr, smiles):
        gene = torch.relu(self.feature_extractor.gene_fc1(gene_expr))
        drug = torch.relu(self.feature_extractor.smiles_fc(smiles))
        return self.output(torch.cat((gene, drug), dim=1))


def make_incidence():
    return torch.tensor([
        [1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1],
    ], dtype=torch.float32)


def test_pathway_buffer_is_row_normalized_and_random_map_has_same_interface():
    true_map = make_incidence()
    random_map = torch.tensor([
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
    ], dtype=torch.float32)
    true_model = PharmaFormerWithPathwayBranch(TinyBackbone(), true_map)
    random_model = PharmaFormerWithPathwayBranch(TinyBackbone(), random_map)

    assert "pathway_gene_weights" in dict(true_model.named_buffers())
    assert not true_model.pathway_gene_weights.requires_grad
    torch.testing.assert_close(true_model.pathway_gene_weights.sum(dim=1), torch.ones(3))
    torch.testing.assert_close(random_model.pathway_gene_weights.sum(dim=1), torch.ones(3))

    gene, smiles = torch.randn(2, 5), torch.randn(2, 4)
    true_result = true_model(gene, smiles, return_components=True)
    random_result = random_model(gene, smiles, return_components=True)
    assert isinstance(true_result, PharmaFormerPathwayOutput)
    assert true_result.prediction.shape == random_result.prediction.shape == (2, 1)
    assert true_result.pathway_contributions.shape == random_result.pathway_contributions.shape == (2, 3)


def test_contributions_reconstruct_final_prediction():
    torch.manual_seed(13)
    model = PharmaFormerWithPathwayBranch(TinyBackbone(), make_incidence(), rank=3)
    gene, smiles = torch.randn(4, 5), torch.randn(4, 4)

    components = model.forward_components(gene, smiles)
    reconstructed = components.base_prediction + components.pathway_contributions.sum(dim=1, keepdim=True)
    torch.testing.assert_close(components.prediction, reconstructed)
    torch.testing.assert_close(model(gene, smiles), components.prediction)


def test_zero_initialized_branch_preserves_base_and_still_receives_gradients():
    torch.manual_seed(17)
    backbone = TinyBackbone()
    model = PharmaFormerWithPathwayBranch(
        backbone, make_incidence(), rank=3, zero_initialize=True)
    gene, smiles = torch.randn(4, 5), torch.randn(4, 4)
    expected = backbone(gene, smiles)

    components = model.forward_components(gene, smiles)
    torch.testing.assert_close(components.prediction, expected, rtol=0, atol=0)
    components.prediction.sum().backward()

    assert model.branch.pathway_factors.requires_grad
    assert model.branch.pathway_factors.grad is not None
    assert model.branch.pathway_factors.grad.abs().sum() > 0


def test_pathway_contributions_depend_on_drug_encoding():
    torch.manual_seed(23)
    model = PharmaFormerWithPathwayBranch(TinyBackbone(), make_incidence(), rank=4)
    with torch.no_grad():
        model.backbone.feature_extractor.smiles_fc.weight.fill_(0.2)
        model.backbone.feature_extractor.smiles_fc.bias.fill_(0.5)
    gene = torch.tensor([[0.4, 1.2, -0.3, 0.8, 0.2]])
    smiles_a = torch.tensor([[0.0, 0.2, 0.4, 0.6]])
    smiles_b = torch.tensor([[1.1, 0.8, 0.5, 0.2]])

    first = model.forward_components(gene, smiles_a).pathway_contributions
    second = model.forward_components(gene, smiles_b).pathway_contributions
    assert not torch.allclose(first, second)


@pytest.mark.parametrize("matrix", [
    torch.ones(5),
    torch.tensor([[1, 0], [0, 0]], dtype=torch.float32),
    torch.tensor([[1.0, 0.5]], dtype=torch.float32),
    torch.tensor([[1.0, float("nan")]], dtype=torch.float32),
])
def test_rejects_malformed_incidence_matrices(matrix):
    with pytest.raises(ValueError, match="incidence_matrix|pathway row"):
        PharmaFormerWithPathwayBranch(TinyBackbone(), matrix)


def test_rejects_incidence_gene_axis_mismatch_and_incompatible_backbone():
    with pytest.raises(ValueError, match="gene count"):
        PharmaFormerWithPathwayBranch(TinyBackbone(), torch.ones(2, 4))

    class MissingSmilesEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_extractor = nn.Module()

        def forward(self, gene_expr, smiles):
            return gene_expr[:, :1]

    with pytest.raises(ValueError, match="smiles_fc"):
        PharmaFormerWithPathwayBranch(MissingSmilesEncoder(), make_incidence())


@pytest.mark.parametrize("gene,smiles,match", [
    (torch.randn(2, 4), torch.randn(2, 4), "gene_expr feature count"),
    (torch.randn(2, 5), torch.randn(2, 3), "smiles feature count"),
    (torch.randn(2, 5), torch.randn(1, 4), "same non-empty batch"),
    (torch.full((2, 5), float("inf")), torch.randn(2, 4), "finite"),
])
def test_validates_input_shapes_and_finiteness(gene, smiles, match):
    model = PharmaFormerWithPathwayBranch(TinyBackbone(), make_incidence())
    with pytest.raises(ValueError, match=match):
        model(gene, smiles)


def test_rejects_backbone_that_does_not_return_one_scalar_per_sample():
    class VectorBackbone(TinyBackbone):
        def forward(self, gene_expr, smiles):
            return torch.zeros((gene_expr.shape[0], 2), device=gene_expr.device)

    model = PharmaFormerWithPathwayBranch(VectorBackbone(), make_incidence())
    with pytest.raises(ValueError, match="one scalar prediction"):
        model(torch.randn(2, 5), torch.randn(2, 4))
