import pytest
import torch
from torch import nn

from drug_screen.models.exp045_dependency_heads import (
    PharmaFormerAuxiliaryOutput,
    PharmaFormerDependencyOutput,
    PharmaFormerMultiTaskOutput,
    PharmaFormerWithDependencyHeads,
)


class TinyPharmaFormer(nn.Module):
    """Synthetic surface matching the official gene and drug encoder API."""

    def __init__(self, gene_features=5, expression_hidden=6,
                 smiles_features=4, drug_hidden=3):
        super().__init__()
        self.feature_extractor = nn.Module()
        self.feature_extractor.gene_fc1 = nn.Linear(gene_features, expression_hidden)
        self.feature_extractor.gene_fc2 = nn.Linear(expression_hidden, expression_hidden)
        self.feature_extractor.smiles_fc = nn.Linear(smiles_features, drug_hidden)
        self.transformer = nn.Linear(expression_hidden + drug_hidden, 1)

    def forward(self, gene_expr, smiles):
        extractor = self.feature_extractor
        expression = torch.relu(extractor.gene_fc1(gene_expr))
        expression = torch.relu(extractor.gene_fc2(expression))
        drug = torch.relu(extractor.smiles_fc(smiles))
        return self.transformer(torch.cat((expression, drug), dim=1))


def make_model(*, depmap_dim=3, nature_dim=7):
    return PharmaFormerWithDependencyHeads(
        TinyPharmaFormer(),
        depmap_target_dim=depmap_dim,
        nature_target_dim=nature_dim,
    )


def test_wrapper_preserves_original_drug_sensitivity_forward():
    torch.manual_seed(5)
    backbone = TinyPharmaFormer()
    model = PharmaFormerWithDependencyHeads(
        backbone, depmap_target_dim=2, nature_target_dim=4)
    gene_expr, smiles = torch.randn(3, 5), torch.randn(3, 4)

    expected = backbone(gene_expr, smiles)
    result = model(gene_expr, smiles)

    assert isinstance(result, PharmaFormerMultiTaskOutput)
    torch.testing.assert_close(result.drug_sensitivity, expected, rtol=0, atol=0)
    assert result.drug_sensitivity.shape == (3, 1)


def test_domain_heads_share_encoder_and_allow_independent_target_dimensions():
    model = make_model(depmap_dim=2, nature_dim=9)
    gene_expr = torch.randn(4, 5)

    representation = model.encode_expression(gene_expr)
    expected_representation = torch.relu(
        model.backbone.feature_extractor.gene_fc2(
            torch.relu(model.backbone.feature_extractor.gene_fc1(gene_expr))))
    predictions = model.predict_dependency_heads(gene_expr)
    full_output = model(gene_expr, torch.randn(4, 4))

    assert isinstance(predictions, PharmaFormerDependencyOutput)
    assert representation.shape == (4, 6)
    torch.testing.assert_close(representation, expected_representation)
    assert predictions.depmap_chronos.shape == (4, 2)
    assert predictions.nature_crispr_lfc.shape == (4, 9)
    assert full_output.depmap_chronos.shape == (4, 2)
    assert full_output.nature_crispr_lfc.shape == (4, 9)
    assert model.depmap_chronos_head is not model.nature_crispr_lfc_head


def test_auxiliary_heads_reconstruct_the_same_ordered_expression_axis():
    model = make_model(depmap_dim=2, nature_dim=9)
    gene_expr = torch.randn(4, 5)

    output = model.predict_auxiliary_heads(gene_expr)

    assert isinstance(output, PharmaFormerAuxiliaryOutput)
    assert output.expression_reconstruction.shape == gene_expr.shape
    assert model.expression_reconstruction_head.out_features == model.gene_input_dim
    output.expression_reconstruction.square().mean().backward()
    assert model.expression_reconstruction_head.weight.grad is not None
    assert model.backbone.feature_extractor.gene_fc1.weight.grad is not None


def test_drug_sensitivity_inference_is_independent_of_auxiliary_heads():
    torch.manual_seed(11)
    model = make_model()
    gene_expr, smiles = torch.randn(2, 5), torch.randn(2, 4)
    expected = model.predict_drug_sensitivity(gene_expr, smiles)

    with torch.no_grad():
        for parameter in model.depmap_chronos_head.parameters():
            parameter.fill_(123.0)
        for parameter in model.nature_crispr_lfc_head.parameters():
            parameter.fill_(-456.0)

    actual = model.predict_drug_sensitivity(gene_expr, smiles)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("depmap_dim,nature_dim", [
    (0, 2),
    (2, -1),
    (True, 3),
    (3, 2.5),
])
def test_rejects_invalid_output_dimensions(depmap_dim, nature_dim):
    with pytest.raises(ValueError, match="positive integer"):
        make_model(depmap_dim=depmap_dim, nature_dim=nature_dim)


@pytest.mark.parametrize("gene_expr,smiles,error", [
    ([[0.0] * 5], torch.randn(1, 4), "gene_expr must be a torch tensor"),
    (torch.randn(2, 1, 5), torch.randn(2, 4), "gene_expr must be a 2D"),
    (torch.randn(2, 4), torch.randn(2, 4), "gene_expr feature count"),
    (torch.randn(2, 5), torch.randn(1, 4), "same batch size"),
    (torch.randn(2, 5), torch.randn(2, 3), "smiles feature count"),
    (torch.ones(2, 5, dtype=torch.int64), torch.ones(2, 4), "gene_expr must use"),
    (torch.ones(2, 5), torch.ones(2, 4, dtype=torch.int64), "smiles must use"),
    (torch.ones(2, 5), torch.ones(2, 4, dtype=torch.float64), "share device and dtype"),
])
def test_validates_drug_inference_inputs(gene_expr, smiles, error):
    with pytest.raises((TypeError, ValueError), match=error):
        make_model().predict_drug_sensitivity(gene_expr, smiles)


@pytest.mark.parametrize("gene_expr,error", [
    ([[0.0] * 5], "gene_expr must be a torch tensor"),
    (torch.randn(2, 1, 5), "gene_expr must be a 2D"),
    (torch.randn(2, 4), "gene_expr feature count"),
    (torch.ones(2, 5, dtype=torch.int64), "gene_expr must use"),
])
def test_validates_expression_only_dependency_inputs(gene_expr, error):
    with pytest.raises((TypeError, ValueError), match=error):
        make_model().predict_dependency_heads(gene_expr)
