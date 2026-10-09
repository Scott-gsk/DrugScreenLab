import copy

import pytest
import torch
from torch import nn

from drug_screen.models.exp046_five_arm import (
    AlignedPathwayMap,
    PathwayAxisContract,
    align_full_axis_pathway_map,
    build_exp046_arm_model,
    capture_cold_start_state,
    expand_mapped_pathway_map,
    restore_cold_start_state,
)
from drug_screen.models.pharmaformer_pathway import PharmaFormerWithPathwayBranch


def seed_cpu(seed):
    torch.random.default_generator.manual_seed(seed)


class TinyBackbone(nn.Module):
    def __init__(self, gene_features=6, drug_features=3, drug_hidden=4):
        super().__init__()
        self.feature_extractor = nn.Module()
        self.feature_extractor.gene_fc1 = nn.Linear(gene_features, 8)
        self.feature_extractor.smiles_fc = nn.Linear(drug_features, drug_hidden)
        self.output = nn.Linear(8 + drug_hidden, 1)

    def forward(self, gene_expr, smiles):
        gene = torch.relu(self.feature_extractor.gene_fc1(gene_expr))
        drug = torch.relu(self.feature_extractor.smiles_fc(smiles))
        return self.output(torch.cat((gene, drug), dim=1))


def toy_maps():
    contract = PathwayAxisContract(
        full_gene_count=6, mapped_gene_count=4, pathway_count=2)
    genes = ("G0", "G1", "G2", "G3", "G4", "G5")
    mapped = ("G0", "G2", "G4", "G5")
    names = ("HALLMARK_A", "HALLMARK_B")
    true_reduced = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]], dtype=torch.float32)
    ordinary_reduced = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)
    random_reduced = torch.tensor([[1, 0, 0, 1], [0, 1, 1, 0]], dtype=torch.float32)

    true_full = torch.zeros((2, 6), dtype=torch.float32)
    true_full[:, [0, 2, 4, 5]] = true_reduced
    true_map = align_full_axis_pathway_map(
        true_full,
        gene_axis=genes,
        mapped_gene_ids=mapped,
        pathway_names=names,
        contract=contract,
    )
    ordinary_map = expand_mapped_pathway_map(
        ordinary_reduced,
        gene_axis=genes,
        mapped_gene_ids=mapped,
        pathway_names=("slot_000", "slot_001"),
        contract=contract,
    )
    random_map = expand_mapped_pathway_map(
        random_reduced,
        gene_axis=genes,
        mapped_gene_ids=mapped,
        pathway_names=names,
        contract=contract,
    )
    return contract, genes, mapped, true_reduced, true_map, ordinary_map, random_map


def backbone_from(template):
    result = TinyBackbone()
    result.load_state_dict(copy.deepcopy(template.state_dict()))
    return result


def test_default_axis_contract_and_explicit_map_expansion_and_axis_rejection():
    assert PathwayAxisContract() == PathwayAxisContract(19_091, 4_366, 50)
    contract, genes, mapped, _, true_map, _, _ = toy_maps()

    assert true_map.incidence_matrix.shape == (2, 6)
    assert true_map.gene_axis == genes
    assert true_map.mapped_gene_ids == mapped
    assert torch.count_nonzero(true_map.incidence_matrix[:, [1, 3]]) == 0

    reduced = torch.ones((2, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match="follow the order"):
        expand_mapped_pathway_map(
            reduced,
            gene_axis=genes,
            mapped_gene_ids=("G2", "G0", "G4", "G5"),
            pathway_names=("A", "B"),
            contract=contract,
        )
    with pytest.raises(ValueError, match="absent from gene_axis"):
        expand_mapped_pathway_map(
            reduced,
            gene_axis=genes,
            mapped_gene_ids=("G0", "G2", "G4", "G99"),
            pathway_names=("A", "B"),
            contract=contract,
        )
    invalid_full = true_map.incidence_matrix.clone()
    invalid_full[0, 1] = 1
    with pytest.raises(ValueError, match="outside mapped_gene_ids"):
        align_full_axis_pathway_map(
            invalid_full,
            gene_axis=genes,
            mapped_gene_ids=mapped,
            pathway_names=true_map.pathway_names,
            contract=contract,
        )


def test_five_arm_construction_binds_the_requested_map_and_shares_branch_shape():
    _, _, _, true_reduced, true_map, ordinary_map, random_map = toy_maps()
    seed_cpu(41)
    template = TinyBackbone()

    p_a_backbone = backbone_from(template)
    p_a = build_exp046_arm_model("P-A", p_a_backbone)
    assert p_a is p_a_backbone

    p_c = build_exp046_arm_model(
        "P-C", backbone_from(template), ordinary_control_map=ordinary_map,
        rank=3, branch_seed=20260922, zero_initialize=False)
    p_r = build_exp046_arm_model(
        "P-R", backbone_from(template), frozen_random_map=random_map,
        rank=3, branch_seed=20260922, zero_initialize=False)
    p_t = build_exp046_arm_model(
        "P-T", backbone_from(template), true_pathway_map=true_map,
        rank=3, branch_seed=20260922, zero_initialize=False)
    p_n = build_exp046_arm_model(
        "P-N", backbone_from(template), true_pathway_map=true_map,
        rank=3, branch_seed=20260922, zero_initialize=False)

    assert isinstance(p_c, PharmaFormerWithPathwayBranch)
    assert isinstance(p_r, PharmaFormerWithPathwayBranch)
    assert isinstance(p_t, PharmaFormerWithPathwayBranch)
    assert isinstance(p_n, PharmaFormerWithPathwayBranch)
    assert p_c.pathway_names == ordinary_map.pathway_names
    assert p_r.pathway_names == random_map.pathway_names
    assert p_t.pathway_names == p_n.pathway_names == true_map.pathway_names
    torch.testing.assert_close(
        p_c.pathway_gene_weights,
        ordinary_map.incidence_matrix / ordinary_map.incidence_matrix.sum(dim=1, keepdim=True),
    )
    torch.testing.assert_close(
        p_r.pathway_gene_weights,
        random_map.incidence_matrix / random_map.incidence_matrix.sum(dim=1, keepdim=True),
    )
    torch.testing.assert_close(
        p_t.pathway_gene_weights,
        true_map.incidence_matrix / true_map.incidence_matrix.sum(dim=1, keepdim=True),
    )
    torch.testing.assert_close(p_n.pathway_gene_weights, p_t.pathway_gene_weights)

    branch_parameter_shapes = [
        [(name, tuple(parameter.shape)) for name, parameter in model.branch.named_parameters()]
        for model in (p_c, p_r, p_t, p_n)
    ]
    assert branch_parameter_shapes.count(branch_parameter_shapes[0]) == 4
    for model in (p_c, p_r, p_t, p_n):
        torch.testing.assert_close(model.branch.pathway_factors, p_t.branch.pathway_factors)
        torch.testing.assert_close(
            model.branch.drug_projection.weight, p_t.branch.drug_projection.weight)

    # The full-axis map is still the subset map placed at the declared gene IDs.
    torch.testing.assert_close(p_t.pathway_gene_weights[:, [0, 2, 4, 5]].sum(dim=1),
                               torch.ones(2))
    assert torch.equal(true_map.incidence_matrix[:, [0, 2, 4, 5]], true_reduced)


def test_all_pathway_arms_reconstruct_their_prediction():
    _, _, _, _, true_map, ordinary_map, random_map = toy_maps()
    seed_cpu(17)
    template = TinyBackbone()
    models = (
        build_exp046_arm_model("P-C", backbone_from(template),
                               ordinary_control_map=ordinary_map, rank=3,
                               branch_seed=17, zero_initialize=False),
        build_exp046_arm_model("P-R", backbone_from(template),
                               frozen_random_map=random_map, rank=3,
                               branch_seed=17, zero_initialize=False),
        build_exp046_arm_model("P-T", backbone_from(template),
                               true_pathway_map=true_map, rank=3,
                               branch_seed=17, zero_initialize=False),
        build_exp046_arm_model("P-N", backbone_from(template),
                               true_pathway_map=true_map, rank=3,
                               branch_seed=17, zero_initialize=False),
    )
    gene_expr = torch.tensor([[0.2, -0.4, 1.0, 0.3, 0.5, -0.1],
                              [0.9, 0.1, -0.3, 0.7, -0.2, 0.4]])
    smiles = torch.tensor([[0.1, 0.6, -0.2], [0.7, -0.1, 0.3]])

    for model in models:
        output = model.forward_components(gene_expr, smiles)
        reconstructed = (output.base_prediction
                         + output.pathway_contributions.sum(dim=1, keepdim=True))
        torch.testing.assert_close(output.prediction, reconstructed)
        assert output.pathway_contributions.shape == (2, 2)


def test_pn_cold_start_restore_copies_parameters_buffers_and_clears_optimizer_state():
    _, _, _, _, true_map, _, _ = toy_maps()
    seed_cpu(23)
    template = TinyBackbone()
    p_t_cold = build_exp046_arm_model(
        "P-T", backbone_from(template), true_pathway_map=true_map,
        rank=3, branch_seed=7, zero_initialize=False)
    p_n = build_exp046_arm_model(
        "P-N", backbone_from(template), true_pathway_map=true_map,
        rank=3, branch_seed=7, zero_initialize=False)
    cold_start = capture_cold_start_state(p_t_cold)
    optimizer = torch.optim.AdamW(p_n.parameters(), lr=0.01)

    gene_expr = torch.randn(3, 6)
    smiles = torch.randn(3, 3)
    optimizer.zero_grad(set_to_none=True)
    p_n(gene_expr, smiles).square().mean().backward()
    optimizer.step()
    assert optimizer.state
    with torch.no_grad():
        p_n.pathway_gene_weights.add_(0.25)

    restore_cold_start_state(p_n, cold_start, optimizer=optimizer)

    assert not optimizer.state
    assert all(parameter.grad is None for parameter in p_n.parameters())
    restored = p_n.state_dict()
    assert set(restored) == set(cold_start)
    for name, expected in cold_start.items():
        torch.testing.assert_close(restored[name].cpu(), expected, rtol=0, atol=0)


def test_arm_requires_its_own_map_and_rejects_map_substitution():
    _, _, _, _, true_map, ordinary_map, random_map = toy_maps()
    backbone = TinyBackbone()
    with pytest.raises(ValueError, match="frozen_random_map is required"):
        build_exp046_arm_model("P-R", backbone)
    with pytest.raises(ValueError, match="accepts only frozen_random_map"):
        build_exp046_arm_model(
            "P-R", TinyBackbone(), frozen_random_map=random_map,
            ordinary_control_map=ordinary_map)
    with pytest.raises(ValueError, match="without a pathway map"):
        build_exp046_arm_model("P-A", TinyBackbone(), true_pathway_map=true_map)
    with pytest.raises(ValueError, match="full gene axis"):
        build_exp046_arm_model(
            "P-T", TinyBackbone(gene_features=5), true_pathway_map=true_map)
    with pytest.raises(ValueError, match="rank must be a positive integer"):
        build_exp046_arm_model(
            "P-T", TinyBackbone(), true_pathway_map=true_map)
