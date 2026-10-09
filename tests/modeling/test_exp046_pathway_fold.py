import copy

import numpy as np
import pytest
import torch
from torch import nn
from types import SimpleNamespace

from drug_screen.models.exp046_five_arm import (
    PathwayAxisContract,
    align_full_axis_pathway_map,
    build_exp046_arm_model,
    capture_cold_start_state,
    expand_mapped_pathway_map,
)
from scripts.modeling.exp046_pathway_fold import (
    BranchInitialization,
    CandidateTraining,
    LabeledPairBatch,
    PairInputBatch,
    SourceFoldInitialization,
    ValidationSelectionState,
    _validate_split,
    build_fold_arm_model,
    compute_state_dict_sha256,
    fit_outer_fold,
    _fit_training_epochs,
    patient_group_indices,
    patient_training_objective,
    require_cuda_training_device,
    select_refit_epoch,
    update_validation_selection,
)
from scripts.modeling import exp036_pharmaformer_fold as exp036
from drug_screen.models.pharmaformer_target import configure_target_variant
from drug_screen.models.pharmaformer_pathway import PharmaFormerWithPathwayBranch


class TinyBackbone(nn.Module):
    def __init__(self, genes=6, drugs=3):
        super().__init__()
        self.feature_extractor = nn.Module()
        self.feature_extractor.gene_fc1 = nn.Linear(genes, 8)
        self.feature_extractor.smiles_fc = nn.Linear(drugs, 4)
        self.output = nn.Linear(12, 1)

    def forward(self, expression, drug_features):
        gene = torch.relu(self.feature_extractor.gene_fc1(expression))
        drug = torch.relu(self.feature_extractor.smiles_fc(drug_features))
        return self.output(torch.cat((gene, drug), dim=1))


class TinyOfficial:
    @staticmethod
    def CombinedModel(**_model_config):
        return TinyBackbone()


class SyntheticExp036Data:
    def __init__(self, batch, device):
        self.labels = batch.responses.detach().cpu().numpy().astype(np.float32)
        self.pair_patient = np.zeros(len(self.labels), dtype=np.int64)
        self.patients = ("synthetic-train-patient",)
        self.expression = batch.expression.to(device)
        self.drug_features = batch.drug_features.to(device)

    def tensor_pair(self, index):
        take = torch.as_tensor(index, dtype=torch.long, device=self.expression.device)
        return self.expression[take], self.drug_features[take]


def seeded_backbone(seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return TinyBackbone()


def toy_maps():
    contract = PathwayAxisContract(full_gene_count=6, mapped_gene_count=4, pathway_count=2)
    genes = ("G0", "G1", "G2", "G3", "G4", "G5")
    mapped = ("G0", "G2", "G4", "G5")
    pathways = ("H0", "H1")
    true_reduced = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]], dtype=torch.float32)
    control_reduced = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)
    random_reduced = torch.tensor([[1, 0, 0, 1], [0, 1, 1, 0]], dtype=torch.float32)
    true_full = torch.zeros((2, 6), dtype=torch.float32)
    true_full[:, [0, 2, 4, 5]] = true_reduced
    true_map = align_full_axis_pathway_map(
        true_full, gene_axis=genes, mapped_gene_ids=mapped,
        pathway_names=pathways, contract=contract)
    control_map = expand_mapped_pathway_map(
        control_reduced, gene_axis=genes, mapped_gene_ids=mapped,
        pathway_names=("slot0", "slot1"), contract=contract)
    random_map = expand_mapped_pathway_map(
        random_reduced, gene_axis=genes, mapped_gene_ids=mapped,
        pathway_names=pathways, contract=contract)
    return true_map, control_map, random_map


def source_for(backbone, fold="outer-0", target_fit=()):
    state = copy.deepcopy(backbone.state_dict())
    return SourceFoldInitialization(
        outer_fold_key=fold,
        state_dict=state,
        source_sha256="a" * 64,
        state_dict_sha256=compute_state_dict_sha256(state),
        target_fit_patient_ids=tuple(target_fit),
    )


def build(arm, source, maps, *, model_seed=13):
    true_map, control_map, random_map = maps
    return build_fold_arm_model(
        arm,
        backbone_factory=TinyBackbone,
        source=source,
        expected_outer_fold_key=source.outer_fold_key,
        model_seed=model_seed,
        branch_initialization=(None if arm == "P-A" else
                               BranchInitialization(rank=3, branch_seed=17,
                                                    zero_initialize=False)),
        ordinary_control_map=control_map if arm == "P-C" else None,
        frozen_random_map=random_map if arm == "P-R" else None,
        true_pathway_map=true_map if arm in ("P-T", "P-N") else None,
    )


def test_p_a_fold_builder_matches_exp036_source_full_backbone_path():
    source_backbone = seeded_backbone(91)
    source = source_for(source_backbone)
    actual = build_fold_arm_model(
        "P-A", backbone_factory=TinyBackbone, source=source,
        expected_outer_fold_key="outer-0", model_seed=27,
        branch_initialization=None, ordinary_control_map=None,
        frozen_random_map=None, true_pathway_map=None)
    expected_backbone = seeded_backbone(27)
    expected_backbone.load_state_dict(source.state_dict, strict=True)
    expected = configure_target_variant(expected_backbone, "source_full", adapter_rank=1)
    assert type(actual) is type(expected) is TinyBackbone
    assert all(parameter.requires_grad for parameter in actual.parameters())
    for name, tensor in actual.state_dict().items():
        torch.testing.assert_close(tensor, expected.state_dict()[name], rtol=0, atol=0)
    expression = torch.randn(2, 6)
    drug_features = torch.randn(2, 3)
    torch.testing.assert_close(actual(expression, drug_features),
                               expected(expression, drug_features))


def test_fold_builder_binds_each_pathway_arm_to_its_declared_map_and_same_branch_shape():
    maps = toy_maps()
    initial = seeded_backbone(5)
    source = source_for(initial)
    models = {arm: build(arm, source, maps) for arm in ("P-A", "P-C", "P-R", "P-T", "P-N")}
    assert models["P-A"] is not None
    assert all(isinstance(models[arm], PharmaFormerWithPathwayBranch)
               for arm in ("P-C", "P-R", "P-T", "P-N"))
    def normalized_map(pathway_map):
        return (pathway_map.incidence_matrix
                / pathway_map.incidence_matrix.sum(dim=1, keepdim=True))

    torch.testing.assert_close(models["P-C"].pathway_gene_weights, normalized_map(maps[1]))
    torch.testing.assert_close(models["P-R"].pathway_gene_weights, normalized_map(maps[2]))
    torch.testing.assert_close(models["P-T"].pathway_gene_weights, normalized_map(maps[0]))
    torch.testing.assert_close(models["P-N"].pathway_gene_weights, normalized_map(maps[0]))
    torch.testing.assert_close(models["P-N"].pathway_gene_weights,
                               models["P-T"].pathway_gene_weights)
    signatures = [
        [(name, tuple(parameter.shape)) for name, parameter in models[arm].branch.named_parameters()]
        for arm in ("P-C", "P-R", "P-T", "P-N")
    ]
    assert all(signature == signatures[0] for signature in signatures[1:])


def test_p_n_resets_full_backbone_branch_and_buffers_to_its_fold_cold_start():
    maps = toy_maps()
    source_backbone = seeded_backbone(81)
    source_state = {name: torch.full_like(value, 11) for name, value in source_backbone.state_dict().items()}
    source = SourceFoldInitialization(
        outer_fold_key="outer-0", state_dict=source_state,
        source_sha256="b" * 64,
        state_dict_sha256=compute_state_dict_sha256(source_state),
        target_fit_patient_ids=())

    cold_backbone = seeded_backbone(29)
    cold_model = build_exp046_arm_model(
        "P-N", cold_backbone, true_pathway_map=maps[0], rank=3,
        branch_seed=17, zero_initialize=False)
    expected_cold_state = capture_cold_start_state(cold_model)
    pn = build("P-N", source, maps, model_seed=29)
    pt = build("P-T", source, maps, model_seed=29)

    assert set(pn.state_dict()) == set(expected_cold_state)
    for name, expected in expected_cold_state.items():
        torch.testing.assert_close(pn.state_dict()[name], expected, rtol=0, atol=0)
    assert any(torch.all(value == 11) for value in pt.backbone.state_dict().values())
    assert not all(torch.all(value == 11) for value in pn.backbone.state_dict().values())


def test_pair_inputs_have_no_label_and_split_receipt_checks_target_exposure():
    expression = torch.zeros((4, 6))
    drugs = torch.zeros((4, 3))
    train = LabeledPairBatch(
        expression, drugs, ("train0", "train0", "train1", "train1"),
        ("d0", "d1", "d0", "d1"), torch.tensor([0.1, 0.4, 0.2, 0.7]))
    valid = LabeledPairBatch(
        expression[:2], drugs[:2], ("valid0", "valid0"),
        ("d0", "d1"), torch.tensor([0.1, 0.5]))
    source = source_for(seeded_backbone(7), target_fit=("train0",))
    _validate_split(train, valid, source, "outer-0")

    outer = PairInputBatch(expression[:2], drugs[:2], ("outer0", "outer0"), ("d0", "d1"))
    outer.validate()
    assert not hasattr(outer, "responses")
    with pytest.raises(TypeError):
        PairInputBatch(expression[:2], drugs[:2], ("outer0", "outer0"),
                       ("d0", "d1"), responses=torch.zeros(2))

    leaking_source = source_for(seeded_backbone(8), target_fit=("valid0",))
    with pytest.raises(ValueError, match="subset of this training slice"):
        _validate_split(train, valid, leaking_source, "outer-0")
    overlapping_validation = LabeledPairBatch(
        expression[:2], drugs[:2], ("train0", "validation0"),
        ("d2", "d3"), torch.tensor([0.1, 0.5]))
    with pytest.raises(ValueError, match="must be disjoint"):
        _validate_split(train, overlapping_validation, source, "outer-0")


def test_source_receipt_binds_actual_state_dict_content():
    state = copy.deepcopy(seeded_backbone(12).state_dict())
    digest = compute_state_dict_sha256(state)
    source = SourceFoldInitialization(
        outer_fold_key="outer-0", state_dict=state, source_sha256="c" * 64,
        state_dict_sha256=digest, target_fit_patient_ids=())
    source.validate()

    forged_receipt = SourceFoldInitialization(
        outer_fold_key="outer-0", state_dict=state, source_sha256="c" * 64,
        state_dict_sha256="d" * 64, target_fit_patient_ids=())
    with pytest.raises(ValueError, match="does not match"):
        forged_receipt.validate()

    first_value = next(iter(state.values()))
    first_value.add_(1)
    with pytest.raises(ValueError, match="does not match"):
        source.validate()


def test_patient_grouped_objective_and_validation_epoch_helper():
    groups = patient_group_indices(("p1", "p1", "p2", "p2", "p2"))
    assert [group.tolist() for group in groups] == [[0, 1], [2, 3, 4]]
    predicted = torch.tensor([0.1, 0.8, 0.3])
    observed = torch.tensor([0.0, 1.0, 0.5])
    objective = patient_training_objective(predicted, observed, ranking_weight=0.25)
    expected = torch.nn.functional.smooth_l1_loss(predicted, observed) + 0.25 * (
        torch.nn.functional.softplus(torch.tensor(-0.7))
        + torch.nn.functional.softplus(torch.tensor(-0.2))
        + torch.nn.functional.softplus(torch.tensor(-0.5))
    ) / 3
    torch.testing.assert_close(objective, expected)
    assert select_refit_epoch((3, 5, 10)) == 5
    with pytest.raises(ValueError, match="positive integers"):
        select_refit_epoch((3, 0))


def test_validation_epoch_updates_choose_best_score_and_stop_after_patience():
    state = ValidationSelectionState(best_score=-float("inf"), best_epoch=0, stale_epochs=0)
    stopped = False
    for epoch, score in enumerate((0.2, 0.4, 0.35), start=1):
        state, stopped = update_validation_selection(
            state, score=score, epoch=epoch, min_improvement=0.01, patience=2)
        if stopped:
            break
    assert not stopped
    assert state.best_epoch == 2
    assert state.best_score == pytest.approx(0.4)
    assert state.stale_epochs == 1
    state, stopped = update_validation_selection(
        state, score=0.39, epoch=4, min_improvement=0.01, patience=2)
    assert stopped
    assert state.best_epoch == 2


def test_outer_entry_rejects_labeled_batch_even_though_it_subclasses_pair_inputs():
    expression = torch.zeros((2, 6))
    drug_features = torch.zeros((2, 3))
    labeled = LabeledPairBatch(expression, drug_features, ("p0", "p0"),
                               ("d0", "d1"), torch.zeros(2))
    with pytest.raises(TypeError, match="exactly PairInputBatch"):
        fit_outer_fold(
            labeled, labeled, arm="P-A", backbone_factory=TinyBackbone,
            source=None, outer_fold_key="outer-0", model_seed=1,
            branch_initialization=None, ordinary_control_map=None,
            frozen_random_map=None, true_pathway_map=None,
            settings=None, selected_epochs=1, device=torch.device("cpu"))


def test_fold_training_entry_rejects_cpu_without_attempting_fallback():
    with pytest.raises(ValueError, match="CPU fallback is disabled"):
        require_cuda_training_device(torch.device("cpu"))


@pytest.mark.parametrize("arm", ("P-A", "P-C", "P-R", "P-T", "P-N"))
def test_each_arm_synthetic_cuda_forward_backward_has_finite_gradients(arm):
    device = torch.device("cuda:0")
    require_cuda_training_device(device)
    maps = toy_maps()
    source = source_for(seeded_backbone(41))
    model = build(arm, source, maps).to(device)
    expression = torch.tensor(
        [[0.2, -0.4, 1.0, 0.3, 0.5, -0.1], [0.9, 0.1, -0.3, 0.7, -0.2, 0.4]],
        dtype=torch.float32, device=device)
    drug_features = torch.tensor([[0.1, 0.6, -0.2], [0.7, -0.1, 0.3]],
                                 dtype=torch.float32, device=device)
    observed = torch.tensor([0.2, 0.8], dtype=torch.float32, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = model(expression, drug_features).float().reshape(-1)
        loss = patient_training_objective(prediction, observed, ranking_weight=0.2)
    assert prediction.dtype == torch.float32
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(loss)
    loss.backward()
    trainable_gradients = [parameter.grad for parameter in model.parameters()
                           if parameter.requires_grad]
    assert trainable_gradients and all(gradient is not None for gradient in trainable_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in trainable_gradients)


def test_p_a_single_step_and_outer_predictions_match_exp036_on_synthetic_cuda():
    device = torch.device("cuda:0")
    require_cuda_training_device(device)
    seed = 1207
    source_backbone = seeded_backbone(101)
    source = source_for(source_backbone)
    train = LabeledPairBatch(
        expression=torch.tensor(
            [[0.1, 0.4, -0.2, 0.8, 0.5, -0.1],
             [0.7, -0.3, 0.9, 0.2, -0.4, 0.6],
             [-0.5, 0.2, 0.3, -0.1, 0.8, 0.4],
             [0.9, 0.5, -0.6, 0.1, 0.3, -0.2]], dtype=torch.float32),
        drug_features=torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, -0.2, 0.1], [-0.3, 0.6, 0.2], [0.7, 0.1, -0.5]],
            dtype=torch.float32),
        patient_ids=("train0", "train0", "train0", "train0"),
        drug_ids=("d0", "d1", "d2", "d3"),
        responses=torch.tensor([0.15, 0.7, 0.4, 0.9], dtype=torch.float32),
    )
    settings = CandidateTraining(
        learning_rate=0.002, weight_decay=0.0001, ranking_weight=0.2,
        max_epochs=1, patience=1, batch_size=4, min_improvement=0.0)

    actual_model = build_fold_arm_model(
        "P-A", backbone_factory=TinyBackbone, source=source,
        expected_outer_fold_key="outer-0", model_seed=seed,
        branch_initialization=None, ordinary_control_map=None,
        frozen_random_map=None, true_pathway_map=None)
    _fit_training_epochs(actual_model, train, validation=None, epochs=1,
                         settings=settings, seed=seed, device=device)
    for parameter in actual_model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()

    candidate = exp036.Candidate(
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        adapter_rank=0,
        ranking_weight=settings.ranking_weight,
    )
    fake_data = SyntheticExp036Data(train, device)
    checkpoint = {
        "model_config": {
            "gene_input_size": 6,
            "gene_hidden_size": 8,
            "drug_hidden_size": 4,
            "feature_dim": 4,
            "nhead": 1,
            "num_layers": 1,
            "dim_feedforward": 8,
            "dropout": 0.0,
        },
        "model_state_dict": source.state_dict,
    }
    reference_model, reference_receipt = exp036.train(
        fake_data, TinyOfficial, checkpoint, "source_full", candidate,
        np.arange(len(train.patient_ids), dtype=np.int64), None,
        seed=seed, device=device, batch_size=settings.batch_size,
        max_epochs=1, patience=1, fixed_epochs=1)
    for name, actual in actual_model.state_dict().items():
        torch.testing.assert_close(actual, reference_model.state_dict()[name],
                                   rtol=2e-5, atol=2e-6)
    assert reference_receipt["best_epoch"] == 1

    outer = PairInputBatch(
        expression=torch.tensor([[0.6, -0.1, 0.3, 0.9, -0.4, 0.2],
                                 [-0.2, 0.8, 0.1, -0.5, 0.7, 0.4]], dtype=torch.float32),
        drug_features=torch.tensor([[0.3, 0.2, -0.1], [-0.4, 0.5, 0.7]], dtype=torch.float32),
        patient_ids=("outer0", "outer0"),
        drug_ids=("d0", "d1"),
    )
    outer_predictions, _ = fit_outer_fold(
        train, outer, arm="P-A", backbone_factory=TinyBackbone,
        source=source, outer_fold_key="outer-0", model_seed=seed,
        branch_initialization=None, ordinary_control_map=None,
        frozen_random_map=None, true_pathway_map=None,
        settings=settings, selected_epochs=1, device=device)
    fake_outer_data = SimpleNamespace(
        tensor_pair=lambda index: (
            outer.expression.to(device)[torch.as_tensor(index, device=device)],
            outer.drug_features.to(device)[torch.as_tensor(index, device=device)],
        )
    )
    reference_standardized = exp036.predict(
        reference_model, fake_outer_data,
        np.arange(len(outer.patient_ids), dtype=np.int64), device, settings.batch_size)
    reference_predictions = (
        reference_standardized * reference_receipt["target_response_scale_train_only"]
        + reference_receipt["target_response_mean_train_only"])
    np.testing.assert_allclose(outer_predictions.predicted_responses,
                               reference_predictions, rtol=2e-5, atol=2e-6)
