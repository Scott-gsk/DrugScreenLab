from types import SimpleNamespace

import pytest
import torch
from torch import nn

from drug_screen.models.exp045_fold_step import (
    Exp045DependencyTargets,
    Exp045ExpressionTargets,
    Exp045FoldStepAudit,
    exp045_auxiliary_batch_step,
)
from drug_screen.models.exp045_multitask_objective import (
    MaskedDependencyBatch,
    MaskedExpressionReconstructionBatch,
    exp045_auxiliary_objective,
)


def dependency_targets(assignment="true", *, empty=False):
    target = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    valid_mask = torch.zeros(4, 2, dtype=torch.bool) if empty else torch.ones(4, 2, dtype=torch.bool)
    if empty:
        target.fill_(float("nan"))
    return Exp045DependencyTargets(
        target=target,
        valid_mask=valid_mask,
        train_mean=1.5,
        train_overall_std=2.0,
        label_assignment=assignment,
    )


def expression_targets(*, empty=False):
    target = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10.0
    valid_mask = torch.ones(4, 3, dtype=torch.bool)
    valid_mask[0, 0] = False
    if empty:
        valid_mask.fill_(False)
        target.fill_(float("nan"))
    return Exp045ExpressionTargets(target=target, valid_mask=valid_mask)


class TinyAuxiliaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalizer = nn.BatchNorm1d(3)
        self.encoder = nn.Linear(3, 4)
        self.reconstruction = nn.Linear(4, 3)
        self.depmap = nn.Linear(4, 2)
        self.nature = nn.Linear(4, 2)
        self.forward_calls = 0

    def predict_auxiliary_heads(self, expression):
        self.forward_calls += 1
        representation = torch.relu(self.encoder(self.normalizer(expression)))
        return SimpleNamespace(
            expression_reconstruction=self.reconstruction(representation),
            depmap_chronos=self.depmap(representation),
            nature_crispr_lfc=self.nature(representation),
        )


def direct_reconstruction_batch():
    return MaskedExpressionReconstructionBatch(
        prediction=torch.tensor(
            [[2.0, 1.0], [0.0, 4.0]], dtype=torch.float32, requires_grad=True),
        target=torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
        valid_mask=torch.tensor([[True, True], [False, True]], dtype=torch.bool),
    )


def dependency_batch(prediction, target, mask, *, mean, std):
    return MaskedDependencyBatch(
        prediction=torch.tensor(prediction, dtype=torch.float32, requires_grad=True),
        target=torch.tensor(target, dtype=torch.float32),
        valid_mask=torch.tensor(mask, dtype=torch.bool),
        train_mean=mean,
        train_overall_std=std,
    )


def test_auxiliary_objective_adds_reconstruction_to_each_scheduled_domain_loss():
    depmap = dependency_batch(
        [[1.0], [-1.0], [4.0], [float("nan")]],
        [[3.0], [3.0], [2.0], [float("nan")]],
        [[True], [True], [True], [False]],
        mean=1.0,
        std=2.0,
    )
    nature = dependency_batch(
        [[5.0], [8.0], [float("nan")]],
        [[9.0], [10.0], [float("nan")]],
        [[True], [True], [False]],
        mean=7.0,
        std=2.0,
    )

    depmap_result = exp045_auxiliary_objective(
        arm="D-CO",
        stage="depmap_source_training",
        lambda_reconstruction=0.5,
        reconstruction_batch=direct_reconstruction_batch(),
        depmap_active=True,
        lambda_depmap=0.5,
        depmap_batch=depmap,
        nature_active=False,
        lambda_nature=0.0,
        nature_batch=None,
    )

    # Reconstruction errors [1, 1, 9] average to 11/3; Chronos errors [1, 4, 1] average to 2.
    torch.testing.assert_close(
        depmap_result.expression_reconstruction.loss, torch.tensor(11.0 / 3.0))
    torch.testing.assert_close(depmap_result.total_loss, torch.tensor(17.0 / 6.0))
    assert depmap_result.expression_reconstruction.valid_denominator == 3
    assert depmap_result.expression_reconstruction.dimensions == (2,)
    assert depmap_result.depmap.valid_denominator == 3
    assert depmap_result.nature.status == "disabled"
    depmap_result.total_loss.backward()
    assert depmap.prediction.grad is not None

    nature_result = exp045_auxiliary_objective(
        arm="D-CO",
        stage="nature_organoid_adaptation",
        lambda_reconstruction=0.5,
        reconstruction_batch=direct_reconstruction_batch(),
        depmap_active=False,
        lambda_depmap=0.0,
        depmap_batch=None,
        nature_active=True,
        lambda_nature=2.0,
        nature_batch=nature,
    )
    torch.testing.assert_close(nature_result.nature.loss, torch.tensor(2.5))
    torch.testing.assert_close(nature_result.total_loss, torch.tensor(41.0 / 6.0))
    assert nature_result.nature.valid_denominator == 2
    nature_result.total_loss.backward()
    assert nature.prediction.grad is not None


@pytest.mark.parametrize(
    "arm, stage, depmap_assignment, nature_assignment",
    [
        ("D-E", "depmap_source_training", None, None),
        ("D-E", "nature_organoid_adaptation", None, None),
        ("D-C", "depmap_source_training", "true", None),
        ("D-C", "nature_organoid_adaptation", None, None),
        ("D-O", "depmap_source_training", None, None),
        ("D-O", "nature_organoid_adaptation", None, "true"),
        ("D-CO", "depmap_source_training", "true", None),
        ("D-CO", "nature_organoid_adaptation", None, "true"),
        ("D-SC", "depmap_source_training", "within_domain_shuffled", None),
        ("D-SC", "nature_organoid_adaptation", None, "true"),
        ("D-SO", "depmap_source_training", "true", None),
        ("D-SO", "nature_organoid_adaptation", None, "within_domain_shuffled"),
    ],
)
def test_all_six_auxiliary_arms_reconstruct_and_step_once_per_batch(
    arm, stage, depmap_assignment, nature_assignment
):
    torch.manual_seed(17)
    model = TinyAuxiliaryModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    audit = Exp045FoldStepAudit(arm=arm, stage=stage)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    shared_expression_targets = expression_targets()

    result = exp045_auxiliary_batch_step(
        model,
        optimizer,
        torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10.0,
        arm=arm,
        stage=stage,
        lambda_reconstruction=0.7,
        reconstruction_targets=shared_expression_targets,
        lambda_depmap=1.0 if depmap_assignment else 0.0,
        depmap_targets=(dependency_targets(depmap_assignment)
                        if depmap_assignment else None),
        lambda_nature=1.0 if nature_assignment else 0.0,
        nature_targets=(dependency_targets(nature_assignment)
                        if nature_assignment else None),
        audit=audit,
    )

    assert model.forward_calls == 1
    assert result.output.expression_reconstruction.shape == (4, 3)
    assert result.objective.expression_reconstruction.dimensions == (3,)
    assert result.objective.expression_reconstruction.valid_denominator == 11
    assert result.objective.should_optimizer_step and result.optimizer_stepped
    assert audit.scheduled_batches == audit.forward_calls == 1
    assert audit.backward_calls == audit.optimizer_steps == 1
    assert audit.failures == 0
    assert audit.reconstruction_dimensions == (3,)
    assert audit.reconstruction_valid_denominator == 11
    assert len(audit.reconstruction_losses) == 1
    assert audit.depmap_valid_denominators == [8 if depmap_assignment else 0]
    assert audit.nature_valid_denominators == [8 if nature_assignment else 0]
    assert audit.depmap_losses == ([result.objective.depmap.loss.item()]
                                   if depmap_assignment else [None])
    assert audit.nature_losses == ([result.objective.nature.loss.item()]
                                   if nature_assignment else [None])
    assert audit.total_losses == [result.objective.total_loss.item()]
    assert any(not torch.equal(old, new)
               for old, new in zip(before, model.parameters()))


def test_de_is_a_trainable_reconstruction_baseline_in_both_stages():
    for stage in ("depmap_source_training", "nature_organoid_adaptation"):
        torch.manual_seed(19)
        model = TinyAuxiliaryModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        audit = Exp045FoldStepAudit(arm="D-E", stage=stage)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        result = exp045_auxiliary_batch_step(
            model,
            optimizer,
            torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10.0,
            arm="D-E",
            stage=stage,
            lambda_reconstruction=1.0,
            reconstruction_targets=expression_targets(),
            lambda_depmap=0.0,
            depmap_targets=None,
            lambda_nature=0.0,
            nature_targets=None,
            audit=audit,
        )
        torch.testing.assert_close(
            result.objective.total_loss,
            result.objective.expression_reconstruction.loss,
        )
        assert result.objective.depmap.status == result.objective.nature.status == "disabled"
        assert result.optimizer_stepped
        assert audit.forward_calls == audit.backward_calls == audit.optimizer_steps == 1
        assert any(not torch.equal(old, new)
                   for old, new in zip(before, model.parameters()))


def test_reconstruction_empty_mask_fails_closed_without_an_optimizer_step():
    model = TinyAuxiliaryModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    audit = Exp045FoldStepAudit(arm="D-E", stage="depmap_source_training")
    before = [parameter.detach().clone() for parameter in model.parameters()]
    with pytest.raises(ValueError, match="reconstruction batch has no valid masked values"):
        exp045_auxiliary_batch_step(
            model,
            optimizer,
            torch.randn(4, 3),
            arm="D-E",
            stage="depmap_source_training",
            lambda_reconstruction=1.0,
            reconstruction_targets=expression_targets(empty=True),
            lambda_depmap=0.0,
            depmap_targets=None,
            lambda_nature=0.0,
            nature_targets=None,
            audit=audit,
        )
    assert audit.scheduled_batches == audit.forward_calls == audit.failures == 1
    assert audit.backward_calls == audit.optimizer_steps == 0
    assert all(torch.equal(old, new) for old, new in zip(before, model.parameters()))


def test_active_empty_dependency_batch_still_fails_closed():
    model = TinyAuxiliaryModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    audit = Exp045FoldStepAudit(arm="D-C", stage="depmap_source_training")
    with pytest.raises(ValueError, match="no valid labels"):
        exp045_auxiliary_batch_step(
            model,
            optimizer,
            torch.randn(4, 3),
            arm="D-C",
            stage="depmap_source_training",
            lambda_reconstruction=1.0,
            reconstruction_targets=expression_targets(),
            lambda_depmap=1.0,
            depmap_targets=dependency_targets(empty=True),
            lambda_nature=0.0,
            nature_targets=None,
            audit=audit,
        )
    assert audit.scheduled_batches == audit.forward_calls == audit.failures == 1
    assert audit.backward_calls == audit.optimizer_steps == 0


def test_arm_assignment_mismatch_fails_before_forward_and_counts_failure():
    model = TinyAuxiliaryModel()
    audit = Exp045FoldStepAudit(arm="D-SC", stage="depmap_source_training")
    with pytest.raises(ValueError, match="within_domain_shuffled"):
        exp045_auxiliary_batch_step(
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            torch.randn(4, 3),
            arm="D-SC",
            stage="depmap_source_training",
            lambda_reconstruction=1.0,
            reconstruction_targets=expression_targets(),
            lambda_depmap=1.0,
            depmap_targets=dependency_targets("true"),
            lambda_nature=0.0,
            nature_targets=None,
            audit=audit,
        )
    assert audit.scheduled_batches == audit.failures == 1
    assert audit.forward_calls == audit.backward_calls == audit.optimizer_steps == 0


def test_all_auxiliary_weights_and_reconstruction_targets_are_explicit():
    model = TinyAuxiliaryModel()
    audit = Exp045FoldStepAudit(arm="D-E", stage="depmap_source_training")
    with pytest.raises(TypeError):
        exp045_auxiliary_batch_step(
            model,
            None,
            torch.randn(4, 3),
            arm="D-E",
            stage="depmap_source_training",
            lambda_depmap=0.0,
            depmap_targets=None,
            lambda_nature=0.0,
            nature_targets=None,
            audit=audit,
        )
    with pytest.raises(TypeError):
        exp045_auxiliary_objective(
            arm="D-E",
            stage="depmap_source_training",
            reconstruction_batch=direct_reconstruction_batch(),
            depmap_active=False,
            lambda_depmap=0.0,
            depmap_batch=None,
            nature_active=False,
            lambda_nature=0.0,
            nature_batch=None,
        )


def test_direct_only_arm_cannot_schedule_an_auxiliary_batch():
    model = TinyAuxiliaryModel()
    audit = Exp045FoldStepAudit(arm="D-A", stage="depmap_source_training")
    with pytest.raises(ValueError, match="direct-only"):
        exp045_auxiliary_batch_step(
            model,
            None,
            torch.randn(4, 3),
            arm="D-A",
            stage="depmap_source_training",
            lambda_reconstruction=1.0,
            reconstruction_targets=expression_targets(),
            lambda_depmap=0.0,
            depmap_targets=None,
            lambda_nature=0.0,
            nature_targets=None,
            audit=audit,
        )
    assert audit.scheduled_batches == audit.forward_calls == audit.failures == 0
