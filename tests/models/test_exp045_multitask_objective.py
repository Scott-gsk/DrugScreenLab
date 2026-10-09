import pytest
import torch
from torch.nn import functional

from drug_screen.models.exp045_multitask_objective import (
    MaskedDependencyBatch,
    MaskedExpressionReconstructionBatch,
    exp045_auxiliary_objective,
    exp045_multitask_objective,
)
from scripts.modeling.exp036_pharmaformer_fold import pairwise_ranking_loss


def dependency_batch(prediction, target, mask, *, mean, std):
    return MaskedDependencyBatch(
        prediction=torch.as_tensor(prediction, dtype=torch.float32),
        target=torch.as_tensor(target, dtype=torch.float32),
        valid_mask=torch.as_tensor(mask, dtype=torch.bool),
        train_mean=mean,
        train_overall_std=std,
    )


def reconstruction_batch(prediction, target, mask):
    prediction_tensor = torch.as_tensor(prediction, dtype=torch.float32).requires_grad_()
    return MaskedExpressionReconstructionBatch(
        prediction=prediction_tensor,
        target=torch.as_tensor(target, dtype=torch.float32),
        valid_mask=torch.as_tensor(mask, dtype=torch.bool),
    )


def auxiliary_objective(*, arm="D-E", stage="depmap_source_training",
                        reconstruction=None, lambda_reconstruction=0.5,
                        depmap_active=False, lambda_depmap=0.0, depmap_batch=None,
                        nature_active=False, lambda_nature=0.0, nature_batch=None):
    if reconstruction is None:
        reconstruction = reconstruction_batch(
            [[2.0, 1.0], [0.0, 4.0]],
            [[1.0, 0.0], [1.0, 1.0]],
            [[True, True], [False, True]],
        )
    return exp045_auxiliary_objective(
        arm=arm,
        stage=stage,
        lambda_reconstruction=lambda_reconstruction,
        reconstruction_batch=reconstruction,
        depmap_active=depmap_active,
        lambda_depmap=lambda_depmap,
        depmap_batch=depmap_batch,
        nature_active=nature_active,
        lambda_nature=lambda_nature,
        nature_batch=nature_batch,
    )


def calculate(prediction, target, groups, *, ranking_weight,
              depmap_weight, depmap, nature_weight, nature):
    return exp045_multitask_objective(
        prediction,
        target,
        groups,
        dss_ranking_weight=ranking_weight,
        lambda_depmap=depmap_weight,
        depmap_batch=depmap,
        lambda_nature=nature_weight,
        nature_batch=nature,
    )


def test_dss_matches_exp036_patient_equal_huber_and_pairwise_ranking():
    prediction = torch.tensor([0.0, 2.0, 1.0, 0.0], requires_grad=True)
    target = torch.tensor([0.0, 1.0, 0.0, 2.0])
    groups = [[0, 1], [2, 3]]
    ranking_weight = 0.4

    result = calculate(
        prediction, target, groups,
        ranking_weight=ranking_weight,
        depmap_weight=0.0,
        depmap=None,
        nature_weight=0.0,
        nature=None,
    )
    patient_huber = [
        functional.smooth_l1_loss(prediction[index], target[index])
        for index in (torch.tensor([0, 1]), torch.tensor([2, 3]))
    ]
    patient_ranking = [
        pairwise_ranking_loss(prediction[index], target[index])
        for index in (torch.tensor([0, 1]), torch.tensor([2, 3]))
    ]
    expected_huber = torch.stack(patient_huber).mean()
    expected_ranking = torch.stack(patient_ranking).mean()

    torch.testing.assert_close(result.dss_huber_loss, expected_huber)
    torch.testing.assert_close(result.dss_ranking_loss, expected_ranking)
    torch.testing.assert_close(
        result.dss_loss, expected_huber + ranking_weight * expected_ranking)
    assert result.dss_patient_denominator == 2
    assert result.should_optimizer_step and result.status == "ready"
    result.total_loss.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_domain_losses_use_separate_train_scales_and_valid_label_denominators():
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

    result = calculate(
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.0, 1.0]),
        [[0, 1]],
        ranking_weight=0.0,
        depmap_weight=0.5,
        depmap=depmap,
        nature_weight=2.0,
        nature=nature,
    )

    # The masked Chronos errors are [1, 4, 1], averaged over 3 labels.
    torch.testing.assert_close(result.depmap.loss, torch.tensor(2.0))
    # The masked Nature errors are [4, 1], averaged over 2 labels.
    torch.testing.assert_close(result.nature.loss, torch.tensor(2.5))
    assert result.depmap.valid_denominator == 3
    assert result.nature.valid_denominator == 2
    torch.testing.assert_close(result.total_loss, torch.tensor(6.0))
    assert result.should_optimizer_step


def test_disabled_head_and_active_empty_head_have_distinct_states_and_no_step():
    empty_nature = dependency_batch(
        [[float("nan")], [float("nan")]],
        [[float("nan")], [float("nan")]],
        [[False], [False]],
        mean=0.0,
        std=1.0,
    )

    result = calculate(
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.0, 1.0]),
        [[0, 1]],
        ranking_weight=0.0,
        depmap_weight=0.0,
        depmap=None,
        nature_weight=1.0,
        nature=empty_nature,
    )

    assert result.depmap.status == "disabled"
    assert result.depmap.loss is None and result.depmap.valid_denominator == 0
    assert result.nature.status == "active_empty"
    assert result.nature.loss is None and result.nature.valid_denominator == 0
    assert result.total_loss is None
    assert not result.should_optimizer_step
    assert result.status == "active_head_empty"


def test_active_head_requires_an_explicit_masked_batch():
    with pytest.raises(ValueError, match="active but has no batch"):
        calculate(
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            [[0]],
            ranking_weight=0.0,
            depmap_weight=1.0,
            depmap=None,
            nature_weight=0.0,
            nature=None,
        )


@pytest.mark.parametrize("std", [0.0, -1.0, float("nan"), float("inf")])
def test_active_domain_rejects_zero_or_invalid_training_scale(std):
    batch = dependency_batch([[1.0]], [[2.0]], [[True]], mean=0.0, std=std)
    with pytest.raises(ValueError, match="train_overall_std"):
        calculate(
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            [[0]],
            ranking_weight=0.0,
            depmap_weight=1.0,
            depmap=batch,
            nature_weight=0.0,
            nature=None,
        )


def test_active_domain_rejects_nonfinite_training_mean():
    batch = dependency_batch([[1.0]], [[2.0]], [[True]], mean=float("nan"), std=1.0)
    with pytest.raises(ValueError, match="train_mean"):
        calculate(
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            [[0]],
            ranking_weight=0.0,
            depmap_weight=1.0,
            depmap=batch,
            nature_weight=0.0,
            nature=None,
        )


def test_active_domain_requires_boolean_mask():
    batch = MaskedDependencyBatch(
        prediction=torch.tensor([[1.0]]),
        target=torch.tensor([[2.0]]),
        valid_mask=torch.tensor([[1]], dtype=torch.int64),
        train_mean=0.0,
        train_overall_std=1.0,
    )
    with pytest.raises(TypeError, match="boolean tensor"):
        calculate(
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            [[0]],
            ranking_weight=0.0,
            depmap_weight=1.0,
            depmap=batch,
            nature_weight=0.0,
            nature=None,
        )


def test_objective_requires_all_weights_from_the_caller():
    with pytest.raises(TypeError):
        exp045_multitask_objective(
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            [[0]],
            depmap_batch=None,
            nature_batch=None,
        )


@pytest.mark.parametrize("weight", [-0.1, float("nan"), float("inf")])
def test_objective_rejects_invalid_explicit_head_weights(weight):
    with pytest.raises(ValueError, match="lambda_depmap"):
        calculate(
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            [[0]],
            ranking_weight=0.0,
            depmap_weight=weight,
            depmap=None,
            nature_weight=0.0,
            nature=None,
        )


def test_auxiliary_objective_combines_masked_reconstruction_and_domain_loss():
    depmap = dependency_batch(
        [[1.0], [-1.0], [4.0], [float("nan")]],
        [[3.0], [3.0], [2.0], [float("nan")]],
        [[True], [True], [True], [False]],
        mean=1.0,
        std=2.0,
    )
    result = auxiliary_objective(
        arm="D-C",
        depmap_active=True,
        lambda_depmap=0.25,
        depmap_batch=depmap,
    )

    # Reconstruction squared errors are [1, 1, 9], averaged over 3 values.
    torch.testing.assert_close(
        result.expression_reconstruction.loss, torch.tensor(11.0 / 3.0))
    assert result.expression_reconstruction.dimensions == (2,)
    assert result.expression_reconstruction.valid_denominator == 3
    torch.testing.assert_close(result.depmap.loss, torch.tensor(2.0))
    torch.testing.assert_close(result.total_loss, torch.tensor(7.0 / 3.0))
    assert result.should_optimizer_step and result.status == "ready"
    result.total_loss.backward()
    assert result.expression_reconstruction.loss.grad_fn is not None


def test_de_objective_is_reconstruction_only_and_requires_positive_explicit_weight():
    result = auxiliary_objective()

    torch.testing.assert_close(
        result.total_loss,
        0.5 * result.expression_reconstruction.loss,
    )
    assert result.depmap.status == result.nature.status == "disabled"
    assert result.should_optimizer_step
    with pytest.raises(ValueError, match="lambda_reconstruction.*positive"):
        auxiliary_objective(lambda_reconstruction=0.0)


def test_auxiliary_objective_requires_reconstruction_weight_and_batch():
    reconstruction = reconstruction_batch([[1.0]], [[0.0]], [[True]])
    with pytest.raises(TypeError):
        exp045_auxiliary_objective(
            arm="D-E",
            stage="depmap_source_training",
            reconstruction_batch=reconstruction,
            depmap_active=False,
            lambda_depmap=0.0,
            depmap_batch=None,
            nature_active=False,
            lambda_nature=0.0,
            nature_batch=None,
        )
    with pytest.raises(ValueError, match="no valid masked values"):
        auxiliary_objective(
            reconstruction=reconstruction_batch([[float("nan")]], [[float("nan")]], [[False]])
        )


def test_auxiliary_objective_rejects_nonboolean_or_misaligned_reconstruction_mask():
    with pytest.raises(TypeError, match="boolean tensor"):
        auxiliary_objective(
            reconstruction=MaskedExpressionReconstructionBatch(
                prediction=torch.tensor([[1.0]]),
                target=torch.tensor([[0.0]]),
                valid_mask=torch.tensor([[1]], dtype=torch.int64),
            )
        )
    with pytest.raises(ValueError, match="align with its target"):
        auxiliary_objective(
            reconstruction=MaskedExpressionReconstructionBatch(
                prediction=torch.tensor([[1.0, 2.0]]),
                target=torch.tensor([[0.0, 0.0]]),
                valid_mask=torch.tensor([[True]], dtype=torch.bool),
            )
        )
