"""Pure loss calculation for the EXP-045 multi-task training contract."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import torch
from torch.nn import functional

from scripts.modeling.exp036_pharmaformer_fold import pairwise_ranking_loss


HeadStatus = Literal["disabled", "active", "active_empty"]
Exp045AuxiliaryArm = Literal["D-E", "D-C", "D-O", "D-CO", "D-SC", "D-SO"]
Exp045AuxiliaryStage = Literal[
    "depmap_source_training", "nature_organoid_adaptation"]

# The protocol schedules the auxiliary domains in two successive stages.
# Empty sets are exposure-only stages, not permission to silently skip labels.
EXP045_STAGE_ACTIVE_DOMAINS: dict[tuple[str, str], frozenset[str]] = {
    ("D-E", "depmap_source_training"): frozenset(),
    ("D-E", "nature_organoid_adaptation"): frozenset(),
    ("D-C", "depmap_source_training"): frozenset({"depmap"}),
    ("D-C", "nature_organoid_adaptation"): frozenset(),
    ("D-O", "depmap_source_training"): frozenset(),
    ("D-O", "nature_organoid_adaptation"): frozenset({"nature"}),
    ("D-CO", "depmap_source_training"): frozenset({"depmap"}),
    ("D-CO", "nature_organoid_adaptation"): frozenset({"nature"}),
    ("D-SC", "depmap_source_training"): frozenset({"depmap"}),
    ("D-SC", "nature_organoid_adaptation"): frozenset({"nature"}),
    ("D-SO", "depmap_source_training"): frozenset({"depmap"}),
    ("D-SO", "nature_organoid_adaptation"): frozenset({"nature"}),
}


@dataclass(frozen=True)
class MaskedDependencyBatch:
    """One dependency domain's predictions, labels, mask, and train-only scale.

    ``train_mean`` and ``train_overall_std`` must be calculated by the caller
    from that domain's permitted training fold. This object never estimates or
    shares statistics between DepMap and Nature.
    """

    prediction: torch.Tensor
    target: torch.Tensor
    valid_mask: torch.Tensor
    train_mean: float
    train_overall_std: float


@dataclass(frozen=True)
class MaskedExpressionReconstructionBatch:
    """Fold-scaled expression reconstruction values and their validity mask.

    The caller fits and applies the expression transform using the permitted
    training fold. This module neither estimates expression statistics nor
    fills missing targets.
    """

    prediction: torch.Tensor
    target: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class DependencyLossTerm:
    """A domain loss, its valid-label denominator, and activation state."""

    loss: torch.Tensor | None
    valid_denominator: int
    status: HeadStatus


@dataclass(frozen=True)
class ExpressionReconstructionLossTerm:
    """Masked reconstruction loss and its feature dimensions and denominator."""

    loss: torch.Tensor
    valid_denominator: int
    dimensions: tuple[int, ...]


@dataclass(frozen=True)
class Exp045ObjectiveResult:
    """Loss components and a fail-closed optimizer-step recommendation."""

    total_loss: torch.Tensor | None
    dss_loss: torch.Tensor
    dss_huber_loss: torch.Tensor
    dss_ranking_loss: torch.Tensor
    dss_patient_denominator: int
    depmap: DependencyLossTerm
    nature: DependencyLossTerm
    should_optimizer_step: bool
    status: Literal["ready", "active_head_empty"]


@dataclass(frozen=True)
class Exp045AuxiliaryObjectiveResult:
    """Expression reconstruction plus the stage's scheduled dependency loss."""

    total_loss: torch.Tensor
    expression_reconstruction: ExpressionReconstructionLossTerm
    depmap: DependencyLossTerm
    nature: DependencyLossTerm
    should_optimizer_step: bool
    status: Literal["ready"]


def _nonnegative_weight(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be an explicitly supplied finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _positive_weight(name: str, value: float) -> float:
    number = _nonnegative_weight(name, value)
    if number <= 0:
        raise ValueError(f"{name} must be explicitly supplied and positive")
    return number


def _training_statistic(name: str, value: float, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite scalar from the training fold")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        condition = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {condition}")
    return number


def _validate_dss_vectors(prediction: torch.Tensor,
                          target: torch.Tensor) -> None:
    if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("DSS prediction and target must be torch tensors")
    if prediction.ndim != 1 or target.shape != prediction.shape:
        raise ValueError("DSS prediction and target must be aligned vectors")
    if prediction.numel() < 1:
        raise ValueError("DSS batch must contain at least one labeled pair")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError("DSS prediction and target must be floating point tensors")
    if prediction.device != target.device or prediction.dtype != target.dtype:
        raise ValueError("DSS prediction and target must share device and dtype")
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("DSS prediction and target must be finite")


def _validated_patient_groups(
    patient_groups: Sequence[object], *, n_rows: int, device: torch.device,
) -> tuple[torch.Tensor, ...]:
    if isinstance(patient_groups, (str, bytes)):
        raise TypeError("patient_groups must contain row-index groups")
    try:
        groups = tuple(patient_groups)
    except TypeError as exc:
        raise TypeError("patient_groups must be a sequence of row-index groups") from exc
    if not groups:
        raise ValueError("patient_groups must contain at least one patient")

    covered = [False] * n_rows
    result: list[torch.Tensor] = []
    for group in groups:
        try:
            indices = torch.as_tensor(group)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("each patient group must contain integer row indices") from exc
        if indices.ndim != 1 or indices.numel() < 1:
            raise ValueError("each patient group must be a non-empty index vector")
        if indices.dtype not in (torch.uint8, torch.int8, torch.int16,
                                 torch.int32, torch.int64):
            raise ValueError("patient group indices must use an integer dtype")
        positions = indices.detach().cpu().tolist()
        for position in positions:
            if position < 0 or position >= n_rows:
                raise ValueError("patient group index is outside the DSS batch")
            if covered[position]:
                raise ValueError("patient groups must partition DSS rows without overlap")
            covered[position] = True
        result.append(torch.tensor(positions, dtype=torch.long, device=device))

    if not all(covered):
        raise ValueError("patient groups must cover every DSS row exactly once")
    return tuple(result)


def _dss_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    patient_groups: Sequence[object],
    ranking_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    _validate_dss_vectors(prediction, target)
    groups = _validated_patient_groups(
        patient_groups, n_rows=prediction.shape[0], device=prediction.device)

    patient_huber: list[torch.Tensor] = []
    patient_ranking: list[torch.Tensor] = []
    for indices in groups:
        patient_prediction = prediction.index_select(0, indices)
        patient_target = target.index_select(0, indices)
        patient_huber.append(functional.smooth_l1_loss(
            patient_prediction, patient_target, reduction="mean"))
        if ranking_weight > 0:
            patient_ranking.append(pairwise_ranking_loss(
                patient_prediction, patient_target))
        else:
            patient_ranking.append(prediction.new_zeros(()))

    huber_loss = torch.stack(patient_huber).mean()
    ranking_loss = torch.stack(patient_ranking).mean()
    return huber_loss, ranking_loss, len(groups)


def _dependency_loss_term(
    name: str,
    batch: MaskedDependencyBatch | None,
    weight: float,
) -> DependencyLossTerm:
    if weight == 0:
        return DependencyLossTerm(loss=None, valid_denominator=0, status="disabled")
    if batch is None:
        raise ValueError(
            f"{name} is active but has no batch; pass an explicit mask, including an empty mask"
        )
    if not isinstance(batch, MaskedDependencyBatch):
        raise TypeError(f"{name} batch must be MaskedDependencyBatch")

    train_mean = _training_statistic(
        f"{name} train_mean", batch.train_mean, positive=False)
    train_std = _training_statistic(
        f"{name} train_overall_std", batch.train_overall_std, positive=True)
    prediction, target, mask = batch.prediction, batch.target, batch.valid_mask
    if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError(f"{name} prediction and target must be torch tensors")
    if prediction.shape != target.shape or prediction.ndim < 1 or prediction.numel() < 1:
        raise ValueError(f"{name} prediction and target must have the same non-empty shape")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError(f"{name} prediction and target must be floating point tensors")
    if prediction.device != target.device or prediction.dtype != target.dtype:
        raise ValueError(f"{name} prediction and target must share device and dtype")
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError(f"{name} valid_mask must be a boolean tensor")
    if mask.shape != target.shape or mask.device != target.device:
        raise ValueError(f"{name} valid_mask must align with its target on the same device")

    valid_denominator = int(mask.sum().item())
    if valid_denominator == 0:
        return DependencyLossTerm(
            loss=None, valid_denominator=0, status="active_empty")

    selected_prediction = prediction[mask]
    selected_target = target[mask]
    if (not bool(torch.isfinite(selected_prediction).all())
            or not bool(torch.isfinite(selected_target).all())):
        raise ValueError(f"{name} valid predictions and targets must be finite")
    mean = prediction.new_tensor(train_mean)
    std = prediction.new_tensor(train_std)
    prediction_scaled = (selected_prediction - mean) / std
    target_scaled = (selected_target - mean) / std
    loss = functional.mse_loss(prediction_scaled, target_scaled, reduction="mean")
    if not bool(torch.isfinite(loss)):
        raise ValueError(f"{name} masked MSE is non-finite")
    return DependencyLossTerm(
        loss=loss, valid_denominator=valid_denominator, status="active")


def _expression_reconstruction_loss(
    batch: MaskedExpressionReconstructionBatch,
) -> ExpressionReconstructionLossTerm:
    if not isinstance(batch, MaskedExpressionReconstructionBatch):
        raise TypeError(
            "expression reconstruction batch must be MaskedExpressionReconstructionBatch"
        )
    prediction, target, mask = batch.prediction, batch.target, batch.valid_mask
    if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("expression reconstruction prediction and target must be tensors")
    if (prediction.ndim != 2 or target.ndim != 2
            or prediction.shape != target.shape
            or prediction.shape[0] < 1 or prediction.shape[1] < 1):
        raise ValueError(
            "expression reconstruction prediction and target must share a non-empty "
            "batch-by-feature shape"
        )
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError("expression reconstruction prediction and target must be floating point")
    if prediction.device != target.device or prediction.dtype != target.dtype:
        raise ValueError(
            "expression reconstruction prediction and target must share device and dtype"
        )
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError("expression reconstruction valid_mask must be a boolean tensor")
    if mask.shape != target.shape or mask.device != target.device:
        raise ValueError("expression reconstruction mask must align with its target")

    valid_denominator = int(mask.sum().item())
    if valid_denominator == 0:
        raise ValueError("expression reconstruction batch has no valid masked values")
    selected_prediction = prediction[mask]
    selected_target = target[mask]
    if (not bool(torch.isfinite(selected_prediction).all())
            or not bool(torch.isfinite(selected_target).all())):
        raise ValueError("valid expression reconstruction values must be finite")
    loss = functional.mse_loss(selected_prediction, selected_target, reduction="mean")
    if not bool(torch.isfinite(loss)):
        raise ValueError("masked expression reconstruction MSE is non-finite")
    return ExpressionReconstructionLossTerm(
        loss=loss,
        valid_denominator=valid_denominator,
        dimensions=tuple(int(value) for value in target.shape[1:]),
    )


def exp045_auxiliary_objective(
    *,
    arm: Exp045AuxiliaryArm,
    stage: Exp045AuxiliaryStage,
    lambda_reconstruction: float,
    reconstruction_batch: MaskedExpressionReconstructionBatch,
    depmap_active: bool,
    lambda_depmap: float,
    depmap_batch: MaskedDependencyBatch | None,
    nature_active: bool,
    lambda_nature: float,
    nature_batch: MaskedDependencyBatch | None,
) -> Exp045AuxiliaryObjectiveResult:
    """Combine fold-scaled expression reconstruction and scheduled domain loss.

    Activity is bound to the preregistered arm and the current two-stage
    schedule. The DepMap source stage can only update the DepMap head; the
    Nature adaptation stage can only update the Nature head. Every auxiliary
    arm must reconstruct the same caller-supplied fold-scaled expression target
    on each scheduled batch. Empty reconstruction or active dependency masks
    fail immediately, so no zero-loss optimizer update can occur.
    """
    if not isinstance(depmap_active, bool) or not isinstance(nature_active, bool):
        raise TypeError("dependency-domain activity flags must be explicit booleans")
    expected_active = EXP045_STAGE_ACTIVE_DOMAINS.get((arm, stage))
    if expected_active is None:
        raise ValueError(f"unsupported EXP-045 auxiliary arm/stage: {arm}/{stage}")
    if depmap_active != ("depmap" in expected_active):
        raise ValueError(f"{arm}/{stage} DepMap activity does not match the frozen stage")
    if nature_active != ("nature" in expected_active):
        raise ValueError(f"{arm}/{stage} Nature activity does not match the frozen stage")
    lambda_reconstruction = _positive_weight(
        "lambda_reconstruction", lambda_reconstruction)
    lambda_depmap = _nonnegative_weight("lambda_depmap", lambda_depmap)
    lambda_nature = _nonnegative_weight("lambda_nature", lambda_nature)

    for name, active, weight, batch in (
        ("DepMap Chronos", depmap_active, lambda_depmap, depmap_batch),
        ("Nature CRISPR-LFC", nature_active, lambda_nature, nature_batch),
    ):
        if active and weight <= 0:
            raise ValueError(f"{name} is active and requires a positive explicit lambda")
        if not active and weight != 0:
            raise ValueError(f"{name} is inactive and requires an explicit zero lambda")
        if active and batch is None:
            raise ValueError(f"{name} is active but has no batch")
        if not active and batch is not None:
            raise ValueError(f"{name} is inactive and must not receive a label batch")

    reconstruction = _expression_reconstruction_loss(reconstruction_batch)
    depmap = _dependency_loss_term(
        "DepMap Chronos", depmap_batch, lambda_depmap)
    nature = _dependency_loss_term(
        "Nature CRISPR-LFC", nature_batch, lambda_nature)
    if depmap.status == "active_empty" or nature.status == "active_empty":
        raise ValueError(
            "a scheduled active dependency domain has no valid labels; this step "
            "fails immediately, and an upper-level driver may reschedule only "
            "under the preregistered sampling rule"
        )

    total_loss = lambda_reconstruction * reconstruction.loss
    if depmap.loss is not None:
        total_loss = total_loss + lambda_depmap * depmap.loss
    if nature.loss is not None:
        total_loss = total_loss + lambda_nature * nature.loss
    if not bool(torch.isfinite(total_loss)):
        raise ValueError("EXP-045 auxiliary total loss is non-finite")
    return Exp045AuxiliaryObjectiveResult(
        total_loss=total_loss,
        expression_reconstruction=reconstruction,
        depmap=depmap,
        nature=nature,
        should_optimizer_step=True,
        status="ready",
    )


def exp045_multitask_objective(
    dss_prediction: torch.Tensor,
    dss_target: torch.Tensor,
    dss_patient_groups: Sequence[object],
    *,
    dss_ranking_weight: float,
    lambda_depmap: float,
    depmap_batch: MaskedDependencyBatch | None,
    lambda_nature: float,
    nature_batch: MaskedDependencyBatch | None,
) -> Exp045ObjectiveResult:
    """Calculate the EXP-045 loss without fitting stats or stepping an optimizer.

    The DSS rows must be grouped by training patient. Their Huber and pairwise
    logistic terms are averaged per patient, matching the EXP-036 patient-equal
    update path; DSS predictions and targets use the response scale prepared by
    the caller from the permitted training fold. Each active dependency loss
    standardizes its own predictions and valid labels with caller-supplied
    training-fold statistics and averages over valid labels only. An active
    empty dependency batch returns no total loss and
    ``should_optimizer_step=False``; an inactive head has weight zero and
    status ``disabled``.
    """
    dss_ranking_weight = _nonnegative_weight(
        "dss_ranking_weight", dss_ranking_weight)
    lambda_depmap = _nonnegative_weight("lambda_depmap", lambda_depmap)
    lambda_nature = _nonnegative_weight("lambda_nature", lambda_nature)

    dss_huber, dss_ranking, dss_patient_denominator = _dss_losses(
        dss_prediction, dss_target, dss_patient_groups, dss_ranking_weight)
    dss_loss = dss_huber + dss_ranking_weight * dss_ranking
    depmap = _dependency_loss_term("DepMap Chronos", depmap_batch, lambda_depmap)
    nature = _dependency_loss_term("Nature CRISPR-LFC", nature_batch, lambda_nature)
    active_empty = depmap.status == "active_empty" or nature.status == "active_empty"

    if active_empty:
        return Exp045ObjectiveResult(
            total_loss=None,
            dss_loss=dss_loss,
            dss_huber_loss=dss_huber,
            dss_ranking_loss=dss_ranking,
            dss_patient_denominator=dss_patient_denominator,
            depmap=depmap,
            nature=nature,
            should_optimizer_step=False,
            status="active_head_empty",
        )

    total_loss = dss_loss
    if depmap.loss is not None:
        total_loss = total_loss + lambda_depmap * depmap.loss
    if nature.loss is not None:
        total_loss = total_loss + lambda_nature * nature.loss
    if not bool(torch.isfinite(total_loss)):
        raise ValueError("EXP-045 total loss is non-finite")
    return Exp045ObjectiveResult(
        total_loss=total_loss,
        dss_loss=dss_loss,
        dss_huber_loss=dss_huber,
        dss_ranking_loss=dss_ranking,
        dss_patient_denominator=dss_patient_denominator,
        depmap=depmap,
        nature=nature,
        should_optimizer_step=True,
        status="ready",
    )
