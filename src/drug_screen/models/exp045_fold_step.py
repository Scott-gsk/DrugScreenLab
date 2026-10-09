"""One auditable auxiliary batch step for the EXP-045 D15 training arms."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Literal

import torch
from torch import nn

from drug_screen.models.exp045_multitask_objective import (
    Exp045AuxiliaryArm,
    Exp045AuxiliaryObjectiveResult,
    Exp045AuxiliaryStage,
    MaskedDependencyBatch,
    MaskedExpressionReconstructionBatch,
    exp045_auxiliary_objective,
)


Exp045Arm = Literal["D-A", "D-E", "D-C", "D-O", "D-CO", "D-SC", "D-SO"]
Exp045Stage = Exp045AuxiliaryStage
LabelAssignment = Literal["true", "within_domain_shuffled"]


@dataclass(frozen=True)
class Exp045DependencyTargets:
    """One domain's fold-fitted target scale, mask, and assigned labels."""

    target: torch.Tensor
    valid_mask: torch.Tensor
    train_mean: float
    train_overall_std: float
    label_assignment: LabelAssignment


@dataclass(frozen=True)
class Exp045ExpressionTargets:
    """One common fold-fitted/scaled expression target and missingness mask."""

    target: torch.Tensor
    valid_mask: torch.Tensor


@dataclass
class Exp045FoldStepAudit:
    """Per-arm counters and loss evidence for scheduled auxiliary batches."""

    arm: Exp045Arm
    stage: Exp045Stage
    scheduled_batches: int = 0
    forward_calls: int = 0
    backward_calls: int = 0
    optimizer_steps: int = 0
    failures: int = 0
    reconstruction_dimensions: tuple[int, ...] | None = None
    reconstruction_valid_denominator: int = 0
    reconstruction_losses: list[float] = field(default_factory=list)
    depmap_valid_denominators: list[int] = field(default_factory=list)
    nature_valid_denominators: list[int] = field(default_factory=list)
    depmap_losses: list[float | None] = field(default_factory=list)
    nature_losses: list[float | None] = field(default_factory=list)
    total_losses: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class Exp045AuxiliaryStepResult:
    """Result from a single auxiliary expression batch."""

    arm: Exp045Arm
    output: object
    objective: Exp045AuxiliaryObjectiveResult
    optimizer_stepped: bool


_EXPECTED_ASSIGNMENTS: dict[tuple[str, str], dict[str, LabelAssignment]] = {
    ("D-E", "depmap_source_training"): {},
    ("D-E", "nature_organoid_adaptation"): {},
    ("D-C", "depmap_source_training"): {"depmap": "true"},
    ("D-C", "nature_organoid_adaptation"): {},
    ("D-O", "depmap_source_training"): {},
    ("D-O", "nature_organoid_adaptation"): {"nature": "true"},
    ("D-CO", "depmap_source_training"): {"depmap": "true"},
    ("D-CO", "nature_organoid_adaptation"): {"nature": "true"},
    ("D-SC", "depmap_source_training"): {"depmap": "within_domain_shuffled"},
    ("D-SC", "nature_organoid_adaptation"): {"nature": "true"},
    ("D-SO", "depmap_source_training"): {"depmap": "true"},
    ("D-SO", "nature_organoid_adaptation"): {"nature": "within_domain_shuffled"},
}


def _explicit_weight(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be an explicitly supplied finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _build_batch(
    prediction: torch.Tensor,
    targets: Exp045DependencyTargets | None,
) -> MaskedDependencyBatch | None:
    if targets is None:
        return None
    return MaskedDependencyBatch(
        prediction=prediction,
        target=targets.target,
        valid_mask=targets.valid_mask,
        train_mean=targets.train_mean,
        train_overall_std=targets.train_overall_std,
    )


def _build_reconstruction_batch(
    prediction: torch.Tensor,
    targets: Exp045ExpressionTargets,
) -> MaskedExpressionReconstructionBatch:
    if not isinstance(targets, Exp045ExpressionTargets):
        raise TypeError("reconstruction_targets must be Exp045ExpressionTargets")
    return MaskedExpressionReconstructionBatch(
        prediction=prediction,
        target=targets.target,
        valid_mask=targets.valid_mask,
    )


def _validate_prediction_output(
    output: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction_prediction = getattr(output, "expression_reconstruction", None)
    depmap_prediction = getattr(output, "depmap_chronos", None)
    nature_prediction = getattr(output, "nature_crispr_lfc", None)
    if not isinstance(reconstruction_prediction, torch.Tensor):
        raise TypeError(
            "auxiliary forward output must expose expression_reconstruction tensor"
        )
    if not isinstance(depmap_prediction, torch.Tensor):
        raise TypeError("dependency forward output must expose depmap_chronos tensor")
    if not isinstance(nature_prediction, torch.Tensor):
        raise TypeError("dependency forward output must expose nature_crispr_lfc tensor")
    return reconstruction_prediction, depmap_prediction, nature_prediction


def exp045_auxiliary_batch_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    expression: torch.Tensor,
    *,
    arm: Exp045Arm,
    stage: Exp045Stage,
    lambda_reconstruction: float,
    reconstruction_targets: Exp045ExpressionTargets,
    lambda_depmap: float,
    depmap_targets: Exp045DependencyTargets | None,
    lambda_nature: float,
    nature_targets: Exp045DependencyTargets | None,
    audit: Exp045FoldStepAudit,
) -> Exp045AuxiliaryStepResult:
    """Run one reconstruction-plus-optional-dependency batch update.

    The caller owns fold-safe batch construction, shuffled-label generation,
    cross-arm opportunity matching, and separate DSS updates. The stage schedule
    permits only the current domain's dependency target: DepMap source training
    first, then Nature organoid adaptation. Every auxiliary arm receives the
    same type of already fold-scaled expression target/mask and performs exactly
    one reconstruction optimizer update per valid batch. D-E has reconstruction
    only; D-A is direct-only and must not call this function.
    """
    if arm == "D-A":
        raise ValueError("D-A is direct-only and has no auxiliary batch steps")
    if (arm, stage) not in _EXPECTED_ASSIGNMENTS:
        raise ValueError(f"unsupported EXP-045 arm/stage: {arm}/{stage}")
    if (not isinstance(audit, Exp045FoldStepAudit)
            or audit.arm != arm or audit.stage != stage):
        raise ValueError("audit counters must be initialized for the same arm and stage")

    audit.scheduled_batches += 1
    try:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch module")
        if not callable(getattr(model, "predict_auxiliary_heads", None)):
            raise TypeError("model must expose predict_auxiliary_heads(expression)")
        if not isinstance(expression, torch.Tensor) or expression.ndim != 2:
            raise ValueError("expression must be a two-dimensional batch-by-gene tensor")
        if expression.shape[0] < 1 or not expression.is_floating_point():
            raise ValueError("expression must be a non-empty floating point batch")
        if not isinstance(reconstruction_targets, Exp045ExpressionTargets):
            raise TypeError("reconstruction_targets must be Exp045ExpressionTargets")

        lambda_reconstruction = _explicit_weight(
            "lambda_reconstruction", lambda_reconstruction)
        if lambda_reconstruction <= 0:
            raise ValueError("lambda_reconstruction must be explicitly positive")
        lambda_depmap = _explicit_weight("lambda_depmap", lambda_depmap)
        lambda_nature = _explicit_weight("lambda_nature", lambda_nature)
        assignments = _EXPECTED_ASSIGNMENTS[(arm, stage)]
        expected_depmap = assignments.get("depmap")
        expected_nature = assignments.get("nature")
        for name, expected, targets, weight in (
            ("DepMap", expected_depmap, depmap_targets, lambda_depmap),
            ("Nature", expected_nature, nature_targets, lambda_nature),
        ):
            if expected is None:
                if targets is not None or weight != 0:
                    raise ValueError(
                        f"{arm} must close {name} with zero weight and no label batch"
                    )
            else:
                if targets is None or weight <= 0:
                    raise ValueError(
                        f"{arm} requires an explicit positive-weight {name} label batch"
                    )
                if targets.label_assignment != expected:
                    raise ValueError(
                        f"{arm} requires {name} labels assigned as {expected}"
                    )

        audit.forward_calls += 1
        output = model.predict_auxiliary_heads(expression)

        reconstruction_prediction, depmap_prediction, nature_prediction = (
            _validate_prediction_output(output)
        )
        objective = exp045_auxiliary_objective(
            arm=arm,
            stage=stage,
            lambda_reconstruction=lambda_reconstruction,
            reconstruction_batch=_build_reconstruction_batch(
                reconstruction_prediction, reconstruction_targets),
            depmap_active=expected_depmap is not None,
            lambda_depmap=lambda_depmap,
            depmap_batch=_build_batch(depmap_prediction, depmap_targets),
            nature_active=expected_nature is not None,
            lambda_nature=lambda_nature,
            nature_batch=_build_batch(nature_prediction, nature_targets),
        )

        if not objective.should_optimizer_step or objective.total_loss is None:
            raise RuntimeError("valid auxiliary objective must provide one update loss")
        if optimizer is None:
            raise ValueError(f"{arm} auxiliary reconstruction requires an optimizer")
        dimensions = objective.expression_reconstruction.dimensions
        if (audit.reconstruction_dimensions is not None
                and audit.reconstruction_dimensions != dimensions):
            raise ValueError("reconstruction feature dimensions changed within the audit stage")
        optimizer.zero_grad(set_to_none=True)
        audit.backward_calls += 1
        objective.total_loss.backward()
        optimizer.step()
        audit.optimizer_steps += 1
        audit.reconstruction_dimensions = dimensions
        audit.reconstruction_valid_denominator += (
            objective.expression_reconstruction.valid_denominator
        )
        audit.reconstruction_losses.append(float(
            objective.expression_reconstruction.loss.detach().cpu()))
        audit.depmap_valid_denominators.append(objective.depmap.valid_denominator)
        audit.nature_valid_denominators.append(objective.nature.valid_denominator)
        audit.depmap_losses.append(
            None if objective.depmap.loss is None
            else float(objective.depmap.loss.detach().cpu()))
        audit.nature_losses.append(
            None if objective.nature.loss is None
            else float(objective.nature.loss.detach().cpu()))
        audit.total_losses.append(float(objective.total_loss.detach().cpu()))
        return Exp045AuxiliaryStepResult(
            arm=arm,
            output=output,
            objective=objective,
            optimizer_stepped=True,
        )
    except Exception:
        audit.failures += 1
        raise
