"""Fail-closed D1 outer-fold driver for EXP-045.

The data adapter is responsible for materializing fold-safe tensors from the
approved manifests. This driver owns the matched seven-arm schedule, training,
inner-fold candidate selection, keyed outer predictions, and the run receipt.
Invoking the CLI without ``--train`` is metadata-only and never opens data.

Receipt JSON contract: all three approval files bind ``experiment_id=EXP-045``,
the ``outer_fold``, ``approved_for_training=true``, and a non-empty
``approved_by``. The budget file binds resource allocation, exact CUDA device,
candidate/seed schedule, and the byte hashes of the exposure/input receipt
files. The exposure receipt contains a recomputable metadata chain: an entity
alias-map digest and exclusion-manifest digest; a mask-execution-log digest,
target-fold mask, checkpoint path/hash; and a lineage receipt binding those
child receipt digests, the requested fold, mask, checkpoint, and exclusion set.
Every nested digest is recomputed from canonical JSON and all links are checked
before the adapter is imported. The input receipt binds the exposure receipt,
fold input/split, target-fold mask, both ordered axes, and the same schedule.
This is an authorization gate over supplied metadata receipts; it does not
independently establish that source exposure or masking actually occurred.
Missing or differently bound fields stop before any adapter data read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

from drug_screen.data.exp045_group_permutation import (
    GroupPermutationError,
    project_dependency_group_permutation,
    verify_dependency_group_permutation,
    verify_dependency_group_permutation_projection,
)
from drug_screen.models.exp045_fold_step import (
    Exp045Arm,
    Exp045DependencyTargets,
    Exp045ExpressionTargets,
    Exp045FoldStepAudit,
    Exp045Stage,
    exp045_auxiliary_batch_step,
)
from drug_screen.models.exp045_multitask_objective import exp045_multitask_objective


ARMS: tuple[Exp045Arm, ...] = (
    "D-A", "D-E", "D-C", "D-O", "D-CO", "D-SC", "D-SO",
)
AUXILIARY_ARMS: tuple[Exp045Arm, ...] = ARMS[1:]
STAGES: tuple[Exp045Stage, ...] = (
    "depmap_source_training", "nature_organoid_adaptation",
)
_ACTIVE_DOMAIN: dict[tuple[str, str], tuple[str, Literal["true", "within_domain_shuffled"]] | None] = {
    ("D-E", STAGES[0]): None,
    ("D-E", STAGES[1]): None,
    ("D-C", STAGES[0]): ("depmap", "true"),
    ("D-C", STAGES[1]): None,
    ("D-O", STAGES[0]): None,
    ("D-O", STAGES[1]): ("nature", "true"),
    ("D-CO", STAGES[0]): ("depmap", "true"),
    ("D-CO", STAGES[1]): ("nature", "true"),
    ("D-SC", STAGES[0]): ("depmap", "within_domain_shuffled"),
    ("D-SC", STAGES[1]): ("nature", "true"),
    ("D-SO", STAGES[0]): ("depmap", "true"),
    ("D-SO", STAGES[1]): ("nature", "within_domain_shuffled"),
}


class FormalFoldError(RuntimeError):
    """An EXP-045 formal-fold contract or authorization failure."""


@dataclass(frozen=True)
class Candidate:
    """One pre-registered candidate shared across all seven arms."""

    candidate_id: str
    learning_rate: float
    weight_decay: float
    dss_ranking_weight: float
    lambda_reconstruction: float
    lambda_depmap: float
    lambda_nature: float
    max_epochs: int
    patience: int


@dataclass(frozen=True)
class TrainingEvent:
    """One ordered optimizer opportunity in a training epoch."""

    kind: Literal["dss", "auxiliary"]
    batch_id: str
    stage: Exp045Stage | None = None


@dataclass(frozen=True)
class KeyedDSSBatch:
    """DSS rows keyed by patient/drug identity; target is train-side scaled."""

    batch_id: str
    row_keys: tuple[tuple[str, str], ...]
    patient_ids: tuple[str, ...]
    expression: torch.Tensor
    drug_features: torch.Tensor
    target: torch.Tensor


@dataclass(frozen=True)
class KeyedPredictionInput:
    """Expression/drug inputs for inner validation or frozen outer prediction."""

    row_keys: tuple[tuple[str, str], ...]
    patient_ids: tuple[str, ...]
    expression: torch.Tensor
    drug_features: torch.Tensor


@dataclass(frozen=True)
class AuxiliaryBatch:
    """One source-domain expression batch and the optional assigned labels.

    ``sample_ids`` preserve observation order; ``biological_unit_ids`` keep
    patient/cell-line repeats auditable. A shuffled-label adapter must carry a
    digest of its frozen within-domain permutation receipt.
    """

    batch_id: str
    stage: Exp045Stage
    sample_ids: tuple[str, ...]
    biological_unit_ids: tuple[str, ...]
    expression: torch.Tensor
    expression_axis: tuple[str, ...]
    reconstruction_target: torch.Tensor
    reconstruction_mask: torch.Tensor
    reconstruction_axis: tuple[str, ...]
    dependency_target: torch.Tensor | None = None
    dependency_mask: torch.Tensor | None = None
    dependency_axis: tuple[str, ...] | None = None
    dependency_train_mean: float | None = None
    dependency_train_overall_std: float | None = None
    label_assignment: Literal["true", "within_domain_shuffled"] | None = None
    permutation_receipt_sha256: str | None = None
    permutation_parent_receipt_sha256: str | None = None
    dependency_strata: tuple[str, ...] | None = None
    permutation_source_target: torch.Tensor | None = None
    permutation_receipt: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ArmPartitionSchedule:
    """The exact ordered batch schedule for one arm and one train partition."""

    fold_id: str
    training_patient_ids: tuple[str, ...]
    dss_batches: Mapping[str, KeyedDSSBatch]
    auxiliary_batches: Mapping[Exp045Stage, Mapping[str, AuxiliaryBatch]]
    events: tuple[TrainingEvent, ...]
    permutation_receipts: Mapping[Exp045Stage, Mapping[str, Any]] = field(default_factory=dict)
    permutation_receipt_sha256s: Mapping[Exp045Stage, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingPartition:
    """One inner or outer training partition, with schedules for every arm."""

    fold_id: str
    response_mean: float
    response_scale: float
    schedules: Mapping[Exp045Arm, ArmPartitionSchedule]
    dependency_manifest_sha256s: Mapping[str, Mapping[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class InnerFold:
    """Inner training data and keyed validation observations."""

    fold_id: str
    training: TrainingPartition
    validation_inputs: KeyedPredictionInput
    validation_labels: tuple[tuple[tuple[str, str], float], ...]


@dataclass(frozen=True)
class FoldPlan:
    """Complete outer-fold work order assembled by an approved data adapter."""

    outer_fold: int
    input_axis: tuple[str, ...]
    dependency_axis: tuple[str, ...]
    inner_folds: tuple[InnerFold, ...]
    outer_training: TrainingPartition
    outer_test_inputs: KeyedPredictionInput
    candidates: tuple[Candidate, ...]
    selection_seeds: tuple[int, ...]
    final_seeds: tuple[int, ...]
    candidate_schedule_by_arm: Mapping[Exp045Arm, tuple[str, ...]]
    selection_seed_schedule_by_arm: Mapping[Exp045Arm, tuple[int, ...]]
    final_seed_schedule_by_arm: Mapping[Exp045Arm, tuple[int, ...]]
    inner_fold_schedule_by_arm: Mapping[Exp045Arm, tuple[str, ...]]
    input_contract_sha256: str
    split_sha256: str
    source_checkpoint_path: str
    source_checkpoint_sha256: str
    exclusion_manifest_sha256: str
    target_fold_mask_sha256: str
    checkpoint_lineage_sha256: str
    dependency_dataset_assets: Mapping[str, tuple[Mapping[str, str], ...]]
    dependency_strata_definition_versions: Mapping[str, str]


@dataclass(frozen=True)
class ReceiptEnvelope:
    """A receipt payload plus the SHA-256 of the exact JSON file bytes."""

    payload: Mapping[str, Any]
    sha256: str
    raw_bytes: bytes
    path: str = "<in-memory>"


@dataclass(frozen=True)
class AuthorizationReceipts:
    budget: ReceiptEnvelope | None
    exposure: ReceiptEnvelope | None
    inputs: ReceiptEnvelope | None


@dataclass(frozen=True)
class KeyedPrediction:
    fold: int
    arm: Exp045Arm
    candidate_id: str
    seed: int
    patient_id: str
    drug_id: str
    prediction: float


@dataclass(frozen=True)
class PatientMetric:
    patient_id: str
    spearman: float | None
    n_drugs: int
    reason: str | None


@dataclass(frozen=True)
class KeyedMetricSummary:
    metric: str
    patient_macro_spearman: float | None
    valid_patients: int
    total_patients: int
    patients: tuple[PatientMetric, ...]


@dataclass
class ArmExecutionAudit:
    arm: Exp045Arm
    dss_scheduled: int = 0
    dss_forward_calls: int = 0
    dss_backward_calls: int = 0
    dss_optimizer_steps: int = 0
    auxiliary: dict[str, Exp045FoldStepAudit] = field(default_factory=dict)
    failures: int = 0


@dataclass(frozen=True)
class CandidateSelection:
    arm: Exp045Arm
    candidate_id: str
    mean_inner_patient_macro_spearman: float
    refit_epochs: int
    inner_scores: tuple[float, ...]


@dataclass(frozen=True)
class FoldRunResult:
    fold: int
    status: str
    selections: tuple[CandidateSelection, ...]
    predictions: tuple[KeyedPrediction, ...]
    audits: tuple[Mapping[str, Any], ...]
    receipt_hashes: Mapping[str, str]
    input_contract_sha256: str
    split_sha256: str
    source_checkpoint_path: str
    source_checkpoint_sha256: str
    exclusion_manifest_sha256: str
    target_fold_mask_sha256: str
    checkpoint_lineage_sha256: str
    started_utc: str
    ended_utc: str
    elapsed_wall_seconds: float
    cuda_device: str
    cuda_device_name: str


ModelFactory = Callable[[Exp045Arm, Candidate, int], nn.Module]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256(encoded.encode("utf-8"))


def dependency_dataset_assets_sha256(plan: FoldPlan) -> str:
    normalized = {
        domain: sorted(
            [dict(asset) for asset in assets],
            key=lambda item: (str(item.get("dataset_id", "")).encode("utf-8"),
                              str(item.get("version", "")).encode("utf-8")),
        )
        for domain, assets in sorted(plan.dependency_dataset_assets.items())
    }
    return _canonical_hash(normalized)


def dependency_partition_manifests_sha256(plan: FoldPlan) -> str:
    partitions = [
        {"fold_id": inner.training.fold_id,
         "manifests": inner.training.dependency_manifest_sha256s}
        for inner in plan.inner_folds
    ]
    partitions.append({
        "fold_id": plan.outer_training.fold_id,
        "manifests": plan.outer_training.dependency_manifest_sha256s,
    })
    return _canonical_hash(partitions)


def dependency_strata_definitions_sha256(plan: FoldPlan) -> str:
    return _canonical_hash(dict(sorted(plan.dependency_strata_definition_versions.items())))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _validate_nested_receipt(
    exposure: Mapping[str, Any], *, name: str, required_fields: set[str], outer_fold: int,
) -> tuple[Mapping[str, Any], str]:
    """Recompute a metadata child receipt digest and reject undeclared payload fields."""
    envelope = exposure.get(name)
    if not isinstance(envelope, Mapping):
        raise FormalFoldError(f"exposure receipt is missing nested {name}")
    payload = envelope.get("payload")
    digest = envelope.get("sha256")
    if not isinstance(payload, Mapping) or set(payload) != required_fields:
        raise FormalFoldError(f"nested {name} payload fields do not match the metadata-only contract")
    if not _is_sha256(digest) or _canonical_hash(payload) != digest:
        raise FormalFoldError(f"nested {name} receipt digest does not match its canonical payload")
    if payload.get("outer_fold") != outer_fold:
        raise FormalFoldError(f"nested {name} receipt is bound to a different target fold")
    if payload.get("status") != "PASS":
        raise FormalFoldError(f"nested {name} receipt status is not PASS")
    return payload, digest


def _validate_receipt_file_sha256(path_value: object, expected_sha256: object, *, name: str) -> None:
    """Hash raw bytes of a declared support artifact without parsing or loading it."""
    if not isinstance(path_value, str) or not path_value.strip():
        raise FormalFoldError(f"{name} path is missing")
    if not _is_sha256(expected_sha256):
        raise FormalFoldError(f"{name} byte SHA-256 is missing or malformed")
    path = Path(path_value)
    try:
        if not path.is_file():
            raise FormalFoldError(f"{name} path is missing or is not a regular file: {path}")
        actual_sha256 = _file_sha256(path)
    except OSError as exc:
        raise FormalFoldError(f"{name} cannot be read for byte-hash verification: {path}") from exc
    if actual_sha256 != expected_sha256:
        raise FormalFoldError(f"{name} raw-byte SHA-256 does not match its receipt: {path}")


def axis_sha256(axis: Sequence[str]) -> str:
    return _sha256("\n".join(axis).encode("utf-8"))


def load_receipt(path: Path) -> ReceiptEnvelope:
    """Load one small approval receipt; this does not open any scientific input."""
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise FormalFoldError(f"receipt must be a JSON object: {path}")
    return ReceiptEnvelope(payload=payload, sha256=_sha256(raw), raw_bytes=raw, path=str(path))


def candidate_seed_schedule_sha256(plan: FoldPlan) -> str:
    schedule = {
        "candidates": [asdict(candidate) for candidate in plan.candidates],
        "selection_seeds": list(plan.selection_seeds),
        "final_seeds": list(plan.final_seeds),
        "inner_folds": [fold.fold_id for fold in plan.inner_folds],
        "candidate_schedule_by_arm": {
            arm: list(plan.candidate_schedule_by_arm[arm]) for arm in ARMS
        },
        "selection_seed_schedule_by_arm": {
            arm: list(plan.selection_seed_schedule_by_arm[arm]) for arm in ARMS
        },
        "final_seed_schedule_by_arm": {
            arm: list(plan.final_seed_schedule_by_arm[arm]) for arm in ARMS
        },
        "inner_fold_schedule_by_arm": {
            arm: list(plan.inner_fold_schedule_by_arm[arm]) for arm in ARMS
        },
    }
    return _canonical_hash(schedule)


def _require_tuple_ids(name: str, values: Sequence[str], *, nonempty: bool = True) -> None:
    if not isinstance(values, tuple) or (nonempty and not values):
        raise FormalFoldError(f"{name} must be a non-empty immutable tuple")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise FormalFoldError(f"{name} contains a missing/non-string identity")
    if len(set(values)) != len(values):
        raise FormalFoldError(f"{name} contains duplicate identities")


def _tensor_equal(left: torch.Tensor | None, right: torch.Tensor | None) -> bool:
    if left is None or right is None:
        return left is right
    return (
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and left.shape == right.shape
        and left.dtype == right.dtype
        and left.device == right.device
        and torch.equal(left, right)
    )


def _tensor_equal_with_nan(left: torch.Tensor | None, right: torch.Tensor | None) -> bool:
    if left is None or right is None:
        return left is right
    if (not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor)
            or left.shape != right.shape or left.dtype != right.dtype or left.device != right.device):
        return False
    if left.is_floating_point() and right.is_floating_point():
        return bool(torch.logical_or(left == right, torch.isnan(left) & torch.isnan(right)).all())
    return torch.equal(left, right)


def _check_tensor(name: str, value: object, *, ndim: int, floating: bool = True) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise FormalFoldError(f"{name} must be a {ndim}D torch tensor")
    if floating and not value.is_floating_point():
        raise FormalFoldError(f"{name} must be floating point")
    if not floating and value.dtype != torch.bool:
        raise FormalFoldError(f"{name} must be boolean")
    if value.numel() < 1:
        raise FormalFoldError(f"{name} must be non-empty")
    return value


def _validate_candidate_table(plan: FoldPlan) -> None:
    if not plan.candidates:
        raise FormalFoldError("the frozen candidate table is missing")
    if any(not isinstance(candidate, Candidate) for candidate in plan.candidates):
        raise FormalFoldError("candidate table contains an invalid entry")
    candidate_ids = tuple(candidate.candidate_id for candidate in plan.candidates)
    _require_tuple_ids("candidate IDs", candidate_ids)
    for candidate in plan.candidates:
        if not isinstance(candidate, Candidate):
            raise FormalFoldError("candidate table contains an invalid entry")
        if not isinstance(candidate.max_epochs, int) or candidate.max_epochs < 1:
            raise FormalFoldError(f"{candidate.candidate_id}: max_epochs must be positive")
        if not isinstance(candidate.patience, int) or not 1 <= candidate.patience <= candidate.max_epochs:
            raise FormalFoldError(f"{candidate.candidate_id}: patience is outside its epoch budget")
        for name, value, positive in (
            ("learning_rate", candidate.learning_rate, True),
            ("weight_decay", candidate.weight_decay, False),
            ("dss_ranking_weight", candidate.dss_ranking_weight, False),
            ("lambda_reconstruction", candidate.lambda_reconstruction, True),
            ("lambda_depmap", candidate.lambda_depmap, True),
            ("lambda_nature", candidate.lambda_nature, True),
        ):
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise FormalFoldError(f"{candidate.candidate_id}: {name} is missing")
            number = float(value)
            if not math.isfinite(number) or number < 0 or (positive and number <= 0):
                raise FormalFoldError(f"{candidate.candidate_id}: invalid {name}")
    for name, seeds in (("candidate-selection", plan.selection_seeds), ("final", plan.final_seeds)):
        if not isinstance(seeds, tuple) or not seeds:
            raise FormalFoldError(f"the frozen {name} seed schedule is missing")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
            raise FormalFoldError(f"{name} seed schedule must contain integer seeds")
        if len(set(seeds)) != len(seeds):
            raise FormalFoldError(f"{name} seed schedule contains duplicate seeds")
    expected_candidates = candidate_ids
    expected_inner = tuple(fold.fold_id for fold in plan.inner_folds)
    if not expected_inner or len(set(expected_inner)) != len(expected_inner):
        raise FormalFoldError("inner fold schedule must be non-empty and unique")
    for arm in ARMS:
        if tuple(plan.candidate_schedule_by_arm.get(arm, ())) != expected_candidates:
            raise FormalFoldError(f"{arm} candidate/search opportunities differ")
        if tuple(plan.selection_seed_schedule_by_arm.get(arm, ())) != plan.selection_seeds:
            raise FormalFoldError(f"{arm} candidate-selection seed schedule differs")
        if tuple(plan.final_seed_schedule_by_arm.get(arm, ())) != plan.final_seeds:
            raise FormalFoldError(f"{arm} final seed schedule differs")
        if tuple(plan.inner_fold_schedule_by_arm.get(arm, ())) != expected_inner:
            raise FormalFoldError(f"{arm} inner-fold schedule differs")


def _validate_dss_batch(batch: KeyedDSSBatch, input_axis: tuple[str, ...]) -> None:
    if not isinstance(batch, KeyedDSSBatch) or not batch.batch_id:
        raise FormalFoldError("invalid DSS training batch")
    if not batch.row_keys or len(batch.row_keys) != len(batch.patient_ids):
        raise FormalFoldError(f"{batch.batch_id}: DSS keys and patient IDs do not align")
    if len(set(batch.row_keys)) != len(batch.row_keys):
        raise FormalFoldError(f"{batch.batch_id}: duplicate patient/drug training keys")
    for key, patient in zip(batch.row_keys, batch.patient_ids):
        if (not isinstance(key, tuple) or len(key) != 2
                or not all(isinstance(part, str) and part for part in key)
                or key[0] != patient):
            raise FormalFoldError(f"{batch.batch_id}: invalid keyed patient/drug row")
    expression = _check_tensor("DSS expression", batch.expression, ndim=2)
    drug = _check_tensor("DSS drug features", batch.drug_features, ndim=2)
    target = _check_tensor("DSS target", batch.target, ndim=1)
    if expression.shape != (len(batch.row_keys), len(input_axis)):
        raise FormalFoldError(f"{batch.batch_id}: DSS expression axis or row count mismatch")
    if drug.shape[0] != len(batch.row_keys) or target.shape != (len(batch.row_keys),):
        raise FormalFoldError(f"{batch.batch_id}: DSS features/labels do not align by key")
    if not bool(torch.isfinite(expression).all() and torch.isfinite(drug).all()
                and torch.isfinite(target).all()):
        raise FormalFoldError(f"{batch.batch_id}: non-finite DSS training values")


def _validate_auxiliary_batch(
    batch: AuxiliaryBatch,
    *,
    arm: Exp045Arm,
    stage: Exp045Stage,
    input_axis: tuple[str, ...],
    dependency_axis: tuple[str, ...],
    fold_id: str,
    input_contract_sha256: str,
    split_sha256: str,
) -> None:
    if not isinstance(batch, AuxiliaryBatch) or batch.stage != stage or not batch.batch_id:
        raise FormalFoldError(f"{arm}/{stage}: invalid auxiliary batch identity")
    _require_tuple_ids(f"{arm}/{stage} sample IDs", batch.sample_ids)
    if not isinstance(batch.biological_unit_ids, tuple) or len(batch.biological_unit_ids) != len(batch.sample_ids):
        raise FormalFoldError(f"{arm}/{stage}: biological-unit IDs do not align with samples")
    if any(not isinstance(unit, str) or not unit for unit in batch.biological_unit_ids):
        raise FormalFoldError(f"{arm}/{stage}: missing biological-unit identity")
    expression = _check_tensor(f"{arm}/{stage} expression", batch.expression, ndim=2)
    target = _check_tensor(f"{arm}/{stage} reconstruction target", batch.reconstruction_target, ndim=2)
    mask = _check_tensor(f"{arm}/{stage} reconstruction mask", batch.reconstruction_mask,
                         ndim=2, floating=False)
    if (batch.expression_axis != input_axis or batch.reconstruction_axis != input_axis
            or expression.shape != target.shape or expression.shape != mask.shape
            or expression.shape != (len(batch.sample_ids), len(input_axis))):
        raise FormalFoldError(f"{arm}/{stage}: expression axis/target/mask mismatch")
    if not bool(mask.any()):
        raise FormalFoldError(f"{arm}/{stage}: no valid expression reconstruction values")
    if not bool(torch.isfinite(expression).all() and torch.isfinite(target[mask]).all()):
        raise FormalFoldError(f"{arm}/{stage}: non-finite valid expression values")

    expected = _ACTIVE_DOMAIN[(arm, stage)]
    if expected is None:
        if any(value is not None for value in (
            batch.dependency_target, batch.dependency_mask, batch.dependency_axis,
            batch.dependency_train_mean, batch.dependency_train_overall_std,
            batch.label_assignment, batch.permutation_receipt_sha256,
            batch.permutation_parent_receipt_sha256,
            batch.dependency_strata, batch.permutation_source_target,
            batch.permutation_receipt,
        )):
            raise FormalFoldError(f"{arm}/{stage}: inactive dependency labels must be absent")
        return

    domain, assignment = expected
    if batch.label_assignment != assignment:
        raise FormalFoldError(f"{arm}/{stage}: expected {domain} labels assigned as {assignment}")
    dep_target = _check_tensor(f"{arm}/{stage} dependency target", batch.dependency_target, ndim=2)
    dep_mask = _check_tensor(f"{arm}/{stage} dependency mask", batch.dependency_mask,
                             ndim=2, floating=False)
    if (batch.dependency_axis != dependency_axis
            or dep_target.shape != dep_mask.shape
            or dep_target.shape != (len(batch.sample_ids), len(dependency_axis))
            or not bool(dep_mask.any())):
        raise FormalFoldError(f"{arm}/{stage}: missing, empty, or misaligned dependency supervision")
    mean = batch.dependency_train_mean
    std = batch.dependency_train_overall_std
    if (isinstance(mean, bool) or not isinstance(mean, (float, int)) or not math.isfinite(float(mean))
            or isinstance(std, bool) or not isinstance(std, (float, int))
            or not math.isfinite(float(std)) or float(std) <= 0):
        raise FormalFoldError(f"{arm}/{stage}: dependency scale must be fit and valid on training data")
    if not bool(torch.isfinite(dep_target[dep_mask]).all()):
        raise FormalFoldError(f"{arm}/{stage}: non-finite valid dependency labels")
    if (not isinstance(batch.dependency_strata, tuple)
            or len(batch.dependency_strata) != len(batch.sample_ids)
            or any(not isinstance(value, str) or not value for value in batch.dependency_strata)):
        raise FormalFoldError(f"{arm}/{stage}: dependency strata do not align with samples")
    if assignment == "within_domain_shuffled":
        source_target = _check_tensor(
            f"{arm}/{stage} permutation source target", batch.permutation_source_target, ndim=2,
        )
        if (source_target.shape != dep_target.shape
                or not isinstance(batch.permutation_receipt, Mapping)
                or not _is_sha256(batch.permutation_receipt_sha256)
                or not _is_sha256(batch.permutation_parent_receipt_sha256)):
            raise FormalFoldError(
                f"{arm}/{stage}: shuffled labels require original targets and parent/projection receipts"
            )
        projection = batch.permutation_receipt
        if (projection.get("schema") != "exp045_group_permutation_projection_v1"
                or projection.get("batch_id") != batch.batch_id
                or projection.get("parent_receipt_sha256") != batch.permutation_parent_receipt_sha256
                or projection.get("ordered_observation_ids") != list(batch.sample_ids)):
            raise FormalFoldError(f"{arm}/{stage}: batch projection receipt is bound to different rows")
        if _canonical_hash(projection) != batch.permutation_receipt_sha256:
            raise FormalFoldError(f"{arm}/{stage}: batch projection receipt digest is invalid")
    elif (batch.permutation_receipt_sha256 is not None
          or batch.permutation_parent_receipt_sha256 is not None
          or batch.permutation_source_target is not None
          or batch.permutation_receipt is not None):
        raise FormalFoldError(f"{arm}/{stage}: true labels cannot carry a shuffle receipt or source copy")


def _aux_signature(batch: AuxiliaryBatch) -> tuple[object, ...]:
    return (
        batch.batch_id, batch.stage, batch.sample_ids, batch.biological_unit_ids,
        batch.expression_axis, batch.reconstruction_axis,
        batch.expression, batch.reconstruction_target, batch.reconstruction_mask,
    )


def _same_aux_inputs(left: AuxiliaryBatch, right: AuxiliaryBatch) -> bool:
    left_sig, right_sig = _aux_signature(left), _aux_signature(right)
    return (
        left_sig[:6] == right_sig[:6]
        and _tensor_equal(left.expression, right.expression)
        and _tensor_equal_with_nan(left.reconstruction_target, right.reconstruction_target)
        and _tensor_equal(left.reconstruction_mask, right.reconstruction_mask)
    )


def _collect_dependency_partition_rows(
    batches: Mapping[str, AuxiliaryBatch],
    *,
    use_permuted_output: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray]:
    rows: dict[str, tuple[str, str, np.ndarray, np.ndarray]] = {}
    for batch in batches.values():
        if (batch.dependency_target is None or batch.dependency_mask is None
                or batch.dependency_strata is None):
            raise FormalFoldError(f"{batch.batch_id}: dependency partition row is missing")
        targets = batch.dependency_target
        if not use_permuted_output and batch.label_assignment == "within_domain_shuffled":
            targets = batch.permutation_source_target
        if targets is None:
            raise FormalFoldError(f"{batch.batch_id}: original dependency targets are missing")
        target_values = targets.detach().cpu().numpy().astype(np.float64, copy=True)
        mask_values = batch.dependency_mask.detach().cpu().numpy().astype(np.bool_, copy=True)
        for index, observation_id in enumerate(batch.sample_ids):
            current = (
                batch.biological_unit_ids[index], batch.dependency_strata[index],
                target_values[index], mask_values[index],
            )
            prior = rows.get(observation_id)
            if prior is None:
                rows[observation_id] = current
                continue
            if (prior[0] != current[0] or prior[1] != current[1]
                    or not np.array_equal(prior[2], current[2], equal_nan=True)
                    or not np.array_equal(prior[3], current[3])):
                raise FormalFoldError(
                    f"{observation_id}: repeated training-row exposures have inconsistent identity, label, or mask"
                )
    ordered_ids = tuple(sorted(rows, key=lambda value: value.encode("utf-8")))
    groups = tuple(rows[observation_id][0] for observation_id in ordered_ids)
    strata = tuple(rows[observation_id][1] for observation_id in ordered_ids)
    targets = np.stack([rows[observation_id][2] for observation_id in ordered_ids], axis=0)
    masks = np.stack([rows[observation_id][3] for observation_id in ordered_ids], axis=0)
    return ordered_ids, groups, strata, targets, masks


def _partition_row_manifests(
    observation_ids: Sequence[str], biological_group_ids: Sequence[str],
) -> dict[str, str]:
    ordered_observations = sorted(set(observation_ids), key=lambda value: value.encode("utf-8"))
    ordered_groups = sorted(set(biological_group_ids), key=lambda value: value.encode("utf-8"))
    return {
        "observation_ids_sha256": _canonical_hash(ordered_observations),
        "biological_group_ids_sha256": _canonical_hash(ordered_groups),
    }


def _validate_partition_group_permutation_receipts(
    partition: TrainingPartition,
    plan: FoldPlan,
) -> None:
    required = {
        "D-SC": (STAGES[0], "depmap"),
        "D-SO": (STAGES[1], "nature"),
    }
    for arm in ARMS:
        schedule = partition.schedules[arm]
        expected_stages = {required[arm][0]} if arm in required else set()
        if (set(schedule.permutation_receipts) != expected_stages
                or set(schedule.permutation_receipt_sha256s) != expected_stages):
            raise FormalFoldError(f"{arm}: full-partition permutation receipt set is incomplete or unexpected")
    if set(partition.dependency_manifest_sha256s) != {"depmap", "nature"}:
        raise FormalFoldError("complete DepMap/Nature source training manifests are required")

    for arm, (stage, domain) in required.items():
        shuffled_schedule = partition.schedules[arm]
        parent_receipt = shuffled_schedule.permutation_receipts[stage]
        parent_sha = shuffled_schedule.permutation_receipt_sha256s[stage]
        if not isinstance(parent_receipt, Mapping) or not _is_sha256(parent_sha):
            raise FormalFoldError(f"{arm}/{stage}: malformed full-partition parent receipt")
        if _canonical_hash(parent_receipt) != parent_sha:
            raise FormalFoldError(f"{arm}/{stage}: full-partition parent receipt digest is invalid")

        true_schedule = partition.schedules["D-CO"]
        true_batches = true_schedule.auxiliary_batches[stage]
        shuffled_batches = shuffled_schedule.auxiliary_batches[stage]
        if tuple(true_batches) != tuple(shuffled_batches):
            raise FormalFoldError(f"{arm}/{stage}: full-partition and projected batch schedules differ")
        observation_ids, group_ids, strata, source_targets, _ = _collect_dependency_partition_rows(
            true_batches, use_permuted_output=False,
        )
        output_ids, output_groups, output_strata, output_targets, output_mask = (
            _collect_dependency_partition_rows(shuffled_batches, use_permuted_output=True)
        )
        if (observation_ids != output_ids or group_ids != output_groups or strata != output_strata):
            raise FormalFoldError(f"{arm}/{stage}: shuffled schedule omits or changes full-partition row identities")
        expected_manifests = _partition_row_manifests(observation_ids, group_ids)
        if partition.dependency_manifest_sha256s[domain] != expected_manifests:
            raise FormalFoldError(f"{domain}: full-partition observation/group manifest differs from the adapter input")

        try:
            permutation = verify_dependency_group_permutation(
                source_targets=source_targets,
                output_targets=output_targets,
                output_mask=output_mask,
                observation_ids=observation_ids,
                biological_group_ids=group_ids,
                row_strata=strata,
                dependency_axis=plan.dependency_axis,
                expected_fold_id=partition.fold_id,
                expected_partition_id=f"{partition.fold_id}:training",
                expected_stage=stage,
                expected_domain=domain,
                expected_batch_id=f"{partition.fold_id}:{stage}:{domain}:full_partition",
                expected_dataset_assets=plan.dependency_dataset_assets[domain],
                expected_strata_definition_version=plan.dependency_strata_definition_versions[domain],
                expected_input_contract_sha256=plan.input_contract_sha256,
                expected_split_sha256=plan.split_sha256,
                expected_training_mean=float(true_batches[next(iter(true_batches))].dependency_train_mean),
                expected_training_std=float(true_batches[next(iter(true_batches))].dependency_train_overall_std),
                receipt=parent_receipt,
                receipt_sha256=parent_sha,
            )
        except (GroupPermutationError, TypeError, ValueError) as exc:
            raise FormalFoldError(f"{arm}/{stage}: full-partition group permutation failed recomputation") from exc

        if (parent_receipt.get("training_partition_observation_manifest_sha256")
                != expected_manifests["observation_ids_sha256"]
                or parent_receipt.get("training_partition_group_manifest_sha256")
                != expected_manifests["biological_group_ids_sha256"]):
            raise FormalFoldError(f"{arm}/{stage}: parent receipt does not cover the complete training partition")
        if not np.array_equal(permutation.mask, output_mask):
            raise FormalFoldError(f"{arm}/{stage}: projected effective mask differs from full-partition mask")

        for batch_id, batch in shuffled_batches.items():
            if batch.permutation_parent_receipt_sha256 != parent_sha:
                raise FormalFoldError(f"{arm}/{stage}/{batch_id}: child projection is bound to a different parent")
            try:
                verify_dependency_group_permutation_projection(
                    partition_result=permutation,
                    batch_id=batch_id,
                    observation_ids=batch.sample_ids,
                    output_targets=batch.dependency_target.detach().cpu().numpy(),
                    output_mask=batch.dependency_mask.detach().cpu().numpy(),
                    receipt=batch.permutation_receipt,
                    receipt_sha256=batch.permutation_receipt_sha256,
                )
            except GroupPermutationError as exc:
                raise FormalFoldError(f"{arm}/{stage}/{batch_id}: child projection failed recomputation") from exc

        for true_arm in AUXILIARY_ARMS:
            for batch_id, batch in partition.schedules[true_arm].auxiliary_batches[stage].items():
                if batch.label_assignment != "true":
                    continue
                expected_projection = project_dependency_group_permutation(
                    permutation, batch_id=batch_id, observation_ids=batch.sample_ids,
                )
                observed_mask = batch.dependency_mask.detach().cpu().numpy().astype(np.bool_, copy=False)
                if not np.array_equal(observed_mask, expected_projection.mask):
                    raise FormalFoldError(
                        f"{true_arm}/{stage}/{batch_id}: true and shuffled arms do not share the full-fold effective mask"
                    )


def _same_dss_batch(left: KeyedDSSBatch, right: KeyedDSSBatch) -> bool:
    return (
        left.batch_id == right.batch_id
        and left.row_keys == right.row_keys
        and left.patient_ids == right.patient_ids
        and _tensor_equal(left.expression, right.expression)
        and _tensor_equal(left.drug_features, right.drug_features)
        and _tensor_equal(left.target, right.target)
    )


def validate_partition(partition: TrainingPartition, plan: FoldPlan) -> None:
    """Fail before optimization if any arm's identities, axes, masks, or steps differ."""
    if not isinstance(partition, TrainingPartition) or not partition.fold_id:
        raise FormalFoldError("training partition/fold identity is missing")
    if set(partition.schedules) != set(ARMS):
        raise FormalFoldError("the partition must contain exactly all seven registered arms")
    if (not math.isfinite(partition.response_mean) or not math.isfinite(partition.response_scale)
            or partition.response_scale <= 0):
        raise FormalFoldError("DSS response transform must be finite and fit on this training partition")
    baseline = partition.schedules["D-A"]
    if baseline.fold_id != partition.fold_id:
        raise FormalFoldError("D-A schedule is bound to a different fold")
    _require_tuple_ids("partition training patient IDs", baseline.training_patient_ids)

    baseline_dss_projection = tuple(event for event in baseline.events if event.kind == "dss")
    baseline_dss_batches = baseline.dss_batches
    for arm in ARMS:
        schedule = partition.schedules[arm]
        if schedule.fold_id != partition.fold_id:
            raise FormalFoldError(f"{arm} schedule is bound to a different fold")
        if schedule.training_patient_ids != baseline.training_patient_ids:
            raise FormalFoldError(f"{arm} training patient set/order differs from D-A")
        if not schedule.events:
            raise FormalFoldError(f"{arm} update schedule is empty")
        for event in schedule.events:
            if not isinstance(event, TrainingEvent) or not event.batch_id:
                raise FormalFoldError(f"{arm} contains an invalid scheduled event")
            if event.kind == "dss":
                if event.stage is not None or event.batch_id not in schedule.dss_batches:
                    raise FormalFoldError(f"{arm} has an invalid DSS schedule reference")
            elif event.kind == "auxiliary":
                if arm == "D-A" or event.stage not in STAGES:
                    raise FormalFoldError(f"{arm} has an invalid auxiliary schedule event")
                if event.batch_id not in schedule.auxiliary_batches.get(event.stage, {}):
                    raise FormalFoldError(f"{arm}/{event.stage} references a missing auxiliary batch")
            else:
                raise FormalFoldError(f"{arm} has an unknown update event kind")
        dss_ids = {event.batch_id for event in schedule.events if event.kind == "dss"}
        if not dss_ids or dss_ids != set(schedule.dss_batches):
            raise FormalFoldError(f"{arm} DSS schedule omits or adds training batches")
        if tuple(event for event in schedule.events if event.kind == "dss") != baseline_dss_projection:
            raise FormalFoldError(f"{arm} DSS update opportunities/order differ from D-A")
        if set(schedule.dss_batches) != set(baseline_dss_batches):
            raise FormalFoldError(f"{arm} DSS batch identities differ from D-A")
        for batch_id, batch in schedule.dss_batches.items():
            _validate_dss_batch(batch, plan.input_axis)
            if not _same_dss_batch(batch, baseline_dss_batches[batch_id]):
                raise FormalFoldError(f"{arm} DSS batch content/order differs from D-A: {batch_id}")

        if arm == "D-A":
            if schedule.auxiliary_batches or any(event.kind == "auxiliary" for event in schedule.events):
                raise FormalFoldError("D-A must remain a DSS-only baseline")
            continue

        if set(schedule.auxiliary_batches) != set(STAGES):
            raise FormalFoldError(f"{arm} must provide both registered auxiliary stages")
        for stage in STAGES:
            stage_events = [event for event in schedule.events
                            if event.kind == "auxiliary" and event.stage == stage]
            stage_batches = schedule.auxiliary_batches[stage]
            if not stage_events or {event.batch_id for event in stage_events} != set(stage_batches):
                raise FormalFoldError(f"{arm}/{stage} scheduled batches are missing or unused")
            first_seen_batch_order = tuple(dict.fromkeys(event.batch_id for event in stage_events))
            if first_seen_batch_order != tuple(stage_batches):
                raise FormalFoldError(f"{arm}/{stage} batch-map order differs from the frozen schedule")
            for batch_id, batch in stage_batches.items():
                if batch_id != batch.batch_id:
                    raise FormalFoldError(f"{arm}/{stage} mapping key does not match batch identity")
                _validate_auxiliary_batch(
                    batch, arm=arm, stage=stage,
                    input_axis=plan.input_axis, dependency_axis=plan.dependency_axis,
                    fold_id=partition.fold_id,
                    input_contract_sha256=plan.input_contract_sha256,
                    split_sha256=plan.split_sha256,
                )
        stage_order = [STAGES.index(event.stage) for event in schedule.events
                       if event.kind == "auxiliary" and event.stage in STAGES]
        if stage_order != sorted(stage_order):
            raise FormalFoldError(f"{arm} violates the frozen DepMap-then-Nature stage order")

    # The six auxiliary arms share the ordered samples, expression target/mask,
    # and opportunities. Dependency targets may differ only by the registered
    # true-versus-within-domain-shuffled assignment.
    reference = partition.schedules[AUXILIARY_ARMS[0]]
    reference_aux_events = tuple(event for event in reference.events if event.kind == "auxiliary")
    for arm in AUXILIARY_ARMS[1:]:
        schedule = partition.schedules[arm]
        if tuple(event for event in schedule.events if event.kind == "auxiliary") != reference_aux_events:
            raise FormalFoldError(f"{arm} auxiliary sample/batch opportunities differ")
        for stage in STAGES:
            left_batches = reference.auxiliary_batches[stage]
            right_batches = schedule.auxiliary_batches[stage]
            if tuple(left_batches) != tuple(right_batches):
                raise FormalFoldError(f"{arm}/{stage} auxiliary batch order differs")
            for batch_id in left_batches:
                left, right = left_batches[batch_id], right_batches[batch_id]
                if not _same_aux_inputs(left, right):
                    raise FormalFoldError(
                        f"{arm}/{stage}/{batch_id} expression identity/axis/target/mask differs"
                    )
    # D-E intentionally has no dependency labels, so compare supervision among
    # the arms that actually use this stage's domain instead of using D-E as the
    # reference for that part of the contract.
    for stage in STAGES:
        # Compare supervision across arms for each scheduled batch separately.
        # Different batches are expected to contain different sample IDs and
        # may have different valid-label masks; only their same-batch copies
        # must agree across arms.
        stage_batch_ids = tuple(reference.auxiliary_batches[stage])
        for batch_id in stage_batch_ids:
            active_batches = [
                partition.schedules[arm].auxiliary_batches[stage][batch_id]
                for arm in AUXILIARY_ARMS
                if partition.schedules[arm].auxiliary_batches[stage][batch_id].dependency_target is not None
            ]
            if not active_batches:
                raise FormalFoldError(
                    f"{stage}/{batch_id} has no available supervised target in any registered arm"
                )
            reference_target = active_batches[0]
            for batch in active_batches[1:]:
                if (batch.batch_id != reference_target.batch_id
                        or batch.sample_ids != reference_target.sample_ids
                        or batch.dependency_axis != reference_target.dependency_axis
                        or batch.dependency_strata != reference_target.dependency_strata
                        or batch.dependency_train_mean != reference_target.dependency_train_mean
                        or batch.dependency_train_overall_std != reference_target.dependency_train_overall_std
                        or not _tensor_equal(batch.dependency_mask, reference_target.dependency_mask)):
                    raise FormalFoldError(
                        f"{batch.batch_id}/{stage} dependency axis/mask/scale differs across supervised arms"
                    )
            true_targets = [batch for batch in active_batches if batch.label_assignment == "true"]
            if not true_targets:
                raise FormalFoldError(f"{stage}/{batch_id} has no true-label reference arm")
            true_reference = true_targets[0]
            for batch in true_targets[1:]:
                if not _tensor_equal_with_nan(batch.dependency_target, true_reference.dependency_target):
                    raise FormalFoldError(
                        f"{batch.batch_id}/{stage} true dependency labels differ across arms"
                    )
            for batch in active_batches:
                if batch.label_assignment == "within_domain_shuffled" and not _tensor_equal_with_nan(
                    batch.permutation_source_target, true_reference.dependency_target,
                ):
                    raise FormalFoldError(
                        f"{batch.batch_id}/{stage} shuffle source differs from matched true-label arm"
                    )

    _validate_partition_group_permutation_receipts(partition, plan)


def validate_fold_plan(plan: FoldPlan) -> None:
    """Validate every inner and outer schedule before constructing a model."""
    if not isinstance(plan, FoldPlan) or isinstance(plan.outer_fold, bool) or plan.outer_fold not in range(5):
        raise FormalFoldError("outer fold must be one of the five frozen patient folds")
    _require_tuple_ids("input gene axis", plan.input_axis)
    _require_tuple_ids("dependency output gene axis", plan.dependency_axis)
    if set(plan.dependency_dataset_assets) != {"depmap", "nature"}:
        raise FormalFoldError("registered DepMap and Nature source asset manifests are required")
    if set(plan.dependency_strata_definition_versions) != {"depmap", "nature"}:
        raise FormalFoldError("DepMap and Nature cancer/library strata definitions are required")
    for domain in ("depmap", "nature"):
        assets = plan.dependency_dataset_assets[domain]
        if not isinstance(assets, tuple) or not assets:
            raise FormalFoldError(f"{domain}: registered source assets must be a non-empty tuple")
        seen_assets: set[tuple[str, str]] = set()
        for asset in assets:
            if (not isinstance(asset, Mapping)
                    or not isinstance(asset.get("dataset_id"), str)
                    or not asset["dataset_id"].strip()
                    or not isinstance(asset.get("version"), str)
                    or not asset["version"].strip()
                    or not _is_sha256(asset.get("sha256"))):
                raise FormalFoldError(f"{domain}: invalid registered asset/version/checksum entry")
            asset_identity = (asset["dataset_id"], asset["version"])
            if asset_identity in seen_assets:
                raise FormalFoldError(f"{domain}: duplicate registered asset identity")
            seen_assets.add(asset_identity)
        strata_version = plan.dependency_strata_definition_versions[domain]
        if not isinstance(strata_version, str) or not strata_version.strip():
            raise FormalFoldError(f"{domain}: cancer/library strata definition version is missing")
    for name, digest in (
        ("input contract", plan.input_contract_sha256),
        ("split", plan.split_sha256),
        ("source checkpoint", plan.source_checkpoint_sha256),
        ("exclusion manifest", plan.exclusion_manifest_sha256),
        ("target-fold mask", plan.target_fold_mask_sha256),
        ("checkpoint exposure lineage", plan.checkpoint_lineage_sha256),
    ):
        if not _is_sha256(digest):
            raise FormalFoldError(f"{name} SHA-256 is missing or malformed")
    if not isinstance(plan.source_checkpoint_path, str) or not plan.source_checkpoint_path.strip():
        raise FormalFoldError("source checkpoint path is missing from the fold plan")
    _validate_candidate_table(plan)
    expected_inner_ids = tuple(fold.fold_id for fold in plan.inner_folds)
    if not plan.inner_folds:
        raise FormalFoldError("frozen inner patient folds are missing")
    if len(set(expected_inner_ids)) != len(expected_inner_ids):
        raise FormalFoldError("inner patient fold identities are duplicated")
    for arm in ARMS:
        if tuple(plan.inner_fold_schedule_by_arm[arm]) != expected_inner_ids:
            raise FormalFoldError(f"{arm} inner patient split schedule differs")
    for inner in plan.inner_folds:
        if inner.training.fold_id != inner.fold_id:
            raise FormalFoldError(f"inner fold {inner.fold_id} training schedule is misbound")
        validate_partition(inner.training, plan)
        _validate_prediction_inputs(inner.validation_inputs, plan.input_axis)
        _validate_keyed_labels(inner.validation_labels, inner.validation_inputs.row_keys)
    if plan.outer_training.fold_id != f"outer-{plan.outer_fold}":
        raise FormalFoldError("outer training schedule is not bound to the requested outer fold")
    validate_partition(plan.outer_training, plan)
    _validate_prediction_inputs(plan.outer_test_inputs, plan.input_axis)


def _validate_prediction_inputs(inputs: KeyedPredictionInput, input_axis: tuple[str, ...]) -> None:
    if not isinstance(inputs, KeyedPredictionInput) or not inputs.row_keys:
        raise FormalFoldError("keyed validation/test prediction inputs are missing")
    if len(set(inputs.row_keys)) != len(inputs.row_keys) or len(inputs.row_keys) != len(inputs.patient_ids):
        raise FormalFoldError("prediction pair keys must be unique and aligned to patients")
    if any(key[0] != patient or not key[1] for key, patient in zip(inputs.row_keys, inputs.patient_ids)):
        raise FormalFoldError("prediction patient/drug keys are inconsistent")
    expression = _check_tensor("keyed prediction expression", inputs.expression, ndim=2)
    drugs = _check_tensor("keyed prediction drug features", inputs.drug_features, ndim=2)
    if expression.shape != (len(inputs.row_keys), len(input_axis)) or drugs.shape[0] != len(inputs.row_keys):
        raise FormalFoldError("keyed prediction feature shape does not match the frozen axes")
    if not bool(torch.isfinite(expression).all() and torch.isfinite(drugs).all()):
        raise FormalFoldError("keyed prediction inputs contain non-finite values")


def _validate_keyed_labels(
    labels: tuple[tuple[tuple[str, str], float], ...],
    expected_keys: tuple[tuple[str, str], ...],
) -> None:
    if not isinstance(labels, tuple) or not labels:
        raise FormalFoldError("inner validation labels are missing")
    label_keys = tuple(key for key, _ in labels)
    if len(set(label_keys)) != len(label_keys) or set(label_keys) != set(expected_keys):
        raise FormalFoldError("inner validation labels do not exactly match keyed prediction rows")
    if any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(float(value))
           for _, value in labels):
        raise FormalFoldError("inner validation labels contain missing or non-finite values")


def _require_envelope(name: str, envelope: ReceiptEnvelope | None) -> Mapping[str, Any]:
    if envelope is None:
        raise FormalFoldError(f"required {name} authorization receipt is missing")
    if not isinstance(envelope, ReceiptEnvelope) or not _is_sha256(envelope.sha256):
        raise FormalFoldError(f"required {name} receipt is not hash-bound")
    payload = envelope.payload
    if not isinstance(payload, Mapping):
        raise FormalFoldError(f"{name} receipt payload must be an object")
    if not isinstance(envelope.raw_bytes, bytes) or not envelope.raw_bytes:
        raise FormalFoldError(f"{name} receipt has no original JSON bytes for digest verification")
    if _sha256(envelope.raw_bytes) != envelope.sha256:
        raise FormalFoldError(f"{name} receipt SHA-256 does not match its original JSON bytes")
    try:
        if json.loads(envelope.raw_bytes.decode("utf-8")) != payload:
            raise FormalFoldError(f"{name} receipt payload differs from its original JSON bytes")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormalFoldError(f"{name} receipt original JSON bytes are invalid") from exc
    if (payload.get("experiment_id") != "EXP-045"
            or payload.get("approved_for_training") is not True
            or not isinstance(payload.get("approved_by"), str)
            or not payload.get("approved_by", "").strip()):
        raise FormalFoldError(f"{name} receipt does not authorize EXP-045 training")
    return payload


def _validate_exposure_receipt_chain(
    exposure: Mapping[str, Any], *, outer_fold: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Check metadata-only alias -> mask execution -> checkpoint lineage bindings."""
    alias_required = {
        "outer_fold", "status", "source_entity_alias_mapping_path",
        "source_entity_alias_mapping_sha256", "exclusion_manifest_path",
        "exclusion_manifest_sha256",
    }
    mask_required = {
        "outer_fold", "status", "mask_execution_log_path", "mask_execution_log_sha256",
        "target_fold_mask_path", "target_fold_mask_sha256", "exclusion_manifest_path",
        "exclusion_manifest_sha256", "source_checkpoint_path", "source_checkpoint_sha256",
    }
    lineage_required = {
        "outer_fold", "status", "checkpoint_lineage_content_sha256",
        "source_entity_alias_receipt_sha256", "mask_execution_log_receipt_sha256",
        "source_entity_alias_mapping_path", "source_entity_alias_mapping_sha256",
        "mask_execution_log_path", "mask_execution_log_sha256", "target_fold_mask_path",
        "target_fold_mask_sha256", "exclusion_manifest_path", "exclusion_manifest_sha256",
        "source_checkpoint_path", "source_checkpoint_sha256",
    }
    alias, alias_digest = _validate_nested_receipt(
        exposure, name="source_entity_alias_receipt", required_fields=alias_required,
        outer_fold=outer_fold,
    )
    mask, mask_digest = _validate_nested_receipt(
        exposure, name="mask_execution_log_receipt", required_fields=mask_required,
        outer_fold=outer_fold,
    )
    lineage, lineage_digest = _validate_nested_receipt(
        exposure, name="checkpoint_lineage_receipt", required_fields=lineage_required,
        outer_fold=outer_fold,
    )
    for name, payload in (("entity alias mapping", alias), ("mask execution log", mask),
                          ("checkpoint lineage content", lineage)):
        digest_field = {
            "entity alias mapping": "source_entity_alias_mapping_sha256",
            "mask execution log": "mask_execution_log_sha256",
            "checkpoint lineage content": "checkpoint_lineage_content_sha256",
        }[name]
        if not _is_sha256(payload.get(digest_field)):
            raise FormalFoldError(f"nested {name} content digest is missing or malformed")
    for label, payload, path_key, digest_key in (
        ("entity alias mapping", alias, "source_entity_alias_mapping_path",
         "source_entity_alias_mapping_sha256"),
        ("mask execution log", mask, "mask_execution_log_path", "mask_execution_log_sha256"),
        ("target-fold mask", mask, "target_fold_mask_path", "target_fold_mask_sha256"),
        ("exclusion manifest", alias, "exclusion_manifest_path", "exclusion_manifest_sha256"),
    ):
        if not isinstance(payload[path_key], str) or not payload[path_key].strip():
            raise FormalFoldError(f"nested {label} path is missing")
        if not _is_sha256(payload[digest_key]):
            raise FormalFoldError(f"nested {label} digest is missing or malformed")
    for key in ("exclusion_manifest_path", "exclusion_manifest_sha256"):
        if mask[key] != alias[key] or lineage[key] != alias[key]:
            raise FormalFoldError("alias, mask, and lineage receipts bind different exclusion manifests")
    for key in (
        "source_entity_alias_mapping_path", "source_entity_alias_mapping_sha256",
        "mask_execution_log_path", "mask_execution_log_sha256", "target_fold_mask_path",
        "target_fold_mask_sha256", "source_checkpoint_path", "source_checkpoint_sha256",
    ):
        if key not in mask or key not in lineage or mask[key] != lineage[key]:
            if key.startswith("source_entity_alias"):
                if lineage[key] != alias[key]:
                    raise FormalFoldError(f"lineage receipt does not bind the alias mapping {key}")
            else:
                raise FormalFoldError(f"mask and lineage receipts do not bind the same {key}")
    if not _is_sha256(mask["target_fold_mask_sha256"]):
        raise FormalFoldError("mask execution receipt target-fold mask digest is malformed")
    if not isinstance(mask["source_checkpoint_path"], str) or not mask["source_checkpoint_path"].strip():
        raise FormalFoldError("mask execution receipt checkpoint path is missing")
    if not _is_sha256(mask["source_checkpoint_sha256"]):
        raise FormalFoldError("mask execution receipt checkpoint digest is malformed")
    if (lineage["source_entity_alias_receipt_sha256"] != alias_digest
            or lineage["mask_execution_log_receipt_sha256"] != mask_digest):
        raise FormalFoldError("checkpoint lineage receipt does not bind its alias/mask child receipts")
    for key, child_value in (
        ("target_fold_mask_path", mask["target_fold_mask_path"]),
        ("target_fold_mask_sha256", mask["target_fold_mask_sha256"]),
        ("exclusion_manifest_path", alias["exclusion_manifest_path"]),
        ("exclusion_manifest_sha256", alias["exclusion_manifest_sha256"]),
        ("source_checkpoint_path", mask["source_checkpoint_path"]),
        ("source_checkpoint_sha256", mask["source_checkpoint_sha256"]),
    ):
        if exposure.get(key) != child_value:
            raise FormalFoldError(f"top-level exposure receipt differs from nested chain at {key}")
    if exposure.get("checkpoint_lineage_sha256") != lineage_digest:
        raise FormalFoldError("top-level checkpoint lineage digest differs from nested lineage receipt")
    if exposure.get("source_checkpoint_lineage_status") != "PASS":
        raise FormalFoldError("source checkpoint to fold-mask lineage status is not PASS")
    if exposure.get("target_fold_mask_exposure_status") != "PASS":
        raise FormalFoldError("target-fold/mask exposure status is not PASS")
    _validate_receipt_file_sha256(
        alias["source_entity_alias_mapping_path"], alias["source_entity_alias_mapping_sha256"],
        name="source entity/alias mapping",
    )
    _validate_receipt_file_sha256(
        mask["mask_execution_log_path"], mask["mask_execution_log_sha256"],
        name="mask execution log",
    )
    _validate_receipt_file_sha256(
        mask["target_fold_mask_path"], mask["target_fold_mask_sha256"],
        name="target-fold mask",
    )
    _validate_receipt_file_sha256(
        alias["exclusion_manifest_path"], alias["exclusion_manifest_sha256"],
        name="exclusion manifest",
    )
    _validate_receipt_file_sha256(
        mask["source_checkpoint_path"], mask["source_checkpoint_sha256"],
        name="source checkpoint (raw bytes only)",
    )
    return alias, mask, lineage


def _validate_receipt_headers(
    receipts: AuthorizationReceipts, *, outer_fold: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Validate receipt presence and hard budget/identity fields before input loading."""
    budget = _require_envelope("budget", receipts.budget)
    exposure = _require_envelope("exposure", receipts.exposure)
    inputs = _require_envelope("input", receipts.inputs)
    for role, payload in (("budget", budget), ("exposure", exposure), ("input", inputs)):
        if payload.get("outer_fold") != outer_fold:
            raise FormalFoldError(f"{role} receipt is bound to a different outer fold")
    if budget.get("training_go") is not True:
        raise FormalFoldError("budget receipt does not contain the final training GO")
    if budget.get("exposure_receipt_sha256") != receipts.exposure.sha256:
        raise FormalFoldError("budget receipt does not bind the exact exposure receipt bytes")
    if budget.get("input_receipt_sha256") != receipts.inputs.sha256:
        raise FormalFoldError("budget receipt does not bind the exact input receipt bytes")
    if inputs.get("exposure_receipt_sha256") != receipts.exposure.sha256:
        raise FormalFoldError("input receipt does not bind the exact exposure receipt bytes")
    remaining = budget.get("package_gpu_hours_remaining")
    reserved = budget.get("reserved_gpu_hours")
    wall_limit = budget.get("run_wall_hours_max")
    numeric = (remaining, reserved, wall_limit)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value)) for value in numeric):
        raise FormalFoldError("budget receipt is missing a finite resource allocation")
    if (float(remaining) < 0 or float(remaining) > 48
            or float(reserved) <= 0 or float(reserved) > float(remaining)
            or float(wall_limit) <= 0 or float(wall_limit) > 20):
        raise FormalFoldError("budget receipt has missing, exhausted, or over-limit resources")
    for name, digest in (
        ("source checkpoint", exposure.get("source_checkpoint_sha256")),
        ("safe exclusion manifest", exposure.get("exclusion_manifest_sha256")),
        ("target-fold mask", exposure.get("target_fold_mask_sha256")),
        ("checkpoint exposure lineage", exposure.get("checkpoint_lineage_sha256")),
        ("input contract", inputs.get("input_contract_sha256")),
        ("split", inputs.get("split_sha256")),
        ("input target-fold mask", inputs.get("target_fold_mask_sha256")),
        ("expression axis", inputs.get("input_axis_sha256")),
        ("dependency output axis", inputs.get("dependency_axis_sha256")),
        ("budget candidate/seed schedule", budget.get("candidate_seed_schedule_sha256")),
        ("input candidate/seed schedule", inputs.get("candidate_seed_schedule_sha256")),
    ):
        if not _is_sha256(digest):
            raise FormalFoldError(f"{name} digest is missing or malformed in its receipt")
    _validate_exposure_receipt_chain(exposure, outer_fold=outer_fold)
    if exposure.get("target_fold_mask_sha256") != inputs.get("target_fold_mask_sha256"):
        raise FormalFoldError("exposure and input receipts do not bind the same target-fold mask")
    return budget, exposure, inputs


def validate_authorization(
    plan: FoldPlan,
    receipts: AuthorizationReceipts,
    *,
    device: str,
    check_cuda: bool = True,
) -> tuple[torch.device, str]:
    """Check approved budget, exposure, input, schedule, and exact CUDA device."""
    budget, exposure, inputs = _validate_receipt_headers(receipts, outer_fold=plan.outer_fold)
    if exposure.get("source_checkpoint_sha256") != plan.source_checkpoint_sha256:
        raise FormalFoldError("exposure receipt checkpoint differs from the fold plan")
    if exposure.get("source_checkpoint_path") != plan.source_checkpoint_path:
        raise FormalFoldError("exposure receipt checkpoint path differs from the fold plan")
    if exposure.get("exclusion_manifest_sha256") != plan.exclusion_manifest_sha256:
        raise FormalFoldError("exposure receipt exclusion set differs from the fold plan")
    if (exposure.get("target_fold_mask_sha256") != plan.target_fold_mask_sha256
            or inputs.get("target_fold_mask_sha256") != plan.target_fold_mask_sha256):
        raise FormalFoldError("target-fold mask receipt differs from the frozen fold plan")
    if exposure.get("checkpoint_lineage_sha256") != plan.checkpoint_lineage_sha256:
        raise FormalFoldError("source checkpoint exposure-lineage receipt differs from the fold plan")
    if inputs.get("input_contract_sha256") != plan.input_contract_sha256:
        raise FormalFoldError("input receipt does not bind the provided fold tensors/manifest")
    if inputs.get("split_sha256") != plan.split_sha256:
        raise FormalFoldError("input receipt split digest differs from the fold plan")
    if inputs.get("input_axis_sha256") != axis_sha256(plan.input_axis):
        raise FormalFoldError("input receipt expression axis differs from the frozen gene order")
    if inputs.get("dependency_axis_sha256") != axis_sha256(plan.dependency_axis):
        raise FormalFoldError("input receipt dependency output axis differs from the frozen gene order")
    if inputs.get("dependency_dataset_assets_sha256") != dependency_dataset_assets_sha256(plan):
        raise FormalFoldError("input receipt does not bind the registered dependency source assets")
    if (inputs.get("dependency_strata_definitions_sha256")
            != dependency_strata_definitions_sha256(plan)):
        raise FormalFoldError("input receipt does not bind the frozen cancer/library strata definitions")
    if (inputs.get("dependency_partition_manifests_sha256")
            != dependency_partition_manifests_sha256(plan)):
        raise FormalFoldError("input receipt does not bind every inner/outer source partition manifest")
    schedule_digest = candidate_seed_schedule_sha256(plan)
    if budget.get("candidate_seed_schedule_sha256") != schedule_digest:
        raise FormalFoldError("budget receipt does not approve this exact candidate/seed schedule")
    if inputs.get("candidate_seed_schedule_sha256") != schedule_digest:
        raise FormalFoldError("input receipt does not bind this exact candidate/seed schedule")
    approved_device = budget.get("cuda_device")
    approved_name = budget.get("cuda_device_name")
    if not isinstance(device, str) or re.fullmatch(r"cuda:[0-9]+", device) is None:
        raise FormalFoldError("formal training requires an explicit exact CUDA device such as cuda:0")
    if device != approved_device or not isinstance(approved_name, str) or not approved_name.strip():
        raise FormalFoldError("requested CUDA device/name differs from the budget receipt")
    if check_cuda:
        if not torch.cuda.is_available():
            raise FormalFoldError("CUDA is unavailable; formal training will not fall back to CPU")
        index = int(device.split(":", 1)[1])
        if index >= torch.cuda.device_count():
            raise FormalFoldError(f"approved CUDA device index is unavailable: {device}")
        actual_name = torch.cuda.get_device_name(index)
        if actual_name != approved_name:
            raise FormalFoldError(
                f"CUDA device mismatch: receipt={approved_name!r}, runtime={actual_name!r}"
            )
        torch.cuda.set_device(index)
        if torch.cuda.current_device() != index:
            raise FormalFoldError("runtime did not select the approved CUDA device")
    else:
        actual_name = approved_name
    return torch.device(device), actual_name


def _patient_groups(patient_ids: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
    by_patient: dict[str, list[int]] = {}
    for index, patient in enumerate(patient_ids):
        by_patient.setdefault(patient, []).append(index)
    return tuple(tuple(indices) for indices in by_patient.values())


def _active_weights(arm: Exp045Arm, stage: Exp045Stage, candidate: Candidate) -> tuple[float, float, float]:
    active = _ACTIVE_DOMAIN[(arm, stage)]
    depmap = candidate.lambda_depmap if active and active[0] == "depmap" else 0.0
    nature = candidate.lambda_nature if active and active[0] == "nature" else 0.0
    return candidate.lambda_reconstruction, depmap, nature


def _move_dependency(batch: AuxiliaryBatch, device: torch.device) -> Exp045DependencyTargets | None:
    if batch.dependency_target is None:
        return None
    assert batch.dependency_mask is not None
    assert batch.dependency_train_mean is not None
    assert batch.dependency_train_overall_std is not None
    assert batch.label_assignment is not None
    return Exp045DependencyTargets(
        target=batch.dependency_target.to(device),
        valid_mask=batch.dependency_mask.to(device),
        train_mean=float(batch.dependency_train_mean),
        train_overall_std=float(batch.dependency_train_overall_std),
        label_assignment=batch.label_assignment,
    )


def _new_audit(arm: Exp045Arm) -> ArmExecutionAudit:
    return ArmExecutionAudit(
        arm=arm,
        auxiliary={stage: Exp045FoldStepAudit(arm=arm, stage=stage) for stage in STAGES},
    )


def _execute_partition_schedule(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    arm: Exp045Arm,
    schedule: ArmPartitionSchedule,
    plan: FoldPlan,
    candidate: Candidate,
    epochs: int,
    device: torch.device,
    audit: ArmExecutionAudit | None = None,
    budget_check: Callable[[], None] | None = None,
) -> ArmExecutionAudit:
    """Execute a validated schedule; formal callers reach this after CUDA/receipt gates.

    Kept device-agnostic so tests can exercise the exact optimizer choreography
    with tiny synthetic CPU tensors. The CLI never exposes a CPU training mode.
    """
    if not isinstance(model, nn.Module) or not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("model and optimizer must be concrete PyTorch training objects")
    if epochs < 1 or arm not in ARMS:
        raise ValueError("invalid arm or epoch count")
    result = audit or _new_audit(arm)
    if result.arm != arm:
        raise ValueError("audit arm does not match the running arm")
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError(f"{arm} has no trainable parameters")
    try:
        for _epoch in range(epochs):
            for event in schedule.events:
                if budget_check is not None:
                    budget_check()
                if event.kind == "dss":
                    batch = schedule.dss_batches[event.batch_id]
                    expression = batch.expression.to(device)
                    drug = batch.drug_features.to(device)
                    target = batch.target.to(device)
                    if not callable(getattr(model, "predict_drug_sensitivity", None)):
                        raise TypeError("model must expose predict_drug_sensitivity(expression, drug)")
                    result.dss_scheduled += 1
                    result.dss_forward_calls += 1
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
                    ):
                        prediction = model.predict_drug_sensitivity(expression, drug).float().reshape(-1)
                        objective = exp045_multitask_objective(
                            prediction, target, _patient_groups(batch.patient_ids),
                            dss_ranking_weight=candidate.dss_ranking_weight,
                            lambda_depmap=0.0, depmap_batch=None,
                            lambda_nature=0.0, nature_batch=None,
                        )
                        loss = objective.total_loss
                    if loss is None or not bool(torch.isfinite(loss)):
                        raise RuntimeError(f"{arm}/{event.batch_id}: DSS loss is undefined")
                    result.dss_backward_calls += 1
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters, 1.0)
                    optimizer.step()
                    result.dss_optimizer_steps += 1
                    if budget_check is not None:
                        budget_check()
                    continue

                assert event.stage is not None
                if arm == "D-A":
                    raise FormalFoldError("D-A cannot execute an auxiliary update")
                batch = schedule.auxiliary_batches[event.stage][event.batch_id]
                if not callable(getattr(model, "predict_auxiliary_heads", None)):
                    raise TypeError("auxiliary model must expose predict_auxiliary_heads(expression)")
                lambda_r, lambda_c, lambda_n = _active_weights(arm, event.stage, candidate)
                exp045_auxiliary_batch_step(
                    model, optimizer, batch.expression.to(device), arm=arm, stage=event.stage,
                    lambda_reconstruction=lambda_r,
                    reconstruction_targets=Exp045ExpressionTargets(
                        target=batch.reconstruction_target.to(device),
                        valid_mask=batch.reconstruction_mask.to(device),
                    ),
                    lambda_depmap=lambda_c,
                    depmap_targets=_move_dependency(batch, device) if lambda_c > 0 else None,
                    lambda_nature=lambda_n,
                    nature_targets=_move_dependency(batch, device) if lambda_n > 0 else None,
                    audit=result.auxiliary[event.stage],
                )
                if budget_check is not None:
                    budget_check()
        _assert_epoch_opportunities(arm, schedule, epochs, result)
    except Exception:
        result.failures += 1
        raise
    return result


def _assert_epoch_opportunities(
    arm: Exp045Arm,
    schedule: ArmPartitionSchedule,
    epochs: int,
    audit: ArmExecutionAudit,
) -> None:
    expected_dss = sum(event.kind == "dss" for event in schedule.events) * epochs
    if not (audit.dss_scheduled == audit.dss_forward_calls == audit.dss_backward_calls
            == audit.dss_optimizer_steps == expected_dss):
        raise FormalFoldError(f"{arm} DSS update opportunity was skipped or duplicated")
    for stage in STAGES:
        expected_aux = sum(
            event.kind == "auxiliary" and event.stage == stage for event in schedule.events
        ) * epochs
        stage_audit = audit.auxiliary[stage]
        if arm == "D-A":
            expected_aux = 0
        if not (stage_audit.scheduled_batches == stage_audit.forward_calls
                == stage_audit.backward_calls == stage_audit.optimizer_steps == expected_aux):
            raise FormalFoldError(f"{arm}/{stage} auxiliary update opportunity was skipped or duplicated")
        if stage_audit.failures:
            raise FormalFoldError(f"{arm}/{stage} recorded a failed auxiliary update")


def _model_backbone_digest(model: nn.Module) -> str:
    backbone = getattr(model, "backbone", None)
    if not isinstance(backbone, nn.Module):
        raise FormalFoldError("D1 model must expose the reusable PharmaFormer backbone")
    digest = hashlib.sha256()
    for name, tensor in sorted(backbone.state_dict().items()):
        if not isinstance(tensor, torch.Tensor):
            raise FormalFoldError("backbone state contains a non-tensor value")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _model_state_digest(model: nn.Module) -> str:
    """Hash every registered parameter and buffer to audit arm initialization."""
    digest = hashlib.sha256()
    state = model.state_dict()
    if not state:
        raise FormalFoldError("model state is empty")
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise FormalFoldError("model state contains a non-tensor value")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _verify_shared_initialization(
    model_factory: ModelFactory,
    candidates: Sequence[Candidate],
    seeds: Sequence[int],
    budget_check: Callable[[], None] | None = None,
) -> None:
    """Confirm shared backbone and auxiliary-arm model states per trial."""
    for candidate in candidates:
        for seed in seeds:
            expected_backbone: str | None = None
            expected_auxiliary_state: str | None = None
            for arm in ARMS:
                if budget_check is not None:
                    budget_check()
                torch.manual_seed(seed)
                model = model_factory(arm, candidate, seed)
                if not isinstance(model, nn.Module):
                    raise TypeError("model factory must return a torch module")
                if not callable(getattr(model, "predict_drug_sensitivity", None)):
                    raise TypeError("every arm must expose the frozen DSS prediction path")
                if arm != "D-A" and not callable(getattr(model, "predict_auxiliary_heads", None)):
                    raise TypeError("auxiliary arms must expose the registered output heads")
                backbone_digest = _model_backbone_digest(model)
                if expected_backbone is None:
                    expected_backbone = backbone_digest
                elif backbone_digest != expected_backbone:
                    raise FormalFoldError(
                        f"{arm} initial PharmaFormer backbone differs for {candidate.candidate_id}/seed{seed}"
                    )
                if arm != "D-A":
                    state_digest = _model_state_digest(model)
                    if expected_auxiliary_state is None:
                        expected_auxiliary_state = state_digest
                    elif state_digest != expected_auxiliary_state:
                        raise FormalFoldError(
                            f"{arm} initial auxiliary model state differs for "
                            f"{candidate.candidate_id}/seed{seed}"
                        )
                del model
                if budget_check is not None:
                    budget_check()


@torch.no_grad()
def predict_keyed(
    model: nn.Module,
    inputs: KeyedPredictionInput,
    *,
    device: torch.device,
    response_mean: float,
    response_scale: float,
    batch_size: int = 128,
) -> tuple[tuple[tuple[str, str], float], ...]:
    """Predict by keyed patient/drug rows and invert the train-only DSS scale."""
    if not callable(getattr(model, "predict_drug_sensitivity", None)):
        raise TypeError("model must expose predict_drug_sensitivity(expression, drug)")
    if batch_size < 1 or not math.isfinite(response_mean) or not math.isfinite(response_scale) or response_scale <= 0:
        raise ValueError("invalid prediction batch size or train-fitted DSS transform")
    model.eval()
    values: list[float] = []
    for begin in range(0, len(inputs.row_keys), batch_size):
        end = min(len(inputs.row_keys), begin + batch_size)
        expression = inputs.expression[begin:end].to(device)
        drugs = inputs.drug_features[begin:end].to(device)
        prediction = model.predict_drug_sensitivity(expression, drugs).float().reshape(-1)
        raw_scale = prediction * response_scale + response_mean
        if raw_scale.shape != (end - begin,) or not bool(torch.isfinite(raw_scale).all()):
            raise ValueError("model returned missing, misaligned, or non-finite keyed predictions")
        values.extend(float(value) for value in raw_scale.cpu().tolist())
    return tuple(zip(inputs.row_keys, values))


def evaluate_keyed_predictions(
    predictions: Sequence[tuple[tuple[str, str], float]],
    observed_labels: Mapping[tuple[str, str], float],
) -> KeyedMetricSummary:
    """Join predictions and labels by patient/drug key and compute patient-macro Spearman."""
    prediction_keys = tuple(key for key, _ in predictions)
    if len(set(prediction_keys)) != len(prediction_keys):
        raise FormalFoldError("keyed predictions contain duplicate patient/drug rows")
    if set(prediction_keys) != set(observed_labels):
        missing = len(set(prediction_keys) - set(observed_labels))
        extra = len(set(observed_labels) - set(prediction_keys))
        raise FormalFoldError(f"prediction/label key mismatch (missing={missing}, extra={extra})")
    if any(not math.isfinite(float(value)) for _, value in predictions):
        raise FormalFoldError("keyed predictions contain non-finite values")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value)) for value in observed_labels.values()):
        raise FormalFoldError("observed keyed labels contain missing or non-finite values")
    predicted_by_patient: dict[str, list[float]] = defaultdict(list)
    observed_by_patient: dict[str, list[float]] = defaultdict(list)
    for (patient, drug), prediction in predictions:
        predicted_by_patient[patient].append(float(prediction))
        observed_by_patient[patient].append(float(observed_labels[(patient, drug)]))
    rows: list[PatientMetric] = []
    for patient in sorted(predicted_by_patient):
        predicted = np.asarray(predicted_by_patient[patient], dtype=np.float64)
        observed = np.asarray(observed_by_patient[patient], dtype=np.float64)
        if len(predicted) < 3:
            rows.append(PatientMetric(patient, None, len(predicted), "FEWER_THAN_THREE_DRUGS"))
        elif np.ptp(observed) == 0:
            rows.append(PatientMetric(patient, None, len(predicted), "CONSTANT_OBSERVED"))
        elif np.ptp(predicted) == 0:
            rows.append(PatientMetric(patient, None, len(predicted), "CONSTANT_PREDICTED"))
        else:
            rho = float(spearmanr(observed, predicted).statistic)
            rows.append(PatientMetric(
                patient, rho if math.isfinite(rho) else None, len(predicted),
                None if math.isfinite(rho) else "UNDEFINED_CORRELATION",
            ))
    valid = [row.spearman for row in rows if row.spearman is not None]
    macro = float(np.mean(valid)) if valid else None
    return KeyedMetricSummary(
        metric="patient_macro_within_patient_spearman",
        patient_macro_spearman=macro,
        valid_patients=len(valid),
        total_patients=len(rows),
        patients=tuple(rows),
    )


def _optimizer(model: nn.Module, candidate: Candidate) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise FormalFoldError("model has no trainable parameters")
    return torch.optim.AdamW(parameters, lr=candidate.learning_rate, weight_decay=candidate.weight_decay)


def _model_for_arm(model_factory: ModelFactory, arm: Exp045Arm, candidate: Candidate,
                   seed: int, device: torch.device) -> nn.Module:
    torch.manual_seed(seed)
    model = model_factory(arm, candidate, seed)
    if not isinstance(model, nn.Module):
        raise TypeError("model factory must return a torch module")
    model = model.to(device)
    if arm == "D-A":
        # The direct baseline retains the original DSS branch and does not train
        # the newly attached auxiliary output heads.
        for name in ("expression_reconstruction_head", "depmap_chronos_head", "nature_crispr_lfc_head"):
            head = getattr(model, name, None)
            if isinstance(head, nn.Module):
                for parameter in head.parameters():
                    parameter.requires_grad_(False)
    return model


def run_formal_fold(
    plan: FoldPlan,
    receipts: AuthorizationReceipts,
    *,
    device: str,
    model_factory: ModelFactory,
    prediction_batch_size: int = 128,
    deadline_monotonic: float | None = None,
) -> FoldRunResult:
    """Run inner selection and one frozen outer fold after every hard gate passes."""
    validate_fold_plan(plan)
    actual_device, device_name = validate_authorization(plan, receipts, device=device)
    if not callable(model_factory):
        raise TypeError("an approved data/model adapter must supply a model factory")
    if prediction_batch_size < 1:
        raise ValueError("prediction_batch_size must be positive")
    started_at = datetime.now(timezone.utc).isoformat()
    start_clock = time.perf_counter()
    budget = receipts.budget.payload
    allowed_hours = min(
        float(budget["run_wall_hours_max"]),
        float(budget["reserved_gpu_hours"]),
        float(budget["package_gpu_hours_remaining"]),
    )
    deadline = deadline_monotonic if deadline_monotonic is not None else start_clock + allowed_hours * 3600.0

    def check_budget() -> None:
        if time.perf_counter() >= deadline:
            raise FormalFoldError(
                "approved wall/GPU-hour reservation reached; stopping before another optimizer step"
            )

    check_budget()

    # This parity check runs on CPU before any optimizer exists or CUDA weights
    # are updated. It verifies the shared source backbone across every arm/trial.
    initialization_seeds = tuple(dict.fromkeys((*plan.selection_seeds, *plan.final_seeds)))
    _verify_shared_initialization(
        model_factory, plan.candidates, initialization_seeds, budget_check=check_budget,
    )

    inner_scores: dict[tuple[str, str], list[float]] = {
        (arm, candidate.candidate_id): [] for arm in ARMS for candidate in plan.candidates
    }
    inner_epochs: dict[tuple[str, str], list[int]] = {
        (arm, candidate.candidate_id): [] for arm in ARMS for candidate in plan.candidates
    }
    inner_trial_rows: list[dict[str, Any]] = []
    all_audits: list[dict[str, Any]] = []

    for arm in ARMS:
        for candidate in plan.candidates:
            for inner in plan.inner_folds:
                schedule = inner.training.schedules[arm]
                for seed in plan.selection_seeds:
                    check_budget()
                    model = _model_for_arm(model_factory, arm, candidate, seed, actual_device)
                    optimizer = _optimizer(model, candidate)
                    best_score = -math.inf
                    best_epoch = 0
                    stale = 0
                    for epoch in range(1, candidate.max_epochs + 1):
                        audit = _execute_partition_schedule(
                            model, optimizer, arm=arm, schedule=schedule, plan=plan,
                            candidate=candidate, epochs=1, device=actual_device,
                            budget_check=check_budget,
                        )
                        _assert_epoch_opportunities(arm, schedule, 1, audit)
                        inner_predictions = predict_keyed(
                            model, inner.validation_inputs, device=actual_device,
                            response_mean=inner.training.response_mean,
                            response_scale=inner.training.response_scale,
                            batch_size=prediction_batch_size,
                        )
                        check_budget()
                        label_map = dict(inner.validation_labels)
                        metric = evaluate_keyed_predictions(inner_predictions, label_map)
                        score = metric.patient_macro_spearman
                        if score is None or not math.isfinite(score):
                            raise FormalFoldError(
                                f"{arm}/{candidate.candidate_id}/{inner.fold_id}/seed{seed}: "
                                "inner selection metric is undefined"
                            )
                        if score > best_score + 1e-6:
                            best_score, best_epoch, stale = score, epoch, 0
                        else:
                            stale += 1
                            if stale >= candidate.patience:
                                break
                    key = (arm, candidate.candidate_id)
                    inner_scores[key].append(best_score)
                    inner_epochs[key].append(best_epoch)
                    inner_trial_rows.append({
                        "arm": arm, "candidate_id": candidate.candidate_id,
                        "inner_fold": inner.fold_id, "seed": seed,
                        "best_epoch": best_epoch, "patient_macro_within_patient_spearman": best_score,
                        "dss_optimizer_steps_per_epoch": sum(event.kind == "dss" for event in schedule.events),
                        "auxiliary_optimizer_steps_per_epoch": {
                            stage: sum(event.kind == "auxiliary" and event.stage == stage
                                       for event in schedule.events)
                            for stage in STAGES
                        },
                    })
                    del optimizer, model
                    if actual_device.type == "cuda":
                        torch.cuda.empty_cache()

    selections: list[CandidateSelection] = []
    selected_candidates: dict[str, Candidate] = {}
    for arm in ARMS:
        ranked: list[tuple[float, str, Candidate]] = []
        for candidate in plan.candidates:
            values = inner_scores[(arm, candidate.candidate_id)]
            if len(values) != len(plan.inner_folds) * len(plan.selection_seeds):
                raise FormalFoldError(f"{arm}/{candidate.candidate_id}: incomplete inner candidate schedule")
            ranked.append((float(np.mean(values)), candidate.candidate_id, candidate))
        # Fixed tie rule: lexicographically smaller registered ID wins.
        score, candidate_id, selected = min(ranked, key=lambda item: (-item[0], item[1]))
        selected_candidates[arm] = selected
        epochs = inner_epochs[(arm, candidate_id)]
        refit_epochs = max(1, int(round(float(np.median(epochs)))))
        selections.append(CandidateSelection(
            arm=arm, candidate_id=candidate_id,
            mean_inner_patient_macro_spearman=score, refit_epochs=refit_epochs,
            inner_scores=tuple(inner_scores[(arm, candidate_id)]),
        ))

    outer_predictions: list[KeyedPrediction] = []
    outer_audits: list[dict[str, Any]] = []
    for arm in ARMS:
        candidate = selected_candidates[arm]
        selection = next(item for item in selections if item.arm == arm)
        for seed in plan.final_seeds:
            check_budget()
            model = _model_for_arm(model_factory, arm, candidate, seed, actual_device)
            optimizer = _optimizer(model, candidate)
            audit = _execute_partition_schedule(
                model, optimizer, arm=arm,
                schedule=plan.outer_training.schedules[arm], plan=plan,
                candidate=candidate, epochs=selection.refit_epochs, device=actual_device,
                budget_check=check_budget,
            )
            _assert_epoch_opportunities(
                arm, plan.outer_training.schedules[arm], selection.refit_epochs, audit,
            )
            predictions = predict_keyed(
                model, plan.outer_test_inputs, device=actual_device,
                response_mean=plan.outer_training.response_mean,
                response_scale=plan.outer_training.response_scale,
                batch_size=prediction_batch_size,
            )
            check_budget()
            for (patient, drug), prediction in predictions:
                outer_predictions.append(KeyedPrediction(
                    fold=plan.outer_fold, arm=arm, candidate_id=candidate.candidate_id,
                    seed=seed, patient_id=patient, drug_id=drug, prediction=prediction,
                ))
            outer_audits.append(_audit_to_dict(audit))
            del optimizer, model
            if actual_device.type == "cuda":
                torch.cuda.empty_cache()

    ended_at = datetime.now(timezone.utc).isoformat()
    elapsed = time.perf_counter() - start_clock
    receipt_hashes = {
        "budget": receipts.budget.sha256 if receipts.budget else "",
        "exposure": receipts.exposure.sha256 if receipts.exposure else "",
        "input": receipts.inputs.sha256 if receipts.inputs else "",
    }
    return FoldRunResult(
        fold=plan.outer_fold,
        status="FOLD_COMPLETE_PENDING_OUTER_LABEL_EVALUATION_AND_INDEPENDENT_REVIEW",
        selections=tuple(selections), predictions=tuple(outer_predictions),
        audits=tuple(outer_audits), receipt_hashes=receipt_hashes,
        input_contract_sha256=plan.input_contract_sha256,
        split_sha256=plan.split_sha256,
        source_checkpoint_sha256=plan.source_checkpoint_sha256,
        source_checkpoint_path=plan.source_checkpoint_path,
        exclusion_manifest_sha256=plan.exclusion_manifest_sha256,
        target_fold_mask_sha256=plan.target_fold_mask_sha256,
        checkpoint_lineage_sha256=plan.checkpoint_lineage_sha256,
        started_utc=started_at, ended_utc=ended_at, elapsed_wall_seconds=elapsed,
        cuda_device=str(actual_device), cuda_device_name=device_name,
    )


def _audit_to_dict(audit: ArmExecutionAudit) -> dict[str, Any]:
    return {
        "arm": audit.arm,
        "dss": {
            "scheduled": audit.dss_scheduled,
            "forward_calls": audit.dss_forward_calls,
            "backward_calls": audit.dss_backward_calls,
            "optimizer_steps": audit.dss_optimizer_steps,
        },
        "auxiliary": {
            stage: {
                "scheduled_batches": stage_audit.scheduled_batches,
                "forward_calls": stage_audit.forward_calls,
                "backward_calls": stage_audit.backward_calls,
                "optimizer_steps": stage_audit.optimizer_steps,
                "failures": stage_audit.failures,
                "reconstruction_dimensions": stage_audit.reconstruction_dimensions,
                "reconstruction_valid_denominator": stage_audit.reconstruction_valid_denominator,
                "depmap_valid_denominators": stage_audit.depmap_valid_denominators,
                "nature_valid_denominators": stage_audit.nature_valid_denominators,
                "reconstruction_losses": stage_audit.reconstruction_losses,
                "depmap_losses": stage_audit.depmap_losses,
                "nature_losses": stage_audit.nature_losses,
                "total_losses": stage_audit.total_losses,
            }
            for stage, stage_audit in audit.auxiliary.items()
        },
        "failures": audit.failures,
    }


def require_wsl_conda_runtime() -> None:
    """Formal CLI guard: all real Python work must use the approved WSL Conda env."""
    if sys.platform != "linux" or "microsoft" not in Path("/proc/version").read_text(
        encoding="utf-8", errors="ignore"
    ).lower():
        raise FormalFoldError("formal EXP-045 training is supported only in WSL2")
    if os.environ.get("CONDA_DEFAULT_ENV") != "drugscreening-gpu":
        raise FormalFoldError("formal EXP-045 training requires Conda environment drugscreening-gpu")


def _load_adapter(specification: str) -> Any:
    if ":" not in specification:
        raise FormalFoldError("adapter must use module:function syntax")
    module_name, function_name = specification.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name, None)
    if not callable(factory):
        raise FormalFoldError("data adapter does not expose the requested callable")
    return factory


def _plain(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _plain(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_plain(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_provenance(adapter_factory: Callable[..., Any]) -> dict[str, Any]:
    files = {
        "driver": Path(__file__).resolve(),
        "fold_step": Path(__file__).resolve().parents[2] / "src/drug_screen/models/exp045_fold_step.py",
        "objective": Path(__file__).resolve().parents[2] / "src/drug_screen/models/exp045_multitask_objective.py",
        "dependency_heads": Path(__file__).resolve().parents[2] / "src/drug_screen/models/exp045_dependency_heads.py",
    }
    adapter_source = getattr(adapter_factory, "__code__", None)
    adapter_file = getattr(adapter_source, "co_filename", None)
    if adapter_file and Path(adapter_file).is_file():
        files["data_adapter"] = Path(adapter_file).resolve()
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        changed = subprocess.run(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[2],
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_sha, changed = "unavailable", []
    return {
        "git_head": git_sha,
        "working_tree_dirty": bool(changed),
        "changed_path_count": len(changed),
        "code_file_sha256": {name: _file_sha256(path) for name, path in files.items() if path.is_file()},
    }


def _write_result(output_dir: Path, result: FoldRunResult, provenance: Mapping[str, Any]) -> dict[str, str]:
    prediction_path = output_dir / "keyed_predictions.jsonl"
    with prediction_path.open("x", encoding="utf-8", newline="\n") as handle:
        for prediction in result.predictions:
            handle.write(json.dumps(asdict(prediction), ensure_ascii=False, sort_keys=True) + "\n")
    file_hashes = {prediction_path.name: _file_sha256(prediction_path)}
    summary = {
        **asdict(result),
        "predictions": "keyed_predictions.jsonl",
        "prediction_rows": len(result.predictions),
        "code_provenance": provenance,
        "files_sha256": file_hashes,
        "scientific_status": "predictions_materialized_outer_labels_not_read_or_evaluated",
    }
    _atomic_write_json(output_dir / "fold_summary.json", summary)
    file_hashes["fold_summary.json"] = _file_sha256(output_dir / "fold_summary.json")
    return file_hashes


def run(args: argparse.Namespace) -> dict[str, Any]:
    """CLI dispatcher. Dry-run is the default and never opens the scientific data."""
    if not args.train:
        return {
            "experiment_id": "EXP-045", "mode": "DRY_RUN",
            "status": "NO_DATA_READ_NO_MODEL_BUILT_NO_TRAINING",
            "required_before_training": [
                "approved budget receipt with final training GO and reserved GPU-hours",
                "recomputed fold-bound alias-map/mask-log/checkpoint-lineage receipt chain",
                "fold-specific input/split/axis and candidate-seed receipt",
                "explicit exact CUDA device and matching runtime hardware",
                "WSL2 Conda drugscreening-gpu adapter returning a validated FoldPlan",
            ],
        }
    require_wsl_conda_runtime()
    if not all((args.budget_receipt, args.exposure_receipt, args.input_receipt, args.adapter,
                args.output_dir, args.device)):
        raise FormalFoldError(
            "--train requires budget, exposure, and input receipts, an adapter, output directory, and exact CUDA device"
        )
    receipts = AuthorizationReceipts(
        load_receipt(args.budget_receipt), load_receipt(args.exposure_receipt),
        load_receipt(args.input_receipt),
    )
    # The adapter is imported only after the receipt and CUDA gates. Its contract
    # is prepare_fold(fold_id, device, receipts) -> (FoldPlan, model_factory).
    budget, exposure, _inputs = _validate_receipt_headers(receipts, outer_fold=args.fold)
    alias_receipt, mask_receipt, lineage_receipt = _validate_exposure_receipt_chain(
        exposure, outer_fold=args.fold,
    )
    device_obj = validate_authorized_device(
        args.device, budget.get("cuda_device"), budget.get("cuda_device_name")
    )
    if args.output_dir.exists():
        raise FileExistsError(f"formal fold output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run_started = datetime.now(timezone.utc).isoformat()
    run_clock = time.perf_counter()
    receipt_hashes = {
        "budget": receipts.budget.sha256,
        "exposure": receipts.exposure.sha256,
        "input": receipts.inputs.sha256,
    }
    base_receipt = {
        "experiment_id": "EXP-045", "fold": args.fold,
        "mode": "FORMAL_TRAINING", "status": "AUTHORIZED_PRECHECK_DATA_NOT_YET_READ",
        "started_utc": run_started, "requested_cuda_device": str(device_obj),
        "cuda_device_name": torch.cuda.get_device_name(device_obj.index or 0),
        "receipt_hashes": receipt_hashes, "command_line": sys.argv,
        "source_checkpoint_sha256": receipts.exposure.payload.get("source_checkpoint_sha256"),
        "source_checkpoint_path": receipts.exposure.payload.get("source_checkpoint_path"),
        "source_entity_alias_mapping_path": alias_receipt["source_entity_alias_mapping_path"],
        "source_entity_alias_mapping_sha256": alias_receipt["source_entity_alias_mapping_sha256"],
        "mask_execution_log_path": mask_receipt["mask_execution_log_path"],
        "mask_execution_log_sha256": mask_receipt["mask_execution_log_sha256"],
        "target_fold_mask_path": mask_receipt["target_fold_mask_path"],
        "exclusion_manifest_sha256": receipts.exposure.payload.get("exclusion_manifest_sha256"),
        "exclusion_manifest_path": alias_receipt["exclusion_manifest_path"],
        "target_fold_mask_sha256": receipts.exposure.payload.get("target_fold_mask_sha256"),
        "checkpoint_lineage_sha256": receipts.exposure.payload.get("checkpoint_lineage_sha256"),
        "exposure_chain_gate": "RECOMPUTED_METADATA_CHAIN_PASS_SOURCE_EXPOSURE_NOT_INDEPENDENTLY_VERIFIED",
        "source_entity_alias_receipt_sha256": exposure["source_entity_alias_receipt"]["sha256"],
        "mask_execution_log_receipt_sha256": exposure["mask_execution_log_receipt"]["sha256"],
        "checkpoint_lineage_receipt_sha256": exposure["checkpoint_lineage_receipt"]["sha256"],
        "checkpoint_lineage_content_sha256": lineage_receipt["checkpoint_lineage_content_sha256"],
    }
    _atomic_write_json(args.output_dir / "run_receipt.json", base_receipt)
    try:
        adapter_factory = _load_adapter(args.adapter)
        plan, model_factory = adapter_factory(args.fold, device_obj, receipts)
        validate_fold_plan(plan)
        result = run_formal_fold(
            plan, receipts, device=args.device, model_factory=model_factory,
            prediction_batch_size=args.prediction_batch_size,
            deadline_monotonic=run_clock + min(
                float(budget["run_wall_hours_max"]),
                float(budget["reserved_gpu_hours"]),
                float(budget["package_gpu_hours_remaining"]),
            ) * 3600.0,
        )
        provenance = _code_provenance(adapter_factory)
        file_hashes = _write_result(args.output_dir, result, provenance)
        final_receipt = {
            **base_receipt,
            "status": result.status,
            "ended_utc": result.ended_utc,
            "elapsed_wall_seconds": time.perf_counter() - run_clock,
            "gpu_hours": (time.perf_counter() - run_clock) / 3600.0,
            "input_contract_sha256": result.input_contract_sha256,
            "split_sha256": result.split_sha256,
            "source_checkpoint_sha256": result.source_checkpoint_sha256,
            "source_checkpoint_path": result.source_checkpoint_path,
            "exclusion_manifest_sha256": result.exclusion_manifest_sha256,
            "target_fold_mask_sha256": result.target_fold_mask_sha256,
            "checkpoint_lineage_sha256": result.checkpoint_lineage_sha256,
            "prediction_rows": len(result.predictions),
            "files_sha256": file_hashes,
            "code_provenance": provenance,
            "scientific_status": "outer_predictions_only_pending_frozen_label_evaluation_and_independent_review",
        }
        _atomic_write_json(args.output_dir / "run_receipt.json", final_receipt)
    except Exception as exc:
        failed_receipt = {
            **base_receipt,
            "status": "FAILED_CLOSED",
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_wall_seconds": time.perf_counter() - run_clock,
            "gpu_hours": (time.perf_counter() - run_clock) / 3600.0,
            "error_type": type(exc).__name__, "error": str(exc),
            "scientific_status": "failed_or_incomplete_run_no_prediction_claim",
        }
        _atomic_write_json(args.output_dir / "run_receipt.json", failed_receipt)
        raise
    return {
        "experiment_id": "EXP-045", "fold": args.fold, "status": result.status,
        "output_dir": str(args.output_dir), "prediction_rows": len(result.predictions),
        "files_sha256": file_hashes,
        "gpu_hours": float(json.loads((args.output_dir / "run_receipt.json").read_text())['gpu_hours']),
        "scientific_status": "outer_predictions_only_pending_frozen_label_evaluation_and_independent_review",
    }


def validate_authorized_device(requested: str, approved: object, approved_name: object) -> torch.device:
    """CLI's pre-adapter CUDA gate, so data loading cannot precede device validation."""
    if re.fullmatch(r"cuda:[0-9]+", requested or "") is None:
        raise FormalFoldError("formal training requires an explicit exact CUDA device")
    if requested != approved or not isinstance(approved_name, str) or not approved_name.strip():
        raise FormalFoldError("requested CUDA device/name differs from approved budget receipt")
    if not torch.cuda.is_available():
        raise FormalFoldError("CUDA is unavailable; the driver will not fall back to CPU")
    index = int(requested.split(":", 1)[1])
    if index >= torch.cuda.device_count():
        raise FormalFoldError(f"approved CUDA device index is unavailable: {requested}")
    if torch.cuda.get_device_name(index) != approved_name:
        raise FormalFoldError("runtime CUDA hardware does not match the approved exact device")
    torch.cuda.set_device(index)
    if torch.cuda.current_device() != index:
        raise FormalFoldError("runtime did not select the approved CUDA device")
    return torch.device(requested)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", dest="train", action="store_false", default=False,
                        help="default metadata-only path; opens no scientific data")
    parser.add_argument("--train", dest="train", action="store_true",
                        help="request formal training after receipt, identity, and CUDA gates")
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--budget-receipt", type=Path)
    parser.add_argument("--exposure-receipt", type=Path)
    parser.add_argument("--input-receipt", type=Path)
    parser.add_argument("--adapter", help="approved data/model adapter as module:function")
    parser.add_argument("--device", help="exact approved CUDA device, for example cuda:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prediction-batch-size", type=int, default=128)
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception as exc:
        print(json.dumps({"status": "FAILED_CLOSED", "error_type": type(exc).__name__,
                          "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
