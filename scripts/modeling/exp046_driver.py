"""Fail-closed, caller-parameterized orchestration for registered EXP-046.

The split validator remains an in-memory check of five outer and three inner
patient partitions. The orchestration accepts all pending search, selection,
initialization, and gate decisions as explicit inputs. It reads only the
registered EXP-046 YAML configuration, has no CLI, and keeps outer-test labels
outside its interface.
"""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable, Mapping

import numpy as np
import torch
import yaml

from scripts.modeling.exp046_pathway_fold import (
    BranchInitialization,
    CandidateTraining,
    InnerFoldReceipt,
    LabeledPairBatch,
    OuterFoldReceipt,
    OuterKeyedPredictions,
    PairInputBatch,
    SourceFoldInitialization,
    fit_inner_candidate,
    fit_outer_fold,
)
from drug_screen.models.exp046_five_arm import (
    AlignedPathwayMap,
    Exp046Arm,
    validate_aligned_pathway_map,
)


class ContractError(ValueError):
    """Raised when an EXP-046 in-memory fold-plan contract is invalid."""


@dataclass(frozen=True)
class InnerFoldSlice:
    """Pre-sliced labeled inputs for one inner train/validation split."""

    fold_key: str
    train: LabeledPairBatch = field(repr=False)
    validation: LabeledPairBatch = field(repr=False)


@dataclass(frozen=True)
class OuterFoldSlice:
    """Outer training labels and label-free outer prediction inputs.

    There is intentionally no outer-test label field. ``outer_test`` must be
    exactly ``PairInputBatch`` so a label-bearing subclass cannot cross this
    boundary.
    """

    fold_key: str
    outer_train: LabeledPairBatch = field(repr=False)
    outer_test: PairInputBatch = field(repr=False)
    inner_folds: tuple[InnerFoldSlice, ...] = field(repr=False)


@dataclass(frozen=True)
class FoldPlan:
    """All folds plus the explicit target-patient universe for this run."""

    cohort_patient_ids: frozenset[str] = field(repr=False)
    outer_folds: tuple[OuterFoldSlice, ...] = field(repr=False)


def _validate_key(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ContractError(f"{name} must be a non-empty trimmed string")
    return value


def _patient_ids(batch: object, expected_type: type, name: str) -> frozenset[str]:
    if type(batch) is not expected_type:
        raise ContractError(f"{name} must be exactly {expected_type.__name__}")
    values = getattr(batch, "patient_ids", None)
    if not isinstance(values, tuple) or not values:
        raise ContractError(f"{name}.patient_ids must be a non-empty tuple")
    if any(not isinstance(value, str) or not value or value != value.strip()
           for value in values):
        raise ContractError(f"{name}.patient_ids must contain trimmed identities")
    # Patient IDs repeat across drug rows; the set is the split unit.
    return frozenset(values)


def validate_fold_plan(plan: FoldPlan) -> None:
    """Validate the fixed EXP-046 five-outer/three-inner patient split shape.

    Only object types and patient identity keys are read. Response tensors,
    expression, drug features, and all other values are left untouched.
    """
    if type(plan) is not FoldPlan:
        raise ContractError("plan must be exactly FoldPlan")
    if (
        not isinstance(plan.cohort_patient_ids, frozenset)
        or not plan.cohort_patient_ids
        or any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in plan.cohort_patient_ids
        )
    ):
        raise ContractError(
            "cohort_patient_ids must be a non-empty frozenset of trimmed identities")
    cohort_ids = plan.cohort_patient_ids
    if not isinstance(plan.outer_folds, tuple) or len(plan.outer_folds) != 5:
        raise ContractError("EXP-046 requires exactly five outer folds")

    outer_keys: set[str] = set()
    outer_test_union: set[str] = set()
    for outer in plan.outer_folds:
        if type(outer) is not OuterFoldSlice:
            raise ContractError("each outer fold must be exactly OuterFoldSlice")
        outer_key = _validate_key(outer.fold_key, "outer fold key")
        if outer_key in outer_keys:
            raise ContractError("duplicate outer fold key")
        outer_keys.add(outer_key)

        outer_train_ids = _patient_ids(
            outer.outer_train, LabeledPairBatch, "outer_train")
        outer_test_ids = _patient_ids(
            outer.outer_test, PairInputBatch, "outer_test")
        if outer_train_ids.intersection(outer_test_ids):
            raise ContractError("outer train/test patients overlap")
        if not outer_train_ids.issubset(cohort_ids) or not outer_test_ids.issubset(cohort_ids):
            raise ContractError("outer fold contains patients outside the declared cohort")
        if outer_test_union.intersection(outer_test_ids):
            raise ContractError("outer-test folds overlap")
        outer_test_union.update(outer_test_ids)
        if outer_train_ids != cohort_ids.difference(outer_test_ids):
            raise ContractError("outer training split is not the cohort complement")
        if not isinstance(outer.inner_folds, tuple) or len(outer.inner_folds) != 3:
            raise ContractError("each outer fold requires exactly three inner folds")

        inner_keys: set[str] = set()
        validation_union: set[str] = set()
        for inner in outer.inner_folds:
            if type(inner) is not InnerFoldSlice:
                raise ContractError("each inner fold must be exactly InnerFoldSlice")
            inner_key = _validate_key(inner.fold_key, "inner fold key")
            if inner_key in inner_keys:
                raise ContractError("duplicate inner fold key")
            inner_keys.add(inner_key)

            train_ids = _patient_ids(
                inner.train, LabeledPairBatch, "inner_train")
            validation_ids = _patient_ids(
                inner.validation, LabeledPairBatch, "inner_validation")
            if train_ids.intersection(validation_ids):
                raise ContractError(
                    "inner train/validation patients overlap")
            if not train_ids.issubset(outer_train_ids):
                raise ContractError(
                    "inner training patients fall outside outer training")
            if not validation_ids.issubset(outer_train_ids):
                raise ContractError(
                    "inner validation patients fall outside outer training")
            if validation_ids.intersection(outer_test_ids):
                raise ContractError(
                    "outer-test patients appear in inner validation")
            repeated_validation_ids = validation_union.intersection(validation_ids)
            if repeated_validation_ids:
                raise ContractError(
                    "inner validation folds overlap; "
                    f"overlap_count={len(repeated_validation_ids)}")
            validation_union.update(validation_ids)
            if train_ids != outer_train_ids.difference(validation_ids):
                raise ContractError(
                    "inner training split is not the outer-training complement")

        if validation_union != set(outer_train_ids):
            missing_count = len(set(outer_train_ids).difference(validation_union))
            extra_count = len(validation_union.difference(outer_train_ids))
            raise ContractError(
                f"three inner validation folds must cover outer training patients "
                f"missing_count={missing_count}, extra_count={extra_count}")

    if outer_test_union != set(cohort_ids):
        missing_count = len(set(cohort_ids).difference(outer_test_union))
        extra_count = len(outer_test_union.difference(cohort_ids))
        raise ContractError(
            f"outer-test folds must partition the declared cohort; "
            f"missing_count={missing_count}, extra_count={extra_count}")


# The order is part of the orchestration contract and is intentionally not
# derived from the caller's mapping insertion order.
ARM_ORDER: tuple[Exp046Arm, ...] = ("P-A", "P-C", "P-R", "P-T", "P-N")


@dataclass(frozen=True)
class CandidateDefinition:
    """One caller-defined candidate; its values are never chosen here."""

    candidate_id: str
    settings: CandidateTraining


@dataclass(frozen=True)
class ArmSearchPlan:
    """Ordered candidate definitions and search seeds for one arm."""

    candidates: tuple[CandidateDefinition, ...]
    search_seeds: tuple[int, ...]


@dataclass(frozen=True)
class SourceIdentityGroupMapping:
    """In-memory PDO-group to source-entity and alias crosswalk row."""

    pdo_patient_group_key: str = field(repr=False)
    source_entity_group_ids: tuple[str, ...] = field(repr=False)
    source_alias_ids: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True)
class SourceHoldoutIdentityManifest:
    """Frozen identity crosswalk subset for one outer/inner holdout slice."""

    outer_fold_key: str
    inner_fold_key: str | None
    manifest_version: str
    manifest_sha256: str
    identity_mapping_key: str
    identity_mapping_version: str
    identity_mapping_sha256: str
    mappings: tuple[SourceIdentityGroupMapping, ...] = field(repr=False)


@dataclass(frozen=True)
class ArmModelInputs:
    """Caller-injected architecture, source state, lineage references, and maps."""

    backbone_factory: Callable[[], torch.nn.Module]
    source: SourceFoldInitialization = field(repr=False)
    branch_initialization: BranchInitialization | None
    ordinary_control_map: AlignedPathwayMap | None = field(repr=False)
    frozen_random_map: AlignedPathwayMap | None = field(repr=False)
    true_pathway_map: AlignedPathwayMap | None = field(repr=False)
    source_input_sha256: str
    checkpoint_artifact_sha256: str
    checkpoint_lineage_sha256: str
    mask_execution_log_sha256: str
    source_input_ref: str = field(repr=False)
    checkpoint_artifact_ref: str = field(repr=False)
    mask_execution_log_ref: str = field(repr=False)
    source_holdout_manifest: SourceHoldoutIdentityManifest = field(repr=False)


@dataclass(frozen=True)
class RegisteredRandomMap:
    map_seed: int
    sha256: str


@dataclass(frozen=True)
class Exp046Registration:
    """The current EXP-046 YAML content bound to its registered map entries."""

    config_sha256: str
    final_model_seeds: tuple[int, ...]
    random_maps: tuple[RegisteredRandomMap, ...]

    @property
    def map_seeds(self) -> tuple[int, ...]:
        return tuple(item.map_seed for item in self.random_maps)

    @property
    def map_sha256_by_seed(self) -> dict[int, str]:
        return {item.map_seed: item.sha256 for item in self.random_maps}


@dataclass(frozen=True)
class PRInnerMapAssignment:
    """One predeclared P-R candidate/fold/search-seed map opportunity."""

    outer_fold_position: int
    outer_fold_key: str
    candidate_id: str
    search_seed: int
    model_seed: int
    map_seed: int
    inner_fold_keys: tuple[str, ...]


@dataclass(frozen=True)
class WorkloadPlan:
    """Exact registered map schedule and fit-call shape for the budget gate."""

    outer_fold_count: int
    inner_fold_count_per_outer: int
    outer_fold_keys: tuple[str, ...]
    inner_fold_keys_by_outer: tuple[tuple[str, tuple[str, ...]], ...]
    arms: tuple[Exp046Arm, ...]
    candidates: tuple[CandidateDefinition, ...]
    search_seeds: tuple[int, ...]
    final_seeds: tuple[int, ...]
    config_sha256: str
    registered_maps: tuple[RegisteredRandomMap, ...]
    inner_map_assignments: tuple[PRInnerMapAssignment, ...]
    inner_fit_call_count: int
    outer_fit_call_count: int


@dataclass(frozen=True)
class SourceRetentionGateRequest:
    """Metadata and source setup for a caller-supplied retention gate.

    This request contains no outer-test inputs or labels. The driver's
    structural checks on source declarations do not prove which identities
    were actually used by upstream source training.
    """

    outer_fold_key: str
    arm: Exp046Arm
    inner_fold_key: str | None
    model_seed: int
    map_seed: int | None
    model_inputs: ArmModelInputs = field(repr=False)


@dataclass(frozen=True)
class SourceExposureAuditRequest:
    """Inputs for a separate identity-exposure audit callback."""

    experiment_id: str
    outer_fold_key: str
    arm: Exp046Arm
    inner_fold_key: str | None
    search_seed: int | None
    model_seed: int
    map_seed: int | None
    pathway_map_sha256: str | None
    source_input_sha256: str
    checkpoint_artifact_sha256: str
    checkpoint_lineage_sha256: str
    loaded_state_dict_sha256: str
    expected_mask_execution_log_sha256: str
    source_input_ref: str = field(repr=False)
    checkpoint_artifact_ref: str = field(repr=False)
    mask_execution_log_ref: str = field(repr=False)
    manifest: SourceHoldoutIdentityManifest = field(repr=False)
    expected_pdo_group_count: int
    expected_source_group_sha256: str
    expected_source_group_count: int
    expected_source_alias_sha256: str
    expected_source_alias_count: int


@dataclass(frozen=True)
class SourceExposureAuditReceipt:
    """Redacted receipt from a mask/log and checkpoint-lineage audit callback.

    The driver checks receipt formatting, context/hash bindings and reported
    intersections. It does not prove that an external audit was executed
    truthfully or that upstream artifacts were completely captured.
    """

    experiment_id: str
    outer_fold_key: str
    arm: Exp046Arm
    inner_fold_key: str | None
    search_seed: int | None
    model_seed: int
    map_seed: int | None
    pathway_map_sha256: str | None
    source_input_sha256: str
    checkpoint_artifact_sha256: str
    checkpoint_lineage_sha256: str
    loaded_state_dict_sha256: str
    mask_execution_log_sha256: str
    manifest_version: str
    manifest_sha256: str
    identity_mapping_key: str
    identity_mapping_version: str
    identity_mapping_sha256: str
    identity_set_hash_scheme: str
    heldout_pdo_group_count: int
    heldout_source_group_sha256: str
    heldout_source_group_count: int
    heldout_source_alias_sha256: str
    heldout_source_alias_count: int
    actual_exposed_source_group_sha256: str
    actual_exposed_source_group_count: int
    actual_exposed_source_alias_sha256: str
    actual_exposed_source_alias_count: int
    source_group_intersection_count: int
    source_alias_intersection_count: int


@dataclass(frozen=True)
class SourceExposureAuditRecord:
    request: SourceExposureAuditRequest = field(repr=False)
    receipt: SourceExposureAuditReceipt


@dataclass(frozen=True)
class InnerOpportunityReceipt:
    """Inner-fit receipts for one candidate/search-seed opportunity."""

    candidate_id: str
    search_seed: int
    model_seed: int
    map_seed: int | None
    pathway_map_sha256: str | None
    inner_fold_keys: tuple[str, ...]
    receipts: tuple[InnerFoldReceipt, ...]


@dataclass(frozen=True)
class CandidateSelectionPacket:
    """Only candidate identifiers, search seeds, and inner receipts."""

    arm: Exp046Arm
    outer_fold_key: str
    registered_maps: tuple[RegisteredRandomMap, ...]
    map_seed_by_search_seed: tuple[tuple[int, int | None], ...]
    opportunities: tuple[InnerOpportunityReceipt, ...]


@dataclass(frozen=True)
class EpochSelectionPacket:
    """Only selected-candidate inner receipts and explicit final seeds."""

    arm: Exp046Arm
    outer_fold_key: str
    candidate_id: str
    registered_maps: tuple[RegisteredRandomMap, ...]
    map_seed_by_search_seed: tuple[tuple[int, int | None], ...]
    opportunities: tuple[InnerOpportunityReceipt, ...]
    final_seeds: tuple[int, ...]


@dataclass(frozen=True)
class InnerCallReceipt:
    """One receipt returned by the existing inner-fold fit primitive."""

    candidate_id: str
    search_seed: int
    model_seed: int
    map_seed: int | None
    pathway_map_sha256: str | None
    inner_fold_key: str
    receipt: InnerFoldReceipt


@dataclass(frozen=True)
class OuterSeedReceipt:
    """One outer refit receipt keyed by arm/fold/model seed/map seed."""

    arm: Exp046Arm
    outer_fold_key: str
    model_seed: int
    map_seed: int | None
    pathway_map_sha256: str | None
    candidate_id: str
    selected_epochs: int
    receipt: OuterFoldReceipt


@dataclass(frozen=True)
class ArmFoldReceipt:
    """Candidate search, selection, and cost primitives for one arm/fold."""

    arm: Exp046Arm
    outer_fold_key: str
    candidate_opportunities: tuple[InnerOpportunityReceipt, ...]
    selected_candidate_id: str
    selected_epochs_by_seed: tuple[tuple[int, int], ...]
    inner_calls: tuple[InnerCallReceipt, ...]
    outer_seed_receipts: tuple[OuterSeedReceipt, ...]
    outer_train_patient_count: int
    outer_test_patient_count: int
    inner_fit_api_elapsed_seconds_sum: float
    outer_fit_api_elapsed_seconds_sum: float
    reported_fit_api_elapsed_seconds_sum: float
    cost_scope: str


@dataclass(frozen=True)
class KeyedPredictionBatch:
    """Outer predictions with the arm, fold, selected candidate, and seed key."""

    arm: Exp046Arm
    outer_fold_key: str
    candidate_id: str
    model_seed: int
    map_seed: int | None
    pathway_map_sha256: str | None
    predictions: OuterKeyedPredictions = field(repr=False)


@dataclass(frozen=True)
class Exp046RunResult:
    """In-memory keyed predictions and engineering receipts; contains no labels."""

    workload: WorkloadPlan
    arm_fold_receipts: tuple[ArmFoldReceipt, ...]
    predictions: tuple[KeyedPredictionBatch, ...]
    outer_seed_receipts: tuple[OuterSeedReceipt, ...]
    source_exposure_audits: tuple[SourceExposureAuditRecord, ...]
    reported_fit_api_elapsed_seconds_sum: float
    cost_scope: str

    def validate(self) -> None:
        expected_arm_folds = self.workload.outer_fold_count * len(self.workload.arms)
        expected_outer_runs = self.workload.outer_fold_count * (
            4 * len(self.workload.final_seeds)
            + len(self.workload.final_seeds) * len(self.workload.registered_maps))
        expected_inner_fits = (
            self.workload.outer_fold_count * len(self.workload.arms)
            * len(self.workload.candidates) * len(self.workload.search_seeds)
            * self.workload.inner_fold_count_per_outer
        )
        if self.workload.inner_fit_call_count != expected_inner_fits:
            raise ContractError("workload inner-fit count disagrees with its declared opportunities")
        if self.workload.outer_fit_call_count != expected_outer_runs:
            raise ContractError("workload outer-fit count disagrees with the registered map cross")
        if len(self.arm_fold_receipts) != expected_arm_folds:
            raise ContractError("run result does not cover every arm/outer-fold pair")
        if (len(self.outer_seed_receipts) != expected_outer_runs
                or len(self.predictions) != expected_outer_runs):
            raise ContractError("run result does not cover every arm/fold/model-seed/map-seed run")

        map_sha_by_seed = {item.map_seed: item.sha256
                           for item in self.workload.registered_maps}
        map_seeds = set(map_sha_by_seed)
        receipt_keys: set[tuple[str, str, int, int | None]] = set()
        for receipt in self.outer_seed_receipts:
            key = (receipt.arm, receipt.outer_fold_key, receipt.model_seed,
                   receipt.map_seed)
            if key in receipt_keys:
                raise ContractError("duplicate arm/fold/model-seed/map-seed engineering receipt")
            if receipt.arm == "P-R":
                if receipt.map_seed not in map_seeds:
                    raise ContractError("P-R outer receipt has an unregistered map_seed")
                if receipt.pathway_map_sha256 != map_sha_by_seed[receipt.map_seed]:
                    raise ContractError("P-R outer receipt has a mismatched registered map hash")
            elif receipt.map_seed is not None:
                raise ContractError("non-P-R outer receipt must have no map_seed")
            elif receipt.pathway_map_sha256 is not None:
                raise ContractError("non-P-R outer receipt must have no random-map hash")
            receipt_keys.add(key)

        expected_receipt_keys = {
            (arm, outer_key, model_seed, map_seed)
            for outer_key in self.workload.outer_fold_keys
            for arm in self.workload.arms
            for model_seed in self.workload.final_seeds
            for map_seed in (
                tuple(sorted(map_seeds)) if arm == "P-R" else (None,))
        }
        if receipt_keys != expected_receipt_keys:
            raise ContractError("outer receipts do not match the registered model/map-seed schedule")

        prediction_run_keys: set[tuple[str, str, int, int | None]] = set()
        prediction_keys: set[tuple[str, str, int, int | None, str, str]] = set()
        for batch in self.predictions:
            run_key = (batch.arm, batch.outer_fold_key, batch.model_seed,
                       batch.map_seed)
            if run_key in prediction_run_keys:
                raise ContractError("duplicate keyed prediction batch for arm/fold/model-seed/map-seed")
            prediction_run_keys.add(run_key)
            if run_key not in receipt_keys:
                raise ContractError("prediction batch has no matching arm/fold/model-seed/map-seed receipt")
            matching_receipt = next(
                item for item in self.outer_seed_receipts
                if (item.arm, item.outer_fold_key, item.model_seed, item.map_seed) == run_key
            )
            if batch.pathway_map_sha256 != matching_receipt.pathway_map_sha256:
                raise ContractError("prediction batch map hash does not match its fit receipt")
            if type(batch.predictions) is not OuterKeyedPredictions:
                raise ContractError("predictions must be exactly OuterKeyedPredictions")
            try:
                batch.predictions.validate()
            except (TypeError, ValueError, AttributeError) as exc:
                raise ContractError("outer keyed predictions failed validation") from exc
            for patient_id, drug_id in zip(
                    batch.predictions.patient_ids, batch.predictions.drug_ids):
                key = (batch.arm, batch.outer_fold_key, batch.model_seed,
                       batch.map_seed,
                       patient_id, drug_id)
                if key in prediction_keys:
                    raise ContractError("duplicate keyed prediction in run result")
                prediction_keys.add(key)
        if prediction_run_keys != receipt_keys:
            raise ContractError("prediction batches do not match the run receipt keys")

        expected_assignment_keys = set()
        if len(self.workload.registered_maps) != 3:
            raise ContractError("workload must contain exactly three registered P-R maps")
        map_seeds_in_order = tuple(item.map_seed for item in self.workload.registered_maps)
        if len(set(map_seeds_in_order)) != 3:
            raise ContractError("workload repeats a registered map_seed")
        for outer_position, outer_key in enumerate(self.workload.outer_fold_keys):
            for candidate in self.workload.candidates:
                for search_position, search_seed in enumerate(self.workload.search_seeds):
                    map_seed = map_seeds_in_order[
                        (search_position + outer_position) % len(map_seeds_in_order)]
                    expected_assignment_keys.add((
                        outer_position, outer_key, candidate.candidate_id,
                        search_seed, search_seed, map_seed,
                        dict(self.workload.inner_fold_keys_by_outer)[outer_key],
                    ))
        observed_assignment_keys = {
            (item.outer_fold_position, item.outer_fold_key, item.candidate_id,
             item.search_seed, item.model_seed, item.map_seed, item.inner_fold_keys)
            for item in self.workload.inner_map_assignments
        }
        if observed_assignment_keys != expected_assignment_keys:
            raise ContractError("workload P-R map assignments do not match the frozen rotation")

        opportunity_map = {
            (assignment.outer_fold_key, assignment.candidate_id, assignment.search_seed):
            assignment.map_seed
            for assignment in self.workload.inner_map_assignments
        }
        expected_arm_fold_keys = {
            (outer_key, arm)
            for outer_key in self.workload.outer_fold_keys
            for arm in self.workload.arms
        }
        observed_arm_fold_keys = {
            (receipt.outer_fold_key, receipt.arm)
            for receipt in self.arm_fold_receipts
        }
        if (len(observed_arm_fold_keys) != len(self.arm_fold_receipts)
                or observed_arm_fold_keys != expected_arm_fold_keys):
            raise ContractError("arm/fold receipts do not uniquely cover the full schedule")
        for fold_receipt in self.arm_fold_receipts:
            expected_opportunities = tuple(
                (candidate.candidate_id, search_seed)
                for candidate in self.workload.candidates
                for search_seed in self.workload.search_seeds
            )
            observed_opportunities = tuple(
                (item.candidate_id, item.search_seed)
                for item in fold_receipt.candidate_opportunities
            )
            if observed_opportunities != expected_opportunities:
                raise ContractError("arm/fold candidate opportunities do not match the common search plan")
            for item in fold_receipt.candidate_opportunities:
                expected_map_seed = (
                    opportunity_map[(fold_receipt.outer_fold_key,
                                     item.candidate_id, item.search_seed)]
                    if fold_receipt.arm == "P-R" else None
                )
                expected_map_hash = (
                    map_sha_by_seed[expected_map_seed]
                    if expected_map_seed is not None else None
                )
                if (item.model_seed != item.search_seed
                        or item.map_seed != expected_map_seed
                        or item.pathway_map_sha256 != expected_map_hash
                        or item.inner_fold_keys
                        != dict(self.workload.inner_fold_keys_by_outer)[fold_receipt.outer_fold_key]):
                    raise ContractError("inner opportunity receipt has a mismatched seed/map/fold binding")
            expected_outer_receipts = (
                len(self.workload.final_seeds) * len(self.workload.registered_maps)
                if fold_receipt.arm == "P-R" else len(self.workload.final_seeds)
            )
            if len(fold_receipt.outer_seed_receipts) != expected_outer_receipts:
                raise ContractError("arm/fold outer receipt count disagrees with its map-seed schedule")

        if not self.source_exposure_audits:
            raise ContractError("run result lacks pre-fit source exposure audit receipts")
        audit_keys: set[tuple[str, str, str | None, int | None, int, int | None]] = set()
        for record in self.source_exposure_audits:
            request = record.request
            key = (request.outer_fold_key, request.arm, request.inner_fold_key,
                   request.search_seed, request.model_seed, request.map_seed)
            if key in audit_keys:
                raise ContractError("duplicate source exposure audit receipt key")
            audit_keys.add(key)
            _validate_source_exposure_receipt(record.receipt, request)
        expected_audit_keys = set()
        inner_by_outer = dict(self.workload.inner_fold_keys_by_outer)
        map_by_outer_search: dict[tuple[str, int], int] = {}
        for assignment in self.workload.inner_map_assignments:
            map_by_outer_search[(assignment.outer_fold_key,
                                 assignment.search_seed)] = assignment.map_seed
        for outer_position, outer_key in enumerate(self.workload.outer_fold_keys):
            inner_keys = inner_by_outer[outer_key]
            for arm in self.workload.arms:
                for search_seed in self.workload.search_seeds:
                    map_seed = (
                        map_by_outer_search[(outer_key, search_seed)]
                        if arm == "P-R" else None
                    )
                    expected_audit_keys.update(
                        (outer_key, arm, inner_key, search_seed, search_seed, map_seed)
                        for inner_key in inner_keys
                    )
                for model_seed in self.workload.final_seeds:
                    for map_seed in (
                        tuple(item.map_seed for item in self.workload.registered_maps)
                        if arm == "P-R" else (None,)
                    ):
                        expected_audit_keys.add(
                            (outer_key, arm, None, None, model_seed, map_seed))
        if audit_keys != expected_audit_keys:
            raise ContractError("source exposure audits do not cover the complete fit schedule")

        for receipt in self.arm_fold_receipts:
            expected_inner = sum(item.receipt.elapsed_seconds for item in receipt.inner_calls)
            expected_outer = sum(item.receipt.elapsed_seconds
                                 for item in receipt.outer_seed_receipts)
            if (not math.isclose(receipt.inner_fit_api_elapsed_seconds_sum,
                                 expected_inner, rel_tol=0.0, abs_tol=1e-12)
                    or not math.isclose(receipt.outer_fit_api_elapsed_seconds_sum,
                                        expected_outer, rel_tol=0.0, abs_tol=1e-12)
                    or not math.isclose(receipt.reported_fit_api_elapsed_seconds_sum,
                                        expected_inner + expected_outer,
                                        rel_tol=0.0, abs_tol=1e-12)):
                raise ContractError("arm/fold cost summary must equal reported primitive elapsed_seconds")
        expected_total = sum(
            receipt.reported_fit_api_elapsed_seconds_sum
            for receipt in self.arm_fold_receipts)
        if not math.isclose(self.reported_fit_api_elapsed_seconds_sum,
                            expected_total, rel_tol=0.0, abs_tol=1e-12):
            raise ContractError("run cost summary must equal arm/fold primitive cost summaries")

        if not math.isfinite(self.reported_fit_api_elapsed_seconds_sum) \
                or self.reported_fit_api_elapsed_seconds_sum < 0:
            raise ContractError("reported primitive elapsed-time sum is invalid")


IDENTITY_SET_HASH_SCHEME = "EXP046-IDSET-JSON-UTF8-SORTED-v1"


_COST_SCOPE = (
    "sum of elapsed_seconds reported by fit_inner_candidate and fit_outer_fold "
    "calls only; excludes source preparation, input materialization, gates, "
    "selection callbacks, driver overhead, setup, and any unreported work; "
    "not a full experiment cost or budget estimate"
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, field_name: str) -> str:
    if not _is_sha256(value):
        raise ContractError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _identity_set_sha256(values: set[str] | frozenset[str]) -> str:
    ordered = sorted(values, key=lambda value: value.encode("utf-8"))
    payload = json.dumps(
        [IDENTITY_SET_HASH_SCHEME, ordered],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_manifest_content_sha256(
    manifest: SourceHoldoutIdentityManifest,
) -> str:
    rows = []
    for row in sorted(
            manifest.mappings,
            key=lambda item: item.pdo_patient_group_key.encode("utf-8")):
        rows.append({
            "pdo_patient_group_key": row.pdo_patient_group_key,
            "source_entity_group_ids": sorted(
                row.source_entity_group_ids, key=lambda value: value.encode("utf-8")),
            "source_alias_ids": sorted(
                row.source_alias_ids, key=lambda value: value.encode("utf-8")),
        })
    payload = {
        "outer_fold_key": manifest.outer_fold_key,
        "inner_fold_key": manifest.inner_fold_key,
        "manifest_version": manifest.manifest_version,
        "identity_mapping_key": manifest.identity_mapping_key,
        "identity_mapping_version": manifest.identity_mapping_version,
        "identity_mapping_sha256": manifest.identity_mapping_sha256,
        "mappings": rows,
    }
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _load_exp046_registration() -> Exp046Registration:
    """Read only the registered EXP-046 config, never data or fold-label files."""
    config_path = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "EXP-046.yaml"
    try:
        raw = config_path.read_bytes()
        config = yaml.safe_load(raw.decode("utf-8"))
    except Exception:
        raise ContractError("EXP-046 config could not be read or parsed") from None
    if not isinstance(config, Mapping):
        raise ContractError("EXP-046 config must be a mapping")
    cohort = config.get("cohort")
    random_maps = config.get("random_pathway_maps")
    if not isinstance(cohort, Mapping) or not isinstance(random_maps, Mapping):
        raise ContractError("EXP-046 config is missing cohort or random-map registration")
    outer_fold_count = cohort.get("outer_patient_folds")
    inner_fold_count = cohort.get("inner_patient_folds")
    if (isinstance(outer_fold_count, bool) or not isinstance(outer_fold_count, int)
            or outer_fold_count != 5
            or isinstance(inner_fold_count, bool) or not isinstance(inner_fold_count, int)
            or inner_fold_count != 3):
        raise ContractError(
            "EXP-046 config must register exactly five outer and three inner patient folds")
    final_seeds = cohort.get("final_seeds")
    if (not isinstance(final_seeds, list) or len(final_seeds) != 3
            or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                   for seed in final_seeds)
            or len(set(final_seeds)) != 3):
        raise ContractError("EXP-046 config must register exactly three unique final model seeds")

    map_seeds = random_maps.get("seeds")
    map_count = random_maps.get("count")
    hashes = random_maps.get("map_sha256")
    if (not isinstance(map_seeds, list) or len(map_seeds) != 3
            or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                   for seed in map_seeds)
            or len(set(map_seeds)) != 3
            or map_count != 3
            or not isinstance(hashes, Mapping)):
        raise ContractError("EXP-046 config must register exactly three random-map seeds and hashes")
    normalized_hashes: dict[int, str] = {}
    for key, value in hashes.items():
        try:
            seed = int(key)
        except (TypeError, ValueError):
            raise ContractError("EXP-046 random-map registry has an invalid seed key") from None
        if seed in normalized_hashes:
            raise ContractError("EXP-046 random-map registry repeats a seed key")
        normalized_hashes[seed] = _require_sha256(value, "registered map hash")
    if set(normalized_hashes) != set(map_seeds):
        raise ContractError("EXP-046 random-map seed and hash registries disagree")
    if random_maps.get("use_only_registered_seed_to_sha256_entries") is not True:
        raise ContractError("EXP-046 config must forbid unregistered random maps")
    if random_maps.get("map_seed_is_independent_of_model_initialization_seed") is not True:
        raise ContractError("EXP-046 config must separate map and model seeds")

    inner_assignment = random_maps.get("inner_assignment")
    final_evaluation = random_maps.get("final_outer_evaluation")
    if not isinstance(inner_assignment, Mapping) or not isinstance(final_evaluation, Mapping):
        raise ContractError("EXP-046 config is missing the registered random-map schedule")
    rotation = inner_assignment.get("fixed_balanced_rotation")
    if (inner_assignment.get("search_seed_count_must_be_multiple_of_map_count") is not True
            or inner_assignment.get("equal_map_opportunities_per_candidate_per_outer_fold") is not True
            or not isinstance(rotation, str)
            or "(j+k) mod 3" not in rotation
            or "all three inner folds" not in rotation
            or "Same schedule for every candidate" not in rotation):
        raise ContractError("EXP-046 config does not contain the frozen balanced inner-map rotation")
    if (final_evaluation.get("cross_all_registered_maps_with_all_three_final_initializations") is not True
            or final_evaluation.get("P_R_fits_per_outer_fold") != 9
            or final_evaluation.get("other_arm_fits_per_outer_fold") != 3
            or final_evaluation.get("map_seed_and_model_seed_both_in_prediction_receipt_and_budget") is not True):
        raise ContractError("EXP-046 config does not contain the frozen final P-R map/model cross")

    return Exp046Registration(
        config_sha256=hashlib.sha256(raw).hexdigest(),
        final_model_seeds=tuple(final_seeds),
        random_maps=tuple(
            RegisteredRandomMap(seed, normalized_hashes[seed])
            for seed in map_seeds
        ),
    )


def _pathway_map_sha256(pathway_map: AlignedPathwayMap) -> str:
    """Recompute the registered canonical edge digest from a map tensor."""
    if type(pathway_map) is not AlignedPathwayMap:
        raise ContractError("P-R map must be exactly AlignedPathwayMap")
    matrix = pathway_map.incidence_matrix
    if matrix.device.type != "cpu":
        raise ContractError("registered pathway maps must be supplied on CPU for preflight")
    array = matrix.detach().numpy()
    edges = [
        (pathway_map.gene_axis[gene_index], pathway_map.pathway_names[pathway_index])
        for pathway_index, gene_index in np.argwhere(array != 0)
    ]
    ordered = sorted(edges, key=lambda edge: (
        edge[0].encode("utf-8"), edge[1].encode("utf-8")))
    payload = "".join(f"{gene}\t{pathway}\n" for gene, pathway in ordered).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_source_holdout_manifest(
    manifest: SourceHoldoutIdentityManifest,
    *,
    outer_fold_key: str,
    inner_fold_key: str | None,
    expected_pdo_group_keys: frozenset[str],
) -> tuple[int, str, int, str, int]:
    if type(manifest) is not SourceHoldoutIdentityManifest:
        raise ContractError("source_holdout_manifest must be exactly SourceHoldoutIdentityManifest")
    if manifest.outer_fold_key != outer_fold_key or manifest.inner_fold_key != inner_fold_key:
        raise ContractError("source holdout manifest is bound to a different fold slice")
    for value, name in (
        (manifest.manifest_version, "manifest_version"),
        (manifest.identity_mapping_key, "identity_mapping_key"),
        (manifest.identity_mapping_version, "identity_mapping_version"),
    ):
        _validate_key(value, name)
    _require_sha256(manifest.manifest_sha256, "source holdout manifest SHA-256")
    _require_sha256(manifest.identity_mapping_sha256, "identity mapping SHA-256")
    if not isinstance(manifest.mappings, tuple) or not manifest.mappings:
        raise ContractError("source holdout manifest mappings must be a non-empty tuple")

    pdo_keys: set[str] = set()
    source_groups: set[str] = set()
    source_aliases: set[str] = set()
    for row in manifest.mappings:
        if type(row) is not SourceIdentityGroupMapping:
            raise ContractError("source holdout manifest rows must be exactly SourceIdentityGroupMapping")
        for value, name in (
            (row.pdo_patient_group_key, "PDO group key"),
        ):
            _validate_key(value, name)
        if row.pdo_patient_group_key in pdo_keys:
            raise ContractError("source holdout manifest repeats a PDO group key")
        pdo_keys.add(row.pdo_patient_group_key)
        for values, name, target in (
            (row.source_entity_group_ids, "source entity groups", source_groups),
            (row.source_alias_ids, "source aliases", source_aliases),
        ):
            if not isinstance(values, tuple) or not values:
                raise ContractError(f"source holdout manifest requires mapped {name}")
            if any(not isinstance(value, str) or not value or value != value.strip()
                   for value in values):
                raise ContractError(f"source holdout manifest {name} must contain trimmed identities")
            if len(set(values)) != len(values):
                raise ContractError(f"source holdout manifest repeats a value in {name}")
            target.update(values)
    if pdo_keys != set(expected_pdo_group_keys):
        raise ContractError(
            "source holdout manifest does not exactly cover the PDO holdout groups; "
            f"expected_count={len(expected_pdo_group_keys)}, observed_count={len(pdo_keys)}")
    if _source_manifest_content_sha256(manifest) != manifest.manifest_sha256:
        raise ContractError("source holdout manifest content does not match its declared SHA-256")
    return (
        len(pdo_keys),
        _identity_set_sha256(source_groups),
        len(source_groups),
        _identity_set_sha256(source_aliases),
        len(source_aliases),
    )


def _validate_source_exposure_receipt(
    receipt: SourceExposureAuditReceipt,
    request: SourceExposureAuditRequest,
) -> None:
    if type(receipt) is not SourceExposureAuditReceipt:
        raise ContractError("source_exposure_gate must return exactly SourceExposureAuditReceipt")
    bindings = (
        ("experiment_id", receipt.experiment_id, request.experiment_id),
        ("outer_fold_key", receipt.outer_fold_key, request.outer_fold_key),
        ("arm", receipt.arm, request.arm),
        ("inner_fold_key", receipt.inner_fold_key, request.inner_fold_key),
        ("search_seed", receipt.search_seed, request.search_seed),
        ("model_seed", receipt.model_seed, request.model_seed),
        ("map_seed", receipt.map_seed, request.map_seed),
        ("pathway_map_sha256", receipt.pathway_map_sha256, request.pathway_map_sha256),
        ("source_input_sha256", receipt.source_input_sha256, request.source_input_sha256),
        ("checkpoint_artifact_sha256", receipt.checkpoint_artifact_sha256,
         request.checkpoint_artifact_sha256),
        ("checkpoint_lineage_sha256", receipt.checkpoint_lineage_sha256,
         request.checkpoint_lineage_sha256),
        ("loaded_state_dict_sha256", receipt.loaded_state_dict_sha256,
         request.loaded_state_dict_sha256),
        ("mask_execution_log_sha256", receipt.mask_execution_log_sha256,
         request.expected_mask_execution_log_sha256),
        ("manifest_version", receipt.manifest_version, request.manifest.manifest_version),
        ("manifest_sha256", receipt.manifest_sha256, request.manifest.manifest_sha256),
        ("identity_mapping_key", receipt.identity_mapping_key,
         request.manifest.identity_mapping_key),
        ("identity_mapping_version", receipt.identity_mapping_version,
         request.manifest.identity_mapping_version),
        ("identity_mapping_sha256", receipt.identity_mapping_sha256,
         request.manifest.identity_mapping_sha256),
        ("identity_set_hash_scheme", receipt.identity_set_hash_scheme,
         IDENTITY_SET_HASH_SCHEME),
        ("heldout_pdo_group_count", receipt.heldout_pdo_group_count,
         request.expected_pdo_group_count),
        ("heldout_source_group_sha256", receipt.heldout_source_group_sha256,
         request.expected_source_group_sha256),
        ("heldout_source_group_count", receipt.heldout_source_group_count,
         request.expected_source_group_count),
        ("heldout_source_alias_sha256", receipt.heldout_source_alias_sha256,
         request.expected_source_alias_sha256),
        ("heldout_source_alias_count", receipt.heldout_source_alias_count,
         request.expected_source_alias_count),
    )
    if any(observed != expected for _, observed, expected in bindings):
        mismatches = tuple(name for name, observed, expected in bindings
                           if observed != expected)
        raise ContractError(f"source exposure receipt binding mismatch; fields={mismatches}")
    for value, name in (
        (receipt.actual_exposed_source_group_sha256,
         "actual_exposed_source_group_sha256"),
        (receipt.actual_exposed_source_alias_sha256,
         "actual_exposed_source_alias_sha256"),
    ):
        _require_sha256(value, name)
    for value, name in (
        (receipt.actual_exposed_source_group_count,
         "actual_exposed_source_group_count"),
        (receipt.actual_exposed_source_alias_count,
         "actual_exposed_source_alias_count"),
        (receipt.source_group_intersection_count,
         "source_group_intersection_count"),
        (receipt.source_alias_intersection_count,
         "source_alias_intersection_count"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError(f"source exposure receipt {name} must be a non-negative integer")
    if receipt.source_group_intersection_count != 0:
        raise ContractError("source exposure audit reports a source-group intersection")
    if receipt.source_alias_intersection_count != 0:
        raise ContractError("source exposure audit reports a source-alias intersection")


def _valid_seed(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{name} must be an explicit non-negative integer")
    return value


def _validate_search_plans(
    plans_by_arm: Mapping[Exp046Arm, ArmSearchPlan],
    final_seeds: tuple[int, ...],
    registration: Exp046Registration,
) -> tuple[tuple[CandidateDefinition, ...], tuple[int, ...]]:
    if not isinstance(plans_by_arm, Mapping):
        raise ContractError("search_plans_by_arm must be an explicit arm mapping")
    if set(plans_by_arm) != set(ARM_ORDER):
        raise ContractError("search_plans_by_arm must define exactly the five EXP-046 arms")
    if not isinstance(final_seeds, tuple) or len(final_seeds) != 3:
        raise ContractError("final_seeds must explicitly contain the three registered initializations")
    for seed in final_seeds:
        _valid_seed(seed, "final seed")
    if len(set(final_seeds)) != len(final_seeds):
        raise ContractError("final_seeds must be unique")
    if final_seeds != registration.final_model_seeds:
        raise ContractError("final_seeds must match the three registered model seeds in EXP-046 config")

    reference_candidates: tuple[CandidateDefinition, ...] | None = None
    reference_seeds: tuple[int, ...] | None = None
    for arm in ARM_ORDER:
        plan = plans_by_arm[arm]
        if type(plan) is not ArmSearchPlan:
            raise ContractError(f"search plan for {arm} must be exactly ArmSearchPlan")
        if not isinstance(plan.candidates, tuple) or not plan.candidates:
            raise ContractError(f"search plan for {arm} requires explicit candidates")
        candidate_ids: set[str] = set()
        for candidate in plan.candidates:
            if type(candidate) is not CandidateDefinition:
                raise ContractError("candidate entries must be exactly CandidateDefinition")
            _validate_key(candidate.candidate_id, "candidate_id")
            if candidate.candidate_id in candidate_ids:
                raise ContractError(f"duplicate candidate ID in arm {arm}")
            candidate_ids.add(candidate.candidate_id)
            if type(candidate.settings) is not CandidateTraining:
                raise ContractError("candidate settings must be exactly CandidateTraining")
            candidate.settings.validate()

        if not isinstance(plan.search_seeds, tuple) or not plan.search_seeds:
            raise ContractError(f"search plan for {arm} requires explicit search seeds")
        for seed in plan.search_seeds:
            _valid_seed(seed, "search seed")
        if len(set(plan.search_seeds)) != len(plan.search_seeds):
            raise ContractError(f"search seeds for {arm} must be unique")

        if reference_candidates is None:
            reference_candidates = plan.candidates
            reference_seeds = plan.search_seeds
        elif plan.candidates != reference_candidates:
            raise ContractError(
                "candidate IDs, values, and order must match across all five arms")
        elif plan.search_seeds != reference_seeds:
            raise ContractError("search seed opportunities and order must match across all five arms")

    assert reference_candidates is not None and reference_seeds is not None
    if len(reference_seeds) % len(registration.random_maps) != 0:
        raise ContractError("search seed count must be a multiple of the registered map count")
    return reference_candidates, reference_seeds


def _validate_model_inputs(
    *,
    outer_fold_key: str,
    arm: Exp046Arm,
    map_seed: int | None,
    registration: Exp046Registration,
    inputs: ArmModelInputs,
    permitted_train_patient_ids: frozenset[str],
) -> None:
    if type(inputs) is not ArmModelInputs:
        raise ContractError("model input provider must return exactly ArmModelInputs")
    if not callable(inputs.backbone_factory):
        raise ContractError("backbone_factory must be callable")
    for value, name in (
        (inputs.source_input_sha256, "source_input_sha256"),
        (inputs.checkpoint_artifact_sha256, "checkpoint_artifact_sha256"),
        (inputs.checkpoint_lineage_sha256, "checkpoint_lineage_sha256"),
        (inputs.mask_execution_log_sha256, "mask_execution_log_sha256"),
    ):
        _require_sha256(value, name)
    for value, name in (
        (inputs.source_input_ref, "source_input_ref"),
        (inputs.checkpoint_artifact_ref, "checkpoint_artifact_ref"),
        (inputs.mask_execution_log_ref, "mask_execution_log_ref"),
    ):
        if not isinstance(value, str) or not value or value != value.strip():
            raise ContractError(f"{name} must be an explicit non-empty reference")
    if type(inputs.source) is not SourceFoldInitialization:
        raise ContractError("source initialization must be exactly SourceFoldInitialization")
    try:
        inputs.source.validate()
    except (TypeError, ValueError, RuntimeError):
        raise ContractError(
            "caller-supplied source initialization failed structural validation") from None
    if inputs.source.outer_fold_key != outer_fold_key:
        raise ContractError("source initialization belongs to a different outer fold")
    if not set(inputs.source.target_fit_patient_ids).issubset(permitted_train_patient_ids):
        raise ContractError(
            "declared source target-fit identities fall outside this training slice")

    expected_map_presence = {
        "P-A": (False, False, False),
        "P-C": (True, False, False),
        "P-R": (False, True, False),
        "P-T": (False, False, True),
        "P-N": (False, False, True),
    }[arm]
    supplied_maps = (inputs.ordinary_control_map,
                     inputs.frozen_random_map,
                     inputs.true_pathway_map)
    if tuple(value is not None for value in supplied_maps) != expected_map_presence:
        raise ContractError(f"pathway map inputs do not match arm {arm}")
    for pathway_map in supplied_maps:
        if pathway_map is not None:
            if (type(pathway_map) is not AlignedPathwayMap
                    or not isinstance(pathway_map.incidence_matrix, torch.Tensor)):
                raise ContractError("pathway maps must be exactly AlignedPathwayMap values")
            if pathway_map.incidence_matrix.device.type != "cpu":
                raise ContractError("pathway maps must be supplied on CPU for preflight")
            try:
                validate_aligned_pathway_map(pathway_map)
            except (TypeError, ValueError, RuntimeError):
                raise ContractError("caller-supplied pathway map failed structural validation") from None
    if arm == "P-R":
        if map_seed not in registration.map_sha256_by_seed:
            raise ContractError("P-R fit requires an explicitly registered map_seed")
        random_map_hash = _pathway_map_sha256(inputs.frozen_random_map)
        if random_map_hash != registration.map_sha256_by_seed[map_seed]:
            raise ContractError("P-R pathway map content does not match the registered map_seed SHA-256")
    elif map_seed is not None:
        raise ContractError("non-P-R fit must not receive a map_seed")

    if arm == "P-A":
        if inputs.branch_initialization is not None:
            raise ContractError("P-A must receive an explicit None branch initialization")
    else:
        if type(inputs.branch_initialization) is not BranchInitialization:
            raise ContractError(f"{arm} requires explicit BranchInitialization")
        try:
            inputs.branch_initialization.validate()
        except (TypeError, ValueError):
            raise ContractError("caller-supplied branch initialization is invalid") from None


def _same_pathway_map(
    left: AlignedPathwayMap | None,
    right: AlignedPathwayMap | None,
) -> bool:
    if left is None or right is None:
        return left is right
    return (
        type(left) is AlignedPathwayMap
        and type(right) is AlignedPathwayMap
        and left.gene_axis == right.gene_axis
        and left.mapped_gene_ids == right.mapped_gene_ids
        and left.pathway_names == right.pathway_names
        and left.contract == right.contract
        and left.incidence_matrix.shape == right.incidence_matrix.shape
        and left.incidence_matrix.dtype == right.incidence_matrix.dtype
        and torch.equal(left.incidence_matrix, right.incidence_matrix)
    )


def _validate_receipt_fields(receipt: InnerFoldReceipt, settings: CandidateTraining) -> None:
    if type(receipt) is not InnerFoldReceipt:
        raise ContractError("inner fitter must return exactly InnerFoldReceipt")
    finite_values = (receipt.best_validation_score,
                     receipt.response_mean_train_only,
                     receipt.response_scale_train_only,
                     receipt.elapsed_seconds)
    try:
        finite = all(math.isfinite(value) for value in finite_values)
    except TypeError as exc:
        raise ContractError("inner receipt numeric fields are invalid") from exc
    if not finite or receipt.response_scale_train_only <= 0 or receipt.elapsed_seconds < 0:
        raise ContractError("inner receipt contains invalid numeric values")
    for name, value in (("best_epoch", receipt.best_epoch),
                        ("epochs_run", receipt.epochs_run),
                        ("train_patient_count", receipt.train_patient_count),
                        ("validation_patient_count", receipt.validation_patient_count)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ContractError(f"inner receipt {name} must be a positive integer")
    if receipt.best_epoch > receipt.epochs_run or receipt.epochs_run > settings.max_epochs:
        raise ContractError("inner receipt epoch counts exceed the explicit candidate limit")


def _validate_outer_receipt(
    receipt: OuterFoldReceipt,
    *,
    selected_epochs: int,
    expected_train_patient_count: int,
    expected_train_pair_count: int,
) -> None:
    if type(receipt) is not OuterFoldReceipt:
        raise ContractError("outer fitter must return exactly OuterFoldReceipt")
    finite_values = (receipt.response_mean_train_only,
                     receipt.response_scale_train_only,
                     receipt.elapsed_seconds)
    try:
        finite = all(math.isfinite(value) for value in finite_values)
    except TypeError as exc:
        raise ContractError("outer receipt numeric fields are invalid") from exc
    if not finite or receipt.response_scale_train_only <= 0 or receipt.elapsed_seconds < 0:
        raise ContractError("outer receipt contains invalid numeric values")
    if receipt.selected_epochs != selected_epochs:
        raise ContractError("outer fitter receipt does not match the selected epoch count")
    if receipt.train_patient_count != expected_train_patient_count:
        raise ContractError("outer fitter receipt has a mismatched training-patient count")
    if receipt.train_pair_count != expected_train_pair_count:
        raise ContractError("outer fitter receipt has a mismatched training-pair count")


def run_exp046(
    *,
    fold_plan: FoldPlan,
    search_plans_by_arm: Mapping[Exp046Arm, ArmSearchPlan],
    final_seeds: tuple[int, ...],
    model_inputs_provider: Callable[
        [str, Exp046Arm, str | None, int, int | None], ArmModelInputs
    ],
    source_retention_gate: Callable[[SourceRetentionGateRequest], bool],
    source_exposure_gate: Callable[
        [SourceExposureAuditRequest], SourceExposureAuditReceipt
    ],
    execution_budget_gate: Callable[[WorkloadPlan], bool],
    candidate_selector: Callable[[CandidateSelectionPacket], str],
    epoch_selector: Callable[[EpochSelectionPacket], Mapping[int, int]],
    device: torch.device,
) -> Exp046RunResult:
    """Execute the parameterized EXP-046 fold orchestration after full preflight.

    Pending protocol decisions are required callback/provider inputs. P-R maps
    are resolved only from the current config registration. The independent
    exposure callback must audit the actual mask execution log and checkpoint
    lineage. This driver checks receipt shape, bindings and reported zero
    intersections; it cannot prove the external audit was performed truthfully.
    """
    validate_fold_plan(fold_plan)
    registration = _load_exp046_registration()
    candidates, search_seeds = _validate_search_plans(
        search_plans_by_arm, final_seeds, registration)
    for name, callback in (
        ("model_inputs_provider", model_inputs_provider),
        ("source_retention_gate", source_retention_gate),
        ("source_exposure_gate", source_exposure_gate),
        ("execution_budget_gate", execution_budget_gate),
        ("candidate_selector", candidate_selector),
        ("epoch_selector", epoch_selector),
    ):
        if not callable(callback):
            raise ContractError(f"{name} is a required callable")
    if not isinstance(device, torch.device):
        raise ContractError("device must be an explicit torch.device")

    map_count = len(registration.random_maps)
    map_seeds = registration.map_seeds
    outer_fold_keys = tuple(outer.fold_key for outer in fold_plan.outer_folds)
    inner_fold_keys_by_outer = tuple(
        (outer.fold_key, tuple(inner.fold_key for inner in outer.inner_folds))
        for outer in fold_plan.outer_folds
    )
    inner_map_assignments = tuple(
        PRInnerMapAssignment(
            outer_fold_position=outer_position,
            outer_fold_key=outer.fold_key,
            candidate_id=candidate.candidate_id,
            search_seed=search_seed,
            model_seed=search_seed,
            map_seed=map_seeds[(search_position + outer_position) % map_count],
            inner_fold_keys=tuple(inner.fold_key for inner in outer.inner_folds),
        )
        for outer_position, outer in enumerate(fold_plan.outer_folds)
        for candidate in candidates
        for search_position, search_seed in enumerate(search_seeds)
    )
    inner_call_count = (
        len(fold_plan.outer_folds) * len(ARM_ORDER)
        * len(candidates) * len(search_seeds) * 3
    )
    outer_call_count = len(fold_plan.outer_folds) * (
        4 * len(final_seeds) + len(final_seeds) * map_count
    )
    workload = WorkloadPlan(
        outer_fold_count=len(fold_plan.outer_folds),
        inner_fold_count_per_outer=3,
        outer_fold_keys=outer_fold_keys,
        inner_fold_keys_by_outer=inner_fold_keys_by_outer,
        arms=ARM_ORDER,
        candidates=candidates,
        search_seeds=search_seeds,
        final_seeds=final_seeds,
        config_sha256=registration.config_sha256,
        registered_maps=registration.random_maps,
        inner_map_assignments=inner_map_assignments,
        inner_fit_call_count=inner_call_count,
        outer_fit_call_count=outer_call_count,
    )

    # Resolve every exact fit context before any fit starts. Search seed is the
    # inner model seed; map_seed is a separate, config-bound graph opportunity.
    resolved_inputs: dict[
        tuple[str, Exp046Arm, str | None, int, int | None], ArmModelInputs
    ] = {}
    manifest_summaries: dict[
        tuple[str, str | None], tuple[int, str, int, str, int]
    ] = {}

    def resolve_inputs(
        outer: OuterFoldSlice,
        arm: Exp046Arm,
        split_key: str | None,
        model_seed: int,
        map_seed: int | None,
        permitted_train: frozenset[str],
        expected_holdout: frozenset[str],
    ) -> None:
        key = (outer.fold_key, arm, split_key, model_seed, map_seed)
        if key in resolved_inputs:
            raise ContractError("duplicate preflight fit context")
        try:
            inputs = model_inputs_provider(
                outer.fold_key, arm, split_key, model_seed, map_seed)
        except Exception:
            raise ContractError("model_inputs_provider failed for a required fit context") from None
        _validate_model_inputs(
            outer_fold_key=outer.fold_key,
            arm=arm,
            map_seed=map_seed,
            registration=registration,
            inputs=inputs,
            permitted_train_patient_ids=permitted_train,
        )
        summary = _validate_source_holdout_manifest(
            inputs.source_holdout_manifest,
            outer_fold_key=outer.fold_key,
            inner_fold_key=split_key,
            expected_pdo_group_keys=expected_holdout,
        )
        manifest_key = (outer.fold_key, split_key)
        if manifest_key in manifest_summaries:
            prior = manifest_summaries[manifest_key]
            prior_manifest = next(
                existing.source_holdout_manifest
                for resolved_key, existing in resolved_inputs.items()
                if resolved_key[0] == outer.fold_key and resolved_key[2] == split_key
            )
            if (
                prior != summary
                or prior_manifest.manifest_sha256
                != inputs.source_holdout_manifest.manifest_sha256
                or prior_manifest.identity_mapping_sha256
                != inputs.source_holdout_manifest.identity_mapping_sha256
            ):
                raise ContractError("source holdout crosswalk differs across matched fit contexts")
        else:
            manifest_summaries[manifest_key] = summary
        resolved_inputs[key] = inputs

    for outer_position, outer in enumerate(fold_plan.outer_folds):
        for arm in ARM_ORDER:
            for search_position, search_seed in enumerate(search_seeds):
                map_seed = (
                    map_seeds[(search_position + outer_position) % map_count]
                    if arm == "P-R" else None
                )
                for inner in outer.inner_folds:
                    expected_holdout = frozenset(outer.outer_test.patient_ids).union(
                        inner.validation.patient_ids)
                    resolve_inputs(
                        outer, arm, inner.fold_key, search_seed, map_seed,
                        frozenset(inner.train.patient_ids), expected_holdout)
            outer_test_ids = frozenset(outer.outer_test.patient_ids)
            for model_seed in final_seeds:
                for map_seed in (map_seeds if arm == "P-R" else (None,)):
                    resolve_inputs(
                        outer, arm, None, model_seed, map_seed,
                        frozenset(outer.outer_train.patient_ids), outer_test_ids)

    # Check matched source lineage and branch setup across arms and P-R maps.
    for outer in fold_plan.outer_folds:
        split_keys = (*tuple(inner.fold_key for inner in outer.inner_folds), None)
        for split_key in split_keys:
            model_seeds = search_seeds if split_key is not None else final_seeds
            for model_seed in model_seeds:
                values_by_arm: dict[Exp046Arm, tuple[ArmModelInputs, ...]] = {}
                for arm in ARM_ORDER:
                    contexts = tuple(
                        value for key, value in resolved_inputs.items()
                        if key[0] == outer.fold_key and key[1] == arm
                        and key[2] == split_key and key[3] == model_seed
                    )
                    expected_contexts = 3 if arm == "P-R" and split_key is None else 1
                    if len(contexts) != expected_contexts:
                        raise ContractError("preflight did not resolve the full map-seed context")
                    values_by_arm[arm] = contexts
                reference = values_by_arm[ARM_ORDER[0]][0]
                source_signature = (
                    reference.source.state_dict_sha256,
                    reference.source.target_fit_patient_ids,
                    reference.source.source_sha256,
                    reference.source_input_sha256,
                    reference.checkpoint_artifact_sha256,
                    reference.checkpoint_lineage_sha256,
                    reference.mask_execution_log_sha256,
                )
                all_inputs = tuple(
                    value for arm in ARM_ORDER for value in values_by_arm[arm])
                for value in all_inputs:
                    observed_signature = (
                        value.source.state_dict_sha256,
                        value.source.target_fit_patient_ids,
                        value.source.source_sha256,
                        value.source_input_sha256,
                        value.checkpoint_artifact_sha256,
                        value.checkpoint_lineage_sha256,
                        value.mask_execution_log_sha256,
                    )
                    if observed_signature != source_signature:
                        raise ContractError(
                            "matched arms/maps must share source state, declared fit IDs, and bound source lineage")
                branch_values = tuple(
                    value.branch_initialization
                    for arm in ARM_ORDER if arm != "P-A"
                    for value in values_by_arm[arm]
                )
                if len(set(branch_values)) != 1:
                    raise ContractError(
                        "P-C/P-R/P-T/P-N must receive identical explicit branch initialization per seed")
                if any(not _same_pathway_map(
                        values_by_arm["P-T"][0].true_pathway_map,
                        item.true_pathway_map)
                       for item in values_by_arm["P-N"]):
                    raise ContractError("P-T and P-N must use the same true pathway map")

    reference_ordinary_map: AlignedPathwayMap | None = None
    reference_true_map: AlignedPathwayMap | None = None
    random_map_by_seed: dict[int, AlignedPathwayMap] = {}
    for key, inputs in resolved_inputs.items():
        _, arm, _, _, map_seed = key
        if arm == "P-C":
            if reference_ordinary_map is None:
                reference_ordinary_map = inputs.ordinary_control_map
            elif not _same_pathway_map(reference_ordinary_map, inputs.ordinary_control_map):
                raise ContractError("P-C ordinary map must remain fixed across folds and seeds")
        if arm == "P-T":
            if reference_true_map is None:
                reference_true_map = inputs.true_pathway_map
            elif not _same_pathway_map(reference_true_map, inputs.true_pathway_map):
                raise ContractError("P-T true pathway map must remain fixed across folds and seeds")
        if arm == "P-R":
            assert map_seed is not None and inputs.frozen_random_map is not None
            if map_seed in random_map_by_seed:
                if not _same_pathway_map(
                        random_map_by_seed[map_seed], inputs.frozen_random_map):
                    raise ContractError("registered P-R map changed across fold or fit context")
            else:
                random_map_by_seed[map_seed] = inputs.frozen_random_map
    if set(random_map_by_seed) != set(map_seeds):
        raise ContractError("preflight did not resolve all three registered P-R maps")

    if execution_budget_gate(workload) is not True:
        raise ContractError("execution budget gate did not return explicit approval")

    # This separate delegated identity audit covers all resolved outer/inner,
    # final-refit, model-seed and map-seed contexts before any fit is called.
    source_exposure_audits: list[SourceExposureAuditRecord] = []
    for (outer_key, arm, split_key, model_seed, map_seed), inputs in resolved_inputs.items():
        manifest = inputs.source_holdout_manifest
        holdout_summary = manifest_summaries[(outer_key, split_key)]
        pathway_hash = (
            registration.map_sha256_by_seed[map_seed]
            if map_seed is not None else None
        )
        request = SourceExposureAuditRequest(
            experiment_id="EXP-046",
            outer_fold_key=outer_key,
            arm=arm,
            inner_fold_key=split_key,
            search_seed=model_seed if split_key is not None else None,
            model_seed=model_seed,
            map_seed=map_seed,
            pathway_map_sha256=pathway_hash,
            source_input_sha256=inputs.source_input_sha256,
            checkpoint_artifact_sha256=inputs.checkpoint_artifact_sha256,
            checkpoint_lineage_sha256=inputs.checkpoint_lineage_sha256,
            loaded_state_dict_sha256=inputs.source.state_dict_sha256,
            expected_mask_execution_log_sha256=inputs.mask_execution_log_sha256,
            source_input_ref=inputs.source_input_ref,
            checkpoint_artifact_ref=inputs.checkpoint_artifact_ref,
            mask_execution_log_ref=inputs.mask_execution_log_ref,
            manifest=manifest,
            expected_pdo_group_count=holdout_summary[0],
            expected_source_group_sha256=holdout_summary[1],
            expected_source_group_count=holdout_summary[2],
            expected_source_alias_sha256=holdout_summary[3],
            expected_source_alias_count=holdout_summary[4],
        )
        try:
            audit_receipt = source_exposure_gate(request)
        except Exception:
            raise ContractError("source_exposure_gate callback failed for a required fit context") from None
        _validate_source_exposure_receipt(audit_receipt, request)
        source_exposure_audits.append(SourceExposureAuditRecord(
            request=request, receipt=audit_receipt))

    # Source-retention approval remains separate from the exposure audit.
    for (outer_key, arm, split_key, model_seed, map_seed), inputs in resolved_inputs.items():
        request = SourceRetentionGateRequest(
            outer_fold_key=outer_key,
            arm=arm,
            inner_fold_key=split_key,
            model_seed=model_seed,
            map_seed=map_seed,
            model_inputs=inputs,
        )
        try:
            approved = source_retention_gate(request)
        except Exception:
            raise ContractError("source_retention_gate callback failed") from None
        if approved is not True:
            raise ContractError("source retention gate did not return explicit approval")

    arm_fold_receipts: list[ArmFoldReceipt] = []
    outer_seed_receipts: list[OuterSeedReceipt] = []
    prediction_batches: list[KeyedPredictionBatch] = []

    for outer_position, outer in enumerate(fold_plan.outer_folds):
        for arm in ARM_ORDER:
            search_plan = search_plans_by_arm[arm]
            inner_opportunity_receipts: list[InnerOpportunityReceipt] = []
            all_inner_calls: list[InnerCallReceipt] = []
            for candidate in search_plan.candidates:
                for search_position, search_seed in enumerate(search_plan.search_seeds):
                    map_seed = (
                        map_seeds[(search_position + outer_position) % map_count]
                        if arm == "P-R" else None
                    )
                    fold_keys: list[str] = []
                    fold_receipts: list[InnerFoldReceipt] = []
                    for inner in outer.inner_folds:
                        inputs = resolved_inputs[
                            (outer.fold_key, arm, inner.fold_key, search_seed, map_seed)]
                        try:
                            receipt = fit_inner_candidate(
                                inner.train,
                                inner.validation,
                                arm=arm,
                                backbone_factory=inputs.backbone_factory,
                                source=inputs.source,
                                outer_fold_key=outer.fold_key,
                                inner_fold_key=inner.fold_key,
                                model_seed=search_seed,
                                branch_initialization=inputs.branch_initialization,
                                ordinary_control_map=inputs.ordinary_control_map,
                                frozen_random_map=inputs.frozen_random_map,
                                true_pathway_map=inputs.true_pathway_map,
                                settings=candidate.settings,
                                device=device,
                            )
                        except Exception:
                            raise ContractError("inner fit primitive failed") from None
                        _validate_receipt_fields(receipt, candidate.settings)
                        expected_train_patients = len(set(inner.train.patient_ids))
                        expected_valid_patients = len(set(inner.validation.patient_ids))
                        if (receipt.train_patient_count != expected_train_patients
                                or receipt.validation_patient_count != expected_valid_patients):
                            raise ContractError("inner fitter receipt has mismatched patient counts")
                        fold_keys.append(inner.fold_key)
                        fold_receipts.append(receipt)
                        all_inner_calls.append(InnerCallReceipt(
                            candidate_id=candidate.candidate_id,
                            search_seed=search_seed,
                            model_seed=search_seed,
                            map_seed=map_seed,
                            pathway_map_sha256=(
                                registration.map_sha256_by_seed[map_seed]
                                if map_seed is not None else None
                            ),
                            inner_fold_key=inner.fold_key,
                            receipt=receipt,
                        ))
                    inner_opportunity_receipts.append(InnerOpportunityReceipt(
                        candidate_id=candidate.candidate_id,
                        search_seed=search_seed,
                        model_seed=search_seed,
                        map_seed=map_seed,
                        pathway_map_sha256=(
                            registration.map_sha256_by_seed[map_seed]
                            if map_seed is not None else None
                        ),
                        inner_fold_keys=tuple(fold_keys),
                        receipts=tuple(fold_receipts),
                    ))

            map_seed_by_search_seed = tuple(
                (
                    seed,
                    map_seeds[(position + outer_position) % map_count]
                    if arm == "P-R" else None,
                )
                for position, seed in enumerate(search_plan.search_seeds)
            )
            selection_packet = CandidateSelectionPacket(
                arm=arm,
                outer_fold_key=outer.fold_key,
                registered_maps=registration.random_maps,
                map_seed_by_search_seed=map_seed_by_search_seed,
                opportunities=tuple(inner_opportunity_receipts),
            )
            try:
                selected_candidate_id = candidate_selector(selection_packet)
            except Exception:
                raise ContractError("candidate_selector callback failed") from None
            if (not isinstance(selected_candidate_id, str)
                    or selected_candidate_id not in {item.candidate_id for item in candidates}):
                raise ContractError("candidate selector returned an unknown candidate ID")
            selected_candidate = next(
                item for item in candidates if item.candidate_id == selected_candidate_id)
            selected_evidence = tuple(
                item for item in inner_opportunity_receipts
                if item.candidate_id == selected_candidate_id)
            epoch_packet = EpochSelectionPacket(
                arm=arm,
                outer_fold_key=outer.fold_key,
                candidate_id=selected_candidate_id,
                registered_maps=registration.random_maps,
                map_seed_by_search_seed=map_seed_by_search_seed,
                opportunities=selected_evidence,
                final_seeds=final_seeds,
            )
            try:
                selected_epochs = epoch_selector(epoch_packet)
            except Exception:
                raise ContractError("epoch_selector callback failed") from None
            if not isinstance(selected_epochs, Mapping):
                raise ContractError("epoch selector must explicitly map every final seed to an epoch count")
            if set(selected_epochs) != set(final_seeds):
                raise ContractError("epoch selector output must cover exactly the final seed set")
            for seed in final_seeds:
                value = selected_epochs[seed]
                if (isinstance(value, bool) or not isinstance(value, int)
                        or value < 1 or value > selected_candidate.settings.max_epochs):
                    raise ContractError(
                        "epoch selector returned a count outside the selected candidate limit")

            fold_outer_receipts: list[OuterSeedReceipt] = []
            for final_seed in final_seeds:
                selected_epoch_count = selected_epochs[final_seed]
                final_map_seeds = map_seeds if arm == "P-R" else (None,)
                for map_seed in final_map_seeds:
                    outer_inputs = resolved_inputs[
                        (outer.fold_key, arm, None, final_seed, map_seed)]
                    try:
                        keyed, outer_receipt = fit_outer_fold(
                            outer.outer_train,
                            outer.outer_test,
                            arm=arm,
                            backbone_factory=outer_inputs.backbone_factory,
                            source=outer_inputs.source,
                            outer_fold_key=outer.fold_key,
                            model_seed=final_seed,
                            branch_initialization=outer_inputs.branch_initialization,
                            ordinary_control_map=outer_inputs.ordinary_control_map,
                            frozen_random_map=outer_inputs.frozen_random_map,
                            true_pathway_map=outer_inputs.true_pathway_map,
                            settings=selected_candidate.settings,
                            selected_epochs=selected_epoch_count,
                            device=device,
                        )
                    except Exception:
                        raise ContractError("outer fit primitive failed") from None
                    _validate_outer_receipt(
                        outer_receipt,
                        selected_epochs=selected_epoch_count,
                        expected_train_patient_count=len(set(outer.outer_train.patient_ids)),
                        expected_train_pair_count=len(outer.outer_train.patient_ids),
                    )
                    if type(keyed) is not OuterKeyedPredictions:
                        raise ContractError(
                            "outer fitter must return exactly label-free OuterKeyedPredictions")
                    try:
                        keyed.validate()
                    except (TypeError, ValueError, AttributeError):
                        raise ContractError("outer fitter returned invalid keyed predictions") from None
                    expected_keys = tuple(zip(outer.outer_test.patient_ids,
                                              outer.outer_test.drug_ids))
                    observed_keys = tuple(zip(keyed.patient_ids, keyed.drug_ids))
                    if observed_keys != expected_keys:
                        raise ContractError(
                            "outer fitter predictions must preserve exact unlabeled input keys and order")

                    seed_receipt = OuterSeedReceipt(
                        arm=arm,
                        outer_fold_key=outer.fold_key,
                        model_seed=final_seed,
                        map_seed=map_seed,
                        pathway_map_sha256=(
                            registration.map_sha256_by_seed[map_seed]
                            if map_seed is not None else None
                        ),
                        candidate_id=selected_candidate_id,
                        selected_epochs=selected_epoch_count,
                        receipt=outer_receipt,
                    )
                    fold_outer_receipts.append(seed_receipt)
                    outer_seed_receipts.append(seed_receipt)
                    prediction_batches.append(KeyedPredictionBatch(
                        arm=arm,
                        outer_fold_key=outer.fold_key,
                        candidate_id=selected_candidate_id,
                        model_seed=final_seed,
                        map_seed=map_seed,
                        pathway_map_sha256=(
                            registration.map_sha256_by_seed[map_seed]
                            if map_seed is not None else None
                        ),
                        predictions=keyed,
                    ))

            inner_elapsed = sum(call.receipt.elapsed_seconds for call in all_inner_calls)
            outer_elapsed = sum(item.receipt.elapsed_seconds for item in fold_outer_receipts)
            arm_fold_receipts.append(ArmFoldReceipt(
                arm=arm,
                outer_fold_key=outer.fold_key,
                candidate_opportunities=tuple(inner_opportunity_receipts),
                selected_candidate_id=selected_candidate_id,
                selected_epochs_by_seed=tuple(
                    (seed, selected_epochs[seed]) for seed in final_seeds),
                inner_calls=tuple(all_inner_calls),
                outer_seed_receipts=tuple(fold_outer_receipts),
                outer_train_patient_count=len(set(outer.outer_train.patient_ids)),
                outer_test_patient_count=len(set(outer.outer_test.patient_ids)),
                inner_fit_api_elapsed_seconds_sum=inner_elapsed,
                outer_fit_api_elapsed_seconds_sum=outer_elapsed,
                reported_fit_api_elapsed_seconds_sum=inner_elapsed + outer_elapsed,
                cost_scope=_COST_SCOPE,
            ))

    result = Exp046RunResult(
        workload=workload,
        arm_fold_receipts=tuple(arm_fold_receipts),
        predictions=tuple(prediction_batches),
        outer_seed_receipts=tuple(outer_seed_receipts),
        source_exposure_audits=tuple(source_exposure_audits),
        reported_fit_api_elapsed_seconds_sum=sum(
            receipt.reported_fit_api_elapsed_seconds_sum
            for receipt in arm_fold_receipts),
        cost_scope=_COST_SCOPE,
    )
    result.validate()
    return result
