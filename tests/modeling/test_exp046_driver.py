"""Synthetic in-memory checks for the EXP-046 fold-plan contract."""

from dataclasses import dataclass, replace
import hashlib

import numpy as np
import pytest
import torch

import scripts.modeling.exp046_driver as driver
from scripts.modeling.exp046_driver import (
    ARM_ORDER,
    ArmModelInputs,
    ArmSearchPlan,
    CandidateDefinition,
    CandidateSelectionPacket,
    ContractError,
    EpochSelectionPacket,
    FoldPlan,
    InnerFoldSlice,
    InnerOpportunityReceipt,
    LabeledPairBatch,
    OuterFoldSlice,
    PairInputBatch,
    SourceRetentionGateRequest,
    validate_fold_plan,
)
from drug_screen.models.exp046_five_arm import (
    AlignedPathwayMap,
    PathwayAxisContract,
)
from scripts.modeling.exp046_pathway_fold import (
    BranchInitialization,
    CandidateTraining,
    InnerFoldReceipt,
    OuterFoldReceipt,
    OuterKeyedPredictions,
    SourceFoldInitialization,
    compute_state_dict_sha256,
)


@dataclass(frozen=True)
class LabelBearingPairInputBatch(PairInputBatch):
    responses: torch.Tensor


def _labeled_batch(patient_ids: tuple[str, ...]) -> LabeledPairBatch:
    # Tiny synthetic CPU tensors make the typed batch objects well formed.
    expression = torch.zeros((len(patient_ids), 1), dtype=torch.float32)
    drug_features = torch.zeros((len(patient_ids), 1), dtype=torch.float32)
    return LabeledPairBatch(
        expression=expression,
        drug_features=drug_features,
        patient_ids=patient_ids,
        drug_ids=tuple(f"drug-{index}" for index in range(len(patient_ids))),
        responses=torch.zeros((len(patient_ids),), dtype=torch.float32),
    )


def _input_batch(patient_ids: tuple[str, ...]) -> PairInputBatch:
    expression = torch.zeros((len(patient_ids), 1), dtype=torch.float32)
    drug_features = torch.zeros((len(patient_ids), 1), dtype=torch.float32)
    return PairInputBatch(
        expression=expression,
        drug_features=drug_features,
        patient_ids=patient_ids,
        drug_ids=tuple(f"drug-{index}" for index in range(len(patient_ids))),
    )


def _outer_fold(
    patients: tuple[str, ...], outer_index: int, heldout_patient: str
) -> OuterFoldSlice:
    outer_train_ids = tuple(
        patient for patient in patients if patient != heldout_patient)
    validation_parts = (
        outer_train_ids[:2],
        outer_train_ids[2:3],
        outer_train_ids[3:],
    )
    inner_folds = tuple(
        InnerFoldSlice(
            fold_key=f"inner-{inner_index}",
            train=_labeled_batch(tuple(
                patient for patient in outer_train_ids
                if patient not in validation_ids)),
            validation=_labeled_batch(validation_ids),
        )
        for inner_index, validation_ids in enumerate(validation_parts)
    )
    return OuterFoldSlice(
        fold_key=f"outer-{outer_index}",
        outer_train=_labeled_batch(outer_train_ids),
        outer_test=_input_batch((heldout_patient,)),
        inner_folds=inner_folds,
    )


def _plan_for_cohort(patients: tuple[str, ...]) -> FoldPlan:
    outer_folds = tuple(
        _outer_fold(patients, index, heldout)
        for index, heldout in enumerate(patients[:5]))
    return FoldPlan(frozenset(patients), outer_folds)


def _valid_plan() -> FoldPlan:
    patients = tuple(f"patient-{index}" for index in range(5))
    return _plan_for_cohort(patients)


def _with_outer_fold(plan: FoldPlan, index: int, outer: OuterFoldSlice) -> FoldPlan:
    folds = list(plan.outer_folds)
    folds[index] = outer
    return FoldPlan(plan.cohort_patient_ids, tuple(folds))


def test_accepts_five_outer_by_three_inner_patient_fold_plan():
    assert validate_fold_plan(_valid_plan()) is None


def test_requires_exactly_five_outer_folds():
    plan = _valid_plan()
    with pytest.raises(ContractError, match="exactly five outer folds"):
        validate_fold_plan(FoldPlan(plan.cohort_patient_ids, plan.outer_folds[:-1]))


def test_outer_test_folds_must_partition_the_declared_patient_cohort():
    plan = _plan_for_cohort(tuple(f"patient-{index}" for index in range(6)))
    with pytest.raises(ContractError, match="outer-test folds must partition"):
        validate_fold_plan(plan)


def test_rejects_overlapping_outer_test_folds_without_exposing_patient_ids():
    plan = _valid_plan()
    overlapping = _outer_fold(
        tuple(sorted(plan.cohort_patient_ids)), 1, "patient-0")
    malformed = _with_outer_fold(plan, 1, overlapping)
    with pytest.raises(ContractError, match="outer-test folds overlap") as error:
        validate_fold_plan(malformed)
    assert "patient-0" not in str(error.value)


def test_requires_exactly_three_inner_folds_per_outer_fold():
    plan = _valid_plan()
    first = plan.outer_folds[0]
    with pytest.raises(ContractError, match="exactly three inner folds"):
        validate_fold_plan(_with_outer_fold(
            plan, 0, replace(first, inner_folds=first.inner_folds[:2])))


def test_rejects_outer_train_test_patient_overlap():
    plan = _valid_plan()
    first = plan.outer_folds[0]
    overlapping_test = _input_batch((first.outer_train.patient_ids[0],))
    with pytest.raises(ContractError, match="outer train/test patients overlap"):
        validate_fold_plan(_with_outer_fold(
            plan, 0, replace(first, outer_test=overlapping_test)))


def test_rejects_outer_training_split_that_is_not_cohort_complement():
    plan = _valid_plan()
    first = plan.outer_folds[0]
    incomplete_train = _labeled_batch(first.outer_train.patient_ids[:-1])
    with pytest.raises(ContractError, match="cohort complement"):
        validate_fold_plan(_with_outer_fold(
            plan, 0, replace(first, outer_train=incomplete_train)))


def test_rejects_duplicate_outer_fold_key():
    plan = _valid_plan()
    first, second = plan.outer_folds[:2]
    duplicated = replace(second, fold_key=first.fold_key)
    with pytest.raises(ContractError, match="duplicate outer fold key"):
        validate_fold_plan(_with_outer_fold(plan, 1, duplicated))


def test_rejects_duplicate_inner_fold_key():
    plan = _valid_plan()
    first = plan.outer_folds[0]
    inner0, inner1 = first.inner_folds[:2]
    duplicated = replace(inner1, fold_key=inner0.fold_key)
    updated_outer = replace(
        first, inner_folds=(inner0, duplicated, *first.inner_folds[2:]))
    with pytest.raises(ContractError, match="duplicate inner fold key"):
        validate_fold_plan(_with_outer_fold(plan, 0, updated_outer))


def test_rejects_outer_patient_not_in_declared_cohort():
    plan = _valid_plan()
    first = plan.outer_folds[0]
    unknown_patient = _input_batch(("patient-outside-cohort",))
    with pytest.raises(ContractError, match="outside the declared cohort"):
        validate_fold_plan(_with_outer_fold(
            plan, 0, replace(first, outer_test=unknown_patient)))


def test_rejects_invalid_patient_identity_without_echoing_value():
    plan = _valid_plan()
    first = plan.outer_folds[0]
    malformed_test = _input_batch((" patient-0 ",))
    with pytest.raises(ContractError, match="trimmed identities") as error:
        validate_fold_plan(_with_outer_fold(
            plan, 0, replace(first, outer_test=malformed_test)))
    assert "patient-0" not in str(error.value)


def test_rejects_label_bearing_outer_test_batch():
    plan = _valid_plan()
    first = plan.outer_folds[0]
    labeled_outer_test = _labeled_batch(first.outer_test.patient_ids)
    with pytest.raises(ContractError, match="exactly PairInputBatch"):
        validate_fold_plan(_with_outer_fold(
            plan, 0, replace(first, outer_test=labeled_outer_test)))


def test_rejects_label_bearing_outer_test_subclass():
    plan = _valid_plan()
    first = plan.outer_folds[0]
    base = _input_batch(first.outer_test.patient_ids)
    labeled_subclass = LabelBearingPairInputBatch(
        expression=base.expression,
        drug_features=base.drug_features,
        patient_ids=base.patient_ids,
        drug_ids=base.drug_ids,
        responses=torch.zeros((len(base.patient_ids),), dtype=torch.float32),
    )
    with pytest.raises(ContractError, match="exactly PairInputBatch"):
        validate_fold_plan(_with_outer_fold(
            plan, 0, replace(first, outer_test=labeled_subclass)))


def test_rejects_inner_train_validation_patient_overlap():
    plan = _valid_plan()
    first = plan.outer_folds[0]
    inner = first.inner_folds[0]
    overlapping_train = _labeled_batch(
        inner.train.patient_ids + (inner.validation.patient_ids[0],))
    updated_inner = replace(inner, train=overlapping_train)
    updated_outer = replace(
        first, inner_folds=(updated_inner, *first.inner_folds[1:]))
    with pytest.raises(ContractError, match="inner train/validation patients overlap") as error:
        validate_fold_plan(_with_outer_fold(plan, 0, updated_outer))
    assert "patient-" not in str(error.value)


def test_rejects_inner_training_split_that_is_not_outer_training_complement():
    plan = _valid_plan()
    first = plan.outer_folds[0]
    inner = first.inner_folds[0]
    incomplete_train = _labeled_batch(inner.train.patient_ids[:-1])
    updated_inner = replace(inner, train=incomplete_train)
    updated_outer = replace(
        first, inner_folds=(updated_inner, *first.inner_folds[1:]))
    with pytest.raises(ContractError, match="outer-training complement"):
        validate_fold_plan(_with_outer_fold(plan, 0, updated_outer))


def test_rejects_inner_validation_patient_outside_outer_train():
    plan = _valid_plan()
    first = plan.outer_folds[0]
    inner = first.inner_folds[0]
    outside_validation = _labeled_batch(first.outer_test.patient_ids)
    updated_inner = replace(inner, validation=outside_validation)
    updated_outer = replace(
        first, inner_folds=(updated_inner, *first.inner_folds[1:]))
    with pytest.raises(
        ContractError, match="inner validation patients fall outside outer training"
    ):
        validate_fold_plan(_with_outer_fold(plan, 0, updated_outer))


def test_requires_inner_validation_sets_to_cover_outer_train_exactly_once():
    plan = _valid_plan()
    first = plan.outer_folds[0]
    inner0, inner1, inner2 = first.inner_folds
    patient0, patient1, patient2, patient3 = first.outer_train.patient_ids

    # Each inner training set remains its matching complement, but patient2
    # appears in two validation folds. The three folds must partition patients.
    inner0 = replace(inner0, validation=_labeled_batch((patient0, patient2)))
    inner0 = replace(inner0, train=_labeled_batch((patient1, patient3)))
    inner1 = replace(inner1, validation=_labeled_batch((patient1, patient2)))
    inner1 = replace(inner1, train=_labeled_batch((patient0, patient3)))
    inner2 = replace(inner2, validation=_labeled_batch((patient3,)))
    inner2 = replace(inner2, train=_labeled_batch((patient0, patient1, patient2)))
    updated_outer = replace(first, inner_folds=(inner0, inner1, inner2))

    with pytest.raises(ContractError, match="inner validation folds overlap"):
        validate_fold_plan(_with_outer_fold(plan, 0, updated_outer))


def test_rejects_inner_validation_sets_that_do_not_cover_outer_train():
    plan = _valid_plan()
    first = plan.outer_folds[0]
    inner0, inner1, inner2 = first.inner_folds
    patient0, patient1, patient2, patient3 = first.outer_train.patient_ids
    inner0 = replace(inner0, train=_labeled_batch((patient1, patient2, patient3)))
    inner0 = replace(inner0, validation=_labeled_batch((patient0,)))
    inner1 = replace(inner1, train=_labeled_batch((patient0, patient2, patient3)))
    inner1 = replace(inner1, validation=_labeled_batch((patient1,)))
    inner2 = replace(inner2, train=_labeled_batch((patient0, patient1, patient3)))
    inner2 = replace(inner2, validation=_labeled_batch((patient2,)))
    updated_outer = replace(first, inner_folds=(inner0, inner1, inner2))

    with pytest.raises(ContractError, match="must cover outer training patients"):
        validate_fold_plan(_with_outer_fold(plan, 0, updated_outer))


def _candidate_settings(max_epochs: int = 4) -> CandidateTraining:
    return CandidateTraining(
        learning_rate=0.001,
        weight_decay=0.0,
        ranking_weight=0.5,
        max_epochs=max_epochs,
        patience=2,
        batch_size=8,
        min_improvement=0.0,
    )


def _search_plans() -> dict[str, ArmSearchPlan]:
    candidates = (
        CandidateDefinition("candidate-a", _candidate_settings()),
        CandidateDefinition("candidate-b", _candidate_settings(max_epochs=3)),
    )
    return {
        arm: ArmSearchPlan(candidates=candidates, search_seeds=(101, 202, 303))
        for arm in ARM_ORDER
    }


def _tiny_pathway_map(index: int = 0) -> AlignedPathwayMap:
    matrices = (
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
        [[1, 1], [0, 1]],
    )
    contract = PathwayAxisContract(
        full_gene_count=2, mapped_gene_count=2, pathway_count=2)
    return AlignedPathwayMap(
        gene_axis=("gene-a", "gene-b"),
        mapped_gene_ids=("gene-a", "gene-b"),
        pathway_names=("slot-0", "slot-1"),
        incidence_matrix=torch.tensor(matrices[index], dtype=torch.uint8),
        contract=contract,
    )


def _synthetic_registration() -> driver.Exp046Registration:
    random_maps = tuple(_tiny_pathway_map(index) for index in range(3))
    return driver.Exp046Registration(
        config_sha256="f" * 64,
        final_model_seeds=(7, 8, 9),
        random_maps=tuple(
            driver.RegisteredRandomMap(
                map_seed=seed,
                sha256=driver._pathway_map_sha256(pathway_map),
            )
            for seed, pathway_map in zip((11, 12, 13), random_maps)
        ),
    )


def _manifest_for(
    outer_fold_key: str,
    inner_fold_key: str | None,
    pdo_group_keys: frozenset[str],
) -> driver.SourceHoldoutIdentityManifest:
    mappings = tuple(
        driver.SourceIdentityGroupMapping(
            pdo_patient_group_key=patient,
            source_entity_group_ids=(
                f"source-entity-{hashlib.sha256(patient.encode()).hexdigest()[:12]}",
            ),
            source_alias_ids=(
                f"source-alias-{hashlib.sha256(patient.encode()).hexdigest()[:12]}",
            ),
        )
        for patient in sorted(pdo_group_keys)
    )
    provisional = driver.SourceHoldoutIdentityManifest(
        outer_fold_key=outer_fold_key,
        inner_fold_key=inner_fold_key,
        manifest_version="synthetic-holdout-v1",
        manifest_sha256="0" * 64,
        identity_mapping_key="synthetic-pdo-source-crosswalk",
        identity_mapping_version="synthetic-crosswalk-v1",
        identity_mapping_sha256="a" * 64,
        mappings=mappings,
    )
    return replace(
        provisional,
        manifest_sha256=driver._source_manifest_content_sha256(provisional),
    )


def _source_exposure_receipt(
    request: driver.SourceExposureAuditRequest,
    *,
    source_group_intersection_count: int = 0,
    alias_intersection_count: int = 0,
    mask_log_sha256: str | None = None,
) -> driver.SourceExposureAuditReceipt:
    empty_set_sha = driver._identity_set_sha256(set())
    manifest = request.manifest
    return driver.SourceExposureAuditReceipt(
        experiment_id=request.experiment_id,
        outer_fold_key=request.outer_fold_key,
        arm=request.arm,
        inner_fold_key=request.inner_fold_key,
        search_seed=request.search_seed,
        model_seed=request.model_seed,
        map_seed=request.map_seed,
        pathway_map_sha256=request.pathway_map_sha256,
        source_input_sha256=request.source_input_sha256,
        checkpoint_artifact_sha256=request.checkpoint_artifact_sha256,
        checkpoint_lineage_sha256=request.checkpoint_lineage_sha256,
        loaded_state_dict_sha256=request.loaded_state_dict_sha256,
        mask_execution_log_sha256=(
            mask_log_sha256 or request.expected_mask_execution_log_sha256),
        manifest_version=manifest.manifest_version,
        manifest_sha256=manifest.manifest_sha256,
        identity_mapping_key=manifest.identity_mapping_key,
        identity_mapping_version=manifest.identity_mapping_version,
        identity_mapping_sha256=manifest.identity_mapping_sha256,
        identity_set_hash_scheme=driver.IDENTITY_SET_HASH_SCHEME,
        heldout_pdo_group_count=request.expected_pdo_group_count,
        heldout_source_group_sha256=request.expected_source_group_sha256,
        heldout_source_group_count=request.expected_source_group_count,
        heldout_source_alias_sha256=request.expected_source_alias_sha256,
        heldout_source_alias_count=request.expected_source_alias_count,
        actual_exposed_source_group_sha256=empty_set_sha,
        actual_exposed_source_group_count=0,
        actual_exposed_source_alias_sha256=empty_set_sha,
        actual_exposed_source_alias_count=0,
        source_group_intersection_count=source_group_intersection_count,
        source_alias_intersection_count=alias_intersection_count,
    )


def _model_inputs_provider(plan: FoldPlan, events: list[tuple]):
    state_dict = {"weight": torch.tensor([0.25], dtype=torch.float32)}
    state_hash = compute_state_dict_sha256(state_dict)
    pathway_map = _tiny_pathway_map(0)
    random_maps = {
        seed: _tiny_pathway_map(index)
        for index, seed in enumerate((11, 12, 13))
    }
    branch = BranchInitialization(rank=1, branch_seed=303, zero_initialize=True)
    outer_by_key = {outer.fold_key: outer for outer in plan.outer_folds}

    def provide(outer_fold_key: str, arm: str, inner_fold_key: str | None,
                model_seed: int, map_seed: int | None) -> ArmModelInputs:
        events.append(("provide", outer_fold_key, arm, inner_fold_key,
                       model_seed, map_seed))
        outer = outer_by_key[outer_fold_key]
        if inner_fold_key is None:
            heldout = set(outer.outer_test.patient_ids)
            permitted_train = set(outer.outer_train.patient_ids)
        else:
            inner = next(
                item for item in outer.inner_folds if item.fold_key == inner_fold_key)
            heldout = set(outer.outer_test.patient_ids).union(
                inner.validation.patient_ids)
            permitted_train = set(inner.train.patient_ids)
        assert not heldout.intersection(permitted_train)
        manifest = _manifest_for(
            outer_fold_key, inner_fold_key, frozenset(heldout))
        source = SourceFoldInitialization(
            outer_fold_key=outer_fold_key,
            state_dict=state_dict,
            source_sha256="b" * 64,
            state_dict_sha256=state_hash,
            target_fit_patient_ids=(),
        )
        return ArmModelInputs(
            backbone_factory=lambda: torch.nn.Linear(1, 1),
            source=source,
            branch_initialization=None if arm == "P-A" else branch,
            ordinary_control_map=pathway_map if arm == "P-C" else None,
            frozen_random_map=(
                random_maps[map_seed] if arm == "P-R" and map_seed is not None
                else None
            ),
            true_pathway_map=pathway_map if arm in {"P-T", "P-N"} else None,
            source_input_sha256="c" * 64,
            checkpoint_artifact_sha256="d" * 64,
            checkpoint_lineage_sha256="e" * 64,
            mask_execution_log_sha256="9" * 64,
            source_input_ref="synthetic://source-input",
            checkpoint_artifact_ref="synthetic://checkpoint",
            mask_execution_log_ref="synthetic://mask-log",
            source_holdout_manifest=manifest,
        )

    return provide


def _run_test_driver(monkeypatch, *,
                     candidate_selector=None,
                     epoch_selector=None,
                     source_retention_gate=None,
                     source_exposure_gate=None,
                     execution_budget_gate=None,
                     search_plans_by_arm=None,
                     final_seeds=(7, 8, 9),
                     model_inputs_provider_override=None,
                     fit_counter=None,
                     duplicate_outer_predictions=False,
                     event_log=None):
    plan = _valid_plan()
    events = event_log if event_log is not None else []
    registration = _synthetic_registration()
    monkeypatch.setattr(driver, "_load_exp046_registration", lambda: registration)
    outer_batch_ids = {
        outer.fold_key: outer.outer_test for outer in plan.outer_folds
    }
    map_sha_by_seed = registration.map_sha256_by_seed

    def fake_inner(train, validation, **kwargs):
        if fit_counter is not None:
            fit_counter.append("inner")
        random_map = kwargs["frozen_random_map"]
        map_sha = (
            driver._pathway_map_sha256(random_map)
            if random_map is not None else None
        )
        events.append(("inner", kwargs["outer_fold_key"], kwargs["arm"],
                       kwargs["inner_fold_key"], kwargs["model_seed"], map_sha))
        return InnerFoldReceipt(
            best_validation_score=0.25,
            best_epoch=2,
            epochs_run=3,
            response_mean_train_only=0.0,
            response_scale_train_only=1.0,
            train_patient_count=len(set(train.patient_ids)),
            validation_patient_count=len(set(validation.patient_ids)),
            elapsed_seconds=0.125,
        )

    def fake_outer(train, outer_inputs, **kwargs):
        if fit_counter is not None:
            fit_counter.append("outer")
        random_map = kwargs["frozen_random_map"]
        map_sha = (
            driver._pathway_map_sha256(random_map)
            if random_map is not None else None
        )
        events.append(("outer", kwargs["outer_fold_key"], kwargs["arm"],
                       kwargs["model_seed"], map_sha, kwargs["selected_epochs"]))
        assert type(outer_inputs) is PairInputBatch
        assert "responses" not in vars(outer_inputs)
        assert outer_inputs is outer_batch_ids[kwargs["outer_fold_key"]]
        if duplicate_outer_predictions:
            patients = (outer_inputs.patient_ids[0], outer_inputs.patient_ids[0])
            drugs = (outer_inputs.drug_ids[0], outer_inputs.drug_ids[0])
            predictions = np.asarray([0.1, 0.2], dtype=np.float64)
        else:
            patients = outer_inputs.patient_ids
            drugs = outer_inputs.drug_ids
            predictions = np.arange(len(patients), dtype=np.float64)
        return (
            OuterKeyedPredictions(patients, drugs, predictions),
            OuterFoldReceipt(
                selected_epochs=kwargs["selected_epochs"],
                response_mean_train_only=0.0,
                response_scale_train_only=1.0,
                train_patient_count=len(set(train.patient_ids)),
                train_pair_count=len(train.patient_ids),
                elapsed_seconds=0.25,
            ),
        )

    monkeypatch.setattr(driver, "fit_inner_candidate", fake_inner)
    monkeypatch.setattr(driver, "fit_outer_fold", fake_outer)

    def retention_gate(request: SourceRetentionGateRequest) -> bool:
        assert type(request) is SourceRetentionGateRequest
        assert not hasattr(request, "outer_test")
        events.append(("source-gate", request.outer_fold_key, request.arm,
                       request.inner_fold_key, request.model_seed, request.map_seed))
        return True

    def exposure_gate(request: driver.SourceExposureAuditRequest):
        assert type(request) is driver.SourceExposureAuditRequest
        assert not hasattr(request, "outer_test")
        assert "patient-" not in repr(request)
        assert len(request.loaded_state_dict_sha256) == 64
        assert request.map_seed is None or request.pathway_map_sha256 == map_sha_by_seed[
            request.map_seed]
        events.append(("exposure-gate", request.outer_fold_key, request.arm,
                       request.inner_fold_key, request.model_seed, request.map_seed))
        return _source_exposure_receipt(request)

    def budget_gate(workload):
        events.append(("budget-gate", workload.inner_fit_call_count,
                       workload.outer_fit_call_count))
        return True

    def select_candidate(packet: CandidateSelectionPacket) -> str:
        assert type(packet) is CandidateSelectionPacket
        assert not hasattr(packet, "outer_test")
        assert not hasattr(packet, "outer_train")
        assert packet.opportunities
        assert all(type(item) is InnerOpportunityReceipt
                   for item in packet.opportunities)
        assert all(item.model_seed == item.search_seed for item in packet.opportunities)
        assert tuple(item.map_seed for item in packet.registered_maps) == (11, 12, 13)
        expected_maps = tuple(
            (seed, packet.registered_maps[position].map_seed
             if packet.arm == "P-R" else None)
            for position, seed in enumerate((101, 202, 303))
        )
        if packet.outer_fold_key == "outer-0":
            assert packet.map_seed_by_search_seed == expected_maps
        assert all(type(receipt) is InnerFoldReceipt
                   for item in packet.opportunities for receipt in item.receipts)
        events.append(("select-candidate", packet.outer_fold_key, packet.arm))
        return "candidate-b"

    def select_epochs(packet: EpochSelectionPacket) -> dict[int, int]:
        assert type(packet) is EpochSelectionPacket
        assert packet.candidate_id in {"candidate-a", "candidate-b"}
        assert not hasattr(packet, "outer_test")
        assert not hasattr(packet, "outer_train")
        assert packet.opportunities
        assert all(type(item) is InnerOpportunityReceipt
                   for item in packet.opportunities)
        events.append(("select-epochs", packet.outer_fold_key, packet.arm))
        return {seed: 2 for seed in packet.final_seeds}

    provider = model_inputs_provider_override or _model_inputs_provider(plan, events)
    result = driver.run_exp046(
        fold_plan=plan,
        search_plans_by_arm=search_plans_by_arm or _search_plans(),
        final_seeds=final_seeds,
        model_inputs_provider=provider,
        source_retention_gate=source_retention_gate or retention_gate,
        source_exposure_gate=source_exposure_gate or exposure_gate,
        execution_budget_gate=execution_budget_gate or budget_gate,
        candidate_selector=candidate_selector or select_candidate,
        epoch_selector=epoch_selector or select_epochs,
        device=torch.device("cpu"),
    )
    return result, events


def test_registered_random_map_config_and_map_hashes_are_read_from_current_config():
    registration = driver._load_exp046_registration()
    assert registration.final_model_seeds == (20260922, 20260923, 20260924)
    assert registration.map_seeds == (20260922, 20260923, 20260924)
    assert all(len(item.sha256) == 64 for item in registration.random_maps)
    assert len(set(item.sha256 for item in registration.random_maps)) == 3
    assert len(registration.config_sha256) == 64


@pytest.mark.parametrize(
    ("cohort_field", "wrong_count"),
    [("outer_patient_folds", 4), ("inner_patient_folds", 2)],
)
def test_registration_rejects_config_with_wrong_outer_or_inner_fold_count(
    monkeypatch, cohort_field, wrong_count
):
    config_path = (
        driver.Path(driver.__file__).resolve().parents[2]
        / "configs" / "experiments" / "EXP-046.yaml"
    )
    config = driver.yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["cohort"][cohort_field] = wrong_count
    monkeypatch.setattr(driver.yaml, "safe_load", lambda _raw: config)

    with pytest.raises(ContractError, match="five outer and three inner"):
        driver._load_exp046_registration()


def test_pathway_map_hash_matches_registered_edge_digest_implementation():
    from drug_screen.transfer.pathway_controls import _edge_digest

    for map_index in range(3):
        pathway_map = _tiny_pathway_map(map_index)
        matrix = pathway_map.incidence_matrix.detach().cpu().numpy()
        edges = [
            (pathway_map.gene_axis[gene_index],
             pathway_map.pathway_names[pathway_index])
            for pathway_index, gene_index in np.argwhere(matrix != 0)
        ]
        assert driver._pathway_map_sha256(pathway_map) == _edge_digest(edges)


def test_orchestrates_fair_inner_rotation_and_nine_final_pr_maps_per_seed(monkeypatch):
    result, events = _run_test_driver(monkeypatch)

    assert "patient-" not in repr(result)
    assert "patient-" not in repr(_valid_plan())
    assert result.workload.inner_fit_call_count == 5 * 5 * 2 * 3 * 3
    assert result.workload.outer_fit_call_count == 5 * (4 * 3 + 9)
    assert len(result.arm_fold_receipts) == 25
    assert len(result.outer_seed_receipts) == 105
    assert len(result.predictions) == 105
    assert len(result.source_exposure_audits) == 330
    assert "patient-" not in repr(result.source_exposure_audits[0])
    assert tuple((item.outer_fold_key, item.arm)
                 for item in result.arm_fold_receipts[:5]) == tuple(
                     ("outer-0", arm) for arm in ARM_ORDER)

    assignment_counts = {}
    for item in result.workload.inner_map_assignments:
        assignment_counts.setdefault(
            (item.outer_fold_key, item.candidate_id), {}).setdefault(
                item.map_seed, 0)
        assignment_counts[(item.outer_fold_key, item.candidate_id)][item.map_seed] += 1
    assert all(tuple(counts[seed] for seed in (11, 12, 13)) == (1, 1, 1)
               for counts in assignment_counts.values())
    assert [
        (item.search_seed, item.map_seed)
        for item in result.arm_fold_receipts[2].candidate_opportunities[:3]
    ] == [(101, 11), (202, 12), (303, 13)]
    assert [
        (item.search_seed, item.map_seed)
        for item in result.arm_fold_receipts[7].candidate_opportunities[:3]
    ] == [(101, 12), (202, 13), (303, 11)]

    p_r_outer = [item for item in result.outer_seed_receipts
                 if item.arm == "P-R" and item.outer_fold_key == "outer-0"]
    assert len(p_r_outer) == 9
    assert {
        (item.model_seed, item.map_seed) for item in p_r_outer
    } == {(seed, map_seed) for seed in (7, 8, 9) for map_seed in (11, 12, 13)}
    assert all(
        item.pathway_map_sha256
        == result.workload.registered_maps[
            (11, 12, 13).index(item.map_seed)
        ].sha256
        for item in p_r_outer
    )
    assert all(
        len([item for item in result.outer_seed_receipts
             if item.arm == arm and item.outer_fold_key == "outer-0"]) == 3
        for arm in ("P-A", "P-C", "P-T", "P-N")
    )

    fit_positions = [i for i, event in enumerate(events)
                     if event[0] in {"inner", "outer"}]
    gate_positions = [i for i, event in enumerate(events)
                      if event[0] in {"budget-gate", "exposure-gate", "source-gate"}]
    assert gate_positions and max(gate_positions) < min(fit_positions)

    first_arm_calls = [event for event in events
                       if event[0] in {"inner", "outer"}
                       and event[1] == "outer-0" and event[2] == "P-A"]
    assert first_arm_calls[:3] == [
        ("inner", "outer-0", "P-A", "inner-0", 101, None),
        ("inner", "outer-0", "P-A", "inner-1", 101, None),
        ("inner", "outer-0", "P-A", "inner-2", 101, None),
    ]
    assert [item[:5] for item in first_arm_calls[-3:]] == [
        ("outer", "outer-0", "P-A", 7, None),
        ("outer", "outer-0", "P-A", 8, None),
        ("outer", "outer-0", "P-A", 9, None),
    ]

    first_pa = result.arm_fold_receipts[0]
    first_pr = result.arm_fold_receipts[2]
    assert first_pa.inner_fit_api_elapsed_seconds_sum == 2.25
    assert first_pa.outer_fit_api_elapsed_seconds_sum == 0.75
    assert first_pa.reported_fit_api_elapsed_seconds_sum == 3.0
    assert first_pr.outer_fit_api_elapsed_seconds_sum == 2.25
    assert "elapsed_seconds" in first_pa.cost_scope
    assert "not a full experiment cost" in first_pa.cost_scope
    assert result.reported_fit_api_elapsed_seconds_sum == 82.5

    unique_prediction_keys = {
        (item.arm, item.outer_fold_key, item.model_seed, item.map_seed,
         patient_id, drug_id)
        for item in result.predictions
        for patient_id, drug_id in zip(item.predictions.patient_ids,
                                       item.predictions.drug_ids)
    }
    assert len(unique_prediction_keys) == sum(
        len(item.predictions.patient_ids) for item in result.predictions)


@pytest.mark.parametrize("mutation, message", [
    ("missing_arm", "exactly the five EXP-046 arms"),
    ("duplicate_candidate", "duplicate candidate ID"),
    ("different_seed_order", "search seed opportunities and order"),
    ("different_candidate_opportunity", "candidate IDs, values, and order"),
    ("search_seed_count_not_balanced", "multiple of the registered map count"),
])
def test_rejects_missing_duplicate_or_mismatched_search_opportunities(
        monkeypatch, mutation, message):
    monkeypatch.setattr(
        driver, "_load_exp046_registration", lambda: _synthetic_registration())
    plans = _search_plans()
    if mutation == "missing_arm":
        plans.pop("P-N")
    elif mutation == "duplicate_candidate":
        plans["P-A"] = ArmSearchPlan(
            candidates=(plans["P-A"].candidates[0], plans["P-A"].candidates[0]),
            search_seeds=plans["P-A"].search_seeds,
        )
    elif mutation == "different_seed_order":
        plans["P-N"] = replace(plans["P-N"], search_seeds=(202, 101, 303))
    elif mutation == "search_seed_count_not_balanced":
        plans["P-N"] = replace(plans["P-N"], search_seeds=(101, 202))
        plans["P-A"] = replace(plans["P-A"], search_seeds=(101, 202))
        plans["P-C"] = replace(plans["P-C"], search_seeds=(101, 202))
        plans["P-R"] = replace(plans["P-R"], search_seeds=(101, 202))
        plans["P-T"] = replace(plans["P-T"], search_seeds=(101, 202))
    else:
        plans["P-N"] = replace(
            plans["P-N"],
            candidates=(*plans["P-N"].candidates,
                        CandidateDefinition("candidate-c", _candidate_settings())),
        )

    events: list[tuple] = []
    with pytest.raises(ContractError, match=message):
        driver.run_exp046(
            fold_plan=_valid_plan(),
            search_plans_by_arm=plans,
            final_seeds=(7, 8, 9),
            model_inputs_provider=_model_inputs_provider(_valid_plan(), events),
            source_retention_gate=lambda request: True,
            source_exposure_gate=lambda request: _source_exposure_receipt(request),
            execution_budget_gate=lambda workload: True,
            candidate_selector=lambda packet: "candidate-a",
            epoch_selector=lambda packet: {7: 1, 8: 1, 9: 1},
            device=torch.device("cpu"),
        )
    assert not any(event[0] == "inner" for event in events)


def test_rejects_missing_required_callbacks_and_unregistered_final_seeds(monkeypatch):
    plan = _valid_plan()
    events: list[tuple] = []
    monkeypatch.setattr(
        driver, "_load_exp046_registration", lambda: _synthetic_registration())
    with pytest.raises(ContractError, match="source_exposure_gate is a required callable"):
        driver.run_exp046(
            fold_plan=plan,
            search_plans_by_arm=_search_plans(),
            final_seeds=(7, 8, 9),
            model_inputs_provider=_model_inputs_provider(plan, events),
            source_retention_gate=lambda request: True,
            source_exposure_gate=None,
            execution_budget_gate=lambda workload: True,
            candidate_selector=lambda packet: "candidate-a",
            epoch_selector=lambda packet: {7: 1, 8: 1, 9: 1},
            device=torch.device("cpu"),
        )
    assert not events

    with pytest.raises(ContractError, match="match the three registered model seeds"):
        driver.run_exp046(
            fold_plan=plan,
            search_plans_by_arm=_search_plans(),
            final_seeds=(7, 8, 10),
            model_inputs_provider=_model_inputs_provider(plan, events),
            source_retention_gate=lambda request: True,
            source_exposure_gate=lambda request: _source_exposure_receipt(request),
            execution_budget_gate=lambda workload: True,
            candidate_selector=lambda packet: "candidate-a",
            epoch_selector=lambda packet: {7: 1, 8: 1, 10: 1},
            device=torch.device("cpu"),
        )
    assert not events


def test_rejects_duplicate_final_seed_values(monkeypatch):
    plan = _valid_plan()
    events: list[tuple] = []
    monkeypatch.setattr(
        driver, "_load_exp046_registration", lambda: _synthetic_registration())
    with pytest.raises(ContractError, match="final_seeds must be unique"):
        driver.run_exp046(
            fold_plan=plan,
            search_plans_by_arm=_search_plans(),
            final_seeds=(7, 8, 8),
            model_inputs_provider=_model_inputs_provider(plan, events),
            source_retention_gate=lambda request: True,
            source_exposure_gate=lambda request: _source_exposure_receipt(request),
            execution_budget_gate=lambda workload: True,
            candidate_selector=lambda packet: "candidate-a",
            epoch_selector=lambda packet: {7: 1, 8: 1},
            device=torch.device("cpu"),
        )
    assert not events


@pytest.mark.parametrize("failure", ["budget", "retention", "exposure_intersection",
                                    "exposure_binding", "exposure_callback"])
def test_fails_closed_before_any_fit_for_rejected_preflight_gate(monkeypatch, failure):
    def bad_retention(_request):
        return False

    def bad_exposure(request):
        if failure == "exposure_callback":
            raise RuntimeError("synthetic alias-SENSITIVE-0")
        if failure == "exposure_intersection":
            return _source_exposure_receipt(request, alias_intersection_count=1)
        if failure == "exposure_binding":
            return _source_exposure_receipt(
                request, mask_log_sha256="0" * 64)
        return _source_exposure_receipt(request)

    fit_counter: list[str] = []
    with pytest.raises(ContractError) as error:
        _run_test_driver(
            monkeypatch,
            fit_counter=fit_counter,
            execution_budget_gate=(
                (lambda workload: False) if failure == "budget" else None
            ),
            source_retention_gate=(
                bad_retention if failure == "retention" else None
            ),
            source_exposure_gate=bad_exposure,
        )
    assert fit_counter == []
    assert "SENSITIVE-0" not in str(error.value)


@pytest.mark.parametrize("failure", ["registered_map_hash", "missing_crosswalk"])
def test_rejects_map_or_crosswalk_mismatch_before_budget_or_fit(monkeypatch, failure):
    plan = _valid_plan()
    events: list[tuple] = []
    base_provider = _model_inputs_provider(plan, events)
    fit_counter: list[str] = []

    def bad_provider(outer_key, arm, inner_key, model_seed, map_seed):
        inputs = base_provider(outer_key, arm, inner_key, model_seed, map_seed)
        if failure == "registered_map_hash" and arm == "P-R" and map_seed == 11:
            return replace(inputs, frozen_random_map=_tiny_pathway_map(1))
        if failure == "missing_crosswalk":
            manifest = inputs.source_holdout_manifest
            row = replace(
                manifest.mappings[0],
                pdo_patient_group_key="SYNTHETIC-UNMAPPED-PDO-ID",
            )
            malformed = replace(
                manifest,
                mappings=(row, *manifest.mappings[1:]),
                manifest_sha256="0" * 64,
            )
            malformed = replace(
                malformed,
                manifest_sha256=driver._source_manifest_content_sha256(malformed),
            )
            return replace(inputs, source_holdout_manifest=malformed)
        return inputs

    with pytest.raises(ContractError) as error:
        _run_test_driver(
            monkeypatch,
            model_inputs_provider_override=bad_provider,
            fit_counter=fit_counter,
            event_log=events,
        )
    assert fit_counter == []
    assert not any(event[0] in {"budget-gate", "exposure-gate", "source-gate"}
                   for event in events)
    assert "SYNTHETIC-UNMAPPED-PDO-ID" not in str(error.value)
    if failure == "registered_map_hash":
        assert "registered map_seed SHA-256" in str(error.value)
    else:
        assert "expected_count=" in str(error.value)


def test_exposure_manifest_repr_and_validation_errors_do_not_echo_identity_values():
    plan = _valid_plan()
    outer = plan.outer_folds[0]
    sensitive = "SYNTHETIC-PDO-IDENTITY-DO-NOT-PRINT"
    manifest = _manifest_for(outer.fold_key, None, frozenset({sensitive}))
    assert sensitive not in repr(manifest)
    assert "source-entity" not in repr(manifest)
    sensitive_alias = "SYNTHETIC-SOURCE-ALIAS-DO-NOT-PRINT"
    updated_mapping = replace(
        manifest.mappings[0],
        source_alias_ids=(sensitive_alias,),
    )
    manifest = replace(
        manifest,
        mappings=(updated_mapping,),
        manifest_sha256="0" * 64,
    )
    manifest = replace(
        manifest,
        manifest_sha256=driver._source_manifest_content_sha256(manifest),
    )
    assert sensitive_alias not in repr(manifest)

    changed_row = replace(
        manifest.mappings[0],
        pdo_patient_group_key="SYNTHETIC-PDO-IDENTITY-DO-NOT-PRINT-OTHER",
    )
    malformed = replace(manifest, mappings=(changed_row,))
    with pytest.raises(ContractError) as error:
        driver._validate_source_holdout_manifest(
            malformed,
            outer_fold_key=outer.fold_key,
            inner_fold_key=None,
            expected_pdo_group_keys=frozenset(outer.outer_test.patient_ids),
        )
    assert sensitive not in str(error.value)
    assert "SYNTHETIC-PDO-IDENTITY-DO-NOT-PRINT-OTHER" not in str(error.value)


def test_inner_selection_receives_only_inner_receipts_not_outer_objects(monkeypatch):
    seen = []

    def selector(packet):
        assert isinstance(packet, CandidateSelectionPacket)
        assert not hasattr(packet, "outer_test")
        assert not hasattr(packet, "outer_train")
        assert not hasattr(packet, "fold_plan")
        assert all(not hasattr(item, "patient_ids")
                   for opportunity in packet.opportunities
                   for item in opportunity.receipts)
        seen.append(packet.outer_fold_key)
        return "candidate-a"

    result, _events = _run_test_driver(monkeypatch, candidate_selector=selector)
    assert len(seen) == 25
    assert len(result.predictions) == 105


def test_rejects_outer_prediction_duplicate_pair_keys(monkeypatch):
    with pytest.raises(ContractError, match="invalid keyed predictions"):
        _run_test_driver(monkeypatch, duplicate_outer_predictions=True)
