"""Synthetic CPU contracts for the EXP-045 formal outer-fold driver."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from drug_screen.data.exp045_group_permutation import (
    permute_dependency_groups,
    project_dependency_group_permutation,
)
from drug_screen.models.exp045_dependency_heads import PharmaFormerWithDependencyHeads
from scripts.modeling.exp045_formal_fold import (
    ARMS,
    AUXILIARY_ARMS,
    STAGES,
    ArmPartitionSchedule,
    AuthorizationReceipts,
    AuxiliaryBatch,
    Candidate,
    FoldPlan,
    FormalFoldError,
    InnerFold,
    KeyedDSSBatch,
    KeyedPredictionInput,
    ReceiptEnvelope,
    TrainingEvent,
    TrainingPartition,
    _execute_partition_schedule,
    _new_audit,
    _optimizer,
    _same_aux_inputs,
    _validate_auxiliary_batch,
    axis_sha256,
    candidate_seed_schedule_sha256,
    dependency_dataset_assets_sha256,
    dependency_partition_manifests_sha256,
    dependency_strata_definitions_sha256,
    evaluate_keyed_predictions,
    predict_keyed,
    run,
    run_formal_fold,
    validate_authorization,
    validate_fold_plan,
    validate_partition,
    _verify_shared_initialization,
)


INPUT_AXIS = ("gene-a", "gene-b", "gene-c")
DEPENDENCY_AXIS = ("gene-x", "gene-y")
VALID_MASK = torch.tensor([[True, True], [True, True]], dtype=torch.bool)
EXPRESSION_MASK = torch.tensor([[True, True, False], [True, True, True]], dtype=torch.bool)
DEPENDENCY_ASSETS = {
    "depmap": ({"dataset_id": "synthetic-depmap", "version": "1", "sha256": "c" * 64},),
    "nature": ({"dataset_id": "synthetic-nature", "version": "1", "sha256": "d" * 64},),
}
STRATA_VERSIONS = {"depmap": "synthetic-depmap-strata-v1", "nature": "synthetic-nature-strata-v1"}


class TinyFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gene_fc1 = nn.Linear(3, 4)
        self.gene_fc2 = nn.Linear(4, 4)
        self.smiles_fc = nn.Linear(2, 3)


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = TinyFeatureExtractor()
        self.response = nn.Linear(7, 1)

    def forward(self, expression: torch.Tensor, smiles: torch.Tensor) -> torch.Tensor:
        genes = torch.relu(self.feature_extractor.gene_fc2(
            torch.relu(self.feature_extractor.gene_fc1(expression))))
        drug = torch.relu(self.feature_extractor.smiles_fc(smiles))
        return self.response(torch.cat((genes, drug), dim=1)).flatten()


def model_factory(arm: str, candidate: Candidate, seed: int) -> nn.Module:
    del arm, candidate
    torch.manual_seed(seed)
    return PharmaFormerWithDependencyHeads(
        TinyBackbone(), depmap_target_dim=len(DEPENDENCY_AXIS),
        nature_target_dim=len(DEPENDENCY_AXIS),
    )


def candidate() -> Candidate:
    return Candidate(
        candidate_id="candidate-001", learning_rate=1e-2, weight_decay=0.0,
        dss_ranking_weight=0.1, lambda_reconstruction=0.5,
        lambda_depmap=0.25, lambda_nature=0.4, max_epochs=1, patience=1,
    )


def _dss_batch() -> KeyedDSSBatch:
    keys = tuple((patient, drug) for patient in ("p-1", "p-2")
                 for drug in ("drug-a", "drug-b", "drug-c"))
    return KeyedDSSBatch(
        batch_id="dss-000", row_keys=keys,
        patient_ids=tuple(key[0] for key in keys),
        expression=torch.tensor([
            [0.1, 0.2, 0.3], [0.2, 0.1, 0.4], [0.0, 0.4, 0.3],
            [0.5, 0.3, 0.2], [0.4, 0.6, 0.1], [0.3, 0.2, 0.7],
        ]),
        drug_features=torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [0.5, 0.5],
            [1.0, 0.0], [0.0, 1.0], [0.5, 0.5],
        ]),
        target=torch.tensor([-0.8, 0.1, 0.7, 0.9, -0.1, -0.8]),
    )


def _dependency_batch(stage: str, assignment: str | None, fold_id: str,
                      batch_index: int = 0) -> AuxiliaryBatch:
    del fold_id
    batch_id = f"{stage}-batch-{batch_index:03d}"
    if stage == STAGES[0]:
        samples = (f"cell-{batch_index}-a", f"cell-{batch_index}-b")
        units = (f"cell-{batch_index}-a", f"cell-{batch_index}-b")
        strata = ("colorectal|eligible_crispr",) * 2
        expression = torch.tensor([[0.2, 0.5, 0.3], [0.9, 0.4, 0.1]])
        offset = 0.0
    else:
        samples = (f"pdo-{batch_index}-a-r1", f"pdo-{batch_index}-b-r1")
        units = (f"patient-{batch_index}-a", f"patient-{batch_index}-b")
        strata = ("colorectal|minLib",) * 2
        expression = torch.tensor([[0.3, 0.1, 0.7], [0.4, 0.8, 0.2]])
        offset = 0.3
    reconstruction = expression + 0.1
    dep_target = torch.tensor([[1.0, 3.0], [2.0, 5.0]]) + offset + batch_index
    dep_mask = VALID_MASK.clone() if assignment else None
    return AuxiliaryBatch(
        batch_id=batch_id, stage=stage,
        sample_ids=samples, biological_unit_ids=units,
        expression=expression, expression_axis=INPUT_AXIS,
        reconstruction_target=reconstruction, reconstruction_mask=EXPRESSION_MASK.clone(),
        reconstruction_axis=INPUT_AXIS,
        dependency_target=dep_target if assignment else None,
        dependency_mask=dep_mask,
        dependency_axis=DEPENDENCY_AXIS if assignment else None,
        dependency_train_mean=1.2 if assignment else None,
        dependency_train_overall_std=0.8 if assignment else None,
        label_assignment=assignment,
        permutation_receipt_sha256=None,
        dependency_strata=strata if assignment else None,
        permutation_source_target=None,
        permutation_receipt=None,
    )


def _active_assignment(arm: str, stage: str) -> str | None:
    return {
        ("D-E", STAGES[0]): None,
        ("D-E", STAGES[1]): None,
        ("D-C", STAGES[0]): "true",
        ("D-C", STAGES[1]): None,
        ("D-O", STAGES[0]): None,
        ("D-O", STAGES[1]): "true",
        ("D-CO", STAGES[0]): "true",
        ("D-CO", STAGES[1]): "true",
        ("D-SC", STAGES[0]): "within_domain_shuffled",
        ("D-SC", STAGES[1]): "true",
        ("D-SO", STAGES[0]): "true",
        ("D-SO", STAGES[1]): "within_domain_shuffled",
    }[(arm, stage)]


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _manifest_for(batches: dict[str, AuxiliaryBatch]) -> dict[str, str]:
    observations = sorted({sample for batch in batches.values() for sample in batch.sample_ids})
    groups = sorted({group for batch in batches.values() for group in batch.biological_unit_ids})
    return {
        "observation_ids_sha256": _canonical_sha256(observations),
        "biological_group_ids_sha256": _canonical_sha256(groups),
    }


def _attach_partition_shuffle(
    schedules: dict[str, ArmPartitionSchedule], *, fold_id: str, stage: str, domain: str,
    shuffled_arm: str, true_arms: tuple[str, ...],
) -> None:
    source_batches = schedules["D-CO"].auxiliary_batches[stage]
    ordered_ids = tuple(sorted(
        {sample for batch in source_batches.values() for sample in batch.sample_ids},
    ))
    rows: dict[str, tuple[str, str, np.ndarray]] = {}
    for batch in source_batches.values():
        assert batch.dependency_target is not None and batch.dependency_strata is not None
        targets = batch.dependency_target.detach().cpu().numpy()
        for index, observation_id in enumerate(batch.sample_ids):
            rows[observation_id] = (
                batch.biological_unit_ids[index], batch.dependency_strata[index], targets[index].copy(),
            )
    groups = tuple(rows[observation_id][0] for observation_id in ordered_ids)
    strata = tuple(rows[observation_id][1] for observation_id in ordered_ids)
    targets = np.stack([rows[observation_id][2] for observation_id in ordered_ids])
    result = permute_dependency_groups(
        targets,
        observation_ids=ordered_ids,
        biological_group_ids=groups,
        row_strata=strata,
        dependency_axis=DEPENDENCY_AXIS,
        fold_id=fold_id,
        partition_id=f"{fold_id}:training",
        stage=stage,
        domain=domain,
        batch_id=f"{fold_id}:{stage}:{domain}:full_partition",
        dataset_assets=DEPENDENCY_ASSETS[domain],
        strata_definition_version=STRATA_VERSIONS[domain],
        input_contract_sha256="a" * 64,
        split_sha256="b" * 64,
        seed=20260922,
        training_mean=1.2,
        training_std=0.8,
    )

    shuffled_schedule = schedules[shuffled_arm]
    shuffled_batches = dict(shuffled_schedule.auxiliary_batches[stage])
    true_batch_maps = {arm: dict(schedules[arm].auxiliary_batches[stage]) for arm in true_arms}
    for batch_id, batch in shuffled_batches.items():
        projection = project_dependency_group_permutation(
            result, batch_id=batch_id, observation_ids=batch.sample_ids,
        )
        source_batch = source_batches[batch_id]
        assert source_batch.dependency_target is not None
        shuffled_batches[batch_id] = replace(
            batch,
            dependency_target=torch.from_numpy(np.array(projection.targets, copy=True)).float(),
            dependency_mask=torch.from_numpy(np.array(projection.mask, copy=True)),
            permutation_source_target=source_batch.dependency_target.clone(),
            permutation_receipt=projection.receipt,
            permutation_receipt_sha256=projection.receipt_sha256,
            permutation_parent_receipt_sha256=result.receipt_sha256,
        )
        for arm, batches in true_batch_maps.items():
            true_batch = batches[batch_id]
            batches[batch_id] = replace(
                true_batch, dependency_mask=torch.from_numpy(np.array(projection.mask, copy=True)),
            )
    schedules[shuffled_arm] = replace(
        shuffled_schedule,
        auxiliary_batches={**shuffled_schedule.auxiliary_batches, stage: shuffled_batches},
        permutation_receipts={**shuffled_schedule.permutation_receipts, stage: result.receipt},
        permutation_receipt_sha256s={
            **shuffled_schedule.permutation_receipt_sha256s, stage: result.receipt_sha256,
        },
    )
    for arm, batches in true_batch_maps.items():
        schedule = schedules[arm]
        schedules[arm] = replace(
            schedule, auxiliary_batches={**schedule.auxiliary_batches, stage: batches},
        )


def _partition(fold_id: str, *, multiple_auxiliary_batches: bool = False) -> TrainingPartition:
    dss = _dss_batch()
    schedules = {}
    for arm in ARMS:
        dss_batches = {dss.batch_id: dss}
        if arm == "D-A":
            events = (
                TrainingEvent("dss", dss.batch_id),
                TrainingEvent("dss", dss.batch_id),
                TrainingEvent("dss", dss.batch_id),
            )
            aux_batches = {}
        else:
            aux_batches = {}
            for stage in STAGES:
                batch_count = 2 if multiple_auxiliary_batches else 1
                scheduled_batches = tuple(
                    _dependency_batch(stage, _active_assignment(arm, stage), fold_id, batch_index)
                    for batch_index in range(batch_count)
                )
                aux_batches[stage] = {item.batch_id: item for item in scheduled_batches}
            events = (
                TrainingEvent("dss", dss.batch_id),
                *(TrainingEvent("auxiliary", batch_id, STAGES[0])
                  for batch_id in aux_batches[STAGES[0]]),
                TrainingEvent("dss", dss.batch_id),
                *(TrainingEvent("auxiliary", batch_id, STAGES[1])
                  for batch_id in aux_batches[STAGES[1]]),
                TrainingEvent("dss", dss.batch_id),
            )
        schedules[arm] = ArmPartitionSchedule(
            fold_id=fold_id, training_patient_ids=("p-1", "p-2"),
            dss_batches=dss_batches, auxiliary_batches=aux_batches, events=events,
        )
    _attach_partition_shuffle(
        schedules, fold_id=fold_id, stage=STAGES[0], domain="depmap", shuffled_arm="D-SC",
        true_arms=("D-C", "D-CO", "D-SO"),
    )
    _attach_partition_shuffle(
        schedules, fold_id=fold_id, stage=STAGES[1], domain="nature", shuffled_arm="D-SO",
        true_arms=("D-O", "D-CO", "D-SC"),
    )
    dependency_manifests = {
        "depmap": _manifest_for(schedules["D-CO"].auxiliary_batches[STAGES[0]]),
        "nature": _manifest_for(schedules["D-CO"].auxiliary_batches[STAGES[1]]),
    }
    return TrainingPartition(
        fold_id=fold_id, response_mean=0.2, response_scale=1.3, schedules=schedules,
        dependency_manifest_sha256s=dependency_manifests,
    )


def _prediction_inputs(prefix: str) -> KeyedPredictionInput:
    keys = tuple((patient, drug) for patient in (f"{prefix}-1", f"{prefix}-2")
                 for drug in ("drug-a", "drug-b", "drug-c"))
    return KeyedPredictionInput(
        row_keys=keys, patient_ids=tuple(key[0] for key in keys),
        expression=torch.tensor([
            [0.1, 0.5, 0.3], [0.2, 0.4, 0.6], [0.7, 0.3, 0.1],
            [0.4, 0.1, 0.9], [0.5, 0.7, 0.2], [0.8, 0.2, 0.3],
        ]),
        drug_features=torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [0.5, 0.5],
            [1.0, 0.0], [0.0, 1.0], [0.5, 0.5],
        ]),
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _receipt_envelope(payload: dict) -> ReceiptEnvelope:
    raw = _canonical_bytes(payload)
    return ReceiptEnvelope(payload=payload, sha256=hashlib.sha256(raw).hexdigest(), raw_bytes=raw)


def _rebind_authorization(receipts: AuthorizationReceipts, *, exposure_payload: dict | None = None,
                          input_payload: dict | None = None) -> AuthorizationReceipts:
    exposure = _receipt_envelope(dict(exposure_payload or receipts.exposure.payload))
    input_data = dict(input_payload or receipts.inputs.payload)
    input_data["exposure_receipt_sha256"] = exposure.sha256
    inputs = _receipt_envelope(input_data)
    budget_data = dict(receipts.budget.payload)
    budget_data["exposure_receipt_sha256"] = exposure.sha256
    budget_data["input_receipt_sha256"] = inputs.sha256
    budget = _receipt_envelope(budget_data)
    return AuthorizationReceipts(budget=budget, exposure=exposure, inputs=inputs)


def _nested_receipt(payload: dict) -> dict:
    return {"payload": payload, "sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest()}


def _support_chain(outer_fold: int, *, alias_path: str, alias_sha: str,
                   mask_log_path: str, mask_log_sha: str, checkpoint_path: str,
                   checkpoint_sha: str, exclusion_path: str, exclusion_sha: str,
                   target_mask_path: str, target_mask_sha: str) -> tuple[dict, dict, dict]:
    alias = _nested_receipt({
        "outer_fold": outer_fold, "status": "PASS",
        "source_entity_alias_mapping_path": alias_path,
        "source_entity_alias_mapping_sha256": alias_sha,
        "exclusion_manifest_path": exclusion_path,
        "exclusion_manifest_sha256": exclusion_sha,
    })
    mask = _nested_receipt({
        "outer_fold": outer_fold, "status": "PASS",
        "mask_execution_log_path": mask_log_path,
        "mask_execution_log_sha256": mask_log_sha,
        "target_fold_mask_path": target_mask_path,
        "target_fold_mask_sha256": target_mask_sha,
        "exclusion_manifest_path": exclusion_path,
        "exclusion_manifest_sha256": exclusion_sha,
        "source_checkpoint_path": checkpoint_path,
        "source_checkpoint_sha256": checkpoint_sha,
    })
    lineage = _nested_receipt({
        "outer_fold": outer_fold, "status": "PASS",
        "checkpoint_lineage_content_sha256": "c3" * 32,
        "source_entity_alias_receipt_sha256": alias["sha256"],
        "mask_execution_log_receipt_sha256": mask["sha256"],
        "source_entity_alias_mapping_path": alias["payload"]["source_entity_alias_mapping_path"],
        "source_entity_alias_mapping_sha256": alias["payload"]["source_entity_alias_mapping_sha256"],
        "mask_execution_log_path": mask["payload"]["mask_execution_log_path"],
        "mask_execution_log_sha256": mask["payload"]["mask_execution_log_sha256"],
        "target_fold_mask_path": target_mask_path,
        "target_fold_mask_sha256": target_mask_sha,
        "exclusion_manifest_path": exclusion_path,
        "exclusion_manifest_sha256": exclusion_sha,
        "source_checkpoint_path": checkpoint_path,
        "source_checkpoint_sha256": checkpoint_sha,
    })
    return alias, mask, lineage


def _support_artifacts(support_dir: Path, outer_fold: int) -> dict[str, dict[str, str]]:
    support_dir.mkdir(parents=True, exist_ok=True)
    contents = {
        "alias": b"synthetic alias-map bytes; no entity data",
        "mask_log": b"synthetic mask-log bytes; no sample data",
        "target_mask": b"synthetic target-fold mask bytes",
        "exclusion": b"synthetic exclusion-manifest bytes",
        "checkpoint": b"synthetic checkpoint bytes; never deserialized",
    }
    names = {
        "alias": f"outer-{outer_fold}-alias-map.bin",
        "mask_log": f"outer-{outer_fold}-mask-log.bin",
        "target_mask": f"outer-{outer_fold}-target-mask.bin",
        "exclusion": f"outer-{outer_fold}-exclusions.bin",
        "checkpoint": f"depmap-source-{outer_fold}.pt",
    }
    result = {}
    for key, filename in names.items():
        path = support_dir / filename
        path.write_bytes(contents[key])
        result[key] = {"path": str(path), "sha256": hashlib.sha256(contents[key]).hexdigest()}
    return result


def _default_support_artifacts() -> dict[str, dict[str, str]]:
    return {
        "alias": {"path": "synthetic/source-entity-alias-map.json", "sha256": "a1" * 32},
        "mask_log": {"path": "synthetic/mask-execution-log.json", "sha256": "b2" * 32},
        "target_mask": {"path": "synthetic/masks/outer-0-mask.json", "sha256": "1" * 64},
        "exclusion": {"path": "synthetic/receipts/outer-0-exclusions.json", "sha256": "d" * 64},
        "checkpoint": {"path": "synthetic/checkpoints/depmap-source.pt", "sha256": "c" * 64},
    }


def _plan(support_dir: Path | None = None) -> FoldPlan:
    candidate_item = candidate()
    inner = InnerFold(
        fold_id="inner-0", training=_partition("inner-0"),
        validation_inputs=_prediction_inputs("val"),
        validation_labels=tuple(
            (key, float(index % 3 - 1)) for index, key in enumerate(_prediction_inputs("val").row_keys)
        ),
    )
    files = _support_artifacts(support_dir, 0) if support_dir is not None else _default_support_artifacts()
    checkpoint_path = files["checkpoint"]["path"]
    exclusion_path = files["exclusion"]["path"]
    target_mask_path = files["target_mask"]["path"]
    chain = _support_chain(
        0, alias_path=files["alias"]["path"], alias_sha=files["alias"]["sha256"],
        mask_log_path=files["mask_log"]["path"], mask_log_sha=files["mask_log"]["sha256"],
        checkpoint_path=checkpoint_path, checkpoint_sha=files["checkpoint"]["sha256"],
        exclusion_path=exclusion_path, exclusion_sha=files["exclusion"]["sha256"],
        target_mask_path=target_mask_path, target_mask_sha=files["target_mask"]["sha256"],
    )
    return FoldPlan(
        outer_fold=0, input_axis=INPUT_AXIS, dependency_axis=DEPENDENCY_AXIS,
        inner_folds=(inner,), outer_training=_partition("outer-0"),
        outer_test_inputs=_prediction_inputs("test"), candidates=(candidate_item,),
        selection_seeds=(7,), final_seeds=(7,),
        candidate_schedule_by_arm={arm: (candidate_item.candidate_id,) for arm in ARMS},
        selection_seed_schedule_by_arm={arm: (7,) for arm in ARMS},
        final_seed_schedule_by_arm={arm: (7,) for arm in ARMS},
        inner_fold_schedule_by_arm={arm: ("inner-0",) for arm in ARMS},
        input_contract_sha256="a" * 64, split_sha256="b" * 64,
        source_checkpoint_path=checkpoint_path, source_checkpoint_sha256=files["checkpoint"]["sha256"],
        exclusion_manifest_sha256=files["exclusion"]["sha256"],
        target_fold_mask_sha256=files["target_mask"]["sha256"],
        checkpoint_lineage_sha256=chain[2]["sha256"],
        dependency_dataset_assets=DEPENDENCY_ASSETS,
        dependency_strata_definition_versions=STRATA_VERSIONS,
    )


def _receipts(plan: FoldPlan, support_dir: Path | None = None, *,
              cuda_device: str = "cuda:0") -> AuthorizationReceipts:
    schedule_hash = candidate_seed_schedule_sha256(plan)
    files = _support_artifacts(support_dir, plan.outer_fold) if support_dir is not None else _default_support_artifacts()
    chain = _support_chain(
        plan.outer_fold,
        alias_path=files["alias"]["path"], alias_sha=files["alias"]["sha256"],
        mask_log_path=files["mask_log"]["path"], mask_log_sha=files["mask_log"]["sha256"],
        checkpoint_path=plan.source_checkpoint_path,
        checkpoint_sha=plan.source_checkpoint_sha256,
        exclusion_path=files["exclusion"]["path"],
        exclusion_sha=plan.exclusion_manifest_sha256,
        target_mask_path=files["target_mask"]["path"],
        target_mask_sha=plan.target_fold_mask_sha256,
    )
    exposure = _receipt_envelope({
        "experiment_id": "EXP-045", "outer_fold": plan.outer_fold,
        "approved_for_training": True, "approved_by": "GPT-6-Sol/medium",
        "source_checkpoint_path": plan.source_checkpoint_path,
        "source_checkpoint_sha256": plan.source_checkpoint_sha256,
        "exclusion_manifest_path": files["exclusion"]["path"],
        "exclusion_manifest_sha256": plan.exclusion_manifest_sha256,
        "target_fold_mask_path": files["target_mask"]["path"],
        "target_fold_mask_sha256": plan.target_fold_mask_sha256,
        "checkpoint_lineage_sha256": plan.checkpoint_lineage_sha256,
        "source_checkpoint_lineage_status": "PASS",
        "target_fold_mask_exposure_status": "PASS",
        "source_entity_alias_receipt": chain[0],
        "mask_execution_log_receipt": chain[1],
        "checkpoint_lineage_receipt": chain[2],
    })
    inputs = _receipt_envelope({
            "experiment_id": "EXP-045", "outer_fold": plan.outer_fold,
            "approved_for_training": True, "approved_by": "GPT-6-Sol/medium",
            "input_contract_sha256": plan.input_contract_sha256,
            "split_sha256": plan.split_sha256,
            "exposure_receipt_sha256": exposure.sha256,
            "target_fold_mask_sha256": plan.target_fold_mask_sha256,
            "input_axis_sha256": axis_sha256(plan.input_axis),
            "dependency_axis_sha256": axis_sha256(plan.dependency_axis),
            "candidate_seed_schedule_sha256": schedule_hash,
            "dependency_dataset_assets_sha256": dependency_dataset_assets_sha256(plan),
            "dependency_strata_definitions_sha256": dependency_strata_definitions_sha256(plan),
            "dependency_partition_manifests_sha256": dependency_partition_manifests_sha256(plan),
    })
    budget = _receipt_envelope({
        "experiment_id": "EXP-045", "outer_fold": plan.outer_fold,
        "approved_for_training": True, "approved_by": "GPT-6-Sol/medium",
        "training_go": True, "package_gpu_hours_remaining": 24.0,
        "reserved_gpu_hours": 2.0, "run_wall_hours_max": 20.0,
        "candidate_seed_schedule_sha256": schedule_hash,
        "exposure_receipt_sha256": exposure.sha256,
        "input_receipt_sha256": inputs.sha256,
        "cuda_device": cuda_device, "cuda_device_name": "NVIDIA RTX A5000",
    })
    return AuthorizationReceipts(budget=budget, exposure=exposure, inputs=inputs)


def test_seven_arm_contract_has_matching_batches_dss_updates_and_auxiliary_opportunities() -> None:
    plan = _plan()
    validate_fold_plan(plan)
    partition = plan.outer_training
    audits = {}
    for arm in ARMS:
        torch.manual_seed(7)
        model = model_factory(arm, plan.candidates[0], 7)
        optimizer = _optimizer(model, plan.candidates[0])
        audit = _execute_partition_schedule(
            model, optimizer, arm=arm, schedule=partition.schedules[arm], plan=plan,
            candidate=plan.candidates[0], epochs=1, device=torch.device("cpu"),
        )
        audits[arm] = audit
    assert {audit.dss_optimizer_steps for audit in audits.values()} == {3}
    assert all(audits[arm].dss_scheduled == audits[arm].dss_forward_calls
               == audits[arm].dss_backward_calls == audits[arm].dss_optimizer_steps
               for arm in ARMS)
    assert audits["D-A"].auxiliary[STAGES[0]].optimizer_steps == 0
    assert audits["D-A"].auxiliary[STAGES[1]].optimizer_steps == 0
    for arm in AUXILIARY_ARMS:
        assert audits[arm].auxiliary[STAGES[0]].optimizer_steps == 1
        assert audits[arm].auxiliary[STAGES[1]].optimizer_steps == 1
    assert len({audits[arm].auxiliary[STAGES[0]].optimizer_steps for arm in AUXILIARY_ARMS}) == 1
    assert len({audits[arm].auxiliary[STAGES[1]].optimizer_steps for arm in AUXILIARY_ARMS}) == 1


def test_d_e_reconstructs_in_both_stages_and_dependency_terms_are_loss_increments() -> None:
    plan = _plan()
    audits = {}
    for arm in ("D-E", "D-CO"):
        torch.manual_seed(7)
        model = model_factory(arm, plan.candidates[0], 7)
        optimizer = _optimizer(model, plan.candidates[0])
        audits[arm] = _execute_partition_schedule(
            model, optimizer, arm=arm, schedule=plan.outer_training.schedules[arm],
            plan=plan, candidate=plan.candidates[0], epochs=1, device=torch.device("cpu"),
        )
    for stage in STAGES:
        control = audits["D-E"].auxiliary[stage]
        combined = audits["D-CO"].auxiliary[stage]
        assert control.optimizer_steps == combined.optimizer_steps == 1
        assert control.reconstruction_valid_denominator == combined.reconstruction_valid_denominator
        assert control.depmap_losses == [None]
        assert control.nature_losses == [None]
        assert control.total_losses[0] == pytest.approx(
            plan.candidates[0].lambda_reconstruction * control.reconstruction_losses[0],
            rel=1e-6,
        )
    depmap_control = audits["D-E"].auxiliary[STAGES[0]]
    depmap_combined = audits["D-CO"].auxiliary[STAGES[0]]
    assert depmap_combined.depmap_losses[0] is not None
    assert depmap_combined.total_losses[0] - (
        plan.candidates[0].lambda_reconstruction * depmap_combined.reconstruction_losses[0]
    ) == pytest.approx(
        plan.candidates[0].lambda_depmap * depmap_combined.depmap_losses[0], rel=1e-5, abs=1e-7
    )
    assert depmap_control.total_losses[0] == pytest.approx(
        plan.candidates[0].lambda_reconstruction * depmap_control.reconstruction_losses[0], rel=1e-6
    )


def test_shared_auxiliary_inputs_accept_matching_masked_nan_only() -> None:
    stage = STAGES[0]
    base = _dependency_batch(stage, None, "fold-nan")
    mask = torch.tensor([[True, True, False], [True, False, False]], dtype=torch.bool)

    left_target = base.reconstruction_target.clone()
    left_target[0, 2] = torch.nan
    left = replace(base, reconstruction_target=left_target, reconstruction_mask=mask)

    matching_target = left_target.clone()
    matching = replace(base, reconstruction_target=matching_target, reconstruction_mask=mask.clone())
    for batch in (left, matching):
        _validate_auxiliary_batch(
            batch,
            arm="D-E",
            stage=stage,
            input_axis=INPUT_AXIS,
            dependency_axis=DEPENDENCY_AXIS,
            fold_id="fold-nan",
            input_contract_sha256="a" * 64,
            split_sha256="b" * 64,
        )
    assert _same_aux_inputs(left, matching)

    changed_valid_target = matching.reconstruction_target.clone()
    changed_valid_target[0, 0] += 1.0
    changed_valid = replace(matching, reconstruction_target=changed_valid_target)
    _validate_auxiliary_batch(
        changed_valid,
        arm="D-E",
        stage=stage,
        input_axis=INPUT_AXIS,
        dependency_axis=DEPENDENCY_AXIS,
        fold_id="fold-nan",
        input_contract_sha256="a" * 64,
        split_sha256="b" * 64,
    )
    assert not _same_aux_inputs(left, changed_valid)

    moved_nan_target = matching.reconstruction_target.clone()
    moved_nan_target[0, 2] = base.reconstruction_target[0, 2]
    moved_nan_target[1, 2] = torch.nan
    moved_nan = replace(matching, reconstruction_target=moved_nan_target)
    _validate_auxiliary_batch(
        moved_nan,
        arm="D-E",
        stage=stage,
        input_axis=INPUT_AXIS,
        dependency_axis=DEPENDENCY_AXIS,
        fold_id="fold-nan",
        input_contract_sha256="a" * 64,
        split_sha256="b" * 64,
    )
    assert not _same_aux_inputs(left, moved_nan)


def test_d_a_is_dss_only_and_same_matched_dss_baseline() -> None:
    plan = _plan()
    validate_partition(plan.outer_training, plan)
    torch.manual_seed(7)
    model = model_factory("D-A", plan.candidates[0], 7)
    optimizer = _optimizer(model, plan.candidates[0])
    audit = _execute_partition_schedule(
        model, optimizer, arm="D-A", schedule=plan.outer_training.schedules["D-A"],
        plan=plan, candidate=plan.candidates[0], epochs=2, device=torch.device("cpu"),
    )
    assert audit.dss_optimizer_steps == 6
    assert all(audit.auxiliary[stage].optimizer_steps == 0 for stage in STAGES)


def test_auxiliary_arms_must_share_the_full_initial_model_state() -> None:
    _verify_shared_initialization(model_factory, (candidate(),), (7,))

    def mismatched_auxiliary_factory(arm: str, config: Candidate, seed: int) -> nn.Module:
        model = model_factory(arm, config, seed)
        if arm == "D-CO":
            with torch.no_grad():
                model.expression_reconstruction_head.weight[0, 0].add_(0.25)
        return model

    with pytest.raises(FormalFoldError, match="initial auxiliary model state differs"):
        _verify_shared_initialization(mismatched_auxiliary_factory, (candidate(),), (7,))


@pytest.mark.parametrize("mutation", [
    "sample_order", "sample_missing", "expression_mask", "expression_axis",
    "dependency_mask", "dependency_axis",
])
def test_shared_identity_axis_and_mask_mismatches_fail_closed(mutation: str) -> None:
    plan = _plan()
    partition = plan.outer_training
    schedules = dict(partition.schedules)
    schedule = schedules["D-SO"]
    stage = STAGES[1] if mutation in {"dependency_mask", "dependency_axis"} else STAGES[0]
    batches = {key: dict(value) for key, value in schedule.auxiliary_batches.items()}
    old = batches[stage][f"{stage}-batch-000"]
    changes = {}
    if mutation == "sample_order":
        changes["sample_ids"] = tuple(reversed(old.sample_ids))
    elif mutation == "sample_missing":
        changes["sample_ids"] = old.sample_ids[:-1]
    elif mutation == "expression_mask":
        changes["reconstruction_mask"] = ~old.reconstruction_mask
    elif mutation == "expression_axis":
        changes["reconstruction_axis"] = tuple(reversed(old.reconstruction_axis))
    elif mutation == "dependency_mask":
        changes["dependency_mask"] = ~old.dependency_mask
    else:
        changes["dependency_axis"] = tuple(reversed(old.dependency_axis))
    batches[stage][old.batch_id] = replace(old, **changes)
    schedules["D-SO"] = replace(schedule, auxiliary_batches=batches)
    changed_partition = replace(partition, schedules=schedules)
    with pytest.raises(FormalFoldError):
        validate_partition(changed_partition, plan)


def test_two_scheduled_batches_per_stage_pass_cross_arm_contract() -> None:
    plan = _plan()
    partition = _partition(plan.outer_training.fold_id, multiple_auxiliary_batches=True)
    validate_partition(partition, plan)

    shuffled = partition.schedules["D-SC"]
    parent = shuffled.permutation_receipts[STAGES[0]]
    assert parent["n_groups_input"] == 4
    assert parent["n_observations_input"] == 4
    group_batch = {
        group: batch_id
        for batch_id, batch in partition.schedules["D-CO"].auxiliary_batches[STAGES[0]].items()
        for group in batch.biological_unit_ids
    }
    assert any(
        group_batch[recipient] != group_batch[donor]
        for recipient, donor in parent["group_mapping"].items()
    )
    assert all(
        batch.permutation_parent_receipt_sha256 == shuffled.permutation_receipt_sha256s[STAGES[0]]
        for batch in shuffled.auxiliary_batches[STAGES[0]].values()
    )


def test_batch_projection_tampering_fails_against_full_partition_parent() -> None:
    plan = _plan()
    partition = _partition(plan.outer_training.fold_id, multiple_auxiliary_batches=True)
    schedules = dict(partition.schedules)
    schedule = schedules["D-SC"]
    batches = dict(schedule.auxiliary_batches[STAGES[0]])
    batch = batches[f"{STAGES[0]}-batch-000"]
    changed_receipt = dict(batch.permutation_receipt)
    changed_receipt["row_mapping"] = list(changed_receipt["row_mapping"])
    changed_receipt["row_mapping"][0] = dict(changed_receipt["row_mapping"][0])
    changed_receipt["row_mapping"][0]["source_observation_id"] = "tampered-source"
    changed_batch = replace(
        batch,
        permutation_receipt=changed_receipt,
        permutation_receipt_sha256=_canonical_sha256(changed_receipt),
    )
    batches[batch.batch_id] = changed_batch
    schedules["D-SC"] = replace(
        schedule,
        auxiliary_batches={**schedule.auxiliary_batches, STAGES[0]: batches},
    )
    with pytest.raises(FormalFoldError, match="child projection failed recomputation"):
        validate_partition(replace(partition, schedules=schedules), plan)


@pytest.mark.parametrize("mutation", ["dependency_mask", "true_dependency_target"])
def test_second_batch_supervision_mismatch_fails_closed(mutation: str) -> None:
    plan = _plan()
    partition = _partition(plan.outer_training.fold_id, multiple_auxiliary_batches=True)
    schedules = dict(partition.schedules)
    schedule = schedules["D-SO"]
    batches = {key: dict(value) for key, value in schedule.auxiliary_batches.items()}
    second_id = f"{STAGES[0]}-batch-001"
    old = batches[STAGES[0]][second_id]
    if mutation == "dependency_mask":
        changes = {"dependency_mask": ~old.dependency_mask}
        # Keep the mask non-empty so failure specifically reflects cross-arm
        # mismatch, rather than a malformed batch.
        changes["dependency_mask"][0, 0] = True
    else:
        changes = {"dependency_target": old.dependency_target + 1.0}
    batches[STAGES[0]][second_id] = replace(old, **changes)
    schedules["D-SO"] = replace(schedule, auxiliary_batches=batches)
    with pytest.raises(FormalFoldError):
        validate_partition(replace(partition, schedules=schedules), plan)


@pytest.mark.parametrize("arm,stage", [("D-C", STAGES[0]), ("D-O", STAGES[1]), ("D-SC", STAGES[0])])
def test_missing_or_misassigned_supervision_fails_closed(arm: str, stage: str) -> None:
    plan = _plan()
    partition = plan.outer_training
    schedules = dict(partition.schedules)
    schedule = schedules[arm]
    batches = {key: dict(value) for key, value in schedule.auxiliary_batches.items()}
    old = batches[stage][f"{stage}-batch-000"]
    batches[stage][old.batch_id] = replace(
        old, dependency_target=None, dependency_mask=None, dependency_axis=None,
        dependency_train_mean=None, dependency_train_overall_std=None,
        label_assignment=None, permutation_receipt_sha256=None,
    )
    schedules[arm] = replace(schedule, auxiliary_batches=batches)
    with pytest.raises(FormalFoldError):
        validate_partition(replace(partition, schedules=schedules), plan)


def test_candidate_seed_and_inner_fold_schedules_must_match_all_arms() -> None:
    plan = _plan()
    changed = dict(plan.final_seed_schedule_by_arm)
    changed["D-SO"] = (8,)
    with pytest.raises(FormalFoldError, match="seed schedule"):
        validate_fold_plan(replace(plan, final_seed_schedule_by_arm=changed))
    changed_candidates = dict(plan.candidate_schedule_by_arm)
    changed_candidates["D-C"] = ()
    with pytest.raises(FormalFoldError, match="candidate/search opportunities"):
        validate_fold_plan(replace(plan, candidate_schedule_by_arm=changed_candidates))


def test_missing_inner_validation_labels_fail_before_model_construction() -> None:
    plan = _plan()
    inner = replace(plan.inner_folds[0], validation_labels=())
    with pytest.raises(FormalFoldError, match="inner validation labels are missing"):
        validate_fold_plan(replace(plan, inner_folds=(inner,)))


def test_missing_budget_exposure_or_input_receipt_blocks_formal_run(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    receipts = _receipts(plan, tmp_path)
    for missing in ("budget", "exposure", "inputs"):
        incomplete = replace(receipts, **{missing: None})
        with pytest.raises(FormalFoldError, match="receipt is missing"):
            validate_authorization(plan, incomplete, device="cuda:0", check_cuda=False)


def test_formal_entrypoint_checks_receipts_before_calling_model_factory() -> None:
    plan = _plan()

    def forbidden_factory(arm: str, candidate_item: Candidate, seed: int) -> nn.Module:
        raise AssertionError("receipt failure must happen before model construction")

    with pytest.raises(FormalFoldError, match="budget.*receipt is missing"):
        run_formal_fold(
            plan, AuthorizationReceipts(None, None, None), device="cuda:0",
            model_factory=forbidden_factory,
        )


def test_receipts_must_bind_the_exact_fold_axes_and_candidate_schedule(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    receipts = _receipts(plan, tmp_path)
    payload = dict(receipts.inputs.payload)
    payload["candidate_seed_schedule_sha256"] = "0" * 64
    invalid = _rebind_authorization(receipts, input_payload=payload)
    with pytest.raises(FormalFoldError, match="candidate/seed schedule"):
        validate_authorization(plan, invalid, device="cuda:0", check_cuda=False)


@pytest.mark.parametrize("field,value", [
    ("source_checkpoint_lineage_status", "UNVERIFIED"),
    ("target_fold_mask_exposure_status", "UNVERIFIED"),
])
def test_source_checkpoint_or_target_fold_mask_exposure_must_be_verified(
    field: str, value: str, tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    receipts = _receipts(plan, tmp_path)
    payload = dict(receipts.exposure.payload)
    payload[field] = value
    invalid = _rebind_authorization(receipts, exposure_payload=payload)
    with pytest.raises(FormalFoldError, match="status is not PASS"):
        validate_authorization(plan, invalid, device="cuda:0", check_cuda=False)


def test_exposure_gate_rejects_pseudo_pass_without_supporting_receipts(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    receipts = _receipts(plan, tmp_path)
    payload = dict(receipts.exposure.payload)
    payload.pop("source_entity_alias_receipt")
    invalid = _rebind_authorization(receipts, exposure_payload=payload)
    with pytest.raises(FormalFoldError, match="missing nested source_entity_alias_receipt"):
        validate_authorization(plan, invalid, device="cuda:0", check_cuda=False)


def test_exposure_gate_rejects_nested_receipt_for_wrong_target_fold(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    receipts = _receipts(plan, tmp_path)
    exposure = dict(receipts.exposure.payload)
    bad_child = dict(exposure["mask_execution_log_receipt"]["payload"])
    bad_child["outer_fold"] = 1
    exposure["mask_execution_log_receipt"] = _nested_receipt(bad_child)
    lineage = dict(exposure["checkpoint_lineage_receipt"]["payload"])
    lineage["mask_execution_log_receipt_sha256"] = exposure["mask_execution_log_receipt"]["sha256"]
    exposure["checkpoint_lineage_receipt"] = _nested_receipt(lineage)
    exposure["checkpoint_lineage_sha256"] = exposure["checkpoint_lineage_receipt"]["sha256"]
    invalid = _rebind_authorization(receipts, exposure_payload=exposure)
    with pytest.raises(FormalFoldError, match="different target fold"):
        validate_authorization(plan, invalid, device="cuda:0", check_cuda=False)


def test_exposure_gate_rejects_mismatched_nested_receipt_digest(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    receipts = _receipts(plan, tmp_path)
    exposure = dict(receipts.exposure.payload)
    stale = dict(exposure["source_entity_alias_receipt"])
    stale["sha256"] = "0" * 64
    exposure["source_entity_alias_receipt"] = stale
    invalid = _rebind_authorization(receipts, exposure_payload=exposure)
    with pytest.raises(FormalFoldError, match="nested source_entity_alias_receipt receipt digest"):
        validate_authorization(plan, invalid, device="cuda:0", check_cuda=False)


def test_budget_receipt_must_bind_exact_input_and_exposure_bytes(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    receipts = _receipts(plan, tmp_path)
    payload = dict(receipts.budget.payload)
    payload["exposure_receipt_sha256"] = "0" * 64
    invalid = replace(receipts, budget=_receipt_envelope(payload))
    with pytest.raises(FormalFoldError, match="exact exposure receipt bytes"):
        validate_authorization(plan, invalid, device="cuda:0", check_cuda=False)


def test_support_artifact_paths_and_raw_hashes_pass_without_checkpoint_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path)
    receipts = _receipts(plan, tmp_path)

    def forbidden_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("checkpoint authorization must hash bytes only")

    monkeypatch.setattr(torch, "load", forbidden_load)
    device, device_name = validate_authorization(
        plan, receipts, device="cuda:0", check_cuda=False,
    )
    assert str(device) == "cuda:0"
    assert device_name == "NVIDIA RTX A5000"


@pytest.mark.parametrize("artifact,label", [
    ("source_entity_alias_receipt", "source entity/alias mapping"),
    ("mask_execution_log_receipt", "mask execution log"),
    ("target_fold_mask_path", "target-fold mask"),
    ("exclusion_manifest_path", "exclusion manifest"),
    ("source_checkpoint_path", "source checkpoint"),
])
@pytest.mark.parametrize("failure", ["missing", "mismatch"])
def test_support_artifact_missing_or_changed_bytes_fail_closed(
    artifact: str, label: str, failure: str, tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    receipts = _receipts(plan, tmp_path)
    exposure = receipts.exposure.payload
    if artifact == "source_entity_alias_receipt":
        path = Path(exposure[artifact]["payload"]["source_entity_alias_mapping_path"])
    elif artifact == "mask_execution_log_receipt":
        path = Path(exposure[artifact]["payload"]["mask_execution_log_path"])
    else:
        path = Path(exposure[artifact])
    if failure == "missing":
        path.unlink()
        expected = "missing or is not a regular file"
    else:
        path.write_bytes(path.read_bytes() + b" changed")
        expected = "raw-byte SHA-256 does not match"
    with pytest.raises(FormalFoldError, match=expected):
        validate_authorization(plan, receipts, device="cuda:0", check_cuda=False)


def test_cuda_unavailable_or_device_mismatch_fails_without_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    receipts = _receipts(plan, tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(FormalFoldError, match="will not fall back to CPU"):
        validate_authorization(plan, receipts, device="cuda:0", check_cuda=True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(FormalFoldError, match="differs from the budget receipt"):
        validate_authorization(plan, receipts, device="cuda:1", check_cuda=True)


def test_keyed_evaluation_joins_by_keys_and_rejects_missing_label_rows() -> None:
    rows = (
        (("p-1", "drug-a"), 0.1), (("p-1", "drug-b"), 0.3), (("p-1", "drug-c"), 0.8),
        (("p-2", "drug-a"), 0.9), (("p-2", "drug-b"), 0.2), (("p-2", "drug-c"), 0.5),
    )
    labels = {
        ("p-1", "drug-a"): 1.0, ("p-1", "drug-b"): 2.0, ("p-1", "drug-c"): 3.0,
        ("p-2", "drug-a"): 2.0, ("p-2", "drug-b"): 1.0, ("p-2", "drug-c"): 3.0,
    }
    result = evaluate_keyed_predictions(tuple(reversed(rows)), labels)
    assert result.metric == "patient_macro_within_patient_spearman"
    assert result.patient_macro_spearman == pytest.approx(0.75)
    assert result.valid_patients == 2
    with pytest.raises(FormalFoldError, match="key mismatch"):
        evaluate_keyed_predictions(rows, {key: value for key, value in labels.items()
                                          if key != ("p-1", "drug-a")})


def test_prediction_interface_returns_predictions_in_input_key_order() -> None:
    plan = _plan()
    inputs = _prediction_inputs("patient")
    torch.manual_seed(7)
    model = model_factory("D-A", plan.candidates[0], 7)
    predictions = predict_keyed(
        model, inputs, device=torch.device("cpu"), response_mean=2.5,
        response_scale=1.2, batch_size=2,
    )
    assert tuple(key for key, _ in predictions) == inputs.row_keys
    assert len(predictions) == len(inputs.row_keys)
    assert all(torch.isfinite(torch.tensor(value)) for _, value in predictions)


def test_failure_from_a_model_is_propagated_and_counted() -> None:
    plan = _plan()
    schedule = plan.outer_training.schedules["D-A"]

    class BrokenModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def predict_drug_sensitivity(self, expression: torch.Tensor,
                                     drug: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("synthetic model failure")

    model = BrokenModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    audit = _new_audit("D-A")
    with pytest.raises(RuntimeError, match="synthetic model failure"):
        _execute_partition_schedule(
            model, optimizer, arm="D-A", schedule=schedule, plan=plan,
            candidate=plan.candidates[0], epochs=1, device=torch.device("cpu"), audit=audit,
        )
    assert audit.failures == 1


def test_cli_defaults_to_metadata_only_dry_run() -> None:
    args = argparse.Namespace(train=False)
    result = run(args)
    assert result["mode"] == "DRY_RUN"
    assert result["status"] == "NO_DATA_READ_NO_MODEL_BUILT_NO_TRAINING"
