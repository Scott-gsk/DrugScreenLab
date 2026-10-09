from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.modeling.exp037_xpert_fold import (
    ContractError,
    FoldDataLoader,
    FoldRows,
    SPLIT_V1_SHA256,
    SPLIT_V2_SHA256,
    build_official_loss,
    validate_drug_feature_index,
    validate_expected_digest,
    validate_formal_training_plan,
    validate_gene_map,
    validate_key_axes,
    validate_split_file,
    validate_split_payload,
)


def _synthetic_split() -> dict:
    all_cells = [f"cell_{i:02d}" for i in range(70)]
    folds = []
    for fold in range(5):
        test_ids = all_cells[fold * 14:(fold + 1) * 14]
        train_ids = [cell for cell in all_cells if cell not in set(test_ids)]
        validation_parts = (train_ids[:19], train_ids[19:38], train_ids[38:])
        inner_folds = []
        for inner_fold, validation in enumerate(validation_parts):
            inner_folds.append({
                "inner_fold": inner_fold,
                "train_ids": [cell for cell in train_ids if cell not in set(validation)],
                "validation_ids": validation,
            })
        folds.append({
            "fold": fold,
            "outer_train_ids": train_ids,
            "outer_test_ids": test_ids,
            "label_fractions": [{
                "fraction": 1.0,
                "train_ids": train_ids,
                "inner_folds": inner_folds,
            }],
        })
    return {
        "protocol": "dual_route_lincs_level3_cell_holdout_20260922_v2_parseable_structure",
        "split_seed": 20260922,
        "n_eligible_instances": 661762,
        "n_eligible_cells": 70,
        "n_eligible_drugs": 20336,
        "folds": folds,
    }


def _valid_inchikey(letter: str) -> str:
    return f"{letter * 14}-{letter * 10}-{letter}"


def _formal_plan() -> dict:
    return {
        "experiment_id": "EXP-037",
        "split_sha256": SPLIT_V2_SHA256,
        "condition_delta_sha256": "29317d50b44933452126ddd213017709d54b37f6c0dcc2f403cf5ac5ff8112fb",
        "condition_manifest_sha256": "a" * 64,
        "input_manifest_sha256": "b" * 64,
        "xpert_asset_audit_sha256": "04d46c22c48e295574ea97a333aecc9d1b862ec9e051cb5ac5b9fc63c56a751a",
        "shared_drug_ids_sha256": "c" * 64,
        "shared_drug_comparison_id": "smoke-fixture-only",
        "initialization_seeds": [20260922, 20260923, 20260924],
        "batch_size": 2,
        "per_run_wallclock_hours": 20,
        "experiment_gpu_hours": 48,
        "candidates": [{
            "candidate_id": "fixture",
            "model_config_sha256": "d" * 64,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "max_epochs": 10,
            "patience": 2,
            "loss_weights": [0.2, 0.003, 0.2, 1.0],
        }],
    }


def test_v2_split_digest_is_explicit_and_v1_cannot_be_passed_as_v2(tmp_path):
    with pytest.raises(ContractError, match="explicit"):
        validate_expected_digest(None, canonical=SPLIT_V2_SHA256, name="split-sha256")
    with pytest.raises(ContractError, match="legacy v1"):
        validate_expected_digest(SPLIT_V1_SHA256, canonical=SPLIT_V2_SHA256,
                                 name="split-sha256")
    changed = tmp_path / "lincs_cell_splits_20260922_v2.json"
    changed.write_text("{}", encoding="utf-8")
    with pytest.raises(ContractError, match="registered v2 digest"):
        validate_split_file(changed, expected_sha256=SPLIT_V2_SHA256, fold_index=0)


def test_split_payload_enforces_five_outer_and_three_inner_identity_partitions():
    split = validate_split_payload(_synthetic_split(), fold_index=3)
    assert split.fold == 3
    assert len(split.outer_train_ids) == 56
    assert len(split.outer_test_ids) == 14
    assert len(split.inner_folds) == 3
    assert set().union(*(item.validation_ids for item in split.inner_folds)) == set(
        split.outer_train_ids)

    invalid = _synthetic_split()
    invalid["folds"][0]["label_fractions"][0]["inner_folds"][0][
        "validation_ids"][0] = invalid["folds"][0]["outer_test_ids"][0]
    with pytest.raises(ContractError, match="inner identities"):
        validate_split_payload(invalid, fold_index=0)


def test_gene_map_must_cover_the_exact_978_position_axis():
    target_genes = [f"gene_{i:03d}" for i in range(978)]
    mapping = np.arange(978, dtype=np.int64)[::-1]
    result = validate_gene_map(target_genes, mapping.tolist())
    assert result.shape == (978,)
    assert np.array_equal(result, mapping)
    with pytest.raises(ContractError, match="permutation"):
        validate_gene_map(target_genes, [0] * 978)
    with pytest.raises(ContractError, match="unique IDs"):
        validate_gene_map(target_genes[:-1] + [target_genes[0]], mapping.tolist())


def test_drug_axis_uses_complete_inchikey_and_matching_full_feature_rows():
    target = pd.DataFrame({
        "drug_index": [0, 1, 2],
        "pert_id": ["D1", "D2", "D3"],
        "inchi_key": [_valid_inchikey("A"), _valid_inchikey("B"), _valid_inchikey("C")],
    })
    source = pd.DataFrame({
        "pert_id": ["D1", "D2", "D3", "D4"],
        "inchi_key": [_valid_inchikey("A"), _valid_inchikey("B"),
                      _valid_inchikey("D"), _valid_inchikey("E")],
        "drug_node_idx": [0, 1, 2, 3],
    })
    result = validate_drug_feature_index(
        target, source, (4, 122, 514), (4, 256),
        hidden_size=256, expected_shared_count=2,
    )
    assert result.exact_shared_pert_ids == frozenset({"D1", "D2"})
    assert result.node_index_by_pert_id == {"D1": 0, "D2": 1}
    assert result.n_structure_mismatches == 1

    truncated = source.copy()
    truncated.loc[0, "inchi_key"] = "AAAAAAAAAAAAAA-BBBBBBBBBB"
    with pytest.raises(ContractError, match="incomplete InChIKey"):
        validate_drug_feature_index(
            target, truncated, (4, 122, 514), (4, 256),
            hidden_size=256, expected_shared_count=2,
        )
    with pytest.raises(ContractError, match="complete encoded drug axis"):
        validate_drug_feature_index(
            target, source, (4, 122, 514), (5, 256),
            hidden_size=256, expected_shared_count=2,
        )
    with pytest.raises(ContractError, match="complete encoded feature axis"):
        validate_drug_feature_index(
            target, source.iloc[:3].copy(), (4, 122, 514), (4, 256),
            hidden_size=256, expected_shared_count=2,
        )


def test_condition_key_exposure_and_matrix_axes_must_align():
    conditions = pd.DataFrame({
        "condition_id": [0, 1],
        "cell_id": ["C1", "C2"],
        "pert_id": ["D1", "D2"],
        "pert_dose": [0.5, 1.0],
        "pert_dose_unit": ["um", "um"],
        "pert_time": [6, 24],
        "pert_time_unit": ["h", "h"],
    })
    exposure = pd.DataFrame({
        "condition_id": [0, 1],
        "log10_dose_um": np.log10([0.5, 1.0]),
        "time_hours": [6, 24],
    })
    drugs = pd.DataFrame({
        "drug_index": [0, 1],
        "pert_id": ["D1", "D2"],
        "inchi_key": [_valid_inchikey("A"), _valid_inchikey("B")],
    })
    validate_key_axes(
        conditions, exposure, drugs,
        delta_shape=(2, 978), control_shape=(2, 978), morgan_shape=(2, 2048),
    )
    with pytest.raises(ContractError, match="condition exposure key order"):
        validate_key_axes(
            conditions, exposure.iloc[::-1].reset_index(drop=True), drugs,
            delta_shape=(2, 978), control_shape=(2, 978), morgan_shape=(2, 2048),
        )
    with pytest.raises(ContractError, match="condition/gene axes"):
        validate_key_axes(
            conditions, exposure, drugs,
            delta_shape=(2, 977), control_shape=(2, 978), morgan_shape=(2, 2048),
        )


def test_outer_test_response_access_is_rejected_before_any_array_read():
    loader = object.__new__(FoldDataLoader)
    rows = FoldRows(role="outer_test", positions=np.asarray([0], dtype=np.int64))
    with pytest.raises(ContractError, match="response values are locked"):
        loader.batch(rows, np.asarray([0]), device=torch.device("cpu"))


def test_fold_rows_limit_outer_test_and_inner_validation_to_training_seen_drugs():
    split_fold = validate_split_payload(_synthetic_split(), fold_index=0)
    first_inner = split_fold.inner_folds[0]
    train_cell = sorted(first_inner.train_ids)[0]
    validation_cell = sorted(first_inner.validation_ids)[0]
    outer_test_cell = sorted(split_fold.outer_test_ids)[0]
    loader = object.__new__(FoldDataLoader)
    cells = sorted(split_fold.outer_train_ids)
    records = [{"cell_id": cell, "pert_id": "D1"} for cell in cells]
    records.append({"cell_id": train_cell, "pert_id": "D2"})
    records.append({"cell_id": validation_cell, "pert_id": "D3"})
    records.extend([
        {"cell_id": outer_test_cell, "pert_id": "D1"},
        {"cell_id": outer_test_cell, "pert_id": "D4"},
    ])
    loader.keys = pd.DataFrame(records)
    loader.drug_index = type("Index", (), {
        "exact_shared_pert_ids": frozenset({"D1", "D2", "D3", "D4"}),
    })()
    loader.split_fold = split_fold

    rows = loader.fold_rows(shared_drug_ids=frozenset({"D1", "D2", "D3", "D4"}))
    assert loader.keys.iloc[rows["outer_test"].positions]["pert_id"].tolist() == ["D1"]
    validation_drugs = loader.keys.iloc[
        rows["inner_folds"][0]["validation"].positions]["pert_id"].tolist()
    assert len(validation_drugs) == 19
    assert set(validation_drugs) == {"D1"}


def test_official_loss_adapter_uses_all_four_explicit_weighted_terms():
    treatment_true = torch.zeros((2, 3))
    control_true = torch.ones((2, 3))
    delta_true = treatment_true - control_true
    outputs = (
        torch.ones((2, 3)),
        torch.zeros((2, 3)),
        torch.zeros((2, 3)),
        treatment_true,
        control_true,
    )

    def official_mse(pred, target):
        return torch.mean((pred - target) ** 2, dim=1).sum()

    def official_pcc(pred, target):
        return torch.tensor(2.0)

    loss, terms = build_official_loss(
        outputs, (0.2, 0.003, 0.2, 1.0), official_mse, official_pcc)
    expected = 0.2 * 2 + 0.003 * 2 + 0.2 * 2 + 1.0 * 2
    assert loss.item() == pytest.approx(expected)
    assert set(terms) == {"treatment_mse", "control_mse", "delta_mse", "delta_pearson"}


def test_formal_training_plan_has_no_implicit_candidates_or_budget():
    with pytest.raises(ContractError, match="missing required fields"):
        validate_formal_training_plan({})
    valid = _formal_plan()
    validate_formal_training_plan(valid)
    over_budget = _formal_plan()
    over_budget["per_run_wallclock_hours"] = 20.1
    with pytest.raises(ContractError, match="at most 20 hours"):
        validate_formal_training_plan(over_budget)
    incomplete_candidate = _formal_plan()
    incomplete_candidate["candidates"][0].pop("patience")
    with pytest.raises(ContractError, match="candidate is missing"):
        validate_formal_training_plan(incomplete_candidate)
