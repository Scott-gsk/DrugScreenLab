"""Synthetic contract tests for the EXP-066 response predictor."""

from __future__ import annotations

import inspect
import numpy as np
import pandas as pd
import pytest

import scripts.modeling.run_exp066_response_prediction as exp066_runner
from drug_screen.models.exp066_response_prediction import (
    FoldFeatureEncoder,
    FoldFeatureSpec,
    ReducedRankRidge,
    build_fold_local_shuffled_cell_ids,
    generate_nested_oof_response_features,
    per_drug_training_mean,
    validate_object_shuffle,
)
from scripts.modeling.run_exp066_response_prediction import (
    _object_equal_weighted_variance,
    _per_condition_metrics,
    _normalize_condition_metadata,
    _dataset_inputs,
)


def test_feature_encoder_uses_train_rows_and_fixed_dimensions() -> None:
    rng = np.random.default_rng(610066)
    rna = rng.normal(size=(18, 978))
    drug = rng.integers(0, 2, size=(18, 2048)).astype(np.float64)
    exposure = rng.normal(size=(36, 2))
    cell_index = np.repeat(np.arange(18), 2)
    drug_index = np.tile(np.arange(18), 2)
    train_rows = np.arange(32)

    encoder = FoldFeatureEncoder(FoldFeatureSpec()).fit(
        rna, drug, cell_index, drug_index, exposure, train_rows,
    )
    candidate = encoder.transform_candidate(rna, drug, cell_index, drug_index, exposure)
    no_state = encoder.transform_no_state(drug, drug_index, exposure)

    assert candidate.shape == (36, 3284)
    assert no_state.shape == (36, 3284)
    assert np.isfinite(candidate).all()
    assert np.isfinite(no_state).all()
    np.testing.assert_allclose(encoder.rna_scaler.mean_, rna[:16].mean(axis=0))
    np.testing.assert_allclose(encoder.exposure_scaler.mean_, exposure[train_rows].mean(axis=0))
    # Held-out RNA and exposure rows are present in the transform but did not
    # change the fitted training-side means.
    assert not np.allclose(encoder.rna_scaler.mean_, rna.mean(axis=0))


def test_reduced_rank_ridge_dual_and_primal_solutions_are_rank_limited() -> None:
    rng = np.random.default_rng(660066)
    x_dual = rng.normal(size=(28, 52))
    y_dual = rng.normal(size=(28, 24))
    dual_model = ReducedRankRidge(rank=16, alpha=1.0).fit(x_dual, y_dual)
    assert dual_model.coefficient is not None
    assert np.linalg.matrix_rank(dual_model.coefficient, tol=1e-9) <= 16
    assert dual_model.predict(x_dual).shape == y_dual.shape

    x_primal = rng.normal(size=(64, 24))
    y_primal = rng.normal(size=(64, 20))
    primal_model = ReducedRankRidge(rank=16, alpha=1.0).fit(x_primal, y_primal)
    assert primal_model.coefficient is not None
    assert np.linalg.matrix_rank(primal_model.coefficient, tol=1e-9) <= 16
    assert np.isfinite(primal_model.predict(x_primal)).all()


def test_patient_equal_variance_keeps_within_object_condition_variation() -> None:
    y = np.asarray([[0.0], [2.0], [10.0]])
    objects = np.asarray(["A", "A", "B"])
    variance = _object_equal_weighted_variance(y, objects)
    # Each object receives total weight 1/2; A's two conditions split its
    # weight, while B's single condition receives its full object weight.
    np.testing.assert_allclose(variance, [20.75])


def test_condition_metrics_keep_ties_and_undefined_predictions_explicit() -> None:
    truth = np.asarray([
        [0.0, 1.0, -1.0, 2.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    prediction = np.asarray([
        [0.0, 0.0, -2.0, 3.0],
        [0.5, 0.5, 0.5, 0.5],
        [np.nan, np.nan, np.nan, np.nan],
    ])
    variance = np.asarray([2.0, 1e-8, 4.0, 0.0])
    result = _per_condition_metrics(truth, prediction, variance, 1e-8)

    assert result["valid_target_gene_count"].tolist() == [2, 2, 2]
    assert result["excluded_target_gene_count"].tolist() == [2, 2, 2]
    assert result["directional_agreement"][0] == pytest.approx(0.75)
    assert result["nonzero_directional_agreement"][0] == pytest.approx(1.0)
    assert np.isnan(result["spearman"][1])
    assert np.isnan(result["raw_mse"][2])
    assert np.isnan(result["directional_agreement"][2])


def test_fold_local_shuffle_is_seeded_and_partition_local() -> None:
    cell_ids = np.asarray([f"cell_{patient}_{rep}" for patient in range(10) for rep in range(2)])
    object_ids = np.asarray([f"patient_{patient}" for patient in range(10) for _ in range(2)])
    fold_ids = np.asarray([patient // 2 for patient in range(10) for _ in range(2)], dtype=np.int64)

    shuffled = build_fold_local_shuffled_cell_ids(cell_ids, object_ids, fold_ids, seed=660067)
    repeated = build_fold_local_shuffled_cell_ids(cell_ids, object_ids, fold_ids, seed=660067)
    assert shuffled.shape == (5, len(cell_ids))
    np.testing.assert_array_equal(shuffled, repeated)
    for fold in range(5):
        donor_to_object = {cell: obj for cell, obj in zip(cell_ids, object_ids)}
        donor_objects = np.asarray([donor_to_object[cell] for cell in shuffled[fold]])
        validate_object_shuffle(
            object_ids,
            donor_objects,
            np.where(fold_ids == fold, "heldout", "train"),
        )
        assert np.all(donor_objects[fold_ids == fold] != object_ids[fold_ids == fold])
        assert np.all(donor_objects[fold_ids != fold] != object_ids[fold_ids != fold])


def test_loader_metadata_aliases_follow_shared_contract_names() -> None:
    frame = pd.DataFrame({
        "cell_id": ["C1"],
        "target_axis_sha256": ["abc"],
        "log10_dose_uM": [-1.0],
        "log10_time_hours": [1.0],
    })
    normalized = _normalize_condition_metadata(frame)
    assert normalized.loc[0, "lincs_cell_id"] == "C1"
    assert normalized.loc[0, "target_axis_hash"] == "abc"
    assert normalized.loc[0, "log10_dose_um"] == -1.0


def test_metadata_gate_is_checked_before_target_payload_access() -> None:
    class NoTargetRead(dict):
        def __getitem__(self, key: str):
            if key == "delta_by_condition":
                raise AssertionError("target values must not be accessed before metadata PASS")
            return super().__getitem__(key)

    dataset = NoTargetRead(metadata_gate_status="PENDING")
    config = {"metadata_gate_status": "PASS", "metadata_gate_receipt_sha256": "digest"}
    with pytest.raises(ValueError, match="metadata gate PASS"):
        _dataset_inputs(dataset, config)


def _synthetic_domain_with_unreferenced_morgan_rows() -> tuple[dict, dict]:
    key = "AAAAAAAAAAAAAA-BBBBBBBBBB-C"
    metadata = pd.DataFrame({
        "condition_id": [f"C{index}" for index in range(5)],
        "lincs_cell_id": [f"CELL{index}" for index in range(5)],
        "depmap_model_id": [f"MODEL{index}" for index in range(5)],
        "source_object_id": [f"PATIENT{index}" for index in range(5)],
        "fold_id": np.arange(5),
        "rna_row_index": np.arange(5),
        "drug_feature_row_index": np.zeros(5, dtype=np.int64),
        "response_row_index": np.arange(5),
        "pert_id": [f"PERT{index}" for index in range(5)],
        "full_inchikey": [key] * 5,
        "canonical_smiles": ["C"] * 5,
        "source_treatment_ids": [f"TRT{index}" for index in range(5)],
        "source_control_ids": [f"CTL{index}" for index in range(5)],
        "target_axis_hash": ["axis-hash"] * 5,
        "log10_dose_um": [-1.0] * 5,
        "log10_time_hours": [1.0] * 5,
    })
    dataset = {
        "metadata_gate_status": "PASS",
        "metadata_gate_receipt_sha256": "receipt-hash",
        "condition_metadata": metadata,
        "rna_by_model": np.zeros((5, 978), dtype=np.float32),
        # Rows 1 and 2 are frozen response-blind structure keys without an admitted condition.
        "drug_morgan_by_key": np.zeros((3, 2048), dtype=np.float32),
        "delta_by_condition": np.zeros((5, 978), dtype=np.float32),
        "target_gene_ids": [f"G{index:03d}" for index in range(978)],
        "target_axis_hash": "axis-hash",
        "shuffled_rna_row_by_fold": np.zeros((5, 5), dtype=np.int64),
    }
    config = {
        "metadata_gate_status": "PASS",
        "metadata_gate_receipt_sha256": "receipt-hash",
        "target_axis_hash": "axis-hash",
    }
    return dataset, config


def test_dataset_input_accepts_two_unreferenced_morgan_rows() -> None:
    dataset, config = _synthetic_domain_with_unreferenced_morgan_rows()
    normalized, rna, drug, _, indices, *_ = _dataset_inputs(dataset, config)
    assert len(normalized) == 5
    assert rna.shape == (5, 978)
    assert drug.shape == (3, 2048)
    assert set(indices[1].tolist()) == {0}


def test_dataset_input_rejects_out_of_range_drug_row() -> None:
    dataset, config = _synthetic_domain_with_unreferenced_morgan_rows()
    dataset["condition_metadata"].loc[0, "full_inchikey"] = "CCCCCCCCCCCCCC-DDDDDDDDDD-E"
    dataset["condition_metadata"].loc[0, "drug_feature_row_index"] = 3
    dataset["condition_metadata"].loc[0, "canonical_smiles"] = "CC"
    with pytest.raises(ValueError, match="condition row index exceeds an input feature axis"):
        _dataset_inputs(dataset, config)


def test_dataset_input_rejects_one_key_rebound_to_multiple_morgan_rows() -> None:
    dataset, config = _synthetic_domain_with_unreferenced_morgan_rows()
    dataset["condition_metadata"].loc[1, "drug_feature_row_index"] = 1
    with pytest.raises(ValueError, match="drug feature rows and complete InChIKeys must map one-to-one"):
        _dataset_inputs(dataset, config)


def test_missing_drug_mean_stays_undefined_when_requested() -> None:
    y_train = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    train_drugs = np.asarray([0, 0], dtype=np.int64)
    predictions = per_drug_training_mean(
        y_train,
        train_drugs,
        np.asarray([0, 1], dtype=np.int64),
        allow_missing=True,
    )
    np.testing.assert_allclose(predictions[0], [2.0, 3.0])
    assert np.isnan(predictions[1]).all()


def test_nested_response_api_has_pair_metadata_parameter() -> None:
    assert list(inspect.signature(generate_nested_oof_response_features).parameters) == [
        "domain", "train_patient_ids", "predict_patient_ids", "outer_fold_id",
        "config", "prediction_metadata",
    ]


def test_nested_pair_api_keeps_pair_order_and_unmatched_cell_key_rows(monkeypatch) -> None:
    patient_ids = [f"P{index:02d}" for index in range(10)]
    keys = ["AAAAAAAAAAAAAA-BBBBBBBBBB-C", "CCCCCCCCCCCCCC-DDDDDDDDDD-E"]
    conditions = []
    pairs = []
    condition_id = 0
    for patient_index, patient in enumerate(patient_ids):
        cell = f"LINCS_{patient}"
        raw_row = patient_index + 100
        fold = patient_index % 5
        for drug_row, key in enumerate(keys):
            exposure = (-1.0 + drug_row, 1.0)
            if not (patient == "P09" and drug_row == 1):
                conditions.append({
                    "condition_id": f"cond_{condition_id}",
                    "lincs_cell_id": cell,
                    "depmap_model_id": f"MODEL_{patient}",
                    "source_object_id": patient,
                    "fold_id": fold,
                    "rna_row_index": patient_index,
                    "depmap_source_row_index": raw_row,
                    "drug_feature_row_index": drug_row,
                    "response_row_index": condition_id,
                    "pert_id": f"PERT_{drug_row}",
                    "full_inchikey": key,
                    "canonical_smiles": f"C{drug_row}",
                    "source_treatment_ids": f"TRT_{condition_id}",
                    "source_control_ids": f"CTL_{condition_id}",
                    "target_axis_hash": "axis-hash",
                    "log10_dose_um": exposure[0],
                    "log10_time_hours": exposure[1],
                })
                condition_id += 1
            pairs.append({
                "source_object_id": patient,
                "lincs_cell_id": cell,
                "rna_row_index": raw_row,
                "full_inchikey": key,
                "drug_feature_row_index": drug_row,
                "log10_dose_um": exposure[0],
                "log10_time_hours": exposure[1],
                "pair_id": f"pair::{patient}::{drug_row}",
            })
    metadata = pd.DataFrame(conditions)
    pair_metadata = pd.DataFrame(pairs)
    rna = np.zeros((len(patient_ids), 978), dtype=np.float32)
    drug = np.zeros((len(keys), 2048), dtype=np.float32)
    targets = np.zeros((len(metadata), 978), dtype=np.float32)
    exposure = metadata[["log10_dose_um", "log10_time_hours"]].to_numpy(dtype=np.float64)
    index_arrays = np.stack((metadata["rna_row_index"], metadata["drug_feature_row_index"])).astype(np.int64)
    gene_ids = [f"G{index:03d}" for index in range(978)]

    monkeypatch.setattr(
        exp066_runner, "_normalize_frozen_config",
        lambda config, domain: {**config, "experiment_id": "EXP-066", "data_load_cpu_seconds": 0.0},
    )
    monkeypatch.setattr(
        exp066_runner, "_dataset_inputs",
        lambda domain, config: (
            metadata.copy(), rna, drug, targets, index_arrays, gene_ids, exposure,
            np.zeros((5, len(metadata)), dtype=np.int64), np.empty((5, len(metadata)), dtype=str),
        ),
    )

    def fake_fit(metadata_arg, rna_arg, drug_arg, y_arg, exposure_arg, train_rows,
                 predict_pairs, train_shuffle, predict_shuffle):
        n = len(predict_pairs)
        return (
            np.full((n, 978), 0.25, dtype=np.float32),
            np.full((n, 978), -0.25, dtype=np.float32),
            {
                "normal_model": {"rank": 16, "alpha": 1.0},
                "shuffled_model": {"rank": 16, "alpha": 1.0},
                "training_patient_ids": sorted(set(metadata_arg.iloc[train_rows]["source_object_id"])),
                "prediction_patient_ids": sorted(set(predict_pairs["source_object_id"])),
                "training_condition_count": len(train_rows),
                "prediction_pair_count": n,
            },
        )

    monkeypatch.setattr(exp066_runner, "_fit_candidate_and_shuffle_pairs", fake_fit)
    domain = {
        "metadata_gate_status": "PASS",
        "metadata_gate_receipt_sha256": "receipt-hash",
    }
    result = exp066_runner.generate_nested_oof_response_features(
        domain,
        [patient for patient in patient_ids if int(patient[1:]) % 5 != 0],
        [patient for patient in patient_ids if int(patient[1:]) % 5 == 0],
        0,
        {"metadata_gate_status": "PASS", "metadata_gate_receipt_sha256": "receipt-hash"},
        pair_metadata,
    )

    expected_train_ids = pair_metadata.loc[
        pair_metadata["source_object_id"].map(
            metadata.drop_duplicates("source_object_id").set_index("source_object_id")["fold_id"]
        ).ne(0), "pair_id",
    ].astype(str).tolist()
    expected_predict_ids = pair_metadata.loc[
        pair_metadata["source_object_id"].map(
            metadata.drop_duplicates("source_object_id").set_index("source_object_id")["fold_id"]
        ).eq(0), "pair_id",
    ].astype(str).tolist()
    assert result["train"]["pair_id"].tolist() == expected_train_ids
    assert result["predict"]["pair_id"].tolist() == expected_predict_ids
    assert result["cost"]["input_pair_rows"] == len(pair_metadata)
    assert result["train"]["prediction"].shape == (len(expected_train_ids), 978)
    assert result["train"]["shuffled_prediction"].shape == (len(expected_train_ids), 978)
    assert result["predict"]["prediction"].shape == (len(expected_predict_ids), 978)
    # P09/key1 has no measured LINCS condition but remains in pair-ordered output.
    unmatched_pair = "pair::P09::1"
    assert unmatched_pair in result["train"]["pair_id"]
    assert result["train"]["nested_fold_id"].tolist() == pair_metadata.loc[
        pair_metadata["source_object_id"].map(
            metadata.drop_duplicates("source_object_id").set_index("source_object_id")["fold_id"]
        ).ne(0), "source_object_id",
    ].map(
        metadata.drop_duplicates("source_object_id").set_index("source_object_id")["fold_id"]
    ).to_numpy().tolist()
    heldout_patients = set(result["predict_patient_ids"])
    assert result["outer_train_patient_ids"] == result["train_patient_ids"]
    assert result["outer_predict_patient_ids"] == result["predict_patient_ids"]
    for fold in result["train"]["model_provenance"]["folds"]:
        assert not heldout_patients.intersection(fold["training_patient_ids"])
        assert set(fold["prediction_patient_ids"]).isdisjoint(fold["training_patient_ids"])
    assert all(source.startswith("outer_0_inner_") for source in result["train"]["model_source"])
    assert set(result["predict"]["model_provenance"]["training_patient_ids"]) == set(
        result["outer_train_patient_ids"]
    )
    assert set(result["predict"]["model_provenance"]["prediction_patient_ids"]) == heldout_patients
    train_objects = result["train"]["source_object_id"]
    train_donors = result["train"]["shuffle_donor_object_id"]
    assert np.all(train_objects != train_donors)
