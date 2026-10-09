import io
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from scripts.evaluation import score_exp044_prism_pharmaformer as scoring

paired_statistics = scoring.paired_statistics
score_cell_pairs = scoring.score_cell_pairs


def test_scores_by_cell_after_exact_key_join_and_retains_pairing():
    predictions = pd.DataFrame([
        ("cell-a", "drug-1", 3.0, 2.0),
        ("cell-a", "drug-2", 2.0, 1.0),
        ("cell-a", "drug-3", 1.0, 3.0),
        ("cell-b", "drug-1", 1.0, 2.0),
        ("cell-b", "drug-2", 2.0, 1.0),
        ("cell-b", "drug-3", 3.0, 3.0),
    ], columns=[
        "cell_line_id", "drug_id", "predicted_source_pharmaformer",
        "predicted_source_train_mean",
    ])
    responses = pd.DataFrame([
        ("cell-a", "drug-1", 3.0),
        ("cell-a", "drug-2", 2.0),
        ("cell-a", "drug-3", 1.0),
        ("cell-b", "drug-1", 1.0),
        ("cell-b", "drug-2", 2.0),
        ("cell-b", "drug-3", 3.0),
    ], columns=["sample_id", "drug_id", "response"])

    scores, coverage = score_cell_pairs(
        predictions,
        responses,
        {"cell-a", "cell-b"},
        {"drug-1", "drug-2", "drug-3"},
        expected_target_keys=6,
        expected_eligible_cells=2,
    )

    assert scores["cell_line_id"].tolist() == ["cell-a", "cell-b"]
    assert scores["n_matched_drugs"].tolist() == [3, 3]
    assert np.allclose(scores["pharmaformer_spearman"], [1.0, 1.0])
    assert coverage["n_cells_with_at_least_three_drugs"] == 2
    assert coverage["n_cells_with_defined_paired_spearman"] == 2


def test_pair_statistics_report_percentile_interval_and_finite_p_value():
    result = paired_statistics(np.array([0.1, 0.2, 0.3, 0.4]),
                               bootstrap_draws=200, sign_flip_draws=500)

    assert result["n_paired_cells"] == 4
    assert result["mean_paired_spearman_difference"] == pytest.approx(0.25)
    assert result["bootstrap_95pct_interval"][0] <= 0.25
    assert result["bootstrap_95pct_interval"][1] >= 0.25
    assert 0 < result["sign_flip_two_sided_p"] <= 1


def test_constant_model_output_is_left_undefined_instead_of_set_to_zero():
    predictions = pd.DataFrame([
        ("cell-a", "drug-1", 1.0, 2.0),
        ("cell-a", "drug-2", 1.0, 1.0),
        ("cell-a", "drug-3", 1.0, 3.0),
    ], columns=[
        "cell_line_id", "drug_id", "predicted_source_pharmaformer",
        "predicted_source_train_mean",
    ])
    responses = pd.DataFrame([
        ("cell-a", "drug-1", 3.0),
        ("cell-a", "drug-2", 2.0),
        ("cell-a", "drug-3", 1.0),
    ], columns=["sample_id", "drug_id", "response"])

    scores, coverage = score_cell_pairs(
        predictions, responses, {"cell-a"}, {"drug-1", "drug-2", "drug-3"},
        expected_target_keys=3, expected_eligible_cells=1,
    )

    assert np.isnan(scores.loc[0, "pharmaformer_spearman"])
    assert coverage["n_cells_with_defined_paired_spearman"] == 0
    assert coverage["undefined_metric_cells"] == ["cell-a"]


def test_pair_join_rejects_unmatched_keys():
    predictions = pd.DataFrame([
        ("cell-a", "drug-1", 1.0, 2.0),
    ], columns=[
        "cell_line_id", "drug_id", "predicted_source_pharmaformer",
        "predicted_source_train_mean",
    ])
    responses = pd.DataFrame([
        ("cell-a", "drug-2", 2.0),
    ], columns=["sample_id", "drug_id", "response"])

    with pytest.raises(ValueError, match="no frozen prediction"):
        score_cell_pairs(
            predictions, responses, {"cell-a"}, {"drug-1", "drug-2"},
            expected_target_keys=1, expected_eligible_cells=0,
        )


def _small_key_audit(tmp_path):
    frame = pd.DataFrame({
        "sample_id": ["cell-a", "cell-a", "cell-a"],
        "drug_id": ["drug-1", "drug-2", "drug-3"],
    })
    normalized = frame.rename(columns={"sample_id": "cell_line_id"})
    source_path = tmp_path / "pairs.parquet"
    source_path.write_bytes(b"fixture parquet bytes")
    snapshot = scoring._snapshot_file(source_path)
    audit = {
        "source_pair_table_cell_identity_column": "sample_id",
        "source_pair_table_columns_requested": ["sample_id", "drug_id"],
        "source_pair_table_opened_for_keys_only": True,
        "source_pair_table_cell_identity_is_verified_sample_axis": True,
        "response_value_columns_requested": [],
        "target_response_values_read": False,
        "target_response_values_loaded": False,
        "source_pair_table_file_sha256": snapshot.sha256,
        "n_source_pair_key_rows": 3,
        "n_unique_source_pair_keys": 3,
        "n_revised_cohort_panel_cell_drug_keys": 3,
        "source_pair_table_key_rows_sha256": scoring._canonical_key_sha256(
            normalized, unique=False,
        ),
        "source_pair_table_unique_keys_sha256": scoring._canonical_key_sha256(normalized),
        "response_blind_cohort_key_sha256": scoring._canonical_key_sha256(normalized),
        "eligible_cell_ids": ["cell-a"],
        "ineligible_cell_ids": [],
        "unique_panel_drugs_per_cell": {"cell-a": 3},
    }
    return frame, source_path, audit, snapshot


def test_changed_response_keys_fail_before_value_projection(tmp_path, monkeypatch):
    expected, source_path, audit, snapshot = _small_key_audit(tmp_path)
    corrupt = expected.copy()
    corrupt.loc[2, "drug_id"] = "drug-other"
    monkeypatch.setattr(scoring, "EXPECTED_TARGET_KEYS", 3)
    monkeypatch.setattr(scoring, "EXPECTED_ELIGIBLE_CELLS", 1)
    requested_columns = []

    def fake_read_parquet(path, *, columns=None, **kwargs):
        assert isinstance(path, io.BytesIO)
        requested_columns.append(columns)
        if "response" in columns:
            raise AssertionError("response values opened before the key mismatch was rejected")
        return corrupt.copy()

    monkeypatch.setattr(scoring.pd, "read_parquet", fake_read_parquet)
    with pytest.raises(ValueError, match="frozen response-blind key audit"):
        scoring._load_response_values_after_key_gate(
            snapshot, audit, {"cell-a"}, {"drug-1", "drug-2", "drug-3"},
        )
    assert requested_columns == [["sample_id", "drug_id"]]


def test_response_file_digest_mismatch_fails_before_value_projection(tmp_path, monkeypatch):
    expected, source_path, audit, snapshot = _small_key_audit(tmp_path)
    audit["source_pair_table_file_sha256"] = "0" * 64
    monkeypatch.setattr(scoring, "EXPECTED_TARGET_KEYS", 3)
    monkeypatch.setattr(scoring, "EXPECTED_ELIGIBLE_CELLS", 1)
    requested_columns = []

    def fake_read_parquet(path, *, columns=None, **kwargs):
        assert isinstance(path, io.BytesIO)
        requested_columns.append(columns)
        if "response" in columns:
            raise AssertionError("response values opened before the source digest was rejected")
        return expected.copy()

    monkeypatch.setattr(scoring.pd, "read_parquet", fake_read_parquet)
    with pytest.raises(ValueError, match="source file bytes"):
        scoring._load_response_values_after_key_gate(
            snapshot, audit, {"cell-a"}, {"drug-1", "drug-2", "drug-3"},
        )
    assert requested_columns == [["sample_id", "drug_id"]]


def test_response_values_are_parsed_from_the_keys_only_snapshot_after_path_change(
    tmp_path, monkeypatch,
):
    keys, source_path, audit, snapshot = _small_key_audit(tmp_path)
    responses = keys.assign(response=[0.3, 0.2, 0.1])
    original_bytes = snapshot.content
    requested = []

    def fake_read_parquet(source, *, columns=None, **kwargs):
        assert isinstance(source, io.BytesIO)
        assert source.getvalue() == original_bytes
        requested.append(columns)
        if "response" not in columns:
            source_path.write_bytes(b"changed after the response-key gate")
            return keys.loc[:, columns].copy()
        return responses.loc[:, columns].copy()

    monkeypatch.setattr(scoring, "EXPECTED_TARGET_KEYS", 3)
    monkeypatch.setattr(scoring, "EXPECTED_ELIGIBLE_CELLS", 1)
    monkeypatch.setattr(scoring.pd, "read_parquet", fake_read_parquet)

    loaded, gated = scoring._load_response_values_after_key_gate(
        snapshot, audit, {"cell-a"}, {"drug-1", "drug-2", "drug-3"},
    )

    assert source_path.read_bytes() != original_bytes
    assert requested == [["sample_id", "drug_id"], ["sample_id", "drug_id", "response"]]
    assert loaded["response"].tolist() == [0.3, 0.2, 0.1]
    assert scoring._canonical_key_sha256(gated) == audit["response_blind_cohort_key_sha256"]


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _small_scoring_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(scoring, "EXPECTED_CELLS", 2)
    monkeypatch.setattr(scoring, "EXPECTED_DRUGS", 3)
    monkeypatch.setattr(scoring, "EXPECTED_TARGET_KEYS", 6)
    monkeypatch.setattr(scoring, "EXPECTED_ELIGIBLE_CELLS", 2)
    prediction_dir = tmp_path / "prediction_v3"
    input_dir = tmp_path / "input_v3"
    prediction_dir.mkdir()
    input_dir.mkdir()
    prediction_path = prediction_dir / "predictions.parquet"
    prediction_summary_path = prediction_dir / "prediction_summary.json"
    response_path = input_dir / "pairs.parquet"
    key_audit_path = tmp_path / "key_audit_v3.json"
    numeric_path = tmp_path / "numeric_v3.json"
    verification_path = tmp_path / "verification_v3.json"
    identity_path = input_dir / "identity_resolution.json"
    manifest_path = input_dir / "manifest.json"
    cell_axis_path = input_dir / "cell_axis.json"
    drug_axis_path = input_dir / "drug_axis.parquet"
    strict_path = input_dir / "strict_unseen_cell_lines_v2.csv"

    cells = ["cell-a", "cell-b"]
    drugs = ["drug-1", "drug-2", "drug-3"]
    prediction_rows = []
    response_rows = []
    for cell, model_values, baseline_values, response_values in (
        ("cell-a", [3.0, 2.0, 1.0], [2.0, 1.0, 3.0], [3.0, 2.0, 1.0]),
        ("cell-b", [1.0, 2.0, 3.0], [2.0, 3.0, 1.0], [1.0, 2.0, 3.0]),
    ):
        for drug, model, baseline, response in zip(
                drugs, model_values, baseline_values, response_values):
            prediction_rows.append((cell, drug, model, baseline))
            response_rows.append((cell, drug, response))
    predictions = pd.DataFrame(prediction_rows, columns=[
        "cell_line_id", "drug_id", "predicted_source_pharmaformer",
        "predicted_source_train_mean",
    ])
    responses = pd.DataFrame(response_rows, columns=["sample_id", "drug_id", "response"])
    predictions.to_parquet(prediction_path, index=False)
    responses.to_parquet(response_path, index=False)
    _write_json(cell_axis_path, cells)
    pd.DataFrame({"drug_id": drugs}).to_parquet(drug_axis_path, index=False)
    pd.DataFrame({"cell_line_id": cells}).to_csv(strict_path, index=False)
    manifest = {
        "experiment_id": "EXP-044",
        "scientific_strict_unseen_accepted": True,
        "target_response_values_read": False,
        "response_label_files_opened": [],
    }
    identity = {
        "experiment_id": "EXP-044",
        "status": "RESOLVED_ORTHOGONAL_AUTHORITATIVE_CROSSWALK",
        "strict_unseen_accepted": True,
        "target_response_values_read": False,
    }
    _write_json(manifest_path, manifest)
    _write_json(identity_path, identity)
    verification = {
        "experiment_id": "EXP-044",
        "status": "PASS_RESPONSE_BLIND_IDENTITY_RESOLVED",
        "feature_verification_status": "PASS",
        "input_manifest_sha256": scoring.sha256(manifest_path),
    }
    _write_json(verification_path, verification)

    normalized = responses.rename(columns={"sample_id": "cell_line_id"})[
        ["cell_line_id", "drug_id"]
    ]
    cohort = normalized.drop_duplicates(["cell_line_id", "drug_id"])
    audit = {
        "experiment_id": "EXP-044",
        "status": "PASS_RESPONSE_BLIND_IDENTITY_KEY_COVERAGE",
        "source_pair_table_path": str(response_path.resolve()),
        "source_pair_table_cell_identity_column": "sample_id",
        "source_pair_table_columns_requested": ["sample_id", "drug_id"],
        "source_pair_table_opened_for_keys_only": True,
        "source_pair_table_cell_identity_is_verified_sample_axis": True,
        "source_pair_table_file_sha256": scoring.sha256(response_path),
        "source_pair_table_key_rows_sha256": scoring._canonical_key_sha256(normalized, unique=False),
        "source_pair_table_unique_keys_sha256": scoring._canonical_key_sha256(normalized),
        "response_blind_cohort_key_sha256": scoring._canonical_key_sha256(cohort),
        "n_source_pair_key_rows": 6,
        "n_unique_source_pair_keys": 6,
        "n_revised_cohort_panel_cell_drug_keys": 6,
        "eligible_cell_ids": cells,
        "ineligible_cell_ids": [],
        "unique_panel_drugs_per_cell": {cell: 3 for cell in cells},
        "response_value_columns_requested": [],
        "target_response_values_read": False,
        "target_response_values_loaded": False,
        "input_verification_path": str(verification_path.resolve()),
        "identity_resolution_status": identity["status"],
        "identity_resolution_sha256": scoring.sha256(identity_path),
        "input_verification_sha256": scoring.sha256(verification_path),
        "input_manifest_sha256": scoring.sha256(manifest_path),
        "cell_axis_sha256": scoring.sha256(cell_axis_path),
        "strict_identity_list_sha256": scoring.sha256(strict_path),
        "drug_axis_sha256": scoring.sha256(drug_axis_path),
        "n_revised_strict_unseen_cells": 2,
        "n_prism_panel_drugs": 3,
        "n_cells_with_at_least_three_panel_drugs": 2,
    }
    _write_json(key_audit_path, audit)
    key_audit_hash = scoring.sha256(key_audit_path)
    prediction_summary = {
        "experiment_id": "EXP-044",
        "run_id": prediction_dir.name,
        "status": "PREDICTIONS_FROZEN_NO_PRISM_LABELS_LOADED_PENDING_RESPONSE_EVALUATION",
        "target_response_values_read": False,
        "target_response_files_opened": [],
        "prediction_file_sha256": scoring.sha256(prediction_path),
        "identity_key_audit_sha256": key_audit_hash,
        "n_strict_unseen_cohort_cells": 2,
        "n_prism_panel_drugs": 3,
        "n_prediction_keys": 6,
        "inputs_sha256": {
            "identity_key_coverage_audit": key_audit_hash,
            "input_manifest": scoring.sha256(manifest_path),
        },
    }
    _write_json(prediction_summary_path, prediction_summary)
    numeric = {
        "experiment_id": "EXP-044",
        "run_id": prediction_dir.name,
        "status": "PASS_INDEPENDENT_NUMERICAL_RECOMPUTATION_NO_TARGET_LABELS",
        "target_response_values_read": False,
        "target_response_files_opened": [],
        "comparison_tolerances_frozen_before_target_response_evaluation": True,
        "n_cells": 2,
        "n_drugs": 3,
        "comparison": {
            "n_prediction_keys": 6,
            "model_prediction_absolute_tolerance": scoring.EXPECTED_PREDICTION_ATOL,
            "max_abs_model_prediction_error": 0.0,
            "source_mean_absolute_tolerance": scoring.EXPECTED_BASELINE_ATOL,
            "max_abs_source_mean_error": 0.0,
        },
        "inputs_sha256": {
            "frozen_predictions": scoring.sha256(prediction_path),
            "prediction_summary": scoring.sha256(prediction_summary_path),
            "identity_key_audit": key_audit_hash,
            "input_manifest": scoring.sha256(manifest_path),
            "input_verification": scoring.sha256(verification_path),
            "cell_axis": scoring.sha256(cell_axis_path),
            "drug_axis": scoring.sha256(drug_axis_path),
        },
    }
    _write_json(numeric_path, numeric)
    return {
        "prediction_dir": prediction_dir,
        "input_dir": input_dir,
        "response_path": response_path,
        "prediction_path": prediction_path,
        "key_audit_path": key_audit_path,
        "numeric_path": numeric_path,
        "verification_path": verification_path,
        "manifest_path": manifest_path,
        "identity_path": identity_path,
        "output_dir": tmp_path / "evaluation_test",
        "predictions": predictions,
        "responses": responses,
    }


def test_scoring_uses_the_prediction_bytes_that_passed_the_digest_gate(
    tmp_path, monkeypatch,
):
    paths = _small_scoring_inputs(tmp_path, monkeypatch)
    original_prediction_bytes = paths["prediction_path"].read_bytes()
    original_digest = scoring._snapshot_file(paths["prediction_path"]).sha256
    original_read_parquet = scoring.pd.read_parquet
    mutated = False

    def change_prediction_after_digest_gate(source, *args, **kwargs):
        nonlocal mutated
        from_snapshot = (hasattr(source, "getvalue")
                         and source.getvalue() == original_prediction_bytes)
        from_path = (isinstance(source, (str, Path))
                     and Path(source) == paths["prediction_path"])
        if from_snapshot or from_path:
            paths["prediction_path"].write_bytes(b"changed after prediction digest validation")
            mutated = True
        return original_read_parquet(source, *args, **kwargs)

    monkeypatch.setattr(scoring.pd, "read_parquet", change_prediction_after_digest_gate)
    result = scoring.run(
        prediction_dir=paths["prediction_dir"],
        response_pairs_path=paths["response_path"],
        input_dir=paths["input_dir"],
        identity_key_audit_path=paths["key_audit_path"],
        numeric_recheck_path=paths["numeric_path"],
        output_dir=paths["output_dir"],
    )

    assert mutated
    assert paths["prediction_path"].read_bytes() != original_prediction_bytes
    assert result["inputs_sha256"]["frozen_predictions"] == original_digest
    assert result["inputs_sha256"]["response_pairs"] == scoring.sha256(paths["response_path"])
    summary = json.loads((paths["output_dir"] / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert summary["experiment_id"] == "EXP-044"
    assert summary["run_id"] == paths["prediction_dir"].name
    assert summary["evaluation_id"] == paths["output_dir"].name
    assert summary["outputs_sha256"]["per_cell_scores.parquet"] == scoring.sha256(
        paths["output_dir"] / "per_cell_scores.parquet"
    )
    marker_snapshot = scoring._snapshot_file(
        paths["output_dir"] / scoring.EVALUATION_COMMIT_NAME,
    )
    assert scoring._json_from_snapshot(marker_snapshot) == scoring._evaluation_commit_payload(
        paths["prediction_dir"].name, paths["output_dir"].name,
        scoring.sha256(paths["output_dir"] / "per_cell_scores.parquet"),
        scoring.sha256(paths["output_dir"] / "evaluation_summary.json"),
    )
    assert marker_snapshot.content == scoring._canonical_json_bytes(
        scoring._json_from_snapshot(marker_snapshot),
    )


@pytest.mark.parametrize(("receipt_name", "field", "replacement", "delete_field"), [
    ("prediction_summary", "experiment_id", None, True),
    ("prediction_summary", "experiment_id", "", False),
    ("prediction_summary", "experiment_id", 44, False),
    ("prediction_summary", "experiment_id", "EXP-045", False),
    ("prediction_summary", "run_id", None, True),
    ("prediction_summary", "run_id", "", False),
    ("prediction_summary", "run_id", 44, False),
    ("prediction_summary", "run_id", "another_run", False),
    ("numeric_recheck", "experiment_id", "EXP-045", False),
    ("numeric_recheck", "run_id", None, True),
    ("numeric_recheck", "run_id", "", False),
    ("numeric_recheck", "run_id", 44, False),
    ("numeric_recheck", "run_id", "another_run", False),
    ("key_audit", "experiment_id", "EXP-045", False),
    ("verification", "experiment_id", "EXP-045", False),
    ("manifest", "experiment_id", "EXP-045", False),
    ("identity_resolution", "experiment_id", "EXP-045", False),
])
def test_receipt_identity_errors_fail_before_response_values_are_decoded(
    tmp_path, monkeypatch, receipt_name, field, replacement, delete_field,
):
    paths = _small_scoring_inputs(tmp_path, monkeypatch)
    receipt_paths = {
        "prediction_summary": paths["prediction_dir"] / "prediction_summary.json",
        "numeric_recheck": paths["numeric_path"],
        "key_audit": paths["key_audit_path"],
        "verification": paths["verification_path"],
        "manifest": paths["manifest_path"],
        "identity_resolution": paths["identity_path"],
    }
    receipt_path = receipt_paths[receipt_name]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if delete_field:
        receipt.pop(field)
    else:
        receipt[field] = replacement
    _write_json(receipt_path, receipt)
    paths["response_path"].write_bytes(b"response values must not be decoded")

    def unexpected_response_decode(*args, **kwargs):
        raise AssertionError("response values were decoded before receipt identity validation")

    monkeypatch.setattr(scoring, "_load_response_values_after_key_gate", unexpected_response_decode)
    expected_error = "run_id" if field == "run_id" else "experiment_id"
    with pytest.raises(ValueError, match=expected_error):
        scoring.run(
            prediction_dir=paths["prediction_dir"],
            response_pairs_path=paths["response_path"],
            input_dir=paths["input_dir"],
            identity_key_audit_path=paths["key_audit_path"],
            numeric_recheck_path=paths["numeric_path"],
            output_dir=paths["output_dir"],
        )
    assert not paths["output_dir"].exists()


def test_global_run_id_cannot_replace_missing_numeric_recheck_run_id(tmp_path, monkeypatch):
    paths = _small_scoring_inputs(tmp_path, monkeypatch)
    numeric = json.loads(paths["numeric_path"].read_text(encoding="utf-8"))
    numeric.pop("run_id")
    _write_json(paths["numeric_path"], numeric)
    manifest = json.loads(paths["manifest_path"].read_text(encoding="utf-8"))
    manifest["run_id"] = paths["prediction_dir"].name
    _write_json(paths["manifest_path"], manifest)

    def unexpected_response_decode(*args, **kwargs):
        raise AssertionError("response values decoded before numeric run_id validation")

    monkeypatch.setattr(scoring, "_load_response_values_after_key_gate", unexpected_response_decode)
    with pytest.raises(ValueError, match="numeric recheck run_id"):
        scoring.run(
            prediction_dir=paths["prediction_dir"],
            response_pairs_path=paths["response_path"],
            input_dir=paths["input_dir"],
            identity_key_audit_path=paths["key_audit_path"],
            numeric_recheck_path=paths["numeric_path"],
            output_dir=paths["output_dir"],
        )
    assert not paths["output_dir"].exists()


@pytest.mark.parametrize("failed_name", [
    "per_cell_scores.parquet", "evaluation_summary.json", "evaluation_commit.json",
])
def test_each_exclusive_publication_stage_failure_leaves_no_commit_marker(
    tmp_path, monkeypatch, failed_name,
):
    paths = _small_scoring_inputs(tmp_path, monkeypatch)
    original_write = scoring._write_bytes_exclusive

    def fail_selected_stage(path, content):
        if path.name == failed_name:
            raise OSError(f"simulated exclusive write failure: {failed_name}")
        return original_write(path, content)

    monkeypatch.setattr(scoring, "_write_bytes_exclusive", fail_selected_stage)
    with pytest.raises(OSError, match="simulated exclusive write failure"):
        scoring.run(
            prediction_dir=paths["prediction_dir"],
            response_pairs_path=paths["response_path"],
            input_dir=paths["input_dir"],
            identity_key_audit_path=paths["key_audit_path"],
            numeric_recheck_path=paths["numeric_path"],
            output_dir=paths["output_dir"],
        )
    assert paths["output_dir"].is_dir()
    assert not (paths["output_dir"] / scoring.EVALUATION_COMMIT_NAME).exists()


@pytest.mark.parametrize("failed_name", [
    "per_cell_scores.parquet", "evaluation_summary.json", "evaluation_commit.json",
])
def test_fsync_failure_cleans_partial_file_and_leaves_no_commit_marker(
    tmp_path, monkeypatch, failed_name,
):
    paths = _small_scoring_inputs(tmp_path, monkeypatch)
    original_fsync = scoring._fsync_file

    def fail_selected_stage(stream):
        if Path(stream.name).name == failed_name:
            raise OSError(f"simulated fsync failure: {failed_name}")
        original_fsync(stream)

    monkeypatch.setattr(scoring, "_fsync_file", fail_selected_stage)
    with pytest.raises(OSError, match="simulated fsync failure"):
        scoring.run(
            prediction_dir=paths["prediction_dir"],
            response_pairs_path=paths["response_path"],
            input_dir=paths["input_dir"],
            identity_key_audit_path=paths["key_audit_path"],
            numeric_recheck_path=paths["numeric_path"],
            output_dir=paths["output_dir"],
        )
    assert not (paths["output_dir"] / failed_name).exists()
    assert not (paths["output_dir"] / scoring.EVALUATION_COMMIT_NAME).exists()


@pytest.mark.parametrize("failed_name", [
    "per_cell_scores.parquet", "evaluation_summary.json", "evaluation_commit.json",
])
def test_readback_failure_cleans_partial_file_and_leaves_no_commit_marker(
    tmp_path, monkeypatch, failed_name,
):
    paths = _small_scoring_inputs(tmp_path, monkeypatch)
    original_snapshot = scoring._snapshot_file

    def fail_selected_readback(path):
        if path.name == failed_name:
            raise OSError(f"simulated readback failure: {failed_name}")
        return original_snapshot(path)

    monkeypatch.setattr(scoring, "_snapshot_file", fail_selected_readback)
    with pytest.raises(OSError, match="simulated readback failure"):
        scoring.run(
            prediction_dir=paths["prediction_dir"],
            response_pairs_path=paths["response_path"],
            input_dir=paths["input_dir"],
            identity_key_audit_path=paths["key_audit_path"],
            numeric_recheck_path=paths["numeric_path"],
            output_dir=paths["output_dir"],
        )
    assert not (paths["output_dir"] / failed_name).exists()
    assert not (paths["output_dir"] / scoring.EVALUATION_COMMIT_NAME).exists()


def test_commit_validation_failure_removes_commit_marker(tmp_path, monkeypatch):
    paths = _small_scoring_inputs(tmp_path, monkeypatch)

    def reject_commit(*args, **kwargs):
        raise ValueError("simulated final commit readback validation failure")

    monkeypatch.setattr(scoring, "_validate_evaluation_commit_snapshots", reject_commit)
    with pytest.raises(ValueError, match="final commit readback"):
        scoring.run(
            prediction_dir=paths["prediction_dir"],
            response_pairs_path=paths["response_path"],
            input_dir=paths["input_dir"],
            identity_key_audit_path=paths["key_audit_path"],
            numeric_recheck_path=paths["numeric_path"],
            output_dir=paths["output_dir"],
        )
    assert not (paths["output_dir"] / scoring.EVALUATION_COMMIT_NAME).exists()


def test_evaluation_directory_exclusive_create_refuses_existing_destination(tmp_path):
    output_dir = tmp_path / "evaluation_test"
    output_dir.mkdir()
    sentinel = output_dir / "owner.txt"
    sentinel.write_text("existing owner", encoding="utf-8")
    with pytest.raises(FileExistsError):
        scoring._create_evaluation_dir_exclusive(output_dir)
    assert sentinel.read_text(encoding="utf-8") == "existing owner"


def test_evaluation_directory_creation_failure_writes_no_outputs(tmp_path, monkeypatch):
    paths = _small_scoring_inputs(tmp_path, monkeypatch)

    def fail_create(path):
        raise OSError("simulated exclusive directory creation failure")

    monkeypatch.setattr(scoring, "_create_evaluation_dir_exclusive", fail_create)
    with pytest.raises(OSError, match="directory creation failure"):
        scoring.run(
            prediction_dir=paths["prediction_dir"],
            response_pairs_path=paths["response_path"],
            input_dir=paths["input_dir"],
            identity_key_audit_path=paths["key_audit_path"],
            numeric_recheck_path=paths["numeric_path"],
            output_dir=paths["output_dir"],
        )
    assert not paths["output_dir"].exists()


def test_drvfs_concurrent_directory_and_file_creation_are_exclusive():
    drvfs_tmp = Path("/mnt/d/Code/DrugScreenLab/tmp")
    if not drvfs_tmp.is_dir():
        pytest.skip("requires the configured WSL project path on /mnt/d")
    with tempfile.TemporaryDirectory(prefix="exp044-drvfs-exclusive-test-", dir=drvfs_tmp) as parent:
        root = Path(parent)
        output_dir = root / "evaluation_test"
        directory_barrier = threading.Barrier(2)

        def create_directory():
            directory_barrier.wait()
            try:
                scoring._create_evaluation_dir_exclusive(output_dir)
                return "created"
            except FileExistsError:
                return "exists"

        with ThreadPoolExecutor(max_workers=2) as pool:
            directory_outcomes = list(pool.map(lambda _: create_directory(), (1, 2)))
        assert sorted(directory_outcomes) == ["created", "exists"]

        output_file = output_dir / "evaluation_summary.json"
        file_barrier = threading.Barrier(2)

        def create_file(value):
            file_barrier.wait()
            try:
                scoring._write_bytes_exclusive(output_file, f"value-{value}".encode())
                return "created"
            except FileExistsError:
                return "exists"

        with ThreadPoolExecutor(max_workers=2) as pool:
            file_outcomes = list(pool.map(create_file, (1, 2)))
        assert sorted(file_outcomes) == ["created", "exists"]
        assert output_file.read_bytes() in {b"value-1", b"value-2"}


@pytest.mark.parametrize("operation", ["directory", "file"])
def test_drvfs_independent_processes_race_for_exclusive_create(operation):
    drvfs_tmp = Path("/mnt/d/Code/DrugScreenLab/tmp")
    if not drvfs_tmp.is_dir():
        pytest.skip("requires the configured WSL project path on /mnt/d")
    child_code = r"""
import sys
import time
from pathlib import Path
from scripts.evaluation import score_exp044_prism_pharmaformer as scoring

operation, release_name, target_name, ready_name, value = sys.argv[1:]
release = Path(release_name)
target = Path(target_name)
Path(ready_name).write_text("ready", encoding="utf-8")
deadline = time.monotonic() + 60
while not release.exists():
    if time.monotonic() >= deadline:
        raise TimeoutError("parent did not release concurrent publishers")
    time.sleep(0.005)
try:
    if operation == "directory":
        scoring._create_evaluation_dir_exclusive(target)
    else:
        scoring._write_bytes_exclusive(target, value.encode("utf-8"))
    print("created")
except FileExistsError:
    print("exists")
"""
    with tempfile.TemporaryDirectory(prefix="exp044-drvfs-process-race-", dir=drvfs_tmp) as parent:
        root = Path(parent)
        if operation == "directory":
            target = root / "exclusive_directory"
        else:
            output_dir = root / "exclusive_file_directory"
            output_dir.mkdir()
            target = output_dir / "shared.txt"
        release = root / "release_publishers"
        processes = []
        for index in range(2):
            processes.append(subprocess.Popen(
                [sys.executable, "-c", child_code, operation, str(release),
                 str(target), str(root / f"ready_{index}"), f"publisher-{index}"],
                cwd=Path.cwd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
            ))
        ready_paths = [root / "ready_0", root / "ready_1"]
        deadline = time.monotonic() + 90
        try:
            while not all(path.is_file() for path in ready_paths):
                if any(process.poll() is not None for process in processes):
                    outputs = [process.communicate(timeout=5) for process in processes]
                    pytest.fail(f"exclusive-create child exited before synchronization: {outputs}")
                if time.monotonic() >= deadline:
                    pytest.fail("exclusive-create child processes did not become ready")
                time.sleep(0.01)
            release.touch()
            results = [process.communicate(timeout=60) for process in processes]
            assert all(process.returncode == 0 for process in processes), results
            assert sorted(stdout.strip() for stdout, _ in results) == ["created", "exists"]
            if operation == "file":
                assert target.read_text(encoding="utf-8") in {"publisher-0", "publisher-1"}
                assert target.stat().st_size == len(target.read_bytes())
            else:
                assert target.is_dir()
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)


@pytest.mark.parametrize("operation", ["open", "fsync", "readback"])
@pytest.mark.parametrize("failed_name", [
    "per_cell_scores.parquet", "evaluation_summary.json", "evaluation_commit.json",
])
def test_drvfs_each_publication_stage_failure_leaves_no_commit_marker(
    tmp_path, monkeypatch, operation, failed_name,
):
    drvfs_tmp = Path("/mnt/d/Code/DrugScreenLab/tmp")
    if not drvfs_tmp.is_dir():
        pytest.skip("requires the configured WSL project path on /mnt/d")
    paths = _small_scoring_inputs(tmp_path, monkeypatch)
    with tempfile.TemporaryDirectory(prefix="exp044-drvfs-stage-test-", dir=drvfs_tmp) as parent:
        output_dir = Path(parent) / "evaluation_test"
        if operation == "open":
            original_open = Path.open

            def fail_open(path, mode="r", *args, **kwargs):
                if path.name == failed_name and mode == "xb":
                    raise OSError(f"simulated DrvFS exclusive-open failure: {failed_name}")
                return original_open(path, mode, *args, **kwargs)

            monkeypatch.setattr(Path, "open", fail_open)
        elif operation == "fsync":
            original_fsync = scoring._fsync_file

            def fail_fsync(stream):
                if Path(stream.name).name == failed_name:
                    raise OSError(f"simulated DrvFS fsync failure: {failed_name}")
                original_fsync(stream)

            monkeypatch.setattr(scoring, "_fsync_file", fail_fsync)
        else:
            original_snapshot = scoring._snapshot_file

            def fail_readback(path):
                if path.name == failed_name:
                    raise OSError(f"simulated DrvFS readback failure: {failed_name}")
                return original_snapshot(path)

            monkeypatch.setattr(scoring, "_snapshot_file", fail_readback)

        with pytest.raises(OSError, match="simulated DrvFS .*failure"):
            scoring.run(
                prediction_dir=paths["prediction_dir"],
                response_pairs_path=paths["response_path"],
                input_dir=paths["input_dir"],
                identity_key_audit_path=paths["key_audit_path"],
                numeric_recheck_path=paths["numeric_path"],
                output_dir=output_dir,
            )
        assert output_dir.is_dir()
        assert not (output_dir / scoring.EVALUATION_COMMIT_NAME).exists()
        assert not (output_dir / failed_name).exists()
