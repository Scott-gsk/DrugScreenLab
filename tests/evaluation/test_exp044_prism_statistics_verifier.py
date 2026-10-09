import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from scripts.evaluation import score_exp044_prism_pharmaformer as scoring
from scripts.evaluation import verify_exp044_prism_pharmaformer_statistics as verifier


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _inputs(root: Path, corruption: str):
    prediction_dir = root / "prediction_v1"
    prediction_dir.mkdir()
    prediction_path = prediction_dir / "predictions.parquet"
    prediction_path.write_bytes(b"frozen prediction placeholder")
    prediction_hash = hashlib.sha256(prediction_path.read_bytes()).hexdigest()

    prediction_summary = {
        "experiment_id": "EXP-044",
        "run_id": prediction_dir.name,
        "status": "PREDICTIONS_FROZEN_NO_PRISM_LABELS_LOADED_PENDING_RESPONSE_EVALUATION",
        "target_response_values_read": False,
        "target_response_files_opened": [],
        "prediction_file_sha256": prediction_hash,
        "n_strict_unseen_cohort_cells": 352,
        "n_prism_panel_drugs": 13,
        "n_prediction_keys": 4576,
    }

    numeric = {
        "experiment_id": "EXP-044",
        "run_id": prediction_dir.name,
        "status": "PASS_INDEPENDENT_NUMERICAL_RECOMPUTATION_NO_TARGET_LABELS",
        "target_response_values_read": False,
        "target_response_files_opened": [],
        "inputs_sha256": {"frozen_predictions": prediction_hash},
        "n_cells": 352,
        "n_drugs": 13,
        "comparison_tolerances_frozen_before_target_response_evaluation": True,
        "comparison": {
            "n_prediction_keys": 4576,
            "n_exact_model_predictions": 4576,
            "max_abs_model_prediction_error": 0.0,
            "model_prediction_absolute_tolerance": 1e-6,
            "max_abs_source_mean_error": 0.0,
            "source_mean_absolute_tolerance": 1e-12,
            "source_mean_values_exact": True,
        },
    }
    key_audit = {
        "experiment_id": "EXP-044",
        "status": "PASS_RESPONSE_BLIND_IDENTITY_KEY_COVERAGE",
        "response_value_columns_requested": [],
        "target_response_values_read": False,
        "n_revised_strict_unseen_cells": 352,
        "n_prism_panel_drugs": 13,
        "n_revised_cohort_panel_cell_drug_keys": 2245,
        "n_cells_with_at_least_three_panel_drugs": 161,
        "target_response_values_loaded": False,
        "source_pair_table_file_sha256": "a" * 64,
        "response_blind_cohort_key_sha256": "b" * 64,
    }
    if corruption == "numeric_target_values_read":
        numeric["target_response_values_read"] = True
    elif corruption == "numeric_target_file_opened":
        numeric["target_response_files_opened"] = ["prism_pairs.parquet"]
    elif corruption == "numeric_prediction_key_count":
        numeric["comparison"]["n_prediction_keys"] = 4575
    elif corruption == "numeric_prediction_error":
        numeric["comparison"]["max_abs_model_prediction_error"] = 1e-3
    elif corruption == "key_audit_status":
        key_audit["status"] = "FAIL_RESPONSE_BLIND_IDENTITY_KEY_COVERAGE"
    elif corruption == "key_audit_response_column":
        key_audit["response_value_columns_requested"] = ["response"]
    elif corruption == "key_audit_target_values_read":
        key_audit["target_response_values_read"] = True
    elif corruption == "key_audit_target_values_loaded":
        key_audit["target_response_values_loaded"] = True
    elif corruption == "key_audit_missing_source_file_digest":
        key_audit.pop("source_pair_table_file_sha256")
    elif corruption == "prediction_summary_target_values_read":
        prediction_summary["target_response_values_read"] = True
    elif corruption == "prediction_summary_target_file_opened":
        prediction_summary["target_response_files_opened"] = ["prism_pairs.parquet"]
    _write_json(prediction_dir / "prediction_summary.json", prediction_summary)

    numeric_path = root / "numeric_recheck.json"
    key_audit_path = root / "key_audit.json"
    _write_json(numeric_path, numeric)
    _write_json(key_audit_path, key_audit)

    response_path = root / "prism_pairs.parquet"
    response_path.write_bytes(b"must not be opened when a gate is invalid")
    input_dir = root / "inputs"
    input_dir.mkdir()
    _write_json(input_dir / "manifest.json", {"experiment_id": "EXP-044"})
    _write_json(input_dir / "identity_resolution.json", {"experiment_id": "EXP-044"})
    (input_dir / "cell_axis.json").write_text("[]", encoding="utf-8")
    (input_dir / "drug_axis.parquet").write_bytes(b"unused before gate")
    (input_dir / "strict_unseen_cell_lines_v2.csv").write_text("cell_line_id\n", encoding="utf-8")

    evaluation_dir = root / "evaluation_test"
    evaluation_dir.mkdir()
    scores_path = evaluation_dir / "per_cell_scores.parquet"
    scores_path.write_bytes(b"unused before gate")
    summary = {
        "experiment_id": "EXP-044",
        "run_id": prediction_dir.name,
        "evaluation_id": evaluation_dir.name,
        "status": "PRISM_HISTORICAL_REEVALUATION_SCORED",
        "outputs_sha256": {
            "per_cell_scores.parquet": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
        },
    }
    summary_path = evaluation_dir / "evaluation_summary.json"
    _write_json(summary_path, summary)
    marker = scoring._evaluation_commit_payload(
        prediction_dir.name, evaluation_dir.name,
        hashlib.sha256(scores_path.read_bytes()).hexdigest(),
        hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    )
    (evaluation_dir / scoring.EVALUATION_COMMIT_NAME).write_bytes(
        scoring._canonical_json_bytes(marker),
    )
    return {
        "prediction_dir": prediction_dir,
        "response_path": response_path,
        "input_dir": input_dir,
        "key_audit_path": key_audit_path,
        "numeric_path": numeric_path,
        "evaluation_dir": evaluation_dir,
        "output_receipt": root / "statistics_recheck.json",
    }


@pytest.mark.parametrize("corruption", [
    "numeric_target_values_read",
    "numeric_target_file_opened",
    "numeric_prediction_key_count",
    "numeric_prediction_error",
    "key_audit_status",
    "key_audit_response_column",
    "key_audit_target_values_read",
    "key_audit_target_values_loaded",
    "key_audit_missing_source_file_digest",
    "prediction_summary_target_values_read",
    "prediction_summary_target_file_opened",
])
def test_invalid_no_label_gate_fails_before_target_response_parquet_read(
    tmp_path, monkeypatch, corruption,
):
    paths = _inputs(tmp_path, corruption)
    original_read_parquet = verifier.pd.read_parquet
    response_reads = []

    def guarded_read_parquet(path, *args, **kwargs):
        if Path(path) == paths["response_path"]:
            response_reads.append(Path(path))
            raise AssertionError("target response parquet was read before all gates passed")
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(verifier.pd, "read_parquet", guarded_read_parquet)

    with pytest.raises(ValueError, match="gate"):
        verifier.run(
            prediction_dir=paths["prediction_dir"],
            response_pairs_path=paths["response_path"],
            input_dir=paths["input_dir"],
            identity_key_audit_path=paths["key_audit_path"],
            numeric_recheck_path=paths["numeric_path"],
            evaluation_dir=paths["evaluation_dir"],
            output_receipt=paths["output_receipt"],
        )

    assert response_reads == []
    assert not paths["output_receipt"].exists()


def _valid_scored_case(root: Path, monkeypatch, *, evaluation_dir_override: Path | None = None,
                       constant_model_cell: str | None = None):
    for module in (scoring, verifier):
        monkeypatch.setattr(module, "EXPECTED_CELLS", 2)
        monkeypatch.setattr(module, "EXPECTED_DRUGS", 3)
        monkeypatch.setattr(module, "EXPECTED_TARGET_KEYS", 6)
        monkeypatch.setattr(module, "EXPECTED_ELIGIBLE_CELLS", 2)
    monkeypatch.setattr(verifier, "EXPECTED_PREDICTION_KEYS", 6)

    prediction_dir = root / "prediction_test"
    input_dir = root / "input_test"
    evaluation_dir = evaluation_dir_override or root / "evaluation_test"
    prediction_dir.mkdir()
    input_dir.mkdir()
    cells = ["cell-a", "cell-b"]
    drugs = ["drug-1", "drug-2", "drug-3"]
    prediction_rows = []
    response_rows = []
    for cell, model_values, baseline_values, response_values in (
        ("cell-a", [3.0, 2.0, 1.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]),
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
    if constant_model_cell is not None:
        mask = predictions["cell_line_id"].eq(constant_model_cell)
        assert mask.any()
        predictions.loc[mask, "predicted_source_pharmaformer"] = 4.0
    responses = pd.DataFrame(response_rows, columns=["sample_id", "drug_id", "response"])
    prediction_path = prediction_dir / "predictions.parquet"
    prediction_summary_path = prediction_dir / "prediction_summary.json"
    response_path = input_dir / "pairs.parquet"
    key_audit_path = root / "key_audit_test.json"
    numeric_path = root / "numeric_test.json"
    verification_path = root / "verification_test.json"
    identity_path = input_dir / "identity_resolution.json"
    manifest_path = input_dir / "manifest.json"
    cell_axis_path = input_dir / "cell_axis.json"
    drug_axis_path = input_dir / "drug_axis.parquet"
    strict_path = input_dir / "strict_unseen_cell_lines_v2.csv"

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
    scoring.run(
        prediction_dir=prediction_dir,
        response_pairs_path=response_path,
        input_dir=input_dir,
        identity_key_audit_path=key_audit_path,
        numeric_recheck_path=numeric_path,
        output_dir=evaluation_dir,
    )
    return {
        "prediction_dir": prediction_dir,
        "input_dir": input_dir,
        "manifest_path": input_dir / "manifest.json",
        "identity_path": input_dir / "identity_resolution.json",
        "evaluation_dir": evaluation_dir,
        "response_path": response_path,
        "prediction_path": prediction_path,
        "prediction_summary_path": prediction_summary_path,
        "numeric_path": numeric_path,
        "key_audit_path": key_audit_path,
        "verification_path": verification_path,
        "manifest_path": manifest_path,
        "identity_path": identity_path,
        "scores_path": evaluation_dir / "per_cell_scores.parquet",
        "evaluation_summary_path": evaluation_dir / "evaluation_summary.json",
        "evaluation_commit_path": evaluation_dir / scoring.EVALUATION_COMMIT_NAME,
        "output_receipt": root / "statistics_recheck.json",
    }


def test_verifier_uses_only_the_input_snapshots_after_files_change(tmp_path, monkeypatch):
    paths = _valid_scored_case(tmp_path, monkeypatch)
    expected_hashes = {
        "frozen_predictions": verifier._snapshot_file(paths["prediction_path"]).sha256,
        "response_pairs": verifier._snapshot_file(paths["response_path"]).sha256,
        "per_cell_scores": verifier._snapshot_file(paths["scores_path"]).sha256,
        "evaluation_summary": verifier._snapshot_file(paths["evaluation_summary_path"]).sha256,
        "evaluation_commit": verifier._snapshot_file(paths["evaluation_commit_path"]).sha256,
    }
    original_parquet_reader = verifier._parquet_from_snapshot
    response_snapshots = []
    changed_paths = False

    def mutate_files_after_validation(snapshot, **kwargs):
        nonlocal changed_paths
        if not changed_paths:
            for path in (paths["prediction_path"], paths["response_path"],
                         paths["scores_path"], paths["evaluation_summary_path"],
                         paths["evaluation_commit_path"]):
                path.write_bytes(b"path content changed after snapshot validation")
            changed_paths = True
        if snapshot.path == paths["response_path"]:
            response_snapshots.append(snapshot.content)
        return original_parquet_reader(snapshot, **kwargs)

    monkeypatch.setattr(verifier, "_parquet_from_snapshot", mutate_files_after_validation)
    receipt = verifier.run(
        prediction_dir=paths["prediction_dir"],
        response_pairs_path=paths["response_path"],
        input_dir=paths["input_dir"],
        identity_key_audit_path=paths["key_audit_path"],
        numeric_recheck_path=paths["numeric_path"],
        evaluation_dir=paths["evaluation_dir"],
        output_receipt=paths["output_receipt"],
    )

    assert changed_paths
    assert receipt["status"] == "PASS_INDEPENDENT_NUMERICAL_RECOMPUTATION_OF_PRISM_EVALUATION"
    assert receipt["inputs_sha256"]["frozen_predictions"] == expected_hashes["frozen_predictions"]
    assert receipt["inputs_sha256"]["response_pairs"] == expected_hashes["response_pairs"]
    assert receipt["inputs_sha256"]["per_cell_scores"] == expected_hashes["per_cell_scores"]
    assert receipt["inputs_sha256"]["evaluation_summary"] == expected_hashes["evaluation_summary"]
    assert receipt["inputs_sha256"]["evaluation_commit"] == expected_hashes["evaluation_commit"]
    assert len(response_snapshots) == 2
    assert response_snapshots[0] == response_snapshots[1]


@pytest.mark.parametrize(("receipt_name", "field", "replacement", "delete_field"), [
    ("prediction_summary", "experiment_id", None, True),
    ("prediction_summary", "experiment_id", "", False),
    ("prediction_summary", "experiment_id", 44, False),
    ("prediction_summary", "experiment_id", "EXP-045", False),
    ("numeric_recheck", "experiment_id", "EXP-045", False),
    ("key_audit", "experiment_id", "EXP-045", False),
    ("verification", "experiment_id", "EXP-045", False),
    ("manifest", "experiment_id", "EXP-045", False),
    ("identity_resolution", "experiment_id", "EXP-045", False),
    ("evaluation_summary", "experiment_id", "EXP-045", False),
    ("prediction_summary", "run_id", None, True),
    ("prediction_summary", "run_id", "", False),
    ("prediction_summary", "run_id", 44, False),
    ("prediction_summary", "run_id", "another_run", False),
    ("numeric_recheck", "run_id", None, True),
    ("numeric_recheck", "run_id", "", False),
    ("numeric_recheck", "run_id", 44, False),
    ("numeric_recheck", "run_id", "another_run", False),
    ("evaluation_summary", "run_id", "another_run", False),
    ("evaluation_summary", "evaluation_id", "another_eval", False),
])
def test_identity_and_version_receipt_mismatch_fails_before_response_decode(
    tmp_path, monkeypatch, receipt_name, field, replacement, delete_field,
):
    paths = _valid_scored_case(tmp_path, monkeypatch)
    receipt_paths = {
        "prediction_summary": paths["prediction_summary_path"],
        "numeric_recheck": paths["numeric_path"],
        "key_audit": paths["key_audit_path"],
        "verification": paths["verification_path"],
        "manifest": paths["manifest_path"],
        "identity_resolution": paths["identity_path"],
        "evaluation_summary": paths["evaluation_summary_path"],
    }
    receipt_path = receipt_paths[receipt_name]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if delete_field:
        receipt.pop(field)
    else:
        receipt[field] = replacement
    _write_json(receipt_path, receipt)
    if receipt_name == "evaluation_summary":
        marker_path = paths["evaluation_commit_path"]
        marker = scoring._evaluation_commit_payload(
            paths["prediction_dir"].name, paths["evaluation_dir"].name,
            scoring.sha256(paths["scores_path"]), scoring.sha256(receipt_path),
        )
        marker_path.write_bytes(scoring._canonical_json_bytes(marker))
    paths["response_path"].write_bytes(b"response values must not be decoded")

    original_reader = verifier._parquet_from_snapshot

    def guarded_reader(snapshot, **kwargs):
        if snapshot.path == paths["response_path"]:
            raise AssertionError("response parquet was decoded before receipt identities were checked")
        return original_reader(snapshot, **kwargs)

    monkeypatch.setattr(verifier, "_parquet_from_snapshot", guarded_reader)
    expected_error = field if field in {"run_id", "evaluation_id"} else "experiment_id"
    with pytest.raises(ValueError, match=expected_error):
        verifier.run(
            prediction_dir=paths["prediction_dir"],
            response_pairs_path=paths["response_path"],
            input_dir=paths["input_dir"],
            identity_key_audit_path=paths["key_audit_path"],
            numeric_recheck_path=paths["numeric_path"],
            evaluation_dir=paths["evaluation_dir"],
            output_receipt=paths["output_receipt"],
        )
    assert not paths["output_receipt"].exists()


def test_global_run_id_cannot_replace_missing_numeric_recheck_run_id(tmp_path, monkeypatch):
    paths = _valid_scored_case(tmp_path, monkeypatch)
    numeric = json.loads(paths["numeric_path"].read_text(encoding="utf-8"))
    numeric.pop("run_id")
    _write_json(paths["numeric_path"], numeric)
    manifest_path = paths["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_id"] = paths["prediction_dir"].name
    _write_json(manifest_path, manifest)

    original_reader = verifier._parquet_from_snapshot

    def guarded_reader(snapshot, **kwargs):
        if snapshot.path == paths["response_path"]:
            raise AssertionError("response parquet decoded before numeric run_id validation")
        return original_reader(snapshot, **kwargs)

    monkeypatch.setattr(verifier, "_parquet_from_snapshot", guarded_reader)
    with pytest.raises(ValueError, match="numeric recheck run_id"):
        verifier.run(
            prediction_dir=paths["prediction_dir"],
            response_pairs_path=paths["response_path"],
            input_dir=paths["input_dir"],
            identity_key_audit_path=paths["key_audit_path"],
            numeric_recheck_path=paths["numeric_path"],
            evaluation_dir=paths["evaluation_dir"],
            output_receipt=paths["output_receipt"],
        )
    assert not paths["output_receipt"].exists()


@pytest.mark.parametrize("tamper", [
    "scores", "summary", "marker",
    "scores_truncated", "summary_truncated", "marker_truncated",
])
def test_invalid_commit_or_changed_output_fails_before_response_decode(
    tmp_path, monkeypatch, tamper,
):
    paths = _valid_scored_case(tmp_path, monkeypatch)
    target_name = tamper.removesuffix("_truncated")
    target = {
        "scores": paths["scores_path"],
        "summary": paths["evaluation_summary_path"],
        "marker": paths["evaluation_commit_path"],
    }[target_name]
    if tamper == "marker":
        marker = json.loads(target.read_text(encoding="utf-8"))
        marker["run_id"] = "changed-run"
        _write_json(target, marker)
    elif tamper.endswith("_truncated"):
        target.write_bytes(target.read_bytes()[:-1])
    else:
        target.write_bytes(target.read_bytes() + b" ")

    original_reader = verifier._parquet_from_snapshot
    original_json_reader = verifier._json_from_snapshot

    def guarded_reader(snapshot, **kwargs):
        if snapshot.path == paths["response_path"]:
            raise AssertionError("response parquet decoded before commit validation")
        return original_reader(snapshot, **kwargs)

    def guarded_json_reader(snapshot):
        if snapshot.path == paths["evaluation_summary_path"]:
            raise AssertionError("evaluation summary parsed before the commit marker passed")
        return original_json_reader(snapshot)

    monkeypatch.setattr(verifier, "_parquet_from_snapshot", guarded_reader)
    monkeypatch.setattr(verifier, "_json_from_snapshot", guarded_json_reader)
    with pytest.raises(ValueError, match="commit marker|summary identity"):
        verifier.run(
            prediction_dir=paths["prediction_dir"],
            response_pairs_path=paths["response_path"],
            input_dir=paths["input_dir"],
            identity_key_audit_path=paths["key_audit_path"],
            numeric_recheck_path=paths["numeric_path"],
            evaluation_dir=paths["evaluation_dir"],
            output_receipt=paths["output_receipt"],
        )
    assert not paths["output_receipt"].exists()


def test_saved_nan_metric_is_rejected_with_self_consistent_commit_hashes(tmp_path, monkeypatch):
    paths = _valid_scored_case(tmp_path, monkeypatch)
    scores = pd.read_parquet(paths["scores_path"])
    assert np.isfinite(scores.loc[0, "pharmaformer_spearman"])
    scores.loc[0, "pharmaformer_spearman"] = np.nan
    scores_bytes = scores.to_parquet(index=False)
    paths["scores_path"].write_bytes(scores_bytes)

    summary = json.loads(paths["evaluation_summary_path"].read_text(encoding="utf-8"))
    scores_sha256 = hashlib.sha256(scores_bytes).hexdigest()
    summary["outputs_sha256"]["per_cell_scores.parquet"] = scores_sha256
    summary_bytes = scoring._json_bytes(summary)
    paths["evaluation_summary_path"].write_bytes(summary_bytes)
    marker = scoring._evaluation_commit_payload(
        paths["prediction_dir"].name, paths["evaluation_dir"].name,
        scores_sha256, hashlib.sha256(summary_bytes).hexdigest(),
    )
    paths["evaluation_commit_path"].write_bytes(scoring._canonical_json_bytes(marker))

    with pytest.raises(ValueError, match="finite-value mask"):
        verifier.run(
            prediction_dir=paths["prediction_dir"],
            response_pairs_path=paths["response_path"],
            input_dir=paths["input_dir"],
            identity_key_audit_path=paths["key_audit_path"],
            numeric_recheck_path=paths["numeric_path"],
            evaluation_dir=paths["evaluation_dir"],
            output_receipt=paths["output_receipt"],
        )
    assert not paths["output_receipt"].exists()


def test_saved_finite_score_drift_is_rejected_with_self_consistent_commit_hashes(
    tmp_path, monkeypatch,
):
    paths = _valid_scored_case(tmp_path, monkeypatch)
    scores = pd.read_parquet(paths["scores_path"])
    original = float(scores.loc[1, "pharmaformer_spearman"])
    assert np.isfinite(original)
    scores.loc[1, "pharmaformer_spearman"] = original - 0.01
    assert np.isfinite(scores.loc[1, "pharmaformer_spearman"])
    scores_bytes = scores.to_parquet(index=False)
    paths["scores_path"].write_bytes(scores_bytes)

    summary = json.loads(paths["evaluation_summary_path"].read_text(encoding="utf-8"))
    scores_sha256 = hashlib.sha256(scores_bytes).hexdigest()
    summary["outputs_sha256"]["per_cell_scores.parquet"] = scores_sha256
    summary_bytes = scoring._json_bytes(summary)
    paths["evaluation_summary_path"].write_bytes(summary_bytes)
    marker = scoring._evaluation_commit_payload(
        paths["prediction_dir"].name, paths["evaluation_dir"].name,
        scores_sha256, hashlib.sha256(summary_bytes).hexdigest(),
    )
    paths["evaluation_commit_path"].write_bytes(scoring._canonical_json_bytes(marker))

    with pytest.raises(ValueError, match="saved numeric value differs"):
        verifier.run(
            prediction_dir=paths["prediction_dir"],
            response_pairs_path=paths["response_path"],
            input_dir=paths["input_dir"],
            identity_key_audit_path=paths["key_audit_path"],
            numeric_recheck_path=paths["numeric_path"],
            evaluation_dir=paths["evaluation_dir"],
            output_receipt=paths["output_receipt"],
        )
    assert not paths["output_receipt"].exists()


def test_undefined_spearman_uses_same_complete_pair_rule_in_scoring_and_recheck(
    tmp_path, monkeypatch,
):
    paths = _valid_scored_case(tmp_path, monkeypatch, constant_model_cell="cell-a")
    saved_scores = pd.read_parquet(paths["scores_path"]).set_index("cell_line_id")
    assert np.isnan(saved_scores.loc["cell-a", "pharmaformer_spearman"])
    assert np.isnan(saved_scores.loc["cell-a", "paired_difference"])
    assert np.isfinite(saved_scores.loc["cell-a", "source_train_mean_spearman"])

    summary = json.loads(paths["evaluation_summary_path"].read_text(encoding="utf-8"))
    assert summary["statistics"]["n_paired_cells"] == 1
    assert summary["coverage"]["undefined_metric_cells"] == ["cell-a"]
    assert summary["coverage"]["n_cells_with_defined_paired_spearman"] == 1

    receipt = verifier.run(
        prediction_dir=paths["prediction_dir"],
        response_pairs_path=paths["response_path"],
        input_dir=paths["input_dir"],
        identity_key_audit_path=paths["key_audit_path"],
        numeric_recheck_path=paths["numeric_path"],
        evaluation_dir=paths["evaluation_dir"],
        output_receipt=paths["output_receipt"],
    )
    assert receipt["n_eligible_cells"] == 2
    assert receipt["n_defined_paired_cells"] == 1
    assert receipt["undefined_metric_cells"] == ["cell-a"]
    assert receipt["recomputed_statistics"]["n_paired_cells"] == 1


def test_statistics_receipt_exclusive_write_is_verified_and_refuses_overwrite(tmp_path):
    output = tmp_path / "independent_statistical_recheck.json"
    snapshot = verifier._write_json_exclusive_verified(
        output, {"experiment_id": "EXP-044", "value": 1},
    )
    assert snapshot.content == output.read_bytes()
    assert snapshot.sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    original_bytes = output.read_bytes()
    with pytest.raises(FileExistsError):
        verifier._write_json_exclusive_verified(output, {"experiment_id": "EXP-044", "value": 2})
    assert output.read_bytes() == original_bytes

    failed_output = tmp_path / "failed_recheck.json"
    with pytest.raises(ValueError):
        verifier._write_json_exclusive_verified(failed_output, {"invalid": float("nan")})
    assert not failed_output.exists()


@pytest.mark.parametrize("case", ["one_sided_nan", "infinity", "both_protocol_nan"])
def test_numeric_comparison_enforces_finite_masks_and_protocol_nan(case):
    if case == "one_sided_nan":
        with pytest.raises(ValueError, match="finite-value mask"):
            verifier._compare_numeric_values(
                [np.nan], [0.25], name="spearman", absolute_tolerance=1e-12,
                allow_protocol_nan=True,
            )
    elif case == "infinity":
        with pytest.raises(ValueError, match="infinity"):
            verifier._compare_numeric_values(
                [np.inf], [np.inf], name="spearman", absolute_tolerance=1e-12,
                allow_protocol_nan=True,
            )
    else:
        assert verifier._compare_numeric_values(
            [np.nan], [np.nan], name="spearman", absolute_tolerance=1e-12,
            allow_protocol_nan=True,
        ) == 0.0


def test_statistics_receipt_fails_closed_on_fsync_error(tmp_path, monkeypatch):
    output = tmp_path / "independent_statistical_recheck.json"

    def fail_fsync(stream):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(verifier, "_fsync_file", fail_fsync)
    with pytest.raises(OSError, match="fsync failure"):
        verifier._write_json_exclusive_verified(output, {"experiment_id": "EXP-044"})
    assert not output.exists()


def test_drvfs_concurrent_statistics_receipt_create_is_exclusive():
    drvfs_tmp = Path("/mnt/d/Code/DrugScreenLab/tmp")
    if not drvfs_tmp.is_dir():
        pytest.skip("requires the configured WSL project path on /mnt/d")
    with tempfile.TemporaryDirectory(prefix="exp044-drvfs-receipt-test-", dir=drvfs_tmp) as parent:
        output = Path(parent) / "independent_statistical_recheck.json"
        barrier = threading.Barrier(2)

        def publish(value):
            barrier.wait()
            try:
                verifier._write_json_exclusive_verified(
                    output, {"experiment_id": "EXP-044", "value": value},
                )
                return "created"
            except FileExistsError:
                return "exists"

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(publish, (1, 2)))
        assert sorted(outcomes) == ["created", "exists"]
        saved = json.loads(output.read_text(encoding="utf-8"))
        assert saved["experiment_id"] == "EXP-044"
        assert saved["value"] in {1, 2}


def test_drvfs_scoring_to_statistics_commit_chain(tmp_path, monkeypatch):
    drvfs_tmp = Path("/mnt/d/Code/DrugScreenLab/tmp")
    if not drvfs_tmp.is_dir():
        pytest.skip("requires the configured WSL project path on /mnt/d")
    with tempfile.TemporaryDirectory(prefix="exp044-drvfs-chain-test-", dir=drvfs_tmp) as parent:
        root = Path(parent)
        evaluation_dir = root / "evaluation_test"
        paths = _valid_scored_case(
            tmp_path, monkeypatch, evaluation_dir_override=evaluation_dir,
        )
        paths["output_receipt"] = root / "statistics_recheck.json"
        receipt = verifier.run(
            prediction_dir=paths["prediction_dir"],
            response_pairs_path=paths["response_path"],
            input_dir=paths["input_dir"],
            identity_key_audit_path=paths["key_audit_path"],
            numeric_recheck_path=paths["numeric_path"],
            evaluation_dir=paths["evaluation_dir"],
            output_receipt=paths["output_receipt"],
        )
        assert receipt["status"] == "PASS_INDEPENDENT_NUMERICAL_RECOMPUTATION_OF_PRISM_EVALUATION"
        assert (evaluation_dir / scoring.EVALUATION_COMMIT_NAME).is_file()
        assert paths["output_receipt"].is_file()
        assert receipt["inputs_sha256"]["evaluation_commit"] == hashlib.sha256(
            (evaluation_dir / scoring.EVALUATION_COMMIT_NAME).read_bytes()
        ).hexdigest()


@pytest.mark.parametrize("operation", ["fsync", "readback"])
def test_drvfs_statistics_receipt_stage_failure_leaves_no_success_file(
    operation, monkeypatch,
):
    drvfs_tmp = Path("/mnt/d/Code/DrugScreenLab/tmp")
    if not drvfs_tmp.is_dir():
        pytest.skip("requires the configured WSL project path on /mnt/d")
    with tempfile.TemporaryDirectory(prefix="exp044-drvfs-stat-stage-test-", dir=drvfs_tmp) as parent:
        output = Path(parent) / "statistics_recheck.json"
        if operation == "fsync":
            def fail_fsync(stream):
                raise OSError("simulated DrvFS statistics fsync failure")

            monkeypatch.setattr(verifier, "_fsync_file", fail_fsync)
        else:
            original_snapshot = verifier._snapshot_file

            def fail_readback(path):
                if path == output:
                    raise OSError("simulated DrvFS statistics readback failure")
                return original_snapshot(path)

            monkeypatch.setattr(verifier, "_snapshot_file", fail_readback)
        with pytest.raises(OSError, match="simulated DrvFS statistics .*failure"):
            verifier._write_json_exclusive_verified(
                output, {"experiment_id": "EXP-044", "status": "PASS"},
            )
        assert not output.exists()
