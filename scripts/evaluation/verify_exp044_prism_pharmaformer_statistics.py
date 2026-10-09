"""Independent recomputation of EXP-044 cell scores and paired statistics."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata


SEED = 20260922
BOOTSTRAP_DRAWS = 2000
SIGN_FLIP_DRAWS = 10000
MIN_DRUGS = 3
ABS_TOL = 1e-12
EXPECTED_CELLS = 352
EXPECTED_DRUGS = 13
EXPECTED_PREDICTION_KEYS = EXPECTED_CELLS * EXPECTED_DRUGS
EXPECTED_TARGET_KEYS = 2245
EXPECTED_ELIGIBLE_CELLS = 161
EXPECTED_PREDICTION_ATOL = 1e-6
EXPECTED_BASELINE_ATOL = 1e-12
EVALUATION_COMMIT_NAME = "evaluation_commit.json"
EVALUATION_COMMIT_SCHEMA = "EXP-044_EVALUATION_COMMIT_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FileSnapshot:
    """Immutable bytes and digest captured from one input-path read."""

    path: Path
    content: bytes
    sha256: str


def _snapshot_file(path: Path) -> FileSnapshot:
    content = path.read_bytes()
    return FileSnapshot(path=path, content=content,
                        sha256=hashlib.sha256(content).hexdigest())


def _json_from_snapshot(snapshot: FileSnapshot) -> Any:
    return json.loads(snapshot.content.decode("utf-8"))


def _parquet_from_snapshot(snapshot: FileSnapshot, **kwargs: Any) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(snapshot.content), **kwargs)


def _csv_from_snapshot(snapshot: FileSnapshot, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(snapshot.content), **kwargs)


def _require_receipt_experiment_id(receipt: Any, receipt_name: str) -> None:
    if (not isinstance(receipt, dict)
            or not isinstance(receipt.get("experiment_id"), str)
            or receipt["experiment_id"] != "EXP-044"):
        raise ValueError(f"{receipt_name} experiment_id must equal EXP-044")


def _require_run_id(receipt: Any, expected_run_id: str, receipt_name: str) -> None:
    if (not isinstance(receipt, dict)
            or not isinstance(receipt.get("run_id"), str)
            or not receipt["run_id"].strip()
            or receipt["run_id"] != expected_run_id):
        raise ValueError(f"{receipt_name} run_id must match the prediction directory name")


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _evaluation_commit_payload(run_id: str, evaluation_id: str,
                               scores_sha256: str, summary_sha256: str) -> dict[str, Any]:
    return {
        "schema": EVALUATION_COMMIT_SCHEMA,
        "experiment_id": "EXP-044",
        "run_id": run_id,
        "evaluation_id": evaluation_id,
        "outputs_sha256": {
            "per_cell_scores.parquet": scores_sha256,
            "evaluation_summary.json": summary_sha256,
        },
    }


def _validate_evaluation_commit_snapshots(
    marker_snapshot: FileSnapshot,
    scores_snapshot: FileSnapshot,
    summary_snapshot: FileSnapshot,
    *, run_id: str, evaluation_id: str,
) -> dict[str, Any]:
    expected = _evaluation_commit_payload(
        run_id, evaluation_id, scores_snapshot.sha256, summary_snapshot.sha256,
    )
    marker = _json_from_snapshot(marker_snapshot)
    if marker != expected or marker_snapshot.content != _canonical_json_bytes(expected):
        raise ValueError("EXP-044 evaluation commit marker is invalid or output bytes changed")
    summary = _json_from_snapshot(summary_snapshot)
    _require_receipt_experiment_id(summary, "evaluation summary")
    if summary.get("run_id") != run_id:
        raise ValueError("evaluation summary run_id must match the prediction directory name")
    if summary.get("evaluation_id") != evaluation_id:
        raise ValueError("evaluation summary evaluation_id must match the evaluation directory name")
    if (summary.get("status") != "PRISM_HISTORICAL_REEVALUATION_SCORED"
            or summary.get("outputs_sha256") != {
                "per_cell_scores.parquet": scores_snapshot.sha256,
            }):
        raise ValueError("EXP-044 committed evaluation summary identity or score hash differs")
    return summary


def _fsync_file(stream) -> None:
    stream.flush()
    os.fsync(stream.fileno())


def _write_json_exclusive_verified(path: Path, value: dict[str, Any]) -> FileSnapshot:
    """Create the receipt exclusively and verify bytes; parent-entry durability is not claimed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(content)
            _fsync_file(stream)
        snapshot = _snapshot_file(path)
        if snapshot.content != content:
            raise RuntimeError(f"statistics receipt failed byte-for-byte readback: {path.name}")
        return snapshot
    except Exception:
        if created and path.exists():
            path.unlink()
        raise


def _canonical_key_sha256(key_frame: pd.DataFrame, *, unique: bool = True) -> str:
    keys = list(zip(
        key_frame["cell_line_id"].astype(str),
        key_frame["drug_id"].astype(str),
    ))
    if unique:
        keys = list(set(keys))
    keys.sort()
    digest = hashlib.sha256()
    for cell_id, drug_id in keys:
        digest.update(cell_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(drug_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _response_keys_from_projection(frame: pd.DataFrame,
                                   key_audit: dict[str, Any]) -> pd.DataFrame:
    cell_column = key_audit.get("source_pair_table_cell_identity_column")
    selected_columns = key_audit.get("source_pair_table_columns_requested")
    if (cell_column not in {"sample_id", "cell_line_id"}
            or selected_columns not in ([cell_column, "drug_id"],
                                        ["sample_id", "cell_line_id", "drug_id"])
            or key_audit.get("source_pair_table_opened_for_keys_only") is not True
            or key_audit.get("source_pair_table_cell_identity_is_verified_sample_axis") is not True
            or key_audit.get("response_value_columns_requested") != []
            or key_audit.get("target_response_values_read") is not False
            or key_audit.get("target_response_values_loaded") is not False
            or not isinstance(key_audit.get("source_pair_table_file_sha256"), str)
            or len(key_audit.get("source_pair_table_file_sha256", "")) != 64
            or set(selected_columns) - set(frame.columns)):
        raise ValueError("response-blind audit lacks a valid keys-only projection")
    keys = frame.loc[:, selected_columns].copy()
    for column in selected_columns:
        if keys[column].isna().any():
            raise ValueError(f"response identity key {column} has missing values")
        keys[column] = keys[column].astype(str).str.strip()
        if keys[column].eq("").any():
            raise ValueError(f"response identity key {column} has blank values")
    if cell_column == "sample_id" and "cell_line_id" in keys:
        if not keys["sample_id"].equals(keys["cell_line_id"]):
            raise ValueError("response sample_id and cell_line_id identity keys disagree")
        return keys.drop(columns=["sample_id"]).loc[:, ["cell_line_id", "drug_id"]]
    return keys.rename(columns={cell_column: "cell_line_id"}).loc[:, ["cell_line_id", "drug_id"]]


def _validate_response_key_projection(keys: pd.DataFrame, key_audit: dict[str, Any],
                                      strict_ids: set[str], drug_ids: set[str]) -> pd.DataFrame:
    unique = keys.drop_duplicates(["cell_line_id", "drug_id"])
    cohort = unique.loc[
        unique["cell_line_id"].isin(strict_ids) & unique["drug_id"].isin(drug_ids)
    ]
    per_cell = cohort.groupby("cell_line_id")["drug_id"].nunique()
    per_cell = per_cell.reindex(sorted(strict_ids), fill_value=0)
    eligible = sorted(per_cell[per_cell >= MIN_DRUGS].index.astype(str))
    ineligible = sorted(per_cell[per_cell < MIN_DRUGS].index.astype(str))
    per_cell_counts = {str(cell): int(count) for cell, count in per_cell.items()}
    if (len(keys) != key_audit.get("n_source_pair_key_rows")
            or len(unique) != key_audit.get("n_unique_source_pair_keys")
            or len(cohort) != EXPECTED_TARGET_KEYS
            or len(cohort) != key_audit.get("n_revised_cohort_panel_cell_drug_keys")
            or _canonical_key_sha256(keys, unique=False)
            != key_audit.get("source_pair_table_key_rows_sha256")
            or _canonical_key_sha256(keys) != key_audit.get("source_pair_table_unique_keys_sha256")
            or _canonical_key_sha256(cohort)
            != key_audit.get("response_blind_cohort_key_sha256")
            or eligible != key_audit.get("eligible_cell_ids")
            or ineligible != key_audit.get("ineligible_cell_ids")
            or per_cell_counts != key_audit.get("unique_panel_drugs_per_cell")
            or len(eligible) != EXPECTED_ELIGIBLE_CELLS):
        raise ValueError("current PRISM keys differ from the response-blind frozen key audit")
    return keys


def _within_frozen_tolerance(value: Any, tolerance: float) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    numeric_value = float(value)
    return bool(np.isfinite(numeric_value) and 0.0 <= numeric_value <= tolerance)


def validate_prediction_gate(numeric: dict[str, Any], prediction_summary: dict[str, Any],
                             key_audit: dict[str, Any], prediction_sha256: str) -> None:
    comparison = numeric.get("comparison")
    if not isinstance(comparison, dict):
        comparison = {}
    input_hashes = numeric.get("inputs_sha256")
    if not isinstance(input_hashes, dict):
        input_hashes = {}
    if (numeric.get("status") != "PASS_INDEPENDENT_NUMERICAL_RECOMPUTATION_NO_TARGET_LABELS"
            or numeric.get("target_response_values_read") is not False
            or numeric.get("target_response_files_opened") != []
            or input_hashes.get("frozen_predictions") != prediction_sha256
            or numeric.get("n_cells") != EXPECTED_CELLS
            or numeric.get("n_drugs") != EXPECTED_DRUGS
            or comparison.get("n_prediction_keys") != EXPECTED_PREDICTION_KEYS
            or comparison.get("model_prediction_absolute_tolerance") != EXPECTED_PREDICTION_ATOL
            or not _within_frozen_tolerance(
                comparison.get("max_abs_model_prediction_error"), EXPECTED_PREDICTION_ATOL
            )
            or comparison.get("source_mean_absolute_tolerance") != EXPECTED_BASELINE_ATOL
            or not _within_frozen_tolerance(
                comparison.get("max_abs_source_mean_error"), EXPECTED_BASELINE_ATOL
            )
            or numeric.get("comparison_tolerances_frozen_before_target_response_evaluation") is not True):
        raise ValueError("independent model recomputation has not passed the frozen numerical gate")
    if (prediction_summary.get("status") != "PREDICTIONS_FROZEN_NO_PRISM_LABELS_LOADED_PENDING_RESPONSE_EVALUATION"
            or prediction_summary.get("target_response_values_read") is not False
            or prediction_summary.get("target_response_files_opened") != []
            or prediction_summary.get("prediction_file_sha256") != prediction_sha256
            or prediction_summary.get("n_strict_unseen_cohort_cells") != EXPECTED_CELLS
            or prediction_summary.get("n_prism_panel_drugs") != EXPECTED_DRUGS
            or prediction_summary.get("n_prediction_keys") != EXPECTED_PREDICTION_KEYS):
        raise ValueError("frozen prediction summary does not pass the response-blind cohort gate")
    if (key_audit.get("status") != "PASS_RESPONSE_BLIND_IDENTITY_KEY_COVERAGE"
            or key_audit.get("response_value_columns_requested") != []
            or key_audit.get("target_response_values_read") is not False
            or key_audit.get("target_response_values_loaded") is not False
            or not isinstance(key_audit.get("source_pair_table_file_sha256"), str)
            or len(key_audit.get("source_pair_table_file_sha256", "")) != 64
            or key_audit.get("n_revised_strict_unseen_cells") != EXPECTED_CELLS
            or key_audit.get("n_prism_panel_drugs") != EXPECTED_DRUGS
            or key_audit.get("n_revised_cohort_panel_cell_drug_keys") != EXPECTED_TARGET_KEYS
            or key_audit.get("n_cells_with_at_least_three_panel_drugs") != EXPECTED_ELIGIBLE_CELLS
            or not isinstance(key_audit.get("response_blind_cohort_key_sha256"), str)
            or len(key_audit.get("response_blind_cohort_key_sha256", "")) != 64):
        raise ValueError("response-blind identity/key-coverage gate does not pass for the frozen cohort")


def spearman_by_rank_covariance(left: np.ndarray, right: np.ndarray) -> float:
    """Compute Spearman correlation through average ranks and Pearson covariance."""
    if left.size < MIN_DRUGS or right.size != left.size:
        return float("nan")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Spearman inputs must be finite; only protocol-defined degeneracy may be undefined")
    rank_left = rankdata(left, method="average")
    rank_right = rankdata(right, method="average")
    centered_left = rank_left - rank_left.mean()
    centered_right = rank_right - rank_right.mean()
    denominator = np.sqrt(np.dot(centered_left, centered_left)
                          * np.dot(centered_right, centered_right))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(centered_left, centered_right) / denominator)


def _runtime_provenance() -> dict[str, Any]:
    return {
        "script_path": Path(__file__).resolve().as_posix(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "command_argv": list(sys.argv),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
    }


def _paired_resample(differences: np.ndarray) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("paired resampling requires a non-empty finite vector of defined pairs")
    observed = float(values.mean())
    sample_rng = np.random.default_rng(SEED)
    indices = sample_rng.integers(0, values.size, size=(BOOTSTRAP_DRAWS, values.size))
    bootstrap_means = values[indices].mean(axis=1)
    interval = np.quantile(bootstrap_means, [0.025, 0.975])

    sign_rng = np.random.default_rng(SEED)
    sign_matrix = sign_rng.choice(np.array([-1.0, 1.0]),
                                  size=(SIGN_FLIP_DRAWS, values.size))
    null_distribution = (sign_matrix * values).mean(axis=1)
    exceedances = int(np.count_nonzero(np.abs(null_distribution) >= abs(observed)))
    return {
        "n_paired_cells": int(values.size),
        "mean_paired_spearman_difference": observed,
        "bootstrap_95pct_interval": [float(interval[0]), float(interval[1])],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "sign_flip_two_sided_p": float((exceedances + 1) / (SIGN_FLIP_DRAWS + 1)),
        "sign_flip_exceedances": exceedances,
        "sign_flip_draws": SIGN_FLIP_DRAWS,
        "random_seed": SEED,
    }


def _defined_paired_mask(scores: pd.DataFrame) -> np.ndarray:
    """Use the preregistered complete-pair rule used by the scoring entrypoint."""
    return (
        np.isfinite(scores["pharmaformer_spearman"].to_numpy(dtype=np.float64))
        & np.isfinite(scores["source_train_mean_spearman"].to_numpy(dtype=np.float64))
        & np.isfinite(scores["paired_difference"].to_numpy(dtype=np.float64))
    )


def _compare_numeric_values(saved: Any, recomputed: Any, *, name: str,
                             absolute_tolerance: float, allow_protocol_nan: bool) -> float:
    saved_values = np.asarray(saved, dtype=np.float64)
    recomputed_values = np.asarray(recomputed, dtype=np.float64)
    if saved_values.shape != recomputed_values.shape:
        raise ValueError(f"saved numeric shape differs from independent recomputation: {name}")
    if np.isinf(saved_values).any() or np.isinf(recomputed_values).any():
        raise ValueError(f"saved or recomputed numeric values contain infinity: {name}")
    saved_nan = np.isnan(saved_values)
    recomputed_nan = np.isnan(recomputed_values)
    if not np.array_equal(saved_nan, recomputed_nan):
        raise ValueError(f"saved numeric finite-value mask differs from independent recomputation: {name}")
    if saved_nan.any() and not allow_protocol_nan:
        raise ValueError(f"undefined NaN is not permitted for this numeric field: {name}")
    finite = ~recomputed_nan
    errors = np.abs(saved_values[finite] - recomputed_values[finite])
    maximum_error = float(errors.max(initial=0.0))
    if maximum_error > absolute_tolerance:
        raise ValueError(f"saved numeric value differs from independent recomputation: {name}")
    return maximum_error


def run(*, prediction_dir: Path, response_pairs_path: Path, input_dir: Path,
        identity_key_audit_path: Path, numeric_recheck_path: Path,
        evaluation_dir: Path, output_receipt: Path) -> dict[str, Any]:
    if output_receipt.exists():
        raise FileExistsError(f"refusing to overwrite an independent statistics receipt: {output_receipt}")
    prediction_path = prediction_dir / "predictions.parquet"
    prediction_summary_path = prediction_dir / "prediction_summary.json"
    score_path = evaluation_dir / "per_cell_scores.parquet"
    evaluation_summary_path = evaluation_dir / "evaluation_summary.json"
    evaluation_commit_path = evaluation_dir / EVALUATION_COMMIT_NAME
    paths = {
        "frozen_predictions": prediction_path,
        "prediction_summary": prediction_summary_path,
        "numeric_recheck": numeric_recheck_path,
        "response_pairs": response_pairs_path,
        "identity_key_audit": identity_key_audit_path,
        "cell_axis": input_dir / "cell_axis.json",
        "drug_axis": input_dir / "drug_axis.parquet",
        "strict_ids": input_dir / "strict_unseen_cell_lines_v2.csv",
        "per_cell_scores": score_path,
        "evaluation_summary": evaluation_summary_path,
        "evaluation_commit": evaluation_commit_path,
        "input_manifest": input_dir / "manifest.json",
        "identity_resolution": input_dir / "identity_resolution.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"independent EXP-044 statistics inputs are missing: {missing}")

    snapshots = {name: _snapshot_file(path) for name, path in paths.items()}
    numeric = _json_from_snapshot(snapshots["numeric_recheck"])
    prediction_summary = _json_from_snapshot(snapshots["prediction_summary"])
    key_audit = _json_from_snapshot(snapshots["identity_key_audit"])
    input_manifest = _json_from_snapshot(snapshots["input_manifest"])
    identity_resolution = _json_from_snapshot(snapshots["identity_resolution"])
    for receipt_name, receipt in {
        "prediction summary": prediction_summary,
        "numeric recheck": numeric,
        "response-blind key audit": key_audit,
        "input manifest": input_manifest,
        "identity resolution": identity_resolution,
    }.items():
        _require_receipt_experiment_id(receipt, receipt_name)
    prediction_run_id = prediction_summary.get("run_id") if isinstance(prediction_summary, dict) else None
    _require_run_id(numeric, prediction_dir.name, "numeric recheck")
    if (not isinstance(prediction_run_id, str) or not prediction_run_id.strip()
            or prediction_run_id != prediction_dir.name):
        raise ValueError("prediction summary run_id must match the prediction directory name")
    saved_summary = _validate_evaluation_commit_snapshots(
        snapshots["evaluation_commit"], snapshots["per_cell_scores"],
        snapshots["evaluation_summary"],
        run_id=prediction_dir.name, evaluation_id=evaluation_dir.name,
    )
    prediction_hash = snapshots["frozen_predictions"].sha256
    validate_prediction_gate(numeric, prediction_summary, key_audit, prediction_hash)

    verification_path = Path(str(key_audit.get("input_verification_path", "")))
    if not verification_path.is_absolute():
        verification_path = Path.cwd() / verification_path
    source_pair_path = Path(str(key_audit.get("source_pair_table_path", "")))
    if not source_pair_path.is_absolute():
        source_pair_path = Path.cwd() / source_pair_path
    if (not verification_path.is_file()
            or source_pair_path.resolve() != response_pairs_path.resolve()):
        raise ValueError("identity, verification, or response table path differs from the frozen audit")
    verification_snapshot = _snapshot_file(verification_path)
    verification = _json_from_snapshot(verification_snapshot)
    _require_receipt_experiment_id(verification, "response-blind input verification")
    input_manifest_hash = snapshots["input_manifest"].sha256
    key_audit_hash = snapshots["identity_key_audit"].sha256
    numeric_hashes = numeric.get("inputs_sha256", {})
    prediction_hashes = prediction_summary.get("inputs_sha256", {})
    evaluation_hashes = saved_summary.get("inputs_sha256", {})
    output_hashes = saved_summary.get("outputs_sha256", {})
    if not all(isinstance(x, dict) for x in (
            numeric_hashes, prediction_hashes, evaluation_hashes, output_hashes)):
        raise ValueError("EXP-044 receipts lack complete input hash maps")
    if (key_audit.get("identity_resolution_sha256") != snapshots["identity_resolution"].sha256
            or key_audit.get("input_verification_sha256") != verification_snapshot.sha256
            or key_audit.get("input_manifest_sha256") != input_manifest_hash
            or key_audit.get("cell_axis_sha256") != snapshots["cell_axis"].sha256
            or key_audit.get("strict_identity_list_sha256") != snapshots["strict_ids"].sha256
            or key_audit.get("drug_axis_sha256") != snapshots["drug_axis"].sha256
            or numeric_hashes.get("identity_key_audit") != key_audit_hash
            or numeric_hashes.get("input_manifest") != input_manifest_hash
            or numeric_hashes.get("input_verification") != verification_snapshot.sha256
            or numeric_hashes.get("cell_axis") != snapshots["cell_axis"].sha256
            or numeric_hashes.get("drug_axis") != snapshots["drug_axis"].sha256
            or prediction_summary.get("identity_key_audit_sha256") != key_audit_hash
            or prediction_hashes.get("identity_key_coverage_audit") != key_audit_hash
            or prediction_hashes.get("input_manifest") != input_manifest_hash
            or evaluation_hashes.get("frozen_predictions") != prediction_hash
            or evaluation_hashes.get("identity_key_audit") != key_audit_hash
            or evaluation_hashes.get("numeric_recheck") != snapshots["numeric_recheck"].sha256
            or saved_summary.get("target_labels_used_for_training_or_model_selection") is not False
            or saved_summary.get("status") != "PRISM_HISTORICAL_REEVALUATION_SCORED"):
        raise ValueError("EXP-044 input/audit/prediction/evaluation hashes are not consistently bound")
    if (identity_resolution.get("status") != "RESOLVED_ORTHOGONAL_AUTHORITATIVE_CROSSWALK"
            or identity_resolution.get("strict_unseen_accepted") is not True
            or identity_resolution.get("target_response_values_read") is not False
            or verification.get("status") != "PASS_RESPONSE_BLIND_IDENTITY_RESOLVED"
            or verification.get("feature_verification_status") != "PASS"
            or verification.get("input_manifest_sha256") != input_manifest_hash):
        raise ValueError("EXP-044 identity resolution or feature verification is not valid")

    scorer_path = Path(__file__).with_name("score_exp044_prism_pharmaformer.py")
    scorer_provenance = saved_summary.get("provenance", {})
    scorer_snapshot = _snapshot_file(scorer_path)
    source_pair_sha256 = snapshots["response_pairs"].sha256
    if (key_audit.get("source_pair_table_file_sha256") != source_pair_sha256
            or evaluation_hashes.get("response_pairs") != source_pair_sha256
            or output_hashes.get("per_cell_scores.parquet") != snapshots["per_cell_scores"].sha256
            or not isinstance(scorer_provenance, dict)
            or scorer_provenance.get("script_sha256") != scorer_snapshot.sha256
            or not isinstance(scorer_provenance.get("command_argv"), list)
            or not scorer_provenance.get("python_version")
            or "conda_env" not in scorer_provenance):
        raise ValueError("scoring receipt does not bind the exact response, score, and implementation files")

    axis_cells = [str(value) for value in _json_from_snapshot(snapshots["cell_axis"])]
    cells = set(axis_cells)
    drugs_frame = _parquet_from_snapshot(snapshots["drug_axis"])
    drugs = set(drugs_frame["drug_id"].astype(str))
    strict_ids = set(_csv_from_snapshot(snapshots["strict_ids"], usecols=["cell_line_id"])
                     ["cell_line_id"].astype(str))
    if cells != strict_ids or len(cells) != EXPECTED_CELLS or len(drugs) != EXPECTED_DRUGS:
        raise ValueError("independent verifier axes differ from the response-blind frozen cohort")

    key_columns = key_audit.get("source_pair_table_columns_requested")
    if not isinstance(key_columns, list) or "response" in key_columns:
        raise ValueError("response-blind audit does not expose a keys-only projection")
    response_keys_raw = _parquet_from_snapshot(
        snapshots["response_pairs"], columns=key_columns,
    )
    gated_keys = _response_keys_from_projection(response_keys_raw, key_audit)
    gated_keys = _validate_response_key_projection(gated_keys, key_audit, strict_ids, drugs)
    predictions = _parquet_from_snapshot(snapshots["frozen_predictions"])
    for column in ("predicted_source_pharmaformer", "predicted_source_train_mean"):
        if not np.isfinite(predictions[column].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"frozen prediction column {column} contains non-finite values")
    response_frame = _parquet_from_snapshot(
        snapshots["response_pairs"],
        columns=list(dict.fromkeys([*key_columns, "response"])),
    )
    response_cell_column = key_audit["source_pair_table_cell_identity_column"]
    scored_keys = _response_keys_from_projection(response_frame, key_audit)
    if _canonical_key_sha256(scored_keys, unique=False) != _canonical_key_sha256(
            gated_keys, unique=False):
        raise ValueError("PRISM response key rows changed between key-only and scoring reads")
    if response_cell_column == "cell_line_id":
        response_frame = response_frame.rename(columns={"cell_line_id": "sample_id"})
    elif "cell_line_id" in response_frame:
        response_frame = response_frame.drop(columns=["cell_line_id"])
    response_frame = response_frame.rename(columns={response_cell_column: "sample_id"})
    response_frame["sample_id"] = response_frame["sample_id"].astype(str)
    response_frame["drug_id"] = response_frame["drug_id"].astype(str)
    response_frame = response_frame.loc[
        response_frame["sample_id"].isin(strict_ids)
        & response_frame["drug_id"].isin(drugs)
    ].copy()
    if not np.isfinite(response_frame["response"].to_numpy(dtype=np.float64)).all():
        raise ValueError("registered target response values contain non-finite observations")
    predictions["cell_line_id"] = predictions["cell_line_id"].astype(str)
    predictions["drug_id"] = predictions["drug_id"].astype(str)
    response_frame = response_frame.merge(
        predictions,
        left_on=["sample_id", "drug_id"],
        right_on=["cell_line_id", "drug_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if (len(response_frame) != EXPECTED_TARGET_KEYS
            or not response_frame["_merge"].eq("both").all()):
        raise ValueError(
            f"independent keyed join differs from the registered {EXPECTED_TARGET_KEYS} response keys"
        )

    score_rows: list[dict[str, Any]] = []
    for cell_id, group in response_frame.groupby("sample_id", sort=True):
        if group["drug_id"].nunique() < MIN_DRUGS:
            continue
        group = group.sort_values("drug_id", kind="stable")
        labels = group["response"].to_numpy(dtype=np.float64)
        model = group["predicted_source_pharmaformer"].to_numpy(dtype=np.float64)
        baseline = group["predicted_source_train_mean"].to_numpy(dtype=np.float64)
        score_rows.append({
            "cell_line_id": str(cell_id),
            "n_matched_drugs": int(group["drug_id"].nunique()),
            "pharmaformer_spearman": spearman_by_rank_covariance(model, labels),
            "source_train_mean_spearman": spearman_by_rank_covariance(baseline, labels),
        })
    recomputed = pd.DataFrame(score_rows)
    recomputed["paired_difference"] = (
        recomputed["pharmaformer_spearman"] - recomputed["source_train_mean_spearman"]
    )
    scored = _parquet_from_snapshot(snapshots["per_cell_scores"])
    columns = ["cell_line_id", "n_matched_drugs", "pharmaformer_spearman",
               "source_train_mean_spearman", "paired_difference"]
    joined_scores = recomputed[columns].merge(
        scored[columns], on="cell_line_id", how="outer", validate="one_to_one",
        suffixes=("_recomputed", "_saved"), indicator=True,
    )
    if (len(joined_scores) != EXPECTED_ELIGIBLE_CELLS
            or not joined_scores["_merge"].eq("both").all()):
        raise ValueError("saved and independently recomputed patient/cell score keys differ")
    numeric_cols = [column for column in columns if column != "cell_line_id"]
    maximum_errors: dict[str, float] = {}
    for column in numeric_cols:
        left = joined_scores[f"{column}_recomputed"].to_numpy(dtype=np.float64)
        right = joined_scores[f"{column}_saved"].to_numpy(dtype=np.float64)
        maximum_errors[column] = _compare_numeric_values(
            right, left, name=column,
            absolute_tolerance=0.0 if column == "n_matched_drugs" else ABS_TOL,
            allow_protocol_nan=column in {
                "pharmaformer_spearman", "source_train_mean_spearman", "paired_difference",
            },
        )

    defined_paired = _defined_paired_mask(recomputed)
    valid_recomputed = recomputed.loc[defined_paired].copy()
    differences = valid_recomputed["paired_difference"].to_numpy(dtype=np.float64)
    statistics = _paired_resample(differences)
    saved_stats = saved_summary.get("statistics")
    if not isinstance(saved_stats, dict):
        raise ValueError("saved paired statistics must be an object")
    statistics.update({
        "mean_pharmaformer_cell_spearman": float(
            valid_recomputed["pharmaformer_spearman"].mean()
        ),
        "mean_source_train_mean_cell_spearman": float(
            valid_recomputed["source_train_mean_spearman"].mean()
        ),
    })
    summary_errors: dict[str, float] = {}
    floating_summary_fields = (
        "mean_paired_spearman_difference",
        "mean_pharmaformer_cell_spearman",
        "mean_source_train_mean_cell_spearman",
    )
    for field in floating_summary_fields:
        summary_errors[field] = _compare_numeric_values(
            saved_stats.get(field), statistics[field], name=f"statistics.{field}",
            absolute_tolerance=ABS_TOL, allow_protocol_nan=False,
        )
    saved_interval = saved_stats.get("bootstrap_95pct_interval")
    expected_interval = statistics["bootstrap_95pct_interval"]
    summary_errors["bootstrap_low"] = _compare_numeric_values(
        saved_interval[0] if isinstance(saved_interval, (list, tuple)) and len(saved_interval) == 2 else None,
        expected_interval[0], name="statistics.bootstrap_95pct_interval[0]",
        absolute_tolerance=ABS_TOL, allow_protocol_nan=False,
    )
    summary_errors["bootstrap_high"] = _compare_numeric_values(
        saved_interval[1] if isinstance(saved_interval, (list, tuple)) and len(saved_interval) == 2 else None,
        expected_interval[1], name="statistics.bootstrap_95pct_interval[1]",
        absolute_tolerance=ABS_TOL, allow_protocol_nan=False,
    )
    for field in ("sign_flip_two_sided_p",):
        summary_errors[field] = _compare_numeric_values(
            saved_stats.get(field), statistics[field], name=f"statistics.{field}",
            absolute_tolerance=ABS_TOL, allow_protocol_nan=False,
        )
    exact_summary_fields = (
        "n_paired_cells", "bootstrap_draws", "sign_flip_exceedances",
        "sign_flip_draws", "random_seed",
    )
    for field in exact_summary_fields:
        summary_errors[field] = _compare_numeric_values(
            saved_stats.get(field), statistics[field], name=f"statistics.{field}",
            absolute_tolerance=0.0, allow_protocol_nan=False,
        )

    saved_coverage = saved_summary.get("coverage")
    if not isinstance(saved_coverage, dict):
        raise ValueError("saved coverage summary must be an object")
    expected_coverage = {
        "n_response_keys": int(len(response_frame)),
        "n_response_cells": int(response_frame["sample_id"].nunique()),
        "n_cells_with_at_least_three_drugs": int(len(recomputed)),
        "n_cells_with_defined_paired_spearman": int(len(valid_recomputed)),
        "mean_matched_drugs_per_eligible_cell": float(recomputed["n_matched_drugs"].mean()),
        "min_matched_drugs_per_eligible_cell": int(recomputed["n_matched_drugs"].min()),
        "max_matched_drugs_per_eligible_cell": int(recomputed["n_matched_drugs"].max()),
    }
    for field, expected_value in expected_coverage.items():
        summary_errors[f"coverage.{field}"] = _compare_numeric_values(
            saved_coverage.get(field), expected_value,
            name=f"coverage.{field}",
            absolute_tolerance=ABS_TOL if field.startswith("mean_") else 0.0,
            allow_protocol_nan=False,
        )
    expected_undefined_cells = recomputed.loc[
        ~defined_paired, "cell_line_id",
    ].astype(str).tolist()
    if saved_coverage.get("undefined_metric_cells") != expected_undefined_cells:
        raise ValueError("saved coverage undefined-metric cell list differs from recomputation")

    receipt = {
        "experiment_id": "EXP-044",
        "run_id": prediction_dir.name,
        "evaluation_id": evaluation_dir.name,
        "status": "PASS_INDEPENDENT_NUMERICAL_RECOMPUTATION_OF_PRISM_EVALUATION",
        "review_type": "separate_rank_covariance_and_resampling_implementation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "response_values_read_after_prediction_gate": True,
        "target_response_values_used_for_training_or_selection": False,
        "n_joined_response_keys": int(len(response_frame)),
        "n_eligible_cells": int(len(recomputed)),
        "n_defined_paired_cells": int(len(valid_recomputed)),
        "undefined_metric_cells": recomputed.loc[~defined_paired, "cell_line_id"].astype(str).tolist(),
        "max_abs_per_cell_errors": maximum_errors,
        "max_abs_summary_errors": summary_errors,
        "absolute_tolerance": ABS_TOL,
        "recomputed_statistics": statistics,
        "inputs_sha256": {
            **{name: snapshot.sha256 for name, snapshot in snapshots.items()},
            "input_verification": verification_snapshot.sha256,
            "scorer_script": scorer_snapshot.sha256,
        },
        "provenance": _runtime_provenance(),
    }
    _write_json_exclusive_verified(output_receipt, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--response-pairs", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--identity-key-audit", required=True, type=Path)
    parser.add_argument("--numeric-recheck", required=True, type=Path)
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    parser.add_argument("--output-receipt", required=True, type=Path)
    args = parser.parse_args()
    result = run(
        prediction_dir=args.prediction_dir,
        response_pairs_path=args.response_pairs,
        input_dir=args.input_dir,
        identity_key_audit_path=args.identity_key_audit,
        numeric_recheck_path=args.numeric_recheck,
        evaluation_dir=args.evaluation_dir,
        output_receipt=args.output_receipt,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
