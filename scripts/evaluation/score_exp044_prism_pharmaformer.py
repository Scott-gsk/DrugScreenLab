"""Evaluate the frozen EXP-044 predictions against keyed PRISM responses.

The evaluator refuses to open target responses until the response-blind input
gate and independent model/source-mean numerical recomputation have passed.
"""

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
from scipy.stats import spearmanr


EXPECTED_CELLS = 352
EXPECTED_DRUGS = 13
EXPECTED_TARGET_KEYS = 2245
EXPECTED_ELIGIBLE_CELLS = 161
MIN_DRUGS_PER_CELL = 3
EXPECTED_PREDICTION_ATOL = 1e-6
EXPECTED_BASELINE_ATOL = 1e-12
BOOTSTRAP_DRAWS = 2000
SIGN_FLIP_DRAWS = 10000
RANDOM_SEED = 20260922
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


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _fsync_file(stream) -> None:
    stream.flush()
    os.fsync(stream.fileno())


def _create_evaluation_dir_exclusive(path: Path) -> None:
    """Create an absent directory exclusively; no power-loss durability is claimed.

    The parent directory is not fsynced, and no directory rename is used or claimed.
    """
    path.mkdir()


def _write_bytes_exclusive(path: Path, content: bytes) -> FileSnapshot:
    """Create one file exclusively and verify bytes; parent-entry durability is not claimed."""
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(content)
            _fsync_file(stream)
        snapshot = _snapshot_file(path)
        if snapshot.content != content:
            raise RuntimeError(f"exclusive output failed byte-for-byte readback: {path.name}")
        return snapshot
    except Exception:
        if created and path.exists():
            path.unlink()
        raise


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
    marker = _json_from_snapshot(marker_snapshot)
    expected = _evaluation_commit_payload(
        run_id, evaluation_id, scores_snapshot.sha256, summary_snapshot.sha256,
    )
    if marker != expected or marker_snapshot.content != _canonical_json_bytes(expected):
        raise ValueError("EXP-044 evaluation commit marker is invalid or output bytes changed")
    summary = _json_from_snapshot(summary_snapshot)
    _require_receipt_experiment_id(summary, "evaluation summary")
    if (not isinstance(summary, dict)
            or summary.get("run_id") != run_id
            or summary.get("evaluation_id") != evaluation_id
            or summary.get("status") != "PRISM_HISTORICAL_REEVALUATION_SCORED"
            or summary.get("outputs_sha256") != {
                "per_cell_scores.parquet": scores_snapshot.sha256,
            }):
        raise ValueError("EXP-044 committed evaluation summary identity or score hash differs")
    return marker


def _require_receipt_experiment_id(receipt: Any, receipt_name: str) -> None:
    if (not isinstance(receipt, dict)
            or not isinstance(receipt.get("experiment_id"), str)
            or receipt["experiment_id"] != "EXP-044"):
        raise ValueError(f"{receipt_name} experiment_id must equal EXP-044")


def _require_prediction_run_id(receipt: Any, prediction_dir: Path,
                              receipt_name: str = "prediction summary") -> None:
    if (not isinstance(receipt, dict)
            or not isinstance(receipt.get("run_id"), str)
            or not receipt["run_id"].strip()
            or receipt["run_id"] != prediction_dir.name):
        raise ValueError(f"{receipt_name} run_id must match the prediction directory name")


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


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


def _validated_keys(frame: pd.DataFrame, key_audit: dict[str, Any],
                    strict_cell_ids: set[str], drug_ids: set[str]) -> pd.DataFrame:
    cell_column = key_audit.get("source_pair_table_cell_identity_column")
    selected_columns = key_audit.get("source_pair_table_columns_requested")
    allowed_columns = (
        [cell_column, "drug_id"],
        ["sample_id", "cell_line_id", "drug_id"],
    )
    if (cell_column not in {"sample_id", "cell_line_id"}
            or selected_columns not in allowed_columns
            or "response" in selected_columns
            or key_audit.get("source_pair_table_opened_for_keys_only") is not True
            or key_audit.get("source_pair_table_cell_identity_is_verified_sample_axis") is not True
            or key_audit.get("response_value_columns_requested") != []
            or key_audit.get("target_response_values_read") is not False
            or key_audit.get("target_response_values_loaded") is not False
            or not isinstance(key_audit.get("source_pair_table_file_sha256"), str)
            or len(key_audit.get("source_pair_table_file_sha256", "")) != 64):
        raise ValueError("response-blind audit does not certify a keys-only identity read")
    required = set(selected_columns)
    if required - set(frame.columns):
        raise ValueError("current response key projection differs from its audit column contract")
    keys = frame.loc[:, selected_columns].copy()
    for column in selected_columns:
        if keys[column].isna().any():
            raise ValueError(f"response key column {column} contains missing values")
        keys[column] = keys[column].astype(str).str.strip()
        if keys[column].eq("").any():
            raise ValueError(f"response key column {column} contains blank values")
    if cell_column == "sample_id" and "cell_line_id" in keys:
        if not keys["sample_id"].equals(keys["cell_line_id"]):
            raise ValueError("current response sample_id and cell_line_id keys disagree")
        normalized = keys.drop(columns=["sample_id"]).loc[:, ["cell_line_id", "drug_id"]]
    else:
        normalized = keys.rename(columns={cell_column: "cell_line_id"})
        normalized = normalized.loc[:, ["cell_line_id", "drug_id"]]
    unique_keys = normalized.drop_duplicates(["cell_line_id", "drug_id"])
    cohort_keys = unique_keys.loc[
        unique_keys["cell_line_id"].isin(strict_cell_ids)
        & unique_keys["drug_id"].isin(drug_ids)
    ]
    per_cell = cohort_keys.groupby("cell_line_id")["drug_id"].nunique()
    per_cell = per_cell.reindex(sorted(strict_cell_ids), fill_value=0)
    eligible = sorted(per_cell[per_cell >= MIN_DRUGS_PER_CELL].index.astype(str))
    ineligible = sorted(per_cell[per_cell < MIN_DRUGS_PER_CELL].index.astype(str))
    expected_per_cell = {str(cell): int(count) for cell, count in per_cell.items()}
    if (len(normalized) != key_audit.get("n_source_pair_key_rows")
            or len(unique_keys) != key_audit.get("n_unique_source_pair_keys")
            or len(cohort_keys) != key_audit.get("n_revised_cohort_panel_cell_drug_keys")
            or _canonical_key_sha256(normalized, unique=False)
            != key_audit.get("source_pair_table_key_rows_sha256")
            or _canonical_key_sha256(normalized)
            != key_audit.get("source_pair_table_unique_keys_sha256")
            or _canonical_key_sha256(cohort_keys)
            != key_audit.get("response_blind_cohort_key_sha256")
            or eligible != key_audit.get("eligible_cell_ids")
            or ineligible != key_audit.get("ineligible_cell_ids")
            or expected_per_cell != key_audit.get("unique_panel_drugs_per_cell")
            or len(cohort_keys) != EXPECTED_TARGET_KEYS
            or len(eligible) != EXPECTED_ELIGIBLE_CELLS):
        raise ValueError("current response keys do not match the frozen response-blind key audit")
    return normalized


def _load_response_values_after_key_gate(
    response_snapshot: FileSnapshot,
    key_audit: dict[str, Any],
    strict_cell_ids: set[str],
    drug_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bind exact response keys and source-file bytes before loading labels."""
    key_columns = key_audit.get("source_pair_table_columns_requested")
    if not isinstance(key_columns, list) or "response" in key_columns:
        raise ValueError("response-blind audit does not provide a response-free key projection")
    key_frame = _parquet_from_snapshot(response_snapshot, columns=key_columns)
    gated_keys = _validated_keys(key_frame, key_audit, strict_cell_ids, drug_ids)
    if response_snapshot.sha256 != key_audit.get("source_pair_table_file_sha256"):
        raise ValueError("PRISM source file bytes differ from the response-blind key audit")

    value_columns = list(dict.fromkeys([*key_columns, "response"]))
    responses = _parquet_from_snapshot(response_snapshot, columns=value_columns)
    scored_keys = _validated_keys(responses, key_audit, strict_cell_ids, drug_ids)
    if (_canonical_key_sha256(scored_keys, unique=False)
            != _canonical_key_sha256(gated_keys, unique=False)):
        raise ValueError("response table keys changed between the keys-only gate and scoring read")
    return responses, gated_keys


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


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < MIN_DRUGS_PER_CELL or right.size != left.size:
        return float("nan")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Spearman inputs must be finite; only protocol-defined degeneracy may be undefined")
    if np.unique(left).size < 2 or np.unique(right).size < 2:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def _defined_paired_mask(scores: pd.DataFrame) -> np.ndarray:
    """Cells enter paired statistics only when both correlations are defined."""
    return (
        np.isfinite(scores["pharmaformer_spearman"].to_numpy(dtype=np.float64))
        & np.isfinite(scores["source_train_mean_spearman"].to_numpy(dtype=np.float64))
        & np.isfinite(scores["paired_difference"].to_numpy(dtype=np.float64))
    )


def score_cell_pairs(predictions: pd.DataFrame, responses: pd.DataFrame,
                     strict_cell_ids: set[str], drug_ids: set[str], *,
                     expected_target_keys: int = EXPECTED_TARGET_KEYS,
                     expected_eligible_cells: int = EXPECTED_ELIGIBLE_CELLS
                     ) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_prediction = {
        "cell_line_id", "drug_id", "predicted_source_pharmaformer", "predicted_source_train_mean"
    }
    required_response = {"sample_id", "drug_id", "response"}
    if required_prediction - set(predictions.columns):
        raise ValueError("frozen predictions lack one or more registered columns")
    if required_response - set(responses.columns):
        raise ValueError("PRISM response pairs lack sample_id, drug_id, or response")
    pred = predictions.loc[:, list(required_prediction)].copy()
    resp = responses.loc[:, ["sample_id", "drug_id", "response"]].copy()
    pred["cell_line_id"] = pred["cell_line_id"].astype(str)
    pred["drug_id"] = pred["drug_id"].astype(str)
    resp["sample_id"] = resp["sample_id"].astype(str)
    resp["drug_id"] = resp["drug_id"].astype(str)
    if pred.duplicated(["cell_line_id", "drug_id"]).any():
        raise ValueError("frozen predictions have duplicate cell/drug keys")
    if resp.duplicated(["sample_id", "drug_id"]).any():
        raise ValueError("PRISM response table has duplicate cell/drug keys")

    pred = pred.loc[pred["cell_line_id"].isin(strict_cell_ids)
                    & pred["drug_id"].isin(drug_ids)].copy()
    for column in ("predicted_source_pharmaformer", "predicted_source_train_mean"):
        if not np.isfinite(pred[column].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"frozen prediction column {column} contains non-finite values")
    resp = resp.loc[resp["sample_id"].isin(strict_cell_ids)
                    & resp["drug_id"].isin(drug_ids)].copy()
    resp_values = resp["response"].to_numpy(dtype=np.float64)
    if not np.isfinite(resp_values).all():
        raise ValueError("registered target response values contain non-finite observations")
    if len(resp) != expected_target_keys:
        raise ValueError(f"expected {expected_target_keys} target keys, found {len(resp)}")
    if resp.groupby("sample_id")["drug_id"].nunique().ge(MIN_DRUGS_PER_CELL).sum() != expected_eligible_cells:
        raise ValueError("response-key eligibility differs from the expected cohort")

    # The prediction grid covers all 352 x 13 keys, while target responses are
    # present for only a preregistered subset. Every response key must join to
    # a prediction; unlabeled prediction keys are intentionally not scored.
    joined = resp.merge(
        pred,
        left_on=["sample_id", "drug_id"],
        right_on=["cell_line_id", "drug_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("one or more gated PRISM response keys have no frozen prediction")
    if len(joined) != expected_target_keys:
        raise ValueError("joined cohort does not match the response-blind key-coverage receipt")

    rows: list[dict[str, Any]] = []
    for cell_id, group in joined.groupby("cell_line_id", sort=True):
        group = group.sort_values("drug_id", kind="stable")
        n_drugs = int(group["drug_id"].nunique())
        if n_drugs < MIN_DRUGS_PER_CELL:
            continue
        observed = group["response"].to_numpy(dtype=np.float64)
        model = group["predicted_source_pharmaformer"].to_numpy(dtype=np.float64)
        source_mean = group["predicted_source_train_mean"].to_numpy(dtype=np.float64)
        rows.append({
            "cell_line_id": str(cell_id),
            "n_matched_drugs": n_drugs,
            "pharmaformer_spearman": _spearman(model, observed),
            "source_train_mean_spearman": _spearman(source_mean, observed),
        })

    scores = pd.DataFrame(rows)
    if len(scores) != expected_eligible_cells:
        raise ValueError(f"expected {expected_eligible_cells} eligible cells, scored {len(scores)}")
    scores["paired_difference"] = (
        scores["pharmaformer_spearman"] - scores["source_train_mean_spearman"]
    )
    defined_paired = _defined_paired_mask(scores)
    undefined = scores.loc[~defined_paired, "cell_line_id"].astype(str).tolist()
    valid = scores.loc[defined_paired].copy()
    counts = {
        "n_response_keys": int(len(resp)),
        "n_response_cells": int(resp["sample_id"].nunique()),
        "n_cells_with_at_least_three_drugs": int(len(scores)),
        "n_cells_with_defined_paired_spearman": int(len(valid)),
        "undefined_metric_cells": undefined,
        "mean_matched_drugs_per_eligible_cell": float(scores["n_matched_drugs"].mean()),
        "min_matched_drugs_per_eligible_cell": int(scores["n_matched_drugs"].min()),
        "max_matched_drugs_per_eligible_cell": int(scores["n_matched_drugs"].max()),
    }
    return scores, counts


def paired_statistics(differences: np.ndarray, *, seed: int = RANDOM_SEED,
                      bootstrap_draws: int = BOOTSTRAP_DRAWS,
                      sign_flip_draws: int = SIGN_FLIP_DRAWS) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("paired statistics require a non-empty finite vector")
    observed = float(values.mean())
    bootstrap_rng = np.random.default_rng(seed)
    bootstrap_indices = bootstrap_rng.integers(0, len(values), size=(bootstrap_draws, len(values)))
    bootstrap_means = values[bootstrap_indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])

    sign_rng = np.random.default_rng(seed)
    signs = sign_rng.choice(np.array([-1.0, 1.0]), size=(sign_flip_draws, len(values)))
    null_means = (signs * values).mean(axis=1)
    exceedances = int(np.count_nonzero(np.abs(null_means) >= abs(observed)))
    p_value = (exceedances + 1) / (sign_flip_draws + 1)
    return {
        "n_paired_cells": int(len(values)),
        "mean_paired_spearman_difference": observed,
        "bootstrap_95pct_interval": [float(low), float(high)],
        "bootstrap_draws": int(bootstrap_draws),
        "sign_flip_two_sided_p": float(p_value),
        "sign_flip_exceedances": exceedances,
        "sign_flip_draws": int(sign_flip_draws),
        "random_seed": int(seed),
    }


def run(*, prediction_dir: Path, response_pairs_path: Path, input_dir: Path,
        identity_key_audit_path: Path, numeric_recheck_path: Path,
        output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite an EXP-044 evaluation directory: {output_dir}")
    prediction_path = prediction_dir / "predictions.parquet"
    prediction_summary_path = prediction_dir / "prediction_summary.json"
    response_blind_paths = {
        "manifest": input_dir / "manifest.json",
        "cell_axis": input_dir / "cell_axis.json",
        "drug_axis": input_dir / "drug_axis.parquet",
        "strict_ids": input_dir / "strict_unseen_cell_lines_v2.csv",
    }
    identity_resolution_path = input_dir / "identity_resolution.json"
    required_paths = [prediction_path, prediction_summary_path, response_pairs_path,
                      identity_key_audit_path, numeric_recheck_path,
                      identity_resolution_path, *response_blind_paths.values()]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"EXP-044 evaluation inputs are missing: {missing}")

    snapshot_paths = {
        "frozen_predictions": prediction_path,
        "prediction_summary": prediction_summary_path,
        "response_pairs": response_pairs_path,
        "identity_key_audit": identity_key_audit_path,
        "numeric_recheck": numeric_recheck_path,
        "manifest": response_blind_paths["manifest"],
        "cell_axis": response_blind_paths["cell_axis"],
        "drug_axis": response_blind_paths["drug_axis"],
        "strict_ids": response_blind_paths["strict_ids"],
        "identity_resolution": identity_resolution_path,
    }
    snapshots = {name: _snapshot_file(path) for name, path in snapshot_paths.items()}
    numeric = _json_from_snapshot(snapshots["numeric_recheck"])
    key_audit = _json_from_snapshot(snapshots["identity_key_audit"])
    prediction_summary = _json_from_snapshot(snapshots["prediction_summary"])
    manifest = _json_from_snapshot(snapshots["manifest"])
    identity_resolution = _json_from_snapshot(snapshots["identity_resolution"])
    verification_path = Path(key_audit.get("input_verification_path", ""))
    if not verification_path.is_absolute():
        verification_path = Path.cwd() / verification_path
    if not verification_path.is_file():
        raise FileNotFoundError("response-blind key audit does not resolve its input verification receipt")
    verification_snapshot = _snapshot_file(verification_path)
    verification = _json_from_snapshot(verification_snapshot)
    upstream_receipts = {
        "prediction summary": prediction_summary,
        "numeric recheck": numeric,
        "response-blind key audit": key_audit,
        "response-blind input verification": verification,
        "input manifest": manifest,
        "identity resolution": identity_resolution,
    }
    for receipt_name, receipt in upstream_receipts.items():
        _require_receipt_experiment_id(receipt, receipt_name)
    _require_prediction_run_id(prediction_summary, prediction_dir)
    _require_prediction_run_id(numeric, prediction_dir, "numeric recheck")
    key_audit_sha256 = snapshots["identity_key_audit"].sha256
    manifest_sha256 = snapshots["manifest"].sha256
    input_hashes = numeric.get("inputs_sha256")
    summary_hashes = prediction_summary.get("inputs_sha256")
    comparison = numeric.get("comparison")
    if not isinstance(input_hashes, dict):
        input_hashes = {}
    if not isinstance(summary_hashes, dict):
        summary_hashes = {}
    if not isinstance(comparison, dict):
        comparison = {}

    source_pair_path = Path(str(key_audit.get("source_pair_table_path", "")))
    if not source_pair_path.is_absolute():
        source_pair_path = Path.cwd() / source_pair_path
    if source_pair_path.resolve() != response_pairs_path.resolve():
        raise ValueError("response-blind key audit refers to a different PRISM pair table")
    if (numeric.get("status") != "PASS_INDEPENDENT_NUMERICAL_RECOMPUTATION_NO_TARGET_LABELS"
            or numeric.get("target_response_values_read") is not False
            or numeric.get("target_response_files_opened") != []
            or numeric.get("comparison_tolerances_frozen_before_target_response_evaluation") is not True
            or numeric.get("n_cells") != EXPECTED_CELLS
            or numeric.get("n_drugs") != EXPECTED_DRUGS
            or comparison.get("n_prediction_keys") != EXPECTED_CELLS * EXPECTED_DRUGS
            or comparison.get("model_prediction_absolute_tolerance") != EXPECTED_PREDICTION_ATOL
            or not isinstance(comparison.get("max_abs_model_prediction_error"), (int, float))
            or not np.isfinite(comparison.get("max_abs_model_prediction_error"))
            or not 0 <= comparison["max_abs_model_prediction_error"] <= EXPECTED_PREDICTION_ATOL
            or comparison.get("source_mean_absolute_tolerance") != EXPECTED_BASELINE_ATOL
            or not isinstance(comparison.get("max_abs_source_mean_error"), (int, float))
            or not np.isfinite(comparison.get("max_abs_source_mean_error"))
            or not 0 <= comparison["max_abs_source_mean_error"] <= EXPECTED_BASELINE_ATOL
            or input_hashes.get("frozen_predictions") != snapshots["frozen_predictions"].sha256
            or input_hashes.get("prediction_summary") != snapshots["prediction_summary"].sha256
            or input_hashes.get("identity_key_audit") != key_audit_sha256
            or input_hashes.get("input_manifest") != manifest_sha256
            or input_hashes.get("input_verification") != verification_snapshot.sha256
            or input_hashes.get("cell_axis") != snapshots["cell_axis"].sha256
            or input_hashes.get("drug_axis") != snapshots["drug_axis"].sha256):
        raise ValueError("independent numerical receipt does not bind a passing no-label cohort and exact inputs")
    if (key_audit.get("status") != "PASS_RESPONSE_BLIND_IDENTITY_KEY_COVERAGE"
            or key_audit.get("identity_resolution_status") != identity_resolution.get("status")
            or key_audit.get("identity_resolution_sha256") != snapshots["identity_resolution"].sha256
            or key_audit.get("input_verification_sha256") != verification_snapshot.sha256
            or key_audit.get("input_manifest_sha256") != manifest_sha256
            or key_audit.get("cell_axis_sha256") != snapshots["cell_axis"].sha256
            or key_audit.get("strict_identity_list_sha256") != snapshots["strict_ids"].sha256
            or key_audit.get("drug_axis_sha256") != snapshots["drug_axis"].sha256
            or key_audit.get("response_value_columns_requested") != []
            or key_audit.get("target_response_values_read") is not False
            or key_audit.get("n_revised_strict_unseen_cells") != EXPECTED_CELLS
            or key_audit.get("n_prism_panel_drugs") != EXPECTED_DRUGS
            or key_audit.get("n_revised_cohort_panel_cell_drug_keys") != EXPECTED_TARGET_KEYS
            or key_audit.get("n_cells_with_at_least_three_panel_drugs") != EXPECTED_ELIGIBLE_CELLS):
        raise ValueError("response-blind identity/key audit does not bind the frozen cohort inputs")
    if (prediction_summary.get("status") != "PREDICTIONS_FROZEN_NO_PRISM_LABELS_LOADED_PENDING_RESPONSE_EVALUATION"
            or prediction_summary.get("target_response_values_read") is not False
            or prediction_summary.get("target_response_files_opened") != []
            or prediction_summary.get("prediction_file_sha256") != snapshots["frozen_predictions"].sha256
            or prediction_summary.get("identity_key_audit_sha256") != key_audit_sha256
            or prediction_summary.get("n_strict_unseen_cohort_cells") != EXPECTED_CELLS
            or prediction_summary.get("n_prism_panel_drugs") != EXPECTED_DRUGS
            or prediction_summary.get("n_prediction_keys") != EXPECTED_CELLS * EXPECTED_DRUGS
            or summary_hashes.get("identity_key_coverage_audit") != key_audit_sha256
            or summary_hashes.get("input_manifest") != manifest_sha256):
        raise ValueError("frozen prediction summary does not bind the current response-blind audit")
    if (identity_resolution.get("status") != "RESOLVED_ORTHOGONAL_AUTHORITATIVE_CROSSWALK"
            or identity_resolution.get("strict_unseen_accepted") is not True
            or identity_resolution.get("target_response_values_read") is not False
            or manifest.get("scientific_strict_unseen_accepted") is not True
            or manifest.get("target_response_values_read") is not False
            or manifest.get("response_label_files_opened") != []
            or verification.get("status") != "PASS_RESPONSE_BLIND_IDENTITY_RESOLVED"
            or verification.get("feature_verification_status") != "PASS"
            or verification.get("input_manifest_sha256") != manifest_sha256):
        raise ValueError("response-blind input manifest records target label access")
    cells = set(map(str, json.loads(snapshots["cell_axis"].content.decode("utf-8"))))
    drugs_frame = _parquet_from_snapshot(snapshots["drug_axis"])
    drugs = set(drugs_frame["drug_id"].astype(str))
    strict_frame = _csv_from_snapshot(snapshots["strict_ids"], usecols=["cell_line_id"])
    strict_ids = set(strict_frame["cell_line_id"].astype(str))
    if len(cells) != EXPECTED_CELLS or cells != strict_ids or len(drugs) != EXPECTED_DRUGS:
        raise ValueError("frozen cell or drug axes differ from the registered evaluation cohort")

    predictions = _parquet_from_snapshot(snapshots["frozen_predictions"])
    # The response keys and values are parsed from the same immutable bytes.
    responses, _ = _load_response_values_after_key_gate(
        snapshots["response_pairs"], key_audit, strict_ids, drugs
    )
    response_cell_column = key_audit["source_pair_table_cell_identity_column"]
    if response_cell_column == "cell_line_id":
        responses = responses.rename(columns={"cell_line_id": "sample_id"})
    elif "cell_line_id" in responses:
        responses = responses.drop(columns=["cell_line_id"])
    scores, coverage = score_cell_pairs(
        predictions, responses, strict_ids, drugs,
        expected_target_keys=EXPECTED_TARGET_KEYS,
        expected_eligible_cells=EXPECTED_ELIGIBLE_CELLS,
    )
    valid = scores.loc[_defined_paired_mask(scores)].copy()
    statistics = paired_statistics(valid["paired_difference"].to_numpy())
    statistics.update({
        "mean_pharmaformer_cell_spearman": float(valid["pharmaformer_spearman"].mean()),
        "mean_source_train_mean_cell_spearman": float(valid["source_train_mean_spearman"].mean()),
    })

    inputs_hash = {
        "frozen_predictions": snapshots["frozen_predictions"].sha256,
        "prediction_summary": snapshots["prediction_summary"].sha256,
        "numeric_recheck": snapshots["numeric_recheck"].sha256,
        "identity_key_audit": snapshots["identity_key_audit"].sha256,
        "response_pairs": snapshots["response_pairs"].sha256,
        "identity_resolution": snapshots["identity_resolution"].sha256,
        "input_verification": verification_snapshot.sha256,
        **{name: snapshots[name].sha256 for name in response_blind_paths},
    }
    result = {
        "experiment_id": "EXP-044",
        "run_id": prediction_dir.name,
        "evaluation_id": output_dir.name,
        "status": "PRISM_HISTORICAL_REEVALUATION_SCORED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": "strict_unseen_cell_macro_spearman_pharmaformer_minus_fold0_source_train_drug_mean",
        "response_source": "PRISM Extended Primary historical cohort previously examined in EXP-019",
        "response_direction": "larger neg_extended_primary_lfc means greater sensitivity",
        "pairing_unit": "cell_line_id",
        "target_labels_used_for": "fixed_evaluation_only_after_frozen_prediction_and_independent_recomputation",
        "target_labels_used_for_training_or_model_selection": False,
        "eligibility_rule": f"at least {MIN_DRUGS_PER_CELL} matched panel drugs per cell",
        "coverage": coverage,
        "statistics": statistics,
        "statistical_family": "single preregistered EXP-044 primary comparison; not in PDO A1/A2/A3 Holm family",
        "bootstrap_interval": "percentile 95% interval across paired eligible cells",
        "inputs_sha256": inputs_hash,
        "outputs_sha256": {},
        "provenance": _runtime_provenance(),
        "interpretation_limit": (
            "Historical PRISM re-evaluation on a previously examined cohort; not independent external confirmation, "
            "PDO transfer, or evidence of clinical benefit."
        ),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _create_evaluation_dir_exclusive(output_dir)
    commit_path = output_dir / EVALUATION_COMMIT_NAME
    commit_snapshot = None
    try:
        scores_bytes = scores.to_parquet(index=False)
        if not isinstance(scores_bytes, bytes):
            raise RuntimeError("per-cell score serializer did not return an immutable byte payload")
        score_sha256 = hashlib.sha256(scores_bytes).hexdigest()
        result["outputs_sha256"] = {"per_cell_scores.parquet": score_sha256}
        summary_bytes = _json_bytes(result)

        scores_path = output_dir / "per_cell_scores.parquet"
        scores_snapshot = _write_bytes_exclusive(scores_path, scores_bytes)
        summary_path = output_dir / "evaluation_summary.json"
        summary_snapshot = _write_bytes_exclusive(summary_path, summary_bytes)

        reopened_scores = _parquet_from_snapshot(scores_snapshot)
        try:
            pd.testing.assert_frame_equal(scores, reopened_scores, check_dtype=True, check_exact=True)
        except AssertionError as error:
            raise RuntimeError("staged per-cell scores do not round-trip exactly") from error
        reopened_summary = _json_from_snapshot(summary_snapshot)
        if (not isinstance(reopened_summary, dict)
                or reopened_summary.get("experiment_id") != "EXP-044"
                or reopened_summary.get("run_id") != prediction_dir.name
                or reopened_summary.get("evaluation_id") != output_dir.name
                or reopened_summary.get("outputs_sha256") != {
                    "per_cell_scores.parquet": scores_snapshot.sha256,
                }):
            raise RuntimeError("staged evaluation summary does not bind its score file and identities")

        commit_bytes = _canonical_json_bytes(_evaluation_commit_payload(
            prediction_dir.name, output_dir.name,
            scores_snapshot.sha256, summary_snapshot.sha256,
        ))
        commit_snapshot = _write_bytes_exclusive(commit_path, commit_bytes)

        # Re-read all three final files after the last exclusive marker write.
        final_scores = _snapshot_file(scores_path)
        final_summary = _snapshot_file(summary_path)
        final_marker = _snapshot_file(commit_snapshot.path)
        _validate_evaluation_commit_snapshots(
            final_marker, final_scores, final_summary,
            run_id=prediction_dir.name, evaluation_id=output_dir.name,
        )
    except Exception:
        # A partial output directory remains uncommitted and cannot pass readers.
        if commit_snapshot is not None and commit_path.exists():
            commit_path.unlink()
        raise
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--response-pairs", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--identity-key-audit", required=True, type=Path)
    parser.add_argument("--numeric-recheck", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = run(
        prediction_dir=args.prediction_dir,
        response_pairs_path=args.response_pairs,
        input_dir=args.input_dir,
        identity_key_audit_path=args.identity_key_audit,
        numeric_recheck_path=args.numeric_recheck,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
