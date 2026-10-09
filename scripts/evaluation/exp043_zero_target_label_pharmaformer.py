"""Evaluate frozen source-only PharmaFormer predictions on CRC PDO labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from drug_screen.evaluation.zero_target_label import (
    expression_identity_permutation_test,
    paired_resampling_summary,
    score_zero_target_predictions,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def evaluate(run_dir: Path, input_dir: Path, source_pairs_path: Path,
             source_indices_path: Path) -> dict[str, Any]:
    """Score only after confirming the no-target-label predictions are frozen."""
    started = time.perf_counter()
    summary_path = run_dir / "evaluation_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"evaluation is already present: {summary_path}")
    prediction_path = run_dir / "predictions.parquet"
    prediction_receipt_path = run_dir / "prediction_summary.json"
    prediction_receipt = json.loads(prediction_receipt_path.read_text(encoding="utf-8"))
    if (prediction_receipt.get("experiment_id") != "EXP-043"
            or prediction_receipt.get("status") != "PREDICTIONS_FROZEN_PENDING_LABEL_EVALUATION"
            or prediction_receipt.get("target_response_loaded") is not False):
        raise ValueError("prediction receipt is incomplete or does not prove label-free freezing")
    prediction_sha = _sha256(prediction_path)
    if prediction_sha != prediction_receipt.get("prediction_file_sha256"):
        raise ValueError("frozen prediction checksum differs from its receipt")
    predictions = pd.read_parquet(prediction_path)
    if {"response", "observed", "target_response"} & set(predictions.columns):
        raise ValueError("frozen prediction file contains target outcome values")
    expected_prediction_columns = {
        "patient_id", "expression_donor_patient_id", "drug_id",
        "predicted_source_pharmaformer", "predicted_source_train_mean",
    }
    if set(predictions.columns) != expected_prediction_columns:
        raise ValueError("frozen prediction columns differ from the preregistered schema")

    manifest_path = input_dir / "manifest.json"
    verification_path = input_dir / "verification.json"
    target_pairs_path = input_dir / "patient_pairs.parquet"
    patient_axis_path = input_dir / "patient_axis.json"
    drug_axis_path = input_dir / "drug_axis.parquet"
    for path in (manifest_path, verification_path, target_pairs_path, patient_axis_path,
                 drug_axis_path, source_pairs_path, source_indices_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    input_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if input_verification.get("status") != "PASS":
        raise ValueError("frozen target input verification is not PASS")
    target_pairs_sha = _sha256(target_pairs_path)
    if input_manifest.get("files_sha256", {}).get("patient_pairs.parquet") != target_pairs_sha:
        raise ValueError("target response table checksum differs from its frozen input contract")
    if input_verification.get("input_manifest_sha256") != _sha256(manifest_path):
        raise ValueError("target input verification receipt does not match the manifest")
    if prediction_receipt.get("inputs_sha256", {}).get("input_manifest") != _sha256(manifest_path):
        raise ValueError("target input manifest differs from the one used for prediction")
    if (prediction_receipt.get("inputs_sha256", {}).get("patient_axis")
            != _sha256(patient_axis_path)
            or prediction_receipt.get("inputs_sha256", {}).get("drug_axis")
            != _sha256(drug_axis_path)):
        raise ValueError("patient or drug identity axis differs from the one used for prediction")
    if (prediction_receipt.get("inputs_sha256", {}).get("source_pairs") != _sha256(source_pairs_path)
            or prediction_receipt.get("inputs_sha256", {}).get("source_indices") != _sha256(source_indices_path)):
        raise ValueError("source baseline assets differ from those used for prediction")

    patient_axis = set(map(str, json.loads(patient_axis_path.read_text(encoding="utf-8"))))
    drug_axis = pd.read_parquet(drug_axis_path)
    drug_ids = set(drug_axis["drug_id"].astype(str))
    if len(patient_axis) != 51 or len(drug_ids) != 14:
        raise ValueError("the frozen PDO axis differs from the preregistered 51-patient/14-drug panel")
    predictions["patient_id"] = predictions["patient_id"].astype(str)
    predictions["expression_donor_patient_id"] = predictions["expression_donor_patient_id"].astype(str)
    predictions["drug_id"] = predictions["drug_id"].astype(str)
    if set(predictions["patient_id"]) != patient_axis or set(predictions["drug_id"]) != drug_ids:
        raise ValueError("frozen prediction axes differ from the admitted PDO contract")
    if not predictions["patient_id"].eq(predictions["expression_donor_patient_id"]).all():
        raise ValueError("true-alignment predictions do not use each recipient's own expression")
    if len(predictions) != len(patient_axis) * len(drug_ids):
        raise ValueError("frozen prediction grid is incomplete")

    source_indices = json.loads(source_indices_path.read_text(encoding="utf-8"))
    if source_indices.get("stage") != "source" or int(source_indices.get("fold_id", -1)) != 0:
        raise ValueError("matched source baseline is not tied to source fold0")
    source_train_ids = {
        str(identity) for identity, role in source_indices["group_assignments"].items()
        if role == "train"
    }
    source_pairs = pd.read_parquet(source_pairs_path).copy()
    source_pairs["sample_id"] = source_pairs["sample_id"].astype(str)
    source_pairs["drug_id"] = source_pairs["drug_id"].astype(str)
    source_train = source_pairs.loc[source_pairs["sample_id"].isin(source_train_ids)]
    if len(source_train_ids) != 577 or source_train.duplicated(["sample_id", "drug_id"]).any():
        raise ValueError("source fold0 training identities or response keys fail the preregistered contract")
    source_mean = source_train.groupby("drug_id", sort=True)["response"].mean()
    expected_mean = predictions["drug_id"].map(source_mean).to_numpy(dtype=np.float64)
    saved_mean = predictions["predicted_source_train_mean"].to_numpy(dtype=np.float64)
    mean_error = float(np.max(np.abs(expected_mean - saved_mean)))
    if set(source_mean.index.astype(str)) != drug_ids or mean_error > 1e-12:
        raise ValueError("saved source baseline differs from the 577-cell fold0 training mean")

    # This is the first point in the pipeline that reads target response labels.
    outcomes = pd.read_parquet(target_pairs_path)
    if not {"patient_id", "drug_id", "response"} <= set(outcomes.columns):
        raise ValueError("target response table is missing required identity or response columns")
    outcomes["patient_id"] = outcomes["patient_id"].astype(str)
    outcomes["drug_id"] = outcomes["drug_id"].astype(str)
    if set(outcomes["patient_id"]) != patient_axis or not set(outcomes["drug_id"]) <= drug_ids:
        raise ValueError("target response keys fall outside the admitted PDO patient/drug panel")
    drug_counts = outcomes.groupby("patient_id")["drug_id"].nunique()
    if len(drug_counts) != 51 or int(drug_counts.min()) < 3:
        raise ValueError("at least one PDO patient has fewer than the preregistered three measured drugs")

    scored_pairs, patient_scores = score_zero_target_predictions(predictions, outcomes)
    valid_pair = np.isfinite(patient_scores["difference_model_minus_mean"].to_numpy(dtype=float))
    paired = paired_resampling_summary(patient_scores)
    permutation_summary, permutation_scores = expression_identity_permutation_test(
        predictions, outcomes, permutations=10000, seed=20260922,
    )
    permutation_summary["mean_baseline_macro_spearman_same_denominator"] = float(
        patient_scores.loc[valid_pair, "spearman_source_train_mean"].mean()
    )
    permutation_summary["true_alignment_minus_source_mean"] = float(
        patient_scores.loc[valid_pair, "difference_model_minus_mean"].mean()
    )

    scored_path = run_dir / "scored_pairs.parquet"
    patient_path = run_dir / "patient_scores.parquet"
    permutation_path = run_dir / "expression_permutation_scores.parquet"
    scored_pairs.to_parquet(scored_path, index=False)
    patient_scores.to_parquet(patient_path, index=False)
    permutation_scores.to_parquet(permutation_path, index=False)
    summary = {
        "experiment_id": "EXP-043",
        "run_id": run_dir.name,
        "status": "EVALUATION_COMPLETE_PENDING_INDEPENDENT_VERIFICATION",
        "primary_metric": "patient-macro Spearman correlation across at least three measured drugs",
        "primary_comparison": "frozen source PharmaFormer minus source fold0 training per-drug mean",
        "n_patients_in_contract": len(patient_axis),
        "n_patients_with_target_labels": int(outcomes["patient_id"].nunique()),
        "n_target_patient_drug_labels": int(len(outcomes)),
        "n_scored_patients_model_and_baseline": int(valid_pair.sum()),
        "n_scored_patient_drug_pairs": int(len(scored_pairs)),
        "n_source_training_identities": int(len(source_train_ids)),
        "source_mean_max_abs_error": mean_error,
        "prediction_file_sha256": prediction_sha,
        "target_response_sha256": target_pairs_sha,
        "target_responses_loaded_after_predictions_frozen": True,
        "paired_patient_statistics": paired,
        "expression_identity_permutation": permutation_summary,
        "files_sha256": {
            scored_path.name: _sha256(scored_path),
            patient_path.name: _sha256(patient_path),
            permutation_path.name: _sha256(permutation_path),
        },
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    _write_json(summary_path, summary)
    _write_json(run_dir / "evaluation_provenance.json", {
        "experiment_id": "EXP-043", "run_id": run_dir.name,
        "status": "TARGET_LABELS_READ_AFTER_PREDICTION_HASH_VERIFIED",
        "prediction_summary_sha256": _sha256(prediction_receipt_path),
        "prediction_file_sha256": prediction_sha,
        "target_response_sha256": target_pairs_sha,
        "target_input_manifest_sha256": _sha256(manifest_path),
        "source_pairs_sha256": _sha256(source_pairs_path),
        "source_indices_sha256": _sha256(source_indices_path),
        "evaluation_code_sha256": _sha256(Path(__file__).resolve()),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--source-pairs", type=Path, required=True)
    parser.add_argument("--source-indices", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = evaluate(args.run_dir, args.input_dir, args.source_pairs,
                          args.source_indices)
    except Exception as error:
        _write_json(args.run_dir / "EVALUATION_FAILURE.json", {
            "experiment_id": "EXP-043", "run_id": args.run_dir.name,
            "status": "FAIL", "error_type": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(),
        })
        raise
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
