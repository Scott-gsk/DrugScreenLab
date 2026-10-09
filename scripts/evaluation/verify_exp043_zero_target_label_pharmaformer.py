"""Independently verify EXP-043 frozen inference, labels, metrics, and permutations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _load_official(path: Path):
    spec = importlib.util.spec_from_file_location("pharmaformer_official_verifier", path)
    if spec is None or spec.loader is None:
        raise ValueError("official PharmaFormer source could not be loaded for verification")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rho(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, str | None]:
    if len(observed) < 3:
        return np.nan, "FEWER_THAN_THREE_DRUGS"
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        return np.nan, "NONFINITE_VALUE"
    if np.ptp(observed) == 0:
        return np.nan, "CONSTANT_OBSERVED"
    if np.ptp(predicted) == 0:
        return np.nan, "CONSTANT_PREDICTED"
    value = float(spearmanr(observed, predicted).statistic)
    return (value, None) if np.isfinite(value) else (np.nan, "UNDEFINED_CORRELATION")


def _reference_patient_scores(predictions: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    joined = outcomes.merge(predictions, on=["patient_id", "drug_id"], how="left",
                            validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("a target label has no frozen prediction")
    joined = joined.drop(columns="_merge")
    rows = []
    for patient_id, group in joined.groupby("patient_id", sort=True):
        observed = group["response"].to_numpy(dtype=np.float64)
        model_score, model_reason = _rho(
            observed, group["predicted_source_pharmaformer"].to_numpy(dtype=np.float64))
        mean_score, mean_reason = _rho(
            observed, group["predicted_source_train_mean"].to_numpy(dtype=np.float64))
        rows.append({
            "patient_id": str(patient_id), "n_drugs": int(group["drug_id"].nunique()),
            "spearman_source_pharmaformer": model_score,
            "reason_source_pharmaformer": model_reason,
            "spearman_source_train_mean": mean_score,
            "reason_source_train_mean": mean_reason,
            "difference_model_minus_mean": model_score - mean_score,
        })
    return joined, pd.DataFrame(rows)


def _reference_paired_summary(scores: pd.DataFrame) -> dict[str, Any]:
    values = scores["difference_model_minus_mean"].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("no valid patient-level paired ranking differences")
    rng = np.random.default_rng(20260922)
    bootstrap = np.empty(2000, dtype=np.float64)
    for index in range(len(bootstrap)):
        bootstrap[index] = values[rng.integers(0, len(values), len(values))].mean()
    signs = rng.choice(np.array([-1.0, 1.0]), size=(10000, len(values)))
    null = np.abs((signs * values[None, :]).mean(axis=1))
    observed = abs(float(values.mean()))
    return {
        "estimate_model_minus_source_mean": float(values.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "sign_flip_p_two_sided": float((1 + np.count_nonzero(null >= observed)) / 10001),
        "n_patients": int(len(values)), "bootstrap_replicates": 2000,
        "sign_flip_replicates": 10000, "seed": 20260922,
    }


def _reference_permutation(predictions: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    recipients = sorted(outcomes["patient_id"].astype(str).unique())
    donors = sorted(predictions["expression_donor_patient_id"].astype(str).unique())
    if recipients != donors:
        raise ValueError("permutation donor identities differ from target patient identities")
    donor_predictions = {
        donor: group.set_index("drug_id")["predicted_source_pharmaformer"]
        for donor, group in predictions.groupby("expression_donor_patient_id", sort=True)
    }
    outcome_groups = {patient: group for patient, group in outcomes.groupby("patient_id", sort=True)}
    matrix = np.full((len(recipients), len(donors)), np.nan, dtype=np.float64)
    for i, patient in enumerate(recipients):
        group = outcome_groups[patient]
        drug_ids = group["drug_id"].astype(str).tolist()
        observed = group["response"].to_numpy(dtype=np.float64)
        for j, donor in enumerate(donors):
            donor_values = donor_predictions[donor]
            if not set(drug_ids) <= set(donor_values.index.astype(str)):
                raise ValueError(f"expression donor {donor} lacks a target measured drug")
            matrix[i, j] = _rho(observed, donor_values.reindex(drug_ids).to_numpy(dtype=np.float64))[0]
    diagonal = matrix[np.arange(len(recipients)), np.arange(len(recipients))]
    fixed = np.isfinite(diagonal)
    if not fixed.any():
        raise ValueError("true alignment has no valid patient scores")
    observed_macro = float(diagonal[fixed].mean())
    rng = np.random.default_rng(20260922)
    rows = []
    for index in range(10000):
        assignment = rng.permutation(len(donors))
        draw = matrix[np.arange(len(recipients)), assignment]
        finite = np.isfinite(draw)
        comparable = bool(finite[fixed].all())
        rows.append({
            "permutation": index,
            "macro_spearman": float(draw[fixed].mean()) if comparable else np.nan,
            "n_fixed_patients_scored": int(fixed.sum()),
            "n_finite_donor_patient_scores": int(finite.sum()),
            "comparable_to_observed_denominator": comparable,
        })
    table = pd.DataFrame(rows)
    valid = table.loc[table["comparable_to_observed_denominator"], "macro_spearman"]
    if not len(valid):
        raise ValueError("no permutation draws have a denominator comparable to the observed score")
    summary = {
        "observed_true_alignment_macro_spearman": observed_macro,
        "permutation_null_mean": float(valid.mean()),
        "permutation_null_median": float(valid.median()),
        "permutation_null_95th_percentile": float(valid.quantile(0.95)),
        "upper_tail_empirical_p": float((1 + np.count_nonzero(valid.to_numpy() >= observed_macro))
                                         / (len(valid) + 1)),
        "n_fixed_patients_scored": int(fixed.sum()),
        "n_permutations_requested": 10000,
        "n_permutations_with_comparable_denominator": int(len(valid)),
        "seed": 20260922,
    }
    return summary, table


def verify(run_dir: Path, input_dir: Path, source_pairs_path: Path,
           source_indices_path: Path, checkpoint_path: Path,
           official_source: Path) -> dict[str, Any]:
    """Refit predictions and recalculate all summaries from frozen identities."""
    verification_path = run_dir / "verification.json"
    if verification_path.exists():
        raise FileExistsError(f"verification receipt already exists: {verification_path}")
    prediction_path = run_dir / "predictions.parquet"
    prediction_summary_path = run_dir / "prediction_summary.json"
    prediction_provenance_path = run_dir / "run_provenance.json"
    evaluation_summary_path = run_dir / "evaluation_summary.json"
    evaluation_provenance_path = run_dir / "evaluation_provenance.json"
    prediction_summary = json.loads(prediction_summary_path.read_text(encoding="utf-8"))
    provenance = json.loads(prediction_provenance_path.read_text(encoding="utf-8"))
    evaluation_summary = json.loads(evaluation_summary_path.read_text(encoding="utf-8"))
    evaluation_provenance = json.loads(evaluation_provenance_path.read_text(encoding="utf-8"))
    if prediction_summary.get("status") != "PREDICTIONS_FROZEN_PENDING_LABEL_EVALUATION":
        raise ValueError("prediction receipt is not frozen in the no-label state")
    if provenance.get("status") != "PREDICTIONS_FROZEN_NO_TARGET_LABELS_LOADED":
        raise ValueError("prediction provenance does not confirm label-free inference")
    if evaluation_summary.get("status") != "EVALUATION_COMPLETE_PENDING_INDEPENDENT_VERIFICATION":
        raise ValueError("evaluation receipt is incomplete")
    if _sha256(prediction_path) != prediction_summary.get("prediction_file_sha256"):
        raise ValueError("frozen prediction checksum mismatch")
    if evaluation_provenance.get("prediction_file_sha256") != _sha256(prediction_path):
        raise ValueError("evaluation did not use the exact frozen prediction file")
    if (evaluation_summary.get("prediction_file_sha256") != _sha256(prediction_path)
            or evaluation_summary.get("target_response_sha256")
            != evaluation_provenance.get("target_response_sha256")):
        raise ValueError("evaluation summary identifies different frozen predictions or target labels")

    root = Path(__file__).resolve().parents[2]
    input_manifest_path = input_dir / "manifest.json"
    input_verification_path = input_dir / "verification.json"
    patient_axis_path = input_dir / "patient_axis.json"
    patient_expression_path = input_dir / "patient_expression.npy"
    drug_axis_path = input_dir / "drug_axis.parquet"
    drug_vectors_path = input_dir / "drug_ordinal128.npy"
    target_pairs_path = input_dir / "patient_pairs.parquet"
    asset_dir = root / "artifacts" / "experiments" / "EXP-036" / "20260923_pharmaformer_asset_audit"
    asset_audit_path = asset_dir / "asset_audit.json"
    strict_smoke_path = asset_dir / "strict_reload_smoke.json"
    real_smoke_path = asset_dir / "real_input_smoke.json"
    inputs = {
        "input_manifest": input_manifest_path,
        "patient_axis": patient_axis_path,
        "patient_expression": patient_expression_path,
        "drug_axis": drug_axis_path,
        "drug_vectors": drug_vectors_path,
        "source_pairs": source_pairs_path,
        "source_indices": source_indices_path,
        "source_checkpoint": checkpoint_path,
        "official_source": official_source,
        "input_verification": input_verification_path,
        "asset_audit": asset_audit_path,
        "strict_reload_smoke": strict_smoke_path,
        "real_input_smoke": real_smoke_path,
    }
    actual_hashes = {name: _sha256(path) for name, path in inputs.items()}
    if actual_hashes != prediction_summary.get("inputs_sha256"):
        raise ValueError("independent input checksums differ from the frozen prediction receipt")
    if actual_hashes != provenance.get("inputs_sha256"):
        raise ValueError("prediction provenance input checksums differ from the frozen receipt")
    predictor_script = root / "scripts" / "modeling" / "exp043_zero_label_pharmaformer.py"
    if provenance.get("code_sha256") != _sha256(predictor_script):
        raise ValueError("prediction code changed after the frozen run")
    if b"patient_pairs.parquet" in predictor_script.read_bytes():
        raise ValueError("prediction-only code references the target response table")
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    input_verification = json.loads(input_verification_path.read_text(encoding="utf-8"))
    asset_audit = json.loads(asset_audit_path.read_text(encoding="utf-8"))
    fold0_audit = next(row for row in asset_audit["checkpoints"] if int(row["fold"]) == 0)
    strict_smoke = json.loads(strict_smoke_path.read_text(encoding="utf-8"))
    real_smoke = json.loads(real_smoke_path.read_text(encoding="utf-8"))
    checkpoint_sha = _sha256(checkpoint_path)
    official_sha = _sha256(official_source)
    if (fold0_audit.get("checkpoint_sha256") != checkpoint_sha
            or int(fold0_audit.get("n_source_fit_cells", 0)) != 577
            or int(fold0_audit.get("n_target_drugs_covered", 0)) != 14
            or strict_smoke.get("status") != "STRICT_OFFICIAL_RELOAD_PASS_ENGINEERING_ONLY"
            or real_smoke.get("status") != "STRICT_RELOAD_AND_REAL_INPUT_SMOKE_PASS_ENGINEERING_ONLY"
            or strict_smoke.get("checkpoint_sha256") != checkpoint_sha
            or real_smoke.get("checkpoint_sha256") != checkpoint_sha
            or strict_smoke.get("official_source_sha256") != official_sha
            or real_smoke.get("official_source_sha256") != official_sha):
        raise ValueError("source weight audit or official model smoke receipts are inconsistent")
    if (input_verification.get("status") != "PASS"
            or input_verification.get("input_manifest_sha256") != _sha256(input_manifest_path)
            or input_verification.get("source_checkpoint_sha256") != _sha256(checkpoint_path)):
        raise ValueError("frozen input verification receipt is inconsistent")
    if input_manifest.get("files_sha256", {}).get("patient_pairs.parquet") != _sha256(target_pairs_path):
        raise ValueError("target response table differs from the frozen input contract")
    if evaluation_provenance.get("target_response_sha256") != _sha256(target_pairs_path):
        raise ValueError("evaluation receipt identifies a different target response table")
    if evaluation_provenance.get("target_input_manifest_sha256") != _sha256(input_manifest_path):
        raise ValueError("evaluation receipt identifies a different target input manifest")
    if (evaluation_provenance.get("source_pairs_sha256") != _sha256(source_pairs_path)
            or evaluation_provenance.get("source_indices_sha256") != _sha256(source_indices_path)):
        raise ValueError("evaluation receipt identifies different source baseline assets")
    if evaluation_provenance.get("evaluation_code_sha256") != _sha256(
            root / "scripts" / "evaluation" / "exp043_zero_target_label_pharmaformer.py"):
        raise ValueError("evaluation code changed after the formal run")

    predictions = pd.read_parquet(prediction_path)
    forbidden = {"response", "observed", "target_response"}
    if forbidden & set(predictions.columns):
        raise ValueError("frozen predictions contain target outcome values")
    expected_columns = {
        "patient_id", "expression_donor_patient_id", "drug_id",
        "predicted_source_pharmaformer", "predicted_source_train_mean",
    }
    if set(predictions.columns) != expected_columns:
        raise ValueError("frozen prediction schema differs from the preregistration")
    predictions["patient_id"] = predictions["patient_id"].astype(str)
    predictions["expression_donor_patient_id"] = predictions["expression_donor_patient_id"].astype(str)
    predictions["drug_id"] = predictions["drug_id"].astype(str)
    patient_axis = list(map(str, json.loads(patient_axis_path.read_text(encoding="utf-8"))))
    drugs = pd.read_parquet(drug_axis_path)
    drug_ids = drugs["drug_id"].astype(str).tolist()
    if len(set(patient_axis)) != 51 or len(set(drug_ids)) != 14:
        raise ValueError("admitted patient/drug axes differ from the fixed contract")
    if len(predictions) != 51 * 14 or predictions.duplicated(["patient_id", "drug_id"]).any():
        raise ValueError("prediction keys are incomplete or duplicated")
    expected_keys = pd.MultiIndex.from_product([patient_axis, drug_ids], names=["patient_id", "drug_id"])
    actual_keys = pd.MultiIndex.from_frame(predictions[["patient_id", "drug_id"]])
    if set(actual_keys) != set(expected_keys):
        raise ValueError("prediction keys do not exactly cover the patient/drug Cartesian product")
    if not predictions["patient_id"].eq(predictions["expression_donor_patient_id"]).all():
        raise ValueError("prediction expression donor does not match its target recipient")

    source_indices = json.loads(source_indices_path.read_text(encoding="utf-8"))
    if source_indices.get("stage") != "source" or int(source_indices.get("fold_id", -1)) != 0:
        raise ValueError("source fold index differs from the preregistered fold0 asset")
    train_ids = {str(key) for key, role in source_indices["group_assignments"].items() if role == "train"}
    if len(train_ids) != 577:
        raise ValueError("source fold0 training identity count is not 577")
    source_pairs = pd.read_parquet(source_pairs_path).copy()
    source_pairs["sample_id"] = source_pairs["sample_id"].astype(str)
    source_pairs["drug_id"] = source_pairs["drug_id"].astype(str)
    train_pairs = source_pairs.loc[source_pairs["sample_id"].isin(train_ids)]
    if train_pairs.duplicated(["sample_id", "drug_id"]).any():
        raise ValueError("duplicate source training sample/drug response key")
    means = train_pairs.groupby("drug_id", sort=True)["response"].mean()
    expected_baseline = predictions["drug_id"].map(means).to_numpy(dtype=np.float64)
    baseline_error = float(np.max(np.abs(
        expected_baseline - predictions["predicted_source_train_mean"].to_numpy(dtype=np.float64)
    )))
    if set(means.index.astype(str)) != set(drug_ids) or baseline_error > 1e-12:
        raise ValueError("frozen baseline differs from the source fold0 training means")

    # Recompute the entire 51 x 14 model grid independently on CPU before using target labels.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint.get("variant") != "M2" or checkpoint.get("model_variant") != "CombinedModel"
            or checkpoint.get("upstream_commit") != "f526ddda4091151f082b38367571a94823314aba"):
        raise ValueError("source checkpoint architecture or official revision differs from the frozen contract")
    if set(map(str, checkpoint["gene_transform"]["fit_sample_ids"])) != train_ids:
        raise ValueError("checkpoint preprocessing identities differ from the source fold0 training set")
    config = checkpoint["model_config"]
    official = _load_official(official_source)
    model = official.CombinedModel(
        gene_input_size=config["gene_input_size"], gene_hidden_size=config["gene_hidden_size"],
        drug_hidden_size=config["drug_hidden_size"], feature_dim=config["feature_dim"],
        nhead=config["nhead"], num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"], dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model.to("cpu")
    expression = np.load(patient_expression_path, mmap_mode="r")
    drug_vectors = np.load(drug_vectors_path, mmap_mode="r")
    if expression.shape != (len(patient_axis), int(config["gene_input_size"])):
        raise ValueError("frozen expression matrix shape differs from official model input")
    if drug_vectors.shape != (len(drug_ids), 128):
        raise ValueError("frozen drug vector shape differs from the fixed 128-token drug input")
    expression_tensor = torch.from_numpy(np.array(expression, dtype=np.float32, copy=True))
    drug_tensor = torch.from_numpy(np.array(drug_vectors, dtype=np.float32, copy=True))
    grid_patient = np.repeat(np.arange(len(patient_axis), dtype=np.int64), len(drug_ids))
    grid_drug = np.tile(np.arange(len(drug_ids), dtype=np.int64), len(patient_axis))
    expected_predictions = np.empty(len(grid_patient), dtype=np.float32)
    torch.set_num_threads(1)
    with threadpool_limits(limits=1), torch.inference_mode():
        for begin in range(0, len(grid_patient), 32):
            end = min(begin + 32, len(grid_patient))
            x = expression_tensor[torch.from_numpy(grid_patient[begin:end])]
            d = drug_tensor[torch.from_numpy(grid_drug[begin:end])]
            expected_predictions[begin:end] = model(x, d).flatten().cpu().numpy()
    expected = pd.DataFrame({
        "patient_id": [patient_axis[i] for i in grid_patient],
        "drug_id": [drug_ids[i] for i in grid_drug],
        "expected": expected_predictions,
    })
    joined_predictions = predictions.merge(expected, on=["patient_id", "drug_id"], validate="one_to_one")
    inference_error = float(np.max(np.abs(
        joined_predictions["predicted_source_pharmaformer"].to_numpy(dtype=np.float64)
        - joined_predictions["expected"].to_numpy(dtype=np.float64)
    )))
    if inference_error > 1e-6:
        raise ValueError(f"independent model inference differs from frozen values (max error {inference_error})")

    # Access target labels only after the prediction hash, keys, and model values passed.
    outcomes = pd.read_parquet(target_pairs_path).copy()
    outcomes["patient_id"] = outcomes["patient_id"].astype(str)
    outcomes["drug_id"] = outcomes["drug_id"].astype(str)
    if outcomes.duplicated(["patient_id", "drug_id"]).any():
        raise ValueError("duplicate target patient/drug response key")
    joined_reference, reference_patient_scores = _reference_patient_scores(predictions, outcomes)
    saved_pairs = pd.read_parquet(run_dir / "scored_pairs.parquet")
    saved_patients = pd.read_parquet(run_dir / "patient_scores.parquet")
    saved_permutations = pd.read_parquet(run_dir / "expression_permutation_scores.parquet")
    saved_pairs["patient_id"] = saved_pairs["patient_id"].astype(str)
    saved_pairs["drug_id"] = saved_pairs["drug_id"].astype(str)
    expected_pair_keys = set(pd.MultiIndex.from_frame(joined_reference[["patient_id", "drug_id"]]))
    saved_pair_keys = set(pd.MultiIndex.from_frame(saved_pairs[["patient_id", "drug_id"]]))
    if saved_pairs.duplicated(["patient_id", "drug_id"]).any() or saved_pair_keys != expected_pair_keys:
        raise ValueError("saved scored patient/drug keys differ from the exact keyed label join")
    reference_pairs = joined_reference.set_index(["patient_id", "drug_id"]).sort_index()
    saved_pairs = saved_pairs.set_index(["patient_id", "drug_id"]).sort_index()
    for column in ("response", "predicted_source_pharmaformer", "predicted_source_train_mean"):
        value_error = float(np.max(np.abs(reference_pairs[column].to_numpy(dtype=float)
                                          - saved_pairs[column].to_numpy(dtype=float))))
        if value_error > 1e-12:
            raise ValueError(f"saved scored-pair value differs from independent keyed join: {column}")
    reference_patient_scores["patient_id"] = reference_patient_scores["patient_id"].astype(str)
    saved_patients["patient_id"] = saved_patients["patient_id"].astype(str)
    if (saved_patients["patient_id"].duplicated().any()
            or set(reference_patient_scores["patient_id"]) != set(saved_patients["patient_id"])):
        raise ValueError("saved patient score keys differ from independently scored identities")
    merged_scores = reference_patient_scores.merge(saved_patients, on="patient_id", suffixes=("_ref", "_saved"),
                                                   validate="one_to_one")
    score_error = float(np.nanmax(np.abs(
        merged_scores["spearman_source_pharmaformer_ref"].to_numpy(dtype=float)
        - merged_scores["spearman_source_pharmaformer_saved"].to_numpy(dtype=float)
    )))
    baseline_score_error = float(np.nanmax(np.abs(
        merged_scores["spearman_source_train_mean_ref"].to_numpy(dtype=float)
        - merged_scores["spearman_source_train_mean_saved"].to_numpy(dtype=float)
    )))
    difference_error = float(np.nanmax(np.abs(
        merged_scores["difference_model_minus_mean_ref"].to_numpy(dtype=float)
        - merged_scores["difference_model_minus_mean_saved"].to_numpy(dtype=float)
    )))
    if max(score_error, baseline_score_error, difference_error) > 1e-12:
        raise ValueError("saved patient scores differ from independent keyed rescoring")
    paired_reference = _reference_paired_summary(reference_patient_scores)
    paired_saved = evaluation_summary.get("paired_patient_statistics", {})
    for key, value in paired_reference.items():
        if key not in paired_saved or not np.isclose(float(paired_saved[key]), value, rtol=0, atol=1e-12):
            raise ValueError(f"paired patient statistic differs from independent calculation: {key}")
    permutation_reference, permutation_table = _reference_permutation(predictions, outcomes)
    for key, value in permutation_reference.items():
        saved_value = evaluation_summary.get("expression_identity_permutation", {}).get(key)
        if saved_value is None or not np.isclose(float(saved_value), value, rtol=0, atol=1e-12):
            raise ValueError(f"expression permutation statistic differs from independent calculation: {key}")
    if len(saved_permutations) != len(permutation_table):
        raise ValueError("saved expression permutation table is incomplete")
    for column in ("permutation", "n_fixed_patients_scored", "n_finite_donor_patient_scores",
                   "comparable_to_observed_denominator"):
        if not saved_permutations[column].equals(permutation_table[column]):
            raise ValueError(f"saved expression permutation denominator differs: {column}")
    if not np.allclose(saved_permutations["macro_spearman"].to_numpy(dtype=float),
                       permutation_table["macro_spearman"].to_numpy(dtype=float),
                       rtol=0, atol=1e-12, equal_nan=True):
        raise ValueError("saved expression permutation scores differ from the independent randomization")

    for filename, digest in evaluation_summary.get("files_sha256", {}).items():
        if _sha256(run_dir / filename) != digest:
            raise ValueError(f"evaluation output checksum mismatch: {filename}")
    result = {
        "experiment_id": "EXP-043", "run_id": run_dir.name,
        "status": "PASS", "verified_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_file_sha256": _sha256(prediction_path),
        "target_response_sha256": _sha256(target_pairs_path),
        "n_verified_prediction_keys": int(len(predictions)),
        "n_target_patient_drug_labels": int(len(outcomes)),
        "n_patients_scored": int(reference_patient_scores["difference_model_minus_mean"].notna().sum()),
        "max_prediction_abs_error": inference_error,
        "max_source_mean_abs_error": baseline_error,
        "max_patient_model_score_abs_error": score_error,
        "max_patient_baseline_score_abs_error": baseline_score_error,
        "max_patient_difference_abs_error": difference_error,
        "paired_statistics_independently_recomputed": True,
        "expression_permutation_independently_recomputed": True,
        "target_labels_read_after_predictions_verified": True,
    }
    _write_json(verification_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--source-pairs", type=Path, required=True)
    parser.add_argument("--source-indices", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.run_dir, args.input_dir, args.source_pairs,
                        args.source_indices, args.checkpoint, args.official_source)
    except Exception as error:
        failure = {
            "experiment_id": "EXP-043", "run_id": args.run_dir.name,
            "status": "FAIL", "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(args.run_dir / "VERIFICATION_FAILURE.json", failure)
        raise
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
