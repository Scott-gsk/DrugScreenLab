"""Independent numerical recomputation for the frozen EXP-044 predictions.

This verifier deliberately has no argument or code path for PRISM response
tables. It reloads the frozen source model and response-blind features, rebuilds
the source-only drug-mean baseline, and compares keyed values to the immutable
prediction parquet.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits


EXPECTED_CHECKPOINT_SHA256 = "1dbe98a2f52c4affdcee091cf011f3e52e29b51992c9bb3d57a60b08fa464de6"
EXPECTED_OFFICIAL_SOURCE_SHA256 = "c7c1dc2389e4050df1f007093a6042620be1c0abdeb4792f04d583fe2820f567"
EXPECTED_CELLS = 352
EXPECTED_DRUGS = 13
PREDICTION_ATOL = 1e-6
BASELINE_ATOL = 1e-12

PREDICTION_COLUMNS = (
    "cell_line_id",
    "drug_id",
    "predicted_source_pharmaformer",
    "predicted_source_train_mean",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_official_model_module(source_path: Path):
    spec = importlib.util.spec_from_file_location("pharmaformer_official_exp044_recheck", source_path)
    if spec is None or spec.loader is None:
        raise ValueError("could not load the pinned official PharmaFormer source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_train_means(source_pairs: pd.DataFrame, train_ids: set[str],
                       drug_ids: list[str]) -> tuple[dict[str, float], int]:
    required = {"sample_id", "drug_id", "response"}
    missing = required - set(source_pairs.columns)
    if missing:
        raise ValueError(f"source training table is missing columns: {sorted(missing)}")
    frame = source_pairs.loc[:, ["sample_id", "drug_id", "response"]].copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["drug_id"] = frame["drug_id"].astype(str)
    if frame.duplicated(["sample_id", "drug_id"]).any():
        raise ValueError("source training table has duplicate cell/drug keys")
    frame = frame.loc[frame["sample_id"].isin(train_ids) & frame["drug_id"].isin(set(drug_ids))]
    response = frame["response"].to_numpy(dtype=np.float64)
    if frame.empty or not np.isfinite(response).all():
        raise ValueError("source-fold responses for the frozen drug panel are empty or non-finite")
    means = frame.groupby("drug_id", sort=True)["response"].mean()
    result = {str(key): float(value) for key, value in means.items()}
    if set(result) != set(drug_ids):
        raise ValueError("source-training means do not cover the exact frozen drug panel")
    return result, len(frame)


def compare_keyed_predictions(actual: pd.DataFrame, expected: pd.DataFrame, *,
                              prediction_atol: float = PREDICTION_ATOL,
                              baseline_atol: float = BASELINE_ATOL) -> dict[str, Any]:
    """Compare complete cell/drug grids independent of row ordering."""
    required = set(PREDICTION_COLUMNS)
    if required - set(actual.columns) or required - set(expected.columns):
        raise ValueError("prediction tables do not contain the frozen four-column schema")
    actual_frame = actual.loc[:, PREDICTION_COLUMNS].copy()
    expected_frame = expected.loc[:, PREDICTION_COLUMNS].copy()
    for frame in (actual_frame, expected_frame):
        frame["cell_line_id"] = frame["cell_line_id"].astype(str)
        frame["drug_id"] = frame["drug_id"].astype(str)
        if frame.duplicated(["cell_line_id", "drug_id"]).any():
            raise ValueError("prediction table contains duplicate cell/drug keys")
    joined = actual_frame.merge(
        expected_frame,
        on=["cell_line_id", "drug_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_actual", "_expected"),
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("actual and recomputed predictions do not have identical keyed coverage")
    pred_actual = joined["predicted_source_pharmaformer_actual"].to_numpy(dtype=np.float64)
    pred_expected = joined["predicted_source_pharmaformer_expected"].to_numpy(dtype=np.float64)
    base_actual = joined["predicted_source_train_mean_actual"].to_numpy(dtype=np.float64)
    base_expected = joined["predicted_source_train_mean_expected"].to_numpy(dtype=np.float64)
    if not (np.isfinite(pred_actual).all() and np.isfinite(pred_expected).all()
            and np.isfinite(base_actual).all() and np.isfinite(base_expected).all()):
        raise ValueError("prediction or source-mean baseline contains non-finite values")
    pred_abs = np.abs(pred_actual - pred_expected)
    base_abs = np.abs(base_actual - base_expected)
    if np.any(pred_abs > prediction_atol):
        raise ValueError("recomputed model predictions exceed the predeclared absolute tolerance")
    if np.any(base_abs > baseline_atol):
        raise ValueError("recomputed source means exceed the predeclared absolute tolerance")
    return {
        "n_prediction_keys": int(len(joined)),
        "n_exact_model_predictions": int(np.count_nonzero(pred_abs == 0.0)),
        "max_abs_model_prediction_error": float(pred_abs.max(initial=0.0)),
        "model_prediction_absolute_tolerance": float(prediction_atol),
        "max_abs_source_mean_error": float(base_abs.max(initial=0.0)),
        "source_mean_absolute_tolerance": float(baseline_atol),
        "source_mean_values_exact": bool(np.all(base_abs == 0.0)),
    }


def run(*, input_dir: Path, input_verification_path: Path, identity_key_audit_path: Path,
        source_pairs_path: Path, source_indices_path: Path, checkpoint_path: Path,
        official_source_path: Path, prediction_dir: Path, output_receipt: Path,
        batch_size: int = 32) -> dict[str, Any]:
    if output_receipt.exists():
        raise FileExistsError(f"refusing to overwrite a numerical-recheck receipt: {output_receipt}")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if sha256(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint differs from the preregistered EXP-044 source checkpoint")
    if sha256(official_source_path) != EXPECTED_OFFICIAL_SOURCE_SHA256:
        raise ValueError("official model source differs from the audited source snapshot")

    manifest_path = input_dir / "manifest.json"
    cell_axis_path = input_dir / "cell_axis.json"
    expression_path = input_dir / "cell_expression.npy"
    drug_axis_path = input_dir / "drug_axis.parquet"
    drug_vectors_path = input_dir / "drug_ordinal128.npy"
    prediction_path = prediction_dir / "predictions.parquet"
    summary_path = prediction_dir / "prediction_summary.json"
    paths = {
        "input_manifest": manifest_path,
        "input_verification": input_verification_path,
        "identity_key_audit": identity_key_audit_path,
        "cell_axis": cell_axis_path,
        "cell_expression": expression_path,
        "drug_axis": drug_axis_path,
        "drug_vectors": drug_vectors_path,
        "source_pairs": source_pairs_path,
        "source_indices": source_indices_path,
        "source_checkpoint": checkpoint_path,
        "official_source": official_source_path,
        "frozen_predictions": prediction_path,
        "prediction_summary": summary_path,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"numerical recheck inputs are missing: {missing}")

    verification = read_json(input_verification_path)
    manifest = read_json(manifest_path)
    key_audit = read_json(identity_key_audit_path)
    summary = read_json(summary_path)
    if (verification.get("status") != "PASS_RESPONSE_BLIND_IDENTITY_RESOLVED"
            or verification.get("target_response_values_read") is not False):
        raise ValueError("EXP-044 input verification is not a response-blind pass")
    if (manifest.get("experiment_id") != "EXP-044"
            or manifest.get("target_response_values_read") is not False
            or manifest.get("response_label_files_opened") != []
            or manifest.get("source_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256):
        raise ValueError("EXP-044 materialized manifest is not the frozen response-blind input set")
    if (key_audit.get("status") != "PASS_RESPONSE_BLIND_IDENTITY_KEY_COVERAGE"
            or key_audit.get("response_value_columns_requested") != []
            or key_audit.get("target_response_values_read") is not False
            or key_audit.get("n_revised_strict_unseen_cells") != EXPECTED_CELLS
            or key_audit.get("n_prism_panel_drugs") != EXPECTED_DRUGS):
        raise ValueError("identity/key-coverage gate does not match the revised 352 by 13 cohort")
    if (summary.get("status") != "PREDICTIONS_FROZEN_NO_PRISM_LABELS_LOADED_PENDING_RESPONSE_EVALUATION"
            or summary.get("target_response_values_read") is not False
            or summary.get("target_response_files_opened") != []
            or summary.get("prediction_file_sha256") != sha256(prediction_path)
            or summary.get("n_strict_unseen_cohort_cells") != EXPECTED_CELLS
            or summary.get("n_prism_panel_drugs") != EXPECTED_DRUGS):
        raise ValueError("frozen prediction summary does not bind the expected response-blind output")
    if sha256(manifest_path) != verification.get("input_manifest_sha256"):
        raise ValueError("input verification does not bind the current input manifest")

    cells = [str(item) for item in json.loads(cell_axis_path.read_text(encoding="utf-8"))]
    drugs_frame = pd.read_parquet(drug_axis_path)
    if "drug_id" not in drugs_frame.columns:
        raise ValueError("frozen drug axis has no drug_id column")
    drugs = drugs_frame["drug_id"].astype(str).tolist()
    if (len(cells) != EXPECTED_CELLS or len(set(cells)) != EXPECTED_CELLS
            or len(drugs) != EXPECTED_DRUGS or len(set(drugs)) != EXPECTED_DRUGS):
        raise ValueError("frozen feature axes do not have exactly 352 unique cells by 13 unique drugs")

    expression = np.load(expression_path, mmap_mode="r", allow_pickle=False)
    drug_vectors = np.load(drug_vectors_path, mmap_mode="r", allow_pickle=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint.get("variant") != "M2" or checkpoint.get("model_variant") != "CombinedModel"
            or checkpoint.get("upstream_commit") != "f526ddda4091151f082b38367571a94823314aba"):
        raise ValueError("source checkpoint is not the audited official M2 fold-0 model")
    if (expression.ndim != 2 or expression.shape[0] != len(cells)
            or expression.shape[1] != int(checkpoint["model_config"]["gene_input_size"])
            or drug_vectors.shape != (len(drugs), 128)
            or not np.isfinite(np.asarray(expression)).all()
            or not np.isfinite(np.asarray(drug_vectors)).all()):
        raise ValueError("materialized expression or drug vectors do not match the model axes")

    index_receipt = read_json(source_indices_path)
    if index_receipt.get("stage") != "source" or int(index_receipt.get("fold_id", -1)) != 0:
        raise ValueError("source index receipt is not fold 0")
    train_ids = {
        str(identity) for identity, role in index_receipt.get("group_assignments", {}).items()
        if role == "train"
    }
    fitted_ids = {str(identity) for identity in checkpoint["gene_transform"]["fit_sample_ids"]}
    if len(train_ids) != 577 or train_ids != fitted_ids:
        raise ValueError("checkpoint preprocessing identities differ from the frozen 577 source IDs")

    # Only the source GDSC training table is opened here. The target PRISM response
    # table has no CLI parameter and is never loaded by this module.
    source_pairs = pd.read_parquet(source_pairs_path, columns=["sample_id", "drug_id", "response"])
    means, n_source_pairs = source_train_means(source_pairs, train_ids, drugs)

    official = load_official_model_module(official_source_path)
    config = checkpoint["model_config"]
    model = official.CombinedModel(
        gene_input_size=config["gene_input_size"],
        gene_hidden_size=config["gene_hidden_size"],
        drug_hidden_size=config["drug_hidden_size"],
        feature_dim=config["feature_dim"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().to(torch.device("cpu"))

    cell_positions = np.repeat(np.arange(len(cells), dtype=np.int64), len(drugs))
    drug_positions = np.tile(np.arange(len(drugs), dtype=np.int64), len(cells))
    predictions = np.empty(len(cell_positions), dtype=np.float32)
    expression_tensor = torch.from_numpy(np.array(expression, dtype=np.float32, copy=True))
    drug_tensor = torch.from_numpy(np.array(drug_vectors, dtype=np.float32, copy=True))
    torch.set_num_threads(1)
    with threadpool_limits(limits=1), torch.inference_mode():
        for start in range(0, len(cell_positions), batch_size):
            stop = min(start + batch_size, len(cell_positions))
            cell_index = torch.from_numpy(cell_positions[start:stop])
            drug_index = torch.from_numpy(drug_positions[start:stop])
            predictions[start:stop] = model(
                expression_tensor[cell_index], drug_tensor[drug_index]
            ).reshape(-1).cpu().numpy()

    expected = pd.DataFrame({
        "cell_line_id": np.repeat(np.asarray(cells, dtype=object), len(drugs)),
        "drug_id": np.tile(np.asarray(drugs, dtype=object), len(cells)),
        "predicted_source_pharmaformer": predictions,
        "predicted_source_train_mean": np.tile(np.asarray([means[drug] for drug in drugs]), len(cells)),
    })
    actual = pd.read_parquet(prediction_path)
    comparison = compare_keyed_predictions(actual, expected)
    if comparison["n_prediction_keys"] != EXPECTED_CELLS * EXPECTED_DRUGS:
        raise ValueError("frozen prediction file does not contain the complete 352 by 13 key grid")

    hashes = {name: sha256(path) for name, path in paths.items()}
    receipt = {
        "experiment_id": "EXP-044",
        "run_id": prediction_dir.name,
        "status": "PASS_INDEPENDENT_NUMERICAL_RECOMPUTATION_NO_TARGET_LABELS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "separate_script_reloads_official_model_and_recomputes_source_train_means",
        "target_response_values_read": False,
        "target_response_files_opened": [],
        "source_response_usage": "fold0_GDSC_training_pairs_only_for_source_mean_baseline",
        "n_source_training_identities": len(train_ids),
        "n_source_training_pairs_for_panel": n_source_pairs,
        "n_cells": len(cells),
        "n_drugs": len(drugs),
        "comparison": comparison,
        "inputs_sha256": hashes,
        "comparison_tolerances_frozen_before_target_response_evaluation": True,
    }
    output_receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_receipt.with_suffix(output_receipt.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(output_receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--input-verification", required=True, type=Path)
    parser.add_argument("--identity-key-audit", required=True, type=Path)
    parser.add_argument("--source-pairs", required=True, type=Path)
    parser.add_argument("--source-indices", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--official-source", required=True, type=Path)
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--output-receipt", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    receipt = run(
        input_dir=args.input_dir,
        input_verification_path=args.input_verification,
        identity_key_audit_path=args.identity_key_audit,
        source_pairs_path=args.source_pairs,
        source_indices_path=args.source_indices,
        checkpoint_path=args.checkpoint,
        official_source_path=args.official_source,
        prediction_dir=args.prediction_dir,
        output_receipt=args.output_receipt,
        batch_size=args.batch_size,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
