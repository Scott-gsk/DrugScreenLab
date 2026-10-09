"""Freeze zero-target-label PharmaFormer predictions and matched source means."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_official(source: Path):
    spec = importlib.util.spec_from_file_location("pharmaformer_official_pretrain", source)
    if spec is None or spec.loader is None:
        raise ValueError("official PharmaFormer source could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_revision(root: Path) -> tuple[str | None, str | None]:
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                  check=True, capture_output=True, text=True).stdout.strip()
        diff = subprocess.run(["git", "diff", "--binary"], cwd=root,
                              check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return revision, hashlib.sha256(diff).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def run(input_dir: Path, source_pairs_path: Path, source_indices_path: Path,
        checkpoint_path: Path, official_source: Path, output_dir: Path,
        *, device_name: str = "cpu", batch_size: int = 32) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"run directory already exists: {output_dir}")
    if device_name != "cpu":
        raise ValueError("EXP-043 prediction is CPU-only while the approved GPU queue is occupied")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    root = Path(__file__).resolve().parents[2]
    tracked_code = Path(__file__).resolve()
    input_paths = {
        "input_manifest": input_dir / "manifest.json",
        "patient_axis": input_dir / "patient_axis.json",
        "patient_expression": input_dir / "patient_expression.npy",
        "drug_axis": input_dir / "drug_axis.parquet",
        "drug_vectors": input_dir / "drug_ordinal128.npy",
        "source_pairs": source_pairs_path,
        "source_indices": source_indices_path,
        "source_checkpoint": checkpoint_path,
        "official_source": official_source,
        "input_verification": input_dir / "verification.json",
        "asset_audit": root / "artifacts" / "experiments" / "EXP-036" / "20260923_pharmaformer_asset_audit" / "asset_audit.json",
        "strict_reload_smoke": root / "artifacts" / "experiments" / "EXP-036" / "20260923_pharmaformer_asset_audit" / "strict_reload_smoke.json",
        "real_input_smoke": root / "artifacts" / "experiments" / "EXP-036" / "20260923_pharmaformer_asset_audit" / "real_input_smoke.json",
    }
    inputs_hashes = {name: sha256(path) for name, path in input_paths.items()}
    revision, diff_hash = _git_revision(root)
    provenance = {
        "experiment_id": "EXP-043", "run_id": output_dir.name,
        "started_utc": started_utc, "status": "PREDICTION_IN_PROGRESS_NO_TARGET_LABELS_LOADED",
        "code_sha256": sha256(tracked_code), "inputs_sha256": inputs_hashes,
        "git_head": revision, "tracked_worktree_diff_sha256": diff_hash,
        "command": os.sys.argv,
        "configuration": {"device": "cpu", "batch_size": batch_size,
                          "target_labels_loaded_by_predictor": False,
                          "checkpoint_fold": 0, "target_label_fraction": 0.0},
    }
    _write_json(output_dir / "run_provenance.json", provenance)
    try:
        input_manifest = json.loads(input_paths["input_manifest"].read_text(encoding="utf-8"))
        if input_manifest.get("status") != "PHARMAFORMER_TARGET_INPUTS_MATERIALIZED_PENDING_INDEPENDENT_VERIFY":
            raise ValueError("target model-input contract has an unexpected status")
        input_verification = json.loads(input_paths["input_verification"].read_text(encoding="utf-8"))
        if (input_verification.get("status") != "PASS"
                or input_verification.get("input_manifest_sha256") != inputs_hashes["input_manifest"]
                or input_verification.get("source_checkpoint_sha256") != inputs_hashes["source_checkpoint"]):
            raise ValueError("frozen model input has no matching independent verification receipt")
        asset_audit = json.loads(input_paths["asset_audit"].read_text(encoding="utf-8"))
        fold0_audit = next(row for row in asset_audit["checkpoints"] if int(row["fold"]) == 0)
        if (fold0_audit.get("checkpoint_sha256") != inputs_hashes["source_checkpoint"]
                or int(fold0_audit.get("n_source_fit_cells", 0)) != 577
                or int(fold0_audit.get("n_target_drugs_covered", 0)) != 14):
            raise ValueError("source checkpoint identity audit differs from the frozen model assets")
        strict_smoke = json.loads(input_paths["strict_reload_smoke"].read_text(encoding="utf-8"))
        real_smoke = json.loads(input_paths["real_input_smoke"].read_text(encoding="utf-8"))
        if (strict_smoke.get("status") != "STRICT_OFFICIAL_RELOAD_PASS_ENGINEERING_ONLY"
                or strict_smoke.get("checkpoint_sha256") != inputs_hashes["source_checkpoint"]
                or strict_smoke.get("official_source_sha256") != inputs_hashes["official_source"]
                or real_smoke.get("status") != "STRICT_RELOAD_AND_REAL_INPUT_SMOKE_PASS_ENGINEERING_ONLY"
                or real_smoke.get("checkpoint_sha256") != inputs_hashes["source_checkpoint"]
                or real_smoke.get("official_source_sha256") != inputs_hashes["official_source"]):
            raise ValueError("official model reload or real-input engineering gate failed")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("variant") != "M2" or checkpoint.get("model_variant") != "CombinedModel":
            raise ValueError("source checkpoint is not the audited official M2 model")
        if checkpoint.get("upstream_commit") != "f526ddda4091151f082b38367571a94823314aba":
            raise ValueError("source checkpoint official-code revision differs from the frozen audit")
        if input_manifest.get("source_checkpoint_sha256") != inputs_hashes["source_checkpoint"]:
            raise ValueError("target input preprocessing was built from a different source checkpoint")

        patient_axis = list(map(str, json.loads(input_paths["patient_axis"].read_text(encoding="utf-8"))))
        drug_axis = pd.read_parquet(input_paths["drug_axis"])
        patients = np.load(input_paths["patient_expression"], mmap_mode="r")
        drug_vectors = np.load(input_paths["drug_vectors"], mmap_mode="r")
        if len(set(patient_axis)) != len(patient_axis) or len(patient_axis) != 51:
            raise ValueError("patient axis must contain 51 unique identities")
        if drug_axis["drug_id"].astype(str).duplicated().any() or len(drug_axis) != 14:
            raise ValueError("drug axis must contain 14 unique drugs")
        drug_ids = drug_axis["drug_id"].astype(str).tolist()
        if patients.shape != (len(patient_axis), int(checkpoint["model_config"]["gene_input_size"])):
            raise ValueError("patient expression does not match official model input size")
        if drug_vectors.shape != (len(drug_ids), 128):
            raise ValueError("drug vectors do not match the frozen 128-wide drug axis")
        if not np.isfinite(np.asarray(patients)).all() or not np.isfinite(np.asarray(drug_vectors)).all():
            raise ValueError("frozen target model inputs contain non-finite values")
        if sha256(input_paths["patient_expression"]) != input_manifest["files_sha256"]["patient_expression.npy"]:
            raise ValueError("patient expression checksum differs from its frozen input receipt")
        if sha256(input_paths["drug_vectors"]) != input_manifest["files_sha256"]["drug_ordinal128.npy"]:
            raise ValueError("drug vector checksum differs from its frozen input receipt")

        source_indices = json.loads(source_indices_path.read_text(encoding="utf-8"))
        if int(source_indices.get("fold_id", -1)) != 0 or source_indices.get("stage") != "source":
            raise ValueError("source checkpoint fold assignment is not fold0 source training")
        source_train_ids = {str(identity) for identity, role in source_indices["group_assignments"].items()
                            if role == "train"}
        checkpoint_fit_ids = set(map(str, checkpoint["gene_transform"]["fit_sample_ids"]))
        if checkpoint_fit_ids != source_train_ids:
            raise ValueError("checkpoint preprocessing identities differ from source fold0 training identities")
        source_pairs = pd.read_parquet(source_pairs_path)
        source_pairs["sample_id"] = source_pairs["sample_id"].astype(str)
        source_pairs["drug_id"] = source_pairs["drug_id"].astype(str)
        source_train = source_pairs.loc[source_pairs["sample_id"].isin(source_train_ids)]
        if source_train.empty or source_train.duplicated(["sample_id", "drug_id"]).any():
            raise ValueError("source training response keys are empty or duplicated")
        if set(source_train["sample_id"]) - source_train_ids:
            raise ValueError("source mean includes a non-training biological identity")
        source_mean = source_train.groupby("drug_id", sort=True)["response"].mean()
        if set(source_mean.index.astype(str)) != set(drug_ids):
            raise ValueError("matched source mean does not cover the complete 14-drug PDO panel")

        official = load_official(official_source)
        config = checkpoint["model_config"]
        model = official.CombinedModel(
            gene_input_size=config["gene_input_size"], gene_hidden_size=config["gene_hidden_size"],
            drug_hidden_size=config["drug_hidden_size"], feature_dim=config["feature_dim"],
            nhead=config["nhead"], num_layers=config["num_layers"],
            dim_feedforward=config["dim_feedforward"], dropout=config["dropout"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        model.to(torch.device("cpu"))
        expression_tensor = torch.from_numpy(np.array(patients, dtype=np.float32, copy=True))
        drug_tensor = torch.from_numpy(np.array(drug_vectors, dtype=np.float32, copy=True))
        grid_patient = np.repeat(np.arange(len(patient_axis), dtype=np.int64), len(drug_ids))
        grid_drug = np.tile(np.arange(len(drug_ids), dtype=np.int64), len(patient_axis))
        predicted = np.empty(len(grid_patient), dtype=np.float32)
        with torch.inference_mode():
            for begin in range(0, len(grid_patient), batch_size):
                end = min(begin + batch_size, len(grid_patient))
                expression_batch = expression_tensor[torch.from_numpy(grid_patient[begin:end])]
                drug_batch = drug_tensor[torch.from_numpy(grid_drug[begin:end])]
                predicted[begin:end] = model(expression_batch, drug_batch).flatten().cpu().numpy()
        if not np.isfinite(predicted).all():
            raise ValueError("official source model produced non-finite PDO predictions")

        frame = pd.DataFrame({
            "patient_id": [patient_axis[index] for index in grid_patient],
            "expression_donor_patient_id": [patient_axis[index] for index in grid_patient],
            "drug_id": [drug_ids[index] for index in grid_drug],
            "predicted_source_pharmaformer": predicted,
            "predicted_source_train_mean": [float(source_mean[drug_ids[index]]) for index in grid_drug],
        })
        if len(frame) != 51 * 14 or frame.duplicated(["patient_id", "drug_id"]).any():
            raise ValueError("prediction grid is incomplete or has duplicate patient/drug keys")
        prediction_path = output_dir / "predictions.parquet"
        frame.to_parquet(prediction_path, index=False)
        summary = {
            "experiment_id": "EXP-043", "run_id": output_dir.name,
            "status": "PREDICTIONS_FROZEN_PENDING_LABEL_EVALUATION",
            "n_patients": len(patient_axis), "n_drugs": len(drug_ids),
            "n_prediction_keys": len(frame), "n_source_training_cells": len(source_train_ids),
            "n_source_training_response_pairs": len(source_train),
            "checkpoint_sha256": inputs_hashes["source_checkpoint"],
            "official_source_sha256": inputs_hashes["official_source"],
            "inputs_sha256": inputs_hashes,
            "prediction_file_sha256": sha256(prediction_path),
            "prediction_columns": list(frame.columns),
            "target_response_loaded": False,
            "resources": {"device": "cpu", "elapsed_seconds": time.perf_counter() - started,
                          "python": platform.python_version(), "torch": torch.__version__,
                          "numpy": np.__version__, "pandas": pd.__version__},
        }
        _write_json(output_dir / "prediction_summary.json", summary)
        provenance["status"] = "PREDICTIONS_FROZEN_NO_TARGET_LABELS_LOADED"
        provenance["completed_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(output_dir / "run_provenance.json", provenance)
        return summary
    except Exception as error:
        _write_json(output_dir / "FAILURE.json", {
            "experiment_id": "EXP-043", "run_id": output_dir.name,
            "error_type": type(error).__name__, "error": str(error),
            "elapsed_seconds": time.perf_counter() - started,
            "target_response_loaded": False,
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--source-pairs", type=Path, required=True)
    parser.add_argument("--source-indices", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        result = run(args.input_dir, args.source_pairs, args.source_indices,
                     args.checkpoint, args.official_source, args.output_dir,
                     device_name=args.device, batch_size=args.batch_size)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
