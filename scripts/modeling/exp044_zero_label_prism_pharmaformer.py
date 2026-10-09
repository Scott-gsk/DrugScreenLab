"""Freeze label-free PharmaFormer predictions for the EXP-044 PRISM cohort."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits


EXPECTED_CHECKPOINT_SHA256 = "1dbe98a2f52c4affdcee091cf011f3e52e29b51992c9bb3d57a60b08fa464de6"
EXPECTED_OFFICIAL_COMMIT = "f526ddda4091151f082b38367571a94823314aba"
EXPECTED_STRICT_UNSEEN_CELL_COUNT = 352
EXPECTED_PRISM_PANEL_DRUG_COUNT = 13


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def load_official(source: Path):
    spec = importlib.util.spec_from_file_location("pharmaformer_official_exp044", source)
    if spec is None or spec.loader is None:
        raise ValueError("official PharmaFormer source could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_provenance(root: Path) -> tuple[str | None, str | None]:
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                  capture_output=True, text=True).stdout.strip()
        patch = subprocess.run(["git", "diff", "--binary"], cwd=root, check=True,
                               capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return revision, hashlib.sha256(patch).hexdigest()


def source_means(source_pairs: pd.DataFrame, train_ids: set[str], drug_ids: list[str]) -> tuple[pd.Series, int]:
    required = {"sample_id", "drug_id", "response"}
    if required - set(source_pairs.columns):
        raise ValueError("source training pair table lacks sample_id, drug_id, or response")
    frame = source_pairs.loc[:, ["sample_id", "drug_id", "response"]].copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["drug_id"] = frame["drug_id"].astype(str)
    if frame.duplicated(["sample_id", "drug_id"]).any():
        raise ValueError("source response table contains duplicate sample/drug keys")
    frame = frame.loc[frame["sample_id"].isin(train_ids) & frame["drug_id"].isin(set(drug_ids))]
    if frame.empty or not np.isfinite(frame["response"].to_numpy(dtype=np.float64)).all():
        raise ValueError("source fold training responses are empty or non-finite")
    means = frame.groupby("drug_id", sort=True)["response"].mean()
    if set(means.index.astype(str)) != set(drug_ids):
        raise ValueError("fold-0 source training identities do not cover the 13-drug PRISM panel")
    return means, len(frame)


def build_prediction_frame(cell_ids: list[str], drug_ids: list[str], predictions: np.ndarray,
                           means: pd.Series) -> pd.DataFrame:
    if not cell_ids or len(cell_ids) != len(set(cell_ids)):
        raise ValueError("cell axis must contain unique identities")
    if not drug_ids or len(drug_ids) != len(set(drug_ids)):
        raise ValueError("drug axis must contain unique identities")
    expected = len(cell_ids) * len(drug_ids)
    values = np.asarray(predictions, dtype=np.float32).reshape(-1)
    if values.shape != (expected,) or not np.isfinite(values).all():
        raise ValueError("prediction vector does not cover the finite cell/drug grid")
    if set(means.index.astype(str)) != set(drug_ids):
        raise ValueError("source mean baseline does not match the exact PRISM drug panel")
    patient_position = np.repeat(np.arange(len(cell_ids), dtype=np.int64), len(drug_ids))
    drug_position = np.tile(np.arange(len(drug_ids), dtype=np.int64), len(cell_ids))
    ordered_drugs = [str(value) for value in drug_ids]
    ordered_cells = [str(value) for value in cell_ids]
    frame = pd.DataFrame({
        "cell_line_id": [ordered_cells[index] for index in patient_position],
        "drug_id": [ordered_drugs[index] for index in drug_position],
        "predicted_source_pharmaformer": values,
        "predicted_source_train_mean": [float(means[ordered_drugs[index]]) for index in drug_position],
    })
    if len(frame) != expected or frame.duplicated(["cell_line_id", "drug_id"]).any():
        raise ValueError("prediction grid contains missing or duplicate cell/drug keys")
    return frame


def validate_frozen_prediction_axes(cell_ids: list[str], drug_ids: list[str]) -> None:
    if (len(cell_ids) != EXPECTED_STRICT_UNSEEN_CELL_COUNT
            or len(drug_ids) != EXPECTED_PRISM_PANEL_DRUG_COUNT
            or len(set(cell_ids)) != EXPECTED_STRICT_UNSEEN_CELL_COUNT
            or len(set(drug_ids)) != EXPECTED_PRISM_PANEL_DRUG_COUNT):
        raise ValueError(
            "frozen strict-unseen cohort must contain "
            f"{EXPECTED_STRICT_UNSEEN_CELL_COUNT} unique cells and "
            f"{EXPECTED_PRISM_PANEL_DRUG_COUNT} unique drugs"
        )


def run(input_dir: Path, input_verification_path: Path, source_pairs_path: Path,
        source_indices_path: Path, checkpoint_path: Path, official_source_path: Path,
        asset_audit_path: Path, strict_reload_smoke_path: Path, real_input_smoke_path: Path,
        identity_key_audit_path: Path, output_dir: Path, *, batch_size: int = 32) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"EXP-044 prediction directory already exists: {output_dir}")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if sha256(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("EXP-044 requires the preregistered fold-0 source checkpoint")

    root = Path(__file__).resolve().parents[2]
    script_path = Path(__file__).resolve()
    input_paths = {
        "input_manifest": input_dir / "manifest.json",
        "identity_resolution": input_dir / "identity_resolution.json",
        "strict_unseen_ids_v2": input_dir / "strict_unseen_cell_lines_v2.csv",
        "source_overlap_ids_v2": input_dir / "excluded_source_overlap_cell_lines_v2.csv",
        "input_verification": input_verification_path,
        "identity_key_coverage_audit": identity_key_audit_path,
        "cell_axis": input_dir / "cell_axis.json",
        "cell_expression": input_dir / "cell_expression.npy",
        "drug_axis": input_dir / "drug_axis.parquet",
        "drug_vectors": input_dir / "drug_ordinal128.npy",
        "source_pairs": source_pairs_path,
        "source_indices": source_indices_path,
        "source_checkpoint": checkpoint_path,
        "official_source": official_source_path,
        "asset_audit": asset_audit_path,
        "strict_reload_smoke": strict_reload_smoke_path,
        "real_input_smoke": real_input_smoke_path,
    }
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    hashes = {name: sha256(path) for name, path in input_paths.items()}
    revision, diff_hash = git_provenance(root)
    provenance = {
        "experiment_id": "EXP-044", "run_id": output_dir.name,
        "started_utc": started_utc, "status": "PREDICTION_IN_PROGRESS_NO_PRISM_LABELS_LOADED",
        "code_sha256": sha256(script_path), "inputs_sha256": hashes,
        "git_head": revision, "tracked_worktree_diff_sha256": diff_hash,
        "command": os.sys.argv,
        "configuration": {
            "device": "cpu", "batch_size": batch_size, "checkpoint_fold": 0,
            "target_labels_loaded_by_predictor": False,
            "source_training_response_file_opened": str(source_pairs_path),
            "target_response_files_opened": [], "target_response_values_read": False,
        },
    }
    write_json(output_dir / "run_provenance.json", provenance)
    try:
        torch.set_num_threads(1)
        verification = json.loads(input_verification_path.read_text(encoding="utf-8"))
        if (verification.get("status") != "PASS_RESPONSE_BLIND_IDENTITY_RESOLVED"
                or verification.get("identity_resolution_status") != "RESOLVED_ORTHOGONAL_AUTHORITATIVE_CROSSWALK"
                or verification.get("scientific_strict_unseen_accepted") is not True
                or verification.get("target_response_values_read") is not False):
            raise ValueError("prediction requires independently verified response-blind resolved inputs")
        manifest = json.loads(input_paths["input_manifest"].read_text(encoding="utf-8"))
        if (manifest.get("experiment_id") != "EXP-044"
                or manifest.get("target_response_values_read") is not False
                or manifest.get("response_label_files_opened") != []
                or manifest.get("source_checkpoint_sha256") != hashes["source_checkpoint"]
                or manifest.get("identity_resolution_status") != "RESOLVED_ORTHOGONAL_AUTHORITATIVE_CROSSWALK"
                or manifest.get("scientific_strict_unseen_accepted") is not True
                or manifest.get("n_strict_unseen_prism_cells") != EXPECTED_STRICT_UNSEEN_CELL_COUNT):
            raise ValueError("source-input manifest differs from the frozen response-blind experiment inputs")
        identity_resolution = json.loads(input_paths["identity_resolution"].read_text(encoding="utf-8"))
        if (identity_resolution.get("status") != "RESOLVED_ORTHOGONAL_AUTHORITATIVE_CROSSWALK"
                or identity_resolution.get("strict_unseen_accepted") is not True
                or identity_resolution.get("target_response_values_read") is not False
                or identity_resolution.get("response_label_files_opened") != []
                or identity_resolution.get("strict_unseen_ids_sha256")
                != manifest.get("strict_unseen_ids_sha256")):
            raise ValueError("identity-resolution receipt does not match the accepted no-label cohort")
        key_audit = json.loads(identity_key_audit_path.read_text(encoding="utf-8"))
        if (key_audit.get("status") != "PASS_RESPONSE_BLIND_IDENTITY_KEY_COVERAGE"
                or key_audit.get("identity_resolution_sha256") != hashes["identity_resolution"]
                or key_audit.get("cell_axis_sha256") != hashes["cell_axis"]
                or key_audit.get("drug_axis_sha256") != hashes["drug_axis"]
                or key_audit.get("target_response_values_read") is not False
                or key_audit.get("response_value_columns_requested") != []
                or key_audit.get("n_revised_strict_unseen_cells") != EXPECTED_STRICT_UNSEEN_CELL_COUNT
                or key_audit.get("n_prism_panel_drugs") != EXPECTED_PRISM_PANEL_DRUG_COUNT):
            raise ValueError("response-blind drug-key eligibility audit is missing or bound to different inputs")
        strict_ids = pd.read_csv(input_paths["strict_unseen_ids_v2"], usecols=["cell_line_id"])
        overlap_ids = pd.read_csv(input_paths["source_overlap_ids_v2"], usecols=["cell_line_id"])
        if (set(strict_ids["cell_line_id"].astype(str)) != set(map(str, json.loads(
                input_paths["cell_axis"].read_text(encoding="utf-8"))))
                or set(strict_ids["cell_line_id"].astype(str))
                & set(overlap_ids["cell_line_id"].astype(str))):
            raise ValueError("saved revised identity lists do not match the prediction cell axis")
        identity_audit = manifest.get("identity_bridge_audit", {})
        if (identity_audit.get("unmapped_source_sanger_ids") != []
                or identity_audit.get("revised_partition_from_frozen_exp019") != {
                    "strict_before": 353, "overlap_before": 524,
                    "strict_after": EXPECTED_STRICT_UNSEEN_CELL_COUNT, "overlap_after": 525,
                    "moved_from_strict_to_overlap": ["ACH-000499"],
                }):
            raise ValueError("resolved input manifest lacks the approved one-cell identity correction")
        if (hashes["input_manifest"] != verification.get("input_manifest_sha256")
                or hashes["source_checkpoint"] != verification.get("source_checkpoint_sha256")):
            raise ValueError("independent input verification does not bind the current manifest and checkpoint")

        audit = json.loads(asset_audit_path.read_text(encoding="utf-8"))
        fold0 = next(row for row in audit["checkpoints"] if int(row["fold"]) == 0)
        if (fold0.get("checkpoint_sha256") != hashes["source_checkpoint"]
                or int(fold0.get("n_source_fit_cells", 0)) != 577
                or int(fold0.get("n_target_drugs_covered", 0)) != 14):
            raise ValueError("source checkpoint does not match the prior identity and coverage audit")
        for path, name in ((strict_reload_smoke_path, "strict reload"),
                           (real_input_smoke_path, "real-input smoke")):
            smoke = json.loads(path.read_text(encoding="utf-8"))
            expected_status = ("STRICT_OFFICIAL_RELOAD_PASS_ENGINEERING_ONLY"
                               if name == "strict reload"
                               else "STRICT_RELOAD_AND_REAL_INPUT_SMOKE_PASS_ENGINEERING_ONLY")
            if (smoke.get("checkpoint_sha256") != hashes["source_checkpoint"]
                    or smoke.get("official_source_sha256") != hashes["official_source"]
                    or smoke.get("status") != expected_status):
                raise ValueError(f"{name} receipt is for different source assets")
        if sha256(official_source_path) != "c7c1dc2389e4050df1f007093a6042620be1c0abdeb4792f04d583fe2820f567":
            raise ValueError("official PharmaFormer source code differs from the audited frozen revision")

        cell_ids = list(map(str, json.loads(input_paths["cell_axis"].read_text(encoding="utf-8"))))
        drug_axis = pd.read_parquet(input_paths["drug_axis"])
        drug_ids = drug_axis["drug_id"].astype(str).tolist()
        expression = np.load(input_paths["cell_expression"], mmap_mode="r", allow_pickle=False)
        drug_vectors = np.load(input_paths["drug_vectors"], mmap_mode="r", allow_pickle=False)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (checkpoint.get("variant") != "M2" or checkpoint.get("model_variant") != "CombinedModel"
                or checkpoint.get("upstream_commit") != EXPECTED_OFFICIAL_COMMIT):
            raise ValueError("checkpoint is not the preregistered official M2 fold-0 model")
        validate_frozen_prediction_axes(cell_ids, drug_ids)
        gene_count = int(checkpoint["model_config"]["gene_input_size"])
        if (expression.shape != (len(cell_ids), gene_count)
                or drug_vectors.shape != (len(drug_ids), 128)
                or not np.isfinite(np.asarray(expression)).all()
                or not np.isfinite(np.asarray(drug_vectors)).all()):
            raise ValueError("materialized expression or drug encoding does not match the source model")
        if sha256(input_paths["cell_expression"]) != manifest["files_sha256"]["cell_expression.npy"]:
            raise ValueError("materialized expression checksum differs from its input manifest")
        if sha256(input_paths["drug_vectors"]) != manifest["files_sha256"]["drug_ordinal128.npy"]:
            raise ValueError("materialized drug vector checksum differs from its input manifest")

        source_indices = json.loads(source_indices_path.read_text(encoding="utf-8"))
        if source_indices.get("stage") != "source" or int(source_indices.get("fold_id", -1)) != 0:
            raise ValueError("source checkpoint identity receipt is not fold 0")
        train_ids = {str(identity) for identity, role in source_indices["group_assignments"].items()
                     if role == "train"}
        fitted_ids = set(map(str, checkpoint["gene_transform"]["fit_sample_ids"]))
        if len(train_ids) != 577 or train_ids != fitted_ids:
            raise ValueError("source checkpoint fit identities differ from the frozen 577 fold-0 source IDs")
        source_pairs = pd.read_parquet(source_pairs_path)
        means, n_source_pairs = source_means(source_pairs, train_ids, drug_ids)

        official = load_official(official_source_path)
        config = checkpoint["model_config"]
        model = official.CombinedModel(
            gene_input_size=config["gene_input_size"], gene_hidden_size=config["gene_hidden_size"],
            drug_hidden_size=config["drug_hidden_size"], feature_dim=config["feature_dim"],
            nhead=config["nhead"], num_layers=config["num_layers"],
            dim_feedforward=config["dim_feedforward"], dropout=config["dropout"],
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval().to(torch.device("cpu"))
        expression_tensor = torch.from_numpy(np.array(expression, dtype=np.float32, copy=True))
        drug_tensor = torch.from_numpy(np.array(drug_vectors, dtype=np.float32, copy=True))
        row_patient = np.repeat(np.arange(len(cell_ids), dtype=np.int64), len(drug_ids))
        row_drug = np.tile(np.arange(len(drug_ids), dtype=np.int64), len(cell_ids))
        predicted = np.empty(len(row_patient), dtype=np.float32)
        with threadpool_limits(limits=1), torch.inference_mode():
            for begin in range(0, len(row_patient), batch_size):
                end = min(begin + batch_size, len(row_patient))
                p = torch.from_numpy(row_patient[begin:end])
                d = torch.from_numpy(row_drug[begin:end])
                predicted[begin:end] = model(expression_tensor[p], drug_tensor[d]).flatten().cpu().numpy()
        frame = build_prediction_frame(cell_ids, drug_ids, predicted, means)
        prediction_path = output_dir / "predictions.parquet"
        frame.to_parquet(prediction_path, index=False)
        summary = {
            "experiment_id": "EXP-044", "run_id": output_dir.name,
            "status": "PREDICTIONS_FROZEN_NO_PRISM_LABELS_LOADED_PENDING_RESPONSE_EVALUATION",
            "identity_scope": (
                "response_blind_Cellosaurus_crosswalk_moves_ACH-000499_to_source-overlap; "
                f"revised_strict_cohort_{EXPECTED_STRICT_UNSEEN_CELL_COUNT}"
            ),
            "identity_revision": "cellosaurus_crosswalk_v1_applied_to_exp019_frozen_v2_partition",
            "scientific_strict_unseen_accepted": True,
            "n_strict_unseen_cohort_cells": len(cell_ids), "n_prism_panel_drugs": len(drug_ids),
            "n_prediction_keys": len(frame), "n_source_training_identities": len(train_ids),
            "n_source_training_pairs_for_panel": n_source_pairs,
            "identity_key_audit_sha256": hashes["identity_key_coverage_audit"],
            "n_cells_with_at_least_three_panel_drugs": key_audit[
                "n_cells_with_at_least_three_panel_drugs"],
            "unmapped_gdsc_source_ids": [],
            "ambiguous_gdsc_source_bridge": {
                "SIDM00117": ["ACH-000837", "ACH-002172"],
                "prism_member": "ACH-000837",
                "handling": "conservative_union_of_all_candidate_ModelIDs",
            },
            "source_checkpoint_sha256": hashes["source_checkpoint"],
            "official_source_sha256": hashes["official_source"],
            "inputs_sha256": hashes,
            "prediction_file_sha256": sha256(prediction_path),
            "prediction_columns": list(frame.columns),
            "target_response_values_read": False, "target_response_files_opened": [],
            "resources": {"device": "cpu", "elapsed_seconds": time.perf_counter() - started,
                          "python": platform.python_version(), "torch": torch.__version__,
                          "numpy": np.__version__, "pandas": pd.__version__},
        }
        write_json(output_dir / "prediction_summary.json", summary)
        provenance["status"] = "PREDICTIONS_FROZEN_NO_PRISM_LABELS_LOADED"
        provenance["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output_dir / "run_provenance.json", provenance)
        return summary
    except Exception as error:
        write_json(output_dir / "FAILURE.json", {
            "experiment_id": "EXP-044", "status": "FAIL",
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(), "target_response_values_read": False,
            "target_response_files_opened": [],
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--input-verification", type=Path, required=True)
    parser.add_argument("--source-pairs", type=Path, required=True)
    parser.add_argument("--source-indices", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--asset-audit", type=Path, required=True)
    parser.add_argument("--strict-reload-smoke", type=Path, required=True)
    parser.add_argument("--real-input-smoke", type=Path, required=True)
    parser.add_argument("--identity-key-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    result = run(
        args.input_dir, args.input_verification, args.source_pairs, args.source_indices,
        args.checkpoint, args.official_source, args.asset_audit, args.strict_reload_smoke,
        args.real_input_smoke, args.identity_key_audit, args.output_dir,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
