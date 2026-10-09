"""Measure frozen-model prediction sensitivity to the fixed 64-gene input mask.

This is a source-only engineering diagnostic. It reads inner-training control
profiles and inner-validation control inputs/predictions, but never reads a
response label, outer-test value, or organoid treated expression.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
EXP037 = ROOT / "artifacts/experiments/EXP-037/20260928_lincs_scale_16_121"
PREPARED = EXP037 / "prepared"
TRANSIGEN = EXP037 / "transigen_121"
XPERT = EXP037 / "xpert_121"
CONTINUOUS = ROOT / "data/processed/lincs/GSE92742/continuous_contract_v2"
V7 = ROOT / "artifacts/experiments/EXP-050/20260930_keyed_response_v7"
RUN_ID = "20260930_missing_gene_input_sensitivity_v1"
RESERVED_GPU_HOURS = 0.25
ATOL = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def import_helper(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import audited helper at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summarize_change(
    baseline: np.ndarray, masked: np.ndarray, visible_indices: np.ndarray
) -> dict[str, Any]:
    before = np.asarray(baseline[:, visible_indices], dtype=np.float64)
    after = np.asarray(masked[:, visible_indices], dtype=np.float64)
    if before.shape != after.shape or not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("visible-gene prediction arrays have invalid shape or values")

    absolute_change = np.abs(after - before)
    before_centered = before - before.mean(axis=1, keepdims=True)
    after_centered = after - after.mean(axis=1, keepdims=True)
    numerator = np.sum(before_centered * after_centered, axis=1)
    denominator = np.sqrt(
        np.sum(before_centered ** 2, axis=1) * np.sum(after_centered ** 2, axis=1)
    )
    valid = denominator > 0
    row_correlations = numerator[valid] / denominator[valid]
    both_nonzero = (before != 0.0) & (after != 0.0)
    sign_pairs = int(both_nonzero.sum())
    sign_agree = int(((np.sign(before) == np.sign(after)) & both_nonzero).sum())
    return {
        "visible_gene_count": int(before.shape[1]),
        "validation_condition_count": int(before.shape[0]),
        "prediction_units": "model-native output units; no cross-model magnitude comparison",
        "mean_absolute_prediction_change": float(absolute_change.mean()),
        "median_absolute_prediction_change": float(np.median(absolute_change)),
        "p95_absolute_prediction_change": float(np.quantile(absolute_change, 0.95)),
        "max_absolute_prediction_change": float(absolute_change.max()),
        "per_condition_pearson_visible914": {
            "valid_condition_count": int(valid.sum()),
            "median": float(np.median(row_correlations)) if len(row_correlations) else None,
            "p10": float(np.quantile(row_correlations, 0.10)) if len(row_correlations) else None,
        },
        "visible_gene_sign_agreement": {
            "nonzero_prediction_pairs": sign_pairs,
            "agreement_fraction": float(sign_agree / sign_pairs) if sign_pairs else None,
        },
    }


def run(output: Path, run_id: str, device_index: int) -> dict[str, Any]:
    if sys.platform != "linux" or os.environ.get("CONDA_DEFAULT_ENV") != "drugscreening-gpu":
        raise RuntimeError("EXP-050 requires WSL2 Conda environment drugscreening-gpu")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing diagnostic output: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible for the frozen source-model diagnostic")

    output.mkdir(parents=True)
    ledger = ROOT / "artifacts/experiments/EXP-050/gpu_ledger.jsonl"
    torch.set_num_threads(4)
    device = torch.device(f"cuda:{device_index}")
    device_name = torch.cuda.get_device_name(device)
    attempt_id = f"EXP050-{run_id.upper()}"
    started = time.monotonic()
    append_jsonl(ledger, {
        "experiment_id": "EXP-050",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "event": "attempt_started",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reserved_gpu_hours": RESERVED_GPU_HOURS,
        "device": str(device),
        "device_name": device_name,
        "operation": "source-only 64-gene mask prediction sensitivity; no training or response labels",
    })

    try:
        transigen_helper = import_helper(
            ROOT / "scripts/data/audit_exp050_frozen_transigen_validation_replay.py",
            "exp050_missing_gene_transigen_replay",
        )
        xpert_helper = import_helper(
            ROOT / "scripts/data/audit_exp050_frozen_xpert_validation_replay.py",
            "exp050_missing_gene_xpert_replay",
        )
        config_path = ROOT / "configs/experiments/EXP-050.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        axis_path = V7 / "EXP050_common_gene_axis_914.tsv"
        axis = pd.read_csv(axis_path, sep="\t", dtype={"LINCS_Entrez_id": str})
        genes_path = CONTINUOUS / "genes.json"
        model_gene_ids = [str(value) for value in json.loads(genes_path.read_text(encoding="utf-8"))]
        v7_manifest = json.loads((V7 / "run_manifest.json").read_text(encoding="utf-8"))
        axis_output_hash = v7_manifest["output_hashes"][axis_path.name]["sha256"]
        axis_contract = config["comparison"]["frozen_gene_axis"]
        if (len(model_gene_ids) != 978 or len(set(model_gene_ids)) != 978
                or len(axis) != 914 or axis["LINCS_Entrez_id"].nunique() != 914
                or axis_contract.get("count") != 914
                or axis_contract.get("hash") != sha256(axis_path)
                or axis_output_hash != sha256(axis_path)):
            raise ValueError("frozen common914 axis does not match the model exact978 contract")
        visible_ids = set(axis["LINCS_Entrez_id"].astype(str))
        visible_indices = np.asarray(
            [index for index, gene_id in enumerate(model_gene_ids) if gene_id in visible_ids],
            dtype=np.int64,
        )
        missing_indices = np.asarray(
            [index for index, gene_id in enumerate(model_gene_ids) if gene_id not in visible_ids],
            dtype=np.int64,
        )
        if len(visible_indices) != 914 or len(missing_indices) != 64:
            raise ValueError("frozen target/model gene overlap is not exactly 914/978")

        manifest_path = PREPARED / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("status") != "PREPARED_TRAIN_VALIDATION_ONLY"
                or manifest.get("n_train121") != 32094
                or manifest.get("n_validation") != 4954
                or manifest.get("official_test_responses_read") is not False):
            raise ValueError("source train/validation-only tier differs from the frozen protocol")
        prepared_paths = {
            "conditions.parquet": PREPARED / "conditions.parquet",
            "control.npy": PREPARED / "control.npy",
            "ecfp4.npy": PREPARED / "ecfp4.npy",
            "train121.npy": PREPARED / "train121.npy",
            "validation.npy": PREPARED / "validation.npy",
            "source_to_target.npy": PREPARED / "source_to_target.npy",
        }
        prepared_hashes = {name: sha256(path) for name, path in prepared_paths.items()}
        hash_cache = {prepared_paths[name]: digest for name, digest in prepared_hashes.items()}
        if any(prepared_hashes[name] != manifest["files_sha256"].get(name)
               for name in prepared_paths):
            raise ValueError("prepared source inputs differ from their frozen EXP-037 manifest")
        keys = pd.read_parquet(PREPARED / "conditions.parquet")
        train_positions = np.load(PREPARED / "train121.npy", allow_pickle=False)
        validation_positions = np.load(PREPARED / "validation.npy", allow_pickle=False)
        source_to_target = np.load(PREPARED / "source_to_target.npy", allow_pickle=False)
        control_path = PREPARED / "control.npy"
        ecfp_path = PREPARED / "ecfp4.npy"
        control = np.load(control_path, mmap_mode="r", allow_pickle=False)
        ecfp4 = np.load(ecfp_path, mmap_mode="r", allow_pickle=False)
        if (train_positions.ndim != 1 or validation_positions.ndim != 1
                or len(train_positions) != 32094 or len(validation_positions) != 4954
                or len(train_positions) + len(validation_positions) != len(keys)
                or train_positions.min() < 0 or validation_positions.min() < 0
                or train_positions.max() >= len(keys) or validation_positions.max() >= len(keys)
                or len(np.unique(train_positions)) != len(train_positions)
                or len(np.unique(validation_positions)) != len(validation_positions)
                or np.intersect1d(train_positions, validation_positions).size
                or control.shape != (len(keys), 978)
                or ecfp4.shape != (len(keys), 2048)
                or source_to_target.shape != (978,)
                or not np.array_equal(np.sort(source_to_target), np.arange(978))):
            raise ValueError("source condition or train/validation row axes are invalid")
        if (not np.allclose(pd.to_numeric(keys.pert_dose).to_numpy(np.float64), 10.0)
                or not np.allclose(pd.to_numeric(keys.pert_time).to_numpy(np.float64), 24.0)):
            raise ValueError("source inputs include conditions outside the fixed 10 uM x 24 h scope")

        train_keys = keys.iloc[train_positions]
        unique_cells = train_keys.cell_id.astype(str).drop_duplicates().tolist()
        per_cell_profiles = []
        train_cell_array = train_keys.cell_id.astype(str).to_numpy()
        for cell_id in unique_cells:
            local_positions = train_positions[train_cell_array == cell_id]
            cell_controls = np.asarray(control[local_positions], dtype=np.float32)
            if not np.isfinite(cell_controls).all():
                raise ValueError(f"nonfinite training control input for {cell_id}")
            per_cell_profiles.append(np.median(cell_controls, axis=0))
        train_median = np.median(np.stack(per_cell_profiles), axis=0).astype(np.float32)
        if len(unique_cells) < 2 or train_median.shape != (978,) or not np.isfinite(train_median).all():
            raise ValueError("training-cell-balanced imputation vector is invalid")

        validation_control = np.asarray(control[validation_positions], dtype=np.float32)
        validation_features = np.asarray(ecfp4[validation_positions], dtype=np.float32)
        if not np.isfinite(validation_control).all() or not np.isfinite(validation_features).all():
            raise ValueError("source validation control/drug inputs contain nonfinite values")
        masked_validation_control = validation_control.copy()
        masked_validation_control[:, missing_indices] = train_median[missing_indices]
        local_validation_indices = np.arange(len(validation_positions), dtype=np.int64)

        # Bind XPert's large external feature matrix and code snapshot before
        # either frozen model is run, so an input-path or provenance failure
        # cannot consume an inference attempt and leave only partial outputs.
        xpert_adapter, xpert_external_files = xpert_helper.load_snapshot_adapter()
        paths = xpert_adapter.ProjectPaths.from_root(ROOT)
        xpert_snapshot_path = xpert_helper.SNAPSHOT / "exp037_xpert_fold.py"
        xpert_source_table_path = (
            paths.xpert_dir / "processed_data/l1000_sdst_drugs_info_8432.csv"
        )
        unimol_path = paths.xpert_dir / "processed_data/all_drugs_unimol_arr.npy"
        graph_path = paths.xpert_dir / "HG_data/saved_embedding/HG_drug_embeddings.npy"
        ppi_path = paths.xpert_dir / "processed_data/PPI_gene_vector_128d.npy"
        target_drugs_path = paths.input_dir / "drugs.parquet"
        replay_receipt_path = (
            ROOT / "artifacts/experiments/EXP-050/20260930_xpert_validation_replay_v4/summary.json"
        )
        replay_receipt = json.loads(replay_receipt_path.read_text(encoding="utf-8"))
        replay_unimol = replay_receipt["input_hashes"]["official_unimol_features"]
        required_xpert_paths = [
            xpert_snapshot_path, XPERT / "best.pt", XPERT / "protocol.json",
            XPERT / "validation_prediction.npy", xpert_source_table_path,
            unimol_path, graph_path, ppi_path, target_drugs_path,
            replay_receipt_path,
            *xpert_external_files.values(),
        ]
        missing_xpert_paths = [str(path) for path in required_xpert_paths if not path.is_file()]
        if missing_xpert_paths:
            raise FileNotFoundError(f"XPert frozen replay inputs are missing: {missing_xpert_paths}")
        current_unimol_sha = sha256(unimol_path)
        if (replay_receipt.get("status") != "PASS"
                or unimol_path.stat().st_size != replay_unimol["bytes"]
                or replay_unimol["sha256"] !=
                "d5b6aab93e9c5ee8b9b62171379ca6a4d9e9667aaf26691d3cf8e5483a617cc9"
                or current_unimol_sha != replay_unimol["sha256"]):
            raise ValueError("XPert UniMol matrix differs from its frozen full-file replay receipt")
        hash_cache[unimol_path] = current_unimol_sha
        xpert_source_table = pd.read_csv(
            xpert_source_table_path,
            dtype={"pert_id": str, "inchi_key": str},
        )
        unimol = np.load(unimol_path, mmap_mode="r", allow_pickle=False)
        graph = np.load(graph_path, mmap_mode="r", allow_pickle=False)
        model_config = xpert_adapter.load_model_config(paths)
        target_drugs = pd.read_parquet(target_drugs_path)
        drug_index = xpert_adapter.validate_drug_feature_index(
            target_drugs, xpert_source_table, unimol.shape, graph.shape,
            hidden_size=int(model_config["model"]["ATTN"]["hidden_size"]),
            expected_shared_count=6836, exclude_conflicting_nodes=True,
        )

        transigen_saved = np.load(
            TRANSIGEN / "transigen_predictions.npy", mmap_mode="r", allow_pickle=False
        )
        transigen_model = transigen_helper.load_model(transigen_helper.SOURCE_MODEL, device)
        transigen_baseline = transigen_helper.predict_validation(
            transigen_model, validation_control, validation_features,
            local_validation_indices, device,
        )
        if (transigen_baseline.shape != transigen_saved.shape
                or not np.allclose(transigen_baseline, transigen_saved, rtol=0.0, atol=ATOL)):
            raise ValueError("TranSiGen full-input validation replay did not match its saved prediction")
        transigen_masked = transigen_helper.predict_validation(
            transigen_model, masked_validation_control, validation_features,
            local_validation_indices, device,
        )
        transigen_result = {
            "baseline_replay_matches_saved_prediction": True,
            "baseline_max_absolute_difference": float(
                np.max(np.abs(transigen_baseline.astype(np.float64)
                              - np.asarray(transigen_saved, dtype=np.float64)))
            ),
            "missing_gene_mask_sensitivity": summarize_change(
                transigen_baseline, transigen_masked, visible_indices
            ),
        }
        del transigen_model
        torch.cuda.empty_cache()

        train_cells_set = frozenset(train_keys.cell_id.astype(str))
        validation_keys = keys.iloc[validation_positions]
        validation_cells_set = frozenset(validation_keys.cell_id.astype(str))
        split = xpert_adapter.SplitFold(
            0, train_cells_set | validation_cells_set, frozenset(),
            (xpert_adapter.InnerFold(0, train_cells_set, validation_cells_set),),
        )
        zero_delta = np.zeros((len(keys), 978), dtype=np.float32)
        xpert_loader = xpert_adapter.FoldDataLoader(
            keys, keys, keys, zero_delta, control, unimol, graph,
            source_to_target,
            drug_index, split, model_config,
        )
        validation_rows = xpert_adapter.FoldRows("inner_validation", validation_positions)
        protocol_path = XPERT / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if (protocol.get("seed") != 20260922
                or protocol.get("model_class") != "XPertNet"
                or protocol.get("training_rows") != train_positions.tolist()
                or protocol.get("validation_rows") != validation_positions.tolist()
                or not np.array_equal(
                    xpert_loader.source_to_target_gene_indices, protocol.get("gene_map")
                )):
            raise ValueError("XPert source split/gene-order map differs from the frozen protocol")
        checkpoint_path = XPERT / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("protocol_sha256") != xpert_helper.fingerprint(protocol):
            raise ValueError("XPert checkpoint is not bound to its frozen protocol")
        xpert_model, _, _ = xpert_adapter.load_xpert_model(
            paths, xpert_loader.config, device=device, seed=int(protocol["seed"])
        )
        xpert_model.load_state_dict(checkpoint["model"])
        xpert_saved = np.load(
            XPERT / "validation_prediction.npy", mmap_mode="r", allow_pickle=False
        )
        xpert_baseline = xpert_helper.build_prediction(
            xpert_model, xpert_loader, validation_rows, device, int(protocol["batch_size"])
        )
        if (xpert_baseline.shape != xpert_saved.shape
                or not np.allclose(xpert_baseline, xpert_saved, rtol=0.0, atol=ATOL)):
            raise ValueError("XPert full-input validation replay did not match its saved prediction")
        masked_control = np.asarray(control, dtype=np.float32).copy()
        masked_control[np.ix_(validation_positions, missing_indices)] = train_median[missing_indices]
        xpert_loader.control = masked_control
        xpert_masked = xpert_helper.build_prediction(
            xpert_model, xpert_loader, validation_rows, device, int(protocol["batch_size"])
        )
        xpert_result = {
            "baseline_replay_matches_saved_prediction": True,
            "baseline_max_absolute_difference": float(
                np.max(np.abs(xpert_baseline.astype(np.float64)
                              - np.asarray(xpert_saved, dtype=np.float64)))
            ),
            "missing_gene_mask_sensitivity": summarize_change(
                xpert_baseline, xpert_masked, visible_indices
            ),
        }
        del xpert_model
        torch.cuda.empty_cache()

        transigen_masked_path = output / "transigen_masked_validation_prediction.npy"
        xpert_masked_path = output / "xpert_masked_validation_prediction.npy"
        np.save(transigen_masked_path, transigen_masked, allow_pickle=False)
        np.save(xpert_masked_path, xpert_masked, allow_pickle=False)

        input_paths = {
            "experiment_config": config_path,
            "v7_axis": axis_path,
            "v7_manifest": V7 / "run_manifest.json",
            "continuous_gene_order": genes_path,
            "prepared_manifest": manifest_path,
            "prepared_conditions": PREPARED / "conditions.parquet",
            "prepared_control": control_path,
            "prepared_ecfp4": ecfp_path,
            "prepared_train_rows": PREPARED / "train121.npy",
            "prepared_validation_rows": PREPARED / "validation.npy",
            "prepared_gene_map": PREPARED / "source_to_target.npy",
            "transigen_checkpoint": TRANSIGEN / "transigen_best.pt",
            "transigen_preprocess": TRANSIGEN / "transigen_preprocess.npz",
            "transigen_saved_validation_prediction": TRANSIGEN / "transigen_predictions.npy",
            "xpert_adapter_snapshot": xpert_snapshot_path,
            "xpert_checkpoint": checkpoint_path,
            "xpert_protocol": protocol_path,
            "xpert_saved_validation_prediction": XPERT / "validation_prediction.npy",
            "xpert_identity_table": paths.xpert_dir / "processed_data/l1000_sdst_drugs_info_8432.csv",
            "xpert_project_drug_table": paths.input_dir / "drugs.parquet",
            "xpert_graph_features": graph_path,
            "xpert_ppi_features": paths.xpert_dir / "processed_data/PPI_gene_vector_128d.npy",
            "xpert_replay_helper": ROOT / "scripts/data/audit_exp050_frozen_xpert_validation_replay.py",
            "transigen_replay_helper": ROOT / "scripts/data/audit_exp050_frozen_transigen_validation_replay.py",
            "official_transigen_model_source": transigen_helper.SOURCE_MODEL,
        }
        input_paths.update({f"official_xpert_{name}": path
                            for name, path in xpert_external_files.items()})
        input_hashes = {}
        for name, path in input_paths.items():
            digest = hash_cache.get(path)
            if digest is None:
                digest = sha256(path)
            input_hashes[name] = {"bytes": path.stat().st_size, "sha256": digest}
        for filename, expected_sha in prepared_hashes.items():
            actual_name = {
                "conditions.parquet": "prepared_conditions",
                "control.npy": "prepared_control",
                "ecfp4.npy": "prepared_ecfp4",
                "train121.npy": "prepared_train_rows",
                "validation.npy": "prepared_validation_rows",
                "source_to_target.npy": "prepared_gene_map",
            }[filename]
            if input_hashes[actual_name]["sha256"] != expected_sha:
                raise ValueError(f"prepared {filename} changed during the audit")
        elapsed = time.monotonic() - started
        result = {
            "experiment_id": "EXP-050",
            "run_id": run_id,
            "status": "PASS",
            "environment": "WSL2 Conda drugscreening-gpu",
            "device": str(device),
            "device_name": device_name,
            "source_partition": {
                "inner_train_conditions": int(len(train_positions)),
                "unique_inner_train_cells": int(len(unique_cells)),
                "inner_validation_conditions": int(len(validation_positions)),
                "inner_validation_cells": int(validation_keys.cell_id.nunique()),
                "dose_uM": 10,
                "time_hours": 24,
            },
            "target_gene_mask": {
                "model_axis_gene_count": 978,
                "visible_target_gene_count": 914,
                "masked_gene_count": 64,
                "masked_gene_entrez_ids": [model_gene_ids[index] for index in missing_indices],
                "imputation": "median of per-cell median inner-training control profiles; one vote per unique training cell",
                "imputation_vector_sha256": hashlib.sha256(train_median.tobytes()).hexdigest(),
            },
            "models": {"TranSiGen": transigen_result, "XPert": xpert_result},
            "validation_response_values_read": False,
            "outer_test_response_values_read": False,
            "target_treated_responses_read": False,
            "model_or_imputation_selection_performed": False,
            "interpretation_boundary": (
                "Source-only model-output sensitivity to one fixed 64-gene median-imputation rule; "
                "does not validate target scale, target imputation accuracy, target prediction, or efficacy"
            ),
            "input_hashes": input_hashes,
            "xpert_unimol_feature_matrix_provenance": {
                "bytes": int(unimol_path.stat().st_size),
                "sha256_verified_by_prior_replay": replay_unimol["sha256"],
                "prior_replay_receipt": str(replay_receipt_path.relative_to(ROOT)),
                "prior_replay_receipt_sha256": sha256(replay_receipt_path),
            },
            "masked_prediction_artifacts": {
                "TranSiGen": {
                    "path": transigen_masked_path.relative_to(ROOT).as_posix(),
                    "bytes": transigen_masked_path.stat().st_size,
                    "sha256": sha256(transigen_masked_path),
                },
                "XPert": {
                    "path": xpert_masked_path.relative_to(ROOT).as_posix(),
                    "bytes": xpert_masked_path.stat().st_size,
                    "sha256": sha256(xpert_masked_path),
                },
            },
            "gpu_hours_charged": elapsed / 3600.0,
        }
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
        append_jsonl(ledger, {
            "experiment_id": "EXP-050",
            "run_id": run_id,
            "attempt_id": attempt_id,
            "event": "attempt_finished",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gpu_hours_charged": elapsed / 3600.0,
            "status": result["status"],
            "summary_sha256": sha256(summary_path),
            "masked_transigen_prediction_sha256": result["masked_prediction_artifacts"]["TranSiGen"]["sha256"],
            "masked_xpert_prediction_sha256": result["masked_prediction_artifacts"]["XPert"]["sha256"],
        })
        return result
    except Exception as exc:
        elapsed = time.monotonic() - started
        append_jsonl(ledger, {
            "experiment_id": "EXP-050",
            "run_id": run_id,
            "attempt_id": attempt_id,
            "event": "attempt_finished",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gpu_hours_charged": elapsed / 3600.0,
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(
        f"artifacts/experiments/EXP-050/{RUN_ID}"))
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    print(json.dumps(run(output, args.run_id, args.device_index), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
