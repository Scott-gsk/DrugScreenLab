"""Replay the frozen EXP-037 XPert checkpoint on its inner-validation inputs.

The stored XPert forward signature includes treatment fields for its training
loss, but the frozen official model uses control expression plus drug and
exposure features to produce its delta output. This replay supplies synthetic
zero deltas to the unused treatment fields, so it does not read validation
response values or outer-test responses.
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


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/experiments/EXP-037/20260928_lincs_scale_16_121"
PREPARED = RUN / "prepared"
SNAPSHOT = RUN / "wrapper_xpert121/code_snapshot"
MODEL_OUTPUT = RUN / "xpert_121"
RUN_ID = "20260930_xpert_validation_replay_v4"
ATTEMPT_ID = "EXP050-20260930-XPERT-VALIDATION-REPLAY-V4"
RESERVED_GPU_HOURS = 0.25
ATOL = 1e-5

EXPECTED_HASHES = {
    "exp037_lincs_scale_probe.py": "bfc5ef97bbaf846d39bad52e44a3e8567988a663b710b0dbedf1f44fb623ae4c",
    "exp037_xpert_fold.py": "5e6e47f7b0ab82269300c688ce69bfbfb9435c6ad75319b0c487e3d296de39b0",
    "exp037_xpert_train.py": "2c9064fe7fe7b966e1a0142be38f36b77071d84317c58ef025b6fac474383e1f",
    "model_XPert.py": "87b08745515f72813a622170d2df774d3b2aac0975e0b93e07008ad67772e5ba",
    "model_utils.py": "b2cffc612266ca3094bb509b312baba605103936c599c86150634838be1237d6",
    "xpert_utils.py": "da15f152943a602b3c38d8e1ba2365e4e47c74c16b16e37423d28f8132faa95a",
    "MyDataset.py": "d4cb1ed1d293d66ddff0e4f71c4931b2719c4cbbea7ece32da7d62651c6c2eda",
    "config_l1000.yaml": "94a2206ced6e679ee9556595459cec4a871219cf94487cd74f9f83f9d1a96b41",
}


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


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_snapshot_adapter():
    probe_path = SNAPSHOT / "exp037_lincs_scale_probe.py"
    adapter_path = SNAPSHOT / "exp037_xpert_fold.py"
    trainer_path = SNAPSHOT / "exp037_xpert_train.py"
    if sha256(probe_path) != EXPECTED_HASHES[probe_path.name]:
        raise ValueError("XPert preparation/training entrypoint differs from the frozen EXP-037 provenance")
    if sha256(adapter_path) != EXPECTED_HASHES[adapter_path.name]:
        raise ValueError("XPert adapter snapshot differs from the frozen EXP-037 provenance")
    if sha256(trainer_path) != EXPECTED_HASHES[trainer_path.name]:
        raise ValueError("XPert trainer snapshot differs from the frozen EXP-037 provenance")

    source_root = ROOT / "data/external/xpert_source"
    external_files = {
        "model_XPert.py": source_root / "models/model_XPert.py",
        "model_utils.py": source_root / "models/model_utils.py",
        "xpert_utils.py": source_root / "utils.py",
        "MyDataset.py": source_root / "datasets/MyDataset.py",
        "config_l1000.yaml": source_root / "configs/config_l1000.yaml",
    }
    for name, path in external_files.items():
        if sha256(path) != EXPECTED_HASHES[name]:
            raise ValueError(f"official XPert source hash mismatch: {name}")

    if str(SNAPSHOT) not in sys.path:
        sys.path.insert(0, str(SNAPSHOT))
    spec = importlib.util.spec_from_file_location("exp050_xpert_adapter", adapter_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load the frozen EXP-037 XPert adapter")
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    return adapter, external_files


def build_prediction(model, loader, rows, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows.positions), batch_size):
            positions = rows.positions[start:start + batch_size]
            # loader.delta is replaced with zeros before this call. The official
            # model's hashed forward code uses ctl_raw_data_binned, drug, dose,
            # and time to make deg_output; treatment fields are not prediction inputs.
            batch = loader.batch(rows, positions, device=device)
            pred = model(batch.model_inputs)[2]
            if not torch.isfinite(pred).all():
                raise ValueError("nonfinite XPert validation replay prediction")
            predictions.append(pred.cpu().numpy())
    source_order = np.concatenate(predictions, axis=0)
    # source_to_target maps each source column to its target-axis position.
    return source_order[:, np.argsort(loader.source_to_target_gene_indices)]


def run(output: Path, device_index: int) -> dict[str, Any]:
    if sys.platform != "linux" or os.environ.get("CONDA_DEFAULT_ENV") != "drugscreening-gpu":
        raise RuntimeError("EXP-050 requires WSL2 Conda environment drugscreening-gpu")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing replay output: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible for the frozen XPert checkpoint replay")

    output.mkdir(parents=True)
    ledger = ROOT / "artifacts/experiments/EXP-050/gpu_ledger.jsonl"
    torch.set_num_threads(4)
    device = torch.device(f"cuda:{device_index}")
    device_name = torch.cuda.get_device_name(device)
    started = time.monotonic()
    append_jsonl(ledger, {
        "experiment_id": "EXP-050", "run_id": RUN_ID, "attempt_id": ATTEMPT_ID,
        "event": "attempt_started",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reserved_gpu_hours": RESERVED_GPU_HOURS,
        "device": str(device), "device_name": device_name,
        "operation": "frozen checkpoint inner-validation replay; no training",
    })

    try:
        adapter, external_files = load_snapshot_adapter()
        paths = adapter.ProjectPaths.from_root(ROOT)
        prepared_manifest = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))
        if (prepared_manifest.get("status") != "PREPARED_TRAIN_VALIDATION_ONLY"
                or prepared_manifest.get("n_train121") != 32094
                or prepared_manifest.get("n_validation") != 4954
                or prepared_manifest.get("official_test_responses_read") is not False):
            raise ValueError("prepared EXP-037 inputs are not the frozen train/validation-only tier")
        manifest_hashes = prepared_manifest["files_sha256"]
        for filename in (
            "control.npy", "ecfp4.npy", "source_to_target.npy", "train121.npy",
            "validation.npy", "conditions.parquet", "gene_scale.npy",
        ):
            actual = sha256(PREPARED / filename)
            if actual != manifest_hashes[filename]:
                raise ValueError(f"prepared XPert input differs from its frozen manifest: {filename}")

        # Reconstruct the exact 10 uM x 24 h filtered inputs used by the
        # original fit-xpert code. The general XPert adapter also includes
        # other dose/time rows and is not the checkpoint's trained partition.
        keys = pd.read_parquet(PREPARED / "conditions.parquet")
        control = np.load(PREPARED / "control.npy", mmap_mode="r", allow_pickle=False)
        delta = np.load(PREPARED / "delta.npy", mmap_mode="r", allow_pickle=False)
        mapping = np.load(PREPARED / "source_to_target.npy", allow_pickle=False)
        train_positions = np.load(PREPARED / "train121.npy", allow_pickle=False)
        validation_positions = np.load(PREPARED / "validation.npy", allow_pickle=False)
        if (control.shape != (len(keys), 978) or delta.shape != control.shape
                or mapping.shape != (978,) or not np.array_equal(np.sort(mapping), np.arange(978))
                or train_positions.ndim != 1 or validation_positions.ndim != 1
                or len(train_positions) != 32094 or len(validation_positions) != 4954
                or len(train_positions) + len(validation_positions) != len(keys)
                or np.intersect1d(train_positions, validation_positions).size):
            raise ValueError("prepared XPert axes or frozen partition sizes are invalid")
        if (not np.allclose(pd.to_numeric(keys.pert_dose).to_numpy(np.float64), 10.0)
                or not np.allclose(pd.to_numeric(keys.pert_time).to_numpy(np.float64), 24.0)):
            raise ValueError("prepared XPert conditions include an unexpected dose or time")

        target_drugs = pd.read_parquet(paths.input_dir / "drugs.parquet")
        source_drugs = pd.read_csv(
            paths.xpert_dir / "processed_data/l1000_sdst_drugs_info_8432.csv",
            dtype={"pert_id": str, "inchi_key": str},
        )
        unimol_path = paths.xpert_dir / "processed_data/all_drugs_unimol_arr.npy"
        hg_path = paths.xpert_dir / "HG_data/saved_embedding/HG_drug_embeddings.npy"
        unimol = np.load(unimol_path, mmap_mode="r", allow_pickle=False)
        hg = np.load(hg_path, mmap_mode="r", allow_pickle=False)
        config = adapter.load_model_config(paths)
        drug_index = adapter.validate_drug_feature_index(
            target_drugs, source_drugs, unimol.shape, hg.shape,
            hidden_size=int(config["model"]["ATTN"]["hidden_size"]),
            expected_shared_count=6836, exclude_conflicting_nodes=True,
        )
        train_cells = frozenset(keys.iloc[train_positions].cell_id.astype(str))
        validation_cells = frozenset(keys.iloc[validation_positions].cell_id.astype(str))
        split = adapter.SplitFold(
            0, train_cells | validation_cells, frozenset(),
            (adapter.InnerFold(0, train_cells, validation_cells),),
        )
        loader = adapter.FoldDataLoader(
            keys, keys, keys, delta, control, unimol, hg, mapping, drug_index, split, config
        )
        train_rows = adapter.FoldRows("inner_train", train_positions)
        validation_rows = adapter.FoldRows("inner_validation", validation_positions)

        protocol = json.loads((MODEL_OUTPUT / "protocol.json").read_text(encoding="utf-8"))
        if (protocol.get("conditioning_bins") is not None
                or protocol.get("seed") != 20260922
                or protocol.get("batch_size") != 16
                or protocol.get("model_class") != "XPertNet"):
            raise ValueError("frozen XPert protocol does not match the expected validation replay")
        def row_comparison(actual: np.ndarray, expected: list[int]) -> dict[str, Any]:
            expected_array = np.asarray(expected, dtype=np.int64)
            return {
                "actual_n": int(len(actual)), "expected_n": int(len(expected_array)),
                "same_order": bool(np.array_equal(actual, expected_array)),
                "same_rows": bool(np.array_equal(np.sort(actual), np.sort(expected_array))),
                "actual_first_8": actual[:8].tolist(),
                "expected_first_8": expected_array[:8].tolist(),
                "n_actual_only": int(len(set(actual.tolist()) - set(expected))),
                "n_expected_only": int(len(set(expected) - set(actual.tolist()))),
            }
        train_row_comparison = row_comparison(train_rows.positions, protocol["training_rows"])
        validation_row_comparison = row_comparison(
            validation_rows.positions, protocol["validation_rows"]
        )
        if (not train_row_comparison["same_order"]
                or not validation_row_comparison["same_order"]
                or len(train_rows.positions) != 32094
                or len(validation_rows.positions) != 4954):
            raise ValueError(json.dumps({
                "message": "reconstructed inner partition does not match the frozen XPert row lists",
                "train_rows": train_row_comparison,
                "validation_rows": validation_row_comparison,
            }, sort_keys=True))
        if not np.array_equal(loader.source_to_target_gene_indices, protocol["gene_map"]):
            raise ValueError("XPert gene-order map differs from the frozen protocol")
        expected_contract = protocol.get("input_contract", {})
        plan_path = SNAPSHOT / "exp037_lincs_scale_probe_20260928.json"
        if (expected_contract.get("prepared_manifest_sha256") != sha256(PREPARED / "manifest.json")
                or expected_contract.get("scale_probe_plan_sha256") != sha256(plan_path)
                or expected_contract.get("gene_scale_sha256") != sha256(PREPARED / "gene_scale.npy")
                or expected_contract.get("tier") != 121
                or expected_contract.get("official_test_responses_read") is not False):
            raise ValueError("XPert checkpoint input contract is not bound to the frozen 121-cell tier")

        saved_keys = pd.read_parquet(MODEL_OUTPUT / "validation_prediction_keys.parquet")
        actual_keys = loader.keys.iloc[validation_rows.positions]
        key_columns = ["condition_id", "cell_id", "pert_id"]
        if any(column not in saved_keys or column not in actual_keys for column in key_columns):
            raise ValueError("saved XPert validation prediction keys lack required identities")
        if not saved_keys[key_columns].reset_index(drop=True).equals(
                actual_keys[key_columns].reset_index(drop=True)):
            raise ValueError("saved XPert validation prediction identity/order differs from the split")

        checkpoint_path = MODEL_OUTPUT / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("protocol_sha256") != fingerprint(protocol):
            raise ValueError("XPert checkpoint protocol hash does not match protocol.json")

        # The upstream model returns its input treatment fields for auxiliary
        # losses but does not use them to construct deg_output. Synthetic zeros
        # here prevent any validation response values from being accessed.
        loader.delta = np.zeros((len(loader.keys), adapter.GENE_COUNT), dtype=np.float32)
        model, _, _ = adapter.load_xpert_model(
            paths, loader.config, device=device, seed=int(protocol["seed"])
        )
        model.load_state_dict(checkpoint["model"])
        replay = build_prediction(model, loader, validation_rows, device, int(protocol["batch_size"]))
        saved = np.load(MODEL_OUTPUT / "validation_prediction.npy", mmap_mode="r", allow_pickle=False)
        if replay.shape != saved.shape or replay.shape != (4954, 978):
            raise ValueError("replayed and saved XPert validation prediction shapes differ")
        if not np.isfinite(replay).all() or not np.isfinite(saved).all():
            raise ValueError("XPert validation prediction contains nonfinite values")
        difference = np.abs(replay.astype(np.float64) - np.asarray(saved, dtype=np.float64))
        matches = bool(np.allclose(replay, saved, rtol=0.0, atol=ATOL))

        input_paths = {
            "prepared_manifest": PREPARED / "manifest.json",
            "prepared_conditions": PREPARED / "conditions.parquet",
            "prepared_control": PREPARED / "control.npy",
            "prepared_ecfp4": PREPARED / "ecfp4.npy",
            "prepared_gene_map": PREPARED / "source_to_target.npy",
            "prepared_train_rows": PREPARED / "train121.npy",
            "prepared_validation_rows": PREPARED / "validation.npy",
            "prepared_gene_scale": PREPARED / "gene_scale.npy",
            "adapter_snapshot": SNAPSHOT / "exp037_xpert_fold.py",
            "trainer_snapshot": SNAPSHOT / "exp037_xpert_train.py",
            "experiment_plan_snapshot": SNAPSHOT / "exp037_lincs_scale_probe_20260928.json",
            "preparation_code_snapshot": SNAPSHOT / "exp037_lincs_scale_probe.py",
            "official_config": external_files["config_l1000.yaml"],
            "official_drug_identity_table": paths.xpert_dir / "processed_data/l1000_sdst_drugs_info_8432.csv",
            "official_unimol_features": unimol_path,
            "official_graph_features": hg_path,
            "official_ppi_features": paths.xpert_dir / "processed_data/PPI_gene_vector_128d.npy",
            "checkpoint": checkpoint_path,
            "protocol": MODEL_OUTPUT / "protocol.json",
            "training_gene_scale": MODEL_OUTPUT / "training_gene_scale_source_order.npy",
            "saved_validation_keys": MODEL_OUTPUT / "validation_prediction_keys.parquet",
            "saved_validation_prediction": MODEL_OUTPUT / "validation_prediction.npy",
            "official_model": external_files["model_XPert.py"],
            "official_model_utils": external_files["model_utils.py"],
            "official_dataset_utils": external_files["xpert_utils.py"],
            "official_dataset": external_files["MyDataset.py"],
        }
        hashes = {name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                  for name, path in input_paths.items()}
        elapsed = time.monotonic() - started
        result: dict[str, Any] = {
            "experiment_id": "EXP-050", "run_id": RUN_ID,
            "status": "PASS" if matches else "MISMATCH",
            "environment": "WSL2 Conda drugscreening-gpu",
            "device": str(device), "device_name": device_name,
            "original_unambiguous_drug_count": len(drug_index.exact_shared_pert_ids),
            "inner_train_conditions": int(len(train_rows.positions)),
            "inner_validation_conditions": int(len(validation_rows.positions)),
            "validation_cells": int(actual_keys.cell_id.nunique()),
            "validation_drugs": int(actual_keys.pert_id.nunique()),
            "gene_count": 978,
            "checkpoint_loaded": True,
            "checkpoint_protocol_match": True,
            "validation_prediction_keys_match": True,
            "replay_match": matches,
            "absolute_tolerance": ATOL,
            "max_absolute_difference": float(difference.max()),
            "mean_absolute_difference": float(difference.mean()),
            "p99_absolute_difference": float(np.quantile(difference, 0.99)),
            "validation_response_values_read": False,
            "validation_response_values_used_for_scoring_or_selection": False,
            "outer_test_response_values_read": False,
            "target_treated_responses_read": False,
            "loader_scope": {
                "data": "EXP-037 prepared 10 uM x 24 h condition subset",
                "validation_partition": "inner validation only",
                "drug_identity_count": len(drug_index.exact_shared_pert_ids),
                "training_rows_sha256": sha256(PREPARED / "train121.npy"),
                "validation_rows_sha256": sha256(PREPARED / "validation.npy"),
                "prepared_delta_values_read": False,
            },
            "protocol_sha256": fingerprint(protocol),
            "prepared_manifest_sha256": sha256(PREPARED / "manifest.json"),
            "prepared_delta_sha256_declared_not_recomputed": prepared_manifest["files_sha256"]["delta.npy"],
            "inference_boundary": "source inner-validation prediction replay only; no target-domain score or patient-response claim",
            "input_hashes": hashes,
            "elapsed_seconds": elapsed,
            "gpu_hours_charged": elapsed / 3600.0,
        }
        replay_path = output / "replayed_validation_prediction.npy"
        np.save(replay_path, replay, allow_pickle=False)
        result["replayed_prediction"] = {
            "path": replay_path.relative_to(ROOT).as_posix(),
            "bytes": replay_path.stat().st_size,
            "sha256": sha256(replay_path),
        }
        summary_path = output / "summary.json"
        summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
        append_jsonl(ledger, {
            "experiment_id": "EXP-050", "run_id": RUN_ID, "attempt_id": ATTEMPT_ID,
            "event": "attempt_finished",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gpu_hours_charged": elapsed / 3600.0,
            "status": result["status"], "summary_sha256": sha256(summary_path),
            "replayed_prediction_sha256": result["replayed_prediction"]["sha256"],
        })
        return result
    except Exception as exc:
        elapsed = time.monotonic() - started
        append_jsonl(ledger, {
            "experiment_id": "EXP-050", "run_id": RUN_ID, "attempt_id": ATTEMPT_ID,
            "event": "attempt_finished",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gpu_hours_charged": elapsed / 3600.0, "status": "FAILED",
            "error_type": type(exc).__name__, "error": str(exc),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(
        f"artifacts/experiments/EXP-050/{RUN_ID}"))
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    print(json.dumps(run(output, args.device_index), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
