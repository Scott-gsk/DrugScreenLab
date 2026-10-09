#!/usr/bin/env python3
"""Root-side readback for an immutable EXP-063 epoch-one cost-gate closeout.

Reads only the authorized fixed EXP-062 panel archive and the new EXP-063
epoch-one outputs. It does not reopen the source H5AD or any forbidden cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


REPO = Path("/mnt/d/Code/DrugScreenLab")
SOURCE = REPO / "artifacts/experiments/EXP-062/20261008_exp062_coverage20_fixed_v3"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_macro_nmse(truth: np.ndarray, prediction: np.ndarray,
                      objects: np.ndarray, gene_variance: np.ndarray) -> dict[str, Any]:
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 978:
        raise ValueError(f"keyed target/prediction shape mismatch: {truth.shape} vs {prediction.shape}")
    use_gene = np.isfinite(gene_variance) & (gene_variance > 1e-12)
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all() or not use_gene.any():
        raise ValueError("non-finite target/prediction or empty TRAIN-fitted gene denominator")
    err = prediction.astype(np.float64) - truth.astype(np.float64)
    per_row = np.mean(np.square(err[:, use_gene]) / gene_variance[use_gene][None, :], axis=1)
    frame = pd.DataFrame({"object": objects.astype(str), "nmse": per_row})
    per_object = frame.groupby("object", sort=True)["nmse"].mean()
    return {"object_macro_nmse": float(per_object.mean()), "conditions": int(len(truth)),
            "objects": int(len(per_object)), "genes_in_denominator": int(use_gene.sum()),
            "finite_target_values": int(truth.size), "finite_prediction_values": int(prediction.size)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="artifacts/experiments/EXP-063/coverage20_official_v1")
    parser.add_argument("--attempt-id", default="fit_attempt_003")
    args = parser.parse_args()
    started = time.perf_counter()
    attempt = REPO / args.run_root / args.attempt_id
    receipt_path = attempt / "root_independent_cost_gate_readback.json"
    if receipt_path.exists():
        raise FileExistsError(f"immutable root readback already exists: {receipt_path}")
    try:
        gate = json.loads((attempt / "first_epoch_cost_gate.json").read_text(encoding="utf-8"))
        manifest = json.loads((attempt / "run_manifest.json").read_text(encoding="utf-8"))
        cpu = json.loads((attempt / "cpu_resource_ledger.json").read_text(encoding="utf-8"))
        curve = json.loads((attempt / "training_curve.jsonl").read_text(encoding="utf-8").splitlines()[0])
        metadata_root = json.loads((REPO / args.run_root / "preflight/metadata_axis_v6/root_independent_gate_readback.json").read_text(encoding="utf-8"))
        numeric_root = json.loads((SOURCE / "root_independent_readback_fit_attempt_002_v5/root_independent_readback.json").read_text(encoding="utf-8"))
        metadata_manifest = json.loads((SOURCE / "metadata_freeze_manifest.json").read_text(encoding="utf-8"))
        if metadata_root.get("status") != "PASS_ROOT_INDEPENDENT_METADATA_GATE_READBACK":
            raise ValueError("EXP-063 metadata gate lacked root PASS")
        if numeric_root.get("status") != "PASS_ROOT_INDEPENDENT_NUMERIC_READBACK" or numeric_root.get("source_values_independently_reopened") is not False:
            raise ValueError("EXP-062 numeric source was not bound to the root-verified frozen package")
        if gate.get("status") != "FAIL_CLOSED_SEAL_EPOCH1_COST_GATE":
            raise ValueError(f"expected a sealed epoch-one cost-gate failure, got {gate.get('status')}")
        cap = float(gate["effective_gpu_cap_hours"])
        projection = float(gate["projection_with_margin_gpu_hours"])
        if not projection > cap:
            raise ValueError("cost-gate failure is inconsistent with the projected 2500-epoch schedule")
        if gate.get("selected_rows_only") is not True or gate.get("forbidden_numeric_values_read") != []:
            raise ValueError("fit gate does not attest the authorized row-only domain")

        keys = pd.read_csv(SOURCE / "response_blind_conditions.tsv", sep="\t", dtype=str, keep_default_na=False)
        train_mask = keys["effective_split_role"].eq("train").to_numpy()
        dev_mask = keys["effective_split_role"].eq("inner_dev").to_numpy()
        train_keys, dev_keys = keys.loc[train_mask].reset_index(drop=True), keys.loc[dev_mask].reset_index(drop=True)
        pred_path = attempt / "epoch001_keyed_predictions.npz"
        with np.load(pred_path, allow_pickle=False) as z:
            required = {
                "train_condition_key", "inner_dev_condition_key", "train_source_object_id", "inner_dev_source_object_id",
                "train_full_inchikey", "inner_dev_full_inchikey", "genes", "train_target_delta", "inner_dev_target_delta",
                "XPert_epoch001_train_delta", "XPert_epoch001_inner_dev_delta",
            }
            missing = sorted(required - set(z.files))
            if missing:
                raise KeyError(f"epoch-one keyed archive missing fields: {missing}")
            arrays = {name: z[name] for name in required}
        for field, expected in (
                ("train_condition_key", train_keys["full_condition_key"].to_numpy(dtype=str)),
                ("inner_dev_condition_key", dev_keys["full_condition_key"].to_numpy(dtype=str)),
                ("train_source_object_id", train_keys["source_object_id"].to_numpy(dtype=str)),
                ("inner_dev_source_object_id", dev_keys["source_object_id"].to_numpy(dtype=str)),
                ("train_full_inchikey", train_keys["full_inchikey"].to_numpy(dtype=str)),
                ("inner_dev_full_inchikey", dev_keys["full_inchikey"].to_numpy(dtype=str))):
            if not np.array_equal(arrays[field].astype(str), expected):
                raise ValueError(f"epoch-one key readback does not match the frozen EXP-062 rows: {field}")
        with np.load(SOURCE / "fit_attempt_002/keyed_predictions.npz", allow_pickle=False) as source_archive:
            if not np.array_equal(arrays["genes"].astype(str), source_archive["gene_axis"].astype(str)):
                raise ValueError("epoch-one gene axis differs from EXP-062 root-bound gene axis")
        if len(keys) != 1982 or int(train_mask.sum()) != 1291 or int(dev_mask.sum()) != 691:
            raise ValueError("frozen row-role counts changed")
        if train_keys.source_object_id.nunique() != 93 or dev_keys.source_object_id.nunique() != 48:
            raise ValueError("frozen object denominators changed")
        if set(train_keys.source_object_id) & set(dev_keys.source_object_id):
            raise ValueError("TRAIN and inner-dev source objects overlap")
        if keys.full_inchikey.nunique() != 20 or arrays["genes"].size != 978:
            raise ValueError("frozen drug/gene axis denominator changed")
        if train_keys.full_condition_key.duplicated().any() or dev_keys.full_condition_key.duplicated().any():
            raise ValueError("duplicate condition key in saved epoch-one predictions")

        train_y = arrays["train_target_delta"].astype(np.float32)
        dev_y = arrays["inner_dev_target_delta"].astype(np.float32)
        train_pred = arrays["XPert_epoch001_train_delta"].astype(np.float32)
        dev_pred = arrays["XPert_epoch001_inner_dev_delta"].astype(np.float32)
        if train_y.shape != (1291, 978) or dev_y.shape != (691, 978):
            raise ValueError("epoch-one output dimensions differ from frozen row/gene contract")
        gene_variance = np.var(train_y.astype(np.float64), axis=0, ddof=0)
        xpert_train = object_macro_nmse(train_y, train_pred, arrays["train_source_object_id"], gene_variance)
        xpert_dev = object_macro_nmse(dev_y, dev_pred, arrays["inner_dev_source_object_id"], gene_variance)
        per_key_mean = {}
        for key in sorted(train_keys.full_inchikey.astype(str).unique()):
            per_key_mean[key] = train_y[train_keys.full_inchikey.astype(str).to_numpy() == key].mean(axis=0, dtype=np.float64).astype(np.float32)
        train_mean_pred = np.stack([per_key_mean[str(key)] for key in train_keys.full_inchikey.astype(str)])
        dev_mean_pred = np.stack([per_key_mean[str(key)] for key in dev_keys.full_inchikey.astype(str)])
        if not np.isfinite(dev_mean_pred).all():
            raise ValueError("non-finite exact-key mean reconstruction")
        mean_train = object_macro_nmse(train_y, train_mean_pred, train_keys.source_object_id.to_numpy(dtype=str), gene_variance)
        mean_dev = object_macro_nmse(dev_y, dev_mean_pred, dev_keys.source_object_id.to_numpy(dtype=str), gene_variance)

        checkpoint = torch.load(attempt / "XPert_epoch001_checkpoint.pt", map_location="cpu", weights_only=True)
        best = torch.load(attempt / "XPert_best_inner_dev.pt", map_location="cpu", weights_only=True)
        if checkpoint.get("epoch_one_based") != 1 or best.get("epoch_one_based") != 1 or checkpoint.get("seed") != 20261006:
            raise ValueError("epoch-one checkpoint identity/readback mismatch")
        def tensor_check(state: Any) -> tuple[int, int]:
            tensors = [v for v in state.values() if isinstance(v, torch.Tensor) and v.is_floating_point()]
            bad = sum(int((~torch.isfinite(v)).sum().item()) for v in tensors)
            return len(tensors), bad
        model_tensor_count, model_nonfinite = tensor_check(checkpoint["model_state_dict"])
        best_tensor_count, best_nonfinite = tensor_check(best["model_state_dict"])
        if model_nonfinite or best_nonfinite or model_tensor_count == 0 or best_tensor_count != model_tensor_count:
            raise ValueError("epoch-one checkpoint weights are incomplete or non-finite")
        if not np.allclose(xpert_train["object_macro_nmse"], curve["train_metrics"]["object_macro_nmse"], rtol=0, atol=1e-10):
            raise ValueError("independent TRAIN NMSE does not match epoch-one training curve")
        if not np.allclose(xpert_dev["object_macro_nmse"], curve["inner_dev_metrics"]["object_macro_nmse"], rtol=0, atol=1e-10):
            raise ValueError("independent inner-dev NMSE does not match epoch-one training curve")
        if not math.isclose(xpert_train["object_macro_nmse"], float(gate["epoch1_train_nmse"]), rel_tol=0, abs_tol=1e-10):
            raise ValueError("independent TRAIN NMSE does not match cost-gate receipt")
        if not math.isclose(xpert_dev["object_macro_nmse"], float(gate["epoch1_inner_dev_nmse_descriptive_only"]), rel_tol=0, abs_tol=1e-10):
            raise ValueError("independent inner-dev NMSE does not match cost-gate receipt")
        if not math.isclose(mean_train["object_macro_nmse"], float(json.loads((attempt / "baseline_metrics.json").read_text(encoding="utf-8"))["metrics"]["per_drug_mean"]["train"]["object_macro_nmse"]), rel_tol=0, abs_tol=1e-10):
            raise ValueError("independent exact-key mean TRAIN baseline differs from execution receipt")
        if not math.isclose(mean_dev["object_macro_nmse"], float(json.loads((attempt / "baseline_metrics.json").read_text(encoding="utf-8"))["metrics"]["per_drug_mean"]["inner_dev"]["object_macro_nmse"]), rel_tol=0, abs_tol=1e-10):
            raise ValueError("independent exact-key mean inner-dev baseline differs from execution receipt")

        output_names = ["baseline_metrics.json", "cpu_resource_ledger.json", "epoch001_keyed_predictions.npz",
                        "first_epoch_cost_gate.json", "initial_checkpoint.pt", "run_manifest.json",
                        "training_curve.jsonl", "XPert_best_inner_dev.pt", "XPert_epoch001_checkpoint.pt"]
        outputs = {name: sha256_file(attempt / name) for name in output_names}
        receipt = {
            "experiment_id": "EXP-063", "run_id": "coverage20_official_v1", "attempt_id": args.attempt_id,
            "status": "PASS_ROOT_INDEPENDENT_EPOCH1_COST_GATE_READBACK_CLOSED_NO_FULL_FIT",
            "cost_gate_status": gate["status"], "effective_gpu_cap_hours": cap,
            "full_schedule_projection_gpu_hours": projection,
            "projection_excess_gpu_hours": projection - cap,
            "actual_gpu_hours_after_epoch1": manifest["gpu_hours_after_epoch1"],
            "denominators": {"rows": len(keys), "full_inchikeys": int(keys.full_inchikey.nunique()),
                             "genes": int(arrays["genes"].size), "train_conditions": int(train_mask.sum()),
                             "train_objects": int(train_keys.source_object_id.nunique()),
                             "inner_dev_conditions": int(dev_mask.sum()), "inner_dev_objects": int(dev_keys.source_object_id.nunique()),
                             "train_dev_object_overlap": 0, "finite_target_values": int(train_y.size + dev_y.size)},
            "root_recomputed_metrics": {"epoch1_XPert_train": xpert_train, "epoch1_XPert_inner_dev_descriptive_only": xpert_dev,
                                         "exact_key_train_mean_train": mean_train, "exact_key_train_mean_inner_dev": mean_dev},
            "checkpoint_readback": {"weights_only": True, "epoch_one_based": 1, "seed": checkpoint["seed"],
                                     "floating_model_tensors": model_tensor_count, "nonfinite_model_values": model_nonfinite},
            "source_lineage": {"exp062_numeric_root_sha256": sha256_file(SOURCE / "root_independent_readback_fit_attempt_002_v5/root_independent_readback.json"),
                               "exp063_axis_root_readback_sha256": sha256_file(REPO / args.run_root / "preflight/metadata_axis_v6/root_independent_gate_readback.json"),
                               "exp062_key_file_sha256": metadata_manifest["frozen_files_sha256"]["response_blind_conditions.tsv"]},
            "output_sha256": outputs,
            "forbidden_numeric_values_read": [],
            "claim_boundary": "first-epoch cost-gate artifact only; no full XPert fit or candidate validation result",
            "resources": {"cpu_prior_package_seconds": cpu["current_cpu_cumulative_seconds"],
                          "root_readback_wall_seconds": time.perf_counter() - started,
                          "cpu_hard_cap_seconds": cpu["cpu_hard_cap_seconds"], "gpu_hours": 0},
        }
        receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path),
                          "projection_gpu_hours": projection, "gpu_cap_hours": cap,
                          "epoch1_train_nmse": xpert_train["object_macro_nmse"],
                          "epoch1_inner_dev_nmse_descriptive_only": xpert_dev["object_macro_nmse"]}, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        failure = {"experiment_id": "EXP-063", "run_id": "coverage20_official_v1", "attempt_id": args.attempt_id,
                   "status": "FAILED_CLOSED_ROOT_READBACK", "error_type": type(exc).__name__, "error": str(exc),
                   "forbidden_numeric_values_read": [], "resources": {"cpu_hard_cap_seconds": 900, "gpu_hours": 0}}
        receipt_path.write_text(json.dumps(failure, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(failure, indent=2, ensure_ascii=False), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
