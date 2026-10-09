#!/usr/bin/env python3
"""Run the frozen EXP-063 original-XPert fit on the EXP-062 fixed source panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import yaml


REPO = Path("/mnt/d/Code/DrugScreenLab")
XPERT = Path("/mnt/d/Code/XPert")
SOURCE = REPO / "artifacts/experiments/EXP-062/20261008_exp062_coverage20_fixed_v3"
EXP051 = REPO / "artifacts/experiments/EXP-051/20261006_exp051_contract_fit_smoke_v1"
GLOBAL_GPU_SETTLED_HOURS = 1.670268035804


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(torch: Any, value: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def finite_state(torch: Any, state: dict[str, Any]) -> tuple[bool, list[str]]:
    bad = [name for name, value in state.items() if isinstance(value, torch.Tensor)
           and value.is_floating_point() and not bool(torch.isfinite(value).all().item())]
    return not bad, bad


def finite_optimizer(torch: Any, optimizer_state: dict[str, Any]) -> tuple[bool, list[str]]:
    bad: list[str] = []
    def walk(value: Any, prefix: str) -> None:
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            if not bool(torch.isfinite(value).all().item()):
                bad.append(prefix)
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{prefix}.{key}")
        elif isinstance(value, (list, tuple)):
            for ix, item in enumerate(value):
                walk(item, f"{prefix}.{ix}")
    walk(optimizer_state, "optimizer")
    return not bad, bad[:20]


def load_selected_values(keys: pd.DataFrame, h5ad_path: Path, expected_genes: int) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    rows = pd.to_numeric(keys["source_row"], errors="raise").to_numpy(dtype=np.int64)
    if len(rows) != 1982 or len(np.unique(rows)) != 1982:
        raise ValueError("frozen EXP-062 source rows are not 1982 unique rows")
    order = np.argsort(rows, kind="mergesort")
    sorted_rows = rows[order]
    h5 = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if h5.shape != (78453, expected_genes) or "X_ctl" not in h5.obsm:
            raise ValueError(f"source H5AD shape/control contract changed: shape={h5.shape}, obsm={list(h5.obsm.keys())}")
        x_raw = h5.X[sorted_rows, :]
        ctl_raw = h5.obsm["X_ctl"][sorted_rows, :]
        if hasattr(x_raw, "to_memory"):
            x_raw = x_raw.to_memory()
        if hasattr(ctl_raw, "to_memory"):
            ctl_raw = ctl_raw.to_memory()
        if hasattr(x_raw, "toarray"):
            x_raw = x_raw.toarray()
        if hasattr(ctl_raw, "toarray"):
            ctl_raw = ctl_raw.toarray()
        x_sorted = np.asarray(x_raw, dtype=np.float32)
        ctl_sorted = np.asarray(ctl_raw, dtype=np.float32)
        x = np.empty_like(x_sorted)
        ctl = np.empty_like(ctl_sorted)
        x[order] = x_sorted
        ctl[order] = ctl_sorted
        genes = np.asarray(h5.var_names.astype(str), dtype=np.str_)
    finally:
        h5.file.close()
    if x.shape != (1982, expected_genes) or ctl.shape != x.shape or len(genes) != expected_genes:
        raise ValueError(f"selected source numeric dimensions changed: X={x.shape}, ctl={ctl.shape}, genes={len(genes)}")
    if not np.isfinite(x).all() or not np.isfinite(ctl).all():
        raise FloatingPointError("non-finite source values in the exact frozen rows")
    delta = x - ctl
    if not np.isfinite(delta).all() or not np.allclose(x.astype(np.float64) - ctl.astype(np.float64), delta.astype(np.float64), rtol=0, atol=1e-6):
        raise FloatingPointError("selected X-X_ctl failed finite/round-trip validation")
    return {"X": x, "X_ctl": ctl, "delta": delta, "genes": genes}, rows, {
        "matrix_rows_read_exactly_selected": True, "selected_rows": len(rows),
        "x_shape": list(x.shape), "x_ctl_shape": list(ctl.shape), "delta_shape": list(delta.shape),
        "all_selected_X_finite": True, "all_selected_X_ctl_finite": True,
        "all_selected_delta_finite": True,
        "max_abs_X_minus_Xctl_roundtrip_error": float(np.max(np.abs((x.astype(np.float64)-ctl.astype(np.float64))-delta.astype(np.float64)))),
        "forbidden_numeric_values_read": [],
    }


def load_baselines(keys: pd.DataFrame, values: dict[str, np.ndarray], train_mask: np.ndarray,
                   dev_mask: np.ndarray, gene_variance: np.ndarray, archive_path: Path,
                   expected_archive_sha: str, metric_fn: Any) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    actual_sha = sha256_file(archive_path)
    if actual_sha != expected_archive_sha:
        raise ValueError("EXP-062 verified baseline prediction archive SHA mismatch")
    train_keys = keys.loc[train_mask].reset_index(drop=True)
    dev_keys = keys.loc[dev_mask].reset_index(drop=True)
    train_y, dev_y = values["delta"][train_mask], values["delta"][dev_mask]
    n_train, n_dev = len(train_y), len(dev_y)
    suffixes = {
        "zero_change": ("zero_change_train", "zero_change_inner_dev"),
        "per_drug_mean": ("per_drug_mean_train", "per_drug_mean_inner_dev"),
        "linear_state_ridge": ("linear_state_residual_train", "linear_state_residual_inner_dev"),
        "nonlinear_state_residual": ("nonlinear_state_residual_train", "nonlinear_state_residual_inner_dev"),
        "capacity_matched_drug_only": ("capacity_matched_drug_only_train", "capacity_matched_drug_only_inner_dev"),
        "whole_object_state_shuffle": ("shuffled_state_train", "shuffled_state_inner_dev"),
    }
    predictions: dict[str, dict[str, np.ndarray]] = {}
    with np.load(archive_path, allow_pickle=False) as saved:
        fields = set(saved.files)
        for label, required in suffixes.items():
            missing = [name for name in required if name not in fields]
            if missing:
                raise KeyError(f"EXP-062 baseline archive lacks required fixed arms: {missing}")
            tr = np.asarray(saved[required[0]], dtype=np.float32)
            dv = np.asarray(saved[required[1]], dtype=np.float32)
            if tr.shape != train_y.shape or dv.shape != dev_y.shape or not np.isfinite(tr).all() or not np.isfinite(dv).all():
                raise ValueError(f"EXP-062 baseline shape/finiteness mismatch for {label}")
            predictions[label] = {"train": tr, "inner_dev": dv}
    # The global-mean comparator is reconstructed from the authorized TRAIN target only.
    train_mean = train_y.mean(axis=0, dtype=np.float64).astype(np.float32)
    predictions["global_train_mean"] = {
        "train": np.tile(train_mean, (n_train, 1)), "inner_dev": np.tile(train_mean, (n_dev, 1))}
    # Recompute the exact-key mean from this same frozen TRAIN target and ensure it matches the saved baseline.
    mean_by_key = {key: train_y[train_keys.full_inchikey.astype(str).to_numpy() == key].mean(axis=0, dtype=np.float64).astype(np.float32)
                   for key in sorted(train_keys.full_inchikey.astype(str).unique())}
    recomputed_train = np.stack([mean_by_key[str(k)] for k in train_keys.full_inchikey.astype(str)])
    recomputed_dev = np.stack([mean_by_key[str(k)] for k in dev_keys.full_inchikey.astype(str)])
    if not np.allclose(recomputed_train, predictions["per_drug_mean"]["train"], rtol=0, atol=2e-6):
        raise ValueError("saved EXP-062 per-drug TRAIN baseline differs from the frozen target recomputation")
    if not np.allclose(recomputed_dev, predictions["per_drug_mean"]["inner_dev"], rtol=0, atol=2e-6):
        raise ValueError("saved EXP-062 per-drug inner-dev baseline differs from the frozen target recomputation")

    metrics: dict[str, Any] = {}
    for name, by_role in predictions.items():
        metrics[name] = {
            "train": metric_fn(train_keys, train_y, by_role["train"], gene_variance),
            "inner_dev": metric_fn(dev_keys, dev_y, by_role["inner_dev"], gene_variance),
        }
    return {"source_archive_sha256": actual_sha, "metrics": metrics}, predictions


def nvidia_snapshot(uuid: str) -> dict[str, Any]:
    query = subprocess.run(["nvidia-smi", f"--id={uuid}", "--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True)
    rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"approved GPU UUID did not resolve to exactly one device: {rows}")
    apps = subprocess.run(["nvidia-smi", f"--id={uuid}", "--query-compute-apps=pid,process_name,gpu_uuid", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True)
    app_rows = [line.strip() for line in apps.stdout.splitlines() if line.strip() and "No running processes" not in line]
    fields = [x.strip() for x in rows[0].split(",")]
    if len(fields) != 5 or fields[0] != uuid or "RTX A5000" not in fields[1]:
        raise RuntimeError(f"approved A5000 identity mismatch: {rows[0]}")
    if int(fields[2]) != 0 or int(fields[3]) > 128 or app_rows:
        raise RuntimeError(f"approved A5000 is not idle at launch: {rows[0]} apps={app_rows}")
    return {"row": rows[0], "active_compute_processes": app_rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/EXP-063-xpert-coverage20.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--phase", choices=["fit"], required=True)
    args = parser.parse_args()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    start_cpu = time.perf_counter()
    config_path = REPO / args.config
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.run_id != cfg["run_id"] or args.run_id != "coverage20_official_v1" or args.attempt_id != "fit_attempt_003":
        raise ValueError("run/attempt identifier differs from the immutable EXP-063 contract")
    run_root = REPO / cfg["output"]["root"]
    attempt = run_root / args.attempt_id
    if attempt.exists():
        raise FileExistsError(f"immutable EXP-063 attempt exists; refusing overwrite: {attempt}")
    attempt.mkdir(parents=True, exist_ok=False)
    manifest_path = attempt / "run_manifest.json"
    fit_started = None
    gpu_started = None
    try:
        gate_path = run_root / "preflight/metadata_axis_v6/xpert_native_axis_metadata_gate.json"
        if not gate_path.is_file():
            raise FileNotFoundError("root-blind XPert native feature-axis metadata gate is missing")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if gate.get("status") != "PASS_RESPONSE_BLIND_XPERT_NATIVE_AXIS_GATE":
            raise RuntimeError("metadata-axis gate is not PASS")
        if gate["data_access"]["h5ad_X_values_read"] or gate["data_access"]["h5ad_X_ctl_values_read"] or gate["data_access"]["exp062_prediction_values_read"]:
            raise RuntimeError("metadata-axis gate unexpectedly accessed numeric values")
        gate_sha = sha256_file(gate_path)
        prior_cpu_seconds = float(gate["resources"]["cumulative_cpu_seconds"])
        if prior_cpu_seconds > float(cfg["resources"]["cpu_hard_cap_seconds"]):
            raise TimeoutError("response-blind gate plus prior work already exceeds EXP-063 CPU cap")
        if gate["asset_sha256"].get("exp063_runner_script") != sha256_file(Path(__file__).resolve()):
            raise RuntimeError("XPert fit runner changed after the response-blind metadata gate")
        helper_path = REPO / "scripts/modeling/exp051_contract_fit_smoke.py"
        if gate["asset_sha256"].get("exp051_adapter_script") != sha256_file(helper_path):
            raise RuntimeError("EXP-051 XPert adapter changed after the response-blind metadata gate")
        metadata_manifest = SOURCE / "metadata_freeze_manifest.json"
        metadata_root = SOURCE / "root_metadata_readback.json"
        numeric_root = SOURCE / "root_independent_readback_fit_attempt_002_v5/root_independent_readback.json"
        values_manifest = SOURCE / "fit_attempt_002/run_manifest.json"
        prediction_path = SOURCE / "fit_attempt_002/keyed_predictions.npz"
        keys_path = SOURCE / "response_blind_conditions.tsv"
        expected = cfg["data"]
        for path, expected_sha, label in (
                (metadata_manifest, expected["metadata_freeze_sha256"], "metadata freeze"),
                (metadata_root, expected["metadata_root_readback_sha256"], "metadata root receipt"),
                (numeric_root, expected["numeric_root_readback_sha256"], "numeric root receipt")):
            if sha256_file(path) != expected_sha:
                raise RuntimeError(f"frozen EXP-062 {label} hash mismatch")
        root_numeric = json.loads(numeric_root.read_text(encoding="utf-8"))
        source_manifest = json.loads(values_manifest.read_text(encoding="utf-8"))
        if (root_numeric.get("status") != "PASS_ROOT_INDEPENDENT_NUMERIC_READBACK"
                or root_numeric.get("forbidden_numeric_values_read") not in ([], None)
                or root_numeric.get("attempt_keyed_predictions_sha256") != source_manifest["outputs_sha256"]["keyed_predictions.npz"]
                or sha256_file(prediction_path) != root_numeric["attempt_keyed_predictions_sha256"]):
            raise RuntimeError("EXP-062 source prediction package is not bound to its PASS root readback")
        if sha256_file(keys_path) != json.loads(metadata_manifest.read_text(encoding="utf-8"))["frozen_files_sha256"]["response_blind_conditions.tsv"]:
            raise RuntimeError("frozen EXP-062 condition key table changed")

        # Reconfirm XPert source-code and feature bytes against the response-blind gate.
        ghash = gate["asset_sha256"]
        files_to_check = {
            "xpert_model": XPERT / "models/model_XPert.py",
            "xpert_train": XPERT / "train_xpert.py",
            "xpert_dataset": XPERT / "datasets/MyDataset.py",
            "xpert_utils": XPERT / "utils.py",
            "xpert_config": XPERT / "configs/config_l1000.yaml",
            "xpert_drug_table": XPERT / "processed_data/l1000_sdst_drugs_info_8432.csv",
            "exp051_gene_axis_mapping": EXP051 / "gene_axis_mapping.tsv",
            "unimol": XPERT / "processed_data/all_drugs_unimol_arr.npy",
            "hg_embeddings": XPERT / "HG_data/saved_embedding/HG_drug_embeddings.npy",
            "ppi_gene_vectors": XPERT / "processed_data/PPI_gene_vector_128d.npy",
        }
        for label, path in files_to_check.items():
            key = f"feature_asset:{label}" if label in {"unimol", "hg_embeddings", "ppi_gene_vectors"} else label
            if sha256_file(path) != ghash[key]:
                raise RuntimeError(f"XPert source input changed after metadata-axis gate: {label}")
        if sha256_file(config_path) != ghash["exp063_config"]:
            raise RuntimeError("EXP-063 configuration changed after metadata-axis gate")

        keys = pd.read_csv(keys_path, sep="\t", dtype=str, keep_default_na=False)
        keys["source_row"] = pd.to_numeric(keys["source_row"], errors="raise").astype(np.int64)
        keys["pert_idx"] = pd.to_numeric(keys["pert_idx"], errors="raise").astype(np.int64)
        keys["drug_node_idx"] = keys["pert_idx"].astype(np.int64)
        keys["pert_dose_num"] = pd.to_numeric(keys["pert_dose"], errors="raise").astype(float)
        keys["pert_time_num"] = pd.to_numeric(keys["pert_time"], errors="raise").astype(float)
        role = keys["effective_split_role"].astype(str)
        train_mask, dev_mask = role.eq("train").to_numpy(), role.eq("inner_dev").to_numpy()
        train_keys, dev_keys = keys.loc[train_mask].reset_index(drop=True), keys.loc[dev_mask].reset_index(drop=True)
        if (len(keys) != 1982 or int(train_mask.sum()) != 1291 or int(dev_mask.sum()) != 691
                or train_keys.source_object_id.nunique() != 93 or dev_keys.source_object_id.nunique() != 48
                or set(train_keys.source_object_id) & set(dev_keys.source_object_id)):
            raise ValueError("frozen EXP-062 object split/denominator changed")
        train_objects = set(train_keys.source_object_id.astype(str))
        dev_objects = set(dev_keys.source_object_id.astype(str))
        if len(set(keys.full_inchikey.astype(str))) != 20 or keys["source_row"].duplicated().any():
            raise ValueError("frozen condition keys or source-row domain changed")

        # Numeric access is restricted to the 1982 rows frozen by EXP-062 Stage M.
        data_started = time.perf_counter()
        values, rows, value_receipt = load_selected_values(
            keys, Path(expected["source_h5ad"]), int(expected["genes"]))
        if sha256_file(config_path) != ghash["exp063_config"]:
            raise RuntimeError("EXP-063 config hash changed during preparation")
        gene_axis_from_archive = None
        with np.load(prediction_path, allow_pickle=False) as archive:
            if "gene_axis" not in archive.files:
                raise KeyError("EXP-062 root-verified numeric archive lacks the frozen gene axis")
            gene_axis_from_archive = archive["gene_axis"].astype(str)
        if not np.array_equal(values["genes"].astype(str), gene_axis_from_archive.astype(str)):
            raise ValueError("H5AD and root-verified EXP-062 gene axes differ in order")
        values["genes"] = gene_axis_from_archive
        data_cpu_seconds = time.perf_counter() - data_started
        gene_variance = np.var(values["delta"][train_mask].astype(np.float64), axis=0, ddof=0)
        baseline, baseline_predictions = load_baselines(
            keys, values, train_mask, dev_mask, gene_variance, prediction_path,
            root_numeric["attempt_keyed_predictions_sha256"],
            __import__("exp051_contract_fit_smoke").keyed_metrics)
        baseline_path = attempt / "baseline_metrics.json"
        atomic_json(baseline_path, baseline)
        baseline_predictions["exact_key_train_mean"] = baseline_predictions.pop("per_drug_mean")

        # Approved-device identity and idle check occurs immediately before CUDA initialization.
        approved_uuid = cfg["resources"]["allowed_gpu_uuid"]
        device_snapshot = nvidia_snapshot(approved_uuid)
        gpu_started = time.perf_counter()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible != approved_uuid:
            raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be the single approved UUID, got {visible!r}")
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("the approved single A5000 is not the sole visible CUDA device")
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        if "A5000" not in gpu_name:
            raise RuntimeError(f"visible GPU is not the approved A5000: {gpu_name}")
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

        helper = __import__("exp051_contract_fit_smoke")
        model_module, xp_cfg, logger, _ = helper._training_modules(torch)
        xp_cfg["dataset"]["drug_unimol_path"] = str(XPERT / "processed_data/all_drugs_unimol_arr.npy")
        xp_cfg["model"]["ATTN"]["ppi_gene_vector_path"] = str(XPERT / "processed_data/PPI_gene_vector_128d.npy")
        xp_cfg["model"]["HG"]["drug_hg_pretrained_embed_path"] = str(XPERT / "HG_data/saved_embedding/HG_drug_embeddings.npy")
        train_cfg = xp_cfg["train"]
        if (int(train_cfg["batch_size"]) != 128 or int(train_cfg["num_epochs"]) != 2500
                or int(train_cfg["init_epoch"]) != 70 or int(train_cfg["patience"]) != 50
                or abs(float(train_cfg["train_lr"]) - 0.004) > 1e-12
                or abs(float(train_cfg["weight_decay"]) - 1e-5) > 1e-12
                or list(map(float, train_cfg["loss_weight"])) != [0.2, 0.003, 0.2, 1.0]):
            raise RuntimeError("official XPert config deviates from the frozen full schedule/objective")
        unimol = np.load(XPERT / "processed_data/all_drugs_unimol_arr.npy", mmap_mode="r", allow_pickle=False)
        if unimol.shape != (8981, 122, 514):
            raise ValueError(f"XPert UniMol feature header changed: {unimol.shape}")

        source_train, _, pcc_loss_func, source_receipt = helper._source_functions(
            XPERT / "train_xpert.py", XPERT / "utils.py")
        seed = int(cfg["model"]["seed"])
        model, train_view = helper._new_xpert(torch, model_module, xp_cfg, device, seed)
        opt_raw = torch.optim.Adam(model.parameters(), lr=float(train_cfg["train_lr"]),
                                   weight_decay=float(train_cfg["weight_decay"]))
        def progress_check() -> None:
            for name, param in model.named_parameters():
                if not bool(torch.isfinite(param).all().item()):
                    raise FloatingPointError(f"non-finite XPert parameter before optimizer step: {name}")
        opt = helper.ProbeOptimizer(opt_raw, model, torch, progress_check=progress_check)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt_raw, lr_lambda=lambda epoch: 1.0 if epoch < 40 else 0.5)
        train_positions = np.flatnonzero(train_mask)
        dev_positions = np.flatnonzero(dev_mask)
        tensors = helper._xpert_tensors(torch, values, keys, train_positions, unimol, xp_cfg, training_targets=True)
        from torch.utils.data import DataLoader, TensorDataset
        loader_generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(TensorDataset(*tensors), batch_size=128, shuffle=True, drop_last=False,
                            num_workers=0, generator=loader_generator)
        initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        initial_optimizer = helper._cpu_copy_tree(torch, opt_raw.state_dict())
        atomic_torch_save(torch, {"model_state_dict": initial_state, "optimizer_state_dict": initial_optimizer,
                                  "epoch_one_based": 0, "seed": seed, "checkpoint_type": "fresh_initialization",
                                  "torch_rng_state": torch.get_rng_state().cpu(),
                                  "loader_rng_state": loader_generator.get_state().cpu()}, attempt / "initial_checkpoint.pt")

        curve_path = attempt / "training_curve.jsonl"
        projection_path = attempt / "first_epoch_cost_gate.json"
        run_manifest = {
            "experiment_id": "EXP-063", "run_id": args.run_id, "attempt_id": args.attempt_id,
            "status": "RUNNING_FIXED_XPERT_OFFICIAL_SCHEDULE", "device": gpu_name,
            "allowed_gpu_uuid": approved_uuid, "cuda_visible_devices": visible,
            "device_snapshot_at_launch": device_snapshot, "pid": os.getpid(),
            "seed": seed, "source_receipt_sha256": sha256_file(gate_path),
            "source_exp062_numeric_readback_sha256": expected["numeric_root_readback_sha256"],
            "source_prediction_archive_sha256": root_numeric["attempt_keyed_predictions_sha256"],
            "selected_source_rows_read": 1982, "forbidden_numeric_values_opened": [],
            "model": "upstream XPertNet; fresh init_weights; UniMol atom features + HG embeddings",
            "upstream_train_function_sha256": source_receipt["train_function_source_sha256"],
            "upstream_loss_function_sha256": source_receipt["loss_functions_source_sha256"],
            "upstream_config_sha256": sha256_file(XPERT / "configs/config_l1000.yaml"),
            "config_sha256": sha256_file(config_path), "runner_sha256": sha256_file(Path(__file__).resolve()),
            "data_receipt": value_receipt,
            "schedule": {"batch_size": 128, "learning_rate": 0.004, "weight_decay": 1e-5,
                         "loss_weights": [0.2, 0.003, 0.2, 1], "max_epochs": 2500,
                         "transition_epoch_index": 70, "author_patience": 50,
                         "author_selection": "minimum inner-dev val_loss4 only"},
            "resources": {"cpu_prior_conservative_seconds": prior_cpu_seconds,
                          "cpu_preparation_seconds": data_cpu_seconds,
                          "cpu_cap_seconds": int(cfg["resources"]["cpu_hard_cap_seconds"]),
                          "global_gpu_settled_baseline_hours": GLOBAL_GPU_SETTLED_HOURS,
                          "package_gpu_cap_hours": float(cfg["resources"]["package_gpu_cap_hours"]),
                          "single_attempt_gpu_cap_hours": float(cfg["resources"]["single_run_gpu_cap_hours"]),
                          "global_gpu_remaining_hours": 48.0 - GLOBAL_GPU_SETTLED_HOURS,
                          "effective_gpu_cap_hours": min(float(cfg["resources"]["package_gpu_cap_hours"]),
                                                          float(cfg["resources"]["single_run_gpu_cap_hours"]),
                                                          48.0 - GLOBAL_GPU_SETTLED_HOURS),
                          "gpu_hours": 0.0},
        }
        atomic_json(manifest_path, run_manifest)
        atomic_json(attempt / "cpu_resource_ledger.json", {
            "prior_package_seconds_conservatively_charged": prior_cpu_seconds,
            "metadata_axis_gate_v1_failed_cpu_seconds": 5.0,
            "metadata_axis_gate_v2_elapsed_seconds": next(
                (item["elapsed_cpu_wall_seconds"] for item in gate.get("predecessor_gates", []) if item.get("gate_id") == "metadata_axis_v2"), None),
            "metadata_axis_gate_v3_elapsed_seconds": gate["resources"]["elapsed_cpu_wall_seconds"],
            "metadata_axis_gate_v4_elapsed_seconds": next(
                (item["elapsed_cpu_wall_seconds"] for item in gate.get("predecessor_gates", []) if item.get("gate_id") == "metadata_axis_v4"), None),
            "failed_fit_attempt_001_seconds_conservatively_charged": 15.0,
            "cpu_data_preparation_seconds": data_cpu_seconds,
            "current_cpu_cumulative_seconds": prior_cpu_seconds + data_cpu_seconds,
            "cpu_hard_cap_seconds": 900, "gpu_hours": 0.0})
        if prior_cpu_seconds + data_cpu_seconds > 900:
            raise TimeoutError("EXP-063 CPU prep/readback hard cap exceeded before training")

        # Charge all CPU-only preparation before deciding whether any optimizer update may run.
        cpu_pre_fit_wall_seconds = time.perf_counter() - start_cpu
        cpu_cumulative_pre_fit = prior_cpu_seconds + cpu_pre_fit_wall_seconds
        cpu_ledger = {
            "prior_package_seconds_conservatively_charged": prior_cpu_seconds,
            "metadata_axis_gate_v1_failed_cpu_seconds": 5.0,
            "metadata_axis_gate_v2_elapsed_seconds": next(
                (item["elapsed_cpu_wall_seconds"] for item in gate.get("predecessor_gates", []) if item.get("gate_id") == "metadata_axis_v2"), None),
            "metadata_axis_gate_v3_elapsed_seconds": next(
                (item["elapsed_cpu_wall_seconds"] for item in gate.get("predecessor_gates", []) if item.get("gate_id") == "metadata_axis_v3"), None),
            "metadata_axis_gate_v4_elapsed_seconds": next(
                (item["elapsed_cpu_wall_seconds"] for item in gate.get("predecessor_gates", []) if item.get("gate_id") == "metadata_axis_v4"), None),
            "metadata_axis_gate_v5_elapsed_seconds": next(
                (item["elapsed_cpu_wall_seconds"] for item in gate.get("predecessor_gates", []) if item.get("gate_id") == "metadata_axis_v5"), None),
            "metadata_axis_gate_v6_elapsed_seconds": gate["resources"]["elapsed_cpu_wall_seconds"],
            "failed_fit_attempt_001_seconds_conservatively_charged": 15.0,
            "failed_fit_attempt_002_seconds_conservatively_charged": 50.0,
            "cpu_pre_fit_wall_seconds": cpu_pre_fit_wall_seconds,
            "current_cpu_cumulative_seconds": cpu_cumulative_pre_fit,
            "cpu_hard_cap_seconds": int(cfg["resources"]["cpu_hard_cap_seconds"]),
            "gpu_hours": 0.0,
        }
        atomic_json(attempt / "cpu_resource_ledger.json", cpu_ledger)
        if cpu_cumulative_pre_fit > float(cfg["resources"]["cpu_hard_cap_seconds"]):
            raise TimeoutError("EXP-063 CPU preparation hard cap exceeded before training updates")

        # One complete real epoch includes training, full TRAIN/inner-dev evaluation, curve, checkpoint and prediction IO.
        fit_started = time.perf_counter()
        loss_start = time.perf_counter()
        grad_start = len(opt.gradient_norms)
        class Args:
            pretrained_model = False
        losses = source_train(train_view, opt, loader, Args(), xp_cfg, scaler=None, epoch=0)
        torch.cuda.synchronize()
        if not losses or any(not math.isfinite(float(value)) for value in losses):
            raise FloatingPointError("non-finite XPert raw training loss in epoch 1")
        state_ok, bad_state = finite_state(torch, model.state_dict())
        optim_copy = helper._cpu_copy_tree(torch, opt_raw.state_dict())
        optimizer_ok, bad_optimizer = finite_optimizer(torch, optim_copy)
        if not state_ok or not optimizer_ok:
            raise FloatingPointError(f"non-finite model/optimizer after epoch 1: {bad_state} {bad_optimizer}")
        train_pred = helper._xpert_predict(torch, model, values, keys, train_positions, unimol, xp_cfg, batch_size=64)
        dev_pred = helper._xpert_predict(torch, model, values, keys, dev_positions, unimol, xp_cfg, batch_size=64)
        if not np.isfinite(train_pred).all() or not np.isfinite(dev_pred).all():
            raise FloatingPointError("non-finite XPert prediction after epoch 1")
        train_y, dev_y = values["delta"][train_mask], values["delta"][dev_mask]
        train_metrics = helper.keyed_metrics(train_keys, train_y, train_pred, gene_variance)
        dev_metrics = helper.keyed_metrics(dev_keys, dev_y, dev_pred, gene_variance)
        if not math.isfinite(float(train_metrics["object_macro_nmse"])) or not math.isfinite(float(dev_metrics["object_macro_nmse"])):
            raise FloatingPointError("non-finite XPert epoch-1 metric")
        val_loss4 = helper._xpert_official_selection_loss(torch, pcc_loss_func, dev_pred, dev_y, device)
        epoch_seconds = time.perf_counter() - loss_start
        gradients = opt.gradient_norms[grad_start:]
        if not gradients or not all(math.isfinite(float(x)) for x in gradients):
            raise FloatingPointError("missing/non-finite XPert epoch-1 gradients")
        epoch_row = {"epoch": 1, "epoch_index_zero_based": 0,
                     "schedule_stage": "per_sample_mean_root_mse", "learning_rate": float(opt_raw.param_groups[0]["lr"]),
                     "official_train_loss_terms": [float(v) for v in losses],
                     "official_inner_dev_selection": "val_loss4_mean_delta_pcc_loss",
                     "inner_dev_val_loss4": val_loss4,
                     "gradient_norm_mean": float(np.mean(gradients)), "gradient_norm_max": float(np.max(gradients)),
                     "gradient_norm_n_updates": len(gradients), "train_metrics": train_metrics,
                     "inner_dev_metrics": dev_metrics, "epoch_seconds_including_full_eval": epoch_seconds}
        curve_path.write_text(json.dumps(epoch_row, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        best_optimizer = helper._cpu_copy_tree(torch, opt_raw.state_dict())
        best_payload = {"model_state_dict": best_state, "optimizer_state_dict": best_optimizer,
                        "epoch_one_based": 1, "seed": seed, "selection_metric": "minimum inner-dev val_loss4",
                        "selected_val_loss4": val_loss4,
                        "official_train_function_sha256": source_receipt["train_function_source_sha256"],
                        "torch_rng_state": torch.get_rng_state().cpu(),
                        "loader_rng_state": loader_generator.get_state().cpu()}
        atomic_torch_save(torch, best_payload, attempt / "XPert_best_inner_dev.pt")
        atomic_torch_save(torch, best_payload, attempt / "XPert_epoch001_checkpoint.pt")
        atomic_npz(attempt / "epoch001_keyed_predictions.npz",
                   train_condition_key=train_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
                   inner_dev_condition_key=dev_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
                   train_source_object_id=train_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
                   inner_dev_source_object_id=dev_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
                   train_full_inchikey=train_keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
                   inner_dev_full_inchikey=dev_keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
                   genes=values["genes"].astype(str), train_target_delta=train_y,
                   inner_dev_target_delta=dev_y, XPert_epoch001_train_delta=train_pred,
                   XPert_epoch001_inner_dev_delta=dev_pred)
        # Read checkpoint/prediction headers and verify exact epoch-one state/prediction roundtrip.
        loaded = torch.load(attempt / "XPert_epoch001_checkpoint.pt", map_location="cpu", weights_only=True)
        if set(loaded["model_state_dict"]) != set(best_state):
            raise RuntimeError("epoch-1 checkpoint state keys failed readback")
        for name, value in best_state.items():
            if not torch.equal(loaded["model_state_dict"][name], value):
                raise RuntimeError(f"epoch-1 checkpoint state failed exact readback: {name}")
        with np.load(attempt / "epoch001_keyed_predictions.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["XPert_epoch001_inner_dev_delta"], dev_pred):
                raise RuntimeError("epoch-1 keyed prediction failed exact archive readback")
        torch.cuda.synchronize()
        first_epoch_total = time.perf_counter() - fit_started
        post_overhead_reserve = 180.0
        max_epochs = int(train_cfg["num_epochs"])
        setup_gpu_seconds = fit_started - float(gpu_started)
        projection_without_margin = setup_gpu_seconds + max_epochs * first_epoch_total + post_overhead_reserve
        projection_with_margin = projection_without_margin * (1.0 + float(cfg["resources"]["first_epoch_full_package_projection_margin"]))
        effective_cap = min(float(cfg["resources"]["package_gpu_cap_hours"]),
                            float(cfg["resources"]["single_run_gpu_cap_hours"]),
                            48.0 - GLOBAL_GPU_SETTLED_HOURS)
        cost_pass = projection_with_margin / 3600.0 <= effective_cap
        gate_result = {
            "status": "PASS_CONTINUE_SAME_ATTEMPT" if cost_pass else "FAIL_CLOSED_SEAL_EPOCH1_COST_GATE",
            "epoch1_total_seconds_including_eval_checkpoint_and_prediction_io": first_epoch_total,
            "gpu_setup_seconds_before_epoch1": setup_gpu_seconds,
            "full_2500_epoch_projection_without_margin_seconds": projection_without_margin,
            "post_fit_eval_save_readback_reserve_seconds": post_overhead_reserve,
            "projection_margin_fraction": float(cfg["resources"]["first_epoch_full_package_projection_margin"]),
            "projection_with_margin_gpu_hours": projection_with_margin / 3600.0,
            "effective_gpu_cap_hours": effective_cap,
            "package_cap_hours": float(cfg["resources"]["package_gpu_cap_hours"]),
            "single_run_cap_hours": float(cfg["resources"]["single_run_gpu_cap_hours"]),
            "global_gpu_settled_baseline_hours": GLOBAL_GPU_SETTLED_HOURS,
            "global_gpu_remaining_hours": 48.0 - GLOBAL_GPU_SETTLED_HOURS,
            "inner_dev_access": "one full frozen evaluation per epoch for author val_loss4 selection; no project-NMSE-based selection",
            "epoch1_train_nmse": train_metrics["object_macro_nmse"],
            "epoch1_inner_dev_nmse_descriptive_only": dev_metrics["object_macro_nmse"],
            "epoch1_inner_dev_val_loss4": val_loss4,
            "selected_checkpoint_epoch": 1,
            "selected_rows_only": True,
            "forbidden_numeric_values_read": [],
        }
        atomic_json(projection_path, gate_result)
        atomic_json(attempt / "baseline_metrics.json", baseline)
        atomic_json(attempt / "training_progress.json", {"status": gate_result["status"], "completed_epochs": 1,
                                                          "max_epochs": 2500, "best_epoch_by_val_loss4": 1,
                                                          "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        run_manifest["status"] = gate_result["status"]
        run_manifest["epoch1_completed_seconds"] = first_epoch_total
        run_manifest["epoch1_cost_gate"] = gate_result
        run_manifest["gpu_hours_after_epoch1"] = (time.perf_counter() - float(gpu_started)) / 3600.0
        run_manifest["output_sha256_after_epoch1"] = {
            name: sha256_file(attempt / name) for name in ["training_curve.jsonl", "XPert_best_inner_dev.pt",
                                                          "XPert_epoch001_checkpoint.pt", "epoch001_keyed_predictions.npz",
                                                          "first_epoch_cost_gate.json", "baseline_metrics.json"]}
        atomic_json(manifest_path, run_manifest)
        print(json.dumps({"stage": "EXP063_FIRST_EPOCH_COST_GATE", "status": gate_result["status"],
                          "epoch1_seconds": first_epoch_total, "projection_gpu_hours": projection_with_margin / 3600.0,
                          "cap_gpu_hours": effective_cap, "pid": os.getpid(), "receipt": str(projection_path)}, ensure_ascii=False), flush=True)
        if not cost_pass:
            return 0

        # Continue the same initialized model/optimizer/loader only if the full maximum schedule fits the package cap.
        # The upstream loop advances LambdaLR after validation and before the next training epoch.
        scheduler.step()
        patience = int(train_cfg["patience"])
        epochs_since_best = 0
        best_val_loss4 = val_loss4
        best_epoch = 1
        best_train, best_dev = train_pred.copy(), dev_pred.copy()
        epoch_durations = [first_epoch_total]
        completed_epochs = 1
        for epoch in range(1, max_epochs):
            epoch_start = time.perf_counter()
            grad_start = len(opt.gradient_norms)
            learning_rate = float(opt_raw.param_groups[0]["lr"])
            losses = source_train(train_view, opt, loader, Args(), xp_cfg, scaler=None, epoch=epoch)
            torch.cuda.synchronize()
            if not losses or any(not math.isfinite(float(value)) for value in losses):
                raise FloatingPointError(f"non-finite XPert raw training loss at epoch {epoch + 1}")
            state_ok, bad_state = finite_state(torch, model.state_dict())
            optimizer_state = helper._cpu_copy_tree(torch, opt_raw.state_dict())
            optimizer_ok, bad_optimizer = finite_optimizer(torch, optimizer_state)
            if not state_ok or not optimizer_ok:
                raise FloatingPointError(f"non-finite model/optimizer at epoch {epoch + 1}: {bad_state} {bad_optimizer}")
            train_pred = helper._xpert_predict(torch, model, values, keys, train_positions, unimol, xp_cfg, batch_size=64)
            dev_pred = helper._xpert_predict(torch, model, values, keys, dev_positions, unimol, xp_cfg, batch_size=64)
            if not np.isfinite(train_pred).all() or not np.isfinite(dev_pred).all():
                raise FloatingPointError(f"non-finite XPert prediction at epoch {epoch + 1}")
            train_metrics = helper.keyed_metrics(train_keys, train_y, train_pred, gene_variance)
            dev_metrics = helper.keyed_metrics(dev_keys, dev_y, dev_pred, gene_variance)
            val_loss4 = helper._xpert_official_selection_loss(torch, pcc_loss_func, dev_pred, dev_y, device)
            gradients = opt.gradient_norms[grad_start:]
            if not gradients or any(not math.isfinite(float(x)) for x in gradients):
                raise FloatingPointError(f"non-finite/missing gradient at epoch {epoch + 1}")
            duration = time.perf_counter() - epoch_start
            epoch_durations.append(duration)
            completed_epochs = epoch + 1
            row = {"epoch": completed_epochs, "epoch_index_zero_based": epoch,
                   "schedule_stage": "per_sample_mean_root_mse" if epoch < int(train_cfg["init_epoch"]) else "raw_weighted_sum",
                   "learning_rate": learning_rate, "official_train_loss_terms": [float(v) for v in losses],
                   "official_inner_dev_selection": "val_loss4_mean_delta_pcc_loss", "inner_dev_val_loss4": val_loss4,
                   "gradient_norm_mean": float(np.mean(gradients)), "gradient_norm_max": float(np.max(gradients)),
                   "gradient_norm_n_updates": len(gradients), "train_metrics": train_metrics,
                   "inner_dev_metrics": dev_metrics, "epoch_seconds_including_full_eval": duration}
            with curve_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            if val_loss4 < best_val_loss4:
                best_val_loss4 = val_loss4
                best_epoch = completed_epochs
                best_train, best_dev = train_pred.copy(), dev_pred.copy()
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                best_optimizer = helper._cpu_copy_tree(torch, opt_raw.state_dict())
                best_payload = {"model_state_dict": best_state, "optimizer_state_dict": best_optimizer,
                                "epoch_one_based": best_epoch, "seed": seed,
                                "selection_metric": "minimum inner-dev val_loss4", "selected_val_loss4": best_val_loss4,
                                "official_train_function_sha256": source_receipt["train_function_source_sha256"],
                                "torch_rng_state": torch.get_rng_state().cpu(),
                                "loader_rng_state": loader_generator.get_state().cpu()}
                atomic_torch_save(torch, best_payload, attempt / "XPert_best_inner_dev.pt")
                epochs_since_best = 0
            else:
                epochs_since_best += 1
            elapsed_gpu_like = time.perf_counter() - float(gpu_started)
            if elapsed_gpu_like / 3600 >= effective_cap:
                raise TimeoutError("EXP-063 package GPU budget reached")
            atomic_json(attempt / "training_progress.json", {
                "status": "RUNNING", "completed_epochs": completed_epochs, "max_epochs": max_epochs,
                "best_epoch_by_official_val_loss4": best_epoch, "best_val_loss4": best_val_loss4,
                "elapsed_gpu_hours_conservative": elapsed_gpu_like / 3600,
                "epoch_seconds": duration,
                "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            if completed_epochs == 10 or completed_epochs % 50 == 0:
                print(json.dumps({"stage": "EXP063_TRAIN_PROGRESS", "epoch": completed_epochs,
                                  "best_epoch": best_epoch, "best_val_loss4": best_val_loss4,
                                  "epoch_seconds": duration, "elapsed_gpu_hours": elapsed_gpu_like / 3600}, ensure_ascii=False), flush=True)
            scheduler.step()
            if epochs_since_best >= patience:
                break

        # Freeze by the author's val_loss4 checkpoint, restore and create keyed outputs.
        best_loaded = torch.load(attempt / "XPert_best_inner_dev.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(best_loaded["model_state_dict"])
        best_train_replay = helper._xpert_predict(torch, model, values, keys, train_positions, unimol, xp_cfg, batch_size=64)
        best_dev_replay = helper._xpert_predict(torch, model, values, keys, dev_positions, unimol, xp_cfg, batch_size=64)
        torch.cuda.synchronize()
        if (not np.allclose(best_train_replay, best_train, rtol=1e-5, atol=1e-6)
                or not np.allclose(best_dev_replay, best_dev, rtol=1e-5, atol=1e-6)):
            raise RuntimeError("XPert best-checkpoint restore changed TRAIN/inner-dev predictions")
        best_train, best_dev = best_train_replay, best_dev_replay
        final_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        final_name = f"XPert_final_stop_epoch_{completed_epochs}.pt"
        atomic_torch_save(torch, {"model_state_dict": final_state, "epoch_one_based": completed_epochs,
                                  "seed": seed, "official_train_function_sha256": source_receipt["train_function_source_sha256"]}, attempt / final_name)
        atomic_npz(attempt / "full_keyed_predictions.npz",
                   train_condition_key=train_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
                   inner_dev_condition_key=dev_keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
                   train_source_object_id=train_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
                   inner_dev_source_object_id=dev_keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
                   train_full_inchikey=train_keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
                   inner_dev_full_inchikey=dev_keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
                   genes=values["genes"].astype(str), train_target_delta=train_y, inner_dev_target_delta=dev_y,
                   XPert_best_inner_dev_train_delta=best_train, XPert_best_inner_dev_inner_dev_delta=best_dev)

        selected_metrics = {
            "XPert_original_architecture": {
                "train": helper.keyed_metrics(train_keys, train_y, best_train, gene_variance),
                "inner_dev": helper.keyed_metrics(dev_keys, dev_y, best_dev, gene_variance),
                "official_inner_dev_val_loss4": best_val_loss4, "selected_epoch_by_val_loss4": best_epoch,
                "epochs_completed": completed_epochs, "max_epochs": max_epochs,
                "stopped_by_author_patience": epochs_since_best >= patience,
            }
        }
        object_metrics = []
        paired = []
        per_drug = []
        all_predictions = {"XPert_original_architecture": {"train": best_train, "inner_dev": best_dev}, **baseline_predictions}
        for method, by_split in all_predictions.items():
            for role_name, split_keys, split_y, pred in (
                    ("train", train_keys, train_y, by_split["train"]),
                    ("inner_dev", dev_keys, dev_y, by_split["inner_dev"])):
                metrics = helper.keyed_metrics(split_keys, split_y, pred, gene_variance)
                selected_metrics.setdefault(method, {})[role_name] = metrics
                gene_use = np.isfinite(gene_variance) & (gene_variance > 1e-12)
                err = pred.astype(np.float64) - split_y.astype(np.float64)
                row_nmse = np.mean(np.square(err[:, gene_use]) / gene_variance[gene_use][None, :], axis=1)
                frame = pd.DataFrame({"source_object_id": split_keys.source_object_id.astype(str).to_numpy(),
                                      "full_inchikey": split_keys.full_inchikey.astype(str).to_numpy(),
                                      "nmse": row_nmse})
                obj = frame.groupby("source_object_id", sort=True)["nmse"].mean()
                for object_id, value in obj.items():
                    object_metrics.append({"method": method, "role": role_name, "source_object_id": object_id,
                                           "object_macro_nmse": float(value)})
                drug = frame.groupby("full_inchikey", sort=True)["nmse"].mean()
                for full_key, value in drug.items():
                    per_drug.append({"method": method, "role": role_name, "full_inchikey": full_key,
                                     "object_macro_nmse": float(value),
                                     "n_objects": int(frame.loc[frame.full_inchikey == full_key, "source_object_id"].nunique())})
        xpert_obj = pd.DataFrame([r for r in object_metrics if r["method"] == "XPert_original_architecture" and r["role"] == "inner_dev"])
        mean_obj = pd.DataFrame([r for r in object_metrics if r["method"] == "exact_key_train_mean" and r["role"] == "inner_dev"])
        joined = xpert_obj.merge(mean_obj, on="source_object_id", suffixes=("_xpert", "_mean"), validate="one_to_one")
        for row in joined.itertuples(index=False):
            paired.append({"source_object_id": row.source_object_id,
                           "xpert_object_macro_nmse": row.object_macro_nmse_xpert,
                           "exact_key_mean_object_macro_nmse": row.object_macro_nmse_mean,
                           "mean_minus_xpert_absolute_gain": row.object_macro_nmse_mean - row.object_macro_nmse_xpert})
        pd.DataFrame(object_metrics).to_csv(attempt / "per_object_metrics.tsv", sep="\t", index=False)
        pd.DataFrame(per_drug).to_csv(attempt / "per_drug_metrics.tsv", sep="\t", index=False)
        pd.DataFrame(paired).to_csv(attempt / "paired_object_differences.tsv", sep="\t", index=False)
        atomic_json(attempt / "metrics_summary.json", {
            "experiment_id": "EXP-063", "run_id": args.run_id, "attempt_id": args.attempt_id,
            "status": "COMPLETED_FIXED_SOURCE_DEVELOPMENT_XPERT_FIT",
            "denominators": {"conditions": 1982, "train_conditions": len(train_y), "train_objects": len(train_objects),
                             "inner_dev_conditions": len(dev_y), "inner_dev_objects": len(dev_objects),
                             "train_inner_dev_object_overlap": 0, "full_inchikeys": int(keys.full_inchikey.nunique()),
                             "genes": int(values["delta"].shape[1]), "nmse_genes": int(np.sum(gene_variance > 1e-12))},
            "selection": {"official_checkpoint_selection": "minimum XPert val_loss4 on inner-dev; patience=50",
                          "project_NMSE_not_used_for_selection": True, "best_epoch": best_epoch,
                          "best_official_val_loss4": best_val_loss4, "epochs_completed": completed_epochs},
            "metrics": selected_metrics,
            "paired_object_differences_vs_exact_key_mean": paired,
            "claim_boundary": "fixed XPert source-development panel only; no independent-drug or PDO efficacy claim",
            "forbidden_numeric_values_read": [],
        })
        elapsed_total = time.perf_counter() - float(gpu_started)
        atomic_json(attempt / "training_progress.json", {
            "status": "COMPLETED", "completed_epochs": completed_epochs, "max_epochs": max_epochs,
            "best_epoch_by_official_val_loss4": best_epoch, "best_official_val_loss4": best_val_loss4,
            "elapsed_gpu_hours_conservative": elapsed_total / 3600,
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        output_files = [p for p in attempt.iterdir() if p.is_file() and p.name != "run_manifest.json"]
        run_manifest["status"] = "COMPLETED_FIXED_SOURCE_DEVELOPMENT_XPERT_FIT"
        run_manifest["completed_epochs"] = completed_epochs
        run_manifest["best_epoch_by_official_val_loss4"] = best_epoch
        run_manifest["best_official_val_loss4"] = best_val_loss4
        run_manifest["actual_gpu_hours_conservative"] = elapsed_total / 3600
        run_manifest["output_sha256"] = {p.name: sha256_file(p) for p in sorted(output_files)}
        atomic_json(manifest_path, run_manifest)
        print(json.dumps({"stage": "EXP063_FIT_COMPLETE", "best_epoch": best_epoch,
                          "val_loss4": best_val_loss4, "inner_dev_nmse": selected_metrics["XPert_original_architecture"]["inner_dev"]["object_macro_nmse"],
                          "gpu_hours": elapsed_total / 3600, "output": str(attempt)}, ensure_ascii=False), flush=True)
        return 0
    except BaseException as exc:
        elapsed_gpu = 0.0 if gpu_started is None else time.perf_counter() - gpu_started
        failure = {"experiment_id": "EXP-063", "run_id": args.run_id, "attempt_id": args.attempt_id,
                   "status": "FAILED_CLOSED", "error_type": type(exc).__name__, "error": str(exc),
                   "pid": os.getpid(), "elapsed_gpu_hours_conservative": elapsed_gpu / 3600,
                   "forbidden_numeric_values_read": [],
                   "resources": {"gpu_hours": elapsed_gpu / 3600,
                                 "cpu_prior_conservative_seconds": float(locals().get("prior_cpu_seconds", 0.0)),
                                 "cpu_hard_cap_seconds": 900}}
        atomic_json(attempt / "failure_receipt.json", failure)
        if manifest_path.exists():
            try:
                old = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                old = {"experiment_id": "EXP-063", "attempt_id": args.attempt_id}
            old.update({"status": "FAILED_CLOSED", "failure_receipt": "failure_receipt.json",
                        "failure_type": type(exc).__name__, "failure": str(exc),
                        "actual_gpu_hours_conservative": elapsed_gpu / 3600})
            atomic_json(manifest_path, old)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
