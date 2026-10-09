"""One-shot, fixed-domain TranSiGen low-learning-rate stability candidate.

Only the learning rate differs from the source-recipe TRAIN run. The first 20
epochs are an in-run cost/stability gate; when it passes, training continues in
the same immutable attempt to epoch 300. Inner development is evaluated only
after the frozen TRAIN fit gate passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp051_contract_fit_smoke as fit  # noqa: E402

BASELINE_SETTLED_GPUH = 0.584830054434155
EXPECTED_PRIOR_SETTLED_GPUH = 1.223878605611392
RESERVE_REPAIR_ALLOCATION_GPUH = 5.415169945565845
TRAIN_BASELINE_NMSE = 1.044271
INNER_BASELINE_NMSE = 1.059270
TRAIN_EPOCHS = 300
SMOKE_EPOCHS = 20
TRAIN_ROWS = 419
TRAIN_OBJECTS = 110
INNER_ROWS = 93
INNER_OBJECTS = 24
N_GENES = 978
N_DRUGS = 5
LR = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 64
SEED = 20261006
GPU_UUID = "GPU-9c366c2e-bfa5-c7b3-b16a-1117c40e6871"
TOTAL_ATTEMPT_CAP_GPUH = 1.05
SMOKE_CAP_GPUH = 0.05
PROJECTED_FULL_CAP_GPUH = 1.0
SAFETY_LIMIT = 1e6
LOSS_NAMES = ("raw_total_loss", "mse_x1", "mse_x2", "mse_pert", "kld_x1", "kld_x2", "kld_pert")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2,
                                allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def sha256(path: Path) -> str:
    return fit.sha256_file(path)


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def receipt_string_field(path: Path, field: str) -> str:
    """Extract one string receipt field without parsing unrelated response summaries."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r'"' + re.escape(field) + r'"\s*:\s*"([^"]+)"', text)
    if not match:
        raise ValueError(f"required receipt field {field!r} is missing from {path.name}")
    return match.group(1)


def read_role_expression(keys: pd.DataFrame, expected_genes: np.ndarray) -> dict[str, np.ndarray]:
    """Read only the requested role's source_row values from the frozen H5AD."""
    import anndata as ad
    import h5py

    rows = pd.to_numeric(keys.source_row, errors="raise").to_numpy(dtype=np.int64)
    if len(rows) != len(keys) or len(np.unique(rows)) != len(rows):
        raise ValueError("requested role contains duplicate or invalid H5AD source rows")
    order = np.argsort(rows, kind="mergesort")
    sorted_rows = rows[order]
    meta = ad.read_h5ad(fit.H5AD, backed="r")
    try:
        genes = meta.var_names.astype(str).to_numpy(dtype=str)
        if not np.array_equal(genes, expected_genes.astype(str)):
            raise ValueError("H5AD gene axis differs from the frozen 978-gene axis")
        source_names = meta.obs_names.astype(str).to_numpy(dtype=str)[rows]
        if not np.array_equal(source_names, keys.obs_name.astype(str).to_numpy(dtype=str)):
            raise ValueError("requested role source rows do not match frozen observation identities")
        selected_obs = meta.obs.iloc[rows]
        if not np.array_equal(selected_obs.cell_iname.astype(str).to_numpy(dtype=str),
                              keys.source_object_id.astype(str).to_numpy(dtype=str)):
            raise ValueError("requested role source rows do not match frozen source objects")
        if not np.array_equal(selected_obs.pert_id.astype(str).to_numpy(dtype=str),
                              keys.pert_id.astype(str).to_numpy(dtype=str)):
            raise ValueError("requested role source rows do not match frozen perturbation identities")
    finally:
        meta.file.close()

    # H5AD obsm is a dense matrix in this source. Direct h5py row indexing avoids
    # materializing the combined TRAIN+inner-dev array before the fit gate.
    with h5py.File(fit.H5AD, "r") as handle:
        x_sorted = np.asarray(handle["X"][sorted_rows, :], dtype=np.float32)
        ctl_sorted = np.asarray(handle["obsm"]["X_ctl"][sorted_rows, :], dtype=np.float32)
    x = np.empty((len(rows), N_GENES), dtype=np.float32)
    x_ctl = np.empty_like(x)
    x[order] = x_sorted
    x_ctl[order] = ctl_sorted
    delta = x - x_ctl
    if x.shape != (len(keys), N_GENES) or not np.isfinite(x).all() or not np.isfinite(x_ctl).all():
        raise ValueError("selected role expression values have invalid shape or non-finite entries")
    residual = float(np.max(np.abs(x.astype(np.float64) - x_ctl.astype(np.float64)
                                  - delta.astype(np.float64))))
    if residual > 1e-6:
        raise ValueError(f"selected role violates treated=control+delta: {residual}")
    return {"X": x, "X_ctl": x_ctl, "delta": delta,
            "max_abs_decomposition_error": np.asarray(residual, dtype=np.float64)}


def device_snapshot() -> dict[str, Any]:
    snapshot = fit._device_snapshot()
    gpu = snapshot["selected_gpu"]
    if gpu["uuid"] != GPU_UUID or gpu["memory_used_mib"] > 1024 or gpu["utilization_percent"] > 20:
        raise RuntimeError(f"approved A5000 is not idle enough: {gpu}")
    return snapshot


def read_frozen_source(source_run_id: str) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    fit.configure_source_run(source_run_id)
    source = fit.SOURCE_RUN_DIR
    needed = ("metadata_freeze_receipt.json", "gene_axis_receipt.json", "value_contract_receipt.json",
              "root_values_readback_receipt.json", "transigen_ecfp4_feature_audit_receipt.json",
              "response_blind_conditions.tsv", "frozen_train_inner_values.npz")
    for name in needed:
        if not (source / name).is_file():
            raise FileNotFoundError(f"missing approved frozen source input {name}")
    root_receipt_path = source / "root_values_readback_receipt.json"
    root_receipt_text = root_receipt_path.read_text(encoding="utf-8")
    if not re.search(r'"result"\s*:\s*"PASS"', root_receipt_text):
        raise ValueError("frozen source root value readback does not pass")
    axis = json.loads((source / "gene_axis_receipt.json").read_text(encoding="utf-8"))
    if axis.get("status") != "PASS_EXACT_978_AXIS":
        raise ValueError("frozen source root value/gene-axis receipts do not pass")
    key_path, values_path = source / "response_blind_conditions.tsv", source / "frozen_train_inner_values.npz"
    expected_value_sha = receipt_string_field(source / "value_contract_receipt.json",
                                              "frozen_train_inner_values_npz")
    if sha256(values_path) != expected_value_sha:
        raise ValueError("frozen TRAIN/inner value archive hash does not match its receipt")
    if receipt_string_field(root_receipt_path, "input_sha256") != expected_value_sha:
        raise ValueError("root value readback is not bound to the frozen source archive")
    keys = pd.read_csv(key_path, sep="\t", dtype=str, keep_default_na=False)
    axis_map_path = source / "gene_axis_mapping.tsv"
    axis_map = pd.read_csv(axis_map_path, sep="\t", dtype=str, keep_default_na=False)
    axis_sequence_sha = hashlib.sha256(("\n".join(axis_map.h5ad_var_annotation.astype(str)) + "\n")
                                       .encode("utf-8")).hexdigest()
    if axis_sequence_sha != axis.get("mapping_sha256"):
        raise ValueError("gene axis annotation sequence differs from the frozen axis receipt")
    if (len(axis_map) != N_GENES or axis_map.axis_index.astype(int).tolist() != list(range(N_GENES))
            or axis_map.h5ad_var_name.nunique() != N_GENES
            or axis_map.h5ad_var_annotation.nunique() != N_GENES):
        raise ValueError("frozen gene axis mapping is not exactly 978 unique, ordered genes")
    genes = axis_map.h5ad_var_name.astype(str).to_numpy(dtype=str)
    train_mask = keys.split_role.eq("train").to_numpy()
    inner_mask = keys.split_role.eq("inner_dev").to_numpy()
    train_keys = keys.loc[train_mask].reset_index(drop=True)
    inner_keys = keys.loc[inner_mask].reset_index(drop=True)
    if (len(train_keys) != TRAIN_ROWS or train_keys.source_object_id.nunique() != TRAIN_OBJECTS
            or len(inner_keys) != INNER_ROWS or inner_keys.source_object_id.nunique() != INNER_OBJECTS
            or set(train_keys.source_object_id) & set(inner_keys.source_object_id)
            or train_keys.full_inchikey.nunique() != N_DRUGS
            or len(genes) != N_GENES or len(set(genes.tolist())) != N_GENES):
        raise ValueError("frozen split, object, complete InChIKey, or 978-gene denominators changed")
    if not train_keys.full_condition_key.is_unique or not inner_keys.full_condition_key.is_unique:
        raise ValueError("frozen condition keys are not unique within each role")
    train_values = read_role_expression(train_keys, genes)
    values = {"X_train": train_values["X"], "X_ctl_train": train_values["X_ctl"],
              "delta_train": train_values["delta"], "genes": genes.copy()}
    residual = float(train_values["max_abs_decomposition_error"])
    train_keys.attrs["inner_keys"] = inner_keys
    train_keys.attrs["source_value_sha256"] = sha256(values_path)
    train_keys.attrs["inner_dev_values_read"] = False
    return train_keys, values, {"decomposition_max_abs_error": float(residual),
                                "source_run_id": source_run_id, "source_values_sha256": sha256(values_path),
                                "source_key_table_sha256": sha256(key_path),
                                "train_conditions": TRAIN_ROWS, "train_objects": TRAIN_OBJECTS,
                                "inner_dev_conditions": INNER_ROWS, "inner_dev_objects": INNER_OBJECTS,
                                "object_overlap": 0, "full_inchikeys": N_DRUGS, "genes": N_GENES,
                                "inner_dev_scored": False}


def per_object_nmse(keys: pd.DataFrame, truth: np.ndarray, pred: np.ndarray,
                    gene_variance: np.ndarray) -> dict[str, float]:
    use_gene = np.isfinite(gene_variance) & (gene_variance > 1e-12)
    error = (pred.astype(np.float64) - truth.astype(np.float64))[:, use_gene]
    normalized = np.square(error) / gene_variance[use_gene][None, :]
    row_scores = normalized.mean(axis=1)
    frame = pd.DataFrame({"object": keys.source_object_id.astype(str).to_numpy(), "nmse": row_scores})
    return {str(k): float(v) for k, v in frame.groupby("object", sort=True).nmse.mean().items()}


def baseline_metrics(train_keys: pd.DataFrame, train_delta: np.ndarray,
                     gene_variance: np.ndarray) -> dict[str, Any]:
    zero = np.zeros_like(train_delta)
    by_key = {str(key): train_delta[train_keys.full_inchikey.astype(str).to_numpy() == str(key)].mean(
        axis=0, dtype=np.float64).astype(np.float32)
        for key in sorted(train_keys.full_inchikey.astype(str).unique())}
    per_drug = np.stack([by_key[str(key)] for key in train_keys.full_inchikey.astype(str)], axis=0)
    metrics = {
        "zero_change": fit.keyed_metrics(train_keys, train_delta, zero, gene_variance),
        "per_drug_mean_delta": fit.keyed_metrics(train_keys, train_delta, per_drug, gene_variance),
    }
    return {"metrics": {name: {"train": value} for name, value in metrics.items()},
            "per_drug_mean_predictions": per_drug}


def max_bn_running_var(model: Any) -> float:
    values = [float(torch_value.detach().float().max().cpu())
              for name, torch_value in model.named_buffers()
              if "running_var" in name]
    return max(values, default=0.0)


def params_finite(torch: Any, model: Any) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


def object_improvement_count(keys: pd.DataFrame, truth: np.ndarray, pred: np.ndarray,
                             baseline_pred: np.ndarray, gene_variance: np.ndarray) -> tuple[int, dict[str, Any]]:
    candidate = per_object_nmse(keys, truth, pred, gene_variance)
    baseline = per_object_nmse(keys, truth, baseline_pred, gene_variance)
    if set(candidate) != set(baseline) or len(candidate) != INNER_OBJECTS:
        raise ValueError("inner-dev paired object denominator changed")
    rows = {key: {"candidate_nmse": candidate[key], "per_drug_mean_nmse": baseline[key],
                  "improved": candidate[key] < baseline[key]} for key in sorted(candidate)}
    return sum(row["improved"] for row in rows.values()), rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg.get("experiment_id") != "EXP-052" or args.run_id != cfg.get("run_id"):
        raise ValueError("config, experiment, and run identifiers do not match")
    if cfg["model"]["learning_rate"] != LR or cfg["model"]["epochs"] != TRAIN_EPOCHS:
        raise ValueError("only the frozen lr=1e-4, 300-epoch candidate is accepted")
    if args.source_run_id != cfg["data"]["source_run_id"]:
        raise ValueError("source run differs from the registered EXP-052 config")
    if args.attempt_id != "fit_attempt_001":
        raise ValueError("this is a one-shot package; only fit_attempt_001 is authorized")
    config_sha = sha256(config_path)
    package_cap = float(cfg["authorization"]["package_gpu_cap_hours"])
    if package_cap != TOTAL_ATTEMPT_CAP_GPUH:
        raise ValueError("candidate package cap differs from its registered one-shot limit")
    experiment_root = ROOT / "artifacts/experiments/EXP-051"
    prior = fit.resource_accounting(experiment_root, "20261006_exp051_representative_compare_v1",
                                    10.0, 48.0, BASELINE_SETTLED_GPUH)
    if abs(prior["exp_settled_gpu_hours"] - EXPECTED_PRIOR_SETTLED_GPUH) > 1e-8:
        raise RuntimeError("actual EXP-051 receipts differ from the registered prior settlement; stop for reconciliation")
    if prior["exp_artifact_ledger_settled_gpu_hours"] > RESERVE_REPAIR_ALLOCATION_GPUH:
        raise RuntimeError("prior authorized receipts exceed the repair reserve; no EXP-052 GPU may start")
    run_dir = ROOT / "artifacts/experiments/EXP-052" / args.run_id
    attempt = run_dir / args.attempt_id
    if run_dir.exists() or attempt.exists():
        raise FileExistsError("refusing to reuse any prior EXP-052 run directory")
    run_dir.mkdir(parents=True, exist_ok=False)
    process_id = os.getpid()
    pre_gpu = {"experiment_id": "EXP-052", "run_id": args.run_id, "attempt": args.attempt_id,
               "phase": "CPU_PREFLIGHT", "pid": process_id, "config_sha256": config_sha,
               "started_at_utc": fit.utc_now(), "reserved_gpu_hours": TOTAL_ATTEMPT_CAP_GPUH,
               "source_settlement_gpu_hours": prior["exp_settled_gpu_hours"],
               "source_run_id": args.source_run_id}
    atomic_json(run_dir / "pre_gpu_preflight.json", pre_gpu)
    ledger_path: Path | None = None
    gpu_start: float | None = None
    reservation_started = False
    first_abnormal: dict[str, Any] | None = None
    phase = "CPU_PREFLIGHT"
    try:
        if cfg.get("authorization", {}).get("registration_status") != "registered_before_run":
            raise ValueError("EXP-052 must be registered before execution")
        snapshot = device_snapshot()
        train_keys, values, denominator = read_frozen_source(args.source_run_id)
        inner_keys = train_keys.attrs.pop("inner_keys")
        inner_dev_values_read = train_keys.attrs.pop("inner_dev_values_read")
        full_keys = pd.read_csv(fit.SOURCE_RUN_DIR / "response_blind_conditions.tsv", sep="\t",
                                dtype=str, keep_default_na=False)
        train_mask = full_keys.split_role.eq("train").to_numpy()
        inner_mask = full_keys.split_role.eq("inner_dev").to_numpy()
        if (int(train_mask.sum()) != TRAIN_ROWS or int(inner_mask.sum()) != INNER_ROWS
                or not np.array_equal(full_keys.loc[train_mask, "full_condition_key"].astype(str).to_numpy(),
                                      train_keys.full_condition_key.astype(str).to_numpy())):
            raise ValueError("response-blind condition-role table differs from the frozen TRAIN inputs")
        if inner_dev_values_read:
            raise ValueError("inner-development numeric values were opened before the TRAIN gate")
        feature_by_key, feature_receipt = fit._load_audited_ecfp4(fit.SOURCE_RUN_DIR)
        train_features = fit._full_key_features(full_keys, np.flatnonzero(train_mask), feature_by_key)
        inner_features = fit._full_key_features(full_keys, np.flatnonzero(inner_mask), feature_by_key)
        if train_features.shape != (TRAIN_ROWS, 2048) or inner_features.shape != (INNER_ROWS, 2048):
            raise ValueError("approved full-key ECFP4 source features have the wrong shape")
        train_y = values["delta_train"]
        gene_variance = np.var(train_y.astype(np.float64), axis=0, ddof=0)
        baselines = baseline_metrics(train_keys, train_y, gene_variance)
        if abs(baselines["metrics"]["per_drug_mean_delta"]["train"]["object_macro_nmse"]
               - TRAIN_BASELINE_NMSE) > 0.02:
            raise ValueError("TRAIN per-drug baseline drifted from the approved EXP-051 comparator")
        source_hashes = fit._source_hashes()
        train_receipt_status = receipt_string_field(fit.SOURCE_RUN_DIR / "value_contract_receipt.json", "status")
        axis_receipt = json.loads((fit.SOURCE_RUN_DIR / "gene_axis_receipt.json").read_text(encoding="utf-8"))
        atomic_json(run_dir / "run_manifest.json", {
            **pre_gpu, "status": "PREFLIGHT_PASS", "runner_sha256": sha256(Path(__file__).resolve()),
            "source_model_and_reference_code_sha256": source_hashes,
            "source_value_receipt_status": train_receipt_status,
            "source_gene_axis_receipt_status": axis_receipt.get("status"),
            "feature_receipt_sha256": sha256(fit.SOURCE_RUN_DIR / "transigen_ecfp4_feature_audit_receipt.json"),
            "denominator": denominator,
            "train_inchikeys": sorted(train_keys.full_inchikey.astype(str).unique().tolist()),
            "training_contract": {"model": "upstream original TranSiGen, scratch initialization",
                                  "objective": "raw summed source objective; unchanged component weights",
                                  "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
                                  "batch_size": BATCH_SIZE, "shuffle": True,
                                  "max_epochs": TRAIN_EPOCHS, "gradient_clip": None,
                                  "batchnorm": "ordinary training updates only; no calibration",
                                  "one_changed_factor": "learning rate 1e-3 -> 1e-4"},
            "smoke_and_stop": {"first_20_epochs_are_part_of_same_run": True,
                                "smoke_cap_gpu_hours": SMOKE_CAP_GPUH,
                                "projected_300_epoch_cap_with_20pct_margin_gpu_hours": PROJECTED_FULL_CAP_GPUH,
                                "attempt_absolute_cap_gpu_hours": TOTAL_ATTEMPT_CAP_GPUH,
                                "stop_if_nonfinite": True,
                                "stop_if_gradient_norm_or_prediction_or_bn_running_var_exceeds": SAFETY_LIMIT},
            "resource_accounting_before_attempt": prior,
            "resource_allocation": {"source": "existing authorized failure/repair reserve",
                                     "reserve_allocation_gpu_hours": RESERVE_REPAIR_ALLOCATION_GPUH,
                                     "candidate_cap_gpu_hours": TOTAL_ATTEMPT_CAP_GPUH,
                                     "reserve_after_candidate_allocation_gpu_hours": RESERVE_REPAIR_ALLOCATION_GPUH - TOTAL_ATTEMPT_CAP_GPUH,
                                     "no_other_package_caps_moved": True},
            "fit_gate": {"strongest_train_baseline": "best of zero/per-drug mean",
                         "required_relative_nmse_reduction": 0.10,
                         "train_nmse_threshold_approx": TRAIN_BASELINE_NMSE * 0.9,
                         "amplitude_ratio_range": [0.5, 2.0]},
            "inner_gate_if_train_passes": {"nmse_threshold_approx": INNER_BASELINE_NMSE * 0.95,
                                           "objects_improved_minimum": 15,
                                           "objects_total": INNER_OBJECTS,
                                           "amplitude_ratio_range": [0.5, 2.0]},
            "numeric_response_roles_read_before_fit_gate": ["TRAIN"],
            "inner_dev_numeric_response_read_before_fit_gate": False,
            "forbidden_roles_opened": []})
        attempt.mkdir(parents=False, exist_ok=False)
        atomic_json(attempt / "pre_gpu_attempt.json", {**pre_gpu, "phase": "attempt_created"})
        atomic_json(attempt / "train_input_access_receipt.json", {
            "status": "PASS_TRAIN_ONLY_NUMERIC_INPUT_READ",
            "source_run_id": args.source_run_id,
            "source_values_archive_sha256": denominator["source_values_sha256"],
            "source_key_table_sha256": denominator["source_key_table_sha256"],
            "runner_sha256": sha256(Path(__file__).resolve()),
            "numeric_response_roles_read_before_fit_gate": ["TRAIN"],
            "inner_dev_numeric_response_read_before_fit_gate": False,
            "train_condition_count": TRAIN_ROWS, "train_object_count": TRAIN_OBJECTS,
            "source_row_index_sha256": array_sha256(train_keys.source_row.astype(np.int64).to_numpy()),
            "train_treated_matrix_sha256": array_sha256(values["X_train"]),
            "train_control_matrix_sha256": array_sha256(values["X_ctl_train"]),
            "train_delta_matrix_sha256": array_sha256(values["delta_train"]),
            "decomposition_max_abs_error": denominator["decomposition_max_abs_error"],
            "forbidden_roles_opened": []})
        ledger_path = attempt / "resource_ledger.jsonl"
        append_jsonl(ledger_path, {"event": "GPU_ATTEMPT_STARTED", "at_utc": fit.utc_now(),
                                  "experiment_id": "EXP-052", "run_id": args.run_id,
                                  "attempt_id": args.attempt_id, "package_id": "authorized_repair_reserve",
                                  "pid": process_id, "device_uuid": GPU_UUID,
                                  "reserved_gpu_hours": TOTAL_ATTEMPT_CAP_GPUH,
                                  "source_settled_gpu_hours": prior["exp_settled_gpu_hours"],
                                  "config_sha256": config_sha, "runner_sha256": sha256(Path(__file__).resolve()),
                                  "device_snapshot": snapshot["selected_gpu"]})
        reservation_started = True

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in WSL2 Conda drugscreening-gpu")
        torch.cuda.set_device(torch.device("cuda:0"))
        device = torch.device("cuda:0")
        fit.RUN_ID = args.run_id
        fit.RUN_DIR = attempt
        fit.SEED = SEED
        model, transigen_module = fit._load_transigen(torch, device, SEED, features_dim=2048)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loader = DataLoader(TensorDataset(torch.from_numpy(values["X_ctl_train"]),
                                          torch.from_numpy(values["X_train"]),
                                          torch.from_numpy(train_features)),
                            batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0,
                            generator=torch.Generator().manual_seed(SEED))
        trajectory_path = attempt / "training_curve.jsonl"
        gpu_start = time.perf_counter()
        phase = "TRAIN_ACTIVE"
        atomic_json(attempt / "attempt_progress.json", {"phase": phase, "pid": process_id,
                    "updated_at_utc": fit.utc_now(), "completed_epochs": 0, "target_epochs": TRAIN_EPOCHS,
                    "actual_gpu_hours_upper_bound": 0.0, "forbidden_roles_opened": []})
        for epoch in range(1, TRAIN_EPOCHS + 1):
            model.train()
            sum_losses = np.zeros(len(LOSS_NAMES), dtype=np.float64)
            grad_norms: list[float] = []
            seen = 0
            for batch_index, batch in enumerate(loader):
                batch = tuple(tensor.to(device) for tensor in batch)
                seen += int(batch[0].shape[0])
                optimizer.zero_grad(set_to_none=True)
                _, _, pred_treated, terms = fit._transigen_forward_batch(torch, model, batch, device)
                if not all(bool(torch.isfinite(term).all()) for term in terms):
                    first_abnormal = {"stage": "TRAIN_LOSS_NONFINITE", "epoch": epoch,
                                      "batch_index": batch_index, "loss_names": list(LOSS_NAMES)}
                    raise FloatingPointError("non-finite TRAIN raw-sum objective")
                terms[0].backward()
                norm = fit._grad_global_norm(torch, model)
                if not math.isfinite(norm) or norm > SAFETY_LIMIT or any(
                        parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                        for parameter in model.parameters()):
                    first_abnormal = {"stage": "TRAIN_GRADIENT_SAFETY_STOP", "epoch": epoch,
                                      "batch_index": batch_index, "gradient_norm": norm,
                                      "safety_limit": SAFETY_LIMIT}
                    raise FloatingPointError("TRAIN gradient nonfinite or exceeded 1e6 stop limit")
                grad_norms.append(norm)
                sum_losses += np.asarray([float(term.detach().cpu()) for term in terms], dtype=np.float64)
                optimizer.step()
                if not params_finite(torch, model):
                    first_abnormal = {"stage": "POST_UPDATE_PARAMETER_NONFINITE", "epoch": epoch,
                                      "batch_index": batch_index}
                    raise FloatingPointError("TranSiGen parameters became nonfinite after optimizer update")
                bn_var = max_bn_running_var(model)
                if not math.isfinite(bn_var) or bn_var > SAFETY_LIMIT:
                    first_abnormal = {"stage": "BATCHNORM_SAFETY_STOP", "epoch": epoch,
                                      "batch_index": batch_index, "max_running_var": bn_var,
                                      "safety_limit": SAFETY_LIMIT}
                    raise FloatingPointError("BatchNorm running variance nonfinite or exceeded 1e6")
                if float(torch.max(torch.abs(pred_treated.detach())).cpu()) > SAFETY_LIMIT:
                    first_abnormal = {"stage": "TRAIN_OUTPUT_SAFETY_STOP", "epoch": epoch,
                                      "batch_index": batch_index, "safety_limit": SAFETY_LIMIT}
                    raise FloatingPointError("model output exceeded the 1e6 safety limit")
            if seen != TRAIN_ROWS:
                raise RuntimeError(f"TRAIN epoch processed {seen} conditions rather than {TRAIN_ROWS}")
            torch.cuda.synchronize(device)
            torch.manual_seed(SEED + 10000 + epoch)
            torch.cuda.manual_seed_all(SEED + 10000 + epoch)
            train_pred, eval_losses = fit._transigen_predict_and_losses(
                torch, model, values["X_ctl_train"], values["X_train"], train_features,
                BATCH_SIZE, device, SEED + 10000 + epoch)
            if (not np.isfinite(train_pred).all() or float(np.max(np.abs(train_pred))) > SAFETY_LIMIT
                    or max_bn_running_var(model) > SAFETY_LIMIT):
                first_abnormal = {"stage": "TRAIN_EVAL_OUTPUT_OR_BN_SAFETY_STOP", "epoch": epoch,
                                  "prediction_max_abs": float(np.nanmax(np.abs(train_pred))) if train_pred.size else None,
                                  "max_bn_running_var": max_bn_running_var(model), "safety_limit": SAFETY_LIMIT}
                raise FloatingPointError("TRAIN evaluation output/BN state failed the 1e6 safety gate")
            train_metric = fit.keyed_metrics(train_keys, train_y, train_pred, gene_variance)
            elapsed = time.perf_counter() - gpu_start
            projected_full = (elapsed / epoch) * TRAIN_EPOCHS * 1.20
            row = {"epoch": epoch, "train_loss_sum_per_condition": {
                       name: float(value / TRAIN_ROWS) for name, value in zip(LOSS_NAMES, sum_losses)},
                   "train_eval_loss_per_condition": eval_losses,
                   "train_metrics": train_metric, "gradient_norm_max": max(grad_norms),
                   "gradient_norm_mean": float(np.mean(grad_norms)),
                   "batchnorm_running_var_max": max_bn_running_var(model),
                   "learning_rate": LR, "elapsed_gpu_hours_upper_bound": elapsed / 3600,
                   "projected_300_epoch_gpu_hours_with_20pct": projected_full}
            append_jsonl(trajectory_path, row)
            if epoch == SMOKE_EPOCHS:
                smoke_pass = elapsed / 3600 <= SMOKE_CAP_GPUH and projected_full <= PROJECTED_FULL_CAP_GPUH
                smoke_receipt = {"status": "PASS_CONTINUE_SAME_RUN" if smoke_pass else "STOP_COST_OR_SMOKE_GATE",
                                 "first_20_epochs_are_part_of_final_candidate": True,
                                 "completed_smoke_epochs": epoch, "smoke_gpu_hours": elapsed / 3600,
                                 "smoke_cap_gpu_hours": SMOKE_CAP_GPUH,
                                 "projected_300_epoch_gpu_hours_with_20pct": projected_full,
                                 "projected_cap_gpu_hours": PROJECTED_FULL_CAP_GPUH,
                                 "last_train_metrics": train_metric,
                                 "forbidden_roles_opened": []}
                atomic_json(attempt / "20_epoch_cost_stability_smoke.json", smoke_receipt)
                print(json.dumps({"stage": "EXP052_20_EPOCH_SMOKE", **smoke_receipt}, ensure_ascii=False), flush=True)
                if not smoke_pass:
                    phase = "CLOSED_COST_GATE"
                    break
            if elapsed / 3600 >= TOTAL_ATTEMPT_CAP_GPUH:
                raise TimeoutError("EXP-052 one-shot absolute GPU-hour cap reached")
            atomic_json(attempt / "attempt_progress.json", {"phase": phase, "pid": process_id,
                        "updated_at_utc": fit.utc_now(), "completed_epochs": epoch,
                        "target_epochs": TRAIN_EPOCHS, "actual_gpu_hours_upper_bound": elapsed / 3600,
                        "last_train_nmse": train_metric["object_macro_nmse"],
                        "projected_full_gpu_hours_with_20pct": projected_full,
                        "first_abnormal": first_abnormal, "forbidden_roles_opened": []})
            if epoch == 1 or epoch % 10 == 0 or epoch == TRAIN_EPOCHS:
                print(json.dumps({"stage": "EXP052_TRAIN_PROGRESS", "epoch": epoch,
                                  "train_nmse": train_metric["object_macro_nmse"],
                                  "amplitude_ratio": train_metric["object_macro_amplitude_ratio"],
                                  "actual_gpu_hours": elapsed / 3600,
                                  "projected_full_gpu_hours_with_20pct": projected_full}, ensure_ascii=False), flush=True)
        elapsed = time.perf_counter() - gpu_start
        torch.cuda.synchronize(device)
        terminal_epoch = min(TRAIN_EPOCHS, len([1 for line in trajectory_path.read_text(encoding="utf-8").splitlines()
                                                if line.strip()]))
        curve_rows = [json.loads(line) for line in trajectory_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        last_row = curve_rows[-1]
        train_pred, final_losses = fit._transigen_predict_and_losses(
            torch, model, values["X_ctl_train"], values["X_train"], train_features,
            BATCH_SIZE, device, SEED + 10000 + terminal_epoch)
        train_metric = fit.keyed_metrics(train_keys, train_y, train_pred, gene_variance)
        fit_gate = fit._fit_gate(train_metric, baselines)
        inner_result: dict[str, Any] = {"status": "NOT_EVALUATED_TRAIN_GATE_FAILED"}
        output_arrays: dict[str, Any] = {
            "train_condition_key": train_keys.full_condition_key.astype(str).to_numpy(dtype=str),
            "train_source_object_id": train_keys.source_object_id.astype(str).to_numpy(dtype=str),
            "train_full_inchikey": train_keys.full_inchikey.astype(str).to_numpy(dtype=str),
            "genes": values["genes"], "train_target_delta": train_y,
            "train_prediction_delta": train_pred.astype(np.float32),
        }
        if fit_gate["status"] == "PASS":
            atomic_json(attempt / "inner_dev_score_open_receipt.json", {
                "status": "OPENED_AFTER_TRAIN_FIT_GATE_PASS",
                "opened_at_utc": fit.utc_now(),
                "fit_gate_metrics": fit_gate,
                "inner_dev_condition_count": INNER_ROWS,
                "inner_dev_object_count": INNER_OBJECTS,
                "source_row_index_sha256": array_sha256(inner_keys.source_row.astype(np.int64).to_numpy()),
                "forbidden_roles_opened": []})
            inner_values = read_role_expression(inner_keys, values["genes"])
            torch.manual_seed(SEED + 20000)
            torch.cuda.manual_seed_all(SEED + 20000)
            inner_pred, inner_losses = fit._transigen_predict_and_losses(
                torch, model, inner_values["X_ctl"], inner_values["X"], inner_features,
                BATCH_SIZE, device, SEED + 20000)
            # The comparator is the per-drug mean fitted on TRAIN only.
            train_by_key = {str(key): train_y[train_keys.full_inchikey.astype(str).to_numpy() == str(key)].mean(
                axis=0, dtype=np.float64).astype(np.float32)
                for key in sorted(train_keys.full_inchikey.astype(str).unique())}
            inner_baseline = np.stack([train_by_key[str(key)] for key in inner_keys.full_inchikey.astype(str)], axis=0)
            inner_metric = fit.keyed_metrics(inner_keys, inner_values["delta"], inner_pred, gene_variance)
            improved_n, paired = object_improvement_count(inner_keys, inner_values["delta"], inner_pred,
                                                           inner_baseline, gene_variance)
            inner_fit_pass = (inner_metric["object_macro_nmse"] is not None
                              and inner_metric["object_macro_nmse"] <= INNER_BASELINE_NMSE * 0.95
                              and improved_n >= 15
                              and inner_metric["object_macro_amplitude_ratio"] is not None
                              and 0.5 <= inner_metric["object_macro_amplitude_ratio"] <= 2.0)
            inner_result = {"status": "PASS_DEVELOPMENT_GATE" if inner_fit_pass else "FAIL_DEVELOPMENT_GATE",
                            "metrics": inner_metric, "per_object_paired_to_per_drug_mean": paired,
                            "n_objects_improved": improved_n, "n_objects_total": INNER_OBJECTS,
                            "required_improved_objects": 15,
                            "nmse_threshold": INNER_BASELINE_NMSE * 0.95,
                            "amplitude_ratio_range": [0.5, 2.0],
                            "raw_sum_objective_per_condition": inner_losses,
                            "inner_dev_scored_once_after_train_gate": True}
            output_arrays.update({"inner_dev_condition_key": inner_keys.full_condition_key.astype(str).to_numpy(dtype=str),
                                  "inner_dev_source_object_id": inner_keys.source_object_id.astype(str).to_numpy(dtype=str),
                                  "inner_dev_full_inchikey": inner_keys.full_inchikey.astype(str).to_numpy(dtype=str),
                                  "inner_dev_target_delta": inner_values["delta"],
                                  "inner_dev_prediction_delta": inner_pred.astype(np.float32),
                                  "inner_dev_per_drug_mean_delta": inner_baseline})
        np.savez_compressed(attempt / "keyed_predictions.npz", **output_arrays)
        torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "model": "upstream original TranSiGen", "epoch": terminal_epoch,
                    "seed": SEED, "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
                    "training_contract": "raw-sum/no-clip, batch64 shuffle, ordinary BatchNorm; low-lr stability variant"},
                   attempt / "transigen_final_state.pt")
        summary = {"status": "COMPLETED_FIXED_EPOCH_CANDIDATE" if terminal_epoch == TRAIN_EPOCHS
                   else "CLOSED_COST_GATE_BEFORE_FULL_RUN" if phase == "CLOSED_COST_GATE" else "INCOMPLETE",
                   "experiment_id": "EXP-052", "run_id": args.run_id, "attempt_id": args.attempt_id,
                   "epochs_completed": terminal_epoch, "epochs_planned": TRAIN_EPOCHS,
                   "training_loss": last_row["train_loss_sum_per_condition"],
                   "train_metrics": train_metric, "train_fit_gate": fit_gate,
                   "inner_dev": inner_result,
                   "resource": {"gpu_uuid": GPU_UUID, "actual_gpu_hours_upper_bound": elapsed / 3600,
                               "absolute_attempt_cap_gpu_hours": TOTAL_ATTEMPT_CAP_GPUH,
                               "projected_300_epoch_with_margin": last_row["projected_300_epoch_gpu_hours_with_20pct"]},
                   "denominator": {"train_conditions": TRAIN_ROWS, "train_objects": TRAIN_OBJECTS,
                                   "inner_dev_conditions": INNER_ROWS if inner_result["status"] != "NOT_EVALUATED_TRAIN_GATE_FAILED" else 0,
                                   "inner_dev_objects": INNER_OBJECTS if inner_result["status"] != "NOT_EVALUATED_TRAIN_GATE_FAILED" else 0,
                                   "full_inchikeys": N_DRUGS, "genes": N_GENES,
                                   "forbidden_roles_opened": []},
                   "source_receipt": denominator, "source_feature_sha256": feature_receipt["feature_npz_sha256"],
                   "final_train_eval_loss_per_condition": final_losses,
                   "claim_boundary": "single fixed-source stability candidate; development evidence only, not PDO/generalization validation"}
        atomic_json(attempt / "result.json", summary)
        manifest_rows = []
        atomic_json(attempt / "attempt_progress.json", {"phase": "CLOSED", "pid": process_id,
                    "updated_at_utc": fit.utc_now(), "completed_epochs": terminal_epoch,
                    "actual_gpu_hours_upper_bound": elapsed / 3600,
                    "train_gate": fit_gate["status"], "inner_gate": inner_result["status"],
                    "forbidden_roles_opened": []})
        final_snapshot = fit._device_snapshot()["selected_gpu"]
        close = {"status": summary["status"], "experiment_id": "EXP-052", "run_id": args.run_id,
                 "attempt": args.attempt_id, "pid": process_id, "device_uuid": GPU_UUID,
                 "charged_gpu_hours_conservative_upper_bound": elapsed / 3600,
                 "reserved_gpu_hours": TOTAL_ATTEMPT_CAP_GPUH,
                 "unused_reservation_released_gpu_hours": max(0.0, TOTAL_ATTEMPT_CAP_GPUH - elapsed / 3600),
                 "prior_global_settled_gpu_hours": prior["exp_settled_gpu_hours"],
                 "global_settled_gpu_hours_after_close": prior["exp_settled_gpu_hours"] + elapsed / 3600,
                 "result_sha256": sha256(attempt / "result.json"),
                 "device_snapshot_at_close": final_snapshot, "forbidden_roles_opened": []}
        atomic_json(attempt / "attempt_close_receipt.json", close)
        for path in sorted(attempt.iterdir()):
            if path.is_file() and path.name not in {"sha256_manifest.tsv", "resource_ledger.jsonl"}:
                manifest_rows.append(f"{sha256(path)}\t{path.name}")
        (attempt / "sha256_manifest.tsv").write_text("sha256\tfile\n" + "\n".join(manifest_rows) + "\n",
                                                     encoding="utf-8")
        append_jsonl(ledger_path, {"event": "GPU_ATTEMPT_COMPLETED", "at_utc": fit.utc_now(),
                                  "experiment_id": "EXP-052", "run_id": args.run_id,
                                  "attempt_id": args.attempt_id, "package_id": "authorized_repair_reserve",
                                  "pid": process_id, "device_uuid": GPU_UUID,
                                  "actual_gpu_hours": elapsed / 3600,
                                  "charged_gpu_hours_conservative_upper_bound": elapsed / 3600,
                                  "reserved_gpu_hours": TOTAL_ATTEMPT_CAP_GPUH,
                                  "unused_reservation_released_gpu_hours": max(0.0, TOTAL_ATTEMPT_CAP_GPUH - elapsed / 3600),
                                  "close_receipt": "attempt_close_receipt.json"})
        print(json.dumps({"stage": summary["status"], "train_gate": fit_gate["status"],
                          "inner_gate": inner_result["status"], "gpu_hours": elapsed / 3600,
                          "train_nmse": train_metric["object_macro_nmse"]}, ensure_ascii=False), flush=True)
        return 0
    except BaseException as error:
        if reservation_started and ledger_path is not None:
            elapsed = max(0.0, time.perf_counter() - gpu_start) if gpu_start is not None else 0.0
            try:
                partial = {"status": "FAILED_CLOSED", "experiment_id": "EXP-052", "run_id": args.run_id,
                           "attempt_id": args.attempt_id, "phase": phase,
                           "exception_type": type(error).__name__, "exception_message": str(error),
                           "first_abnormal": first_abnormal, "charged_gpu_hours_conservative_upper_bound": elapsed / 3600,
                           "forbidden_roles_opened": []}
                atomic_json(attempt / "failure_close_receipt.json", partial)
                append_jsonl(ledger_path, {"event": "GPU_ATTEMPT_FAILED_CLOSED", "at_utc": fit.utc_now(),
                                          "experiment_id": "EXP-052", "run_id": args.run_id,
                                          "attempt_id": args.attempt_id, "package_id": "authorized_repair_reserve",
                                          "pid": process_id, "device_uuid": GPU_UUID,
                                          "charged_gpu_hours_conservative_upper_bound": elapsed / 3600,
                                          "reserved_gpu_hours": TOTAL_ATTEMPT_CAP_GPUH,
                                          "unused_reservation_released_gpu_hours": max(0.0, TOTAL_ATTEMPT_CAP_GPUH - elapsed / 3600),
                                          "failure_receipt": "failure_close_receipt.json"})
                atomic_json(attempt / "attempt_progress.json", {"phase": "FAILED_CLOSED",
                            "pid": process_id, "updated_at_utc": fit.utc_now(),
                            "actual_gpu_hours_upper_bound": elapsed / 3600,
                            "first_abnormal": first_abnormal, "exception": str(error),
                            "forbidden_roles_opened": []})
            except Exception as close_error:
                print(json.dumps({"stage": "CLOSE_RECEIPT_WRITE_FAILED", "error": str(close_error)}, ensure_ascii=False), flush=True)
        elif run_dir.exists() and not (run_dir / "pre_gpu_failure_receipt.json").exists():
            atomic_json(run_dir / "pre_gpu_failure_receipt.json", {
                "status": "CPU_PREFLIGHT_FAILED_CLOSED_NO_GPU", "experiment_id": "EXP-052",
                "run_id": args.run_id, "attempt": args.attempt_id, "gpu_smoke_started": False,
                "charged_gpu_hours": 0.0, "exception_type": type(error).__name__,
                "exception_message": str(error), "observed_close_time_utc": fit.utc_now()})
        print(json.dumps({"stage": "FAILED_CLOSED", "exception_type": type(error).__name__,
                          "exception_message": str(error), "first_abnormal": first_abnormal}, ensure_ascii=False), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
