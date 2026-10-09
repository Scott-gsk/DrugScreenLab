#!/usr/bin/env python3
"""Run the frozen, CPU-only EXP-060 inference randomness comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.modeling import exp051_contract_fit_smoke as runner


PARENT_RUN_ID = "20261008_exp059_transigen_lr1e4_bindingfix_v2"
PARENT_ATTEMPT_ID = "fit_attempt_001"
EXPECTED_CHECKPOINT_SHA256 = "fa75ce08d6013c42ee52f503230cfc0f1bbe4a30036b11c4e705f7d2981d92f2"
EXPECTED_TRAIN_METRICS_SHA256 = "57605394f80d78e37fb8ebd48d5d35b2554db6976a45ac2698054cf64bd0f0ac"
EXPECTED_DEV_RECOVERY_SHA256 = "0bb8f91eca7e0127cbcb19cb57a13ded58d12512678ea1e0ed789aba56504e82"
EXPECTED_DEV_ROOT_RECEIPT_SHA256 = "327c82173369a79c8f81ef8660d824f0f42423b06d4f1fe97b95c753d033f8b1"
BOOTSTRAP_SEED = 2060
BOOTSTRAP_REPEATS = 2000
MODEL_SEED = 20261006
INFERENCE_SEEDS = tuple(range(20261006, 20261022))


class InferenceFailure(FloatingPointError):
    def __init__(self, message: str, evidence: dict[str, Any]):
        super().__init__(message)
        self.evidence = evidence


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def tensor_tree_hash(state: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode("utf-8"))
        h.update(str(array.dtype).encode("ascii"))
        h.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        h.update(array.tobytes(order="C"))
    return h.hexdigest()


def snapshot_model_state(model: Any) -> tuple[dict[str, Any], str, dict[str, Any]]:
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    bn_buffers: dict[str, Any] = {}
    for module_name, module in model.named_modules():
        if module.__class__.__name__ in {"BatchNorm1d", "BatchNorm2d", "BatchNorm3d"}:
            for buffer_name in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(module, buffer_name, None)
                if value is not None:
                    name = f"{module_name}.{buffer_name}" if module_name else buffer_name
                    bn_buffers[name] = value.detach().cpu().clone()
    return state, tensor_tree_hash(state), bn_buffers


def exact_state_equal(torch: Any, model: Any, frozen_state: dict[str, Any]) -> dict[str, Any]:
    current = model.state_dict()
    if set(current) != set(frozen_state):
        return {"exact": False, "reason": "state_dict_key_set_changed",
                "expected_keys": len(frozen_state), "observed_keys": len(current)}
    mismatched = [name for name in sorted(current)
                  if not torch.equal(current[name].detach().cpu(), frozen_state[name])]
    return {"exact": not mismatched, "mismatched_tensor_names": mismatched,
            "tensor_count": len(current), "state_tree_sha256": tensor_tree_hash(current)}


def module_modes_eval(model: Any) -> bool:
    return all(not module.training for module in model.modules())


def config_and_parent_context(config_path: Path, run_id: str) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("EXP-060 requires CUDA_VISIBLE_DEVICES to be empty")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("experiment_id") != "EXP-060":
        raise ValueError("EXP-060 configuration is missing or has a different experiment id")
    if cfg.get("authorization", {}).get("user_scope") != "EXP060-FROZEN-INFERENCE-0103":
        raise ValueError("configuration is not bound to the frozen EXP-060 authorization")
    if cfg.get("execution", {}).get("run_id") != run_id:
        raise ValueError("requested run_id differs from the frozen EXP-060 configuration")
    if cfg.get("model", {}).get("stochastic_inference_seeds") != list(INFERENCE_SEEDS):
        raise ValueError("fixed inference seed list changed")
    if cfg.get("data", {}).get("checkpoint", {}).get("sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("frozen checkpoint SHA in config changed")
    parent = ROOT / "artifacts/experiments/EXP-059" / PARENT_RUN_ID / PARENT_ATTEMPT_ID
    recovery = ROOT / "artifacts/experiments/EXP-059" / PARENT_RUN_ID / "postcheckpoint_inner_dev_evaluation_recovery_v1"
    checkpoint = parent / "TranSiGen_final_epoch_300.pt"
    train_baseline = parent / "train_baseline_metrics.json"
    dev_recovery = recovery / "recovered_fixed300_evaluation.json"
    dev_root = recovery / "root_independent_readback.json"
    required = [
        checkpoint, train_baseline, dev_recovery, dev_root,
        parent / "checkpoint_freeze_receipt.json", parent / "epoch_300_endpoint_receipt.json",
        parent / "attempt_close_receipt.json", parent / "fit_smoke_manifest.json",
        parent / "TranSiGen_epoch_300_train_keyed_predictions.npz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("required frozen EXP-059 evidence is missing: " + ", ".join(missing))
    checkpoint_sha = sha256(checkpoint)
    train_baseline_sha = sha256(train_baseline)
    # Bind the frozen parent files by their preregistered hashes, but do not scan or
    # deserialize their development scores before the current TRAIN selection is frozen.
    dev_recovery_sha = EXPECTED_DEV_RECOVERY_SHA256
    dev_root_sha = EXPECTED_DEV_ROOT_RECEIPT_SHA256
    freeze = read_json(parent / "checkpoint_freeze_receipt.json")
    endpoint = read_json(parent / "epoch_300_endpoint_receipt.json")
    close = read_json(parent / "attempt_close_receipt.json")
    if (checkpoint_sha != EXPECTED_CHECKPOINT_SHA256
            or freeze.get("status") != "PASS_FROZEN_EPOCH_300_CHECKPOINT"
            or freeze.get("checkpoint_sha256") != checkpoint_sha
            or freeze.get("epoch") != 300
            or endpoint.get("prediction_archive_written") is not True
            or endpoint.get("train_main_delta_finite") is not True
            or endpoint.get("train_main_delta_denominator") != 419 * 978
            or close.get("last_completed_finite_epoch") != 300
            or close.get("gpu_hours") != 0.0
            or train_baseline_sha != EXPECTED_TRAIN_METRICS_SHA256
            or dev_recovery_sha != EXPECTED_DEV_RECOVERY_SHA256
            or dev_root_sha != EXPECTED_DEV_ROOT_RECEIPT_SHA256):
        raise RuntimeError("frozen EXP-059 checkpoint, endpoint, baseline or root-evaluation binding failed")
    return {
        "config": cfg, "config_sha256": sha256(config_path), "config_path": str(config_path.relative_to(ROOT)),
        "parent_attempt": parent, "recovery_dir": recovery, "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha, "train_baseline": train_baseline,
        "train_baseline_sha256": train_baseline_sha, "dev_recovery": dev_recovery,
        "dev_recovery_sha256": dev_recovery_sha, "dev_root_receipt": dev_root,
        "dev_root_receipt_sha256": dev_root_sha, "dev_root_result_path": dev_root,
        "freeze_receipt": freeze,
        "endpoint_receipt": endpoint, "close_receipt": close,
    }


def load_train_only() -> tuple[Any, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    runner.configure_experiment("EXP-059")
    runner.configure_run(PARENT_RUN_ID)
    gate = runner._reuse_root_verified_exp055_train_gate()
    source_run = runner.SOURCE_RUN_DIR
    keys, values, feature_by_key, evidence = runner._load_exp055_trainpath_train_inputs_reusing_gate(
        gate, source_run / "response_blind_conditions.tsv", source_run)
    if (len(keys) != 419 or keys.source_object_id.nunique() != 110
            or keys.full_inchikey.nunique() != 5 or values["delta"].shape != (419, 978)
            or values["X_ctl"].shape != (419, 978) or values["genes"].shape != (978,)
            or evidence.get("inner_dev_numeric_rows_loaded") is not False):
        raise ValueError("TRAIN-only input loader did not return the registered 419x978 identity contract")
    feature_matrix = np.stack([feature_by_key[str(key)] for key in keys.full_inchikey.astype(str)])
    if feature_matrix.shape != (419, 2048) or not np.isfinite(feature_matrix).all():
        raise ValueError("TRAIN full-key ECFP4 feature matrix is not finite 419x2048")
    return keys, values, feature_by_key, {
        "train_gate": gate, "train_input_evidence": evidence,
        "feature_matrix_sha256": hashlib.sha256(feature_matrix.astype(np.float32).tobytes()).hexdigest(),
        "inner_dev_numeric_rows_loaded": False,
        "pdo_outer_locked_numeric_values_opened": False,
        "panacea_numeric_values_opened": False,
    }


def run_inference(torch: Any, model: Any, x_ctl: np.ndarray, features: np.ndarray,
                  *, seed: int | None, deterministic_mu: bool, batch_size: int,
                  progress_check: Any | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    if seed is None:
        raise ValueError("every inference must use a fixed registered seed")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.random.default_generator.manual_seed(seed)
    model.eval()
    if not module_modes_eval(model):
        raise RuntimeError("model or a child module is not in eval mode")
    call_count = 0
    original_method = model.sample_latent

    def observed_sample(mu: Any, logvar: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if deterministic_mu:
            return mu
        return original_method(mu, logvar)

    model.sample_latent = observed_sample
    chunks: list[np.ndarray] = []
    finite_count = 0
    denominator = int(x_ctl.shape[0] * x_ctl.shape[1])
    try:
        with torch.no_grad():
            for start in range(0, len(x_ctl), batch_size):
                if progress_check is not None:
                    progress_check()
                stop = min(start + batch_size, len(x_ctl))
                x1 = torch.as_tensor(x_ctl[start:stop], dtype=torch.float32, device="cpu")
                feat = torch.as_tensor(features[start:stop], dtype=torch.float32, device="cpu")
                output = model.forward(x1, feat)
                if len(output) < 4:
                    raise RuntimeError("original TranSiGen forward returned fewer than four outputs")
                delta = (output[3] - x1).detach()
                if tuple(delta.shape) != (stop - start, 978):
                    raise RuntimeError(f"prediction shape is not {(stop - start, 978)}: {tuple(delta.shape)}")
                finite = torch.isfinite(delta)
                finite_count += int(finite.sum().item())
                if not bool(finite.all()):
                    bad = torch.nonzero(~finite, as_tuple=False)
                    raise InferenceFailure("non-finite full-key prediction",
                        {"seed": seed, "candidate": "mu_both_levels" if deterministic_mu else "stochastic_single",
                         "batch_start": start, "batch_stop": stop, "full_denominator": denominator,
                         "finite_n_observed": finite_count, "nonfinite_n_observed": denominator - finite_count,
                         "batch_nonfinite_n": int((~finite).sum().item()),
                         "first_nonfinite_positions": bad[:32].cpu().tolist(),
                         "inner_dev_numeric_rows_loaded": False})
                chunks.append(delta.cpu().numpy().astype(np.float32, copy=True))
                if progress_check is not None:
                    progress_check()
    finally:
        delattr(model, "sample_latent")
    prediction = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 978), dtype=np.float32)
    if prediction.shape != x_ctl.shape or not np.isfinite(prediction).all():
        raise InferenceFailure("incomplete or non-finite full-key prediction",
            {"seed": seed, "candidate": "mu_both_levels" if deterministic_mu else "stochastic_single",
             "full_denominator": denominator, "finite_n_observed": finite_count,
             "nonfinite_n_observed": denominator - finite_count})
    expected_calls = 2 * int(math.ceil(len(x_ctl) / batch_size))
    if call_count != expected_calls:
        raise RuntimeError(f"expected two latent paths per batch ({expected_calls}), observed {call_count}")
    return prediction, {"seed": seed, "deterministic_mu_at_both_latent_levels": deterministic_mu,
                        "latent_reparameterization_calls": call_count,
                        "expected_latent_reparameterization_calls": expected_calls,
                        "prediction_denominator": denominator, "finite_n": finite_count,
                        "nonfinite_n": denominator - finite_count,
                        "batch_size": batch_size, "model_eval": module_modes_eval(model),
                        "no_grad": True}


def metric_per_drug(keys: Any, truth: np.ndarray, prediction: np.ndarray,
                    gene_variance: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(keys.full_inchikey.astype(str).unique()):
        pos = np.flatnonzero(keys.full_inchikey.astype(str).to_numpy() == key)
        subkeys = keys.iloc[pos].reset_index(drop=True)
        metric = runner.keyed_metrics(subkeys, truth[pos], prediction[pos], gene_variance)
        rows.append({"full_inchikey": str(key), "n_conditions": len(pos),
                     "n_source_objects": int(subkeys.source_object_id.nunique()), **metric})
    return rows


def object_macro_nmse(keys: Any, truth: np.ndarray, prediction: np.ndarray,
                      gene_variance: np.ndarray) -> tuple[list[str], np.ndarray]:
    use_gene = np.isfinite(gene_variance) & (gene_variance > 1e-12)
    if not use_gene.any():
        raise ValueError("TRAIN-only gene variance has no positive denominator")
    err = prediction.astype(np.float64) - truth.astype(np.float64)
    row_nmse = np.mean(np.square(err[:, use_gene]) / gene_variance[use_gene][None, :], axis=1)
    frame = keys[["source_object_id"]].copy().reset_index(drop=True)
    frame["nmse"] = row_nmse
    grouped = frame.groupby("source_object_id", sort=True).nmse.mean()
    return grouped.index.astype(str).tolist(), grouped.to_numpy(dtype=np.float64)


def bootstrap_paired_diff(reference: np.ndarray, candidate: np.ndarray,
                          rng: np.random.Generator, repeats: int = BOOTSTRAP_REPEATS) -> dict[str, Any]:
    if reference.shape != candidate.shape or reference.ndim != 1 or not len(reference):
        raise ValueError("paired source-object bootstrap arrays must have equal nonempty shape")
    diff = candidate.astype(np.float64) - reference.astype(np.float64)
    sampled = rng.integers(0, len(diff), size=(repeats, len(diff)))
    means = diff[sampled].mean(axis=1)
    return {"n_objects": int(len(diff)), "n_resamples": int(repeats),
            "difference_definition": "candidate minus reference object-macro NMSE; negative favors candidate",
            "observed_mean_difference": float(diff.mean()),
            "ci_95_low": float(np.quantile(means, 0.025)),
            "ci_95_high": float(np.quantile(means, 0.975)),
            "selection_use": False, "rng_seed": BOOTSTRAP_SEED}


def save_keyed(path: Path, keys: Any, genes: np.ndarray, target: np.ndarray,
               prediction: np.ndarray, *, role: str, method: str,
               seed: int | None) -> None:
    arrays: dict[str, Any] = {
        "condition_key": keys.full_condition_key.astype(str).to_numpy(dtype=np.str_),
        "source_object_id": keys.source_object_id.astype(str).to_numpy(dtype=np.str_),
        "full_inchikey": keys.full_inchikey.astype(str).to_numpy(dtype=np.str_),
        "genes": np.asarray(genes, dtype=np.str_),
        "target_delta": np.asarray(target, dtype=np.float32),
        "predicted_delta": np.asarray(prediction, dtype=np.float32),
        "role": np.asarray(role, dtype=np.str_),
        "method": np.asarray(method, dtype=np.str_),
    }
    if seed is not None:
        arrays["inference_seed"] = np.asarray([seed], dtype=np.int64)
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(arrays):
            raise RuntimeError(f"prediction archive fields do not round-trip exactly: {path.name}")
        for name, expected in arrays.items():
            if not np.array_equal(archive[name], expected):
                raise RuntimeError(f"prediction archive array changed during round-trip: {path.name}/{name}")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, lineterminator="\n")


def baseline_predictions(keys: Any, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means: dict[str, np.ndarray] = {}
    key_values = keys.full_inchikey.astype(str).to_numpy()
    for key in sorted(set(key_values.tolist())):
        positions = np.flatnonzero(key_values == key)
        means[key] = y[positions].mean(axis=0, dtype=np.float64).astype(np.float32)
    per_drug = np.stack([means[key] for key in key_values]).astype(np.float32)
    zero = np.zeros_like(y, dtype=np.float32)
    return per_drug, zero


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def run_package(config_path: Path, run_id: str, device: str) -> dict[str, Any]:
    if device != "cpu":
        raise ValueError("EXP-060 accepts only --device cpu")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty")
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    context = config_and_parent_context(config_path, run_id)
    run_root = ROOT / "artifacts/experiments/EXP-060" / run_id
    if run_root.exists():
        raise FileExistsError(f"EXP-060 refuses to overwrite an existing run directory: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    attempt = run_root / "fit_attempt_001"
    attempt.mkdir(parents=False, exist_ok=False)
    ledger = attempt / "resource_ledger.jsonl"
    config = context["config"]
    precharge = float(config["authorization"]["new_package_pre_registration_shell_failure_seconds"])
    new_cap = float(config["authorization"]["new_package_cpu_cap_seconds"])
    inherited_remaining = float(config["authorization"]["inherited_exp059_cpu_remaining_seconds"])
    def resource_guard(stage: str) -> None:
        charged = precharge + (time.perf_counter() - started_wall) + (time.process_time() - started_cpu)
        cap = min(new_cap, inherited_remaining)
        if charged > cap:
            raise TimeoutError(f"EXP-060 conservative CPU charge exceeded its registered cap during {stage}: "
                               f"{charged:.3f}s > {cap:.3f}s")
    append_jsonl(ledger, {"event": "PACKAGE_STARTED", "at_utc": runner.utc_now(),
                         "run_id": run_id, "attempt_id": attempt.name, "device": "cpu",
                         "cuda_visible_devices": "", "gpu_hours": 0.0,
                         "pre_registration_shell_failure_charge_seconds": precharge,
                         "new_package_cpu_cap_seconds": new_cap,
                         "inherited_exp059_cpu_remaining_seconds": inherited_remaining})
    manifest = {
        "experiment_id": "EXP-060", "run_id": run_id, "attempt_id": attempt.name,
        "authorization": config["authorization"]["user_scope"], "config_path": context["config_path"],
        "config_sha256": context["config_sha256"],
        "runner_sha256": sha256(Path(__file__).resolve()),
        "checkpoint_path": str(context["checkpoint"].relative_to(ROOT)),
        "checkpoint_sha256_before": context["checkpoint_sha256"],
        "parent_attempt_close_receipt_sha256": sha256(context["parent_attempt"] / "attempt_close_receipt.json"),
        "parent_train_baseline_sha256": context["train_baseline_sha256"],
        "parent_inner_dev_evaluation_sha256": context["dev_recovery_sha256"],
        "parent_inner_dev_root_readback_sha256": context["dev_root_receipt_sha256"],
        "train_conditions": 419, "train_objects": 110, "full_inchikeys": 5, "genes": 978,
        "inner_dev_conditions": 93, "inner_dev_objects": 24, "object_overlap": 0,
        "device": "cpu", "cuda_visible_devices": "", "gpu_hours_cap": 0,
        "inference_seeds": list(INFERENCE_SEEDS), "model_selection": "none for checkpoint; inference-only TRAIN gate",
        "inner_dev_numeric_rows_loaded": False, "pdo_outer_locked_numeric_values_opened": False,
        "panacea_numeric_values_opened": False, "created_at_utc": runner.utc_now(),
    }
    atomic_json(attempt / "run_manifest.json", manifest)
    atomic_json(attempt / "attempt_progress.json", {"phase": "FROZEN_INPUTS_BOUND",
        "updated_at_utc": runner.utc_now(), "pid": os.getpid(), "inner_dev_numeric_rows_loaded": False})
    torch = None
    try:
        keys, values, feature_by_key, input_evidence = load_train_only()
        train_y = values["delta"].astype(np.float32, copy=False)
        gene_variance = np.var(train_y.astype(np.float64), axis=0, ddof=0)
        positive = np.isfinite(gene_variance) & (gene_variance > 1e-12)
        if positive.sum() != 978:
            raise ValueError("the frozen TRAIN denominator changed from 978 positive-variance genes")
        train_features = runner._full_key_features(
            keys, np.arange(len(keys), dtype=np.int64), feature_by_key)
        import torch as torch_module
        torch = torch_module
        torch.set_num_threads(1)
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA was initialized in a CPU-only package")
        model, _ = runner._load_transigen(torch, torch.device("cpu"), MODEL_SEED,
                                          features_dim=2048, seed_cuda=False)
        payload = torch.load(context["checkpoint"], map_location="cpu", weights_only=True)
        if (payload.get("epoch") != 300
                or payload.get("checkpoint_role") != "frozen_fixed_epoch_300_main_endpoint"
                or set(payload.get("state_dict", {})) != set(model.state_dict())):
            raise RuntimeError("checkpoint role, endpoint or architecture state keys do not match")
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        if not module_modes_eval(model):
            raise RuntimeError("not all TranSiGen modules entered eval mode")
        frozen_state, frozen_state_sha, frozen_bn = snapshot_model_state(model)
        atomic_json(attempt / "model_state_before_inference.json", {
            "state_tree_sha256": frozen_state_sha, "state_tensor_count": len(frozen_state),
            "bn_buffer_count": len(frozen_bn), "bn_buffer_sha256": tensor_tree_hash(frozen_bn),
            "all_modules_eval": module_modes_eval(model), "checkpoint_sha256": sha256(context["checkpoint"]),
            "optimizer_loaded": False, "optimizer_updates": 0})
        atomic_json(attempt / "input_receipt.json", {
            "input_evidence": input_evidence, "train_conditions": len(keys),
            "train_objects": int(keys.source_object_id.nunique()),
            "full_inchikeys": int(keys.full_inchikey.nunique()), "genes": len(values["genes"]),
            "train_condition_keys_sha256": hashlib.sha256(
                ("\n".join(keys.full_condition_key.astype(str).tolist()) + "\n").encode()).hexdigest(),
            "gene_axis_sha256": hashlib.sha256(
                ("\n".join(values["genes"].astype(str).tolist()) + "\n").encode()).hexdigest(),
            "target_delta_sha256": hashlib.sha256(train_y.tobytes()).hexdigest(),
            "training_source_rows_sha256": input_evidence["train_input_evidence"]["train_source_rows_sha256"],
            "inner_dev_numeric_rows_loaded": False, "pdo_outer_locked_numeric_values_opened": False,
            "panacea_numeric_values_opened": False, "checkpoint_sha256": context["checkpoint_sha256"]})

        stochastic_predictions: list[np.ndarray] = []
        stochastic_metrics: list[dict[str, Any]] = []
        per_drug_rows: list[dict[str, Any]] = []
        per_object_rows: list[dict[str, Any]] = []
        inference_costs: list[float] = []
        stochastic_failure: dict[str, Any] | None = None
        for index, seed in enumerate(INFERENCE_SEEDS):
            before_wall, before_cpu = time.perf_counter(), time.process_time()
            try:
                pred, evidence = run_inference(torch, model, values["X_ctl"], train_features,
                    seed=seed, deterministic_mu=False, batch_size=int(config["model"]["batch_size"]),
                    progress_check=lambda: resource_guard(f"random seed {seed}"))
            except InferenceFailure as exc:
                stochastic_failure = {"status": "FAILED_CLOSED_NONFINITE_FIXED_RANDOM_SEED",
                    "failure": exc.evidence, "exception": str(exc), "seed_index": index,
                    "seed_count_fixed": len(INFERENCE_SEEDS)}
                atomic_json(attempt / f"random_seed_{seed}_failure.json", stochastic_failure)
                append_jsonl(ledger, {"event": "FIXED_RANDOM_SEED_FAILED_NONFINITE", "seed": seed,
                    "at_utc": runner.utc_now(), "full_denominator": exc.evidence.get("full_denominator"),
                    "finite_n_observed": exc.evidence.get("finite_n_observed"),
                    "nonfinite_n_observed": exc.evidence.get("nonfinite_n_observed"),
                    "gpu_hours": 0.0})
                break
            elapsed = (time.perf_counter() - before_wall) + (time.process_time() - before_cpu)
            inference_costs.append(elapsed)
            stochastic_predictions.append(pred)
            file = attempt / f"random_seed_{seed}_train_keyed_predictions.npz"
            save_keyed(file, keys, values["genes"], train_y, pred,
                       role="TRAIN", method="stochastic_single", seed=seed)
            metric = runner.keyed_metrics(keys, train_y, pred, gene_variance)
            record = {"seed": seed, "metric": metric, "inference": evidence,
                      "prediction_path": file.name, "prediction_sha256": sha256(file)}
            stochastic_metrics.append(record)
            per_drug_rows.extend({"method": f"random_seed_{seed}", **item}
                                 for item in metric_per_drug(keys, train_y, pred, gene_variance))
            object_ids, object_nmse = object_macro_nmse(keys, train_y, pred, gene_variance)
            per_object_rows.extend({"method": f"random_seed_{seed}", "source_object_id": obj,
                                    "object_macro_nmse": float(value)}
                                   for obj, value in zip(object_ids, object_nmse))
            append_jsonl(ledger, {"event": "FIXED_RANDOM_SEED_COMPLETED", "seed": seed,
                "at_utc": runner.utc_now(), "wall_plus_process_seconds": elapsed,
                "full_denominator": evidence["prediction_denominator"], "finite_n": evidence["finite_n"],
                "prediction_sha256": sha256(file), "gpu_hours": 0.0})
            atomic_json(attempt / "attempt_progress.json", {
                "phase": "TRAIN_FIXED_STOCHASTIC_INFERENCE", "completed_fixed_seeds": len(stochastic_predictions),
                "next_seed": INFERENCE_SEEDS[index + 1] if index + 1 < len(INFERENCE_SEEDS) else None,
                "updated_at_utc": runner.utc_now(), "inner_dev_numeric_rows_loaded": False})
            if index == 0:
                elapsed_package = precharge + (time.perf_counter() - started_wall) + (time.process_time() - started_cpu)
                seed_cost = max(elapsed, 1e-6)
                # Include 15 stochastic passes, deterministic TRAIN, a potential 16-pass DEV ensemble,
                # selection/bootstrap, serialization, independent readback and fixed overhead.
                fixed_overhead = float(config["execution"]["fixed_post_inference_and_readback_overhead_seconds"])
                projected = elapsed_package + seed_cost * 32 + fixed_overhead
                projected_with_margin = projected * (1.0 + float(config["authorization"]["margin_fraction"]))
                gate = {"status": "PASS_CONTINUE_FULL_FIXED_PACKAGE" if
                        projected_with_margin <= min(new_cap, inherited_remaining)
                        else "FAIL_CLOSE_AFTER_FIRST_FORMAL_INFERENCE",
                    "first_inference_seed": seed, "first_inference_wall_plus_process_seconds": elapsed,
                    "package_elapsed_plus_precharge_seconds": elapsed_package,
                    "remaining_new_package_cap_seconds": new_cap - elapsed_package,
                    "remaining_inherited_exp059_cap_seconds": inherited_remaining - elapsed_package,
                    "projected_full_package_seconds_before_margin": projected,
                    "projected_full_package_seconds_with_20pct_margin": projected_with_margin,
                    "projection_components": {"remaining_train_random_passes": 15,
                        "deterministic_train_pass": 1, "potential_selected_dev_mc16_passes": 16,
                        "fixed_overhead_seconds": fixed_overhead},
                    "caps": {"new_package_seconds": new_cap,
                        "inherited_exp059_remaining_seconds": inherited_remaining},
                    "inner_dev_numeric_rows_loaded": False, "gpu_hours": 0.0}
                atomic_json(attempt / "first_inference_cost_gate.json", gate)
                if gate["status"] != "PASS_CONTINUE_FULL_FIXED_PACKAGE":
                    atomic_json(attempt / "train_selection_receipt.json", {
                        "status": "CLOSED_BEFORE_SELECTION_COST_GATE_FAILED",
                        "experiment_id": "EXP-060", "run_id": run_id,
                        "attempt_id": attempt.name, "selected_candidate": None,
                        "inner_dev_numeric_rows_loaded": False,
                        "selection_used_inner_dev": False,
                        "cost_gate_path": "first formal seed retained; full fixed comparison not run",
                        "frozen_at_utc": runner.utc_now()})
                    state_after = exact_state_equal(torch, model, frozen_state)
                    if not state_after["exact"]:
                        raise RuntimeError("model state changed during the cost-gated first inference")
                    state_after["bn_buffer_sha256"] = tensor_tree_hash(snapshot_model_state(model)[2])
                    state_after["bn_buffer_count"] = len(frozen_bn)
                    atomic_json(attempt / "model_state_after_inference.json", state_after)
                    result = {"status": gate["status"], "experiment_id": "EXP-060", "run_id": run_id,
                        "attempt_id": attempt.name, "first_formal_seed": seed,
                        "first_seed_prediction_sha256": sha256(file), "cost_gate": gate,
                        "inner_dev_numeric_rows_loaded": False, "valid_full_comparison": False,
                        "science_interpretation": "A single fixed inference is insufficient for the frozen 16-seed comparison."}
                    atomic_json(attempt / "train_comparison_result.json", result)
                    return close_attempt(attempt, ledger, result, started_wall, started_cpu,
                                         precharge, "CLOSED_COST_GATE_NO_FULL_COMPARISON")
                append_jsonl(ledger, {"event": "FIRST_INFERENCE_COST_GATE_PASS", "at_utc": runner.utc_now(),
                    "projected_seconds_with_margin": projected_with_margin, "gpu_hours": 0.0})

        train_baseline = read_json(context["train_baseline"])
        drug_pred, zero_pred = baseline_predictions(keys, train_y)
        baseline_metrics = {
            "per_drug_mean_delta": runner.keyed_metrics(keys, train_y, drug_pred, gene_variance),
            "zero_change": runner.keyed_metrics(keys, train_y, zero_pred, gene_variance)}
        for name, computed in baseline_metrics.items():
            registered = train_baseline["metrics"][name]
            if abs(float(computed["object_macro_nmse"]) - float(registered["object_macro_nmse"])) > 1e-10:
                raise RuntimeError(f"reconstructed existing TRAIN baseline changed for {name}")
        baseline_drug_obj_ids, baseline_drug_obj = object_macro_nmse(keys, train_y, drug_pred, gene_variance)
        baseline_zero_obj_ids, baseline_zero_obj = object_macro_nmse(keys, train_y, zero_pred, gene_variance)
        if baseline_drug_obj_ids != baseline_zero_obj_ids:
            raise RuntimeError("TRAIN baseline object axes differ")
        per_object_rows.extend({"method": "per_drug_mean_delta", "source_object_id": obj,
                                "object_macro_nmse": float(value)}
                               for obj, value in zip(baseline_drug_obj_ids, baseline_drug_obj))
        per_object_rows.extend({"method": "zero_change", "source_object_id": obj,
                                "object_macro_nmse": float(value)}
                               for obj, value in zip(baseline_zero_obj_ids, baseline_zero_obj))
        candidate_predictions: dict[str, np.ndarray] = {}
        candidate_metrics: dict[str, Any] = {}
        candidate_inference: dict[str, Any] = {}
        random_reference_objects = None
        random_reference_nmse = None
        if len(stochastic_predictions) == len(INFERENCE_SEEDS):
            stack = np.stack(stochastic_predictions, axis=0).astype(np.float32, copy=False)
            mc_mean = stack.mean(axis=0, dtype=np.float64).astype(np.float32)
            if not np.isfinite(mc_mean).all():
                raise InferenceFailure("non-finite mean of fixed 16 stochastic predictions",
                    {"full_denominator": 419 * 978,
                     "finite_n": int(np.isfinite(mc_mean).sum()),
                     "nonfinite_n": int(mc_mean.size - np.isfinite(mc_mean).sum())})
            candidate_predictions["mc16_mean"] = mc_mean
            candidate_metrics["mc16_mean"] = runner.keyed_metrics(keys, train_y, mc_mean, gene_variance)
            random_reference_nmse = float(np.mean([item["metric"]["object_macro_nmse"]
                for item in stochastic_metrics], dtype=np.float64))
            object_id_order, seed_object_metrics = object_macro_nmse(keys, train_y,
                stochastic_predictions[0], gene_variance)
            seed_obj_stack = []
            for pred in stochastic_predictions:
                obj_ids, obj_values = object_macro_nmse(keys, train_y, pred, gene_variance)
                if obj_ids != object_id_order:
                    raise RuntimeError("source object order changed among fixed stochastic seeds")
                seed_obj_stack.append(obj_values)
            random_reference_objects = np.mean(np.stack(seed_obj_stack, axis=0), axis=0)
            candidate_file = attempt / "mc16_mean_train_keyed_predictions.npz"
            save_keyed(candidate_file, keys, values["genes"], train_y, mc_mean,
                       role="TRAIN", method="mc16_mean", seed=None)
            per_drug_rows.extend({"method": "mc16_mean", **item}
                                 for item in metric_per_drug(keys, train_y, mc_mean, gene_variance))
            obj_ids, obj_values = object_macro_nmse(keys, train_y, mc_mean, gene_variance)
            per_object_rows.extend({"method": "mc16_mean", "source_object_id": obj,
                                    "object_macro_nmse": float(value)}
                                   for obj, value in zip(obj_ids, obj_values))
            pred_var = np.var(stack.astype(np.float64), axis=0, ddof=0)
            standardized_var = pred_var[:, positive] / gene_variance[positive][None, :]
            standardized_sd = np.sqrt(standardized_var)
            variance_summary = {"population_variance_ddof0": True,
                "mean_predictive_variance_over_train_gene_variance": float(np.mean(standardized_var)),
                "median_predictive_variance_over_train_gene_variance": float(np.median(standardized_var)),
                "mean_standardized_predictive_sd": float(np.mean(standardized_sd)),
                "condition_gene_denominator": int(standardized_var.size),
                "interpretation": "fixed 16 inference draws on these 419 source conditions; not biological replication"}
            single_minus_mean_nmse = random_reference_nmse - float(candidate_metrics["mc16_mean"]["object_macro_nmse"])
            candidate_metrics["mc16_mean"]["nmse_gain_vs_random_single_reference"] = (
                (random_reference_nmse - float(candidate_metrics["mc16_mean"]["object_macro_nmse"])) /
                random_reference_nmse if random_reference_nmse > 0 else None)
            candidate_metrics["mc16_mean"]["nmse_gain_vs_per_drug_mean"] = (
                (float(baseline_metrics["per_drug_mean_delta"]["object_macro_nmse"])
                 - float(candidate_metrics["mc16_mean"]["object_macro_nmse"])) /
                float(baseline_metrics["per_drug_mean_delta"]["object_macro_nmse"]))
            candidate_metrics["mc16_mean"]["passes_amplitude_gate"] = (
                0.5 <= float(candidate_metrics["mc16_mean"]["object_macro_amplitude_ratio"]) <= 2.0)
            candidate_inference["mc16_mean"] = {"prediction_path": candidate_file.name,
                "prediction_sha256": sha256(candidate_file), "n_random_predictions": 16}
            variance_summary["random_single_reference_object_macro_nmse"] = random_reference_nmse
            variance_summary["mc16_mean_prediction_object_macro_nmse"] = candidate_metrics["mc16_mean"]["object_macro_nmse"]
            variance_summary["single_reference_minus_mc16_mean_nmse"] = single_minus_mean_nmse
            atomic_json(attempt / "stochastic_prediction_variance.json", variance_summary)
        else:
            atomic_json(attempt / "stochastic_prediction_variance.json", {
                "status": "UNAVAILABLE_FIXED_STOCHASTIC_ARM_FAILED",
                "failure": stochastic_failure, "full_seed_count_required": 16})

        before_det_wall, before_det_cpu = time.perf_counter(), time.process_time()
        try:
            mu_pred, mu_evidence = run_inference(torch, model, values["X_ctl"], train_features,
                seed=MODEL_SEED, deterministic_mu=True, batch_size=int(config["model"]["batch_size"]),
                progress_check=lambda: resource_guard("deterministic TRAIN inference"))
            mu_file = attempt / "mu_both_levels_train_keyed_predictions.npz"
            save_keyed(mu_file, keys, values["genes"], train_y, mu_pred,
                       role="TRAIN", method="mu_both_levels", seed=None)
            candidate_predictions["mu_both_levels"] = mu_pred
            candidate_metrics["mu_both_levels"] = runner.keyed_metrics(keys, train_y, mu_pred, gene_variance)
            candidate_inference["mu_both_levels"] = {"prediction_path": mu_file.name,
                "prediction_sha256": sha256(mu_file), **mu_evidence}
            per_drug_rows.extend({"method": "mu_both_levels", **item}
                                 for item in metric_per_drug(keys, train_y, mu_pred, gene_variance))
            obj_ids, obj_values = object_macro_nmse(keys, train_y, mu_pred, gene_variance)
            per_object_rows.extend({"method": "mu_both_levels", "source_object_id": obj,
                                    "object_macro_nmse": float(value)}
                                   for obj, value in zip(obj_ids, obj_values))
            if random_reference_nmse is not None:
                nmse = float(candidate_metrics["mu_both_levels"]["object_macro_nmse"])
                candidate_metrics["mu_both_levels"]["nmse_gain_vs_random_single_reference"] = (
                    (random_reference_nmse - nmse) / random_reference_nmse if random_reference_nmse > 0 else None)
                drug_nmse = float(baseline_metrics["per_drug_mean_delta"]["object_macro_nmse"])
                candidate_metrics["mu_both_levels"]["nmse_gain_vs_per_drug_mean"] = (
                    (drug_nmse - nmse) / drug_nmse if drug_nmse > 0 else None)
                amp = candidate_metrics["mu_both_levels"]["object_macro_amplitude_ratio"]
                candidate_metrics["mu_both_levels"]["passes_amplitude_gate"] = (
                    amp is not None and 0.5 <= float(amp) <= 2.0)
        except InferenceFailure as exc:
            candidate_inference["mu_both_levels"] = {"status": "FAILED_CLOSED_NONFINITE",
                "failure": exc.evidence, "exception": str(exc)}
        deterministic_elapsed = (time.perf_counter() - before_det_wall) + (time.process_time() - before_det_cpu)
        append_jsonl(ledger, {"event": "DETERMINISTIC_MU_BOTH_LEVELS_COMPLETED"
            if "mu_both_levels" in candidate_predictions else "DETERMINISTIC_MU_BOTH_LEVELS_FAILED",
            "at_utc": runner.utc_now(), "wall_plus_process_seconds": deterministic_elapsed,
            "full_denominator": 419 * 978, "gpu_hours": 0.0})

        # TRAIN baselines are reconstructed with the already frozen definition and checked against the old artifact.
        random_reference_pass = len(stochastic_predictions) == len(INFERENCE_SEEDS)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        paired: list[dict[str, Any]] = []
        if random_reference_pass and random_reference_objects is not None:
            for candidate_name, pred in candidate_predictions.items():
                ids, object_values = object_macro_nmse(keys, train_y, pred, gene_variance)
                if ids != object_id_order:
                    raise RuntimeError("candidate object axis differs from the stochastic reference")
                paired.append({"role": "TRAIN", "candidate": candidate_name,
                    "reference": "mean_of_16_single_random_object_losses",
                    **bootstrap_paired_diff(random_reference_objects, object_values, rng)})
                paired.append({"role": "TRAIN", "candidate": candidate_name,
                    "reference": "TRAIN_per_full_inchikey_mean",
                    **bootstrap_paired_diff(baseline_drug_obj, object_values, rng)})
                paired.append({"role": "TRAIN", "candidate": candidate_name,
                    "reference": "zero_change",
                    **bootstrap_paired_diff(baseline_zero_obj, object_values, rng)})
        bootstrap_path = attempt / "paired_object_bootstrap.tsv"
        write_tsv(bootstrap_path, paired)
        write_tsv(attempt / "train_per_drug_metrics.tsv", per_drug_rows)
        write_tsv(attempt / "train_object_metrics.tsv", per_object_rows)

        thresholds = config["selection"]
        passing: list[str] = []
        selection_tests: dict[str, Any] = {}
        for name, metric in candidate_metrics.items():
            single_gain = metric.get("nmse_gain_vs_random_single_reference")
            mean_gain = metric.get("nmse_gain_vs_per_drug_mean")
            amp = metric.get("object_macro_amplitude_ratio")
            rand_pass = single_gain is not None and float(single_gain) >= float(
                thresholds["candidate_min_nmse_reduction_vs_random_reference"])
            mean_pass = mean_gain is not None and float(mean_gain) >= float(
                thresholds["candidate_min_nmse_reduction_vs_train_per_full_inchikey_mean"])
            amp_pass = amp is not None and 0.5 <= float(amp) <= 2.0
            selection_tests[name] = {"nmse_gain_vs_random_single_reference": single_gain,
                "nmse_gain_vs_train_per_drug_mean": mean_gain, "amplitude_ratio": amp,
                "passes_random_reference_10pct_gate": rand_pass,
                "passes_per_drug_mean_10pct_gate": mean_pass,
                "passes_amplitude_0.5_to_2_gate": amp_pass,
                "passes_all_train_gates": bool(rand_pass and mean_pass and amp_pass)}
            if rand_pass and mean_pass and amp_pass:
                passing.append(name)
        selected: str | None = None
        if passing:
            selected = min(passing, key=lambda name: (
                float(candidate_metrics[name]["object_macro_nmse"]),
                0 if name == "mu_both_levels" else 1))
        selection_receipt = {"status": "FROZEN_TRAIN_SELECTION",
            "experiment_id": "EXP-060", "run_id": run_id, "attempt_id": attempt.name,
            "candidate_tests": selection_tests, "passing_candidates": passing,
            "selected_candidate": selected, "selection_rule": "lowest TRAIN object-macro NMSE among candidates passing all frozen gates; exact tie favors mu_both_levels",
            "random_reference": {"n_fixed_seeds_completed": len(stochastic_metrics),
                "fixed_seed_values": [row["seed"] for row in stochastic_metrics],
                "definition": "arithmetic mean of single-seed object-macro NMSE, not the best seed",
                "object_macro_nmse": random_reference_nmse},
            "train_only_candidate_metrics": candidate_metrics,
            "baseline_train_metrics_reused_and_recomputed": baseline_metrics,
            "baseline_artifact_path": str(context["train_baseline"].relative_to(ROOT)),
            "baseline_artifact_sha256": context["train_baseline_sha256"],
            "inner_dev_numeric_rows_loaded": False, "selection_used_inner_dev": False,
            "pdo_outer_locked_numeric_values_opened": False, "panacea_numeric_values_opened": False,
            "frozen_at_utc": runner.utc_now()}
        atomic_json(attempt / "train_selection_receipt.json", selection_receipt)

        dev_result: dict[str, Any] | None = None
        dev_access_evidence: dict[str, Any] = {"inner_dev_numeric_rows_loaded": False}
        dev_cost = 0.0
        if selected is not None:
            dev_started_wall, dev_started_cpu = time.perf_counter(), time.process_time()
            # This is the first point at which EXP-060 opens inner-dev numeric rows.
            source_run = runner.SOURCE_RUN_DIR
            dev_keys, dev_values, dev_evidence = runner._load_exp055_inner_dev_after_checkpoint_freeze(
                keys, values, source_run)
            # Do not even deserialize prior inner-dev scores until the current TRAIN-only
            # candidate selection receipt is frozen and exactly one candidate has passed.
            dev_eval = read_json(context["dev_recovery"])
            dev_root_result = read_json(context["dev_root_result_path"])
            if (sha256(context["dev_recovery"]) != context["dev_recovery_sha256"]
                    or sha256(context["dev_root_result_path"]) != context["dev_root_receipt_sha256"]
                    or dev_eval.get("checkpoint_sha256") != context["checkpoint_sha256"]
                    or dev_eval.get("checkpoint_epoch") != 300
                    or dev_eval.get("data_access", {}).get("inner_dev_values_loaded_after_epoch_300_checkpoint_freeze") is not True
                    or dev_root_result.get("status") != "PASS_ROOT_EXP059_FIXED300_POSTCHECKPOINT_EVALUATION_READBACK"):
                raise RuntimeError("existing EXP-059 inner-dev recovery is not root-verified against this checkpoint")
            if (len(dev_keys) != 93 or dev_keys.source_object_id.nunique() != 24
                    or dev_evidence.get("object_overlap") != 0):
                raise ValueError("frozen inner-dev identity or object-disjoint contract changed")
            dev_y = dev_values["delta"].astype(np.float32, copy=False)
            dev_features = runner._full_key_features(
                dev_keys, np.arange(len(dev_keys), dtype=np.int64), feature_by_key)
            selected_predictions: np.ndarray
            selected_inference: list[dict[str, Any]] = []
            if selected == "mu_both_levels":
                selected_predictions, ev = run_inference(torch, model, dev_values["X_ctl"], dev_features,
                    seed=MODEL_SEED, deterministic_mu=True, batch_size=int(config["model"]["batch_size"]),
                    progress_check=lambda: resource_guard("selected deterministic inner-dev inference"))
                selected_inference.append(ev)
            else:
                dev_draws = []
                for seed in INFERENCE_SEEDS:
                    pred, ev = run_inference(torch, model, dev_values["X_ctl"], dev_features,
                        seed=seed, deterministic_mu=False, batch_size=int(config["model"]["batch_size"]),
                        progress_check=lambda: resource_guard(f"selected inner-dev random seed {seed}"))
                    dev_draws.append(pred)
                    selected_inference.append(ev)
                selected_predictions = np.mean(np.stack(dev_draws, axis=0).astype(np.float64),
                                              axis=0).astype(np.float32)
                if not np.isfinite(selected_predictions).all():
                    raise InferenceFailure("non-finite selected mc16 mean on frozen inner-dev",
                        {"full_denominator": 93 * 978,
                         "finite_n": int(np.isfinite(selected_predictions).sum()),
                         "nonfinite_n": int(selected_predictions.size - np.isfinite(selected_predictions).sum())})
            dev_state = exact_state_equal(torch, model, frozen_state)
            if not dev_state["exact"]:
                raise RuntimeError("model/BN state changed during frozen inner-dev inference")
            dev_metric = runner.keyed_metrics(dev_keys, dev_y, selected_predictions, gene_variance)
            dev_baselines = runner._exp055_train_and_dev_baselines(
                keys, train_y, dev_keys, dev_y, gene_variance)
            existing_dev_metrics = dev_eval["metrics_by_role"]["inner_dev"]
            for baseline_name, registered_name in (("per_drug_mean_delta_baseline", "per_drug_mean_delta"),
                                                    ("zero_change_baseline", "zero_change")):
                if abs(float(dev_baselines["metrics"][registered_name]["inner_dev"]["object_macro_nmse"])
                       - float(existing_dev_metrics[baseline_name]["object_macro_nmse"])) > 1e-10:
                    raise RuntimeError(f"recomputed existing inner-dev baseline changed: {registered_name}")
            dev_file = attempt / f"{selected}_inner_dev_keyed_predictions.npz"
            save_keyed(dev_file, dev_keys, dev_values["genes"], dev_y, selected_predictions,
                       role="inner_dev", method=selected, seed=None)
            dev_metric_per_drug = metric_per_drug(dev_keys, dev_y, selected_predictions, gene_variance)
            dev_ids, dev_obj = object_macro_nmse(dev_keys, dev_y, selected_predictions, gene_variance)
            drug_dev_prediction = np.stack([
                train_y[keys.full_inchikey.astype(str).to_numpy() == str(key)].mean(
                    axis=0, dtype=np.float64).astype(np.float32)
                for key in dev_keys.full_inchikey.astype(str).to_numpy()])
            zero_dev_prediction = np.zeros_like(dev_y, dtype=np.float32)
            _, drug_dev_obj = object_macro_nmse(dev_keys, dev_y, drug_dev_prediction, gene_variance)
            _, zero_dev_obj = object_macro_nmse(dev_keys, dev_y, zero_dev_prediction, gene_variance)
            dev_bootstrap = [
                {"candidate": selected, "reference": "inner_dev_per_full_inchikey_mean",
                 **bootstrap_paired_diff(drug_dev_obj, dev_obj, rng)},
                {"candidate": selected, "reference": "inner_dev_zero_change",
                 **bootstrap_paired_diff(zero_dev_obj, dev_obj, rng)},
            ]
            dev_bootstrap_path = attempt / "inner_dev_paired_object_bootstrap.tsv"
            write_tsv(dev_bootstrap_path, dev_bootstrap)
            write_tsv(attempt / "inner_dev_per_drug_metrics.tsv",
                      [{"method": selected, **row} for row in dev_metric_per_drug])
            write_tsv(attempt / "inner_dev_object_metrics.tsv",
                      [{"method": selected, "source_object_id": obj, "object_macro_nmse": float(value)}
                       for obj, value in zip(dev_ids, dev_obj)])
            dev_result = {"status": "SCORED_ONE_TRAIN_SELECTED_CANDIDATE_ONLY",
                "selected_candidate": selected, "n_conditions": len(dev_keys),
                "n_source_objects": int(dev_keys.source_object_id.nunique()), "object_overlap": 0,
                "metrics": {"candidate": dev_metric,
                    "per_drug_mean_delta_baseline": existing_dev_metrics["per_drug_mean_delta_baseline"],
                    "zero_change_baseline": existing_dev_metrics["zero_change_baseline"]},
                "selected_candidate_prediction_path": dev_file.name,
                "selected_candidate_prediction_sha256": sha256(dev_file),
                "per_drug_metrics_path": "inner_dev_per_drug_metrics.tsv",
                "paired_bootstrap_path": dev_bootstrap_path.name,
                "inference_evidence": selected_inference,
                "baseline_reused_from_existing_root_verified_evaluation": {
                    "path": str(context["dev_recovery"].relative_to(ROOT)),
                    "sha256": context["dev_recovery_sha256"]},
                "target_domain_fit_scaling_applied": False,
                "candidate_meaning": "development score for one preselected inference rule; not independent validation",
                "inner_dev_numeric_rows_loaded": True,
                "inner_dev_used_for_candidate_or_seed_selection": False}
            dev_access_evidence = {"inner_dev_numeric_rows_loaded": True,
                "loaded_after_train_selection_receipt": True,
                "selected_candidate_only": selected,
                "condition_count": 93, "object_count": 24, "object_overlap": 0}
            dev_cost = (time.perf_counter() - dev_started_wall) + (time.process_time() - dev_started_cpu)
            atomic_json(attempt / "inner_dev_result.json", dev_result)
            append_jsonl(ledger, {"event": "ONE_TRAIN_SELECTED_INNER_DEV_CANDIDATE_SCORED",
                "at_utc": runner.utc_now(), "selected_candidate": selected,
                "wall_plus_process_seconds": dev_cost, "conditions": 93, "objects": 24,
                "gpu_hours": 0.0})

        state_after = exact_state_equal(torch, model, frozen_state)
        if not state_after["exact"]:
            raise RuntimeError(f"model or BN tensor state changed during read-only inference: {state_after}")
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA initialized during the CPU-only inference package")
        state_after["bn_buffer_sha256"] = tensor_tree_hash(snapshot_model_state(model)[2])
        state_after["bn_buffer_count"] = len(frozen_bn)
        atomic_json(attempt / "model_state_after_inference.json", state_after)
        if tensor_tree_hash(frozen_bn) != tensor_tree_hash(snapshot_model_state(model)[2]):
            raise RuntimeError("BatchNorm buffers changed during read-only inference")

        train_candidate_files = {name: meta for name, meta in candidate_inference.items()}
        report = {
            "status": ("COMPLETED_TRAIN_SELECTION_AND_ONE_DEV_SCORE" if selected is not None else
                       "COMPLETED_TRAIN_ONLY_NO_CANDIDATE_PASSED" if stochastic_failure is None else
                       "PARTIAL_RANDOM_ARM_FAILED_DETERMINISTIC_ONLY_NO_DEV"),
            "experiment_id": "EXP-060", "run_id": run_id, "attempt_id": attempt.name,
            "checkpoint_path": str(context["checkpoint"].relative_to(ROOT)),
            "checkpoint_sha256": context["checkpoint_sha256"], "checkpoint_epoch": 300,
            "checkpoint_role": "frozen_fixed_epoch_300_main_endpoint",
            "model_state_before_sha256": frozen_state_sha,
            "model_state_after_sha256": state_after["state_tree_sha256"],
            "model_state_exactly_unchanged": state_after["exact"],
            "bn_buffer_count": len(frozen_bn), "bn_buffers_exactly_unchanged": True,
            "optimizer_loaded": False, "optimizer_updates": 0, "batchnorm_updates": 0,
            "model_eval": module_modes_eval(model), "no_grad": True, "device": "cpu", "gpu_hours": 0.0,
            "train_denominator": {"conditions": 419, "objects": 110, "full_inchikeys": 5, "genes": 978,
                "finite_main_delta_per_complete_prediction": 419 * 978},
            "fixed_inference_seed_records": stochastic_metrics,
            "fixed_random_seed_failure": stochastic_failure,
            "valid_full_comparison": stochastic_failure is None,
            "primary_random_reference": {"definition": "equal arithmetic mean of 16 individual single-seed object-macro NMSEs; no best seed selection",
                "object_macro_nmse": random_reference_nmse,
                "n_fixed_seed_predictions": len(stochastic_metrics),
                "per_seed_predictions_retained": len(stochastic_predictions) == len(INFERENCE_SEEDS)},
            "candidate_train_metrics": candidate_metrics,
            "candidate_train_prediction_files": train_candidate_files,
            "baseline_train_metrics_reused_and_recomputed": baseline_metrics,
            "paired_bootstrap_train_path": bootstrap_path.name,
            "paired_bootstrap_train_sha256": sha256(bootstrap_path),
            "stochastic_prediction_variance": read_json(attempt / "stochastic_prediction_variance.json"),
            "selection_receipt_path": "train_selection_receipt.json",
            "selection_receipt_sha256": sha256(attempt / "train_selection_receipt.json"),
            "selected_candidate": selected, "inner_dev": dev_result,
            "data_access": {"train_numeric_rows_loaded_for_selection": True,
                "inner_dev_loaded_only_after_selection_freeze": dev_access_evidence["inner_dev_numeric_rows_loaded"],
                "inner_dev_candidate_or_seed_selection": False,
                "pdo_outer_locked_numeric_values_opened": False, "panacea_numeric_values_opened": False},
            "input_evidence": input_evidence,
            "runner_sha256": sha256(Path(__file__).resolve()),
            "config_sha256": context["config_sha256"],
            "parent_checkpoint_sha256_after_inference": sha256(context["checkpoint"]),
            "parent_checkpoint_unchanged": sha256(context["checkpoint"]) == context["checkpoint_sha256"],
            "cpu_elapsed_wall_seconds": time.perf_counter() - started_wall,
            "cpu_process_seconds": time.process_time() - started_cpu,
            "inference_wall_plus_process_seconds": inference_costs,
            "deterministic_inference_wall_plus_process_seconds": deterministic_elapsed,
            "inner_dev_wall_plus_process_seconds": dev_cost,
            "cpu_package_precharge_seconds": precharge,
            "cpu_new_package_cap_seconds": new_cap,
            "cpu_inherited_exp059_remaining_cap_seconds": inherited_remaining,
            "cpu_cap_projection_with_margin_seconds": None,
            "selection_use_of_bootstrap": False,
        }
        # Record the observed total and a post-hoc full-package estimate for accounting.
        elapsed_charge = precharge + report["cpu_elapsed_wall_seconds"] + report["cpu_process_seconds"]
        report["cpu_charged_conservative_seconds"] = elapsed_charge
        remaining_hypothetical = 0.0
        if len(stochastic_predictions) == len(INFERENCE_SEEDS):
            remaining_hypothetical = float(np.mean(inference_costs) * 16)
        projection = (elapsed_charge + remaining_hypothetical + 120.0) * 1.2
        report["cpu_cap_projection_with_margin_seconds"] = projection
        report["cpu_cost_gate_full_package_with_margin_pass"] = (
            projection <= new_cap and elapsed_charge <= inherited_remaining)
        report["science_interpretation"] = (
            "The fixed checkpoint was evaluated under the predeclared inference rules; this does not establish a retrained model result or PDO transfer."
            if selected is not None else
            "No TRAIN candidate passed the frozen mechanism and baseline gates; no inner-dev numbers were accessed.")
        atomic_json(attempt / "comparison_result.json", report)
        return close_attempt(attempt, ledger, report, started_wall, started_cpu, precharge,
            "COMPLETED_TRAIN_SELECTION_AND_ONE_DEV_SCORE" if selected is not None else
            "COMPLETED_TRAIN_ONLY_NO_CANDIDATE_PASSED" if stochastic_failure is None else
            "PARTIAL_RANDOM_ARM_FAILED_DETERMINISTIC_ONLY_NO_DEV")
    except Exception as exc:
        elapsed_wall = time.perf_counter() - started_wall
        elapsed_cpu = time.process_time() - started_cpu
        failure = {"status": "FAILED_CLOSED", "experiment_id": "EXP-060", "run_id": run_id,
            "attempt_id": attempt.name, "exception_type": type(exc).__name__, "exception": str(exc),
            "traceback": traceback.format_exc(), "cpu_elapsed_wall_seconds": elapsed_wall,
            "cpu_process_seconds": elapsed_cpu, "precharge_seconds": precharge,
            "cpu_charged_conservative_seconds": precharge + elapsed_wall + elapsed_cpu,
            "device": "cpu", "gpu_hours": 0.0,
            "inner_dev_numeric_rows_loaded": (attempt / "inner_dev_result.json").is_file(),
            "pdo_outer_locked_numeric_values_opened": False, "panacea_numeric_values_opened": False}
        atomic_json(attempt / "failure_receipt.json", failure)
        append_jsonl(ledger, {"event": "PACKAGE_FAILED_CLOSED", "at_utc": runner.utc_now(),
            "exception_type": type(exc).__name__, "exception": str(exc),
            "wall_seconds": elapsed_wall, "process_seconds": elapsed_cpu,
            "conservative_charge_seconds": failure["cpu_charged_conservative_seconds"],
            "gpu_hours": 0.0})
        atomic_json(attempt / "attempt_close_receipt.json", failure)
        raise


def close_attempt(attempt: Path, ledger: Path, result: dict[str, Any], started_wall: float,
                  started_cpu: float, precharge: float, status: str) -> dict[str, Any]:
    wall = time.perf_counter() - started_wall
    cpu = time.process_time() - started_cpu
    charged = precharge + wall + cpu
    close = {"status": status, "run_id": attempt.parent.name, "attempt_id": attempt.name,
        "closed_at_utc": runner.utc_now(), "device": "cpu", "cuda_visible_devices": "",
        "gpu_hours": 0.0, "cpu_wall_seconds": wall, "cpu_process_seconds": cpu,
        "precharge_seconds": precharge, "conservative_charge_seconds": charged,
        "new_package_cap_seconds": float(result.get("cpu_new_package_cap_seconds", 1800)),
        "inherited_exp059_remaining_cap_seconds": float(result.get("cpu_inherited_exp059_remaining_cap_seconds", 4389.796162049892)),
        "fixed_comparison_valid": bool(result.get("valid_full_comparison", True)),
        "selected_candidate": result.get("selected_candidate"),
        "inner_dev_numeric_rows_loaded": bool(result.get("data_access", {}).get(
            "inner_dev_loaded_only_after_selection_freeze", False)),
        "pdo_outer_locked_numeric_values_opened": False}
    atomic_json(attempt / "attempt_close_receipt.json", close)
    append_jsonl(ledger, {"event": "PACKAGE_CLOSED", "at_utc": runner.utc_now(),
        "status": status, "wall_seconds": wall, "process_seconds": cpu,
        "conservative_charge_seconds": charged, "gpu_hours": 0.0})
    atomic_json(attempt / "attempt_progress.json", {"phase": status,
        "updated_at_utc": runner.utc_now(), "active_pid": None,
        "cpu_conservative_charge_seconds": charged, "gpu_hours": 0.0})
    print(json.dumps({"status": status, "attempt": str(attempt), "cpu_charge_seconds": charged,
                      "gpu_hours": 0.0, "selected_candidate": result.get("selected_candidate")},
                     ensure_ascii=False), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=["cpu"], required=True)
    args = parser.parse_args()
    result = run_package(Path(args.config).resolve(), args.run_id, args.device)
    print(json.dumps({"status": result.get("status"), "selected_candidate": result.get("selected_candidate")},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
