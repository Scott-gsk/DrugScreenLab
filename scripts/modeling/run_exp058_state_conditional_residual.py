"""Run the preregistered EXP-058 source-development comparison (CPU only)."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/modeling"))
import exp051_contract_fit_smoke as fit  # noqa: E402
import run_exp056_low_rank_state_drug_interaction as source  # noqa: E402
import run_exp053_drug_shared_state_residual as exp053  # noqa: E402

ARMS = ("nonlinear_state_residual", "capacity_matched_drug_only", "shuffled_state")
ALL_METHODS = ("zero_change", "per_drug_mean", "linear_state_residual", *ARMS)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def object_state_rows(keys: pd.DataFrame, x_ctl: np.ndarray) -> dict[str, np.ndarray]:
    ids = keys["source_object_id"].astype(str).to_numpy()
    if x_ctl.shape != (len(keys), 978) or not np.isfinite(x_ctl).all():
        raise ValueError("X_ctl state matrix must be finite and match the frozen 978-gene axis")
    return {obj: x_ctl[np.flatnonzero(ids == obj)].mean(axis=0, dtype=np.float64)
            for obj in sorted(set(ids))}


def derangement(ids: list[str], seed: int) -> dict[str, str]:
    ordered = sorted(set(map(str, ids)))
    if len(ordered) < 2:
        raise ValueError("object-block shuffle requires at least two objects per split")
    rng = np.random.default_rng(seed)
    perm = np.asarray(ordered, dtype=str)[rng.permutation(len(ordered))]
    offset = int(rng.integers(1, len(ordered)))
    mapping = {str(perm[i]): str(perm[(i + offset) % len(ordered)]) for i in range(len(ordered))}
    if any(k == v for k, v in mapping.items()) or set(mapping) != set(mapping.values()):
        raise RuntimeError("shuffle mapping is not a derangement/permutation")
    return mapping


def make_state_features(train_keys: pd.DataFrame, dev_keys: pd.DataFrame,
                        train_ctl: np.ndarray, dev_ctl: np.ndarray, *,
                        n_components: int, shuffle_seed: int | None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from sklearn.decomposition import PCA

    train_states = object_state_rows(train_keys, train_ctl)
    dev_states = object_state_rows(dev_keys, dev_ctl)
    train_ids = sorted(train_states)
    dev_ids = sorted(dev_states)
    train_map = {obj: obj for obj in train_ids}
    dev_map = {obj: obj for obj in dev_ids}
    if shuffle_seed is not None:
        train_map = derangement(train_ids, shuffle_seed)
        dev_map = derangement(dev_ids, shuffle_seed + 1)
    train_x = np.stack([train_states[train_map[o]] for o in train_ids]).astype(np.float64)
    mean = train_x.mean(axis=0, dtype=np.float64)
    scale = train_x.std(axis=0, ddof=0)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    standardized = (train_x - mean) / scale
    if n_components > min(standardized.shape):
        raise ValueError("registered PCA dimension exceeds TRAIN object capacity")
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(standardized)
    train_projection = pca.transform(standardized).astype(np.float32)
    dev_x = np.stack([dev_states[dev_map[o]] for o in dev_ids]).astype(np.float64)
    dev_projection = pca.transform((dev_x - mean) / scale).astype(np.float32)
    tr_position = {obj: i for i, obj in enumerate(train_ids)}
    dv_position = {obj: i for i, obj in enumerate(dev_ids)}
    train_rows = np.stack([train_projection[tr_position[o]]
                           for o in train_keys.source_object_id.astype(str)]).astype(np.float32)
    dev_rows = np.stack([dev_projection[dv_position[o]]
                         for o in dev_keys.source_object_id.astype(str)]).astype(np.float32)
    receipt = {
        "fit_role": "TRAIN source_object_id object-mean X_ctl only",
        "components": int(n_components), "shuffle_seed": shuffle_seed,
        "train_objects": len(train_ids), "inner_dev_objects": len(dev_ids),
        "train_inner_dev_object_overlap": len(set(train_ids) & set(dev_ids)),
        "train_state_to_donor": train_map, "inner_dev_state_to_donor": dev_map,
        "train_mapping_sha256": hashlib.sha256(json.dumps(train_map, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(),
        "inner_dev_mapping_sha256": hashlib.sha256(json.dumps(dev_map, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(),
        "scaler_mean_sha256": hashlib.sha256(mean.astype("<f8").tobytes()).hexdigest(),
        "scaler_scale_sha256": hashlib.sha256(scale.astype("<f8").tobytes()).hexdigest(),
        "pca_components_sha256": hashlib.sha256(pca.components_.astype("<f8").tobytes()).hexdigest(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
    }
    return train_rows, dev_rows, receipt


def per_drug_predictions(train_keys: pd.DataFrame, train_y: np.ndarray,
                         eval_keys: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    train_drugs = train_keys.full_inchikey.astype(str).to_numpy()
    eval_drugs = eval_keys.full_inchikey.astype(str).to_numpy()
    means = {drug: train_y[train_drugs == drug].mean(axis=0, dtype=np.float64)
             for drug in sorted(set(train_drugs))}
    absent = sorted(set(eval_drugs) - set(means))
    if absent:
        raise ValueError(f"inner-dev drugs lack exact-full-key TRAIN support: {absent}")
    return np.stack([means[d] for d in train_drugs]), means


def build_model(torch: Any, arm: str, n_drugs: int, pca_dim: int, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    has_state = arm != "capacity_matched_drug_only"
    width = 33 if not has_state else 32
    state_dim = pca_dim if has_state else 0

    class ResidualMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(n_drugs, 16)
            self.hidden = torch.nn.Linear(2048 + 16 + state_dim, width)
            self.output = torch.nn.Linear(width, 978)

        def forward(self, fp, drug_idx, state):
            parts = [fp, self.embedding(drug_idx)]
            if has_state:
                parts.append(state)
            return self.output(torch.relu(self.hidden(torch.cat(parts, dim=1))))

    return ResidualMLP(), width, has_state


def score_object_macro_nmse(torch: Any, model: Any, base, fp, drug, state,
                            target, keys: pd.DataFrame, gene_variance) -> float:
    model.eval()
    ids = keys.source_object_id.astype(str).to_numpy()
    with torch.no_grad():
        residual = model(fp, drug, state).detach().cpu().numpy().astype(np.float64)
    base_np = base.detach().cpu().numpy().astype(np.float64)
    target_np = target.detach().cpu().numpy().astype(np.float64)
    variance_np = gene_variance.detach().cpu().numpy().astype(np.float64)
    prediction = base_np + residual
    if not np.isfinite(prediction).all():
        raise FloatingPointError("nonfinite prediction in object-macro NMSE evaluation")
    usable = np.isfinite(variance_np) & (variance_np > 1e-12)
    if not usable.all():
        raise ValueError(f"registered TRAIN gene-variance denominator changed: {int((~usable).sum())} unusable")
    row_nmse = (np.square(prediction[:, usable] - target_np[:, usable]) /
                variance_np[usable][None, :]).mean(axis=1)
    return float(np.mean([row_nmse[ids == obj].mean() for obj in sorted(set(ids))]))


def train_epoch(torch: Any, arm: str, model: Any, optimizer: Any, *,
                fp, drug, state, residual_target, row_weights, gene_variance,
                batch_size: int, seed: int) -> tuple[float, dict[str, Any]]:
    model.train()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(len(residual_target), generator=generator).tolist()
    total_weighted = 0.0
    total_weight = 0.0
    n_batches = 0
    for start in range(0, len(order), batch_size):
        index = torch.as_tensor(order[start:start + batch_size], dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        output = model(fp[index], drug[index], state[index])
        if not torch.isfinite(output).all():
            raise FloatingPointError(f"{arm}: nonfinite training residual prediction")
        err = output - residual_target[index]
        per_row = (err.square() / gene_variance[None, :]).mean(dim=1)
        weights = row_weights[index]
        loss = (per_row * weights).sum() / weights.sum()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{arm}: nonfinite training loss")
        loss.backward()
        bad_grad = [name for name, param in model.named_parameters()
                    if param.grad is not None and not torch.isfinite(param.grad).all()]
        if bad_grad:
            raise FloatingPointError(f"{arm}: nonfinite gradients at {bad_grad}")
        optimizer.step()
        bad_param = [name for name, param in model.named_parameters()
                     if not torch.isfinite(param).all()]
        if bad_param:
            raise FloatingPointError(f"{arm}: nonfinite parameters at {bad_param}")
        for state_dict in optimizer.state.values():
            for value in state_dict.values():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise FloatingPointError(f"{arm}: nonfinite optimizer state")
        batch_weight = float(weights.sum().item())
        total_weighted += float(loss.detach().cpu().item()) * batch_weight
        total_weight += batch_weight
        n_batches += 1
    return total_weighted / total_weight, {"batches": n_batches, "finite": True}


def clone_state(torch: Any, model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_checkpoint(torch: Any, path: Path, arm: str, model_state: dict[str, Any],
                    epoch: int, dev_nmse: float, seed: int) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    started = time.monotonic()
    torch.save({"arm": arm, "model_state_dict": model_state, "selected_epoch": int(epoch),
                "inner_dev_object_macro_nmse": float(dev_nmse), "seed": int(seed)}, temp)
    temp.replace(path)
    return time.monotonic() - started


def input_contract(cfg: dict[str, Any]) -> tuple[Any, ...]:
    source_id = str(cfg["data"]["source_run_id"])
    fit.configure_source_run(source_id)
    train_keys, dev_keys, genes, train_data, dev_data, receipt = source.load_contract(cfg)
    train_ctl, train_y = train_data
    dev_ctl, dev_y = dev_data
    features, feature_receipt = fit._load_audited_ecfp4(fit.SOURCE_RUN_DIR)
    train_fp = fit._full_key_features(train_keys, np.arange(len(train_keys)), features)
    dev_fp = fit._full_key_features(dev_keys, np.arange(len(dev_keys)), features)
    keys = sorted(set(train_keys.full_inchikey.astype(str)))
    if set(keys) != set(dev_keys.full_inchikey.astype(str)) or len(keys) != 5:
        raise ValueError("frozen full-InChIKey domain changed or TRAIN/inner-dev support differs")
    drug_index = {key: i for i, key in enumerate(keys)}
    train_drug = np.asarray([drug_index[k] for k in train_keys.full_inchikey.astype(str)], dtype=np.int64)
    dev_drug = np.asarray([drug_index[k] for k in dev_keys.full_inchikey.astype(str)], dtype=np.int64)
    receipt["feature_audit"] = feature_receipt
    receipt["train_ecfp4_sha256"] = hashlib.sha256(train_fp.astype("<f4").tobytes()).hexdigest()
    receipt["inner_dev_ecfp4_sha256"] = hashlib.sha256(dev_fp.astype("<f4").tobytes()).hexdigest()
    receipt["drug_index"] = drug_index
    receipt["train_numeric_role_read"] = True
    receipt["inner_dev_numeric_role_read_for_registered_early_stopping_and_final_scoring"] = True
    receipt["pdo_numeric_sensitivity_opened"] = False
    receipt["pdo_transcript_response_opened"] = False
    receipt["outer_or_locked_numeric_values_opened"] = False
    receipt["panacea_numeric_values_opened"] = False
    return train_keys, dev_keys, genes, train_ctl, train_y, dev_ctl, dev_y, train_fp, dev_fp, train_drug, dev_drug, keys, receipt


def run(args: argparse.Namespace) -> int:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() not in ("", "-1"):
        raise RuntimeError("EXP-058 is CPU-only; CUDA_VISIBLE_DEVICES must be empty")
    import torch
    from sklearn.linear_model import Ridge

    if torch.cuda.is_available():
        raise RuntimeError("CPU-only EXP-058 refuses to initialize CUDA")
    started = time.monotonic()
    cfg_path = (ROOT / args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("experiment_id") != "EXP-058" or cfg.get("run_id") != args.run_id:
        raise ValueError("CLI run_id/config do not bind to preregistered EXP-058")
    if cfg.get("attempt_id") != args.attempt_id:
        raise ValueError("CLI attempt id differs from frozen configuration")
    if (int(cfg["resources"]["cpu_wall_cap_minutes"]) != 120 or
            float(cfg["resources"]["gpu_hours_cap"]) != 0 or
            int(cfg["training"]["max_epochs"]) != 300 or
            int(cfg["training"]["seed"]) != 58058):
        raise ValueError("EXP-058 frozen resource/training contract changed")
    prior_package_wall = float(cfg["resources"].get("prior_package_wall_seconds", 0.0))
    total_cap_seconds = int(cfg["resources"]["cpu_wall_cap_minutes"]) * 60
    if not math.isfinite(prior_package_wall) or prior_package_wall < 0 or prior_package_wall >= total_cap_seconds:
        raise ValueError("EXP-058 prior package wall charge is invalid or exhausts the fixed CPU cap")
    out_root = ROOT / cfg["output"]["root"]
    attempt = out_root / args.attempt_id
    if out_root.exists():
        raise FileExistsError(f"unique EXP-058 run path exists: {out_root}")
    out_root.mkdir(parents=True, exist_ok=False)
    attempt.mkdir(parents=True, exist_ok=False)
    manifest = {"experiment_id": "EXP-058", "run_id": args.run_id, "attempt_id": args.attempt_id,
                "status": "RUNNING", "config_sha256": sha256(cfg_path),
                "runner_sha256": sha256(Path(__file__).resolve()), "started_monotonic": started,
                "device": "cpu", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "prior_package_wall_seconds_charged": prior_package_wall,
                "total_package_cpu_cap_seconds": total_cap_seconds,
                "forbidden_inputs_opened": []}
    atomic_json(attempt / "run_manifest.json", manifest)
    try:
        data = input_contract(cfg)
        (train_keys, dev_keys, genes, train_ctl, train_y, dev_ctl, dev_y,
         train_fp_np, dev_fp_np, train_drug_np, dev_drug_np, drug_keys, input_receipt) = data
        atomic_json(attempt / "input_receipt.json", finite_json(input_receipt))
        train_base, train_means = per_drug_predictions(train_keys, train_y, train_keys)
        dev_base = np.stack([train_means[k] for k in dev_keys.full_inchikey.astype(str)]).astype(np.float64)
        train_gene_variance = np.var(train_y.astype(np.float64), axis=0, ddof=0)
        usable = np.isfinite(train_gene_variance) & (train_gene_variance > 1e-12)
        if not usable.all():
            raise ValueError(f"registered variance-normalized objective requires all genes; unusable={int((~usable).sum())}")
        if not (np.isfinite(train_y).all() and np.isfinite(dev_y).all() and
                np.isfinite(train_base).all() and np.isfinite(dev_base).all()):
            raise FloatingPointError("nonfinite input/target/baseline; fail-closed")
        train_states, dev_states, state_receipt = make_state_features(
            train_keys, dev_keys, train_ctl, dev_ctl,
            n_components=int(cfg["models"]["nonlinear_state_candidate"]["state_pca_components"]),
            shuffle_seed=None)
        shuf_train_states, shuf_dev_states, shuffle_receipt = make_state_features(
            train_keys, dev_keys, train_ctl, dev_ctl,
            n_components=int(cfg["models"]["nonlinear_state_candidate"]["state_pca_components"]),
            shuffle_seed=58059)
        atomic_json(attempt / "state_transform_receipt.json", finite_json({
            "true_state": state_receipt, "shuffled_state_control": shuffle_receipt}))

        # Existing EXP-053 method is the frozen linear-state comparator.
        object_states = source.object_state_map(
            pd.concat([train_keys, dev_keys], ignore_index=True),
            np.concatenate([train_ctl, dev_ctl], axis=0))
        linear_dev, linear_fit = exp053.fit_state_residual(
            train_keys, train_y, dev_keys, train_base, dev_base,
            object_states, None, alpha=float(cfg["models"]["prior_linear"]["ridge_alpha"]))
        linear_train, _ = exp053.fit_state_residual(
            train_keys, train_y, train_keys, train_base, train_base,
            object_states, None, alpha=float(cfg["models"]["prior_linear"]["ridge_alpha"]))
        predictions_train: dict[str, np.ndarray] = {
            "zero_change": np.zeros_like(train_y), "per_drug_mean": train_base,
            "linear_state_residual": linear_train,
        }
        predictions_dev: dict[str, np.ndarray] = {
            "zero_change": np.zeros_like(dev_y), "per_drug_mean": dev_base,
            "linear_state_residual": linear_dev,
        }
        model_receipts: dict[str, Any] = {"linear_state_residual": linear_fit}
        curves: list[dict[str, Any]] = []
        best_state: dict[str, dict[str, Any]] = {}
        best_epoch: dict[str, int] = {}
        best_nmse: dict[str, float] = {}
        patience_count = {arm: 0 for arm in ARMS}
        active = set(ARMS)

        train_obj_counts = train_keys.source_object_id.astype(str).value_counts().to_dict()
        row_weights_np = np.asarray([1.0 / train_obj_counts[str(o)]
                                     for o in train_keys.source_object_id.astype(str)], dtype=np.float32)
        row_weights_np /= row_weights_np.sum()
        gene_var_t = torch.as_tensor(train_gene_variance.astype(np.float32), dtype=torch.float32)
        train_y_t = torch.as_tensor(train_y.astype(np.float32), dtype=torch.float32)
        dev_y_t = torch.as_tensor(dev_y.astype(np.float32), dtype=torch.float32)
        train_base_t = torch.as_tensor(train_base.astype(np.float32), dtype=torch.float32)
        dev_base_t = torch.as_tensor(dev_base.astype(np.float32), dtype=torch.float32)
        row_weights_t = torch.as_tensor(row_weights_np, dtype=torch.float32)
        train_fp = torch.as_tensor(train_fp_np, dtype=torch.float32)
        dev_fp = torch.as_tensor(dev_fp_np, dtype=torch.float32)
        train_drug = torch.as_tensor(train_drug_np, dtype=torch.long)
        dev_drug = torch.as_tensor(dev_drug_np, dtype=torch.long)
        state_by_arm = {"nonlinear_state_residual": (train_states, dev_states),
                        "capacity_matched_drug_only": (
                            np.zeros((len(train_keys), 16), dtype=np.float32),
                            np.zeros((len(dev_keys), 16), dtype=np.float32)),
                        "shuffled_state": (shuf_train_states, shuf_dev_states)}
        models: dict[str, Any] = {}
        optimizers: dict[str, Any] = {}
        arm_seeds = {"nonlinear_state_residual": 58058,
                     "capacity_matched_drug_only": 58060, "shuffled_state": 58059}
        for arm in ARMS:
            model, width, uses_state = build_model(torch, arm, len(drug_keys), 16, arm_seeds[arm])
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
            models[arm], optimizers[arm] = model, optimizer
            state_by_arm[arm] = (torch.as_tensor(state_by_arm[arm][0], dtype=torch.float32),
                                 torch.as_tensor(state_by_arm[arm][1], dtype=torch.float32))
            model_receipts[arm] = {"seed": arm_seeds[arm], "hidden_width": width,
                                   "uses_state": uses_state, "parameter_count": sum(p.numel() for p in model.parameters()),
                                   "optimizer": "Adam", "learning_rate": 1e-3, "weight_decay": 1e-3}

        residual_target = train_y_t - train_base_t
        if not torch.isfinite(residual_target).all():
            raise FloatingPointError("nonfinite per-drug residual target")
        train_eval_keys = train_keys
        dev_eval_keys = dev_keys
        pre_epoch_elapsed = time.monotonic() - started
        epoch_started = time.monotonic()
        arm_train_loss: dict[str, float] = {}
        for arm in ARMS:
            loss, step_receipt = train_epoch(
                torch, arm, models[arm], optimizers[arm], fp=train_fp, drug=train_drug,
                state=state_by_arm[arm][0], residual_target=residual_target,
                row_weights=row_weights_t, gene_variance=gene_var_t,
                batch_size=64, seed=58058 + arm_seeds[arm])
            arm_train_loss[arm] = loss
            dev_nmse = score_object_macro_nmse(
                torch, models[arm], dev_base_t, dev_fp, dev_drug, state_by_arm[arm][1],
                dev_y_t, dev_eval_keys, gene_var_t)
            best_state[arm] = clone_state(torch, models[arm])
            best_epoch[arm] = 1
            best_nmse[arm] = dev_nmse
            # The immutable first-epoch snapshot is retained for the registered cost gate.
            save_checkpoint(torch, attempt / f"epoch1_{arm}.pt", arm, best_state[arm], 1,
                            dev_nmse, arm_seeds[arm])
            curves.append({"epoch": 1, "arm": arm, "train_object_balanced_normalized_residual_mse": loss,
                           "inner_dev_source_object_macro_nmse": dev_nmse, "improved_best": True,
                           "early_stop_patience_count": 0, "finite": True,
                           "train_batches": step_receipt["batches"]})
        epoch_seconds = time.monotonic() - epoch_started
        checkpoint_io_seconds = sum(float(c.get("io_seconds", 0.0)) for c in [])
        # Include measured first-epoch snapshot saves in the package timing through the epoch wall above.
        final_io_allowance = max(60.0, epoch_seconds * 0.10)
        projected_seconds = (pre_epoch_elapsed + 300.0 * epoch_seconds + final_io_allowance) * 1.20
        projected_package_seconds = prior_package_wall + projected_seconds
        cost_gate = {"status": "PASS" if projected_package_seconds <= total_cap_seconds else "FAIL_CLOSED_COST_GATE",
                     "cpu_cap_seconds": total_cap_seconds,
                     "prior_package_wall_seconds_charged": prior_package_wall,
                     "remaining_package_cpu_seconds_before_attempt": total_cap_seconds - prior_package_wall,
                     "elapsed_to_end_epoch1_current_attempt_seconds": pre_epoch_elapsed + epoch_seconds,
                     "measured_epoch1_all_arm_train_eval_and_checkpoint_wall_seconds": epoch_seconds,
                     "measured_first_epoch_checkpoint_io_included": True,
                     "full_schedule_projected_epochs_per_arm": 300,
                     "post_training_finalization_io_allowance_seconds": final_io_allowance,
                     "projection_includes_margin_fraction": 0.20,
                     "projected_current_schedule_seconds_with_margin": projected_seconds,
                     "projected_full_package_seconds_with_margin": projected_package_seconds,
                     "projected_full_package_minutes_with_margin": projected_package_seconds / 60.0,
                     "arms_all_included": list(ARMS), "schedule_shortened": False,
                     "continued": projected_package_seconds <= total_cap_seconds}
        atomic_json(attempt / "first_epoch_cost_gate.json", finite_json(cost_gate))
        # Establish root-recoverable best snapshots from the first real comparison epoch.
        if projected_package_seconds > total_cap_seconds:
            train_predictions = {"zero_change": predictions_train["zero_change"],
                                 "per_drug_mean": train_base, "linear_state_residual": linear_train}
            dev_predictions = {"zero_change": predictions_dev["zero_change"],
                               "per_drug_mean": dev_base, "linear_state_residual": linear_dev}
            for arm in ARMS:
                train_state = torch.as_tensor(state_by_arm[arm][0], dtype=torch.float32)
                with torch.no_grad():
                    train_predictions[arm] = (train_base_t + models[arm](train_fp, train_drug,
                                                          train_state).cpu()).numpy().astype(np.float64)
                    dev_predictions[arm] = (dev_base_t + models[arm](dev_fp, dev_drug,
                                                         state_by_arm[arm][1]).cpu()).numpy().astype(np.float64)
            write_cost_gate_outputs(attempt, train_keys, dev_keys, genes, train_y, dev_y,
                                    train_predictions, dev_predictions, train_gene_variance,
                                    model_receipts, curves, input_receipt, cost_gate, started, cfg, args)
            return 0

        # Continue the same attempt only when the complete schedule and all fixed arms fit.
        stop_epoch = 1
        for epoch in range(2, 301):
            epoch_start = time.monotonic()
            for arm in list(ARMS):
                if arm not in active:
                    continue
                loss, step_receipt = train_epoch(
                    torch, arm, models[arm], optimizers[arm], fp=train_fp, drug=train_drug,
                    state=state_by_arm[arm][0], residual_target=residual_target,
                    row_weights=row_weights_t, gene_variance=gene_var_t,
                    batch_size=64, seed=58058 + arm_seeds[arm] + epoch)
                dev_nmse = score_object_macro_nmse(
                    torch, models[arm], dev_base_t, dev_fp, dev_drug, state_by_arm[arm][1],
                    dev_y_t, dev_eval_keys, gene_var_t)
                improved = dev_nmse < best_nmse[arm] - 1e-12
                if improved:
                    best_nmse[arm] = dev_nmse
                    best_epoch[arm] = epoch
                    best_state[arm] = clone_state(torch, models[arm])
                    patience_count[arm] = 0
                else:
                    patience_count[arm] += 1
                curves.append({"epoch": epoch, "arm": arm,
                               "train_object_balanced_normalized_residual_mse": loss,
                               "inner_dev_source_object_macro_nmse": dev_nmse,
                               "improved_best": improved,
                               "early_stop_patience_count": patience_count[arm], "finite": True,
                               "train_batches": step_receipt["batches"]})
                if patience_count[arm] >= 30:
                    active.remove(arm)
            stop_epoch = epoch
            if not active:
                break
            elapsed = time.monotonic() - started
            package_elapsed = prior_package_wall + elapsed
            if package_elapsed > total_cap_seconds:
                raise TimeoutError(f"EXP-058 cumulative CPU wall cap exceeded at epoch {epoch}: {package_elapsed:.1f}s>{total_cap_seconds}s")

        # Restore the per-arm checkpoint chosen only by the preregistered inner-dev metric.
        for arm in ARMS:
            models[arm].load_state_dict(best_state[arm], strict=True)
            save_checkpoint(torch, attempt / f"best_{arm}.pt", arm, best_state[arm],
                            best_epoch[arm], best_nmse[arm], arm_seeds[arm])
        train_predictions: dict[str, np.ndarray] = {"zero_change": np.zeros_like(train_y),
            "per_drug_mean": train_base, "linear_state_residual": linear_train}
        dev_predictions: dict[str, np.ndarray] = {"zero_change": np.zeros_like(dev_y),
            "per_drug_mean": dev_base, "linear_state_residual": linear_dev}
        for arm in ARMS:
            with torch.no_grad():
                train_residual = models[arm](train_fp, train_drug, state_by_arm[arm][0]).cpu().numpy()
                dev_residual = models[arm](dev_fp, dev_drug, state_by_arm[arm][1]).cpu().numpy()
            train_predictions[arm] = train_base + train_residual
            dev_predictions[arm] = dev_base + dev_residual
        if any(not np.isfinite(a).all() for a in (*train_predictions.values(), *dev_predictions.values())):
            raise FloatingPointError("selected model or baseline produced nonfinite final predictions")
        model_receipts["early_stopping"] = {arm: {"selected_epoch": best_epoch[arm],
            "best_inner_dev_object_macro_nmse": best_nmse[arm], "max_epochs": 300,
            "patience": 30} for arm in ARMS}
        summary = write_final_outputs(attempt, train_keys, dev_keys, genes, train_y, dev_y,
            train_predictions, dev_predictions, train_gene_variance, model_receipts, curves,
            input_receipt, cost_gate, started, cfg, args, best_epoch, stop_epoch)
        return 0 if summary["status"] == "COMPLETED_FIXED_SPLIT_TRAIN_INNER_DEV_COMPARISON" else 1
    except Exception as exc:
        current_attempt_elapsed = time.monotonic() - started
        close = {"experiment_id": "EXP-058", "run_id": args.run_id, "attempt_id": args.attempt_id,
                 "status": "FAILED_CLOSED", "error_type": type(exc).__name__, "error": str(exc),
                 "traceback": traceback.format_exc(), "elapsed_cpu_wall_seconds": current_attempt_elapsed,
                 "prior_package_wall_seconds_charged": prior_package_wall,
                 "cumulative_package_cpu_wall_seconds": prior_package_wall + current_attempt_elapsed,
                 "gpu_hours": 0, "forbidden_numeric_values_opened": {"pdo": False, "outer": False,
                    "locked": False, "panacea": False}}
        atomic_json(attempt / "failure_close_receipt.json", close)
        manifest.update({"status": "FAILED_CLOSED", "closed_monotonic": time.monotonic()})
        atomic_json(attempt / "run_manifest.json", finite_json(manifest))
        raise


def compute_metrics(keys: pd.DataFrame, truth: np.ndarray, pred: np.ndarray,
                    train_gene_variance: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return source.metrics_for_role(keys, truth, pred, train_gene_variance)


def role_metrics_by_drug(keys: pd.DataFrame, truth: np.ndarray,
                         predictions: dict[str, np.ndarray], variance: np.ndarray,
                         role: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for drug in sorted(keys.full_inchikey.astype(str).unique()):
        idx = np.flatnonzero(keys.full_inchikey.astype(str).to_numpy() == drug)
        subset = keys.iloc[idx].reset_index(drop=True)
        for method, pred in predictions.items():
            metrics, _ = compute_metrics(subset, truth[idx], pred[idx], variance)
            rows.append({"role": role, "full_inchikey": drug, "method": method,
                         **{k: v for k, v in metrics.items() if k != "object_scores"}})
    return rows


def paired_object_differences(keys: pd.DataFrame, metrics_by_method: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ref = metrics_by_method["per_drug_mean"]["inner_dev"]["object_scores"]
    rows = []
    for method in ALL_METHODS:
        if method == "per_drug_mean":
            continue
        scores = metrics_by_method[method]["inner_dev"]["object_scores"]
        for obj in sorted(ref):
            rows.append({"source_object_id": obj, "reference": "per_drug_mean", "method": method,
                         "reference_minus_method_nmse": float(ref[obj]["nmse"] - scores[obj]["nmse"]),
                         "method_direction_agreement": scores[obj]["direction_agreement"],
                         "method_amplitude_ratio": scores[obj]["amplitude_ratio"]})
    return rows


def build_summary(train_keys, dev_keys, genes, train_y, dev_y, train_predictions,
                  dev_predictions, variance, model_receipts, curves, input_receipt,
                  cost_gate, started, cfg, args, *, status, best_epoch=None, stop_epoch=None):
    metrics: dict[str, Any] = {}
    object_rows: list[dict[str, Any]] = []
    for method in ALL_METHODS:
        tr, tr_rows = compute_metrics(train_keys, train_y, train_predictions[method], variance)
        dv, dv_rows = compute_metrics(dev_keys, dev_y, dev_predictions[method], variance)
        metrics[method] = {"train_in_sample": tr, "inner_dev": dv}
        object_rows.extend({"method": method, "role": "train_in_sample", **row} for row in tr_rows)
        object_rows.extend({"method": method, "role": "inner_dev", **row} for row in dv_rows)
    control_methods = ["per_drug_mean", "linear_state_residual", "capacity_matched_drug_only", "shuffled_state"]
    strongest = min(control_methods, key=lambda m: float(metrics[m]["inner_dev"]["object_macro_nmse"]))
    candidate_nmse = float(metrics["nonlinear_state_residual"]["inner_dev"]["object_macro_nmse"])
    strongest_nmse = float(metrics[strongest]["inner_dev"]["object_macro_nmse"])
    gain = (strongest_nmse - candidate_nmse) / strongest_nmse if strongest_nmse > 0 else None
    object_scores = metrics["nonlinear_state_residual"]["inner_dev"]["object_scores"]
    direction_correct = sum(float(v["direction_agreement"]) > 0.5 for v in object_scores.values())
    amp = metrics["nonlinear_state_residual"]["inner_dev"]["object_macro_amplitude_ratio"]
    beats_controls = all(candidate_nmse < float(metrics[m]["inner_dev"]["object_macro_nmse"])
                         for m in control_methods)
    gate = bool(gain is not None and gain >= 0.05 and direction_correct >= 15 and amp is not None
                and 0.5 <= float(amp) <= 2.0 and beats_controls)
    decision = {"strongest_registered_control": strongest, "candidate_inner_dev_nmse": candidate_nmse,
                "strongest_control_inner_dev_nmse": strongest_nmse,
                "relative_nmse_gain_over_strongest_control": gain,
                "direction_correct_objects_definition": "object direction agreement strictly greater than 0.5",
                "direction_correct_objects": int(direction_correct), "inner_dev_objects": len(object_scores),
                "amplitude_ratio": amp, "beats_per_drug_mean_linear_capacity_matched_and_shuffled": beats_controls,
                "expansion_gate_pass": gate,
                "candidate_disposition": "QUALIFIES_FOR_SEPARATE_PREREGISTERED_GENERALIZATION_REVIEW"
                    if gate else "RETAIN_NEGATIVE_AND_STOP_THIS_CANDIDATE",
                "claim_boundary": "fixed source-development split only; no independent-domain, organoid, PDO efficacy, or clinical claim"}
    summary = {"experiment_id": "EXP-058", "run_id": args.run_id, "attempt_id": args.attempt_id,
               "status": status, "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
               "config_sha256": sha256((ROOT / args.config).resolve()),
               "runner_sha256": sha256(Path(__file__).resolve()),
               "cpu_wall_seconds": float(cfg["resources"].get("prior_package_wall_seconds", 0.0)) + time.monotonic() - started,
               "current_attempt_cpu_wall_seconds": time.monotonic() - started,
               "prior_package_wall_seconds_charged": float(cfg["resources"].get("prior_package_wall_seconds", 0.0)),
               "gpu_hours": 0, "denominators": {"train_conditions": len(train_keys),
                   "train_objects": train_keys.source_object_id.nunique(), "inner_dev_conditions": len(dev_keys),
                   "inner_dev_objects": dev_keys.source_object_id.nunique(), "train_dev_object_overlap": 0,
                   "full_inchikeys": train_keys.full_inchikey.nunique(), "genes": len(genes),
                   "train_gene_variance_usable": int((variance > 1e-12).sum())},
               "input_receipt": input_receipt, "model_fit_receipts": model_receipts,
               "cost_gate": cost_gate, "training_curve_rows": len(curves),
               "early_stopping_best_epochs": best_epoch, "last_epoch_completed": stop_epoch,
               "metrics": {m: {r: {k: v for k, v in metrics[m][r].items() if k != "object_scores"}
                               for r in ("train_in_sample", "inner_dev")} for m in ALL_METHODS},
               "decision": decision,
               "forbidden_numeric_values_opened": {"pdo": False, "outer": False, "locked": False, "panacea": False}}
    return summary, metrics, object_rows, decision


def write_npz(attempt, train_keys, dev_keys, genes, train_y, dev_y, train_predictions, dev_predictions):
    path = attempt / "full_keyed_predictions.npz"
    np.savez_compressed(path,
        train_full_condition_key=train_keys.full_condition_key.astype(str).to_numpy(),
        inner_dev_full_condition_key=dev_keys.full_condition_key.astype(str).to_numpy(),
        train_source_object_id=train_keys.source_object_id.astype(str).to_numpy(),
        inner_dev_source_object_id=dev_keys.source_object_id.astype(str).to_numpy(),
        train_full_inchikey=train_keys.full_inchikey.astype(str).to_numpy(),
        inner_dev_full_inchikey=dev_keys.full_inchikey.astype(str).to_numpy(),
        gene_axis=genes.astype(str), train_target_delta=train_y.astype(np.float32),
        inner_dev_target_delta=dev_y.astype(np.float32),
        **{f"{m}_train": train_predictions[m].astype(np.float32) for m in ALL_METHODS},
        **{f"{m}_inner_dev": dev_predictions[m].astype(np.float32) for m in ALL_METHODS})
    return path


def write_cost_gate_outputs(attempt, train_keys, dev_keys, genes, train_y, dev_y,
                            train_predictions, dev_predictions, variance, model_receipts,
                            curves, input_receipt, cost_gate, started, cfg, args):
    baseline_methods = {k: v for k, v in train_predictions.items()}
    summary, metrics, object_rows, decision = build_summary(
        train_keys, dev_keys, genes, train_y, dev_y, train_predictions, dev_predictions,
        variance, model_receipts, curves, input_receipt, cost_gate, started, cfg, args,
        status="CLOSED_AFTER_FIRST_EPOCH_COST_GATE")
    summary["decision"]["expansion_gate_pass"] = False
    summary["decision"]["candidate_disposition"] = "NOT_EVALUABLE_COST_GATE_FAILED_AFTER_FIRST_EPOCH"
    summary["decision"]["reason"] = "Full 300-epoch schedule plus I/O and 20% margin exceeds CPU cap; no schedule shortening or arm pruning."
    npz = write_npz(attempt, train_keys, dev_keys, genes, train_y, dev_y, train_predictions, dev_predictions)
    atomic_json(attempt / "metrics_summary.json", finite_json(summary))
    atomic_json(attempt / "model_fit_receipts.json", finite_json(model_receipts))
    pd.DataFrame(object_rows).to_csv(attempt / "object_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(role_metrics_by_drug(train_keys, train_y, train_predictions, variance, "train_in_sample") +
                 role_metrics_by_drug(dev_keys, dev_y, dev_predictions, variance, "inner_dev")).to_csv(
                     attempt / "per_drug_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(curves).to_csv(attempt / "training_curves.tsv", sep="\t", index=False)
    (attempt / "decision.json").write_text(json.dumps(finite_json(summary["decision"]),
        ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {"experiment_id": "EXP-058", "run_id": args.run_id, "attempt_id": args.attempt_id,
                "status": summary["status"], "config_sha256": sha256((ROOT / args.config).resolve()),
                "runner_sha256": sha256(Path(__file__).resolve()), "prediction_npz_sha256": sha256(npz),
                "elapsed_cpu_wall_seconds": summary["cpu_wall_seconds"], "gpu_hours": 0,
                "finished_at_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    atomic_json(attempt / "run_manifest.json", manifest)


def write_final_outputs(attempt, train_keys, dev_keys, genes, train_y, dev_y,
                        train_predictions, dev_predictions, variance, model_receipts,
                        curves, input_receipt, cost_gate, started, cfg, args, best_epoch, stop_epoch):
    summary, metrics, object_rows, decision = build_summary(
        train_keys, dev_keys, genes, train_y, dev_y, train_predictions, dev_predictions,
        variance, model_receipts, curves, input_receipt, cost_gate, started, cfg, args,
        status="COMPLETED_FIXED_SPLIT_TRAIN_INNER_DEV_COMPARISON",
        best_epoch=best_epoch, stop_epoch=stop_epoch)
    summary["paired_object_differences"] = paired_object_differences(dev_keys, metrics)
    npz = write_npz(attempt, train_keys, dev_keys, genes, train_y, dev_y, train_predictions, dev_predictions)
    atomic_json(attempt / "metrics_summary.json", finite_json(summary))
    atomic_json(attempt / "model_fit_receipts.json", finite_json(model_receipts))
    pd.DataFrame(object_rows).to_csv(attempt / "object_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(role_metrics_by_drug(train_keys, train_y, train_predictions, variance, "train_in_sample") +
                 role_metrics_by_drug(dev_keys, dev_y, dev_predictions, variance, "inner_dev")).to_csv(
                     attempt / "per_drug_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(curves).to_csv(attempt / "training_curves.tsv", sep="\t", index=False)
    pd.DataFrame(summary["paired_object_differences"]).to_csv(attempt / "paired_object_differences.tsv", sep="\t", index=False)
    (attempt / "decision.json").write_text(json.dumps(finite_json(decision), ensure_ascii=False,
        sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    total_wall = time.monotonic() - started
    prior_package_wall = float(cfg["resources"].get("prior_package_wall_seconds", 0.0))
    package_wall = prior_package_wall + total_wall
    if package_wall > int(cfg["resources"]["cpu_wall_cap_minutes"]) * 60:
        raise TimeoutError(f"EXP-058 cumulative wall cap exceeded after finalization: {package_wall:.1f}s")
    summary["cpu_wall_seconds"] = package_wall
    summary["current_attempt_cpu_wall_seconds"] = total_wall
    summary["output_files"] = {p.name: sha256(p) for p in attempt.iterdir() if p.is_file() and p.name != "metrics_summary.json"}
    atomic_json(attempt / "metrics_summary.json", finite_json(summary))
    manifest = {"experiment_id": "EXP-058", "run_id": args.run_id, "attempt_id": args.attempt_id,
                "status": summary["status"], "config_sha256": sha256((ROOT / args.config).resolve()),
                "runner_sha256": sha256(Path(__file__).resolve()), "prediction_npz_sha256": sha256(npz),
                "elapsed_cpu_wall_seconds": package_wall,
                "current_attempt_cpu_wall_seconds": total_wall,
                "prior_package_wall_seconds_charged": prior_package_wall,
                "gpu_hours": 0,
                "finished_at_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    atomic_json(attempt / "run_manifest.json", manifest)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
