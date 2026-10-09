"""Fixed CPU comparison for EXP-056 state-by-drug low-rank residuals.

Only the registered TRAIN and inner-development rows are read.  This runner
never opens PDO, outer, locked, or external response data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/modeling"))
import exp051_contract_fit_smoke as fit  # noqa: E402
import run_exp052_transigen_lr_stability as source_reader  # noqa: E402

METHODS = (
    "per_drug_mean",
    "state_rank4",
    "state_rank8",
    "no_state_rank4",
    "no_state_rank8",
    "shuffled_state_rank4",
    "shuffled_state_rank8",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha(values: list[str] | np.ndarray) -> str:
    return hashlib.sha256(("\n".join(map(str, values)) + "\n").encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                              indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def json_finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds").replace("+00:00", "Z")


def load_contract(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray,
                                                  np.ndarray, np.ndarray, dict[str, Any]]:
    source_id = str(cfg["data"]["source_run_id"])
    fit.configure_source_run(source_id)
    src = fit.SOURCE_RUN_DIR
    required = (
        "metadata_freeze_receipt.json", "gene_axis_receipt.json",
        "value_contract_receipt.json", "root_values_readback_receipt.json",
        "response_blind_conditions.tsv", "gene_axis_mapping.tsv",
        "frozen_train_inner_values.npz",
    )
    missing = [name for name in required if not (src / name).is_file()]
    if missing:
        raise FileNotFoundError(f"frozen source contract is missing: {missing}")

    freeze_path = src / "metadata_freeze_receipt.json"
    axis_path = src / "gene_axis_receipt.json"
    value_path = src / "value_contract_receipt.json"
    readback_path = src / "root_values_readback_receipt.json"
    key_path = src / "response_blind_conditions.tsv"
    mapping_path = src / "gene_axis_mapping.tsv"
    value_archive = src / "frozen_train_inner_values.npz"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    axis = json.loads(axis_path.read_text(encoding="utf-8"))
    value_receipt = json.loads(value_path.read_text(encoding="utf-8"))
    readback = json.loads(readback_path.read_text(encoding="utf-8"))
    expected = cfg["data"]
    if freeze.get("run_id") != source_id:
        raise ValueError("source freeze receipt run_id differs from the registered input")
    if axis.get("status") != "PASS_EXACT_978_AXIS" or readback.get("checks", {}).get("result") != "PASS":
        raise ValueError("frozen gene-axis or root value readback receipt is not PASS")
    archived_sha = value_receipt.get("files_sha256", {}).get("frozen_train_inner_values_npz")
    if not archived_sha or sha256(value_archive) != archived_sha or readback.get("input_sha256") != archived_sha:
        raise ValueError("frozen value archive is not bound to both source receipts")
    if int(axis.get("n_genes", axis.get("gene_axis", {}).get("n", 978))) != 978:
        # Some older receipts store n under a nested field; the mapping below is authoritative too.
        if int(axis.get("gene_axis", {}).get("n", -1)) != 978:
            raise ValueError("registered source axis is not 978 genes")

    mapping = pd.read_csv(mapping_path, sep="\t", dtype=str, keep_default_na=False)
    if (len(mapping) != 978 or mapping["axis_index"].astype(int).tolist() != list(range(978))
            or mapping["h5ad_var_name"].nunique() != 978):
        raise ValueError("source gene mapping is not a unique ordered 978-gene axis")
    genes = mapping["h5ad_var_name"].astype(str).to_numpy(dtype=str)
    annotation_sha = text_sha(mapping["h5ad_var_annotation"].astype(str).to_numpy())
    if annotation_sha != axis.get("mapping_sha256"):
        raise ValueError("source gene annotation order differs from the frozen axis receipt")
    keys_all = pd.read_csv(key_path, sep="\t", dtype=str, keep_default_na=False)
    required_cols = {"split_role", "source_object_id", "source_row", "obs_name", "pert_id",
                     "full_inchikey", "full_condition_key"}
    if not required_cols.issubset(keys_all.columns):
        raise ValueError(f"condition table lacks required fields: {sorted(required_cols - set(keys_all.columns))}")
    if not keys_all["full_condition_key"].is_unique:
        raise ValueError("full condition keys are not unique")
    train_mask = keys_all["split_role"].eq("train")
    dev_mask = keys_all["split_role"].eq("inner_dev")
    if not (train_mask | dev_mask).all() or int((~(train_mask | dev_mask)).sum()) != 0:
        raise ValueError("source table contains unregistered roles outside train/inner_dev")
    train_keys = keys_all.loc[train_mask].reset_index(drop=True)
    dev_keys = keys_all.loc[dev_mask].reset_index(drop=True)
    train_ids = set(train_keys.source_object_id.astype(str))
    dev_ids = set(dev_keys.source_object_id.astype(str))
    counts = (len(train_keys), len(train_ids), len(dev_keys), len(dev_ids),
              keys_all.full_inchikey.nunique(), len(genes))
    expected_counts = (int(expected["train_conditions"]), int(expected["train_objects"]),
                       int(expected["inner_dev_conditions"]), int(expected["inner_dev_objects"]),
                       int(expected["exact_full_inchikeys"]), int(expected["genes"]))
    if counts != expected_counts:
        raise ValueError(f"frozen data denominator changed: actual={counts}, expected={expected_counts}")
    if train_ids & dev_ids:
        raise ValueError("source object overlap between TRAIN and inner-dev is nonzero")
    if set(train_keys.full_inchikey.astype(str)) != set(dev_keys.full_inchikey.astype(str)):
        raise ValueError("TRAIN and inner-dev do not share the same complete InChIKey domain")

    # All numeric input comes from the registered H5AD role-specific row reader.
    train_values = source_reader.read_role_expression(train_keys, genes)
    dev_values = source_reader.read_role_expression(dev_keys, genes)
    train_ctl = train_values["X_ctl"].astype(np.float64)
    train_y = train_values["delta"].astype(np.float64)
    dev_ctl = dev_values["X_ctl"].astype(np.float64)
    dev_y = dev_values["delta"].astype(np.float64)
    if (train_ctl.shape != (419, 978) or train_y.shape != (419, 978)
            or dev_ctl.shape != (93, 978) or dev_y.shape != (93, 978)):
        raise ValueError("role-specific H5AD reads have invalid matrix shape")
    max_decomp = max(float(train_values["max_abs_decomposition_error"]),
                     float(dev_values["max_abs_decomposition_error"]))
    if max_decomp > 1e-6:
        raise ValueError(f"treated-control decomposition error exceeds tolerance: {max_decomp}")

    h5ad = Path(source_reader.fit.H5AD)
    h5ad_sha = sha256(h5ad)
    expected_h5ad_sha = freeze.get("file_sha256", {}).get("h5ad")
    if expected_h5ad_sha and h5ad_sha != expected_h5ad_sha:
        raise ValueError("source H5AD bytes differ from the registered metadata freeze")
    receipt = {
        "status": "PASS_ROLE_SCOPED_SOURCE_READ",
        "source_run_id": source_id,
        "source_conditions_sha256": sha256(key_path),
        "source_value_archive_sha256": archived_sha,
        "source_h5ad_sha256": h5ad_sha,
        "source_axis_mapping_sha256": sha256(mapping_path),
        "gene_annotation_axis_sha256": annotation_sha,
        "train_full_condition_key_sha256": text_sha(train_keys.full_condition_key.to_numpy()),
        "inner_dev_full_condition_key_sha256": text_sha(dev_keys.full_condition_key.to_numpy()),
        "train_object_ids_sha256": text_sha(sorted(train_ids)),
        "inner_dev_object_ids_sha256": text_sha(sorted(dev_ids)),
        "full_inchikeys_sha256": text_sha(sorted(keys_all.full_inchikey.astype(str).unique())),
        "genes": 978,
        "denominators": {"train_conditions": len(train_keys), "train_objects": len(train_ids),
                         "inner_dev_conditions": len(dev_keys), "inner_dev_objects": len(dev_ids),
                         "train_inner_dev_object_overlap": 0,
                         "full_inchikeys": int(keys_all.full_inchikey.nunique())},
        "max_abs_decomposition_error": max_decomp,
        "train_numeric_values_read": True,
        "inner_dev_numeric_values_read_for_registered_selection_and_scoring": True,
        "pdo_numeric_sensitivity_opened": False,
        "outer_locked_response_opened": False,
        "external_response_matrix_opened": False,
    }
    return train_keys, dev_keys, genes, (train_ctl, train_y), (dev_ctl, dev_y), receipt


def object_state_map(keys: pd.DataFrame, x_ctl: np.ndarray) -> dict[str, np.ndarray]:
    ids = keys.source_object_id.astype(str).to_numpy()
    states: dict[str, np.ndarray] = {}
    for obj in sorted(set(ids)):
        idx = np.flatnonzero(ids == obj)
        states[obj] = x_ctl[idx].mean(axis=0, dtype=np.float64)
    return states


def derangement(ids: list[str], rng: np.random.Generator) -> dict[str, str]:
    unique = sorted(set(map(str, ids)))
    if len(unique) < 2:
        raise ValueError("object-block shuffle requires at least two objects")
    order = np.asarray(unique, dtype=str)[rng.permutation(len(unique))]
    return {str(order[i]): str(order[(i + 1) % len(order)]) for i in range(len(order))}


def low_rank_transform(train_states: dict[str, np.ndarray], train_ids: list[str],
                       all_states: dict[str, np.ndarray], rank: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    ordered_train = sorted(set(train_ids))
    matrix = np.stack([train_states[obj] for obj in ordered_train]).astype(np.float64)
    mean = matrix.mean(axis=0, dtype=np.float64)
    scale = matrix.std(axis=0, dtype=np.float64)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    standardized = (matrix - mean) / scale
    # Rank is fixed by the EXP. Use SVD only on the TRAIN object-state matrix.
    _, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    if rank > min(vt.shape[0], vt.shape[1]):
        raise ValueError(f"requested rank {rank} exceeds TRAIN-only SVD capacity")
    components = vt[:rank].copy()
    score_map = {obj: ((np.asarray(all_states[obj], dtype=np.float64) - mean) / scale) @ components.T
                 for obj in sorted(all_states)}
    variance_total = float(np.square(singular).sum())
    variance_fraction = float(np.square(singular[:rank]).sum() / variance_total) if variance_total > 0 else 0.0
    receipt = {"rank": rank, "train_fit_objects": len(ordered_train), "fit_role": "TRAIN object-level X_ctl only",
               "state_gene_mean_sha256": hashlib.sha256(mean.astype("<f8").tobytes()).hexdigest(),
               "state_gene_scale_sha256": hashlib.sha256(scale.astype("<f8").tobytes()).hexdigest(),
               "component_sha256": hashlib.sha256(components.astype("<f8").tobytes()).hexdigest(),
               "train_singular_values": singular[:rank].tolist(),
               "train_state_variance_fraction_at_rank": variance_fraction,
               "dev_state_used_only_for_transform": True}
    return score_map, receipt


def random_embeddings(ids: list[str], rank: int, seed: int) -> dict[str, np.ndarray]:
    unique = sorted(set(map(str, ids)))
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((len(unique), rank))
    return {obj: values[i].astype(np.float64) for i, obj in enumerate(unique)}


def fit_interaction(train_keys: pd.DataFrame, train_y: np.ndarray, eval_keys: pd.DataFrame,
                    drug_base_train: np.ndarray, drug_base_eval: np.ndarray,
                    score_map: dict[str, np.ndarray], alpha: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit drug-specific multi-output Ridge residuals with equal object weight."""
    train_obj = train_keys.source_object_id.astype(str).to_numpy()
    eval_obj = eval_keys.source_object_id.astype(str).to_numpy()
    train_drug = train_keys.full_inchikey.astype(str).to_numpy()
    eval_drug = eval_keys.full_inchikey.astype(str).to_numpy()
    train_x = np.stack([score_map[obj] for obj in train_obj]).astype(np.float64)
    eval_x = np.stack([score_map[obj] for obj in eval_obj]).astype(np.float64)
    residual = train_y - drug_base_train
    train_pred = np.empty_like(train_y, dtype=np.float64)
    eval_pred = np.empty_like(drug_base_eval, dtype=np.float64)
    drug_receipts: dict[str, Any] = {}
    for drug in sorted(set(train_drug)):
        tr = np.flatnonzero(train_drug == drug)
        ev = np.flatnonzero(eval_drug == drug)
        if len(ev) == 0:
            continue
        if len(tr) == 0:
            raise ValueError(f"inner-dev contains unsupported complete InChIKey {drug}")
        obj_counts = pd.Series(train_obj[tr]).value_counts().to_dict()
        weights = np.asarray([1.0 / obj_counts[obj] for obj in train_obj[tr]], dtype=np.float64)
        weights *= len(weights) / weights.sum()
        x = train_x[tr]
        y = residual[tr]
        wsum = float(weights.sum())
        xmean = np.average(x, axis=0, weights=weights)
        ymean = np.average(y, axis=0, weights=weights)
        xc, yc = x - xmean, y - ymean
        gram = xc.T @ (weights[:, None] * xc)
        rhs = xc.T @ (weights[:, None] * yc)
        coef = np.linalg.solve(gram + float(alpha) * np.eye(gram.shape[0]), rhs)
        intercept = ymean - xmean @ coef
        train_pred[tr] = drug_base_train[tr] + (x @ coef + intercept)
        eval_pred[ev] = drug_base_eval[ev] + (eval_x[ev] @ coef + intercept)
        drug_receipts[drug] = {
            "train_conditions": len(tr), "train_objects": int(len(obj_counts)),
            "inner_dev_conditions": len(ev), "weighted_object_total_equalized": True,
            "sample_weight_sum": wsum, "feature_rank": int(x.shape[1]),
            "alpha": float(alpha),
            "residual_coef_frobenius_norm": float(np.linalg.norm(coef)),
            "residual_intercept_l2_norm": float(np.linalg.norm(intercept)),
            "fit_role": "TRAIN rows only",
        }
    if not np.isfinite(train_pred).all() or not np.isfinite(eval_pred).all():
        raise FloatingPointError("interaction fit generated non-finite predictions")
    return train_pred, eval_pred, {"drug_models": drug_receipts, "feature_rank": int(train_x.shape[1]),
                                   "ridge_alpha": float(alpha), "object_total_weight_equalized": True}


def per_drug_means(train_keys: pd.DataFrame, train_y: np.ndarray,
                   eval_keys: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_drugs = train_keys.full_inchikey.astype(str).to_numpy()
    eval_drugs = eval_keys.full_inchikey.astype(str).to_numpy()
    means = {drug: train_y[train_drugs == drug].mean(axis=0, dtype=np.float64)
             for drug in sorted(set(train_drugs))}
    missing = sorted(set(eval_drugs) - set(means))
    if missing:
        raise ValueError(f"inner-dev contains full keys without a TRAIN anchor: {missing}")
    train_pred = np.stack([means[d] for d in train_drugs])
    eval_pred = np.stack([means[d] for d in eval_drugs])
    return train_pred, eval_pred, {d: {"train_conditions": int((train_drugs == d).sum()),
                                      "mean_sha256": hashlib.sha256(means[d].astype("<f8").tobytes()).hexdigest()}
                                  for d in sorted(means)}


def metrics_for_role(keys: pd.DataFrame, truth: np.ndarray, pred: np.ndarray,
                     train_gene_variance: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    usable = np.isfinite(train_gene_variance) & (train_gene_variance > 1e-12)
    if not usable.any():
        raise ValueError("TRAIN-fitted variance normalizer has no usable genes")
    error = pred.astype(np.float64) - truth.astype(np.float64)
    condition_nmse = np.square(error[:, usable]) / train_gene_variance[usable][None, :]
    condition_nmse = condition_nmse.mean(axis=1)
    condition_mae = np.abs(error).mean(axis=1)
    condition_corr = np.full(len(keys), np.nan, dtype=np.float64)
    for i in range(len(keys)):
        a, b = truth[i], pred[i]
        if np.std(a) > 0 and np.std(b) > 0:
            condition_corr[i] = float(np.corrcoef(a, b)[0, 1])
    condition_direction = (np.sign(truth) == np.sign(pred)).mean(axis=1)
    ids = keys.source_object_id.astype(str).to_numpy()
    object_metrics: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for obj in sorted(set(ids)):
        idx = np.flatnonzero(ids == obj)
        finite_corr = condition_corr[idx][np.isfinite(condition_corr[idx])]
        truth_amp = float(np.mean(np.abs(truth[idx])))
        pred_amp = float(np.mean(np.abs(pred[idx])))
        amp = pred_amp / truth_amp if truth_amp > 0 else math.nan
        vals = {
            "nmse": float(condition_nmse[idx].mean()),
            "mae": float(condition_mae[idx].mean()),
            "condition_gene_correlation": float(finite_corr.mean()) if len(finite_corr) else math.nan,
            "direction_agreement": float(condition_direction[idx].mean()),
            "amplitude_ratio": amp,
            "conditions": int(len(idx)),
            "condition_correlations_defined": int(len(finite_corr)),
        }
        object_metrics[obj] = json_finite(vals)
        rows.append({"source_object_id": obj, **json_finite(vals)})
    metric_names = ("nmse", "mae", "condition_gene_correlation", "direction_agreement", "amplitude_ratio")
    summary: dict[str, Any] = {}
    for name in metric_names:
        values = np.asarray([object_metrics[obj][name] if object_metrics[obj][name] is not None else np.nan
                             for obj in object_metrics], dtype=np.float64)
        summary[f"object_macro_{name}"] = (float(values[np.isfinite(values)].mean())
                                            if np.isfinite(values).any() else None)
    summary.update({"objects": int(len(object_metrics)), "conditions": int(len(keys)),
                    "condition_gene_correlations_defined": int(np.isfinite(condition_corr).sum()),
                    "genes_in_nmse": int(usable.sum()),
                    "zero_or_invalid_variance_genes_excluded": int((~usable).sum()),
                    "object_scores": object_metrics})
    return summary, rows


def bootstrap_pairs(object_scores: dict[str, dict[str, float]], methods: list[str],
                    reps: int, seed: int) -> dict[str, Any]:
    objects = sorted(object_scores)
    if len(objects) != 24:
        raise ValueError(f"paired object bootstrap requires exactly 24 inner-dev objects, got {len(objects)}")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(objects), size=(reps, len(objects)))
    outputs: dict[str, Any] = {}
    for candidate in methods:
        if candidate == "per_drug_mean":
            continue
        diff = np.asarray([object_scores[obj]["per_drug_mean"] - object_scores[obj][candidate]
                           for obj in objects], dtype=np.float64)
        boot = diff[indexes].mean(axis=1)
        outputs[f"per_drug_mean_minus_{candidate}"] = {
            "paired_objects": len(objects), "mean_absolute_nmse_improvement": float(diff.mean()),
            "bootstrap_95pct_ci": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
            "positive_favors_candidate": True,
        }
    return outputs


def make_mapping(keys: pd.DataFrame) -> tuple[dict[str, dict[str, str]], dict[str, np.ndarray]]:
    train_ids = sorted(keys.loc[keys.split_role.eq("train"), "source_object_id"].astype(str).unique())
    dev_ids = sorted(keys.loc[keys.split_role.eq("inner_dev"), "source_object_id"].astype(str).unique())
    rng = np.random.default_rng(56056)
    train_map = derangement(train_ids, rng)
    dev_map = derangement(dev_ids, rng)
    payload = {"train": train_map, "inner_dev": dev_map}
    checks = {}
    for role, mapping, expected_ids in (("train", train_map, train_ids), ("inner_dev", dev_map, dev_ids)):
        if set(mapping) != set(expected_ids) or set(mapping.values()) != set(expected_ids):
            raise ValueError(f"{role} shuffle mapping is not a within-role permutation")
        if any(k == v for k, v in mapping.items()):
            raise ValueError(f"{role} shuffle mapping contains fixed points")
        checks[role] = {"objects": len(expected_ids), "fixed_points": 0,
                        "mapping_sha256": hashlib.sha256(json.dumps(mapping, sort_keys=True,
                            separators=(",", ":")).encode()).hexdigest()}
    return payload, {"train": np.asarray(train_ids, dtype=str), "inner_dev": np.asarray(dev_ids, dtype=str)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    cfg_path = (ROOT / args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("experiment_id") != "EXP-056" or cfg.get("run_id") != args.run_id:
        raise ValueError("CLI run_id/config do not bind to the registered EXP-056 run")
    if cfg["models"]["low_rank_state_drug_interaction"]["ranks"] != [4, 8]:
        raise ValueError("EXP-056 rank set changed")
    if float(cfg["resources"]["gpu_hours_max"]) != 2.0 or int(cfg["resources"]["cpu_minutes_max"]) != 240:
        raise ValueError("EXP-056 resource ceilings changed")
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"].strip() not in ("", "-1"):
        raise RuntimeError("CPU-only EXP-056 runner must not expose a CUDA device")
    output = ROOT / "artifacts/experiments/EXP-056" / args.run_id
    if output.exists():
        raise FileExistsError(f"unique EXP-056 run directory already exists: {output}")

    train_keys, dev_keys, genes, train_data, dev_data, input_receipt = load_contract(cfg)
    train_ctl, train_y = train_data
    dev_ctl, dev_y = dev_data
    keys = pd.concat([train_keys, dev_keys], ignore_index=True)
    train_index = np.arange(len(train_keys), dtype=np.int64)
    dev_index = np.arange(len(train_keys), len(train_keys) + len(dev_keys), dtype=np.int64)
    all_ctl = np.concatenate([train_ctl, dev_ctl], axis=0)
    all_y = np.concatenate([train_y, dev_y], axis=0)
    # State inputs are the mean pre-treatment control vector per source object.
    states = object_state_map(keys, all_ctl)
    train_ids = train_keys.source_object_id.astype(str).tolist()
    dev_ids = dev_keys.source_object_id.astype(str).tolist()
    train_drug_pred, dev_drug_pred, drug_means = per_drug_means(train_keys, train_y, dev_keys)
    train_gene_variance = np.var(train_y.astype(np.float64), axis=0, ddof=0)
    predictions_train: dict[str, np.ndarray] = {"per_drug_mean": train_drug_pred}
    predictions_dev: dict[str, np.ndarray] = {"per_drug_mean": dev_drug_pred}
    transform_receipts: dict[str, Any] = {"per_drug_means": drug_means}
    model_fit_receipts: dict[str, Any] = {}

    shuffle_maps, _ = make_mapping(keys)
    mapping_file = {"seed": 56056, "train": shuffle_maps["train"],
                    "inner_dev": shuffle_maps["inner_dev"], "fixed_points": 0}
    mapping_file["checks"] = {
        "train_objects": len(shuffle_maps["train"]), "inner_dev_objects": len(shuffle_maps["inner_dev"]),
        "train_mapping_is_within_role_permutation": True,
        "inner_dev_mapping_is_within_role_permutation": True,
        "no_fixed_points": True, "label_used_for_mapping": False,
        "train_mapping_sha256": hashlib.sha256(json.dumps(shuffle_maps["train"], sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(),
        "inner_dev_mapping_sha256": hashlib.sha256(json.dumps(shuffle_maps["inner_dev"], sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(),
    }

    alpha = float(cfg["models"]["low_rank_state_drug_interaction"]["ridge_alpha"])
    base_train, base_dev = train_drug_pred, dev_drug_pred
    for rank in (4, 8):
        # True pre-treatment state: scaler + SVD fitted on TRAIN object states only.
        true_scores, true_transform = low_rank_transform(states, train_ids, states, rank)
        tr_pred, dv_pred, model_receipt = fit_interaction(
            train_keys, train_y, dev_keys, base_train, base_dev, true_scores, alpha)
        name = f"state_rank{rank}"
        predictions_train[name], predictions_dev[name] = tr_pred, dv_pred
        transform_receipts[name] = true_transform
        model_fit_receipts[name] = model_receipt

        # Matched-capacity control: deterministic object-keyed random coordinates, no X_ctl.
        random_scores = random_embeddings(train_ids + dev_ids, rank, 56056 + rank)
        tr_pred, dv_pred, model_receipt = fit_interaction(
            train_keys, train_y, dev_keys, base_train, base_dev, random_scores, alpha)
        name = f"no_state_rank{rank}"
        predictions_train[name], predictions_dev[name] = tr_pred, dv_pred
        transform_receipts[name] = {"rank": rank, "embedding_seed": 56056 + rank,
                                    "input": "deterministic object-id keyed random embedding; no expression state",
                                    "embedding_sha256": hashlib.sha256(np.stack(
                                        [random_scores[o] for o in sorted(random_scores)]).astype("<f8").tobytes()).hexdigest()}
        model_fit_receipts[name] = model_receipt

        # State permutation is separate inside TRAIN and inner-dev; mapping is label blind.
        shuffled_states = {obj: states[donor] for obj, donor in
                           {**shuffle_maps["train"], **shuffle_maps["inner_dev"]}.items()}
        shuffled_scores, shuffled_transform = low_rank_transform(shuffled_states, train_ids,
                                                                  shuffled_states, rank)
        tr_pred, dv_pred, model_receipt = fit_interaction(
            train_keys, train_y, dev_keys, base_train, base_dev, shuffled_scores, alpha)
        name = f"shuffled_state_rank{rank}"
        predictions_train[name], predictions_dev[name] = tr_pred, dv_pred
        transform_receipts[name] = {**shuffled_transform,
                                    "train_mapping_sha256": mapping_file["checks"].get("train_mapping_sha256"),
                                    "train_mapping": shuffle_maps["train"],
                                    "inner_dev_mapping": shuffle_maps["inner_dev"],
                                    "mapping_label_blind": True}
        model_fit_receipts[name] = model_receipt

    if set(predictions_train) != set(METHODS) or set(predictions_dev) != set(METHODS):
        raise RuntimeError("runner did not produce the fixed seven method/rank outputs")
    for method in METHODS:
        if (predictions_train[method].shape != (419, 978)
                or predictions_dev[method].shape != (93, 978)
                or not np.isfinite(predictions_train[method]).all()
                or not np.isfinite(predictions_dev[method]).all()):
            raise FloatingPointError(f"invalid saved prediction array for {method}")

    # Metrics are object-macro; every arm shares the same keyed denominators.
    metrics: dict[str, Any] = {}
    object_metric_rows: list[dict[str, Any]] = []
    for method in METHODS:
        train_summary, train_rows = metrics_for_role(train_keys, train_y, predictions_train[method], train_gene_variance)
        dev_summary, dev_rows = metrics_for_role(dev_keys, dev_y, predictions_dev[method], train_gene_variance)
        metrics[method] = {"train_in_sample": train_summary, "inner_dev": dev_summary}
        object_metric_rows.extend({"method": method, "role": "train_in_sample", **row} for row in train_rows)
        object_metric_rows.extend({"method": method, "role": "inner_dev", **row} for row in dev_rows)

    dev_object_nmse = {method: metrics[method]["inner_dev"]["object_macro_nmse"] for method in METHODS}
    selected = min(("state_rank4", "state_rank8"), key=lambda m: (dev_object_nmse[m], m))
    controls = ["per_drug_mean", "no_state_rank4", "no_state_rank8",
                "shuffled_state_rank4", "shuffled_state_rank8"]
    strongest_control = min(controls, key=lambda m: (dev_object_nmse[m], m))
    paired_objects = sorted(metrics[selected]["inner_dev"]["object_scores"])
    object_scores = {obj: {m: float(metrics[m]["inner_dev"]["object_scores"][obj]["nmse"])
                           for m in METHODS} for obj in paired_objects}
    bootstrap = bootstrap_pairs(object_scores, METHODS, int(cfg["metrics"]["bootstrap_replicates"]),
                                int(cfg["metrics"]["bootstrap_seed"]))
    diff = np.asarray([object_scores[obj][strongest_control] - object_scores[obj][selected]
                       for obj in paired_objects], dtype=np.float64)
    best_pair = bootstrap.get(f"per_drug_mean_minus_{selected}")
    # Bootstrap against the strongest fixed control using the same paired resamples.
    rng = np.random.default_rng(int(cfg["metrics"]["bootstrap_seed"]))
    idx = rng.integers(0, len(paired_objects), size=(int(cfg["metrics"]["bootstrap_replicates"]),
                                                     len(paired_objects)))
    boot_strong = diff[idx].mean(axis=1)
    selected_nmse = float(dev_object_nmse[selected])
    control_nmse = float(dev_object_nmse[strongest_control])
    relative_reduction = (control_nmse - selected_nmse) / control_nmse if control_nmse > 0 else math.nan
    improved_objects = int(np.sum(diff > 0))
    amp = metrics[selected]["inner_dev"]["object_macro_amplitude_ratio"]
    ci = [float(x) for x in np.quantile(boot_strong, [0.025, 0.975])]
    gate = bool(relative_reduction >= 0.05 and ci[0] > 0 and improved_objects >= 15
                and amp is not None and 0.5 <= float(amp) <= 2.0)
    decision = {
        "selected_interaction_rank": selected,
        "selection_rule": "minimum registered inner-dev object-macro NMSE among rank4 and rank8; ties select rank4",
        "strongest_control": strongest_control,
        "strongest_control_nmse": control_nmse,
        "selected_nmse": selected_nmse,
        "relative_nmse_reduction_vs_strongest_control": relative_reduction,
        "paired_object_nmse_improvement_control_minus_candidate": float(diff.mean()),
        "paired_object_bootstrap_95pct_ci": ci,
        "objects_improved_of_24": improved_objects,
        "selected_amplitude_ratio": amp,
        "development_gate_pass": gate,
        "all_controls_beaten": all(selected_nmse < float(dev_object_nmse[m])
                                    for m in controls),
        "candidate_disposition": "QUALIFIES_FOR_SEPARATE_INDEPENDENT_DOMAIN_ELIGIBILITY_REVIEW"
        if gate and all(selected_nmse < float(dev_object_nmse[m]) for m in controls)
        else "RETAIN_NEGATIVE_AND_STOP_THIS_CANDIDATE",
        "claim_boundary": "fixed source-development split only; no independent-domain, patient, organoid, or PDO efficacy claim",
    }
    paired_summary = bootstrap
    paired_summary[f"{strongest_control}_minus_selected_interaction"] = {
        "reference": strongest_control, "candidate": selected, "paired_objects": len(paired_objects),
        "reference_minus_candidate_nmse_mean": float(diff.mean()), "bootstrap_95pct_ci": ci,
        "positive_favors_candidate": True,
    }

    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "input_receipt.json", input_receipt)
    atomic_json(output / "transform_receipts.json", transform_receipts)
    atomic_json(output / "fit_receipts.json", model_fit_receipts)
    atomic_json(output / "randomization_mappings.json", mapping_file)
    atomic_json(output / "paired_object_bootstrap.json", paired_summary)
    atomic_json(output / "decision.json", decision)
    atomic_json(output / "metrics_summary.json", {m: metrics[m] for m in METHODS})
    pd.DataFrame(object_metric_rows).to_csv(output / "object_metrics.tsv", sep="\t", index=False)

    prediction_path = output / "full_keyed_predictions.npz"
    np.savez_compressed(
        prediction_path,
        train_condition_key=train_keys.full_condition_key.astype(str).to_numpy(),
        inner_dev_condition_key=dev_keys.full_condition_key.astype(str).to_numpy(),
        train_source_object_id=train_keys.source_object_id.astype(str).to_numpy(),
        inner_dev_source_object_id=dev_keys.source_object_id.astype(str).to_numpy(),
        train_full_inchikey=train_keys.full_inchikey.astype(str).to_numpy(),
        inner_dev_full_inchikey=dev_keys.full_inchikey.astype(str).to_numpy(),
        gene_axis=genes.astype(str),
        train_target_delta=train_y.astype(np.float32),
        inner_dev_target_delta=dev_y.astype(np.float32),
        **{f"{m}_train": predictions_train[m].astype(np.float32) for m in METHODS},
        **{f"{m}_inner_dev": predictions_dev[m].astype(np.float32) for m in METHODS},
    )

    elapsed = time.monotonic() - started
    cpu_cap_seconds = int(cfg["resources"]["cpu_minutes_max"]) * 60
    if elapsed > cpu_cap_seconds:
        raise TimeoutError(f"EXP-056 CPU time {elapsed:.1f}s exceeded {cpu_cap_seconds}s")
    ledger = ROOT / "artifacts/experiments/EXP-050/gpu_ledger.jsonl"
    result = {
        "experiment_id": "EXP-056", "run_id": args.run_id,
        "status": "COMPLETED_FIXED_SPLIT_TRAIN_INNER_DEV_COMPARISON",
        "created_at_utc": utc_now(), "config_sha256": sha256(cfg_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "source_input_receipt": input_receipt,
        "denominator": input_receipt["denominators"],
        "methods": {m: {"train_in_sample": metrics[m]["train_in_sample"],
                        "inner_dev": {k: v for k, v in metrics[m]["inner_dev"].items()
                                      if k != "object_scores"}}
                    for m in METHODS},
        "paired_bootstrap": paired_summary,
        "decision": decision,
        "predictions": {"file": prediction_path.name, "sha256": sha256(prediction_path),
                        "full_condition_keyed": True, "gene_axis_bound": True,
                        "numpy_allow_pickle_required": False},
        "forbidden_data_opened": {"PDO_numeric_sensitivity": False,
                                   "outer_or_locked_labels_or_responses": False,
                                   "external_response_matrix": False},
        "resource": {"gpu_hours": 0.0, "cpu_seconds": elapsed,
                     "cpu_minutes_max": int(cfg["resources"]["cpu_minutes_max"]),
                     "gpu_device_used": None,
                     "global_gpu_ledger_sha256_at_close": sha256(ledger) if ledger.is_file() else None,
                     "gpu_attempt_or_reservation_created": False},
        "claim_boundary": "same TRAIN/inner-dev source domain previously exposed by EXP-053/055; development only",
    }
    atomic_json(output / "result.json", json_finite(result))
    manifest_files = sorted(p for p in output.iterdir() if p.is_file() and p.name != "sha256_manifest.tsv")
    manifest = pd.DataFrame([{"file": p.name, "sha256": sha256(p), "bytes": p.stat().st_size}
                             for p in manifest_files])
    manifest.to_csv(output / "sha256_manifest.tsv", sep="\t", index=False)
    print(json.dumps({"status": result["status"], "run_id": args.run_id,
                      "denominator": result["denominator"], "selected": selected,
                      "strongest_control": strongest_control,
                      "inner_dev_nmse": {m: dev_object_nmse[m] for m in METHODS},
                      "relative_reduction_vs_strongest_control": relative_reduction,
                      "paired_ci": ci, "improved_objects": f"{improved_objects}/24",
                      "gate_pass": gate, "gpu_hours": 0.0, "cpu_seconds": elapsed,
                      "output": str(output)}, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
