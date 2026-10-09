#!/usr/bin/env python3
"""EXP-064: fixed TRAIN-only object-fold OOF state-neighbor comparison."""
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
METHODS = ("zero_change", "per_drug_mean", "nearest3_state", "random3_same_drug")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    x = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode())
    h.update(np.asarray(x.shape, dtype=np.int64).tobytes())
    h.update(x.tobytes())
    return h.hexdigest()


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def fold_for(obj: str, seed: int, folds: int) -> int:
    digest = hashlib.sha256(f"{seed}|{obj}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def to_dense(x: Any) -> np.ndarray:
    if hasattr(x, "to_memory"):
        x = x.to_memory()
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x)


def load_train_only(cfg: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import anndata as ad

    d = cfg["data"]
    freeze_root = ROOT / d["frozen_root"]
    conditions_path = ROOT / d["condition_keys"]
    manifest_path = ROOT / d["metadata_freeze"]
    root_gate_path = ROOT / d["metadata_root_receipt"]
    gene_path = ROOT / d["gene_axis"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate = json.loads(root_gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS_ROOT_INDEPENDENT_METADATA_READBACK":
        raise ValueError("EXP-062 root metadata gate is not PASS")
    if manifest.get("status") != "PASS_METADATA_FREEZE_PENDING_ROOT_READBACK":
        raise ValueError("EXP-062 metadata freeze manifest status mismatch")
    expected_meta_hash = manifest.get("frozen_files_sha256", {}).get("response_blind_conditions.tsv")
    if sha256_file(conditions_path) != expected_meta_hash:
        raise ValueError("EXP-062 frozen condition table hash mismatch")
    expected_h5 = manifest.get("source_h5ad_sha256")
    h5ad_path = Path(d["h5ad_path"])
    actual_h5 = sha256_file(h5ad_path)
    if actual_h5 != expected_h5:
        raise ValueError("official source H5AD SHA differs from the EXP-062 freeze")

    table = pd.read_csv(conditions_path, sep="\t", dtype=str, keep_default_na=False)
    train = table.loc[table.effective_split_role.eq("train")].copy()
    numeric_rows = pd.to_numeric(train.source_row, errors="raise").astype(np.int64).to_numpy()
    train["source_row"] = numeric_rows
    for col in ("full_condition_key", "source_object_id", "full_inchikey", "obs_name", "sample_id", "pert_id"):
        train[col] = train[col].astype(str).str.strip()
    if len(train) != int(d["condition_count"]):
        raise ValueError(f"TRAIN row count changed: {len(train)}")
    if train.full_condition_key.duplicated().any() or train.source_row.duplicated().any():
        raise ValueError("frozen TRAIN condition/source rows are not unique")
    if train.source_object_id.nunique() != int(d["object_count"]):
        raise ValueError("frozen TRAIN object count changed")
    if train.full_inchikey.nunique() != int(d["drug_count"]):
        raise ValueError("frozen TRAIN full-InChIKey count changed")
    if not train.full_inchikey.str.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]").all():
        raise ValueError("malformed or incomplete chemical identity in frozen TRAIN")

    genes = pd.read_csv(gene_path, sep="\t", dtype=str, keep_default_na=False)
    if len(genes) != int(d["genes"]) or genes.axis_index.astype(int).tolist() != list(range(int(d["genes"]))):
        raise ValueError("EXP-051 frozen gene mapping is not the expected ordered 978-axis")
    train = train.sort_values("source_row", kind="mergesort").reset_index(drop=True)
    rows = train.source_row.to_numpy(dtype=np.int64)
    a = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if tuple(map(int, a.shape))[1] != int(d["genes"]):
            raise ValueError("source gene dimension changed")
        gene_names = np.asarray(a.var_names.astype(str), dtype=str)
        if not np.array_equal(gene_names, genes.h5ad_var_name.astype(str).to_numpy()):
            raise ValueError("source gene order differs from the frozen EXP-051 axis")
        if "X_ctl" not in a.obsm.keys():
            raise ValueError("source H5AD lacks registered paired X_ctl")
        if not np.array_equal(np.asarray(a.obs_names[rows].astype(str), dtype=str), train.obs_name.to_numpy()):
            raise ValueError("frozen TRAIN observation names do not match their H5AD source rows")
        obs = a.obs.iloc[rows]
        for h5col, keycol in (("cell_iname", "source_object_id"), ("pert_id", "pert_id"), ("sample_id", "sample_id")):
            if h5col not in obs.columns or not np.array_equal(obs[h5col].astype(str).str.strip().to_numpy(), train[keycol].to_numpy()):
                raise ValueError(f"frozen TRAIN row identity mismatch for {h5col}")
        x = to_dense(a.X[rows, :]).astype(np.float64, copy=False)
        ctl = to_dense(a.obsm["X_ctl"][rows, :]).astype(np.float64, copy=False)
    finally:
        a.file.close()
    if x.shape != (len(train), int(d["genes"])) or ctl.shape != x.shape:
        raise ValueError(f"selected TRAIN source array shape mismatch: {x.shape}, {ctl.shape}")
    if not np.isfinite(x).all() or not np.isfinite(ctl).all():
        raise FloatingPointError("non-finite values in permitted TRAIN X/X_ctl rows")
    y = x - ctl
    if not np.isfinite(y).all():
        raise FloatingPointError("non-finite TRAIN treatment-response deltas")
    if not np.allclose(x, ctl + y, rtol=0, atol=1e-10):
        raise FloatingPointError("TRAIN X-X_ctl roundtrip check failed")
    receipt = {
        "source_h5ad_sha256": actual_h5,
        "condition_table_sha256": sha256_file(conditions_path),
        "metadata_freeze_manifest_sha256": sha256_file(manifest_path),
        "metadata_root_gate_sha256": sha256_file(root_gate_path),
        "gene_axis_sha256": sha256_file(gene_path),
        "source_train_rows_read": int(len(train)),
        "source_train_numeric_row_indices_sha256": sha256_array(rows),
        "train_X_numeric_sha256": sha256_array(x),
        "train_X_ctl_numeric_sha256": sha256_array(ctl),
        "train_delta_numeric_sha256": sha256_array(y),
        "train_x_shape": list(x.shape),
        "train_x_ctl_shape": list(ctl.shape),
        "train_delta_finite_count": int(np.isfinite(y).sum()),
        "train_delta_total_count": int(y.size),
        "inner_dev_numeric_values_read": False,
        "outer_locked_pdo_panacea_cdsdb_numeric_values_read": False,
        "layers_numeric_values_read": False,
        "GPU_hours": 0,
    }
    return train, gene_names, y, ctl, receipt


def object_macro_metrics(keys: pd.DataFrame, y: np.ndarray, pred: np.ndarray,
                         fold_var: np.ndarray, mask: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    methodscores: list[dict[str, Any]] = []
    usable_by_fold = np.isfinite(fold_var) & (fold_var > 1e-12)
    row_nmse = np.full(len(keys), np.nan, dtype=np.float64)
    row_mae = np.full(len(keys), np.nan, dtype=np.float64)
    row_dir = np.full(len(keys), np.nan, dtype=np.float64)
    row_amp = np.full(len(keys), np.nan, dtype=np.float64)
    row_corr = np.full(len(keys), np.nan, dtype=np.float64)
    folds = keys.fold_id.to_numpy(dtype=np.int64)
    effective_mask = mask & np.isfinite(y).all(axis=1) & np.isfinite(pred).all(axis=1)
    for i in np.flatnonzero(effective_mask):
        vmask = usable_by_fold[folds[i]]
        if not vmask.any():
            continue
        err = pred[i].astype(np.float64) - y[i].astype(np.float64)
        row_nmse[i] = float(np.mean(np.square(err[vmask]) / fold_var[folds[i], vmask]))
        row_mae[i] = float(np.mean(np.abs(err)))
        row_dir[i] = float(np.mean(np.sign(pred[i]) == np.sign(y[i])))
        ya, pa = y[i].astype(np.float64), pred[i].astype(np.float64)
        ysd, psd = float(np.std(ya)), float(np.std(pa))
        if ysd > 0 and psd > 0:
            row_corr[i] = float(np.corrcoef(ya, pa)[0, 1])
        yabs = float(np.mean(np.abs(ya)))
        if yabs > 0:
            row_amp[i] = float(np.mean(np.abs(pa)) / yabs)
    scored = keys.loc[effective_mask].copy()
    scored["row_nmse"] = row_nmse[effective_mask]
    scored["row_mae"] = row_mae[effective_mask]
    scored["row_direction"] = row_dir[effective_mask]
    scored["row_correlation"] = row_corr[effective_mask]
    scored["row_amplitude_ratio"] = row_amp[effective_mask]
    obj = scored.groupby("source_object_id", sort=True).agg(
        nmse=("row_nmse", "mean"), mae=("row_mae", "mean"),
        direction=("row_direction", "mean"), correlation=("row_correlation", "mean"),
        amplitude_ratio=("row_amplitude_ratio", "mean"), conditions=("full_condition_key", "size"),
    ).reset_index()
    amp_by_object: dict[str, float] = {}
    for obj_id in obj.source_object_id.astype(str):
        rows = np.flatnonzero(effective_mask & keys.source_object_id.astype(str).eq(obj_id).to_numpy())
        target_scale = float(np.mean(np.abs(y[rows].astype(np.float64)))) if len(rows) else 0.0
        amp_by_object[obj_id] = (float(np.mean(np.abs(pred[rows].astype(np.float64))) / target_scale)
                                 if target_scale > 0 else float("nan"))
    obj["amplitude_ratio"] = obj.source_object_id.astype(str).map(amp_by_object)
    finite_obj = np.isfinite(obj.nmse.to_numpy(dtype=float))
    summary = {
        "object_macro_nmse": float(obj.loc[finite_obj, "nmse"].mean()) if finite_obj.any() else None,
        "object_macro_mae": float(obj.loc[np.isfinite(obj.mae), "mae"].mean()) if np.isfinite(obj.mae).any() else None,
        "object_macro_direction_agreement": float(obj.loc[np.isfinite(obj.direction), "direction"].mean()) if np.isfinite(obj.direction).any() else None,
        "object_macro_gene_correlation": float(obj.loc[np.isfinite(obj.correlation), "correlation"].mean()) if np.isfinite(obj.correlation).any() else None,
        "object_macro_amplitude_ratio": float(obj.loc[np.isfinite(obj.amplitude_ratio), "amplitude_ratio"].mean()) if np.isfinite(obj.amplitude_ratio).any() else None,
        "condition_denominator": int(effective_mask.sum()),
        "requested_condition_count": int(mask.sum()),
        "excluded_nonfinite_prediction_conditions": int(mask.sum() - effective_mask.sum()),
        "object_denominator": int(finite_obj.sum()),
        "mae_object_denominator": int(np.isfinite(obj.mae.to_numpy(dtype=float)).sum()),
        "direction_object_denominator": int(np.isfinite(obj.direction.to_numpy(dtype=float)).sum()),
        "correlation_object_denominator": int(np.isfinite(obj.correlation.to_numpy(dtype=float)).sum()),
        "amplitude_object_denominator": int(np.isfinite(obj.amplitude_ratio.to_numpy(dtype=float)).sum()),
        "undefined_correlation_objects": int((~np.isfinite(obj.correlation)).sum()),
        "undefined_amplitude_objects": int((~np.isfinite(obj.amplitude_ratio)).sum()),
    }
    return summary, obj


def run_fold(fold: int, cfg: dict[str, Any], keys: pd.DataFrame, y: np.ndarray,
             ctl: np.ndarray, out: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any], pd.DataFrame]:
    from sklearn.decomposition import PCA

    start = time.monotonic()
    seed = int(cfg["split"]["object_hash_seed"])
    fold_count = int(cfg["split"]["folds"])
    folds = keys.source_object_id.astype(str).map(lambda x: fold_for(x, seed, fold_count)).to_numpy(dtype=np.int16)
    test_mask = folds == fold
    train_mask = ~test_mask
    if not test_mask.any() or not train_mask.any():
        raise ValueError(f"empty OOF split at fold {fold}")
    train_idx, test_idx = np.flatnonzero(train_mask), np.flatnonzero(test_mask)
    train_keys, test_keys = keys.iloc[train_idx].copy(), keys.iloc[test_idx].copy()
    train_objects = sorted(train_keys.source_object_id.astype(str).unique())
    test_objects = sorted(test_keys.source_object_id.astype(str).unique())
    if set(train_objects) & set(test_objects):
        raise ValueError("source-object leakage across this fold")
    # The single treatment-free state for each source object is the mean of its TRAIN X_ctl rows.
    object_ids = sorted(keys.source_object_id.astype(str).unique())
    state_by_obj = {obj: ctl[keys.source_object_id.astype(str).to_numpy() == obj].mean(axis=0) for obj in object_ids}
    train_state = np.stack([state_by_obj[o] for o in train_objects]).astype(np.float64)
    all_state = np.stack([state_by_obj[o] for o in object_ids]).astype(np.float64)
    scaler_mean = train_state.mean(axis=0)
    scaler_scale = train_state.std(axis=0, ddof=0)
    scaler_scale[scaler_scale <= 1e-12] = 1.0
    ztrain = (train_state - scaler_mean) / scaler_scale
    zall = (all_state - scaler_mean) / scaler_scale
    pca = PCA(n_components=int(cfg["preprocessing"]["pca_components"]), svd_solver="full")
    pca.fit(ztrain)
    coords = pca.transform(zall).astype(np.float64)
    coord_by_obj = {obj: coords[i] for i, obj in enumerate(object_ids)}

    gene_var = np.var(y[train_idx].astype(np.float64), axis=0, ddof=0)
    drug_keys = sorted(train_keys.full_inchikey.astype(str).unique())
    object_drug_response: dict[tuple[str, str], np.ndarray] = {}
    for (obj, drug), group in train_keys.groupby(["source_object_id", "full_inchikey"], sort=True):
        idx = group.index.to_numpy(dtype=np.int64)
        object_drug_response[(str(obj), str(drug))] = y[idx].astype(np.float64).mean(axis=0)
    drug_mean: dict[str, np.ndarray] = {}
    drug_objects: dict[str, list[str]] = {}
    for drug in drug_keys:
        objs = sorted(obj for obj, key in object_drug_response if key == drug)
        drug_objects[drug] = objs
        if not objs:
            continue
        drug_mean[drug] = np.stack([object_drug_response[(obj, drug)] for obj in objs]).mean(axis=0)

    predictions = {m: np.full((len(test_idx), y.shape[1]), np.nan, dtype=np.float32) for m in METHODS}
    predictions["zero_change"][:] = 0.0
    neighbors: list[dict[str, Any]] = []
    for local, (row_idx, row) in enumerate(zip(test_idx, test_keys.itertuples(index=False), strict=True)):
        obj, drug = str(row.source_object_id), str(row.full_inchikey)
        if drug in drug_mean:
            predictions["per_drug_mean"][local] = drug_mean[drug].astype(np.float32)
        candidates = drug_objects.get(drug, [])
        nearest_sorted = sorted(candidates, key=lambda cand: (float(np.linalg.norm(coord_by_obj[obj] - coord_by_obj[cand])), cand))
        random_sorted = sorted(candidates, key=lambda cand: (
            hashlib.sha256(f"{int(cfg['neighbor_selection']['random_hash_seed'])}|{fold}|{obj}|{drug}|{cand}".encode()).hexdigest(), cand))
        nearest_ids = nearest_sorted[:3]
        random_ids = random_sorted[:3]
        if len(nearest_ids) == 3:
            predictions["nearest3_state"][local] = np.stack([object_drug_response[(o, drug)] for o in nearest_ids]).mean(axis=0).astype(np.float32)
            predictions["random3_same_drug"][local] = np.stack([object_drug_response[(o, drug)] for o in random_ids]).mean(axis=0).astype(np.float32)
        nearest_distances = [float(np.linalg.norm(coord_by_obj[obj] - coord_by_obj[o])) for o in nearest_ids]
        neighbors.append({
            "fold_id": int(fold), "full_condition_key": str(row.full_condition_key),
            "test_source_object_id": obj, "full_inchikey": drug,
            "candidate_train_objects": len(candidates),
            "nearest3_source_object_ids": ";".join(nearest_ids),
            "nearest3_distances": ";".join(format(v, ".12g") for v in nearest_distances),
            "random3_source_object_ids": ";".join(random_ids),
            "neighbor_eligible": len(nearest_ids) == 3,
        })
    fold_keys = test_keys.copy()
    fold_keys["fold_id"] = int(fold)
    fold_y = y[test_idx].astype(np.float32)
    common = np.ones(len(test_idx), dtype=bool)
    for method in METHODS:
        common &= np.isfinite(predictions[method]).all(axis=1)
    fold_dir = out / "folds" / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(fold_dir / "predictions.npz",
        full_condition_key=fold_keys.full_condition_key.astype(str).to_numpy(dtype=str),
        source_object_id=fold_keys.source_object_id.astype(str).to_numpy(dtype=str),
        full_inchikey=fold_keys.full_inchikey.astype(str).to_numpy(dtype=str),
        fold_id=np.full(len(fold_keys), fold, dtype=np.int16), y_true=fold_y,
        common_denominator=common.astype(np.uint8),
        **{f"pred_{m}": predictions[m] for m in METHODS})
    preproc_payload = {
        "train_object_ids": np.asarray(train_objects, dtype=str),
        "test_object_ids": np.asarray(test_objects, dtype=str),
        "all_object_ids": np.asarray(object_ids, dtype=str),
        "all_object_state_mean_xctl": all_state.astype(np.float32),
        "scaler_mean": scaler_mean.astype(np.float64), "scaler_scale": scaler_scale.astype(np.float64),
        "pca_mean": pca.mean_.astype(np.float64), "pca_components": pca.components_.astype(np.float64),
        "pca_explained_variance": pca.explained_variance_.astype(np.float64),
        "object_pca_coordinates": coords.astype(np.float64), "gene_variance": gene_var.astype(np.float64),
    }
    np.savez_compressed(fold_dir / "preprocessing.npz", **preproc_payload)
    neighbor_frame = pd.DataFrame(neighbors)
    neighbor_frame.to_csv(fold_dir / "neighbor_selection.tsv", sep="\t", index=False)
    # Verify the first real fold's own disk roundtrip before applying the continuation gate.
    with np.load(fold_dir / "predictions.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["full_condition_key"], fold_keys.full_condition_key.to_numpy(dtype=str)):
            raise IOError("fold prediction write/read key mismatch")
        for method in METHODS:
            if not np.array_equal(saved[f"pred_{method}"], predictions[method], equal_nan=True):
                raise IOError(f"fold prediction write/read mismatch for {method}")
    elapsed = time.monotonic() - start
    receipt = {
        "fold_id": int(fold), "status": "PASS_FOLD_PREDICTIONS_WRITTEN_AND_READ_BACK",
        "train_conditions": int(len(train_idx)), "test_conditions": int(len(test_idx)),
        "train_objects": int(len(train_objects)), "test_objects": int(len(test_objects)),
        "train_test_object_overlap": 0,
        "train_drugs": int(len(drug_keys)), "neighbor_missing_conditions": int((~common).sum()),
        "all_methods_finite_on_common_denominator": True,
        "gene_variance_fitted_on_fold_train_deltas_only": True,
        "state_scaler_pca_fitted_on_fold_train_objects_only": True,
        "neighbor_selection_uses_pretreatment_state_only": True,
        "elapsed_wall_seconds": elapsed,
        "prediction_file_sha256": sha256_file(fold_dir / "predictions.npz"),
        "preprocessing_file_sha256": sha256_file(fold_dir / "preprocessing.npz"),
        "neighbor_file_sha256": sha256_file(fold_dir / "neighbor_selection.tsv"),
    }
    write_json(fold_dir / "fold_receipt.json", receipt)
    return fold_keys, {m: predictions[m] for m in METHODS}, preproc_payload, neighbor_frame


def metric_bundle(keys: pd.DataFrame, y: np.ndarray, preds: dict[str, np.ndarray],
                  fold_var: np.ndarray, common: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    object_tables: dict[str, pd.DataFrame] = {}
    full_mask = np.ones(len(keys), dtype=bool)
    for method in METHODS:
        all_summary, _ = object_macro_metrics(keys, y, preds[method], fold_var, full_mask)
        common_summary, obj = object_macro_metrics(keys, y, preds[method], fold_var, common)
        methods[method] = {"full_denominator": all_summary, "common_denominator": common_summary}
        object_tables[method] = obj
    object_common = pd.DataFrame({"source_object_id": sorted(set.intersection(*[
        set(object_tables[m].source_object_id.astype(str)) for m in METHODS]))})
    contrasts = {}
    for comparator in ("per_drug_mean", "random3_same_drug"):
        left = object_tables[comparator].set_index("source_object_id").nmse
        right = object_tables["nearest3_state"].set_index("source_object_id").nmse
        ids = object_common.source_object_id.astype(str).tolist()
        diffs = (left.loc[ids].to_numpy(dtype=float) - right.loc[ids].to_numpy(dtype=float))
        cfg_seed = int(cfg["metrics"]["paired_object_bootstrap_seed"])
        replicates = int(cfg["metrics"]["paired_object_bootstrap_replicates"])
        rng = np.random.default_rng(cfg_seed)
        draws = rng.integers(0, len(diffs), size=(replicates, len(diffs)))
        means = diffs[draws].mean(axis=1)
        contrasts[f"nearest3_vs_{comparator}"] = {
            "objects": len(ids), "object_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "mean_nmse_improvement": float(diffs.mean()),
            "paired_object_bootstrap_95pct_interval": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            "bootstrap_replicates": replicates, "seed": int(cfg_seed),
            "objects_nearest_better": int(np.sum(diffs > 0)),
            "positive_improvement_means_nearest_nmse_lower": True,
        }
    per_drug: list[dict[str, Any]] = []
    scored = keys.loc[common].copy()
    for drug, drug_rows in scored.groupby("full_inchikey", sort=True):
        idx = drug_rows.index.to_numpy(dtype=np.int64)
        drug_out: dict[str, Any] = {"full_inchikey": str(drug), "conditions": int(len(idx)),
                                   "objects": int(drug_rows.source_object_id.nunique())}
        for method in METHODS:
            drug_common = common & np.isin(np.arange(len(keys)), idx)
            local, _ = object_macro_metrics(keys, y, preds[method], fold_var, drug_common)
            drug_out[method] = local
        per_drug.append(drug_out)
    return {
        "primary_scope": "fixed_source TRAIN-only five-fold object-level OOF development comparison",
        "primary_metric": "object-macro NMSE using each row's fold-train gene variance; conditions averaged within object, objects equally weighted",
        "common_condition_count": int(common.sum()), "all_train_condition_count": int(len(keys)),
        "common_object_count": int(object_common.source_object_id.nunique()),
        "methods": methods, "paired_object_contrasts": contrasts, "per_drug_common_denominator": per_drug,
        "inner_dev_numeric_values_read": False, "pdo_outer_locked_panacea_cdsdb_numeric_values_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    cfg_path = (ROOT / args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("experiment_id") != "EXP-064" or cfg.get("run_id") != args.run_id:
        raise ValueError("EXP-064 config/run_id binding mismatch")
    out = ROOT / cfg["output"]["root"]
    attempt = out / cfg["output"]["run_attempt"]
    if args.run_id not in str(out) or attempt.exists():
        raise FileExistsError(f"run output exists or is not uniquely bound: {attempt}")
    if int(cfg["resources"]["gpu_hours_cap"]) != 0:
        raise ValueError("EXP-064 is CPU-only")
    attempt.mkdir(parents=True, exist_ok=False)
    wall_start, cpu_start = time.monotonic(), time.process_time()
    run_manifest: dict[str, Any] = {
        "experiment_id": "EXP-064", "run_id": args.run_id, "attempt_id": "fit_attempt_001",
        "status": "RUNNING_FIRST_COMPLETE_FOLD_COST_GATE", "config_sha256": sha256_file(cfg_path),
        "runner_sha256": sha256_file(Path(__file__)), "gpu_hours": 0,
        "protocol_id": cfg["protocol_id"], "started_unix": time.time(),
        "forbidden_numeric_values_read": {"inner_dev": False, "outer": False, "locked": False,
                                          "pdo": False, "panacea": False, "cdsdb": False},
    }
    write_json(attempt / "run_manifest.json", run_manifest)
    try:
        keys, genes, y, ctl, input_receipt = load_train_only(cfg)
        input_receipt.update({"gene_axis_count": int(len(genes)), "gene_axis_sha256_ordered_names":
                              hashlib.sha256("\n".join(genes.tolist()).encode()).hexdigest(),
                              "selected_train_full_condition_keys_sha256": hashlib.sha256("\n".join(keys.full_condition_key).encode()).hexdigest(),
                              "selected_train_object_ids": int(keys.source_object_id.nunique()),
                              "selected_train_full_inchikeys": int(keys.full_inchikey.nunique()),
                              "feature_access": "TRAIN rows only: X and matching X_ctl; pretreatment state from X_ctl",
                              "dev_expression_values_loaded": False})
        write_json(attempt / "train_only_input_receipt.json", input_receipt)
        seed, folds = int(cfg["split"]["object_hash_seed"]), int(cfg["split"]["folds"])
        obj_to_fold = {o: fold_for(str(o), seed, folds) for o in sorted(keys.source_object_id.astype(str).unique())}
        fold_sizes = {str(f): int(sum(v == f for v in obj_to_fold.values())) for f in range(folds)}
        fold_plan = keys[["full_condition_key", "source_object_id", "full_inchikey"]].copy()
        fold_plan["fold_id"] = fold_plan.source_object_id.astype(str).map(obj_to_fold).astype(int)
        fold_plan.to_csv(attempt / "fold_assignment.tsv", sep="\t", index=False)
        # One true fold includes all TRAIN data access, preprocessing, prediction and output roundtrip.
        load_seconds = float(time.monotonic() - wall_start)
        first = run_fold(0, cfg, keys, y, ctl, attempt)
        fkeys, fpreds, fpre, fneighbors = first
        fold0 = attempt / "folds" / "fold_0"
        fold0_compute = float(json.loads((fold0 / "fold_receipt.json").read_text(encoding="utf-8"))["elapsed_wall_seconds"])
        projected_fold_compute = max(5.0, fold0_compute * 1.5 * max(fold_sizes.values()) / max(1, fold_sizes.get("0", 1)))
        precharge = 60.0
        independent_readback_estimate = 180.0
        aggregate_write_estimate = 60.0
        projected_unmargined = precharge + load_seconds + fold0_compute + 4.0 * (projected_fold_compute + 5.0) + independent_readback_estimate + aggregate_write_estimate
        projected_with_margin = projected_unmargined * (1.0 + float(cfg["resources"]["full_schedule_margin_fraction"]))
        remaining_inherited = float(cfg["resources"]["inherited_exp055_cpu_remaining_start_seconds"])
        gate = {
            "status": "PASS_CONTINUE_ALL_REMAINING_FOLDS" if projected_with_margin <= min(
                float(cfg["resources"]["cpu_package_cap_seconds"]), remaining_inherited) else "CLOSED_COST_GATE_AFTER_FOLD0",
            "fold0_elapsed_wall_seconds": fold0_compute, "input_load_elapsed_wall_seconds": load_seconds,
            "fold_object_counts": fold_sizes, "projected_remaining_fold_seconds_each_with_imbalance_factor": projected_fold_compute,
            "preparation_precharge_seconds": precharge,
            "independent_readback_reserve_seconds": independent_readback_estimate,
            "aggregation_and_closeout_reserve_seconds": aggregate_write_estimate,
            "projected_total_unmargined_seconds": projected_unmargined,
            "projected_total_with_20pct_margin_seconds": projected_with_margin,
            "package_cpu_cap_seconds": cfg["resources"]["cpu_package_cap_seconds"],
            "inherited_exp055_remaining_seconds_at_start": remaining_inherited,
            "gate_limit_seconds": min(float(cfg["resources"]["cpu_package_cap_seconds"]), remaining_inherited),
            "gpu_hours": 0, "full_five_folds_authorized_by_cost_gate": projected_with_margin <= min(
                float(cfg["resources"]["cpu_package_cap_seconds"]), remaining_inherited),
        }
        write_json(attempt / "first_fold_cost_gate.json", gate)
        if gate["status"] != "PASS_CONTINUE_ALL_REMAINING_FOLDS":
            run_manifest.update({"status": "CLOSED_COST_GATE_AFTER_FOLD0", "finished_unix": time.time(),
                                 "elapsed_wall_seconds": time.monotonic() - wall_start,
                                 "process_cpu_seconds": time.process_time() - cpu_start,
                                 "full_five_folds_completed": False})
            write_json(attempt / "run_manifest.json", run_manifest)
            return 2
        fold_results = [first]
        for fold in range(1, folds):
            fold_results.append(run_fold(fold, cfg, keys, y, ctl, attempt))
        pred_by_key: dict[str, dict[str, np.ndarray]] = {}
        fold_variances = np.full((folds, y.shape[1]), np.nan, dtype=np.float64)
        prep_entries: dict[str, np.ndarray] = {}
        neighbor_frames = []
        for fold, (fk, p, prep, nf) in enumerate(fold_results):
            fi = fk.full_condition_key.astype(str).tolist()
            for i, key in enumerate(fi):
                pred_by_key[key] = {m: p[m][i] for m in METHODS}
            fold_variances[fold] = prep["gene_variance"]
            prep_entries[f"fold{fold}_gene_variance"] = prep["gene_variance"]
            for k, v in prep.items():
                prep_entries[f"fold{fold}_{k}"] = v
            neighbor_frames.append(nf)
        ordered = keys.sort_values("full_condition_key", kind="mergesort").reset_index(drop=True)
        y_ordered = np.stack([y[keys.index[keys.full_condition_key.eq(k)].item()] for k in ordered.full_condition_key]).astype(np.float32)
        fold_ids = ordered.source_object_id.astype(str).map(obj_to_fold).to_numpy(dtype=np.int16)
        predictions = {m: np.stack([pred_by_key[k][m] for k in ordered.full_condition_key]).astype(np.float32) for m in METHODS}
        common = np.ones(len(ordered), dtype=bool)
        for m in METHODS:
            common &= np.isfinite(predictions[m]).all(axis=1)
        ordered["fold_id"] = fold_ids
        keyed = {
            "full_condition_key": ordered.full_condition_key.astype(str).to_numpy(dtype=str),
            "source_object_id": ordered.source_object_id.astype(str).to_numpy(dtype=str),
            "full_inchikey": ordered.full_inchikey.astype(str).to_numpy(dtype=str),
            "source_row": ordered.source_row.to_numpy(dtype=np.int64), "fold_id": fold_ids,
            "y_true": y_ordered, "common_denominator": common.astype(np.uint8),
            "fold_gene_variance": fold_variances,
            **{f"pred_{m}": predictions[m] for m in METHODS},
        }
        np.savez_compressed(attempt / cfg["output"]["keyed_predictions"], **keyed)
        np.savez_compressed(attempt / cfg["output"]["fold_preprocessing"], **prep_entries)
        neighbor_table = pd.concat(neighbor_frames, ignore_index=True).sort_values(
            ["fold_id", "full_condition_key"], kind="mergesort")
        neighbor_table.to_csv(attempt / cfg["output"]["neighbor_manifest"], sep="\t", index=False)
        missing_summary = neighbor_table.groupby(["fold_id", "full_inchikey"], sort=True).agg(
            heldout_conditions=("full_condition_key", "size"),
            eligible_conditions=("neighbor_eligible", "sum"),
        ).reset_index()
        missing_summary["missing_conditions"] = missing_summary.heldout_conditions - missing_summary.eligible_conditions
        missing_summary.to_csv(attempt / cfg["output"]["missing_candidates_by_fold_drug"], sep="\t", index=False)
        bundle = metric_bundle(ordered, y_ordered, predictions, fold_variances, common, cfg)
        object_metric_frames = []
        for method in METHODS:
            _, om = object_macro_metrics(ordered, y_ordered, predictions[method], fold_variances, common)
            om = om.rename(columns={c: f"{method}_{c}" for c in om.columns if c != "source_object_id"})
            object_metric_frames.append(om)
        object_table = object_metric_frames[0]
        for om in object_metric_frames[1:]:
            object_table = object_table.merge(om, on="source_object_id", how="outer", validate="one_to_one")
        object_table.to_csv(attempt / "object_metrics.tsv", sep="\t", index=False)
        bundle["folds"] = {str(f): json.loads((attempt / "folds" / f"fold_{f}" / "fold_receipt.json").read_text(encoding="utf-8")) for f in range(folds)}
        bundle["denominators"] = {"train_conditions": len(ordered), "train_objects": ordered.source_object_id.nunique(),
                                  "full_inchikeys": ordered.full_inchikey.nunique(), "genes": int(y.shape[1]),
                                  "common_conditions": int(common.sum()),
                                  "common_objects": int(ordered.loc[common, "source_object_id"].nunique()),
                                  "neighbor_random_missing_conditions": int((~common).sum())}
        write_json(attempt / "metrics.json", bundle)
        manifest = {
            **run_manifest, "status": "COMPLETED_OOF_PENDING_ROOT_READBACK", "finished_unix": time.time(),
            "elapsed_wall_seconds": time.monotonic() - wall_start,
            "process_cpu_seconds": time.process_time() - cpu_start,
            "full_five_folds_completed": True, "train_only_input_receipt_sha256": sha256_file(attempt / "train_only_input_receipt.json"),
            "prediction_sha256": sha256_file(attempt / cfg["output"]["keyed_predictions"]),
            "preprocessing_sha256": sha256_file(attempt / cfg["output"]["fold_preprocessing"]),
            "neighbor_manifest_sha256": sha256_file(attempt / cfg["output"]["neighbor_manifest"]),
            "missing_candidate_summary_sha256": sha256_file(attempt / cfg["output"]["missing_candidates_by_fold_drug"]),
            "object_metrics_sha256": sha256_file(attempt / "object_metrics.tsv"),
            "metrics_sha256": sha256_file(attempt / "metrics.json"),
            "inner_dev_numeric_values_read": False, "gpu_hours": 0,
        }
        write_json(attempt / "run_manifest.json", manifest)
        return 0
    except Exception as exc:
        failure = {"status": "FAILED_CLOSED", "error_type": type(exc).__name__, "error": str(exc),
                   "elapsed_wall_seconds": time.monotonic() - wall_start,
                   "process_cpu_seconds": time.process_time() - cpu_start,
                   "inner_dev_numeric_values_read": False, "forbidden_numeric_values_read": False,
                   "gpu_hours": 0}
        write_json(attempt / "failure_receipt.json", failure)
        run_manifest.update({"status": "FAILED_CLOSED", "finished_unix": time.time(), **failure})
        write_json(attempt / "run_manifest.json", run_manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
