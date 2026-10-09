#!/usr/bin/env python3
"""Independent TRAIN-only numeric and identity readback for EXP-064."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("zero_change", "per_drug_mean", "nearest3_state", "random3_same_drug")


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_text(items: list[str]) -> str:
    return hashlib.sha256("\n".join(items).encode()).hexdigest()


def independently_read_train_values(cfg: dict[str, Any], frozen_keys: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read only the registered TRAIN X and paired X_ctl rows for independent replay."""
    data_cfg = cfg["data"]
    h5ad_path = Path(data_cfg["h5ad_path"])
    manifest_path = ROOT / data_cfg["metadata_freeze"]
    metadata_path = ROOT / data_cfg["condition_keys"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if digest_file(h5ad_path) != manifest.get("source_h5ad_sha256"):
        raise AssertionError("root verifier source H5AD fingerprint mismatch")
    if digest_file(metadata_path) != manifest.get("frozen_files_sha256", {}).get("response_blind_conditions.tsv"):
        raise AssertionError("root verifier frozen metadata fingerprint mismatch")
    train_source = frozen_keys.sort_values("source_row", key=lambda c: pd.to_numeric(c), kind="mergesort").reset_index(drop=True)
    source_rows = pd.to_numeric(train_source.source_row, errors="raise").to_numpy(dtype=np.int64)
    import anndata as ad
    source = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if "X_ctl" not in source.obsm.keys():
            raise AssertionError("root verifier source lacks paired X_ctl")
        gene_axis = pd.read_csv(ROOT / data_cfg["gene_axis"], sep="\t", dtype=str, keep_default_na=False)
        if len(gene_axis) != int(data_cfg["genes"]) or not np.array_equal(
                source.var_names.astype(str).to_numpy(), gene_axis.h5ad_var_name.astype(str).to_numpy()):
            raise AssertionError("root verifier H5AD gene axis differs from frozen mapping")
        expected_names = train_source.obs_name.astype(str).to_numpy()
        if not np.array_equal(np.asarray(source.obs_names[source_rows].astype(str)), expected_names):
            raise AssertionError("root verifier TRAIN observation names differ from frozen metadata")
        source_obs = source.obs.iloc[source_rows]
        for h5col, keycol in (("cell_iname", "source_object_id"), ("pert_id", "pert_id"), ("sample_id", "sample_id")):
            if h5col not in source_obs.columns or not np.array_equal(
                    source_obs[h5col].astype(str).str.strip().to_numpy(), train_source[keycol].astype(str).to_numpy()):
                raise AssertionError(f"root verifier TRAIN row identity mismatch: {h5col}")
        def dense(value: Any) -> np.ndarray:
            if hasattr(value, "toarray"):
                value = value.toarray()
            return np.asarray(value, dtype=np.float64)
        root_x = dense(source.X[source_rows, :])
        root_ctl = dense(source.obsm["X_ctl"][source_rows, :])
    finally:
        source.file.close()
    expected_shape = (len(train_source), int(data_cfg["genes"]))
    if root_x.shape != expected_shape or root_ctl.shape != expected_shape:
        raise AssertionError("root verifier TRAIN numeric source shape mismatch")
    if not np.isfinite(root_x).all() or not np.isfinite(root_ctl).all():
        raise AssertionError("root verifier TRAIN numeric source contains nonfinite values")
    root_y = root_x - root_ctl
    if not np.isfinite(root_y).all() or not np.allclose(root_x, root_ctl + root_y, rtol=0, atol=1e-10):
        raise AssertionError("root verifier TRAIN response reconstruction failed")
    by_key = dict(zip(train_source.full_condition_key.astype(str), root_y, strict=True))
    return root_x, root_ctl, np.stack([by_key[str(k)] for k in frozen_keys.sort_values(
        "full_condition_key", kind="mergesort").full_condition_key]).astype(np.float64)


def object_fold(obj: str, seed: int, nfolds: int) -> int:
    token = hashlib.sha256(f"{seed}|{obj}".encode()).digest()
    return int.from_bytes(token[:8], "big") % nfolds


def close(a: float | None, b: float | None, atol: float = 5e-7, rtol: float = 5e-7) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)


def metrics(keys: pd.DataFrame, truth: np.ndarray, pred: np.ndarray,
            fold_var: np.ndarray, mask: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    fids = keys.fold_id.to_numpy(dtype=int)
    row_nmse = np.full(len(keys), np.nan)
    row_mae = np.full(len(keys), np.nan)
    row_direction = np.full(len(keys), np.nan)
    row_corr = np.full(len(keys), np.nan)
    row_amp = np.full(len(keys), np.nan)
    effective_mask = mask & np.isfinite(truth).all(axis=1) & np.isfinite(pred).all(axis=1)
    for i in np.flatnonzero(effective_mask):
        use = np.isfinite(fold_var[fids[i]]) & (fold_var[fids[i]] > 1e-12)
        if not use.any():
            continue
        t = truth[i].astype(np.float64)
        p = pred[i].astype(np.float64)
        row_nmse[i] = float(np.mean(np.square(p[use] - t[use]) / fold_var[fids[i], use]))
        row_mae[i] = float(np.mean(np.abs(p - t)))
        row_direction[i] = float(np.mean(np.sign(p) == np.sign(t)))
        if float(np.std(t)) > 0 and float(np.std(p)) > 0:
            row_corr[i] = float(np.corrcoef(t, p)[0, 1])
        if float(np.mean(np.abs(t))) > 0:
            row_amp[i] = float(np.mean(np.abs(p)) / np.mean(np.abs(t)))
    frame = keys.loc[effective_mask].copy()
    frame["nmse"] = row_nmse[effective_mask]
    frame["mae"] = row_mae[effective_mask]
    frame["direction"] = row_direction[effective_mask]
    frame["correlation"] = row_corr[effective_mask]
    frame["amplitude_ratio"] = row_amp[effective_mask]
    obj = frame.groupby("source_object_id", sort=True).agg(
        nmse=("nmse", "mean"), mae=("mae", "mean"), direction=("direction", "mean"),
        correlation=("correlation", "mean"), amplitude_ratio=("amplitude_ratio", "mean"),
        conditions=("full_condition_key", "size"),
    ).reset_index()
    amp_by_object: dict[str, float] = {}
    for object_id in obj.source_object_id.astype(str):
        idx = np.flatnonzero(effective_mask & keys.source_object_id.astype(str).eq(object_id).to_numpy())
        target_scale = float(np.mean(np.abs(truth[idx].astype(np.float64)))) if len(idx) else 0.0
        amp_by_object[object_id] = (float(np.mean(np.abs(pred[idx].astype(np.float64))) / target_scale)
                                    if target_scale > 0 else float("nan"))
    obj["amplitude_ratio"] = obj.source_object_id.astype(str).map(amp_by_object)
    result = {
        "object_macro_nmse": float(obj.nmse.mean()) if np.isfinite(obj.nmse).any() else None,
        "object_macro_mae": float(obj.mae.mean()) if np.isfinite(obj.mae).any() else None,
        "object_macro_direction_agreement": float(obj.direction.mean()) if np.isfinite(obj.direction).any() else None,
        "object_macro_gene_correlation": float(obj.correlation.mean()) if np.isfinite(obj.correlation).any() else None,
        "object_macro_amplitude_ratio": float(obj.amplitude_ratio.mean()) if np.isfinite(obj.amplitude_ratio).any() else None,
        "condition_denominator": int(effective_mask.sum()), "requested_condition_count": int(mask.sum()),
        "excluded_nonfinite_prediction_conditions": int(mask.sum() - effective_mask.sum()),
        "object_denominator": int(np.isfinite(obj.nmse).sum()),
        "mae_object_denominator": int(np.isfinite(obj.mae.to_numpy(dtype=float)).sum()),
        "direction_object_denominator": int(np.isfinite(obj.direction.to_numpy(dtype=float)).sum()),
        "correlation_object_denominator": int(np.isfinite(obj.correlation.to_numpy(dtype=float)).sum()),
        "amplitude_object_denominator": int(np.isfinite(obj.amplitude_ratio.to_numpy(dtype=float)).sum()),
        "undefined_correlation_objects": int((~np.isfinite(obj.correlation)).sum()),
        "undefined_amplitude_objects": int((~np.isfinite(obj.amplitude_ratio)).sum()),
    }
    return result, obj


def assert_metrics_equal(a: dict[str, Any], b: dict[str, Any], label: str) -> None:
    for key in ("condition_denominator", "requested_condition_count", "excluded_nonfinite_prediction_conditions",
                "object_denominator", "mae_object_denominator", "direction_object_denominator",
                "correlation_object_denominator", "amplitude_object_denominator",
                "undefined_correlation_objects", "undefined_amplitude_objects"):
        if a.get(key) != b.get(key):
            raise AssertionError(f"{label}.{key} mismatch: {a.get(key)} != {b.get(key)}")
    for key in ("object_macro_nmse", "object_macro_mae", "object_macro_direction_agreement",
                "object_macro_gene_correlation", "object_macro_amplitude_ratio"):
        if not close(a.get(key), b.get(key)):
            raise AssertionError(f"{label}.{key} numeric mismatch: {a.get(key)} != {b.get(key)}")


def verify(config_path: Path, run_id: str) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg.get("experiment_id") != "EXP-064" or cfg.get("run_id") != run_id:
        raise ValueError("config/run identity mismatch")
    attempt = ROOT / cfg["output"]["root"] / cfg["output"]["run_attempt"]
    if run_id not in str(attempt):
        raise ValueError("attempt path is not bound to run id")
    manifest = json.loads((attempt / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED_OOF_PENDING_ROOT_READBACK" or not manifest.get("full_five_folds_completed"):
        raise ValueError("runner did not complete all five folds")
    receipt = json.loads((attempt / "train_only_input_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("inner_dev_numeric_values_read") is not False or receipt.get("outer_locked_pdo_panacea_cdsdb_numeric_values_read") is not False:
        raise ValueError("input receipt does not affirm numeric isolation")
    pred_path = attempt / cfg["output"]["keyed_predictions"]
    prep_path = attempt / cfg["output"]["fold_preprocessing"]
    neighbor_path = attempt / cfg["output"]["neighbor_manifest"]
    missing_path = attempt / cfg["output"]["missing_candidates_by_fold_drug"]
    metric_path = attempt / "metrics.json"
    object_path = attempt / "object_metrics.tsv"
    for p, expected in ((pred_path, manifest["prediction_sha256"]), (prep_path, manifest["preprocessing_sha256"]),
                        (neighbor_path, manifest["neighbor_manifest_sha256"]),
                        (missing_path, manifest["missing_candidate_summary_sha256"]),
                        (object_path, manifest["object_metrics_sha256"]), (metric_path, manifest["metrics_sha256"])):
        if digest_file(p) != expected:
            raise ValueError(f"frozen result output hash mismatch: {p.name}")
    frozen_keys = pd.read_csv(ROOT / cfg["data"]["condition_keys"], sep="\t", dtype=str, keep_default_na=False)
    frozen_keys = frozen_keys.loc[frozen_keys.effective_split_role.eq("train")].copy()
    if len(frozen_keys) != int(cfg["data"]["condition_count"]):
        raise AssertionError("frozen TRAIN table denominator mismatch")
    frozen_keys = frozen_keys.sort_values("full_condition_key", kind="mergesort").reset_index(drop=True)
    with np.load(pred_path, allow_pickle=False) as z:
        keys = {name: z[name] for name in z.files}
    key_names = ["full_condition_key", "source_object_id", "full_inchikey", "source_row", "fold_id", "y_true", "common_denominator", "fold_gene_variance"] + [f"pred_{m}" for m in METHODS]
    if set(keys) != set(key_names):
        raise AssertionError("prediction archive schema mismatch")
    if not np.array_equal(keys["full_condition_key"], frozen_keys.full_condition_key.astype(str).to_numpy()):
        raise AssertionError("condition key set/order differs from the fixed TRAIN domain")
    if not np.array_equal(keys["source_object_id"], frozen_keys.source_object_id.astype(str).to_numpy()):
        raise AssertionError("source-object identities differ from frozen TRAIN metadata")
    if not np.array_equal(keys["full_inchikey"], frozen_keys.full_inchikey.astype(str).to_numpy()):
        raise AssertionError("chemical identity differs from frozen TRAIN metadata")
    if not np.array_equal(keys["source_row"], pd.to_numeric(frozen_keys.source_row).to_numpy(dtype=np.int64)):
        raise AssertionError("source row indices differ from frozen TRAIN metadata")
    if keys["y_true"].shape != (len(frozen_keys), int(cfg["data"]["genes"])):
        raise AssertionError("OOF target shape mismatch")
    root_x, root_ctl, y_root_ordered = independently_read_train_values(cfg, frozen_keys)
    if not np.allclose(y_root_ordered, keys["y_true"], rtol=1e-6, atol=2e-6):
        raise AssertionError("keyed OOF targets differ from independently reconstructed TRAIN X-X_ctl")
    train_source_rows = frozen_keys.sort_values("source_row", key=lambda c: pd.to_numeric(c), kind="mergesort").reset_index(drop=True)
    root_state_by_object: dict[str, np.ndarray] = {}
    for object_id, group in train_source_rows.groupby("source_object_id", sort=True):
        positions = group.index.to_numpy(dtype=np.int64)
        root_state_by_object[str(object_id)] = root_ctl[positions].mean(axis=0)
    if not np.isfinite(keys["y_true"]).all():
        raise AssertionError("non-finite observed TRAIN response target")
    for method in METHODS:
        if keys[f"pred_{method}"].shape != keys["y_true"].shape:
            raise AssertionError(f"prediction shape mismatch for {method}")
    fold_count = int(cfg["split"]["folds"])
    seed = int(cfg["split"]["object_hash_seed"])
    expected_folds = np.asarray([object_fold(str(o), seed, fold_count) for o in keys["source_object_id"]], dtype=np.int16)
    if not np.array_equal(expected_folds, keys["fold_id"]):
        raise AssertionError("fold assignment is not the preregistered object hash")
    if keys["fold_gene_variance"].shape != (fold_count, int(cfg["data"]["genes"])):
        raise AssertionError("fold gene-variance shape mismatch")
    prep_archive = np.load(prep_path, allow_pickle=False)
    try:
        preproc = {name: prep_archive[name] for name in prep_archive.files}
    finally:
        prep_archive.close()
    neighbors = pd.read_csv(neighbor_path, sep="\t", dtype=str, keep_default_na=False)
    if not set(neighbors.neighbor_eligible.unique()).issubset({"True", "False"}):
        raise AssertionError("neighbor eligibility field is not a strict boolean")
    neighbors["neighbor_eligible"] = neighbors.neighbor_eligible.map({"True": True, "False": False})
    neighbors["fold_id"] = pd.to_numeric(neighbors.fold_id, errors="raise").astype(int)
    expected_neighbors: list[dict[str, Any]] = []
    response_recompute_max_error = 0.0
    variance_max_error = 0.0
    for fold in range(fold_count):
        fold_dir = attempt / "folds" / f"fold_{fold}"
        fold_receipt = json.loads((fold_dir / "fold_receipt.json").read_text(encoding="utf-8"))
        if fold_receipt.get("status") != "PASS_FOLD_PREDICTIONS_WRITTEN_AND_READ_BACK":
            raise AssertionError(f"fold {fold} has no successful prediction readback")
        train_mask = keys["fold_id"] != fold
        test_mask = ~train_mask
        train_objects = sorted(set(keys["source_object_id"][train_mask].astype(str)))
        test_objects = sorted(set(keys["source_object_id"][test_mask].astype(str)))
        if set(train_objects) & set(test_objects):
            raise AssertionError(f"object split leak in fold {fold}")
        var = np.var(y_root_ordered[train_mask].astype(np.float64), axis=0, ddof=0)
        variance_max_error = max(variance_max_error, float(np.max(np.abs(var - keys["fold_gene_variance"][fold]))))
        if not np.allclose(var, keys["fold_gene_variance"][fold], atol=1e-10, rtol=1e-10):
            raise AssertionError(f"fold {fold} variance not computed from other TRAIN objects")
        # Rebuild each same-drug object response from only non-held-out OOF targets.
        tr_frame = pd.DataFrame({"obj": keys["source_object_id"][train_mask].astype(str),
                                 "drug": keys["full_inchikey"][train_mask].astype(str),
                                 "idx": np.flatnonzero(train_mask)})
        obj_resp: dict[tuple[str, str], np.ndarray] = {}
        for (obj, drug), g in tr_frame.groupby(["obj", "drug"], sort=True):
            obj_resp[(str(obj), str(drug))] = y_root_ordered[g.idx.to_numpy(dtype=int)].astype(np.float64).mean(axis=0)
        drug_resp: dict[str, np.ndarray] = {}
        drug_objs: dict[str, list[str]] = {}
        for drug in sorted(tr_frame.drug.unique()):
            objs = sorted(obj for obj, d in obj_resp if d == drug)
            drug_objs[str(drug)] = objs
            drug_resp[str(drug)] = np.stack([obj_resp[(o, str(drug))] for o in objs]).mean(axis=0)
        fold_test_idx = np.flatnonzero(test_mask)
        if not np.array_equal(keys["pred_zero_change"][test_mask], np.zeros_like(keys["y_true"][test_mask])):
            raise AssertionError(f"zero-change arm is not zero in fold {fold}")
        for idx in fold_test_idx:
            drug = str(keys["full_inchikey"][idx])
            if drug not in drug_resp:
                raise AssertionError("test drug has no same-drug TRAIN response mean")
            response_recompute_max_error = max(response_recompute_max_error,
                float(np.max(np.abs(keys["pred_per_drug_mean"][idx].astype(np.float64) - drug_resp[drug]))))
            if not np.allclose(keys["pred_per_drug_mean"][idx], drug_resp[drug], atol=2e-6, rtol=2e-6):
                raise AssertionError("per-drug mean prediction does not match other-fold TRAIN targets")
        # Independently recompute preprocessing geometry from the saved pretreatment-only state matrix.
        prefix = f"fold{fold}_"
        needed = ["train_object_ids", "test_object_ids", "all_object_ids", "all_object_state_mean_xctl",
                  "scaler_mean", "scaler_scale", "pca_mean", "pca_components", "object_pca_coordinates"]
        if any(prefix + name not in preproc for name in needed):
            raise AssertionError(f"fold {fold} preprocessing artifacts incomplete")
        if not set(train_objects).issubset(set(preproc[prefix + "train_object_ids"].astype(str))):
            raise AssertionError("preprocessing training object list does not cover exact fold training objects")
        if set(train_objects) != set(preproc[prefix + "train_object_ids"].astype(str)):
            raise AssertionError("preprocessing includes an object outside fold training objects")
        if set(test_objects) != set(preproc[prefix + "test_object_ids"].astype(str)):
            raise AssertionError("preprocessing held-out object list mismatch")
        obj_ids = preproc[prefix + "all_object_ids"].astype(str)
        state = preproc[prefix + "all_object_state_mean_xctl"].astype(np.float64)
        root_state = np.stack([root_state_by_object[str(o)] for o in obj_ids]).astype(np.float64)
        if not np.allclose(root_state, state, atol=2e-6, rtol=1e-7):
            raise AssertionError(f"fold {fold} pretreatment state means differ from independently read TRAIN X_ctl")
        sc_mean = preproc[prefix + "scaler_mean"].astype(np.float64)
        sc_scale = preproc[prefix + "scaler_scale"].astype(np.float64)
        pmean = preproc[prefix + "pca_mean"].astype(np.float64)
        comps = preproc[prefix + "pca_components"].astype(np.float64)
        coord_saved = preproc[prefix + "object_pca_coordinates"].astype(np.float64)
        train_state_root = np.stack([root_state_by_object[str(o)] for o in train_objects]).astype(np.float64)
        expected_sc_mean = train_state_root.mean(axis=0)
        expected_sc_scale = train_state_root.std(axis=0, ddof=0)
        expected_sc_scale[expected_sc_scale <= 1e-12] = 1.0
        if not np.allclose(expected_sc_mean, sc_mean, atol=1e-9, rtol=1e-9) or not np.allclose(expected_sc_scale, sc_scale, atol=1e-9, rtol=1e-9):
            raise AssertionError(f"fold {fold} state scaler was not fitted on the fold training objects")
        from sklearn.decomposition import PCA
        ztrain_root = (train_state_root - expected_sc_mean) / expected_sc_scale
        zall_root = (root_state - expected_sc_mean) / expected_sc_scale
        pca_root = PCA(n_components=int(cfg["preprocessing"]["pca_components"]), svd_solver="full").fit(ztrain_root)
        if not np.allclose(pca_root.mean_, pmean, atol=1e-9, rtol=1e-9) or not np.allclose(pca_root.components_, comps, atol=1e-8, rtol=1e-8):
            raise AssertionError(f"fold {fold} PCA parameters do not refit from fold TRAIN X_ctl objects")
        coord_calc = pca_root.transform(zall_root)
        if not np.allclose(coord_calc, coord_saved, atol=1e-7, rtol=1e-7):
            raise AssertionError(f"fold {fold} pretreatment PCA transform readback mismatch")
        coord_map = {str(o): coord_calc[i] for i, o in enumerate(obj_ids)}
        train_state_from_rows: dict[str, np.ndarray] = {}
        train_indices = np.flatnonzero(train_mask)
        for obj in train_objects:
            ii = train_indices[keys["source_object_id"][train_indices].astype(str) == obj]
            if len(ii) == 0:
                raise AssertionError("fold training object missing response rows")
        fold_neighbor = neighbors.loc[neighbors.fold_id.astype(int).eq(fold)]
        if len(fold_neighbor) != int(test_mask.sum()):
            raise AssertionError(f"fold {fold} neighbor manifest row denominator mismatch")
        # Neighbor choices must be nearest by pretreatment coordinates and same exact drug.
        for idx in fold_test_idx:
            obj = str(keys["source_object_id"][idx])
            drug = str(keys["full_inchikey"][idx])
            row_key = str(keys["full_condition_key"][idx])
            row = fold_neighbor.loc[fold_neighbor.full_condition_key.eq(row_key)]
            if len(row) != 1:
                raise AssertionError("neighbor selection row key is not unique")
            record = row.iloc[0]
            candidates = drug_objs[drug]
            ranked = sorted(candidates, key=lambda cand: (float(np.linalg.norm(coord_map[obj] - coord_map[cand])), cand))
            random_ranked = sorted(candidates, key=lambda cand: (
                hashlib.sha256(f"{int(cfg['neighbor_selection']['random_hash_seed'])}|{fold}|{obj}|{drug}|{cand}".encode()).hexdigest(), cand))
            expected_near, expected_random = ranked[:3], random_ranked[:3]
            if record.nearest3_source_object_ids.split(";") != expected_near:
                raise AssertionError("nearest-object selection is not the frozen pretreatment-only nearest 3")
            if record.random3_source_object_ids.split(";") != expected_random:
                raise AssertionError("random-object comparator differs from its fixed hash draw")
            if len(set(expected_near)) != 3 or len(set(expected_random)) != 3 or not set(expected_near + expected_random).issubset(set(train_objects)):
                raise AssertionError("same-drug neighbor identity or object isolation failed")
            if any((n, drug) not in obj_resp for n in expected_near + expected_random):
                raise AssertionError("selected same-drug source object has no fold-train response")
            mean_near = np.stack([obj_resp[(n, drug)] for n in expected_near]).mean(axis=0)
            mean_random = np.stack([obj_resp[(n, drug)] for n in expected_random]).mean(axis=0)
            response_recompute_max_error = max(response_recompute_max_error,
                float(np.max(np.abs(keys["pred_nearest3_state"][idx].astype(np.float64) - mean_near))),
                float(np.max(np.abs(keys["pred_random3_same_drug"][idx].astype(np.float64) - mean_random))))
            if not np.allclose(keys["pred_nearest3_state"][idx], mean_near, atol=2e-6, rtol=2e-6):
                raise AssertionError("nearest-neighbor response prediction differs from selected TRAIN objects")
            if not np.allclose(keys["pred_random3_same_drug"][idx], mean_random, atol=2e-6, rtol=2e-6):
                raise AssertionError("random-three response prediction differs from selected TRAIN objects")
        # Per-fold output must exactly match the corresponding rows in the aggregate archive.
        with np.load(fold_dir / "predictions.npz", allow_pickle=False) as local:
            local_keys = local["full_condition_key"].astype(str)
            aggregate_pos = {str(k): i for i, k in enumerate(keys["full_condition_key"].astype(str))}
            for local_i, key in enumerate(local_keys):
                ai = aggregate_pos[key]
                if int(keys["fold_id"][ai]) != fold:
                    raise AssertionError("fold archive includes a key from another fold")
                if not np.array_equal(local["y_true"][local_i], keys["y_true"][ai]):
                    raise AssertionError("fold target differs from aggregate keyed target")
                for m in METHODS:
                    if not np.array_equal(local[f"pred_{m}"][local_i], keys[f"pred_{m}"][ai], equal_nan=True):
                        raise AssertionError("fold prediction differs from aggregate OOF prediction")
    # Recreate the intersection denominator and all absolute metrics independently.
    common = np.ones(len(frozen_keys), dtype=bool)
    for method in METHODS:
        common &= np.isfinite(keys[f"pred_{method}"]).all(axis=1)
    if not np.array_equal(common.astype(np.uint8), keys["common_denominator"]):
        raise AssertionError("common denominator mask is not the intersection of all four arms")
    saved_metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    recalc: dict[str, Any] = {}
    readback_objects: dict[str, pd.DataFrame] = {}
    for method in METHODS:
        full_m, full_obj = metrics(frozen_keys.assign(fold_id=keys["fold_id"]), keys["y_true"], keys[f"pred_{method}"], keys["fold_gene_variance"], np.ones(len(frozen_keys), dtype=bool))
        common_m, common_obj = metrics(frozen_keys.assign(fold_id=keys["fold_id"]), keys["y_true"], keys[f"pred_{method}"], keys["fold_gene_variance"], common)
        recalc[method] = {"full_denominator": full_m, "common_denominator": common_m, "object_table": common_obj}
        readback_objects[method] = common_obj
        assert_metrics_equal(saved_metrics["methods"][method]["full_denominator"], full_m, f"{method}.full")
        assert_metrics_equal(saved_metrics["methods"][method]["common_denominator"], common_m, f"{method}.common")
    if int(saved_metrics["common_condition_count"]) != int(common.sum()):
        raise AssertionError("summary common-denominator condition count mismatch")
    if int(saved_metrics["all_train_condition_count"]) != len(frozen_keys):
        raise AssertionError("summary full TRAIN condition count mismatch")
    per_drug_check_rows = []
    for drug, drug_rows in frozen_keys.loc[common].groupby("full_inchikey", sort=True):
        idx = drug_rows.index.to_numpy(dtype=np.int64)
        record: dict[str, Any] = {"full_inchikey": str(drug), "conditions": len(idx),
                                  "objects": int(drug_rows.source_object_id.nunique())}
        for method in METHODS:
            dm, _ = metrics(frozen_keys.assign(fold_id=keys["fold_id"]), keys["y_true"], keys[f"pred_{method}"],
                            keys["fold_gene_variance"], np.isin(np.arange(len(frozen_keys)), idx))
            record[method] = dm
        per_drug_check_rows.append(record)
    if len(per_drug_check_rows) != len(saved_metrics["per_drug_common_denominator"]):
        raise AssertionError("per-drug metric row count mismatch")
    for expected, actual in zip(saved_metrics["per_drug_common_denominator"], per_drug_check_rows, strict=True):
        if expected["full_inchikey"] != actual["full_inchikey"] or expected["conditions"] != actual["conditions"] or expected["objects"] != actual["objects"]:
            raise AssertionError("per-drug metric identity/denominator mismatch")
        for method in METHODS:
            assert_metrics_equal(expected[method], actual[method], f"per-drug.{expected['full_inchikey']}.{method}")
    object_ids = sorted(set.intersection(*[set(readback_objects[m].source_object_id.astype(str)) for m in METHODS]))
    paired = {}
    for comparator in ("per_drug_mean", "random3_same_drug"):
        lm = readback_objects[comparator].set_index("source_object_id").nmse
        rn = readback_objects["nearest3_state"].set_index("source_object_id").nmse
        diffs = lm.loc[object_ids].to_numpy(dtype=float) - rn.loc[object_ids].to_numpy(dtype=float)
        rng = np.random.default_rng(int(cfg["metrics"]["paired_object_bootstrap_seed"]))
        draws = rng.integers(0, len(diffs), size=(int(cfg["metrics"]["paired_object_bootstrap_replicates"]), len(diffs)))
        means = diffs[draws].mean(axis=1)
        interval = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
        stored = saved_metrics["paired_object_contrasts"][f"nearest3_vs_{comparator}"]
        if int(stored["objects"]) != len(object_ids) or not np.allclose(stored["paired_object_bootstrap_95pct_interval"], interval, atol=1e-12, rtol=1e-12):
            raise AssertionError("paired object bootstrap does not reproduce")
        if not close(float(stored["mean_nmse_improvement"]), float(diffs.mean())):
            raise AssertionError("paired object mean contrast mismatch")
        paired[f"nearest3_vs_{comparator}"] = {"objects": len(object_ids), "mean_nmse_improvement": float(diffs.mean()),
                                                 "paired_object_bootstrap_95pct_interval": interval,
                                                 "objects_nearest_better": int(np.sum(diffs > 0))}
    saved_object_table = pd.read_csv(object_path, sep="\t", dtype=str, keep_default_na=False)
    if len(saved_object_table) != len(object_ids) or saved_object_table.source_object_id.astype(str).tolist() != object_ids:
        raise AssertionError("per-object output identity/denominator mismatch")
    for method in METHODS:
        independent_obj = readback_objects[method].sort_values("source_object_id", kind="mergesort").reset_index(drop=True)
        for col in ("nmse", "mae", "direction", "correlation", "amplitude_ratio", "conditions"):
            stored_col = f"{method}_{col}"
            if stored_col not in saved_object_table:
                raise AssertionError(f"per-object output missing {stored_col}")
            expected_values = independent_obj[col].to_numpy(dtype=float)
            stored_values = pd.to_numeric(saved_object_table[stored_col], errors="coerce").to_numpy(dtype=float)
            if not np.allclose(expected_values, stored_values, equal_nan=True, atol=5e-7, rtol=5e-7):
                raise AssertionError(f"per-object output numeric mismatch for {stored_col}")
    missing = pd.read_csv(missing_path, sep="\t", dtype=str, keep_default_na=False)
    independent_missing = neighbors.groupby(["fold_id", "full_inchikey"], sort=True).agg(
        heldout_conditions=("full_condition_key", "size"), eligible_conditions=("neighbor_eligible", "sum")).reset_index()
    independent_missing["missing_conditions"] = independent_missing.heldout_conditions - independent_missing.eligible_conditions
    if not missing.astype(str).equals(independent_missing.astype(str)):
        raise AssertionError("missing-candidate summary does not match per-fold neighbor manifest")
    cpu = time.process_time()
    return {
        "status": "PASS_ROOT_INDEPENDENT_NUMERIC_READBACK",
        "run_id": run_id, "fold_count": fold_count,
        "train_conditions": int(len(frozen_keys)), "train_objects": int(frozen_keys.source_object_id.nunique()),
        "full_inchikey_count": int(frozen_keys.full_inchikey.nunique()), "gene_count": int(keys["y_true"].shape[1]),
        "common_conditions": int(common.sum()), "common_objects": int(frozen_keys.loc[common, "source_object_id"].nunique()),
        "row_condition_key_sha256": digest_text(keys["full_condition_key"].astype(str).tolist()),
        "object_ids_disjoint_by_fold": True, "nearest_selection_recomputed_from_pretreatment_pca_only": True,
        "responses_recomputed_from_other_fold_train_targets_only": True,
        "train_X_Xctl_numeric_values_independently_reconstructed": True,
        "train_X_numeric_sha256": hashlib.sha256(np.ascontiguousarray(root_x).tobytes()).hexdigest(),
        "train_Xctl_numeric_sha256": hashlib.sha256(np.ascontiguousarray(root_ctl).tobytes()).hexdigest(),
        "train_delta_recomputed_sha256": hashlib.sha256(np.ascontiguousarray(root_x - root_ctl).tobytes()).hexdigest(),
        "fold_scaler_and_pca_refitted_from_root_read_train_Xctl": True,
        "max_response_prediction_recalculation_abs_error": response_recompute_max_error,
        "max_fold_variance_recalculation_abs_error": variance_max_error,
        "metrics_independently_recomputed": {m: {"full_denominator": recalc[m]["full_denominator"],
                                                  "common_denominator": recalc[m]["common_denominator"]} for m in METHODS},
        "paired_object_contrasts_independently_recomputed": paired,
        "per_drug_metrics_independently_recomputed": per_drug_check_rows,
        "per_object_metrics_rows": int(len(saved_object_table)),
        "per_drug_metrics_rows": int(len(saved_metrics["per_drug_common_denominator"])),
        "neighbor_selection_rows": int(len(neighbors)),
        "forbidden_numeric_values_read": {"inner_dev": False, "outer": False, "locked": False,
                                           "pdo": False, "panacea": False, "cdsdb": False},
        "elapsed_process_cpu_seconds": cpu,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    config_path = ROOT / args.config
    out = ROOT / "artifacts/experiments/EXP-064" / args.run_id / "fit_attempt_001"
    try:
        receipt = verify(config_path, args.run_id)
    except Exception as exc:
        fail_path = out / "root_independent_readback_failure_v2.json"
        if fail_path.exists():
            raise FileExistsError("root readback failure receipt exists; refusing overwrite") from exc
        failure = {
            "status": "FAILED_ROOT_INDEPENDENT_READBACK_V2",
            "run_id": args.run_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_wall_seconds": time.monotonic() - started,
            "forbidden_numeric_values_read": {"inner_dev": False, "outer": False, "locked": False,
                                                "pdo": False, "panacea": False, "cdsdb": False},
        }
        fail_path.write_text(json.dumps(failure, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        raise
    receipt["elapsed_wall_seconds"] = time.monotonic() - started
    receipt["runner_sha256"] = digest_file(Path(__file__))
    receipt["prediction_sha256"] = digest_file(out / "keyed_oof_predictions.npz")
    receipt["preprocessing_sha256"] = digest_file(out / "fold_preprocessing.npz")
    receipt["neighbor_manifest_sha256"] = digest_file(out / "neighbor_selection.tsv")
    receipt["metrics_sha256"] = digest_file(out / "metrics.json")
    receipt["object_metrics_sha256"] = digest_file(out / "object_metrics.tsv")
    target = out / "root_independent_readback.json"
    if target.exists():
        raise FileExistsError("root readback receipt path already exists; refusing overwrite")
    target.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
