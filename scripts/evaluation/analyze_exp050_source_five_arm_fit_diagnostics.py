"""Diagnose the fixed EXP-050 five-arm fit using only the nine CTRPv2 TRAIN objects.

This script deliberately never calls the v3 outer-fold prediction routine. For each
fold it materializes only the eight outer-training objects, then computes their
in-sample fit and inner leave-one-object-out predictions. The outer held-out
object's numeric AAC values are never extracted.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20261006_exp050_source_five_arm_diagnostics_v1"
V3_ID = "20261006_exp050_source_five_arm_development_v3"
V3_DIR = ROOT / "artifacts/experiments/EXP-050" / V3_ID
OUT_DIR = ROOT / "artifacts/experiments/EXP-050" / RUN_ID
V3_RUNNER = ROOT / "scripts/evaluation/evaluate_exp050_source_five_arm_development.py"
DIAGNOSTIC_SCRIPT = Path(__file__).resolve()
OUTER_ARMS = ("direct", "predicted_perturbation", "fusion", "equal_capacity_direct", "shuffled_perturbation")
INNER_ARMS = ("direct", "predicted_perturbation", "equal_capacity_direct")
CPU_LIMIT_SECONDS = 60 * 60
CPU_WARN_SECONDS = 50 * 60


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def import_v3_runner():
    spec = importlib.util.spec_from_file_location("exp050_source_five_arm_v3", V3_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the already-frozen v3 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_allowed_train_labels(v3: Any, train_objects: set[str]) -> pd.DataFrame:
    """Convert sensitivity to numeric only on CTRPv2 TRAIN rows for registered objects."""
    rows: list[dict[str, Any]] = []
    with (ROOT / v3.PAIRED_REL).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"endpoint", "split_role", "source_object_id", "full_inchikey", "sensitivity"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("paired source table is missing frozen columns")
        for raw in reader:
            if raw["endpoint"] != "CTRPv2" or raw["split_role"] != "train":
                continue
            obj = str(raw["source_object_id"])
            if obj not in train_objects:
                raise ValueError(f"unexpected CTRPv2 TRAIN object: {obj}")
            key = str(raw["full_inchikey"]).strip().upper()
            value = float(raw["sensitivity"])
            if not np.isfinite(value):
                raise ValueError(f"non-finite allowed TRAIN AAC value: {obj}/{key}")
            rows.append({"source_object_id": obj, "full_inchikey": key, "sensitivity": value})
    frame = pd.DataFrame(rows)
    if frame.empty or frame.duplicated(["source_object_id", "full_inchikey"]).any():
        raise ValueError("allowed TRAIN object/key labels are empty or duplicated")
    if set(frame.source_object_id.astype(str)) != train_objects:
        raise ValueError("allowed TRAIN labels do not cover the frozen nine objects")
    if not frame.full_inchikey.astype(str).str.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]").all():
        raise ValueError("an allowed label has an invalid complete InChIKey")
    return frame.sort_values(["source_object_id", "full_inchikey"]).reset_index(drop=True)


def materialize_outer_train(v3: Any, labels: pd.DataFrame, features: pd.DataFrame,
                            support: dict[str, Any], test_obj: str) -> tuple[pd.DataFrame, dict[str, str], str]:
    """Mirror v3's outer-training row construction without retrieving test AAC values."""
    detail = next(row for row in support["fold_support"] if row["test_object"] == test_obj)
    mapping: dict[str, str] = {str(k): str(v) for k, v in detail["outer_train_block_donor_map"].items()}
    train_objects = [str(x) for x in detail["outer_train_objects"]]
    donor_test = str(detail["test_block_donor"])
    candidate = set(map(str, support["candidate_domain"]))
    feature_lookup = features.set_index(["source_object_id", "full_inchikey"])
    labels_by_object = {obj: set(labels.loc[labels.source_object_id.eq(obj), "full_inchikey"].astype(str)) for obj in train_objects}
    rows: list[dict[str, Any]] = []
    for obj in train_objects:
        donor = mapping[obj]
        if donor == obj:
            raise ValueError("frozen whole-block shuffle mapping is not a derangement")
        if obj not in feature_lookup.index.get_level_values(0) or donor not in feature_lookup.index.get_level_values(0):
            raise ValueError("outer-training target or block donor lacks predicted perturbation features")
        keys = sorted(labels_by_object[obj]
                      & set(feature_lookup.loc[obj].index.astype(str))
                      & set(feature_lookup.loc[donor].index.astype(str))
                      & candidate)
        for key in keys:
            y = float(labels.loc[labels.source_object_id.eq(obj) & labels.full_inchikey.eq(key), "sensitivity"].iloc[0])
            rows.append({"source_object_id": obj, "full_inchikey": key, "sensitivity": y,
                         "shuffle_donor": donor,
                         "predicted_perturbation_strength": float(feature_lookup.loc[(obj, key), "predicted_perturbation_strength"])})
    train = pd.DataFrame(rows)
    # Match v3's common residual denominator: each key must have at least two
    # distinct outer-training label objects after target/donor support is applied.
    key_support = train.groupby("full_inchikey").source_object_id.nunique()
    eligible_train_keys = set(key_support.loc[key_support.ge(2)].index.astype(str))
    train = train.loc[train.full_inchikey.astype(str).isin(eligible_train_keys)].copy()
    if train.empty or train.source_object_id.nunique() < 5:
        raise ValueError(f"insufficient outer training support for {test_obj}")
    frozen_train_objects = set(train_objects)
    if set(train.source_object_id.astype(str)) - frozen_train_objects:
        raise ValueError("outer training frame includes a non-training object")
    return train.reset_index(drop=True), mapping, donor_test


def fit_predict(v3: Any, train: pd.DataFrame, valid: pd.DataFrame,
                features: pd.DataFrame, profiles: dict[str, np.ndarray],
                drugs: dict[str, np.ndarray], arm: str, shuffle: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    means = v3.train_mean(train)
    baseline = train.full_inchikey.map(means).to_numpy(dtype=float)
    y_resid = train.sensitivity.to_numpy(dtype=float) - baseline
    x_train, x_valid = v3.transformed_pair(train, valid, features, profiles, drugs, arm, shuffle=shuffle)
    model = Ridge(alpha=v3.RIDGE_ALPHA)
    model.fit(x_train, y_resid, sample_weight=v3.row_weights(train))
    prediction = valid.full_inchikey.map(means).to_numpy(dtype=float) + model.predict(x_valid)
    if not np.isfinite(prediction).all():
        raise ValueError(f"non-finite diagnostic prediction from {arm}")
    info = {"arm": arm, "n_train_pairs": int(len(train)), "n_train_objects": int(train.source_object_id.nunique()),
            "n_features": int(x_train.shape[1]), "ridge_alpha": float(v3.RIDGE_ALPHA),
            "fit_objects": sorted(train.source_object_id.astype(str).unique()),
            "fit_only_train_objects": True, "preprocessing_fit_only_on_train": True}
    return prediction.astype(float), info


def per_object_metrics(frame: pd.DataFrame, prediction: np.ndarray, means: dict[str, float], arm: str,
                       context: str, fold: str) -> list[dict[str, Any]]:
    work = frame[["source_object_id", "full_inchikey", "sensitivity"]].copy()
    work["prediction"] = np.asarray(prediction, dtype=float)
    work["drug_mean"] = work.full_inchikey.map(means).to_numpy(dtype=float)
    work["residual_observed"] = work.sensitivity - work.drug_mean
    work["residual_prediction"] = work.prediction - work.drug_mean
    rows: list[dict[str, Any]] = []
    for obj, group in work.groupby("source_object_id", sort=True):
        error = group.prediction.to_numpy(dtype=float) - group.sensitivity.to_numpy(dtype=float)
        observed_residual = group.residual_observed.to_numpy(dtype=float)
        predicted_residual = group.residual_prediction.to_numpy(dtype=float)
        base_mse = float(np.mean(np.square(observed_residual)))
        model_resid_mse = float(np.mean(np.square(observed_residual - predicted_residual)))
        rows.append({
            "fold_object": fold, "context": context, "arm": arm,
            "source_object_id": str(obj), "n_pairs": int(len(group)),
            "aac_mae": float(np.mean(np.abs(error))), "aac_rmse": float(np.sqrt(np.mean(np.square(error)))),
            "aac_mse": float(np.mean(np.square(error))), "mean_error_aac": float(np.mean(error)),
            "prediction_variance_aac": float(np.var(group.prediction.to_numpy(dtype=float), ddof=0)),
            "residual_mse_vs_train_drug_mean": model_resid_mse,
            "zero_residual_reference_mse": base_mse,
            "residual_explained_fraction": (1.0 - model_resid_mse / base_mse) if base_mse > 0 else None,
            "observed_residual_variance": float(np.var(observed_residual, ddof=0)),
            "predicted_residual_variance": float(np.var(predicted_residual, ddof=0)),
            "residual_prediction_variance_ratio": (float(np.var(predicted_residual, ddof=0) / np.var(observed_residual, ddof=0))
                                                    if np.var(observed_residual, ddof=0) > 0 else None),
        })
    return rows


def object_macro(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None and np.isfinite(float(row[field]))]
    return float(np.mean(values)) if values else None


def build_inner_oof(v3: Any, outer_train: pd.DataFrame, features: pd.DataFrame,
                    profiles: dict[str, np.ndarray], drugs: dict[str, np.ndarray],
                    fold: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    predictions: list[pd.DataFrame] = []
    objects = sorted(outer_train.source_object_id.astype(str).unique())
    for inner_test_obj in objects:
        inner_fit = outer_train.loc[outer_train.source_object_id.astype(str).ne(inner_test_obj)].copy()
        eligible = set(inner_fit.groupby("full_inchikey").source_object_id.nunique().loc[lambda s: s.ge(2)].index.astype(str))
        inner_test = outer_train.loc[(outer_train.source_object_id.astype(str).eq(inner_test_obj))
                                     & outer_train.full_inchikey.astype(str).isin(eligible)].copy()
        if inner_fit.empty or inner_test.empty:
            continue
        means = v3.train_mean(inner_fit)
        part = inner_test[["source_object_id", "full_inchikey", "sensitivity"]].copy()
        part["train_drug_mean"] = part.full_inchikey.map(means).to_numpy(dtype=float)
        part["observed_train_residual"] = part.sensitivity - part.train_drug_mean
        for arm, col in (("direct", "pred_direct"), ("predicted_perturbation", "pred_perturbation"),
                         ("equal_capacity_direct", "pred_direct_dup")):
            pred, _ = fit_predict(v3, inner_fit, inner_test, features, profiles, drugs,
                                  "equal_capacity_direct" if arm == "equal_capacity_direct" else arm)
            part[col] = pred
        part["fold_object"] = fold
        predictions.append(part)
    if not predictions:
        raise ValueError(f"no inner object predictions for outer fold {fold}")
    oof = pd.concat(predictions, ignore_index=True)
    info = {"n_inner_objects": int(oof.source_object_id.nunique()), "n_inner_pairs": int(len(oof)),
            "inner_objects": sorted(oof.source_object_id.astype(str).unique()),
            "inner_split": "leave-one-outer-training-object-out", "outer_test_values_accessed": False,
            "training_only_inner_labels": True}
    return oof, info


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_wall = time.monotonic()
    start_cpu = time.process_time()
    start_utc = utc_now()
    v3 = import_v3_runner()
    manifest_path = V3_DIR / "run_manifest.json"
    support_path = V3_DIR / "support_freeze.json"
    logs_path = V3_DIR / "full/fold_model_logs.json"
    completion_path = V3_DIR / "full/completion_receipt.json"
    manager_path = V3_DIR / "full/manager_review_receipt.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    support = json.loads(support_path.read_text(encoding="utf-8"))
    fold_logs = json.loads(logs_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    manager = json.loads(manager_path.read_text(encoding="utf-8"))
    if manifest.get("run_id") != V3_ID or manifest.get("runner_sha256") != sha256(V3_RUNNER):
        raise ValueError("v3 run identity or runner hash changed from the frozen diagnostic base")
    if manifest.get("outputs_sha256", {}).get("support_freeze.json") != sha256(support_path):
        raise ValueError("frozen support file hash differs from the v3 run manifest")
    if manifest.get("outputs_sha256", {}).get("full/fold_model_logs.json") != sha256(logs_path):
        raise ValueError("v3 fold model log hash differs from the run manifest")
    if completion.get("status") != "PASS_NINE_OBJECT_FIVE_ARM_OOF" or manager.get("status") != "MANAGER_VERIFIED_LIMITED_EXPLORATORY_RESULT":
        raise ValueError("the v3 base run is not in the expected completed/reviewed state")
    if len(fold_logs) != 9 or len(support.get("fold_support", [])) != 9:
        raise ValueError("v3 base run does not contain nine frozen outer folds")
    train_objects = set(map(str, support["label_support"]["pairs_by_object"].keys()))
    labels = read_allowed_train_labels(v3, train_objects)
    if len(labels) != int(support["label_support"]["object_drug_pairs"]):
        raise ValueError("allowed TRAIN label count differs from the frozen support")
    target_keys = pd.read_csv(ROOT / v3.PRED_KEYS_REL, sep="\t",
                              usecols=["condition_id", "source_row", "source_object_id", "full_inchikey"],
                              dtype={"source_object_id": "string", "full_inchikey": "string"})
    target_keys["prediction_row"] = np.arange(len(target_keys), dtype=np.int64)
    features, exposure_audit = v3.verified_xpert_exposure(train_objects, target_keys)
    genes = [str(x).strip() for x in json.loads((ROOT / v3.GENES_REL).read_text(encoding="utf-8"))]
    profiles, expression_audit = v3.load_expression(train_objects, genes)
    drug_features, drug_audit = v3.load_drug_features(set(labels.full_inchikey.astype(str)))
    if exposure_audit["identity_overlap_status"] != "PASS_ZERO_OVERLAP":
        raise ValueError("frozen XPert training identities overlap the permitted source objects")
    if set(labels.full_inchikey.astype(str)) & set(drug_audit["excluded_nonunique_exact_morgan_keys"]) != set(v3.EXPECTED_NONUNIQUE_MORGAN_KEYS):
        raise ValueError("frozen ambiguous Morgan key exclusion differs from v3")

    outer_fit_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    inner_predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    input_timing = {"loaded_utc": utc_now(), "allowed_train_objects": sorted(train_objects),
                    "allowed_train_label_pairs": int(len(labels)), "features_object_key_pairs": int(len(features)),
                    "val_outer_locked_pdo_label_values_loaded": False}

    for fold_detail in support["fold_support"]:
        if time.process_time() - start_cpu > CPU_WARN_SECONDS:
            raise TimeoutError("diagnostic reached its 50-minute process-CPU warning stop")
        fold = str(fold_detail["test_object"])
        frozen_log = next(row for row in fold_logs if str(row["test_object"]) == fold)
        train, mapping, donor_test = materialize_outer_train(v3, labels, features, support, fold)
        frozen_keys = sorted(map(str, fold_detail["test_common_full_inchikeys"]))
        if len(frozen_keys) != int(fold_detail["test_common_all_arm_pairs"]):
            raise ValueError(f"frozen test key list/count mismatch for {fold}")
        if sorted(map(str, frozen_log["test_drug_key_list"])) != frozen_keys:
            raise ValueError(f"v3 logged test key identities changed for {fold}")
        test_feature_keys = set(features.loc[features.source_object_id.astype(str).eq(fold), "full_inchikey"].astype(str))
        donor_feature_keys = set(features.loc[features.source_object_id.astype(str).eq(donor_test), "full_inchikey"].astype(str))
        covered_target = sorted(set(frozen_keys) & test_feature_keys & set(drug_features))
        covered_donor = sorted(set(frozen_keys) & test_feature_keys & donor_feature_keys & set(drug_features))
        if covered_donor != frozen_keys:
            raise ValueError(f"heldout whole-block donor support no longer covers frozen denominator for {fold}")
        if len(train) != int(frozen_log["outer_train_rows"]):
            raise ValueError(f"reconstructed outer training rows differ from fixed v3 model log for {fold}")
        train_keys = set(train.full_inchikey.astype(str))
        min_two_train_object_keys = set(
            train.groupby("full_inchikey").source_object_id.nunique().loc[lambda s: s.ge(2)].index.astype(str)
        )
        block_target_coverage = sum(
            len(set(labels.loc[labels.source_object_id.eq(obj), "full_inchikey"].astype(str))
                & set(features.loc[features.source_object_id.astype(str).eq(obj), "full_inchikey"].astype(str))
                & set(features.loc[features.source_object_id.astype(str).eq(mapping[obj]), "full_inchikey"].astype(str))
                & set(support["candidate_domain"])
                & min_two_train_object_keys)
            for obj in mapping
        )
        if int(block_target_coverage) != len(train):
            raise ValueError(f"train target/donor block coverage differs from common rows for {fold}")
        coverage_rows.append({
            "outer_test_object_identity_only": fold, "outer_train_objects": len(mapping),
            "outer_train_common_rows_all_arms": int(len(train)), "outer_train_unique_exact_keys": int(train.full_inchikey.nunique()),
            "outer_test_frozen_common_key_count": int(len(frozen_keys)),
            "outer_test_target_delta_coverage": int(len(covered_target)),
            "outer_test_block_donor": donor_test, "outer_test_block_donor_delta_coverage": int(len(covered_donor)),
            "outer_test_shared_exact_key_coverage": int(len(set(covered_target) & set(covered_donor) & set(drug_features))),
            "outer_test_frozen_denominator_sha256": v3.canonical_sha256(frozen_keys),
            "train_block_shuffled_row_support": int(block_target_coverage),
            "all_model_arms_same_frozen_denominator": True,
            "outer_test_numeric_aac_read": False,
        })

        # Inner object cross-predictions are built exclusively from this fold's outer-training rows.
        inner_oof, inner_info = build_inner_oof(v3, train, features, profiles, drug_features, fold)
        fusion_weight, fusion_info = v3.choose_mix(inner_oof.rename(columns={"pred_direct": "pred_direct",
                                                                             "pred_perturbation": "pred_perturbation"}),
                                                   "pred_direct", "pred_perturbation")
        equal_weight, equal_info = v3.choose_mix(inner_oof, "pred_direct", "pred_direct_dup")
        # Reproduce v3's fold choices exactly; weights are learned only from this fold's inner OOF.
        if abs(fusion_weight - float(frozen_log["fusion_weight_direct"])) > 1e-12:
            raise ValueError(f"recomputed inner-OOF fusion weight differs from v3 for {fold}")
        if abs(equal_weight - float(frozen_log["equal_capacity_direct_weight"])) > 1e-12:
            raise ValueError(f"recomputed duplicate-direct weight differs from v3 for {fold}")
        inner_oof["pred_fusion_selected_on_same_inner_oof"] = (
            fusion_weight * inner_oof.pred_direct + (1.0 - fusion_weight) * inner_oof.pred_perturbation)
        inner_oof["pred_equal_capacity_selected_on_same_inner_oof"] = (
            equal_weight * inner_oof.pred_direct + (1.0 - equal_weight) * inner_oof.pred_direct_dup)
        inner_oof["outer_fold"] = fold
        inner_predictions.append(inner_oof)
        # Inner fold drug-mean baselines are saved per row during construction.
        inner_means_by_object: dict[str, dict[str, float]] = {}
        for inner_obj in sorted(train.source_object_id.astype(str).unique()):
            inner_fit = train.loc[train.source_object_id.astype(str).ne(inner_obj)]
            inner_means_by_object[inner_obj] = v3.train_mean(inner_fit)
        for arm, col in (("direct", "pred_direct"), ("predicted_perturbation", "pred_perturbation"),
                         ("equal_capacity_direct", "pred_direct_dup"),
                         ("fusion_selected_on_same_inner_oof", "pred_fusion_selected_on_same_inner_oof")):
            for obj, group in inner_oof.groupby("source_object_id", sort=True):
                means = inner_means_by_object[str(obj)]
                rows = per_object_metrics(group, group[col].to_numpy(dtype=float), means, arm, "inner_oof", fold)
                inner_rows.extend(rows)
        # Outer training in-sample fit: no outer-test numerical labels or predictions are accessed.
        means = v3.train_mean(train)
        direct, direct_info = fit_predict(v3, train, train, features, profiles, drug_features, "direct")
        perturb, perturb_info = fit_predict(v3, train, train, features, profiles, drug_features, "predicted_perturbation")
        duplicate, duplicate_info = fit_predict(v3, train, train, features, profiles, drug_features, "equal_capacity_direct")
        shuffled, shuffled_info = fit_predict(v3, train, train, features, profiles, drug_features,
                                             "shuffled_perturbation", shuffle=True)
        if not np.allclose(direct, duplicate, rtol=1e-7, atol=1e-8):
            raise ValueError(f"fixed duplicate direct fit differs from direct fit for {fold}")
        outer_predictions = {
            "direct": direct,
            "predicted_perturbation": perturb,
            "fusion": fusion_weight * direct + (1.0 - fusion_weight) * perturb,
            "equal_capacity_direct": equal_weight * direct + (1.0 - equal_weight) * duplicate,
            "shuffled_perturbation": shuffled,
        }
        fit_arm_info = {"direct": direct_info, "predicted_perturbation": perturb_info,
                        "equal_capacity_direct": duplicate_info, "shuffled_perturbation": shuffled_info}
        fold_arm_rows: dict[str, list[dict[str, Any]]] = {}
        for arm, pred in outer_predictions.items():
            rows = per_object_metrics(train, pred, means, arm, "outer_train_in_sample", fold)
            outer_fit_rows.extend(rows)
            fold_arm_rows[arm] = rows
        fold_summary = {"outer_fold": fold, "outer_train_rows": int(len(train)),
                        "outer_train_objects": int(train.source_object_id.nunique()),
                        "outer_test_key_count_identity_only": int(len(frozen_keys)),
                        "fusion_weight_direct": float(fusion_weight), "fusion_inner_object_macro_mse": float(fusion_info["inner_oof_object_macro_mse"]),
                        "fusion_inner_oof_objects": int(fusion_info["n_inner_oof_objects"]),
                        "equal_capacity_direct_weight": float(equal_weight),
                        "equal_capacity_inner_object_macro_mse": float(equal_info["inner_oof_object_macro_mse"]),
                        "inner_oof_pairs": int(inner_info["n_inner_pairs"]), "inner_oof_objects": int(inner_info["n_inner_objects"])}
        for arm, rows in fold_arm_rows.items():
            fold_summary[f"{arm}_train_object_macro_aac_rmse"] = object_macro(rows, "aac_rmse")
            fold_summary[f"{arm}_train_object_macro_residual_mse"] = object_macro(rows, "residual_mse_vs_train_drug_mean")
            fold_summary[f"{arm}_train_object_macro_zero_residual_mse"] = object_macro(rows, "zero_residual_reference_mse")
            fractions = [r["residual_explained_fraction"] for r in rows if r["residual_explained_fraction"] is not None]
            fold_summary[f"{arm}_train_object_macro_residual_explained_fraction"] = float(np.mean(fractions)) if fractions else None
        for arm in INNER_ARMS + ("fusion_selected_on_same_inner_oof",):
            selected_col = {"direct": "pred_direct", "predicted_perturbation": "pred_perturbation",
                            "equal_capacity_direct": "pred_direct_dup",
                            "fusion_selected_on_same_inner_oof": "pred_fusion_selected_on_same_inner_oof"}[arm]
            relevant = [r for r in inner_rows if r["fold_object"] == fold and r["arm"] == arm]
            fold_summary[f"{arm}_inner_object_macro_aac_rmse"] = object_macro(relevant, "aac_rmse")
            fold_summary[f"{arm}_inner_object_macro_residual_mse"] = object_macro(relevant, "residual_mse_vs_train_drug_mean")
            fold_summary[f"{arm}_inner_object_macro_residual_explained_fraction"] = object_macro(relevant, "residual_explained_fraction")
            fold_summary[f"{arm}_inner_object_macro_prediction_variance_aac"] = object_macro(relevant, "prediction_variance_aac")
        fold_rows.append(fold_summary)
        weight_rows.append({"outer_fold": fold, "fusion_weight_on_direct": float(fusion_weight),
                            "fusion_weight_on_predicted_perturbation": float(1.0 - fusion_weight),
                            "fusion_inner_object_macro_mse": float(fusion_info["inner_oof_object_macro_mse"]),
                            "fusion_inner_oof_objects": int(fusion_info["n_inner_oof_objects"]),
                            "fusion_weight_recomputed_matches_v3": True,
                            "equal_capacity_direct_weight_on_copy_1": float(equal_weight),
                            "equal_capacity_duplicate_model_is_identical": True,
                            "weight_source": "outer-training objects' inner leave-one-object-out predictions only"})

    if time.process_time() - start_cpu > CPU_LIMIT_SECONDS:
        raise TimeoutError("diagnostic exceeded its 60-minute process-CPU hard stop")
    outer_df = pd.DataFrame(outer_fit_rows)
    inner_df = pd.DataFrame(inner_rows)
    inner_pred_df = pd.concat(inner_predictions, ignore_index=True)
    fold_df = pd.DataFrame(fold_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    weights_df = pd.DataFrame(weight_rows)
    # Add a uniform residual reference and predictions to the inner row-level table.
    mean_lookup = {(str(row.outer_fold), str(row.source_object_id), str(row.full_inchikey)): float(row.train_drug_mean)
                   for row in inner_pred_df.itertuples(index=False)}
    if len(mean_lookup) != len(inner_pred_df):
        raise ValueError("inner OOF row identity is duplicated")
    # Round-trip all saved row keys and numeric fields from allowed TRAIN objects only.
    outer_df.to_csv(OUT_DIR / "outer_train_fit_by_object.tsv", sep="\t", index=False, na_rep="")
    inner_df.to_csv(OUT_DIR / "inner_oof_by_object_metrics.tsv", sep="\t", index=False, na_rep="")
    inner_pred_df.to_csv(OUT_DIR / "inner_oof_predictions_training_only.tsv", sep="\t", index=False, na_rep="")
    fold_df.to_csv(OUT_DIR / "per_outer_fold_diagnostics.tsv", sep="\t", index=False, na_rep="")
    weights_df.to_csv(OUT_DIR / "fusion_weights.tsv", sep="\t", index=False, na_rep="")
    coverage_df.to_csv(OUT_DIR / "common_key_and_block_shuffle_coverage.tsv", sep="\t", index=False, na_rep="")

    summary: dict[str, Any] = {
        "run_id": RUN_ID, "status": "COMPLETED_TRAINING_ONLY_FIXED_MODEL_DIAGNOSTIC",
        "base_run_id": V3_ID, "base_run_status": manager["status"], "n_train_objects": len(train_objects),
        "outer_folds": int(len(fold_df)), "outer_test_numeric_labels_read": False,
        "forbidden_numeric_values_loaded": {"VAL": False, "outer": False, "locked": False, "PDO": False, "PRISM": False},
        "label_exposure_boundary": "These nine CTRPv2 TRAIN labels were already exposed in EXP-050 section 3.4; this is a mechanism-oriented development diagnostic, not independent evidence.",
        "protocol": {"ridge_alpha": 1.0, "feature_scaling": "same v3 fold-local StandardScaler", "target": "AAC residual from current fit's exact-key training mean", "outer_train_fit": "in-sample diagnostic only", "inner_oof": "leave one outer-training source object out", "fusion_weight": "recomputed on current outer-training inner OOF only; same inner OOF is used for the diagnostic blend score", "no_candidates_or_search": True},
        "input_lineage": {"v3_run_manifest_sha256": sha256(manifest_path), "v3_manager_review_sha256": sha256(manager_path),
                          "v3_support_freeze_sha256": sha256(support_path), "v3_fold_logs_sha256": sha256(logs_path),
                          "v3_completion_receipt_sha256": sha256(completion_path),
                          "inherited_v3_input_sha256": manifest.get("inputs_sha256", {}),
                          "v3_full_oof_values_used": False, "v3_full_oof_identity_projection_used": False},
        "runtime_input_audit": {**input_timing,
                                 "xpert_fit_training_identity_overlap": exposure_audit["fit_train_ctrpv2_train_identity_overlap"],
                                 "selected_expression_objects": expression_audit["selected_patient_objects"],
                                 "selected_expression_gene_count": expression_audit["gene_count"],
                                 "Morgan_key_count": len(drug_features), "excluded_ambiguous_keys": sorted(drug_audit["excluded_nonunique_exact_morgan_keys"])},
        "outer_train_fit_object_macro_means": {}, "inner_oof_object_macro_means_by_outer_fold": {},
        "fusion_weight_summary": {"mean_weight_on_direct": float(weights_df.fusion_weight_on_direct.mean()),
                                  "median_weight_on_direct": float(weights_df.fusion_weight_on_direct.median()),
                                  "range_weight_on_direct": [float(weights_df.fusion_weight_on_direct.min()), float(weights_df.fusion_weight_on_direct.max())],
                                  "folds_weight_on_direct_at_least_0_8": int((weights_df.fusion_weight_on_direct >= 0.8).sum()),
                                  "n_outer_folds": int(len(weights_df))},
        "shared_denominator": {"n_outer_test_object_drug_pairs": int(coverage_df.outer_test_frozen_common_key_count.sum()),
                               "all_folds_match_v3_frozen_test_key_identity": bool((coverage_df.outer_test_shared_exact_key_coverage == coverage_df.outer_test_frozen_common_key_count).all()),
                               "all_five_model_arms_use_identical_denominator_per_fold": True,
                               "total_outer_train_common_rows_all_folds": int(coverage_df.outer_train_common_rows_all_arms.sum())},
        "interpretation": {},
        "next_action": "Coordinator uses the training-fit and inner-object OOF evidence to record a scoped conclusion and resume the frozen independent-source intake/transfer deliverable; no new model candidate or parameter search.",
        "cost": {"cpu_cap_minutes": 60.0, "cpu_warning_minutes": 50.0,
                 "process_cpu_minutes": float((time.process_time() - start_cpu) / 60.0),
                 "wall_minutes": float((time.monotonic() - start_wall) / 60.0), "gpu_hours": 0.0,
                 "process_id": os.getpid(),
                 "environment": "WSL2 Conda drugscreening-gpu", "python": sys.version,
                 "python_executable": sys.executable, "platform": platform.platform(),
                 "started_utc": start_utc, "finished_utc": utc_now()},
    }
    for arm in OUTER_ARMS:
        rows = outer_df.loc[outer_df.arm.eq(arm)]
        summary["outer_train_fit_object_macro_means"][arm] = {
            "aac_mae": float(rows.aac_mae.mean()), "aac_rmse": float(rows.aac_rmse.mean()),
            "residual_mse_vs_train_drug_mean": float(rows.residual_mse_vs_train_drug_mean.mean()),
            "zero_residual_reference_mse": float(rows.zero_residual_reference_mse.mean()),
            "residual_explained_fraction": float(rows.residual_explained_fraction.mean()),
            "prediction_variance_aac": float(rows.prediction_variance_aac.mean()),
            "n_outer_train_objects_across_folds_not_independent": int(rows.source_object_id.nunique()),
        }
    for arm in INNER_ARMS + ("fusion_selected_on_same_inner_oof",):
        rows = inner_df.loc[inner_df.arm.eq(arm)]
        summary["inner_oof_object_macro_means_by_outer_fold"][arm] = {
            "aac_rmse": float(rows.groupby("fold_object").aac_rmse.mean().mean()),
            "aac_mae": float(rows.groupby("fold_object").aac_mae.mean().mean()),
            "residual_mse_vs_inner_train_drug_mean": float(rows.groupby("fold_object").residual_mse_vs_train_drug_mean.mean().mean()),
            "zero_residual_reference_mse": float(rows.groupby("fold_object").zero_residual_reference_mse.mean().mean()),
            "residual_explained_fraction": float(rows.groupby("fold_object").residual_explained_fraction.mean().mean()),
            "prediction_variance_aac": float(rows.groupby("fold_object").prediction_variance_aac.mean().mean()),
            "mean_inner_objects_per_outer_fold": float(rows.groupby("fold_object").source_object_id.nunique().mean()),
            "n_outer_folds": int(rows.fold_object.nunique()),
        }
    direct_inner = summary["inner_oof_object_macro_means_by_outer_fold"]["direct"]
    pert_inner = summary["inner_oof_object_macro_means_by_outer_fold"]["predicted_perturbation"]
    direct_fit = summary["outer_train_fit_object_macro_means"]["direct"]
    pert_fit = summary["outer_train_fit_object_macro_means"]["predicted_perturbation"]
    summary["interpretation"] = {
        "did_direct_fit_reduce_training_residual_error_vs_zero_residual_reference": direct_fit["residual_explained_fraction"] > 0,
        "did_perturbation_fit_reduce_training_residual_error_vs_zero_residual_reference": pert_fit["residual_explained_fraction"] > 0,
        "does_inner_oof_show_direct_residual_error_reduction": direct_inner["residual_explained_fraction"] > 0,
        "does_inner_oof_show_perturbation_residual_error_reduction": pert_inner["residual_explained_fraction"] > 0,
        "inner_direct_minus_perturbation_aac_rmse": direct_inner["aac_rmse"] - pert_inner["aac_rmse"],
        "inner_direct_minus_perturbation_residual_mse": direct_inner["residual_mse_vs_inner_train_drug_mean"] - pert_inner["residual_mse_vs_inner_train_drug_mean"],
        "fusion_weight_direction": "a direct weight near 1 means selected blend favors direct prediction; this is selection within already exposed source-domain development data",
        "scientific_limit": "Results characterize this fixed realization and exposed nine-object dataset only; they do not establish that object-specific perturbation information generally exists or does not exist.",
    }
    summary_path = OUT_DIR / "diagnostic_summary.json"
    write_json(summary_path, summary)

    # Numeric/key round-trip is confined to training-only diagnostic tables.
    outer_readback = pd.read_csv(OUT_DIR / "outer_train_fit_by_object.tsv", sep="\t")
    inner_readback = pd.read_csv(OUT_DIR / "inner_oof_predictions_training_only.tsv", sep="\t")
    if len(outer_readback) != len(outer_df) or len(inner_readback) != len(inner_pred_df):
        raise ValueError("diagnostic TSV readback row count mismatch")
    if not set(outer_readback.source_object_id.astype(str)).issubset(train_objects) or not set(inner_readback.source_object_id.astype(str)).issubset(train_objects):
        raise ValueError("diagnostic readback includes a non-TRAIN object")
    outputs = sorted(path for path in OUT_DIR.iterdir() if path.is_file() and path.name not in {"run_manifest.json", "completion_receipt.json"})
    output_hashes = {str(path.relative_to(OUT_DIR)).replace("\\", "/"): sha256(path) for path in outputs}
    run_manifest = {"run_id": RUN_ID, "base_run_id": V3_ID, "status": summary["status"],
                    "diagnostic_script_sha256": sha256(DIAGNOSTIC_SCRIPT),
                    "v3_runner_sha256": sha256(V3_RUNNER), "v3_config_sha256": manifest["config_sha256"],
                    "v3_inputs_sha256_inherited_from_manager_reviewed_run_manifest": manifest["inputs_sha256"],
                    "v3_output_hashes_used": {"support_freeze.json": sha256(support_path),
                                              "full/fold_model_logs.json": sha256(logs_path),
                                              "full/completion_receipt.json": sha256(completion_path),
                                              "full/manager_review_receipt.json": sha256(manager_path)},
                    "numeric_data_role": "CTRPv2 TRAIN labels for nine already-exposed source objects only",
                    "forbidden_numeric_values_loaded": {"VAL": False, "outer": False, "locked": False, "PDO": False, "PRISM": False},
                    "cost": summary["cost"], "outputs_sha256": output_hashes}
    write_json(OUT_DIR / "run_manifest.json", run_manifest)
    receipt = {"status": "PASS_TRAIN_ONLY_FIXED_MODEL_DIAGNOSTICS", "run_id": RUN_ID,
               "n_outer_folds": int(len(fold_df)), "n_outer_train_fit_object_arm_rows": int(len(outer_df)),
               "n_inner_oof_object_arm_rows": int(len(inner_df)), "inner_oof_training_only_rows": int(len(inner_pred_df)),
               "all_frozen_fold_weights_recomputed_exactly": True,
               "all_common_key_and_block_donor_coverage_checks_passed": True,
               "outer_test_numeric_labels_read": False, "forbidden_numeric_values_loaded": False,
               "gpu_hours": 0.0, "cpu_minutes": summary["cost"]["process_cpu_minutes"],
               "summary_sha256": sha256(summary_path), "run_manifest_sha256": sha256(OUT_DIR / "run_manifest.json")}
    write_json(OUT_DIR / "completion_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
