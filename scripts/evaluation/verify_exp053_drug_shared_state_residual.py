"""Independent readback for EXP-053 TRAIN-only object OOF outputs."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
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

METHODS = ("zero_change", "per_drug_mean", "state_residual", "shuffled_state_control")


def sha256(path: Path) -> str:
    return fit.sha256_file(path)


def derangement(ids: list[str], rng: np.random.Generator) -> dict[str, str]:
    unique = sorted(set(ids))
    if len(unique) < 2:
        raise ValueError("cannot reconstruct an object-block derangement with fewer than two objects")
    perm = np.asarray(unique, dtype=str)[rng.permutation(len(unique))]
    return {str(perm[i]): str(perm[(i + 1) % len(perm)]) for i in range(len(perm))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", help="legacy same-directory completed run id")
    parser.add_argument("--source-run-id", help="immutable run containing saved OOF predictions")
    parser.add_argument("--recovery-run-id", help="derived recovery run containing recovered result and manifest")
    args = parser.parse_args()
    started = time.monotonic()
    if args.source_run_id or args.recovery_run_id:
        if not args.source_run_id or not args.recovery_run_id or args.run_id:
            raise ValueError("provide both --source-run-id and --recovery-run-id, without --run-id")
        source = ROOT / "artifacts/experiments/EXP-053" / args.source_run_id
        out = ROOT / "artifacts/experiments/EXP-053" / args.recovery_run_id
        run_id = args.recovery_run_id
        source_run_id = args.source_run_id
    else:
        if not args.run_id:
            raise ValueError("provide --run-id or a source/recovery run pair")
        source = out = ROOT / "artifacts/experiments/EXP-053" / args.run_id
        run_id = source_run_id = args.run_id
    result_path = out / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError("EXP-053 result.json is absent from the result/recovery directory")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    cfg = yaml.safe_load((ROOT / "configs/experiments/EXP-053-drug-shared-state-residual.yaml").read_text(
        encoding="utf-8"))
    recovered = result.get("status") == "RECOVERED_FROM_SAVED_TRAIN_OOF_AFTER_RESULT_SERIALIZATION_FAILURE"
    if result.get("status") not in ("COMPLETED_TRAIN_ONLY_OBJECT_OOF_DEVELOPMENT",
                                     "RECOVERED_FROM_SAVED_TRAIN_OOF_AFTER_RESULT_SERIALIZATION_FAILURE"):
        raise ValueError("runner did not complete the registered OOF comparison")
    if result.get("source_run_id", source_run_id) != source_run_id:
        raise ValueError("recovered result refers to a different immutable source run")
    pred_path = source / "train_object_oof_predictions.npz"
    with np.load(pred_path, allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    fit.configure_source_run("20261006_exp051_contract_fit_smoke_v1")
    src = fit.SOURCE_RUN_DIR
    keys_all = pd.read_csv(src / "response_blind_conditions.tsv", sep="\t", dtype=str,
                           keep_default_na=False)
    keys = keys_all.loc[keys_all.split_role.eq("train")].reset_index(drop=True)
    if len(keys) != 419 or not keys.full_condition_key.is_unique:
        raise ValueError("frozen response-blind TRAIN key count changed")
    if not np.array_equal(arrays["condition_key"].astype(str), keys.full_condition_key.astype(str).to_numpy()):
        raise ValueError("OOF keys differ from the frozen TRAIN key order")
    if not np.array_equal(arrays["source_object_id"].astype(str), keys.source_object_id.astype(str).to_numpy()):
        raise ValueError("OOF source-object axis differs from frozen TRAIN rows")
    if not np.array_equal(arrays["full_inchikey"].astype(str), keys.full_inchikey.astype(str).to_numpy()):
        raise ValueError("OOF full InChIKey axis differs from frozen TRAIN rows")
    axis_map = pd.read_csv(src / "gene_axis_mapping.tsv", sep="\t", dtype=str, keep_default_na=False)
    expected_genes = axis_map.h5ad_var_name.astype(str).to_numpy(dtype=str)
    if not np.array_equal(arrays["gene_axis"].astype(str), expected_genes):
        raise ValueError("OOF gene axis differs from frozen ordered gene axis")
    if (arrays["truth_delta"].shape != (419, 978) or arrays["fold_id"].shape != (419,)
            or any(arrays[m].shape != (419, 978) for m in METHODS)):
        raise ValueError("OOF prediction, truth, fold, or gene dimensions differ")
    if any(not np.isfinite(arrays[m]).all() for m in ("truth_delta", *METHODS)):
        raise FloatingPointError("OOF numeric output has non-finite values")
    object_to_fold: dict[str, int] = {}
    for obj, fold in zip(arrays["source_object_id"].astype(str), arrays["fold_id"].astype(int)):
        if obj in object_to_fold and object_to_fold[obj] != fold:
            raise ValueError("a source object appears in multiple held-out folds")
        object_to_fold[obj] = fold
    if len(object_to_fold) != 110 or set(object_to_fold.values()) != set(range(5)):
        raise ValueError("OOF object/fold denominator differs from 110 objects/5 folds")
    objects = np.asarray(sorted(keys.source_object_id.astype(str).unique().tolist()), dtype=str)
    fold_rng = np.random.default_rng(int(cfg["cv"]["seed"]))
    ordered_objects = objects[fold_rng.permutation(len(objects))]
    expected_fold = {obj: fold for fold in range(5) for obj in ordered_objects[fold::5]}
    if object_to_fold != expected_fold:
        raise ValueError("OOF assignments differ from the frozen deterministic label-blind object split")
    fold_fit = pd.read_csv(source / "fold_fit_and_permutation_receipt.tsv", sep="\t", dtype=str)
    fold_metrics = pd.read_csv(source / "fold_method_metrics.tsv", sep="\t")
    if (len(fold_fit) != 5 or set(fold_fit.training_drug_keys.astype(int)) != {5}
            or len(fold_metrics) != 20):
        raise ValueError("fold receipt lacks five full-key fits and four methods per fold")
    full_drugs = set(keys.full_inchikey.astype(str))
    for row in fold_fit.itertuples(index=False):
        fold = int(row.fold)
        train_idx = np.flatnonzero(arrays["fold_id"].astype(int) != fold)
        test_objects = set(arrays["source_object_id"].astype(str)[arrays["fold_id"].astype(int) == fold])
        train_objects = set(arrays["source_object_id"].astype(str)[arrays["fold_id"].astype(int) != fold])
        if (set(keys.full_inchikey.astype(str).to_numpy()[train_idx]) != full_drugs
                or set(row.test_source_objects.split(";")) != test_objects
                or len(train_objects) != 88 or len(test_objects) + len(train_objects) != 110):
            raise ValueError(f"fold {fold} lacks full five-drug training support or the expected held-out objects")
        for encoded_model in (row.training_state_model, row.shuffled_state_model):
            model = ast.literal_eval(encoded_model)
            if (model.get("ridge_alpha") != float(cfg["models"]["state_residual"]["ridge_alpha"])
                    or model.get("fit_objects") != len(train_objects)
                    or model.get("fit_conditions") != len(train_idx)
                    or model.get("feature_genes") != 978
                    or model.get("scaler_fit_objects") != len(train_objects)
                    or model.get("training_object_total_weight_equalized") is not True
                    or model.get("state_source") != "object-mean paired X_ctl; fold-train standardization"):
                raise ValueError(f"fold {fold} receipt does not bind preprocessing/fitting to train objects only")
    source_values = source_reader.read_role_expression(keys, expected_genes)
    truth_delta = arrays["truth_delta"].astype(np.float64)
    truth_error = float(np.max(np.abs(truth_delta - source_values["delta"].astype(np.float64))))
    if truth_error > 1e-6:
        raise ValueError(f"stored TRAIN truth differs from the frozen H5AD TRAIN rows: {truth_error}")

    object_scores: dict[str, dict[str, float]] = {obj: {} for obj in object_to_fold}
    object_detail: dict[tuple[str, str], dict[str, float]] = {}
    metric_max_abs_error = 0.0
    for fold in range(5):
        test_idx = np.flatnonzero(arrays["fold_id"] == fold)
        train_idx = np.flatnonzero(arrays["fold_id"] != fold)
        train_y = truth_delta[train_idx]
        variance = np.var(train_y, axis=0, ddof=0)
        use = np.isfinite(variance) & (variance > 1e-12)
        if not use.any():
            raise ValueError("no fold-training response genes have nonzero variance")
        for method in METHODS:
            pred = arrays[method].astype(np.float64)[test_idx]
            truth = truth_delta[test_idx]
            err = pred - truth
            row_nmse = (np.square(err[:, use]) / variance[use][None, :]).mean(axis=1)
            row_mae = np.abs(err).mean(axis=1)
            row_corr = []
            for y, p in zip(truth, pred):
                row_corr.append(float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else np.nan)
            row_corr = np.asarray(row_corr, dtype=np.float64)
            row_dir = (np.sign(truth) == np.sign(pred)).mean(axis=1)
            ids = arrays["source_object_id"].astype(str)[test_idx]
            for obj in sorted(set(ids)):
                ix = np.flatnonzero(ids == obj)
                finite_corr = row_corr[ix][np.isfinite(row_corr[ix])]
                truth_amp = float(np.mean(np.abs(truth[ix])))
                pred_amp = float(np.mean(np.abs(pred[ix])))
                detail = {"nmse": float(row_nmse[ix].mean()), "mae": float(row_mae[ix].mean()),
                          "gene_correlation": float(finite_corr.mean()) if len(finite_corr) else float("nan"),
                          "direction_agreement": float(row_dir[ix].mean()),
                          "amplitude_ratio": pred_amp / truth_amp if truth_amp > 0 else float("nan"),
                          "conditions": int(len(ix))}
                object_detail[(obj, method)] = detail
                object_scores[obj][method] = detail["nmse"]
    nullable_metrics: dict[str, list[str]] = {}
    for method in METHODS:
        rows = [object_detail[(obj, method)] for obj in sorted(object_scores)]
        recalculated = {"object_macro_nmse": float(np.mean([x["nmse"] for x in rows])),
                        "object_macro_mae": float(np.mean([x["mae"] for x in rows])),
                        "object_macro_gene_correlation": (float(np.mean([x["gene_correlation"] for x in rows
                            if np.isfinite(x["gene_correlation"])])) if any(
                                np.isfinite(x["gene_correlation"]) for x in rows) else None),
                        "object_macro_direction_agreement": float(np.mean([x["direction_agreement"] for x in rows])),
                        "object_macro_amplitude_ratio": (float(np.mean([x["amplitude_ratio"] for x in rows
                            if np.isfinite(x["amplitude_ratio"])])) if any(
                                np.isfinite(x["amplitude_ratio"]) for x in rows) else None)}
        for metric, value in recalculated.items():
            recorded = result["methods"][method][metric]
            if recorded is None:
                if value is not None:
                    raise ValueError(f"{method}/{metric} is recorded undefined but recomputation is finite")
                nullable_metrics.setdefault(method, []).append(metric)
                continue
            if value is None:
                raise ValueError(f"{method}/{metric} is finite in result but undefined on readback")
            delta = abs(value - float(recorded))
            metric_max_abs_error = max(metric_max_abs_error, delta)
            if delta > 2e-5:
                raise ValueError(f"independent {method}/{metric} differs by {delta}")
    saved_object = pd.read_csv(source / "per_object_method_metrics.tsv", sep="\t",
                               dtype={"source_object_id": str})
    recovered_object = pd.DataFrame([
        {"source_object_id": obj, "method": method, **object_detail[(obj, method)]}
        for obj in sorted(object_scores) for method in METHODS
    ])
    if len(saved_object) != 440:
        raise ValueError("saved per-object table does not contain 110 objects x 4 methods")
    joined_object = recovered_object.merge(saved_object, on=["source_object_id", "method"],
                                           suffixes=("_root", "_saved"), validate="one_to_one")
    for metric in ("nmse", "mae", "gene_correlation", "direction_agreement", "amplitude_ratio", "conditions"):
        root_values = pd.to_numeric(joined_object[f"{metric}_root"], errors="coerce").to_numpy(dtype=float)
        saved_values = pd.to_numeric(joined_object[f"{metric}_saved"], errors="coerce").to_numpy(dtype=float)
        if metric == "conditions":
            if not np.array_equal(root_values.astype(int), saved_values.astype(int)):
                raise ValueError("saved per-object condition denominators differ on independent readback")
        elif not np.allclose(root_values, saved_values, rtol=0, atol=2e-5, equal_nan=True):
            raise ValueError(f"saved per-object {metric} differs on independent readback")
    root_fold_rows = []
    for fold in range(5):
        objects_in_fold = [obj for obj, obj_fold in object_to_fold.items() if obj_fold == fold]
        for method in METHODS:
            rows = [object_detail[(obj, method)] for obj in objects_in_fold]
            corr = [row["gene_correlation"] for row in rows if np.isfinite(row["gene_correlation"])]
            amp = [row["amplitude_ratio"] for row in rows if np.isfinite(row["amplitude_ratio"])]
            root_fold_rows.append({"fold": fold, "method": method,
                                   "object_macro_nmse": float(np.mean([row["nmse"] for row in rows])),
                                   "object_macro_mae": float(np.mean([row["mae"] for row in rows])),
                                   "object_macro_gene_correlation": float(np.mean(corr)) if corr else np.nan,
                                   "object_macro_direction_agreement": float(np.mean([row["direction_agreement"] for row in rows])),
                                   "object_macro_amplitude_ratio": float(np.mean(amp)) if amp else np.nan})
    root_fold = pd.DataFrame(root_fold_rows)
    joined_fold = root_fold.merge(fold_metrics, on=["fold", "method"], suffixes=("_root", "_saved"),
                                  validate="one_to_one")
    for metric in ("object_macro_nmse", "object_macro_mae", "object_macro_gene_correlation",
                   "object_macro_direction_agreement", "object_macro_amplitude_ratio"):
        root_values = pd.to_numeric(joined_fold[f"{metric}_root"], errors="coerce").to_numpy(dtype=float)
        saved_values = pd.to_numeric(joined_fold[f"{metric}_saved"], errors="coerce").to_numpy(dtype=float)
        if not np.allclose(root_values, saved_values, rtol=0, atol=2e-5, equal_nan=True):
            raise ValueError(f"saved fold {metric} differs on independent readback")
    boot = json.loads((source / "paired_object_bootstrap.json").read_text(encoding="utf-8"))
    pairs = (("per_drug_mean", "state_residual"), ("per_drug_mean", "shuffled_state_control"),
             ("shuffled_state_control", "state_residual"))
    bootstrap_max_abs_error = 0.0
    recovered_pairs = result.get("paired_object_bootstrap", {}).get("pairs", [])
    if recovered and len(recovered_pairs) != len(pairs):
        raise ValueError("recovered result has the wrong paired-bootstrap comparison count")
    for i, (ref, cand) in enumerate(pairs):
        d = np.asarray([object_scores[o][ref] - object_scores[o][cand] for o in sorted(object_scores)])
        rng = np.random.default_rng(53053 + i)
        idx = rng.integers(0, len(d), size=(20000, len(d)))
        boot_samples = d[idx].mean(axis=1)
        expected = np.asarray([float(d.mean()), *np.quantile(boot_samples, [0.025, 0.975])])
        pair_record = boot["pairs"][i]
        got = np.asarray([pair_record["reference_minus_candidate_nmse"], *pair_record["bootstrap_95pct_ci"]])
        if recovered:
            recovered_record = recovered_pairs[i]
            recovered_got = np.asarray([recovered_record["reference_minus_candidate_nmse"],
                                        *recovered_record["bootstrap_95pct_ci"]], dtype=np.float64)
            if (recovered_record.get("reference") != ref or recovered_record.get("candidate") != cand
                    or recovered_record.get("objects") != 110
                    or recovered_record.get("positive_favors_candidate") is not True
                    or np.max(np.abs(expected - recovered_got)) > 2e-5):
                raise ValueError(f"recovered paired-bootstrap comparison {i} differs from independent OOF readback")
        delta = float(np.max(np.abs(expected - got)))
        bootstrap_max_abs_error = max(bootstrap_max_abs_error, delta)
        if delta > 2e-5:
            raise ValueError(f"paired bootstrap row {i} differs by {delta}")
    if recovered:
        failure_path = source / "failure_receipt.json"
        if not failure_path.is_file():
            raise FileNotFoundError("immutable source failure receipt is absent")
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        provenance = result.get("recovery_provenance", {})
        if (failure.get("status") != "FAILED_CLOSED"
                or provenance.get("source_failure_status") != failure.get("status")
                or provenance.get("source_failure_message") != failure.get("error")
                or provenance.get("fitting_performed") is not False
                or provenance.get("new_fold_assignment_created") is not False):
            raise ValueError("recovery provenance does not preserve the original failed run without refit")
        for name, record in provenance.get("source_files", {}).items():
            p = source / name
            if not p.is_file() or sha256(p) != record.get("sha256") or p.stat().st_size != record.get("bytes"):
                raise ValueError(f"immutable source artifact hash changed after recovery: {name}")
        nullable = result.get("undefined_metric_denominators", {})
        for method in METHODS:
            rows = [object_detail[(obj, method)] for obj in sorted(object_scores)]
            condition_corr_n = int(sum(
                1 for idx, obj in enumerate(arrays["source_object_id"].astype(str))
                if np.isfinite(arrays[method][idx]).all() and np.std(truth_delta[idx]) > 0
                and np.std(arrays[method][idx]) > 0))
            expected_counts = {
                "objects_total": 110,
                "objects_gene_correlation_defined": int(sum(np.isfinite(row["gene_correlation"]) for row in rows)),
                "objects_amplitude_ratio_defined": int(sum(np.isfinite(row["amplitude_ratio"]) for row in rows)),
                "conditions_gene_correlation_defined": condition_corr_n,
                "conditions_total": 419,
            }
            if nullable.get(method) != expected_counts:
                raise ValueError(f"recovered {method} undefined-metric denominators differ from independent readback")
        zero = nullable.get("zero_change", {})
        if (not nullable_metrics.get("zero_change")
                or zero.get("objects_gene_correlation_defined") != 0
                or zero.get("conditions_gene_correlation_defined") != 0):
            raise ValueError("zero-change correlation must remain undefined with a zero valid denominator")
        shuffle_seed = int(cfg["models"]["shuffled_state_capacity_control"]["seed"])
        shuffle_rng = np.random.default_rng(shuffle_seed)
        expected_shuffle_receipts = []
        for fold in range(5):
            train_idx = np.flatnonzero(arrays["fold_id"].astype(int) != fold)
            test_idx = np.flatnonzero(arrays["fold_id"].astype(int) == fold)
            train_ids = sorted(set(arrays["source_object_id"].astype(str)[train_idx]))
            test_ids = sorted(set(arrays["source_object_id"].astype(str)[test_idx]))
            train_map, test_map = derangement(train_ids, shuffle_rng), derangement(test_ids, shuffle_rng)
            mapping_json = json.dumps({"train": train_map, "test": test_map}, sort_keys=True,
                                      separators=(",", ":"), ensure_ascii=False)
            expected_shuffle_receipts.append({
                "fold": fold, "seed": shuffle_seed, "train_objects": len(train_ids), "test_objects": len(test_ids),
                "mapping_sha256": hashlib.sha256(mapping_json.encode()).hexdigest(),
                "mapping_reconstructed_from_registered_seed": True, "model_refit_during_recovery": False,
            })
        if result.get("shuffle_mapping_receipts") != expected_shuffle_receipts:
            raise ValueError("recovered deterministic within-partition object-shuffle receipts differ from seed replay")
    manifest = pd.read_csv(out / "sha256_manifest.tsv", sep="\t", dtype=str)
    hash_checks = {}
    for row in manifest.itertuples(index=False):
        p = out / row.file
        hash_checks[row.file] = p.is_file() and sha256(p) == row.sha256
    if not hash_checks or not all(hash_checks.values()):
        raise ValueError("one or more run manifest SHA-256 entries failed")
    if result["source_input_receipt"].get("inner_dev_numeric_response_read") is not False:
        raise ValueError("runner receipt reports a forbidden inner response read")
    if result.get("forbidden_roles_opened") != [] or any(result.get("forbidden_numeric_values_read", {}).values()):
        raise ValueError("runner receipt reports a forbidden data role")
    receipt = {"status": "PASS_ROOT_INDEPENDENT_RECOVERY_READBACK" if recovered else
                          "PASS_ROOT_INDEPENDENT_TRAIN_OBJECT_OOF_READBACK",
               "run_id": run_id, "source_run_id": source_run_id, "verified_at_utc": fit.utc_now(),
               "checks": {"train_rows": len(keys), "train_objects": len(object_to_fold),
                          "full_inchikeys": int(keys.full_inchikey.nunique()), "genes": len(expected_genes),
                          "folds": 5, "fold_object_exclusivity": True,
                          "all_condition_keys_exactly_once": True, "all_predictions_finite": True,
                          "training_drug_support_each_fold": 5, "inner_dev_numeric_values_read": False,
                          "pdo_outer_locked_opened": False, "source_train_delta_max_abs_error": truth_error,
                          "metric_max_abs_error": metric_max_abs_error,
                          "bootstrap_max_abs_error": bootstrap_max_abs_error,
                          "manifest_sha_checks": hash_checks,
                          "recovered_without_refit": recovered,
                          "undefined_result_metrics": nullable_metrics},
               "gpu_hours": 0.0, "cpu_seconds": time.monotonic() - started,
               "claim_boundary": "TRAIN-only source-object OOF development result; not an independent PDO/generalization test"}
    fit.atomic_json(out / "root_independent_readback.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
