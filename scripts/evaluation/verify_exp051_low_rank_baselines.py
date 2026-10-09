"""Independent keyed readback for the EXP-051 CPU low-rank baseline package.

This verifier consumes only the frozen response-blind TRAIN/inner-dev values and
the new low-rank prediction artifact. It does not import the result-generation
functions, so its metric and key checks are independent of the runner.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/experiments/EXP-051/20261006_exp051_contract_fit_smoke_v1"
RUN = ROOT / "artifacts/experiments/EXP-051/20261007_exp051_low_rank_baselines_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def metrics(keys: pd.DataFrame, truth: np.ndarray, pred: np.ndarray,
            train_variance: np.ndarray) -> dict[str, Any]:
    if truth.shape != pred.shape or truth.ndim != 2 or truth.shape[1] != 978:
        raise ValueError("prediction/target axes differ from the 978-gene contract")
    use = np.isfinite(train_variance) & (train_variance > 1e-12)
    if not use.any():
        raise ValueError("no positive TRAIN-fitted gene-variance denominator")
    err = pred.astype(np.float64) - truth.astype(np.float64)
    tgt = truth.astype(np.float64)
    nmse = np.mean(err[:, use] ** 2 / train_variance[use][None, :], axis=1)
    mae = np.mean(np.abs(err), axis=1)
    per_row = pd.DataFrame({"object": keys.source_object_id.astype(str), "nmse": nmse,
                            "mae": mae, "correlation": np.nan, "direction": np.nan})
    # Preserve row positions for undefined per-condition correlations/directions.
    for row_index, (y, p) in enumerate(zip(tgt, pred.astype(np.float64))):
        if np.ptp(y) > 0 and np.ptp(p) > 0:
            value = float(np.corrcoef(y, p)[0, 1])
            if np.isfinite(value):
                per_row.loc[row_index, "correlation"] = value
        nz = np.abs(y) > 1e-12
        if nz.any():
            per_row.loc[row_index, "direction"] = float(np.mean(np.sign(y[nz]) == np.sign(p[nz])))
    per_object = per_row.groupby("object", sort=True).mean()
    object_amplitudes = []
    for object_id, ix in keys.groupby("source_object_id", sort=True).indices.items():
        ix = np.asarray(ix, dtype=np.int64)
        observed = float(np.mean(np.abs(tgt[ix])))
        predicted = float(np.mean(np.abs(pred[ix].astype(np.float64))))
        if observed > 0:
            object_amplitudes.append(predicted / observed)
    return {
        "n_conditions": int(len(keys)), "n_source_objects": int(keys.source_object_id.nunique()),
        "n_full_inchikeys": int(keys.full_inchikey.nunique()), "n_genes_in_nmse_denominator": int(use.sum()),
        "n_zero_variance_genes_excluded": int((~use).sum()),
        "object_macro_nmse": _safe_float(float(per_object.nmse.mean())),
        "object_macro_mae": _safe_float(float(per_object.mae.mean())),
        "object_macro_gene_correlation": _safe_float(float(per_object.correlation.mean())) if per_object.correlation.notna().any() else None,
        "object_macro_direction_agreement": _safe_float(float(per_object.direction.mean())) if per_object.direction.notna().any() else None,
        "object_macro_amplitude_ratio": _safe_float(float(np.mean(object_amplitudes))) if object_amplitudes else None,
        "n_amplitude_objects_with_nonzero_observed_mean_abs": len(object_amplitudes),
    }


def require_close(generated: dict[str, Any], recomputed: dict[str, Any], label: str) -> float:
    diffs = []
    for key, actual in recomputed.items():
        expected = generated.get(key)
        if actual is None or expected is None:
            if actual is not expected:
                raise ValueError(f"{label}.{key}: undefined-value semantics differ")
            continue
        diffs.append(abs(float(actual) - float(expected)))
    maximum = max(diffs, default=0.0)
    if maximum > 2e-6:
        raise ValueError(f"{label}: independent metric mismatch, max_abs_diff={maximum}")
    return maximum


def main() -> None:
    receipts = {
        "freeze": SOURCE / "metadata_freeze_receipt.json",
        "axis": SOURCE / "gene_axis_receipt.json",
        "value": SOURCE / "value_contract_receipt.json",
        "root_readback": SOURCE / "root_values_readback_receipt.json",
        "feature": SOURCE / "transigen_ecfp4_feature_audit_receipt.json",
    }
    for name, path in receipts.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen {name} receipt: {path}")
    value_path = SOURCE / "frozen_train_inner_values.npz"
    value_receipt = json.loads(receipts["value"].read_text(encoding="utf-8"))
    if sha256(value_path) != value_receipt.get("files_sha256", {}).get("frozen_train_inner_values_npz"):
        raise ValueError("frozen value NPZ checksum differs from its receipt")
    root_receipt = json.loads(receipts["root_readback"].read_text(encoding="utf-8"))
    if root_receipt.get("checks", {}).get("result") != "PASS":
        raise ValueError("the frozen values lack an independent root readback PASS")

    keys = pd.read_csv(SOURCE / "response_blind_conditions.tsv", sep="\t", dtype=str)
    # This trusted, checksum-bound artifact contains a small number of categorical
    # object arrays. Explicitly select only TRAIN and inner-dev rows by frozen role.
    train = keys.split_role.eq("train").to_numpy()
    dev = keys.split_role.eq("inner_dev").to_numpy()
    if (int(train.sum()), int(dev.sum()), keys.loc[train, "source_object_id"].nunique(),
            keys.loc[dev, "source_object_id"].nunique()) != (419, 93, 110, 24):
        raise ValueError("frozen condition/object denominator differs from 419/110 and 93/24")
    train_keys, dev_keys = keys.loc[train].reset_index(drop=True), keys.loc[dev].reset_index(drop=True)
    if set(train_keys.source_object_id) & set(dev_keys.source_object_id):
        raise ValueError("source object crosses TRAIN and inner-dev roles")
    with np.load(value_path, allow_pickle=True) as archive:
        source_roles = archive["split_role"].astype(str)
        if not np.array_equal(source_roles, keys.split_role.astype(str).to_numpy()):
            raise ValueError("response-blind key roles do not match frozen values")
        train_y = archive["delta"][train].astype(np.float32, copy=True)
        dev_y = archive["delta"][dev].astype(np.float32, copy=True)
        genes = archive["genes"].astype(str)
        frozen_train_keys = archive["full_inchikey"][train].astype(str)
        frozen_dev_keys = archive["full_inchikey"][dev].astype(str)
    if len(genes) != 978 or len(set(genes.tolist())) != 978:
        raise ValueError("frozen gene axis is not 978 unique genes")
    if (not np.array_equal(frozen_train_keys, train_keys.full_inchikey.astype(str).to_numpy())
            or not np.array_equal(frozen_dev_keys, dev_keys.full_inchikey.astype(str).to_numpy())):
        raise ValueError("frozen full-InChIKey order differs from the keyed condition table")
    variance = np.var(train_y.astype(np.float64), axis=0, ddof=0)

    result_path = RUN / "result.json"
    pred_path = RUN / "low_rank_predictions.npz"
    baseline_path = RUN / "baseline_predictions.npz"
    if not all(path.is_file() for path in (result_path, pred_path, baseline_path)):
        raise FileNotFoundError("low-rank run has not emitted its complete result and keyed predictions")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "COMPLETED_CPU_ONLY_LOW_RANK_BASELINES":
        raise ValueError("low-rank result is not a closed successful CPU-only run")
    if result.get("run_id") != RUN.name:
        raise ValueError("run_id embedded in the result does not match the artifact directory")
    if result.get("source_files", {}).get("values_npz_sha256") != sha256(value_path):
        raise ValueError("result does not bind the frozen TRAIN/inner value archive")
    if sha256(pred_path) != result["predictions"]["sha256"]:
        raise ValueError("prediction artifact SHA differs from result receipt")
    if sha256(baseline_path) != result["fixed_baseline"]["prediction_sha256"]:
        raise ValueError("baseline prediction artifact SHA differs from result receipt")
    # The runner's keyed NPZ contains NumPy object arrays only for categorical
    # strings. It is locally generated by the audited runner and now bound by
    # the closed result receipt and SHA-256; validate that only the allowlisted
    # key/label arrays use object dtype before comparing them to frozen tables.
    with np.load(pred_path, allow_pickle=True) as pred:
        allowed_object_arrays = {"train_condition_key", "inner_dev_condition_key",
                                 "train_source_object_id", "inner_dev_source_object_id",
                                 "train_full_inchikey", "inner_dev_full_inchikey", "genes"}
        for name in pred.files:
            if pred[name].dtype.hasobject and name not in allowed_object_arrays:
                raise ValueError(f"unexpected object dtype in prediction array {name}")
        expected = {
            "train_condition_key": train_keys.full_condition_key.astype(str).to_numpy(),
            "inner_dev_condition_key": dev_keys.full_condition_key.astype(str).to_numpy(),
            "train_source_object_id": train_keys.source_object_id.astype(str).to_numpy(),
            "inner_dev_source_object_id": dev_keys.source_object_id.astype(str).to_numpy(),
            "train_full_inchikey": train_keys.full_inchikey.astype(str).to_numpy(),
            "inner_dev_full_inchikey": dev_keys.full_inchikey.astype(str).to_numpy(),
            "genes": genes,
            "train_target_delta": train_y,
            "inner_dev_target_delta": dev_y,
        }
        for name, want in expected.items():
            if not np.array_equal(pred[name], want):
                raise ValueError(f"prediction archive keyed/source readback differs: {name}")
        recomputed: dict[str, dict[str, Any]] = {}
        for model_name in ("structural_only", "state_plus_structure"):
            train_pred = pred[f"{model_name}_train"]
            dev_pred = pred[f"{model_name}_inner_dev"]
            if train_pred.shape != train_y.shape or dev_pred.shape != dev_y.shape:
                raise ValueError(f"{model_name} prediction dimensions differ from frozen targets")
            if not np.isfinite(train_pred).all() or not np.isfinite(dev_pred).all():
                raise FloatingPointError(f"{model_name} contains non-finite predictions")
            recomputed[model_name] = {
                "train": metrics(train_keys, train_y, train_pred, variance),
                "inner_dev": metrics(dev_keys, dev_y, dev_pred, variance),
            }
    with np.load(baseline_path, allow_pickle=True) as base:
        for name in base.files:
            if base[name].dtype.hasobject and name not in {"inner_dev_condition_key"}:
                raise ValueError(f"unexpected object dtype in baseline array {name}")
        if not np.array_equal(base["inner_dev_condition_key"].astype(str), dev_keys.full_condition_key.astype(str).to_numpy()):
            raise ValueError("per-drug reference condition order differs")
        base_pred = base["per_drug_mean_delta_inner_dev"].astype(np.float32, copy=True)
    if not np.isfinite(base_pred).all():
        raise FloatingPointError("fixed per-drug reference is non-finite")
    ref_metrics = metrics(dev_keys, dev_y, base_pred, variance)
    ref_objects = pd.DataFrame({"object": dev_keys.source_object_id.astype(str),
                                "nmse": metrics(dev_keys, dev_y, base_pred, variance)["object_macro_nmse"]})
    # Calculate the proportion of inner-dev objects improved over the frozen per-drug mean.
    per_object_score: dict[str, dict[str, float]] = {}
    for model_name in ("structural_only", "state_plus_structure"):
        pred_npz = np.load(pred_path, allow_pickle=False)
        try:
            p = pred_npz[f"{model_name}_inner_dev"].astype(np.float64, copy=True)
        finally:
            pred_npz.close()
        row_nmse = np.mean((p.astype(np.float64) - dev_y.astype(np.float64)) ** 2 /
                           variance[None, :], axis=1)
        frame = pd.DataFrame({"object": dev_keys.source_object_id.astype(str), "nmse": row_nmse})
        candidate_obj = frame.groupby("object", sort=True).nmse.mean().to_dict()
        base_nmse = np.mean((base_pred.astype(np.float64) - dev_y.astype(np.float64)) ** 2 /
                            variance[None, :], axis=1)
        base_frame = pd.DataFrame({"object": dev_keys.source_object_id.astype(str), "nmse": base_nmse})
        reference_obj = base_frame.groupby("object", sort=True).nmse.mean().to_dict()
        if set(candidate_obj) != set(reference_obj) or len(candidate_obj) != 24:
            raise ValueError("paired baseline comparison does not share the 24-object denominator")
        per_object_score[model_name] = candidate_obj

    metric_diffs: dict[str, float] = {}
    for model_name in recomputed:
        for split in ("train", "inner_dev"):
            metric_diffs[f"{model_name}.{split}"] = require_close(
                result["methods"][model_name][split], recomputed[model_name][split],
                f"{model_name}.{split}")
    dev_base_nmse = ref_metrics["object_macro_nmse"]
    comparisons = {}
    for model_name, obj in per_object_score.items():
        candidate = float(np.mean(list(obj.values())))
        reference_by_object = {}
        row_nmse = np.mean((base_pred.astype(np.float64) - dev_y.astype(np.float64)) ** 2 /
                           variance[None, :], axis=1)
        for source_object, idx in dev_keys.groupby("source_object_id", sort=True).indices.items():
            reference_by_object[str(source_object)] = float(np.mean(row_nmse[np.asarray(idx, dtype=np.int64)]))
        differences = np.asarray([reference_by_object[key] - obj[key] for key in sorted(obj)], dtype=np.float64)
        comparisons[model_name] = {
            "inner_dev_nmse": candidate,
            "per_drug_mean_inner_dev_nmse": dev_base_nmse,
            "absolute_nmse_improvement": float(dev_base_nmse - candidate),
            "relative_nmse_reduction": float((dev_base_nmse - candidate) / dev_base_nmse) if dev_base_nmse else None,
            "objects_improved": int(np.sum(differences > 0)), "n_paired_objects": len(differences),
            "mean_object_paired_difference_reference_minus_candidate": float(differences.mean()),
        }
    receipt = {
        "status": "ROOT_INDEPENDENT_KEYED_READBACK_PASS",
        "run_id": result["run_id"], "run_result_sha256": sha256(result_path),
        "prediction_sha256": sha256(pred_path), "baseline_prediction_sha256": sha256(baseline_path),
        "frozen_value_sha256": sha256(value_path),
        "denominator": {"train_conditions": len(train_keys), "train_objects": int(train_keys.source_object_id.nunique()),
                        "inner_dev_conditions": len(dev_keys), "inner_dev_objects": int(dev_keys.source_object_id.nunique()),
                        "object_overlap": 0, "full_inchikeys": int(keys.full_inchikey.nunique()), "genes": len(genes)},
        "checks": {"keys_and_object_order": "PASS", "targets_match_frozen_values": "PASS",
                   "train_inner_role_separation": "PASS", "all_predictions_finite": "PASS",
                   "metric_recomputation_max_abs_diffs": metric_diffs},
        "per_drug_mean_reference_inner_dev": ref_metrics,
        "comparisons_vs_per_drug_mean": comparisons,
        "claim_boundary": "Fixed EXP-051 inner-development split only; this is not independent or PDO validation.",
    }
    out = RUN / "root_independent_readback.json"
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
