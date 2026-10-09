"""Independent key/metric readback for the XPert-only EXP-051 fixed100 fit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluation.verify_exp051_low_rank_baselines import metrics, sha256
SOURCE = ROOT / "artifacts/experiments/EXP-051/20261006_exp051_contract_fit_smoke_v1"
ATTEMPT = ROOT / "artifacts/experiments/EXP-051/20261006_exp051_representative_compare_v1/fit_attempt_004"


def main() -> None:
    receipt_path = ATTEMPT / "attempt_close_receipt.json"
    fit_path = ATTEMPT / "full_fit_result.json"
    pred_path = ATTEMPT / "full_keyed_predictions.npz"
    manifest_path = ATTEMPT / "fit_smoke_manifest.json"
    baseline_path = ATTEMPT / "baseline_predictions.npz"
    value_path = SOURCE / "frozen_train_inner_values.npz"
    values_receipt_path = SOURCE / "value_contract_receipt.json"
    for path in (receipt_path, fit_path, pred_path, manifest_path, baseline_path,
                 value_path, values_receipt_path):
        if not path.is_file():
            raise FileNotFoundError(f"required frozen fit artifact is missing: {path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    value_receipt = json.loads(values_receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "COMPLETED_XPERT_ONLY" or manifest.get("selected_models") != ["xpert"]:
        raise ValueError("attempt was not a completed, selected XPert-only fit")
    if fit.get("unselected_models_initialized_or_evaluated") != []:
        raise ValueError("an unselected model was initialized or evaluated")
    if sha256(value_path) != value_receipt.get("files_sha256", {}).get("frozen_train_inner_values_npz"):
        raise ValueError("source frozen-value hash differs from its receipt")
    if sha256(pred_path) != fit.get("predictions_sha256"):
        raise ValueError("full-key prediction hash differs from fit summary")
    if sha256(ATTEMPT / "XPert_best_inner_dev.pt") != fit["best_metrics"]["XPert"]["checkpoint_best_sha256"]:
        raise ValueError("selected XPert checkpoint SHA differs")
    if sha256(ATTEMPT / "XPert_final_epoch_100.pt") != fit["best_metrics"]["XPert"]["checkpoint_final_sha256"]:
        raise ValueError("XPert epoch-100 checkpoint SHA differs")

    keys = pd.read_csv(SOURCE / "response_blind_conditions.tsv", sep="\t", dtype=str)
    train = keys.split_role.eq("train").to_numpy()
    dev = keys.split_role.eq("inner_dev").to_numpy()
    train_keys = keys.loc[train].reset_index(drop=True)
    dev_keys = keys.loc[dev].reset_index(drop=True)
    if (int(train.sum()), int(dev.sum()), train_keys.source_object_id.nunique(),
            dev_keys.source_object_id.nunique()) != (419, 93, 110, 24):
        raise ValueError("frozen TRAIN/inner-dev denominator changed")
    if set(train_keys.source_object_id) & set(dev_keys.source_object_id):
        raise ValueError("source object leaks across TRAIN and inner-dev")
    with np.load(value_path, allow_pickle=True) as source_values:
        source_roles = source_values["split_role"].astype(str)
        if not np.array_equal(source_roles, keys.split_role.astype(str).to_numpy()):
            raise ValueError("frozen value/key roles disagree")
        train_y = source_values["delta"][train].astype(np.float32, copy=True)
        dev_y = source_values["delta"][dev].astype(np.float32, copy=True)
        genes = source_values["genes"].astype(str)
    if len(genes) != 978 or len(set(genes.tolist())) != 978:
        raise ValueError("frozen gene axis is not exactly 978 unique genes")
    train_var = np.var(train_y.astype(np.float64), axis=0, ddof=0)
    # The executor wrote its categorical axes as NumPy object arrays. The file
    # is checksum-bound to this run's manifest/result receipt before deserialization.
    with np.load(pred_path, allow_pickle=True) as prediction:
        expected = {
            "train_condition_key": train_keys.full_condition_key.astype(str).to_numpy(),
            "inner_dev_condition_key": dev_keys.full_condition_key.astype(str).to_numpy(),
            "train_source_object_id": train_keys.source_object_id.astype(str).to_numpy(),
            "inner_dev_source_object_id": dev_keys.source_object_id.astype(str).to_numpy(),
            "train_full_inchikey": train_keys.full_inchikey.astype(str).to_numpy(),
            "inner_dev_full_inchikey": dev_keys.full_inchikey.astype(str).to_numpy(),
            "genes": genes, "train_target_delta": train_y, "inner_dev_target_delta": dev_y,
        }
        for name, want in expected.items():
            if not np.array_equal(prediction[name], want):
                raise ValueError(f"keyed output does not match frozen axis/order for {name}")
        train_pred = prediction["XPert_best_inner_dev_train_delta"].astype(np.float32, copy=True)
        dev_pred = prediction["XPert_best_inner_dev_dev_delta"].astype(np.float32, copy=True)
    if not np.isfinite(train_pred).all() or not np.isfinite(dev_pred).all():
        raise FloatingPointError("best inner-dev-selected XPert predictions contain non-finite values")
    train_metrics = metrics(train_keys, train_y, train_pred, train_var)
    dev_metrics = metrics(dev_keys, dev_y, dev_pred, train_var)
    reported = fit["best_metrics"]["XPert"]
    diffs = {}
    for role, recomputed in (("train", train_metrics), ("inner_dev", dev_metrics)):
        for key, value in recomputed.items():
            expected_value = reported[role].get(key)
            if value is None or expected_value is None:
                if value is not expected_value:
                    raise ValueError(f"{role}.{key} undefined handling differs")
                continue
            diffs[f"{role}.{key}"] = abs(float(value) - float(expected_value))
    max_diff = max(diffs.values(), default=0.0)
    if max_diff > 2e-6:
        raise ValueError(f"root metric readback differs; maximum absolute difference={max_diff}")

    with np.load(baseline_path, allow_pickle=False) as base:
        if not np.array_equal(base["train_condition_key"].astype(str), train_keys.full_condition_key.astype(str).to_numpy()):
            raise ValueError("baseline TRAIN key denominator/order differs")
        if not np.array_equal(base["inner_dev_condition_key"].astype(str), dev_keys.full_condition_key.astype(str).to_numpy()):
            raise ValueError("baseline inner-dev key denominator/order differs")
        drug_train = base["per_drug_mean_delta_train"].astype(np.float32, copy=True)
        drug_dev = base["per_drug_mean_delta_inner_dev"].astype(np.float32, copy=True)
    baseline_train = metrics(train_keys, train_y, drug_train, train_var)
    baseline_dev = metrics(dev_keys, dev_y, drug_dev, train_var)
    gate = reported["fit_gate"]
    receipt_out = {
        "status": "ROOT_INDEPENDENT_XPERT_KEYED_READBACK_PASS",
        "run_id": manifest["run_id"], "attempt_id": ATTEMPT.name,
        "gpu_hours_actual": receipt["actual_gpu_hours"],
        "prediction_sha256": sha256(pred_path), "checkpoint_best_sha256": sha256(ATTEMPT / "XPert_best_inner_dev.pt"),
        "denominator": {"train_conditions": len(train_keys), "train_objects": int(train_keys.source_object_id.nunique()),
                        "inner_dev_conditions": len(dev_keys), "inner_dev_objects": int(dev_keys.source_object_id.nunique()),
                        "object_overlap": 0, "full_inchikeys": int(keys.full_inchikey.nunique()), "genes": len(genes)},
        "metrics_recomputed_from_keyed_predictions": {"train": train_metrics, "inner_dev": dev_metrics},
        "max_metric_abs_difference_from_runner": max_diff,
        "per_drug_mean_reference": {"train": baseline_train, "inner_dev": baseline_dev},
        "fit_gate": gate,
        "interpretation": "The frozen XPert implementation was trained for all 100 prescribed epochs, but its training-side NMSE advantage over the per-drug mean is only about 0.5% and its prediction amplitude is below the registered 0.5 lower bound. This is a current-implementation fit-gate failure, not evidence that XPert lacks information or that PDO transfer fails.",
        "forbidden_roles_opened": [],
    }
    output = ATTEMPT / "root_independent_xpert_readback.json"
    output.write_text(json.dumps(receipt_out, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt_out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
