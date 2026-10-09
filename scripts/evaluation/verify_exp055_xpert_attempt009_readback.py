"""Root-independent recovery/readback for the closed EXP-055 XPert attempt009.

This script never trains, restores a model for inference, or reads PDO/outer/locked
labels. It validates the closed run's keyed prediction archive, the TRAIN/inner-dev
split contract, selected-epoch identity, and metrics against the saved curve.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from verify_exp051_low_rank_baselines import metrics, sha256


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "artifacts/experiments/EXP-051/20261006_exp051_contract_fit_smoke_v1"
ATTEMPT = ROOT / "artifacts/experiments/EXP-055/20261007_exp055_representative_models_v1/fit_attempt_009"
BASELINE = ROOT / "artifacts/experiments/EXP-055/20261007_exp055_representative_models_v1/fit_attempt_009/baseline_metrics.json"


def main() -> None:
    manifest_path = ATTEMPT / "fit_smoke_manifest.json"
    values_path = SOURCE / "frozen_train_inner_values.npz"
    values_receipt_path = SOURCE / "value_contract_receipt.json"
    keys_path = SOURCE / "response_blind_conditions.tsv"
    predictions_path = ATTEMPT / "full_keyed_predictions.npz"
    curve_path = ATTEMPT / "XPert_training_curve.jsonl"
    close_path = ATTEMPT / "failure_close_receipt.json"
    baseline_path = ATTEMPT / "baseline_metrics.json"
    selected_checkpoint = ATTEMPT / "XPert_best_inner_dev.pt"
    final_checkpoint = ATTEMPT / "XPert_final_stop_epoch_189.pt"
    required = (manifest_path, values_path, values_receipt_path, keys_path,
                predictions_path, curve_path, close_path, baseline_path,
                selected_checkpoint, final_checkpoint)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("required closed-run evidence missing: " + ", ".join(missing))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    value_receipt = json.loads(values_receipt_path.read_text(encoding="utf-8"))
    close = json.loads(close_path.read_text(encoding="utf-8"))
    progress = json.loads((ATTEMPT / "training_progress.json").read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if manifest.get("fit_attempt") != ATTEMPT.name or manifest.get("selected_models") != ["xpert"]:
        raise ValueError("manifest does not bind the single XPert attempt009")
    if manifest.get("experiment_id") != "EXP-055" or manifest.get("phase") != "EXP055_XPERT_FRESH_ORIGINAL_FULL_2500_OR_PATIENCE50":
        raise ValueError("manifest differs from the approved full-schedule XPert phase")
    if close.get("status") != "FAILED_CLOSED" or close.get("attempt") != ATTEMPT.name:
        raise ValueError("attempt009 is not the immutable closed attempt described by its receipt")
    if close.get("exception_type") != "TypeError" or "Tensor is not JSON serializable" not in close.get("exception_message", ""):
        raise ValueError("closed-run failure is not the recorded post-fit serialization error")

    conditions = pd.read_csv(keys_path, sep="\t", dtype=str)
    if sha256(keys_path) != manifest["frozen_files"]["conditions_tsv_sha256"]:
        raise ValueError("frozen response-blind condition table SHA differs from manifest")
    if sha256(values_path) != manifest["frozen_files"]["values_npz_sha256"]:
        raise ValueError("frozen TRAIN/inner-dev values SHA differs from manifest")
    if sha256(values_path) != value_receipt.get("files_sha256", {}).get("frozen_train_inner_values_npz"):
        raise ValueError("frozen values do not match the existing value-contract receipt")
    train_mask = conditions["split_role"].eq("train").to_numpy()
    dev_mask = conditions["split_role"].eq("inner_dev").to_numpy()
    train_keys = conditions.loc[train_mask].reset_index(drop=True)
    dev_keys = conditions.loc[dev_mask].reset_index(drop=True)
    denominator = (int(train_mask.sum()), int(train_keys.source_object_id.nunique()),
                   int(dev_mask.sum()), int(dev_keys.source_object_id.nunique()))
    if denominator != (419, 110, 93, 24):
        raise ValueError(f"frozen TRAIN/inner-dev denominator changed: {denominator}")
    overlap = set(train_keys.source_object_id) & set(dev_keys.source_object_id)
    if overlap:
        raise ValueError(f"source objects cross TRAIN and inner-dev: {sorted(overlap)[:3]}")

    # This frozen project archive was already root-readback verified and is
    # checksum-bound above. Its metadata also contains categorical object arrays
    # (for example pert_id); the response matrix itself must remain numeric.
    with np.load(values_path, allow_pickle=True) as values:
        if values["delta"].dtype.hasobject:
            raise ValueError("frozen response delta matrix unexpectedly uses object dtype")
        roles = values["split_role"].astype(str)
        genes = values["genes"].astype(str)
        train_y = values["delta"][train_mask].astype(np.float32, copy=True)
        dev_y = values["delta"][dev_mask].astype(np.float32, copy=True)
        frozen_ik_train = values["full_inchikey"][train_mask].astype(str)
        frozen_ik_dev = values["full_inchikey"][dev_mask].astype(str)
    if not np.array_equal(roles, conditions["split_role"].astype(str).to_numpy()):
        raise ValueError("response-blind row roles differ from the frozen values")
    if len(genes) != 978 or len(set(genes.tolist())) != 978:
        raise ValueError("frozen gene axis is not exactly 978 unique genes")
    for got, want, label in (
        (frozen_ik_train, train_keys.full_inchikey.astype(str).to_numpy(), "TRAIN InChIKeys"),
        (frozen_ik_dev, dev_keys.full_inchikey.astype(str).to_numpy(), "inner-dev InChIKeys"),
    ):
        if not np.array_equal(got, want):
            raise ValueError(f"frozen values/key table mismatch: {label}")

    curve = [json.loads(line) for line in curve_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not curve or [row["epoch"] for row in curve] != list(range(1, len(curve) + 1)):
        raise ValueError("training curve is empty or has missing/duplicate epochs")
    best_epoch = int(progress["best_inner_dev_epoch"])
    completed_epochs = int(progress["completed_epochs"])
    if completed_epochs != len(curve) or completed_epochs != 189:
        raise ValueError("completed epoch count does not match the immutable 189-row curve")
    best_row = min(curve, key=lambda row: float(row["inner_dev_val_loss4"]))
    final_row = curve[-1]
    if int(best_row["epoch"]) != best_epoch or best_epoch != 139:
        raise ValueError("selected epoch is not the minimum author val_loss4 checkpoint")

    # weights_only=True avoids unrestricted checkpoint pickle loading.
    best_state = torch.load(selected_checkpoint, map_location="cpu", weights_only=True)
    final_state = torch.load(final_checkpoint, map_location="cpu", weights_only=True)
    if int(best_state.get("epoch_one_based", -1)) != best_epoch:
        raise ValueError("selected checkpoint epoch metadata differs from the curve")
    if best_state.get("selection_metric") != "minimum inner_dev val_loss4":
        raise ValueError("selected checkpoint does not identify the approved selection metric")
    if int(final_state.get("epoch", -1)) != completed_epochs:
        raise ValueError(f"final-stop checkpoint epoch metadata differs from the curve: {final_state.get('epoch')!r} versus {completed_epochs}")

    train_var = np.var(train_y.astype(np.float64), axis=0, ddof=0)
    with np.load(predictions_path, allow_pickle=False) as pred:
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
            if name not in pred.files or not np.array_equal(pred[name].astype(str) if pred[name].dtype.kind in "US" else pred[name], want.astype(str) if np.asarray(want).dtype.kind in "US" else want):
                raise ValueError(f"keyed prediction readback differs: {name}")
        train_pred = pred["XPert_best_inner_dev_train_delta"].astype(np.float32, copy=True)
        dev_pred = pred["XPert_best_inner_dev_dev_delta"].astype(np.float32, copy=True)
        final_train_pred = pred["XPert_final_stop_train_delta"].astype(np.float32, copy=True)
        final_dev_pred = pred["XPert_final_stop_inner_dev_delta"].astype(np.float32, copy=True)
        object_arrays = [name for name in pred.files if pred[name].dtype.hasobject]
        if object_arrays:
            raise ValueError(f"prediction archive is not safe to read without pickle: {object_arrays}")
    for label, arr in (("selected TRAIN", train_pred), ("selected inner-dev", dev_pred),
                       ("final TRAIN", final_train_pred), ("final inner-dev", final_dev_pred)):
        if not np.isfinite(arr).all():
            raise FloatingPointError(f"{label} predictions contain non-finite values")

    train_metric = metrics(train_keys, train_y, train_pred, train_var)
    dev_metric = metrics(dev_keys, dev_y, dev_pred, train_var)
    final_train_metric = metrics(train_keys, train_y, final_train_pred, train_var)
    final_dev_metric = metrics(dev_keys, dev_y, final_dev_pred, train_var)
    metric_fields = ("object_macro_nmse", "object_macro_mae", "object_macro_gene_correlation",
                     "object_macro_direction_agreement", "object_macro_amplitude_ratio")
    curve_metrics = {
        "train": best_row["train_metrics"], "inner_dev": best_row["inner_dev_metrics"],
    }
    runner_metrics = {"train": train_metric, "inner_dev": dev_metric}
    diffs: dict[str, float] = {}
    for role in ("train", "inner_dev"):
        for name in metric_fields:
            actual = runner_metrics[role].get(name)
            recorded = curve_metrics[role].get(name)
            if actual is None or recorded is None:
                if actual is not recorded:
                    raise ValueError(f"undefined metric handling differs for {role}.{name}")
            else:
                diffs[f"{role}.{name}"] = abs(float(actual) - float(recorded))
    max_metric_diff = max(diffs.values(), default=0.0)
    if max_metric_diff > 2e-6:
        raise ValueError(f"recomputed metrics disagree with selected curve row: {max_metric_diff}")

    baseline_train_nmse = float(baseline["metrics"]["per_drug_mean_delta"]["train"]["object_macro_nmse"])
    baseline_dev_nmse = float(baseline["metrics"]["per_drug_mean_delta"]["inner_dev"]["object_macro_nmse"])
    train_nmse_reduction = 1.0 - float(train_metric["object_macro_nmse"]) / baseline_train_nmse
    dev_nmse_reduction = 1.0 - float(dev_metric["object_macro_nmse"]) / baseline_dev_nmse
    amplitude = float(train_metric["object_macro_amplitude_ratio"])
    gate = {
        "train_nmse_relative_reduction_vs_per_drug_mean": train_nmse_reduction,
        "minimum_required_train_nmse_reduction": 0.10,
        "train_amplitude_ratio": amplitude,
        "amplitude_ratio_range": [0.5, 2.0],
        "train_amplitude_gate_passed": 0.5 <= amplitude <= 2.0,
        "train_nmse_gate_passed": train_nmse_reduction >= 0.10,
        "overall_fit_gate_passed": train_nmse_reduction >= 0.10 and 0.5 <= amplitude <= 2.0,
    }
    out = {
        "status": "ROOT_INDEPENDENT_ATTEMPT009_KEYED_METRIC_READBACK_PASS",
        "experiment_id": "EXP-055",
        "attempt_id": ATTEMPT.name,
        "interpretation_boundary": "Successful fit artifact recovery after runner post-fit JSON serialization failure; not a successful runner close receipt and not a PDO/outer/locked transfer result.",
        "source_hashes": {"manifest": sha256(manifest_path), "frozen_values": sha256(values_path),
                          "frozen_conditions": sha256(keys_path), "baseline_metrics": sha256(baseline_path),
                          "training_curve": sha256(curve_path), "keyed_predictions": sha256(predictions_path),
                          "selected_checkpoint": sha256(selected_checkpoint), "final_stop_checkpoint": sha256(final_checkpoint)},
        "denominator": {"train_conditions": 419, "train_objects": 110, "inner_dev_conditions": 93,
                        "inner_dev_objects": 24, "source_object_overlap": 0, "full_inchikeys": 5, "genes": 978},
        "fit_schedule": {"completed_epochs": completed_epochs, "selected_epoch_by_minimum_val_loss4": best_epoch,
                         "final_stop_epoch": int(final_row["epoch"]), "stopped_at_patience_boundary": completed_epochs - best_epoch == 50,
                         "selected_val_loss4": float(best_row["inner_dev_val_loss4"]),
                         "final_stop_val_loss4": float(final_row["inner_dev_val_loss4"])},
        "selected_checkpoint_metrics_recomputed": {"train": train_metric, "inner_dev": dev_metric},
        "final_stop_metrics_recomputed": {"train": final_train_metric, "inner_dev": final_dev_metric},
        "per_drug_mean_nmse_reference": {"train": baseline_train_nmse, "inner_dev": baseline_dev_nmse},
        "relative_nmse_reduction_vs_per_drug_mean": {"train": train_nmse_reduction, "inner_dev": dev_nmse_reduction},
        "fit_gate": gate,
        "maximum_absolute_metric_difference_from_curve": max_metric_diff,
        "forbidden_roles_opened": [],
    }
    out_path = ATTEMPT / "root_independent_attempt009_readback.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
