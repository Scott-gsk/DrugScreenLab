#!/usr/bin/env python3
"""Independent, read-only verification of the EXP-055 v12 TRAIN cost gate.

Purpose: verify the closed one-epoch TranSiGen diagnostic, exact keyed TRAIN
prediction/target alignment, independently recomputed TRAIN metrics, cost stop,
and resource settlement without opening any inner-dev numeric values.
Reuse barrier: the generic attempt-009 verifier is XPert-specific, while the
older TranSiGen verifier expects a nonfinite-diagnostic schema; neither checks
this finite-but-severely-miscalibrated epoch1 artifact and fixed-300 cost gate.
Retention: keep as the root readback entrypoint for this immutable v12 result.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20261007_exp055_transigen_trainpath_recovery_v12"
RUN_DIR = ROOT / "artifacts/experiments/EXP-055" / RUN_ID
ATTEMPT_DIR = RUN_DIR / "fit_attempt_001"
SOURCE_DIR = ROOT / "artifacts/experiments/EXP-051/20261006_exp051_contract_fit_smoke_v1"
RUNNER = ROOT / "scripts/modeling/exp051_contract_fit_smoke.py"
CONFIG = ROOT / "configs/experiments/EXP-055-representative-models.yaml"
OUTPUT = RUN_DIR / "root_independent_readback_v2.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def close_float(actual: Any, expected: Any, *, rel: float = 1e-7, abs_: float = 1e-10) -> bool:
    try:
        a, e = float(actual), float(expected)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and math.isfinite(e) and math.isclose(a, e, rel_tol=rel, abs_tol=abs_)


def metrics(keys: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    # Recompute the registered TRAIN-only gene variance denominator from targets.
    variance = np.var(truth.astype(np.float64), axis=0, ddof=0)
    use_gene = np.isfinite(variance) & (variance > 1e-12)
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    per_row_nmse = np.mean(np.square(error[:, use_gene]) / variance[use_gene][None, :], axis=1)
    per_row_mae = np.mean(np.abs(error), axis=1)
    row_metrics: list[dict[str, Any]] = []
    y = truth.astype(np.float64)
    p = prediction.astype(np.float64)
    for i in range(len(keys)):
        corr = None if np.ptp(y[i]) <= 0 or np.ptp(p[i]) <= 0 else float(np.corrcoef(y[i], p[i])[0, 1])
        nonzero = np.abs(y[i]) > 1e-12
        direction = None if not nonzero.any() else float(np.mean(np.sign(y[i, nonzero]) == np.sign(p[i, nonzero])))
        row_metrics.append({"object": str(keys.iloc[i]["source_object_id"]), "nmse": float(per_row_nmse[i]),
                            "mae": float(per_row_mae[i]), "correlation": corr, "direction": direction})
    frame = pd.DataFrame(row_metrics)
    per_object = frame.groupby("object", sort=True).mean(numeric_only=True)
    amplitudes: list[float] = []
    for obj, positions in keys.groupby("source_object_id", sort=True).indices.items():
        ix = np.asarray(positions, dtype=np.int64)
        observed = float(np.mean(np.abs(y[ix])))
        predicted = float(np.mean(np.abs(p[ix])))
        if observed > 0:
            amplitudes.append(predicted / observed)
    return {
        "n_conditions": int(len(keys)),
        "n_source_objects": int(keys.source_object_id.nunique()),
        "n_full_inchikeys": int(keys.full_inchikey.nunique()),
        "n_genes": int(truth.shape[1]),
        "n_genes_in_nmse_denominator": int(use_gene.sum()),
        "n_zero_variance_genes_excluded": int((~use_gene).sum()),
        "object_macro_nmse": float(per_object.nmse.mean()),
        "object_macro_mae": float(per_object.mae.mean()),
        "object_macro_gene_correlation": None if per_object.correlation.isna().all() else float(per_object.correlation.mean()),
        "object_macro_direction_agreement": None if per_object.direction.isna().all() else float(per_object.direction.mean()),
        "object_macro_amplitude_ratio": None if not amplitudes else float(np.mean(amplitudes)),
        "n_amplitude_objects_with_nonzero_observed_mean_abs": int(len(amplitudes)),
        "aggregation": "conditions averaged within source cell-line object, then objects equally weighted",
        "normalizer": "per-gene population variance recomputed from the 419 TRAIN target rows",
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace root readback: {OUTPUT}")
    preflight = load_json(RUN_DIR / "run_preflight_receipt.json")
    manifest = load_json(ATTEMPT_DIR / "fit_smoke_manifest.json")
    trace = load_json(ATTEMPT_DIR / "trainpath_epoch1_trace.json")
    cost = load_json(ATTEMPT_DIR / "trainpath_cost_gate.json")
    close = load_json(ATTEMPT_DIR / "attempt_close_receipt.json")
    progress = load_json(ATTEMPT_DIR / "attempt_progress.json")
    checkpoint_receipt = load_json(ATTEMPT_DIR / "last_finite_checkpoint_receipt.json")
    ledger = [json.loads(line) for line in (ATTEMPT_DIR / "resource_ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    npz_path = ATTEMPT_DIR / "TranSiGen_epoch1_train_keyed_predictions.npz"
    with np.load(npz_path, allow_pickle=False) as arrays:
        required = {"train_condition_key", "train_source_object_id", "train_full_inchikey", "genes", "train_target_delta", "train_delta"}
        if set(arrays.files) != required:
            raise ValueError(f"Unexpected keyed prediction archive fields: {arrays.files}")
        condition_key = arrays["train_condition_key"].astype(str)
        object_id = arrays["train_source_object_id"].astype(str)
        inchikey = arrays["train_full_inchikey"].astype(str)
        genes = arrays["genes"].astype(str)
        truth = np.asarray(arrays["train_target_delta"], dtype=np.float32)
        prediction = np.asarray(arrays["train_delta"], dtype=np.float32)

    metadata = pd.read_csv(SOURCE_DIR / "response_blind_conditions.tsv", sep="\t", dtype=str, keep_default_na=False)
    train_meta = metadata.loc[metadata["split_role"].eq("train"), ["full_condition_key", "source_object_id", "full_inchikey"]].copy()
    axis = pd.read_csv(SOURCE_DIR / "gene_axis_mapping.tsv", sep="\t", dtype=str, keep_default_na=False)
    axis["axis_index_numeric"] = pd.to_numeric(axis["axis_index"], errors="raise")
    expected_genes = axis.sort_values("axis_index_numeric", kind="stable")["h5ad_var_name"].astype(str).to_numpy()

    npz_keys = pd.DataFrame({"full_condition_key": condition_key, "source_object_id": object_id, "full_inchikey": inchikey})
    merged = npz_keys.merge(train_meta, on="full_condition_key", how="outer", suffixes=("_pred", "_meta"), indicator=True, validate="one_to_one")
    identity_ok = bool(
        len(merged) == 419 and merged["_merge"].eq("both").all()
        and merged["source_object_id_pred"].eq(merged["source_object_id_meta"]).all()
        and merged["full_inchikey_pred"].eq(merged["full_inchikey_meta"]).all()
        and npz_keys["full_condition_key"].nunique() == 419
    )
    train_keys = npz_keys.rename(columns={"full_condition_key": "condition_key"})
    checks = {
        "immutable_run_and_attempt_identity": manifest.get("run_id") == RUN_ID and manifest.get("fit_attempt") == "fit_attempt_001" and progress.get("completed_epochs") == 1,
        "preflight_and_current_code_hashes_match": (preflight.get("status") == "PASS_CPU_TRAIN_IDENTITY_BUDGET_AND_DEVICE_GATE_GPU_NOT_STARTED"
            and preflight.get("runner_sha256") == sha256(RUNNER) == manifest.get("current_runner_sha256")
            and preflight.get("config_sha256") == sha256(CONFIG) == manifest.get("config_sha256")
            and close.get("runner_sha256") == sha256(RUNNER)),
        "keyed_rows_match_frozen_response_blind_train_metadata": identity_ok,
        "gene_axis_matches_frozen_axis_metadata": len(genes) == 978 and len(np.unique(genes)) == 978 and np.array_equal(genes, expected_genes),
        "train_target_rows_match_registered_sha": (truth.shape == (419, 978) and np.isfinite(truth).all()
            and hashlib.sha256(truth.tobytes()).hexdigest() == preflight["reused_root_verified_train_input_gate"]["train_delta_sha256"]),
        "main_delta_is_finite_over_full_denominator": (prediction.shape == (419, 978) and np.isfinite(prediction).all()
            and int(prediction.size) == 409782 and int(trace["train_prediction_metrics"]["main_delta_nonfinite_n"]) == 0),
        "only_train_and_fixed_output_roles_used": (manifest["data_roles"]["inner_dev_numeric_rows_loaded"] is False
            and manifest["data_roles"]["inner_dev_used_for_model_selection"] is False
            and manifest["data_roles"]["outer_or_locked_numeric_values_opened"] is False
            and manifest["data_roles"]["pdo_numeric_sensitivity_or_expression_opened"] is False
            and cost["inner_dev_numeric_rows_loaded"] is False and close["inner_dev_numeric_rows_loaded"] is False),
        "batch_losses_and_gradients_finite_for_one_epoch": (len(trace["batch_loss_and_gradient_trace"]) == 7
            and all(math.isfinite(float(row["gradient_global_l2"])) and all(math.isfinite(float(x)) for x in row["losses"].values())
                    for row in trace["batch_loss_and_gradient_trace"])),
        "cost_projection_includes_required_300_schedule_and_margin_and_exceeds_cap": (cost["status"] == "STOP_AFTER_TRAIN_ONLY_DIAGNOSTIC_COST_OVER_CAP"
            and cost["projected_remaining_epochs"] == 299 and cost["fixed_epoch100_diagnostic_cost_included"] is True
            and cost["full_300_epoch_projection_including_20pct_margin_gpu_hours"] > cost["attempt_cap_gpu_hours"]),
        "closed_ledger_matches_owner_charge_and_reservation_release": (close["status"] == "CLOSED_AFTER_TRAIN_ONLY_TRACE_COST_GATE"
            and close["actual_gpu_hours"] == progress["actual_gpu_hours"]
            and close["released_unused_reservation_gpu_hours"] > 0
            and any(item.get("event") == "GPU_ATTEMPT_STARTED" for item in ledger)
            and any(item.get("event") == "GPU_ATTEMPT_CLOSED_AFTER_COST_GATE" for item in ledger)),
        "checkpoint_prediction_trace_hashes_match_close_receipt": (sha256(npz_path) == close["train_prediction_sha256"]
            and sha256(ATTEMPT_DIR / "trainpath_epoch1_trace.json") == close["trainpath_epoch1_trace_sha256"]
            and sha256(ATTEMPT_DIR / "TranSiGen_epoch_1_cost_gate_checkpoint.pt") == close["epoch1_cost_gate_checkpoint_sha256"]
            and checkpoint_receipt["optimizer_updates"] == 7 and checkpoint_receipt["status"] == "PASS_LAST_FINITE_COMPLETED_TRAIN_EPOCH_CHECKPOINT"),
    }
    recalculated = metrics(train_keys, truth, prediction)
    owner_metrics = trace["train_prediction_metrics"]
    metric_checks = {key: close_float(recalculated[key], owner_metrics.get(key), rel=2e-7, abs_=1e-8)
                     for key in ("object_macro_nmse", "object_macro_mae", "object_macro_gene_correlation",
                                 "object_macro_direction_agreement", "object_macro_amplitude_ratio")}
    checks["independent_object_macro_metrics_match_owner"] = all(metric_checks.values())

    baseline = trace["train_fit_gate"]
    cap = float(cost["attempt_cap_gpu_hours"])
    projected = float(cost["full_300_epoch_projection_including_20pct_margin_gpu_hours"])
    actual = float(close["actual_gpu_hours"])
    recovery_balance = cap - actual
    receipt = {
        "status": "ROOT_INDEPENDENT_V12_EPOCH1_READBACK_PASS" if all(checks.values()) else "ROOT_INDEPENDENT_V12_EPOCH1_READBACK_FAIL",
        "run_id": RUN_ID,
        "attempt_id": "fit_attempt_001",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "checks": checks,
        "metric_checks": metric_checks,
        "independent_train_metrics": recalculated,
        "owner_train_metrics": owner_metrics,
        "train_fit_gate": baseline,
        "cost_gate": {
            "actual_one_epoch_gpu_hours": actual,
            "recovery_cap_before_attempt_gpu_hours": cap,
            "remaining_recovery_cap_gpu_hours": recovery_balance,
            "projected_full_300_with_all_registered_evaluations_io_and_20pct_margin_gpu_hours": projected,
            "excess_over_remaining_cap_gpu_hours": projected - cap,
            "fixed_epoch100_diagnostic_included": cost["fixed_epoch100_diagnostic_cost_included"],
            "full_300_proceeds": False,
        },
        "input_hashes": {
            "runner": sha256(RUNNER), "config": sha256(CONFIG),
            "preflight": sha256(RUN_DIR / "run_preflight_receipt.json"),
            "manifest": sha256(ATTEMPT_DIR / "fit_smoke_manifest.json"),
            "prediction_npz": sha256(npz_path), "trace": sha256(ATTEMPT_DIR / "trainpath_epoch1_trace.json"),
            "cost_gate": sha256(ATTEMPT_DIR / "trainpath_cost_gate.json"),
            "close_receipt": sha256(ATTEMPT_DIR / "attempt_close_receipt.json"),
            "checkpoint": sha256(ATTEMPT_DIR / "TranSiGen_epoch_1_cost_gate_checkpoint.pt"),
        },
        "denominator": {"train_conditions": 419, "train_objects": 110, "full_inchikeys": 5,
                        "genes": 978, "prediction_elements": 409782, "prediction_nonfinite": 0,
                        "inner_dev_numeric_rows_opened": 0, "pdo_outer_locked_values_opened": False},
        "interpretation": "This is a finite but profoundly miscalibrated one-epoch TRAIN diagnostic, not a completed fit or transfer score. Its registered fixed-300 schedule including required evaluations, serialization and 20% margin projects above the remaining dedicated recovery cap, so the attempt correctly closed after epoch 1; no shorter schedule is authorized.",
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
