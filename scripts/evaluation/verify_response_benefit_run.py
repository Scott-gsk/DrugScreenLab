#!/usr/bin/env python
"""Independently verify the keyed outputs and patient-level metrics of EXP-067."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ARMS = (
    "pretreatment_rna_plus_structure",
    "pretreatment_rna_plus_structure_plus_crossfitted_predicted_response",
    "equal_capacity_direct_rff",
    "object_shuffled_crossfitted_response",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _recompute(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    y = frame["aac_published"].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for patient, positions in frame.groupby("source_object_id", sort=True).indices.items():
        ix = np.asarray(positions, dtype=np.int64)
        row: dict[str, Any] = {"source_object_id": str(patient), "n_pairs": len(ix)}
        for arm in ARMS:
            pred = frame[f"prediction__{arm}"].to_numpy(dtype=np.float64)
            error = pred[ix] - y[ix]
            row[f"{arm}__rmse"] = float(np.sqrt(np.mean(error * error)))
            rhos: list[float] = []
            agreements: list[float] = []
            local = frame.iloc[ix]
            for _, cell_positions in local.groupby("cellid", sort=True).indices.items():
                cell_ix = ix[np.asarray(cell_positions, dtype=np.int64)]
                if len(cell_ix) >= 2 and np.ptp(y[cell_ix]) > 0 and np.ptp(pred[cell_ix]) > 0:
                    rho = spearmanr(y[cell_ix], pred[cell_ix]).statistic
                    if np.isfinite(rho):
                        rhos.append(float(rho))
                if len(cell_ix) >= 2:
                    y_centered = y[cell_ix] - np.mean(y[cell_ix])
                    p_centered = pred[cell_ix] - np.mean(pred[cell_ix])
                    agreements.append(float(np.mean(np.sign(y_centered) == np.sign(p_centered))))
            row[f"{arm}__within_cell_drug_spearman"] = float(np.mean(rhos)) if rhos else None
            row[f"{arm}__within_cell_centered_directional_agreement"] = (
                float(np.mean(agreements)) if agreements else None
            )
            row[f"{arm}__spearman_cell_denominator"] = len(rhos)
        rows.append(row)
    objects = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    for arm in ARMS:
        rho = pd.to_numeric(objects[f"{arm}__within_cell_drug_spearman"], errors="coerce")
        direction = pd.to_numeric(objects[f"{arm}__within_cell_centered_directional_agreement"], errors="coerce")
        summary[arm] = {
            "patient_macro_rmse": float(objects[f"{arm}__rmse"].mean()),
            "patient_macro_rmse_denominator": int(objects[f"{arm}__rmse"].notna().sum()),
            "patient_macro_within_cell_drug_spearman": float(rho.mean()) if rho.notna().any() else None,
            "patient_spearman_denominator": int(rho.notna().sum()),
            "patient_macro_within_cell_centered_directional_agreement": float(direction.mean()) if direction.notna().any() else None,
            "patient_count": int(len(objects)), "pair_count": int(objects["n_pairs"].sum()),
        }
    return summary, objects


def _close(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return bool(np.isclose(float(a), float(b), rtol=tol, atol=tol))


def _paired_bootstrap(reference: np.ndarray, candidate: np.ndarray, seed: int,
                      iterations: int) -> dict[str, Any]:
    """Independent PatientID-paired percentile interval for reference minus candidate RMSE."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError("paired object RMSE arrays must be aligned one-dimensional vectors")
    valid = np.isfinite(reference) & np.isfinite(candidate)
    difference = reference[valid] - candidate[valid]
    if len(difference) < 2:
        return {"objects": int(len(difference)), "mean_absolute_rmse_reduction": None,
                "ci95": [None, None], "mean_relative_rmse_reduction": None}
    rng = np.random.default_rng(int(seed))
    sample_ids = rng.integers(0, len(difference), size=(int(iterations), len(difference)))
    absolute = difference[sample_ids].mean(axis=1)
    base = reference[valid]
    base_samples = base[sample_ids].mean(axis=1)
    relative = absolute / base_samples if np.all(base_samples > 0) else np.full(int(iterations), np.nan)
    return {
        "objects": int(len(difference)),
        "mean_absolute_rmse_reduction": float(difference.mean()),
        "ci95": [float(np.quantile(absolute, .025)), float(np.quantile(absolute, .975))],
        "mean_relative_rmse_reduction": float(difference.mean() / base.mean()) if base.mean() > 0 else None,
        "relative_ci95": [float(np.nanquantile(relative, .025)), float(np.nanquantile(relative, .975))]
        if np.isfinite(relative).any() else [None, None],
        "bootstrap_iterations": int(iterations),
        "bootstrap_unit": "PatientID, paired within object; not patient-population inference",
    }


def verify(run_dir: Path, expected_receipt_sha256: str) -> dict[str, Any]:
    summary_path = run_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("experiment_id") != "EXP-067" or summary.get("metadata_gate_receipt_sha256") != expected_receipt_sha256:
        raise ValueError("EXP-067 summary has the wrong experiment or frozen metadata receipt")
    if summary.get("status") not in {"COMPLETE_CANDIDATE_PASS", "COMPLETE_CANDIDATE_CLOSED"}:
        raise ValueError("Only a complete five-fold run can receive a scientific readback PASS")
    if summary.get("outer_folds_completed") != 5 or summary.get("pair_count", 0) < 1:
        raise ValueError("EXP-067 five-fold or pair denominator is incomplete")
    frame = pd.read_csv(run_dir / "oof_predictions.tsv", sep="\t", dtype={
        "pair_id": "string", "source_object_id": "string", "cellid": "string",
        "full_inchikey": "string",
    })
    required = {"pair_id", "source_object_id", "cellid", "full_inchikey", "fold_id", "aac_published"}
    required |= {f"prediction__{arm}" for arm in ARMS}
    if required - set(frame.columns):
        raise ValueError(f"OOF table lacks required fields: {sorted(required-set(frame.columns))}")
    if frame["pair_id"].isna().any() or frame["pair_id"].duplicated().any():
        raise ValueError("OOF pair keys are missing or duplicated")
    if frame[["source_object_id", "cellid", "full_inchikey"]].isna().any().any():
        raise ValueError("OOF identity or complete drug key contains missing values")
    if not frame["full_inchikey"].astype(str).str.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]").all():
        raise ValueError("OOF output contains an incomplete InChIKey")
    if set(pd.to_numeric(frame["fold_id"], errors="raise").astype(int).unique()) != set(range(5)):
        raise ValueError("OOF output does not cover five fixed outer folds")
    object_folds = frame[["source_object_id", "fold_id"]].drop_duplicates()
    if object_folds["source_object_id"].duplicated().any():
        raise ValueError("PatientID is split across outer folds")
    numeric = frame[["aac_published", *[f"prediction__{arm}" for arm in ARMS]]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("the common four-arm OOF domain includes nonfinite labels or predictions")
    if int(len(frame)) != int(summary["pair_count"]):
        raise ValueError("OOF row denominator differs from the run summary")
    if int(frame["source_object_id"].nunique()) != int(summary["object_count"]):
        raise ValueError("OOF PatientID denominator differs from the run summary")
    if int(frame["full_inchikey"].nunique()) != int(summary["drug_count"]):
        raise ValueError("OOF full-key denominator differs from the run summary")
    by_fold = frame.groupby("fold_id")["pair_id"].apply(lambda s: set(s.astype(str)))
    common_by_pair = frame.set_index("pair_id", drop=False)
    fold_by_patient = frame[["source_object_id", "fold_id"]].drop_duplicates().set_index(
        "source_object_id")["fold_id"].astype(int).to_dict()
    for fold in range(5):
        fold_path = run_dir / f"fold_{fold}_pair_predictions.tsv"
        if not fold_path.is_file():
            raise FileNotFoundError(f"missing outer-fold keyed prediction table: {fold_path.name}")
        fold_frame = pd.read_csv(fold_path, sep="\t", dtype={"pair_id": "string"})
        if set(fold_frame["pair_id"].astype(str)) != by_fold.loc[fold]:
            raise ValueError(f"fold {fold} rows do not match its combined OOF keys")
        fold_frame["pair_id"] = fold_frame["pair_id"].astype(str)
        expected_fold_rows = common_by_pair.loc[fold_frame["pair_id"].tolist()].reset_index(drop=True)
        for identity_column in ("source_object_id", "cellid", "full_inchikey", "fold_id"):
            if identity_column not in fold_frame or fold_frame[identity_column].astype(str).tolist() != expected_fold_rows[identity_column].astype(str).tolist():
                raise ValueError(f"fold {fold} identity or key differs from combined OOF for {identity_column}")
        for output_column in ("aac_published", *[f"prediction__{arm}" for arm in ARMS]):
            if output_column not in fold_frame or not np.allclose(
                fold_frame[output_column].to_numpy(dtype=np.float64),
                expected_fold_rows[output_column].to_numpy(dtype=np.float64), rtol=0, atol=1e-12,
            ):
                raise ValueError(f"fold {fold} prediction/label values differ from the combined keyed OOF table")
        fold_summary = json.loads((run_dir / f"fold_{fold}_summary.json").read_text(encoding="utf-8"))
        nested = fold_summary.get("fold", {}).get("nested_response_receipt", {})
        if int(nested.get("outer_fold_id", -1)) != fold:
            raise ValueError(f"fold {fold} nested receipt has a mismatched outer-fold ID")
        outer_train = set(map(str, nested.get("outer_train_patient_ids", [])))
        outer_predict = set(map(str, nested.get("outer_predict_patient_ids", [])))
        expected_predict = {p for p, assigned in fold_by_patient.items() if assigned == fold}
        expected_train = set(fold_by_patient) - expected_predict
        if outer_train != expected_train or outer_predict != expected_predict:
            raise ValueError(f"fold {fold} nested receipt does not match fixed outer PatientID assignment")
        inner = nested.get("inner_fold_provenance", [])
        if {int(item.get("inner_validation_fold_id", -1)) for item in inner} != set(range(5)) - {fold}:
            raise ValueError(f"fold {fold} nested receipt is missing one or more fixed inner folds")
        covered_train: set[str] = set()
        for item in inner:
            inner_fold = int(item["inner_validation_fold_id"])
            inner_predict = set(map(str, item.get("prediction_patient_ids", [])))
            inner_train = set(map(str, item.get("training_patient_ids", [])))
            expected_inner_predict = {p for p in expected_train if fold_by_patient[p] == inner_fold}
            if inner_predict != expected_inner_predict or inner_train != expected_train - expected_inner_predict:
                raise ValueError(f"fold {fold}/inner {inner_fold} response training IDs leak or differ from frozen folds")
            if inner_train & (inner_predict | expected_predict) or covered_train & inner_predict:
                raise ValueError(f"fold {fold}/inner {inner_fold} response training/validation IDs overlap")
            covered_train |= inner_predict
        if covered_train != expected_train:
            raise ValueError(f"fold {fold} inner response predictions do not partition outer TRAIN objects")
        outer_model = nested.get("outer_model_provenance", {})
        if (set(map(str, outer_model.get("training_patient_ids", []))) != expected_train
                or set(map(str, outer_model.get("prediction_patient_ids", []))) != expected_predict):
            raise ValueError(f"fold {fold} outer response model was fit on an incorrect PatientID set")
        donor_sets = []
        for part_key in ("train_shuffle_donor_ids", "heldout_shuffle_donor_ids"):
            records = fold_summary["fold"][part_key]
            for item in records:
                if item["source_object_id"] == item["shuffle_donor_object_id"]:
                    raise ValueError(f"fold {fold} shuffle arm maps a PatientID to itself")
                donor_fold = fold_by_patient.get(str(item["shuffle_donor_object_id"]))
                if donor_fold is None or donor_fold != int(item["nested_fold_id"]):
                    raise ValueError(f"fold {fold} shuffle donor crosses its frozen PatientID fold")
            donor_sets.append(records)
    recomputed, object_metrics = _recompute(frame)
    for arm in ARMS:
        for metric, value in recomputed[arm].items():
            if metric == "patient_count" or metric == "pair_count" or metric.endswith("denominator"):
                if int(summary["overall_metrics"][arm][metric]) != int(value):
                    raise ValueError(f"summary denominator differs for {arm}/{metric}")
            elif not _close(summary["overall_metrics"][arm][metric], value):
                raise ValueError(f"independent OOF metric differs for {arm}/{metric}")
    object_path = run_dir / "oof_object_metrics.tsv"
    object_metrics.to_csv(run_dir / "root_recomputed_object_metrics.tsv", sep="\t", index=False, lineterminator="\n")
    # The fixed gate is recomputed from the independently reconstructed object scores.
    direct = object_metrics[f"{ARMS[0]}__rmse"].to_numpy(dtype=float)
    candidate = object_metrics[f"{ARMS[1]}__rmse"].to_numpy(dtype=float)
    capacity = object_metrics[f"{ARMS[2]}__rmse"].to_numpy(dtype=float)
    shuffled = object_metrics[f"{ARMS[3]}__rmse"].to_numpy(dtype=float)
    direct_rel = (direct.mean() - candidate.mean()) / direct.mean()
    capacity_rel = (capacity.mean() - candidate.mean()) / capacity.mean()
    improved_direct = float(np.mean(candidate < direct))
    improved_capacity = float(np.mean(candidate < capacity))
    bootstrap = summary.get("bootstrap_protocol", {})
    seed, iterations = int(bootstrap.get("seed", -1)), int(bootstrap.get("iterations", -1))
    if seed != 670067 or iterations != 2000:
        raise ValueError("EXP-067 paired bootstrap does not match the frozen seed/iteration contract")
    paired_readback: dict[str, Any] = {}
    positive_ci = True
    for reference_arm, reference_values in ((ARMS[0], direct), (ARMS[2], capacity)):
        name = f"{ARMS[1]}_vs_{reference_arm}"
        expected_ci = _paired_bootstrap(reference_values, candidate, seed, iterations)
        stored_ci = summary.get("paired_patient_bootstrap_vs_direct", {}).get(name)
        if not isinstance(stored_ci, dict):
            raise ValueError(f"run summary is missing paired patient interval {name}")
        for metric, value in expected_ci.items():
            if isinstance(value, list):
                actual = stored_ci.get(metric)
                if not isinstance(actual, list) or len(actual) != len(value) or any(
                    not _close(a, b) for a, b in zip(actual, value)
                ):
                    raise ValueError(f"stored paired interval differs from independent bootstrap for {name}/{metric}")
            elif isinstance(value, (int, float)):
                if not _close(stored_ci.get(metric), value):
                    raise ValueError(f"stored paired interval differs from independent bootstrap for {name}/{metric}")
        positive_ci &= expected_ci["ci95"][0] is not None and expected_ci["ci95"][0] > 0
        paired_readback[name] = expected_ci
    gate_pass = (direct_rel >= .05 and capacity_rel >= .05 and positive_ci
                 and improved_direct >= .60 and improved_capacity >= .60
                 and candidate.mean() < shuffled.mean())
    if bool(summary["practical_gate_pass"]) != bool(gate_pass):
        raise ValueError("candidate closure decision differs from independent fixed-gate recomputation")
    receipt = {
        "schema": "exp067_root_independent_readback_v1",
        "status": "PASS_ROOT_INDEPENDENT_EXP067_NUMERIC_READBACK",
        "run_id": summary["run_id"], "metadata_gate_receipt_sha256": expected_receipt_sha256,
        "run_summary_sha256": _sha256(summary_path),
        "oof_predictions_sha256": _sha256(run_dir / "oof_predictions.tsv"),
        "fold_prediction_sha256": {str(i): _sha256(run_dir / f"fold_{i}_pair_predictions.tsv") for i in range(5)},
        "pair_count": len(frame), "object_count": int(frame["source_object_id"].nunique()),
        "full_inchikey_count": int(frame["full_inchikey"].nunique()),
        "gene_axis_count": 978, "all_arms_common_finite_domain": True,
        "independently_recomputed_patient_macro_metrics": recomputed,
        "independent_gate": {
            "relative_rmse_improvement_vs_direct": float(direct_rel),
            "relative_rmse_improvement_vs_capacity_match": float(capacity_rel),
            "fraction_objects_improved_vs_direct": improved_direct,
            "fraction_objects_improved_vs_capacity_match": improved_capacity,
            "candidate_rmse_below_shuffled": bool(candidate.mean() < shuffled.mean()),
            "positive_paired_object_ci_vs_direct_and_capacity_match": bool(positive_ci),
            "paired_bootstrap": paired_readback,
            "pass": bool(gate_pass),
        },
        "root_recomputed_object_metrics_sha256": _sha256(run_dir / "root_recomputed_object_metrics.tsv"),
        "source_roles_read": ["frozen development TRAIN+DEV OOF outputs only"],
        "forbidden_data_read": [],
    }
    receipt_path = run_dir / "root_independent_readback.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--metadata-receipt-sha256", required=True)
    args = parser.parse_args()
    receipt = verify(args.run_dir.resolve(), args.metadata_receipt_sha256)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
