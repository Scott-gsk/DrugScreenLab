"""Source-only reconstruction of LINCS 24-hour control profiles from DepMap RNA-seq.

This EXP-050 development analysis does not read PDO expression, perturbed LINCS
responses, direct drug labels, outer-test data, or locked-evaluation values.
The independent unit is the frozen DepMap PatientID source origin. Technical
LINCS control instances are averaged within control group, control groups are
averaged within cell line, and cell lines sharing one verified PatientID are
averaged to one paired source object.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
INTAKE = ROOT / "artifacts/experiments/EXP-007/CCLE_RNASEQ_INTAKE.json"
EXPECTED_TRAIN_OBJECTS = 13
EXPECTED_VAL_OBJECTS = 2
EXPECTED_GENES = 978


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def safe_corr(function: Any, left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    value = float(function(left, right).statistic)
    return value if np.isfinite(value) else None


def expression_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    train_sd: np.ndarray,
) -> dict[str, float | int | None]:
    valid = np.isfinite(truth) & np.isfinite(prediction) & np.isfinite(train_sd) & (train_sd > 0)
    if not np.any(valid):
        raise ValueError("no genes have a finite nonzero training-fold LINCS control SD")
    residual = prediction - truth
    scaled_mse = float(np.mean(np.square(residual[valid] / train_sd[valid])))
    return {
        "n_genes_scaled_mse": int(valid.sum()),
        "n_genes_excluded_zero_or_nonfinite_train_sd": int((~valid).sum()),
        "scaled_mse": scaled_mse,
        "profile_pearson": safe_corr(pearsonr, truth, prediction),
        "profile_spearman": safe_corr(spearmanr, truth, prediction),
        "profile_mae": float(np.mean(np.abs(residual))),
        "profile_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "profile_mean_bias_prediction_minus_truth": float(np.mean(residual)),
    }


def fit_candidate_1(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centered_source = source - source_mean
    variance = np.mean(np.square(centered_source), axis=0)
    covariance = np.mean(centered_source * (target - target_mean), axis=0)
    valid = variance > 0
    slope = np.zeros(target.shape[1], dtype=np.float64)
    slope[valid] = covariance[valid] / variance[valid]
    intercept = target_mean - slope * source_mean
    return intercept, slope, int((~valid).sum())


def predict_candidate_1(source: np.ndarray, intercept: np.ndarray, slope: np.ndarray) -> np.ndarray:
    return intercept + slope * source


def fit_candidate_2(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    source_median = np.median(source, axis=0)
    target_median = np.median(target, axis=0)
    source_mad = np.median(np.abs(source - source_median), axis=0)
    target_mad = np.median(np.abs(target - target_median), axis=0)
    scale = np.zeros(target.shape[1], dtype=np.float64)
    valid = (source_mad > 0) & (target_mad > 0)
    scale[valid] = target_mad[valid] / source_mad[valid]
    # Zero-MAD genes map to the training-fold target median by preregistration.
    return source_median, source_mad, target_median, scale, int((~valid).sum())


def predict_candidate_2(
    source: np.ndarray,
    source_median: np.ndarray,
    target_median: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return target_median + (source - source_median) * scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/EXP-050.yaml"))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol = config["source_rnaseq_to_lincs_scale_reconstruction"]
    if args.run_id != protocol["run_id"]:
        raise ValueError(f"run id mismatch: expected {protocol['run_id']}, got {args.run_id}")
    if protocol["status"] != "preregistered_pending_execution":
        raise ValueError(f"unexpected preregistration state: {protocol['status']}")

    output = ROOT / protocol["outputs"]["directory"]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {output}")
    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite incomplete staging directory: {staging}")
    started_at = datetime.now(timezone.utc)

    depmap_matrix_path = ROOT / protocol["inputs"]["depmap_expression"]
    depmap_mapping_path = ROOT / protocol["inputs"]["depmap_mapping"]
    identity_path = ROOT / protocol["inputs"]["depmap_identity"]
    controls_path = ROOT / protocol["inputs"]["lincs_controls"]
    lincs_matrix_path = ROOT / protocol["inputs"]["lincs_control_matrix"]
    genes_path = ROOT / protocol["inputs"]["lincs_gene_axis_contract"]
    input_paths = {
        "protocol_config": config_path,
        "depmap_expression": depmap_matrix_path,
        "depmap_mapping": depmap_mapping_path,
        "depmap_intake_audit": INTAKE,
        "frozen_identity_split": identity_path,
        "lincs_control_metadata": controls_path,
        "lincs_control_matrix": lincs_matrix_path,
        "lincs_gene_axis": genes_path,
    }
    for name, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing registered input {name}: {path}")

    # Verify the frozen gene contract before opening either expression matrix.
    genes = [str(value).strip() for value in json.loads(genes_path.read_text(encoding="utf-8"))]
    if len(genes) != EXPECTED_GENES or len(set(genes)) != EXPECTED_GENES:
        raise ValueError("LINCS exact978 gene axis must contain 978 unique ordered identifiers")
    intake = json.loads(INTAKE.read_text(encoding="utf-8"))
    expected_axis_hash = intake["gene_harmonization"]["ordered_gene_ids_sha256"]
    actual_axis_hash = hashlib.sha256(("\n".join(genes) + "\n").encode("utf-8")).hexdigest()
    if actual_axis_hash != expected_axis_hash:
        raise ValueError("LINCS genes.json order differs from the verified DepMap exact978 intake axis")

    # Resolve exact identities and frozen roles using metadata only. locked_eval
    # identities are dropped before any cache_row is selected or expression is read.
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("response_blind") is not True:
        raise ValueError("EXP-007 identity/split file is not marked response-blind")
    assignments = identity.get("split_assignments", [])
    assignment_by_id: dict[str, dict[str, Any]] = {}
    for row in assignments:
        depmap_id = str(row["depmap_id"])
        if depmap_id in assignment_by_id:
            raise ValueError(f"duplicate frozen split assignment for {depmap_id}")
        assignment_by_id[depmap_id] = row

    identity_rows = identity.get("identity_rows", [])
    frozen_joined = []
    roles_by_patient: dict[str, set[str]] = {}
    for row in identity_rows:
        if not row.get("expression_present") or row.get("exact978_row") is None:
            continue
        depmap_id = str(row["depmap_id"])
        assignment = assignment_by_id.get(depmap_id)
        if assignment is None:
            continue
        role = str(assignment["split_role"])
        patient_id = str(row.get("patient_id", "")).strip()
        if not patient_id:
            raise ValueError(f"missing PatientID source-origin key for exact identity {depmap_id}")
        roles_by_patient.setdefault(patient_id, set()).add(role)
        frozen_joined.append({**row, "split_role": role})
    cross_role_patients = {key: sorted(value) for key, value in roles_by_patient.items() if len(value) > 1}
    if cross_role_patients:
        raise ValueError(f"frozen split leaks shared PatientID source origins across roles: {cross_role_patients}")

    train_val_rows = [row for row in frozen_joined if row["split_role"] in {"train", "val"}]
    if any(row["split_role"] == "locked_eval" for row in train_val_rows):
        raise AssertionError("locked_eval identity unexpectedly entered training/validation scope")
    mapping_data = json.loads(depmap_mapping_path.read_text(encoding="utf-8"))
    mapping_by_id = {str(row["depmap_id"]): row for row in mapping_data}
    if len(mapping_by_id) != len(mapping_data):
        raise ValueError("DepMap exact978 mapping contains duplicate DepMap model IDs")

    exact_records = []
    for row in train_val_rows:
        depmap_id = str(row["depmap_id"])
        exact_name = str(row.get("stripped_cell_line_name", "")).strip().upper()
        lincs_name = str(row.get("lincs_cell_id", "")).strip().upper()
        mapping_row = mapping_by_id.get(depmap_id)
        if not exact_name or exact_name != lincs_name:
            continue
        if row.get("join_status") != "EXACT_STRIPPED_NAME_WITH_RNASEQ":
            continue
        if row.get("alias_note") not in (None, "") or row.get("alias_depmap_id") not in (None, ""):
            continue
        if mapping_row is None:
            raise ValueError(f"exact DepMap identity lacks a matrix row: {depmap_id}")
        if int(mapping_row["exact978_row"]) != int(row["exact978_row"]):
            raise ValueError(f"identity and RNA-seq matrix row disagree for {depmap_id}")
        if str(mapping_row["stripped_cell_line_name"]).strip().upper() != exact_name:
            raise ValueError(f"identity and matrix cell-line name disagree for {depmap_id}")
        exact_records.append({**row, "depmap_id": depmap_id, "cell_name": exact_name,
                              "lincs_cell_id": lincs_name,
                              "exact978_row": int(mapping_row["exact978_row"])})
    if len({row["lincs_cell_id"] for row in exact_records}) != len(exact_records):
        raise ValueError("one LINCS cell identity maps to more than one included DepMap row")

    controls_columns = ["inst_id", "cache_row", "control_group_id", "cell_id",
                        "pert_time", "pert_time_unit", "control_finite"]
    controls = pd.read_parquet(controls_path, columns=controls_columns)
    controls["cell_id_norm"] = controls["cell_id"].astype(str).str.strip().str.upper()
    exact_cell_ids = {row["lincs_cell_id"] for row in exact_records}
    all_finite_24h = controls.loc[
        controls["pert_time"].astype(str).eq("24")
        & controls["pert_time_unit"].astype(str).str.lower().eq("h")
        & controls["control_finite"].astype(bool)
    ].copy()
    finite_24h = all_finite_24h.loc[all_finite_24h["cell_id_norm"].isin(exact_cell_ids)].copy()
    if not finite_24h.empty:
        group_identity = finite_24h.groupby("control_group_id", sort=False).agg(
            cells=("cell_id_norm", "nunique"), times=("pert_time", "nunique"),
            units=("pert_time_unit", "nunique"))
        if (group_identity[["cells", "times", "units"]] != 1).any().any():
            raise ValueError("control_group_id crosses cell, time, or unit identity")

    rows_by_cell = {
        str(cell): frame.copy()
        for cell, frame in finite_24h.groupby("cell_id_norm", sort=True)
    }
    records_with_controls = [row for row in exact_records if row["lincs_cell_id"] in rows_by_cell]
    grouped_by_patient: dict[str, list[dict[str, Any]]] = {}
    for row in records_with_controls:
        grouped_by_patient.setdefault(str(row["patient_id"]), []).append(row)

    def role_for_patient(rows: list[dict[str, Any]]) -> str:
        roles = {str(row["split_role"]) for row in rows}
        if len(roles) != 1:
            raise ValueError(f"source object is represented in multiple split roles: {roles}")
        return roles.pop()

    # Audit exact 24-hour coverage before opening any numerical matrix values.
    object_role_counts: dict[str, int] = {}
    for patient_id, rows in grouped_by_patient.items():
        role = role_for_patient(rows)
        object_role_counts[role] = object_role_counts.get(role, 0) + 1
    n_train = object_role_counts.get("train", 0)
    n_val = object_role_counts.get("val", 0)
    if n_train != EXPECTED_TRAIN_OBJECTS or n_val != EXPECTED_VAL_OBJECTS:
        raise ValueError(
            "pre-registered exact 24-hour coverage mismatch before numerical read: "
            f"expected train/val={EXPECTED_TRAIN_OBJECTS}/{EXPECTED_VAL_OBJECTS}, got {n_train}/{n_val}"
        )

    # Ensure the two cohorts are distinct independent PatientID origins.
    train_patient_ids = sorted(pid for pid, rows in grouped_by_patient.items()
                               if role_for_patient(rows) == "train")
    val_patient_ids = sorted(pid for pid, rows in grouped_by_patient.items()
                             if role_for_patient(rows) == "val")
    if set(train_patient_ids) & set(val_patient_ids):
        raise ValueError("train and validation share a PatientID source origin")

    cache_rows_by_cell: dict[str, pd.DataFrame] = rows_by_cell
    source_objects: list[dict[str, Any]] = []
    crosswalk_rows: list[dict[str, Any]] = []

    # Only after identity, role, group, 13/2 coverage, and gene-axis gates pass
    # are train/validation control and DepMap expression rows read from arrays.
    lincs_cache = np.load(lincs_matrix_path, mmap_mode="r", allow_pickle=False)
    depmap_matrix = np.load(depmap_matrix_path, mmap_mode="r", allow_pickle=False)
    if lincs_cache.ndim != 2 or lincs_cache.shape[1] != EXPECTED_GENES:
        raise ValueError(f"unexpected LINCS cache shape: {lincs_cache.shape}")
    if depmap_matrix.ndim != 2 or depmap_matrix.shape[1] != EXPECTED_GENES:
        raise ValueError(f"unexpected DepMap exact978 matrix shape: {depmap_matrix.shape}")
    if len(mapping_data) != depmap_matrix.shape[0]:
        raise ValueError("DepMap row mapping count does not match RNA-seq matrix rows")

    depmap_profiles: dict[str, np.ndarray] = {}
    lincs_profiles: dict[str, np.ndarray] = {}
    for patient_id in sorted(grouped_by_patient):
        lines = sorted(grouped_by_patient[patient_id], key=lambda x: x["lincs_cell_id"])
        role = role_for_patient(lines)
        line_depmap_profiles: list[np.ndarray] = []
        line_lincs_profiles: list[np.ndarray] = []
        for row in lines:
            index = int(row["exact978_row"])
            x_line = np.asarray(depmap_matrix[index, :], dtype=np.float64)
            if x_line.shape != (EXPECTED_GENES,) or not np.isfinite(x_line).all():
                raise ValueError(f"nonfinite or malformed DepMap source profile: {row['depmap_id']}")

            line_controls = cache_rows_by_cell[row["lincs_cell_id"]]
            group_vectors: list[np.ndarray] = []
            group_count = 0
            instance_count = 0
            for _, group in line_controls.groupby("control_group_id", sort=True):
                cache_rows = group["cache_row"].to_numpy(dtype=np.int64)
                if cache_rows.size == 0 or cache_rows.min() < 0 or cache_rows.max() >= lincs_cache.shape[0]:
                    raise ValueError(f"LINCS control cache index out of bounds for {row['lincs_cell_id']}")
                technical = np.asarray(lincs_cache[cache_rows, :], dtype=np.float64)
                if not np.isfinite(technical).all():
                    raise ValueError(
                        f"control_finite metadata disagrees with cache values in group {group['control_group_id'].iloc[0]}"
                    )
                group_vectors.append(technical.mean(axis=0))
                group_count += 1
                instance_count += int(cache_rows.size)
            if not group_vectors:
                raise ValueError(f"no finite 24-hour control groups for {row['lincs_cell_id']}")
            y_line = np.stack(group_vectors, axis=0).mean(axis=0)
            if not np.isfinite(y_line).all():
                raise ValueError(f"nonfinite LINCS control profile for {row['lincs_cell_id']}")
            line_depmap_profiles.append(x_line)
            line_lincs_profiles.append(y_line)
            crosswalk_rows.append({
                "patient_id_source_origin": patient_id,
                "split_role": role,
                "depmap_id": row["depmap_id"],
                "cell_line_name": row["cell_name"],
                "lincs_cell_id": row["lincs_cell_id"],
                "exact978_row": index,
                "n_finite_24h_control_groups": group_count,
                "n_finite_24h_control_instances": instance_count,
                "aggregation": "technical_instances_to_control_group_then_equal_control_groups_to_cell_line",
            })
        source_objects.append({
            "patient_id": patient_id,
            "split_role": role,
            "cell_lines": [line["cell_name"] for line in lines],
            "source": np.stack(line_depmap_profiles, axis=0).mean(axis=0),
            "target": np.stack(line_lincs_profiles, axis=0).mean(axis=0),
        })

    source_objects.sort(key=lambda item: (item["split_role"], item["patient_id"]))
    training = [item for item in source_objects if item["split_role"] == "train"]
    validation = [item for item in source_objects if item["split_role"] == "val"]
    if len(training) != EXPECTED_TRAIN_OBJECTS or len(validation) != EXPECTED_VAL_OBJECTS:
        raise AssertionError("source-object count changed during numeric aggregation")

    X_train = np.stack([item["source"] for item in training], axis=0)
    Y_train = np.stack([item["target"] for item in training], axis=0)
    X_val = np.stack([item["source"] for item in validation], axis=0)
    Y_val = np.stack([item["target"] for item in validation], axis=0)

    # Frozen 13-object leave-one-source-origin-out comparison.
    cv_records: list[dict[str, Any]] = []
    for heldout in range(len(training)):
        fit_indices = np.asarray([index for index in range(len(training)) if index != heldout], dtype=np.int64)
        x_fit, y_fit = X_train[fit_indices], Y_train[fit_indices]
        x_test, y_test = X_train[heldout], Y_train[heldout]
        sd = y_fit.std(axis=0, ddof=1)
        baseline = np.median(y_fit, axis=0)
        baseline_metrics = expression_metrics(y_test, baseline, sd)
        intercept, slope, affine_zero_variance = fit_candidate_1(x_fit, y_fit)
        affine_pred = predict_candidate_1(x_test, intercept, slope)
        affine_metrics = expression_metrics(y_test, affine_pred, sd)
        source_median, source_mad, target_median, robust_scale, robust_fallback = fit_candidate_2(x_fit, y_fit)
        robust_pred = predict_candidate_2(x_test, source_median, target_median, robust_scale)
        robust_metrics = expression_metrics(y_test, robust_pred, sd)
        cv_records.append({
            "heldout_patient_id_source_origin": training[heldout]["patient_id"],
            "heldout_cell_lines": ";".join(training[heldout]["cell_lines"]),
            "n_train_objects_in_fold": int(len(fit_indices)),
            "n_genes_excluded_zero_or_nonfinite_train_sd": baseline_metrics["n_genes_excluded_zero_or_nonfinite_train_sd"],
            "baseline_train_median_scaled_mse": baseline_metrics["scaled_mse"],
            "candidate_1_affine_scaled_mse": affine_metrics["scaled_mse"],
            "candidate_1_delta_vs_baseline": float(affine_metrics["scaled_mse"] - baseline_metrics["scaled_mse"]),
            "candidate_1_profile_pearson": affine_metrics["profile_pearson"],
            "candidate_1_profile_spearman": affine_metrics["profile_spearman"],
            "candidate_1_profile_mae": affine_metrics["profile_mae"],
            "candidate_1_profile_rmse": affine_metrics["profile_rmse"],
            "candidate_1_profile_mean_bias": affine_metrics["profile_mean_bias_prediction_minus_truth"],
            "candidate_1_zero_source_variance_genes": affine_zero_variance,
            "candidate_2_robust_location_scale_scaled_mse": robust_metrics["scaled_mse"],
            "candidate_2_delta_vs_baseline": float(robust_metrics["scaled_mse"] - baseline_metrics["scaled_mse"]),
            "candidate_2_profile_pearson": robust_metrics["profile_pearson"],
            "candidate_2_profile_spearman": robust_metrics["profile_spearman"],
            "candidate_2_profile_mae": robust_metrics["profile_mae"],
            "candidate_2_profile_rmse": robust_metrics["profile_rmse"],
            "candidate_2_profile_mean_bias": robust_metrics["profile_mean_bias_prediction_minus_truth"],
            "candidate_2_zero_source_or_target_mad_genes": robust_fallback,
        })

    cv_frame = pd.DataFrame(cv_records)
    baseline_mean = float(cv_frame["baseline_train_median_scaled_mse"].mean())
    candidate_means = {
        "candidate_1_affine": float(cv_frame["candidate_1_affine_scaled_mse"].mean()),
        "candidate_2_robust_location_scale": float(cv_frame["candidate_2_robust_location_scale_scaled_mse"].mean()),
    }
    eligible_candidates = [(name, score) for name, score in candidate_means.items() if score < baseline_mean]
    eligible_candidates.sort(key=lambda item: (item[1], item[0]))
    selected_candidate = eligible_candidates[0][0] if eligible_candidates else None

    validation_records: list[dict[str, Any]] = []
    params: dict[str, np.ndarray] = {
        "gene_ids": np.asarray(genes, dtype="U16"),
        "selected_candidate": np.asarray(selected_candidate or "none", dtype="U64"),
        "training_median_baseline": np.median(Y_train, axis=0).astype(np.float64),
    }
    selected_summary: dict[str, Any] | None = None
    if selected_candidate == "candidate_1_affine":
        intercept, slope, affine_zero_variance = fit_candidate_1(X_train, Y_train)
        params["candidate_1_intercept"] = intercept
        params["candidate_1_slope"] = slope
        params["candidate_1_zero_source_variance_genes"] = np.asarray(affine_zero_variance, dtype=np.int64)
        predict = lambda x: predict_candidate_1(x, intercept, slope)
        selected_summary = {"zero_source_variance_genes": affine_zero_variance}
    elif selected_candidate == "candidate_2_robust_location_scale":
        source_median, source_mad, target_median, robust_scale, robust_fallback = fit_candidate_2(X_train, Y_train)
        params["candidate_2_source_median"] = source_median
        params["candidate_2_source_mad"] = source_mad
        params["candidate_2_target_median"] = target_median
        params["candidate_2_scale"] = robust_scale
        params["candidate_2_zero_source_or_target_mad_genes"] = np.asarray(robust_fallback, dtype=np.int64)
        predict = lambda x: predict_candidate_2(x, source_median, target_median, robust_scale)
        selected_summary = {"zero_source_or_target_mad_genes": robust_fallback}

    if selected_candidate:
        train_sd = Y_train.std(axis=0, ddof=1)
        full_train_baseline = np.median(Y_train, axis=0)
        for index, item in enumerate(validation):
            baseline_metrics = expression_metrics(Y_val[index], full_train_baseline, train_sd)
            candidate_metrics = expression_metrics(Y_val[index], predict(X_val[index]), train_sd)
            validation_records.append({
                "patient_id_source_origin": item["patient_id"],
                "cell_lines": ";".join(item["cell_lines"]),
                "selected_candidate": selected_candidate,
                "n_training_source_objects": int(len(training)),
                "n_genes_excluded_zero_or_nonfinite_train_sd": candidate_metrics["n_genes_excluded_zero_or_nonfinite_train_sd"],
                "baseline_scaled_mse": baseline_metrics["scaled_mse"],
                "candidate_scaled_mse": candidate_metrics["scaled_mse"],
                "candidate_delta_vs_baseline": float(candidate_metrics["scaled_mse"] - baseline_metrics["scaled_mse"]),
                "baseline_profile_pearson": baseline_metrics["profile_pearson"],
                "candidate_profile_pearson": candidate_metrics["profile_pearson"],
                "baseline_profile_spearman": baseline_metrics["profile_spearman"],
                "candidate_profile_spearman": candidate_metrics["profile_spearman"],
                "candidate_profile_mae": candidate_metrics["profile_mae"],
                "candidate_profile_rmse": candidate_metrics["profile_rmse"],
                "candidate_profile_mean_bias": candidate_metrics["profile_mean_bias_prediction_minus_truth"],
            })

    staging.mkdir(parents=True)
    selected_crosswalk = pd.DataFrame(crosswalk_rows).sort_values(
        ["split_role", "patient_id_source_origin", "cell_line_name"]
    )
    selected_crosswalk.to_csv(staging / "source_object_crosswalk.tsv", sep="\t", index=False)
    cv_frame.to_csv(staging / "cv_scores.tsv", sep="\t", index=False, na_rep="")
    validation_columns = [
        "patient_id_source_origin", "cell_lines", "selected_candidate", "n_training_source_objects",
        "n_genes_excluded_zero_or_nonfinite_train_sd", "baseline_scaled_mse", "candidate_scaled_mse",
        "candidate_delta_vs_baseline", "baseline_profile_pearson", "candidate_profile_pearson",
        "baseline_profile_spearman", "candidate_profile_spearman", "candidate_profile_mae",
        "candidate_profile_rmse", "candidate_profile_mean_bias",
    ]
    pd.DataFrame(validation_records, columns=validation_columns).to_csv(
        staging / "validation_scores.tsv", sep="\t", index=False, na_rep=""
    )
    np.savez_compressed(staging / "model_parameters.npz", **params)

    excluded_locked_with_controls = []
    for row in frozen_joined:
        if row["split_role"] != "locked_eval":
            continue
        cell = str(row.get("lincs_cell_id", "")).strip().upper()
        groups = all_finite_24h.loc[all_finite_24h["cell_id_norm"].eq(cell), "control_group_id"].nunique()
        if groups:
            excluded_locked_with_controls.append({"cell_line": cell, "control_groups": int(groups),
                                                  "numerical_values_read": False})
    coverage = {
        "experiment_id": "EXP-050",
        "run_id": args.run_id,
        "status": "metadata_preflight_passed_and_source_values_evaluated",
        "analysis_scope": "response-blind source-side DepMap RNA-seq to LINCS 24-hour control profile reconstruction",
        "target_pdo_expression_read": False,
        "target_treated_expression_read": False,
        "perturbed_lincs_response_read": False,
        "drug_response_endpoint_read": False,
        "outer_test_values_read": False,
        "locked_eval_values_read": False,
        "identity_contract": {
            "frozen_response_blind": True,
            "exact978_gene_count": int(len(genes)),
            "ordered_gene_ids_sha256": actual_axis_hash,
            "cross_role_patient_id_conflicts": 0,
            "included_train_source_objects": len(training),
            "included_validation_source_objects": len(validation),
            "included_exact_depmap_lincs_lines": len(crosswalk_rows),
            "train_patient_ids": train_patient_ids,
            "validation_patient_ids": val_patient_ids,
            "eligible_locked_eval_lines_with_24h_controls_excluded_before_matrix_read": excluded_locked_with_controls,
        },
        "aggregation": {
            "control_time": "24 h exact",
            "technical_instances": "mean within control_group_id",
            "control_groups": "equal-weight mean within LINCS cell line",
            "cell_lines_with_shared_patient_id": "equal-weight mean to one source object",
            "source_object_weight": "equal across 13 training and 2 validation units",
        },
        "candidate_selection": {
            "leave_one_source_object_out_folds": len(training),
            "candidate_means_scaled_mse": candidate_means,
            "training_fold_median_baseline_mean_scaled_mse": baseline_mean,
            "selected_candidate": selected_candidate,
            "validation_evaluated_once": bool(selected_candidate),
        },
    }
    json_write(staging / "coverage.json", coverage)

    summary = {
        "experiment_id": "EXP-050",
        "run_id": args.run_id,
        "status": "completed_source_candidate_selected" if selected_candidate else "completed_no_candidate_improved_over_baseline",
        "scientific_question": "Does a simple source-side map reconstruct LINCS 24-hour control profiles better than a fold-fitted per-gene median among the 13 available training source origins, and how do two frozen validation origins score once?",
        "source_train_object_count": len(training),
        "source_validation_object_count": len(validation),
        "gene_count": len(genes),
        "primary_metric": "equal-object, equal-gene MSE scaled by each fold's training LINCS-control gene SD squared; lower is better",
        "cv_mean_scaled_mse": {"median_baseline": baseline_mean, **candidate_means},
        "cv_candidate_delta_vs_baseline": {
            name: score - baseline_mean for name, score in candidate_means.items()
        },
        "selected_candidate_from_training_cv_only": selected_candidate,
        "selected_candidate_full_train_details": selected_summary,
        "validation_scores": validation_records,
        "interpretation_boundary": "These source results only describe reconstruction of LINCS basal control profiles in this 13/2 source-object set. They do not validate GSE287429 author-normalized scale, target missing-gene handling, dose/time matching, drug response, PDO transfer, or screening utility.",
        "validation_inference": "Two frozen source objects are shown separately; no p-value, confidence interval, broad generalization claim, or tuning from validation.",
        "gpu_hours": 0,
    }
    json_write(staging / "summary.json", summary)

    # Hash the frozen inputs and all outputs for a complete, traceable CPU run.
    input_hashes = {
        name: {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
              "sha256": sha256_file(path)}
        for name, path in input_paths.items()
    }
    output_names = [
        "coverage.json", "source_object_crosswalk.tsv", "cv_scores.tsv",
        "validation_scores.tsv", "model_parameters.npz", "summary.json",
    ]
    manifest = {
        "experiment_id": "EXP-050",
        "run_id": args.run_id,
        "status": "completed",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": "WSL2 Conda drugscreening-gpu; CPU-only",
        "gpu_hours": 0,
        "code": {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                 "sha256": sha256_file(Path(__file__).resolve())},
        "inputs": input_hashes,
        "outputs": {
            name: {"bytes": (staging / name).stat().st_size,
                   "sha256": sha256_file(staging / name)}
            for name in output_names
        },
        "preregistered_protocol": "configs/experiments/EXP-050.yaml:source_rnaseq_to_lincs_scale_reconstruction",
        "uses_target_or_drug_labels": False,
    }
    json_write(staging / "run_manifest.json", manifest)
    staging.rename(output)
    print(json.dumps({"run_id": args.run_id, "status": summary["status"],
                      "selected_candidate": selected_candidate,
                      "cv_mean_scaled_mse": summary["cv_mean_scaled_mse"],
                      "validation_scores": validation_records,
                      "output": str(output)}, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
