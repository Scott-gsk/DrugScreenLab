"""Run the frozen EXP-066 five-fold cell-object response comparison.

This file exposes ``run_fold_comparison(dataset, frozen_config, outdir)`` for
the shared response-blind loader.  The loader owns input admission and must
verify the root metadata readback before exposing LINCS targets.  This runner
does no network access, data discovery, target selection, hyperparameter
search, or implicit fold creation.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from drug_screen.data.response_benefit_contract import load_frozen_response_domain
from drug_screen.models.exp066_response_prediction import (
    FoldFeatureEncoder,
    FoldFeatureSpec,
    ReducedRankRidge,
    build_fold_local_shuffled_cell_ids,
    model_receipt,
    per_drug_training_mean,
    validate_object_shuffle,
    zero_change,
)


FULL_INCHIKEY_PATTERN = r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$"
EXPECTED_INPUT_COLUMNS = {
    "condition_id", "lincs_cell_id", "depmap_model_id", "source_object_id",
    "fold_id", "rna_row_index", "drug_feature_row_index", "response_row_index",
    "pert_id", "full_inchikey", "canonical_smiles", "source_treatment_ids",
    "source_control_ids", "target_axis_hash",
}
ARMS = (
    "zero_change",
    "per_drug_training_mean",
    "rank16_rrr",
    "no_state_equal_capacity_rank16_rrr",
    "fold_local_object_state_shuffle_rank16_rrr",
)


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_float(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _finite_matrix(value: Any, name: str, columns: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"{name} must have {columns} columns; got {array.shape[1]}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains missing or non-finite values")
    return array.astype(np.float64, copy=False)


def _normalize_condition_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.copy()
    if "lincs_cell_id" not in metadata.columns and "cell_id" in metadata.columns:
        metadata["lincs_cell_id"] = metadata["cell_id"]
    if "full_inchikey" not in metadata.columns:
        for alias in ("InChIKey", "inchikey", "inchi_key"):
            if alias in metadata.columns:
                metadata["full_inchikey"] = metadata[alias]
                break
    if "target_axis_hash" not in metadata.columns:
        for alias in ("target_axis_sha256",):
            if alias in metadata.columns:
                metadata["target_axis_hash"] = metadata[alias]
                break
    if "log10_dose_um" not in metadata.columns:
        dose_name = next((name for name in ("log10_dose_uM", "dose_uM", "dose_um", "pert_dose")
                          if name in metadata.columns), None)
        if dose_name is None:
            raise ValueError("condition metadata needs log10_dose_um or a raw dose column")
        if dose_name == "log10_dose_uM":
            metadata["log10_dose_um"] = pd.to_numeric(metadata[dose_name], errors="coerce")
        elif dose_name == "pert_dose":
            if "pert_dose_unit" not in metadata.columns:
                raise ValueError("raw pert_dose metadata needs its unit")
            unit = metadata["pert_dose_unit"].astype(str).str.strip().str.lower().str.replace("μ", "u", regex=False)
            if not unit.eq("um").all():
                raise ValueError("EXP-066 exposure input requires micromolar dose metadata")
            dose = pd.to_numeric(metadata[dose_name], errors="coerce")
            if dose.isna().any() or (dose <= 0).any():
                raise ValueError("eligible dose metadata must be positive and finite")
            metadata["log10_dose_um"] = np.log10(dose.astype(np.float64))
        else:
            dose = pd.to_numeric(metadata[dose_name], errors="coerce")
            if dose.isna().any() or (dose <= 0).any():
                raise ValueError("eligible dose metadata must be positive and finite")
            metadata["log10_dose_um"] = np.log10(dose.astype(np.float64))
    if "log10_time_hours" not in metadata.columns:
        time_name = next((name for name in ("time_hours", "time_h", "pert_time")
                          if name in metadata.columns), None)
        if time_name is None:
            raise ValueError("condition metadata needs log10_time_hours or a raw time column")
        if time_name == "pert_time":
            if "pert_time_unit" not in metadata.columns:
                raise ValueError("raw pert_time metadata needs its unit")
            unit = metadata["pert_time_unit"].astype(str).str.strip().str.lower()
            if not unit.eq("h").all():
                raise ValueError("EXP-066 exposure input requires hour metadata")
        duration = pd.to_numeric(metadata[time_name], errors="coerce")
        if duration.isna().any() or (duration <= 0).any():
            raise ValueError("eligible time metadata must be positive and finite")
        metadata["log10_time_hours"] = np.log10(duration.astype(np.float64))
    return metadata


def _dataset_inputs(dataset: Any, config: Mapping[str, Any]) -> tuple[pd.DataFrame, np.ndarray,
                                                                        np.ndarray, np.ndarray,
                                                                        np.ndarray, list[str],
                                                                        np.ndarray, np.ndarray,
                                                                        np.ndarray]:
    """Validate the shared loader payload before the first fit."""
    gate_status = str(_value(dataset, "metadata_gate_status", ""))
    configured_gate_status = str(
        config.get("metadata_gate_status", config.get("root_metadata_gate_status", ""))
    )
    if gate_status != "PASS" or configured_gate_status != "PASS":
        raise ValueError("EXP-066 fitting requires a root-verified metadata gate PASS")
    receipt_hash = str(_value(dataset, "metadata_gate_receipt_sha256", ""))
    if not receipt_hash or receipt_hash != str(config.get("metadata_gate_receipt_sha256", "")):
        raise ValueError("dataset and frozen config metadata-gate receipt hashes differ")

    metadata = _value(dataset, "condition_metadata")
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("shared loader must return condition_metadata as a pandas DataFrame")
    metadata = _normalize_condition_metadata(metadata)
    missing = EXPECTED_INPUT_COLUMNS - set(metadata.columns)
    if missing:
        raise ValueError(f"shared condition contract lacks required fields: {sorted(missing)}")
    metadata = metadata.reset_index(drop=True).copy()
    if metadata.empty or len(metadata) > 20_000:
        raise ValueError("EXP-066 domain must contain 1..20,000 frozen conditions")
    if metadata["condition_id"].isna().any() or metadata["condition_id"].duplicated().any():
        raise ValueError("condition_id must be present and unique")
    if metadata["source_object_id"].isna().any() or metadata["fold_id"].isna().any():
        raise ValueError("PatientID object and frozen fold assignments must be complete")
    if metadata["full_inchikey"].isna().any() or not metadata["full_inchikey"].astype(str).str.fullmatch(
            FULL_INCHIKEY_PATTERN).all():
        raise ValueError("each condition needs one complete InChIKey")
    if metadata["canonical_smiles"].isna().any() or metadata["canonical_smiles"].astype(str).str.strip().eq("").any():
        raise ValueError("each admitted structure key needs a canonical SMILES")
    for source_column in ("source_treatment_ids", "source_control_ids"):
        if (metadata[source_column].isna().any()
                or metadata[source_column].astype(str).str.strip().eq("").any()):
            raise ValueError(f"each selected response condition needs traceable {source_column}")
    for col in ("rna_row_index", "drug_feature_row_index", "response_row_index", "fold_id"):
        numeric = pd.to_numeric(metadata[col], errors="coerce")
        if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{col} must be a complete integer axis")
        metadata[col] = numeric.astype(np.int64)
    if set(metadata["fold_id"].unique()) != set(range(5)):
        raise ValueError("frozen metadata must cover exactly fold IDs 0..4")
    object_fold_n = metadata.groupby("source_object_id", dropna=False)["fold_id"].nunique()
    if object_fold_n.ne(1).any():
        raise ValueError("a PatientID source object appears in multiple folds")
    if metadata["lincs_cell_id"].isna().any() or metadata["depmap_model_id"].isna().any():
        raise ValueError("LINCS-to-DepMap exact identity keys must be present")
    cell_bindings = metadata[["lincs_cell_id", "depmap_model_id", "rna_row_index"]].drop_duplicates()
    if cell_bindings["lincs_cell_id"].duplicated().any() or cell_bindings["depmap_model_id"].duplicated().any():
        raise ValueError("LINCS cell line must map one-to-one to DepMap ModelID/RNA row")
    key_bindings = metadata[["full_inchikey", "drug_feature_row_index", "canonical_smiles"]].drop_duplicates()
    if key_bindings["full_inchikey"].duplicated().any() or key_bindings["drug_feature_row_index"].duplicated().any():
        raise ValueError("drug feature rows and complete InChIKeys must map one-to-one")
    if not np.isfinite(pd.to_numeric(metadata["log10_dose_um"], errors="coerce")).all():
        raise ValueError("admitted dose metadata must be finite log10 micromolar values")
    if not np.isfinite(pd.to_numeric(metadata["log10_time_hours"], errors="coerce")).all():
        raise ValueError("admitted time metadata must be finite log10 hour values")
    hashes = metadata["target_axis_hash"].dropna().astype(str).unique()
    expected_hash = str(_value(dataset, "target_axis_hash", _value(dataset, "target_axis_sha256", "")))
    configured_axis_hash = str(config.get("target_axis_hash", config.get("target_axis_sha256", "")))
    if len(hashes) != 1 or not expected_hash or hashes[0] != expected_hash:
        raise ValueError("condition target-axis hashes do not match the loader axis")
    if not configured_axis_hash or hashes[0] != configured_axis_hash:
        raise ValueError("condition target-axis hash differs from frozen config")

    rna = _finite_matrix(_value(dataset, "rna_by_model"), "DepMap baseline RNA", 978)
    drug = _finite_matrix(_value(dataset, "drug_morgan_by_key"), "Morgan fingerprints", 2048)
    target = _finite_matrix(_value(dataset, "delta_by_condition"), "LINCS target deltas", 978)
    if len(target) != len(metadata):
        raise ValueError("loader delta rows must match condition_metadata row order")
    if metadata["response_row_index"].duplicated().any():
        raise ValueError("response row indices must be unique in the frozen condition set")
    source_rna_row_ids = _value(dataset, "rna_row_ids")
    if source_rna_row_ids is not None:
        source_rna_row_ids = np.asarray(source_rna_row_ids, dtype=np.int64)
        if source_rna_row_ids.ndim != 1 or len(source_rna_row_ids) != len(rna) or len(np.unique(source_rna_row_ids)) != len(rna):
            raise ValueError("rna_row_ids must bind every loaded DepMap row to one unique source row")
        source_index = metadata["rna_row_index"].to_numpy(dtype=np.int64)
        row_lookup = {int(source_row): local_row for local_row, source_row in enumerate(source_rna_row_ids.tolist())}
        missing_rna_rows = sorted(set(map(int, source_index)) - set(row_lookup))
        if missing_rna_rows:
            raise ValueError(f"condition metadata references {len(missing_rna_rows)} unreturned DepMap source rows")
        metadata["depmap_source_row_index"] = source_index
        metadata["rna_row_index"] = np.asarray([row_lookup[int(row)] for row in source_index], dtype=np.int64)
    if (metadata["rna_row_index"].min() < 0 or metadata["drug_feature_row_index"].min() < 0
            or metadata["rna_row_index"].max() >= len(rna)
            or metadata["drug_feature_row_index"].max() >= len(drug)):
        raise ValueError("condition row index exceeds an input feature axis")
    if set(metadata["rna_row_index"].unique()) != set(range(len(rna))):
        raise ValueError("loaded DepMap RNA rows are not exactly covered by admitted LINCS identities")
    # The frozen Morgan table may retain rows for response-blind structure keys
    # that do not survive condition admission. Every admitted key is bound
    # one-to-one above, and train-fold PCA selects only rows referenced by TRAIN.
    cell_bindings = metadata[["lincs_cell_id", "depmap_model_id", "rna_row_index"]].drop_duplicates()
    if cell_bindings["lincs_cell_id"].duplicated().any() or cell_bindings["depmap_model_id"].duplicated().any():
        raise ValueError("loaded LINCS cell line does not retain its one-to-one ModelID/RNA row binding")
    gene_ids = [str(gene) for gene in _value(dataset, "target_gene_ids", [])]
    if len(gene_ids) != 978 or len(set(gene_ids)) != 978:
        raise ValueError("target_gene_ids must provide the frozen unique 978-gene axis")
    shuffle_rows = np.asarray(_value(dataset, "shuffled_rna_row_by_fold"))
    if shuffle_rows.shape != (5, len(metadata)) or not np.issubdtype(shuffle_rows.dtype, np.integer):
        raise ValueError("shuffled_rna_row_by_fold must be an integer [5, n_conditions] matrix")
    if shuffle_rows.min() < 0 or shuffle_rows.max() >= len(rna):
        raise ValueError("shuffled RNA row index exceeds the DepMap expression axis")
    donor_ids = _value(dataset, "shuffled_donor_model_id_by_fold")
    if donor_ids is not None:
        donor_ids = np.asarray(donor_ids).astype(str)
        if donor_ids.shape != shuffle_rows.shape:
            raise ValueError("shuffled donor model IDs do not align with fold/condition rows")
        model_id_by_row = cell_bindings.drop_duplicates("rna_row_index").set_index("rna_row_index")["depmap_model_id"].astype(str)
        expected_ids = np.asarray([[model_id_by_row.loc[int(idx)] for idx in row]
                                   for row in shuffle_rows], dtype=str)
        if not np.array_equal(donor_ids, expected_ids):
            raise ValueError("shuffled donor ModelID does not match its frozen RNA row")
    else:
        model_by_local_row = cell_bindings.drop_duplicates("rna_row_index").set_index(
            "rna_row_index")["depmap_model_id"].astype(str)
        donor_ids = np.asarray(
            [[model_by_local_row.loc[int(idx)] for idx in row] for row in shuffle_rows], dtype=str,
        )
    condition_cell = metadata["rna_row_index"].to_numpy(dtype=np.int64)
    condition_drug = metadata["drug_feature_row_index"].to_numpy(dtype=np.int64)
    exposure = metadata[["log10_dose_um", "log10_time_hours"]].to_numpy(dtype=np.float64)
    return (
        metadata, rna, drug, target,
        np.stack((condition_cell, condition_drug), axis=0), gene_ids,
        exposure, shuffle_rows, donor_ids,
    )


def _per_condition_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           train_variance: np.ndarray,
                           variance_epsilon: float) -> dict[str, np.ndarray]:
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError("true and predicted response arrays must be aligned matrices")
    error = y_pred - y_true
    valid_genes = np.isfinite(train_variance) & (train_variance > variance_epsilon)
    finite_prediction = np.isfinite(y_pred).all(axis=1)
    norm_mse = np.full(len(y_true), np.nan, dtype=np.float64)
    raw_mse = np.full(len(y_true), np.nan, dtype=np.float64)
    if finite_prediction.any():
        raw_mse[finite_prediction] = np.square(error[finite_prediction]).mean(axis=1)
        if valid_genes.any():
            normalized = np.square(error[finite_prediction][:, valid_genes]) / train_variance[valid_genes]
            norm_mse[finite_prediction] = normalized.mean(axis=1)
    spearman = np.full(len(y_true), np.nan, dtype=np.float64)
    direction = np.full(len(y_true), np.nan, dtype=np.float64)
    nonzero_direction = np.full(len(y_true), np.nan, dtype=np.float64)
    for row in range(len(y_true)):
        if not finite_prediction[row]:
            continue
        if np.ptp(y_true[row]) > 0 and np.ptp(y_pred[row]) > 0:
            spearman[row] = float(spearmanr(y_true[row], y_pred[row]).statistic)
        direction[row] = float(np.mean(np.sign(y_true[row]) == np.sign(y_pred[row])))
        nonzero = (y_true[row] != 0) & (y_pred[row] != 0)
        if nonzero.any():
            nonzero_direction[row] = float(
                np.mean(np.sign(y_true[row, nonzero]) == np.sign(y_pred[row, nonzero]))
            )
    return {
        "normalized_mse": norm_mse,
        "raw_mse": raw_mse,
        "spearman": spearman,
        "directional_agreement": direction,
        "nonzero_directional_agreement": nonzero_direction,
        "valid_target_gene_count": np.full(len(y_true), int(valid_genes.sum()), dtype=np.int64),
        "excluded_target_gene_count": np.full(len(y_true), int((~valid_genes).sum()), dtype=np.int64),
    }


def _object_macro(condition_values: np.ndarray, object_ids: np.ndarray) -> dict[str, Any]:
    frame = pd.DataFrame({"source_object_id": object_ids.astype(str), "value": condition_values})
    finite = frame.loc[np.isfinite(frame["value"])].copy()
    total_objects = int(frame["source_object_id"].nunique())
    if finite.empty:
        return {
            "object_macro_mean": None,
            "total_objects": total_objects,
            "valid_objects": 0,
            "undefined_objects": total_objects,
            "valid_conditions": 0,
            "undefined_conditions": int(len(frame)),
        }
    per_object = finite.groupby("source_object_id", sort=True)["value"].mean()
    return {
        "object_macro_mean": float(per_object.mean()),
        "total_objects": total_objects,
        "valid_objects": int(per_object.size),
        "undefined_objects": int(total_objects - per_object.size),
        "valid_conditions": int(len(finite)),
        "undefined_conditions": int(frame["value"].isna().sum()),
    }


def _object_equal_weighted_variance(y_train: np.ndarray,
                                    object_ids: np.ndarray) -> np.ndarray:
    """Population response variance with equal total weight per PatientID.

    Within each PatientID every TRAIN condition receives equal weight, and each
    PatientID contributes the same total weight regardless of its condition
    count.  This retains within-object condition variation without allowing an
    object with many conditions to dominate the gene-wise variance.
    """
    y = _finite_matrix(y_train, "training responses")
    raw_objects = np.asarray(object_ids)
    if raw_objects.ndim != 1 or len(raw_objects) != len(y) or pd.isna(raw_objects).any():
        raise ValueError("training source-object IDs must align with response rows")
    objects = raw_objects.astype(str)
    codes, unique_objects = pd.factorize(objects, sort=True)
    if len(unique_objects) < 2:
        raise ValueError("training variance needs at least two source objects")
    counts = np.bincount(codes, minlength=len(unique_objects)).astype(np.float64)
    weights = 1.0 / counts[codes]
    mean = np.sum(y * weights[:, None], axis=0) / float(len(unique_objects))
    variance = np.sum(np.square(y - mean) * weights[:, None], axis=0) / float(len(unique_objects))
    return variance


def _summarize_metrics(y_true: np.ndarray, predictions: Mapping[str, np.ndarray],
                       train_variance: np.ndarray, metadata: pd.DataFrame,
                       variance_epsilon: float) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    objects = metadata["source_object_id"].astype(str).to_numpy()
    drugs = metadata["full_inchikey"].astype(str).to_numpy()
    per_arm: dict[str, Any] = {}
    detailed: dict[str, dict[str, np.ndarray]] = {}
    for arm, y_pred in predictions.items():
        condition_metrics = _per_condition_metrics(y_true, y_pred, train_variance, variance_epsilon)
        detailed[arm] = condition_metrics
        summary = {
            key: _object_macro(condition_metrics[key], objects)
            for key in ("normalized_mse", "raw_mse", "spearman", "directional_agreement",
                        "nonzero_directional_agreement")
        }
        per_drug: dict[str, Any] = {}
        frame = pd.DataFrame({"source_object_id": objects, "full_inchikey": drugs})
        for key in ("normalized_mse", "raw_mse", "spearman", "directional_agreement",
                    "nonzero_directional_agreement"):
            frame[key] = condition_metrics[key]
        for inchikey, group in frame.groupby("full_inchikey", sort=True):
            drug_object_count = int(group["source_object_id"].nunique())
            drug_summary: dict[str, Any] = {
                "conditions": int(len(group)),
                "objects": drug_object_count,
            }
            for metric in ("normalized_mse", "raw_mse", "spearman", "directional_agreement",
                           "nonzero_directional_agreement"):
                finite = group.loc[np.isfinite(group[metric])]
                by_object = finite.groupby("source_object_id", sort=True)[metric].mean()
                drug_summary[metric] = {
                    "object_macro_mean": float(by_object.mean()) if len(by_object) else None,
                    "total_objects": drug_object_count,
                    "valid_objects": int(len(by_object)),
                    "undefined_objects": int(drug_object_count - len(by_object)),
                    "valid_conditions": int(len(finite)),
                    "undefined_conditions": int(len(group) - len(finite)),
                }
            per_drug[str(inchikey)] = drug_summary
        per_arm[arm] = {
            "object_macro": summary,
            "per_drug": per_drug,
            "denominators": {
                "conditions": int(len(y_true)),
                "objects": int(pd.Series(objects).nunique()),
                "prediction_valid_conditions": int(np.isfinite(condition_metrics["raw_mse"]).sum()),
                "prediction_undefined_conditions": int(np.isnan(condition_metrics["raw_mse"]).sum()),
                "spearman_valid_conditions": int(np.isfinite(condition_metrics["spearman"]).sum()),
                "spearman_undefined_conditions": int(np.isnan(condition_metrics["spearman"]).sum()),
                "direction_valid_conditions": int(np.isfinite(condition_metrics["directional_agreement"]).sum()),
                "direction_undefined_conditions": int(np.isnan(condition_metrics["directional_agreement"]).sum()),
                "nonzero_direction_valid_conditions": int(np.isfinite(condition_metrics["nonzero_directional_agreement"]).sum()),
                "nonzero_direction_undefined_conditions": int(np.isnan(condition_metrics["nonzero_directional_agreement"]).sum()),
                "valid_target_genes_per_condition": int(condition_metrics["valid_target_gene_count"][0]),
                "excluded_near_zero_variance_genes": int(condition_metrics["excluded_target_gene_count"][0]),
            },
        }
    return per_arm, detailed


def _summarize_metric_arrays(condition_metrics_by_arm: Mapping[str, Mapping[str, np.ndarray]],
                             metadata: pd.DataFrame) -> dict[str, Any]:
    """Aggregate already-computed OOF condition metrics object first."""
    objects = metadata["source_object_id"].astype(str).to_numpy()
    drugs = metadata["full_inchikey"].astype(str).to_numpy()
    result: dict[str, Any] = {}
    metrics = ("normalized_mse", "raw_mse", "spearman", "directional_agreement",
               "nonzero_directional_agreement")
    for arm, condition_metrics in condition_metrics_by_arm.items():
        summary = {metric: _object_macro(condition_metrics[metric], objects) for metric in metrics}
        frame = pd.DataFrame({"source_object_id": objects, "full_inchikey": drugs})
        for metric in metrics:
            frame[metric] = condition_metrics[metric]
        per_drug: dict[str, Any] = {}
        for inchikey, group in frame.groupby("full_inchikey", sort=True):
            drug_summary: dict[str, Any] = {
                "conditions": int(len(group)),
                "objects": int(group["source_object_id"].nunique()),
            }
            for metric in metrics:
                finite = group.loc[np.isfinite(group[metric])]
                by_object = finite.groupby("source_object_id", sort=True)[metric].mean()
                total_objects = int(group["source_object_id"].nunique())
                drug_summary[metric] = {
                    "object_macro_mean": float(by_object.mean()) if len(by_object) else None,
                    "total_objects": total_objects,
                    "valid_objects": int(len(by_object)),
                    "undefined_objects": int(total_objects - len(by_object)),
                    "valid_conditions": int(len(finite)),
                    "undefined_conditions": int(len(group) - len(finite)),
                }
            per_drug[str(inchikey)] = drug_summary
        result[arm] = {
            "object_macro": summary,
            "per_drug": per_drug,
            "denominators": {
                "conditions": int(len(metadata)),
                "objects": int(pd.Series(objects).nunique()),
                "prediction_valid_conditions": int(np.isfinite(condition_metrics["raw_mse"]).sum()),
                "prediction_undefined_conditions": int(np.isnan(condition_metrics["raw_mse"]).sum()),
                "spearman_valid_conditions": int(np.isfinite(condition_metrics["spearman"]).sum()),
                "spearman_undefined_conditions": int(np.isnan(condition_metrics["spearman"]).sum()),
                "direction_valid_conditions": int(np.isfinite(condition_metrics["directional_agreement"]).sum()),
                "direction_undefined_conditions": int(np.isnan(condition_metrics["directional_agreement"]).sum()),
                "nonzero_direction_valid_conditions": int(np.isfinite(condition_metrics["nonzero_directional_agreement"]).sum()),
                "nonzero_direction_undefined_conditions": int(np.isnan(condition_metrics["nonzero_directional_agreement"]).sum()),
                "valid_target_genes_per_condition_min": int(np.min(condition_metrics["valid_target_gene_count"])),
                "valid_target_genes_per_condition_max": int(np.max(condition_metrics["valid_target_gene_count"])),
                "valid_target_gene_condition_pairs": int(np.sum(condition_metrics["valid_target_gene_count"])),
                "excluded_near_zero_variance_genes_per_condition_min": int(np.min(condition_metrics["excluded_target_gene_count"])),
                "excluded_near_zero_variance_genes_per_condition_max": int(np.max(condition_metrics["excluded_target_gene_count"])),
                "excluded_gene_condition_pairs": int(np.sum(condition_metrics["excluded_target_gene_count"])),
            },
        }
    return result


def _fold_fit(dataset: Any, config: Mapping[str, Any], metadata: pd.DataFrame,
              rna: np.ndarray, drug: np.ndarray, y: np.ndarray,
              exposure: np.ndarray, shuffle_rows: np.ndarray,
              fold: int) -> tuple[dict[str, Any], dict[str, Any]]:
    condition_cell, condition_drug = (
        metadata["rna_row_index"].to_numpy(dtype=np.int64),
        metadata["drug_feature_row_index"].to_numpy(dtype=np.int64),
    )
    folds = metadata["fold_id"].to_numpy(dtype=np.int64)
    train_rows = np.flatnonzero(folds != fold)
    test_rows = np.flatnonzero(folds == fold)
    if train_rows.size == 0 or test_rows.size == 0:
        raise ValueError(f"fold {fold} has an empty train or held-out partition")
    train_objects = set(metadata.iloc[train_rows]["source_object_id"].astype(str))
    test_objects = set(metadata.iloc[test_rows]["source_object_id"].astype(str))
    if train_objects & test_objects:
        raise ValueError(f"fold {fold} has source-object leakage")
    y_train = y[train_rows]
    y_test = y[test_rows]
    train_object_ids = metadata.iloc[train_rows]["source_object_id"].astype(str).to_numpy()
    train_variance = _object_equal_weighted_variance(y_train, train_object_ids)
    variance_epsilon = float(config["train_variance_epsilon"])

    spec = FoldFeatureSpec(
        rna_components=16,
        drug_components=16,
        no_state_seed=660066,
        fixed_nonlinear_components=256,
        fixed_rff_components=978,
        response_rank=16,
        ridge_alpha=1.0,
    )
    encoder = FoldFeatureEncoder(spec).fit(
        rna, drug, condition_cell, condition_drug, exposure, train_rows,
    )
    candidate_x = encoder.transform_candidate(rna, drug, condition_cell, condition_drug, exposure)
    candidate = ReducedRankRidge(rank=16, alpha=1.0).fit(candidate_x[train_rows], y_train)
    candidate_train_pred = candidate.predict(candidate_x[train_rows])
    candidate_test_pred = candidate.predict(candidate_x[test_rows])

    no_state_x = encoder.transform_no_state(drug, condition_drug, exposure)
    no_state = ReducedRankRidge(rank=16, alpha=1.0).fit(no_state_x[train_rows], y_train)
    no_state_train_pred = no_state.predict(no_state_x[train_rows])
    no_state_test_pred = no_state.predict(no_state_x[test_rows])

    shuffled_cell = shuffle_rows[fold]
    shuffled_encoder = FoldFeatureEncoder(spec).fit(
        rna, drug, shuffled_cell, condition_drug, exposure, train_rows,
    )
    shuffled_x = shuffled_encoder.transform_candidate(
        rna, drug, shuffled_cell, condition_drug, exposure,
    )
    shuffled_model = ReducedRankRidge(rank=16, alpha=1.0).fit(shuffled_x[train_rows], y_train)
    shuffled_train_pred = shuffled_model.predict(shuffled_x[train_rows])
    shuffled_test_pred = shuffled_model.predict(shuffled_x[test_rows])

    train_drug_index = condition_drug[train_rows]
    test_drug_index = condition_drug[test_rows]
    predictions = {
        "zero_change": zero_change(len(test_rows), y.shape[1]),
        "per_drug_training_mean": per_drug_training_mean(
            y_train, train_drug_index, test_drug_index, allow_missing=True,
        ),
        "rank16_rrr": candidate_test_pred,
        "no_state_equal_capacity_rank16_rrr": no_state_test_pred,
        "fold_local_object_state_shuffle_rank16_rrr": shuffled_test_pred,
    }
    train_predictions = {
        "zero_change": zero_change(len(train_rows), y.shape[1]),
        "per_drug_training_mean": per_drug_training_mean(
            y_train, train_drug_index, train_drug_index, allow_missing=True,
        ),
        "rank16_rrr": candidate_train_pred,
        "no_state_equal_capacity_rank16_rrr": no_state_train_pred,
        "fold_local_object_state_shuffle_rank16_rrr": shuffled_train_pred,
    }
    # Score the same float32 values that are retained in the keyed prediction
    # artifacts, so a readback verifier can reproduce each reported metric.
    predictions = {
        arm: np.asarray(matrix, dtype=np.float32).astype(np.float64)
        for arm, matrix in predictions.items()
    }
    train_predictions = {
        arm: np.asarray(matrix, dtype=np.float32).astype(np.float64)
        for arm, matrix in train_predictions.items()
    }
    test_metadata = metadata.iloc[test_rows].reset_index(drop=True)
    train_metadata = metadata.iloc[train_rows].reset_index(drop=True)
    test_summary, test_detail = _summarize_metrics(
        y_test, predictions, train_variance, test_metadata, variance_epsilon,
    )
    train_summary, _ = _summarize_metrics(
        y_train, train_predictions, train_variance, train_metadata, variance_epsilon,
    )
    fold_result = {
        "fold_id": int(fold),
        "train_conditions": int(len(train_rows)),
        "heldout_conditions": int(len(test_rows)),
        "train_objects": int(len(train_objects)),
        "heldout_objects": int(len(test_objects)),
        "train_drugs": int(metadata.iloc[train_rows]["full_inchikey"].nunique()),
        "heldout_drugs": int(metadata.iloc[test_rows]["full_inchikey"].nunique()),
        "train_variance_excluded_genes": int(np.count_nonzero(train_variance <= variance_epsilon)),
        "train_variance_fitting": {
            "rule": "population variance; each PatientID has equal total weight and its TRAIN conditions have equal within-object weight",
            "training_conditions": int(len(train_rows)),
            "training_objects": int(len(train_objects)),
            "epsilon_delta_units_squared": float(variance_epsilon),
            "eligible_genes": int(np.count_nonzero(train_variance > variance_epsilon)),
            "excluded_genes": int(np.count_nonzero(train_variance <= variance_epsilon)),
        },
        "train_metrics": train_summary,
        "heldout_metrics": test_summary,
        "models": {
            "rank16_rrr": model_receipt(candidate, encoder),
            "no_state_equal_capacity_rank16_rrr": model_receipt(no_state, encoder),
            "fold_local_object_state_shuffle_rank16_rrr": model_receipt(shuffled_model, shuffled_encoder),
        },
        "test_indices": test_rows,
        "test_predictions": predictions,
        "test_metric_detail": test_detail,
        "test_true": y_test,
        "train_variance": train_variance,
        "test_metadata": test_metadata,
    }
    return fold_result, {"train_variance": train_variance}


def _fit_candidate_predict(metadata: pd.DataFrame, rna: np.ndarray, drug: np.ndarray,
                           y: np.ndarray, exposure: np.ndarray,
                           index_arrays: np.ndarray, train_rows: np.ndarray,
                           predict_rows: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one frozen candidate model and predict a disjoint condition block."""
    condition_cell, condition_drug = index_arrays
    spec = FoldFeatureSpec()
    encoder = FoldFeatureEncoder(spec).fit(
        rna, drug, condition_cell, condition_drug, exposure, train_rows,
    )
    train_x = encoder.transform_candidate(
        rna, drug, condition_cell[train_rows], condition_drug[train_rows], exposure[train_rows],
    )
    predict_x = encoder.transform_candidate(
        rna, drug, condition_cell[predict_rows], condition_drug[predict_rows], exposure[predict_rows],
    )
    model = ReducedRankRidge(rank=16, alpha=1.0).fit(train_x, y[train_rows])
    prediction = np.asarray(model.predict(predict_x), dtype=np.float32)
    if prediction.shape != (len(predict_rows), 978) or not np.isfinite(prediction).all():
        raise ValueError("rank16 candidate returned an invalid response-feature block")
    receipt = model_receipt(model, encoder)
    receipt.update({
        "training_conditions": int(len(train_rows)),
        "prediction_conditions": int(len(predict_rows)),
        "training_objects": int(metadata.iloc[train_rows]["source_object_id"].nunique()),
        "prediction_objects": int(metadata.iloc[predict_rows]["source_object_id"].nunique()),
        "training_drugs": int(metadata.iloc[train_rows]["full_inchikey"].nunique()),
        "prediction_drugs": int(metadata.iloc[predict_rows]["full_inchikey"].nunique()),
    })
    return prediction, receipt


def _normalize_prediction_metadata(prediction_metadata: pd.DataFrame,
                                  condition_metadata: pd.DataFrame,
                                  rna: np.ndarray, drug: np.ndarray) -> pd.DataFrame:
    """Bind each B sensitivity pair to frozen DepMap and Morgan feature axes.

    ``rna_row_index`` is the full DepMap source row index. The returned frame
    preserves the caller's row order and adds only private local-index fields.
    """
    if not isinstance(prediction_metadata, pd.DataFrame):
        raise TypeError("prediction_metadata must be a pandas DataFrame")
    frame = prediction_metadata.copy().reset_index(drop=True)
    required = {
        "source_object_id", "lincs_cell_id", "rna_row_index", "full_inchikey",
        "drug_feature_row_index", "log10_dose_um", "log10_time_hours", "pair_id",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction_metadata lacks required fields: {sorted(missing)}")
    if frame.empty or frame["pair_id"].isna().any() or frame["pair_id"].astype(str).duplicated().any():
        raise ValueError("prediction_metadata must contain unique, nonempty pair_id rows")
    for column in ("source_object_id", "lincs_cell_id", "full_inchikey", "pair_id"):
        frame[column] = frame[column].astype(str)
    if not frame["full_inchikey"].str.fullmatch(FULL_INCHIKEY_PATTERN).all():
        raise ValueError("each prediction pair needs one complete InChIKey")
    for column in ("rna_row_index", "drug_feature_row_index"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"prediction pair {column} must be a complete integer axis")
        frame[column] = numeric.astype(np.int64)
    for column in ("log10_dose_um", "log10_time_hours"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[["log10_dose_um", "log10_time_hours"]].to_numpy(dtype=np.float64)).all():
        raise ValueError("prediction pair dose/time features must be finite log10 values")

    frame["_input_row"] = np.arange(len(frame), dtype=np.int64)
    cell_bindings = condition_metadata[[
        "source_object_id", "lincs_cell_id", "depmap_model_id",
        "depmap_source_row_index", "rna_row_index",
    ]].drop_duplicates()
    if cell_bindings["lincs_cell_id"].duplicated().any():
        raise ValueError("frozen LINCS cell IDs do not bind one-to-one to DepMap rows")
    cell_bindings = cell_bindings.rename(columns={
        "depmap_source_row_index": "_expected_source_rna_row_index",
        "rna_row_index": "_rna_local_row_index",
    })
    frame = frame.merge(
        cell_bindings,
        on=["source_object_id", "lincs_cell_id"],
        how="left",
        validate="many_to_one",
        sort=False,
    ).sort_values("_input_row", kind="mergesort").reset_index(drop=True)
    if frame["_rna_local_row_index"].isna().any():
        raise ValueError("a prediction pair cell is absent from the frozen LINCS-to-DepMap identity map")
    if not np.array_equal(
        frame["rna_row_index"].to_numpy(dtype=np.int64),
        frame["_expected_source_rna_row_index"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("prediction pair source RNA row disagrees with its frozen LINCS cell binding")
    frame["_rna_local_row_index"] = frame["_rna_local_row_index"].astype(np.int64)
    if frame["_rna_local_row_index"].max() >= len(rna):
        raise ValueError("prediction pair RNA row exceeds the loader-local DepMap axis")

    key_bindings = condition_metadata[[
        "full_inchikey", "drug_feature_row_index", "log10_dose_um", "log10_time_hours",
    ]].drop_duplicates()
    if key_bindings.groupby("full_inchikey", dropna=False).size().ne(1).any():
        raise ValueError("each frozen InChIKey must have one unique dose/time and Morgan row")
    key_bindings = key_bindings.rename(columns={
        "drug_feature_row_index": "_expected_drug_feature_row_index",
        "log10_dose_um": "_expected_log10_dose_um",
        "log10_time_hours": "_expected_log10_time_hours",
    })
    frame = frame.merge(
        key_bindings, on="full_inchikey", how="left", validate="many_to_one", sort=False,
    ).sort_values("_input_row", kind="mergesort").reset_index(drop=True)
    if frame["_expected_drug_feature_row_index"].isna().any():
        raise ValueError("a prediction pair InChIKey is absent from the frozen Morgan feature axis")
    if not np.array_equal(
        frame["drug_feature_row_index"].to_numpy(dtype=np.int64),
        frame["_expected_drug_feature_row_index"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("prediction pair full InChIKey disagrees with its Morgan feature row")
    if not np.allclose(
        frame[["log10_dose_um", "log10_time_hours"]].to_numpy(dtype=np.float64),
        frame[["_expected_log10_dose_um", "_expected_log10_time_hours"]].to_numpy(dtype=np.float64),
        rtol=0.0, atol=1e-12,
    ):
        raise ValueError("prediction pair dose/time differs from the key's frozen LINCS exposure")
    if frame["drug_feature_row_index"].max() >= len(drug):
        raise ValueError("prediction pair Morgan row exceeds the loaded drug feature axis")
    frame["_drug_feature_row_index"] = frame["drug_feature_row_index"].astype(np.int64)
    return frame


def _partition_shuffle_bindings(identity: pd.DataFrame, patient_ids: set[str],
                                outer_fold_id: int, partition: str,
                                seed: int = 660067) -> dict[tuple[str, str], dict[str, Any]]:
    """Create a deterministic PatientID-block derangement within one partition."""
    identity = identity.drop_duplicates().copy()
    identity["source_object_id"] = identity["source_object_id"].astype(str)
    identity["lincs_cell_id"] = identity["lincs_cell_id"].astype(str)
    identity["depmap_model_id"] = identity["depmap_model_id"].astype(str)
    objects = sorted(patient_ids)
    if len(objects) < 2:
        raise ValueError(f"partition {partition} needs at least two PatientIDs for object derangement")
    ordered = sorted(
        objects,
        key=lambda patient: hashlib.sha256(
            f"{seed}|shuffle|{outer_fold_id}|{partition}|{patient}".encode("utf-8")
        ).hexdigest(),
    )
    # Keep the same deterministic +1 cycle as the shared frozen metadata contract.
    donor_for = {patient: ordered[(i + 1) % len(ordered)]
                 for i, patient in enumerate(ordered)}
    cells_by_object: dict[str, list[str]] = {}
    model_by_cell: dict[str, str] = {}
    local_row_by_cell: dict[str, int] = {}
    for row in identity.itertuples(index=False):
        patient = str(row.source_object_id)
        cell = str(row.lincs_cell_id)
        if patient in patient_ids:
            cells_by_object.setdefault(patient, []).append(cell)
            model_by_cell[cell] = str(row.depmap_model_id)
            local_row_by_cell[cell] = int(row.rna_row_index)
    if set(cells_by_object) != patient_ids:
        raise ValueError("partition PatientID list does not match its frozen LINCS cell identities")
    cells_by_object = {patient: sorted(set(cells)) for patient, cells in cells_by_object.items()}
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for patient in objects:
        donor_patient = donor_for[patient]
        target_cells = cells_by_object[patient]
        donor_cells = cells_by_object[donor_patient]
        for rank, cell in enumerate(target_cells):
            donor_cell = donor_cells[rank % len(donor_cells)]
            result[(patient, cell)] = {
                "donor_object_id": donor_patient,
                "donor_lincs_cell_id": donor_cell,
                "donor_depmap_model_id": model_by_cell[donor_cell],
                "donor_rna_row_index": local_row_by_cell[donor_cell],
            }
    if any(patient == binding["donor_object_id"]
           for (patient, _), binding in result.items()):
        raise ValueError("partition derangement left a PatientID in place")
    return result


def _fit_candidate_and_shuffle_pairs(
    metadata: pd.DataFrame, rna: np.ndarray, drug: np.ndarray, y: np.ndarray,
    condition_exposure: np.ndarray, train_rows: np.ndarray, predict_pairs: pd.DataFrame,
    train_shuffle: Mapping[tuple[str, str], Mapping[str, Any]],
    predict_shuffle: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit normal and state-shuffled rank-16 models on identical TRAIN labels."""
    condition_cell = metadata["rna_row_index"].to_numpy(dtype=np.int64)
    condition_drug = metadata["drug_feature_row_index"].to_numpy(dtype=np.int64)
    pair_cell = predict_pairs["_rna_local_row_index"].to_numpy(dtype=np.int64)
    pair_drug = predict_pairs["_drug_feature_row_index"].to_numpy(dtype=np.int64)
    pair_exposure = predict_pairs[["log10_dose_um", "log10_time_hours"]].to_numpy(dtype=np.float64)

    spec = FoldFeatureSpec()
    encoder = FoldFeatureEncoder(spec).fit(
        rna, drug, condition_cell, condition_drug, condition_exposure, train_rows,
    )
    train_x = encoder.transform_candidate(
        rna, drug, condition_cell[train_rows], condition_drug[train_rows], condition_exposure[train_rows],
    )
    predict_x = encoder.transform_candidate(rna, drug, pair_cell, pair_drug, pair_exposure)
    model = ReducedRankRidge(rank=16, alpha=1.0).fit(train_x, y[train_rows])
    normal_prediction = np.asarray(model.predict(predict_x), dtype=np.float32)

    shuffled_condition_cell = condition_cell.copy()
    for row in train_rows:
        patient = str(metadata.iloc[int(row)]["source_object_id"])
        cell = str(metadata.iloc[int(row)]["lincs_cell_id"])
        shuffled_condition_cell[int(row)] = int(train_shuffle[(patient, cell)]["donor_rna_row_index"])
    pair_shuffle_evidence = [
        predict_shuffle[(str(row.source_object_id), str(row.lincs_cell_id))]
        for row in predict_pairs.itertuples(index=False)
    ]
    shuffled_pair_cell = np.asarray(
        [int(item["donor_rna_row_index"]) for item in pair_shuffle_evidence], dtype=np.int64,
    )
    shuffled_encoder = FoldFeatureEncoder(spec).fit(
        rna, drug, shuffled_condition_cell, condition_drug, condition_exposure, train_rows,
    )
    shuffled_train_x = shuffled_encoder.transform_candidate(
        rna, drug, shuffled_condition_cell[train_rows], condition_drug[train_rows], condition_exposure[train_rows],
    )
    shuffled_predict_x = shuffled_encoder.transform_candidate(
        rna, drug, shuffled_pair_cell, pair_drug, pair_exposure,
    )
    shuffled_model = ReducedRankRidge(rank=16, alpha=1.0).fit(shuffled_train_x, y[train_rows])
    shuffled_prediction = np.asarray(shuffled_model.predict(shuffled_predict_x), dtype=np.float32)
    expected_shape = (len(predict_pairs), 978)
    if (normal_prediction.shape != expected_shape or shuffled_prediction.shape != expected_shape
            or not np.isfinite(normal_prediction).all() or not np.isfinite(shuffled_prediction).all()):
        raise ValueError("nested response model returned an invalid pair prediction block")
    return normal_prediction, shuffled_prediction, {
        "normal_model": model_receipt(model, encoder),
        "shuffled_model": model_receipt(shuffled_model, shuffled_encoder),
        "training_patient_ids": sorted(set(metadata.iloc[train_rows]["source_object_id"].astype(str))),
        "prediction_patient_ids": sorted(set(predict_pairs["source_object_id"].astype(str))),
        "training_condition_count": int(len(train_rows)),
        "prediction_pair_count": int(len(predict_pairs)),
    }


def generate_nested_oof_response_features(
    domain: Any,
    train_patient_ids: list[str] | tuple[str, ...],
    predict_patient_ids: list[str] | tuple[str, ...],
    outer_fold_id: int,
    config: Mapping[str, Any],
    prediction_metadata: pd.DataFrame,
) -> dict[str, Any]:
    """Generate pair-ordered nested OOF response features for B.

    Models train only on approved EXP-066 LINCS conditions. ``prediction_metadata``
    supplies every B cell-by-key row, including combinations with no measured
    LINCS condition. Its row order and all pair rows are preserved.
    """
    normalized = _normalize_frozen_config(config, domain)
    if normalized.get("experiment_id") != "EXP-066":
        raise ValueError("nested response features must use the frozen EXP-066 model contract")
    outer_fold_id = int(outer_fold_id)
    if outer_fold_id not in range(5):
        raise ValueError("outer_fold_id must be one of the frozen folds 0..4")
    metadata, rna, drug, y, index_arrays, target_gene_ids, exposure, _, _ = _dataset_inputs(
        domain, normalized,
    )
    fold_ids = metadata["fold_id"].to_numpy(dtype=np.int64)
    expected_predict = set(metadata.loc[fold_ids == outer_fold_id, "source_object_id"].astype(str))
    expected_train = set(metadata.loc[fold_ids != outer_fold_id, "source_object_id"].astype(str))
    supplied_train = [str(value) for value in train_patient_ids]
    supplied_predict = [str(value) for value in predict_patient_ids]
    if len(set(supplied_train)) != len(supplied_train) or len(set(supplied_predict)) != len(supplied_predict):
        raise ValueError("explicit PatientID lists must be unique")
    if set(supplied_train) != expected_train or set(supplied_predict) != expected_predict:
        raise ValueError("explicit PatientID lists do not match the frozen outer-fold assignment")
    if set(supplied_train) & set(supplied_predict):
        raise ValueError("outer train and prediction PatientID sets overlap")

    outer_train_rows = np.flatnonzero(fold_ids != outer_fold_id)
    pair_metadata = _normalize_prediction_metadata(prediction_metadata, metadata, rna, drug)
    pair_objects = pair_metadata["source_object_id"].astype(str)
    if set(pair_objects) != expected_train | expected_predict:
        raise ValueError("prediction_metadata PatientIDs do not cover the frozen outer-fold domain exactly")
    fold_by_object = metadata.drop_duplicates("source_object_id").set_index(
        "source_object_id"
    )["fold_id"].astype(int).to_dict()
    pair_fold = pair_metadata["source_object_id"].map(fold_by_object)
    if pair_fold.isna().any():
        raise ValueError("prediction_metadata contains a PatientID absent from the frozen fold assignment")
    pair_metadata["_fold_id"] = pair_fold.astype(np.int64)
    train_pair_positions = np.flatnonzero(pair_metadata["_fold_id"].to_numpy() != outer_fold_id)
    predict_pair_positions = np.flatnonzero(pair_metadata["_fold_id"].to_numpy() == outer_fold_id)
    train_pairs = pair_metadata.iloc[train_pair_positions].reset_index(drop=True)
    predict_pairs = pair_metadata.iloc[predict_pair_positions].reset_index(drop=True)
    if len(train_pairs) + len(predict_pairs) != len(pair_metadata):
        raise ValueError("nested pair partitions do not cover every prediction row")
    train_oof = np.full((len(train_pairs), 978), np.nan, dtype=np.float32)
    train_oof_shuffled = np.full_like(train_oof, np.nan)
    train_oof_fold = np.full(len(train_pairs), -1, dtype=np.int8)
    train_oof_model_source = np.full(len(train_pairs), "", dtype=object)
    identity = metadata[[
        "source_object_id", "lincs_cell_id", "depmap_model_id", "rna_row_index",
    ]].drop_duplicates()
    inner_provenance: list[dict[str, Any]] = []
    train_donor_rows: list[dict[str, Any] | None] = [None] * len(train_pairs)
    started_cpu = time.process_time()

    for inner_fold_id in sorted(set(range(5)) - {outer_fold_id}):
        inner_patient_ids = set(metadata.loc[
            metadata["fold_id"].eq(inner_fold_id), "source_object_id"
        ].astype(str))
        inner_train_patient_ids = expected_train - inner_patient_ids
        inner_train_rows = np.flatnonzero(
            metadata["source_object_id"].astype(str).isin(inner_train_patient_ids).to_numpy()
        )
        pair_positions = np.flatnonzero(train_pairs["_fold_id"].to_numpy() == inner_fold_id)
        inner_pairs = train_pairs.iloc[pair_positions].reset_index(drop=True)
        if not len(inner_train_rows) or inner_pairs.empty:
            raise ValueError(f"nested fold {inner_fold_id} has an empty train or prediction partition")
        train_shuffle = _partition_shuffle_bindings(
            identity, inner_train_patient_ids, outer_fold_id, f"inner_{inner_fold_id}_train",
        )
        predict_shuffle = _partition_shuffle_bindings(
            identity, inner_patient_ids, outer_fold_id, f"inner_{inner_fold_id}_heldout",
        )
        inner_started = time.process_time()
        normal_prediction, shuffled_prediction, model_provenance = _fit_candidate_and_shuffle_pairs(
            metadata, rna, drug, y, exposure, inner_train_rows, inner_pairs,
            train_shuffle, predict_shuffle,
        )
        inner_elapsed = time.process_time() - inner_started
        train_oof[pair_positions] = normal_prediction
        train_oof_shuffled[pair_positions] = shuffled_prediction
        train_oof_fold[pair_positions] = int(inner_fold_id)
        train_oof_model_source[pair_positions] = f"outer_{outer_fold_id}_inner_{inner_fold_id}_oof"
        for local_position, row in enumerate(inner_pairs.itertuples(index=False)):
            train_donor_rows[int(pair_positions[local_position])] = predict_shuffle[
                (str(row.source_object_id), str(row.lincs_cell_id))
            ]
        inner_provenance.append({
            "inner_validation_fold_id": int(inner_fold_id),
            "training_patient_ids": model_provenance["training_patient_ids"],
            "prediction_patient_ids": model_provenance["prediction_patient_ids"],
            "training_condition_count": model_provenance["training_condition_count"],
            "prediction_pair_count": model_provenance["prediction_pair_count"],
            "cpu_seconds": _safe_float(inner_elapsed),
            "model_source": f"outer_{outer_fold_id}_inner_{inner_fold_id}_oof",
            "models": {
                "rank16_rrr": model_provenance["normal_model"],
                "fold_local_object_state_shuffle_rank16_rrr": model_provenance["shuffled_model"],
            },
        })

    if (not np.isfinite(train_oof).all() or not np.isfinite(train_oof_shuffled).all()
            or np.any(train_oof_fold < 0) or any(item is None for item in train_donor_rows)):
        raise ValueError("nested 4-fold OOF predictions do not cover every outer-training pair exactly once")

    outer_train_shuffle = _partition_shuffle_bindings(
        identity, expected_train, outer_fold_id, "outer_train",
    )
    outer_predict_shuffle = _partition_shuffle_bindings(
        identity, expected_predict, outer_fold_id, "outer_heldout",
    )
    outer_started = time.process_time()
    outer_normal, outer_shuffled, outer_model_provenance = _fit_candidate_and_shuffle_pairs(
        metadata, rna, drug, y, exposure, outer_train_rows, predict_pairs,
        outer_train_shuffle, outer_predict_shuffle,
    )
    outer_elapsed = time.process_time() - outer_started
    total_model_cpu = time.process_time() - started_cpu

    def output_block(frame: pd.DataFrame, normal: np.ndarray, shuffled: np.ndarray,
                     nested_fold: np.ndarray, model_source: np.ndarray,
                     donor_evidence: list[Mapping[str, Any]],
                     provenance: Mapping[str, Any]) -> dict[str, Any]:
        frame = frame.reset_index(drop=True)
        return {
            "pair_id": frame["pair_id"].astype(str).to_numpy(),
            "source_object_id": frame["source_object_id"].astype(str).to_numpy(),
            "lincs_cell_id": frame["lincs_cell_id"].astype(str).to_numpy(),
            "full_inchikey": frame["full_inchikey"].astype(str).to_numpy(),
            "drug_feature_row_index": frame["drug_feature_row_index"].to_numpy(dtype=np.int64),
            "log10_dose_um": frame["log10_dose_um"].to_numpy(dtype=np.float64),
            "log10_time_hours": frame["log10_time_hours"].to_numpy(dtype=np.float64),
            "nested_fold_id": nested_fold,
            "model_source": model_source,
            "prediction": normal,
            "shuffled_prediction": shuffled,
            "shuffle_donor_object_id": np.asarray([item["donor_object_id"] for item in donor_evidence], dtype=str),
            "shuffle_donor_lincs_cell_id": np.asarray([item["donor_lincs_cell_id"] for item in donor_evidence], dtype=str),
            "shuffle_donor_depmap_model_id": np.asarray([item["donor_depmap_model_id"] for item in donor_evidence], dtype=str),
            "training_patient_ids": provenance.get(
                "training_patient_ids", provenance.get("training_patient_ids_by_fold", {})
            ),
            "prediction_patient_ids": provenance.get(
                "prediction_patient_ids", provenance.get("prediction_patient_ids_by_fold", {})
            ),
            "model_provenance": provenance,
        }

    train_provenance = {
        "model_source": "four_fold_inner_oof_within_outer_train",
        "outer_train_patient_ids": sorted(expected_train),
        "outer_predict_patient_ids": sorted(expected_predict),
        "folds": inner_provenance,
        "training_patient_ids_by_fold": {
            str(item["inner_validation_fold_id"]): item["training_patient_ids"]
            for item in inner_provenance
        },
        "prediction_patient_ids_by_fold": {
            str(item["inner_validation_fold_id"]): item["prediction_patient_ids"]
            for item in inner_provenance
        },
    }
    predict_provenance = {
        "model_source": f"outer_{outer_fold_id}_train_refit",
        "outer_train_patient_ids": sorted(expected_train),
        "outer_predict_patient_ids": sorted(expected_predict),
        "training_patient_ids": outer_model_provenance["training_patient_ids"],
        "prediction_patient_ids": outer_model_provenance["prediction_patient_ids"],
        "training_condition_count": outer_model_provenance["training_condition_count"],
        "prediction_pair_count": outer_model_provenance["prediction_pair_count"],
        "cpu_seconds": _safe_float(outer_elapsed),
        "models": {
            "rank16_rrr": outer_model_provenance["normal_model"],
            "fold_local_object_state_shuffle_rank16_rrr": outer_model_provenance["shuffled_model"],
        },
    }
    predict_donor_rows = [
        outer_predict_shuffle[(str(row.source_object_id), str(row.lincs_cell_id))]
        for row in predict_pairs.itertuples(index=False)
    ]
    train_block = output_block(
        train_pairs, train_oof, train_oof_shuffled, train_oof_fold,
        train_oof_model_source.astype(str), train_donor_rows, train_provenance,
    )
    predict_block = output_block(
        predict_pairs, outer_normal, outer_shuffled,
        np.full(len(predict_pairs), outer_fold_id, dtype=np.int8),
        np.full(len(predict_pairs), f"outer_{outer_fold_id}_train_refit", dtype=object),
        predict_donor_rows, predict_provenance,
    )
    return {
        "schema": "exp066_nested_pair_response_features_v2",
        "experiment_id": "EXP-066",
        "outer_fold_id": outer_fold_id,
        "train_patient_ids": sorted(expected_train),
        "predict_patient_ids": sorted(expected_predict),
        "outer_train_patient_ids": sorted(expected_train),
        "outer_predict_patient_ids": sorted(expected_predict),
        "train": train_block,
        "predict": predict_block,
        "target_gene_ids": np.asarray(target_gene_ids, dtype=str),
        "target_axis_hash": str(metadata["target_axis_hash"].iloc[0]),
        "inner_fold_provenance": inner_provenance,
        "outer_model_provenance": outer_model_provenance,
        "cost": {
            "model_generation_cpu_seconds": _safe_float(total_model_cpu),
            "data_loader_cpu_seconds": _safe_float(float(normalized["data_load_cpu_seconds"])),
            "gpu_hours": 0.0,
            "fits": 10,
            "inner_folds": 4,
            "input_pair_rows": int(len(pair_metadata)),
            "outer_train_pair_rows": int(len(train_pairs)),
            "outer_predict_pair_rows": int(len(predict_pairs)),
        },
        "read_roles": {
            "lincs_targets": "EXP-066 approved training-response rows only for fitting; prediction rows are never used to fit their own model",
            "ctrp": "not read",
            "prism": "not read",
            "pdo": "not read",
            "locked": "not read",
        },
    }


def _write_fold_predictions(outdir: Path, fold_result: Mapping[str, Any], target_gene_ids: list[str]) -> dict[str, str]:
    fold = int(fold_result["fold_id"])
    metadata = fold_result["test_metadata"]
    path = outdir / "fold_predictions" / f"fold_{fold}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".npz.tmp")
    arrays: dict[str, np.ndarray] = {
        "condition_id": metadata["condition_id"].astype(str).to_numpy(dtype="U"),
        "pert_id": metadata["pert_id"].astype(str).to_numpy(dtype="U"),
        "lincs_cell_id": metadata["lincs_cell_id"].astype(str).to_numpy(dtype="U"),
        "depmap_model_id": metadata["depmap_model_id"].astype(str).to_numpy(dtype="U"),
        "source_object_id": metadata["source_object_id"].astype(str).to_numpy(dtype="U"),
        "full_inchikey": metadata["full_inchikey"].astype(str).to_numpy(dtype="U"),
        "canonical_smiles": metadata["canonical_smiles"].astype(str).to_numpy(dtype="U"),
        "source_treatment_ids": metadata["source_treatment_ids"].astype(str).to_numpy(dtype="U"),
        "source_control_ids": metadata["source_control_ids"].astype(str).to_numpy(dtype="U"),
        "fold_id": np.full(len(metadata), fold, dtype=np.int8),
        "rna_row_index": metadata["rna_row_index"].to_numpy(dtype=np.int64),
        "depmap_source_row_index": metadata.get(
            "depmap_source_row_index", metadata["rna_row_index"],
        ).to_numpy(dtype=np.int64),
        "response_row_index": metadata["response_row_index"].to_numpy(dtype=np.int64),
        "target_axis_hash": metadata["target_axis_hash"].astype(str).to_numpy(dtype="U"),
        "log10_dose_um": metadata["log10_dose_um"].to_numpy(dtype=np.float64),
        "log10_time_hours": metadata["log10_time_hours"].to_numpy(dtype=np.float64),
        "target_gene_ids": np.asarray(target_gene_ids, dtype="U"),
        "y_true": np.asarray(fold_result["test_true"], dtype=np.float32),
        "train_gene_variance": np.asarray(fold_result["train_variance"], dtype=np.float64),
    }
    for arm, matrix in fold_result["test_predictions"].items():
        arrays[f"pred__{arm}"] = np.asarray(matrix, dtype=np.float32)
        arrays[f"prediction_valid__{arm}"] = np.isfinite(matrix).all(axis=1)
        for metric, values in fold_result["test_metric_detail"][arm].items():
            arrays[f"metric__{arm}__{metric}"] = np.asarray(values)
    with temp.open("wb") as stream:
        np.savez(stream, **arrays)
    temp.replace(path)
    return {"path": str(path), "sha256": _hash_file(path), "bytes": int(path.stat().st_size)}


def _readback_fold_predictions(path: Path, fold_result: Mapping[str, Any],
                               target_gene_ids: list[str],
                               expected_sha256: str) -> int:
    """Read one completed prediction artifact and verify its scientific keys."""
    if _hash_file(path) != expected_sha256:
        raise IOError(f"fold {fold_result['fold_id']} prediction checksum changed on readback")
    metadata = fold_result["test_metadata"]
    with np.load(path, allow_pickle=False) as payload:
        expected_ids = metadata["condition_id"].astype(str).to_numpy()
        if not np.array_equal(payload["condition_id"].astype(str), expected_ids):
            raise IOError("fold prediction condition IDs changed on readback")
        for output_column, metadata_column in (
            ("pert_id", "pert_id"),
            ("lincs_cell_id", "lincs_cell_id"),
            ("depmap_model_id", "depmap_model_id"),
            ("source_object_id", "source_object_id"),
            ("full_inchikey", "full_inchikey"),
            ("canonical_smiles", "canonical_smiles"),
            ("target_axis_hash", "target_axis_hash"),
            ("source_treatment_ids", "source_treatment_ids"),
            ("source_control_ids", "source_control_ids"),
        ):
            if not np.array_equal(
                payload[output_column].astype(str), metadata[metadata_column].astype(str).to_numpy()
            ):
                raise IOError(f"fold prediction identity field changed on readback: {output_column}")
        expected_source_rows = metadata.get(
            "depmap_source_row_index", metadata["rna_row_index"],
        ).to_numpy(dtype=np.int64)
        if not np.array_equal(payload["depmap_source_row_index"], expected_source_rows):
            raise IOError("fold prediction DepMap source row index changed on readback")
        if not np.array_equal(payload["target_gene_ids"].astype(str), np.asarray(target_gene_ids, dtype=str)):
            raise IOError("fold prediction target-gene axis changed on readback")
        if not np.array_equal(payload["y_true"], np.asarray(fold_result["test_true"], dtype=np.float32)):
            raise IOError("fold prediction labels changed on readback")
        for arm, values in fold_result["test_predictions"].items():
            restored = payload[f"pred__{arm}"]
            expected = np.asarray(values, dtype=np.float32)
            if restored.shape != expected.shape or not np.array_equal(restored, expected, equal_nan=True):
                raise IOError(f"fold prediction values changed on readback for arm {arm}")
            valid = payload[f"prediction_valid__{arm}"].astype(bool)
            if not np.array_equal(valid, np.isfinite(restored).all(axis=1)):
                raise IOError(f"fold prediction validity mask is inconsistent for arm {arm}")
            for metric, values in fold_result["test_metric_detail"][arm].items():
                restored_metric = payload[f"metric__{arm}__{metric}"]
                expected_metric = np.asarray(values)
                if not np.array_equal(restored_metric, expected_metric, equal_nan=True):
                    raise IOError(f"fold metric changed on readback for arm {arm} / {metric}")
        return int(path.stat().st_size)


def _project_cost(cpu_used: float, fold_cpu_seconds: list[float], folds_done: int,
                  data_load_cpu_seconds: float, n_folds: int = 5) -> float:
    per_fold = float(np.mean(fold_cpu_seconds)) if fold_cpu_seconds else 0.0
    projected_without_margin = data_load_cpu_seconds + cpu_used + (n_folds - folds_done) * per_fold
    return projected_without_margin * 1.2


def _normalize_frozen_config(raw_config: Mapping[str, Any], dataset: Any) -> dict[str, Any]:
    """Flatten the approved EXP-066 YAML fields needed by the runner."""
    config = dict(raw_config)
    resources = raw_config.get("resources", {})
    model = raw_config.get("model", {})
    data = raw_config.get("data", {})
    metrics = raw_config.get("metrics", {})
    shared_contract = raw_config.get("shared_contract", {})
    if not isinstance(resources, Mapping) or not isinstance(model, Mapping):
        raise TypeError("frozen model/resources config blocks must be mappings")
    if not isinstance(data, Mapping) or not isinstance(metrics, Mapping):
        raise TypeError("frozen data/metrics config blocks must be mappings")
    if not isinstance(shared_contract, Mapping):
        raise TypeError("shared_contract config block must be a mapping")
    shared_data = shared_contract.get("data", {})
    if not isinstance(shared_data, Mapping):
        raise TypeError("shared_contract.data config block must be a mapping")

    config.setdefault("a_cpu_limit_seconds", resources.get("cpu_seconds_package_cap", 1500.0))
    config.setdefault("a_b_cpu_cap_seconds", resources.get("a_b_cpu_cap_seconds", 3600.0))
    if "a_b_cpu_spent_before_seconds" not in config:
        prior = resources.get("a_b_cpu_spent_before_seconds")
        if prior is None:
            raise ValueError("frozen config must include the measured A+B CPU spend before this package")
        config["a_b_cpu_spent_before_seconds"] = prior
    config.setdefault(
        "train_variance_epsilon",
        data.get("train_gene_variance_epsilon", metrics.get("train_gene_variance_epsilon", 1e-8)),
    )
    config.setdefault("shuffle_seed", model.get("seed", 660067))
    config.setdefault(
        "target_axis_hash",
        data.get("target_axis_sha256", shared_data.get("target_axis_sha256", "")),
    )
    if not config.get("metadata_gate_status"):
        config["metadata_gate_status"] = raw_config.get("root_metadata_gate_status", "")
    if not config.get("metadata_gate_receipt_sha256"):
        config["metadata_gate_receipt_sha256"] = raw_config.get("root_metadata_receipt_sha256", "")
    if "data_load_cpu_seconds" not in config:
        config["data_load_cpu_seconds"] = _value(dataset, "loader_cpu_seconds")
    return config


def run_fold_comparison(dataset: Any, frozen_config: Mapping[str, Any],
                        outdir: str | Path) -> dict[str, Any]:
    """Fit the frozen five-fold package, stopping if the first-fold cost gate fails.

    The first full fold is retained as the measured cost gate.  If measured
    CPU projection for all five folds, readback, and a 20% margin exceeds the
    A/combined CPU envelopes, the run closes without a scientific comparison.
    Otherwise the same run continues through the remaining four folds.
    """
    frozen_config = _normalize_frozen_config(frozen_config, dataset)
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    if frozen_config.get("experiment_id") != "EXP-066":
        raise ValueError("frozen config must identify EXP-066")
    run_id = str(frozen_config.get("run_id", ""))
    if not run_id:
        raise ValueError("frozen config must provide a unique run_id")
    cpu_a_limit = float(frozen_config.get("a_cpu_limit_seconds", 1500.0))
    cpu_ab_cap = float(frozen_config.get("a_b_cpu_cap_seconds", 3600.0))
    cpu_ab_prior = float(frozen_config["a_b_cpu_spent_before_seconds"])
    data_load_raw = frozen_config.get("data_load_cpu_seconds")
    if data_load_raw is None:
        raise ValueError("shared loader must return measured loader_cpu_seconds before the fit cost gate")
    data_load_cpu = float(data_load_raw)
    variance_epsilon = float(frozen_config.get("train_variance_epsilon", 1e-8))
    if cpu_a_limit <= 0 or cpu_a_limit > 1200 or cpu_ab_cap <= 0 or cpu_ab_prior < 0:
        raise ValueError("A and combined CPU envelopes exceed their frozen limits")
    if cpu_ab_prior >= cpu_ab_cap or cpu_a_limit > cpu_ab_cap - cpu_ab_prior:
        raise ValueError("A CPU allocation does not fit the remaining A/B combined envelope")
    if not np.isfinite(data_load_cpu) or data_load_cpu < 0:
        raise ValueError("shared loader CPU time must be finite and nonnegative")
    if not np.isfinite(variance_epsilon) or variance_epsilon != 1e-8:
        raise ValueError("EXP-066 train variance epsilon is frozen at 1e-8 delta units squared")
    shuffle_seed = int(frozen_config.get("shuffle_seed", 660067))
    destination = Path(outdir).resolve()
    if destination.exists():
        raise FileExistsError(f"EXP-066 run output already exists: {destination}")

    metadata, rna, drug, y, index_arrays, target_gene_ids, exposure, shuffle_rows, donor_model_ids = _dataset_inputs(
        dataset, frozen_config,
    )
    condition_cell, condition_drug = index_arrays
    expected_shuffle = build_fold_local_shuffled_cell_ids(
        metadata["lincs_cell_id"].astype(str).to_numpy(),
        metadata["source_object_id"].astype(str).to_numpy(),
        metadata["fold_id"].to_numpy(dtype=np.int64),
        n_folds=5,
        seed=shuffle_seed,
    )
    expected_shuffle_rows = np.empty_like(shuffle_rows)
    for fold in range(5):
        cell_id_to_row = metadata[["lincs_cell_id", "rna_row_index"]].drop_duplicates().set_index(
            "lincs_cell_id")["rna_row_index"]
        expected_shuffle_rows[fold] = [cell_id_to_row.loc[cell_id] for cell_id in expected_shuffle[fold]]
    if not np.array_equal(shuffle_rows, expected_shuffle_rows):
        raise ValueError("shared frozen object-shuffle mapping differs from the registered deterministic rule")
    donor_object_by_fold = np.empty((5, len(metadata)), dtype=object)
    model_to_object = metadata[["depmap_model_id", "source_object_id"]].drop_duplicates().set_index(
        "depmap_model_id")["source_object_id"].astype(str)
    for fold in range(5):
        donor_object_by_fold[fold] = [model_to_object.loc[model_id] for model_id in donor_model_ids[fold]]
        validate_object_shuffle(
            metadata["source_object_id"].astype(str).to_numpy(),
            donor_object_by_fold[fold].astype(str),
            np.where(metadata["fold_id"].to_numpy() == fold, "heldout", "train"),
        )

    # Writes start only after all response-blind and frozen-config gates pass.
    destination.mkdir(parents=True)
    metrics_by_fold: list[dict[str, Any]] = []
    files: dict[str, Any] = {}
    fold_cpu_seconds: list[float] = []
    metric_names = (
        "normalized_mse", "raw_mse", "spearman", "directional_agreement",
        "nonzero_directional_agreement", "valid_target_gene_count",
        "excluded_target_gene_count",
    )
    out_of_fold_metrics = {
        arm: {
            metric: np.full(len(metadata), np.nan, dtype=np.float64)
            for metric in metric_names
        }
        for arm in ARMS
    }
    completed_folds = 0
    cost_gate: dict[str, Any] | None = None
    cost_gates: list[dict[str, Any]] = []
    for fold in range(5):
        fold_started = time.process_time()
        fold_result, _ = _fold_fit(
            dataset, frozen_config, metadata, rna, drug, y, exposure, shuffle_rows, fold,
        )
        file_info = _write_fold_predictions(destination, fold_result, target_gene_ids)
        readback_bytes = _readback_fold_predictions(
            Path(file_info["path"]), fold_result, target_gene_ids, file_info["sha256"],
        )
        fold_elapsed = time.process_time() - fold_started
        fold_cpu_seconds.append(fold_elapsed)
        completed_folds += 1
        file_info["readback_bytes_verified"] = int(readback_bytes)
        files[f"fold_{fold}_predictions"] = file_info
        test_indices = np.asarray(fold_result["test_indices"], dtype=np.int64)
        for arm, details in fold_result["test_metric_detail"].items():
            for metric in metric_names:
                out_of_fold_metrics[arm][metric][test_indices] = details[metric]
        metrics_by_fold.append({key: value for key, value in fold_result.items()
                                if key not in {"test_indices", "test_predictions", "test_metric_detail",
                                               "test_true", "test_metadata", "train_variance"}})
        # First-fold denominator/cost receipt is available before any next fold.
        estimated = _project_cost(
            time.process_time() - started_cpu,
            fold_cpu_seconds,
            completed_folds,
            data_load_cpu,
        )
        cost_gate = {
            "status": "PASS" if estimated <= cpu_a_limit and estimated <= cpu_ab_cap - cpu_ab_prior else "FAIL",
            "run_id": run_id,
            "measured_fold_id": int(fold),
            "measured_fold_cpu_seconds": _safe_float(fold_elapsed),
            "completed_folds": int(completed_folds),
            "total_folds": 5,
            "projected_total_cpu_seconds_with_20pct_margin": _safe_float(estimated),
            "a_cpu_limit_seconds": _safe_float(cpu_a_limit),
            "a_b_cpu_cap_seconds": _safe_float(cpu_ab_cap),
            "a_b_cpu_spent_before_seconds": _safe_float(cpu_ab_prior),
            "data_loader_cpu_seconds": _safe_float(data_load_cpu),
            "projected_readback_scope": "all five fold prediction artifacts, based on measured fold fit/write/readback CPU and 20 percent margin",
            "denominator": {
                "conditions": int(len(metadata)),
                "fold_train_conditions": int(fold_result["train_conditions"]),
                "fold_heldout_conditions": int(fold_result["heldout_conditions"]),
                "fold_train_objects": int(fold_result["train_objects"]),
                "fold_heldout_objects": int(fold_result["heldout_objects"]),
                "fold_train_drugs": int(fold_result["train_drugs"]),
                "fold_heldout_drugs": int(fold_result["heldout_drugs"]),
                "heldout_metric_denominators_by_arm": {
                    arm: summary["denominators"]
                    for arm, summary in fold_result["heldout_metrics"].items()
                },
            },
        }
        cost_gates.append(cost_gate)
        gate_path = (
            destination / "first_fold_cost_gate.json"
            if fold == 0 else destination / f"cost_gate_after_fold_{fold + 1}.json"
        )
        _canonical_json(gate_path, cost_gate)
        if completed_folds == 1 and cost_gate["status"] != "PASS":
            break
        if estimated > cpu_a_limit or estimated > cpu_ab_cap - cpu_ab_prior:
            break

    complete = completed_folds == 5
    elapsed_cpu = time.process_time() - started_cpu
    total_package_cpu = data_load_cpu + elapsed_cpu
    elapsed_wall = time.monotonic() - started_wall
    budget_compliant = bool(
        total_package_cpu <= cpu_a_limit
        and total_package_cpu <= cpu_ab_cap - cpu_ab_prior
    )
    status = (
        "COMPLETE" if complete and budget_compliant
        else "COMPLETE_OVER_BUDGET" if complete
        else "COST_GATE_FAILED" if cost_gate and cost_gate["status"] == "FAIL"
        else "STOPPED_BUDGET"
    )
    summary = {
        "schema": "exp066_response_prediction_run_v1",
        "experiment_id": "EXP-066",
        "run_id": run_id,
        "status": status,
        "scientific_comparison_valid": bool(complete),
        "budget_compliant": budget_compliant,
        "metadata_gate_status": "PASS",
        "metadata_gate_receipt_sha256": str(frozen_config["metadata_gate_receipt_sha256"]),
        "target_axis_hash": str(metadata["target_axis_hash"].iloc[0]),
        "drug_feature_provenance": {
            "rdkit_version": "2023.09.5",
            "fingerprint": "Morgan radius 2, 2048 bits",
            "bit_axis_order": "bit_index 0..2047",
            "fingerprint_contract_sha256": str(
                frozen_config.get("metadata_gate", {}).get(
                    "fingerprint_contract_sha256", ""
                )
            ),
            "root_metadata_receipt_path": str(
                Path(str(frozen_config.get("metadata_gate", {}).get("shared_dir", "")))
                / "root_metadata_readback.json"
            ),
            "root_metadata_receipt_sha256": str(frozen_config["metadata_gate_receipt_sha256"]),
            "drug_rows_bound_by": "drug_feature_row_index and full_inchikey in frozen conditions",
            "morgan_rows_loaded": int(len(drug)),
            "morgan_rows_used_by_admitted_conditions": int(metadata["drug_feature_row_index"].nunique()),
            "unreferenced_morgan_rows": int(
                len(drug) - metadata["drug_feature_row_index"].nunique()
            ),
        },
        "condition_count": int(len(metadata)),
        "target_gene_count": 978,
        "fold_count_completed": int(completed_folds),
        "fold_count_required": 5,
        "arms": list(ARMS),
        "feature_contract": {
            "rna_genes": 978,
            "morgan_bits": 2048,
            "log_exposures": 2,
            "interaction_components": 256,
            "candidate_and_no_state_dimensions": 3284,
            "rna_pca_components": 16,
            "drug_pca_components": 16,
            "response_rank": 16,
            "ridge_alpha": 1.0,
            "no_state_seed": 660066,
        },
        "variance_epsilon": variance_epsilon,
        "train_variance_policy": "population variance over TRAIN conditions with equal total weight per PatientID and equal within-object condition weights; genes at or below the frozen epsilon are excluded and counted",
        "metric_policy": "per condition: raw MSE over all genes; NMSE averages gene squared errors divided by fold TRAIN variance over genes with variance > epsilon; Spearman across 978 genes is undefined for constant target/prediction; direction uses literal sign equality including 0==0, with a separate nonzero/nonzero subset; average valid conditions within each PatientID then equal-weight PatientIDs; per-drug metrics use the same object-first aggregation within each full InChIKey",
        "metric_rules": {
            "variance_epsilon_delta_units_squared": variance_epsilon,
            "variance_aggregation": "population weighted variance; each source object has equal total weight and its TRAIN conditions equal within-object weights",
            "spearman": "condition-wise across the ordered 978 genes; constant target or prediction is undefined and serialized as null",
            "directional_agreement": "mean(sign(prediction) == sign(actual)); zero equals zero",
            "nonzero_directional_agreement": "same direction rule on genes where both actual and predicted response are nonzero; undefined when the subset is empty",
            "missing_prediction": "undefined metric and prediction_valid=false; never filled with zero",
            "object_aggregation": "mean finite conditions within PatientID, then equal-weight PatientIDs",
            "per_drug_aggregation": "within each complete InChIKey, mean finite conditions within PatientID, then equal-weight contributing PatientIDs",
        },
        "cost": {
            "cpu_seconds": _safe_float(elapsed_cpu),
            "data_loader_cpu_seconds": _safe_float(data_load_cpu),
            "total_package_cpu_seconds_including_loader": _safe_float(total_package_cpu),
            "wall_seconds": _safe_float(elapsed_wall),
            "gpu_hours": 0.0,
            "a_cpu_limit_seconds": _safe_float(cpu_a_limit),
            "a_b_cpu_cap_seconds": _safe_float(cpu_ab_cap),
            "a_b_cpu_spent_before_seconds": _safe_float(cpu_ab_prior),
            "estimated_total_cpu_seconds_with_20pct_margin": cost_gate.get("projected_total_cpu_seconds_with_20pct_margin") if cost_gate else None,
            "a_b_cpu_spent_after_package_seconds": _safe_float(cpu_ab_prior + total_package_cpu),
        },
        "folds": metrics_by_fold,
        "out_of_fold_metrics": _summarize_metric_arrays(out_of_fold_metrics, metadata) if complete else None,
        "first_fold_cost_gate": cost_gates[0] if cost_gates else None,
        "cost_gates_by_fold": cost_gates,
        "files": files,
        "read_roles": {
            "lincs_targets": "selected frozen condition_id rows only",
            "depmap": "baseline RNA context only; never used to construct targets",
            "ctrp": "not read by A runner",
            "prism": "not read",
            "pdo": "not read",
            "locked": "not read",
        },
    }
    summary_path = destination / "run_summary.json"
    summary_readback_started = time.process_time()
    _canonical_json(summary_path, summary)
    reread_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if reread_summary != summary:
        raise IOError("EXP-066 run summary differs after JSON readback")
    summary_readback_cpu = time.process_time() - summary_readback_started
    _canonical_json(destination / "run_readback.json", {
        "status": "PASS_RUN_OUTPUT_READBACK" if complete else "PASS_PARTIAL_COST_GATE_READBACK",
        "run_id": run_id,
        "run_summary_sha256": _hash_file(summary_path),
        "run_summary_bytes": int(summary_path.stat().st_size),
        "metadata_gate_receipt_sha256": str(frozen_config["metadata_gate_receipt_sha256"]),
        "target_axis_hash": str(metadata["target_axis_hash"].iloc[0]),
        "drug_feature_provenance": summary["drug_feature_provenance"],
        "prediction_artifact_count": int(len(files)),
        "prediction_artifact_sha256": {
            key: value["sha256"] for key, value in files.items()
        },
        "summary_readback_cpu_seconds": _safe_float(summary_readback_cpu),
        "read_roles": summary["read_roles"],
    })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/EXP-066-response-prediction.yaml"))
    parser.add_argument("--run-id", required=True, help="Unique immutable EXP-066 run ID")
    parser.add_argument("--shared-dir", type=Path, help="Override the frozen shared metadata directory")
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    config_path = args.config if args.config.is_absolute() else repository_root / args.config
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("experiment_id") != "EXP-066":
        raise ValueError("A runner requires the frozen EXP-066 configuration")
    if raw.get("metadata_gate_status") != "PASS":
        raise PermissionError("A runner requires root-verified metadata gate PASS")
    metadata_gate = raw.get("metadata_gate", {})
    if not isinstance(metadata_gate, Mapping):
        raise TypeError("EXP-066 metadata_gate config must be a mapping")
    expected_receipt = str(raw.get("metadata_gate_receipt_sha256", ""))
    data_config = raw.get("data", {})
    resources = raw.get("resources", {})
    if not isinstance(data_config, Mapping) or not isinstance(resources, Mapping):
        raise TypeError("EXP-066 data/resources config blocks must be mappings")
    expected_axis = str(raw.get(
        "target_axis_hash",
        raw.get("target_axis_sha256", metadata_gate.get(
            "target_axis_sha256", data_config.get("target_axis_sha256", "")
        )),
    ))
    if not expected_receipt or not expected_axis:
        raise ValueError("frozen config must bind the root receipt SHA and target axis SHA")
    shared_dir_config = metadata_gate.get("shared_dir")
    if args.shared_dir is None and not shared_dir_config:
        raise ValueError("frozen config must provide the verified shared_dir")
    shared_value = args.shared_dir if args.shared_dir is not None else Path(str(shared_dir_config))
    shared_dir = shared_value if shared_value.is_absolute() else repository_root / shared_value

    # Reject an incomplete or over-budget envelope before any numeric input is loaded.
    a_cpu_limit = float(raw.get(
        "a_cpu_limit_seconds", resources.get("a_cpu_limit_seconds", 0.0),
    ))
    a_b_cpu_cap = float(raw.get(
        "a_b_cpu_cap_seconds", resources.get("a_b_cpu_cap_seconds", 3600.0),
    ))
    a_b_cpu_spent_before = raw.get(
        "a_b_cpu_spent_before_seconds", resources.get("a_b_cpu_spent_before_seconds"),
    )
    if a_b_cpu_spent_before is None:
        raise ValueError("frozen config must record the measured shared A+B CPU spend before loading values")
    a_b_cpu_spent_before = float(a_b_cpu_spent_before)
    if (a_cpu_limit <= 0 or a_cpu_limit > 1200 or a_b_cpu_cap > 3600
            or a_b_cpu_spent_before < 0 or a_cpu_limit > a_b_cpu_cap - a_b_cpu_spent_before):
        raise ValueError("frozen A CPU allocation does not fit the approved 1200/3600-second envelopes")

    # This is the sole numeric-stage entry point: CTRP labels are explicitly disabled.
    dataset = load_frozen_response_domain(
        shared_dir, read_lincs_targets=True, read_ctrp_labels=False,
    )
    if str(dataset.get("metadata_gate_status", "")) != "PASS":
        raise PermissionError("shared loader did not return metadata gate PASS")
    if str(dataset.get("metadata_gate_receipt_sha256", "")) != expected_receipt:
        raise PermissionError("shared loader receipt SHA differs from frozen EXP-066 config")
    if str(dataset.get("target_axis_hash", dataset.get("target_axis_sha256", ""))) != expected_axis:
        raise PermissionError("shared loader target axis differs from frozen EXP-066 config")
    if bool(dataset.get("value_access_audit", {}).get("ctrp_labels_loaded", True)):
        raise PermissionError("EXP-066 loader unexpectedly opened CTRP labels")

    run_config = dict(raw)
    run_config["run_id"] = str(args.run_id)
    run_config["metadata_gate_status"] = "PASS"
    run_config["metadata_gate_receipt_sha256"] = expected_receipt
    run_config["target_axis_hash"] = expected_axis
    run_config["data_load_cpu_seconds"] = float(dataset.get("load_cpu_seconds", np.nan))
    output_config = raw.get("outputs", {})
    if not isinstance(output_config, Mapping) or not output_config.get("root_dir"):
        raise ValueError("EXP-066 frozen config must specify outputs.root_dir")
    outdir = repository_root / str(output_config["root_dir"]) / str(args.run_id)
    summary = run_fold_comparison(dataset, run_config, outdir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.get("status") in {
        "COMPLETE", "COMPLETE_CANDIDATE_PASS", "COMPLETE_CANDIDATE_CLOSED", "COST_GATE_FAILED",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
