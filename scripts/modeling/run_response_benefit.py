#!/usr/bin/env python
"""Run the fixed nested-crossfit EXP-067 comparison on the shared frozen domain."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from drug_screen.data.response_benefit_contract import load_frozen_response_domain
from drug_screen.models.exp066_response_prediction import generate_nested_oof_response_features


ARMS = (
    "pretreatment_rna_plus_structure",
    "pretreatment_rna_plus_structure_plus_crossfitted_predicted_response",
    "equal_capacity_direct_rff",
    "object_shuffled_crossfitted_response",
)
FOLD0_REUSE_RUN_ID = "20261008_exp067_nested_response_benefit_v3"
FOLD0_REUSE_ROOT_RECEIPT_SHA256 = "f88d4e6962543ac268e50d093f4d4a69cfdf7ba9e413af57d60f539ddc322df7"
METADATA_RECEIPT_SHA256 = "dc4b649d960a326916bb53c1734c0204f67bd877e4cce9467f35a255779633c9"
KNOWN_B_PACKAGE_CPU_DEBIT_SECONDS = 1605.5639088
KNOWN_SHARED_AB_CPU_DEBIT_SECONDS = 2555.5639088


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _project_cpu_seconds(elapsed: float, fold_reference_seconds: list[float],
                         folds_remaining: int, readback_allowance: float,
                         margin: float = 0.2) -> float:
    """Project remaining CPU using completed model-fold costs, including reused-fold history."""
    values = [float(value) for value in fold_reference_seconds]
    numbers = [float(elapsed), float(readback_allowance), float(margin), *values]
    if not all(np.isfinite(value) for value in numbers) or elapsed < 0 or readback_allowance < 0:
        raise ValueError("CPU projection inputs must be finite and non-negative")
    if folds_remaining < 0 or not 0 <= margin < 10 or not values:
        raise ValueError("CPU projection requires completed-fold cost evidence and a valid remaining count")
    if any(value <= 0 for value in values):
        raise ValueError("fold reference costs must be positive")
    return (elapsed + folds_remaining * float(np.mean(values)) + readback_allowance) * (1 + margin)


def _project_package_cpu_seconds(preflight: float, elapsed: float, fold_reference_seconds: list[float],
                                 folds_remaining: int, readback_allowance: float
                                 ) -> tuple[float, float]:
    if not np.isfinite(preflight) or preflight < 0:
        raise ValueError("cumulative package CPU debit must be finite and non-negative")
    projected = _project_cpu_seconds(elapsed, fold_reference_seconds, folds_remaining, readback_allowance)
    return projected, float(preflight) + projected


def _optional_cpu_cap(value: Any, field: str) -> float | None:
    """Parse a legacy/reporting CPU cap without making it a required runtime limit."""
    if value is None or value is False:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "disabled", "off"}:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative number, null, or disabled")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{field} must be finite and non-negative when set")
    return numeric


def _effective_cpu_debit(resources: Mapping[str, Any], field: str, known_floor: float) -> tuple[float, float]:
    """Keep known cumulative CPU charges even when loading a stale/nullable config."""
    configured_value = resources.get(field)
    configured = 0.0 if configured_value is None else float(configured_value)
    if not np.isfinite(configured) or configured < 0:
        raise ValueError(f"{field} must preserve a finite non-negative cumulative CPU debit")
    return configured, max(configured, known_floor)


def _configured_cpu_threads(resources: Mapping[str, Any]) -> int:
    value = resources.get("cpu_threads", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("resources.cpu_threads must be a positive integer")
    return value


def _load_reusable_fold0(source_run_dir: Path, expected_frame: pd.DataFrame
                         ) -> tuple[dict[str, np.ndarray], dict[str, Any], float]:
    """Read the independently verified v3 fold-0 artifact without changing its source run."""
    if source_run_dir.name != FOLD0_REUSE_RUN_ID:
        raise ValueError("only the independently verified EXP-067 v3 fold-0 artifact is eligible for reuse")
    receipt_path = source_run_dir / "root_cost_gate_readback.json"
    if not receipt_path.is_file() or _sha256(receipt_path) != FOLD0_REUSE_ROOT_RECEIPT_SHA256:
        raise IOError("the fixed fold-0 source root receipt is missing or changed")
    root_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    summary_path = source_run_dir / "run_summary.json"
    fold_summary_path = source_run_dir / "fold_0_summary.json"
    prediction_path = source_run_dir / "fold_0_pair_predictions.tsv"
    object_path = source_run_dir / "fold_0_object_metrics.tsv"
    drug_path = source_run_dir / "fold_0_drug_metrics.tsv"
    gate_path = source_run_dir / "first_outer_fold_cost_gate.json"
    pair_audit_path = source_run_dir / "pair_domain_audit.json"
    required = (summary_path, fold_summary_path, prediction_path, object_path, drug_path, gate_path,
                pair_audit_path)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("one or more independently verified fold-0 source artifacts are missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fold_summary = json.loads(fold_summary_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    hashes = root_receipt.get("hashes", {})
    if (root_receipt.get("root_verification_status") != "PASS_PARTIAL_FOLD_AND_COST_GATE_ONLY"
            or root_receipt.get("run_id") != FOLD0_REUSE_RUN_ID
            or root_receipt.get("metadata_gate_receipt_sha256") != METADATA_RECEIPT_SHA256
            or _sha256(summary_path) != hashes.get("run_summary_sha256")
            or _sha256(fold_summary_path) != hashes.get("fold0_summary_sha256")
            or _sha256(prediction_path) != hashes.get("fold0_predictions_sha256")
            or _sha256(object_path) != hashes.get("fold0_object_metrics_sha256")
            or _sha256(drug_path) != hashes.get("fold0_drug_metrics_sha256")
            or _sha256(pair_audit_path) != hashes.get("pair_domain_audit_sha256")
            or _sha256(gate_path) != hashes.get("cost_gate_sha256")):
        raise IOError("fold-0 source outputs do not match the immutable root receipt")
    denominators = root_receipt.get("denominators", {})
    if (denominators.get("frozen_source_conditions") != 2830
            or denominators.get("frozen_source_objects") != 34
            or denominators.get("full_chemical_keys") != 138
            or denominators.get("genes") != 978
            or denominators.get("fold0_heldout_pairs") != 816
            or denominators.get("fold0_heldout_objects") != 7):
        raise ValueError("fold-0 root receipt does not match the frozen source/fold denominators")
    if (summary.get("run_id") != FOLD0_REUSE_RUN_ID
            or summary.get("status") != "COST_GATE_FAILED" or summary.get("scientific_comparison_valid")
            or summary.get("outer_folds_completed") != 1
            or summary.get("pair_count") != 3965 or summary.get("object_count") != 34
            or summary.get("drug_count") != 138 or summary.get("target_gene_count") != 978
            or summary.get("metadata_gate_receipt_sha256") != METADATA_RECEIPT_SHA256
            or fold_summary.get("fold_id") != 0 or gate.get("measured_outer_fold") != 0
            or gate.get("status") != "FAIL"):
        raise ValueError("fold-0 source is not the registered incomplete cost-gate run")
    if (fold_summary.get("prediction_sha256") != _sha256(prediction_path)
            or fold_summary.get("object_metrics_sha256") != _sha256(object_path)
            or fold_summary.get("drug_metrics_sha256") != _sha256(drug_path)):
        raise IOError("fold-0 summary does not bind its predictions and metrics")
    expected_columns = ["pair_id", "source_object_id", "cellid", "full_inchikey", "fold_id", "aac_published"]
    expected = expected_frame.loc[:, expected_columns].reset_index(drop=True)
    if len(expected) != 816 or expected["fold_id"].astype(int).ne(0).any():
        raise ValueError("current frozen fold-0 key frame does not match its 816-row identity contract")
    cached = pd.read_csv(prediction_path, sep="\t", dtype={
        "pair_id": "string", "source_object_id": "string", "cellid": "string",
        "full_inchikey": "string", "fold_id": "int64",
    })
    if len(cached) != len(expected) or not np.array_equal(
            cached[expected_columns[:5]].astype(str).to_numpy(),
            expected[expected_columns[:5]].astype(str).to_numpy()):
        raise ValueError("cached fold-0 keys/order differ from the current frozen fold")
    if not np.allclose(cached["aac_published"].to_numpy(float), expected["aac_published"].to_numpy(float),
                       rtol=0, atol=1e-12):
        raise ValueError("cached fold-0 labels differ from the same frozen keyed rows")
    predictions: dict[str, np.ndarray] = {}
    for arm in ARMS:
        column = f"prediction__{arm}"
        if column not in cached:
            raise ValueError(f"cached fold-0 output is missing arm {arm}")
        values = cached[column].to_numpy(dtype=np.float64)
        if values.shape != (len(expected),) or not np.isfinite(values).all():
            raise ValueError(f"cached fold-0 predictions for {arm} have an invalid shape or non-finite values")
        predictions[arm] = values
    historical_seconds = float(gate["measured_fold_cpu_seconds"])
    if not np.isfinite(historical_seconds) or historical_seconds <= 0:
        raise ValueError("fold-0 source lacks a valid measured model-fold cost")
    return predictions, fold_summary["fold"], historical_seconds


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _portable_output_path(path: Path, root: Path) -> str:
    """Prefer repository-relative artifact links; retain valid paths for external test roots."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path.resolve())


def _bind_lincs_cell_id(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose the frozen LINCS cell alias used by EXP-066 nested prediction metadata."""
    if "cell_id" not in frame.columns or frame["cell_id"].isna().any():
        raise ValueError("frozen identity binding must provide a non-missing LINCS cell_id")
    cell_ids = frame["cell_id"].astype(str)
    if "lincs_cell_id" in frame.columns:
        lincs_ids = frame["lincs_cell_id"].astype(str)
        if frame["lincs_cell_id"].isna().any() or not lincs_ids.equals(cell_ids):
            raise ValueError("lincs_cell_id must exactly match the frozen LINCS cell_id")
    bound = frame.copy()
    bound["lincs_cell_id"] = cell_ids
    return bound


def _bind_measured_response_loader_cpu(config: Mapping[str, Any], loader_cpu_seconds: float) -> dict[str, Any]:
    """Pass the caller's measured shared-loader cost into the nested response API."""
    value = float(loader_cpu_seconds)
    if not np.isfinite(value) or value < 0:
        raise ValueError("measured response loader CPU seconds must be finite and non-negative")
    bound = dict(config)
    bound["data_load_cpu_seconds"] = value
    return bound


def _load_pair_frame(domain: Mapping[str, Any], shared_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Bind development AAC rows to frozen identity, key, exposure, and split metadata."""
    conditions = domain["condition_metadata"].copy()
    pairs = pd.read_csv(shared_dir / "sensitivity_cell_drug_pairs.tsv", sep="\t", dtype="string")
    identity = pd.read_csv(shared_dir / "cell_identity_frozen.tsv", sep="\t", dtype="string")
    labels = domain.get("ctrp_aac_rows")
    if not isinstance(labels, pd.DataFrame) or not {"cellid", "drugid", "aac_published"}.issubset(labels.columns):
        raise ValueError("the authorized shared loader did not return selected CTRP AAC labels")
    required_pair = {"source_object_id", "full_inchikey", "cellid", "drugid"}
    if required_pair - set(pairs.columns):
        raise ValueError(f"frozen CTRP pair metadata lacks {sorted(required_pair - set(pairs.columns))}")
    key_exposure = conditions.groupby("full_inchikey", as_index=False).agg(
        exposure_rows=("condition_id", "size"),
        dose_values=("log10_dose_um", "nunique"),
        time_values=("log10_time_hours", "nunique"),
        log10_dose_um=("log10_dose_um", "first"),
        log10_time_hours=("log10_time_hours", "first"),
        drug_feature_row_index=("drug_feature_row_index", "first"),
    )
    if key_exposure[["dose_values", "time_values"]].ne(1).to_numpy().any():
        raise ValueError("shared LINCS conditions do not define one frozen exposure per exact key")
    if conditions.groupby("full_inchikey")["drug_feature_row_index"].nunique().ne(1).any():
        raise ValueError("a complete key maps to more than one Morgan row")
    cell_bindings = identity[["ctrp_cell_id", "cell_id", "depmap_model_id", "source_object_id", "rna_row_index"]].copy()
    if cell_bindings["ctrp_cell_id"].duplicated().any() or cell_bindings["source_object_id"].isna().any():
        raise ValueError("frozen CTRP cell to PatientID/RNA bindings are not unique")
    fold_map = conditions[["source_object_id", "fold_id"]].drop_duplicates()
    if fold_map["source_object_id"].duplicated().any():
        raise ValueError("a PatientID has multiple frozen outer folds")

    frame = pairs.merge(
        cell_bindings, left_on="cellid", right_on="ctrp_cell_id", how="left", validate="many_to_one",
        suffixes=("", "_identity"),
    )
    if frame[["cell_id", "depmap_model_id", "rna_row_index"]].isna().any().any():
        raise ValueError("a frozen AAC pair has no exact DepMap/LINCS pretreatment identity")
    if not frame["source_object_id"].astype(str).eq(frame["source_object_id_identity"].astype(str)).all():
        raise ValueError("CTRP pair PatientID differs from the frozen LINCS/DepMap identity")
    frame = frame.drop(columns=["source_object_id_identity"])
    frame = frame.merge(key_exposure, on="full_inchikey", how="inner", validate="many_to_one")
    frame = frame.merge(fold_map, on="source_object_id", how="inner", validate="many_to_one")
    frame = _bind_lincs_cell_id(frame)
    if len(frame) != len(pairs):
        raise ValueError("frozen AAC pair domain lost an exact key or PatientID fold during binding")
    if frame.duplicated(["cellid", "full_inchikey"]).any():
        raise ValueError("the CTRP pair domain contains duplicate cell × exact-key rows")

    metadata_pair_count = int(len(frame))
    numeric_labels = labels[["cellid", "drugid", "aac_published"]].copy()
    numeric_labels["aac_published"] = pd.to_numeric(numeric_labels["aac_published"], errors="coerce")
    numeric_labels = numeric_labels.loc[np.isfinite(numeric_labels["aac_published"])].copy()
    if numeric_labels.empty:
        raise ValueError("no finite AAC labels remain in the frozen development domain")
    label_summary = numeric_labels.groupby(["cellid", "drugid"], as_index=False, sort=True).agg(
        aac_published=("aac_published", "mean"),
        n_aac_rows=("aac_published", "size"),
    )
    frame = frame.merge(label_summary, on=["cellid", "drugid"], how="inner", validate="one_to_one")
    if frame.empty:
        raise ValueError("no labeled AAC rows remain after exact metadata binding")
    pair_audit = {
        "metadata_pair_count": metadata_pair_count,
        "finite_aac_row_count_before_cell_drug_aggregation": int(len(numeric_labels)),
        "unique_finite_cell_drug_label_count": int(len(label_summary)),
        "duplicate_finite_aac_rows_averaged": int(len(numeric_labels) - len(label_summary)),
        "metadata_pairs_without_finite_aac": int(metadata_pair_count - len(frame)),
        "included_finite_aac_pairs": int(len(frame)),
    }
    frame["source_object_id"] = frame["source_object_id"].astype(str)
    frame["full_inchikey"] = frame["full_inchikey"].astype(str)
    frame["pair_id"] = frame["cellid"].astype(str) + "|" + frame["full_inchikey"]
    if frame["pair_id"].duplicated().any():
        raise ValueError("cell × InChIKey pair identifier is not unique")
    frame["fold_id"] = pd.to_numeric(frame["fold_id"], errors="raise").astype(int)
    frame["rna_row_index"] = pd.to_numeric(frame["rna_row_index"], errors="raise").astype(int)
    frame["drug_feature_row_index"] = pd.to_numeric(frame["drug_feature_row_index"], errors="raise").astype(int)
    frame["log10_dose_um"] = pd.to_numeric(frame["log10_dose_um"], errors="raise").astype(float)
    frame["log10_time_hours"] = pd.to_numeric(frame["log10_time_hours"], errors="raise").astype(float)
    if set(frame["fold_id"].unique()) != set(range(5)):
        raise ValueError("AAC rows do not cover every frozen PatientID outer fold")
    return (
        frame.sort_values(["fold_id", "source_object_id", "full_inchikey"], kind="mergesort").reset_index(drop=True),
        pair_audit,
    )


def _pair_inputs(domain: Mapping[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rna = np.asarray(domain["rna_by_model"], dtype=np.float64)
    drug = np.asarray(domain["drug_morgan_by_key"], dtype=np.float64)
    source_rna_ids = np.asarray(domain["rna_row_ids"], dtype=np.int64)
    rna_lookup = {int(source): i for i, source in enumerate(source_rna_ids.tolist())}
    if len(rna_lookup) != len(source_rna_ids):
        raise ValueError("DepMap RNA source row axis is not unique")
    metadata = frame.copy()
    metadata["local_rna_row_index"] = metadata["rna_row_index"].map(rna_lookup)
    if metadata["local_rna_row_index"].isna().any():
        raise ValueError("some CTRP cell rows do not have a loaded exact978 RNA row")
    metadata["local_rna_row_index"] = metadata["local_rna_row_index"].astype(int)
    if metadata["local_rna_row_index"].max() >= len(rna):
        raise ValueError("pair RNA index exceeds loaded DepMap matrix")
    if metadata["drug_feature_row_index"].max() >= len(drug):
        raise ValueError("pair drug row exceeds frozen Morgan feature matrix")
    return rna, drug, metadata


def _direct_matrix(frame: pd.DataFrame, rna: np.ndarray, drug: np.ndarray) -> np.ndarray:
    rows = frame["local_rna_row_index"].to_numpy(dtype=np.int64)
    drug_rows = frame["drug_feature_row_index"].to_numpy(dtype=np.int64)
    x = np.concatenate((rna[rows], drug[drug_rows]), axis=1)
    if x.shape[1] != 3026 or not np.isfinite(x).all():
        raise ValueError("direct AAC feature block must be finite exact978 RNA + exact2048 Morgan")
    return x


def _fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
                 *, seed: int, arm: str, alpha: float) -> tuple[np.ndarray, dict[str, Any]]:
    if train_x.ndim != 2 or test_x.ndim != 2 or train_x.shape[1] != test_x.shape[1]:
        raise ValueError(f"{arm} train/test feature axes do not match")
    if not np.isfinite(train_x).all() or not np.isfinite(test_x).all() or not np.isfinite(train_y).all():
        raise ValueError(f"{arm} inputs contain nonfinite values")
    x_scaler = StandardScaler().fit(train_x)
    train_z = x_scaler.transform(train_x)
    test_z = x_scaler.transform(test_x)
    model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr", tol=1e-6)
    model.fit(train_z, train_y)
    prediction = np.asarray(model.predict(test_z), dtype=np.float64).reshape(-1)
    if not np.isfinite(prediction).all():
        raise ValueError(f"{arm} produced nonfinite AAC predictions")
    receipt = {
        "arm": arm,
        "estimator": f"sklearn.linear_model.Ridge(alpha={alpha}, solver=lsqr, tol=1e-6, fit_intercept=true)",
        "standardizer": "StandardScaler fitted on the corresponding outer TRAIN rows only",
        "feature_count": int(train_x.shape[1]),
        "training_pair_count": int(len(train_y)),
        "prediction_pair_count": int(len(test_x)),
        "seed": int(seed),
    }
    return prediction, receipt


def _metrics_by_object(frame: pd.DataFrame, y: np.ndarray, predictions: Mapping[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for patient, idx in frame.groupby("source_object_id", sort=True).indices.items():
        ix = np.asarray(idx, dtype=np.int64)
        row: dict[str, Any] = {
            "source_object_id": str(patient),
            "fold_id": int(frame.iloc[ix]["fold_id"].iloc[0]),
            "n_pairs": int(len(ix)),
            "n_drugs": int(frame.iloc[ix]["full_inchikey"].nunique()),
        }
        for arm, pred in predictions.items():
            error = np.asarray(pred, dtype=np.float64)[ix] - y[ix]
            row[f"{arm}__rmse"] = float(np.sqrt(np.mean(error * error))) if np.isfinite(error).all() else None
            per_cell_rho: list[float] = []
            centered_directions: list[float] = []
            obj_frame = frame.iloc[ix]
            for _, cell_indices in obj_frame.groupby("cellid", sort=True).indices.items():
                cix = ix[np.asarray(cell_indices, dtype=np.int64)]
                if len(cix) >= 2 and np.ptp(y[cix]) > 0 and np.ptp(pred[cix]) > 0:
                    rho = spearmanr(y[cix], pred[cix]).statistic
                    if np.isfinite(rho):
                        per_cell_rho.append(float(rho))
                if len(cix) >= 2:
                    y_centered = y[cix] - np.mean(y[cix])
                    pred_centered = pred[cix] - np.mean(pred[cix])
                    centered_directions.append(float(np.mean(np.sign(y_centered) == np.sign(pred_centered))))
            row[f"{arm}__within_cell_drug_spearman"] = float(np.mean(per_cell_rho)) if per_cell_rho else None
            row[f"{arm}__spearman_cell_denominator"] = int(len(per_cell_rho))
            row[f"{arm}__within_cell_centered_directional_agreement"] = (
                float(np.mean(centered_directions)) if centered_directions else None
            )
        rows.append(row)
    object_metrics = pd.DataFrame(rows)
    summary: dict[str, Any] = {}
    for arm in predictions:
        rmse = pd.to_numeric(object_metrics[f"{arm}__rmse"], errors="coerce")
        rho = pd.to_numeric(object_metrics[f"{arm}__within_cell_drug_spearman"], errors="coerce")
        summary[arm] = {
            "patient_macro_rmse": float(rmse.mean()),
            "patient_macro_rmse_denominator": int(rmse.notna().sum()),
            "patient_macro_within_cell_drug_spearman": float(rho.mean()) if rho.notna().any() else None,
            "patient_spearman_denominator": int(rho.notna().sum()),
            "patient_macro_within_cell_centered_directional_agreement": (
                float(pd.to_numeric(object_metrics[
                    f"{arm}__within_cell_centered_directional_agreement"
                ], errors="coerce").mean())
                if pd.to_numeric(object_metrics[
                    f"{arm}__within_cell_centered_directional_agreement"
                ], errors="coerce").notna().any() else None
            ),
            "patient_count": int(len(object_metrics)),
            "pair_count": int(len(object_metrics) and object_metrics["n_pairs"].sum()),
        }
    return object_metrics, summary


def _metrics_by_drug(frame: pd.DataFrame, y: np.ndarray,
                     predictions: Mapping[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, positions in frame.groupby("full_inchikey", sort=True).indices.items():
        ix = np.asarray(positions, dtype=np.int64)
        row: dict[str, Any] = {
            "full_inchikey": str(key),
            "pair_count": int(len(ix)),
            "object_count": int(frame.iloc[ix]["source_object_id"].nunique()),
        }
        for arm, pred in predictions.items():
            pred = np.asarray(pred, dtype=np.float64)
            per_object_rmse: list[float] = []
            per_object_mae: list[float] = []
            for _, local in frame.iloc[ix].groupby("source_object_id", sort=True).indices.items():
                obj_ix = ix[np.asarray(local, dtype=np.int64)]
                err = pred[obj_ix] - y[obj_ix]
                if np.isfinite(err).all():
                    per_object_rmse.append(float(np.sqrt(np.mean(err * err))))
                    per_object_mae.append(float(np.mean(np.abs(err))))
            row[f"{arm}__patient_macro_rmse"] = float(np.mean(per_object_rmse)) if per_object_rmse else None
            row[f"{arm}__patient_macro_mae"] = float(np.mean(per_object_mae)) if per_object_mae else None
            row[f"{arm}__valid_object_denominator"] = int(len(per_object_rmse))
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_bootstrap(object_metrics: pd.DataFrame, arm: str, reference: str,
                      seed: int, iterations: int = 2000) -> dict[str, Any]:
    a = pd.to_numeric(object_metrics[f"{reference}__rmse"], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(object_metrics[f"{arm}__rmse"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    improvement = a[valid] - b[valid]
    if len(improvement) < 2:
        return {"objects": int(len(improvement)), "mean_absolute_rmse_reduction": None,
                "ci95": [None, None], "mean_relative_rmse_reduction": None}
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(improvement), size=(iterations, len(improvement)))
    absolute = improvement[samples].mean(axis=1)
    base_mean = a[valid].mean()
    base_samples = a[valid][samples].mean(axis=1)
    relative = absolute / base_samples if np.all(base_samples > 0) else np.full(iterations, np.nan)
    return {
        "objects": int(len(improvement)),
        "mean_absolute_rmse_reduction": float(improvement.mean()),
        "ci95": [float(np.quantile(absolute, .025)), float(np.quantile(absolute, .975))],
        "mean_relative_rmse_reduction": float(improvement.mean() / base_mean) if base_mean > 0 else None,
        "relative_ci95": [float(np.nanquantile(relative, .025)), float(np.nanquantile(relative, .975))]
        if np.isfinite(relative).any() else [None, None],
        "bootstrap_iterations": int(iterations),
        "bootstrap_unit": "PatientID, paired within object; not patient-population inference",
    }


def _nested_partition_frame(payload: Mapping[str, Any], partition: str) -> pd.DataFrame:
    """Convert the EXP-066 array block to a row table without flattening provenance maps."""
    block = payload.get(partition)
    if isinstance(block, pd.DataFrame):
        return block.reset_index(drop=True).copy()
    if not isinstance(block, Mapping):
        raise TypeError(f"nested {partition} response block must be a mapping of keyed arrays")
    columns = (
        "pair_id", "source_object_id", "lincs_cell_id", "full_inchikey",
        "drug_feature_row_index", "log10_dose_um", "log10_time_hours",
        "nested_fold_id", "model_source", "prediction", "shuffled_prediction",
        "shuffle_donor_object_id", "shuffle_donor_lincs_cell_id",
        "shuffle_donor_depmap_model_id",
    )
    missing = set(columns) - set(block)
    if missing:
        raise ValueError(f"nested {partition} response block lacks fields: {sorted(missing)}")
    arrays = {column: np.asarray(block[column]) for column in columns}
    vector_columns = {"prediction", "shuffled_prediction"}
    row_counts: set[int] = set()
    for column, value in arrays.items():
        if column in vector_columns:
            if value.ndim != 2 or value.shape[1] != 978:
                raise ValueError(f"nested {partition} {column} must have shape [rows,978]")
            row_counts.add(int(value.shape[0]))
        else:
            if value.ndim != 1:
                raise ValueError(f"nested {partition} {column} must be a one-dimensional keyed array")
            row_counts.add(int(len(value)))
    if len(row_counts) != 1:
        raise ValueError(f"nested {partition} keyed arrays have inconsistent row counts")
    frame = pd.DataFrame({
        column: value for column, value in arrays.items() if column not in vector_columns
    })
    for column in vector_columns:
        # Keep each 978-gene prediction as one row value. Constructing a DataFrame
        # directly from the 2-D block would incorrectly treat genes as columns.
        frame[column] = pd.Series([row.copy() for row in arrays[column]], dtype=object)
    return frame.loc[:, columns]


def _validate_nested_patient_provenance(
    nested: Mapping[str, Any], pair_frame: pd.DataFrame, outer_fold: int,
    expected_train: set[str], expected_predict: set[str],
    response_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact outer/inner PatientID membership before using any predicted response."""
    if int(nested.get("outer_fold_id", -1)) != int(outer_fold):
        raise ValueError("nested response receipt belongs to a different outer PatientID fold")
    if set(map(str, nested.get("outer_train_patient_ids", []))) != expected_train:
        raise ValueError("nested response model did not use the exact outer TRAIN PatientID set")
    if set(map(str, nested.get("outer_predict_patient_ids", []))) != expected_predict:
        raise ValueError("nested response model did not predict the exact outer held-out PatientID set")
    expected_axis = str(response_config.get("target_axis_hash", ""))
    if (str(nested.get("target_axis_hash", "")) != expected_axis
            or len(nested.get("target_gene_ids", [])) != 978
            or len(set(map(str, nested.get("target_gene_ids", [])))) != 978):
        raise ValueError("nested response receipt does not bind the frozen unique 978-gene axis")

    fold_map = pair_frame[["source_object_id", "fold_id"]].drop_duplicates()
    if fold_map["source_object_id"].duplicated().any():
        raise ValueError("B pair table assigns a PatientID to multiple frozen folds")
    fold_by_patient = fold_map.set_index("source_object_id")["fold_id"].astype(int).to_dict()
    inner = nested.get("inner_fold_provenance")
    expected_inner_folds = set(range(5)) - {int(outer_fold)}
    if not isinstance(inner, list) or {int(item.get("inner_validation_fold_id", -1)) for item in inner} != expected_inner_folds:
        raise ValueError("nested response receipt must contain each of the four fixed inner folds once")
    seen_prediction: set[str] = set()
    validated_inner: list[dict[str, Any]] = []
    for item in inner:
        inner_fold = int(item["inner_validation_fold_id"])
        expected_inner_predict = {patient for patient in expected_train if fold_by_patient.get(patient) == inner_fold}
        expected_inner_train = expected_train - expected_inner_predict
        train_ids = set(map(str, item.get("training_patient_ids", [])))
        predict_ids = set(map(str, item.get("prediction_patient_ids", [])))
        if train_ids != expected_inner_train or predict_ids != expected_inner_predict:
            raise ValueError(f"inner response model {inner_fold} used an incorrect PatientID training/prediction set")
        if train_ids & (predict_ids | expected_predict):
            raise ValueError(f"inner response model {inner_fold} includes a held-out PatientID")
        if seen_prediction & predict_ids:
            raise ValueError("an outer-TRAIN PatientID is predicted by more than one inner fold")
        seen_prediction |= predict_ids
        validated_inner.append({
            "inner_validation_fold_id": inner_fold,
            "training_patient_ids": sorted(train_ids),
            "prediction_patient_ids": sorted(predict_ids),
            "training_condition_count": int(item.get("training_condition_count", -1)),
            "prediction_pair_count": int(item.get("prediction_pair_count", -1)),
            "cpu_seconds": float(item.get("cpu_seconds", 0.0)),
            "model_source": str(item.get("model_source", "")),
            "models": item.get("models", {}),
        })
    if seen_prediction != expected_train:
        raise ValueError("the four inner folds do not predict each outer-TRAIN PatientID exactly once")
    outer_model = nested.get("outer_model_provenance")
    if not isinstance(outer_model, Mapping):
        raise ValueError("nested response receipt is missing outer-model provenance")
    if (set(map(str, outer_model.get("training_patient_ids", []))) != expected_train
            or set(map(str, outer_model.get("prediction_patient_ids", []))) != expected_predict):
        raise ValueError("outer response model training/prediction PatientID sets are incorrect")
    return {
        "outer_fold_id": int(outer_fold),
        "outer_train_patient_ids": sorted(expected_train),
        "outer_predict_patient_ids": sorted(expected_predict),
        "inner_fold_provenance": validated_inner,
        "outer_model_provenance": {
            "training_patient_ids": sorted(map(str, outer_model["training_patient_ids"])),
            "prediction_patient_ids": sorted(map(str, outer_model["prediction_patient_ids"])),
            "training_condition_count": int(outer_model.get("training_condition_count", -1)),
            "prediction_pair_count": int(outer_model.get("prediction_pair_count", -1)),
            "cpu_seconds": float(outer_model.get("cpu_seconds", 0.0)),
            "models": outer_model.get("models", {}),
        },
        "target_axis_hash": expected_axis,
        "target_gene_count": 978,
        "nested_response_cost": nested.get("cost", {}),
    }


def _build_fold_predictions(domain: Mapping[str, Any], pair_frame: pd.DataFrame,
                            rna: np.ndarray, drug: np.ndarray,
                            raw_config: Mapping[str, Any], fold: int,
                            response_config: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    train_patients = sorted(pair_frame.loc[pair_frame["fold_id"].ne(fold), "source_object_id"].unique())
    test_patients = sorted(pair_frame.loc[pair_frame["fold_id"].eq(fold), "source_object_id"].unique())
    if set(train_patients) & set(test_patients):
        raise ValueError("outer PatientID groups overlap")
    nested_meta = pair_frame[[
        "pair_id", "source_object_id", "cell_id", "lincs_cell_id", "rna_row_index",
        "full_inchikey", "drug_feature_row_index", "log10_dose_um", "log10_time_hours", "fold_id",
    ]].rename(columns={"cell_id": "cell_id"}).copy()
    nested = generate_nested_oof_response_features(
        domain=domain,
        train_patient_ids=train_patients,
        predict_patient_ids=test_patients,
        outer_fold_id=int(fold),
        config=response_config,
        prediction_metadata=nested_meta,
    )
    # The nested response model is the source of all treatment-response columns.
    if not {"train", "predict"}.issubset(nested):
        raise ValueError("nested EXP-066 API must return separate train and heldout pair tables")
    train_meta = pair_frame.loc[pair_frame["fold_id"].ne(fold)].copy().set_index("pair_id", drop=False)
    test_meta = pair_frame.loc[pair_frame["fold_id"].eq(fold)].copy().set_index("pair_id", drop=False)
    provenance_receipt = _validate_nested_patient_provenance(
        nested, pair_frame, fold,
        set(train_patients), set(test_patients), response_config,
    )
    train_nested = _nested_partition_frame(nested, "train")
    predict_nested = _nested_partition_frame(nested, "predict")
    if list(train_nested["pair_id"].astype(str)) != train_meta.index.astype(str).tolist():
        raise ValueError("nested OOF pair identity/order does not exactly match outer TRAIN AAC rows")
    if list(predict_nested["pair_id"].astype(str)) != test_meta.index.astype(str).tolist():
        raise ValueError("nested heldout response predictions do not exactly match outer TEST AAC rows")
    for nested_frame, expected_frame, partition_name in (
        (train_nested, train_meta.reset_index(drop=True), "outer TRAIN"),
        (predict_nested, test_meta.reset_index(drop=True), "outer TEST"),
    ):
        for column in ("source_object_id", "lincs_cell_id", "full_inchikey"):
            if nested_frame[column].astype(str).tolist() != expected_frame[column].astype(str).tolist():
                raise ValueError(f"nested {partition_name} identity/key differs for {column}")
        expected_fold = expected_frame["fold_id"].to_numpy(dtype=np.int64)
        nested_fold = pd.to_numeric(nested_frame["nested_fold_id"], errors="raise").to_numpy(dtype=np.int64)
        if not np.array_equal(nested_fold, expected_fold):
            raise ValueError(f"nested {partition_name} rows were not predicted by their frozen PatientID fold")
        if nested_frame["shuffle_donor_object_id"].astype(str).eq(
            nested_frame["source_object_id"].astype(str)
        ).any():
            raise ValueError(f"nested {partition_name} shuffled donor equals the target object")
        patient_fold = pair_frame[["source_object_id", "fold_id"]].drop_duplicates().set_index(
            "source_object_id")["fold_id"].astype(int)
        donor_fold = nested_frame["shuffle_donor_object_id"].astype(str).map(patient_fold)
        if donor_fold.isna().any() or not np.array_equal(
            donor_fold.to_numpy(dtype=np.int64), nested_fold,
        ):
            raise ValueError(f"nested {partition_name} shuffle donor crossed its response-blind fold partition")
    x = _direct_matrix(pair_frame, rna, drug)
    row_index = {str(pair_id): i for i, pair_id in enumerate(pair_frame["pair_id"].astype(str))}
    train_ix = np.asarray([row_index[str(value)] for value in train_nested["pair_id"]], dtype=np.int64)
    test_ix = np.asarray([row_index[str(value)] for value in predict_nested["pair_id"]], dtype=np.int64)
    y = pair_frame["aac_published"].to_numpy(dtype=np.float64)
    ordinary_train = np.stack(train_nested["prediction"].to_numpy())
    ordinary_test = np.stack(predict_nested["prediction"].to_numpy())
    shuffled_train = np.stack(train_nested["shuffled_prediction"].to_numpy())
    shuffled_test = np.stack(predict_nested["shuffled_prediction"].to_numpy())
    for name, matrix, expected in (
        ("train OOF", ordinary_train, len(train_ix)), ("outer test", ordinary_test, len(test_ix)),
        ("train shuffled OOF", shuffled_train, len(train_ix)), ("outer shuffled", shuffled_test, len(test_ix)),
    ):
        if matrix.shape != (expected, 978) or not np.isfinite(matrix).all():
            raise ValueError(f"{name} response feature matrix is not finite [n,978]")
    train_y, test_y = y[train_ix], y[test_ix]
    train_x, test_x = x[train_ix], x[test_ix]
    # Fold-local, fixed random Fourier expansion supplies 978 additional direct-only degrees of freedom.
    direct_scaler = StandardScaler().fit(train_x)
    train_standardized = direct_scaler.transform(train_x)
    test_standardized = direct_scaler.transform(test_x)
    rff_spec = raw_config["models"]["capacity_match_random_fourier"]
    if int(rff_spec["n_components"]) != 978 or not np.isclose(
        float(rff_spec["gamma"]), 1.0 / float(train_x.shape[1]), rtol=0, atol=1e-16,
    ):
        raise ValueError("frozen direct-only RFF must add 978 dimensions with gamma=1/3026")
    rff = RBFSampler(gamma=float(rff_spec["gamma"]), n_components=978,
                     random_state=int(raw_config["models"]["capacity_match_seed"]))
    rff_train = rff.fit_transform(train_standardized)
    rff_test = rff.transform(test_standardized)
    train_features = {
        ARMS[0]: train_x,
        ARMS[1]: np.concatenate((train_x, ordinary_train), axis=1),
        ARMS[2]: np.concatenate((train_x, rff_train), axis=1),
        ARMS[3]: np.concatenate((train_x, shuffled_train), axis=1),
    }
    test_features = {
        ARMS[0]: test_x,
        ARMS[1]: np.concatenate((test_x, ordinary_test), axis=1),
        ARMS[2]: np.concatenate((test_x, rff_test), axis=1),
        ARMS[3]: np.concatenate((test_x, shuffled_test), axis=1),
    }
    predictions: dict[str, np.ndarray] = {}
    receipts: list[dict[str, Any]] = []
    for arm in ARMS:
        pred, receipt = _fit_predict(train_features[arm], train_y, test_features[arm],
                                     seed=int(raw_config["splits"]["seed"]), arm=arm,
                                     alpha=float(raw_config["models"]["alpha"]))
        predictions[arm] = pred
        receipts.append(receipt)
    test_rows = pair_frame.iloc[test_ix].copy().reset_index(drop=True)
    fold_detail = {
        "fold_id": int(fold), "train_patient_count": len(train_patients),
        "test_patient_count": len(test_patients), "train_pair_count": len(train_ix),
        "test_pair_count": len(test_ix),
        "nested_response_receipt": provenance_receipt,
        "train_shuffle_donor_ids": train_nested[["pair_id", "source_object_id", "nested_fold_id", "shuffle_donor_object_id", "shuffle_donor_lincs_cell_id", "shuffle_donor_depmap_model_id"]].to_dict("records"),
        "heldout_shuffle_donor_ids": predict_nested[["pair_id", "source_object_id", "nested_fold_id", "shuffle_donor_object_id", "shuffle_donor_lincs_cell_id", "shuffle_donor_depmap_model_id"]].to_dict("records"),
        "models": receipts,
    }
    return predictions, receipts, {"test_indices": test_ix, "test_frame": test_rows, "test_y": test_y,
                                   "train_y": train_y, "fold_detail": fold_detail}


def run_exp067(config_path: Path, run_id: str,
               resume_fold0_from_run_id: str | None = None) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw.get("experiment_id") != "EXP-067" or raw.get("metadata_gate_status") != "PASS":
        raise PermissionError("EXP-067 requires the frozen root-readback metadata PASS")
    if float(raw["models"]["alpha"]) != 1.0:
        raise ValueError("EXP-067 Ridge alpha is frozen at 1.0")
    if raw.get("metadata_gate_receipt_sha256") != "dc4b649d960a326916bb53c1734c0204f67bd877e4cce9467f35a255779633c9":
        raise PermissionError("EXP-067 config is not bound to the accepted shared metadata receipt")
    if resume_fold0_from_run_id not in (None, FOLD0_REUSE_RUN_ID):
        raise ValueError("only the fixed independently verified v3 fold-0 artifact may be reused")
    resources = raw["resources"]
    b_preflight_configured, b_preflight = _effective_cpu_debit(
        resources, "b_package_cpu_spent_before_seconds", KNOWN_B_PACKAGE_CPU_DEBIT_SECONDS,
    )
    shared_preflight_configured, shared_preflight = _effective_cpu_debit(
        resources, "a_b_cpu_spent_before_seconds", KNOWN_SHARED_AB_CPU_DEBIT_SECONDS,
    )
    package_cap = _optional_cpu_cap(resources.get("cpu_seconds_package_cap"), "cpu_seconds_package_cap")
    shared_cap = _optional_cpu_cap(
        resources.get("cpu_seconds_shared_AB_absolute_cap"), "cpu_seconds_shared_AB_absolute_cap",
    )
    historical_package_cap = _optional_cpu_cap(
        resources.get("historical_cpu_seconds_package_cap", resources.get("cpu_seconds_package_cap")),
        "historical_cpu_seconds_package_cap",
    )
    historical_shared_cap = _optional_cpu_cap(
        resources.get("historical_cpu_seconds_shared_AB_absolute_cap",
                      resources.get("cpu_seconds_shared_AB_absolute_cap")),
        "historical_cpu_seconds_shared_AB_absolute_cap",
    )
    root = Path(__file__).resolve().parents[2]
    shared = root / raw["shared_contract"]
    response_config = yaml.safe_load((root / "configs/experiments/EXP-066-response-prediction.yaml").read_text(encoding="utf-8"))
    if response_config.get("metadata_gate_receipt_sha256") != raw["metadata_gate_receipt_sha256"]:
        raise PermissionError("EXP-066 and EXP-067 metadata receipts differ")
    destination = root / raw["outputs"]["root_dir"] / run_id
    if destination.exists():
        raise FileExistsError(f"EXP-067 immutable run directory already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    start_cpu = time.process_time()
    loader = load_frozen_response_domain(
        shared, read_lincs_targets=True, read_ctrp_labels=True,
    )
    loader_cpu = float(loader["load_cpu_seconds"])
    response_config = _bind_measured_response_loader_cpu(response_config, loader_cpu)
    pair_frame, pair_audit = _load_pair_frame(loader, shared)
    rna, drug, pair_frame = _pair_inputs(loader, pair_frame)
    _write_json(destination / "pair_domain_audit.json", pair_audit)
    y = pair_frame["aac_published"].to_numpy(dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("AAC labels must be finite after the frozen metadata join")
    elapsed_start = time.process_time()
    fold_reference_seconds: list[float] = []
    fold_summaries: list[dict[str, Any]] = []
    oof = {arm: np.full(len(pair_frame), np.nan, dtype=np.float64) for arm in ARMS}
    cost_gates: list[dict[str, Any]] = []
    for fold in range(5):
        fold_started = time.process_time()
        reused_from = None
        if fold == 0 and resume_fold0_from_run_id is not None:
            source_run_dir = root / raw["outputs"]["root_dir"] / resume_fold0_from_run_id
            predictions, cached_fold_detail, historical_fold_seconds = _load_reusable_fold0(
                source_run_dir, pair_frame.loc[pair_frame["fold_id"].eq(0)].reset_index(drop=True),
            )
            test_ix = np.flatnonzero(pair_frame["fold_id"].to_numpy(dtype=np.int64) == 0)
            train_ix = np.flatnonzero(pair_frame["fold_id"].to_numpy(dtype=np.int64) != 0)
            details = {"test_indices": test_ix, "test_y": y[test_ix], "train_y": y[train_ix],
                       "fold_detail": cached_fold_detail}
            reused_from = resume_fold0_from_run_id
        else:
            predictions, _, details = _build_fold_predictions(loader, pair_frame, rna, drug, raw, fold, response_config)
        test_ix = details["test_indices"]
        for arm in ARMS:
            oof[arm][test_ix] = predictions[arm]
        fold_frame = pair_frame.iloc[test_ix].copy().reset_index(drop=True)
        obj_metrics, metric_summary = _metrics_by_object(fold_frame, details["test_y"], predictions)
        drug_metrics = _metrics_by_drug(fold_frame, details["test_y"], predictions)
        pred_table = fold_frame[["pair_id", "source_object_id", "cellid", "full_inchikey", "fold_id", "aac_published"]].copy()
        for arm in ARMS:
            pred_table[f"prediction__{arm}"] = predictions[arm]
        fold_path = destination / f"fold_{fold}_pair_predictions.tsv"
        pred_table.to_csv(fold_path, sep="\t", index=False, lineterminator="\n")
        reread = pd.read_csv(fold_path, sep="\t", dtype={"pair_id": "string", "full_inchikey": "string"})
        if reread["pair_id"].astype(str).tolist() != pred_table["pair_id"].astype(str).tolist():
            raise IOError("EXP-067 fold prediction row keys changed on independent readback")
        for arm in ARMS:
            if not np.allclose(reread[f"prediction__{arm}"].to_numpy(float), predictions[arm], rtol=0, atol=1e-12):
                raise IOError(f"EXP-067 fold prediction readback differs for {arm}")
        obj_path = destination / f"fold_{fold}_object_metrics.tsv"
        obj_metrics.to_csv(obj_path, sep="\t", index=False, lineterminator="\n")
        drug_path = destination / f"fold_{fold}_drug_metrics.tsv"
        drug_metrics.to_csv(drug_path, sep="\t", index=False, lineterminator="\n")
        _write_json(destination / f"fold_{fold}_summary.json", {
            "fold_id": fold, "metrics": metric_summary, "fold": details["fold_detail"],
            "prediction_sha256": _sha256(fold_path), "object_metrics_sha256": _sha256(obj_path),
            "drug_metrics_sha256": _sha256(drug_path),
        })
        elapsed = time.process_time() - fold_started
        fold_summaries.append({"fold_id": fold, "metrics": metric_summary,
                               "fold": details["fold_detail"], "prediction_path": _portable_output_path(fold_path, root),
                               "object_metrics_path": _portable_output_path(obj_path, root),
                               "drug_metrics_path": _portable_output_path(drug_path, root),
                               "reused_from_run_id": reused_from})
        total_since_fit = time.process_time() - elapsed_start
        total_since_package_start = time.process_time() - start_cpu
        fold_reference_seconds.append(historical_fold_seconds if reused_from is not None else elapsed)
        mean_fold = float(np.mean(fold_reference_seconds))
        readback_allowance = float(raw["resources"]["root_readback_cpu_allowance_seconds"])
        projected, projected_package_total = _project_package_cpu_seconds(
            b_preflight, total_since_package_start, fold_reference_seconds, 4 - fold, readback_allowance,
        )
        projected_shared_total = shared_preflight + projected
        gate = {
            "status": "INFORMATIONAL_ONLY",
            "cpu_budget_enforcement": False,
            "run_id": run_id, "measured_outer_fold": fold,
            "measured_fold_cpu_seconds": elapsed, "completed_outer_folds": fold + 1,
            "projection_reference_fold_cpu_seconds": mean_fold,
            "reused_fold0_from_run_id": reused_from,
            "required_outer_folds": 5, "shared_loader_cpu_seconds": loader_cpu,
            "projected_model_and_closeout_cpu_seconds_with_20pct_margin": projected,
            "b_package_cpu_spent_before_seconds": b_preflight,
            "projected_total_package_cpu_seconds_including_preflight_and_margin": projected_package_total,
            "package_cpu_cap_seconds": package_cap,
            "shared_AB_cpu_cap_seconds": shared_cap,
            "historical_package_cpu_cap_seconds": historical_package_cap,
            "historical_shared_AB_cpu_cap_seconds": historical_shared_cap,
            "configured_b_package_cpu_spent_before_seconds": b_preflight_configured,
            "b_package_cpu_spent_before_seconds": b_preflight,
            "configured_a_b_cpu_spent_before_seconds": shared_preflight_configured,
            "a_b_cpu_spent_before_seconds": shared_preflight,
            "projected_shared_AB_cumulative_cpu_seconds": projected_shared_total,
            "historical_package_cap_exceeded_information_only": (
                projected_package_total > historical_package_cap if historical_package_cap is not None else None
            ),
            "historical_shared_AB_cap_exceeded_information_only": (
                projected_shared_total > historical_shared_cap if historical_shared_cap is not None else None
            ),
            "projected_remaining_shared_AB_seconds": None,
            "root_readback_cpu_allowance_seconds": readback_allowance,
            "fold_denominator": {"training_pairs": len(details["train_y"]), "test_pairs": len(details["test_y"]),
                                 "training_objects": details["fold_detail"]["train_patient_count"],
                                 "test_objects": details["fold_detail"]["test_patient_count"]},
            "all_arms_and_nested_models_included": True,
        }
        cost_gates.append(gate)
        _write_json(destination / ("first_outer_fold_cost_gate.json" if fold == 0 else f"cost_gate_after_fold_{fold+1}.json"), gate)

    complete = len(fold_summaries) == 5 and all(np.isfinite(oof[arm]).all() for arm in ARMS)
    if complete:
        obj_metrics, overall = _metrics_by_object(pair_frame, y, oof)
        obj_path = destination / "oof_object_metrics.tsv"
        obj_metrics.to_csv(obj_path, sep="\t", index=False, lineterminator="\n")
        drug_metrics = _metrics_by_drug(pair_frame, y, oof)
        drug_path = destination / "oof_drug_metrics.tsv"
        drug_metrics.to_csv(drug_path, sep="\t", index=False, lineterminator="\n")
        summary_metrics = overall
        paired = {}
        for arm in (ARMS[1], ARMS[2], ARMS[3]):
            for reference in (ARMS[0], ARMS[2]):
                paired[f"{arm}_vs_{reference}"] = _paired_bootstrap(
                    obj_metrics, arm, reference, seed=int(raw["metrics"]["bootstrap_seed"]),
                    iterations=int(raw["metrics"]["bootstrap_iterations"]),
                )
        improvement_fraction = {}
        for comparator in (ARMS[0], ARMS[2]):
            reference = obj_metrics[f"{comparator}__rmse"].to_numpy(dtype=float)
            candidate = obj_metrics[f"{ARMS[1]}__rmse"].to_numpy(dtype=float)
            improvement_fraction[comparator] = float(np.mean(candidate < reference))
        comparisons = [paired[f"{ARMS[1]}_vs_{reference}"] for reference in (ARMS[0], ARMS[2])]
        gate_pass = all(
            result["mean_relative_rmse_reduction"] is not None
            and result["mean_relative_rmse_reduction"] >= .05
            and result["ci95"][0] > 0
            for result in comparisons
        ) and (
            improvement_fraction[ARMS[0]] >= .60
            and improvement_fraction[ARMS[2]] >= .60
            and summary_metrics[ARMS[1]]["patient_macro_rmse"] < summary_metrics[ARMS[3]]["patient_macro_rmse"]
        )
        status = "COMPLETE_CANDIDATE_PASS" if gate_pass else "COMPLETE_CANDIDATE_CLOSED"
        all_predictions = pair_frame[["pair_id", "source_object_id", "cellid", "full_inchikey", "fold_id", "aac_published"]].copy()
        for arm in ARMS:
            all_predictions[f"prediction__{arm}"] = oof[arm]
        all_predictions.to_csv(destination / "oof_predictions.tsv", sep="\t", index=False, lineterminator="\n")
        if not np.array_equal(pd.read_csv(destination / "oof_predictions.tsv", sep="\t")["pair_id"].astype(str).to_numpy(),
                              pair_frame["pair_id"].astype(str).to_numpy()):
            raise IOError("combined OOF prediction key readback failed")
    else:
        overall, paired, improvement_fraction, gate_pass = None, None, None, False
        status = "INCOMPLETE"
    total_cpu = time.process_time() - start_cpu
    summary = {
        "schema": "exp067_crossfitted_response_benefit_run_v1", "experiment_id": "EXP-067",
        "run_id": run_id, "status": status, "scientific_comparison_valid": bool(complete),
        "practical_gate_pass": bool(gate_pass), "metadata_gate_status": "PASS",
        "metadata_gate_receipt_sha256": raw["metadata_gate_receipt_sha256"],
        "condition_count": int(len(loader["condition_metadata"])),
        "pair_count": int(len(pair_frame)), "object_count": int(pair_frame["source_object_id"].nunique()),
        "drug_count": int(pair_frame["full_inchikey"].nunique()), "target_gene_count": 978,
        "pair_domain_audit": pair_audit,
        "outer_folds_completed": len(fold_summaries), "outer_folds_required": 5,
        "arms": list(ARMS), "folds": fold_summaries, "overall_metrics": overall,
        "paired_patient_bootstrap_vs_direct": paired, "improved_object_fraction_vs_direct_and_capacity_match": improvement_fraction,
        "bootstrap_protocol": {
            "seed": int(raw["metrics"]["bootstrap_seed"]),
            "iterations": int(raw["metrics"]["bootstrap_iterations"]),
            "unit": "paired PatientID objects; interval does not imply patient-population generalization",
            "interval": "percentile 95 percent bootstrap of absolute RMSE reduction",
        },
        "fit_gate_pass": bool(gate_pass),
        "aggregation": "AAC duplicate rows averaged within frozen CTRP cellid × drugid; scores aggregated within cell then equal PatientID",
        "feature_contract": {"direct": "DepMap exact978 pretreatment RNA + full-key Morgan2048",
            "response": "nested 4-fold PatientID OOF predicted LINCS treatment-minus-matched-control delta, exact978",
            "capacity_match": "direct 3026 standardized dimensions plus 978 RBFSampler RFF dimensions; gamma=1/3026, seed fixed in config",
            "ridge_alpha": 1.0, "rff_dimensions": 978, "rff_gamma": 1.0/3026.0},
        "read_roles": {"lincs_targets": "frozen 2,830-row exact978 source-development response targets only",
            "depmap": "exact978 pretreatment feature rows only", "ctrp": "metadata-eligible TRAIN+DEV AAC only",
            "pdo": "not read", "outer_locked": "not read", "panacea_cds_b": "not read"},
        "cost": {"loader_cpu_seconds": loader_cpu, "package_cpu_seconds": total_cpu,
            "package_cumulative_cpu_seconds": b_preflight + total_cpu,
            "package_preflight_cpu_seconds": b_preflight,
            "package_preflight_cpu_seconds_configured": b_preflight_configured,
            "shared_AB_cumulative_cpu_seconds": shared_preflight + total_cpu,
            "shared_AB_preflight_cpu_seconds": shared_preflight,
            "shared_AB_preflight_cpu_seconds_configured": shared_preflight_configured,
            "estimated_with_20pct_margin": cost_gates[-1]["projected_total_package_cpu_seconds_including_preflight_and_margin"] if cost_gates else None,
            "package_cpu_cap_seconds": package_cap, "shared_AB_cpu_cap_seconds": shared_cap,
            "budget_enforcement": "informational_only", "gpu_hours": 0.0},
        "cost_gates": cost_gates,
    }
    _write_json(destination / "run_summary.json", summary)
    summary_readback = json.loads((destination / "run_summary.json").read_text(encoding="utf-8"))
    if summary_readback != summary:
        raise IOError("EXP-067 summary JSON readback differs")
    _write_json(destination / "run_readback.json", {
        "status": "PASS_ROOT_INPUT_RUNNER_READBACK", "run_id": run_id,
        "run_summary_sha256": _sha256(destination / "run_summary.json"),
        "fold_prediction_sha256": {f"fold_{i}": _sha256(destination / f"fold_{i}_pair_predictions.tsv")
                                    for i in range(len(fold_summaries))},
        "metric_scope": "same full-key domain and object folds across all four arms",
    })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume-fold0-from-run-id", choices=[FOLD0_REUSE_RUN_ID],
                        help="read-only reuse of the root-verified v3 fold 0; not an independent test")
    args = parser.parse_args()
    config_path = args.config.resolve()
    started_cpu = time.process_time()
    started_wall = time.monotonic()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[2]
    output_root = root / raw["outputs"]["root_dir"]
    destination = output_root / args.run_id
    existed_before = destination.exists()
    try:
        cpu_threads = _configured_cpu_threads(raw.get("resources", {}))
        with threadpool_limits(limits=cpu_threads):
            summary = run_exp067(config_path, args.run_id, args.resume_fold0_from_run_id)
    except Exception as exc:
        # Preserve a failed immutable attempt without overwriting any pre-existing run.
        if not existed_before:
            destination.mkdir(parents=True, exist_ok=True)
            failure = {
                "schema": "exp067_failure_close_receipt_v1",
                "status": "FAILED_CLOSED",
                "experiment_id": "EXP-067",
                "run_id": args.run_id,
                "config_path": str(config_path),
                "metadata_gate_receipt_sha256": raw.get("metadata_gate_receipt_sha256"),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "process_cpu_seconds": float(time.process_time() - started_cpu),
                "wall_seconds": float(time.monotonic() - started_wall),
                "cpu_accounting": _failure_cpu_accounting(
                    raw.get("resources", {}), float(time.process_time() - started_cpu),
                ),
                "scientific_result_available": False,
                "gpu_hours": 0.0,
                "note": "Failure receipt only; no scientific score is implied.",
            }
            _write_json(destination / "failure_close_receipt.json", failure)
        raise
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] in {"COMPLETE_CANDIDATE_PASS", "COMPLETE_CANDIDATE_CLOSED"} else 2


def _failure_cpu_accounting(resources: Mapping[str, Any], process_cpu_seconds: float) -> dict[str, Any]:
    """Keep historical CPU charges visible even when a new immutable attempt fails."""
    try:
        configured_b, effective_b = _effective_cpu_debit(
            resources, "b_package_cpu_spent_before_seconds", KNOWN_B_PACKAGE_CPU_DEBIT_SECONDS,
        )
        configured_shared, effective_shared = _effective_cpu_debit(
            resources, "a_b_cpu_spent_before_seconds", KNOWN_SHARED_AB_CPU_DEBIT_SECONDS,
        )
    except (TypeError, ValueError) as exc:
        return {
            "status": "PRIOR_CPU_DEBIT_INVALID_UNRESOLVED",
            "error": str(exc),
            "process_cpu_seconds": process_cpu_seconds,
            "gpu_hours": 0.0,
        }
    return {
        "status": "INFORMATIONAL_ONLY",
        "configured_b_package_cpu_spent_before_seconds": configured_b,
        "effective_b_package_cpu_spent_before_seconds": effective_b,
        "configured_shared_AB_cpu_spent_before_seconds": configured_shared,
        "effective_shared_AB_cpu_spent_before_seconds": effective_shared,
        "process_cpu_seconds": process_cpu_seconds,
        "b_package_cumulative_cpu_seconds": effective_b + process_cpu_seconds,
        "shared_AB_cumulative_cpu_seconds": effective_shared + process_cpu_seconds,
        "cpu_budget_enforcement": False,
        "gpu_hours": 0.0,
    }


if __name__ == "__main__":
    raise SystemExit(main())
