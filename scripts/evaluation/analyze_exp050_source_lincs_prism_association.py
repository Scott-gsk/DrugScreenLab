#!/usr/bin/env python3
"""Descriptive source-domain LINCS perturbation vs PRISM association.

This EXP-050 analysis is a frozen external cell-line reference only. It does
not fit or select a drug-response model and does not evaluate organoid efficacy.
PRISM outcomes are scanned only for exact frozen train/validation sample IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_RUN_ID = "20261001_lincs_prism_transfer_association_v1"
MIN_ORIGINS = 10
FULL_INCHIKEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as stream:
        temp = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temp, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".parquet") as stream:
        temp = Path(stream.name)
    try:
        frame.to_parquet(temp, index=False)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def required_columns(actual: set[str], expected: set[str], label: str) -> None:
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def normalize_key(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.upper().replace("", pd.NA)


def read_contract(root: Path, config_path: Path, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    contract = config.get("source_lincs_prism_association_exploratory")
    if not isinstance(contract, dict) or contract.get("run_id") != run_id:
        raise ValueError("EXP-050 PRISM preregistration or run_id does not match")
    if contract.get("gpu_hours") != 0 or contract.get("status") not in {
        "preregistered_pending_execution", "running", "completed_exploratory"
    }:
        raise ValueError("PRISM association status or zero-GPU budget is invalid")
    for relative in contract["inputs"].values():
        # Some input values include a human-readable semicolon-delimited scale note.
        candidate = str(relative).split(";", 1)[0].strip()
        if candidate.endswith((".json", ".parquet")) and not (root / candidate).is_file():
            raise FileNotFoundError(root / candidate)
    return contract


def read_frozen_source_map(root: Path) -> tuple[pd.DataFrame, int]:
    identity_path = root / "artifacts/experiments/EXP-007/CCLE_CONTEXT_IDENTITY_SPLIT.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("status") != "IDENTITY_SPLIT_FROZEN":
        raise ValueError("EXP-007 response-blind identity split is not frozen")
    rows = pd.DataFrame(identity.get("identity_rows", []))
    required_columns(
        set(rows.columns),
        {"depmap_id", "patient_id", "expression_present", "join_status"},
        "frozen LINCS/DepMap identity crosswalk",
    )
    rows = rows.loc[
        rows["expression_present"].astype(bool)
        & rows["join_status"].eq("EXACT_STRIPPED_NAME_WITH_RNASEQ"),
        ["depmap_id", "patient_id"],
    ].dropna().drop_duplicates()
    if rows["depmap_id"].duplicated().any():
        raise ValueError("Frozen identity crosswalk has duplicate DepMap ModelIDs")

    split = pd.DataFrame(identity.get("split_assignments", []))
    required_columns(set(split.columns), {"depmap_id", "split_role"}, "frozen split assignments")
    split = split[["depmap_id", "split_role"]].dropna().drop_duplicates()
    if split["depmap_id"].duplicated().any():
        raise ValueError("Frozen response-blind split has duplicate DepMap ModelIDs")
    rows = rows.merge(split, on="depmap_id", how="inner", validate="one_to_one")
    if not rows["split_role"].isin(["train", "val", "locked_eval"]).all():
        raise ValueError("Frozen split contains an unrecognized role")

    models = pd.read_csv(
        root / "data/raw/depmap/24q2/Model.csv",
        usecols=["ModelID", "PatientID"], dtype="string",
    ).dropna(subset=["ModelID", "PatientID"]).drop_duplicates()
    if models["ModelID"].duplicated().any():
        raise ValueError("DepMap ModelID does not map uniquely to PatientID")
    check = rows.merge(models, left_on="depmap_id", right_on="ModelID", how="inner", validate="one_to_one")
    if len(check) != len(rows) or not check["patient_id"].astype(str).eq(check["PatientID"].astype(str)).all():
        raise ValueError("Frozen PatientID disagrees with DepMap 24Q2 Model.csv")
    rows = rows.rename(columns={"depmap_id": "model_id", "patient_id": "source_origin_id"})
    locked_count = int(rows["split_role"].eq("locked_eval").sum())
    return rows, locked_count


def read_prism_samples(root: Path, frozen: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    sample_path = root / "data/processed/exp012/prism_extended_primary_samples.parquet"
    schema = set(pq.ParquetFile(sample_path).schema_arrow.names)
    required_columns(schema, {"sample_id", "cell_line_id"}, "PRISM sample metadata")
    samples = pd.read_parquet(sample_path, columns=["sample_id", "cell_line_id"])
    samples["sample_id"] = normalize_key(samples["sample_id"])
    samples["cell_line_id"] = normalize_key(samples["cell_line_id"])
    if samples["sample_id"].isna().any() or samples["sample_id"].duplicated().any():
        raise ValueError("PRISM sample_id must be nonmissing and unique")
    exact = samples.merge(
        frozen, left_on="sample_id", right_on="model_id", how="inner", validate="one_to_one"
    )
    if not exact["cell_line_id"].eq(exact["model_id"]).all():
        raise ValueError("PRISM sample_id/cell_line_id is not the exact frozen DepMap ModelID")
    allowed = exact.loc[exact["split_role"].isin(["train", "val"])].copy()
    locked = exact.loc[exact["split_role"].eq("locked_eval")].copy()
    if set(allowed["sample_id"]) & set(locked["sample_id"]):
        raise ValueError("A PRISM sample_id appears in both allowed and locked roles")
    identity_summary = {
        "frozen_exact_identity_rows": int(len(frozen)),
        "frozen_train_identity_rows": int(frozen["split_role"].eq("train").sum()),
        "frozen_val_identity_rows": int(frozen["split_role"].eq("val").sum()),
        "frozen_locked_eval_identity_rows_excluded": int(frozen["split_role"].eq("locked_eval").sum()),
        "prism_exact_sample_matches": int(len(exact)),
        "prism_allowed_samples_present": int(len(allowed)),
        "prism_allowed_train_samples_present": int(allowed["split_role"].eq("train").sum()),
        "prism_allowed_val_samples_present": int(allowed["split_role"].eq("val").sum()),
        "prism_locked_eval_samples_present_and_excluded": int(len(locked)),
        "prism_allowed_samples_missing_from_bundle": int(
            frozen["split_role"].isin(["train", "val"]).sum() - len(allowed)
        ),
    }
    return allowed[["sample_id", "model_id", "source_origin_id", "split_role"]], identity_summary


def read_prism_drugs(root: Path) -> pd.DataFrame:
    path = root / "data/processed/exp012/prism_extended_primary_drugs.parquet"
    schema = set(pq.ParquetFile(path).schema_arrow.names)
    required_columns(schema, {"drug_id", "inchikey"}, "PRISM drug metadata")
    drugs = pd.read_parquet(path, columns=["drug_id", "inchikey"])
    drugs["drug_id"] = normalize_key(drugs["drug_id"])
    drugs["inchikey"] = normalize_key(drugs["inchikey"])
    drugs = drugs.dropna(subset=["drug_id", "inchikey"])
    drugs = drugs.loc[drugs["inchikey"].str.match(FULL_INCHIKEY)]
    id_count = drugs.groupby("drug_id")["inchikey"].nunique()
    key_count = drugs.groupby("inchikey")["drug_id"].nunique()
    drugs = drugs.loc[
        drugs["drug_id"].map(id_count).eq(1) & drugs["inchikey"].map(key_count).eq(1)
    ].drop_duplicates()
    if drugs["drug_id"].duplicated().any() or drugs["inchikey"].duplicated().any():
        raise ValueError("PRISM exact drug identity map is not one-to-one")
    return drugs


def read_frozen_lincs_features(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = root / "artifacts/experiments/EXP-050/20261001_lincs_ctrp_transfer_association_v3"
    per_drug_path = base / "per_drug_association.tsv"
    selected = pd.read_csv(
        per_drug_path, sep="\t", usecols=["inchikey", "dose_uM", "time_h"], dtype={"inchikey": "string"}
    )
    selected["inchikey"] = normalize_key(selected["inchikey"])
    selected["dose_uM"] = pd.to_numeric(selected["dose_uM"], errors="coerce")
    selected["time_h"] = pd.to_numeric(selected["time_h"], errors="coerce")
    selected = selected.dropna(subset=["inchikey", "dose_uM", "time_h"])
    if selected["inchikey"].duplicated().any():
        raise ValueError("Frozen CTRP-v3 LINCS exposure is not unique per exact compound")

    pairs_path = base / "patient_level_pairs.parquet"
    schema = set(pq.ParquetFile(pairs_path).schema_arrow.names)
    required_columns(
        schema,
        {"source_origin_id", "inchikey", "dose_uM", "time_h", "response_magnitude"},
        "frozen LINCS source features",
    )
    # Explicit projection excludes every CTRPv2 response/AAC field.
    lincs = pd.read_parquet(
        pairs_path,
        columns=["source_origin_id", "inchikey", "dose_uM", "time_h", "response_magnitude"],
    )
    lincs["source_origin_id"] = lincs["source_origin_id"].astype("string").str.strip()
    lincs["inchikey"] = normalize_key(lincs["inchikey"])
    for col in ["dose_uM", "time_h", "response_magnitude"]:
        lincs[col] = pd.to_numeric(lincs[col], errors="coerce")
    lincs = lincs.merge(
        selected, on=["inchikey", "dose_uM", "time_h"], how="inner", validate="many_to_one"
    )
    lincs = lincs.loc[np.isfinite(lincs["response_magnitude"])].copy()
    if lincs.duplicated(["source_origin_id", "inchikey"]).any():
        raise ValueError("Frozen LINCS feature is not unique per source-origin x compound")
    return selected, lincs


def scan_allowed_prism_rows(root: Path, allowed_sample_ids: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    pairs_path = root / "data/processed/exp012/prism_extended_primary_pairs.parquet"
    parquet_file = pq.ParquetFile(pairs_path)
    schema = set(parquet_file.schema_arrow.names)
    required_columns(schema, {"sample_id", "drug_id", "response"}, "PRISM response pairs")
    if not allowed_sample_ids:
        raise ValueError("No exact frozen train/validation PRISM sample IDs are available")

    # Filter by the frozen train/validation sample-ID set in Arrow Dataset before
    # converting any response rows to pandas/NumPy. Locked-eval labels are never
    # returned to numeric analysis, even though the source artifact has one row group.
    dataset = ds.dataset(str(pairs_path), format="parquet")
    predicate = ds.field("sample_id").isin(sorted(allowed_sample_ids))
    table = dataset.to_table(columns=["sample_id", "drug_id", "response"], filter=predicate)
    raw = table.to_pandas()
    raw["sample_id"] = normalize_key(raw["sample_id"])
    raw["drug_id"] = normalize_key(raw["drug_id"])
    returned_ids = set(raw["sample_id"].dropna().astype(str))
    unexpected = returned_ids - allowed_sample_ids
    if unexpected:
        raise ValueError("Arrow response scan returned a sample outside frozen train/validation IDs")
    response = pd.to_numeric(raw["response"], errors="coerce")
    raw["prism_minus_lfc_sensitivity"] = response
    finite = np.isfinite(raw["prism_minus_lfc_sensitivity"])
    raw = raw.loc[finite, ["sample_id", "drug_id", "prism_minus_lfc_sensitivity"]].copy()
    scan_summary = {
        "pairs_parquet_row_groups": int(parquet_file.metadata.num_row_groups),
        "pairs_parquet_rows_total_metadata_only": int(parquet_file.metadata.num_rows),
        "arrow_filter": "sample_id IN exact frozen EXP-007 train/val PRISM ModelIDs",
        "response_rows_returned_after_arrow_filter": int(len(table)),
        "finite_allowed_response_rows": int(len(raw)),
        "returned_sample_ids": int(len(returned_ids)),
        "unexpected_or_locked_sample_ids_returned": 0,
        "response_column": "EXP-012 cohort_index_manifest response=neg_extended_primary_lfc; higher_is_sensitive=true",
    }
    return raw, scan_summary


def calculate_associations(
    root: Path,
    allowed_samples: pd.DataFrame,
    drugs: pd.DataFrame,
    selected: pd.DataFrame,
    lincs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    allowed_ids = set(allowed_samples["sample_id"].astype(str))
    prism_rows, scan_summary = scan_allowed_prism_rows(root, allowed_ids)
    unexpected = set(prism_rows["sample_id"].astype(str)) - allowed_ids
    if unexpected:
        raise ValueError("Returned PRISM rows include a sample outside the allowed frozen roles")

    # Collapse replicate pairs within exact sample x drug, then cell lines sharing
    # the same DepMap PatientID into a single independent source-origin unit.
    prism_cell = (
        prism_rows.groupby(["sample_id", "drug_id"], as_index=False)
        .agg(
            prism_minus_lfc_sensitivity=("prism_minus_lfc_sensitivity", "median"),
            n_prism_response_rows=("prism_minus_lfc_sensitivity", "size"),
        )
        .merge(allowed_samples, on="sample_id", how="inner", validate="many_to_one")
        .merge(drugs, on="drug_id", how="inner", validate="many_to_one")
    )
    prism_origin = (
        prism_cell.groupby(["source_origin_id", "split_role", "inchikey"], as_index=False)
        .agg(
            prism_minus_lfc_sensitivity=("prism_minus_lfc_sensitivity", "median"),
            n_prism_cell_lines=("sample_id", "nunique"),
            n_prism_sample_drug_pairs=("sample_id", "size"),
            n_prism_response_rows=("n_prism_response_rows", "sum"),
        )
    )
    selected_keys = set(selected["inchikey"].astype(str))
    prism_origin = prism_origin.loc[prism_origin["inchikey"].isin(selected_keys)].copy()
    joined = prism_origin.merge(
        lincs,
        on=["source_origin_id", "inchikey"],
        how="inner",
        validate="one_to_one",
    )
    if joined.duplicated(["source_origin_id", "inchikey"]).any():
        raise ValueError("Joined source-origin x exact drug rows are not unique")
    if not joined["split_role"].isin(["train", "val"]).all():
        raise ValueError("Locked or unrecognized source role entered PRISM association output")

    summaries: list[dict[str, Any]] = []
    for inchikey, group in joined.groupby("inchikey", sort=True):
        n = int(group["source_origin_id"].nunique())
        rho: float | None = None
        reason = "below_minimum_source_origins" if n < MIN_ORIGINS else ""
        if n >= MIN_ORIGINS:
            x = group["response_magnitude"]
            y = group["prism_minus_lfc_sensitivity"]
            if x.nunique(dropna=True) < 2:
                reason = "constant_lincs_response_magnitude"
            elif y.nunique(dropna=True) < 2:
                reason = "constant_prism_sensitivity"
            else:
                value = x.corr(y, method="spearman")
                if pd.notna(value) and math.isfinite(float(value)):
                    rho = float(value)
                else:
                    reason = "spearman_undefined"
        dose_time = selected.loc[selected["inchikey"].eq(inchikey)].iloc[0]
        summaries.append({
            "inchikey": inchikey,
            "lincs_dose_uM": float(dose_time["dose_uM"]),
            "lincs_time_h": float(dose_time["time_h"]),
            "n_unique_depmap_patient_ids": n,
            "n_train_source_origins": int(group["source_origin_id"].nunique()),
            "n_val_source_origins": int(group.loc[group["split_role"].eq("val"), "source_origin_id"].nunique()),
            "n_prism_cell_lines": int(group["n_prism_cell_lines"].sum()),
            "spearman_rho_prism_minus_lfc_vs_lincs_mean_absolute_978_gene_delta": rho,
            "undefined_reason": reason,
            "median_prism_minus_lfc_sensitivity": float(group["prism_minus_lfc_sensitivity"].median()),
            "median_lincs_response_magnitude": float(group["response_magnitude"].median()),
            "exploratory_only": True,
        })
    columns = [
        "inchikey", "lincs_dose_uM", "lincs_time_h", "n_unique_depmap_patient_ids",
        "n_train_source_origins", "n_val_source_origins", "n_prism_cell_lines",
        "spearman_rho_prism_minus_lfc_vs_lincs_mean_absolute_978_gene_delta",
        "undefined_reason", "median_prism_minus_lfc_sensitivity", "median_lincs_response_magnitude",
        "exploratory_only",
    ]
    per_drug = pd.DataFrame(summaries, columns=columns).sort_values("inchikey").reset_index(drop=True)
    pair_columns = [
        "source_origin_id", "split_role", "inchikey", "dose_uM", "time_h", "response_magnitude",
        "prism_minus_lfc_sensitivity", "n_prism_cell_lines", "n_prism_sample_drug_pairs",
        "n_prism_response_rows",
    ]
    patient_pairs = joined.rename(columns={"dose_uM": "lincs_dose_uM", "time_h": "lincs_time_h"})
    pair_columns = [c if c not in {"dose_uM", "time_h"} else {"dose_uM": "lincs_dose_uM", "time_h": "lincs_time_h"}[c] for c in pair_columns]
    patient_pairs = patient_pairs[pair_columns].sort_values(["inchikey", "source_origin_id"]).reset_index(drop=True)
    join_summary = {
        "allowed_prism_cell_drug_rows_after_identity_and_drug_join": int(len(prism_cell)),
        "allowed_prism_source_origin_drug_rows": int(len(prism_origin)),
        "frozen_lincs_source_feature_rows_at_selected_exposure": int(len(lincs)),
        "matched_source_origin_drug_rows": int(len(patient_pairs)),
        "matched_exact_full_inchikeys": int(patient_pairs["inchikey"].nunique()),
        "drugs_meeting_minimum_n_with_defined_rho": int(per_drug["spearman_rho_prism_minus_lfc_vs_lincs_mean_absolute_978_gene_delta"].notna().sum()),
        "minimum_n_unique_source_origins": MIN_ORIGINS,
        "excluded_ctrp_aac_values": True,
        "association_method": "Per-exact-InChIKey Spearman correlation across unique DepMap PatientID source origins; no pooled drugs or p-values.",
    }
    return patient_pairs, per_drug, {**scan_summary, **join_summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/EXP-050.yaml"))
    parser.add_argument("--run-id", default=EXPECTED_RUN_ID)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    contract = read_contract(root, config_path, args.run_id)
    output_dir = root / contract["outputs"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest = output_dir / "run_manifest.json"
    if existing_manifest.exists():
        raise FileExistsError(f"Completed run already exists: {existing_manifest}")
    started = datetime.now(timezone.utc).isoformat()

    # Confirm the public EXP-012 semantics before reading any response values.
    exp012 = json.loads((root / "data/processed/exp012/cohort_index_manifest.json").read_text(encoding="utf-8"))
    cohorts = [c for c in exp012.get("cohorts", []) if c.get("cohort_id") == "prism_extended_primary"]
    if len(cohorts) != 1:
        raise ValueError("EXP-012 PRISM cohort manifest is missing or ambiguous")
    cohort = cohorts[0]
    if cohort.get("response_name") != "neg_extended_primary_lfc" or cohort.get("higher_is_sensitive") is not True:
        raise ValueError("PRISM response scale differs from the frozen -LFC, higher-sensitive contract")
    if cohort.get("role") != "FROZEN_EXTERNAL_CELL_LINE":
        raise ValueError("PRISM cohort role is not frozen external cell-line reference")

    frozen, n_locked_identity = read_frozen_source_map(root)
    allowed_samples, identity_summary = read_prism_samples(root, frozen)
    drugs = read_prism_drugs(root)
    selected, lincs = read_frozen_lincs_features(root)
    patient_pairs, per_drug, analysis_summary = calculate_associations(
        root, allowed_samples, drugs, selected, lincs
    )

    if patient_pairs["split_role"].eq("locked_eval").any():
        raise ValueError("Locked-eval rows are present in the association output")
    if patient_pairs["source_origin_id"].isin(
        frozen.loc[frozen["split_role"].eq("locked_eval"), "source_origin_id"]
    ).any():
        raise ValueError("Locked-eval source-origin identity is present in the association output")

    pair_path = output_dir / "patient_level_pairs.parquet"
    drug_path = output_dir / "per_drug_association.tsv"
    coverage_path = output_dir / "coverage_summary.json"
    manifest_path = output_dir / "run_manifest.json"
    atomic_parquet(pair_path, patient_pairs)
    per_drug.to_csv(drug_path, sep="\t", index=False, na_rep="")
    coverage = {
        "run_id": args.run_id,
        "status": "COMPLETED_EXPLORATORY",
        "source_reference_role": "PRISM 24Q2 Extended Primary -LFC sensitivity reference; not training/selection and not PDO efficacy",
        "identity_coverage": identity_summary,
        "exact_compound_coverage": {
            "prism_unique_full_inchikeys": int(drugs["inchikey"].nunique()),
            "frozen_lincs_selected_compounds": int(selected["inchikey"].nunique()),
            "exact_full_inchikey_intersection": int(len(set(drugs["inchikey"]) & set(selected["inchikey"]))),
        },
        "analysis": analysis_summary,
        "interpretation": "Descriptive external source-cell-line association only. Per-drug signs may vary; no pooled estimate, p-value, confidence interval, model/drug selection, ranking, PDO efficacy, or independent generalization claim.",
        "locked_eval_exclusion": {
            "frozen_locked_eval_identity_rows": n_locked_identity,
            "arrow_response_scan_sample_filter": "train and val sample_id allowlist only",
            "locked_eval_rows_returned": 0,
            "locked_eval_rows_written": 0,
        },
    }
    atomic_json(coverage_path, coverage)

    input_paths = [
        root / "configs/experiments/EXP-050.yaml",
        root / "data/processed/exp012/cohort_index_manifest.json",
        root / "data/processed/exp012/prism_extended_primary_raw_manifest.json",
        root / "data/processed/exp012/prism_extended_primary_pairs.parquet",
        root / "data/processed/exp012/prism_extended_primary_samples.parquet",
        root / "data/processed/exp012/prism_extended_primary_drugs.parquet",
        root / "artifacts/experiments/EXP-007/CCLE_CONTEXT_IDENTITY_SPLIT.json",
        root / "data/raw/depmap/24q2/Model.csv",
        root / "artifacts/experiments/EXP-050/20261001_lincs_ctrp_transfer_association_v3/patient_level_pairs.parquet",
        root / "artifacts/experiments/EXP-050/20261001_lincs_ctrp_transfer_association_v3/per_drug_association.tsv",
        root / "scripts/evaluation/analyze_exp050_source_lincs_prism_association.py",
    ]
    finished = datetime.now(timezone.utc).isoformat()
    manifest = {
        "run_id": args.run_id,
        "status": "COMPLETED_EXPLORATORY",
        "started_at_utc": started,
        "finished_at_utc": finished,
        "exit_code": 0,
        "environment": "WSL2 Conda drugscreening-gpu",
        "compute": {"device": "CPU-only", "gpu_hours": 0.0},
        "protocol": "configs/experiments/EXP-050.yaml:source_lincs_prism_association_exploratory",
        "response_values_read": True,
        "response_read_contract": "Only Arrow-filtered rows for exact frozen train/val PRISM sample IDs were returned to numeric analysis; CTRPv2 AAC was excluded by column projection.",
        "pre_execution_failures": [
            {
                "attempted_on": "2026-10-01",
                "attempt_time_precision": "not captured by the failed shell launch",
                "stage": "WSL Conda launcher",
                "failure": "The first shell invocation could not resolve `conda` because this non-interactive WSL shell did not initialize it.",
                "response_values_read": False,
                "gpu_hours": 0.0,
                "recovery": "Use the documented explicit Linux Conda path /home/dell/miniconda3/bin/conda and run scripts/check_environment.py before analysis.",
            },
            {
                "attempted_on": "2026-10-01",
                "attempt_time_precision": "not captured by command output",
                "stage": "dataset registry validation before analysis",
                "failure": "The documented registry module invocation was missing PYTHONPATH=src, so Python could not import drug_screen; the chained PRISM analyzer did not start.",
                "response_values_read": False,
                "gpu_hours": 0.0,
                "recovery": "Rerun the registry validator with PYTHONPATH=src as documented in docs/ENVIRONMENT.md, then run the frozen analysis.",
            }
        ],
        "target_expression_or_labels_read": False,
        "locked_eval_response_rows_returned_or_written": 0,
        "identity_summary": identity_summary,
        "analysis_summary": analysis_summary,
        "inputs": {
            str(p.relative_to(root)).replace("\\", "/"): sha256_file(p)
            for p in input_paths
        },
        "outputs": {
            "coverage_summary.json": sha256_file(coverage_path),
            "patient_level_pairs.parquet": sha256_file(pair_path),
            "per_drug_association.tsv": sha256_file(drug_path),
        },
        "output_rows": {
            "patient_level_pairs": int(len(patient_pairs)),
            "per_drug_association": int(len(per_drug)),
        },
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps({
        "run_id": args.run_id,
        "status": "COMPLETED_EXPLORATORY",
        "patient_level_pairs": int(len(patient_pairs)),
        "drugs_with_rho": int(per_drug["spearman_rho_prism_minus_lfc_vs_lincs_mean_absolute_978_gene_delta"].notna().sum()),
        "gpu_hours": 0.0,
        "output_dir": str(output_dir),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
