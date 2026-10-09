#!/usr/bin/env python3
"""Exploratory, source-domain LINCS response magnitude vs CTRPv2 AAC association.

This is an EXP-050 section 3.4 descriptive analysis, not a drug-response model
evaluation. Cell and compound identities use frozen exact-key contracts. It
never reads organoid treated expression or outer-test labels.
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


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_RUN_ID = "20261001_lincs_ctrp_transfer_association_v3"
MIN_ORIGINS = 10


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


def required_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def normalize_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper().replace("", pd.NA)


def read_frozen_cells(root: Path) -> pd.DataFrame:
    mapping_path = root / "data/artifacts/resource_acquisition_20260916/ctrp_parse/ctrp_depmap_identity_map.csv"
    model_path = root / "data/raw/depmap/24q2/Model.csv"
    identity_path = root / "artifacts/experiments/EXP-007/CCLE_CONTEXT_IDENTITY_SPLIT.json"

    ctrp_map = pd.read_csv(mapping_path, dtype="string")
    required_columns(
        ctrp_map,
        {"cellid", "ModelID", "match_method", "has_depmap_expression"},
        "frozen CTRPv2-DepMap identity map",
    )
    ctrp_map = ctrp_map.loc[
        ctrp_map["match_method"].eq("exact_unique_cellosaurus_rrid")
        & ctrp_map["has_depmap_expression"].astype("string").str.lower().eq("true"),
        ["cellid", "ModelID"],
    ].dropna().drop_duplicates()
    if ctrp_map["cellid"].duplicated().any() or ctrp_map["ModelID"].duplicated().any():
        raise ValueError("CTRPv2 exact RRID mapping is not one-to-one")

    models = pd.read_csv(model_path, usecols=["ModelID", "PatientID"], dtype="string")
    models = models.dropna(subset=["ModelID", "PatientID"]).drop_duplicates()
    if models["ModelID"].duplicated().any():
        raise ValueError("DepMap ModelID does not identify a unique PatientID")

    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("status") != "IDENTITY_SPLIT_FROZEN":
        raise ValueError("LINCS/DepMap identity split is not frozen")
    rows = pd.DataFrame(identity.get("identity_rows", []))
    required_columns(
        rows,
        {"lincs_cell_id", "depmap_id", "patient_id", "expression_present", "join_status"},
        "frozen LINCS/DepMap identity rows",
    )
    rows = rows.loc[
        rows["expression_present"].astype(bool)
        & rows["join_status"].eq("EXACT_STRIPPED_NAME_WITH_RNASEQ"),
        ["lincs_cell_id", "depmap_id", "patient_id"],
    ].dropna().drop_duplicates()
    if rows["lincs_cell_id"].duplicated().any() or rows["depmap_id"].duplicated().any():
        raise ValueError("Frozen LINCS/DepMap identity crosswalk is not one-to-one")
    split = pd.DataFrame(identity.get("split_assignments", []))
    required_columns(split, {"depmap_id", "split_role"}, "frozen response-blind split assignments")
    split = split[["depmap_id", "split_role"]].dropna().drop_duplicates()
    if split["depmap_id"].duplicated().any():
        raise ValueError("Frozen response-blind split has duplicate DepMap IDs")
    rows = rows.merge(split, on="depmap_id", how="inner", validate="one_to_one")
    if not rows["split_role"].isin(["train", "val", "locked_eval"]).all():
        raise ValueError("Frozen split contains an unrecognized role")
    n_locked_eval_excluded = int(rows["split_role"].eq("locked_eval").sum())
    rows = rows.loc[rows["split_role"].isin(["train", "val"])].copy()

    # Verify hashed donor-origin keys agree across the two frozen DepMap sources.
    check = rows.merge(
        models, left_on="depmap_id", right_on="ModelID", how="inner", validate="one_to_one"
    )
    if not check["patient_id"].astype(str).eq(check["PatientID"].astype(str)).all():
        raise ValueError("Frozen identity PatientID disagrees with DepMap Model.csv")

    cells = ctrp_map.merge(models, left_on="ModelID", right_on="ModelID", how="inner", validate="many_to_one")
    cells = cells.merge(
        rows[["depmap_id", "lincs_cell_id", "patient_id", "split_role"]],
        left_on="ModelID",
        right_on="depmap_id",
        how="inner",
        validate="many_to_one",
    )
    if not cells["PatientID"].astype(str).eq(cells["patient_id"].astype(str)).all():
        raise ValueError("CTRPv2-to-LINCS donor-origin keys disagree")
    cells = cells[["cellid", "ModelID", "PatientID", "lincs_cell_id", "split_role"]].drop_duplicates()
    cells.attrs["locked_eval_cells_excluded"] = n_locked_eval_excluded
    return cells


def read_exact_drugs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ctrp_path = root / "data/processed/exp013/ctrp_v2/drugs.csv"
    ctrp = pd.read_csv(ctrp_path, usecols=["drugid", "inchikey"], dtype="string")
    ctrp["inchikey"] = normalize_key(ctrp["inchikey"])
    ctrp = ctrp.dropna(subset=["drugid", "inchikey"])
    ctrp_key_counts = ctrp.groupby("inchikey")["drugid"].nunique()
    ctrp_unique = ctrp.loc[ctrp["inchikey"].map(ctrp_key_counts).eq(1)].drop_duplicates()
    ctrp_unique = ctrp_unique[["drugid", "inchikey"]].drop_duplicates()
    if ctrp_unique["drugid"].duplicated().any():
        raise ValueError("CTRPv2 drugid maps to multiple exact InChIKeys")

    lincs_path = root / "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_pert_info.txt.gz"
    lincs = pd.read_csv(
        lincs_path,
        sep="\t",
        compression="gzip",
        usecols=["pert_id", "pert_type", "inchi_key"],
        dtype="string",
    )
    lincs = lincs.loc[lincs["pert_type"].eq("trt_cp")].copy()
    lincs["inchikey"] = normalize_key(lincs["inchi_key"])
    lincs = lincs.dropna(subset=["pert_id", "inchikey"])
    return ctrp_unique, lincs[["pert_id", "inchikey"]].drop_duplicates()


def summarize_ctrp(root: Path, cells: pd.DataFrame, drugs: pd.DataFrame) -> pd.DataFrame:
    response_path = root / "data/processed/exp013/ctrp_v2/responses.csv"
    allowed_cells = set(cells["cellid"].astype(str))
    eligible_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        response_path,
        usecols=["cellid", "drugid", "aac_published"],
        dtype={"cellid": "string", "drugid": "string"},
        chunksize=50_000,
    ):
        chunk = chunk.loc[chunk["cellid"].astype(str).isin(allowed_cells)].copy()
        if chunk.empty:
            continue
        chunk["aac_published"] = pd.to_numeric(chunk["aac_published"], errors="coerce")
        eligible_chunks.append(chunk.loc[np.isfinite(chunk["aac_published"])])
    response = (
        pd.concat(eligible_chunks, ignore_index=True)
        if eligible_chunks
        else pd.DataFrame(columns=["cellid", "drugid", "aac_published"])
    )
    response_cell = (
        response.groupby(["cellid", "drugid"], as_index=False)
        .agg(aac_median=("aac_published", "median"), n_aac_rows=("aac_published", "size"))
    )
    response_cell = response_cell.merge(cells, on="cellid", how="inner", validate="many_to_one")
    response_cell = response_cell.merge(drugs, on="drugid", how="inner", validate="many_to_one")
    # Cell lines derived from the same DepMap PatientID form one source-origin unit.
    return (
        response_cell.groupby(["PatientID", "inchikey"], as_index=False)
        .agg(
            aac=("aac_median", "median"),
            n_cell_lines=("ModelID", "nunique"),
            n_aac_rows=("n_aac_rows", "sum"),
        )
    )


def summarize_lincs(root: Path, cells: pd.DataFrame, lincs_drugs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    base = root / "data/processed/lincs/GSE92742/continuous_conditions_v2"
    conditions = pd.read_parquet(base / "conditions.parquet")
    required_columns(
        conditions,
        {
            "condition_id", "cell_id", "pert_id", "pert_dose", "pert_dose_unit",
            "pert_time", "pert_time_unit",
        },
        "LINCS conditions",
    )
    condition_ids = pd.to_numeric(conditions["condition_id"], errors="coerce")
    if condition_ids.isna().any() or condition_ids.duplicated().any():
        raise ValueError("LINCS condition_id must be finite and unique")
    conditions["condition_id"] = condition_ids.astype(np.int64)

    selected = conditions.merge(
        cells[["lincs_cell_id", "PatientID"]].drop_duplicates(),
        left_on="cell_id",
        right_on="lincs_cell_id",
        how="inner",
        validate="many_to_one",
    ).merge(lincs_drugs, on="pert_id", how="inner", validate="many_to_many")
    selected["dose"] = pd.to_numeric(selected["pert_dose"], errors="coerce")
    selected["time"] = pd.to_numeric(selected["pert_time"], errors="coerce")
    dose_unit = selected["pert_dose_unit"].astype("string").str.strip().str.lower().str.replace("μ", "u", regex=False)
    time_unit = selected["pert_time_unit"].astype("string").str.strip().str.lower()
    eligible = (
        dose_unit.eq("um")
        & time_unit.eq("h")
        & np.isfinite(selected["dose"])
        & np.isfinite(selected["time"])
    )
    selected = selected.loc[eligible].copy()

    delta_path = base / "condition_delta.npy"
    delta = np.load(delta_path, mmap_mode="r", allow_pickle=False)
    if delta.ndim != 2 or delta.shape[1] != 978 or int(conditions["condition_id"].max()) >= delta.shape[0]:
        raise ValueError(f"LINCS delta array does not match the frozen 978-gene contract: {delta.shape}")
    selected = selected.sort_values("condition_id").reset_index(drop=True)
    magnitude = np.empty(len(selected), dtype=np.float64)
    ids = selected["condition_id"].to_numpy(dtype=np.int64)
    for start in range(0, len(ids), 8192):
        end = min(start + 8192, len(ids))
        block = np.asarray(delta[ids[start:end], :], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("Selected LINCS response contains non-finite values")
        magnitude[start:end] = np.abs(block).mean(axis=1)
    selected["response_magnitude"] = magnitude

    # Give each LINCS cell line equal weight before collapsing shared origins.
    cell_level = (
        selected.groupby(["PatientID", "cell_id", "inchikey", "dose", "time"], as_index=False)
        .agg(
            response_magnitude=("response_magnitude", "median"),
            n_condition_rows=("condition_id", "nunique"),
        )
    )
    patient_level = (
        cell_level.groupby(["PatientID", "inchikey", "dose", "time"], as_index=False)
        .agg(
            response_magnitude=("response_magnitude", "median"),
            n_cell_lines=("cell_id", "nunique"),
            n_condition_rows=("n_condition_rows", "sum"),
        )
    )
    coverage = {
        "conditions_total": int(len(conditions)),
        "cells_after_exact_identity_join": int(conditions["cell_id"].isin(set(cells["lincs_cell_id"])).sum()),
        "pert_id_rows_after_exact_inchikey_join": int(conditions["pert_id"].isin(set(lincs_drugs["pert_id"])).sum()),
        "conditions_after_cell_drug_dose_time_filter": int(len(selected)),
        "patient_drug_exposure_rows": int(len(patient_level)),
        "unique_lincs_cells": int(selected["cell_id"].nunique()),
        "unique_compound_keys": int(selected["inchikey"].nunique()),
        "unit_policy": "dose unit exactly micromolar; time unit exactly hours",
    }
    return patient_level, coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = (root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))
    contract = config.get("source_lincs_ctrp_association_exploratory", {})
    if args.run_id != EXPECTED_RUN_ID or contract.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("Run ID does not match the frozen EXP-050 association contract")
    if contract.get("status") != "preregistered_exploratory_pending_execution":
        raise ValueError("Exploratory protocol is not in the expected frozen state")

    cells = read_frozen_cells(root)
    ctrp_drugs, lincs_drugs = read_exact_drugs(root)
    ctrp_key_set = set(ctrp_drugs["inchikey"])
    lincs_key_set = set(lincs_drugs["inchikey"])
    common_drugs = ctrp_key_set.intersection(lincs_key_set)
    n_ctrp_unique_keys = len(ctrp_key_set)
    n_lincs_unique_keys = len(lincs_key_set)
    ctrp_drugs = ctrp_drugs.loc[ctrp_drugs["inchikey"].isin(common_drugs)].copy()
    lincs_drugs = lincs_drugs.loc[lincs_drugs["inchikey"].isin(common_drugs)].copy()
    ctrp_patient = summarize_ctrp(root, cells, ctrp_drugs)
    lincs_patient, lincs_coverage = summarize_lincs(root, cells, lincs_drugs)

    joined = lincs_patient.merge(
        ctrp_patient,
        on=["PatientID", "inchikey"],
        how="inner",
        suffixes=("_lincs", "_ctrp"),
        validate="many_to_one",
    )
    summaries: list[dict[str, Any]] = []
    pair_columns = [
        "PatientID", "inchikey", "dose", "time", "response_magnitude", "aac",
        "n_cell_lines_lincs", "n_condition_rows", "n_cell_lines_ctrp", "n_aac_rows", "n_origins",
    ]
    if joined.empty:
        paired = pd.DataFrame(columns=pair_columns)
    else:
        exposure_coverage = (
            joined.groupby(["inchikey", "dose", "time"], as_index=False)
            .agg(n_origins=("PatientID", "nunique"))
            .sort_values(["inchikey", "n_origins", "dose", "time"], ascending=[True, False, True, True])
        )
        chosen = exposure_coverage.drop_duplicates("inchikey", keep="first")
        paired = joined.merge(chosen, on=["inchikey", "dose", "time"], how="inner", validate="many_to_one")

        for key, group in paired.groupby("inchikey", sort=True):
            group = group.sort_values("PatientID").reset_index(drop=True)
            n = int(group["PatientID"].nunique())
            rho: float | None = None
            undefined_reason: str | None = None
            if n < MIN_ORIGINS:
                undefined_reason = f"fewer_than_{MIN_ORIGINS}_unique_source_origins"
            elif group["aac"].nunique(dropna=True) < 2:
                undefined_reason = "constant_aac"
            elif group["response_magnitude"].nunique(dropna=True) < 2:
                undefined_reason = "constant_response_magnitude"
            else:
                value = group["aac"].corr(group["response_magnitude"], method="spearman")
                rho = float(value) if pd.notna(value) and math.isfinite(float(value)) else None
                if rho is None:
                    undefined_reason = "spearman_undefined"
            summaries.append(
                {
                    "inchikey": key,
                    "dose_uM": float(group["dose"].iloc[0]),
                    "time_h": float(group["time"].iloc[0]),
                    "n_unique_depmap_patient_ids": n,
                    "n_cell_lines_ctrp": int(group["n_cell_lines_ctrp"].sum()),
                    "n_cell_lines_lincs": int(group["n_cell_lines_lincs"].sum()),
                    "spearman_rho_aac_vs_mean_absolute_978_gene_delta": rho,
                    "undefined_reason": undefined_reason,
                    "median_aac": float(group["aac"].median()),
                    "median_response_magnitude": float(group["response_magnitude"].median()),
                    "exploratory_only": True,
                }
            )
    summary_columns = [
        "inchikey", "dose_uM", "time_h", "n_unique_depmap_patient_ids", "n_cell_lines_ctrp",
        "n_cell_lines_lincs", "spearman_rho_aac_vs_mean_absolute_978_gene_delta", "undefined_reason",
        "median_aac", "median_response_magnitude", "exploratory_only",
    ]
    summary_frame = pd.DataFrame(summaries, columns=summary_columns).sort_values("inchikey").reset_index(drop=True)
    output_dir = root / contract["outputs"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=False)
    pair_output = paired[pair_columns].rename(
        columns={"PatientID": "source_origin_id", "dose": "dose_uM", "time": "time_h"}
    )
    atomic_parquet(output_dir / "patient_level_pairs.parquet", pair_output)
    summary_frame.to_csv(output_dir / "per_drug_association.tsv", sep="\t", index=False, na_rep="")

    input_files = [
        root / "data/processed/exp013/ctrp_v2/responses.csv",
        root / "data/processed/exp013/ctrp_v2/drugs.csv",
        root / "data/artifacts/resource_acquisition_20260916/ctrp_parse/ctrp_depmap_identity_map.csv",
        root / "data/raw/depmap/24q2/Model.csv",
        root / "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_pert_info.txt.gz",
        root / "data/processed/lincs/GSE92742/continuous_conditions_v2/conditions.parquet",
        root / "data/processed/lincs/GSE92742/continuous_conditions_v2/condition_delta.npy",
        root / "artifacts/experiments/EXP-007/CCLE_CONTEXT_IDENTITY_SPLIT.json",
        config_path,
        Path(__file__).resolve(),
    ]
    manifest = {
        "run_id": args.run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETED_EXPLORATORY" if not joined.empty else "BLOCKED_NO_MATCHED_SOURCE_PAIRS",
        "exploratory_only_reason": "Part of CTRPv2 responses.csv was inadvertently emitted in prior tool output before protocol freeze; no independent or confirmatory claim is permitted.",
        "protocol": "EXP-050 section 3.4 source-domain association; per-drug Spearman between LINCS 978-gene mean absolute matched-control delta and CTRPv2 published AAC.",
        "labels": {"endpoint": "aac_published", "higher_is_more_sensitive": True, "outer_test_or_organoid_treated_labels_read": False},
        "gpu_hours": 0,
        "n_cell_identity_rows": int(len(cells)),
        "excluded_locked_eval_cells_before_endpoint_read": int(cells.attrs.get("locked_eval_cells_excluded", 0)),
        "included_split_roles": {str(k): int(v) for k, v in cells["split_role"].value_counts().items()},
        "n_ctrp_unique_full_inchikeys_before_intersection": int(n_ctrp_unique_keys),
        "n_lincs_unique_full_inchikeys_before_intersection": int(n_lincs_unique_keys),
        "n_common_exact_full_inchikeys": int(len(common_drugs)),
        "n_patient_level_ctrp_pairs": int(len(ctrp_patient)),
        "n_patient_level_lincs_conditions": int(len(lincs_patient)),
        "n_joined_patient_drug_exposures": int(len(joined)),
        "n_selected_drugs": int(len(summary_frame)),
        "n_drugs_meeting_minimum_n": int(summary_frame["spearman_rho_aac_vs_mean_absolute_978_gene_delta"].notna().sum()),
        "minimum_n_unique_source_origins": MIN_ORIGINS,
        "identity_policy": "Exact unique Cellosaurus RRID for CTRPv2-to-DepMap; frozen exact LINCS/DepMap identity rows; exact full InChIKey; cell lines sharing DepMap PatientID count once.",
        "exposure_policy": "Per compound select the dose/time with greatest matched source-origin coverage; ties lower dose then shorter time. All selected units are micromolar and hours.",
        "lincs_coverage": lincs_coverage,
        "unmatched_exact_compound_keys": int(len(ctrp_key_set.symmetric_difference(lincs_key_set))),
        "inputs_sha256": {str(path.relative_to(root)): sha256_file(path) for path in input_files},
        "outputs": {
            "run_manifest": "run_manifest.json",
            "coverage_summary": "coverage_summary.json",
            "patient_level_pairs": "patient_level_pairs.parquet",
            "per_drug_association": "per_drug_association.tsv",
        },
    }
    coverage_summary = {
        "run_id": args.run_id,
        "status": manifest["status"],
        "n_cell_identity_rows": manifest["n_cell_identity_rows"],
        "n_common_exact_full_inchikeys": manifest["n_common_exact_full_inchikeys"],
        "n_patient_level_ctrp_pairs": manifest["n_patient_level_ctrp_pairs"],
        "n_patient_level_lincs_conditions": manifest["n_patient_level_lincs_conditions"],
        "n_joined_patient_drug_exposures": manifest["n_joined_patient_drug_exposures"],
        "n_selected_drugs": manifest["n_selected_drugs"],
        "n_drugs_with_defined_rho": manifest["n_drugs_meeting_minimum_n"],
        "minimum_n_unique_source_origins": MIN_ORIGINS,
        "lincs_coverage": lincs_coverage,
        "interpretation": (
            "Per-drug descriptive source-domain association only; no pooled estimate, p-value, interval, model selection, or independent generalization claim."
            if not joined.empty
            else "No patient-level pairs remain under exact frozen identities; no association estimate is available."
        ),
    }
    atomic_json(output_dir / "run_manifest.json", manifest)
    atomic_json(output_dir / "coverage_summary.json", coverage_summary)
    print(json.dumps({k: manifest[k] for k in ["run_id", "status", "n_cell_identity_rows", "n_common_exact_full_inchikeys", "n_joined_patient_drug_exposures", "n_selected_drugs", "n_drugs_meeting_minimum_n", "gpu_hours"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
