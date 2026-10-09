#!/usr/bin/env python3
"""Response-blind identity and condition inventory for EXP-068 breast lines.

The script reads only identity/condition metadata. It projects exactly two
columns from CTRPv2 responses.csv and the first (ModelID) column from the
DepMap expression CSV. It never opens response values, expression values, or
LINCS perturbation expression/delta arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


FULL_INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
APPROVED_LINCS_DATASET = "GSE92742"
APPROVED_LINCS_STATUS = "base_cell_id mapped to unique DepMap ModelID"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def projection_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(
        sep="\t", index=False, lineterminator="\n", na_rep=""
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def exact_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def exact_full_inchikey(value: object) -> str:
    candidate = exact_text(value)
    return candidate if FULL_INCHIKEY_RE.fullmatch(candidate) else ""


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required metadata columns: {sorted(missing)}")


def require_unique_columns(frame: pd.DataFrame, label: str) -> None:
    duplicated = frame.columns[frame.columns.duplicated()].astype(str).tolist()
    if duplicated:
        raise ValueError(f"{label} contains duplicate column labels: {duplicated}")


def write_tsv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n", na_rep="")


def validate_tsv_readback(frame: pd.DataFrame, path: Path, label: str) -> dict[str, Any]:
    require_unique_columns(frame, label)
    if frame.shape[1] == 0:
        if path.read_text(encoding="utf-8").strip():
            raise ValueError(f"{label} has no columns but serialized content is not empty")
        return {"rows": 0, "columns": []}
    loaded = pd.read_csv(path, sep="\t", dtype="string", keep_default_na=False)
    require_unique_columns(loaded, f"{label} TSV readback")
    if list(loaded.columns) != list(frame.columns):
        raise ValueError(f"{label} TSV readback changed column order or names")
    if len(loaded) != len(frame):
        raise ValueError(f"{label} TSV readback changed row count")
    for key_column in (
        "identity_key",
        "CTRP_cellid",
        "DepMap_ModelID",
        "CTRP_full_InChIKey_exact",
        "LINCS_full_InChIKey_exact",
        "LINCS_condition_id",
    ):
        if key_column in frame.columns:
            expected = frame[key_column].fillna("").astype(str).tolist()
            observed = loaded[key_column].tolist()
            if observed != expected:
                raise ValueError(f"{label} TSV readback changed identity key {key_column}")
    return {"rows": int(len(loaded)), "columns": list(loaded.columns)}


def unique_key_map(
    frame: pd.DataFrame, key_column: str, value_column: str
) -> tuple[dict[str, str], set[str]]:
    valid = frame.loc[
        frame[key_column].fillna("").astype(str).ne("")
        & frame[value_column].fillna("").astype(str).ne(""),
        [key_column, value_column],
    ].drop_duplicates()
    n_values = valid.groupby(key_column, dropna=False)[value_column].nunique()
    ambiguous = set(n_values[n_values.gt(1)].index.astype(str))
    unique = valid.loc[~valid[key_column].astype(str).isin(ambiguous)].drop_duplicates()
    return (
        dict(zip(unique[key_column].astype(str), unique[value_column].astype(str))),
        ambiguous,
    )


def load_metadata(root: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    ctrp_dir = root / "data/processed/exp013/ctrp_v2"
    depmap_dir = root / "data/raw/depmap/24q2"
    lincs_raw = root / "data/raw/lincs/GSE92742"
    lincs_processed = root / "data/processed/lincs/GSE92742"
    crosswalk_path = (
        root
        / "artifacts/experiments/EXP-050/20261006_breast_subtype_measured_transfer_v1/"
        / "lincs_breast_cell_model_identity.tsv"
    )

    paths = {
        "ctrp_cells": ctrp_dir / "cells.csv",
        "ctrp_drugs": ctrp_dir / "drugs.csv",
        "ctrp_responses": ctrp_dir / "responses.csv",
        "depmap_model": depmap_dir / "Model.csv",
        "depmap_expression": depmap_dir / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        "lincs_cell_info": lincs_raw / "GSE92742_Broad_LINCS_cell_info.txt.gz",
        "lincs_pert_info": lincs_raw / "GSE92742_Broad_LINCS_pert_info.txt.gz",
        "lincs_conditions": lincs_processed / "continuous_conditions_v2/conditions.parquet",
        "approved_breast_crosswalk": crosswalk_path,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required identity/condition metadata missing: " + "; ".join(missing))

    ctrp_cells = pd.read_csv(
        paths["ctrp_cells"],
        usecols=["cellid", "Cellosaurus.Accession.id", "ccle_primary_site"],
        dtype="string",
    )
    require_columns(
        ctrp_cells,
        {"cellid", "Cellosaurus.Accession.id", "ccle_primary_site"},
        "CTRPv2 cell identity metadata",
    )
    ctrp_cells["cellid"] = ctrp_cells["cellid"].astype("string").str.strip()
    ctrp_cells["rrid_exact"] = ctrp_cells["Cellosaurus.Accession.id"].map(exact_text)

    # Strict response projection: no other responses.csv column is selected.
    ctrp_response_projection = pd.read_csv(
        paths["ctrp_responses"],
        usecols=["cellid", "drugid"],
        dtype={"cellid": "string", "drugid": "string"},
    )
    ctrp_response_projection["cellid"] = ctrp_response_projection["cellid"].str.strip()
    ctrp_response_projection["drugid"] = ctrp_response_projection["drugid"].str.strip()
    response_valid_mask = (
        ctrp_response_projection["cellid"].notna()
        & ctrp_response_projection["cellid"].ne("")
        & ctrp_response_projection["drugid"].notna()
        & ctrp_response_projection["drugid"].ne("")
    )
    response_projection_rows = ctrp_response_projection.loc[response_valid_mask].copy()
    response_projection_invalid_rows = int((~response_valid_mask).sum())
    response_pair_counts = (
        response_projection_rows.groupby(["cellid", "drugid"], as_index=False, dropna=False)
        .size()
        .rename(columns={"size": "CTRP_response_metadata_record_count"})
    )

    ctrp_drugs = pd.read_csv(
        paths["ctrp_drugs"],
        usecols=["drugid", "inchikey"],
        dtype="string",
    )
    ctrp_drugs["drugid"] = ctrp_drugs["drugid"].astype("string").str.strip()
    ctrp_drugs["full_inchikey"] = ctrp_drugs["inchikey"].map(exact_full_inchikey)

    depmap_model = pd.read_csv(
        paths["depmap_model"],
        usecols=["ModelID", "PatientID", "RRID", "OncotreeLineage", "OncotreePrimaryDisease"],
        dtype="string",
    )
    depmap_model["ModelID"] = depmap_model["ModelID"].astype("string").str.strip()
    depmap_model["RRID_exact"] = depmap_model["RRID"].map(exact_text)
    depmap_model["PatientID"] = depmap_model["PatientID"].astype("string").str.strip()
    depmap_model["is_breast_oncotree"] = (
        depmap_model["OncotreeLineage"].fillna("").str.strip().str.casefold().eq("breast")
    )

    # Only the first CSV column (the DepMap ModelID row axis) is read.
    depmap_expression_ids_frame = pd.read_csv(
        paths["depmap_expression"],
        header=None,
        skiprows=1,
        usecols=[0],
        names=["ModelID"],
        dtype="string",
    )
    depmap_expression_ids_frame["ModelID"] = depmap_expression_ids_frame["ModelID"].str.strip()
    depmap_expression_ids = set(
        depmap_expression_ids_frame["ModelID"].dropna().astype(str).loc[lambda values: values.ne("")]
    )
    depmap_expression_duplicate_ids = int(
        depmap_expression_ids_frame["ModelID"].dropna().duplicated().sum()
    )

    # Read only the identity fields of the already approved breast crosswalk.
    approved_crosswalk = pd.read_csv(
        paths["approved_breast_crosswalk"],
        sep="\t",
        usecols=["dataset", "LINCS_cell_id", "LINCS_base_cell_id", "DepMap_ModelID", "identity_status"],
        dtype="string",
    )
    approved_crosswalk = approved_crosswalk.loc[
        approved_crosswalk["dataset"].eq(APPROVED_LINCS_DATASET)
    ].copy()
    approved_crosswalk["LINCS_cell_id"] = approved_crosswalk["LINCS_cell_id"].astype("string").str.strip()
    approved_crosswalk["LINCS_base_cell_id"] = approved_crosswalk["LINCS_base_cell_id"].astype("string").str.strip()
    approved_crosswalk["DepMap_ModelID"] = approved_crosswalk["DepMap_ModelID"].astype("string").str.strip()
    approved_crosswalk["crosswalk_approved"] = approved_crosswalk["identity_status"].eq(APPROVED_LINCS_STATUS)

    lincs_cell_info = pd.read_csv(
        paths["lincs_cell_info"],
        sep="\t",
        compression="gzip",
        usecols=["cell_id", "base_cell_id", "primary_site"],
        dtype="string",
    )
    lincs_cell_info["cell_id"] = lincs_cell_info["cell_id"].astype("string").str.strip()
    lincs_cell_info["base_cell_id"] = lincs_cell_info["base_cell_id"].astype("string").str.strip()
    lincs_cell_info["primary_site"] = lincs_cell_info["primary_site"].astype("string").str.strip()

    lincs_pert_info = pd.read_csv(
        paths["lincs_pert_info"],
        sep="\t",
        compression="gzip",
        usecols=["pert_id", "pert_type", "inchi_key"],
        dtype="string",
    )
    lincs_pert_info["pert_id"] = lincs_pert_info["pert_id"].astype("string").str.strip()
    lincs_pert_info["full_inchikey"] = lincs_pert_info["inchi_key"].map(exact_full_inchikey)

    condition_columns = [
        "condition_id",
        "cell_id",
        "pert_id",
        "pert_dose",
        "pert_dose_unit",
        "pert_time",
        "pert_time_unit",
        "n_treatment_instances",
        "n_control_groups",
    ]
    lincs_conditions = pd.read_parquet(
        paths["lincs_conditions"], columns=condition_columns
    )
    lincs_conditions["cell_id"] = lincs_conditions["cell_id"].astype("string").str.strip()
    lincs_conditions["pert_id"] = lincs_conditions["pert_id"].astype("string").str.strip()
    if lincs_conditions["condition_id"].duplicated().any():
        raise ValueError("LINCS condition_id is not unique in the approved condition metadata")

    frames = {
        "ctrp_cells": ctrp_cells,
        "ctrp_response_projection": response_projection_rows,
        "response_pair_counts": response_pair_counts,
        "ctrp_drugs": ctrp_drugs,
        "depmap_model": depmap_model,
        "depmap_expression_ids": depmap_expression_ids_frame,
        "approved_crosswalk": approved_crosswalk,
        "lincs_cell_info": lincs_cell_info,
        "lincs_pert_info": lincs_pert_info,
        "lincs_conditions": lincs_conditions,
    }
    source = {
        name: {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }
    source["ctrp_responses"]["projection_columns"] = ["cellid", "drugid"]
    source["ctrp_responses"]["projection_sha256"] = projection_sha256(ctrp_response_projection)
    source["depmap_expression"]["read_columns"] = ["first column: ModelID row identity"]
    source["depmap_expression"]["modelid_axis_sha256"] = projection_sha256(depmap_expression_ids_frame)
    source["approved_breast_crosswalk"]["projection_columns"] = [
        "dataset", "LINCS_cell_id", "LINCS_base_cell_id", "DepMap_ModelID", "identity_status"
    ]
    source["approved_breast_crosswalk"]["projection_sha256"] = projection_sha256(approved_crosswalk)
    source["lincs_conditions"]["projection_columns"] = condition_columns
    source["lincs_conditions"]["projection_sha256"] = projection_sha256(lincs_conditions)
    for key in ("ctrp_cells", "ctrp_drugs", "depmap_model", "lincs_cell_info", "lincs_pert_info"):
        source[key]["projection_sha256"] = projection_sha256(frames[key])

    return {
        "paths": paths,
        "source": source,
        "response_projection_rows": response_projection_rows,
        "response_projection_rows_projected_raw": ctrp_response_projection,
        "response_projection_invalid_rows": response_projection_invalid_rows,
        "response_pair_counts": response_pair_counts,
        "depmap_expression_ids": depmap_expression_ids,
        "depmap_expression_duplicate_ids": depmap_expression_duplicate_ids,
    }, frames


def build_outputs(
    root: Path, run_id: str, outdir: Path, meta: dict[str, Any], frames: dict[str, pd.DataFrame]
) -> dict[str, Any]:
    ctrp_cells = frames["ctrp_cells"].copy()
    ctrp_drugs = frames["ctrp_drugs"].copy()
    depmap_model = frames["depmap_model"].copy()
    crosswalk = frames["approved_crosswalk"].copy()
    lincs_cell_info = frames["lincs_cell_info"].copy()
    lincs_pert_info = frames["lincs_pert_info"].copy()
    lincs_conditions = frames["lincs_conditions"].copy()
    response_pair_counts = meta["response_pair_counts"].copy()
    expression_ids = meta["depmap_expression_ids"]

    model_identity_fields = ["PatientID", "RRID_exact", "OncotreeLineage", "OncotreePrimaryDisease"]
    model_identity_counts = depmap_model.groupby("ModelID", dropna=False)[model_identity_fields].apply(
        lambda block: len(block.drop_duplicates())
    )
    duplicate_model_ids = set(model_identity_counts[model_identity_counts.gt(1)].index.astype(str))
    rrid_to_model, ambiguous_depmap_rrids = unique_key_map(depmap_model, "RRID_exact", "ModelID")

    approved_rows = crosswalk.loc[crosswalk["crosswalk_approved"]].copy()
    crosswalk_models = set(approved_rows["DepMap_ModelID"].dropna().astype(str))
    _, ambiguous_lincs_ids = unique_key_map(approved_rows, "LINCS_cell_id", "DepMap_ModelID")
    cellinfo_base_values = lincs_cell_info.groupby("cell_id")["base_cell_id"].nunique(dropna=True)
    cellinfo_site_values = lincs_cell_info.groupby("cell_id")["primary_site"].nunique(dropna=True)
    ambiguous_cell_info_ids = set(
        cellinfo_base_values[cellinfo_base_values.gt(1)].index.astype(str)
    ) | set(cellinfo_site_values[cellinfo_site_values.gt(1)].index.astype(str))
    cellinfo_unique = lincs_cell_info.loc[
        ~lincs_cell_info["cell_id"].astype(str).isin(ambiguous_cell_info_ids)
    ].drop_duplicates("cell_id")
    cellinfo_map = cellinfo_unique.set_index("cell_id")
    conditions_per_lincs_cell = lincs_conditions.groupby("cell_id").size().to_dict()

    accessions_per_ctrp_cell = ctrp_cells.groupby("cellid")["rrid_exact"].nunique(dropna=True)
    ctrp_cellids_per_accession = ctrp_cells.loc[
        ctrp_cells["rrid_exact"].ne(""), ["rrid_exact", "cellid"]
    ].drop_duplicates().groupby("rrid_exact")["cellid"].nunique()

    depmap_by_id = depmap_model.drop_duplicates("ModelID").set_index("ModelID")
    breast_models = set(depmap_model.loc[depmap_model["is_breast_oncotree"], "ModelID"].dropna().astype(str))

    cell_audit_rows: list[dict[str, Any]] = []
    eligible_cell_rows: list[dict[str, Any]] = []
    response_pair_count_by_cellid = response_pair_counts.groupby("cellid").size().to_dict()
    response_record_count_by_cellid = response_pair_counts.groupby("cellid")[
        "CTRP_response_metadata_record_count"
    ].sum().to_dict()
    for row in ctrp_cells.itertuples(index=False):
        cellid = safe_text(getattr(row, "cellid"))
        rrid = safe_text(getattr(row, "rrid_exact"))
        primary_site = getattr(row, "ccle_primary_site")
        reasons: list[str] = []
        model_id = rrid_to_model.get(rrid)
        matched_model = depmap_by_id.loc[model_id] if model_id in depmap_by_id.index else None

        if not rrid:
            reasons.append("missing_CTRP_Cellosaurus_accession")
        elif rrid in ambiguous_depmap_rrids:
            reasons.append("ambiguous_DepMap_RRID_to_ModelID")
        elif rrid not in rrid_to_model:
            reasons.append("no_exact_DepMap_RRID_match")
        elif accessions_per_ctrp_cell.get(cellid, 0) > 1:
            reasons.append("one_CTRP_cellid_has_multiple_Cellosaurus_accessions")
        elif ctrp_cellids_per_accession.get(rrid, 0) > 1:
            reasons.append("one_Cellosaurus_accession_has_multiple_CTRP_cellids")
        elif model_id in duplicate_model_ids:
            reasons.append("duplicate_DepMap_ModelID_identity_rows")
        elif not bool(matched_model["is_breast_oncotree"]):
            reasons.append("DepMap_OncotreeLineage_not_Breast")
        elif model_id not in expression_ids:
            reasons.append("ModelID_absent_from_DepMap_expression_row_axis")
        elif model_id not in crosswalk_models:
            reasons.append("no_approved_GSE92742_breast_crosswalk_for_ModelID")

        mapped_lincs_ids = sorted(
            crosswalk.loc[
                crosswalk["crosswalk_approved"] & crosswalk["DepMap_ModelID"].eq(model_id),
                "LINCS_cell_id",
            ].dropna().astype(str).unique().tolist()
        ) if model_id else []
        observed_lincs_ids = [cell for cell in mapped_lincs_ids if cell in cellinfo_map.index]
        bad_lincs_ids = [cell for cell in mapped_lincs_ids if cell not in cellinfo_map.index]
        if not reasons and not observed_lincs_ids:
            reasons.append("approved_crosswalk_cell_id_missing_official_LINCS_cell_info")

        matched_site = ""
        lineage = ""
        patient_id = ""
        primary_disease = ""
        if matched_model is not None:
            matched_site = safe_text(primary_site)
            lineage = safe_text(matched_model["OncotreeLineage"])
            patient_id = safe_text(matched_model["PatientID"])
            primary_disease = safe_text(matched_model["OncotreePrimaryDisease"])
        if not reasons and observed_lincs_ids:
            eligible_cell_rows.append(
                {
                    "CTRP_cellid": cellid,
                    "CTRP_Cellosaurus_accession": rrid,
                    "DepMap_ModelID": model_id,
                    "DepMap_RRID": safe_text(matched_model["RRID"]),
                    "DepMap_PatientID": patient_id,
                    "DepMap_OncotreeLineage": lineage,
                    "DepMap_OncotreePrimaryDisease": primary_disease,
                    "DepMap_expression_ModelID_row_present": True,
                    "LINCS_cell_ids": ";".join(observed_lincs_ids),
                    "CTRP_primary_site_metadata": matched_site,
                }
            )

        reason = ";".join(reasons)
        if not reason:
            reason = "exact_identity_RNA_crosswalk_and_LINCS_cell_info_pass"
        cell_audit_rows.append(
            {
                "CTRP_cellid": cellid,
                "CTRP_Cellosaurus_accession": rrid,
                "CTRP_primary_site_metadata": safe_text(primary_site),
                "DepMap_ModelID_exact_RRID_match": model_id or "",
                "DepMap_RRID": safe_text(matched_model["RRID"]) if matched_model is not None else "",
                "DepMap_PatientID": patient_id,
                "DepMap_OncotreeLineage": lineage,
                "DepMap_OncotreePrimaryDisease": primary_disease,
                "DepMap_expression_ModelID_row_present": bool(model_id and model_id in expression_ids),
                "approved_GSE92742_LINCS_cell_ids": ";".join(mapped_lincs_ids),
                "LINCS_cell_ids_in_official_cell_info": ";".join(observed_lincs_ids),
                "LINCS_cell_ids_missing_official_cell_info": ";".join(bad_lincs_ids),
                "n_LINCS_conditions_for_approved_cells": int(
                    sum(conditions_per_lincs_cell.get(cell, 0) for cell in observed_lincs_ids)
                ),
                "CTRP_response_metadata_pair_count": int(response_pair_count_by_cellid.get(cellid, 0)),
                "CTRP_response_metadata_record_count": int(response_record_count_by_cellid.get(cellid, 0)),
                "identity_exclusion_or_status": reason,
            }
        )

    cell_audit = pd.DataFrame(cell_audit_rows).drop_duplicates()
    eligible_cells = pd.DataFrame(eligible_cell_rows).drop_duplicates()

    # Exact full-key drug maps. Aliases sharing one key are retained as separate source IDs.
    ctrp_drugid_key_counts = ctrp_drugs.groupby("drugid")["full_inchikey"].nunique(dropna=True)
    ambiguous_ctrp_drugids = set(ctrp_drugid_key_counts[ctrp_drugid_key_counts.gt(1)].index.astype(str))
    ctrp_drug_map = ctrp_drugs.loc[
        ctrp_drugs["full_inchikey"].ne("")
        & ~ctrp_drugs["drugid"].astype(str).isin(ambiguous_ctrp_drugids),
        ["drugid", "full_inchikey"],
    ].drop_duplicates()

    pert_info = lincs_pert_info.loc[lincs_pert_info["pert_type"].eq("trt_cp")].copy()
    pert_id_key_counts = pert_info.groupby("pert_id")["full_inchikey"].nunique(dropna=True)
    ambiguous_pert_ids = set(pert_id_key_counts[pert_id_key_counts.gt(1)].index.astype(str))
    pert_map = pert_info.loc[
        pert_info["full_inchikey"].ne("")
        & ~pert_info["pert_id"].astype(str).isin(ambiguous_pert_ids),
        ["pert_id", "full_inchikey"],
    ].drop_duplicates()

    ctrp_pairs_breast = response_pair_counts.merge(
        cell_audit.loc[
            cell_audit["DepMap_ModelID_exact_RRID_match"].fillna("").astype(str).ne("")
            & cell_audit["DepMap_OncotreeLineage"].astype(str).str.casefold().eq("breast")
            & ~cell_audit["identity_exclusion_or_status"].str.contains(
                "ambiguous|multiple|duplicate", case=False, regex=True
            ),
            ["CTRP_cellid", "DepMap_ModelID_exact_RRID_match", "DepMap_PatientID", "DepMap_RRID", "CTRP_Cellosaurus_accession"],
        ].drop_duplicates(),
        left_on="cellid",
        right_on="CTRP_cellid",
        how="inner",
        validate="many_to_one",
    )
    # The right-hand audit already carries CTRP_cellid. Drop the left join key
    # before renaming cellid so the merged frame has one unambiguous cell axis.
    ctrp_pairs_breast = ctrp_pairs_breast.drop(columns=["cellid"])
    require_unique_columns(ctrp_pairs_breast, "CTRP cell identity merge")
    ctrp_pairs_breast = ctrp_pairs_breast.merge(
        ctrp_drug_map,
        left_on="drugid",
        right_on="drugid",
        how="left",
        validate="many_to_one",
    )
    require_unique_columns(ctrp_pairs_breast, "CTRP drug identity merge")
    ctrp_pairs_breast["CTRP_full_InChIKey_exact"] = ctrp_pairs_breast["full_inchikey"].fillna("")
    ctrp_pairs_breast["CTRP_drug_identity_status"] = "exact_full_InChIKey"
    ctrp_pairs_breast.loc[
        ctrp_pairs_breast["drugid"].astype(str).isin(ambiguous_ctrp_drugids),
        "CTRP_drug_identity_status",
    ] = "excluded_ambiguous_drugid_to_multiple_full_InChIKeys"
    ctrp_pairs_breast.loc[
        ctrp_pairs_breast["CTRP_full_InChIKey_exact"].eq("")
        & ~ctrp_pairs_breast["drugid"].astype(str).isin(ambiguous_ctrp_drugids),
        "CTRP_drug_identity_status",
    ] = "excluded_missing_or_incomplete_full_InChIKey"

    model_lincs_map = crosswalk.loc[
        crosswalk["crosswalk_approved"]
        & crosswalk["DepMap_ModelID"].isin(breast_models),
        ["LINCS_cell_id", "LINCS_base_cell_id", "DepMap_ModelID"],
    ].drop_duplicates()
    official_cell_info = cellinfo_unique.rename(
        columns={
            "cell_id": "LINCS_official_cell_id",
            "base_cell_id": "LINCS_official_base_cell_id",
            "primary_site": "LINCS_primary_site",
        }
    )
    model_lincs_map = model_lincs_map.merge(
        official_cell_info[["LINCS_official_cell_id", "LINCS_official_base_cell_id", "LINCS_primary_site"]],
        left_on="LINCS_cell_id",
        right_on="LINCS_official_cell_id",
        how="inner",
        validate="one_to_one",
    )
    require_unique_columns(model_lincs_map, "LINCS approved cell crosswalk merge")
    model_lincs_map = model_lincs_map.loc[
        model_lincs_map["LINCS_base_cell_id"].fillna("").str.strip()
        .eq(model_lincs_map["LINCS_official_base_cell_id"].fillna("").str.strip())
    ].copy()
    if model_lincs_map["LINCS_cell_id"].duplicated().any():
        raise ValueError("Approved LINCS breast crosswalk has duplicate LINCS cell identities")

    conditions = lincs_conditions.merge(
        model_lincs_map,
        left_on="cell_id",
        right_on="LINCS_cell_id",
        how="inner",
        validate="many_to_one",
    )
    require_unique_columns(conditions, "LINCS condition to approved cell crosswalk merge")
    if not conditions["cell_id"].astype("string").eq(conditions["LINCS_cell_id"]).all():
        raise ValueError("LINCS condition cell_id does not equal its approved crosswalk cell identity")
    conditions = conditions.drop(columns=["cell_id"])
    conditions = conditions.merge(pert_map, on="pert_id", how="left", validate="many_to_one")
    require_unique_columns(conditions, "LINCS condition to perturbagen identity merge")
    conditions["full_inchikey"] = conditions["full_inchikey"].fillna("")
    conditions["dose_time_key"] = (
        conditions["pert_dose"].astype("string").fillna("").str.strip()
        + " " + conditions["pert_dose_unit"].astype("string").fillna("").str.strip()
        + " × " + conditions["pert_time"].astype("string").fillna("").str.strip()
        + " " + conditions["pert_time_unit"].astype("string").fillna("").str.strip()
    ).str.strip()
    conditions = conditions.rename(
        columns={
            "condition_id": "LINCS_condition_id",
            "pert_dose": "LINCS_pert_dose",
            "pert_dose_unit": "LINCS_pert_dose_unit",
            "pert_time": "LINCS_pert_time",
            "pert_time_unit": "LINCS_pert_time_unit",
            "n_treatment_instances": "LINCS_n_treatment_instances_metadata",
            "n_control_groups": "LINCS_n_control_groups_metadata",
        }
    )
    ctrp_pairs_breast = ctrp_pairs_breast.rename(
        columns={
            "cellid": "CTRP_cellid",
            "drugid": "CTRP_drugid",
            "DepMap_ModelID_exact_RRID_match": "DepMap_ModelID",
            "DepMap_PatientID": "DepMap_PatientID",
        }
    )
    conditions_for_join = conditions[
        [
            "LINCS_condition_id", "LINCS_cell_id", "LINCS_base_cell_id", "DepMap_ModelID",
            "pert_id", "full_inchikey", "LINCS_pert_dose", "LINCS_pert_dose_unit",
            "LINCS_pert_time", "LINCS_pert_time_unit", "LINCS_n_treatment_instances_metadata",
            "LINCS_n_control_groups_metadata", "LINCS_official_base_cell_id", "LINCS_primary_site", "dose_time_key",
        ]
    ].rename(columns={"pert_id": "LINCS_pert_id", "full_inchikey": "LINCS_full_InChIKey_exact"})
    require_unique_columns(conditions_for_join, "LINCS condition join projection")

    # Full exact condition-key join. CTRP presence means a metadata record exists;
    # numeric sensitivity labels have not been opened or checked for finiteness.
    ctrp_pairs_with_full_key = ctrp_pairs_breast.loc[
        ctrp_pairs_breast["CTRP_full_InChIKey_exact"].ne("")
    ].copy()
    conditions_with_full_key = conditions_for_join.loc[
        conditions_for_join["LINCS_full_InChIKey_exact"].ne("")
    ].copy()
    condition_join_all = ctrp_pairs_with_full_key.merge(
        conditions_with_full_key,
        left_on=["DepMap_ModelID", "CTRP_full_InChIKey_exact"],
        right_on=["DepMap_ModelID", "LINCS_full_InChIKey_exact"],
        how="inner",
        validate="many_to_many",
    )
    require_unique_columns(condition_join_all, "exact full-InChIKey condition join")
    triple_conditions = condition_join_all.loc[
        condition_join_all["DepMap_ModelID"].isin(expression_ids)
    ].copy()
    triple_conditions["identity_key"] = (
        triple_conditions["DepMap_ModelID"].astype(str)
        + "|" + triple_conditions["CTRP_cellid"].astype(str)
        + "|" + triple_conditions["CTRP_drugid"].astype(str)
        + "|" + triple_conditions["CTRP_full_InChIKey_exact"].astype(str)
        + "|" + triple_conditions["LINCS_cell_id"].astype(str)
        + "|" + triple_conditions["LINCS_pert_id"].astype(str)
        + "|" + triple_conditions["LINCS_condition_id"].astype(str)
    )
    triple_conditions = triple_conditions.sort_values(
        ["DepMap_PatientID", "DepMap_ModelID", "CTRP_full_InChIKey_exact", "LINCS_pert_dose", "LINCS_pert_time", "LINCS_condition_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    condition_counts = (
        condition_join_all.groupby(
            ["DepMap_ModelID", "CTRP_cellid", "CTRP_drugid", "CTRP_full_InChIKey_exact"],
            as_index=False,
            dropna=False,
        )
        .agg(
            n_LINCS_conditions=("LINCS_condition_id", "nunique"),
            n_LINCS_cell_ids=("LINCS_cell_id", "nunique"),
            n_LINCS_pert_ids=("LINCS_pert_id", "nunique"),
            LINCS_condition_ids=("LINCS_condition_id", lambda values: ";".join(sorted(set(values.astype(str))))),
            LINCS_cell_ids=("LINCS_cell_id", lambda values: ";".join(sorted(set(values.astype(str))))),
            LINCS_pert_ids=("LINCS_pert_id", lambda values: ";".join(sorted(set(values.astype(str))))),
            LINCS_dose_time_conditions=("dose_time_key", lambda values: ";".join(sorted(set(values.astype(str))))),
        )
    )
    pair_table = ctrp_pairs_breast.merge(
        condition_counts,
        left_on=["DepMap_ModelID", "CTRP_cellid", "CTRP_drugid", "CTRP_full_InChIKey_exact"],
        right_on=["DepMap_ModelID", "CTRP_cellid", "CTRP_drugid", "CTRP_full_InChIKey_exact"],
        how="left",
        validate="one_to_one",
    )
    for col in ("n_LINCS_conditions", "n_LINCS_cell_ids", "n_LINCS_pert_ids"):
        pair_table[col] = pair_table[col].fillna(0).astype(int)
    for col in ("LINCS_condition_ids", "LINCS_cell_ids", "LINCS_pert_ids", "LINCS_dose_time_conditions"):
        pair_table[col] = pair_table[col].fillna("")
    pair_table["RNA_expression_row_present"] = pair_table["DepMap_ModelID"].isin(expression_ids)
    pair_table["CTRP_response_numeric_value_read"] = False
    pair_table["threeway_exact_condition_coverage"] = (
        pair_table["RNA_expression_row_present"]
        & pair_table["n_LINCS_conditions"].gt(0)
        & pair_table["CTRP_full_InChIKey_exact"].ne("")
    )
    pair_table["coverage_status"] = "no_exact_LINCS_condition_for_same_ModelID_and_full_InChIKey"
    pair_table.loc[
        pair_table["CTRP_drug_identity_status"].ne("exact_full_InChIKey"),
        "coverage_status",
    ] = "CTRP_response_record_without_usable_full_InChIKey"
    pair_table.loc[
        pair_table["n_LINCS_conditions"].gt(0) & ~pair_table["RNA_expression_row_present"],
        "coverage_status",
    ] = "exact_CTRP_and_LINCS_identity_but_no_DepMap_expression_ModelID_row"
    pair_table.loc[
        pair_table["threeway_exact_condition_coverage"], "coverage_status"
    ] = "exact_RNA_CTRP_record_and_LINCS_condition"

    # Drug denominator and cell-drug coverage audit, limited to CTRP drug IDs
    # present in the permitted response projection.
    observed_ctrp_drugids = set(response_pair_counts["drugid"].dropna().astype(str))
    response_drug_counts = response_pair_counts.groupby("drugid")["cellid"].nunique().to_dict()
    observed_ctrp_drugs = ctrp_drugs.loc[ctrp_drugs["drugid"].astype(str).isin(observed_ctrp_drugids)].copy()
    lincs_key_counts = pert_map.groupby("full_inchikey")["pert_id"].nunique().to_dict()
    breast_condition_key_counts = conditions.loc[conditions["full_inchikey"].ne("")].groupby("full_inchikey")["LINCS_condition_id"].nunique().to_dict()
    drug_audit_rows: list[dict[str, Any]] = []
    for drugid in sorted(observed_ctrp_drugids):
        rows = observed_ctrp_drugs.loc[observed_ctrp_drugs["drugid"].astype(str).eq(drugid)]
        keys = sorted(set(rows["full_inchikey"].dropna().astype(str)) - {""})
        full_key = keys[0] if len(keys) == 1 and drugid not in ambiguous_ctrp_drugids else ""
        if drugid in ambiguous_ctrp_drugids:
            status = "excluded_ambiguous_CTRP_drugid_to_multiple_full_InChIKeys"
        elif not full_key:
            status = "excluded_missing_or_incomplete_full_InChIKey"
        elif full_key not in set(pert_map["full_inchikey"].astype(str)):
            status = "no_exact_full_InChIKey_in_LINCS_pert_info"
        elif not breast_condition_key_counts.get(full_key, 0):
            status = "exact_LINCS_structure_metadata_but_no_approved_breast_line_condition"
        else:
            status = "exact_full_InChIKey_and_breast_LINCS_condition_metadata"
        drug_audit_rows.append(
            {
                "CTRP_drugid": drugid,
                "CTRP_full_InChIKey_exact": full_key,
                "CTRP_identity_row_count": int(len(rows)),
                "n_CTRP_cellids_with_response_metadata_record": int(response_drug_counts.get(drugid, 0)),
                "n_LINCS_pert_ids_with_exact_full_InChIKey": int(lincs_key_counts.get(full_key, 0)) if full_key else 0,
                "n_approved_breast_LINCS_conditions_for_key": int(breast_condition_key_counts.get(full_key, 0)) if full_key else 0,
                "drug_exclusion_or_status": status,
            }
        )
    drug_audit = pd.DataFrame(drug_audit_rows)

    lincs_cell_audit_rows: list[dict[str, Any]] = []
    cell_info_unique_map, _ = unique_key_map(
        cellinfo_unique, "cell_id", "base_cell_id"
    )
    for row in crosswalk.itertuples(index=False):
        lincs_id = safe_text(getattr(row, "LINCS_cell_id"))
        base_id = safe_text(getattr(row, "LINCS_base_cell_id"))
        model_id = safe_text(getattr(row, "DepMap_ModelID"))
        approved = bool(getattr(row, "crosswalk_approved"))
        reasons: list[str] = []
        info_base = cell_info_unique_map.get(lincs_id, "")
        if not approved:
            reasons.append("crosswalk_identity_status_not_approved_exact_mapping")
        if lincs_id in ambiguous_lincs_ids:
            reasons.append("approved_crosswalk_LINCS_cell_id_maps_to_multiple_ModelIDs")
        if lincs_id in ambiguous_cell_info_ids:
            reasons.append("LINCS_cell_id_has_conflicting_official_cell_info_metadata")
        elif not info_base:
            reasons.append("LINCS_cell_id_absent_or_ambiguous_in_official_cell_info")
        elif info_base != base_id:
            reasons.append("approved_crosswalk_base_cell_id_conflicts_with_official_cell_info")
        if model_id not in breast_models:
            reasons.append("DepMap_ModelID_not_Breast_by_OncotreeLineage")
        if model_id not in expression_ids:
            reasons.append("ModelID_absent_from_DepMap_expression_row_axis")
        if not reasons:
            reasons.append("approved_crosswalk_and_RNA_row_identity_pass")
        lincs_cell_audit_rows.append(
            {
                "LINCS_dataset": APPROVED_LINCS_DATASET,
                "LINCS_cell_id": lincs_id,
                "LINCS_base_cell_id_from_approved_crosswalk": base_id,
                "LINCS_base_cell_id_from_official_cell_info": info_base,
                "DepMap_ModelID": model_id,
                "DepMap_PatientID": safe_text(depmap_by_id.loc[model_id, "PatientID"]) if model_id in depmap_by_id.index else "",
                "crosswalk_identity_status": safe_text(getattr(row, "identity_status")),
                "DepMap_expression_ModelID_row_present": bool(model_id and model_id in expression_ids),
                "n_LINCS_condition_metadata_rows": int(conditions_per_lincs_cell.get(lincs_id, 0)),
                "identity_exclusion_or_status": ";".join(reasons),
            }
        )
    lincs_cell_audit = pd.DataFrame(lincs_cell_audit_rows).drop_duplicates()

    # Include only source identifiers and metadata; never use subtype fields from the approved crosswalk.
    identity_columns = [
        "identity_key", "DepMap_ModelID", "DepMap_PatientID", "DepMap_RRID",
        "CTRP_cellid", "CTRP_Cellosaurus_accession", "CTRP_drugid",
        "CTRP_full_InChIKey_exact", "LINCS_full_InChIKey_exact", "CTRP_response_metadata_record_count",
        "RNA_expression_row_present", "LINCS_cell_id", "LINCS_base_cell_id",
        "LINCS_pert_id", "LINCS_condition_id", "LINCS_pert_dose", "LINCS_pert_dose_unit",
        "LINCS_pert_time", "LINCS_pert_time_unit", "LINCS_n_treatment_instances_metadata",
        "LINCS_n_control_groups_metadata", "LINCS_official_base_cell_id", "LINCS_primary_site",
    ]
    # `RNA_expression_row_present` follows from an exact eligible identity join;
    # numeric expression values and the gene header were not read.
    triple_conditions["RNA_expression_row_present"] = triple_conditions["DepMap_ModelID"].isin(expression_ids)
    for column in identity_columns:
        if column not in triple_conditions.columns:
            triple_conditions[column] = ""
    triple_conditions_out = triple_conditions[identity_columns].drop_duplicates()

    # Independent units are patient-origin IDs, not condition/instance counts.
    matched_models = set(pair_table.loc[pair_table["threeway_exact_condition_coverage"], "DepMap_ModelID"].dropna().astype(str))
    matched_patients = set(
        pair_table.loc[pair_table["threeway_exact_condition_coverage"], "DepMap_PatientID"].dropna().astype(str)
    ) - {""}
    model_patient = depmap_model.loc[
        depmap_model["ModelID"].astype(str).isin(matched_models), ["ModelID", "PatientID"]
    ].drop_duplicates()
    patient_model_counts = model_patient.loc[model_patient["PatientID"].notna() & model_patient["PatientID"].ne("")].groupby("PatientID")["ModelID"].nunique()
    shared_patient_groups = {
        str(patient): sorted(model_patient.loc[model_patient["PatientID"].eq(patient), "ModelID"].astype(str).unique().tolist())
        for patient in patient_model_counts[patient_model_counts.gt(1)].index
    }

    condition_rows_with_keys = conditions.loc[conditions["full_inchikey"].ne("")].copy()
    summary = {
        "run_id": run_id,
        "status": "COMPLETED_RESPONSE_BLIND_THREEWAY_METADATA_INVENTORY",
        "run_scope": "identity_and_condition_metadata_only_no_training_or_label_scoring",
        "identity_contract": {
            "CTRP_to_DepMap": "exact DepMap Model.RRID == CTRPv2 Cellosaurus.Accession.id after trim and uppercase only; punctuation preserved",
            "LINCS_to_DepMap": "approved EXP-050 GSE92742 breast crosswalk used as frozen; no name-based rematching",
            "drug_identity": "complete standard full InChIKey exact equality only; malformed/partial keys excluded",
            "RNA_presence": "DepMap expression CSV first column (ModelID row identity) only",
            "breast_definition": "DepMap Model.csv OncotreeLineage exactly Breast; no basal_B or other subtype inference",
        },
        "read_scope": {
            "CTRP_responses.csv_columns": ["cellid", "drugid"],
            "DepMap_expression_file_columns": ["first column only: ModelID row identity"],
            "LINCS_condition_columns": [
                "condition_id", "cell_id", "pert_id", "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit",
                "n_treatment_instances", "n_control_groups",
            ],
            "LINCS_pert_info_columns": ["pert_id", "pert_type", "inchi_key"],
            "approved_crosswalk_columns": ["dataset", "LINCS_cell_id", "LINCS_base_cell_id", "DepMap_ModelID", "identity_status"],
            "not_read": [
                "CTRP response numeric columns (AAC/EC50/IC50 and all other responses columns)",
                "LINCS treatment/control/delta/expression arrays",
                "DepMap expression values and gene header",
                "PDO/outer/locked/PANACEA/CDS-B numeric data or labels",
                "crosswalk subtype/status columns such as LegacyMolecularSubtype/basal_B/verified_TNBC",
                "EXP-007 train/val/locked roles",
            ],
        },
        "coverage_denominators": {
            "CTRP_response_projection_rows_cellid_drugid_only": int(len(meta["response_projection_rows_projected_raw"])),
            "CTRP_response_projection_rows_with_nonempty_cellid_and_drugid": int(len(meta["response_projection_rows"])),
            "CTRP_response_projection_rows_missing_or_empty_cellid_or_drugid": int(meta["response_projection_invalid_rows"]),
            "CTRP_unique_cellid_drugid_pairs_all": int(len(response_pair_counts)),
            "DepMap_expression_ModelID_axis_duplicate_rows": int(meta["depmap_expression_duplicate_ids"]),
            "DepMap_expression_ModelID_axis_unique_ids": int(len(expression_ids)),
            "CTRP_cell_identity_rows": int(len(ctrp_cells)),
            "CTRP_cells_exact_RRID_matched_to_DepMap_breast": int(
                cell_audit.loc[
                    cell_audit["DepMap_ModelID_exact_RRID_match"].fillna("").astype(str).ne("")
                    & cell_audit["DepMap_OncotreeLineage"].astype(str).str.casefold().eq("breast")
                    & ~cell_audit["identity_exclusion_or_status"].str.contains(
                        "ambiguous|multiple|duplicate", case=False, regex=True
                    ),
                    "CTRP_cellid",
                ].nunique()
            ),
            "CTRP_cells_exact_RRID_matched_to_DepMap_breast_with_expression_ModelID_row": int(
                cell_audit.loc[
                    cell_audit["DepMap_ModelID_exact_RRID_match"].fillna("").astype(str).ne("")
                    & cell_audit["DepMap_OncotreeLineage"].astype(str).str.casefold().eq("breast")
                    & cell_audit["DepMap_expression_ModelID_row_present"]
                    & ~cell_audit["identity_exclusion_or_status"].str.contains(
                        "ambiguous|multiple|duplicate", case=False, regex=True
                    ),
                    "CTRP_cellid",
                ].nunique()
            ),
            "CTRP_cellids_with_exact_breast_RNA_crosswalk_and_official_LINCS_cell_info": int(
                cell_audit.loc[
                    cell_audit["identity_exclusion_or_status"].eq("exact_identity_RNA_crosswalk_and_LINCS_cell_info_pass"),
                    "CTRP_cellid",
                ].nunique()
            ),
            "CTRP_unique_cell_drug_pairs_on_exactly_mapped_breast_lines": int(len(ctrp_pairs_breast)),
            "CTRP_response_metadata_records_on_exactly_mapped_breast_lines": int(
                ctrp_pairs_breast["CTRP_response_metadata_record_count"].sum()
            ),
            "approved_GSE92742_breast_crosswalk_rows": int(crosswalk["crosswalk_approved"].sum()),
            "approved_GSE92742_crosswalk_cell_ids": int(
                crosswalk.loc[crosswalk["crosswalk_approved"], "LINCS_cell_id"].nunique()
            ),
            "approved_LINCS_cells_with_condition_metadata": int(lincs_cell_audit["n_LINCS_condition_metadata_rows"].gt(0).sum()),
            "breast_LINCS_condition_metadata_rows_after_exact_identity": int(len(conditions)),
            "breast_LINCS_condition_metadata_rows_with_DepMap_expression_ModelID_row": int(
                conditions["DepMap_ModelID"].isin(expression_ids).sum()
            ),
            "breast_LINCS_condition_metadata_rows_with_complete_full_InChIKey": int(len(condition_rows_with_keys)),
            "threeway_exact_cell_drug_condition_rows": int(len(triple_conditions_out)),
            "threeway_unique_CTRP_cell_drug_pairs_with_at_least_one_LINCS_condition": int(
                pair_table["threeway_exact_condition_coverage"].sum()
            ),
            "threeway_unique_DepMap_ModelID_InChIKey_pairs": int(
                pair_table.loc[pair_table["threeway_exact_condition_coverage"], ["DepMap_ModelID", "CTRP_full_InChIKey_exact"]].drop_duplicates().shape[0]
            ),
            "threeway_unique_DepMap_ModelID_cells": int(len(matched_models)),
            "threeway_unique_DepMap_PatientID_independent_objects_known": int(len(matched_patients)),
            "threeway_rows_with_unknown_PatientID": int(
                pair_table.loc[pair_table["threeway_exact_condition_coverage"], "DepMap_PatientID"].isna().sum()
                + pair_table.loc[pair_table["threeway_exact_condition_coverage"], "DepMap_PatientID"].astype("string").eq("").sum()
            ),
            "threeway_unique_CTRP_drugid": int(
                pair_table.loc[pair_table["threeway_exact_condition_coverage"], "CTRP_drugid"].nunique()
            ),
            "threeway_unique_full_InChIKeys": int(
                pair_table.loc[pair_table["threeway_exact_condition_coverage"], "CTRP_full_InChIKey_exact"].nunique()
            ),
            "unique_LINCS_condition_ids_in_threeway": int(triple_conditions_out["LINCS_condition_id"].nunique()),
            "LINCS_treatment_instances_metadata_total_for_unique_conditions": int(
                pd.to_numeric(
                    triple_conditions_out.drop_duplicates("LINCS_condition_id")["LINCS_n_treatment_instances_metadata"],
                    errors="coerce",
                ).fillna(0).sum()
            ),
            "LINCS_control_groups_metadata_total_for_unique_conditions": int(
                pd.to_numeric(
                    triple_conditions_out.drop_duplicates("LINCS_condition_id")["LINCS_n_control_groups_metadata"],
                    errors="coerce",
                ).fillna(0).sum()
            ),
            "CTRP_response_numeric_values_read": 0,
            "LINCS_expression_or_delta_values_read": 0,
            "DepMap_expression_values_read": 0,
        },
        "independence_and_repetition": {
            "independent_object_key": "DepMap PatientID; condition IDs, CTRP response rows, LINCS treatment instances, and LINCS condition rows are not independent-object counts",
            "matched_PatientID_groups_with_multiple_DepMap_ModelIDs": shared_patient_groups,
            "interpretation": "Cross-study identity overlap represents the same cell-line-derived model across modalities, not independent validation. LINCS condition and instance counts are metadata only and do not prove independent biological culture replicates.",
            "CTRP_repeats_limit": "Only cellid/drugid were read; repeated rows cannot be assigned to experiments, dose levels, or independent cultures from this projection.",
            "exposure_alignment_limit": "CTRP response metadata projection has no dose/time; LINCS dose/time are reported as measured condition metadata and no matched-exposure claim is made.",
        },
        "fold_isolation": {
            "status": "NOT_APPLICABLE_METADATA_ONLY_NO_MODEL_FOLDS_CREATED",
            "folds_created": 0,
            "reason": "EXP-068 package scope is response-blind identity/coverage inventory only; no model fitting, selection, or label use occurs.",
            "split_role_files_read": False,
            "locked_or_outer_numeric_values_read": False,
        },
        "gene_axis": {
            "status": "NOT_EVALUATED_IN_THIS_METADATA_ONLY_INVENTORY",
            "reason": "Only the DepMap expression ModelID row identity was read. The expression header and values were not read; downstream training must bind its frozen gene-axis contract separately.",
        },
        "cell_identity_exclusion_counts": dict(Counter(cell_audit["identity_exclusion_or_status"].astype(str))),
        "drug_identity_status_counts": dict(Counter(drug_audit.get("drug_exclusion_or_status", pd.Series(dtype="string")).astype(str))),
        "source_metadata": meta["source"],
        "runtime": {
            "gpu_hours": 0,
            "cpu_seconds": None,
            "wall_seconds": None,
            "environment": "WSL2 Conda drugscreening-gpu",
        },
    }

    # The pair table contains all exact mapped breast CTRP metadata pairs,
    # including pairs with no exact LINCS condition, so missingness stays visible.
    pair_columns = [
        "CTRP_cellid", "CTRP_Cellosaurus_accession", "DepMap_ModelID", "DepMap_RRID",
        "DepMap_PatientID", "CTRP_drugid", "CTRP_full_InChIKey_exact",
        "CTRP_response_metadata_record_count", "CTRP_drug_identity_status",
        "RNA_expression_row_present", "n_LINCS_conditions", "n_LINCS_cell_ids", "n_LINCS_pert_ids",
        "LINCS_condition_ids", "LINCS_cell_ids", "LINCS_pert_ids", "LINCS_dose_time_conditions",
        "threeway_exact_condition_coverage", "CTRP_response_numeric_value_read", "coverage_status",
    ]
    for column in pair_columns:
        if column not in pair_table.columns:
            pair_table[column] = ""
    pair_table_out = pair_table[pair_columns].sort_values(
        ["DepMap_PatientID", "DepMap_ModelID", "CTRP_full_InChIKey_exact", "CTRP_cellid", "CTRP_drugid"],
        kind="mergesort",
    ).reset_index(drop=True)

    output_tables = {
        "threeway_cell_drug_condition_keys.tsv": triple_conditions_out,
        "ctrp_cell_drug_coverage.tsv": pair_table_out,
        "ctrp_cell_identity_audit.tsv": cell_audit.sort_values(["CTRP_cellid", "CTRP_Cellosaurus_accession"], kind="mergesort"),
        "ctrp_drug_identity_audit.tsv": drug_audit.sort_values(["CTRP_drugid"], kind="mergesort") if not drug_audit.empty else drug_audit,
        "lincs_breast_cell_identity_audit.tsv": lincs_cell_audit.sort_values(["LINCS_cell_id"], kind="mergesort"),
    }
    for filename, frame in output_tables.items():
        write_tsv(frame, outdir / filename)
    summary["serialization_roundtrip"] = {
        filename: validate_tsv_readback(frame, outdir / filename, filename)
        for filename, frame in output_tables.items()
    }
    summary["runtime"]["cpu_seconds"] = round(float(time.process_time()), 6)
    summary["output_files_sha256"] = {
        filename: sha256_file(outdir / filename) for filename in output_tables
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_readback = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary_readback.get("run_id") != run_id or summary_readback.get("status") != summary["status"]:
        raise ValueError("summary.json readback does not match the in-memory run identity")
    output_hashes = dict(summary["output_files_sha256"])
    output_hashes["summary.json"] = sha256_file(summary_path)
    return {
        "summary": summary,
        "output_tables": output_tables,
        "output_hashes": output_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/experiments/EXP-068/threeway_inventory_v1"),
    )
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base_out = args.output if args.output.is_absolute() else root / args.output
    base_out = base_out.resolve()
    outdir = base_out / args.run_id
    if outdir.exists():
        raise FileExistsError(f"Immutable output directory already exists: {outdir}")
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    outdir.mkdir(parents=True, exist_ok=False)

    meta, frames = load_metadata(root)
    result = build_outputs(root, args.run_id, outdir, meta, frames)
    wall_seconds = time.perf_counter() - start_wall
    cpu_seconds = time.process_time() - start_cpu
    result["summary"]["runtime"]["cpu_seconds"] = round(float(cpu_seconds), 6)
    result["summary"]["runtime"]["wall_seconds"] = round(float(wall_seconds), 6)
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(result["summary"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if json.loads(summary_path.read_text(encoding="utf-8")).get("run_id") != args.run_id:
        raise ValueError("Final summary.json readback does not match --run-id")
    result["output_hashes"]["summary.json"] = sha256_file(summary_path)

    scope_text = """# EXP-068 乳腺癌三方精确身份与条件覆盖清点\n\n\
本产物只做响应盲身份、基线RNA行存在性、药物全结构键和LINCS条件元数据交集。\n\n\
## 读取范围\n\n\
- CTRPv2 `responses.csv`：仅 `cellid`、`drugid` 两列；这些行只表示存在响应记录，不代表已读取或核实数值标签。\n\
- CTRPv2 `cells.csv`：`cellid`、`Cellosaurus.Accession.id`、`ccle_primary_site`。\n\
- CTRPv2 `drugs.csv`：`drugid`、`inchikey`；只接受完整 InChIKey。\n\
- DepMap `Model.csv`：`ModelID`、`PatientID`、`RRID`、`OncotreeLineage`、`OncotreePrimaryDisease`。\n\
- DepMap表达CSV：只读第一列 `ModelID` 行身份；不读表达值或基因表头。\n\
- LINCS：读取已批准乳腺 crosswalk 的身份字段、官方 cell_info 的 `cell_id/base_cell_id/primary_site`、pert_info 的 `pert_id/pert_type/inchi_key`，以及 conditions.parquet 的 condition/cell/drug/dose/time/重复计数元数据。\n\n\
## 禁读范围\n\n\
没有读取CTRP数值响应列、LINCS扰动表达/变化数组、DepMap表达值、PDO/outer/locked/PANACEA/CDS-B数值或标签；没有读取乳腺crosswalk的分子亚型字段，也没有用 `basal_B` 推断TNBC。没有按名称重新连接细胞。\n\n\
本包不产生训练/验证折：这是EXP-068响应盲覆盖清点，未创建模型分折、未读取split role或标签。独立对象分母使用DepMap `PatientID`；CTRP重复响应记录与LINCS条件/处理实例数不能替代独立培养物分母。\n"""
    (outdir / "read_scope.md").write_text(scope_text, encoding="utf-8")

    manifest = {
        "run_id": args.run_id,
        "status": "COMPLETED_RESPONSE_BLIND_THREEWAY_METADATA_INVENTORY",
        "command": (
            "/home/dell/miniconda3/bin/conda run --no-capture-output -n drugscreening-gpu python "
            "/mnt/d/Code/DrugScreenLab/scripts/data/build_exp068_breast_threeway_inventory.py "
            f"--root {root} --output {base_out} --run-id {args.run_id}"
        ),
        "environment": "WSL2 Conda drugscreening-gpu",
        "gpu_hours": 0,
        "cpu_seconds": round(float(cpu_seconds), 6),
        "wall_seconds": round(float(wall_seconds), 6),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "source_metadata": meta["source"],
        "output_files_sha256": {
            **result["output_hashes"],
            "read_scope.md": sha256_file(outdir / "read_scope.md"),
        },
        "forbidden_numeric_files_opened": [],
        "raw_sources_modified": False,
    }
    manifest_path = outdir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if json.loads(manifest_path.read_text(encoding="utf-8")).get("run_id") != args.run_id:
        raise ValueError("run_manifest.json readback does not match --run-id")
    print(json.dumps({
        "output_dir": str(outdir),
        "summary": str(summary_path),
        "run_manifest": str(manifest_path),
        "status": result["summary"]["status"],
        "coverage_denominators": result["summary"]["coverage_denominators"],
        "gpu_hours": 0,
        "cpu_seconds": round(float(cpu_seconds), 6),
        "wall_seconds": round(float(wall_seconds), 6),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
