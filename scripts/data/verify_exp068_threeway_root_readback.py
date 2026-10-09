#!/usr/bin/env python3
"""Independent response-blind root readback for EXP-068 three-way keys.

Only approved identity and condition metadata columns are projected. No CTRP
response values, DepMap expression values, LINCS response arrays, or restricted
PDO/outer/locked/PANACEA/CDS-B values are opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd


FULL_INCHIKEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
APPROVED_DATASET = "GSE92742"
APPROVED_STATUS = "base_cell_id mapped to unique DepMap ModelID"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def projection_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(sep="\t", index=False, lineterminator="\n", na_rep="").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def exact_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def full_key(value: object) -> str:
    candidate = exact_text(value)
    return candidate if FULL_INCHIKEY.fullmatch(candidate) else ""


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype="string", keep_default_na=False)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--attempt-dir",
        type=Path,
        default=Path("artifacts/experiments/EXP-068/threeway_inventory_v1/20261009_attempt_005"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    attempt = (root / args.attempt_dir).resolve()
    output = (root / args.output).resolve() if args.output else attempt / "root_independent_readback.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt: {output}")

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    summary_path = attempt / "summary.json"
    manifest_path = attempt / "run_manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(summary.get("run_id") == manifest.get("run_id") == "20261009_attempt_005", "run identity mismatch")
    require(
        summary.get("status") == manifest.get("status") == "COMPLETED_RESPONSE_BLIND_THREEWAY_METADATA_INVENTORY",
        "run status mismatch",
    )

    output_hashes: dict[str, Any] = {}
    for name, expected_hash in manifest["output_files_sha256"].items():
        path = attempt / name
        actual = sha256_file(path)
        require(actual == expected_hash, f"output hash mismatch: {name}")
        output_hashes[name] = {"bytes": path.stat().st_size, "sha256": actual}

    # Read only the explicitly approved identity/condition projections.
    ctrp_dir = root / "data/processed/exp013/ctrp_v2"
    depmap_dir = root / "data/raw/depmap/24q2"
    lincs_raw = root / "data/raw/lincs/GSE92742"
    conditions_path = root / "data/processed/lincs/GSE92742/continuous_conditions_v2/conditions.parquet"
    crosswalk_path = root / "artifacts/experiments/EXP-050/20261006_breast_subtype_measured_transfer_v1/lincs_breast_cell_model_identity.tsv"

    ctrp_cells = pd.read_csv(
        ctrp_dir / "cells.csv",
        usecols=["cellid", "Cellosaurus.Accession.id", "ccle_primary_site"],
        dtype="string",
    )
    ctrp_cells["cellid"] = ctrp_cells["cellid"].astype("string").str.strip()
    ctrp_cells["rrid_exact"] = ctrp_cells["Cellosaurus.Accession.id"].map(exact_text)

    response_projection_raw = pd.read_csv(
        ctrp_dir / "responses.csv",
        usecols=["cellid", "drugid"],
        dtype={"cellid": "string", "drugid": "string"},
    )
    response_projection = response_projection_raw.copy()
    response_projection["cellid"] = response_projection["cellid"].str.strip()
    response_projection["drugid"] = response_projection["drugid"].str.strip()
    valid = response_projection["cellid"].notna() & response_projection["cellid"].ne("") & response_projection["drugid"].notna() & response_projection["drugid"].ne("")
    response_pairs = (
        response_projection.loc[valid]
        .groupby(["cellid", "drugid"], as_index=False, dropna=False)
        .size()
        .rename(columns={"size": "CTRP_response_metadata_record_count"})
    )

    ctrp_drugs = pd.read_csv(ctrp_dir / "drugs.csv", usecols=["drugid", "inchikey"], dtype="string")
    ctrp_drugs["drugid"] = ctrp_drugs["drugid"].astype("string").str.strip()
    ctrp_drugs["full_inchikey"] = ctrp_drugs["inchikey"].map(full_key)
    ctrp_key_counts = ctrp_drugs.groupby("drugid")["full_inchikey"].nunique(dropna=True)
    ambiguous_drugids = set(ctrp_key_counts[ctrp_key_counts.gt(1)].index.astype(str))
    ctrp_drug_map = ctrp_drugs.loc[
        ctrp_drugs["full_inchikey"].ne("") & ~ctrp_drugs["drugid"].astype(str).isin(ambiguous_drugids),
        ["drugid", "full_inchikey"],
    ].drop_duplicates()

    model = pd.read_csv(
        depmap_dir / "Model.csv",
        usecols=["ModelID", "PatientID", "RRID", "OncotreeLineage", "OncotreePrimaryDisease"],
        dtype="string",
    )
    model["ModelID"] = model["ModelID"].astype("string").str.strip()
    model["RRID_exact"] = model["RRID"].map(exact_text)
    model["PatientID"] = model["PatientID"].astype("string").str.strip()
    model["is_breast"] = model["OncotreeLineage"].fillna("").str.strip().str.casefold().eq("breast")
    require(not model["ModelID"].duplicated().any(), "duplicate DepMap ModelID metadata rows")
    breast = model.loc[model["is_breast"]].copy()
    breast_ids = set(breast["ModelID"].dropna().astype(str))

    # Read only first-column row identities; no header or expression values.
    expression_ids_frame = pd.read_csv(
        depmap_dir / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        header=None,
        skiprows=1,
        usecols=[0],
        names=["ModelID"],
        dtype="string",
    )
    expression_ids_frame["ModelID"] = expression_ids_frame["ModelID"].str.strip()
    require(not expression_ids_frame["ModelID"].dropna().duplicated().any(), "duplicate expression ModelID rows")
    expression_ids = set(expression_ids_frame["ModelID"].dropna().astype(str))

    crosswalk = pd.read_csv(
        crosswalk_path,
        sep="\t",
        usecols=["dataset", "LINCS_cell_id", "LINCS_base_cell_id", "DepMap_ModelID", "identity_status"],
        dtype="string",
    )
    crosswalk = crosswalk.loc[crosswalk["dataset"].eq(APPROVED_DATASET)].copy()
    crosswalk["LINCS_cell_id"] = crosswalk["LINCS_cell_id"].astype("string").str.strip()
    crosswalk["LINCS_base_cell_id"] = crosswalk["LINCS_base_cell_id"].astype("string").str.strip()
    crosswalk["DepMap_ModelID"] = crosswalk["DepMap_ModelID"].astype("string").str.strip()
    crosswalk = crosswalk.loc[crosswalk["identity_status"].eq(APPROVED_STATUS)].copy()

    official_cells = pd.read_csv(
        lincs_raw / "GSE92742_Broad_LINCS_cell_info.txt.gz",
        sep="\t",
        compression="gzip",
        usecols=["cell_id", "base_cell_id", "primary_site"],
        dtype="string",
    )
    for col in ("cell_id", "base_cell_id", "primary_site"):
        official_cells[col] = official_cells[col].astype("string").str.strip()
    base_counts = official_cells.groupby("cell_id")["base_cell_id"].nunique(dropna=True)
    site_counts = official_cells.groupby("cell_id")["primary_site"].nunique(dropna=True)
    ambiguous_cellids = set(base_counts[base_counts.gt(1)].index.astype(str)) | set(site_counts[site_counts.gt(1)].index.astype(str))
    official_unique = official_cells.loc[~official_cells["cell_id"].astype(str).isin(ambiguous_cellids)].drop_duplicates("cell_id")

    pert_info = pd.read_csv(
        lincs_raw / "GSE92742_Broad_LINCS_pert_info.txt.gz",
        sep="\t",
        compression="gzip",
        usecols=["pert_id", "pert_type", "inchi_key"],
        dtype="string",
    )
    pert_info["pert_id"] = pert_info["pert_id"].astype("string").str.strip()
    pert_info["full_inchikey"] = pert_info["inchi_key"].map(full_key)
    pert_info = pert_info.loc[pert_info["pert_type"].eq("trt_cp")].copy()
    pert_key_counts = pert_info.groupby("pert_id")["full_inchikey"].nunique(dropna=True)
    ambiguous_pertids = set(pert_key_counts[pert_key_counts.gt(1)].index.astype(str))
    pert_map = pert_info.loc[
        pert_info["full_inchikey"].ne("") & ~pert_info["pert_id"].astype(str).isin(ambiguous_pertids),
        ["pert_id", "full_inchikey"],
    ].drop_duplicates()

    condition_columns = [
        "condition_id", "cell_id", "pert_id", "pert_dose", "pert_dose_unit",
        "pert_time", "pert_time_unit", "n_treatment_instances", "n_control_groups",
    ]
    conditions = pd.read_parquet(conditions_path, columns=condition_columns)
    conditions["cell_id"] = conditions["cell_id"].astype("string").str.strip()
    conditions["pert_id"] = conditions["pert_id"].astype("string").str.strip()
    require(not conditions["condition_id"].duplicated().any(), "LINCS condition_id duplicate in metadata")

    # Recreate the strictly exact identity path independently.
    rrid_to_model_counts = model.loc[model["RRID_exact"].ne("")].groupby("RRID_exact")["ModelID"].nunique()
    unique_rrids = set(rrid_to_model_counts[rrid_to_model_counts.eq(1)].index.astype(str))
    unique_model_by_rrid = (
        model.loc[model["RRID_exact"].isin(unique_rrids)]
        .drop_duplicates(["RRID_exact", "ModelID"])
        .set_index("RRID_exact")["ModelID"].to_dict()
    )
    cellid_accession_counts = ctrp_cells.loc[ctrp_cells["rrid_exact"].ne("")].groupby("cellid")["rrid_exact"].nunique()
    cellid_ambiguous = set(cellid_accession_counts[cellid_accession_counts.gt(1)].index.astype(str))
    accession_cell_counts = ctrp_cells.loc[ctrp_cells["rrid_exact"].ne(""), ["rrid_exact", "cellid"]].drop_duplicates().groupby("rrid_exact")["cellid"].nunique()
    accession_ambiguous = set(accession_cell_counts[accession_cell_counts.gt(1)].index.astype(str))
    model_by_id = model.drop_duplicates("ModelID").set_index("ModelID")
    mapped_cells: list[dict[str, str]] = []
    for row in ctrp_cells.itertuples(index=False):
        cellid = text(getattr(row, "cellid"))
        rrid = text(getattr(row, "rrid_exact"))
        if not rrid or rrid not in unique_model_by_rrid or cellid in cellid_ambiguous or rrid in accession_ambiguous:
            continue
        model_id = str(unique_model_by_rrid[rrid])
        if model_id not in breast_ids:
            continue
        matched_model = model_by_id.loc[model_id]
        mapped_cells.append(
            {
                "CTRP_cellid": cellid,
                "CTRP_RRID": rrid,
                "DepMap_ModelID": model_id,
                "DepMap_PatientID": text(matched_model["PatientID"]),
            }
        )
    mapped_cell_df = pd.DataFrame(mapped_cells).drop_duplicates()
    require(not mapped_cell_df["CTRP_cellid"].duplicated().any(), "eligible CTRP cellid is not one-to-one")
    require(len(mapped_cell_df) == 40, "exact breast CTRP cell identity denominator mismatch")

    mapped_cell_lincs = crosswalk.loc[crosswalk["DepMap_ModelID"].isin(breast_ids)].merge(
        official_unique.rename(columns={"cell_id": "official_cell_id", "base_cell_id": "official_base_cell_id", "primary_site": "official_primary_site"})[
            ["official_cell_id", "official_base_cell_id", "official_primary_site"]
        ],
        left_on="LINCS_cell_id",
        right_on="official_cell_id",
        how="inner",
        validate="one_to_one",
    )
    mapped_cell_lincs = mapped_cell_lincs.loc[
        mapped_cell_lincs["LINCS_base_cell_id"].fillna("").astype(str).str.strip().eq(
            mapped_cell_lincs["official_base_cell_id"].fillna("").astype(str).str.strip()
        )
    ].copy()
    require(not mapped_cell_lincs["LINCS_cell_id"].duplicated().any(), "approved LINCS cell ID duplicates")
    model_lincs = mapped_cell_lincs[
        ["LINCS_cell_id", "LINCS_base_cell_id", "DepMap_ModelID", "official_base_cell_id", "official_primary_site"]
    ].copy()
    cond = conditions.merge(model_lincs, left_on="cell_id", right_on="LINCS_cell_id", how="inner", validate="many_to_one")
    cond = cond.merge(pert_map, on="pert_id", how="left", validate="many_to_one")
    cond = cond.loc[cond["full_inchikey"].fillna("").ne("")].copy()
    cond["LINCS_full_InChIKey_exact"] = cond["full_inchikey"]
    cond = cond.rename(
        columns={
            "condition_id": "LINCS_condition_id", "pert_dose": "LINCS_pert_dose",
            "pert_dose_unit": "LINCS_pert_dose_unit", "pert_time": "LINCS_pert_time",
            "pert_time_unit": "LINCS_pert_time_unit", "n_treatment_instances": "LINCS_n_treatment_instances_metadata",
            "n_control_groups": "LINCS_n_control_groups_metadata", "pert_id": "LINCS_pert_id",
            "official_primary_site": "LINCS_primary_site", "official_base_cell_id": "LINCS_official_base_cell_id",
        }
    )
    cond = cond.loc[cond["LINCS_base_cell_id"].astype(str).str.strip().eq(cond["LINCS_official_base_cell_id"].astype(str).str.strip())]

    pairs = response_pairs.merge(
        mapped_cell_df,
        left_on="cellid",
        right_on="CTRP_cellid",
        how="inner",
        validate="many_to_one",
    ).merge(ctrp_drug_map, on="drugid", how="left", validate="many_to_one")
    pairs = pairs.loc[pairs["full_inchikey"].fillna("").ne("")].copy()
    pairs = pairs.drop(columns=["cellid"])
    pairs = pairs.rename(
        columns={
            "drugid": "CTRP_drugid", "full_inchikey": "CTRP_full_InChIKey_exact",
            "cellid": "CTRP_cellid",
        }
    )
    pair_lincs = pairs.merge(
        cond,
        left_on=["DepMap_ModelID", "CTRP_full_InChIKey_exact"],
        right_on=["DepMap_ModelID", "LINCS_full_InChIKey_exact"],
        how="inner",
        validate="many_to_many",
        suffixes=("", "_cond"),
    )
    pair_lincs["identity_key"] = (
        pair_lincs["DepMap_ModelID"].astype(str)
        + "|" + pair_lincs["CTRP_cellid"].astype(str)
        + "|" + pair_lincs["CTRP_drugid"].astype(str)
        + "|" + pair_lincs["CTRP_full_InChIKey_exact"].astype(str)
        + "|" + pair_lincs["LINCS_cell_id"].astype(str)
        + "|" + pair_lincs["LINCS_pert_id"].astype(str)
        + "|" + pair_lincs["LINCS_condition_id"].astype(str)
    )
    expected_keys = set(pair_lincs["identity_key"].astype(str))

    # Read back only result metadata and test every exact condition key/axis.
    key_table = read_tsv(attempt / "threeway_cell_drug_condition_keys.tsv")
    key_table["identity_key"] = key_table["identity_key"].astype(str)
    require(len(key_table) == 1832 == summary["coverage_denominators"]["threeway_exact_cell_drug_condition_rows"], "condition denominator mismatch")
    require(not key_table["identity_key"].duplicated().any(), "duplicate serialized identity_key")
    require(set(key_table["identity_key"]) == expected_keys, "serialized condition keys differ from independently recomputed exact join")
    require(key_table["CTRP_full_InChIKey_exact"].eq(key_table["LINCS_full_InChIKey_exact"]).all(), "CTRP and LINCS full InChIKeys differ")
    require(key_table["CTRP_full_InChIKey_exact"].map(lambda x: bool(FULL_INCHIKEY.fullmatch(str(x)))).all(), "malformed full InChIKey in output")
    require(key_table["RNA_expression_row_present"].eq("True").all(), "condition output contains a model without RNA row identity")
    require(key_table["LINCS_pert_dose"].ne("").all() and key_table["LINCS_pert_time"].ne("").all(), "dose/time metadata missing on condition keys")
    require(key_table["LINCS_condition_id"].nunique() == 1832, "LINCS condition_id denominator mismatch")
    require(key_table["DepMap_ModelID"].nunique() == 5 and key_table["DepMap_PatientID"].nunique() == 5, "independent cell-model denominator mismatch")
    require(key_table["CTRP_drugid"].nunique() == 128 and key_table["CTRP_full_InChIKey_exact"].nunique() == 128, "drug denominator mismatch")

    condition_fields = [
        "LINCS_pert_dose", "LINCS_pert_dose_unit", "LINCS_pert_time", "LINCS_pert_time_unit",
        "LINCS_n_treatment_instances_metadata", "LINCS_n_control_groups_metadata",
        "LINCS_base_cell_id", "LINCS_official_base_cell_id", "LINCS_primary_site",
    ]
    expected_condition_fields = pair_lincs[["identity_key", *condition_fields]].copy()
    observed_condition_fields = key_table[["identity_key", *condition_fields]].copy()
    for col in condition_fields:
        expected_condition_fields[col] = expected_condition_fields[col].fillna("").astype(str).str.strip()
        observed_condition_fields[col] = observed_condition_fields[col].fillna("").astype(str).str.strip()
    joined_fields = observed_condition_fields.merge(
        expected_condition_fields.drop_duplicates("identity_key"),
        on="identity_key",
        how="left",
        validate="one_to_one",
        suffixes=("_out", "_src"),
    )
    for col in condition_fields:
        require(
            joined_fields[f"{col}_out"].eq(joined_fields[f"{col}_src"]).all(),
            f"serialized LINCS condition metadata does not match source projection: {col}",
        )

    pair_table = read_tsv(attempt / "ctrp_cell_drug_coverage.tsv")
    require(len(pair_table) == 16825 == summary["coverage_denominators"]["CTRP_unique_cell_drug_pairs_on_exactly_mapped_breast_lines"], "cell-drug coverage denominator mismatch")
    require(not pair_table[["CTRP_cellid", "CTRP_drugid"]].duplicated().any(), "duplicate cell-drug pair rows")
    require(pair_table["coverage_status"].value_counts().to_dict() == {
        "CTRP_response_record_without_usable_full_InChIKey": 9794,
        "no_exact_LINCS_condition_for_same_ModelID_and_full_InChIKey": 6802,
        "exact_RNA_CTRP_record_and_LINCS_condition": 229,
    }, "cell-drug exclusion denominators mismatch")
    require(int(pair_table["threeway_exact_condition_coverage"].eq("True").sum()) == 229, "exact pair coverage mismatch")
    require(pair_table.loc[pair_table["CTRP_full_InChIKey_exact"].ne(""), "CTRP_full_InChIKey_exact"].map(lambda x: bool(FULL_INCHIKEY.fullmatch(str(x)))).all(), "malformed key in cell-drug coverage output")

    cell_audit = read_tsv(attempt / "ctrp_cell_identity_audit.tsv")
    drug_audit = read_tsv(attempt / "ctrp_drug_identity_audit.tsv")
    lincs_audit = read_tsv(attempt / "lincs_breast_cell_identity_audit.tsv")
    require(len(cell_audit) == 887 and len(drug_audit) == 544 and len(lincs_audit) == 7, "audit denominators mismatch")
    require(summary["coverage_denominators"]["CTRP_unique_cellid_drugid_pairs_all"] == int(response_pairs.shape[0]), "all-source CTRP unique pair count mismatch")
    require(summary["coverage_denominators"]["CTRP_response_projection_rows_cellid_drugid_only"] == len(response_projection_raw), "CTRP response projection row count mismatch")
    require(summary["coverage_denominators"]["DepMap_expression_ModelID_axis_unique_ids"] == len(expression_ids), "expression identity axis count mismatch")
    require(summary["coverage_denominators"]["DepMap_expression_ModelID_axis_duplicate_rows"] == int(expression_ids_frame["ModelID"].dropna().duplicated().sum()), "expression identity duplicate count mismatch")
    require(summary["coverage_denominators"]["CTRP_cell_identity_rows"] == len(ctrp_cells), "CTRP cell identity row count mismatch")
    require(summary["coverage_denominators"]["CTRP_cells_exact_RRID_matched_to_DepMap_breast_with_expression_ModelID_row"] == 40, "exact breast RNA/CTRP denominator mismatch")
    require(summary["coverage_denominators"]["approved_LINCS_cells_with_condition_metadata"] == 5, "approved LINCS condition-cell denominator mismatch")
    require(summary["coverage_denominators"]["breast_LINCS_condition_metadata_rows_after_exact_identity"] == 30456, "LINCS condition metadata denominator mismatch")
    require(summary["coverage_denominators"]["LINCS_treatment_instances_metadata_total_for_unique_conditions"] == 7779, "LINCS treatment-instance metadata count mismatch")
    require(summary["coverage_denominators"]["LINCS_control_groups_metadata_total_for_unique_conditions"] == 7175, "LINCS control-group metadata count mismatch")

    # Verify the response-blind column projection fingerprints recorded by the run.
    projection_checks = {
        "ctrp_responses": projection_sha256(response_projection),
        "depmap_expression": projection_sha256(expression_ids_frame),
    }
    for name, actual in projection_checks.items():
        expected = manifest["source_metadata"][name]["projection_sha256" if name != "depmap_expression" else "modelid_axis_sha256"]
        require(actual == expected, f"projected metadata fingerprint mismatch: {name}")

    receipt = {
        "status": "PASS_ROOT_INDEPENDENT_THREEWAY_METADATA_READBACK",
        "experiment": "EXP-068",
        "run_id": manifest["run_id"],
        "attempt_dir": str(attempt.relative_to(root)),
        "read_scope": {
            "opened": [
                "CTRP response identity columns cellid/drugid only",
                "CTRP cell and drug identity metadata only",
                "DepMap model metadata and first expression ModelID column only",
                "approved LINCS cell crosswalk, official cell_info, pert_info and condition dose/time/count metadata only",
            ],
            "not_opened": [
                "CTRP response numeric labels", "DepMap expression values/header", "LINCS expression/delta/control arrays",
                "PDO/outer/locked/PANACEA/CDS-B values or labels",
            ],
            "ctpr_exposure_alignment": "CTRP projection has no dose/time, so no matched exposure claim is made",
        },
        "denominators": {
            "CTRP_response_identity_rows": int(len(response_projection_raw)),
            "CTRP_unique_cell_drug_pairs_all": int(len(response_pairs)),
            "DepMap_breast_models_by_OncotreeLineage": int(len(breast)),
            "DepMap_expression_ModelID_axis_unique": int(len(expression_ids)),
            "exact_breast_CTRP_RRID_plus_RNA_models": 40,
            "approved_LINCS_cells_with_conditions": 5,
            "exact_cell_drug_pairs_with_condition": 229,
            "unique_ModelID": int(key_table["DepMap_ModelID"].nunique()),
            "unique_PatientID": int(key_table["DepMap_PatientID"].nunique()),
            "unique_CTRP_drugid": int(key_table["CTRP_drugid"].nunique()),
            "unique_full_InChIKey": int(key_table["CTRP_full_InChIKey_exact"].nunique()),
            "unique_condition_id": int(key_table["LINCS_condition_id"].nunique()),
            "condition_key_rows": int(len(key_table)),
            "LINCS_treatment_instance_metadata_total": 7779,
            "LINCS_control_group_metadata_total": 7175,
            "cell_drug_pair_coverage_rows": int(len(pair_table)),
            "cell_pair_exclusions": pair_table["coverage_status"].value_counts().to_dict(),
        },
        "checks": {
            "all_manifest_output_hashes_match": True,
            "exact_condition_identity_join_recomputed": True,
            "full_InChIKey_format_and_equality": True,
            "dose_time_metadata_matches_condition_projection": True,
            "patient_model_and_drug_denominators_match": True,
            "pair_exclusion_denominators_match": True,
            "response_and_expression_numeric_values_opened": False,
        },
        "verified_output_files": output_hashes,
        "source_projection_fingerprints": projection_checks,
        "verifier_runtime": {
            "wall_seconds": round(time.perf_counter() - started_wall, 6),
            "cpu_process_seconds": round(time.process_time() - started_cpu, 6),
            "gpu_hours": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
