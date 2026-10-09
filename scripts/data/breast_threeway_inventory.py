#!/usr/bin/env python3
"""Inventory exact breast cell-line overlap across DepMap, CTRPv2, and LINCS.

This is an identity/metadata audit only. It reads DepMap expression row IDs,
CTRP cell/drug pair IDs, and LINCS condition metadata; it never reads expression,
drug-response, or perturbation values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("artifacts/data_audits/breast_threeway_inventory_20261009")
INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
TUMOR_LINCS_IDS = {"BT20", "HS578T", "MCF7", "MDAMB231", "SKBR3"}
EXPECTED_BREAST_MODELS = 97


def normalize_stable_id(value: object) -> str:
    """Normalize punctuation and case for exact stable-ID comparisons."""
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def normalize_inchikey(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def is_full_inchikey(value: object) -> bool:
    return bool(INCHIKEY_RE.fullmatch(normalize_inchikey(value)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent,
        suffix=".tmp", delete=False,
    ) as stream:
        temp = Path(stream.name)
        frame.to_csv(stream, index=False)
    os.replace(temp, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        suffix=".tmp", delete=False,
    ) as stream:
        temp = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temp, path)


def _read_expression_model_ids(path: Path) -> set[str]:
    """Read only the first CSV column (DepMap row identity), in chunks."""
    model_ids: set[str] = set()
    for chunk in pd.read_csv(path, usecols=[0], dtype="string", chunksize=50_000):
        model_ids.update(chunk.iloc[:, 0].dropna().astype(str))
    return model_ids


def _unique_stable_id_map(
    frame: pd.DataFrame, source_col: str, target_col: str,
) -> tuple[dict[str, str], set[str]]:
    work = frame[[source_col, target_col]].dropna().copy()
    work["_stable_key"] = work[source_col].map(normalize_stable_id)
    work = work.loc[work._stable_key.ne("")].drop_duplicates(["_stable_key", target_col])
    counts = work.groupby("_stable_key")[target_col].nunique()
    ambiguous = set(counts.loc[counts.gt(1)].index.astype(str))
    unique = work.loc[~work._stable_key.isin(ambiguous)]
    return unique.set_index("_stable_key")[target_col].astype(str).to_dict(), ambiguous


def _make_exact_structure_map(
    frame: pd.DataFrame, id_col: str, key_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return unambiguous ID->full-InChIKey mappings and row-level exclusions."""
    work = frame[[id_col, key_col]].copy()
    work[id_col] = work[id_col].astype("string")
    work["full_inchikey"] = work[key_col].map(normalize_inchikey)
    valid = work.loc[
        work[id_col].notna() & work.full_inchikey.map(is_full_inchikey),
        [id_col, "full_inchikey"],
    ].drop_duplicates()
    per_id = valid.groupby(id_col).full_inchikey.nunique()
    ambiguous_ids = set(per_id.loc[per_id.gt(1)].index.astype(str))
    good = valid.loc[~valid[id_col].astype(str).isin(ambiguous_ids)].drop_duplicates()
    excluded = work[[id_col, "full_inchikey"]].drop_duplicates().copy()
    excluded["identity_status"] = "verified_full_inchikey"
    excluded.loc[~excluded.full_inchikey.map(is_full_inchikey), "identity_status"] = "missing_or_invalid_full_inchikey"
    excluded.loc[excluded[id_col].astype(str).isin(ambiguous_ids), "identity_status"] = "ambiguous_id_to_multiple_inchikeys"
    return good, excluded


def _require_unique(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    if frame.duplicated(keys).any():
        samples = frame.loc[frame.duplicated(keys, keep=False), keys].head(5).to_dict("records")
        raise ValueError(f"{name} key is not unique on {keys}: {samples}")


def _read_inputs(root: Path) -> dict[str, Any]:
    depmap_dir = root / "data/raw/depmap/24q2"
    ctrp_dir = root / "data/processed/exp013/ctrp_v2"
    lincs_raw = root / "data/raw/lincs/GSE92742"
    lincs_conditions = root / "data/processed/lincs/GSE92742/continuous_conditions_v2"

    model_path = depmap_dir / "Model.csv"
    expression_path = depmap_dir / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    ctrp_cells_path = ctrp_dir / "cells.csv"
    ctrp_drugs_path = ctrp_dir / "drugs.csv"
    ctrp_pairs_path = ctrp_dir / "responses.csv"
    ctrp_map_path = root / "data/artifacts/resource_acquisition_20260916/ctrp_parse/ctrp_depmap_identity_map.csv"
    lincs_pert_path = lincs_raw / "GSE92742_Broad_LINCS_pert_info.txt.gz"
    lincs_conditions_path = lincs_conditions / "conditions.parquet"
    lincs_crosswalk_path = root / "artifacts/experiments/EXP-007/CCLE_CONTEXT_IDENTITY_SPLIT.json"

    required_paths = [
        model_path, expression_path, ctrp_cells_path, ctrp_drugs_path,
        ctrp_pairs_path, ctrp_map_path, lincs_pert_path, lincs_conditions_path,
        lincs_crosswalk_path,
    ]
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Required local source files are missing: {missing_paths}")

    models = pd.read_csv(
        model_path,
        usecols=["ModelID", "PatientID", "CellLineName", "StrippedCellLineName", "RRID",
                 "DepmapModelType", "OncotreeLineage", "OncotreePrimaryDisease"],
        dtype="string",
    )
    if models.ModelID.duplicated().any():
        raise ValueError("DepMap ModelID is not unique in Model.csv")
    breast = models.loc[models.OncotreeLineage.str.strip().str.casefold().eq("breast")].copy()
    if len(breast) != EXPECTED_BREAST_MODELS:
        raise ValueError(f"Expected {EXPECTED_BREAST_MODELS} DepMap 24Q2 breast-lineage records, found {len(breast)}")
    breast["baseline_RNA_present"] = breast.ModelID.isin(_read_expression_model_ids(expression_path))

    ctrp_cells = pd.read_csv(ctrp_cells_path, dtype="string")
    ctrp_map = pd.read_csv(ctrp_map_path, dtype="string")
    required_map_columns = {
        "cellid", "Cellosaurus.Accession.id", "ModelID", "match_method",
        "has_depmap_expression", "candidate_ModelID",
    }
    if required_map_columns.difference(ctrp_map.columns):
        raise ValueError(f"Existing verified CTRP-DepMap crosswalk lacks columns {sorted(required_map_columns.difference(ctrp_map.columns))}")
    # Recheck, rather than trust, each retained bridge against source stable IDs.
    model_rrid_map, ambiguous_model_rrids = _unique_stable_id_map(models, "RRID", "ModelID")
    ctrp_cells = ctrp_cells.merge(
        ctrp_map[["cellid", "ModelID", "match_method", "has_depmap_expression", "candidate_ModelID"]],
        on="cellid", how="left", validate="one_to_one",
    )
    ctrp_cells["_rrid_key"] = ctrp_cells["Cellosaurus.Accession.id"].map(normalize_stable_id)
    expected_model = ctrp_cells._rrid_key.map(model_rrid_map).astype("string")
    retained_exact = ctrp_cells.match_method.eq("exact_unique_cellosaurus_rrid")
    mismatch = retained_exact & ~ctrp_cells.ModelID.astype("string").eq(expected_model)
    if mismatch.any():
        raise ValueError("Saved CTRP-DepMap crosswalk disagrees with direct stable RRID mapping")
    ctrp_cells["depmap_model_id"] = ctrp_cells.ModelID.astype("string")
    ctrp_cells["ctrp_identity_status"] = "unmatched_rrid"
    ctrp_cells.loc[ctrp_cells._rrid_key.isin(ambiguous_model_rrids), "ctrp_identity_status"] = "ambiguous_depmap_rrid"
    ctrp_cells.loc[retained_exact & expected_model.notna(), "ctrp_identity_status"] = "exact_unique_cellosaurus_rrid"
    ctrp_cells = ctrp_cells.merge(
        breast[["ModelID", "PatientID", "CellLineName", "RRID", "baseline_RNA_present"]],
        left_on="depmap_model_id", right_on="ModelID", how="left", validate="many_to_one",
        suffixes=("", "_depmap"),
    )
    ctrp_breast_identity_cells = ctrp_cells.loc[
        ctrp_cells.depmap_model_id.isin(set(breast.ModelID.astype(str)))
        & ctrp_cells.ctrp_identity_status.eq("exact_unique_cellosaurus_rrid")
    ].copy()
    _require_unique(ctrp_breast_identity_cells, ["cellid"], "breast CTRP cell crosswalk")
    ctrp_breast_cells = ctrp_breast_identity_cells.loc[
        ctrp_breast_identity_cells.baseline_RNA_present.map(lambda value: False if pd.isna(value) else bool(value))
    ].copy()

    # The frozen LINCS-to-DepMap map is the approved exact stripped-name contract.
    identity_json = json.loads(lincs_crosswalk_path.read_text(encoding="utf-8"))
    if identity_json.get("status") != "IDENTITY_SPLIT_FROZEN":
        raise ValueError("LINCS-to-DepMap exact cell identity contract is not frozen")
    identity_rows = pd.DataFrame(identity_json.get("identity_rows", []))
    required_identity = {"lincs_cell_id", "depmap_id", "expression_present", "join_status"}
    if required_identity.difference(identity_rows.columns):
        raise ValueError(f"Frozen LINCS identity map lacks {sorted(required_identity.difference(identity_rows.columns))}")
    lincs_cell_map = identity_rows.loc[
        identity_rows.join_status.eq("EXACT_STRIPPED_NAME_WITH_RNASEQ")
        & identity_rows.expression_present.astype(bool)
        & identity_rows.depmap_id.astype(str).isin(set(breast.ModelID.astype(str))),
        ["lincs_cell_id", "depmap_id", "join_status"],
    ].drop_duplicates()
    _require_unique(lincs_cell_map, ["lincs_cell_id"], "LINCS cell identity map")
    _require_unique(lincs_cell_map, ["depmap_id"], "DepMap model to LINCS cell identity map")
    lincs_cell_map = lincs_cell_map.merge(
        breast[["ModelID", "PatientID", "CellLineName", "RRID", "baseline_RNA_present"]],
        left_on="depmap_id", right_on="ModelID", how="inner", validate="one_to_one",
    )
    if "patient_id" in lincs_cell_map.columns and not lincs_cell_map.patient_id.astype(str).eq(lincs_cell_map.PatientID.astype(str)).all():
        raise ValueError("Frozen LINCS identity patient key disagrees with DepMap PatientID")
    lincs_cell_map = lincs_cell_map.loc[
        lincs_cell_map.lincs_cell_id.astype(str).str.upper().isin(TUMOR_LINCS_IDS)
    ].copy()

    ctrp_drugs = pd.read_csv(ctrp_drugs_path, usecols=["drugid", "cpd_name", "cpd_smiles", "inchikey", "cid"], dtype="string")
    ctrp_drugs["ctrp_inchikey_source"] = ctrp_drugs.inchikey
    ctrp_drug_map, ctrp_drug_identity_rows = _make_exact_structure_map(ctrp_drugs, "drugid", "inchikey")
    ctrp_drugs = ctrp_drugs.merge(ctrp_drug_map, on="drugid", how="inner", validate="one_to_one")
    ctrp_key_counts = ctrp_drugs.groupby("full_inchikey").drugid.nunique()
    ctrp_drugs["ctrp_key_has_multiple_ids"] = ctrp_drugs.full_inchikey.map(ctrp_key_counts).gt(1)

    lincs_pert_cols = ["pert_id", "pert_type", "pert_iname", "inchi_key"]
    lincs_pert = pd.read_csv(
        lincs_pert_path, sep="\t", compression="gzip", usecols=lincs_pert_cols,
        dtype="string",
    )
    lincs_pert["full_inchikey"] = lincs_pert.inchi_key.map(normalize_inchikey)
    lincs_cp = lincs_pert.loc[lincs_pert.pert_type.eq("trt_cp")].copy()
    lincs_drug_mapping_frame, lincs_drug_identity_rows = _make_exact_structure_map(lincs_cp, "pert_id", "inchi_key")
    lincs_drug_map = lincs_drug_mapping_frame.set_index("pert_id").full_inchikey.astype(str).to_dict()
    lincs_drug_names = lincs_cp[["pert_id", "pert_iname"]].drop_duplicates().groupby(
        "pert_id", as_index=False
    ).agg(pert_iname=("pert_iname", lambda x: " | ".join(sorted(set(x.dropna().astype(str))))))
    lincs_drugs = lincs_drug_mapping_frame.copy()
    lincs_drugs = lincs_drugs.merge(lincs_drug_names, on="pert_id", how="left", validate="one_to_one")
    lincs_pert_type = lincs_pert.groupby("pert_id", as_index=False).agg(
        pert_type=("pert_type", lambda x: "trt_cp" if set(x.dropna().astype(str)) == {"trt_cp"} else "ambiguous_or_noncompound"),
        pert_iname=("pert_iname", lambda x: " | ".join(sorted(set(x.dropna().astype(str))))),
        lincs_inchikey_source=("inchi_key", lambda x: " | ".join(sorted(set(x.dropna().astype(str))))),
    )
    lincs_pert_for_conditions = lincs_pert_type.merge(
        lincs_drugs[["pert_id", "full_inchikey"]], on="pert_id", how="left", validate="one_to_one"
    )

    # Read only the condition metadata columns. The perturbation matrix is not opened.
    condition_cols = [
        "condition_id", "cell_id", "pert_id", "pert_dose", "pert_dose_unit",
        "pert_time", "pert_time_unit", "n_treatment_instances", "n_control_groups",
    ]
    conditions = pd.read_parquet(lincs_conditions_path, columns=condition_cols)
    _require_unique(conditions, ["condition_id"], "LINCS condition metadata")

    # CTRP response table is projected to identities only; response values are never read.
    ctrp_pairs = pd.read_csv(
        ctrp_pairs_path, usecols=["cellid", "drugid"],
        dtype={"cellid": "string", "drugid": "string"},
    ).dropna(subset=["cellid", "drugid"]).drop_duplicates()
    ctrp_pairs = ctrp_pairs.merge(
        ctrp_breast_identity_cells[["cellid", "depmap_model_id", "PatientID", "_rrid_key", "ccl_name",
                           "Cellosaurus.Accession.id", "CellLineName", "baseline_RNA_present"]],
        on="cellid", how="inner", validate="many_to_one",
    )
    ctrp_pairs = ctrp_pairs.merge(
        ctrp_drugs[["drugid", "full_inchikey", "ctrp_inchikey_source", "cpd_name", "cpd_smiles", "cid", "ctrp_key_has_multiple_ids"]],
        on="drugid", how="left", validate="many_to_one",
    )
    ctrp_pairs["pair_key_valid"] = ctrp_pairs.full_inchikey.map(is_full_inchikey).fillna(False).astype(bool)
    _require_unique(ctrp_pairs, ["cellid", "drugid"], "CTRP breast cell-drug metadata pairs")

    # Full metadata condition table for the five exact tumor-cell LINCS backgrounds.
    lincs_cell_ids = set(lincs_cell_map.lincs_cell_id.astype(str))
    tumor_conditions = conditions.loc[conditions.cell_id.astype(str).isin(lincs_cell_ids)].copy()
    tumor_conditions = tumor_conditions.merge(
        lincs_cell_map[["lincs_cell_id", "depmap_id", "PatientID", "CellLineName", "RRID", "baseline_RNA_present"]],
        left_on="cell_id", right_on="lincs_cell_id", how="inner", validate="many_to_one",
    )
    tumor_conditions = tumor_conditions.merge(
        lincs_pert_for_conditions,
        on="pert_id", how="left", validate="many_to_one",
    )
    tumor_conditions["lincs_compound_identity_valid"] = (
        tumor_conditions.pert_type.eq("trt_cp")
        & tumor_conditions.full_inchikey.map(is_full_inchikey).fillna(False)
        & tumor_conditions.pert_id.astype(str).isin(lincs_drug_map)
    )
    valid_ctrp_keys = set(ctrp_drugs.full_inchikey.dropna().astype(str))
    tumor_conditions["exact_CTRP_structure_available"] = tumor_conditions.full_inchikey.astype(str).isin(valid_ctrp_keys)

    # Join by stable DepMap ModelID through the CTRP cell map.
    pair_lookup = ctrp_pairs.loc[
        ctrp_pairs.pair_key_valid,
        ["cellid", "drugid", "depmap_model_id", "_rrid_key", "ccl_name",
         "Cellosaurus.Accession.id", "full_inchikey", "ctrp_inchikey_source", "cpd_name", "cpd_smiles", "cid"],
    ].drop_duplicates()
    cond_struct = tumor_conditions.loc[
        tumor_conditions.lincs_compound_identity_valid
        & tumor_conditions.exact_CTRP_structure_available
    ].copy()
    cond_struct = cond_struct.merge(
        pair_lookup,
        left_on=["depmap_id", "full_inchikey"],
        right_on=["depmap_model_id", "full_inchikey"],
        how="left",
        validate="many_to_many",
    )
    threeway = cond_struct.loc[cond_struct.cellid.notna()].copy()
    threeway = threeway.drop(columns=["depmap_model_id"])
    threeway = threeway.rename(columns={
        "depmap_id": "depmap_model_id",
        "PatientID": "depmap_patient_id",
        "_rrid_key": "depmap_rrid_normalized",
        "ccl_name": "ctrp_cell_name",
        "Cellosaurus.Accession.id": "ctrp_cellosaurus_accession",
        "pert_iname": "lincs_pert_iname",
    })
    _require_unique(
        threeway,
        ["condition_id", "depmap_model_id", "cellid", "drugid", "pert_id", "full_inchikey"],
        "three-way condition identity key",
    )
    threeway["identity_key"] = (
        threeway.depmap_model_id.astype(str) + "|" + threeway.cellid.astype(str)
        + "|" + threeway.drugid.astype(str) + "|" + threeway.full_inchikey.astype(str)
        + "|" + threeway.lincs_cell_id.astype(str) + "|" + threeway.pert_id.astype(str)
        + "|" + threeway.condition_id.astype(str)
    )

    # Give every breast CTRP cell-drug pair one exclusive reason for inclusion/exclusion.
    any_lincs_key = set(lincs_drugs.full_inchikey.astype(str))
    any_cell_key = set(
        tumor_conditions.loc[tumor_conditions.lincs_compound_identity_valid, ["depmap_id", "full_inchikey"]]
        .astype(str).itertuples(index=False, name=None)
    )
    ctrp_pairs["lincs_cell_mapped"] = ctrp_pairs.depmap_model_id.astype(str).isin(set(lincs_cell_map.depmap_id.astype(str)))
    ctrp_pairs["lincs_exact_structure_exists"] = ctrp_pairs.full_inchikey.astype(str).isin(any_lincs_key)
    ctrp_pairs["same_cell_lincs_condition_exists"] = [
        (str(model_id), str(key)) in any_cell_key
        for model_id, key in zip(ctrp_pairs.depmap_model_id, ctrp_pairs.full_inchikey)
    ]
    joined_pair_keys = set(
        threeway[["cellid", "drugid"]].drop_duplicates().astype(str).itertuples(index=False, name=None)
    )
    ctrp_pairs["threeway_pair_present"] = [
        (str(cell_id), str(drug_id)) in joined_pair_keys
        for cell_id, drug_id in zip(ctrp_pairs.cellid, ctrp_pairs.drugid)
    ]
    ctrp_pairs["exclusion_reason"] = "matched_threeway_condition"
    has_baseline_rna = ctrp_pairs.baseline_RNA_present.map(lambda value: False if pd.isna(value) else bool(value))
    ctrp_pairs.loc[~has_baseline_rna, "exclusion_reason"] = "no_DepMap_baseline_RNA"
    ctrp_pairs.loc[has_baseline_rna & ~ctrp_pairs.lincs_cell_mapped, "exclusion_reason"] = "no_exact_LINCS_tumor_cell_mapping"
    ctrp_pairs.loc[ctrp_pairs.lincs_cell_mapped & ~ctrp_pairs.pair_key_valid, "exclusion_reason"] = "no_unambiguous_full_CTRP_InChIKey"
    ctrp_pairs.loc[ctrp_pairs.pair_key_valid & ~ctrp_pairs.lincs_exact_structure_exists, "exclusion_reason"] = "no_exact_LINCS_full_InChIKey"
    ctrp_pairs.loc[
        ctrp_pairs.pair_key_valid & ctrp_pairs.lincs_exact_structure_exists
        & ~ctrp_pairs.same_cell_lincs_condition_exists,
        "exclusion_reason",
    ] = "no_condition_for_exact_cell_and_drug"
    ctrp_pairs.loc[ctrp_pairs.threeway_pair_present, "exclusion_reason"] = "matched_threeway_condition"

    # Exclusive reason for every LINCS condition from the five tumor backgrounds.
    has_same_cell_ctrp = set(pair_lookup[["depmap_model_id", "full_inchikey"]].astype(str).itertuples(index=False, name=None))
    tumor_conditions["condition_exclusion_reason"] = "matched_threeway_condition"
    non_compound = ~tumor_conditions.pert_type.eq("trt_cp")
    tumor_conditions.loc[non_compound, "condition_exclusion_reason"] = "not_small_molecule_trt_cp"
    invalid_lincs = ~tumor_conditions.lincs_compound_identity_valid & ~non_compound
    tumor_conditions.loc[invalid_lincs, "condition_exclusion_reason"] = "missing_invalid_or_ambiguous_LINCS_InChIKey"
    no_ctrp_key = (
        tumor_conditions.lincs_compound_identity_valid
        & ~tumor_conditions.exact_CTRP_structure_available
    )
    tumor_conditions.loc[no_ctrp_key, "condition_exclusion_reason"] = "no_exact_CTRP_full_InChIKey"
    no_same_cell_pair = [
        (str(model_id), str(key)) not in has_same_cell_ctrp
        for model_id, key in zip(tumor_conditions.depmap_id, tumor_conditions.full_inchikey)
    ]
    no_same_cell_pair = pd.Series(no_same_cell_pair, index=tumor_conditions.index)
    mask = (
        tumor_conditions.lincs_compound_identity_valid
        & tumor_conditions.exact_CTRP_structure_available
        & no_same_cell_pair
    )
    tumor_conditions.loc[mask, "condition_exclusion_reason"] = "no_same_cell_CTRP_cell_drug_pair"

    # Row-level condition ledger preserves an explicit reason for every
    # processed LINCS condition from the five tumor backgrounds.
    ctrp_ids_by_model_key = pair_lookup.groupby(
        ["depmap_model_id", "full_inchikey"], as_index=False
    ).agg(ctrp_drug_ids_same_cell_key=("drugid", lambda x: "|".join(sorted(set(x.dropna().astype(str))))))
    ctrp_cells_by_model = ctrp_breast_identity_cells.groupby("depmap_model_id", as_index=False).agg(
        ctrp_cellids=("cellid", lambda x: "|".join(sorted(set(x.dropna().astype(str)))))
    )
    condition_ledger = tumor_conditions.merge(
        ctrp_cells_by_model, left_on="depmap_id", right_on="depmap_model_id",
        how="left", validate="many_to_one",
    ).drop(columns=["depmap_model_id"])
    condition_ledger = condition_ledger.merge(
        ctrp_ids_by_model_key,
        left_on=["depmap_id", "full_inchikey"],
        right_on=["depmap_model_id", "full_inchikey"],
        how="left", validate="many_to_one",
    ).drop(columns=["depmap_model_id"])
    condition_ledger = condition_ledger.rename(columns={
        "depmap_id": "depmap_model_id",
        "PatientID": "depmap_patient_id",
    })
    condition_ledger["condition_identity_key"] = (
        condition_ledger.depmap_model_id.astype(str) + "|"
        + condition_ledger.lincs_cell_id.astype(str) + "|"
        + condition_ledger.pert_id.astype(str) + "|"
        + condition_ledger.condition_id.astype(str)
    )
    condition_ledger_cols = [
        "condition_identity_key", "condition_id", "depmap_model_id", "depmap_patient_id",
        "CellLineName", "RRID", "baseline_RNA_present", "lincs_cell_id", "cell_id",
        "ctrp_cellids", "pert_id", "pert_type", "pert_iname", "lincs_inchikey_source",
        "full_inchikey", "lincs_compound_identity_valid", "exact_CTRP_structure_available",
        "ctrp_drug_ids_same_cell_key", "pert_dose", "pert_dose_unit", "pert_time",
        "pert_time_unit", "n_treatment_instances", "n_control_groups",
        "condition_exclusion_reason",
    ]
    condition_ledger = condition_ledger[condition_ledger_cols]

    # Pair-level and condition-level output ledgers.
    pair_cols = [
        "depmap_model_id", "PatientID", "CellLineName", "RRID", "cellid", "ccl_name",
        "Cellosaurus.Accession.id", "drugid", "cpd_name", "cpd_smiles", "cid",
        "ctrp_inchikey_source", "full_inchikey", "ctrp_key_has_multiple_ids", "baseline_RNA_present",
        "lincs_cell_mapped", "lincs_exact_structure_exists",
        "same_cell_lincs_condition_exists", "threeway_pair_present", "exclusion_reason",
    ]
    ctrp_pair_ledger = ctrp_pairs[[col for col in pair_cols if col in ctrp_pairs.columns]].copy()

    condition_cols_out = [
        "identity_key", "condition_id", "depmap_model_id", "depmap_patient_id", "CellLineName", "RRID",
        "baseline_RNA_present", "lincs_cell_id", "cell_id", "cellid", "ctrp_cell_name",
        "ctrp_cellosaurus_accession", "drugid", "cpd_name", "cpd_smiles", "cid",
        "ctrp_inchikey_source", "pert_id", "lincs_pert_iname", "lincs_inchikey_source",
        "pert_type", "full_inchikey", "pert_dose", "pert_dose_unit", "pert_time",
        "pert_time_unit", "n_treatment_instances", "n_control_groups",
    ]
    threeway_out = threeway[[col for col in condition_cols_out if col in threeway.columns]].copy()

    condition_reason_counts = (
        tumor_conditions.condition_exclusion_reason.value_counts(dropna=False)
        .rename_axis("reason").reset_index(name="condition_count")
    )
    pair_reason_counts = (
        ctrp_pair_ledger.exclusion_reason.value_counts(dropna=False)
        .rename_axis("reason").reset_index(name="cell_drug_pair_count")
    )
    if not threeway.empty:
        strata = threeway.groupby(
            ["pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit", "n_control_groups"],
            dropna=False,
        ).agg(
            condition_count=("condition_id", "nunique"),
            cell_count=("depmap_model_id", "nunique"),
            drug_key_count=("full_inchikey", "nunique"),
            ctrp_drug_id_count=("drugid", "nunique"),
            pert_id_count=("pert_id", "nunique"),
            treatment_instance_sum=("n_treatment_instances", "sum"),
        ).reset_index()
    else:
        strata = pd.DataFrame(columns=["pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit", "n_control_groups", "condition_count", "cell_count", "drug_key_count", "ctrp_drug_id_count", "pert_id_count", "treatment_instance_sum"])

    # Stable identity crosswalk shared with the parallel subtype work package.
    lincs_identity_for_model = lincs_cell_map.set_index("depmap_id")["lincs_cell_id"].to_dict()
    ctrp_cellids_for_model = ctrp_breast_identity_cells.groupby("depmap_model_id").cellid.apply(
        lambda values: "|".join(sorted(set(values.astype(str))))
    ).to_dict()
    ctrp_accessions_for_model = ctrp_breast_identity_cells.groupby("depmap_model_id")["Cellosaurus.Accession.id"].apply(
        lambda values: "|".join(sorted(set(values.dropna().astype(str))))
    ).to_dict()
    ctrp_names_for_model = ctrp_breast_identity_cells.groupby("depmap_model_id").ccl_name.apply(
        lambda values: "|".join(sorted(set(values.dropna().astype(str))))
    ).to_dict()
    lincs_status_for_model = {
        row.depmap_id: row.join_status for row in lincs_cell_map.itertuples(index=False)
    }
    crosswalk = breast[["ModelID", "PatientID", "CellLineName", "DepmapModelType", "OncotreeLineage", "OncotreePrimaryDisease", "RRID", "baseline_RNA_present"]].copy()
    crosswalk = crosswalk.rename(columns={
        "ModelID": "depmap_model_id", "DepmapModelType": "depmap_model_type",
        "OncotreeLineage": "depmap_oncotree_lineage",
        "OncotreePrimaryDisease": "depmap_primary_disease",
        "PatientID": "depmap_patient_id",
    })
    crosswalk["lincs_cell_id"] = crosswalk.depmap_model_id.map(lincs_identity_for_model).fillna("")
    crosswalk["lincs_identity_status"] = crosswalk.depmap_model_id.map(lincs_status_for_model).fillna("no_exact_LINCS_RNA_cell_mapping")
    crosswalk["ctrp_cellid"] = crosswalk.depmap_model_id.map(ctrp_cellids_for_model).fillna("")
    crosswalk["ctrp_cell_name"] = crosswalk.depmap_model_id.map(ctrp_names_for_model).fillna("")
    crosswalk["ctrp_cellosaurus_accession"] = crosswalk.depmap_model_id.map(ctrp_accessions_for_model).fillna("")
    crosswalk["ctrp_identity_status"] = crosswalk.ctrp_cellid.map(lambda value: "exact_unique_cellosaurus_rrid" if value else "no_exact_CTRP_cell_mapping")
    crosswalk["threeway_CTRP_pair_count"] = crosswalk.depmap_model_id.map(
        ctrp_pair_ledger.loc[ctrp_pair_ledger.exclusion_reason.eq("matched_threeway_condition")]
        .groupby("depmap_model_id").drugid.nunique()
    ).fillna(0).astype(int)
    crosswalk["threeway_condition_count"] = crosswalk.depmap_model_id.map(
        threeway.groupby("depmap_model_id").condition_id.nunique()
    ).fillna(0).astype(int)
    crosswalk["threeway_unique_full_inchikeys"] = crosswalk.depmap_model_id.map(
        threeway.groupby("depmap_model_id").full_inchikey.nunique()
    ).fillna(0).astype(int)
    crosswalk = crosswalk.rename(columns={"depmap_model_id": "ModelID"})

    if not threeway.empty:
        dose_time_control_columns = [
            "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit", "n_control_groups"
        ]
        threeway["_dose_time_control_key"] = (
            threeway[dose_time_control_columns].astype("string").fillna("<NA>").agg("|".join, axis=1)
        )
        drug_summary = threeway.groupby("full_inchikey", as_index=False).agg(
            depmap_cell_count=("depmap_model_id", "nunique"),
            ctrp_drug_id_count=("drugid", "nunique"),
            lincs_pert_id_count=("pert_id", "nunique"),
            condition_count=("condition_id", "nunique"),
            dose_time_control_strata_count=("_dose_time_control_key", "nunique"),
            ctrp_names=("cpd_name", lambda x: " | ".join(sorted(set(x.dropna().astype(str))))),
            lincs_names=("lincs_pert_iname", lambda x: " | ".join(sorted(set(x.dropna().astype(str))))),
        )
    else:
        drug_summary = pd.DataFrame(columns=[
            "full_inchikey", "depmap_cell_count", "ctrp_drug_id_count", "lincs_pert_id_count",
            "condition_count", "dose_time_control_strata_count", "ctrp_names", "lincs_names",
        ])

    # Source/contract hashes support exact replay and make source revisions visible.
    input_paths = {
        "depmap_model": model_path,
        "depmap_expression_row_identity_only": expression_path,
        "ctrp_cells": ctrp_cells_path,
        "ctrp_drugs": ctrp_drugs_path,
        "ctrp_response_pair_identities_only": ctrp_pairs_path,
        "verified_ctrp_depmap_identity_crosswalk": ctrp_map_path,
        "lincs_pert_info": lincs_pert_path,
        "lincs_condition_metadata_only": lincs_conditions_path,
        "frozen_lincs_depmap_cell_identity_contract": lincs_crosswalk_path,
    }
    provenance = {
        name: {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for name, path in input_paths.items()
    }

    summary = {
        "audit_id": "breast_threeway_inventory_20261009",
        "scope": "metadata_only_identity_inventory; no DepMap expression values, CTRP response values, or LINCS perturbation matrix values read",
        "source_roles": {
            "DepMap_24Q2": "pretreatment RNA row identity availability plus breast cell metadata",
            "CTRPv2_local_processed": "cell-drug pair identities only; no response-value column projected",
            "LINCS_GSE92742": "small-molecule perturbation identity and condition/dose/time/control metadata only",
        },
        "identity_contracts": {
            "breast_model_domain": "DepMap 24Q2 OncotreeLineage exactly Breast",
            "CTRP_to_DepMap": "reuse existing verified crosswalk only where match_method=exact_unique_cellosaurus_rrid; independently recheck normalized Cellosaurus.Accession.id == DepMap RRID; no name-only candidate retained",
            "LINCS_to_DepMap": "reuse frozen EXP-007 exact stripped-name identity contract with join_status=EXACT_STRIPPED_NAME_WITH_RNASEQ; split roles are not applied or changed",
            "drug_identity": "exact full 27-character InChIKey equality; partial/invalid keys and ambiguous source IDs excluded; drug names are not identity keys",
            "threeway_key": "DepMap ModelID + CTRP cellid + CTRP drugid + full InChIKey + LINCS cell_id + pert_id + condition_id",
            "exposure_filter": "none; all observed metadata dose/time strata retained and reported",
        },
        "counts": {
            "depmap_breast_model_records": int(len(breast)),
            "depmap_breast_models_with_pretreatment_rna_row": int(breast.baseline_RNA_present.sum()),
        "ctrp_breast_exact_stable_id_cells": int(ctrp_breast_identity_cells.cellid.nunique()),
        "ctrp_breast_exact_stable_id_cells_with_rna": int(ctrp_breast_cells.cellid.nunique()),
            "ctrp_breast_cell_drug_pairs_identity_rows": int(len(ctrp_pair_ledger)),
            "ctrp_breast_unique_drug_ids_in_pairs": int(ctrp_pair_ledger.drugid.nunique()),
            "lincs_exact_tumor_cell_models": int(lincs_cell_map.depmap_id.nunique()),
            "lincs_exact_tumor_cell_ids": sorted(lincs_cell_map.lincs_cell_id.astype(str).unique().tolist()),
            "lincs_tumor_condition_metadata_rows": int(len(tumor_conditions)),
            "lincs_tumor_unique_pert_ids": int(tumor_conditions.pert_id.nunique()),
            "lincs_tumor_unique_full_inchikeys": int(tumor_conditions.loc[tumor_conditions.lincs_compound_identity_valid, "full_inchikey"].nunique()),
            "threeway_cell_models": int(threeway.depmap_model_id.nunique()),
            "threeway_independent_depmap_patient_ids": int(threeway.depmap_patient_id.nunique()),
            "threeway_ctrp_cell_ids": int(threeway.cellid.nunique()),
            "threeway_lincs_cell_ids": int(threeway.lincs_cell_id.nunique()),
            "threeway_unique_full_inchikeys": int(threeway.full_inchikey.nunique()),
            "threeway_unique_ctrp_drug_ids": int(threeway.drugid.nunique()),
            "threeway_unique_cell_drug_pairs": int(threeway[["cellid", "drugid"]].drop_duplicates().shape[0]),
            "threeway_unique_cell_structure_pairs": int(threeway[["cellid", "full_inchikey"]].drop_duplicates().shape[0]),
            "threeway_unique_lincs_pert_ids": int(threeway.pert_id.nunique()),
            "threeway_unique_conditions": int(threeway.condition_id.nunique()),
            "threeway_condition_cell_drug_rows": int(len(threeway)),
            "threeway_dose_time_control_strata": int(len(strata)),
            "threeway_drugs_by_cell": {
                str(row.CellLineName): int(row.threeway_unique_full_inchikeys)
                for row in crosswalk.loc[crosswalk.threeway_unique_full_inchikeys.gt(0), ["CellLineName", "threeway_unique_full_inchikeys"]].itertuples(index=False)
            },
        },
        "exclusion_counts": {
            "ctrp_cell_drug_pairs_by_reason": {str(row.reason): int(row.cell_drug_pair_count) for row in pair_reason_counts.itertuples(index=False)},
            "lincs_tumor_conditions_by_reason": {str(row.reason): int(row.condition_count) for row in condition_reason_counts.itertuples(index=False)},
            "breast_models_without_exact_LINCS_RNA_mapping": int((crosswalk.lincs_cell_id.eq("")).sum()),
        },
        "condition_metadata_policy": {
            "dose_fields": ["pert_dose", "pert_dose_unit"],
            "time_fields": ["pert_time", "pert_time_unit"],
            "control_field": "n_control_groups",
            "treatment_instance_field": "n_treatment_instances",
            "no_condition_is_inferred_from_other_doses_or_times": True,
        },
        "files_sha256": provenance,
        "limitations": [
            "A condition record documents an exposure and metadata; it does not establish a usable finite response label or model performance.",
            "CTRP identity rows reflect presence in the source response table; finite AAC or assay-quality values were not read or checked.",
            "LINCS condition rows reflect processed condition metadata; expression/delta values and control expression were not read.",
            "Dose/time/control strata differ across platforms; the inventory does not claim assay equivalence or causal transfer.",
            "CTRP drug IDs sharing an identical complete InChIKey are reported separately as source records while unique chemical keys are counted once.",
        ],
    }

    return {
        "crosswalk": crosswalk,
        "ctrp_pairs": ctrp_pair_ledger,
        "threeway_conditions": threeway_out,
        "pair_exclusion_counts": pair_reason_counts,
        "condition_exclusion_counts": condition_reason_counts,
        "condition_strata": strata,
        "condition_ledger": condition_ledger,
        "drug_summary": drug_summary,
        "summary": summary,
        "lincs_cell_map": lincs_cell_map,
    }


def run(root: Path, output: Path) -> dict[str, Any]:
    result = _read_inputs(root)
    script_path = Path(__file__).resolve()
    result["summary"]["implementation"] = {
        "path": script_path.relative_to(root).as_posix(),
        "sha256": _sha256(script_path),
    }
    result["summary"]["runtime"] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
    }
    output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "breast_cell_identity_crosswalk.csv": result["crosswalk"],
        "ctrp_breast_pair_identity_inventory.csv": result["ctrp_pairs"],
        "threeway_condition_inventory.csv": result["threeway_conditions"],
        "lincs_tumor_condition_inventory.csv": result["condition_ledger"],
        "threeway_drug_summary.csv": result["drug_summary"],
        "ctrp_pair_exclusion_counts.csv": result["pair_exclusion_counts"],
        "lincs_condition_exclusion_counts.csv": result["condition_exclusion_counts"],
        "threeway_dose_time_control_strata.csv": result["condition_strata"],
    }
    for name, frame in outputs.items():
        _atomic_csv(frame, output / name)
    # Independent serialized readback: verify the shared denominator, full keys,
    # exclusion accounting, and the complete condition-level identity key.
    crosswalk_back = pd.read_csv(output / "breast_cell_identity_crosswalk.csv", dtype="string")
    pair_back = pd.read_csv(output / "ctrp_breast_pair_identity_inventory.csv", dtype="string")
    condition_back = pd.read_csv(output / "threeway_condition_inventory.csv", dtype="string")
    condition_ledger_back = pd.read_csv(output / "lincs_tumor_condition_inventory.csv", dtype="string")
    pair_exclusion_back = pd.read_csv(output / "ctrp_pair_exclusion_counts.csv")
    condition_exclusion_back = pd.read_csv(output / "lincs_condition_exclusion_counts.csv")
    required_shared_columns = {
        "ModelID", "depmap_patient_id", "CellLineName", "RRID", "ctrp_cellid", "ctrp_cell_name", "ctrp_cellosaurus_accession",
        "ctrp_identity_status", "baseline_RNA_present", "lincs_cell_id", "lincs_identity_status",
    }
    if required_shared_columns.difference(crosswalk_back.columns):
        raise ValueError(f"Serialized shared identity crosswalk lacks {sorted(required_shared_columns.difference(crosswalk_back.columns))}")
    if crosswalk_back.ModelID.nunique() != result["summary"]["counts"]["depmap_breast_model_records"]:
        raise ValueError("Serialized shared identity crosswalk changed the DepMap breast denominator")
    if pair_back.duplicated(["cellid", "drugid"]).any():
        raise ValueError("Serialized CTRP breast pair identities are not unique")
    if int(pair_exclusion_back.cell_drug_pair_count.sum()) != len(pair_back):
        raise ValueError("CTRP cell-drug exclusion reasons do not account for every source identity pair")
    matched_pair_count = int(pair_exclusion_back.loc[pair_exclusion_back.reason.eq("matched_threeway_condition"), "cell_drug_pair_count"].sum())
    if matched_pair_count != result["summary"]["counts"]["threeway_unique_cell_drug_pairs"]:
        raise ValueError("CTRP matched-pair count disagrees with the condition-level three-way intersection")
    if int(condition_exclusion_back.condition_count.sum()) != result["summary"]["counts"]["lincs_tumor_condition_metadata_rows"]:
        raise ValueError("LINCS condition exclusion reasons do not account for every tumor-cell condition")
    if condition_ledger_back.condition_id.nunique() != result["summary"]["counts"]["lincs_tumor_condition_metadata_rows"]:
        raise ValueError("Serialized LINCS condition ledger lost or duplicated condition identities")
    ledger_reason_counts = condition_ledger_back.condition_exclusion_reason.value_counts().to_dict()
    recorded_reason_counts = dict(zip(condition_exclusion_back.reason, condition_exclusion_back.condition_count.astype(int)))
    if ledger_reason_counts != recorded_reason_counts:
        raise ValueError("Serialized LINCS condition reasons disagree with the summary table")
    matched_ledger_ids = set(condition_ledger_back.loc[condition_ledger_back.condition_exclusion_reason.eq("matched_threeway_condition"), "condition_id"])
    if matched_ledger_ids != set(condition_back.condition_id):
        raise ValueError("Matched LINCS conditions disagree between exclusion and three-way ledgers")
    if condition_back.empty or not condition_back.full_inchikey.map(is_full_inchikey).all():
        raise ValueError("Three-way condition inventory contains an empty result or incomplete structural key")
    for source_column in ["ctrp_inchikey_source", "lincs_inchikey_source"]:
        if source_column not in condition_back.columns:
            raise ValueError(f"Three-way condition inventory lacks source key column {source_column}")
        for source_value, full_key in zip(condition_back[source_column], condition_back.full_inchikey):
            source_keys = {normalize_inchikey(part) for part in str(source_value).split("|") if part.strip()}
            if source_keys != {normalize_inchikey(full_key)}:
                raise ValueError(f"Source structural key mismatch in serialized {source_column}")
    if condition_back.duplicated(["identity_key"]).any():
        raise ValueError("Serialized three-way condition identity keys are not unique")
    if not set(condition_back.lincs_cell_id.astype(str)).issubset(TUMOR_LINCS_IDS):
        raise ValueError("Three-way intersection contains an out-of-domain LINCS cell background")
    result["summary"]["validation"] = {
        "status": "PASS",
        "checks": {
            "97_model_shared_identity_crosswalk_keyed_by_ModelID": True,
            "CTRP_cell_drug_identity_pairs_unique": True,
            "CTRP_pair_exclusion_counts_sum_to_pair_denominator": True,
            "LINCS_condition_exclusion_counts_sum_to_tumor_condition_denominator": True,
            "every_tumor_condition_has_a_row_level_reason": True,
            "matched_condition_ids_agree_between_ledgers": True,
            "threeway_rows_have_full_InChIKey": True,
            "CTRP_and_LINCS_source_InChIKeys_exactly_equal_full_key": True,
            "threeway_condition_identity_key_unique": True,
            "threeway_cell_domain_is_five_tumor_backgrounds": True,
        },
    }
    result["summary"]["output_files_sha256"] = {
        name: _sha256(output / name) for name in outputs
    }
    result["summary"]["output_rows"] = {name: int(len(frame)) for name, frame in outputs.items()}
    _atomic_json(result["summary"], output / "summary.json")
    return result["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = run(args.root.resolve(), args.output.resolve())
    print(json.dumps({"summary_path": str(args.output.resolve() / "summary.json"), "counts": summary["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
