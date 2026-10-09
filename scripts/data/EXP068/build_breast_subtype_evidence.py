#!/usr/bin/env python
"""Build an identity-keyed, metadata-only breast cell-line subtype evidence pack.

This script reads only DepMap Model.csv model metadata, CTRPv2 cell identity
metadata, and the already-approved LINCS/DepMap crosswalk. It does not open
CTRP response tables, LINCS expression, DepMap expression, or any PDO/outer/
locked/PANACEA/CDS-DB numeric asset. Published receptor annotations below are
coded facts from the cited tables and are not used to select the cell lines.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEP_MAP = ROOT / "data/raw/depmap/24q2/Model.csv"
CTRP_CELLS = ROOT / "data/processed/exp013/ctrp_v2/cells.csv"
CTRP_RDS = ROOT / "data/raw/exp013/ctrp_v2/CTRPv2.rds"
LINCS_CROSSWALK = (
    ROOT
    / "artifacts/experiments/EXP-050/20261006_breast_subtype_measured_transfer_v1/"
    / "lincs_breast_cell_model_identity.tsv"
)
LINCS_ROOT_RECEIPT = (
    ROOT
    / "artifacts/experiments/EXP-050/20261006_breast_subtype_measured_transfer_v1/"
    / "root_independent_readback_receipt.json"
)
LINCS_CELL_INFO = ROOT / "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_cell_info.txt.gz"
CTRP_MATERIALIZATION = (
    ROOT / "data/artifacts/resource_acquisition_20260916/ctrp_parse/"
    / "ctrp_v2_materialization_registry.json"
)
SOURCE_AUDIT_ROOT = ROOT / "artifacts/data_audits/breast_subtype_evidence_20261009/source_annotations"
DAI_TABLE1 = SOURCE_AUDIT_ROOT / "dai2017_table1_receptors.csv"
DAI_TABLE3 = SOURCE_AUDIT_ROOT / "dai2017_table3_conflicts.csv"
DAI_ARTICLE_HTML = SOURCE_AUDIT_ROOT / "dai2017_article_page.html"
CELLOSAURUS_VALIDATION = SOURCE_AUDIT_ROOT / "cellosaurus_rrid_validation.tsv"
CELLOSAURUS_RELEASE_INFO = SOURCE_AUDIT_ROOT / "cellosaurus_release_info.json"
PUBLIC_SOURCE_MANIFEST = SOURCE_AUDIT_ROOT / "source_manifest.json"

MODEL_COLUMNS = [
    "ModelID",
    "PatientID",
    "CellLineName",
    "RRID",
    "DepmapModelType",
    "OncotreeLineage",
    "OncotreePrimaryDisease",
    "OncotreeSubtype",
    "OncotreeCode",
    "LegacyMolecularSubtype",
    "LegacySubSubtype",
    "PatientMolecularSubtype",
]
CTRP_COLUMNS = [
    "master_ccl_id",
    "ccl_name",
    "cellid",
    "Cellosaurus.Accession.id",
]
LINCS_COLUMNS = [
    "dataset",
    "LINCS_cell_id",
    "LINCS_base_cell_id",
    "LINCS_official_subtype",
    "DepMap_ModelID",
    "DepMap_CellLineName",
    "identity_status",
]

CELLOSAURUS_URLS = {
    "CVCL_0178": "https://www.cellosaurus.org/CVCL_0178",
    "CVCL_0332": "https://www.cellosaurus.org/CVCL_0332",
    "CVCL_0031": "https://www.cellosaurus.org/CVCL_0031",
    "CVCL_0062": "https://www.cellosaurus.org/CVCL_0062",
    "CVCL_0033": "https://www.cellosaurus.org/CVCL_0033",
}

# Raw receptor statuses and legacy labels transcribed from Table 1 of
# Güttler et al. 2021 (Western blot). These are keyed by RRID after identity
# review, not used as model-selection criteria.
WB_2021 = {
    "CVCL_0178": {"source_cell_name": "BT-20", "er": "-", "pr": "-", "her2": "-", "subtype": "basal A"},
    "CVCL_0332": {"source_cell_name": "HS578T", "er": "-", "pr": "-", "her2": "-", "subtype": "basal B"},
    "CVCL_0031": {"source_cell_name": "MCF-7", "er": "+", "pr": "-", "her2": "-", "subtype": "luminal"},
    "CVCL_0062": {"source_cell_name": "MDA-MB-231", "er": "-", "pr": "-", "her2": "-", "subtype": "basal B"},
    "CVCL_0033": {"source_cell_name": "SKBR3", "er": "-", "pr": "-", "her2": "+", "subtype": "luminal"},
}

# Exact line names in Subik et al. 2010 Table 3. The paper uses Allred scores
# for ER/PR and HER2 IHC 0-1+ versus 3+. Its "HS598T" spelling is retained as
# an orphan source row because the paper does not identify it as CVCL_0332.
IHC_2010 = {
    "CVCL_0178": {"source_cell_name": "BT-20", "er": "0", "pr": "0", "her2": "0-1+"},
    "CVCL_0031": {"source_cell_name": "MCF-7", "er": "6", "pr": "6", "her2": "0-1+"},
    "CVCL_0062": {"source_cell_name": "MDA-MB-231", "er": "0", "pr": "0", "her2": "0-1+"},
    "CVCL_0033": {"source_cell_name": "SKBR-3", "er": "0", "pr": "0", "her2": "3+"},
}
IHC_2010_ORPHANS = [
    {
        "source_cell_name": "HS598T",
        "er": "0",
        "pr": "0",
        "her2": "0-1+",
        "mapping_status": "unresolved_name_not_mapped_to_Hs_578T",
        "mapping_note": "Paper uses HS598T in methods and table; no source evidence establishes this as Cellosaurus CVCL_0332.",
    }
]


def read_selected_delimited(path: Path, delimiter: str, columns: list[str]) -> list[dict[str, str]]:
    """Read selected metadata columns only; all other columns are ignored."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader)
        index = {name: i for i, name in enumerate(header)}
        missing = [name for name in columns if name not in index]
        if missing:
            raise ValueError(f"{path} missing required metadata columns: {missing}")
        result: list[dict[str, str]] = []
        for line_no, row in enumerate(reader, start=2):
            if len(row) < len(header):
                raise ValueError(f"short row in {path}:{line_no}")
            result.append({name: row[index[name]].strip() for name in columns})
        return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm_rrid(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def norm_cell_name(value: str) -> str:
    """Normalize punctuation/case only for deterministic literature-name joins."""
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def receptor_state(source_id: str, field: str, raw: str) -> str:
    """Map an explicitly reported source value to a receptor state."""
    value = (raw or "").strip()
    if source_id in {"Dai_2017_Table1", "Guttler_2021_WB_Table1"}:
        return {"+": "positive", "-": "negative", "+/-": "equivocal"}.get(value, "missing")
    if source_id == "Subik_2010_IHC_Table3":
        if field in {"ER", "PR"}:
            return {"0": "negative", "6": "positive"}.get(value, "missing")
        if field == "HER2":
            return {"0-1+": "negative", "3+": "positive"}.get(value, "missing")
    return "missing"


def is_breast(model: dict[str, str]) -> bool:
    joined = " ".join(
        model.get(field, "")
        for field in ("OncotreeLineage", "OncotreePrimaryDisease")
    ).lower()
    return "breast" in joined


def csv_write(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def json_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check_inputs() -> list[Path]:
    required = [
        DEP_MAP, CTRP_CELLS, CTRP_RDS, LINCS_CROSSWALK, LINCS_ROOT_RECEIPT,
        LINCS_CELL_INFO, CTRP_MATERIALIZATION, DAI_TABLE1, DAI_TABLE3,
        DAI_ARTICLE_HTML, CELLOSAURUS_VALIDATION, CELLOSAURUS_RELEASE_INFO,
        PUBLIC_SOURCE_MANIFEST,
    ]
    missing = [p.as_posix() for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"required input file(s) missing: {missing}")
    return required


def build(output_dir: Path) -> dict[str, Any]:
    if not sys.platform.startswith("linux") or os.environ.get("CONDA_DEFAULT_ENV") != "drugscreening-gpu":
        raise RuntimeError("must run under WSL2 Conda environment drugscreening-gpu")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    input_paths = check_inputs()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    models = read_selected_delimited(DEP_MAP, ",", MODEL_COLUMNS)
    ctrp_cells = read_selected_delimited(CTRP_CELLS, ",", CTRP_COLUMNS)
    lincs_rows = read_selected_delimited(LINCS_CROSSWALK, "\t", LINCS_COLUMNS)
    dai_rows = read_selected_delimited(
        DAI_TABLE1,
        ",",
        ["article_cell_line", "article_er", "article_pr", "article_her2", "article_historical_subtype", "article_source_basis", "article_reference_ids"],
    )
    dai_by_normalized_name: dict[str, list[dict[str, str]]] = {}
    for row in dai_rows:
        dai_by_normalized_name.setdefault(norm_cell_name(row["article_cell_line"]), []).append(row)
    dai_table3_rows = read_selected_delimited(
        DAI_TABLE3,
        ",",
        ["article_cell_line", "article_er", "article_pr", "article_her2", "article_historical_subtype", "article_reference_ids"],
    )

    model_by_id: dict[str, dict[str, str]] = {}
    models_by_rrid: dict[str, list[dict[str, str]]] = {}
    for row in models:
        mid = row["ModelID"]
        if mid in model_by_id:
            raise ValueError(f"duplicate DepMap ModelID: {mid}")
        model_by_id[mid] = row
        rrid = norm_rrid(row["RRID"])
        if rrid:
            models_by_rrid.setdefault(rrid, []).append(row)

    ctrp_by_rrid: dict[str, list[dict[str, str]]] = {}
    for row in ctrp_cells:
        rrid = norm_rrid(row["Cellosaurus.Accession.id"])
        if rrid:
            ctrp_by_rrid.setdefault(rrid, []).append(row)

    breast_models = [row for row in models if is_breast(row)]
    breast_with_ctrp = {
        row["ModelID"]
        for row in breast_models
        if len(ctrp_by_rrid.get(norm_rrid(row["RRID"]), [])) == 1
        and len(models_by_rrid.get(norm_rrid(row["RRID"]), [])) == 1
    }

    # The approved crosswalk includes GSE70138 and GSE92742. This package uses
    # only GSE92742 identity rows and only these identity fields.
    lincs_gse92742 = [row for row in lincs_rows if row["dataset"] == "GSE92742"]
    by_base: dict[str, list[dict[str, str]]] = {}
    for row in lincs_gse92742:
        if row["LINCS_base_cell_id"] and row["DepMap_ModelID"]:
            by_base.setdefault(row["LINCS_base_cell_id"], []).append(row)

    candidate_rows: list[dict[str, Any]] = []
    intersection_rows: list[dict[str, Any]] = []
    source_label_rows: list[dict[str, Any]] = []
    receptor_evidence_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    subtype_disagreements: list[dict[str, Any]] = []
    excluded_lincs_base_rows: list[dict[str, Any]] = []
    source_alias_rows: list[dict[str, Any]] = []

    for base_id in sorted(by_base):
        records = by_base[base_id]
        model_ids = sorted({row["DepMap_ModelID"] for row in records})
        if len(model_ids) != 1:
            excluded_lincs_base_rows.append(
                {"LINCS_base_cell_id": base_id, "status": "ambiguous_DepMap_model_mapping", "model_ids": ";".join(model_ids)}
            )
            continue
        mid = model_ids[0]
        model = model_by_id.get(mid)
        if model is None:
            excluded_lincs_base_rows.append(
                {"LINCS_base_cell_id": base_id, "status": "DepMap_ModelID_absent_from_Model_csv", "model_ids": mid}
            )
            continue
        rrid_key = norm_rrid(model["RRID"])
        ctrp_matches = ctrp_by_rrid.get(rrid_key, [])
        depmap_rrid_matches = models_by_rrid.get(rrid_key, [])
        has_breast_annotation = is_breast(model)
        if len(ctrp_matches) > 1:
            ctrp_unique_ids = {match["master_ccl_id"] for match in ctrp_matches}
            ctrp_unique_names = {match["cellid"] for match in ctrp_matches}
            if len(ctrp_unique_ids) != 1 or len(ctrp_unique_names) != 1:
                ctrp_status = "ambiguous_CTRP_exact_RRID_matches"
                ctrp = {}
            else:
                ctrp_status = "duplicate_rows_same_CTRP_identity"
                ctrp = ctrp_matches[0]
        elif len(ctrp_matches) == 1:
            ctrp_status = "exact_unique_RRID_match"
            ctrp = ctrp_matches[0]
        else:
            ctrp_status = "no_exact_CTRP_RRID_match"
            ctrp = {}

        if len(depmap_rrid_matches) > 1:
            status = "ambiguous_DepMap_RRID_maps_to_multiple_models"
        elif not has_breast_annotation:
            status = "not_breast_by_DepMap_model_metadata"
        elif ctrp_status in ("exact_unique_RRID_match", "duplicate_rows_same_CTRP_identity"):
            status = "three_way_exact_identity_match"
        else:
            status = ctrp_status

        lincs_ids = sorted({row["LINCS_cell_id"] for row in records})
        lincs_official_subtypes = sorted(
            {
                row.get("LINCS_official_subtype", "")
                for row in lincs_rows
                if row["dataset"] == "GSE92742"
                and row["LINCS_base_cell_id"] == base_id
            }
        )
        candidate = {
            "LINCS_dataset": "GSE92742",
            "LINCS_base_cell_id": base_id,
            "LINCS_cell_ids": ";".join(lincs_ids),
            "n_LINCS_cell_ids_collapsed_to_base": len(lincs_ids),
            "DepMap_ModelID": mid,
            "DepMap_CellLineName": model["CellLineName"],
            "DepMap_RRID": model["RRID"],
            "DepMap_PatientID": model["PatientID"],
            "DepmapModelType_raw": model["DepmapModelType"],
            "OncotreeLineage_raw": model["OncotreeLineage"],
            "OncotreePrimaryDisease_raw": model["OncotreePrimaryDisease"],
            "OncotreeSubtype_raw": model["OncotreeSubtype"],
            "OncotreeCode_raw": model["OncotreeCode"],
            "LegacyMolecularSubtype_raw": model["LegacyMolecularSubtype"],
            "LegacySubSubtype_raw": model["LegacySubSubtype"],
            "PatientMolecularSubtype_raw": model["PatientMolecularSubtype"],
            "LINCS_official_subtype_raw": ";".join(x for x in lincs_official_subtypes if x),
            "CTRP_master_ccl_id": ctrp.get("master_ccl_id", ""),
            "CTRP_cellid": ctrp.get("cellid", ""),
            "CTRP_ccl_name": ctrp.get("ccl_name", ""),
            "CTRP_Cellosaurus_Accession": ctrp.get("Cellosaurus.Accession.id", ""),
            "CTRP_exact_RRID_status": ctrp_status,
            "three_way_status": status,
            "Cellosaurus_record_url": CELLOSAURUS_URLS.get(model["RRID"], ""),
        }
        candidate_rows.append(candidate)
        if status != "three_way_exact_identity_match":
            excluded_lincs_base_rows.append(
                {"LINCS_base_cell_id": base_id, "status": status, "model_ids": mid}
            )
            continue
        intersection_rows.append(candidate)

        dai_name_key = norm_cell_name(model["CellLineName"])
        dai_name_matches = dai_by_normalized_name.get(dai_name_key, [])
        if len(dai_name_matches) != 1:
            raise ValueError(
                f"Dai 2017 Table 1 name is not a unique punctuation/case-normalized match for {model['CellLineName']}: {len(dai_name_matches)} rows"
            )
        dai = dai_name_matches[0]
        source_alias_rows.append(
            {
                "DepMap_ModelID": mid,
                "DepMap_CellLineName": model["CellLineName"],
                "RRID": model["RRID"],
                "source_id": "Dai_2017_Table1",
                "source_cell_name": dai["article_cell_line"],
                "match_rule": "case_and_punctuation_normalized_exact",
                "mapping_status": "unique_exact_normalized_name_match",
                "table_locator": "Table 1",
            }
        )

        # Source-level raw subtype fields stay separate. No harmonization of
        # luminal/basal/HER2 labels into PAM50 is attempted.
        source_label_rows.extend(
            [
                {
                    "DepMap_ModelID": mid,
                    "DepMap_CellLineName": model["CellLineName"],
                    "RRID": model["RRID"],
                    "source_id": "DepMap_24Q2_Model.csv",
                    "source_version": "DepMap Public 24Q2; Model.csv",
                    "source_field": "LegacyMolecularSubtype",
                    "raw_value": model["LegacyMolecularSubtype"],
                    "ontology_note": "Historical molecular subtype; not a PAM50 call.",
                    "source_url": "https://doi.org/10.25452/figshare.plus.25880521.v1",
                },
                {
                    "DepMap_ModelID": mid,
                    "DepMap_CellLineName": model["CellLineName"],
                    "RRID": model["RRID"],
                    "source_id": "DepMap_24Q2_Model.csv",
                    "source_version": "DepMap Public 24Q2; Model.csv",
                    "source_field": "LegacySubSubtype",
                    "raw_value": model["LegacySubSubtype"],
                    "ontology_note": "Historical sub-subtype; no conversion to receptor class or PAM50.",
                    "source_url": "https://doi.org/10.25452/figshare.plus.25880521.v1",
                },
                {
                    "DepMap_ModelID": mid,
                    "DepMap_CellLineName": model["CellLineName"],
                    "RRID": model["RRID"],
                    "source_id": "DepMap_24Q2_Model.csv",
                    "source_version": "DepMap Public 24Q2; Model.csv",
                    "source_field": "PatientMolecularSubtype",
                    "raw_value": model["PatientMolecularSubtype"],
                    "ontology_note": "Per-model source annotation; blank means missing, not a biological negative.",
                    "source_url": "https://doi.org/10.25452/figshare.plus.25880521.v1",
                },
                {
                    "DepMap_ModelID": mid,
                    "DepMap_CellLineName": model["CellLineName"],
                    "RRID": model["RRID"],
                    "source_id": "LINCS_GSE92742_approved_identity_crosswalk",
                    "source_version": "EXP-050 approved, root-readback-verified crosswalk",
                    "source_field": "LINCS_official_subtype",
                    "raw_value": ";".join(x for x in lincs_official_subtypes if x),
                    "ontology_note": "Broad histology label (for example carcinoma/adenocarcinoma), not an intrinsic breast subtype.",
                    "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742",
                },
                {
                    "DepMap_ModelID": mid,
                    "DepMap_CellLineName": model["CellLineName"],
                    "RRID": model["RRID"],
                    "source_id": "Guttler_2021_WB_Table1",
                    "source_version": "Güttler et al., Int J Mol Sci 2021;22(16):8808; Table 1",
                    "source_field": "Subtype",
                    "raw_value": WB_2021[model["RRID"]]["subtype"],
                    "ontology_note": "Source-specific cell-line subtype label; not a PAM50 call and kept separate from receptor-derived group.",
                    "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8395940/",
                },
                {
                    "DepMap_ModelID": mid,
                    "DepMap_CellLineName": model["CellLineName"],
                    "RRID": model["RRID"],
                    "source_id": "Dai_2017_Table1",
                    "source_version": "Dai et al., J Cancer 2017;8(16):3131-3141; DOI 10.7150/jca.18457; Table 1",
                    "source_field": "article_historical_subtype",
                    "raw_value": dai["article_historical_subtype"],
                    "ontology_note": "Dai's five-group ER/PR/HER2-based cell-line scheme (LA/LB/H/TNA/TNB); original source label, not PAM50.",
                    "source_url": "https://www.jcancer.org/v08p3131.htm",
                },
            ]
        )

        wb = WB_2021.get(model["RRID"], {})
        ihc = IHC_2010.get(model["RRID"], {})
        if not wb:
            raise ValueError(f"no curated primary receptor evidence for RRID {model['RRID']}")
        dai_receptor_record = {
            "source_cell_name": dai["article_cell_line"],
            "er": dai["article_er"],
            "pr": dai["article_pr"],
            "her2": dai["article_her2"],
            "subtype": dai["article_historical_subtype"],
            "source_basis": dai["article_source_basis"],
            "source_reference_ids": dai["article_reference_ids"],
        }
        for source_id, record, method, source_year, table_ref, source_url, source_license in [
            (
                "Dai_2017_Table1",
                dai_receptor_record,
                "Literature synthesis of ER/PR/HER2 receptor annotations; Table 1 source/reference columns retained",
                "2017",
                "Table 1",
                "https://www.jcancer.org/v08p3131.htm",
                "CC BY-NC 4.0",
            ),
            (
                "Guttler_2021_WB_Table1",
                wb,
                "Western blot (Table 1; Figure 2 describes ER/PR/HER2 protein detection)",
                "2021",
                "Table 1",
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC8395940/",
                "CC BY 4.0",
            ),
            (
                "Subik_2010_IHC_Table3",
                ihc,
                "Immunohistochemistry on cell blocks; ER/PR Allred scores, HER2 IHC score",
                "2010",
                "Table 3 and Methods",
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC2914277/",
                "Open access; unrestricted non-commercial reuse with attribution; not a CC BY license",
            ),
        ]:
            if not record:
                continue
            receptor_evidence_rows.append(
                {
                    "DepMap_ModelID": mid,
                    "DepMap_CellLineName": model["CellLineName"],
                    "RRID": model["RRID"],
                    "source_id": source_id,
                    "source_year": source_year,
                    "source_cell_name": record["source_cell_name"],
                    "source_cell_name_identity_status": (
                        "case_and_punctuation_normalized_exact_match_to_DepMap_name"
                        if source_id == "Dai_2017_Table1"
                        else "exact_name_reviewed_against_RRID"
                        if source_id.endswith("2021_WB_Table1")
                        else "exact_cell_line_name_in_paper"
                    ),
                    "assay": method,
                    "source_table": table_ref,
                    "source_basis": record.get("source_basis", ""),
                    "source_reference_ids": record.get("source_reference_ids", ""),
                    "ER_raw": record["er"],
                    "PR_raw": record["pr"],
                    "HER2_raw": record["her2"],
                    "raw_subtype_label": record.get("subtype", ""),
                    "source_url": source_url,
                    "source_license": source_license,
                }
            )

        marker_raw_values = {
            "ER": [("Dai_2017_Table1", dai["article_er"]), ("Guttler_2021_WB_Table1", wb["er"])],
            "PR": [("Dai_2017_Table1", dai["article_pr"]), ("Guttler_2021_WB_Table1", wb["pr"])],
            "HER2": [("Dai_2017_Table1", dai["article_her2"]), ("Guttler_2021_WB_Table1", wb["her2"])],
        }
        if ihc:
            marker_raw_values["ER"].append(("Subik_2010_IHC_Table3", ihc["er"]))
            marker_raw_values["PR"].append(("Subik_2010_IHC_Table3", ihc["pr"]))
            marker_raw_values["HER2"].append(("Subik_2010_IHC_Table3", ihc["her2"]))

        marker_states: dict[str, str] = {}
        marker_sources: dict[str, str] = {}
        marker_conflicts: dict[str, str] = {}
        for field, values in marker_raw_values.items():
            normalized = [(source, receptor_state(source, field, raw)) for source, raw in values]
            explicit = {state for _, state in normalized if state in {"positive", "negative"}}
            marker_sources[field] = ";".join(source for source, state in normalized if state != "missing")
            if len(explicit) > 1:
                marker_states[field] = "conflict"
                marker_conflicts[field] = "; ".join(
                    f"{source}={state}" for source, state in normalized if state != "missing"
                )
            elif explicit:
                marker_states[field] = next(iter(explicit))
                marker_conflicts[field] = ""
            else:
                marker_states[field] = "unknown"
                marker_conflicts[field] = "no explicit positive/negative source status"

        er = marker_states["ER"]
        pr_consensus = marker_states["PR"]
        her2 = marker_states["HER2"]
        if er == "negative" and pr_consensus == "negative" and her2 == "negative":
            receptor_group = "triple_negative_receptor_phenotype"
        elif (er == "positive" or pr_consensus == "positive") and her2 == "negative":
            receptor_group = "HR_positive_HER2_negative"
        elif er == "negative" and pr_consensus == "negative" and her2 == "positive":
            receptor_group = "ER_negative_PR_negative_HER2_positive"
        else:
            receptor_group = "unresolved_receptor_group"
        normalized_rows.append(
            {
                "DepMap_ModelID": mid,
                "DepMap_CellLineName": model["CellLineName"],
                "RRID": model["RRID"],
                "ER_normalized": er,
                "ER_sources": marker_sources["ER"],
                "ER_conflict_detail": marker_conflicts["ER"],
                "PR_normalized": pr_consensus,
                "PR_sources": marker_sources["PR"],
                "PR_conflict_detail": marker_conflicts["PR"],
                "HER2_normalized": her2,
                "HER2_sources": marker_sources["HER2"],
                "HER2_conflict_detail": marker_conflicts["HER2"],
                "receptor_defined_group": receptor_group,
                "normalization_rule": "receptor_group_v2: source values are mapped only when explicitly positive/negative; conflicting marker calls remain conflict; HR-positive/HER2-negative requires either ER or PR positive plus HER2 negative; triple-negative requires ER, PR, and HER2 all negative; unresolved values are not imputed.",
                "PAM50_status": "not_present_in_audited_metadata",
                "PAM50_subtype": "",
                "PAM50_note": "DepMap Model.csv, CTRP identity metadata, and approved LINCS crosswalk contain no PAM50 call; no expression-based classifier was run.",
                "raw_paper_subtype_2021": wb.get("subtype", ""),
                "raw_paper_subtype_2017": dai.get("article_historical_subtype", ""),
                "raw_DepMap_LegacyMolecularSubtype": model["LegacyMolecularSubtype"],
                "raw_DepMap_LegacySubSubtype": model["LegacySubSubtype"],
                "raw_DepMap_PatientMolecularSubtype": model["PatientMolecularSubtype"],
                "raw_LINCS_official_subtype": ";".join(x for x in lincs_official_subtypes if x),
                "raw_DepMap_and_Guttler_subtype_agree_textually": str(
                    re.sub(r"[^a-z0-9]", "", (wb.get("subtype", "")).lower())
                    == re.sub(r"[^a-z0-9]", "", (model["LegacyMolecularSubtype"] or "").lower())
                ).lower(),
            }
        )
        if normalized_rows[-1]["raw_DepMap_and_Guttler_subtype_agree_textually"] == "false":
            subtype_disagreements.append(
                {
                    "DepMap_ModelID": mid,
                    "CellLineName": model["CellLineName"],
                    "RRID": model["RRID"],
                    "DepMap_LegacyMolecularSubtype": model["LegacyMolecularSubtype"],
                    "Guttler_2021_Table1_subtype": wb.get("subtype", ""),
                    "Dai_2017_Table1_subtype": dai.get("article_historical_subtype", ""),
                    "interpretation": "DepMap and Güttler source-specific labels differ textually; Dai uses a five-group receptor-based scheme. Retain all raw labels and do not coerce to PAM50.",
                }
            )

    # Explicitly preserve the 2010 source's nonmatching Hs 598T string rather
    # than silently using it as Hs 578T evidence.
    orphan_rows = []
    for row in IHC_2010_ORPHANS:
        orphan_rows.append(
            {
                "source_id": "Subik_2010_IHC_Table3",
                "source_cell_name": row["source_cell_name"],
                "RRID": "",
                "ER_raw": row["er"],
                "PR_raw": row["pr"],
                "HER2_raw": row["her2"],
                "mapping_status": row["mapping_status"],
                "mapping_note": row["mapping_note"],
                "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC2914277/",
            }
        )
        source_alias_rows.append(
            {
                "DepMap_ModelID": "",
                "DepMap_CellLineName": "",
                "RRID": "",
                "source_id": "Subik_2010_IHC_Table3",
                "source_cell_name": row["source_cell_name"],
                "match_rule": "rejected_nonmatching_spelling_without_identity_key",
                "mapping_status": row["mapping_status"],
                "candidate_reference_only": "Hs 578T / CVCL_0332; not assigned",
                "table_locator": "Table 3 and Methods",
            }
        )

    intersection_rows.sort(key=lambda row: row["DepMap_ModelID"])
    candidate_rows.sort(key=lambda row: row["DepMap_ModelID"])
    normalized_rows.sort(key=lambda row: row["DepMap_ModelID"])

    # The expectation is based on identity keys only; fail closed on any change
    # in the previously approved model-level intersection.
    expected_rrids = {"CVCL_0178", "CVCL_0332", "CVCL_0031", "CVCL_0062", "CVCL_0033"}
    actual_rrids = {row["DepMap_RRID"] for row in intersection_rows}
    if actual_rrids != expected_rrids:
        raise ValueError(
            f"unexpected three-way breast identity intersection: expected {sorted(expected_rrids)}, got {sorted(actual_rrids)}"
        )
    patient_ids = [row["DepMap_PatientID"] for row in intersection_rows]
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("intersection contains repeated DepMap PatientID; review independent cell-line identities")

    output_dir.mkdir(parents=True, exist_ok=False)
    intersection_path = output_dir / "three_way_breast_model_intersection.tsv"
    candidate_path = output_dir / "lincs_breast_candidate_join_audit.tsv"
    labels_path = output_dir / "source_label_evidence.tsv"
    receptor_path = output_dir / "receptor_status_evidence.tsv"
    normalized_path = output_dir / "normalized_receptor_groups.tsv"
    disagreement_path = output_dir / "subtype_label_disagreements.tsv"
    orphan_path = output_dir / "unresolved_source_line_names.tsv"
    alias_path = output_dir / "source_alias_mapping.tsv"
    manifest_path = output_dir / "source_manifest.json"
    summary_path = output_dir / "summary.json"
    readme_path = output_dir / "README.md"

    csv_write(intersection_path, list(intersection_rows[0].keys()), intersection_rows)
    csv_write(candidate_path, list(candidate_rows[0].keys()) if candidate_rows else ["status"], candidate_rows)
    source_label_columns = [
        "DepMap_ModelID", "DepMap_CellLineName", "RRID", "source_id", "source_version",
        "source_field", "raw_value", "ontology_note", "source_url",
    ]
    csv_write(labels_path, source_label_columns, source_label_rows)
    receptor_columns = [
        "DepMap_ModelID", "DepMap_CellLineName", "RRID", "source_id", "source_year",
        "source_cell_name", "source_cell_name_identity_status", "assay", "source_table",
        "source_basis", "source_reference_ids", "ER_raw", "PR_raw", "HER2_raw",
        "raw_subtype_label", "source_url", "source_license",
    ]
    csv_write(receptor_path, receptor_columns, receptor_evidence_rows)
    csv_write(normalized_path, list(normalized_rows[0].keys()), normalized_rows)
    csv_write(
        disagreement_path,
        ["DepMap_ModelID", "CellLineName", "RRID", "DepMap_LegacyMolecularSubtype", "Guttler_2021_Table1_subtype", "Dai_2017_Table1_subtype", "interpretation"],
        subtype_disagreements,
    )
    csv_write(
        orphan_path,
        ["source_id", "source_cell_name", "RRID", "ER_raw", "PR_raw", "HER2_raw", "mapping_status", "mapping_note", "source_url"],
        orphan_rows,
    )
    csv_write(
        alias_path,
        ["DepMap_ModelID", "DepMap_CellLineName", "RRID", "source_id", "source_cell_name", "match_rule", "mapping_status", "candidate_reference_only", "table_locator"],
        source_alias_rows,
    )

    ctrp_materialization = json.loads(CTRP_MATERIALIZATION.read_text(encoding="utf-8"))
    lincs_receipt = json.loads(LINCS_ROOT_RECEIPT.read_text(encoding="utf-8"))
    public_source_manifest = json.loads(PUBLIC_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    cellosaurus_release = json.loads(CELLOSAURUS_RELEASE_INFO.read_text(encoding="utf-8"))
    cellosaurus_rows = read_selected_delimited(
        CELLOSAURUS_VALIDATION, "\t", ["id", "ac", "acas", "dt", "problematic", "di", "ca"]
    )
    cellosaurus_by_accession: dict[str, list[dict[str, str]]] = {}
    for row in cellosaurus_rows:
        cellosaurus_by_accession.setdefault(norm_rrid(row["ac"]), []).append(row)
    if any(len(cellosaurus_by_accession.get(norm_rrid(rrid), [])) != 1 for rrid in expected_rrids):
        raise ValueError("cached Cellosaurus snapshot does not uniquely contain all five exact intersection RRIDs")
    target_name_keys = {norm_cell_name(row["DepMap_CellLineName"]) for row in intersection_rows}
    dai_table3_target_name_overlaps = sorted(
        {row["article_cell_line"] for row in dai_table3_rows if norm_cell_name(row["article_cell_line"]) in target_name_keys}
    )
    input_manifest = []
    for path in input_paths:
        item: dict[str, Any] = {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        if path == LINCS_CROSSWALK:
            item["approved_crosswalk_root_receipt"] = "artifacts/experiments/EXP-050/20261006_breast_subtype_measured_transfer_v1/root_independent_readback_receipt.json"
            item["approved_crosswalk_status"] = lincs_receipt.get("status", "")
        if path == CTRP_CELLS:
            item["materialization_sha256_registered"] = ctrp_materialization["materialization"]["exports"]["root.cell"]["sha256"]
            item["materialization_registered_rows"] = ctrp_materialization["materialization"]["exports"]["root.cell"]["rows"]
        input_manifest.append(item)

    source_manifest = {
        "scope": "EXP-068 metadata-only breast cell-line identity intersection and subtype evidence; no response/expression numeric data read",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [
            {
                "source_id": "depmap_24q2_model_metadata",
                "version": "DepMap Public 24Q2; Model.csv",
                "url": "https://doi.org/10.25452/figshare.plus.25880521.v1",
                "license": "CC BY 4.0",
                "license_note": "The release page identifies CC BY 4.0.",
                "data_use": "ModelID, PatientID, CellLineName, RRID, Oncotree and legacy subtype metadata only.",
            },
            {
                "source_id": "ctrpv2_pharmacoset_identity_metadata",
                "version": "ORCESTRA CTRPv2 PharmacoSet Zenodo v1 (published 2020-06-24); locally materialized cells.csv",
                "url": "https://doi.org/10.5281/zenodo.3905470",
                "license": "CC BY 4.0 per Zenodo/DepMap data usage statement",
                "license_note": "The processed cell metadata are derived from the deposited PharmacoSet; no response table was opened.",
                "data_use": "Cellosaurus.Accession.id and cell identifiers only.",
            },
            {
                "source_id": "approved_lincs_identity_crosswalk",
                "version": "EXP-050 approved crosswalk; GSE92742 rows only; prior root readback status recorded in input manifest",
                "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742",
                "license": "No explicit reuse license identified in GEO series metadata reviewed; attribution to GEO/Broad LINCS retained.",
                "data_use": "LINCS cell ID, LINCS base-cell ID, and already-approved DepMap ModelID mapping only.",
            },
            {
                "source_id": "cellosaurus_identity_records",
                "version": f"Cellosaurus release {cellosaurus_release['Cellosaurus']['header']['release']['version']}; updated {cellosaurus_release['Cellosaurus']['header']['release']['updated']}; 2026-10-09 local query snapshot",
                "url": "https://api.cellosaurus.org/search/cell-line?q=ac:<RRID>&fields=id,ac,acas,dt,prob,di,ca&format=tsv",
                "license": "CC BY 4.0",
                "license_note": "Current Cellosaurus FAQ identifies CC BY 4.0. The cached release/API snapshot returns all five intersection RRIDs; no receptor status is inferred from Cellosaurus.",
                "data_use": "RRID identity and cell-line name only.",
            },
            {
                "source_id": "dai_2017_receptor_and_historical_subtype_table",
                "version": "Dai et al., J Cancer 2017;8(16):3131-3141; DOI 10.7150/jca.18457; Table 1 and Table 3; local HTML/CSV snapshot retrieved 2026-10-09",
                "url": "https://www.jcancer.org/v08p3131.htm",
                "license": "CC BY-NC 4.0",
                "license_note": "Publisher page states Creative Commons Attribution-NonCommercial 4.0; non-commercial reuse with attribution only.",
                "data_use": "Table 1 receptor values, original five-group subtype abbreviations, and cited source/reference columns; Table 3 used to preserve documented label conflicts.",
            },
            {
                "source_id": "guttler_2021_receptor_evidence",
                "version": "Güttler et al., Int J Mol Sci 2021;22(16):8808; DOI 10.3390/ijms22168808; Table 1; accessed 2026-10-09",
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8395940/",
                "license": "CC BY 4.0",
                "data_use": "Raw ER/PR/HER2 signs and source subtype labels; Western blot evidence.",
            },
            {
                "source_id": "subik_2010_receptor_corroboration",
                "version": "Subik et al., Breast Cancer (Auckl) 2010;4:35-41; PMID 20697531; PMCID PMC2914277; Table 3; accessed 2026-10-09",
                "url": "https://pubmed.ncbi.nlm.nih.gov/20697531/",
                "license": "Open access; unrestricted non-commercial use with attribution according to article copyright notice; not a CC BY license.",
                "data_use": "Independent IHC corroboration for exact-name matches; HS598T retained as an unresolved orphan source name.",
            },
        ],
        "inputs": input_manifest,
        "external_source_snapshot": {
            "retrieved_at_utc": public_source_manifest.get("retrieved_at_utc", ""),
            "cellosaurus_release_version": cellosaurus_release["Cellosaurus"]["header"]["release"]["version"],
            "cellosaurus_updated": cellosaurus_release["Cellosaurus"]["header"]["release"]["updated"],
            "cellosaurus_requested_primary_rrids": public_source_manifest["primary_sources"]["cellosaurus"]["requested_primary_rrids"],
            "cellosaurus_returned_primary_rrids": public_source_manifest["primary_sources"]["cellosaurus"]["returned_primary_rrids"],
            "cellosaurus_unreturned_rrids": public_source_manifest["primary_sources"]["cellosaurus"]["unreturned_rrids"],
            "intersection_rrids_verified_in_cached_cellosaurus": sorted(expected_rrids),
            "intersection_rrid_rows_checked": len(expected_rrids),
            "dai2017_page_url": public_source_manifest["primary_sources"]["dai_2017_article"]["url"],
            "dai2017_doi": public_source_manifest["primary_sources"]["dai_2017_article"]["doi"],
            "dai2017_table1_locator": "84-line Table 1 parsed to dai2017_table1_receptors.csv",
            "dai2017_table3_locator": "8 conflict records parsed to dai2017_table3_conflicts.csv",
            "dai2017_table3_conflict_names_matching_intersection": dai_table3_target_name_overlaps,
            "dai2017_external_hashes": {
                "article_html_sha256": public_source_manifest["primary_sources"]["dai_2017_article"]["page_sha256"],
                "table1_csv_sha256": public_source_manifest["primary_sources"]["dai_2017_article"]["table1_sha256"],
                "table3_csv_sha256": public_source_manifest["primary_sources"]["dai_2017_article"]["table3_sha256"],
            },
            "cellosaurus_external_hashes": {
                "validation_tsv_sha256": public_source_manifest["primary_sources"]["cellosaurus"]["records_sha256"],
                "release_info_json_sha256": public_source_manifest["primary_sources"]["cellosaurus"]["release_info_sha256"],
            },
            "source_files": [
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in [DAI_ARTICLE_HTML, DAI_TABLE1, DAI_TABLE3, CELLOSAURUS_VALIDATION, CELLOSAURUS_RELEASE_INFO, PUBLIC_SOURCE_MANIFEST]
            ],
            "source_refresh_resource_ledger": {
                "preserved_refresh_group_started_utc_approx": "2026-10-09T03:11:52Z",
                "last_complete_snapshot_and_process_check_utc_approx": "2026-10-09T03:14:32Z",
                "observed_wall_seconds_approx": 160,
                "cpu_seconds": "not instrumented; post-hoc process sample was 3.2 percent at about 133 seconds elapsed, so no exact total is claimed",
                "gpu_hours": 0,
                "external_paid_cost": 0,
                "duplicate_refresh_pids_1832_1845": "not running when checked; no refresh was started by this evidence builder",
            },
        },
        "input_use_boundary": {
            "opened": [
                "DepMap Model.csv selected metadata columns",
                "CTRPv2 materialized cells.csv identity columns",
                "approved crosswalk GSE92742 identity columns only",
                "existing CTRP materialization receipt and LINCS root readback receipt for provenance/hash binding",
            ],
            "not_opened": [
                "CTRP responses.csv or sensitivity_profiles.csv values",
                "DepMap expression matrices",
                "LINCS expression or perturbation response matrices",
                "PDO, outer, locked, PANACEA, or CDS-DB values",
            ],
            "selection_policy": "Cell lines were selected solely by the existing approved identity crosswalk and exact Cellosaurus RRID join; receptor and subtype annotations were attached afterward.",
        },
    }
    json_write(manifest_path, source_manifest)

    runtime = {
        "wall_seconds": round(time.perf_counter() - wall_start, 6),
        "cpu_process_seconds": round(time.process_time() - cpu_start, 6),
        "gpu_hours": 0,
        "external_paid_cost": 0,
    }
    conflicts = []
    for row in normalized_rows:
        for field in ("ER", "PR", "HER2"):
            if row[f"{field}_normalized"] == "conflict":
                conflicts.append(
                    {
                        "type": "receptor_status_conflict",
                        "field": field,
                        "model_id": row["DepMap_ModelID"],
                        "rrid": row["RRID"],
                        "sources": row[f"{field}_conflict_detail"],
                        "resolution": "Preserved as conflict; receptor group is derived only when the remaining explicit marker evidence supports it.",
                    }
                )
    if subtype_disagreements:
        conflicts.extend(
            {
                "type": "source_specific_subtype_label_difference",
                "field": "legacy molecular subtype",
                "model_id": row["DepMap_ModelID"],
                "rrid": row["RRID"],
                "sources": f"DepMap={row['DepMap_LegacyMolecularSubtype']}; Güttler2021={row['Guttler_2021_Table1_subtype']}; Dai2017={row['Dai_2017_Table1_subtype']}",
                "resolution": "All raw labels kept; different source taxonomies are not forced into a common PAM50 label.",
            }
            for row in subtype_disagreements
        )
    summary = {
        "status": "METADATA_ONLY_SUBTYPE_EVIDENCE_COMPLETE",
        "runtime": runtime,
        "cohort_counts": {
            "DepMap_24Q2_breast_models_in_Model_csv": len(breast_models),
            "DepMap_breast_models_with_unique_exact_CTRP_RRID_metadata_match": len(breast_with_ctrp),
            "approved_GSE92742_LINCS_unique_base_cell_ids": len(by_base),
            "LINCS_candidates_in_breast_lineage": sum(
                1 for row in candidate_rows if row["three_way_status"] != "not_breast_by_DepMap_model_metadata"
            ),
            "complete_three_way_unique_cell_line_models": len(intersection_rows),
            "complete_three_way_unique_DepMap_PatientID": len(set(patient_ids)),
            "complete_three_way_CTRP_exact_identity_coverage": f"{sum(1 for r in intersection_rows if r['CTRP_exact_RRID_status'] == 'exact_unique_RRID_match')}/{len(intersection_rows)}",
            "complete_three_way_LINCS_base_cell_coverage": f"{len({r['LINCS_base_cell_id'] for r in intersection_rows})}/{len(intersection_rows)}",
        },
        "receptor_evidence_counts": {
            "Dai_2017_literature_synthesis_rows_exactly_mapped": len(intersection_rows),
            "Guttler_2021_WB_source_rows_exactly_mapped": len(intersection_rows),
            "Subik_2010_IHC_source_rows_exactly_mapped": len(IHC_2010),
            "IHC_ambiguous_HS598T_orphan_rows_not_mapped": len(orphan_rows),
            "source_alias_mappings_exactly_attached": sum(1 for row in source_alias_rows if row["mapping_status"] == "unique_exact_normalized_name_match"),
            "source_aliases_rejected_as_unresolved": sum(1 for row in source_alias_rows if row["mapping_status"] != "unique_exact_normalized_name_match"),
            "PR_source_conflicts": sum(1 for row in normalized_rows if row["PR_normalized"] == "conflict"),
            "ER_source_conflicts": sum(1 for row in normalized_rows if row["ER_normalized"] == "conflict"),
            "HER2_source_conflicts": sum(1 for row in normalized_rows if row["HER2_normalized"] == "conflict"),
            "DepMap_Model_csv_PR_field_missing_across_intersection": len(intersection_rows),
            "PAM50_calls_available_in_audited_metadata": 0,
            "PAM50_missing_denominator": len(intersection_rows),
        },
        "dai2017_table3_intersection_name_conflicts": len(dai_table3_target_name_overlaps),
        "subtype_raw_label_disagreement_count": len(subtype_disagreements),
        "subtype_raw_label_disagreement_denominator": len(intersection_rows),
        "conflicts_and_limits": conflicts
        + [
            {
                "type": "PAM50_unavailable_in_audited_metadata",
                "field": "PAM50",
                "denominator": len(intersection_rows),
                "available": 0,
                "resolution": "Keep PAM50 missing; historical basal_A/basal_B/luminal/HER2_amp labels, Dai five-group labels, and 2021 paper labels are not relabeled as PAM50.",
            },
            {
                "type": "ambiguous_published_cell_line_spelling",
                "field": "2010 IHC line identity",
                "denominator": len(intersection_rows),
                "affected_source_name": "HS598T",
                "resolution": "Not linked to Hs 578T (CVCL_0332) without identity evidence.",
            },
        ],
        "normalization_rule": "ER, PR, HER2 stay separate. The receptor-defined group is derived only from those markers. basal_B is never used as a proxy for TNBC. No PAM50 calls are imputed.",
        "denominator_definition": "DepMap breast denominator includes Model.csv rows whose OncotreeLineage or OncotreePrimaryDisease names breast; OncotreeSubtype is excluded from this filter because it can name a mammary fibroblast model rather than a breast cancer model.",
        "outputs": {},
    }
    output_paths = [intersection_path, candidate_path, labels_path, receptor_path, normalized_path, disagreement_path, orphan_path, alias_path, manifest_path]
    summary["outputs"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in output_paths
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme_path.write_text(
        "# EXP-068 乳腺细胞系身份与分型证据\n\n"
        "本目录只含身份及公开受体/分型注释元数据。三方交集按 LINCS base cell ID → 已批准 DepMap ModelID → DepMap RRID 与 CTRP Cellosaurus accession 精确连接。受体标记 ER、PR、HER2 分列；原始亚型来源保留在 `source_label_evidence.tsv`，统一标签只按受体定义生成。\n\n"
        "没有读取 CTRP 响应、DepMap/LINCS 表达数值、PDO、outer、locked、PANACEA 或 CDS-DB 数据。PAM50 在所审查的来源元数据中缺失，没有通过表达数据计算。Dai 2017 的 LA/LB/H/TNA/TNB、Güttler 2021 的 luminal/basal A/B 与 DepMap 历史字段分别保留，均不转换成 PAM50。`HS598T` 是 2010 IHC 论文中的原始名称，未映射成 Hs 578T。MCF7 的 PR 来源冲突保持显式标记。\n\n"
        "文件哈希、版本、URL、许可和运行费用见 `source_manifest.json` 与 `summary.json`。\n",
        encoding="utf-8",
    )
    summary["outputs"][readme_path.name] = {"bytes": readme_path.stat().st_size, "sha256": sha256(readme_path)}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run_manifest_path = output_dir / "run_manifest.json"
    run_manifest = {
        "run_id": "EXP-068_subtype_evidence_v1_attempt_005",
        "command": "conda run --no-capture-output -n drugscreening-gpu python scripts/data/EXP068/build_breast_subtype_evidence.py --output-dir artifacts/experiments/EXP-068/subtype_evidence_v1/attempt_005",
        "environment": "WSL2 Conda drugscreening-gpu",
        "scope": "metadata-only identity intersection and literature-coded receptor evidence; no model training or expression/response matrix access",
        "code_sha256": sha256(Path(__file__).resolve()),
        "runtime": runtime,
        "preserved_prior_attempt": {
            "path": "artifacts/experiments/EXP-068/subtype_evidence_v1/attempt_004",
            "reason_for_new_attempt": "Keep the unresolved HS598T alias fully unassigned and place all DepMap, LINCS, Dai 2017, and Güttler 2021 raw subtype labels together in the unified model table; prior files remain unchanged.",
        },
        "pre-finalization_failed_build_attempts": [
            {"attempt": 1, "reason": "non-target duplicate DepMap RRID was initially treated as a global error; no output written", "observed_tool_wall_seconds": 3.8, "gpu_hours": 0},
            {"attempt": 2, "reason": "intersection output key was initially named DepMap_RRID rather than RRID; no output written", "observed_tool_wall_seconds": 4.0, "gpu_hours": 0},
            {"attempt": 3, "reason": "Cellosaurus accession lookup compared normalized RRIDs with unnormalized keys; partial TSV files were written under attempt_003 before failure and were preserved unchanged", "observed_tool_wall_seconds": 3.7, "gpu_hours": 0},
            {"attempt": 4, "reason": "retry refused to overwrite the partial attempt_003 directory; no files changed", "observed_tool_wall_seconds": 3.7, "gpu_hours": 0},
        ],
        "superseded_preliminary_output": {
            "path": "artifacts/experiments/EXP-068/subtype_evidence_v1/",
            "issue": "Initial denominator filter included six mammary fibroblast models because it searched OncotreeSubtype; the current attempt uses OncotreeLineage/OncotreePrimaryDisease and reports 97 breast models.",
            "preserved_unchanged": True,
            "observed_tool_wall_seconds_for_preliminary_build": 4.7,
            "gpu_hours": 0,
        },
        "denominator_readback": {
            "method": "read-only DepMap Model.csv metadata count in WSL2 Conda drugscreening-gpu",
            "observed_tool_wall_seconds": 3.4,
            "gpu_hours": 0,
        },
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in output_paths + [summary_path, readme_path]
        },
        "limitations": [
            "MCF7 PR remains assay/source-dependent and conflicted (Dai 2017 and Subik 2010 support positive; Güttler 2021 Western blot reports negative).",
            "PAM50 is absent from audited metadata; no expression-based subtype classifier was run.",
            "The cached Cellosaurus API snapshot identifies release 56.0 updated 2026-06-25; one unrelated requested RRID CVCL_V618 was not returned.",
            "Subik 2010 reports the string HS598T; it is not linked to Hs 578T (CVCL_0332) without identity evidence.",
        ],
    }
    json_write(run_manifest_path, run_manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/experiments/EXP-068/subtype_evidence_v1",
    )
    args = parser.parse_args()
    result = build(args.output_dir.resolve())
    print(json.dumps({"status": result["status"], "runtime": result["runtime"], "cohort_counts": result["cohort_counts"], "outputs": result["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
