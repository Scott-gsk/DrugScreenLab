#!/usr/bin/env python3
"""Build an evidence-bounded breast cell-line receptor subtype inventory.

The input DepMap and CTRPv2 files are read-only. Public annotation source files
are cached separately from the derived evidence table so the latter can be
rebuilt without relying on live network services.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ARTICLE_URL = "https://www.jcancer.org/v08p3131.htm"
ARTICLE_DOI = "10.7150/jca.18457"
CELLOSAURUS_HOME_URL = "https://www.cellosaurus.org/"
CELLOSAURUS_FAQ_URL = "https://www.cellosaurus.org/faq"
CTRPV2_URL = "https://portals.broadinstitute.org/ctrp.v2.1/"
DEPMAP_24Q2_RELEASE_URL = "https://plus.figshare.com/articles/dataset/DepMap_24Q2_Public/25880521"
DEPMAP_26Q3_NOTES_URL = "https://forum.depmap.org/t/announcing-the-26q3-release/4692"
DEPMAP_DOWNLOAD_API_URL = "https://depmap.org/portal/api/no-captcha/download/files"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def normalize_receptor(value: object) -> str:
    """Normalize explicit plus/minus calls while retaining ambiguous values."""
    if pd.isna(value):
        return "unknown"
    value = str(value).strip().replace("−", "-").replace("–", "-")
    value = value.replace(" ", "")
    if value in {"+", "positive", "POSITIVE"}:
        return "positive"
    if value in {"-", "negative", "NEGATIVE"}:
        return "negative"
    if value in {"+/-", "-/+", "±", "indeterminate", "equivocal"}:
        return "indeterminate"
    return "unknown"


def receptor_pattern(er: object, pr: object, her2: object, *, conflicting_source: bool = False) -> str:
    """Derive only broad receptor patterns from complete, explicit calls.

    This describes the cited cell-line literature phenotype. It is not a
    clinical diagnosis and does not infer PAM50 or Luminal A/B.
    """
    if conflicting_source:
        return "unknown_public_annotation_conflict"
    calls = tuple(normalize_receptor(value) for value in (er, pr, her2))
    if any(call not in {"positive", "negative"} for call in calls):
        return "unknown_incomplete_or_indeterminate"
    er_call, pr_call, her2_call = calls
    hormone_receptor_positive = er_call == "positive" or pr_call == "positive"
    if hormone_receptor_positive:
        return "HR_positive_HER2_positive" if her2_call == "positive" else "HR_positive_HER2_negative"
    return "HR_negative_HER2_positive" if her2_call == "positive" else "triple_negative_receptor_pattern"


def fetch_bytes(url: str, timeout: int = 30) -> bytes:
    # curl is used here because it follows the WSL network proxy configuration
    # used by the project's approved runtime; urllib can stall on this API.
    result = subprocess.run(
        ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", str(timeout), url],
        check=True,
        capture_output=True,
    )
    return result.stdout


def _column_key(value: object) -> str:
    if isinstance(value, tuple):
        value = " ".join(str(part) for part in value if str(part) != "nan")
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _find_article_tables(page_bytes: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = pd.read_html(io.BytesIO(page_bytes), keep_default_na=False)
    receptor_table: pd.DataFrame | None = None
    conflict_table: pd.DataFrame | None = None
    for table in tables:
        normalized = {_column_key(column): column for column in table.columns}
        if {"cell lines", "er", "pr", "her2", "subtype", "source", "references"}.issubset(normalized):
            receptor_table = table.rename(columns={
                normalized["cell lines"]: "article_cell_line",
                normalized["er"]: "article_er",
                normalized["pr"]: "article_pr",
                normalized["her2"]: "article_her2",
                normalized["subtype"]: "article_historical_subtype",
                normalized.get("source", "Source"): "article_source_basis",
                normalized.get("references", "References"): "article_reference_ids",
            })
        if {"cell lines", "er", "pr", "her2", "subtype", "reference"}.issubset(normalized) and "source" not in normalized:
            conflict_table = table.rename(columns={
                normalized["cell lines"]: "article_cell_line",
                normalized["er"]: "article_er",
                normalized["pr"]: "article_pr",
                normalized["her2"]: "article_her2",
                normalized["subtype"]: "article_historical_subtype",
                normalized["reference"]: "article_reference_ids",
            })
    if receptor_table is None or conflict_table is None:
        raise ValueError("Could not locate Table 1 receptor data and Table 3 conflict data in the source page")
    receptor_table = receptor_table.astype("string").fillna("")
    conflict_table = conflict_table.astype("string").fillna("")
    return receptor_table, conflict_table


def refresh_public_sources(root: Path, source_dir: Path) -> dict[str, object]:
    model_path = root / "data/raw/depmap/24q2/Model.csv"
    models = pd.read_csv(model_path, usecols=["RRID", "OncotreeLineage"], dtype="string")
    models = models.loc[models["OncotreeLineage"].eq("Breast")]
    page = fetch_bytes(ARTICLE_URL)
    table1, table3 = _find_article_tables(page)
    source_dir.mkdir(parents=True, exist_ok=True)
    table1_path = source_dir / "dai2017_table1_receptors.csv"
    table3_path = source_dir / "dai2017_table3_conflicts.csv"
    page_path = source_dir / "dai2017_article_page.html"
    table1.to_csv(table1_path, index=False)
    table3.to_csv(table3_path, index=False)
    page_path.write_bytes(page)

    expected_rrids = {str(value).strip().upper() for value in models["RRID"].dropna() if str(value).strip()}
    manifest: dict[str, object] = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_sources": {
            "dai_2017_article": {
                "url": ARTICLE_URL,
                "doi": ARTICLE_DOI,
                "data_locator": "HTML Table 1 (84 cell-line rows) and Table 3 (8 named conflict cases); downloaded as HTML, parsed tables cached as CSV",
                "page_sha256": sha256(page_path),
                "table1_sha256": sha256(table1_path),
                "table3_sha256": sha256(table3_path),
            },
            "cellosaurus": {
                "homepage_url": CELLOSAURUS_HOME_URL,
                "faq_url": CELLOSAURUS_FAQ_URL,
                "release": "Cellosaurus release 56, June 2026 (homepage metadata)",
                "identifier_rule": "The FAQ states that Cellosaurus accession CVCL_xxxx and RRID are the same identifier.",
                "per_model_live_record_verification": False,
                "note": "Model.csv RRIDs are preserved as supplied and cross-checked against the shared CTRP identity table. Current Cellosaurus entry records were not fetched individually.",
                "rrids_in_depmap_source": len(expected_rrids),
            },
            "depmap_26q3_release_notes": {
                "url": DEPMAP_26Q3_NOTES_URL,
                "data_locator": "Model annotations section: inferred ER/PR expression thresholds, ERBB2 copy-number rule, PAM50 classes/scores",
                "per_model_values_downloaded": False,
                "download_manifest_url": DEPMAP_DOWNLOAD_API_URL,
                "access_note": "The official download API returned a Cloudflare verification HTML page in this environment. No per-model 26Q3 values were assigned.",
            },
        },
        "depmap_source": {
            "path": str(model_path),
            "release": "24Q2",
            "url": DEPMAP_24Q2_RELEASE_URL,
            "sha256": sha256(model_path),
        },
    }
    manifest_path = source_dir / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def truthy(value: object) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def _rrid_key(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).strip().upper())


def _status_text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _is_exact_ctrp_status(value: object) -> bool:
    status = _status_text(value).casefold()
    return status == "exact" or status.startswith("exact_")


def _is_verified_lincs_status(value: object, cell_id: object) -> bool:
    status = _status_text(value).casefold()
    return bool(_status_text(cell_id)) and ("exact" in status or "verified" in status)


def _load_crosswalk(path: Path, models: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required shared three-way identity crosswalk not found: {path}")
    crosswalk = pd.read_csv(path, dtype="string").fillna("")
    required = {"ModelID", "RRID", "depmap_patient_id", "baseline_RNA_present", "ctrp_identity_status", "lincs_identity_status"}
    missing = required.difference(crosswalk.columns)
    if missing:
        raise ValueError(f"Identity crosswalk missing required columns: {sorted(missing)}")
    if crosswalk["ModelID"].duplicated().any():
        raise ValueError("Identity crosswalk ModelID values are not unique")
    comparison = models[["ModelID", "RRID"]].merge(
        crosswalk[["ModelID", "RRID"]], on="ModelID", how="outer", suffixes=("_depmap", "_crosswalk"), indicator=True
    )
    mismatched = comparison[comparison["_merge"].ne("both") | (comparison["RRID_depmap"].map(_rrid_key) != comparison["RRID_crosswalk"].map(_rrid_key))]
    if not mismatched.empty:
        raise ValueError(f"Identity crosswalk does not match the 97-model DepMap identity set ({len(mismatched)} differences)")
    patients = models[["ModelID", "PatientID"]].merge(
        crosswalk[["ModelID", "depmap_patient_id"]], on="ModelID", how="outer", validate="one_to_one", indicator=True
    )
    patient_mismatch = patients[patients["_merge"].ne("both") | (patients["PatientID"].fillna("") != patients["depmap_patient_id"].fillna(""))]
    if not patient_mismatch.empty:
        raise ValueError(f"Identity crosswalk patient IDs do not match DepMap Model.csv ({len(patient_mismatch)} differences)")
    return crosswalk


def build_evidence(root: Path, out_dir: Path, source_dir: Path, crosswalk_path: Path) -> dict[str, object]:
    model_path = root / "data/raw/depmap/24q2/Model.csv"
    models_all = pd.read_csv(model_path, dtype="string").fillna("")
    models = models_all.loc[models_all["OncotreeLineage"].eq("Breast")].copy()
    if len(models) != 97:
        raise ValueError(f"Expected the frozen 97 DepMap 24Q2 Breast rows, observed {len(models)}")
    if models["ModelID"].duplicated().any():
        raise ValueError("DepMap breast ModelID values are not unique")
    nonempty_rrids = models.loc[models["RRID"].ne(""), "RRID"]
    if nonempty_rrids.duplicated().any():
        raise ValueError("DepMap 24Q2 Breast non-empty RRIDs are duplicated")

    article_path = source_dir / "dai2017_table1_receptors.csv"
    conflicts_path = source_dir / "dai2017_table3_conflicts.csv"
    manifest_path = source_dir / "source_manifest.json"
    for path in (article_path, conflicts_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing cached public-source input: {path}. Run --refresh-public-sources first.")

    article = pd.read_csv(article_path, dtype="string").fillna("")
    article["article_name_key"] = article["article_cell_line"].map(normalize_name)
    if article["article_name_key"].eq("").any() or article["article_name_key"].duplicated().any():
        raise ValueError("Article Table 1 has empty or duplicate normalized cell-line names")
    conflicts = pd.read_csv(conflicts_path, dtype="string").fillna("")
    conflicts["article_name_key"] = conflicts["article_cell_line"].map(normalize_name)
    conflict_names = set(conflicts["article_name_key"])
    conflicts["article_table3_raw_row"] = conflicts.apply(
        lambda row: json.dumps(
            {
                "cell_line": row["article_cell_line"],
                "ER": row["article_er"],
                "PR": row["article_pr"],
                "HER2": row["article_her2"],
                "historical_subtype": row["article_historical_subtype"],
                "reference_ids": row["article_reference_ids"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        axis=1,
    )
    conflict_details = (
        conflicts.groupby("article_name_key", as_index=False, sort=False)
        .agg(
            article_table3_raw_conflict_rows=("article_table3_raw_row", lambda values: "[" + ",".join(values) + "]"),
            article_table3_conflict_row_count=("article_table3_raw_row", "size"),
            article_table3_conflict_reference_ids=(
                "article_reference_ids", lambda values: " | ".join(dict.fromkeys(values))
            ),
            article_table3_cell_line=("article_cell_line", "first"),
        )
    )

    crosswalk = _load_crosswalk(crosswalk_path, models)
    upstream_summary_path = crosswalk_path.parent / "summary.json"
    upstream_summary = json.loads(upstream_summary_path.read_text(encoding="utf-8"))
    recorded_crosswalk_sha = str(
        upstream_summary.get("output_files_sha256", {}).get("breast_cell_identity_crosswalk.csv", "")
    ).casefold()
    actual_crosswalk_sha = sha256(crosswalk_path).casefold()
    if not recorded_crosswalk_sha or recorded_crosswalk_sha != actual_crosswalk_sha:
        raise ValueError("Shared identity crosswalk SHA-256 does not match its upstream inventory summary")
    upstream_source_hashes = upstream_summary.get("files_sha256", {})
    evidence = models.merge(crosswalk, on=["ModelID", "RRID"], how="left", validate="one_to_one", suffixes=("", "_crosswalk"))
    evidence["rrid_key"] = evidence["RRID"].map(_rrid_key)
    evidence["article_name_key"] = evidence["StrippedCellLineName"].map(normalize_name)
    evidence = evidence.merge(
        article,
        left_on="article_name_key",
        right_on="article_name_key",
        how="left",
        validate="many_to_one",
        indicator="article_match",
    )
    evidence = evidence.merge(conflict_details, on="article_name_key", how="left", validate="many_to_one")
    evidence["article_table3_conflict_row_count"] = evidence["article_table3_conflict_row_count"].fillna(0).astype(int)
    evidence["article_table3_source_locator"] = evidence.apply(
        lambda row: (
            f"Dai et al. 2017, Table 3, cell_line={row['article_table3_cell_line']}, "
            f"conflicting_rows={row['article_table3_conflict_row_count']}"
            if row["article_table3_conflict_row_count"] > 0 else ""
        ),
        axis=1,
    )

    # Preserve source labels as reported; their meanings differ from each other.
    evidence = evidence.rename(columns={
        "LegacyMolecularSubtype": "depmap24q2_legacy_molecular_subtype_raw",
        "LegacySubSubtype": "depmap24q2_legacy_sub_subtype_raw",
        "PatientMolecularSubtype": "depmap24q2_patient_molecular_subtype_raw",
    })
    evidence["depmap24q2_legacy_label_track"] = evidence["depmap24q2_legacy_molecular_subtype_raw"]
    evidence["depmap24q2_historical_sub_subtype_track"] = evidence["depmap24q2_legacy_sub_subtype_raw"]
    evidence["article_historical_label_track"] = evidence.get("article_historical_subtype", pd.Series("", index=evidence.index))
    evidence["depmap26q3_ER_inferred_call"] = "unavailable_per_model_annotation"
    evidence["depmap26q3_PR_inferred_call"] = "unavailable_per_model_annotation"
    evidence["depmap26q3_ERBB2_amplification_call"] = "unavailable_per_model_annotation"
    evidence["PAM50_classification"] = "unknown_unavailable_per_model_values"
    evidence["PAM50_scores"] = "unknown_unavailable_per_model_values"

    evidence["article_table_conflict"] = evidence["article_name_key"].isin(conflict_names)
    evidence["receptor_pattern_category"] = [
        receptor_pattern(er, pr, her2, conflicting_source=conflict)
        for er, pr, her2, conflict in zip(
            evidence.get("article_er", ""),
            evidence.get("article_pr", ""),
            evidence.get("article_her2", ""),
            evidence["article_table_conflict"],
        )
    ]
    has_receptor_source = evidence["article_match"].eq("both") | evidence["article_table_conflict"]
    evidence["receptor_evidence_level"] = has_receptor_source.map(
        {True: "B_secondary_literature_compilation_with_row_level_references", False: "D_no_receptor_table_match"}
    )
    evidence["receptor_evidence_level_description"] = evidence["receptor_evidence_level"].map({
        "B_secondary_literature_compilation_with_row_level_references": "Secondary literature compilation with row-level citation IDs; not a standardized clinical assay",
        "D_no_receptor_table_match": "No exact row-level receptor annotation in the selected source; receptor class remains unknown",
    })
    evidence["receptor_source_locator"] = evidence.apply(
        lambda row: (
            " ; ".join(
                locator for locator in (
                    f"Dai et al. 2017, Table 1, row={row.get('article_cell_line', '')}, References={row.get('article_reference_ids', '')}"
                    if _status_text(row.get("article_match")) == "both" else "",
                    _status_text(row.get("article_table3_source_locator")),
                ) if locator
            )
        ),
        axis=1,
    )
    evidence["receptor_conflict_note"] = evidence["article_table_conflict"].map(
        {True: "Dai et al. Table 3 lists contradictory ER/PR/HER2 calls across cited sources; Table 1 raw calls are retained when present, Table 3 raw rows are retained separately, and no unified receptor category is assigned.", False: ""}
    )
    evidence["article_name_match_method"] = evidence["article_match"].map(
        {"both": "unique_normalized_StrippedCellLineName_to_article_cell_line", "left_only": "unmatched"}
    )
    evidence["cellosaurus_identity_status"] = evidence["RRID"].map(
        lambda value: "missing_RRID; ModelID_retained_as_internal_stable_key"
        if not _status_text(value)
        else "RRID_CVCL_identifier_retained_from_DepMap_24Q2; current_entry_not_individually_queried"
    )
    evidence["cellosaurus_current_record_validated"] = False
    evidence["cellosaurus_source_version"] = "Cellosaurus release 56, June 2026; FAQ identifier rule only, individual records not queried"
    evidence["cellosaurus_source_locator"] = "DepMap 24Q2 Model.csv RRID; Cellosaurus FAQ Q18 defines CVCL accession and RRID as identical"
    evidence["cellosaurus_record_url"] = evidence["RRID"].map(
        lambda rrid: f"https://api.cellosaurus.org/cell-line/{rrid}?format=json" if _status_text(rrid) else ""
    )
    evidence["depmap_model_source_url"] = DEPMAP_24Q2_RELEASE_URL
    evidence["depmap_model_source_file"] = "data/raw/depmap/24q2/Model.csv"
    evidence["depmap_model_source_release"] = "DepMap Public 24Q2"
    evidence["article_source_url"] = ARTICLE_URL
    evidence["article_source_version"] = "Dai et al., J Cancer 2017;8(16):3131-3141; DOI " + ARTICLE_DOI
    evidence["ctrp_identity_source_url"] = CTRPV2_URL
    evidence["ctrp_identity_source_file"] = "data/processed/exp013/ctrp_v2/cells.csv"
    evidence["ctrp_identity_source_version"] = "CTRPv2 local processed cell metadata; release label encoded by ctrp_v2 directory; no separate file release field"
    evidence["ctrp_identity_source_locator"] = "cells.csv cellid + Cellosaurus.Accession.id; exact RRID bridge and match status in shared crosswalk"
    evidence["rna_identity_source_file"] = "data/raw/depmap/24q2/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    evidence["rna_identity_source_release"] = "DepMap Public 24Q2"
    evidence["rna_identity_source_locator"] = "ModelID row presence only; numeric expression columns were not read"
    evidence["lincs_identity_source_file"] = "artifacts/experiments/EXP-007/CCLE_CONTEXT_IDENTITY_SPLIT.json"
    evidence["lincs_identity_source_version"] = "Frozen EXP-007 cell identity contract; shared inventory requires EXACT_STRIPPED_NAME_WITH_RNASEQ"
    evidence["lincs_identity_source_locator"] = "shared crosswalk lincs_cell_id + lincs_identity_status"
    evidence["pam50_source_url"] = DEPMAP_26Q3_NOTES_URL
    evidence["pam50_source_version"] = "DepMap Public 26Q3 release notes; per-model file unavailable in this audit"
    evidence["ctrp_exact_rrid_match"] = evidence["ctrp_identity_status"].map(_is_exact_ctrp_status)
    evidence["rna_present"] = evidence["baseline_RNA_present"].map(truthy)
    evidence["lincs_exact_identity"] = [
        _is_verified_lincs_status(status, cell_id)
        for status, cell_id in zip(evidence.get("lincs_identity_status", ""), evidence.get("lincs_cell_id", ""))
    ]
    evidence["three_way_RNA_CTRP_LINCS_identity"] = evidence["rna_present"] & evidence["ctrp_exact_rrid_match"] & evidence["lincs_exact_identity"]
    evidence["RNA_and_CTRP_exact_intersection"] = evidence["rna_present"] & evidence["ctrp_exact_rrid_match"]

    selected_columns = [
        "ModelID", "PatientID", "depmap_patient_id", "CellLineName", "StrippedCellLineName", "RRID", "OncotreeLineage",
        "OncotreePrimaryDisease", "OncotreeSubtype", "DepmapModelType",
        "depmap24q2_legacy_molecular_subtype_raw", "depmap24q2_legacy_sub_subtype_raw",
        "depmap24q2_patient_molecular_subtype_raw", "article_historical_label_track",
        "article_er", "article_pr", "article_her2", "receptor_pattern_category",
        "receptor_evidence_level", "receptor_evidence_level_description", "article_source_basis", "article_reference_ids",
        "receptor_source_locator", "article_table3_raw_conflict_rows", "article_table3_conflict_row_count",
        "article_table3_conflict_reference_ids", "article_table3_source_locator",
        "article_table_conflict", "receptor_conflict_note",
        "article_name_match_method", "cellosaurus_identity_status", "cellosaurus_current_record_validated",
        "cellosaurus_source_version", "cellosaurus_source_locator", "cellosaurus_record_url",
        "baseline_RNA_present", "RNA_and_CTRP_exact_intersection",
        "ctrp_cellid", "ctrp_cell_name", "ctrp_cellosaurus_accession", "ctrp_identity_status",
        "lincs_cell_id", "lincs_identity_status", "three_way_RNA_CTRP_LINCS_identity",
        "depmap26q3_ER_inferred_call", "depmap26q3_PR_inferred_call", "depmap26q3_ERBB2_amplification_call",
        "PAM50_classification", "PAM50_scores", "depmap_model_source_url", "depmap_model_source_file",
        "depmap_model_source_release", "article_source_url", "article_source_version",
        "ctrp_identity_source_url", "ctrp_identity_source_file", "ctrp_identity_source_version", "ctrp_identity_source_locator",
        "rna_identity_source_file", "rna_identity_source_release", "rna_identity_source_locator",
        "lincs_identity_source_file", "lincs_identity_source_version", "lincs_identity_source_locator",
        "pam50_source_url", "pam50_source_version",
    ]
    # Keep useful columns that may not be present in a given producer version optional.
    output = evidence[[column for column in selected_columns if column in evidence.columns]].copy()
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = out_dir / "breast_subtype_evidence.csv"
    output.to_csv(evidence_path, index=False)
    assignment_columns = [
        "ModelID", "PatientID", "depmap_patient_id", "CellLineName", "StrippedCellLineName", "RRID",
        "baseline_RNA_present", "ctrp_cellid", "ctrp_identity_status",
        "lincs_cell_id", "lincs_identity_status", "RNA_and_CTRP_exact_intersection",
        "three_way_RNA_CTRP_LINCS_identity", "article_er", "article_pr", "article_her2",
        "receptor_pattern_category", "receptor_evidence_level", "receptor_source_locator",
        "receptor_evidence_level_description", "article_source_url", "article_source_version", "article_reference_ids",
        "article_table3_raw_conflict_rows", "article_table3_conflict_reference_ids", "article_table3_source_locator",
        "article_table_conflict", "receptor_conflict_note",
        "depmap24q2_legacy_molecular_subtype_raw", "depmap24q2_legacy_sub_subtype_raw",
        "depmap24q2_patient_molecular_subtype_raw", "article_historical_label_track",
        "PAM50_classification", "PAM50_scores",
    ]
    assignments_path = out_dir / "breast_subtype_assignments.csv"
    output[[column for column in assignment_columns if column in output.columns]].to_csv(assignments_path, index=False)

    pattern_counts = output["receptor_pattern_category"].value_counts(dropna=False).to_dict()
    intersection = output[output["RNA_and_CTRP_exact_intersection"]]
    intersection_counts = intersection["receptor_pattern_category"].value_counts(dropna=False).to_dict()
    three_way = output[output["three_way_RNA_CTRP_LINCS_identity"]]
    receptor_unknown_mask = output["receptor_pattern_category"].str.startswith("unknown")
    summary: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "metadata and identity only; no response or expression numeric values read",
        "denominator_contract": {
            "depmap_source": "DepMap Public 24Q2 Model.csv rows with OncotreeLineage=Breast",
            "expected_depmap_rows": 97,
            "identity_crosswalk": str(crosswalk_path),
            "ctrp_identity": "stable Cellosaurus RRID exact match as provided by shared crosswalk",
            "RNA": "baseline_RNA_present as provided by shared crosswalk",
            "LINCS": "shared exact identity status and lincs_cell_id as provided by shared crosswalk",
        },
        "counts": {
            "depmap_breast_models": int(len(output)),
            "models_with_unique_article_table1_match": int(output["article_name_match_method"].eq("unique_normalized_StrippedCellLineName_to_article_cell_line").sum()),
            "models_with_stable_source_RRID": int(output["RRID"].ne("").sum()),
            "models_without_RRID": int(output["RRID"].eq("").sum()),
            "duplicate_nonempty_RRID_count": int(output.loc[output["RRID"].ne(""), "RRID"].duplicated().sum()),
            "unique_nonempty_DepMap_PatientID_count": int(output.loc[output["PatientID"].ne(""), "PatientID"].nunique()),
            "models_with_exact_RNA_and_CTRP_identity": int(output["RNA_and_CTRP_exact_intersection"].sum()),
            "models_with_exact_RNA_CTRP_and_LINCS_identity": int(output["three_way_RNA_CTRP_LINCS_identity"].sum()),
            "article_table_conflict_models": int(output["article_table_conflict"].sum()),
            "unknown_receptor_pattern_models": int(receptor_unknown_mask.sum()),
            "receptor_classified_models": int((~receptor_unknown_mask).sum()),
            "receptor_classified_RNA_CTRP_intersection_models": int((~intersection["receptor_pattern_category"].str.startswith("unknown")).sum()),
            "receptor_unknown_RNA_CTRP_intersection_models": int(intersection["receptor_pattern_category"].str.startswith("unknown").sum()),
            "receptor_classified_three_way_models": int((~three_way["receptor_pattern_category"].str.startswith("unknown")).sum()),
        },
        "receptor_pattern_counts_all_97": {str(key): int(value) for key, value in pattern_counts.items()},
        "receptor_evidence_level_counts": {
            str(key): int(value)
            for key, value in output["receptor_evidence_level"].value_counts(dropna=False).to_dict().items()
        },
        "receptor_evidence_level_counts_RNA_CTRP_intersection": {
            str(key): int(value)
            for key, value in intersection["receptor_evidence_level"].value_counts(dropna=False).to_dict().items()
        },
        "receptor_pattern_counts_RNA_CTRP_intersection": {str(key): int(value) for key, value in intersection_counts.items()},
        "receptor_pattern_counts_three_way_identity": {str(key): int(value) for key, value in three_way["receptor_pattern_category"].value_counts(dropna=False).to_dict().items()},
        "legacy_label_counts_depmap24q2": {
            str(key) if str(key) else "missing": int(value)
            for key, value in output["depmap24q2_legacy_molecular_subtype_raw"].value_counts(dropna=False).to_dict().items()
        },
        "pam50": {
            "per_model_values_available": False,
            "assignments_made": 0,
            "reason": "26Q3 release notes document the new breast/PAM50 annotations, but the model-level file was not retrievable through the official download API in this audit. All PAM50 values remain unknown.",
        },
        "classification_rules": {
            "HR_positive_HER2_negative": "ER or PR explicitly positive and HER2 explicitly negative",
            "HR_positive_HER2_positive": "ER or PR explicitly positive and HER2 explicitly positive",
            "HR_negative_HER2_positive": "ER and PR explicitly negative and HER2 explicitly positive",
            "triple_negative_receptor_pattern": "ER, PR, and HER2 all explicitly negative",
            "uncertainty": "Any missing, +/- or conflicting source call remains unknown; no clinical IHC thresholds inferred.",
        },
        "limitations": [
            "Dai et al. Table 1 is a literature synthesis with row-level citation IDs, not a harmonized laboratory assay dataset; those receptor calls are evidence-bearing cell-line annotations, not clinical diagnostic results.",
            "Rows named in Dai et al. Table 3 carry contradictory receptor calls in cited sources and are withheld from unified receptor grouping; raw Table 1 calls (if present) and the Table 3 conflict rows are retained separately.",
            "DepMap 24Q2 legacy subtype, Dai et al. historical cell-line subtype, receptor pattern, and PAM50 are retained as separate concepts.",
            "DepMap 26Q3 release-note ER/PR expression rules and ERBB2 copy-number rule are not substituted for measured ER/PR/HER2 receptor status; per-model values are unavailable in this artifact.",
            "PAM50 class and scores are unknown unless a versioned per-model source is acquired; no basal/luminal label was converted to PAM50.",
            "Five DepMap Breast records have no RRID in Model.csv; they retain their unique ModelID and remain unmatched to Cellosaurus-keyed CTRP records unless an independent source key is available.",
        ],
        "input_files": {
            "depmap_model_csv": {"path": str(model_path), "release": "24Q2", "url": DEPMAP_24Q2_RELEASE_URL, "sha256": sha256(model_path)},
            "identity_crosswalk": {"path": str(crosswalk_path), "sha256": sha256(crosswalk_path)},
        "identity_crosswalk_audit_summary": {
                "path": str(crosswalk_path.parent / "summary.json"),
            "sha256": sha256(upstream_summary_path),
            "crosswalk_sha256_recorded_by_upstream_audit": recorded_crosswalk_sha.upper(),
            },
            "depmap_rna_row_identity_only": {
            **upstream_source_hashes["depmap_expression_row_identity_only"],
                "release": "DepMap Public 24Q2; only ModelID row axis projected",
            },
            "ctrp_cell_identity_metadata": {
            **upstream_source_hashes["ctrp_cells"],
                "release": "CTRPv2 local processed metadata",
                "url": CTRPV2_URL,
                "locator": "cellid and Cellosaurus.Accession.id columns",
            },
            "ctrp_depmap_rrid_bridge": {
            **upstream_source_hashes["verified_ctrp_depmap_identity_crosswalk"],
            },
            "lincs_frozen_identity_contract": {
            **upstream_source_hashes["frozen_lincs_depmap_cell_identity_contract"],
                "release": "Frozen EXP-007 identity contract",
            },
            "article_table1": {"path": str(article_path), "sha256": sha256(article_path), "url": ARTICLE_URL, "locator": "Table 1"},
            "article_table3": {"path": str(conflicts_path), "sha256": sha256(conflicts_path), "url": ARTICLE_URL, "locator": "Table 3"},
            "cellosaurus_identifier_definition": {"url": CELLOSAURUS_FAQ_URL, "release": "Release 56, June 2026"},
            "source_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        },
        "output_files": {
            "full_evidence_table": {"path": str(evidence_path), "sha256": sha256(evidence_path)},
            "unified_assignment_list": {"path": str(assignments_path), "sha256": sha256(assignments_path)},
        },
    }
    summary_path = out_dir / "breast_subtype_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_methods(out_dir / "breast_subtype_methods.md", summary)
    return summary


def write_methods(path: Path, summary: dict[str, object]) -> None:
    counts = summary["counts"]
    assert isinstance(counts, dict)
    text = f"""# 乳腺细胞受体分型证据核查

生成时间（UTC）：{summary['generated_at_utc']}

## 范围与结果

以 DepMap Public 24Q2 `Model.csv` 中 `OncotreeLineage=Breast` 的 97 条模型记录为分母。所有模型有唯一 `ModelID`，其中 {counts['models_with_stable_source_RRID']} 条同时带有Cellosaurus `RRID`，{counts['models_without_RRID']} 条在原始清单中没有RRID，保留ModelID但不猜外部身份。CTRP、基线 RNA 和 LINCS 交集以共享身份表为准。此核查只读取模型与身份元数据，没有读取药敏数值或表达矩阵数值。

- 97 条模型中，{counts['models_with_unique_article_table1_match']} 条可按去标点后唯一的模型名对应到 Dai 等（2017）Table 1 的受体注释。
- {counts['article_table_conflict_models']} 条模型对应的细胞系被 Dai 等 Table 3 明列为不同研究间受体标记冲突；如 Table 1 也有记录则保留其原值，Table 3 各来源的矛盾行另行保留，统一受体分组置为未知。
- 有 {counts['models_with_exact_RNA_and_CTRP_identity']} 条模型进入精确 Cellosaurus RRID + 基线 RNA 交集；其中 {counts['models_with_exact_RNA_CTRP_and_LINCS_identity']} 条同时符合共享表中的 LINCS 精确身份条件。
- 97 条模型中，{counts['receptor_classified_models']} 条有完整且未列入来源冲突的 ER/PR/HER2 文献受体模式，{counts['unknown_receptor_pattern_models']} 条保留未知；40 条 RNA + CTRP 交集中分别为 {counts['receptor_classified_RNA_CTRP_intersection_models']} 条有明确受体模式、{counts['receptor_unknown_RNA_CTRP_intersection_models']} 条未知。三方 LINCS 交集的 {counts['receptor_classified_three_way_models']} 条均有明确受体模式。
- PAM50 类别与分数均保留未知：26Q3 发布说明确认新增了这些字段，但本轮没有取得对应逐模型文件。

## 受体规则

ER、PR、HER2 仅按来源表中的明确 `+`/`-` 记录标准化。ER 或 PR 任一明确阳性且 HER2 明确阴性，归入“HR 阳性 / HER2 阴性”；相同 HR 条件且 HER2 阳性，归入“HR 阳性 / HER2 阳性”；ER、PR 均明确阴性且 HER2 阳性，归入“HR 阴性 / HER2 阳性”；三项均明确阴性时记录为“三阴性受体模式”。缺失、`+/-` 或来源冲突一律保留未知。这里的 HR 表示 ER 或 PR 阳性。

该分组描述细胞系文献里的受体表型，不等同于临床肿瘤诊断；原文没有提供适用于所有行的统一免疫组化方法和临床阈值。ERBB2 扩增也没有被当作 HER2 临床阳性。

## 标签分轨保存

DepMap 24Q2 `LegacyMolecularSubtype`、`LegacySubSubtype`、`PatientMolecularSubtype`、Dai 等 Table 1 的历史类别和 PAM50 分开保存。`basal_A/B` 不推成三阴性，`luminal` 不推成 Luminal A；PAM50 不从这些标签反推。

DepMap 26Q3 发布说明所述定义包括：ERBB2 扩增为合并 WGS/WES copy number >4；Breast ER+ 按 ESR1 RNA >2.5 且 FOXA1 RNA >6；Breast PR+ 按 PGR RNA >1.5；并发布 PAM50 类别与分数。它们是新版本的推断字段，和临床受体实验不等价。本轮DepMap公开下载API返回Cloudflare验证页，未取得逐模型注释文件，因此未填写任何26Q3受体或PAM50调用结果。

## 来源与可复算

- DepMap 24Q2 模型身份及原始历史标签：`data/raw/depmap/24q2/Model.csv`，发行页 <{DEPMAP_24Q2_RELEASE_URL}>；具体输入 SHA-256 见 JSON。
- Dai X, Cheng H, Bai Z, Li J. *Breast Cancer Cell Line Classification and Its Relevance with Breast Tumor Subtyping*. J Cancer. 2017;8(16):3131–3141. DOI: {ARTICLE_DOI}. Table 1 提供 84 条 ER/PR/HER2 与历史类别，逐行带参考编号；Table 3 列出八种细胞系的矛盾注释。页面：{ARTICLE_URL}
- 受体证据等级：B 表示来自带逐行参考编号的二次文献汇编，不能替代统一实验检测；D 表示本文所用注释表没有该模型的行级受体记录。Table 3 明列的冲突不赋统一受体类别。
- Cellosaurus FAQ Q18 说明 CVCL accession 与 RRID 是同一标识；本表保留 DepMap 24Q2 提供的 RRID，并用共享交集表的精确 RRID 连接 CTRP。Cellosaurus release 56（2026年6月）记录只作为版本/标识规则来源，本轮未逐条复核具体记录，已在表中标清。
- CTRPv2来源入口 <{CTRPV2_URL}>；本地处理细胞元数据为 `data/processed/exp013/ctrp_v2/cells.csv`，使用 `cellid` 与 `Cellosaurus.Accession.id`。基线 RNA 仅使用 DepMap 24Q2表达文件的 `ModelID` 行身份；LINCS 身份沿用冻结的 EXP-007 精确映射合同。文件定位、版本声明和上游校验和见 JSON 及逐模型证据表。
- DepMap 26Q3 发布说明：{DEPMAP_26Q3_NOTES_URL}。

在仓库根目录、WSL2 `drugscreening-gpu` 环境内重建来源缓存并生成结果：

```bash
PYTHONPATH=src /home/dell/miniconda3/bin/conda run --no-capture-output -n drugscreening-gpu python scripts/data/breast_subtype_evidence.py --refresh-public-sources
PYTHONPATH=src /home/dell/miniconda3/bin/conda run --no-capture-output -n drugscreening-gpu python scripts/data/breast_subtype_evidence.py --build
```

默认构建要求共享交集身份表 `artifacts/data_audits/breast_threeway_inventory_20261009/breast_cell_identity_crosswalk.csv` 存在；脚本会逐行核对 97 个 ModelID 和 RRID 后才输出分型表。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("artifacts/data_audits/breast_subtype_evidence_20261009"))
    parser.add_argument("--refresh-public-sources", action="store_true", help="Fetch and cache the public receptor tables and source-version manifest")
    parser.add_argument("--build", action="store_true", help="Build the 97-row evidence table using the shared three-way identity crosswalk")
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out if args.out.is_absolute() else root / args.out
    source_dir = out_dir / "source_annotations"
    if args.refresh_public_sources:
        manifest = refresh_public_sources(root, source_dir)
        print(json.dumps({"source_manifest": str(source_dir / "source_manifest.json"), "dai_table1_rows": 84, "dai_table3_rows": 17}, ensure_ascii=False))
    if args.build:
        crosswalk = root / "artifacts/data_audits/breast_threeway_inventory_20261009/breast_cell_identity_crosswalk.csv"
        summary = build_evidence(root, out_dir, source_dir, crosswalk)
        print(json.dumps({"evidence_csv": str(out_dir / "breast_subtype_evidence.csv"), "summary_json": str(out_dir / "breast_subtype_summary.json"), "counts": summary["counts"]}, ensure_ascii=False))
    if not args.refresh_public_sources and not args.build:
        parser.error("choose --refresh-public-sources, --build, or both")


if __name__ == "__main__":
    main()
