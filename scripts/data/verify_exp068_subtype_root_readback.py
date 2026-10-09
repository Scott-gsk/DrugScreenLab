#!/usr/bin/env python3
"""Independently verify the EXP-068 subtype metadata artifact and identities.

This verifier reads only cell/model identity, published receptor annotations,
and output metadata. It never opens CTRP response rows or expression matrices
except the first DepMap expression CSV identity column.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tsv(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(8192)
    delimiter = csv.Sniffer().sniff(sample, delimiters="\t,").delimiter
    return pd.read_csv(path, sep=delimiter, dtype=str, keep_default_na=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--attempt-dir",
        type=Path,
        default=Path("artifacts/experiments/EXP-068/subtype_evidence_v1/attempt_005"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    attempt = (root / args.attempt_dir).resolve()
    output = (root / args.output).resolve() if args.output else attempt / "root_independent_readback.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite root receipt: {output}")

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    summary_path = attempt / "summary.json"
    run_manifest_path = attempt / "run_manifest.json"
    source_manifest_path = attempt / "source_manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))

    file_checks: dict[str, Any] = {}
    for name, expected in summary["outputs"].items():
        path = attempt / name
        actual = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if actual != expected:
            raise ValueError(f"summary output hash/size mismatch: {name}: {actual} != {expected}")
        manifest_expected = run_manifest.get("outputs", {}).get(name)
        if manifest_expected is not None and manifest_expected != actual:
            raise ValueError(f"run-manifest output hash/size mismatch: {name}")
        file_checks[name] = actual

    input_checks: dict[str, Any] = {}
    intentionally_not_opened: list[str] = []
    independently_verified_input_paths = {
        "data/raw/depmap/24q2/Model.csv",
        "data/processed/exp013/ctrp_v2/cells.csv",
        "artifacts/experiments/EXP-050/20261006_breast_subtype_measured_transfer_v1/lincs_breast_cell_model_identity.tsv",
        "artifacts/experiments/EXP-050/20261006_breast_subtype_measured_transfer_v1/root_independent_readback_receipt.json",
        "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_cell_info.txt.gz",
        "data/artifacts/resource_acquisition_20260916/ctrp_parse/ctrp_v2_materialization_registry.json",
        "artifacts/data_audits/breast_subtype_evidence_20261009/source_annotations/dai2017_table1_receptors.csv",
        "artifacts/data_audits/breast_subtype_evidence_20261009/source_annotations/dai2017_table3_conflicts.csv",
        "artifacts/data_audits/breast_subtype_evidence_20261009/source_annotations/dai2017_article_page.html",
        "artifacts/data_audits/breast_subtype_evidence_20261009/source_annotations/cellosaurus_rrid_validation.tsv",
        "artifacts/data_audits/breast_subtype_evidence_20261009/source_annotations/cellosaurus_release_info.json",
        "artifacts/data_audits/breast_subtype_evidence_20261009/source_annotations/source_manifest.json",
    }
    post_run_input_mutations: list[dict[str, Any]] = []
    for entry in source_manifest.get("inputs", []):
        if entry["path"] not in independently_verified_input_paths:
            intentionally_not_opened.append(entry["path"])
            continue
        path = root / entry["path"]
        actual = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        if actual != {"bytes": entry["bytes"], "sha256": entry["sha256"]}:
            if entry["path"].endswith("/source_manifest.json"):
                post_run_input_mutations.append(
                    {
                        "path": entry["path"],
                        "attempt_bound_bytes": entry["bytes"],
                        "attempt_bound_sha256": entry["sha256"],
                        "current_bytes": actual["bytes"],
                        "current_sha256": actual["sha256"],
                        "handling": "preserved as a post-run provenance-file mutation; source data hashes are checked separately",
                    }
                )
            else:
                raise ValueError(f"source input hash/size mismatch: {entry['path']}")
        else:
            input_checks[entry["path"]] = actual

    model_path = root / "data/raw/depmap/24q2/Model.csv"
    model_columns = [
        "ModelID", "PatientID", "CellLineName", "RRID", "OncotreeLineage",
        "OncotreePrimaryDisease", "LegacyMolecularSubtype", "LegacySubSubtype",
        "PatientMolecularSubtype",
    ]
    models = pd.read_csv(model_path, usecols=model_columns, dtype=str, keep_default_na=False)
    breast_mask = (
        models["OncotreeLineage"].str.contains("breast", case=False, regex=False)
        | models["OncotreePrimaryDisease"].str.contains("breast", case=False, regex=False)
    )
    breast = models.loc[breast_mask].copy()
    if breast.ModelID.duplicated().any() or len(breast) != 97:
        raise ValueError(f"DepMap breast metadata denominator mismatch: {len(breast)}")
    breast_ids = set(breast.ModelID)

    expression_path = root / "data/raw/depmap/24q2/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    expression_ids = pd.read_csv(expression_path, usecols=[0], dtype=str, keep_default_na=False).iloc[:, 0]
    if expression_ids.duplicated().any():
        raise ValueError("DepMap expression identity column contains duplicate ModelID")
    expression_id_set = set(expression_ids)

    ctrp_map_path = root / "data/artifacts/resource_acquisition_20260916/ctrp_parse/ctrp_depmap_identity_map.csv"
    ctrp_map = pd.read_csv(
        ctrp_map_path,
        usecols=["ModelID", "match_method", "has_depmap_expression"],
        dtype=str,
        keep_default_na=False,
    )
    ctrp_exact = ctrp_map.loc[
        ctrp_map.ModelID.isin(breast_ids)
        & ctrp_map.match_method.eq("exact_unique_cellosaurus_rrid")
        & ctrp_map.has_depmap_expression.eq("True"),
        "ModelID",
    ]
    ctrp_exact_ids = set(ctrp_exact)
    if len(ctrp_exact_ids) != 40:
        raise ValueError(f"Exact CTRP-RRID breast identity count mismatch: {len(ctrp_exact_ids)}")

    lincs_path = root / "artifacts/experiments/EXP-050/20261006_breast_subtype_measured_transfer_v1/lincs_breast_cell_model_identity.tsv"
    lincs = load_tsv(lincs_path)
    lincs = lincs.loc[
        lincs.dataset.eq("GSE92742")
        & lincs.identity_status.eq("base_cell_id mapped to unique DepMap ModelID")
        & lincs.DepMap_ModelID.isin(breast_ids)
    ]
    if lincs.LINCS_base_cell_id.nunique() != 5 or lincs.DepMap_ModelID.nunique() != 5:
        raise ValueError("Approved LINCS breast crosswalk denominator mismatch")
    lincs_ids = set(lincs.DepMap_ModelID)

    intersection = load_tsv(attempt / "three_way_breast_model_intersection.tsv")
    required = {
        "DepMap_ModelID", "DepMap_RRID", "DepMap_PatientID", "CTRP_cellid",
        "CTRP_Cellosaurus_Accession", "LINCS_base_cell_id", "three_way_status",
    }
    if required - set(intersection.columns):
        raise ValueError(f"intersection columns missing: {sorted(required - set(intersection.columns))}")
    if intersection.DepMap_ModelID.duplicated().any() or intersection.DepMap_RRID.duplicated().any():
        raise ValueError("three-way intersection has duplicate model/RRID identities")
    if intersection.DepMap_PatientID.duplicated().any():
        raise ValueError("three-way intersection has non-independent DepMap PatientID rows")
    if len(intersection) != 5 or set(intersection.DepMap_ModelID) != (breast_ids & expression_id_set & ctrp_exact_ids & lincs_ids):
        raise ValueError("three-way model intersection does not equal independently recomputed exact identity join")
    if not intersection.three_way_status.eq("three_way_exact_identity_match").all():
        raise ValueError("three-way identity status is not exact for every row")
    if not intersection.DepMap_RRID.eq(intersection.CTRP_Cellosaurus_Accession).all():
        raise ValueError("DepMap and CTRP RRIDs differ in the three-way rows")

    normalized = load_tsv(attempt / "normalized_receptor_groups.tsv")
    if len(normalized) != 5 or normalized.DepMap_ModelID.duplicated().any():
        raise ValueError("normalized receptor table denominator/key mismatch")
    if not normalized.PAM50_status.eq("not_present_in_audited_metadata").all() or normalized.PAM50_subtype.ne("").any():
        raise ValueError("PAM50 missing values were filled or inferred")
    if normalized.receptor_defined_group.value_counts().to_dict() != {
        "triple_negative_receptor_phenotype": 3,
        "HR_positive_HER2_negative": 1,
        "ER_negative_PR_negative_HER2_positive": 1,
    }:
        raise ValueError("receptor-defined group counts mismatch")
    mc = normalized.loc[normalized.DepMap_ModelID.eq("ACH-000019")].iloc[0]
    if mc.PR_normalized != "conflict" or "Dai_2017_Table1=positive" not in mc.PR_conflict_detail:
        raise ValueError("MCF7 PR conflict not preserved")
    if "Guttler_2021_WB_Table1=negative" not in mc.PR_conflict_detail:
        raise ValueError("MCF7 PR source disagreement not preserved")
    alias = load_tsv(attempt / "source_alias_mapping.tsv")
    hs = alias.loc[alias.source_cell_name.eq("HS598T")].iloc[0]
    if hs.RRID != "" or hs.mapping_status != "unresolved_name_not_mapped_to_Hs_578T":
        raise ValueError("HS598T was assigned an unverified cell-line identity")

    if not summary["outputs"]:
        raise ValueError("summary has no output hashes")
    status = {
        "status": (
            "PASS_ROOT_INDEPENDENT_SUBTYPE_METADATA_READBACK_WITH_POSTRUN_PROVENANCE_MUTATION"
            if post_run_input_mutations
            else "PASS_ROOT_INDEPENDENT_SUBTYPE_METADATA_READBACK"
        ),
        "experiment": "EXP-068",
        "attempt_dir": str(attempt.relative_to(root)),
        "cohort_counts": {
            "DepMap_breast_models": len(breast),
            "breast_models_with_exact_CTRP_RRID_and_expression_identity": len(ctrp_exact_ids),
            "approved_LINCS_breast_base_cells": int(lincs.LINCS_base_cell_id.nunique()),
            "exact_three_way_models": len(intersection),
            "unique_DepMap_PatientID": int(intersection.DepMap_PatientID.nunique()),
            "receptor_defined_groups": normalized.receptor_defined_group.value_counts().to_dict(),
            "PAM50_calls_available": 0,
            "PAM50_missing": len(normalized),
        },
        "checks": {
            "file_hashes_sizes_match_summary_and_run_manifest": True,
            "all_verified_source_data_files_match_attempt_manifest": True,
            "source_provenance_manifest_matches_attempt": not post_run_input_mutations,
            "numeric_or_nonprojected_raw_files_not_opened_or_rehashed": intentionally_not_opened,
            "post_run_provenance_file_mutations": post_run_input_mutations,
            "DepMap_breast_metadata_rows_recomputed": True,
            "expression_identity_only_breast_overlap_recomputed": int((breast_ids & expression_id_set).__len__()),
            "CTRP_exact_identity_count_recomputed": True,
            "LINCS_exact_identity_count_recomputed": True,
            "three_way_exact_join_recomputed": True,
            "independent_PatientID_denominator": True,
            "receptor_source_conflict_retained": True,
            "unverified_HS598T_identity_not_assigned": True,
            "PAM50_left_missing_not_inferred": True,
            "forbidden_numeric_data_opened_by_verifier": False,
            "outer_locked_PDO_PANACEA_CDS_B_values_opened": False,
        },
        "verified_output_files": file_checks,
        "verified_input_files": input_checks,
        "verifier_runtime": {
            "wall_seconds": time.perf_counter() - wall_start,
            "cpu_process_seconds": time.process_time() - cpu_start,
            "gpu_hours": 0,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
