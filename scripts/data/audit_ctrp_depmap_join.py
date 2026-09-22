#!/usr/bin/env python3
"""Audit an identity-only CTRPv2 to DepMap 24Q2 expression join.

This script never edits source files and never selects a response threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


def norm_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    root = args.root.resolve()
    ctrp = root / "processed/exp013/ctrp_v2"
    depmap = root / "raw/depmap/24q2"
    out = root / "artifacts/resource_acquisition_20260916/ctrp_parse"
    out.mkdir(parents=True, exist_ok=True)

    cells_path = ctrp / "cells.csv"
    drugs_path = ctrp / "drugs.csv"
    responses_path = ctrp / "responses.csv"
    model_path = depmap / "Model.csv"
    expression_path = depmap / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"

    cells = pd.read_csv(cells_path, dtype="string")
    models = pd.read_csv(
        model_path,
        usecols=["ModelID", "CellLineName", "StrippedCellLineName", "RRID", "CCLEName"],
        dtype="string",
    )
    expression_ids = set(
        pd.read_csv(expression_path, usecols=[0], dtype="string").iloc[:, 0].dropna()
    )

    cells["rrid_key"] = cells["Cellosaurus.Accession.id"].map(norm_text)
    models["rrid_key"] = models["RRID"].map(norm_text)
    rrid_counts = models.loc[models.rrid_key.ne(""), "rrid_key"].value_counts()
    unique_rrid = models[models.rrid_key.map(rrid_counts).eq(1)].copy()
    rrid_lookup = unique_rrid.set_index("rrid_key")["ModelID"].to_dict()

    cells["ModelID"] = cells["rrid_key"].map(rrid_lookup).astype("string")
    cells["match_method"] = cells["ModelID"].notna().map(
        {True: "exact_unique_cellosaurus_rrid", False: "unmatched"}
    )

    # Unique normalized-name matches are retained only as candidates, not verified joins.
    model_name_pairs = []
    for col in ["CellLineName", "StrippedCellLineName", "CCLEName"]:
        tmp = models[["ModelID", col]].dropna().copy()
        tmp["name_key"] = tmp[col].map(norm_text)
        model_name_pairs.append(tmp[["ModelID", "name_key"]])
    name_table = pd.concat(model_name_pairs).drop_duplicates()
    name_counts = name_table.loc[name_table.name_key.ne(""), "name_key"].value_counts()
    name_lookup = (
        name_table[name_table.name_key.map(name_counts).eq(1)]
        .drop_duplicates("name_key")
        .set_index("name_key")["ModelID"]
        .to_dict()
    )
    cells["name_key"] = cells["ccl_name"].map(norm_text)
    unmatched = cells["ModelID"].isna()
    cells.loc[unmatched, "candidate_ModelID"] = cells.loc[unmatched, "name_key"].map(name_lookup)

    cells["has_depmap_expression"] = cells["ModelID"].isin(expression_ids)
    mapping_columns = [
        "cellid",
        "ccl_name",
        "Cellosaurus.Accession.id",
        "ModelID",
        "match_method",
        "has_depmap_expression",
        "candidate_ModelID",
    ]
    mapping = cells[mapping_columns].copy()
    mapping_path = out / "ctrp_depmap_identity_map.csv"
    mapping.to_csv(mapping_path, index=False)

    verified_cells = set(mapping.loc[mapping.has_depmap_expression, "cellid"].dropna())
    drugs = pd.read_csv(drugs_path, dtype="string")
    valid_source_smiles_drugs = set(drugs.loc[drugs.cpd_smiles.notna() & drugs.cpd_smiles.ne(""), "drugid"])

    response_rows = 0
    finite_aac_rows = 0
    verified_expression_rows = 0
    verified_expression_smiles_rows = 0
    verified_response_cells: set[str] = set()
    verified_response_drugs: set[str] = set()
    for chunk in pd.read_csv(
        responses_path,
        usecols=["cellid", "drugid", "aac_published"],
        dtype={"cellid": "string", "drugid": "string", "aac_published": "float64"},
        chunksize=100_000,
    ):
        response_rows += len(chunk)
        finite = chunk.aac_published.notna()
        finite_aac_rows += int(finite.sum())
        matched = chunk.cellid.isin(verified_cells)
        verified_expression_rows += int(matched.sum())
        joint = matched & chunk.drugid.isin(valid_source_smiles_drugs)
        verified_expression_smiles_rows += int(joint.sum())
        verified_response_cells.update(chunk.loc[joint, "cellid"].dropna().astype(str))
        verified_response_drugs.update(chunk.loc[joint, "drugid"].dropna().astype(str))

    report = {
        "audit_version": 1,
        "purpose": "identity_and_input_availability_only_no_training_or_evaluation",
        "identity_contract": {
            "verified_key": "CTRPv2 Cellosaurus.Accession.id == DepMap Model.RRID after punctuation-insensitive uppercase normalization",
            "name_matches": "reported as candidates only and excluded from verified counts",
        },
        "source_counts": {
            "ctrp_cells": int(len(cells)),
            "depmap_models": int(len(models)),
            "depmap_expression_rows": int(len(expression_ids)),
            "ctrp_drugs": int(len(drugs)),
            "ctrp_response_rows": int(response_rows),
        },
        "join_counts": {
            "exact_unique_rrid_matches": int(mapping.ModelID.notna().sum()),
            "exact_matches_with_expression": int(mapping.has_depmap_expression.sum()),
            "unmatched_cells": int(mapping.ModelID.isna().sum()),
            "unmatched_with_unique_name_candidate": int(mapping.candidate_ModelID.notna().sum()),
            "response_rows_with_verified_expression": int(verified_expression_rows),
            "response_rows_with_verified_expression_and_source_smiles": int(verified_expression_smiles_rows),
            "cells_with_verified_expression_and_response": len(verified_response_cells),
            "drugs_with_source_smiles_and_verified_expression_response": len(verified_response_drugs),
            "finite_aac_published_rows_total": int(finite_aac_rows),
        },
        "files": {
            "identity_map": {
                "path": str(mapping_path.relative_to(root)),
                "sha256": sha256(mapping_path),
            },
            "depmap_expression": {
                "path": str(expression_path.relative_to(root)),
                "not_duplicated": True,
            },
        },
        "admission": "CANDIDATE_CONTRACT_READY_PENDING_ENDPOINT_DRUG_ID_AND_SPLIT_APPROVAL",
        "limitations": [
            "No name-only candidate is counted as a verified expression join.",
            "This audit does not choose a response endpoint or threshold.",
            "This audit does not create train, validation, or test splits.",
        ],
    }
    report_path = out / "ctrp_depmap_join_audit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
