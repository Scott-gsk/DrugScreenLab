#!/usr/bin/env python3
"""Audit the fixed IMPROVE CSA package without training or evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile


STUDIES = ["CCLE", "gCSI", "GDSCv1", "GDSCv2", "CTRPv2"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_lines(zf: ZipFile, member: str):
    with zf.open(member) as raw:
        yield from io.TextIOWrapper(raw, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "raw/resource_acquisition_20260916/improve_chempert/improve_csa_data_zenodo_15258883.zip"
    out = root / "artifacts/resource_acquisition_20260916/improve_contract"
    out.mkdir(parents=True, exist_ok=True)

    base = "csa_data/raw_data/"
    response_member = base + "y_data/response.tsv"
    split_base = base + "splits/"
    with ZipFile(source) as zf:
        names = set(zf.namelist())
        reader = csv.DictReader(text_lines(zf, response_member), delimiter="\t")
        source_counts: Counter[str] = Counter()
        source_cells: dict[str, set[str]] = {s: set() for s in STUDIES}
        source_drugs: dict[str, set[str]] = {s: set() for s in STUDIES}
        endpoint_nonmissing: dict[str, Counter[str]] = {s: Counter() for s in STUDIES}
        response_fields = reader.fieldnames or []
        response_rows = 0
        response_identity: list[tuple[str, str, str]] = []
        for row in reader:
            response_rows += 1
            study = row["source"]
            response_identity.append((study, row["improve_sample_id"], row["improve_chem_id"]))
            source_counts[study] += 1
            source_cells.setdefault(study, set()).add(row["improve_sample_id"])
            source_drugs.setdefault(study, set()).add(row["improve_chem_id"])
            for endpoint in ["auc", "ic50", "ec50", "aac1", "auc1", "dss1"]:
                if row.get(endpoint, "") != "":
                    endpoint_nonmissing.setdefault(study, Counter())[endpoint] += 1

        split_summary = {}
        for study in STUDIES:
            all_member = f"{split_base}{study}_all.txt"
            all_ids = {line.strip() for line in text_lines(zf, all_member) if line.strip()}
            per_split = []
            for split in range(10):
                parts = {}
                for role in ["train", "val", "test"]:
                    member = f"{split_base}{study}_split_{split}_{role}.txt"
                    ids = {line.strip() for line in text_lines(zf, member) if line.strip()}
                    parts[role] = ids
                union = parts["train"] | parts["val"] | parts["test"]
                overlap = (
                    (parts["train"] & parts["val"])
                    | (parts["train"] & parts["test"])
                    | (parts["val"] & parts["test"])
                )
                role_cells = {
                    role: {response_identity[int(i)][1] for i in ids}
                    for role, ids in parts.items()
                }
                role_drugs = {
                    role: {response_identity[int(i)][2] for i in ids}
                    for role, ids in parts.items()
                }
                cell_overlap = (
                    (role_cells["train"] & role_cells["val"])
                    | (role_cells["train"] & role_cells["test"])
                    | (role_cells["val"] & role_cells["test"])
                )
                drug_overlap = (
                    (role_drugs["train"] & role_drugs["val"])
                    | (role_drugs["train"] & role_drugs["test"])
                    | (role_drugs["val"] & role_drugs["test"])
                )
                per_split.append(
                    {
                        "split": split,
                        "train": len(parts["train"]),
                        "val": len(parts["val"]),
                        "test": len(parts["test"]),
                        "pairwise_overlap": len(overlap),
                        "union_equals_all": union == all_ids,
                        "cells": {role: len(values) for role, values in role_cells.items()},
                        "cell_identity_overlap": len(cell_overlap),
                        "drugs": {role: len(values) for role, values in role_drugs.items()},
                        "drug_identity_overlap": len(drug_overlap),
                    }
                )
            split_summary[study] = {"all_rows": len(all_ids), "splits": per_split}

        x_members = {}
        for member in [
            base + "x_data/cancer_gene_expression.tsv",
            base + "x_data/drug_SMILES.tsv",
            base + "x_data/drug_info.tsv",
        ]:
            first = next(text_lines(zf, member)).rstrip("\n\r").split("\t")
            x_members[member.removeprefix(base)] = {
                "bytes": zf.getinfo(member).file_size,
                "columns": len(first),
                "first_columns": first[:12],
            }

    report = {
        "audit_version": 1,
        "purpose": "fixed_benchmark_contract_only_no_training_or_evaluation",
        "source": {
            "path": str(source.relative_to(root)),
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
        },
        "response": {
            "rows": response_rows,
            "columns": response_fields,
            "per_study": {
                s: {
                    "rows": source_counts[s],
                    "samples": len(source_cells[s]),
                    "drugs": len(source_drugs[s]),
                    "nonmissing_endpoints": dict(endpoint_nonmissing[s]),
                }
                for s in STUDIES
            },
            "endpoint_contract": {
                "official_csa_default": "auc",
                "evidence": "IMPROVE v0.1.0 workflows/csa/parsl/csa_params.ini and model example parameter files set y_col_name = auc",
                "transformation": "raw auc regression target; no project-side reversal selected by this audit",
            },
        },
        "splits": split_summary,
        "feature_files": x_members,
        "duplicate_boundary": {
            "existing_project_sources": ["CCLE", "GDSCv2", "CTRPv2"],
            "new_study_names_in_fixed_package": ["gCSI", "GDSCv1"],
            "rule": "Keep the package as an immutable benchmark version; do not merge its duplicate studies into current project source tables silently.",
        },
        "identity_isolation": {
            "result": "FAILED_FOR_FORMAL_DRUGSCREENLAB_EVALUATION",
            "interpretation": "Rows are disjoint, but nearly all cell identities and all drug identities recur across train, validation, and test. These are mixed response-row splits, not cold-cell or cold-drug splits.",
            "permitted_role": "official IMPROVE benchmark reproduction with explicit leakage boundary",
            "not_permitted_role": "formal strict-unseen cell-line or drug generalization claim",
        },
        "admission": "FIXED_BENCHMARK_CONTRACT_AUDITED; IDENTITY_ISOLATION_REQUIRES_REVIEW_BEFORE_FORMAL_PROJECT_EVALUATION",
    }
    path = out / "improve_csa_contract_audit.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
