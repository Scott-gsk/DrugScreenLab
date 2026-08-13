"""Select a bounded, response-blind genetic cohort for the E2 FAST probe."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def select_cohort(
    instances: pd.DataFrame,
    local_genes: set[str],
    *,
    max_genes: int,
    min_rows: int,
    min_cells: int,
) -> pd.DataFrame:
    required = {"cell_id", "pert_id", "pert_iname", "pert_type", "pert_time", "pert_time_unit"}
    missing = sorted(required.difference(instances.columns))
    if missing:
        raise ValueError(f"inst_info missing columns: {missing}")
    selected = instances.loc[
        instances["pert_type"].eq("trt_sh")
        & instances["pert_time"].eq(96)
        & instances["pert_time_unit"].eq("h")
    ].copy()
    selected["gene_symbol"] = selected["pert_iname"].astype(str).str.strip().str.upper()
    selected = selected.loc[selected["gene_symbol"].isin(local_genes)]
    summary = (
        selected.groupby("gene_symbol", as_index=False)
        .agg(
            rows=("pert_id", "size"),
            cells=("cell_id", "nunique"),
            perturbagen_ids=("pert_id", "nunique"),
        )
    )
    summary = summary.loc[
        (summary["rows"] >= min_rows) & (summary["cells"] >= min_cells)
    ].sort_values(
        ["rows", "cells", "perturbagen_ids", "gene_symbol"],
        ascending=[False, False, False, True],
    )
    result = summary.head(max_genes).reset_index(drop=True)
    if result.empty:
        raise ValueError("no genetic genes satisfy the cohort criteria")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inst-info", type=Path, required=True)
    parser.add_argument("--unipert-source", type=Path, required=True)
    parser.add_argument("--output-genes", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--max-genes", type=int, default=256)
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--min-cells", type=int, default=2)
    args = parser.parse_args()
    if args.max_genes < 1 or args.min_rows < 1 or args.min_cells < 1:
        raise ValueError("cohort limits must be positive")

    instances = pd.read_csv(args.inst_info, sep="\t", low_memory=False)
    reference_targets = pd.read_csv(args.unipert_source / "data" / "ref_targets.csv", low_memory=False)
    local_genes = set(reference_targets["Approved symbol"].astype(str).str.strip().str.upper())
    cohort = select_cohort(
        instances,
        local_genes,
        max_genes=args.max_genes,
        min_rows=args.min_rows,
        min_cells=args.min_cells,
    )
    args.output_genes.parent.mkdir(parents=True, exist_ok=True)
    args.output_genes.write_text(
        "\n".join(cohort["gene_symbol"].tolist()) + "\n", encoding="utf-8"
    )
    audit = {
        "format": "e2_genetic_cohort_selection_v1",
        "status": "COHORT_FROZEN_RESPONSE_BLIND",
        "source": str(args.inst_info),
        "source_sha256": digest(args.inst_info),
        "condition": {"pert_type": "trt_sh", "time_h": 96.0},
        "control_policy": "same_rna_plate_same_cell_same_time_ctl_vector_preferred_ctl_untrt_fallback",
        "selection": {
            "minimum_rows_per_gene": args.min_rows,
            "minimum_cells_per_gene": args.min_cells,
            "maximum_genes": args.max_genes,
            "local_unipert_reference_only": True,
        },
        "candidate_genes": int(len(cohort)),
        "candidate_rows": int(cohort["rows"].sum()),
        "candidate_cells": int(
            instances.loc[
                instances["pert_iname"].astype(str).str.strip().str.upper().isin(
                    set(cohort["gene_symbol"])
                )
                & instances["pert_type"].eq("trt_sh")
                & instances["pert_time"].eq(96)
                & instances["pert_time_unit"].eq("h"),
                "cell_id",
            ].nunique()
        ),
        "response_values_read": False,
        "gene_list": str(args.output_genes),
        "gene_list_sha256": None,
        "genes": cohort.to_dict(orient="records"),
    }
    audit["gene_list_sha256"] = digest(args.output_genes)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
