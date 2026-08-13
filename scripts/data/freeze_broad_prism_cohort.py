"""Freeze Broad PRISM identity/context cohort without reading response values."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

import pandas as pd


BASE_ID = re.compile(r"^(BRD-[A-Z]\d{8})")


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--treatment-info", type=Path, required=True)
    parser.add_argument("--cell-line-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-candidates", type=int, default=1)
    args = parser.parse_args()
    if args.min_candidates < 1:
        raise ValueError("--min-candidates must be positive")

    bridge = pd.read_csv(args.bridge, low_memory=False)
    treatment = pd.read_csv(args.treatment_info, low_memory=False)
    lines = pd.read_csv(args.cell_line_info, low_memory=False)
    required_bridge = {"prism_broad_id", "prism_broad_id_base", "match_status", "match_method"}
    if not required_bridge.issubset(bridge.columns):
        raise ValueError("bridge lacks required identity columns")
    formal = bridge.loc[bridge["match_status"].eq("MATCHED_IDENTITY")].copy()
    formal = formal.drop_duplicates("prism_broad_id")
    treatment["prism_broad_id_base"] = treatment["broad_id"].astype(str).str.extract(BASE_ID, expand=False)
    treatment = treatment.loc[treatment["prism_broad_id_base"].notna()].copy()
    treatment = treatment.drop_duplicates(["prism_broad_id_base", "broad_id", "column_name"])
    formal = formal.merge(
        treatment[["prism_broad_id_base", "broad_id", "column_name"]],
        left_on=["prism_broad_id_base", "prism_broad_id"],
        right_on=["prism_broad_id_base", "broad_id"],
        how="inner",
        validate="one_to_many",
    )
    candidate_counts = formal.groupby("prism_broad_id_base")["column_name"].nunique()
    eligible_base_ids = sorted(candidate_counts[candidate_counts >= args.min_candidates].index.astype(str))
    lines["passed_str_profiling"] = lines["passed_str_profiling"].astype(str).str.upper().eq("TRUE")
    eligible_lines = lines.loc[
        lines["passed_str_profiling"] & lines["primary_tissue"].notna() & lines["depmap_id"].notna()
    ].drop_duplicates("depmap_id")
    crc_lines = eligible_lines.loc[eligible_lines["primary_tissue"].astype(str).str.lower().eq("colorectal")]

    payload = {
        "format": "broad_prism_cohort_v1",
        "status": "FROZEN_IDENTITY_CONTEXT_COHORT",
        "response_values_read": False,
        "selection": {
            "identity": "MATCHED_IDENTITY only; exact pert_id/InChIKey bridge; aliases/ambiguous/unmatched excluded",
            "treatment_deduplication": "base_id, broad_id, column_name",
            "line_deduplication": "unique depmap_id, passed_str_profiling=TRUE, primary_tissue non-null",
            "minimum_candidates_per_base_id": args.min_candidates,
        },
        "counts": {
            "formal_identity_broad_rows": int(formal["prism_broad_id"].nunique()),
            "eligible_base_ids": len(eligible_base_ids),
            "eligible_cancer_lines": int(eligible_lines["depmap_id"].nunique()),
            "eligible_crc_lines": int(crc_lines["depmap_id"].nunique()),
            "candidate_columns": int(formal["column_name"].nunique()),
        },
        "eligible_base_ids": eligible_base_ids,
        "eligible_cancer_depmap_ids": sorted(eligible_lines["depmap_id"].astype(str)),
        "eligible_crc_depmap_ids": sorted(crc_lines["depmap_id"].astype(str)),
        "source_checksums": {
            "bridge": digest(args.bridge),
            "treatment_info": digest(args.treatment_info),
            "cell_line_info": digest(args.cell_line_info),
        },
        "source_paths": {
            "bridge": str(args.bridge),
            "treatment_info": str(args.treatment_info),
            "cell_line_info": str(args.cell_line_info),
        },
        "downstream_boundary": "response archive must be read only after this identity/context freeze; observed oracle is operational reversal correlation, not efficacy ground truth",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
