#!/usr/bin/env python3
"""Read-back contract check for the registered GSE288798 GEX summaries."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20261001_gse288798_expression_only_v1"
OUT = ROOT / "artifacts/experiments/EXP-050" / RUN_ID
INTAKE = OUT / "intake_summary.json"
MANIFEST = OUT / "run_manifest.json"
GENE_SUMMARY = OUT / "cluster_expression_summary.parquet"
PATHWAY_SUMMARY = OUT / "descriptive_pathway_scores.tsv"
RECEIPT = OUT / "readback_verification.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as stream:
        temp = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temp, path)


def main() -> None:
    intake = json.loads(INTAKE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if intake["run_id"] != RUN_ID or intake["status"] != "DESCRIPTIVE_GEX_SUMMARIES_COMPLETE":
        raise ValueError("intake summary does not report the registered completed run")
    if manifest["run_id"] != RUN_ID or manifest["status"] != "DESCRIPTIVE_GEX_SUMMARIES_COMPLETE":
        raise ValueError("run manifest does not report the registered completed run")

    parquet = pq.ParquetFile(GENE_SUMMARY)
    expected_rows = int(intake["summaries"]["expression_rows"])
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(f"Parquet metadata rows {parquet.metadata.num_rows} != intake rows {expected_rows}")
    gene = parquet.read().to_pandas()
    key_columns = ["cluster_field", "cluster", "treatment", "gene"]
    if gene.duplicated(key_columns).any():
        raise ValueError("duplicate cluster/treatment/gene keys in expression summary")
    n_genes = int(intake["selected_assay"]["dimensions_genes_by_cells"][0])
    group_columns = ["cluster_field", "cluster", "treatment"]
    groups = gene.groupby(group_columns, dropna=False, sort=True)
    per_group_genes = groups["gene"].nunique()
    if not (per_group_genes == n_genes).all():
        raise ValueError("one or more expression groups do not contain the full named RNA gene axis")
    if not np.isfinite(gene["mean_rna_data"].to_numpy(dtype=np.float64)).all():
        raise ValueError("expression summary contains nonfinite RNA.data means")
    descriptive = gene["contrast_status"].eq("DESCRIPTIVE_MATCHED_CONTROL")
    no_control = gene["contrast_status"].eq("NO_MATCHED_CONTROL_CELLS")
    if not (descriptive | no_control).all():
        raise ValueError("expression summary contains an unrecognized contrast status")
    if gene.loc[descriptive, "delta_from_control"].isna().any() or gene.loc[no_control, "delta_from_control"].notna().any():
        raise ValueError("delta missingness does not follow the matched-control status")
    if gene.loc[descriptive, "control_mean_rna_data"].isna().any() or gene.loc[no_control, "control_mean_rna_data"].notna().any():
        raise ValueError("control mean missingness does not follow the matched-control status")
    control_rows = gene[gene["treatment"].eq("control")]
    if control_rows.empty or not np.allclose(control_rows["delta_from_control"].to_numpy(dtype=float), 0.0):
        raise ValueError("control self-difference should be zero for groups with control observations")

    mapping = intake["official_sample_mapping"]
    dataset_treatment = mapping["rds_dataset_to_treatment"]
    dataset_gsm = mapping["rds_dataset_to_gsm"]
    if set(dataset_treatment.values()) != {"control", "jq1", "ms177"} or set(dataset_gsm.values()) != {
        "GSM8775686", "GSM8775688", "GSM8775690"
    }:
        raise ValueError("official GEX sample mapping is not a complete one-to-one 3-GSM map")
    expected_pairs = {
        (dataset, treatment, dataset_gsm[dataset])
        for dataset, treatment in dataset_treatment.items()
    }
    observed_pairs = set(map(tuple, gene[["source_dataset", "treatment", "gsm_id"]].drop_duplicates().to_numpy()))
    if not observed_pairs.issubset(expected_pairs):
        raise ValueError("summary rows contain an unmapped dataset/treatment/GSM tuple")

    pathway = pd.read_csv(PATHWAY_SUMMARY, sep="\t")
    expected_pathway_rows = int(intake["summaries"]["pathway_rows"])
    if len(pathway) != expected_pathway_rows:
        raise ValueError(f"Hallmark output rows {len(pathway)} != intake rows {expected_pathway_rows}")
    pathway_group = ["cluster_field", "cluster", "treatment", "hallmark"]
    if pathway.duplicated(pathway_group).any():
        raise ValueError("duplicate cluster/treatment/Hallmark keys in pathway summary")
    per_pathway_group = pathway.groupby(["cluster_field", "cluster", "treatment"], sort=False)["hallmark"].nunique()
    if not (per_pathway_group == 50).all():
        raise ValueError("each described treatment/group contrast must carry all 50 Hallmark sets")
    score_ok = pathway["score_status"].eq("DESCRIPTIVE_ONLY")
    score_missing = pathway["score_status"].eq("NOT_COMPUTED_NO_MATCHED_CONTROL_OR_GENES")
    if not (score_ok | score_missing).all():
        raise ValueError("Hallmark output contains an unrecognized score status")
    if not np.isfinite(pathway.loc[score_ok, "mean_gene_expression_delta"].to_numpy(dtype=float)).all():
        raise ValueError("a descriptive Hallmark mean is nonfinite")
    if pathway.loc[score_missing, "mean_gene_expression_delta"].notna().any():
        raise ValueError("uncomputed Hallmark scores should remain missing, not zero")
    if not pathway["gene_coverage"].between(0, 1, inclusive="both").all():
        raise ValueError("Hallmark gene coverage is outside [0, 1]")

    verified_files = (INTAKE, GENE_SUMMARY, PATHWAY_SUMMARY)
    observed_hashes = {path.name: sha256_file(path) for path in verified_files}
    if observed_hashes != manifest.get("output_sha256"):
        raise ValueError("output SHA-256 values do not round-trip to run_manifest")
    analyzer = ROOT / "scripts/evaluation/analyze_exp050_gse288798_expression.py"
    receipt: dict[str, Any] = {
        "run_id": RUN_ID,
        "status": "PASS",
        "verified_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verification_scope": "read-back keys, row counts, finite means, explicit missingness, official GEX mapping, Hallmark coverage, and output hashes; no expression re-analysis",
        "analyzer_sha256": sha256_file(analyzer),
        "checks": {
            "expression_rows": int(len(gene)),
            "expression_groups": int(len(per_group_genes)),
            "unique_genes_per_group": n_genes,
            "duplicate_expression_keys": 0,
            "nonfinite_expression_means": 0,
            "matched_control_groups": int(groups["contrast_status"].first().eq("DESCRIPTIVE_MATCHED_CONTROL").sum()),
            "groups_without_control": int(groups["contrast_status"].first().eq("NO_MATCHED_CONTROL_CELLS").sum()),
            "pathway_rows": int(len(pathway)),
            "pathway_groups": int(len(per_pathway_group)),
            "hallmark_sets_per_group": 50,
            "uncomputed_hallmark_scores_kept_missing": int(score_missing.sum()),
            "official_gex_gsm_ids": sorted(set(gene["gsm_id"].astype(str))),
            "summary_output_hashes_match_manifest": True,
        },
        "output_sha256": observed_hashes,
    }
    atomic_json(RECEIPT, receipt)
    manifest["analyzer_sha256"] = receipt["analyzer_sha256"]
    manifest["readback_verification"] = {"status": "PASS", "receipt": str(RECEIPT.relative_to(ROOT)).replace("\\", "/"),
                                          "checks": receipt["checks"], "receipt_sha256": sha256_file(RECEIPT)}
    atomic_json(MANIFEST, manifest)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
