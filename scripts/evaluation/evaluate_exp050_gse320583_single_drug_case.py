#!/usr/bin/env python3
"""Create a bounded, text-evidence case table for three GSE320583 donors.

The script only joins approved identity/source claims to already-computed
expression summaries and summarizes the complete existing 50-set direction
table. It never reads outcome images, digitizes efficacy values, or fits a
response-to-drug-sensitivity model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


RUN_ID = "20261006_gse320583_single_drug_case_evidence_v1"
IDENTITY_RUN = "20261006_pdo_public_evidence_completion_v1"
TRANSFER_RUN = "20261001_gse320583_birinapant_transfer_v3"
EXPECTED_MODELS = {"HUB-C2-120", "HUB-C2-123", "HUB-C2-152"}
EXPECTED_KEY = "PKWRMUKBEYJEIX-DXXQBUJASA-N"
CASE_FIELDS = [
    "article_donor_index",
    "GEO_model_id",
    "single_agent_condition",
    "paper_chemical_identity_status",
    "author_reported_single_agent_phenotype",
    "author_assay_evidence_and_replicates",
    "prior_gene_spearman_rho_vs_MCF7",
    "prior_gene_pearson_r_vs_MCF7",
    "prior_nonzero_gene_sign_agreement",
    "prior_nonzero_gene_sign_denominator",
    "prior_common_gene_count",
    "prior_hallmark_same_direction_vs_MCF7_n_of_50",
    "prior_hallmark_opposite_direction_vs_MCF7_n_of_50",
    "prior_hallmark_zero_or_unavailable_n_of_50",
    "prior_hallmark_source_coverage_min_median_max",
    "cross_donor_hallmark_direction_sets_same_n_of_50",
    "cross_donor_hallmark_direction_sets_mixed_n_of_50",
    "transcript_vehicle_and_normalization",
    "assay_vehicle_and_normalization",
    "replicate_and_pooled_library_limit",
    "what_this_case_can_support",
    "what_this_case_cannot_support",
]
HALLMARK_FIELDS = [
    "article_donor_index",
    "GEO_model_id",
    "hallmark_set",
    "gmt_gene_count",
    "source_common_gene_count",
    "source_gene_coverage_of_gmt",
    "target_direction",
    "source_direction",
    "direction_relation",
    "within_set_gene_spearman_rho",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/experiments/EXP-050") / RUN_ID)
    args = parser.parse_args()
    root = args.root.resolve()
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    run_dir = run_dir.resolve()

    identity_path = root / "artifacts/experiments/EXP-050" / IDENTITY_RUN / "GSE320583_donor_model_crosswalk.tsv"
    gene_path = root / "artifacts/experiments/EXP-050" / TRANSFER_RUN / "gse320583_birinapant_cross_system_gene_agreement.tsv"
    hallmark_path = root / "artifacts/experiments/EXP-050" / TRANSFER_RUN / "gse320583_hallmark_cross_system_directions.tsv"
    consistency_path = root / "artifacts/experiments/EXP-050" / TRANSFER_RUN / "gse320583_birinapant_hallmark_model_consistency.tsv"
    source_evidence_path = run_dir / "single_agent_source_evidence.json"
    for path in (identity_path, gene_path, hallmark_path, consistency_path, source_evidence_path):
        if not path.is_file():
            raise FileNotFoundError(f"required registered input missing: {path}")

    start = time.perf_counter()
    crosswalk = read_tsv(identity_path)
    donor_by_model = {row["GEO_model_id"]: row for row in crosswalk}
    if len(donor_by_model) != 3 or set(donor_by_model) != EXPECTED_MODELS:
        raise ValueError("article donor/model crosswalk is not the frozen 3-model set")
    gene_rows = [r for r in read_tsv(gene_path) if r["target_dataset"] == "GSE320583" and r["drug_label"].casefold() == "birinapant"]
    genes_by_model = {r["model_id"]: r for r in gene_rows}
    if len(gene_rows) != 3 or set(genes_by_model) != EXPECTED_MODELS:
        raise ValueError("prior expression comparison does not have exactly one row per frozen model")
    for row in gene_rows:
        if row["source_dataset"] != "GSE70138" or row["source_cell_id"] != "MCF7":
            raise ValueError("prior source-cell comparison differs from the frozen case")
        if row["full_inchi_key"] != EXPECTED_KEY or row["dose_um"] != "10.0" or row["time_h"] != "24.0":
            raise ValueError("prior expression comparison differs from registered key/10uM/24h condition")
        if int(row["n_common_landmark_genes"]) != 961:
            raise ValueError("prior expression comparison gene denominator differs from 961")
        if row["absolute_amplitudes_comparable"].casefold() != "false" or row["inferential_p_value"]:
            raise ValueError("prior comparison has an unexpected inferential/amplitude status")

    hallmark_rows = read_tsv(hallmark_path)
    hallmark_by_model: dict[str, list[dict[str, str]]] = {model: [] for model in EXPECTED_MODELS}
    for row in hallmark_rows:
        if row["model_id"] in hallmark_by_model:
            hallmark_by_model[row["model_id"]].append(row)
    if any(len(rows) != 50 for rows in hallmark_by_model.values()):
        raise ValueError("the complete 50-set Hallmark direction table is required for each model")
    consistency_rows = [r for r in read_tsv(consistency_path) if r["condition_code"] == "Bi"]
    if len(consistency_rows) != 50 or {r["hallmark_set"] for r in consistency_rows} != {r["hallmark_set"] for r in hallmark_rows}:
        raise ValueError("the complete shared 50-set cross-donor consistency table is required")
    consistency_same = sum(r["cross_model_direction_consistent"].casefold() == "true" for r in consistency_rows)
    consistency_mixed = sum(r["cross_model_direction_consistent"].casefold() == "false" for r in consistency_rows)
    if consistency_same + consistency_mixed != 50:
        raise ValueError("unexpected unresolved cross-donor Hallmark directions")

    source = json.loads(source_evidence_path.read_text(encoding="utf-8"))
    donor_claims = source["donor_claims"]
    if set(donor_claims) != {"1", "2", "3"}:
        raise ValueError("single-agent source evidence must cover all three anonymous donor indices")

    started_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    cases: list[dict[str, str]] = []
    hallmark_out: list[dict[str, str]] = []
    for model_id, mapping in sorted(donor_by_model.items(), key=lambda pair: int(pair[1]["article_donor_index"])):
        donor = mapping["article_donor_index"]
        expr = genes_by_model[model_id]
        hrows = hallmark_by_model[model_id]
        counts: dict[str, int] = {}
        for item in hrows:
            relation = item["direction_relation"]
            counts[relation] = counts.get(relation, 0) + 1
            hallmark_out.append(
                {
                    "article_donor_index": donor,
                    "GEO_model_id": model_id,
                    "hallmark_set": item["hallmark_set"],
                    "gmt_gene_count": item["gmt_gene_count"],
                    "source_common_gene_count": item["source_common_gene_count"],
                    "source_gene_coverage_of_gmt": item["source_gene_coverage_of_gmt"],
                    "target_direction": item["target_direction"],
                    "source_direction": item["source_direction"],
                    "direction_relation": relation,
                    "within_set_gene_spearman_rho": item["within_set_gene_spearman_rho"],
                }
            )
        if sum(counts.values()) != 50:
            raise ValueError(f"Hallmark relationship categories do not sum to 50 for {model_id}")
        zero_unavailable = counts.get("zero_or_unavailable", 0) + counts.get("zero", 0) + counts.get("unavailable", 0)
        coverage = [float(item["source_gene_coverage_of_gmt"]) for item in hrows]
        if not all(math.isfinite(x) for x in coverage):
            raise ValueError(f"nonfinite Hallmark coverage for {model_id}")
        claim = donor_claims[donor]
        cases.append(
            {
                "article_donor_index": donor,
                "GEO_model_id": model_id,
                "single_agent_condition": "论文命名Birinapant，10 µM，24 h；全化学结构键未由该论文商品目录号/CAS闭合",
                "paper_chemical_identity_status": source["chemical_identity_status"],
                "author_reported_single_agent_phenotype": claim["donor_specific_text_evidence"],
                "author_assay_evidence_and_replicates": source["assay_and_replicate_evidence"],
                "prior_gene_spearman_rho_vs_MCF7": expr["gene_spearman_rho"],
                "prior_gene_pearson_r_vs_MCF7": expr["gene_pearson_r"],
                "prior_nonzero_gene_sign_agreement": expr["nonzero_gene_sign_agreement"],
                "prior_nonzero_gene_sign_denominator": expr["nonzero_gene_sign_denominator"],
                "prior_common_gene_count": expr["n_common_landmark_genes"],
                "prior_hallmark_same_direction_vs_MCF7_n_of_50": str(counts.get("same_sign", 0)),
                "prior_hallmark_opposite_direction_vs_MCF7_n_of_50": str(counts.get("opposite_sign", 0)),
                "prior_hallmark_zero_or_unavailable_n_of_50": str(zero_unavailable),
                "prior_hallmark_source_coverage_min_median_max": f"{min(coverage):.4f}/{statistics.median(coverage):.4f}/{max(coverage):.4f}",
                "cross_donor_hallmark_direction_sets_same_n_of_50": str(consistency_same),
                "cross_donor_hallmark_direction_sets_mixed_n_of_50": str(consistency_mixed),
                "transcript_vehicle_and_normalization": source["transcript_vehicle_and_normalization"],
                "assay_vehicle_and_normalization": source["assay_vehicle_and_normalization"],
                "replicate_and_pooled_library_limit": source["replicate_and_pooled_library_limit"],
                "what_this_case_can_support": "论文所述匿名供体模型级单药病例描述，与既有模型级转录响应描述并列；50个Hallmark方向完整列出，支持与分歧都保留。",
                "what_this_case_cannot_support": "无逐模型数值活性表/可绑定商品身份，不能算PDO转录—药敏相关、单药排序、效应量、推断区间、预测准确率、因果或临床获益；不能把单库细胞数当独立复孔。",
            }
        )
    case_path = run_dir / "gse320583_single_agent_three_donor_case_evidence.tsv"
    hallmark_path_out = run_dir / "gse320583_single_agent_hallmark_directions_with_donor_identity.tsv"
    write_tsv(case_path, cases, CASE_FIELDS)
    write_tsv(hallmark_path_out, hallmark_out, HALLMARK_FIELDS)
    case_readback = read_tsv(case_path)
    hallmark_readback = read_tsv(hallmark_path_out)
    if len(case_readback) != 3 or len(hallmark_readback) != 150:
        raise RuntimeError("case/Hallmark output row-count readback failed")
    if sum(int(row["prior_hallmark_same_direction_vs_MCF7_n_of_50"]) + int(row["prior_hallmark_opposite_direction_vs_MCF7_n_of_50"]) + int(row["prior_hallmark_zero_or_unavailable_n_of_50"]) for row in case_readback) != 150:
        raise RuntimeError("case summaries do not account for every donor-set direction relation")

    elapsed = time.perf_counter() - start
    finished_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    output_hashes = {case_path.name: sha256(case_path), hallmark_path_out.name: sha256(hallmark_path_out)}
    summary = {
        "experiment_id": "EXP-050",
        "run_id": RUN_ID,
        "status": "COMPLETED_EXPLORATORY_SINGLE_AGENT_CASE_DESCRIPTION; NUMERIC_DRUG_EFFECT_ASSOCIATION_NOT_ESTIMATED",
        "article": source["article"],
        "donor_model_identity": {r["article_donor_index"]: r["GEO_model_id"] for r in crosswalk},
        "fixed_condition": {"paper_named_drug": "Birinapant", "dose_um": 10, "time_h": 24, "source_cell": "MCF7", "common_genes": 961},
        "case_table_rows": len(case_readback),
        "complete_hallmark_donor_rows": len(hallmark_readback),
        "hallmark_direction_counts_by_donor": [
            {"article_donor_index": r["article_donor_index"], "GEO_model_id": r["GEO_model_id"], "same_of_50": r["prior_hallmark_same_direction_vs_MCF7_n_of_50"], "opposite_of_50": r["prior_hallmark_opposite_direction_vs_MCF7_n_of_50"], "zero_or_unavailable_of_50": r["prior_hallmark_zero_or_unavailable_n_of_50"]}
            for r in case_readback
        ],
        "cross_donor_hallmark_consistency": {"all_three_same_direction_of_50": consistency_same, "mixed_direction_of_50": consistency_mixed, "unresolved_of_50": 0},
        "drug_sensitivity_values_digitized_or_used": False,
        "efficacy_correlation_ranking_inference_or_prediction_performance_calculated": False,
        "numeric_association_status": "NOT_ESTIMATED; article figure supplies an assay design and replicate count but no machine-readable per-model numeric viability values were collected in this bounded package.",
        "interpretation_limit": "Exploratory public case description only. Existing gene-level and Hallmark quantities are copied or counted from frozen expression outputs, not new scores; cross-platform amplitudes remain incomparable.",
        "outputs_sha256": output_hashes,
        "gpu_hours": 0,
        "cpu_wall_seconds": elapsed,
    }
    summary_path = run_dir / "summary.json"
    write_json(summary_path, summary)
    manifest = {
        "experiment_id": "EXP-050",
        "run_id": RUN_ID,
        "started_at_local": started_at.isoformat(timespec="seconds"),
        "finished_at_local": finished_at.isoformat(timespec="seconds"),
        "runtime_cpu_wall_minutes": elapsed / 60,
        "gpu_hours": 0,
        "cpu_wall_cap_minutes": 60,
        "python": sys.version,
        "platform": platform.platform(),
        "script_path": "scripts/evaluation/evaluate_exp050_gse320583_single_drug_case.py",
        "script_sha256": sha256(Path(__file__).resolve()),
        "inputs": {p.name: {"path": str(p.relative_to(root)), "sha256": sha256(p)} for p in (identity_path, gene_path, hallmark_path, consistency_path, source_evidence_path)},
        "outputs_sha256": {**output_hashes, "summary.json": sha256(summary_path)},
        "protocol_snapshot_sha256": sha256(run_dir / "protocol_snapshot.json"),
        "start_receipt_sha256": sha256(run_dir / "start_receipt.json"),
        "all_50_hallmark_sets_retained_per_model": True,
        "donor_model_join_exact": True,
        "interpretation_limit": summary["interpretation_limit"],
    }
    manifest_path = run_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    sha_items = [
        run_dir / "protocol_snapshot.json",
        run_dir / "start_receipt.json",
        source_evidence_path,
        case_path,
        hallmark_path_out,
        summary_path,
        manifest_path,
    ]
    sha_manifest = run_dir / "sha256_manifest.tsv"
    with sha_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["relative_path", "sha256", "bytes"])
        for path in sha_items:
            writer.writerow([path.name, sha256(path), path.stat().st_size])
    print(json.dumps({"run_id": RUN_ID, "status": summary["status"], "case_rows": len(case_readback), "hallmark_rows": len(hallmark_readback), "hallmark_counts": summary["hallmark_direction_counts_by_donor"], "elapsed_seconds": elapsed, "gpu_hours": 0, "outputs": output_hashes}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
