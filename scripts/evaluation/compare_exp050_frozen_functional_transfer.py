#!/usr/bin/env python3
"""Describe functional-direction transfer using frozen EXP-050 DOX/BTZ contrasts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


DEFAULT_RUN_ID = "20261001_frozen_functional_transfer_v1"
DEFAULT_REPLICATES = 1000
DEFAULT_SEED = 20261001


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty result without an explicit schema: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_gmt(path: Path) -> tuple[dict[str, list[str]], set[str]]:
    sets: dict[str, list[str]] = {}
    universe: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            name = fields[0]
            genes = list(dict.fromkeys(g.strip() for g in fields[2:] if g.strip()))
            if name in sets:
                raise ValueError(f"Duplicate gene-set name in GMT: {name}")
            sets[name] = genes
            universe.update(genes)
    if len(sets) != 50:
        raise ValueError(f"Expected frozen Hallmark v2023.2.Hs to contain 50 sets, found {len(sets)}")
    return sets, universe


def sign(value: float) -> str:
    if not np.isfinite(value) or value == 0:
        return "zero_or_unavailable"
    return "positive" if value > 0 else "negative"


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    result = spearmanr(left, right)
    value = float(result.statistic)
    return value if np.isfinite(value) else None


def random_panel_rank_controls(
    target: np.ndarray,
    source: np.ndarray,
    matched_sizes: np.ndarray,
    n_replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate panels of random gene groups exactly matched on observed size/coverage."""
    n_genes = len(target)
    draws = np.zeros((n_replicates, len(matched_sizes), 2), dtype=float)
    for panel_index in range(n_replicates):
        for group_index, k in enumerate(matched_sizes):
            if k <= 0 or k > n_genes:
                raise ValueError(f"Invalid matched in-axis group size: {k}")
            selected = rng.choice(n_genes, size=int(k), replace=False)
            draws[panel_index, group_index] = (
                target[selected].mean(), source[selected].mean()
            )
    target_scores, source_scores = draws[:, :, 0], draws[:, :, 1]
    target_ranks = rankdata(target_scores, axis=1, method="average")
    source_ranks = rankdata(source_scores, axis=1, method="average")
    target_centered = target_ranks - target_ranks.mean(axis=1, keepdims=True)
    source_centered = source_ranks - source_ranks.mean(axis=1, keepdims=True)
    numerator = np.mean(target_centered * source_centered, axis=1)
    denominator = target_centered.std(axis=1) * source_centered.std(axis=1)
    correlations = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )
    return correlations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--random-panels", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.random_panels < 100:
        raise ValueError("At least 100 size/coverage-matched random panels are required")

    root = args.repo_root.resolve()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    base = root / "artifacts/experiments/EXP-050/20260930_keyed_response_v7"
    vector_path = base / "GSE287429_MCF7_all_exposure_response_vectors.parquet"
    similarity_path = base / "GSE287429_MCF7_all_exposure_similarity_56.tsv"
    exposure_path = base / "LINCS_MCF7_DOX_BTZ_QC_eligible_exposures.tsv"
    axis_path = base / "EXP050_common_gene_axis_914.tsv"
    gmt_path = root / "data/raw/exp013/msigdb/h.all.v2023.2.Hs.symbols.gmt"

    vectors = pd.read_parquet(vector_path)
    similarities = pd.read_csv(similarity_path, sep="\t")
    exposures = pd.read_csv(exposure_path, sep="\t")
    axis = pd.read_csv(axis_path, sep="\t")
    gene_sets, gmt_universe = load_gmt(gmt_path)

    required_vector_columns = {
        "target_contrast", "source_pert_id", "source_dose", "source_dose_unit",
        "source_time", "source_time_unit", "gene_index", "GeneSymbol",
        "target_author_scale_delta", "LINCS_condition_delta",
    }
    if not required_vector_columns.issubset(vectors.columns):
        raise ValueError(f"Frozen vector parquet missing columns: {sorted(required_vector_columns - set(vectors.columns))}")
    if len(axis) != 914 or axis["HGNC_symbol"].nunique() != 914:
        raise ValueError("Frozen EXP050 common axis does not contain 914 unique HGNC symbols")
    if len(vectors) != 56 * 914 or len(similarities) != 56:
        raise ValueError(f"Expected 56 frozen comparisons x 914 genes, found {len(similarities)} comparisons / {len(vectors)} rows")
    if similarities["source_cell_line"].nunique() != 1 or similarities["source_cell_line"].iloc[0] != "MCF7":
        raise ValueError("Frozen comparison no longer resolves to a single MCF7 source line")

    exposure_keys = ["pert_id", "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit"]
    exposure_lookup: dict[tuple[str, float, str, str, str], dict[str, Any]] = {}
    for row in exposures.to_dict(orient="records"):
        key = (
            str(row["pert_id"]), float(row["pert_dose"]), str(row["pert_dose_unit"]),
            str(row["pert_time"]), str(row["pert_time_unit"]),
        )
        if key in exposure_lookup:
            raise ValueError(f"Exposure key is not unique in the frozen MCF7 condition contract: {key}")
        exposure_lookup[key] = row

    first = vectors.sort_values("gene_index").groupby("target_contrast", sort=False)
    target_vectors: dict[str, np.ndarray] = {}
    gene_symbols: list[str] | None = None
    for target_name, group in first:
        if group.groupby("gene_index")["target_author_scale_delta"].nunique(dropna=False).max() != 1:
            raise ValueError(f"Frozen target delta differs between source-exposure rows for {target_name}")
        unique = group.sort_values("gene_index").drop_duplicates("gene_index")
        if len(unique) != 914:
            raise ValueError(f"Target vector {target_name} does not have 914 genes")
        values = unique["target_author_scale_delta"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite frozen target response in {target_name}")
        symbols = unique["GeneSymbol"].astype(str).tolist()
        if gene_symbols is None:
            gene_symbols = symbols
        elif symbols != gene_symbols:
            raise ValueError("Frozen target vector gene ordering changed across contrasts")
        if not np.array_equal(unique["gene_index"].to_numpy(), np.arange(914)):
            raise ValueError(f"Gene-index axis changed for {target_name}")
        target_vectors[str(target_name)] = values

    assert gene_symbols is not None
    if gene_symbols != axis["HGNC_symbol"].astype(str).tolist():
        raise ValueError("The 914 response-vector symbols do not exactly match the frozen axis file order")
    gene_index = {symbol: i for i, symbol in enumerate(gene_symbols)}

    hallmark_names = list(gene_sets)
    hallmark_indices: list[np.ndarray] = []
    hallmark_total_sizes: list[int] = []
    for name in hallmark_names:
        members = gene_sets[name]
        indices = np.array([gene_index[g] for g in members if g in gene_index], dtype=int)
        if len(indices) != len(set(indices.tolist())):
            raise ValueError(f"Duplicate gene symbol after exact-axis membership in {name}")
        if len(indices) == 0:
            raise ValueError(f"Hallmark set has no members on the frozen 914-gene axis: {name}")
        hallmark_indices.append(indices)
        hallmark_total_sizes.append(len(members))
    matched_sizes = np.array([len(ix) for ix in hallmark_indices], dtype=int)

    source_vectors: dict[tuple[str, float, str, str, str], np.ndarray] = {}
    comparison_rows: list[dict[str, Any]] = []
    pathway_rows: list[dict[str, Any]] = []
    counterexamples: list[dict[str, Any]] = []
    random_control_rows: list[dict[str, Any]] = []
    random_spearman_by_comparison: dict[str, np.ndarray] = {}
    comp_key_fields = ["target_contrast", "source_pert_id", "source_dose", "source_dose_unit", "source_time", "source_time_unit"]
    rng = np.random.default_rng(args.seed)

    for comparison in similarities.to_dict(orient="records"):
        key = (
            str(comparison["source_pert_id"]), float(comparison["source_dose"]),
            str(comparison["source_dose_unit"]), str(comparison["source_time"]),
            str(comparison["source_time_unit"]),
        )
        if key not in exposure_lookup:
            raise ValueError(f"Frozen source condition not found by exact exposure tuple: {key}")
        exposure = exposure_lookup[key]
        target_name = str(comparison["target_contrast"])
        subset = vectors.loc[
            (vectors["target_contrast"] == target_name)
            & (vectors["source_pert_id"] == key[0])
            & np.isclose(vectors["source_dose"].astype(float), key[1])
            & (vectors["source_dose_unit"].astype(str) == key[2])
            & (vectors["source_time"].astype(str) == key[3])
            & (vectors["source_time_unit"].astype(str) == key[4])
        ].sort_values("gene_index")
        if len(subset) != 914 or subset["gene_index"].duplicated().any():
            raise ValueError(f"Frozen comparison vector has bad key/axis binding: {target_name} {key}")
        if not np.array_equal(subset["gene_index"].to_numpy(), np.arange(914)):
            raise ValueError(f"Frozen comparison vector axis order failed: {target_name} {key}")
        target = subset["target_author_scale_delta"].to_numpy(dtype=float)
        source = subset["LINCS_condition_delta"].to_numpy(dtype=float)
        if not np.isfinite(target).all() or not np.isfinite(source).all():
            raise ValueError(f"Non-finite values in frozen comparison: {target_name} {key}")
        if str(exposure["cell_id"]) != "MCF7":
            raise ValueError(f"Source line changed from the approved single-line frame: {exposure['cell_id']}")
        source_key = key
        if source_key in source_vectors and not np.array_equal(source_vectors[source_key], source):
            raise ValueError(f"Repeated source condition has inconsistent response values: {key}")
        source_vectors[source_key] = source

        hallmark_target = np.array([target[ix].mean() for ix in hallmark_indices])
        hallmark_source = np.array([source[ix].mean() for ix in hallmark_indices])
        rank_rho = safe_spearman(hallmark_target, hallmark_source)
        same_direction = 0
        changed_direction = 0
        unresolved_direction = 0
        for i, name in enumerate(hallmark_names):
            ix = hallmark_indices[i]
            target_mean = float(hallmark_target[i])
            source_mean = float(hallmark_source[i])
            target_sign, source_sign = sign(target_mean), sign(source_mean)
            if target_sign == "zero_or_unavailable" or source_sign == "zero_or_unavailable":
                direction_status = "cannot_judge_zero_mean"
                unresolved_direction += 1
            elif target_sign == source_sign:
                direction_status = "preserved_same_sign"
                same_direction += 1
            else:
                direction_status = "changed_opposite_sign"
                changed_direction += 1
            both_nonzero = (target[ix] != 0) & (source[ix] != 0)
            n_both_nonzero = int(both_nonzero.sum())
            gene_sign_agreement = (
                float(np.mean(np.sign(target[ix][both_nonzero]) == np.sign(source[ix][both_nonzero])))
                if n_both_nonzero else None
            )
            row = {
                "target_contrast": target_name,
                "drug": str(comparison["target_drug"]),
                "target_time_hours": int(comparison["target_time_hours"]),
                "source_cell_line": "MCF7",
                "source_condition_id": str(exposure["condition_id"]),
                "source_dose": float(comparison["source_dose"]),
                "source_dose_unit": str(comparison["source_dose_unit"]),
                "source_time": str(comparison["source_time"]),
                "source_time_unit": str(comparison["source_time_unit"]),
                "hallmark_set": name,
                "full_hallmark_size": hallmark_total_sizes[i],
                "n_genes_on_frozen_914_axis": int(len(ix)),
                "axis_coverage_fraction": float(len(ix) / hallmark_total_sizes[i]),
                "target_set_mean_direction": target_sign,
                "source_set_mean_direction": source_sign,
                "direction_status": direction_status,
                "target_set_mean_descriptive_score": target_mean,
                "source_set_mean_descriptive_score": source_mean,
                "n_axis_genes_nonzero_on_both_sides": n_both_nonzero,
                "gene_level_sign_agreement_within_set": gene_sign_agreement,
                "rank_desc_target": float(rankdata(-hallmark_target, method="average")[i]),
                "rank_desc_source": float(rankdata(-hallmark_source, method="average")[i]),
                "rank_displacement_abs": float(abs(rankdata(-hallmark_target, method="average")[i] - rankdata(-hallmark_source, method="average")[i])),
                "interpretation_boundary": "Descriptive direction/rank comparison only; score magnitudes are not compared across platforms.",
            }
            pathway_rows.append(row)
            if direction_status == "changed_opposite_sign":
                counterexamples.append({**row, "counterexample_type": "source_vs_target_function_direction_reversal"})

        comparison_id = f"{target_name}|{exposure['condition_id']}"
        random_rhos = random_panel_rank_controls(
            target, source, matched_sizes, args.random_panels, rng
        )
        random_spearman_by_comparison[comparison_id] = random_rhos
        finite_random = random_rhos[np.isfinite(random_rhos)]
        random_q025, random_q50, random_q975 = np.quantile(finite_random, [0.025, 0.5, 0.975]) if len(finite_random) else (np.nan, np.nan, np.nan)
        rank_percentile = (
            float(100 * np.mean(finite_random <= rank_rho))
            if len(finite_random) and rank_rho is not None else None
        )
        comparison_rows.append({
            "comparison_id": comparison_id,
            "target_contrast": target_name,
            "drug": str(comparison["target_drug"]),
            "target_time_hours": int(comparison["target_time_hours"]),
            "target_treatment_samples": int(comparison["target_n_treatment_samples"]),
            "target_control_samples": int(comparison["target_n_control_samples"]),
            "source_dataset": "LINCS GSE92742",
            "source_cell_line": "MCF7",
            "source_pert_id": str(comparison["source_pert_id"]),
            "source_condition_id": str(exposure["condition_id"]),
            "source_dose": float(comparison["source_dose"]),
            "source_dose_unit": str(comparison["source_dose_unit"]),
            "source_time": str(comparison["source_time"]),
            "source_time_unit": str(comparison["source_time_unit"]),
            "source_treatment_instances": int(exposure["n_treatment_instances"]),
            "source_matched_control_groups": int(exposure["n_control_groups"]),
            "common_genes": 914,
            "hallmark_sets_compared": len(hallmark_names),
            "hallmark_same_mean_direction_sets": same_direction,
            "hallmark_opposite_mean_direction_sets": changed_direction,
            "hallmark_zero_or_unavailable_mean_sets": unresolved_direction,
            "hallmark_mean_score_rank_spearman": rank_rho,
            "matched_random_panel_count": args.random_panels,
            "matched_random_panel_rank_spearman_median": float(random_q50),
            "matched_random_panel_rank_spearman_2_5pct": float(random_q025),
            "matched_random_panel_rank_spearman_97_5pct": float(random_q975),
            "hallmark_rank_random_control_percentile": rank_percentile,
            "fraction_random_panels_at_or_above_hallmark_rank": float(np.mean(finite_random >= rank_rho)) if len(finite_random) and rank_rho is not None else None,
            "randomization_interpretation": "Descriptive matched-gene-set reference distribution; not a p-value, confidence interval, or biological replicate test.",
            "cross_time_boundary": f"Target {comparison['target_time_hours']} h vs LINCS {comparison['source_time']} {comparison['source_time_unit']}; target dose is unknown.",
        })
        random_control_rows.append({
            "comparison_id": comparison_id,
            "random_panel_count": args.random_panels,
            "hallmark_rank_spearman": rank_rho,
            "random_median": float(random_q50),
            "random_2_5pct": float(random_q025),
            "random_97_5pct": float(random_q975),
            "hallmark_percentile_in_random_distribution": rank_percentile,
            "random_panels_at_or_above_hallmark_fraction": float(np.mean(finite_random >= rank_rho)) if len(finite_random) and rank_rho is not None else None,
            "control_matching": "Each of 50 random groups matches one Hallmark set's total GMT size and exact number of genes on the shared 914-gene axis; scored members are sampled without replacement from that axis.",
        })

    # Pairwise target-time direction reversals are explicit counterexamples to time-invariant summaries.
    pathway_frame = pd.DataFrame(pathway_rows)
    for drug, short in [("doxorubicin", "DOX"), ("bortezomib", "BTZ")]:
        left = pathway_frame.loc[pathway_frame["target_contrast"] == f"{short}_23h"]
        right = pathway_frame.loc[pathway_frame["target_contrast"] == f"{short}_47h"]
        for condition_id in sorted(set(left["source_condition_id"]) & set(right["source_condition_id"])):
            left_cond = left.loc[left["source_condition_id"] == condition_id].set_index("hallmark_set")
            right_cond = right.loc[right["source_condition_id"] == condition_id].set_index("hallmark_set")
            for name in hallmark_names:
                a, b = left_cond.loc[name], right_cond.loc[name]
                if a["target_set_mean_direction"] in {"positive", "negative"} and b["target_set_mean_direction"] in {"positive", "negative"} and a["target_set_mean_direction"] != b["target_set_mean_direction"]:
                    counterexamples.append({
                        "counterexample_type": "target_cross_time_direction_reversal",
                        "target_contrast": f"{short}_23h_to_47h",
                        "drug": drug,
                        "target_time_hours": "23->47",
                        "source_cell_line": "MCF7",
                        "source_condition_id": condition_id,
                        "source_dose": a["source_dose"],
                        "source_dose_unit": a["source_dose_unit"],
                        "source_time": a["source_time"],
                        "source_time_unit": a["source_time_unit"],
                        "hallmark_set": name,
                        "full_hallmark_size": a["full_hallmark_size"],
                        "n_genes_on_frozen_914_axis": a["n_genes_on_frozen_914_axis"],
                        "axis_coverage_fraction": a["axis_coverage_fraction"],
                        "target_set_mean_direction": f"{a['target_set_mean_direction']} -> {b['target_set_mean_direction']}",
                        "source_set_mean_direction": a["source_set_mean_direction"],
                        "direction_status": "target direction changed between 23 h and 47 h",
                        "target_set_mean_descriptive_score": f"{a['target_set_mean_descriptive_score']} -> {b['target_set_mean_descriptive_score']}",
                        "source_set_mean_descriptive_score": a["source_set_mean_descriptive_score"],
                        "n_axis_genes_nonzero_on_both_sides": None,
                        "gene_level_sign_agreement_within_set": None,
                        "rank_desc_target": None,
                        "rank_desc_source": None,
                        "rank_displacement_abs": None,
                        "interpretation_boundary": "Target response direction changes between its two measured times for the same drug; no time-matched MCF7 exposure exists because target dose/time differs.",
                    })

    by_target_rows: list[dict[str, Any]] = []
    for target_name, group in pd.DataFrame(comparison_rows).groupby("target_contrast", sort=True):
        rand_median = group["matched_random_panel_rank_spearman_median"].to_numpy(dtype=float)
        hallmark_rho = group["hallmark_mean_score_rank_spearman"].dropna().to_numpy(dtype=float)
        percentile = group["hallmark_rank_random_control_percentile"].dropna().to_numpy(dtype=float)
        by_target_rows.append({
            "target_contrast": target_name,
            "drug": group["drug"].iloc[0],
            "target_time_hours": int(group["target_time_hours"].iloc[0]),
            "source_condition_count_descriptive_only": int(len(group)),
            "source_line_count": 1,
            "source_line": "MCF7",
            "hallmark_rank_spearman_median_across_source_conditions": float(np.median(hallmark_rho)) if len(hallmark_rho) else None,
            "hallmark_rank_spearman_min": float(np.min(hallmark_rho)) if len(hallmark_rho) else None,
            "hallmark_rank_spearman_max": float(np.max(hallmark_rho)) if len(hallmark_rho) else None,
            "matched_random_rank_median_across_source_conditions": float(np.median(rand_median)),
            "median_hallmark_random_percentile": float(np.median(percentile)) if len(percentile) else None,
            "fraction_source_conditions_hallmark_above_random_97_5pct": float(np.mean(group["hallmark_mean_score_rank_spearman"] > group["matched_random_panel_rank_spearman_97_5pct"])),
            "median_hallmark_sets_same_mean_direction": float(group["hallmark_same_mean_direction_sets"].median()),
            "median_hallmark_sets_opposite_mean_direction": float(group["hallmark_opposite_mean_direction_sets"].median()),
            "interpretation_boundary": "Across-exposure distribution for one target contrast and one source cell line; source conditions are not independent biological replicates and are not ranked for model selection.",
        })

    # Summary across exposures per target/pathway: a compact preservation/change/unknown view.
    target_function_rows: list[dict[str, Any]] = []
    for (target_name, name), group in pathway_frame.groupby(["target_contrast", "hallmark_set"], sort=True):
        same = int((group["direction_status"] == "preserved_same_sign").sum())
        changed = int((group["direction_status"] == "changed_opposite_sign").sum())
        unknown = int((group["direction_status"] == "cannot_judge_zero_mean").sum())
        source_signs = set(group["source_set_mean_direction"])
        target_signs = set(group["target_set_mean_direction"])
        target_function_rows.append({
            "target_contrast": target_name,
            "drug": group["drug"].iloc[0],
            "target_time_hours": int(group["target_time_hours"].iloc[0]),
            "hallmark_set": name,
            "full_hallmark_size": int(group["full_hallmark_size"].iloc[0]),
            "n_genes_on_914_axis": int(group["n_genes_on_frozen_914_axis"].iloc[0]),
            "axis_coverage_fraction": float(group["axis_coverage_fraction"].iloc[0]),
            "target_mean_direction": group["target_set_mean_direction"].iloc[0],
            "source_conditions_with_same_direction": same,
            "source_conditions_with_opposite_direction": changed,
            "source_conditions_zero_or_unavailable": unknown,
            "source_condition_count_descriptive_only": int(len(group)),
            "source_direction_varies_across_doses_or_times": len(source_signs - {"zero_or_unavailable"}) > 1,
            "target_direction_varies_across_source_condition_rows": len(target_signs - {"zero_or_unavailable"}) > 1,
            "median_gene_level_sign_agreement": float(group["gene_level_sign_agreement_within_set"].median()),
            "interpretation_boundary": "Fifty Hallmark summaries across each frozen source-condition set; no inferential denominator and no cross-platform magnitude interpretation.",
        })

    out = root / "artifacts/experiments/EXP-050" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "comparison": out / "functional_transfer_by_source_condition.tsv",
        "pathway": out / "hallmark_direction_and_ranks.tsv",
        "counterexamples": out / "functional_direction_counterexamples.tsv",
        "target_summary": out / "functional_transfer_by_target_contrast.tsv",
        "target_pathways": out / "hallmark_status_by_target_contrast.tsv",
        "random_controls": out / "matched_random_set_control_summary.tsv",
        "summary": out / "summary.json",
        "manifest": out / "run_manifest.json",
    }
    write_tsv(paths["comparison"], comparison_rows)
    write_tsv(paths["pathway"], pathway_rows)
    if counterexamples:
        write_tsv(paths["counterexamples"], counterexamples)
    else:
        with paths["counterexamples"].open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["counterexample_type", *pathway_rows[0].keys()], delimiter="\t")
            writer.writeheader()
    write_tsv(paths["target_summary"], by_target_rows)
    write_tsv(paths["target_pathways"], target_function_rows)
    write_tsv(paths["random_controls"], random_control_rows)

    direction_counts = {
        key: int((pathway_frame["direction_status"] == key).sum())
        for key in ["preserved_same_sign", "changed_opposite_sign", "cannot_judge_zero_mean"]
    }
    summary = {
        "experiment_id": "EXP-050",
        "run_id": args.run_id,
        "status": "completed_frozen_functional_direction_transfer_descriptive",
        "frame": "All 56 previously frozen GSE287429 DOX/BTZ versus QC-eligible LINCS MCF7 comparisons; 4 target contrasts x 10/18 source conditions x 914 genes.",
        "source_cell_line_count": 1,
        "source_cell_line": "MCF7",
        "target_contrast_count": 4,
        "source_condition_count": len(source_vectors),
        "comparison_count": len(comparison_rows),
        "common_gene_axis_count": 914,
        "hallmark_resource": "MSigDB Hallmark v2023.2.Hs symbols; exact symbol membership on the frozen 914-gene axis.",
        "hallmark_set_count": len(hallmark_names),
        "hallmark_gene_set_direction_counts_over_all_56_comparisons": direction_counts,
        "cross_system_ranking_metric": "Spearman rank correlation across the 50 Hallmark per-set mean response scores; reports ordering agreement and does not compare absolute magnitudes.",
        "coverage_rule": "Per Hallmark set, record full GMT size, exact members on the 914-gene axis, and their ratio; all comparison vectors must be finite for the complete 914-gene axis.",
        "matched_random_group_control": {
            "replicate_panels_per_frozen_comparison": args.random_panels,
            "seed": args.seed,
            "groups_per_panel": 50,
            "matching": "For each Hallmark set, match original GMT total size and exact number of genes intersecting the 914-gene axis; use without-replacement random genes from the 914-gene scoring universe for the intersecting members.",
            "comparison": "Compare the Hallmark-panel rank correlation to the descriptive distribution across matched random panels; no p-value or confidence interval is reported.",
        },
        "cross_time_and_scale_boundary": "Target conditions are 23/47 h with dose not stated in GEO; source is the same drug at LINCS MCF7 6/24 h conditions. All eligible exposures are retained with no best-condition selection. Absolute cross-platform amplitudes are not compared.",
        "counterexample_count": len(counterexamples),
        "counterexample_types": {
            "source_vs_target_function_direction_reversal": sum(r["counterexample_type"] == "source_vs_target_function_direction_reversal" for r in counterexamples),
            "target_cross_time_direction_reversal": sum(r["counterexample_type"] == "target_cross_time_direction_reversal" for r in counterexamples),
        },
        "interpretation": "Functional set direction and rank are descriptive gene-set summaries. Gene sets and source exposure conditions are not independent biological n; no patient-level or pathway-activity claim is made. Counterexamples are retained row by row.",
        "gpu_hours": 0,
        "outputs": {name: path.relative_to(root).as_posix() for name, path in paths.items()},
        "by_target_contrast": by_target_rows,
    }
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    inputs = [vector_path, similarity_path, exposure_path, axis_path, gmt_path]
    run_manifest = {
        "experiment_id": "EXP-050",
        "run_id": args.run_id,
        "started_at_local": started_at,
        "finished_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": f"conda run --no-capture-output -n drugscreening-gpu python scripts/evaluation/compare_exp050_frozen_functional_transfer.py --run-id {args.run_id} --random-panels {args.random_panels} --seed {args.seed}",
        "environment": "WSL2 Conda drugscreening-gpu",
        "gpu_hours": 0,
        "scope": "CPU-only evaluation of existing frozen 56-row transcript-response vectors; no model fit, target PDO prediction, exposure selection, or external contact.",
        "inputs": {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in inputs},
        "code": {"path": "scripts/evaluation/compare_exp050_frozen_functional_transfer.py", "sha256": sha256(Path(__file__).resolve())},
        "outputs": {name: {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)} for name, path in paths.items() if path.exists()},
        "exit_code": 0,
    }
    paths["manifest"].write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
