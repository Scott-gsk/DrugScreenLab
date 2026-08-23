"""Frozen breast-PDO validation and domain diagnostics for EXP-010."""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from drug_screen.evaluation.phase1_prism import spearman

TORG_ORDER = ("TORG40", "TORG51", "TORG58", "TORG139", "TORG146", "TORG43", "TORG155")
TORG_DEATH = (1.19, 1.14, 1.14, 1.48, 3.06, 1.58, 1.59)
TORG_AREA = (1.11, 0.60, 0.80, 1.24, 0.38, 1.03, 0.68)


def gse303509_observed_scores() -> dict[str, float]:
    return {sample: death / area for sample, death, area in zip(TORG_ORDER, TORG_DEATH, TORG_AREA, strict=True)}


def gse303509_rank_validation(predictions: Mapping[str, float]) -> dict[str, object]:
    if set(predictions) != set(TORG_ORDER):
        raise ValueError("GSE303509 evaluation requires the frozen seven-TORG cohort")
    observed_map = gse303509_observed_scores()
    predicted = np.asarray([predictions[x] for x in TORG_ORDER], dtype=float)
    observed = np.asarray([observed_map[x] for x in TORG_ORDER], dtype=float)
    rho = spearman(predicted, observed)
    concordant = int(sum(bool((predicted[i] - predicted[j]) * (observed[i] - observed[j]) > 0)
                         for i, j in combinations(range(7), 2)))
    if rho is not None and rho >= 0.30 and concordant >= 14:
        status = "SUPPORTS"
    elif (rho is not None and rho <= 0) or concordant <= 10:
        status = "DOES_NOT_SUPPORT"
    else:
        status = "INCONCLUSIVE"
    return {"format": "exp010_gse303509_rank_v1", "status": status, "cohort_size": 7,
            "spearman": rho, "pairwise_concordant": concordant, "pairwise_total": 21,
            "claim_scope": "cisplatin-containing combination directional concordance only",
            "rows": [{"torg_id": x, "predicted_score": float(predictions[x]), "observed_death_area_ratio": observed_map[x]} for x in TORG_ORDER]}


def gse287429_contrast_manifest(sample_table: pd.DataFrame) -> dict[str, object]:
    required = {"sample_id", "treatment", "time_hours", "replicate"}
    if not required.issubset(sample_table):
        raise ValueError(f"missing sample metadata: {sorted(required - set(sample_table))}")
    contrasts = []
    for drug in ("DOX", "BTZ"):
        for time in (23, 47):
            treated = sample_table.query("treatment == @drug and time_hours == @time")
            control = sample_table.query("treatment == 'CONTROL' and time_hours == @time")
            if len(treated) < 2 or len(control) < 2:
                raise ValueError(f"contrast {drug} {time}h requires >=2 treated and controls")
            contrasts.append({"contrast_id": f"{drug}_{time}h_vs_CONTROL_{time}h", "treatment": drug,
                              "time_hours": time, "treated_samples": sorted(treated.sample_id.astype(str)),
                              "control_samples": sorted(control.sample_id.astype(str))})
    return {"format": "exp010_gse287429_contrasts_v1", "interpretation": "perturbation pathway response, not drug sensitivity", "contrasts": contrasts}


def reactome_overrepresentation(ranked_genes: Sequence[str], gene_sets: Mapping[str, Sequence[str]], *,
                                top_n: int = 200, universe: Sequence[str] | None = None) -> dict[str, object]:
    """Response-blind Reactome overrepresentation on a precomputed gene ranking."""
    ranked = list(dict.fromkeys(map(str, ranked_genes)))
    background = set(map(str, universe)) if universe is not None else set(ranked)
    selected = set(ranked[: min(top_n, len(ranked))]) & background
    if not selected or len(background) < 2:
        raise ValueError("pathway analysis requires selected genes and a non-trivial universe")
    rows = []
    for pathway, members in sorted(gene_sets.items()):
        member_set = set(map(str, members)) & background
        overlap = selected & member_set
        if not overlap:
            continue
        pvalue = float(hypergeom.sf(len(overlap) - 1, len(background), len(member_set), len(selected)))
        rows.append({"pathway": str(pathway), "overlap_count": len(overlap), "pathway_size": len(member_set),
                     "selected_count": len(selected), "pvalue": pvalue, "overlap_genes": sorted(overlap)})
    # Benjamini-Hochberg correction across all tested pathways, including zero-overlap sets.
    tested = max(1, len(gene_sets))
    order = sorted(range(len(rows)), key=lambda i: rows[i]["pvalue"])
    running = 1.0
    for reverse_rank, index in enumerate(reversed(order), start=1):
        rank = len(order) - reverse_rank + 1
        running = min(running, rows[index]["pvalue"] * tested / rank)
        rows[index]["fdr_bh"] = float(running)
    rows.sort(key=lambda row: (row["fdr_bh"], row["pvalue"], row["pathway"]))
    return {"format": "exp010_reactome_ora_v1", "top_n": top_n, "universe_size": len(background),
            "tested_pathway_count": len(gene_sets), "rows": rows,
            "interpretation": "perturbation pathway response, not drug sensitivity"}


def gse276609_domain_shift(reference: pd.DataFrame, target: pd.DataFrame, *, n_components: int = 10,
                          bootstrap_repeats: int = 500, seed: int = 20260823) -> dict[str, object]:
    """Report symmetric centroid distance after joint gene-wise standardization/PCA."""
    genes = sorted(set(reference.columns) & set(target.columns))
    if len(genes) < 2 or min(len(reference), len(target)) < 2:
        raise ValueError("domain diagnostic needs >=2 shared genes and samples per domain")
    matrix = np.vstack([reference[genes].to_numpy(float), target[genes].to_numpy(float)])
    mean, std = matrix.mean(axis=0), matrix.std(axis=0)
    keep = std > 0
    matrix = (matrix[:, keep] - mean[keep]) / std[keep]
    _, _, vt = np.linalg.svd(matrix - matrix.mean(axis=0), full_matrices=False)
    scores = matrix @ vt[: min(n_components, len(vt))].T
    split = len(reference)
    distance = float(np.linalg.norm(scores[:split].mean(axis=0) - scores[split:].mean(axis=0)))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(bootstrap_repeats):
        left = scores[rng.integers(0, split, split)]
        right = scores[split + rng.integers(0, len(target), len(target))]
        boot.append(float(np.linalg.norm(left.mean(axis=0) - right.mean(axis=0))))
    return {"format": "exp010_gse276609_domain_shift_v1", "shared_gene_count": len(genes),
            "reference_samples": len(reference), "target_samples": len(target), "centroid_distance": distance,
            "bootstrap_repeats": bootstrap_repeats, "seed": seed,
            "centroid_distance_ci_95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "interpretation": "domain shift only; no drug-sensitivity claim"}


def multiregional_public_gate(tables: Sequence[pd.DataFrame], *, required_drugs: Sequence[str]) -> dict[str, object]:
    columns = set().union(*(set(table.columns) for table in tables)) if tables else set()
    identity = {"patient_id", "pdo_id"}.issubset(columns)
    response = any(name in columns for name in ("ic50", "auc", "dss", "drug_response")) and "drug_id" in columns
    expression = bool({"gene_id", "expression"}.issubset(columns) or "expression_path" in columns)
    drug_values = set()
    for table in tables:
        if "drug_id" in table:
            drug_values.update(table["drug_id"].dropna().astype(str))
    drugs_ok = set(required_drugs).issubset(drug_values)
    ready = identity and response and expression and drugs_ok
    return {"format": "exp010_multiregional_public_gate_v1", "status": "EVALUABLE" if ready else "NOT_EVALUABLE_PUBLIC_MATRIX",
            "checks": {"patient_pdo_identity": identity, "response_matrix": response, "expression_features": expression,
                       "all_required_drugs": drugs_ok}, "required_drugs": list(required_drugs)}
