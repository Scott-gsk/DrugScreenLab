#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-027: unified dual-route comparison under stratified endpoints.

Route A (direct drug sensitivity; predictions reused from disk, no retraining):
  A1 B0, A2 M1_source_scaled (representative), A3 M3   <- EXP-021 official runs
  A4 pathway_pca_best_reg                              <- EXP-024 gainwave
  A5 target_drugstd__mlp, A6 target_residual__mlp,
  A7 ref__mlp                                           <- EXP-026 ablation
Route B (perturbation mediation, zero-shot, no Kryzei response training):
  B1 lincs_reversal        <- EXP-025 features (deterministic)
  B2 chempert_reversal     <- EXP-027 upstream (3 seeds)
  B3 chempert_pathway_reversal <- EXP-027 upstream (3 seeds)

Endpoints: evaluate_stratified triple (nonstructural NDCG@5 primary,
per-drug cross-patient Spearman, full-panel reference). Pre-registered paired
patient bootstrap (2000x, seed=20260918):
  A2 vs A1, A4 vs A1, A5/A6 vs A7, B1/B2/B3 vs A1, A2 vs B2 (cross-route),
  per-drug Spearman macro vs 0 for A2/B1/B2/B3.

Outputs:
  artifacts/exp027_route_comparison/route_comparison_summary.json
  artifacts/exp027_route_comparison/bootstrap_route_comparison.json
  docs/reports/route_comparison_EXP027_20260920.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from drug_screen.evaluation.stratified_metrics import (
    evaluate_stratified,
    full_panel_patient_values,
    load_structural_drugs,
    nonstructural_patient_values,
)

ROOT = Path("/mnt/d/Code/DrugScreenLab")
STRUCTURAL_PATH = ROOT / "data/processed/identity/kryzei_structural_drugs_FROZEN.json"
SPLIT = ROOT / "data/splits/kryeziu_patient_outer5fold_FROZEN.json"
PROC = ROOT / "data/processed/transfer/mcrc_v3"
OUT_DIR = ROOT / "artifacts/exp027_route_comparison"
REPORT = ROOT / "docs/reports/route_comparison_EXP027_20260920.md"

OFFICIAL_RUNS = {
    "20260912": "run_b2000_20260918_174702",
    "20260913": "run_b2000_seed20260913_20260918_223357",
    "20260919": "run_b2000_20260919_20260919_121930",
}
GAINWAVE = ROOT / "artifacts/route1_stage2_gain_wave1_DRAFT"
ABLATION_PRED = ROOT / "artifacts/exp026_ablation/ablation_predictions.parquet"
LINCS_FEATS = ROOT / "artifacts/joint_feasibility_EXP025_DRAFT/kryzei_perturb_features.parquet"
CHEMPERT_FEATS = OUT_DIR / "chempert_kryzei_features.parquet"

BOOT_RS = 20260918
BOOT_N = 2000
ROUTE_A_REPR = "M1_source_scaled"


# ---------------------------------------------------------------- helpers
def normalize_official(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["algorithm"] = out["algorithm"].str.replace(r"_seed_\d+", "", regex=True)
    return out


def per_patient_scores(known: pd.DataFrame, structural) -> dict[str, dict]:
    ns = nonstructural_patient_values(known, structural)
    fp = full_panel_patient_values(known)
    ns_p = ns.rename("v").reset_index().groupby("level_0")["v"].mean()
    fp_p = fp.rename("v").reset_index().groupby("level_0")["v"].mean()
    return {"nonstructural": ns_p.to_dict(), "full_panel": fp_p.to_dict()}


def patient_bootstrap(diff: dict[str, float], seed: int = BOOT_RS,
                       resamples: int = BOOT_N) -> dict | None:
    vals = [v for v in diff.values() if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return None
    point = float(np.mean(vals))
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals)
    idx = rng.integers(0, len(vals), (resamples, len(vals)))
    draws = vals[idx].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"point": point, "low": float(low), "high": float(high),
            "n": len(vals), "resamples": resamples}


def load_tables():
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    samples = pd.read_parquet(PROC / "samples.parquet")
    pairs = pd.read_parquet(PROC / "pairs.parquet")
    samples = samples.copy()
    samples["sample_id"] = samples["sample_id"].astype(str)
    samples["patient_id"] = samples["patient_id"].astype(str)
    pairs = pairs.copy()
    pairs["sample_id"] = pairs["sample_id"].astype(str)
    pairs["drug_id"] = pairs["drug_id"].astype(str)
    merged = (pairs.merge(samples[["sample_id", "patient_id"]], on="sample_id",
                          how="inner").dropna(subset=["response"])
              .rename(columns={"response": "y"}))
    # restrict to split patients (EXP-025 convention)
    all_p: set[str] = set()
    for fold in split["folds"]:
        all_p |= set(map(str, fold.get("train_patient_ids")
                         or fold.get("train_patients") or []))
        all_p |= set(map(str, fold.get("test_patient_ids")
                         or fold.get("test_patients") or []))
    merged = merged[merged["patient_id"].isin(all_p)].copy()
    # patient -> outer test fold
    p2fold = {}
    for fold in split["folds"]:
        f_id = int(fold.get("fold", fold.get("fold_id")))
        for p in (fold.get("test_patient_ids") or fold.get("test_patients") or []):
            p2fold[str(p)] = f_id
    return merged, p2fold


def route_b_frame(feats: pd.DataFrame, score_col: str, arm: str,
                  merged: pd.DataFrame, p2fold: dict,
                  known_set: frozenset[str], structural) -> pd.DataFrame:
    """Zero-shot perturbation features -> standard long table (known drugs)."""
    f = feats.copy()
    f["sample_id"] = f["sample_id"].astype(str)
    f["drug_id"] = f["drug_id"].astype(str)
    known = merged[merged["drug_id"].isin(known_set)]
    j = f.merge(known[["sample_id", "drug_id", "patient_id", "y"]],
                on=["sample_id", "drug_id"], how="inner")
    out = pd.DataFrame({
        "patient_id": j["patient_id"], "model_id": j["sample_id"],
        "drug_id": j["drug_id"],
        "fold": j["patient_id"].map(p2fold),
        "algorithm": arm, "observed": j["y"], "predicted": j[score_col],
        "source_drug_status": "known",
    })
    if out["fold"].isna().any():
        raise RuntimeError("route B rows without fold assignment")
    out.attrs["seed"] = (list(sorted(map(int, feats["seed"].unique())))
                         if "seed" in feats.columns else None)
    return out


def macro_per_drug_spearman(known: pd.DataFrame,
                            min_patients: int = 5) -> float | None:
    per_patient = (known.groupby(["drug_id", "patient_id"], sort=True)
                   .agg(observed=("observed", "mean"),
                        predicted=("predicted", "mean")).reset_index())
    vals = []
    for _, g in per_patient.groupby("drug_id", sort=True):
        y = g["observed"].to_numpy(float)
        p = g["predicted"].to_numpy(float)
        if len(y) < min_patients or np.ptp(y) == 0 or np.ptp(p) == 0:
            continue
        rho = spearmanr(y, p).statistic
        if np.isfinite(rho):
            vals.append(float(rho))
    return float(np.mean(vals)) if vals else None


def spearman_macro_bootstrap(known: pd.DataFrame, seed: int = BOOT_RS,
                             resamples: int = BOOT_N,
                             min_patients: int = 5) -> dict | None:
    """Patient-level bootstrap of the per-drug Spearman macro (vs 0)."""
    per_patient = (known.groupby(["drug_id", "patient_id"], sort=True)
                   .agg(observed=("observed", "mean"),
                        predicted=("predicted", "mean")).reset_index())
    drugs = sorted(per_patient["drug_id"].unique())
    tables = {d: per_patient[per_patient["drug_id"] == d].set_index("patient_id")
              for d in drugs}
    patients = sorted(per_patient["patient_id"].unique())
    point = macro_per_drug_spearman(known, min_patients)
    if point is None:
        return None
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        pick = rng.choice(patients, size=len(patients), replace=True)
        vals = []
        for d, tb in tables.items():
            sub = tb.reindex(pick).dropna()
            if len(sub) < min_patients:
                continue
            y = sub["observed"].to_numpy(float)
            p = sub["predicted"].to_numpy(float)
            if np.ptp(y) == 0 or np.ptp(p) == 0:
                continue
            rho = spearmanr(y, p).statistic
            if np.isfinite(rho):
                vals.append(float(rho))
        if vals:
            draws.append(float(np.mean(vals)))
    if len(draws) < resamples // 2:
        return None
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"point": point, "low": float(low), "high": float(high),
            "n_draws_valid": len(draws), "resamples": resamples}


def fmt(x, sd=None):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{x:.4f}"
    if sd is not None and not np.isnan(sd):
        s += f" ±{sd:.4f}"
    return s


# ---------------------------------------------------------------- main
def main() -> None:
    structural = load_structural_drugs(STRUCTURAL_PATH)
    known_set = structural.structural | structural.nonstructural
    merged, p2fold = load_tables()
    results = []      # arm-level rows
    patient_ns = {}   # arm -> {patient -> mean-over-seeds nonstructural value}

    # ============ Route A: EXP-021 official runs ============
    for seed, run in OFFICIAL_RUNS.items():
        run_dir = (ROOT / "artifacts/route1_stage2_official_transfer_kryeziu_DRAFT"
                   / run)
        df = normalize_official(pd.read_parquet(
            run_dir / "outer_oof_predictions.parquet"))
        for alg in ["B0", "M1", "M1_source_scaled", "M2", "M3"]:
            sub = df[(df.source_drug_status == "known") & (df.algorithm == alg)]
            summary, _, _ = evaluate_stratified(df, structural, algorithm=alg)
            results.append({"route": "A", "source": "EXP021_official",
                            "arm": alg, "seed": seed,
                            **{k: v for k, v in summary.as_dict().items()
                               if k != "algorithm"}})
            pat = per_patient_scores(sub, structural)
            for pid, v in pat["nonstructural"].items():
                patient_ns.setdefault(alg, {}).setdefault(pid, []).append(v)

    # ============ Route A: EXP-024 gainwave pathway arm ============
    pcr_rows = []
    for pq in sorted((GAINWAVE / "pathway_pca_best_reg").glob(
            "pathway_pca_wd02_seed*_predictions.parquet")):
        name = pq.name.replace("_predictions.parquet", "")
        seed = name.split("_seed")[1].split("_")[0]
        df = pd.read_parquet(pq)
        df = df[df["drug_id"].isin(known_set)].copy()
        norm = pd.DataFrame({
            "patient_id": df["patient_id"], "model_id": df["sample_id"],
            "drug_id": df["drug_id"], "fold": df["fold"],
            "algorithm": "pathway_pca_best_reg", "observed": df["y"],
            "predicted": df["pred"], "source_drug_status": "known"})
        summary, _, _ = evaluate_stratified(norm, structural,
                                            algorithm="pathway_pca_best_reg")
        results.append({"route": "A", "source": "EXP024_gainwave",
                        "arm": "pathway_pca_best_reg", "seed": seed,
                        **{k: v for k, v in summary.as_dict().items()
                           if k != "algorithm"}})
        pat = per_patient_scores(norm, structural)
        for pid, v in pat["nonstructural"].items():
            patient_ns.setdefault("pathway_pca_best_reg", {}).setdefault(
                pid, []).append(v)
        pcr_rows.append((seed, norm))

    # ============ Route A: EXP-026 ablation arms ============
    abl = pd.read_parquet(ABLATION_PRED)
    abl_arms = {
        "target_drugstd__mlp": "A5_target_drugstd",
        "target_residual__mlp": "A6_target_residual",
        "ref__mlp": "A7_ref_mlp",
        "B0__ridge": "A1_ablation_B0",
    }
    for alg_key, arm in abl_arms.items():
        sub = abl[abl["algorithm"] == alg_key].copy()
        seeds = sorted(sub["seed"].unique())
        for seed in seeds:
            s = sub[sub["seed"] == seed]
            summary, _, _ = evaluate_stratified(s, structural)
            results.append({"route": "A", "source": "EXP026_ablation",
                            "arm": arm, "seed": str(seed),
                            **{k: v for k, v in summary.as_dict().items()
                               if k != "algorithm"}})
            pat = per_patient_scores(s, structural)
            for pid, v in pat["nonstructural"].items():
                patient_ns.setdefault(arm, {}).setdefault(pid, []).append(v)

    # ============ Route B ============
    lincs = pd.read_parquet(LINCS_FEATS)
    b1 = route_b_frame(lincs, "reversal_score", "B1_lincs_reversal",
                       merged, p2fold, known_set, structural)
    summary, _, _ = evaluate_stratified(b1, structural)
    results.append({"route": "B", "source": "EXP025_lincs", "arm": "B1_lincs_reversal",
                    "seed": "single",
                    **{k: v for k, v in summary.as_dict().items() if k != "algorithm"}})
    for pid, v in per_patient_scores(b1, structural)["nonstructural"].items():
        patient_ns.setdefault("B1_lincs_reversal", {})[pid] = [v]

    chempert = pd.read_parquet(CHEMPERT_FEATS)
    for feat_col, arm in [("reversal_score", "B2_chempert_reversal"),
                          ("pathway_reversal", "B3_chempert_pathway_reversal")]:
        for seed in sorted(chempert["seed"].unique()):
            fs = chempert[chempert["seed"] == seed]
            b = route_b_frame(fs, feat_col, arm, merged, p2fold,
                              known_set, structural)
            summary, _, _ = evaluate_stratified(b, structural)
            results.append({"route": "B", "source": "EXP027_chempert", "arm": arm,
                            "seed": str(seed),
                            **{k: v for k, v in summary.as_dict().items()
                               if k != "algorithm"}})
            for pid, v in per_patient_scores(b, structural)["nonstructural"].items():
                patient_ns.setdefault(arm, {}).setdefault(pid, []).append(v)

    frame = pd.DataFrame(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_json(OUT_DIR / "route_comparison_summary.json", orient="records",
                  indent=2, force_ascii=False)

    # ============ aggregate ============
    agg = (frame.groupby(["route", "source", "arm"])
           .agg(ns_mean=("nonstructural_ndcg5_macro", "mean"),
                ns_sd=("nonstructural_ndcg5_macro", "std"),
                pd_mean=("per_drug_patient_spearman_macro", "mean"),
                pd_sd=("per_drug_patient_spearman_macro", "std"),
                fp_mean=("full_panel_ndcg5_macro", "mean"),
                fp_sd=("full_panel_ndcg5_macro", "std"),
                n_seeds=("seed", "nunique"))
           .reset_index().sort_values(["route", "ns_mean"], ascending=[True, False]))

    # seed-averaged patient-level values
    pat_avg = {arm: {pid: float(np.mean(v)) for pid, v in d.items()
                     if v and np.isfinite(np.mean(v))}
               for arm, d in patient_ns.items()}

    def paired(arm_x: str, arm_y: str) -> dict | None:
        """bootstrap of mean(patient_ns[x] - patient_ns[y]) on intersection."""
        common = set(pat_avg.get(arm_x, {})) & set(pat_avg.get(arm_y, {}))
        diff = {p: pat_avg[arm_x][p] - pat_avg[arm_y][p] for p in common}
        return patient_bootstrap(diff) if diff else None

    boot = {
        "A2_M1_source_scaled_vs_A1_B0": paired(ROUTE_A_REPR, "B0"),
        "A4_pathway_pca_vs_A1_B0": paired("pathway_pca_best_reg", "B0"),
        "A5_target_drugstd_vs_A7_ref": paired("A5_target_drugstd", "A7_ref_mlp"),
        "A6_target_residual_vs_A7_ref": paired("A6_target_residual", "A7_ref_mlp"),
        "B1_lincs_vs_A1_B0": paired("B1_lincs_reversal", "B0"),
        "B2_chempert_vs_A1_B0": paired("B2_chempert_reversal", "B0"),
        "B3_chempert_pathway_vs_A1_B0": paired("B3_chempert_pathway_reversal", "B0"),
        "CROSS_ROUTE_A2_vs_B2": paired(ROUTE_A_REPR, "B2_chempert_reversal"),
    }

    # per-drug Spearman macro vs 0 (patient bootstrap)
    sp_boot = {}
    for label, frame_arm in [
            ("A2_M1_source_scaled",
             None),  # filled below from official runs (seed-merged)
            ("B1_lincs_reversal", b1)]:
        if frame_arm is not None:
            sp_boot[label] = spearman_macro_bootstrap(frame_arm)
    # A2: merge official seeds into one frame
    a2_frames = []
    for seed, run in OFFICIAL_RUNS.items():
        run_dir = (ROOT / "artifacts/route1_stage2_official_transfer_kryeziu_DRAFT"
                   / run)
        df = normalize_official(pd.read_parquet(
            run_dir / "outer_oof_predictions.parquet"))
        a2_frames.append(df[(df.source_drug_status == "known")
                            & (df.algorithm == ROUTE_A_REPR)])
    a2_all = pd.concat(a2_frames, ignore_index=True)
    sp_boot["A2_M1_source_scaled"] = spearman_macro_bootstrap(a2_all)
    # B2: merge chempert seeds
    b2_frames = [route_b_frame(chempert[chempert["seed"] == s],
                              "reversal_score", "B2_chempert_reversal",
                              merged, p2fold, known_set, structural)
                 for s in sorted(chempert["seed"].unique())]
    sp_boot["B2_chempert_reversal"] = spearman_macro_bootstrap(
        pd.concat(b2_frames, ignore_index=True))
    b3_frames = [route_b_frame(chempert[chempert["seed"] == s],
                              "pathway_reversal", "B3_chempert_pathway_reversal",
                              merged, p2fold, known_set, structural)
                 for s in sorted(chempert["seed"].unique())]
    sp_boot["B3_chempert_pathway_reversal"] = spearman_macro_bootstrap(
        pd.concat(b3_frames, ignore_index=True))
    boot["per_drug_spearman_macro_vs0"] = sp_boot

    (OUT_DIR / "bootstrap_route_comparison.json").write_text(
        json.dumps(boot, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")

    # ============ per-drug stratification ============
    per_drug_rows = []
    # observed cross-patient SD per drug (known drugs)
    obs_sd = (merged[merged["drug_id"].isin(known_set)]
              .groupby("drug_id")["y"].std())
    # per-drug spearman for each main arm
    per_drug_tables = {"A2_M1_source_scaled": a2_all}
    b2_all = pd.concat(b2_frames, ignore_index=True)
    per_drug_tables["B1_lincs_reversal"] = b1
    per_drug_tables["B2_chempert_reversal"] = b2_all
    per_drug_tables["B3_chempert_pathway_reversal"] = pd.concat(
        b3_frames, ignore_index=True)
    for arm, fr in per_drug_tables.items():
        per_patient = (fr.groupby(["drug_id", "patient_id"], sort=True)
                       .agg(observed=("observed", "mean"),
                            predicted=("predicted", "mean")).reset_index())
        for drug, g in per_patient.groupby("drug_id", sort=True):
            y = g["observed"].to_numpy(float)
            p = g["predicted"].to_numpy(float)
            if len(y) < 5 or np.ptp(y) == 0 or np.ptp(p) == 0:
                rho = float("nan")
            else:
                rho = float(spearmanr(y, p).statistic)
            per_drug_rows.append({"arm": arm, "drug_id": drug,
                                  "spearman": rho,
                                  "n_patients": int(len(g))})
    pd_frame = pd.DataFrame(per_drug_rows)
    pivot = (pd_frame.pivot(index="drug_id", columns="arm", values="spearman")
             .assign(obs_sd=obs_sd)
             .assign(structural=lambda d: [
                 "structural" if i in structural.structural else "nonstructural"
                 for i in d.index])
             .sort_values(["structural", "obs_sd"], ascending=[True, False]))
    pivot.reset_index().to_json(OUT_DIR / "per_drug_stratification.json",
                                orient="records", indent=2, force_ascii=False)

    # ============ report ============
    lines = [
        "# EXP-027 双路线统一对比报告（路线 A 直接药敏 vs 路线 B 扰动中介）",
        "",
        "- 生成：scripts/evaluation/exp027_route_comparison.py（EXP-027）",
        f"- 终点：非结构药子集 NDCG@5（主）/ 逐药跨患者 Spearman（个性化）/ "
        "全面板 NDCG@5（参照）",
        "- 路线 A 全部复用已落盘预测；路线 B 零样本（不使用 Kryzei 药敏标签）",
        "",
        "## 一、双路线总表（多种子 mean±SD；患者宏平均）",
        "",
        "| 路线 | 臂 | 非结构 NDCG@5 | 逐药跨患者 Spearman | 全面板参照 | 种子数 |",
        "|---|---|---|---|---|---|",
    ]
    order = {"A1_B0": 0, "A2": 1}
    for _, r in agg.iterrows():
        lines.append(
            f"| {r['route']} | {r['arm']} | {fmt(r.ns_mean, r.ns_sd)} | "
            f"{fmt(r.pd_mean, r.pd_sd)} | {fmt(r.fp_mean, r.fp_sd)} | "
            f"{int(r.n_seeds)} |")
    lines += [
        "",
        "## 二、预登记配对比较（患者级 2000 次 bootstrap，非结构 NDCG@5）",
        "",
        "| 比较 | 点估计 | 95% CI | n |",
        "|---|---|---|---|",
    ]
    for key in ["A2_M1_source_scaled_vs_A1_B0", "A4_pathway_pca_vs_A1_B0",
                "A5_target_drugstd_vs_A7_ref", "A6_target_residual_vs_A7_ref",
                "B1_lincs_vs_A1_B0", "B2_chempert_vs_A1_B0",
                "B3_chempert_pathway_vs_A1_B0", "CROSS_ROUTE_A2_vs_B2"]:
        b = boot.get(key)
        if b is None:
            lines.append(f"| {key} | 不可算 | — | — |")
        else:
            lines.append(f"| {key} | {b['point']:+.4f} | "
                         f"[{b['low']:+.4f}, {b['high']:+.4f}] | {b['n']} |")
    lines += [
        "",
        "## 三、逐药跨患者 Spearman 宏平均 vs 0（患者级 bootstrap）",
        "",
        "| 臂 | 宏平均 | 95% CI |",
        "|---|---|---|",
    ]
    for k, b in sp_boot.items():
        if b is None:
            lines.append(f"| {k} | 不可算 | — |")
        else:
            lines.append(f"| {k} | {b['point']:+.4f} | "
                         f"[{b['low']:+.4f}, {b['high']:+.4f}] |")
    lines += [
        "",
        "## 四、按药分层（14 已知药；Spearman 为跨患者个性化度量）",
        "",
        "| 药物 | 结构性 | 观测跨患者 SD | A2 Spearman | B1 | B2 | B3 |",
        "|---|---|---|---|---|---|---|",
    ]
    for drug_id, r in pivot.iterrows():
        lines.append(
            f"| {drug_id} | {r['structural']} | {r['obs_sd']:.2f} | "
            f"{fmt(r.get('A2_M1_source_scaled'))} | "
            f"{fmt(r.get('B1_lincs_reversal'))} | "
            f"{fmt(r.get('B2_chempert_reversal'))} | "
            f"{fmt(r.get('B3_chempert_pathway_reversal'))} |")
    lines += [
        "",
        "## 五、结果判读",
        "",
        "（待执行后回填；判读框架见 experiments/records/EXP-027.md 预登记。）",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("REPORT ->", REPORT)
    print(agg.to_string())
    print("\nBOOTSTRAP:")
    for k, v in boot.items():
        if k == "per_drug_spearman_macro_vs0":
            for kk, vv in v.items():
                print(" ", kk, vv)
        else:
            print(" ", k, v)


if __name__ == "__main__":
    main()
