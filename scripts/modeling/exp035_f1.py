#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-035 F1：控制疾病构成的域差异分析（工作包 F 预登记）。

只读诊断 + 描述性分析；遵守 EXP-035.md 预登记约束：
- 域分类报 平衡准确率 + 多数类基线 + 置换基线（修正审计发现 13）。
- 表示重建比例只称「该模型下的重构方差贡献」（修正审计发现 12）。
- 距离/误差相关只在有明确关联时才陈述；相关不写因果。
- 疾病/平台/培养体系完全混杂处明确「无法独立识别原因」。

分析量：分布距离、最近邻、域分类、表示重建（PCA 共享/私有分解）、预测误差关系。
配套对照：泛癌来源 vs 同癌种来源、等来源样本量 bootstrap、同癌种跨研究细胞系。

基因空间 = GDSC∩Kryzei∩biobank = 14,900 共享基因，域内逐基因 z-score（EXP-029 口径），
距离 = 1 − Pearson 相关。样本集：mCRC 类器官 = Kryzei 白名单 60；
同癌种细胞系 = GDSC COREAD（结直肠腺癌）；泛癌 = GDSC 全部 963；
同癌种跨研究细胞系 = DepMap 24Q2 RNA-seq 结直肠系；类器官库 = biobank 255。

域分类效率口径：特征先做 标准化 + PCA(50)（无监督，不含标签），再在得到的
50 维主成分上做逻辑回归 5 折交叉验证；置换基线与等样本量 bootstrap 仅重拟合逻辑回归。
输出：artifacts/exp035_f1/f1_summary.json（报告 f1_report.md 另写）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from drug_screen.transfer.cohorts import load_registered_cohort, sha256
from drug_screen.evaluation.exp031_eval import holm_correct

PROJ = Path("/mnt/d/Code/DrugScreenLab")
OUT_DIR = PROJ / "artifacts/exp035_f1"
BIOBANK_CSV = PROJ / ("data/raw/transfer_public_assets/organoid_biobank/"
                      "data_RNAseq_ALL-organoids.csv")
BIOBANK_SHA = "92859e18417c3737956236e6990f8001a0d2cc352890b9355b7ed9603b6a6b1e"
DEPMAP_RNA = PROJ / "data/raw/depmap/24q2/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
DEPMAP_MODEL = PROJ / "data/raw/depmap/24q2/Model.csv"
AUDIT_JSON = PROJ / "data/processed/identity/kryzei_admission_audit_20260921.json"
EXP032_PRED = PROJ / "artifacts/exp032_l3l4/predictions.parquet"
EXP032_L3 = PROJ / "artifacts/exp032_l3l4/l3_summary.json"

SEED_CV = 20260921
SEED_BOOT = 20260918
N_BOOT = 2000
N_PERM = 100
N_EQUAL_BOOT = 200
PCA_N = 50
SHARED_PC = 64
PRIVATE_PC = 32
FULL_PC = SHARED_PC + PRIVATE_PC


def log(msg):
    print(msg, flush=True)


def zscore(mat: np.ndarray) -> np.ndarray:
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (mat - mu) / sd


def fill_nan(mat: np.ndarray):
    if not np.isnan(mat).any():
        return mat, 0
    with np.errstate(all="ignore"):
        med = np.nanmedian(mat, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isnan(mat), med[None, :], mat), int(np.isnan(mat).any(axis=0).sum())


def corr_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 - Pearson；返回 (n_b, n_a)：行 = 第二参数样本。"""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = b @ a.T
    den = np.linalg.norm(b, axis=1)[:, None] * np.linalg.norm(a, axis=1)[None, :]
    return 1.0 - num / np.clip(den, 1e-12, None)


def spearman_p(a, b):
    r = spearmanr(a, b)
    return float(r.statistic), float(r.pvalue)


def project(X: np.ndarray, n_components: int = PCA_N) -> np.ndarray:
    """标准化 + PCA（无监督，不含类别标签），返回主成分矩阵。"""
    Xs = StandardScaler().fit_transform(X)
    return PCA(n_components=n_components, random_state=0).fit_transform(Xs)


def _clf():
    return LogisticRegression(max_iter=3000, class_weight="balanced")


def logreg_ba_cv(Xp: np.ndarray, y: np.ndarray, seed: int) -> float:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return float(cross_val_score(
        _clf(), Xp, y, cv=skf, scoring="balanced_accuracy").mean())


def logreg_confusion_cv(Xp: np.ndarray, y: np.ndarray, seed: int):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = cross_val_predict(_clf(), Xp, y, cv=skf)
    return confusion_matrix(y, pred, labels=[0, 1]).tolist()


def classify_report(X: np.ndarray, y: np.ndarray, *, label: str,
                    n_perm: int = N_PERM):
    Xp = project(X)
    ba = logreg_ba_cv(Xp, y, SEED_CV)
    cm = logreg_confusion_cv(Xp, y, SEED_CV)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    majority_acc = max(n0, n1) / len(y)
    perm_ba = []
    rng = np.random.default_rng(SEED_CV)
    for _ in range(n_perm):
        yp = rng.permutation(y)
        perm_ba.append(logreg_ba_cv(Xp, yp, SEED_CV))
    perm_ba = np.asarray(perm_ba)
    return {
        "label": label,
        "n_class0": n0, "n_class1": n1,
        "feature": "StandardScaler + PCA(50) (unsupervised) + LogisticRegression(class_weight=balanced), 5-fold stratified",
        "balanced_accuracy": ba,
        "majority_baseline_balanced_acc": 0.5,   # 常数多数类预测 -> 平衡准确率恒为 (1+0)/2
        "majority_baseline_accuracy": majority_acc,
        "permutation_baseline": {
            "n_perm": n_perm,
            "mean": float(perm_ba.mean()),
            "p95": float(np.percentile(perm_ba, 95)),
            "p_observed": float((perm_ba >= ba).mean()),
        },
        "confusion_matrix": cm,
    }


def r2(x: np.ndarray, recon: np.ndarray) -> float:
    sse = float(((x - recon) ** 2).sum())
    sst = float(((x - x.mean(axis=0, keepdims=True)) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


def q(a):
    a = np.asarray(a, dtype=float)
    return {"median": float(np.median(a)),
            "q25": float(np.percentile(a, 25)),
            "q75": float(np.percentile(a, 75))}


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summ: dict = {"experiment_id": "EXP-035", "task": "F1", "status": "RAN"}

    log("[exp035_f1] load cohorts")
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    wl_patients = set(audit["patient_whitelist"])
    wl_samples = set(audit["pdo_whitelist"])

    gdsc = load_registered_cohort(
        "gdsc2_v2", root=PROJ / "data",
        registry=PROJ / "data/registry/exp015_source.json")
    g_expr = np.asarray(gdsc.expression, dtype=float)
    g_genes = [str(x) for x in gdsc.gene_ids]
    g_sid = [str(x) for x in gdsc.sample_ids]
    g_tissue = pd.Series(
        [str(x) if not pd.isna(x) else "" for x in gdsc.samples["sample_tissue"]],
        index=[str(x) for x in gdsc.samples["sample_id"]]).reindex(g_sid)
    crc_sids = [s for s in g_sid if g_tissue.get(s) == "COREAD"]
    li_sids = [s for s in g_sid if g_tissue.get(s) == "LARGE_INTESTINE"]

    kry = load_registered_cohort(
        "kryeziu_mcrc_v3", root=PROJ,
        registry=PROJ / "data/registry/transfer_cohorts.json")
    k_expr = np.asarray(kry.expression, dtype=float)
    k_genes = [str(x) for x in kry.gene_ids]
    k_sid = [str(x) for x in kry.sample_ids]
    k_pos = {s: i for i, s in enumerate(k_sid)}
    k_wl_idx = [k_pos[s] for s in wl_samples if s in k_pos]

    if sha256(BIOBANK_CSV) != BIOBANK_SHA:
        raise ValueError("biobank raw csv sha mismatch")
    bio = pd.read_csv(BIOBANK_CSV, index_col=0)
    bio.columns = [str(c)[:-4] for c in bio.columns]
    b_genes = [str(c) for c in bio.columns]
    b_expr = bio.to_numpy(float)
    b_ids = [str(s) for s in bio.index]

    log("[exp035_f1] gene space")
    inter = sorted(set(g_genes) & set(k_genes) & set(b_genes))
    if len(inter) != 14900:
        raise AssertionError(f"expected 14900 intersection genes, got {len(inter)}")
    g_idx = {g: i for i, g in enumerate(g_genes)}
    k_idx = {g: i for i, g in enumerate(k_genes)}
    b_idx = {g: i for i, g in enumerate(b_genes)}
    gi = np.array([g_idx[g] for g in inter])
    ki = np.array([k_idx[g] for g in inter])
    bi = np.array([b_idx[g] for g in inter])

    g_int_fill, g_fill_cols = fill_nan(g_expr[:, gi])
    g_z = zscore(g_int_fill)
    k_wl_expr = k_expr[k_wl_idx][:, ki]
    k_z = zscore(k_wl_expr)
    b_z = zscore(b_expr[:, bi])

    g_pos = {s: i for i, s in enumerate(g_sid)}
    crc_i = np.array([g_pos[s] for s in crc_sids])
    g_crc_z = g_z[crc_i]

    d_gg = corr_dist(g_z, g_z)
    np.fill_diagonal(d_gg, np.inf)
    g_nn_min = d_gg.min(axis=1)
    thresh = float(np.percentile(g_nn_min, 95))

    summ["gene_space"] = {
        "intersection_genes": len(inter),
        "gdsc_genes": len(g_genes), "kryzei_genes": len(k_genes),
        "biobank_genes": len(b_genes),
        "gdsc_intersection_cols_nan_filled": g_fill_cols,
        "zscore": "within-domain per-gene z on 14900 shared genes (EXP-029 convention)",
        "distance": "1 - Pearson correlation",
        "no_neighbor_threshold_p95_within_gdsc": thresh,
        "exp029_threshold_reference": 0.6703208316919416,
    }
    summ["sample_sets"] = {
        "kryzei_whitelist_samples": int(len(k_wl_idx)),
        "kryzei_whitelist_patients": int(len(wl_patients)),
        "gdsc_all": int(len(g_sid)),
        "gdsc_crc_coread": int(len(crc_sids)),
        "gdsc_large_intestine": int(len(li_sids)),
        "biobank": int(len(b_ids)),
        "evidence_gdsc_tissue": "sample_tissue column of gdsc2_v2 (EXP-015 source) samples.parquet; COREAD = 结直肠腺癌",
    }

    log("[exp035_f1] distance + nearest neighbor")
    d_k_all = corr_dist(g_z, k_z)
    d_k_crc = corr_dist(g_crc_z, k_z)
    k_min_all = d_k_all.min(axis=1)
    k_min_crc = d_k_crc.min(axis=1)
    d_b_all = corr_dist(g_z, b_z)
    b_min_all = d_b_all.min(axis=1)
    d_crc_crc = corr_dist(g_crc_z, g_crc_z)
    np.fill_diagonal(d_crc_crc, np.inf)
    crc_nn_min = d_crc_crc.min(axis=1)

    summ["distance"] = {
        "kryzei_to_nearest_gdsc_all": q(k_min_all),
        "kryzei_to_nearest_gdsc_crc": q(k_min_crc),
        "within_gdsc_nn": q(g_nn_min),
        "within_gdsc_crc_nn": q(crc_nn_min),
        "biobank_to_nearest_gdsc_all": q(b_min_all),
    }
    summ["nearest_neighbor"] = {
        "kryzei_no_neighbor_vs_all_gdsc": int((k_min_all > thresh).sum()),
        "kryzei_no_neighbor_vs_crc": int((k_min_crc > thresh).sum()),
        "kryzei_n": int(len(k_wl_idx)),
        "biobank_no_neighbor_vs_all_gdsc": int((b_min_all > thresh).sum()),
        "biobank_n": int(len(b_ids)),
        "threshold_used": thresh,
    }

    log("[exp035_f1] domain classification (main)")
    X_main = np.vstack([g_crc_z, k_z])
    y_main = np.concatenate([np.zeros(len(g_crc_z)), np.ones(len(k_z))]).astype(int)
    summ["domain_classification_main"] = classify_report(
        X_main, y_main, label="同癌种细胞系(GDSC COREAD) vs mCRC类器官(60)")
    X_main_proj = project(X_main)  # 供等样本量对照复用

    log("[exp035_f1] domain classification (pancancer)")
    X_pan = np.vstack([g_z, k_z])
    y_pan = np.concatenate([np.zeros(len(g_z)), np.ones(len(k_z))]).astype(int)
    summ["domain_classification_pancancer"] = classify_report(
        X_pan, y_pan, label="泛癌细胞系(GDSC 963) vs mCRC类器官(60)")

    log("[exp035_f1] domain classification (cancer composition)")
    y_cc = np.concatenate([np.zeros(len(g_z)), np.ones(len(g_crc_z))]).astype(int)
    summ["domain_classification_cancer_composition"] = classify_report(
        np.vstack([g_z, g_crc_z]), y_cc,
        label="泛癌GDSC(963) vs 结直肠GDSC(COREAD 46) —— 同平台癌种可分性")

    log("[exp035_f1] domain classification (main, raw expression)")
    X_main_raw = np.vstack([g_int_fill[crc_i], k_wl_expr])
    summ["domain_classification_main_raw_expression"] = classify_report(
        X_main_raw, y_main, label="主比较·原始log2表达(域间全局标准化)")

    log("[exp035_f1] reconstruction (PCA shared/private)")
    domains = [g_z, k_z, b_z]
    dom_names = ["gdsc", "kryzei", "biobank"]
    pooled = np.vstack(domains)
    shared_pca = PCA(n_components=SHARED_PC, random_state=0).fit(pooled)
    recon = {}
    for name, d in zip(dom_names, domains):
        priv = PCA(n_components=PRIVATE_PC, random_state=0).fit(d)
        full_pc = min(FULL_PC, d.shape[0])
        full = PCA(n_components=full_pc, random_state=0).fit(d)
        r_shared = r2(d, shared_pca.inverse_transform(shared_pca.transform(d)))
        r_private = r2(d, priv.inverse_transform(priv.transform(d)))
        r_full = r2(d, full.inverse_transform(full.transform(d)))
        recon[name] = {
            "recon_variance_full": r_full,
            "full_n_components": int(full_pc),
            "recon_variance_shared_64comp": r_shared,
            "recon_variance_private_32comp": r_private,
            "shared_subspace_variance_contribution": (
                r_shared / r_full if np.isfinite(r_full) and r_full != 0 else float("nan")),
        }
    summ["representation_reconstruction"] = {
        "model": "PCA shared(64 components, fitted on pooled 3-domain z) + per-domain private(32); R2 = 1 - SSE/SST",
        "note": "比例一律表述为「该模型下的重构方差贡献」，不作生物共享比例解释",
        "domains": recon,
    }

    log("[exp035_f1] prediction error relationship")
    pred = pd.read_parquet(EXP032_PRED)
    zs = pred[pred["arm"] == "source_zero_shot"].copy()
    l3 = json.loads(EXP032_L3.read_text(encoding="utf-8"))
    known14 = set(l3["known14_drugs"])
    zs["known"] = zs["drug_id"].astype(str).isin(known14)
    zs["y_true"] = pd.to_numeric(zs["y_true"], errors="coerce").astype(float)
    zs["y_pred"] = pd.to_numeric(zs["y_pred"], errors="coerce").astype(float)

    def sample_spearman(sub):
        if sub["drug_id"].nunique() < 3:
            return np.nan
        r = spearmanr(sub["y_true"].to_numpy(float),
                      sub["y_pred"].to_numpy(float)).statistic
        return float(r) if np.isfinite(r) else np.nan

    rows = []
    for mid, g in zs.groupby("model_id", sort=True):
        rows.append({
            "model_id": str(mid),
            "patient_id": str(g["patient_id"].iloc[0]),
            "err_known14": sample_spearman(g[g["known"]]),
            "err_native48": sample_spearman(g),
        })
    err = pd.DataFrame(rows).set_index("model_id")
    mid_to_krow = {k_sid[k_wl_idx[i]]: i for i in range(len(k_wl_idx))}
    err = err[err.index.isin(mid_to_krow)].copy()
    err["min_dist_crc"] = [k_min_crc[mid_to_krow[m]] for m in err.index]
    err["min_dist_all"] = [k_min_all[mid_to_krow[m]] for m in err.index]

    def boot_spearman_ci(a, b, n=N_BOOT, seed=SEED_BOOT):
        a = np.asarray(a, float); b = np.asarray(b, float)
        rng = np.random.default_rng(seed)
        m = len(a)
        stats = []
        for _ in range(n):
            idx = rng.integers(0, m, size=m)
            r = spearmanr(a[idx], b[idx]).statistic
            stats.append(float(r) if np.isfinite(r) else np.nan)
        stats = np.asarray(stats)
        return {"low": float(np.nanpercentile(stats, 2.5)),
                "high": float(np.nanpercentile(stats, 97.5)),
                "n_units": m, "n": n, "seed": seed}

    def corr_entry(a, b):
        rho, p = spearman_p(np.asarray(a, float), np.asarray(b, float))
        return {"rho": rho, "p": p, "ci95": boot_spearman_ci(a, b),
                "n": int(len(np.asarray(a, float)))}

    sample_known = corr_entry(err["err_known14"].dropna(),
                              err.loc[err["err_known14"].notna(), "min_dist_crc"])
    sample_native = corr_entry(err["err_native48"].dropna(),
                               err.loc[err["err_native48"].notna(), "min_dist_crc"])
    pat = err.groupby("patient_id").agg(
        err_known14=("err_known14", "mean"),
        err_native48=("err_native48", "mean"),
        min_dist_crc=("min_dist_crc", "mean"),
    ).dropna(subset=["err_known14"])
    pat_known = corr_entry(pat["err_known14"], pat["min_dist_crc"])
    pat_native = corr_entry(pat["err_native48"], pat["min_dist_crc"])

    corr_family = {
        "sample_err_known14_vs_min_dist_crc": sample_known,
        "sample_err_native48_vs_min_dist_crc": sample_native,
        "patient_err_known14_vs_min_dist_crc": pat_known,
        "patient_err_native48_vs_min_dist_crc": pat_native,
    }
    holm = holm_correct([v["p"] for v in corr_family.values()])
    for i, k in enumerate(corr_family):
        corr_family[k]["holm_p"] = holm[i]
    summ["prediction_error_relationship"] = {
        "zero_shot_source": "EXP-032 L3 source_zero_shot (EXP-015 source model, frozen)",
        "error_metric": "per-sample Spearman(y_true, y_pred) over known14 / native48 drugs",
        "distance_metric": "min distance (1-Pearson, z-space) to nearest GDSC COREAD cell line",
        "correlations": corr_family,
        "n_samples": int(len(err)),
        "n_patients": int(pat.shape[0]),
    }

    log("[exp035_f1] control: pancancer vs same-cancer (integrated in classification)")

    log("[exp035_f1] control: equal sample size bootstrap")
    n_eq = min(len(g_crc_z), len(k_z))
    rng = np.random.default_rng(SEED_BOOT)
    eq_ba = []
    for _ in range(N_EQUAL_BOOT):
        ci = rng.integers(0, len(g_crc_z), size=n_eq)
        ki_ = rng.integers(0, len(k_z), size=n_eq)
        Xp = np.vstack([X_main_proj[ci], X_main_proj[len(g_crc_z) + ki_]])
        y = np.concatenate([np.zeros(n_eq), np.ones(n_eq)]).astype(int)
        eq_ba.append(logreg_ba_cv(Xp, y, SEED_CV))
    eq_ba = np.asarray(eq_ba)
    summ["control_equal_sample_size"] = {
        "n_per_class": n_eq,
        "n_boot": N_EQUAL_BOOT,
        "balanced_accuracy_mean": float(eq_ba.mean()),
        "balanced_accuracy_ci95": [float(np.percentile(eq_ba, 2.5)),
                                   float(np.percentile(eq_ba, 97.5))],
        "full_sample_balanced_acc": summ["domain_classification_main"]["balanced_accuracy"],
        "note": "在固定 PCA(50) 投影（主比较拟合）上，两类别各重采样 n_eq 个样本后重拟合逻辑回归",
    }

    log("[exp035_f1] control: cross-study cell line (DepMap 24Q2)")
    try:
        model = pd.read_csv(DEPMAP_MODEL, low_memory=False)
        crc_models = set(model.loc[
            model["OncotreePrimaryDisease"] == "Colorectal Adenocarcinoma",
            "ModelID"].astype(str))
        header = pd.read_csv(DEPMAP_RNA, nrows=0)
        hcols = list(header.columns)
        sym_col = {}
        for c in hcols[1:]:
            sym = c.split(" (")[0]
            if sym in inter:
                sym_col[sym] = c
        depmap_genes = [g for g in inter if g in sym_col]
        usecols = [hcols[0]] + [sym_col[g] for g in depmap_genes]
        dpm = pd.read_csv(DEPMAP_RNA, usecols=usecols, index_col=0)
        dpm_ids = [str(x) for x in dpm.index]
        depmap_crc_ids = sorted(crc_models & set(dpm_ids))
        dpm_crc_idx = [dpm_ids.index(i) for i in depmap_crc_ids]
        dpm_all_z = zscore(dpm.to_numpy(float))
        dpm_crc_z = dpm_all_z[dpm_crc_idx]

        inter_pos = {g: j for j, g in enumerate(inter)}
        idp = np.array([inter_pos[g] for g in depmap_genes])
        g_crc_dep_z = g_crc_z[:, idp]
        k_dep_z = k_z[:, idp]

        d_dep_g = corr_dist(g_crc_dep_z, dpm_crc_z)   # (n_dep, n_gdsc_crc)
        d_g_dep = corr_dist(dpm_crc_z, g_crc_dep_z)   # (n_gdsc_crc, n_dep)
        d_dep_dep = corr_dist(dpm_crc_z, dpm_crc_z)
        np.fill_diagonal(d_dep_dep, np.inf)
        dep_nn = d_dep_dep.min(axis=1)
        d_k_dep = corr_dist(dpm_crc_z, k_dep_z)       # (60, n_dep)
        summ["control_cross_study_cell_line"] = {
            "depmap_crc_lines_with_rna": int(len(depmap_crc_ids)),
            "depmap_genetic_matched_genes": int(len(depmap_genes)),
            "evidence": "DepMap 24Q2 RNA-seq raw CSV (depmap_public_24q2_rnaseq_raw_v1); "
                        "label = OncotreePrimaryDisease 'Colorectal Adenocarcinoma'",
            "within_gdsc_crc_nn": q(crc_nn_min),
            "within_depmap_crc_nn": q(dep_nn),
            "depmap_crc_to_nearest_gdsc_crc": q(d_dep_g.min(axis=1)),
            "gdsc_crc_to_nearest_depmap_crc": q(d_g_dep.min(axis=1)),
            "kryzei_to_nearest_depmap_crc": q(d_k_dep.min(axis=1)),
            "confound": "跨研究比较混有平台/文库/培养差异，无法独立识别原因",
        }
    except Exception as e:  # noqa: BLE001
        log(f"[exp035_f1] DepMap control failed: {e}")
        summ["control_cross_study_cell_line"] = {
            "available": False, "reason": f"asset construction failed: {e}",
            "confound": "无法独立识别原因"}

    summ["elapsed_seconds"] = round(time.time() - t0, 1)
    (OUT_DIR / "f1_summary.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    log("WROTE " + str(OUT_DIR / "f1_summary.json"))
    log(json.dumps(summ, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()