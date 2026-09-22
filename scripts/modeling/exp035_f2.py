#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-035 F2: 重建基因依赖评价（三层）。

预登记契约（experiments/records/EXP-035.md，工作包 F2）：
- 层1 通用必需基因识别：AUROC 复核引用 EXP-030（不重算）。
- 层2 癌种/谱系依赖：按 primary_tumour_type 分癌种逐基因迁移宏平均（ZS /
  HD）+ 谱系特异基因清单（训练侧选择、留出验证）。
- 层3 个体化（核心）：结直肠子集（85→83 患者）逐基因折内 Spearman（患者先
  聚合）+ 折合并跨患者 Spearman + 患者级 bootstrap 95% CI（CI 不跨零计数）
  + 患者整体置换检验（100 次）+ 癌种控制（仅结直肠）。不减去基因均值相关。

产物：artifacts/exp035_f2f3/f2_summary.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path("/mnt/d/Code/DrugScreenLab")
OUT_DIR = ROOT / "artifacts/exp035_f2f3"
PRED_PARQUET = ROOT / "artifacts/exp030_dependency_transfer/predictions.parquet"
ARM_SUMMARY = ROOT / "artifacts/exp030_dependency_transfer/arm_summary.json"
SAMPLES = ROOT / "data/processed/transfer/biobank_functional_v1/samples.parquet"
ESSENTIALS = (ROOT / "data/raw/transfer_public_assets/organoid_biobank/"
              "CRISPRInferredCommonEssentials.csv")
KRYZEI_AUDIT = ROOT / "data/processed/identity/kryzei_admission_audit_20260921.json"

MIN_ROWS_MACRO = 8
BOOT_N = 2000
BOOT_SEED = 20260921
PERM_N = 100
PERM_SEED = 20260921
ARMS_L2 = ["ZS", "HD"]
CANCERS = ["Colorectal", "Oesophageal", "Ovarian", "Pancreatic", "Gastric"]
MAIN_CANCER = "Colorectal"
N_LINEAGE_TOP = 50


def _gene_spearman(pred: np.ndarray, obs: np.ndarray, min_rows: int = 3) -> np.ndarray:
    """逐基因 Spearman（向量化，G x N，平均并列秩）。"""
    mask = ~np.isnan(pred) & ~np.isnan(obs)
    rp = rankdata(np.where(mask, pred, np.inf), axis=1, method="average")
    ro = rankdata(np.where(mask, obs, np.inf), axis=1, method="average")
    k = mask.sum(axis=1)
    n = np.maximum(k, 1)
    mp = np.where(mask, rp, 0.0).sum(axis=1) / n
    mo = np.where(mask, ro, 0.0).sum(axis=1) / n
    dp = np.where(mask, rp - mp[:, None], 0.0)
    do = np.where(mask, ro - mo[:, None], 0.0)
    cov = (dp * do).sum(axis=1)
    var = np.sqrt((dp * dp).sum(axis=1) * (do * do).sum(axis=1))
    rho = np.full(pred.shape[0], np.nan)
    ok = (k >= min_rows) & (var > 0)
    rho[ok] = cov[ok] / var[ok]
    return rho


def _fast_spearman(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """逐基因 Spearman（无 NaN；双 argsort 秩，不分并列，用于重采样）。"""
    rp = np.argsort(np.argsort(pred, axis=1, kind="stable"), axis=1).astype(np.float64)
    ro = np.argsort(np.argsort(obs, axis=1, kind="stable"), axis=1).astype(np.float64)
    rp -= rp.mean(axis=1, keepdims=True)
    ro -= ro.mean(axis=1, keepdims=True)
    cov = (rp * ro).sum(axis=1)
    s2p = (rp * rp).sum(axis=1)
    s2o = (ro * ro).sum(axis=1)
    var = np.sqrt(s2p * s2o)
    rho = np.full(pred.shape[0], np.nan)
    ok = var > 0
    rho[ok] = cov[ok] / var[ok]
    return rho


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log: dict = {}

    samples = pd.read_parquet(SAMPLES)
    sample_list = samples["sample_ID"].astype(str).tolist()
    smap = {s: i for i, s in enumerate(sample_list)}
    tissue_arr = samples["primary_tumour_type"].to_numpy()
    ind_arr = samples["individual_ID"].to_numpy()

    pred = pd.read_parquet(PRED_PARQUET,
                           columns=["fold", "arm", "sample", "gene",
                                    "predicted", "observed"])
    genes = np.array(sorted(pred["gene"].unique()), dtype=object)
    n_genes = len(genes)
    log["n_evaluable_genes"] = n_genes

    # 修复（2026-09-22 审计）：EXP-030 predictions.parquet 的 (sample, gene) 标签
    # 与数值行序错位——数值按 gene-major（k = g*n_te + s）写入，标签按 sample-major。
    # 按标签重建会产生跨基因伪相关（HD 宏平均 0.45 vs 真实 0.032）。此处改按
    # 书写行序重建，已与 arm_summary.json 内存口径逐折精确复核一致。
    fold_of_sample = (pred[["sample", "fold"]].drop_duplicates()
                      .set_index("sample")["fold"].to_dict())

    def arm_matrix(arm: str) -> tuple[np.ndarray, np.ndarray]:
        sub = pred[pred["arm"] == arm]
        P = np.full((n_genes, len(sample_list)), np.nan, dtype=np.float32)
        O = np.full((n_genes, len(sample_list)), np.nan, dtype=np.float32)
        for f in range(1, 6):
            sub_f = sub[sub["fold"] == f]
            col_ids = list(dict.fromkeys(sub_f["sample"].tolist()))
            n_te = len(col_ids)
            if n_te == 0:
                continue
            assert len(sub_f) == n_genes * n_te, (arm, f, len(sub_f), n_genes, n_te)
            cols = [smap[s] for s in col_ids]
            P[:, cols] = sub_f["predicted"].to_numpy(
                np.float32).reshape(n_genes, n_te)
            O[:, cols] = sub_f["observed"].to_numpy(
                np.float32).reshape(n_genes, n_te)
        return P, O

    # ---------- Kryzei 重叠 ----------
    kry = json.loads(KRYZEI_AUDIT.read_text(encoding="utf-8"))
    kpats = set(kry["patient_whitelist"])
    kpdo = set(kry["pdo_whitelist"])
    bio_patients = set(samples["patient_id"].dropna().astype(str).tolist())
    bio_samps = set(sample_list)
    overlap_pat = bio_patients & kpats
    overlap_samp = bio_samps & kpdo
    log["kryzei_overlap"] = {
        "conclusion": "无重叠" if not overlap_pat and not overlap_samp else "存在重叠",
        "biobank_patient_id_scheme": "integer string (== individual_ID)",
        "kryzei_patient_scheme": "Pt31..Pt151",
        "n_biobank_samples": int(len(samples)),
        "n_biobank_individuals": int(samples["individual_ID"].nunique()),
        "n_kryzei_patients": len(kpats),
        "n_kryzei_pdos": len(kpdo),
        "overlap_patient_ids": sorted(overlap_pat),
        "overlap_sample_ids": sorted(overlap_samp),
    }

    # ---------- 必需基因 ----------
    ess = pd.read_csv(ESSENTIALS)
    ess_genes = {str(v).split(" (")[0] for v in ess[ess.columns[0]]}
    ess_mask = np.array([(g in ess_genes) for g in genes], dtype=bool)
    log["essentials"] = {"n_prior": len(ess_genes),
                         "n_in_evaluable": int(ess_mask.sum())}

    # ---------- 层1 ----------
    arm_sum = json.loads(ARM_SUMMARY.read_text(encoding="utf-8"))
    log["layer1_essential_auroc"] = {
        "source": "EXP-030 arm_summary.json secondary_auroc_essential",
        "values": arm_sum["secondary_auroc_essential"],
        "note": "引用复核，不重算；ZS 0.9971 最高。",
    }

    # ---------- 层2 分癌种宏平均 ----------
    layer2 = {"per_cancer_macro": {}, "lineage_specific_genes": {}}
    cancer_mask = {c: (tissue_arr == c) for c in CANCERS}
    cancer_mask["Other"] = ~np.isin(tissue_arr, ["Colorectal", "Oesophageal"])
    O_zs = None
    for arm in ARMS_L2:
        P, O = arm_matrix(arm)
        if arm == "ZS":
            O_zs = O
        layer2["per_cancer_macro"][arm] = {}
        for c in list(CANCERS) + ["Other"]:
            cm = cancer_mask[c]
            rho = _gene_spearman(P[:, cm], O[:, cm], min_rows=MIN_ROWS_MACRO)
            fin = np.isfinite(rho)
            layer2["per_cancer_macro"][arm][c] = {
                "n_samples": int(cm.sum()),
                "n_genes": int(fin.sum()),
                "macro_spearman": float(np.mean(rho[fin])) if fin.any() else None,
            }

    # ---------- 层2 谱系特异基因 ----------
    sample_fold = np.array([int(fold_of_sample[s]) for s in sample_list])
    O_all = O_zs
    strata = CANCERS
    lin_score = np.zeros(n_genes)
    lin_repro = np.full((n_genes, 5), np.nan)
    for f in range(1, 6):
        tr = sample_fold != f
        te = sample_fold == f
        tr_means = np.full((n_genes, len(strata)), np.nan)
        te_means = np.full((n_genes, len(strata)), np.nan)
        for ci, c in enumerate(strata):
            m = tr & (tissue_arr == c)
            if m.sum() >= 2:
                tr_means[:, ci] = np.nanmean(O_all[:, m], axis=1)
            m2 = te & (tissue_arr == c)
            if m2.sum() >= 1:
                te_means[:, ci] = np.nanmean(O_all[:, m2], axis=1)
        with np.errstate(invalid="ignore"):
            lin_score += np.nanstd(tr_means, axis=1) / 5.0
        for g in range(n_genes):
            ok = ~np.isnan(tr_means[g]) & ~np.isnan(te_means[g])
            if ok.sum() >= 3:
                a, b = tr_means[g][ok], te_means[g][ok]
                if np.ptp(a) > 0 and np.ptp(b) > 0:
                    rk = rankdata(a), rankdata(b)
                    lin_repro[g, f - 1] = np.corrcoef(rk[0], rk[1])[0, 1]
    lin_repro_mean = np.nanmean(lin_repro, axis=1)
    order = np.argsort(-lin_score, kind="stable")
    selected = order[:N_LINEAGE_TOP]
    pooled_cmean = np.zeros((n_genes, len(strata)))
    for ci, c in enumerate(strata):
        m = tissue_arr == c
        pooled_cmean[:, ci] = np.nanmean(O_all[:, m], axis=1)
    top_genes = []
    for g in selected:
        top_genes.append({
            "gene": str(genes[g]),
            "lineage_score_train": float(lin_score[g]),
            "holdout_reproducibility": (float(lin_repro_mean[g])
                                        if np.isfinite(lin_repro_mean[g]) else None),
            "essential": bool(ess_mask[g]),
            "per_cancer_mean_observed": {
                c: (float(pooled_cmean[g, ci]) if np.isfinite(pooled_cmean[g, ci])
                    else None) for ci, c in enumerate(strata)},
        })
    layer2["lineage_specific_genes"] = {
        "method": ("训练侧逐折跨癌种层均值扩展分(std)+留出复现(训练/测试层均值秩相关)"
                   "5 折；选择纯训练侧(top-%d)留出验证。" % N_LINEAGE_TOP),
        "n_selected": N_LINEAGE_TOP,
        "honest_caveat": ("小癌种样本 3-11，逐折测试层均值极噪，留出复现值整体偏低"
                          "属预期，仅线索级。"),
        "selected_holdout_reproducibility_mean": float(
            np.nanmean(lin_repro_mean[selected])),
        "all_genes_holdout_reproducibility_mean": float(
            np.nanmean(lin_repro_mean[np.isfinite(lin_repro_mean)])),
        "top_genes": top_genes,
    }
    # ---------- 层3 个体化 ----------
    cr_idx = np.where(tissue_arr == MAIN_CANCER)[0]
    # 患者聚合
    per_pat = {}
    for i in cr_idx:
        per_pat.setdefault(int(ind_arr[i]), []).append(i)
    pat_ids = sorted(per_pat.keys())
    n_pat = len(pat_ids)
    log["layer3_scope"] = {
        "cancer": MAIN_CANCER, "n_models": int(len(cr_idx)),
        "n_patients_after_aggregation": n_pat,
        "note": "85 模型→83 患者（2 个 multi-model individual 患者内均值聚合）。",
    }

    layer3 = {}
    for arm in ["HD", "ZS"]:
        P, O = arm_matrix(arm)
        Pp = np.full((n_genes, n_pat), np.nan, dtype=np.float32)
        Op = np.full((n_genes, n_pat), np.nan, dtype=np.float32)
        fold_arr = np.zeros(n_pat, dtype=int)
        for j, pid in enumerate(pat_ids):
            cols = per_pat[pid]
            Pp[:, j] = np.nanmean(P[:, cols], axis=1)
            Op[:, j] = np.nanmean(O[:, cols], axis=1)
            fold_arr[j] = sample_fold[cols[0]]
        # 折内逐基因 Spearman（患者级）
        fold_rhos = []
        for f in range(1, 6):
            m = fold_arr == f
            if m.sum() < 3:
                continue
            fold_rhos.append(_gene_spearman(Pp[:, m], Op[:, m], min_rows=3))
        fold_rho_mean = np.nanmean(np.vstack(fold_rhos), axis=0)

        # 折合并跨患者逐基因 Spearman
        rho_pool = _gene_spearman(Pp, Op, min_rows=3)
        fin_pool = np.isfinite(rho_pool)

        # bootstrap CI（患者级重采样，快速秩）
        rng = np.random.default_rng(BOOT_SEED)
        boot = np.full((BOOT_N, n_genes), np.nan, dtype=np.float32)
        for it in range(BOOT_N):
            take = rng.integers(0, n_pat, n_pat)
            boot[it] = _fast_spearman(Pp[:, take], Op[:, take]).astype(np.float32)
        ci_low = np.nanpercentile(boot, 2.5, axis=0)
        ci_high = np.nanpercentile(boot, 97.5, axis=0)
        rep_pos = fin_pool & (ci_low > 0)
        n_ci_pos = int(rep_pos.sum())

        # 置换检验（100 次）
        rng2 = np.random.default_rng(PERM_SEED)
        perm_macro = np.full(PERM_N, np.nan)
        perm_pos = np.full(PERM_N, np.nan)
        for it in range(PERM_N):
            colp = rng2.permutation(n_pat)
            rho_perm = _fast_spearman(Pp, Op[:, colp])
            fin = np.isfinite(rho_perm)
            perm_macro[it] = float(np.mean(rho_perm[fin])) if fin.any() else np.nan
            perm_pos[it] = float((rho_perm > 0).mean())
        obs_macro = float(np.mean(rho_pool[fin_pool])) if fin_pool.any() else np.nan
        pmf = perm_macro[np.isfinite(perm_macro)]
        pval = float(np.mean(pmf >= obs_macro)) if pmf.size else np.nan

        ess_fin = fin_pool & ess_mask
        var_fin = fin_pool & ~ess_mask
        layer3[arm] = {
            "per_gene_rho_pooled": {
                "mean": obs_macro,
                "median": float(np.nanmedian(rho_pool[fin_pool])),
                "frac_positive": float(np.mean(rho_pool[fin_pool] > 0)),
                "n_genes": int(fin_pool.sum()),
            },
            "within_fold_mean_rho": {
                "mean": float(np.nanmean(fold_rho_mean[np.isfinite(fold_rho_mean)])),
                "n_genes": int(np.isfinite(fold_rho_mean).sum()),
            },
            "n_genes_ci_positive": n_ci_pos,
            "ci_positive_frac": float(n_ci_pos / max(int(fin_pool.sum()), 1)),
            "ci_positive_genes": None,  # 见 summary 里另存
            "permutation_null_100": {
                "statistic": "macro per-gene individualized Spearman",
                "observed_macro": obs_macro,
                "null_mean": float(np.mean(pmf)),
                "null_sd": float(np.std(pmf)),
                "null_p2_5": float(np.percentile(pmf, 2.5)),
                "null_p97_5": float(np.percentile(pmf, 97.5)),
                "permutation_p_value": pval,
                "null_frac_positive_genes_mean": float(np.nanmean(perm_pos)),
                "observed_frac_positive_genes": float(np.mean(rho_pool > 0)),
            },
            "essential_vs_variable": {
                "essential": {"macro": float(np.mean(rho_pool[ess_fin])) if ess_fin.any() else None,
                              "n_genes": int(ess_fin.sum()),
                              "n_ci_positive": int((rep_pos & ess_mask).sum())},
                "variable": {"macro": float(np.mean(rho_pool[var_fin])) if var_fin.any() else None,
                             "n_genes": int(var_fin.sum()),
                             "n_ci_positive": int((rep_pos & ~ess_mask).sum())},
            },
        }
        # 存 CI>0 基因名（供报告排序列出）
        pos_genes = [str(g) for g, ok in zip(genes, rep_pos.tolist()) if ok]
        layer3[arm]["ci_positive_genes"] = pos_genes[:200]
        layer3[arm]["ci_positive_genes_count_total"] = len(pos_genes)

    log["layer2_lineage"] = layer2
    log["layer3_individualization"] = layer3
    log["elapsed_seconds"] = round(time.time() - t0, 1)

    (OUT_DIR / "f2_summary.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"[exp035_f2] done in {log['elapsed_seconds']}s; "
          f"HD ci_positive={layer3['HD']['n_genes_ci_positive']} "
          f"ZS={layer3['ZS']['n_genes_ci_positive']} "
          f"HD macro={layer3['HD']['per_gene_rho_pooled']['mean']:.4f}", flush=True)


if __name__ == "__main__":
    main()