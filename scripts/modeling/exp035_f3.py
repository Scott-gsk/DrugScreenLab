#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-035 F3: 归纳偏置与学习曲线（四模型 + 等规模随机分组对照）。

预登记契约（experiments/records/EXP-035.md，工作包 F3）：
- 四模型：①逐基因独立 Ridge；②低秩多任务（因子化 W=UV^T 谱分解）；
  ③非线性共享表示（MLP 共享编码器 + 逐基因线性头）；④固定通路结构模型
  （Hallmark 通路共成员掩码）+ 等规模随机分组对照（同掩码尺寸、随机基因分组）。
- 相同输入（表达→依赖，基因级表达 z 分数）、同患者划分（复用 EXP-030
  individual 整组 5 折，seed=20260918）、同搜索预算（每模型 1 个固定候选
  <=20）、训练量档位 25%/50%/100%（训练患者嵌套子集，测试患者不变）。
- 报告训练与验证表现（宏观 Spearman + 必需基因分层）。

产物：artifacts/exp035_f2f3/f3_summary.json
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold

from drug_screen.data.organoid_depmap import resolve_dual_library

ROOT = Path("/mnt/d/Code/DrugScreenLab")
OUT_DIR = ROOT / "artifacts/exp035_f2f3"
SAMPLES = ROOT / "data/processed/transfer/biobank_functional_v1/samples.parquet"
EXPR = ROOT / "data/processed/transfer/biobank_functional_v1/expression_author_log2tpm.parquet"
DEP = ROOT / "data/processed/transfer/biobank_functional_v1/dependency_by_library.parquet"
ESSENTIALS = (ROOT / "data/raw/transfer_public_assets/organoid_biobank/"
              "CRISPRInferredCommonEssentials.csv")
GMT = ROOT / "data/raw/exp013/msigdb/h.all.v2023.2.Hs.symbols.gmt"

N_FOLDS = 5
SPLIT_SEED = 20260918
MIN_ROWS = 30
ALPHA = 1.0
LOW_RANK = 32
ALS_ITERS = 20
MLP_HIDDEN = [512, 128]
MLP_EPOCHS = 300
MLP_PATIENCE = 25
MLP_LR = 1e-3
MLP_WD = 1e-4
MLP_SEEDS = [42, 43, 44]
TIERS = [0.25, 0.50, 1.00]
CANCER_MAIN = "Colorectal"
CANCER_SECOND = "Oesophageal"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SMOKE = "--smoke" in sys.argv
if SMOKE:
    MLP_EPOCHS, MLP_SEEDS = 3, [42]
    ALS_ITERS = 3


def _symbol(column: str) -> str:
    if column.endswith("_exp"):
        return column[:-4]
    return column.split(" (")[0]


def _gene_spearman(pred: np.ndarray, obs: np.ndarray, min_rows: int = MIN_ROWS) -> np.ndarray:
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


def _macro(rho: np.ndarray) -> float:
    fin = np.isfinite(rho)
    return float(np.mean(rho[fin])) if fin.any() else float("nan")


def _zscore_fit_apply(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (Xtr - mu) / sd, (Xte - mu) / sd


# ----------------------------------------------------------------- models --
# 每个模型返回 (pred_on_train, pred_on_test)，训练/验证表现用同一拟合模型。

def model_ridge(Xtr, Ytr, Xte):
    K = Xtr @ Xtr.T
    A = np.linalg.solve(K + ALPHA * np.eye(len(Xtr)), Ytr)
    return (K @ A), ((Xte @ Xtr.T) @ A)


def model_lowrank(Xtr, Ytr, Xte, r=LOW_RANK, iters=ALS_ITERS):
    n_tr, D = Xtr.shape
    G = Ytr.shape[1]
    r = min(r, n_tr, G)
    svd = TruncatedSVD(n_components=r, random_state=0)
    svd.fit(Ytr)
    V = svd.components_.T.astype(np.float64)   # G x r
    K = Xtr @ Xtr.T
    XtYfree = Xtr.T @ np.linalg.inv(K + ALPHA * np.eye(n_tr))
    for _ in range(iters):
        Z = Ytr @ V
        U = XtYfree @ Z                          # D x r
        H = Xtr @ U                              # n_tr x r
        V = np.linalg.solve(H.T @ H + ALPHA * np.eye(r), H.T @ Ytr).T
    return ((Xtr @ U) @ V.T), ((Xte @ U) @ V.T)


class _MLP(nn.Module):
    def __init__(self, D, G):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(D, MLP_HIDDEN[0]), nn.ReLU(),
            nn.Linear(MLP_HIDDEN[0], MLP_HIDDEN[1]))
        self.head = nn.Linear(MLP_HIDDEN[1], G)

    def forward(self, x):
        return self.head(self.enc(x))


def model_mlp(Xtr, Ytr, Xte, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    D, G = Xtr.shape[1], Ytr.shape[1]
    model = _MLP(D, G).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    Yt = torch.tensor(Ytr, dtype=torch.float32, device=DEVICE)
    Xe = torch.tensor(Xte, dtype=torch.float32, device=DEVICE)
    perm = np.random.permutation(len(Xtr))
    n_val = max(int(0.2 * len(Xtr)), 1)
    tr_idx, va_idx = perm[n_val:], perm[:n_val]
    best_loss, best_state, bad = float("inf"), None, 0
    for _ in range(MLP_EPOCHS):
        model.train()
        loss = ((model(Xt) - Yt) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(((model(Xt[va_idx]) - Yt[va_idx]) ** 2).mean().item())
        if vl < best_loss - 1e-6:
            best_loss, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= MLP_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return (model(Xt).cpu().numpy().astype(np.float64),
                model(Xe).cpu().numpy().astype(np.float64))


def _build_masks(evaluable, col_pos, gmt_path, rng):
    """逐输出基因掩码（输入列索引）。

    pathway = 输出基因 g 的 Hallmark 共成员输入列 ∪ {g 自身}；
    random  = {g 自身} ∪ 同尺寸随机输入列（镜像对角+分组结构，仅分组随机化）。
    """
    p2g, gene2path = {}, {}
    for ln in gmt_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        parts = ln.split("\t")
        name, gs = parts[0], parts[2:]
        p2g[name] = set(gs)
        for g in gs:
            gene2path.setdefault(g, set()).add(name)
    D = len(col_pos)
    masks, rand_masks, sizes = [], [], []
    for g in evaluable:
        self_i = col_pos[g]
        co = {self_i}
        for p in gene2path.get(g, ()):
            for gg in p2g[p]:
                if gg in col_pos:
                    co.add(col_pos[gg])
        idx = np.asarray(sorted(co), dtype=np.int64)
        masks.append(idx)
        sizes.append(len(idx))
    for g, sz in zip(evaluable, sizes):
        self_i = col_pos[g]
        others = [j for j in range(D) if j != self_i]
        rnd = rng.choice(others, size=sz - 1, replace=False)
        rand_masks.append(np.asarray(sorted(list(rnd) + [self_i]), dtype=np.int64))
    return masks, rand_masks


def _masked_ridge_both(Xtr, Ytr, Xte, masks):
    """遮罩岭回归：返回 (pred_tr, pred_te)。单基因掩码向量化，多基因掩码逐基因对偶解。"""
    n_tr = len(Xtr)
    n_te = len(Xte)
    G = len(masks)
    pred_tr = np.full((n_tr, G), np.nan, dtype=np.float64)
    pred_te = np.full((n_te, G), np.nan, dtype=np.float64)
    eye = ALPHA * np.eye(n_tr)
    # 单基因掩码（size==1）向量化：仅一列特征 x_j，闭式 w = <x_j,y>/ (<x_j,x_j>+alpha)
    single = [i for i in range(G) if len(masks[i]) == 1]
    if single:
        js = np.array([int(masks[i][0]) for i in single], dtype=np.int64)
        xv = Xtr[:, js]                       # n_tr x m
        xv_te = Xte[:, js]
        denom = (xv * xv).sum(axis=0) + ALPHA
        num = (xv * Ytr[:, single]).sum(axis=0)
        w = num / denom                       # m
        pred_tr[:, single] = xv * w[None, :]
        pred_te[:, single] = xv_te * w[None, :]
    # 多基因掩码逐基因对偶解
    for i in range(G):
        if len(masks[i]) <= 1:
            continue
        S = masks[i]
        Xs = Xtr[:, S]
        Gs = Xs @ Xs.T
        a = np.linalg.solve(Gs + eye, Ytr[:, i])
        pred_tr[:, i] = Gs @ a
        pred_te[:, i] = (Xte[:, S] @ Xs.T) @ a
    return pred_tr, pred_te


# ------------------------------------------------------------------- main --
def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log: dict = {"device": str(DEVICE), "smoke": SMOKE}

    samples = pd.read_parquet(SAMPLES)
    expr = pd.read_parquet(EXPR)
    sample_ids = samples["sample_ID"].astype(str).tolist()
    assert list(expr.index.astype(str)) == sample_ids
    expr.columns = [_symbol(str(c)) for c in expr.columns]
    expr = expr.loc[:, ~expr.columns.duplicated()]

    dep = pd.read_parquet(DEP)
    dep_res, _ = resolve_dual_library(dep)
    dep_wide = dep_res.pivot(index="sample_ID", columns="gene", values="LFC")
    dep_wide = dep_wide.reindex(index=sample_ids)

    expr_genes = list(expr.columns)
    dep_genes = list(dep_wide.columns)
    evaluable = sorted(set(expr_genes) & set(dep_genes))
    D = len(expr_genes)
    G = len(evaluable)
    log["gene_spaces"] = {"n_expr_genes_input": D, "n_evaluable_output": G}

    X = expr.reindex(columns=expr_genes).to_numpy(np.float32)
    Y = dep_wide.reindex(columns=evaluable).to_numpy(np.float32)
    log["y_observed_frac"] = float(1.0 - np.isnan(Y).mean())

    ess = pd.read_csv(ESSENTIALS)
    ess_genes = {_symbol(str(v)) for v in ess[ess.columns[0]]}
    ess_mask = np.array([(g in ess_genes) for g in evaluable], dtype=bool)
    log["essentials_n_in_evaluable"] = int(ess_mask.sum())

    rng_mask = np.random.default_rng(20260921)
    col_pos = {c: i for i, c in enumerate(expr_genes)}
    pmasks, rmasks = _build_masks(evaluable, col_pos, GMT, rng_mask)
    n_multi = sum(1 for m in pmasks if len(m) > 1)
    log["hallmark"] = {"n_genes_multi_gene_mask": n_multi}

    # fold split (replicate EXP-030 exactly)
    ind_ids = samples["individual_ID"].unique()
    tissue = (samples.groupby("individual_ID")["primary_tumour_type"]
              .agg(lambda s: s.mode().iat[0]))
    ind_group = tissue.reindex(ind_ids).map(
        lambda t: t if t in (CANCER_MAIN, CANCER_SECOND) else "Other")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    fold_pairs = []
    for _, te_i in skf.split(ind_ids, ind_group.to_numpy()):
        te_inds = set(ind_ids[te_i].tolist())
        te_mask = samples["individual_ID"].isin(te_inds).to_numpy()
        te_pos = np.where(te_mask)[0]
        tr_pos = np.where(~te_mask)[0]
        fold_pairs.append((tr_pos, te_pos))
    log["folds"] = {"test_sizes": [int(len(te)) for _, te in fold_pairs]}

    model_fns = {
        "ridge_independent": model_ridge,
        "lowrank_multitask": model_lowrank,
        "pathway_masked": lambda Xtr, Ytr, Xte: _masked_ridge_both(Xtr, Ytr, Xte, pmasks),
        "random_group_control": lambda Xtr, Ytr, Xte: _masked_ridge_both(Xtr, Ytr, Xte, rmasks),
    }
    results: dict = {m: {} for m in model_fns}
    results["mlp_shared"] = {}

    fold_ids = range(N_FOLDS) if not SMOKE else [0]
    tiers = TIERS if not SMOKE else [1.00]

    for fi in fold_ids:
        tr_pos, te_pos = fold_pairs[fi]
        tr_inds = samples.iloc[tr_pos]["individual_ID"].unique()
        rng_tier = np.random.default_rng(SPLIT_SEED + 1000 + fi)
        order = rng_tier.permutation(len(tr_inds))
        for tier in tiers:
            k = max(int(round(tier * len(tr_inds))), 8)
            keep_inds = set(tr_inds[order[:k]].tolist())
            tr_sub = np.where(samples["individual_ID"].isin(keep_inds).to_numpy())[0]
            Xtr_raw = X[tr_sub].astype(np.float64)
            Ytr_raw = Y[tr_sub].astype(np.float64)
            Xte_raw = X[te_pos].astype(np.float64)
            med = np.nanmedian(Xtr_raw, axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            Xtr_f = np.where(np.isnan(Xtr_raw), med[None, :], Xtr_raw)
            Xte_f = np.where(np.isnan(Xte_raw), med[None, :], Xte_raw)
            Xtr_z, Xte_z = _zscore_fit_apply(Xtr_f, Xte_f)
            gmean = np.nanmean(Ytr_raw, axis=0)
            gmean = np.where(np.isfinite(gmean), gmean, 0.0)
            Ytr_f = np.where(np.isnan(Ytr_raw), gmean[None, :], Ytr_raw)
            Yte = Y[te_pos].astype(np.float64)
            key = f"{tier:.2f}"

            for mname, fn in model_fns.items():
                pred_tr, pred_te = fn(Xtr_z, Ytr_f, Xte_z)
                rho_te = _gene_spearman(pred_te.T, Yte.T)
                rho_tr = _gene_spearman(pred_tr.T, Ytr_f.T)
                results[mname].setdefault(key, []).append({
                    "fold": fi + 1, "train_macro": _macro(rho_tr),
                    "val_macro": _macro(rho_te),
                    "val_essential_macro": _macro(rho_te[ess_mask]),
                    "val_variable_macro": _macro(rho_te[~ess_mask]),
                    "n_train": int(len(tr_sub)), "n_test": int(len(te_pos)),
                    "n_val_genes": int(np.isfinite(rho_te).sum())})
            # MLP (multi-seed average)
            pred_tr = np.zeros((len(tr_sub), G), dtype=np.float64)
            pred_te = np.zeros((len(te_pos), G), dtype=np.float64)
            for sd in MLP_SEEDS:
                ptr, pte = model_mlp(Xtr_z, Ytr_f, Xte_z, sd)
                pred_tr += ptr / len(MLP_SEEDS)
                pred_te += pte / len(MLP_SEEDS)
            rho_te = _gene_spearman(pred_te.T, Yte.T)
            rho_tr = _gene_spearman(pred_tr.T, Ytr_f.T)
            results["mlp_shared"].setdefault(key, []).append({
                "fold": fi + 1, "train_macro": _macro(rho_tr),
                "val_macro": _macro(rho_te),
                "val_essential_macro": _macro(rho_te[ess_mask]),
                "val_variable_macro": _macro(rho_te[~ess_mask]),
                "n_train": int(len(tr_sub)), "n_test": int(len(te_pos)),
                "n_val_genes": int(np.isfinite(rho_te).sum())})
            print(f"  fold {fi+1} tier {tier:.2f} done", flush=True)

    summary = []
    for mname, tiers_dict in results.items():
        for tier, folds_list in sorted(tiers_dict.items()):
            def agg(field):
                vals = [r[field] for r in folds_list if np.isfinite(r[field])]
                return float(np.mean(vals)) if vals else float("nan")
            summary.append({
                "model": mname, "tier": tier,
                "train_macro_mean": agg("train_macro"),
                "val_macro_mean": agg("val_macro"),
                "val_essential_macro_mean": agg("val_essential_macro"),
                "val_variable_macro_mean": agg("val_variable_macro"),
                "n_folds": len(folds_list),
                "mean_train_samples": float(np.mean([r["n_train"] for r in folds_list])),
                "mean_test_samples": float(np.mean([r["n_test"] for r in folds_list])),
                "mean_val_genes": float(np.mean([r["n_val_genes"] for r in folds_list])),
                "per_fold": folds_list,
            })

    log["summary_table"] = summary
    log["config"] = {
        "input": "biobank gene-level expression, within-tier-train z-score",
        "output": "biobank LFC dependency (evaluable = expr∩dep)",
        "split": "EXP-030 individual-grouped 5-fold, seed=20260918",
        "tiers": "25/50/100% nested train-patient subsets, test fixed",
        "search_budget_per_model": "1 fixed candidate (<=20); MLP uses 3 init seeds",
        "random_control_matches_pathway": "same mask sizes + alpha, random gene grouping",
    }
    log["elapsed_seconds"] = round(time.time() - t0, 1)
    (OUT_DIR / "f3_summary.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"[exp035_f3] done in {log['elapsed_seconds']}s", flush=True)
    for row in summary:
        print(f"  {row['model']:22s} tier={row['tier']:4s} "
              f"train={row['train_macro_mean']:.4f} val={row['val_macro_mean']:.4f} "
              f"ess={row['val_essential_macro_mean']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())