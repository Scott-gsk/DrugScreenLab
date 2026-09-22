#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-026 数据清洗消融（任务 4）：4 维度 × 3 版本，单变量替换。

规格（预登记于 experiments/records/EXP-026.md 任务 4 与第二波方案 §三 EXP-026）：
- 固定基线（reference）＝ top-2000 训练折方差基因 + z-score + 基因直接输入
  + 原始 DSS 响应；
- 单变量替换其余固定：基因空间（GDSC∩Kryzei 交集全谱 / Hallmark 基因并集）、
  归一化（分位数 / 秩）、通路特征（Hallmark ssGSEA / Reactome 通路均值 PCA /
  PROGENy 式 14 通路）、响应口径（逐药标准化 / 残差化）；
- 同一 Ridge（联合 [表达特征, 药物 one-hot]，alpha=1.0，确定性算法只跑 1 轮）
  + 同一小 MLP（DualMLP h64 depth1 药物嵌入 64，AdamW wd=0.02，epochs=50，
  batch=256，3 种子）；
- 协议：FROZEN 5 折患者隔离；全部 48 药训练行参与拟合（每折约 2500 行），
  评估仅 known 14 药（947 行）；所有预处理与 y 变换只在训练折拟合；
- 终点：三分层终点（evaluate_stratified）+ 每维度内版本 vs 版本 A 的
  患者级配对 bootstrap（2000 次，非结构 NDCG@5 差值，MLP 患者分数先跨种子平均）；
- B0 参照行（训练折全部行逐药均值）同台报告。

PROGENy 式说明：14 条经典信号通路的 Hallmark 映射基因均值聚合（footprint
均值口径），非官方 PROGENy 加权模型（权重未入库），映射表预登记于本脚本。

输出：
- artifacts/exp026_ablation/ablation_results.json
- artifacts/exp026_ablation/ablation_predictions.parquet
- docs/reports/exp026_datacleaning_ablation.md（单独报告脚本生成）
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset

from drug_screen.data.normalization import (
    QuantileNormalizer,
    RankNormalizer,
    ZScoreScaler,
)
from drug_screen.evaluation.stratified_metrics import (
    evaluate_stratified,
    full_panel_patient_values,
    load_structural_drugs,
    nonstructural_patient_values,
)

ROOT = Path("/mnt/d/Code/DrugScreenLab")
PROC = ROOT / "data/processed/transfer/mcrc_v3"
SPLIT = ROOT / "data/splits/kryeziu_patient_outer5fold_FROZEN.json"
SSGSEA = ROOT / "data/processed/identity/pathway/kryeziu_hallmark_ssgsea_DRAFT.parquet"
HALLMARK_JSON = ROOT / "data/processed/transfer/pathways_v2/hallmark.json"
REACTOME_JSON = ROOT / "data/processed/transfer/pathways_v2/reactome.json"
INTERSECTION_TXT = (
    ROOT / "data/processed/identity/transfer/gdsc_kryeziu_gene_intersection_ordered_DRAFT.txt")
STRUCTURAL_PATH = ROOT / "data/processed/identity/kryzei_structural_drugs_FROZEN.json"
OUT_DIR = ROOT / "artifacts/exp026_ablation"

SEEDS = [20260919, 20260920, 20260921]
PCA_DIM = 64
TOP_K_GENES = 2000
RIDGE_ALPHA = 1.0
MLP_HIDDEN, MLP_DEPTH, MLP_DRUG_DIM = 64, 1, 64
MLP_EPOCHS, MLP_BATCH, MLP_LR, MLP_WD = 50, 256, 1e-3, 0.02
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260919

# PROGENy 式 14 通路：经典信号轴 → Hallmark 映射（近似，预登记）
PROGENY_LIKE_HALLMARK = {
    "androgen": "HALLMARK_ANDROGEN_RESPONSE",
    "estrogen": "HALLMARK_ESTROGEN_RESPONSE_EARLY",
    "hypoxia": "HALLMARK_HYPOXIA",
    "jak_stat": "HALLMARK_IL6_JAK_STAT3_SIGNALING",
    "ras_mapk": "HALLMARK_KRAS_SIGNALING_UP",
    "nfkb_tnf": "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "pi3k_akt": "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
    "mtorc1": "HALLMARK_MTORC1_SIGNALING",
    "tgfb": "HALLMARK_TGF_BETA_SIGNALING",
    "interferon": "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    "vegf_angio": "HALLMARK_ANGIOGENESIS",
    "wnt": "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
    "p53": "HALLMARK_P53_PATHWAY",
    "notch": "HALLMARK_NOTCH_SIGNALING",
}

# config -> (维度内角色, 基因空间, 归一化, 特征族, 响应口径)
CONFIGS = {
    "ref": ("reference", "topvar", "zscore", "genes", "raw"),
    "gene_intersect": ("B", "intersect", "zscore", "genes", "raw"),
    "gene_hallmark": ("C", "hallmark_union", "zscore", "genes", "raw"),
    "norm_quantile": ("B", "topvar", "quantile", "genes", "raw"),
    "norm_rank": ("C", "topvar", "rank", "genes", "raw"),
    "pathway_ssgsea": ("A", None, "zscore", "ssgsea", "raw"),
    "pathway_reactome_pca": ("B", None, "zscore", "reactome_pca", "raw"),
    "pathway_progeny": ("C", None, "zscore", "progeny", "raw"),
    "target_drugstd": ("B", "topvar", "zscore", "genes", "drugstd"),
    "target_residual": ("C", "topvar", "zscore", "genes", "residual"),
}
DIMENSIONS = {
    "gene_space": ["ref", "gene_intersect", "gene_hallmark"],
    "normalization": ["ref", "norm_quantile", "norm_rank"],
    "pathway_features": ["pathway_ssgsea", "pathway_reactome_pca", "pathway_progeny"],
    "response_scale": ["ref", "target_drugstd", "target_residual"],
}


class DualMLP(nn.Module):
    """EXP-024 同构小 MLP（h64 depth1 + 药物嵌入 64 + 128 头）。"""

    def __init__(self, n_in: int, n_drugs: int, drug_dim: int = 64,
                 hidden: int = 64, depth: int = 1):
        super().__init__()
        self.drug_emb = nn.Embedding(n_drugs, drug_dim)
        layers: list[nn.Module] = []
        d = n_in
        for _ in range(max(1, depth)):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.2)]
            d = hidden
        if d != 128:
            layers += [nn.Linear(d, 128), nn.ReLU()]
        self.expr = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(128 + drug_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x_expr, drug_idx):
        h = torch.cat([self.expr(x_expr), self.drug_emb(drug_idx)], dim=-1)
        return self.head(h).squeeze(-1)


def load_data():
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    samples = pd.read_parquet(PROC / "samples.parquet")
    pairs = pd.read_parquet(PROC / "pairs.parquet")
    npz = np.load(PROC / "indexed.npz", allow_pickle=True)
    sids = [str(x) for x in np.asarray(npz["sample_id"]).tolist()]
    expr = np.asarray(npz["expression"], dtype=np.float32)
    genes = [str(g) for g in np.asarray(npz["ordered_genes"]).tolist()]
    sid_to_idx = {s: i for i, s in enumerate(sids)}
    gene_to_col = {g: i for i, g in enumerate(genes)}
    samples = samples.copy()
    samples["sample_id"] = samples["sample_id"].astype(str)
    samples["patient_id"] = samples["patient_id"].astype(str)
    pairs = pairs.copy()
    pairs["sample_id"] = pairs["sample_id"].astype(str)
    pairs["drug_id"] = pairs["drug_id"].astype(str)
    merged = pairs.merge(
        samples[["sample_id", "patient_id"]],
        on="sample_id", how="inner").dropna(subset=["response"])
    ss = pd.read_parquet(SSGSEA)
    if "sample_id" in ss.columns:
        ss = ss.set_index("sample_id")
    ss.index = ss.index.astype(str)
    hall = json.loads(HALLMARK_JSON.read_text(encoding="utf-8"))["pathways"]
    react = json.loads(REACTOME_JSON.read_text(encoding="utf-8"))["pathways"]
    inter_genes = [l.strip() for l in INTERSECTION_TXT.read_text(
        encoding="utf-8").splitlines() if l.strip()]
    structural = load_structural_drugs(STRUCTURAL_PATH)
    return split, merged, expr, genes, sid_to_idx, gene_to_col, ss, hall, \
        react, inter_genes, structural


class FeatureSpace:
    """每折每配置的特征构建（训练折拟合，测试折变换）。

    gene_space / normalization 只作用于 ``genes`` 特征族；通路特征族
    （ssgsea / reactome_pca / progeny）内部自带基因层 z-score（训练折拟合）。
    """

    def __init__(self, expr, genes, gene_to_col, sid_to_idx, ss_matrix,
                 hall, react, inter_genes):
        self.expr = expr
        self.genes = genes
        self.gene_to_col = gene_to_col
        self.sid_to_idx = sid_to_idx
        self.ss = ss_matrix
        self.hall = hall
        self.react = react
        self.inter_genes = inter_genes

    def build(self, train_sids, test_sids, gene_space, normalization, family):
        """返回 (Xtr_sample, Xte_sample, n_features)：样本级特征矩阵。"""
        tr_i = np.asarray([self.sid_to_idx[s] for s in train_sids], int)
        te_i = np.asarray([self.sid_to_idx[s] for s in test_sids], int)

        if family == "ssgsea":
            cols = list(self.ss.columns)
            Xtr = self.ss.loc[train_sids, cols].to_numpy(np.float64)
            Xte = self.ss.loc[test_sids, cols].to_numpy(np.float64)
            sc = ZScoreScaler().fit(Xtr)
            return (sc.transform(Xtr).astype(np.float32),
                    sc.transform(Xte).astype(np.float32), len(cols))

        if family in ("reactome_pca", "progeny"):
            pws = ({k: self.hall[v] for k, v in PROGENY_LIKE_HALLMARK.items()}
                   if family == "progeny" else self.react)
            # 基因层 z-score（训练折拟合，全部表达基因）
            sc = ZScoreScaler().fit(self.expr[tr_i].astype(np.float64))
            Ztr = sc.transform(self.expr[tr_i].astype(np.float64))
            Zte = sc.transform(self.expr[te_i].astype(np.float64))
            # 通路均值聚合（Reactome 过滤：表达矩阵内基因数 10..500）
            sets = []
            for name, glist in pws.items():
                cols = [self.gene_to_col[g] for g in glist
                        if g in self.gene_to_col]
                if family == "progeny" or 10 <= len(cols) <= 500:
                    sets.append((name, cols))
            if not sets:
                raise RuntimeError(f"no pathway sets survived filter: {family}")
            Ttr = np.stack([Ztr[:, cols].mean(axis=1) for _, cols in sets], axis=1)
            Tte = np.stack([Zte[:, cols].mean(axis=1) for _, cols in sets], axis=1)
            if family == "progeny":
                # 通路均值特征 z-score（训练折拟合，与 ssGSEA 口径对齐）
                sc = ZScoreScaler().fit(Ttr)
                return (sc.transform(Ttr).astype(np.float32),
                        sc.transform(Tte).astype(np.float32), len(sets))
            mu = Ttr.mean(axis=0, keepdims=True)
            Xc = Ttr - mu
            k = min(PCA_DIM, Xc.shape[0] - 1, Xc.shape[1])
            _, _, vt = np.linalg.svd(Xc, full_matrices=False)
            W = vt[:k].T
            Ptr = Xc @ W
            Pte = (Tte - mu) @ W
            # PCA 分量 z-score（训练折拟合）：消除分量间尺度差，
            # 否则低方差分量使 Ridge 数值病态（消融 smoke 检查发现）
            sc = ZScoreScaler().fit(Ptr)
            return (sc.transform(Ptr).astype(np.float32),
                    sc.transform(Pte).astype(np.float32), int(k))

        # genes 族
        if gene_space == "topvar":
            var = self.expr[tr_i].var(axis=0)
            gene_idx = np.argsort(-var)[:TOP_K_GENES]
        elif gene_space == "intersect":
            names = [g for g in self.inter_genes if g in self.gene_to_col]
            gene_idx = np.asarray([self.gene_to_col[g] for g in names], int)
        elif gene_space == "hallmark_union":
            union = set()
            for v in self.hall.values():
                union.update(v)
            names = sorted(g for g in union if g in self.gene_to_col)
            gene_idx = np.asarray([self.gene_to_col[g] for g in names], int)
        else:
            raise ValueError(gene_space)
        Xtr = self.expr[np.ix_(tr_i, gene_idx)].astype(np.float64)
        Xte = self.expr[np.ix_(te_i, gene_idx)].astype(np.float64)
        if normalization == "zscore":
            sc = ZScoreScaler().fit(Xtr)
        elif normalization == "quantile":
            sc = QuantileNormalizer().fit(Xtr)
        elif normalization == "rank":
            sc = RankNormalizer().fit(Xtr)
        else:
            raise ValueError(normalization)
        return (sc.transform(Xtr).astype(np.float32),
                sc.transform(Xte).astype(np.float32), int(Xtr.shape[1]))


def drug_stats(dtr, ytr, n_drugs, kind):
    """训练折逐药统计（drugstd: mean/std；residual: mean）。"""
    counts = np.bincount(dtr, minlength=n_drugs)
    sums = np.bincount(dtr, weights=ytr.astype(float), minlength=n_drugs)
    sqs = np.bincount(dtr, weights=ytr.astype(float) ** 2, minlength=n_drugs)
    gm = float(np.mean(ytr))
    mu = np.where(counts > 0, sums / np.maximum(counts, 1), gm)
    if kind == "residual":
        return mu, None
    var = np.where(counts > 1,
                   (sqs - counts * mu ** 2) / np.maximum(counts - 1, 1), 1.0)
    return mu, np.sqrt(np.maximum(var, 1e-6))


def fit_ridge(Xtr, dtr, ytr, Xte, dte, n_drugs):
    """联合 Ridge：[表达特征, 药物 one-hot] → y。返回测试预测。"""
    Dtr = np.zeros((len(dtr), n_drugs), np.float64)
    Dtr[np.arange(len(dtr)), dtr] = 1.0
    Dte = np.zeros((len(dte), n_drugs), np.float64)
    Dte[np.arange(len(dte)), dte] = 1.0
    A = np.concatenate([Xtr.astype(np.float64), Dtr], axis=1)
    B = np.concatenate([Xte.astype(np.float64), Dte], axis=1)
    m = Ridge(alpha=RIDGE_ALPHA, solver="lsqr", fit_intercept=True,
              random_state=0)
    m.fit(A, ytr.astype(np.float64))
    return m.predict(B)


def fit_mlp(device, Xtr, dtr, ytr, Xte, dte, n_drugs, seed):
    """小 MLP（DualMLP）单折训练，返回测试预测。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = DualMLP(Xtr.shape[1], n_drugs, drug_dim=MLP_DRUG_DIM,
                    hidden=MLP_HIDDEN, depth=MLP_DEPTH).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr),
                      torch.from_numpy(dtr.astype(np.int64)),
                      torch.from_numpy(ytr.astype(np.float32))),
        batch_size=MLP_BATCH, shuffle=True)
    model.train()
    for _ in range(MLP_EPOCHS):
        for xb, db, yb in loader:
            xb, db, yb = xb.to(device), db.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb, db), yb).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte).to(device),
                     torch.from_numpy(dte.astype(np.int64)).to(device))
    return pred.cpu().numpy().astype(np.float64)


def _invert(pred, inv, d_idx):
    if inv is None:
        return pred
    if inv[0] == "add":
        return pred + inv[1][d_idx]
    if inv[0] == "affine":
        return pred * inv[2][d_idx] + inv[1][d_idx]
    raise ValueError(inv)


def _frame(fold, te_df, yte, pred, cfg, model, seed, known_set):
    return pd.DataFrame({
        "fold": fold,
        "patient_id": te_df["patient_id"].to_numpy(),
        "model_id": te_df["sample_id"].to_numpy(),
        "drug_id": te_df["drug_id"].to_numpy(),
        "observed": yte,
        "predicted": pred,
        "config": cfg,
        "model": model,
        "seed": seed,
        "algorithm": f"{cfg}__{model}",
        "source_drug_status": np.where(
            te_df["drug_id"].isin(known_set), "known", "probe"),
    })


def patient_bootstrap(diff: dict, seed: int = BOOTSTRAP_SEED,
                      resamples: int = BOOTSTRAP_RESAMPLES):
    ids = sorted(diff)
    vals = [diff[i] for i in ids]
    finite = [v for v in vals if v is not None and np.isfinite(v)]
    if len(finite) < 2:
        return None
    rng = np.random.default_rng(seed)
    arr = np.asarray(finite)
    draws = rng.choice(arr, size=(resamples, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"point": float(np.mean(arr)), "low": float(lo), "high": float(hi),
            "n": len(arr), "resamples": resamples}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def main():
    t0 = time.time()
    smoke = os.environ.get("SMOKE") == "1"
    global MLP_EPOCHS
    if smoke:
        MLP_EPOCHS = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE={device} SMOKE={smoke}", flush=True)
    split, merged, expr, genes, sid_to_idx, gene_to_col, ss, hall, react, \
        inter_genes, structural = load_data()
    known_set = structural.structural | structural.nonstructural
    fs = FeatureSpace(expr, genes, gene_to_col, sid_to_idx, ss, hall, react,
                      inter_genes)

    drugs = sorted(merged["drug_id"].unique())
    drug_to_i = {d: i for i, d in enumerate(drugs)}
    n_drugs = len(drugs)
    print(f"n_drugs={n_drugs} n_pairs={len(merged)} "
          f"n_patients={merged['patient_id'].nunique()}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_rows: list[pd.DataFrame] = []
    fold_meta = []
    feature_dims: dict[str, list[int]] = {}
    folds_iter = split["folds"][:1] if smoke else split["folds"]
    for fold in folds_iter:
        f_id = int(fold["fold"])
        train_p = set(map(str, fold["train_patient_ids"]))
        test_p = set(map(str, fold["test_patient_ids"]))
        tr_df = merged[merged["patient_id"].isin(train_p)]
        te_df = merged[merged["patient_id"].isin(test_p)]
        tr_sids = tr_df["sample_id"].tolist()
        te_sids = te_df["sample_id"].tolist()
        uniq_tr = list(dict.fromkeys(tr_sids))
        uniq_te = list(dict.fromkeys(te_sids))
        row_of_tr = {s: i for i, s in enumerate(uniq_tr)}
        row_of_te = {s: i for i, s in enumerate(uniq_te)}
        dtr = np.asarray([drug_to_i[d] for d in tr_df["drug_id"]], np.int64)
        dte = np.asarray([drug_to_i[d] for d in te_df["drug_id"]], np.int64)
        ytr = tr_df["response"].to_numpy(np.float64)
        yte = te_df["response"].to_numpy(np.float64)
        fold_meta.append({
            "fold": f_id, "n_train_patients": len(train_p),
            "n_test_patients": len(test_p), "n_train_rows": int(len(tr_df)),
            "n_test_rows": int(len(te_df)),
        })
        print(f"FOLD {f_id}: train_rows={len(tr_df)} test_rows={len(te_df)}",
              flush=True)

        # B0：训练折全部行逐药均值（未见药回落全局均值）
        mu_all, _ = drug_stats(dtr, ytr, n_drugs, "residual")
        gm = float(np.mean(ytr))
        seen = np.zeros(n_drugs, bool)
        seen[np.unique(dtr)] = True
        b0_te = np.where(seen[dte], mu_all[dte], gm)
        pred_rows.append(_frame(f_id, te_df, yte, b0_te, "B0", "ridge", 0,
                                known_set))

        for cfg_name, (role, space, norm, family, target) in CONFIGS.items():
            Xs_tr, Xs_te, n_feat = fs.build(uniq_tr, uniq_te, space, norm,
                                             family)
            feature_dims.setdefault(cfg_name, []).append(int(n_feat))
            # 样本级 → pair 行级扩展
            Xtr = Xs_tr[[row_of_tr[s] for s in tr_sids]]
            Xte = Xs_te[[row_of_te[s] for s in te_sids]]
            if target == "raw":
                ytr_fit, inv = ytr, None
            elif target == "residual":
                mu, _ = drug_stats(dtr, ytr, n_drugs, "residual")
                ytr_fit = ytr - mu[dtr]
                inv = ("add", mu)
            elif target == "drugstd":
                mu, sd = drug_stats(dtr, ytr, n_drugs, "drugstd")
                ytr_fit = (ytr - mu[dtr]) / sd[dtr]
                inv = ("affine", mu, sd)
            else:
                raise ValueError(target)

            pr = fit_ridge(Xtr, dtr, ytr_fit, Xte, dte, n_drugs)
            pr = _invert(pr, inv, dte)
            pred_rows.append(_frame(f_id, te_df, yte, pr, cfg_name, "ridge", 0,
                                    known_set))
            for seed in SEEDS:
                pm = fit_mlp(device, Xtr, dtr, ytr_fit, Xte, dte, n_drugs,
                             seed)
                pm = _invert(pm, inv, dte)
                pred_rows.append(_frame(f_id, te_df, yte, pm, cfg_name, "mlp",
                                        seed, known_set))
            print(f"  cfg={cfg_name} role={role} n_features={n_feat} "
                  f"done ({time.time()-t0:.0f}s)", flush=True)

    preds = pd.concat(pred_rows, ignore_index=True)
    pred_path = OUT_DIR / "ablation_predictions.parquet"
    preds.to_parquet(pred_path, index=False)
    print("WROTE", pred_path, flush=True)

    # ---- 评估：三分层终点 ----
    known = preds[preds["source_drug_status"] == "known"].copy()
    eval_rows = []
    per_patient_ns: dict[tuple, dict] = {}
    for (cfg, model, seed), g in known.groupby(["config", "model", "seed"]):
        summary, _, _ = evaluate_stratified(g, structural)
        eval_rows.append({
            "config": cfg, "model": model, "seed": int(seed),
            "nonstructural_ndcg5": summary.nonstructural_ndcg5_macro,
            "per_drug_spearman": summary.per_drug_patient_spearman_macro,
            "full_panel_ndcg5": summary.full_panel_ndcg5_macro,
            "n_patients": int(summary.n_patients_nonstructural),
        })
        ns = nonstructural_patient_values(g, structural)
        frame = ns.rename("ns").reset_index()
        pats = frame.groupby(frame.columns[0])["ns"].mean().to_dict()
        per_patient_ns[(cfg, model, int(seed))] = pats

    eval_df = pd.DataFrame(eval_rows)
    agg = (eval_df.groupby(["config", "model"])
           .agg(ns_mean=("nonstructural_ndcg5", "mean"),
                ns_sd=("nonstructural_ndcg5", "std"),
                sp_mean=("per_drug_spearman", "mean"),
                sp_sd=("per_drug_spearman", "std"),
                fp_mean=("full_panel_ndcg5", "mean"),
                fp_sd=("full_panel_ndcg5", "std"),
                n_seeds=("seed", "count")).reset_index())

    # ---- 配对 bootstrap：每维度内 B/C vs A ----
    def patient_scores(cfg: str, model: str) -> dict:
        per: dict[str, list] = {}
        for (c, m, _), pats in per_patient_ns.items():
            if c == cfg and m == model:
                for pid, v in pats.items():
                    per.setdefault(pid, []).append(v)
        return {pid: float(np.nanmean(v)) for pid, v in per.items()}

    boot = {}
    for dim, members in DIMENSIONS.items():
        base = members[0]
        for model in ("ridge", "mlp"):
            a = patient_scores(base, model)
            for other in members[1:]:
                b = patient_scores(other, model)
                diff = {}
                for pid in a:
                    if pid in b and np.isfinite(a[pid]) and np.isfinite(b[pid]):
                        diff[pid] = b[pid] - a[pid]
                boot[f"{dim}:{other}_vs_{base}:{model}"] = patient_bootstrap(diff)

    results = {
        "experiment": "EXP-026_task4_datacleaning_ablation",
        "protocol": {
            "split": str(SPLIT), "n_folds": 5,
            "fold_meta": fold_meta,
            "known_drugs": sorted(known_set),
            "ridge": (f"joint [expr, drug-onehot] Ridge alpha={RIDGE_ALPHA} "
                      "solver=lsqr (deterministic, single run)"),
            "mlp": (f"DualMLP hidden={MLP_HIDDEN} depth={MLP_DEPTH} "
                    f"drug_emb={MLP_DRUG_DIM} AdamW(lr={MLP_LR}, wd={MLP_WD}) "
                    f"epochs={MLP_EPOCHS} batch={MLP_BATCH}"),
            "seeds": SEEDS,
            "preprocessing": "all fitted on training folds only "
                             "(patient-isolated outer CV)",
            "endpoint_module": "drug_screen.evaluation.stratified_metrics",
            "feature_dims_per_fold": feature_dims,
        },
        "configs": {k: {"role": v[0], "gene_space": v[1],
                         "normalization": v[2], "feature_family": v[3],
                         "response": v[4]} for k, v in CONFIGS.items()},
        "progeny_like_mapping": PROGENY_LIKE_HALLMARK,
        "eval_rows": eval_rows,
        "aggregate": agg.to_dict(orient="records"),
        "bootstrap": boot,
        "seconds": time.time() - t0,
    }
    out_path = OUT_DIR / "ablation_results.json"
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False,
                   default=_json_default) + "\n", encoding="utf-8")
    print("WROTE", out_path, flush=True)
    print(f"TOTAL {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
