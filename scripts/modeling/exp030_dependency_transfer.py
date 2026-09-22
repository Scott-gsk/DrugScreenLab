#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-030 dependency-map transfer test: DepMap cell lines -> organoid CRISPR
dependency (five arms vs gene-mean baseline).

Pre-registered contract (experiments/records/EXP-030.md, pinned BEFORE run):
- Task: per-gene dependency regression. Source = DepMap 24Q2 cell lines
  (log2TPM+1 expression, Chronos gene effect, negative = essential);
  target = organoid biobank 162 models (author log2TPM expression, CRISPR
  LFC dependency, minLib dual-library rule from EXP-026).
- Gene spaces: expression genes = source expr symbols n biobank expr
  symbols; dependency genes = CRISPRGeneEffect symbols n biobank LFC genes;
  evaluable = both intersections. If evaluable < 2000 -> preregistered
  downgrade (abort with message).
- Split: target 162 organoids grouped by individual_ID, StratifiedKFold 5
  folds on cancer strata (Colorectal / Oesophageal / Other); source DepMap
  fully used for source models + 20% source-internal validation (report
  only).
- Five arms per fold: B0 gene mean (train-fold per-gene LFC mean); ZS
  zero-shot (source multi-output Ridge on source PCA100, applied to target
  via the FROZEN source fill+z+PCA+scaler chain); CAL output calibration
  (ZS prediction + per-gene 2-parameter linear fit on train fold, closed
  form); DA domain adaptation (EXP-029 OrthoAE shared/private decomposition,
  2 domains = source lines + biobank 255 full expression corpus incl. 93
  non-CRISPR organoids, shared 64-d code -> per-fold multi-output Ridge);
  HD from-scratch (train-fold multi-output Ridge on target PCA100).
- Feature engineering: expression-intersection genes, within-domain z
  (source fit on all source lines; target fit on train fold), PCA top-100
  (source frozen on all source lines; target fit on train fold), Ridge
  alpha=1.0 on standardized features.
- Y-matrix NaN policy (registered decision): per gene, train-fold NaN frac
  < 10% -> fill with train-fold gene mean; >= 10% -> gene marked NaN for
  ALL arms in that fold (keeps arms paired-comparable). Same rule on the
  source Y matrix decides which genes can train a ZS/CAL source model.
- Primary endpoint: per-fold test per-gene Spearman (prediction vs
  observed LFC across test samples; >= 30 test rows per gene), macro
  averaged across the all-arm evaluable gene set; 5-fold mean; each arm vs
  B0 paired difference with individual-level bootstrap (2000, seed
  20260918) 95% CI computed on pooled 5-fold predictions.
- Secondary: common-essential AUROC (per test sample, negated prediction as
  score, prior = biobank CRISPRInferredCommonEssentials only -- the DepMap
  AchillesCommonEssentialControls file is not part of the local registered
  assets, a preregistered deviation logged in audit_log).
- Stratified reporting (must): neighbor vs no-neighbor organoids (min
  distance to all source lines in expression z space, 1-Pearson, threshold
  = P95 of within-source nearest-neighbor distances); cancer strata
  (Colorectal main layer); core-essential vs variable genes.
- Products: predictions.parquet (fold/arm/sample/gene/predicted/observed),
  gene_transferability_map.parquet (per-gene per-arm mean fold Spearman),
  arm_summary.json, stratified_report.json, audit_log.json.
- Verdict rules pinned in EXP-030.md (transfer arm CI lower bound > 0 etc.).

Implementation decisions where the record was terse (all logged in
audit_log["decisions"]): see DECISIONS below main().
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from drug_screen.data.organoid_depmap import resolve_dual_library
from drug_screen.transfer.cohorts import sha256

ROOT = Path("/mnt/d/Code/DrugScreenLab")
OUT_DIR = ROOT / "artifacts/exp030_dependency_transfer"
DEPMAP_DIR = ROOT / "data/raw/depmap/24q2"
GE_CSV = DEPMAP_DIR / "CRISPRGeneEffect.csv"
EXPR_CSV = DEPMAP_DIR / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
BIO_DIR = ROOT / "data/processed/transfer/biobank_functional_v1"
BIO_SAMPLES = BIO_DIR / "samples.parquet"
BIO_EXPR = BIO_DIR / "expression_author_log2tpm.parquet"
BIO_DEP = BIO_DIR / "dependency_by_library.parquet"
BIO_RAW_CSV = (ROOT / "data/raw/transfer_public_assets/organoid_biobank/"
               "data_RNAseq_ALL-organoids.csv")
ESSENTIALS_CSV = (ROOT / "data/raw/transfer_public_assets/organoid_biobank/"
                  "CRISPRInferredCommonEssentials.csv")

GE_SHA = "d155149181308f8e16b6a4677cde3e5af1ddef68ada1947486ae8c0bcb231452"
EXPR_SHA = "39eff342bbbbe0a40208545e93a69cc85a945899592f9d683f87e3b2bb121670"
BIO_RAW_SHA = "92859e18417c3737956236e6990f8001a0d2cc352890b9355b7ed9603b6a6b1e"

N_FOLDS = 5
PCA_COMPONENTS = 100
RIDGE_ALPHA = 1.0
Y_NAN_MAX = 0.10
MIN_TEST_ROWS = 30
BOOT_N, BOOT_SEED = 2000, 20260918
SPLIT_SEED = 20260918
SOURCE_VAL_FRAC = 0.20
AE_SEEDS = [41, 42, 43]
DA_MAIN_SEED = 41
AE_EPOCHS, AE_BATCH, AE_LR, AE_WD, LAMBDA_ORTH = 30, 128, 1e-3, 1e-5, 1.0
SHARED_DIM, PRIVATE_DIM, HIDDEN = 64, 32, 512
ARMS = ["B0", "ZS", "CAL", "DA", "HD"]
TRANSFER_ARMS = ["ZS", "CAL", "DA"]
DELTA_ARMS = ["ZS", "CAL", "DA", "HD"]
MIN_EVALUABLE_GENES = 2000
CANCER_MAIN = "Colorectal"
CANCER_SECOND = "Oesophageal"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DECISIONS = [
    "Y-matrix NaN: train-fold gene NaN frac <10% -> fill train-fold mean; "
    ">=10% -> gene NaN for ALL arms in that fold (paired comparability); "
    "same rule on source Y decides ZS/CAL trainable genes",
    "ZS/CAL cross-domain alignment: frozen source fill/z/PCA100/scaler "
    "chain (fit on all source lines) applied to target expression; target "
    "NaN filled with train-fold column median",
    "Ridge inputs standardized on training rows, alpha=1.0 (EXP-029 style)",
    "DA main analysis uses the seed-41 shared code; AE seeds 41/42/43 all "
    "trained and reported as stability metrics",
    "source-internal validation: ZS arm only, 80/20 (seed 20260918), "
    "frozen source PCA/z/scaler chain",
    "neighbor diagnostics on within-domain z over all target samples "
    "(EXP-029 T3 convention)",
    "bootstrap pairing on genes valid for both arm and B0 in the resample; "
    "stratified supplementary statistic = within-gene standardized-rank "
    "pooled Pearson (used when stratum < 30 rows per gene)",
    "common-essential prior = biobank CRISPRInferredCommonEssentials only; "
    "DepMap AchillesCommonEssentialControls not in local registered assets "
    "(preregistered deviation from the two-domain union wording)",
    "biobank expression gene ids: '_exp' suffix stripped, duplicated "
    "symbols keep first column (count logged)",
]


# ---------------------------------------------------------------- helpers --

def _symbol(column: str) -> str:
    """'A1BG (1)' / 'A1BG_exp' / 'A1BG' -> 'A1BG'."""
    if column.endswith("_exp"):
        return column[:-4]
    return column.split(" (")[0]


def _fill_nan(mat: np.ndarray) -> tuple[np.ndarray, int]:
    """域内列中位数填充；全 NaN 列填 0。填充列数登记进 audit_log。"""
    if not np.isnan(mat).any():
        return mat, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(mat, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    filled = np.where(np.isnan(mat), med[None, :], mat)
    return filled, int(np.isnan(mat).any(axis=0).sum())


def _zscore(mat: np.ndarray) -> np.ndarray:
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (mat - mu) / sd


def _corr_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 - Pearson; rows are samples. Returns (n_b, n_a)."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = b @ a.T
    den = (np.linalg.norm(b, axis=1)[:, None]
           * np.linalg.norm(a, axis=1)[None, :])
    return 1.0 - num / np.clip(den, 1e-12, None)


def _rank_pair(pred: np.ndarray, obs: np.ndarray) -> tuple:
    """逐基因秩对。pred/obs 形状 (n_genes, n_samples)，NaN=缺失。

    NaN 位置置 +inf 排在秩末尾、随后由掩码剔除，等价于在非缺失子集内
    取平均秩（ties 正确处理，重采样重复样本安全）。
    """
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    r_p = rankdata(np.where(mask, pred, np.inf), axis=1, method="average")
    r_o = rankdata(np.where(mask, obs, np.inf), axis=1, method="average")
    return mask, r_p, r_o


def _macro_from_ranks(mask: np.ndarray, r_p: np.ndarray, r_o: np.ndarray,
                      min_rows: int) -> np.ndarray:
    """主口径：逐基因 Spearman（秩的 Pearson），行数 >= min_rows 才有效。"""
    k = mask.sum(axis=1)
    n = np.maximum(k, 1)
    mp = np.where(mask, r_p, 0.0).sum(axis=1) / n
    mo = np.where(mask, r_o, 0.0).sum(axis=1) / n
    dp = np.where(mask, r_p - mp[:, None], 0.0)
    do = np.where(mask, r_o - mo[:, None], 0.0)
    cov = (dp * do).sum(axis=1)
    var = np.sqrt((dp * dp).sum(axis=1) * (do * do).sum(axis=1))
    rho = np.full(mask.shape[0], np.nan)
    ok = (k >= min_rows) & (var > 0)
    rho[ok] = cov[ok] / var[ok]
    return rho


def _spearman_gene_vec(pred: np.ndarray, obs: np.ndarray,
                       min_rows: int) -> np.ndarray:
    mask, r_p, r_o = _rank_pair(pred, obs)
    return _macro_from_ranks(mask, r_p, r_o, min_rows)


def _pooled_from_ranks(mask: np.ndarray, r_p: np.ndarray,
                       r_o: np.ndarray) -> float:
    """分层补充口径：基因内标准化秩合并 Pearson。"""
    k = mask.sum(axis=1)
    n = np.maximum(k, 1)
    mp = np.where(mask, r_p, 0.0).sum(axis=1) / n
    mo = np.where(mask, r_o, 0.0).sum(axis=1) / n
    dp = np.where(mask, r_p - mp[:, None], 0.0)
    do = np.where(mask, r_o - mo[:, None], 0.0)
    sd_p = np.sqrt((dp * dp).sum(axis=1) / n)
    sd_o = np.sqrt((do * do).sum(axis=1) / n)
    ok = (k >= 2) & (sd_p > 0) & (sd_o > 0)
    if not ok.any():
        return float("nan")
    zp = dp[ok] / sd_p[ok, None]
    zo = do[ok] / sd_o[ok, None]
    vp = zp[mask[ok]].ravel()
    vo = zo[mask[ok]].ravel()
    vp = vp - vp.mean()
    vo = vo - vo.mean()
    den = np.sqrt((vp * vp).sum() * (vo * vo).sum())
    return float((vp * vo).sum() / den) if den > 0 else float("nan")


def _pooled_std_spearman(pred: np.ndarray, obs: np.ndarray) -> float:
    mask, r_p, r_o = _rank_pair(pred, obs)
    return _pooled_from_ranks(mask, r_p, r_o)


# ---------------------------------------------------------------- OrthoAE --

class _MLP(nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i != len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OrthoAE(nn.Module):
    """2-domain shared/private decomposition (EXP-029 architecture)."""

    def __init__(self, n_genes: int):
        super().__init__()
        self.enc_shared = _MLP([n_genes, HIDDEN, 128, SHARED_DIM])
        self.enc_private = nn.ModuleList(
            [_MLP([n_genes, HIDDEN, 128, PRIVATE_DIM]) for _ in range(2)])
        self.decoders = nn.ModuleList(
            [_MLP([SHARED_DIM + PRIVATE_DIM, 128, HIDDEN, n_genes])
             for _ in range(2)])

    def forward(self, x: torch.Tensor, domain: int,
                *, zero_private: bool = False, zero_shared: bool = False):
        zs = self.enc_shared(x)
        zp = self.enc_private[domain](x)
        zs = torch.zeros_like(zs) if zero_shared else zs
        zp = torch.zeros_like(zp) if zero_private else zp
        return self.decoders[domain](torch.cat([zs, zp], dim=-1)), zs, zp


def _ortho_loss(zs: torch.Tensor, zp: torch.Tensor) -> torch.Tensor:
    cross = zs.T @ zp / zs.size(0)
    return (cross ** 2).sum()


def _r2(x: np.ndarray, recon: np.ndarray) -> float:
    sse = float(((x - recon) ** 2).sum())
    sst = float(((x - x.mean(axis=0, keepdims=True)) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


def train_ae(domains: list[np.ndarray], seed: int) -> tuple[OrthoAE, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = OrthoAE(domains[0].shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=AE_LR, weight_decay=AE_WD)
    tensors = [torch.tensor(d, dtype=torch.float32, device=DEVICE)
               for d in domains]
    sizes = [len(d) for d in domains]
    pooled = np.concatenate([np.stack([np.full(n, g), np.arange(n)], axis=1)
                             for g, n in enumerate(sizes)])
    n_total = len(pooled)
    last_loss = float("nan")
    for epoch in range(1, AE_EPOCHS + 1):
        model.train()
        perm = np.random.permutation(n_total)
        tot, batches = 0.0, 0
        for start in range(0, n_total, AE_BATCH):
            sel = pooled[perm[start:start + AE_BATCH]]
            loss = torch.zeros((), device=DEVICE)
            for g in range(2):
                rows = sel[sel[:, 0] == g][:, 1]
                if len(rows) == 0:
                    continue
                x = tensors[g][torch.tensor(rows, device=DEVICE)]
                recon, zs, zp = model(x, g)
                loss = loss + ((recon - x) ** 2).mean()
                loss = loss + LAMBDA_ORTH * _ortho_loss(zs, zp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.item())
            batches += 1
        last_loss = tot / max(batches, 1)
        if not np.isfinite(last_loss):
            raise RuntimeError(f"AE loss not finite at epoch {epoch}")
        if epoch == 1 or epoch % 10 == 0 or epoch == AE_EPOCHS:
            print(f"    ae seed {seed} epoch {epoch}/{AE_EPOCHS} "
                  f"loss={last_loss:.5f}", flush=True)
    model.eval()
    metrics: dict[str, object] = {"final_loss": last_loss}
    with torch.no_grad():
        codes_s, codes_p, labels = [], [], []
        for g, name in enumerate(("depmap", "biobank")):
            x = tensors[g]
            full, _, _ = model(x, g)
            shared_only, _, _ = model(x, g, zero_private=True)
            private_only, _, _ = model(x, g, zero_shared=True)
            xn = x.cpu().numpy()
            metrics[f"{name}_r2_full"] = _r2(xn, full.cpu().numpy())
            metrics[f"{name}_r2_shared"] = _r2(xn, shared_only.cpu().numpy())
            metrics[f"{name}_r2_private"] = _r2(xn, private_only.cpu().numpy())
            codes_s.append(model.enc_shared(x).cpu().numpy())
            codes_p.append(model.enc_private[g](x).cpu().numpy())
            labels.append(np.full(len(x), g))
    Xs = np.concatenate(codes_s)
    Xp = np.concatenate(codes_p)
    y = np.concatenate(labels)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    metrics["domain_acc_shared"] = float(
        cross_val_score(clf, Xs, y, cv=cv, scoring="accuracy").mean())
    metrics["domain_acc_private"] = float(
        cross_val_score(clf, Xp, y, cv=cv, scoring="accuracy").mean())
    return model, metrics


# ------------------------------------------------------------------- main --

def main() -> None:
    t0 = time.time()
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise ValueError("output must be new or empty")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log: dict[str, object] = {"device": str(DEVICE), "decisions": DECISIONS}
    print(f"[exp030] device={DEVICE}", flush=True)

    # ---------- checksums ----------
    for path, want in ((GE_CSV, GE_SHA), (EXPR_CSV, EXPR_SHA),
                       (BIO_RAW_CSV, BIO_RAW_SHA)):
        if sha256(path) != want:
            raise ValueError(f"sha256 mismatch: {path.name}")
    log["sha256_verified"] = [GE_CSV.name, EXPR_CSV.name, BIO_RAW_CSV.name]
    log["essentials_sha256"] = sha256(ESSENTIALS_CSV)
    print("[exp030] checksums ok", flush=True)

    # ---------- target data ----------
    samples = pd.read_parquet(BIO_SAMPLES)
    assert len(samples) == 162 and samples["sample_ID"].is_unique
    sample_ids = samples["sample_ID"].astype(str).tolist()
    bio_expr = pd.read_parquet(BIO_EXPR)
    assert list(bio_expr.index.astype(str)) == sample_ids, (
        "biobank expression rows must match samples.parquet order")
    bio_expr.columns = [_symbol(str(c)) for c in bio_expr.columns]
    n_dup_bio = int(bio_expr.columns.duplicated().sum())
    bio_expr = bio_expr.loc[:, ~bio_expr.columns.duplicated()]
    bio_dep = pd.read_parquet(BIO_DEP)
    dep_res, dep_audit = resolve_dual_library(bio_dep)
    log["dependency_dedup"] = dep_audit
    dep_wide = dep_res.pivot(index="sample_ID", columns="gene", values="LFC")
    log["target"] = {
        "samples": int(len(samples)),
        "individuals": int(samples["individual_ID"].nunique()),
        "expression_genes": int(bio_expr.shape[1]),
        "dependency_genes_pivot": int(dep_wide.shape[1]),
        "biobank_expr_duplicate_symbol_cols": n_dup_bio,
        "tissue_counts": {str(k): int(v) for k, v in
                          samples["primary_tumour_type"]
                          .value_counts().items()},
    }
    del bio_dep, dep_res
    tissue = (samples.groupby("individual_ID")["primary_tumour_type"]
              .agg(lambda s: s.mode().iat[0]))

    essentials = pd.read_csv(ESSENTIALS_CSV)
    ess_genes = sorted({_symbol(str(v)) for v in
                        essentials[essentials.columns[0]]})
    log["essentials_prior"] = {"n_genes": len(ess_genes)}

    # ---------- source data ----------
    print("[exp030] loading DepMap matrices (large CSVs)", flush=True)
    ge = pd.read_csv(GE_CSV, index_col=0)
    ge.columns = [_symbol(str(c)) for c in ge.columns]
    n_dup_ge = int(ge.columns.duplicated().sum())
    ge = ge.loc[:, ~ge.columns.duplicated()]
    expr = pd.read_csv(EXPR_CSV, index_col=0)
    expr.columns = [_symbol(str(c)) for c in expr.columns]
    n_dup_ex = int(expr.columns.duplicated().sum())
    expr = expr.loc[:, ~expr.columns.duplicated()]
    src_ids = sorted(set(ge.index) & set(expr.index))
    if len(src_ids) < 800:
        raise AssertionError(f"source overlap too small: {len(src_ids)}")
    log["source"] = {
        "gene_effect_cell_lines": int(ge.shape[0]),
        "expression_profiles": int(expr.shape[0]),
        "overlap_cell_lines": len(src_ids),
        "gene_effect_genes": int(ge.shape[1]),
        "expression_genes": int(expr.shape[1]),
        "gene_effect_duplicate_symbol_cols": n_dup_ge,
        "expression_duplicate_symbol_cols": n_dup_ex,
    }

    # ---------- gene spaces ----------
    expr_genes = sorted(set(expr.columns) & set(bio_expr.columns))
    dep_genes = sorted(set(ge.columns) & set(dep_wide.columns))
    evaluable = sorted(set(expr_genes) & set(dep_genes))
    log["gene_spaces"] = {
        "expression_intersection": len(expr_genes),
        "dependency_intersection": len(dep_genes),
        "evaluable": len(evaluable),
    }
    if len(evaluable) < MIN_EVALUABLE_GENES:
        raise RuntimeError(
            f"PREREGISTERED DOWNGRADE: evaluable genes {len(evaluable)} < "
            f"{MIN_EVALUABLE_GENES}; downgrade to gene-set stratified report")
    print(f"[exp030] genes: expr={len(expr_genes)} dep={len(dep_genes)} "
          f"evaluable={len(evaluable)}", flush=True)

    # ---------- matrices ----------
    Xs = expr.loc[src_ids, expr_genes].to_numpy(float)    # n_src x G_expr
    Ys = ge.loc[src_ids, evaluable].to_numpy(float)      # n_src x G_eval
    del ge, expr
    Xt = bio_expr.reindex(columns=expr_genes).to_numpy(float)  # 162 x G_expr
    Yt = dep_wide.reindex(index=sample_ids, columns=evaluable)
    Yt = Yt.to_numpy(float)                              # 162 x G_eval
    log["target"]["dependency_observed_frac"] = float(1.0 - np.isnan(Yt).mean())
    log["target"]["expression_missing_frac"] = float(np.isnan(Xt).mean())
    del dep_wide, bio_expr

    # ---------- source pipeline (frozen on all source lines) ----------
    print("[exp030] source pipeline (fill/z/PCA100/scaler frozen)", flush=True)
    Xs_f, src_fill_cols = _fill_nan(Xs)
    mu_s = Xs_f.mean(axis=0, keepdims=True)
    sd_s = Xs_f.std(axis=0, keepdims=True)
    sd_s = np.where(sd_s < 1e-8, 1.0, sd_s)
    z_src = (Xs_f - mu_s) / sd_s
    pca_src = PCA(n_components=PCA_COMPONENTS).fit(z_src)
    Xs_pca = pca_src.transform(z_src)
    sc_src = StandardScaler().fit(Xs_pca)
    Xs_std = sc_src.transform(Xs_pca)
    src_nan_frac = np.isnan(Ys).mean(axis=0)
    src_trainable = src_nan_frac < Y_NAN_MAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        src_fill_mean = np.nanmean(Ys, axis=0)
    Ys_fit = np.where(np.isnan(Ys),
                      np.where(src_trainable, src_fill_mean, 0.0), Ys)
    log["source_pipeline"] = {
        "fill_cols_filled": src_fill_cols,
        "y_genes_source_trainable": int(src_trainable.sum()),
        "y_genes_dropped_nan_ge_10pct": int((~src_trainable).sum()),
        "pca_explained_var_sum": float(
            pca_src.explained_variance_ratio_.sum()),
    }
    zs_ridge = Ridge(alpha=RIDGE_ALPHA).fit(
        Xs_std, Ys_fit[:, src_trainable])

    # source-internal validation (ZS only, 80/20, frozen chain)
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(src_ids))
    n_val = int(round(SOURCE_VAL_FRAC * len(src_ids)))
    val_pos, tr_pos = perm[:n_val], perm[n_val:]
    ridge_val = Ridge(alpha=RIDGE_ALPHA).fit(
        Xs_std[tr_pos], Ys_fit[tr_pos][:, src_trainable])
    pred_val = ridge_val.predict(Xs_std[val_pos])
    rho_val = _spearman_gene_vec(
        pred_val.T, Ys[val_pos][:, src_trainable].T, MIN_TEST_ROWS)
    fin_val = np.isfinite(rho_val)
    src_val_macro = (float(np.mean(rho_val[fin_val]))
                     if fin_val.any() else float("nan"))
    log["source_internal_validation"] = {
        "arm": "ZS", "n_train": int(len(tr_pos)), "n_val": int(len(val_pos)),
        "macro_spearman": src_val_macro, "n_genes": int(fin_val.sum())}
    print(f"[exp030] source-internal validation ZS macro rho="
          f"{src_val_macro:.4f}", flush=True)
    del Xs, Ys, Ys_fit, pred_val, rho_val, ridge_val, Xs_f, Xs_pca

    # ---------- OrthoAE pretraining (DA arm) ----------
    print("[exp030] orthogonal AE pretraining (2 domains)", flush=True)
    bio_raw = pd.read_csv(BIO_RAW_CSV, index_col=0)
    bio_raw.columns = [_symbol(str(c)) for c in bio_raw.columns]
    bio_raw = bio_raw.loc[:, ~bio_raw.columns.duplicated()]
    pos162 = bio_raw.index.get_indexer(sample_ids)
    assert (pos162 >= 0).all(), "biobank CSV must contain all 162 samples"
    log["biobank_raw_corpus"] = {"samples": int(bio_raw.shape[0]),
                                 "genes": int(bio_raw.shape[1])}
    X255 = bio_raw.reindex(columns=expr_genes).to_numpy(float)
    del bio_raw
    X255_f, tgt_fill_cols_255 = _fill_nan(X255)
    z_bio255 = _zscore(X255_f)
    del X255
    domains_ae = [z_src, z_bio255]
    ae_metrics: dict[int, dict] = {}
    da_model = None
    for seed in AE_SEEDS:
        print(f"  ae seed {seed}", flush=True)
        model, metrics = train_ae(domains_ae, seed)
        ae_metrics[seed] = metrics
        if seed == DA_MAIN_SEED:
            da_model = model
    log["ortho_ae"] = {
        "corpus": {"depmap": list(domains_ae[0].shape),
                   "biobank": list(domains_ae[1].shape)},
        "target_fill_cols_filled_255": tgt_fill_cols_255,
        "seeds": {str(s): m for s, m in ae_metrics.items()},
        "da_main_seed": DA_MAIN_SEED,
    }
    da_available = True
    Z_da = np.full((len(sample_ids), SHARED_DIM), np.nan)
    try:
        with torch.no_grad():
            codes_255 = da_model.enc_shared(
                torch.tensor(z_bio255, dtype=torch.float32, device=DEVICE)
            ).cpu().numpy()
        Z_da = codes_255[pos162]
        del codes_255
    except Exception as exc:  # pragma: no cover - preregistered failure path
        da_available = False
        log["ortho_ae"]["da_failure"] = repr(exc)
        print(f"[exp030] DA arm unavailable: {exc!r}", flush=True)
    del z_bio255, domains_ae
    log["ortho_ae"]["da_available"] = da_available

    # ---------- neighbor diagnostics ----------
    print("[exp030] neighbor diagnostics", flush=True)
    Xt_f_all, _ = _fill_nan(Xt)
    z_tgt_all = _zscore(Xt_f_all)
    del Xt_f_all
    d_ts = _corr_dist(z_src, z_tgt_all)               # 162 x n_src
    min_dist = d_ts.min(axis=1)
    del d_ts
    d_ss = _corr_dist(z_src, z_src)                   # n_src x n_src
    np.fill_diagonal(d_ss, np.inf)
    thresh = float(np.percentile(d_ss.min(axis=1), 95))
    del d_ss
    has_neighbor = min_dist <= thresh
    is_colorectal = (samples["primary_tumour_type"]
                     == CANCER_MAIN).to_numpy()
    log["neighbor"] = {
        "threshold_p95": thresh,
        "min_distance_median": float(np.median(min_dist)),
        "n_with_neighbor": int(has_neighbor.sum()),
        "n_without_neighbor": int((~has_neighbor).sum()),
    }

    # ---------- folds (individual-grouped, cancer-stratified) ----------
    ind_ids = samples["individual_ID"].unique()        # native dtype (int64)
    ind_group = tissue.reindex(ind_ids)
    assert ind_group.notna().all(), "individual tissue group lookup failed"
    ind_group = ind_group.map(
        lambda t: t if t in (CANCER_MAIN, CANCER_SECOND) else "Other")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=SPLIT_SEED)
    fold_pairs = []
    for _, te_i in skf.split(ind_ids, ind_group.to_numpy()):
        te_inds = set(ind_ids[te_i].tolist())
        te_mask = samples["individual_ID"].isin(te_inds).to_numpy()
        te_pos = np.where(te_mask)[0]
        tr_pos = np.where(~te_mask)[0]
        fold_pairs.append((tr_pos, te_pos))
    log["folds"] = {"n_folds": N_FOLDS,
                    "test_sizes": [int(len(te)) for _, te in fold_pairs],
                    "train_sizes": [int(len(tr)) for tr, _ in fold_pairs]}
    if min(len(te) for _, te in fold_pairs) < MIN_TEST_ROWS:
        raise RuntimeError("preregistered failure: a test fold has fewer "
                           f"than {MIN_TEST_ROWS} organoids")

    # ---------- per-fold five arms ----------
    n_eval = len(evaluable)
    P = {arm: np.full((n_eval, len(sample_ids)), np.nan) for arm in ARMS}
    O = Yt.T.copy()                                     # n_eval x 162
    fold_gene_rho: dict[str, list[np.ndarray]] = {a: [] for a in ARMS}
    fold_macro: dict[str, list[float]] = {a: [] for a in ARMS}
    fold_gene_counts: dict[str, list[int]] = {a: [] for a in ARMS}

    pred_schema = pa.schema([
        ("fold", pa.int8()), ("arm", pa.string()), ("sample", pa.string()),
        ("gene", pa.string()), ("predicted", pa.float32()),
        ("observed", pa.float32())])
    pred_writer = pq.ParquetWriter(OUT_DIR / "predictions.parquet",
                                   pred_schema, compression="snappy",
                                   use_dictionary=["arm", "sample", "gene"])
    gene_np = np.array(evaluable, dtype=object)

    for fold_i, (tr_pos, te_pos) in enumerate(fold_pairs):
        t_fold = time.time()
        n_tr, n_te = len(tr_pos), len(te_pos)
        te_ids = [sample_ids[i] for i in te_pos]
        # target pipeline: fill (train-fold median) -> z (train fold) -> PCA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            tr_mat = Xt[tr_pos]
            med_tr = (np.nanmedian(tr_mat, axis=0)
                     if np.isnan(tr_mat).any()
                     else np.full(Xt.shape[1], np.nan))
        med_tr = np.where(np.isfinite(med_tr), med_tr, 0.0)
        Xt_tr = np.where(np.isnan(tr_mat), med_tr[None, :], tr_mat)
        Xt_te = np.where(np.isnan(Xt[te_pos]), med_tr[None, :], Xt[te_pos])
        mu_t = Xt_tr.mean(axis=0, keepdims=True)
        sd_t = Xt_tr.std(axis=0, keepdims=True)
        sd_t = np.where(sd_t < 1e-8, 1.0, sd_t)
        z_tr = (Xt_tr - mu_t) / sd_t
        z_te = (Xt_te - mu_t) / sd_t
        pca_t = PCA(n_components=PCA_COMPONENTS).fit(z_tr)
        Xp_tr = pca_t.transform(z_tr)
        Xp_te = pca_t.transform(z_te)
        sc_hd = StandardScaler().fit(Xp_tr)
        # ZS features: frozen source chain applied to filled target expr
        Xz_tr = sc_src.transform(pca_src.transform((Xt_tr - mu_s) / sd_s))
        Xz_te = sc_src.transform(pca_src.transform((Xt_te - mu_s) / sd_s))

        # Y handling (train fold)
        Y_tr = Yt[tr_pos]
        nan_frac = np.isnan(Y_tr).mean(axis=0)
        fold_trainable = nan_frac < Y_NAN_MAX
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fold_mean = np.nanmean(Y_tr, axis=0)
        fold_mean = np.where(fold_trainable, fold_mean, np.nan)
        Y_fit = np.where(np.isnan(Y_tr),
                         np.where(fold_trainable, fold_mean, 0.0), Y_tr)
        paired = fold_trainable & src_trainable

        preds: dict[str, np.ndarray] = {}
        # B0: train-fold gene mean (constant per gene)
        b0 = np.full((n_te, n_eval), np.nan)
        cols_ft = np.where(fold_trainable)[0]
        b0[:, cols_ft] = np.broadcast_to(fold_mean[cols_ft],
                                         (n_te, len(cols_ft)))
        preds["B0"] = b0
        # ZS: frozen source ridge applied to target
        zs_te = np.full((n_te, n_eval), np.nan)
        zs_te[:, src_trainable] = zs_ridge.predict(Xz_te)
        preds["ZS"] = zs_te
        # CAL: per-gene 2-parameter linear calibration of ZS (closed form)
        zs_tr = np.full((n_tr, n_eval), np.nan)
        zs_tr[:, src_trainable] = zs_ridge.predict(Xz_tr)
        cal = np.full((n_te, n_eval), np.nan)
        calc_ok = fold_trainable & src_trainable
        cols = np.where(calc_ok)[0]
        if len(cols):
            xo = zs_tr[:, cols]
            yo = Y_tr[:, cols]
            m = ~np.isnan(yo)
            xv = np.where(m, xo, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                xbar = np.nanmean(xv, axis=0)
                ybar = np.nanmean(np.where(m, yo, np.nan), axis=0)
                exy = np.nanmean(np.where(m, xo * yo, np.nan), axis=0)
                vx = np.nanvar(xv, axis=0)
            a = np.where(vx > 1e-12,
                         (exy - xbar * ybar) / np.where(vx > 1e-12, vx, 1.0),
                         0.0)
            b = ybar - a * xbar
            cal[:, cols] = zs_te[:, cols] * a + b
        preds["CAL"] = cal
        # DA: per-fold multi-output ridge on frozen shared codes
        if da_available:
            sc_da = StandardScaler().fit(Z_da[tr_pos])
            ridge_da = Ridge(alpha=RIDGE_ALPHA).fit(
                sc_da.transform(Z_da[tr_pos]), Y_fit[:, fold_trainable])
            da_te = np.full((n_te, n_eval), np.nan)
            da_te[:, fold_trainable] = ridge_da.predict(
                sc_da.transform(Z_da[te_pos]))
            preds["DA"] = da_te
        else:
            preds["DA"] = np.full((n_te, n_eval), np.nan)
        # HD: per-fold multi-output ridge on target PCA100
        ridge_hd = Ridge(alpha=RIDGE_ALPHA).fit(
            sc_hd.transform(Xp_tr), Y_fit[:, fold_trainable])
        hd_te = np.full((n_te, n_eval), np.nan)
        hd_te[:, fold_trainable] = ridge_hd.predict(sc_hd.transform(Xp_te))
        preds["HD"] = hd_te

        obs_te = Yt[te_pos]
        for arm in ARMS:
            P[arm][:, te_pos] = preds[arm].T
            rho = _spearman_gene_vec(preds[arm].T, obs_te.T, MIN_TEST_ROWS)
            fold_gene_rho[arm].append(rho)
            sel = paired & np.isfinite(rho)
            fold_macro[arm].append(float(np.mean(rho[sel]))
                                   if sel.any() else float("nan"))
            fold_gene_counts[arm].append(int(sel.sum()))
        for arm in ARMS:
            pm = preds[arm]
            n_rows = pm.size
            tbl = pa.table({
                "fold": pa.array(np.full(n_rows, fold_i + 1, np.int8())),
                "arm": pa.array(np.full(n_rows, arm, dtype=object)),
                "sample": pa.array(np.repeat(
                    np.array(te_ids, dtype=object), n_eval)),
                "gene": pa.array(np.tile(gene_np, n_te)),
                "predicted": pa.array(pm.T.ravel().astype(np.float32)),
                "observed": pa.array(obs_te.T.ravel().astype(np.float32)),
            }, schema=pred_schema)
            pred_writer.write_table(tbl)
        print(f"[exp030] fold {fold_i + 1}/{N_FOLDS} done "
              f"({time.time() - t_fold:.1f}s) n_test={n_te} "
              f"paired_genes={int(paired.sum())}", flush=True)
    pred_writer.close()

    # ---------- gene transferability map ----------
    map_rows: dict[str, np.ndarray] = {
        "gene": gene_np.astype(str)}
    for arm in ARMS:
        mat = np.vstack(fold_gene_rho[arm])            # n_folds x n_eval
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            map_rows[arm] = np.nanmean(mat, axis=0)
    pd.DataFrame(map_rows).to_parquet(
        OUT_DIR / "gene_transferability_map.parquet", index=False)

    # ---------- secondary: essential AUROC ----------
    ess_gene_mask = np.isin(np.array(evaluable), ess_genes)
    log["essentials_prior"]["n_in_evaluable"] = int(ess_gene_mask.sum())
    auroc: dict[str, float] = {}
    for arm in ARMS:
        vals = []
        for s in range(len(sample_ids)):
            v = ~np.isnan(P[arm][:, s])
            if ess_gene_mask[v].any() and (~ess_gene_mask[v]).any():
                vals.append(roc_auc_score(ess_gene_mask[v], -P[arm][v, s]))
        auroc[arm] = float(np.mean(vals)) if vals else float("nan")
    log["auroc_essential"] = auroc

    # ---------- stratified point estimates ----------
    def _stratum_stats(colmask: np.ndarray) -> dict:
        out: dict[str, dict[str, object]] = {}
        o_s = O[:, colmask]
        rho_b0 = _spearman_gene_vec(P["B0"][:, colmask], o_s, MIN_TEST_ROWS)
        fin_b0 = np.isfinite(rho_b0)
        pooled_b0 = _pooled_std_spearman(P["B0"][:, colmask], o_s)
        for arm in ARMS:
            rho = _spearman_gene_vec(P[arm][:, colmask], o_s, MIN_TEST_ROWS)
            fin = np.isfinite(rho)
            both = fin & fin_b0
            out[arm] = {
                "macro_spearman": (float(np.mean(rho[fin]))
                                   if fin.any() else None),
                "n_genes": int(fin.sum()),
                "pooled_std_spearman": _pooled_std_spearman(
                    P[arm][:, colmask], o_s),
            }
            if both.any():
                out[arm]["delta_vs_b0_macro"] = float(
                    np.mean(rho[both]) - np.mean(rho_b0[both]))
                out[arm]["delta_vs_b0_pooled"] = float(
                    out[arm]["pooled_std_spearman"] - pooled_b0)
        return out

    stratified: dict[str, object] = {
        "neighbor": {
            "threshold_p95": thresh,
            "min_distance_median": float(np.median(min_dist)),
            "n_with_neighbor": int(has_neighbor.sum()),
            "n_without_neighbor": int((~has_neighbor).sum()),
            "with_neighbor": _stratum_stats(has_neighbor),
            "without_neighbor": _stratum_stats(~has_neighbor)},
        "cancer": {
            "n_colorectal": int(is_colorectal.sum()),
            "n_non_colorectal": int((~is_colorectal).sum()),
            "colorectal": _stratum_stats(is_colorectal),
            "non_colorectal": _stratum_stats(~is_colorectal)},
        "gene_type": {
            "n_essential_genes": int(ess_gene_mask.sum()),
            "n_variable_genes": int((~ess_gene_mask).sum()),
            "essential": {arm: (float(np.nanmean(map_rows[arm][ess_gene_mask]))
                                if ess_gene_mask.any() else None)
                          for arm in ARMS},
            "variable": {arm: (float(np.nanmean(map_rows[arm][~ess_gene_mask]))
                               if (~ess_gene_mask).any() else None)
                         for arm in ARMS}},
    }

    # ---------- bootstrap (individual level, pooled 5 folds) ----------
    print(f"[exp030] bootstrap {BOOT_N} (individual level)", flush=True)
    P32 = {arm: P[arm].astype(np.float32) for arm in ARMS}
    O32 = O.astype(np.float32)
    near_mask = has_neighbor
    ind_pos = [np.where(samples["individual_ID"].to_numpy() == i)[0]
               for i in ind_ids]
    rng = np.random.default_rng(BOOT_SEED)
    boot_delta = {arm: [] for arm in DELTA_ARMS}
    boot_gain = {arm: [] for arm in TRANSFER_ARMS}
    t_boot = time.time()
    for it in range(BOOT_N):
        take = rng.integers(0, len(ind_pos), len(ind_pos))
        idx = np.concatenate([ind_pos[i] for i in take])
        o_b = O32[:, idx]
        near_cols = near_mask[idx]
        mask_b0, rp_b0, ro_b0 = _rank_pair(P32["B0"][:, idx], o_b)
        rho_b0 = _macro_from_ranks(mask_b0, rp_b0, ro_b0, MIN_TEST_ROWS)
        fin_b0 = np.isfinite(rho_b0)
        pl_b0_near = _pooled_from_ranks(
            *_rank_pair(P32["B0"][:, idx][:, near_cols], o_b[:, near_cols]))
        pl_b0_far = _pooled_from_ranks(
            *_rank_pair(P32["B0"][:, idx][:, ~near_cols],
                        o_b[:, ~near_cols]))
        for arm in DELTA_ARMS:
            mask_a, rp_a, ro_a = _rank_pair(P32[arm][:, idx], o_b)
            rho_a = _macro_from_ranks(mask_a, rp_a, ro_a, MIN_TEST_ROWS)
            fin_a = np.isfinite(rho_a)
            both = fin_a & fin_b0
            boot_delta[arm].append(
                float(np.mean(rho_a[both]) - np.mean(rho_b0[both]))
                if both.any() else float("nan"))
        for arm in TRANSFER_ARMS:
            pl_near = _pooled_from_ranks(
                *_rank_pair(P32[arm][:, idx][:, near_cols],
                            o_b[:, near_cols]))
            pl_far = _pooled_from_ranks(
                *_rank_pair(P32[arm][:, idx][:, ~near_cols],
                            o_b[:, ~near_cols]))
            boot_gain[arm].append((pl_near - pl_b0_near)
                                  - (pl_far - pl_b0_far))
        if (it + 1) % 250 == 0:
            print(f"  bootstrap {it + 1}/{BOOT_N} "
                  f"({time.time() - t_boot:.0f}s)", flush=True)
    del P32, O32

    def _ci(vals: list) -> list:
        arr = np.asarray([v for v in vals if np.isfinite(v)], float)
        if not len(arr):
            return [float("nan"), float("nan")]
        return [float(x) for x in np.percentile(arr, [2.5, 97.5])]

    # ---------- arm summary ----------
    five_fold = {arm: float(np.nanmean(fold_macro[arm])) for arm in ARMS}
    arm_summary: dict[str, object] = {
        "primary_endpoint": {
            "metric": "macro per-gene Spearman on test folds "
                      "(>= 30 rows per gene, all-arm evaluable set)",
            "per_fold_macro": {a: fold_macro[a] for a in ARMS},
            "five_fold_mean": five_fold,
            "paired_genes_per_fold": {a: fold_gene_counts[a] for a in ARMS},
            "delta_vs_b0_bootstrap": {}},
        "secondary_auroc_essential": auroc,
        "source_internal_validation": log["source_internal_validation"],
        "n_evaluable_genes": n_eval,
        "da_arm_available": da_available,
    }
    for arm in DELTA_ARMS:
        ci = _ci(boot_delta[arm])
        arm_summary["primary_endpoint"]["delta_vs_b0_bootstrap"][arm] = {
            "point_five_fold_mean": five_fold[arm] - five_fold["B0"],
            "pooled_point": float(np.nanmean(boot_delta[arm])),
            "ci95": ci,
            "ci_low_positive": bool(ci[0] > 0),
            "boot_frac_positive": float(np.mean(
                np.asarray(boot_delta[arm], float) > 0)),
        }
    ci_low = {arm: arm_summary["primary_endpoint"]
              ["delta_vs_b0_bootstrap"][arm]["ci95"][0]
              for arm in TRANSFER_ARMS}
    pooled_pt = {arm: arm_summary["primary_endpoint"]
                 ["delta_vs_b0_bootstrap"][arm]["pooled_point"]
                 for arm in TRANSFER_ARMS}
    if any(v > 0 for v in ci_low.values()):
        verdict = ("RULE1_TRANSFER_SUPPORTED_ZS_DIRECT" if ci_low["ZS"] > 0
                   else "RULE2_LOW_DOF_ADAPTATION_REPAIRS_DOMAIN_SHIFT")
    else:
        verdict = "RULE3_TRANSFER_METHODOLOGY_NOT_SUPPORTED"
    best = max(TRANSFER_ARMS,
               key=lambda a: pooled_pt[a] if np.isfinite(pooled_pt[a])
               else -np.inf)
    arm_summary["primary_endpoint"]["verdict"] = {
        "rule": verdict,
        "per_arm_ci_low": ci_low,
        "best_transfer_arm": best,
    }
    stratified["neighbor"]["transfer_gain_with_minus_without"] = {
        arm: {"point": float(np.nanmean(boot_gain[arm])),
              "ci95": _ci(boot_gain[arm])} for arm in TRANSFER_ARMS}
    arm_summary["stratified_summary"] = {
        "neighbor_macro": {
            "with": {a: stratified["neighbor"]["with_neighbor"][a]
                     ["macro_spearman"] for a in ARMS},
            "without": {a: stratified["neighbor"]["without_neighbor"][a]
                        ["macro_spearman"] for a in ARMS}},
        "neighbor_gain_ci_low": {
            a: stratified["neighbor"]["transfer_gain_with_minus_without"]
            [a]["ci95"][0] for a in TRANSFER_ARMS},
        "colorectal_macro": {
            a: stratified["cancer"]["colorectal"][a]["macro_spearman"]
            for a in ARMS},
        "essential_vs_variable_macro": {
            g: stratified["gene_type"][g] for g in
            ("essential", "variable")},
    }
    (OUT_DIR / "arm_summary.json").write_text(
        json.dumps(arm_summary, indent=2, ensure_ascii=False, default=str))
    (OUT_DIR / "stratified_report.json").write_text(
        json.dumps(stratified, indent=2, ensure_ascii=False, default=str))

    log["arm_summary_ref"] = "arm_summary.json"
    log["stratified_ref"] = "stratified_report.json"
    log["elapsed_seconds"] = round(time.time() - t0, 1)
    (OUT_DIR / "audit_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False, default=str))
    print(f"[exp030] done in {log['elapsed_seconds']}s verdict={verdict}",
          flush=True)


if __name__ == "__main__":
    main()
