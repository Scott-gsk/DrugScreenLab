#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-029 cell-line -> organoid interpretable transfer audit (T1-T4).

Pre-registered contract (experiments/records/EXP-029.md, pinned BEFORE run):
- Read-only audit: no new drug-sensitivity training. Zero-shot source->Kryzei
  inference (EXP-015 source model, pure predict) feeds T3/T4.
- T1 pathway map: Hallmark-50 single-sample mean-z activities; per-drug Ridge
  (alpha=1.0, standardized) on GDSC (cell lines) vs Kryzei (samples, primary;
  patient-level sensitivity); per-pathway sign agreement across 14 known drugs
  (binomial two-sided, BH-FDR 0.05).
- T2 shared/private decomposition: orthogonal-constrained AE on the 14,900-gene
  triple intersection (GDSC 963 / Kryzei ~91 / biobank 255, within-domain z);
  shared enc 14900-512-128-64, private enc (per domain) ...-32, decoders
  (64+32)-128-512-14900; Adam lr 1e-3 wd 1e-5 batch 128 epochs 30 lambda_orth 1
  seeds 41/42/43. Metrics: R2_full / R2_shared / R2_private per domain, shared
  capture = R2_shared/R2_full (primary, Kryzei), 3-class domain classifier on
  shared vs private codes (5-fold logistic), per-Kryzei-sample shared capture
  vs zero-shot performance Spearman.
- T3 neighbor diagnostics: distance = 1 - Pearson on the same z space; neighbor
  pool = 193 source validation cell lines with per-line Spearman error (>=50
  drugs); k=10; patient-level Spearman(PDO zero-shot performance, mean neighbor
  error) primary (two-sided p<0.05 AND positive estimate = significant);
  no-neighbor flag = min distance to all 963 GDSC lines > P95 of within-GDSC
  nearest-neighbor distances (Kryzei + biobank 255 counted).
- T4 drug boundary: per known drug source across-cell-line SD vs target
  across-patient SD (Spearman across 14 drugs); transferable drug = zero-shot
  per-drug cross-patient Spearman > 0 with patient bootstrap (2000, seed
  20260918) 95% CI excluding 0; structural/nonstructural counted separately.
- Verdict rules pinned in EXP-029.md.

Outputs: artifacts/exp029_transfer_audit/* (see record).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import binomtest, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from drug_screen.transfer.benchmark import _table
from drug_screen.transfer.chemistry import morgan_fingerprint
from drug_screen.transfer.cohorts import load_registered_cohort, sha256
from drug_screen.transfer.models import DualEncoderPairModel
from drug_screen.transfer.preprocessing import ExpressionPreprocessor

ROOT = Path("/mnt/d/Code/DrugScreenLab")
OUT_DIR = ROOT / "artifacts/exp029_transfer_audit"
SOURCE_DIR = ROOT / "data/artifacts/EXP-015/source_seed20260912"
SRC_REGISTRY = ROOT / "data/registry/exp015_source.json"
BIOBANK_CSV = ROOT / ("data/raw/transfer_public_assets/organoid_biobank/"
                      "data_RNAseq_ALL-organoids.csv")
BIOBANK_SHA = "92859e18417c3737956236e6990f8001a0d2cc352890b9355b7ed9603b6a6b1e"
HALLMARK_GMT = ROOT / "data/raw/exp013/msigdb/h.all.v2023.2.Hs.symbols.gmt"
STRUCTURAL_JSON = ROOT / "data/processed/identity/kryzei_structural_drugs_FROZEN.json"
ROUTEA_PRED = ROOT / "artifacts/exp028_fusion/routeA_frozen_predictions.parquet"

AE_SEEDS = [41, 42, 43]
AE_EPOCHS, AE_BATCH, AE_LR, AE_WD, LAMBDA_ORTH = 30, 128, 1e-3, 1e-5, 1.0
SHARED_DIM, PRIVATE_DIM, HIDDEN = 64, 32, 512
RIDGE_ALPHA = 1.0
K_NEIGHBORS = 10
MIN_DRUGS_FOR_LINE_ERROR = 50
BOOT_N, BOOT_SEED = 2000, 20260918
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _zscore(mat: np.ndarray) -> np.ndarray:
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (mat - mu) / sd


def _fill_nan(mat: np.ndarray) -> tuple[np.ndarray, int]:
    """域内列中位数填充（GDSC 未测基因列：13 个交集基因在全部细胞系缺失）。

    全 NaN 列填 0（z 分数后即无信息）。填充列数登记进 audit_log。
    """
    if not np.isnan(mat).any():
        return mat, 0
    with np.errstate(all="ignore"):
        med = np.nanmedian(mat, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    filled = np.where(np.isnan(mat), med[None, :], mat)
    return filled, int(np.isnan(mat).any(axis=0).sum())


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    rho = spearmanr(a, b).statistic
    return float(rho) if np.isfinite(rho) else float("nan")


def _r2(x: np.ndarray, recon: np.ndarray) -> float:
    sse = float(((x - recon) ** 2).sum())
    sst = float(((x - x.mean(axis=0, keepdims=True)) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


class _MLP(nn.Module):
    def __init__(self, dims: list[int], *, relu_last: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            last = i == len(dims) - 2
            if (not last) or relu_last:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OrthoAE(nn.Module):
    def __init__(self, n_genes: int):
        super().__init__()
        self.enc_shared = _MLP([n_genes, HIDDEN, 128, SHARED_DIM])
        self.enc_private = nn.ModuleList(
            [_MLP([n_genes, HIDDEN, 128, PRIVATE_DIM]) for _ in range(3)])
        self.decoders = nn.ModuleList(
            [_MLP([SHARED_DIM + PRIVATE_DIM, 128, HIDDEN, n_genes])
             for _ in range(3)])

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


def train_ae(domains: list[np.ndarray], seed: int) -> dict:
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
    for epoch in range(1, AE_EPOCHS + 1):
        model.train()
        perm = np.random.permutation(n_total)
        tot, batches = 0.0, 0
        for start in range(0, n_total, AE_BATCH):
            sel = pooled[perm[start:start + AE_BATCH]]
            loss = torch.zeros((), device=DEVICE)
            for g in range(3):
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
            tot += float(loss.item()); batches += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == AE_EPOCHS:
            print(f"    ae epoch {epoch}/{AE_EPOCHS} loss={tot / max(batches, 1):.5f}",
                  flush=True)
    model.eval()
    with torch.no_grad():
        metrics: dict[str, object] = {}
        codes_shared, codes_private, labels = [], [], []
        per_sample = {}
        for g, name in enumerate(("gdsc", "kryzei", "biobank")):
            x = tensors[g]
            full, zs, zp = model(x, g)
            shared_only, _, _ = model(x, g, zero_private=True)
            private_only, _, _ = model(x, g, zero_shared=True)
            metrics[f"{name}_r2_full"] = _r2(x.cpu().numpy(), full.cpu().numpy())
            metrics[f"{name}_r2_shared"] = _r2(x.cpu().numpy(),
                                               shared_only.cpu().numpy())
            metrics[f"{name}_r2_private"] = _r2(x.cpu().numpy(),
                                               private_only.cpu().numpy())
            per_sample[name] = (
                1.0 - ((x - shared_only) ** 2).sum(dim=1).cpu().numpy()
                / (((x - x.mean(dim=0, keepdims=True)) ** 2)
                   .sum(dim=1).clamp(min=1e-12)).cpu().numpy())
            codes_shared.append(zs.cpu().numpy())
            codes_private.append(zp.cpu().numpy())
            labels.append(np.full(len(x), g))
        for name in ("gdsc", "kryzei", "biobank"):
            metrics[f"{name}_shared_capture"] = (
                metrics[f"{name}_r2_shared"] / metrics[f"{name}_r2_full"])
        Xs = np.concatenate(codes_shared)
        Xp = np.concatenate(codes_private)
        y = np.concatenate(labels)
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        acc_s = cross_val_score(clf, Xs, y, cv=cv, scoring="accuracy").mean()
        acc_p = cross_val_score(clf, Xp, y, cv=cv, scoring="accuracy").mean()
        metrics["domain_acc_shared"] = float(acc_s)
        metrics["domain_acc_private"] = float(acc_p)
    return {"metrics": metrics, "per_sample_shared_capture": per_sample}


def main() -> None:
    t0 = time.time()
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise ValueError("output must be new or empty")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log: dict[str, object] = {"device": str(DEVICE)}
    print(f"[exp029] device={DEVICE}", flush=True)

    # ---------- data loading ----------
    src_reg = json.loads(SRC_REGISTRY.read_text())
    entry = src_reg["cohorts"][0]
    assert entry["cohort_id"] == "gdsc2_v2" and entry["higher_is_sensitive"] is True
    assets = {}
    for key, spec in entry["assets"].items():
        path = ROOT / "data" / spec["path"]
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"gdsc asset sha mismatch: {key}")
        assets[key] = path
    g_samples = pd.read_parquet(assets["samples"])
    g_pairs = pd.read_parquet(assets["pairs"])
    with np.load(assets["expression"], allow_pickle=True) as z:
        g_expr = np.asarray(z["expression"], float)
        g_genes = [str(x) for x in z["ordered_genes"]]
    g_sid = [str(s) for s in g_samples["sample_id"]]
    pos = {s: i for i, s in enumerate(g_sid)}
    g_pairs = g_pairs.copy()
    g_pairs["sample_id"] = g_pairs["sample_id"].astype(str)
    assert not g_pairs.duplicated(["sample_id", "drug_id"]).any()
    assert set(g_pairs["sample_id"]) <= set(pos)
    g_pairs["expression_index"] = g_pairs["sample_id"].map(pos)
    log["gdsc"] = {"cell_lines": len(g_sid), "genes": len(g_genes),
                   "pairs": len(g_pairs),
                   "drugs": int(g_pairs["drug_id"].nunique())}

    bundle = load_registered_cohort(
        "kryeziu_mcrc_v3", root=ROOT,
        registry=ROOT / "data/registry/transfer_cohorts.json")
    table = _table(bundle)
    k_expr = np.asarray(bundle.expression, float)
    k_genes = [str(x) for x in bundle.gene_ids]
    k_samples = [str(s) for s in bundle.sample_ids]
    assert not table.duplicated(["patient_id", "model_id", "drug_id"]).any()
    log["kryzei"] = {"samples_in_indexed": len(k_samples),
                     "table_rows": len(table),
                     "patients": int(table.patient_id.nunique()),
                     "genes": len(k_genes)}

    if sha256(BIOBANK_CSV) != BIOBANK_SHA:
        raise ValueError("biobank raw csv sha mismatch")
    bio = pd.read_csv(BIOBANK_CSV, index_col=0)
    bio.columns = [c[:-4] for c in bio.columns]  # strip _exp
    b_genes = [str(c) for c in bio.columns]
    b_expr = bio.to_numpy(float)
    log["biobank"] = {"samples": b_expr.shape[0], "genes": b_expr.shape[1]}

    frozen = json.loads((SOURCE_DIR / "frozen_run.json").read_text())
    source_drugs = set(frozen["source_drug_ids"])
    known = sorted(set(table.drug_id.astype(str)) & source_drugs)
    if len(known) != 14:
        raise AssertionError(f"expected 14 known drugs, got {len(known)}")
    structural = json.loads(STRUCTURAL_JSON.read_text())
    assert structural["status"] == "FROZEN"
    drug_kind = {d: ("structural" if d in set(structural["pan_effective"])
                    | set(structural["pan_resistant"]) else "nonstructural")
                 for d in known}

    inter = sorted(set(g_genes) & set(k_genes) & set(b_genes))
    if len(inter) != 14900:
        raise AssertionError(f"expected 14900 intersection genes, got {len(inter)}")
    g_idx = {g: i for i, g in enumerate(g_genes)}
    k_idx = {g: i for i, g in enumerate(k_genes)}
    b_idx = {g: i for i, g in enumerate(b_genes)}
    gi = np.array([g_idx[g] for g in inter])
    ki = np.array([k_idx[g] for g in inter])
    bi = np.array([b_idx[g] for g in inter])
    log["intersection_genes"] = len(inter)

    # ---------- zero-shot source -> Kryzei ----------
    print("[exp029] zero-shot inference", flush=True)
    prep = ExpressionPreprocessor.load(SOURCE_DIR / "expression_transform.npz")
    feats, report = prep.transform_with_report(k_expr, np.array(k_genes, object))
    log["zeroshot_alignment"] = {
        "n_input_genes": report.n_input_genes,
        "n_aligned_genes": report.n_aligned_genes,
        "n_missing_frozen_genes": report.n_missing_frozen_genes}
    model = DualEncoderPairModel.load(SOURCE_DIR / "source_model.pt")
    drugs_frame = bundle.drugs
    field = next(k for k in ("canonical_smiles", "canonical_isomeric_smiles",
                             "smiles") if k in drugs_frame.columns)
    lookup = dict(zip(drugs_frame.drug_id.astype(str), drugs_frame[field]))
    fps = {str(d): morgan_fingerprint(str(lookup[str(d)]), radius=2, n_bits=2048)
           .astype(float) for d in table.drug_id.unique()}
    rows = table
    pred = model.predict(
        feats[rows.expression_index.to_numpy(int)],
        np.vstack([fps[str(d)] for d in rows.drug_id]))
    zs = pd.DataFrame({
        "patient_id": rows.patient_id.astype(str).to_numpy(),
        "model_id": rows.model_id.astype(str).to_numpy(),
        "drug_id": rows.drug_id.astype(str).to_numpy(),
        "source_drug_status": np.where(
            rows.drug_id.astype(str).isin(source_drugs), "known", "unknown"),
        "observed": rows.response.to_numpy(float),
        "predicted": pred})
    zs.to_parquet(OUT_DIR / "zeroshot_kryzei_predictions.parquet", index=False)
    log["zeroshot_rows"] = len(zs)

    # per-sample / per-patient zero-shot performance (known drugs)
    known_rows = zs[zs.source_drug_status == "known"]

    def _spearman_group(g: pd.DataFrame) -> float:
        if g["drug_id"].nunique() < 5:
            return float("nan")
        return _spearman(g["observed"].to_numpy(float),
                         g["predicted"].to_numpy(float))

    perf = {mid: _spearman_group(gdf)
            for mid, gdf in known_rows.groupby("model_id", sort=True)}
    sample_patient = (table[["model_id", "patient_id"]]
                      .drop_duplicates().set_index("model_id")["patient_id"])
    perf_sample = pd.DataFrame({"zeroshot_spearman": pd.Series(perf)})
    perf_sample.index.name = "model_id"
    perf_sample = perf_sample.join(sample_patient)

    # ---------- T1 pathway map ----------
    print("[exp029] T1 pathway map", flush=True)
    pathways: dict[str, list[str]] = {}
    with open(HALLMARK_GMT) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            pathways[parts[0]] = [p for p in parts[2:] if p]

    def _activities(expr: np.ndarray, genes: list[str]) -> np.ndarray:
        zmat = _zscore(expr)
        pos = {g: i for i, g in enumerate(genes)}
        cols = []
        for pw, members in pathways.items():
            hit = [pos[m] for m in members if m in pos]
            if len(hit) < 10:
                raise AssertionError(f"pathway too small: {pw}")
            cols.append(zmat[:, hit].mean(axis=1))
        return np.column_stack(cols)

    g_act = _activities(g_expr, g_genes)
    k_act = _activities(k_expr, k_genes)
    pw_names = sorted(pathways)

    # response matrices: rows aligned to activity rows
    g_resp = g_pairs[["expression_index", "drug_id", "response"]]
    k_resp = table[["expression_index", "drug_id", "response"]]

    def _ridge_coefs(act: np.ndarray, resp: pd.DataFrame,
                     drug: str) -> np.ndarray | None:
        sub = resp[resp["drug_id"] == drug]
        if len(sub) < 20:
            return None
        X = act[sub["expression_index"].to_numpy(int)]
        y = sub["response"].to_numpy(float)
        sc = StandardScaler().fit(X)
        ridge = Ridge(alpha=RIDGE_ALPHA).fit(sc.transform(X), y)
        return ridge.coef_

    t1_drug_rows = []
    coefs_g, coefs_k = {}, {}
    for drug in known:
        cg = _ridge_coefs(g_act, g_resp, drug)
        ck = _ridge_coefs(k_act, k_resp, drug)
        if cg is None or ck is None:
            raise AssertionError(f"insufficient rows for drug {drug}")
        coefs_g[drug], coefs_k[drug] = cg, ck
        cos = float(np.dot(cg, ck) / (np.linalg.norm(cg) * np.linalg.norm(ck)))
        t1_drug_rows.append({"drug_id": drug, "kind": drug_kind[drug],
                             "cosine": cos})
    cg_mat = np.vstack([coefs_g[d] for d in known])
    ck_mat = np.vstack([coefs_k[d] for d in known])
    agree = (np.sign(cg_mat) == np.sign(ck_mat)).mean(axis=0)
    pvals = [float(binomtest(int(round(a * len(known))), len(known), 0.5)
             .pvalue) for a in agree]
    order = np.argsort(pvals)
    fdr = np.empty(len(pvals))
    m = len(pvals)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        idx = m - rank
        val = pvals[i] * m / idx
        prev = min(prev, val)
        fdr[i] = prev
    t1 = pd.DataFrame({"pathway": pw_names, "agreement": agree,
                       "p_value": pvals, "fdr": fdr,
                       "gdsc_mean_abs_coef": np.abs(cg_mat).mean(axis=0),
                       "kryzei_mean_abs_coef": np.abs(ck_mat).mean(axis=0)})
    t1.to_parquet(OUT_DIR / "t1_pathway_map.parquet", index=False)
    pd.DataFrame(t1_drug_rows).to_parquet(
        OUT_DIR / "t1_drug_cosine.parquet", index=False)
    log["t1"] = {"pathways_fdr_below_0.05": int((t1["fdr"] < 0.05).sum()),
                 "median_agreement": float(np.median(agree))}

    # ---------- T2 orthogonal AE ----------
    print("[exp029] T2 orthogonal autoencoder", flush=True)
    k_table_samples = sorted(table.expression_index.unique().tolist())
    g_int_fill, g_int_filled_cols = _fill_nan(g_expr[:, gi])
    log["t2_nan_fill"] = {"gdsc_intersection_cols_filled": g_int_filled_cols,
                          "rule": "within-domain column median; all-NaN -> 0"}
    domains = [ _zscore(g_int_fill),
                _zscore(k_expr[np.array(k_table_samples, int)][:, ki]),
                _zscore(b_expr[:, bi]) ]
    domain_names = ["gdsc", "kryzei", "biobank"]
    seed_metrics = []
    per_sample_capture = None
    for seed in AE_SEEDS:
        print(f"  seed {seed}", flush=True)
        out = train_ae(domains, seed)
        seed_metrics.append(out["metrics"])
        per_sample_capture = out["per_sample_shared_capture"]
    t2: dict[str, object] = {}
    for name in domain_names:
        for key in ("r2_full", "r2_shared", "r2_private", "shared_capture"):
            vals = [m[f"{name}_{key}"] for m in seed_metrics]
            t2[f"{name}_{key}"] = float(np.mean(vals))
            t2[f"{name}_{key}_sd"] = float(np.std(vals))
    for key in ("domain_acc_shared", "domain_acc_private"):
        vals = [m[key] for m in seed_metrics]
        t2[key] = float(np.mean(vals))
        t2[f"{key}_sd"] = float(np.std(vals))
    # per-sample capture vs zero-shot performance (Kryzei)
    k_capture = per_sample_capture["kryzei"]
    k_model_ids = [k_samples[i] for i in k_table_samples]
    cap_series = pd.Series(k_capture, index=k_model_ids, name="shared_capture")
    joined = perf_sample.join(cap_series, how="inner").dropna()
    t2["sample_corr_capture_vs_zeroshot"] = _spearman(
        joined["shared_capture"].to_numpy(float),
        joined["zeroshot_spearman"].to_numpy(float))
    pat = joined.groupby("patient_id")[["shared_capture",
                                        "zeroshot_spearman"]].mean()
    t2["patient_corr_capture_vs_zeroshot"] = _spearman(
        pat["shared_capture"].to_numpy(float),
        pat["zeroshot_spearman"].to_numpy(float))
    (OUT_DIR / "t2_ae_metrics.json").write_text(
        json.dumps(t2, indent=2, ensure_ascii=False))
    log["t2"] = {"kryzei_shared_capture": t2["kryzei_shared_capture"],
                 "domain_acc_shared": t2["domain_acc_shared"],
                 "domain_acc_private": t2["domain_acc_private"]}

    # ---------- T3 neighbor diagnostics ----------
    print("[exp029] T3 neighbor diagnostics", flush=True)
    sv = pd.read_parquet(SOURCE_DIR / "source_validation_predictions.parquet")
    line_err = {}
    for sid, gdf in sv.groupby("sample_id"):
        if len(gdf) >= MIN_DRUGS_FOR_LINE_ERROR:
            line_err[str(sid)] = _spearman(
                gdf["response"].to_numpy(float),
                gdf["predicted"].to_numpy(float))
    val_lines = sorted(s for s in line_err if s in pos)
    g_z = domains[0]
    val_mat = g_z[[pos[s] for s in val_lines]]
    err_vec = np.array([line_err[s] for s in val_lines])

    def _corr_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        num = b @ a.T
        den = (np.linalg.norm(b, axis=1)[:, None]
               * np.linalg.norm(a, axis=1)[None, :])
        return 1.0 - num / np.clip(den, 1e-12, None)

    k_mat = domains[1]
    b_mat = domains[2]
    # _corr_dist(a, b) returns (n_b, n_a): rows = second arg's samples.
    d_kv = _corr_dist(val_mat, k_mat)           # n_k x n_val
    d_kb = _corr_dist(val_mat, b_mat)           # n_b x n_val
    d_ka = _corr_dist(g_z, k_mat)               # n_k x 963
    d_ba = _corr_dist(g_z, b_mat)               # n_b x 963

    def _neighbor_stats(dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nn = np.argsort(dist, axis=1)[:, :K_NEIGHBORS]
        mean_err = err_vec[nn].mean(axis=1)
        mean_dist = np.take_along_axis(dist, nn, axis=1).mean(axis=1)
        return mean_err, mean_dist

    k_nerr, k_ndist = _neighbor_stats(d_kv)
    b_nerr, b_ndist = _neighbor_stats(d_kb)
    k_min_all = d_ka.min(axis=1)
    b_min_all = d_ba.min(axis=1)

    # within-GDSC nearest-neighbor distance P95 (all 963 lines)
    d_gg = _corr_dist(g_z, g_z)
    np.fill_diagonal(d_gg, np.inf)
    g_nn_min = d_gg.min(axis=1)
    thresh = float(np.percentile(g_nn_min, 95))

    t3_rows = []
    for j, mid in enumerate(k_model_ids):
        pat_id = str(sample_patient.get(mid, ""))
        t3_rows.append({
            "cohort": "kryzei", "model_id": mid, "patient_id": pat_id,
            "mean_neighbor_error": float(k_nerr[j]),
            "mean_neighbor_distance": float(k_ndist[j]),
            "min_distance_all_gdsc": float(k_min_all[j]),
            "no_neighbor": bool(k_min_all[j] > thresh),
            "zeroshot_spearman": float(perf_sample.loc[mid,
                                                      "zeroshot_spearman"]
                                       if mid in perf_sample.index else np.nan)})
    bio_ids = [str(s) for s in bio.index]
    for j, bid in enumerate(bio_ids):
        t3_rows.append({
            "cohort": "biobank", "model_id": bid, "patient_id": "",
            "mean_neighbor_error": float(b_nerr[j]),
            "mean_neighbor_distance": float(b_ndist[j]),
            "min_distance_all_gdsc": float(b_min_all[j]),
            "no_neighbor": bool(b_min_all[j] > thresh),
            "zeroshot_spearman": float("nan")})
    t3 = pd.DataFrame(t3_rows)
    t3.to_parquet(OUT_DIR / "t3_neighbors.parquet", index=False)

    kt3 = t3[t3.cohort == "kryzei"].dropna(subset=["zeroshot_spearman"])
    pat_level = kt3.groupby("patient_id")[["zeroshot_spearman",
                                           "mean_neighbor_error",
                                           "mean_neighbor_distance"]].mean()
    corr_pat_err = _spearman(pat_level["zeroshot_spearman"].to_numpy(float),
                             pat_level["mean_neighbor_error"].to_numpy(float))
    corr_pat_dist = _spearman(pat_level["zeroshot_spearman"].to_numpy(float),
                              pat_level["mean_neighbor_distance"].to_numpy(float))
    corr_smp_err = _spearman(kt3["zeroshot_spearman"].to_numpy(float),
                             kt3["mean_neighbor_error"].to_numpy(float))

    def _rho_p(a, b):
        r = spearmanr(a, b)
        return float(r.statistic), float(r.pvalue)

    rho_pe, p_pe = _rho_p(pat_level["zeroshot_spearman"].to_numpy(float),
                          pat_level["mean_neighbor_error"].to_numpy(float))
    rho_pd, p_pd = _rho_p(pat_level["zeroshot_spearman"].to_numpy(float),
                          pat_level["mean_neighbor_distance"].to_numpy(float))
    t3_json = {
        "patient_neighbor_error_corr": {"rho": corr_pat_err, "p": p_pe},
        "patient_neighbor_distance_corr": {"rho": corr_pat_dist, "p": p_pd},
        "sample_neighbor_error_corr": {"rho": corr_smp_err},
        "no_neighbor_threshold_p95": thresh,
        "kryzei_no_neighbor": int(t3[(t3.cohort == "kryzei")]
                                  ["no_neighbor"].sum()),
        "biobank_no_neighbor": int(t3[(t3.cohort == "biobank")]
                                   ["no_neighbor"].sum()),
        "n_validation_lines": len(val_lines),
    }
    (OUT_DIR / "t3_correlations.json").write_text(
        json.dumps(t3_json, indent=2, ensure_ascii=False))
    log["t3"] = t3_json

    # ---------- T4 drug boundary ----------
    print("[exp029] T4 drug boundary", flush=True)
    rng = np.random.default_rng(BOOT_SEED)
    t4_rows = []
    known_pair = known_rows.copy()
    pat_mean = (known_pair.groupby(["drug_id", "patient_id"])
                .agg(observed=("observed", "mean"),
                     predicted=("predicted", "mean")).reset_index())
    patients = sorted(pat_mean.patient_id.unique())
    for drug in known:
        sub = pat_mean[pat_mean.drug_id == drug]
        obs = sub["observed"].to_numpy(float)
        prd = sub["predicted"].to_numpy(float)
        point = _spearman(obs, prd)
        boots = []
        for _ in range(BOOT_N):
            take = rng.integers(0, len(obs), len(obs))
            boots.append(_spearman(obs[take], prd[take]))
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        gsub = g_resp[g_resp["drug_id"] == drug]["response"].to_numpy(float)
        source_sd = float(np.std(gsub, ddof=1))
        tsub = sub["observed"].to_numpy(float)
        target_sd = float(np.std(tsub, ddof=1))
        t4_rows.append({
            "drug_id": drug, "kind": drug_kind[drug],
            "source_sd": source_sd, "target_sd": target_sd,
            "zeroshot_patient_spearman": point,
            "ci_low": float(lo), "ci_high": float(hi),
            "transferable": bool(point > 0 and lo > 0)})
    t4 = pd.DataFrame(t4_rows)
    t4.to_parquet(OUT_DIR / "t4_drug_table.parquet", index=False)
    sd_corr = _spearman(t4["source_sd"].to_numpy(float),
                        t4["target_sd"].to_numpy(float))
    ns = t4[t4.kind == "nonstructural"]
    log["t4"] = {"source_target_sd_spearman": sd_corr,
                 "transferable_nonstructural": int(ns["transferable"].sum()),
                 "transferable_structural": int(
                     t4[t4.kind == "structural"]["transferable"].sum())}

    # ---------- verdict ----------
    shared_capture = float(t2["kryzei_shared_capture"])
    t3_sig = bool(rho_pe < 0.05 and corr_pat_err > 0)
    ns_transferable = int(ns["transferable"].sum())
    if shared_capture > 0.60 and t3_sig:
        verdict = "LABEL_PROTOCOL_BOTTLENECK_EXP031_EXPAND_SOURCE"
    elif shared_capture < 0.30:
        verdict = "DOMAIN_SHIFT_BOTTLENECK_EXP031_DOMAIN_ADAPTATION"
    else:
        verdict = "INTERMEDIATE_REPORT_AS_IS"
    if ns_transferable < 3:
        verdict += " | INSUFFICIENT_EVIDENCE_AT_THIS_COHORT_SIZE"
    log["verdict"] = {"kryzei_shared_capture": shared_capture,
                      "t3_patient_error_corr": corr_pat_err,
                      "t3_patient_error_p": p_pe,
                      "t4_transferable_nonstructural": ns_transferable,
                      "verdict": verdict}

    log["elapsed_seconds"] = round(time.time() - t0, 1)
    (OUT_DIR / "audit_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False, default=str))
    print(f"[exp029] done in {log['elapsed_seconds']}s verdict={verdict}",
          flush=True)


if __name__ == "__main__":
    main()
