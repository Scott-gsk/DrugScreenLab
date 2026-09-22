#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-033 工作包 D 任务 1+2：扰动介导路线能力梯度（零目标标签）。

任务 1（C4 输入）：按 EXP-027 协议对白名单 60 样本/51 患者 × 48 药生成
ctrl_morgan 零样本特征（reversal_score / pathway_reversal / perturb_magnitude，
3 种子），逆转评分参考 = 60 样本队列均值（sample-difference，NOT tumor-vs-normal）。
输出 artifacts/exp033_routeB/whitelist_chempert_features.parquet。

任务 2（路线 B 梯度，全零目标标签）：
  - global_mean_signature（全局扰动均值基线，常数/药，EXP-023 契约）
  - ctrl_only（线性 795->795，药不变，机制对照）
  - drug_only（Emb64->MLP512 药单独信息；Kryzei 药经 SMILES 桥映射 ChemPert
    词表，未见药=UNK 索引 0）
  - ctrl_morgan（任务 1 同名上游，= 任务 1 特征，3 种子均值）
  - XPert（较高臂，独立脚本 scripts/modeling/exp033_routeB_xpert.py）
排序对象 = 白名单 60 样本 48 药观测行；统计 = 患者内全药排序 Spearman /
逐药跨患者 Spearman / 前1~3选择质量 / NDCG@5(探索)；与 global_mean 基线
配对 bootstrap 2000 + Holm（主比较集）。

监督条件：全部零目标标签（不接触 Kryzei 药敏标签 fit），与路线 A 有监督臂
不直接对比。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("/mnt/d/Code/DrugScreenLab")
TABLE_PATH = ROOT / "data/processed/chempert/chempert_v2_response_table.parquet"
SLICE_PATH = ROOT / "data/processed/chempert/chempert_v2_control_expression_slice.parquet"
SMILES_PATH = ROOT / "data/processed/chempert/chempert_v2_chem_smiles.parquet"
PROC = ROOT / "data/processed/transfer/mcrc_v3"
IDENT = ROOT / "data/processed/identity/kryeziu_drug_identity_48.json"
AUDIT_PATH = ROOT / "data/processed/identity/kryzei_admission_audit_20260921.json"
HALLMARK_GMT = ROOT / "data/raw/exp013/msigdb/h.all.v2023.2.Hs.symbols.gmt"
OUT_DIR = ROOT / "artifacts" / "exp033_routeB"

SEEDS = [23, 24, 25]
CTRL_SEED = 23
EPOCHS, BATCH, LR, WD = 40, 512, 1e-3, 1e-4
HIDDEN = 512
EMB_DIM = 64
TOPK_PATHWAYS = 8
BOOT_SEED = 20260918
BOOT_N = 2000


# ---------------------------------------------------------------- models
class FeatCond(nn.Module):
    def __init__(self, n_genes, feat_dim, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes + feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_genes))

    def forward(self, ctrl, feat):
        return self.net(torch.cat([ctrl, feat], dim=-1))


class CtrlOnly(nn.Module):
    def __init__(self, n_genes):
        super().__init__()
        self.net = nn.Linear(n_genes, n_genes)

    def forward(self, ctrl, feat):
        return self.net(ctrl)


class DrugOnlyEmb(nn.Module):
    """EXP-023 DrugOnly 契约：Emb(n_drugs+1, 64) -> MLP(512) -> n_genes."""

    def __init__(self, n_genes, n_drugs, emb_dim=EMB_DIM, hidden=HIDDEN):
        super().__init__()
        self.emb = nn.Embedding(n_drugs + 1, emb_dim)  # index 0 = UNK
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_genes))

    def forward(self, drug_idx):
        return self.net(self.emb(drug_idx))


def train(model, X, F, Y, device):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X, F, Y), batch_size=BATCH,
                        shuffle=True, drop_last=False)
    model.train()
    for epoch in range(1, EPOCHS + 1):
        tot, n = 0.0, 0
        for xb, fb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb, fb), yb)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(f"    epoch {epoch}/{EPOCHS} train_mse={tot / max(n, 1):.6f}",
                  flush=True)


def train_drugonly(model, idx, Y, device):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(idx, Y), batch_size=BATCH,
                        shuffle=True, drop_last=False)
    model.train()
    for epoch in range(1, EPOCHS + 1):
        tot, n = 0.0, 0
        for ib, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(ib), yb)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * ib.size(0)
            n += ib.size(0)
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(f"    [drugonly] epoch {epoch}/{EPOCHS} train_mse={tot / max(n, 1):.6f}",
                  flush=True)


def morgan_fp(smiles: str):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
    arr = np.zeros(1024, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def canonical_smiles(smiles: str):
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)


def load_chempert():
    table = pd.read_parquet(TABLE_PATH)
    sl = pd.read_parquet(SLICE_PATH)
    genes = sl["gene_symbol"].tolist()
    prof_cols = [c for c in sl.columns if c != "gene_symbol"]
    slice_mat = sl[prof_cols].to_numpy(dtype=np.float32)
    col_pos = {c: j for j, c in enumerate(prof_cols)}
    gene_pos = {g: i for i, g in enumerate(genes)}

    sigs, cov = [], []
    for up, down in zip(table["up_genes"], table["down_genes"]):
        v = np.zeros(len(genes), dtype=np.float32)
        for g in str(up).split(";"):
            if g.strip() in gene_pos:
                v[gene_pos[g.strip()]] = 1.0
        for g in str(down).split(";"):
            if g.strip() in gene_pos:
                v[gene_pos[g.strip()]] = -1.0
        cov.append(int(v.any()))
        sigs.append(v)
    Y = np.stack(sigs)
    X = slice_mat[:, [col_pos[e] for e in table["Expression_ID"]]].T
    smiles = pd.read_parquet(SMILES_PATH).dropna(subset=["canonical_smiles"])
    chem2smi = dict(zip(smiles["Chemical_ID"],
                        smiles["canonical_smiles"].astype(str)))
    return table, X, Y, genes, cov, chem2smi


def reversal_from_delta(pred_delta, sample_diff, present):
    """pred_delta:(S,D,795) sample_diff:(S,795) -> reversal (S,D)."""
    pd_sig = pred_delta[:, :, present].astype(np.float64)
    sdiff = sample_diff[:, present]
    num = np.einsum("sdk,sk->sd", pd_sig, sdiff)
    den = (np.linalg.norm(pd_sig, axis=2)
           * np.linalg.norm(sdiff, axis=1)[:, None]) + 1e-12
    return -(num / den)


# ---------------------------------------------------------------- task 2 metrics
def _agg_patient_drug(frame):
    f = frame[["patient_id", "drug_id", "observed", "predicted"]].copy()
    f["observed"] = pd.to_numeric(f["observed"], errors="raise").astype(float)
    f["predicted"] = pd.to_numeric(f["predicted"], errors="raise").astype(float)
    return f.groupby(["patient_id", "drug_id"], sort=True, as_index=False).agg(
        observed=("observed", "mean"), predicted=("predicted", "mean"))


def _sp(y, p, min_drugs=3):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if len(y) < min_drugs or np.ptp(y) == 0 or np.ptp(p) == 0:
        return np.nan
    r = spearmanr(y, p).statistic
    return float(r) if np.isfinite(r) else np.nan


def macro_bootstrap(vals, seed=BOOT_SEED, n=BOOT_N):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) == 0:
        return {"point_estimate": None, "low": None, "high": None, "n_units": 0}
    rng = np.random.default_rng(seed)
    boot = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return {"point_estimate": float(v.mean()),
            "low": float(np.quantile(boot, 0.025)),
            "high": float(np.quantile(boot, 0.975)),
            "n_units": int(len(v)), "n": n, "seed": seed}


def arm_eval(frame, min_drugs=3):
    agg = _agg_patient_drug(frame)
    per_patient_rows = []
    valid_values = []
    reasons = {}
    for patient, g in agg.groupby("patient_id", sort=True):
        y = g["observed"].to_numpy(float)
        p = g["predicted"].to_numpy(float)
        n_drugs = int(len(y))
        if n_drugs < min_drugs:
            reasons["insufficient_drugs"] = reasons.get("insufficient_drugs", 0) + 1
            per_patient_rows.append((patient, np.nan))
            continue
        if np.ptp(y) == 0:
            reasons["constant_observed"] = reasons.get("constant_observed", 0) + 1
            per_patient_rows.append((patient, np.nan))
            continue
        if np.ptp(p) == 0:
            reasons["constant_predicted"] = reasons.get("constant_predicted", 0) + 1
            per_patient_rows.append((patient, np.nan))
            continue
        rho = float(spearmanr(y, p).statistic)
        rho = rho if np.isfinite(rho) else np.nan
        per_patient_rows.append((patient, rho))
        if np.isfinite(rho):
            valid_values.append(rho)
    intra_macro = float(np.mean(valid_values)) if valid_values else None

    # per-drug cross-patient spearman (min 5 patients)
    drug_rhos = []
    for drug, g in agg.groupby("drug_id", sort=True):
        y = g["observed"].to_numpy(float)
        p = g["predicted"].to_numpy(float)
        if len(y) >= 5 and np.ptp(y) > 0 and np.ptp(p) > 0:
            r = spearmanr(y, p).statistic
            if np.isfinite(r):
                drug_rhos.append(float(r))
    per_drug = float(np.mean(drug_rhos)) if drug_rhos else None

    topk = {}
    for k in (1, 2, 3):
        vals = []
        for _, g in agg.groupby("patient_id", sort=True):
            y = g["observed"].to_numpy(float)
            p = g["predicted"].to_numpy(float)
            if len(y) >= k and np.ptp(y) > 0 and np.ptp(p) > 0:
                top_y = set(np.argsort(-y, kind="mergesort")[:k])
                top_p = set(np.argsort(-p, kind="mergesort")[:k])
                vals.append(len(top_y & top_p) / k)
        topk[k] = float(np.mean(vals)) if vals else None

    ndcgs = []
    for _, g in agg.groupby("patient_id", sort=True):
        y = g["observed"].to_numpy(float)
        p = g["predicted"].to_numpy(float)
        if len(y) >= 10 and np.ptp(y) > 0 and np.ptp(p) > 0:
            v = _ndcg_at_k(y, p, 5)
            if np.isfinite(v):
                ndcgs.append(float(v))
    return {
        "intra_patient_spearman": intra_macro,
        "intra_patient_spearman_ci": macro_bootstrap(valid_values),
        "n_valid_patients": int(len(valid_values)),
        "n_patients": int(frame.patient_id.nunique()),
        "n_drugs": int(frame.drug_id.nunique()),
        "per_drug_spearman": per_drug,
        "top1": topk[1], "top2": topk[2], "top3": topk[3],
        "ndcg5": float(np.mean(ndcgs)) if ndcgs else None,
        "reason_counts": reasons,
        "_per_patient": [float(v) for v in [r[1] for r in per_patient_rows]],
    }


def _ndcg_at_k(observed, predicted, k):
    from scipy.stats import rankdata
    y = np.asarray(observed, float)
    p = np.asarray(predicted, float)
    if len(y) < k or np.ptp(y) == 0:
        return np.nan
    gains = (rankdata(y, method="average") - 1) / (len(y) - 1)
    discounts = 1 / np.log2(np.arange(2, k + 2))
    ideal = np.sort(gains)[::-1][:k] @ discounts
    total, offset = 0.0, 0
    for score in np.unique(p)[::-1]:
        mask = p == score
        count = int(mask.sum())
        total += gains[mask].mean() * discounts[offset:min(offset + count, k)].sum()
        offset += count
        if offset >= k:
            break
    return float(total / ideal) if ideal > 0 else np.nan


def per_drug_paired_bootstrap(arm_frame, base_frame, seed=BOOT_SEED, n=BOOT_N):
    """配对比：arm 与 base 的逐药跨患者 Spearman 宏平均差（患者级重采样）。"""
    A = _agg_patient_drug(arm_frame)
    B = _agg_patient_drug(base_frame)
    drugs = sorted(set(A["drug_id"]) & set(B["drug_id"]))
    patients = sorted(set(A["patient_id"]) & set(B["patient_id"]))
    ta = {d: A[A["drug_id"] == d].set_index("patient_id") for d in drugs}
    tb = {d: B[B["drug_id"] == d].set_index("patient_id") for d in drugs}

    def macro(ts, pick):
        vals = []
        for d, t in ts.items():
            sub = t.reindex(pick).dropna()
            if len(sub) < 5:
                continue
            r = _sp(sub["observed"].to_numpy(float), sub["predicted"].to_numpy(float))
            if np.isfinite(r):
                vals.append(r)
        return float(np.mean(vals)) if vals else np.nan

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        pick = rng.choice(patients, size=len(patients), replace=True)
        ma = macro(ta, pick)
        mb = macro(tb, pick)
        if np.isfinite(ma) and np.isfinite(mb):
            deltas.append(ma - mb)
    if not deltas:
        return {"point_estimate": None, "low": None, "high": None, "n_valid": 0}
    deltas = np.asarray(deltas)
    return {"point_estimate": float(deltas.mean()),
            "low": float(np.quantile(deltas, 0.025)),
            "high": float(np.quantile(deltas, 0.975)),
            "n_valid": int(len(deltas)), "n": n, "seed": seed,
            "p_greater_than_zero": float(np.mean(deltas <= 0))}


def holm_correct(p_values):
    p = np.asarray(list(p_values), dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * np.arange(m, 0, -1)
    adjusted = np.maximum.accumulate(adjusted)
    adjusted = np.minimum(adjusted, 1.0)
    out = np.empty(m)
    out[order] = adjusted
    return out.tolist()


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[exp033-routeB] device={device}", flush=True)

    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    whitelist = set(audit["pdo_whitelist"])

    # ---------- ChemPert 载入（同 EXP-027） ----------
    table, X, Y, genes, cov, chem2smi = load_chempert()
    nz = np.array(cov) > 0
    chem_ids = table["Chemical_ID"].to_numpy()

    fp_all = {}
    canon = {}
    for c in sorted(set(chem_ids[nz]) & set(chem2smi)):
        fp_all[c] = morgan_fp(chem2smi[c])
        canon[c] = canonical_smiles(chem2smi[c])
    parseable = {c for c, f in fp_all.items() if f is not None}
    encodable = nz & np.array([c in parseable for c in chem_ids])
    print(f"[chempert] responses={len(table)} nonzero={int(nz.sum())} "
          f"encodable={int(encodable.sum())}", flush=True)

    mu = X[nz].mean(axis=0, keepdims=True)
    sd = X[nz].std(axis=0, keepdims=True) + 1e-6
    Xz_all = ((X - mu) / sd).astype(np.float32)
    y_mu = Y[nz].mean(axis=0, keepdims=True).astype(np.float32)  # global mean sig

    # ---------- Kryzei 白名单样本 ----------
    npz = np.load(PROC / "indexed.npz", allow_pickle=True)
    kry_expr_all = np.asarray(npz["expression"], dtype=np.float32)
    kry_genes = [str(g) for g in np.asarray(npz["ordered_genes"]).tolist()]
    kry_sid_all = [str(s) for s in np.asarray(npz["sample_id"]).tolist()]
    sids = [(i, s) for i, s in enumerate(kry_sid_all) if s in whitelist]
    sid_idx = [i for i, _ in sids]
    kry_sid = [s for _, s in sids]
    kry_expr = kry_expr_all[sid_idx]
    print(f"[kryzei] whitelist samples={len(kry_sid)} (from {len(kry_sid_all)})",
          flush=True)

    gpos = {g: i for i, g in enumerate(kry_genes)}
    pos795 = np.array([gpos.get(g, -1) for g in genes])
    present = pos795 >= 0
    print(f"[kryzei] 795∩kryzei present={int(present.sum())} "
          f"missing={int((~present).sum())}", flush=True)

    Z_k = np.zeros((len(kry_sid), len(genes)), dtype=np.float32)
    Z_k[:, present] = kry_expr[:, pos795[present]]
    Z_k = ((Z_k - mu) / sd).astype(np.float32)
    sample_diff = (Z_k - Z_k.mean(axis=0, keepdims=True)).astype(np.float64)

    # Kryzei 药物指纹（48）
    ident = json.loads(IDENT.read_text(encoding="utf-8"))
    kry_drugs = ident["drugs"]
    fps, drug_ids = {}, []
    for d in kry_drugs:
        fp = morgan_fp(d["smiles"])
        if fp is not None:
            fps[d["drug_id"]] = fp
        drug_ids.append(d["drug_id"])
    F_k = np.stack([fps[d] for d in drug_ids]).astype(np.float32)
    print(f"[kryzei] drugs fp={len(fps)}/48", flush=True)

    # Kryzei -> ChemPert 词表映射（SMILES 桥）
    smi2chem = {}
    for c, cn in canon.items():
        smi2chem.setdefault(cn, c)
    kry_canon = {d["drug_id"]: canonical_smiles(d["smiles"]) for d in kry_drugs}
    kry_seen_chem = {}
    seen, unseen = 0, 0
    for d in kry_drugs:
        cs = kry_canon[d["drug_id"]]
        if cs is not None and cs in smi2chem:
            kry_seen_chem[d["drug_id"]] = smi2chem[cs]
            seen += 1
        else:
            unseen += 1
    print(f"[drugonly] kryzei seen-in-chempert={seen} unseen(UNK)={unseen}",
          flush=True)

    # drug_only 词表（全非零签名行）
    vocab = {}
    for c in chem_ids[nz]:
        if c not in vocab:
            vocab[c] = len(vocab) + 1
    n_vocab = len(vocab)

    # top-8 Hallmark 通路
    hallmark = {}
    with open(HALLMARK_GMT, "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                hallmark[parts[0]] = [g for g in parts[2:] if g]
    gene_set795 = set(genes)
    pw_cov = {p: len([g for g in gs if g in gene_set795]) for p, gs in hallmark.items()}
    top_pw = sorted(hallmark, key=lambda p: (-pw_cov[p], p))[:TOPK_PATHWAYS]
    pos795_map = {g: i for i, g in enumerate(genes)}
    pw_cols, pw_sample_cols = [], []
    for p in top_pw:
        cols = [pos795_map[g] for g in hallmark[p]
                if g in pos795_map and present[pos795_map[g]]]
        pw_cols.append(cols)
        pw_sample_cols.append(cols)
    sd_pw = np.stack([sample_diff[:, cols].mean(axis=1) for cols in pw_sample_cols],
                     axis=1)

    Zt = torch.from_numpy(Z_k).to(device)
    Ft = torch.from_numpy(F_k).to(device)

    idx_cm = np.where(encodable)[0]
    X_cm = torch.from_numpy(Xz_all[idx_cm]).to(device)
    F_cm_np = np.zeros((len(idx_cm), 1024), dtype=np.float32)
    for i, c in enumerate(chem_ids[idx_cm]):
        F_cm_np[i] = fp_all[c]
    F_cm = torch.from_numpy(F_cm_np).to(device)
    Y_cm = torch.from_numpy(Y[idx_cm]).to(device)

    idx_co = np.where(nz)[0]

    log = {"status": "RAN", "env": "drugscreening-gpu",
           "n_whitelist_samples": len(kry_sid), "n_drugs": len(drug_ids),
           "genes_795_in_kryzei": int(present.sum()),
           "genes_795_missing_in_kryzei": int((~present).sum()),
           "drug_only_seen_in_chempert": seen, "drug_only_unseen": unseen,
           "global_mean_note": "global_mean_signature = ChemPert 非零签名训练均值 y_mu（常数/药）",
           "hyperparams": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "wd": WD,
                           "hidden": HIDDEN, "emb_dim": EMB_DIM, "seeds": SEEDS}}

    # ---------------- ctrl_morgan（任务 1 特征） + ctrl_only + drug_only ----------------
    feat_rows = []       # 任务 1 输出
    arm_rev = {}         # arm -> {seed|'mean': (S, D) reversal}
    cm_seeds = {}

    for seed in SEEDS:
        # ctrl_morgan
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = FeatCond(len(genes), 1024).to(device)
        train(model, X_cm, F_cm, Y_cm, device)
        with torch.no_grad():
            model.eval()
            preds = []
            for si in range(len(kry_sid)):
                z_rep = Zt[si].unsqueeze(0).expand(len(drug_ids), -1)
                preds.append(model(z_rep, Ft).cpu().numpy())
            pred_delta = np.stack(preds)          # (S,48,795)
        magnitude = np.linalg.norm(pred_delta, axis=2)
        reversal = reversal_from_delta(pred_delta, sample_diff, present)
        pw_delta = np.stack([pred_delta[:, :, cols].mean(axis=2) for cols in pw_cols], axis=2)
        pnum = np.einsum("sdp,sp->sd", pw_delta, sd_pw)
        pden = (np.linalg.norm(pw_delta, axis=2)
                * np.linalg.norm(sd_pw, axis=1)[:, None]) + 1e-12
        pw_rev = -(pnum / pden)
        for si, sid in enumerate(kry_sid):
            for di, did in enumerate(drug_ids):
                feat_rows.append({"sample_id": sid, "drug_id": did, "seed": seed,
                                  "reversal_score": float(reversal[si, di]),
                                  "pathway_reversal": float(pw_rev[si, di]),
                                  "perturb_magnitude": float(magnitude[si, di])})
        cm_seeds[seed] = reversal
        del model
        torch.cuda.empty_cache()
        print(f"[ctrl_morgan] seed {seed} reversal mean={reversal.mean():.4f}",
              flush=True)

    # ctrl_only (seed 23, 全非零行)
    torch.manual_seed(CTRL_SEED)
    np.random.seed(CTRL_SEED)
    co = CtrlOnly(len(genes)).to(device)
    train(co, torch.from_numpy(Xz_all[idx_co]).to(device),
          torch.zeros((len(idx_co), 1), dtype=torch.float32, device=device),
          torch.from_numpy(Y[idx_co]).to(device), device)
    with torch.no_grad():
        co.eval()
        pred_co = co(Zt, torch.zeros((len(kry_sid), 1), device=device)).cpu().numpy()
    pred_co_delta = np.repeat(pred_co[:, None, :], len(drug_ids), axis=1)
    co_rev = reversal_from_delta(pred_co_delta, sample_diff, present)
    arm_rev["ctrl_only"] = {"mean": co_rev}
    del co
    torch.cuda.empty_cache()

    # drug_only (Embedding, 3 seeds, 全非零行)
    do_seeds = {}
    idx_co_drug = torch.from_numpy(np.array([vocab.get(c, 0) for c in chem_ids[idx_co]], dtype=np.int64))
    Y_co = torch.from_numpy(Y[idx_co]).to(device)
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        dm = DrugOnlyEmb(len(genes), n_vocab).to(device)
        train_drugonly(dm, idx_co_drug.to(device), Y_co, device)
        with torch.no_grad():
            dm.eval()
            kry_drug_idx = torch.tensor(
                [vocab.get(kry_seen_chem.get(d), 0) for d in drug_ids],
                dtype=torch.long, device=device)
            delta_do = dm(kry_drug_idx).cpu().numpy()      # (48, 795)
        pred_do_delta = np.repeat(delta_do[None, :, :], len(kry_sid), axis=0)
        do_rev = reversal_from_delta(pred_do_delta, sample_diff, present)
        do_seeds[seed] = do_rev
        del dm
        torch.cuda.empty_cache()
    arm_rev["drug_only"] = {"mean": np.mean(list(do_seeds.values()), axis=0)}

    # ctrl_morgan / drug_only / ctrl_only / global_mean 均值
    arm_rev["ctrl_morgan"] = {"mean": np.mean(list(cm_seeds.values()), axis=0)}
    gm_delta = np.repeat(np.repeat(y_mu[None, :, :], len(kry_sid), axis=0), len(drug_ids), axis=1)
    arm_rev["global_mean"] = {"mean": reversal_from_delta(gm_delta, sample_diff, present)}

    # ---------------- 落盘任务 1 特征 ----------------
    feats = pd.DataFrame(feat_rows)
    feats.to_parquet(OUT_DIR / "whitelist_chempert_features.parquet", index=False)
    print(f"[task1] whitelist_chempert_features rows={len(feats)} "
          f"({len(feats) // (len(kry_sid) * len(drug_ids))} seeds)", flush=True)

    # ---------------- 任务 2：排序评估 ----------------
    samples = pd.read_parquet(PROC / "samples.parquet")
    pairs = pd.read_parquet(PROC / "pairs.parquet")
    samples = samples.copy()
    samples["sample_id"] = samples["sample_id"].astype(str)
    samples["patient_id"] = samples["patient_id"].astype(str)
    pairs = pairs.copy()
    pairs["sample_id"] = pairs["sample_id"].astype(str)
    pairs["drug_id"] = pairs["drug_id"].astype(str)
    obs = pairs.merge(samples[["sample_id", "patient_id"]], on="sample_id", how="inner")
    obs = obs[obs["sample_id"].isin(set(kry_sid))].copy()
    obs = obs.rename(columns={"response": "observed"})
    obs["observed"] = pd.to_numeric(obs["observed"], errors="raise").astype(float)

    def arm_frame(rev):
        rows = []
        for si, sid in enumerate(kry_sid):
            for di, did in enumerate(drug_ids):
                rows.append({"sample_id": sid, "drug_id": did,
                             "predicted": float(rev[si, di])})
        f = pd.DataFrame(rows).merge(obs[["sample_id", "drug_id", "patient_id", "observed"]],
                                     on=["sample_id", "drug_id"], how="inner")
        return f

    eval_out = {}
    frames = {}
    for arm in ["global_mean", "ctrl_only", "drug_only", "ctrl_morgan"]:
        frames[arm] = arm_frame(arm_rev[arm]["mean"])
        eval_out[arm] = {k: v for k, v in arm_eval(frames[arm]).items()
                         if not k.startswith("_")}

    # 配对 vs global_mean（逐药跨患者 Spearman，患者级 bootstrap）+ Holm
    base_frame = frames["global_mean"]
    pvals = []
    paired = {}
    comparators = ["ctrl_only", "drug_only", "ctrl_morgan"]
    for arm in comparators:
        pb = per_drug_paired_bootstrap(frames[arm], base_frame)
        pt = pb["point_estimate"]
        p_greater = pb.get("p_greater_than_zero")
        p_two = None
        if p_greater is not None:
            p_two = min(1.0, 2.0 * min(p_greater, 1.0 - p_greater))
        paired[f"{arm}_minus_global_mean"] = {
            "supervision": "both zero-target-label",
            "metric": "per_drug_cross_patient_spearman",
            "point_estimate": pt, "low": pb["low"], "high": pb["high"],
            "n_valid_resamples": pb["n_valid"], "p_greater_than_zero": p_greater,
            "p_two_sided": p_two}
        if p_two is not None:
            pvals.append(p_two)
    paired["holm_p"] = holm_correct(pvals) if pvals else None
    paired["holm_applied_over"] = [f"{a}_minus_global_mean" for a in comparators]
    paired["note"] = ("global_mean/ctrl_only 预测药不变，患者内全药排序为常值 "
                      "（intra_patient_spearman 未定义、n_valid=0）；配对主比较在 "
                      "逐药跨患者 Spearman（个性化终点，各臂均有定义）上进行。")

    summary = {
        "experiment_id": "EXP-033", "task": "route_B_gradient (D1)",
        "status": "RAN",
        "supervision": "all_zero_target_label",
        "ranking_object": "whitelist 60 samples x 48 drugs observation rows",
        "seed_avg_note": "ctrl_morgan/drug_only 为 3 种子均值；ctrl_only 单种子 23；"
                         "global_mean 无随机性",
        "logs": log,
        "arms": eval_out,
        "paired_vs_global_mean": paired,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT_DIR / "routeB_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8")
    log["elapsed_sec"] = round(time.time() - t0, 1)
    (OUT_DIR / "routeB_upstream_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[exp033-routeB] DONE", flush=True)
    print(json.dumps({a: {k: eval_out[a][k] for k in
                          ("intra_patient_spearman", "per_drug_spearman",
                           "top1", "ndcg5")} for a in eval_out},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()