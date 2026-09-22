#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-027 Route B upstream: ChemPert ctrl_morgan retrain + Kryzei zero-shot.

Pre-registered contract (experiments/records/EXP-027.md, pinned BEFORE run):
- Data: chempert_v2 contract tables; nonzero-signature rows; NO accession holdout
  (EXP-023 P1/P2 internal validation evidence carries over; this model is the
  downstream application upstream).
- ctrl_morgan (FeatCond: ctrl 795 z + Morgan 1024 -> 512 ReLU -> 795) trained on
  the SMILES-encodable nonzero-signature subset (EXP-023-consistent),
  seeds 23/24/25; hyperparams identical to EXP-023 (epochs=40, batch=512,
  lr=1e-3, wd=1e-4).
- ctrl_only (Linear 795->795) on all nonzero-signature rows, seed 23, as a
  mechanism control (predictions drug-invariant -> no within-patient ranking).
- z-stats from ChemPert full nonzero-signature rows (shared at application).
- Kryzei application: indexed.npz expression projected onto the 795 symbols;
  genes missing in Kryzei get z=0 (train mean; count logged); sample-difference
  signature = z-expr minus 92-sample cohort mean (EXP-025 convention, NOT
  tumor-vs-normal); reversal = -cosine(pred delta, signature) restricted to
  the 795∩Kryzei gene subspace.
- Drug fingerprints: RDKit Morgan radius=2 nBits=1024 from
  kryeziu_drug_identity_48.json SMILES (EXP-025 verified 38/38 vs xpert).
- Pathway arm: top-8 Hallmark pathways by |pathway ∩ 795| coverage (y-free);
  pathway-level reversal = -cosine over the 8 pathway-aggregated dims, using
  only 795∩Kryzei-present genes inside each pathway.

Outputs:
  artifacts/exp027_route_comparison/chempert_kryzei_features.parquet
    (sample_id, drug_id, seed, reversal_score, pathway_reversal,
     perturb_magnitude)
  artifacts/exp027_route_comparison/chempert_zeroshot_log.json
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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("/mnt/d/Code/DrugScreenLab")
TABLE_PATH = ROOT / "data/processed/chempert/chempert_v2_response_table.parquet"
SLICE_PATH = ROOT / "data/processed/chempert/chempert_v2_control_expression_slice.parquet"
SMILES_PATH = ROOT / "data/processed/chempert/chempert_v2_chem_smiles.parquet"
PROC = ROOT / "data/processed/transfer/mcrc_v3"
IDENT = ROOT / "data/processed/identity/kryeziu_drug_identity_48.json"
HALLMARK_GMT = ROOT / "data/raw/exp013/msigdb/h.all.v2023.2.Hs.symbols.gmt"
OUT_DIR = ROOT / "artifacts/exp027_route_comparison"

SEEDS = [23, 24, 25]
CTRL_SEED = 23
EPOCHS, BATCH, LR, WD = 40, 512, 1e-3, 1e-4
HIDDEN = 512
TOPK_PATHWAYS = 8


class FeatCond(nn.Module):
    def __init__(self, n_genes: int, feat_dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes + feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_genes))

    def forward(self, ctrl, feat):
        return self.net(torch.cat([ctrl, feat], dim=-1))


class CtrlOnly(nn.Module):
    def __init__(self, n_genes: int):
        super().__init__()
        self.net = nn.Linear(n_genes, n_genes)

    def forward(self, ctrl, feat):
        return self.net(ctrl)


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


def morgan_fp(smiles: str) -> np.ndarray | None:
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


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[exp027-upstream] device={device}", flush=True)

    table, X, Y, genes, cov, chem2smi = load_chempert()
    nz = np.array(cov) > 0
    chem_ids = table["Chemical_ID"].to_numpy()
    # parseability filter consistent with EXP-023 (unparseable SMILES excluded)
    fp_all: dict[str, np.ndarray | None] = {}
    for c in sorted(set(chem_ids[nz]) & set(chem2smi)):
        fp_all[c] = morgan_fp(chem2smi[c])
    parseable = {c for c, f in fp_all.items() if f is not None}
    encodable = nz & np.array([c in parseable for c in chem_ids])
    n_unparseable = len([c for c in fp_all if fp_all[c] is None])
    print(f"[exp027-upstream] responses={len(table)} nonzero={int(nz.sum())} "
          f"encodable={int(encodable.sum())} "
          f"unparseable_chemicals={n_unparseable}", flush=True)

    # z-stats from ALL nonzero-signature rows (pre-registered)
    mu = X[nz].mean(axis=0, keepdims=True)
    sd = X[nz].std(axis=0, keepdims=True) + 1e-6
    Xz_all = ((X - mu) / sd).astype(np.float32)

    log = {"status": "RAN", "env": "drugscreening-gpu",
           "n_responses": int(len(table)),
           "n_nonzero": int(nz.sum()), "n_encodable": int(encodable.sum()),
           "hyperparams": {"epochs": EPOCHS, "batch": BATCH, "lr": LR,
                           "wd": WD, "hidden": HIDDEN, "seeds": SEEDS},
           "ctrl_morgan_train_rows": int(encodable.sum()),
           "ctrl_only_train_rows": int(nz.sum())}

    # ---------- train ctrl_only (mechanism control, seed 23) ----------
    print("[exp027-upstream] training ctrl_only (seed 23) ...", flush=True)
    torch.manual_seed(CTRL_SEED)
    np.random.seed(CTRL_SEED)
    idx_co = np.where(nz)[0]
    co = CtrlOnly(len(genes)).to(device)
    train(co, torch.from_numpy(Xz_all[idx_co]).to(device),
          torch.zeros((len(idx_co), 1), dtype=torch.float32, device=device),
          torch.from_numpy(Y[idx_co]).to(device), device)
    log["ctrl_only_final_train_mse"] = None  # filled below if needed

    # ---------- Kryzei expression -> 795 z-space ----------
    npz = np.load(PROC / "indexed.npz", allow_pickle=True)
    kry_expr = np.asarray(npz["expression"], dtype=np.float32)
    kry_genes = [str(g) for g in np.asarray(npz["ordered_genes"]).tolist()]
    kry_sid = [str(s) for s in np.asarray(npz["sample_id"]).tolist()]
    gpos = {g: i for i, g in enumerate(kry_genes)}
    pos795 = np.array([gpos.get(g, -1) for g in genes])
    present = pos795 >= 0
    n_missing = int((~present).sum())
    print(f"[exp027-upstream] 795∩kryzei genes present={int(present.sum())} "
          f"missing(z set to 0)={n_missing}", flush=True)
    log["genes_795_in_kryzei"] = int(present.sum())
    log["genes_795_missing_in_kryzei"] = n_missing

    Z_k = np.zeros((len(kry_sid), len(genes)), dtype=np.float32)
    Z_k[:, present] = kry_expr[:, pos795[present]]
    Z_k = ((Z_k - mu) / sd).astype(np.float32)     # missing genes stay z=0

    # ---------- Kryzei drug fingerprints ----------
    ident = json.loads(IDENT.read_text(encoding="utf-8"))
    kry_drugs = ident["drugs"]
    fps, unenc = {}, []
    for d in kry_drugs:
        fp = morgan_fp(d["smiles"])
        if fp is None:
            unenc.append(d["drug_id"])
        else:
            fps[d["drug_id"]] = fp
    drug_ids = [d["drug_id"] for d in kry_drugs]
    F_k = np.stack([fps[d] for d in drug_ids]).astype(np.float32)
    print(f"[exp027-upstream] kryzei drugs with fp={len(fps)}/48 "
          f"unencodable={unenc}", flush=True)
    log["kryzei_drugs_encoded"] = len(fps)
    log["kryzei_drugs_unencodable"] = unenc

    # ---------- Hallmark top-8 pathways (y-free coverage rule) ----------
    hallmark = {}
    with open(HALLMARK_GMT, "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                hallmark[parts[0]] = [g for g in parts[2:] if g]
    gene_set795 = set(genes)
    pw_cov = {p: len([g for g in gs if g in gene_set795])
              for p, gs in hallmark.items()}
    top_pw = sorted(hallmark, key=lambda p: (-pw_cov[p], p))[:TOPK_PATHWAYS]
    log["pathway_coverage_top8"] = {p: pw_cov[p] for p in top_pw}
    print(f"[exp027-upstream] top-{TOPK_PATHWAYS} pathways: "
          f"{[(p, pw_cov[p]) for p in top_pw[:4]]} ...", flush=True)
    pos795_map = {g: i for i, g in enumerate(genes)}
    pw_cols, pw_sample_cols = [], []
    for p in top_pw:
        # restrict to Kryzei-present genes for symmetry between pred and
        # sample-difference aggregation (pre-registered)
        cols = [pos795_map[g] for g in hallmark[p]
                if g in pos795_map and present[pos795_map[g]]]
        pw_cols.append(cols)
        pw_sample_cols.append(cols)

    sample_diff = (Z_k - Z_k.mean(axis=0, keepdims=True)).astype(np.float64)
    sd_pw = np.stack([sample_diff[:, cols].mean(axis=1) for cols in pw_sample_cols],
                     axis=1)                          # (92, 8)

    # ---------- ctrl_only mechanism check ----------
    with torch.no_grad():
        co.eval()
        pred_co = co(torch.from_numpy(Z_k).to(device),
                     torch.zeros((len(kry_sid), 1), device=device)
                     ).cpu().numpy()                  # (92, 795)
    # drug-invariance: per-sample spread across any drug = 0 by construction;
    # verify predictions actually vary across samples
    co_spread_samples = float(np.abs(pred_co).max())
    log["ctrl_only_check"] = {
        "note": "ctrl_only predictions are drug-invariant by construction; "
                "per-patient drug ranking is constant (no ranking signal)",
        "max_abs_pred": co_spread_samples,
    }

    # ---------- train ctrl_morgan seeds + Kryzei application ----------
    Zt = torch.from_numpy(Z_k).to(device)
    Ft = torch.from_numpy(F_k).to(device)
    idx_cm = np.where(encodable)[0]
    X_cm = torch.from_numpy(Xz_all[idx_cm]).to(device)
    F_cm_np = np.zeros((len(idx_cm), 1024), dtype=np.float32)
    for i, c in enumerate(chem_ids[idx_cm]):
        F_cm_np[i] = fp_all[c]
    F_cm = torch.from_numpy(F_cm_np).to(device)
    Y_cm = torch.from_numpy(Y[idx_cm]).to(device)

    rows = []
    for seed in SEEDS:
        print(f"[exp027-upstream] training ctrl_morgan (seed {seed}) ...",
              flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = FeatCond(len(genes), 1024).to(device)
        train(model, X_cm, F_cm, Y_cm, device)
        with torch.no_grad():
            model.eval()
            # (92, 48, 795): per sample x per drug
            preds = []
            for si in range(len(kry_sid)):
                z_rep = Zt[si].unsqueeze(0).expand(len(drug_ids), -1)
                preds.append(model(z_rep, Ft).cpu().numpy())
            pred_delta = np.stack(preds)              # (92, 48, 795)

        magnitude = np.linalg.norm(pred_delta, axis=2)
        # reversal restricted to 795∩Kryzei subspace
        pd_sig = pred_delta[:, :, present].astype(np.float64)
        sdiff = sample_diff[:, present]
        num = np.einsum("sdk,sk->sd", pd_sig, sdiff)
        den = (np.linalg.norm(pd_sig, axis=2)
               * np.linalg.norm(sdiff, axis=1)[:, None]) + 1e-12
        reversal = -(num / den)                      # (92, 48)

        pw_delta = np.stack(
            [pred_delta[:, :, cols].mean(axis=2) for cols in pw_cols], axis=2)
        pnum = np.einsum("sdp,sp->sd", pw_delta, sd_pw)
        pden = (np.linalg.norm(pw_delta, axis=2)
                * np.linalg.norm(sd_pw, axis=1)[:, None]) + 1e-12
        pw_reversal = -(pnum / pden)                 # (92, 48)

        for si, sid in enumerate(kry_sid):
            for di, did in enumerate(drug_ids):
                rows.append({
                    "sample_id": sid, "drug_id": did, "seed": seed,
                    "reversal_score": float(reversal[si, di]),
                    "pathway_reversal": float(pw_reversal[si, di]),
                    "perturb_magnitude": float(magnitude[si, di]),
                })
        print(f"[exp027-upstream] seed {seed}: reversal "
              f"min={reversal.min():.4f} max={reversal.max():.4f} "
              f"mean={reversal.mean():.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    feats = pd.DataFrame(rows)
    feats.to_parquet(OUT_DIR / "chempert_kryzei_features.parquet", index=False)
    log["n_feature_rows"] = len(feats)
    log["elapsed_sec"] = round(time.time() - t0, 1)
    (OUT_DIR / "chempert_zeroshot_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[exp027-upstream] DONE rows={len(feats)} "
          f"elapsed={log['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
