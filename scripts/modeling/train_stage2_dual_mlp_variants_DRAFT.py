#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-2 DualMLP variants: deeper MLP or PCA pathway-proxy features. One GPU process."""
from __future__ import annotations

import argparse
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
PROC = ROOT / "data/processed/transfer/mcrc_v3"
SPLIT_DRAFT = ROOT / "data/splits/kryeziu_patient_outer5fold_DRAFT.json"
SPLIT_FROZEN = ROOT / "data/splits/kryeziu_patient_outer5fold_FROZEN.json"
# frozen contract preferred (patient folds verified identical to DRAFT 2026-09-19)
SPLIT = SPLIT_FROZEN if SPLIT_FROZEN.exists() else SPLIT_DRAFT
OUT_DIR = ROOT / "artifacts/route1_stage2_dual_mlp_variants_DRAFT"
REPORT_DIR = ROOT / "docs/reports"


def inner_val_patients(train_p, seed):
    """Pinned inner-val algorithm (docs/reports/split_freeze_checklist_DRAFT.md, 2026-09-18):
    patient-level 4:1 split drawn from the outer fold's train_patient_ids ONLY; lexicographic sort,
    then numpy.random.RandomState(seed).permutation; the LAST ceil(n/5) patients are val
    (at least 1 val when n<5; skip inner val when n==1). Outer test patients never participate."""
    pats = sorted(str(p) for p in train_p)
    n = len(pats)
    if n <= 1:
        return [], True
    order = np.random.RandomState(seed).permutation(n)
    n_val = max(1, int(np.ceil(n / 5)))
    val = sorted(pats[i] for i in order[-n_val:])
    return val, False


def spearman(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 3:
        return float("nan")
    y, p = y[m], p[m]
    ry = y.argsort().argsort().astype(float)
    rp = p.argsort().argsort().astype(float)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    rp = (rp - rp.mean()) / (rp.std() + 1e-12)
    return float(np.mean(ry * rp))


def ndcg_at_k(y, p, k=5):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if len(y) == 0:
        return float("nan")
    k = min(k, len(y))
    order = np.argsort(-p)[:k]
    gains = y[order]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains * disc))
    ideal = y[np.argsort(-y)[:k]]
    idcg = float(np.sum(ideal * disc))
    return dcg / idcg if idcg > 0 else float("nan")


def patient_metrics(patients, y, pred):
    nds, sps = [], []
    patients = np.asarray(patients)
    for pid in np.unique(patients):
        m = patients == pid
        if m.sum() < 2:
            continue
        nds.append(ndcg_at_k(y[m], pred[m], 5))
        sps.append(spearman(y[m], pred[m]))
    return {
        "n_patients": len(nds),
        "ndcg@5": float(np.nanmean(nds)) if nds else float("nan"),
        "spearman": float(np.nanmean(sps)) if sps else float("nan"),
    }


class DualMLP(nn.Module):
    def __init__(self, n_in: int, n_drugs: int, drug_dim: int = 64, hidden: int = 256, depth: int = 2):
        super().__init__()
        self.drug_emb = nn.Embedding(n_drugs, drug_dim)
        layers = []
        d = n_in
        h = hidden
        for i in range(max(1, depth)):
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.2)]
            d = h
            h = max(128, h // 2) if i < depth - 1 else 128
        if not layers or layers[-1] is None:
            pass
        # ensure final expr dim 128
        if d != 128:
            layers += [nn.Linear(d, 128), nn.ReLU()]
        self.expr = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(128 + drug_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x_expr, drug_idx):
        h = torch.cat([self.expr(x_expr), self.drug_emb(drug_idx)], dim=-1)
        return self.head(h).squeeze(-1)


def load_tables():
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    samples = pd.read_parquet(PROC / "samples.parquet")
    pairs = pd.read_parquet(PROC / "pairs.parquet")
    npz = np.load(PROC / "indexed.npz", allow_pickle=True)
    sids = [str(x) for x in np.asarray(npz["sample_id"]).tolist()]
    expr = np.asarray(npz["expression"], dtype=np.float32)
    sid_to_idx = {s: i for i, s in enumerate(sids)}
    s_pid, s_sid, p_sid, p_did, p_y = "patient_id", "sample_id", "sample_id", "drug_id", "response"
    samp = samples.copy()
    samp[s_sid] = samp[s_sid].astype(str)
    samp[s_pid] = samp[s_pid].astype(str)
    pr = pairs.copy()
    pr[p_sid] = pr[p_sid].astype(str)
    pr[p_did] = pr[p_did].astype(str)
    merged = pr.merge(samp[[s_sid, s_pid]], left_on=p_sid, right_on=s_sid, how="inner").dropna(subset=[p_y])
    all_p = set()
    for f in split["folds"]:
        all_p |= set(map(str, f.get("train_patient_ids") or f.get("train_patients") or []))
        all_p |= set(map(str, f.get("test_patient_ids") or f.get("test_patients") or []))
    merged = merged[merged[s_pid].isin(all_p)].copy()
    drugs = sorted(merged[p_did].astype(str).unique())
    drug_to_i = {d: i for i, d in enumerate(drugs)}
    return split, merged, expr, sid_to_idx, drug_to_i, s_pid, p_sid, p_did, p_y


def fold_matrices(merged, expr, sid_to_idx, drug_to_i, train_p, test_p, s_pid, p_sid, p_did, p_y, feat: str, pca_dim: int, val_p=None):
    """Build train/(optional val)/test matrices. All preprocessing (gene variance
    filter, standardization, PCA) is fitted on the train rows ONLY; val/test are
    transformed with the train-fitted parameters."""

    def take(df):
        rows = []
        for _, r in df.iterrows():
            j = sid_to_idx.get(str(r[p_sid]))
            if j is None:
                continue
            rows.append((j, drug_to_i[str(r[p_did])], float(r[p_y]), str(r[s_pid]), str(r[p_sid]), str(r[p_did])))
        return rows

    tr = take(merged[merged[s_pid].isin(train_p)])
    te = take(merged[merged[s_pid].isin(test_p)])
    va = take(merged[merged[s_pid].isin(set(val_p))]) if val_p else []
    tr_i = np.array([x[0] for x in tr], int)
    te_i = np.array([x[0] for x in te], int)
    va_i = np.array([x[0] for x in va], int) if va else np.zeros(0, int)
    Xtr_raw = expr[tr_i]
    Xte_raw = expr[te_i]
    Xva_raw = expr[va_i] if va else np.zeros((0, Xtr_raw.shape[1]), np.float32)
    # select genes on train only
    gene_idx = np.argsort(-Xtr_raw.var(axis=0))[:2000]
    Xtr = Xtr_raw[:, gene_idx]
    Xte = Xte_raw[:, gene_idx]
    Xva = Xva_raw[:, gene_idx] if va else Xva_raw
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    if va:
        Xva = (Xva - mu) / sd
    if feat == "pathway_hallmark_ssgsea":
        import json
        import pandas as pd
        pw = ROOT / "data/processed/identity/pathway/kryeziu_hallmark_ssgsea_DRAFT.parquet"
        order_path = ROOT / "data/processed/identity/pathway/kryeziu_hallmark_feature_order_DRAFT.json"
        order = json.loads(order_path.read_text(encoding="utf-8"))
        feats = order.get("features") or order.get("feature_order") or order.get("pathways")
        mat = pd.read_parquet(pw)
        if "sample_id" in mat.columns:
            mat = mat.set_index("sample_id")
        mat.index = mat.index.astype(str)
        if feats is None:
            feats = [c for c in mat.columns if c != "patient_id"]
        mat = mat.reindex(columns=list(feats)).fillna(0.0).astype(np.float32)
        tr_s = [str(x[4]) for x in tr]
        te_s = [str(x[4]) for x in te]
        va_s = [str(x[4]) for x in va]
        Xtr = mat.reindex(tr_s).fillna(0.0).to_numpy(dtype=np.float32)
        Xte = mat.reindex(te_s).fillna(0.0).to_numpy(dtype=np.float32)
        Xva = mat.reindex(va_s).fillna(0.0).to_numpy(dtype=np.float32) if va else Xva
        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True) + 1e-6
        Xtr = ((Xtr - mu) / sd).astype(np.float32)
        Xte = ((Xte - mu) / sd).astype(np.float32)
        if va:
            Xva = ((Xva - mu) / sd).astype(np.float32)
    elif feat == "pathway_pca":
        # DRAFT pathway proxy: train-fit PCA on top genes (no Hallmark GMT frozen yet)
        Xc = Xtr - Xtr.mean(axis=0, keepdims=True)
        # economy SVD
        k = min(pca_dim, Xc.shape[0] - 1, Xc.shape[1])
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        W = vt[:k].T  # genes x k
        Xtr = (Xtr @ W).astype(np.float32)
        Xte = (Xte @ W).astype(np.float32)
        if va:
            Xva = (Xva @ W).astype(np.float32)
    else:
        Xtr = Xtr.astype(np.float32)
        Xte = Xte.astype(np.float32)
        if va:
            Xva = Xva.astype(np.float32)

    def pack(rows):
        if not rows:
            return (np.zeros(0, np.int64), np.zeros(0, np.float32),
                    np.zeros(0, dtype=object), np.zeros(0, dtype=object), np.zeros(0, dtype=object))
        return (
            np.array([x[1] for x in rows], np.int64),
            np.array([x[2] for x in rows], np.float32),
            np.array([x[3] for x in rows], dtype=object),
            np.array([x[4] for x in rows], dtype=object),
            np.array([x[5] for x in rows], dtype=object),
        )

    dtr, ytr, ptr, sitr, drtr = pack(tr)
    dva, yva, pva, siva, drva = pack(va)
    dte, yte, pte, site, drte = pack(te)
    return Xtr, dtr, ytr, ptr, sitr, drtr, Xva, dva, yva, pva, siva, drva, Xte, dte, yte, pte, site, drte


def train_fold(device, Xtr, dtr, ytr, eval_sets, n_drugs, epochs, batch, hidden, depth, weight_decay=1e-4):
    """Train on (Xtr, dtr, ytr); eval_sets: dict name -> (X, d). Returns raw preds dict."""
    model = DualMLP(Xtr.shape[1], n_drugs, hidden=hidden, depth=depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(dtr), torch.from_numpy(ytr)),
        batch_size=batch,
        shuffle=True,
    )
    model.train()
    for ep in range(epochs):
        total = 0.0
        n = 0
        for xb, db, yb in loader:
            xb, db, yb = xb.to(device), db.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb, db), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
            n += len(yb)
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1}/{epochs} loss={total/max(n,1):.4f}", flush=True)
    model.eval()
    preds = {}
    with torch.no_grad():
        for name, (X, d) in eval_sets.items():
            if len(d) == 0:
                preds[name] = np.zeros(0, np.float32)
                continue
            preds[name] = model(torch.from_numpy(X).to(device), torch.from_numpy(d).to(device)).cpu().numpy()
    return preds, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--folds", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--feat", type=str, default="genes", choices=["genes", "pathway_pca", "pathway_hallmark_ssgsea"])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=None, help="training seed (torch/numpy); None keeps legacy unseeded behavior")
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR), help="output directory for checkpoints and results.json")
    ap.add_argument("--target", type=str, default="raw", choices=["raw", "b0_residual"],
                    help="raw: fit y; b0_residual: fit y - per-drug train mean, add it back at prediction (EXP-024 V1)")
    ap.add_argument("--hold-out-inner-val", action="store_true",
                    help="exclude pinned inner-val patients from training rows; preprocessing fitted on inner-train only; "
                         "val predictions dumped alongside test (EXP-024 V2 ensemble members)")
    ap.add_argument("--dump-preds", action="store_true",
                    help="dump per-sample predictions (patient_id, sample_id, drug_id, y, pred, fold, split, seed)")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(
        f"START tag={args.tag} gpu={args.gpu} device={device} folds={args.folds} feat={args.feat} hidden={args.hidden} depth={args.depth} wd={args.weight_decay} seed={args.seed} target={args.target} hold_out_inner_val={args.hold_out_inner_val} dump_preds={args.dump_preds} out_dir={args.out_dir}",
        flush=True,
    )
    if device.type == "cuda":
        print("GPU_NAME", torch.cuda.get_device_name(0), flush=True)
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    split, merged, expr, sid_to_idx, drug_to_i, s_pid, p_sid, p_did, p_y = load_tables()
    fold_ids = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    inner_seed = split.get("random_seed")
    inner_seed = int(inner_seed) if inner_seed else 20260918
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    pred_rows = []
    t0 = time.time()
    for fold in split["folds"]:
        f_id = int(fold.get("fold", fold.get("fold_id")))
        if f_id not in fold_ids:
            continue
        train_p = set(map(str, fold.get("train_patient_ids") or fold.get("train_patients")))
        test_p = set(map(str, fold.get("test_patient_ids") or fold.get("test_patients")))
        print(f"FOLD {f_id} train_p={len(train_p)} test_p={len(test_p)}", flush=True)
        val_p, inner_val_skipped = inner_val_patients(train_p, inner_seed)
        materialized = fold.get("val_patient_ids")
        if materialized:
            materialized = sorted(map(str, materialized))
            if materialized != val_p and not inner_val_skipped:
                raise SystemExit(f"fold{f_id}: pinned inner-val algorithm {val_p} != materialized {materialized}")
        n_ti = len(train_p) - len(val_p)
        print(
            f"INNER_VAL fold={f_id} random_seed={inner_seed} n_train_inner_patients={n_ti} n_val_patients={len(val_p)} "
            f"val_patient_ids={val_p}{' inner_val_skipped' if inner_val_skipped else ''}",
            flush=True,
        )
        if args.hold_out_inner_val and val_p and not inner_val_skipped:
            fit_p = train_p - set(val_p)
        else:
            fit_p = train_p
        Xtr, dtr, ytr, ptr, sitr, drtr, Xva, dva, yva, pva, siva, drva, Xte, dte, yte, pte, site, drte = fold_matrices(
            merged, expr, sid_to_idx, drug_to_i, fit_p, test_p, s_pid, p_sid, p_did, p_y, args.feat, args.pca_dim,
            val_p=(val_p if (args.hold_out_inner_val and val_p and not inner_val_skipped) else None),
        )
        ytr_fit = ytr
        drug_mean_vec = None
        if args.target == "b0_residual":
            # per-drug mean from the FIT rows only (train patients; unseen by test)
            n_drugs_total = len(drug_to_i)
            counts = np.bincount(dtr, minlength=n_drugs_total)
            sums = np.bincount(dtr, weights=ytr.astype(float), minlength=n_drugs_total)
            gm = float(np.mean(ytr))
            drug_mean_vec = np.where(counts > 0, sums / np.maximum(counts, 1), gm).astype(np.float32)
            ytr_fit = ytr - drug_mean_vec[dtr]
        eval_sets = {"test": (Xte, dte)}
        if args.hold_out_inner_val and len(dva) > 0:
            eval_sets["val"] = (Xva, dva)
        preds, model = train_fold(
            device, Xtr, dtr, ytr_fit, eval_sets,
            n_drugs=len(drug_to_i), epochs=args.epochs, batch=args.batch,
            hidden=args.hidden, depth=args.depth, weight_decay=args.weight_decay,
        )
        pred_test = preds["test"] + (drug_mean_vec[dte] if drug_mean_vec is not None else 0.0)
        metrics = patient_metrics(pte, yte, pred_test)
        metrics["mae"] = float(np.mean(np.abs(yte - pred_test)))
        metrics["val"] = None
        if "val" in preds and len(dva) > 0:
            pred_val = preds["val"] + (drug_mean_vec[dva] if drug_mean_vec is not None else 0.0)
            vm = patient_metrics(pva, yva, pred_val)
            metrics["val"] = {"n_patients": vm["n_patients"], "ndcg@5": vm["ndcg@5"], "spearman": vm["spearman"]}
        if args.dump_preds:
            pred_rows.append(pd.DataFrame({
                "fold": f_id, "split": "test", "patient_id": pte, "sample_id": site, "drug_id": drte,
                "y": yte, "pred": pred_test, "train_seed": args.seed,
            }))
            if "val" in preds and len(dva) > 0:
                pred_val = preds["val"] + (drug_mean_vec[dva] if drug_mean_vec is not None else 0.0)
                pred_rows.append(pd.DataFrame({
                    "fold": f_id, "split": "val", "patient_id": pva, "sample_id": siva, "drug_id": drva,
                    "y": yva, "pred": pred_val, "train_seed": args.seed,
                }))
        metrics.update({
            "fold": f_id,
            "n_train_rows": int(len(ytr)),
            "n_test_rows": int(len(yte)),
            "n_features": int(Xtr.shape[1]),
            "n_drugs": int(len(drug_to_i)),
            "gpu": args.gpu,
            "tag": args.tag,
            "feat": args.feat,
            "hidden": args.hidden,
            "depth": args.depth,
            "weight_decay": args.weight_decay,
            "target": args.target,
            "hold_out_inner_val": bool(args.hold_out_inner_val),
            "train_seed": args.seed,
            "inner_val_random_seed": inner_seed,
            "n_train_inner_patients": int(n_ti),
            "n_val_patients": int(len(val_p)),
            "val_patient_ids": val_p,
            "inner_val_skipped": bool(inner_val_skipped),
        })
        results.append(metrics)
        ckpt = out_dir / f"{args.tag}_gpu{args.gpu}_fold{f_id}.pt"
        torch.save({"model": model.state_dict(), "metrics": metrics}, ckpt)
        print(f"FOLD {f_id} DONE {metrics}", flush=True)

    out = {
        "status": "RAN_LOCAL_WSL",
        "env": "drugscreening-gpu",
        "physical_gpu": args.gpu,
        "tag": args.tag,
        "feat": args.feat,
        "hidden": args.hidden,
        "depth": args.depth,
        "weight_decay": args.weight_decay,
        "target": args.target,
        "hold_out_inner_val": bool(args.hold_out_inner_val),
        "train_seed": args.seed,
        "inner_val_random_seed": inner_seed,
        "inner_val_algorithm": "patient-level 4:1 from outer train_patient_ids only; lexicographic sort + numpy.random.RandomState(seed).permutation; last ceil(n/5) patients as val; outer test never used for model selection/early stopping/feature selection/thresholds (split_freeze_checklist_DRAFT 2026-09-18)",
        "folds": fold_ids,
        "data": str(PROC),
        "split": str(SPLIT),
        "what_trained": {
            "model": f"DualMLP feat={args.feat} hidden={args.hidden} depth={args.depth}",
            "label": "drug response (higher = more sensitive)",
            "target": args.target,
            "pathway_note": "pathway_pca is a DRAFT proxy (train-fit PCA on top-2000 genes); not Hallmark scores",
            "train": "patients in fold train_* only"
            + (" minus pinned inner-val patients (hold_out_inner_val)" if args.hold_out_inner_val else ""),
            "test": "patients in fold test_* (unseen)",
        },
        "seconds": time.time() - t0,
        "results": results,
    }
    out_path = out_dir / f"{args.tag}_gpu{args.gpu}_results.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("WROTE", out_path, flush=True)
    if args.dump_preds and pred_rows:
        pred_path = out_dir / f"{args.tag}_gpu{args.gpu}_predictions.parquet"
        pd.concat(pred_rows, ignore_index=True).to_parquet(pred_path, index=False)
        print("WROTE", pred_path, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
