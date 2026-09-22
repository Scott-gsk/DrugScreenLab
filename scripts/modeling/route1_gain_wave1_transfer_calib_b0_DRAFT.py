#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-024 V3 transfer_calib_b0: per-drug mean/scale calibration of the GDSC source
prior blended onto B0 (target per-drug train means). Deterministic (no SGD).

Protocol (registered in experiments/records/EXP-024.md before training):
- source: exp014 gdsc_v3 pairs (response=1_minus_auc, higher_is_sensitive=True)
- intersection drugs (InChIKey ids) between GDSC and the Kryeziu merged table
- z_s(d): source per-drug mean standardized across intersection drugs
- per fold: OLS a,b fitting per-drug train means m_d ~ a + b*z_s(d) on intersection
  drugs only (train patients only); prior p_s(d) = a + b*z_s(d)
- blend pred(d) = w*m_d + (1-w)*p_s(d) for intersection drugs, m_d otherwise;
  w in {0,0.25,0.5,0.75,1.0} selected on pinned inner-val patient NDCG@5
  (m_d, a, b refit on inner-train only during selection; full train for test)
- tie-break: larger w (closer to B0)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/d/Code/DrugScreenLab")
PROC = ROOT / "data/processed/transfer/mcrc_v3"
SPLIT = ROOT / "data/splits/kryeziu_patient_outer5fold_FROZEN.json"
GDSC_PAIRS = ROOT / "data/processed/exp014/gdsc_v3/pairs.parquet"
OUT = ROOT / "artifacts/route1_stage2_gain_wave1_DRAFT/transfer_calib_b0"
W_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def inner_val_patients(train_p, seed):
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


def patient_metrics(df):
    nds, sps = [], []
    for pid, g in df.groupby("patient_id", sort=True):
        if len(g) < 2:
            continue
        nds.append(ndcg_at_k(g["y"].to_numpy(), g["pred"].to_numpy(), 5))
        sps.append(spearman(g["y"].to_numpy(), g["pred"].to_numpy()))
    return {"n_patients": len(nds), "ndcg@5": float(np.nanmean(nds)) if nds else float("nan"),
            "spearman": float(np.nanmean(sps)) if sps else float("nan")}


def drug_means(rows: pd.DataFrame) -> tuple[dict, float]:
    g = rows.groupby("drug_id")["y"].mean()
    gm = float(rows["y"].mean())
    return {str(k): float(v) for k, v in g.items()}, gm


def fit_affine(m_d: dict, z_s: dict):
    """OLS m ~ a + b*z over intersection drugs present in m_d."""
    ds = [d for d in m_d if d in z_s]
    if len(ds) < 3:
        gm = float(np.mean(list(m_d.values()))) if m_d else 0.0
        return gm, 0.0
    x = np.array([z_s[d] for d in ds], float)
    y = np.array([m_d[d] for d in ds], float)
    b = float(np.cov(x, y, ddof=0)[0, 1] / (np.var(x) + 1e-12))
    a = float(np.mean(y) - b * np.mean(x))
    return a, b


def blend_pred(rows: pd.DataFrame, m_d: dict, gm: float, a: float, b: float, z_s: dict, w: float):
    def one(d):
        d = str(d)
        base = m_d.get(d, gm)
        if d in z_s:
            return w * base + (1.0 - w) * (a + b * z_s[d])
        return base
    return np.array([one(d) for d in rows["drug_id"]], float)


def main():
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    inner_seed = int(split.get("random_seed") or 20260918)
    samples = pd.read_parquet(PROC / "samples.parquet")
    pairs = pd.read_parquet(PROC / "pairs.parquet")
    samples = samples.copy()
    samples["sample_id"] = samples["sample_id"].astype(str)
    samples["patient_id"] = samples["patient_id"].astype(str)
    pairs = pairs.copy()
    pairs["sample_id"] = pairs["sample_id"].astype(str)
    pairs["drug_id"] = pairs["drug_id"].astype(str)
    merged = pairs.merge(samples[["sample_id", "patient_id"]], on="sample_id", how="inner").dropna(subset=["response"])
    merged = merged.rename(columns={"response": "y"})
    # same cohort filter as train_stage2_dual_mlp_variants_DRAFT.load_tables: only patients in the split
    all_p = set()
    for fold in split["folds"]:
        all_p |= set(map(str, fold.get("train_patient_ids") or fold.get("train_patients") or []))
        all_p |= set(map(str, fold.get("test_patient_ids") or fold.get("test_patients") or []))
    merged = merged[merged["patient_id"].isin(all_p)].copy()

    gdsc = pd.read_parquet(GDSC_PAIRS)
    gdsc = gdsc.copy()
    gdsc["drug_id"] = gdsc["drug_id"].astype(str)
    target_drugs = set(merged["drug_id"].unique())
    inter = sorted(target_drugs & set(gdsc["drug_id"].unique()))
    src_mean = gdsc[gdsc["drug_id"].isin(inter)].groupby("drug_id")["response"].mean()
    mu_s, sd_s = float(src_mean.mean()), float(src_mean.std(ddof=0))
    z_s = {d: (float(src_mean[d]) - mu_s) / (sd_s + 1e-12) for d in inter}
    print(f"INTERSECTION n_drugs={len(inter)} source_mean_mu={mu_s:.4f} sd={sd_s:.4f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    pred_rows = []
    t0 = time.time()
    for fold in split["folds"]:
        f_id = int(fold.get("fold", fold.get("fold_id")))
        train_p = set(map(str, fold["train_patient_ids"]))
        test_p = set(map(str, fold["test_patient_ids"]))
        val_p, _ = inner_val_patients(train_p, inner_seed)
        val_set = set(val_p)
        it = merged[merged["patient_id"].isin(train_p - val_set)]
        va = merged[merged["patient_id"].isin(val_set)]
        tr = merged[merged["patient_id"].isin(train_p)]
        te = merged[merged["patient_id"].isin(test_p)].copy()

        # --- w selection on inner val (inner-train fits only) ---
        m_it, gm_it = drug_means(it)
        a_it, b_it = fit_affine(m_it, z_s)
        val_scores = {}
        for w in W_GRID:
            v = va.copy()
            v["pred"] = blend_pred(va, m_it, gm_it, a_it, b_it, z_s, w)
            val_scores[w] = patient_metrics(v)["ndcg@5"]
        w_star = max(W_GRID, key=lambda w: (round(val_scores[w], 12), w))  # tie-break: larger w
        print(f"FOLD {f_id} val_ndcg_by_w={ {w: round(s,4) for w,s in val_scores.items()} } w*={w_star}", flush=True)

        # --- final test predictions (full-train fits) ---
        m_tr, gm_tr = drug_means(tr)
        a_tr, b_tr = fit_affine(m_tr, z_s)
        te["pred"] = blend_pred(te, m_tr, gm_tr, a_tr, b_tr, z_s, w_star)
        tm = patient_metrics(te)
        # B0 reference on the same test rows (full-train means)
        te_b0 = te.copy()
        te_b0["pred"] = [m_tr.get(str(d), gm_tr) for d in te_b0["drug_id"]]
        b0m = patient_metrics(te_b0)
        rec = {
            "fold": f_id,
            "w_star": w_star,
            "val_ndcg@5_by_w": {str(w): val_scores[w] for w in W_GRID},
            "n_intersection_drugs": len(inter),
            "a_full_train": a_tr, "b_full_train": b_tr,
            "ndcg@5": tm["ndcg@5"], "spearman": tm["spearman"], "n_patients": tm["n_patients"],
            "b0_ndcg@5_same_rows": b0m["ndcg@5"],
            "n_train_rows": int(len(tr)), "n_test_rows": int(len(te)),
        }
        results.append(rec)
        keep = te[["patient_id", "sample_id", "drug_id", "y", "pred"]].copy()
        keep["fold"] = f_id
        keep["split"] = "test"
        keep["train_seed"] = None  # deterministic
        pred_rows.append(keep)
        print(f"FOLD {f_id} DONE ndcg={tm['ndcg@5']:.4f} (b0 same rows {b0m['ndcg@5']:.4f})", flush=True)

    out = {
        "status": "RAN_LOCAL_WSL",
        "env": "drugscreening-gpu",
        "tag": "transfer_calib_b0",
        "deterministic": True,
        "seed_invariance_note": "no SGD; output independent of training seed; single run registered (EXP-024 V3)",
        "source": str(GDSC_PAIRS.relative_to(ROOT)),
        "split": str(SPLIT.relative_to(ROOT)),
        "intersection_drugs": inter,
        "z_source": z_s,
        "weight_grid": W_GRID,
        "weight_selection": "inner-val patient NDCG@5 only; tie-break larger w",
        "seconds": time.time() - t0,
        "results": results,
    }
    out_path = OUT / "transfer_calib_b0_gpu0_results.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("WROTE", out_path, flush=True)
    pred_path = OUT / "transfer_calib_b0_gpu0_predictions.parquet"
    pd.concat(pred_rows, ignore_index=True).to_parquet(pred_path, index=False)
    print("WROTE", pred_path, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
