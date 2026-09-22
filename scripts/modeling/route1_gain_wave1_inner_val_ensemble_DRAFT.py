#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-024 V2 inner_val_ensemble: combine head-variant members (trained on inner-train,
predictions dumped by train_stage2_dual_mlp_variants_DRAFT.py with --hold-out-inner-val
--dump-preds) using weights proportional to inner-val patient NDCG@5.

Weights use ONLY pinned inner-val patients; outer test never participates in selection.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/d/Code/DrugScreenLab")
WAVE = ROOT / "artifacts/route1_stage2_gain_wave1_DRAFT/inner_val_ensemble"

MEMBERS = ["ensm_genes_h64_wd02", "ensm_pathway_pca_reg", "ensm_tiny_wd05"]
SEEDS = [20260919, 20260920, 20260921]


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
    return {
        "n_patients": len(nds),
        "ndcg@5": float(np.nanmean(nds)) if nds else float("nan"),
        "spearman": float(np.nanmean(sps)) if sps else float("nan"),
    }


def load_member(tag: str, seed: int) -> pd.DataFrame:
    cands = sorted(WAVE.glob(f"{tag}_seed{seed}_gpu*_predictions.parquet"))
    if len(cands) != 1:
        raise SystemExit(f"expected exactly 1 prediction dump for {tag} seed{seed}, got {cands}")
    return pd.read_parquet(cands[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=str(WAVE))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    combined_rows = []
    t0 = time.time()
    for seed in SEEDS:
        member_preds = {m: load_member(m, seed) for m in MEMBERS}
        for fold in sorted(member_preds[MEMBERS[0]]["fold"].unique()):
            # inner-val NDCG@5 per member -> weights (inner val ONLY)
            val_ndcg = {}
            for m in MEMBERS:
                v = member_preds[m][(member_preds[m]["fold"] == fold) & (member_preds[m]["split"] == "val")]
                val_ndcg[m] = patient_metrics(v)["ndcg@5"]
            total = sum(val_ndcg.values())
            weights = {m: val_ndcg[m] / total for m in MEMBERS}
            # weighted ensemble of test predictions
            key = ["fold", "split", "patient_id", "sample_id", "drug_id", "y"]
            sorted_te = {}
            for m in MEMBERS:
                t = member_preds[m][(member_preds[m]["fold"] == fold) & (member_preds[m]["split"] == "test")]
                sorted_te[m] = t.sort_values(key).reset_index(drop=True)
            ref_keys = sorted_te[MEMBERS[0]][key].reset_index(drop=True)
            for m in MEMBERS[1:]:
                if not ref_keys.equals(sorted_te[m][key].reset_index(drop=True)):
                    raise SystemExit(f"member row mismatch fold{fold} seed{seed} member {m}")
            ens_pred = np.zeros(len(ref_keys), dtype=np.float64)
            for m in MEMBERS:
                ens_pred += weights[m] * sorted_te[m]["pred"].to_numpy()
            te = sorted_te[MEMBERS[0]].copy()
            te["pred"] = ens_pred
            tm = patient_metrics(te)
            rec = {
                "fold": int(fold),
                "train_seed": seed,
                "val_ndcg@5": val_ndcg,
                "weights": weights,
                "n_patients": tm["n_patients"],
                "ndcg@5": tm["ndcg@5"],
                "spearman": tm["spearman"],
                "member_test_ndcg@5": {
                    m: patient_metrics(
                        member_preds[m][(member_preds[m]["fold"] == fold) & (member_preds[m]["split"] == "test")]
                    )["ndcg@5"]
                    for m in MEMBERS
                },
            }
            results.append(rec)
            te["train_seed"] = seed
            combined_rows.append(te)
            print(f"SEED {seed} FOLD {fold} val_ndcg={ {m: round(v,4) for m,v in val_ndcg.items()} } w={ {m: round(v,4) for m,v in weights.items()} } test_ndcg={tm['ndcg@5']:.4f}", flush=True)

    out = {
        "status": "RAN_LOCAL_WSL",
        "env": "drugscreening-gpu",
        "tag": "inner_val_ensemble",
        "members": MEMBERS,
        "seeds": SEEDS,
        "weight_rule": "w_i = inner_val_patient_ndcg5_i / sum_i(inner_val_patient_ndcg5_i); inner val only; outer test never used",
        "members_trained_on": "inner-train (outer train minus pinned inner-val patients)",
        "seconds": time.time() - t0,
        "results": results,
    }
    out_path = out_dir / "inner_val_ensemble_gpu0_results.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("WROTE", out_path, flush=True)
    pred_path = out_dir / "inner_val_ensemble_gpu0_predictions.parquet"
    pd.concat(combined_rows, ignore_index=True).to_parquet(pred_path, index=False)
    print("WROTE", pred_path, flush=True)
    for seed in SEEDS:
        nds = [r["ndcg@5"] for r in results if r["train_seed"] == seed]
        print(f"seed {seed} macro ndcg@5 = {float(np.mean(nds)):.4f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
