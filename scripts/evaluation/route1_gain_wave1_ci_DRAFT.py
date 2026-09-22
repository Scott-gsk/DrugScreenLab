#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-024 wave-1 aggregation: seed-mean metrics, fold-CI and patient-level paired
bootstrap CI (2000 resamples) vs B0, dual-gate judgment.

Pre-registered aggregation (EXP-024, before training):
- config fold value (fold gate) = mean over seeds of that fold's macro NDCG@5 under the
  stage2 raw-row scoring (identical to all prior stage2 rows); B0 fold values = frozen
  stage1 CSV (EXP-020 口径, B0 macro = 0.9089).
- primary patient gate: per-patient NDCG@5 under the stage1 Task-B protocol
  ((patient, drug) dedup by mean; B0 refit on stage1 carve fit-patients) for BOTH the
  config predictions and B0; delta per patient per seed, averaged over seeds;
  2000x patient-unit paired bootstrap.
- secondary sensitivity: fully stage2-protocol patient CI (raw rows; B0 means on all
  outer-train patients) and seed-ensemble (prediction-averaged) CI.
- gates: point estimate > 0 AND fold-CI excludes 0 AND patient-CI excludes 0.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from drug_screen.evaluation.protocols import judge_model_gain, paired_resample_ci
from drug_screen.evaluation.tasks import evaluate_task_b
from drug_screen.transfer.baselines import fit_b0_drug_means, predict_b0

ROOT = Path("/mnt/d/Code/DrugScreenLab")
WAVE = ROOT / "artifacts/route1_stage2_gain_wave1_DRAFT"
PROC = ROOT / "data/processed/transfer/mcrc_v3"
SPLIT = ROOT / "data/splits/kryeziu_patient_outer5fold_FROZEN.json"
B0_FOLD_CSV = ROOT / "data/artifacts/route1_stage1_DRAFT/kryeziu_stage1_baseline_metrics_by_fold_DRAFT.csv"
OUT_CI_CSV = ROOT / "docs/reports/route1_gain_wave1_ci_vs_B0.csv"
FOLD_CSV = ROOT / "docs/reports/stage2_fold_metrics_SUMMARY_DRAFT.csv"
SEEDS = [20260919, 20260920, 20260921]
STAGE1_SEED = 20260918

CONFIGS = {
    "dual_mlp_genes_h64_wd02_3seed": [WAVE / f"multiseed/dual_mlp_genes_h64_wd02_seed{s}_gpu0_predictions.parquet" for s in SEEDS],
    "dual_mlp_pathway_pca_reg_3seed": [WAVE / f"multiseed/dual_mlp_pathway_pca_reg_seed{s}_gpu1_predictions.parquet" for s in SEEDS],
    "b0_residual_genes_h64_wd02": [WAVE / f"b0_residual/b0_residual_genes_h64_wd02_seed{s}_gpu0_predictions.parquet" for s in SEEDS],
    "dual_mlp_pathway_pca_wd02": [WAVE / f"pathway_pca_best_reg/pathway_pca_wd02_seed{s}_gpu1_predictions.parquet" for s in SEEDS],
    "inner_val_ensemble": [WAVE / "inner_val_ensemble/inner_val_ensemble_gpu0_predictions.parquet"],
    "transfer_calib_b0": [WAVE / "transfer_calib_b0/transfer_calib_b0_gpu0_predictions.parquet"],
}


# ---------------- stage2 raw-row scoring (matches train script patient_metrics) ----------------

def spearman_lin(y, p):
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


def ndcg_lin(y, p, k=5):
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


def per_patient_raw(df):
    """stage2 protocol: all rows as-is; patients with >=2 rows."""
    out = {}
    for pid, g in df.groupby("patient_id", sort=True):
        if len(g) < 2:
            continue
        out[pid] = (ndcg_lin(g["y"].to_numpy(), g["pred"].to_numpy(), 5),
                    spearman_lin(g["y"].to_numpy(), g["pred"].to_numpy()))
    return out


def fold_macro_raw(df):
    pp = per_patient_raw(df)
    nds = [v[0] for v in pp.values()]
    sps = [v[1] for v in pp.values()]
    return (float(np.mean(nds)) if nds else float("nan"),
            float(np.mean(sps)) if sps else float("nan"),
            len(nds))


# ---------------- stage1 Task-B scoring (official B0=0.9089 protocol) ----------------

def carve_patients_stage1(train_patients, fold):
    rng = np.random.default_rng(STAGE1_SEED + int(fold))
    pts = sorted(map(str, train_patients))
    if len(pts) < 5:
        return pts, []
    n_val = max(1, int(round(len(pts) * 0.2)))
    n_val = min(n_val, len(pts) - 1)
    order = rng.permutation(len(pts))
    val = sorted(pts[i] for i in order[:n_val])
    fit = sorted(pts[i] for i in order[n_val:])
    return fit, val


def per_patient_taskb(df):
    """stage1 Task-B protocol: dedup (patient, drug) by mean; evaluate_task_b per patient."""
    agg = df.groupby(["patient_id", "drug_id"], as_index=False)[["y", "pred"]].mean()
    out = {}
    for pid, g in agg.groupby("patient_id", sort=True):
        if len(g) < 2:
            continue
        r = evaluate_task_b(patient_id=str(pid), y_true=g["y"].to_numpy(dtype=float),
                            y_score=g["pred"].to_numpy(dtype=float), k=5)
        out[pid] = r.ndcg_at_5
    return out


def load_tables():
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
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
    all_p = set()
    for fold in split["folds"]:
        all_p |= set(map(str, fold.get("train_patient_ids") or fold.get("train_patients") or []))
        all_p |= set(map(str, fold.get("test_patient_ids") or fold.get("test_patients") or []))
    merged = merged[merged["patient_id"].isin(all_p)].copy()
    return split, merged


def load_b0():
    """Reconstruct the official stage1 B0 per-patient (Task-B protocol) AND the stage2
    raw-row variant; verify Task-B fold macros against the frozen CSV."""
    split, merged = load_tables()
    b0_fold_csv = {}
    with B0_FOLD_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model"] == "B0_drug_mean" and row["split"] == "test":
                b0_fold_csv[int(row["fold"])] = float(row["ndcg@5"])
    folds = {}
    for fold in split["folds"]:
        f_id = int(fold.get("fold", fold.get("fold_id")))
        train_p = set(map(str, fold["train_patient_ids"]))
        test_p = set(map(str, fold["test_patient_ids"]))
        tr = merged[merged["patient_id"].isin(train_p)]
        te = merged[merged["patient_id"].isin(test_p)].copy()

        # --- official stage1 B0: fit on stage1-carve fit patients, Task-B dedup scoring ---
        fit_p, _val_p = carve_patients_stage1(train_p, f_id)
        fit_rows = tr[tr["patient_id"].isin(fit_p)]
        b0 = fit_b0_drug_means(fit_rows["drug_id"].tolist(), fit_rows["y"].tolist())
        te_official = te.copy()
        te_official["pred"] = predict_b0(b0, te_official["drug_id"].tolist())
        pat_taskb = per_patient_taskb(te_official)
        fold_taskb = float(np.mean([v for v in pat_taskb.values()])) if pat_taskb else float("nan")
        assert abs(fold_taskb - b0_fold_csv[f_id]) < 1e-9, \
            f"stage1 B0 recompute mismatch fold{f_id}: {fold_taskb} vs {b0_fold_csv[f_id]}"

        # --- stage2 raw-row B0: means on ALL outer-train patients ---
        m_d = tr.groupby("drug_id")["y"].mean().to_dict()
        gm = float(tr["y"].mean())
        te_raw = te.copy()
        te_raw["pred"] = [m_d.get(str(d), gm) for d in te_raw["drug_id"]]
        pat_raw = per_patient_raw(te_raw)

        folds[f_id] = {
            "df": te, "pat_taskb": pat_taskb, "fold_taskb": fold_taskb,
            "pat_raw": pat_raw, "ndcg_raw": fold_macro_raw(te_raw)[0],
        }
    print(f"stage1 B0 (Task-B protocol) verified vs frozen CSV for {len(folds)} folds; "
          f"macro={np.mean([v['fold_taskb'] for v in folds.values()]):.4f}; "
          f"stage2-raw B0 macro={np.mean([v['ndcg_raw'] for v in folds.values()]):.4f}")
    return folds


def keys_match(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Cross-source key alignment: string cols exact, y approx (parquet dumps float32)."""
    for c in ("patient_id", "sample_id", "drug_id"):
        if not a[c].reset_index(drop=True).equals(b[c].reset_index(drop=True)):
            return False
    ya = a["y"].to_numpy(dtype=np.float64)
    yb = b["y"].to_numpy(dtype=np.float64)
    return bool(np.allclose(ya, yb, rtol=1e-6, atol=1e-9))


def main():
    b0 = load_b0()
    folds_sorted = sorted(b0)
    pat_ids = sorted({p for v in b0.values() for p in v["pat_taskb"]})
    b0_taskb = np.array([np.nanmean([b0[f]["pat_taskb"].get(p, np.nan) for f in folds_sorted]) for p in pat_ids])
    b0_raw = np.array([np.nanmean([b0[f]["pat_raw"].get(p, (np.nan, np.nan))[0] for f in folds_sorted]) for p in pat_ids])
    b0_fold_vals = [b0[f]["fold_taskb"] for f in folds_sorted]

    ci_rows = []
    fold_rows = []
    summary = {}
    for name, paths in CONFIGS.items():
        dfs = []
        for p in paths:
            if not p.exists():
                raise SystemExit(f"missing predictions: {p}")
            d = pd.read_parquet(p)
            d = d[d["split"] == "test"].copy()
            d["y"] = d["y"].astype(np.float64)
            dfs.append(d)
        multi = "train_seed" in dfs[0].columns and dfs[0]["train_seed"].notna().any()
        if len(dfs) > 1:
            # one parquet per seed (multiseed configs): each file is one seed
            per_seed = dfs
            seeds = []
            for d in per_seed:
                if "train_seed" in d.columns and d["train_seed"].notna().any():
                    seeds.append(int(d["train_seed"].iloc[0]))
                else:
                    seeds.append(None)
        elif multi:
            # single parquet holding several seeds (inner_val_ensemble combiner output)
            seeds = sorted(dfs[0]["train_seed"].dropna().unique().tolist())
            per_seed = [dfs[0][dfs[0]["train_seed"] == s].copy() for s in seeds]
        else:
            # single deterministic run without seed marks (transfer_calib_b0)
            per_seed = dfs
            seeds = [None]
        key_cols = ["patient_id", "sample_id", "drug_id", "y"]
        ref = per_seed[0].sort_values(key_cols).reset_index(drop=True)
        for d in per_seed[1:]:
            if not ref[key_cols].equals(d.sort_values(key_cols).reset_index(drop=True)[key_cols]):
                raise SystemExit(f"{name}: test keys differ across seeds")
        for f_id in folds_sorted:
            bk = b0[f_id]["df"].sort_values(key_cols).reset_index(drop=True)
            rk = ref[ref["fold"] == f_id].sort_values(key_cols).reset_index(drop=True)
            if len(bk) != len(rk) or not keys_match(bk, rk):
                raise SystemExit(f"{name}: test keys differ from B0 rows in fold {f_id}")

        fold_nd = {f: [] for f in folds_sorted}
        fold_sp = {f: [] for f in folds_sorted}
        seed_macros = []
        pat_raw_by_seed = []
        pat_taskb_by_seed = []
        for d in per_seed:
            macros_nd, macros_sp = [], []
            praw, ptask = {}, {}
            for f_id in folds_sorted:
                g = d[d["fold"] == f_id]
                nd, sp, _n = fold_macro_raw(g)
                fold_nd[f_id].append(nd)
                fold_sp[f_id].append(sp)
                macros_nd.append(nd)
                macros_sp.append(sp)
                for pid, (n_, s_) in per_patient_raw(g).items():
                    praw[pid] = n_
                for pid, n_ in per_patient_taskb(g).items():
                    ptask[pid] = n_
            seed_macros.append((float(np.mean(macros_nd)), float(np.mean(macros_sp))))
            pat_raw_by_seed.append(praw)
            pat_taskb_by_seed.append(ptask)
        seed_macros = np.array(seed_macros)

        # fold gate (pre-registered): seed-mean raw-row fold values vs frozen stage1 B0 CSV
        cfg_fold_nd = [float(np.mean(fold_nd[f])) for f in folds_sorted]
        cfg_fold_sp = [float(np.mean(fold_sp[f])) for f in folds_sorted]
        ci_fold = paired_resample_ci(cfg_fold_nd, b0_fold_vals, n_resamples=2000, unit="fold", random_state=20260918)
        j_fold = judge_model_gain(ci_fold)

        # primary patient gate: Task-B protocol, seed-mean per-patient deltas
        pt = np.array([[p.get(pid, np.nan) for pid in pat_ids] for p in pat_taskb_by_seed])
        prim_taskb = np.nanmean(pt, axis=0)
        ok = ~np.isnan(prim_taskb) & ~np.isnan(b0_taskb)
        ci_pat = paired_resample_ci(prim_taskb[ok], b0_taskb[ok], n_resamples=2000, unit="patient", random_state=20260918)
        j_pat = judge_model_gain(ci_pat)

        # secondary 1: stage2 raw protocol patient CI
        pr = np.array([[p.get(pid, np.nan) for pid in pat_ids] for p in pat_raw_by_seed])
        prim_raw = np.nanmean(pr, axis=0)
        okr = ~np.isnan(prim_raw) & ~np.isnan(b0_raw)
        ci_raw = paired_resample_ci(prim_raw[okr], b0_raw[okr], n_resamples=2000, unit="patient", random_state=20260918)

        # secondary 2: seed-ensemble (prediction-averaged), Task-B protocol
        merged_pred = per_seed[0].sort_values(key_cols).reset_index(drop=True)[key_cols + ["fold"]].copy()
        acc = np.zeros(len(merged_pred))
        for d in per_seed:
            acc += d.sort_values(key_cols).reset_index(drop=True)["pred"].to_numpy()
        merged_pred["pred"] = acc / len(per_seed)
        ens_taskb = {}
        for f_id in folds_sorted:
            for pid, n_ in per_patient_taskb(merged_pred[merged_pred["fold"] == f_id]).items():
                ens_taskb[pid] = n_
        ens_vals = np.array([ens_taskb.get(pid, np.nan) for pid in pat_ids])
        oke = ~np.isnan(ens_vals) & ~np.isnan(b0_taskb)
        ci_ens = paired_resample_ci(ens_vals[oke], b0_taskb[oke], n_resamples=2000, unit="patient", random_state=20260918)

        gate_pass = (ci_pat.estimate > 0) and (not ci_fold.crosses_zero) and (not ci_pat.crosses_zero)
        if gate_pass:
            claim = "候选增益（过双闸门）"
        elif ci_fold.crosses_zero or ci_pat.crosses_zero:
            claim = "证据不足"
        else:
            claim = "弱于对照"
        summary[name] = {
            "n_seeds": len(seeds), "seed_macros": seed_macros,
            "cfg_macro_ndcg": float(np.mean(cfg_fold_nd)),
            "fold_ci": ci_fold, "patient_ci": ci_pat, "patient_ci_raw": ci_raw, "ens_ci": ci_ens,
            "gate_pass": gate_pass, "claim": claim,
        }
        print(f"\n== {name} (seeds={[s if s is not None else 'deterministic' for s in seeds]})")
        print(f"   seed macro NDCG@5(raw): mean={seed_macros[:, 0].mean():.4f} "
              f"std={seed_macros[:, 0].std(ddof=1) if len(seeds) > 1 else 0:.4f} "
              f"per-seed={[round(x, 4) for x in seed_macros[:, 0]]}")
        print(f"   FOLD-GATE : d={ci_fold.estimate:+.4f} [{ci_fold.low:.4f},{ci_fold.high:.4f}] "
              f"cross={ci_fold.crosses_zero} -> {j_fold.claim_level}")
        print(f"   PAT-GATE  : d={ci_pat.estimate:+.4f} [{ci_pat.low:.4f},{ci_pat.high:.4f}] "
              f"cross={ci_pat.crosses_zero} n={int(ok.sum())} -> {j_pat.claim_level}")
        print(f"   sens-raw  : d={ci_raw.estimate:+.4f} [{ci_raw.low:.4f},{ci_raw.high:.4f}] cross={ci_raw.crosses_zero}")
        print(f"   sens-ens  : d={ci_ens.estimate:+.4f} [{ci_ens.low:.4f},{ci_ens.high:.4f}] cross={ci_ens.crosses_zero}")
        print(f"   GATE: {claim}")

        for ci, unit, note in [
            (ci_fold, "fold", "primary gate1: seed-mean raw-row fold values vs frozen stage1 B0 (EXP-020 口径)"),
            (ci_pat, "patient", "primary gate2: Task-B protocol per-patient deltas, seed-mean"),
            (ci_raw, "patient", "sensitivity: stage2 raw-row protocol"),
            (ci_ens, "patient", "sensitivity: seed-ensemble, Task-B protocol"),
        ]:
            jj = judge_model_gain(ci)
            ci_rows.append({
                "model": name, "vs": "B0_drug_mean", "unit": unit,
                "n_units": len(folds_sorted) if unit == "fold" else int(ok.sum()),
                "delta": ci.estimate, "ci95_low": ci.low, "ci95_high": ci.high,
                "crosses_zero": ci.crosses_zero, "claim_level": jj.claim_level, "note": note,
            })
        for i, f_id in enumerate(folds_sorted):
            fold_rows.append({
                "fold": f_id, "model": name, "n_test_patients": len(b0[f_id]["pat_taskb"]),
                "ndcg@5": cfg_fold_nd[i], "spearman_B": cfg_fold_sp[i],
                "top5_overlap": np.nan, "spearman_A": np.nan, "n_drugs_A": np.nan,
                "ndcg@5_std": np.std(fold_nd[f_id], ddof=1) if len(seeds) > 1 else np.nan,
                "spearman_B_std": np.std(fold_sp[f_id], ddof=1) if len(seeds) > 1 else np.nan,
            })
        fold_rows.append({
            "fold": "macro", "model": name, "n_test_patients": np.nan,
            "ndcg@5": float(np.mean(cfg_fold_nd)), "spearman_B": float(np.mean(cfg_fold_sp)),
            "top5_overlap": np.nan, "spearman_A": np.nan, "n_drugs_A": np.nan,
            "ndcg@5_std": float(np.std(cfg_fold_nd, ddof=1)), "spearman_B_std": float(np.std(cfg_fold_sp, ddof=1)),
        })

    with OUT_CI_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "vs", "unit", "n_units", "delta", "ci95_low", "ci95_high",
                                          "crosses_zero", "claim_level", "note"])
        w.writeheader()
        for r in ci_rows:
            w.writerow(r)
    print(f"\nwrote {OUT_CI_CSV}")

    old = pd.read_csv(FOLD_CSV)
    new = pd.DataFrame(fold_rows)
    pd.concat([old, new], ignore_index=True).to_csv(FOLD_CSV, index=False)
    print(f"appended {len(fold_rows)} rows -> {FOLD_CSV}")

    summary_path = WAVE / "wave1_dual_gate_summary.json"
    summary_path.write_text(json.dumps({
        k: {
            "n_seeds": v["n_seeds"],
            "seed_macro_ndcg_mean": float(v["seed_macros"][:, 0].mean()),
            "seed_macro_ndcg_std": float(v["seed_macros"][:, 0].std(ddof=1)) if v["n_seeds"] > 1 else 0.0,
            "seed_macro_ndcg_per_seed": [round(float(x), 6) for x in v["seed_macros"][:, 0]],
            "seed_macro_spearman_mean": float(v["seed_macros"][:, 1].mean()),
            "cfg_macro_ndcg": v["cfg_macro_ndcg"],
            "fold_ci": {"delta": v["fold_ci"].estimate, "low": v["fold_ci"].low, "high": v["fold_ci"].high, "crosses_zero": v["fold_ci"].crosses_zero},
            "patient_ci_taskb": {"delta": v["patient_ci"].estimate, "low": v["patient_ci"].low, "high": v["patient_ci"].high, "crosses_zero": v["patient_ci"].crosses_zero},
            "patient_ci_stage2raw": {"delta": v["patient_ci_raw"].estimate, "low": v["patient_ci_raw"].low, "high": v["patient_ci_raw"].high, "crosses_zero": v["patient_ci_raw"].crosses_zero},
            "seed_ensemble_ci_taskb": {"delta": v["ens_ci"].estimate, "low": v["ens_ci"].low, "high": v["ens_ci"].high, "crosses_zero": v["ens_ci"].crosses_zero},
            "gate_pass": v["gate_pass"],
            "claim": v["claim"],
        } for k, v in summary.items()
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
