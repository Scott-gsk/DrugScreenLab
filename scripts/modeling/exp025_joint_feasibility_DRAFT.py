#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-025 joint feasibility slice (spec Task 6): three-way comparison.

Pre-registered design (EXP-025, pinned BEFORE any training in this script):
1. Direct arm: B0 (drug-mean) + pathway_pca_reg (strongest EXP-024 wave-1 config,
   3 seeds). Inner-val OOF predictions reused from EXP-024 V2 ensemble members
   (ensm_pathway_pca_reg, hold_out_inner_val=True, identical hyperparams,
   trained on inner-train patients only); outer-test predictions reused from
   EXP-024 multiseed runs. B0 stacking features refit per fold: inner-train
   (val rows) / outer-train (test rows), (patient,drug)-dedup drug means.
2. Perturbation arm (zero-shot, no Kryzei training, no y leakage):
   LINCS drug-conditioned Ridge (ctrl z-scored exact978 restricted to
   landmark∩Kryzei genes + Morgan 1024 -> delta 978, alpha=1.0, 80/20
   pair-random split seed=20260918; the 80% model is applied to Kryzei; the
   20% is a quality report only, never used for hyperparameter or feature
   selection). Kryzei sample-level baseline expression projected on the same
   landmark subset; drug fingerprints via InChIKey bridge to the xpert
   registry (Morgan radius=2 nBits=1024, verified 38/38 identical) and
   RDKit-computed for the 10 unmatched drugs. Derived features (pinned):
   - perturb_magnitude = L2 norm of predicted delta
   - reversal_score = -cosine(pred delta, sample-difference signature);
     signature = sample baseline - cohort mean baseline (all 92 samples),
     landmark z-space, restricted to the same landmark∩Kryzei genes. This is
     a SAMPLE-DIFFERENCE feature, NOT a tumor-vs-normal disease reversal
     (方案 §六.2, no normal reference in Kryzei).
   - pathway_delta: mean predicted delta per Hallmark pathway over
     landmark∩Kryzei∩pathway genes; top-8 pathways selected by gene coverage
     (|pathway ∩ landmark ∩ Kryzei genes|, descending; y-free rule).
   - reliability proxy: coverage flag (drug pert_id present in LINCS 80%
     train pairs) + magnitude already in inputs.
3. Joint model: second-layer Ridge (alpha=1.0, StandardScaler on inner-val
   rows) trained ONLY on inner-val OOF rows of each outer fold (8/9 patients);
   outer test predictions come from the outer-train-fitted first-layer models.
   Inputs (14): [pathway_pca_reg OOF, B0 OOF, magnitude, reversal, 8 pathway
   deltas, coverage]. No outer-test contact during training/selection.
4. Three-way comparison (Task-B protocol: (patient,drug) dedup by mean;
   per-patient NDCG@5 + Spearman; patient-equal-weight and fold-equal-weight
   macros): direct (B0 official carve-fit protocol; pathway_pca_reg seed-mean),
   perturbation ranking (reversal_score only), joint.
   Primary endpoint: patient-level paired bootstrap (2000x, seed 20260918)
   95% CI of dNDCG@5 (joint - pathway_pca_reg).
"""
from __future__ import annotations

import csv
import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from drug_screen.data.measured_perturbation_loader_DRAFT import (
    MeasuredPerturbationPaths,
    propose_lincs_control_pairs,
)
from drug_screen.evaluation.protocols import paired_resample_ci
from drug_screen.evaluation.tasks import evaluate_task_b
from drug_screen.transfer.baselines import fit_b0_drug_means, predict_b0

ROOT = Path("/mnt/d/Code/DrugScreenLab")
PROC = ROOT / "data/processed/transfer/mcrc_v3"
SPLIT = ROOT / "data/splits/kryeziu_patient_outer5fold_FROZEN.json"
WAVE = ROOT / "artifacts/route1_stage2_gain_wave1_DRAFT"
B0_FOLD_CSV = ROOT / "data/artifacts/route1_stage1_DRAFT/kryeziu_stage1_baseline_metrics_by_fold_DRAFT.csv"
OUT_DIR = ROOT / "artifacts/joint_feasibility_EXP025_DRAFT"
HALLMARK_GMT = ROOT / "data/raw/exp013/msigdb/h.all.v2023.2.Hs.symbols.gmt"
XPERT_REG = ROOT / "mvp/foundation/xpert/DRUG_REGISTRY.json"
XPERT_MORGAN = ROOT / "data/external/xpert_source/processed_data/all_drugs_idx2morgan.npy"
LINCS_LINEAR_CKPT = ROOT / "artifacts/experiments/route2_lincs_linear_gpu_DRAFT/linear_ctrl_to_delta.pt"

SEEDS = [20260919, 20260920, 20260921]
STAGE1_SEED = 20260918
BOOT_RS = 20260918
LINCS_SPLIT_SEED = 20260918
N_PAIRS = 12000          # identical to route2 linear DRAFT script (mu/sd anchor)
MAX_SCAN_ROWS = 400_000
LINCS_TEST_FRAC = 0.2
RIDGE_ALPHA = 1.0        # pinned for BOTH layers; no search anywhere
TOPK_PATHWAYS = 8
MORGAN_RADIUS, MORGAN_BITS = 2, 1024


# ---------------------------------------------------------------- helpers

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
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(y[order] * disc))
    idcg = float(np.sum(y[np.argsort(-y)[:k]] * disc))
    return dcg / idcg if idcg > 0 else float("nan")


def taskb_patient_scores(df):
    """Task-B protocol: dedup (patient,drug) by mean; per-patient NDCG@5 + Spearman."""
    agg = df.groupby(["patient_id", "drug_id"], as_index=False)[["y", "pred"]].mean()
    out = {}
    for pid, g in agg.groupby("patient_id", sort=True):
        if len(g) < 2:
            continue
        r = evaluate_task_b(patient_id=str(pid), y_true=g["y"].to_numpy(float),
                            y_score=g["pred"].to_numpy(float), k=5)
        out[pid] = (r.ndcg_at_5, r.spearman)
    return out


def fold_macro_taskb(df):
    pp = taskb_patient_scores(df)
    nds = [v[0] for v in pp.values()]
    sps = [v[1] for v in pp.values()]
    return (float(np.nanmean(nds)), float(np.nanmean(sps)), len(nds))


def rawrow_patient_scores(df):
    out = {}
    for pid, g in df.groupby("patient_id", sort=True):
        if len(g) < 2:
            continue
        out[pid] = (ndcg_lin(g["y"].to_numpy(), g["pred"].to_numpy(), 5),
                    spearman_lin(g["y"].to_numpy(), g["pred"].to_numpy()))
    return out


def build_inst_index(path: Path, expected_n: int) -> dict[str, int]:
    index = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh, delimiter="\t")):
            index[row["inst_id"]] = i
    if len(index) != expected_n:
        raise RuntimeError(f"inst_info={len(index)} != cache rows={expected_n}")
    return index


def load_morgan_maps(root: Path):
    reg = json.loads((root / "mvp/foundation/xpert/DRUG_REGISTRY.json").read_text(encoding="utf-8"))
    morgan_obj = np.load(root / "data/external/xpert_source/processed_data/all_drugs_idx2morgan.npy",
                         allow_pickle=True).item()
    out = {}
    for d in reg["drugs"]:
        pid, idx = d.get("pert_id"), d.get("pert_idx")
        if pid is None or idx is None:
            continue
        vec = morgan_obj.get(int(idx))
        if vec is not None:
            out[str(pid)] = np.asarray(vec, dtype=np.float32)
    return out


def rdkit_morgan(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"unparseable SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)
    arr = np.zeros(MORGAN_BITS, dtype=np.int64)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return num / den


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


def b0_dedup_drug_means(rows: pd.DataFrame) -> dict[str, float]:
    """B0 stacking feature fit: (patient,drug)-dedup mean rows -> per-drug mean."""
    agg = rows.groupby(["patient_id", "drug_id"], as_index=False)["y"].mean()
    m = agg.groupby("drug_id")["y"].mean().to_dict()
    gm = float(agg["y"].mean())
    return {"means": m, "global": gm}


def b0_predict(b0: dict, drug_ids) -> np.ndarray:
    return np.array([b0["means"].get(str(d), b0["global"]) for d in drug_ids], dtype=np.float64)


# ---------------------------------------------------------------- main

def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # read the frozen B0 fold CSV early (DrvFs on /mnt/d occasionally returns
    # transient ENOENT after heavy mmap I/O; retry to ride it out)
    b0_fold_csv = {}
    for attempt in range(3):
        try:
            with B0_FOLD_CSV.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["model"] == "B0_drug_mean" and row["split"] == "test":
                        b0_fold_csv[int(row["fold"])] = float(row["ndcg@5"])
            break
        except FileNotFoundError:
            if attempt == 2:
                raise
            time.sleep(3.0)
    assert len(b0_fold_csv) == 5, f"B0 frozen CSV folds = {sorted(b0_fold_csv)}"
    run_log = {"status": "RAN_LOCAL_WSL", "env": "drugscreening-gpu",
               "pinned_before_training": {
                   "lincs_split": f"pair_random 80/20, seed={LINCS_SPLIT_SEED}",
                   "lincs_ridge_alpha": RIDGE_ALPHA,
                   "joint_ridge_alpha": RIDGE_ALPHA,
                   "morgan": f"rdkit radius={MORGAN_RADIUS} nBits={MORGAN_BITS} (verified vs xpert 38/38)",
                   "topk_pathways": TOPK_PATHWAYS,
                   "topk_rule": "|pathway ∩ landmark ∩ kryzei-genes| descending (y-free)",
                   "cohort_mean": "all 92 Kryzei samples, landmark z-space (sample-difference feature, not tumor-vs-normal)",
                   "joint_inputs": ["ppr_oof", "b0_oof", "perturb_magnitude", "reversal_score",
                                    "pathway_delta_top8", "lincs_coverage_flag"],
                   "seeds": SEEDS, "bootstrap": f"2000x patient paired, random_state={BOOT_RS}"},
               }

    # ============ landmark genes (order per gene_info row order) ==================
    lm_syms = []
    with gzip.open(ROOT / "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_gene_info.txt.gz",
                   "rt", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row.get("pr_is_lm") == "1":
                lm_syms.append(row["pr_gene_symbol"])
    assert len(lm_syms) == 978, f"landmark genes = {len(lm_syms)}"

    # Kryzei gene contract -> landmark subset shared by both spaces (pinned input space)
    npz = np.load(PROC / "indexed.npz", allow_pickle=True)
    kry_expr_full = np.asarray(npz["expression"], dtype=np.float32)
    kry_genes = [str(g) for g in np.asarray(npz["ordered_genes"]).tolist()]
    kry_sid = [str(s) for s in np.asarray(npz["sample_id"]).tolist()]
    gpos = {g: i for i, g in enumerate(kry_genes)}
    keep = [i for i, g in enumerate(lm_syms) if g in gpos]      # positions within 978
    lm_keep_syms = [lm_syms[i] for i in keep]
    n_hit = len(keep)
    print(f"[exp025] landmark∩kryzei genes = {n_hit}/978 (input space of the Ridge)", flush=True)
    run_log["landmark_kryzei_overlap"] = n_hit

    paths = MeasuredPerturbationPaths.from_root(ROOT)
    manifest = json.loads((paths.lincs_exact978 / "asset_manifest.json").read_text(encoding="utf-8"))
    n_rows, n_genes = manifest["cache_shape"]
    Xmm = np.load(paths.lincs_exact978 / "exact978_cache.npy", mmap_mode="r")
    inst_index = build_inst_index(paths.lincs_raw / "GSE92742_Broad_LINCS_inst_info.txt.gz", n_rows)

    # ============ Part 0: landmark order anchor (vs frozen ctrl-only model) ======
    print("[exp025] Part 0: landmark order anchor vs route2 linear checkpoint", flush=True)
    anchor_pairs = propose_lincs_control_pairs(ROOT, perturbation_class="chemical",
                                               limit_treatments=N_PAIRS, max_scan_rows=MAX_SCAN_ROWS)
    anchor_usable = [p for p in anchor_pairs if p.treatment_inst_id in inst_index and p.control_inst_id in inst_index]
    rng7 = np.random.default_rng(7)
    compounds7 = sorted({p.pert_id for p in anchor_usable})
    rng7.shuffle(compounds7)
    n_test7 = max(1, int(round(len(compounds7) * LINCS_TEST_FRAC)))
    test_c7 = set(compounds7[:n_test7])
    anchor_train = [p for p in anchor_usable if p.pert_id not in test_c7]
    mu_anchor = np.zeros((n_genes,), dtype=np.float64)
    for p in anchor_train:
        mu_anchor += np.asarray(Xmm[inst_index[p.control_inst_id]], dtype=np.float64)
    mu_anchor /= len(anchor_train)
    sq = np.zeros((n_genes,), dtype=np.float64)
    for p in anchor_train:
        c = np.asarray(Xmm[inst_index[p.control_inst_id]], dtype=np.float64)
        sq += (c - mu_anchor) ** 2
    sd_anchor = np.sqrt(sq / len(anchor_train))
    ck = torch.load(LINCS_LINEAR_CKPT, map_location="cpu", weights_only=False)
    d_mu = float(np.abs(mu_anchor[None, :] - np.asarray(ck["mu"], dtype=np.float64)).max())
    d_sd = float(np.abs(sd_anchor[None, :] - np.asarray(ck["sd"], dtype=np.float64)).max())
    print(f"[exp025] anchor: n_train={len(anchor_train)} max|mu-mu_ckpt|={d_mu:.2e} max|sd-sd_ckpt|={d_sd:.2e}", flush=True)
    assert d_mu < 1e-3 and d_sd < 1e-3, "landmark gene order anchor FAILED vs frozen ctrl-only model"
    run_log["landmark_anchor"] = {"n_train": len(anchor_train), "max_abs_mu_diff": d_mu,
                                  "max_abs_sd_diff": d_sd, "verdict": "PASS (gene order consistent)"}

    # ============ Part 1: LINCS drug-conditioned Ridge (perturbation arm) =======
    print("[exp025] Part 1: train LINCS Ridge ctrl+Morgan -> delta (input restricted to landmark∩Kryzei)", flush=True)
    morgan_by_pid = load_morgan_maps(ROOT)
    usable = anchor_usable  # same pair proposal as the anchor/linear script
    covered = [p for p in usable if p.pert_id in morgan_by_pid]
    print(f"[exp025] usable={len(usable)} with_morgan={len(covered)}", flush=True)
    rng = np.random.default_rng(LINCS_SPLIT_SEED)
    idx = rng.permutation(len(covered))
    cut = max(1, int(len(covered) * (1 - LINCS_TEST_FRAC)))
    train_pairs = [covered[i] for i in idx[:cut]]
    holdout_pairs = [covered[i] for i in idx[cut:]]

    def pack(plist):
        ctl, delta, feats = [], [], []
        for p in plist:
            c_full = np.asarray(Xmm[inst_index[p.control_inst_id]], dtype=np.float32)
            t_full = np.asarray(Xmm[inst_index[p.treatment_inst_id]], dtype=np.float32)
            ctl.append(c_full[keep])        # model INPUT: landmark∩Kryzei subspace (n_hit)
            delta.append(t_full - c_full)   # supervision: full 978 output space
            feats.append(morgan_by_pid[p.pert_id])
        return np.stack(ctl), np.stack(delta), np.stack(feats).astype(np.float32)

    ctl_tr, y_tr, f_tr = pack(train_pairs)
    ctl_ho, y_ho, f_ho = pack(holdout_pairs)
    mu_c, sd_c = ctl_tr.mean(0, keepdims=True), ctl_tr.std(0, keepdims=True) + 1e-6
    mu_f, sd_f = f_tr.mean(0, keepdims=True), f_tr.std(0, keepdims=True) + 1e-6
    zc_tr, zf_tr = ((ctl_tr - mu_c) / sd_c).astype(np.float32), ((f_tr - mu_f) / sd_f).astype(np.float32)
    zc_ho, zf_ho = ((ctl_ho - mu_c) / sd_c).astype(np.float32), ((f_ho - mu_f) / sd_f).astype(np.float32)

    def lin_scores(pred, true, name):
        cos = cosine_rows(pred.astype(np.float64), true.astype(np.float64))
        return {"name": name, "n": int(pred.shape[0]), "mean_cosine": float(cos.mean())}

    holdout_scores = [lin_scores(np.zeros_like(y_ho), y_ho, "null_zero"),
                      lin_scores(np.repeat(y_tr.mean(0, keepdims=True), len(y_ho), 0), y_ho, "global_mean_delta")]
    ctrl_only = Ridge(alpha=RIDGE_ALPHA).fit(zc_tr, y_tr)
    holdout_scores.append(lin_scores(ctrl_only.predict(zc_ho), y_ho, "ctrl_only_ridge"))
    morgan_only = Ridge(alpha=RIDGE_ALPHA).fit(zf_tr, y_tr)
    holdout_scores.append(lin_scores(morgan_only.predict(zf_ho), y_ho, "morgan_only_ridge"))
    drug_cond = Ridge(alpha=RIDGE_ALPHA).fit(np.hstack([zc_tr, zf_tr]), y_tr)
    holdout_scores.append(lin_scores(drug_cond.predict(np.hstack([zc_ho, zf_ho])), y_ho, "ctrl_plus_morgan_ridge"))
    for s in holdout_scores:
        print(f"[exp025]   {s['name']}: n={s['n']} mean_cosine={s['mean_cosine']:.4f}", flush=True)
    run_log["lincs_holdout"] = {"n_train": len(train_pairs), "n_holdout": len(holdout_pairs),
                                "n_train_compounds": len({p.pert_id for p in train_pairs}), "scores": holdout_scores}
    np.savez(OUT_DIR / "lincs_ridge_drugcond.npz",
             coef=drug_cond.coef_.astype(np.float32), intercept=drug_cond.intercept_.astype(np.float32),
             ctrl_mu=mu_c.astype(np.float32), ctrl_sd=sd_c.astype(np.float32),
             fp_mu=mu_f.astype(np.float32), fp_sd=sd_f.astype(np.float32),
             landmark_input_symbols=np.array(lm_keep_syms),
             landmark_output_symbols=np.array(lm_syms))
    train_pert_ids = set(p.pert_id for p in train_pairs)

    # ============ Part 2: zero-shot Kryzei application ===========================
    print("[exp025] Part 2: zero-shot Kryzei perturbation features", flush=True)
    expr_lm_raw = kry_expr_full[:, [gpos[g] for g in lm_keep_syms]]   # 92 x n_hit (raw log space)
    ctrl_z_k = ((expr_lm_raw - mu_c) / sd_c).astype(np.float32)       # same z-space as LINCS train

    ident = json.loads((ROOT / "data/processed/identity/kryeziu_drug_identity_48.json").read_text(encoding="utf-8"))
    kry_drugs = ident["drugs"]
    xreg = json.loads(XPERT_REG.read_text(encoding="utf-8"))["drugs"]
    ik2pid = {d["inchi_key"]: d["pert_id"] for d in xreg if d.get("inchi_key")}
    drug_fps, drug_bridge, coverage = {}, {}, {}
    n_bridge = n_rdkit = 0
    for d in kry_drugs:
        ik = d["drug_id"]
        pid = ik2pid.get(ik)
        if pid is not None and pid in morgan_by_pid:
            drug_fps[ik] = morgan_by_pid[pid]
            drug_bridge[ik] = pid
            n_bridge += 1
            coverage[ik] = 1 if pid in train_pert_ids else 0
        else:
            drug_fps[ik] = rdkit_morgan(d["smiles"])
            drug_bridge[ik] = None
            n_rdkit += 1
            coverage[ik] = 0
    assert len(drug_fps) == 48
    print(f"[exp025] drug bridge: registry={n_bridge} rdkit={n_rdkit} "
          f"lincs-covered={sum(coverage.values())}/48", flush=True)
    run_log["drug_bridge"] = {"registry_inchikey": n_bridge, "rdkit_from_smiles": n_rdkit,
                              "lincs_train_covered": int(sum(coverage.values())),
                              "not_covered_names": [d["original_name"] for d in kry_drugs if coverage[d["drug_id"]] == 0]}

    drug_ids = [d["drug_id"] for d in kry_drugs]
    F_z = np.stack([(drug_fps[k] - mu_f[0]) / sd_f[0] for k in drug_ids]).astype(np.float32)  # 48 x 1024
    W = drug_cond.coef_.astype(np.float32)          # 978_out x (n_hit + 1024)
    Wc, Wf = W[:, :n_hit], W[:, n_hit:]
    b = drug_cond.intercept_.astype(np.float32)
    pred_delta = b[None, None, :] + ctrl_z_k[:, None, :] @ Wc.T + F_z[None, :, :] @ Wf.T  # 92 x 48 x 978
    magnitude = np.linalg.norm(pred_delta, axis=2)                                   # 92 x 48

    cohort_mean_z = ctrl_z_k.mean(axis=0, keepdims=True)                             # 1 x n_hit
    sample_diff = ctrl_z_k - cohort_mean_z                                           # 92 x n_hit
    # cosine in the shared landmark∩Kryzei subspace (pinned)
    pd_sig = pred_delta[:, :, keep]
    num = np.einsum("sdk,sk->sd", pd_sig.astype(np.float64), sample_diff.astype(np.float64))
    den = (np.linalg.norm(pd_sig, axis=2) * np.linalg.norm(sample_diff, axis=1)[:, None]) + 1e-12
    reversal = -(num / den)                                                          # 92 x 48

    # Hallmark pathway deltas over landmark∩Kryzei genes
    hallmark = {}
    with open(HALLMARK_GMT, "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                hallmark[parts[0]] = [g for g in parts[2:] if g]
    hit_set = set(lm_keep_syms)
    pw_cov = {p: len([g for g in gs if g in hit_set]) for p, gs in hallmark.items()}
    top_pw = sorted(hallmark, key=lambda p: (-pw_cov[p], p))[:TOPK_PATHWAYS]
    run_log["pathway_coverage"] = {p: pw_cov[p] for p in top_pw}
    lm_out_pos = {g: i for i, g in enumerate(lm_syms)}
    pw_delta = {}
    for p in top_pw:
        cols = [lm_out_pos[g] for g in hallmark[p] if g in lm_out_pos]
        pw_delta[p] = pred_delta[:, :, cols].mean(axis=2)   # 92 x 48
    print(f"[exp025] top-{TOPK_PATHWAYS} pathway coverage: {[(p, pw_cov[p]) for p in top_pw[:4]]} ...", flush=True)

    feat_rows = []
    for si, sid in enumerate(kry_sid):
        for di, did in enumerate(drug_ids):
            feat_rows.append({
                "sample_id": sid, "drug_id": did,
                "perturb_magnitude": float(magnitude[si, di]),
                "reversal_score": float(reversal[si, di]),
                "lincs_coverage": int(coverage[did]),
                **{f"pw_{p}": float(pw_delta[p][si, di]) for p in top_pw},
            })
    feats_df = pd.DataFrame(feat_rows)
    feats_df.to_parquet(OUT_DIR / "kryzei_perturb_features.parquet", index=False)
    print(f"[exp025] wrote {len(feats_df)} perturbation feature rows", flush=True)

    # ============ Part 3: direct arm predictions =================================
    print("[exp025] Part 3: direct arm (B0 + pathway_pca_reg)", flush=True)
    split, merged = load_tables()
    ppr_val, ppr_test = [], []
    for s in SEEDS:
        dv = pd.read_parquet(WAVE / f"inner_val_ensemble/ensm_pathway_pca_reg_seed{s}_gpu1_predictions.parquet")
        ppr_val.append(dv[dv["split"] == "val"].copy())
        dt = pd.read_parquet(WAVE / f"multiseed/dual_mlp_pathway_pca_reg_seed{s}_gpu1_predictions.parquet")
        ppr_test.append(dt[dt["split"] == "test"].copy())

    def seed_mean(dfs):
        keys = ["fold", "split", "patient_id", "sample_id", "drug_id", "y"]
        base = dfs[0].sort_values(keys).reset_index(drop=True)[keys].copy()
        acc = np.zeros(len(base))
        for d in dfs:
            dd = d.sort_values(keys).reset_index(drop=True)
            assert dd[keys].equals(base[keys]), "seed keys mismatch"
            acc += dd["pred"].to_numpy()
        base["pred"] = acc / len(dfs)
        return base

    ppr_val_m = seed_mean(ppr_val)
    ppr_test_m = seed_mean(ppr_test)
    mchk = merged[["sample_id", "drug_id", "y"]].copy()
    for df, tag in [(ppr_test_m, "test"), (ppr_val_m, "val")]:
        j = df.merge(mchk, on=["sample_id", "drug_id"], suffixes=("", "_m"))
        assert np.allclose(j["y"].astype(float), j["y_m"].astype(float), rtol=1e-5, atol=1e-6), f"y mismatch ({tag})"

    b0_fold_csv = {}
    with B0_FOLD_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model"] == "B0_drug_mean" and row["split"] == "test":
                b0_fold_csv[int(row["fold"])] = float(row["ndcg@5"])

    feat_by_sample = feats_df.set_index(["sample_id", "drug_id"])
    pw_cols = [f"pw_{p}" for p in top_pw]

    def assemble(rows: pd.DataFrame, direct_pred: np.ndarray, b0_pred: np.ndarray) -> np.ndarray:
        keys = list(zip(rows["sample_id"].astype(str), rows["drug_id"].astype(str)))
        fm = np.array([feat_by_sample.loc[k, "perturb_magnitude"] for k in keys])
        rv = np.array([feat_by_sample.loc[k, "reversal_score"] for k in keys])
        cv = np.array([feat_by_sample.loc[k, "lincs_coverage"] for k in keys])
        pw = np.column_stack([np.array([feat_by_sample.loc[k, c] for k in keys]) for c in pw_cols])
        return np.column_stack([direct_pred, b0_pred, fm, rv, pw, cv]).astype(np.float64)

    # ============ Part 4: second-layer joint model (per outer fold) ==============
    print("[exp025] Part 4: joint second layer (inner-val OOF only)", flush=True)
    joint_rows, fold_stats = [], []
    for fold in split["folds"]:
        f_id = int(fold.get("fold", fold.get("fold_id")))
        train_p = set(map(str, fold["train_patient_ids"]))
        test_p = set(map(str, fold["test_patient_ids"]))
        val_p = sorted(map(str, fold["val_patient_ids"]))
        assert set(val_p) <= train_p and not (set(val_p) & test_p)
        inner_train_p = train_p - set(val_p)

        val_rows = ppr_val_m[ppr_val_m["fold"] == f_id].copy()
        test_rows = ppr_test_m[ppr_test_m["fold"] == f_id].copy()
        b0_inner = b0_dedup_drug_means(merged[merged["patient_id"].isin(inner_train_p)])
        b0_full = b0_dedup_drug_means(merged[merged["patient_id"].isin(train_p)])
        Xval = assemble(val_rows, val_rows["pred"].to_numpy(), b0_predict(b0_inner, val_rows["drug_id"]))
        yval = val_rows["y"].to_numpy(float)
        Xte = assemble(test_rows, test_rows["pred"].to_numpy(), b0_predict(b0_full, test_rows["drug_id"]))
        yte = test_rows["y"].to_numpy(float)

        scaler = StandardScaler().fit(Xval)
        joint = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(Xval), yval)
        pred_te = joint.predict(scaler.transform(Xte))
        pred_val = joint.predict(scaler.transform(Xval))
        jr = test_rows[["patient_id", "sample_id", "drug_id", "y"]].copy()
        jr["pred"] = pred_te
        jr["fold"] = f_id
        joint_rows.append(jr)
        # official B0 on test (carve-fit protocol; verify vs frozen CSV)
        fit_p1, _ = carve_patients_stage1(train_p, f_id)
        fit_rows = merged[merged["patient_id"].isin(fit_p1)]
        b0m = fit_b0_drug_means(fit_rows["drug_id"].tolist(), fit_rows["y"].tolist())
        b0_te = test_rows[["patient_id", "sample_id", "drug_id", "y"]].copy()
        b0_te["pred"] = predict_b0(b0m, b0_te["drug_id"].tolist())
        nd_b0 = fold_macro_taskb(b0_te)
        assert abs(nd_b0[0] - b0_fold_csv[f_id]) < 1e-8, f"B0 mismatch fold{f_id}: {nd_b0[0]} vs {b0_fold_csv[f_id]}"
        fold_stats.append({
            "fold": f_id, "n_val_rows": len(val_rows), "n_val_patients": len(val_p),
            "n_test_rows": len(test_rows), "n_test_patients": len(test_p),
            "b0_official_taskb_ndcg5": nd_b0[0],
            "joint_val_mae": float(np.mean(np.abs(pred_val - yval))),
        })
        print(f"[exp025] fold{f_id}: val_rows={len(val_rows)} test_rows={len(test_rows)} "
              f"B0_official={nd_b0[0]:.4f} joint_val_mae={fold_stats[-1]['joint_val_mae']:.3f}", flush=True)
    joint_df = pd.concat(joint_rows, ignore_index=True)
    run_log["folds"] = fold_stats

    # ============ Part 5: three-way comparison ====================================
    print("[exp025] Part 5: three-way comparison + CI", flush=True)
    arms = {"B0_official": {}, "pathway_pca_reg": {}, "reversal_only": {}, "joint": {}}
    raw_macros = {"B0_official": [], "pathway_pca_reg": [], "reversal_only": [], "joint": []}
    for fold in split["folds"]:
        f_id = int(fold.get("fold", fold.get("fold_id")))
        train_p = set(map(str, fold["train_patient_ids"]))
        te = ppr_test_m[ppr_test_m["fold"] == f_id].copy()
        fit_p1, _ = carve_patients_stage1(train_p, f_id)
        fit_rows = merged[merged["patient_id"].isin(fit_p1)]
        b0m = fit_b0_drug_means(fit_rows["drug_id"].tolist(), fit_rows["y"].tolist())
        b0_te = te.copy()
        b0_te["pred"] = predict_b0(b0m, b0_te["drug_id"].tolist())
        b0_full = b0_dedup_drug_means(merged[merged["patient_id"].isin(train_p)])
        b0_raw = te.copy()
        b0_raw["pred"] = b0_predict(b0_full, b0_raw["drug_id"])
        rev_te = te[["patient_id", "sample_id", "drug_id", "y"]].copy()
        keys = list(zip(rev_te["sample_id"].astype(str), rev_te["drug_id"].astype(str)))
        rev_te["pred"] = [float(feat_by_sample.loc[k, "reversal_score"]) for k in keys]
        jt = joint_df[joint_df["fold"] == f_id].copy()
        arms["B0_official"][f_id] = b0_te
        arms["pathway_pca_reg"][f_id] = te
        arms["reversal_only"][f_id] = rev_te
        arms["joint"][f_id] = jt
        raw_macros["B0_official"].append(rawrow_patient_scores(b0_raw))
        raw_macros["pathway_pca_reg"].append(rawrow_patient_scores(te))
        raw_macros["reversal_only"].append(rawrow_patient_scores(rev_te))
        raw_macros["joint"].append(rawrow_patient_scores(jt))

    fold_ids = sorted(arms["joint"])
    pat_ids = sorted({p for f in fold_ids for p in taskb_patient_scores(arms["joint"][f])})
    per_pat = {}
    for arm, byf in arms.items():
        vals = {}
        for f in fold_ids:
            for pid, (n_, s_) in taskb_patient_scores(byf[f]).items():
                vals.setdefault(pid, []).append((n_, s_))
        per_pat[arm] = {pid: (float(np.nanmean([v[0] for v in vs])), float(np.nanmean([v[1] for v in vs])))
                        for pid, vs in vals.items()}
    macro = {}
    for arm in arms:
        nd = [v[0] for v in per_pat[arm].values()]
        sp = [v[1] for v in per_pat[arm].values()]
        fold_macro = [fold_macro_taskb(arms[arm][f]) for f in fold_ids]
        raw_nd = [float(np.nanmean([v[0] for v in rm.values()])) for rm in raw_macros[arm]]
        macro[arm] = {
            "patient_macro_ndcg5_taskb": float(np.nanmean(nd)),
            "patient_macro_spearman_taskb": float(np.nanmean(sp)),
            "fold_equal_weight_macro_ndcg5_taskb": float(np.nanmean([m[0] for m in fold_macro])),
            "fold_equal_weight_macro_spearman_taskb": float(np.nanmean([m[1] for m in fold_macro])),
            "stage2_raw_row_macro_ndcg5_sensitivity": float(np.nanmean(raw_nd)),
            "n_patients": len(nd),
        }
        print(f"[exp025] {arm}: taskB patient-macro NDCG@5={macro[arm]['patient_macro_ndcg5_taskb']:.4f} "
              f"Spearman={macro[arm]['patient_macro_spearman_taskb']:.4f} "
              f"fold-macro={macro[arm]['fold_equal_weight_macro_ndcg5_taskb']:.4f} "
              f"raw-macro={macro[arm]['stage2_raw_row_macro_ndcg5_sensitivity']:.4f}", flush=True)

    jv = np.array([per_pat["joint"][p][0] for p in pat_ids])
    pv = np.array([per_pat["pathway_pca_reg"][p][0] for p in pat_ids])
    bv = np.array([per_pat["B0_official"][p][0] for p in pat_ids])
    ok = ~(np.isnan(jv) | np.isnan(pv) | np.isnan(bv))
    ci_primary = paired_resample_ci(jv[ok], pv[ok], n_resamples=2000, unit="patient", random_state=BOOT_RS)
    ci_vs_b0 = paired_resample_ci(jv[ok], bv[ok], n_resamples=2000, unit="patient", random_state=BOOT_RS)
    jf = [fold_macro_taskb(arms["joint"][f])[0] for f in fold_ids]
    pf = [fold_macro_taskb(arms["pathway_pca_reg"][f])[0] for f in fold_ids]
    bf = [fold_macro_taskb(arms["B0_official"][f])[0] for f in fold_ids]
    ci_fold_ppr = paired_resample_ci(jf, pf, n_resamples=2000, unit="fold", random_state=BOOT_RS)
    ci_fold_b0 = paired_resample_ci(jf, bf, n_resamples=2000, unit="fold", random_state=BOOT_RS)
    print(f"[exp025] PRIMARY joint-ppr: d={ci_primary.estimate:+.4f} "
          f"[{ci_primary.low:.4f},{ci_primary.high:.4f}] crosses_zero={ci_primary.crosses_zero} n={int(ok.sum())}", flush=True)

    verdict = "EXPAND" if (ci_primary.estimate > 0 and not ci_primary.crosses_zero) else "DO_NOT_EXPAND"
    run_log["three_way"] = macro
    run_log["endpoints"] = {
        "primary_joint_minus_ppr_patient_ci": {"delta": ci_primary.estimate, "low": ci_primary.low,
                                               "high": ci_primary.high, "crosses_zero": ci_primary.crosses_zero,
                                               "n_patients": int(ok.sum()), "n_resamples": 2000},
        "secondary_joint_minus_b0_patient_ci": {"delta": ci_vs_b0.estimate, "low": ci_vs_b0.low,
                                                "high": ci_vs_b0.high, "crosses_zero": ci_vs_b0.crosses_zero},
        "secondary_joint_minus_ppr_fold_ci": {"delta": ci_fold_ppr.estimate, "low": ci_fold_ppr.low,
                                              "high": ci_fold_ppr.high, "crosses_zero": ci_fold_ppr.crosses_zero},
        "secondary_joint_minus_b0_fold_ci": {"delta": ci_fold_b0.estimate, "low": ci_fold_b0.low,
                                             "high": ci_fold_b0.high, "crosses_zero": ci_fold_b0.crosses_zero},
        "verdict": verdict,
    }
    run_log["elapsed_sec"] = round(time.time() - t0, 1)

    joint_df.to_parquet(OUT_DIR / "joint_predictions.parquet", index=False)
    per_pat_df = pd.DataFrame([
        {"patient_id": p, "arm": arm, "ndcg5_taskb": v[0], "spearman_taskb": v[1]}
        for arm in arms for p, v in per_pat[arm].items()])
    per_pat_df.to_parquet(OUT_DIR / "three_way_patient_scores.parquet", index=False)
    (OUT_DIR / "joint_feasibility_results.json").write_text(
        json.dumps(run_log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[exp025] DONE verdict={verdict} elapsed={run_log['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
