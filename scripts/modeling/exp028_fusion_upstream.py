#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-028 upstream: route A frozen-split retrain + route B Hallmark-50 features.

Pre-registered contract (experiments/records/EXP-028.md, pinned BEFORE run):

Part 1 (route A, M1_source_scaled precise EXP-021 replication):
- kryeziu_mcrc_v3 registered cohort (hash-checked); frozen 5-fold patient split
  (kryeziu_patient_outer5fold_FROZEN.json, 51 patients; per fold train 40 incl
  val 8 / test 11). Official runner's seed-dependent fold assignment disagrees
  with the frozen split (13/51 fold overlap) -> retrain on the frozen split.
- Features: EXP-015 source preprocessor (512-dim) transform + Morgan 2048
  (radius=2); random init; DualEncoderPairModel official _PARAMS
  (hidden_dim=32, epochs=50, lr=1e-3, ranking_weight=0.2, max_rank_pairs=128,
  batch_groups=8); seeds 20260912/20260913/20260919.
- 105 trainings: 15 outer (train 40 -> test 11, role=outer_test),
  15 inner (train-val 32 -> val 8, role=inner_val),
  75 cross-fit (per outer fold, train patients split into 5 groups via
  assign_patient_folds seed=20260919; each group predicted by a model trained
  on the remaining 32, role=xfit_oof).
- B0_frozen reference: per-fold patient-mean-then-drug-mean drug means on the
  40 train patients -> test predictions (official B0 protocol).

Part 2 (route B, EXP-027 contract replication + extended output):
- ChemPert v2 ctrl_morgan (FeatCond 795+1024->512->795), SMILES-encodable
  nonzero-signature subset, seeds 23/24/25, epochs=40/batch=512/lr=1e-3/wd=1e-4.
- Kryzei zero-shot application identical to EXP-027 (785/795 present genes,
  missing z=0; 92-sample cohort-mean difference signature; reversal restricted
  to the 795∩Kryzei subspace).
- NEW: predicted perturbation delta aggregated to ALL 50 Hallmark pathways
  (pathway ∩ 795 predicted dims mean, NOT clipped to Kryzei-present genes;
  pure upstream output; coverage logged). GPU nondeterminism => independent
  sample of the same contract (preregistered note).

Outputs:
  artifacts/exp028_fusion/routeA_frozen_predictions.parquet
    (patient_id, model_id, drug_id, fold, role, seed, algorithm, observed,
     predicted, source_drug_status [, xgroup for xfit_oof])
  artifacts/exp028_fusion/routeB_features_exp028.parquet
    (sample_id, drug_id, seed, reversal_score, perturb_magnitude,
     pathway_reversal [EXP-027 B3 consistency check], pw_<HALLMARK_*>*50)
  artifacts/exp028_fusion/upstream_log.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from drug_screen.transfer.cohorts import load_registered_cohort
from drug_screen.transfer.models import DualEncoderPairModel
from drug_screen.transfer.preprocessing import ExpressionPreprocessor
from drug_screen.transfer.splits import assign_patient_folds
from drug_screen.transfer.transfer_experiment import (
    _PARAMS, _arrays, _fingerprints, _fit_b0, _predict_b0, _table,
)

ROOT = Path("/mnt/d/Code/DrugScreenLab")
REGISTRY = ROOT / "data/registry/transfer_cohorts.json"
COHORT_ID = "kryeziu_mcrc_v3"
SOURCE_DIR = ROOT / "data/artifacts/EXP-015/source_seed20260912"
SPLIT = ROOT / "data/splits/kryeziu_patient_outer5fold_FROZEN.json"
OUT_DIR = ROOT / "artifacts/exp028_fusion"

SEEDS_A = [20260912, 20260913, 20260919]
XFIT_SEED = 20260919

# ---- route B (EXP-027 contract) ----
TABLE_PATH = ROOT / "data/processed/chempert/chempert_v2_response_table.parquet"
SLICE_PATH = ROOT / "data/processed/chempert/chempert_v2_control_expression_slice.parquet"
SMILES_PATH = ROOT / "data/processed/chempert/chempert_v2_chem_smiles.parquet"
PROC = ROOT / "data/processed/transfer/mcrc_v3"
IDENT = ROOT / "data/processed/identity/kryeziu_drug_identity_48.json"
HALLMARK_GMT = ROOT / "data/raw/exp013/msigdb/h.all.v2023.2.Hs.symbols.gmt"
SEEDS_B = [23, 24, 25]
EPOCHS, BATCH, LR, WD = 40, 512, 1e-3, 1e-4
HIDDEN = 512
TOPK_PATHWAYS = 8


# ================================================================ part 1
def _frame(rows: pd.DataFrame, prediction: np.ndarray, fold: int, seed: int,
           role: str, algorithm: str, source_drugs: set[str],
           xgroup: int | None = None) -> pd.DataFrame:
    out = pd.DataFrame({
        "patient_id": rows.patient_id.astype(str),
        "model_id": rows.model_id.astype(str),
        "drug_id": rows.drug_id.astype(str),
        "fold": fold,
        "role": role,
        "seed": seed,
        "algorithm": algorithm,
        "observed": rows.response.to_numpy(float),
        "predicted": np.asarray(prediction, dtype=float),
        "source_drug_status": np.where(
            rows.drug_id.astype(str).isin(source_drugs), "known", "unknown"),
    })
    if xgroup is not None:
        out["xgroup"] = xgroup
    return out


def _fit_predict(features: np.ndarray, fingerprints: dict, params: dict,
                 fit_rows: pd.DataFrame, pred_rows: pd.DataFrame,
                 device: str) -> np.ndarray:
    model = DualEncoderPairModel(**params)
    model.fit(*_arrays(features, fingerprints, fit_rows),
              pdo_ids=fit_rows.model_id.astype(str).to_numpy())
    return model.predict(
        features[pred_rows.expression_index.to_numpy(int)],
        np.vstack([fingerprints[str(d)] for d in pred_rows.drug_id]))


def route_a(device: str) -> dict:
    t0 = time.time()
    bundle = load_registered_cohort(COHORT_ID, root=ROOT, registry=REGISTRY)
    table = _table(bundle)
    expression = np.asarray(bundle.expression, dtype=float)
    genes = np.asarray(bundle.gene_ids, dtype=str)
    source_prep = ExpressionPreprocessor.load(SOURCE_DIR / "expression_transform.npz")
    source_x = source_prep.transform(expression, genes)
    frozen = json.loads((SOURCE_DIR / "frozen_run.json").read_text(encoding="utf-8"))
    source_drugs = set(frozen["source_drug_ids"])
    fingerprints = _fingerprints(bundle, table.drug_id)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    folds = split["folds"]

    # split sanity
    all_test: set[str] = set()
    for fd in folds:
        train_p = set(map(str, fd["train_patient_ids"]))
        val_p = set(map(str, fd["val_patient_ids"]))
        test_p = set(map(str, fd["test_patient_ids"]))
        assert not (train_p & test_p) and not (val_p & test_p), "patient leakage"
        assert val_p <= train_p, "val must be a subset of train"
        assert len(train_p) + len(test_p) == 51 and 8 <= len(val_p) <= 9
        all_test |= test_p
    assert len(all_test) == 51, f"test folds must partition 51 patients, got {len(all_test)}"

    rows: list[pd.DataFrame] = []
    n_trainings = 0
    for seed in SEEDS_A:
        for fd in folds:
            fold = int(fd["fold"])
            train_p = set(map(str, fd["train_patient_ids"]))
            val_p = set(map(str, fd["val_patient_ids"]))
            test_p = set(map(str, fd["test_patient_ids"]))
            train = table.loc[table.patient_id.isin(train_p)].copy()
            val = table.loc[table.patient_id.isin(val_p)].copy()
            test = table.loc[table.patient_id.isin(test_p)].copy()
            inner_train = table.loc[table.patient_id.isin(train_p - val_p)].copy()
            params = _PARAMS | {"seed": seed, "device": device}
            print(f"[routeA] seed={seed} fold={fold} "
                  f"train={train.patient_id.nunique()} inner={inner_train.patient_id.nunique()} "
                  f"val={val.patient_id.nunique()} test={test.patient_id.nunique()}",
                  flush=True)

            # outer model -> test rows
            pred = _fit_predict(source_x, fingerprints, params, train, test, device)
            rows.append(_frame(test, pred, fold, seed, "outer_test",
                               "M1_source_scaled", source_drugs))
            n_trainings += 1
            # B0_frozen reference (outer only)
            b0 = _fit_b0(train)
            rows.append(_frame(test, _predict_b0(b0, test.drug_id), fold, seed,
                               "outer_test", "B0", source_drugs))

            # inner model -> val rows
            pred_i = _fit_predict(source_x, fingerprints, params, inner_train, val, device)
            rows.append(_frame(val, pred_i, fold, seed, "inner_val",
                               "M1_source_scaled", source_drugs))
            n_trainings += 1

            # cross-fit models -> train-patient rows (xfit_oof)
            assignment = assign_patient_folds(sorted(train_p), n_folds=5,
                                              seed=XFIT_SEED)
            for group in range(5):
                hold_p = {p for p, v in assignment.patient_to_test_fold.items()
                          if v == group}
                fit_p = train_p - hold_p
                fit_df = table.loc[table.patient_id.isin(fit_p)].copy()
                hold_df = table.loc[table.patient_id.isin(hold_p)].copy()
                pred_x = _fit_predict(source_x, fingerprints, params,
                                      fit_df, hold_df, device)
                rows.append(_frame(hold_df, pred_x, fold, seed, "xfit_oof",
                                   "M1_source_scaled", source_drugs, xgroup=group))
                n_trainings += 1

    frame = pd.concat(rows, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT_DIR / "routeA_frozen_predictions.parquet", index=False)
    log = {
        "n_trainings": n_trainings,
        "n_rows": len(frame),
        "rows_by_role": frame.role.value_counts().to_dict(),
        "rows_by_algorithm": frame.algorithm.value_counts().to_dict(),
        "elapsed_sec": round(time.time() - t0, 1),
        "device": device,
    }
    print(f"[routeA] DONE trainings={n_trainings} rows={len(frame)} "
          f"elapsed={log['elapsed_sec']}s", flush=True)
    return log


# ================================================================ part 2
class FeatCond(nn.Module):
    def __init__(self, n_genes: int, feat_dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes + feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_genes))

    def forward(self, ctrl, feat):
        return self.net(torch.cat([ctrl, feat], dim=-1))


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


def route_b(device) -> dict:
    t0 = time.time()
    table, X, Y, genes, cov, chem2smi = load_chempert()
    nz = np.array(cov) > 0
    chem_ids = table["Chemical_ID"].to_numpy()
    fp_all: dict[str, np.ndarray | None] = {}
    for c in sorted(set(chem_ids[nz]) & set(chem2smi)):
        fp_all[c] = morgan_fp(chem2smi[c])
    parseable = {c for c, f in fp_all.items() if f is not None}
    encodable = nz & np.array([c in parseable for c in chem_ids])
    print(f"[routeB] responses={len(table)} nonzero={int(nz.sum())} "
          f"encodable={int(encodable.sum())}", flush=True)

    mu = X[nz].mean(axis=0, keepdims=True)
    sd = X[nz].std(axis=0, keepdims=True) + 1e-6
    Xz_all = ((X - mu) / sd).astype(np.float32)

    # Kryzei expression -> 795 z-space
    npz = np.load(PROC / "indexed.npz", allow_pickle=True)
    kry_expr = np.asarray(npz["expression"], dtype=np.float32)
    kry_genes = [str(g) for g in np.asarray(npz["ordered_genes"]).tolist()]
    kry_sid = [str(s) for s in np.asarray(npz["sample_id"]).tolist()]
    gpos = {g: i for i, g in enumerate(kry_genes)}
    pos795 = np.array([gpos.get(g, -1) for g in genes])
    present = pos795 >= 0
    print(f"[routeB] 795 genes present={int(present.sum())} "
          f"missing(z=0)={int((~present).sum())}", flush=True)

    Z_k = np.zeros((len(kry_sid), len(genes)), dtype=np.float32)
    Z_k[:, present] = kry_expr[:, pos795[present]]
    Z_k = ((Z_k - mu) / sd).astype(np.float32)

    # Kryzei drug fingerprints
    ident = json.loads(IDENT.read_text(encoding="utf-8"))
    kry_drugs = ident["drugs"]
    fps = {}
    for d in kry_drugs:
        fp = morgan_fp(d["smiles"])
        if fp is not None:
            fps[d["drug_id"]] = fp
    drug_ids = [d["drug_id"] for d in kry_drugs]
    F_k = np.stack([fps[d] for d in drug_ids]).astype(np.float32)
    print(f"[routeB] kryzei drugs encoded={len(fps)}/48", flush=True)

    # Hallmark pathways: top-8 (EXP-027 B3 consistency) + all-50 deltas
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
    pos795_map = {g: i for i, g in enumerate(genes)}
    # top-8: Kryzei-present restricted (EXP-027 contract)
    pw_cols_top = [ [pos795_map[g] for g in hallmark[p]
                     if g in pos795_map and present[pos795_map[g]]]
                    for p in top_pw]
    # all-50: full predicted dims (NOT clipped to Kryzei-present; prereg)
    all_pw = sorted(hallmark)
    pw_cols_all = [ [pos795_map[g] for g in hallmark[p] if g in pos795_map]
                    for p in all_pw]
    print(f"[routeB] hallmark pathways={len(all_pw)} "
          f"top-{TOPK_PATHWAYS}={[(p, pw_cov[p]) for p in top_pw[:3]]} ...",
          flush=True)

    sample_diff = (Z_k - Z_k.mean(axis=0, keepdims=True)).astype(np.float64)
    sd_pw = np.stack([sample_diff[:, cols].mean(axis=1)
                      for cols in pw_cols_top], axis=1)      # (92, 8)

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
    reversal_stats = {}
    for seed in SEEDS_B:
        print(f"[routeB] training ctrl_morgan (seed {seed}) ...", flush=True)
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
            pred_delta = np.stack(preds)          # (92, 48, 795)

        magnitude = np.linalg.norm(pred_delta, axis=2)
        pd_sig = pred_delta[:, :, present].astype(np.float64)
        sdiff = sample_diff[:, present]
        num = np.einsum("sdk,sk->sd", pd_sig, sdiff)
        den = (np.linalg.norm(pd_sig, axis=2)
               * np.linalg.norm(sdiff, axis=1)[:, None]) + 1e-12
        reversal = -(num / den)                   # (92, 48)

        pw_delta_top = np.stack(
            [pred_delta[:, :, cols].mean(axis=2) for cols in pw_cols_top],
            axis=2)
        pnum = np.einsum("sdp,sp->sd", pw_delta_top, sd_pw)
        pden = (np.linalg.norm(pw_delta_top, axis=2)
                * np.linalg.norm(sd_pw, axis=1)[:, None]) + 1e-12
        pw_reversal = -(pnum / pden)              # (92, 48)

        # all-50 pathway deltas (full predicted dims)
        pw_all = np.stack(
            [pred_delta[:, :, cols].mean(axis=2) for cols in pw_cols_all],
            axis=2)                               # (92, 48, 50)

        for si, sid in enumerate(kry_sid):
            for di, did in enumerate(drug_ids):
                row = {
                    "sample_id": sid, "drug_id": did, "seed": seed,
                    "reversal_score": float(reversal[si, di]),
                    "perturb_magnitude": float(magnitude[si, di]),
                    "pathway_reversal": float(pw_reversal[si, di]),
                }
                for pi, p in enumerate(all_pw):
                    row[f"pw_{p}"] = float(pw_all[si, di, pi])
                rows.append(row)
        reversal_stats[str(seed)] = {
            "min": float(reversal.min()), "max": float(reversal.max()),
            "mean": float(reversal.mean())}
        print(f"[routeB] seed {seed}: reversal mean={reversal.mean():.4f} "
              f"min={reversal.min():.4f} max={reversal.max():.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    feats = pd.DataFrame(rows)
    feats.to_parquet(OUT_DIR / "routeB_features_exp028.parquet", index=False)
    log = {
        "n_responses": int(len(table)),
        "n_nonzero": int(nz.sum()),
        "n_encodable": int(encodable.sum()),
        "genes_795_in_kryzei": int(present.sum()),
        "genes_795_missing_in_kryzei": int((~present).sum()),
        "kryzei_drugs_encoded": len(fps),
        "hyperparams": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "wd": WD,
                        "hidden": HIDDEN, "seeds": SEEDS_B},
        "pathway_coverage_all50": {p: pw_cov[p] for p in all_pw},
        "top8_pathways": top_pw,
        "reversal_stats": reversal_stats,
        "n_feature_rows": len(feats),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    print(f"[routeB] DONE rows={len(feats)} elapsed={log['elapsed_sec']}s",
          flush=True)
    return log


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[exp028-upstream] device={device}", flush=True)
    log_a = route_a(device)
    log_b = route_b(torch.device(device))
    log = {
        "status": "RAN", "env": "drugscreening-gpu", "device": device,
        "route_a": log_a, "route_b": log_b,
        "elapsed_sec_total": round(time.time() - t0, 1),
    }
    (OUT_DIR / "upstream_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[exp028-upstream] ALL DONE elapsed={log['elapsed_sec_total']}s",
          flush=True)


if __name__ == "__main__":
    main()
