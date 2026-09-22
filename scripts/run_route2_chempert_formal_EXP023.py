#!/usr/bin/env python3
"""EXP-023: ChemPert formal drug-conditioned perturbation experiment.

Tests whether drug-conditioned heads beat a ctrl_only baseline when predicting
ChemPert up/down response signatures from control (initial-state) expression
profiles, under accession-group-isolated splits with frozen v2 contract data.

Arms (same capacity scale as the LINCS EXP-022 ablations):
  - null_zero, global_mean_signature (references)
  - ctrl_only      : Linear(795 -> 795) on z-scored control profile
  - ctrl_drug_id   : ctrl + learnable drug embedding -> MLP(512)
  - ctrl_morgan    : ctrl + 1024-bit Morgan FP (SMILES-matched subset only)
  - drug_only      : drug embedding only (control-deletion ablation)

Protocols (whole accession_group isolation per the frozen leakage rule):
  - P1 test=GSE92742 (train = GSE70138 + other)
  - P2 test=GSE70138 (train = GSE92742 + other)

Primary endpoint: paired mean-cosine delta (arm - ctrl_only) with compound-
level grouped bootstrap 95% CI (2000 resamples).

Targets: responses RDS up/down gene lists mapped onto the 795-gene landmark
intersection (up=+1 / down=-1 / other=0); mapping coverage reported.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

RAW_SMILES_PATH = "data/processed/chempert/chempert_v2_chem_smiles.parquet"
TABLE_PATH = "data/processed/chempert/chempert_v2_response_table.parquet"
SLICE_PATH = "data/processed/chempert/chempert_v2_control_expression_slice.parquet"


@dataclass
class Score:
    name: str
    n_eval: int
    mean_cosine: float
    mean_pearson: float


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom < 1e-12 else float(np.dot(a, b) / denom)


def pearson_np(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False) - a.mean()
    b = b.astype(np.float64, copy=False) - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom < 1e-12 else float(np.dot(a, b) / denom)


def score_rows(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Row-wise cosine; assumes rows already filtered to nonzero-signature rows."""
    p = pred.astype(np.float64)
    t = true.astype(np.float64)
    num = (p * t).sum(axis=1)
    denom = np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1)
    out = np.zeros(len(p))
    ok = denom > 1e-12
    out[ok] = num[ok] / denom[ok]
    return out


def pearson_rows(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    p = pred.astype(np.float64)
    t = true.astype(np.float64)
    pc = p - p.mean(axis=1, keepdims=True)
    tc = t - t.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(pc, axis=1) * np.linalg.norm(tc, axis=1)
    out = np.zeros(len(p))
    ok = denom > 1e-12
    out[ok] = (pc * tc).sum(axis=1)[ok] / denom[ok]
    return out


class CtrlOnly(nn.Module):
    def __init__(self, n_genes: int):
        super().__init__()
        self.net = nn.Linear(n_genes, n_genes)

    def forward(self, ctrl, _drug, _feat):
        return self.net(ctrl)


class DrugCond(nn.Module):
    def __init__(self, n_genes: int, n_drugs: int, emb_dim: int = 64, hidden: int = 512):
        super().__init__()
        self.emb = nn.Embedding(n_drugs + 1, emb_dim)  # index 0 = UNK
        self.net = nn.Sequential(
            nn.Linear(n_genes + emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_genes),
        )

    def forward(self, ctrl, drug, _feat):
        return self.net(torch.cat([ctrl, self.emb(drug)], dim=-1))


class FeatCond(nn.Module):
    def __init__(self, n_genes: int, feat_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes + feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_genes),
        )

    def forward(self, ctrl, _drug, feat):
        return self.net(torch.cat([ctrl, feat], dim=-1))


class DrugOnly(nn.Module):
    def __init__(self, n_genes: int, n_drugs: int, emb_dim: int = 64, hidden: int = 512):
        super().__init__()
        self.emb = nn.Embedding(n_drugs + 1, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_genes),
        )

    def forward(self, _ctrl, drug, _feat):
        return self.net(self.emb(drug))


def train_model(model, x, y, drug, feat, device, epochs, batch_size, lr, wd):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(x, drug, feat, y), batch_size=batch_size, shuffle=True, drop_last=False
    )
    model.train()
    hist = []
    for epoch in range(1, epochs + 1):
        total, n = 0.0, 0
        for xb, db, fb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb, db, fb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        avg = total / max(n, 1)
        hist.append({"epoch": epoch, "train_mse": avg})
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"    epoch {epoch}/{epochs} train_mse={avg:.6f}", flush=True)
    return hist


@torch.no_grad()
def predict(model, x, drug, feat):
    model.eval()
    return model(x, drug, feat).detach().cpu().numpy()


def grouped_bootstrap_ci(deltas: np.ndarray, groups: np.ndarray, n_boot: int, seed: int):
    """Compound-level grouped bootstrap of the mean paired delta."""
    rng = np.random.default_rng(seed)
    uniq = pd.unique(groups)
    group_idx = {g: np.where(groups == g)[0] for g in uniq}
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([group_idx[g] for g in pick])
        stats[b] = deltas[idx].mean()
    return {
        "delta_mean": float(deltas.mean()),
        "ci_low": float(np.percentile(stats, 2.5)),
        "ci_high": float(np.percentile(stats, 97.5)),
        "n_bootstrap": n_boot,
        "n_groups": int(len(uniq)),
    }


def load_data(root: Path):
    table = pd.read_parquet(root / TABLE_PATH)
    sl = pd.read_parquet(root / SLICE_PATH)
    genes = sl["gene_symbol"].tolist()
    prof_cols = [c for c in sl.columns if c != "gene_symbol"]
    slice_mat = sl[prof_cols].to_numpy(dtype=np.float32)  # (795, n_profiles)
    col_pos = {c: j for j, c in enumerate(prof_cols)}
    gene_pos = {g: i for i, g in enumerate(genes)}

    def signature_vector(up: str, down: str) -> tuple[np.ndarray, int, int]:
        v = np.zeros(len(genes), dtype=np.float32)
        n_up = n_down = 0
        for g in str(up).split(";"):
            g = g.strip()
            if g in gene_pos:
                v[gene_pos[g]] = 1.0
                n_up += 1
        for g in str(down).split(";"):
            g = g.strip()
            if g in gene_pos:
                v[gene_pos[g]] = -1.0
                n_down += 1
        return v, n_up, n_down

    sigs, cov_up, cov_down, n_up_all, n_down_all = [], [], [], [], []
    for up, down in zip(table["up_genes"], table["down_genes"]):
        v, nu, nd = signature_vector(up, down)
        su = sum(1 for g in str(up).split(";") if g.strip() in gene_pos)
        sd = sum(1 for g in str(down).split(";") if g.strip() in gene_pos)
        tot_up = sum(1 for g in str(up).split(";") if g.strip())
        tot_down = sum(1 for g in str(down).split(";") if g.strip())
        sigs.append(v)
        cov_up.append(su)
        cov_down.append(sd)
        n_up_all.append(tot_up)
        n_down_all.append(tot_down)
    Y = np.stack(sigs)
    X = slice_mat[:, [col_pos[e] for e in table["Expression_ID"]]].T  # (n, 795)

    smiles = pd.read_parquet(root / RAW_SMILES_PATH)
    return table, X, Y, genes, smiles, np.array(cov_up), np.array(cov_down), np.array(n_up_all), np.array(n_down_all)


def morgan_fp(smiles: str) -> np.ndarray | None:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
    arr = np.zeros(1024, dtype=np.float32)
    for b in fp.GetOnBits():
        arr[b] = 1.0
    return arr


def run_arm(name, model_kind, tr_idx, te_idx, Xz, Y, drug, feat, device, hp, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_genes = Xz.shape[1]
    if model_kind == "ctrl_only":
        model = CtrlOnly(n_genes).to(device)
    elif model_kind == "ctrl_drug_id":
        model = DrugCond(n_genes, hp["n_vocab"], hp["emb_dim"], hp["hidden"]).to(device)
    elif model_kind == "ctrl_morgan":
        model = FeatCond(n_genes, feat.shape[1], hp["hidden"]).to(device)
    elif model_kind == "drug_only":
        model = DrugOnly(n_genes, hp["n_vocab"], hp["emb_dim"], hp["hidden"]).to(device)
    else:
        raise ValueError(model_kind)
    hist = train_model(
        model,
        torch.from_numpy(Xz[tr_idx]).to(device),
        torch.from_numpy(Y[tr_idx]).to(device),
        torch.from_numpy(drug[tr_idx]).to(device),
        torch.from_numpy(feat[tr_idx]).to(device),
        device,
        hp["epochs"], hp["batch_size"], hp["lr"], hp["wd"],
    )
    pred = predict(
        model,
        torch.from_numpy(Xz[te_idx]).to(device),
        torch.from_numpy(drug[te_idx]).to(device),
        torch.from_numpy(feat[te_idx]).to(device),
    )
    return pred, hist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/mnt/d/Code/DrugScreenLab"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    root = args.root

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this run")
    device = torch.device("cuda:0")

    out_dir = root / "artifacts" / "route2_chempert_EXP023_DRAFT"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("[exp023] loading v2 contract data ...", flush=True)
    table, X, Y, genes, smiles_tab, cov_up, cov_down, n_up_all, n_down_all = load_data(root)
    print(f"[exp023] responses={len(table)} genes={len(genes)}", flush=True)

    # signature mapping coverage
    cov = {
        "n_responses": int(len(table)),
        "mean_up_genes_total": float(n_up_all.mean()),
        "mean_down_genes_total": float(n_down_all.mean()),
        "mean_up_mapped_795": float(cov_up.mean()),
        "mean_down_mapped_795": float(cov_down.mean()),
        "frac_up_mapped": float(cov_up.sum() / max(n_up_all.sum(), 1)),
        "frac_down_mapped": float(cov_down.sum() / max(n_down_all.sum(), 1)),
        "n_zero_signature_rows": int(((cov_up + cov_down) == 0).sum()),
    }

    # SMILES -> morgan features
    smi_tab = smiles_tab.dropna(subset=["canonical_smiles"])
    chem2smi = dict(zip(smi_tab["Chemical_ID"], smi_tab["canonical_smiles"].astype(str)))
    chem_ids = table["Chemical_ID"].to_numpy()
    has_smiles = np.array([c in chem2smi for c in chem_ids])
    fp_cache: dict[str, np.ndarray] = {}
    feat_morgan = np.zeros((len(table), 1024), dtype=np.float32)
    n_unencodable = 0
    for i, c in enumerate(chem_ids):
        if c not in chem2smi:
            continue
        if c not in fp_cache:
            fp = morgan_fp(chem2smi[c])
            if fp is None:
                fp_cache[c] = None
            else:
                fp_cache[c] = fp
        fp = fp_cache[c]
        if fp is None:
            n_unencodable += 1
        else:
            feat_morgan[i] = fp
    encodable = has_smiles & np.array(
        [(fp_cache.get(c) is not None) for c in chem_ids]
    )
    print(f"[exp023] smiles-encodable responses={int(encodable.sum())} "
          f"unencodable={n_unencodable}", flush=True)

    groups = table["accession_group"].to_numpy()
    protocols = [
        {"name": "P1_test_GSE92742", "test_group": "GSE92742"},
        {"name": "P2_test_GSE70138", "test_group": "GSE70138"},
    ]

    all_pred_rows = []
    results = {"protocols": [], "coverage": cov, "seed": args.seed}
    for proto in protocols:
        te_mask = groups == proto["test_group"]
        tr_mask = ~te_mask
        te_idx = np.where(te_mask)[0]
        tr_idx = np.where(tr_mask)[0]
        # drop zero-signature rows from BOTH train and eval (uninformative)
        nz = (cov_up + cov_down) > 0
        tr_idx = tr_idx[nz[tr_idx]]
        te_idx = te_idx[nz[te_idx]]

        # z-score with train statistics
        mu = X[tr_idx].mean(axis=0, keepdims=True)
        sd = X[tr_idx].std(axis=0, keepdims=True) + 1e-6
        Xz = (X - mu) / sd

        # drug vocab from train only; 0 = UNK
        vocab = {}
        for c in chem_ids[tr_idx]:
            if c not in vocab:
                vocab[c] = len(vocab) + 1
        drug = np.array([vocab.get(c, 0) for c in chem_ids], dtype=np.int64)
        n_unk_te = int((drug[te_idx] == 0).sum())
        te_chems = set(chem_ids[te_idx])
        tr_chems = set(chem_ids[tr_idx])
        visible = len(te_chems & tr_chems)

        # global mean baseline (train signature mean)
        y_mu = Y[tr_idx].mean(axis=0, keepdims=True)

        hp = {
            "n_vocab": len(vocab), "emb_dim": args.emb_dim, "hidden": args.hidden,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "wd": args.weight_decay,
        }
        print(f"[{proto['name']}] train={len(tr_idx)} test={len(te_idx)} vocab={len(vocab)} "
              f"test_unk_drug_rows={n_unk_te} test_chem={len(te_chems)} visible_in_train={visible}",
              flush=True)

        preds = {}
        # reference baselines
        preds["null_zero"] = np.zeros((len(te_idx), Y.shape[1]), dtype=np.float32)
        preds["global_mean_signature"] = np.repeat(y_mu, len(te_idx), axis=0).astype(np.float32)

        # ctrl_only / ctrl_drug_id / drug_only on the full (nonzero-signature) sets
        for arm, kind in [("ctrl_only", "ctrl_only"), ("ctrl_drug_id", "ctrl_drug_id"),
                          ("drug_only", "drug_only")]:
            print(f"[{proto['name']}] training {arm} ...", flush=True)
            pred, hist = run_arm(arm, kind, tr_idx, te_idx, Xz, Y, drug,
                                 feat_morgan, device, hp, args.seed)
            preds[arm] = pred

        # ctrl_morgan on the SMILES-encodable subset
        tr_sm = tr_idx[encodable[tr_idx]]
        te_sm = te_idx[encodable[te_idx]]
        print(f"[{proto['name']}] training ctrl_morgan on encodable subset "
              f"train={len(tr_sm)} test={len(te_sm)} ...", flush=True)
        # need an index translation: run_arm slices Xz[tr_idx] etc. Use subset arrays.
        pred_m, _ = run_arm("ctrl_morgan", "ctrl_morgan", tr_sm, te_sm, Xz, Y, drug,
                            feat_morgan, device, hp, args.seed)

        # scores
        Y_te = Y[te_idx]
        arm_scores = {}
        for arm, pred in preds.items():
            cos = score_rows(pred, Y_te)
            pea = pearson_rows(pred, Y_te)
            arm_scores[arm] = Score(arm, len(te_idx), float(cos.mean()), float(pea.mean()))
            for i_local, i_global in enumerate(te_idx):
                all_pred_rows.append({
                    "protocol": proto["name"], "arm": arm,
                    "Response_ID": table["Response_ID"].iloc[i_global],
                    "Chemical_ID": chem_ids[i_global],
                    "cosine": float(cos[i_local]), "pearson": float(pea[i_local]),
                })
        # morgan scores on its subset
        cos_m = score_rows(pred_m, Y[te_sm])
        pea_m = pearson_rows(pred_m, Y[te_sm])
        arm_scores["ctrl_morgan"] = Score("ctrl_morgan", len(te_sm), float(cos_m.mean()), float(pea_m.mean()))
        for k_local, k_global in enumerate(te_sm):
            all_pred_rows.append({
                "protocol": proto["name"], "arm": "ctrl_morgan",
                "Response_ID": table["Response_ID"].iloc[k_global],
                "Chemical_ID": chem_ids[k_global],
                "cosine": float(cos_m[k_local]), "pearson": float(pea_m[k_local]),
            })

        # paired deltas vs ctrl_only with compound-level grouped bootstrap
        te_chem_arr = chem_ids[te_idx]
        deltas = {}
        for arm in ["ctrl_drug_id", "drug_only"]:
            d = arm_scores[arm].mean_cosine - arm_scores["ctrl_only"].mean_cosine
            per_row = score_rows(preds[arm], Y_te) - score_rows(preds["ctrl_only"], Y_te)
            ci = grouped_bootstrap_ci(per_row, te_chem_arr, args.n_boot, args.seed)
            deltas[arm] = {"scope": "full_test", **ci}
        # morgan delta on the encodable subset (ctrl_only re-scored on same rows,
        # same order: te_sm is an order-preserving subset of te_idx)
        cos_ctrl_full = score_rows(preds["ctrl_only"], Y_te)
        te_sm_set = set(te_sm.tolist())
        ctrl_cos_on_sm = np.array([cos_ctrl_full[k] for k in range(len(te_idx))
                                   if te_idx[k] in te_sm_set])
        per_row_m = cos_m - ctrl_cos_on_sm
        ci_m = grouped_bootstrap_ci(per_row_m, chem_ids[te_sm], args.n_boot, args.seed)
        deltas["ctrl_morgan"] = {"scope": "smiles_encodable_test_subset", **ci_m}

        results["protocols"].append({
            "protocol": proto["name"],
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "n_vocab_drugs": int(len(vocab)),
            "n_test_unk_drug_rows": n_unk_te,
            "n_test_chemicals": int(len(te_chems)),
            "n_test_chemicals_visible_in_train": int(visible),
            "n_morgan_train": int(len(tr_sm)),
            "n_morgan_test": int(len(te_sm)),
            "scores": [asdict(s) for s in arm_scores.values()],
            "paired_delta_vs_ctrl_only_95CI": deltas,
        })
        print(f"[{proto['name']}] " + " | ".join(
            f"{s.name}={s.mean_cosine:.4f}" for s in arm_scores.values()), flush=True)
        for arm, ci in deltas.items():
            print(f"  delta[{arm}] = {ci['delta_mean']:.4f} "
                  f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]", flush=True)

    # artifacts
    pred_df = pd.DataFrame(all_pred_rows)
    pred_path = out_dir / "per_response_scores.parquet"
    pred_df.to_parquet(pred_path, index=False)

    metrics = {
        "status": "DRAFT",
        "experiment_id": "EXP-023",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "device": {
            "requested_physical_gpu": args.gpu,
            "name": torch.cuda.get_device_name(0),
        },
        "seed": args.seed,
        "data": {
            "contract": "chempert_contract_v2_supplement_20260919",
            "n_responses_v2_true": int(len(table)),
            "gene_space": "795 landmark intersection genes",
            "input": "z-scored control (initial-state) expression profile, per-response "
                     "Expression_ID column of the v2 slice",
            "target": "up/down signature mapped to the 795-gene space (up=+1/down=-1/other=0)",
            "coverage": cov,
            "smiles_encodable_responses": int(encodable.sum()),
            "smiles_unencodable_chemicals": int(n_unencodable),
        },
        "model": {
            "ctrl_only": "Linear(795->795)",
            "ctrl_drug_id": f"Emb({args.emb_dim})+MLP({args.hidden}) -> 795",
            "ctrl_morgan": "Morgan1024(r=2)+MLP(512) -> 795 (SMILES-matched subset)",
            "drug_only": f"Emb({args.emb_dim})->MLP({args.hidden}) -> 795 (control deletion)",
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "weight_decay": args.weight_decay,
        },
        "split_rule": "whole accession_group isolation; P1 test=GSE92742, P2 test=GSE70138; "
                      "zero-signature rows excluded from train and eval (count in coverage)",
        "bootstrap": "compound-level grouped bootstrap, 2000 resamples, percentile 95% CI",
        "results": results,
        "elapsed_sec": round(time.time() - t0, 1),
        "notes": [
            "EXPLORATORY: ChemPert data license not explicitly stated by the download page "
            "(audit leftover); research use, no redistribution.",
            "ctrl_morgan arm evaluated only on the SMILES-encodable subset; its paired delta "
            "vs ctrl_only is computed on the same subset rows.",
            "Do not read high ctrl_only cosine as drug-specific perturbation learning.",
        ],
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[exp023] wrote {out_dir / 'metrics.json'} and {pred_path}")
    print(f"[exp023] elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
