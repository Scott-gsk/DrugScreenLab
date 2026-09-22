#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-033 工作包 D 任务 2 较高臂：XPert 预训练扰动模型官方推理（冻结检查点）。

冻结检查点 data/external/xpert_source/saved_model/l1000_sdst_warm_split.pth；
官方复现环境（models/model_XPert.py + model_utils.py + flash_attn 本地回退 +
config_l1000）。978 landmark 基因（Kryzei 按 gene_symbol 对齐），对照输入
= X_ctl（Kryzei 表达 binned 到 n_bins=128，min=0/max=15.002），药物 = 预计算
UniMol(122,514)+HG(256)，drug_idx=InChIKey 匹配 l1000_sdst_drugs_info_8432 的
drug_node_idx。预测 delta = deg_output（978 维），逆转评分 = -cosine(deg,
sample_diff_978)，限 978∩Kryzei 基因子空间。

产出 artifacts/exp033_routeB/xpert_reversal.parquet + xpert_summary.json。
失败条件：官方推理不可运行 → 上报并记录，不换候选。
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
import yaml

ROOT = Path("/mnt/d/Code/DrugScreenLab")
XPERT = ROOT / "data/external/xpert_source"
OUT = ROOT / "artifacts" / "exp033_routeB"

SEED = 2024  # 官方 set_random_seed 默认
BOOT_SEED = 20260918
N_BOOT = 2000


def setup_xpert_env():
    sys.path.insert(0, str(XPERT))
    os.chdir(str(XPERT))
    # 本地 flash_attn 回退
    sys.path.insert(0, str(XPERT / "flash_attn"))


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    setup_xpert_env()

    from models.model_XPert import XPertNet

    audit = json.loads((ROOT / "data/processed/identity/kryzei_admission_audit_20260921.json").read_text(encoding="utf-8"))
    whitelist = set(audit["pdo_whitelist"])
    ident = json.loads((ROOT / "data/processed/identity/kryeziu_drug_identity_48.json").read_text(encoding="utf-8"))

    # ---- config + args ----
    with open(str(XPERT / "configs/config_l1000.yaml")) as f:
        config = yaml.safe_load(f)
    args = argparse_namespace(
        dataset="l1000_sdst", drug_feat="unimol", pretrained_mode="global",
        include_cell_idx=True, mode="test", output_attention=False,
        output_cls_embed=False, use_gene_pos_emed=False, wo_HG=False,
        wo_atom=False, wo_atom_HG=False, wo_unimol=False, wo_ppi=False,
        use_gradscaler=False)

    logger = logging.getLogger("xpert")
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- model + checkpoint ----
    model = XPertNet(args, config, device, logger)
    ckpt = torch.load(str(XPERT / "saved_model/l1000_sdst_warm_split.pth"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    log = {"status": "RAN", "checkpoint": "saved_model/l1000_sdst_warm_split.pth",
           "n_params_M": round(sum(p.nelement() for p in model.parameters()) / 1e6, 2)}

    # ---- assets ----
    unimol = np.load(str(XPERT / "processed_data/all_drugs_unimol_arr.npy"),
                     allow_pickle=True).astype(np.float32)
    hg = np.load(str(XPERT / "HG_data/saved_embedding/HG_drug_embeddings.npy"),
                 allow_pickle=True).astype(np.float32)
    info = pd.read_csv(str(XPERT / "processed_data/l1000_sdst_drugs_info_8432.csv"))
    info["inchi_key"] = info["inchi_key"].astype(str).str.strip()
    info["canonical_smiles"] = info["canonical_smiles"].astype(str)
    ik2node = {row.inchi_key: int(row.drug_node_idx) for row in info.itertuples(index=False)}

    # Kryzei 药物映射
    from rdkit import Chem
    drug_rows = []
    for d in ident["drugs"]:
        node = ik2node.get(d["inchikey"].strip())
        drug_rows.append({"drug_id": d["drug_id"], "node": node,
                          "inchi_key": d["inchikey"].strip()})
    matched = [r for r in drug_rows if r["node"] is not None]
    missed = [r["drug_id"] for r in drug_rows if r["node"] is None]
    log["drug_matched"] = len(matched)
    log["drug_missed"] = missed
    print(f"[xpert] drugs matched {len(matched)}/48, missed={missed}", flush=True)

    # Kryzei 表达 -> 978 landmark（gene_symbol 对齐，gene_idx 顺序）
    npz = np.load(str(ROOT / "data/processed/transfer/mcrc_v3/indexed.npz"), allow_pickle=True)
    kry_expr_all = np.asarray(npz["expression"], dtype=np.float32)
    kry_genes = [str(g) for g in np.asarray(npz["ordered_genes"])]
    kry_sid_all = [str(s) for s in np.asarray(npz["sample_id"])]
    gpos = {g: i for i, g in enumerate(kry_genes)}
    gi = pd.read_csv(str(XPERT / "processed_data/l1000_gene_info_978.csv"))
    gi = gi.sort_values("gene_idx")
    sym978 = [str(s) for s in gi["gene_symbol"]]
    col978 = np.array([gpos.get(s, -1) for s in sym978])   # 978 per-symbol Kryzei col
    pos978 = col978 >= 0
    log["genes_978_in_kryzei"] = int(pos978.sum())
    log["genes_978_missing_in_kryzei"] = int((~pos978).sum())

    sids = [(i, s) for i, s in enumerate(kry_sid_all) if s in whitelist]
    sid_idx = [i for i, _ in sids]
    kry_sid = [s for _, s in sids]
    kry_expr = kry_expr_all[sid_idx]

    kry978 = np.zeros((len(kry_sid), 978), dtype=np.float32)
    kry978[:, pos978] = kry_expr[:, col978[pos978]]

    # 978-sample 差异签名（60 样本队列均值）
    sample_diff = (kry978 - kry978.mean(axis=0, keepdims=True)).astype(np.float64)

    # binning（n_bins=128, min=0, max=15.002）
    min_v, max_v = float(config["dataset"]["min_value"]), float(config["dataset"]["max_value"])
    n_bins = int(config["dataset"]["n_bins"])
    bins = np.quantile(np.array([min_v, max_v]), np.linspace(0, 1, n_bins - 1))
    ctl_raw = kry978.astype(np.float64)
    ctl_binned = np.digitize(ctl_raw, bins).astype(np.int64).clip(0, n_bins - 1)

    sdiff_present = sample_diff[:, pos978]

    def run_drug(node):
        feat = unimol[node:node + 1]                      # (1,122,514)
        drug_idx = torch.tensor([node], dtype=torch.long, device=device)
        drug_feat = torch.tensor(feat, dtype=torch.float32, device=device)
        hg_embed = torch.tensor(hg[node:node + 1], dtype=torch.float32, device=device)
        degs = []
        for si in range(len(kry_sid)):
            trt = torch.tensor(ctl_raw[si:si + 1], dtype=torch.float32, device=device)
            ctl = torch.tensor(ctl_raw[si:si + 1], dtype=torch.float32, device=device)
            trtb = torch.tensor(ctl_binned[si:si + 1], dtype=torch.int64, device=device)
            ctlb = torch.tensor(ctl_binned[si:si + 1], dtype=torch.int64, device=device)
            dose = torch.tensor([0], dtype=torch.int64, device=device)
            time_idx = torch.tensor([0], dtype=torch.int64, device=device)
            cidx = torch.tensor([0], dtype=torch.long, device=device)
            tidx = torch.tensor([0], dtype=torch.long, device=device)
            data = (trt, ctl, trtb, ctlb, drug_feat, dose, time_idx,
                    drug_idx, cidx, tidx)
            with torch.no_grad():
                out = model(data)   # mode='test' -> 8-tuple
            deg = out[2].cpu().numpy()[0]                 # (978,)
            degs.append(deg)
        return np.asarray(degs)                            # (S, 978)

    rows = []
    mag_rows = []
    for r in matched:
        node = r["node"]
        deg = run_drug(node)                                # (60, 978)
        mag = np.linalg.norm(deg, axis=1)
        pd_sig = deg[:, pos978].astype(np.float64)
        num = np.einsum("sk,sk->s", pd_sig, sdiff_present)
        den = (np.linalg.norm(pd_sig, axis=1)
               * np.linalg.norm(sdiff_present, axis=1)) + 1e-12
        rev = -(num / den)
        for si, sid in enumerate(kry_sid):
            rows.append({"sample_id": sid, "drug_id": r["drug_id"],
                         "reversal_score": float(rev[si]),
                         "perturb_magnitude": float(mag[si])})
        print(f"[xpert] drug {r['drug_id']} reversal mean={rev.mean():.4f}", flush=True)

    feats = pd.DataFrame(rows)
    feats.to_parquet(OUT / "xpert_reversal.parquet", index=False)
    log["n_feature_rows"] = len(feats)
    log["n_samples"] = len(kry_sid)
    log["reproduce_command"] = (
        "cd /mnt/d/Code/DrugScreenLab/data/external/xpert_source && "
        "python /mnt/d/Code/DrugScreenLab/scripts/modeling/exp033_routeB_xpert.py")
    log["elapsed_sec"] = round(time.time() - t0, 1)

    # ---- 排序评估（38 药覆盖） vs observed ----
    samples = pd.read_parquet(ROOT / "data/processed/transfer/mcrc_v3/samples.parquet")
    pairs = pd.read_parquet(ROOT / "data/processed/transfer/mcrc_v3/pairs.parquet")
    samples = samples.copy()
    samples["sample_id"] = samples["sample_id"].astype(str)
    samples["patient_id"] = samples["patient_id"].astype(str)
    pairs = pairs.copy()
    pairs["sample_id"] = pairs["sample_id"].astype(str)
    pairs["drug_id"] = pairs["drug_id"].astype(str)
    obs = pairs.merge(samples[["sample_id", "patient_id"]], on="sample_id", how="inner")
    obs = obs[obs["sample_id"].isin(set(kry_sid))]
    obs = obs.rename(columns={"response": "observed"})
    obs["observed"] = pd.to_numeric(obs["observed"], errors="raise").astype(float)
    frame = feats.rename(columns={"reversal_score": "predicted"}).merge(
        obs[["sample_id", "drug_id", "patient_id", "observed"]],
        on=["sample_id", "drug_id"], how="inner")

    def agg_pd(f):
        g = f[["patient_id", "drug_id", "observed", "predicted"]].copy()
        for c in ("observed", "predicted"):
            g[c] = pd.to_numeric(g[c], errors="raise").astype(float)
        return g.groupby(["patient_id", "drug_id"], sort=True, as_index=False).agg(
            observed=("observed", "mean"), predicted=("predicted", "mean"))

    agg = agg_pd(frame)
    valid = []
    reasons = {}
    for p, g in agg.groupby("patient_id", sort=True):
        y = g.observed.to_numpy(float); pr = g.predicted.to_numpy(float)
        if len(y) < 3:
            reasons["insufficient_drugs"] = reasons.get("insufficient_drugs", 0) + 1
            continue
        if np.ptp(y) == 0 or np.ptp(pr) == 0:
            reasons["constant"] = reasons.get("constant", 0) + 1
            continue
        r = spearmanr(y, pr).statistic
        if np.isfinite(r):
            valid.append(float(r))
    intra = float(np.mean(valid)) if valid else None
    drug_rhos = []
    for d, g in agg.groupby("drug_id", sort=True):
        y = g.observed.to_numpy(float); pr = g.predicted.to_numpy(float)
        if len(y) >= 5 and np.ptp(y) > 0 and np.ptp(pr) > 0:
            rr = spearmanr(y, pr).statistic
            if np.isfinite(rr):
                drug_rhos.append(float(rr))
    per_drug = float(np.mean(drug_rhos)) if drug_rhos else None
    top1 = []
    for _, g in agg.groupby("patient_id", sort=True):
        y = g.observed.to_numpy(float); pr = g.predicted.to_numpy(float)
        if len(y) >= 1 and np.ptp(y) > 0 and np.ptp(pr) > 0:
            top1.append(int(np.argmax(-y) == np.argmax(-pr)))
    top1_v = float(np.mean(top1)) if top1 else None

    summary = {
        "experiment_id": "EXP-033", "task": "route_B_higher_tier_XPert",
        "status": "RAN",
        "supervision": "zero_target_label (frozen checkpoint, no finetune)",
        "coverage": {
            "drug_matched": len(matched), "drug_missed": missed,
            "genes_978_in_kryzei": int(pos978.sum()),
            "genes_978_missing_in_kryzei": int((~pos978).sum()),
        },
        "metrics": {
            "intra_patient_spearman": intra,
            "n_valid_patients": len(valid),
            "n_patients": int(frame.patient_id.nunique()),
            "n_drugs": int(frame.drug_id.nunique()),
            "per_drug_spearman": per_drug,
            "top1": top1_v,
            "reason_counts": reasons,
        },
        "log": log,
    }
    (OUT / "xpert_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8")
    print("WROTE xpert_reversal.parquet + xpert_summary.json")
    print(json.dumps(summary["metrics"], indent=2, ensure_ascii=False))


class argparse_namespace:
    def __init__(self, **kw):
        self.__dict__.update(kw)


if __name__ == "__main__":
    main()