"""事后描述性补充分析（2026-09-20，应用户要求执行）。

为 EXP-023（ChemPert 药物条件扰动预测）补充基因级排序指标：
原实验只报告了每条响应的余弦相似度与皮尔逊相关，本次补充
**每条响应在 795 个预测基因上的 Spearman 排序一致性**（预测能否把
上调基因排在下调基因前面），并沿用原协议做化合物分组 bootstrap 配对比较。

- 不改变 EXP-023 的任何判定：同种子(23)、同超参、同划分重训各臂
  （原始逐基因预测未落盘，只能重跑推理）；
- 复现质量校验：重跑的 mean cosine 与原 metrics.json 对比；
- 输出：artifacts/supplementary_ranking_metrics_20260920/
    chempert_spearman_supplement.json + chempert_per_response_spearman.parquet
"""
from __future__ import annotations

import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

ROOT = Path("/mnt/d/Code/DrugScreenLab")
OUT_DIR = ROOT / "artifacts/supplementary_ranking_metrics_20260920"
EXP023_SCRIPT = ROOT / "scripts/run_route2_chempert_formal_EXP023.py"
EXP023_METRICS = ROOT / "artifacts/route2_chempert_EXP023_DRAFT/metrics.json"

# 导入 EXP-023 脚本（同种子同协议复用其数据加载与模型定义）
import sys

spec = importlib.util.spec_from_file_location("exp023_formal", EXP023_SCRIPT)
exp023 = importlib.util.module_from_spec(spec)
sys.modules["exp023_formal"] = exp023
spec.loader.exec_module(exp023)


def spearman_rows(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """逐行（每条响应，跨 795 基因）Spearman：秩变换后做皮尔逊。"""
    p = pred.astype(np.float64)
    t = true.astype(np.float64)
    try:
        pr = rankdata(p, axis=1)
        tr = rankdata(t, axis=1)
    except TypeError:  # 旧 scipy 无 axis 参数
        pr = np.apply_along_axis(rankdata, 1, p)
        tr = np.apply_along_axis(rankdata, 1, t)
    pc = pr - pr.mean(axis=1, keepdims=True)
    tc = tr - tr.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(pc, axis=1) * np.linalg.norm(tc, axis=1)
    out = np.zeros(len(p))
    ok = denom > 1e-12
    out[ok] = (pc * tc).sum(axis=1)[ok] / denom[ok]
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[supp] device={device}", flush=True)
    t0 = time.time()

    table, X, Y, genes, smiles_tab, cov_up, cov_down, _, _ = exp023.load_data(ROOT)
    chem_ids = table["Chemical_ID"].to_numpy()
    groups = table["accession_group"].to_numpy()

    # SMILES -> Morgan（与原脚本一致的缓存逻辑）
    smi_tab = smiles_tab.dropna(subset=["canonical_smiles"])
    chem2smi = dict(zip(smi_tab["Chemical_ID"], smi_tab["canonical_smiles"].astype(str)))
    feat_morgan = np.zeros((len(table), 1024), dtype=np.float32)
    encodable = np.zeros(len(table), dtype=bool)
    for i, c in enumerate(chem_ids):
        if c not in chem2smi:
            continue
        fp = exp023.morgan_fp(chem2smi[c])
        if fp is not None:
            feat_morgan[i] = fp
            encodable[i] = True

    orig = json.loads(EXP023_METRICS.read_text())
    orig_cos = {}
    for proto in orig["results"]["protocols"]:
        for s in proto["scores"]:
            orig_cos[(proto["protocol"], s["name"])] = s["mean_cosine"]

    seed = orig.get("seed", 23)
    hp_defaults = dict(epochs=orig["model"]["epochs"],
                       batch_size=orig["model"]["batch_size"])
    protocols = [{"name": "P1_test_GSE92742", "test_group": "GSE92742"},
                 {"name": "P2_test_GSE70138", "test_group": "GSE70138"}]

    per_response_rows = []
    out_protocols = []
    for proto in protocols:
        te_mask = groups == proto["test_group"]
        tr_idx = np.where(~te_mask)[0]
        te_idx = np.where(te_mask)[0]
        nz = (cov_up + cov_down) > 0
        tr_idx = tr_idx[nz[tr_idx]]
        te_idx = te_idx[nz[te_idx]]

        mu = X[tr_idx].mean(axis=0, keepdims=True)
        sd = X[tr_idx].std(axis=0, keepdims=True) + 1e-6
        Xz = (X - mu) / sd

        vocab = {}
        for c in chem_ids[tr_idx]:
            if c not in vocab:
                vocab[c] = len(vocab) + 1
        drug = np.array([vocab.get(c, 0) for c in chem_ids], dtype=np.int64)
        y_mu = Y[tr_idx].mean(axis=0, keepdims=True)
        hp = {"n_vocab": len(vocab), "emb_dim": 64, "hidden": 512,
              "epochs": hp_defaults["epochs"], "batch_size": hp_defaults["batch_size"],
              "lr": 1e-3, "wd": 1e-4}

        preds = {
            "null_zero": np.zeros((len(te_idx), Y.shape[1]), dtype=np.float32),
            "global_mean_signature": np.repeat(y_mu, len(te_idx), axis=0).astype(np.float32),
        }
        for arm, kind in [("ctrl_only", "ctrl_only"),
                          ("ctrl_drug_id", "ctrl_drug_id"),
                          ("drug_only", "drug_only")]:
            print(f"[{proto['name']}] retraining {arm} (seed={seed}) ...", flush=True)
            pred, _ = exp023.run_arm(arm, kind, tr_idx, te_idx, Xz, Y, drug,
                                     feat_morgan, device, hp, seed)
            preds[arm] = pred

        tr_sm = tr_idx[encodable[tr_idx]]
        te_sm = te_idx[encodable[te_idx]]
        print(f"[{proto['name']}] retraining ctrl_morgan on encodable subset ...", flush=True)
        pred_m, _ = exp023.run_arm("ctrl_morgan", "ctrl_morgan", tr_sm, te_sm, Xz, Y,
                                   drug, feat_morgan, device, hp, seed)

        # ---- 打分：cosine（复现校验）+ Spearman（本次补充） ----
        Y_te = Y[te_idx]
        arm_stats = {}
        for arm, pred in preds.items():
            cos = exp023.score_rows(pred, Y_te)
            spr = spearman_rows(pred, Y_te)
            arm_stats[arm] = {
                "n_eval": int(len(te_idx)),
                "mean_cosine_reproduced": float(cos.mean()),
                "mean_cosine_original": orig_cos.get((proto["name"], arm)),
                "mean_spearman": float(spr.mean()),
            }
            for i_local, i_global in enumerate(te_idx):
                per_response_rows.append({
                    "protocol": proto["name"], "arm": arm,
                    "Response_ID": table["Response_ID"].iloc[i_global],
                    "Chemical_ID": chem_ids[i_global],
                    "spearman": float(spr[i_local]),
                    "cosine_reproduced": float(cos[i_local]),
                })
        cos_m = exp023.score_rows(pred_m, Y[te_sm])
        spr_m = spearman_rows(pred_m, Y[te_sm])
        arm_stats["ctrl_morgan"] = {
            "n_eval": int(len(te_sm)),
            "mean_cosine_reproduced": float(cos_m.mean()),
            "mean_cosine_original": orig_cos.get((proto["name"], "ctrl_morgan")),
            "mean_spearman": float(spr_m.mean()),
        }
        for k_local, k_global in enumerate(te_sm):
            per_response_rows.append({
                "protocol": proto["name"], "arm": "ctrl_morgan",
                "Response_ID": table["Response_ID"].iloc[k_global],
                "Chemical_ID": chem_ids[k_global],
                "spearman": float(spr_m[k_local]),
                "cosine_reproduced": float(cos_m[k_local]),
            })

        # ---- Spearman 口径的配对差（化合物分组 bootstrap，同原协议） ----
        deltas = {}
        spr_full = {a: spearman_rows(p, Y_te) for a, p in preds.items()}
        for arm in ["ctrl_drug_id", "drug_only"]:
            per_row = spr_full[arm] - spr_full["ctrl_only"]
            ci = exp023.grouped_bootstrap_ci(per_row, chem_ids[te_idx], 2000, seed)
            deltas[arm] = {"scope": "full_test", **ci}
        spr_ctrl_full = spr_full["ctrl_only"]
        te_sm_set = set(te_sm.tolist())
        ctrl_spr_on_sm = np.array([spr_ctrl_full[k] for k in range(len(te_idx))
                                   if te_idx[k] in te_sm_set])
        per_row_m = spr_m - ctrl_spr_on_sm
        ci_m = exp023.grouped_bootstrap_ci(per_row_m, chem_ids[te_sm], 2000, seed)
        deltas["ctrl_morgan"] = {"scope": "smiles_encodable_test_subset", **ci_m}

        out_protocols.append({
            "protocol": proto["name"],
            "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
            "n_morgan_test": int(len(te_sm)),
            "arm_stats": arm_stats,
            "spearman_paired_delta_vs_ctrl_only_95CI": deltas,
        })
        print(f"[{proto['name']}] " + " | ".join(
            f"{a}: cos={v['mean_cosine_reproduced']:.4f}"
            f"(orig {v['mean_cosine_original']:.4f})"
            f" spearman={v['mean_spearman']:.4f}"
            for a, v in arm_stats.items()), flush=True)
        for arm, ci in deltas.items():
            print(f"  spearman delta[{arm}] = {ci['delta_mean']:.4f} "
                  f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]", flush=True)

    pd.DataFrame(per_response_rows).to_parquet(
        OUT_DIR / "chempert_per_response_spearman.parquet", index=False)
    out = {
        "status": "SUPPLEMENT_DESCRIPTIVE",
        "purpose": "为 EXP-023 补充每条响应的基因级 Spearman 排序指标（事后描述性，"
                   "不改判定；同种子同超参重训，逐基因预测原未落盘）",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "seed": seed,
        "protocols": out_protocols,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT_DIR / "chempert_spearman_supplement.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[supp] done elapsed={out['elapsed_sec']}s -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
