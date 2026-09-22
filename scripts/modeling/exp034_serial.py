"""EXP-034 E1 串联架构执行（工作包 E 预登记）。

只写 artifacts/exp034_serial/：
- predictions.parquet（统一 schema 预测）
- summary.json、report.md

口径以 EXP-034.md 预登记 + 工作包 B 合同（exp031_eval）为准：
- 主指标：患者内全药排序 Spearman 患者等权宏平均（≥3 药资格，不填零）。
- 扰动表示 = whitelist_chempert_features 3 种子均值
  （reversal_score / pathway_reversal / perturb_magnitude，3 标量）。不附加
  Hallmark 50 维通路向量：frozen parquet 未含该向量，重跑上游不属本实验范围，
  属预登记"任选其一"口径内的选择。
- 药敏头输入 = 药指纹(Morgan 2048) + 扰动表示(3)。冻结扰动模型 = 仅用其
  零样本输出、不更新其权重（等价表述）。
- 四臂（预登记对照，全做）：
  ① drug_only（不使用扰动）
  ② perturb（使用预测扰动）
  ③ shuffled（折内按样本打乱扰动表示，3 次打乱）
  ④ learned（维度/容量相近的普通学习特征 = 表达 PCA 3 维，训练折内拟合）
- 第二层训练机制：药敏头超参（≤20 候选）在训练折内层 3 折上选择，选定的
  同一头结构/超参四臂共用（仅输入特征不同）；头在各外层折 train 患者训练、
  test 患者预测；3 初始化种子。
- 联合微调臂：不适用（扰动模型为外部冻结数组，无上游可调参数），如实记录。
- 测试折只参与评测；全部学习型预处理（含表达 PCA）只在训练行 fit。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from torch import nn
from torch.nn import functional as F

from drug_screen.transfer.cohorts import load_registered_cohort
from drug_screen.transfer.benchmark import _table
from drug_screen.transfer.preprocessing import ExpressionPreprocessor
from drug_screen.transfer.chemistry import morgan_fingerprint
from drug_screen.data.admission import AdmissionWhitelist
from drug_screen.evaluation.exp031_eval import (
    intra_patient_spearman_patient_macro,
    paired_bootstrap_ci,
    holm_correct,
)

PROJ = Path("/mnt/d/Code/DrugScreenLab")
OUT = PROJ / "artifacts" / "exp034_serial"
AUDIT = json.loads((PROJ / "data/processed/identity/kryzei_admission_audit_20260921.json").read_text(encoding="utf-8"))
SPLITS = json.loads((PROJ / "data/splits/kryzei_shared_splits_v1.json").read_text(encoding="utf-8"))
SOURCE_DIR = PROJ / "data/artifacts/EXP-015/source_seed20260912"
FEAT_PATH = PROJ / "artifacts/exp033_routeB/whitelist_chempert_features.parquet"

EXP_ID = "EXP-034"
CONTRACT_ID = "exp034_serial_contract"
SPLIT_ID = "kryzei_shared_v1"
DEVICE = "cuda"

BOOT_SEED = 20260918
N_BOOT = 2000
INIT_SEEDS = [20260921, 20260922, 20260923]
SHUFFLE_SEEDS = [20260921, 20260922, 20260923]
EPOCHS = 60
CANDIDATES = [
    dict(hidden=16, lr=1e-3, wd=0.0),
    dict(hidden=16, lr=1e-3, wd=1e-4),
    dict(hidden=16, lr=3e-4, wd=0.0),
    dict(hidden=32, lr=1e-3, wd=0.0),
    dict(hidden=32, lr=1e-3, wd=1e-4),
    dict(hidden=32, lr=3e-4, wd=0.0),
    dict(hidden=64, lr=1e-3, wd=0.0),
    dict(hidden=64, lr=3e-4, wd=0.0),
    dict(hidden=64, lr=3e-4, wd=1e-4),
]
ARMS = ["drug_only", "perturb", "shuffled", "learned"]


def load_all():
    bundle = load_registered_cohort(
        "kryeziu_mcrc_v3", root=PROJ,
        registry=PROJ / "data/registry/transfer_cohorts.json")
    table = _table(bundle)
    sp = dict(zip(bundle.samples.sample_id.astype(str), bundle.samples.patient_id.astype(str)))
    wl = AdmissionWhitelist(AUDIT["patient_whitelist"], AUDIT["pdo_whitelist"],
                            patient_of_sample=sp)
    admitted, rejected = wl.reject_non_admitted(
        table.sample_id.astype(str).tolist(), patient_of_sample=sp)
    rows = table[table.sample_id.astype(str).isin(set(admitted))].copy()
    for c in ("patient_id", "model_id", "drug_id", "sample_id"):
        rows[c] = rows[c].astype(str).str.strip()
    rows["observed"] = pd.to_numeric(rows.response, errors="raise").astype(float)
    # 稳定行序：按 (sample_id, drug_id) 排序，保证折切片可复现
    rows = rows.sort_values(["sample_id", "drug_id"], kind="mergesort").reset_index(drop=True)

    expression = np.asarray(bundle.expression, dtype=float)
    genes = np.asarray(bundle.gene_ids, dtype=str)

    chem = bundle.drugs.copy()
    field = "canonical_isomeric_smiles"
    lookup = dict(zip(chem.drug_id.astype(str), chem[field], strict=True))
    fps = {str(d): morgan_fingerprint(str(lookup[str(d)]), radius=2, n_bits=2048).astype(float)
           for d in rows.drug_id.unique()}

    source_prep = ExpressionPreprocessor.load(SOURCE_DIR / "expression_transform.npz")
    source_x = source_prep.transform(expression, genes)

    # 扰动表示：3 种子均值（3 标量）
    feat = pd.read_parquet(FEAT_PATH)
    feat = feat.groupby(["sample_id", "drug_id"], sort=True, as_index=False)[
        ["reversal_score", "pathway_reversal", "perturb_magnitude"]].mean()
    feat.columns = ["sample_id", "drug_id", "reversal", "pathway_rev", "perturb_mag"]
    feat["sample_id"] = feat["sample_id"].astype(str).str.strip()
    feat["drug_id"] = feat["drug_id"].astype(str).str.strip()
    P_lookup = {(s, d): np.array([r, p, m], dtype=float)
                for s, d, r, p, m in zip(feat.sample_id, feat.drug_id,
                                          feat.reversal, feat.pathway_rev, feat.perturb_mag)}

    return dict(bundle=bundle, rows=rows, expression=expression, genes=genes,
                fps=fps, source_prep=source_prep, source_x=source_x, feat=feat,
                P_lookup=P_lookup, rejected=rejected)


def fold_map():
    fm = {}
    for f in SPLITS["outer_folds"]:
        fm[int(f["fold"])] = {
            "test_patients": sorted(f["test_patient_ids"]),
            "train_patients": sorted(f["train_patient_ids"]),
            "inner_folds": [
                {"inner_train": sorted(inf["inner_train_patient_ids"]),
                 "inner_val": sorted(inf["inner_val_patient_ids"])}
                for inf in f["inner_folds"]
            ],
        }
    return fm


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------

def _agg_patient_drug(frame):
    f = frame[["patient_id", "drug_id", "observed", "predicted"]].copy()
    f["observed"] = pd.to_numeric(f["observed"], errors="raise").astype(float)
    f["predicted"] = pd.to_numeric(f["predicted"], errors="raise").astype(float)
    return f.groupby(["patient_id", "drug_id"], sort=True,
                     as_index=False).agg(observed=("observed", "mean"),
                                         predicted=("predicted", "mean"))


def _sp(y, p, min_drugs=3):
    y = np.asarray(y, float); p = np.asarray(p, float)
    if len(y) < min_drugs or np.ptp(y) == 0 or np.ptp(p) == 0:
        return np.nan
    r = spearmanr(y, p).statistic
    return float(r) if np.isfinite(r) else np.nan


def macro_bootstrap(per_patient_vals, seed=BOOT_SEED, n=N_BOOT):
    vals = np.asarray([v for v in per_patient_vals if np.isfinite(v)], float)
    if len(vals) == 0:
        return {"point_estimate": None, "low": None, "high": None, "n_units": 0}
    rng = np.random.default_rng(seed)
    boot = vals[rng.integers(0, len(vals), size=(n, len(vals)))].mean(axis=1)
    return {"point_estimate": float(vals.mean()),
            "low": float(np.quantile(boot, 0.025)),
            "high": float(np.quantile(boot, 0.975)),
            "n_units": int(len(vals)), "n": n, "seed": seed}


def arm_eval(frame, *, min_drugs=3):
    macro, n_valid, per_patient = intra_patient_spearman_patient_macro(
        frame, min_drugs=min_drugs)
    agg = _agg_patient_drug(frame)

    drug_rhos = []
    for _, g in agg.groupby("drug_id", sort=True):
        r = _sp(g.observed.to_numpy(float), g.predicted.to_numpy(float))
        if np.isfinite(r):
            drug_rhos.append(float(r))
    per_drug = float(np.mean(drug_rhos)) if drug_rhos else None

    topk = {}
    for k in (1, 2, 3):
        vals = []
        for _, g in agg.groupby("patient_id", sort=True):
            y = g.observed.to_numpy(float); p = g.predicted.to_numpy(float)
            if len(y) >= k and np.ptp(y) > 0 and np.ptp(p) > 0:
                top_y = set(np.argsort(-y, kind="mergesort")[:k])
                top_p = set(np.argsort(-p, kind="mergesort")[:k])
                vals.append(len(top_y & top_p) / k)
        topk[k] = float(np.mean(vals)) if vals else None

    ndcgs = []
    for _, g in agg.groupby("patient_id", sort=True):
        y = g.observed.to_numpy(float); p = g.predicted.to_numpy(float)
        if len(y) >= 10 and np.ptp(y) > 0 and np.ptp(p) > 0:
            v = _ndcg_at_k(y, p, 5)
            if v is not None and np.isfinite(v):
                ndcgs.append(float(v))
    ndcg5 = float(np.mean(ndcgs)) if ndcgs else None

    pv = per_patient["intra_patient_spearman"].to_numpy(float)
    return {
        "intra_patient_spearman": macro,
        "intra_patient_spearman_ci": macro_bootstrap(pv),
        "n_valid_patients": int(n_valid),
        "n_patients": int(frame.patient_id.nunique()),
        "n_drugs": int(frame.drug_id.nunique()),
        "per_drug_spearman": per_drug,
        "ndcg5": ndcg5,
        "top1": topk[1], "top2": topk[2], "top3": topk[3],
        "reason_counts": {str(k): int(v) for k, v in
                          per_patient.loc[per_patient.reason.notna(), "reason"].value_counts().items()},
        "_per_patient_vals": [float(v) if np.isfinite(v) else float("nan") for v in pv],
    }


def _ndcg_at_k(observed, predicted, k):
    from scipy.stats import rankdata
    y = np.asarray(observed, float); p = np.asarray(predicted, float)
    if len(y) < k or np.ptp(y) == 0:
        return np.nan
    gains = (rankdata(y, method="average") - 1) / (len(y) - 1)
    discounts = 1 / np.log2(np.arange(2, k + 2))
    ideal = np.sort(gains)[::-1][:k] @ discounts
    total, offset = 0.0, 0
    for score in np.unique(p)[::-1]:
        mask = p == score
        count = int(mask.sum())
        total += gains[mask].mean() * discounts[offset:min(offset + count, k)].sum()
        offset += count
        if offset >= k:
            break
    return float(total / ideal) if ideal > 0 else np.nan


def paired_contrast(a, b, *, seed=BOOT_SEED, n=N_BOOT):
    A = _agg_patient_drug(a); B = _agg_patient_drug(b)
    common = sorted(set(A.patient_id) & set(B.patient_id))
    valid = []
    for p in common:
        ga = A[A.patient_id == p]; gb = B[B.patient_id == p]
        if np.isfinite(_sp(ga.observed.to_numpy(float), ga.predicted.to_numpy(float))) and \
           np.isfinite(_sp(gb.observed.to_numpy(float), gb.predicted.to_numpy(float))):
            valid.append(p)
    if len(valid) < 2:
        return {"point_estimate": None, "low": None, "high": None,
                "n_units": len(valid), "n": n,
                "note": "fewer than 2 comparable patients"}
    Av = A[A.patient_id.isin(valid)].copy()
    Bv = B[B.patient_id.isin(valid)].copy()
    ci = paired_bootstrap_ci(Av, Bv, "patient_id", _sp, n=n, seed=seed,
                             truth_col="observed", pred_col="predicted")
    deltas = np.asarray(ci.deltas, float)
    rng = np.random.default_rng(seed)
    boots = deltas[rng.integers(0, len(deltas), size=(n, len(deltas)))].mean(axis=1)
    p_greater = float(np.mean(boots <= 0))
    p_two = float(min(1.0, 2.0 * min(p_greater, 1.0 - p_greater)))
    return {"point_estimate": ci.point_estimate, "low": ci.low, "high": ci.high,
            "n_units": ci.n_units, "n": n, "seed": seed,
            "p_greater_than_zero": round(1.0 - p_greater, 6),
            "p_two_sided": round(p_two, 6)}


# ---------------------------------------------------------------------------
# 药敏头
# ---------------------------------------------------------------------------

class _SensHead(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def fit_head(X, y, *, seed, hidden, lr, wd, epochs=EPOCHS):
    torch.manual_seed(int(seed))
    model = _SensHead(X.shape[1], hidden).to(DEVICE)
    mean = float(np.mean(y)); std = float(np.std(y)) or 1.0
    xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor((y - mean) / std, dtype=torch.float32, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(int(epochs)):
        model.train(); opt.zero_grad()
        loss = F.huber_loss(model(xt), yt); loss.backward(); opt.step()
    model.eval()
    return model, mean, std


def predict_head(model, mean, std, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32, device=DEVICE)).cpu().numpy() * std + mean


# ---------------------------------------------------------------------------
# 特征
# ---------------------------------------------------------------------------

def build_F(data, sub):
    """返回 F(N,2048)，行序与 sub 一致。"""
    fps = data["fps"]
    return np.vstack([fps[str(d)] for d in sub.drug_id])


def build_P(data, sub, sample_map=None):
    """返回 P(N,3)，行序与 sub 一致；sample_map 可重排样本（打乱对照）。"""
    P = data["P_lookup"]
    out = []
    for s, d in zip(sub.sample_id.tolist(), sub.drug_id.tolist()):
        ss = s if sample_map is None else sample_map[s]
        out.append(P[(ss, d)])
    return np.vstack(out)


def pca3_fit_transform(data, sub, train_mask):
    """表达 PCA 3 维（512 源预处理表达），训练行 fit，全行 transform。"""
    src_x = data["source_x"]
    idx = sub.expression_index.to_numpy(int)
    X = src_x[idx]
    tr = X[train_mask]
    mean = tr.mean(axis=0)
    centered = tr - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    comps = vt[:3]
    return (X - mean) @ comps.T


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_all()
    fm = fold_map()
    rows = data["rows"]

    all_F = build_F(data, rows)
    all_P = build_P(data, rows)

    pred_rows = []
    meta = []
    selected_configs = {}

    uniq_samples = sorted(rows.sample_id.unique().tolist())
    sample_to_idx = {s: i for i, s in enumerate(uniq_samples)}

    for fold in sorted(fm):
        finfo = fm[fold]
        train_pat = set(finfo["train_patients"])
        test_pat = set(finfo["test_patients"])
        tr_mask = rows.patient_id.isin(train_pat).to_numpy()
        te_mask = rows.patient_id.isin(test_pat).to_numpy()
        tr_idx = np.flatnonzero(tr_mask)
        te_idx = np.flatnonzero(te_mask)
        tr = rows.iloc[tr_idx].reset_index(drop=True)
        te = rows.iloc[te_idx].reset_index(drop=True)
        F_tr, P_tr = all_F[tr_idx], all_P[tr_idx]
        F_te, P_te = all_F[te_idx], all_P[te_idx]
        y_tr = tr.observed.to_numpy(float)

        # ---- 内层选择（在"使用预测扰动"臂上，1 种子）----
        inner_scores = {tuple(sorted(c.items())): [] for c in CANDIDATES}
        for inf in finfo["inner_folds"]:
            in_tr_pat = set(inf["inner_train"])
            in_val_pat = set(inf["inner_val"])
            in_tr_mask = tr.patient_id.isin(in_tr_pat).to_numpy()
            in_val_mask = tr.patient_id.isin(in_val_pat).to_numpy()
            in_tr_i = np.flatnonzero(in_tr_mask)
            in_val_i = np.flatnonzero(in_val_mask)
            Xitr = np.hstack([F_tr[in_tr_i], P_tr[in_tr_i]])
            Xiva = np.hstack([F_tr[in_val_i], P_tr[in_val_i]])
            yitr = y_tr[in_tr_i]
            val_frame = tr.iloc[in_val_i][["patient_id", "drug_id"]].copy()
            val_frame["observed"] = y_tr[in_val_i]
            for cand in CANDIDATES:
                key = tuple(sorted(cand.items()))
                m, mu, sd = fit_head(Xitr, yitr, seed=INIT_SEEDS[0], **cand)
                val_frame["predicted"] = predict_head(m, mu, sd, Xiva)
                macro, _, _ = intra_patient_spearman_patient_macro(val_frame)
                inner_scores[key].append(macro if macro is not None else -1.0)
        cfg = dict(min(inner_scores, key=lambda k: (-np.mean(inner_scores[k]), k)))
        selected_configs[fold] = cfg

        # ---- 最终：训练 3 种子头，预测 test ----
        P_pca = pca3_fit_transform(data, rows, tr_mask)

        for seed in INIT_SEEDS:
            for arm, Xtr, Xte in [
                ("drug_only", F_tr, F_te),
                ("perturb", np.hstack([F_tr, P_tr]), np.hstack([F_te, P_te])),
                ("learned", np.hstack([F_tr, P_pca[tr_idx]]), np.hstack([F_te, P_pca[te_idx]])),
            ]:
                m, mu, sd = fit_head(Xtr, y_tr, seed=seed, **cfg)
                pred = predict_head(m, mu, sd, Xte)
                _emit(pred_rows, fold, seed, arm, te, pred)

            # ③ shuffled（3 次打乱 × 3 种子）
            for sh in SHUFFLE_SEEDS:
                rng = np.random.default_rng(sh)
                s_perm = rng.permutation(len(uniq_samples))
                sample_map = {s: uniq_samples[int(s_perm[sample_to_idx[s]])] for s in uniq_samples}
                P_sh = build_P(data, rows, sample_map=sample_map)
                m, mu, sd = fit_head(np.hstack([F_tr, P_sh[tr_idx]]), y_tr, seed=seed, **cfg)
                pred = predict_head(m, mu, sd, np.hstack([F_te, P_sh[te_idx]]))
                _emit(pred_rows, fold, seed, "shuffled", te, pred, shuffle=sh)

        meta.append({"fold": fold, "n_train_rows": int(len(tr)), "n_test_rows": int(len(te)),
                     "selected_config": cfg})

    schema = pd.DataFrame(pred_rows)
    schema.to_parquet(OUT / "predictions.parquet", index=False)

    summary = build_summary(schema, meta, selected_configs, time.perf_counter() - t0)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    (OUT / "report.md").write_text(build_report(summary), encoding="utf-8")
    print("WROTE exp034_serial summary.json + predictions.parquet + report.md")
    print(json.dumps({a: {k: summary["arms"][a][k] for k in
                          ("intra_patient_spearman_mean", "intra_patient_spearman_std",
                           "per_drug_spearman", "ndcg5")} for a in ARMS},
                     indent=2, ensure_ascii=False))
    return summary


def _emit(pred_rows, fold, seed, arm, te, pred, shuffle=None):
    for i, r in te.iterrows():
        pred_rows.append({
            "experiment_id": EXP_ID, "run_id": f"exp034_serial_{arm}",
            "contract_id": CONTRACT_ID, "split_id": SPLIT_ID,
            "fold": int(fold), "seed": int(seed),
            "patient_id": r["patient_id"], "model_id": r["model_id"],
            "sample_id": r["sample_id"], "drug_id": r["drug_id"],
            "gene_id": None, "y_true": float(r["observed"]),
            "y_pred": None if not np.isfinite(pred[i]) else float(pred[i]),
            "evaluation_role": "test", "arm": arm,
            "shuffle": None if shuffle is None else int(shuffle),
        })


def _frame_for_arm(schema, arm, seed=None, shuffle=None):
    f = schema[schema["arm"] == arm].copy()
    if seed is not None:
        f = f[f["seed"] == int(seed)]
    if shuffle is not None:
        f = f[f["shuffle"] == int(shuffle)]
    f = f.rename(columns={"y_true": "observed", "y_pred": "predicted"})
    for c in ("observed", "predicted"):
        f[c] = pd.to_numeric(f[c], errors="coerce").astype(float)
    return f


def build_summary(schema, meta, selected_configs, elapsed):
    arms = {}
    per_seed = {}
    for arm in ARMS:
        seed_evals = []
        seed_pv = {}
        for seed in INIT_SEEDS:
            if arm == "shuffled":
                f = pd.concat([_frame_for_arm(schema, arm, seed=seed, shuffle=sh)
                               for sh in SHUFFLE_SEEDS], ignore_index=True)
            else:
                f = _frame_for_arm(schema, arm, seed=seed)
            e = arm_eval(f)
            seed_evals.append(e)
            seed_pv[seed] = e["_per_patient_vals"]
        pvs = np.asarray([np.asarray(seed_pv[s], float) for s in INIT_SEEDS], float)
        patient_mean = np.nanmean(pvs, axis=0)
        ci = macro_bootstrap(patient_mean)
        pts = [e["intra_patient_spearman"] for e in seed_evals if e["intra_patient_spearman"] is not None]
        arms[arm] = {
            "intra_patient_spearman_mean": float(np.mean(pts)) if pts else None,
            "intra_patient_spearman_std": float(np.std(pts)) if len(pts) > 1 else 0.0,
            "per_seed": pts,
            "intra_patient_spearman_ci_seed_avg": ci,
            "per_drug_spearman": seed_evals[0]["per_drug_spearman"],
            "ndcg5": seed_evals[0]["ndcg5"],
            "top1": seed_evals[0]["top1"], "top2": seed_evals[0]["top2"], "top3": seed_evals[0]["top3"],
            "n_valid_patients": seed_evals[0]["n_valid_patients"],
            "n_patients": seed_evals[0]["n_patients"],
            "n_drugs": seed_evals[0]["n_drugs"],
        }
        per_seed[arm] = {str(s): {k: v for k, v in e.items() if not k.startswith("_")}
                         for s, e in zip(INIT_SEEDS, seed_evals)}

    def seed_avg_frame(arm):
        frs = []
        for seed in INIT_SEEDS:
            if arm == "shuffled":
                frs.append(pd.concat([_frame_for_arm(schema, arm, seed=seed, shuffle=sh)
                                      for sh in SHUFFLE_SEEDS], ignore_index=True))
            else:
                frs.append(_frame_for_arm(schema, arm, seed=seed))
        conc = pd.concat(frs, ignore_index=True)
        return conc.groupby(["patient_id", "drug_id"], sort=True, as_index=False).agg(
            observed=("observed", "mean"), predicted=("predicted", "mean"))

    base = seed_avg_frame("drug_only")
    pvals = []
    paired = {}
    comparators = ["perturb", "shuffled", "learned"]
    for arm in comparators:
        pb = paired_contrast(seed_avg_frame(arm), base)
        paired[f"{arm}_minus_drug_only"] = {
            "supervision": "both target-label (same fold train subset)",
            "metric": "patient_macro_intra_patient_spearman (seed-averaged patient level)",
            **pb}
        if pb.get("p_two_sided") is not None:
            pvals.append(pb["p_two_sided"])
    paired["holm_p"] = holm_correct(pvals) if pvals else None
    paired["holm_applied_over"] = [f"{a}_minus_drug_only" for a in comparators]
    paired["note"] = ("主比较集 = 预测扰动/打乱/维度匹配 learned 各自的增量相对 drug_only "
                      "（patient 级 seed 平均），Holm 校正。判定链：主臂 CI 不跨零 + "
                      "打乱与维度匹配对照无同量级增益。")

    return {
        "experiment_id": EXP_ID, "architecture": "E1_serial", "status": "RAN",
        "preregistration": {
            "perturbation_representation": "whitelist_chempert_features 3-seed mean "
                                           "(reversal_score/pathway_reversal/perturb_magnitude = 3 scalars)",
            "hallmark_vector": "not_used (frozen parquet lacks 50-dim pathway vector)",
            "frozen_perturb_model": "external frozen array; zero-shot outputs only, "
                                    "no upstream weights updated",
            "arms": ARMS,
            "learned_control": "expression PCA 3-dim (512 source-preprocessed), fit on train fold",
            "shuffle_control": "3 permutations of perturbation across samples (fixed seeds)",
            "head_selection": "inner 3-fold over <=20 candidates on 'perturb' arm; "
                              "same head config shared by all 4 arms",
            "init_seeds": INIT_SEEDS, "shuffle_seeds": SHUFFLE_SEEDS,
        },
        "joint_finetune_arm": {
            "applicable": False,
            "reason": "perturbation model is an external frozen array; "
                      "no upstream trainable parameters to jointly fine-tune. "
                      "Joint-finetuning is not applicable to this representation."},
        "arms": arms,
        "per_seed_arms": per_seed,
        "paired_gain": paired,
        "selected_configs_per_fold": selected_configs,
        "meta": meta,
        "elapsed_seconds": elapsed,
    }


def build_report(summary):
    L = []
    L.append("# EXP-034 E1 串联架构报告\n\n")
    L.append(f"- status: {summary['status']}（工作包 E 预登记执行）\n")
    L.append("- 扰动表示：whitelist_chempert_features 3 种子均值（3 标量）。")
    L.append("  Hallmark 50 维通路向量未使用（frozen parquet 未含，重跑上游不属本实验范围；属预登记『任选其一』口径内）。\n")
    L.append("- 冻结扰动模型 = 外部冻结数组零样本输出，无上游可调参数 → **联合微调臂不适用**（如实记录）。\n")
    L.append("\n## 四臂指标（种子 3，患者级 seed 平均）\n\n")
    L.append("| 臂 | 主指标(均值±std) | 逐药 Spearman | NDCG@5 | top1/top2/top3 |\n|---|---|---|---|---|\n")
    for arm, a in summary["arms"].items():
        L.append(f"| {arm} | {a['intra_patient_spearman_mean']:.4f}±{a['intra_patient_spearman_std']:.4f} | "
                 f"{_fmt(a['per_drug_spearman'])} | {_fmt(a['ndcg5'])} | "
                 f"{_fmt(a['top1'])}/{_fmt(a['top2'])}/{_fmt(a['top3'])} |\n")
    L.append("\n## 配对增益（vs drug_only，患者级 2000 bootstrap + Holm）\n\n")
    L.append("| 比较 | 点估计 | 95% CI | p_two |\n|---|---|---|---|\n")
    pg = summary["paired_gain"]
    for key in pg["holm_applied_over"]:
        v = pg[key]
        L.append(f"| {key} | {_fmt(v['point_estimate'])} | [{_fmt(v['low'])}, {_fmt(v['high'])}] | "
                 f"{_fmt(v.get('p_two_sided'))} |\n")
    L.append(f"\nHolm 校正 p：{pg['holm_p']}\n")
    L.append("\n## 判定建议\n\n")
    main = pg["perturb_minus_drug_only"]
    shuf = pg["shuffled_minus_drug_only"]
    learn = pg["learned_minus_drug_only"]
    low, high = main["low"], main["high"]
    sig_main = (low is not None and high is not None and low > 0)
    def gain_mag(v):
        return v["point_estimate"] if v["point_estimate"] is not None else float("-inf")
    shuf_no_gain = gain_mag(shuf) < gain_mag(main) - 1e-9
    learn_no_gain = gain_mag(learn) < gain_mag(main) - 1e-9
    if sig_main and shuf_no_gain and learn_no_gain:
        verdict = "支持（在公平对照下扰动表示提供可归因增量）"
    elif (low is not None and high is not None and low <= 0 <= high):
        verdict = "证据不足（主臂 CI 跨零）"
    else:
        verdict = "不支持（主臂无显著正向增量或对照臂出现同量级增益）"
    L.append(f"- 主臂（perturb vs drug_only）：点估计 {_fmt(main['point_estimate'])}，"
             f"CI [{_fmt(low)}, {_fmt(high)}]。\n")
    L.append(f"- 打乱对照增益 {_fmt(shuf['point_estimate'])}，维度匹配对照增益 {_fmt(learn['point_estimate'])}。\n")
    L.append(f"- **判定：{verdict}**\n")
    L.append("\n## unknown / 限制\n\n")
    L.append("- 实测扰动参考：Kryzei 无同样本同处理条件实测扰动，无法检验该项参考。\n")
    L.append("- 扰动表示仅 3 标量；未含 50 维通路方向向量。\n")
    L.append("- 联合微调臂不适用（外部冻结数组）。\n")
    return "".join(L)


def _fmt(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "None"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


if __name__ == "__main__":
    main()