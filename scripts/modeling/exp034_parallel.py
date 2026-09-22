"""EXP-034 工作包 E2 并联架构（含指纹/扰动增量四组合分离）。

按 EXP-034.md 预登记执行。写 artifacts/exp034_parallel/：
- predictions.parquet（统一 schema，各臂测试折预测）
- arm_summary.json（8 臂主指标 + 三增量 + 打乱/容量对照）
- increment_report.md

口径：
- 直接分支 = 冻结源预处理器 512 表达；扰动分支 = whitelist 扰动特征
  （3 种子均值，3 标量：reversal_score / pathway_reversal / perturb_magnitude；
  白名单产物仅含标量，通路方向由 pathway_reversal 标量承载）。
- 四组合分离：臂1 纯表达、臂2 +指纹(PCA<=64)、臂3 +扰动、臂4 +指纹+扰动；
  指纹增量=臂2−臂1、扰动增量=臂3−臂1、联合增量=臂4−臂1。
- 臂1（纯表达，药物不变）患者内排序 Spearman 为常值→未定义；故三增量主比较
  在逐药跨患者 Spearman（个体化指标，全体臂均有定义）上进行，患者级 2000 次
  bootstrap + Holm。
- 测试折只参与预测；全部拟合/选模仅在训练折（内层 3 折来自冻结划分）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
import torch.nn as nn
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
OUT = PROJ / "artifacts" / "exp034_parallel"
AUDIT = json.loads((PROJ / "data/processed/identity/kryzei_admission_audit_20260921.json").read_text(encoding="utf-8"))
SPLITS = json.loads((PROJ / "data/splits/kryzei_shared_splits_v1.json").read_text(encoding="utf-8"))
SOURCE_DIR = PROJ / "data/artifacts/EXP-015/source_seed20260912"
PERT_PARQUET = PROJ / "artifacts/exp033_routeB/whitelist_chempert_features.parquet"

EXP_ID = "EXP-034"
CONTRACT_ID = "exp034_parallel_contract"
SPLIT_ID = "kryzei_shared_v1"
DEVICE = "cuda"

INIT_SEEDS = [20260921, 20260922, 20260923]
SEED_BASE = 20260921   # 共享表达分支初始化种子（= 划分 model_init_seed）
BOOT_SEED = 20260918
N_BOOT = 2000

EXPR_DIM = 512
FP_BITS = 2048
FP_PCA = 64
PERT_DIM = 3

EPOCHS = 150
LR = 1e-3
HIDDEN_CAND = [16, 32, 64]
WD_CAND = [1e-4, 1e-3, 1e-2]
GATE_RANK_CAND = [2, 4, 8]
DUAL_HD = 64
DUAL_HP = 16
DUAL_HD2 = 64
CAP_H = 64
CAP_H2 = 64

MIN_DRUGS = 3
MIN_PATIENTS_PER_DRUG = 5

ARM_NAMES = ["direct", "direct_fp", "direct_pert", "direct_fp_pert",
             "pert_only", "dual", "dual_gated", "dual_pert_shuffle", "capacity"]


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

class ParallelNet(nn.Module):
    """并联可加模型：h = ReLU(enc(expr)) [+ ReLU(drug(drug_feat))] -> head。

    臂 1~4 共享 enc/head（同初始化种子），仅 drug 分支不同（可加并联）。
    """

    def __init__(self, enc, head, drug_in, hidden, drug_seed):
        super().__init__()
        self.encoder = enc
        self.head = head
        if drug_in > 0:
            torch.manual_seed(drug_seed)
            self.drug = nn.Linear(drug_in, hidden)
        else:
            self.drug = None

    def forward(self, expr, drug=None):
        h = F.relu(self.encoder(expr))
        if self.drug is not None and drug is not None:
            h = h + F.relu(self.drug(drug))
        return self.head(h).squeeze(-1)


class DualBranchNet(nn.Module):
    """双分支：直接 [expr,fp]、扰动 [pert] concat -> head；gate_rank>0 时扰动经 low-rank 门控。"""

    def __init__(self, expr_dim, fp_dim, pert_dim, hd, hp, hd2, gate_rank=0):
        super().__init__()
        self.direct = nn.Linear(expr_dim + fp_dim, hd)
        self.pert = nn.Linear(pert_dim, hp)
        if gate_rank > 0:
            self.gate_a = nn.Linear(hd, gate_rank)
            self.gate_b = nn.Linear(gate_rank, hp)
        else:
            self.gate_a = self.gate_b = None
        self.head = nn.Sequential(nn.Linear(hd + hp, hd2), nn.ReLU(), nn.Linear(hd2, 1))

    def forward(self, expr, fp, pert):
        dh = F.relu(self.direct(torch.cat([expr, fp], dim=-1)))
        ph = F.relu(self.pert(pert))
        if self.gate_a is not None:
            ph = ph * torch.sigmoid(self.gate_b(self.gate_a(dh)))
        return self.head(torch.cat([dh, ph], dim=-1)).squeeze(-1)


class CapacityNet(nn.Module):
    """容量相近直接模型：仅 [expr,fp]，两层隐藏大到与双分支参数量相近。"""

    def __init__(self, expr_dim, fp_dim, h, h2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(expr_dim + fp_dim, h), nn.ReLU(),
                                 nn.Linear(h, h2), nn.ReLU(), nn.Linear(h2, 1))

    def forward(self, expr, fp):
        return self.net(torch.cat([expr, fp], dim=-1)).squeeze(-1)


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------------------------
# 载入
# ---------------------------------------------------------------------------

def _fit_pca(X, k):
    mean = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
    return mean, Vt[:k]


def _apply_pca(X, mean, comps):
    return (X - mean) @ comps.T


def load_all():
    bundle = load_registered_cohort("kryeziu_mcrc_v3", root=PROJ,
                                    registry=PROJ / "data/registry/transfer_cohorts.json")
    table = _table(bundle)
    sp = dict(zip(bundle.samples.sample_id.astype(str), bundle.samples.patient_id.astype(str)))
    wl = AdmissionWhitelist(AUDIT["patient_whitelist"], AUDIT["pdo_whitelist"], patient_of_sample=sp)
    admitted, rejected = wl.reject_non_admitted(table.sample_id.astype(str).tolist(), patient_of_sample=sp)
    rows = table[table.sample_id.astype(str).isin(set(admitted))].copy()
    for c in ("patient_id", "model_id", "drug_id", "sample_id"):
        rows[c] = rows[c].astype(str).str.strip()
    rows["observed"] = pd.to_numeric(rows.response, errors="raise").astype(float)

    expression = np.asarray(bundle.expression, dtype=float)
    genes = np.asarray(bundle.gene_ids, dtype=str)

    source_prep = ExpressionPreprocessor.load(SOURCE_DIR / "expression_transform.npz")
    source_x = source_prep.transform(expression, genes)

    chem = bundle.drugs.copy()
    field = "canonical_isomeric_smiles"
    lookup = dict(zip(chem.drug_id.astype(str), chem[field], strict=True))
    drug_ids = sorted(rows.drug_id.unique())
    fp_raw = np.vstack([morgan_fingerprint(str(lookup[d]), radius=2, n_bits=FP_BITS).astype(float)
                        for d in drug_ids])
    fp_dim = min(FP_PCA, len(drug_ids) - 1)
    fp_mean, fp_comps = _fit_pca(fp_raw, fp_dim)
    fp_dict = {d: _apply_pca(fp_raw[i:i + 1], fp_mean, fp_comps)[0] for i, d in enumerate(drug_ids)}

    feat = pd.read_parquet(PERT_PARQUET)
    feat = (feat.groupby(["sample_id", "drug_id"], sort=True)
                [["reversal_score", "pathway_reversal", "perturb_magnitude"]]
                .mean().reset_index())
    pert_map = {
        (str(r.sample_id), str(r.drug_id)): (float(r.reversal_score),
                                             float(r.pathway_reversal),
                                             float(r.perturb_magnitude))
        for r in feat.itertuples(index=False)
    }
    missing = rows[~rows.apply(lambda r: (r.sample_id, r.drug_id) in pert_map, axis=1)]
    assert len(missing) == 0, f"{len(missing)} rows missing perturbation features"

    return dict(rows=rows, source_x=source_x, fp_dict=fp_dict, pert_map=pert_map,
                fp_pca_dim=int(fp_dim), drug_ids=drug_ids)


def fold_map():
    fm = {}
    for f in SPLITS["outer_folds"]:
        fm[int(f["fold"])] = {
            "test_patients": sorted(f["test_patient_ids"]),
            "train_patients": sorted(f["train_patient_ids"]),
            "inner_folds": f["inner_folds"],
        }
    return fm


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------

def _agg_patient_drug(frame):
    f = frame[["patient_id", "drug_id", "observed", "predicted"]].copy()
    f["observed"] = pd.to_numeric(f["observed"], errors="raise").astype(float)
    f["predicted"] = pd.to_numeric(f["predicted"], errors="raise").astype(float)
    return f.groupby(["patient_id", "drug_id"], sort=True, as_index=False).agg(
        observed=("observed", "mean"), predicted=("predicted", "mean"))


def _sp(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    if len(y) < MIN_DRUGS or np.ptp(y) == 0 or np.ptp(p) == 0:
        return np.nan
    r = spearmanr(y, p).statistic
    return float(r) if (r is not None and np.isfinite(r)) else np.nan


def _sp_drug(y, p, min_patients=MIN_PATIENTS_PER_DRUG):
    y = np.asarray(y, float); p = np.asarray(p, float)
    if len(y) < min_patients or np.ptp(y) == 0 or np.ptp(p) == 0:
        return np.nan
    r = spearmanr(y, p).statistic
    return float(r) if (r is not None and np.isfinite(r)) else np.nan


def macro_bootstrap(vals, seed=BOOT_SEED, n=N_BOOT):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) == 0:
        return {"point_estimate": None, "low": None, "high": None, "n_units": 0}
    rng = np.random.default_rng(seed)
    boot = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return {"point_estimate": float(v.mean()), "low": float(np.quantile(boot, 0.025)),
            "high": float(np.quantile(boot, 0.975)), "n_units": int(len(v)),
            "n": n, "seed": seed}


def _ndcg_at_k(observed, predicted, k):
    from scipy.stats import rankdata
    y = np.asarray(observed, float); p = np.asarray(predicted, float)
    if len(y) < k or np.ptp(y) == 0 or np.ptp(p) == 0:
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


def arm_eval(frame):
    macro, n_valid, per_patient = intra_patient_spearman_patient_macro(frame, min_drugs=MIN_DRUGS)
    agg = _agg_patient_drug(frame)

    drug_rhos = []
    for _, g in agg.groupby("drug_id", sort=True):
        r = _sp_drug(g.observed.to_numpy(float), g.predicted.to_numpy(float))
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
            if np.isfinite(v):
                ndcgs.append(float(v))
    ndcg5 = float(np.mean(ndcgs)) if ndcgs else None

    pv = per_patient["intra_patient_spearman"].to_numpy(float)
    return {
        "intra_patient_spearman": macro,
        "intra_patient_spearman_ci": macro_bootstrap(pv),
        "n_valid_patients": int(n_valid),
        "n_patients": int(frame.patient_id.nunique()),
        "n_rows": int(len(frame)),
        "n_drugs": int(frame.drug_id.nunique()),
        "per_drug_spearman": per_drug,
        "top1": topk[1], "top2": topk[2], "top3": topk[3],
        "ndcg5": ndcg5,
        "reason_counts": {str(k): int(v) for k, v in
                          per_patient.loc[per_patient.reason.notna(), "reason"].value_counts().items()},
    }


# ---------------------------------------------------------------------------
# 配对（患者级 bootstrap 2000）
# ---------------------------------------------------------------------------

def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def per_drug_paired_bootstrap(a, b, seed=BOOT_SEED, n=N_BOOT):
    """逐药跨患者 Spearman 宏平均的配对比（患者级重采样）。

    a/b 共享观测，故 rank(y) 每 (药, 抽) 只算一次；Spearman=Pearson(秩)，
    用 rankdata+融合循环提速。
    """
    from scipy.stats import rankdata
    A = _agg_patient_drug(a); B = _agg_patient_drug(b)
    patients = sorted(set(A.patient_id) & set(B.patient_id))
    n_pat = len(patients)
    p2i = {p: i for i, p in enumerate(patients)}

    def drug_maps(frame):
        obs = {}; pred = {}
        for r in frame.itertuples(index=False):
            obs.setdefault(r.drug_id, {})[r.patient_id] = float(r.observed)
            pred.setdefault(r.drug_id, {})[r.patient_id] = float(r.predicted)
        return obs, pred

    obs_a, pred_a = drug_maps(A)
    obs_b, pred_b = drug_maps(B)
    drugs = sorted(set(obs_a) & set(obs_b))

    posmaps = []; ovals = []; pvals_a = []; pvals_b = []
    for d in drugs:
        oa_d = obs_a[d]; pa_d = pred_a[d]
        ob_d = obs_b[d]; pb_d = pred_b[d]
        idx = []; ov = []; av = []; bv = []
        for p in patients:
            if p in oa_d and p in ob_d:
                idx.append(p2i[p]); ov.append(oa_d[p])
                av.append(pa_d[p]); bv.append(pb_d[p])
        idx = np.asarray(idx, int)
        if len(idx) < MIN_PATIENTS_PER_DRUG:
            continue
        vmap = np.full(n_pat, -1, dtype=int)
        vmap[idx] = np.arange(len(idx))
        posmaps.append(vmap)
        ovals.append(np.asarray(ov, float))
        pvals_a.append(np.asarray(av, float))
        pvals_b.append(np.asarray(bv, float))
    if not posmaps:
        return {"point_estimate": None, "low": None, "high": None,
                "n_valid": 0, "n": n, "seed": seed, "p_two_sided": None}

    def macro_pair(pick):
        vals_a = []; vals_b = []
        for vmap, ov, pva, pvb in zip(posmaps, ovals, pvals_a, pvals_b):
            sel = vmap[pick]
            sel = sel[sel >= 0]
            if len(sel) < MIN_PATIENTS_PER_DRUG:
                continue
            y = ov[sel]
            if np.ptp(y) == 0:
                continue
            pa = pva[sel]; pb = pvb[sel]
            if np.ptp(pa) == 0 or np.ptp(pb) == 0:
                continue
            ry = rankdata(y, method="average")
            ra = _pearson(ry, rankdata(pa, method="average"))
            rb = _pearson(ry, rankdata(pb, method="average"))
            if np.isfinite(ra):
                vals_a.append(ra)
            if np.isfinite(rb):
                vals_b.append(rb)
        ma = float(np.mean(vals_a)) if vals_a else np.nan
        mb = float(np.mean(vals_b)) if vals_b else np.nan
        return ma, mb

    rng = np.random.default_rng(seed)
    base = np.arange(n_pat)
    ma0, mb0 = macro_pair(base)
    point = (ma0 - mb0) if (np.isfinite(ma0) and np.isfinite(mb0)) else np.nan
    draws = rng.integers(0, n_pat, size=(n, n_pat))
    deltas = np.empty(n)
    for t in range(n):
        ma, mb = macro_pair(draws[t])
        deltas[t] = (ma - mb) if (np.isfinite(ma) and np.isfinite(mb)) else np.nan
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) == 0:
        return {"point_estimate": point if np.isfinite(point) else None,
                "low": None, "high": None, "n_valid": 0, "n": n, "seed": seed,
                "p_two_sided": None}
    low, high = np.quantile(deltas, [0.025, 0.975])
    p_greater = float(np.mean(deltas <= 0))
    p_two = float(min(1.0, 2.0 * min(p_greater, 1.0 - p_greater)))
    return {"point_estimate": point if np.isfinite(point) else float(np.mean(deltas)),
            "low": float(low), "high": float(high),
            "n_valid": int(len(deltas)), "n": n, "seed": seed,
            "p_greater_than_zero": round(1.0 - p_greater, 6), "p_two_sided": round(p_two, 6)}


def intra_paired_contrast(a, b, seed=BOOT_SEED, n=N_BOOT):
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
                "n_units": len(valid), "n": n, "p_two_sided": None,
                "note": "fewer than 2 comparable patients"}
    Av = A[A.patient_id.isin(valid)].copy(); Bv = B[B.patient_id.isin(valid)].copy()
    ci = paired_bootstrap_ci(Av, Bv, "patient_id", _sp, n=n, seed=seed,
                             truth_col="observed", pred_col="predicted")
    deltas = np.asarray(ci.deltas, float)
    rng = np.random.default_rng(seed)
    boots = deltas[rng.integers(0, len(deltas), size=(n, len(deltas)))].mean(axis=1)
    p_greater = float(np.mean(boots <= 0))
    p_two = float(min(1.0, 2.0 * min(p_greater, 1.0 - p_greater)))
    return {"point_estimate": ci.point_estimate, "low": ci.low, "high": ci.high,
            "n_units": ci.n_units, "n": n, "seed": seed,
            "p_greater_than_zero": round(1.0 - p_greater, 6), "p_two_sided": round(p_two, 6)}


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def _zscore(train, test):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return (train - mean) / std, (test - mean) / std


def _train(model, xtr, ytr, seed, wd=1e-3, epochs=EPOCHS, lr=LR):
    torch.manual_seed(seed)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    X = [torch.tensor(t, dtype=torch.float32, device=DEVICE) for t in xtr]
    y = torch.tensor(ytr, dtype=torch.float32, device=DEVICE)
    for _ in range(epochs):
        model.train(); opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(*X), y)
        loss.backward(); opt.step()
    model.eval()
    return model


def _predict(model, xte):
    X = [torch.tensor(t, dtype=torch.float32, device=DEVICE) for t in xte]
    with torch.no_grad():
        return model(*X).cpu().numpy()


def _make_parallel(hidden, drug_in, drug_seed):
    torch.manual_seed(SEED_BASE)
    enc = nn.Linear(EXPR_DIM, hidden)
    head = nn.Linear(hidden, 1)
    return ParallelNet(enc, head, drug_in, hidden, drug_seed)


def row_matrices(sub, data):
    """返回 patient/model/sample/drug 身份 + expr/fp/pert 矩阵（按行序）。"""
    expr = data["source_x"][sub.expression_index.to_numpy(int)]
    fp = np.vstack([data["fp_dict"][d] for d in sub.drug_id])
    pert = np.vstack([data["pert_map"][(r.sample_id, r.drug_id)] for r in sub.itertuples(index=False)])
    return expr, fp, pert


# ---------------------------------------------------------------------------
# 折内运行
# ---------------------------------------------------------------------------

def run_fold(data, fm, fold, hidden, wd, gate_rank):
    rows = data["rows"]
    foldinfo = fm[fold]
    tr = rows[rows.patient_id.isin(set(foldinfo["train_patients"]))].sort_values(
        ["sample_id", "drug_id"]).reset_index(drop=True)
    te = rows[rows.patient_id.isin(set(foldinfo["test_patients"]))].sort_values(
        ["sample_id", "drug_id"]).reset_index(drop=True)

    expr_tr, fp_tr, pert_tr = row_matrices(tr, data)
    expr_te, fp_te, pert_te = row_matrices(te, data)
    fp_tr_z, fp_te_z = _zscore(fp_tr, fp_te)
    pert_tr_z, pert_te_z = _zscore(pert_tr, pert_te)
    y_mean, y_std = float(tr.observed.mean()), float(tr.observed.std() or 1.0)
    ytr = ((tr.observed.to_numpy(float) - y_mean) / y_std).astype(float)

    drug_tr = np.concatenate([fp_tr_z, pert_tr_z], axis=1)
    drug_te = np.concatenate([fp_te_z, pert_te_z], axis=1)
    fp_dim = data["fp_pca_dim"]

    preds = {}
    for seed in INIT_SEEDS:
        m = _make_parallel(hidden, 0, seed)
        preds[("direct", seed)] = _predict(_train(m, [expr_tr], ytr, seed, wd), [expr_te]) * y_std + y_mean

        m = _make_parallel(hidden, fp_dim, seed + 1)
        preds[("direct_fp", seed)] = _predict(_train(m, [expr_tr, fp_tr_z], ytr, seed, wd), [expr_te, fp_te_z]) * y_std + y_mean

        m = _make_parallel(hidden, PERT_DIM, seed + 2)
        preds[("direct_pert", seed)] = _predict(_train(m, [expr_tr, pert_tr_z], ytr, seed, wd), [expr_te, pert_te_z]) * y_std + y_mean

        m = _make_parallel(hidden, fp_dim + PERT_DIM, seed + 3)
        preds[("direct_fp_pert", seed)] = _predict(_train(m, [expr_tr, drug_tr], ytr, seed, wd), [expr_te, drug_te]) * y_std + y_mean

        torch.manual_seed(seed)
        m = DualBranchNet(EXPR_DIM, fp_dim, PERT_DIM, DUAL_HD, DUAL_HP, DUAL_HD2, 0)
        preds[("dual", seed)] = _predict(_train(m, [expr_tr, fp_tr_z, pert_tr_z], ytr, seed, wd), [expr_te, fp_te_z, pert_te_z]) * y_std + y_mean

        torch.manual_seed(seed)
        m = DualBranchNet(EXPR_DIM, fp_dim, PERT_DIM, DUAL_HD, DUAL_HP, DUAL_HD2, gate_rank)
        preds[("dual_gated", seed)] = _predict(_train(m, [expr_tr, fp_tr_z, pert_tr_z], ytr, seed, wd), [expr_te, fp_te_z, pert_te_z]) * y_std + y_mean

        torch.manual_seed(seed)
        m = CapacityNet(EXPR_DIM, fp_dim, CAP_H, CAP_H2)
        preds[("capacity", seed)] = _predict(_train(m, [expr_tr, fp_tr_z], ytr, seed, wd), [expr_te, fp_te_z]) * y_std + y_mean

        for s in range(3):
            perm_tr = np.random.default_rng(seed * 7 + s).permutation(len(pert_tr))
            perm_te = np.random.default_rng(seed * 7 + s + 100).permutation(len(pert_te))
            torch.manual_seed(seed)
            m = DualBranchNet(EXPR_DIM, fp_dim, PERT_DIM, DUAL_HD, DUAL_HP, DUAL_HD2, 0)
            preds[("dual_pert_shuffle", seed, s)] = _predict(
                _train(m, [expr_tr, fp_tr_z, pert_tr_z[perm_tr]], ytr, seed, wd),
                [expr_te, fp_te_z, pert_te_z[perm_te]]) * y_std + y_mean

    return preds, tr, te


# ---------------------------------------------------------------------------
# 内层选择（<=20 候选）
# ---------------------------------------------------------------------------

def inner_select(data, fm, fold):
    rows = data["rows"]
    foldinfo = fm[fold]
    fp_dim = data["fp_pca_dim"]

    scores = {}
    for hidden in HIDDEN_CAND:
        for wd in WD_CAND:
            per_inner = []
            for inf in foldinfo["inner_folds"]:
                itr = set(inf["inner_train_patient_ids"]); iva = set(inf["inner_val_patient_ids"])
                sub_tr = rows[rows.patient_id.isin(itr)].sort_values(["sample_id", "drug_id"]).reset_index(drop=True)
                sub_va = rows[rows.patient_id.isin(iva)].sort_values(["sample_id", "drug_id"]).reset_index(drop=True)
                expr_tr, fp_tr, pert_tr = row_matrices(sub_tr, data)
                expr_va, fp_va, pert_va = row_matrices(sub_va, data)
                fp_tr_z, fp_va_z = _zscore(fp_tr, fp_va)
                pert_tr_z, pert_va_z = _zscore(pert_tr, pert_va)
                drug_tr = np.concatenate([fp_tr_z, pert_tr_z], axis=1)
                drug_va = np.concatenate([fp_va_z, pert_va_z], axis=1)
                y_mean, y_std = float(sub_tr.observed.mean()), float(sub_tr.observed.std() or 1.0)
                ytr = ((sub_tr.observed.to_numpy(float) - y_mean) / y_std).astype(float)
                m = _make_parallel(hidden, fp_dim + PERT_DIM, 1)
                m = _train(m, [expr_tr, drug_tr], ytr, SEED_BASE, wd)
                vpred = _predict(m, [expr_va, drug_va]) * y_std + y_mean
                frame = pd.DataFrame({"patient_id": sub_va.patient_id, "drug_id": sub_va.drug_id,
                                      "observed": sub_va.observed.to_numpy(float), "predicted": vpred})
                macro, _, _ = intra_patient_spearman_patient_macro(frame, min_drugs=MIN_DRUGS)
                if macro is not None and np.isfinite(macro):
                    per_inner.append(macro)
            scores[(hidden, wd)] = float(np.mean(per_inner)) if per_inner else float("-inf")
    best = max(scores, key=lambda k: scores[k])

    gate_scores = {}
    for r in GATE_RANK_CAND:
        per_inner = []
        for inf in foldinfo["inner_folds"]:
            itr = set(inf["inner_train_patient_ids"]); iva = set(inf["inner_val_patient_ids"])
            sub_tr = rows[rows.patient_id.isin(itr)].sort_values(["sample_id", "drug_id"]).reset_index(drop=True)
            sub_va = rows[rows.patient_id.isin(iva)].sort_values(["sample_id", "drug_id"]).reset_index(drop=True)
            expr_tr, fp_tr, pert_tr = row_matrices(sub_tr, data)
            expr_va, fp_va, pert_va = row_matrices(sub_va, data)
            fp_tr_z, fp_va_z = _zscore(fp_tr, fp_va)
            pert_tr_z, pert_va_z = _zscore(pert_tr, pert_va)
            y_mean, y_std = float(sub_tr.observed.mean()), float(sub_tr.observed.std() or 1.0)
            ytr = ((sub_tr.observed.to_numpy(float) - y_mean) / y_std).astype(float)
            torch.manual_seed(SEED_BASE)
            m = DualBranchNet(EXPR_DIM, fp_dim, PERT_DIM, DUAL_HD, DUAL_HP, DUAL_HD2, r)
            m = _train(m, [expr_tr, fp_tr_z, pert_tr_z], ytr, SEED_BASE, best[1])
            vpred = _predict(m, [expr_va, fp_va_z, pert_va_z]) * y_std + y_mean
            frame = pd.DataFrame({"patient_id": sub_va.patient_id, "drug_id": sub_va.drug_id,
                                  "observed": sub_va.observed.to_numpy(float), "predicted": vpred})
            macro, _, _ = intra_patient_spearman_patient_macro(frame, min_drugs=MIN_DRUGS)
            if macro is not None and np.isfinite(macro):
                per_inner.append(macro)
        gate_scores[r] = float(np.mean(per_inner)) if per_inner else float("-inf")
    best_gate = max(gate_scores, key=lambda r: gate_scores[r])
    return {"hidden": best[0], "wd": best[1], "gate_rank": best_gate,
            "hidden_wd_scores": {f"{k[0]}/{k[1]}": v for k, v in scores.items()},
            "gate_rank_scores": {str(k): v for k, v in gate_scores.items()}}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _frame_from_preds(preds, key, te, fold):
    p = preds[key]
    seed = key[1]
    shuffle = int(key[2]) if len(key) == 3 else None
    rows = []
    for i, r in enumerate(te.itertuples(index=False)):
        rows.append({"fold": int(fold), "seed": int(seed), "arm": key[0],
                     "shuffle": shuffle,
                     "patient_id": r.patient_id, "model_id": r.model_id,
                     "sample_id": r.sample_id, "drug_id": r.drug_id,
                     "observed": float(r.observed), "predicted": float(p[i])})
    return pd.DataFrame(rows)


def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_all()
    fm = fold_map()
    rows = data["rows"]

    # arm5 扰动单分支（零目标标签排序 = reversal_score 3 种子均值）
    arm5 = pd.DataFrame({
        "fold": -1, "seed": 0, "arm": "pert_only",
        "patient_id": rows.patient_id, "model_id": rows.model_id,
        "sample_id": rows.sample_id, "drug_id": rows.drug_id,
        "observed": rows.observed.to_numpy(float),
        "predicted": [data["pert_map"][(r.sample_id, r.drug_id)][0]
                      for r in rows.itertuples(index=False)],
    })

    frames_by_arm = {a: [] for a in ARM_NAMES}
    frames_by_arm["pert_only"].append(arm5)
    meta = []
    n_params_log = None

    for fold in sorted(fm):
        sel = inner_select(data, fm, fold)
        hidden, wd, gate_rank = sel["hidden"], sel["wd"], sel["gate_rank"]
        preds, tr, te = run_fold(data, fm, fold, hidden, wd, gate_rank)
        meta.append({"fold": int(fold), **sel})
        for seed in INIT_SEEDS:
            for arm in ["direct", "direct_fp", "direct_pert", "direct_fp_pert",
                        "dual", "dual_gated", "capacity"]:
                frames_by_arm[arm].append(_frame_from_preds(preds, (arm, seed), te, fold))
            for s in range(3):
                frames_by_arm["dual_pert_shuffle"].append(
                    _frame_from_preds(preds, ("dual_pert_shuffle", seed, s), te, fold))
        if n_params_log is None:
            n_params_log = _measure_params(hidden, data["fp_pca_dim"], gate_rank)

    arm_frames = {a: pd.concat(frames_by_arm[a], ignore_index=True) for a in ARM_NAMES}

    # predictions.parquet（统一 schema）
    schema_list = []
    for arm in ARM_NAMES:
        f = arm_frames[arm]
        for r in f.itertuples(index=False):
            sh = int(r.shuffle) if (arm == "dual_pert_shuffle" and getattr(r, "shuffle", None) is not None) else None
            run_id = f"exp034_parallel_{arm}_seed{r.seed}"
            if sh is not None:
                run_id += f"_shuf{sh}"
            schema_list.append({
                "experiment_id": EXP_ID,
                "run_id": run_id,
                "contract_id": CONTRACT_ID, "split_id": SPLIT_ID,
                "fold": int(r.fold), "seed": int(r.seed),
                "patient_id": r.patient_id, "model_id": r.model_id,
                "sample_id": r.sample_id, "drug_id": r.drug_id,
                "gene_id": None, "y_true": float(r.observed),
                "y_pred": None if not np.isfinite(r.predicted) else float(r.predicted),
                "evaluation_role": "test", "arm": arm, "frac": None,
            })
    pd.DataFrame(schema_list).to_parquet(OUT / "predictions.parquet", index=False)

    # 逐臂 eval
    eval_out = {a: arm_eval(arm_frames[a]) for a in ARM_NAMES}

    # 三增量（逐药跨患者 Spearman，患者级 bootstrap 2000）
    increments = {}
    pvals = []
    for label, a, b in [("fingerprint_gain", "direct_fp", "direct"),
                        ("perturbation_gain", "direct_pert", "direct"),
                        ("joint_gain", "direct_fp_pert", "direct")]:
        inc = per_drug_paired_bootstrap(arm_frames[a], arm_frames[b])
        increments[label] = {"comparison": f"{a}_minus_{b}",
                             "metric": "per_drug_cross_patient_spearman", **inc}
        if inc.get("p_two_sided") is not None:
            pvals.append(inc["p_two_sided"])
    increments["holm_p"] = holm_correct(pvals) if pvals else None
    increments["holm_applied_over"] = ["fingerprint_gain", "perturbation_gain", "joint_gain"]

    # 容量对照（患者内 Spearman）
    cap_inc = intra_paired_contrast(arm_frames["capacity"], arm_frames["direct_fp"])
    cap_inc_perdrug = per_drug_paired_bootstrap(arm_frames["capacity"], arm_frames["direct_fp"])

    # 探索：患者内排序 Spearman 臂间增量（臂1 未定义，仅 2/3/4 可算）
    intra_explore = {}
    for a, b in [("direct_fp_pert", "direct_fp"), ("direct_fp_pert", "direct_pert"),
                 ("direct_fp", "direct_pert")]:
        intra_explore[f"{a}_minus_{b}"] = intra_paired_contrast(arm_frames[a], arm_frames[b])

    summary = {
        "experiment_id": EXP_ID, "status": "RAN", "architecture": "E2_parallel",
        "env": "drugscreening-gpu (RTX A5000, cuda)",
        "n_patients": int(rows.patient_id.nunique()), "n_samples": int(rows.model_id.nunique()),
        "n_rows": int(len(rows)), "n_drugs": int(rows.drug_id.nunique()),
        "fingerprint": {"n_bits": FP_BITS, "pca_dim": data["fp_pca_dim"],
                        "pca_note": "PCA(min(64, n_drugs-1)=47，48 药最大非平凡维数)"},
        "perturbation_branch": {"n_features": PERT_DIM,
                                "features": ["reversal_score", "pathway_reversal", "perturb_magnitude"],
                                "seed_mean_over": [23, 24, 25],
                                "note": "白名单产物仅含 3 标量；通路方向由 pathway_reversal 标量承载"},
        "direct_branch": {"source_preprocessor": str(SOURCE_DIR / "expression_transform.npz"),
                          "n_features": EXPR_DIM, "frozen": True},
        "hyperparams_selected_per_fold": meta,
        "inner_selection_note": "hidden ∈ {16,32,64} × wd ∈ {1e-4,1e-3,1e-2} (9 候选) 于 arm4 内层 3 折；"
                                "gate_rank ∈ {2,4,8} (3 候选) 于 arm6b；合计 12 <= 20",
        "model_params": n_params_log,
        "init_seeds": INIT_SEEDS, "bootstrap_seed": BOOT_SEED, "n_boot": N_BOOT,
        "arms": eval_out,
        "increments_per_drug_metric": increments,
        "controls": {
            "shuffled_perturbation": {
                "note": "3 次折内打乱 × 3 种子合并后相对 arm1 的逐药增量（打乱扰动对照）",
                "shuffled_minus_direct": per_drug_paired_bootstrap(arm_frames["dual_pert_shuffle"], arm_frames["direct"]),
            },
            "capacity_vs_direct_fp": {"patient_level": cap_inc, "per_drug": cap_inc_perdrug},
        },
        "intra_patient_exploratory_increments": intra_explore,
        "arm_definition_note": {
            "arm1_direct": "纯表达（药物不变，患者内排序 Spearman 未定义）",
            "arm2_direct_fp": f"直接 + 指纹 PCA{data['fp_pca_dim']}",
            "arm3_direct_pert": "直接 + 扰动 3 标量",
            "arm4_direct_fp_pert": "直接 + 指纹 + 扰动",
            "arm5_pert_only": "扰动单分支（零目标标签，reversal_score 排名，只报绝对值）",
            "arm6_dual": "双分支（直接[expr,fp] + 扰动[pert] 并联 concat）",
            "arm6b_dual_gated": "双分支条件化权重变体（扰动分支 low-rank 门控，次级臂）",
            "arm7_dual_pert_shuffle": "双分支打乱扰动（折内 3 次打乱 × 3 种子）",
            "arm8_capacity": "容量相近直接模型（[expr,fp] 两层隐藏 ≈ 双分支参数量）",
        },
        "elapsed_seconds": time.perf_counter() - t0,
    }
    (OUT / "arm_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    _write_report(summary)
    print("WROTE predictions.parquet + arm_summary.json + increment_report.md")
    _print_table(summary)


def _measure_params(hidden, fp_dim, gate_rank):
    torch.manual_seed(0)
    enc = nn.Linear(EXPR_DIM, hidden); head = nn.Linear(hidden, 1)
    return {
        "hidden_selected": hidden, "fp_pca_dim": fp_dim,
        "arm1_direct": n_params(ParallelNet(enc, head, 0, hidden, 0)),
        "arm2_direct_fp": n_params(ParallelNet(enc, head, fp_dim, hidden, 0)),
        "arm3_direct_pert": n_params(ParallelNet(enc, head, PERT_DIM, hidden, 0)),
        "arm4_direct_fp_pert": n_params(ParallelNet(enc, head, fp_dim + PERT_DIM, hidden, 0)),
        "arm6_dual": n_params(DualBranchNet(EXPR_DIM, fp_dim, PERT_DIM, DUAL_HD, DUAL_HP, DUAL_HD2, 0)),
        "arm6b_dual_gated": n_params(DualBranchNet(EXPR_DIM, fp_dim, PERT_DIM, DUAL_HD, DUAL_HP, DUAL_HD2, gate_rank)),
        "arm8_capacity": n_params(CapacityNet(EXPR_DIM, fp_dim, CAP_H, CAP_H2)),
        "note": "arm8_capacity 应与 arm6_dual 参数量相近（容量对照）",
    }


def _f(x):
    if x is None:
        return "NA"
    if isinstance(x, float) and not np.isfinite(x):
        return "NaN"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _write_report(s):
    inc = s["increments_per_drug_metric"]
    holm = inc["holm_p"] or []
    lines = [
        "# EXP-034 E2 并联架构：指纹/扰动增量四组合分离",
        "",
        f"- status: RAN（{s['n_patients']} 患者 / {s['n_samples']} 样本 / {s['n_rows']} 观测行 / {s['n_drugs']} 药）",
        "- 直接分支：冻结源预处理器 512 表达；扰动分支：whitelist 3 标量（3 种子均值）",
        f"- 指纹：Morgan 2048 -> PCA{s['fingerprint']['pca_dim']}",
        "",
        "## 臂主指标（逐种子合并测试折，患者内 Spearman 为主指标 / 逐药跨患者 Spearman 为个体化指标）",
        "",
        "| 臂 | 患者内 Spearman | 逐药跨患者 Spearman | top1 | top3 | NDCG@5 | n_valid |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in ["direct", "direct_fp", "direct_pert", "direct_fp_pert", "pert_only",
              "dual", "dual_gated", "dual_pert_shuffle", "capacity"]:
        e = s["arms"][a]
        lines.append(f"| {a} | {_f(e['intra_patient_spearman'])} | {_f(e['per_drug_spearman'])} | "
                     f"{_f(e['top1'])} | {_f(e['top3'])} | {_f(e['ndcg5'])} | {e['n_valid_patients']} |")
    lines += [
        "",
        "> 臂1（纯表达，药物不变）患者内排序未定义（常值）；臂5 零目标标签排序。",
        "",
        "## 三增量（逐药跨患者 Spearman，患者级 2000 次 bootstrap，arm_n − arm1）",
        "",
        "| 增量 | 点估计 | 95% CI | Holm p |",
        "|---|---|---|---|",
    ]
    for i, label in enumerate(["fingerprint_gain", "perturbation_gain", "joint_gain"]):
        r = inc[label]
        hp = holm[i] if i < len(holm) else None
        lines.append(f"| {label} | {_f(r['point_estimate'])} | [{_f(r['low'])}, {_f(r['high'])}] | {_f(hp)} |")
    lines += [
        "",
        "## 打乱扰动对照（逐药跨患者 Spearman，dual_pert_shuffle 合并 − arm1）",
        "",
    ]
    c = s["controls"]["shuffled_perturbation"]["shuffled_minus_direct"]
    lines.append(f"- shuffled_minus_direct: point={_f(c['point_estimate'])} "
                 f"CI=[{_f(c['low'])}, {_f(c['high'])}] p={_f(c.get('p_two_sided'))}")
    lines += ["", "## 容量对照（capacity 直接模型 vs direct_fp）", ""]
    cap = s["controls"]["capacity_vs_direct_fp"]
    lines.append(f"- 患者内 Spearman: point={_f(cap['patient_level'].get('point_estimate'))} "
                 f"CI=[{_f(cap['patient_level'].get('low'))}, {_f(cap['patient_level'].get('high'))}] "
                 f"p={_f(cap['patient_level'].get('p_two_sided'))}")
    lines.append(f"- 逐药跨患者 Spearman: point={_f(cap['per_drug'].get('point_estimate'))} "
                 f"CI=[{_f(cap['per_drug'].get('low'))}, {_f(cap['per_drug'].get('high'))}] "
                 f"p={_f(cap['per_drug'].get('p_two_sided'))}")
    lines += ["", "## 患者内排序探索增量（臂1 未定义，仅 2/3/4 可算）", "",
              "| 比较 | 点估计 | 95% CI | p(双侧) |", "|---|---|---|---|"]
    for k, v in s["intra_patient_exploratory_increments"].items():
        lines.append(f"| {k} | {_f(v.get('point_estimate'))} | "
                     f"[{_f(v.get('low'))}, {_f(v.get('high'))}] | {_f(v.get('p_two_sided'))} |")
    lines += [
        "",
        "## 判定建议",
        "",
        "核心判定链：配对增益 CI 不跨零（Holm 后）＋ 打乱对照无同量级增益 ＋ 容量对照无同量级增益。",
        "详细数值见 arm_summary.json。",
        "",
    ]
    (OUT / "increment_report.md").write_text("\n".join(lines), encoding="utf-8")


def _print_table(s):
    print("\n=== ARM MAIN METRICS ===")
    for a in ["direct", "direct_fp", "direct_pert", "direct_fp_pert", "pert_only",
              "dual", "dual_gated", "dual_pert_shuffle", "capacity"]:
        e = s["arms"][a]
        print(f"  {a}: intra={_f(e['intra_patient_spearman'])} per_drug={_f(e['per_drug_spearman'])} "
              f"top1={_f(e['top1'])} ndcg5={_f(e['ndcg5'])} n_valid={e['n_valid_patients']}")
    print("\n=== INCREMENTS (per_drug) ===")
    for label in ["fingerprint_gain", "perturbation_gain", "joint_gain"]:
        r = s["increments_per_drug_metric"][label]
        print(f"  {label}: {_f(r['point_estimate'])} [{_f(r['low'])}, {_f(r['high'])}] p={_f(r.get('p_two_sided'))}")


if __name__ == "__main__":
    main()