"""EXP-034 E3 联合学习执行（部分共享表示多任务，工作包 E 预登记）。

只写 artifacts/exp034_joint/：
- predictions.parquet（统一 schema 预测）
- summary.json、report.md

口径以 EXP-034.md 预登记 + 工作包 B 合同（exp031_eval）为准：
- 双任务：药敏（目标标签）与扰动预测（ChemPert 上游标签）。
- 共享编码器从表达出发（512 源预处理表达）；两个头分别预测药敏分数与
  扰动 delta（795 维基因签名）。Kryzei 行与 ChemPert 行各自经共享编码器，
  损失只算各自存在的标签（Kryzei 行无扰动标签、ChemPert 行无药敏标签；
  缺失不填零）。
- 辅助权重 × 共享维度 仅在训练/内层验证选择（≤20 候选）。
- 药敏头在各外层折 train 患者训练（联合训练批次混采样）、test 预测；
  主指标 = 患者内全药排序 Spearman。3 初始化种子。
- 容量对照：同参数量级的纯药敏模型（单任务）。
- 重要限制：源预处理器 512 基因与 ChemPert 控制表达 795 基因仅 48 个重叠，
  ChemPert 侧经共享编码器的表达信息有限（464/512 维为源中位数插补），如实记录。
- 测试折只参与评测；全部学习型预处理只在训练行 fit。
"""

from __future__ import annotations

import json
import os
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
OUT = PROJ / "artifacts" / "exp034_joint"
AUDIT = json.loads((PROJ / "data/processed/identity/kryzei_admission_audit_20260921.json").read_text(encoding="utf-8"))
SPLITS = json.loads((PROJ / "data/splits/kryzei_shared_splits_v1.json").read_text(encoding="utf-8"))
SOURCE_DIR = PROJ / "data/artifacts/EXP-015/source_seed20260912"
CHEMPERT_TABLE = PROJ / "data/processed/chempert/chempert_v2_response_table.parquet"
CHEMPERT_SLICE = PROJ / "data/processed/chempert/chempert_v2_control_expression_slice.parquet"
CHEMPERT_SMILES = PROJ / "data/processed/chempert/chempert_v2_chem_smiles.parquet"

EXP_ID = "EXP-034"
CONTRACT_ID = "exp034_joint_contract"
SPLIT_ID = "kryzei_shared_v1"
DEVICE = "cuda"

BOOT_SEED = 20260918
N_BOOT = 2000
INIT_SEEDS = [20260921, 20260922, 20260923]

# 预登记调参预算：辅助权重 × 共享维度（≤20 候选）
AUX_WEIGHTS = [0.1, 0.3, 1.0, 3.0]
SHARED_DIMS = [64, 128]
HID = 256          # 两个头的隐藏维（固定）
EPOCHS_FINAL = 15
EPOCHS_SELECT = 6
LR = 1e-3
BATCH_C = 1024     # ChemPert 小批

SMOKE = os.environ.get("EXP034_SMOKE") == "1"
if SMOKE:
    AUX_WEIGHTS = [0.3, 1.0]
    SHARED_DIMS = [64]
    EPOCHS_FINAL = 2
    EPOCHS_SELECT = 2
    INIT_SEEDS = INIT_SEEDS[:1]


def load_kryzei():
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
    rows = rows.sort_values(["sample_id", "drug_id"], kind="mergesort").reset_index(drop=True)

    expression = np.asarray(bundle.expression, dtype=float)
    genes = np.asarray(bundle.gene_ids, dtype=str)

    chem = bundle.drugs.copy()
    field = "canonical_isomeric_smiles"
    lookup = dict(zip(chem.drug_id.astype(str), chem[field], strict=True))
    fps = {str(d): morgan_fingerprint(str(lookup[str(d)]), radius=2, n_bits=2048).astype(float)
           for d in rows.drug_id.unique()}

    source_prep = ExpressionPreprocessor.load(SOURCE_DIR / "expression_transform.npz")
    source_x = source_prep.transform(expression, genes)   # (n_samples, 512)

    return dict(bundle=bundle, rows=rows, expression=expression, genes=genes,
                fps=fps, source_prep=source_prep, source_x=source_x, rejected=rejected)


def load_chempert(source_prep):
    table = pd.read_parquet(CHEMPERT_TABLE)
    sl = pd.read_parquet(CHEMPERT_SLICE)
    genes = sl["gene_symbol"].tolist()
    prof_cols = [c for c in sl.columns if c != "gene_symbol"]
    slice_mat = sl[prof_cols].to_numpy(dtype=np.float32)
    col_pos = {c: j for j, c in enumerate(prof_cols)}
    gene_pos = {g: i for i, g in enumerate(genes)}

    smiles = pd.read_parquet(CHEMPERT_SMILES).dropna(subset=["canonical_smiles"])
    chem2smi = dict(zip(smiles["Chemical_ID"], smiles["canonical_smiles"].astype(str)))

    sigs, cov = [], []
    enc_rows = []
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
    Y_all = np.stack(sigs)
    nz = np.array(cov) > 0

    # Ctrl 表达
    eids = table["Expression_ID"].to_numpy()
    X_ctrl = np.stack([slice_mat[:, col_pos[e]] for e in eids]).astype(np.float32)  # (N, 795)

    # Kryzei -> 512 源空间投影（共享编码器输入）
    X512 = source_prep.transform(X_ctrl, genes).astype(np.float32)  # (N, 512)

    # 药物指纹（去重指纹）
    chem_ids = table["Chemical_ID"].to_numpy()
    uniq = sorted(set(chem_ids[nz]))
    fp_map = {}
    parseable = set()
    for c in uniq:
        smi = chem2smi.get(c)
        if smi is None:
            continue
        try:
            fp = morgan_fingerprint(str(smi), radius=2, n_bits=2048)
        except Exception:
            continue
        fp_map[c] = fp
        parseable.add(c)

    encodable = nz & np.array([c in parseable for c in chem_ids])
    idx = np.where(encodable)[0]
    F = np.stack([fp_map[chem_ids[i]] for i in idx]).astype(np.float32)

    if SMOKE:
        idx = idx[:8000]

    return {
        "X512": X512[idx],                  # (n, 512) 控制表达 -> 512 源空间
        "F": F,                             # (n, 2048) 药物指纹
        "Y": Y_all[idx],                    # (n, 795) 扰动签名标签
        "n_encodable_total": int(len(np.where(encodable)[0])),
        "n_chempert_genes": len(genes),
        "n_overlap_source_genes": int(sum(1 for g in source_prep.gene_ids if g in set(genes))),
    }


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
# 模型
# ---------------------------------------------------------------------------

class _Encoder(nn.Module):
    def __init__(self, in_dim, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class _SensHead(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 2048, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, 1))

    def forward(self, z, fp):
        return self.net(torch.cat([z, fp], dim=-1)).squeeze(1)


class _PertHead(nn.Module):
    def __init__(self, d, h, out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 2048, h), nn.ReLU(), nn.Linear(h, out))

    def forward(self, z, fp):
        return self.net(torch.cat([z, fp], dim=-1))


class _SingleNet(nn.Module):
    """容量匹配单任务：编码器(512->d) + 药敏头(d+2048 -> h_s -> 1)。"""

    def __init__(self, d, h_s):
        super().__init__()
        self.enc = _Encoder(512, d)
        self.sens = _SensHead(d, h_s)

    def forward(self, x, fp):
        return self.sens(self.enc(x), fp)


def _count(m):
    return sum(p.numel() for p in m.parameters())


def capacity_h_s(d, h):
    """计算单任务药敏头隐藏宽 h_s，使单任务总参数量 ≈ 联合模型总参数量。"""
    enc = _Encoder(512, d)
    sens = _SensHead(d, h)
    pert = _PertHead(d, h, 795)
    joint = _count(enc) + _count(sens) + _count(pert)
    single_enc = _count(enc)
    # single_enc(512->d) + sens_head(d+2048 -> h_s -> 1) == joint
    # sens_head params = (d+2049)*h_s + 1
    target = joint - single_enc
    h_s = int(round((target - 1) / (d + 2049)))
    h_s = max(8, h_s)
    return h_s


# ---------------------------------------------------------------------------
# 指标（与 E1 一致）
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
    macro, n_valid, per_patient = intra_patient_spearman_patient_macro(frame, min_drugs=min_drugs)
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
                "n_units": len(valid), "n": n, "note": "fewer than 2 comparable patients"}
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
# 训练
# ---------------------------------------------------------------------------

def joint_predict(enc, sens, X, F, mean, std):
    enc.eval(); sens.eval()
    with torch.no_grad():
        z = enc(torch.tensor(X, dtype=torch.float32, device=DEVICE))
        out = sens(z, torch.tensor(F, dtype=torch.float32, device=DEVICE))
        return out.cpu().numpy() * std + mean


def _kryzei_matrices(rows_sub, source_x, fps):
    idx = rows_sub.expression_index.to_numpy(int)
    X = source_x[idx]
    F = np.vstack([fps[str(d)] for d in rows_sub.drug_id])
    return X, F


def train_joint(Xk, Fk, yk, Xc, Fc, Yc, *, seed, d, h, aux, epochs):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    enc = _Encoder(512, d).to(DEVICE)
    sens = _SensHead(d, h).to(DEVICE)
    pert = _PertHead(d, h, 795).to(DEVICE)
    params = list(enc.parameters()) + list(sens.parameters()) + list(pert.parameters())
    opt = torch.optim.Adam(params, lr=LR)

    mean = float(np.mean(yk)); std = float(np.std(yk)) or 1.0
    Xk_t = torch.tensor(Xk, dtype=torch.float32, device=DEVICE)
    Fk_t = torch.tensor(Fk, dtype=torch.float32, device=DEVICE)
    yk_t = torch.tensor((yk - mean) / std, dtype=torch.float32, device=DEVICE)
    Xc_np, Fc_np, Yc_np = Xc, Fc, Yc
    nc = len(Xc_np)

    hist = []
    for epoch in range(int(epochs)):
        enc.train(); sens.train(); pert.train()
        order = np.random.permutation(nc)
        n_steps = int(np.ceil(nc / BATCH_C))
        tot_sens = 0.0; tot_pert = 0.0; n_steps_done = 0
        for st in range(n_steps):
            ci = order[st * BATCH_C:(st + 1) * BATCH_C]
            xb = torch.tensor(Xc_np[ci], dtype=torch.float32, device=DEVICE)
            fb = torch.tensor(Fc_np[ci], dtype=torch.float32, device=DEVICE)
            yb = torch.tensor(Yc_np[ci], dtype=torch.float32, device=DEVICE)
            opt.zero_grad(set_to_none=True)
            zk = enc(Xk_t)
            sens_loss = F.huber_loss(sens(zk, Fk_t), yk_t)
            zc = enc(xb)
            pert_loss = F.mse_loss(pert(zc, fb), yb)
            loss = sens_loss + aux * pert_loss
            loss.backward(); opt.step()
            tot_sens += float(sens_loss.detach().cpu())
            tot_pert += float(pert_loss.detach().cpu())
            n_steps_done += 1
        hist.append({"epoch": epoch + 1, "sens_loss": tot_sens / n_steps_done,
                     "pert_loss": tot_pert / n_steps_done})
    enc.eval(); sens.eval(); pert.eval()
    return enc, sens, mean, std, hist


def train_single(Xk, Fk, yk, *, seed, d, h_s, epochs):
    torch.manual_seed(int(seed))
    model = _SingleNet(d, h_s).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mean = float(np.mean(yk)); std = float(np.std(yk)) or 1.0
    Xt = torch.tensor(Xk, dtype=torch.float32, device=DEVICE)
    Ft = torch.tensor(Fk, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor((yk - mean) / std, dtype=torch.float32, device=DEVICE)
    for _ in range(int(epochs)):
        model.train(); opt.zero_grad()
        loss = F.huber_loss(model(Xt, Ft), yt); loss.backward(); opt.step()
    model.eval()
    return model, mean, std


def single_predict(model, X, F, mean, std):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32, device=DEVICE),
                     torch.tensor(F, dtype=torch.float32, device=DEVICE)).cpu().numpy() * std + mean


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_kryzei()
    chempert = load_chempert(data["source_prep"])
    fm = fold_map()
    rows = data["rows"]
    source_x = data["source_x"]
    fps = data["fps"]

    pred_rows = []
    meta = []
    selected_configs = {}
    convergence = {}

    folds = list(sorted(fm)) if not SMOKE else [0]

    for fold in folds:
        finfo = fm[fold]
        train_pat = set(finfo["train_patients"])
        test_pat = set(finfo["test_patients"])
        tr = rows[rows.patient_id.isin(train_pat)].reset_index(drop=True)
        te = rows[rows.patient_id.isin(test_pat)].reset_index(drop=True)
        Xtr, Ftr = _kryzei_matrices(tr, source_x, fps)
        ytr = tr.observed.to_numpy(float)
        Xte, Fte = _kryzei_matrices(te, source_x, fps)
        Xc, Fc, Yc = chempert["X512"], chempert["F"], chempert["Y"]

        # ---- 内层选择（辅助权重 × 共享维度）----
        cand_scores = {}
        for d in SHARED_DIMS:
            for aux in AUX_WEIGHTS:
                key = (d, aux)
                cand_scores[key] = []
                for inf in finfo["inner_folds"]:
                    in_tr_pat = set(inf["inner_train"])
                    in_val_pat = set(inf["inner_val"])
                    itr = tr[tr.patient_id.isin(in_tr_pat)].reset_index(drop=True)
                    iva = tr[tr.patient_id.isin(in_val_pat)].reset_index(drop=True)
                    Xitr, Fitr = _kryzei_matrices(itr, source_x, fps)
                    Xiva, Fiva = _kryzei_matrices(iva, source_x, fps)
                    yitr = itr.observed.to_numpy(float)
                    enc, sens, mean, std, hist = train_joint(
                        Xitr, Fitr, yitr, Xc, Fc, Yc, seed=INIT_SEEDS[0],
                        d=d, h=HID, aux=aux, epochs=EPOCHS_SELECT)
                    pred = joint_predict(enc, sens, Xiva, Fiva, mean, std)
                    vf = iva[["patient_id", "drug_id"]].copy()
                    vf["observed"] = iva.observed.to_numpy(float)
                    vf["predicted"] = pred
                    macro, _, _ = intra_patient_spearman_patient_macro(vf)
                    cand_scores[key].append(macro if macro is not None else -1.0)
                    del enc, sens
                # 记录收敛性：取最后一个候选的一次 history（第 1 个 inner fold 已覆盖）
        best = min(cand_scores, key=lambda k: (-np.mean(cand_scores[k]), k))
        d_sel, aux_sel = best
        selected_configs[fold] = {"shared_dim": d_sel, "aux_weight": aux_sel,
                                  "inner_scores": {f"{d}x{aux}": float(np.mean(v)) for (d, aux), v in cand_scores.items()}}
        print(f"[fold {fold}] selected shared_dim={d_sel} aux={aux_sel} "
              f"inner={selected_configs[fold]['inner_scores']}", flush=True)

        h_s = capacity_h_s(d_sel, HID)

        # ---- 最终训练：joint（3 种子）+ single 容量对照（3 种子）----
        for seed in INIT_SEEDS:
            enc, sens, mean, std, hist = train_joint(
                Xtr, Ftr, ytr, Xc, Fc, Yc, seed=seed, d=d_sel, h=HID,
                aux=aux_sel, epochs=EPOCHS_FINAL)
            pred = joint_predict(enc, sens, Xte, Fte, mean, std)
            _emit(pred_rows, fold, seed, "joint", te, pred)
            convergence.setdefault("joint", []).append(hist[-1] if hist else {})

            sm, mu, sd = train_single(Xtr, Ftr, ytr, seed=seed, d=d_sel, h_s=h_s, epochs=EPOCHS_FINAL)
            pred_s = single_predict(sm, Xte, Fte, mu, sd)
            _emit(pred_rows, fold, seed, "single", te, pred_s)

        meta.append({"fold": fold, "n_train_rows": int(len(tr)), "n_test_rows": int(len(te)),
                     "selected": selected_configs[fold], "capacity_h_s": h_s,
                     "joint_params_est": None})

    schema = pd.DataFrame(pred_rows)
    schema.to_parquet(OUT / "predictions.parquet", index=False)

    summary = build_summary(schema, meta, selected_configs, convergence, chempert,
                            time.perf_counter() - t0)
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    (OUT / "report.md").write_text(build_report(summary), encoding="utf-8")
    print("WROTE exp034_joint summary.json + predictions.parquet + report.md")
    return summary


def _emit(pred_rows, fold, seed, arm, te, pred):
    for i, r in te.iterrows():
        pred_rows.append({
            "experiment_id": EXP_ID, "run_id": f"exp034_joint_{arm}",
            "contract_id": CONTRACT_ID, "split_id": SPLIT_ID,
            "fold": int(fold), "seed": int(seed),
            "patient_id": r["patient_id"], "model_id": r["model_id"],
            "sample_id": r["sample_id"], "drug_id": r["drug_id"],
            "gene_id": None, "y_true": float(r["observed"]),
            "y_pred": None if not np.isfinite(pred[i]) else float(pred[i]),
            "evaluation_role": "test", "arm": arm,
        })


def _frame_for_arm(schema, arm, seed=None):
    f = schema[schema["arm"] == arm].copy()
    if seed is not None:
        f = f[f["seed"] == int(seed)]
    f = f.rename(columns={"y_true": "observed", "y_pred": "predicted"})
    for c in ("observed", "predicted"):
        f[c] = pd.to_numeric(f[c], errors="coerce").astype(float)
    return f


def build_summary(schema, meta, selected_configs, convergence, chempert, elapsed):
    arms = {}
    per_seed = {}
    for arm in ["joint", "single"]:
        seed_evals = []
        seed_pv = {}
        for seed in INIT_SEEDS:
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
        frs = [_frame_for_arm(schema, arm, seed=seed) for seed in INIT_SEEDS]
        conc = pd.concat(frs, ignore_index=True)
        return conc.groupby(["patient_id", "drug_id"], sort=True, as_index=False).agg(
            observed=("observed", "mean"), predicted=("predicted", "mean"))

    pb = paired_contrast(seed_avg_frame("joint"), seed_avg_frame("single"))
    paired = {"joint_minus_single": {
        "supervision": "joint: Kryzei target labels + ChemPert upstream labels; "
                       "single: Kryzei target labels only (capacity-matched)",
        "metric": "patient_macro_intra_patient_spearman (seed-averaged patient level)",
        **pb}}
    paired["holm_p"] = holm_correct([pb["p_two_sided"]]) if pb.get("p_two_sided") is not None else None
    paired["holm_applied_over"] = ["joint_minus_single"]

    return {
        "experiment_id": EXP_ID, "architecture": "E3_joint", "status": "RAN",
        "smoke": SMOKE,
        "preregistration": {
            "shared_encoder": "expression -> z (512 source-preprocessed), both Kryzei and ChemPert rows",
            "sensitivity_head": "[z, morgan2048] -> 1 (huber)",
            "perturbation_head": "[z, morgan2048] -> 795 gene signature (MSE)",
            "label_rule": "loss only on rows with the respective label; no zero-filling",
            "chempert_label": "795-gene {-1,0,+1} up/down signature (non-zero + SMILES-encodable)",
            "auxiliary_selection": "aux_weight x shared_dim inner 3-fold (<=20 candidates)",
            "capacity_control": "single-task sensitivity model with matched total parameter count",
            "init_seeds": INIT_SEEDS, "HID": HID, "LR": LR,
            "chempert_n_encodable_total": chempert["n_encodable_total"],
            "source_chempert_gene_overlap": chempert["n_overlap_source_genes"],
            "limitation": f"source 512 genes overlap ChemPert 795 control genes by only "
                          f"{chempert['n_overlap_source_genes']}; ChemPert expression through the "
                          f"shared encoder is largely median-imputed (documented).",
        },
        "arms": arms,
        "per_seed_arms": per_seed,
        "paired_gain": paired,
        "capacity_control_difference": {
            "joint_minus_single_intra": (arms["joint"]["intra_patient_spearman_mean"] -
                                         arms["single"]["intra_patient_spearman_mean"]
                                         if arms["joint"]["intra_patient_spearman_mean"] is not None
                                         and arms["single"]["intra_patient_spearman_mean"] is not None else None),
        },
        "selected_configs_per_fold": selected_configs,
        "convergence_last_epoch": convergence,
        "meta": meta,
        "elapsed_seconds": elapsed,
    }


def build_report(summary):
    L = []
    L.append("# EXP-034 E3 联合学习报告\n\n")
    L.append(f"- status: {summary['status']}（工作包 E 预登记执行）\n")
    L.append(f"- ChemPert 可编码扰动训练行（总）：{summary['preregistration']['chempert_n_encodable_total']}\n")
    L.append(f"- 源 512 基因与 ChemPert 795 控制表达基因仅重叠 "
             f"{summary['preregistration']['source_chempert_gene_overlap']} 个 "
             f"（ChemPert 侧表达经共享编码器大部分为中位数插补，已记录）。\n")
    L.append("\n## 臂指标（种子 3）\n\n")
    L.append("| 臂 | 主指标(均值±std) | 逐药 Spearman | NDCG@5 | top1/top2/top3 |\n|---|---|---|---|---|\n")
    for arm, a in summary["arms"].items():
        L.append(f"| {arm} | {a['intra_patient_spearman_mean']:.4f}±{a['intra_patient_spearman_std']:.4f} | "
                 f"{_fmt(a['per_drug_spearman'])} | {_fmt(a['ndcg5'])} | "
                 f"{_fmt(a['top1'])}/{_fmt(a['top2'])}/{_fmt(a['top3'])} |\n")
    L.append("\n## 辅助权重/共享维度选择（每折）\n\n")
    L.append("| fold | shared_dim | aux_weight |\n|---|---|---|\n")
    for fold, cfg in summary["selected_configs_per_fold"].items():
        L.append(f"| {fold} | {cfg['shared_dim']} | {cfg['aux_weight']} |\n")
    L.append("\n## 配对增益（joint vs single 容量对照，患者级 2000 bootstrap + Holm）\n\n")
    pg = summary["paired_gain"]
    v = pg["joint_minus_single"]
    L.append(f"- joint_minus_single 点估计 {_fmt(v['point_estimate'])}，"
             f"CI [{_fmt(v['low'])}, {_fmt(v['high'])}]，p_two {_fmt(v.get('p_two_sided'))}。\n")
    L.append(f"- Holm p：{pg['holm_p']}\n")
    L.append("\n## 判定建议\n\n")
    low, high = v["low"], v["high"]
    if low is not None and high is not None and low > 0:
        verdict = "支持（联合学习相对容量匹配单任务提供可归因增量）"
    elif low is not None and high is not None and low <= 0 <= high:
        verdict = "证据不足（joint 相对 single 的 CI 跨零）"
    else:
        verdict = "不支持（联合学习无显著正向增量）"
    L.append(f"- **判定：{verdict}**\n")
    L.append("\n## unknown / 限制\n\n")
    L.append("- ChemPert 侧共享表达仅 48/512 基因为实测，其余为中位数插补；"
             "辅助任务的表达共享有限。\n")
    L.append("- 实测扰动参考：Kryzei 无同样本同处理条件实测扰动，无法检验该项参考。\n")
    return "".join(L)


def _fmt(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "None"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


if __name__ == "__main__":
    main()