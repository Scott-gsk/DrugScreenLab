#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-033 工作包 D 任务 3：D3-C4 臂（扰动特征 + 目标药敏监督）。

按 EXP-032 L4-100% 的 C1 协议（DualEncoderPairModel，hidden=32，native
ExpressionPreprocessor 512 维表达 + Morgan 2048 指纹），额外拼接任务 1 的
whitelist 扰动特征（reversal_score / pathway_reversal / perturb_magnitude 的
3 种子均值，逐 (sample,drug) 行标准化到训练折统计）→ C4。

对比：C4 vs C1（无扰动特征）同折同子集配对（患者级 intra-patient Spearman
差值，2000 次 bootstrap）+ Holm。C1 复用 EXP-032 落盘预测
(artifacts/exp032_l3l4/predictions.parquet, arm=C1 frac=1.00 seed=20260921)；
C4 以同一 fold map / 子集 RNG / 预处理协议训练。

输出 artifacts/exp033_routeB/c4_summary.json。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from drug_screen.transfer.cohorts import load_registered_cohort
from drug_screen.transfer.benchmark import _table
from drug_screen.transfer.models import DualEncoderPairModel
from drug_screen.transfer.preprocessing import ExpressionPreprocessor
from drug_screen.transfer.chemistry import morgan_fingerprint
from drug_screen.data.admission import AdmissionWhitelist
from drug_screen.evaluation.exp031_eval import (
    intra_patient_spearman_patient_macro,
    paired_bootstrap_ci,
    holm_correct,
)

PROJ = Path("/mnt/d/Code/DrugScreenLab")
OUT = PROJ / "artifacts" / "exp033_routeB"
AUDIT = json.loads((PROJ / "data/processed/identity/kryzei_admission_audit_20260921.json").read_text(encoding="utf-8"))
SPLITS = json.loads((PROJ / "data/splits/kryzei_shared_splits_v1.json").read_text(encoding="utf-8"))
FEATS = PROJ / "artifacts/exp033_routeB/whitelist_chempert_features.parquet"
EXP032_PRED = PROJ / "artifacts/exp032_l3l4/predictions.parquet"

PARAMS = dict(hidden_dim=32, epochs=50, learning_rate=1e-3, ranking_weight=0.2,
              max_rank_pairs=128, batch_groups=8)
SUBSET_RNG_SEED = 20260921
BOOT_SEED = 20260918
N_BOOT = 2000
SEED = 20260921  # L4-100% 单种子
FRAC = 1.00
DEVICE = "cuda"


def load_all():
    bundle = load_registered_cohort(
        "kryeziu_mcrc_v3", root=PROJ,
        registry=PROJ / "data/registry/transfer_cohorts.json")
    table = _table(bundle)
    sp = dict(zip(bundle.samples.sample_id.astype(str), bundle.samples.patient_id.astype(str)))
    wl = AdmissionWhitelist(AUDIT["patient_whitelist"], AUDIT["pdo_whitelist"],
                            patient_of_sample=sp)
    admitted, _ = wl.reject_non_admitted(
        table.sample_id.astype(str).tolist(), patient_of_sample=sp)
    rows = table[table.sample_id.astype(str).isin(set(admitted))].copy()
    for c in ("patient_id", "model_id", "drug_id", "sample_id"):
        rows[c] = rows[c].astype(str).str.strip()
    rows["observed"] = pd.to_numeric(rows.response, errors="raise").astype(float)

    expression = np.asarray(bundle.expression, dtype=float)
    genes = np.asarray(bundle.gene_ids, dtype=str)
    chem = bundle.drugs.copy()
    field = "canonical_isomeric_smiles"
    lookup = dict(zip(chem.drug_id.astype(str), chem[field], strict=True))
    fps = {str(d): morgan_fingerprint(str(lookup[str(d)]), radius=2, n_bits=2048).astype(float)
           for d in rows.drug_id.unique()}
    return dict(bundle=bundle, rows=rows, expression=expression, genes=genes, fps=fps)


def fold_map():
    fm = {}
    for f in SPLITS["outer_folds"]:
        fm[int(f["fold"])] = {
            "test_patients": sorted(f["test_patient_ids"]),
            "train_patients": sorted(f["train_patient_ids"])}
    return fm


def _sp(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    if len(y) < 3 or np.ptp(y) == 0 or np.ptp(p) == 0:
        return np.nan
    r = spearmanr(y, p).statistic
    return float(r) if np.isfinite(r) else np.nan


def arm_eval(frame):
    macro, n_valid, per_patient = intra_patient_spearman_patient_macro(
        frame, min_drugs=3)
    return {"intra_patient_spearman": macro, "n_valid_patients": n_valid,
            "n_patients": int(frame.patient_id.nunique()),
            "n_drugs": int(frame.drug_id.nunique()),
            "reason_counts": {str(k): int(v) for k, v in
                              per_patient.loc[per_patient.reason.notna(), "reason"].value_counts().items()}}


def load_perturb_features():
    f = pd.read_parquet(FEATS)
    f["sample_id"] = f["sample_id"].astype(str)
    f["drug_id"] = f["drug_id"].astype(str)
    agg = f.groupby(["sample_id", "drug_id"], sort=True).agg(
        reversal=("reversal_score", "mean"),
        pathway=("pathway_reversal", "mean"),
        magnitude=("perturb_magnitude", "mean")).reset_index()
    return {(r.sample_id, r.drug_id): [r.reversal, r.pathway, r.magnitude]
            for r in agg.itertuples(index=False)}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_all()
    fm = fold_map()
    rows = data["rows"].reset_index(drop=True)
    fps = data["fps"]
    expression = data["expression"]
    genes = data["genes"]
    pert = load_perturb_features()

    # per-row perturb features aligned to rows order
    perturb_matrix = np.array([pert[(s, d)] for s, d in
                               zip(rows.sample_id, rows.drug_id)], dtype=float)
    missing = int(np.isnan(perturb_matrix).sum())
    if missing:
        perturb_matrix = np.nan_to_num(perturb_matrix, nan=0.0)

    c4_rows = []
    for fold in sorted(fm):
        foldinfo = fm[fold]
        test_patients = set(foldinfo["test_patients"])
        train_patients = set(foldinfo["train_patients"])
        tr = rows[rows.patient_id.isin(train_patients)].copy()
        te = rows[rows.patient_id.isin(test_patients)].copy()

        tr_pos = tr.index.to_numpy()   # positions in `rows`
        te_pos = te.index.to_numpy()

        native_prep = ExpressionPreprocessor.fit(
            expression[np.unique(tr.expression_index.to_numpy(int))], genes,
            n_features=512)
        native_x = native_prep.transform(expression, genes)

        # perturbation features standardized on train rows (fit-on-train)
        p_tr = perturb_matrix[tr_pos]
        p_mu = p_tr.mean(axis=0)
        p_sd = p_tr.std(axis=0) + 1e-6
        p_all = (perturb_matrix - p_mu) / p_sd

        # C4 输入：每行 [512 表达 | 3 扰动]
        tr_features = np.concatenate(
            [native_x[tr.expression_index.to_numpy(int)], p_all[tr_pos]], axis=1)
        te_features = np.concatenate(
            [native_x[te.expression_index.to_numpy(int)], p_all[te_pos]], axis=1)
        tr_vec = np.vstack([fps[str(d)] for d in tr.drug_id])
        te_vec = np.vstack([fps[str(d)] for d in te.drug_id])

        torch.manual_seed(SEED)
        model = DualEncoderPairModel(**PARAMS, seed=SEED, device=DEVICE)
        model.fit(tr_features, tr_vec, tr.observed.to_numpy(float),
                  pdo_ids=tr.model_id.astype(str).to_numpy())
        pred = model.predict(te_features, te_vec)

        c4_rows.append(pd.DataFrame({
            "patient_id": te.patient_id, "drug_id": te.drug_id,
            "sample_id": te.sample_id, "observed": te.observed.to_numpy(float),
            "predicted": pred, "fold": fold}))

    c4 = pd.concat(c4_rows, ignore_index=True)
    c4_eval = arm_eval(c4)

    # C1 复用 EXP-032 落盘
    ep = pd.read_parquet(EXP032_PRED)
    c1 = ep[(ep["arm"] == "C1") & (ep["frac"] == 1.00)
            & (ep["seed"] == SEED)].copy()
    c1 = c1.rename(columns={"y_true": "observed", "y_pred": "predicted"})
    c1_eval = arm_eval(c1)

    # 配对 bootstrap（患者级 intra-patient Spearman 差）
    def agg_pd(fr):
        f = fr[["patient_id", "drug_id", "observed", "predicted"]].copy()
        for c in ("observed", "predicted"):
            f[c] = pd.to_numeric(f[c], errors="coerce").astype(float)
        return f.groupby(["patient_id", "drug_id"], sort=True, as_index=False).agg(
            observed=("observed", "mean"), predicted=("predicted", "mean"))

    gan = paired_bootstrap_ci(agg_pd(c4), agg_pd(c1), "patient_id", _sp,
                              n=N_BOOT, seed=BOOT_SEED, truth_col="observed",
                              pred_col="predicted")
    deltas = np.asarray(gan.deltas, float)
    rng = np.random.default_rng(BOOT_SEED)
    boots = deltas[rng.integers(0, len(deltas), size=(N_BOOT, len(deltas)))].mean(axis=1)
    p_greater = float(np.mean(boots <= 0))
    p_two = float(min(1.0, 2.0 * min(p_greater, 1.0 - p_greater)))

    summary = {
        "experiment_id": "EXP-033", "task": "D3-C4 (perturbation features + target supervision)",
        "status": "RAN",
        "C1_protocol": "EXP-032 L4-100% (hidden32 dual encoder, native 512 + Morgan2048)",
        "C4": "C1 + 3 whitellist perturb features (reversal/pathway/magnitude, 3-seed mean, "
              "per-fold train-standardized), sample_dim=515",
        "seed": SEED, "frac": FRAC, "subset_rng_seed": SUBSET_RNG_SEED,
        "perturb_missing_rows_filled_zero": missing,
        "C1_source": str(EXP032_PRED),
        "C1_eval": c1_eval,
        "C4_eval": c4_eval,
        "paired_C4_minus_C1": {
            "point_estimate": gan.point_estimate, "low": gan.low, "high": gan.high,
            "n_units": gan.n_units, "n": N_BOOT, "seed": BOOT_SEED,
            "p_greater_than_zero": round(1.0 - p_greater, 6),
            "p_two_sided": round(p_two, 6),
            "holm_p": holm_correct([p_two])[0],
            "metric": "intra_patient_spearman (patient-level paired bootstrap)",
        },
        "drug_identity_dominance_check": {
            "note": "按预登记口径报告增量与残差相关检查；C1 药身份主导检查见 EXP-032 "
                    "L4 (target_fold_drug_mean 常数臂为逐药均值锚点)。C4 vs C1 增量 "
                    "即扰动特征的边际价值（药身份在此对比中已同时存在于两侧）。",
        },
        "supervision": "C1/C4 均用目标训练折药敏标签（同监督条件，可配对比较）",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT / "c4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8")
    # 也落盘 C4 预测长表便于审计
    c4.assign(arm="C4", frac=1.00).to_parquet(OUT / "c4_predictions.parquet", index=False)
    print("WROTE c4_summary.json + c4_predictions.parquet")
    print(json.dumps({
        "C1_intra": c1_eval["intra_patient_spearman"],
        "C4_intra": c4_eval["intra_patient_spearman"],
        "C4_minus_C1_point": summary["paired_C4_minus_C1"]["point_estimate"],
        "CI": [summary["paired_C4_minus_C1"]["low"], summary["paired_C4_minus_C1"]["high"]],
        "holm_p": summary["paired_C4_minus_C1"]["holm_p"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()