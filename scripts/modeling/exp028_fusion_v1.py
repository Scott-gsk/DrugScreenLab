#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EXP-028 fusion v1: nonlinear second layer over dual-route features.

Pre-registered contract (experiments/records/EXP-028.md, pinned BEFORE run):

- Per fold (frozen split): second layer trains on VAL patients' rows (~490)
  with route A feature = inner-model predictions (no leakage); test rows use
  outer-model predictions; route B features are zero-shot.
- Feature vector (2103 dims): route A pred seed-mean (1) + route B scalars
  (reversal/magnitude mean+var over seeds 23/24/25, 4) + Hallmark-50 pathway
  delta seed-mean (50) + Morgan 2048 (radius=2, interaction capacity).
- Arms:
    F0_GBM_predonly   HistGBR on [routeA_pred]
    F1_GBM_full       HistGBR on all 2103
    F0_MLP_predonly   MLP 2103-slot->32 ReLU->1 on [routeA_pred], seeds 31/32/33
    F2_MLP_full       MLP on all 2103
    F1_GBM_full_xfit  HistGBR full, trained on TRAIN patients' rows with
                      cross-fit route A features (xfit_oof), test on test rows
  References: A2_frozen (route A outer OOF), B0_frozen (per-fold drug means).
- GBM pinned: lr=0.05, max_iter=200, max_depth=3, min_samples_leaf=20,
  l2=1.0, early_stopping=False, random_state=20260918.
- MLP pinned: 32 hidden, Adam lr=1e-3 wd=1e-2, epochs=100, batch=128,
  StandardScaler fit on training rows.
- Endpoints: evaluate_stratified triple (nonstructural NDCG@5 primary /
  per-drug cross-patient Spearman / full-panel reference); MLP arms mean±SD.
- Pre-registered patient bootstrap (2000x, seed=20260918, seed-averaged
  patient values, patient intersection):
    F1 vs A2 (primary), F2 vs A2, F0_GBM vs A2, F0_MLP vs A2,
    F1 vs F0_GBM, F2 vs F0_MLP, F1_xfit vs A2, A2 vs B0.

Outputs:
  artifacts/exp028_fusion/fusion_predictions.parquet
  artifacts/exp028_fusion/fusion_summary.json
  artifacts/exp028_fusion/bootstrap_fusion.json
  artifacts/exp028_fusion/per_drug_fusion.json
  docs/reports/fusion_EXP028_20260920.md
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from drug_screen.evaluation.stratified_metrics import (
    evaluate_stratified, load_structural_drugs,
    nonstructural_patient_values, full_panel_patient_values,
)
from drug_screen.transfer.cohorts import load_registered_cohort
from drug_screen.transfer.transfer_experiment import _fingerprints

ROOT = Path("/mnt/d/Code/DrugScreenLab")
REGISTRY = ROOT / "data/registry/transfer_cohorts.json"
COHORT_ID = "kryeziu_mcrc_v3"
SOURCE_DIR = ROOT / "data/artifacts/EXP-015/source_seed20260912"
SPLIT = ROOT / "data/splits/kryeziu_patient_outer5fold_FROZEN.json"
STRUCTURAL_PATH = ROOT / "data/processed/identity/kryzei_structural_drugs_FROZEN.json"
IN_DIR = ROOT / "artifacts/exp028_fusion"
REPORT = ROOT / "docs/reports/fusion_EXP028_20260920.md"

SEEDS_A = [20260912, 20260913, 20260919]
MLP_SEEDS = [31, 32, 33]
BOOT_RS, BOOT_N = 20260918, 2000

GBM_PARAMS = dict(learning_rate=0.05, max_iter=200, max_depth=3,
                  min_samples_leaf=20, l2_regularization=1.0,
                  early_stopping=False, random_state=BOOT_RS)

SCALAR_COLS = ["reversal_mean", "reversal_var",
               "magnitude_mean", "magnitude_var"]


# ---------------------------------------------------------------- helpers
class LayerMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(),
                                 nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(x_train: np.ndarray, y_train: np.ndarray, dim: int,
            seed: int) -> LayerMLP:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LayerMLP(dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(torch.from_numpy(x_train.astype(np.float32)),
                      torch.from_numpy(y_train.astype(np.float32)))
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    model.train()
    for _ in range(100):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def mlp_predict(model: LayerMLP, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x.astype(np.float32))).numpy().astype(float)


def patient_bootstrap(diff: dict[str, float], seed: int = BOOT_RS,
                      resamples: int = BOOT_N) -> dict | None:
    vals = [v for v in diff.values() if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return None
    point = float(np.mean(vals))
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals)
    idx = rng.integers(0, len(vals), (resamples, len(vals)))
    draws = vals[idx].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"point": point, "low": float(low), "high": float(high),
            "n": len(vals), "resamples": resamples}


def fmt(x, sd=None):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{x:.4f}"
    if sd is not None and not np.isnan(sd):
        s += f" ±{sd:.4f}"
    return s


# ---------------------------------------------------------------- features
def build_side_features() -> tuple[pd.DataFrame, dict[str, np.ndarray], list[str]]:
    """Route B seed-aggregated features per (sample, drug) + Morgan 2048."""
    rb = pd.read_parquet(IN_DIR / "routeB_features_exp028.parquet")
    pw_cols = [c for c in rb.columns if c.startswith("pw_")]
    scalars = (rb.groupby(["sample_id", "drug_id"], sort=True)
               .agg(reversal_mean=("reversal_score", "mean"),
                    reversal_var=("reversal_score", "var"),
                    magnitude_mean=("perturb_magnitude", "mean"),
                    magnitude_var=("perturb_magnitude", "var"))
               .reset_index())
    pw_mean = (rb.groupby(["sample_id", "drug_id"], sort=True)[pw_cols]
               .mean().reset_index())
    side = scalars.merge(pw_mean, on=["sample_id", "drug_id"], validate="one_to_one")
    assert side[SCALAR_COLS + pw_cols].notna().all().all(), "NaN in side features"

    bundle = load_registered_cohort(COHORT_ID, root=ROOT, registry=REGISTRY)
    drugs = pd.Series(sorted(rb["drug_id"].astype(str).unique()))
    fps = _fingerprints(bundle, drugs)
    return side, fps, pw_cols


def role_frame(ra: pd.DataFrame, fold: int, role: str,
               patients: set[str]) -> pd.DataFrame:
    """(model_id, drug_id) -> route A pred (seed-mean) + observed + identity."""
    sub = ra[(ra["algorithm"] == "M1_source_scaled") & (ra["role"] == role)
             & (ra["fold"] == fold) & (ra["patient_id"].isin(patients))]
    assert sub["seed"].nunique() == len(SEEDS_A), f"{role}: missing route A seeds"
    base = sub[sub["seed"] == SEEDS_A[0]][
        ["patient_id", "model_id", "drug_id", "observed", "source_drug_status"]]
    pred = (sub.groupby(["model_id", "drug_id"], sort=True)["predicted"]
            .mean().reset_index().rename(columns={"predicted": "routeA_pred"}))
    out = base.merge(pred, on=["model_id", "drug_id"],
                     how="left", validate="one_to_one")
    assert out["routeA_pred"].notna().all(), f"{role}: missing route A pred"
    return out


def design_matrix(rows: pd.DataFrame, side: pd.DataFrame,
                  pw_cols: list[str], fps: dict[str, np.ndarray]) -> np.ndarray:
    n = len(rows)
    merged = rows.merge(side, left_on=["model_id", "drug_id"],
                        right_on=["sample_id", "drug_id"],
                        how="left", validate="one_to_one", suffixes=("", "_side"))
    assert merged[SCALAR_COLS + pw_cols].notna().all().all(), "missing side feats"
    X = np.empty((n, 1 + len(SCALAR_COLS) + len(pw_cols) + 2048), dtype=np.float64)
    X[:, 0] = merged["routeA_pred"].to_numpy(float)
    X[:, 1:1 + len(SCALAR_COLS)] = merged[SCALAR_COLS].to_numpy(float)
    X[:, 1 + len(SCALAR_COLS):1 + len(SCALAR_COLS) + len(pw_cols)] = \
        merged[pw_cols].to_numpy(float)
    X[:, 1 + len(SCALAR_COLS) + len(pw_cols):] = \
        np.vstack([fps[str(d)] for d in merged["drug_id"]])
    assert X.shape[1] == 2103, f"expected 2103 feature dims, got {X.shape[1]}"
    return X


# ---------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    structural = load_structural_drugs(STRUCTURAL_PATH)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    ra = pd.read_parquet(IN_DIR / "routeA_frozen_predictions.parquet")
    frozen = json.loads((SOURCE_DIR / "frozen_run.json").read_text(encoding="utf-8"))
    source_drugs = set(frozen["source_drug_ids"])
    side, fps, pw_cols = build_side_features()
    print(f"[fusion] side features: {len(side)} rows, "
          f"{1 + len(SCALAR_COLS) + len(pw_cols) + 2048} design dims", flush=True)

    pred_rows: list[pd.DataFrame] = []
    for fd in split["folds"]:
        fold = int(fd["fold"])
        val_p = set(map(str, fd["val_patient_ids"]))
        test_p = set(map(str, fd["test_patient_ids"]))
        train_p = set(map(str, fd["train_patient_ids"]))

        val_rows = role_frame(ra, fold, "inner_val", val_p)
        test_rows = role_frame(ra, fold, "outer_test", test_p)
        xfit_rows = role_frame(ra, fold, "xfit_oof", train_p)
        X_val = design_matrix(val_rows, side, pw_cols, fps)
        X_test = design_matrix(test_rows, side, pw_cols, fps)
        X_xfit = design_matrix(xfit_rows, side, pw_cols, fps)
        y_val = val_rows["observed"].to_numpy(float)
        y_xfit = xfit_rows["observed"].to_numpy(float)
        print(f"[fusion] fold={fold} val_rows={len(val_rows)} "
              f"xfit_rows={len(xfit_rows)} test_rows={len(test_rows)}", flush=True)

        def emit(arm: str, pred: np.ndarray, rows: pd.DataFrame,
                 layer_seed: int | None = None) -> None:
            pred_rows.append(pd.DataFrame({
                "patient_id": rows["patient_id"].astype(str),
                "model_id": rows["model_id"].astype(str),
                "drug_id": rows["drug_id"].astype(str),
                "fold": fold, "algorithm": arm,
                "layer_seed": layer_seed if layer_seed is not None else -1,
                "observed": rows["observed"].to_numpy(float),
                "predicted": np.asarray(pred, dtype=float),
                "source_drug_status": rows["source_drug_status"],
            }))

        pred_only_val, pred_only_test = X_val[:, :1], X_test[:, :1]

        # ---- GBM arms (val-holdout mechanism) ----
        for arm, xv, xt in (("F0_GBM_predonly", pred_only_val, pred_only_test),
                            ("F1_GBM_full", X_val, X_test)):
            gbm = HistGradientBoostingRegressor(**GBM_PARAMS)
            gbm.fit(xv, y_val)
            emit(arm, gbm.predict(xt), test_rows)

        # ---- MLP arms (layer seeds 31/32/33) ----
        for arm, xv, xt in (("F0_MLP_predonly", pred_only_val, pred_only_test),
                            ("F2_MLP_full", X_val, X_test)):
            for ls in MLP_SEEDS:
                scaler = StandardScaler().fit(xv)
                model = fit_mlp(scaler.transform(xv), y_val, xv.shape[1], ls)
                emit(arm, mlp_predict(model, scaler.transform(xt)),
                     test_rows, layer_seed=ls)

        # ---- sensitivity arm: cross-fit training rows ----
        gbm_x = HistGradientBoostingRegressor(**GBM_PARAMS)
        gbm_x.fit(X_xfit, y_xfit)
        emit("F1_GBM_full_xfit", gbm_x.predict(X_test), test_rows)

    fusion = pd.concat(pred_rows, ignore_index=True)
    fusion.to_parquet(IN_DIR / "fusion_predictions.parquet", index=False)
    print(f"[fusion] predictions rows={len(fusion)} "
          f"algorithms={sorted(fusion.algorithm.unique())}", flush=True)

    # ============ evaluation ============
    def ref_frame(algorithm: str, role: str, algo_src: str,
                  seed: int | None) -> pd.DataFrame:
        m = (ra["algorithm"] == algo_src) & (ra["role"] == role)
        if seed is not None:
            m &= ra["seed"] == seed
        sub = ra.loc[m]
        return pd.DataFrame({
            "patient_id": sub["patient_id"].astype(str),
            "model_id": sub["model_id"].astype(str),
            "drug_id": sub["drug_id"].astype(str),
            "fold": sub["fold"], "algorithm": algorithm,
            "layer_seed": seed if seed is not None else -1,
            "observed": sub["observed"].to_numpy(float),
            "predicted": sub["predicted"].to_numpy(float),
            "source_drug_status": sub["source_drug_status"],
        })

    slices: list[tuple[str, int | None, pd.DataFrame]] = []
    for (arm, ls), df in fusion.groupby(["algorithm", "layer_seed"], sort=True):
        slices.append((arm, int(ls), df))
    for seed in SEEDS_A:
        slices.append(("A2_frozen", seed,
                       ref_frame("A2_frozen", "outer_test", "M1_source_scaled", seed)))
    slices.append(("B0_frozen", None, ref_frame(
        "B0_frozen", "outer_test", "B0", SEEDS_A[0])))

    results = []
    patient_ns: dict[str, dict[str, list[float]]] = {}
    for arm, ls, df in slices:
        summary, _, _ = evaluate_stratified(df, structural, algorithm=arm)
        results.append({"arm": arm, "layer_seed": ls,
                        **{k: v for k, v in summary.as_dict().items()
                           if k != "algorithm"}})
        known = df[df["source_drug_status"] == "known"]
        ns = nonstructural_patient_values(known, structural)
        ns_p = ns.rename("v").reset_index().groupby("level_0")["v"].mean()
        for pid, v in ns_p.items():
            if np.isfinite(v):
                patient_ns.setdefault(arm, {}).setdefault(pid, []).append(float(v))

    frame = pd.DataFrame(results)
    IN_DIR.joinpath("fusion_summary.json").write_text(
        frame.to_json(orient="records", indent=2, force_ascii=False) + "\n",
        encoding="utf-8")

    # ============ pre-registered paired bootstrap ============
    pat_avg = {arm: {pid: float(np.mean(v)) for pid, v in d.items()}
               for arm, d in patient_ns.items()}

    def paired(arm_x: str, arm_y: str) -> dict | None:
        common = sorted(set(pat_avg.get(arm_x, {})) & set(pat_avg.get(arm_y, {})))
        diff = {p: pat_avg[arm_x][p] - pat_avg[arm_y][p] for p in common}
        return patient_bootstrap(diff) if diff else None

    boot = {
        "F1_GBM_full_vs_A2_frozen": paired("F1_GBM_full", "A2_frozen"),
        "F2_MLP_full_vs_A2_frozen": paired("F2_MLP_full", "A2_frozen"),
        "F0_GBM_predonly_vs_A2_frozen": paired("F0_GBM_predonly", "A2_frozen"),
        "F0_MLP_predonly_vs_A2_frozen": paired("F0_MLP_predonly", "A2_frozen"),
        "F1_GBM_full_vs_F0_GBM_predonly": paired("F1_GBM_full", "F0_GBM_predonly"),
        "F2_MLP_full_vs_F0_MLP_predonly": paired("F2_MLP_full", "F0_MLP_predonly"),
        "F1_GBM_full_xfit_vs_A2_frozen": paired("F1_GBM_full_xfit", "A2_frozen"),
        "A2_frozen_vs_B0_frozen": paired("A2_frozen", "B0_frozen"),
    }
    (IN_DIR / "bootstrap_fusion.json").write_text(
        json.dumps(boot, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")

    # ============ per-drug stratification ============
    per_drug_rows = []
    drug_tables = {
        "A2_frozen": pd.concat([s for a, _, s in slices if a == "A2_frozen"],
                               ignore_index=True),
        "F1_GBM_full": next(s for a, _, s in slices if a == "F1_GBM_full"),
        "F1_GBM_full_xfit": next(s for a, _, s in slices if a == "F1_GBM_full_xfit"),
    }
    for arm, fr in drug_tables.items():
        known = fr[fr["source_drug_status"] == "known"]
        per_patient = (known.groupby(["drug_id", "patient_id"], sort=True)
                       .agg(observed=("observed", "mean"),
                            predicted=("predicted", "mean")).reset_index())
        for drug, g in per_patient.groupby("drug_id", sort=True):
            y = g["observed"].to_numpy(float)
            p = g["predicted"].to_numpy(float)
            if len(y) < 5 or np.ptp(y) == 0 or np.ptp(p) == 0:
                rho = float("nan")
            else:
                rho = float(spearmanr(y, p).statistic)
            per_drug_rows.append({"arm": arm, "drug_id": drug, "spearman": rho,
                                  "n_patients": int(len(g))})
    pd_frame = pd.DataFrame(per_drug_rows)
    pivot = pd_frame.pivot(index="drug_id", columns="arm", values="spearman")
    pivot["delta_F1_minus_A2"] = pivot["F1_GBM_full"] - pivot["A2_frozen"]
    pivot["delta_xfit_minus_A2"] = (pivot["F1_GBM_full_xfit"]
                                    - pivot["A2_frozen"])
    pivot = pivot.assign(structural=lambda d: [
        "structural" if i in structural.structural else "nonstructural"
        for i in d.index]).sort_values(["structural", "delta_F1_minus_A2"],
                                       ascending=[True, False])
    pivot.reset_index().to_json(IN_DIR / "per_drug_fusion.json",
                                orient="records", indent=2, force_ascii=False)

    # ============ report ============
    agg = (frame.groupby("arm")
           .agg(ns_mean=("nonstructural_ndcg5_macro", "mean"),
                ns_sd=("nonstructural_ndcg5_macro", "std"),
                pd_mean=("per_drug_patient_spearman_macro", "mean"),
                pd_sd=("per_drug_patient_spearman_macro", "std"),
                fp_mean=("full_panel_ndcg5_macro", "mean"),
                fp_sd=("full_panel_ndcg5_macro", "std"),
                n_runs=("layer_seed", "count")).reset_index())
    order = ["B0_frozen", "A2_frozen", "F0_GBM_predonly", "F0_MLP_predonly",
             "F1_GBM_full", "F2_MLP_full", "F1_GBM_full_xfit"]
    agg["rank"] = agg["arm"].map({a: i for i, a in enumerate(order)})
    agg = agg.sort_values("rank").drop(columns="rank")

    lines = [
        "# EXP-028 双路线非线性融合 v1 报告",
        "",
        "- 生成：scripts/modeling/exp028_fusion_v1.py（EXP-028 预登记协议）",
        "- 机制：第二层在 val 患者行上训练（路线 A 特征=内层模型预测，无泄漏），"
        "预测 test 患者行；路线 B 特征零样本",
        "- 特征 2,103 维：路线 A 预测（3 种子均值）1 + 路线 B 标量 4 + "
        "Hallmark 通路 delta 50 + Morgan 2048",
        "",
        "## 一、臂级总表（三终点；MLP 为 3 种子 mean±SD）",
        "",
        "| 臂 | 非结构 NDCG@5（主） | 逐药跨患者 Spearman | 全面板参照 |",
        "|---|---|---|---|",
    ]
    for _, r in agg.iterrows():
        lines.append(f"| {r['arm']} | {fmt(r.ns_mean, r.ns_sd)} | "
                      f"{fmt(r.pd_mean, r.pd_sd)} | {fmt(r.fp_mean, r.fp_sd)} |")
    lines += [
        "",
        "## 二、预登记配对比较（患者级 2000 次 bootstrap，非结构 NDCG@5）",
        "",
        "| 比较 | 点估计 | 95% CI | n |",
        "|---|---|---|---|",
    ]
    for key, b in boot.items():
        if b is None:
            lines.append(f"| {key} | 不可算 | — | — |")
        else:
            lines.append(f"| {key} | {b['point']:+.4f} | "
                         f"[{b['low']:+.4f}, {b['high']:+.4f}] | {b['n']} |")
    lines += [
        "",
        "## 三、按药分层（哪些药融合获益；ΔSpearman = 融合 − 路线 A）",
        "",
        "| 药物 | 结构性 | A2 | F1 | F1_xfit | Δ(F1−A2) | Δ(xfit−A2) |",
        "|---|---|---|---|---|---|---|",
    ]
    for drug_id, r in pivot.iterrows():
        lines.append(
            f"| {drug_id} | {r['structural']} | {fmt(r.get('A2_frozen'))} | "
            f"{fmt(r.get('F1_GBM_full'))} | {fmt(r.get('F1_GBM_full_xfit'))} | "
            f"{fmt(r.get('delta_F1_minus_A2'))} | "
            f"{fmt(r.get('delta_xfit_minus_A2'))} |")
    lines += [
        "",
        "## 四、结果判读",
        "",
        "（待执行后回填；判读框架见 experiments/records/EXP-028.md 预登记。）",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("REPORT ->", REPORT)
    print(agg.to_string())
    print("\nBOOTSTRAP:")
    for k, v in boot.items():
        print(" ", k, v)
    print(f"\n[fusion] DONE elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
