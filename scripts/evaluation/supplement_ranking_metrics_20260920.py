"""事后描述性补充分析（2026-09-20，应用户要求执行）。

补充两个整体排序指标（此前预登记终点未包含，本次只读落盘预测补算，
不训练任何模型、不改变任何预登记判定）：

1. 患者内全药列表 Spearman：每个类器官培养物在其全部已知药物（资格 >=10）
   上，预测排序与实际排序的 Spearman 一致性；培养物值先在患者内等权平均，
   再做患者宏平均。
2. 患者内非结构药子集 Spearman：同上，但只在 7 个患者间差异大的药物上
   （资格 >=5）。

数据源（全部为已落盘预测/特征）：
- EXP-021 官方迁移三真种子预测（B0/M1/M1_source_scaled/M2/M3）；
- EXP-028 冻结协议直接路线预测（M1_source_scaled=A2、B0）与融合五臂；
- EXP-027/028 间接路线零样本特征（B2=逆转评分、B3=通路逆转评分，3 种子平均）。

输出：artifacts/supplementary_ranking_metrics_20260920/
  supplement_results.json + supplement_tables.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from drug_screen.evaluation.stratified_metrics import load_structural_drugs

ROOT = Path("/mnt/d/Code/DrugScreenLab")
STRUCTURAL_PATH = ROOT / "data/processed/identity/kryzei_structural_drugs_FROZEN.json"
OUT_DIR = ROOT / "artifacts/supplementary_ranking_metrics_20260920"

OFFICIAL_RUNS = {
    "20260912": "run_b2000_20260918_174702",
    "20260913": "run_b2000_seed20260913_20260918_223357",
    "20260919": "run_b2000_20260919_20260919_121930",
}
MISLABELED_RUN = "run_b2000_seed2_20260918_223331"  # 不计入（重复轮，EXP-021 口径）

ROUTEA = ROOT / "artifacts/exp028_fusion/routeA_frozen_predictions.parquet"
FUSION = ROOT / "artifacts/exp028_fusion/fusion_predictions.parquet"
ROUTEB = ROOT / "artifacts/exp028_fusion/routeB_features_exp028.parquet"

SCOPES = {
    "full_panel_spearman": {"min_panel": 10, "drugs": "known"},
    "nonstructural_spearman": {"min_panel": 5, "drugs": "nonstructural"},
}
BOOT_RESAMPLES = 2000
BOOT_SEED = 0


def load_structural() -> tuple[frozenset, frozenset]:
    sd = load_structural_drugs(STRUCTURAL_PATH)
    return frozenset(sd.structural), frozenset(sd.nonstructural)


def find_official_dir() -> Path:
    """路径直写会因文件系统异常失败，改用枚举定位真实目录。"""
    art = ROOT / "artifacts"
    for child in art.iterdir():
        if "official_transfer" in child.name and child.is_dir():
            return child
    raise FileNotFoundError("official transfer dir not found via enumeration")


def per_pdo_spearman(df: pd.DataFrame, drug_set: frozenset | None,
                     min_panel: int) -> pd.Series:
    """逐培养物（model_id）的预测-观测 Spearman。"""
    if drug_set is not None:
        df = df[df["drug_id"].isin(drug_set)]
    vals: dict[str, float] = {}
    for mid, g in df.groupby("model_id"):
        g = g.dropna(subset=["observed", "predicted"])
        if len(g) < min_panel:
            continue
        if g["observed"].nunique() < 2 or g["predicted"].nunique() < 2:
            continue
        vals[mid] = float(spearmanr(g["predicted"], g["observed"])[0])
    return pd.Series(vals)


def patient_values(pdo_vals: pd.Series,
                   pdo_to_patient: pd.Series) -> pd.Series:
    """培养物值 → 患者内等权平均。"""
    if pdo_vals.empty:
        return pd.Series(dtype=float)
    pat = pdo_to_patient.reindex(pdo_vals.index).dropna()
    return pdo_vals.reindex(pat.index).groupby(pat).mean()


def arm_patient_values(df: pd.DataFrame, known: frozenset,
                       nonstructural: frozenset) -> dict[str, pd.Series]:
    """对单个臂（单种子或已平均）计算两个口径的患者级值。"""
    out = {}
    base = df[df["source_drug_status"] == "known"] if "source_drug_status" in df else df
    pdo_to_patient = base.groupby("model_id")["patient_id"].first()
    for scope, cfg in SCOPES.items():
        drug_set = known if cfg["drugs"] == "known" else nonstructural
        pdo_vals = per_pdo_spearman(base, drug_set, cfg["min_panel"])
        out[scope] = patient_values(pdo_vals, pdo_to_patient)
    return out


def mean_across_seeds(per_seed: list[dict[str, pd.Series]]) -> dict[str, pd.Series]:
    """对每口径把多个种子的患者级值平均。"""
    out = {}
    for scope in SCOPES:
        sers = [d[scope] for d in per_seed if not d[scope].empty]
        if not sers:
            out[scope] = pd.Series(dtype=float)
        else:
            out[scope] = pd.concat(sers, axis=1).mean(axis=1)
    return out


def summarize(name: str, vals: dict[str, pd.Series], rows: list) -> None:
    for scope, s in vals.items():
        if s.empty:
            rows.append({"arm": name, "scope": scope,
                         "macro_mean": None, "n_patients": 0})
        else:
            rows.append({"arm": name, "scope": scope,
                         "macro_mean": float(s.mean()),
                         "n_patients": int(s.size)})


def paired_bootstrap(a: pd.Series, b: pd.Series) -> dict | None:
    """患者级配对差 bootstrap（与既有报告同口径：2000 次、百分位 95%）。"""
    idx = a.index.intersection(b.index)
    diff = (a.loc[idx] - b.loc[idx]).dropna()
    if len(diff) < 2:
        return None
    point = float(diff.mean())
    rng = np.random.default_rng(BOOT_SEED)
    arr = diff.to_numpy()
    draws = rng.choice(arr, size=(BOOT_RESAMPLES, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"point": point, "low": float(lo), "high": float(hi),
            "n": int(len(diff))}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    structural, nonstructural = load_structural()
    known = structural | nonstructural
    rows: list[dict] = []
    patient_vals: dict[str, dict[str, pd.Series]] = {}

    # ---------- 1. EXP-021 官方协议（三真种子） ----------
    official_dir = find_official_dir()
    official_seed_vals: dict[str, list[dict[str, pd.Series]]] = {}
    observed_ref = None
    for seed, run in OFFICIAL_RUNS.items():
        df = pd.read_parquet(official_dir / run / "outer_oof_predictions.parquet")
        df["algorithm"] = df["algorithm"].str.replace(r"_seed_\d+", "", regex=True)
        if observed_ref is None:
            b0 = df[df["algorithm"] == "B0"]
            observed_ref = b0[["patient_id", "model_id", "drug_id",
                               "observed", "source_drug_status"]].drop_duplicates(
                subset=["model_id", "drug_id"])
        for algo in sorted(df["algorithm"].unique()):
            sub = df[df["algorithm"] == algo]
            vals = arm_patient_values(sub, known, nonstructural)
            official_seed_vals.setdefault(algo, []).append(vals)
            summarize(f"official/{algo}/seed{seed}", vals, rows)
    for algo, seeds in official_seed_vals.items():
        avg = mean_across_seeds(seeds)
        patient_vals[f"official/{algo}"] = avg
        summarize(f"official/{algo}", avg, rows)

    # ---------- 2. EXP-028 冻结协议：直接路线（A2 / B0） ----------
    ra = pd.read_parquet(ROUTEA)
    ra = ra[(ra["role"] == "outer_test")]
    for algo in sorted(ra["algorithm"].unique()):
        sub = ra[ra["algorithm"] == algo]
        per_seed = []
        for seed in sorted(sub["seed"].unique()):
            vals = arm_patient_values(sub[sub["seed"] == seed], known, nonstructural)
            per_seed.append(vals)
        avg = mean_across_seeds(per_seed)
        patient_vals[f"frozen/{algo}"] = avg
        summarize(f"frozen/{algo}", avg, rows)

    # ---------- 3. EXP-028 融合五臂 ----------
    fu = pd.read_parquet(FUSION)
    for algo in sorted(fu["algorithm"].unique()):
        sub = fu[fu["algorithm"] == algo]
        per_seed = []
        for ls in sorted(sub["layer_seed"].unique()):
            vals = arm_patient_values(sub[sub["layer_seed"] == ls], known, nonstructural)
            per_seed.append(vals)
        avg = mean_across_seeds(per_seed)
        patient_vals[f"fusion/{algo}"] = avg
        summarize(f"fusion/{algo}", avg, rows)

    # ---------- 4. 间接臂（B2=逆转评分，B3=通路逆转；3 种子平均） ----------
    rb = pd.read_parquet(ROUTEB)
    rb_mean = rb.groupby(["sample_id", "drug_id"], as_index=False)[
        ["reversal_score", "pathway_reversal"]].mean()
    obs = observed_ref.rename(columns={"model_id": "sample_id"})
    rb_j = rb_mean.merge(obs[["sample_id", "drug_id", "observed",
                              "source_drug_status"]],
                         on=["sample_id", "drug_id"], how="inner")
    rb_j = rb_j[rb_j["source_drug_status"] == "known"]
    # sample_id -> patient_id（如 Pt31_t1 -> Pt31）
    rb_j["patient_id"] = rb_j["sample_id"].str.replace(r"_t\d+$", "", regex=True)
    for arm, col in [("B2_chempert_reversal", "reversal_score"),
                     ("B3_chempert_pathway", "pathway_reversal")]:
        sub = rb_j[["patient_id", "sample_id", "drug_id", "observed", col]].rename(
            columns={col: "predicted", "sample_id": "model_id"})
        sub["source_drug_status"] = "known"
        vals = arm_patient_values(sub, known, nonstructural)
        patient_vals[f"indirect/{arm}"] = vals
        summarize(f"indirect/{arm}", vals, rows)

    # ---------- 5. 关键配对比较（患者级 2000 次 bootstrap） ----------
    comparisons = [
        ("official/M1_source_scaled", "official/B0"),
        ("official/M3", "official/B0"),
        ("official/M1", "official/B0"),
        ("official/M2", "official/B0"),
        ("frozen/M1_source_scaled", "frozen/B0"),
        ("fusion/F1_GBM_full", "frozen/M1_source_scaled"),
        ("fusion/F2_MLP_full", "frozen/M1_source_scaled"),
        ("fusion/F0_MLP_predonly", "frozen/M1_source_scaled"),
        ("fusion/F1_GBM_full_xfit", "frozen/M1_source_scaled"),
        ("indirect/B2_chempert_reversal", "frozen/B0"),
        ("indirect/B2_chempert_reversal", "frozen/M1_source_scaled"),
        ("indirect/B3_chempert_pathway", "frozen/B0"),
    ]
    boot_rows = []
    for a, b in comparisons:
        va, vb = patient_vals.get(a), patient_vals.get(b)
        if va is None or vb is None:
            continue
        for scope in SCOPES:
            res = paired_bootstrap(va[scope], vb[scope])
            if res is not None:
                boot_rows.append({"comparison": f"{a} - {b}", "scope": scope, **res})

    # ---------- 输出 ----------
    res_df = pd.DataFrame(rows)
    boot_df = pd.DataFrame(boot_rows)
    res_df.to_json(OUT_DIR / "supplement_results.json", orient="records",
                   indent=1, force_ascii=False)
    boot_df.to_json(OUT_DIR / "supplement_bootstrap.json", orient="records",
                    indent=1, force_ascii=False)

    lines = ["# 补充排序指标（事后描述性分析，2026-09-20）", "",
             "口径：患者内全药列表 Spearman（已知 14 药，资格>=10）与",
             "患者内非结构药子集 Spearman（7 药，资格>=5）；培养物→患者内等权→患者宏平均。", "",
             "## 各臂宏平均", ""]
    lines.append(res_df.to_markdown(index=False, floatfmt=".4f"))
    lines += ["", "## 关键配对比较（患者级 2000 次 bootstrap）", ""]
    lines.append(boot_df.to_markdown(index=False, floatfmt=".4f"))
    (OUT_DIR / "supplement_tables.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
