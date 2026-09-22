"""周报数据补充（2026-09-20，只读描述性分析）。

两个目的（均为事后描述性，不训练模型、不改变任何预登记判定）：
1. 主队列结构：患者/PDO/有效观测数的精确统计（附录 J.1 数据来源）；
2. 路线 A（直接预测）与路线 B（扰动逆转评分）预测结果的原始/残差相关性
   （附录 J.4 数据来源）。

输入：
- artifacts/exp028_fusion/routeA_frozen_predictions.parquet（冻结协议，51 患者）
- artifacts/exp028_fusion/routeB_features_exp028.parquet（零样本路线 B 特征）
- data/processed/identity/kryzei_structural_drugs_FROZEN.json

输出：artifacts/supplementary_ranking_metrics_20260920/
  cohort_route_correlation_supplement.json
"""
import json
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
A_PATH = ROOT / "artifacts/exp028_fusion/routeA_frozen_predictions.parquet"
B_PATH = ROOT / "artifacts/exp028_fusion/routeB_features_exp028.parquet"
SD_PATH = ROOT / "data/processed/identity/kryzei_structural_drugs_FROZEN.json"
OUT_PATH = (ROOT / "artifacts/supplementary_ranking_metrics_20260920"
            / "cohort_route_correlation_supplement.json")

OUT: dict = {}

# ---------- 1. 主队列结构（冻结协议口径） ----------
A = pd.read_parquet(A_PATH)
one = A[(A["source_drug_status"] == "known") & (A["role"] == "outer_test")
        & (A["algorithm"] == "M1_source_scaled")
        & (A["seed"] == A["seed"].unique()[0])]

pdo_per_patient = one.groupby("patient_id")["model_id"].nunique()
drugs_per_pdo = one.groupby("model_id")["drug_id"].nunique()
drugs_per_patient = one.groupby("patient_id")["drug_id"].nunique()
pairs = one.groupby(["patient_id", "drug_id"]).size().reset_index()

OUT["cohort_structure_frozen_protocol"] = {
    "note": ("EXP-028 冻结协议（准入白名单，不含 Pt120）；官方评价协议为 "
             "52 患者 / 91 PDO / 947 条已知观测（含 Pt120 的 1 枚 PDO）"),
    "n_patients": int(one["patient_id"].nunique()),
    "n_pdo_models": int(one["model_id"].nunique()),
    "n_known_rows": int(len(one)),
    "patient_drug_pairs": int(len(pairs)),
    "pdo_per_patient_hist": {str(k): int(v) for k, v in
                             pdo_per_patient.value_counts().sort_index().items()},
    "known_drugs_per_pdo_min_median_max": [int(drugs_per_pdo.min()),
                                           float(drugs_per_pdo.median()),
                                           int(drugs_per_pdo.max())],
    "known_drugs_per_patient_min_median_max": [int(drugs_per_patient.min()),
                                               float(drugs_per_patient.median()),
                                               int(drugs_per_patient.max())],
}

# ---------- 2. 路线 A vs 路线 B 预测相关性 ----------
a = A[(A["source_drug_status"] == "known") & (A["role"] == "outer_test")
      & (A["algorithm"] == "M1_source_scaled")].groupby(
    ["patient_id", "drug_id"]).agg(
    a_pred=("predicted", "mean"), obs=("observed", "mean")).reset_index()

B = pd.read_parquet(B_PATH)
B["patient_id"] = B["sample_id"].str.replace(r"_t\d+$", "", regex=True)
b = B.groupby(["patient_id", "drug_id"]).agg(
    b_rev=("reversal_score", "mean"),
    b_pwr=("pathway_reversal", "mean")).reset_index()

m = a.merge(b, on=["patient_id", "drug_id"], how="inner")
OUT["route_correlation"] = {"n_patient_drug_pairs": int(len(m))}


def _corr(x, y):
    pr, sr = pearsonr(x, y), spearmanr(x, y)
    return {"pearson": round(float(pr.statistic), 4),
            "spearman": round(float(sr.statistic), 4),
            "p_pearson": float(pr.pvalue)}


OUT["route_correlation"]["raw_A_vs_B_reversal"] = _corr(m["a_pred"], m["b_rev"])


def _residual_corr(df, col_a, col_b):
    """同药内去药物均值后的残差相关（只保留患者间差异成分）。"""
    dm = df[[col_a, col_b, "drug_id"]].copy()
    dm["ra"] = dm[col_a] - dm.groupby("drug_id")[col_a].transform("mean")
    dm["rb"] = dm[col_b] - dm.groupby("drug_id")[col_b].transform("mean")
    out = _corr(dm["ra"], dm["rb"])
    out["n"] = int(len(dm))
    return out


for ca, cb, name in [
        ("a_pred", "b_rev", "residual_A_vs_B_reversal"),
        ("a_pred", "b_pwr", "residual_A_vs_B_pathway_reversal"),
        ("obs", "b_rev", "residual_observed_vs_B_reversal"),
        ("obs", "a_pred", "residual_observed_vs_A_pred")]:
    OUT["route_correlation"][name] = _residual_corr(m, ca, cb)

nonstruct = set(json.loads(SD_PATH.read_text())["nonstructural"])
m_ns = m[m["drug_id"].isin(nonstruct)]
OUT["route_correlation"]["residual_A_vs_B_reversal_nonstructural"] = (
    _residual_corr(m_ns, "a_pred", "b_rev"))

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps(OUT, indent=2, ensure_ascii=False))
print(json.dumps(OUT, indent=2, ensure_ascii=False))
print(f"\nsaved -> {OUT_PATH}")
