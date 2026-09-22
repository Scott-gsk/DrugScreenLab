"""EXP-026 分层评价终点：非结构药子集 + 逐药跨患者个性化度量。

评价改革依据 ``experiments/records/EXP-026.md``（2026-09-19 三轮数据审计：
14 已知药中 7 个结构性极端、NDCG@5 随机基线 0.598、B0=0.9089 主要由
药物主效应贡献）。

三个并列终点（防止选择性汇报）：
1. ``nonstructural``  非结构药子集患者宏平均 NDCG@5（任务 B 应用口径，
   剔除送分结构药后的真实排序能力）；
2. ``per_drug``       逐药跨患者 Spearman 宏平均（主方案 §三.3 预登记的
   患者间差异度量；药物主效应被逐药自然消去，B0 的折常数预测与观测独立，
   零假设统计有效——直接度量个性化成分）；
3. ``full_panel``     全面板 NDCG@5 参照列，必须与第一波官方数字一致
   （同一 ndcg 实现，见 ``drug_screen.transfer.metrics_v2``）。

设计更正（先于任何结果产出，见 EXP-026 记录注记）：原设想的「观测减训练折
药均值 + 预测减训练折预测均值」的 PDO 内残差 Spearman 存在折间均值伪影
（B0 逐折常数 ≠ 该折训练均值，产生与观测残差系统负相关的伪信号），
改用主方案 §三.3 已预登记的逐药跨患者 Spearman，两者同为去药物主效应的
个性化度量。

患者多 PDO 时先患者内等权平均再跨患者 Spearman（患者等权口径）。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from drug_screen.transfer.metrics_v2 import ndcg_at_k

__all__ = [
    "StructuralDrugs",
    "load_structural_drugs",
    "nonstructural_patient_values",
    "full_panel_patient_values",
    "per_drug_patient_values",
    "evaluate_stratified",
    "StratifiedEndpoints",
]

_REQUIRED_COLUMNS = (
    "patient_id",
    "model_id",
    "drug_id",
    "fold",
    "algorithm",
    "observed",
    "predicted",
    "source_drug_status",
)


@dataclass(frozen=True)
class StructuralDrugs:
    """Kryzei 结构性极端药物 FROZEN 清单（EXP-026）。"""

    pan_effective: frozenset[str]
    pan_resistant: frozenset[str]
    nonstructural: frozenset[str]

    @property
    def structural(self) -> frozenset[str]:
        return self.pan_effective | self.pan_resistant

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "pan_effective": sorted(self.pan_effective),
            "pan_resistant": sorted(self.pan_resistant),
            "nonstructural": sorted(self.nonstructural),
        }


def load_structural_drugs(path: str | Path) -> StructuralDrugs:
    """加载 ``kryzei_structural_drugs_FROZEN.json``。"""
    obj = json.loads(Path(path).read_text())
    if obj.get("status") != "FROZEN":
        raise ValueError(f"structural drugs list not FROZEN: {path}")
    for key in ("pan_effective", "pan_resistant", "nonstructural"):
        if key not in obj:
            raise ValueError(f"missing key {key!r} in {path}")
    overlap = (set(obj["pan_effective"]) & set(obj["pan_resistant"])) | (
        set(obj["pan_effective"]) & set(obj["nonstructural"])) | (
        set(obj["pan_resistant"]) & set(obj["nonstructural"]))
    if overlap:
        raise ValueError(f"overlapping structural categories: {sorted(overlap)}")
    return StructuralDrugs(
        pan_effective=frozenset(obj["pan_effective"]),
        pan_resistant=frozenset(obj["pan_resistant"]),
        nonstructural=frozenset(obj["nonstructural"]),
    )


def _known_frame(df: pd.DataFrame, algorithm: str | None) -> pd.DataFrame:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    known = df[df["source_drug_status"] == "known"].copy()
    if known.empty:
        raise ValueError("no known-drug rows")
    if algorithm is None:
        if known["algorithm"].nunique() > 1:
            raise ValueError(
                "multiple algorithms in frame; pass algorithm= to select one")
    else:
        known = known[known["algorithm"] == algorithm]
        if known.empty:
            raise ValueError(f"no known rows for algorithm={algorithm!r}")
    return known


def _patient_macro(per_pdo: pd.Series) -> float | None:
    """PDO 值 → 患者等权（跳过 nan）→ 宏平均。

    输入为按 ``(patient_id, model_id)`` MultiIndex 的逐 PDO 值。
    """
    frame = per_pdo.rename("value").reset_index()
    patient_col = frame.columns[0]  # MultiIndex (patient_id, model_id) 第一层
    patients = frame.groupby(patient_col)["value"].mean()
    finite = patients.dropna()
    return float(finite.mean()) if len(finite) else None


def _pdo_values(known: pd.DataFrame, drug_set: frozenset[str] | None,
                name: str, *, k: int, min_panel: int) -> pd.Series:
    """每 PDO NDCG@k（子集筛选后药数不足 ``min_panel`` 或零方差 → NaN）。"""
    sub = known if drug_set is None else known[known["drug_id"].isin(drug_set)]
    out = {}
    for (patient, pdo), g in sub.groupby(["patient_id", "model_id"], sort=True):
        y = g["observed"].to_numpy(float)
        p = g["predicted"].to_numpy(float)
        value = ndcg_at_k(y, p, k=k) if len(y) >= min_panel else float("nan")
        if value is not None and not math.isfinite(value):
            value = float("nan")
        out[(patient, pdo)] = value
    return pd.Series(out, name=name, dtype=float)


def nonstructural_patient_values(
    known: pd.DataFrame, structural: StructuralDrugs, *, k: int = 5,
) -> pd.Series:
    """每 PDO 非结构药子集 NDCG@k（资格：非结构药数 ≥ k）。"""
    return _pdo_values(known, structural.nonstructural, "nonstructural_ndcg",
                       k=k, min_panel=k)


def full_panel_patient_values(
    known: pd.DataFrame, *, k: int = 5, min_panel: int = 10,
) -> pd.Series:
    """每 PDO 全面板 NDCG@k（官方资格：已知药数 ≥ 10，参照列）。

    与第一波官方口径一致（``metrics_v2`` 的 ``(5, len(y) >= 10)`` 规则），
    保证全面板参照列可与第一波官方数字精确对齐。
    """
    return _pdo_values(known, None, "full_panel_ndcg", k=k, min_panel=min_panel)


def per_drug_patient_values(
    known: pd.DataFrame, *, min_patients: int = 5,
) -> pd.Series:
    """逐药跨患者 Spearman（患者内先等权平均，任务 A 个性化度量）。

    对每个药物：患者平均观测 vs 患者平均预测的跨患者 Spearman。
    患者数不足 ``min_patients``、零方差或预测零方差 → NaN（不计入宏平均）。
    B0 的逐折常数预测与各折测试患者观测独立 → 零假设下期望 0 的有效零对照。
    """
    per_patient = (
        known.groupby(["drug_id", "patient_id"], sort=True)
        .agg(observed=("observed", "mean"), predicted=("predicted", "mean"))
        .reset_index()
    )
    out = {}
    for drug, g in per_patient.groupby("drug_id", sort=True):
        y = g["observed"].to_numpy(float)
        p = g["predicted"].to_numpy(float)
        if (len(y) < min_patients or np.ptp(y) == 0 or np.ptp(p) == 0):
            out[drug] = float("nan")
            continue
        rho = spearmanr(y, p).statistic
        out[drug] = float(rho) if math.isfinite(rho) else float("nan")
    return pd.Series(out, name="per_drug_patient_spearman", dtype=float)


@dataclass(frozen=True)
class StratifiedEndpoints:
    """单算法（单种子）三分层终点汇总。"""

    algorithm: str
    nonstructural_ndcg5_macro: float | None
    per_drug_patient_spearman_macro: float | None
    full_panel_ndcg5_macro: float | None
    n_patients_nonstructural: int
    n_pdos_nonstructural: int
    nonstructural_drugs_covered: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "nonstructural_ndcg5_macro": self.nonstructural_ndcg5_macro,
            "per_drug_patient_spearman_macro": self.per_drug_patient_spearman_macro,
            "full_panel_ndcg5_macro": self.full_panel_ndcg5_macro,
            "n_patients_nonstructural": self.n_patients_nonstructural,
            "n_pdos_nonstructural": self.n_pdos_nonstructural,
            "nonstructural_drugs_covered": list(self.nonstructural_drugs_covered),
        }


def evaluate_stratified(
    df: pd.DataFrame,
    structural: StructuralDrugs,
    *,
    algorithm: str | None = None,
    k: int = 5,
    min_patients_per_drug: int = 5,
) -> tuple[StratifiedEndpoints, pd.DataFrame]:
    """对单算法预测计算三个分层终点。

    Parameters
    ----------
    df:
        预测长表（需含 ``_REQUIRED_COLUMNS``）；含多算法/多种子行时必须用
        ``algorithm`` 参数选定单算法切片。

    Returns
    -------
    (summary, pdo_detail, drug_detail) ：汇总、逐 PDO 明细
        （``nonstructural_ndcg`` / ``full_panel_ndcg`` 两列）与逐药明细
        （``per_drug_patient_spearman``，按 drug_id 索引）。
    """
    known = _known_frame(df, algorithm)
    ns_values = nonstructural_patient_values(known, structural, k=k)
    fp_values = full_panel_patient_values(known, k=k)
    pd_values = per_drug_patient_values(known, min_patients=min_patients_per_drug)

    detail = pd.concat([ns_values, fp_values], axis=1).reset_index()
    detail = detail.rename(
        columns={"level_0": "patient_id", "level_1": "model_id"})
    drug_detail = pd.DataFrame({"drug_id": pd_values.index,
                                "per_drug_patient_spearman": pd_values.values})

    ns_rows = known[known["drug_id"].isin(structural.nonstructural)]
    summary = StratifiedEndpoints(
        algorithm=(algorithm if algorithm is not None
                   else str(known["algorithm"].iloc[0])),
        nonstructural_ndcg5_macro=_patient_macro(ns_values),
        per_drug_patient_spearman_macro=(
            float(pd_values.dropna().mean()) if pd_values.dropna().size else None),
        full_panel_ndcg5_macro=_patient_macro(fp_values),
        n_patients_nonstructural=int(ns_rows["patient_id"].nunique()),
        n_pdos_nonstructural=int(ns_rows["model_id"].nunique()),
        nonstructural_drugs_covered=tuple(sorted(ns_rows["drug_id"].unique())),
    )
    return summary, detail, drug_detail
