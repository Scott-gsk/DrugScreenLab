"""Permutation / substitute contrast hooks (个体信息贡献, 方案第三/八章).

Hooks only — callers supply the model ``predict_fn`` and expression matrix.
Does not run heavy training.

W3-09 / W3-11 修复新增（2026-09-22 审计）：

- ``permutation_p_value``：置换 p 值有限次数修正
  ``p = (1 + #{置换统计量 >= 观测}) / (n_perm + 1)``，禁止输出 p=0.0。
- ``permute_patient_identity_broadcast``：患者身份置换——身份映射后广播
  到该患者全部药物记录行（药物-患者对应保持，划内置换），替代会产生
  表达-药物错配的整行重排置换。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd

PredictFn = Callable[[np.ndarray], np.ndarray]
MetricFn = Callable[[np.ndarray, np.ndarray], float]


class ExpressionBatch(Protocol):
    """Minimal view of an expression matrix (n_samples, n_features)."""

    def __array__(self) -> np.ndarray: ...


@dataclass(frozen=True)
class PermutationContrastResult:
    """Contrast between original metric and disrupted-expression metric."""

    metric_name: str
    baseline_score: float
    disrupted_score: float
    delta: float  # baseline - disrupted; >0 means expression helped
    n_repeats: int
    disrupted_scores: tuple[float, ...]
    p_value: float | None = None  # 有限修正置换 p（经 permutation_p_value 计算后填充）


def permutation_p_value(
    observed_stat: float,
    perm_stats: Sequence[float],
    *,
    alternative: str = "greater",
) -> float:
    """置换检验 p 值（有限次数修正，W3-09）。

    ``p = (1 + #{置换统计量 >= 观测}) / (n_perm + 1)``

    - ``alternative="greater"``（默认）：置换统计量不小于观测计入极端；
      ``"less"`` 反向（计数 ``<=``）。
    - 永不输出 0.0：观测比全部置换更极端时 ``p = 1/(n_perm+1)``。
    - NaN 置换统计量不计入极端计数但计入分母（保守口径）；观测非有限
      → 返回 NaN（未定义，配合上游常数守卫先过滤）。
    """
    obs = float(observed_stat)
    if not np.isfinite(obs):
        return float("nan")
    stats = np.asarray(list(perm_stats), dtype=float).reshape(-1)
    n_perm = int(stats.size)
    if n_perm < 1:
        raise ValueError("perm_stats must contain at least one statistic")
    if alternative == "greater":
        count = int(np.sum(stats >= obs))
    elif alternative == "less":
        count = int(np.sum(stats <= obs))
    else:
        raise ValueError("alternative must be 'greater' or 'less'")
    return float((1 + count) / (n_perm + 1))


@dataclass(frozen=True)
class PatientIdentityPermutation:
    """一次患者身份置换（W3-11）。

    ``mapping``：记录患者 → 信息供体患者（允许划分内的双射）。
    ``frame``：``info_cols`` 给定时为广播后的数据帧副本（行结构与药物-
    患者对应不变，仅患者级信息列换成供体值），否则为 ``None``。
    """

    mapping: Mapping[str, str]
    frame: pd.DataFrame | None


def permute_patient_identity_broadcast(
    df: pd.DataFrame,
    n_perm: int,
    patient_col: str = "patient_id",
    drug_cols: Sequence[str] = ("drug_id",),
    *,
    info_cols: Sequence[str] | None = None,
    split_col: str | None = None,
    seed: int = 0,
) -> Iterator[PatientIdentityPermutation]:
    """患者身份置换：身份映射后广播到药物记录（W3-11）。

    每次置换：

    1. 在允许的划分内（``split_col`` 给定时逐划分独立置换）生成患者
       身份映射 π（划分内双射）；
    2. 把 π 广播到患者全部药物记录行——患者 i 的行获得患者 π(i) 的
       ``info_cols`` 值（患者级信息随身份整体换人）；
    3. 药物-患者对应保持：每患者的药物行集合与标签不变，仅信息来源
       换人——杜绝整行重排导致的表达-药物错配（EXP-030 parquet 同类风险）。

    Parameters
    ----------
    df:
        药物记录长表（每行 = 患者 × 药物），需含 ``patient_col`` 与
        ``drug_cols``；``(patient, drug)`` 组合不得重复。
    n_perm:
        置换次数（>= 1）。
    info_cols:
        随身份广播的患者级信息列（如表达特征），要求每患者内取值唯一
        （患者级信息），否则抛错。``None`` 时仅返回身份映射
        （``frame=None``），由调用方自行广播。
    split_col:
        划分列（如 ``fold`` / ``evaluation_role``）。给定时置换在该列
        每个取值内独立进行（train/test 患者身份不互换——不允许把置换
        样本送进训练侧统计）。

    Yields
    ------
    ``PatientIdentityPermutation``（每次置换一个）。
    """
    if n_perm < 1:
        raise ValueError("n_perm must be >= 1")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    missing = [c for c in (patient_col, *drug_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    work = df.copy()
    work[patient_col] = work[patient_col].astype(str).str.strip()
    for col in drug_cols:
        work[col] = work[col].astype(str).str.strip()
    if work.duplicated(subset=[patient_col, *drug_cols]).any():
        raise ValueError(
            f"duplicate ({patient_col}, {list(drug_cols)}) rows; "
            "patient-drug correspondence must be unique")

    if split_col is not None:
        if split_col not in work.columns:
            raise ValueError(f"split_col {split_col!r} not in frame")
        groups: dict[str, list[str]] = {
            str(split): sorted(sub[patient_col].unique())
            for split, sub in work.groupby(split_col, sort=True)
        }
    else:
        groups = {"__all__": sorted(work[patient_col].unique())}

    info_list: list[str] | None = None
    donor_info: pd.DataFrame | None = None
    if info_cols is not None:
        info_list = list(info_cols)
        missing_info = [c for c in info_list if c not in work.columns]
        if missing_info:
            raise ValueError(f"missing info columns: {missing_info}")
        nunique = work.groupby(patient_col)[info_list].nunique(dropna=False)
        bad = nunique[(nunique > 1).any(axis=1)]
        if len(bad):
            raise ValueError(
                "info columns must be constant within each patient "
                f"(patient-level information): {sorted(bad.index)}")
        donor_info = work.groupby(patient_col, sort=True)[info_list].first()

    rng = np.random.default_rng(seed)
    for _ in range(n_perm):
        mapping: dict[str, str] = {}
        for members in groups.values():
            perm = rng.permutation(len(members))
            for i, member in enumerate(members):
                mapping[member] = members[int(perm[i])]
        if donor_info is None or info_list is None:
            yield PatientIdentityPermutation(mapping=mapping, frame=None)
        else:
            out = work.copy()
            donor_index = out[patient_col].map(mapping)
            for col in info_list:
                out[col] = donor_info.loc[donor_index, col].to_numpy()
            yield PatientIdentityPermutation(mapping=mapping, frame=out)


def _as_matrix(x: ExpressionBatch | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError("expression matrix must be 2-D (n_samples, n_features)")
    return arr


@dataclass
class ExpressionPermutationContrast:
    """同癌种（或同队列）表达整行置换对照.

    Shuffle sample rows of the expression block so that drug / label
    alignment is preserved but individual expression identity is broken.
    """

    predict_fn: PredictFn
    metric_fn: MetricFn
    metric_name: str = "metric"
    n_repeats: int = 100
    random_state: int | None = 0

    def run(
        self,
        expression: ExpressionBatch | np.ndarray,
        y_true: Sequence[float],
        *,
        baseline_pred: Sequence[float] | None = None,
    ) -> PermutationContrastResult:
        expr = _as_matrix(expression)
        truth = np.asarray(y_true, dtype=float).reshape(-1)
        if expr.shape[0] != truth.size:
            raise ValueError("expression rows must match y_true length")

        if baseline_pred is None:
            baseline_pred_arr = np.asarray(self.predict_fn(expr), dtype=float).reshape(-1)
        else:
            baseline_pred_arr = np.asarray(baseline_pred, dtype=float).reshape(-1)
        if baseline_pred_arr.size != truth.size:
            raise ValueError("baseline_pred length mismatch")

        baseline = float(self.metric_fn(truth, baseline_pred_arr))
        rng = np.random.default_rng(self.random_state)
        disrupted: list[float] = []
        for _ in range(self.n_repeats):
            perm = rng.permutation(expr.shape[0])
            pred = np.asarray(self.predict_fn(expr[perm]), dtype=float).reshape(-1)
            disrupted.append(float(self.metric_fn(truth, pred)))

        mean_disrupted = float(np.mean(disrupted)) if disrupted else float("nan")
        return PermutationContrastResult(
            metric_name=self.metric_name,
            baseline_score=baseline,
            disrupted_score=mean_disrupted,
            delta=baseline - mean_disrupted,
            n_repeats=len(disrupted),
            disrupted_scores=tuple(disrupted),
        )


@dataclass
class MeanExpressionSubstituteContrast:
    """训练均值表达替代对照.

    Replace every sample's expression with the training-set mean vector
    (broadcast), then re-score. Large drop vs baseline supports
    individualized signal (方案第八章: 表达置换无影响 → 不足以支持个体化).
    """

    predict_fn: PredictFn
    metric_fn: MetricFn
    metric_name: str = "metric"
    train_expression: ExpressionBatch | np.ndarray | None = None

    def run(
        self,
        expression: ExpressionBatch | np.ndarray,
        y_true: Sequence[float],
        *,
        baseline_pred: Sequence[float] | None = None,
    ) -> PermutationContrastResult:
        expr = _as_matrix(expression)
        truth = np.asarray(y_true, dtype=float).reshape(-1)
        if expr.shape[0] != truth.size:
            raise ValueError("expression rows must match y_true length")

        if baseline_pred is None:
            baseline_pred_arr = np.asarray(self.predict_fn(expr), dtype=float).reshape(-1)
        else:
            baseline_pred_arr = np.asarray(baseline_pred, dtype=float).reshape(-1)

        baseline = float(self.metric_fn(truth, baseline_pred_arr))

        source = (
            _as_matrix(self.train_expression)
            if self.train_expression is not None
            else expr
        )
        mean_vec = source.mean(axis=0, keepdims=True)
        substituted = np.repeat(mean_vec, expr.shape[0], axis=0)
        pred = np.asarray(self.predict_fn(substituted), dtype=float).reshape(-1)
        disrupted = float(self.metric_fn(truth, pred))

        return PermutationContrastResult(
            metric_name=self.metric_name,
            baseline_score=baseline,
            disrupted_score=disrupted,
            delta=baseline - disrupted,
            n_repeats=1,
            disrupted_scores=(disrupted,),
        )
