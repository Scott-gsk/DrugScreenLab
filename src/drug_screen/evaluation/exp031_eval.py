"""EXP-031 / 工作包 B 评估与统计核心.

实现工作包 B（数据合同、工程修复与统一评估）中评估侧可复用的原语：

- 主指标：患者内全药物排序 Spearman 患者等权宏平均
  ``intra_patient_spearman_patient_macro``（B4）。
- A4 三类均值基线：目标药物均值 / 来源药物均值 / 目标基因依赖均值
  ``target_drug_mean_baseline``、``source_drug_mean_baseline``、
  ``gene_dependency_mean_baseline``。
- 统一预测 schema 校验器 ``validate_prediction_frame``（B3）。
- 比较前检查器 ``assert_comparable``（B3）。
- 配对 bootstrap + Holm 校正 ``paired_bootstrap_ci``、``holm_correct``（B4）。
- 常数预测守卫 ``constant_prediction_guard``（B5 #6）。

W3-07~W3-11 审计修复新增（2026-09-22，docs/reports/audit_repair_wave3_
20260922.md）：

- W3-07 种子统计口径：``aggregate_seed_patient_scores``（先逐种子计算
  患者分数 → 患者的种子分数平均 → 患者级配对统计；预测平均集成另列为
  独立模型）与 ``patient_score_paired_bootstrap``（患者分数级配对统计）。
- W3-08 折内同药跨患者相关性：``per_fold_drug_patient_correlation``
  （按折评价；常数记 NaN+原因；严禁跨折池化负基线）与
  ``per_fold_gain``（折内对齐增益，基线缺失折不跨折借用）。
- W3-10 多重比较 family：``ComparisonFamily`` 声明式结构（可序列化
  预注册）+ ``holm_correct_family``（Holm 在 family 内执行；跨种子
  平均 p 值拒绝）。
- 统一 canonical 入口：``run_formal_comparison``——正式实验比较固定
  走 比较检查器 → 患者等权主指标 → 患者级配对统计 链路。

研究性口径一律按工作包 B 合同实现，不自行改动判定规则（例如：资格不足
记 NaN 并列明原因、宏平均跳过 NaN 不填零、药物 / 标签不一致拒绝比较而
不取交集）。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from drug_screen.evaluation.metrics import masked_spearman

__all__ = [
    "intra_patient_spearman_patient_macro",
    "target_drug_mean_baseline",
    "source_drug_mean_baseline",
    "gene_dependency_mean_baseline",
    "validate_prediction_frame",
    "assert_comparable",
    "PairedBootstrapCI",
    "paired_bootstrap_ci",
    "holm_correct",
    "constant_prediction_guard",
    # W3-07 种子统计口径
    "SeedAggregation",
    "aggregate_seed_patient_scores",
    "patient_score_paired_bootstrap",
    # W3-08 折内同药跨患者相关性
    "per_fold_drug_patient_correlation",
    "per_fold_gain",
    # W3-10 多重比较 family
    "ComparisonSpec",
    "ComparisonFamily",
    "ComparisonPValue",
    "holm_correct_family",
    # 统一 canonical 入口
    "FormalComparisonResult",
    "run_formal_comparison",
]

REASON_INSUFFICIENT_DRUGS = "insufficient_drugs"
REASON_CONSTANT_OBSERVED = "constant_observed"
REASON_CONSTANT_PREDICTED = "constant_predicted"
REASON_CONSTANT_WITHIN_FOLD = "constant_within_fold"
REASON_CONSTANT_ACROSS_FOLDS = "constant_across_folds"
REASON_INSUFFICIENT_PATIENTS = "insufficient_patients"

_REQUIRED_SCHEMA_COLUMNS = (
    "experiment_id",
    "run_id",
    "contract_id",
    "split_id",
    "fold",
    "seed",
    "sample_id",
    "drug_id",
    "gene_id",
    "y_true",
    "y_pred",
    "model_id",
    "evaluation_role",
)
_IDENTITY_COLUMNS = ("patient_id", "model_id")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{what} must be a pandas DataFrame")
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{what} missing columns: {missing}")


def intra_patient_spearman_patient_macro(
    pred: pd.DataFrame,
    *,
    min_drugs: int = 3,
    algorithm: str | None = None,
) -> tuple[float | None, int, pd.DataFrame]:
    """患者内全药物排序 Spearman 的患者等权宏平均（B4 主指标）。

    对每个患者：先在患者内对同一药物的 ``observed`` / ``predicted`` 取均值
    （多 PDO 等权），再计算该患者的全药物排序 Spearman。共同实测药物数
    ``< min_drugs``、观测零方差、预测零方差时该患者记 NaN 并注明原因；宏
    平均跳过 NaN（不填零）。

    Parameters
    ----------
    pred:
        预测长表，需含 ``patient_id``、``drug_id``、``observed``、
        ``predicted``；可含多算法列 ``algorithm``。含多算法且未指定
        ``algorithm`` 时抛错。
    min_drugs:
        每患者共同实测药物数资格阈值（含该值，默认 3）。
    algorithm:
        单算法切片；``pred`` 含 ``algorithm`` 列且多于一个取值时为必需。

    Returns
    -------
    (macro_value, n_valid_patients, per_patient_frame)
    """
    _require_columns(pred, ("patient_id", "drug_id", "observed", "predicted"),
                     "prediction frame")
    df = pred
    if "algorithm" in df.columns:
        if algorithm is None:
            if df["algorithm"].nunique() > 1:
                raise ValueError(
                    "multiple algorithms in frame; pass algorithm= to select one")
        else:
            df = df[df["algorithm"] == algorithm]
            if df.empty:
                raise ValueError(f"no rows for algorithm={algorithm!r}")

    table = df[["patient_id", "drug_id", "observed", "predicted"]].copy()
    table["patient_id"] = table["patient_id"].astype(str).str.strip()
    table["drug_id"] = table["drug_id"].astype(str).str.strip()
    for col in ("observed", "predicted"):
        table[col] = pd.to_numeric(table[col], errors="raise").astype(float)

    per_drug = (
        table.groupby(["patient_id", "drug_id"], sort=True)
        .agg(observed=("observed", "mean"), predicted=("predicted", "mean"))
        .dropna(subset=["observed", "predicted"])
        .reset_index()
    )

    rows: list[dict[str, object]] = []
    valid_count = 0
    valid_values: list[float] = []
    for patient, group in per_drug.groupby("patient_id", sort=True):
        y = group["observed"].to_numpy(float)
        p = group["predicted"].to_numpy(float)
        n_drugs = int(len(y))
        if n_drugs < min_drugs:
            rows.append({
                "patient_id": patient,
                "n_drugs": n_drugs,
                "intra_patient_spearman": float("nan"),
                "reason": REASON_INSUFFICIENT_DRUGS,
            })
            continue
        if np.ptp(y) == 0:
            rows.append({
                "patient_id": patient,
                "n_drugs": n_drugs,
                "intra_patient_spearman": float("nan"),
                "reason": REASON_CONSTANT_OBSERVED,
            })
            continue
        if np.ptp(p) == 0:
            rows.append({
                "patient_id": patient,
                "n_drugs": n_drugs,
                "intra_patient_spearman": float("nan"),
                "reason": REASON_CONSTANT_PREDICTED,
            })
            continue
        rho = spearmanr(y, p).statistic
        rho = float(rho) if rho is not None and np.isfinite(rho) else float("nan")
        rows.append({
            "patient_id": patient,
            "n_drugs": n_drugs,
            "intra_patient_spearman": rho,
            "reason": None,
        })
        if np.isfinite(rho):
            valid_count += 1
            valid_values.append(rho)

    per_patient = pd.DataFrame(rows, columns=[
        "patient_id", "n_drugs", "intra_patient_spearman", "reason"])
    macro = float(np.mean(valid_values)) if valid_values else None
    return macro, valid_count, per_patient


def target_drug_mean_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    """目标类器官药物均值基线（A4）。

    仅用每个 fold 的训练折行（``evaluation_role == "train"``）计算：先在
    患者内对同药物取均值（多 PDO 等权），再对药物取患者等权平均，得到该
    fold 的逐药常数预测，应用到该 fold 的所有样本，并保留 / 写入
    ``evaluation_role`` 列标记 train/test。
    """
    _require_columns(
        frame, ("patient_id", "drug_id", "fold", "observed", "evaluation_role"),
        "target frame")
    df = frame.copy()
    df["drug_id"] = df["drug_id"].astype(str).str.strip()
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["observed"] = pd.to_numeric(df["observed"], errors="raise").astype(float)

    train = df[df["evaluation_role"] == "train"]
    if train.empty:
        raise ValueError("target_drug_mean_baseline requires training-fold rows")

    constants: dict[tuple[object, str], float] = {}
    for fold, sub in train.groupby("fold", sort=True):
        patient_mean = sub.groupby(["patient_id", "drug_id"], sort=True)["observed"].mean()
        drug_mean = patient_mean.groupby("drug_id", sort=True).mean()
        for drug, value in drug_mean.items():
            constants[(fold, drug)] = float(value)

    predicted = [constants.get((fold, drug), float("nan"))
                 for fold, drug in zip(df["fold"], df["drug_id"])]
    df["predicted"] = predicted
    return df


def source_drug_mean_baseline(
    source_frame: pd.DataFrame,
    *,
    source_study: str | None = None,
    study_col: str = "study_id",
    evaluation_role_col: str = "evaluation_role",
    train_role: str = "train",
) -> pd.DataFrame:
    """来源细胞系药物均值基线（A4）。

    仅用来源（训练）细胞计算逐药均值。PRISM 用途时须显式 ``source_study``
    过滤（如 ``"GDSC"``），均值来源记录在返回表的 ``source_study`` 列。

    Returns
    -------
    ``[drug_id, predicted, source_study, evaluation_role]`` 逐药常数预测表。
    """
    _require_columns(source_frame, ("drug_id", "observed"), "source frame")
    df = source_frame.copy()
    df["drug_id"] = df["drug_id"].astype(str).str.strip()
    df["observed"] = pd.to_numeric(df["observed"], errors="raise").astype(float)

    if evaluation_role_col in df.columns:
        df = df[df[evaluation_role_col] == train_role]

    if study_col in df.columns:
        studies = sorted(df[study_col].dropna().unique())
        if source_study is None:
            if len(studies) > 1:
                raise ValueError(
                    "multiple source studies present; pass source_study="
                    f" (e.g. 'GDSC') to select: {studies}")
            effective_study = studies[0] if studies else None
        else:
            df = df[df[study_col] == source_study]
            if df.empty:
                raise ValueError(f"no source rows for study={source_study!r}")
            effective_study = source_study
    else:
        if source_study is not None:
            raise ValueError(
                f"source_study={source_study!r} given but no {study_col!r} column")
        effective_study = None

    means = df.groupby("drug_id", sort=True)["observed"].mean()
    out = pd.DataFrame({
        "drug_id": means.index.astype(str),
        "predicted": means.to_numpy(float),
        "source_study": effective_study,
        "evaluation_role": "train",
    })
    return out


def gene_dependency_mean_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    """目标基因依赖均值基线（A4）。

    按 fold 的训练折求逐基因 ``y`` 均值，作为该 fold 的逐基因常数预测，
    应用到该 fold 的所有样本；保留 / 写入 ``evaluation_role`` 列。
    """
    _require_columns(frame, ("gene_id", "y", "fold", "evaluation_role"),
                     "gene dependency frame")
    df = frame.copy()
    df["gene_id"] = df["gene_id"].astype(str).str.strip()
    df["y"] = pd.to_numeric(df["y"], errors="raise").astype(float)

    train = df[df["evaluation_role"] == "train"]
    if train.empty:
        raise ValueError("gene_dependency_mean_baseline requires training-fold rows")

    constants: dict[tuple[object, str], float] = {}
    for fold, sub in train.groupby("fold", sort=True):
        gene_mean = sub.groupby("gene_id", sort=True)["y"].mean()
        for gene, value in gene_mean.items():
            constants[(fold, gene)] = float(value)

    predicted = [constants.get((fold, gene), float("nan"))
                 for fold, gene in zip(df["fold"], df["gene_id"])]
    df["predicted"] = predicted
    return df


def validate_prediction_frame(
    df: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame]:
    """校验统一预测长表（B3），返回 (errors, missing_prediction_rows)。

    校验项：必含列缺失、重复行（同 run/fold/sample/drug/gene）、NaN 预测
    记录（返回对应行，不静默丢弃）。``gene_id`` 可空但列必须存在；
    ``patient_id`` / ``model_id`` 至少其一为生物单元键。
    """
    errors: list[str] = []
    if not isinstance(df, pd.DataFrame):
        return [f"input must be a pandas DataFrame, got {type(df).__name__}"], pd.DataFrame()

    for col in _REQUIRED_SCHEMA_COLUMNS:
        if col not in df.columns:
            errors.append(f"missing required column: {col}")
    if not any(c in df.columns for c in _IDENTITY_COLUMNS):
        errors.append(
            f"missing biological identity column (one of {_IDENTITY_COLUMNS})")
    if errors:
        return errors, pd.DataFrame()

    duplicate_keys = ["run_id", "fold", "sample_id", "drug_id", "gene_id"]
    duplicate_keys = [c for c in duplicate_keys if c in df.columns]
    dup_mask = df.duplicated(subset=duplicate_keys, keep=False)
    if dup_mask.any():
        n_uniq = int(df[duplicate_keys].drop_duplicates().shape[0])
        errors.append(
            f"duplicate rows on {duplicate_keys}: {n_uniq} unique but "
            f"{int(dup_mask.sum())} rows involved in duplicates")

    nan_mask = df["y_pred"].isna()
    missing = pd.DataFrame()
    if nan_mask.any():
        missing = df[nan_mask].copy()
        errors.append(
            f"{int(nan_mask.sum())} rows have NaN y_pred (returned, not dropped)")

    return errors, missing


def _combos(frame: pd.DataFrame) -> frozenset[tuple[str, str]]:
    return frozenset(
        (str(fold), str(seed))
        for fold, seed in zip(frame["fold"], frame["seed"])
    )


def _value_set(series: pd.Series) -> set[str]:
    return {str(v) for v in series.dropna().unique()}


def _symmetric_diff(a: set[str], b: set[str]) -> tuple[list[str], list[str]]:
    return sorted(a - b), sorted(b - a)


def assert_comparable(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    allow_seed_mismatch: bool = False,
) -> None:
    """比较前检查（B3）：拒绝评价键 / 面板 / 标签 / 权重 / 种子不一致的比较。

    检查失败时抛 ``ValueError`` 并列出全部差异（不自动取交集）。
    """
    _require_columns(a, ("fold", "seed", "y_true", "sample_id"), "frame a")
    _require_columns(b, ("fold", "seed", "y_true", "sample_id"), "frame b")
    errors: list[str] = []

    for key in ("experiment_id", "contract_id", "split_id"):
        if key in a.columns and key in b.columns:
            if _value_set(a[key]) != _value_set(b[key]):
                errors.append(f"{key} differs between frames")

    def _item_cols(frame: pd.DataFrame) -> list[str]:
        return [c for c in ("drug_id", "gene_id")
                if c in frame.columns and frame[c].notna().any()]
    item_a = _item_cols(a)
    item_b = _item_cols(b)
    if not item_a or not item_b:
        errors.append("missing drug_id/gene_id item identity")
    if item_a != item_b:
        errors.append(f"item columns differ: {item_a} vs {item_b}")

    seeds_a = _value_set(a["seed"])
    seeds_b = _value_set(b["seed"])
    seed_mismatch = seeds_a != seeds_b
    if seed_mismatch and not allow_seed_mismatch:
        errors.append(
            f"seed sets differ (a={sorted(seeds_a)} vs b={sorted(seeds_b)}); "
            "deterministic single-seed vs multi-seed requires allow_seed_mismatch=True")
    if not seed_mismatch:
        combos_a = _combos(a)
        combos_b = _combos(b)
        missing_a, missing_b = _symmetric_diff(
            {f"{f}:{s}" for f, s in combos_a}, {f"{f}:{s}" for f, s in combos_b})
        if missing_a or missing_b:
            errors.append(
                f"fold×seed combos differ: only-in-a={missing_a}, only-in-b={missing_b}")

    for item in item_a:
        if item in a.columns and item in b.columns:
            only_a, only_b = _symmetric_diff(
                _value_set(a[item]), _value_set(b[item]))
            if only_a or only_b:
                errors.append(
                    f"{item} panel differs: only-in-a={only_a}, only-in-b={only_b}")

    only_a, only_b = _symmetric_diff(
        _value_set(a["sample_id"]), _value_set(b["sample_id"]))
    if only_a or only_b:
        errors.append(
            f"sample_id set differs: only-in-a={only_a}, only-in-b={only_b}")

    def _unit_col(frame: pd.DataFrame) -> str | None:
        for c in ("patient_id", "model_id"):
            if c in frame.columns and frame[c].notna().any():
                return c
        return None
    unit_a = _unit_col(a)
    unit_b = _unit_col(b)
    if unit_a is None or unit_b is None:
        errors.append("missing patient_id/model_id unit identity")
    elif unit_a != unit_b:
        errors.append(f"unit columns differ: {unit_a} vs {unit_b}")
    else:
        counts_a = a.groupby(unit_a).size()
        counts_b = b.groupby(unit_b).size()
        patients_a = sorted(counts_a.index)
        patients_b = sorted(counts_b.index)
        only_a, only_b = _symmetric_diff(set(patients_a), set(patients_b))
        if only_a or only_b:
            errors.append(
                f"patient list differs: only-in-a={only_a}, only-in-b={only_b}")
        else:
            ca = counts_a.reindex(patients_a).to_numpy(float)
            cb = counts_b.reindex(patients_b).to_numpy(float)
            ratio = ca / cb
            if not np.allclose(ratio, ratio[0], rtol=1e-9, atol=1e-12):
                errors.append(
                    "per-patient row-count ratios are not constant (patient weights differ)")

    if item_a and item_a == item_b:
        keys = ["fold", "sample_id"] + item_a
        if not (allow_seed_mismatch and seed_mismatch):
            keys = ["fold", "seed", "sample_id"] + item_a
        truth_a = a.drop_duplicates(subset=keys)[keys + ["y_true"]]
        truth_b = b.drop_duplicates(subset=keys)[keys + ["y_true"]]
        merged = truth_a.merge(truth_b, on=keys, suffixes=("_a", "_b"))
        if len(merged) != max(len(truth_a), len(truth_b)):
            errors.append(
                f"y_true rows do not align on keys {keys} "
                f"(matched {len(merged)} vs {max(len(truth_a), len(truth_b))})")
        else:
            mismatch = ~np.isclose(
                merged["y_true_a"].to_numpy(float),
                merged["y_true_b"].to_numpy(float),
                rtol=1e-9, atol=1e-12)
            if mismatch.any():
                errors.append(
                    f"y_true differs between frames in {int(mismatch.sum())} aligned rows")

    if errors:
        raise ValueError("frames are not comparable: " + " | ".join(errors))


@dataclass(frozen=True)
class PairedBootstrapCI:
    """以独立单位为重采样单元的配对差值 bootstrap 区间。"""

    point_estimate: float
    low: float
    high: float
    n_units: int
    n_resamples: int
    seed: int
    unit: str
    deltas: tuple[float, ...] = ()


def paired_bootstrap_ci(
    a_frame: pd.DataFrame,
    b_frame: pd.DataFrame,
    unit_col: str,
    metric_fn,
    *,
    n: int = 2000,
    seed: int = 0,
    truth_col: str = "y_true",
    pred_col: str = "y_pred",
) -> PairedBootstrapCI:
    """配对差值 bootstrap（B4）：以患者 / 细胞为单位重采样，配对差值 95% CI。

    每个单位分别用 ``metric_fn(truth, pred)`` 计算 a / b 得分，配对差值
    ``metric_a - metric_b`` 作为重采样观测；point estimate 为单位差值均值，
    区间为 ``n`` 次单位重采样的 2.5% / 97.5% 分位数。单位集合不一致或存在
    未定义（NaN）差值时抛错（配合 ``constant_prediction_guard`` 先过滤）。
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    for frame, name in ((a_frame, "a_frame"), (b_frame, "b_frame")):
        _require_columns(frame, (unit_col, truth_col, pred_col), name)
    units_a = sorted(a_frame[unit_col].unique())
    units_b = sorted(b_frame[unit_col].unique())
    if units_a != units_b:
        raise ValueError(
            "unit sets differ between frames; run assert_comparable first")

    deltas: list[float] = []
    undefined: list[str] = []
    for unit in units_a:
        ga = a_frame[a_frame[unit_col] == unit]
        gb = b_frame[b_frame[unit_col] == unit]
        ma = metric_fn(ga[truth_col].to_numpy(float), ga[pred_col].to_numpy(float))
        mb = metric_fn(gb[truth_col].to_numpy(float), gb[pred_col].to_numpy(float))
        delta = float(ma - mb)
        if not np.isfinite(delta):
            undefined.append(str(unit))
            continue
        deltas.append(delta)
    if undefined:
        raise ValueError(
            f"undefined paired delta for units (constant / ineligible): "
            f"{undefined}; filter via constant_prediction_guard first")

    deltas_arr = np.asarray(deltas, dtype=float)
    point = float(np.mean(deltas_arr))
    rng = np.random.default_rng(seed)
    boots = np.empty(n, dtype=float)
    m = deltas_arr.size
    for i in range(n):
        idx = rng.integers(0, m, size=m)
        boots[i] = float(np.mean(deltas_arr[idx]))
    low = float(np.quantile(boots, 0.025))
    high = float(np.quantile(boots, 0.975))
    return PairedBootstrapCI(
        point_estimate=point,
        low=low,
        high=high,
        n_units=m,
        n_resamples=n,
        seed=seed,
        unit=unit_col,
        deltas=tuple(deltas_arr.tolist()),
    )


def holm_correct(p_values: Iterable[float]) -> list[float]:
    """Holm-Bonferroni 逐步校正，返回校正后的 p 值（保持原始顺序）。"""
    p = np.asarray(list(p_values), dtype=float).reshape(-1)
    if p.size == 0:
        return []
    if not np.all(np.isfinite(p)):
        raise ValueError("p-values must be finite")
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * np.arange(m, 0, -1)
    adjusted = np.maximum.accumulate(adjusted)
    adjusted = np.minimum(adjusted, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adjusted
    return out.tolist()


def constant_prediction_guard(
    per_fold_frames,
    *,
    fold_col: str = "fold",
    drug_col: str = "drug_id",
    pred_col: str = "predicted",
) -> pd.DataFrame:
    """常数预测守卫（B5 #6）：标记 ``(fold, drug)`` 上预测无变化的组合。

    折内常数（同一 ``(fold, drug)`` 内预测零方差）标 ``constant_within_fold``；
    若某药物在所有折中均为常数且取值相同（跨折全局常数）标
    ``constant_across_folds``。返回新增 ``undefined_constant_reason`` 列的
    长表（非标记行为 None）。
    """
    if isinstance(per_fold_frames, pd.DataFrame):
        frame = per_fold_frames
    elif isinstance(per_fold_frames, Mapping):
        frame = pd.concat(list(per_fold_frames.values()), ignore_index=True)
    else:
        frame = pd.concat(list(per_fold_frames), ignore_index=True)
    _require_columns(frame, (fold_col, drug_col, pred_col), "prediction frame")

    df = frame.copy()
    df[drug_col] = df[drug_col].astype(str).str.strip()
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce").astype(float)

    key = [fold_col, drug_col]
    unique_per_group = df.groupby(key, sort=False)[pred_col].nunique(dropna=True)
    within_constant = unique_per_group <= 1

    reason = pd.Series([None] * len(df), index=df.index, dtype=object)
    for (fold, drug), is_const in within_constant.items():
        if not bool(is_const):
            continue
        rows_mask = (df[fold_col] == fold) & (df[drug_col] == drug)
        drug_folds = sorted(df.loc[df[drug_col] == drug, fold_col].unique())
        across = False
        if len(drug_folds) >= 2:
            values = []
            all_const = True
            for f in drug_folds:
                vals = df.loc[(df[drug_col] == drug) & (df[fold_col] == f),
                              pred_col].dropna().unique()
                if vals.size != 1:
                    all_const = False
                    break
                values.append(float(vals[0]))
            if all_const and len(set(values)) == 1:
                across = True
        reason[rows_mask] = (
            REASON_CONSTANT_ACROSS_FOLDS if across else REASON_CONSTANT_WITHIN_FOLD
        )

    df["undefined_constant_reason"] = reason
    return df


# ---------------------------------------------------------------------------
# W3-07 种子统计口径（2026-09-22 审计修复）
# ---------------------------------------------------------------------------


def _assert_single_seed(frame: pd.DataFrame, what: str) -> None:
    """多种子输入守卫：先经 ``aggregate_seed_patient_scores`` 明确口径。

    逐种子患者分数平均（主分析）与预测平均集成（独立模型）是两种
    不同口径，直接把多种子行混入会静默退化为跨种子平均预测。
    """
    if "seed" in frame.columns:
        n = int(frame["seed"].nunique(dropna=False))
        if n > 1:
            raise ValueError(
                f"{what} contains {n} seeds; aggregate via "
                "aggregate_seed_patient_scores first (逐种子患者分数平均与"
                "预测平均集成是两种独立口径，不得混用)")


@dataclass(frozen=True)
class SeedAggregation:
    """多种子三种统计口径（W3-07）。

    - ``per_seed_patient_scores``：逐种子患者分数
      （index=patient_id，columns=seed，未定义为 NaN）
    - ``seed_mean_patient_scores``：患者的种子分数平均（主分析口径，
      跳过 NaN 种子；``n_seeds_used`` 记录每患者实际使用的种子数）
    - ``ensemble_predictions`` / ``ensemble_patient_scores``：预测平均
      集成（逐 (patient, drug) 跨种子平均预测后重算）——独立模型口径，
      不得与逐种子平均混淆
    """

    per_seed_patient_scores: pd.DataFrame
    seed_mean_patient_scores: pd.Series
    n_seeds_used: pd.Series
    ensemble_predictions: pd.DataFrame
    ensemble_patient_scores: pd.Series


def aggregate_seed_patient_scores(
    preds_by_seed: Mapping[int, pd.DataFrame] | pd.DataFrame,
    *,
    seed_col: str = "seed",
    patient_col: str = "patient_id",
    drug_col: str = "drug_id",
    observed_col: str = "observed",
    predicted_col: str = "predicted",
    min_drugs: int = 3,
) -> SeedAggregation:
    """多种子统计口径聚合（W3-07）。

    主分析口径：先逐种子计算患者分数（患者内全药物排序 Spearman），
    再对患者的种子分数平均（``seed_mean_patient_scores``），最后做患者
    级配对统计（``patient_score_paired_bootstrap``）。预测平均集成
    （``ensemble_*``）另列为一个独立模型，不得与逐种子平均混淆。

    Parameters
    ----------
    preds_by_seed:
        ``{seed: 预测长表}`` 或含 ``seed_col`` 的单一长表（每行
        patient / drug / observed / predicted）。各 seed 的
        ``(patient, drug)`` 键集合与 observed 值必须一致（否则抛错，
        不取交集——先经 ``assert_comparable``）。
    min_drugs:
        每患者共同实测药物数资格阈值（透传主指标）。
    """
    required = (patient_col, drug_col, observed_col, predicted_col)
    if isinstance(preds_by_seed, pd.DataFrame):
        _require_columns(preds_by_seed, (seed_col, *required), "prediction frame")
        frames: dict[str, pd.DataFrame] = {
            str(s): sub for s, sub in preds_by_seed.groupby(seed_col, sort=True)}
    elif isinstance(preds_by_seed, Mapping):
        frames = {}
        for s, sub in preds_by_seed.items():
            _require_columns(sub, required, "prediction frame")
            frames[str(s)] = sub
    else:
        raise ValueError(
            "preds_by_seed must be a DataFrame or a mapping seed -> DataFrame")
    if not frames:
        raise ValueError("preds_by_seed must contain at least one seed")

    agg: dict[str, pd.DataFrame] = {}
    for seed, sub in frames.items():
        table = sub[list(required)].copy()
        table[patient_col] = table[patient_col].astype(str).str.strip()
        table[drug_col] = table[drug_col].astype(str).str.strip()
        for col in (observed_col, predicted_col):
            table[col] = pd.to_numeric(table[col], errors="raise").astype(float)
        agg[seed] = (
            table.groupby([patient_col, drug_col], sort=True)
            .agg(observed=(observed_col, "mean"),
                 predicted=(predicted_col, "mean"))
            .reset_index()
        )

    reference = next(iter(agg.values()))
    ref_obs = {
        (patient, drug): value
        for patient, drug, value in zip(
            reference[patient_col], reference[drug_col], reference["observed"])
    }
    for seed, table in agg.items():
        got = {
            (patient, drug): value
            for patient, drug, value in zip(
                table[patient_col], table[drug_col], table["observed"])
        }
        if set(got) != set(ref_obs):
            raise ValueError(
                f"seed {seed}: (patient, drug) key set differs across seeds; "
                "run assert_comparable first (拒绝取交集)")
        for key, value in ref_obs.items():
            if not np.isclose(got[key], value, rtol=1e-9, atol=1e-12):
                raise ValueError(
                    f"seed {seed}: observed differs at {key} "
                    f"({got[key]} vs {value}); labels must match across seeds")

    seeds = sorted(agg)
    per_seed: dict[str, pd.Series] = {}
    for seed in seeds:
        _, _, per_patient = intra_patient_spearman_patient_macro(
            agg[seed], min_drugs=min_drugs)
        per_seed[seed] = per_patient.set_index("patient_id")[
            "intra_patient_spearman"]
    all_patients = sorted({
        patient for series in per_seed.values() for patient in series.index})
    per_seed_frame = pd.DataFrame(
        {seed: per_seed[seed].reindex(all_patients) for seed in seeds},
        index=all_patients)
    seed_mean = per_seed_frame.mean(axis=1, skipna=True)
    n_used = per_seed_frame.notna().sum(axis=1)

    base = reference[[patient_col, drug_col, "observed"]].copy()
    key_index = pd.MultiIndex.from_frame(base[[patient_col, drug_col]])
    pred_matrix = np.column_stack([
        agg[seed].set_index([patient_col, drug_col])["predicted"]
        .reindex(key_index).to_numpy()
        for seed in seeds
    ])
    base["predicted"] = pred_matrix.mean(axis=1)
    ensemble_predictions = base.reset_index(drop=True)
    _, _, ens_per_patient = intra_patient_spearman_patient_macro(
        ensemble_predictions, min_drugs=min_drugs)
    ensemble_scores = ens_per_patient.set_index("patient_id")[
        "intra_patient_spearman"]

    return SeedAggregation(
        per_seed_patient_scores=per_seed_frame,
        seed_mean_patient_scores=seed_mean.rename("seed_mean_patient_score"),
        n_seeds_used=n_used.rename("n_seeds_used"),
        ensemble_predictions=ensemble_predictions,
        ensemble_patient_scores=ensemble_scores.rename("ensemble_patient_score"),
    )


def patient_score_paired_bootstrap(
    scores_a: pd.Series,
    scores_b: pd.Series,
    *,
    n: int = 2000,
    seed: int = 0,
) -> PairedBootstrapCI:
    """患者分数级配对 bootstrap（W3-07 主统计口径）。

    输入为"患者 → 分数"Series：逐种子平均后的患者分数（主分析）、
    预测平均集成的患者分数（独立模型）或单模型患者分数皆可——集成作为
    独立模型参与比较，不得与逐种子平均混淆。

    索引按患者对齐（集合不一致抛错）；差值存在 NaN 抛错（先经资格
    过滤与对齐检查）。
    """
    if not isinstance(scores_a, pd.Series) or not isinstance(scores_b, pd.Series):
        raise ValueError("scores must be pandas Series indexed by patient")
    if n < 2:
        raise ValueError("n must be >= 2")

    def _normalized(series: pd.Series) -> pd.Series:
        return pd.Series(
            series.to_numpy(dtype=float),
            index=[str(i) for i in series.index],
        ).sort_index()

    a = _normalized(scores_a)
    b = _normalized(scores_b)
    if not a.index.equals(b.index):
        raise ValueError(
            "patient sets differ between score series; run "
            "assert_comparable first")
    delta = a.to_numpy(float) - b.to_numpy(float)
    if not np.all(np.isfinite(delta)):
        raise ValueError(
            "undefined (NaN) paired patient scores; filter ineligible "
            "patients first")

    point = float(np.mean(delta))
    rng = np.random.default_rng(seed)
    m = delta.size
    boots = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, m, size=m)
        boots[i] = float(np.mean(delta[idx]))
    return PairedBootstrapCI(
        point_estimate=point,
        low=float(np.quantile(boots, 0.025)),
        high=float(np.quantile(boots, 0.975)),
        n_units=m,
        n_resamples=n,
        seed=seed,
        unit="patient_id",
        deltas=tuple(delta.tolist()),
    )


# ---------------------------------------------------------------------------
# W3-08 折内同药跨患者相关性（2026-09-22 审计修复）
# ---------------------------------------------------------------------------


def per_fold_drug_patient_correlation(
    pred: pd.DataFrame,
    *,
    fold_col: str = "fold",
    patient_col: str = "patient_id",
    drug_col: str = "drug_id",
    observed_col: str = "observed",
    predicted_col: str = "predicted",
    min_patients: int = 3,
) -> pd.DataFrame:
    """折内同药物跨患者相关性（W3-08）。

    按 ``(fold, drug)`` 分组：先患者内等权平均（多 PDO / 多行），
    再在折内计算跨患者 Spearman（``masked_spearman``：正确并列秩 +
    pairwise complete 缺失掩码）。

    - 有效患者数 < ``min_patients`` → NaN + ``insufficient_patients``
    - 折内观测零方差 → NaN + ``constant_observed``
    - 折内预测零方差 → NaN + ``constant_predicted``
    - 严禁跨折池化：跨折池化含折间伪变异，其负基线不得作为增益依据
      （增益计算见 ``per_fold_gain``，基线缺失折不跨折借用）。

    Returns
    -------
    ``[fold, drug_id, n_patients, spearman, reason]`` 长表。
    """
    _require_columns(
        pred, (fold_col, patient_col, drug_col, observed_col, predicted_col),
        "prediction frame")
    _assert_single_seed(pred, "prediction frame")
    table = pred[[fold_col, patient_col, drug_col, observed_col,
                  predicted_col]].copy()
    table[patient_col] = table[patient_col].astype(str).str.strip()
    table[drug_col] = table[drug_col].astype(str).str.strip()
    for col in (observed_col, predicted_col):
        table[col] = pd.to_numeric(table[col], errors="coerce").astype(float)

    per_patient = (
        table.groupby([fold_col, patient_col, drug_col], sort=True)
        .agg(observed=(observed_col, "mean"), predicted=(predicted_col, "mean"))
        .dropna(subset=["observed", "predicted"])
        .reset_index()
    )

    rows: list[dict[str, object]] = []
    for (fold, drug), group in per_patient.groupby([fold_col, drug_col],
                                                   sort=True):
        y = group["observed"].to_numpy(float)
        p = group["predicted"].to_numpy(float)
        n_patients = int(len(y))
        row: dict[str, object] = {
            "fold": fold, "drug_id": drug, "n_patients": n_patients,
            "spearman": float("nan"), "reason": None,
        }
        if n_patients < min_patients:
            row["reason"] = REASON_INSUFFICIENT_PATIENTS
            rows.append(row)
            continue
        if np.ptp(y) == 0:
            row["reason"] = REASON_CONSTANT_OBSERVED
            rows.append(row)
            continue
        if np.ptp(p) == 0:
            row["reason"] = REASON_CONSTANT_PREDICTED
            rows.append(row)
            continue
        rho, _ = masked_spearman(y, p)
        row["spearman"] = rho
        rows.append(row)
    return pd.DataFrame(
        rows, columns=["fold", "drug_id", "n_patients", "spearman", "reason"])


def per_fold_gain(
    model_per_fold: pd.DataFrame,
    baseline_per_fold: pd.DataFrame,
    *,
    value_col: str = "spearman",
    fold_col: str = "fold",
    drug_col: str = "drug_id",
) -> pd.DataFrame:
    """折内增益（W3-08）：按 ``(fold, drug)`` 对齐求差，禁止跨折借用基线。

    输入为 ``per_fold_drug_patient_correlation`` 输出（或含
    ``[fold, drug, value_col]`` 的等价表）。baseline 缺失该
    ``(fold, drug)`` 行或值为 NaN → 该格 gain 为 NaN（不跨折借用；
    跨折池化负基线不得作为增益依据）。

    Returns
    -------
    ``[fold, drug_id, model, baseline, gain, gain_defined, baseline_reason]``。
    """
    _require_columns(model_per_fold, (fold_col, drug_col, value_col),
                     "model frame")
    _require_columns(baseline_per_fold, (fold_col, drug_col, value_col),
                     "baseline frame")
    merged = model_per_fold[[fold_col, drug_col, value_col]].merge(
        baseline_per_fold[[fold_col, drug_col, value_col]],
        on=[fold_col, drug_col], how="outer",
        suffixes=("_model", "_baseline"))
    merged = merged.rename(columns={
        f"{value_col}_model": "model",
        f"{value_col}_baseline": "baseline",
    })
    if "reason" in baseline_per_fold.columns:
        reasons = baseline_per_fold[[fold_col, drug_col, "reason"]].rename(
            columns={"reason": "baseline_reason"})
        merged = merged.merge(reasons, on=[fold_col, drug_col], how="left")
    else:
        merged["baseline_reason"] = None
    merged["gain"] = merged["model"] - merged["baseline"]
    merged["gain_defined"] = merged["gain"].notna()
    return merged


# ---------------------------------------------------------------------------
# W3-10 预注册多重比较 family（2026-09-22 审计修复）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonSpec:
    """单个预注册比较（W3-10）。"""

    name: str
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("comparison name must be a non-empty string")


@dataclass(frozen=True)
class ComparisonFamily:
    """声明式多重比较 family（W3-10）：比较范围由预注册明确。

    family 定义可序列化（``to_dict`` / ``from_dict``）落盘登记；
    Holm 校正只在 family 内执行（``holm_correct_family``）。
    """

    name: str
    comparisons: tuple[ComparisonSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "comparisons", tuple(self.comparisons))
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("family name must be a non-empty string")
        if not self.comparisons:
            raise ValueError("family must contain at least one comparison")
        names = [c.name for c in self.comparisons]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate comparison names: {names}")

    @property
    def comparison_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.comparisons)

    def to_dict(self) -> dict[str, object]:
        """序列化（供预注册落盘登记）。"""
        return {
            "name": self.name,
            "comparisons": [
                {"name": c.name, "description": c.description}
                for c in self.comparisons
            ],
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, object]) -> "ComparisonFamily":
        if "name" not in obj or "comparisons" not in obj:
            raise ValueError("family dict requires 'name' and 'comparisons'")
        comps = obj["comparisons"]
        if isinstance(comps, (str, bytes)) or not isinstance(comps, Sequence):
            raise ValueError("'comparisons' must be a sequence of dicts")
        specs = []
        for item in comps:
            if not isinstance(item, Mapping) or "name" not in item:
                raise ValueError("each comparison requires a 'name'")
            specs.append(ComparisonSpec(
                name=str(item["name"]),
                description=str(item.get("description", "")),
            ))
        return cls(name=str(obj["name"]), comparisons=tuple(specs))


@dataclass(frozen=True)
class ComparisonPValue:
    """带来源纪律的比较 p 值（W3-10）。

    ``n_seeds_averaged > 1`` 表示跨种子平均 p 值——禁止充当汇总检验
    （``holm_correct_family`` 拒绝；多种子应先按
    ``aggregate_seed_patient_scores`` 聚合患者分数再检验）。
    """

    comparison: str
    p_value: float
    n_seeds_averaged: int = 1

    def __post_init__(self) -> None:
        if not np.isfinite(self.p_value) or not 0.0 <= self.p_value <= 1.0:
            raise ValueError("p_value must be finite within [0, 1]")
        if self.n_seeds_averaged < 1:
            raise ValueError("n_seeds_averaged must be >= 1")


def holm_correct_family(
    family: ComparisonFamily,
    p_values: Mapping[str, float | ComparisonPValue],
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """family 内 Holm 校正（W3-10）。

    - ``p_values`` 键必须与 family 预注册比较一一对应（多余 / 缺失抛错
      ——比较范围由预注册明确，不得事后增删）
    - ``ComparisonPValue.n_seeds_averaged > 1`` → 抛错（跨种子平均 p
      值不得充当汇总检验；多种子按 ``aggregate_seed_patient_scores``
      口径先聚合分数再检验）

    Returns
    -------
    ``[comparison, p_raw, p_holm, reject_holm]``（保持 family 注册顺序）。
    """
    given = dict(p_values)
    registered = list(family.comparison_names)
    missing = [c for c in registered if c not in given]
    extra = [c for c in given if c not in set(registered)]
    if missing or extra:
        raise ValueError(
            "p_values must match the pre-registered family exactly: "
            f"missing={missing}, extra={extra}")

    raw: dict[str, float] = {}
    for name in registered:
        value = given[name]
        if isinstance(value, ComparisonPValue):
            if value.comparison != name:
                raise ValueError(
                    f"ComparisonPValue comparison {value.comparison!r} "
                    f"does not match key {name!r}")
            if value.n_seeds_averaged > 1:
                raise ValueError(
                    f"comparison {name!r}: 跨种子平均 p 值不得充当汇总检验"
                    f"（n_seeds_averaged={value.n_seeds_averaged}）；"
                    "先按 aggregate_seed_patient_scores 聚合患者分数")
            raw[name] = float(value.p_value)
        else:
            raw[name] = float(value)

    adjusted = holm_correct([raw[name] for name in registered])
    rows = pd.DataFrame({
        "comparison": registered,
        "p_raw": [raw[name] for name in registered],
        "p_holm": adjusted,
    })
    rows["reject_holm"] = rows["p_holm"] < alpha
    return rows


# ---------------------------------------------------------------------------
# 统一 canonical 入口（W3-07~W3-11 接入点）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FormalComparisonResult:
    """正式比较 canonical 链路（``run_formal_comparison``）的结果。"""

    comparison: str
    point_estimate: float
    low: float
    high: float
    crosses_zero: bool
    n_patients: int
    model_patient_scores: pd.Series
    baseline_patient_scores: pd.Series


def run_formal_comparison(
    a_frame: pd.DataFrame,
    b_frame: pd.DataFrame,
    *,
    comparison: str = "model_minus_baseline",
    min_drugs: int = 3,
    n_bootstrap: int = 2000,
    seed: int = 0,
    allow_seed_mismatch: bool = False,
) -> FormalComparisonResult:
    """正式实验比较的统一 canonical 入口（W3-07~11）。

    链路顺序固定：

    1. ``assert_comparable``——比较检查器（面板 / 标签 / 键 / 种子一致）
    2. ``intra_patient_spearman_patient_macro``——患者等权主指标
    3. ``patient_score_paired_bootstrap``——患者级配对统计

    多种子输入必须先经 ``aggregate_seed_patient_scores`` 明确口径：
    逐种子患者分数平均（主分析）或预测平均集成（独立模型，另列），
    不得混用；多重比较经 ``holm_correct_family`` 在预注册 family 内
    执行。患者级配对在双侧均有定义的患者上成对（pairwise complete）。
    """
    _assert_single_seed(a_frame, "a_frame")
    _assert_single_seed(b_frame, "b_frame")
    assert_comparable(a_frame, b_frame, allow_seed_mismatch=allow_seed_mismatch)

    _, _, per_a = intra_patient_spearman_patient_macro(
        a_frame, min_drugs=min_drugs)
    _, _, per_b = intra_patient_spearman_patient_macro(
        b_frame, min_drugs=min_drugs)
    scores_a = per_a.set_index("patient_id")["intra_patient_spearman"]
    scores_b = per_b.set_index("patient_id")["intra_patient_spearman"]
    common = sorted(
        set(scores_a.index[scores_a.notna()])
        & set(scores_b.index[scores_b.notna()]))
    if not common:
        raise ValueError(
            "no patients with defined scores on both sides "
            "(constant / ineligible); comparison undefined")

    ci = patient_score_paired_bootstrap(
        scores_a.reindex(common), scores_b.reindex(common),
        n=n_bootstrap, seed=seed)
    return FormalComparisonResult(
        comparison=comparison,
        point_estimate=ci.point_estimate,
        low=ci.low,
        high=ci.high,
        crosses_zero=ci.low <= 0.0 <= ci.high,
        n_patients=ci.n_units,
        model_patient_scores=scores_a.reindex(common).rename("model"),
        baseline_patient_scores=scores_b.reindex(common).rename("baseline"),
    )