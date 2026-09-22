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
]

REASON_INSUFFICIENT_DRUGS = "insufficient_drugs"
REASON_CONSTANT_OBSERVED = "constant_observed"
REASON_CONSTANT_PREDICTED = "constant_predicted"
REASON_CONSTANT_WITHIN_FOLD = "constant_within_fold"
REASON_CONSTANT_ACROSS_FOLDS = "constant_across_folds"

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