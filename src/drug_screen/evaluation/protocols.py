"""Evaluation protocols: leakage checks, paired-resample CI, judgment rules.

Aligned with 方案第八章 §2 必须通过的检查 / §3 结果判断.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from drug_screen.evaluation.permutation import PermutationContrastResult


@dataclass(frozen=True)
class LeakageCheckReport:
    """划分无泄漏检查结果 (第八章 §2)."""

    ok: bool
    patient_overlap: frozenset[str] = frozenset()
    cell_overlap: frozenset[str] = frozenset()
    drug_overlap: frozenset[str] = frozenset()
    lincs_chempert_dup_cross_split: int = 0
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "patient_overlap": sorted(self.patient_overlap),
            "cell_overlap": sorted(self.cell_overlap),
            "drug_overlap": sorted(self.drug_overlap),
            "lincs_chempert_dup_cross_split": self.lincs_chempert_dup_cross_split,
            "notes": list(self.notes),
        }


def _overlap(a: Iterable[str], b: Iterable[str]) -> frozenset[str]:
    return frozenset(set(map(str, a)) & set(map(str, b)))


def check_split_leakage(
    *,
    train_patients: Iterable[str] = (),
    test_patients: Iterable[str] = (),
    train_cells: Iterable[str] = (),
    test_cells: Iterable[str] = (),
    train_drugs: Iterable[str] = (),
    test_drugs: Iterable[str] = (),
    allow_drug_overlap: bool = True,
    lincs_chempert_duplicate_ids: Iterable[str] = (),
    train_perturbation_ids: Iterable[str] = (),
    test_perturbation_ids: Iterable[str] = (),
    extra_notes: Sequence[str] = (),
) -> LeakageCheckReport:
    """Check identity overlaps between train and test partitions.

    Parameters
    ----------
    allow_drug_overlap:
        Main PDO task is *known drugs on unseen PDO*; drug overlap is
        expected. Set False for leave-drug-out / scaffold-out secondary tasks.
    lincs_chempert_duplicate_ids:
        IDs known to be duplicated across LINCS and ChemPert; if any appear
        in both train and test perturbation sets, count as leakage.
    """
    notes = list(extra_notes)
    patient_ol = _overlap(train_patients, test_patients)
    cell_ol = _overlap(train_cells, test_cells)
    drug_ol = _overlap(train_drugs, test_drugs)

    if patient_ol:
        notes.append(f"patient overlap: {sorted(patient_ol)}")
    if cell_ol:
        notes.append(f"cell overlap: {sorted(cell_ol)}")
    if drug_ol and not allow_drug_overlap:
        notes.append(f"drug overlap (disallowed for this split): {sorted(drug_ol)}")

    dup_set = set(map(str, lincs_chempert_duplicate_ids))
    train_p = set(map(str, train_perturbation_ids))
    test_p = set(map(str, test_perturbation_ids))
    cross = train_p & test_p & dup_set if dup_set else train_p & test_p
    # Only flag ChemPert/LINCS duplicates when the dup registry is provided;
    # otherwise still flag any identical perturbation record IDs across splits.
    n_cross = len(cross)
    if n_cross:
        notes.append(f"perturbation record overlap across splits: {n_cross}")

    ok = (
        len(patient_ol) == 0
        and len(cell_ol) == 0
        and (allow_drug_overlap or len(drug_ol) == 0)
        and n_cross == 0
    )
    return LeakageCheckReport(
        ok=ok,
        patient_overlap=patient_ol,
        cell_overlap=cell_ol,
        drug_overlap=drug_ol if not allow_drug_overlap else frozenset(),
        lincs_chempert_dup_cross_split=n_cross,
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class PairedResampleCI:
    """配对重采样置信区间 (第八章 §3: 2,000 次, 95%)."""

    estimate: float
    low: float
    high: float
    n_resamples: int
    alpha: float
    unit: str  # e.g. "patient" | "model"
    crosses_zero: bool
    deltas: tuple[float, ...] = ()


def paired_resample_ci(
    scores_model: Sequence[float],
    scores_baseline: Sequence[float],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    unit: str = "patient",
    random_state: int | None = 0,
    return_deltas: bool = False,
) -> PairedResampleCI:
    """Bootstrap the mean paired gain ``model - baseline`` over units.

    Each index is one patient (or model) unit; resampling is with replacement.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    a = np.asarray(scores_model, dtype=float).reshape(-1)
    b = np.asarray(scores_baseline, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("scores_model and scores_baseline must match")
    if a.size == 0:
        raise ValueError("empty scores")

    paired = a - b
    estimate = float(np.mean(paired))
    rng = np.random.default_rng(random_state)
    boots = np.empty(n_resamples, dtype=float)
    n = paired.size
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(paired[idx]))

    low = float(np.quantile(boots, alpha / 2.0))
    high = float(np.quantile(boots, 1.0 - alpha / 2.0))
    crosses = low <= 0.0 <= high
    return PairedResampleCI(
        estimate=estimate,
        low=low,
        high=high,
        n_resamples=n_resamples,
        alpha=alpha,
        unit=unit,
        crosses_zero=crosses,
        deltas=tuple(boots.tolist()) if return_deltas else (),
    )


@dataclass(frozen=True)
class ResultJudgment:
    """第八章 §3 结果判断口径结构化输出."""

    claim_ok: bool
    claim_level: str
    reasons: tuple[str, ...]
    ci: PairedResampleCI | None = None
    permutation: PermutationContrastResult | None = None
    extras: Mapping[str, object] = field(default_factory=dict)


def judge_model_gain(
    ci: PairedResampleCI,
    *,
    permutation: PermutationContrastResult | None = None,
    task_a_improved: bool | None = None,
    task_b_improved: bool | None = None,
    perturbation_improved: bool | None = None,
    ranking_improved: bool | None = None,
    learning_curve_suggests_sample_limit: bool | None = None,
) -> ResultJudgment:
    """Map numeric evidence to allowed claim language (第八章 §3).

    Rules (simplified encoding of the protocol draft):
    - CI crosses zero / very wide → 证据不足
    - ranking↑ but permutation delta≈0 → 总体候选排序, 不足个体化
    - Task A↑ but Task B not → 敏感性分层, 不宣称候选药物排序改善
    - 扰动预测↑ but 药敏排序 not → 转录预测能力, 不宣称药筛收益
    """
    reasons: list[str] = []

    if ci.crosses_zero:
        reasons.append("配对增益 95% 区间跨零")
        if learning_curve_suggests_sample_limit:
            reasons.append("学习曲线提示样本量限制")
        return ResultJudgment(
            claim_ok=False,
            claim_level="证据不足",
            reasons=tuple(reasons),
            ci=ci,
            permutation=permutation,
        )

    if ci.estimate <= 0:
        reasons.append("配对增益点估计≤0 且 95% 区间不跨零（弱于或等于对照）")
        return ResultJudgment(
            claim_ok=False,
            claim_level="弱于对照",
            reasons=tuple(reasons),
            ci=ci,
            permutation=permutation,
        )

    # width heuristic: half-width larger than |estimate|
    half_width = 0.5 * (ci.high - ci.low)
    if abs(ci.estimate) > 0 and half_width > abs(ci.estimate):
        reasons.append("区间较宽（半宽 > |点估计|）")
        if learning_curve_suggests_sample_limit:
            reasons.append("学习曲线提示样本量限制")
        return ResultJudgment(
            claim_ok=False,
            claim_level="证据不足",
            reasons=tuple(reasons),
            ci=ci,
            permutation=permutation,
        )

    if (
        ranking_improved is True
        and permutation is not None
        and abs(permutation.delta) <= 1e-12
    ):
        reasons.append("排名改善但表达置换/替代几乎无影响")
        return ResultJudgment(
            claim_ok=True,
            claim_level="总体候选排序（不足个体化）",
            reasons=tuple(reasons),
            ci=ci,
            permutation=permutation,
        )

    if task_a_improved is True and task_b_improved is False:
        reasons.append("逐药患者差异改善但患者内排序未改善")
        return ResultJudgment(
            claim_ok=True,
            claim_level="敏感性分层（不宣称候选药物排序改善）",
            reasons=tuple(reasons),
            ci=ci,
            permutation=permutation,
        )

    if perturbation_improved is True and ranking_improved is False:
        reasons.append("扰动预测改善但药敏排序未改善")
        return ResultJudgment(
            claim_ok=True,
            claim_level="转录预测能力（不宣称药筛收益）",
            reasons=tuple(reasons),
            ci=ci,
            permutation=permutation,
        )

    reasons.append("配对增益区间不跨零且未触发限制性条款")
    return ResultJudgment(
        claim_ok=True,
        claim_level="可报告模型增益（仍需结合学习曲线与对照）",
        reasons=tuple(reasons),
        ci=ci,
        permutation=permutation,
    )
