"""Task A / Task B evaluation interfaces (方案第三章).

Task A — 同一种药在不同患者之间排序（逐药跨患者 Spearman）。
Task B — 同一个患者的不同药物之间排序（NDCG@5 主指标）。

Patient with multiple PDOs: aggregate within patient first, then
patient-equal-weight statistics (caller supplies already-aggregated
rows when required).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from drug_screen.evaluation.metrics import (
    mae,
    ndcg_at_k,
    rmse,
    spearman_rho,
    top_k_overlap,
)


@dataclass(frozen=True)
class TaskAResult:
    """Per-drug cross-patient ranking (任务 A)."""

    drug_id: str
    spearman: float
    n_patients: int
    mae: float | None = None
    rmse: float | None = None


@dataclass(frozen=True)
class TaskBResult:
    """Within-patient cross-drug ranking (任务 B)."""

    patient_id: str
    ndcg_at_5: float
    spearman: float
    top5_overlap: float
    n_drugs: int
    mae: float | None = None
    rmse: float | None = None


@dataclass(frozen=True)
class TaskASummary:
    per_drug: tuple[TaskAResult, ...]
    mean_spearman: float
    n_drugs_scored: int


@dataclass(frozen=True)
class TaskBSummary:
    per_patient: tuple[TaskBResult, ...]
    mean_ndcg_at_5: float
    mean_spearman: float
    mean_top5_overlap: float
    n_patients_scored: int
    extras: Mapping[str, float] = field(default_factory=dict)


def evaluate_task_a(
    *,
    drug_id: str,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    compute_error: bool = False,
) -> TaskAResult:
    """Score one drug across patients (aligned arrays, one row per patient)."""
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if truth.size != pred.size:
        raise ValueError("y_true and y_pred length mismatch")
    result_mae = mae(truth, pred) if compute_error else None
    result_rmse = rmse(truth, pred) if compute_error else None
    return TaskAResult(
        drug_id=str(drug_id),
        spearman=spearman_rho(truth, pred),
        n_patients=int(truth.size),
        mae=result_mae,
        rmse=result_rmse,
    )


def summarize_task_a(results: Sequence[TaskAResult]) -> TaskASummary:
    rhos = [r.spearman for r in results if np.isfinite(r.spearman)]
    mean_rho = float(np.mean(rhos)) if rhos else float("nan")
    return TaskASummary(
        per_drug=tuple(results),
        mean_spearman=mean_rho,
        n_drugs_scored=len(rhos),
    )


def evaluate_task_b(
    *,
    patient_id: str,
    y_true: Sequence[float],
    y_score: Sequence[float],
    drug_ids: Sequence[object] | None = None,
    k: int = 5,
    compute_error: bool = False,
) -> TaskBResult:
    """Score one patient's drug ranking on a comparable candidate set.

    Caller must restrict to the same study / comparable screen condition
    and drop untested drugs (方案第三章).
    """
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    score = np.asarray(y_score, dtype=float).reshape(-1)
    if truth.size != score.size:
        raise ValueError("y_true and y_score length mismatch")
    result_mae = mae(truth, score) if compute_error else None
    result_rmse = rmse(truth, score) if compute_error else None
    return TaskBResult(
        patient_id=str(patient_id),
        ndcg_at_5=ndcg_at_k(truth, score, k=k),
        spearman=spearman_rho(truth, score),
        top5_overlap=top_k_overlap(truth, score, k=k, item_ids=drug_ids),
        n_drugs=int(truth.size),
        mae=result_mae,
        rmse=result_rmse,
    )


def summarize_task_b(results: Sequence[TaskBResult]) -> TaskBSummary:
    if not results:
        return TaskBSummary(
            per_patient=(),
            mean_ndcg_at_5=float("nan"),
            mean_spearman=float("nan"),
            mean_top5_overlap=float("nan"),
            n_patients_scored=0,
        )
    ndcgs = [r.ndcg_at_5 for r in results]
    rhos = [r.spearman for r in results if np.isfinite(r.spearman)]
    overlaps = [r.top5_overlap for r in results]
    return TaskBSummary(
        per_patient=tuple(results),
        mean_ndcg_at_5=float(np.mean(ndcgs)),
        mean_spearman=float(np.mean(rhos)) if rhos else float("nan"),
        mean_top5_overlap=float(np.mean(overlaps)),
        n_patients_scored=len(results),
    )
