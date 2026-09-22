"""Evaluation package for DrugScreenLab.

Combines legacy leakage-safe primitives (``protocol``) with the Task A/B
P0 skeleton aligned to chapters 3 and 8 of ``docs/研究实施方案.md``.
"""

from __future__ import annotations

from drug_screen.evaluation.metrics import (
    mae,
    ndcg_at_k,
    rmse,
    spearman_rho,
    top_k_overlap,
)
from drug_screen.evaluation.permutation import (
    ExpressionPermutationContrast,
    MeanExpressionSubstituteContrast,
    PermutationContrastResult,
)
from drug_screen.evaluation.protocol import (
    CORE_GENE_COUNT,
    BootstrapInterval,
    EvaluationRecord,
    VectorMetrics,
    assert_cold_split_isolation,
    bootstrap_macro_mean,
    compute_vector_metrics,
)
from drug_screen.evaluation.protocols import (
    LeakageCheckReport,
    PairedResampleCI,
    ResultJudgment,
    check_split_leakage,
    judge_model_gain,
    paired_resample_ci,
)
from drug_screen.evaluation.stratified_metrics import (
    StratifiedEndpoints,
    StructuralDrugs,
    evaluate_stratified,
    full_panel_patient_values,
    load_structural_drugs,
    nonstructural_patient_values,
    per_drug_patient_values,
)
from drug_screen.evaluation.tasks import (
    TaskAResult,
    TaskASummary,
    TaskBResult,
    TaskBSummary,
    evaluate_task_a,
    evaluate_task_b,
    summarize_task_a,
    summarize_task_b,
)

__all__ = [
    # legacy (protocol.py)
    "CORE_GENE_COUNT",
    "BootstrapInterval",
    "EvaluationRecord",
    "VectorMetrics",
    "assert_cold_split_isolation",
    "bootstrap_macro_mean",
    "compute_vector_metrics",
    # metrics
    "ndcg_at_k",
    "spearman_rho",
    "top_k_overlap",
    "mae",
    "rmse",
    # tasks
    "TaskAResult",
    "TaskASummary",
    "TaskBResult",
    "TaskBSummary",
    "evaluate_task_a",
    "evaluate_task_b",
    "summarize_task_a",
    "summarize_task_b",
    # permutation
    "ExpressionPermutationContrast",
    "MeanExpressionSubstituteContrast",
    "PermutationContrastResult",
    # protocols (ch.8)
    "LeakageCheckReport",
    "PairedResampleCI",
    "ResultJudgment",
    "check_split_leakage",
    "paired_resample_ci",
    "judge_model_gain",
    # stratified endpoints (EXP-026)
    "StratifiedEndpoints",
    "StructuralDrugs",
    "evaluate_stratified",
    "full_panel_patient_values",
    "load_structural_drugs",
    "nonstructural_patient_values",
    "per_drug_patient_values",
]
