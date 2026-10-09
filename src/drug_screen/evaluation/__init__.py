"""Evaluation package for DrugScreenLab.

Combines legacy leakage-safe primitives (``protocol``) with the Task A/B
P0 skeleton aligned to chapters 3 and 8 of ``docs/研究实施方案.md``.
"""

from __future__ import annotations

from drug_screen.evaluation.exp031_eval import (
    ComparisonFamily,
    ComparisonPValue,
    ComparisonSpec,
    FormalComparisonResult,
    SeedAggregation,
    aggregate_seed_patient_scores,
    holm_correct_family,
    patient_score_paired_bootstrap,
    per_fold_drug_patient_correlation,
    per_fold_gain,
    run_formal_comparison,
)
from drug_screen.evaluation.metrics import (
    mae,
    masked_pearson,
    masked_spearman,
    ndcg_at_k,
    rmse,
    spearman_rho,
    top_k_overlap,
)
from drug_screen.evaluation.permutation import (
    ExpressionPermutationContrast,
    MeanExpressionSubstituteContrast,
    PatientIdentityPermutation,
    PermutationContrastResult,
    permutation_p_value,
    permute_patient_identity_broadcast,
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
from drug_screen.evaluation.reversal import reversal_score
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
    # 统一评估接口（EXP-031 / W3-07~11 审计修复）
    "aggregate_seed_patient_scores",
    "SeedAggregation",
    "patient_score_paired_bootstrap",
    "per_fold_drug_patient_correlation",
    "per_fold_gain",
    "ComparisonSpec",
    "ComparisonFamily",
    "ComparisonPValue",
    "holm_correct_family",
    "FormalComparisonResult",
    "run_formal_comparison",
    "reversal_score",
    "masked_spearman",
    "masked_pearson",
    "permutation_p_value",
    "PatientIdentityPermutation",
    "permute_patient_identity_broadcast",
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
