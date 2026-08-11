"""Leakage-safe evaluation primitives for DrugScreenLab experiments."""

from .protocol import (
    CORE_GENE_COUNT,
    BootstrapInterval,
    EvaluationRecord,
    VectorMetrics,
    assert_cold_split_isolation,
    bootstrap_macro_mean,
    compute_vector_metrics,
)

__all__ = [
    "CORE_GENE_COUNT",
    "BootstrapInterval",
    "EvaluationRecord",
    "VectorMetrics",
    "assert_cold_split_isolation",
    "bootstrap_macro_mean",
    "compute_vector_metrics",
]
