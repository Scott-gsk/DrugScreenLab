"""Backward-compatible re-exports for expression contracts (DRAFT).

Prefer importing from ``drug_screen.data.expression_contracts`` and
``drug_screen.data.identity_contracts`` in new code.
"""

from __future__ import annotations

from drug_screen.data.expression_contracts import (
    ExpressionMatrix,
    GeneExpressionContract,
    PathwayActivityContract,
    PathwayMatrix,
    PerturbationGeneContract,
    PerturbationGeneRecord,
    PerturbationRecord,
    StubGeneExpressionLoader,
    StubPathwayActivityLoader,
    StubPerturbationGeneLoader,
    registry_hint,
)

__all__ = [
    "ExpressionMatrix",
    "GeneExpressionContract",
    "PathwayActivityContract",
    "PathwayMatrix",
    "PerturbationGeneContract",
    "PerturbationGeneRecord",
    "PerturbationRecord",
    "StubGeneExpressionLoader",
    "StubPathwayActivityLoader",
    "StubPerturbationGeneLoader",
    "registry_hint",
]
