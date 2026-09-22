"""Three expression-input contracts for data-phase P0 (DRAFT).

Aligned with ``docs/研究实施方案.md`` §二.2:
1. gene expression
2. pathway activity
3. perturbation genes

Design rules (non-negotiable):
- ``data/raw`` is read-only.
- Preprocessing statistics (gene filters, scalers, pathway ranks) fit on the
  **training fold only**, then apply to val/test.
- Unmeasured drug responses must not be imputed as zeros.
- Different models may consume different contracts; do not force all assays
  onto the 978-gene L1000 panel.
- Level-3 deltas and Level-5 signatures must not be mixed silently.
- Do not treat prior EXP outcomes as model-selection evidence.

This module defines Protocols / dataclasses only. Concrete loaders belong in
sibling modules and should register versions via ``data/registry``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class ExpressionMatrix:
    """Sample × gene expression block under a named contract version.

    Attributes
    ----------
    sample_ids:
        Row identifiers (cell line / PDO / patient-model ids).
    gene_ids:
        Column identifiers (HGNC symbols or Ensembl IDs — document which).
    values:
        Dense or sparse numeric matrix; shape ``(n_samples, n_genes)``.
        Concrete loaders may use ``numpy.ndarray`` or ``scipy.sparse``;
        the contract only requires a 2-d numeric array-like.
    feature_space:
        e.g. ``"genome_wide"``, ``"l1000_978"``, ``"custom_panel"``.
    units:
        e.g. ``"log2_tpm"``, ``"rma"``, ``"level3_z"``.
    gene_order_contract_id:
        Frozen id for the gene/feature order (registry checksum of ordered ids).
    """

    sample_ids: Sequence[str]
    gene_ids: Sequence[str]
    values: object
    feature_space: str
    units: str
    gene_order_contract_id: str = "unset"
    metadata: Mapping[str, object] = field(default_factory=dict)
    contract_version: str = "draft-0.1"


@dataclass(frozen=True)
class PathwayMatrix:
    """Sample × pathway activity scores.

    Default pathway set for stage-1 modeling is Hallmark (ssGSEA / rank-based
    single-sample scores). Reactome is a sensitivity analysis, not the default.

    Any rank/scale fit must use ``fit_sample_ids`` (train only).
    """

    sample_ids: Sequence[str]
    pathway_ids: Sequence[str]
    values: object
    gene_set_collection: str  # e.g. "MSigDB_Hallmark_v2024.1"
    scoring_method: str  # e.g. "ssgsea_rank", "gsva", "mean_z"
    feature_order_contract_id: str = "unset"
    metadata: Mapping[str, object] = field(default_factory=dict)
    contract_version: str = "draft-0.1"


@dataclass(frozen=True)
class PerturbationGeneRecord:
    """One chemical (or genetic) perturbation gene signature instance.

    Level-3 expression deltas and Level-5 moderated signatures must not be
    mixed silently — keep ``signature_level`` explicit.
    """

    record_id: str
    compound_id: str
    cell_or_model_id: str
    dose: Optional[float]
    dose_unit: Optional[str]
    time_hr: Optional[float]
    signature_level: str  # "level3_delta" | "level5_signature" | ...
    control_id: Optional[str]
    gene_ids: Sequence[str]
    values: object  # gene-aligned vector or matrix row
    gene_order_contract_id: str = "unset"
    perturbation_class: str = "chemical"  # vs "genetic"
    source: str = ""  # "LINCS_GSE92742" | "ChemPert" | ...
    metadata: Mapping[str, object] = field(default_factory=dict)
    contract_version: str = "draft-0.1"


# Backward-compatible alias used by early DRAFT imports / tests.
PerturbationRecord = PerturbationGeneRecord


@runtime_checkable
class GeneExpressionContract(Protocol):
    """Contract: load baseline (or untreated) gene expression for modeling.

    Implementations must document:
    - ``version`` id (frozen in registry);
    - gene/feature order contract id;
    - identity join keys to cell / PDO / patient tables;
    - train-only fit steps for normalization / filtering.
    """

    name: str
    version: str

    def list_sources(self) -> Sequence[str]:
        """Registered source study ids (e.g. GDSC2, Kryeziu, Zhu)."""
        ...

    def load(
        self,
        *,
        source: str,
        sample_ids: Optional[Sequence[str]] = None,
        gene_ids: Optional[Sequence[str]] = None,
        feature_space: str = "genome_wide",
    ) -> ExpressionMatrix:
        """Load expression for ``source``; subset if ids provided."""
        ...

    def align_genes(
        self,
        matrices: Sequence[ExpressionMatrix],
        *,
        how: str = "intersection",
        fit_sample_ids: Optional[Sequence[str]] = None,
    ) -> Sequence[ExpressionMatrix]:
        """Align matrices to a shared gene axis.

        If any filter/scale is fit, use ``fit_sample_ids`` (train only).
        """
        ...


@runtime_checkable
class PathwayActivityContract(Protocol):
    """Contract: fixed gene-set → single-sample pathway activity.

    Stage-1 default: Hallmark collection + rank-based single-sample score.
    Pathway features used for model selection must be frozen in the registry
    (collection name + version + scoring method + feature order id).
    """

    name: str
    version: str

    def gene_set_collection(self) -> str:
        ...

    def scoring_method(self) -> str:
        ...

    def from_expression(
        self,
        expression: ExpressionMatrix,
        *,
        fit_sample_ids: Optional[Sequence[str]] = None,
    ) -> PathwayMatrix:
        """Score pathways; fit any rank/scale on ``fit_sample_ids`` only."""
        ...

    def load_cached(
        self,
        *,
        source: str,
        sample_ids: Optional[Sequence[str]] = None,
    ) -> PathwayMatrix:
        """Load a previously frozen pathway matrix from ``data/processed``."""
        ...


@runtime_checkable
class PerturbationGeneContract(Protocol):
    """Contract: chemical perturbation gene signatures (LINCS / ChemPert).

    Hard rules from 方案第六章:
    - separate chemical vs genetic perturbations;
    - do not conflate Level-3 deltas with Level-5 signatures;
    - group related replicates for splitting;
    - ChemPert∩LINCS duplicates must not cross train/test;
    - records without a reliable control are not delta-supervision labels.
    """

    name: str
    version: str

    def list_sources(self) -> Sequence[str]:
        ...

    def iter_records(
        self,
        *,
        source: str,
        signature_level: str,
        compound_ids: Optional[Sequence[str]] = None,
        cell_ids: Optional[Sequence[str]] = None,
    ) -> Iterable[PerturbationGeneRecord]:
        ...

    def leakage_groups(
        self, records: Sequence[PerturbationGeneRecord]
    ) -> Mapping[str, Sequence[str]]:
        """Return split-atomic group id → record ids (families, cross-source dups)."""
        ...


class StubGeneExpressionLoader:
    """Importable stub documenting the gene-expression entry point."""

    name = "gene_expression"
    version = "draft-0.1"

    def list_sources(self) -> Sequence[str]:
        return ("GDSC2", "CTRPv2", "PRISM", "Kryeziu", "Zhu", "LINCS_GSE92742")

    def load(self, *, source: str, sample_ids=None, gene_ids=None, feature_space: str = "genome_wide"):
        raise NotImplementedError(
            f"Stub only — implement train-only fit loaders for source={source!r}; raw read-only"
        )

    def align_genes(self, matrices, *, how: str = "intersection", fit_sample_ids=None):
        raise NotImplementedError("Stub only — align using train-fold gene stats if needed")


class StubPathwayActivityLoader:
    """Importable stub for Hallmark pathway activity (stage-1 default)."""

    name = "pathway_activity"
    version = "draft-0.1"

    def gene_set_collection(self) -> str:
        return "MSigDB_Hallmark_v2024.1"

    def scoring_method(self) -> str:
        return "ssgsea_rank"

    def from_expression(self, expression, *, fit_sample_ids=None):
        raise NotImplementedError(
            "Stub only — fit pathway ranks on train ids only "
            f"(fit_sample_ids={fit_sample_ids!r})"
        )

    def load_cached(self, *, source: str, sample_ids=None):
        raise NotImplementedError(f"Stub only — load frozen pathway matrix for {source!r}")


class StubPerturbationGeneLoader:
    """Importable stub for LINCS / ChemPert perturbation-gene contracts."""

    name = "perturbation_genes"
    version = "draft-0.1"

    def list_sources(self) -> Sequence[str]:
        return ("LINCS_GSE92742", "ChemPert")

    def iter_records(self, *, source: str, signature_level: str, compound_ids=None, cell_ids=None):
        raise NotImplementedError(
            f"Stub only — iterate {source!r} at signature_level={signature_level!r}"
        )

    def leakage_groups(self, records):
        raise NotImplementedError("Stub only — return compound-family / cross-source groups")


def registry_hint() -> Path:
    """Default relative path for dataset registry manifests (project-root relative)."""
    return Path("data/registry")
