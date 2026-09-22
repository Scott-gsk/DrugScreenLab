"""Versioned GMT resources and sample-local pathway expression scores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def _gene(value: str) -> str:
    return value.strip().upper()


@dataclass(frozen=True)
class PathwayResource:
    resource_id: str
    version: str
    pathways: dict[str, tuple[str, ...]]

    @property
    def metadata(self) -> dict[str, object]:
        return {"resource_id": self.resource_id, "version": self.version, "n_pathways": len(self.pathways)}


@dataclass(frozen=True)
class PathwayScores:
    pathway_names: tuple[str, ...]
    values: np.ndarray
    coverage: np.ndarray
    resource_metadata: dict[str, object]


def parse_gmt(
    source: str | Path | Iterable[str], *, resource_id: str, version: str
) -> PathwayResource:
    """Parse a GMT file or lines, preserving resource provenance and gene sets."""

    if not resource_id or not version:
        raise ValueError("resource_id and version are required")
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and Path(source).exists()):
        lines = Path(source).read_text(encoding="utf-8").splitlines()
    elif isinstance(source, str):
        lines = source.splitlines()
    else:
        lines = list(source)
    pathways: dict[str, tuple[str, ...]] = {}
    for line_number, line in enumerate(lines, start=1):
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 3 or not fields[0].strip():
            raise ValueError(f"invalid GMT row at line {line_number}")
        name = fields[0].strip()
        genes = tuple(dict.fromkeys(gene for gene in (_gene(item) for item in fields[2:]) if gene))
        if not genes:
            raise ValueError(f"GMT pathway {name!r} has no genes")
        if name in pathways:
            raise ValueError(f"duplicate GMT pathway {name!r}")
        pathways[name] = genes
    if not pathways:
        raise ValueError("GMT contains no pathways")
    return PathwayResource(resource_id=resource_id, version=version, pathways=pathways)


def gene_to_pathways(resource: PathwayResource) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, list[str]] = {}
    for pathway, genes in resource.pathways.items():
        for gene in genes:
            mapping.setdefault(gene, []).append(pathway)
    return {gene: tuple(pathways) for gene, pathways in mapping.items()}


def require_distinct_resources(*resources: PathwayResource) -> None:
    """Reject accidental reuse where separate pathway resources are required."""
    identities = [(resource.resource_id, resource.version) for resource in resources]
    if len(set(identities)) != len(identities):
        raise ValueError("pathway resources must have distinct resource_id/version metadata")


def score_pathways(
    expression: np.ndarray,
    gene_ids: Iterable[str],
    resource: PathwayResource,
    *,
    min_coverage: float = 0.5,
) -> PathwayScores:
    """Mean each sample's observed pathway genes; no cross-sample fit occurs."""
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    values = np.asarray(expression, dtype=np.float64)
    ids = tuple(_gene(str(gene)) for gene in gene_ids)
    if values.ndim != 2 or values.shape[1] != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("expression columns must match unique gene_ids")
    index = {gene: position for position, gene in enumerate(ids)}
    names = tuple(resource.pathways)
    scores = np.full((values.shape[0], len(names)), np.nan, dtype=np.float64)
    coverage = np.zeros_like(scores)
    for column, name in enumerate(names):
        genes = resource.pathways[name]
        present = [index[gene] for gene in genes if gene in index]
        if not present:
            continue
        selected = values[:, present]
        observed = np.isfinite(selected)
        coverage[:, column] = observed.sum(axis=1) / len(genes)
        eligible = coverage[:, column] >= min_coverage
        if np.any(eligible):
            sums = np.where(observed, selected, 0.0).sum(axis=1)
            counts = observed.sum(axis=1)
            scores[eligible, column] = sums[eligible] / counts[eligible]
    return PathwayScores(names, scores, coverage, resource.metadata)


def score_progeny(expression=None, gene_ids=None, *, weights=None, version=None, min_coverage=0.5) -> PathwayScores:
    """Apply explicit signed gene/pathway coefficients to preprocessed expression.

    weights maps pathway -> {HGNC: coefficient}; never derives weights from test
    samples. This is a signed linear baseline, not a claim of reproducing all
    published PROGENy inference procedures.
    """
    if weights is None:
        raise NotImplementedError("signed PROGENy scoring requires an explicit, versioned weight resource")
    if not version or not isinstance(weights, dict) or not weights or not 0 < min_coverage <= 1:
        raise ValueError('version, nonempty pathway weights and valid coverage required')
    values = np.asarray(expression, dtype=float)
    ids = tuple(gene_ids)
    if values.ndim != 2 or values.shape[1] != len(ids) or len(set(ids)) != len(ids) or np.isinf(values).any():
        raise ValueError('invalid expression alignment')
    index = {gene: i for i, gene in enumerate(ids)}
    scores = np.full((len(values), len(weights)), np.nan)
    coverage = np.zeros_like(scores)
    for j, (name, members) in enumerate(weights.items()):
        if not members or not all(np.isfinite(float(w)) for w in members.values()):
            raise ValueError('invalid pathway coefficients')
        genes = [g for g in members if g in index]
        if not genes: continue
        block = values[:, [index[g] for g in genes]]
        observed = np.isfinite(block)
        coverage[:, j] = observed.sum(axis=1) / len(members)
        eligible = coverage[:, j] >= min_coverage
        scores[eligible, j] = np.where(observed, block, 0)[eligible] @ np.array([members[g] for g in genes])
    return PathwayScores(tuple(weights), scores, coverage, {'resource_id': 'PROGENy', 'version': version, 'method': 'signed_linear'})
