"""Strict source-to-common gene-axis slicing for the EXP-045 input adapter."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch


def _normalize_axis(values: Sequence[str], name: str) -> tuple[str, ...]:
    axis = tuple(str(value).strip() for value in values)
    if not axis or any(not gene for gene in axis):
        raise ValueError(f"{name} must be non-empty and contain no empty symbols")
    if len(axis) != len(set(axis)):
        raise ValueError(f"{name} contains duplicate symbols")
    return axis


def gene_axis_sha256(genes: Sequence[str]) -> str:
    axis = _normalize_axis(genes, "genes")
    return hashlib.sha256(("\n".join(axis) + "\n").encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Exp045GeneAxisSlice:
    """Verified column map from an ordered source axis to its common subset."""

    source_genes: tuple[str, ...]
    target_genes: tuple[str, ...]
    positions: tuple[int, ...]
    source_sha256: str
    target_sha256: str

    def __post_init__(self) -> None:
        _validate_frozen_axis(self.source_genes, "source_genes")
        _validate_frozen_axis(self.target_genes, "target_genes")

        if self.source_sha256 != gene_axis_sha256(self.source_genes):
            raise ValueError("source gene axis checksum does not match its frozen axis")
        if self.target_sha256 != gene_axis_sha256(self.target_genes):
            raise ValueError("target gene axis checksum does not match its frozen axis")

        if not isinstance(self.positions, tuple):
            raise ValueError("positions must be an immutable tuple")
        if len(self.positions) != len(self.target_genes):
            raise ValueError("positions length must equal the target gene axis length")
        if any(type(position) is not int for position in self.positions):
            raise ValueError("positions must contain strict integers")
        if any(position < 0 or position >= len(self.source_genes) for position in self.positions):
            raise ValueError("positions must be within the source gene axis")
        if any(left >= right for left, right in zip(self.positions, self.positions[1:])):
            raise ValueError("positions must be strictly increasing")
        if tuple(self.source_genes[position] for position in self.positions) != self.target_genes:
            raise ValueError("positions do not map the source axis to the target gene axis")


def _validate_frozen_axis(axis: object, name: str) -> None:
    """Require a canonical ordered tuple at the public immutable-plan boundary."""
    if not isinstance(axis, tuple):
        raise ValueError(f"{name} must be a tuple")
    if not axis or any(type(gene) is not str or not gene or gene != gene.strip() for gene in axis):
        raise ValueError(f"{name} must contain non-empty normalized gene symbols")
    if len(axis) != len(set(axis)):
        raise ValueError(f"{name} contains duplicate symbols")


def build_exp045_gene_axis_slice(
    source_genes: Sequence[str],
    target_genes: Sequence[str],
    *,
    expected_source_sha256: str,
    expected_target_sha256: str,
) -> Exp045GeneAxisSlice:
    """Build a strict order-preserving map; aliases and zero fill are forbidden."""
    source = _normalize_axis(source_genes, "source_genes")
    target = _normalize_axis(target_genes, "target_genes")
    source_sha = gene_axis_sha256(source)
    target_sha = gene_axis_sha256(target)
    if source_sha != expected_source_sha256:
        raise ValueError("source gene axis checksum does not match the frozen contract")
    if target_sha != expected_target_sha256:
        raise ValueError("target gene axis checksum does not match the frozen contract")
    positions_by_gene = {gene: index for index, gene in enumerate(source)}
    try:
        positions = tuple(positions_by_gene[gene] for gene in target)
    except KeyError as exc:
        raise ValueError("target gene axis contains a gene absent from source axis") from exc
    if positions != tuple(sorted(positions)):
        raise ValueError("target gene order must preserve the source checkpoint order")
    return Exp045GeneAxisSlice(
        source_genes=source,
        target_genes=target,
        positions=positions,
        source_sha256=source_sha,
        target_sha256=target_sha,
    )


def slice_exp045_gene_fc1_weight(weight: torch.Tensor, plan: Exp045GeneAxisSlice) -> torch.Tensor:
    """Return a new first-layer weight with exactly the frozen common input columns."""
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise TypeError("gene_fc1 weight must be a 2D torch tensor")
    if weight.shape[1] != len(plan.source_genes):
        raise ValueError("gene_fc1 input width differs from the frozen source axis")
    index = torch.tensor(plan.positions, dtype=torch.long, device=weight.device)
    return weight.index_select(1, index).clone()


def slice_exp045_transform_vectors(
    transform: Mapping[str, object], plan: Exp045GeneAxisSlice
) -> dict[str, np.ndarray]:
    """Select the same columns from frozen source imputation/scaling vectors."""
    required = ("mean", "scale", "min", "range")
    selected: dict[str, np.ndarray] = {}
    for key in required:
        if key not in transform:
            raise ValueError(f"source transform is missing {key}")
        vector = np.asarray(transform[key])
        if vector.ndim != 1 or vector.shape[0] != len(plan.source_genes):
            raise ValueError(f"source transform {key} does not match the frozen source axis")
        selected[key] = vector[np.asarray(plan.positions, dtype=np.int64)].copy()
    return selected
