"""Train-fitted expression feature processing with frozen gene alignment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ExpressionTransformReport:
    n_input_genes: int
    n_aligned_genes: int
    n_missing_frozen_genes: int
    n_imputed_values: int
    observed_fraction: float


@dataclass(frozen=True)
class ExpressionPreprocessor:
    """Feature state derived solely from a training expression matrix."""

    gene_ids: tuple[str, ...]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    min_gene_coverage: float
    scaled: bool

    @classmethod
    def fit(
        cls,
        expression: np.ndarray,
        gene_ids: list[str] | tuple[str, ...] | np.ndarray,
        *,
        min_gene_coverage: float = 0.8,
        n_features: int | None = None,
        scale: bool = True,
    ) -> "ExpressionPreprocessor":
        values, ids = _validate_input(expression, gene_ids)
        if not 0 < min_gene_coverage <= 1:
            raise ValueError("min_gene_coverage must be in (0, 1]")
        if n_features is not None and n_features < 1:
            raise ValueError("n_features must be positive when supplied")
        coverage = np.isfinite(values).mean(axis=0)
        eligible = np.flatnonzero(coverage >= min_gene_coverage)
        if not len(eligible):
            raise ValueError("no genes meet the training coverage threshold")
        eligible_values = values[:, eligible]
        medians = np.nanmedian(eligible_values, axis=0)
        imputed = np.where(np.isfinite(eligible_values), eligible_values, medians)
        variances = np.var(imputed, axis=0)
        # mergesort keeps original gene order when variances tie.
        ranking = np.argsort(-variances, kind="mergesort")
        if n_features is not None:
            ranking = ranking[: min(n_features, len(ranking))]
        selected = eligible[ranking]
        selected_values = imputed[:, ranking]
        means = selected_values.mean(axis=0)
        scales = selected_values.std(axis=0)
        scales = np.where(scales > 0, scales, 1.0)
        return cls(
            gene_ids=tuple(ids[index] for index in selected),
            medians=np.asarray(medians[ranking], dtype=np.float64),
            means=np.asarray(means, dtype=np.float64),
            scales=np.asarray(scales, dtype=np.float64),
            min_gene_coverage=min_gene_coverage,
            scaled=scale,
        )

    def transform(
        self, expression: np.ndarray, gene_ids: list[str] | tuple[str, ...] | np.ndarray
    ) -> np.ndarray:
        values, _ = self.transform_with_report(expression, gene_ids)
        return values

    def transform_with_report(
        self, expression: np.ndarray, gene_ids: list[str] | tuple[str, ...] | np.ndarray
    ) -> tuple[np.ndarray, ExpressionTransformReport]:
        source, ids = _validate_input(expression, gene_ids)
        locations = {gene: index for index, gene in enumerate(ids)}
        aligned = np.full((source.shape[0], len(self.gene_ids)), np.nan, dtype=np.float64)
        found = np.array([gene in locations for gene in self.gene_ids], dtype=bool)
        for output_index, gene in enumerate(self.gene_ids):
            if gene in locations:
                aligned[:, output_index] = source[:, locations[gene]]
        missing = ~np.isfinite(aligned)
        imputed = np.where(missing, self.medians, aligned)
        transformed = (imputed - self.means) / self.scales if self.scaled else imputed
        report = ExpressionTransformReport(
            n_input_genes=len(ids), n_aligned_genes=int(found.sum()),
            n_missing_frozen_genes=int((~found).sum()), n_imputed_values=int(missing.sum()),
            observed_fraction=float(np.isfinite(aligned).mean()) if aligned.size else 0.0,
        )
        return transformed, report

    def to_dict(self) -> dict[str, Any]:
        return {
            "gene_ids": list(self.gene_ids), "medians": self.medians.tolist(),
            "means": self.means.tolist(), "scales": self.scales.tolist(),
            "min_gene_coverage": self.min_gene_coverage, "scaled": self.scaled,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "ExpressionPreprocessor":
        try:
            ids = _validate_gene_ids(state["gene_ids"])
            medians = np.asarray(state["medians"], dtype=np.float64)
            means = np.asarray(state["means"], dtype=np.float64)
            scales = np.asarray(state["scales"], dtype=np.float64)
            coverage = float(state["min_gene_coverage"])
            scaled = bool(state["scaled"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid expression preprocessor state") from error
        _validate_fitted_state(ids, medians, means, scales, coverage)
        return cls(ids, medians, means, scales, coverage, scaled)

    def save(self, path: str | Path) -> None:
        """Save compact, dependency-free fitted state in an NPZ archive."""
        state = self.to_dict()
        np.savez_compressed(Path(path), metadata=json.dumps({
            "gene_ids": state["gene_ids"], "min_gene_coverage": state["min_gene_coverage"],
            "scaled": state["scaled"],
        }), medians=self.medians, means=self.means, scales=self.scales)

    @classmethod
    def load(cls, path: str | Path) -> "ExpressionPreprocessor":
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            return cls.from_dict({
                "gene_ids": metadata["gene_ids"], "medians": archive["medians"],
                "means": archive["means"], "scales": archive["scales"],
                "min_gene_coverage": metadata["min_gene_coverage"], "scaled": metadata["scaled"],
            })


def _validate_input(expression: np.ndarray, gene_ids: object) -> tuple[np.ndarray, tuple[str, ...]]:
    values = np.asarray(expression, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("expression must be a non-empty two-dimensional array")
    if np.isinf(values).any():
        raise ValueError("expression cannot contain positive or negative infinity")
    ids = _validate_gene_ids(gene_ids)
    if len(ids) != values.shape[1]:
        raise ValueError("gene_ids must be non-empty and match expression columns")
    return values, ids


def _validate_gene_ids(gene_ids: object) -> tuple[str, ...]:
    try:
        raw_ids = tuple(gene_ids)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("gene_ids must be an iterable of strings") from error
    if not raw_ids or any(not isinstance(gene, str) or not gene.strip() for gene in raw_ids):
        raise ValueError("gene_ids must be non-empty strings")
    ids = tuple(gene.strip() for gene in raw_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("gene_ids must be unique")
    return ids


def _validate_fitted_state(
    ids: tuple[str, ...], medians: np.ndarray, means: np.ndarray, scales: np.ndarray, coverage: float
) -> None:
    arrays = (medians, means, scales)
    if any(array.ndim != 1 or len(array) != len(ids) for array in arrays):
        raise ValueError("fitted statistics must be one-dimensional and match gene_ids")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("fitted statistics must be finite")
    if not np.all(scales > 0):
        raise ValueError("fitted scales must be positive")
    if not 0 < coverage <= 1:
        raise ValueError("min_gene_coverage must be in (0, 1]")
