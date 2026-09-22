"""Leakage-safe B0 drug-mean and B1 per-drug linear residual adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class B0Model:
    drug_means: Mapping[str, float]
    global_mean: float


@dataclass(frozen=True)
class B1DrugModel:
    drug_id: str
    intercept: float
    weights: np.ndarray
    gene_indices: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    alpha: float


@dataclass(frozen=True)
class B1Model:
    models: Mapping[str, B1DrugModel]
    fallback: B0Model
    n_hvg: int
    alpha: float


def fit_b0_drug_means(
    drug_ids: Sequence[str],
    responses: Sequence[float],
) -> B0Model:
    grouped: dict[str, list[float]] = {}
    for drug_id, response in zip(drug_ids, responses, strict=True):
        if not math_isfinite(response):
            continue
        grouped.setdefault(drug_id, []).append(float(response))
    if not grouped:
        raise ValueError("B0 requires finite training responses")
    means = {drug: sum(values) / len(values) for drug, values in grouped.items()}
    all_values = [value for values in grouped.values() for value in values]
    return B0Model(drug_means=means, global_mean=sum(all_values) / len(all_values))


def predict_b0(model: B0Model, drug_ids: Sequence[str]) -> list[float]:
    return [model.drug_means.get(drug_id, model.global_mean) for drug_id in drug_ids]


def fit_b1_per_drug_ridge(
    expression: np.ndarray,
    sample_indices: Sequence[int],
    drug_ids: Sequence[str],
    responses: Sequence[float],
    *,
    n_hvg: int = 512,
    alpha: float = 10.0,
    min_train: int = 8,
) -> B1Model:
    fallback = fit_b0_drug_means(drug_ids, responses)
    matrix = np.asarray(expression, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("expression must be samples x genes")
    unique_train = np.unique(np.asarray(sample_indices, dtype=np.int64))
    filled = np.nan_to_num(matrix[unique_train], nan=0.0)
    variances = filled.var(axis=0)
    gene_count = min(n_hvg, int((variances > 0).sum()) or 1)
    shared_genes = np.argsort(variances)[-gene_count:]
    models: dict[str, B1DrugModel] = {}
    by_drug: dict[str, list[tuple[int, float]]] = {}
    for sample_index, drug_id, response in zip(sample_indices, drug_ids, responses, strict=True):
        if not math_isfinite(response):
            continue
        by_drug.setdefault(drug_id, []).append((int(sample_index), float(response)))
    for drug_id, rows in by_drug.items():
        if len(rows) < min_train:
            continue
        indices = np.array([item[0] for item in rows], dtype=np.int64)
        y = np.array([item[1] for item in rows], dtype=np.float64)
        block = np.nan_to_num(matrix[indices], nan=0.0)
        features = block[:, shared_genes]
        center = features.mean(axis=0)
        scale = features.std(axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        standardized = (features - center) / scale
        design = np.column_stack([np.ones(len(y)), standardized])
        gram = design.T @ design
        penalty = np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(gram + alpha * penalty, design.T @ y)
        models[drug_id] = B1DrugModel(
            drug_id=drug_id,
            intercept=float(coefficients[0]),
            weights=coefficients[1:].astype(np.float64),
            gene_indices=shared_genes,
            center=center,
            scale=scale,
            alpha=alpha,
        )
    return B1Model(models=models, fallback=fallback, n_hvg=n_hvg, alpha=alpha)


def predict_b1(
    model: B1Model,
    expression: np.ndarray,
    sample_indices: Sequence[int],
    drug_ids: Sequence[str],
) -> list[float]:
    matrix = np.asarray(expression, dtype=np.float64)
    predictions: list[float] = []
    fallback = predict_b0(model.fallback, drug_ids)
    for sample_index, drug_id, mean_pred in zip(sample_indices, drug_ids, fallback, strict=True):
        drug_model = model.models.get(drug_id)
        if drug_model is None:
            predictions.append(mean_pred)
            continue
        row = np.nan_to_num(matrix[int(sample_index)], nan=0.0)
        features = (row[drug_model.gene_indices] - drug_model.center) / drug_model.scale
        predictions.append(float(drug_model.intercept + features @ drug_model.weights))
    return predictions


def math_isfinite(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
