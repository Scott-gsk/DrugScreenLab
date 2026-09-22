"""Read-only loader for EXP-012 indexed PDO/cell-line bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from drug_screen.data.root import data_root


@dataclass(frozen=True)
class IndexedBundle:
    cohort_id: str
    role: str
    samples: pd.DataFrame
    drugs: pd.DataFrame
    pairs: pd.DataFrame
    expression: np.ndarray
    sample_ids: np.ndarray
    gene_ids: np.ndarray
    expression_unit: str
    higher_is_sensitive: bool
    response_name: str


def processed_exp012_root(root: Path | None = None) -> Path:
    return (root or data_root()) / "processed" / "exp012"


def load_cohort_index(root: Path | None = None) -> dict:
    path = processed_exp012_root(root) / "cohort_index_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_indexed_bundle(cohort_id: str, *, root: Path | None = None) -> IndexedBundle:
    index = load_cohort_index(root)
    match = next((row for row in index["cohorts"] if row["cohort_id"] == cohort_id), None)
    if match is None:
        raise KeyError(f"unknown cohort: {cohort_id}")
    bundle_dir = processed_exp012_root(root)
    samples = pd.read_parquet(bundle_dir / match["bundle"]["samples"])
    drugs = pd.read_parquet(bundle_dir / match["bundle"]["drugs"])
    pairs = pd.read_parquet(bundle_dir / match["bundle"]["pairs"])
    archive = np.load(bundle_dir / match["bundle"]["raw_npz"], allow_pickle=True)
    return IndexedBundle(
        cohort_id=cohort_id,
        role=match["role"],
        samples=samples,
        drugs=drugs,
        pairs=pairs,
        expression=np.asarray(archive["expression"]),
        sample_ids=np.asarray(archive["sample_id"]),
        gene_ids=np.asarray(archive["ordered_genes"]),
        expression_unit=match["expression_unit"],
        higher_is_sensitive=bool(match["higher_is_sensitive"]),
        response_name=match["response_name"],
    )


def sample_index_map(bundle: IndexedBundle) -> dict[str, int]:
    return {str(sample_id): index for index, sample_id in enumerate(bundle.sample_ids)}
