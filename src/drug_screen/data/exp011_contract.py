"""Immutable, index-based data contracts for EXP-011.

EXP-011 deliberately stores one expression vector per biological sample and
an index table for sample--drug responses.  It must never materialize a dense
``n_pairs x n_genes`` matrix: that would duplicate source expression roughly
two hundred thousand times and makes accidental split leakage harder to audit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


REQUIRED_MANIFEST_FIELDS = (
    "cohort_id", "role", "sample_type", "patient_id", "pdo_id", "drug_id",
    "canonical_smiles", "inchikey", "response_name", "higher_is_sensitive",
    "expression_unit", "counts", "hashes",
)

SAMPLE_COLUMNS = ("sample_id", "patient_id", "pdo_id", "cell_line_id", "sample_tissue")
DRUG_COLUMNS = ("drug_id", "canonical_smiles", "inchikey", "drug_scaffold")
PAIR_COLUMNS = (
    "cohort_id", "sample_id", "drug_id", "response", "response_name",
    "response_scale", "higher_is_sensitive",
)


def sha256_file(path: Path) -> str:
    """Return the byte-level digest used in each immutable source record."""
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gene_digest(ordered_genes: list[str] | np.ndarray) -> str:
    values = [str(gene).strip().upper() for gene in ordered_genes]
    return sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResponseContract:
    """Semantics of one response scale before any fold-specific normalization."""

    response_name: str
    response_scale: str
    higher_is_sensitive: bool
    transform: str

    def __post_init__(self) -> None:
        if not self.response_name or not self.response_scale:
            raise ValueError("response_name and response_scale are required")


@dataclass(frozen=True)
class CohortManifest:
    """Cohort-level provenance and biological-unit contract."""

    cohort_id: str
    role: str
    sample_type: str
    patient_id: str
    pdo_id: str
    drug_id: str
    canonical_smiles: str
    inchikey: str
    response_name: str
    higher_is_sensitive: bool
    expression_unit: str
    counts: Mapping[str, int]
    hashes: Mapping[str, str]
    response_scale: str
    notes: str = ""

    def __post_init__(self) -> None:
        missing = [field for field in REQUIRED_MANIFEST_FIELDS if getattr(self, field, None) in (None, "")]
        if missing:
            raise ValueError(f"cohort manifest missing required fields: {missing}")
        if not self.counts or not self.hashes:
            raise ValueError("cohort manifest must record counts and source hashes")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IndexedDrugResponseDataset:
    """One cohort's expressions, structures and response index.

    ``pairs`` stores a response once.  The corresponding sample and molecular
    features live in ``samples`` and ``drugs`` respectively and are joined by
    integer positions in the NPZ payload.  ``patient_id`` and ``pdo_id`` are
    retained even when empty so downstream split code has a stable schema.
    """

    def __init__(
        self,
        *,
        cohort_id: str,
        expression: np.ndarray,
        ordered_genes: list[str],
        samples: pd.DataFrame,
        drugs: pd.DataFrame,
        pairs: pd.DataFrame,
        expression_unit: str,
        source_hashes: Mapping[str, str],
    ) -> None:
        self.cohort_id = str(cohort_id)
        self.expression = np.asarray(expression, dtype=np.float32)
        self.ordered_genes = [str(gene).strip().upper() for gene in ordered_genes]
        self.samples = samples.copy()
        self.drugs = drugs.copy()
        self.pairs = pairs.copy()
        self.expression_unit = str(expression_unit)
        self.source_hashes = dict(source_hashes)
        self._validate()

    def _validate(self) -> None:
        if not self.cohort_id:
            raise ValueError("cohort_id is required")
        if self.expression.ndim != 2:
            raise ValueError("expression must have shape [n_samples, n_genes]")
        if self.expression.shape != (len(self.samples), len(self.ordered_genes)):
            raise ValueError(
                "expression shape does not match samples/ordered_genes: "
                f"{self.expression.shape} != {(len(self.samples), len(self.ordered_genes))}"
            )
        if len(self.ordered_genes) != 19_091:
            raise ValueError(f"EXP-011 requires exactly 19,091 ordered genes, got {len(self.ordered_genes)}")
        if self.ordered_genes != sorted(self.ordered_genes) or len(set(self.ordered_genes)) != len(self.ordered_genes):
            raise ValueError("ordered_genes must be unique, uppercase and lexicographically ordered")
        for column in SAMPLE_COLUMNS:
            if column not in self.samples.columns:
                raise ValueError(f"samples missing {column}")
        for column in DRUG_COLUMNS:
            if column not in self.drugs.columns:
                raise ValueError(f"drugs missing {column}")
        for column in PAIR_COLUMNS:
            if column not in self.pairs.columns:
                raise ValueError(f"pairs missing {column}")
        if self.samples["sample_id"].astype(str).duplicated().any():
            raise ValueError("sample_id must be unique")
        if self.drugs["drug_id"].astype(str).duplicated().any():
            raise ValueError("drug_id must be unique")
        if not self.drugs["canonical_smiles"].astype(str).str.len().gt(0).all():
            raise ValueError("all retained drugs require canonical_smiles")
        if not self.drugs["inchikey"].astype(str).str.len().gt(0).all():
            raise ValueError("all retained drugs require inchikey")
        if not self.pairs["cohort_id"].astype(str).eq(self.cohort_id).all():
            raise ValueError("pair cohort_id does not match dataset cohort_id")
        if not np.isfinite(pd.to_numeric(self.pairs["response"], errors="coerce")).all():
            raise ValueError("response must be finite")
        if not self.pairs["higher_is_sensitive"].isin([True, False]).all():
            raise ValueError("higher_is_sensitive must be boolean for every response")
        unknown_samples = set(self.pairs["sample_id"].astype(str)).difference(self.samples["sample_id"].astype(str))
        unknown_drugs = set(self.pairs["drug_id"].astype(str)).difference(self.drugs["drug_id"].astype(str))
        if unknown_samples or unknown_drugs:
            raise ValueError(f"pair identity is not exact: samples={sorted(unknown_samples)}, drugs={sorted(unknown_drugs)}")
        duplicate_columns = ["sample_id", "drug_id", "response_name"]
        if self.pairs.duplicated(duplicate_columns).any():
            raise ValueError(f"duplicate response pairs by {duplicate_columns}")
        if not np.isfinite(self.expression).all():
            raise ValueError("expression contains non-finite values")

    @property
    def counts(self) -> dict[str, int]:
        return {
            "samples": int(len(self.samples)),
            "patients": int(self.samples.loc[self.samples["patient_id"].astype(str).ne(""), "patient_id"].nunique()),
            "pdo_samples": int(self.samples.loc[self.samples["pdo_id"].astype(str).ne(""), "pdo_id"].nunique()),
            "cell_lines": int(self.samples.loc[self.samples["cell_line_id"].astype(str).ne(""), "cell_line_id"].nunique()),
            "drugs": int(len(self.drugs)),
            "response_rows": int(len(self.pairs)),
            "genes": int(len(self.ordered_genes)),
        }

    def manifest_record(self, *, role: str, sample_type: str, notes: str = "") -> CohortManifest:
        names = sorted(self.pairs["response_name"].astype(str).unique())
        directions = sorted(self.pairs["higher_is_sensitive"].astype(bool).unique().tolist())
        return CohortManifest(
            cohort_id=self.cohort_id,
            role=role,
            sample_type=sample_type,
            patient_id="samples.patient_id",
            pdo_id="samples.pdo_id",
            drug_id="drugs.drug_id",
            canonical_smiles="drugs.canonical_smiles",
            inchikey="drugs.inchikey",
            response_name="|".join(names),
            higher_is_sensitive=directions == [True],
            expression_unit=self.expression_unit,
            counts=self.counts,
            hashes=self.source_hashes,
            response_scale="|".join(sorted(self.pairs["response_scale"].astype(str).unique())),
            notes=notes,
        )

    @staticmethod
    def _text(values: pd.Series | list[str]) -> np.ndarray:
        return np.asarray([str(value) for value in values], dtype="U")

    def write(self, directory: Path) -> dict[str, str]:
        """Write an auditable, non-expanded NPZ/Parquet cohort bundle."""
        directory.mkdir(parents=True, exist_ok=True)
        samples = self.samples.loc[:, SAMPLE_COLUMNS].copy()
        drugs = self.drugs.loc[:, DRUG_COLUMNS].copy()
        pairs = self.pairs.loc[:, PAIR_COLUMNS].copy()
        sample_index = {value: index for index, value in enumerate(samples["sample_id"].astype(str))}
        drug_index = {value: index for index, value in enumerate(drugs["drug_id"].astype(str))}
        pairs["pair_sample_idx"] = pairs["sample_id"].astype(str).map(sample_index).astype(np.int32)
        pairs["pair_drug_idx"] = pairs["drug_id"].astype(str).map(drug_index).astype(np.int32)
        prefix = self.cohort_id
        sample_path = directory / f"{prefix}_samples.parquet"
        drug_path = directory / f"{prefix}_drugs.parquet"
        pair_path = directory / f"{prefix}_pairs.parquet"
        npz_path = directory / f"{prefix}_indexed.npz"
        samples.to_parquet(sample_path, index=False)
        drugs.to_parquet(drug_path, index=False)
        pairs.to_parquet(pair_path, index=False)
        np.savez_compressed(
            npz_path,
            expression=self.expression,
            ordered_genes=np.asarray(self.ordered_genes, dtype="U"),
            sample_id=self._text(samples["sample_id"]),
            patient_id=self._text(samples["patient_id"]),
            pdo_id=self._text(samples["pdo_id"]),
            cell_line_id=self._text(samples["cell_line_id"]),
            sample_tissue=self._text(samples["sample_tissue"]),
            drug_id=self._text(drugs["drug_id"]),
            canonical_smiles=self._text(drugs["canonical_smiles"]),
            inchikey=self._text(drugs["inchikey"]),
            drug_scaffold=self._text(drugs["drug_scaffold"]),
            pair_sample_idx=pairs["pair_sample_idx"].to_numpy(np.int32),
            pair_drug_idx=pairs["pair_drug_idx"].to_numpy(np.int32),
            pair_response=pairs["response"].to_numpy(np.float32),
            pair_cohort_id=self._text(pairs["cohort_id"]),
            pair_response_name=self._text(pairs["response_name"]),
            pair_response_scale=self._text(pairs["response_scale"]),
            pair_higher_is_sensitive=pairs["higher_is_sensitive"].to_numpy(bool),
        )
        return {
            "indexed_npz": sha256_file(npz_path),
            "samples_parquet": sha256_file(sample_path),
            "drugs_parquet": sha256_file(drug_path),
            "pairs_parquet": sha256_file(pair_path),
            "gene_universe": gene_digest(self.ordered_genes),
        }

    def write_json_record(self, path: Path, *, role: str, sample_type: str, notes: str = "") -> None:
        path.write_text(json.dumps(self.manifest_record(role=role, sample_type=sample_type, notes=notes).to_dict(), indent=2) + "\n", encoding="utf-8")
