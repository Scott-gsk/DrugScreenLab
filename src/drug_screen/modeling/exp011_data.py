"""Raw indexed-pair contract and fold construction for EXP-011.

The data stage emits no precomputed split.  This module retains one expression
matrix and one drug table, while the model stage creates auditable patient or
cell-line grouped folds from the response pair indices.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from drug_screen.foundation.pharmaformer_exp011 import OFFICIAL_GENE_INPUT_SIZE


RAW_MANIFEST_SCHEMA = "drugscreenlab.exp011.raw-indexed-pairs.v1"
REQUIRED_ROOT_ARRAYS = (
    "expression", "ordered_genes", "sample_id", "patient_id", "pdo_id", "cell_line_id",
    "drug_id", "canonical_smiles", "inchikey", "drug_scaffold",
)
REQUIRED_PAIR_ARRAYS = (
    "pair_sample_idx", "pair_drug_idx", "pair_response", "pair_cohort_id",
    "pair_response_name", "pair_response_scale", "pair_higher_is_sensitive",
)


def _strings(value: np.ndarray | Sequence[object]) -> np.ndarray:
    return np.asarray(value).astype(str)


def _resolve(manifest_path: Path, raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def _require_unique(name: str, values: np.ndarray) -> None:
    if values.ndim != 1 or not len(values):
        raise ValueError(f"{name} must be a non-empty 1D array")
    if len(set(values.tolist())) != len(values):
        raise ValueError(f"{name} must be unique")


@dataclass(frozen=True)
class IndexedPairData:
    expression: np.ndarray
    ordered_genes: np.ndarray
    sample_id: np.ndarray
    patient_id: np.ndarray
    pdo_id: np.ndarray
    cell_line_id: np.ndarray
    drug_id: np.ndarray
    canonical_smiles: np.ndarray
    inchikey: np.ndarray
    drug_scaffold: np.ndarray
    pair_sample_idx: np.ndarray
    pair_drug_idx: np.ndarray
    pair_response: np.ndarray
    pair_cohort_id: np.ndarray
    pair_response_name: np.ndarray
    pair_response_scale: np.ndarray
    pair_higher_is_sensitive: np.ndarray
    extra_arrays: Mapping[str, np.ndarray]
    manifest: Mapping[str, Any]
    manifest_path: Path

    @property
    def pair_count(self) -> int:
        return int(len(self.pair_response))

    @property
    def gene_count(self) -> int:
        return int(self.expression.shape[1])

    @property
    def formal_gene_contract(self) -> bool:
        return bool(self.manifest.get("formal_gene_contract", True))


@dataclass(frozen=True)
class FoldIndices:
    fold_id: int
    seed: int
    stage: str
    group_field: str
    stratification_field: str
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    group_assignments: Mapping[str, str]

    def as_manifest(self, data: IndexedPairData) -> dict[str, Any]:
        return {
            "schema": "drugscreenlab.exp011.fold-indices.v1",
            "fold_id": self.fold_id,
            "seed": self.seed,
            "stage": self.stage,
            "group_field": self.group_field,
            "stratification_field": self.stratification_field,
            "pair_indices": {"train": self.train.astype(int).tolist(), "validation": self.validation.astype(int).tolist(), "test": self.test.astype(int).tolist()},
            "group_assignments": dict(self.group_assignments),
            "pair_counts": {"train": int(len(self.train)), "validation": int(len(self.validation)), "test": int(len(self.test))},
            "indexed_manifest": str(data.manifest_path),
        }


def _read_manifest(path: Path, cohort_id: str | None) -> tuple[dict[str, Any], Path]:
    """Resolve either one raw manifest or the top-level cohort index manifest."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("manifest must be a JSON object")
    if document.get("schema") == RAW_MANIFEST_SCHEMA:
        if cohort_id is not None and str(document.get("cohort_id", "")) != cohort_id:
            raise ValueError(f"raw manifest cohort_id differs from requested --cohort-id: {cohort_id}")
        return document, path
    cohorts = document.get("cohorts")
    if not isinstance(cohorts, list):
        raise ValueError(f"manifest must be {RAW_MANIFEST_SCHEMA} or a top-level cohort index with cohorts[]")
    if not cohort_id:
        raise ValueError("top-level cohort_index_manifest requires --cohort-id")
    records = [record for record in cohorts if isinstance(record, dict) and str(record.get("cohort_id", "")) == cohort_id]
    if len(records) != 1:
        raise ValueError(f"cohort_id {cohort_id!r} is not uniquely present in top-level manifest")
    record = records[0]
    bundle = record.get("bundle", {})
    raw_ref = record.get("raw_manifest")
    if raw_ref is None and isinstance(bundle, dict):
        raw_ref = bundle.get("raw_manifest")
    if not isinstance(raw_ref, str) or not raw_ref:
        raise ValueError(f"cohort {cohort_id!r} record has no raw_manifest")
    raw_path = _resolve(path, raw_ref)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != RAW_MANIFEST_SCHEMA:
        raise ValueError(f"cohort {cohort_id!r} raw_manifest does not use {RAW_MANIFEST_SCHEMA}")
    if str(raw.get("cohort_id", "")) != cohort_id:
        raise ValueError("top-level cohort record and raw manifest cohort_id differ")
    # Preserve the aggregate provenance path so formal checkpoints can record
    # both the selected raw asset and the cohort index that selected it.
    resolved = dict(raw)
    resolved["cohort_index_manifest"] = str(path)
    resolved["cohort_index_record"] = record
    return resolved, raw_path


def load_indexed_pair_data(manifest_path: str | Path, *, cohort_id: str | None = None) -> IndexedPairData:
    """Load a raw cohort bundle selected directly or via the top-level index."""
    path = Path(manifest_path).resolve()
    if path.suffix == ".npz":
        if cohort_id is not None:
            raise ValueError("--cohort-id is only valid with a JSON manifest, not a direct NPZ development input")
        # This supports dry-run integration before data provenance is frozen.
        # The training CLI rejects it under --formal.
        manifest: dict[str, Any] = {
            "schema": RAW_MANIFEST_SCHEMA,
            "raw_npz": str(path),
            "development_input": True,
            "formal_gene_contract": False,
        }
        archive_path = path
    else:
        manifest, path = _read_manifest(path, cohort_id)
        raw_path = manifest.get("raw_npz")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("raw manifest requires non-empty raw_npz")
        archive_path = _resolve(path, raw_path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"raw indexed NPZ is missing: {archive_path}")
    with np.load(archive_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    missing = set(REQUIRED_ROOT_ARRAYS + REQUIRED_PAIR_ARRAYS) - set(arrays)
    if missing:
        raise ValueError(f"raw indexed NPZ misses formal fields: {sorted(missing)}")

    expression = np.asarray(arrays["expression"], dtype=np.float32)
    ordered_genes = _strings(arrays["ordered_genes"])
    sample_id, patient_id, pdo_id, cell_line_id = (_strings(arrays[key]) for key in ("sample_id", "patient_id", "pdo_id", "cell_line_id"))
    drug_id, canonical_smiles, inchikey, drug_scaffold = (_strings(arrays[key]) for key in ("drug_id", "canonical_smiles", "inchikey", "drug_scaffold"))
    if expression.ndim != 2 or expression.shape != (len(sample_id), len(ordered_genes)):
        raise ValueError("expression must have shape [unique samples, ordered_genes]")
    if not np.isfinite(expression).all():
        raise ValueError("expression contains non-finite values; imputation belongs in data preparation")
    _require_unique("sample_id", sample_id); _require_unique("ordered_genes", ordered_genes); _require_unique("drug_id", drug_id)
    if any(len(values) != len(sample_id) for values in (patient_id, pdo_id, cell_line_id)):
        raise ValueError("patient_id/pdo_id/cell_line_id must align to sample_id")
    if any(len(values) != len(drug_id) for values in (canonical_smiles, inchikey, drug_scaffold)):
        raise ValueError("canonical_smiles/inchikey/drug_scaffold must align to drug_id")
    if not np.all(np.char.str_len(canonical_smiles) > 0):
        raise ValueError("canonical_smiles must be non-empty for every drug")
    if bool(manifest.get("formal_gene_contract", True)) and len(ordered_genes) != OFFICIAL_GENE_INPUT_SIZE:
        raise ValueError(f"formal EXP-011 data must contain {OFFICIAL_GENE_INPUT_SIZE} ordered genes")

    pair_sample_idx = np.asarray(arrays["pair_sample_idx"], dtype=np.int64)
    pair_drug_idx = np.asarray(arrays["pair_drug_idx"], dtype=np.int64)
    pair_response = np.asarray(arrays["pair_response"], dtype=np.float32)
    pair_cohort_id, pair_response_name, pair_response_scale = (_strings(arrays[key]) for key in ("pair_cohort_id", "pair_response_name", "pair_response_scale"))
    pair_higher = np.asarray(arrays["pair_higher_is_sensitive"])
    length = len(pair_response)
    if not length or any(len(value) != length for value in (pair_sample_idx, pair_drug_idx, pair_cohort_id, pair_response_name, pair_response_scale, pair_higher)):
        raise ValueError("all pair arrays must have the same non-zero length")
    if not np.isfinite(pair_response).all():
        raise ValueError("pair_response must be finite")
    if pair_sample_idx.min() < 0 or pair_sample_idx.max() >= len(sample_id):
        raise ValueError("pair_sample_idx references a missing sample")
    if pair_drug_idx.min() < 0 or pair_drug_idx.max() >= len(drug_id):
        raise ValueError("pair_drug_idx references a missing drug")
    if not np.asarray(pair_higher, dtype=bool).all():
        raise ValueError("pair_higher_is_sensitive must be true for every EXP-011 response")
    if any(not value for value in pair_cohort_id) or any(not value for value in pair_response_scale):
        raise ValueError("pair_cohort_id and pair_response_scale must be non-empty")
    extra = {name: value for name, value in arrays.items() if name not in set(REQUIRED_ROOT_ARRAYS + REQUIRED_PAIR_ARRAYS)}
    return IndexedPairData(expression, ordered_genes, sample_id, patient_id, pdo_id, cell_line_id, drug_id, canonical_smiles,
                           inchikey, drug_scaffold, pair_sample_idx, pair_drug_idx, pair_response, pair_cohort_id,
                           pair_response_name, pair_response_scale, np.asarray(pair_higher, dtype=bool), extra, manifest, path)


def _groups_for_stage(data: IndexedPairData, stage: str) -> tuple[str, np.ndarray, str, np.ndarray]:
    if stage == "source":
        groups = data.cell_line_id[data.pair_sample_idx]
        tissue = data.extra_arrays.get("sample_tissue")
        if tissue is not None and len(tissue) == len(data.sample_id):
            return "cell_line_id", groups, "sample_tissue", _strings(tissue)[data.pair_sample_idx]
        return "cell_line_id", groups, "pair_cohort_id_fallback", data.pair_cohort_id
    if stage == "target":
        patient, pdo = data.patient_id[data.pair_sample_idx], data.pdo_id[data.pair_sample_idx]
        groups = np.where(np.char.str_len(patient) > 0, patient, pdo)
        if any(not value for value in groups):
            raise ValueError("target fold requires patient_id or pdo_id for every response pair")
        return "patient_id_or_pdo_id", groups, "pair_cohort_id", data.pair_cohort_id
    raise ValueError("stage must be source or target")


def build_fold_indices(data: IndexedPairData, *, stage: str, fold_id: int, seed: int = 20260823, folds: int = 5, validation_offset: int = 1) -> FoldIndices:
    """Construct deterministic grouped train/validation/test folds at model time."""
    if not 0 <= fold_id < folds or not 1 <= validation_offset < folds:
        raise ValueError("invalid fold_id or validation_offset")
    group_field, groups, strat_field, labels = _groups_for_stage(data, stage)
    if any(not value for value in groups):
        raise ValueError(f"{group_field} must be non-empty for every {stage} response pair")
    if len(np.unique(groups)) < folds:
        raise ValueError(f"{stage} requires at least {folds} distinct grouped identities")
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
    assigned = np.full(data.pair_count, -1, dtype=np.int64)
    strategy = strat_field
    try:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        for current, (_, held) in enumerate(splitter.split(np.zeros(data.pair_count), labels, groups)):
            assigned[held] = current
    except ValueError:
        strategy = f"{strat_field};GroupKFold_fallback"
        splitter = GroupKFold(n_splits=folds)
        for current, (_, held) in enumerate(splitter.split(np.zeros(data.pair_count), groups=groups)):
            assigned[held] = current
    validation_fold = (fold_id + validation_offset) % folds
    train = np.flatnonzero((assigned != fold_id) & (assigned != validation_fold)).astype(np.int64)
    validation, test = np.flatnonzero(assigned == validation_fold).astype(np.int64), np.flatnonzero(assigned == fold_id).astype(np.int64)
    if not len(train) or not len(validation) or not len(test):
        raise RuntimeError("fold construction yielded an empty partition")
    assignments: dict[str, str] = {}
    for group in np.unique(groups).tolist():
        observed = np.unique(assigned[groups == group])
        if len(observed) != 1:
            raise RuntimeError(f"group leakage after fold construction: {group}")
        assignments[str(group)] = "test" if observed[0] == fold_id else "validation" if observed[0] == validation_fold else "train"
    return FoldIndices(fold_id, seed, stage, group_field, strategy, train, validation, test, assignments)


def build_external_indices(data: IndexedPairData, *, seed: int = 20260823) -> FoldIndices:
    """Represent all external rows as one frozen, no-fit evaluation partition."""
    all_pairs = np.arange(data.pair_count, dtype=np.int64)
    return FoldIndices(
        fold_id=-1,
        seed=int(seed),
        stage="external",
        group_field="none_external_frozen",
        stratification_field="none_external_frozen",
        train=np.empty(0, dtype=np.int64),
        validation=np.empty(0, dtype=np.int64),
        test=all_pairs,
        group_assignments={},
    )


class IndexedPairDataset:
    """Torch-compatible map dataset that lazily gathers unique expression/drug rows."""

    def __init__(self, data: IndexedPairData, pair_indices: Sequence[int], encoded_drugs: np.ndarray) -> None:
        import torch
        indices = np.asarray(pair_indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices) or indices.min() < 0 or indices.max() >= data.pair_count:
            raise ValueError("pair_indices must be non-empty valid pair indices")
        drugs = np.asarray(encoded_drugs, dtype=np.float32)
        if drugs.shape != (len(data.drug_id), 128):
            raise ValueError("encoded drugs must have shape [unique drugs, 128]")
        self.data, self.pair_indices = data, indices
        self.expression, self.encoded_drugs = torch.as_tensor(data.expression, dtype=torch.float32), torch.as_tensor(drugs, dtype=torch.float32)
        self.sample_idx, self.drug_idx = torch.as_tensor(data.pair_sample_idx[indices], dtype=torch.long), torch.as_tensor(data.pair_drug_idx[indices], dtype=torch.long)
        self.targets = torch.as_tensor(data.pair_response[indices], dtype=torch.float32)
        group_text = np.asarray([f"{data.pair_cohort_id[i]}\x1f{data.pair_drug_idx[i]}" for i in indices], dtype=str)
        _, group_idx = np.unique(group_text, return_inverse=True)
        self.group_idx, self.global_pair_idx = torch.as_tensor(group_idx, dtype=torch.long), torch.as_tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.pair_indices)

    def __getitem__(self, index: int) -> tuple[Any, Any, Any, Any, Any]:
        sample, drug = self.sample_idx[index], self.drug_idx[index]
        return self.expression[sample], self.encoded_drugs[drug], self.targets[index], self.group_idx[index], self.global_pair_idx[index]


class CohortDrugBalancedBatchSampler:
    """Target batches contain ranking pairs from the same cohort--drug group."""

    def __init__(self, dataset: IndexedPairDataset, batch_size: int, *, seed: int = 20260823) -> None:
        if batch_size < 2:
            raise ValueError("target ranking requires batch_size >= 2")
        self.dataset, self.batch_size, self.seed = dataset, int(batch_size), int(seed)
        groups: dict[int, list[int]] = {}
        for local_index, group in enumerate(dataset.group_idx.cpu().numpy().tolist()):
            groups.setdefault(int(group), []).append(local_index)
        self.groups = {group: values for group, values in groups.items() if len(values) >= 2}
        if not self.groups:
            raise ValueError("target ranking requires one cohort--drug group with at least two training pairs")

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed); keys = np.asarray(sorted(self.groups), dtype=np.int64); groups_per_batch = max(1, self.batch_size // 2)
        for _ in range(len(self)):
            selected = rng.choice(keys, size=groups_per_batch, replace=len(keys) < groups_per_batch); batch: list[int] = []
            for group in selected.tolist():
                choices = np.asarray(self.groups[int(group)], dtype=np.int64)
                batch.extend(int(value) for value in rng.choice(choices, size=2, replace=len(choices) < 2).tolist())
            if len(batch) < self.batch_size:
                batch.append(int(rng.integers(0, len(self.dataset))))
            yield batch[:self.batch_size]

    def __len__(self) -> int:
        return max(1, int(np.ceil(len(self.dataset) / self.batch_size)))
