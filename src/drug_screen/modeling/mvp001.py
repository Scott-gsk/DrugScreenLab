"""MVP-001's deliberately small learned perturbation predictor.

This module consumes a Data Steward compact manifest, never a GCTX file.  A
manifest must explicitly bind every target to two rows of the registered
exact-978 cache; matching controls or creating splits is outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


EXPERIMENT_ID = "MVP-001"
MANIFEST_FORMAT = "mvp001_compact_cache_manifest_v1"
PREDICTION_FORMAT = "mvp001_predicted_delta978_v1"
GENE_COUNT = 978
PHASES = frozenset({"tiny", "small", "mvp_full"})
SPLITS = frozenset({"train", "validation", "test"})


class IntegrationError(RuntimeError):
    """A required Data Contract input is missing or inconsistent."""


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _canonical_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise IntegrationError(f"manifest {name} must be a canonical non-empty string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IntegrationError(f"manifest {name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class PerturbationRow:
    sample_id: str
    treatment_group_id: str
    drug_id: str
    dose_id: str
    time_id: str
    split: str
    treatment_cache_row: int
    control_cache_row: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PerturbationRow":
        required = {
            "sample_id", "treatment_group_id", "drug_id", "dose_id", "time_id", "split",
            "treatment_cache_row", "control_cache_row",
        }
        missing = required.difference(value)
        if missing:
            raise IntegrationError(f"manifest record missing fields: {sorted(missing)}")
        split = _canonical_text(value["split"], "record.split")
        if split not in SPLITS:
            raise IntegrationError(f"manifest record has unsupported split: {split}")
        return cls(
            sample_id=_canonical_text(value["sample_id"], "record.sample_id"),
            treatment_group_id=_canonical_text(value["treatment_group_id"], "record.treatment_group_id"),
            drug_id=_canonical_text(value["drug_id"], "record.drug_id"),
            dose_id=_canonical_text(value["dose_id"], "record.dose_id"),
            time_id=_canonical_text(value["time_id"], "record.time_id"),
            split=split,
            treatment_cache_row=_integer(value["treatment_cache_row"], "record.treatment_cache_row"),
            control_cache_row=_integer(value["control_cache_row"], "record.control_cache_row"),
        )


@dataclass(frozen=True)
class CompactManifest:
    cache_path: Path
    cache_sha256: str
    cache_shape: tuple[int, int]
    records: tuple[PerturbationRow, ...]
    source_digest: str

    @classmethod
    def load(cls, path: Path | str, *, data_root: Path | str | None = None) -> "CompactManifest":
        manifest_path = Path(path)
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise IntegrationError(f"compact manifest is unreadable: {manifest_path}") from error
        if payload.get("format") != MANIFEST_FORMAT:
            raise IntegrationError(f"manifest format must be {MANIFEST_FORMAT}")
        cache = payload.get("cache")
        if not isinstance(cache, Mapping):
            raise IntegrationError("manifest must provide a cache mapping")
        relative = _canonical_text(cache.get("relative_path"), "cache.relative_path")
        cache_path = Path(data_root) / relative if data_root is not None else Path(relative)
        cache_sha256 = _canonical_text(cache.get("sha256"), "cache.sha256")
        shape = cache.get("shape")
        if not isinstance(shape, list) or len(shape) != 2 or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in shape
        ):
            raise IntegrationError("manifest cache.shape must contain two positive integers")
        if shape[1] != GENE_COUNT:
            raise IntegrationError("manifest cache must retain exact 978 output genes")
        records_payload = payload.get("records")
        if not isinstance(records_payload, list) or not records_payload:
            raise IntegrationError("manifest records must be a non-empty list")
        records = tuple(PerturbationRow.from_mapping(v) for v in records_payload if isinstance(v, Mapping))
        if len(records) != len(records_payload) or len({r.sample_id for r in records}) != len(records):
            raise IntegrationError("manifest record identities must be unique mappings")
        group_splits: dict[str, str] = {}
        for row in records:
            previous = group_splits.setdefault(row.treatment_group_id, row.split)
            if previous != row.split:
                raise IntegrationError("manifest treatment groups must not cross splits")
        if not any(row.split == "train" for row in records):
            raise IntegrationError("manifest must contain train perturbations")
        if not any(row.split == "test" for row in records):
            raise IntegrationError("manifest must contain held-out test perturbations")
        for row in records:
            if row.treatment_cache_row >= shape[0] or row.control_cache_row >= shape[0]:
                raise IntegrationError("manifest cache row is outside declared cache shape")
        return cls(cache_path, cache_sha256, (shape[0], shape[1]), records, _digest(manifest_path))

    def load_cache(self) -> np.ndarray:
        if not self.cache_path.is_file():
            raise IntegrationError(f"registered cache is absent: {self.cache_path}")
        if _digest(self.cache_path) != self.cache_sha256:
            raise IntegrationError("registered cache checksum does not match compact manifest")
        try:
            values = np.load(self.cache_path, mmap_mode="r")
        except (OSError, ValueError) as error:
            raise IntegrationError("registered cache is not a readable NPY array") from error
        if values.dtype != np.float32 or values.shape != self.cache_shape:
            raise IntegrationError("registered cache dtype or shape differs from compact manifest")
        return values


class _Rows(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, rows: Sequence[PerturbationRow], targets: np.ndarray, vocabularies: Mapping[str, Mapping[str, int]]):
        self.rows, self.targets, self.vocabularies = tuple(rows), targets, vocabularies

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        return {
            "drug": torch.tensor(self.vocabularies["drug"][row.drug_id]),
            "dose": torch.tensor(self.vocabularies["dose"][row.dose_id]),
            "time": torch.tensor(self.vocabularies["time"][row.time_id]),
            "target": torch.from_numpy(self.targets[index]),
        }


class DrugDoseTimePredictor(nn.Module):
    """A single low-capacity additive categorical representation and 978-gene head."""

    def __init__(self, drug_count: int, dose_count: int, time_count: int, embedding_dim: int):
        super().__init__()
        self.drug = nn.Embedding(drug_count, embedding_dim)
        self.dose = nn.Embedding(dose_count, embedding_dim)
        self.time = nn.Embedding(time_count, embedding_dim)
        self.output = nn.Linear(embedding_dim, GENE_COUNT)

    def forward(self, drug: torch.Tensor, dose: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self.output(self.drug(drug) + self.dose(dose) + self.time(time))


def _vocabulary(rows: Sequence[PerturbationRow], field: str) -> dict[str, int]:
    return {value: index for index, value in enumerate(sorted({getattr(row, field) for row in rows}))}


def _pearson(observed: np.ndarray, predicted: np.ndarray) -> float | None:
    if observed.size < 2 or np.std(observed) == 0.0 or np.std(predicted) == 0.0:
        return None
    return float(np.corrcoef(observed, predicted)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(observed: np.ndarray, predicted: np.ndarray) -> float | None:
    return _pearson(_rank(observed), _rank(predicted))


def _unit_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    return {
        "pearson": _pearson(observed, predicted),
        "spearman": _spearman(observed, predicted),
        "rmse": float(np.sqrt(np.mean((observed - predicted) ** 2))),
        "mae": float(np.mean(np.abs(observed - predicted))),
    }


def _grouped_metrics(rows: Sequence[PerturbationRow], observed: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(row.treatment_group_id, []).append(index)
    units: list[dict[str, Any]] = []
    for group_id, indices in sorted(grouped.items()):
        first = rows[indices[0]]
        units.append({
            "treatment_group_id": group_id,
            "drug_id": first.drug_id,
            "replicate_count": len(indices),
            "metrics": _unit_metrics(np.mean(observed[indices], axis=0), np.mean(predicted[indices], axis=0)),
        })
    by_drug: dict[str, list[dict[str, Any]]] = {}
    for unit in units:
        by_drug.setdefault(unit["drug_id"], []).append(unit)
    macro_by_drug = {}
    for drug, values in sorted(by_drug.items()):
        macro_by_drug[drug] = {"eligible_group_count": len(values), **{
            metric: (float(np.mean([value["metrics"][metric] for value in values if value["metrics"][metric] is not None])) if any(value["metrics"][metric] is not None for value in values) else None)
            for metric in ("pearson", "spearman", "rmse", "mae")
        }}
    return {"units": units, "macro_by_drug": macro_by_drug, "macro_across_drugs": {
        metric: (float(np.mean([values[metric] for values in macro_by_drug.values() if values[metric] is not None])) if any(values[metric] is not None for values in macro_by_drug.values()) else None)
        for metric in ("pearson", "spearman", "rmse", "mae")
    }}


def _grouped_targets(rows: Sequence[PerturbationRow], targets: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Return one equal-weight target vector per treatment group.

    Technical replicates are an observation detail, not additional training
    groups.  Keeping this aggregation explicit prevents a group with more
    replicates from changing the train-only baselines (and makes the
    diagnostic agree with the group-level evaluation unit).
    """
    if len(rows) != len(targets):
        raise ValueError("rows and targets must have the same length")
    indices: dict[str, list[int]] = {}
    drugs: dict[str, str] = {}
    for index, row in enumerate(rows):
        indices.setdefault(row.treatment_group_id, []).append(index)
        previous = drugs.setdefault(row.treatment_group_id, row.drug_id)
        if previous != row.drug_id:
            raise IntegrationError("treatment groups must not contain multiple drugs")
    means = {
        group_id: np.mean(targets[group_indices], axis=0, dtype=np.float32)
        for group_id, group_indices in sorted(indices.items())
    }
    return means, drugs


def _select_subset(records: Sequence[PerturbationRow], max_records: int | None) -> tuple[PerturbationRow, ...]:
    """Select a deterministic bounded subset without splitting treatment groups.

    The manifest order is frozen by the Data Steward.  A row-count bound is
    applied to complete treatment groups in that order; train groups needed to
    provide the selected test groups' categorical vocabulary are then added as
    complete groups.  The latter may make the final count exceed the bound,
    which is intentional and mirrors the previous required-support behavior.
    """
    if max_records is None:
        return tuple(records)

    groups: dict[str, list[PerturbationRow]] = {}
    for row in records:
        groups.setdefault(row.treatment_group_id, []).append(row)
    selected: list[PerturbationRow] = []
    selected_group_ids: set[str] = set()
    for group_id, group_rows in groups.items():
        if len(selected) and len(selected) + len(group_rows) > max_records:
            break
        if not selected and len(group_rows) > max_records:
            raise IntegrationError("data.max_records is smaller than the first treatment group")
        selected.extend(group_rows)
        selected_group_ids.add(group_id)

    if not selected:
        raise IntegrationError("data.max_records produced an empty deterministic subset")

    required_conditions = {
        (row.drug_id, row.dose_id, row.time_id)
        for row in selected
        if row.split == "test"
    }
    for group_id, group_rows in groups.items():
        if group_id in selected_group_ids:
            continue
        if any(
            row.split == "train"
            and (row.drug_id, row.dose_id, row.time_id) in required_conditions
            for row in group_rows
        ):
            selected.extend(group_rows)
            selected_group_ids.add(group_id)
    return tuple(selected)


def _load_checkpoint(path: Path | str) -> dict[str, Any]:
    """Load a locally generated MVP checkpoint on CPU.

    ``weights_only=False`` is intentional here: the checkpoint contains the
    small categorical vocabularies alongside the state dict.  The fallback
    keeps the helper compatible with older supported PyTorch releases.
    """
    checkpoint_path = Path(path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise IntegrationError("MVP-001 checkpoint must contain a mapping")
    if checkpoint.get("experiment_id") != EXPERIMENT_ID:
        raise IntegrationError("checkpoint experiment_id does not match MVP-001")
    if not isinstance(checkpoint.get("model_state_dict"), Mapping) or not isinstance(checkpoint.get("vocabularies"), Mapping):
        raise IntegrationError("MVP-001 checkpoint is missing model state or vocabularies")
    return checkpoint


def write_prediction_artifact(
    checkpoint_path: Path | str,
    manifest_path: Path | str,
    output_path: Path | str,
    *,
    data_root: Path | str | None = None,
    max_records: int | None = None,
) -> dict[str, Any]:
    """Write compact predicted-Delta978 vectors for manifest conditions.

    The helper performs no training and never loads the expression cache.  It
    uses the checkpoint's frozen categorical vocabularies and the manifest's
    deterministic condition cohort, then writes one row per unique
    ``(drug, dose, time)`` plus a coordinate-wise per-drug median.  The cache
    checksum is retained as provenance from the manifest; cache values are not
    read during prediction generation.
    """
    if max_records is not None and (isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 1):
        raise ValueError("max_records must be a positive integer or null")
    manifest = CompactManifest.load(manifest_path, data_root=data_root)
    checkpoint = _load_checkpoint(checkpoint_path)
    vocabularies = checkpoint["vocabularies"]
    if not all(isinstance(vocabularies.get(kind), Mapping) for kind in ("drug", "dose", "time")):
        raise IntegrationError("checkpoint vocabularies must contain drug, dose, and time mappings")
    records = _select_subset(manifest.records, max_records)
    conditions = sorted({(row.drug_id, row.dose_id, row.time_id) for row in records})
    if not conditions:
        raise IntegrationError("prediction cohort contains no unique drug/dose/time conditions")
    for drug, dose, time in conditions:
        for kind, value in (("drug", drug), ("dose", dose), ("time", time)):
            if value not in vocabularies[kind]:
                raise IntegrationError(f"prediction cohort has unseen {kind}: {value}")

    state = checkpoint["model_state_dict"]
    try:
        embedding_dim = int(state["drug.weight"].shape[1])
        model = DrugDoseTimePredictor(
            len(vocabularies["drug"]),
            len(vocabularies["dose"]),
            len(vocabularies["time"]),
            embedding_dim,
        )
        model.load_state_dict(state)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise IntegrationError("MVP-001 checkpoint state is incompatible with its vocabularies") from error
    model.eval()
    drug_index = torch.tensor([vocabularies["drug"][drug] for drug, _, _ in conditions], dtype=torch.long)
    dose_index = torch.tensor([vocabularies["dose"][dose] for _, dose, _ in conditions], dtype=torch.long)
    time_index = torch.tensor([vocabularies["time"][time] for _, _, time in conditions], dtype=torch.long)
    with torch.no_grad():
        predicted = model(drug_index, dose_index, time_index).numpy().astype(np.float32, copy=False)
    condition_counts = np.asarray(
        [sum((row.drug_id, row.dose_id, row.time_id) == condition for row in records) for condition in conditions],
        dtype=np.int64,
    )
    drug_ids = sorted({condition[0] for condition in conditions})
    drug_medians = np.stack([
        np.median(predicted[[index for index, condition in enumerate(conditions) if condition[0] == drug]], axis=0)
        for drug in drug_ids
    ]).astype(np.float32, copy=False)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        format=np.asarray(PREDICTION_FORMAT),
        gene_count=np.asarray(GENE_COUNT, dtype=np.int64),
        condition_drug_id=np.asarray([condition[0] for condition in conditions]),
        condition_dose_id=np.asarray([condition[1] for condition in conditions]),
        condition_time_id=np.asarray([condition[2] for condition in conditions]),
        condition_group_count=condition_counts,
        delta978=predicted,
        drug_id=np.asarray(drug_ids),
        drug_median_delta978=drug_medians,
    )
    artifact_digest = _digest(output)
    metadata = {
        "format": PREDICTION_FORMAT,
        "experiment_id": EXPERIMENT_ID,
        "status": "PREDICTION_ARTIFACT_COMPLETE",
        "checkpoint": str(Path(checkpoint_path)),
        "checkpoint_sha256": _digest(Path(checkpoint_path)),
        "manifest": str(Path(manifest_path)),
        "manifest_sha256": manifest.source_digest,
        "cache_sha256": manifest.cache_sha256,
        "cache_values_loaded": False,
        "gene_count": GENE_COUNT,
        "requested_max_records": max_records,
        "selected_rows": len(records),
        "condition_count": len(conditions),
        "drug_count": len(drug_ids),
        "aggregation": "coordinate-wise median across unique manifest conditions per drug",
        "candidate_selection": "predeclared manifest drug/dose/time cohort; no PRISM response labels read",
        "artifact": str(output),
        "artifact_sha256": artifact_digest,
    }
    metadata_path = output.with_suffix(".json")
    metadata["metadata"] = str(metadata_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID or config.get("phase") not in PHASES:
        raise ValueError("MVP-001 config has an invalid experiment_id or phase")
    model, training, data = config.get("model"), config.get("training"), config.get("data")
    if not all(isinstance(value, Mapping) for value in (model, training, data)):
        raise ValueError("MVP-001 config requires data, model, and training mappings")
    if data.get("manifest_format") != MANIFEST_FORMAT or data.get("gene_count") != GENE_COUNT:
        raise ValueError("MVP-001 config changes the compact manifest or exact-978 contract")
    for section, name in ((model, "embedding_dim"), (training, "batch_size"), (training, "epochs")):
        if isinstance(section.get(name), bool) or not isinstance(section.get(name), int) or section[name] < 1:
            raise ValueError(f"MVP-001 config {name} must be positive")
    if not isinstance(training.get("seed"), int) or isinstance(training["seed"], bool):
        raise ValueError("MVP-001 config seed must be an integer")
    if not isinstance(training.get("learning_rate"), (float, int)) or training["learning_rate"] <= 0:
        raise ValueError("MVP-001 config learning_rate must be positive")


def load_config(path: Path | str) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_config(config)
    return config


def run(config: Mapping[str, Any], manifest_path: Path | str, output_dir: Path | str, *, data_root: Path | str | None = None) -> dict[str, Any]:
    """Train one seed and write only compact summary/checkpoint artifacts."""
    _validate_config(config)
    manifest = CompactManifest.load(manifest_path, data_root=data_root)
    cache = manifest.load_cache()
    max_records = config["data"].get("max_records")
    if max_records is not None:
        if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 1:
            raise ValueError("data.max_records must be a positive integer or null")
        records = _select_subset(manifest.records, max_records)
    else:
        records = manifest.records
    train_rows = tuple(row for row in records if row.split == "train")
    test_rows = tuple(row for row in records if row.split == "test")
    if not train_rows or not test_rows:
        raise IntegrationError("configured deterministic subset must retain train and test rows")
    # Within-drug holdout is defined by the frozen manifest, not reconstructed here.
    train_drugs = {row.drug_id for row in train_rows}
    if any(row.drug_id not in train_drugs for row in test_rows):
        raise IntegrationError("held-out perturbation test rows must retain a train occurrence of each drug")
    targets_by_id = {
        row.sample_id: np.asarray(
            cache[row.treatment_cache_row] - cache[row.control_cache_row], dtype=np.float32
        )
        for row in records
    }
    targets = np.stack([targets_by_id[row.sample_id] for row in records])
    if not np.isfinite(targets).all():
        raise IntegrationError("cache rows selected by manifest produce non-finite Delta978 targets")
    vocabularies = {"drug": _vocabulary(train_rows, "drug_id"), "dose": _vocabulary(train_rows, "dose_id"), "time": _vocabulary(train_rows, "time_id")}
    for row in test_rows:
        for kind, field in (("drug", "drug_id"), ("dose", "dose_id"), ("time", "time_id")):
            if getattr(row, field) not in vocabularies[kind]:
                raise IntegrationError(f"held-out perturbation has unseen {kind}; MVP-001 supports within-vocabulary holdout only")
    seed = config["training"]["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = DrugDoseTimePredictor(len(vocabularies["drug"]), len(vocabularies["dose"]), len(vocabularies["time"]), config["model"]["embedding_dim"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"].get("weight_decay", 0.0)))
    train_targets = np.stack([targets_by_id[row.sample_id] for row in train_rows])
    loader = DataLoader(_Rows(train_rows, train_targets, vocabularies), batch_size=config["training"]["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    model.train()
    for _ in range(config["training"]["epochs"]):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(batch["drug"], batch["dose"], batch["time"]) - batch["target"]) ** 2)
            loss.backward(); optimizer.step()
    test_targets = np.stack([targets_by_id[row.sample_id] for row in test_rows])
    model.eval()
    with torch.no_grad():
        test_dataset = _Rows(test_rows, test_targets, vocabularies)
        predicted = np.stack([model(item["drug"].view(1), item["dose"].view(1), item["time"].view(1)).squeeze(0).numpy() for item in test_dataset])
    train_group_means, train_group_drugs = _grouped_targets(train_rows, train_targets)
    # Baselines are defined over treatment groups, not technical replicate
    # rows.  This keeps a group with three replicates from receiving 3x the
    # weight of a one-replicate group.
    constant_vector = np.mean(np.stack(list(train_group_means.values())), axis=0, dtype=np.float32)
    constant = np.repeat(constant_vector[None, :], len(test_rows), axis=0)
    drug_group_means: dict[str, list[np.ndarray]] = {}
    for group_id, target in train_group_means.items():
        drug_group_means.setdefault(train_group_drugs[group_id], []).append(target)
    drug_means = {
        drug: np.mean(np.stack(group_targets), axis=0, dtype=np.float32)
        for drug, group_targets in drug_group_means.items()
    }
    diagnostic = np.stack([drug_means[row.drug_id] for row in test_rows])
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    torch.save({"experiment_id": EXPERIMENT_ID, "seed": seed, "model_state_dict": model.state_dict(), "vocabularies": vocabularies}, checkpoint)
    result = {"experiment_id": EXPERIMENT_ID, "phase": config["phase"], "status": "MODEL_STAGE_COMPLETE", "config_sha256": sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "manifest_sha256": manifest.source_digest, "cache_sha256": manifest.cache_sha256, "dataset": {"train_rows": len(train_rows), "test_rows": len(test_rows), "gene_count": GENE_COUNT, "split": "predeclared_within_drug_perturbation_holdout"}, "subset": {"requested_max_records": max_records, "selection": "manifest_order_complete_treatment_groups_plus_train_support", "treatment_groups_are_atomic": True}, "held_out_metrics": {"learned": _grouped_metrics(test_rows, test_targets, predicted), "constant_train_mean_baseline": _grouped_metrics(test_rows, test_targets, constant), "train_drug_mean_diagnostic": _grouped_metrics(test_rows, test_targets, diagnostic)}, "artifacts": {"checkpoint": str(checkpoint), "checkpoint_sha256": _digest(checkpoint), "predictions": "not_serialized; compact artifact contains no large prediction matrix"}, "known_deviations": ["No external PDO ranking or activity labels are read by the model stage.", "This is a single-seed within-drug perturbation holdout, not cold-drug/context generalization.", "Train optimization currently consumes cache rows; train-only baselines aggregate technical replicates to equal-weight treatment groups."]}
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result
