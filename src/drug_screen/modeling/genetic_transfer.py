"""Unified chemical/genetic Delta978 response model for the E2 FAST probe.

The module keeps perturbagen modality and genetic direction explicit.  It is
intentionally small: the purpose of E2 is to test whether genetic supervision
reduces the amount of chemical supervision needed, not to introduce a new
large architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


GENE_COUNT = 978
MODALITIES = ("chemical", "genetic")
DIRECTIONS = ("small_molecule", "knockdown", "overexpression", "mutant")
_SPLITS = {"train", "validation", "test"}


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a canonical non-empty string")
    return value


@dataclass(frozen=True)
class UnifiedResponseRecord:
    """One exact-978 response pair with explicit modality and direction."""

    sample_id: str
    treatment_group_id: str
    perturbagen_id: str
    modality: str
    perturbation_direction: str
    context_id: str
    dose_um: float
    time_h: float
    split: str
    treatment_cache_row: int
    control_cache_row: int
    perturbagen_feature_row: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "UnifiedResponseRecord":
        required = {
            "sample_id",
            "treatment_group_id",
            "perturbagen_id",
            "modality",
            "perturbation_direction",
            "context_id",
            "dose_um",
            "time_h",
            "split",
            "treatment_cache_row",
            "control_cache_row",
            "perturbagen_feature_row",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise ValueError(f"record missing fields: {missing}")
        modality = _required_text(value["modality"], "record.modality")
        direction = _required_text(
            value["perturbation_direction"], "record.perturbation_direction"
        )
        if modality not in MODALITIES:
            raise ValueError(f"unsupported modality: {modality}")
        if direction not in DIRECTIONS:
            raise ValueError(f"unsupported perturbation_direction: {direction}")
        if modality == "chemical" and direction != "small_molecule":
            raise ValueError("chemical records must use small_molecule direction")
        if modality == "genetic" and direction == "small_molecule":
            raise ValueError("genetic records require an explicit genetic direction")
        split = _required_text(value["split"], "record.split")
        if split not in _SPLITS:
            raise ValueError(f"unsupported split: {split}")
        dose_um = float(value["dose_um"])
        time_h = float(value["time_h"])
        if not np.isfinite(dose_um) or dose_um <= 0 or not np.isfinite(time_h) or time_h <= 0:
            raise ValueError("dose_um and time_h must be finite positive values")

        def nonnegative_int(field: str) -> int:
            raw = value[field]
            if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)) or int(raw) < 0:
                raise ValueError(f"{field} must be a non-negative integer")
            return int(raw)

        return cls(
            sample_id=_required_text(value["sample_id"], "record.sample_id"),
            treatment_group_id=_required_text(
                value["treatment_group_id"], "record.treatment_group_id"
            ),
            perturbagen_id=_required_text(value["perturbagen_id"], "record.perturbagen_id"),
            modality=modality,
            perturbation_direction=direction,
            context_id=_required_text(value["context_id"], "record.context_id"),
            dose_um=dose_um,
            time_h=time_h,
            split=split,
            treatment_cache_row=nonnegative_int("treatment_cache_row"),
            control_cache_row=nonnegative_int("control_cache_row"),
            perturbagen_feature_row=nonnegative_int("perturbagen_feature_row"),
        )


class UnifiedResponseModel(nn.Module):
    """Low-capacity Context × Perturbagen × Dose/Time response model."""

    def __init__(
        self,
        chemical_dim: int,
        context_dim: int = GENE_COUNT,
        hidden_dim: int = 64,
        gene_count: int = GENE_COUNT,
    ) -> None:
        super().__init__()
        if chemical_dim < 1 or context_dim < 1 or hidden_dim < 1 or gene_count < 1:
            raise ValueError("model dimensions must be positive")
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.perturbagen_projection = nn.Linear(chemical_dim, hidden_dim)
        self.dose_time_projection = nn.Linear(2, hidden_dim)
        self.modality_embedding = nn.Embedding(len(MODALITIES), hidden_dim)
        self.direction_embedding = nn.Embedding(len(DIRECTIONS), hidden_dim)
        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(),
        )
        self.output = nn.Linear(hidden_dim, gene_count)

    def forward(
        self,
        context: torch.Tensor | np.ndarray,
        perturbagen: torch.Tensor | np.ndarray,
        dose_time: torch.Tensor | np.ndarray,
        modality: torch.Tensor | np.ndarray,
        direction: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        context_tensor = torch.as_tensor(context, dtype=torch.float32)
        perturbagen_tensor = torch.as_tensor(perturbagen, dtype=torch.float32)
        dose_time_tensor = torch.as_tensor(dose_time, dtype=torch.float32)
        modality_tensor = torch.as_tensor(modality, dtype=torch.long)
        direction_tensor = torch.as_tensor(direction, dtype=torch.long)
        context_z = self.context_projection(context_tensor)
        perturbagen_z = self.perturbagen_projection(perturbagen_tensor)
        dose_time_z = self.dose_time_projection(dose_time_tensor)
        modality_z = self.modality_embedding(modality_tensor)
        direction_z = self.direction_embedding(direction_tensor)
        return self.output(
            self.interaction(
                torch.cat(
                    [context_z, perturbagen_z, dose_time_z, modality_z, direction_z],
                    dim=-1,
                )
            )
        )


def group_atomic_subset(
    records: Sequence[UnifiedResponseRecord], *, fraction: float, seed: int
) -> tuple[UnifiedResponseRecord, ...]:
    """Select a deterministic fraction of complete perturbation groups."""
    if not np.isfinite(fraction) or fraction <= 0 or fraction > 1:
        raise ValueError("fraction must be in (0, 1]")
    groups: dict[str, list[UnifiedResponseRecord]] = {}
    for record in records:
        groups.setdefault(record.treatment_group_id, []).append(record)
    if not groups:
        raise ValueError("records must not be empty")
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest(),
    )
    count = max(1, math.ceil(len(ordered) * fraction))
    selected_groups = set(ordered[:count])
    return tuple(record for record in records if record.treatment_group_id in selected_groups)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    observed_flat = observed.reshape(-1)
    predicted_flat = predicted.reshape(-1)
    pearson = None
    if np.std(observed_flat) > 0 and np.std(predicted_flat) > 0:
        pearson = float(np.corrcoef(observed_flat, predicted_flat)[0, 1])
    observed_rank = _rank(observed_flat)
    predicted_rank = _rank(predicted_flat)
    spearman = None
    if np.std(observed_rank) > 0 and np.std(predicted_rank) > 0:
        spearman = float(np.corrcoef(observed_rank, predicted_rank)[0, 1])
    nonzero = observed_flat != 0
    return {
        "pearson": pearson,
        "spearman": spearman,
        "rmse": float(np.sqrt(np.mean((observed_flat - predicted_flat) ** 2))),
        "mae": float(np.mean(np.abs(observed_flat - predicted_flat))),
        "direction_accuracy": float(
            np.mean((observed_flat[nonzero] > 0) == (predicted_flat[nonzero] > 0))
        )
        if np.any(nonzero)
        else None,
    }


def _normalization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return center, scale


def _arrays(
    records: Sequence[UnifiedResponseRecord],
    cache: np.ndarray,
    perturbagen_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("records must not be empty")
    context = np.asarray([cache[row.control_cache_row] for row in records], dtype=np.float32)
    perturbagen = np.asarray(
        [perturbagen_features[row.perturbagen_feature_row] for row in records],
        dtype=np.float32,
    )
    dose_time = np.asarray([[row.dose_um, row.time_h] for row in records], dtype=np.float32)
    modality = np.asarray([MODALITIES.index(row.modality) for row in records], dtype=np.int64)
    direction = np.asarray(
        [DIRECTIONS.index(row.perturbation_direction) for row in records], dtype=np.int64
    )
    targets = np.asarray(
        [cache[row.treatment_cache_row] - cache[row.control_cache_row] for row in records],
        dtype=np.float32,
    )
    return context, perturbagen, dose_time, modality, direction, targets


def _validate_arrays(cache: np.ndarray, perturbagen_features: np.ndarray) -> None:
    if cache.ndim != 2 or cache.shape[1] != GENE_COUNT or not np.isfinite(cache).all():
        raise ValueError("cache must be finite with shape [n, 978]")
    if perturbagen_features.ndim != 2 or perturbagen_features.shape[1] < 1:
        raise ValueError("perturbagen_features must be a non-empty 2D array")
    if not np.isfinite(perturbagen_features).all():
        raise ValueError("perturbagen_features must be finite")


def fit_transfer_probe(
    *,
    genetic_records: Sequence[UnifiedResponseRecord],
    chemical_train_records: Sequence[UnifiedResponseRecord],
    chemical_test_records: Sequence[UnifiedResponseRecord],
    cache: np.ndarray,
    perturbagen_features: np.ndarray,
    chemical_fraction: float,
    genetic_epochs: int = 8,
    chemical_epochs: int = 12,
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 20260813,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run one E2 low-data comparison with a frozen chemical test set."""
    _validate_arrays(cache, perturbagen_features)
    if not genetic_records or not chemical_train_records or not chemical_test_records:
        raise ValueError("genetic, chemical train, and chemical test records are required")
    if genetic_epochs < 0 or chemical_epochs < 1:
        raise ValueError("genetic_epochs must be non-negative and chemical_epochs positive")
    chemical_train = group_atomic_subset(
        chemical_train_records, fraction=chemical_fraction, seed=seed
    )
    if any(row.split != "train" for row in chemical_train):
        raise ValueError("chemical training records must all have split=train")
    if any(row.split != "train" for row in genetic_records):
        raise ValueError("genetic pretraining records must all have split=train")
    if any(row.split != "test" for row in chemical_test_records):
        raise ValueError("chemical test records must all have split=test")

    chemical_context, chemical_perturbagen, chemical_dose, chemical_modality, chemical_direction, chemical_targets = _arrays(
        chemical_train, cache, perturbagen_features
    )
    test_context, test_perturbagen, test_dose, test_modality, test_direction, test_targets = _arrays(
        chemical_test_records, cache, perturbagen_features
    )
    genetic_context, genetic_perturbagen, genetic_dose, genetic_modality, genetic_direction, genetic_targets = _arrays(
        genetic_records, cache, perturbagen_features
    )

    context_center, context_scale = _normalization(chemical_context)
    dose_center, dose_scale = _normalization(np.log1p(np.maximum(chemical_dose, 0.0)))
    target_center, target_scale = _normalization(chemical_targets)

    def normalize(
        context: np.ndarray,
        perturbagen: np.ndarray,
        dose: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            (context - context_center) / context_scale,
            perturbagen,
            (np.log1p(np.maximum(dose, 0.0)) - dose_center) / dose_scale,
            (target - target_center) / target_scale,
        )

    chemical_context, chemical_perturbagen, chemical_dose, chemical_targets = normalize(
        chemical_context, chemical_perturbagen, chemical_dose, chemical_targets
    )
    test_context, test_perturbagen, test_dose, test_targets = normalize(
        test_context, test_perturbagen, test_dose, test_targets
    )
    genetic_context, genetic_perturbagen, genetic_dose, genetic_targets = normalize(
        genetic_context, genetic_perturbagen, genetic_dose, genetic_targets
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    def tensors(
        context: np.ndarray,
        perturbagen: np.ndarray,
        dose: np.ndarray,
        modality: np.ndarray,
        direction: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[torch.Tensor, ...]:
        return (
            torch.from_numpy(context).to(requested_device),
            torch.from_numpy(perturbagen).to(requested_device),
            torch.from_numpy(dose).to(requested_device),
            torch.from_numpy(modality).to(requested_device),
            torch.from_numpy(direction).to(requested_device),
            torch.from_numpy(targets).to(requested_device),
        )

    chemical_tensors = tensors(
        chemical_context,
        chemical_perturbagen,
        chemical_dose,
        chemical_modality,
        chemical_direction,
        chemical_targets,
    )
    genetic_tensors = tensors(
        genetic_context,
        genetic_perturbagen,
        genetic_dose,
        genetic_modality,
        genetic_direction,
        genetic_targets,
    )
    test_tensors = tensors(
        test_context,
        test_perturbagen,
        test_dose,
        test_modality,
        test_direction,
        test_targets,
    )

    def train_model(*, pretrain_genetic: bool) -> dict[str, Any]:
        torch.manual_seed(seed)
        model = UnifiedResponseModel(
            chemical_dim=perturbagen_features.shape[1],
            context_dim=GENE_COUNT,
            hidden_dim=hidden_dim,
            gene_count=GENE_COUNT,
        ).to(requested_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        def train_epochs(batch: tuple[torch.Tensor, ...], epochs: int) -> None:
            if epochs < 1:
                return
            model.train()
            for _ in range(epochs):
                optimizer.zero_grad(set_to_none=True)
                predicted = model(*batch[:-1])
                loss = torch.mean((predicted - batch[-1]) ** 2)
                loss.backward()
                optimizer.step()

        if pretrain_genetic:
            train_epochs(genetic_tensors, genetic_epochs)
        train_epochs(chemical_tensors, chemical_epochs)
        model.eval()
        with torch.no_grad():
            train_pred_norm = model(*chemical_tensors[:-1]).cpu().numpy()
            test_pred_norm = model(*test_tensors[:-1]).cpu().numpy()
        train_pred = train_pred_norm * target_scale + target_center
        test_pred = test_pred_norm * target_scale + target_center
        raw_chemical_targets = chemical_targets * target_scale + target_center
        raw_test_targets = test_targets * target_scale + target_center
        return {
            "train_metrics": _metrics(raw_chemical_targets, train_pred),
            "test_metrics": _metrics(raw_test_targets, test_pred),
            "prediction_variance": float(np.var(test_pred)),
            "pretrained_on_genetic": pretrain_genetic,
        }

    return {
        "format": "genetic_chemical_transfer_fast_probe_v1",
        "chemical_fraction": float(chemical_fraction),
        "chemical_train_group_count": len({row.treatment_group_id for row in chemical_train}),
        "chemical_test_group_count": len(
            {row.treatment_group_id for row in chemical_test_records}
        ),
        "genetic_pretrain_group_count": len(
            {row.treatment_group_id for row in genetic_records}
        ),
        "normalization_fit_scope": "chemical_train_only",
        "chemical_test_frozen": True,
        "models": {
            "chemical_only": train_model(pretrain_genetic=False),
            "genetic_pretrain_then_chemical": train_model(pretrain_genetic=True),
        },
    }
