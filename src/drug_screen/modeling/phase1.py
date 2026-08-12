"""Minimal Phase-1 context-conditioned exact-978 perturbation backbone."""

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


GENE_COUNT = 978
MANIFEST_FORMAT = "phase1_context_chemical_manifest_v1"
SUPPORTED_MANIFEST_FORMATS = {
    MANIFEST_FORMAT,
    "phase1_context_unipert_manifest_v1",
    "phase1_context_mechanism_manifest_v1",
}


class Phase1IntegrationError(RuntimeError):
    """Raised when the Phase-1 data contract is incomplete or unsafe."""


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise Phase1IntegrationError(f"{field} must be a canonical non-empty string")
    return value


def _required_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Phase1IntegrationError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class Phase1Record:
    sample_id: str
    treatment_group_id: str
    drug_id: str
    context_id: str
    dose_um: float
    time_h: float
    split: str
    treatment_cache_row: int
    control_cache_row: int
    chemical_feature_row: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Phase1Record":
        required = {
            "sample_id", "treatment_group_id", "drug_id", "context_id", "dose_um",
            "time_h", "split", "treatment_cache_row", "control_cache_row",
            "chemical_feature_row",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise Phase1IntegrationError(f"record missing fields: {missing}")
        split = _required_text(value["split"], "record.split")
        if split not in {"train", "validation", "test"}:
            raise Phase1IntegrationError(f"unsupported split: {split}")
        dose_um = float(value["dose_um"])
        time_h = float(value["time_h"])
        if not np.isfinite(dose_um) or dose_um <= 0 or not np.isfinite(time_h) or time_h <= 0:
            raise Phase1IntegrationError("dose_um and time_h must be finite positive values")
        return cls(
            sample_id=_required_text(value["sample_id"], "record.sample_id"),
            treatment_group_id=_required_text(value["treatment_group_id"], "record.treatment_group_id"),
            drug_id=_required_text(value["drug_id"], "record.drug_id"),
            context_id=_required_text(value["context_id"], "record.context_id"),
            dose_um=dose_um,
            time_h=time_h,
            split=split,
            treatment_cache_row=_required_int(value["treatment_cache_row"], "record.treatment_cache_row"),
            control_cache_row=_required_int(value["control_cache_row"], "record.control_cache_row"),
            chemical_feature_row=_required_int(value["chemical_feature_row"], "record.chemical_feature_row"),
        )


@dataclass(frozen=True)
class Phase1Manifest:
    path: Path
    cache_path: Path
    cache_sha256: str
    chemical_features_path: Path
    chemical_features_sha256: str
    cache_shape: tuple[int, int]
    chemical_features_shape: tuple[int, int]
    records: tuple[Phase1Record, ...]

    def load_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if _digest(self.cache_path) != self.cache_sha256:
            raise Phase1IntegrationError("cache checksum does not match manifest")
        if _digest(self.chemical_features_path) != self.chemical_features_sha256:
            raise Phase1IntegrationError("chemical feature checksum does not match manifest")
        cache = np.load(self.cache_path, mmap_mode="r")
        chemical = np.load(self.chemical_features_path, mmap_mode="r")
        if cache.dtype != np.float32 or cache.shape != self.cache_shape:
            raise Phase1IntegrationError("cache dtype or shape differs from manifest")
        if chemical.dtype != np.float32 or chemical.shape != self.chemical_features_shape:
            raise Phase1IntegrationError("chemical feature dtype or shape differs from manifest")
        return cache, chemical


def load_phase1_manifest(path: Path | str, *, root: Path | str | None = None) -> Phase1Manifest:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Phase1IntegrationError(f"manifest is unreadable: {manifest_path}") from error
    if payload.get("format") not in SUPPORTED_MANIFEST_FORMATS:
        raise Phase1IntegrationError(
            f"manifest format must be one of {sorted(SUPPORTED_MANIFEST_FORMATS)}"
        )
    if payload.get("gene_count") != GENE_COUNT:
        raise Phase1IntegrationError("Phase-1 target must contain exactly 978 genes")
    base = Path(root) if root is not None else manifest_path.parent
    cache = payload.get("cache")
    features = payload.get("chemical_features")
    records_payload = payload.get("records")
    if not isinstance(cache, Mapping) or not isinstance(features, Mapping) or not isinstance(records_payload, list) or not records_payload:
        raise Phase1IntegrationError("manifest requires cache, chemical_features, and records")
    cache_path = base / _required_text(cache.get("relative_path"), "cache.relative_path")
    chemical_path = base / _required_text(features.get("relative_path"), "chemical_features.relative_path")
    cache_sha256 = _required_text(cache.get("sha256"), "cache.sha256")
    chemical_sha256 = _required_text(features.get("sha256"), "chemical_features.sha256")
    control_policy = payload.get("control_policy", "strict_no_cross_split")
    if control_policy not in {"strict_no_cross_split", "pre_treatment_context_feature"}:
        raise Phase1IntegrationError("unsupported control_policy")
    cache_shape_raw = cache.get("shape")
    feature_shape_raw = features.get("shape")
    if not isinstance(cache_shape_raw, list) or len(cache_shape_raw) != 2 or cache_shape_raw[1] != GENE_COUNT:
        raise Phase1IntegrationError("cache.shape must be [rows, 978]")
    if not isinstance(feature_shape_raw, list) or len(feature_shape_raw) != 2 or any(int(v) < 1 for v in feature_shape_raw):
        raise Phase1IntegrationError("chemical_features.shape must contain two positive integers")
    records = tuple(Phase1Record.from_mapping(row) for row in records_payload if isinstance(row, Mapping))
    if len(records) != len(records_payload) or len({row.sample_id for row in records}) != len(records):
        raise Phase1IntegrationError("record identities must be unique mappings")
    group_splits: dict[str, str] = {}
    control_splits: dict[int, str] = {}
    for row in records:
        previous = group_splits.setdefault(row.treatment_group_id, row.split)
        if previous != row.split:
            raise Phase1IntegrationError("treatment groups cannot cross splits")
        if control_policy == "strict_no_cross_split":
            previous_control = control_splits.setdefault(row.control_cache_row, row.split)
            if previous_control != row.split:
                raise Phase1IntegrationError("control row cannot be shared across splits")
        if row.treatment_cache_row == row.control_cache_row:
            raise Phase1IntegrationError("treatment and control rows must differ")
        if row.treatment_cache_row >= int(cache_shape_raw[0]) or row.control_cache_row >= int(cache_shape_raw[0]):
            raise Phase1IntegrationError("cache row is outside declared shape")
        if row.chemical_feature_row >= int(feature_shape_raw[0]):
            raise Phase1IntegrationError("chemical feature row is outside declared shape")
    if not any(row.split == "train" for row in records) or not any(row.split == "test" for row in records):
        raise Phase1IntegrationError("manifest requires train and test records")
    return Phase1Manifest(
        path=manifest_path,
        cache_path=cache_path,
        cache_sha256=cache_sha256,
        chemical_features_path=chemical_path,
        chemical_features_sha256=chemical_sha256,
        cache_shape=(int(cache_shape_raw[0]), int(cache_shape_raw[1])),
        chemical_features_shape=(int(feature_shape_raw[0]), int(feature_shape_raw[1])),
        records=records,
    )


class ContextChemicalDoseTimePredictor(nn.Module):
    """Low-capacity concat+MLP interaction model with a gene-specific Δ978 head."""

    def __init__(self, chemical_dim: int, context_dim: int = GENE_COUNT, hidden_dim: int = 64, gene_count: int = GENE_COUNT):
        super().__init__()
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.chemical_projection = nn.Linear(chemical_dim, hidden_dim)
        self.dose_time_projection = nn.Linear(2, hidden_dim)
        self.interaction = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU())
        self.output = nn.Linear(hidden_dim, gene_count)

    def forward(self, context: torch.Tensor, chemical: torch.Tensor, dose_time: torch.Tensor) -> torch.Tensor:
        context_z = self.context_projection(context)
        chemical_z = self.chemical_projection(chemical)
        dose_time_z = self.dose_time_projection(dose_time)
        return self.output(self.interaction(torch.cat([context_z, chemical_z, dose_time_z], dim=-1)))


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    residual = observed - predicted
    observed_flat = observed.reshape(-1)
    predicted_flat = predicted.reshape(-1)
    pearson = None
    if observed_flat.size > 1 and np.std(observed_flat) > 0 and np.std(predicted_flat) > 0:
        pearson = float(np.corrcoef(observed_flat, predicted_flat)[0, 1])
    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty(values.size, dtype=np.float64)
        start = 0
        while start < values.size:
            stop = start + 1
            while stop < values.size and values[order[stop]] == values[order[start]]:
                stop += 1
            result[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
            start = stop
        return result
    ranked_observed = rank(observed_flat)
    ranked_predicted = rank(predicted_flat)
    spearman = None
    if np.std(ranked_observed) > 0 and np.std(ranked_predicted) > 0:
        spearman = float(np.corrcoef(ranked_observed, ranked_predicted)[0, 1])
    return {
        "pearson": pearson,
        "spearman": spearman,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "direction_accuracy": float(np.mean((observed_flat[observed_flat != 0] > 0) == (predicted_flat[observed_flat != 0] > 0)))
        if np.any(observed_flat != 0)
        else None,
    }


def _group_macro_metrics(
    rows: Sequence[Phase1Record], observed: np.ndarray, predicted: np.ndarray
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(row.treatment_group_id, []).append(index)
    metrics = [_metrics(np.mean(observed[indexes], axis=0), np.mean(predicted[indexes], axis=0)) for indexes in grouped.values()]
    summary: dict[str, float | int | None] = {"group_count": len(metrics)}
    for name in ("pearson", "spearman", "rmse", "mae", "direction_accuracy"):
        values = [float(item[name]) for item in metrics if item[name] is not None]
        summary[name] = float(np.mean(values)) if values else None
    return summary


def _bounded_records(records: Sequence[Phase1Record], max_records: int | None) -> tuple[Phase1Record, ...]:
    """Select complete groups while retaining train and test support."""
    if max_records is None:
        return tuple(records)
    if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 2:
        raise ValueError("max_records must be at least 2 or null")
    groups_by_split: dict[str, list[list[Phase1Record]]] = {"train": [], "validation": [], "test": []}
    current: dict[tuple[str, str], list[Phase1Record]] = {}
    for row in records:
        current.setdefault((row.split, row.treatment_group_id), []).append(row)
    for (split, _), group_rows in current.items():
        groups_by_split[split].append(group_rows)
    train_quota = max(1, int(max_records * 0.7))
    validation_quota = int(max_records * 0.1)
    test_quota = max(1, max_records - train_quota - validation_quota)
    selected: list[Phase1Record] = []
    for split, quota in (("train", train_quota), ("validation", validation_quota), ("test", test_quota)):
        used = 0
        for group_rows in groups_by_split[split]:
            if used and used + len(group_rows) > quota:
                break
            selected.extend(group_rows)
            used += len(group_rows)
    if not any(row.split == "train" for row in selected) or not any(row.split == "test" for row in selected):
        raise Phase1IntegrationError("bounded subset must retain train and test records")
    return tuple(selected)


def run_tiny(
    manifest_path: Path | str,
    *,
    root: Path | str | None = None,
    hidden_dim: int = 64,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    seed: int = 20260812,
    max_records: int | None = 2048,
    checkpoint_path: Path | str | None = None,
) -> dict[str, Any]:
    manifest = load_phase1_manifest(manifest_path, root=root)
    cache, chemical_features = manifest.load_arrays()
    records = _bounded_records(manifest.records, max_records)
    train = [row for row in records if row.split == "train"]
    test = [row for row in records if row.split == "test"]
    if not train or not test:
        raise Phase1IntegrationError("Tiny requires non-empty train and test records")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    def arrays(rows: Sequence[Phase1Record]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        context = np.asarray([cache[row.control_cache_row] for row in rows], dtype=np.float32)
        chemical = np.asarray([chemical_features[row.chemical_feature_row] for row in rows], dtype=np.float32)
        dose_time = np.asarray([[row.dose_um, row.time_h] for row in rows], dtype=np.float32)
        targets = np.asarray([cache[row.treatment_cache_row] - cache[row.control_cache_row] for row in rows], dtype=np.float32)
        return context, chemical, dose_time, targets
    train_context, train_chemical, train_dose_time, train_targets = arrays(train)
    test_context, test_chemical, test_dose_time, test_targets = arrays(test)
    context_center = train_context.mean(axis=0, dtype=np.float64).astype(np.float32)
    context_scale = train_context.std(axis=0, dtype=np.float64).astype(np.float32)
    context_scale[context_scale < 1e-6] = 1.0
    train_context_norm = (train_context - context_center) / context_scale
    test_context_norm = (test_context - context_center) / context_scale
    dose_time_raw = np.log1p(np.maximum(train_dose_time, 0.0)).astype(np.float32)
    test_dose_time_raw = np.log1p(np.maximum(test_dose_time, 0.0)).astype(np.float32)
    dose_time_center = dose_time_raw.mean(axis=0, dtype=np.float64).astype(np.float32)
    dose_time_scale = dose_time_raw.std(axis=0, dtype=np.float64).astype(np.float32)
    dose_time_scale[dose_time_scale < 1e-6] = 1.0
    train_dose_time_norm = (dose_time_raw - dose_time_center) / dose_time_scale
    test_dose_time_norm = (test_dose_time_raw - dose_time_center) / dose_time_scale
    target_center = train_targets.mean(axis=0, dtype=np.float64).astype(np.float32)
    target_scale = train_targets.std(axis=0, dtype=np.float64).astype(np.float32)
    target_scale[target_scale < 1e-6] = 1.0
    train_targets_norm = (train_targets - target_center) / target_scale
    context_tensor = torch.from_numpy(train_context_norm)
    chemical_tensor = torch.from_numpy(train_chemical)
    dose_time_tensor = torch.from_numpy(train_dose_time_norm)
    target_tensor = torch.from_numpy(train_targets_norm)
    test_context_tensor = torch.from_numpy(test_context_norm)
    test_chemical_tensor = torch.from_numpy(test_chemical)
    test_dose_time_tensor = torch.from_numpy(test_dose_time_norm)

    def fit_model(include_context: bool) -> dict[str, Any]:
        torch.manual_seed(seed)
        model = ContextChemicalDoseTimePredictor(
            chemical_dim=chemical_features.shape[1], context_dim=GENE_COUNT, hidden_dim=hidden_dim, gene_count=GENE_COUNT
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        fit_context = context_tensor if include_context else torch.zeros_like(context_tensor)
        eval_context = test_context_tensor if include_context else torch.zeros_like(test_context_tensor)
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(fit_context, chemical_tensor, dose_time_tensor) - target_tensor) ** 2)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_pred_norm = model(fit_context, chemical_tensor, dose_time_tensor).numpy()
            test_pred_norm = model(eval_context, test_chemical_tensor, test_dose_time_tensor).numpy()
        train_pred = train_pred_norm * target_scale + target_center
        test_pred = test_pred_norm * target_scale + target_center
        if include_context and checkpoint_path is not None:
            checkpoint = {
                "format": "phase1_context_candidate_checkpoint_v1",
                "model_state_dict": model.state_dict(),
                "chemical_dim": int(chemical_features.shape[1]),
                "context_dim": GENE_COUNT,
                "hidden_dim": hidden_dim,
                "gene_count": GENE_COUNT,
                "seed": seed,
                "manifest": str(manifest.path),
                "manifest_sha256": _digest(manifest.path),
                "chemical_features_sha256": manifest.chemical_features_sha256,
                "normalization": {
                    "context_center": context_center,
                    "context_scale": context_scale,
                    "dose_time_center": dose_time_center,
                    "dose_time_scale": dose_time_scale,
                    "target_center": target_center,
                    "target_scale": target_scale,
                },
            }
            checkpoint_file = Path(checkpoint_path)
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, checkpoint_file)
        return {
            "train_metrics": _metrics(train_targets, train_pred),
            "test_metrics": _metrics(test_targets, test_pred),
            "group_metrics": _group_macro_metrics(test, test_targets, test_pred),
            "prediction_variance": float(np.var(test_pred)),
        }

    models = {
        "chemical_only": fit_model(False),
        "chemical_context": fit_model(True),
    }
    global_prediction = np.broadcast_to(train_targets.mean(axis=0, dtype=np.float64), test_targets.shape).astype(np.float32)
    models["global_mean"] = {
        "train_metrics": _metrics(train_targets, np.broadcast_to(train_targets.mean(axis=0), train_targets.shape)),
        "test_metrics": _metrics(test_targets, global_prediction),
        "group_metrics": _group_macro_metrics(test, test_targets, global_prediction),
        "prediction_variance": 0.0,
    }
    context_result = models["chemical_context"]
    return {
        "format": "phase1_tiny_summary_v1",
        "status": "TINY_COMPLETE",
        "gene_count": GENE_COUNT,
        "train_rows": len(train),
        "test_rows": len(test),
        "target_variance": float(np.var(train_targets)),
        "context_variance": float(np.var(train_context)),
        "normalization": {
            "fit_scope": "train_only",
            "context": "train_mean_std",
            "dose_time": "log1p_then_train_mean_std",
            "target": "train_mean_std",
        },
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "train_metrics": context_result["train_metrics"],
        "test_metrics": context_result["test_metrics"],
        "prediction_variance": context_result["prediction_variance"],
        "models": models,
    }
