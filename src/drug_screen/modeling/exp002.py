"""Frozen dataloader, simple baselines, and evaluation wiring for EXP-002."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from drug_screen.evaluation.protocol import (
    bootstrap_macro_mean,
    compute_vector_metrics,
    paired_group_bootstrap_difference,
)


EXPERIMENT_ID = "EXP-002"
DATASET_ID = "lincs_gse92742_raw_level3_v1"
DERIVED_STORAGE_ID = "lincs_gse92742_level3_level4_level5"
CONTRACT_ARTIFACT_SHA256 = (
    "900592f841d736f3087077936b86c6f78ceb17f9a79f4485e619561d63e2ee9c"
)
ORDERED_GENE_IDS_SHA256 = (
    "b4e2fca877c5cfdcc1c712ad0fd67e97a88b6f7566b013e4bab065f699ebb623"
)
GENE_COUNT = 978
SPLIT_SCHEMES = ("cold_drug", "cold_context")
SPLITS = ("train", "validation", "test")
METRICS = ("pearson", "spearman", "rmse", "mae", "direction_accuracy")


def _stream_digest(values: Sequence[str]) -> str:
    digest = sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a canonical non-empty string")
    return value


@dataclass(frozen=True)
class Delta978Record:
    """One materialized treatment target under the frozen output contract."""

    experiment_id: str
    split_scheme: str
    split: str
    treatment_inst_id: str
    drug_id: str
    context_id: str
    replicate_family_id: str
    rna_plate: str
    dose: tuple[str, str]
    time: tuple[str, str]
    control_inst_ids: tuple[str, ...]
    ordered_gene_ids_sha256: str
    delta978: np.ndarray

    def __post_init__(self) -> None:
        if self.experiment_id != EXPERIMENT_ID:
            raise ValueError(f"experiment_id must be {EXPERIMENT_ID}")
        if self.split_scheme not in SPLIT_SCHEMES:
            raise ValueError(f"unsupported split_scheme: {self.split_scheme}")
        if self.split not in SPLITS:
            raise ValueError(f"unsupported split: {self.split}")
        for field in (
            "treatment_inst_id",
            "drug_id",
            "context_id",
            "replicate_family_id",
            "rna_plate",
        ):
            _text(getattr(self, field), field)
        if len(self.dose) != 2 or len(self.time) != 2:
            raise ValueError("dose and time must each contain value and unit")
        if not self.control_inst_ids or len(set(self.control_inst_ids)) != len(
            self.control_inst_ids
        ):
            raise ValueError("control_inst_ids must be non-empty and unique")
        if self.ordered_gene_ids_sha256 != ORDERED_GENE_IDS_SHA256:
            raise ValueError("record gene identity/order digest is not frozen exact-978")
        values = np.asarray(self.delta978)
        if values.shape != (GENE_COUNT,) or values.dtype != np.float32:
            raise ValueError("delta978 must have dtype float32 and shape (978,)")
        if not np.isfinite(values).all():
            raise ValueError("delta978 must contain only finite values")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Delta978Record":
        required = {
            "experiment_id",
            "split_scheme",
            "split",
            "treatment_inst_id",
            "drug_id",
            "context_id",
            "replicate_family_id",
            "rna_plate",
            "dose",
            "time",
            "control_inst_ids",
            "ordered_gene_ids_sha256",
            "delta978",
        }
        missing = required.difference(value)
        if missing:
            raise ValueError(f"materialized record is missing fields: {sorted(missing)}")
        return cls(
            experiment_id=value["experiment_id"],
            split_scheme=value["split_scheme"],
            split=value["split"],
            treatment_inst_id=value["treatment_inst_id"],
            drug_id=value["drug_id"],
            context_id=value["context_id"],
            replicate_family_id=value["replicate_family_id"],
            rna_plate=value["rna_plate"],
            dose=tuple(value["dose"]),
            time=tuple(value["time"]),
            control_inst_ids=tuple(value["control_inst_ids"]),
            ordered_gene_ids_sha256=value["ordered_gene_ids_sha256"],
            delta978=np.asarray(value["delta978"], dtype=np.float32),
        )


class Delta978Dataset(Dataset[dict[str, Any]]):
    """Validated in-memory view of a materialized JSONL split.

    The Data Steward owns creation of the full manifest. This reader deliberately
    refuses raw GCTX input so modeling cannot silently redefine matching or splits.
    """

    def __init__(self, records: Sequence[Delta978Record], gene_ids: Sequence[str]):
        self.records = tuple(records)
        self.gene_ids = tuple(gene_ids)
        if len(self.gene_ids) != GENE_COUNT or len(set(self.gene_ids)) != GENE_COUNT:
            raise ValueError("genes.json must contain 978 unique gene IDs")
        if _stream_digest(self.gene_ids) != ORDERED_GENE_IDS_SHA256:
            raise ValueError("genes.json does not match the frozen exact-978 order")
        if not self.records:
            raise ValueError("materialized split cannot be empty")
        self._validate_records()

    def _validate_records(self) -> None:
        treatment_ids: set[str] = set()
        schemes = {record.split_scheme for record in self.records}
        splits = {record.split for record in self.records}
        if len(schemes) != 1 or len(splits) != 1:
            raise ValueError("one dataset file must contain exactly one scheme and split")
        for record in self.records:
            if record.treatment_inst_id in treatment_ids:
                raise ValueError("treatment_inst_id values must be unique within a split")
            treatment_ids.add(record.treatment_inst_id)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "treatment_inst_id": record.treatment_inst_id,
            "drug_id": record.drug_id,
            "context_id": record.context_id,
            "replicate_family_id": record.replicate_family_id,
            "delta978": torch.from_numpy(record.delta978.copy()),
        }

    @classmethod
    def from_directory(
        cls, root: Path | str, split_scheme: str, split: str
    ) -> "Delta978Dataset":
        if split_scheme not in SPLIT_SCHEMES or split not in SPLITS:
            raise ValueError("unsupported split scheme or split")
        root = Path(root)
        genes = json.loads((root / "genes.json").read_text(encoding="utf-8"))
        path = root / split_scheme / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(
                f"materialized EXP-002 split is absent: {path}; raw GCTX is not a model input"
            )
        with path.open("rt", encoding="utf-8") as handle:
            records = [
                Delta978Record.from_mapping(json.loads(line))
                for line in handle
                if line.strip()
            ]
        return cls(records, genes)


def validate_materialization(root: Path | str) -> dict[str, Any]:
    """Verify small provenance metadata and every declared materialized file hash."""
    root = Path(root)
    path = root / "materialization.json"
    if not path.is_file():
        raise FileNotFoundError(f"EXP-002 materialization metadata is absent: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "source_contract_artifact_sha256": CONTRACT_ARTIFACT_SHA256,
        "ordered_gene_ids_sha256": ORDERED_GENE_IDS_SHA256,
        "gene_count": GENE_COUNT,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"materialization changes frozen contract field: {key}")
    files = metadata.get("files")
    expected_files = {"genes.json"}.union(
        f"{scheme}/{split}.jsonl" for scheme in SPLIT_SCHEMES for split in SPLITS
    )
    if not isinstance(files, dict) or set(files) != expected_files:
        raise ValueError("materialization metadata must hash every required split file")
    for relative, expected_digest in files.items():
        file_path = root / relative
        if not file_path.is_file():
            raise FileNotFoundError(f"declared materialized file is absent: {file_path}")
        digest = _file_digest(file_path)
        if digest != expected_digest:
            raise ValueError(f"materialized file digest mismatch: {relative}")
    return metadata


def assert_manifest_isolation(
    datasets: Sequence[Delta978Dataset], split_scheme: str
) -> None:
    """Reject treatment, family, raw-control, and frozen cold-axis leakage."""
    if split_scheme not in SPLIT_SCHEMES:
        raise ValueError(f"unsupported split_scheme: {split_scheme}")
    observed: dict[str, dict[str, str]] = {
        "treatment": {},
        "family": {},
        "control": {},
        "cold_axis": {},
    }
    for dataset in datasets:
        for record in dataset.records:
            if record.split_scheme != split_scheme:
                raise ValueError("dataset split scheme does not match evaluation stratum")
            identities = {
                "treatment": (record.treatment_inst_id,),
                "family": (record.replicate_family_id,),
                "control": record.control_inst_ids,
                "cold_axis": (
                    record.drug_id if split_scheme == "cold_drug" else record.context_id,
                ),
            }
            for kind, values in identities.items():
                for identity in values:
                    previous = observed[kind].setdefault(identity, record.split)
                    if previous != record.split:
                        raise ValueError(
                            f"{kind} leakage for {identity}: {previous} vs {record.split}"
                        )


def create_dataloader(
    dataset: Delta978Dataset, *, batch_size: int, shuffle: bool, seed: int
) -> DataLoader[dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


class GlobalMeanBaseline:
    """Predict the per-gene arithmetic mean fitted on training targets only."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None

    def fit(self, records: Iterable[Delta978Record]) -> "GlobalMeanBaseline":
        total = np.zeros(GENE_COUNT, dtype=np.float64)
        count = 0
        for record in records:
            if record.split == "train":
                total += record.delta978
                count += 1
        if count == 0:
            raise ValueError("global baseline requires training records")
        self.mean_ = (total / count).astype(np.float32)
        return self

    def predict(self, records: Sequence[Delta978Record]) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("global baseline is not fitted")
        return np.repeat(self.mean_[None, :], len(records), axis=0)


class ContextConditionedMeanBaseline:
    """Predict train context means with a frozen global-mean unseen fallback."""

    fallback = "global_train_mean"

    def __init__(self) -> None:
        self.global_mean_: np.ndarray | None = None
        self.context_means_: dict[str, np.ndarray] = {}

    def fit(self, records: Iterable[Delta978Record]) -> "ContextConditionedMeanBaseline":
        global_total = np.zeros(GENE_COUNT, dtype=np.float64)
        global_count = 0
        context_totals: dict[str, np.ndarray] = {}
        context_counts: dict[str, int] = defaultdict(int)
        for record in records:
            if record.split != "train":
                continue
            global_total += record.delta978
            global_count += 1
            context_totals.setdefault(
                record.context_id, np.zeros(GENE_COUNT, dtype=np.float64)
            )
            context_totals[record.context_id] += record.delta978
            context_counts[record.context_id] += 1
        if global_count == 0:
            raise ValueError("context baseline requires training records")
        self.global_mean_ = (global_total / global_count).astype(np.float32)
        self.context_means_ = {
            context: (total / context_counts[context]).astype(np.float32)
            for context, total in context_totals.items()
        }
        return self

    def predict(self, records: Sequence[Delta978Record]) -> tuple[np.ndarray, np.ndarray]:
        if self.global_mean_ is None:
            raise RuntimeError("context baseline is not fitted")
        fallback = np.asarray(
            [record.context_id not in self.context_means_ for record in records], dtype=bool
        )
        predictions = np.stack(
            [
                self.context_means_.get(record.context_id, self.global_mean_)
                for record in records
            ]
        )
        return predictions, fallback


def _validate_config(config: Mapping[str, Any]) -> None:
    frozen = config.get("frozen_contract", {})
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "dataset_registry_id": DATASET_ID,
        "derived_storage_registry_id": DERIVED_STORAGE_ID,
        "contract_artifact_sha256": CONTRACT_ARTIFACT_SHA256,
        "ordered_gene_ids_sha256": ORDERED_GENE_IDS_SHA256,
        "gene_count": GENE_COUNT,
        "split_schemes": list(SPLIT_SCHEMES),
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise ValueError(f"config changes frozen contract field: {key}")
    evaluation = config.get("evaluation", {})
    if evaluation.get("metrics") != list(METRICS):
        raise ValueError("config must retain all pre-registered metrics in frozen order")
    if evaluation.get("primary_metric") != "pearson":
        raise ValueError("primary_metric must be pre-registered Pearson")
    if evaluation.get("direction_zero_epsilon") != 0.0:
        raise ValueError("direction_zero_epsilon is frozen at 0.0")
    if evaluation.get("unseen_context_fallback") != "global_train_mean":
        raise ValueError("unseen context fallback must be global_train_mean")
    if evaluation.get("replicate_aggregation") != "arithmetic_mean":
        raise ValueError("replicate aggregation must be arithmetic_mean")
    if evaluation.get("bootstrap_resamples", 0) < 2:
        raise ValueError("bootstrap_resamples must be at least 2")
    if isinstance(evaluation.get("bootstrap_seed"), bool) or not isinstance(
        evaluation.get("bootstrap_seed"), int
    ):
        raise ValueError("bootstrap_seed must be an integer")
    if evaluation.get("paired_improvement_threshold") != 0.0:
        raise ValueError("paired improvement threshold is pre-registered at 0.0")
    if evaluation.get("seed_consistency_rule") != "all_seeds_positive_paired_improvement":
        raise ValueError("seed consistency rule does not match pre-registration")
    if evaluation.get("minimum_eligible_group_count", 0) < 2:
        raise ValueError("minimum_eligible_group_count must be at least 2")
    seeds = config.get("reproducibility", {}).get("seeds", [])
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("at least two unique seeds are required")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("seeds must be integers")


def load_exp002_config(path: Path | str) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_config(config)
    return config


def _group_records(
    records: Sequence[Delta978Record], predictions: np.ndarray
) -> list[tuple[str, str, np.ndarray, np.ndarray, int]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(record.drug_id, record.context_id)].append(index)
    result = []
    for (drug_id, context_id), indices in sorted(grouped.items()):
        observed = np.mean(
            np.stack([records[index].delta978 for index in indices]), axis=0
        )
        predicted = np.mean(predictions[indices], axis=0)
        result.append((drug_id, context_id, observed, predicted, len(indices)))
    return result


def _metric_mapping(metrics: object) -> dict[str, float]:
    return {name: float(getattr(metrics, name)) for name in METRICS}


def _compute_metrics_preserving_defined_errors(
    observed: np.ndarray, predicted: np.ndarray, gene_ids: Sequence[str]
) -> tuple[dict[str, float | None], tuple[str, ...]]:
    try:
        metrics = compute_vector_metrics(
            observed.tolist(),
            predicted.tolist(),
            observed_gene_ids=gene_ids,
            predicted_gene_ids=gene_ids,
            expected_gene_ids=gene_ids,
        )
        return _metric_mapping(metrics), ()
    except ValueError as error:
        if "correlation undefined" not in str(error):
            raise
    errors = observed - predicted
    eligible = observed != 0.0
    direction = (
        float(np.mean((observed[eligible] > 0) == (predicted[eligible] > 0)))
        if eligible.any()
        else None
    )
    undefined = ["pearson", "spearman"]
    if direction is None:
        undefined.append("direction_accuracy")
    return (
        {
            "pearson": None,
            "spearman": None,
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mae": float(np.mean(np.abs(errors))),
            "direction_accuracy": direction,
        },
        tuple(undefined),
    )


def _interval_mapping(interval: object) -> dict[str, float | int]:
    return {
        "point_estimate": float(getattr(interval, "point_estimate")),
        "low": float(getattr(interval, "low")),
        "high": float(getattr(interval, "high")),
        "seed": int(getattr(interval, "seed")),
        "resamples": int(getattr(interval, "resamples")),
    }


def run_evaluation(
    config: Mapping[str, Any], materialized_root: Path | str
) -> dict[str, Any]:
    """Fit both train-only baselines and score test groups for both OOD schemes."""
    _validate_config(config)
    root = Path(materialized_root)
    validate_materialization(root)
    gene_ids = tuple(json.loads((root / "genes.json").read_text(encoding="utf-8")))
    results: dict[str, Any] = {}
    seeds = config["reproducibility"]["seeds"]
    for scheme in SPLIT_SCHEMES:
        train = Delta978Dataset.from_directory(root, scheme, "train")
        validation = Delta978Dataset.from_directory(root, scheme, "validation")
        test = Delta978Dataset.from_directory(root, scheme, "test")
        assert_manifest_isolation((train, validation, test), scheme)
        global_model = GlobalMeanBaseline().fit(train.records)
        context_model = ContextConditionedMeanBaseline().fit(train.records)
        global_predictions = global_model.predict(test.records)
        context_predictions, fallback = context_model.predict(test.records)
        method_predictions = {
            "global_train_mean": global_predictions,
            "context_train_mean": context_predictions,
        }
        scheme_rows = []
        scheme_summaries = []
        undefined_counts = {method: {metric: 0 for metric in METRICS} for method in method_predictions}
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            for method, predictions in method_predictions.items():
                for drug_id, context_id, observed, predicted, replicate_count in _group_records(
                    test.records, predictions
                ):
                    values, undefined = _compute_metrics_preserving_defined_errors(
                        observed, predicted, gene_ids
                    )
                    for metric in undefined:
                        undefined_counts[method][metric] += 1
                    scheme_rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "drug_id": drug_id,
                            "context_id": context_id,
                            "replicate_count": replicate_count,
                            "metrics": values,
                        }
                    )
            seed_rows = [row for row in scheme_rows if row["seed"] == seed]
            bootstrap_seed = config["evaluation"]["bootstrap_seed"] + seed
            resamples = config["evaluation"]["bootstrap_resamples"]
            for metric in METRICS:
                by_method: dict[str, list[dict[str, Any]]] = {
                    method: [
                        row
                        for row in seed_rows
                        if row["method"] == method and row["metrics"][metric] is not None
                    ]
                    for method in method_predictions
                }
                for method, rows in by_method.items():
                    interval = (
                        bootstrap_macro_mean(
                            [row["metrics"][metric] for row in rows],
                            seed=bootstrap_seed,
                            resamples=resamples,
                        )
                        if rows
                        else None
                    )
                    scheme_summaries.append(
                        {
                            "seed": seed,
                            "metric": metric,
                            "method": method,
                            "eligible_group_count": len(rows),
                            "macro_bootstrap": _interval_mapping(interval)
                            if interval is not None
                            else None,
                        }
                    )
                global_by_group = {
                    (row["drug_id"], row["context_id"]): row
                    for row in by_method["global_train_mean"]
                }
                context_by_group = {
                    (row["drug_id"], row["context_id"]): row
                    for row in by_method["context_train_mean"]
                }
                paired_keys = sorted(set(global_by_group).intersection(context_by_group))
                group_ids = [
                    key[0] if scheme == "cold_drug" else key[1] for key in paired_keys
                ]
                interval = None
                if len(set(group_ids)) >= 2:
                    interval = paired_group_bootstrap_difference(
                        [context_by_group[key]["metrics"][metric] for key in paired_keys],
                        [global_by_group[key]["metrics"][metric] for key in paired_keys],
                        group_ids,
                        higher_is_better=metric
                        in {"pearson", "spearman", "direction_accuracy"},
                        seed=bootstrap_seed,
                        resamples=resamples,
                    )
                scheme_summaries.append(
                    {
                        "seed": seed,
                        "metric": metric,
                        "comparison": "context_train_mean_minus_global_train_mean_improvement",
                        "paired_group_count": len(set(group_ids)),
                        "paired_bootstrap": _interval_mapping(interval)
                        if interval is not None
                        else None,
                    }
                )
        results[scheme] = {
            "eligible_treatment_count": len(test.records),
            "eligible_group_count": len({(r.drug_id, r.context_id) for r in test.records}),
            "context_fallback_count": int(fallback.sum()),
            "dropped_count": 0,
            "undefined_metric_counts": undefined_counts,
            "rows": scheme_rows,
            "summaries": scheme_summaries,
        }
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "EVALUATION_COMPLETE",
        "config_sha256": sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "frozen_contract": config["frozen_contract"],
        "results": results,
    }
