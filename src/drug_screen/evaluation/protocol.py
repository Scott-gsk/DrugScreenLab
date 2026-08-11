"""Reusable assertions and metrics for pre-registered perturbation evaluation."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

CORE_GENE_COUNT = 978
DIRECTION_ZERO_EPSILON = 0.0
_VALID_SPLITS = frozenset({"train", "validation", "test"})
_VALID_STRATA = frozenset({"cold_drug", "cold_context"})
_VALID_METRICS = frozenset(
    {"pearson", "spearman", "rmse", "mae", "direction_accuracy"}
)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _canonical_gene_ids(values: Sequence[str], field: str) -> list[str]:
    identifiers = list(values)
    normalized = [_text(identifier, field) for identifier in identifiers]
    if identifiers != normalized:
        raise ValueError(f"{field} must contain canonical trimmed identifiers")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{field} must contain unique identifiers")
    return identifiers


@dataclass(frozen=True)
class EvaluationRecord:
    """Identity-only view used to verify a frozen split manifest.

    ``context_id`` is the Data Contract's canonical context identity.  It is not
    reconstructed by evaluation code, preventing silent changes to its fields.
    """

    sample_id: str
    compound_id: str | None
    context_id: str
    control_id: str | None
    replicate_family_id: str
    is_control: bool
    control_match_key: str

    def __post_init__(self) -> None:
        _text(self.sample_id, "sample_id")
        _text(self.context_id, "context_id")
        _text(self.replicate_family_id, "replicate_family_id")
        _text(self.control_match_key, "control_match_key")
        if not self.is_control:
            _text(self.compound_id, "compound_id")
            _text(self.control_id, "control_id")


def assert_cold_split_isolation(
    records: Iterable[EvaluationRecord], assignments: Mapping[str, str], stratum: str
) -> None:
    """Assert pair, replicate-family, and requested cold-axis isolation.

    Cold-drug isolates canonical compounds across all three splits.  Cold-context
    isolates canonical contexts.  Both variants always isolate matched controls
    and replicate families, but intentionally do not impose the other axis.
    """
    if stratum not in _VALID_STRATA:
        raise ValueError(f"unsupported OOD stratum: {stratum}")
    values = list(records)
    by_id = {record.sample_id: record for record in values}
    if len(by_id) != len(values):
        raise ValueError("sample_id values must be unique")
    if not values:
        raise ValueError("evaluation manifest cannot be empty")

    family_splits: dict[str, str] = {}
    cold_axis_splits: dict[str, str] = {}
    treatment_splits: set[str] = set()
    for record in values:
        split = _text(assignments.get(record.sample_id), "split")
        if split not in _VALID_SPLITS:
            raise ValueError(f"unsupported split: {split}")
        previous_family = family_splits.setdefault(record.replicate_family_id, split)
        if previous_family != split:
            raise ValueError(
                f"replicate-family leakage for {record.replicate_family_id}: "
                f"{previous_family} vs {split}"
            )
        if record.is_control:
            continue

        treatment_splits.add(split)
        control = by_id.get(record.control_id or "")
        if control is None or not control.is_control:
            raise ValueError(f"treatment {record.sample_id} has no matched control")
        control_split = _text(assignments.get(control.sample_id), "split")
        if control_split != split:
            raise ValueError(
                f"matched-control leakage for {record.sample_id}: "
                f"{control_split} vs {split}"
            )
        if control.control_match_key != record.control_match_key:
            raise ValueError(
                f"matched-control key mismatch for {record.sample_id}: "
                f"{record.control_match_key} vs {control.control_match_key}"
            )

        axis = record.compound_id if stratum == "cold_drug" else record.context_id
        previous_axis = cold_axis_splits.setdefault(axis or "", split)
        if previous_axis != split:
            name = "compound" if stratum == "cold_drug" else "context"
            raise ValueError(f"{name} leakage for {axis}: {previous_axis} vs {split}")

    if "test" not in treatment_splits:
        raise ValueError(f"{stratum} manifest must contain treatment test examples")
    if not treatment_splits.intersection({"train", "validation"}):
        raise ValueError(f"{stratum} manifest must contain development treatment examples")


def assert_test_label_isolation(
    selection_sample_ids: Iterable[str],
    final_evaluation_sample_ids: Iterable[str],
    assignments: Mapping[str, str],
    *,
    configuration_locked: bool,
) -> None:
    """Ensure selection is development-only and final labels are opened after lock."""
    selection_ids = list(selection_sample_ids)
    evaluation_ids = list(final_evaluation_sample_ids)
    if not selection_ids:
        raise ValueError("selection sample IDs cannot be empty")
    if not evaluation_ids:
        raise ValueError("final evaluation sample IDs cannot be empty")
    if not configuration_locked:
        raise ValueError("configuration must be locked before test-label access")
    if set(selection_ids).intersection(evaluation_ids):
        raise ValueError("selection and final evaluation samples must be disjoint")
    for sample_id in selection_ids:
        split = _text(assignments.get(_text(sample_id, "selection sample_id")), "split")
        if split not in {"train", "validation"}:
            raise ValueError(f"test-label leakage during selection for {sample_id}")
    for sample_id in evaluation_ids:
        split = _text(assignments.get(_text(sample_id, "evaluation sample_id")), "split")
        if split != "test":
            raise ValueError(f"final evaluation must be test-only: {sample_id} is {split}")


@dataclass(frozen=True)
class VectorMetrics:
    pearson: float
    spearman: float
    rmse: float
    mae: float
    direction_accuracy: float


@dataclass(frozen=True)
class BootstrapInterval:
    point_estimate: float
    low: float
    high: float
    seed: int
    resamples: int


@dataclass(frozen=True)
class StratifiedMetricRecord:
    """One auditable result row retaining every pre-registered reporting axis."""

    evaluation_unit_id: str
    group_id: str
    drug_id: str
    context_id: str
    gene_stratum: str
    ood_stratum: str
    failure_stratum: str
    seed: int
    metric_name: str
    candidate_value: float
    baseline_value: float

    def __post_init__(self) -> None:
        for field in (
            "evaluation_unit_id",
            "group_id",
            "drug_id",
            "context_id",
            "gene_stratum",
            "failure_stratum",
        ):
            _text(getattr(self, field), field)
        if self.ood_stratum not in _VALID_STRATA:
            raise ValueError(f"unsupported OOD stratum: {self.ood_stratum}")
        if self.metric_name not in _VALID_METRICS:
            raise ValueError(f"unsupported metric: {self.metric_name}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not math.isfinite(self.candidate_value) or not math.isfinite(self.baseline_value):
            raise ValueError("stratified metric values must be finite")


def assert_stratified_reporting_complete(
    records: Iterable[StratifiedMetricRecord], *, required_seeds: Iterable[int]
) -> None:
    """Require separate OOD, metric, and multi-seed evidence before aggregation."""
    values = list(records)
    seeds = tuple(required_seeds)
    if not values:
        raise ValueError("stratified reporting records cannot be empty")
    if len(set(seeds)) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("required_seeds must contain at least two unique seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("required_seeds must contain integers")

    result_keys = [
        (
            record.evaluation_unit_id,
            record.ood_stratum,
            record.seed,
            record.metric_name,
        )
        for record in values
    ]
    if len(set(result_keys)) != len(result_keys):
        raise ValueError("stratified reporting rows must be unique")

    observed = {
        (record.ood_stratum, record.seed, record.metric_name) for record in values
    }
    required = {
        (stratum, seed, metric)
        for stratum in _VALID_STRATA
        for seed in seeds
        for metric in _VALID_METRICS
    }
    missing = required.difference(observed)
    if missing:
        raise ValueError(f"missing stratified metric cells: {sorted(missing)}")
    unexpected_seeds = {record.seed for record in values}.difference(seeds)
    if unexpected_seeds:
        raise ValueError(f"unregistered result seeds: {sorted(unexpected_seeds)}")


def _rank(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2
        for original_index, _ in ordered[index:end]:
            ranks[original_index] = average_rank
        index = end
    return ranks


def _pearson(observed: Sequence[float], predicted: Sequence[float]) -> float:
    observed_mean = sum(observed) / len(observed)
    predicted_mean = sum(predicted) / len(predicted)
    numerator = sum((x - observed_mean) * (y - predicted_mean) for x, y in zip(observed, predicted))
    observed_scale = math.sqrt(sum((x - observed_mean) ** 2 for x in observed))
    predicted_scale = math.sqrt(sum((y - predicted_mean) ** 2 for y in predicted))
    if observed_scale == 0 or predicted_scale == 0:
        raise ValueError("constant vectors make correlation undefined")
    return numerator / (observed_scale * predicted_scale)


def compute_vector_metrics(
    observed: Sequence[float],
    predicted: Sequence[float],
    *,
    observed_gene_ids: Sequence[str],
    predicted_gene_ids: Sequence[str],
    expected_gene_ids: Sequence[str],
    expected_gene_count: int = CORE_GENE_COUNT,
    direction_zero_epsilon: float = DIRECTION_ZERO_EPSILON,
) -> VectorMetrics:
    """Compute pre-registered metrics for one exact-landmark Delta978 vector."""
    if len(observed) != expected_gene_count or len(predicted) != expected_gene_count:
        raise ValueError(f"vectors must each contain exactly {expected_gene_count} genes")
    expected_ids = _canonical_gene_ids(expected_gene_ids, "expected_gene_ids")
    observed_ids = _canonical_gene_ids(observed_gene_ids, "observed_gene_ids")
    predicted_ids = _canonical_gene_ids(predicted_gene_ids, "predicted_gene_ids")
    if len(expected_ids) != expected_gene_count:
        raise ValueError(f"expected gene IDs must contain exactly {expected_gene_count} genes")
    if observed_ids != expected_ids:
        raise ValueError("observed gene identity/order does not match the frozen universe")
    if predicted_ids != expected_ids:
        raise ValueError("predicted gene identity/order does not match the frozen universe")
    if direction_zero_epsilon != DIRECTION_ZERO_EPSILON:
        raise ValueError("direction_zero_epsilon is frozen at 0.0")
    if not all(math.isfinite(value) for value in (*observed, *predicted)):
        raise ValueError("metrics require finite observed and predicted values")
    absolute_errors = [abs(x - y) for x, y in zip(observed, predicted)]
    eligible = [
        (x, y)
        for x, y in zip(observed, predicted)
        if abs(x) > direction_zero_epsilon
    ]
    if not eligible:
        raise ValueError("direction accuracy has no eligible non-zero targets")
    return VectorMetrics(
        pearson=_pearson(observed, predicted),
        spearman=_pearson(_rank(observed), _rank(predicted)),
        rmse=math.sqrt(sum(error**2 for error in absolute_errors) / expected_gene_count),
        mae=sum(absolute_errors) / expected_gene_count,
        direction_accuracy=sum((x > 0) == (y > 0) for x, y in eligible) / len(eligible),
    )


def bootstrap_macro_mean(
    values: Sequence[float], *, seed: int, resamples: int = 2_000
) -> BootstrapInterval:
    """Deterministically bootstrap a macro metric over independent evaluation units."""
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("bootstrap values must be non-empty and finite")
    if resamples < 2:
        raise ValueError("resamples must be at least 2")
    generator = random.Random(seed)
    count = len(values)
    samples = sorted(
        sum(values[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    )
    return BootstrapInterval(
        point_estimate=sum(values) / count,
        low=samples[round(0.025 * (resamples - 1))],
        high=samples[round(0.975 * (resamples - 1))],
        seed=seed,
        resamples=resamples,
    )


def paired_group_bootstrap_difference(
    candidate_values: Sequence[float],
    baseline_values: Sequence[float],
    group_ids: Sequence[str],
    *,
    higher_is_better: bool,
    seed: int,
    resamples: int = 2_000,
) -> BootstrapInterval:
    """Bootstrap paired candidate improvements by independent group ID."""
    if not (
        len(candidate_values) == len(baseline_values) == len(group_ids)
        and candidate_values
    ):
        raise ValueError("paired values and group IDs must have equal non-zero length")
    if not all(
        math.isfinite(value) for value in (*candidate_values, *baseline_values)
    ):
        raise ValueError("paired bootstrap values must be finite")
    if not isinstance(higher_is_better, bool):
        raise TypeError("higher_is_better must be boolean")
    if resamples < 2:
        raise ValueError("resamples must be at least 2")

    grouped: dict[str, list[float]] = {}
    for candidate, baseline, group_id in zip(
        candidate_values, baseline_values, group_ids
    ):
        group = _text(group_id, "group_id")
        difference = candidate - baseline if higher_is_better else baseline - candidate
        grouped.setdefault(group, []).append(difference)
    if len(grouped) < 2:
        raise ValueError("group bootstrap requires at least two independent groups")

    group_means = [sum(values) / len(values) for values in grouped.values()]
    generator = random.Random(seed)
    count = len(group_means)
    samples = sorted(
        sum(group_means[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    )
    return BootstrapInterval(
        point_estimate=sum(group_means) / count,
        low=samples[round(0.025 * (resamples - 1))],
        high=samples[round(0.975 * (resamples - 1))],
        seed=seed,
        resamples=resamples,
    )
