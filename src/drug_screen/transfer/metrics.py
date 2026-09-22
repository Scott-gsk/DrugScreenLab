"""Patient-equal ranking metrics and EXP-013 verdict rules."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from drug_screen.evaluation.protocol import BootstrapInterval, paired_group_bootstrap_difference

MIN_PAIRS = 3


def spearman(observed: Sequence[float], predicted: Sequence[float]) -> float:
    if len(observed) != len(predicted) or len(observed) < MIN_PAIRS:
        raise ValueError("spearman requires at least three paired finite values")
    if not all(math.isfinite(value) for value in (*observed, *predicted)):
        raise ValueError("spearman requires finite values")
    return _pearson(_rank(observed), _rank(predicted))


def ndcg_at_k(observed: Sequence[float], predicted: Sequence[float], *, k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    if len(observed) != len(predicted) or not observed:
        raise ValueError("ndcg requires aligned non-empty vectors")
    if len(observed) < k:
        return float("nan")
    gains = _positive_linear_gains(observed)
    ideal = sorted(range(len(observed)), key=lambda index: observed[index], reverse=True)
    ranking = sorted(range(len(predicted)), key=lambda index: predicted[index], reverse=True)
    return _dcg(gains, ranking, k) / _dcg(gains, ideal, k)


def _positive_linear_gains(observed: Sequence[float]) -> list[float]:
    minimum = min(observed)
    shifted = [value - minimum for value in observed]
    scale = max(shifted) if shifted else 0.0
    if scale == 0:
        return [0.0 for _ in observed]
    return [value / scale for value in shifted]


def _dcg(gains: Sequence[float], order: Sequence[int], k: int) -> float:
    total = 0.0
    for rank, index in enumerate(order[:k], start=1):
        total += gains[index] / math.log2(rank + 1)
    return total


def _rank(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average = (index + 1 + end) / 2
        for original, _ in ordered[index:end]:
            ranks[original] = average
        index = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale == 0 or right_scale == 0:
        raise ValueError("constant vectors make correlation undefined")
    return numerator / (left_scale * right_scale)


@dataclass(frozen=True)
class TransferVerdict:
    label: str
    primary_endpoint: str
    candidate_minus_baseline: float
    ci_low: float
    ci_high: float
    notes: tuple[str, ...]


def evaluate_predictions(
    patient_ids: Sequence[str],
    drug_ids: Sequence[str],
    observed: Sequence[float],
    predicted: Sequence[float],
    baseline: Sequence[float],
    *,
    min_drugs_per_patient: int,
    seed: int,
    ndcg_k: int = 5,
) -> dict[str, object]:
    if not (len(patient_ids) == len(drug_ids) == len(observed) == len(predicted) == len(baseline)):
        raise ValueError("prediction tables must be aligned")

    by_patient: dict[str, list[tuple[str, float, float, float]]] = {}
    by_drug: dict[str, list[tuple[str, float, float, float]]] = {}
    for patient, drug, y, yhat, y0 in zip(patient_ids, drug_ids, observed, predicted, baseline):
        by_patient.setdefault(patient, []).append((drug, y, yhat, y0))
        by_drug.setdefault(drug, []).append((patient, y, yhat, y0))

    within_rows = []
    for patient, rows in sorted(by_patient.items()):
        if len(rows) < min_drugs_per_patient:
            continue
        y = [item[1] for item in rows]
        yhat = [item[2] for item in rows]
        y0 = [item[3] for item in rows]
        candidate = _safe_spearman(y, yhat)
        control = _safe_spearman(y, y0)
        ndcg = ndcg_at_k(y, yhat, k=ndcg_k)
        ndcg0 = ndcg_at_k(y, y0, k=ndcg_k)
        within_rows.append(
            {
                "patient_id": patient,
                "n_drugs": len(rows),
                "spearman": candidate,
                "baseline_spearman": control,
                "ndcg": ndcg,
                "baseline_ndcg": ndcg0,
            }
        )

    per_drug_rows = []
    for drug, rows in sorted(by_drug.items()):
        if len(rows) < MIN_PAIRS:
            continue
        y = [item[1] for item in rows]
        yhat = [item[2] for item in rows]
        y0 = [item[3] for item in rows]
        per_drug_rows.append(
            {
                "drug_id": drug,
                "n_patients": len({item[0] for item in rows}),
                "spearman": _safe_spearman(y, yhat),
                "baseline_spearman": _safe_spearman(y, y0),
            }
        )

    within_delta = [
        row["spearman"] - row["baseline_spearman"] for row in within_rows
    ]
    per_drug_delta = [
        row["spearman"] - row["baseline_spearman"] for row in per_drug_rows
    ]
    within_interval = _optional_bootstrap(
        [row["spearman"] for row in within_rows],
        [row["baseline_spearman"] for row in within_rows],
        [row["patient_id"] for row in within_rows],
        seed,
    )
    per_drug_interval = _optional_bootstrap(
        [row["spearman"] for row in per_drug_rows],
        [row["baseline_spearman"] for row in per_drug_rows],
        [row["drug_id"] for row in per_drug_rows],
        seed + 1,
    )
    verdict = decide_verdict(
        within_interval,
        per_drug_interval,
        n_patients=len(within_rows),
        n_drugs=len(per_drug_rows),
    )
    return {
        "n_scored_patients": len(within_rows),
        "n_scored_drugs": len(per_drug_rows),
        "within_patient_spearman": _mean([row["spearman"] for row in within_rows]),
        "within_patient_baseline_spearman": _mean(
            [row["baseline_spearman"] for row in within_rows]
        ),
        "within_patient_ndcg": _mean(
            [row["ndcg"] for row in within_rows if math.isfinite(row["ndcg"])]
        ),
        "per_drug_spearman": _mean([row["spearman"] for row in per_drug_rows]),
        "per_drug_baseline_spearman": _mean(
            [row["baseline_spearman"] for row in per_drug_rows]
        ),
        "within_patient_delta": _mean(within_delta),
        "per_drug_delta": _mean(per_drug_delta),
        "within_patient_delta_ci": _interval_dict(within_interval),
        "per_drug_delta_ci": _interval_dict(per_drug_interval),
        "verdict": {
            "label": verdict.label,
            "primary_endpoint": verdict.primary_endpoint,
            "candidate_minus_baseline": verdict.candidate_minus_baseline,
            "ci_low": verdict.ci_low,
            "ci_high": verdict.ci_high,
            "notes": list(verdict.notes),
        },
        "patients": within_rows,
        "drugs": per_drug_rows,
    }


def decide_verdict(
    within: BootstrapInterval | None,
    per_drug: BootstrapInterval | None,
    *,
    n_patients: int,
    n_drugs: int,
) -> TransferVerdict:
    notes = []
    if n_patients < 8 or n_drugs < 5:
        return TransferVerdict(
            "INCONCLUSIVE_SMALL_SUPPORT",
            "per_drug_patient_spearman",
            float("nan"),
            float("nan"),
            float("nan"),
            ("too few scored patients or drugs",),
        )
    if per_drug is None or within is None:
        return TransferVerdict(
            "INCONCLUSIVE",
            "per_drug_patient_spearman",
            float("nan"),
            float("nan"),
            float("nan"),
            ("bootstrap undefined",),
        )
    if per_drug.point_estimate >= 0.05 and per_drug.low > 0:
        label = "INDIVIDUALIZED_SIGNAL"
        notes = ("B1 minus B0 per-drug Spearman meets the pre-registered gain rule",)
        endpoint = "per_drug_patient_spearman"
        interval = per_drug
    elif abs(per_drug.point_estimate) < 0.02 and within.point_estimate <= 0.02:
        label = "MEAN_EFFECT_DOMINATED"
        notes = (
            "within-patient ranking is not above the drug-mean baseline; per-drug residual is near zero",
        )
        endpoint = "per_drug_patient_spearman"
        interval = per_drug
    elif within.low > 0 and per_drug.high < 0.05:
        label = "WITHIN_RANKING_GAIN_POSSIBLY_MEAN"
        notes = ("within-patient Spearman improved but per-drug residual does not support patient specificity",)
        endpoint = "within_patient_spearman"
        interval = within
    else:
        label = "INCONCLUSIVE"
        notes = ("interval includes both negligible and meaningful per-drug gain",)
        endpoint = "per_drug_patient_spearman"
        interval = per_drug
    return TransferVerdict(
        label,
        endpoint,
        interval.point_estimate,
        interval.low,
        interval.high,
        tuple(notes),
    )


def _optional_bootstrap(
    candidate: Sequence[float],
    baseline: Sequence[float],
    groups: Sequence[str],
    seed: int,
) -> BootstrapInterval | None:
    if len(set(groups)) < 2:
        return None
    return paired_group_bootstrap_difference(
        candidate,
        baseline,
        groups,
        higher_is_better=True,
        seed=seed,
        resamples=2000,
    )


def _safe_spearman(observed: Sequence[float], predicted: Sequence[float]) -> float:
    """Undefined correlations (constants) score as 0, not as missing drugs/patients."""
    try:
        return spearman(observed, predicted)
    except ValueError:
        return 0.0


def _mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def _interval_dict(interval: BootstrapInterval | None) -> dict[str, float] | None:
    if interval is None:
        return None
    return {
        "point_estimate": interval.point_estimate,
        "low": interval.low,
        "high": interval.high,
        "seed": interval.seed,
        "resamples": interval.resamples,
    }
