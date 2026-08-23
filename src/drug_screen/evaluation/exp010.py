"""Pre-registered, patient-safe evaluation primitives for EXP-010.

This module deliberately consumes predictions only.  It does not fit models,
select features, calibrate thresholds, or inspect test labels during selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from drug_screen.evaluation.phase1_prism import spearman


@dataclass(frozen=True)
class PredictionRecord:
    exp_id: str
    model_id: str
    split: str
    fold: int
    patient_id: str
    pdo_id: str
    drug_id: str
    observed_dss: float
    predicted_dss: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def patient_outer_folds(patient_ids: Iterable[str], *, n_folds: int = 5, seed: int = 20260823) -> dict[str, int]:
    """Assign each patient to exactly one deterministic, balanced outer fold."""
    patients = sorted({str(value).strip() for value in patient_ids if str(value).strip()})
    if n_folds < 3 or len(patients) < n_folds:
        raise ValueError("patient folds require at least n_folds patients and n_folds >= 3")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(patients, dtype=object)
    rng.shuffle(shuffled)
    return {str(patient): int(index % n_folds) for index, patient in enumerate(shuffled)}


def fold_roles(assignments: Mapping[str, int], test_fold: int, *, n_folds: int = 5) -> dict[str, str]:
    """Use the next fold for validation and all remaining folds for training."""
    if test_fold not in range(n_folds):
        raise ValueError("invalid test fold")
    validation_fold = (test_fold + 1) % n_folds
    roles = {
        patient: ("test" if fold == test_fold else "validation" if fold == validation_fold else "train")
        for patient, fold in assignments.items()
    }
    if set(roles.values()) != {"train", "validation", "test"}:
        raise ValueError("fold assignment does not contain all required roles")
    return roles


def assert_patient_isolation(frame: pd.DataFrame) -> None:
    required = {"patient_id", "pdo_id", "split"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing columns: {sorted(required - set(frame.columns))}")
    if frame[list(required)].isna().any().any():
        raise ValueError("split identities cannot be null")
    patient_counts = frame.groupby("patient_id", observed=True)["split"].nunique()
    pdo_counts = frame.groupby("pdo_id", observed=True)["split"].nunique()
    if (patient_counts > 1).any():
        raise ValueError("patient leakage across splits")
    if (pdo_counts > 1).any():
        raise ValueError("PDO leakage across splits")


def _ndcg_at_k(observed: np.ndarray, predicted: np.ndarray, k: int) -> float:
    k = min(int(k), len(observed))
    if k < 1:
        return float("nan")
    relevance = observed - np.min(observed)
    pred_order = np.argsort(-predicted, kind="mergesort")[:k]
    ideal_order = np.argsort(-observed, kind="mergesort")[:k]
    discounts = np.log2(np.arange(2, k + 2, dtype=float))
    dcg = float(np.sum(relevance[pred_order] / discounts))
    ideal = float(np.sum(relevance[ideal_order] / discounts))
    return dcg / ideal if ideal > 0 else float("nan")


def patient_ranking_metrics(records: Sequence[PredictionRecord] | pd.DataFrame, *, k: int = 5) -> dict[str, object]:
    """Aggregate per-PDO rankings to patient-equal macro metrics.

    Each PDO must contain at least two drugs.  Metrics are computed per PDO,
    averaged within patient, and only then macro-averaged across patients.
    """
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame([row.to_dict() for row in records])
    required = {"patient_id", "pdo_id", "drug_id", "observed_dss", "predicted_dss"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing prediction columns: {sorted(required - set(frame.columns))}")
    if frame.duplicated(["pdo_id", "drug_id"]).any():
        raise ValueError("duplicate PDO-drug predictions must be resolved before evaluation")
    rows: list[dict[str, object]] = []
    for (patient, pdo), group in frame.groupby(["patient_id", "pdo_id"], sort=True, observed=True):
        if len(group) < 2:
            continue
        observed = group["observed_dss"].to_numpy(float)
        predicted = group["predicted_dss"].to_numpy(float)
        if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
            raise ValueError("metrics require finite observations and predictions")
        rows.append({"patient_id": str(patient), "pdo_id": str(pdo), "drug_count": len(group),
                     "spearman": spearman(predicted, observed), "ndcg_at_5": _ndcg_at_k(observed, predicted, k),
                     "mae": float(np.mean(np.abs(predicted - observed)))})
    per_pdo = pd.DataFrame(rows)
    if per_pdo.empty:
        raise ValueError("no evaluable PDO has at least two drugs")
    per_patient = per_pdo.groupby("patient_id", sort=True, observed=True)[["spearman", "ndcg_at_5", "mae"]].mean()
    macro = per_patient.mean()
    return {"format": "exp010_patient_macro_metrics_v1", "patient_count": int(len(per_patient)),
            "pdo_count": int(len(per_pdo)), "per_pdo": per_pdo.to_dict("records"),
            "per_patient": per_patient.reset_index().to_dict("records"),
            "macro": {name: float(value) for name, value in macro.items()}}


def cluster_bootstrap_delta(candidate: pd.DataFrame, baseline: pd.DataFrame, *, metric: str = "spearman",
                            repeats: int = 2000, seed: int = 20260823) -> dict[str, object]:
    """Paired patient-cluster bootstrap for M3 minus M2."""
    if repeats < 100:
        raise ValueError("at least 100 bootstrap repeats are required")
    left = candidate.set_index("patient_id")[metric].rename("candidate")
    right = baseline.set_index("patient_id")[metric].rename("baseline")
    joined = pd.concat([left, right], axis=1, join="inner").dropna()
    if len(joined) < 2 or set(left.index) != set(right.index):
        raise ValueError("paired bootstrap requires the same patients in both models")
    delta = (joined["candidate"] - joined["baseline"]).to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = np.mean(delta[rng.integers(0, len(delta), size=(repeats, len(delta)))], axis=1)
    return {"format": "exp010_patient_cluster_bootstrap_v1", "metric": metric,
            "patient_count": int(len(delta)), "repeats": repeats, "seed": seed,
            "delta_mean": float(np.mean(delta)), "ci_95": [float(x) for x in np.quantile(samples, [0.025, 0.975])]}


def promising_fast_decision(m3: Mapping[str, float], m2: Mapping[str, float], ci_lower: float) -> dict[str, object]:
    delta = float(m3["spearman"] - m2["spearman"])
    ndcg_ok = float(m3["ndcg_at_5"]) >= float(m2["ndcg_at_5"])
    mae_ok = float(m3["mae"]) <= float(m2["mae"]) * 1.05
    passed = delta >= 0.05 and ci_lower > 0 and ndcg_ok and mae_ok
    return {"status": "PROMISING_FAST" if passed else "NO_GO", "spearman_delta": delta,
            "ci_lower_gt_zero": ci_lower > 0, "ndcg_no_decline": ndcg_ok, "mae_within_5_percent": mae_ok}
