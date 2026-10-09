"""Cached CPU evaluation for fixed-panel perturbation-reversal validation.

The observed validation responses, grouping keys and disease signature are fixed
within an inner fold. This module computes their grouped reversal scores once,
then scores each new prediction without rebuilding pandas groups or rescoring
the unchanged observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from drug_screen.evaluation.reversal import reversal_score


@dataclass(frozen=True)
class _DrugGroup:
    cell_id: str
    drug_id: str
    positions: np.ndarray
    measured_reversal: float


class ReversalValidationCache:
    """Reuse fixed observed scores and group indices across validation epochs."""

    def __init__(self, keys: pd.DataFrame, observed_delta: np.ndarray,
                 genes: Sequence[str], disease_up: Sequence[str],
                 disease_down: Sequence[str]) -> None:
        required = {"cell_id", "pert_id"}
        if required - set(keys.columns):
            raise ValueError("validation keys require cell_id and pert_id")
        observed = np.asarray(observed_delta, dtype=np.float32)
        self.n_rows = len(keys)
        self.genes = tuple(map(str, genes))
        self.disease_up = tuple(map(str, disease_up))
        self.disease_down = tuple(map(str, disease_down))
        if observed.ndim != 2 or len(observed) != len(keys):
            raise ValueError("observed validation delta and key rows differ")
        if observed.shape[1] != len(self.genes):
            raise ValueError("observed validation delta and gene axis differ")

        normalized_keys = keys[["cell_id", "pert_id"]].reset_index(drop=True)
        groups = normalized_keys.groupby(["cell_id", "pert_id"], sort=True).indices
        self.drug_groups: list[_DrugGroup] = []
        for (cell_id, drug_id), positions in groups.items():
            positions = np.asarray(positions, dtype=np.int64)
            measured = observed[positions].mean(axis=0, dtype=np.float64)
            self.drug_groups.append(_DrugGroup(
                str(cell_id), str(drug_id), positions,
                reversal_score(measured, self.genes,
                               self.disease_up, self.disease_down),
            ))

        self.cell_groups: dict[str, list[int]] = {}
        for group_index, group in enumerate(self.drug_groups):
            self.cell_groups.setdefault(group.cell_id, []).append(group_index)

    def score(self, prediction_delta: np.ndarray) -> tuple[float, pd.DataFrame]:
        """Return the original equal-cell mean Spearman and its cell-level rows."""
        prediction = np.asarray(prediction_delta, dtype=np.float32)
        if prediction.ndim != 2 or prediction.shape[0] != self.n_rows:
            raise ValueError("prediction rows differ from cached validation rows")
        if prediction.shape[1] != len(self.genes):
            raise ValueError("prediction gene axis differs from cached validation axis")

        drug_scores = []
        for group in self.drug_groups:
            predicted = prediction[group.positions].mean(axis=0, dtype=np.float64)
            drug_scores.append({
                "cell_id": group.cell_id,
                "pert_id": group.drug_id,
                "measured_reversal": group.measured_reversal,
                "predicted_reversal": reversal_score(
                    predicted, self.genes, self.disease_up, self.disease_down),
                "n_conditions": len(group.positions),
            })

        cell_rows = []
        drug_frame = pd.DataFrame(drug_scores)
        for cell_id, group_indices in self.cell_groups.items():
            frame = drug_frame.iloc[group_indices]
            observed = frame["measured_reversal"].to_numpy(dtype=np.float64)
            predicted = frame["predicted_reversal"].to_numpy(dtype=np.float64)
            reason = None
            if len(frame) < 5:
                reason = "FEWER_THAN_FIVE_DRUGS"
            elif np.ptp(observed) == 0:
                reason = "CONSTANT_MEASURED_REVERSAL"
            elif np.ptp(predicted) == 0:
                reason = "CONSTANT_PREDICTED_REVERSAL"
            value = float(spearmanr(observed, predicted).statistic) if reason is None else float("nan")
            if reason is None and not np.isfinite(value):
                reason = "UNDEFINED_CORRELATION"
            cell_rows.append({
                "cell_id": cell_id,
                "spearman": value,
                "n_drugs": len(frame),
                "reason": reason,
                "n_conditions": int(frame["n_conditions"].sum()),
            })

        cell_scores = pd.DataFrame(cell_rows)
        valid = cell_scores["spearman"].dropna()
        if len(valid) != len(cell_scores) or len(valid) < 3:
            return float("nan"), cell_scores
        return float(valid.mean()), cell_scores
