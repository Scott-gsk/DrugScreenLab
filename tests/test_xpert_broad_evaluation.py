from __future__ import annotations

import numpy as np
import pandas as pd

from drug_screen.evaluation.xpert_broad import rank_metrics


def test_rank_metrics_reports_perfect_rank_and_top_k_overlap() -> None:
    frame = pd.DataFrame(
        {
            "pert_id": ["A", "B", "C", "D"],
            "predicted": [4.0, 3.0, 2.0, 1.0],
            "sensitivity_score": [40.0, 30.0, 20.0, 10.0],
        }
    )
    result = rank_metrics(frame, score_column="predicted", ks=(2, 3), minimum_candidates=3)
    assert result["eligible"] is True
    assert result["spearman"] == 1.0
    assert result["top_k"]["2"]["overlap_rate"] == 1.0
    assert result["top_k"]["3"]["overlap_rate"] == 1.0


def test_rank_metrics_rejects_constant_or_small_input() -> None:
    frame = pd.DataFrame(
        {
            "pert_id": ["A", "B"],
            "predicted": [1.0, 1.0],
            "sensitivity_score": [2.0, 1.0],
        }
    )
    result = rank_metrics(frame, score_column="predicted", minimum_candidates=3)
    assert result["eligible"] is False
    assert result["reason"] == "fewer_than_minimum_candidates"


def test_rank_metrics_drops_nonfinite_values() -> None:
    frame = pd.DataFrame(
        {
            "pert_id": ["A", "B", "C"],
            "predicted": [3.0, np.nan, 1.0],
            "sensitivity_score": [3.0, 2.0, 1.0],
        }
    )
    result = rank_metrics(frame, score_column="predicted", minimum_candidates=2)
    assert result["candidate_count"] == 2
    assert result["eligible"] is True
