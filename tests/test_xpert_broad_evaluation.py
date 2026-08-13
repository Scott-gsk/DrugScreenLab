from __future__ import annotations

import numpy as np
import pandas as pd

from drug_screen.evaluation.xpert_broad import audit_oracle_coverage, rank_metrics


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


def test_rank_metrics_random_null_is_seeded_and_reports_lift() -> None:
    frame = pd.DataFrame(
        {
            "pert_id": list("ABCDE"),
            "predicted": [5.0, 4.0, 3.0, 2.0, 1.0],
            "sensitivity_score": [50.0, 40.0, 30.0, 20.0, 10.0],
        }
    )
    first = rank_metrics(frame, score_column="predicted", ks=(2,), minimum_candidates=3, null_repeats=32, null_seed=9)
    second = rank_metrics(frame, score_column="predicted", ks=(2,), minimum_candidates=3, null_repeats=32, null_seed=9)
    assert first["null_baseline"] == second["null_baseline"]
    assert first["top_k"]["2"]["overlap_lift"] > 1.0
    assert first["top_k"]["2"]["delta_ndcg"] > 0.0


def test_audit_oracle_coverage_uses_unique_context_perturbagen_pairs() -> None:
    prism = pd.DataFrame({"context_id": ["C1", "C1", "C2"], "pert_id": ["A", "A", "B"]})
    oracle = pd.DataFrame({"cell_iname": ["C1", "C3"], "pert_id": ["A", "X"]})
    result = audit_oracle_coverage(prism, oracle)
    assert result["status"] == "READY"
    assert result["prism_unique_pairs"] == 2
    assert result["oracle_unique_pairs"] == 2
    assert result["overlap_unique_pairs"] == 1
    assert result["pair_coverage_rate"] == 0.5
