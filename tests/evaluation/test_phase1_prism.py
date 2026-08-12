from __future__ import annotations

import numpy as np
import pandas as pd

from drug_screen.evaluation.phase1_prism import line_metrics, reversal_score


def test_reversal_score_prefers_opposite_signed_perturbation():
    disease = np.asarray([1.0, 2.0, -1.0, -2.0])
    opposite = np.asarray([-1.0, -2.0, 1.0, 2.0])
    aligned = np.asarray([1.0, 2.0, -1.0, -2.0])
    assert reversal_score(disease, opposite) > reversal_score(disease, aligned)


def test_line_metrics_marks_missing_candidate_as_ineligible():
    frame = pd.DataFrame(
        {
            "pert_id": ["a", "b"],
            "predicted_score": [0.2, np.nan],
            "sensitivity_score": [0.5, 0.1],
        }
    )
    result = line_metrics(frame, "predicted_score", minimum_candidates=3)
    assert result["eligible"] is False
    assert result["reason"] == "fewer_than_minimum_candidates"
