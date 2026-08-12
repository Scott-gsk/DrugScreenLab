from pathlib import Path

import pandas as pd
import pytest

from drug_screen.evaluation.cross_study import CrossStudyBlockedError, evaluate_frozen_cross_study


def test_cross_study_requires_external_asset(tmp_path: Path) -> None:
    with pytest.raises(CrossStudyBlockedError, match="unavailable"):
        evaluate_frozen_cross_study(
            labels_path=tmp_path / "missing.tsv",
            predictions_path=tmp_path / "pred.tsv",
            output_path=tmp_path / "out.json",
            frozen_candidate_ids=["A"],
        )


def test_cross_study_has_strict_candidate_set(tmp_path: Path) -> None:
    labels = tmp_path / "labels.tsv"
    predictions = tmp_path / "pred.tsv"
    pd.DataFrame({
        "study_id": ["S", "S"], "context_id": ["C", "C"],
        "pert_id": ["A", "B"], "sensitivity_score": [1.0, 0.0],
    }).to_csv(labels, sep="\t", index=False)
    pd.DataFrame({
        "study_id": ["S", "S"], "context_id": ["C", "C"],
        "pert_id": ["A", "B"], "reversal_predicted": [0.9, 0.1],
    }).to_csv(predictions, sep="\t", index=False)
    result = evaluate_frozen_cross_study(
        labels_path=labels,
        predictions_path=predictions,
        output_path=tmp_path / "out.json",
        frozen_candidate_ids=["A", "B"],
    )
    assert result["status"] == "COMPLETE"
    assert result["labels_used_for_tuning"] is False
