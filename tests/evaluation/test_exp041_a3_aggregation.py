from __future__ import annotations

import pandas as pd
import pytest

from scripts.evaluation.aggregate_exp041_a3 import (
    ARMS,
    SEEDS,
    analyze_predictions,
    validate_prediction_keys,
)


def _predictions(*, constant_adapter_patient: str | None = None) -> pd.DataFrame:
    rows = []
    for patient in ("p1", "p2", "p3", "p4"):
        for drug_rank in range(1, 5):
            for seed in SEEDS:
                observed = float(drug_rank)
                adapter = (
                    1.0 if patient == constant_adapter_patient else observed
                )
                values = {
                    "random_full": float(5 - drug_rank),
                    "source_head": observed,
                    "source_full": float(5 - drug_rank),
                    "source_adapter": adapter,
                }
                for arm in ARMS:
                    rows.append({
                        "fold": (int(patient[-1]) - 1) % 5,
                        "fraction": 0.5,
                        "arm": arm,
                        "seed": seed,
                        "patient_id": patient,
                        "drug_id": f"d{drug_rank}",
                        "observed": observed,
                        "predicted": values[arm],
                    })
    return pd.DataFrame(rows)


def test_a3_uses_patient_scores_averaged_across_seeds_and_signed_adapter_minus_full() -> None:
    summary, patient_scores = analyze_predictions(_predictions())

    assert summary["primary_comparison"] == "source_adapter_minus_source_full"
    assert summary["main"]["gain"] == pytest.approx(2.0)
    assert summary["main"]["n_valid_patients"] == 4
    assert summary["n_patients_excluded_for_undefined_or_incomplete_scores"] == 0
    assert patient_scores["gain_adapter_minus_full"].mean() == pytest.approx(2.0)
    assert summary["seed_stability"]["mean_sd_difference_adapter_minus_full"] == pytest.approx(0.0)
    assert summary["holm_adjustment"] is None


def test_a3_reports_patients_with_undefined_scores_instead_of_filling_them() -> None:
    summary, patient_scores = analyze_predictions(
        _predictions(constant_adapter_patient="p4")
    )

    assert summary["main"]["n_valid_patients"] == 3
    assert summary["excluded_patient_ids"] == ["p4"]
    p4 = patient_scores.loc[patient_scores["patient_id"].eq("p4")].iloc[0]
    assert pd.isna(p4["source_adapter_seed_mean_spearman"])
    assert pd.isna(p4["gain_adapter_minus_full"])


def test_prediction_key_gate_rejects_missing_or_extra_test_pairs() -> None:
    frame = _predictions().query("fold == 0")
    expected = pd.DataFrame({
        "patient_id": ["p1"],
        "drug_id": ["d1"],
        "response": [1.0],
    })
    # One expected pair is insufficient: every arm and seed must carry the full panel.
    with pytest.raises(ValueError, match="frozen test keys differ"):
        validate_prediction_keys(frame, expected, fold=0)
