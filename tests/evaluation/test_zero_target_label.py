from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drug_screen.evaluation.zero_target_label import (
    expression_identity_permutation_test,
    paired_resampling_summary,
    score_zero_target_predictions,
)


def test_score_joins_frozen_predictions_by_keys_when_rows_are_reordered() -> None:
    predictions = pd.DataFrame([
        {"patient_id": patient, "expression_donor_patient_id": patient, "drug_id": drug,
         "predicted_source_pharmaformer": model, "predicted_source_train_mean": mean}
        for patient, model_values in {
            "p1": [3.0, 2.0, 1.0], "p2": [1.0, 2.0, 3.0],
        }.items()
        for drug, model, mean in zip(["d1", "d2", "d3"], model_values, [3.0, 2.0, 1.0])
    ])
    outcomes = pd.DataFrame([
        {"patient_id": patient, "drug_id": drug, "response": response}
        for patient, values in {"p1": [3.0, 2.0, 1.0], "p2": [1.0, 2.0, 3.0]}.items()
        for drug, response in zip(["d1", "d2", "d3"], values)
    ]).sample(frac=1.0, random_state=19)

    scored_pairs, patient_scores = score_zero_target_predictions(predictions, outcomes)

    joined_values = scored_pairs.sort_values(["patient_id", "drug_id"]).reset_index(drop=True)
    target_values = outcomes.sort_values(["patient_id", "drug_id"]).reset_index(drop=True)
    assert joined_values[["patient_id", "drug_id", "response"]].equals(
        target_values[["patient_id", "drug_id", "response"]]
    )
    scores = patient_scores.set_index("patient_id")
    assert scores.loc["p1", "spearman_source_pharmaformer"] == pytest.approx(1.0)
    assert scores.loc["p2", "spearman_source_pharmaformer"] == pytest.approx(1.0)
    assert scores.loc["p1", "spearman_source_train_mean"] == pytest.approx(1.0)


def test_score_marks_undefined_patient_scores_without_filling_them() -> None:
    rows = []
    for drug, model, mean, response in [
        ("d1", 1.0, 3.0, 2.0), ("d2", 2.0, 2.0, 2.0), ("d3", 3.0, 1.0, 2.0),
    ]:
        rows.append({"patient_id": "p", "expression_donor_patient_id": "p", "drug_id": drug,
                     "predicted_source_pharmaformer": model, "predicted_source_train_mean": mean,
                     "response": response})
    frame = pd.DataFrame(rows)

    _, scores = score_zero_target_predictions(frame, frame[["patient_id", "drug_id", "response"]])

    assert scores.loc[0, "reason_source_pharmaformer"] == "CONSTANT_OBSERVED"
    assert np.isnan(scores.loc[0, "spearman_source_pharmaformer"])
    assert scores.loc[0, "reason_source_train_mean"] == "CONSTANT_OBSERVED"
    assert np.isnan(scores.loc[0, "difference_model_minus_mean"])


def test_paired_resampling_uses_only_finite_patient_differences_and_is_reproducible() -> None:
    scores = pd.DataFrame({"difference_model_minus_mean": [0.2, 0.4, np.nan, -0.1]})

    first = paired_resampling_summary(scores, bootstrap_replicates=200, sign_flip_replicates=500)
    second = paired_resampling_summary(scores, bootstrap_replicates=200, sign_flip_replicates=500)

    assert first == second
    assert first["n_patients"] == 3
    assert first["estimate_model_minus_source_mean"] == pytest.approx(0.5 / 3)
    assert first["ci95_low"] <= first["estimate_model_minus_source_mean"] <= first["ci95_high"]


def test_expression_permutation_repeats_deterministically_and_keeps_patient_denominator() -> None:
    donors = {
        "p1": [3.0, 2.0, 1.0],
        "p2": [1.0, 2.0, 3.0],
        "p3": [2.0, 3.0, 1.0],
    }
    predictions = pd.DataFrame([
        {"expression_donor_patient_id": patient, "drug_id": drug,
         "predicted_source_pharmaformer": value}
        for patient, values in donors.items()
        for drug, value in zip(["d1", "d2", "d3"], values)
    ])
    outcomes = pd.DataFrame([
        {"patient_id": patient, "drug_id": drug, "response": value}
        for patient, values in donors.items()
        for drug, value in zip(["d1", "d2", "d3"], values)
    ]).sample(frac=1.0, random_state=7)

    first_summary, first_draws = expression_identity_permutation_test(
        predictions, outcomes, permutations=30, seed=81,
    )
    second_summary, second_draws = expression_identity_permutation_test(
        predictions, outcomes, permutations=30, seed=81,
    )

    assert first_summary == second_summary
    pd.testing.assert_frame_equal(first_draws, second_draws)
    assert first_summary["observed_true_alignment_macro_spearman"] == pytest.approx(1.0)
    assert first_summary["n_fixed_patients_scored"] == 3
    assert len(first_draws) == 30
    assert first_draws["n_fixed_patients_scored"].eq(3).all()
    assert first_draws["comparable_to_observed_denominator"].all()


def test_expression_permutation_rejects_missing_or_duplicate_identity_keys() -> None:
    predictions = pd.DataFrame({
        "expression_donor_patient_id": ["p1", "p1", "p1"],
        "drug_id": ["d1", "d1", "d2"],
        "predicted_source_pharmaformer": [1.0, 2.0, 3.0],
    })
    outcomes = pd.DataFrame({
        "patient_id": ["p1", "p1", "p1"],
        "drug_id": ["d1", "d2", "d3"],
        "response": [1.0, 2.0, 3.0],
    })

    with pytest.raises(ValueError, match="duplicate expression-donor/drug"):
        expression_identity_permutation_test(predictions, outcomes, permutations=2)
