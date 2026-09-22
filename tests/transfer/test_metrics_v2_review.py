"""Independent audit cases for EXP-014 metrics; these are not implementation mirrors."""

import numpy as np
import pandas as pd
import pytest

from drug_screen.transfer.metrics_v2 import evaluate_predictions


def _panel(patient, pdo, *, direction=1, shift=0.0):
    values = np.arange(10, dtype=float) + shift
    return pd.DataFrame({
        "patient_id": patient, "model_id": pdo,
        "drug_id": [f"d{i}" for i in range(10)],
        "observed": values, "predicted": direction * values,
        "baseline": -values,
    })


def test_extra_pdo_changes_only_that_patients_equal_weighted_average():
    # p1 has one correct and one reversed PDO, p2 has one correct PDO.  PDO
    # pooling would give 1/3; patient-first aggregation must give 1/2.
    table = pd.concat([
        _panel("p1", "o1", direction=1),
        _panel("p1", "o2", direction=-1),
        _panel("p2", "o3", direction=1),
    ], ignore_index=True)
    report = evaluate_predictions(table, resamples=8)
    assert next(row for row in report["patients"] if row["patient_id"] == "p1")["spearman"] == pytest.approx(0)
    assert report["patient_average"]["spearman"] == pytest.approx(0.5)


def test_per_drug_averages_pdos_before_patient_correlation():
    # For d0, p1's two PDO values average to 0.  The three patient-level
    # values are then [0, 1, 2]: candidate rho=1, baseline rho=-1.
    frames = []
    for patient, pdo, shift in (("p1", "o1", -1), ("p1", "o2", 1), ("p2", "o3", 1), ("p3", "o4", 2)):
        values = np.arange(3, dtype=float) + shift
        frames.append(pd.DataFrame({"patient_id": patient, "model_id": pdo,
            "drug_id": ["d0", "d1", "d2"], "observed": values,
            "predicted": values, "baseline": -values}))
    row = next(row for row in evaluate_predictions(pd.concat(frames), resamples=4)["drugs"] if row["drug_id"] == "d0")
    assert row["n_patients"] == 3
    assert row["spearman_difference"] == pytest.approx(2.0)


def test_patient_bootstrap_keeps_duplicate_draws_as_distinct_blocks():
    # Patient p1 has a +2 rho difference and p2 has 0.  Reproduce the exact
    # seeded patient draws: merging duplicate p1/p2 draws would change this CI.
    table = pd.concat([_panel("p1", "o1", direction=1), _panel("p2", "o2", direction=1)], ignore_index=True)
    table.loc[table.patient_id == "p2", "baseline"] = table.loc[table.patient_id == "p2", "observed"]
    seed, resamples = 11, 9
    report = evaluate_predictions(table, seed=seed, resamples=resamples)
    draws = np.random.default_rng(seed).integers(0, 2, (resamples, 2))
    expected = [np.mean([[2.0, 0.0][index] for index in draw]) for draw in draws]
    interval = report["bootstrap"]["spearman_difference"]
    assert interval["low"] == pytest.approx(np.quantile(expected, .025))
    assert interval["high"] == pytest.approx(np.quantile(expected, .975))


def test_constant_predictions_have_an_explicit_undefined_reason():
    report = evaluate_predictions(_panel("p1", "o1").assign(predicted=0.0), resamples=2)
    row = report["pdos"][0]
    assert row["candidate_spearman_undefined_reason"] == "constant_predicted"
    assert row["baseline_spearman_undefined_reason"] is None


@pytest.mark.parametrize("resamples", [True, 1, 2.0])
def test_resamples_must_be_an_integer_of_at_least_two(resamples):
    with pytest.raises(ValueError, match="resamples"):
        evaluate_predictions(_panel("p1", "o1"), resamples=resamples)


@pytest.mark.parametrize("k", [True, 0, 3.0])
def test_ndcg_k_must_be_a_positive_integer(k):
    with pytest.raises(ValueError, match="k"):
        from drug_screen.transfer.metrics_v2 import ndcg_at_k
        ndcg_at_k([0.0, 1.0], [0.0, 1.0], k=k)
