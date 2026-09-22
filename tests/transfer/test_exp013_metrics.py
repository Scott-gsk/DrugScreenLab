import math

import pytest

from drug_screen.transfer.metrics import evaluate_predictions, ndcg_at_k, spearman


def test_spearman_is_one_for_monotone_pairs():
    assert spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)


def test_ndcg_is_one_when_ranking_matches():
    observed = [0.1, 0.5, 0.9, 0.2]
    assert ndcg_at_k(observed, observed, k=3) == pytest.approx(1.0)
    assert math.isnan(ndcg_at_k(observed, observed, k=5))


def test_evaluate_predictions_flags_mean_effect_when_only_drug_means_work():
    patients = []
    drugs = []
    observed = []
    baseline = []
    for patient, offset in [
        ("p1", 0.0),
        ("p2", 0.1),
        ("p3", -0.2),
        ("p4", 0.2),
        ("p5", 0.05),
        ("p6", -0.1),
        ("p7", 0.15),
        ("p8", -0.05),
    ]:
        for drug, mean in [("a", 3.0), ("b", 1.0), ("c", 2.0), ("d", 0.5), ("e", 2.5)]:
            patients.append(patient)
            drugs.append(drug)
            observed.append(mean + offset)
            baseline.append(mean)
    result = evaluate_predictions(
        patients,
        drugs,
        observed,
        baseline,
        baseline,
        min_drugs_per_patient=3,
        seed=7,
    )
    assert result["within_patient_spearman"] == pytest.approx(1.0)
    assert result["per_drug_spearman"] == result["per_drug_baseline_spearman"]
    assert result["verdict"]["label"] == "MEAN_EFFECT_DOMINATED"
