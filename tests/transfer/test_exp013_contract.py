from drug_screen.transfer.contract import (
    EXPLORE_MIN_PATIENTS,
    TransferObservation,
    assert_higher_is_sensitive,
    eligibility_report,
)
import pytest


def _row(patient: str, drug: str, response: float) -> TransferObservation:
    return TransferObservation(
        patient_id=patient,
        model_id=f"{patient}-m",
        study_id="toy",
        domain="pdo",
        cancer_type="PDAC",
        drug_id=drug,
        response=response,
        response_name="dss",
        higher_is_sensitive=True,
        split_id="train",
    )


def test_label_direction_is_frozen():
    rows = [_row("p1", "d1", 1.0), _row("p2", "d1", 2.0)]
    assert_higher_is_sensitive(rows)
    bad = [
        TransferObservation(
            "p1",
            "m",
            "toy",
            "pdo",
            "PDAC",
            "d1",
            1.0,
            "auc",
            False,
            "train",
        )
    ]
    with pytest.raises(ValueError, match="higher-is-sensitive"):
        assert_higher_is_sensitive(bad)


def test_eligibility_flags_small_and_shared_drugs():
    rows = [_row(f"p{i}", f"d{j}", float(i + j)) for i in range(25) for j in range(12)]
    report = eligibility_report(rows, source_drug_ids=[f"d{j}" for j in range(10)])
    assert report["n_patients"] == 25
    assert report["scale"] == "exploratory_baselines_and_learning_curve"
    assert report["min_shared_drugs_met"] is True
    assert report["n_patients"] >= EXPLORE_MIN_PATIENTS
