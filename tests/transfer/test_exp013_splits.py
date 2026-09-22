import pytest

from drug_screen.transfer.contract import TransferObservation
from drug_screen.transfer.splits import assign_patient_folds, assert_patient_isolation, split_for_fold


def test_patient_folds_keep_all_models_together():
    patients = ["p1", "p1", "p2", "p2", "p3", "p4"]
    assignment = assign_patient_folds(patients, n_folds=2, seed=1)
    assert set(assignment.patient_to_test_fold) == {"p1", "p2", "p3", "p4"}
    fold0 = {patient: split_for_fold(assignment, patient, 0) for patient in assignment.patient_to_test_fold}
    rows = [
        TransferObservation(
            patient,
            f"{patient}-m",
            "toy",
            "pdo",
            "PDAC",
            "d1",
            1.0,
            "dss",
            True,
            fold0[patient],
        )
        for patient in patients
    ]
    assert_patient_isolation(rows, fold0)


def test_patient_isolation_detects_split_conflict():
    assignment = assign_patient_folds(["p1", "p2"], n_folds=2, seed=0)
    fold0 = {patient: split_for_fold(assignment, patient, 0) for patient in assignment.patient_to_test_fold}
    leaked = fold0 | {"p1": "train" if fold0["p1"] != "train" else "test"}
    rows = [
        TransferObservation("p1", "m", "toy", "pdo", "PDAC", "d1", 1.0, "dss", True, fold0["p1"]),
        TransferObservation("p1", "m2", "toy", "pdo", "PDAC", "d2", 2.0, "dss", True, leaked["p1"]),
        TransferObservation("p2", "m", "toy", "pdo", "PDAC", "d1", 1.0, "dss", True, fold0["p2"]),
    ]
    with pytest.raises(ValueError, match="patient leakage|split_id mismatch"):
        assert_patient_isolation(rows, leaked)
