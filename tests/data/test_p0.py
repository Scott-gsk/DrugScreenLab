import pytest
from pathlib import Path

from drug_screen.data.p0 import (
    DatasetReadiness,
    PerturbationRecord,
    assert_matched_controls,
    assert_no_split_leakage,
    assert_task_role,
    assert_level3_landmark_core,
    count_same_plate_vehicle_candidates,
    deterministic_metadata_digest,
)
from drug_screen.data.registry import validate_readiness_matrix


def _control(sample_id="control"):
    return PerturbationRecord(sample_id, None, None, "A375", None, None, "rf-1", "P1", "24h", True)


def _treatment(sample_id="treatment", control_id="control", compound_id="CHEMBL:1"):
    return PerturbationRecord(sample_id, compound_id, None, "A375", None, control_id, "rf-1", "P1", "24h", False)


def test_readiness_schema_requires_explicit_blockers_and_valid_role(tmp_path):
    row = {
        "dataset_id": "x", "accession_or_version": "v1", "intended_role": "TRAIN",
        "local_availability": "absent", "source_availability": "public", "checksum_evidence": "none",
        "metadata_status": "absent", "license_status": "unknown", "preprocessing_status": "not_started",
        "blockers": ["missing"],
    }
    assert DatasetReadiness.from_mapping(row).blockers == ("missing",)
    row["intended_role"] = "TRAINING"
    with pytest.raises(ValueError, match="intended_role"):
        DatasetReadiness.from_mapping(row)
    matrix = tmp_path / "matrix.json"
    matrix.write_text("[]", encoding="utf-8")
    assert validate_readiness_matrix(matrix) == []


def test_exp001_readiness_matrix_is_schema_valid():
    path = Path("experiments/records/EXP-001/readiness_matrix.json")
    assert validate_readiness_matrix(path) == []


def test_matched_control_must_share_context_plate_and_timepoint():
    assert_matched_controls([_control(), _treatment()])
    bad = PerturbationRecord("bad", "CHEMBL:2", None, "A375", None, "control", "rf-1", "P2", "24h", False)
    with pytest.raises(ValueError, match="plate_id"):
        assert_matched_controls([_control(), bad])


def test_split_leakage_detects_canonical_compound_across_splits():
    second_control = _control("control-2")
    second = _treatment("treatment-2", control_id="control-2")
    with pytest.raises(ValueError, match="compound_id=CHEMBL:1"):
        assert_no_split_leakage(
            {
                "control": "train",
                "treatment": "train",
                "control-2": "test",
                "treatment-2": "test",
            },
            [_control(), _treatment(), second_control, second],
            ["compound_id"],
        )


def test_split_leakage_rejects_a_matched_control_in_another_split():
    with pytest.raises(ValueError, match="matched control"):
        assert_no_split_leakage(
            {"control": "test", "treatment": "train"},
            [_control(), _treatment()],
            ["compound_id"],
        )


def test_external_test_role_cannot_be_used_for_fitting():
    assert_task_role("EXTERNAL_TEST", "evaluate")
    with pytest.raises(ValueError, match="cannot"):
        assert_task_role("EXTERNAL_TEST", "tune")


def test_metadata_digest_is_deterministic():
    assert deterministic_metadata_digest({"b": [2, 1], "a": "x"}) == deterministic_metadata_digest({"a": "x", "b": [2, 1]})


def test_level3_landmark_core_allows_inferred_non_landmarks_but_requires_978_direct_core():
    assert_level3_landmark_core((3, 979), [f"g{i}" for i in range(979)], [1] * 978 + [0], ["i1", "i2", "i3"], ["i3", "i2", "i1"])
    with pytest.raises(ValueError, match="978"):
        assert_level3_landmark_core((3, 979), [f"g{i}" for i in range(979)], [1] * 977 + [0, 0], ["i1", "i2", "i3"], ["i1", "i2", "i3"])


def test_same_plate_vehicle_pairing_counts_candidates_without_materializing_delta():
    chemical = [("P1", "A375", 24, "h"), ("P2", "A375", 24, "h")]
    vehicle = [("P1", "A375", 24, "h")]
    assert count_same_plate_vehicle_candidates(chemical, vehicle) == (2, 1)
