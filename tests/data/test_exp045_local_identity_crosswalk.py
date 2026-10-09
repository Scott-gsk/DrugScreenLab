import pandas as pd

from scripts.data.audit_exp045_local_identity_crosswalk import (
    _bijective_candidates,
    _exact_patient_candidates,
    _identity_table_audit,
    _identity_projection_sha256,
    _status_and_exit_code,
    _target_identity_audit,
)


def test_exact_sample_key_maps_target_patient_to_nature_individual():
    target = pd.DataFrame([{"patient_id": "PDO-P1", "sample_id": "SAMPLE-1"}])
    nature = pd.DataFrame(
        [{"sample_ID": "SAMPLE-1", "sanger_ID": "SIDM-1", "individual_ID": "IND-1"}]
    )

    candidates, evidence = _exact_patient_candidates(target, nature)

    assert candidates == {"PDO-P1": {"IND-1"}}
    assert evidence == {"PDO-P1": {"sample_id==sample_ID"}}


def test_same_exact_key_linking_to_multiple_patients_is_ambiguous():
    target = pd.DataFrame([{"patient_id": "PDO-P1", "sample_id": "SAMPLE-1"}])
    nature = pd.DataFrame(
        [
            {"sample_ID": "SAMPLE-1", "sanger_ID": "SIDM-1", "individual_ID": "IND-1"},
            {"sample_ID": "SAMPLE-1", "sanger_ID": "SIDM-2", "individual_ID": "IND-2"},
        ]
    )

    candidates, _ = _exact_patient_candidates(target, nature)

    assert candidates["PDO-P1"] == {"IND-1", "IND-2"}


def test_different_case_or_punctuation_does_not_create_a_link():
    target = pd.DataFrame([{"patient_id": "PDO-P1", "sample_id": "sample_1"}])
    nature = pd.DataFrame(
        [{"sample_ID": "SAMPLE-1", "sanger_ID": "SIDM-1", "individual_ID": "IND-1"}]
    )

    candidates, evidence = _exact_patient_candidates(target, nature)

    assert not candidates
    assert not evidence


def test_bijective_candidates_reject_reverse_many_to_one_links():
    accepted, forward_ambiguous, reverse_ambiguous = _bijective_candidates(
        {"PDO-P1", "PDO-P2"},
        {"PDO-P1": {"IND-1"}, "PDO-P2": {"IND-1"}},
    )

    assert accepted == {}
    assert forward_ambiguous == 0
    assert reverse_ambiguous == 1


def test_identity_table_audit_fails_closed_on_missing_required_keys():
    official = pd.DataFrame(
        [{"sample_ID": "SAMPLE-1", "sanger_ID": "SIDM-1", "individual_ID": "IND-1"}]
    )
    paired = pd.DataFrame(
        [{"sample_ID": None, "sanger_ID": "SIDM-1", "individual_ID": "IND-1"}]
    )

    audit = _identity_table_audit(official, paired)

    assert audit["missing_identity_values"]["paired"]["sample_ID"] == 1
    assert audit["required_sample_and_individual_fields_complete"] is False
    assert audit["paired_identity_rows_exact_subset_of_official_availability"] is False


def test_identity_table_audit_reports_duplicate_sample_rows():
    official = pd.DataFrame(
        [
            {"sample_ID": "SAMPLE-1", "sanger_ID": "SIDM-1", "individual_ID": "IND-1"},
            {"sample_ID": "SAMPLE-1", "sanger_ID": "SIDM-1", "individual_ID": "IND-1"},
        ]
    )
    audit = _identity_table_audit(official, official.copy())

    assert audit["official_duplicate_sample_ids"] == 1
    assert audit["paired_duplicate_sample_ids"] == 1


def test_target_identity_audit_reports_missing_and_duplicate_keys():
    target = pd.DataFrame(
        [
            {"patient_id": "PDO-P1", "sample_id": "SAMPLE-1"},
            {"patient_id": "PDO-P1", "sample_id": "SAMPLE-1"},
            {"patient_id": None, "sample_id": "SAMPLE-2"},
        ]
    )

    audit = _target_identity_audit(target)

    assert audit["missing_patient_id"] == 1
    assert audit["missing_sample_id"] == 0
    assert audit["duplicate_patient_sample_rows"] == 1
    assert audit["all_patient_and_sample_keys_complete"] is False


def test_blocked_identity_result_returns_nonzero_status_exit():
    assert _status_and_exit_code(
        registered_provenance_pass=True, pairing_pass=True, target_pass=False, source_pass=False
    ) == ("DATA_BLOCKED", 2)


def test_crosswalk_pass_and_invalid_inputs_have_distinct_exit_codes():
    assert _status_and_exit_code(
        registered_provenance_pass=True, pairing_pass=True, target_pass=True, source_pass=True
    ) == ("IDENTITY_CROSSWALK_PASS", 0)
    assert _status_and_exit_code(
        registered_provenance_pass=False, pairing_pass=True, target_pass=False, source_pass=False
    ) == ("AUDIT_INVALID", 1)


def test_identity_projection_digest_ignores_non_identity_columns():
    left = pd.DataFrame([{"patient_id": "P1", "sample_id": "S1", "unused": "a"}])
    right = pd.DataFrame([{"patient_id": "P1", "sample_id": "S1", "unused": "b"}])
    changed_identity = pd.DataFrame([{"patient_id": "P2", "sample_id": "S1", "unused": "a"}])

    left_hash = _identity_projection_sha256(left, ("patient_id", "sample_id"))

    assert left_hash == _identity_projection_sha256(right, ("patient_id", "sample_id"))
    assert left_hash != _identity_projection_sha256(changed_identity, ("patient_id", "sample_id"))
