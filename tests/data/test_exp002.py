import pytest

from drug_screen.data.exp002 import (
    assert_exclusive_assignments,
    deterministic_split,
    deterministic_vehicle_partition,
    has_all_splits,
    is_non_degenerate,
    strict_components,
)


def test_legacy_single_control_plate_rule_reproduces_false_single_component():
    components = strict_components([
        ("drug:A", "control:V1"),
        ("control:V1", "plate:P1"),
        ("drug:B", "control:V1"),
        ("control:V2", "plate:P1"),
    ])
    assert components["drug:A"] == components["drug:B"]


def test_identity_first_split_does_not_depend_on_plate_or_control():
    identity = "cold_drug:INCHIKEY:AAAA"
    assert deterministic_split(identity) == "train"
    assert deterministic_split(identity) == deterministic_split(identity)


def test_vehicle_partition_is_disjoint_deterministic_and_populates_active_splits():
    first = deterministic_vehicle_partition(
        ("P1", "MCF7", "24", "h"),
        (control for control in ["V1", "V2", "V3", "V4"]),
        (split_name for split_name in ["train", "test"]),
    )
    second = deterministic_vehicle_partition(
        ("P1", "MCF7", "24", "h"), ["V4", "V3", "V2", "V1"], ["test", "train"]
    )
    assert first == second
    assert set(first) == {"V1", "V2", "V3", "V4"}
    assert set(first.values()) == {"train", "test"}


def test_vehicle_partition_rejects_insufficient_controls():
    with pytest.raises(ValueError, match="1 controls for 2 active splits"):
        deterministic_vehicle_partition(("P1", "MCF7", "24", "h"), ["V1"], ["train", "test"])


def test_replicate_family_cannot_cross_cold_entity_splits():
    with pytest.raises(ValueError, match="replicate family crosses splits"):
        assert_exclusive_assignments(
            {"drug:A": "train", "drug:B": "test"},
            {"family:bad": ["drug:A", "drug:B"]},
            {"V1": "train", "V2": "test"},
        )


def test_non_degenerate_helpers_distinguish_ids_from_assignments():
    assert not is_non_degenerate(["one-component"])
    assert has_all_splits(["train", "validation", "test"])
    assert not has_all_splits(["train", "test"])
