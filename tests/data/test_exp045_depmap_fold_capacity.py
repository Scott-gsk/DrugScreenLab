from __future__ import annotations

from scripts.data.audit_exp045_depmap_fold_capacity import (
    _build_group_signatures,
    _fold_partition_summary,
    compatibility_summary,
)


def test_group_signatures_use_complete_ordered_stratum_and_mask_profile():
    models = ["m1", "m2", "m3", "m4", "m5"]
    metadata = {
        "m1": ("p1", "CRC"),
        "m2": ("p1", "CRC"),
        "m3": ("p2", "CRC"),
        "m4": ("p3", "CRC"),
        "m5": ("p4", "CRC"),
    }
    profiles = {
        model: (("Library-A", "type"),)
        for model in models
    }
    masks = {
        "m1": b"\x01",
        "m2": b"\x01",
        "m3": b"\x01",
        "m4": b"\x01",
        "m5": b"\x01",
    }
    profiles["m5"] = (("Library-B", "type"),)

    signatures, patient_models, invalid = _build_group_signatures(
        models, masks, metadata, profiles
    )
    result = compatibility_summary(patient_models, signatures)

    # p1 and p2 have the same two-row/single-row observation patterns only if
    # their group sizes match; p1 remains separate from the one-row groups.
    assert result["raw_patient_group_count"] == 4
    assert result["mask_and_profile_complete_group_count"] == 4
    assert result["nonidentity_exchangeable_group_count"] == 2
    assert result["nonidentity_exchangeable_class_count"] == 1
    assert result["pairwise_exchangeable_pair_count"] == 1
    assert invalid["invalid_patient_groups"] == 0
    assert len(patient_models["p1"]) == 2


def test_fold_capacity_closes_manifest_and_candidate_exclusions_by_patient():
    sidecar_keys = {"a", "b", "c", "d"}
    patient_models = {
        "p1": {"a", "b"},
        "p2": {"c"},
        "p3": {"d"},
    }
    metadata = {
        "a": ("p1", "CRC"),
        "b": ("p1", "CRC"),
        "c": ("p2", "CRC"),
        "d": ("p3", "CRC"),
        # This excluded ModelID is absent from the sidecar but still makes p2
        # an excluded whole PatientID group.
        "outside_sidecar": ("p2", "CRC"),
    }
    signatures = {"p1": ("sig-1",), "p2": ("sig-2",), "p3": ("sig-3",)}

    result = _fold_partition_summary(
        level="outer",
        outer_fold=0,
        inner_fold=None,
        excluded_model_ids=["a", "outside_sidecar"],
        unresolved_sanger_ids=["SIDM00117"],
        candidate_ids_by_sanger={"SIDM00117": ["d"], "SIDM00427": [], "SIDM01117": []},
        sidecar_keys=sidecar_keys,
        patient_models=patient_models,
        metadata_by_model=metadata,
        signatures=signatures,
    )

    split = result["frozen_manifest_split_exclusion"]
    assert split["excluded_model_ids_matching_sidecar_keys"] == 1
    assert split["excluded_model_ids_missing_from_sidecar_keys"] == 1
    assert split["patient_group_count_with_sidecar_observations"] == 2
    assert split["split_excluded_sidecar_group_count_with_partial_key_level_membership"] == 1
    group = result["group_partition"]
    assert group["key_level_training_model_id_count_before_patient_group_closure"] == 3
    assert group["patient_group_closed_training_group_count_before_candidate_exclusion"] == 1
    assert group["candidate_patient_group_count_added_to_exclusion_union"] == 1
    assert group["final_patient_group_closed_training_group_count_after_candidate_exclusion"] == 0
    candidate = result["candidate_identity_exclusions"]["per_registered_sanger_id"]["SIDM00117"]
    assert candidate["candidate_model_ids_in_sidecar"] == 1
    assert candidate["candidate_patient_groups_with_sidecar_observations"] == 1
    assert result["unresolved_sanger_ids_registered_in_frozen_slot"]["count"] == 1
    assert result["slot_training_gate"] == "DATA_BLOCKED"


def test_compatibility_capacity_ignores_singleton_and_unusable_groups():
    summary = compatibility_summary(
        {"p1", "p2", "p3", "p4"},
        {"p1": (b"x",), "p2": (b"x",), "p3": (b"z",)},
    )
    assert summary["raw_patient_group_count"] == 4
    assert summary["mask_and_profile_complete_group_count"] == 3
    assert summary["unusable_group_count"] == 1
    assert summary["nonidentity_exchangeable_group_count"] == 2
    assert summary["singleton_class_count"] == 1
    assert summary["compatibility_class_size_histogram"] == {"1": 1, "2": 1}
