from __future__ import annotations

import pytest
import numpy as np

from scripts.data.audit_exp045_identity_fold_preflight import (
    PreflightError,
    _mask_summary,
    _mask_summary_for_indices,
    group_exclusion_status,
    select_fraction_inner_folds,
)


def test_select_fraction_inner_folds_uses_five_outer_and_three_inner_slots():
    folds = []
    for outer in range(5):
        inner = [
            {"fraction": fraction, "inner_fold": inner_id}
            for fraction in (0.25, 0.5, 1.0)
            for inner_id in range(3)
        ]
        folds.append({"fold": outer, "inner": inner})

    selected = select_fraction_inner_folds(folds, fraction=0.5)

    assert len(selected) == 15
    assert {(outer, row["inner_fold"]) for outer, row in selected} == {
        (outer, inner) for outer in range(5) for inner in range(3)
    }


def test_repeated_samples_must_be_excluded_as_a_whole_individual_group():
    allowlist = {"person-a": ("sample-a1", "sample-a2"), "person-b": ("sample-b1",)}

    complete = group_exclusion_status(allowlist, ["person-a"], ["sample-a1", "sample-a2"])
    partial = group_exclusion_status(allowlist, ["person-a"], ["sample-a1"])

    assert complete["status"] == "PASS_GROUP_ATOMIC"
    assert complete["excluded_group_count"] == 1
    assert complete["expected_full_group_sample_count"] == 2
    assert partial["status"] == "REPEAT_GROUP_EXCLUSION_INCOMPLETE"


def test_missing_fold_specific_nature_exclusion_does_not_claim_pass():
    result = group_exclusion_status({"person-a": ("sample-a",)}, [], [])
    assert result["status"] == "NO_FOLD_SPECIFIC_EXCLUSION_KEYS"
    assert result["excluded_group_count"] == 0


def test_missing_registered_fraction_or_fold_fails_closed():
    folds = [
        {"fold": fold, "inner": [{"fraction": 1.0, "inner_fold": inner} for inner in range(3)]}
        for fold in range(5)
    ]
    with pytest.raises(PreflightError):
        select_fraction_inner_folds(folds, fraction=0.5)


def test_packed_mask_summary_ignores_padding_bits_after_frozen_axis():
    packed = np.array([[0b10101101, 0b11111111]], dtype=np.uint8)

    summary = _mask_summary(packed, gene_count=10)

    assert summary["object_count"] == 1
    assert summary["gene_positions_per_object"] == 10
    assert summary["valid_bit_count"] == 7
    assert summary["missing_or_nonfinite_bit_count"] == 3
    assert summary["objects_with_any_valid_bit"] == 1
    assert summary["objects_with_all_gene_positions_valid"] == 0


def test_empty_fold_projection_has_zero_mask_bits_not_a_complete_axis():
    packed = np.zeros((2, 2), dtype=np.uint8)

    summary = _mask_summary_for_indices(packed, [], gene_count=10)

    assert summary["object_count"] == 0
    assert summary["valid_bit_count"] == 0
    assert summary["missing_or_nonfinite_bit_count"] == 0
    assert summary["objects_with_all_gene_positions_valid"] == 0
