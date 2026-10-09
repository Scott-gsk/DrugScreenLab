from __future__ import annotations

import numpy as np
import pytest

from scripts.data.audit_exp045_nature_fold_capacity import (
    CapacityAuditError,
    build_group_signatures,
    compatibility_summary,
    selected_inner_folds,
)


def test_whole_individual_signatures_use_library_and_exact_packed_mask_only():
    packed = np.asarray(
        [
            [0b00000101],
            [0b00000101],
            [0b00000101],
            [0b00000111],
            [0b00000101],
        ],
        dtype=np.uint8,
    )
    signatures, info = build_group_signatures(
        sample_to_group={
            "a-sample": "a",
            "b-sample": "b",
            "c-sample": "c",
            "d-sample-1": "d",
            "d-sample-2": "d",
        },
        sidecar_sample_ids=["a-sample", "b-sample", "c-sample", "d-sample-1", "d-sample-2"],
        sidecar_libraries=["minLib", "minLib", "v1_1", "minLib", "minLib"],
        packed_masks=packed,
        excluded_group_ids={"c"},
    )

    assert set(signatures) == {"a", "b", "d"}
    assert signatures["a"] == signatures["b"]
    assert signatures["d"][0] == 2
    assert signatures["a"] != signatures["d"]
    assert info["excluded_quarantine_group_count_present_in_sidecar"] == 1
    assert info["unmapped_sample_count"] == 0


def test_compatibility_summary_counts_only_exact_whole_group_classes():
    summary = compatibility_summary(
        {
            "a": ("one-sample", "minLib", b"mask-1"),
            "b": ("one-sample", "minLib", b"mask-1"),
            "c": ("one-sample", "v1_1", b"mask-1"),
        }
    )

    assert summary["valid_group_count"] == 3
    assert summary["compatibility_class_count"] == 2
    assert summary["nonidentity_exchangeable_group_count"] == 2
    assert summary["class_size_histogram"] == {"1": 1, "2": 1}
    assert summary["nonidentity_group_permutation_count_decimal"] == "1"


def test_selected_inner_folds_requires_five_outer_and_three_half_fraction_folds():
    folds = [
        {
            "fold": outer,
            "inner": [
                {"fraction": fraction, "inner_fold": inner}
                for fraction in (0.25, 0.5, 1.0)
                for inner in range(3)
            ],
        }
        for outer in range(5)
    ]

    selected = selected_inner_folds(folds)

    assert len(selected) == 15
    assert {(outer, inner["inner_fold"]) for outer, inner in selected} == {
        (outer, inner) for outer in range(5) for inner in range(3)
    }


def test_selected_inner_folds_rejects_missing_half_fraction_entry():
    folds = [
        {
            "fold": outer,
            "inner": [
                {"fraction": fraction, "inner_fold": inner}
                for fraction in (0.25, 0.5, 1.0)
                for inner in range(3)
                if not (outer == 0 and fraction == 0.5 and inner == 2)
            ],
        }
        for outer in range(5)
    ]

    with pytest.raises(CapacityAuditError):
        selected_inner_folds(folds)
