from __future__ import annotations

import numpy as np
import pytest

from drug_screen.data.exp045_group_permutation import (
    GroupPermutationError,
    permute_dependency_groups,
    verify_dependency_group_permutation,
)


INPUT_SHA = "a" * 64
SPLIT_SHA = "b" * 64
AXIS = ("dep-a", "dep-b", "dep-c")
DATASET_ASSETS = ({"dataset_id": "synthetic-dependency", "version": "1",
                   "sha256": "c" * 64},)
STRATA_VERSION = "synthetic-cancer-library-v1"


def _case():
    targets = np.asarray([
        [1.0, 2.0, np.nan], [3.0, 4.0, np.nan],
        [5.0, 6.0, np.nan], [7.0, 8.0, np.nan],
        [9.0, 10.0, np.nan], [11.0, 12.0, np.nan],
        [13.0, np.nan, 14.0], [15.0, np.nan, 16.0],
    ])
    observation_ids = tuple(f"sample-{i}" for i in range(8))
    groups = ("cell-a", "cell-a", "cell-b", "cell-b", "cell-c", "cell-c",
              "cell-singleton", "cell-singleton")
    strata = ("colorectal|minLib",) * 6 + ("oesophageal|minLib",) * 2
    return targets, observation_ids, groups, strata


def _run(seed: int = 71):
    targets, observations, groups, strata = _case()
    return permute_dependency_groups(
        targets,
        observation_ids=observations,
        biological_group_ids=groups,
        row_strata=strata,
        dependency_axis=AXIS,
        fold_id="outer-2",
        partition_id="outer-2-training",
        stage="depmap_source_training",
        domain="depmap",
        batch_id="depmap-batch-000",
        dataset_assets=DATASET_ASSETS,
        strata_definition_version=STRATA_VERSION,
        input_contract_sha256=INPUT_SHA,
        split_sha256=SPLIT_SHA,
        seed=seed,
        training_mean=1.0,
        training_std=2.0,
    )


def test_whole_groups_shuffle_with_repeats_masks_and_strata_bound_in_receipt():
    targets, observations, groups, strata = _case()
    result = _run()
    mapping = result.receipt["group_mapping"]
    assert set(mapping) == {"cell-a", "cell-b", "cell-c"}
    assert all(recipient != donor for recipient, donor in mapping.items())
    assert result.receipt["excluded_groups"] == {"cell-singleton": "no_compatible_peer"}
    assert result.receipt["n_groups_exchangeable"] == 3
    assert result.receipt["n_groups_excluded"] == 1
    assert np.array_equal(result.mask[:6], np.isfinite(targets[:6]))
    assert not result.mask[6:].any()

    for recipient, donor in mapping.items():
        dst = [index for index, value in enumerate(groups) if value == recipient]
        src = [index for index, value in enumerate(groups) if value == donor]
        dst = sorted(dst, key=lambda index: (strata[index], observations[index]))
        src = sorted(src, key=lambda index: (strata[index], observations[index]))
        assert np.array_equal(result.targets[dst], targets[src], equal_nan=True)

    verify_dependency_group_permutation(
        source_targets=targets,
        output_targets=result.targets,
        output_mask=result.mask,
        observation_ids=observations,
        biological_group_ids=groups,
        row_strata=strata,
        dependency_axis=AXIS,
        expected_fold_id="outer-2",
        expected_partition_id="outer-2-training",
        expected_stage="depmap_source_training",
        expected_domain="depmap",
        expected_batch_id="depmap-batch-000",
        expected_dataset_assets=DATASET_ASSETS,
        expected_strata_definition_version=STRATA_VERSION,
        expected_input_contract_sha256=INPUT_SHA,
        expected_split_sha256=SPLIT_SHA,
        expected_training_mean=1.0,
        expected_training_std=2.0,
        receipt=result.receipt,
        receipt_sha256=result.receipt_sha256,
    )


def test_result_is_deterministic_for_same_contract_and_seed():
    left = _run(seed=72)
    right = _run(seed=72)
    assert left.receipt == right.receipt
    assert left.receipt_sha256 == right.receipt_sha256
    assert np.array_equal(left.targets, right.targets, equal_nan=True)
    assert np.array_equal(left.mask, right.mask)


def test_mask_mismatch_makes_group_unexchangeable_and_removes_it_from_loss():
    targets, observations, groups, strata = _case()
    targets[0, 0] = np.nan
    targets[2, 1] = np.nan
    with pytest.raises(GroupPermutationError, match="no non-trivial compatible"):
        permute_dependency_groups(
            targets,
            observation_ids=observations,
            biological_group_ids=groups,
            row_strata=strata,
            dependency_axis=AXIS,
            fold_id="outer-2",
            partition_id="outer-2-training",
            stage="depmap_source_training",
            domain="depmap",
            batch_id="depmap-batch-000",
            dataset_assets=DATASET_ASSETS,
            strata_definition_version=STRATA_VERSION,
            input_contract_sha256=INPUT_SHA,
            split_sha256=SPLIT_SHA,
            seed=73,
            training_mean=0.0,
            training_std=1.0,
        )


def test_group_exclusion_is_applied_to_shared_dependency_mask():
    targets, observations, groups, strata = _case()
    result = permute_dependency_groups(
        targets,
        observation_ids=observations,
        biological_group_ids=groups,
        row_strata=strata,
        dependency_axis=AXIS,
        fold_id="inner-1",
        partition_id="inner-1-training",
        stage="nature_organoid_adaptation",
        domain="nature",
        batch_id="nature-batch-000",
        dataset_assets=DATASET_ASSETS,
        strata_definition_version=STRATA_VERSION,
        input_contract_sha256=INPUT_SHA,
        split_sha256=SPLIT_SHA,
        seed=74,
        training_mean=0.0,
        training_std=1.0,
        excluded_groups={"cell-singleton": "identity_quarantine"},
    )
    assert result.receipt["excluded_groups"]["cell-singleton"] == "identity_quarantine"
    assert not result.mask[6:].any()
    verify_dependency_group_permutation(
        source_targets=targets,
        output_targets=result.targets,
        output_mask=result.mask,
        observation_ids=observations,
        biological_group_ids=groups,
        row_strata=strata,
        dependency_axis=AXIS,
        expected_fold_id="inner-1",
        expected_partition_id="inner-1-training",
        expected_stage="nature_organoid_adaptation",
        expected_domain="nature",
        expected_batch_id="nature-batch-000",
        expected_dataset_assets=DATASET_ASSETS,
        expected_strata_definition_version=STRATA_VERSION,
        expected_input_contract_sha256=INPUT_SHA,
        expected_split_sha256=SPLIT_SHA,
        expected_training_mean=0.0,
        expected_training_std=1.0,
        receipt=result.receipt,
        receipt_sha256=result.receipt_sha256,
    )


def test_receipt_tampering_or_output_change_fails_recomputation():
    targets, observations, groups, strata = _case()
    result = _run()
    tampered = dict(result.receipt)
    tampered["group_mapping"] = dict(result.receipt["group_mapping"])
    recipient = next(iter(tampered["group_mapping"]))
    tampered["group_mapping"][recipient] = recipient
    with pytest.raises(GroupPermutationError, match="recomputed group mapping"):
        verify_dependency_group_permutation(
            source_targets=targets,
            output_targets=result.targets,
            output_mask=result.mask,
            observation_ids=observations,
            biological_group_ids=groups,
            row_strata=strata,
            dependency_axis=AXIS,
            expected_fold_id="outer-2",
            expected_partition_id="outer-2-training",
            expected_stage="depmap_source_training",
            expected_domain="depmap",
            expected_batch_id="depmap-batch-000",
            expected_dataset_assets=DATASET_ASSETS,
            expected_strata_definition_version=STRATA_VERSION,
            expected_input_contract_sha256=INPUT_SHA,
            expected_split_sha256=SPLIT_SHA,
            expected_training_mean=1.0,
            expected_training_std=2.0,
            receipt=tampered,
            receipt_sha256=result.receipt_sha256,
        )

    changed_output = np.array(result.targets, copy=True)
    changed_output[0, 0] += 1
    with pytest.raises(GroupPermutationError, match="labels or effective mask differ"):
        verify_dependency_group_permutation(
            source_targets=targets,
            output_targets=changed_output,
            output_mask=result.mask,
            observation_ids=observations,
            biological_group_ids=groups,
            row_strata=strata,
            dependency_axis=AXIS,
            expected_fold_id="outer-2",
            expected_partition_id="outer-2-training",
            expected_stage="depmap_source_training",
            expected_domain="depmap",
            expected_batch_id="depmap-batch-000",
            expected_dataset_assets=DATASET_ASSETS,
            expected_strata_definition_version=STRATA_VERSION,
            expected_input_contract_sha256=INPUT_SHA,
            expected_split_sha256=SPLIT_SHA,
            expected_training_mean=1.0,
            expected_training_std=2.0,
            receipt=result.receipt,
            receipt_sha256=result.receipt_sha256,
        )


@pytest.mark.parametrize("mutation", ["duplicate_observation", "unknown_exclusion", "infinite_label"])
def test_invalid_inputs_fail_closed(mutation: str):
    targets, observations, groups, strata = _case()
    kwargs = {}
    if mutation == "duplicate_observation":
        observations = (observations[0], *observations[0:1], *observations[2:])
    elif mutation == "unknown_exclusion":
        kwargs["excluded_groups"] = {"outside-training-fold": "identity_quarantine"}
    else:
        targets[0, 0] = np.inf
    with pytest.raises(GroupPermutationError):
        permute_dependency_groups(
            targets,
            observation_ids=observations,
            biological_group_ids=groups,
            row_strata=strata,
            dependency_axis=AXIS,
            fold_id="outer-2",
            partition_id="outer-2-training",
            stage="depmap_source_training",
            domain="depmap",
            batch_id="depmap-batch-000",
            dataset_assets=DATASET_ASSETS,
            strata_definition_version=STRATA_VERSION,
            input_contract_sha256=INPUT_SHA,
            split_sha256=SPLIT_SHA,
            seed=75,
            training_mean=0.0,
            training_std=1.0,
            **kwargs,
        )
