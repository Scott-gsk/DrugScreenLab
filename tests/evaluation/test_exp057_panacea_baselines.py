"""Synthetic checks for fixed EXP-057 PANACEA baseline scoring contracts."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd
import pytest

from scripts.evaluation import evaluate_panacea_external_development_baselines as runner


def test_train_mean_baselines_match_hand_calculation() -> None:
    y = np.asarray([[1.0, 3.0], [3.0, 5.0], [5.0, 7.0], [7.0, 9.0]])
    keys = np.asarray(["A", "A", "B", "B"])
    global_mean, key_mean, n_key = runner._compute_train_means(y, keys, "A")
    np.testing.assert_allclose(global_mean, [4.0, 6.0], rtol=0, atol=0)
    np.testing.assert_allclose(key_mean, [2.0, 4.0], rtol=0, atol=0)
    assert n_key == 2


def test_source_numeric_gene_labels_resolve_through_frozen_author_map() -> None:
    source_labels = ["0", "1", "2"]
    mapping = pd.DataFrame({
        "axis_index": ["0", "1", "2"],
        "h5ad_var_name": ["0", "1", "2"],
        "h5ad_var_annotation": ["202", "101", "303"],
        "l1000_gene_info_value": ["202", "101", "303"],
    })
    gene_info = pd.DataFrame({
        "gene_id": ["101", "202", "303"],
        "gene_symbol": ["GENE_A", "GENE_B", "GENE_C"],
    })
    reindex, alignment, evidence = runner._resolve_source_gene_reindex(
        source_labels, mapping, gene_info, ["GENE_A", "GENE_B", "GENE_C"]
    )

    np.testing.assert_array_equal(reindex, [1, 0, 2])
    assert alignment["gene_symbol"].tolist() == ["GENE_A", "GENE_B", "GENE_C"]
    assert alignment["source_npz_gene_label"].tolist() == ["1", "0", "2"]
    assert evidence["gene_symbol_membership_matches_target"] is True
    assert evidence["gene_symbol_order_already_matches_target"] is False


def test_source_train_loader_reorders_only_train_values_into_symbol_axis(tmp_path) -> None:
    condition_keys = [f"COND_{i:03d}" for i in range(419)]
    key_table = pd.DataFrame({
        "source_object_id": [f"OBJ_{i % 110:03d}" for i in range(419)],
        "full_inchikey": [f"KEY_{i % 5}" for i in range(419)],
        "pert_id": [f"DRUG_{i % 5}" for i in range(419)],
        "split_role": ["train"] * 419,
        "full_condition_key": condition_keys,
    })
    key_path = tmp_path / "conditions.tsv"
    key_table.to_csv(key_path, sep="\t", index=False)

    source_rows = np.arange(419, dtype=np.float64)
    source_delta = np.column_stack((source_rows, 2 * source_rows, -source_rows))
    values_path = tmp_path / "source.npz"
    np.savez_compressed(
        values_path,
        train_condition_key=np.asarray(condition_keys, dtype=str),
        genes=np.asarray(["0", "1", "2"], dtype=str),
        train_target_delta=source_delta,
        inner_dev_target_delta=np.full((2, 3), 9.0),
    )

    mapping = pd.DataFrame({
        "axis_index": ["0", "1", "2"],
        "h5ad_var_name": ["0", "1", "2"],
        "h5ad_var_annotation": ["202", "101", "303"],
        "l1000_gene_info_value": ["202", "101", "303"],
    })
    mapping_path = tmp_path / "gene_axis_mapping.tsv"
    mapping.to_csv(mapping_path, sep="\t", index=False)
    gene_info = pd.DataFrame({
        "gene_id": ["101", "202", "303"],
        "gene_symbol": ["GENE_A", "GENE_B", "GENE_C"],
    })
    gene_info_path = tmp_path / "gene_info.csv"
    gene_info.to_csv(gene_info_path, index=False)
    receipt_path = tmp_path / "gene_axis_receipt.json"
    receipt_path.write_text(json.dumps({
        "files_sha256": {"gene_axis_mapping.tsv": runner.sha256(mapping_path)},
        "mapping_rows": 3,
        "mapping_proof": "exact row-by-row shared-column match",
        "matched_columns": {"reference": "gene_id"},
        "X_read": False,
        "X_ctl_read": False,
    }), encoding="utf-8")

    global_mean, key_mean, ordered_meta, evidence, alignment = runner._load_source_train_means(
        values_path,
        key_path,
        ["GENE_A", "GENE_B", "GENE_C"],
        "KEY_0",
        mapping_path,
        receipt_path,
        gene_info_path,
        runner.sha256(mapping_path),
        runner.sha256(gene_info_path),
    )

    np.testing.assert_allclose(global_mean, [418.0, 209.0, -209.0])
    np.testing.assert_allclose(key_mean, [415.0, 207.5, -207.5])
    assert len(ordered_meta) == 419
    assert evidence["numeric_array_names_loaded"] == ["train_condition_key", "genes", "train_target_delta"]
    assert evidence["inner_dev_or_other_role_numeric_arrays_loaded"] is False
    assert alignment["source_column_position"].tolist() == [1, 0, 2]


def test_direction_rank_and_hallmark_metrics_match_hand_calculation() -> None:
    target = np.asarray([1.0, 0.0, -2.0, 4.0])
    prediction = np.asarray([3.0, 10.0, -1.0, 0.0])
    sets = {"SET_A": np.asarray([0, 1]), "SET_B": np.asarray([2, 3]), "TARGET_ZERO": np.asarray([1])}
    scores, gene_rows, hallmark_rows = runner._score_condition(target, prediction, sets)

    assert scores["gene_direction_numerator"] == 2
    assert scores["gene_direction_denominator_target_nonzero"] == 3
    assert scores["gene_direction_agreement"] == pytest.approx(2 / 3)
    assert scores["gene_spearman"] == pytest.approx(0.2)
    assert scores["gene_spearman_status"] == "DEFINED"
    assert scores["hallmark_direction_numerator"] == 1
    assert scores["hallmark_direction_denominator_target_nonzero"] == 2
    assert scores["hallmark_direction_agreement"] == pytest.approx(0.5)
    assert len(gene_rows) == 4
    assert len(hallmark_rows) == 3


def test_zero_change_keeps_constant_rank_undefined_and_counts_no_direction() -> None:
    target = np.asarray([1.0, 0.0, -2.0, 4.0])
    prediction = np.zeros_like(target)
    sets = {"SET_A": np.asarray([0, 1]), "SET_B": np.asarray([2, 3]), "TARGET_ZERO": np.asarray([1])}
    scores, _, _ = runner._score_condition(target, prediction, sets)

    assert scores["gene_direction_agreement"] == 0.0
    assert scores["gene_direction_denominator_target_nonzero"] == 3
    assert scores["gene_spearman"] is None
    assert scores["gene_spearman_status"] == "UNDEFINED_CONSTANT_PREDICTION_VECTOR"
    assert scores["hallmark_direction_agreement"] == 0.0
    assert scores["hallmark_direction_denominator_target_nonzero"] == 2
    assert scores["hallmark_prediction_nonzero_set_count"] == 0


def test_selected_dense_reader_preserves_key_order_and_reads_only_requested_rows(tmp_path) -> None:
    expected = np.arange(20, dtype=np.float64).reshape(5, 4)
    path = tmp_path / "toy.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("X", data=expected)
    with h5py.File(path, "r") as h5:
        actual = runner._read_dense_selected_rows(h5["X"], np.asarray([3, 1]), expected_cols=4)
    np.testing.assert_array_equal(actual, expected[[3, 1]])


def test_run_gate_failure_propagates_before_numeric_access() -> None:
    metadata_gate = {"metadata_gate_all_pass": False, "run_id": "synthetic"}
    frozen = {"status": "FROZEN_PENDING_ROOT_INDEPENDENT_READBACK", "run_id": "synthetic"}
    root_readback = {
        "status": "ROOT_INDEPENDENT_METADATA_READBACK_PASS_SYNTHETIC",
        "run_id": "synthetic",
        "independent_assertions_pass": True,
        "data_boundary": {
            "metadata_numeric_matrix_accessed": False,
            "h5ad_X_values_accessed": False,
            "h5ad_layers_values_accessed": False,
            "matched_control_values_accessed": False,
            "pdo_outer_locked_values_accessed": False,
        },
    }
    with pytest.raises(RuntimeError, match="numeric run blocked: metadata-only gate did not pass"):
        runner._require_run_gate_ready(metadata_gate, frozen, root_readback)


def test_standard_root_readback_schema_passes_only_with_all_false_boundaries() -> None:
    metadata_gate = {"metadata_gate_all_pass": True, "run_id": "synthetic"}
    frozen = {"status": "FROZEN_PENDING_ROOT_INDEPENDENT_READBACK", "run_id": "synthetic"}
    root_readback = {
        "status": "ROOT_INDEPENDENT_METADATA_READBACK_PASS_SYNTHETIC",
        "run_id": "synthetic",
        "independent_assertions_pass": True,
        "data_boundary": {
            "metadata_numeric_matrix_accessed": False,
            "h5ad_X_values_accessed": False,
            "h5ad_layers_values_accessed": False,
            "matched_control_values_accessed": False,
            "pdo_outer_locked_values_accessed": False,
        },
    }
    runner._require_run_gate_ready(metadata_gate, frozen, root_readback)

    root_readback["data_boundary"]["matched_control_values_accessed"] = True
    with pytest.raises(RuntimeError, match="numeric-access flags are false"):
        runner._require_run_gate_ready(metadata_gate, frozen, root_readback)


def test_paired_bootstrap_difference_is_object_paired() -> None:
    left = np.asarray([0.8, 0.4, 0.6])
    right = np.asarray([0.5, 0.4, 0.1])
    point, low, high, n = runner._paired_bootstrap_difference(left, right, seed=13, replicates=1000)
    assert point == pytest.approx((0.3 + 0.0 + 0.5) / 3)
    assert low is not None and high is not None and low <= point <= high
    assert n == 3
