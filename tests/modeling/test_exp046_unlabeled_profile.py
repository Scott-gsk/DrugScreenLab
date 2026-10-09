"""Contract tests for the label-free EXP-046 input profiling boundary."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.modeling.profile_exp046_unlabeled_inputs as profiler
from scripts.modeling.profile_exp046_unlabeled_inputs import (
    INPUT_RELATIVE_PATHS,
    RUN_ID,
    profile_unlabeled_inputs,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_input_manifest(root: Path) -> None:
    inputs = root / "data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1"
    manifest_path = inputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files_sha256"] = {
        name: _sha(inputs / name)
        for name in ("patient_expression.npy", "patient_axis.json",
                     "drug_ordinal128.npy", "drug_axis.parquet")
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _write_fixture(root: Path) -> None:
    contract = root / "data/processed/dual_route/EXP-036/direct_response_contract_v1"
    inputs = root / "data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1"
    split_dir = root / "data/splits"
    contract.mkdir(parents=True)
    inputs.mkdir(parents=True)
    split_dir.mkdir(parents=True)

    genes = [f"GENE{index}" for index in range(3)]
    patients = [f"PATIENT{index}" for index in range(5)]
    drugs = [f"DRUG{index}" for index in range(2)]
    (contract / "genes.json").write_text(json.dumps(genes), encoding="utf-8")
    contract_manifest = {"files_sha256": {"genes.json": _sha(contract / "genes.json")}}
    (contract / "manifest.json").write_text(json.dumps(contract_manifest), encoding="utf-8")
    np.save(inputs / "patient_expression.npy", np.arange(15, dtype=np.float32).reshape(5, 3))
    (inputs / "patient_axis.json").write_text(json.dumps(patients), encoding="utf-8")
    np.save(inputs / "drug_ordinal128.npy", np.ones((2, 128), dtype=np.float32))
    pd.DataFrame({"drug_id": drugs, "drug_index": range(len(drugs))}).to_parquet(
        inputs / "drug_axis.parquet", index=False)

    files = ["patient_expression.npy", "patient_axis.json",
             "drug_ordinal128.npy", "drug_axis.parquet"]
    manifest = {"contract_manifest_sha256": _sha(contract / "manifest.json"),
                "files_sha256": {name: _sha(inputs / name) for name in files}}
    (inputs / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    folds = []
    for heldout in patients:
        test_ids = [heldout]
        train_ids = [patient for patient in patients if patient != heldout]
        folds.append({"outer_train_ids": train_ids, "outer_test_ids": test_ids})
    (split_dir / "dual_route_splits_20260922_v1.json").write_text(
        json.dumps({"target_folds": folds}), encoding="utf-8")


def test_profile_reads_only_unlabeled_allowlist_and_never_echoes_axis_values(tmp_path):
    root = tmp_path / "workspace"
    _write_fixture(root)
    assert "patient_pairs" not in " ".join(
        path.as_posix() for path in INPUT_RELATIVE_PATHS.values())
    result = profile_unlabeled_inputs(root, tmp_path / "out")

    assert result["receipt"]["status"] == "UNLABELED_INPUT_STAGE_PROFILE_ONLY"
    assert result["receipt"]["labels_read"] is False
    assert result["receipt"]["gpu_hours"] == 0.0
    assert result["receipt"]["measured_summary"]["split_patient_partition_valid"] is True
    assert result["receipt"]["measured_summary"]["n_patients"] == 5
    assert result["receipt_write_seconds"] >= 0
    serialized = Path(result["receipt_path"]).read_text(encoding="utf-8")
    assert "PATIENT0" not in serialized
    assert "DRUG0" not in serialized
    assert "GENE0" not in serialized


def test_profile_fails_closed_on_overlapping_outer_test_groups(tmp_path):
    root = tmp_path / "workspace"
    _write_fixture(root)
    split_path = root / "data/splits/dual_route_splits_20260922_v1.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["target_folds"][1]["outer_test_ids"] = ["PATIENT0"]
    split_path.write_text(json.dumps(split), encoding="utf-8")

    with pytest.raises(ValueError, match="partition contract"):
        profile_unlabeled_inputs(root, tmp_path / "out")
    assert not (tmp_path / "out" / RUN_ID).exists()


def test_profile_refuses_to_overwrite_existing_run_receipt(tmp_path):
    root = tmp_path / "workspace"
    _write_fixture(root)
    output = tmp_path / "out"
    (output / RUN_ID).mkdir(parents=True)

    with pytest.raises(FileExistsError, match="overwrite"):
        profile_unlabeled_inputs(root, output)


def test_profile_rejects_symlink_in_allowlisted_input_path(tmp_path):
    root = tmp_path / "workspace"
    _write_fixture(root)
    contract = root / "data/processed/dual_route/EXP-036/direct_response_contract_v1"
    genes_path = contract / "genes.json"
    genes_path.unlink()
    outside = tmp_path / "outside-genes.json"
    outside.write_text(json.dumps(["OUTSIDE"]), encoding="utf-8")
    genes_path.symlink_to(outside)

    with pytest.raises(ValueError, match="symbolic link"):
        profile_unlabeled_inputs(root, tmp_path / "out")
    assert not (tmp_path / "out" / RUN_ID).exists()


def test_allowlisted_read_stays_on_pinned_directory_if_path_is_replaced(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    _write_fixture(root)
    contract = root / "data/processed/dual_route/EXP-036/direct_response_contract_v1"
    original_genes = (contract / "genes.json").read_bytes()
    parked = contract.with_name("direct_response_contract_v1_pinned")
    outside = tmp_path / "outside-contract"
    outside.mkdir()
    (outside / "genes.json").write_text(json.dumps(["OUTSIDE"]), encoding="utf-8")
    real_open = profiler.os.open
    replaced = False

    def replace_after_directory_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal replaced
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == "direct_response_contract_v1" and not replaced:
            contract.rename(parked)
            contract.symlink_to(outside, target_is_directory=True)
            replaced = True
        return descriptor

    monkeypatch.setattr(profiler.os, "open", replace_after_directory_open)

    root_fd = profiler._open_pinned_directory(root)
    try:
        actual = profiler._read_allowlisted_blob(root_fd, INPUT_RELATIVE_PATHS["gene_axis"])
    finally:
        os.close(root_fd)

    assert replaced is True
    assert actual == original_genes


def test_root_directory_replacement_does_not_redirect_pinned_read(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    _write_fixture(root)
    original_genes = (root / INPUT_RELATIVE_PATHS["gene_axis"]).read_bytes()
    parked = tmp_path / "workspace-pinned"
    outside = tmp_path / "outside-workspace"
    outside.mkdir()
    (outside / "data").mkdir()
    real_open = profiler.os.open
    replaced = False

    def replace_after_root_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal replaced
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == "workspace" and not replaced:
            root.rename(parked)
            root.symlink_to(outside, target_is_directory=True)
            replaced = True
        return descriptor

    monkeypatch.setattr(profiler.os, "open", replace_after_root_open)
    root_fd = profiler._open_pinned_directory(root)
    try:
        actual = profiler._read_allowlisted_blob(root_fd, INPUT_RELATIVE_PATHS["gene_axis"])
    finally:
        os.close(root_fd)

    assert replaced is True
    assert actual == original_genes


def test_root_walk_closes_new_descriptor_if_fstat_fails(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    _write_fixture(root)
    real_fstat = profiler.os.fstat
    opened_descriptors = []

    def fail_fstat(descriptor):
        opened_descriptors.append(descriptor)
        raise OSError("injected fstat failure")

    monkeypatch.setattr(profiler.os, "fstat", fail_fstat)
    with pytest.raises(OSError, match="injected fstat failure"):
        profiler._open_pinned_directory(root)

    assert len(opened_descriptors) == 1
    with pytest.raises(OSError):
        real_fstat(opened_descriptors[0])


def test_allowlisted_walk_closes_new_descriptor_if_fstat_fails(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    _write_fixture(root)
    root_fd = profiler._open_pinned_directory(root)
    real_fstat = profiler.os.fstat
    opened_descriptors = []

    def fail_fstat(descriptor):
        opened_descriptors.append(descriptor)
        raise OSError("injected fstat failure")

    monkeypatch.setattr(profiler.os, "fstat", fail_fstat)
    try:
        with pytest.raises(OSError, match="injected fstat failure"):
            profiler._read_allowlisted_blob(root_fd, INPUT_RELATIVE_PATHS["gene_axis"])
        assert len(opened_descriptors) == 1
        with pytest.raises(OSError):
            real_fstat(opened_descriptors[0])
    finally:
        os.close(root_fd)


def test_allowlisted_walk_tracks_next_descriptor_before_old_close_error(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    _write_fixture(root)
    root_fd = profiler._open_pinned_directory(root)
    real_open = profiler.os.open
    real_close = profiler.os.close
    real_fstat = profiler.os.fstat
    opened_descriptors = []
    injected = False

    def track_open(path, flags, mode=0o777, *, dir_fd=None):
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        opened_descriptors.append(descriptor)
        return descriptor

    def fail_after_closing_old_descriptor(descriptor):
        nonlocal injected
        if opened_descriptors and descriptor == opened_descriptors[0] and not injected:
            injected = True
            real_close(descriptor)
            raise OSError("injected close error")
        real_close(descriptor)

    monkeypatch.setattr(profiler.os, "open", track_open)
    monkeypatch.setattr(profiler.os, "close", fail_after_closing_old_descriptor)
    try:
        with pytest.raises(OSError, match="injected close error"):
            profiler._read_allowlisted_blob(root_fd, INPUT_RELATIVE_PATHS["gene_axis"])
        assert injected is True
        assert len(opened_descriptors) >= 2
        for descriptor in opened_descriptors[:2]:
            with pytest.raises(OSError):
                real_fstat(descriptor)
        real_fstat(root_fd)
    finally:
        real_close(root_fd)


def test_profile_rejects_hardlinked_allowlisted_asset(tmp_path):
    root = tmp_path / "workspace"
    _write_fixture(root)
    patient_axis = root / INPUT_RELATIVE_PATHS["patient_axis"]
    outside = tmp_path / "outside-patient-axis.json"
    outside.write_bytes(patient_axis.read_bytes())
    patient_axis.unlink()
    os.link(outside, patient_axis)
    _refresh_input_manifest(root)

    with pytest.raises(ValueError, match="hard-link count"):
        profile_unlabeled_inputs(root, tmp_path / "out")
    assert not (tmp_path / "out" / RUN_ID).exists()


def test_profile_rejects_fractional_drug_index_instead_of_truncating(tmp_path):
    root = tmp_path / "workspace"
    _write_fixture(root)
    inputs = root / "data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1"
    pd.DataFrame({"drug_id": ["DRUG0", "DRUG1"], "drug_index": [0.0, 1.5]}).to_parquet(
        inputs / "drug_axis.parquet", index=False)
    _refresh_input_manifest(root)

    with pytest.raises(ValueError, match="integer-valued"):
        profile_unlabeled_inputs(root, tmp_path / "out")
    assert not (tmp_path / "out" / RUN_ID).exists()


def test_profile_rejects_empty_outer_test_fold(tmp_path):
    root = tmp_path / "workspace"
    _write_fixture(root)
    split_path = root / "data/splits/dual_route_splits_20260922_v1.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["target_folds"][0]["outer_test_ids"] = []
    split_path.write_text(json.dumps(split), encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty train/test"):
        profile_unlabeled_inputs(root, tmp_path / "out")
    assert not (tmp_path / "out" / RUN_ID).exists()
