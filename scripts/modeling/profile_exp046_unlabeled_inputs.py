"""Profile registered EXP-046 inputs without opening response-label files."""

from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import platform
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RUN_ID = "20260924_unlabeled_input_profile_v1"
INPUT_RELATIVE_PATHS = {
    "gene_axis": Path("data/processed/dual_route/EXP-036/direct_response_contract_v1/genes.json"),
    "contract_manifest": Path("data/processed/dual_route/EXP-036/direct_response_contract_v1/manifest.json"),
    "patient_expression": Path("data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1/patient_expression.npy"),
    "patient_axis": Path("data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1/patient_axis.json"),
    "drug_codes": Path("data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1/drug_ordinal128.npy"),
    "drug_axis": Path("data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1/drug_axis.parquet"),
    "input_manifest": Path("data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1/manifest.json"),
    "frozen_split": Path("data/splits/dual_route_splits_20260922_v1.json"),
}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _open_pinned_directory(path: Path) -> int:
    """Open an absolute directory path component by component without symlinks."""
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    absolute = Path(os.path.abspath(path))
    current_fd = os.open(absolute.anchor, directory_flags)
    try:
        for component in absolute.parts[1:]:
            next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            try:
                if not stat.S_ISDIR(os.fstat(next_fd).st_mode):
                    raise ValueError("profile root contains a non-directory path component")
            except BaseException:
                os.close(next_fd)
                raise
            previous_fd = current_fd
            current_fd = next_fd
            os.close(previous_fd)
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _read_allowlisted_blob(root_fd: int, relative: Path) -> bytes:
    """Read below a pinned root fd, refusing symlinks at every path component."""
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current_fd = root_fd
    try:
        for component in relative.parts[:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            try:
                if not stat.S_ISDIR(os.fstat(next_fd).st_mode):
                    raise ValueError("registered unlabeled input contains a non-directory component")
            except BaseException:
                os.close(next_fd)
                raise
            previous_fd = current_fd
            current_fd = next_fd
            if previous_fd != root_fd:
                os.close(previous_fd)

        file_flags = os.O_RDONLY | os.O_NOFOLLOW
        descriptor = os.open(relative.parts[-1], file_flags, dir_fd=current_fd)
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError("registered unlabeled input is not a regular file")
            if file_stat.st_nlink != 1:
                raise ValueError("registered unlabeled input has an unexpected hard-link count")
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = -1
                return stream.read()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    finally:
        if current_fd != root_fd:
            os.close(current_fd)


def _read_and_validate(root_fd: int) -> tuple[dict[str, Any], dict[str, float]]:
    read_start = time.perf_counter()
    blobs: dict[str, bytes] = {}
    for name, relative in INPUT_RELATIVE_PATHS.items():
        try:
            blobs[name] = _read_allowlisted_blob(root_fd, relative)
        except FileNotFoundError:
            raise FileNotFoundError(f"registered unlabeled input is missing: {name}") from None
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ValueError(
                    f"registered unlabeled input {name} traverses a symbolic link or non-directory"
                ) from None
            raise
    read_seconds = time.perf_counter() - read_start

    hash_start = time.perf_counter()
    hashes = {name: _sha256(blob) for name, blob in blobs.items()}
    hash_seconds = time.perf_counter() - hash_start

    decode_start = time.perf_counter()
    genes = json.loads(blobs["gene_axis"].decode("utf-8"))
    contract_manifest = json.loads(blobs["contract_manifest"].decode("utf-8"))
    patient_axis = json.loads(blobs["patient_axis"].decode("utf-8"))
    manifest = json.loads(blobs["input_manifest"].decode("utf-8"))
    split = json.loads(blobs["frozen_split"].decode("utf-8"))
    expression_raw = np.load(io.BytesIO(blobs["patient_expression"]), allow_pickle=False)
    drug_codes_raw = np.load(io.BytesIO(blobs["drug_codes"]), allow_pickle=False)
    drug_axis = pd.read_parquet(
        io.BytesIO(blobs["drug_axis"]), columns=["drug_id", "drug_index"]
    )
    decode_seconds = time.perf_counter() - decode_start

    preprocess_start = time.perf_counter()
    if (not isinstance(genes, list) or not genes
            or any(not isinstance(value, str) or not value for value in genes)
            or len(set(genes)) != len(genes)):
        raise ValueError("ordered gene axis is malformed")
    if (not isinstance(patient_axis, list) or not patient_axis
            or any(not isinstance(value, str) or not value for value in patient_axis)
            or len(set(patient_axis)) != len(patient_axis)):
        raise ValueError("patient axis is malformed")
    drug_ids = drug_axis["drug_id"].astype(str).tolist()
    raw_drug_indices = drug_axis["drug_index"].tolist()
    if any(isinstance(value, (bool, np.bool_))
           or not isinstance(value, (int, np.integer))
           for value in raw_drug_indices):
        raise ValueError("drug code axis row indices must be integer-valued")
    drug_indices = [int(value) for value in raw_drug_indices]
    if (not drug_ids or len(set(drug_ids)) != len(drug_ids)
            or drug_indices != list(range(len(drug_ids)))):
        raise ValueError("drug code axis has duplicate identifiers or invalid row order")

    expression = np.array(expression_raw, dtype=np.float32, order="C", copy=True)
    drug_codes = np.array(drug_codes_raw, dtype=np.float32, order="C", copy=True)
    if (expression.shape != (len(patient_axis), len(genes))
            or drug_codes.shape != (len(drug_ids), 128)
            or not np.isfinite(expression).all()
            or not np.isfinite(drug_codes).all()):
        raise ValueError("unlabeled expression/drug arrays are non-finite or axis-mismatched")

    files_sha = manifest.get("files_sha256", {})
    for name, file_name in (
        ("patient_expression", "patient_expression.npy"),
        ("patient_axis", "patient_axis.json"),
        ("drug_codes", "drug_ordinal128.npy"),
        ("drug_axis", "drug_axis.parquet"),
    ):
        if files_sha.get(file_name) != hashes[name]:
            raise ValueError(f"registered input manifest hash mismatch for {file_name}")
    if manifest.get("contract_manifest_sha256") != hashes["contract_manifest"]:
        raise ValueError("registered input contract manifest hash mismatch")
    if contract_manifest.get("files_sha256", {}).get("genes.json") != hashes["gene_axis"]:
        raise ValueError("ordered gene axis does not match the registered contract manifest")

    target_folds = split.get("target_folds")
    if not isinstance(target_folds, list) or len(target_folds) != 5:
        raise ValueError("frozen target split must contain exactly five outer folds")
    cohort = set(patient_axis)
    test_union: set[str] = set()
    fold_counts = []
    for fold in target_folds:
        train_ids = fold.get("outer_train_ids")
        test_ids = fold.get("outer_test_ids")
        if (not isinstance(train_ids, list) or not train_ids
                or not isinstance(test_ids, list) or not test_ids):
            raise ValueError("frozen target fold requires non-empty train/test patient group keys")
        train_set, test_set = set(train_ids), set(test_ids)
        if (len(train_set) != len(train_ids) or len(test_set) != len(test_ids)
                or train_set.intersection(test_set)
                or train_set != cohort.difference(test_set)
                or test_union.intersection(test_set)):
            raise ValueError("frozen target patient folds fail the unlabeled partition contract")
        test_union.update(test_set)
        fold_counts.append({"train_patient_count": len(train_set),
                            "test_patient_count": len(test_set)})
    if test_union != cohort:
        raise ValueError("frozen outer-test groups do not cover the expression patient axis")

    summary = {
        "n_genes": len(genes),
        "n_patients": len(patient_axis),
        "n_drugs": len(drug_ids),
        "expression_shape": list(expression.shape),
        "drug_codes_shape": list(drug_codes.shape),
        "outer_fold_patient_counts": fold_counts,
        "expression_finite": True,
        "drug_codes_finite": True,
        "split_patient_partition_valid": True,
    }
    preprocess_seconds = time.perf_counter() - preprocess_start
    summary["input_sha256"] = hashes
    summary["input_bytes"] = {name: len(blob) for name, blob in blobs.items()}
    summary["unlabeled_input_identity"] = {
        "patient_axis_sha256": hashes["patient_axis"],
        "drug_axis_sha256": hashes["drug_axis"],
        "gene_axis_sha256": hashes["gene_axis"],
    }
    return summary, {
        "file_read_seconds": read_seconds,
        "sha256_seconds": hash_seconds,
        "decode_seconds": decode_seconds,
        "cpu_preprocess_and_contract_seconds": preprocess_seconds,
    }


def _environment() -> dict[str, str]:
    release = platform.release().lower()
    if (platform.system() != "Linux"
            or not ("microsoft" in release or os.environ.get("WSL_DISTRO_NAME"))):
        raise RuntimeError("this profile must run inside WSL2 Linux")
    if os.environ.get("CONDA_DEFAULT_ENV") != "drugscreening-gpu":
        raise RuntimeError("this profile requires Conda environment drugscreening-gpu")
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "conda_env": os.environ["CONDA_DEFAULT_ENV"],
        "device_used": "CPU only; no CUDA context created",
    }


def profile_unlabeled_inputs(root: Path, output_root: Path) -> dict[str, Any]:
    """Read only the explicit allowlist, validate axes/splits, and write one receipt."""
    root = Path(os.path.abspath(root))
    output_root = Path(output_root)
    output_dir = output_root / RUN_ID
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite profile output {RUN_ID}")
    environment = _environment()
    started = datetime.now(timezone.utc)
    wall_start = time.perf_counter()
    root_fd = _open_pinned_directory(root)
    try:
        summary, stages = _read_and_validate(root_fd)
    finally:
        os.close(root_fd)
    finished = datetime.now(timezone.utc)
    receipt = {
        "experiment_id": "EXP-046",
        "run_id": RUN_ID,
        "status": "UNLABELED_INPUT_STAGE_PROFILE_ONLY",
        "runtime": environment,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "profiled_stages_seconds": stages,
        "measured_summary": summary,
        "training_performed": False,
        "labels_read": False,
        "response_label_files_opened": [],
        "gpu_hours": 0.0,
        "not_measured": ["host_to_device_transfer", "forward_backward", "cpu_validation",
                         "prediction_or_checkpoint_write", "source_stage", "full_five_arm_cost"],
        "scope_limit": (
            "One read of the registered unlabeled input allowlist; filesystem cache state is uncontrolled. "
            "These stage times do not estimate a full training run or authorize model selection."
        ),
    }
    wall_seconds = time.perf_counter() - wall_start
    receipt["profile_wall_seconds_excluding_receipt_write"] = wall_seconds
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "receipt.json"
    payload = json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    write_start = time.perf_counter()
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    write_seconds = time.perf_counter() - write_start
    return {"receipt": receipt, "receipt_path": output_path.as_posix(),
            "receipt_sha256": _sha256(output_path.read_bytes()),
            "receipt_write_seconds": write_seconds}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    result = profile_unlabeled_inputs(root, root / "artifacts/experiments/EXP-046")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
