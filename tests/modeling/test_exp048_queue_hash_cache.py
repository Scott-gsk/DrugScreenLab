import hashlib
import json
import os

import pytest

from run_exp048_fivefold_queue import (
    ARMS,
    SEEDS,
    FUSION_CANDIDATES,
    prepare_fold_output_dir,
    sha256,
)


def test_sha256_reuses_unchanged_file_and_rehashes_changed_file(tmp_path):
    path = tmp_path / "checkpoint.bin"
    path.write_bytes(b"first version")

    first = sha256(path)
    assert first == hashlib.sha256(b"first version").hexdigest()
    assert sha256(path) == first

    previous = path.stat()
    path.write_bytes(b"second version")
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000))
    second = sha256(path)
    assert second == hashlib.sha256(b"second version").hexdigest()
    assert second != first


def test_prepare_fold_output_dir_writes_matching_manifest_and_final_dirs(tmp_path):
    run_dir = tmp_path / "fold_3"
    trainer_hashes = {"config": "c", "split": "s", "fold_runner": "f"}

    prepare_fold_output_dir(run_dir, 3, trainer_hashes)

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["fold"] == 3
    assert manifest["candidate_grid"] == [vars(item) for item in FUSION_CANDIDATES]
    assert manifest["arms"] == ARMS
    assert manifest["seeds"] == SEEDS
    assert manifest["input_and_code_sha256"] == trainer_hashes
    assert all((run_dir / "units" / "final" / arm).is_dir() for arm in ARMS)

    prepare_fold_output_dir(run_dir, 3, trainer_hashes)


def test_prepare_fold_output_dir_rejects_nonempty_directory_without_manifest(tmp_path):
    run_dir = tmp_path / "fold_3"
    run_dir.mkdir()
    (run_dir / "partial.bin").write_bytes(b"preserve")

    with pytest.raises(FileExistsError):
        prepare_fold_output_dir(run_dir, 3, {"config": "c"})
