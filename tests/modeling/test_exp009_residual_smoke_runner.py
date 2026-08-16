from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts.modeling.run_exp009_residual_smoke import (
    assert_readonly_checkpoint,
    checkpoint_sha256,
    make_checkpoint_manifest,
)


def test_checkpoint_manifest_records_sha256_readonly_identity_and_seed(tmp_path: Path):
    checkpoint = tmp_path / "official_xpert.pth"
    checkpoint.write_bytes(b"frozen XPert checkpoint")
    checkpoint.chmod(0o444)

    manifest = make_checkpoint_manifest(checkpoint, seed=2026)

    assert manifest["seed"] == 2026
    assert manifest["checkpoint"]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert manifest["checkpoint"]["readonly"] is True
    assert checkpoint_sha256(checkpoint) == manifest["checkpoint"]["sha256"]
    assert_readonly_checkpoint(checkpoint)
