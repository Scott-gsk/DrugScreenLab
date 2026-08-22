from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts.modeling.run_exp009_residual_smoke import (
    RESIDUAL_ACTIVATION_RAW_GAMMA,
    assert_readonly_checkpoint,
    checkpoint_sha256,
    make_checkpoint_manifest,
    residual_learnability_audit,
)


def test_residual_learnability_audit_marks_zero_gradient_or_update_as_broken():
    audit = residual_learnability_audit(
        raw_gamma_before=RESIDUAL_ACTIVATION_RAW_GAMMA,
        raw_gamma_after=RESIDUAL_ACTIVATION_RAW_GAMMA,
        gradient_norms={"output.weight": 0.0},
        parameter_max_abs_updates={"output.weight": 0.0},
        loss_finite=True,
    )

    assert audit["status"] == "BROKEN"
    assert audit["at_least_one_residual_gradient_nonzero"] is False
    assert audit["at_least_one_residual_parameter_updated"] is False


def test_residual_learnability_audit_accepts_any_nonzero_gradient_and_update():
    audit = residual_learnability_audit(
        raw_gamma_before=RESIDUAL_ACTIVATION_RAW_GAMMA,
        raw_gamma_after=9.9e-4,
        gradient_norms={"hidden.0.weight": 0.0, "output.weight": 1e-6},
        parameter_max_abs_updates={"hidden.0.weight": 0.0, "output.weight": 1e-4},
        loss_finite=True,
    )

    assert audit["status"] == "COMPLETE"
    assert audit["at_least_one_residual_gradient_nonzero"] is True


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
