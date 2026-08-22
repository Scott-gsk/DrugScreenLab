from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.modeling.run_exp009_residual_smoke import (
    DEFAULT_SOFT_TARGET_FEATURES,
    load_soft_target_features_for_batch,
)


def test_morgan_feature_default_is_the_verified_complete_8418_by_64_artifact():
    expected = (
        Path("artifacts/experiments/EXP-009/teacher_morgan_probe_100k/xpert_sdst_features")
        / "xpert_sdst_soft_target_features.npz"
    )

    assert DEFAULT_SOFT_TARGET_FEATURES.as_posix().endswith(expected.as_posix())


def test_morgan_feature_loader_returns_probabilities_in_requested_pert_id_order(tmp_path: Path):
    artifact = tmp_path / "features.npz"
    np.savez_compressed(
        artifact,
        pert_id=np.asarray(["drug-b", "drug-a"]),
        pert_idx=np.asarray([2, 1]),
        soft_target_probabilities=np.asarray([[0.2, 0.3], [0.4, 0.5]], dtype=np.float32),
        feature_valid=np.asarray([True, True]),
    )

    values, audit = load_soft_target_features_for_batch(artifact, ["drug-a", "drug-b"], expected_target_dim=2)

    np.testing.assert_array_equal(values, np.asarray([[0.4, 0.5], [0.2, 0.3]], dtype=np.float32))
    assert audit["feature_source"] == "precomputed_morgan_soft_target_probabilities"
    assert audit["rows"] == 2
    assert audit["target_dim"] == 2


def test_morgan_feature_loader_rejects_invalid_or_missing_batch_features(tmp_path: Path):
    artifact = tmp_path / "features.npz"
    np.savez_compressed(
        artifact,
        pert_id=np.asarray(["drug-a"]),
        soft_target_probabilities=np.asarray([[0.4, 0.5]], dtype=np.float32),
        feature_valid=np.asarray([False]),
    )

    with pytest.raises(ValueError, match="invalid"):
        load_soft_target_features_for_batch(artifact, ["drug-a"], expected_target_dim=2)
    with pytest.raises(ValueError, match="does not contain"):
        load_soft_target_features_for_batch(artifact, ["drug-b"], expected_target_dim=2)
