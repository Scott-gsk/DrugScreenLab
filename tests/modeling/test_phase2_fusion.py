from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from drug_screen.modeling.phase2_fusion import (
    Phase2FusionError,
    build_kpgt_unipert_features,
)


def _write_mapping(path: Path, mapping: dict[str, int]) -> None:
    path.write_text(json.dumps(mapping), encoding="utf-8")


def test_additive_fusion_aligns_by_pert_id_not_row_order(tmp_path: Path) -> None:
    kpgt_path = tmp_path / "kpgt.npy"
    unipert_path = tmp_path / "unipert.npy"
    np.save(kpgt_path, np.asarray([[10.0, 11.0], [20.0, 21.0]], dtype=np.float32))
    np.save(unipert_path, np.asarray([[200.0], [100.0]], dtype=np.float32))
    kpgt_mapping = tmp_path / "kpgt.json"
    unipert_mapping = tmp_path / "unipert.json"
    _write_mapping(kpgt_mapping, {"drug_b": 0, "drug_a": 1})
    _write_mapping(unipert_mapping, {"drug_a": 0, "drug_b": 1})

    features, mapping, audit = build_kpgt_unipert_features(
        ["drug_b", "drug_a"],
        kpgt_features_path=kpgt_path,
        kpgt_mapping_path=kpgt_mapping,
        unipert_features_path=unipert_path,
        unipert_mapping_path=unipert_mapping,
    )

    assert mapping == {"drug_a": 0, "drug_b": 1}
    np.testing.assert_allclose(features, [[20.0, 21.0, 200.0], [10.0, 11.0, 100.0]])
    assert audit["alignment_key"] == "pert_id"
    assert audit["labels_used"] is False


def test_additive_fusion_rejects_missing_identity(tmp_path: Path) -> None:
    kpgt_path = tmp_path / "kpgt.npy"
    unipert_path = tmp_path / "unipert.npy"
    np.save(kpgt_path, np.ones((1, 2), dtype=np.float32))
    np.save(unipert_path, np.ones((1, 1), dtype=np.float32))
    kpgt_mapping = tmp_path / "kpgt.json"
    unipert_mapping = tmp_path / "unipert.json"
    _write_mapping(kpgt_mapping, {"drug_a": 0})
    _write_mapping(unipert_mapping, {"drug_a": 0})

    with pytest.raises(Phase2FusionError, match="missing_kpgt"):
        build_kpgt_unipert_features(
            ["drug_a", "drug_b"],
            kpgt_features_path=kpgt_path,
            kpgt_mapping_path=kpgt_mapping,
            unipert_features_path=unipert_path,
            unipert_mapping_path=unipert_mapping,
        )
