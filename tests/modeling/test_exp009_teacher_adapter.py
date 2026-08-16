from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from drug_screen.foundation.exp009_teacher_adapter import (
    build_teacher_soft_target_features,
    select_teacher_soft_target_batch,
    write_teacher_soft_target_features,
)


def _registry(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "format": "xpert_drug_registry_v1",
                "drugs": [
                    {
                        "pert_id": "BRD-B",
                        "pert_idx": 2,
                        "inchi_key": "bbb-key",
                        "canonical_smiles": "CCN",
                        "global_inference_eligible": True,
                    },
                    {
                        "pert_id": "BRD-A",
                        "pert_idx": 1,
                        "inchi_key": "aaa-key",
                        "canonical_smiles": "CCO",
                        "global_inference_eligible": True,
                    },
                    {
                        "pert_id": "BRD-INELIGIBLE",
                        "pert_idx": 3,
                        "inchi_key": "ccc-key",
                        "canonical_smiles": "CCC",
                        "global_inference_eligible": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _teacher(path: Path) -> None:
    torch.save(
        {
            "state_dict": {
                "weight": torch.zeros((64, 2048), dtype=torch.float32),
                "bias": torch.zeros(64, dtype=torch.float32),
            },
            "input_dim": 2048,
            "target_dim": 64,
            "targets": [f"P{index:05d}" for index in range(64)],
        },
        path,
    )


def test_teacher_adapter_generates_ordered_64d_logits_and_probabilities_for_every_eligible_drug(tmp_path: Path):
    registry = tmp_path / "registry.json"
    checkpoint = tmp_path / "teacher.pt"
    _registry(registry)
    _teacher(checkpoint)

    payload = build_teacher_soft_target_features(registry, checkpoint, batch_size=1)

    assert payload.pert_ids.tolist() == ["BRD-A", "BRD-B"]
    assert payload.pert_indices.tolist() == [1, 2]
    assert payload.inchi_keys.tolist() == ["AAA-KEY", "BBB-KEY"]
    assert payload.logits.shape == (2, 64)
    assert payload.probabilities.shape == (2, 64)
    np.testing.assert_array_equal(payload.logits, np.zeros((2, 64), dtype=np.float32))
    np.testing.assert_array_equal(payload.probabilities, np.full((2, 64), 0.5, dtype=np.float32))
    assert payload.audit["response_values_read"] is False
    assert payload.audit["eligible_drug_count"] == 2
    assert payload.audit["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_teacher_adapter_writes_npz_and_identity_audit_without_checkpoint_mutation(tmp_path: Path):
    registry = tmp_path / "registry.json"
    checkpoint = tmp_path / "teacher.pt"
    output_dir = tmp_path / "features"
    _registry(registry)
    _teacher(checkpoint)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    payload = build_teacher_soft_target_features(registry, checkpoint)
    audit = write_teacher_soft_target_features(payload, output_dir)

    artifact = np.load(output_dir / "xpert_sdst_soft_target_features.npz", allow_pickle=False)
    assert artifact["pert_id"].tolist() == ["BRD-A", "BRD-B"]
    assert artifact["soft_target_logits"].shape == (2, 64)
    assert artifact["soft_target_probabilities"].shape == (2, 64)
    assert audit["artifact_sha256"] == hashlib.sha256((output_dir / "xpert_sdst_soft_target_features.npz").read_bytes()).hexdigest()
    assert audit["checkpoint_sha256"] == checkpoint_sha
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_sha
    assert json.loads((output_dir / "identity_audit.json").read_text(encoding="utf-8"))["checkpoint_access"] == "read_only_load"


def test_teacher_adapter_selects_sdst_batch_by_exact_pert_id_and_rejects_unknown_id(tmp_path: Path):
    registry = tmp_path / "registry.json"
    checkpoint = tmp_path / "teacher.pt"
    _registry(registry)
    _teacher(checkpoint)
    payload = build_teacher_soft_target_features(registry, checkpoint)

    batch = select_teacher_soft_target_batch(payload, ["BRD-B", "BRD-A"])

    assert batch["pert_id"].tolist() == ["BRD-B", "BRD-A"]
    assert batch["soft_target_logits"].shape == (2, 64)
    assert batch["feature_valid"].tolist() == [True, True]
    with pytest.raises(ValueError, match="does not contain"):
        select_teacher_soft_target_batch(payload, ["BRD-MISSING"])


def test_teacher_adapter_marks_unparseable_registry_structure_missing_without_a_fake_zero_feature(tmp_path: Path):
    registry = tmp_path / "registry.json"
    checkpoint = tmp_path / "teacher.pt"
    _registry(registry)
    raw = json.loads(registry.read_text(encoding="utf-8"))
    raw["drugs"].append(
        {
            "pert_id": "BRD-BAD",
            "pert_idx": 4,
            "inchi_key": "",
            "canonical_smiles": "not-smiles",
            "global_inference_eligible": True,
        }
    )
    registry.write_text(json.dumps(raw), encoding="utf-8")
    _teacher(checkpoint)

    payload = build_teacher_soft_target_features(registry, checkpoint)

    bad = payload.pert_ids.tolist().index("BRD-BAD")
    assert payload.feature_valid.tolist() == [True, True, False]
    assert payload.confidence.tolist() == [0.5, 0.5, 0.0]
    assert np.isnan(payload.logits[bad]).all()
    assert np.isnan(payload.probabilities[bad]).all()
    assert payload.audit["invalid_structure_count"] == 1
    assert payload.audit["missing_inchi_key_count"] == 1

def test_teacher_adapter_confidence_is_max_probability_for_valid_rows(tmp_path: Path):
    registry = tmp_path / "registry.json"
    checkpoint = tmp_path / "teacher.pt"
    _registry(registry)
    torch.save(
        {
            "state_dict": {"weight": torch.zeros((64, 2048)), "bias": torch.tensor([2.0] + [0.0] * 63)},
            "input_dim": 2048,
            "target_dim": 64,
            "targets": [f"P{index:05d}" for index in range(64)],
        },
        checkpoint,
    )
    payload = build_teacher_soft_target_features(registry, checkpoint)
    assert payload.audit["confidence_definition"] == "max_probability_across_64_targets"
    np.testing.assert_allclose(payload.confidence, [1.0 / (1.0 + np.exp(-2.0))] * 2, rtol=1e-6)


def test_teacher_adapter_rejects_checkpoint_with_wrong_feature_width(tmp_path: Path):
    registry = tmp_path / "registry.json"
    checkpoint = tmp_path / "bad_teacher.pt"
    _registry(registry)
    torch.save(
        {
            "state_dict": {"weight": torch.zeros((64, 10)), "bias": torch.zeros(64)},
            "input_dim": 10,
            "target_dim": 64,
            "targets": [f"P{index:05d}" for index in range(64)],
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="input_dim=2048"):
        build_teacher_soft_target_features(registry, checkpoint)
