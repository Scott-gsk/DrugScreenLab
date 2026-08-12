from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import torch

from drug_screen.modeling.phase1 import (
    GENE_COUNT,
    MANIFEST_FORMAT,
    ContextChemicalDoseTimePredictor,
    Phase1IntegrationError,
    load_phase1_manifest,
    run_tiny,
)


def _write_manifest(tmp_path: Path) -> Path:
    cache = np.zeros((8, GENE_COUNT), dtype=np.float32)
    cache[0, 0] = 1.0
    cache[1, 0] = 0.25
    cache[2, 0] = 2.0
    cache[3, 0] = 0.50
    cache_path = tmp_path / "cache.npy"
    np.save(cache_path, cache)
    records = [
        {
            "sample_id": "train-a",
            "treatment_group_id": "drug-a|context-a|10um|6h",
            "drug_id": "drug-a",
            "context_id": "context-a",
            "dose_um": 10.0,
            "time_h": 6.0,
            "split": "train",
            "treatment_cache_row": 0,
            "control_cache_row": 1,
            "chemical_feature_row": 0,
        },
        {
            "sample_id": "test-a",
            "treatment_group_id": "drug-a|context-b|10um|6h",
            "drug_id": "drug-a",
            "context_id": "context-b",
            "dose_um": 10.0,
            "time_h": 6.0,
            "split": "test",
            "treatment_cache_row": 2,
            "control_cache_row": 3,
            "chemical_feature_row": 0,
        },
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": MANIFEST_FORMAT,
                "gene_count": GENE_COUNT,
                "cache": {"relative_path": "cache.npy", "sha256": "", "shape": [8, GENE_COUNT]},
                "chemical_features": {
                    "relative_path": "chemical.npy",
                    "sha256": "",
                    "shape": [1, 8],
                },
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    np.save(tmp_path / "chemical.npy", np.ones((1, 8), dtype=np.float32))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["cache"]["sha256"] = sha256(cache_path.read_bytes()).hexdigest()
    payload["chemical_features"]["sha256"] = sha256((tmp_path / "chemical.npy").read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def test_phase1_model_changes_prediction_when_biological_context_changes():
    model = ContextChemicalDoseTimePredictor(
        chemical_dim=8,
        context_dim=GENE_COUNT,
        hidden_dim=16,
        gene_count=GENE_COUNT,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.context_projection.weight[0, 0] = 1.0
        model.interaction[0].weight[0, 0] = 1.0
        model.output.weight[0, 0] = 1.0
    chemical = torch.zeros(1, 8)
    dose_time = torch.tensor([[10.0, 6.0]])
    context_a = torch.zeros(1, GENE_COUNT)
    context_b = context_a.clone()
    context_b[0, 0] = 1.0
    prediction_a = model(context_a, chemical, dose_time)
    prediction_b = model(context_b, chemical, dose_time)
    assert not torch.equal(prediction_a, prediction_b)


def test_phase1_manifest_rejects_missing_control_or_cross_split_group(tmp_path: Path):
    manifest_path = _write_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["records"][1]["control_cache_row"] = 1
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Phase1IntegrationError, match="control row cannot be shared"):
        load_phase1_manifest(manifest_path, root=tmp_path)


def test_phase1_tiny_overfits_synthetic_context_conditioned_targets(tmp_path: Path):
    manifest_path = _write_manifest(tmp_path)
    checkpoint_path = tmp_path / "candidate.pt"
    result = run_tiny(
        manifest_path,
        root=tmp_path,
        hidden_dim=16,
        epochs=80,
        learning_rate=0.03,
        seed=7,
        max_records=2,
        checkpoint_path=checkpoint_path,
    )
    assert result["status"] == "TINY_COMPLETE"
    assert result["gene_count"] == GENE_COUNT
    assert result["train_metrics"]["mae"] < 0.05
    assert set(result["models"]) == {"chemical_only", "chemical_context", "global_mean"}
    assert "global_mean" in result["models"]
    assert result["target_variance"] >= 0
    assert "group_metrics" in result["models"]["chemical_context"]
    assert result["normalization"]["fit_scope"] == "train_only"
    assert checkpoint_path.is_file()
