from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drug_screen.modeling.phase2_fast import (
    Phase2FastError,
    UNIPERT_DIM,
    build_unipert_chemical_features,
)


def test_unipert_probe_requires_a_valid_checkpoint(tmp_path: Path) -> None:
    frame = pd.DataFrame({"pert_id": ["A"], "canonical_smiles": ["CCO"]})
    with pytest.raises(Phase2FastError, match="checkpoint"):
        build_unipert_chemical_features(frame, model_path=tmp_path / "missing.pt")


def test_unipert_probe_uses_frozen_chemical_head(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = {
        "cp_encoder": {
            "linear_layer.weight": torch.ones((UNIPERT_DIM, 2048)),
            "linear_layer.bias": torch.zeros((UNIPERT_DIM,)),
        }
    }
    model_path = tmp_path / "unipert.pt"
    torch.save(checkpoint, model_path)
    frame = pd.DataFrame({"pert_id": ["B", "A"], "canonical_smiles": ["CCO", "CC"]})
    features, mapping, audit = build_unipert_chemical_features(frame, model_path=model_path)
    assert features.shape == (2, UNIPERT_DIM)
    assert mapping == {"A": 0, "B": 1}
    assert np.all(features > 0)
    assert audit["output_dim"] == UNIPERT_DIM
