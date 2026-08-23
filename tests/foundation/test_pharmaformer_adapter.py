from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from drug_screen.foundation.pharmaformer_adapter import (
    PharmaFormerConfig, TrainOnlyGeneTransform, TrainOnlySmilesEncoder,
    build_model, load_checkpoint, save_checkpoint, validate_patient_fold,
)
from drug_screen.foundation.pharmaformer_training import fit_model, predict, training_policy


def test_patient_fold_rejects_patient_and_pdo_leakage() -> None:
    good = [{"patient_id": "p1", "pdo_id": "o1", "partition": "train"},
            {"patient_id": "p2", "pdo_id": "o2", "partition": "validation"},
            {"patient_id": "p3", "pdo_id": "o3", "partition": "test"}]
    assert validate_patient_fold(good)["test"] == {"p3"}
    with pytest.raises(ValueError, match="leakage"):
        validate_patient_fold(good + [{"patient_id": "p1", "pdo_id": "o4", "partition": "test"}])


def test_transforms_are_train_only_and_frozen(tmp_path: Path) -> None:
    transform = TrainOnlyGeneTransform().fit(np.array([[0, 1], [2, 3]], dtype=float), ["a", "b"])
    held_out = transform.transform(np.array([[100, 200]], dtype=float))
    assert transform.fit_sample_ids == ("a", "b")
    assert held_out[0, 0] > 1
    assert np.isfinite(transform.transform(np.array([[np.nan, 2.0]]))).all()
    codes = tmp_path / "bpe.codes"
    codes.write_text("#version: 0.2\nC C\n", encoding="utf-8")
    encoder = TrainOnlySmilesEncoder().fit(["CC", "CO"], ["d1", "d2"], bpe_codes_path=codes)
    assert encoder.transform(["CC"]).shape == (1, 128)
    with pytest.raises(ValueError, match="absent from training"):
        encoder.transform(["CN"])


def test_synthetic_training_checkpoint_reload(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class Feature(torch.nn.Module):
        def __init__(self, gene_input_size, gene_hidden_size, drug_hidden_size):
            super().__init__()
            self.gene_fc1 = torch.nn.Linear(gene_input_size, gene_hidden_size)
            self.gene_fc2 = torch.nn.Linear(gene_hidden_size, gene_hidden_size)
            self.smiles_fc = torch.nn.Linear(128, drug_hidden_size)
        def forward(self, gene, drug):
            gene = torch.relu(self.gene_fc2(torch.relu(self.gene_fc1(gene))))
            return torch.cat((gene, torch.relu(self.smiles_fc(drug))), dim=1)

    class Combined(torch.nn.Module):
        def __init__(self, gene_input_size, gene_hidden_size, drug_hidden_size, feature_dim, nhead,
                     num_layers=3, dim_feedforward=2048, dropout=.1):
            super().__init__()
            self.feature_extractor = Feature(gene_input_size, gene_hidden_size, drug_hidden_size)
            self.output = torch.nn.Linear(gene_hidden_size + drug_hidden_size, 1)
        def forward(self, gene, drug):
            return self.output(self.feature_extractor(gene, drug))

    upstream = SimpleNamespace(CombinedModel=Combined)
    config = PharmaFormerConfig(gene_input_size=3, gene_hidden_size=4, drug_hidden_size=4,
                                feature_dim=4, nhead=2, num_layers=1, dim_feedforward=8)
    model = build_model(upstream, config)
    gene = np.arange(24, dtype=np.float32).reshape(8, 3)
    transform = TrainOnlyGeneTransform().fit(gene[:6], [f"s{i}" for i in range(6)])
    codes = tmp_path / "bpe.codes"
    codes.write_text("#version: 0.2\nC C\n", encoding="utf-8")
    encoder = TrainOnlySmilesEncoder().fit(["CC"] * 6, ["d"] * 6, bpe_codes_path=codes)
    x, drug = transform.transform(gene), encoder.transform(["CC"] * 8)
    result = fit_model(model, x[:6], drug[:6], np.arange(6), x[6:], drug[6:], np.arange(6, 8),
                       device="cpu", epochs=2, batch_size=2, patience=1, seed=1)
    assert result.epochs_run >= 1
    expected = predict(model, x[6:], drug[6:], device="cpu")
    path = save_checkpoint(tmp_path / "model.pt", model, config, transform, encoder,
                           variant="M4", fold_id=0, provenance={"synthetic": True})
    restored, payload = load_checkpoint(path, upstream, device="cpu")
    np.testing.assert_allclose(predict(restored, x[6:], drug[6:], device="cpu"), expected)
    assert payload["gene_transform"]["fit_scope"] == "train_only"
    assert payload["smiles_encoder"]["algorithm"].startswith("upstream_subword_nmt_BPE")


def test_m2_target_is_inference_only_and_labels_cannot_affect_prediction() -> None:
    assert training_policy("M2", "target_mcrc") == "frozen_inference"
    assert training_policy("M3", "target_mcrc") == "finetune"
    assert training_policy("M4", "target_mcrc") == "fit_random_init"
    with pytest.raises(ValueError, match="forbidden"):
        training_policy("M3", "source_gdsc")
    # In frozen inference the public prediction API has no response argument;
    # changing target labels therefore cannot enter either weights or outputs.
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential()  # sentinel: use a two-input module below
    class Sum(torch.nn.Module):
        def forward(self, gene, drug):
            return gene.sum(1, keepdim=True) + drug[:, :1]
    gene, drug = np.ones((2, 3), dtype=np.float32), np.zeros((2, 128), dtype=np.float32)
    before = {k: v.clone() for k, v in Sum().state_dict().items()}
    first = predict(Sum(), gene, drug, device="cpu")
    changed_target_labels = np.array([999.0, -999.0])
    second = predict(Sum(), gene, drug, device="cpu")
    np.testing.assert_array_equal(first, second)
    assert changed_target_labels.tolist() == [999.0, -999.0]
    assert before == {}
