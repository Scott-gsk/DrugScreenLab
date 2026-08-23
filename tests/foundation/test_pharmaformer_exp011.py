from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from drug_screen.foundation.pharmaformer_adapter import TrainOnlyGeneTransform, TrainOnlySmilesEncoder
from drug_screen.foundation.pharmaformer_exp011 import (
    Exp011ModelConfig,
    build_m3_adapter,
    build_official_m2,
    load_exp011_checkpoint,
    save_exp011_checkpoint,
)
from drug_screen.modeling.exp011_models import assert_m3_zero_initial_delta


def _upstream() -> SimpleNamespace:
    torch = pytest.importorskip("torch")

    class Feature(torch.nn.Module):
        def __init__(self, gene_input_size, gene_hidden_size, drug_hidden_size):
            super().__init__()
            self.gene_fc1 = torch.nn.Linear(gene_input_size, gene_hidden_size)
            self.gene_fc2 = torch.nn.Linear(gene_hidden_size, gene_hidden_size)
            self.smiles_fc = torch.nn.Linear(128, drug_hidden_size)

    class Trans(torch.nn.Module):
        def __init__(self, feature_dim, nhead, seq_len, dim_feedforward, dropout, num_layers):
            super().__init__()
            layer = torch.nn.TransformerEncoderLayer(feature_dim, nhead, dim_feedforward, dropout, batch_first=True)
            self.transformer_encoder = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
            self.output = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(seq_len * feature_dim, 1))

        def forward(self, value):
            return self.output(self.transformer_encoder(value))

    class Combined(torch.nn.Module):
        def __init__(self, gene_input_size, gene_hidden_size, drug_hidden_size, feature_dim, nhead,
                     num_layers=3, dim_feedforward=2048, dropout=.1):
            super().__init__()
            self.feature_extractor = Feature(gene_input_size, gene_hidden_size, drug_hidden_size)
            self.feature_dim = feature_dim
            self.seq_len = (gene_hidden_size + drug_hidden_size) // feature_dim
            self.transformer = Trans(feature_dim, nhead, self.seq_len, dim_feedforward, dropout, num_layers)

        def forward(self, gene, drug):
            feature = self.feature_extractor
            merged = torch.cat((torch.relu(feature.gene_fc2(torch.relu(feature.gene_fc1(gene)))), torch.relu(feature.smiles_fc(drug))), 1)
            return self.transformer(merged.view(len(gene), self.seq_len, self.feature_dim))

    return SimpleNamespace(CombinedModel=Combined)


def _config() -> Exp011ModelConfig:
    return Exp011ModelConfig(gene_input_size=3, gene_hidden_size=4, drug_hidden_size=4, feature_dim=4,
                             nhead=2, num_layers=1, dim_feedforward=8, adapter_bottleneck=2,
                             adapter_dropout=0.3, test_only_geometry=True)


def test_m3_adapter_is_exactly_source_at_initialization() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(3)
    source = build_official_m2(_upstream(), _config())
    adapter = build_m3_adapter(source, _config())
    gene, drug = torch.rand(2, 3), torch.rand(2, 128)
    assert_m3_zero_initial_delta(adapter, gene, drug, source)
    assert torch.count_nonzero(adapter.adapter.up.weight) == 0
    assert torch.count_nonzero(adapter.adapter.up.bias) == 0


def test_exp011_checkpoint_strictly_reloads_adapter_and_transforms(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    config = _config()
    source = build_official_m2(_upstream(), config)
    model = build_m3_adapter(source, config)
    transform = TrainOnlyGeneTransform().fit(np.array([[0, 1, 2], [2, 1, 0]], dtype=float), ["s1", "s2"])
    code_path = tmp_path / "frozen.codes"
    code_path.write_text("#version: 0.2\nC C\n", encoding="utf-8")
    encoder = TrainOnlySmilesEncoder().fit(["CC"], ["d1"], bpe_codes_path=code_path)
    path = save_exp011_checkpoint(
        tmp_path / "adapter.pt", model, config, transform, encoder, variant="M3-adapter", fold_id=2,
        gene_contract_sha256="a" * 64, provenance={"synthetic": True},
        response_transform_state={"algorithm": "train_only_per_cohort_drug_zscore"},
    )
    restored, payload = load_exp011_checkpoint(path, _upstream(), device="cpu")
    gene, drug = torch.rand(2, 3), torch.rand(2, 128)
    model.eval()
    np.testing.assert_array_equal(model(gene, drug).detach().numpy(), restored(gene, drug).detach().numpy())
    assert payload["response_transform"]["algorithm"] == "train_only_per_cohort_drug_zscore"
    assert payload["smiles_encoder_object"].bpe_codes is not None
