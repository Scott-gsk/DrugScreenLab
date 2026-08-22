from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.data import DataLoader, Dataset

from drug_screen.foundation.xpert_extension import load_xpert_checkpoint
from scripts.modeling.run_exp008_sparse_exploratory import (
    SparseMechanismResidual,
    align_feature_matrix,
    build_parser,
    align_contract_features,
)


class _FrozenBackbone(nn.Module):
    def __init__(self, n_genes: int = 978) -> None:
        super().__init__()
        self.projection = nn.Linear(n_genes, n_genes, bias=False)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.projection(delta)


def test_residual_freezes_backbone_and_uses_bounded_gamma_and_zero_output() -> None:
    backbone = _FrozenBackbone()
    model = SparseMechanismResidual(backbone, feature_dim=978, n_genes=978)
    assert all(not p.requires_grad for p in backbone.parameters())
    assert model.raw_gamma.requires_grad
    assert torch.equal(model.output.weight, torch.zeros_like(model.output.weight))
    assert torch.equal(model.output.bias, torch.zeros_like(model.output.bias))
    assert float(model.gamma.detach()) == pytest.approx(0.05 * torch.tanh(model.raw_gamma.detach()).item())
    assert float(model.gamma.detach()) <= 0.05
    assert float(model.gamma.detach()) >= -0.05


def test_zero_input_and_zero_gate_are_exact_backbone_equivalents() -> None:
    model = SparseMechanismResidual(_FrozenBackbone(), feature_dim=978, n_genes=978)
    delta = torch.randn(3, 978)
    features = torch.zeros(3, 978)
    baseline = model.backbone(delta)
    torch.testing.assert_close(model(delta, features), baseline, rtol=0.0, atol=1e-6)
    with torch.no_grad():
        model.raw_gamma.fill_(0.0)
    torch.testing.assert_close(model(delta, torch.randn(3, 978)), baseline, rtol=0.0, atol=1e-6)


def test_official_batch_uses_deg_output_as_frozen_baseline() -> None:
    class OfficialBatchStub(nn.Module):
        def forward(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            delta = batch[0] - batch[1]
            return (delta, delta, delta, None, None, None, None, None)

    model = SparseMechanismResidual(OfficialBatchStub(), feature_dim=978, n_genes=978)
    batch = tuple(torch.zeros(2, 978) for _ in range(10)) + (torch.zeros(2, 978),)
    output = model(batch)
    torch.testing.assert_close(output, torch.zeros(2, 978), rtol=0.0, atol=1e-6)


def test_contract_alignment_uses_row_positions_and_retains_missing_drugs() -> None:
    features = np.zeros((3, 978), dtype=np.float32)
    features[1, 0] = 7.0
    aligned = align_contract_features(features, [10, 20, 40], np.array([40, 10, 30, 20]))
    assert aligned.shape == (4, 978)
    assert aligned[0, 0] == 0.0
    assert aligned[1, 0] == 0.0
    assert aligned[2, 0] == 0.0
    assert aligned[3, 0] == 7.0


def test_full_feature_alignment_requires_all_8276_drugs() -> None:
    features = torch.zeros(8276, 978)
    aligned = align_contract_features(features, list(range(8276)), np.arange(8276))
    assert aligned.shape == (8276, 978)
    with pytest.raises(ValueError):
        align_contract_features(features[:-1], list(range(8276)), np.arange(8276))


def test_official_checkpoint_audit_and_freezing(tmp_path: Path) -> None:
    model = _FrozenBackbone(4)
    checkpoint = {"model_state_dict": model.state_dict()}
    audit = load_xpert_checkpoint(model, checkpoint, strict_official=True)
    assert audit["missing_official"] == []
    assert audit["unexpected"] == []
    assert model.official_checkpoint_loaded is True
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_smoke_cli_has_explicit_one_batch_mode() -> None:
    parser = build_parser()
    args = parser.parse_args(["--full-sdst", "--smoke-batches", "1"])
    assert args.smoke_batches == 1


def test_smoke_uses_only_one_batch_and_feature_extra() -> None:
    from scripts.modeling.run_exp008_sparse_exploratory import _FeatureDataset

    class TinyDataset(Dataset):
        def __len__(self):
            return 3
        def __getitem__(self, index):
            return tuple(torch.zeros(978) for _ in range(10))

    dataset = _FeatureDataset(TinyDataset(), torch.zeros(3, 978))
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)))
    assert len(batch) == 11
    assert batch[0].shape == (2, 978)
    assert batch[10].shape == (2, 978)


def test_cli_rejects_fast_and_max_rows() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--full-sdst", "--fast"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--full-sdst", "--max-rows", "10"])


def test_cli_script_rejects_fast_prefix() -> None:
    script = Path(__file__).parents[2] / "scripts/modeling/run_exp008_sparse_exploratory.py"
    result = subprocess.run(
        [sys.executable, str(script), "--full-sdst", "--fast"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "fast" in (result.stderr + result.stdout).lower()
