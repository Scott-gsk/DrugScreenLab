from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from drug_screen.foundation.xpert_extension import build_xpert_additive_model


class _OfficialStub(nn.Module):
    def __init__(self, args, config, device, logger):
        super().__init__()
        self.device = torch.device("cpu")
        self.drug_emb = nn.Identity()

    def forward(self, data, mode="ST"):
        return self.drug_emb(data[0])


def _model():
    cls = build_xpert_additive_model(_OfficialStub, use_kpgt=True, use_unipert=False)
    return cls(None, {"model": {"ATTN": {"hidden_size": 4}}}, "cpu", None)


def test_gate_zero_is_exact_and_nonzero_residual_preserves_sequence_length():
    model = _model()
    base = torch.randn(2, 3, 4)
    kpgt = torch.randn(2, 2304)

    out_zero = model((base, *([None] * 9), kpgt))
    torch.testing.assert_close(out_zero, base, rtol=0.0, atol=0.0)

    with torch.no_grad():
        model.additive_gate.fill_(1.0)
    out_residual = model((base, *([None] * 9), kpgt))
    assert out_residual.shape == base.shape
    torch.testing.assert_close(out_residual[:, 1:, :], base[:, 1:, :])
    assert not torch.equal(out_residual[:, 0, :], base[:, 0, :])

