from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drug_screen.foundation.exp006_transfer import (
    GeneticPerturbagenAdapter,
    assign_compound_level_splits,
    build_xpert_genetic_transfer_model,
    delta978_metrics,
    sample_unique_compounds,
    select_dual_coverage_contexts,
    summarize_context_coverage,
)


def _genetic_frame() -> pd.DataFrame:
    rows = []
    # CTX-A: 4 unique genes, CTX-B: 3, CTX-C: 1, CTX-D: 4
    for gene in ("G1", "G2", "G3", "G4"):
        rows.append(
            {
                "cell_id": "CTX-A",
                "gene_symbol": gene,
                "pert_id": f"SH-{gene}",
                "pert_type": "trt_sh",
                "has_matched_control": True,
                "unipert_mappable": True,
            }
        )
        rows.append(
            {
                "cell_id": "CTX-D",
                "gene_symbol": gene,
                "pert_id": f"SH2-{gene}",
                "pert_type": "trt_sh",
                "has_matched_control": True,
                "unipert_mappable": True,
            }
        )
    for gene in ("G1", "G2", "G5"):
        rows.append(
            {
                "cell_id": "CTX-B",
                "gene_symbol": gene,
                "pert_id": f"SHB-{gene}",
                "pert_type": "trt_sh",
                "has_matched_control": True,
                "unipert_mappable": True,
            }
        )
    rows.append(
        {
            "cell_id": "CTX-C",
            "gene_symbol": "G9",
            "pert_id": "SH-C",
            "pert_type": "trt_sh",
            "has_matched_control": True,
            "unipert_mappable": True,
        }
    )
    # Duplicate records must not inflate unique-gene diversity.
    rows.append(
        {
            "cell_id": "CTX-A",
            "gene_symbol": "G1",
            "pert_id": "SH-G1-REP",
            "pert_type": "trt_sh",
            "has_matched_control": True,
            "unipert_mappable": True,
        }
    )
    # Unmappable / unmatched rows must be ignored.
    rows.append(
        {
            "cell_id": "CTX-A",
            "gene_symbol": "GX",
            "pert_id": "SH-GX",
            "pert_type": "trt_sh",
            "has_matched_control": False,
            "unipert_mappable": True,
        }
    )
    return pd.DataFrame(rows)


def _chemical_frame() -> pd.DataFrame:
    rows = []
    for pert in ("C1", "C2", "C3", "C4", "C5"):
        rows.append({"cell_iname": "CTX-A", "pert_id": pert})
        rows.append({"cell_iname": "CTX-D", "pert_id": pert})
    for pert in ("C1", "C6"):
        rows.append({"cell_iname": "CTX-B", "pert_id": pert})
    rows.append({"cell_iname": "CTX-C", "pert_id": "C9"})
    rows.append({"cell_iname": "CTX-A", "pert_id": "C1"})  # duplicate row
    return pd.DataFrame(rows)


def test_coverage_uses_unique_genes_and_compounds_not_rows() -> None:
    coverage = summarize_context_coverage(_genetic_frame(), _chemical_frame())
    by_context = {row["context_id"]: row for row in coverage}
    assert by_context["CTX-A"]["unique_genes"] == 4
    assert by_context["CTX-A"]["genetic_records"] == 5
    assert by_context["CTX-A"]["unique_compounds"] == 5
    assert by_context["CTX-A"]["chemical_records"] == 6


def test_context_selection_is_response_blind_and_ranks_by_coverage() -> None:
    decision = select_dual_coverage_contexts(
        _genetic_frame(),
        _chemical_frame(),
        min_unique_genes=2,
        min_unique_compounds=2,
        max_contexts=3,
        target_unique_genes=4,
    )
    assert decision["selected_contexts"] == ["CTX-A", "CTX-D", "CTX-B"]
    assert "CTX-C" not in decision["selected_contexts"]
    assert decision["selection_policy"]["response_values_used"] is False
    assert decision["selection_policy"]["test_performance_used"] is False
    assert decision["per_context"][0]["meets_target_unique_genes"] is True


def test_compound_level_split_is_frozen_and_nonoverlapping() -> None:
    compounds = [f"BRD-{index:03d}" for index in range(20)]
    assignment = assign_compound_level_splits(compounds, seed=20260813)
    train = set(assignment["train"])
    test = set(assignment["test"])
    validation = set(assignment["validation"])
    assert not train & test
    assert not train & validation
    assert not test & validation
    assert set(compounds) == train | test | validation
    again = assign_compound_level_splits(compounds, seed=20260813)
    assert again == assignment


def test_fraction_sampling_is_unique_compound_atomic() -> None:
    train = [f"D{index}" for index in range(10)]
    selected = sample_unique_compounds(train, fraction=0.2, seed=20260813)
    assert len(selected) == 2
    assert sample_unique_compounds(train, fraction=0.2, seed=20260813) == selected
    assert sample_unique_compounds(train, fraction=1.0, seed=20260813) == sorted(train)


def test_delta978_metrics_include_direction_consistency() -> None:
    true = np.asarray([[1.0, -2.0, 3.0], [2.0, 1.0, -4.0]], dtype=np.float32)
    pred = np.asarray([[0.8, -1.0, 2.5], [1.5, 0.5, -3.0]], dtype=np.float32)
    metrics = delta978_metrics(true, pred)
    assert metrics["rows"] == 2
    assert metrics["direction_consistency"] == 1.0
    assert metrics["pearson_row_mean"] is not None
    assert metrics["spearman_row_mean"] is not None
    assert metrics["mse"] > 0


def test_genetic_adapter_is_a_single_projected_token() -> None:
    torch = pytest.importorskip("torch")
    adapter = GeneticPerturbagenAdapter(unipert_dim=8, hidden_size=16, dropout=0.0)
    token = adapter(torch.ones((3, 8)), direction=torch.zeros(3, dtype=torch.long))
    assert token.shape == (3, 16)
    assert torch.isfinite(token).all()


def test_genetic_transfer_model_replaces_drug_token_without_lengthening_sequence() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    class _Attn(nn.Module):
        def forward(self, cell_embed, drug_embed, cell_mask, drug_mask, *args, **kwargs):
            return cell_embed, drug_embed, drug_mask

    class _OfficialStub(nn.Module):
        def __init__(self, args, config, device, logger):
            super().__init__()
            self.device = torch.device("cpu")
            self.drug_emb = nn.Identity()
            self.attnEncoder_trt = _Attn()

        def forward(self, data, mode="ST"):
            drug = self.drug_emb(data[0])
            return self.attnEncoder_trt(data[1], drug, None, "official-mask")

    cls = build_xpert_genetic_transfer_model(
        _OfficialStub, unipert_dim=8, hidden_size=4, dropout=0.0
    )
    model = cls(None, {"model": {"ATTN": {"hidden_size": 4}}}, "cpu", None)
    base_drug = torch.randn(2, 5, 4)
    cell = torch.randn(2, 3, 4)
    genetic = torch.randn(2, 8)
    direction = torch.zeros(2, dtype=torch.long)

    official = model((base_drug, cell))
    assert official[1].shape == (2, 5, 4)

    genetic_out = model((base_drug, cell, genetic, direction))
    assert genetic_out[1].shape == (2, 1, 4)
    assert genetic_out[2] is None
    assert not torch.equal(genetic_out[1][:, 0, :], official[1][:, 0, :])


def test_coverage_contract_roundtrip(tmp_path: Path) -> None:
    decision = select_dual_coverage_contexts(
        _genetic_frame(),
        _chemical_frame(),
        min_unique_genes=2,
        min_unique_compounds=2,
        max_contexts=3,
        target_unique_genes=4,
    )
    path = tmp_path / "CONTEXT_COVERAGE.json"
    path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["selected_contexts"] == ["CTX-A", "CTX-D", "CTX-B"]
    assert payload["downsample"]["applied"] is False
