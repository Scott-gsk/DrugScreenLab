"""Small target-domain variants around the pinned official PharmaFormer backbone."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class GeneResidualAdapter(nn.Module):
    """A zero-initialized residual bottleneck after the official gene encoder."""

    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        if hidden_size < 1 or rank < 1:
            raise ValueError("hidden size and rank must be positive")
        self.down = nn.Linear(hidden_size, rank)
        self.up = nn.Linear(rank, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.up(functional.gelu(self.down(value)))


class PharmaFormerWithAdapter(nn.Module):
    """Preserve official drug encoder/transformer and adapt its gene representation."""

    def __init__(self, backbone: nn.Module, rank: int) -> None:
        super().__init__()
        if not all(hasattr(backbone.feature_extractor, name)
                   for name in ("gene_fc1", "gene_fc2", "smiles_fc")):
            raise ValueError("official feature extractor interface changed")
        self.backbone = backbone
        self.adapter = GeneResidualAdapter(
            backbone.feature_extractor.gene_fc2.out_features, rank)

    def forward(self, gene_expr: torch.Tensor, smiles: torch.Tensor) -> torch.Tensor:
        extractor = self.backbone.feature_extractor
        gene = functional.relu(extractor.gene_fc2(functional.relu(extractor.gene_fc1(gene_expr))))
        gene = self.adapter(gene)
        drug = functional.relu(extractor.smiles_fc(smiles))
        features = torch.cat((gene, drug), dim=1)
        features = features.view(features.shape[0], self.backbone.seq_len, self.backbone.feature_dim)
        return self.backbone.transformer(features)


def configure_target_variant(backbone: nn.Module, arm: str, *, adapter_rank: int = 8) -> nn.Module:
    """Configure trainable parameters for four fixed EXP-036 comparison arms."""
    if arm not in {"random_full", "source_head", "source_full", "source_adapter"}:
        raise ValueError(f"unknown target arm: {arm}")
    if arm in {"random_full", "source_full"}:
        for parameter in backbone.parameters():
            parameter.requires_grad = True
        return backbone
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    for parameter in backbone.transformer.output.parameters():
        parameter.requires_grad = True
    if arm == "source_head":
        return backbone
    return PharmaFormerWithAdapter(backbone, adapter_rank)
