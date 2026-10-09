"""EXP-037 的共享药物效应与细胞状态低秩交互网络。"""

from __future__ import annotations

import torch
from torch import nn


class SharedEffectInteraction(nn.Module):
    """Predict standardized delta from control state, drug fingerprint and exposure."""

    def __init__(self, *, n_genes: int = 978, drug_features: int = 2048,
                 exposure_features: int = 2, hidden: int = 256, rank: int = 32):
        super().__init__()
        condition_features = drug_features + exposure_features
        self.shared = nn.Sequential(
            nn.Linear(condition_features, hidden), nn.GELU(), nn.Linear(hidden, n_genes))
        self.state_encoder = nn.Linear(n_genes, rank)
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_features, hidden), nn.GELU(), nn.Linear(hidden, rank))
        self.interaction_decoder = nn.Linear(rank, n_genes, bias=False)

    def forward(self, control_state: torch.Tensor, drug: torch.Tensor,
                exposure: torch.Tensor, *, use_state: bool = True) -> torch.Tensor:
        condition = torch.cat([drug, exposure], dim=1)
        shared = self.shared(condition)
        state_code = self.state_encoder(control_state)
        if not use_state:
            state_code = torch.zeros_like(state_code)
        interaction = self.interaction_decoder(state_code * self.condition_encoder(condition))
        return shared + interaction
