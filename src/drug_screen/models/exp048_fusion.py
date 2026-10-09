"""Five frozen EXP-048 input arms around the registered PharmaFormer backbone."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


SOURCE_PERTURBATION_DIM = 978
ADDED_OUTPUT_INIT_STD = 1e-4
ARMS = ("F-A", "F-B", "F-F", "F-C", "F-S")


def capacity_matched_rank(gene_hidden_size: int, input_size: int = SOURCE_PERTURBATION_DIM) -> int:
    """Choose the nearest rank matching Linear(input_size, h) parameter count."""
    if gene_hidden_size < 1 or input_size < 1:
        raise ValueError("model dimensions must be positive")
    return max(1, int(round(input_size * gene_hidden_size / (2 * gene_hidden_size + 1))))


def added_parameter_counts(gene_hidden_size: int, rank: int) -> dict[str, int | float]:
    if gene_hidden_size < 1 or rank < 1:
        raise ValueError("model dimensions and adapter rank must be positive")
    fusion = SOURCE_PERTURBATION_DIM * gene_hidden_size + gene_hidden_size
    direct = 2 * gene_hidden_size * rank + rank + gene_hidden_size
    return {
        "F-F_projection": fusion,
        "F-C_residual_adapter": direct,
        "absolute_difference": abs(fusion - direct),
        "relative_difference": abs(fusion - direct) / fusion,
    }


def _init_small_output(layer: nn.Linear, *, generator: torch.Generator | None = None) -> None:
    nn.init.normal_(layer.weight, mean=0.0, std=ADDED_OUTPUT_INIT_STD, generator=generator)
    nn.init.zeros_(layer.bias)


class EXP048FusionArm(nn.Module):
    """Apply one EXP-048 arm while preserving the backbone's sequence geometry.

    ``F-B`` requires ``expression=None``. Its projected perturbation fills the
    positions occupied by the official expression latent; it never evaluates
    the checkpoint's patient-expression encoder.
    """

    def __init__(self, backbone: nn.Module, arm: str, *, seed: int) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown EXP-048 arm: {arm}")
        self.backbone = backbone
        self.arm = arm
        extractor = backbone.feature_extractor
        if not all(hasattr(extractor, name) for name in ("gene_fc1", "gene_fc2", "smiles_fc")):
            raise ValueError("official PharmaFormer feature-extractor interface changed")
        self.gene_hidden_size = int(extractor.gene_fc2.out_features)
        self.drug_hidden_size = int(extractor.smiles_fc.out_features)
        self.feature_dim = int(backbone.feature_dim)
        total_width = self.gene_hidden_size + self.drug_hidden_size
        if (self.gene_hidden_size % self.feature_dim != 0
                or self.drug_hidden_size % self.feature_dim != 0
                or total_width % self.feature_dim != 0):
            raise ValueError("EXP-048 requires complete expression and drug sequence slots")
        self.expression_slots = self.gene_hidden_size // self.feature_dim
        self.drug_slots = self.drug_hidden_size // self.feature_dim
        self.sequence_length = total_width // self.feature_dim
        if int(backbone.seq_len) != self.sequence_length:
            raise ValueError("official sequence length differs from dimensions used by CombinedModel")

        self.capacity_rank = capacity_matched_rank(self.gene_hidden_size)
        self.perturbation_projection: nn.Linear | None = None
        self.adapter_down: nn.Linear | None = None
        self.adapter_up: nn.Linear | None = None

        # Initialize only the new modules from a stable per-seed stream. The
        # official backbone initialization and all caller RNG state are intact.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if arm in {"F-B", "F-F", "F-S"}:
                self.perturbation_projection = nn.Linear(
                    SOURCE_PERTURBATION_DIM, self.gene_hidden_size)
                _init_small_output(self.perturbation_projection)
            if arm == "F-C":
                self.adapter_down = nn.Linear(self.gene_hidden_size, self.capacity_rank)
                self.adapter_up = nn.Linear(self.capacity_rank, self.gene_hidden_size)
                _init_small_output(self.adapter_up)

        if arm == "F-B":
            # These checkpoint layers are not evaluated by the perturbation-only arm.
            for parameter in extractor.gene_fc1.parameters():
                parameter.requires_grad_(False)
            for parameter in extractor.gene_fc2.parameters():
                parameter.requires_grad_(False)

    def encode_expression(self, expression: torch.Tensor) -> torch.Tensor:
        extractor = self.backbone.feature_extractor
        return F.relu(extractor.gene_fc2(F.relu(extractor.gene_fc1(expression))))

    def encode_drug(self, smiles: torch.Tensor) -> torch.Tensor:
        return F.relu(self.backbone.feature_extractor.smiles_fc(smiles))

    def perturbation_delta(self, perturbation: torch.Tensor) -> torch.Tensor:
        if self.perturbation_projection is None:
            raise ValueError(f"arm {self.arm} has no perturbation projection")
        return F.gelu(self.perturbation_projection(perturbation))

    def direct_residual(self, expression_latent: torch.Tensor) -> torch.Tensor:
        if self.arm != "F-C" or self.adapter_down is None or self.adapter_up is None:
            raise ValueError(f"arm {self.arm} has no direct residual adapter")
        return self.adapter_up(F.gelu(self.adapter_down(expression_latent)))

    def forward(
        self,
        expression: torch.Tensor | None,
        smiles: torch.Tensor,
        perturbation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected = (smiles.shape[0], SOURCE_PERTURBATION_DIM)
        if self.arm == "F-B":
            if expression is not None:
                raise ValueError("F-B must not receive or compute PDO expression")
            if perturbation is None or tuple(perturbation.shape) != expected:
                raise ValueError(f"F-B perturbation must have shape {expected}")
            gene = self.perturbation_delta(perturbation)
        else:
            if expression is None:
                raise ValueError(f"arm {self.arm} requires PDO expression")
            gene = self.encode_expression(expression)
            if self.arm in {"F-F", "F-S"}:
                if perturbation is None or tuple(perturbation.shape) != expected:
                    raise ValueError(f"{self.arm} perturbation must have shape {expected}")
                gene = gene + self.perturbation_delta(perturbation)
            elif perturbation is not None:
                raise ValueError(f"arm {self.arm} must not receive perturbation input")
            if self.arm == "F-C":
                gene = gene + self.direct_residual(gene)

        drug = self.encode_drug(smiles)
        features = torch.cat((gene, drug), dim=1)
        if features.shape[1] != self.sequence_length * self.feature_dim:
            raise ValueError("input latent width does not fill the frozen official sequence")
        sequence = features.view(features.shape[0], self.sequence_length, self.feature_dim)
        return self.backbone.transformer(sequence)

    def added_module_parameter_count(self) -> int:
        if self.arm in {"F-B", "F-F", "F-S"}:
            assert self.perturbation_projection is not None
            return sum(parameter.numel() for parameter in self.perturbation_projection.parameters())
        if self.arm == "F-C":
            assert self.adapter_down is not None and self.adapter_up is not None
            return sum(parameter.numel() for parameter in self.adapter_down.parameters()) + sum(
                parameter.numel() for parameter in self.adapter_up.parameters())
        return 0

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def added_module_parameters(self) -> tuple[nn.Parameter, ...]:
        if self.perturbation_projection is not None:
            return tuple(self.perturbation_projection.parameters())
        if self.adapter_down is not None and self.adapter_up is not None:
            return tuple(self.adapter_down.parameters()) + tuple(self.adapter_up.parameters())
        return ()


def check_capacity_tolerance(gene_hidden_size: int, *, maximum_relative_difference: float = 0.002) -> dict[str, int | float]:
    rank = capacity_matched_rank(gene_hidden_size)
    report = added_parameter_counts(gene_hidden_size, rank)
    if float(report["relative_difference"]) > maximum_relative_difference:
        raise ValueError("F-C/F-F added module parameter mismatch exceeds frozen 0.2% gate")
    return {"rank": rank, **report}


def finite_positive_gradient(parameters: tuple[nn.Parameter, ...]) -> dict[str, float | int]:
    if not parameters:
        return {"parameter_tensors": 0, "finite_gradient_tensors": 0, "nonzero_gradient_tensors": 0,
                "gradient_l2": 0.0}
    norms = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise ValueError("EXP-048 added module produced non-finite gradients")
        norms.append(float(torch.linalg.vector_norm(parameter.grad.detach()).cpu()))
    nonzero = sum(value > 0.0 for value in norms)
    total = math.sqrt(sum(value * value for value in norms))
    return {
        "parameter_tensors": len(parameters),
        "finite_gradient_tensors": len(norms),
        "nonzero_gradient_tensors": nonzero,
        "gradient_l2": total,
    }
