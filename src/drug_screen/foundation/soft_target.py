"""Minimal EXP-009 soft-target components.

The components in this module are data-contract agnostic.  They consume only
already-provided structure or mechanism features and intentionally perform no
dataset loading, checkpoint loading, or training orchestration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from torch import Tensor, nn
import torch


class SoftTargetHead(nn.Module):
    """Map a frozen structure representation to target probabilities and confidence."""

    def __init__(self, structure_dim: int, target_dim: int = 256) -> None:
        super().__init__()
        self.target_logits = nn.Linear(structure_dim, target_dim)
        self.confidence_logit = nn.Linear(structure_dim, 1)

    def forward(self, structure_features: Tensor) -> tuple[Tensor, Tensor]:
        """Return bounded per-target probabilities and per-sample confidence."""
        probabilities = torch.sigmoid(self.target_logits(structure_features))
        confidence = torch.sigmoid(self.confidence_logit(structure_features))
        return probabilities, confidence


class MechanismResidual(nn.Module):
    """Zero-gated mechanism residual that preserves a supplied baseline at start."""

    def __init__(
        self,
        feature_dim: int = 385,
        hidden_dim: int = 256,
        n_genes: int = 978,
    ) -> None:
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU())
        self.output = nn.Linear(hidden_dim, n_genes)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.raw_gamma = nn.Parameter(torch.zeros(()))

    def forward(self, features: Tensor, baseline: Tensor) -> Tensor:
        """Add a learnable, initially inactive mechanism residual to ``baseline``."""
        if features.ndim != 2:
            raise ValueError("mechanism features must be a rank-2 tensor")
        expected_feature_dim = self.hidden[0].in_features
        if features.shape[1] != expected_feature_dim:
            raise ValueError(
                "mechanism feature width must match "
                f"feature_dim={expected_feature_dim}, received {features.shape[1]}"
            )
        residual = self.output(self.hidden(features))
        gamma = 0.05 * torch.tanh(self.raw_gamma).to(dtype=residual.dtype)
        return baseline + gamma * residual


def count_trainable_parameters(module: nn.Module) -> int:
    """Return the exact number of parameters with gradient updates enabled."""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class _EXP009ResidualWrapper(nn.Module):
    """Attach a zero-gated residual to the Delta978 prediction of frozen XPert.

    The wrapper neither changes XPert's input contract nor its upstream output
    contract: it replaces only output index 2 (the official Delta978 tensor).
    """

    def __init__(
        self,
        official: nn.Module,
        *,
        variant: Literal["B", "C"],
        structure_dim: int,
        soft_target_dim: int,
        hidden_dim: int,
        n_genes: int,
    ) -> None:
        super().__init__()
        self.official = official
        self.variant = variant
        if variant not in {"B", "C"}:  # pragma: no cover - public builder validates
            raise ValueError("EXP-009 residual variant must be B or C")
        # Both overlays consume an identically sized residual input. B carries
        # an exact-zero 64-d placeholder; C replaces it with teacher outputs.
        feature_dim = structure_dim + soft_target_dim
        self.residual = MechanismResidual(
            feature_dim=feature_dim, hidden_dim=hidden_dim, n_genes=n_genes
        )
        # B gets a zero-valued parameter-free placeholder to ensure C uses the
        # same residual dimensions/trainable parameter count without adding an
        # unapproved feature stream.
        self.structure_only_padding_dim = soft_target_dim if variant == "B" else 0
        for parameter in self.official.parameters():
            parameter.requires_grad_(False)
        self.official_parameters_frozen = True

    def _residual_features(
        self,
        *,
        structure_features: Tensor | None,
        soft_targets: Tensor | None,
    ) -> Tensor:
        if structure_features is None:
            raise ValueError("EXP-009 residual requires structure_features")
        if structure_features.ndim != 2:
            raise ValueError("structure_features must be rank-2")
        if self.variant == "B":
            padding = structure_features.new_zeros(
                (structure_features.shape[0], self.structure_only_padding_dim)
            )
            return torch.cat((structure_features, padding), dim=1)
        if soft_targets is None:
            raise ValueError("EXP-009 soft-target residual requires soft_targets")
        if soft_targets.ndim != 2:
            raise ValueError("soft_targets must be rank-2")
        if soft_targets.shape[0] != structure_features.shape[0]:
            raise ValueError("structure_features and soft_targets must share batch size")
        return torch.cat((structure_features, soft_targets), dim=1)

    def forward(
        self,
        data: Any,
        *,
        structure_features: Tensor | None = None,
        soft_targets: Tensor | None = None,
    ) -> Any:
        # Running the officially frozen module normally preserves the official
        # runtime behavior while autograd holds no gradients for its weights.
        output = self.official(data)
        if not isinstance(output, Sequence) or len(output) < 3:
            raise ValueError("official XPert output must be a sequence containing Delta978 at index 2")
        features = self._residual_features(
            structure_features=structure_features, soft_targets=soft_targets
        )
        delta = self.residual(features.to(output[2]), output[2])
        if isinstance(output, tuple):
            return (*output[:2], delta, *output[3:])
        result = list(output)
        result[2] = delta
        return result

    def gradient_audit(self) -> dict[str, Any]:
        """Return a checkpoint-safe audit proving XPert remains frozen."""
        official_trainable = [
            name for name, parameter in self.official.named_parameters() if parameter.requires_grad
        ]
        official_with_grad = [
            name for name, parameter in self.official.named_parameters() if parameter.grad is not None
        ]
        residual_gradient_norms = {
            name: float(parameter.grad.detach().norm().cpu()) if parameter.grad is not None else 0.0
            for name, parameter in self.residual.named_parameters()
        }
        return {
            "official_parameters_frozen": self.official_parameters_frozen,
            "official_trainable_parameter_count": sum(
                parameter.numel() for parameter in self.official.parameters() if parameter.requires_grad
            ),
            "official_parameters_with_grad": official_with_grad,
            "official_trainable_parameter_names": official_trainable,
            "residual_trainable_parameter_count": count_trainable_parameters(self.residual),
            "residual_gradient_norms": residual_gradient_norms,
            "raw_gamma": float(self.residual.raw_gamma.detach().cpu()),
            "residual_output_zero_initialized": bool(
                torch.count_nonzero(self.residual.output.weight).item() == 0
                and torch.count_nonzero(self.residual.output.bias).item() == 0
            ),
        }


def build_exp009_residual_wrapper(
    official: nn.Module,
    *,
    variant: Literal["B", "C"],
    structure_dim: int,
    soft_target_dim: int = 64,
    hidden_dim: int = 256,
    n_genes: int = 978,
) -> _EXP009ResidualWrapper:
    """Build a parameter-count-matched, fully frozen-XPert EXP-009 overlay."""
    if variant not in {"B", "C"}:
        raise ValueError("EXP-009 residual variant must be B or C")
    # Parameter matching: both use structure_dim + 64 inputs. B replaces the
    # contract-approved soft-target channel with exact zeros, C receives 64-d
    # teacher outputs. The trainable residual architecture is otherwise exact.
    return _EXP009ResidualWrapper(
        official,
        variant=variant,
        structure_dim=structure_dim,
        soft_target_dim=soft_target_dim,
        hidden_dim=hidden_dim,
        n_genes=n_genes,
    )
