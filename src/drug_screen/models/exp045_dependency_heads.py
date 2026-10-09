"""Shared PharmaFormer expression representation with EXP-045 output heads."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as functional


class PharmaFormerDependencyOutput(NamedTuple):
    """Predictions for the two independently scaled dependency resources."""

    depmap_chronos: torch.Tensor
    nature_crispr_lfc: torch.Tensor


class PharmaFormerAuxiliaryOutput(NamedTuple):
    """Expression reconstruction and both separate dependency predictions."""

    expression_reconstruction: torch.Tensor
    depmap_chronos: torch.Tensor
    nature_crispr_lfc: torch.Tensor


class PharmaFormerMultiTaskOutput(NamedTuple):
    """Original drug-sensitivity prediction and both auxiliary predictions."""

    drug_sensitivity: torch.Tensor
    depmap_chronos: torch.Tensor
    nature_crispr_lfc: torch.Tensor


def _positive_dimension(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class PharmaFormerWithDependencyHeads(nn.Module):
    """Attach separate DepMap and Nature heads to PharmaFormer's gene encoder.

    The drug-sensitivity path delegates directly to the wrapped PharmaFormer,
    preserving its original forward computation and output. Both dependency
    heads use the same ``gene_fc1``/``gene_fc2`` parameters as its expression
    branch, while retaining separate output layers and target dimensions.
    """

    def __init__(self, backbone: nn.Module, *, depmap_target_dim: int,
                 nature_target_dim: int) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a PyTorch module")
        if not callable(backbone):
            raise ValueError("backbone must be callable")

        feature_extractor = getattr(backbone, "feature_extractor", None)
        gene_fc1 = getattr(feature_extractor, "gene_fc1", None)
        gene_fc2 = getattr(feature_extractor, "gene_fc2", None)
        smiles_fc = getattr(feature_extractor, "smiles_fc", None)
        if not all(isinstance(layer, nn.Module) and callable(layer)
                   for layer in (gene_fc1, gene_fc2, smiles_fc)):
            raise ValueError(
                "backbone must expose callable feature_extractor.gene_fc1, "
                "gene_fc2, and smiles_fc modules"
            )

        gene_input_dim = getattr(gene_fc1, "in_features", None)
        gene_hidden_dim = getattr(gene_fc1, "out_features", None)
        second_input_dim = getattr(gene_fc2, "in_features", None)
        second_output_dim = getattr(gene_fc2, "out_features", None)
        smiles_input_dim = getattr(smiles_fc, "in_features", None)
        for name, value in (
            ("gene input dimension", gene_input_dim),
            ("gene hidden dimension", gene_hidden_dim),
            ("second gene input dimension", second_input_dim),
            ("expression representation dimension", second_output_dim),
            ("drug structure dimension", smiles_input_dim),
        ):
            _positive_dimension(name, value)
        if gene_hidden_dim != second_input_dim:
            raise ValueError("gene_fc1 output width must match gene_fc2 input width")

        depmap_target_dim = _positive_dimension("depmap_target_dim", depmap_target_dim)
        nature_target_dim = _positive_dimension("nature_target_dim", nature_target_dim)

        self.backbone = backbone
        self.gene_input_dim = gene_input_dim
        self.expression_dim = second_output_dim
        self.smiles_input_dim = smiles_input_dim
        self.depmap_target_dim = depmap_target_dim
        self.nature_target_dim = nature_target_dim
        self.expression_reconstruction_head = nn.Linear(second_output_dim, gene_input_dim)
        self.depmap_chronos_head = nn.Linear(second_output_dim, depmap_target_dim)
        self.nature_crispr_lfc_head = nn.Linear(second_output_dim, nature_target_dim)

    def _validate_expression(self, gene_expr: torch.Tensor) -> None:
        if not isinstance(gene_expr, torch.Tensor):
            raise TypeError("gene_expr must be a torch tensor")
        if gene_expr.ndim != 2:
            raise ValueError("gene_expr must be a 2D batch-by-gene tensor")
        if gene_expr.shape[0] < 1:
            raise ValueError("gene_expr batch must be non-empty")
        if gene_expr.shape[1] != self.gene_input_dim:
            raise ValueError("gene_expr feature count does not match gene_fc1 input")
        if not gene_expr.is_floating_point():
            raise ValueError("gene_expr must use a floating point dtype")

    def _validate_inputs(self, gene_expr: torch.Tensor,
                         smiles: torch.Tensor) -> None:
        self._validate_expression(gene_expr)
        if not isinstance(smiles, torch.Tensor):
            raise TypeError("smiles must be a torch tensor")
        if smiles.ndim != 2:
            raise ValueError("smiles must be a 2D batch-by-feature tensor")
        if smiles.shape[0] != gene_expr.shape[0]:
            raise ValueError("gene_expr and smiles must have the same batch size")
        if smiles.shape[1] != self.smiles_input_dim:
            raise ValueError("smiles feature count does not match smiles_fc input")
        if not smiles.is_floating_point():
            raise ValueError("smiles must use a floating point dtype")
        if smiles.device != gene_expr.device or smiles.dtype != gene_expr.dtype:
            raise ValueError("gene_expr and smiles must share device and dtype")

    def encode_expression(self, gene_expr: torch.Tensor) -> torch.Tensor:
        """Return the official two-layer PharmaFormer expression representation."""
        self._validate_expression(gene_expr)
        feature_extractor = self.backbone.feature_extractor
        representation = functional.relu(feature_extractor.gene_fc1(gene_expr))
        return functional.relu(feature_extractor.gene_fc2(representation))

    def _predict_drug_sensitivity(self, gene_expr: torch.Tensor,
                                  smiles: torch.Tensor) -> torch.Tensor:
        prediction = self.backbone(gene_expr, smiles)
        if (not isinstance(prediction, torch.Tensor)
                or prediction.shape not in ((gene_expr.shape[0],),
                                            (gene_expr.shape[0], 1))):
            raise ValueError(
                "backbone must return one drug-sensitivity value per sample"
            )
        return prediction

    def predict_drug_sensitivity(self, gene_expr: torch.Tensor,
                                 smiles: torch.Tensor) -> torch.Tensor:
        """Infer drug sensitivity using only pretreatment expression and drug structure."""
        self._validate_inputs(gene_expr, smiles)
        return self._predict_drug_sensitivity(gene_expr, smiles)

    def _predict_dependency_heads(self,
                                 gene_expr: torch.Tensor) -> PharmaFormerDependencyOutput:
        representation = self.encode_expression(gene_expr)
        return PharmaFormerDependencyOutput(
            depmap_chronos=self.depmap_chronos_head(representation),
            nature_crispr_lfc=self.nature_crispr_lfc_head(representation),
        )

    def _predict_auxiliary_heads(
        self, gene_expr: torch.Tensor,
    ) -> PharmaFormerAuxiliaryOutput:
        representation = self.encode_expression(gene_expr)
        return PharmaFormerAuxiliaryOutput(
            expression_reconstruction=self.expression_reconstruction_head(representation),
            depmap_chronos=self.depmap_chronos_head(representation),
            nature_crispr_lfc=self.nature_crispr_lfc_head(representation),
        )

    def predict_dependency_heads(self,
                                 gene_expr: torch.Tensor) -> PharmaFormerDependencyOutput:
        """Predict each dependency resource from expression alone."""
        self._validate_expression(gene_expr)
        return self._predict_dependency_heads(gene_expr)

    def predict_auxiliary_heads(
        self, gene_expr: torch.Tensor,
    ) -> PharmaFormerAuxiliaryOutput:
        """Predict the fold-scaled expression target and dependency domains.

        The reconstruction output uses the same ordered expression feature
        axis accepted by ``gene_fc1``. Fold-specific expression scaling and
        missing-value masks are supplied by the caller.
        """
        self._validate_expression(gene_expr)
        return self._predict_auxiliary_heads(gene_expr)

    def forward(self, gene_expr: torch.Tensor,
                smiles: torch.Tensor) -> PharmaFormerMultiTaskOutput:
        """Return the original drug score plus the two separate dependency outputs."""
        self._validate_inputs(gene_expr, smiles)
        drug_sensitivity = self._predict_drug_sensitivity(gene_expr, smiles)
        dependencies = self._predict_dependency_heads(gene_expr)
        return PharmaFormerMultiTaskOutput(
            drug_sensitivity=drug_sensitivity,
            depmap_chronos=dependencies.depmap_chronos,
            nature_crispr_lfc=dependencies.nature_crispr_lfc,
        )
