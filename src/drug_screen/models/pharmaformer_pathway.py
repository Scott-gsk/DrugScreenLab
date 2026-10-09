"""A small, drug-conditioned pathway branch for the official PharmaFormer."""

from __future__ import annotations

from math import sqrt
from typing import NamedTuple, Sequence

import torch
from torch import nn
from torch.nn import functional as functional


class PharmaFormerPathwayOutput(NamedTuple):
    """Prediction decomposition returned by ``forward_components``.

    ``prediction`` and ``base_prediction`` retain the backbone's scalar output
    shape, either ``(batch,)`` or ``(batch, 1)``. ``pathway_contributions`` has
    shape ``(batch, pathways)`` and its row sum is the pathway adjustment.
    """

    prediction: torch.Tensor
    base_prediction: torch.Tensor
    pathway_contributions: torch.Tensor


class DrugConditionedPathwayBranch(nn.Module):
    """Score each pathway activity using a shared low-rank drug code."""

    def __init__(self, n_pathways: int, drug_features: int, rank: int,
                 *, zero_initialize: bool) -> None:
        super().__init__()
        if n_pathways < 1 or drug_features < 1 or rank < 1:
            raise ValueError("pathway count, drug feature count, and rank must be positive")

        self.pathway_factors = nn.Parameter(torch.empty(n_pathways, rank))
        self.drug_projection = nn.Linear(drug_features, rank, bias=False)
        if zero_initialize:
            nn.init.zeros_(self.pathway_factors)
        else:
            nn.init.xavier_uniform_(self.pathway_factors)

    def forward(self, pathway_activity: torch.Tensor,
                drug_encoding: torch.Tensor) -> torch.Tensor:
        drug_code = torch.tanh(self.drug_projection(drug_encoding))
        coefficients = torch.einsum("pr,br->bp", self.pathway_factors, drug_code)
        coefficients = coefficients / sqrt(self.pathway_factors.shape[1])
        return pathway_activity * coefficients


class PharmaFormerWithPathwayBranch(nn.Module):
    """Wrap an official PharmaFormer with an additive pathway contribution.

    The same interface accepts true or random pathway-by-gene incidence maps.
    The provided map is checked as binary membership, row-normalized, detached,
    and stored as a fixed model buffer. By default ``forward`` returns only the
    prediction, preserving the official model's tensor interface. Call
    ``forward_components`` (or pass ``return_components=True``) to inspect the
    base score and each drug-conditioned pathway contribution.
    """

    def __init__(self, backbone: nn.Module, incidence_matrix: torch.Tensor,
                 *, rank: int = 8, zero_initialize: bool = False,
                 pathway_names: Sequence[str] | None = None) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module) or not callable(backbone):
            raise ValueError("backbone must be a callable PyTorch module")
        feature_extractor = getattr(backbone, "feature_extractor", None)
        smiles_fc = getattr(feature_extractor, "smiles_fc", None)
        if not isinstance(smiles_fc, nn.Module):
            raise ValueError("backbone must expose feature_extractor.smiles_fc")
        drug_features = getattr(smiles_fc, "out_features", None)
        if not isinstance(drug_features, int) or drug_features < 1:
            raise ValueError("backbone feature_extractor.smiles_fc must expose out_features")

        try:
            raw_matrix = torch.as_tensor(incidence_matrix)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("incidence_matrix must be a numeric pathway-by-gene matrix") from exc
        if raw_matrix.is_complex():
            raise ValueError("incidence_matrix must contain real values")
        if raw_matrix.ndim != 2 or min(raw_matrix.shape, default=0) < 1:
            raise ValueError("incidence_matrix must be a non-empty 2D pathway-by-gene matrix")

        matrix = raw_matrix.detach().to(dtype=torch.float32).clone()
        if not bool(torch.isfinite(matrix).all()):
            raise ValueError("incidence_matrix must contain only finite values")
        if not bool(((matrix == 0) | (matrix == 1)).all()):
            raise ValueError("incidence_matrix must contain binary membership values")
        row_sizes = matrix.sum(dim=1)
        if bool((row_sizes == 0).any()):
            raise ValueError("every pathway row must contain at least one gene")

        gene_fc1 = getattr(feature_extractor, "gene_fc1", None)
        expected_genes = getattr(gene_fc1, "in_features", None)
        if isinstance(expected_genes, int) and expected_genes != matrix.shape[1]:
            raise ValueError(
                "incidence_matrix gene count does not match backbone feature_extractor.gene_fc1"
            )

        if pathway_names is None:
            names = tuple(f"pathway_{index:03d}" for index in range(matrix.shape[0]))
        else:
            if isinstance(pathway_names, str):
                raise ValueError("pathway_names must be a sequence of pathway names")
            names = tuple(pathway_names)
            if (len(names) != matrix.shape[0]
                    or any(not isinstance(name, str) or not name.strip() for name in names)
                    or len(set(names)) != len(names)):
                raise ValueError("pathway_names must be unique non-empty names matching matrix rows")

        self.backbone = backbone
        self.pathway_names = names
        self.register_buffer("pathway_gene_weights", matrix / row_sizes.unsqueeze(1))
        self.branch = DrugConditionedPathwayBranch(
            n_pathways=matrix.shape[0], drug_features=drug_features, rank=rank,
            zero_initialize=zero_initialize)

    def _validate_inputs(self, gene_expr: torch.Tensor,
                         smiles: torch.Tensor) -> None:
        if not isinstance(gene_expr, torch.Tensor) or not isinstance(smiles, torch.Tensor):
            raise TypeError("gene_expr and smiles must be torch tensors")
        if gene_expr.ndim != 2 or smiles.ndim != 2:
            raise ValueError("gene_expr and smiles must be 2D batch-by-feature tensors")
        if gene_expr.shape[0] < 1 or gene_expr.shape[0] != smiles.shape[0]:
            raise ValueError("gene_expr and smiles must have the same non-empty batch")
        if gene_expr.shape[1] != self.pathway_gene_weights.shape[1]:
            raise ValueError("gene_expr feature count does not match incidence_matrix columns")
        expected_smiles = getattr(self.backbone.feature_extractor.smiles_fc, "in_features", None)
        if isinstance(expected_smiles, int) and smiles.shape[1] != expected_smiles:
            raise ValueError("smiles feature count does not match backbone smiles_fc input")
        if not gene_expr.is_floating_point() or not smiles.is_floating_point():
            raise ValueError("gene_expr and smiles must use floating point dtypes")
        if gene_expr.device != smiles.device or gene_expr.dtype != smiles.dtype:
            raise ValueError("gene_expr and smiles must share device and dtype")
        if gene_expr.device != self.pathway_gene_weights.device:
            raise ValueError("gene_expr and smiles must be on the model device")
        if not bool(torch.isfinite(gene_expr).all()) or not bool(torch.isfinite(smiles).all()):
            raise ValueError("gene_expr and smiles must contain only finite values")

    def forward_components(self, gene_expr: torch.Tensor,
                           smiles: torch.Tensor) -> PharmaFormerPathwayOutput:
        """Return the additive prediction and its per-pathway decomposition."""
        self._validate_inputs(gene_expr, smiles)

        base_prediction = self.backbone(gene_expr, smiles)
        batch_size = gene_expr.shape[0]
        if (not isinstance(base_prediction, torch.Tensor)
                or base_prediction.shape not in ((batch_size,), (batch_size, 1))):
            raise ValueError("backbone must return exactly one scalar prediction per sample")
        if not bool(torch.isfinite(base_prediction).all()):
            raise ValueError("backbone returned non-finite predictions")

        pathway_activity = functional.linear(
            gene_expr, self.pathway_gene_weights.to(dtype=gene_expr.dtype))
        drug_encoding = functional.relu(self.backbone.feature_extractor.smiles_fc(smiles))
        if (drug_encoding.ndim != 2 or drug_encoding.shape[0] != batch_size
                or drug_encoding.shape[1] != self.branch.drug_projection.in_features):
            raise ValueError("backbone smiles_fc must return the expected batch-by-feature tensor")
        if not bool(torch.isfinite(drug_encoding).all()):
            raise ValueError("backbone smiles_fc returned non-finite drug features")

        pathway_contributions = self.branch(pathway_activity, drug_encoding)
        if not bool(torch.isfinite(pathway_contributions).all()):
            raise ValueError("pathway branch returned non-finite contributions")
        prediction = (base_prediction.reshape(batch_size)
                      + pathway_contributions.sum(dim=1))
        prediction = prediction.reshape_as(base_prediction)
        return PharmaFormerPathwayOutput(
            prediction=prediction,
            base_prediction=base_prediction,
            pathway_contributions=pathway_contributions,
        )

    def forward(self, gene_expr: torch.Tensor, smiles: torch.Tensor,
                *, return_components: bool = False
                ) -> torch.Tensor | PharmaFormerPathwayOutput:
        """Return the prediction tensor, optionally with inspectable components."""
        components = self.forward_components(gene_expr, smiles)
        return components if return_components else components.prediction
