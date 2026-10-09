"""Model construction helpers for the EXP-046 five-arm comparison.

This module handles model structure and frozen pathway-map alignment only. It
does not load study data, labels, checkpoints, or train a model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.optim import Optimizer

from drug_screen.models.pharmaformer_pathway import PharmaFormerWithPathwayBranch


Exp046Arm = Literal["P-A", "P-C", "P-R", "P-T", "P-N"]


@dataclass(frozen=True)
class PathwayAxisContract:
    """Expected dimensions for a full ordered gene axis and its pathway subset."""

    full_gene_count: int = 19_091
    mapped_gene_count: int = 4_366
    pathway_count: int = 50

    def __post_init__(self) -> None:
        values = (self.full_gene_count, self.mapped_gene_count, self.pathway_count)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in values):
            raise ValueError("pathway axis contract counts must be positive integers")
        if self.mapped_gene_count > self.full_gene_count:
            raise ValueError("mapped gene count cannot exceed the full gene count")


EXP046_PATHWAY_AXIS = PathwayAxisContract()


@dataclass(frozen=True)
class AlignedPathwayMap:
    """A validated binary map expanded onto one complete ordered gene axis."""

    gene_axis: tuple[str, ...]
    mapped_gene_ids: tuple[str, ...]
    pathway_names: tuple[str, ...]
    incidence_matrix: torch.Tensor
    contract: PathwayAxisContract


def _identifiers(values: Sequence[str], *, name: str, expected_count: int
                 ) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of identifiers")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of identifiers") from exc
    if len(result) != expected_count:
        raise ValueError(f"{name} must contain exactly {expected_count} identifiers")
    if any(not isinstance(value, str) or not value or value != value.strip()
           for value in result):
        raise ValueError(f"{name} must contain non-empty, whitespace-trimmed strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique identifiers")
    return result


def _pathway_names(values: Sequence[str], contract: PathwayAxisContract
                   ) -> tuple[str, ...]:
    return _identifiers(values, name="pathway_names",
                        expected_count=contract.pathway_count)


def _gene_positions(gene_axis: Sequence[str], mapped_gene_ids: Sequence[str],
                    contract: PathwayAxisContract
                    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    genes = _identifiers(gene_axis, name="gene_axis",
                         expected_count=contract.full_gene_count)
    mapped = _identifiers(mapped_gene_ids, name="mapped_gene_ids",
                           expected_count=contract.mapped_gene_count)
    position = {gene: index for index, gene in enumerate(genes)}
    try:
        indices = tuple(position[gene] for gene in mapped)
    except KeyError as exc:
        raise ValueError("mapped_gene_ids contains a gene absent from gene_axis") from exc
    if tuple(sorted(indices)) != indices:
        raise ValueError("mapped_gene_ids must follow the order of gene_axis")
    return genes, mapped, indices


def _binary_incidence(value: object, *, expected_shape: tuple[int, int],
                      name: str) -> torch.Tensor:
    try:
        raw = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a numeric pathway-by-gene matrix") from exc
    if raw.is_complex():
        raise ValueError(f"{name} must contain real values")
    if raw.ndim != 2 or tuple(raw.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    matrix = raw.detach().to(dtype=torch.float32).clone()
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} must contain only finite values")
    if not bool(((matrix == 0) | (matrix == 1)).all()):
        raise ValueError(f"{name} must contain binary membership values")
    if bool((matrix.sum(dim=1) == 0).any()):
        raise ValueError(f"{name} must not contain an empty pathway row")
    return matrix


def expand_mapped_pathway_map(
    incidence_matrix: object,
    *,
    gene_axis: Sequence[str],
    mapped_gene_ids: Sequence[str],
    pathway_names: Sequence[str],
    contract: PathwayAxisContract = EXP046_PATHWAY_AXIS,
) -> AlignedPathwayMap:
    """Expand a 50-by-4,366-style map by explicit gene IDs, never by row order.

    ``mapped_gene_ids`` must be the ordered subsequence of ``gene_axis`` used
    for the input matrix columns. Columns for genes outside that subset are
    filled with zero membership.
    """
    genes, mapped, positions = _gene_positions(gene_axis, mapped_gene_ids, contract)
    names = _pathway_names(pathway_names, contract)
    reduced = _binary_incidence(
        incidence_matrix,
        expected_shape=(contract.pathway_count, contract.mapped_gene_count),
        name="incidence_matrix",
    )
    if bool((reduced.sum(dim=0) == 0).any()):
        raise ValueError("every mapped gene must occur in at least one pathway")

    full = torch.zeros(
        (contract.pathway_count, contract.full_gene_count),
        dtype=reduced.dtype,
        device=reduced.device,
    )
    index = torch.tensor(positions, dtype=torch.long, device=reduced.device)
    full[:, index] = reduced
    return AlignedPathwayMap(genes, mapped, names, full, contract)


def align_full_axis_pathway_map(
    incidence_matrix: object,
    *,
    gene_axis: Sequence[str],
    mapped_gene_ids: Sequence[str],
    pathway_names: Sequence[str],
    contract: PathwayAxisContract = EXP046_PATHWAY_AXIS,
) -> AlignedPathwayMap:
    """Validate a full-axis map and reduce/expand it through explicit gene IDs.

    This is for a resource map built on the complete gene axis, such as the
    Hallmark membership matrix. It rejects nonzero membership outside the
    declared mapped-gene subset before returning the aligned full-width map.
    """
    genes, mapped, positions = _gene_positions(gene_axis, mapped_gene_ids, contract)
    names = _pathway_names(pathway_names, contract)
    full_input = _binary_incidence(
        incidence_matrix,
        expected_shape=(contract.pathway_count, contract.full_gene_count),
        name="incidence_matrix",
    )
    mapped_index = torch.tensor(positions, dtype=torch.long, device=full_input.device)
    mapped_columns = full_input.index_select(1, mapped_index)
    mapped_set = set(positions)
    other_positions = tuple(index for index in range(contract.full_gene_count)
                            if index not in mapped_set)
    if other_positions:
        other_index = torch.tensor(other_positions, dtype=torch.long,
                                   device=full_input.device)
        if bool((full_input.index_select(1, other_index) != 0).any()):
            raise ValueError("full-axis map has membership outside mapped_gene_ids")
    return expand_mapped_pathway_map(
        mapped_columns,
        gene_axis=genes,
        mapped_gene_ids=mapped,
        pathway_names=names,
        contract=contract,
    )


def validate_aligned_pathway_map(pathway_map: AlignedPathwayMap) -> None:
    """Recheck a map at the model boundary in case its tensor was mutated."""
    if not isinstance(pathway_map, AlignedPathwayMap):
        raise TypeError("pathway maps must be AlignedPathwayMap instances")
    genes, mapped, positions = _gene_positions(
        pathway_map.gene_axis, pathway_map.mapped_gene_ids, pathway_map.contract)
    names = _pathway_names(pathway_map.pathway_names, pathway_map.contract)
    matrix = _binary_incidence(
        pathway_map.incidence_matrix,
        expected_shape=(pathway_map.contract.pathway_count,
                        pathway_map.contract.full_gene_count),
        name="aligned incidence_matrix",
    )
    mapped_index = torch.tensor(positions, dtype=torch.long, device=matrix.device)
    if bool((matrix.index_select(1, mapped_index).sum(dim=0) == 0).any()):
        raise ValueError("aligned map omits a declared mapped gene")
    mapped_set = set(positions)
    other_positions = tuple(index for index in range(pathway_map.contract.full_gene_count)
                            if index not in mapped_set)
    if other_positions:
        other_index = torch.tensor(other_positions, dtype=torch.long, device=matrix.device)
        if bool((matrix.index_select(1, other_index) != 0).any()):
            raise ValueError("aligned map has membership outside mapped_gene_ids")
    if genes != pathway_map.gene_axis or mapped != pathway_map.mapped_gene_ids:
        raise ValueError("aligned map gene identifiers changed after construction")
    if names != pathway_map.pathway_names:
        raise ValueError("aligned map pathway names changed after construction")


def build_exp046_arm_model(
    arm: Exp046Arm | str,
    backbone: nn.Module,
    *,
    ordinary_control_map: AlignedPathwayMap | None = None,
    frozen_random_map: AlignedPathwayMap | None = None,
    true_pathway_map: AlignedPathwayMap | None = None,
    rank: int | None = None,
    branch_seed: int | None = None,
    zero_initialize: bool | None = None,
) -> nn.Module:
    """Construct one EXP-046 arm using the matching frozen map.

    P-R receives exactly one caller-selected frozen random map. P-C, P-R, P-T,
    and P-N use the same branch class, rank, and seed-controlled cold start.
    Branch rank, seed, and zero-initialization choice are mandatory for branch
    arms so this helper does not freeze pending EXP-046 search choices.
    Pass a separate, identically initialized backbone instance for each arm.
    """
    if not isinstance(backbone, nn.Module):
        raise TypeError("backbone must be a torch module")
    if arm not in ("P-A", "P-C", "P-R", "P-T", "P-N"):
        raise ValueError("arm must be one of P-A, P-C, P-R, P-T, or P-N")
    provided = {
        "P-C": ordinary_control_map,
        "P-R": frozen_random_map,
        "P-T": true_pathway_map,
        "P-N": true_pathway_map,
    }
    required_name = {
        "P-C": "ordinary_control_map",
        "P-R": "frozen_random_map",
        "P-T": "true_pathway_map",
        "P-N": "true_pathway_map",
    }.get(arm)
    all_maps = {
        "ordinary_control_map": ordinary_control_map,
        "frozen_random_map": frozen_random_map,
        "true_pathway_map": true_pathway_map,
    }
    if arm == "P-A":
        if any(value is not None for value in all_maps.values()):
            raise ValueError("P-A must be constructed without a pathway map")
        if any(value is not None for value in (rank, branch_seed, zero_initialize)):
            raise ValueError("P-A does not accept pathway branch initialization settings")
        return backbone

    assert required_name is not None
    selected = provided[arm]
    if selected is None:
        raise ValueError(f"{required_name} is required for {arm}")
    if any(name != required_name and value is not None
           for name, value in all_maps.items()):
        raise ValueError(f"{arm} accepts only {required_name}")
    validate_aligned_pathway_map(selected)

    feature_extractor = getattr(backbone, "feature_extractor", None)
    gene_layer = getattr(feature_extractor, "gene_fc1", None)
    gene_input_size = getattr(gene_layer, "in_features", None)
    if gene_input_size != selected.contract.full_gene_count:
        raise ValueError("backbone gene input width does not match the full gene axis")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer")
    if (isinstance(branch_seed, bool) or not isinstance(branch_seed, int)
            or branch_seed < 0 or branch_seed > 0x7FFF_FFFF_FFFF_FFFF):
        raise ValueError("branch_seed must be a non-negative 63-bit integer")
    if not isinstance(zero_initialize, bool):
        raise TypeError("zero_initialize must be a bool")

    # Restrict branch initialization to the CPU RNG. This keeps construction
    # deterministic without initializing or querying CUDA.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(branch_seed)
        return PharmaFormerWithPathwayBranch(
            backbone,
            selected.incidence_matrix,
            rank=rank,
            zero_initialize=zero_initialize,
            pathway_names=selected.pathway_names,
        )


def capture_cold_start_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy every parameter and buffer to an immutable-by-convention CPU snapshot."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch module")
    state = model.state_dict()
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("model state must contain only tensors")
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def restore_cold_start_state(
    model: nn.Module,
    cold_start_state: Mapping[str, torch.Tensor],
    *,
    optimizer: Optimizer | None = None,
) -> None:
    """Restore all model parameters/buffers and clear optimizer and gradients.

    This is the mechanism P-N can use after matched source-stage computation.
    It performs no source training itself.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch module")
    if not isinstance(cold_start_state, Mapping):
        raise TypeError("cold_start_state must be a state mapping")
    if optimizer is not None and not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a torch optimizer or None")

    current = model.state_dict()
    if set(current) != set(cold_start_state):
        missing = sorted(set(current) - set(cold_start_state))
        extra = sorted(set(cold_start_state) - set(current))
        raise ValueError(f"cold-start state keys differ; missing={missing}, extra={extra}")
    for name, value in cold_start_state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"cold-start state entry {name} must be a tensor")
        if value.shape != current[name].shape or value.dtype != current[name].dtype:
            raise ValueError(f"cold-start state entry {name} has an incompatible shape or dtype")

    model.load_state_dict(
        {name: value.detach().clone() for name, value in cold_start_state.items()},
        strict=True,
    )
    for parameter in model.parameters():
        parameter.grad = None
    if optimizer is not None:
        optimizer.state.clear()
