"""Deterministic, label-blind pathway control maps for EXP-046."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np


PC_SALT = "EXP-046-PC-ordinary-v1"
PR_PREFIX = "EXP-046-PR-v1"
PR_MIN_ACCEPTED_PER_EDGE = 20
PR_MAX_PROPOSALS_PER_EDGE = 2_000
PR_MIN_EDGE_HAMMING_FRACTION = 0.90


@dataclass(frozen=True)
class PathwayControlMap:
    """A binary pathway-by-gene control map and its construction receipt."""

    gene_ids: tuple[str, ...]
    pathway_names: tuple[str, ...]
    incidence: np.ndarray
    edges: tuple[tuple[str, str], ...]
    edge_sha256: str
    method: str
    metadata: dict[str, int | float | str]


def _sha256_integer(payload: str) -> int:
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest(), "big")


def _edge_digest(edges: Sequence[tuple[str, str]]) -> str:
    ordered = sorted(edges, key=lambda edge: (edge[0].encode("utf-8"), edge[1].encode("utf-8")))
    payload = "".join(f"{gene}\t{pathway}\n" for gene, pathway in ordered).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_axis(gene_ids: Sequence[str], pathway_names: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    genes = tuple(gene_ids)
    pathways = tuple(pathway_names)
    if not genes or not pathways:
        raise ValueError("gene and pathway axes must be non-empty")
    if any(not isinstance(value, str) or not value for value in genes + pathways):
        raise ValueError("gene and pathway identifiers must be non-empty strings")
    if len(set(genes)) != len(genes) or len(set(pathways)) != len(pathways):
        raise ValueError("gene and pathway axes must contain unique identifiers")
    return genes, pathways


def _slot_targets(n_edges: int, n_slots: int) -> tuple[int, ...]:
    quotient, remainder = divmod(n_edges, n_slots)
    return tuple(quotient + (slot < remainder) for slot in range(n_slots))


def build_ordinary_sparse_control(
    gene_ids: Sequence[str],
    *,
    n_slots: int = 50,
    n_edges: int = 7_299,
    salt: str = PC_SALT,
) -> PathwayControlMap:
    """Build P-C using anonymous slots and exact balanced slot capacities.

    Genes receive one deterministic base edge by hash rank modulo ``n_slots``.
    The first ``n_edges - n_genes`` genes in that order receive one additional
    edge. NetworkX's frozen ``network_simplex`` implementation assigns those
    edges to meet the remaining balanced slot capacities at minimum SHA-256
    integer cost. No pathway membership or response label is used.
    """

    genes, _ = _validate_axis(gene_ids, ("anonymous",))
    if n_slots < 1 or n_edges < len(genes) or n_edges > 2 * len(genes):
        raise ValueError("n_slots and n_edges must permit one or two edges per gene")
    if not isinstance(salt, str) or not salt:
        raise ValueError("salt must be a non-empty string")

    ranked = sorted(
        genes,
        key=lambda gene: (
            hashlib.sha256(f"{salt}|gene|{gene}".encode("utf-8")).digest(),
            gene.encode("utf-8"),
        ),
    )
    n_second_edges = n_edges - len(ranked)
    slot_for_gene = {gene: rank % n_slots for rank, gene in enumerate(ranked)}
    base_counts = [0] * n_slots
    for slot in slot_for_gene.values():
        base_counts[slot] += 1

    targets = _slot_targets(n_edges, n_slots)
    remaining = tuple(target - current for target, current in zip(targets, base_counts))
    if any(value < 0 for value in remaining) or sum(remaining) != n_second_edges:
        raise ValueError("requested slot capacities cannot be reached from the base assignment")
    if not n_second_edges:
        edges = tuple((gene, f"control_slot_{slot_for_gene[gene]:02d}") for gene in ranked)
        return _ordinary_result(genes, n_slots, edges, salt, targets, n_second_edges, 0)

    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - environment contract guard
        raise RuntimeError("NetworkX is required to construct the frozen P-C minimum-cost flow") from exc

    source = ("source",)
    sink = ("sink",)
    gene_nodes = {gene: ("gene", rank) for rank, gene in enumerate(ranked[:n_second_edges])}
    slot_nodes = [("slot", slot) for slot in range(n_slots)]
    graph = nx.DiGraph()
    graph.add_node(source, demand=-n_second_edges)
    for gene in ranked[:n_second_edges]:
        graph.add_node(gene_nodes[gene], demand=0)
    for node in slot_nodes:
        graph.add_node(node, demand=0)
    graph.add_node(sink, demand=n_second_edges)

    for gene in ranked[:n_second_edges]:
        graph.add_edge(source, gene_nodes[gene], capacity=1, weight=0)
    for gene in ranked[:n_second_edges]:
        base_slot = slot_for_gene[gene]
        for slot in range(n_slots):
            if slot == base_slot:
                continue
            weight = _sha256_integer(f"{salt}|second|{gene}|{slot}")
            graph.add_edge(gene_nodes[gene], slot_nodes[slot], capacity=1, weight=weight)
    for slot, node in enumerate(slot_nodes):
        graph.add_edge(node, sink, capacity=remaining[slot], weight=0)

    try:
        total_cost, flow = nx.network_simplex(graph)
    except (nx.NetworkXUnfeasible, nx.NetworkXError) as exc:
        raise RuntimeError("P-C minimum-cost flow failed; keep EXP-046 at NO-GO") from exc

    edges = [(gene, f"control_slot_{slot_for_gene[gene]:02d}") for gene in ranked]
    second_edges: list[tuple[str, int]] = []
    for gene in ranked[:n_second_edges]:
        gene_node = gene_nodes[gene]
        selected = [slot for slot, node in enumerate(slot_nodes) if flow[gene_node].get(node, 0) == 1]
        if len(selected) != 1:
            raise RuntimeError("P-C flow did not assign exactly one second slot per selected gene")
        edges.append((gene, f"control_slot_{selected[0]:02d}"))
        second_edges.append((gene, selected[0]))

    recomputed_cost = sum(
        _sha256_integer(f"{salt}|second|{gene}|{slot}")
        for gene, slot in second_edges
    )
    if total_cost != recomputed_cost:
        raise RuntimeError("P-C reported minimum cost does not match the selected edges")
    return _ordinary_result(genes, n_slots, tuple(edges), salt, targets, n_second_edges, total_cost)


def _ordinary_result(
    genes: tuple[str, ...],
    n_slots: int,
    edges: Sequence[tuple[str, str]],
    salt: str,
    targets: Sequence[int],
    n_second_edges: int,
    total_cost: int,
) -> PathwayControlMap:
    slot_names = tuple(f"control_slot_{slot:02d}" for slot in range(n_slots))
    matrix = np.zeros((n_slots, len(genes)), dtype=np.uint8)
    gene_index = {gene: index for index, gene in enumerate(genes)}
    slot_index = {slot: index for index, slot in enumerate(slot_names)}
    for gene, slot in edges:
        if matrix[slot_index[slot], gene_index[gene]]:
            raise RuntimeError("P-C contains a duplicate gene-slot edge")
        matrix[slot_index[slot], gene_index[gene]] = 1
    if int(matrix.sum()) != len(edges) or tuple(map(int, matrix.sum(axis=1))) != tuple(targets):
        raise RuntimeError("P-C failed edge-count or slot-capacity validation")
    if not np.all(matrix.sum(axis=0) >= 1) or not np.all(matrix.sum(axis=0) <= 2):
        raise RuntimeError("P-C failed gene-coverage or degree validation")
    digest = _edge_digest(edges)
    return PathwayControlMap(
        gene_ids=genes,
        pathway_names=slot_names,
        incidence=matrix,
        edges=tuple(edges),
        edge_sha256=digest,
        method="minimum_cost_balanced_anonymous_sparse_partition",
        metadata={
            "salt": salt,
            "slot_count": n_slots,
            "edge_count": len(edges),
            "second_edge_count": n_second_edges,
            "slot_targets": ",".join(map(str, targets)),
            "minimum_cost_sha256_integer": str(total_cost),
        },
    )


def randomize_bipartite_pathways(
    gene_ids: Sequence[str],
    pathway_names: Sequence[str],
    incidence: np.ndarray,
    *,
    seed: int,
    min_accepted_swaps_per_edge: int = PR_MIN_ACCEPTED_PER_EDGE,
    min_edge_hamming_fraction: float = PR_MIN_EDGE_HAMMING_FRACTION,
    max_proposals_per_edge: int = PR_MAX_PROPOSALS_PER_EDGE,
) -> PathwayControlMap:
    """Degree-preserving P-R randomization with a label-blind hash proposal stream."""

    genes, pathways = _validate_axis(gene_ids, pathway_names)
    matrix = np.asarray(incidence)
    if matrix.ndim != 2 or matrix.shape != (len(pathways), len(genes)):
        raise ValueError("incidence must be pathway-by-gene and match the supplied axes")
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError("incidence must contain only binary membership values")
    original = matrix.astype(np.uint8, copy=True)
    if not np.all(original.sum(axis=0) > 0) or not np.all(original.sum(axis=1) > 0):
        raise ValueError("P-R input must contain only covered genes and non-empty pathways")
    if seed < 0 or min_accepted_swaps_per_edge < 0 or max_proposals_per_edge < 1:
        raise ValueError("seed and swap/proposal limits are invalid")
    if not 0 < min_edge_hamming_fraction <= 1:
        raise ValueError("min_edge_hamming_fraction must be in (0, 1]")

    gene_index = {gene: index for index, gene in enumerate(genes)}
    pathway_index = {pathway: index for index, pathway in enumerate(pathways)}
    true_edges = {
        (gene_index[gene], pathway_index[pathway])
        for pathway in pathways
        for gene in genes
        if original[pathway_index[pathway], gene_index[gene]]
    }
    edge_list = sorted(
        true_edges,
        key=lambda edge: (genes[edge[0]].encode("utf-8"), pathways[edge[1]].encode("utf-8")),
    )
    edge_count = len(edge_list)
    min_accepted = min_accepted_swaps_per_edge * edge_count
    max_proposals = max_proposals_per_edge * edge_count
    if min_accepted > max_proposals:
        raise ValueError("maximum proposals cannot reach the minimum accepted-swap count")

    adjacency = set(edge_list)
    true_intersection = edge_count
    accepted = 0
    proposal = 0
    target_fraction = float(min_edge_hamming_fraction)
    while proposal < max_proposals:
        left = _proposal_index(seed, proposal, "0", edge_count)
        right = _proposal_index(seed, proposal, "1", edge_count)
        proposal += 1
        if left == right:
            continue
        gene_left, pathway_left = edge_list[left]
        gene_right, pathway_right = edge_list[right]
        if gene_left == gene_right or pathway_left == pathway_right:
            continue
        cross_left = (gene_left, pathway_right)
        cross_right = (gene_right, pathway_left)
        if cross_left in adjacency or cross_right in adjacency:
            continue

        old_left = edge_list[left]
        old_right = edge_list[right]
        adjacency.remove(old_left)
        adjacency.remove(old_right)
        adjacency.add(cross_left)
        adjacency.add(cross_right)
        edge_list[left] = cross_left
        edge_list[right] = cross_right
        true_intersection -= int(old_left in true_edges) + int(old_right in true_edges)
        true_intersection += int(cross_left in true_edges) + int(cross_right in true_edges)
        accepted += 1

        hamming_fraction = (edge_count - true_intersection) / edge_count
        if accepted >= min_accepted and hamming_fraction >= target_fraction:
            break

    hamming_fraction = (edge_count - true_intersection) / edge_count
    if accepted < min_accepted or hamming_fraction < target_fraction:
        raise RuntimeError(
            "P-R graph construction failed its frozen swap or edge-difference gate; "
            f"accepted={accepted}, proposals={proposal}, edge_hamming_fraction={hamming_fraction:.6f}"
        )

    randomized = np.zeros_like(original)
    named_edges = []
    for gene_idx, pathway_idx in edge_list:
        randomized[pathway_idx, gene_idx] = 1
        named_edges.append((genes[gene_idx], pathways[pathway_idx]))
    if int(randomized.sum()) != edge_count:
        raise RuntimeError("P-R edge count changed during randomization")
    if not np.array_equal(randomized.sum(axis=0), original.sum(axis=0)):
        raise RuntimeError("P-R did not preserve per-gene degree")
    if not np.array_equal(randomized.sum(axis=1), original.sum(axis=1)):
        raise RuntimeError("P-R did not preserve pathway size")
    if not np.array_equal(randomized.sum(axis=0) > 0, original.sum(axis=0) > 0):
        raise RuntimeError("P-R did not preserve gene coverage")
    if not np.array_equal(randomized.sum(axis=1) > 0, original.sum(axis=1) > 0):
        raise RuntimeError("P-R did not preserve non-empty pathways")
    calculated_fraction = len(true_edges.symmetric_difference(adjacency)) / (2 * edge_count)
    if calculated_fraction != hamming_fraction:
        raise RuntimeError("P-R incremental and final edge-difference calculations disagree")

    return PathwayControlMap(
        gene_ids=genes,
        pathway_names=pathways,
        incidence=randomized,
        edges=tuple(named_edges),
        edge_sha256=_edge_digest(named_edges),
        method="degree_preserving_bipartite_double_edge_swap",
        metadata={
            "seed": int(seed),
            "proposal_count": proposal,
            "accepted_swaps": accepted,
            "edge_count": edge_count,
            "edge_hamming_fraction": hamming_fraction,
            "minimum_accepted_swaps": min_accepted,
            "maximum_proposals": max_proposals,
        },
    )


def _proposal_index(seed: int, proposal: int, lane: str, edge_count: int) -> int:
    payload = f"{PR_PREFIX}|{seed}|{proposal}|{lane}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big") % edge_count
