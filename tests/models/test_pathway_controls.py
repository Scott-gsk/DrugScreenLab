import numpy as np
import pytest

from drug_screen.transfer.pathway_controls import (
    build_ordinary_sparse_control,
    randomize_bipartite_pathways,
)


def test_ordinary_control_is_deterministic_balanced_and_uses_each_gene_once_or_twice():
    genes = tuple(f"GENE_{index:02d}" for index in range(13))
    first = build_ordinary_sparse_control(genes, n_slots=4, n_edges=19)
    second = build_ordinary_sparse_control(genes, n_slots=4, n_edges=19)

    assert first.method == "minimum_cost_balanced_anonymous_sparse_partition"
    assert first.edge_sha256 == second.edge_sha256
    np.testing.assert_array_equal(first.incidence, second.incidence)
    assert first.incidence.shape == (4, 13)
    assert int(first.incidence.sum()) == 19
    assert tuple(map(int, first.incidence.sum(axis=1))) == (5, 5, 5, 4)
    assert np.all(first.incidence.sum(axis=0) >= 1)
    assert np.all(first.incidence.sum(axis=0) <= 2)
    assert int(first.metadata["second_edge_count"]) == 6
    assert len(first.edges) == len(set(first.edges))


def test_randomized_control_preserves_degrees_and_reaches_frozen_difference_gate():
    genes = ("G1", "G2", "G3", "G4")
    pathways = ("P1", "P2", "P3", "P4")
    incidence = np.asarray(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    result = randomize_bipartite_pathways(
        genes,
        pathways,
        incidence,
        seed=20260922,
        min_accepted_swaps_per_edge=1,
        min_edge_hamming_fraction=0.25,
        max_proposals_per_edge=2_000,
    )
    repeated = randomize_bipartite_pathways(
        genes,
        pathways,
        incidence,
        seed=20260922,
        min_accepted_swaps_per_edge=1,
        min_edge_hamming_fraction=0.25,
        max_proposals_per_edge=2_000,
    )

    assert result.edge_sha256 == repeated.edge_sha256
    np.testing.assert_array_equal(result.incidence, repeated.incidence)
    np.testing.assert_array_equal(result.incidence.sum(axis=0), incidence.sum(axis=0))
    np.testing.assert_array_equal(result.incidence.sum(axis=1), incidence.sum(axis=1))
    assert int(result.incidence.sum()) == int(incidence.sum()) == 8
    assert result.metadata["accepted_swaps"] >= 8
    assert result.metadata["edge_hamming_fraction"] >= 0.25
    assert len(result.edges) == len(set(result.edges))


def test_randomized_control_rejects_uncovered_gene_or_empty_pathway():
    with pytest.raises(ValueError, match="covered genes"):
        randomize_bipartite_pathways(
            ("G1", "G2"),
            ("P1",),
            np.asarray([[1, 0]], dtype=np.uint8),
            seed=1,
            min_accepted_swaps_per_edge=0,
            min_edge_hamming_fraction=0.01,
        )
