"""Identity-first split and matched-control contracts for EXP-002."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
import json


SPLITS = ("train", "validation", "test")
DEFAULT_SEED = "EXP-002-identity-first-v2"


class _UnionFind:
    """Legacy feasibility helper retained to regression-test the old blocker."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while value != root:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def strict_components(edges: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Return legacy component labels for explicitly non-separable identities."""
    graph = _UnionFind()
    for left, right in edges:
        graph.union(left, right)
    return {vertex: graph.find(vertex) for vertex in graph.parent}


def deterministic_split(entity_id: str, seed: str = DEFAULT_SEED) -> str:
    """Assign a cold entity directly; controls and plates never define its split."""
    bucket = int(sha256(f"{seed}|entity|{entity_id}".encode("utf-8")).hexdigest(), 16) % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def is_non_degenerate(component_ids: Iterable[str], seed: str = DEFAULT_SEED) -> bool:
    """Return whether independently hashed IDs populate every required split."""
    assignments = {deterministic_split(component, seed) for component in set(component_ids)}
    return set(SPLITS).issubset(assignments)


def has_all_splits(assignments: Iterable[str]) -> bool:
    """Return whether an already assigned manifest contains every split."""
    return set(SPLITS).issubset(set(assignments))


def canonical_digest(value: object) -> str:
    """Hash a JSON-compatible contract value with stable serialization."""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def deterministic_vehicle_partition(
    matching_key: Sequence[object],
    control_ids: Iterable[str],
    active_splits: Iterable[str],
    seed: str = DEFAULT_SEED,
) -> dict[str, str]:
    """Assign each raw vehicle instance to exactly one active split.

    Every active split receives at least one vehicle. Remaining replicates are
    distributed round-robin after stable hash ranking. A plate is deliberately
    not treated as a split identity.
    """
    controls = set(control_ids)
    if not controls:
        raise ValueError("matching key has no vehicle controls")
    active = set(active_splits)
    unknown = active.difference(SPLITS)
    if unknown:
        raise ValueError(f"unknown splits: {sorted(unknown)}")
    splits = [split_name for split_name in SPLITS if split_name in active]
    if not splits:
        raise ValueError("matching key has no active splits")
    if len(controls) < len(splits):
        raise ValueError(
            f"matching key has {len(controls)} controls for {len(splits)} active splits"
        )
    key_digest = canonical_digest(list(matching_key))
    ranked = sorted(
        controls,
        key=lambda control_id: sha256(
            f"{seed}|vehicle|{key_digest}|{control_id}".encode("utf-8")
        ).hexdigest(),
    )
    return {control_id: splits[index % len(splits)] for index, control_id in enumerate(ranked)}


def assert_exclusive_assignments(
    entity_splits: Mapping[str, str],
    family_members: Mapping[str, Iterable[str]],
    control_splits: Mapping[str, str],
) -> None:
    """Validate cold-entity, treatment-family, and raw-control isolation."""
    for identity, split_name in entity_splits.items():
        if not identity or split_name not in SPLITS:
            raise ValueError("invalid cold-entity assignment")
    for family_id, members in family_members.items():
        splits = {entity_splits[member] for member in members}
        if len(splits) != 1:
            raise ValueError(f"replicate family crosses splits: {family_id}")
    if any(
        not control_id or split_name not in SPLITS
        for control_id, split_name in control_splits.items()
    ):
        raise ValueError("invalid raw-control assignment")
