"""Patient-grouped fold assignment with leakage checks."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .contract import TransferObservation

_VALID_SPLITS = frozenset({"train", "validation", "test"})


@dataclass(frozen=True)
class PatientFoldAssignment:
    n_folds: int
    seed: int
    patient_to_test_fold: Mapping[str, int]


def assign_patient_folds(
    patient_ids: Sequence[str],
    *,
    n_folds: int,
    seed: int,
) -> PatientFoldAssignment:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    unique = sorted({_require_id(value, "patient_id") for value in patient_ids})
    if len(unique) < n_folds:
        raise ValueError("need at least one patient per fold")
    generator = random.Random(seed)
    shuffled = unique[:]
    generator.shuffle(shuffled)
    mapping = {patient: index % n_folds for index, patient in enumerate(shuffled)}
    return PatientFoldAssignment(n_folds=n_folds, seed=seed, patient_to_test_fold=mapping)


def split_for_fold(assignment: PatientFoldAssignment, patient_id: str, fold: int) -> str:
    test_fold = assignment.patient_to_test_fold[_require_id(patient_id, "patient_id")]
    if test_fold == fold:
        return "test"
    if test_fold == (fold + 1) % assignment.n_folds:
        return "validation"
    return "train"


def assert_patient_isolation(
    observations: Iterable[TransferObservation],
    assignments: Mapping[str, str],
) -> None:
    seen: dict[str, str] = {}
    for row in observations:
        split = assignments.get(row.patient_id)
        if split not in _VALID_SPLITS:
            raise ValueError(f"unsupported split for {row.patient_id}: {split}")
        previous = seen.setdefault(row.patient_id, split)
        if previous != split:
            raise ValueError(f"patient leakage for {row.patient_id}: {previous} vs {split}")
        if row.split_id and row.split_id != split:
            raise ValueError(
                f"split_id mismatch for {row.patient_id}: {row.split_id} vs {split}"
            )


def _require_id(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()
