"""Sample-level admission whitelist and patient-unit permutation (work-package B).

Two contract pieces live here:

1. ``AdmissionWhitelist`` — the sample-level admission gate. Kryzei admission is
   **sample-level** (60 S1_PDO_patient_verified samples across 51 patients), so
   a sample whose patient is admitted must *still* be rejected when the sample
   itself is not in the whitelist (e.g. ``Pt102_t2`` under admitted ``Pt102``).
2. ``patient_unit_permutation`` — a row permutation that moves rows in whole
   *patient* blocks. It is used by the patient-level permutation tests so that
   every drug record of a patient references the same permuted expression
   vector.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["AdmissionWhitelist", "SampleVerdict", "patient_unit_permutation"]


@dataclass(frozen=True)
class SampleVerdict:
    sample_id: str
    patient_id: str
    admitted: bool
    reason: Optional[str] = None


class AdmissionWhitelist:
    """Sample-level admission gate backed by patient/sample whitelists."""

    def __init__(
        self,
        patient_whitelist: Iterable[object],
        sample_whitelist: Iterable[object],
        *,
        patient_of_sample: Optional[Mapping[object, object]] = None,
    ) -> None:
        self.patients = frozenset(str(p) for p in patient_whitelist)
        self.samples = frozenset(str(s) for s in sample_whitelist)
        self._patient_of_sample: dict[str, str] = {
            str(s): str(p) for s, p in (patient_of_sample or {}).items()
        }

    def adjudicate(self, sample_id: object, patient_id: Optional[object] = None) -> SampleVerdict:
        sid = str(sample_id)
        pid = str(patient_id) if patient_id is not None else self._patient_of_sample.get(sid, "?")
        if sid in self.samples:
            return SampleVerdict(sample_id=sid, patient_id=pid, admitted=True)
        if pid == "?":
            return SampleVerdict(sample_id=sid, patient_id=pid, admitted=False,
                                 reason="unknown_patient_mapping")
        if pid in self.patients:
            return SampleVerdict(sample_id=sid, patient_id=pid, admitted=False,
                                 reason="non_admitted_sample_of_admitted_patient")
        return SampleVerdict(sample_id=sid, patient_id=pid, admitted=False,
                             reason="non_admitted_patient")

    def reject_non_admitted(
        self,
        sample_ids: Sequence[object],
        *,
        patient_of_sample: Optional[Mapping[object, object]] = None,
    ) -> tuple[list[str], list[SampleVerdict]]:
        """Return ``(admitted_sample_ids, rejected_verdicts)`` in input order."""
        mapping = patient_of_sample or self._patient_of_sample
        admitted: list[str] = []
        rejected: list[SampleVerdict] = []
        for sid in sample_ids:
            verdict = self.adjudicate(sid, patient_id=mapping.get(sid))
            if verdict.admitted:
                admitted.append(str(sid))
            else:
                rejected.append(verdict)
        return admitted, rejected

def patient_unit_permutation(
    patient_ids: Sequence[object],
    rng: Optional[object] = None,
) -> np.ndarray:
    """Return a row permutation moving rows in whole *patient* blocks.

    ``patient_ids[k]`` is the patient of the k-th row. The returned integer
    array ``perm`` satisfies: ``np.asarray(patient_ids)[perm]`` keeps every
    patient's rows contiguous, and the multiset of row counts per patient is
    preserved. When ``rng`` has a ``permutation`` method (``np.random.Generator``
    or ``np.random.RandomState``) the patient block order is shuffled.
    """
    ids = np.asarray([str(p) for p in patient_ids])
    groups: dict[str, list[int]] = {}
    for position, patient in enumerate(ids):
        groups.setdefault(patient, []).append(position)
    order = sorted(groups)
    if rng is not None and hasattr(rng, "permutation"):
        shuffled = rng.permutation(len(order))
        order = [order[int(i)] for i in shuffled]
    return np.asarray(
        [index for patient in order for index in groups[patient]], dtype=int
    )


