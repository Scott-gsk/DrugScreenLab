"""Frozen observation schema and eligibility gates for EXP-013."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

REQUIRED_FIELDS = (
    "patient_id",
    "model_id",
    "study_id",
    "domain",
    "cancer_type",
    "drug_id",
    "response",
    "response_name",
    "higher_is_sensitive",
    "split_id",
)

SUGGESTED_MIN_PATIENTS = 30
EXPLORE_MIN_PATIENTS = 20
MIN_SHARED_DRUGS = 10
MIN_DRUGS_FOR_NDCG = 10
MIN_DRUGS_PER_PATIENT = 5


@dataclass(frozen=True)
class TransferObservation:
    patient_id: str
    model_id: str
    study_id: str
    domain: str
    cancer_type: str
    drug_id: str
    response: float
    response_name: str
    higher_is_sensitive: bool
    split_id: str
    smiles: str | None = None
    assay: str | None = None


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def assert_higher_is_sensitive(observations: Iterable[TransferObservation]) -> None:
    values = list(observations)
    if not values:
        raise ValueError("observations cannot be empty")
    names = {row.response_name for row in values}
    if len(names) != 1:
        raise ValueError(f"mixed response names are not a single label contract: {sorted(names)}")
    flags = {row.higher_is_sensitive for row in values}
    if flags != {True}:
        raise ValueError("EXP-013 requires a frozen higher-is-sensitive label direction")


def eligibility_report(
    observations: Sequence[TransferObservation],
    *,
    source_drug_ids: Sequence[str] | None = None,
) -> Mapping[str, object]:
    patients = {row.patient_id for row in observations}
    drugs = {row.drug_id for row in observations}
    shared = set(source_drug_ids or []).intersection(drugs)
    n_patients = len(patients)
    if n_patients < EXPLORE_MIN_PATIENTS:
        scale = "too_small_for_architecture_search"
    elif n_patients < SUGGESTED_MIN_PATIENTS:
        scale = "exploratory_baselines_and_learning_curve"
    else:
        scale = "planned_nested_cv_eligible"
    return {
        "n_patients": n_patients,
        "n_drugs": len(drugs),
        "n_observations": len(observations),
        "n_shared_source_drugs": len(shared),
        "within_patient_ranking_eligible": all(
            sum(1 for row in observations if row.patient_id == patient) >= MIN_DRUGS_PER_PATIENT
            for patient in patients
        )
        if patients
        else False,
        "ndcg_primary_allowed": len(drugs) >= MIN_DRUGS_FOR_NDCG,
        "scale": scale,
        "min_shared_drugs_met": len(shared) >= MIN_SHARED_DRUGS if source_drug_ids is not None else None,
    }
