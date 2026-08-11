"""Phase-0 data contracts for auditable perturbation benchmarks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json


ALLOWED_TASK_ROLES = frozenset({"TRAIN", "CALIBRATION", "EXTERNAL_TEST", "KNOWLEDGE_PRIOR"})


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class DatasetReadiness:
    """Minimal per-dataset P0 schema; unavailable resources remain explicit."""

    dataset_id: str
    accession_or_version: str
    intended_role: str
    local_availability: str
    source_availability: str
    checksum_evidence: str
    metadata_status: str
    license_status: str
    preprocessing_status: str
    blockers: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "DatasetReadiness":
        fields = (
            "dataset_id", "accession_or_version", "intended_role", "local_availability",
            "source_availability", "checksum_evidence", "metadata_status", "license_status",
            "preprocessing_status", "blockers",
        )
        missing = [field for field in fields if field not in value]
        if missing:
            raise ValueError(f"missing readiness fields: {', '.join(missing)}")
        role = _required_text(value["intended_role"], "intended_role")
        if role not in ALLOWED_TASK_ROLES:
            raise ValueError(f"unsupported intended_role: {role}")
        blockers = value["blockers"]
        if not isinstance(blockers, list) or not all(isinstance(item, str) for item in blockers):
            raise ValueError("blockers must be a list of strings")
        return cls(
            **{field: _required_text(value[field], field) for field in fields[:-1]},
            blockers=tuple(blockers),
        )


@dataclass(frozen=True)
class PerturbationRecord:
    """Identity fields needed to calculate one matched-control perturbation."""

    sample_id: str
    compound_id: str | None
    gene_id: str | None
    context_id: str
    donor_id: str | None
    control_id: str | None
    replicate_family_id: str
    plate_id: str
    timepoint: str
    is_control: bool

    def __post_init__(self) -> None:
        _required_text(self.sample_id, "sample_id")
        _required_text(self.context_id, "context_id")
        _required_text(self.replicate_family_id, "replicate_family_id")
        _required_text(self.plate_id, "plate_id")
        _required_text(self.timepoint, "timepoint")
        if not self.is_control:
            if not (self.compound_id or self.gene_id):
                raise ValueError("treatment requires compound_id or gene_id")
            _required_text(self.control_id, "control_id")


def assert_matched_controls(records: Iterable[PerturbationRecord]) -> None:
    """Require each treatment's named control in the same context, plate, and timepoint."""
    values = list(records)
    by_id = {record.sample_id: record for record in values}
    if len(by_id) != len(values):
        raise ValueError("sample_id values must be unique")
    for record in values:
        if record.is_control:
            continue
        control = by_id.get(record.control_id or "")
        if control is None or not control.is_control:
            raise ValueError(f"treatment {record.sample_id} has no matched control")
        for field in ("context_id", "plate_id", "timepoint"):
            if getattr(record, field) != getattr(control, field):
                raise ValueError(f"treatment {record.sample_id} mismatches control on {field}")


def assert_no_split_leakage(
    assignments: Mapping[str, str], records: Iterable[PerturbationRecord], keys: Sequence[str]
) -> None:
    """Forbid identity and matched-control leakage across assigned splits."""
    values = list(records)
    by_id = {record.sample_id: record for record in values}
    if len(by_id) != len(values):
        raise ValueError("sample_id values must be unique")
    seen: dict[tuple[str, str], str] = {}
    for record in values:
        split = _required_text(assignments.get(record.sample_id), "split")
        if not record.is_control:
            control = by_id.get(record.control_id or "")
            if control is None or not control.is_control:
                raise ValueError(f"treatment {record.sample_id} has no matched control")
            control_split = _required_text(assignments.get(control.sample_id), "split")
            if control_split != split:
                raise ValueError(
                    f"split leakage for matched control {control.sample_id}: "
                    f"{control_split} vs {split}"
                )
        for key in keys:
            if not hasattr(record, key):
                raise ValueError(f"unknown leakage key: {key}")
            value = getattr(record, key)
            if value is None:
                continue
            marker = (key, str(value))
            previous = seen.setdefault(marker, split)
            if previous != split:
                raise ValueError(f"split leakage for {key}={value}: {previous} vs {split}")


def assert_task_role(role: str, use: str) -> None:
    """Keep frozen EXTERNAL_TEST resources out of fitting and tuning."""
    if role not in ALLOWED_TASK_ROLES:
        raise ValueError(f"unsupported task role: {role}")
    if role == "EXTERNAL_TEST" and use in {"train", "tune", "calibrate"}:
        raise ValueError("EXTERNAL_TEST datasets cannot be used for train, tune, or calibrate")


def deterministic_metadata_digest(metadata: Mapping[str, object]) -> str:
    """Return a stable fingerprint independent of input mapping insertion order."""
    encoded = json.dumps(metadata, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def assert_level3_landmark_core(
    matrix_shape: tuple[int, int],
    row_gene_ids: Sequence[str],
    landmark_flags: Sequence[int],
    column_instance_ids: Sequence[str],
    metadata_instance_ids: Sequence[str],
) -> None:
    """Validate that Level-3 exposes a direct-measurement 978-gene core.

    Level-3 also contains inferred non-landmark genes.  This assertion deliberately
    validates only the `pr_is_lm` subset and does not promote the remaining genes
    to ground truth.
    """
    if len(matrix_shape) != 2:
        raise ValueError("matrix_shape must have two dimensions")
    if matrix_shape[1] != len(row_gene_ids) or len(row_gene_ids) != len(landmark_flags):
        raise ValueError("Level-3 gene dimension does not match gene metadata")
    if sum(landmark_flags) != 978:
        raise ValueError("Level-3 landmark core must contain exactly 978 genes")
    if len(set(row_gene_ids)) != len(row_gene_ids):
        raise ValueError("Level-3 row gene IDs must be unique")
    if matrix_shape[0] != len(column_instance_ids):
        raise ValueError("Level-3 instance dimension does not match GCTX columns")
    if set(column_instance_ids) != set(metadata_instance_ids):
        raise ValueError("GCTX instance IDs do not match instance metadata")


def count_same_plate_vehicle_candidates(
    chemical_keys: Iterable[tuple[str, str, object, str]],
    vehicle_keys: Iterable[tuple[str, str, object, str]],
) -> tuple[int, int]:
    """Return chemical-instance count and same-plate vehicle candidate count."""
    vehicle_key_set = set(vehicle_keys)
    total = 0
    matched = 0
    for key in chemical_keys:
        total += 1
        if key in vehicle_key_set:
            matched += 1
    return total, matched
