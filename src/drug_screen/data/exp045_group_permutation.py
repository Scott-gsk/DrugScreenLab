"""Auditable whole-biological-group label shuffling for EXP-045 D-SC/D-SO.

Only training-side batches may be passed to this module. Groups are exchangeable
only when their complete observation strata and per-observation missing-label
masks match exactly. Groups without a compatible peer are excluded from the
dependency mask so the true-label and shuffled-label arms can use the same
effective denominator.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


class GroupPermutationError(ValueError):
    """Raised when a fold-safe, auditable group permutation cannot be built."""


@dataclass(frozen=True)
class GroupPermutationResult:
    targets: np.ndarray
    mask: np.ndarray
    receipt: Mapping[str, Any]
    receipt_sha256: str


@dataclass(frozen=True)
class GroupPermutationProjection:
    targets: np.ndarray
    mask: np.ndarray
    receipt: Mapping[str, Any]
    receipt_sha256: str


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _require_ids(values: Sequence[str], name: str, *, unique: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise GroupPermutationError(f"{name} must be a sequence of IDs")
    result = tuple(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise GroupPermutationError(f"{name} contains an empty or non-string ID")
    if unique and len(set(result)) != len(result):
        raise GroupPermutationError(f"{name} contains duplicate IDs")
    return result


def _mask_sha256(mask: np.ndarray) -> str:
    packed = np.packbits(np.asarray(mask, dtype=np.bool_).reshape(-1), bitorder="little")
    header = _canonical_bytes({"shape": list(mask.shape), "bitorder": "little"})
    return _sha256(header + b"\0" + packed.tobytes())


def _target_sha256(targets: np.ndarray, mask: np.ndarray) -> str:
    values = np.asarray(targets, dtype="<f8", order="C")
    valid = np.asarray(mask, dtype=np.bool_)
    if values.shape != valid.shape:
        raise GroupPermutationError("target values and mask have different shapes")
    if not np.isfinite(values[valid]).all():
        raise GroupPermutationError("valid dependency targets must be finite")
    header = _canonical_bytes({"shape": list(values.shape), "dtype": "<f8"})
    return _sha256(header + b"\0" + np.packbits(valid.reshape(-1), bitorder="little").tobytes()
                   + b"\0" + values[valid].tobytes(order="C"))


def _normalize_dataset_assets(assets: Sequence[Mapping[str, str]]) -> tuple[dict[str, str], ...]:
    if not isinstance(assets, Sequence) or isinstance(assets, (str, bytes)) or not assets:
        raise GroupPermutationError("registered source asset identities are required")
    normalized: list[dict[str, str]] = []
    for asset in assets:
        if not isinstance(asset, Mapping):
            raise GroupPermutationError("registered source asset entry must be a mapping")
        dataset_id = asset.get("dataset_id")
        version = asset.get("version")
        checksum = asset.get("sha256")
        if (not isinstance(dataset_id, str) or not dataset_id.strip()
                or not isinstance(version, str) or not version.strip()
                or not _valid_sha256(checksum)):
            raise GroupPermutationError("registered source assets require dataset_id, version, and SHA-256")
        normalized.append({"dataset_id": dataset_id, "version": version, "sha256": checksum})
    normalized.sort(key=lambda item: (item["dataset_id"].encode("utf-8"), item["version"].encode("utf-8")))
    identities = [(item["dataset_id"], item["version"]) for item in normalized]
    if len(set(identities)) != len(identities):
        raise GroupPermutationError("registered source asset identities must be unique")
    return tuple(normalized)


def _row_order(indices: Sequence[int], strata: tuple[str, ...], observation_ids: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(sorted(
        indices,
        key=lambda index: (strata[index].encode("utf-8"), observation_ids[index].encode("utf-8")),
    ))


def _mask_signature(mask: np.ndarray, rows: tuple[int, ...], strata: tuple[str, ...]) -> tuple[tuple[str, ...], str]:
    ordered_strata = tuple(strata[index] for index in rows)
    ordered_mask = np.ascontiguousarray(mask[np.asarray(rows, dtype=np.int64)], dtype=np.bool_)
    return ordered_strata, _mask_sha256(ordered_mask)


def _seeded_derangement(group_ids: tuple[str, ...], *, seed: int, signature_sha256: str) -> dict[str, str]:
    """Build a deterministic Sattolo cycle with no fixed points."""
    ordered = list(group_ids)
    seed_material = _canonical_bytes({"seed": int(seed), "signature_sha256": signature_sha256})
    local_seed = int.from_bytes(hashlib.sha256(seed_material).digest(), "big")
    rng = random.Random(local_seed)
    donors = list(ordered)
    for index in range(len(donors) - 1, 0, -1):
        # Sattolo's algorithm: draw uniformly from [0, index), never index.
        swap_index = rng.randrange(index)
        donors[index], donors[swap_index] = donors[swap_index], donors[index]
    mapping = dict(zip(ordered, donors, strict=True))
    if any(recipient == donor for recipient, donor in mapping.items()):
        raise GroupPermutationError("internal derangement construction produced a fixed point")
    return mapping


def permute_dependency_groups(
    targets: np.ndarray,
    *,
    observation_ids: Sequence[str],
    biological_group_ids: Sequence[str],
    row_strata: Sequence[str],
    dependency_axis: Sequence[str],
    fold_id: str,
    partition_id: str,
    stage: str,
    domain: str,
    batch_id: str,
    dataset_assets: Sequence[Mapping[str, str]],
    strata_definition_version: str,
    input_contract_sha256: str,
    split_sha256: str,
    seed: int,
    training_mean: float,
    training_std: float,
    excluded_groups: Mapping[str, str] | None = None,
) -> GroupPermutationResult:
    """Shuffle complete groups within exact stratum/shape/mask-compatible classes.

    The caller must pass only rows belonging to the current training partition.
    For every output ``eligible`` group, all observations move together to a
    different group with the same ordered row strata and exact missing-label
    pattern. Unexchangeable groups remain in the batch with an all-false loss
    mask; the same effective mask must be applied to the corresponding true-label
    arms.
    """
    if isinstance(targets, (list, tuple)):
        targets = np.asarray(targets)
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or values.size == 0:
        raise GroupPermutationError("dependency targets must be a non-empty 2D array")
    if np.isinf(values).any():
        raise GroupPermutationError("infinite dependency labels are invalid; only NaN marks missingness")
    sample_ids = _require_ids(observation_ids, "observation IDs", unique=True)
    group_ids = _require_ids(biological_group_ids, "biological group IDs")
    strata = _require_ids(row_strata, "row strata")
    axis = _require_ids(dependency_axis, "dependency axis", unique=True)
    n_rows, n_genes = values.shape
    if not (len(sample_ids) == len(group_ids) == len(strata) == n_rows):
        raise GroupPermutationError("observation, biological-group, and stratum IDs must align to target rows")
    if len(axis) != n_genes:
        raise GroupPermutationError("dependency axis does not match target width")
    for name, value in (("fold_id", fold_id), ("partition_id", partition_id),
                        ("stage", stage), ("domain", domain), ("batch_id", batch_id)):
        if not isinstance(value, str) or not value.strip():
            raise GroupPermutationError(f"{name} is required")
    if not isinstance(strata_definition_version, str) or not strata_definition_version.strip():
        raise GroupPermutationError("strata definition version is required")
    source_assets = _normalize_dataset_assets(dataset_assets)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise GroupPermutationError("permutation seed must be an integer")
    if not _valid_sha256(input_contract_sha256) or not _valid_sha256(split_sha256):
        raise GroupPermutationError("input contract and split SHA-256 digests are required")
    if (isinstance(training_mean, bool) or not isinstance(training_mean, (int, float, np.number))
            or not np.isfinite(float(training_mean))
            or isinstance(training_std, bool) or not isinstance(training_std, (int, float, np.number))
            or not np.isfinite(float(training_std)) or float(training_std) <= 0):
        raise GroupPermutationError("fold-fitted dependency scale must be finite with positive standard deviation")

    observed_mask = np.isfinite(values)
    effective_mask = observed_mask.copy()
    shuffled = values.copy()
    by_group: dict[str, list[int]] = {}
    for row, group_id in enumerate(group_ids):
        by_group.setdefault(group_id, []).append(row)

    declared_exclusions = dict(excluded_groups or {})
    excluded = dict(declared_exclusions)
    if any(not isinstance(group, str) or not group or not isinstance(reason, str) or not reason.strip()
           for group, reason in excluded.items()):
        raise GroupPermutationError("excluded group IDs and reasons must be non-empty strings")
    unknown_exclusions = set(excluded) - set(by_group)
    if unknown_exclusions:
        raise GroupPermutationError("exclusion manifest names groups absent from this training batch")

    ordered_rows = {group: _row_order(rows, strata, sample_ids)
                    for group, rows in by_group.items()}
    group_signatures: dict[str, dict[str, Any]] = {}
    signature_buckets: dict[str, list[str]] = {}
    group_signature_hash: dict[str, str] = {}
    for group in sorted(by_group, key=lambda value: value.encode("utf-8")):
        if group in excluded:
            continue
        rows = ordered_rows[group]
        ordered_strata, mask_digest = _mask_signature(observed_mask, rows, strata)
        signature = {
            "n_observations": len(rows),
            "ordered_row_strata": list(ordered_strata),
            "ordered_missing_mask_sha256": mask_digest,
        }
        signature_digest = _sha256(_canonical_bytes(signature))
        group_signatures[group] = signature
        group_signature_hash[group] = signature_digest
        signature_buckets.setdefault(signature_digest, []).append(group)

    mappings: dict[str, str] = {}
    excluded_reasons = dict(excluded)
    compatible_classes: list[dict[str, Any]] = []
    for signature_digest in sorted(signature_buckets):
        members = tuple(sorted(signature_buckets[signature_digest], key=lambda value: value.encode("utf-8")))
        if len(members) < 2:
            excluded_reasons[members[0]] = "no_compatible_peer"
            continue
        class_mapping = _seeded_derangement(members, seed=int(seed), signature_sha256=signature_digest)
        mappings.update(class_mapping)
        compatible_classes.append({
            "signature_sha256": signature_digest,
            "n_groups": len(members),
            "group_ids": list(members),
            "group_mapping": {member: class_mapping[member] for member in members},
        })

    for group, donor in mappings.items():
        destination_rows = ordered_rows[group]
        source_rows = ordered_rows[donor]
        # Exact signatures guarantee same row count, stratum order, and mask bits.
        for destination_row, source_row in zip(destination_rows, source_rows, strict=True):
            shuffled[destination_row] = values[source_row]
            effective_mask[destination_row] = observed_mask[source_row]

    row_mapping_by_observation: dict[str, dict[str, Any]] = {}
    for recipient, rows in ordered_rows.items():
        donor = mappings.get(recipient)
        donor_rows = ordered_rows[donor] if donor is not None else ()
        for position, target_row in enumerate(rows):
            source_row = donor_rows[position] if donor is not None else None
            row_mapping_by_observation[sample_ids[target_row]] = {
                "target_observation_id": sample_ids[target_row],
                "source_observation_id": sample_ids[source_row] if source_row is not None else None,
                "target_group_id": recipient,
                "source_group_id": donor,
                "ordered_row_position_in_group": position,
                "assignment": "permuted" if donor is not None else "excluded",
            }

    for group in excluded_reasons:
        effective_mask[np.asarray(ordered_rows[group], dtype=np.int64)] = False

    # A valid D-SC/D-SO shuffled batch must have at least one exchange class and
    # retain at least one observed target after applying the shared exclusion mask.
    if not mappings or not effective_mask.any():
        raise GroupPermutationError("training batch has no non-trivial compatible group permutation")

    row_manifest = [
        {
            "observation_id": sample_ids[index],
            "biological_group_id": group_ids[index],
            "stratum": strata[index],
            "ordered_row_position_in_group": ordered_rows[group_ids[index]].index(index),
        }
        for index in range(n_rows)
    ]
    payload: dict[str, Any] = {
        "schema": "exp045_group_permutation_v1",
        "algorithm": "Sattolo_cycle_python_random_v1",
        "fold_id": fold_id,
        "partition_id": partition_id,
        "stage": stage,
        "domain": domain,
        "arm_assignment": "within_domain_shuffled",
        "batch_id": batch_id,
        "seed": int(seed),
        "dataset_assets": list(source_assets),
        "dataset_assets_manifest_sha256": _sha256(_canonical_bytes(list(source_assets))),
        "strata_definition_version": strata_definition_version,
        "input_contract_sha256": input_contract_sha256,
        "split_sha256": split_sha256,
        "dependency_training_mean": float(training_mean),
        "dependency_training_population_std": float(training_std),
        "dependency_axis_sha256": _sha256(_canonical_bytes(list(axis))),
        "ordered_observation_ids": list(sample_ids),
        "ordered_biological_group_ids": list(group_ids),
        "ordered_row_strata": list(strata),
        "training_partition_observation_manifest_sha256": _sha256(_canonical_bytes(
            sorted(sample_ids, key=lambda value: value.encode("utf-8")),
        )),
        "training_partition_group_ids": sorted(by_group, key=lambda value: value.encode("utf-8")),
        "training_partition_group_manifest_sha256": _sha256(_canonical_bytes(
            sorted(by_group, key=lambda value: value.encode("utf-8")),
        )),
        "row_mapping": [row_mapping_by_observation[sample] for sample in sample_ids],
        "rows": row_manifest,
        "group_signatures": {group: group_signatures[group] for group in sorted(group_signatures)},
        "compatible_classes": compatible_classes,
        "group_mapping": {group: mappings[group] for group in sorted(mappings)},
        "declared_exclusions": {
            group: declared_exclusions[group]
            for group in sorted(declared_exclusions, key=lambda value: value.encode("utf-8"))
        },
        "excluded_groups": {group: excluded_reasons[group]
                            for group in sorted(excluded_reasons, key=lambda value: value.encode("utf-8"))},
        "n_groups_input": len(by_group),
        "n_groups_exchangeable": len(mappings),
        "n_groups_excluded": len(excluded_reasons),
        "n_observations_input": n_rows,
        "n_observations_exchangeable": sum(len(ordered_rows[group]) for group in mappings),
        "n_valid_labels_input": int(observed_mask.sum()),
        "n_valid_labels_effective": int(effective_mask.sum()),
        "n_valid_labels_output": int((np.isfinite(shuffled) & effective_mask).sum()),
        "input_target_sha256": _target_sha256(values, observed_mask),
        "input_mask_sha256": _mask_sha256(observed_mask),
        "effective_target_sha256": _target_sha256(shuffled, effective_mask),
        "effective_mask_sha256": _mask_sha256(effective_mask),
        "output_target_sha256": _target_sha256(shuffled, effective_mask),
        "output_mask_sha256": _mask_sha256(effective_mask),
    }
    receipt_sha256 = _sha256(_canonical_bytes(payload))
    output_targets = np.array(shuffled, copy=True)
    output_mask = np.array(effective_mask, dtype=np.bool_, copy=True)
    output_targets.setflags(write=False)
    output_mask.setflags(write=False)
    return GroupPermutationResult(
        targets=output_targets,
        mask=output_mask,
        receipt=payload,
        receipt_sha256=receipt_sha256,
    )


def project_dependency_group_permutation(
    result: GroupPermutationResult,
    *,
    batch_id: str,
    observation_ids: Sequence[str],
) -> GroupPermutationProjection:
    """Project a frozen partition permutation onto one scheduled batch by row ID."""
    if not isinstance(result, GroupPermutationResult):
        raise GroupPermutationError("partition permutation result is required")
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise GroupPermutationError("projection batch ID is required")
    selected_ids = _require_ids(observation_ids, "projection observation IDs", unique=True)
    parent_ids = _require_ids(result.receipt.get("ordered_observation_ids", ()),
                              "partition observation IDs", unique=True)
    parent_index = {sample_id: index for index, sample_id in enumerate(parent_ids)}
    unknown = set(selected_ids) - set(parent_index)
    if unknown:
        raise GroupPermutationError("batch projection includes an observation outside the training partition")
    indices = np.asarray([parent_index[sample_id] for sample_id in selected_ids], dtype=np.int64)
    projected_targets = np.array(result.targets[indices], copy=True)
    projected_mask = np.array(result.mask[indices], dtype=np.bool_, copy=True)
    mapping = {item["target_observation_id"]: item
               for item in result.receipt.get("row_mapping", ())}
    if set(mapping) != set(parent_ids):
        raise GroupPermutationError("partition receipt row mapping is incomplete")
    projection_payload: dict[str, Any] = {
        "schema": "exp045_group_permutation_projection_v1",
        "batch_id": batch_id,
        "parent_receipt_sha256": result.receipt_sha256,
        "ordered_observation_ids": list(selected_ids),
        "row_mapping": [mapping[sample_id] for sample_id in selected_ids],
        "n_observations": len(selected_ids),
        "n_valid_labels": int(projected_mask.sum()),
        "projected_target_sha256": _target_sha256(projected_targets, projected_mask),
        "projected_mask_sha256": _mask_sha256(projected_mask),
    }
    digest = _sha256(_canonical_bytes(projection_payload))
    projected_targets.setflags(write=False)
    projected_mask.setflags(write=False)
    return GroupPermutationProjection(projected_targets, projected_mask, projection_payload, digest)


def verify_dependency_group_permutation_projection(
    *,
    partition_result: GroupPermutationResult,
    batch_id: str,
    observation_ids: Sequence[str],
    output_targets: np.ndarray,
    output_mask: np.ndarray,
    receipt: Mapping[str, Any],
    receipt_sha256: str,
) -> None:
    """Recompute one child projection and bind it to the whole-partition receipt."""
    projected = project_dependency_group_permutation(
        partition_result, batch_id=batch_id, observation_ids=observation_ids,
    )
    observed_targets = np.asarray(output_targets, dtype=np.float64)
    observed_mask = np.asarray(output_mask, dtype=np.bool_)
    if (dict(receipt) != dict(projected.receipt)
            or receipt_sha256 != projected.receipt_sha256
            or observed_targets.shape != projected.targets.shape
            or not np.array_equal(observed_targets, projected.targets, equal_nan=True)
            or not np.array_equal(observed_mask, projected.mask)):
        raise GroupPermutationError("batch projection differs from the recomputed full-partition mapping")


def verify_dependency_group_permutation(
    *,
    source_targets: np.ndarray,
    output_targets: np.ndarray,
    output_mask: np.ndarray,
    observation_ids: Sequence[str],
    biological_group_ids: Sequence[str],
    row_strata: Sequence[str],
    dependency_axis: Sequence[str],
    expected_fold_id: str,
    expected_partition_id: str,
    expected_stage: str,
    expected_domain: str,
    expected_batch_id: str,
    expected_dataset_assets: Sequence[Mapping[str, str]],
    expected_strata_definition_version: str,
    expected_input_contract_sha256: str,
    expected_split_sha256: str,
    expected_training_mean: float,
    expected_training_std: float,
    receipt: Mapping[str, Any],
    receipt_sha256: str,
) -> GroupPermutationResult:
    """Rebuild the shuffle and fail closed unless payload, labels, and mask match."""
    if not isinstance(receipt, Mapping) or not _valid_sha256(receipt_sha256):
        raise GroupPermutationError("permutation receipt or digest is missing")
    seed = receipt.get("seed")
    excluded = receipt.get("declared_exclusions")
    if not isinstance(seed, int) or isinstance(seed, bool) or not isinstance(excluded, Mapping):
        raise GroupPermutationError("permutation receipt seed or exclusion mapping is malformed")
    recomputed = permute_dependency_groups(
        source_targets,
        observation_ids=observation_ids,
        biological_group_ids=biological_group_ids,
        row_strata=row_strata,
        dependency_axis=dependency_axis,
        fold_id=expected_fold_id,
        partition_id=expected_partition_id,
        stage=expected_stage,
        domain=expected_domain,
        batch_id=expected_batch_id,
        dataset_assets=expected_dataset_assets,
        strata_definition_version=expected_strata_definition_version,
        input_contract_sha256=expected_input_contract_sha256,
        split_sha256=expected_split_sha256,
        seed=seed,
        training_mean=expected_training_mean,
        training_std=expected_training_std,
        excluded_groups=excluded,
    )
    if dict(receipt) != dict(recomputed.receipt):
        raise GroupPermutationError("permutation receipt differs from its recomputed group mapping or input digests")
    if receipt_sha256 != recomputed.receipt_sha256:
        raise GroupPermutationError("permutation receipt SHA-256 is invalid")
    observed_output = np.asarray(output_targets, dtype=np.float64)
    observed_mask = np.asarray(output_mask, dtype=np.bool_)
    if (observed_output.shape != recomputed.targets.shape
            or not np.array_equal(observed_output, recomputed.targets, equal_nan=True)
            or not np.array_equal(observed_mask, recomputed.mask)):
        raise GroupPermutationError("shuffled labels or effective mask differ from the recomputed result")
    return recomputed
