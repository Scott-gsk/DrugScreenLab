"""Measure foldwise DepMap whole-PatientID mask-exchange capacity (CPU only).

Only the previously verified finite/missing sidecar, registered DepMap identity
and CRISPR screen metadata, and frozen source-fold exclusions are read. No
dependency magnitudes, expression, drug-response labels, predictions, scores,
or checkpoints are opened. The result is a conditional capacity diagnostic;
unknown aliases and checkpoint exposure keep every slot DATA_BLOCKED.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "artifacts/experiments/EXP-045/20260925_depmap_fold_capacity_v1"
OUTPUT_RECEIPT = OUTPUT_DIR / "receipt.json"
ENVIRONMENT = "drugscreening-gpu"
GENE_COUNT = 16_100
AXIS_SHA256 = "9881e9f8184b3586c2211b3554f4fa9e76522055ff4384918f60b27576254965"

SIDECAR = ROOT / "artifacts/experiments/EXP-045/20260925_validity_mask_sidecars_v1/availability/dependency_validity_masks.npz"
SIDECAR_RECEIPT = ROOT / "artifacts/experiments/EXP-045/20260925_validity_mask_sidecars_v1/availability/run_receipt.json"
FOLD_MANIFEST = ROOT / "artifacts/experiments/EXP-045/identity_review_20260923_v3b/identity_manifest.json"
CANDIDATE_BINDING = ROOT / "artifacts/experiments/EXP-045/20260925_sanger_fold_candidate_binding_v1/receipt.json"
MODEL_CSV = ROOT / "data/raw/depmap/24q2/Model.csv"
SCREEN_MAP_CSV = ROOT / "data/raw/depmap/24q2/CRISPRScreenMap.csv"
SEQUENCE_MAP_CSV = ROOT / "data/raw/depmap/24q2/ScreenSequenceMap.csv"

EXPECTED_SHA256 = {
    "validity_mask_sidecar": "9eab5d6fcd693061bccd7550efe413098f756a59ee34f39e16dd46078f76ddf3",
    "depmap_model_csv": "a4cac376131b41aa10b60a075b11c80264bfa860a5509d22cde259c5e85867f8",
    "depmap_crispr_screen_map_csv": "7dd110f8eceff81e547e6529f5f00fd734239369e7a0e63836b4fdf533b805e0",
    "depmap_screen_sequence_map_csv": "fabaf0bfb091bb0ca1528a998dc28c91a5baae306231e358686116e1133ef0be",
    "frozen_source_fold_manifest": "c44e50edd88e93eb70cbe96de56327818d061c309ca4778c46d6ea550715edfc",
}
EXPECTED_CANDIDATES = {
    "SIDM00117": ("ACH-000837", "ACH-002172"),
    "SIDM00427": (),
    "SIDM01117": ("ACH-003071",),
}
EXPECTED_DEPMap_SOURCE_SHA256 = "d155149181308f8e16b6a4677cde3e5af1ddef68ada1947486ae8c0bcb231452"
EXPECTED_DEPMap_MASK_ARRAY_SHA256 = "405de9ecf83d709a015c148b7f3850d3b4d31738295a139aee9a94d485d94754"
EXPECTED_SOURCE_OBJECT_KEY_COUNT = 1_150


class CapacityAuditError(RuntimeError):
    """An input contract or runtime requirement failed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_values(values: Iterable[str]) -> str:
    payload = json.dumps(sorted(set(values)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CapacityAuditError(f"metadata JSON unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise CapacityAuditError(f"metadata JSON must be an object: {path.name}")
    return value


def _require_runtime() -> dict[str, Any]:
    release = platform.release()
    if sys.platform != "linux" or "microsoft" not in release.casefold():
        raise CapacityAuditError("run this audit in WSL2 Linux")
    if os.environ.get("CONDA_DEFAULT_ENV") != ENVIRONMENT:
        raise CapacityAuditError(f"run this audit in Conda {ENVIRONMENT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise CapacityAuditError("CPU audit requires CUDA_VISIBLE_DEVICES to be empty")
    return {
        "platform": sys.platform,
        "platform_release": release,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "conda_environment": ENVIRONMENT,
        "cuda_visible_devices": "",
        "device": "cpu",
        "gpu_used": False,
        "gpu_hours": 0,
    }


def _load_verified_sidecar() -> tuple[list[str], np.ndarray, dict[str, Any]]:
    actual_hash = _sha256_file(SIDECAR)
    if actual_hash != EXPECTED_SHA256["validity_mask_sidecar"]:
        raise CapacityAuditError("verified dependency validity sidecar checksum mismatch")
    source_receipt = _read_json(SIDECAR_RECEIPT)
    if (
        source_receipt.get("status") != "VALIDITY_MASK_SIDECAR_READY"
        or source_receipt.get("sidecar", {}).get("sha256") != actual_hash
        or source_receipt.get("sidecar", {}).get("contains_dependency_magnitudes") is not False
        or source_receipt.get("run", {}).get("loss_computed") is not False
    ):
        raise CapacityAuditError("sidecar source receipt does not bind the verified value-blind artifact")

    try:
        with np.load(SIDECAR, allow_pickle=False) as payload:
            required = {
                "schema_version", "axis_gene_count", "axis_sha256", "depmap_source_sha256",
                "nature_dependency_source_sha256", "nature_samples_source_sha256",
                "depmap_object_keys", "depmap_masks_packed", "nature_sample_ids",
                "nature_library_keys", "nature_masks_packed",
            }
            if set(payload.files) != required:
                raise CapacityAuditError("sidecar NPZ field set changed")
            if int(payload["schema_version"].item()) != 1:
                raise CapacityAuditError("sidecar schema version changed")
            gene_count = int(payload["axis_gene_count"].item())
            axis_sha = str(payload["axis_sha256"].item())
            source_sha = str(payload["depmap_source_sha256"].item())
            keys = [str(value) for value in payload["depmap_object_keys"].tolist()]
            packed = np.asarray(payload["depmap_masks_packed"])
            embedded_mask_hash = hashlib.sha256(packed.tobytes()).hexdigest()
            shape = tuple(int(value) for value in packed.shape)
            dtype = str(packed.dtype)
    except CapacityAuditError:
        raise
    except Exception as exc:
        raise CapacityAuditError("verified sidecar NPZ could not be read without pickle") from exc

    expected_bytes = (GENE_COUNT + 7) // 8
    if gene_count != GENE_COUNT or axis_sha != AXIS_SHA256:
        raise CapacityAuditError("sidecar frozen gene axis changed")
    if source_sha != EXPECTED_DEPMap_SOURCE_SHA256:
        raise CapacityAuditError("sidecar DepMap source byte binding changed")
    if len(keys) != EXPECTED_SOURCE_OBJECT_KEY_COUNT or len(set(keys)) != len(keys):
        raise CapacityAuditError("sidecar DepMap object keys are missing or duplicated")
    if shape != (len(keys), expected_bytes) or dtype != "uint8":
        raise CapacityAuditError("sidecar finite/missing bitmap shape or dtype changed")
    if embedded_mask_hash != EXPECTED_DEPMap_MASK_ARRAY_SHA256:
        raise CapacityAuditError("sidecar finite/missing bitmap array hash mismatch")
    # Arrays in the NPZ are copies, so the memory released here cannot affect the source artifact.
    with np.load(SIDECAR, allow_pickle=False) as payload:
        masks = np.asarray(payload["depmap_masks_packed"], dtype=np.uint8).copy()
    info = {
        "path": str(SIDECAR.relative_to(ROOT)).replace("\\", "/"),
        "sha256": actual_hash,
        "source_receipt_sha256": _sha256_file(SIDECAR_RECEIPT),
        "source_dependency_sha256_embedded": source_sha,
        "source_mask_array_sha256": embedded_mask_hash,
        "source_model_key_count": len(keys),
        "gene_count": gene_count,
        "ordered_gene_axis_sha256": axis_sha,
        "mask_representation": "finite/missing availability bits only; packed little-bit-order",
    }
    return keys, masks, info


def _verify_metadata_hashes() -> dict[str, str]:
    paths = {
        "depmap_model_csv": MODEL_CSV,
        "depmap_crispr_screen_map_csv": SCREEN_MAP_CSV,
        "depmap_screen_sequence_map_csv": SEQUENCE_MAP_CSV,
        "frozen_source_fold_manifest": FOLD_MANIFEST,
    }
    result: dict[str, str] = {}
    for name, path in paths.items():
        actual = _sha256_file(path)
        if actual != EXPECTED_SHA256[name]:
            raise CapacityAuditError(f"registered input checksum mismatch: {name}")
        result[str(path.relative_to(ROOT)).replace("\\", "/")] = actual
    return result


def _load_candidate_contract() -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    receipt = _read_json(CANDIDATE_BINDING)
    if (
        receipt.get("receipt_id") != "EXP-045-Sanger-fold-candidate-binding-v1"
        or receipt.get("status") != "IDENTITY_CANDIDATE_BINDING_COMPLETE"
        or receipt.get("provenance", {}).get("v3b_identity_manifest_sha256")
        != EXPECTED_SHA256["frozen_source_fold_manifest"]
    ):
        raise CapacityAuditError("Sanger candidate binding is not linked to the frozen source manifest")
    if receipt.get("provenance", {}).get("official_alias_bridge_sha256") != "8be00c61d9486ffa36fda4d77d8061e7133cb5acb3b69ed3befc78804ec6eb8c":
        raise CapacityAuditError("official Sanger alias bridge hash does not match the reviewed candidate binding")
    rows = receipt.get("per_object")
    if not isinstance(rows, list):
        raise CapacityAuditError("Sanger candidate binding rows are missing")
    candidates: dict[str, tuple[str, ...]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise CapacityAuditError("Sanger candidate binding row schema changed")
        sanger = row.get("Sanger_ID")
        ids = row.get("official_CMP", {}).get("Model_csv_exact_candidate_ModelIDs")
        if sanger in candidates or not isinstance(ids, list):
            raise CapacityAuditError("Sanger candidate binding keys are duplicated or malformed")
        candidates[str(sanger)] = tuple(sorted(str(value) for value in ids))
    if candidates != EXPECTED_CANDIDATES:
        raise CapacityAuditError("Sanger candidate identity set changed from the reviewed bridge")
    return candidates, {
        "path": str(CANDIDATE_BINDING.relative_to(ROOT)).replace("\\", "/"),
        "sha256": _sha256_file(CANDIDATE_BINDING),
        "official_alias_bridge_sha256": receipt["provenance"]["official_alias_bridge_sha256"],
        "frozen_manifest_sha256": receipt["provenance"]["v3b_identity_manifest_sha256"],
    }


def _load_model_metadata() -> dict[str, tuple[str, str]]:
    try:
        frame = pd.read_csv(
            MODEL_CSV,
            usecols=["ModelID", "PatientID", "OncotreeLineage"],
            dtype=str,
            keep_default_na=False,
            low_memory=False,
        )
    except Exception as exc:
        raise CapacityAuditError("DepMap Model.csv identity projection failed") from exc
    if frame["ModelID"].eq("").any() or frame["ModelID"].duplicated().any():
        raise CapacityAuditError("DepMap ModelID key is empty or duplicated")
    return {
        str(row.ModelID).strip(): (str(row.PatientID).strip(), str(row.OncotreeLineage).strip())
        for row in frame.itertuples(index=False)
    }


def _load_library_profiles() -> tuple[dict[str, tuple[tuple[str, str], ...]], dict[str, Any]]:
    try:
        screen_map = pd.read_csv(SCREEN_MAP_CSV, dtype=str, keep_default_na=False, low_memory=False)
        sequence_map = pd.read_csv(SEQUENCE_MAP_CSV, dtype=str, keep_default_na=False, low_memory=False)
    except Exception as exc:
        raise CapacityAuditError("DepMap CRISPR screen metadata projection failed") from exc
    if set(screen_map.columns) != {"ScreenID", "ModelID"}:
        raise CapacityAuditError("CRISPRScreenMap schema changed")
    required_sequence_columns = {"ScreenID", "ModelID", "Library", "ScreenType"}
    if not required_sequence_columns.issubset(sequence_map.columns):
        raise CapacityAuditError("ScreenSequenceMap required library profile fields changed")
    if screen_map[["ScreenID", "ModelID"]].eq("").any().any():
        raise CapacityAuditError("CRISPRScreenMap has empty screen/model keys")

    duplicate_pair_count = int(screen_map.duplicated(["ScreenID", "ModelID"]).sum())
    screen_map = screen_map.drop_duplicates(["ScreenID", "ModelID"])
    models_by_screen: dict[str, set[str]] = defaultdict(set)
    screens_by_model: dict[str, set[str]] = defaultdict(set)
    for screen_id, model_id in screen_map[["ScreenID", "ModelID"]].itertuples(index=False, name=None):
        models_by_screen[str(screen_id).strip()].add(str(model_id).strip())
        screens_by_model[str(model_id).strip()].add(str(screen_id).strip())

    profile_by_screen: dict[str, set[tuple[str, str]]] = defaultdict(set)
    sequence_models_by_screen: dict[str, set[str]] = defaultdict(set)
    for row in sequence_map.itertuples(index=False):
        screen_id = str(row.ScreenID).strip()
        model_id = str(row.ModelID).strip()
        library = str(row.Library).strip()
        screen_type = str(row.ScreenType).strip()
        if library and screen_type:
            profile_by_screen[screen_id].add((library, screen_type))
        if model_id:
            sequence_models_by_screen[screen_id].add(model_id)

    invalid_models: set[str] = set()
    profiles_by_model: dict[str, list[tuple[str, str]]] = defaultdict(list)
    profile_conflict_screen_count = 0
    profile_missing_screen_count = 0
    sequence_model_mismatch_screen_count = 0
    for screen_id, model_ids in models_by_screen.items():
        profiles = profile_by_screen.get(screen_id, set())
        if len(profiles) != 1:
            if profiles:
                profile_conflict_screen_count += 1
            else:
                profile_missing_screen_count += 1
            invalid_models.update(model_ids)
            continue
        sequence_model_ids = sequence_models_by_screen.get(screen_id, set())
        if sequence_model_ids and sequence_model_ids != model_ids:
            sequence_model_mismatch_screen_count += 1
            invalid_models.update(model_ids)
            continue
        profile = next(iter(profiles))
        for model_id in model_ids:
            profiles_by_model[model_id].append(profile)

    valid_profiles = {
        model_id: tuple(sorted(profiles))
        for model_id, profiles in profiles_by_model.items()
        if model_id not in invalid_models
    }
    return valid_profiles, {
        "crispr_screen_map_row_count_after_pair_deduplication": int(len(screen_map)),
        "duplicate_screen_model_pairs": duplicate_pair_count,
        "screen_map_model_count": len(screens_by_model),
        "screen_map_screen_count": len(models_by_screen),
        "valid_model_profile_count": len(valid_profiles),
        "invalid_model_profile_count": len(invalid_models),
        "profile_conflict_screen_count": profile_conflict_screen_count,
        "profile_missing_screen_count": profile_missing_screen_count,
        "sequence_model_mismatch_screen_count": sequence_model_mismatch_screen_count,
        "profile_rule": "per ModelID, map every CRISPRScreenMap ScreenID to the unique ScreenSequenceMap (Library, ScreenType); retain repeated screen profiles",
    }


def _build_group_signatures(
    model_ids: Sequence[str],
    masks_by_model: Mapping[str, bytes],
    metadata_by_model: Mapping[str, tuple[str, str]],
    profiles_by_model: Mapping[str, tuple[tuple[str, str], ...]],
) -> tuple[dict[str, tuple[Any, ...]], dict[str, set[str]], dict[str, int]]:
    patient_models: dict[str, set[str]] = defaultdict(set)
    invalid_reason_counts: Counter[str] = Counter()
    row_stratum_by_model: dict[str, str] = {}
    for model_id in model_ids:
        metadata = metadata_by_model.get(model_id)
        if metadata is None:
            invalid_reason_counts["model_metadata_missing"] += 1
            continue
        patient_id, lineage = metadata
        if not patient_id:
            invalid_reason_counts["patient_id_missing"] += 1
            continue
        patient_models[patient_id].add(model_id)
        if not lineage:
            invalid_reason_counts["oncotree_lineage_missing"] += 1
            continue
        profile = profiles_by_model.get(model_id)
        if not profile:
            invalid_reason_counts["screen_library_profile_missing_or_invalid"] += 1
            continue
        if model_id not in masks_by_model:
            invalid_reason_counts["sidecar_mask_missing"] += 1
            continue
        row_stratum_by_model[model_id] = json.dumps(
            {
                "oncotree_lineage": lineage,
                "library_screen_profile": profile,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    signatures: dict[str, tuple[Any, ...]] = {}
    invalid_groups: set[str] = set()
    for patient_id, members in patient_models.items():
        if any(member not in row_stratum_by_model for member in members):
            invalid_groups.add(patient_id)
            continue
        ordered = sorted(
            members,
            key=lambda model_id: (
                row_stratum_by_model[model_id].encode("utf-8"),
                model_id.encode("utf-8"),
            ),
        )
        signatures[patient_id] = tuple(
            (row_stratum_by_model[model_id], masks_by_model[model_id])
            for model_id in ordered
        )
    invalid_reason_counts["invalid_patient_groups"] = len(invalid_groups)
    return signatures, patient_models, dict(invalid_reason_counts)


def compatibility_summary(
    patient_groups: Iterable[str],
    signatures: Mapping[str, tuple[Any, ...]],
) -> dict[str, Any]:
    groups = set(patient_groups)
    compatible = groups & set(signatures)
    class_sizes = Counter(signatures[group] for group in compatible)
    histogram = Counter(class_sizes.values())
    exchangeable_group_count = sum(size for size in class_sizes.values() if size >= 2)
    exchangeable_pair_count = sum(size * (size - 1) // 2 for size in class_sizes.values() if size >= 2)
    return {
        "raw_patient_group_count": len(groups),
        "mask_and_profile_complete_group_count": len(compatible),
        "unusable_group_count": len(groups - compatible),
        "compatibility_class_count": len(class_sizes),
        "singleton_class_count": int(histogram.get(1, 0)),
        "nonidentity_exchangeable_class_count": sum(size >= 2 for size in class_sizes.values()),
        "nonidentity_exchangeable_group_count": int(exchangeable_group_count),
        "pairwise_exchangeable_pair_count": int(exchangeable_pair_count),
        "maximum_compatibility_class_size": max(class_sizes.values(), default=0),
        "compatibility_class_size_histogram": {
            str(size): int(class_count) for size, class_count in sorted(histogram.items())
        },
        "group_ids_sha256": _sha256_values(groups),
    }


def _extract_folds(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    folds = manifest.get("source_folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise CapacityAuditError("frozen source manifest must contain five outer folds")
    result: list[dict[str, Any]] = []
    outer_ids: set[int] = set()
    for outer in folds:
        if not isinstance(outer, dict) or not isinstance(outer.get("fold"), int):
            raise CapacityAuditError("frozen source outer fold schema changed")
        outer_id = int(outer["fold"])
        if outer_id in outer_ids:
            raise CapacityAuditError("duplicate frozen source outer fold id")
        outer_ids.add(outer_id)
        inner = outer.get("inner")
        if not isinstance(inner, list) or len(inner) != 3:
            raise CapacityAuditError("each frozen source outer fold must contain three inner folds")
        inner_ids: set[int] = set()
        for item in inner:
            if not isinstance(item, dict) or not isinstance(item.get("inner_fold"), int):
                raise CapacityAuditError("frozen source inner fold schema changed")
            inner_id = int(item["inner_fold"])
            if inner_id in inner_ids or item.get("fraction") != 1.0:
                raise CapacityAuditError("frozen source inner folds must be unique 100-percent partitions")
            inner_ids.add(inner_id)
        if inner_ids != {0, 1, 2}:
            raise CapacityAuditError("frozen source inner fold IDs changed")
        result.append(outer)
    if outer_ids != set(range(5)):
        raise CapacityAuditError("frozen source outer fold IDs changed")
    return sorted(result, key=lambda item: int(item["fold"]))


def _fold_partition_summary(
    *,
    level: str,
    outer_fold: int,
    inner_fold: int | None,
    excluded_model_ids: Sequence[str],
    unresolved_sanger_ids: Sequence[str],
    candidate_ids_by_sanger: Mapping[str, Sequence[str]],
    sidecar_keys: set[str],
    patient_models: Mapping[str, set[str]],
    metadata_by_model: Mapping[str, tuple[str, str]],
    signatures: Mapping[str, tuple[Any, ...]],
) -> dict[str, Any]:
    excluded_ids = {str(value).strip() for value in excluded_model_ids}
    if "" in excluded_ids or len(excluded_ids) != len(excluded_model_ids):
        raise CapacityAuditError("frozen fold excluded ModelID keys are empty or duplicated")
    split_patient_ids: set[str] = set()
    split_identity_missing_count = 0
    for model_id in excluded_ids:
        identity = metadata_by_model.get(model_id)
        if identity is None or not identity[0]:
            split_identity_missing_count += 1
        else:
            split_patient_ids.add(identity[0])

    raw_groups = set(patient_models)
    split_excluded_sidecar_groups = raw_groups & split_patient_ids
    key_level_training_keys = sidecar_keys - excluded_ids
    touched_training_groups = {
        patient_id
        for patient_id, members in patient_models.items()
        if members & key_level_training_keys
    }
    partial_split_groups = {
        patient_id
        for patient_id, members in patient_models.items()
        if members & excluded_ids and members & key_level_training_keys
    }
    split_group_closed_train_groups = raw_groups - split_patient_ids

    candidate_patient_ids: set[str] = set()
    candidate_status: dict[str, Any] = {}
    candidate_model_union: set[str] = set()
    for sanger_id, model_ids in candidate_ids_by_sanger.items():
        current = set(model_ids)
        candidate_model_union.update(current)
        model_hits = sorted(model_id for model_id in current if model_id in metadata_by_model)
        sidecar_hits = sorted(current & sidecar_keys)
        split_hits = sorted(current & excluded_ids)
        train_hits = sorted(current & key_level_training_keys)
        mapped_model_ids = {
            model_id for model_id in current
            if model_id in metadata_by_model and metadata_by_model[model_id][0]
        }
        known_patients = {
            metadata_by_model[model_id][0] for model_id in mapped_model_ids
        }
        candidate_patient_ids.update(known_patients)
        group_hits = known_patients & raw_groups
        candidate_status[sanger_id] = {
            "candidate_model_id_count": len(current),
            "candidate_model_metadata_hits": len(model_hits),
            "candidate_model_ids_in_sidecar": len(sidecar_hits),
            "candidate_model_ids_absent_from_sidecar": len(current - set(sidecar_hits)),
            "candidate_model_ids_already_in_fold_manifest_exclusion": len(split_hits),
            "candidate_model_ids_in_key_level_training_partition": len(train_hits),
            "candidate_patient_groups_with_sidecar_observations": len(group_hits),
            "candidate_model_ids_sha256": _sha256_values(current),
            "candidate_sidecar_model_ids_sha256": _sha256_values(sidecar_hits),
            "candidate_patient_groups_sha256": _sha256_values(known_patients),
            "known_identity_mapping_complete": len(mapped_model_ids) == len(current),
        }

    candidate_sidecar_groups = raw_groups & candidate_patient_ids
    candidate_groups_not_already_split_excluded = candidate_sidecar_groups - split_excluded_sidecar_groups
    complete_excluded_patient_ids = split_patient_ids | candidate_patient_ids
    complete_excluded_sidecar_groups = raw_groups & complete_excluded_patient_ids
    final_train_groups = raw_groups - complete_excluded_patient_ids

    pre_candidate_capacity = compatibility_summary(split_group_closed_train_groups, signatures)
    final_capacity = compatibility_summary(final_train_groups, signatures)
    all_sidecar_group_capacity = compatibility_summary(raw_groups, signatures)

    return {
        "level": level,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "training_fraction": 1.0 if level == "inner" else None,
        "unresolved_sanger_ids_registered_in_frozen_slot": {
            "count": len(set(str(value).strip() for value in unresolved_sanger_ids)),
            "ids_sha256": _sha256_values(str(value).strip() for value in unresolved_sanger_ids),
            "identity_or_checkpoint_exposure_resolved": False,
        },
        "frozen_manifest_split_exclusion": {
            "excluded_model_id_count": len(excluded_ids),
            "excluded_model_ids_matching_sidecar_keys": len(excluded_ids & sidecar_keys),
            "excluded_model_ids_missing_from_sidecar_keys": len(excluded_ids - sidecar_keys),
            "excluded_model_ids_resolving_to_model_patient_group": len(excluded_ids) - split_identity_missing_count,
            "excluded_model_ids_without_patient_group_mapping": split_identity_missing_count,
            "excluded_model_ids_sha256": _sha256_values(excluded_ids),
            "patient_group_count_in_model_metadata": len(split_patient_ids),
            "patient_group_count_with_sidecar_observations": len(split_excluded_sidecar_groups),
            "split_excluded_sidecar_group_count_with_partial_key_level_membership": len(partial_split_groups),
            "partial_group_sidecar_model_count_that_would_remain_at_modelid_level": sum(
                len(patient_models[patient_id] & key_level_training_keys)
                for patient_id in partial_split_groups
            ),
            "sidecar_missing_ids_are_not_missing_label_counts": True,
        },
        "candidate_identity_exclusions": {
            "per_registered_sanger_id": candidate_status,
            "candidate_patient_group_count_in_model_metadata": len(candidate_patient_ids),
            "candidate_patient_group_count_with_sidecar_observations": len(candidate_sidecar_groups),
            "candidate_sidecar_group_count_already_split_excluded": len(
                candidate_sidecar_groups & split_excluded_sidecar_groups
            ),
            "candidate_sidecar_group_count_newly_removed_from_fold_training": len(
                candidate_groups_not_already_split_excluded
            ),
            "candidate_patient_groups_sha256": _sha256_values(candidate_patient_ids),
            "candidate_ModelID_set_sha256": _sha256_values(candidate_model_union),
            "unmapped_alias_or_checkpoint_exposure_remains_unknown": True,
        },
        "group_partition": {
            "raw_sidecar_patient_group_count": len(raw_groups),
            "raw_sidecar_model_count": len(sidecar_keys),
            "sidecar_model_ids_without_patient_group": len(sidecar_keys) - sum(
                len(members) for members in patient_models.values()
            ),
            "key_level_training_model_id_count_before_patient_group_closure": len(key_level_training_keys),
            "key_level_training_patient_group_count_touched": len(touched_training_groups),
            "fold_split_excluded_patient_group_count_closure": len(split_excluded_sidecar_groups),
            "candidate_patient_group_count_added_to_exclusion_union": len(
                candidate_groups_not_already_split_excluded
            ),
            "complete_excluded_patient_group_count_in_sidecar_source": len(complete_excluded_sidecar_groups),
            "complete_excluded_patient_group_union_sha256": _sha256_values(complete_excluded_patient_ids),
            "complete_excluded_sidecar_patient_group_sha256": _sha256_values(complete_excluded_sidecar_groups),
            "patient_group_closed_training_group_count_before_candidate_exclusion": len(
                split_group_closed_train_groups
            ),
            "final_patient_group_closed_training_group_count_after_candidate_exclusion": len(final_train_groups),
            "patient_group_closed_training_model_count_after_all_exclusions": sum(
                len(patient_models[group]) for group in final_train_groups
            ),
        },
        "capacity_before_candidate_identity_exclusion": pre_candidate_capacity,
        "capacity_after_complete_patient_group_exclusion": final_capacity,
        "unfiltered_sidecar_capacity_reference": all_sidecar_group_capacity,
        "slot_training_gate": "DATA_BLOCKED",
        "slot_blockers": [
            "SANGER_ALIAS_OR_BIOLOGICAL_DERIVATIVE_SET_INCOMPLETE",
            "SOURCE_CHECKPOINT_FIT_SELECTION_TEST_EXPOSURE_UNKNOWN",
            "FOLD_MASK_EXECUTION_IN_TRAINING_ADAPTER_UNBOUND",
        ],
        "capacity_is_conditional_diagnostic_only": True,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"refusing to replace prior partial artifact: {temporary}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_audit(command: str) -> dict[str, Any]:
    if OUTPUT_RECEIPT.exists():
        raise FileExistsError(f"refusing to overwrite capacity receipt: {OUTPUT_RECEIPT}")
    started_wall = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    runtime = _require_runtime()
    input_hashes = _verify_metadata_hashes()
    candidates, candidate_input = _load_candidate_contract()
    sidecar_keys_list, packed_masks, sidecar_info = _load_verified_sidecar()
    manifest = _read_json(FOLD_MANIFEST)
    folds = _extract_folds(manifest)
    metadata_by_model = _load_model_metadata()
    profiles_by_model, profile_summary = _load_library_profiles()

    sidecar_keys = set(sidecar_keys_list)
    if sidecar_keys != set(sidecar_keys_list):
        raise CapacityAuditError("sidecar ModelID keys are not unique")
    masks_by_model = {
        model_id: packed_masks[row].tobytes()
        for row, model_id in enumerate(sidecar_keys_list)
    }
    signatures, patient_models, invalid_group_summary = _build_group_signatures(
        sidecar_keys_list,
        masks_by_model,
        metadata_by_model,
        profiles_by_model,
    )
    candidate_ids = {
        sanger_id: list(model_ids)
        for sanger_id, model_ids in candidates.items()
    }

    slots: list[dict[str, Any]] = []
    for outer in folds:
        outer_id = int(outer["fold"])
        outer_excluded = outer.get("excluded_model_ids")
        if not isinstance(outer_excluded, list):
            raise CapacityAuditError("outer frozen source exclusions are missing")
        outer_unresolved = outer.get("unresolved_sanger_ids")
        if not isinstance(outer_unresolved, list):
            raise CapacityAuditError("outer frozen source unresolved aliases are missing")
        if int(outer.get("excluded_model_count", -1)) != len(outer_excluded):
            raise CapacityAuditError("outer frozen exclusion count does not match its key list")
        slots.append(_fold_partition_summary(
            level="outer",
            outer_fold=outer_id,
            inner_fold=None,
            excluded_model_ids=outer_excluded,
            unresolved_sanger_ids=outer_unresolved,
            candidate_ids_by_sanger=candidate_ids,
            sidecar_keys=sidecar_keys,
            patient_models=patient_models,
            metadata_by_model=metadata_by_model,
            signatures=signatures,
        ))
        for inner in sorted(outer["inner"], key=lambda item: int(item["inner_fold"])):
            inner_excluded = inner.get("excluded_model_ids")
            if not isinstance(inner_excluded, list):
                raise CapacityAuditError("inner frozen source exclusions are missing")
            inner_unresolved = inner.get("unresolved_sanger_ids")
            if not isinstance(inner_unresolved, list):
                raise CapacityAuditError("inner frozen source unresolved aliases are missing")
            if int(inner.get("excluded_model_count", -1)) != len(inner_excluded):
                raise CapacityAuditError("inner frozen exclusion count does not match its key list")
            slots.append(_fold_partition_summary(
                level="inner",
                outer_fold=outer_id,
                inner_fold=int(inner["inner_fold"]),
                excluded_model_ids=inner_excluded,
                unresolved_sanger_ids=inner_unresolved,
                candidate_ids_by_sanger=candidate_ids,
                sidecar_keys=sidecar_keys,
                patient_models=patient_models,
                metadata_by_model=metadata_by_model,
                signatures=signatures,
            ))

    elapsed = time.perf_counter() - started_wall
    return {
        "receipt_id": "EXP-045-DepMap-fold-PatientID-mask-capacity-v1",
        "status": "CAPACITY_DIAGNOSTIC_COMPLETE_TRAINING_BLOCKED",
        "experiment": "EXP-045",
        "created_at_utc": started_at,
        "scope": {
            "question": "For each frozen source outer/inner training partition, how many whole DepMap PatientID groups remain in exact finite/missing-mask and screen-profile compatibility classes after split group closure and conservative Sanger candidate exclusions?",
            "data_access": "only dependency sidecar key strings and finite/missing bits; DepMap Model.csv identity fields ModelID/PatientID/OncotreeLineage; CRISPRScreenMap ScreenID/ModelID; ScreenSequenceMap ScreenID/ModelID/Library/ScreenType; frozen source fold excluded_model_ids; reviewed Sanger candidate receipt",
            "forbidden_data_read": [
                "dependency magnitudes or labels",
                "expression values",
                "drug sensitivity labels",
                "predictions or scores",
                "checkpoint contents or tensors",
            ],
            "source_group_unit": "all sidecar ModelIDs mapped to one DepMap Model.csv PatientID are one indivisible group",
            "fold_split_semantics": "frozen excluded_model_ids are first applied at ModelID level for direct-key counts, then any PatientID group containing one such ID is fully excluded from fold training",
            "candidate_exclusion_semantics": "every known exact Sanger ModelID candidate resolves through Model.csv PatientID and excludes that complete PatientID group, including sibling sidecar ModelIDs; IDs absent from sidecar remain separately reported",
            "compatibility": "same number of observations and exact ordered per-row OncotreeLineage, sorted Library/ScreenType multiset, and packed finite/missing mask across all frozen output-axis genes",
            "training_or_science_comparison": False,
            "formal_training_go": False,
        },
        "inputs": {
            **{
                path: {"sha256": digest, "expected_sha256": digest, "verified": True}
                for path, digest in input_hashes.items()
            },
            "dependency_validity_sidecar": sidecar_info,
            "sidecar_source_run_receipt": {
                "path": str(SIDECAR_RECEIPT.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sidecar_info["source_receipt_sha256"],
            },
            "sanger_candidate_binding": candidate_input,
        },
        "sanger_identity": {
            "candidate_ModelIDs_by_registered_Sanger_ID": {
                key: list(value) for key, value in candidates.items()
            },
            "unresolved_registered_Sanger_IDs": ["SIDM00117", "SIDM00427", "SIDM01117"],
            "unknown_alias_and_checkpoint_exposure_are_not_resolved_by_this_capacity_run": True,
        },
        "static_input_profile": {
            **profile_summary,
            "raw_sidecar_patient_group_count": len(patient_models),
            "sidecar_model_ids_without_patient_group": len(sidecar_keys) - sum(
                len(members) for members in patient_models.values()
            ),
            "sidecar_groups_invalid_for_mask_exchange": invalid_group_summary.get("invalid_patient_groups", 0),
            "invalid_group_reason_counts": invalid_group_summary,
            "all_counts_are_identity_profile_or_availability_bits_only": True,
        },
        "fold_slots": slots,
        "limitations": [
            "This reports a counterfactual patient-group-closed capacity after the frozen ModelID exclusions; it does not prove the production training adapter used the same exclusions or sidecar masks.",
            "Known Sanger ModelID candidates are conservatively excluded by whole PatientID, but biological aliases/derivatives beyond the reviewed candidates remain unresolved.",
            "Per-fold source checkpoint fit, selection, and test exposure remain unknown; all 20 fold slots remain DATA_BLOCKED regardless of capacity.",
            "Mask compatibility counts are conditional exchangeability diagnostics, not scientific performance, a loss result, a dependency-label summary, or formal training GO.",
            "A frozen excluded ModelID absent from the sidecar is a key-coverage difference, not evidence of a missing dependency label.",
        ],
        "minimum_unlocks": [
            "resolve or quarantine all affected source biological aliases and checkpoint exposure per fold",
            "bind patient-group exclusions and the exact sidecar mask hash to the actual training adapter",
            "then run the separately authorized real-input smoke, independent holdout gate, and cost/budget gate",
        ],
        "run": {
            "command": command,
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 3),
            "environment": runtime,
            "code_sha256": _sha256_file(Path(__file__).resolve()),
            "training_performed": False,
            "labels_or_values_read": False,
            "checkpoint_opened": False,
            "gpu_used": False,
            "gpu_hours": 0,
            "scientific_performance_comparison": False,
            "exit_code": 0,
        },
        "next_action": {
            "action": "Review this foldwise capacity receipt together with the EXP-045 adapter-mask and checkpoint-exposure lineage audit; do not use capacity alone to authorize training.",
            "next_executable_command": "wsl.exe --distribution Ubuntu --cd /mnt/d/Code/DrugScreenLab -- /usr/bin/env CUDA_VISIBLE_DEVICES= /home/dell/miniconda3/bin/conda run --no-capture-output -n drugscreening-gpu python scripts/data/audit_exp045_identity_fold_preflight.py",
            "command_is_expected_to_remain_data_blocked_until_identity_and_lineage_inputs_change": True,
        },
    }


def main() -> int:
    started = time.perf_counter()
    command = (
        "wsl.exe --distribution Ubuntu --cd /mnt/d/Code/DrugScreenLab -- "
        "/usr/bin/env CUDA_VISIBLE_DEVICES= /home/dell/miniconda3/bin/conda run "
        "--no-capture-output -n drugscreening-gpu python "
        "scripts/data/audit_exp045_depmap_fold_capacity.py"
    )
    try:
        receipt = run_audit(command)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if OUTPUT_RECEIPT.exists():
            raise FileExistsError(f"refusing to overwrite capacity receipt: {OUTPUT_RECEIPT}")
        _write_json_atomic(OUTPUT_RECEIPT, receipt)
        overall_elapsed = time.perf_counter() - started
        print(json.dumps({
            "status": receipt["status"],
            "receipt": str(OUTPUT_RECEIPT.relative_to(ROOT)).replace("\\", "/"),
            "fold_slots": len(receipt["fold_slots"]),
            "elapsed_seconds": round(overall_elapsed, 3),
            "gpu_hours": 0,
        }, ensure_ascii=False))
        return 0
    except Exception as exc:
        failure_path = OUTPUT_DIR / "failure.json"
        if not failure_path.exists():
            _write_json_atomic(failure_path, {
                "receipt_id": "EXP-045-DepMap-fold-PatientID-mask-capacity-v1",
                "status": "FAILED_CLOSED",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "code_sha256": _sha256_file(Path(__file__).resolve()),
                "training_performed": False,
                "gpu_used": False,
                "gpu_hours": 0,
            })
        print(f"FAILED_CLOSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
