"""Foldwise Nature CRC mask and whole-individual exchangeability audit.

This CPU-only audit consumes the approved finite/missing sidecar, its keyed
sample/library rows, registered sample identity/cancer fields, the official
Nature overlap table, and frozen target fold metadata. It never reads
dependency magnitudes, expression, drug-response labels, predictions, scores,
or checkpoints. Capacity is conditional on the currently identified
cross-source quarantine being exhaustive; every fold remains DATA_BLOCKED.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
import platform
import sys
import time
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.audit_exp045_identity_fold_preflight import (  # noqa: E402
    group_exclusion_status as shared_group_exclusion_status,
    select_fraction_inner_folds as shared_select_fraction_inner_folds,
)


DEFAULT_OUTPUT = ROOT / "artifacts/experiments/EXP-045/20260925_nature_fold_capacity_v1"
ENVIRONMENT = "drugscreening-gpu"
MASK_SIDECAR = ROOT / "artifacts/experiments/EXP-045/20260925_validity_mask_sidecars_v1/availability/dependency_validity_masks.npz"
MASK_RECEIPT = ROOT / "artifacts/experiments/EXP-045/20260925_validity_mask_sidecars_v1/availability/run_receipt.json"
SAMPLES_PARQUET = ROOT / "data/processed/transfer/biobank_functional_v1/samples.parquet"
OVERLAP_RECEIPT = ROOT / "artifacts/experiments/EXP-045/20260925_source_provenance_overlap_recovery_v1/receipt.json"
FOLD_MANIFEST = ROOT / "artifacts/experiments/EXP-045/identity_review_20260923_v3b/identity_manifest.json"
NATURE_TABLE2_ZIP = ROOT / "data/raw/transfer_public_assets/organoid_biobank/41586_2026_10830_MOESM3_ESM.zip"
NATURE_TABLE2_MEMBER = "supplementary_table_2_revision.xlsx"
NATURE_TABLE2_SHEET = "data_availability"
AXIS_CONTRACT = ROOT / "artifacts/experiments/EXP-045/20260924_gene_axis_audit_v2/axis_contract.json"
EXPECTED_MASK_SHA256 = "9eab5d6fcd693061bccd7550efe413098f756a59ee34f39e16dd46078f76ddf3"
EXPECTED_SIDECAR_SCOPE = "one CPU-only finite/missing mask scan; no dependency values or statistics retained"
EXPECTED_MASK_INPUT_SHA256 = {
    "nature_dependency_by_library": "bf40474b804ab3bb9029c551cd0f7a8604dbdcbead7011ba2e96894f17f20fb7",
    "nature_samples": "2864dc5340d6bca73aea88ece6dfbad533cd100f245152915df1e200773892ac",
}
EXPECTED_NATURE_TABLE2_ZIP_SHA256 = "7e83b7551c9c272de0bc0455ea9a0e7803a8043d15d9cfedf5a1c3e609a2a6e2"
EXPECTED_NATURE_TABLE2_MEMBER_SHA256 = "5b319152fd3d708ad6be4de62d3655b860cb2ecb08bfb3de9892a25898559cc9"
EXPECTED_SAMPLES_SHA256 = "2864dc5340d6bca73aea88ece6dfbad533cd100f245152915df1e200773892ac"
EXPECTED_AXIS_SHA256 = "9881e9f8184b3586c2211b3554f4fa9e76522055ff4384918f60b27576254965"
MASK_FIELDS = {
    "schema_version",
    "axis_gene_count",
    "axis_sha256",
    "depmap_source_sha256",
    "nature_dependency_source_sha256",
    "nature_samples_source_sha256",
    "depmap_object_keys",
    "depmap_masks_packed",
    "nature_sample_ids",
    "nature_library_keys",
    "nature_masks_packed",
}
RUN_COMMAND = (
    "wsl.exe --distribution Ubuntu --cd /mnt/d/Code/DrugScreenLab -- "
    "/bin/bash -lc 'CUDA_VISIBLE_DEVICES= "
    "/home/dell/miniconda3/bin/conda run --no-capture-output "
    "-n drugscreening-gpu python scripts/data/audit_exp045_nature_fold_capacity.py "
    "--output-dir artifacts/experiments/EXP-045/20260925_nature_fold_capacity_v1'"
)


class CapacityAuditError(RuntimeError):
    """Raised when registered identity or sidecar inputs cannot be verified."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _key_set_summary(values: Iterable[str]) -> dict[str, Any]:
    ordered = sorted(set(values))
    return {"count": len(ordered), "sha256": sha256_json(ordered)}


def _pairwise_key_set_summary(
    left_name: str,
    left: set[str],
    right_name: str,
    right: set[str],
) -> dict[str, Any]:
    return {
        "left": left_name,
        "right": right_name,
        "intersection": _key_set_summary(left & right),
        "left_only": _key_set_summary(left - right),
        "right_only": _key_set_summary(right - left),
    }


def require_runtime() -> dict[str, Any]:
    release = platform.release()
    if sys.platform != "linux" or "microsoft" not in release.casefold():
        raise RuntimeError("run this audit in WSL2 Linux")
    if os.environ.get("CONDA_DEFAULT_ENV") != ENVIRONMENT:
        raise RuntimeError(f"run this audit in Conda {ENVIRONMENT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("run this audit with CUDA_VISIBLE_DEVICES empty")
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
        "training_performed": False,
    }


def compatibility_summary(group_signatures: Mapping[str, Any]) -> dict[str, Any]:
    """Use the EXP-045 whole-group exact signature compatibility rule."""
    class_sizes = Counter(group_signatures.values())
    histogram = Counter(class_sizes.values())
    exchangeable_groups = sum(size for size in class_sizes.values() if size >= 2)
    permutation_space = math.prod(math.factorial(size) for size in class_sizes.values())
    return {
        "valid_group_count": len(group_signatures),
        "compatibility_class_count": len(class_sizes),
        "singleton_class_count": histogram.get(1, 0),
        "nonidentity_class_count": sum(size >= 2 for size in class_sizes.values()),
        "nonidentity_exchangeable_group_count": exchangeable_groups,
        "maximum_class_size": max(class_sizes.values(), default=0),
        "class_size_histogram": {
            str(size): int(class_count)
            for size, class_count in sorted(histogram.items())
        },
        "all_group_permutation_space_including_identity_decimal": str(permutation_space),
        "nonidentity_group_permutation_count_decimal": str(max(0, permutation_space - 1)),
    }


def _safe_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def selected_inner_folds(
    folds: Sequence[Mapping[str, Any]], fraction: float = 0.5, expected: int = 3
) -> list[tuple[int, Mapping[str, Any]]]:
    try:
        return shared_select_fraction_inner_folds(
            folds, fraction=fraction, expected=expected
        )
    except Exception as exc:
        raise CapacityAuditError(
            "the existing EXP-045 frozen fold selector rejected the target manifest"
        ) from exc


def _load_sidecar() -> dict[str, Any]:
    if sha256_file(MASK_SIDECAR) != EXPECTED_MASK_SHA256:
        raise CapacityAuditError("approved validity-mask sidecar SHA256 changed")
    side_receipt = json.loads(MASK_RECEIPT.read_text(encoding="utf-8"))
    if (
        side_receipt.get("status") != "VALIDITY_MASK_SIDECAR_READY"
        or side_receipt.get("scope") != EXPECTED_SIDECAR_SCOPE
        or side_receipt.get("sidecar", {}).get("sha256") != EXPECTED_MASK_SHA256
    ):
        raise CapacityAuditError("validity-mask sidecar receipt changed")
    for name, expected in EXPECTED_MASK_INPUT_SHA256.items():
        if side_receipt.get("inputs", {}).get(name, {}).get("actual_sha256") != expected:
            raise CapacityAuditError(f"sidecar source hash binding changed: {name}")
        if side_receipt.get("inputs", {}).get(name, {}).get("sha256_match") is not True:
            raise CapacityAuditError(f"sidecar source checksum was not matched: {name}")
    arrays: dict[str, Any] = {}
    with np.load(MASK_SIDECAR, allow_pickle=False) as archive:
        if set(archive.files) != MASK_FIELDS:
            raise CapacityAuditError("validity-mask sidecar field set changed")
        arrays = {name: archive[name].copy() for name in archive.files}
    gene_count = int(arrays["axis_gene_count"])
    if (
        int(arrays["schema_version"]) != 1
        or gene_count != 16_100
        or str(arrays["axis_sha256"]) != EXPECTED_AXIS_SHA256
        or str(arrays["nature_dependency_source_sha256"])
        != EXPECTED_MASK_INPUT_SHA256["nature_dependency_by_library"]
        or str(arrays["nature_samples_source_sha256"])
        != EXPECTED_MASK_INPUT_SHA256["nature_samples"]
    ):
        raise CapacityAuditError("sidecar schema, gene axis, or Nature source binding changed")
    sample_ids = arrays["nature_sample_ids"].astype(str)
    libraries = arrays["nature_library_keys"].astype(str)
    masks = arrays["nature_masks_packed"]
    expected_width = (gene_count + 7) // 8
    if (
        masks.shape != (85, expected_width)
        or len(sample_ids) != 85
        or len(libraries) != 85
        or len(set(sample_ids.tolist())) != 85
        or len(set(zip(sample_ids.tolist(), libraries.tolist()))) != 85
        or any(not value for value in sample_ids.tolist())
        or any(not value for value in libraries.tolist())
    ):
        raise CapacityAuditError("Nature sidecar keys or packed-mask dimensions changed")
    array_hashes = side_receipt.get("sidecar", {}).get("validity_array_hashes", {})
    key_hash = hashlib.sha256(
        "\n".join(
            f"{sample}\t{library}" for sample, library in zip(sample_ids, libraries)
        ).encode("utf-8")
    ).hexdigest()
    if key_hash != array_hashes.get("nature_key_array_sha256"):
        raise CapacityAuditError("Nature sample/library key digest changed")
    if hashlib.sha256(masks.tobytes()).hexdigest() != array_hashes.get(
        "nature_mask_array_sha256"
    ):
        raise CapacityAuditError("Nature validity-mask digest changed")
    unpacked = np.unpackbits(masks, axis=1, count=gene_count, bitorder="little")
    observed_valid = int(np.count_nonzero(unpacked))
    expected_valid = int(side_receipt["nature_colorectal"]["valid_bit_count"])
    if observed_valid != expected_valid or observed_valid != 1_368_500:
        raise CapacityAuditError("Nature validity-bit aggregate differs from the receipt")
    return {
        "receipt": side_receipt,
        "sample_ids": sample_ids,
        "libraries": libraries,
        "packed_masks": masks,
        "gene_count": gene_count,
        "sample_to_row": {sample: row for row, sample in enumerate(sample_ids)},
        "valid_bit_count": observed_valid,
        "missing_bit_count": int(masks.shape[0] * gene_count - observed_valid),
        "key_array_sha256": key_hash,
        "mask_array_sha256": hashlib.sha256(masks.tobytes()).hexdigest(),
    }


def _load_official_overlap(
    overlap_receipt: Mapping[str, Any],
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any], dict[str, Any]]:
    if sha256_file(NATURE_TABLE2_ZIP) != EXPECTED_NATURE_TABLE2_ZIP_SHA256:
        raise CapacityAuditError("registered official Nature supplement ZIP changed")
    try:
        with zipfile.ZipFile(NATURE_TABLE2_ZIP) as archive:
            workbook = archive.read(NATURE_TABLE2_MEMBER)
    except Exception as exc:
        raise CapacityAuditError("official Nature Table 2 workbook is unavailable") from exc
    if hashlib.sha256(workbook).hexdigest() != EXPECTED_NATURE_TABLE2_MEMBER_SHA256:
        raise CapacityAuditError("official Nature Table 2 member SHA256 changed")
    try:
        official = pd.read_excel(
            BytesIO(workbook),
            sheet_name=NATURE_TABLE2_SHEET,
            usecols=[
                "sample_ID",
                "individual_ID",
                "primary_tumour_type",
                "overlap_Broad_DepMap",
            ],
            dtype=str,
            keep_default_na=False,
        )
    except Exception as exc:
        raise CapacityAuditError("official identity/cancer fields could not be read") from exc
    for column in official.columns:
        official[column] = official[column].astype(str).str.strip()
    if (
        official["sample_ID"].eq("").any()
        or official["individual_ID"].eq("").any()
        or official["sample_ID"].duplicated().any()
    ):
        raise CapacityAuditError("official Nature Table 2 sample/group keys are incomplete")
    yes = official[
        official["overlap_Broad_DepMap"].str.casefold().eq("yes")
    ]
    yes_sample_ids = sorted(set(yes["sample_ID"]))
    yes_individual_ids = sorted(set(yes["individual_ID"]))
    registered_groups = overlap_receipt.get("nature_overlap_result", {}).get(
        "individual_to_sample_ID_groups"
    )
    if not isinstance(registered_groups, dict):
        raise CapacityAuditError("registered 13-group quarantine mapping is missing")
    normalized_registered = {
        str(individual): tuple(sorted(str(sample).strip() for sample in sample_ids))
        for individual, sample_ids in registered_groups.items()
    }
    registered_sample_ids = {
        sample for sample_ids in normalized_registered.values() for sample in sample_ids
    }
    official_group_by_sample = dict(zip(official["sample_ID"], official["individual_ID"]))
    official_cancer_by_sample = dict(
        zip(official["sample_ID"], official["primary_tumour_type"])
    )
    if (
        len(normalized_registered) != 13
        or len(registered_sample_ids) != 14
        or len(yes) != 18
        or len(yes_sample_ids) != 18
        or len(yes_individual_ids) != 17
        or any(
            official_group_by_sample.get(sample_id) != individual_id
            for individual_id, sample_ids in normalized_registered.items()
            for sample_id in sample_ids
        )
    ):
        raise CapacityAuditError(
            "official Table 2 or the registered 13-group/14-sample quarantine mapping changed"
        )
    nature_result = overlap_receipt.get("nature_overlap_result", {})
    expected_registered_counts = {
        "explicit_overlap_Broad_DepMap_yes_rows": 13,
        "unique_yes_sample_IDs": 13,
        "unique_individual_ID_groups": 13,
        "samples_after_grouping_duplicate_samples": 14,
        "sample_ID_library_pairs": 17,
    }
    if any(nature_result.get(key) != value for key, value in expected_registered_counts.items()):
        raise CapacityAuditError("recorded approved 13/14/17 quarantine counts changed")
    result = {
        "official_table2_row_count": int(len(official)),
        "full_official_Broad_DepMap_yes_row_count": int(len(yes)),
        "full_official_yes_sample_id_count": len(yes_sample_ids),
        "full_official_yes_individual_group_count": len(yes_individual_ids),
        "approved_current_input_quarantine_individual_group_count": len(normalized_registered),
        "approved_current_input_quarantine_sample_id_count": len(registered_sample_ids),
        "approved_current_input_quarantine_sample_library_pair_count": nature_result.get(
            "sample_ID_library_pairs"
        ),
        "registered_group_membership_matches_official_sample_individual_keys": True,
        "group_map_sha256": sha256_json(
            sorted(
                (individual, list(sample_ids))
                for individual, sample_ids in normalized_registered.items()
            )
        ),
        "sample_set_sha256": sha256_json(sorted(registered_sample_ids)),
        "explicit_yes_individual_id_set_sha256": sha256_json(yes_individual_ids),
        "fields_used": [
            "sample_ID",
            "individual_ID",
            "primary_tumour_type",
            "overlap_Broad_DepMap",
        ],
    }
    group_map_hash_safe = {
        group: tuple(members) for group, members in normalized_registered.items()
    }
    return group_map_hash_safe, result, {
        "sample_to_group": official_group_by_sample,
        "sample_to_cancer": official_cancer_by_sample,
        "full_official_yes_sample_ids": set(yes_sample_ids),
        "full_official_yes_individual_ids": set(yes_individual_ids),
    }


def _load_sample_identity() -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    if sha256_file(SAMPLES_PARQUET) != EXPECTED_SAMPLES_SHA256:
        raise CapacityAuditError("registered Nature sample metadata checksum changed")
    samples = pd.read_parquet(
        SAMPLES_PARQUET,
        columns=[
            "sample_ID",
            "individual_ID",
            "primary_tumour_type",
            "overlap_Broad_DepMap",
        ],
    )
    for column in (
        "sample_ID",
        "individual_ID",
        "primary_tumour_type",
        "overlap_Broad_DepMap",
    ):
        samples[column] = samples[column].map(_safe_id)
    if (
        samples["sample_ID"].eq("").any()
        or samples["individual_ID"].eq("").any()
        or samples["sample_ID"].duplicated().any()
    ):
        raise CapacityAuditError("Nature sample metadata has missing or duplicate keys")
    crc = samples[
        samples["primary_tumour_type"].str.casefold().eq("colorectal")
    ].copy()
    crc_sample_to_group = dict(zip(crc["sample_ID"], crc["individual_ID"]))
    yes = samples[
        samples["overlap_Broad_DepMap"].str.casefold().eq("yes")
    ]
    yes_groups = set(yes["individual_ID"])
    expanded_registered_samples = set(
        samples.loc[samples["individual_ID"].isin(yes_groups), "sample_ID"]
    )
    return crc_sample_to_group, {
        "samples_metadata_sha256": EXPECTED_SAMPLES_SHA256,
        "selected_columns": [
            "sample_ID",
            "individual_ID",
            "primary_tumour_type",
            "overlap_Broad_DepMap",
        ],
        "sample_metadata_row_count": int(len(samples)),
        "colorectal_sample_object_count": int(len(crc)),
        "colorectal_individual_group_count": int(crc["individual_ID"].nunique()),
        "registered_input_Broad_DepMap_yes_row_count": int(len(yes)),
        "registered_input_Broad_DepMap_yes_sample_id_count": int(yes["sample_ID"].nunique()),
        "registered_input_Broad_DepMap_yes_individual_group_count": len(yes_groups),
        "registered_input_whole_yes_group_sample_id_count": len(
            expanded_registered_samples
        ),
        "registered_input_yes_sample_id_set_sha256": sha256_json(
            sorted(set(yes["sample_ID"]))
        ),
        "registered_input_yes_individual_id_set_sha256": sha256_json(
            sorted(yes_groups)
        ),
        "registered_input_expanded_group_sample_id_set_sha256": sha256_json(
            sorted(expanded_registered_samples)
        ),
        "all_registered_sample_id_set_sha256": sha256_json(
            sorted(set(samples["sample_ID"]))
        ),
        "colorectal_group_size_histogram": {
            str(size): int(count)
            for size, count in sorted(Counter(crc.groupby("individual_ID").size()).items())
        },
    }, {
        "all_sample_ids": set(samples["sample_ID"]),
        "all_sample_to_group": dict(zip(samples["sample_ID"], samples["individual_ID"])),
        "processed_yes_sample_ids": set(yes["sample_ID"]),
        "approved_group_ids": yes_groups,
        "expanded_approved_sample_ids": expanded_registered_samples,
    }


def build_group_signatures(
    sample_to_group: Mapping[str, str],
    sidecar_sample_ids: Sequence[str],
    sidecar_libraries: Sequence[str],
    packed_masks: np.ndarray,
    excluded_group_ids: Iterable[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build complete individual_ID signatures using only library keys and bitmaps."""
    if not (
        len(sidecar_sample_ids) == len(sidecar_libraries) == packed_masks.shape[0]
    ):
        raise CapacityAuditError("key and mask row counts differ")
    excluded = {str(value) for value in excluded_group_ids}
    observations: dict[str, list[tuple[str, str, bytes]]] = defaultdict(list)
    unmapped_samples: list[str] = []
    for sample, library, mask in zip(sidecar_sample_ids, sidecar_libraries, packed_masks):
        individual = sample_to_group.get(str(sample))
        if individual is None:
            unmapped_samples.append(str(sample))
            continue
        observations[individual].append(
            (str(sample), str(library), np.asarray(mask, dtype=np.uint8).tobytes())
        )
    signatures: dict[str, Any] = {}
    for individual, group_observations in observations.items():
        if individual in excluded:
            continue
        group_observations.sort(key=lambda item: item[0])
        signatures[individual] = (
            len(group_observations),
            tuple(
                ("Colorectal", library, packed_mask)
                for _, library, packed_mask in group_observations
            ),
        )
    return signatures, {
        "sidecar_sample_count": len(sidecar_sample_ids),
        "mapped_sample_count": len(sidecar_sample_ids) - len(unmapped_samples),
        "unmapped_sample_count": len(unmapped_samples),
        "unmapped_sample_id_set_sha256": sha256_json(sorted(unmapped_samples)),
        "sidecar_group_count_before_exclusion": len(observations),
        "excluded_quarantine_group_count_present_in_sidecar": len(
            set(observations) & excluded
        ),
        "excluded_quarantine_sidecar_sample_count": sum(
            len(observations[group]) for group in set(observations) & excluded
        ),
    }


def _slot_exclusions(
    slot: Mapping[str, Any],
    sample_to_group: Mapping[str, str],
) -> tuple[set[str], dict[str, Any]]:
    group_ids = {_safe_id(value) for value in slot.get("excluded_nature_individual_ids", [])}
    sample_ids = {_safe_id(value) for value in slot.get("excluded_nature_sample_ids", [])}
    sample_ids.discard("")
    group_ids.discard("")
    unresolved_samples = sorted(sample for sample in sample_ids if sample not in sample_to_group)
    group_ids.update(
        sample_to_group[sample] for sample in sample_ids if sample in sample_to_group
    )
    unknown_group_count = sum(
        1 for group_id in group_ids if group_id not in set(sample_to_group.values())
    )
    group_to_samples: dict[str, list[str]] = defaultdict(list)
    for sample_id, individual_id in sample_to_group.items():
        group_to_samples[individual_id].append(sample_id)
    atomic_check = shared_group_exclusion_status(
        group_to_samples,
        slot.get("excluded_nature_individual_ids", []),
        slot.get("excluded_nature_sample_ids", []),
    )
    return group_ids, {
        "manifest_excluded_individual_key_count": len(slot.get("excluded_nature_individual_ids", [])),
        "manifest_excluded_sample_key_count": len(slot.get("excluded_nature_sample_ids", [])),
        "unmapped_manifest_sample_key_count": len(unresolved_samples),
        "unmapped_manifest_sample_key_set_sha256": sha256_json(unresolved_samples),
        "unmapped_manifest_individual_key_count": unknown_group_count,
        "manifest_group_exclusion_sha256": sha256_json(sorted(group_ids)),
        "shared_identity_helper_status": atomic_check["status"],
        "fold_exclusion_expanded_to_whole_individual_groups": True,
    }


def _fold_row(
    level: str,
    outer: Mapping[str, Any],
    inner: Mapping[str, Any] | None,
    fraction: float,
    signatures: Mapping[str, Any],
    base_group_info: Mapping[str, Any],
    global_quarantine_groups: set[str],
    sample_to_group: Mapping[str, str],
) -> dict[str, Any]:
    source = outer if inner is None else inner
    fold_specific_groups, fold_exclusion = _slot_exclusions(source, sample_to_group)
    # Rebuild to apply every individual_ID exclusion atomically. The passed
    # signatures already omit the global quarantine set, so these fold-only
    # keys are applied by selecting from that safe signature pool.
    fold_excluded_from_safe_pool = fold_specific_groups - global_quarantine_groups
    remaining = {
        group: signature
        for group, signature in signatures.items()
        if group not in fold_excluded_from_safe_pool
    }
    capacity = compatibility_summary(remaining)
    status = "DATA_BLOCKED"
    return {
        "level": level,
        "outer_fold": int(outer["fold"]),
        "inner_fold": None if inner is None else int(inner["inner_fold"]),
        "inner_label_fraction": None if inner is None else fraction,
        "target_outer_test_patient_count": outer.get("outer_test_patient_count"),
        "target_inner_validation_patient_count": None
        if inner is None
        else inner.get("validation_patient_count"),
        "target_manifest_linked_patient_count": source.get("linked_patient_count"),
        "unresolved_target_patient_count": len(source.get("unresolved_patient_ids", [])),
        "target_patient_to_Nature_one_to_one_matching_required": False,
        "Nature_partition": {
            "selected_colorectal_sample_count_before_quarantine": base_group_info[
                "sidecar_sample_count"
            ],
            "individual_group_count_before_quarantine": base_group_info[
                "sidecar_group_count_before_exclusion"
            ],
            "known_overlap_group_count_quarantined_globally": len(
                global_quarantine_groups
            ),
            "known_overlap_groups_present_in_current_colorectal_sidecar": base_group_info[
                "excluded_quarantine_group_count_present_in_sidecar"
            ],
            "target_fold_specific_whole_group_exclusion": fold_exclusion,
            "fold_specific_group_count_removed_after_global_quarantine": len(
                fold_excluded_from_safe_pool & set(signatures)
            ),
            "remaining_group_count_after_known_quarantine_and_fold_exclusions": len(
                remaining
            ),
            "mask_compatibility_and_exchangeability": capacity,
        },
        "capacity_scope": "conditional on the 13 identified Nature individual_ID groups exhausting cross-source overlap",
        "status": status,
        "training_go": False,
        "blocking_reasons": [
            "NATURE_TO_DEPMAP_BRIDGE_OR_ABSENCE_UNRESOLVED",
            "FOLD_MASK_EXECUTION_UNBOUND",
            "SOURCE_CHECKPOINT_EXPOSURE_LINEAGE_UNBOUND",
        ],
        "capacity_is_not_training_or_performance_evidence": True,
    }


def run_audit(output_dir: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    output_dir = output_dir.resolve()
    runtime = require_runtime()
    for path in (
        MASK_SIDECAR,
        MASK_RECEIPT,
        SAMPLES_PARQUET,
        OVERLAP_RECEIPT,
        FOLD_MANIFEST,
        NATURE_TABLE2_ZIP,
        AXIS_CONTRACT,
    ):
        if not path.is_file():
            raise CapacityAuditError(f"required registered input is missing: {path.name}")
    input_hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
        for path in (
            MASK_SIDECAR,
            MASK_RECEIPT,
            SAMPLES_PARQUET,
            OVERLAP_RECEIPT,
            FOLD_MANIFEST,
            NATURE_TABLE2_ZIP,
            AXIS_CONTRACT,
        )
    }
    if input_hashes["data/processed/transfer/biobank_functional_v1/samples.parquet"] != EXPECTED_SAMPLES_SHA256:
        raise CapacityAuditError("registered Nature samples file checksum changed")
    if input_hashes["artifacts/experiments/EXP-045/20260924_gene_axis_audit_v2/axis_contract.json"] == "":
        raise CapacityAuditError("gene-axis contract checksum could not be read")
    overlap_receipt = json.loads(OVERLAP_RECEIPT.read_text(encoding="utf-8"))
    if overlap_receipt.get("status") not in {"PASS", "PASS_WITH_LIMITATIONS", "SOURCE_PROVENANCE_OVERLAP_REVIEW_COMPLETE"}:
        # The current receipt status is checked below by its required result;
        # this branch still permits the registered stage's exact named status.
        if "nature_overlap_result" not in overlap_receipt:
            raise CapacityAuditError("source overlap receipt lacks the approved Nature quarantine")
    fold_manifest = json.loads(FOLD_MANIFEST.read_text(encoding="utf-8"))
    if fold_manifest.get("status") not in {
        "SOURCE_AND_TARGET_IDENTITY_REVIEW_COMPLETE",
        "PASS_WITH_UNRESOLVED_CROSS_SOURCE_BRIDGE",
        "DATA_BLOCKED",
        "REVIEW_COMPLETE",
    }:
        if not isinstance(fold_manifest.get("target_folds"), list):
            raise CapacityAuditError("frozen target fold identity manifest is missing")
    target_folds = fold_manifest.get("target_folds")
    if not isinstance(target_folds, list):
        raise CapacityAuditError("frozen target outer folds are missing")
    inner_selected = selected_inner_folds(target_folds)

    (
        quarantine_group_map,
        quarantine_summary,
        official_identity,
    ) = _load_official_overlap(overlap_receipt)
    global_quarantine_groups = set(quarantine_group_map)
    sample_to_group, sample_summary, registered_sample_keys = _load_sample_identity()
    sidecar = _load_sidecar()

    # Reproduce the approved current-input quarantine from the 162-object
    # registered sample table, then compare it with the official full-table
    # Yes keys. The five broader publisher entries must remain outside this
    # frozen input for the 13-group contract to be applicable as recorded.
    registered_groups_from_table = registered_sample_keys["approved_group_ids"]
    registered_quarantine_samples = registered_sample_keys[
        "expanded_approved_sample_ids"
    ]
    receipt_quarantine_samples = {
        sample_id for values in quarantine_group_map.values() for sample_id in values
    }
    if (
        registered_groups_from_table != global_quarantine_groups
        or registered_quarantine_samples != receipt_quarantine_samples
        or registered_sample_keys["processed_yes_sample_ids"]
        != official_identity["full_official_yes_sample_ids"]
        & registered_sample_keys["all_sample_ids"]
    ):
        raise CapacityAuditError(
            "approved 13-group quarantine does not reproduce from the frozen 162-sample identity fields"
        )
    global_atomic_check = shared_group_exclusion_status(
        quarantine_group_map,
        global_quarantine_groups,
        receipt_quarantine_samples,
    )
    if global_atomic_check["status"] != "PASS_GROUP_ATOMIC":
        raise CapacityAuditError("shared identity helper rejected the 13/14 whole-group quarantine")
    quarantine_summary["shared_identity_helper_group_check"] = global_atomic_check
    full_official_yes = official_identity["full_official_yes_sample_ids"]
    frozen_samples = registered_sample_keys["all_sample_ids"]
    sidecar_sample_keys = set(sidecar["sample_ids"].tolist())
    if len(full_official_yes - frozen_samples) != 5:
        raise CapacityAuditError(
            "the five broader official Yes sample keys are not all outside the frozen 162-sample input"
        )
    if len(full_official_yes & frozen_samples) != 13:
        raise CapacityAuditError("official Yes and frozen sample table intersection is not 13 keys")
    if len(full_official_yes & sidecar_sample_keys) != 4:
        raise CapacityAuditError("official Yes and the selected colorectal sidecar intersection is not 4 keys")
    key_set_relations = {
        "sets": {
            "full_official_table2_Broad_DepMap_yes_sample_IDs": _key_set_summary(
                full_official_yes
            ),
            "registered_samples_parquet_162_sample_IDs": _key_set_summary(
                frozen_samples
            ),
            "selected_colorectal_sidecar_85_sample_IDs": _key_set_summary(
                sidecar_sample_keys
            ),
        },
        "pairwise": [
            _pairwise_key_set_summary(
                "full_official_table2_Broad_DepMap_yes_sample_IDs",
                full_official_yes,
                "registered_samples_parquet_162_sample_IDs",
                frozen_samples,
            ),
            _pairwise_key_set_summary(
                "full_official_table2_Broad_DepMap_yes_sample_IDs",
                full_official_yes,
                "selected_colorectal_sidecar_85_sample_IDs",
                sidecar_sample_keys,
            ),
            _pairwise_key_set_summary(
                "registered_samples_parquet_162_sample_IDs",
                frozen_samples,
                "selected_colorectal_sidecar_85_sample_IDs",
                sidecar_sample_keys,
            ),
        ],
        "triple_intersection": _key_set_summary(
            full_official_yes & frozen_samples & sidecar_sample_keys
        ),
        "official_yes_keys_outside_frozen_input_count": len(
            full_official_yes - frozen_samples
        ),
        "official_yes_keys_outside_frozen_input_set_sha256": sha256_json(
            sorted(full_official_yes - frozen_samples)
        ),
        "interpretation": (
            "The five full-supplement Yes sample_IDs outside the frozen 162-sample "
            "input are not added to the current 13-group quarantine; the approved "
            "13 groups are verified against the current input's own Yes rows."
        ),
    }

    # The processed sample table and publisher table must agree on the official
    # identity/cancer fields for every selected CRC sample. No label field is read.
    side_samples = sidecar["sample_ids"].tolist()
    if set(side_samples) != set(sample_to_group):
        # sample_to_group contains all 85 official local CRC sample keys.
        raise CapacityAuditError(
            "selected Nature sidecar sample keys do not exactly cover registered colorectal sample keys"
        )
    for sample_id in side_samples:
        if sample_id not in official_identity["sample_to_group"]:
            raise CapacityAuditError("selected Nature sample key is absent from publisher Table 2")
        official_group = official_identity["sample_to_group"][sample_id]
        official_tumour = official_identity["sample_to_cancer"][sample_id]
        if (
            official_group != sample_to_group[sample_id]
            or official_tumour.casefold() != "colorectal"
        ):
            raise CapacityAuditError(
                "selected Nature sample/group/cancer identity differs from official Table 2"
            )

    quarantine_sample_ids = {
        sample_id for values in quarantine_group_map.values() for sample_id in values
    }
    sidecar_samples = set(side_samples)
    quarantine_present = quarantine_sample_ids & sidecar_samples
    quarantine_missing = quarantine_sample_ids - sidecar_samples
    observed_quarantine_groups = {
        sample_to_group[sample_id]
        for sample_id in quarantine_present
        if sample_id in sample_to_group
    }
    if observed_quarantine_groups != (global_quarantine_groups & set(sample_to_group.values())):
        # An individual can have a colorectal sample in the local cohort even
        # when its listed Table 2 member is not among the 85 sidecar keys. Group
        # identity is therefore taken from the official IDs, not from sample
        # ID intersection alone; record this case without silently accepting
        # it as proof of no overlap.
        missing_group_keys = (
            global_quarantine_groups & set(sample_to_group.values())
        ) - observed_quarantine_groups
    else:
        missing_group_keys = set()

    all_signatures, group_info = build_group_signatures(
        sample_to_group,
        sidecar_samples,
        sidecar["libraries"].tolist(),
        sidecar["packed_masks"],
        excluded_group_ids=(),
    )
    if group_info["unmapped_sample_count"] != 0:
        raise CapacityAuditError("a sidecar sample key has no registered colorectal individual_ID")
    if set(all_signatures) != set(sample_to_group.values()):
        raise CapacityAuditError("not every colorectal individual group has a complete sidecar mask")
    safe_signatures = {
        group: signature
        for group, signature in all_signatures.items()
        if group not in global_quarantine_groups
    }
    if len(all_signatures) - len(safe_signatures) != len(
        global_quarantine_groups & set(all_signatures)
    ):
        raise CapacityAuditError("whole-individual quarantine projection is inconsistent")
    group_info["excluded_quarantine_group_count_present_in_sidecar"] = len(
        global_quarantine_groups & set(all_signatures)
    )
    group_info["excluded_quarantine_sidecar_sample_count"] = sum(
        1
        for sample_id in side_samples
        if sample_to_group[sample_id] in global_quarantine_groups
    )
    group_info["quarantine_sample_ids_present_in_selected_colorectal_sidecar"] = len(
        quarantine_present
    )
    group_info["quarantine_sample_ids_missing_from_selected_colorectal_sidecar"] = len(
        quarantine_missing
    )
    group_info["quarantine_sample_id_set_sha256"] = sha256_json(
        sorted(quarantine_sample_ids)
    )
    group_info["quarantine_present_sample_id_set_sha256"] = sha256_json(
        sorted(quarantine_present)
    )
    group_info["quarantine_missing_sample_id_set_sha256"] = sha256_json(
        sorted(quarantine_missing)
    )
    group_info["quarantined_groups_with_crc_samples_not_named_in_table2_member_list"] = len(
        missing_group_keys
    )
    group_info["quarantine_is_applied_by_whole_individual_id"] = True
    group_info["projection_only_not_executed_in_training"] = True
    group_info["registered_input_overlap_identity_fields"] = {
        "yes_row_count": sample_summary["registered_input_Broad_DepMap_yes_row_count"],
        "yes_sample_ID_count": sample_summary[
            "registered_input_Broad_DepMap_yes_sample_id_count"
        ],
        "yes_individual_ID_group_count": sample_summary[
            "registered_input_Broad_DepMap_yes_individual_group_count"
        ],
        "all_samples_in_those_groups_count": sample_summary[
            "registered_input_whole_yes_group_sample_id_count"
        ],
        "group_key_set_sha256": sample_summary[
            "registered_input_yes_individual_id_set_sha256"
        ],
        "expanded_sample_key_set_sha256": sample_summary[
            "registered_input_expanded_group_sample_id_set_sha256"
        ],
    }

    slots: list[dict[str, Any]] = []
    for outer in sorted(target_folds, key=lambda row: row["fold"]):
        slots.append(
            _fold_row(
                "outer",
                outer,
                None,
                0.5,
                safe_signatures,
                group_info,
                global_quarantine_groups,
                sample_to_group,
            )
        )
    for outer_id, inner in sorted(inner_selected, key=lambda item: (item[0], item[1]["inner_fold"])):
        outer = next(row for row in target_folds if row["fold"] == outer_id)
        slots.append(
            _fold_row(
                "inner",
                outer,
                inner,
                0.5,
                safe_signatures,
                group_info,
                global_quarantine_groups,
                sample_to_group,
            )
        )
    if len(slots) != 20 or any(slot["status"] != "DATA_BLOCKED" for slot in slots):
        raise CapacityAuditError("expected 20 blocked target outer/50% inner capacity slots")
    completed = datetime.now(timezone.utc)
    runtime["elapsed_seconds"] = round(time.perf_counter() - started_clock, 6)
    fold_counts = Counter(slot["level"] for slot in slots)
    receipt = {
        "receipt_id": "EXP-045-nature-fold-capacity-v1",
        "experiment": "EXP-045",
        "run_id": "20260925_nature_fold_capacity_v1",
        "status": "CAPACITY_AUDIT_COMPLETE_TRAINING_DATA_BLOCKED",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "command": RUN_COMMAND,
        "execution": {
            **runtime,
            "formal_training": False,
            "scientific_performance_comparison": False,
            "dependency_values_read": False,
            "expression_values_read": False,
            "drug_response_labels_read": False,
            "predictions_scores_checkpoints_read": False,
            "gpu_used": False,
            "gpu_hours": 0,
        },
        "inputs": {
            "sha256": input_hashes,
            "validity_sidecar_sha256_verified": EXPECTED_MASK_SHA256,
            "validity_sidecar_key_array_sha256": sidecar["key_array_sha256"],
            "validity_sidecar_mask_array_sha256": sidecar["mask_array_sha256"],
            "frozen_gene_axis_sha256": EXPECTED_AXIS_SHA256,
            "nature_source_data_receipt": {
                "status": sidecar["receipt"]["status"],
                "dependency_source_sha256_from_prior_receipt": EXPECTED_MASK_INPUT_SHA256[
                    "nature_dependency_by_library"
                ],
                "samples_source_sha256": EXPECTED_SAMPLES_SHA256,
                "raw_dependency_table_read_this_run": False,
            },
            "nature_official_overlap_identity": quarantine_summary,
            "nature_sample_metadata": sample_summary,
            "three_source_key_set_relations": key_set_relations,
            "sidecar_mask_summary": {
                "sample_library_object_count": int(len(side_samples)),
                "gene_positions_per_object": int(sidecar["gene_count"]),
                "valid_bit_count": sidecar["valid_bit_count"],
                "missing_or_nonfinite_bit_count": sidecar["missing_bit_count"],
            },
            "fold_manifest_status": fold_manifest.get("status"),
            "fold_manifest_target_outer_count": len(target_folds),
            "fold_manifest_target_inner_fraction": 0.5,
        },
        "nature_partition": group_info,
        "fold_slots": slots,
        "slot_summary": {
            "outer_slot_count": fold_counts.get("outer", 0),
            "inner_50_percent_slot_count": fold_counts.get("inner", 0),
            "pass_count": 0,
            "data_blocked_count": len(slots),
            "training_go": False,
        },
        "identity_and_lineage_limits": [
            "The 13 official overlap individual_ID groups and all 14 Table 2 sample_ID members are conservatively quarantined as whole groups; the registered overlap receipt records 17 sample_ID/library pairs.",
            "Four of those 14 sample_ID keys occur in the selected 85-row colorectal sidecar; ten are absent from this selected subset. This is not evidence of absence from other assets or the wider Nature study.",
            "The target PDO manifest has no recorded Nature patient link for these slots. No one-to-one PDO-to-Nature patient matching is required; this audit does not infer global independence from empty mapping fields.",
            "The post-quarantine group capacity is conditional on the known 13-group quarantine exhausting cross-source overlap. The Nature-to-DepMap object bridge or authoritative absence is still unresolved.",
            "Whole-group exclusion and mask projection are preflight calculations only; this audit does not execute a model mask, establish checkpoint exposure lineage, permit training, or compare scientific performance.",
        ],
        "next_action": (
            "Resolve the Nature/HCMI/DepMap bridge or authoritative absence for the 13 groups, "
            "bind the whole-group exclusions and exact sidecar masks to every frozen outer/inner "
            "adapter input, then rerun the preflight before any training-budget gate."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "receipt.json"
    receipt["output"] = {
        "receipt_path": str(receipt_path.relative_to(ROOT)).replace("\\", "/"),
        "code_sha256": sha256_file(Path(__file__)),
        "code_path": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
    }
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        receipt = run_audit(args.output_dir)
    except Exception as exc:
        print(json.dumps({"status": "CAPACITY_AUDIT_BLOCKED", "error": type(exc).__name__, "reason": str(exc)}))
        return 2
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": receipt["output"]["receipt_path"],
                "outer_slots": receipt["slot_summary"]["outer_slot_count"],
                "inner_50_percent_slots": receipt["slot_summary"]["inner_50_percent_slot_count"],
                "data_blocked_slots": receipt["slot_summary"]["data_blocked_count"],
                "gpu_hours": receipt["execution"]["gpu_hours"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
