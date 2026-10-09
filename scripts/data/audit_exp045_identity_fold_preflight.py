"""Project EXP-045 finite/missing masks onto frozen folds (CPU, no values)."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import os
import platform
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/experiments/EXP-045/20260925_fold_mask_preflight_v1"
ENVIRONMENT = "drugscreening-gpu"
NATURE_DIR = ROOT / "data/processed/transfer/biobank_functional_v1"
NATURE_SAMPLES = NATURE_DIR / "samples.parquet"
NATURE_DEPENDENCY = NATURE_DIR / "dependency_by_library.parquet"
NATURE_TABLE2_ZIP = ROOT / "data/raw/transfer_public_assets/organoid_biobank/41586_2026_10830_MOESM3_ESM.zip"
DEPMAP_MODEL_CSV = ROOT / "data/raw/depmap/24q2/Model.csv"
OVERLAP = ROOT / "artifacts/experiments/EXP-045/20260925_source_provenance_overlap_recovery_v1/receipt.json"
FOLD_MANIFEST = ROOT / "artifacts/experiments/EXP-045/identity_review_20260923_v3b/identity_manifest.json"
SOURCE_BINDING = ROOT / "artifacts/experiments/EXP-045/20260925_sanger_fold_candidate_binding_v1/receipt.json"
AXES = ROOT / "artifacts/experiments/EXP-045/20260924_gene_axis_audit_v2/axis_contract.json"
DRUGS = ROOT / "data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1/manifest.json"
MASK_SIDECAR = ROOT / "artifacts/experiments/EXP-045/20260925_validity_mask_sidecars_v1/availability/dependency_validity_masks.npz"
MASK_RUN_RECEIPT = ROOT / "artifacts/experiments/EXP-045/20260925_validity_mask_sidecars_v1/availability/run_receipt.json"
MASK_SIDECAR_SHA256 = "9eab5d6fcd693061bccd7550efe413098f756a59ee34f39e16dd46078f76ddf3"
MASK_ARRAY_KEYS = {
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
EXPECTED_SOURCE_SHA256 = {
    "depmap_crispr_gene_effect": "d155149181308f8e16b6a4677cde3e5af1ddef68ada1947486ae8c0bcb231452",
    "nature_dependency_by_library": "bf40474b804ab3bb9029c551cd0f7a8604dbdcbead7011ba2e96894f17f20fb7",
    "nature_samples": "2864dc5340d6bca73aea88ece6dfbad533cd100f245152915df1e200773892ac",
}
EXPECTED_AXIS_SHA256 = "9881e9f8184b3586c2211b3554f4fa9e76522055ff4384918f60b27576254965"
EXPECTED_NATURE_ZIP_SHA256 = "7e83b7551c9c272de0bc0455ea9a0e7803a8043d15d9cfedf5a1c3e609a2a6e2"
EXPECTED_NATURE_TABLE2_SHA256 = "5b319152fd3d708ad6be4de62d3655b860cb2ecb08bfb3de9892a25898559cc9"
EXPECTED_DEPMAP_MODEL_SHA256 = "a4cac376131b41aa10b60a075b11c80264bfa860a5509d22cde259c5e85867f8"
NATURE_COLUMNS = ("sample_ID", "individual_ID")
LIBRARY_COLUMNS = ("sample_ID", "library")
SANGER_MODELS = {
    "SIDM00117": {"ACH-000837", "ACH-002172"},
    "SIDM01117": {"ACH-003071"},
    "SIDM00427": set(),
}


class PreflightError(RuntimeError):
    pass


def _hash_obj(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreflightError(f"expected metadata object: {path.name}")
    return value


def _require_runtime() -> dict[str, Any]:
    release = platform.release()
    if sys.platform != "linux" or "microsoft" not in release.casefold():
        raise RuntimeError("run this preflight in WSL2")
    if os.environ.get("CONDA_DEFAULT_ENV") != ENVIRONMENT:
        raise RuntimeError(f"run this preflight in Conda {ENVIRONMENT}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("run this preflight with CUDA_VISIBLE_DEVICES empty")
    return {
        "platform": sys.platform,
        "platform_release": release,
        "python_version": platform.python_version(),
        "conda_environment": ENVIRONMENT,
        "cuda_visible_devices": "",
        "cpu_only": True,
        "gpu_used": False,
        "formal_gpu_hours": 0,
    }


def select_fraction_inner_folds(
    folds: Sequence[Mapping[str, Any]], fraction: float = 0.5, expected: int = 3
) -> list[tuple[int, Mapping[str, Any]]]:
    selected: list[tuple[int, Mapping[str, Any]]] = []
    ids: set[int] = set()
    for fold in folds:
        fold_id = fold.get("fold")
        if not isinstance(fold_id, int) or fold_id in ids:
            raise PreflightError("outer-fold IDs are missing or duplicated")
        ids.add(fold_id)
        inner = fold.get("inner")
        if not isinstance(inner, list):
            raise PreflightError("inner-fold metadata is missing")
        chosen = [item for item in inner if item.get("fraction") == fraction]
        if len(chosen) != expected or {item.get("inner_fold") for item in chosen} != set(range(expected)):
            raise PreflightError("requested label fraction does not contain three registered inner folds")
        selected.extend((fold_id, item) for item in chosen)
    if ids != set(range(5)):
        raise PreflightError("the frozen target manifest must contain five outer folds")
    return selected


def group_exclusion_status(
    individual_to_samples: Mapping[str, Sequence[str]],
    excluded_individuals: Iterable[str],
    excluded_samples: Iterable[str],
) -> dict[str, Any]:
    groups = set(individual_to_samples)
    sample_to_group = {
        sample: individual
        for individual, samples in individual_to_samples.items()
        for sample in samples
    }
    individuals, samples = set(excluded_individuals), set(excluded_samples)
    expected_samples = {
        sample
        for individual in individuals & groups
        for sample in individual_to_samples[individual]
    }
    unknown_groups = individuals - groups
    unknown_samples = samples - set(sample_to_group)
    if unknown_groups or unknown_samples:
        status = "IDENTITY_KEYS_NOT_IN_ALLOWLIST"
    elif not individuals and not samples:
        status = "NO_FOLD_SPECIFIC_EXCLUSION_KEYS"
    elif expected_samples != samples:
        status = "REPEAT_GROUP_EXCLUSION_INCOMPLETE"
    else:
        status = "PASS_GROUP_ATOMIC"
    return {
        "status": status,
        "excluded_group_count": len(individuals),
        "excluded_sample_count": len(samples),
        "expected_full_group_sample_count": len(expected_samples),
        "group_ids_sha256": _hash_obj(sorted(individuals)),
        "sample_ids_sha256": _hash_obj(sorted(samples)),
    }


def _project_parquet(path: Path, columns: Sequence[str], keep_first: set[str] | None = None):
    import pyarrow.parquet as pq

    table = pq.ParquetFile(path)
    if not set(columns).issubset(set(table.schema.names)):
        raise PreflightError(f"required key projection missing from {path.name}")
    rows: set[tuple[str, ...]] = set()
    scanned = 0
    for batch in table.iter_batches(batch_size=65_536, columns=list(columns)):
        fields = [batch.column(i).to_pylist() for i in range(len(columns))]
        for values in zip(*fields):
            scanned += 1
            if any(value is None for value in values):
                continue
            row = tuple(str(value) for value in values)
            if keep_first is None or row[0] in keep_first:
                rows.add(row)
    meta = table.metadata
    return sorted(rows), {
        "selected_columns": list(columns),
        "unique_projected_rows": len(rows),
        "selected_column_rows_scanned": scanned,
        "footer_rows": meta.num_rows,
        "footer_row_groups": meta.num_row_groups,
        "footer_columns": meta.num_columns,
        "full_data_file_hash_computed": False,
    }


def _fold_slot(level: str, outer: int, inner: int | None, counts: Mapping[str, Any], group_status: dict[str, Any]):
    blockers = ["FOLD_MASK_EXECUTION_UNBOUND", "SOURCE_WEIGHT_EXPOSURE_UNKNOWN_UNBOUND"]
    if group_status["status"] != "PASS_GROUP_ATOMIC":
        blockers.append(group_status["status"])
    return {
        "level": level,
        "outer_fold": outer,
        "inner_fold": inner,
        "target_patient_count_from_manifest": counts.get("outer_test_patient_count", counts.get("validation_patient_count")),
        "prior_manifest_linked_patient_count": counts.get("linked_patient_count"),
        "prior_manifest_unresolved_patient_count": len(counts.get("unresolved_patient_ids", [])),
        "nature_group_exclusion": group_status,
        "status": "DATA_BLOCKED",
        "blocking_reasons": blockers,
        "eligible_group_count": None,
        "size_preserving_permutation_count": None,
    }


def _mask_summary(packed: np.ndarray, gene_count: int) -> dict[str, int]:
    """Summarize only keyed validity bits; packed padding bits are ignored."""
    if packed.ndim != 2 or packed.shape[1] != (gene_count + 7) // 8:
        raise PreflightError("packed validity mask width does not match frozen gene axis")
    unpacked = np.unpackbits(packed, axis=1, count=gene_count, bitorder="little")
    per_object = np.count_nonzero(unpacked, axis=1)
    total_bits = int(packed.shape[0] * gene_count)
    valid_bits = int(per_object.sum())
    return {
        "object_count": int(packed.shape[0]),
        "gene_positions_per_object": int(gene_count),
        "valid_bit_count": valid_bits,
        "missing_or_nonfinite_bit_count": total_bits - valid_bits,
        "objects_with_any_valid_bit": int(np.count_nonzero(per_object > 0)),
        "objects_with_all_gene_positions_valid": int(np.count_nonzero(per_object == gene_count)),
    }


def _load_sidecar() -> dict[str, Any]:
    if not MASK_SIDECAR.is_file() or not MASK_RUN_RECEIPT.is_file():
        raise PreflightError("approved validity mask sidecar or run receipt is missing")
    actual_sha = _hash_file(MASK_SIDECAR)
    if actual_sha != MASK_SIDECAR_SHA256:
        raise PreflightError("validity mask sidecar SHA256 differs from approved receipt")
    run_receipt = _read_json(MASK_RUN_RECEIPT)
    if (
        run_receipt.get("status") != "VALIDITY_MASK_SIDECAR_READY"
        or run_receipt.get("sidecar", {}).get("sha256") != MASK_SIDECAR_SHA256
        or run_receipt.get("scope") != "one CPU-only finite/missing mask scan; no dependency values or statistics retained"
    ):
        raise PreflightError("validity mask sidecar run receipt does not match approved scope")
    input_hashes = run_receipt.get("inputs", {})
    for name, expected in EXPECTED_SOURCE_SHA256.items():
        if input_hashes.get(name, {}).get("actual_sha256") != expected or input_hashes.get(name, {}).get("sha256_match") is not True:
            raise PreflightError("validity mask sidecar source binding does not match frozen input SHA256")
    arrays: dict[str, Any] = {}
    try:
        with np.load(MASK_SIDECAR, allow_pickle=False) as archive:
            if set(archive.files) != MASK_ARRAY_KEYS:
                raise PreflightError("validity mask sidecar field set changed")
            arrays = {name: archive[name].copy() for name in archive.files}
    except PreflightError:
        raise
    except Exception as exc:
        raise PreflightError("validity mask sidecar could not be read safely") from exc

    gene_count = int(arrays["axis_gene_count"])
    if gene_count != 16_100 or str(arrays["axis_sha256"]) != EXPECTED_AXIS_SHA256:
        raise PreflightError("validity mask sidecar frozen gene axis changed")
    if int(arrays["schema_version"]) != 1:
        raise PreflightError("validity mask sidecar schema version changed")
    for key, source_name in (
        ("depmap_source_sha256", "depmap_crispr_gene_effect"),
        ("nature_dependency_source_sha256", "nature_dependency_by_library"),
        ("nature_samples_source_sha256", "nature_samples"),
    ):
        if str(arrays[key]) != EXPECTED_SOURCE_SHA256[source_name]:
            raise PreflightError("validity mask sidecar embedded source SHA256 changed")

    depmap_keys = arrays["depmap_object_keys"].astype(str)
    nature_samples = arrays["nature_sample_ids"].astype(str)
    nature_libraries = arrays["nature_library_keys"].astype(str)
    depmap_masks = arrays["depmap_masks_packed"]
    nature_masks = arrays["nature_masks_packed"]
    packed_width = (gene_count + 7) // 8
    if (
        depmap_masks.shape != (1_150, packed_width)
        or nature_masks.shape != (85, packed_width)
        or len(depmap_keys) != len(set(depmap_keys.tolist()))
        or len(nature_samples) != len(set(nature_samples.tolist()))
        or len(nature_libraries) != len(nature_samples)
        or any(not value for value in depmap_keys.tolist())
        or any(not value for value in nature_samples.tolist())
        or any(not value for value in nature_libraries.tolist())
    ):
        raise PreflightError("validity mask key uniqueness or packed shape check failed")

    depmap_summary = _mask_summary(depmap_masks, gene_count)
    nature_summary = _mask_summary(nature_masks, gene_count)
    if (
        depmap_summary["valid_bit_count"] != run_receipt["depmap"]["valid_bit_count"]
        or nature_summary["valid_bit_count"] != run_receipt["nature_colorectal"]["valid_bit_count"]
        or nature_summary["object_count"] != run_receipt["nature_colorectal"]["object_count"]
    ):
        raise PreflightError("validity mask sidecar aggregate bit counts differ from its run receipt")
    array_hashes = run_receipt.get("sidecar", {}).get("validity_array_hashes", {})
    if hashlib.sha256(depmap_masks.tobytes()).hexdigest() != array_hashes.get("depmap_mask_array_sha256"):
        raise PreflightError("DepMap validity bitmap hash differs from run receipt")
    if hashlib.sha256(nature_masks.tobytes()).hexdigest() != array_hashes.get("nature_mask_array_sha256"):
        raise PreflightError("Nature validity bitmap hash differs from run receipt")
    nature_key_hash = hashlib.sha256(
        "\n".join(f"{sample}\t{library}" for sample, library in zip(nature_samples, nature_libraries)).encode("utf-8")
    ).hexdigest()
    if nature_key_hash != array_hashes.get("nature_key_array_sha256"):
        raise PreflightError("Nature sample/library key hash differs from run receipt")
    return {
        "run_receipt": run_receipt,
        "depmap_keys": depmap_keys,
        "depmap_masks": depmap_masks,
        "nature_samples": nature_samples,
        "nature_libraries": nature_libraries,
        "nature_masks": nature_masks,
        "depmap_summary": depmap_summary,
        "nature_summary": nature_summary,
    }


def _mask_summary_for_indices(packed: np.ndarray, indices: Sequence[int], gene_count: int) -> dict[str, int]:
    selected = packed[np.asarray(indices, dtype=np.int64)] if indices else packed[:0]
    return _mask_summary(selected, gene_count)


def _extended_identity_projection(allowed_samples: set[str], depmap_mask_keys: set[str]):
    """Check newly identified registered identity columns by exact key equality only."""
    if _hash_file(NATURE_TABLE2_ZIP) != EXPECTED_NATURE_ZIP_SHA256:
        raise PreflightError("registered Nature supplement ZIP SHA256 changed")
    if _hash_file(DEPMAP_MODEL_CSV) != EXPECTED_DEPMAP_MODEL_SHA256:
        raise PreflightError("registered DepMap Model.csv SHA256 changed")
    member_name = "supplementary_table_2_revision.xlsx"
    try:
        with zipfile.ZipFile(NATURE_TABLE2_ZIP) as archive:
            member_bytes = archive.read(member_name)
    except Exception as exc:
        raise PreflightError("registered Nature identity workbook member is unavailable") from exc
    if hashlib.sha256(member_bytes).hexdigest() != EXPECTED_NATURE_TABLE2_SHA256:
        raise PreflightError("registered Nature identity workbook member SHA256 changed")
    try:
        official = pd.read_excel(
            BytesIO(member_bytes),
            sheet_name="data_availability",
            usecols=["sample_ID", "sanger_ID", "individual_ID"],
            dtype=str,
            keep_default_na=False,
        )
        model_header = pd.read_csv(DEPMAP_MODEL_CSV, nrows=0).columns
        if not {"ModelID", "StrippedCellLineName", "RRID"}.issubset(set(model_header)):
            raise PreflightError("registered DepMap Model.csv identity header is incomplete")
        models = pd.read_csv(
            DEPMAP_MODEL_CSV,
            usecols=["ModelID", "StrippedCellLineName", "RRID"],
            dtype=str,
            keep_default_na=False,
        )
    except PreflightError:
        raise
    except Exception as exc:
        raise PreflightError("registered identity columns could not be projected") from exc
    official["sample_ID"] = official["sample_ID"].astype(str).str.strip()
    official["sanger_ID"] = official["sanger_ID"].astype(str).str.strip()
    official["individual_ID"] = official["individual_ID"].astype(str).str.strip()
    flagged = official.loc[official["sample_ID"].isin(allowed_samples)].copy()
    if (
        len(flagged) != len(allowed_samples)
        or flagged["sample_ID"].duplicated().any()
        or flagged["sanger_ID"].eq("").any()
        or flagged["individual_ID"].eq("").any()
        or models["ModelID"].eq("").any()
        or models["ModelID"].duplicated().any()
    ):
        raise PreflightError("registered identity keys are incomplete or nonunique for the quarantine set")

    official_id_columns = ("sample_ID", "sanger_ID", "individual_ID")
    official_keys_by_type = {
        column: set(flagged[column].astype(str).str.strip().str.casefold())
        for column in official_id_columns
    }
    official_keys = set().union(*official_keys_by_type.values())
    model_ids = models["ModelID"].astype(str).str.strip()
    stripped_names = models["StrippedCellLineName"].astype(str).str.strip().str.casefold()
    rrids = models["RRID"].astype(str).str.strip().str.casefold()
    stripped_matches = stripped_names.isin(official_keys)
    rrid_matches = rrids.isin(official_keys)
    stripped_model_ids = set(model_ids.loc[stripped_matches].tolist())
    rrid_model_ids = set(model_ids.loc[rrid_matches].tolist())
    matched_model_ids = stripped_model_ids | rrid_model_ids

    def match_summary(field_values: pd.Series, match_mask: pd.Series, matched_ids: set[str]) -> dict[str, Any]:
        matched_tokens = set(field_values.loc[match_mask].tolist())
        hit_type_counts = {
            field: len(matched_tokens & tokens)
            for field, tokens in official_keys_by_type.items()
        }
        # Count official tokens that resolve to more than one distinct ModelID.
        model_ids_by_token = {
            token: set(model_ids.loc[field_values.eq(token)].tolist())
            for token in matched_tokens
        }
        return {
            "status": "EXACT_CANDIDATES_FOUND" if matched_ids else "NO_EXACT_MATCH_UNRESOLVED",
            "model_csv_rows_with_exact_full_string_match": int(match_mask.sum()),
            "distinct_candidate_model_ids": len(matched_ids),
            "candidate_model_ids_present_in_crispr_mask_sidecar": len(matched_ids & depmap_mask_keys),
            "candidate_model_ids_absent_from_crispr_mask_sidecar": len(matched_ids - depmap_mask_keys),
            "matched_tokens_by_official_field": hit_type_counts,
            "ambiguous_matched_token_count": sum(len(ids) > 1 for ids in model_ids_by_token.values()),
            "candidate_model_id_set_sha256": _hash_obj(sorted(matched_ids)),
            "status_interpretation": "exact cross-field strings are candidates only; no exact hit does not establish biological absence",
        }

    summary = {
        "source_sha256": {
            "nature_supplement_zip": EXPECTED_NATURE_ZIP_SHA256,
            "nature_table2_member": EXPECTED_NATURE_TABLE2_SHA256,
            "depmap_model_csv": EXPECTED_DEPMAP_MODEL_SHA256,
        },
        "depmap_fields_read": ["ModelID", "StrippedCellLineName", "RRID"],
        "nature_fields_read": ["sample_ID", "sanger_ID", "individual_ID"],
        "comparison_rule": "trim outer whitespace and Unicode casefold, then compare complete strings; no URI, punctuation, or fuzzy normalization",
        "stripped_cell_line_name": {
            **match_summary(stripped_names, stripped_matches, stripped_model_ids),
        },
        "rrid": {
            "depmap_field_present_in_registered_model_csv": True,
            **match_summary(rrids, rrid_matches, rrid_model_ids),
        },
        "combined_candidate_model_id_count": len(matched_model_ids),
        "combined_candidate_model_id_set_sha256": _hash_obj(sorted(matched_model_ids)),
    }
    return summary, {"stripped_cell_line_name": stripped_model_ids, "RRID": rrid_model_ids}


def build_receipt() -> dict[str, Any]:
    paths = {
        "overlap": OVERLAP,
        "fold_manifest": FOLD_MANIFEST,
        "source_binding": SOURCE_BINDING,
        "axis_contract": AXES,
        "drug_manifest": DRUGS,
        "mask_run_receipt": MASK_RUN_RECEIPT,
    }
    docs = {key: _read_json(path) for key, path in paths.items()}
    mask = _load_sidecar()
    gene_count = 16_100

    source_paths = {
        "depmap_crispr_gene_effect": ROOT / "data/raw/depmap/24q2/CRISPRGeneEffect.csv",
        "nature_dependency_by_library": NATURE_DEPENDENCY,
        "nature_samples": NATURE_SAMPLES,
    }
    current_source_hashes = {key: _hash_file(path) for key, path in source_paths.items()}
    if current_source_hashes != EXPECTED_SOURCE_SHA256:
        raise PreflightError("current registered source input SHA256 changed since validity-mask scan")

    overlap = docs["overlap"].get("nature_overlap_result", {})
    groups = overlap.get("individual_to_sample_ID_groups")
    if not isinstance(groups, dict) or len(groups) != 13:
        raise PreflightError("approved Nature allowlist must contain 13 individual groups")
    groups = {str(k): tuple(str(v) for v in values) for k, values in groups.items()}
    allowed_samples = {sample for values in groups.values() for sample in values}
    if len(allowed_samples) != 14 or overlap.get("sample_ID_library_pairs") != 17:
        raise PreflightError("Nature allowlist differs from the approved 13/14/17 counts")

    sample_rows, sample_meta = _project_parquet(
        NATURE_SAMPLES, ("sample_ID", "individual_ID"), allowed_samples
    )
    expected_rows = sorted((sample, individual) for individual, samples in groups.items() for sample in samples)
    if sample_rows != expected_rows:
        raise PreflightError("allowlisted Nature sample/group pairs do not match samples.parquet")
    library_rows, library_meta = _project_parquet(
        NATURE_DEPENDENCY, ("sample_ID", "library"), allowed_samples
    )
    if len(library_rows) != 17 or {sample for sample, _ in library_rows} != allowed_samples:
        raise PreflightError("allowlisted Nature samples do not resolve to 17 unique library pairs")

    nature_key_to_row = {
        (str(sample), str(library)): row
        for row, (sample, library) in enumerate(zip(mask["nature_samples"], mask["nature_libraries"]))
    }
    nature_sample_to_row = {str(sample): row for row, sample in enumerate(mask["nature_samples"])}
    quarantine_samples_present = sorted(allowed_samples & set(nature_sample_to_row))
    quarantine_sample_indices = [nature_sample_to_row[sample] for sample in quarantine_samples_present]
    quarantine_summary = _mask_summary_for_indices(mask["nature_masks"], quarantine_sample_indices, gene_count)
    missing_quarantine_samples = sorted(allowed_samples - set(nature_sample_to_row))
    if len(nature_key_to_row) != len(mask["nature_samples"]):
        raise PreflightError("selected Nature sample/library sidecar keys are not unique")

    depmap_key_to_row = {str(key): row for row, key in enumerate(mask["depmap_keys"])}
    if len(depmap_key_to_row) != len(mask["depmap_keys"]):
        raise PreflightError("DepMap ModelID sidecar keys are not unique")
    extended_identity, extended_identity_candidate_model_ids = _extended_identity_projection(
        allowed_samples, set(depmap_key_to_row)
    )

    def depmap_fold_projection(excluded_model_ids: Iterable[str]) -> dict[str, Any]:
        excluded = {str(model_id) for model_id in excluded_model_ids}
        matched_excluded = excluded & set(depmap_key_to_row)
        train_indices = [
            row for model_id, row in depmap_key_to_row.items()
            if model_id not in excluded
        ]
        excluded_indices = [depmap_key_to_row[model_id] for model_id in matched_excluded]
        return {
            "manifest_excluded_model_id_count": len(excluded),
            "excluded_model_ids_matching_mask_keys": len(matched_excluded),
            "excluded_model_ids_missing_from_mask_keys": len(excluded - matched_excluded),
            "excluded_model_ids_sha256": _hash_obj(sorted(excluded)),
            "remaining_sidecar_model_key_count_before_other_identity_quarantine": len(train_indices),
            "remaining_sidecar_mask_summary_before_other_identity_quarantine": _mask_summary_for_indices(
                mask["depmap_masks"], train_indices, gene_count
            ),
            "manifest_exclusion_mask_summary": _mask_summary_for_indices(
                mask["depmap_masks"], excluded_indices, gene_count
            ),
            "projection_only_not_executed_in_training": True,
        }

    def nature_fold_projection(excluded_samples: Iterable[str]) -> dict[str, Any]:
        excluded = {str(sample) for sample in excluded_samples}
        matched = excluded & set(nature_sample_to_row)
        indices = [nature_sample_to_row[sample] for sample in matched]
        return {
            "manifest_excluded_nature_sample_id_count": len(excluded),
            "excluded_sample_ids_matching_selected_mask_keys": len(matched),
            "excluded_sample_ids_missing_from_selected_mask_keys": len(excluded - matched),
            "excluded_sample_ids_sha256": _hash_obj(sorted(excluded)),
            "manifest_exclusion_mask_summary": _mask_summary_for_indices(
                mask["nature_masks"], indices, gene_count
            ),
            "provisional_cross_source_quarantine_group_count": len(groups),
            "quarantine_samples_present_in_selected_colorectal_mask": len(quarantine_samples_present),
            "quarantine_samples_missing_from_selected_colorectal_mask": len(missing_quarantine_samples),
            "quarantine_mask_summary": quarantine_summary,
            "quarantine_applied_in_fold_manifest": matched == set(quarantine_samples_present),
            "projection_only_not_executed_in_training": True,
        }

    def extended_identity_fold_projection(excluded_model_ids: Iterable[str]) -> dict[str, Any]:
        excluded = {str(model_id) for model_id in excluded_model_ids}
        by_field = {
            field: candidate_ids & set(depmap_key_to_row)
            for field, candidate_ids in extended_identity_candidate_model_ids.items()
        }
        combined = set().union(*by_field.values())
        held_out_by_field = {field: candidate_ids & excluded for field, candidate_ids in by_field.items()}
        remaining_by_field = {field: candidate_ids - excluded for field, candidate_ids in by_field.items()}
        held_out = set().union(*held_out_by_field.values())
        remaining = set().union(*remaining_by_field.values())
        return {
            "exact_candidate_model_count_by_field": {field: len(values) for field, values in by_field.items()},
            "exact_candidate_model_count_combined": len(combined),
            "candidate_model_count_in_frozen_fold_exclusion_by_field": {
                field: len(values) for field, values in held_out_by_field.items()
            },
            "candidate_model_count_outside_frozen_fold_exclusion_by_field": {
                field: len(values) for field, values in remaining_by_field.items()
            },
            "fold_exclusion_status": "CANDIDATE_NOT_EXCLUDED" if remaining else (
                "NO_EXACT_CANDIDATES_UNRESOLVED" if not combined else "CANDIDATES_IN_HOLDOUT_SET_ONLY"
            ),
            "candidate_exclusion_model_ids_sha256": _hash_obj(sorted(held_out)),
            "candidate_remaining_model_ids_sha256": _hash_obj(sorted(remaining)),
            "checkpoint_exposure_resolved": False,
        }

    manifest = docs["fold_manifest"]
    target = manifest.get("target_folds", [])
    source = manifest.get("source_folds", [])
    target_inner = select_fraction_inner_folds(target)
    target_slots = []
    for outer in sorted(target, key=lambda x: x["fold"]):
        check = group_exclusion_status(
            groups,
            outer.get("excluded_nature_individual_ids", []),
            outer.get("excluded_nature_sample_ids", []),
        )
        slot = _fold_slot("outer", outer["fold"], None, outer, check)
        slot["nature_mask_projection"] = nature_fold_projection(outer.get("excluded_nature_sample_ids", []))
        slot["blocking_reasons"].extend([
            "NATURE_CROSS_SOURCE_QUARANTINE_NOT_APPLIED_IN_FOLD_MANIFEST",
            "NATURE_TO_DEPMAP_IDENTITY_BRIDGE_UNRESOLVED",
        ])
        target_slots.append(slot)
    for outer_id, item in sorted(target_inner, key=lambda x: (x[0], x[1]["inner_fold"])):
        outer = next(x for x in target if x["fold"] == outer_id)
        # Use fold-specific inner exclusions when present; otherwise inherit the
        # already registered outer exclusion set without matching other PDO keys.
        counts = dict(item)
        counts.setdefault("excluded_nature_individual_ids", outer.get("excluded_nature_individual_ids", []))
        counts.setdefault("excluded_nature_sample_ids", outer.get("excluded_nature_sample_ids", []))
        check = group_exclusion_status(
            groups,
            counts.get("excluded_nature_individual_ids", []),
            counts.get("excluded_nature_sample_ids", []),
        )
        slot = _fold_slot("inner", outer_id, item["inner_fold"], counts, check)
        slot["nature_mask_projection"] = nature_fold_projection(counts.get("excluded_nature_sample_ids", []))
        slot["blocking_reasons"].extend([
            "NATURE_CROSS_SOURCE_QUARANTINE_NOT_APPLIED_IN_FOLD_MANIFEST",
            "NATURE_TO_DEPMAP_IDENTITY_BRIDGE_UNRESOLVED",
        ])
        target_slots.append(slot)

    # Verify the source candidate binding against the registered split metadata;
    # retain only aggregate counts and hashes, never source IDs.
    bound = docs["source_binding"].get("per_object", [])
    if len(bound) != 3 or len(source) != 5:
        raise PreflightError("source candidate/fold metadata does not match the registered contract")
    by_id = {}
    for obj in bound:
        sid = obj.get("Sanger_ID")
        candidates = set(obj.get("official_CMP", {}).get("Model_csv_exact_candidate_ModelIDs", []))
        if sid not in SANGER_MODELS or sid in by_id or candidates != SANGER_MODELS[sid]:
            raise PreflightError("source candidate mapping differs from the Sol ruling")
        if obj.get("source_checkpoint_training_selection_test_exposure") != "UNKNOWN_UNBOUND":
            raise PreflightError("source exposure status changed; re-review required")
        by_id[sid] = obj

    source_slots = []
    binding_checks = binding_matches = 0
    all_candidates = set().union(*SANGER_MODELS.values())
    for outer in sorted(source, key=lambda x: x["fold"]):
        outer_id = outer["fold"]
        unresolved = set(outer.get("unresolved_sanger_ids", []))
        excluded = set(outer.get("excluded_model_ids", []))
        records = []
        for sid, obj in by_id.items():
            reg = next((x for x in obj.get("fold_identity_registration", []) if x.get("outer_fold") == outer_id), None)
            if reg is None:
                raise PreflightError("source candidate binding omits an outer fold")
            expected = sid in unresolved
            binding_checks += 1
            binding_matches += int(bool(reg.get("outer_registered_member")) == expected)
            records.append({"candidate_count": len(SANGER_MODELS[sid]), "registered_member": expected})
        outer_required_candidates = set().union(*(SANGER_MODELS[sid] for sid in unresolved)) if unresolved else set()
        outer_hits = len(outer_required_candidates & excluded)
        outer_unresolved_affected = sum(
            not SANGER_MODELS[sid] and sid in unresolved for sid in by_id
        )
        outer_identity_projection = extended_identity_fold_projection(excluded)
        outer_blockers = [
            "SOURCE_WEIGHT_EXPOSURE_UNKNOWN_UNBOUND",
            "FOLD_MASK_EXECUTION_UNBOUND",
            "NATURE_TO_DEPMAP_IDENTITY_BRIDGE_UNRESOLVED",
        ]
        if outer_hits != len(outer_required_candidates):
            outer_blockers.append("SOURCE_MODELID_ISOLATION_SET_INCOMPLETE")
        if sum(outer_identity_projection["candidate_model_count_outside_frozen_fold_exclusion_by_field"].values()):
            outer_blockers.append("NATURE_MODELID_ALIAS_CANDIDATE_NOT_IN_FOLD_EXCLUSION_SET")
        if outer_unresolved_affected:
            outer_blockers.append("UNRESOLVED_SOURCE_IDENTITY_AFFECTS_FOLD")
        source_slots.append({
            "level": "outer", "outer_fold": outer_id, "inner_fold": None,
            "source_group_count_from_manifest": outer.get("outer_test_sanger_count"),
            "candidate_model_id_exclusion_hits": outer_hits,
            "candidate_model_id_exclusion_required": len(outer_required_candidates),
            "candidate_model_id_exclusion_required_set_sha256": _hash_obj(sorted(outer_required_candidates)),
            "unresolved_identity_objects_registered_in_slot": outer_unresolved_affected,
            "candidate_identity_object_counts": records,
            "unresolved_sanger_identity_object_count": len(unresolved),
            "extended_nature_identity_projection": outer_identity_projection,
            "source_exposure_status": "UNKNOWN_UNBOUND",
            "fold_bound_mask_receipt_available": False,
            "depmap_mask_projection": depmap_fold_projection(excluded),
            "nature_mask_projection": nature_fold_projection(outer.get("excluded_nature_sample_ids", [])),
            "nature_quarantine_blocked": True,
            "status": "DATA_BLOCKED",
            "blocking_reasons": outer_blockers,
            "eligible_group_count": None,
            "size_preserving_permutation_count": None,
        })
        inner_rows = outer.get("inner", [])
        if len(inner_rows) != 3:
            raise PreflightError("source outer fold does not contain three inner folds")
        for inner in sorted(inner_rows, key=lambda x: x["inner_fold"]):
            unresolved_inner = set(inner.get("unresolved_sanger_ids", []))
            excluded_inner = set(inner.get("excluded_model_ids", []))
            inner_records = []
            for sid, obj in by_id.items():
                outer_reg = next(x for x in obj["fold_identity_registration"] if x.get("outer_fold") == outer_id)
                reg = next(x for x in outer_reg["inner_splits"] if x.get("inner_fold") == inner["inner_fold"])
                expected = sid in unresolved_inner
                binding_checks += 1
                binding_matches += int(bool(reg.get("registered_member")) == expected)
                inner_records.append({"candidate_count": len(SANGER_MODELS[sid]), "registered_member": expected})
            inner_required_candidates = set().union(*(SANGER_MODELS[sid] for sid in unresolved_inner)) if unresolved_inner else set()
            inner_hits = len(inner_required_candidates & excluded_inner)
            inner_unresolved_affected = sum(
                not SANGER_MODELS[sid] and sid in unresolved_inner for sid in by_id
            )
            inner_identity_projection = extended_identity_fold_projection(excluded_inner)
            inner_blockers = [
                "SOURCE_WEIGHT_EXPOSURE_UNKNOWN_UNBOUND",
                "FOLD_MASK_EXECUTION_UNBOUND",
                "NATURE_TO_DEPMAP_IDENTITY_BRIDGE_UNRESOLVED",
            ]
            if inner_hits != len(inner_required_candidates):
                inner_blockers.append("SOURCE_MODELID_ISOLATION_SET_INCOMPLETE")
            if sum(inner_identity_projection["candidate_model_count_outside_frozen_fold_exclusion_by_field"].values()):
                inner_blockers.append("NATURE_MODELID_ALIAS_CANDIDATE_NOT_IN_FOLD_EXCLUSION_SET")
            if inner_unresolved_affected:
                inner_blockers.append("UNRESOLVED_SOURCE_IDENTITY_AFFECTS_FOLD")
            source_slots.append({
                "level": "inner", "outer_fold": outer_id, "inner_fold": inner["inner_fold"],
                "source_group_count_from_manifest": inner.get("validation_sanger_count"),
                "candidate_model_id_exclusion_hits": inner_hits,
                "candidate_model_id_exclusion_required": len(inner_required_candidates),
                "candidate_model_id_exclusion_required_set_sha256": _hash_obj(sorted(inner_required_candidates)),
                "unresolved_identity_objects_registered_in_slot": inner_unresolved_affected,
                "candidate_identity_object_counts": inner_records,
                "unresolved_sanger_identity_object_count": len(unresolved_inner),
                "extended_nature_identity_projection": inner_identity_projection,
                "source_exposure_status": "UNKNOWN_UNBOUND",
                "fold_bound_mask_receipt_available": False,
                "depmap_mask_projection": depmap_fold_projection(excluded_inner),
                "nature_mask_projection": nature_fold_projection(inner.get("excluded_nature_sample_ids", [])),
                "nature_quarantine_blocked": True,
                "status": "DATA_BLOCKED",
                "blocking_reasons": inner_blockers,
                "eligible_group_count": None,
                "size_preserving_permutation_count": None,
            })
    if binding_checks != 60 or binding_matches != binding_checks:
        raise PreflightError("source candidate membership check failed against the frozen fold manifest")

    axis = docs["axis_contract"]
    input_genes = axis.get("feature_input_axis", {}).get("ordered_genes", [])
    output_genes = axis.get("dependency_output_axis", {}).get("ordered_genes", [])
    drug = docs["drug_manifest"]
    if len(input_genes) != 14_857 or len(output_genes) != 16_100 or drug.get("n_drugs") != 14:
        raise PreflightError("frozen gene/drug axis metadata changed")

    target_blocked = sum(x["status"] == "DATA_BLOCKED" for x in target_slots)
    source_blocked = sum(x["status"] == "DATA_BLOCKED" for x in source_slots)
    return {
        "receipt_id": "EXP-045-fold-mask-preflight-v1",
        "status": "DATA_BLOCKED",
        "execution": {"mode": "fold_keyed_validity_mask_preflight_cpu", "formal_training": False, "d1_science_comparison": False, "gpu_used": False, "formal_gpu_hours": 0},
        "read_contract": {
            "Nature samples parquet columns": list(NATURE_COLUMNS),
            "Nature dependency parquet columns": list(LIBRARY_COLUMNS),
            "dependency_magnitudes_or_LFC_values_read": False,
            "DSS/dependency/expression labels, predictions, scores, checkpoints read": False,
            "full PDO-to-Nature patient matching attempted": False,
            "only_finite_missing_bits_and_object_keys_loaded_from_sidecar": True,
        },
        "input_receipt_sha256": {key: _hash_file(path) for key, path in paths.items()},
        "input_sha256": {
            "validity_mask_sidecar": _hash_file(MASK_SIDECAR),
            "current_source_files": current_source_hashes,
            "nature_supplement_zip": extended_identity["source_sha256"]["nature_supplement_zip"],
            "nature_table2_member": extended_identity["source_sha256"]["nature_table2_member"],
            "depmap_model_csv": extended_identity["source_sha256"]["depmap_model_csv"],
            "expected_sidecar_sha256": MASK_SIDECAR_SHA256,
        },
        "nature_identity": {
            "allowlist_groups": len(groups), "allowlist_samples": len(allowed_samples),
            "verified_sample_individual_rows": len(sample_rows), "verified_sample_library_pairs": len(library_rows),
            "group_size_histogram": {str(n): sum(len(x) == n for x in groups.values()) for n in sorted({len(x) for x in groups.values()})},
            "group_membership_sha256": _hash_obj(sorted((k, sorted(v)) for k, v in groups.items())),
            "sample_projection_sha256": _hash_obj(sample_rows), "library_projection_sha256": _hash_obj(library_rows),
            "sample_table_projection": sample_meta, "dependency_table_projection": library_meta,
        },
        "axes": {
            "axis_contract_status": axis.get("status"),
            "input_gene_count": len(input_genes), "input_gene_axis_sha256": _hash_obj(input_genes),
            "dependency_gene_count": len(output_genes), "dependency_gene_axis_sha256": _hash_obj(output_genes),
            "drug_count": drug["n_drugs"], "drug_axis_parquet_sha256": drug["files_sha256"]["drug_axis.parquet"],
            "drug_axis_parquet_opened": False,
        },
        "mask_status": {
            "sidecar_status": "VALIDITY_MASK_SIDECAR_HASH_AND_KEYS_VERIFIED",
            "sidecar_sha256": MASK_SIDECAR_SHA256,
            "axis_gene_count": gene_count,
            "axis_sha256": EXPECTED_AXIS_SHA256,
            "depmap": mask["depmap_summary"],
            "nature_colorectal_selected_sample_library": mask["nature_summary"],
            "nature_library_selection": {
                "selected_minLib_count": mask["run_receipt"]["nature_colorectal"]["selected_minLib_count"],
                "selected_single_library_count": mask["run_receipt"]["nature_colorectal"]["selected_single_library_count"],
            },
            "nature_cross_source_quarantine": {
                "individual_group_count": len(groups),
                "sample_id_count": len(allowed_samples),
                "sample_library_pair_count": len(library_rows),
                "selected_colorectal_sidecar_sample_ids_present": len(quarantine_samples_present),
                "selected_colorectal_sidecar_sample_ids_missing": len(missing_quarantine_samples),
                "selected_colorectal_sidecar_mask_summary": quarantine_summary,
                "quarantined_sample_ids_sha256": _hash_obj(sorted(allowed_samples)),
                "quarantined_group_map_sha256": _hash_obj(sorted((k, sorted(v)) for k, v in groups.items())),
            },
            "extended_identity_key_check": extended_identity,
            "fold_bound_mask_receipts": 0,
            "mask_execution_in_model_adapter_verified": False,
            "eligible_group_count": None,
            "size_preserving_permutation_count": None,
            "null_statistics_or_data_pages_read": False,
        },
        "target_folds": {"slot_count": len(target_slots), "pass_count": 0, "data_blocked_count": target_blocked, "by_fold": target_slots},
        "source_folds": {
            "slot_count": len(source_slots), "pass_count": 0, "data_blocked_count": source_blocked,
            "candidate_model_id_counts": sorted(len(value) for value in SANGER_MODELS.values()),
            "candidate_binding_membership_checks": binding_checks,
            "candidate_binding_membership_matches": binding_matches,
            "by_fold": source_slots,
        },
        "minimum_external_requirements": [
            "Authoritative Nature/HCMI/Broad identity or absence bridge for the 13 quarantined groups and 14 sample_IDs, then apply affected object exclusions to each fold",
            "Resolve or quarantine the three named Sanger source IDs and their biological aliases on the affected source folds",
            "Per-fold checkpoint hash and fit/selection/test lineage bound to object exclusions and the exact executed finite/missing mask",
            "Real adapter input/exclusion evidence, independent functional holdout, and separately authorized candidate/cost/resource gate",
        ],
        "next_action": "Obtain the authoritative Nature overlap bridge or absence evidence and affected Sanger alias/exposure evidence; bind resulting object exclusions and actual mask execution to frozen folds, then rerun this preflight before any formal-training gate.",
    }


def main() -> int:
    started_utc = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument(
        "--command",
        default=(
            "wsl.exe --distribution Ubuntu --cd /mnt/d/Code/DrugScreenLab -- "
            "/bin/bash -lc 'CUDA_VISIBLE_DEVICES= /home/dell/miniconda3/bin/conda run "
            "--no-capture-output -n drugscreening-gpu python "
            "scripts/data/audit_exp045_identity_fold_preflight.py --output-dir "
            "/mnt/d/Code/DrugScreenLab/artifacts/experiments/EXP-045/20260925_fold_mask_preflight_v1'"
        ),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt directory: {args.output_dir}")
    failure: dict[str, Any] | None = None
    try:
        runtime = _require_runtime()
        receipt = build_receipt()
    except PreflightError as exc:
        runtime = {
            "platform": sys.platform,
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_only": True,
            "gpu_used": False,
            "formal_gpu_hours": 0,
        }
        receipt = {
            "receipt_id": "EXP-045-fold-mask-preflight-v1",
            "status": "DATA_BLOCKED",
            "execution": {"mode": "fail_closed_cpu_data_preflight", "formal_training": False, "d1_science_comparison": False},
            "failure": {"type": "PreflightError", "reason": str(exc)},
            "input_sha256": {
                "validity_mask_sidecar": _hash_file(MASK_SIDECAR) if MASK_SIDECAR.is_file() else None,
                "expected_validity_mask_sidecar": MASK_SIDECAR_SHA256,
                "fold_manifest": _hash_file(FOLD_MANIFEST) if FOLD_MANIFEST.is_file() else None,
            },
            "next_action": "Resolve the exact failed data contract, then rerun this fixed CPU preflight into a new output directory.",
        }
        failure = {"type": "PreflightError", "reason": str(exc)}
    except RuntimeError as exc:
        runtime = {
            "platform": sys.platform,
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_only": True,
            "gpu_used": False,
            "formal_gpu_hours": 0,
        }
        receipt = {
            "receipt_id": "EXP-045-fold-mask-preflight-v1",
            "status": "DATA_BLOCKED",
            "execution": {"mode": "runtime_contract_failed_closed", "formal_training": False, "d1_science_comparison": False},
            "failure": {"type": "RuntimeError", "reason": str(exc)},
            "next_action": "Rerun only in WSL2 Conda drugscreening-gpu with CUDA_VISIBLE_DEVICES empty.",
        }
        failure = {"type": "RuntimeError", "reason": str(exc)}
    except Exception as exc:
        runtime = {
            "platform": sys.platform,
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_only": True,
            "gpu_used": False,
            "formal_gpu_hours": 0,
        }
        receipt = {
            "receipt_id": "EXP-045-fold-mask-preflight-v1",
            "status": "DATA_BLOCKED",
            "execution": {"mode": "preflight_error_fail_closed", "formal_training": False, "d1_science_comparison": False},
            "failure": {"type": type(exc).__name__, "reason": "PREFLIGHT_DID_NOT_COMPLETE; no fold pass is claimed"},
            "input_sha256": {
                "validity_mask_sidecar": _hash_file(MASK_SIDECAR) if MASK_SIDECAR.is_file() else None,
                "expected_validity_mask_sidecar": MASK_SIDECAR_SHA256,
                "fold_manifest": _hash_file(FOLD_MANIFEST) if FOLD_MANIFEST.is_file() else None,
            },
            "next_action": "Resolve the recorded preflight implementation or data-contract failure, then rerun into a new output directory.",
        }
        failure = {"type": type(exc).__name__, "reason": "PREFLIGHT_DID_NOT_COMPLETE"}
    completed_at = datetime.now(timezone.utc).isoformat()
    return_code = 2 if receipt.get("status") == "DATA_BLOCKED" else 0
    receipt["execution"] = {
        **receipt.get("execution", {}),
        **runtime,
        "command": args.command,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "elapsed_seconds": round(time.time() - started_utc, 3),
        "exit_status": return_code,
        "preflight_script_sha256": _hash_file(Path(__file__).resolve()),
        "gpu_hours": 0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.output_dir / "receipt.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if "target_folds" in receipt and "source_folds" in receipt:
        print(
            f"status={receipt['status']}; Nature=13 groups/14 samples/17 pairs; "
            f"target={receipt['target_folds']['data_blocked_count']}/{receipt['target_folds']['slot_count']} blocked; "
            f"source={receipt['source_folds']['data_blocked_count']}/{receipt['source_folds']['slot_count']} blocked; "
            f"sha256={_hash_file(path)}"
        )
    else:
        print(f"status={receipt['status']}; failure={failure}; sha256={_hash_file(path)}")
    print(f"receipt={path.relative_to(ROOT).as_posix()}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
