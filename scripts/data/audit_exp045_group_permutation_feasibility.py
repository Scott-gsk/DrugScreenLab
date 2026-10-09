"""Asset-level mask-compatibility preflight for EXP-045 D-SC and D-SO.

This audit reads only the registered DepMap CRISPR effect matrix and its model
metadata, the registered Nature dependency/sample tables, and the frozen
EXP-045 dependency output axis. Dependency values are converted immediately to
finite/missing masks; no label values or row identifiers are written to the
receipt or stdout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_CONDA_ENV = "drugscreening-gpu"
EXPECTED_AXIS_SIZE = 16_100
NATURE_DUAL_LIBRARY_KEEP = "minLib"
NATURE_DUAL_LIBRARY_ALTERNATE = "v1_1"
DEPMAP_INPUT_SHA256 = {
    "CRISPRGeneEffect.csv": "d155149181308f8e16b6a4677cde3e5af1ddef68ada1947486ae8c0bcb231452",
    "Model.csv": "a4cac376131b41aa10b60a075b11c80264bfa860a5509d22cde259c5e85867f8",
    "CRISPRScreenMap.csv": "7dd110f8eceff81e547e6529f5f00fd734239369e7a0e63836b4fdf533b805e0",
    "ScreenSequenceMap.csv": "fabaf0bfb091bb0ca1528a998dc28c91a5baae306231e358686116e1133ef0be",
    "README.txt": "e0538d922dcee4e9f85241b622f0a984cc004fffa37f9fd1f7847d9ce42d323c",
}
RUN_COMMAND = (
    "wsl.exe --distribution Ubuntu --cd /mnt/d/Code/DrugScreenLab -- "
    "/home/dell/miniconda3/bin/conda run --no-capture-output "
    "-n drugscreening-gpu python "
    "scripts/data/audit_exp045_group_permutation_feasibility.py"
)


def _require_supported_runtime() -> dict[str, Any]:
    release = platform.release()
    if sys.platform != "linux" or "microsoft" not in release.casefold():
        raise RuntimeError("audit requires WSL2 Linux")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env != EXPECTED_CONDA_ENV:
        raise RuntimeError("audit requires the registered drugscreening-gpu environment")
    return {
        "platform": sys.platform,
        "platform_release": release,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "conda_environment": conda_env,
        "gpu_used": False,
        "gpu_hours": 0,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _packed_mask(mask: np.ndarray) -> bytes:
    return np.packbits(np.asarray(mask, dtype=np.uint8), bitorder="little").tobytes()


def _compatibility_summary(group_signatures: dict[Any, Any]) -> dict[str, Any]:
    class_sizes = Counter(group_signatures.values())
    histogram = Counter(class_sizes.values())
    exchangeable_groups = sum(size for size in class_sizes.values() if size >= 2)
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
    }


def _depmap_preflight(
    depmap_path: Path,
    model_path: Path,
    screen_map_path: Path,
    screen_sequence_map_path: Path,
    ordered_genes: list[str],
) -> dict[str, Any]:
    with depmap_path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    if len(header) < 2 or header[0] != "":
        raise RuntimeError("registered DepMap matrix key-column schema changed")

    columns_by_symbol: dict[str, list[str]] = defaultdict(list)
    for column in header[1:]:
        columns_by_symbol[column.split(" (", 1)[0]].append(column)
    ordered_columns: list[str] = []
    for gene in ordered_genes:
        candidates = columns_by_symbol.get(gene, [])
        if len(candidates) != 1:
            raise RuntimeError("frozen output axis does not map uniquely to DepMap")
        ordered_columns.append(candidates[0])

    model = pd.read_csv(
        model_path,
        usecols=["ModelID", "PatientID", "OncotreeLineage"],
        dtype=str,
        keep_default_na=False,
    )
    if model["ModelID"].eq("").any() or model["ModelID"].duplicated().any():
        raise RuntimeError("DepMap model metadata key contract is not unique")
    metadata_by_model = {
        row.ModelID: (row.PatientID.strip(), row.OncotreeLineage.strip())
        for row in model.itertuples(index=False)
    }
    del model

    screen_map = pd.read_csv(screen_map_path, dtype=str, keep_default_na=False)
    if set(screen_map.columns) != {"ScreenID", "ModelID"}:
        raise RuntimeError("official ModelID-to-ScreenID mapping schema changed")
    if screen_map[["ScreenID", "ModelID"]].eq("").any().any():
        raise RuntimeError("official ModelID-to-ScreenID mapping contains empty keys")
    duplicate_screen_rows = int(screen_map.duplicated(["ScreenID", "ModelID"]).sum())
    screen_map = screen_map.drop_duplicates(["ScreenID", "ModelID"])
    models_by_screen: dict[str, set[str]] = defaultdict(set)
    screens_by_model: dict[str, set[str]] = defaultdict(set)
    for screen_id, model_id in screen_map[["ScreenID", "ModelID"]].itertuples(index=False, name=None):
        models_by_screen[screen_id].add(model_id)
        screens_by_model[model_id].add(screen_id)

    sequence_map = pd.read_csv(screen_sequence_map_path, dtype=str, keep_default_na=False)
    required_sequence_columns = {
        "ScreenID", "ModelID", "Library", "ScreenType",
        "ExcludeFromCRISPRCombined", "PassesQC",
    }
    if not required_sequence_columns.issubset(sequence_map.columns):
        raise RuntimeError("official screen-sequence library metadata schema changed")
    profiles_by_screen: dict[str, set[tuple[str, str]]] = defaultdict(set)
    sequence_models_by_screen: dict[str, set[str]] = defaultdict(set)
    for row in sequence_map.itertuples(index=False):
        if row.Library.strip() and row.ScreenType.strip():
            profiles_by_screen[row.ScreenID].add((row.Library.strip(), row.ScreenType.strip()))
        if row.ModelID.strip():
            sequence_models_by_screen[row.ScreenID].add(row.ModelID.strip())
    del sequence_map

    screen_profile_conflict_count = 0
    screen_profile_missing_count = 0
    screen_model_mismatch_count = 0
    invalid_profile_models: set[str] = set()
    profiles_by_model: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for screen_id, model_ids in models_by_screen.items():
        profile = profiles_by_screen.get(screen_id, set())
        sequence_model_ids = sequence_models_by_screen.get(screen_id, set())
        if len(profile) != 1:
            if profile:
                screen_profile_conflict_count += 1
            else:
                screen_profile_missing_count += 1
            invalid_profile_models.update(model_ids)
            continue
        if sequence_model_ids and sequence_model_ids != model_ids:
            screen_model_mismatch_count += 1
            invalid_profile_models.update(model_ids)
            continue
        library_screen_type = next(iter(profile))
        for model_id in model_ids:
            profiles_by_model[model_id].append(library_screen_type)
    profile_by_model = {
        model_id: tuple(sorted(profiles))
        for model_id, profiles in profiles_by_model.items()
        if model_id not in invalid_profile_models
    }

    selected_columns = set(ordered_columns)
    selected_columns.add("Unnamed: 0")
    numeric_dtypes = {column: np.float32 for column in ordered_columns}
    observations_by_patient: dict[str, list[tuple[str, str, bytes]]] = defaultdict(list)
    invalid_patient_groups: set[str] = set()
    matrix_model_ids: set[str] = set()
    cancer_strata: set[str] = set()
    seen_model_ids: set[str] = set()
    matrix_row_count = 0
    duplicate_matrix_row_count = 0
    missing_model_metadata_row_count = 0
    missing_patient_id_row_count = 0
    missing_cancer_stratum_row_count = 0
    missing_library_profile_row_count = 0

    for chunk in pd.read_csv(
        depmap_path,
        usecols=lambda column: column in selected_columns,
        index_col=0,
        dtype=numeric_dtypes,
        chunksize=64,
        low_memory=False,
    ):
        chunk = chunk[ordered_columns]
        masks = np.isfinite(chunk.to_numpy(dtype=np.float32, copy=False))
        for row_number, model_id in enumerate(chunk.index.astype(str)):
            matrix_row_count += 1
            matrix_model_ids.add(model_id)
            if model_id in seen_model_ids:
                duplicate_matrix_row_count += 1
                continue
            seen_model_ids.add(model_id)
            model_metadata = metadata_by_model.get(model_id)
            if model_metadata is None:
                missing_model_metadata_row_count += 1
                continue
            patient_id, lineage = model_metadata
            if not patient_id:
                missing_patient_id_row_count += 1
                continue
            if not lineage:
                missing_cancer_stratum_row_count += 1
                invalid_patient_groups.add(patient_id)
                continue
            library_profile = profile_by_model.get(model_id)
            if not library_profile:
                missing_library_profile_row_count += 1
                invalid_patient_groups.add(patient_id)
                continue
            cancer_strata.add(lineage)
            row_stratum = json.dumps(
                {"oncotree_lineage": lineage, "library_screen_profile": library_profile},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            observations_by_patient[patient_id].append(
                (model_id, row_stratum, _packed_mask(masks[row_number]))
            )

    group_signatures: dict[str, Any] = {}
    for patient_id, observations in observations_by_patient.items():
        if patient_id in invalid_patient_groups:
            continue
        # Match the formal permutation ordering: row stratum, then observation ID.
        observations.sort(key=lambda item: (item[1].encode("utf-8"), item[0].encode("utf-8")))
        group_signatures[patient_id] = (
            len(observations),
            tuple((row_stratum, mask) for _, row_stratum, mask in observations),
        )

    profiled_library_counts = Counter(map(len, profile_by_model.values()))
    return {
        "registered_asset": "depmap_public_24q2_crispr_gene_effect_raw_v1",
        "registered_version": "DepMap Public 24Q2, asset v1",
        "matrix_row_count": matrix_row_count,
        "matrix_model_id_count": len(matrix_model_ids),
        "matrix_model_ids_with_patient_metadata_count": len(matrix_model_ids & set(metadata_by_model)),
        "matrix_model_ids_without_model_metadata_count": len(matrix_model_ids - set(metadata_by_model)),
        "matrix_model_ids_without_screen_mapping_count": len(matrix_model_ids - set(screens_by_model)),
        "patient_group_count_in_matrix": len(observations_by_patient),
        "missing_model_metadata_row_count": missing_model_metadata_row_count,
        "missing_patient_id_row_count": missing_patient_id_row_count,
        "missing_cancer_stratum_row_count": missing_cancer_stratum_row_count,
        "missing_library_profile_row_count": missing_library_profile_row_count,
        "duplicate_matrix_row_count": duplicate_matrix_row_count,
        "duplicate_model_screen_mapping_count": duplicate_screen_rows,
        "screen_map_row_count": int(len(screen_map)),
        "screen_map_model_count": len(screens_by_model),
        "screen_map_screen_count": len(models_by_screen),
        "screen_profile_conflict_count": screen_profile_conflict_count,
        "screen_profile_missing_count": screen_profile_missing_count,
        "screen_model_mismatch_count": screen_model_mismatch_count,
        "invalid_patient_group_count": len(invalid_patient_groups),
        "cancer_stratum_count": len(cancer_strata),
        "output_axis_gene_count": len(ordered_genes),
        "library_stratification": "per ModelID, mapped via CRISPRScreenMap to ScreenSequenceMap Library and ScreenType; sorted profile retains repeated screens",
        "profiled_model_count": len(profile_by_model),
        "profiled_model_library_screen_profile_count_distribution": {
            str(size): int(count) for size, count in sorted(profiled_library_counts.items())
        },
        "invalid_group_structure_count": len(invalid_patient_groups),
        "compatibility": _compatibility_summary(group_signatures),
        "scope_limitations": [
            "Model.csv PatientID groups ModelIDs from the same tissue sample; authoritative cross-PatientID aliases and derivatives remain unresolved and need a bounded identity quarantine.",
            "All counts are asset-level and include objects pending fold-safe exclusion; they are not foldwise permutation or training-GO evidence.",
        ],
    }


def _nature_preflight(
    samples_path: Path,
    dependency_path: Path,
    ordered_genes: list[str],
) -> dict[str, Any]:
    samples = pd.read_parquet(
        samples_path,
        columns=["sample_ID", "individual_ID", "primary_tumour_type"],
    )
    if samples["sample_ID"].isna().any() or samples["individual_ID"].isna().any():
        raise RuntimeError("Nature sample/group metadata contains missing keys")
    samples["sample_ID"] = samples["sample_ID"].astype(str)
    samples["individual_ID"] = samples["individual_ID"].astype(str)
    if samples["sample_ID"].duplicated().any():
        raise RuntimeError("Nature sample metadata key is not unique")

    colorectal = samples[
        samples["primary_tumour_type"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .eq("colorectal")
    ].copy()
    expected_sample_count_by_group = colorectal.groupby("individual_ID").size().to_dict()
    colorectal_group_count = len(expected_sample_count_by_group)
    colorectal_sample_ids = set(colorectal["sample_ID"])

    dependency = pd.read_parquet(
        dependency_path,
        columns=["sample_ID", "gene", "library", "LFC"],
    )
    dependency["sample_ID"] = dependency["sample_ID"].astype(str)
    dependency["library"] = dependency["library"].astype(str)
    libraries_by_sample = dependency.groupby("sample_ID", sort=False)["library"].agg(
        lambda values: tuple(pd.unique(values.dropna()))
    )

    selected_library_by_sample: dict[str, str] = {}
    dual_library_minlib_sample_count = 0
    single_library_sample_count = 0
    ambiguous_library_sample_count = 0
    for sample_key, library_values in libraries_by_sample.items():
        libraries = set(library_values)
        if libraries == {NATURE_DUAL_LIBRARY_KEEP, NATURE_DUAL_LIBRARY_ALTERNATE}:
            selected_library_by_sample[sample_key] = NATURE_DUAL_LIBRARY_KEEP
            dual_library_minlib_sample_count += 1
        elif len(libraries) == 1:
            selected_library_by_sample[sample_key] = next(iter(libraries))
            single_library_sample_count += 1
        else:
            ambiguous_library_sample_count += 1

    nature_samples_with_dependency = colorectal_sample_ids & set(libraries_by_sample.index)
    missing_dependency_sample_count = len(colorectal_sample_ids - nature_samples_with_dependency)
    selected_colorectal_libraries = {
        sample_key: library
        for sample_key, library in selected_library_by_sample.items()
        if sample_key in colorectal_sample_ids
    }
    selected_rows = dependency[
        dependency["sample_ID"].isin(selected_colorectal_libraries)
    ].copy()
    expected_libraries = selected_rows["sample_ID"].map(selected_colorectal_libraries)
    selected_rows = selected_rows[selected_rows["library"].eq(expected_libraries)]
    del dependency

    gene_position = {gene: position for position, gene in enumerate(ordered_genes)}
    observation_masks: dict[str, tuple[str, bytes]] = {}
    duplicate_output_gene_sample_keys: set[str] = set()
    complete_output_axis_observation_count = 0
    for (sample_key, library), frame in selected_rows.groupby(
        ["sample_ID", "library"], sort=False
    ):
        positions = frame["gene"].map(gene_position)
        on_axis = positions.notna()
        positions = positions[on_axis].astype(np.int32).to_numpy()
        if len(positions) != len(np.unique(positions)):
            duplicate_output_gene_sample_keys.add(sample_key)
            continue
        finite_mask = np.isfinite(
            pd.to_numeric(frame.loc[on_axis, "LFC"], errors="coerce").to_numpy(
                dtype=np.float64
            )
        )
        mask = np.zeros(len(ordered_genes), dtype=bool)
        mask[positions] = finite_mask
        if len(positions) == len(ordered_genes):
            complete_output_axis_observation_count += 1
        observation_masks[sample_key] = (str(library), _packed_mask(mask))
    del selected_rows

    expected_observation_count_by_group = expected_sample_count_by_group
    individual_by_sample = dict(
        zip(colorectal["sample_ID"], colorectal["individual_ID"])
    )
    invalid_group_keys: set[str] = set()
    observations_by_individual: dict[str, list[tuple[str, str, bytes]]] = defaultdict(list)
    for sample_key, individual_key in individual_by_sample.items():
        if (
            sample_key not in selected_colorectal_libraries
            or sample_key not in observation_masks
            or sample_key in duplicate_output_gene_sample_keys
        ):
            invalid_group_keys.add(individual_key)
            continue
        library, mask = observation_masks[sample_key]
        observations_by_individual[individual_key].append((sample_key, library, mask))

    group_signatures: dict[str, Any] = {}
    for individual_key, observations in observations_by_individual.items():
        if individual_key in invalid_group_keys:
            continue
        if len(observations) != expected_observation_count_by_group[individual_key]:
            invalid_group_keys.add(individual_key)
            continue
        # Sort by the private sample key for deterministic within-group order.
        observations.sort(key=lambda item: item[0])
        group_signatures[individual_key] = (
            len(observations),
            tuple(("Colorectal", library, mask) for _, library, mask in observations),
        )

    return {
        "registered_asset": "transfer_organoid_biobank_paired_dependency_v1",
        "registered_version": "Nature paired dependency, asset v1",
        "source_sample_count": len(samples),
        "colorectal_sample_count": len(colorectal),
        "colorectal_individual_group_count": colorectal_group_count,
        "colorectal_samples_with_dependency_count": len(nature_samples_with_dependency),
        "colorectal_samples_missing_dependency_count": missing_dependency_sample_count,
        "dual_library_minlib_retained_sample_count_all_cohort": dual_library_minlib_sample_count,
        "single_library_sample_count_all_cohort": single_library_sample_count,
        "ambiguous_library_sample_count_all_cohort": ambiguous_library_sample_count,
        "complete_output_axis_observation_count": complete_output_axis_observation_count,
        "duplicate_output_gene_observation_count": len(duplicate_output_gene_sample_keys),
        "invalid_or_incomplete_colorectal_group_count": len(invalid_group_keys),
        "output_axis_gene_count": len(ordered_genes),
        "dual_library_rule": "When both registered minLib and v1_1 libraries exist for a model, retain minLib; otherwise retain the single available library.",
        "compatibility": _compatibility_summary(group_signatures),
        "scope_limitations": [
            "Nature groups use individual_ID as the whole biological-patient group and stable sample-key sorting only in memory.",
            "The currently identified 13 potentially overlapping individual groups and 14 samples do not yet have a frozen safe exclusion set; this preflight does not remove them.",
            "All counts are asset-level and cannot establish foldwise exchangeability, evaluation eligibility, or training GO.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--command",
        default=RUN_COMMAND,
        help="Exact launcher command to preserve in the aggregate receipt.",
    )
    args = parser.parse_args()

    runtime = _require_supported_runtime()
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    depmap_path = root / "data/raw/depmap/24q2/CRISPRGeneEffect.csv"
    depmap_model_path = root / "data/raw/depmap/24q2/Model.csv"
    depmap_screen_map_path = root / "data/raw/depmap/24q2/CRISPRScreenMap.csv"
    depmap_sequence_map_path = root / "data/raw/depmap/24q2/ScreenSequenceMap.csv"
    depmap_readme_path = root / "data/raw/depmap/24q2/README.txt"
    nature_dependency_path = (
        root / "data/processed/transfer/biobank_functional_v1/dependency_by_library.parquet"
    )
    nature_samples_path = (
        root / "data/processed/transfer/biobank_functional_v1/samples.parquet"
    )
    axis_path = root / "artifacts/experiments/EXP-045/20260924_gene_axis_audit_v2/axis_contract.json"
    receipt_path = (
        root
        / "artifacts/experiments/EXP-045/20260925_group_permutation_asset_preflight_v2/receipt.json"
    )
    inputs = [
        depmap_path,
        depmap_model_path,
        depmap_screen_map_path,
        depmap_sequence_map_path,
        depmap_readme_path,
        nature_dependency_path,
        nature_samples_path,
        axis_path,
    ]
    if not all(path.is_file() for path in inputs):
        raise RuntimeError("one or more registered inputs are unavailable")
    for path in (depmap_path, depmap_model_path, depmap_screen_map_path,
                 depmap_sequence_map_path, depmap_readme_path):
        expected = DEPMAP_INPUT_SHA256[path.name]
        if _sha256(path) != expected:
            raise RuntimeError(f"DepMap 24Q2 source file checksum mismatch: {path.name}")

    axis_contract = json.loads(axis_path.read_text(encoding="utf-8"))
    axis_section = axis_contract.get("dependency_output_axis", {})
    ordered_genes = [str(gene) for gene in axis_section.get("ordered_genes", [])]
    if len(ordered_genes) != EXPECTED_AXIS_SIZE or len(set(ordered_genes)) != EXPECTED_AXIS_SIZE:
        raise RuntimeError("frozen EXP-045 dependency output axis is not the registered 16,100-gene axis")

    depmap_result = _depmap_preflight(
        depmap_path, depmap_model_path, depmap_screen_map_path,
        depmap_sequence_map_path, ordered_genes,
    )
    nature_result = _nature_preflight(
        nature_samples_path,
        nature_dependency_path,
        ordered_genes,
    )
    input_receipts = {
        "depmap_crispr_gene_effect": {
            "registered_asset": "depmap_public_24q2_crispr_gene_effect_raw_v1",
            "registered_version": "DepMap Public 24Q2, asset v1",
            "path": str(depmap_path.relative_to(root)).replace("\\", "/"),
            "sha256": _sha256(depmap_path),
        },
        "depmap_model_metadata": {
            "registered_source": "DepMap Public 24Q2",
            "path": str(depmap_model_path.relative_to(root)).replace("\\", "/"),
            "sha256": _sha256(depmap_model_path),
        },
        "depmap_model_screen_mapping": {
            "registered_asset": "depmap_public_24q2_crispr_screen_metadata_raw_v1",
            "registered_version": "DepMap Public 24Q2, asset v1",
            "path": str(depmap_screen_map_path.relative_to(root)).replace("\\", "/"),
            "sha256": _sha256(depmap_screen_map_path),
        },
        "depmap_screen_sequence_library_metadata": {
            "registered_asset": "depmap_public_24q2_crispr_screen_metadata_raw_v1",
            "registered_version": "DepMap Public 24Q2, asset v1",
            "path": str(depmap_sequence_map_path.relative_to(root)).replace("\\", "/"),
            "sha256": _sha256(depmap_sequence_map_path),
        },
        "depmap_release_documentation": {
            "registered_asset": "depmap_public_24q2_crispr_screen_metadata_raw_v1",
            "registered_version": "DepMap Public 24Q2, asset v1",
            "path": str(depmap_readme_path.relative_to(root)).replace("\\", "/"),
            "sha256": _sha256(depmap_readme_path),
        },
        "nature_dependency_by_library": {
            "registered_asset": "transfer_organoid_biobank_paired_dependency_v1",
            "registered_version": "Nature paired dependency, asset v1",
            "path": str(nature_dependency_path.relative_to(root)).replace("\\", "/"),
            "sha256": _sha256(nature_dependency_path),
        },
        "nature_samples": {
            "registered_asset": "transfer_organoid_biobank_paired_dependency_v1",
            "registered_version": "Nature paired dependency, asset v1",
            "path": str(nature_samples_path.relative_to(root)).replace("\\", "/"),
            "sha256": _sha256(nature_samples_path),
        },
    }
    axis_file_sha256 = _sha256(axis_path)
    elapsed = time.perf_counter() - started
    receipt = {
        "schema_version": 2,
        "experiment": "EXP-045",
        "audit": "D-SC_D-SO_asset_level_group_permutation_preflight_with_official_screen_library_mapping",
        "status": "ASSET_CAPACITY_PREFLIGHT_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": input_receipts,
        "frozen_dependency_output_axis": {
            "path": str(axis_path.relative_to(root)).replace("\\", "/"),
            "axis_contract_file_sha256": axis_file_sha256,
            "ordered_axis_sha256": axis_section.get("ordered_axis_sha256"),
            "ordered_gene_count": len(ordered_genes),
        },
        "grouping_rule": {
            "depmap_group_field": "Model.csv:PatientID; all mapped ModelID observations from one tissue sample stay together",
            "depmap_cancer_stratum_field": "Model.csv:OncotreeLineage",
            "depmap_library_screen_profile": "CRISPRGeneEffect row ModelID -> CRISPRScreenMap ScreenID -> ScreenSequenceMap Library and ScreenType; ordered multiset of contributing screen profiles",
            "nature_group_field": "individual_ID",
            "nature_cancer_stratum": "Colorectal",
            "nature_library_rule": "minLib retained over v1_1 when both are present",
            "compatibility_requires": [
                "equal number of observation rows per group",
                "same stable within-group observation order",
            "same cancer stratum and complete contributing library/screen-type profile",
                "exactly equal finite/missing masks at every observation and frozen output-axis gene",
            ],
            "minimum_group_count_for_nonidentity_exchange": 2,
        },
        "depmap": depmap_result,
        "nature": nature_result,
        "run": {
            "command": args.command,
            "exit_code": 0,
            "elapsed_seconds": round(elapsed, 3),
            "environment": runtime,
        },
        "scope_limitations": [
            "Only finite/missing masks were derived from dependency labels; dependency magnitudes and statistics were not emitted or retained.",
            "This is an asset-level capacity preflight, not a fold-specific permutation receipt, scientific performance result, training authorization, or GO decision.",
            "The prior search output exposed identity keys in tool output; those values were not used for this calculation and are not reproduced here.",
            "No model training or GPU execution was performed; no persistent intermediate data were created.",
        ],
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "receipt_path": str(receipt_path.relative_to(root)).replace("\\", "/"),
                "status": receipt["status"],
                "depmap_compatibility": depmap_result["compatibility"],
                "nature_compatibility": nature_result["compatibility"],
                "elapsed_seconds": receipt["run"]["elapsed_seconds"],
                "gpu_hours": 0,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
