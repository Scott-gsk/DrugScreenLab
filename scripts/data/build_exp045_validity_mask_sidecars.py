"""Create the one-time, value-blind EXP-045 dependency validity sidecar.

Only object keys and packed finite/missing bits are retained. Dependency values
are read in memory only to determine ``isfinite`` and are never logged, saved,
summarized, or passed to a model. This is an input-preflight artifact, not a
training authorization.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


EXPECTED_CONDA_ENV = "drugscreening-gpu"
EXPECTED_AXIS_SIZE = 16_100
EXPECTED_AXIS_SHA256 = "9881e9f8184b3586c2211b3554f4fa9e76522055ff4384918f60b27576254965"
EXPECTED_SOURCE_SHA256 = {
    "depmap_crispr_gene_effect": "d155149181308f8e16b6a4677cde3e5af1ddef68ada1947486ae8c0bcb231452",
    "nature_dependency_by_library": "bf40474b804ab3bb9029c551cd0f7a8604dbdcbead7011ba2e96894f17f20fb7",
    "nature_samples": "2864dc5340d6bca73aea88ece6dfbad533cd100f245152915df1e200773892ac",
}
AXIS_CONTRACT_RELATIVE = Path(
    "artifacts/experiments/EXP-045/20260924_gene_axis_audit_v2/axis_contract.json"
)
OUTPUT_RELATIVE = Path(
    "artifacts/experiments/EXP-045/20260925_validity_mask_sidecars_v1/availability"
)
SIDECAR_NAME = "dependency_validity_masks.npz"
RECEIPT_NAME = "run_receipt.json"
PARTIAL_NAME = ".dependency_validity_masks.partial.npz"
NATURE_KEEP_LIBRARY = "minLib"
NATURE_ALTERNATE_LIBRARY = "v1_1"
RUN_COMMAND = (
    "wsl.exe --distribution Ubuntu --cd /mnt/d/Code/DrugScreenLab -- "
    "/bin/bash -lc 'CUDA_VISIBLE_DEVICES= /home/dell/miniconda3/bin/conda run "
    "--no-capture-output -n drugscreening-gpu python "
    "scripts/data/build_exp045_validity_mask_sidecars.py'"
)


class ContractError(RuntimeError):
    """A fixed-message, value-safe fail-closed error."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _require_supported_runtime() -> dict[str, Any]:
    release = platform.release()
    if sys.platform != "linux" or "microsoft" not in release.casefold():
        raise ContractError("RUNTIME_NOT_WSL2_LINUX")
    if os.environ.get("CONDA_DEFAULT_ENV") != EXPECTED_CONDA_ENV:
        raise ContractError("RUNTIME_NOT_REGISTERED_CONDA_ENV")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ContractError("RUNTIME_GPU_VISIBILITY_NOT_DISABLED")
    return {
        "platform": sys.platform,
        "platform_release": release,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": "cpu",
        "gpu_used": False,
        "gpu_hours": 0,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _axis_sha256(axis: list[str]) -> str:
    return hashlib.sha256(("\n".join(axis) + "\n").encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _verify_inputs(root: Path) -> tuple[dict[str, Path], list[str], dict[str, str]]:
    paths = {
        "depmap_crispr_gene_effect": root / "data/raw/depmap/24q2/CRISPRGeneEffect.csv",
        "nature_dependency_by_library": root / "data/processed/transfer/biobank_functional_v1/dependency_by_library.parquet",
        "nature_samples": root / "data/processed/transfer/biobank_functional_v1/samples.parquet",
    }
    axis_path = root / AXIS_CONTRACT_RELATIVE
    if any(not path.is_file() for path in paths.values()) or not axis_path.is_file():
        raise ContractError("REGISTERED_INPUT_MISSING")

    source_hashes: dict[str, str] = {}
    for name, path in paths.items():
        current = _sha256_file(path)
        if current != EXPECTED_SOURCE_SHA256[name]:
            raise ContractError("REGISTERED_SOURCE_SHA256_MISMATCH")
        source_hashes[name] = current

    try:
        contract = json.loads(axis_path.read_text(encoding="utf-8"))
        section = contract["dependency_output_axis"]
        axis = [str(gene).strip() for gene in section["ordered_genes"]]
        recorded_sha = str(section["ordered_axis_sha256"])
    except Exception as exc:
        raise ContractError("FROZEN_AXIS_CONTRACT_INVALID") from exc
    if (
        len(axis) != EXPECTED_AXIS_SIZE
        or any(not gene for gene in axis)
        or len(set(axis)) != EXPECTED_AXIS_SIZE
        or recorded_sha != EXPECTED_AXIS_SHA256
        or _axis_sha256(axis) != EXPECTED_AXIS_SHA256
    ):
        raise ContractError("FROZEN_AXIS_HASH_OR_COVERAGE_MISMATCH")
    source_hashes["frozen_dependency_output_axis"] = EXPECTED_AXIS_SHA256
    source_hashes["axis_contract_file"] = _sha256_file(axis_path)
    return paths, axis, source_hashes


def _depmap_symbol(column: str) -> str:
    normalized = str(column).strip()
    if " (" not in normalized or not normalized.endswith(")"):
        raise ContractError("DEPMAP_FEATURE_HEADER_SCHEMA_INVALID")
    symbol, entrez = normalized.rsplit(" (", 1)
    if not symbol or not entrez[:-1].isdigit():
        raise ContractError("DEPMAP_FEATURE_HEADER_SCHEMA_INVALID")
    return symbol


def _scan_depmap(path: Path, axis: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream))
    if len(header) < 2 or header[0] != "" or len(header) != len(set(header)):
        raise ContractError("DEPMAP_MATRIX_KEY_OR_HEADER_NOT_UNIQUE")

    by_symbol: dict[str, list[str]] = {}
    for column in header[1:]:
        by_symbol.setdefault(_depmap_symbol(column), []).append(column)
    selected_columns: list[str] = []
    for gene in axis:
        candidates = by_symbol.get(gene, [])
        if len(candidates) != 1:
            raise ContractError("DEPMAP_FROZEN_AXIS_MAPPING_NOT_UNIQUE")
        selected_columns.append(candidates[0])

    numeric_dtypes = {column: np.float32 for column in selected_columns}
    selected_set = set(selected_columns)
    object_keys: list[str] = []
    packed_masks: list[bytes] = []
    seen_keys: set[str] = set()
    valid_bits = 0
    row_count = 0
    try:
        chunks = pd.read_csv(
            path,
            usecols=lambda column: column == "Unnamed: 0" or column in selected_set,
            index_col=0,
            dtype=numeric_dtypes,
            chunksize=32,
            low_memory=False,
            encoding="utf-8-sig",
        )
        for chunk in chunks:
            chunk = chunk.loc[:, selected_columns]
            keys = [str(key).strip() for key in chunk.index]
            if any(not key for key in keys) or len(keys) != len(set(keys)):
                raise ContractError("DEPMAP_OBJECT_KEY_EMPTY_OR_DUPLICATE_WITHIN_CHUNK")
            if seen_keys.intersection(keys):
                raise ContractError("DEPMAP_OBJECT_KEY_DUPLICATE_ACROSS_CHUNKS")
            seen_keys.update(keys)

            matrix = chunk.to_numpy(dtype=np.float32, copy=False)
            finite = np.isfinite(matrix)
            if finite.shape != (len(keys), len(axis)):
                raise ContractError("DEPMAP_MASK_SHAPE_MISMATCH")
            for row_index, key in enumerate(keys):
                row_mask = finite[row_index]
                packed_masks.append(np.packbits(row_mask, bitorder="little"))
                object_keys.append(key)
                valid_bits += int(np.count_nonzero(row_mask))
            row_count += len(keys)
            del matrix, finite, chunk
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError("DEPMAP_MATRIX_PARSE_OR_MASK_FAILURE") from exc

    if row_count == 0 or row_count != len(object_keys) or len(seen_keys) != row_count:
        raise ContractError("DEPMAP_OBJECT_COVERAGE_INVALID")
    packed_width = (len(axis) + 7) // 8
    if any(len(mask) != packed_width for mask in packed_masks):
        raise ContractError("DEPMAP_PACKED_MASK_WIDTH_INVALID")

    ordering = np.argsort(np.asarray(object_keys, dtype=np.str_), kind="stable")
    sorted_keys = np.asarray(object_keys, dtype=np.str_)[ordering]
    if len(np.unique(sorted_keys)) != len(sorted_keys):
        raise ContractError("DEPMAP_OBJECT_KEY_DUPLICATE_AFTER_SORT")
    sorted_masks = np.stack([packed_masks[int(index)] for index in ordering]).astype(
        np.uint8, copy=False
    )
    total_bits = row_count * len(axis)
    return sorted_keys, sorted_masks, {
        "object_count": row_count,
        "axis_gene_count": len(axis),
        "axis_positions_covered_per_object": len(axis),
        "valid_bit_count": valid_bits,
        "missing_or_nonfinite_bit_count": total_bits - valid_bits,
        "total_object_gene_bits": total_bits,
        "duplicate_object_key_count": 0,
    }


def _nature_colorectal_samples(path: Path) -> set[str]:
    try:
        samples = pq.read_table(path, columns=["sample_ID", "primary_tumour_type"]).to_pandas()
    except Exception as exc:
        raise ContractError("NATURE_SAMPLE_METADATA_PARSE_FAILURE") from exc
    if samples["sample_ID"].isna().any() or samples["primary_tumour_type"].isna().any():
        raise ContractError("NATURE_SAMPLE_METADATA_KEY_OR_STRATUM_MISSING")
    sample_ids = samples["sample_ID"].astype(str).str.strip()
    if sample_ids.eq("").any() or sample_ids.duplicated().any():
        raise ContractError("NATURE_SAMPLE_KEY_EMPTY_OR_DUPLICATE")
    is_colorectal = (
        samples["primary_tumour_type"].astype(str).str.strip().str.casefold().eq("colorectal")
    )
    selected = set(sample_ids.loc[is_colorectal].tolist())
    if not selected:
        raise ContractError("NATURE_COLORECTAL_TARGET_COHORT_EMPTY")
    del samples, sample_ids, is_colorectal
    return selected


def _scan_nature(
    dependency_path: Path,
    samples_path: Path,
    axis: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    colorectal_samples = _nature_colorectal_samples(samples_path)
    gene_position = {gene: index for index, gene in enumerate(axis)}
    try:
        dataset = ds.dataset(dependency_path, format="parquet")
        required = {"sample_ID", "gene", "library", "LFC"}
        if not required.issubset(set(dataset.schema.names)):
            raise ContractError("NATURE_DEPENDENCY_SCHEMA_MISSING_REQUIRED_FIELDS")
        lfc_type = dataset.schema.field("LFC").type
        if not (pa.types.is_integer(lfc_type) or pa.types.is_floating(lfc_type)):
            raise ContractError("NATURE_DEPENDENCY_VALUE_TYPE_NOT_NUMERIC")
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError("NATURE_DEPENDENCY_SCHEMA_READ_FAILURE") from exc

    library_by_sample: dict[str, set[str]] = {sample: set() for sample in colorectal_samples}
    seen_by_pair: dict[tuple[str, str], np.ndarray] = {}
    mask_by_pair: dict[tuple[str, str], np.ndarray] = {}
    target_axis_row_count = 0

    try:
        target_filter = ds.field("sample_ID").isin(sorted(colorectal_samples))
        key_scanner = dataset.scanner(
            columns=["sample_ID", "library"],
            filter=target_filter,
            batch_size=65_536,
        )
        for batch in key_scanner.to_batches():
            sample_values = batch.column(0).to_pylist()
            library_values = batch.column(1).to_pylist()
            for raw_sample, raw_library in zip(sample_values, library_values):
                if raw_sample is None or raw_library is None:
                    raise ContractError("NATURE_TARGET_OBJECT_OR_LIBRARY_KEY_MISSING")
                sample = str(raw_sample).strip()
                library = str(raw_library).strip()
                if sample not in colorectal_samples or not library:
                    raise ContractError("NATURE_TARGET_OBJECT_OR_LIBRARY_KEY_INVALID")
                library_by_sample[sample].add(library)
            del sample_values, library_values, batch

        if any(not libraries for libraries in library_by_sample.values()):
            raise ContractError("NATURE_COLORECTAL_SAMPLE_HAS_NO_DEPENDENCY_LIBRARY")

        axis_filter = target_filter & ds.field("gene").isin(axis)
        mask_scanner = dataset.scanner(
            columns=["sample_ID", "gene", "library", "LFC"],
            filter=axis_filter,
            batch_size=65_536,
        )
        for batch in mask_scanner.to_batches():
            sample_values = batch.column(0).to_pylist()
            gene_values = batch.column(1).to_pylist()
            library_values = batch.column(2).to_pylist()
            finite_array = pc.fill_null(pc.is_finite(batch.column(3)), False)
            finite_values = finite_array.to_numpy(zero_copy_only=False)
            for raw_sample, raw_gene, raw_library, is_finite in zip(
                sample_values, gene_values, library_values, finite_values
            ):
                if raw_sample is None or raw_gene is None or raw_library is None:
                    raise ContractError("NATURE_TARGET_OBJECT_GENE_OR_LIBRARY_KEY_MISSING")
                sample = str(raw_sample).strip()
                gene = str(raw_gene).strip()
                library = str(raw_library).strip()
                if (
                    sample not in colorectal_samples
                    or library not in library_by_sample[sample]
                    or gene not in gene_position
                ):
                    raise ContractError("NATURE_TARGET_AXIS_KEY_MAPPING_INVALID")
                position = gene_position[gene]
                pair = (sample, library)
                if pair not in seen_by_pair:
                    seen_by_pair[pair] = np.zeros(len(axis), dtype=bool)
                    mask_by_pair[pair] = np.zeros(len(axis), dtype=bool)
                if seen_by_pair[pair][position]:
                    raise ContractError("NATURE_SAMPLE_LIBRARY_GENE_KEY_DUPLICATE")
                seen_by_pair[pair][position] = True
                mask_by_pair[pair][position] = bool(is_finite)
                target_axis_row_count += 1
            del sample_values, gene_values, library_values, finite_array, finite_values, batch
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError("NATURE_DEPENDENCY_PARSE_OR_MASK_FAILURE") from exc

    selected_rows: list[tuple[str, str, np.ndarray]] = []
    minlib_count = 0
    single_library_count = 0
    for sample in sorted(colorectal_samples):
        libraries = library_by_sample.get(sample, set())
        if not libraries:
            raise ContractError("NATURE_COLORECTAL_SAMPLE_HAS_NO_DEPENDENCY_LIBRARY")
        if libraries == {NATURE_KEEP_LIBRARY, NATURE_ALTERNATE_LIBRARY}:
            selected_library = NATURE_KEEP_LIBRARY
            minlib_count += 1
        elif len(libraries) == 1:
            selected_library = next(iter(libraries))
            single_library_count += 1
        else:
            raise ContractError("NATURE_LIBRARY_SELECTION_AMBIGUOUS")

        for library in libraries:
            all_library_seen = seen_by_pair.get((sample, library))
            if all_library_seen is None or not bool(np.all(all_library_seen)):
                raise ContractError("NATURE_LIBRARY_FROZEN_AXIS_COVERAGE_INCOMPLETE")
        pair = (sample, selected_library)
        seen = seen_by_pair.get(pair)
        mask = mask_by_pair.get(pair)
        if seen is None or mask is None or not bool(np.all(seen)):
            raise ContractError("NATURE_FROZEN_AXIS_COVERAGE_INCOMPLETE")
        selected_rows.append((sample, selected_library, mask.copy()))

    if len(selected_rows) != len(colorectal_samples):
        raise ContractError("NATURE_COLORECTAL_SAMPLE_COVERAGE_MISMATCH")

    object_keys = np.asarray([row[0] for row in selected_rows], dtype=np.str_)
    library_keys = np.asarray([row[1] for row in selected_rows], dtype=np.str_)
    packed_masks = np.stack(
        [np.packbits(row[2], bitorder="little") for row in selected_rows]
    ).astype(np.uint8, copy=False)
    total_valid = int(sum(np.count_nonzero(row[2]) for row in selected_rows))
    total_bits = len(selected_rows) * len(axis)
    if len(np.unique(object_keys)) != len(object_keys):
        raise ContractError("NATURE_OUTPUT_OBJECT_KEY_DUPLICATE")
    return object_keys, library_keys, packed_masks, {
        "object_count": len(selected_rows),
        "axis_gene_count": len(axis),
        "axis_positions_covered_per_object": len(axis),
        "valid_bit_count": total_valid,
        "missing_or_nonfinite_bit_count": total_bits - total_valid,
        "total_object_gene_bits": total_bits,
        "duplicate_sample_library_gene_key_count": 0,
        "selected_minLib_count": minlib_count,
        "selected_single_library_count": single_library_count,
        "target_axis_rows_scanned": target_axis_row_count,
        "all_colorectal_sample_keys_covered": True,
    }


def _verify_written_sidecar(
    path: Path,
    axis_count: int,
    axis_sha256: str,
    source_hashes: dict[str, str],
    depmap_stats: dict[str, int],
    nature_stats: dict[str, Any],
) -> dict[str, str]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            expected_fields = {
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
            if set(payload.files) != expected_fields:
                raise ContractError("SIDECAR_FIELD_SET_MISMATCH")
            if int(payload["axis_gene_count"]) != axis_count:
                raise ContractError("SIDECAR_AXIS_SIZE_MISMATCH")
            if str(payload["axis_sha256"]) != axis_sha256:
                raise ContractError("SIDECAR_AXIS_SHA256_MISMATCH")
            if str(payload["depmap_source_sha256"]) != source_hashes["depmap_crispr_gene_effect"]:
                raise ContractError("SIDECAR_DEPMAP_SOURCE_BINDING_MISMATCH")
            if str(payload["nature_dependency_source_sha256"]) != source_hashes["nature_dependency_by_library"]:
                raise ContractError("SIDECAR_NATURE_DEPENDENCY_BINDING_MISMATCH")
            if str(payload["nature_samples_source_sha256"]) != source_hashes["nature_samples"]:
                raise ContractError("SIDECAR_NATURE_SAMPLES_BINDING_MISMATCH")
            depmap_keys = payload["depmap_object_keys"]
            depmap_masks = payload["depmap_masks_packed"]
            nature_ids = payload["nature_sample_ids"]
            nature_libraries = payload["nature_library_keys"]
            nature_masks = payload["nature_masks_packed"]
            packed_width = (axis_count + 7) // 8
            if (
                depmap_masks.shape != (depmap_stats["object_count"], packed_width)
                or nature_masks.shape != (nature_stats["object_count"], packed_width)
                or len(depmap_keys) != depmap_stats["object_count"]
                or len(nature_ids) != nature_stats["object_count"]
                or len(nature_libraries) != len(nature_ids)
                or len(np.unique(depmap_keys)) != len(depmap_keys)
                or len(np.unique(nature_ids)) != len(nature_ids)
            ):
                raise ContractError("SIDECAR_KEY_OR_PACKED_SHAPE_INVALID")
            depmap_unpacked = np.unpackbits(depmap_masks, axis=1, count=axis_count, bitorder="little")
            nature_unpacked = np.unpackbits(nature_masks, axis=1, count=axis_count, bitorder="little")
            if (
                int(np.count_nonzero(depmap_unpacked)) != depmap_stats["valid_bit_count"]
                or int(np.count_nonzero(nature_unpacked)) != nature_stats["valid_bit_count"]
                or depmap_stats["valid_bit_count"] + depmap_stats["missing_or_nonfinite_bit_count"]
                != depmap_stats["total_object_gene_bits"]
                or nature_stats["valid_bit_count"] + nature_stats["missing_or_nonfinite_bit_count"]
                != nature_stats["total_object_gene_bits"]
            ):
                raise ContractError("SIDECAR_VALID_BIT_COUNT_MISMATCH")
            del depmap_unpacked, nature_unpacked
            return {
                "depmap_mask_array_sha256": hashlib.sha256(depmap_masks.tobytes()).hexdigest(),
                "nature_mask_array_sha256": hashlib.sha256(nature_masks.tobytes()).hexdigest(),
                "nature_key_array_sha256": hashlib.sha256(
                    "\n".join(
                        f"{sample}\t{library}"
                        for sample, library in zip(nature_ids.tolist(), nature_libraries.tolist())
                    ).encode("utf-8")
                ).hexdigest(),
            }
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError("SIDECAR_REOPEN_OR_VALIDATION_FAILURE") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", default=RUN_COMMAND)
    args = parser.parse_args()

    started_wall = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    root = Path(__file__).resolve().parents[2]
    output_dir = root / OUTPUT_RELATIVE
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / RECEIPT_NAME
    partial_path = output_dir / PARTIAL_NAME
    sidecar_path = output_dir / SIDECAR_NAME
    if any(output_dir.iterdir()):
        print(json.dumps({"status": "FAILED_CLOSED", "failure_code": "OUTPUT_DIRECTORY_NOT_EMPTY"}))
        return 2

    stage = "runtime"
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "EXP-045",
        "run_id": "20260925_validity_mask_sidecars_v1",
        "status": "IN_PROGRESS",
        "started_at_utc": started_utc,
        "command": args.command,
        "output_directory": str(OUTPUT_RELATIVE).replace("\\", "/"),
        "scope": "one CPU-only finite/missing mask scan; no dependency values or statistics retained",
    }
    _write_json_atomic(receipt_path, receipt)
    try:
        runtime = _require_supported_runtime()
        stage = "input_hash_and_axis_contract"
        input_paths, axis, source_hashes = _verify_inputs(root)
        axis_sha = _axis_sha256(axis)

        stage = "depmap_mask_scan"
        depmap_keys, depmap_masks, depmap_stats = _scan_depmap(
            input_paths["depmap_crispr_gene_effect"], axis
        )

        stage = "nature_mask_scan"
        nature_ids, nature_libraries, nature_masks, nature_stats = _scan_nature(
            input_paths["nature_dependency_by_library"],
            input_paths["nature_samples"],
            axis,
        )

        stage = "write_and_verify_sidecar"
        code_sha256 = _sha256_file(Path(__file__).resolve())
        np.savez_compressed(
            partial_path,
            schema_version=np.asarray(1, dtype=np.int16),
            axis_gene_count=np.asarray(len(axis), dtype=np.int32),
            axis_sha256=np.asarray(axis_sha),
            depmap_source_sha256=np.asarray(source_hashes["depmap_crispr_gene_effect"]),
            nature_dependency_source_sha256=np.asarray(source_hashes["nature_dependency_by_library"]),
            nature_samples_source_sha256=np.asarray(source_hashes["nature_samples"]),
            depmap_object_keys=depmap_keys,
            depmap_masks_packed=depmap_masks,
            nature_sample_ids=nature_ids,
            nature_library_keys=nature_libraries,
            nature_masks_packed=nature_masks,
        )
        sidecar_sha = _sha256_file(partial_path)
        mask_array_hashes = _verify_written_sidecar(
            partial_path,
            len(axis),
            axis_sha,
            source_hashes,
            depmap_stats,
            nature_stats,
        )
        partial_path.replace(sidecar_path)
        sidecar_size = sidecar_path.stat().st_size

        elapsed = round(time.perf_counter() - started_wall, 3)
        receipt.update(
            {
                "status": "VALIDITY_MASK_SIDECAR_READY",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "inputs": {
                    name: {
                        "path": str(path.relative_to(root)).replace("\\", "/"),
                        "expected_sha256": EXPECTED_SOURCE_SHA256[name],
                        "actual_sha256": source_hashes[name],
                        "sha256_match": True,
                    }
                    for name, path in input_paths.items()
                },
                "frozen_axis": {
                    "path": str(AXIS_CONTRACT_RELATIVE).replace("\\", "/"),
                    "gene_count": len(axis),
                    "ordered_axis_sha256_expected": EXPECTED_AXIS_SHA256,
                    "ordered_axis_sha256_actual": axis_sha,
                    "axis_hash_match": True,
                    "axis_contract_file_sha256": source_hashes["axis_contract_file"],
                    "complete_axis_coverage_per_object": True,
                },
                "depmap": depmap_stats,
                "nature_colorectal": nature_stats,
                "sidecar": {
                    "path": str((OUTPUT_RELATIVE / SIDECAR_NAME)).replace("\\", "/"),
                    "sha256": sidecar_sha,
                    "byte_size": sidecar_size,
                    "container": "compressed NumPy NPZ; no object pickle; masks packed little-bit-order per frozen gene position",
                    "validity_array_hashes": mask_array_hashes,
                    "contains_dependency_magnitudes": False,
                    "contains_gene_symbols": False,
                },
                "run": {
                    "stage": "complete",
                    "elapsed_seconds": elapsed,
                    "runtime": runtime,
                    "code_sha256": code_sha256,
                    "numpy_version": np.__version__,
                    "pandas_version": pd.__version__,
                    "pyarrow_version": pa.__version__,
                    "training_performed": False,
                    "loss_computed": False,
                    "gpu_used": False,
                    "gpu_hours": 0,
                },
                "limitations": [
                    "This sidecar is not automatically connected to training.",
                    "Identity exclusions, fold-level mask lineage, fairness, independent holdout, exact weights, costs, and formal GO remain separate gates.",
                    "Finite/missing bits describe availability only and do not summarize dependency values.",
                ],
            }
        )
        _write_json_atomic(receipt_path, receipt)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "receipt_path": str(receipt_path.relative_to(root)).replace("\\", "/"),
                    "sidecar_path": str(sidecar_path.relative_to(root)).replace("\\", "/"),
                    "sidecar_sha256": sidecar_sha,
                    "depmap_objects": depmap_stats["object_count"],
                    "depmap_valid_bits": depmap_stats["valid_bit_count"],
                    "depmap_missing_bits": depmap_stats["missing_or_nonfinite_bit_count"],
                    "nature_colorectal_samples": nature_stats["object_count"],
                    "nature_valid_bits": nature_stats["valid_bit_count"],
                    "nature_missing_bits": nature_stats["missing_or_nonfinite_bit_count"],
                    "elapsed_seconds": elapsed,
                    "gpu_hours": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        try:
            partial_path.unlink(missing_ok=True)
        except Exception:
            pass
        failure_code = exc.code if isinstance(exc, ContractError) else f"UNEXPECTED_{type(exc).__name__}"
        receipt.update(
            {
                "status": "FAILED_CLOSED",
                "failure_stage": stage,
                "failure_code": failure_code,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.perf_counter() - started_wall, 3),
                "run": {
                    "training_performed": False,
                    "loss_computed": False,
                    "gpu_used": False,
                    "gpu_hours": 0,
                },
                "exception_message_saved": False,
            }
        )
        _write_json_atomic(receipt_path, receipt)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "failure_stage": stage,
                    "failure_code": failure_code,
                    "receipt_path": str(receipt_path.relative_to(root)).replace("\\", "/"),
                    "elapsed_seconds": receipt["elapsed_seconds"],
                    "gpu_hours": 0,
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
