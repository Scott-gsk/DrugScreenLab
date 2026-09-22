"""Materialize auditable tables from the locally registered CTRPv2 PharmacoSet RDS.

This command is deliberately local-only: it verifies the immutable source, reads it
with the installed ``rdata`` reader, inventories the decoded S4 slots, and exports
only data frames that are explicitly selected by their decoded paths.  It neither
downloads data nor assigns a response direction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import pandas as pd
import rdata


EXPECTED_BYTES = 43_118_179
EXPECTED_SHA256 = "8587d711265261ee23498c3d18efe74f149134e38fd27dc2047b4259a54ccb28"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def walk(value: Any, path: str = "root", seen: set[int] | None = None) -> Iterator[tuple[str, Any]]:
    """Yield decoded objects without expanding numeric matrices into their cells."""
    seen = set() if seen is None else seen
    if isinstance(value, (pd.DataFrame, np.ndarray, SimpleNamespace, dict, list, tuple)):
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
    yield path, value
    if isinstance(value, SimpleNamespace):
        for key, item in sorted(vars(value).items()):
            yield from walk(item, f"{path}.{key}", seen)
    elif isinstance(value, dict):
        for key, item in sorted(value.items(), key=lambda item: str(item[0])):
            yield from walk(item, f"{path}[{key!r}]", seen)
    elif isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            yield from walk(item, f"{path}[{index}]", seen)


def describe(value: Any) -> dict[str, Any]:
    if isinstance(value, pd.DataFrame):
        return {
            "kind": "data_frame",
            "rows": int(value.shape[0]),
            "columns": [str(column) for column in value.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in value.dtypes.items()},
            "missing_by_column": {str(column): int(value[column].isna().sum()) for column in value.columns},
        }
    if isinstance(value, np.ndarray):
        return {"kind": "ndarray", "shape": [int(part) for part in value.shape], "dtype": str(value.dtype)}
    if isinstance(value, SimpleNamespace):
        return {"kind": "namespace", "attributes": sorted(vars(value))}
    if isinstance(value, dict):
        return {"kind": "dict", "keys": [str(key) for key in sorted(value, key=str)]}
    if isinstance(value, (list, tuple)):
        return {"kind": type(value).__name__, "length": len(value)}
    return {"kind": type(value).__name__, "value": _json_scalar(value)}


def export_frame(frame: pd.DataFrame, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "rows": int(frame.shape[0]),
        "columns": [str(column) for column in frame.columns],
        "missing_by_column": {str(column): int(frame[column].isna().sum()) for column in frame.columns},
    }


def _index_summary(frame: pd.DataFrame) -> dict[str, Any]:
    index = frame.index
    return {
        "name": _json_scalar(index.name),
        "dtype": str(index.dtype),
        "is_unique": bool(index.is_unique),
        "first_values": [_json_scalar(item) for item in index[:3]],
    }


def build_response_table(info: pd.DataFrame, profiles: pd.DataFrame) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Join sensitivity metadata and metrics only when their R row IDs agree."""
    if len(info) != len(profiles):
        return None, {"status": "not_joined", "reason": "sensitivity info and profile row counts differ"}
    info_ids = pd.Index(info["experimentIds"].astype("string"))
    profile_ids = pd.Index(profiles.index.astype("string"))
    if not profile_ids.equals(info_ids):
        return None, {
            "status": "not_joined",
            "reason": "sensitivity profile row IDs do not exactly match sensitivity info experimentIds",
            "info_index": _index_summary(info),
            "profiles_index": _index_summary(profiles),
        }
    duplicate_columns = set(info.columns) & set(profiles.columns)
    if duplicate_columns:
        return None, {"status": "not_joined", "reason": f"duplicate response columns: {sorted(duplicate_columns)}"}
    joined = pd.concat([info.reset_index(drop=True), profiles.reset_index(drop=True)], axis=1)
    return joined, {
        "status": "joined_by_exact_experiment_id",
        "rows": int(len(joined)),
        "info_index": _index_summary(info),
        "profiles_index": _index_summary(profiles),
        "experiment_ids_unique": bool(info_ids.is_unique),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rds", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    raw_path = args.rds.resolve()
    raw_sha256 = sha256_file(raw_path)
    raw_bytes = raw_path.stat().st_size
    source_ok = raw_bytes == EXPECTED_BYTES and raw_sha256 == EXPECTED_SHA256
    if not source_ok:
        raise RuntimeError(f"immutable CTRPv2 source verification failed: bytes={raw_bytes}, sha256={raw_sha256}")

    value = rdata.read_rds(raw_path)
    nodes = {path: item for path, item in walk(value)}
    frames = {path: item for path, item in nodes.items() if isinstance(item, pd.DataFrame)}
    structure = {path: describe(item) for path, item in nodes.items()}
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    (args.audit.parent / "ctrp_v2_structure.json").write_text(
        json.dumps(structure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    exports: dict[str, Any] = {}
    selected = {
        "root.cell": "cells.csv",
        "root.drug": "drugs.csv",
        "root.curation['cell']": "cell_id_map.csv",
        "root.curation['drug']": "drug_id_map.csv",
        "root.curation['tissue']": "tissue_id_map.csv",
        "root.sensitivity['info']": "sensitivity_info.csv",
        "root.sensitivity['profiles']": "sensitivity_profiles.csv",
    }
    for path, filename in selected.items():
        if path in frames:
            exports[path] = export_frame(frames[path], args.processed_dir / filename)
    info, profiles = frames.get("root.sensitivity['info']"), frames.get("root.sensitivity['profiles']")
    joined, alignment = (None, {"status": "not_joined", "reason": "one or both sensitivity tables were not decoded"})
    if info is not None and profiles is not None:
        joined, alignment = build_response_table(info, profiles)
    if joined is not None:
        exports["response_table"] = export_frame(joined, args.processed_dir / "responses.csv")

    endpoint_columns = [] if profiles is None else [str(column) for column in profiles.columns]
    audit = {
        "asset": "ctrp_v2_pharmacoset_zenodo_3905470_v1",
        "raw_path": str(raw_path),
        "raw_immutable": True,
        "bytes": raw_bytes,
        "sha256": raw_sha256,
        "registry_checksum_matches": source_ok,
        "parser": {"status": "parsed", "reader": "rdata.read_rds", "object_type": type(value).__name__},
        "decoded_data_frames": {path: describe(frame) for path, frame in frames.items()},
        "response": {
            "metadata_path": "root.sensitivity['info']",
            "profiles_path": "root.sensitivity['profiles']",
            "endpoint_columns": endpoint_columns,
            "row_alignment": alignment,
            "recommended_metric_for_later_contract": {
                "name": "aac_published",
                "higher_is_more_sensitive": True,
                "evidence": "PharmacoGx computeAUC documentation defines the response area as 1 - viability and states that high AUC implies high drug sensitivity; this applies to the AAC-labelled profile column.",
                "source": "https://bioconductor.org/packages/devel/bioc/manuals/PharmacoGx/man/PharmacoGx.pdf",
                "status": "metric_semantics_resolved_not_selected_for_training_or_evaluation",
            },
            "other_endpoint_directions": "not_assigned; use a single predeclared metric rather than mixing AAC, EC50, IC50, HS, or E_inf.",
        },
        "exports": exports,
        "unexported_data_frames": sorted(set(frames) - set(exports)),
        "baseline_expression": {
            "status": "absent_in_this_rds",
            "decoded_slot": "root.molecularProfiles['rna'].assays.data.listData['exprs']",
            "decoded_shape": [0, 0],
            "reason": "The PharmacoSet contains an RNA slot but its expression assay and its cell/gene annotations are empty, so this asset alone cannot provide baseline expression rows.",
        },
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    materialization_registry = {
        "id": "ctrp_v2_pharmacoset_zenodo_3905470_materialized_20260916",
        "source_asset": audit["asset"],
        "raw": {"path": str(raw_path), "bytes": raw_bytes, "sha256": raw_sha256, "immutable": True},
        "materialization": {
            "status": "parsed_with_local_rdata",
            "generator": str(Path(__file__).resolve()),
            "generator_sha256": sha256_file(Path(__file__).resolve()),
            "audit": str(args.audit),
            "structure": str(args.audit.parent / "ctrp_v2_structure.json"),
            "exports": exports,
        },
        "admission": {
            "training": "not_authorized",
            "evaluation": "not_authorized",
            "reason": "This materialization establishes local table availability only. Cross-study pairing, expression provenance, split contract, and experiment approval remain separate gates.",
        },
    }
    (args.audit.parent / "ctrp_v2_materialization_registry.json").write_text(
        json.dumps(materialization_registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"audit": str(args.audit), "frames": len(frames), "exports": list(exports), "response_alignment": alignment["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
