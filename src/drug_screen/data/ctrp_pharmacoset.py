"""Read and audit the registered CTRPv2 PharmacoSet RDS source."""

from __future__ import annotations

import gzip
import hashlib
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

EXPECTED_SHA256 = "8587d711265261ee23498c3d18efe74f149134e38fd27dc2047b4259a54ccb28"
EXPECTED_BYTES = 43_118_179
SLOT_MARKERS = ("PharmacoSet", "molecularProfiles", "ctrp.cells", "ctrp.drugs", "sensitivityInfo", "sensitivityRaw", "sensitivityProfiles", "sensitivityN", "sensitivity")
FIELD_MARKERS = ("cellid", "smiles", "cpd_smiles")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_gzip_markers(path: Path, markers: tuple[str, ...]) -> list[str]:
    """Find complete ASCII identifiers in a gzip RDS; never parse its contents."""
    found: set[str] = set()
    for marker in markers:
        pattern = re.compile(rb"(?<![A-Za-z0-9_.])" + re.escape(marker.encode("ascii")) + rb"(?![A-Za-z0-9_.])")
        with gzip.open(path, "rb") as handle:
            pending = b""
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                data = pending + block
                if pattern.search(data):
                    found.add(marker)
                    break
                pending = data[-256:]
    return [marker for marker in markers if marker in found]


def _frame_summary(frame: Any) -> dict[str, Any]:
    return {"rows": int(frame.shape[0]), "columns": [str(column) for column in frame.columns]}


def _top_level_frames(value: Any) -> dict[str, Any]:
    import pandas as pd
    if not isinstance(value, SimpleNamespace):
        return {}
    return {name: item for name, item in vars(value).items() if isinstance(item, pd.DataFrame)}


def _rdata_diagnostic(path: Path) -> tuple[Any | None, dict[str, Any]]:
    """Materialize S4 attributes through the installed pure-Python reader."""
    try:
        import rdata
        value = rdata.read_rds(path)
    except Exception as exc:
        return None, {"status": "unsupported", "reader": "rdata.read_rds", "error_type": type(exc).__name__, "message": str(exc)}
    frames = _top_level_frames(value)
    return value, {
        "status": "parsed", "reader": "rdata.read_rds", "object_type": type(value).__name__,
        "s4_slots": sorted(vars(value)) if isinstance(value, SimpleNamespace) else [],
        "decoded_top_level_frames": {name: _frame_summary(frame) for name, frame in frames.items()},
    }


def _find_column(frames: dict[str, Any], column: str) -> list[str]:
    return sorted(name for name, frame in frames.items() if column in frame.columns)


def materialize_ctrp_pharmacoset(path: Path, output_dir: Path) -> dict[str, Any]:
    """Export only decoded top-level data frames; never infer mappings."""
    value, _ = _rdata_diagnostic(path)
    if value is None:
        return {"status": "not_created", "reason": "rdata could not materialize the S4 object."}
    frames = _top_level_frames(value)
    output_dir.mkdir(parents=True, exist_ok=True)
    exports: dict[str, Any] = {}
    for name, frame in frames.items():
        destination = output_dir / f"{name}.csv"
        frame.to_csv(destination, index=False)
        exports[name] = {"path": str(destination), **_frame_summary(frame)}
    return {"status": "created", "format": "csv", "tables": exports}


def audit_ctrp_pharmacoset(path: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """Audit CTRPv2 and optionally export materialized tables."""
    actual_bytes, actual_sha256 = path.stat().st_size, sha256_file(path)
    slots, fields = find_gzip_markers(path, SLOT_MARKERS), find_gzip_markers(path, FIELD_MARKERS)
    value, parser = _rdata_diagnostic(path)
    frames = _top_level_frames(value) if value is not None else {}
    exports = {"status": "not_created", "reason": "No reader materialized a complete S4 object, so no processed tables were created."}
    endpoint_columns = sorted({str(column) for frame in frames.values() for column in frame.columns if str(column).lower() in {"auc", "aac", "ic50", "ec50", "emax"}})
    return {
        "asset": "ctrp_v2_pharmacoset_zenodo_3905470_v1", "raw_path": str(path), "raw_immutable": True,
        "bytes": actual_bytes, "sha256": actual_sha256,
        "registry_checksum_matches": actual_bytes == EXPECTED_BYTES and actual_sha256 == EXPECTED_SHA256,
        "container": "gzip-compressed R serialized object", "embedded_s4_class_marker": "PharmacoSet" in slots,
        "embedded_slot_markers": slots, "embedded_field_markers": fields, "parser": parser,
        "response_endpoint": {"decoded_endpoint_columns": endpoint_columns, "decoded_in_tables": {column: _find_column(frames, column) for column in endpoint_columns}, "direction": "not_inferred", "reason": "The decoded table schema alone does not encode endpoint direction."},
        "drug_structure": {"decoded_in_tables": {field: _find_column(frames, field) for field in ("smiles", "cpd_smiles")}, "status": "materialized" if any(_find_column(frames, field) for field in ("smiles", "cpd_smiles")) else "not_materialized"},
        "cell_identity": {"decoded_in_tables": {"cellid": _find_column(frames, "cellid")}, "status": "materialized" if _find_column(frames, "cellid") else "not_materialized"},
        "processed_export": exports,
    }
