"""Bind the current EXP-011 index file bytes to the historical EXP-036 receipt.

This audit hashes the NPZ payload as an opaque byte stream. It does not open the
archive, inspect its members, or read any scientific values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_INDEX = ROOT / "data/processed/exp011/gdsc2_cell_line_indexed.npz"
PRIOR_AUDIT = ROOT / "artifacts/experiments/EXP-036/20260923_pharmaformer_asset_audit/asset_audit.json"
DEFAULT_OUTPUT = ROOT / "artifacts/experiments/EXP-045/20260924_source_index_binding_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_receipt(
    source_path: Path,
    prior_audit_path: Path,
    *,
    root: Path,
    audit_script_path: Path,
) -> dict[str, Any]:
    prior = json.loads(prior_audit_path.read_text(encoding="utf-8"))
    prior_sha = str(prior["source_index_sha256"])
    current_sha = _sha256_file(source_path)
    matches = current_sha == prior_sha
    return {
        "audit_id": "EXP-045-source-index-byte-binding-v1",
        "status": (
            "SOURCE_INDEX_BYTES_MATCH_PRIOR_SOURCE_ONLY_ASSET_RECEIPT"
            if matches
            else "SOURCE_INDEX_BYTES_DIFFER_FROM_PRIOR_SOURCE_ONLY_ASSET_RECEIPT"
        ),
        "scope": {
            "source_index_interpreted_as_opaque_bytes": True,
            "npz_archive_opened": False,
            "npz_members_read": [],
            "labels_or_scientific_values_read": False,
            "training_authorized": False,
        },
        "source_index": {
            "path": str(source_path.resolve()),
            "repository_relative_path": source_path.resolve().relative_to(root.resolve()).as_posix(),
            "byte_size": source_path.stat().st_size,
            "current_byte_sha256": current_sha,
        },
        "historical_receipt": {
            "path": str(prior_audit_path.resolve()),
            "repository_relative_path": prior_audit_path.resolve().relative_to(root.resolve()).as_posix(),
            "receipt_byte_sha256": _sha256_file(prior_audit_path),
            "source_index_sha256": prior_sha,
            "current_source_index_sha256_matches": matches,
            "historical_source_path_or_code_hash_reconstructed": False,
        },
        "audit_implementation": {
            "path": str(audit_script_path.resolve()),
            "repository_relative_path": audit_script_path.resolve().relative_to(root.resolve()).as_posix(),
            "byte_sha256": _sha256_file(audit_script_path),
        },
        "limitation": (
            "A matching whole-file digest binds the current file bytes to the prior receipt's digest, "
            "but does not establish the historical command path or code hash. Current checkpoint payloads "
            "must still be checked against their prior per-fold hashes immediately before any slice."
        ),
    }


def write_receipt(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"EXP-045 source-index binding output exists: {output_dir}")
    receipt = build_receipt(
        SOURCE_INDEX,
        PRIOR_AUDIT,
        root=ROOT,
        audit_script_path=Path(__file__).resolve(),
    )
    output_dir.mkdir(parents=True)
    (output_dir / "source_index_binding.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt = write_receipt(args.output_dir)
    print(
        "status={}; current_sha256={}; matches_prior={}; training_authorized={}".format(
            receipt["status"],
            receipt["source_index"]["current_byte_sha256"],
            receipt["historical_receipt"]["current_source_index_sha256_matches"],
            receipt["scope"]["training_authorized"],
        )
    )


if __name__ == "__main__":
    main()
