#!/usr/bin/env python3
"""Read-only ChemPert RDS schema audit and compact metadata materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import rdata


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frame_schema(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "column_names": [str(x) for x in frame.columns],
        "index_unique": bool(frame.index.is_unique),
        "missing_by_column": {str(c): int(frame[c].isna().sum()) for c in frame.columns},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    root = args.root.resolve()
    raw = root / "raw/resource_acquisition_20260916/improve_chempert"
    out = root / "processed/resource_acquisition_20260916/chempert"
    audit_dir = root / "artifacts/resource_acquisition_20260916/improve_chempert"
    out.mkdir(parents=True, exist_ok=True)

    files = {
        "responses": raw / "chempert_transcriptional_responses.rds",
        "targets": raw / "chempert_targets_of_perturbagens.rds",
        "expression": raw / "chempert_gene_expression_profiles.rds",
    }
    schemas = {}
    compact_outputs = {}
    join_checks = {}
    parser_limitations = {}
    loaded = {}

    # The response table converts safely with rdata. The two reference objects
    # contain very large/ragged character vectors for which rdata 1.0.0 may try
    # to allocate a pathological fixed-width NumPy Unicode array. Their public
    # CSV sidecars are therefore the authoritative compact metadata here; the
    # original RDS files remain immutable and checksummed.
    response = rdata.read_rds(files["responses"])
    if not isinstance(response, pd.DataFrame):
        raise TypeError(f"{files['responses']} decoded to {type(response)!r}, expected DataFrame")
    loaded["responses"] = response
    schemas["responses"] = frame_schema(response)

    for key in ("targets", "expression"):
        path = files[key]
        try:
            parsed = rdata.parser.parse_file(path)
            top = parsed.object
            schemas[key] = {
                "conversion_status": "parser_tree_readable_full_dataframe_conversion_not_attempted",
                "top_level_r_type": str(top.info.type),
                "top_level_vector_length": len(top.value) if hasattr(top.value, "__len__") else None,
                "has_attributes": bool(top.info.attributes),
            }
            parser_limitations[key] = (
                "Python rdata conversion previously attempted a pathological fixed-width Unicode "
                "allocation. This audit uses the official CSV sidecar for compact metadata and "
                "does not treat the RDS as corrupt."
            )
        except Exception as exc:
            schemas[key] = {"conversion_status": "parser_tree_failed"}
            parser_limitations[key] = f"{type(exc).__name__}: {str(exc)[:500]}"

    # Do not emit another identity table: the official CSV sidecars already
    # provide these fields, and retaining a derived copy would be redundant.

    sidecars = {
        "responses": (raw / "chempert_information_for_transcriptional_responses.csv", ","),
        "expression": (raw / "chempert_information_for_gene_expression.csv", ","),
        # This upstream file uses @ as its delimiter despite the .csv suffix.
        "perturbagens": (raw / "chempert_information_for_perturbagens.csv", "@"),
    }
    sidecar_schemas = {}
    sidecar_frames = {}
    for key, (path, delimiter) in sidecars.items():
        frame = pd.read_csv(path, dtype="string", sep=delimiter)
        sidecar_frames[key] = frame
        sidecar_schemas[key] = frame_schema(frame)

    response_meta_path = sidecars["responses"][0]
    response_meta = sidecar_frames["responses"]
    if "Response ID" in loaded["responses"].columns and "Response_ID" in response_meta.columns:
        rds_ids = loaded["responses"]["Response ID"].astype("string")
        csv_ids = response_meta["Response_ID"].astype("string")
        join_checks["response_id"] = {
            "rds_unique": bool(rds_ids.is_unique),
            "csv_unique": bool(csv_ids.is_unique),
            "rds_ids": int(rds_ids.nunique(dropna=True)),
            "csv_ids": int(csv_ids.nunique(dropna=True)),
            "intersection": int(len(set(rds_ids.dropna()) & set(csv_ids.dropna()))),
            "rds_only_count": int(len(set(rds_ids.dropna()) - set(csv_ids.dropna()))),
            "csv_only_count": int(len(set(csv_ids.dropna()) - set(rds_ids.dropna()))),
            "rds_only_ids": sorted(set(rds_ids.dropna()) - set(csv_ids.dropna())),
            "csv_only_ids": sorted(set(csv_ids.dropna()) - set(rds_ids.dropna())),
        }

    source_col = next((c for c in response_meta.columns if "accession" in c.lower()), None)
    source_counts = {}
    if source_col:
        source_counts = response_meta[source_col].value_counts(dropna=False).head(30).astype(int).to_dict()

    report = {
        "audit_version": 1,
        "purpose": "schema_and_metadata_materialization_only_no_training_or_evaluation",
        "source_files": {
            k: {"path": str(v.relative_to(root)), "bytes": v.stat().st_size, "sha256": sha256(v)}
            for k, v in files.items()
        },
        "decoded_frames": schemas,
        "official_csv_sidecars": {
            k: {
                "path": str(sidecars[k][0].relative_to(root)),
                "delimiter": sidecars[k][1],
                "bytes": sidecars[k][0].stat().st_size,
                "sha256": sha256(sidecars[k][0]),
                "schema": sidecar_schemas[k],
            }
            for k in sidecars
        },
        "parser_limitations": parser_limitations,
        "compact_outputs": compact_outputs,
        "join_checks": join_checks,
        "response_metadata_top_source_counts": source_counts,
        "data_boundary": "Post-perturbation transcriptional response and target/reference data; no viability, AUC, IC50, or PDO response labels.",
        "admission": "PERTURBATION_REFERENCE_PARTIALLY_SCHEMA_AUDITED_PENDING_LICENSE_AND_MODALITY_CONTRACT",
    }
    dest = audit_dir / "chempert_schema_audit.json"
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()
