"""Recompute EXP-044 cell/drug eligibility using keys only, without responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


IDENTITY_STATUS = "RESOLVED_ORTHOGONAL_AUTHORITATIVE_CROSSWALK"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_provenance() -> dict[str, Any]:
    return {
        "script_path": Path(__file__).resolve().as_posix(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "command_argv": list(sys.argv),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
    }


def _validated_ids(values: pd.Series, label: str) -> list[str]:
    if values.isna().any():
        raise ValueError(f"{label} contains missing identities")
    result = values.astype(str).str.strip().tolist()
    if any(not value for value in result):
        raise ValueError(f"{label} contains blank identities")
    return result


def _canonical_key_sha256(key_frame: pd.DataFrame, *, unique: bool = True) -> str:
    """Hash sorted cell/drug keys without reading or hashing response values."""
    keys = list(zip(
        key_frame["cell_line_id"].astype(str),
        key_frame["drug_id"].astype(str),
    ))
    if unique:
        keys = list(set(keys))
    keys.sort()
    digest = hashlib.sha256()
    for cell_id, drug_id in keys:
        digest.update(cell_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(drug_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def audit(pairs_path: Path, input_dir: Path, input_verification_path: Path,
          output_path: Path) -> dict[str, Any]:
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("identity_resolution_status") != IDENTITY_STATUS
            or manifest.get("scientific_strict_unseen_accepted") is not True
            or manifest.get("target_response_values_read") is not False):
        raise ValueError("key coverage requires an accepted response-blind identity revision")

    identity_resolution_path = input_dir / "identity_resolution.json"
    identity_resolution = json.loads(identity_resolution_path.read_text(encoding="utf-8"))
    if (identity_resolution.get("status") != IDENTITY_STATUS
            or identity_resolution.get("strict_unseen_accepted") is not True
            or identity_resolution.get("target_response_values_read") is not False):
        raise ValueError("identity-resolution receipt is missing or invalid")
    verification = json.loads(input_verification_path.read_text(encoding="utf-8"))
    if (verification.get("status") != "PASS_RESPONSE_BLIND_IDENTITY_RESOLVED"
            or verification.get("feature_verification_status") != "PASS"
            or verification.get("identity_resolution_status") != IDENTITY_STATUS
            or verification.get("scientific_strict_unseen_accepted") is not True
            or verification.get("target_response_values_read") is not False
            or verification.get("input_manifest_sha256") != sha256(manifest_path)):
        raise ValueError("independent response-blind feature verification is required before key coverage")

    cell_axis_path = input_dir / "cell_axis.json"
    drug_axis_path = input_dir / "drug_axis.parquet"
    cell_ids = [str(value) for value in json.loads(cell_axis_path.read_text(encoding="utf-8"))]
    if not cell_ids or len(cell_ids) != len(set(cell_ids)):
        raise ValueError("revised strict-unseen cell axis must be non-empty and unique")
    strict_frame = pd.read_csv(input_dir / "strict_unseen_cell_lines_v2.csv", usecols=["cell_line_id"])
    strict_ids = set(_validated_ids(strict_frame["cell_line_id"], "revised strict identity list"))
    if strict_ids != set(cell_ids):
        raise ValueError("revised strict identity list differs from the materialized cell axis")
    drugs = pd.read_parquet(drug_axis_path, columns=["drug_id"])
    drug_ids = _validated_ids(drugs["drug_id"], "frozen PRISM drug axis")
    if not drug_ids or len(drug_ids) != len(set(drug_ids)):
        raise ValueError("frozen PRISM drug axis must be non-empty and unique")
    actual_feature_hashes = {
        name: sha256(input_dir / name)
        for name in ("identity_resolution.json", "strict_unseen_cell_lines_v2.csv",
                     "excluded_source_overlap_cell_lines_v2.csv", "cell_axis.json",
                     "cell_expression.npy", "drug_axis.parquet", "drug_ordinal128.npy")
    }
    if (manifest.get("files_sha256") != actual_feature_hashes
            or identity_resolution.get("strict_unseen_ids_sha256")
            != actual_feature_hashes["strict_unseen_cell_lines_v2.csv"]
            or identity_resolution.get("source_overlap_ids_sha256")
            != actual_feature_hashes["excluded_source_overlap_cell_lines_v2.csv"]):
        raise ValueError("materialized feature and revised identity checksums do not match their receipts")

    # Inspect Parquet schema metadata first, then project only the identity keys.
    # This source stores the PRISM sample identity as `sample_id`; the independently
    # verified sample contract establishes sample_id == cell_line_id.
    source_columns = set(pq.ParquetFile(pairs_path).schema.names)
    available_cell_columns = [name for name in ("sample_id", "cell_line_id") if name in source_columns]
    if not available_cell_columns or "drug_id" not in source_columns:
        raise ValueError("PRISM pair table lacks a supported cell identity or drug_id column")
    selected_columns = [*available_cell_columns, "drug_id"]
    keys = pd.read_parquet(pairs_path, columns=selected_columns)
    cell_key = pd.Series(_validated_ids(keys[available_cell_columns[0]],
                                        f"PRISM response-table {available_cell_columns[0]} keys"))
    if len(available_cell_columns) == 2:
        second_key = pd.Series(_validated_ids(keys[available_cell_columns[1]],
                                              f"PRISM response-table {available_cell_columns[1]} keys"))
        if not cell_key.equals(second_key):
            raise ValueError("PRISM pair table sample_id and cell_line_id columns disagree")
    drug_key = pd.Series(_validated_ids(keys["drug_id"], "PRISM response-table drug keys"))
    key_frame = pd.DataFrame({"cell_line_id": cell_key, "drug_id": drug_key})
    n_key_rows = len(key_frame)
    unique_keys = key_frame.drop_duplicates(["cell_line_id", "drug_id"])
    cohort_panel = unique_keys.loc[
        unique_keys["cell_line_id"].isin(set(cell_ids))
        & unique_keys["drug_id"].isin(set(drug_ids))
    ]
    counts = cohort_panel.groupby("cell_line_id")["drug_id"].nunique()
    counts = counts.reindex(cell_ids, fill_value=0)
    eligible = sorted(counts[counts >= 3].index.astype(str))
    ineligible = sorted(counts[counts < 3].index.astype(str))
    result = {
        "experiment_id": "EXP-044",
        "status": "PASS_RESPONSE_BLIND_IDENTITY_KEY_COVERAGE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "identity_resolution_status": IDENTITY_STATUS,
        "identity_resolution_path": identity_resolution_path.as_posix(),
        "identity_resolution_sha256": sha256(identity_resolution_path),
        "input_verification_path": input_verification_path.as_posix(),
        "input_verification_sha256": sha256(input_verification_path),
        "input_manifest_sha256": sha256(manifest_path),
        "cell_axis_sha256": sha256(cell_axis_path),
        "strict_identity_list_sha256": sha256(input_dir / "strict_unseen_cell_lines_v2.csv"),
        "drug_axis_sha256": sha256(drug_axis_path),
        "source_pair_table_path": pairs_path.as_posix(),
        "source_pair_table_file_sha256": sha256(pairs_path),
        "source_pair_table_file_digest_method": (
            "sha256 of source file bytes; parquet decoding requested only identity key columns"
        ),
        "source_pair_table_opened_for_keys_only": True,
        "source_pair_table_columns_requested": selected_columns,
        "source_pair_table_key_rows_sha256": _canonical_key_sha256(key_frame, unique=False),
        "source_pair_table_unique_keys_sha256": _canonical_key_sha256(key_frame),
        "source_pair_table_cell_identity_column": available_cell_columns[0],
        "source_pair_table_cell_identity_is_verified_sample_axis": (
            available_cell_columns[0] == "sample_id" or len(available_cell_columns) == 2
        ),
        "response_value_columns_requested": [],
        "target_response_values_read": False,
        "target_response_values_loaded": False,
        "n_revised_strict_unseen_cells": len(cell_ids),
        "n_prism_panel_drugs": len(drug_ids),
        "n_source_pair_key_rows": n_key_rows,
        "n_unique_source_pair_keys": len(unique_keys),
        "n_revised_cohort_panel_cell_drug_keys": len(cohort_panel),
        "response_blind_cohort_key_sha256": _canonical_key_sha256(cohort_panel),
        "response_blind_key_hash_contract": (
            "sha256 of sorted unique UTF-8 cell_line_id NUL drug_id LF pairs "
            "within the frozen strict-unseen cell and drug axes"
        ),
        "n_cells_with_at_least_three_panel_drugs": len(eligible),
        "n_cells_below_three_panel_drugs": len(ineligible),
        "eligible_cell_ids_sha256": hashlib.sha256(
            ("\n".join(eligible) + "\n").encode("utf-8")
        ).hexdigest(),
        "strict_unseen_ids_sha256": hashlib.sha256(
            ("\n".join(cell_ids) + "\n").encode("utf-8")
        ).hexdigest(),
        "eligible_cell_ids": eligible,
        "ineligible_cell_ids": ineligible,
        "unique_panel_drugs_per_cell": {str(cell): int(counts[cell]) for cell in cell_ids},
        "provenance": _runtime_provenance(),
    }
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite EXP-044 identity-key audit: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--input-verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.pairs, args.input_dir, args.input_verification, args.output),
                     ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
