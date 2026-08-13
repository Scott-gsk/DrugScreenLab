"""Response-blind Broad PRISM identity freeze and compact response reader."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd


_BASE_ID = re.compile(r"^(BRD-[A-Z]\d{8})")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_frozen_response_mapping(
    *,
    cohort_path: Path | str,
    bridge_path: Path | str,
    treatment_info_path: Path | str,
    registry_path: Path | str,
) -> pd.DataFrame:
    """Return response columns after identity/context and feature freeze.

    No response matrix is opened here.  Full Broad IDs are joined first;
    base-ID membership alone is intentionally insufficient because a base may
    have multiple dose/screen columns.
    """
    cohort = json.loads(Path(cohort_path).read_text(encoding="utf-8"))
    eligible_bases = {str(value) for value in cohort.get("eligible_base_ids", [])}
    bridge = pd.read_csv(bridge_path, dtype=str, keep_default_na=False)
    required_bridge = {
        "prism_broad_id",
        "prism_broad_id_base",
        "lincs_pert_id",
        "match_status",
    }
    missing = sorted(required_bridge.difference(bridge.columns))
    if missing:
        raise ValueError(f"Broad bridge is missing columns: {missing}")
    bridge = bridge.loc[bridge["match_status"].eq("MATCHED_IDENTITY")].copy()
    bridge = bridge.drop_duplicates("prism_broad_id")
    bridge = bridge.loc[bridge["prism_broad_id_base"].isin(eligible_bases)]

    treatment = pd.read_csv(treatment_info_path, dtype=str, keep_default_na=False)
    required_treatment = {"broad_id", "column_name"}
    missing = sorted(required_treatment.difference(treatment.columns))
    if missing:
        raise ValueError(f"PRISM treatment metadata is missing columns: {missing}")
    treatment["prism_broad_id_base"] = treatment["broad_id"].astype(str).str.extract(_BASE_ID, expand=False)
    treatment = treatment.loc[treatment["prism_broad_id_base"].isin(eligible_bases)].copy()
    treatment = treatment.drop_duplicates(["prism_broad_id_base", "broad_id", "column_name"])
    mapping = bridge.merge(
        treatment,
        left_on=["prism_broad_id", "prism_broad_id_base"],
        right_on=["broad_id", "prism_broad_id_base"],
        how="inner",
        validate="one_to_many",
        suffixes=("_bridge", "_treatment"),
    )
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    eligible_pert_ids = {
        str(row["pert_id"])
        for row in registry.get("drugs", [])
        if bool(row.get("broad_inference_eligible"))
    }
    mapping = mapping.loc[mapping["lincs_pert_id"].isin(eligible_pert_ids)].copy()
    if mapping.empty:
        raise ValueError("frozen Broad response mapping is empty after registry eligibility")
    mapping["pert_id"] = mapping["lincs_pert_id"].astype(str)
    if "lincs_pert_iname" in mapping.columns:
        mapping["pert_iname"] = mapping["lincs_pert_iname"].astype(str)
    else:
        mapping["pert_iname"] = ""
    keep = [
        "prism_broad_id_base",
        "prism_broad_id",
        "column_name",
        "pert_id",
        "pert_iname",
    ]
    for column in ("name", "dose", "screen_id", "smiles"):
        if column in mapping.columns:
            keep.append(column)
    mapping = mapping[keep].drop_duplicates().sort_values(
        ["prism_broad_id_base", "prism_broad_id", "column_name"], kind="mergesort"
    )
    return mapping.reset_index(drop=True)


def build_broad_prism_compact(
    *,
    cohort_path: Path | str,
    bridge_path: Path | str,
    treatment_info_path: Path | str,
    cell_info_path: Path | str,
    registry_path: Path | str,
    matrix_path: Path | str,
    output_path: Path | str,
    audit_path: Path | str,
) -> dict[str, Any]:
    """Read only the frozen Broad response columns and CRC lines."""
    cohort_file = Path(cohort_path)
    bridge_file = Path(bridge_path)
    treatment_file = Path(treatment_info_path)
    cell_file = Path(cell_info_path)
    registry_file = Path(registry_path)
    matrix_file = Path(matrix_path)
    output_file = Path(output_path)
    audit_file = Path(audit_path)

    mapping = select_frozen_response_mapping(
        cohort_path=cohort_file,
        bridge_path=bridge_file,
        treatment_info_path=treatment_file,
        registry_path=registry_file,
    )
    cohort = json.loads(cohort_file.read_text(encoding="utf-8"))
    crc_ids = {str(value) for value in cohort.get("eligible_crc_depmap_ids", [])}
    cell_info = pd.read_csv(cell_file, dtype=str, keep_default_na=False)
    crc = cell_info.loc[cell_info["depmap_id"].isin(crc_ids)].drop_duplicates("depmap_id").copy()
    if len(crc) != len(crc_ids):
        missing = sorted(crc_ids.difference(crc["depmap_id"].astype(str)))
        raise ValueError(f"CRC lines absent from PRISM cell metadata: {missing[:5]}")

    header = pd.read_csv(matrix_file, nrows=0).columns.tolist()
    columns = mapping["column_name"].astype(str).tolist()
    missing_columns = sorted(set(columns).difference(header))
    if missing_columns:
        raise ValueError(f"frozen Broad response columns absent from matrix: {missing_columns[:5]}")
    values = pd.read_csv(matrix_file, usecols=[header[0], *columns], index_col=0)
    values.index = values.index.astype(str)
    missing_lines = sorted(crc_ids.difference(values.index))
    if missing_lines:
        raise ValueError(f"CRC lines absent from response matrix: {missing_lines[:5]}")

    cell_lookup = crc.set_index("depmap_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    for record in mapping.itertuples(index=False):
        column = str(record.column_name)
        for depmap_id in sorted(crc_ids):
            raw = pd.to_numeric(values.loc[depmap_id, column], errors="coerce")
            if pd.isna(raw):
                continue
            info = cell_lookup[depmap_id]
            rows.append(
                {
                    "study_id": "PRISM_REPURPOSING_PRIMARY_19Q4",
                    "source_revision": "PRISM primary replicate-collapsed; frozen Broad CRC extension",
                    "depmap_id": depmap_id,
                    "ccle_name": str(info["ccle_name"]),
                    "primary_tissue": str(info["primary_tissue"]),
                    "broad_id": str(record.prism_broad_id),
                    "broad_id_base": str(record.prism_broad_id_base),
                    "column_name": column,
                    "pert_id": str(record.pert_id),
                    "pert_iname": str(getattr(record, "pert_iname", "")),
                    "response_raw": float(raw),
                    "sensitivity_score": float(-raw),
                    "response_unit": "PRISM log2 fold-change",
                    "response_direction": "lower_log2fc_more_sensitive",
                    "source_row_id": f"{depmap_id}|{column}",
                }
            )
    compact = pd.DataFrame(rows)
    if compact.empty:
        raise ValueError("Broad PRISM compact response is empty")
    compact = compact.sort_values(["depmap_id", "pert_id", "column_name"], kind="mergesort").reset_index(drop=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    compact.to_parquet(output_file, index=False)

    payload: dict[str, Any] = {
        "format": "xpert_broad_prism_crc_response_v1",
        "status": "RESPONSE_READ_AFTER_IDENTITY_CONTEXT_FREEZE",
        "response_values_read": True,
        "selection": {
            "cohort": str(cohort_file),
            "cohort_sha256": file_sha256(cohort_file),
            "registry": str(registry_file),
            "registry_sha256": file_sha256(registry_file),
            "identity_join": "matched bridge full prism_broad_id + extracted base ID; then registry broad_inference_eligible",
            "context_join": "frozen eligible_crc_depmap_ids; no response-dependent context selection",
        },
        "source": {
            "matrix": str(matrix_file),
            "matrix_sha256": file_sha256(matrix_file),
            "matrix_bytes": matrix_file.stat().st_size,
            "bridge": str(bridge_file),
            "treatment_info": str(treatment_file),
            "cell_info": str(cell_file),
        },
        "counts": {
            "frozen_identity_base_ids": int(len(cohort.get("eligible_base_ids", []))),
            "registry_eligible_base_ids": int(mapping["prism_broad_id_base"].nunique()),
            "response_columns": int(mapping["column_name"].nunique()),
            "crc_lines": int(compact["depmap_id"].nunique()),
            "registry_eligible_pert_ids": int(compact["pert_id"].nunique()),
            "finite_response_rows": int(len(compact)),
        },
        "output": {
            "path": str(output_file),
            "sha256": file_sha256(output_file),
            "schema": list(compact.columns),
        },
        "large_source_not_tracked": True,
    }
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    audit_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return payload
