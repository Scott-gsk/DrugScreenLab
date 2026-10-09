"""Bridge the approved HCMI lookup to the exact Nature and DepMap ID axes."""

from __future__ import annotations

import csv
import hashlib
import json
import time
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
WHITELIST_RECEIPT = ROOT / "artifacts/experiments/EXP-045/20260925_source_provenance_overlap_recovery_v1/receipt.json"
PREFIX_RECEIPT = ROOT / "artifacts/experiments/EXP-045/20260925_nature_hcmi_sample_id_route_precheck_v1/receipt.json"
BRIDGE_RECEIPT = ROOT / "artifacts/experiments/EXP-045/20260925_nature_hcmi_depmap_bridge_projection_v1/receipt.json"
NATURE_ZIP = ROOT / "data/raw/transfer_public_assets/organoid_biobank/41586_2026_10830_MOESM3_ESM.zip"
REGISTRY = ROOT / "data/registry/datasets.json"
MODEL_CSV = ROOT / "data/raw/depmap/24q2/Model.csv"
RNA_CSV = ROOT / "data/raw/depmap/24q2/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
CRISPR_CSV = ROOT / "data/raw/depmap/24q2/CRISPRGeneEffect.csv"
OUTPUT_DIR = ROOT / "artifacts/experiments/EXP-045/20260925_nature_hcmi_actual_input_axis_bridge_v1"
MEMBER = "supplementary_table_2_revision.xlsx"
SHEET = "data_availability"
NATURE_COLUMNS = ["sample_ID", "sanger_ID", "individual_ID"]
MODEL_COLUMNS = ["ModelID", "PatientID", "CellLineName", "CCLEName", "HCMIID", "WTSIMasterCellID", "SangerModelID", "COSMICID"]
API_BASE = "https://api.cellmodelpassports.sanger.ac.uk"
SOURCE = "HCMI"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def id_hash(value: str | None) -> str | None:
    return sha256_bytes(value.encode("utf-8")) if value else None


def scalar_strings(value: Any) -> set[str]:
    found: set[str] = set()
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, str) and item.strip():
            found.add(item.strip())
    return found


def unique_keys(values: pd.Series) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


def file_entry(registry: Any, dataset_id: str, filename: str) -> dict[str, Any]:
    if not isinstance(registry, list):
        raise SystemExit("Dataset registry must be a list; stopped")
    entries = [item for item in registry if isinstance(item, dict) and item.get("id") == dataset_id]
    if len(entries) != 1:
        raise SystemExit(f"Expected one dataset registry entry for {dataset_id}; stopped")
    entry = entries[0]["files"].get(filename)
    if not isinstance(entry, dict):
        raise SystemExit(f"Missing registered file entry for {filename}; stopped")
    return entry


def axis_summary(path: Path) -> tuple[list[str], str, Counter[str]]:
    frame = pd.read_csv(path, usecols=[0], dtype=str, keep_default_na=False, low_memory=False)
    header = str(frame.columns[0])
    frequencies = Counter(value.strip() for value in frame.iloc[:, 0] if value.strip())
    return sorted(frequencies), header, frequencies


def main() -> None:
    whitelist_receipt = json.loads(WHITELIST_RECEIPT.read_text(encoding="utf-8"))
    prefix_receipt = json.loads(PREFIX_RECEIPT.read_text(encoding="utf-8"))
    prior_bridge = json.loads(BRIDGE_RECEIPT.read_text(encoding="utf-8"))
    whitelist_groups = whitelist_receipt["nature_overlap_result"]["individual_to_sample_ID_groups"]
    sample_ids = sorted({str(value).strip() for group in whitelist_groups.values() for value in group if str(value).strip()})
    if not (
        len(sample_ids) == 14
        and prefix_receipt["status"] == "PASS_AGGREGATE_ONLY"
        and prefix_receipt["scope"]["exact_whitelist_set_matches_official_Table2"] is True
        and prefix_receipt["scope"]["HCM_prefix_total_unique_sample_ID_count"] == 14
    ):
        raise SystemExit("Exact HCMI whitelist-to-Nature bridge precondition failed; stopped")

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    expected_files = {
        MODEL_CSV: file_entry(registry, "depmap_public_24q2_rnaseq_raw_v1", MODEL_CSV.name),
        RNA_CSV: file_entry(registry, "depmap_public_24q2_rnaseq_raw_v1", RNA_CSV.name),
        CRISPR_CSV: file_entry(registry, "depmap_public_24q2_crispr_gene_effect_raw_v1", CRISPR_CSV.name),
    }
    for path, entry in expected_files.items():
        if path.stat().st_size != int(entry["bytes"]):
            raise SystemExit(f"Registered file size mismatch for {path.name}; stopped before API queries")
        if sha256_file(path) != entry["sha256"]:
            raise SystemExit(f"Registered file checksum mismatch for {path.name}; stopped before API queries")

    nature_zip_sha = sha256_file(NATURE_ZIP)
    expected_nature_zip_sha = prior_bridge["inputs"]["sha256"][str(NATURE_ZIP.relative_to(ROOT)).replace("\\", "/")]
    if nature_zip_sha != expected_nature_zip_sha:
        raise SystemExit("Nature Supplementary Table ZIP differs from its approved bridge receipt; stopped")
    with zipfile.ZipFile(NATURE_ZIP) as archive:
        table_bytes = archive.read(MEMBER)
    table = pd.read_excel(BytesIO(table_bytes), sheet_name=SHEET, usecols=NATURE_COLUMNS, dtype=str)
    for column in NATURE_COLUMNS:
        table[column] = table[column].fillna("").astype(str).str.strip()
    table = table.loc[table["sample_ID"].isin(set(sample_ids)), NATURE_COLUMNS].drop_duplicates()
    nature_sample_ids = unique_keys(table["sample_ID"])
    if nature_sample_ids != sample_ids:
        raise SystemExit("The exact whitelist did not reproduce against official Table 2; stopped")
    nature_pairs_per_sample = table.groupby("sample_ID", dropna=False)[["sanger_ID", "individual_ID"]].nunique(dropna=False)
    if not bool((nature_pairs_per_sample.max(axis=1) == 1).all()):
        raise SystemExit("Official Table 2 has a conflicting Sanger/individual mapping for a whitelisted sample; stopped")
    nature_sanger_by_sample = table.drop_duplicates("sample_ID").set_index("sample_ID")["sanger_ID"].to_dict()
    nature_individual_by_sample = table.drop_duplicates("sample_ID").set_index("sample_ID")["individual_ID"].to_dict()

    model = pd.read_csv(MODEL_CSV, usecols=MODEL_COLUMNS, dtype=str, keep_default_na=False)
    model_sets = {column: set(model[column].str.strip()) - {""} for column in MODEL_COLUMNS}
    rna_axis, rna_header, rna_freq = axis_summary(RNA_CSV)
    crispr_axis, crispr_header, crispr_freq = axis_summary(CRISPR_CSV)
    rna_set, crispr_set = set(rna_axis), set(crispr_axis)

    session = requests.Session()
    session.headers.update({"Accept": "application/vnd.api+json"})
    rows: list[dict[str, Any]] = []
    http_statuses: Counter[str] = Counter()
    for index, sample_id in enumerate(sample_ids):
        url = f"{API_BASE}/models/{SOURCE}/{quote(sample_id, safe='')}"
        response = session.get(
            url,
            params={"fields[model]": "identifiers", "include": "identifiers"},
            timeout=30,
        )
        http_statuses[str(response.status_code)] += 1
        try:
            body = response.json()
        except ValueError:
            body = None
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, dict):
            rows.append(
                {
                    "sample_ID_sha256": id_hash(sample_id),
                    "nature_Sanger_ID_sha256": id_hash(nature_sanger_by_sample[sample_id]),
                    "nature_individual_ID_sha256": id_hash(nature_individual_by_sample[sample_id]),
                    "http_status": int(response.status_code),
                    "model_object_returned": False,
                    "cmp_ModelID_sha256": None,
                    "identifier_record_count": 0,
                    "sample_ID_echoed_in_identifiers": False,
                    "Nature_Sanger_ID_echoed_in_identifiers": False,
                    "Model_csv_exact_hit_fields": [],
                    "RNA_axis_unique_identifier_hit_count": 0,
                    "RNA_axis_row_hit_count": 0,
                    "CRISPR_axis_unique_identifier_hit_count": 0,
                    "CRISPR_axis_row_hit_count": 0,
                }
            )
        else:
            cmp_model_id = str(data.get("id", "")).strip() or None
            included = body.get("included", []) if isinstance(body, dict) else []
            identifier_records = [
                item for item in included
                if isinstance(item, dict) and item.get("type") == "model_identifier"
            ] if isinstance(included, list) else []
            identifier_values = set().union(
                *(scalar_strings(item.get("attributes", {})) for item in identifier_records)
            ) if identifier_records else set()
            identity_values = set(identifier_values)
            if cmp_model_id:
                identity_values.add(cmp_model_id)
            model_field_hits = sorted(column for column, values in model_sets.items() if identity_values & values)
            rna_matches = identity_values & rna_set
            crispr_matches = identity_values & crispr_set
            rows.append(
                {
                    "sample_ID_sha256": id_hash(sample_id),
                    "nature_Sanger_ID_sha256": id_hash(nature_sanger_by_sample[sample_id]),
                    "nature_individual_ID_sha256": id_hash(nature_individual_by_sample[sample_id]),
                    "http_status": int(response.status_code),
                    "model_object_returned": True,
                    "cmp_ModelID_sha256": id_hash(cmp_model_id),
                    "identifier_record_count": len(identifier_records),
                    "sample_ID_echoed_in_identifiers": sample_id in identifier_values,
                    "Nature_Sanger_ID_echoed_in_identifiers": nature_sanger_by_sample[sample_id] in identifier_values,
                    "Model_csv_exact_hit_fields": model_field_hits,
                    "RNA_axis_unique_identifier_hit_count": len(rna_matches),
                    "RNA_axis_row_hit_count": sum(rna_freq[value] for value in rna_matches),
                    "CRISPR_axis_unique_identifier_hit_count": len(crispr_matches),
                    "CRISPR_axis_row_hit_count": sum(crispr_freq[value] for value in crispr_matches),
                }
            )
        if index < len(sample_ids) - 1:
            time.sleep(0.15)

    nonnull_rows = [row for row in rows if row["model_object_returned"]]
    status = {
        "query_count": len(rows),
        "http_status_counts": dict(sorted(http_statuses.items())),
        "model_objects_returned": len(nonnull_rows),
        "null_or_nonobject_responses": len(rows) - len(nonnull_rows),
        "unique_CMP_ModelID_hashes": len({row["cmp_ModelID_sha256"] for row in nonnull_rows if row["cmp_ModelID_sha256"]}),
        "query_sample_ID_echoes_in_identifiers": sum(row["sample_ID_echoed_in_identifiers"] for row in rows),
        "Nature_Sanger_ID_echoes_in_identifiers": sum(row["Nature_Sanger_ID_echoed_in_identifiers"] for row in rows),
        "CMP_identity_matches_by_Model_csv_field": {
            column: sum(column in row["Model_csv_exact_hit_fields"] for row in rows)
            for column in MODEL_COLUMNS
        },
        "queries_with_any_Model_csv_identity_match": sum(bool(row["Model_csv_exact_hit_fields"]) for row in rows),
        "queries_with_any_RNA_axis_match": sum(row["RNA_axis_unique_identifier_hit_count"] > 0 for row in rows),
        "unique_RNA_axis_identifier_matches": sum(row["RNA_axis_unique_identifier_hit_count"] for row in rows),
        "RNA_axis_row_matches": sum(row["RNA_axis_row_hit_count"] for row in rows),
        "queries_with_any_CRISPR_axis_match": sum(row["CRISPR_axis_unique_identifier_hit_count"] > 0 for row in rows),
        "unique_CRISPR_axis_identifier_matches": sum(row["CRISPR_axis_unique_identifier_hit_count"] for row in rows),
        "CRISPR_axis_row_matches": sum(row["CRISPR_axis_row_hit_count"] for row in rows),
        "duplicate_RNA_axis_keys": sum(count - 1 for count in rna_freq.values()),
        "duplicate_CRISPR_axis_keys": sum(count - 1 for count in crispr_freq.values()),
    }
    per_key_payload = "\n".join(
        sorted(
            ":".join(
                str(row[key])
                for key in (
                    "sample_ID_sha256",
                    "nature_Sanger_ID_sha256",
                    "cmp_ModelID_sha256",
                    "http_status",
                    "RNA_axis_unique_identifier_hit_count",
                    "CRISPR_axis_unique_identifier_hit_count",
                )
            )
            for row in rows
        )
    ).encode("utf-8")
    receipt = {
        "receipt_id": "EXP-045-Nature-HCMI-actual-input-axis-bridge-v1",
        "status": "IDENTITY_AXIS_AUDIT_COMPLETE",
        "scope": {
            "exact_whitelist_count": 14,
            "source": SOURCE,
            "endpoint": "/models/{source}/{source_id}",
            "api_fieldset_and_include": "fields[model]=identifiers&include=identifiers",
            "Nature_Table2_columns": NATURE_COLUMNS,
            "DepMap_Model_csv_identity_columns": MODEL_COLUMNS,
            "RNA_input_axis": "first CSV column only; all other values excluded",
            "CRISPR_input_axis": "first CSV column only; all other values excluded",
            "raw_API_responses_or_identifiers_saved": False,
            "expression_or_Chronos_values_labels_or_checkpoints_read": False,
            "exact_matching": "case-sensitive exact string equality after trimming surrounding whitespace",
        },
        "provenance": {
            "Nature_quarantine_receipt_sha256": sha256_file(WHITELIST_RECEIPT),
            "Nature_HCM_prefix_receipt_sha256": sha256_file(PREFIX_RECEIPT),
            "Nature_Table2_bridge_receipt_sha256": sha256_file(BRIDGE_RECEIPT),
            "Nature_Supplement_ZIP_sha256": nature_zip_sha,
            "DepMap_24Q2_Model_csv_sha256": expected_files[MODEL_CSV]["sha256"],
            "DepMap_24Q2_RNA_sha256": expected_files[RNA_CSV]["sha256"],
            "DepMap_24Q2_CRISPR_sha256": expected_files[CRISPR_CSV]["sha256"],
            "DepMap_registry_sha256": sha256_file(REGISTRY),
            "official_Table2_Sanger_pair_set_sha256": sha256_bytes(
                "\n".join(sorted(f"{sample}:{nature_sanger_by_sample[sample]}" for sample in sample_ids)).encode("utf-8")
            ),
            "RNA_axis_key_set_sha256": sha256_bytes("\n".join(rna_axis).encode("utf-8")),
            "CRISPR_axis_key_set_sha256": sha256_bytes("\n".join(crispr_axis).encode("utf-8")),
            "per_key_redacted_result_sha256": sha256_bytes(per_key_payload),
            "RNA_first_column_header": rna_header,
            "CRISPR_first_column_header": crispr_header,
            "RNA_axis_row_count": sum(rna_freq.values()),
            "CRISPR_axis_row_count": sum(crispr_freq.values()),
        },
        "results": status,
        "per_key_redacted_status_and_match_counts": rows,
        "interpretation_limit": (
            "A zero exact match means only that the current registered matrix axis did not contain the returned identifier strings. "
            "A 404 or null response means only this route did not resolve that key; neither result proves global model absence. "
            "This is an identity bridge only, not dependency transfer performance or model training evidence."
        ),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    output_path = OUTPUT_DIR / "receipt.json"
    output_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "receipt": str(output_path.relative_to(ROOT)).replace("\\", "/"),
                "receipt_sha256": sha256_file(output_path),
                "results": status,
                "rna_first_column_header": rna_header,
                "crispr_first_column_header": crispr_header,
                "raw_identifiers_emitted": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
