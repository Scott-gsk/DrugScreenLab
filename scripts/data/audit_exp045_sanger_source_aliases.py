"""Audit three approved Sanger aliases against DepMap Public 24Q2 input axes.

This is a narrow, read-only identity bridge. It performs exactly one request for
each registered SIDM key using the documented single-model route, then compares
only the returned model ID/identifier fields with the registered Model.csv
identity columns and the first columns of the RNA and CRISPR input files.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts/experiments/EXP-045/20260925_sanger_source_alias_bridge_v1"
OUT_FILE = OUT_DIR / "receipt.json"
REGISTRY = ROOT / "data/registry/datasets.json"
MODEL_CSV = ROOT / "data/raw/depmap/24q2/Model.csv"
RNA_CSV = ROOT / "data/raw/depmap/24q2/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
CRISPR_CSV = ROOT / "data/raw/depmap/24q2/CRISPRGeneEffect.csv"
API_BASE = "https://api.cellmodelpassports.sanger.ac.uk"
REGISTERED_SANGER_IDS = ("SIDM00117", "SIDM00427", "SIDM01117")
MODEL_COLUMNS = (
    "ModelID",
    "PatientID",
    "CellLineName",
    "CCLEName",
    "HCMIID",
    "WTSIMasterCellID",
    "SangerModelID",
    "COSMICID",
)
EXPECTED_FILES = {
    MODEL_CSV: "depmap_public_24q2_rnaseq_raw_v1",
    RNA_CSV: "depmap_public_24q2_rnaseq_raw_v1",
    CRISPR_CSV: "depmap_public_24q2_crispr_gene_effect_raw_v1",
}


class AuditStop(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_file_entry(registry: Any, dataset_id: str, filename: str) -> dict[str, Any]:
    if not isinstance(registry, list):
        raise AuditStop("DATA_REGISTRY_SCHEMA_UNKNOWN")
    entries = [
        item for item in registry
        if isinstance(item, dict) and item.get("id") == dataset_id
    ]
    if len(entries) != 1:
        raise AuditStop("DATA_REGISTRY_ENTRY_NOT_UNIQUE")
    files = entries[0].get("files")
    if not isinstance(files, dict) or not isinstance(files.get(filename), dict):
        raise AuditStop("DATA_REGISTRY_FILE_ENTRY_MISSING")
    return files[filename]


def verify_inputs() -> dict[str, str]:
    try:
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise AuditStop("DATA_REGISTRY_UNREADABLE") from None
    hashes: dict[str, str] = {}
    for path, dataset_id in EXPECTED_FILES.items():
        try:
            entry = dataset_file_entry(registry, dataset_id, path.name)
            if path.stat().st_size != int(entry["bytes"]):
                raise AuditStop("REGISTERED_INPUT_SIZE_MISMATCH")
            actual_hash = sha256_file(path)
            if actual_hash != str(entry["sha256"]):
                raise AuditStop("REGISTERED_INPUT_CHECKSUM_MISMATCH")
            hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = actual_hash
        except AuditStop:
            raise
        except (OSError, KeyError, TypeError, ValueError):
            raise AuditStop("REGISTERED_INPUT_UNREADABLE_OR_SCHEMA_UNKNOWN") from None
    return hashes


def axis_summary(path: Path) -> tuple[set[str], Counter[str]]:
    try:
        frame = pd.read_csv(path, usecols=[0], dtype=str, keep_default_na=False, low_memory=False)
    except Exception:
        raise AuditStop("INPUT_AXIS_UNREADABLE_OR_SCHEMA_UNKNOWN") from None
    if frame.shape[1] != 1:
        raise AuditStop("INPUT_AXIS_SCHEMA_UNKNOWN")
    values = [str(value).strip() for value in frame.iloc[:, 0] if str(value).strip()]
    return set(values), Counter(values)


def model_identity_index() -> tuple[list[dict[str, str]], dict[str, str]]:
    try:
        frame = pd.read_csv(
            MODEL_CSV,
            usecols=list(MODEL_COLUMNS),
            dtype=str,
            keep_default_na=False,
            low_memory=False,
        )
    except Exception:
        raise AuditStop("MODEL_IDENTITY_COLUMNS_UNREADABLE_OR_SCHEMA_UNKNOWN") from None
    if tuple(frame.columns) != MODEL_COLUMNS:
        # CSV readers preserve requested usecols order for this file in the
        # approved environment; any discrepancy is treated as unknown schema.
        if set(frame.columns) != set(MODEL_COLUMNS):
            raise AuditStop("MODEL_IDENTITY_COLUMNS_SCHEMA_UNKNOWN")
    rows: list[dict[str, str]] = []
    for values in frame.to_dict(orient="records"):
        rows.append({column: str(values[column]).strip() for column in MODEL_COLUMNS})
    return rows, {column: column for column in MODEL_COLUMNS}


def string_leaves(value: Any) -> set[str]:
    """Return nonempty strings only from model_identifier attributes."""
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


def parse_model_projection(response: requests.Response) -> tuple[str | None, set[str], int]:
    try:
        body = response.json()
    except ValueError:
        raise AuditStop("API_SCHEMA_UNKNOWN_STOPPED") from None
    if not isinstance(body, dict) or not isinstance(body.get("data"), dict):
        raise AuditStop("API_SCHEMA_UNKNOWN_STOPPED")
    data = body["data"]
    if data.get("type") != "model" or not isinstance(data.get("id"), str) or not data["id"].strip():
        raise AuditStop("API_SCHEMA_UNKNOWN_STOPPED")
    included = body.get("included", [])
    if not isinstance(included, list):
        raise AuditStop("API_SCHEMA_UNKNOWN_STOPPED")
    identities = {data["id"].strip()}
    identifier_count = 0
    for resource in included:
        if not isinstance(resource, dict) or resource.get("type") != "model_identifier":
            raise AuditStop("API_SCHEMA_UNKNOWN_STOPPED")
        attributes = resource.get("attributes")
        if not isinstance(attributes, dict):
            raise AuditStop("API_SCHEMA_UNKNOWN_STOPPED")
        identifier_count += 1
        identities.update(string_leaves(attributes))
    return data["id"].strip(), identities, identifier_count


def empty_key_result(sanger_id: str, request_status: str) -> dict[str, Any]:
    return {
        "registered_Sanger_ID": sanger_id,
        "http_status": request_status,
        "model_objects_returned": 0,
        "CMP_primary_ModelID": None,
        "DepMap_24Q2_exact_candidate_ModelIDs": [],
        "DepMap_24Q2_unique_ModelID_count": 0,
        "DepMap_24Q2_ambiguous_ModelID_count": 0,
        "Model_csv_exact_hit_fields": [],
        "RNA_axis": {"hit": False, "matched_unique_key_count": 0, "matched_row_count": 0},
        "CRISPR_axis": {"hit": False, "matched_unique_key_count": 0, "matched_row_count": 0},
    }


def main() -> int:
    if OUT_FILE.exists():
        print("OUTPUT_EXISTS_NO_OVERWRITE")
        return 2
    receipt: dict[str, Any] = {
        "receipt_id": "EXP-045-three-Sanger-source-alias-bridge-v1",
        "status": "NOT_STARTED",
        "scope": {
            "registered_Sanger_IDs": list(REGISTERED_SANGER_IDS),
            "endpoint": "/models/{id}",
            "query_parameters": "fields[model]=identifiers&include=identifiers",
            "DepMap_Model_csv_identity_columns": list(MODEL_COLUMNS),
            "RNA_and_CRISPR_axes": "first CSV column only",
            "matching": "case-sensitive exact string equality after trimming surrounding whitespace",
            "raw_API_responses_saved": False,
            "expression_values_labels_predictions_checkpoints_scores_read": False,
            "GPU_used": False,
        },
        "provenance": {},
        "results": {"per_key": []},
    }
    error_code: str | None = None
    try:
        if len(REGISTERED_SANGER_IDS) != 3 or len(set(REGISTERED_SANGER_IDS)) != 3:
            raise AuditStop("REGISTERED_KEY_SET_INVALID")
        input_hashes = verify_inputs()
        receipt["provenance"]["registered_input_sha256"] = input_hashes
        model_rows, _ = model_identity_index()
        rna_keys, rna_freq = axis_summary(RNA_CSV)
        crispr_keys, crispr_freq = axis_summary(CRISPR_CSV)
        axis_keys = {"RNA_axis": (rna_keys, rna_freq), "CRISPR_axis": (crispr_keys, crispr_freq)}
        receipt["provenance"]["axis_unique_key_counts"] = {
            "RNA_axis": len(rna_keys),
            "CRISPR_axis": len(crispr_keys),
        }
        session = requests.Session()
        session.headers.update({"Accept": "application/vnd.api+json"})
        per_key: list[dict[str, Any]] = []
        for sanger_id in REGISTERED_SANGER_IDS:
            try:
                response = session.get(
                    f"{API_BASE}/models/{quote(sanger_id, safe='')}",
                    params={"fields[model]": "identifiers", "include": "identifiers"},
                    timeout=(10, 30),
                )
            except requests.RequestException as exc:
                item = empty_key_result(sanger_id, "REQUEST_ERROR")
                item["failure_code"] = f"API_REQUEST_{type(exc).__name__.upper()}_STOPPED"
                per_key.append(item)
                error_code = item["failure_code"]
                break

            if response.status_code == 404:
                per_key.append(empty_key_result(sanger_id, 404))
                continue
            if response.status_code != 200:
                item = empty_key_result(sanger_id, response.status_code)
                item["failure_code"] = "API_UNEXPECTED_HTTP_STATUS_STOPPED"
                per_key.append(item)
                error_code = item["failure_code"]
                break

            try:
                api_id, identity_values, identifier_count = parse_model_projection(response)
            except AuditStop as exc:
                item = empty_key_result(sanger_id, response.status_code)
                item["failure_code"] = exc.code
                per_key.append(item)
                error_code = exc.code
                break

            candidate_rows = [
                row for row in model_rows
                if any(row[column] and row[column] in identity_values for column in MODEL_COLUMNS)
            ]
            candidates = sorted({row["ModelID"] for row in candidate_rows if row["ModelID"]})
            hit_fields = sorted({
                column
                for row in candidate_rows
                for column in MODEL_COLUMNS
                if row[column] and row[column] in identity_values
            })
            item = {
                "registered_Sanger_ID": sanger_id,
                "http_status": response.status_code,
                "model_objects_returned": 1,
                "CMP_primary_ModelID": api_id,
                "included_model_identifier_record_count": identifier_count,
                "DepMap_24Q2_exact_candidate_ModelIDs": candidates,
                "DepMap_24Q2_unique_ModelID_count": len(candidates) if len(candidates) == 1 else 0,
                "DepMap_24Q2_ambiguous_ModelID_count": len(candidates) if len(candidates) > 1 else 0,
                "Model_csv_exact_hit_fields": hit_fields,
            }
            for axis_name, (keys, frequencies) in axis_keys.items():
                matches = sorted(identity_values & keys)
                item[axis_name] = {
                    "hit": bool(matches),
                    "matched_unique_key_count": len(matches),
                    "matched_row_count": sum(frequencies[key] for key in matches),
                    "matched_keys": matches,
                }
            per_key.append(item)

        receipt["results"]["per_key"] = per_key
        attempted_ids = {row["registered_Sanger_ID"] for row in per_key}
        for sanger_id in REGISTERED_SANGER_IDS:
            if sanger_id not in attempted_ids:
                row = empty_key_result(sanger_id, "NOT_QUERIED_AFTER_STOP")
                row["failure_code"] = error_code or "PRIOR_KEY_FAILURE_STOPPED"
                per_key.append(row)
        receipt["results"]["per_key"] = per_key
        receipt["results"]["query_count"] = sum(row["http_status"] != "NOT_QUERIED_AFTER_STOP" for row in per_key)
        receipt["results"]["http_status_counts"] = dict(sorted(Counter(
            str(row["http_status"]) for row in per_key if row["http_status"] != "NOT_QUERIED_AFTER_STOP"
        ).items()))
        receipt["results"]["model_objects_returned"] = sum(row["model_objects_returned"] for row in per_key)
        receipt["results"]["keys_with_unique_ModelID_match"] = sum(
            row["DepMap_24Q2_unique_ModelID_count"] == 1 for row in per_key
        )
        receipt["results"]["keys_with_ambiguous_ModelID_match"] = sum(
            row["DepMap_24Q2_ambiguous_ModelID_count"] > 0 for row in per_key
        )
        for axis_name in ("RNA_axis", "CRISPR_axis"):
            receipt["results"][axis_name] = {
                "key_hits": sum(bool(row.get(axis_name, {}).get("hit")) for row in per_key),
                "key_misses": sum(
                    row["http_status"] != "NOT_QUERIED_AFTER_STOP" and not bool(row.get(axis_name, {}).get("hit"))
                    for row in per_key
                ),
                "matched_unique_key_count": sum(row.get(axis_name, {}).get("matched_unique_key_count", 0) for row in per_key),
                "matched_row_count": sum(row.get(axis_name, {}).get("matched_row_count", 0) for row in per_key),
            }
        if error_code:
            receipt["status"] = "STOPPED_WITH_FIXED_FAILURE_CODE"
            receipt["failure_code"] = error_code
        else:
            receipt["status"] = "IDENTITY_AXIS_AUDIT_COMPLETE"
    except AuditStop as exc:
        error_code = exc.code
        receipt["status"] = "STOPPED_WITH_FIXED_FAILURE_CODE"
        receipt["failure_code"] = error_code

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "failure_code": receipt.get("failure_code"),
        "receipt": str(OUT_FILE.relative_to(ROOT)).replace("\\", "/"),
        "results": receipt.get("results", {}),
    }, indent=2, ensure_ascii=False))
    return 0 if receipt["status"] == "IDENTITY_AXIS_AUDIT_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
