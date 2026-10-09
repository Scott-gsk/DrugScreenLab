"""Audit open GDC TCGA-COAD/READ STAR-count file-to-sample identity before download."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ENDPOINT = "https://api.gdc.cancer.gov/files"
FIELDS = ",".join((
    "file_id", "file_name", "md5sum", "file_size", "access", "data_type",
    "analysis.workflow_type", "associated_entities.entity_submitter_id",
    "associated_entities.entity_type", "cases.submitter_id", "cases.project.project_id",
    "cases.samples.submitter_id", "cases.samples.sample_type",
    "cases.samples.portions.analytes.aliquots.submitter_id",
))


def fetch() -> tuple[dict, bytes]:
    filters = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id",
                                  "value": ["TCGA-COAD", "TCGA-READ"]}},
        {"op": "=", "content": {"field": "data_type",
                                "value": "Gene Expression Quantification"}},
        {"op": "=", "content": {"field": "analysis.workflow_type",
                                "value": "STAR - Counts"}},
        {"op": "=", "content": {"field": "access", "value": "open"}},
    ]}
    parameters = urllib.parse.urlencode({"filters": json.dumps(filters, separators=(",", ":")),
                                         "fields": FIELDS, "format": "JSON", "size": 3000})
    url = ENDPOINT + "?" + parameters
    request = urllib.request.Request(url, headers={"User-Agent": "DrugScreenLab/EXP-038"})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
    return json.loads(payload), payload


def resolve_sample(file: dict) -> dict:
    cases = file.get("cases", [])
    if len(cases) != 1:
        return {"identity_status": "CASE_COUNT_NOT_ONE"}
    case = cases[0]
    linked = {entity.get("entity_submitter_id") for entity in file.get("associated_entities", [])
              if entity.get("entity_type", "").lower() == "aliquot"}
    matches = []
    for sample in case.get("samples", []):
        for portion in sample.get("portions", []):
            for analyte in portion.get("analytes", []):
                for aliquot in analyte.get("aliquots", []):
                    if aliquot.get("submitter_id") in linked:
                        matches.append((sample, aliquot))
    unique = {(sample.get("submitter_id"), sample.get("sample_type"),
               aliquot.get("submitter_id")) for sample, aliquot in matches}
    if len(unique) != 1:
        return {"identity_status": "FILE_ALIQUOT_SAMPLE_LINK_NOT_UNIQUE",
                "case_id": case.get("submitter_id"), "linked_aliquots": sorted(x for x in linked if x)}
    sample_id, sample_type, aliquot_id = next(iter(unique))
    return {"identity_status": "LINKED", "case_id": case.get("submitter_id"),
            "project_id": case.get("project", {}).get("project_id"),
            "sample_id": sample_id, "sample_type": sample_type, "aliquot_id": aliquot_id}


def audit(output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"availability output already exists: {output_dir}")
    response, raw_payload = fetch()
    hits = response["data"]["hits"]
    total = response["data"]["pagination"]["total"]
    if len(hits) != total or len({item["file_id"] for item in hits}) != total:
        raise ValueError("GDC response is incomplete or repeats a file")
    inventory = []
    for item in hits:
        identity = resolve_sample(item)
        inventory.append({"file_id": item["file_id"], "file_name": item.get("file_name"),
                          "md5sum": item.get("md5sum"), "file_size": item.get("file_size"),
                          "access": item.get("access"), "workflow": item.get("analysis", {}).get("workflow_type"),
                          **identity})
    linked = [item for item in inventory if item["identity_status"] == "LINKED"]
    by_case = defaultdict(set)
    for item in linked:
        by_case[item["case_id"]].add(item["sample_type"])
    paired = sorted(case for case, types in by_case.items()
                    if {"Primary Tumor", "Solid Tissue Normal"} <= types)
    report = {"experiment_id": "EXP-038", "status": "GDC_METADATA_AVAILABILITY_ONLY",
              "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
              "endpoint": ENDPOINT, "filters": "open STAR - Counts Gene Expression Quantification from TCGA-COAD/READ",
              "gdc_response_sha256": hashlib.sha256(raw_payload).hexdigest(),
              "n_files": total, "n_linked_files": len(linked),
              "identity_status_counts": dict(Counter(item["identity_status"] for item in inventory)),
              "sample_type_counts": dict(Counter(item["sample_type"] for item in linked)),
              "n_cases_with_primary_tumor_and_solid_tissue_normal": len(paired),
              "paired_case_ids": paired,
              "linked_file_bytes": sum(int(item["file_size"]) for item in linked),
              "download_status": "NOT_STARTED"}
    output_dir.mkdir(parents=True)
    (output_dir / "gdc_response.json").write_bytes(raw_payload)
    (output_dir / "file_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
                                                    encoding="utf-8")
    (output_dir / "availability.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.output_dir)
    print(json.dumps({key: value for key, value in result.items() if key != "paired_case_ids"},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
