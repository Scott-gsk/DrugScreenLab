#!/usr/bin/env python3
"""Audit exact chemical identity and scale eligibility for GSE288798 JQ1/MS177."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


GEO_SOFT_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE288nnn/GSE288798/soft/GSE288798_family.soft.gz"
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/Title,InChIKey/JSON"
EXPECTED_GEX = {"control": "GSM8775686", "jq1": "GSM8775688", "ms177": "GSM8775690"}
LINCS_PERT_INFO = Path("data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_pert_info.txt.gz")
LINCS_CELL_INFO = Path("data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_cell_info.txt.gz")
LINCS_CONDITIONS = Path("data/processed/lincs/GSE92742/continuous_conditions_v2/conditions.parquet")
LINCS_CONTINUOUS_VERIFICATION = Path("data/processed/lincs/GSE92742/continuous_conditions_v2/verification.json")
GSE288798_INTAKE = Path("artifacts/experiments/EXP-050/20261001_gse288798_expression_only_v1/intake_summary.json")
GSE288798_VERIFICATION = Path("artifacts/experiments/EXP-050/20261001_gse288798_expression_only_v1/readback_verification.json")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "DrugScreenLab/EXP050 public identity audit"})
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except Exception as exc:  # network transient; preserve the final error if retries fail
            last_error = exc
            if attempt < 3:
                time.sleep(2 ** attempt)
    assert last_error is not None
    raise last_error


def parse_soft_drugs(gzip_bytes: bytes) -> tuple[str, dict[str, str]]:
    text = gzip.decompress(gzip_bytes).decode("utf-8", errors="replace")
    series_pmid = ""
    samples: dict[str, str] = {}
    current_gsm = ""
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("!Series_pubmed_id"):
            series_pmid = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("^SAMPLE ="):
            if current_gsm:
                samples[current_gsm] = "\n".join(lines)
            current_gsm = line.split("=", 1)[1].strip()
            lines = []
        elif current_gsm and (line.startswith("!Sample_title") or line.startswith("!Sample_characteristics_ch1")):
            lines.append(line)
    if current_gsm:
        samples[current_gsm] = "\n".join(lines)
    return series_pmid, samples


def pubchem_lookup(label: str) -> tuple[dict[str, Any], bytes, str]:
    url = PUBCHEM_URL.format(urllib.parse.quote(label, safe=""))
    raw = fetch_bytes(url)
    payload = json.loads(raw.decode("utf-8"))
    records = payload.get("PropertyTable", {}).get("Properties", [])
    if len(records) != 1:
        raise ValueError(f"PubChem exact name lookup for {label} returned {len(records)} records")
    record = records[0]
    if not record.get("InChIKey"):
        raise ValueError(f"PubChem did not return a full InChIKey for {label}")
    return record, raw, url


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty audit output: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path.cwd().resolve()
    run_id = "20261001_gse288798_compound_eligibility_v2"
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    soft_raw = fetch_bytes(GEO_SOFT_URL)
    series_pmid, soft_samples = parse_soft_drugs(soft_raw)
    if not all(gsm in soft_samples for gsm in EXPECTED_GEX.values()):
        raise ValueError("Official GSE288798 SOFT does not contain all three registered GEX GSM samples")
    if not series_pmid:
        raise ValueError("Official GSE288798 series record has no PubMed citation")

    intake = json.loads((root / GSE288798_INTAKE).read_text(encoding="utf-8"))
    readback = json.loads((root / GSE288798_VERIFICATION).read_text(encoding="utf-8"))
    registered_map = intake["official_sample_mapping"]["rds_dataset_to_gsm"]
    if registered_map != {
        "JQ1_24hr": EXPECTED_GEX["jq1"],
        "MS177_24hr": EXPECTED_GEX["ms177"],
        "control_24hr": EXPECTED_GEX["control"],
    }:
        raise ValueError("Existing readback sample map changed; stop before drug/key comparison")
    if not readback["checks"]["summary_output_hashes_match_manifest"]:
        raise ValueError("GSE288798 output read-back hash verification is not PASS")

    lookups = {label: pubchem_lookup(label) for label in ["JQ1", "MS177"]}
    pert = pd.read_csv(root / LINCS_PERT_INFO, sep="\t", dtype=str, low_memory=False).fillna("")
    cells = pd.read_csv(root / LINCS_CELL_INFO, sep="\t", dtype=str, low_memory=False).fillna("")
    conditions = pd.read_parquet(root / LINCS_CONDITIONS)
    continuity = json.loads((root / LINCS_CONTINUOUS_VERIFICATION).read_text(encoding="utf-8"))
    if continuity.get("status") != "PASS":
        raise ValueError("LINCS continuous-condition verification is not PASS")
    if len(conditions) != continuity.get("n_conditions"):
        raise ValueError("LINCS condition table row count no longer matches its verification receipt")

    breast = cells.loc[
        (cells["cell_type"].str.lower() == "cell line")
        & (cells["sample_type"].str.lower() == "tumor")
        & (cells["primary_site"].str.lower() == "breast")
    ].copy()
    breast_ids = set(breast["cell_id"])
    condition_breast = conditions.loc[conditions["cell_id"].isin(breast_ids)].copy()
    condition_breast["pert_dose"] = condition_breast["pert_dose"].astype(str)
    condition_breast["pert_time"] = condition_breast["pert_time"].astype(str)
    cell_meta = breast.set_index("cell_id").to_dict(orient="index")

    drug_rows: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []
    lookup_manifests: dict[str, Any] = {}
    raw_lookup_hashes: dict[str, Any] = {}
    for label in ["JQ1", "MS177"]:
        record, raw, url = lookups[label]
        inchikey = str(record["InChIKey"])
        matches = pert.loc[pert["inchi_key"].str.upper() == inchikey.upper()].copy()
        perturbagen_ids = sorted(matches["pert_id"].unique().tolist())
        selected_conditions = condition_breast.loc[condition_breast["pert_id"].isin(perturbagen_ids)]
        for condition in selected_conditions.to_dict(orient="records"):
            cell = cell_meta[str(condition["cell_id"])]
            exposure_rows.append({
                "target_compound_label": label,
                "pubchem_cid": record.get("CID"),
                "exact_pubchem_inchikey": inchikey,
                "lincs_pert_id": str(condition["pert_id"]),
                "condition_id": str(condition["condition_id"]),
                "cell_id": str(condition["cell_id"]),
                "base_cell_id": str(cell.get("base_cell_id", "")),
                "official_cell_subtype_field": str(cell.get("subtype", "")),
                "dose": str(condition["pert_dose"]),
                "dose_unit": str(condition["pert_dose_unit"]),
                "time": str(condition["pert_time"]),
                "time_unit": str(condition["pert_time_unit"]),
                "n_treatment_instances": int(condition["n_treatment_instances"]),
                "n_matched_control_groups": int(condition["n_control_groups"]),
            })
        gex_gsm = EXPECTED_GEX[label.lower()]
        soft_record = soft_samples[gex_gsm]
        gse_exact_key = bool(len(matches))
        independent_model_identity_gate = bool(gse_exact_key and perturbagen_ids)
        scale_description = intake["selected_assay"]["normalization_provenance"]
        scale_gate = False
        drug_rows.append({
            "target_compound_label": label,
            "gse_accession": "GSE288798",
            "series_pubmed_id": series_pmid,
            "gex_gsm": gex_gsm,
            "gse_sample_record_excerpt": soft_record.replace("\t", " "),
            "pubchem_cid": record.get("CID"),
            "pubchem_title": record.get("Title"),
            "pubchem_inchikey": inchikey,
            "identity_evidence": "Exact unique PubChem name lookup for the GEO drug label; candidate PubChem full InChIKey compared byte-for-byte, case-insensitive, with official LINCS perturbagen InChIKey. GEO does not register a catalog/product key.",
            "matching_lincs_pert_ids": ";".join(perturbagen_ids),
            "exact_key_match_to_lincs_catalog": gse_exact_key,
            "breast_lincs_condition_count": int(len(selected_conditions)),
            "breast_source_base_cell_origins": int(selected_conditions["cell_id"].map(lambda x: cell_meta[x].get("base_cell_id", x)).nunique()) if len(selected_conditions) else 0,
            "breast_source_cell_ids": ";".join(sorted(selected_conditions["cell_id"].unique().tolist())),
            "scale_intake": scale_description,
            "scale_gate_for_cross_platform_functional_comparison": scale_gate,
            "scale_gate_reason": "Deposited Seurat-like RNA.data values are finite, but the normalizer/transform command provenance was not recovered; no scale equivalence to the LINCS continuous L1000 contract is established.",
            "same_drug_source_transfer_eligible": bool(independent_model_identity_gate and scale_gate and len(selected_conditions)),
            "inference_boundary": "One TNBC PDO model only; cells are not independent biological replicates. Identity derives from GEO drug label plus PubChem exact name and LINCS full InChIKey, without manufacturer catalog metadata. No transfer comparison is run unless identity and scale gates both pass.",
        })
        lookup_manifests[label] = {
            "url": url,
            "pubchem_record": record,
            "gse_label_to_pubchem_name_resolved": True,
            "exact_inchikey_in_lincs": gse_exact_key,
            "matched_lincs_pert_ids": perturbagen_ids,
            "breast_condition_count": int(len(selected_conditions)),
        }
        raw_lookup_hashes[label] = {"sha256": sha256(raw), "bytes": len(raw)}

    by_target = {row["target_compound_label"]: row for row in drug_rows}
    gse_exact_match = bool(all(by_target[label]["exact_key_match_to_lincs_catalog"] for label in ["JQ1", "MS177"]))
    scale_pass = bool(all(by_target[label]["scale_gate_for_cross_platform_functional_comparison"] for label in ["JQ1", "MS177"]))
    eligible = bool(gse_exact_match and scale_pass and all(by_target[label]["breast_lincs_condition_count"] > 0 for label in ["JQ1", "MS177"]))

    out = root / "artifacts/experiments/EXP-050" / run_id
    out.mkdir(parents=True, exist_ok=True)
    drug_path = out / "gse288798_compound_identity_scale_eligibility.tsv"
    exposure_path = out / "exact_key_breast_lincs_exposure_coverage.tsv"
    summary_path = out / "summary.json"
    manifest_path = out / "run_manifest.json"
    write_tsv(drug_path, drug_rows)
    if exposure_rows:
        write_tsv(exposure_path, exposure_rows)
    else:
        with exposure_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["target_compound_label", "pubchem_cid", "exact_pubchem_inchikey", "lincs_pert_id", "condition_id", "cell_id", "base_cell_id", "official_cell_subtype_field", "dose", "dose_unit", "time", "time_unit", "n_treatment_instances", "n_matched_control_groups"], delimiter="\t")
            writer.writeheader()

    summary = {
        "experiment_id": "EXP-050",
        "run_id": run_id,
        "status": "completed_drug_identity_and_scale_intake",
        "gse_accession": "GSE288798",
        "gse_pubmed_id": series_pmid,
        "official_geo_soft_url": GEO_SOFT_URL,
        "registered_gex_samples": EXPECTED_GEX,
        "gse_soft_sample_count": len(soft_samples),
        "unique_cell_origin_rule": "Count unique base_cell_id; modified derivatives do not add independent origin count.",
        "official_breast_tumor_cell_line_registry_count": int(len(breast)),
        "lincs_condition_table_verification": {"status": continuity["status"], "condition_count": len(conditions)},
        "compound_identity_crosswalks": lookup_manifests,
        "exact_full_key_gate_pass": gse_exact_match,
        "normalized_scale_gate_pass": scale_pass,
        "cross_platform_descriptive_comparison_eligible": eligible,
        "decision": "Eligible" if eligible else "Do not run a cross-study functional comparison yet; the registered GSE288798 RNA.data normalization/transform provenance is unresolved, so the scale gate fails even if the exact chemical key resolves.",
        "scale_source": intake["selected_assay"]["normalization_provenance"],
        "scope": "Public GEO sample-label to PubChem name lookup and PubChem full InChIKey to LINCS exact-key/eligible-condition audit. No reprocessing of the 2.9-GB RDS, expression-value comparison, target prediction, efficacy claim, or external contact.",
        "gpu_hours": 0,
        "outputs": {
            "drug_table": drug_path.relative_to(root).as_posix(),
            "exposure_coverage": exposure_path.relative_to(root).as_posix(),
            "summary": summary_path.relative_to(root).as_posix(),
            "manifest": manifest_path.relative_to(root).as_posix(),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    local_inputs = [
        root / LINCS_PERT_INFO,
        root / LINCS_CELL_INFO,
        root / LINCS_CONDITIONS,
        root / LINCS_CONTINUOUS_VERIFICATION,
        root / GSE288798_INTAKE,
        root / GSE288798_VERIFICATION,
    ]
    manifest = {
        "experiment_id": "EXP-050",
        "run_id": run_id,
        "started_at_local": started,
        "finished_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": f"conda run --no-capture-output -n drugscreening-gpu python scripts/evaluation/audit_exp050_gse288798_compound_eligibility.py",
        "environment": "WSL2 Conda drugscreening-gpu",
        "gpu_hours": 0,
        "network_sources": {
            "NCBI_GEO_SOFT": {"url": GEO_SOFT_URL, "sha256": sha256(soft_raw), "bytes": len(soft_raw)},
            "PubChem_PUG_REST": raw_lookup_hashes,
        },
        "inputs": {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in local_inputs},
        "official_breast_cell_lines": breast[["cell_id", "base_cell_id", "subtype"]].to_dict(orient="records"),
        "output_hashes": {
            p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in [drug_path, exposure_path, summary_path]
        },
        "code": {"path": "scripts/evaluation/audit_exp050_gse288798_compound_eligibility.py", "sha256": sha256_file(Path(__file__).resolve())},
        "exit_code": 0,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
