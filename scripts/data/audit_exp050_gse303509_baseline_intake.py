#!/usr/bin/env python3
"""Audit the public GSE303509 baseline RNA-seq archive without extracting it."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASET_ID = "gse303509_bui_breast_pdo_rnaseq_raw_v1"
SOFT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE303nnn/GSE303509/"
    "soft/GSE303509_family.soft.gz"
)
EXPECTED_HEADER = [
    "gene_id",
    "transcript_id(s)",
    "length",
    "effective_length",
    "expected_count",
    "TPM",
    "FPKM",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_soft(path: Path) -> tuple[dict[str, list[str]], list[dict[str, list[str]]]]:
    text = gzip.open(path, "rt", encoding="utf-8", errors="replace").read()
    series: dict[str, list[str]] = {}
    for line in text.splitlines():
        if line.startswith("!Series_"):
            key, value = line.split("=", 1)
            series.setdefault(key.strip()[len("!Series_") :], []).append(value.strip())

    samples: list[dict[str, list[str]]] = []
    for block in text.split("^"):
        if not block.startswith("SAMPLE"):
            continue
        record: dict[str, list[str]] = {}
        for line in block.splitlines():
            if line.startswith("!Sample_"):
                key, value = line.split("=", 1)
                record.setdefault(key.strip()[len("!Sample_") :], []).append(
                    value.strip()
                )
        samples.append(record)
    return series, samples


def first(record: dict[str, list[str]], key: str) -> str:
    values = record.get(key, [])
    return values[0] if values else ""


def list_text(record: dict[str, list[str]], key: str) -> str:
    return " | ".join(record.get(key, []))


def annotation_for(record: dict[str, list[str]]) -> str:
    processing = " | ".join(record.get("data_processing", []))
    if re.search(r"GENCODE\s*49", processing, re.I):
        return "GENCODE49_GRCh38.p14"
    if re.search(r"Ensemble_75", processing, re.I):
        return "Ensembl75"
    return "unknown"


def parse_member(data: bytes, member_name: str) -> tuple[dict[str, Any], set[str], set[str]]:
    stable_ids: set[str] = set()
    exact_ids: set[str] = set()
    exact_duplicates = 0
    stable_duplicates = 0
    row_count = 0
    detected_count = 0
    detected_tpm = 0
    expected_count_sum = 0.0
    tpm_sum = 0.0
    invalid_numeric = 0
    negative_numeric = 0
    versioned_ids = 0

    with gzip.open(io.BytesIO(data), "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_HEADER:
            raise ValueError(f"Unexpected RSEM header in {member_name}: {reader.fieldnames}")
        for row in reader:
            row_count += 1
            gene_id = (row.get("gene_id") or "").strip()
            if not gene_id:
                invalid_numeric += 1
                continue
            stable_id = gene_id.split(".", 1)[0]
            if gene_id in exact_ids:
                exact_duplicates += 1
            if stable_id in stable_ids:
                stable_duplicates += 1
            exact_ids.add(gene_id)
            stable_ids.add(stable_id)
            if "." in gene_id:
                versioned_ids += 1
            try:
                expected_count = float(row["expected_count"])
                tpm = float(row["TPM"])
                if not (math.isfinite(expected_count) and math.isfinite(tpm)):
                    invalid_numeric += 1
                    continue
                if expected_count < 0 or tpm < 0:
                    negative_numeric += 1
                    continue
            except (TypeError, ValueError, KeyError):
                invalid_numeric += 1
                continue
            expected_count_sum += expected_count
            tpm_sum += tpm
            detected_count += expected_count > 0
            detected_tpm += tpm > 0

    return (
        {
            "member": member_name,
            "compressed_member_sha256": sha256_bytes(data),
            "gene_rows": row_count,
            "unique_exact_gene_ids": len(exact_ids),
            "unique_version_stripped_gene_ids": len(stable_ids),
            "duplicate_exact_gene_id_rows": exact_duplicates,
            "duplicate_version_stripped_gene_id_rows": stable_duplicates,
            "gene_ids_with_version_suffix": versioned_ids,
            "detected_genes_expected_count_gt_zero": detected_count,
            "detected_genes_tpm_gt_zero": detected_tpm,
            "expected_count_sum_rsem_estimates": expected_count_sum,
            "TPM_sum": tpm_sum,
            "invalid_numeric_rows": invalid_numeric,
            "negative_numeric_rows": negative_numeric,
        },
        exact_ids,
        stable_ids,
    )


def write_tsv(path: Path, records: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--soft-gz", type=Path, required=True)
    parser.add_argument("--run-id", default="20261001_gse303509_baseline_intake_v1")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    started = datetime.now(timezone.utc).isoformat()
    registry = json.loads((root / "data/registry/datasets.json").read_text(encoding="utf-8"))
    dataset = next(item for item in registry if item.get("id") == DATASET_ID)
    raw_tar = root / "data" / dataset["path"]["relative"]
    expected_sha = dataset["checksum"]["value"]
    expected_bytes = int(dataset["checksum"]["bytes"])
    if raw_tar.stat().st_size != expected_bytes or sha256_file(raw_tar) != expected_sha:
        raise ValueError("GSE303509 raw archive does not match its registered size/SHA-256")

    soft_path = args.soft_gz.resolve()
    soft_sha = sha256_file(soft_path)
    series, sample_records = read_soft(soft_path)
    if first(series, "geo_accession") != "GSE303509":
        raise ValueError("Family SOFT accession does not match GSE303509")

    sample_by_gsm: dict[str, dict[str, list[str]]] = {}
    for record in sample_records:
        gsm = first(record, "geo_accession")
        if not gsm or gsm in sample_by_gsm:
            raise ValueError(f"Missing or duplicate GSM in SOFT metadata: {gsm!r}")
        sample_by_gsm[gsm] = record

    qc_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    exact_gene_sets: list[set[str]] = []
    stable_gene_sets: list[set[str]] = []
    seen_gsm: set[str] = set()
    with tarfile.open(raw_tar, "r:") as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        for member in members:
            basename = Path(member.name).name
            match = re.match(r"(GSM\d+)_([^/]+)\.genes\.csv\.gz$", basename)
            if not match:
                raise ValueError(f"Unexpected non-count file in raw archive: {member.name}")
            gsm, label = match.groups()
            if gsm not in sample_by_gsm or gsm in seen_gsm:
                raise ValueError(f"GSM missing from SOFT or repeated in archive: {gsm}")
            seen_gsm.add(gsm)
            data_stream = archive.extractfile(member)
            if data_stream is None:
                raise ValueError(f"Unable to read archive member: {member.name}")
            compressed_data = data_stream.read()
            qc, exact_ids, stable_ids = parse_member(compressed_data, member.name)
            record = sample_by_gsm[gsm]
            torg_match = re.search(r"\b(TORG\d+)\b", first(record, "title"), re.I)
            torg = torg_match.group(1).upper() if torg_match else ""
            characteristics = list_text(record, "characteristics_ch1")
            treatment_protocol = list_text(record, "treatment_protocol")
            has_treatment_characteristic = bool(re.search(r"\btreatment\s*:", characteristics, re.I))
            batch_match = re.search(r"\bbatch\s*:\s*([^|]+)", characteristics, re.I)
            qc.update(
                {
                    "gsm": gsm,
                    "torg_label": torg,
                    "batch": batch_match.group(1).strip() if batch_match else "",
                    "annotation": annotation_for(record),
                }
            )
            qc_rows.append(qc)
            manifest_rows.append(
                {
                    "gsm": gsm,
                    "torg_label": torg,
                    "title": first(record, "title"),
                    "batch": qc["batch"],
                    "annotation": qc["annotation"],
                    "library_description": list_text(record, "description"),
                    "characteristics_ch1": characteristics,
                    "treatment_protocol_field_present": "treatment_protocol" in record,
                    "treatment_protocol": treatment_protocol,
                    "treatment_characteristic_present": has_treatment_characteristic,
                    "raw_member": member.name,
                    "raw_member_sha256": qc["compressed_member_sha256"],
                }
            )
            exact_gene_sets.append(exact_ids)
            stable_gene_sets.append(stable_ids)

    if len(sample_records) != int(dataset["schema"]["rna_seq_samples"]):
        raise ValueError("SOFT sample count differs from registered expected count")
    if seen_gsm != set(sample_by_gsm):
        raise ValueError("Archive/SOFT sample accession sets differ")

    exact_all = set.intersection(*exact_gene_sets) if exact_gene_sets else set()
    stable_all = set.intersection(*stable_gene_sets) if stable_gene_sets else set()
    annotation_counts: dict[str, int] = {}
    batch_counts: dict[str, int] = {}
    for row in manifest_rows:
        annotation_counts[row["annotation"]] = annotation_counts.get(row["annotation"], 0) + 1
        batch_counts[row["batch"]] = batch_counts.get(row["batch"], 0) + 1

    output_dir = root / "artifacts/experiments/EXP-050" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "sample_manifest.tsv"
    qc_path = output_dir / "expression_file_qc.tsv"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "run_manifest.json"
    write_tsv(
        sample_path,
        manifest_rows,
        [
            "gsm", "torg_label", "title", "batch", "annotation", "library_description",
            "characteristics_ch1", "treatment_protocol_field_present", "treatment_protocol",
            "treatment_characteristic_present", "raw_member", "raw_member_sha256",
        ],
    )
    write_tsv(
        qc_path,
        qc_rows,
        [
            "gsm", "torg_label", "batch", "annotation", "member", "compressed_member_sha256",
            "gene_rows", "unique_exact_gene_ids", "unique_version_stripped_gene_ids",
            "duplicate_exact_gene_id_rows", "duplicate_version_stripped_gene_id_rows",
            "gene_ids_with_version_suffix", "detected_genes_expected_count_gt_zero",
            "detected_genes_tpm_gt_zero", "expected_count_sum_rsem_estimates", "TPM_sum",
            "invalid_numeric_rows", "negative_numeric_rows",
        ],
    )

    summary = {
        "run_id": args.run_id,
        "dataset_id": DATASET_ID,
        "accession": "GSE303509",
        "series_title": first(series, "title"),
        "series_summary": first(series, "summary"),
        "series_overall_design": first(series, "overall_design"),
        "sample_count": len(manifest_rows),
        "unique_gsm_count": len(seen_gsm),
        "unique_torg_labels": len({r["torg_label"] for r in manifest_rows if r["torg_label"]}),
        "raw_count_files": len(qc_rows),
        "annotation_sample_counts": annotation_counts,
        "batch_sample_counts": batch_counts,
        "samples_with_treatment_protocol_field": sum(bool(r["treatment_protocol_field_present"]) for r in manifest_rows),
        "samples_with_nonempty_treatment_protocol": sum(bool(r["treatment_protocol"]) for r in manifest_rows),
        "samples_with_treatment_characteristic": sum(bool(r["treatment_characteristic_present"]) for r in manifest_rows),
        "common_exact_gene_ids_all_samples": len(exact_all),
        "common_version_stripped_gene_ids_all_samples": len(stable_all),
        "all_gene_files_have_unique_exact_ids": all(r["duplicate_exact_gene_id_rows"] == 0 for r in qc_rows),
        "all_gene_files_have_finite_nonnegative_expression_fields": all(
            r["invalid_numeric_rows"] == 0 and r["negative_numeric_rows"] == 0 for r in qc_rows
        ),
        "expression_contract": "RSEM expected_count estimates, TPM, and FPKM; expected_count values are estimated counts, not direct FASTQ read counts.",
        "interpretation": "GEO provides 14 organoid RNA-seq sample files and no drug-response panel. GEO sample records have no nonempty treatment protocol or treatment characteristic. The files are suitable only for baseline-state context pending sample-level treatment confirmation; do not treat them as drug-induced transcriptional responses or paired PDO efficacy labels.",
        "annotation_boundary": "The 12 Ensembl75 and 2 GENCODE49/GRCh38.p14 samples remain stratified. Shared stable gene identifiers do not harmonize different reference/annotation quantification pipelines.",
        "inference_boundary": "No model training, patient prediction, drug ranking, target-treated expression access, or locked_eval access occurred.",
        "gpu_hours": 0,
        "evidence_files": [sample_path.relative_to(root).as_posix(), qc_path.relative_to(root).as_posix()],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    outputs = [sample_path, qc_path, summary_path]
    run_manifest = {
        "run_id": args.run_id,
        "dataset_id": DATASET_ID,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "conda run --no-capture-output -n drugscreening-gpu python scripts/data/audit_exp050_gse303509_baseline_intake.py --soft-gz tmp/exp050_gse303509_intake/GSE303509_family.soft.gz",
        "gpu_hours": 0,
        "inputs": {
            str(raw_tar.relative_to(root).as_posix()): {"sha256": expected_sha, "bytes": expected_bytes},
            "NCBI_GEO_family_SOFT": {"url": SOFT_URL, "sha256": soft_sha, "bytes": soft_path.stat().st_size},
        },
        "code": {"path": "scripts/data/audit_exp050_gse303509_baseline_intake.py", "sha256": sha256_file(Path(__file__))},
        "outcome": "completed_metadata_and_RSEM_structure_QC",
        "outputs": {
            p.relative_to(root).as_posix(): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in outputs
        },
        "exit_code": 0,
    }
    manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
