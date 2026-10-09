#!/usr/bin/env python3
"""Link published TNBC TORG VP-response cases to GSE303509 baseline RNA samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NCBI_EFETCH = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=pmc&id=PMC13522806&retmode=xml"
)
EXPECTED_PMID = "42561946"
EXPECTED_PMCID = "PMC13522806"
EXPECTED_DOI = "10.1016/j.xcrm.2026.102973"
EXPECTED_TITLE = "Biomarker-guided responses in patient-derived organoids predict effective therapies in breast cancer"


CASES = {
    "TORG40": {
        "predicted_vp_group": "resistant",
        "reported_in_vitro_response": "resistant to cisplatin monotherapy and veliparib-platinum combination",
        "observed_ordinal_group": "resistant",
        "article_evidence": "Figure 4E-F and Results text",
    },
    "TORG58": {
        "predicted_vp_group": "intermediate",
        "reported_in_vitro_response": "resistant to cisplatin monotherapy and veliparib-platinum combination",
        "observed_ordinal_group": "resistant",
        "article_evidence": "Figure 4E-F and Results text",
    },
    "TORG139": {
        "predicted_vp_group": "intermediate",
        "reported_in_vitro_response": "intermediate in vitro response",
        "observed_ordinal_group": "intermediate",
        "article_evidence": "Figure 4E-F and Results text",
    },
    "TORG146": {
        "predicted_vp_group": "sensitive",
        "reported_in_vitro_response": "sensitive to cisplatin with or without veliparib",
        "observed_ordinal_group": "sensitive",
        "article_evidence": "Figure 4E-F and Results text",
    },
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_name(tag: str) -> str:
    return tag.split("}")[-1]


def compact(node: ET.Element) -> str:
    return " ".join("".join(node.itertext()).split())


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parse_soft_series(path: Path) -> dict[str, str]:
    import gzip

    series: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Series_"):
                key, value = line.split("=", 1)
                series[key.strip()[len("!Series_") :]] = value.strip()
    return series


def article_metadata(xml_bytes: bytes) -> tuple[dict[str, Any], str, str, str]:
    root = ET.fromstring(xml_bytes)
    article = next((node for node in root.iter() if local_name(node.tag) == "article"), None)
    if article is None:
        raise ValueError("NCBI EFetch response has no PMC article node")
    meta = next((node for node in article if local_name(node.tag) == "front"), None)
    if meta is None:
        raise ValueError("PMC article has no front matter")
    article_meta = next((node for node in meta if local_name(node.tag) == "article-meta"), None)
    journal_meta = next((node for node in meta if local_name(node.tag) == "journal-meta"), None)
    if article_meta is None or journal_meta is None:
        raise ValueError("PMC article metadata is incomplete")

    identifiers: dict[str, str] = {}
    for node in article_meta:
        if local_name(node.tag) == "article-id":
            identifiers[node.attrib.get("pub-id-type", "unknown")] = compact(node)
    title_node = next((node for node in article_meta.iter() if local_name(node.tag) == "article-title"), None)
    title = compact(title_node) if title_node is not None else ""
    journal_title = next((compact(node) for node in journal_meta.iter() if local_name(node.tag) == "journal-title"), "")
    abstract_node = next((node for node in article_meta if local_name(node.tag) == "abstract"), None)
    abstract = compact(abstract_node) if abstract_node is not None else ""
    body = next((node for node in article if local_name(node.tag) == "body"), None)
    if body is None:
        raise ValueError("PMC article has no body")
    body_text = compact(body)
    resource_section = ""
    for section in body.iter():
        if local_name(section.tag) != "sec":
            continue
        section_title = next((compact(c) for c in section if local_name(c.tag) == "title"), "")
        if section_title.lower() == "data and code availability":
            resource_section = compact(section)
            break

    figures: dict[str, str] = {}
    for node in body.iter():
        if local_name(node.tag) != "fig":
            continue
        label = next((compact(c) for c in node if local_name(c.tag) == "label"), "")
        if label in {"Figure 4", "Figure 4."}:
            figures[label] = compact(node)

    record = {
        "pmid": identifiers.get("pmid", ""),
        # PMC's JATS uses `pub-id-type="pmcid"` in the current EFetch record.
        "pmcid": identifiers.get("pmcid", identifiers.get("pmc", "")),
        "doi": identifiers.get("doi", ""),
        "title": title,
        "journal": journal_title,
        "abstract": abstract,
    }
    return record, body_text, resource_section, " | ".join(figures.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--soft-gz", type=Path, required=True)
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        default=Path("artifacts/experiments/EXP-050/20261001_gse303509_baseline_intake_v1/sample_manifest.tsv"),
    )
    parser.add_argument(
        "--intake-summary",
        type=Path,
        default=Path("artifacts/experiments/EXP-050/20261001_gse303509_baseline_intake_v1/summary.json"),
    )
    parser.add_argument("--run-id", default="20261001_gse303509_published_case_crosswalk_v1")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    started = datetime.now(timezone.utc).isoformat()
    soft_path = args.soft_gz.resolve()
    sample_path = root / args.sample_manifest
    intake_summary_path = root / args.intake_summary
    soft = parse_soft_series(soft_path)
    if soft.get("geo_accession") != "GSE303509" or soft.get("pubmed_id") != EXPECTED_PMID:
        raise ValueError("GSE303509 SOFT citation does not match the expected final article PMID")

    samples = read_tsv(sample_path)
    sample_by_torg = {row["torg_label"]: row for row in samples if row.get("torg_label")}
    if len(sample_by_torg) != len(samples):
        raise ValueError("Sample manifest contains missing or duplicate TORG labels")
    missing = sorted(set(CASES) - set(sample_by_torg))
    if missing:
        raise ValueError(f"Published case labels do not map to GSE303509 sample titles: {missing}")

    request = urllib.request.Request(
        NCBI_EFETCH,
        headers={"User-Agent": "DrugScreenLab/EXP050 public-literature-crosswalk"},
    )
    with urllib.request.urlopen(request, timeout=40) as response:
        xml_bytes = response.read()
    article, body_text, resource_section, figure4 = article_metadata(xml_bytes)
    if article["pmid"] != EXPECTED_PMID or article["pmcid"] != EXPECTED_PMCID or article["doi"] != EXPECTED_DOI:
        raise ValueError(f"Unexpected final article identity: {article}")
    if article["title"] != EXPECTED_TITLE:
        raise ValueError(f"Unexpected published article title: {article['title']}")
    if "GSE303509" not in resource_section or "RSEM" not in resource_section:
        raise ValueError("Published data-availability statement did not confirm GSE303509 RSEM deposit")
    for label in CASES:
        if label not in body_text or label not in figure4:
            raise ValueError(f"Published Figure 4/article body does not contain expected case {label}")
    if "three biological replicates per organoid" not in figure4:
        raise ValueError("Figure 4 legend did not confirm its stated biological replicate count")

    case_rows: list[dict[str, Any]] = []
    for torg, case in CASES.items():
        sample = sample_by_torg[torg]
        case_rows.append(
            {
                "torg_label": torg,
                "gsm": sample["gsm"],
                "baseline_rna_file": sample["raw_member"],
                "baseline_annotation": sample["annotation"],
                "published_model_group": case["predicted_vp_group"],
                "published_in_vitro_group": case["observed_ordinal_group"],
                "exact_ordinal_group_match": case["predicted_vp_group"] == case["observed_ordinal_group"],
                "reported_in_vitro_response": case["reported_in_vitro_response"],
                "published_evidence_location": case["article_evidence"],
                "independent_external_validation": False,
                "treated_rna_available_in_GSE303509": False,
            }
        )

    output_dir = root / "artifacts/experiments/EXP-050" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    crosswalk_path = output_dir / "published_case_response_crosswalk.tsv"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "run_manifest.json"
    columns = list(case_rows[0].keys())
    with crosswalk_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(case_rows)

    intake_summary = json.loads(intake_summary_path.read_text(encoding="utf-8"))
    data_availability = resource_section or ""
    exact_matches = sum(row["exact_ordinal_group_match"] for row in case_rows)
    summary = {
        "run_id": args.run_id,
        "series_accession": "GSE303509",
        "series_pubmed_id": soft["pubmed_id"],
        "publication": article,
        "publication_status": "peer-reviewed Cell Reports Medicine article; PubMed lists 2026 Aug 18, e-published 2026 Aug 6",
        "gse_expression_profiles": intake_summary["sample_count"],
        "case_profiles_crosswalked_by_exact_TORG_title": len(case_rows),
        "case_crosswalk": [{k: r[k] for k in ["torg_label", "gsm", "published_model_group", "published_in_vitro_group", "exact_ordinal_group_match"]} for r in case_rows],
        "published_cell_viability_replicates": "Figure 4E states three biological replicates per organoid, with three technical replicates per biological replicate.",
        "descriptive_exact_ordinal_matches": exact_matches,
        "descriptive_exact_ordinal_denominator": len(case_rows),
        "interpretation": "Three of four selected same-paper cases have matching ordinal labels; TORG58 was predicted intermediate but described as resistant in vitro. This is a descriptive crosswalk of the authors' own selected validation cases, not an independent estimate of model accuracy or of EXP-050 transfer performance.",
        "mainline_boundary": "The linked GEO files are baseline RSEM profiles only. They contain no treated RNA, and this article crosswalk provides drug-response phenotype context only; it cannot evaluate drug-induced transcription-response migration.",
        "data_availability_statement": data_availability,
        "response_data_boundary": "GSE303509 publicly deposits RNA-seq/processed RSEM. The NCBI GEO TAR has no PDO-by-drug response panel. Article figures/text report selected-case summaries, while additional reanalysis information is available from the lead contact upon request; no contact was made and no figure digitization/raw replicate value was performed.",
        "limitations": [
            "Only four hand-selected validation organoids are crosswalked; they are not a random or independent validation cohort.",
            "Exact culture-label matching establishes the GEO sample label correspondence, not an independently verified patient-level identity.",
            "The reported outcomes come from the same publication that developed the clinical-biomarker prediction and are used only as a direct-sensitivity reference.",
            "The article's selected treatment outcomes are not paired post-treatment RNA measurements, so no transcription-response migration statistic can be computed from this crosswalk.",
        ],
        "gpu_hours": 0,
        "output": crosswalk_path.relative_to(root).as_posix(),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output_files = [crosswalk_path, summary_path]
    run_manifest = {
        "run_id": args.run_id,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "conda run --no-capture-output -n drugscreening-gpu python scripts/evaluation/audit_exp050_gse303509_published_case_crosswalk.py --soft-gz tmp/exp050_gse303509_intake/GSE303509_family.soft.gz",
        "gpu_hours": 0,
        "inputs": {
            "official_NCBI_EFetch_PMC_XML": {"url": NCBI_EFETCH, "sha256": sha256_bytes(xml_bytes), "bytes": len(xml_bytes)},
            str(soft_path.relative_to(root).as_posix()): {"sha256": sha256_file(soft_path), "bytes": soft_path.stat().st_size},
            sample_path.relative_to(root).as_posix(): {"sha256": sha256_file(sample_path), "bytes": sample_path.stat().st_size},
            intake_summary_path.relative_to(root).as_posix(): {"sha256": sha256_file(intake_summary_path), "bytes": intake_summary_path.stat().st_size},
        },
        "article_identity": {k: article[k] for k in ["pmid", "pmcid", "doi", "title", "journal"]},
        "article_xml_retention": "Parsed in memory from official NCBI EFetch; the full text XML is not copied into the experiment artifact.",
        "protocol_boundary": "Exact TORG-label metadata crosswalk and literature-text extraction only; no RNA expression values, drug-response measurements, model fitting, prediction, ranking, contact, or figure digitization.",
        "code": {"path": "scripts/evaluation/audit_exp050_gse303509_published_case_crosswalk.py", "sha256": sha256_file(Path(__file__))},
        "outcome": "completed_published_four_case_crosswalk",
        "outputs": {
            path.relative_to(root).as_posix(): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in output_files
        },
        "exit_code": 0,
    }
    manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
