"""Quantify unique perturbations, controls, donor groups and exact978 overlap.

Local files only.  Does not invent counts when a field is absent.
"""

from __future__ import annotations

from collections import defaultdict
import gzip
import json
from pathlib import Path
import re
import sys
import tarfile
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from drug_screen.data.lincs_landmarks import ORDERED_GENE_IDS_SHA256

GENE_INFO = ROOT / "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_gene_info.txt.gz"
FETCH = ROOT / "artifacts/experiments/EXP-007/ORGANOID_GEO_FETCH.json"
OUT = ROOT / "artifacts/experiments/EXP-007/ORGANOID_GENETIC_INVENTORY.json"
EXISTING = ROOT / "mvp/extension/ORGANOID_DATASET_READINESS_AUDIT.json"
_SAMPLE = re.compile(r"^\^SAMPLE = (?P<gsm>GSM\d+)\s*$")
_FIELD = re.compile(r"^!(?P<key>[^=]+)=(?P<value>.*)$")


def _soft_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            sample_match = _SAMPLE.match(line)
            if sample_match:
                if current:
                    samples.append(current)
                current = {"gsm": sample_match.group("gsm"), "fields": defaultdict(list)}
                continue
            if current is None:
                continue
            field = _FIELD.match(line)
            if not field:
                continue
            current["fields"][field.group("key").strip()].append(field.group("value").strip())
        if current:
            samples.append(current)
    rows = []
    for sample in samples:
        fields = sample["fields"]
        rows.append(
            {
                "gsm": sample["gsm"],
                "title": (fields.get("Sample_title") or [None])[0],
                "characteristics": list(fields.get("Sample_characteristics_ch1") or []),
                "source": (fields.get("Sample_source_name_ch1") or [None])[0],
                "platform": (fields.get("Sample_platform_id") or [None])[0],
            }
        )
    return rows


def _char(sample: MappingLike, prefix: str) -> str | None:
    for item in sample.get("characteristics") or []:
        if item.lower().startswith(prefix.lower()):
            return item.split(":", 1)[1].strip()
    return None


MappingLike = dict[str, Any]


def _landmark_symbols() -> tuple[set[str], dict[str, str]]:
    gene_info = pd.read_csv(GENE_INFO, sep="\t", dtype=str)
    landmarks = gene_info.loc[gene_info["pr_is_lm"].astype(str) == "1"]
    id_to_symbol = dict(zip(landmarks["pr_gene_id"].astype(str), landmarks["pr_gene_symbol"].astype(str)))
    symbols = {str(symbol).upper() for symbol in landmarks["pr_gene_symbol"]}
    return symbols, id_to_symbol


def _overlap_from_symbols(symbols: set[str], landmarks: set[str]) -> dict[str, Any]:
    mapped = sorted(symbol for symbol in landmarks if symbol in symbols)
    missing = sorted(symbol for symbol in landmarks if symbol not in symbols)
    return {
        "method": "symbol_upper_vs_GSE92742_pr_gene_symbol",
        "landmark_genes": 978,
        "mapped": len(mapped),
        "missing": len(missing),
        "missing_symbols_head": missing[:20],
        "status": "EXACT978_COUNTED_BY_SYMBOL",
        "caveat": "symbol join, not Entrez; not a frozen 978 adapter",
    }


def _audit_gse280506(landmarks: set[str]) -> dict[str, Any]:
    soft = ROOT / "data/raw/geo/GSE280506/GSE280506_family.soft.gz"
    identities = pd.read_csv(ROOT / "data/raw/geo/GSE280506/GSE280506_cell_identities.csv.gz")
    features = pd.read_csv(ROOT / "data/raw/geo/GSE280506/GSE280506_features.tsv.gz", sep="\t", header=None)
    samples = _soft_samples(soft)
    assigned = identities.loc[identities["guide_identity"].astype(str) != "*"].copy()
    assigned["target"] = assigned["guide_identity"].astype(str).str.split("_").str[0]
    assigned["control_like"] = assigned["guide_identity"].astype(str).str.contains(
        r"NTC|CTRL|CONTROL|NEG|NON.?TARGET", case=False, regex=True
    )
    gem_to_sample = {
        1: "CRISPRi_DMSO",
        2: "CRISPRi_DMSO",
        3: "CRISPRi_DMSO",
        4: "CRISPRi_Cisplatin",
        5: "CRISPRi_Cisplatin",
        6: "CRISPRi_Cisplatin",
        7: "CRISPRa_DMSO",
        8: "CRISPRa_Cisplatin",
    }
    identities["sample_group"] = identities["gemgroup"].map(gem_to_sample)
    symbols = {str(value).upper() for value in features.iloc[:, 1].dropna()}
    return {
        "accession": "GSE280506",
        "official_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE280506",
        "role": "organoid_genetic_adaptation_reference",
        "organism": "Homo sapiens",
        "platform": "GPL24676",
        "modality": "scRNA-seq / CROP-seq",
        "design": "TP53/APC KO primary human gastric organoid; CRISPRi/CRISPRa; DMSO/cisplatin",
        "local_availability": "present",
        "local_expression_matrix": "data/raw/geo/GSE280506/GSE280506_filtered_feature_bc_matrix.h5",
        "reported_samples": 4,
        "soft_samples": [
            {
                "gsm": row["gsm"],
                "title": row["title"],
                "treatment": _char(row, "treatment:"),
                "cell_type": _char(row, "cell type:"),
                "model": _char(row, "cell line:"),
            }
            for row in samples
        ],
        "cells_in_identity_table": int(len(identities)),
        "cells_unassigned_star": int((identities["guide_identity"].astype(str) == "*").sum()),
        "cells_assigned_guide": int(len(assigned)),
        "unique_guide_identities_including_star": int(identities["guide_identity"].nunique(dropna=False)),
        "unique_assigned_guides": int(assigned["guide_identity"].nunique()),
        "unique_assigned_targets": int(assigned["target"].nunique()),
        "assigned_control_like_guides": int(assigned.loc[assigned["control_like"], "guide_identity"].nunique()),
        "assigned_control_like_cells": int(assigned["control_like"].sum()),
        "unique_perturbations": int(assigned["target"].nunique()),
        "matched_controls": (
            "DMSO vs Cisplatin is the chemical contrast; unassigned '*' and control-like guides "
            f"({int(assigned['control_like'].sum())} cells) are the genetic-control candidates. "
            "No LINCS plate-matched vehicle exists."
        ),
        "donor_or_model_grouping": {
            "model": "TP53/APC KO Human Gastric Organoids",
            "n_models": 1,
            "n_donors_named": 0,
            "split_unit_if_used": "single engineered organoid line; donor-cold generalization cannot be claimed",
        },
        "cells_by_sample_group": identities["sample_group"].value_counts(dropna=False).to_dict(),
        "exact978_coverage": _overlap_from_symbols(symbols, landmarks),
        "context_identity": "organoid/PDO namespace; not mergeable with LINCS cell_id",
        "live_geo_query": "NCBI_FTP_OK",
        "cannot_answer": [
            "chemical dose-response ranking ground truth",
            "donor-cold generalization (single engineered line)",
        ],
    }


def _audit_gse145308(landmarks: set[str]) -> dict[str, Any]:
    soft = ROOT / "data/raw/geo/GSE145308/GSE145308_family.soft.gz"
    samples = _soft_samples(soft)
    members: list[str] = []
    gene_symbols: set[str] = set()
    with tarfile.open(ROOT / "data/raw/geo/GSE145308/GSE145308_RAW.tar") as tar:
        members = tar.getnames()
        count_members = [
            name
            for name in members
            if name.endswith(".txt.gz") or name.endswith(".tsv.gz") or name.endswith(".txt")
        ]
        if count_members:
            extracted = tar.extractfile(count_members[0])
            if extracted is not None:
                handle: Any = extracted
                if count_members[0].endswith(".gz"):
                    handle = gzip.GzipFile(fileobj=extracted)
                table = pd.read_csv(handle, sep="\t")
                first = table.columns[0]
                gene_ids = {str(value).split(".")[0].upper() for value in table[first].dropna()}
                gene_symbols = gene_ids
    genotypes = sorted({_char(row, "genotype:") or "UNVERIFIED" for row in samples})
    return {
        "accession": "GSE145308",
        "official_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE145308",
        "role": "organoid_genetic_adaptation_reference",
        "organism": "Homo sapiens",
        "platforms": sorted({row["platform"] for row in samples if row["platform"]}),
        "modality": "bulk RNA-seq (two platforms)",
        "design": "human intestinal organoid; WT/APC/ARID1A/SMARCA4; 0h/24h TGFB; three replicates",
        "local_availability": "present",
        "local_expression_matrix": "data/raw/geo/GSE145308/GSE145308_RAW.tar",
        "reported_samples": len(samples),
        "soft_samples": [
            {
                "gsm": row["gsm"],
                "title": row["title"],
                "genotype": _char(row, "genotype:"),
                "source": row["source"],
                "platform": row["platform"],
            }
            for row in samples
        ],
        "unique_perturbations": len([item for item in genotypes if item != "UNVERIFIED"]),
        "unique_genotypes": genotypes,
        "raw_members": members,
        "matched_controls": "WT genotype + 0h (no TGFB) titles are the within-study controls; not LINCS vehicle",
        "donor_or_model_grouping": {
            "n_donors_named": 0,
            "note": "single human small-intestine organoid system in metadata; donor IDs are not in SOFT",
            "split_unit_if_used": "genotype x timepoint x replicate; donor-cold cannot be claimed",
        },
        "exact978_coverage": {
            "status": "UNVERIFIED_ENSEMBL_ONLY_NO_LOCAL_CROSSWALK",
            "raw_gene_namespace": "Ensembl ENSG in Identifier column",
            "n_raw_ids_first_file": int(len(gene_symbols)),
            "note": (
                "GSE92742 gene_info has Entrez + symbol only. No local Ensembl crosswalk "
                "is registered, so exact978 overlap is not counted."
            ),
        },
        "raw_gene_preview": sorted(gene_symbols)[:8],
        "context_identity": "organoid/PDO namespace; not mergeable with LINCS cell_id",
        "live_geo_query": "NCBI_FTP_OK",
        "cannot_answer": ["chemical sensitivity ground truth", "donor-cold generalization"],
    }


def _audit_gse167285(landmarks: set[str]) -> dict[str, Any]:
    soft = ROOT / "data/raw/geo/GSE167285/GSE167285_family.soft.gz"
    raw = pd.read_csv(ROOT / "data/raw/geo/GSE167285/GSE167285_Human_CRISPR_organoids_raw_counts.txt.gz", sep="\t")
    samples = _soft_samples(soft)
    sample_cols = [col for col in raw.columns if col != "Geneid"]
    donors = sorted({col.split("_")[0] for col in sample_cols})
    conditions = sorted({col.split("_", 1)[1] for col in sample_cols if "_" in col})
    genes = {str(value).upper() for value in raw["Geneid"].dropna()}
    return {
        "accession": "GSE167285",
        "official_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE167285",
        "role": "unseen_donor_genetic_response_test",
        "organism": "Homo sapiens",
        "modality": "bulk RNA-seq",
        "design": "colon organoid; 5 donors; SATB2 KO + Cas9 control",
        "local_availability": "present",
        "local_expression_matrix": "data/raw/geo/GSE167285/GSE167285_Human_CRISPR_organoids_raw_counts.txt.gz",
        "reported_samples": len(sample_cols),
        "soft_samples": [
            {
                "gsm": row["gsm"],
                "title": row["title"],
                "genotyping": _char(row, "genotyping:"),
                "intestine_part": _char(row, "intestine part:"),
            }
            for row in samples
        ],
        "matrix_sample_columns": sample_cols,
        "unique_perturbations": len(conditions),
        "conditions": conditions,
        "matched_controls": "Cas9 control column per donor; 5/5 donors have paired SATB2",
        "donor_or_model_grouping": {
            "donors": donors,
            "n_donors": len(donors),
            "split_unit_if_used": "donor_id; all Cas9/SATB2 profiles of a donor stay together",
        },
        "exact978_coverage": _overlap_from_symbols(genes, landmarks),
        "gene_id_namespace": "Geneid column; treated as symbol/ID as provided",
        "context_identity": "organoid/PDO namespace; not mergeable with LINCS cell_id",
        "live_geo_query": "NCBI_FTP_OK",
        "cannot_answer": ["chemical sensitivity ground truth"],
    }


def _audit_gse241659(landmarks: set[str]) -> dict[str, Any]:
    soft = ROOT / "data/raw/geo/GSE241659/GSE241659_family.soft.gz"
    counts = pd.read_csv(ROOT / "data/raw/geo/GSE241659/GSE241659_counts_normalised.tsv.gz", sep="\t")
    samples = _soft_samples(soft)
    sample_cols = [col for col in counts.columns if col not in {"GeneID", "Symbol", "Description"}]
    genes = {str(value).upper() for value in counts["Symbol"].dropna()} if "Symbol" in counts.columns else set()
    titles = [row["title"] or "" for row in samples]
    wt = [title for title in titles if re.search(r"WT|CTRL|CONTROL", title, re.I)]
    ko = [title for title in titles if re.search(r"KO|PTEN", title, re.I)]
    return {
        "accession": "GSE241659",
        "official_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE241659",
        "role": "pathway_mechanism_external_validation",
        "organism": "Homo sapiens",
        "modality": "bulk RNA-seq",
        "design": "human intestinal organoid PTEN KO RNA-seq",
        "local_availability": "present",
        "local_expression_matrix": "data/raw/geo/GSE241659/GSE241659_counts_normalised.tsv.gz",
        "reported_samples": len(samples),
        "soft_samples": [
            {
                "gsm": row["gsm"],
                "title": row["title"],
                "characteristics": row["characteristics"],
                "source": row["source"],
            }
            for row in samples
        ],
        "matrix_sample_columns": sample_cols,
        "unique_perturbations": 2 if wt and ko else len({title for title in titles}),
        "control_titles": wt,
        "ko_titles": ko,
        "matched_controls": "WT/control titles vs PTEN KO titles in the same study; not LINCS vehicle",
        "donor_or_model_grouping": {
            "n_soft_samples": len(samples),
            "donors": sorted({title.split("_")[0] for title in titles if title}),
            "n_donors": len({title.split("_")[0] for title in titles if title}),
            "split_unit_if_used": "donor prefix in sample title (T427/T474/T561/T640)",
            "titles": titles,
            "note": "T561 has KO only in this series; WT/KO pairing is incomplete for that donor",
        },
        "exact978_coverage": _overlap_from_symbols(genes, landmarks),
        "context_identity": "organoid/PDO namespace; not mergeable with LINCS cell_id",
        "live_geo_query": "NCBI_FTP_OK",
        "cannot_answer": ["chemical sensitivity ground truth"],
    }


def build() -> dict[str, Any]:
    landmarks, _ = _landmark_symbols()
    fetch = json.loads(FETCH.read_text(encoding="utf-8")) if FETCH.exists() else {}
    existing = json.loads(EXISTING.read_text(encoding="utf-8")) if EXISTING.exists() else {}
    datasets = [
        _audit_gse280506(landmarks),
        _audit_gse145308(landmarks),
        _audit_gse167285(landmarks),
        _audit_gse241659(landmarks),
    ]
    local_ok = all(row["local_availability"] == "present" for row in datasets)
    payload = {
        "format": "organoid_genetic_inventory_v2",
        "status": "LOCAL_MATRICES_PRESENT_ADAPTER_NOT_BUILT",
        "data_status": "DATA_PARTIAL",
        "existing_audit": str(EXISTING.relative_to(ROOT).as_posix()) if EXISTING.exists() else None,
        "existing_audit_status": existing.get("status"),
        "ncbi_geo_live_access": "FTP_OK_THIS_SESSION",
        "local_expression_matrices": local_ok,
        "fetch_audit": str(FETCH.relative_to(ROOT).as_posix()) if FETCH.exists() else None,
        "fetch_status": fetch.get("status"),
        "datasets": datasets,
        "exact978_contract": {
            "ordered_gene_ids_sha256": ORDERED_GENE_IDS_SHA256,
            "status": "COUNTED_BY_SYMBOL_NOT_FROZEN_AS_ADAPTER",
            "note": "symbol overlap only; no organoid 978 adapter, X_ctl, or Δ978 was built",
        },
        "forbidden_interpretation": (
            "These accessions are genetic/context adaptation references, not chemical sensitivity ground truth."
        ),
        "cannot_use_as_x_ctl": True,
        "cannot_compute_delta978": True,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    payload = build()
    print(
        json.dumps(
            {
                "status": payload["status"],
                "accessions": [row["accession"] for row in payload["datasets"]],
                "local": payload["local_expression_matrices"],
            },
            sort_keys=True,
        )
    )
