"""Inventory candidate organoid genetic accessions from local files + metadata."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/experiments/EXP-007/ORGANOID_GENETIC_INVENTORY.json"
EXISTING = ROOT / "mvp/extension/ORGANOID_DATASET_READINESS_AUDIT.json"
APPENDIX = ROOT / "附录 A：核心参考文献.md"


CANDIDATES = [
    {
        "accession": "GSE280506",
        "official_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE280506",
        "role": "organoid_genetic_adaptation_reference",
        "organism": "Homo sapiens",
        "platform": "GPL24676",
        "modality": "scRNA-seq / CROP-seq",
        "design": "primary human gastric organoid; CRISPRi/CRISPRa; DMSO/cisplatin",
        "reported_samples": 4,
        "reported_unique_perturbations": "CRISPRi/a against DNA-binding proteins; UNVERIFIED exact count locally",
        "source_of_design": [
            "mvp/extension/ORGANOID_DATASET_READINESS_AUDIT.json",
            "附录 A：核心参考文献.md",
        ],
    },
    {
        "accession": "GSE145308",
        "official_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE145308",
        "role": "organoid_genetic_adaptation_reference",
        "organism": "Homo sapiens",
        "platforms": ["GPL20301", "GPL24676"],
        "modality": "bulk RNA-seq (two platforms)",
        "design": "human intestinal organoid; WT/APC/ARID1A/SMARCA4; 0h/24h; three replicates",
        "reported_samples": 24,
        "reported_unique_perturbations": "WT, APC, ARID1A, SMARCA4 backgrounds; UNVERIFIED exact count locally",
        "source_of_design": [
            "mvp/extension/ORGANOID_DATASET_READINESS_AUDIT.json",
            "附录 A：核心参考文献.md",
        ],
    },
    {
        "accession": "GSE167285",
        "official_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE167285",
        "role": "unseen_donor_genetic_response_test",
        "organism": "Homo sapiens",
        "modality": "bulk RNA-seq",
        "design": "colon organoid; 5 donors; SATB2 KO + control; 10 bulk RNA-seq",
        "reported_samples": 10,
        "reported_unique_perturbations": "SATB2 KO vs control across 5 donors",
        "source_of_design": ["附录 A：核心参考文献.md"],
    },
    {
        "accession": "GSE241659",
        "official_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE241659",
        "role": "pathway_mechanism_external_validation",
        "organism": "Homo sapiens",
        "modality": "bulk RNA-seq",
        "design": "human intestinal organoid PTEN KO RNA-seq",
        "reported_samples": "UNVERIFIED",
        "reported_unique_perturbations": "PTEN KO vs control",
        "source_of_design": ["附录 A：核心参考文献.md"],
    },
]


def _scan_local(accession: str) -> dict[str, object]:
    hits: list[str] = []
    data_root = ROOT / "data"
    if data_root.exists():
        for path in data_root.rglob("*"):
            name = path.name.upper()
            if accession.upper() in name or accession.upper() in str(path).upper():
                hits.append(path.relative_to(ROOT).as_posix())
                if len(hits) >= 20:
                    break
    return {
        "local_files": hits,
        "local_file_count": int(len(hits)),
        "local_availability": "present" if hits else "absent",
    }


def build() -> dict[str, object]:
    if OUT.exists():
        current = json.loads(OUT.read_text(encoding="utf-8"))
        if current.get("format") == "organoid_genetic_inventory_v2":
            return current
    raise RuntimeError(
        "Legacy metadata-only inventory is retired. "
        "Run scripts/data/audit_organoid_local_matrices.py after a live GEO fetch."
    )


if __name__ == "__main__":
    print(json.dumps({"output": str(OUT), "n": len(build()["datasets"])}))
