"""Build the compact CRC tumor-vs-normal signature for MVP-001.

The only input expression summary is the small, processed GEO GSE19163
supplementary table.  Its signed ``Fold-Change(A TUMOR vs. NORMAL)`` values
are linear fold-change magnitudes with a direction sign (the source paper
uses negative and positive cutoffs).  The output is therefore converted to
``sign(fc) * log2(abs(fc))`` before intersecting the frozen LINCS exact-978
gene universe.  No raw CEL or expression matrix is read or written.
"""

from __future__ import annotations

import argparse
import gzip
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE19nnn/GSE19163/suppl/"
    "GSE19163_tumor_vs_normal_tissue.txt.gz"
)
SOURCE_SHA256 = "97a706393d5d6a7e7b7050542bcb41a456638425b208c4dab1f42efce97759fb"
SOURCE_STUDY = "GSE19163"
SOURCE_PLATFORM = "GPL5175"
EXACT978_REGISTRY_ID = "lincs_gse92742_exact978_cache_v1"


def _sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_source(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        table = pd.read_csv(handle, sep="\t")
    required = {
        "Transcript ID",
        "Gene Symbol",
        "RefSeq",
        "p-value(A TUMOR vs. NORMAL)",
        "Fold-Change(A TUMOR vs. NORMAL)",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"GSE19163 source missing columns: {sorted(missing)}")
    table = table.rename(
        columns={
            "Transcript ID": "source_transcript_id",
            "Gene Symbol": "gene_symbol",
            "RefSeq": "source_refseq",
            "p-value(A TUMOR vs. NORMAL)": "source_pvalue",
            "Fold-Change(A TUMOR vs. NORMAL)": "source_fold_change_signed",
        }
    )
    table["gene_symbol"] = table["gene_symbol"].fillna("").astype(str).str.strip().str.upper()
    table["source_fold_change_signed"] = pd.to_numeric(
        table["source_fold_change_signed"], errors="coerce"
    )
    if table["source_fold_change_signed"].isna().any():
        raise ValueError("GSE19163 source contains non-numeric fold-change values")
    if (table["source_fold_change_signed"] == 0).any():
        raise ValueError("zero fold-change cannot receive a signed log2FC")
    return table


def build(root: Path, source: Path, output: Path, metadata_output: Path) -> dict[str, object]:
    source_sha256 = _sha256(source)
    if source_sha256 != SOURCE_SHA256:
        raise ValueError(f"GSE19163 source checksum mismatch: {source_sha256}")
    source_table = _load_source(source)
    gene_info = pd.read_csv(
        root / "raw/lincs/GSE92742/GSE92742_Broad_LINCS_gene_info.txt.gz",
        sep="\t",
        dtype={"pr_gene_id": str, "pr_gene_symbol": str, "pr_is_lm": str},
    )
    exact = gene_info[gene_info["pr_is_lm"].eq("1")].copy()
    exact["gene_symbol"] = exact["pr_gene_symbol"].fillna("").str.strip().str.upper()
    if len(exact) != 978 or not exact["gene_symbol"].is_unique:
        raise ValueError("GSE92742 exact-978 gene universe is not unique or has wrong size")

    symbol_counts = source_table.loc[source_table["gene_symbol"].ne(""), "gene_symbol"].value_counts()
    duplicate_symbols = sorted(symbol_counts[symbol_counts > 1].index.tolist())
    missing_symbol_rows = int(source_table["gene_symbol"].eq("").sum())
    unique_source = source_table[source_table["gene_symbol"].ne("")].copy()
    unique_source = unique_source[~unique_source["gene_symbol"].isin(duplicate_symbols)]
    merged = exact[["pr_gene_id", "gene_symbol"]].merge(
        unique_source,
        on="gene_symbol",
        how="inner",
        validate="one_to_one",
    )
    merged["signed_log2fc"] = np.sign(merged["source_fold_change_signed"]) * np.log2(
        np.abs(merged["source_fold_change_signed"])
    )
    merged["direction"] = np.where(merged["signed_log2fc"] > 0, "up", "down")
    exact_index = {symbol: index for index, symbol in enumerate(exact["gene_symbol"].tolist())}
    merged["gene_index_978"] = merged["gene_symbol"].map(exact_index).astype(int)
    merged["comparison"] = "tumor_vs_normal"
    merged["role"] = "CRC_DISEASE_SIGNATURE"
    columns = [
        "gene_index_978",
        "pr_gene_id",
        "gene_symbol",
        "source_transcript_id",
        "source_refseq",
        "source_pvalue",
        "source_fold_change_signed",
        "signed_log2fc",
        "direction",
        "comparison",
        "role",
    ]
    merged = merged.sort_values("gene_index_978")[columns]
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, sep="\t", index=False, float_format="%.10g")
    gene_digest = sha256("\n".join(exact["gene_symbol"].tolist()).encode()).hexdigest()
    metadata = {
        "format": "mvp001_crc_signature_exact978_v1",
        "experiment_id": "MVP-001",
        "role": "DATA_STEWARD",
        "status": "DATA_PARTIAL",
        "source": {
            "study": SOURCE_STUDY,
            "accession": SOURCE_STUDY,
            "platform": SOURCE_PLATFORM,
            "url": SOURCE_URL,
            "local_path": str(source),
            "sha256": source_sha256,
            "bytes": source.stat().st_size,
            "source_field": "Fold-Change(A TUMOR vs. NORMAL)",
            "comparison_direction": "positive=tumor_higher; negative=tumor_lower",
        },
        "transformation": {
            "formula": "signed_log2fc = sign(source_fold_change_signed) * log2(abs(source_fold_change_signed))",
            "reason": "GSE19163 reports signed linear fold-change magnitudes; negative and positive cutoffs are used in the source paper",
            "normalization": "source supplementary summary; no additional normalization or batch correction",
        },
        "gene_universe": {
            "registry_id": EXACT978_REGISTRY_ID,
            "source": "GSE92742 gene_info pr_is_lm=1",
            "size": int(len(exact)),
            "ordered_gene_symbols_sha256": gene_digest,
        },
        "source_audit": {
            "rows": int(len(source_table)),
            "unique_gene_symbols": int(source_table["gene_symbol"].replace("", np.nan).dropna().nunique()),
            "missing_gene_symbol_rows": missing_symbol_rows,
            "duplicate_gene_symbols": duplicate_symbols,
            "duplicate_rows_excluded": int(len(source_table) - len(unique_source) - missing_symbol_rows),
        },
        "exact978_overlap": {
            "rows": int(len(merged)),
            "gene_symbols": merged["gene_symbol"].tolist(),
            "up": int((merged["direction"] == "up").sum()),
            "down": int((merged["direction"] == "down").sum()),
            "minimum_formal_total": 20,
            "minimum_formal_each_direction": 5,
            "exploratory_only": True,
        },
        "forbidden": [
            "GSE19163 raw CEL files",
            "GSE19163 p-values for model/tuning selection",
            "GSE117548 activity or factor labels",
            "PRISM response labels",
            "inferred/non-landmark genes to pad overlap",
        ],
        "output": {
            "local_path": str(output),
            "sha256": None,
            "schema": columns,
        },
    }
    metadata["output"]["sha256"] = _sha256(output)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("mvp/core_data/_source_cache/GSE19163_tumor_vs_normal_tissue.txt.gz"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("mvp/core_data/crc_disease_signature_exact978.tsv")
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("mvp/core_data/crc_disease_signature_audit.json"),
    )
    args = parser.parse_args()
    metadata = build(args.root, args.source, args.output, args.metadata_output)
    print(json.dumps(metadata["exact978_overlap"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
