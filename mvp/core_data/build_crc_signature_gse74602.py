"""Build the primary MVP-001 CRC tumor-vs-normal exact-978 signature.

GSE74602 is a processed GEO supplementary expression table for 30 paired
normal/tumor colorectal samples (GPL6104).  The table is explicitly the
non-normalized signal matrix; this builder computes a pre-registered paired
log2(T/N) contrast, then takes the median across pairs and (where present)
platform probes for each gene.  It never reads or writes raw CEL files and it
does not use PRISM labels to select genes.

The sample-role and pairing map is read from the official GEO family SOFT
metadata.  The SOFT file may be supplied as a transient local download; only
its checksum and compact pairing map are emitted into the tracked audit.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import h5py


MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE74nnn/GSE74602/suppl/"
    "GSE74602_non_normalized.txt.gz"
)
MATRIX_SHA256 = "381b6cb83e6776d7f571833001765ccfbf3143fc500f69a0e1df5b4aa264b0f6"
SOFT_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE74nnn/GSE74602/soft/GSE74602_family.soft.gz"
PLATFORM_URL = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL6nnn/GPL6104/annot/GPL6104.annot.gz"
PLATFORM_SHA256 = "82ae57d6d9ec26ce2bcff01ccd1db498bb8055ee01471829dec0a5ab5666d518"
SOURCE_STUDY = "GSE74602"
SOURCE_PLATFORM = "GPL6104"
EXACT978_REGISTRY_ID = "lincs_gse92742_exact978_cache_v1"
GCTX_RELATIVE = Path(
    "interim/lincs/GSE92742/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx"
)
ORDERED_GENE_IDS_SHA256 = "b4e2fca877c5cfdcc1c712ad0fd67e97a88b6f7566b013e4bab065f699ebb623"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_soft_samples(path: Path) -> dict[str, dict[str, str]]:
    """Read only the sample title/source/characteristics from GEO SOFT."""

    samples: dict[str, dict[str, str]] = {}
    current: str | None = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("!Sample_title = "):
                current = line.split("=", 1)[1].strip()
                samples[current] = {}
            elif current and line.startswith("!Sample_source_name_ch1 = "):
                samples[current]["source_name"] = line.split("=", 1)[1].strip()
            elif current and line.startswith("!Sample_characteristics_ch1 = "):
                value = line.split("=", 1)[1].strip()
                if value.lower().startswith("tissue type:"):
                    samples[current]["tissue_type"] = value.split(":", 1)[1].strip()
    return samples


def make_pairs(sample_meta: dict[str, dict[str, str]], matrix_samples: list[str]) -> list[dict[str, str]]:
    missing = sorted(set(matrix_samples) - set(sample_meta))
    if missing:
        raise ValueError(f"GSE74602 SOFT is missing matrix samples: {missing[:5]}")
    by_subject: defaultdict[str, list[str]] = defaultdict(list)
    for sample in matrix_samples:
        by_subject[sample.rsplit("_", 1)[0]].append(sample)
    pairs: list[dict[str, str]] = []
    for subject, samples in sorted(by_subject.items()):
        tumors = [
            s for s in samples if sample_meta[s].get("tissue_type", "").lower().startswith("tumor")
        ]
        normals = [
            s
            for s in samples
            if sample_meta[s].get("tissue_type", "").lower().startswith("normal")
        ]
        tumors.sort(key=lambda s: s.rsplit("_", 1)[1])
        normals.sort(key=lambda s: s.rsplit("_", 1)[1])
        if not tumors or len(tumors) != len(normals):
            raise ValueError(f"unbalanced tumor/normal samples for subject {subject}")
        for index, (tumor, normal) in enumerate(zip(tumors, normals), start=1):
            pairs.append(
                {
                    "pair_id": f"{subject}_pair{index}",
                    "subject_id": subject,
                    "tumor_sample": tumor,
                    "normal_sample": normal,
                    "tumor_source_name": sample_meta[tumor].get("source_name", ""),
                    "normal_source_name": sample_meta[normal].get("source_name", ""),
                }
            )
    if len(pairs) != 30:
        raise ValueError(f"expected 30 paired samples, found {len(pairs)}")
    return pairs


def load_platform(path: Path) -> pd.DataFrame:
    rows: list[list[str]] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!platform_table_begin"):
                break
        header = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            if line.startswith("!platform_table_end"):
                break
            rows.append(line.rstrip("\n").split("\t"))
    table = pd.DataFrame(rows, columns=header)
    required = {"ID", "Gene symbol"}
    if not required.issubset(table.columns):
        raise ValueError(f"GPL6104 annotation missing columns: {sorted(required - set(table.columns))}")
    table = table[["ID", "Gene symbol"]].rename(columns={"ID": "probe_id"})
    table["gene_symbol"] = table["Gene symbol"].fillna("").astype(str).str.strip().str.upper()
    return table[["probe_id", "gene_symbol"]]


def build(
    root: Path,
    matrix: Path,
    platform: Path,
    soft: Path,
    output: Path,
    metadata_output: Path,
    pair_output: Path,
) -> dict[str, object]:
    matrix_sha = sha256_file(matrix)
    if matrix_sha != MATRIX_SHA256:
        raise ValueError(f"GSE74602 matrix checksum mismatch: {matrix_sha}")
    platform_sha = sha256_file(platform)
    if platform_sha != PLATFORM_SHA256:
        raise ValueError(f"GPL6104 annotation checksum mismatch: {platform_sha}")
    sample_meta = parse_soft_samples(soft)
    matrix_table = pd.read_csv(matrix, sep="\t", compression="gzip")
    if "ID_REF" not in matrix_table.columns:
        raise ValueError("GSE74602 matrix lacks ID_REF column")
    matrix_samples = [str(c) for c in matrix_table.columns if c != "ID_REF"]
    pairs = make_pairs(sample_meta, matrix_samples)
    values = matrix_table.set_index("ID_REF").apply(pd.to_numeric, errors="coerce")
    if values.index.duplicated().any():
        raise ValueError("GSE74602 matrix contains duplicate probe IDs")
    pair_contrasts: list[np.ndarray] = []
    for pair in pairs:
        tumor = values[pair["tumor_sample"]].to_numpy(dtype=float)
        normal = values[pair["normal_sample"]].to_numpy(dtype=float)
        valid = (tumor > 0) & (normal > 0) & np.isfinite(tumor) & np.isfinite(normal)
        pair_contrasts.append(np.where(valid, np.log2(tumor / normal), np.nan))
    contrasts = np.stack(pair_contrasts, axis=1)
    probe_log2fc = np.nanmedian(contrasts, axis=1)
    probe_n_pairs = np.isfinite(contrasts).sum(axis=1).astype(int)
    annotation = load_platform(platform)
    annotated = pd.DataFrame(
        {
            "probe_id": values.index.astype(str),
            "probe_signed_log2fc": probe_log2fc,
            "n_pairs": probe_n_pairs,
        }
    ).merge(annotation, on="probe_id", how="left", validate="one_to_one")
    annotated = annotated[
        annotated["gene_symbol"].ne("")
        & np.isfinite(annotated["probe_signed_log2fc"])
        & annotated["n_pairs"].ge(20)
    ]
    # Probe-to-gene aggregation is fixed to the median, independent of PRISM.
    gene = (
        annotated.groupby("gene_symbol", sort=True)
        .agg(
            signed_log2fc=("probe_signed_log2fc", "median"),
            n_pairs=("n_pairs", "max"),
            n_probes=("probe_id", "count"),
        )
        .reset_index()
    )
    gene["direction"] = np.where(gene["signed_log2fc"] > 0, "up", "down")
    gene_info = pd.read_csv(
        root / "raw/lincs/GSE92742/GSE92742_Broad_LINCS_gene_info.txt.gz",
        sep="\t",
        dtype={"pr_gene_id": str, "pr_gene_symbol": str, "pr_is_lm": str},
    )
    flags = dict(zip(gene_info["pr_gene_id"].astype(str), gene_info["pr_is_lm"].astype(int)))
    symbols = dict(
        zip(
            gene_info["pr_gene_id"].astype(str),
            gene_info["pr_gene_symbol"].fillna("").astype(str).str.strip().str.upper(),
        )
    )
    with h5py.File(root / GCTX_RELATIVE, "r") as handle:
        row_ids = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in handle["0/META/ROW/id"][:]
        ]
    landmark_ids = [gene_id for gene_id in row_ids if flags.get(gene_id, 0) == 1]
    if len(landmark_ids) != 978:
        raise ValueError(f"GSE92742 exact-978 GCTX landmark order has {len(landmark_ids)} rows")
    actual_gene_id_digest = hashlib.sha256("".join(gene_id + "\n" for gene_id in landmark_ids).encode()).hexdigest()
    if actual_gene_id_digest != ORDERED_GENE_IDS_SHA256:
        raise ValueError(f"GSE92742 exact-978 GCTX gene order checksum mismatch: {actual_gene_id_digest}")
    exact_order = pd.DataFrame(
        {"pr_gene_id": landmark_ids, "gene_symbol": [symbols[gene_id] for gene_id in landmark_ids]}
    )
    if exact_order["gene_symbol"].eq("").any() or not exact_order["gene_symbol"].is_unique:
        raise ValueError("GSE92742 exact-978 GCTX gene universe is not unique or has missing symbols")
    exact_order["gene_index_978"] = np.arange(len(exact_order), dtype=int)
    merged = exact_order.merge(gene, on="gene_symbol", how="inner", validate="one_to_one")
    merged["comparison"] = "tumor_vs_normal"
    merged["role"] = "CRC_DISEASE_SIGNATURE"
    columns = [
        "gene_index_978",
        "pr_gene_id",
        "gene_symbol",
        "signed_log2fc",
        "direction",
        "n_pairs",
        "n_probes",
        "comparison",
        "role",
    ]
    merged = merged.sort_values("gene_index_978")[columns]
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, sep="\t", index=False, float_format="%.10g")
    pair_output.parent.mkdir(parents=True, exist_ok=True)
    pair_output.write_text(json.dumps(pairs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    gene_digest = hashlib.sha256("".join(symbol + "\n" for symbol in exact_order["gene_symbol"]).encode()).hexdigest()
    gene_id_digest = actual_gene_id_digest
    sample_roles = {
        "tumor": int(sum(v.get("tissue_type", "").lower().startswith("tumor") for v in sample_meta.values())),
        "normal": int(sum(v.get("tissue_type", "").lower().startswith("normal") for v in sample_meta.values())),
    }
    metadata = {
        "format": "mvp001_crc_signature_exact978_v2",
        "experiment_id": "MVP-001",
        "role": "DATA_STEWARD",
        "status": "DATA_READY",
        "source": {
            "study": SOURCE_STUDY,
            "accession": SOURCE_STUDY,
            "title": "30 paired normal and tumor colorectal samples",
            "platform": SOURCE_PLATFORM,
            "matrix_url": MATRIX_URL,
            "matrix_local_path": str(matrix),
            "matrix_sha256": matrix_sha,
            "matrix_bytes": matrix.stat().st_size,
            "matrix_semantics": "processed non-normalized signal values",
            "soft_url": SOFT_URL,
            "soft_local_path": str(soft),
            "soft_sha256": sha256_file(soft),
            "platform_annotation_url": PLATFORM_URL,
            "platform_annotation_local_path": str(platform),
            "platform_annotation_sha256": platform_sha,
        },
        "sample_context": {
            "sample_count": len(matrix_samples),
            "tumor_sample_count": sample_roles["tumor"],
            "normal_sample_count": sample_roles["normal"],
            "paired_subject_count": len(pairs),
            "pairing_rule": "same subject prefix before final underscore; sorted tumor and normal source samples paired by ordinal",
            "pair_map_local_path": str(pair_output),
            "pair_map_sha256": sha256_file(pair_output),
            "tissue_roles_from_official_soft": True,
        },
        "transformation": {
            "formula": "pair_log2fc = log2(tumor_signal / normal_signal); gene_signed_log2fc = median(all finite pair_log2fc, then median across probes)",
            "positive_direction": "tumor_higher",
            "negative_direction": "tumor_lower",
            "min_pairs_per_probe": 20,
            "probe_to_gene_aggregation": "median",
            "additional_normalization": "none; source is non-normalized and paired ratios are used",
        },
        "gene_universe": {
            "registry_id": EXACT978_REGISTRY_ID,
            "source": "GSE92742 gene_info pr_is_lm=1",
            "size": int(len(exact_order)),
            "ordered_gene_symbols_sha256": gene_digest,
            "ordered_gene_ids_sha256": gene_id_digest,
        },
        "exact978_overlap": {
            "rows": int(len(merged)),
            "up": int((merged["direction"] == "up").sum()),
            "down": int((merged["direction"] == "down").sum()),
            "zero": int((merged["signed_log2fc"] == 0).sum()),
            "minimum_formal_total": 20,
            "minimum_formal_each_direction": 5,
            "formal_gate_pass": bool(
                len(merged) >= 20
                and int((merged["direction"] == "up").sum()) >= 5
                and int((merged["direction"] == "down").sum()) >= 5
            ),
            "gene_symbols": merged["gene_symbol"].tolist(),
        },
        "forbidden": [
            "GSE74602 raw CEL files",
            "GSE74602 sample-level values for PRISM label or candidate selection",
            "GSE117548 activity/factor labels",
            "PRISM response labels for selecting signature genes",
            "inferred/non-landmark genes outside exact-978",
        ],
        "output": {
            "local_path": str(output),
            "sha256": sha256_file(output),
            "schema": columns,
        },
        "provenance": {
            "generator": "mvp/core_data/build_crc_signature_gse74602.py",
            "generator_sha256": sha256_file(Path(__file__)),
            "creation_command": "curl -L --fail -o mvp/core_data/_source_cache/GSE74602_non_normalized.txt.gz https://ftp.ncbi.nlm.nih.gov/geo/series/GSE74nnn/GSE74602/suppl/GSE74602_non_normalized.txt.gz && curl -L --fail -o mvp/core_data/_source_cache/GPL6104.annot.gz https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL6nnn/GPL6104/annot/GPL6104.annot.gz && curl -L --fail -o /tmp/GSE74602_family.soft.gz https://ftp.ncbi.nlm.nih.gov/geo/series/GSE74nnn/GSE74602/soft/GSE74602_family.soft.gz && python mvp/core_data/build_crc_signature_gse74602.py",
            "environment": "WSL2; conda drugscreening-gpu",
            "large_raw_cel_not_downloaded": True,
        },
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument(
        "--matrix", type=Path, default=Path("mvp/core_data/_source_cache/GSE74602_non_normalized.txt.gz")
    )
    parser.add_argument("--platform", type=Path, default=Path("mvp/core_data/_source_cache/GPL6104.annot.gz"))
    parser.add_argument("--soft", type=Path, default=Path("/tmp/GSE74602_family.soft.gz"))
    parser.add_argument("--output", type=Path, default=Path("mvp/core_data/crc_disease_signature_exact978.tsv"))
    parser.add_argument("--metadata-output", type=Path, default=Path("mvp/core_data/crc_disease_signature_audit.json"))
    parser.add_argument("--pair-output", type=Path, default=Path("mvp/core_data/gse74602_sample_pairs.json"))
    args = parser.parse_args()
    metadata = build(
        args.root,
        args.matrix,
        args.platform,
        args.soft,
        args.output,
        args.metadata_output,
        args.pair_output,
    )
    print(json.dumps(metadata["exact978_overlap"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
