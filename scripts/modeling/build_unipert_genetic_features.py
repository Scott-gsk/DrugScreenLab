"""Build a bounded, response-blind UniPert genetic feature table."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pert-info", type=Path, required=True)
    parser.add_argument("--unipert-source", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-features", type=Path, required=True)
    parser.add_argument("--output-mapping", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--max-genes", type=int, default=256)
    args = parser.parse_args()
    if args.max_genes < 1:
        raise ValueError("--max-genes must be positive")

    perturbagens = pd.read_csv(args.pert_info, sep="\t", low_memory=False)
    required = {"pert_id", "pert_iname", "pert_type"}
    missing = sorted(required.difference(perturbagens.columns))
    if missing:
        raise ValueError(f"pert_info missing columns: {missing}")
    genetic = perturbagens.loc[
        perturbagens["pert_type"].isin(["trt_sh", "trt_sh.cgs", "trt_sh.css", "trt_oe", "trt_oe.mut"])
    ].copy()
    genetic["gene_symbol"] = genetic["pert_iname"].astype(str).str.strip().str.upper()
    genetic = genetic.loc[~genetic["gene_symbol"].isin({"", "NAN", "-666"})]
    source_root = args.unipert_source.resolve()
    reference_targets = pd.read_csv(source_root / "data" / "ref_targets.csv", low_memory=False)
    local_genes = set(reference_targets["Approved symbol"].astype(str).str.upper())
    genes = sorted(set(genetic["gene_symbol"]).intersection(local_genes))[: args.max_genes]
    if not genes:
        raise ValueError("no genetic perturbagen gene symbols are available")

    sys.path.insert(0, str(source_root))
    from unipert.model import UniPert  # type: ignore[import-not-found]

    model = UniPert(
        data_dir=str(source_root / "data"),
        model_dir=str(args.model_dir.resolve()),
    )
    representations, invalid = model.encode_genes(gene_names=genes)
    valid_genes = sorted(representations)
    if not valid_genes:
        raise RuntimeError("UniPert encoded no genetic perturbagens")
    features = np.stack([np.asarray(representations[gene], dtype=np.float32) for gene in valid_genes])
    if features.ndim != 2 or features.shape[1] != 256 or not np.isfinite(features).all():
        raise RuntimeError(f"unexpected UniPert genetic feature shape: {features.shape}")
    gene_to_row = {gene: index for index, gene in enumerate(valid_genes)}
    selected = genetic.loc[genetic["gene_symbol"].isin(gene_to_row), ["pert_id", "gene_symbol", "pert_type"]].copy()
    pert_to_row = {str(row.pert_id): int(gene_to_row[str(row.gene_symbol)]) for row in selected.itertuples(index=False)}

    args.output_features.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_features, features)
    mapping = {"gene_to_row": gene_to_row, "pert_id_to_row": pert_to_row}
    args.output_mapping.parent.mkdir(parents=True, exist_ok=True)
    args.output_mapping.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "format": "unipert_genetic_feature_table_v1",
        "status": "FAST_FEATURES_READY",
        "representation": "official_UniPert_genetic_encoder_256d",
        "pert_info": str(args.pert_info),
        "pert_info_sha256": _sha256(args.pert_info),
        "unipert_source": str(source_root),
        "unipert_model": str(args.model_dir.resolve() / "unipert_model.pt"),
        "unipert_model_sha256": _sha256(args.model_dir.resolve() / "unipert_model.pt"),
        "candidate_gene_limit": args.max_genes,
        "local_reference_only": True,
        "candidate_genes": len(genes),
        "encoded_genes": len(valid_genes),
        "invalid_genes": sorted(set(invalid)),
        "feature_shape": list(features.shape),
        "mapped_genetic_perturbagen_ids": len(pert_to_row),
        "labels_used": False,
        "response_values_read": False,
        "downstream_status": "GENETIC_RESPONSE_ADAPTER_PENDING",
        "features_path": str(args.output_features),
        "features_sha256": _sha256(args.output_features),
        "mapping_path": str(args.output_mapping),
        "mapping_sha256": _sha256(args.output_mapping),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
