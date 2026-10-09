"""Audit existing TranSiGen weights against the frozen EXP-037 cell split."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


PRETRAIN_CELLS = {
    "HT29", "NPC", "A375", "PC3", "ASC", "MCF7", "HCC515", "VCAP", "HA1E", "A549"
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--target-genes", type=Path, required=True)
    parser.add_argument("--target-drugs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data_path = args.source_root / "data/lincs/processed_data.h5"
    genes_path = args.source_root / "data/lincs/geneinfo_processed.csv"
    feature_path = args.source_root / "data/lincs/KPGT_emb2304.pickle"
    checkpoint = args.source_root / (
        "参考代码/TranSiGen/results/trained_models_164_cell_cells_split/364039/"
        "feature_KPGT_init_pretrain_shRNA_all/best_model.pt"
    )
    for path in (data_path, genes_path, feature_path, checkpoint, args.split,
                 args.target_genes, args.target_drugs):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty asset: {path}")

    with h5py.File(data_path) as handle:
        raw_cells = [x.decode("utf-8") for x in handle["cid"][:]]
        n_rows = len(raw_cells)
        if handle["x1"].shape != (n_rows, 978) or handle["x2"].shape != (n_rows, 978):
            raise ValueError("Source expression axis mismatch")
    unique_cells = set(raw_cells)
    counts = pd.DataFrame({"cid": raw_cells}).value_counts("cid").reset_index()
    counts.columns = ["cid", "count"]
    source_test = set(
        counts[~counts.cid.isin(PRETRAIN_CELLS)]
        .sample(n=7, replace=False, random_state=1234).cid
    )
    source_valid = set(
        counts[~counts.cid.isin(PRETRAIN_CELLS | source_test)]
        .sample(n=7, replace=False, random_state=1234).cid
    )
    source_train = unique_cells - source_valid - source_test
    if len(unique_cells) != 164 or len(source_train) != 150:
        raise ValueError("Unexpected source cell split")

    source_gene_table = pd.read_csv(genes_path)
    source_genes = source_gene_table.loc[
        (source_gene_table.pr_is_bing == 1) & (source_gene_table.pr_is_lm == 1),
        "pr_gene_id",
    ].astype(str).tolist()
    target_genes = [str(x) for x in json.loads(args.target_genes.read_text())]
    if len(source_genes) != 978 or len(target_genes) != 978 or set(source_genes) != set(target_genes):
        raise ValueError("Source and target landmark gene sets differ")
    gene_index = {gene: i for i, gene in enumerate(source_genes)}
    target_to_source = [gene_index[gene] for gene in target_genes]

    RDLogger.DisableLog("rdApp.*")
    with feature_path.open("rb") as handle:
        features = pickle.load(handle)  # Trusted local project asset.
    if not isinstance(features, dict):
        raise ValueError("Source KPGT feature cache is not a dictionary")
    inchi_to_feature: dict[str, np.ndarray] = {}
    ambiguous_inchi: set[str] = set()
    for smiles, value in features.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Unparseable source SMILES: {smiles}")
        inchi = Chem.MolToInchiKey(mol)
        vector = np.asarray(value)
        if vector.shape != (2304,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid KPGT feature for {smiles}")
        if inchi in inchi_to_feature and not np.array_equal(inchi_to_feature[inchi], vector):
            ambiguous_inchi.add(inchi)
        else:
            inchi_to_feature.setdefault(inchi, vector)
    target_drugs = pd.read_parquet(args.target_drugs)
    if target_drugs.pert_id.duplicated().any() or target_drugs.inchi_key.isna().any():
        raise ValueError("Target drug identity is not unique or complete")
    matching = target_drugs.inchi_key.isin(inchi_to_feature)
    ambiguous = target_drugs.inchi_key.isin(ambiguous_inchi)
    eligible_drug_ids = sorted(target_drugs.loc[matching & ~ambiguous, "pert_id"].tolist())

    frozen = json.loads(args.split.read_text())
    folds = []
    for fold in frozen["folds"]:
        test = set(fold["outer_test_ids"])
        folds.append({
            "fold": fold["fold"],
            "n_test_cells": len(test),
            "exposed_by_source_train": sorted(test & source_train),
            "n_exposed_by_source_train": len(test & source_train),
            "in_source_valid": sorted(test & source_valid),
            "in_source_test": sorted(test & source_test),
        })

    result = {
        "experiment_id": "EXP-037",
        "status": "TRANSIGEN_ASSET_EXPOSURE_AUDITED",
        "scope": "cell and gene identity only; checkpoint training lineage remains to be independently confirmed",
        "source_data_sha256": sha256(data_path),
        "source_gene_table_sha256": sha256(genes_path),
        "source_feature_cache_sha256": sha256(feature_path),
        "checkpoint_sha256": sha256(checkpoint),
        "target_split_sha256": sha256(args.split),
        "target_genes_sha256": sha256(args.target_genes),
        "target_drugs_sha256": sha256(args.target_drugs),
        "n_source_rows": n_rows,
        "n_source_cells": len(unique_cells),
        "n_source_train_cells_reconstructed": len(source_train),
        "source_valid_cells": sorted(source_valid),
        "source_test_cells": sorted(source_test),
        "source_gene_order_equals_target": source_genes == target_genes,
        "target_to_source_gene_indices": target_to_source,
        "n_source_feature_smiles": len(features),
        "n_source_feature_unique_inchi_keys": len(inchi_to_feature),
        "n_ambiguous_inchi_keys_with_different_vectors": len(ambiguous_inchi),
        "n_target_drugs_with_feature_inchi_match": int(matching.sum()),
        "n_target_drugs_with_ambiguous_feature": int((matching & ambiguous).sum()),
        "n_target_drugs_with_unambiguous_feature": len(eligible_drug_ids),
        "target_drug_ids_with_unambiguous_feature": eligible_drug_ids,
        "strict_unseen_cell_eligible_if_checkpoint_uses_reconstructed_train": all(
            row["n_exposed_by_source_train"] == 0 for row in folds
        ),
        "folds": folds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "source_gene_order_equals_target": result["source_gene_order_equals_target"],
        "n_exposed_by_source_train": [x["n_exposed_by_source_train"] for x in folds],
        "n_target_drugs_with_unambiguous_feature": len(eligible_drug_ids),
    }))


if __name__ == "__main__":
    main()
