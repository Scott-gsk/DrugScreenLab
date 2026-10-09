"""Audit XPert checkpoint exposure and drug/gene identity against EXP-037 before reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(source_h5ad: Path, warm_checkpoint: Path, cold_checkpoint: Path,
          source_drugs: Path, target_drugs: Path, target_contract: Path,
          target_genes: Path, split_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"XPert audit output already exists: {output_dir}")
    source = ad.read_h5ad(source_h5ad, backed="r")
    try:
        obs = source.obs[["sample_id", "cell_iname", "pert_id", "split_1",
                          "split_cold_cell_1"]].copy()
        source_gene_ids = source.var["gene_id"].astype(str).tolist()
    finally:
        source.file.close()
    genes = json.loads(target_genes.read_text(encoding="utf-8"))
    if len(source_gene_ids) != 978 or set(source_gene_ids) != set(genes):
        raise ValueError("XPert and EXP-037 do not share the same 978 gene identities")
    target_position = {gene: index for index, gene in enumerate(genes)}
    source_to_target = [target_position[gene] for gene in source_gene_ids]
    xpert = pd.read_csv(source_drugs)
    target = pd.read_parquet(target_drugs)
    if target["pert_id"].duplicated().any():
        raise ValueError("EXP-037 target drug IDs are not unique")
    grouped = xpert.groupby("pert_id", sort=True)
    if grouped["inchi_key"].nunique().gt(1).any() or grouped["drug_node_idx"].nunique().gt(1).any():
        raise ValueError("XPert source maps a drug ID to multiple structures or nodes")
    unique_source = xpert.drop_duplicates("pert_id")[["pert_id", "inchi_key", "drug_node_idx"]]
    joined = target.merge(unique_source, on="pert_id", how="left", validate="one_to_one",
                          suffixes=("_target", "_xpert"))
    matched = joined["drug_node_idx"].notna() & joined["inchi_key_target"].eq(joined["inchi_key_xpert"])
    matched_drugs = set(joined.loc[matched, "pert_id"])
    mismatched_structure = joined.loc[joined["drug_node_idx"].notna() & ~matched,
                                    ["pert_id", "inchi_key_target", "inchi_key_xpert"]].to_dict("records")
    treatments = pd.read_parquet(target_contract / "treatments.parquet")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    results = {}
    for name, column, checkpoint in (
        ("warm_split_1", "split_1", warm_checkpoint),
        ("cold_cell_1", "split_cold_cell_1", cold_checkpoint),
    ):
        train = obs.loc[obs[column].eq("train")]
        train_cells = set(train["cell_iname"])
        train_instances = {instance.strip() for value in train["sample_id"].astype(str)
                           for instance in value.split(";") if instance.strip()}
        folds = []
        for fold in split["folds"]:
            test_cells = set(fold["outer_test_ids"])
            test_treatment = treatments.loc[treatments["cell_id"].isin(test_cells)]
            exact_instances = set(test_treatment["inst_id"]) & train_instances
            folds.append({"fold": fold["fold"], "n_test_cells": len(test_cells),
                          "exposed_test_cells": sorted(test_cells & train_cells),
                          "n_exposed_test_instances": len(exact_instances),
                          "n_test_instances": len(test_treatment),
                          "n_test_instances_with_xpert_drug_features": int(
                              test_treatment["pert_id"].isin(matched_drugs).sum())})
        results[name] = {"checkpoint_sha256": sha256(checkpoint),
                         "source_training_rows": len(train),
                         "source_training_cells": len(train_cells),
                         "n_source_training_instances_parsed": len(train_instances),
                         "strict_unseen_cell_eligible_all_folds": all(
                             not row["exposed_test_cells"] for row in folds),
                         "folds": folds}
    output_dir.mkdir(parents=True)
    gene_map = output_dir / "source_to_target_gene_indices.json"
    gene_map.write_text(json.dumps(source_to_target) + "\n", encoding="utf-8")
    report = {"experiment_id": "EXP-037", "status": "XPERT_ASSET_EXPOSURE_AUDITED",
              "scope": "checkpoint exposure, exact drug structure, and 978-gene axis only; no formal inference",
              "source_h5ad_sha256": sha256(source_h5ad),
              "target_split_sha256": sha256(split_path),
              "source_drugs_sha256": sha256(source_drugs),
              "target_drugs_sha256": sha256(target_drugs),
              "source_gene_order_equals_target": source_gene_ids == genes,
              "source_to_target_gene_indices_sha256": sha256(gene_map),
              "n_target_drugs": len(target), "n_exact_id_and_structure_matched_drugs": len(matched_drugs),
              "id_matched_but_structure_mismatched": mismatched_structure,
              "checkpoints": results}
    (output_dir / "asset_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    return {"status": report["status"], "n_target_drugs": len(target),
            "n_exact_matched_drugs": len(matched_drugs),
            "source_gene_order_equals_target": source_gene_ids == genes,
            "checkpoint_exposure": {key: [len(row["exposed_test_cells"]) for row in value["folds"]]
                                    for key, value in results.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5ad", type=Path, required=True)
    parser.add_argument("--warm-checkpoint", type=Path, required=True)
    parser.add_argument("--cold-checkpoint", type=Path, required=True)
    parser.add_argument("--source-drugs", type=Path, required=True)
    parser.add_argument("--target-drugs", type=Path, required=True)
    parser.add_argument("--target-contract", type=Path, required=True)
    parser.add_argument("--target-genes", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.source_h5ad, args.warm_checkpoint, args.cold_checkpoint,
                           args.source_drugs, args.target_drugs, args.target_contract,
                           args.target_genes, args.split, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
