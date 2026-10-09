"""Audit reusable source-only PharmaFormer checkpoints for EXP-036 input identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(source_index: Path, contract_dir: Path, checkpoint_root: Path,
          official_repo: Path) -> dict:
    source = np.load(source_index, allow_pickle=True)
    genes = json.loads((contract_dir / "genes.json").read_text(encoding="utf-8"))
    if list(map(str, source["ordered_genes"])) != genes:
        raise ValueError("existing source checkpoint gene axis differs from EXP-036 contract")
    source_ids = set(map(str, source["sample_id"]))
    target_ids = set(pd.read_parquet(contract_dir / "target_samples.parquet")["sample_id"].astype(str))
    target_drugs = set(pd.read_parquet(contract_dir / "drugs.parquet")["canonical_isomeric_smiles"].astype(str))
    repo_commit = subprocess.check_output(["git", "-C", str(official_repo), "rev-parse", "HEAD"],
                                          text=True).strip()
    if repo_commit != "f526ddda4091151f082b38367571a94823314aba":
        raise ValueError("official PharmaFormer checkout changed revision")
    checks = []
    for fold in range(5):
        path = checkpoint_root / f"fold{fold}" / f"M2_fold{fold}.pt"
        summary_path = path.with_name(f"M2_fold{fold}_summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if sha256(path) != summary["checkpoint_sha256"]:
            raise ValueError(f"source checkpoint {fold} checksum differs from its receipt")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (checkpoint["experiment_id"] != "EXP-011" or checkpoint["variant"] != "M2"
                or checkpoint["fold_id"] != fold or checkpoint["upstream_commit"] != repo_commit
                or checkpoint["model_variant"] != "CombinedModel"
                or checkpoint["gene_transform"]["fit_scope"] != "train_only"
                or checkpoint["smiles_encoder"]["fit_scope"] != "source_train_only"):
            raise ValueError(f"source checkpoint {fold} metadata incompatible")
        fitted = set(map(str, checkpoint["gene_transform"]["fit_sample_ids"]))
        if not fitted or not fitted <= source_ids or fitted & target_ids:
            raise ValueError(f"source checkpoint {fold} fit identities overlap target or leave source")
        available_smiles = set(map(str, checkpoint["smiles_encoder"]["allowed_smiles"]))
        if not target_drugs <= available_smiles:
            raise ValueError(f"source checkpoint {fold} lacks a target-panel drug")
        config = checkpoint["model_config"]
        if config["gene_input_size"] != len(genes) or config["drug_hidden_size"] != 256:
            raise ValueError(f"source checkpoint {fold} architecture input mismatch")
        first = checkpoint["model_state_dict"]["feature_extractor.gene_fc1.weight"]
        if tuple(first.shape) != (config["gene_hidden_size"], len(genes)):
            raise ValueError(f"source checkpoint {fold} state dimensions mismatch")
        checks.append({"fold": fold, "checkpoint_sha256": sha256(path),
                       "summary_sha256": sha256(summary_path),
                       "n_source_fit_cells": len(fitted),
                       "n_target_drugs_covered": len(target_drugs),
                       "gene_input_size": len(genes),
                       "state_dict_n_tensors": len(checkpoint["model_state_dict"])})
        del checkpoint
    return {"status": "SOURCE_WEIGHTS_IDENTITY_PASS_PENDING_STRICT_MODEL_RELOAD_AND_NEW_PROTOCOL",
            "official_repo_commit": repo_commit, "source_index_sha256": sha256(source_index),
            "exp036_gene_axis_sha256": sha256(contract_dir / "genes.json"),
            "n_target_panel_drugs": len(target_drugs), "checkpoints": checks,
            "scope": "source-only weight identity; old EXP results do not select models or estimate new performance"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source_index, args.contract_dir, args.checkpoint_root, args.official_repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
