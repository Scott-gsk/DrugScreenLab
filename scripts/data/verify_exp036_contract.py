"""独立核验 EXP-036 合同的轴、响应、结构与患者/细胞划分。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(root: Path, contract_dir: Path, split_path: Path) -> dict:
    manifest = json.loads((contract_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        if sha256(contract_dir / name) != expected:
            raise ValueError(f"contract checksum mismatch: {name}")
    genes = json.loads((contract_dir / "genes.json").read_text())
    samples = pd.read_parquet(contract_dir / "target_samples.parquet")
    source_pairs = pd.read_parquet(contract_dir / "source_pairs.parquet")
    target_pairs = pd.read_parquet(contract_dir / "target_pairs.parquet")
    drugs = pd.read_parquet(contract_dir / "drugs.parquet")
    target_expression = np.load(contract_dir / "target_expression.npy", mmap_mode="r")
    fingerprints = np.load(contract_dir / "drug_morgan2048.npy", mmap_mode="r")
    with np.load(root / "processed/exp012/gdsc2_cell_line_indexed.npz", allow_pickle=True) as source:
        source_genes = source["ordered_genes"].astype(str).tolist()
        source_ids = set(source["sample_id"].astype(str))
    with np.load(root / "processed/transfer/mcrc_v3/indexed.npz", allow_pickle=True) as target:
        target_ids = target["sample_id"].astype(str)
        target_genes = target["ordered_genes"].astype(str)
        position = {value: i for i, value in enumerate(target_ids)}
        gene_position = {value: i for i, value in enumerate(target_genes)}
        expected = np.asarray(target["expression"], dtype=np.float32)[
            np.ix_([position[x] for x in samples["sample_id"].astype(str)],
                   [gene_position[x] for x in genes])]
    if genes != source_genes or len(set(genes)) != len(genes):
        raise ValueError("contract gene order differs from source expression axis")
    if not np.array_equal(expected, target_expression, equal_nan=True):
        raise ValueError("target expression values/axis differ from original indexed matrix")
    original_source = pd.read_parquet(root / "processed/exp012/gdsc2_cell_line_pairs.parquet")
    original_target = pd.read_parquet(root / "processed/transfer/mcrc_v3/pairs.parquet")
    common = set(drugs["drug_id"])
    expected_source = original_source.loc[original_source["drug_id"].isin(common)].reset_index(drop=True)
    expected_target = original_target.loc[
        original_target["drug_id"].isin(common)
        & original_target["sample_id"].isin(set(samples["sample_id"]))].reset_index(drop=True)
    pd.testing.assert_frame_equal(source_pairs.reset_index(drop=True), expected_source, check_exact=True)
    pd.testing.assert_frame_equal(target_pairs.reset_index(drop=True), expected_target, check_exact=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for row in drugs.itertuples(index=False):
        expected_fp = generator.GetFingerprintAsNumPy(
            Chem.MolFromSmiles(row.canonical_isomeric_smiles)).astype(np.uint8)
        if not np.array_equal(expected_fp, fingerprints[row.drug_index]):
            raise ValueError(f"drug fingerprint mismatch: {row.drug_id}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    target_patients = set(samples["patient_id"].astype(str))
    for fold in split["target_folds"]:
        train, test = set(fold["outer_train_ids"]), set(fold["outer_test_ids"])
        if train & test or train | test != target_patients:
            raise ValueError("target patient split overlap or omission")
    for fold in split["source_folds"]:
        train, test = set(fold["outer_train_ids"]), set(fold["outer_test_ids"])
        if train & test or train | test != source_ids:
            raise ValueError("source cell split overlap or omission")
    return {"status": "PASS", "n_genes": len(genes), "n_drugs": len(drugs),
            "n_source_pairs": len(source_pairs), "n_target_pairs": len(target_pairs),
            "n_target_patients": len(target_patients), "verified_outer_folds": 5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root, args.contract_dir, args.split)
    (args.contract_dir / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
