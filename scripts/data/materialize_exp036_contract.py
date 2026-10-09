"""物化 EXP-036 的来源/目标共同药物、共同基因及显式身份合同。"""

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


def materialize(root: Path, source_audit_path: Path, split_path: Path,
                output_dir: Path) -> dict:
    audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    required = ("response", "expression", "cell_identity", "drug_identity")
    if any(audit[name]["status"] != "PASS" for name in required):
        raise ValueError("GDSC source audit has unresolved conflicts")
    source_root = root / "processed/exp012"
    target_root = root / "processed/transfer/mcrc_v3"
    admission_path = root / "processed/identity/transfer/kryeziu_mcrc_target_admission.json"
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    admitted_samples = set(map(str, admission["admitted_sample_ids"]))
    admitted_patients = set(map(str, admission["admitted_patient_ids"]))
    source_samples = pd.read_parquet(source_root / "gdsc2_cell_line_samples.parquet")
    source_drugs = pd.read_parquet(source_root / "gdsc2_cell_line_drugs.parquet")
    source_pairs = pd.read_parquet(source_root / "gdsc2_cell_line_pairs.parquet")
    target_samples = pd.read_parquet(target_root / "samples.parquet")
    target_drugs = pd.read_parquet(target_root / "drugs.parquet")
    target_pairs = pd.read_parquet(target_root / "pairs.parquet")
    target_samples = target_samples.loc[target_samples["sample_id"].astype(str).isin(admitted_samples)].copy()
    if set(target_samples["sample_id"].astype(str)) != admitted_samples:
        raise ValueError("admitted target sample identity mismatch")
    if set(target_samples["patient_id"].astype(str)) != admitted_patients:
        raise ValueError("admitted target patient identity mismatch")
    common_drugs = sorted(set(source_drugs["drug_id"].astype(str)) &
                          set(target_drugs["drug_id"].astype(str)))
    if len(common_drugs) != 14:
        raise ValueError(f"expected 14 structure-identical common drugs, got {len(common_drugs)}")
    source_pairs = source_pairs.loc[source_pairs["drug_id"].astype(str).isin(common_drugs)].copy()
    target_pairs = target_pairs.loc[
        target_pairs["sample_id"].astype(str).isin(admitted_samples)
        & target_pairs["drug_id"].astype(str).isin(common_drugs)].copy()
    for name, frame in (("source", source_pairs), ("target", target_pairs)):
        if frame.duplicated(["sample_id", "drug_id"]).any():
            raise ValueError(f"{name} has duplicate sample-drug response keys")
        if not np.isfinite(frame["response"].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} response contains non-finite values")
    if set(source_pairs["response_name"]) != {"1_minus_auc"} or not source_pairs["higher_is_sensitive"].all():
        raise ValueError("source response direction/endpoint mismatch")
    if set(target_pairs["response_name"]) != {"DSS"} or not target_pairs["higher_is_sensitive"].all():
        raise ValueError("target response direction/endpoint mismatch")

    with np.load(source_root / "gdsc2_cell_line_indexed.npz", allow_pickle=True) as source:
        source_expression = np.asarray(source["expression"], dtype=np.float32)
        source_ids = source["sample_id"].astype(str)
        source_genes = source["ordered_genes"].astype(str)
    with np.load(target_root / "indexed.npz", allow_pickle=True) as target:
        target_expression_all = np.asarray(target["expression"], dtype=np.float32)
        target_ids_all = target["sample_id"].astype(str)
        target_genes = target["ordered_genes"].astype(str)
    if not np.array_equal(source_ids, source_samples["sample_id"].astype(str)):
        raise ValueError("source expression/sample row identity mismatch")
    target_position = {sample: index for index, sample in enumerate(target_ids_all)}
    target_rows = np.array([target_position[x] for x in target_samples["sample_id"].astype(str)], dtype=int)
    target_gene_position = {gene: index for index, gene in enumerate(target_genes)}
    common_genes = [gene for gene in source_genes if gene in target_gene_position]
    if len(common_genes) != len(source_genes) or len(set(common_genes)) != len(common_genes):
        raise ValueError("target does not contain the full unique source gene axis")
    target_columns = np.array([target_gene_position[gene] for gene in common_genes], dtype=int)
    target_expression = target_expression_all[np.ix_(target_rows, target_columns)]
    if not np.isfinite(target_expression).all():
        raise ValueError("admitted target expression contains non-finite values")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    genes_path = output_dir / "genes.json"
    genes_path.write_text(json.dumps(common_genes) + "\n", encoding="utf-8")
    np.save(output_dir / "target_expression.npy", target_expression)
    target_samples.to_parquet(output_dir / "target_samples.parquet", index=False)
    source_pairs.to_parquet(output_dir / "source_pairs.parquet", index=False)
    target_pairs.to_parquet(output_dir / "target_pairs.parquet", index=False)

    target_drug_index = target_drugs.set_index("drug_id", verify_integrity=True)
    fingerprint = np.zeros((len(common_drugs), 2048), dtype=np.uint8)
    names, smiles = [], []
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for index, drug_id in enumerate(common_drugs):
        row = target_drug_index.loc[drug_id]
        molecule = Chem.MolFromSmiles(row["canonical_isomeric_smiles"])
        if molecule is None:
            raise ValueError(f"common drug has invalid structure: {drug_id}")
        fingerprint[index] = generator.GetFingerprintAsNumPy(molecule).astype(np.uint8)
        names.append(row["drug_name"])
        smiles.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
    drug_table = pd.DataFrame({"drug_index": range(len(common_drugs)), "drug_id": common_drugs,
                               "drug_name": names, "canonical_isomeric_smiles": smiles})
    drug_table.to_parquet(output_dir / "drugs.parquet", index=False)
    np.save(output_dir / "drug_morgan2048.npy", fingerprint)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if set().union(*(set(fold["outer_test_ids"]) for fold in split["target_folds"])) != admitted_patients:
        raise ValueError("frozen target folds do not cover admitted patients exactly")
    if set().union(*(set(fold["outer_test_ids"]) for fold in split["source_folds"])) != set(source_ids):
        raise ValueError("frozen source folds do not cover source cells exactly")
    files = ["genes.json", "target_expression.npy", "target_samples.parquet",
             "source_pairs.parquet", "target_pairs.parquet", "drugs.parquet",
             "drug_morgan2048.npy"]
    report = {
        "protocol": "EXP-036 direct-response contract v1", "status": "PASS",
        "source": {"n_cells": len(source_ids), "n_pairs_common_drugs": len(source_pairs),
                   "expression_shape": list(source_expression.shape),
                   "nonfinite_expression_values": int((~np.isfinite(source_expression)).sum()),
                   "endpoint": "1_minus_auc"},
        "target": {"n_patients": len(admitted_patients), "n_samples": len(target_samples),
                   "n_pairs_common_drugs": len(target_pairs),
                   "expression_shape": list(target_expression.shape), "endpoint": "DSS"},
        "n_common_drugs": len(common_drugs), "n_common_genes": len(common_genes),
        "preprocessing": "source missing-value imputation and all scaling must be fit independently inside each training fold",
        "source_audit_sha256": sha256(source_audit_path), "split_sha256": sha256(split_path),
        "admission_sha256": sha256(admission_path),
        "files_sha256": {name: sha256(output_dir / name) for name in files},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.root, args.source_audit, args.split, args.output_dir),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
