"""Build patient-level EXP-036 inputs using frozen source PharmaFormer transforms."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from subword_nmt.apply_bpe import BPE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize(contract_dir: Path, checkpoint_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"PharmaFormer input output exists: {output_dir}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["variant"] != "M2" or checkpoint["gene_transform"]["fit_scope"] != "train_only":
        raise ValueError("checkpoint is not a source-only frozen transform")
    samples = pd.read_parquet(contract_dir / "target_samples.parquet")
    pairs = pd.read_parquet(contract_dir / "target_pairs.parquet")
    drugs = pd.read_parquet(contract_dir / "drugs.parquet")
    expression = np.load(contract_dir / "target_expression.npy", mmap_mode="r")
    genes = json.loads((contract_dir / "genes.json").read_text(encoding="utf-8"))
    if (expression.shape != (len(samples), len(genes))
            or len(genes) != checkpoint["model_config"]["gene_input_size"]
            or samples["sample_id"].duplicated().any()
            or drugs["drug_id"].duplicated().any()
            or pairs.duplicated(["sample_id", "drug_id"]).any()):
        raise ValueError("target expression, sample or drug axes differ from checkpoint/contract")
    if (not pairs["sample_id"].isin(samples["sample_id"]).all()
            or not pairs["drug_id"].isin(drugs["drug_id"]).all()
            or not pairs["higher_is_sensitive"].all()
            or set(pairs["response_name"]) != {"DSS"}):
        raise ValueError("target response identity or direction invalid")
    transform = checkpoint["gene_transform"]
    mean = np.asarray(transform["mean"], dtype=np.float64)
    scale = np.asarray(transform["scale"], dtype=np.float64)
    minimum = np.asarray(transform["min"], dtype=np.float64)
    span = np.asarray(transform["range"], dtype=np.float64)
    if any(array.shape != (len(genes),) for array in (mean, scale, minimum, span)):
        raise ValueError("checkpoint transform axis mismatch")
    if (not all(np.isfinite(array).all() for array in (mean, scale, minimum, span))
            or (scale <= 0).any() or (span <= 0).any()):
        raise ValueError("checkpoint transform has invalid scale")
    raw = np.asarray(expression, dtype=np.float64)
    if np.isinf(raw).any():
        raise ValueError("target expression contains infinite values")
    transformed = ((np.where(np.isnan(raw), mean, raw) - mean) / scale - minimum) / span
    if not np.isfinite(transformed).all():
        raise ValueError("transformed target expression is non-finite")
    patient_axis = sorted(samples["patient_id"].astype(str).unique())
    sample_position = {value: index for index, value in enumerate(samples["sample_id"].astype(str))}
    patient_expression = np.stack([
        transformed[[sample_position[s] for s in samples.loc[samples["patient_id"].astype(str).eq(patient),
                                                              "sample_id"].astype(str)]].mean(axis=0)
        for patient in patient_axis]).astype(np.float32)
    response = pairs.merge(samples[["sample_id", "patient_id"]], on="sample_id",
                           how="left", validate="many_to_one")
    if response["patient_id"].isna().any() or not np.isfinite(response["response"]).all():
        raise ValueError("target responses lack patient or finite label")
    patient_pairs = response.groupby(["patient_id", "drug_id"], sort=True, as_index=False).agg(
        response=("response", "mean"), n_sample_observations=("sample_id", "size"))
    if patient_pairs["patient_id"].nunique() != len(patient_axis):
        raise ValueError("some admitted patients have no drug response")
    frozen_encoder = checkpoint["smiles_encoder"]
    codes = str(frozen_encoder["bpe_codes"])
    if hashlib.sha256(codes.encode()).hexdigest() != frozen_encoder["bpe_sha256"]:
        raise ValueError("frozen source BPE checksum mismatch")
    if not set(drugs["canonical_isomeric_smiles"].astype(str)) <= set(frozen_encoder["allowed_smiles"]):
        raise ValueError("target drug absent from frozen source SMILES vocabulary")
    bpe = BPE(io.StringIO(codes))
    drug_axis = drugs.sort_values("drug_index").reset_index(drop=True)
    drug_ordinal = np.zeros((len(drug_axis), 128), dtype=np.float32)
    for index, smiles in enumerate(drug_axis["canonical_isomeric_smiles"].astype(str)):
        encoded = [ord(char) for char in bpe.process_line(smiles)[:128]]
        drug_ordinal[index, :len(encoded)] = encoded
    output_dir.mkdir(parents=True)
    np.save(output_dir / "patient_expression.npy", patient_expression)
    np.save(output_dir / "drug_ordinal128.npy", drug_ordinal)
    (output_dir / "patient_axis.json").write_text(json.dumps(patient_axis) + "\n", encoding="utf-8")
    drug_axis.to_parquet(output_dir / "drug_axis.parquet", index=False)
    patient_pairs.to_parquet(output_dir / "patient_pairs.parquet", index=False)
    report = {
        "experiment_id": "EXP-036", "status": "PHARMAFORMER_TARGET_INPUTS_MATERIALIZED_PENDING_INDEPENDENT_VERIFY",
        "source_transform": "checkpoint source-train-only mean imputation, StandardScaler then MinMaxScaler",
        "source_drug_encoding": "checkpoint source-train-only BPE then ordinal padded to width 128",
        "patient_aggregation": "transform each admitted sample, then sample-equal mean within patient; mean response within patient/drug",
        "n_patients": len(patient_axis), "n_drugs": len(drug_axis),
        "n_patient_drug_labels": len(patient_pairs), "n_source_sample_drug_labels": len(pairs),
        "n_target_missing_expression_imputed": int(np.isnan(raw).sum()),
        "contract_manifest_sha256": sha256(contract_dir / "manifest.json"),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "frozen_bpe_sha256": frozen_encoder["bpe_sha256"],
        "files_sha256": {name: sha256(output_dir / name) for name in
                         ("patient_expression.npy", "drug_ordinal128.npy", "patient_axis.json",
                          "drug_axis.parquet", "patient_pairs.parquet")},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(args.contract_dir, args.checkpoint, args.output_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
