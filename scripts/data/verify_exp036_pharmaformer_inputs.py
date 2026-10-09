"""Independently recompute EXP-036 patient/drug axes and source-transformed inputs."""

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


def verify(contract_dir: Path, checkpoint_path: Path, input_dir: Path) -> dict:
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["contract_manifest_sha256"] != sha256(contract_dir / "manifest.json")
            or manifest["source_checkpoint_sha256"] != sha256(checkpoint_path)):
        raise ValueError("input manifest source hashes differ from current files")
    for name, expected in manifest["files_sha256"].items():
        if sha256(input_dir / name) != expected:
            raise ValueError(f"materialized input checksum mismatch: {name}")
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    patients = json.loads((input_dir / "patient_axis.json").read_text(encoding="utf-8"))
    samples = pd.read_parquet(contract_dir / "target_samples.parquet")
    pairs = pd.read_parquet(contract_dir / "target_pairs.parquet")
    drugs = pd.read_parquet(contract_dir / "drugs.parquet").sort_values("drug_index").reset_index(drop=True)
    saved_drugs = pd.read_parquet(input_dir / "drug_axis.parquet")
    pd.testing.assert_frame_equal(saved_drugs, drugs)
    raw = np.load(contract_dir / "target_expression.npy", mmap_mode="r")
    saved_expression = np.load(input_dir / "patient_expression.npy")
    if patients != sorted(samples["patient_id"].astype(str).unique()) or saved_expression.shape != (len(patients), raw.shape[1]):
        raise ValueError("patient expression axis does not match admitted samples")
    transform = source["gene_transform"]
    source_fit = set(map(str, transform["fit_sample_ids"]))
    if source_fit & set(samples["sample_id"].astype(str)):
        raise ValueError("source transformation contains target sample")
    mean = np.asarray(transform["mean"], dtype=np.float64)
    scale = np.asarray(transform["scale"], dtype=np.float64)
    minimum = np.asarray(transform["min"], dtype=np.float64)
    span = np.asarray(transform["range"], dtype=np.float64)
    max_error = 0.0
    for position, patient in enumerate(patients):
        sample_rows = np.flatnonzero(samples["patient_id"].astype(str).eq(patient).to_numpy())
        if not len(sample_rows):
            raise ValueError("empty admitted patient")
        patient_values = []
        for sample_position in sample_rows:
            value = np.asarray(raw[sample_position], dtype=np.float64)
            value = np.where(np.isnan(value), mean, value)
            patient_values.append(((value - mean) / scale - minimum) / span)
        expected = np.mean(np.stack(patient_values), axis=0).astype(np.float32)
        error = float(np.max(np.abs(expected - saved_expression[position])))
        max_error = max(max_error, error)
    if max_error > 1e-6:
        raise ValueError(f"source-transformed patient expression mismatch: {max_error}")
    expected_pairs = (pairs.merge(samples[["sample_id", "patient_id"]], on="sample_id",
                                  how="left", validate="many_to_one")
                      .groupby(["patient_id", "drug_id"], sort=True, as_index=False)
                      .agg(response=("response", "mean"), n_sample_observations=("sample_id", "size")))
    actual_pairs = pd.read_parquet(input_dir / "patient_pairs.parquet")
    pd.testing.assert_frame_equal(actual_pairs, expected_pairs)
    ordinal = np.load(input_dir / "drug_ordinal128.npy")
    if ordinal.shape != (len(drugs), 128) or not np.isfinite(ordinal).all():
        raise ValueError("drug ordinal axis or values invalid")
    encoder = source["smiles_encoder"]
    if hashlib.sha256(str(encoder["bpe_codes"]).encode()).hexdigest() != encoder["bpe_sha256"]:
        raise ValueError("frozen BPE codes checksum invalid")
    bpe = BPE(io.StringIO(str(encoder["bpe_codes"])))
    expected_ordinal = np.zeros_like(ordinal)
    for index, smiles in enumerate(drugs["canonical_isomeric_smiles"].astype(str)):
        encoded = np.fromiter((ord(char) for char in bpe.process_line(smiles)[:128]), dtype=np.float32)
        expected_ordinal[index, :len(encoded)] = encoded
    if not np.array_equal(ordinal, expected_ordinal):
        raise ValueError("drug ordinal values differ from frozen source BPE")
    if manifest["n_patients"] != len(patients) or manifest["n_patient_drug_labels"] != len(actual_pairs):
        raise ValueError("manifest patient or label count mismatch")
    return {"status": "PASS", "n_patients": len(patients), "n_drugs": len(drugs),
            "n_patient_drug_labels": len(actual_pairs),
            "max_patient_expression_error": max_error,
            "source_checkpoint_sha256": sha256(checkpoint_path),
            "input_manifest_sha256": sha256(input_dir / "manifest.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.contract_dir, args.checkpoint, args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
