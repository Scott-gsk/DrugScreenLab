"""Strictly reload an existing source-only PharmaFormer weight into official code."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
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


def smoke(checkpoint_path: Path, official_source: Path,
          contract_dir: Path | None = None) -> dict:
    spec = importlib.util.spec_from_file_location("pharmaformer_official_pretrain", official_source)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load official PharmaFormer source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["variant"] != "M2" or checkpoint["model_variant"] != "CombinedModel":
        raise ValueError("checkpoint is not a source-only CombinedModel")
    config = checkpoint["model_config"]
    model = module.CombinedModel(
        gene_input_size=config["gene_input_size"],
        gene_hidden_size=config["gene_hidden_size"],
        drug_hidden_size=config["drug_hidden_size"],
        feature_dim=config["feature_dim"], nhead=config["nhead"],
        num_layers=config["num_layers"], dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, config["gene_input_size"]), torch.zeros(1, 128))
    if output.shape != (1, 1) or not torch.isfinite(output).all():
        raise ValueError("strictly reloaded official model has invalid output")
    result = {"status": "STRICT_OFFICIAL_RELOAD_PASS_ENGINEERING_ONLY",
            "checkpoint_sha256": sha256(checkpoint_path),
            "official_source_sha256": sha256(official_source),
            "n_model_parameters": sum(param.numel() for param in model.parameters()),
            "gene_input_size": config["gene_input_size"],
            "output_shape": list(output.shape),
            "note": "zero-input smoke validates architecture and weights only; not expression preprocessing or efficacy"}
    if contract_dir is not None:
        from subword_nmt.apply_bpe import BPE

        samples = pd.read_parquet(contract_dir / "target_samples.parquet")
        drugs = pd.read_parquet(contract_dir / "drugs.parquet")
        expression = np.load(contract_dir / "target_expression.npy", mmap_mode="r")
        if expression.shape != (len(samples), config["gene_input_size"]):
            raise ValueError("target expression axis differs from official model")
        transform = checkpoint["gene_transform"]
        values = np.asarray(expression[0:1], dtype=np.float64)
        mean = np.asarray(transform["mean"])
        values = np.where(np.isnan(values), mean, values)
        normalized = ((values - mean) / np.asarray(transform["scale"])
                      - np.asarray(transform["min"])) / np.asarray(transform["range"])
        if not np.isfinite(normalized).all():
            raise ValueError("real target expression transforms to non-finite input")
        smiles = str(drugs.iloc[0]["canonical_isomeric_smiles"])
        encoder = checkpoint["smiles_encoder"]
        if smiles not in encoder["allowed_smiles"]:
            raise ValueError("real panel drug missing from frozen source encoder")
        if hashlib.sha256(encoder["bpe_codes"].encode()).hexdigest() != encoder["bpe_sha256"]:
            raise ValueError("checkpoint BPE asset checksum mismatch")
        encoded = [ord(char) for char in BPE(io.StringIO(encoder["bpe_codes"]))
                   .process_line(smiles)[:128]]
        drug_vector = np.zeros((1, 128), dtype=np.float32)
        drug_vector[0, :len(encoded)] = encoded
        with torch.no_grad():
            real_output = model(torch.from_numpy(normalized.astype(np.float32)),
                                torch.from_numpy(drug_vector))
        if real_output.shape != (1, 1) or not torch.isfinite(real_output).all():
            raise ValueError("official model fails on a real target expression/drug input")
        result["status"] = "STRICT_RELOAD_AND_REAL_INPUT_SMOKE_PASS_ENGINEERING_ONLY"
        result["real_input"] = {"sample_id": str(samples.iloc[0]["sample_id"]),
                                "drug_id": str(drugs.iloc[0]["drug_id"]),
                                "frozen_bpe_sha256": encoder["bpe_sha256"],
                                "transformed_expression_finite": True,
                                "output_finite": True}
        result["note"] = "real input validates frozen preprocessing and inference shape only; no target response used"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = smoke(args.checkpoint, args.official_source, args.contract_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
