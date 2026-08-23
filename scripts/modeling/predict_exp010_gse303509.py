"""Frozen, label-blind GSE303509 cisplatin inference for EXP-010."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from drug_screen.foundation.pharmaformer_adapter import load_checkpoint, load_upstream_module  # noqa: E402
from drug_screen.foundation.pharmaformer_training import predict  # noqa: E402

TORGS = ("TORG40", "TORG51", "TORG58", "TORG139", "TORG146", "TORG43", "TORG155")

def file_digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument("--source-genes", type=Path, required=True)
    parser.add_argument("--pubchem", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--m2-checkpoint", type=Path, required=True)
    parser.add_argument("--m3-checkpoints", type=Path, nargs=5, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    genes = args.source_genes.read_text(encoding="utf-8").splitlines()
    if len(genes) != 8192 or len(set(genes)) != 8192:
        raise ValueError("source gene contract must contain 8192 unique ordered genes")
    expression = pd.read_parquet(args.expression)
    if not set(TORGS).issubset(expression.index.astype(str)) or not set(genes).issubset(expression.columns):
        raise ValueError("GSE303509 misses frozen TORGs or source genes")
    expression = expression.loc[list(TORGS), genes]
    structures = json.loads(args.pubchem.read_text(encoding="utf-8"))
    cisplatin = next((x for x in structures if str(x.get("primary_name", "")).casefold() == "cisplatin" and x.get("mapped")), None)
    if cisplatin is None or not cisplatin.get("canonical_smiles"):
        raise ValueError("cisplatin has no frozen PubChem structure")
    smiles = str(cisplatin["canonical_smiles"])
    upstream = load_upstream_module(args.upstream)
    checkpoints = [("M2", args.m2_checkpoint)] + [(f"M3_fold{i}", p) for i, p in enumerate(args.m3_checkpoints)]
    rows: dict[str, np.ndarray] = {}
    audits = []
    for name, checkpoint_path in checkpoints:
        model, payload = load_checkpoint(checkpoint_path, upstream, device=args.device)
        transform = payload["gene_transform_object"]
        encoder = payload["smiles_encoder_object"]
        if encoder.allowed_ is None or smiles not in encoder.allowed_:
            raise ValueError(f"cisplatin is absent from frozen source vocabulary for {name}")
        gene = transform.transform(expression.to_numpy(float))
        drug = encoder.transform([smiles] * len(TORGS))
        values = predict(model, gene, drug, device=args.device, batch_size=4)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite {name} breast predictions")
        rows[name] = values
        audits.append({"model": name, "checkpoint": str(checkpoint_path),
                       "checkpoint_sha256": file_digest(checkpoint_path), "prediction_range": [float(values.min()), float(values.max())]})
    frame = pd.DataFrame({"torg_id": TORGS, **rows})
    frame["M3_ensemble"] = frame[[f"M3_fold{i}" for i in range(5)]].mean(axis=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "gse303509_cisplatin_unlabeled_predictions.csv", index=False)
    for column in ("M2", "M3_ensemble"):
        frame[["torg_id", column]].rename(columns={column: "predicted_score"}).to_csv(
            args.output_dir / f"{column}_evaluation_input.csv", index=False)
    audit = {"experiment_id": "EXP-010", "labels_read": False, "cohort": list(TORGS), "drug": "cisplatin",
             "canonical_smiles": smiles, "pubchem_cid": cisplatin.get("pubchem_cid"),
             "source_vocabulary_supported": True, "source_genes": 8192,
             "source_genes_sha256": file_digest(args.source_genes), "models": audits,
             "ensemble_rule": "unweighted arithmetic mean of all five predeclared M3 patient-fold checkpoints"}
    (args.output_dir / "inference_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
