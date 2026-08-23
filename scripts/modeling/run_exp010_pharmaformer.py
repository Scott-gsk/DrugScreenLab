"""Run one predeclared EXP-010 PharmaFormer fold (M2, M3 or M4).

This command intentionally accepts already-materialized, fold-specific NPZ
arrays.  It never discovers/splits rows and therefore cannot silently recreate
the pair-level KFold logic in the upstream scripts.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from drug_screen.foundation.pharmaformer_adapter import (  # noqa: E402
    PharmaFormerConfig, TrainOnlyGeneTransform, TrainOnlySmilesEncoder,
    build_model, load_checkpoint, load_upstream_module, save_checkpoint,
    validate_patient_fold,
)
from drug_screen.foundation.pharmaformer_training import fit_model, predict, training_policy  # noqa: E402


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _load_rows(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(x, dict) for x in value):
        raise ValueError("fold manifest must be a JSON list of row objects")
    return value


def run(args: argparse.Namespace) -> dict[str, object]:
    rows = _load_rows(args.fold_manifest)
    partitions = validate_patient_fold(rows) if args.domain == "target_mcrc" else {}
    arrays = np.load(args.arrays, allow_pickle=False)
    required = {f"{p}_{x}" for p in ("train", "validation", "test") for x in ("gene", "smiles", "y", "sample_id", "drug_id")}
    missing = required - set(arrays.files)
    if missing:
        raise ValueError(f"fold NPZ misses arrays: {sorted(missing)}")
    train_gene = arrays["train_gene"]
    config = PharmaFormerConfig(gene_input_size=train_gene.shape[1], gene_hidden_size=args.gene_hidden_size,
                                drug_hidden_size=args.drug_hidden_size, feature_dim=args.feature_dim,
                                nhead=args.nhead, num_layers=args.num_layers,
                                dim_feedforward=args.dim_feedforward, dropout=args.dropout)
    upstream = load_upstream_module(args.upstream)
    action = training_policy(args.variant, args.domain)
    frozen_inference = action == "frozen_inference"
    if args.variant == "M3" or frozen_inference:
        if args.source_checkpoint is None:
            raise ValueError("M3 and target-domain M2 require the frozen source-domain M2 checkpoint")
        model, source = load_checkpoint(args.source_checkpoint, upstream, device=args.device)
        source_config = dict(source["model_config"])
        requested_config = dict(config.__dict__)
        source_dropout, requested_dropout = source_config.pop("dropout"), requested_config.pop("dropout")
        if source["variant"] != "M2" or source_config != requested_config:
            raise ValueError("M3 source must be a compatible M2 checkpoint")
        if frozen_inference and source_dropout != requested_dropout:
            raise ValueError("frozen M2 inference must retain source dropout config")
        if args.variant == "M3" and source_dropout != requested_dropout:
            import torch
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = float(requested_dropout)
        transform = source["gene_transform_object"]
        encoder = source["smiles_encoder_object"]
    else:
        if args.source_checkpoint is not None:
            raise ValueError("only M3 may inherit a source checkpoint")
        model = build_model(upstream, config)
        transform = TrainOnlyGeneTransform().fit(train_gene, arrays["train_sample_id"].astype(str))
        if args.bpe_codes is None:
            raise ValueError("fitting requires an explicit frozen source BPE codes asset")
        encoder = TrainOnlySmilesEncoder().fit(arrays["train_smiles"].astype(str), arrays["train_drug_id"].astype(str),
                                               bpe_codes_path=args.bpe_codes)
    # For M2, the test drugs must be represented by the source-training set. For
    # target folds an unseen drug is likewise rejected rather than target-fitted.
    x = {p: transform.transform(arrays[f"{p}_gene"]) for p in ("train", "validation", "test")}
    drug = {p: encoder.transform(arrays[f"{p}_smiles"].astype(str)) for p in ("train", "validation", "test")}
    fit = None if frozen_inference else fit_model(
        model, x["train"], drug["train"], arrays["train_y"], x["validation"], drug["validation"],
        arrays["validation_y"], device=args.device, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, patience=args.patience, seed=args.seed)
    prediction = predict(model, x["test"], drug["test"], device=args.device, batch_size=args.batch_size)
    checkpoint = args.source_checkpoint if frozen_inference else save_checkpoint(
        args.output / f"{args.variant}_fold{args.fold_id}.pt", model, config, transform, encoder,
        variant=args.variant, fold_id=args.fold_id,
        provenance={"training_domain": args.domain, "arrays_sha256": _digest(args.arrays),
                    "fold_manifest_sha256": _digest(args.fold_manifest),
                    "partitions": {k: sorted(v) for k, v in partitions.items()}, "seed": args.seed})
    assert checkpoint is not None
    args.output.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output / f"{args.variant}_fold{args.fold_id}_predictions.npz"
    np.savez_compressed(predictions_path, sample_id=arrays["test_sample_id"], drug_id=arrays["test_drug_id"],
                        observed=arrays["test_y"], predicted=prediction)
    summary = {"experiment_id": "EXP-010", "variant": args.variant, "fold_id": args.fold_id,
               "domain": args.domain, "training_policy": action, "fit": None if fit is None else fit.__dict__,
               "frozen_source_inference": frozen_inference,
               "checkpoint": str(checkpoint), "checkpoint_sha256": _digest(checkpoint),
               "predictions": str(predictions_path), "predictions_sha256": _digest(predictions_path),
               "data_contract": "caller_materialized_patient_fold; transforms fit on train only"}
    summary_path = args.output / f"{args.variant}_fold{args.fold_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("M2", "M3", "M4"), required=True)
    parser.add_argument("--domain", choices=("source_gdsc", "target_mcrc"), required=True)
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--bpe-codes", type=Path,
                        help="required only when fitting a new M2/M4; frozen M2/M3 reads codes from checkpoint")
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--gene-hidden-size", type=int, default=8192)
    parser.add_argument("--drug-hidden-size", type=int, default=256)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.3)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
