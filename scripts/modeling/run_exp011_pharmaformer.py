"""Train or infer an EXP-011 PharmaFormer fold from indexed-pair inputs.

The command reads a data-agent-produced manifest under
``$DRUGSCREEN_DATA_ROOT/processed/exp011`` by default.  It never constructs
splits, discovers response files, or expands sample expression per drug.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from drug_screen.foundation.pharmaformer_exp011 import (  # noqa: E402
    EXP011_EXPERIMENT_ID,
    Exp011ModelConfig,
    architecture_digest,
    build_official_m2,
    load_exp011_checkpoint,
    require_official_gene_contract,
    save_exp011_checkpoint,
    sha256_file,
    upstream_checkout,
)
from drug_screen.modeling.exp011_data import IndexedPairDataset, build_external_indices, build_fold_indices, load_indexed_pair_data  # noqa: E402
from drug_screen.modeling.exp011_models import assert_m3_zero_initial_delta, build_variant  # noqa: E402
from drug_screen.modeling.exp011_training import (  # noqa: E402
    TrainOnlyResponseStandardizer,
    build_source_transforms,
    encode_data_smiles,
    fit_gene_transform,
    fit_source_bpe_codes,
    fit_source_model,
    refit_source_model_full,
    fit_target_model,
    predict_dataset,
    transformed_data,
)


def _default_data_root() -> Path:
    return Path(os.environ.get("DRUGSCREEN_DATA_ROOT", ROOT / "data"))


def _require_domain_contract(data: Any, stage: str) -> None:
    if not data.pair_higher_is_sensitive.all():
        raise ValueError("EXP-011 models require response labels oriented so higher means more sensitive")
    if stage in {"source", "source-final"} and set(data.pair_response_name.tolist()) != {"1_minus_auc"}:
        raise ValueError("source stage requires GDSC pair_response_name='1_minus_auc'")


def _dataset(data: Any, indices: np.ndarray, encoded_drugs: np.ndarray) -> IndexedPairDataset:
    if not len(indices):
        raise ValueError("required partition has no pair rows")
    return IndexedPairDataset(data, indices, encoded_drugs)


def _prediction_frame(
    data: Any,
    global_pair_indices: np.ndarray,
    prediction: np.ndarray,
    *,
    model_id: str,
    fold_id: int,
    evaluation_role: str,
) -> pd.DataFrame:
    """Create the evaluator's long-form schema while preserving all identity fields."""
    pair = np.asarray(global_pair_indices, dtype=np.int64)
    if len(pair) != len(prediction):
        raise ValueError("predictions and pair indices must align")
    manifest = dict(data.manifest)
    sample = data.pair_sample_idx[pair]
    drug = data.pair_drug_idx[pair]
    frame = pd.DataFrame({
        "experiment_id": EXP011_EXPERIMENT_ID,
        "model_id": model_id,
        "evaluation_role": evaluation_role,
        "cohort_id": data.pair_cohort_id[pair],
        "fold_id": -1 if evaluation_role in {"prism_frozen", "external_exploratory"} else int(fold_id),
        "split": "external" if evaluation_role in {"prism_frozen", "external_exploratory"} else "test",
        "patient_id": data.patient_id[sample],
        "sample_id": data.sample_id[sample],
        "pdo_id": data.pdo_id[sample],
        "cell_line_id": data.cell_line_id[sample],
        "drug_id": data.drug_id[drug],
        "inchikey": data.inchikey[drug],
        "drug_scaffold": data.drug_scaffold[drug],
        "response_name": data.pair_response_name[pair],
        "response_scale": data.pair_response_scale[pair],
        "higher_is_sensitive": data.pair_higher_is_sensitive[pair],
        "y_true": data.pair_response[pair],
        "y_pred": np.asarray(prediction, dtype=np.float32),
    })
    return frame


def _checkpoint_provenance(data: Any, args: argparse.Namespace, gene_sha: str, bpe_sha: str | None) -> dict[str, Any]:
    input_archive = Path(data.manifest["raw_npz"])
    if not input_archive.is_absolute():
        input_archive = data.manifest_path.parent / input_archive
    return {
        "manifest_path": str(data.manifest_path),
        "manifest_sha256": sha256_file(data.manifest_path),
        "raw_indexed_npz": str(input_archive),
        "raw_indexed_npz_sha256": sha256_file(input_archive),
        "gene_contract_sha256": gene_sha,
        "bpe_codes_sha256": bpe_sha,
        "seed": args.seed,
        "device": args.device,
        "training_stage": args.stage,
        "source_checkpoint": None if args.source_checkpoint is None else str(args.source_checkpoint),
    }


def _frozen_source_final_epoch(summary_paths: list[Path] | None, gate_path: Path | None) -> tuple[int, dict[str, Any]]:
    """Validate the one-way gate before a full-source, fixed-epoch refit."""
    if not summary_paths or len(summary_paths) != 5:
        raise ValueError("source-final requires exactly five --source-fold-summary files")
    if gate_path is None or not gate_path.is_file():
        raise ValueError("source-final requires an existing --cell-line-gate JSON")
    epochs: list[int] = []
    fold_ids: set[int] = set()
    for path in summary_paths:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if document.get("experiment_id") != EXP011_EXPERIMENT_ID or document.get("stage") != "source" or document.get("model_id") != "M2":
            raise ValueError(f"not an M2 source-fold summary: {path}")
        fold_id = document.get("fold_id")
        best_epoch = document.get("fit", {}).get("best_epoch") if isinstance(document.get("fit"), dict) else None
        if not isinstance(fold_id, int) or not isinstance(best_epoch, int) or best_epoch < 1:
            raise ValueError(f"source-fold summary lacks integer fold_id/best_epoch: {path}")
        if fold_id in fold_ids:
            raise ValueError("source-final summaries must describe five distinct outer folds")
        fold_ids.add(fold_id); epochs.append(best_epoch)
    if fold_ids != {0, 1, 2, 3, 4}:
        raise ValueError("source-final requires source fold summaries exactly for fold IDs 0..4")
    gate_document = json.loads(gate_path.read_text(encoding="utf-8"))
    gate = gate_document.get("cell_line_gate", gate_document)
    if not isinstance(gate, dict) or gate.get("format") != "exp011_cell_line_gate_v1" or gate.get("passed") is not True:
        raise ValueError("source-final requires a passed exp011_cell_line_gate_v1 result")
    prism = gate.get("prism_frozen")
    if not isinstance(prism, dict) or prism.get("labels_used_for_model_selection") is not False:
        raise ValueError("source-final gate must include frozen PRISM evidence not used for model selection")
    sorted_epochs = sorted(epochs)
    median_epoch = sorted_epochs[len(sorted_epochs) // 2]
    return median_epoch, {
        "post_gate_full_source_refit": True,
        "source_fold_summary_paths": [str(path) for path in summary_paths],
        "source_fold_summary_sha256": {str(path): sha256_file(path) for path in summary_paths},
        "outer_fold_best_epochs": epochs,
        "frozen_median_best_epoch": median_epoch,
        "cell_line_gate_path": str(gate_path),
        "cell_line_gate_sha256": sha256_file(gate_path),
        "cell_line_gate_format": gate["format"],
        "prism_labels_used_for_model_selection": prism["labels_used_for_model_selection"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    data = load_indexed_pair_data(args.manifest, cohort_id=getattr(args, "cohort_id", None))
    if args.formal and data.manifest.get("development_input"):
        raise ValueError("formal EXP-011 runs reject direct development NPZ input")
    if args.formal and args.stage in {"source", "source-final"}:
        tissue = data.extra_arrays.get("sample_tissue")
        if tissue is None or len(tissue) != len(data.sample_id) or len(set(np.asarray(tissue).astype(str).tolist())) < 2:
            raise ValueError("formal source runs require sample_tissue for the declared tissue-stratified cold-cell split")
    if args.stage == "external" and args.evaluation_role not in {"prism_frozen", "external_exploratory"}:
        raise ValueError("external stage only permits prism_frozen or external_exploratory evaluation roles")
    _require_domain_contract(data, args.stage)
    config = Exp011ModelConfig(
        gene_input_size=data.gene_count, gene_hidden_size=args.gene_hidden_size, drug_hidden_size=args.drug_hidden_size,
        feature_dim=args.feature_dim, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout, adapter_bottleneck=args.adapter_bottleneck,
        adapter_dropout=args.adapter_dropout, test_only_geometry=args.allow_test_geometry,
    )
    gene_sha = "test-only-geometry" if args.allow_test_geometry else require_official_gene_contract(data.ordered_genes.tolist(), config)
    upstream = upstream_checkout(args.upstream)
    if args.stage == "source-final":
        if not args.formal:
            raise ValueError("source-final is a formal post-gate operation and requires --formal")
        if args.model_id != "M2" or args.fold_id is not None or args.source_checkpoint is not None or args.checkpoint is not None:
            raise ValueError("source-final permits only M2 and forbids fold/checkpoint inputs")
        fixed_epochs, approval = _frozen_source_final_epoch(args.source_fold_summary, args.cell_line_gate)
        args.output.mkdir(parents=True, exist_ok=True)
        all_indices = np.arange(data.pair_count, dtype=np.int64)
        fold_manifest_path = args.output / "source_final_all_source_indices.json"
        fold_manifest_path.write_text(json.dumps({
            "schema": "drugscreenlab.exp011.source-final.v1", "stage": "source-final", "train_pair_indices": all_indices.tolist(),
            "validation_pair_indices": [], "test_pair_indices": [], "approval": approval,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        bpe_path = fit_source_bpe_codes(data.canonical_smiles[np.unique(data.pair_drug_idx[all_indices])], args.output / "M2_source_final.bpe.codes")
        gene_transform, smiles_encoder = build_source_transforms(data, all_indices, bpe_path)
        transformed = transformed_data(data, gene_transform)
        encoded = encode_data_smiles(transformed, smiles_encoder, allow_external_smiles=True)
        model = build_official_m2(upstream, config)
        fit = refit_source_model_full(model, _dataset(transformed, all_indices, encoded), fixed_epochs=fixed_epochs, device=args.device, seed=args.seed)
        bpe_sha = sha256_file(bpe_path)
        provenance = _checkpoint_provenance(data, args, gene_sha, bpe_sha) | approval
        checkpoint_path = save_exp011_checkpoint(
            args.output / "M2_source_final.pt", model, config, gene_transform, smiles_encoder, variant="M2", fold_id=-1,
            gene_contract_sha256=gene_sha, provenance=provenance,
        )
        probe = _dataset(transformed, all_indices[: min(64, len(all_indices))], encoded)
        _, before_reload = predict_dataset(model, probe, device=args.device)
        restored, _ = load_exp011_checkpoint(checkpoint_path, upstream, device=args.device)
        _, after_reload = predict_dataset(restored, probe, device=args.device)
        reload_delta = float(np.max(np.abs(before_reload - after_reload)))
        if reload_delta != 0.0:
            raise RuntimeError(f"strict source-final checkpoint reload changed predictions by {reload_delta}")
        summary = {
            "experiment_id": EXP011_EXPERIMENT_ID, "model_id": "M2", "stage": "source-final", "fold_id": -1,
            "fit": fit.__dict__, "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
            "post_gate_full_source_refit": True, "frozen_median_best_epoch": fixed_epochs,
            "strict_reload_max_abs_delta": reload_delta,
            "predictions": None, "gene_contract_sha256": gene_sha, "bpe_codes_sha256": bpe_sha,
            "pair_counts": {"train": int(len(all_indices)), "validation": 0, "test": 0}, "fold_manifest": str(fold_manifest_path),
            "indexed_input": True, "development_input": False,
        }
        summary_path = args.output / "M2_source_final_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        return summary
    if args.stage == "external":
        if args.fold_id is not None:
            raise ValueError("external stage does not accept --fold-id")
        fold = build_external_indices(data, seed=args.seed)
    else:
        if args.fold_id is None:
            raise ValueError("source and target stages require --fold-id")
        fold = build_fold_indices(data, stage=args.stage, fold_id=args.fold_id, seed=args.seed)
    train_indices, validation_indices, test_indices = fold.train, fold.validation, fold.test
    fold_id = fold.fold_id
    args.output.mkdir(parents=True, exist_ok=True)
    fold_manifest_path = args.output / f"fold{fold_id}_{args.stage}_indices.json"
    fold_manifest_path.write_text(json.dumps(fold.as_manifest(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fit: Any | None = None
    response_transform: TrainOnlyResponseStandardizer | None = None
    source_checkpoint_sha: str | None = None
    bpe_sha: str | None = None
    loaded_checkpoint_path: Path | None = None
    if args.stage == "external":
        if args.source_checkpoint is not None or args.checkpoint is None:
            raise ValueError("external stage requires --checkpoint and forbids --source-checkpoint")
        model, payload = load_exp011_checkpoint(args.checkpoint, upstream, device=args.device)
        if payload["variant"] != args.model_id:
            raise ValueError("--model-id must exactly match the strictly loaded checkpoint variant")
        if payload.get("gene_contract_sha256") != gene_sha:
            raise ValueError("external cohort ordered-gene contract differs from the frozen checkpoint")
        gene_transform = payload["gene_transform_object"]
        smiles_encoder = payload["smiles_encoder_object"]
        bpe_sha = smiles_encoder.bpe_sha256
        source_checkpoint_sha = sha256_file(args.checkpoint)
        loaded_checkpoint_path = args.checkpoint
        data_for_prediction = transformed_data(data, gene_transform)
        encoded = encode_data_smiles(data_for_prediction, smiles_encoder, allow_external_smiles=True)
        prediction_model_id = args.model_id
        checkpoint_variant = None
    elif args.stage == "source":
        if args.model_id != "M2" or args.source_checkpoint is not None or args.checkpoint is not None:
            raise ValueError("source stage only permits a new M2 without checkpoint inputs")
        bpe_path = fit_source_bpe_codes(data.canonical_smiles[np.unique(data.pair_drug_idx[train_indices])], args.output / f"M2_fold{fold_id}.bpe.codes")
        gene_transform, smiles_encoder = build_source_transforms(data, train_indices, bpe_path)
        bpe_sha = sha256_file(bpe_path)
        transformed = transformed_data(data, gene_transform)
        encoded = encode_data_smiles(transformed, smiles_encoder, allow_external_smiles=True)
        model = build_official_m2(upstream, config)
        fit = fit_source_model(model, _dataset(transformed, train_indices, encoded), _dataset(transformed, validation_indices, encoded), device=args.device, seed=args.seed)
        data_for_prediction, prediction_model_id, checkpoint_variant = transformed, "M2", "M2"
    else:
        if args.source_checkpoint is None or args.checkpoint is not None:
            raise ValueError("target stage requires --source-checkpoint and forbids --checkpoint")
        source_model, source_payload = load_exp011_checkpoint(args.source_checkpoint, upstream, device=args.device)
        if source_payload["variant"] != "M2" or source_payload["model_variant"] != "CombinedModel":
            raise ValueError("target stage requires an M2 CombinedModel source checkpoint")
        if source_payload.get("provenance", {}).get("post_gate_full_source_refit") is not True:
            raise ValueError("target stage requires the post-gate full-source M2 checkpoint, not an outer-fold checkpoint")
        source_checkpoint_sha = sha256_file(args.source_checkpoint)
        smiles_encoder = source_payload["smiles_encoder_object"]
        bpe_sha = smiles_encoder.bpe_sha256
        if args.model_id == "M2":
            model = source_model
            gene_transform = source_payload["gene_transform_object"]
            transformed = transformed_data(data, gene_transform)
            encoded = encode_data_smiles(transformed, smiles_encoder, allow_external_smiles=True)
            data_for_prediction, prediction_model_id, checkpoint_variant = transformed, "M2", None
        else:
            if args.model_id == "M4":
                model, gene_transform = build_variant("M4", upstream, config), fit_gene_transform(data, train_indices)
            else:
                source_config = Exp011ModelConfig(**source_payload["model_config"])
                if source_config.gene_input_size != config.gene_input_size:
                    raise ValueError("target expression gene contract differs from the frozen source checkpoint")
                model, gene_transform = build_variant(args.model_id, upstream, config, source_backbone=source_model), source_payload["gene_transform_object"]
                if args.model_id == "M3-adapter":
                    probe_gene, probe_drug = np.zeros((1, data.gene_count), dtype=np.float32), np.zeros((1, 128), dtype=np.float32)
                    import torch
                    assert_m3_zero_initial_delta(model, torch.from_numpy(probe_gene).to(args.device), torch.from_numpy(probe_drug).to(args.device), source_model.to(args.device))
            transformed = transformed_data(data, gene_transform)
            encoded = encode_data_smiles(transformed, smiles_encoder, allow_external_smiles=True)
            response_transform = TrainOnlyResponseStandardizer(min_train_per_group=args.min_train_per_group).fit(data, train_indices)
            standardized = transformed_data(replace(data, pair_response=response_transform.transform(data, np.arange(data.pair_count))), gene_transform)
            fit = fit_target_model(model, args.model_id, _dataset(standardized, train_indices, encoded), _dataset(standardized, validation_indices, encoded), device=args.device, seed=args.seed)
            data_for_prediction, prediction_model_id, checkpoint_variant = standardized, args.model_id, args.model_id

    test_dataset = _dataset(data_for_prediction, test_indices, encoded)
    pair_indices, predicted = predict_dataset(model, test_dataset, device=args.device)
    prediction_raw = response_transform.inverse(data, pair_indices, predicted) if response_transform is not None else predicted
    prediction = _prediction_frame(data, pair_indices, prediction_raw, model_id=prediction_model_id, fold_id=fold_id, evaluation_role=args.evaluation_role)
    prediction_path = args.output / f"{prediction_model_id}_fold{fold_id}_predictions.parquet"
    prediction.to_parquet(prediction_path, index=False)

    checkpoint_path: Path | None = None
    reload_delta: float | None = None
    if checkpoint_variant is not None:
        checkpoint_path = save_exp011_checkpoint(
            args.output / f"{checkpoint_variant}_fold{fold_id}.pt", model, config, gene_transform, smiles_encoder,
            variant=checkpoint_variant, fold_id=fold_id, gene_contract_sha256=gene_sha,
            source_checkpoint_sha256=source_checkpoint_sha, response_transform_state=None if response_transform is None else response_transform.state_dict(),
            provenance=_checkpoint_provenance(data, args, gene_sha, bpe_sha),
        )
        restored, _ = load_exp011_checkpoint(checkpoint_path, upstream, device=args.device)
        _, reloaded = predict_dataset(restored, test_dataset, device=args.device)
        reload_delta = float(np.max(np.abs(reloaded - predicted))) if len(predicted) else 0.0
        if reload_delta != 0.0:
            raise RuntimeError(f"strict checkpoint reload changed predictions by {reload_delta}")
    summary_checkpoint = checkpoint_path or loaded_checkpoint_path
    summary = {
        "experiment_id": EXP011_EXPERIMENT_ID, "model_id": prediction_model_id, "stage": args.stage, "fold_id": fold_id,
        "fit": None if fit is None else fit.__dict__, "checkpoint": None if summary_checkpoint is None else str(summary_checkpoint),
        "checkpoint_sha256": None if summary_checkpoint is None else sha256_file(summary_checkpoint),
        "checkpoint_strictly_loaded": bool(loaded_checkpoint_path is not None), "source_checkpoint_sha256": source_checkpoint_sha,
        "predictions": str(prediction_path), "predictions_sha256": sha256_file(prediction_path), "strict_reload_max_abs_delta": reload_delta,
        "architecture_sha256": architecture_digest(config, gene_sha), "gene_contract_sha256": gene_sha, "bpe_codes_sha256": bpe_sha,
        "pair_counts": {"train": int(len(train_indices)), "validation": int(len(validation_indices)), "test": int(len(test_indices))},
        "fold_manifest": str(fold_manifest_path), "indexed_input": True, "development_input": bool(data.manifest.get("development_input", False)),
    }
    summary_path = args.output / f"{prediction_model_id}_fold{fold_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=_default_data_root() / "processed/exp011/cohort_index_manifest.json")
    parser.add_argument("--cohort-id", help="required when --manifest is a top-level cohort_index_manifest.json")
    parser.add_argument("--upstream", type=Path, required=True, help="verifiable PharmaFormer checkout at commit f526…")
    parser.add_argument("--stage", choices=("source", "source-final", "target", "external"), required=True)
    parser.add_argument("--model-id", choices=("M2", "M3-adapter", "M3-full", "M4"), required=True)
    parser.add_argument("--fold-id", type=int, help="required for source/target; forbidden for external")
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="strictly reloaded frozen checkpoint for --stage external")
    parser.add_argument("--source-fold-summary", type=Path, action="append", help="five completed M2 cold-cell source summaries required by source-final")
    parser.add_argument("--cell-line-gate", type=Path, help="passed exp011_cell_line_gate_v1 required by source-final")
    parser.add_argument("--evaluation-role", choices=("gdsc_cold_cell", "gdsc_cold_drug", "prism_frozen", "pdo_primary", "external_exploratory"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--min-train-per-group", type=int, default=8)
    parser.add_argument("--gene-hidden-size", type=int, default=8192)
    parser.add_argument("--drug-hidden-size", type=int, default=256)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--adapter-bottleneck", type=int, default=256)
    parser.add_argument("--adapter-dropout", type=float, default=0.3)
    parser.add_argument("--allow-test-geometry", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--formal", action="store_true", help="require data-agent cohort_index_manifest.json and emit a formal run")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
