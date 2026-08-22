#!/usr/bin/env python3
"""EXP-009 dedicated full-partition MVP runner.

This is intentionally independent of EXP-005. It evaluates the approved
``split_cold_drug_1`` SDST Delta978 task only: A is frozen XPert forward-only;
B and C are frozen-XPert residual overlays. C consumes the reviewed 8,418x64
precomputed Morgan features by exact ``pert_id`` matching. No efficacy,
disease-signature, PRISM, CTRP, or PDO inputs are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Direct script execution needs the repository root to import the dedicated
# EXP-009 smoke utilities; pytest already supplies it during module imports.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.modeling.run_exp009_residual_smoke import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SOFT_TARGET_FEATURES,
    EXPECTED_SOFT_TARGET_FEATURES_SHA256,
    RESIDUAL_ACTIVATION_RAW_GAMMA,
    _load_config,
    _load_official_model,
    _official_args,
    _official_imports,
    _official_loss,
    _structure_features,
    checkpoint_sha256,
    load_soft_target_features_for_batch,
    make_checkpoint_manifest,
    residual_learnability_audit,
    seed_everything,
)

DEFAULT_OUTPUT = ROOT / "artifacts" / "experiments" / "EXP-009" / "full_mvp"
SPLIT = "split_cold_drug_1"


def execution_scope(max_train_rows: int | None, max_test_rows: int | None) -> str:
    return "FULL" if max_train_rows is None and max_test_rows is None else "DEBUG"


def select_partition_positions(labels: np.ndarray, partition: str, limit: int | None) -> np.ndarray:
    positions = np.flatnonzero(labels.astype(str) == partition)
    if limit is not None:
        if limit <= 0:
            raise ValueError("debug partition limit must be positive")
        positions = positions[:limit]
    if not len(positions):
        raise ValueError(f"official partition is empty after selection: {partition}")
    return positions


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    return result


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError("truth and prediction must share rank-2 Delta978 shape")
    finite = np.isfinite(prediction)
    if not finite.all() or not np.isfinite(truth).all():
        raise ValueError("non-finite Delta978 prediction or target")
    row_spearman = [_corr(_rank(t), _rank(p)) for t, p in zip(truth, prediction, strict=True)]
    flat_truth, flat_prediction = truth.reshape(-1), prediction.reshape(-1)
    return {
        "rows": int(truth.shape[0]), "genes": int(truth.shape[1]),
        "row_macro_spearman": float(np.mean(row_spearman)),
        "flat_spearman": _corr(_rank(flat_truth), _rank(flat_prediction)),
        "flat_pearson": _corr(flat_truth, flat_prediction),
        "mse": float(np.mean((truth - prediction) ** 2)),
        "rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
        "prediction_std": float(np.std(prediction)),
        "finite_rate": float(finite.mean()),
    }


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values.astype(np.float32)).tobytes()).hexdigest()


def load_morgan_feature_table(path: Path | str, *, expected_target_dim: int = 64) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read the fixed complete Morgan artifact once for an entire partition run."""
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"soft-target feature artifact not found: {artifact}")
    with np.load(artifact, allow_pickle=False) as payload:
        required = {"pert_id", "soft_target_probabilities", "feature_valid"}
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"soft-target feature artifact missing arrays: {missing}")
        pert_ids = np.asarray(payload["pert_id"]).astype(str)
        probabilities = np.asarray(payload["soft_target_probabilities"], dtype=np.float32)
        valid = np.asarray(payload["feature_valid"], dtype=bool)
    if probabilities.shape != (len(pert_ids), expected_target_dim) or valid.shape != (len(pert_ids),):
        raise ValueError("Morgan feature artifact shape does not match its pert_id rows and required target dimension")
    if len(set(pert_ids.tolist())) != len(pert_ids):
        raise ValueError("Morgan feature artifact pert_id values must be unique")
    if not valid.all() or not np.isfinite(probabilities).all():
        raise ValueError("complete Morgan feature artifact must contain only valid finite feature rows")
    digest = checkpoint_sha256(artifact)
    if artifact.resolve() == DEFAULT_SOFT_TARGET_FEATURES.resolve() and digest != EXPECTED_SOFT_TARGET_FEATURES_SHA256:
        raise ValueError("default Morgan feature artifact SHA256 mismatch")
    return {pert_id: probabilities[position] for position, pert_id in enumerate(pert_ids.tolist())}, {
        "path": str(artifact.resolve()), "sha256": digest, "rows": int(len(pert_ids)), "target_dim": expected_target_dim,
        "feature_source": "precomputed_morgan_soft_target_probabilities",
    }


def select_feature_rows(table: dict[str, np.ndarray], pert_ids: list[str]) -> np.ndarray:
    missing = sorted(set(pert_ids).difference(table))
    if missing:
        raise ValueError(f"Morgan feature table does not contain requested pert_id values: {missing}")
    return np.stack([table[pert_id] for pert_id in pert_ids]).astype(np.float32, copy=False)


def _batch_features(batch: Any, batch_pert_ids: list[str], variant: str, feature_table: dict[str, np.ndarray] | None, device: Any) -> tuple[Any, Any | None]:
    import torch
    structure = _structure_features(batch).to(device)
    if variant != "C":
        return structure, None
    if feature_table is None:
        raise ValueError("C requires an already validated complete Morgan feature table")
    return structure, torch.from_numpy(select_feature_rows(feature_table, batch_pert_ids)).to(device)


def _predict(model: Any, loader: Any, raw_data: Any, variant: str, feature_table: dict[str, np.ndarray] | None, device: Any) -> tuple[np.ndarray, np.ndarray]:
    import torch
    model.eval()
    truth_rows: list[np.ndarray] = []
    pred_rows: list[np.ndarray] = []
    with torch.no_grad():
        cursor = 0
        for batch in loader:
            count = len(batch[0])
            pert_ids = raw_data.obs["pert_id"].astype(str).iloc[cursor:cursor + count].tolist()
            structure, soft = _batch_features(batch, pert_ids, variant, feature_table, device)
            output = model(batch) if variant == "A" else model(batch, structure_features=structure, soft_targets=soft)
            truth_rows.append((batch[0] - batch[1]).cpu().numpy())
            pred_rows.append(output[2].detach().cpu().numpy())
            cursor += count
    return np.concatenate(truth_rows), np.concatenate(pred_rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import scanpy as sc
    import torch
    from torch.utils.data import DataLoader
    from drug_screen.foundation.soft_target import build_exp009_residual_wrapper, count_trainable_parameters

    if args.variant not in {"A", "B", "C"}:
        raise ValueError("variant must be A, B, or C")
    if args.epochs <= 0:
        raise ValueError("fixed epochs must be positive")
    scope = execution_scope(args.max_train_rows, args.max_test_rows)
    identity = make_checkpoint_manifest(args.checkpoint, seed=args.seed)
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    official, config = _official_imports(), _load_config()
    official_args = _official_args(str(device))
    baseline, checkpoint_audit = _load_official_model(official=official, args=official_args, config=config, device=device, checkpoint=args.checkpoint)
    if str(ROOT / "data" / "external" / "xpert_source") not in sys.path:
        sys.path.insert(0, str(ROOT / "data" / "external" / "xpert_source"))
    from datasets.MyDataset import MyDataset
    backed = sc.read_h5ad(ROOT / "data" / "external" / "xpert_source" / "processed_data" / "l1000_sdst_78453.h5ad", backed="r")
    labels = backed.obs[SPLIT].astype(str).to_numpy()
    train_pos = select_partition_positions(labels, "train", args.max_train_rows)
    test_pos = select_partition_positions(labels, "test", args.max_test_rows)
    train_data, test_data = backed[train_pos].to_memory(), backed[test_pos].to_memory()
    backed.file.close()
    unimol = np.load(ROOT / "data" / "external" / "xpert_source" / "processed_data" / "all_drugs_unimol_arr.npy", mmap_mode="r", allow_pickle=False)
    logger = logging.getLogger("drugscreenlab.exp009.full_mvp")
    common = {"args": official_args, "config": config, "logger": logger, "max_value": config["dataset"]["max_value"], "min_value": config["dataset"]["min_value"]}
    train_loader = DataLoader(MyDataset(train_data, unimol, **common), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(MyDataset(test_data, unimol, **common), batch_size=args.batch_size, shuffle=False, num_workers=0)
    feature_table, feature_audit = (None, None)
    if args.variant == "C":
        feature_table, feature_audit = load_morgan_feature_table(args.soft_target_features, expected_target_dim=64)

    activation, learnability, gradient_audit, initial_equivalence = None, None, None, {"checked": True, "exact": True, "max_abs_delta": 0.0, "scope": "A baseline self-equivalence"}
    epoch_history: list[dict[str, float | int]] = []
    if args.variant == "A":
        model = baseline
        for parameter in model.parameters(): parameter.requires_grad_(False)
        epoch_history = [{"epoch": epoch, "trainable": 0, "status": "forward_only"} for epoch in range(1, args.epochs + 1)]
    else:
        official_model, _ = _load_official_model(official=official, args=official_args, config=config, device=device, checkpoint=args.checkpoint)
        model = build_exp009_residual_wrapper(official_model, variant=args.variant, structure_dim=514, soft_target_dim=64, hidden_dim=args.hidden_dim, n_genes=978).to(device)
        first_batch = next(iter(train_loader))
        first_ids = train_data.obs["pert_id"].astype(str).iloc[:len(first_batch[0])].tolist()
        structure, soft = _batch_features(first_batch, first_ids, args.variant, feature_table, device)
        baseline.eval(); model.eval()
        with torch.no_grad():
            base_output = baseline(first_batch)[2]
            residual_output = model(first_batch, structure_features=structure, soft_targets=soft)[2]
        max_delta = float((base_output - residual_output).abs().max().cpu())
        initial_equivalence = {"checked": True, "exact": max_delta == 0.0, "max_abs_delta": max_delta, "scope": "pre_activation_only"}
        if max_delta != 0.0: raise RuntimeError("B/C failed strict pre-activation baseline equivalence")
        with torch.no_grad(): model.residual.raw_gamma.fill_(RESIDUAL_ACTIVATION_RAW_GAMMA)
        before = {name: parameter.detach().clone() for name, parameter in model.residual.named_parameters()}
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
        first_step = True
        for _epoch in range(args.epochs):
            model.train()
            cursor, batches, loss_sum = 0, 0, 0.0
            finite = True
            for batch in train_loader:
                count = len(batch[0]); ids = train_data.obs["pert_id"].astype(str).iloc[cursor:cursor + count].tolist(); cursor += count
                structure, soft = _batch_features(batch, ids, args.variant, feature_table, device)
                output = model(batch, structure_features=structure, soft_targets=soft)
                loss = _official_loss(_OutputOverride(model, output), batch, config, official)
                finite = bool(torch.isfinite(loss).item())
                optimizer.zero_grad(set_to_none=True)
                if not finite: break
                loss.backward()
                if first_step:
                    gradient_audit = model.gradient_audit(); first_step = False
                optimizer.step()
                batches += 1
                loss_sum += float(loss.detach().cpu())
            epoch_history.append({"epoch": _epoch + 1, "batches": batches, "mean_batch_loss": loss_sum / max(batches, 1), "finite": finite})
            if not finite: break
        updates = {name: float((parameter.detach() - before[name]).abs().max().cpu()) for name, parameter in model.residual.named_parameters()}
        learnability = residual_learnability_audit(raw_gamma_before=RESIDUAL_ACTIVATION_RAW_GAMMA, raw_gamma_after=float(model.residual.raw_gamma.detach().cpu()), gradient_norms=gradient_audit["residual_gradient_norms"], parameter_max_abs_updates=updates, loss_finite=finite)
        activation = {"policy": "after_pre_activation_equivalence_set_raw_gamma_1e-3_before_first_train_batch", "raw_gamma_fixed": RESIDUAL_ACTIVATION_RAW_GAMMA, "max_effective_gate": float(0.05 * torch.tanh(torch.tensor(RESIDUAL_ACTIVATION_RAW_GAMMA))), "gate_bound": 0.05}

    truth, prediction = _predict(model, test_loader, test_data, args.variant, feature_table, device)
    result_metrics = metrics(truth, prediction)
    completed_epochs = len(epoch_history)
    if completed_epochs != args.epochs:
        raise RuntimeError(f"training ended before the fixed epoch budget: completed={completed_epochs}, requested={args.epochs}")
    status = "COMPLETE" if learnability is None or learnability["status"] == "COMPLETE" else "BROKEN"
    output_dir = args.output_dir / args.variant
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "prediction_digest.npz", truth_sha256=_digest(truth), prediction_sha256=_digest(prediction), rows=truth.shape[0], genes=truth.shape[1])
    (output_dir / "metrics.json").write_text(json.dumps(result_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"format": "exp009_full_mvp_v1", "status": status, "experiment": "EXP-009", "variant": args.variant, "scope": scope, "split": SPLIT, "split_counts": {"train": int(len(train_pos)), "test": int(len(test_pos)), "official_full_counts": {"train": 62624, "test": 15829}}, "seed": args.seed, "epochs": args.epochs, "completed_epochs": completed_epochs, "epoch_history": epoch_history, "batch_size": args.batch_size, "selection": "fixed epochs; no validation; never uses test for early stopping", "endpoint": "official Delta978; row-macro Spearman primary", "official_checkpoint": identity["checkpoint"], "checkpoint_inheritance": checkpoint_audit, "soft_target_features": feature_audit, "pathway_feature": False, "transductive_external_pretraining_to_sdst_labels": True, "strict_ood_claim": False, "forbidden_inputs": ["efficacy", "disease signature", "PRISM", "CTRP", "PDO"], "initial_equivalence": initial_equivalence, "activation": activation, "gradient_audit": gradient_audit, "learnability": learnability, "trainable_parameters": count_trainable_parameters(model), "metrics_path": str(output_dir / "metrics.json"), "prediction_digest_path": str(output_dir / "prediction_digest.npz")}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "variant": args.variant, "scope": scope, "metrics": result_metrics}, sort_keys=True))
    return manifest


class _OutputOverride:
    def __init__(self, model: Any, output: Any) -> None: self.device, self._output = model.official.device, output
    def __call__(self, _batch: Any) -> Any: return self._output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["A", "B", "C"], required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--soft-target-features", type=Path, default=DEFAULT_SOFT_TARGET_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-test-rows", type=int)
    args = parser.parse_args()
    run(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
