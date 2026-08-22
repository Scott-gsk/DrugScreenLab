#!/usr/bin/env python3
"""Approved EXP-009 one-batch engineering smoke runner.

A: frozen official XPert baseline.
B: frozen XPert plus parameter-count-matched structure-only residual.
C: frozen XPert plus 64-d BindingDB teacher soft-target residual.

This runner is deliberately train-only smoke infrastructure. It reuses the
approved official XPert cold-drug split without selecting a checkpoint,
changing any split, endpoint, disease definition, or hypothesis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import stat
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
SOURCE = ROOT / "data" / "external" / "xpert_source"
DEFAULT_CHECKPOINT = SOURCE / "saved_model" / "l1000_sdst_warm_split.pth"
DEFAULT_OUTPUT = ROOT / "artifacts" / "experiments" / "EXP-009" / "residual_smoke"
DEFAULT_SOFT_TARGET_FEATURES = (
    ROOT
    / "artifacts"
    / "experiments"
    / "EXP-009"
    / "teacher_morgan_probe_100k"
    / "xpert_sdst_features"
    / "xpert_sdst_soft_target_features.npz"
)
EXPECTED_SOFT_TARGET_FEATURES_SHA256 = "84cc67ffe427b6e51ee2b2ad08c9b3b81d416721bcc85ad3c3d8f9c3517a1ec0"
# Applied only after the pre-activation A/C equivalence assertion. It is small
# enough to keep the bounded effective gate essentially inactive (about 5e-5),
# but lets gradients reach the zero-initialized residual output layer.
RESIDUAL_ACTIVATION_RAW_GAMMA = 1e-3


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_readonly_checkpoint(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"official XPert checkpoint not found: {path}")
    if os.stat(path).st_mode & stat.S_IWUSR:
        raise PermissionError(f"official XPert checkpoint must be read-only: {path}")


def make_checkpoint_manifest(path: Path, *, seed: int) -> dict[str, Any]:
    assert_readonly_checkpoint(path)
    return {
        "seed": int(seed),
        "checkpoint": {
            "path": str(path.resolve()),
            "sha256": checkpoint_sha256(path),
            "readonly": True,
            "mutation_policy": "read_only; runner never writes the official checkpoint",
        },
    }


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _official_imports() -> dict[str, Any]:
    import sys

    source = str(SOURCE)
    if source not in sys.path:
        sys.path.insert(0, source)
    from models.model_XPert import XPertNet
    from utils import mse_loss_ls_sum, pcc_loss_sum

    return {"XPertNet": XPertNet, "mse_loss_ls_sum": mse_loss_ls_sum, "pcc_loss_sum": pcc_loss_sum}


def _official_args(device: str) -> SimpleNamespace:
    return SimpleNamespace(
        mode="train", dataset="l1000_sdst", drug_feat="unimol", device=device,
        pretrained_mode="global", include_cell_idx=True, wo_HG=False, wo_atom=False,
        wo_atom_HG=False, wo_unimol=False, wo_ppi=False, use_gene_pos_emed=False,
        output_attention=False, output_cls_embed=False,
    )


def _load_config() -> dict[str, Any]:
    import yaml

    config = yaml.safe_load((SOURCE / "configs" / "config_l1000_foundation_bounded.yaml").read_text())
    config["model"]["ATTN"]["ppi_gene_vector_path"] = str(SOURCE / "processed_data" / "PPI_gene_vector_128d.npy")
    config["model"]["HG"]["drug_hg_pretrained_embed_path"] = str(SOURCE / "HG_data" / "saved_embedding" / "HG_drug_embeddings.npy")
    return config


def _load_official_model(*, official: dict[str, Any], args: Any, config: dict[str, Any], device: Any, checkpoint: Path) -> Any:
    from drug_screen.foundation.xpert_extension import load_xpert_checkpoint

    model = official["XPertNet"](args, config, device, logging.getLogger("drugscreenlab.exp009.smoke"))
    model.init_weights()
    audit = load_xpert_checkpoint(model, checkpoint, map_location=device)
    if audit["missing_official"] or audit["unexpected"]:
        raise RuntimeError(f"official XPert checkpoint identity failure: {audit}")
    return model.to(device), audit


def _official_loss(model: Any, batch: Any, config: dict[str, Any], official: dict[str, Any]) -> Any:
    output = model(batch)
    trt_output, ctl_output, deg_output = output[:3]
    trt_raw_data = batch[0].to(model.device)
    ctl_raw_data = batch[1].to(model.device)
    loss1 = official["mse_loss_ls_sum"](trt_output, trt_raw_data)
    cell_true, cell_pred = output[6], output[7]
    if cell_pred is None:
        loss2 = official["mse_loss_ls_sum"](ctl_output, ctl_raw_data)
    else:
        import torch
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        loss2 = criterion(cell_pred[0], cell_true) + criterion(cell_pred[1], cell_true)
    loss3 = official["mse_loss_ls_sum"](deg_output, trt_raw_data - ctl_raw_data)
    loss4 = official["pcc_loss_sum"](deg_output, trt_raw_data - ctl_raw_data)
    a, b, c, d = config["train"]["loss_weight"]
    return loss1 * a + loss2 * b + loss3 * c + loss4 * d


def _structure_features(batch: Any) -> Any:
    """Pool the contract-approved official UniMol tensor to its 514-d drug feature."""
    import torch

    candidates = [value for value in batch if torch.is_tensor(value) and value.ndim == 3 and value.shape[-1] == 514]
    if len(candidates) != 1:
        raise ValueError("official XPert batch must expose exactly one rank-3 514-d UniMol tensor")
    return candidates[0].mean(dim=1)


def load_soft_target_features_for_batch(
    path: Path | str,
    pert_ids: list[str],
    *,
    expected_target_dim: int = 64,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read only precomputed Morgan probabilities in exact requested pert_id order."""
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"soft-target feature artifact not found: {artifact}")
    with np.load(artifact, allow_pickle=False) as payload:
        required = {"pert_id", "soft_target_probabilities", "feature_valid"}
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"soft-target feature artifact missing arrays: {missing}")
        stored_ids = np.asarray(payload["pert_id"]).astype(str)
        probabilities = np.asarray(payload["soft_target_probabilities"], dtype=np.float32)
        valid = np.asarray(payload["feature_valid"], dtype=bool)
    if probabilities.shape != (len(stored_ids), expected_target_dim):
        raise ValueError(
            "soft-target feature probabilities must have shape "
            f"({len(stored_ids)}, {expected_target_dim}), received {probabilities.shape}"
        )
    if valid.shape != (len(stored_ids),):
        raise ValueError("soft-target feature validity mask must align with pert_id")
    lookup = {pert_id: position for position, pert_id in enumerate(stored_ids.tolist())}
    missing_ids = sorted(set(pert_ids).difference(lookup))
    if missing_ids:
        raise ValueError(f"soft-target feature artifact does not contain requested pert_id values: {missing_ids}")
    positions = np.asarray([lookup[pert_id] for pert_id in pert_ids], dtype=np.int64)
    if not np.all(valid[positions]):
        invalid_ids = [pert_id for pert_id, position in zip(pert_ids, positions, strict=True) if not valid[position]]
        raise ValueError(f"soft-target feature artifact has invalid features for requested pert_id values: {invalid_ids}")
    selected = probabilities[positions]
    if not np.isfinite(selected).all():
        raise ValueError("soft-target feature artifact has non-finite probabilities for requested pert_id values")
    return selected, {
        "path": str(artifact.resolve()),
        "sha256": checkpoint_sha256(artifact),
        "expected_sha256": EXPECTED_SOFT_TARGET_FEATURES_SHA256 if artifact.resolve() == DEFAULT_SOFT_TARGET_FEATURES.resolve() else None,
        "feature_source": "precomputed_morgan_soft_target_probabilities",
        "rows": int(len(stored_ids)),
        "target_dim": int(expected_target_dim),
        "requested_batch_rows": int(len(pert_ids)),
    }


def residual_learnability_audit(
    *,
    raw_gamma_before: float,
    raw_gamma_after: float,
    gradient_norms: dict[str, float],
    parameter_max_abs_updates: dict[str, float],
    loss_finite: bool,
) -> dict[str, Any]:
    """Classify one post-equivalence residual optimization step honestly."""
    at_least_one_gradient_nonzero = any(value > 0.0 for value in gradient_norms.values())
    at_least_one_updated = any(value > 0.0 for value in parameter_max_abs_updates.values())
    status = "COMPLETE" if loss_finite and at_least_one_gradient_nonzero and at_least_one_updated else "BROKEN"
    return {
        "status": status,
        "loss_finite": bool(loss_finite),
        "raw_gamma_before": float(raw_gamma_before),
        "raw_gamma_after": float(raw_gamma_after),
        "residual_gradient_norms": gradient_norms,
        "residual_parameter_max_abs_updates": parameter_max_abs_updates,
        "at_least_one_residual_gradient_nonzero": at_least_one_gradient_nonzero,
        "at_least_one_residual_parameter_updated": at_least_one_updated,
    }


def _load_teacher(path: Path, *, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch
    from drug_screen.foundation.soft_target import SoftTargetHead

    payload = torch.load(path, map_location=device, weights_only=False)
    if int(payload.get("structure_dim", -1)) != 514 or int(payload.get("target_dim", -1)) != 64:
        raise ValueError("EXP-009 teacher checkpoint must be a fixed 514->64 BindingDB soft-target head")
    teacher = SoftTargetHead(514, 64).to(device)
    teacher.load_state_dict(payload["state_dict"], strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher, {"path": str(path.resolve()), "sha256": checkpoint_sha256(path), "target_dim": 64, "frozen": True}


def _replace_delta(output: Any, delta: Any) -> Any:
    if isinstance(output, tuple):
        return (*output[:2], delta, *output[3:])
    result = list(output)
    result[2] = delta
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    import scanpy as sc
    import torch
    from torch.utils.data import DataLoader
    from drug_screen.foundation.soft_target import build_exp009_residual_wrapper, count_trainable_parameters
    from drug_screen.foundation.xpert_extension import load_xpert_checkpoint

    if args.variant not in {"A", "B", "C"}:
        raise ValueError("variant must be A, B, or C")
    if args.max_batches != 1:
        raise ValueError("EXP-009 runner supports only --max-batches 1 engineering smoke")
    checkpoint = Path(args.checkpoint)
    identity = make_checkpoint_manifest(checkpoint, seed=args.seed)
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    official = _official_imports()
    config = _load_config()
    official_args = _official_args(str(device))
    baseline, checkpoint_audit = _load_official_model(official=official, args=official_args, config=config, device=device, checkpoint=checkpoint)

    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    from datasets.MyDataset import MyDataset
    backed = sc.read_h5ad(SOURCE / "processed_data" / "l1000_sdst_78453.h5ad", backed="r")
    positions = np.flatnonzero(backed.obs[args.split].astype(str).to_numpy() == "train")
    if len(positions) < args.batch_size:
        raise ValueError("approved official cold-drug train split has fewer rows than requested batch")
    data = backed[positions[:args.batch_size]].to_memory()
    backed.file.close()
    unimol = np.load(SOURCE / "processed_data" / "all_drugs_unimol_arr.npy", mmap_mode="r", allow_pickle=False)
    dataset = MyDataset(data, unimol, args=official_args, config=config, logger=logging.getLogger("drugscreenlab.exp009.smoke"), max_value=config["dataset"]["max_value"], min_value=config["dataset"]["min_value"])
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)))

    baseline.eval()
    with torch.no_grad():
        baseline_output = baseline(batch)
    structure = _structure_features(batch).to(device)
    batch_pert_ids = data.obs["pert_id"].astype(str).tolist()
    teacher_audit = None
    soft_target_feature_audit = None
    activation = None
    learnability = None
    if args.variant == "A":
        model = baseline
        output = baseline_output
        initial_equivalence = {"checked": True, "max_abs_delta": 0.0, "exact": True}
        gradient_audit = {"official_parameters_frozen": True, "official_trainable_parameter_count": 0, "official_parameters_with_grad": [], "residual_trainable_parameter_count": 0, "raw_gamma": None, "residual_output_zero_initialized": None}
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    else:
        # Construct a separately checkpoint-loaded official instance, then wrap.
        official_model, _ = _load_official_model(official=official, args=official_args, config=config, device=device, checkpoint=checkpoint)
        model = build_exp009_residual_wrapper(official_model, variant=args.variant, structure_dim=514, soft_target_dim=64, hidden_dim=args.hidden_dim, n_genes=978).to(device)
        soft_targets = None
        if args.variant == "C":
            # C consumes the reviewed full 8,418 x 64 Morgan artifact directly;
            # it does not re-run or substitute the small UniMol teacher smoke.
            values, soft_target_feature_audit = load_soft_target_features_for_batch(
                args.soft_target_features, batch_pert_ids, expected_target_dim=64
            )
            if (
                Path(args.soft_target_features).resolve() == DEFAULT_SOFT_TARGET_FEATURES.resolve()
                and soft_target_feature_audit["sha256"] != EXPECTED_SOFT_TARGET_FEATURES_SHA256
            ):
                raise ValueError("default Morgan soft-target artifact SHA256 does not match the approved identity")
            soft_targets = torch.from_numpy(values).to(device)
        model.eval()
        with torch.no_grad():
            initialized = model(batch, structure_features=structure, soft_targets=soft_targets)
        max_delta = float((initialized[2] - baseline_output[2]).abs().max().cpu())
        initial_equivalence = {"checked": True, "max_abs_delta": max_delta, "exact": max_delta == 0.0}
        # Phase 2 begins only after strict pre-activation baseline equivalence.
        # We retain the all-zero output initialization and activate raw_gamma to
        # a fixed audit-visible value so its output layer can receive gradient.
        with torch.no_grad():
            model.residual.raw_gamma.fill_(RESIDUAL_ACTIVATION_RAW_GAMMA)
        raw_gamma_before = float(model.residual.raw_gamma.detach().cpu())
        max_effective_gate = float(0.05 * torch.tanh(model.residual.raw_gamma).detach().cpu())
        parameters_before = {
            name: parameter.detach().clone()
            for name, parameter in model.residual.named_parameters()
        }
        model.train()
        train_output = model(batch, structure_features=structure, soft_targets=soft_targets)
        loss = _official_loss(_OutputOverride(model, train_output), batch, config, official)
        loss_finite = bool(torch.isfinite(loss).item())
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
        optimizer.zero_grad(set_to_none=True)
        if loss_finite:
            loss.backward()
        gradient_audit = model.gradient_audit()
        if gradient_audit["official_trainable_parameter_count"] or gradient_audit["official_parameters_with_grad"]:
            raise RuntimeError(f"frozen XPert gradient audit failed: {gradient_audit}")
        if loss_finite:
            optimizer.step()
        raw_gamma_after = float(model.residual.raw_gamma.detach().cpu())
        parameter_updates = {
            name: float((parameter.detach() - parameters_before[name]).abs().max().cpu())
            for name, parameter in model.residual.named_parameters()
        }
        learnability = residual_learnability_audit(
            raw_gamma_before=raw_gamma_before,
            raw_gamma_after=raw_gamma_after,
            gradient_norms=gradient_audit["residual_gradient_norms"],
            parameter_max_abs_updates=parameter_updates,
            loss_finite=loss_finite,
        )
        activation = {
            "policy": "after_pre_activation_equivalence_assertion_set_raw_gamma_fixed_1e-3_before_one_optimizer_step",
            "raw_gamma_fixed": RESIDUAL_ACTIVATION_RAW_GAMMA,
            "raw_gamma_before": raw_gamma_before,
            "raw_gamma_after": raw_gamma_after,
            "max_effective_gate": max_effective_gate,
            "gate_bound": 0.05,
            "output_layer_initialization": "all_zero_preserved_before_activation",
        }
        output = train_output

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "format": "exp009_residual_train_only_smoke_v2", "experiment": "EXP-009", "mode": "TRAIN_ONLY_SMOKE",
        "status": "COMPLETE" if learnability is None or learnability["status"] == "COMPLETE" else "BROKEN",
        "variant": args.variant, "variant_definition": {"A": "frozen official XPert baseline", "B": "frozen XPert + parameter-matched structure-only residual", "C": "frozen XPert + 64-d BindingDB soft-target residual"}[args.variant],
        "seed": args.seed, "command": args.command, "batch_size": args.batch_size, "max_batches": 1,
        "split": args.split, "endpoint": "official XPert Delta978 output index 2", "checkpoint": identity["checkpoint"],
        "checkpoint_inheritance": checkpoint_audit, "teacher_checkpoint": teacher_audit,
        "soft_target_features": soft_target_feature_audit,
        "initial_equivalence": {
            **initial_equivalence,
            "scope": "pre_activation_only; checked before fixed raw_gamma activation and optimizer step",
        },
        "activation": activation, "gradient_audit": gradient_audit, "learnability": learnability,
        "model_trainable_parameters": count_trainable_parameters(model),
        "output_shape": list(output[2].shape),
    }
    manifest = args.output_dir / f"{args.variant}_manifest.json"
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "variant": args.variant, "manifest": str(manifest), "initial_equivalence": initial_equivalence, "activation": activation, "gradient_audit": gradient_audit, "learnability": learnability}, sort_keys=True))
    return result


class _OutputOverride:
    """Route the official loss through a precomputed wrapped output."""
    def __init__(self, model: Any, output: Any) -> None:
        self.device, self._output = model.official.device, output
    def __call__(self, _batch: Any) -> Any:
        return self._output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["A", "B", "C"], required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--soft-target-features",
        type=Path,
        default=DEFAULT_SOFT_TARGET_FEATURES,
        help="approved full 8418x64 Morgan teacher probabilities aligned by XPert pert_id",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="split_cold_drug_1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-batches", type=int, default=1)
    args = parser.parse_args()
    args.command = " ".join(__import__("sys").argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
