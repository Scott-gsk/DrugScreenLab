"""Run the bounded EXP-005 XPert additive-token comparison.

This runner imports the official XPert model and dataset classes from the
registered external checkout.  Model A is the unchanged official XPertNet;
Models B/C use the project overlay that appends KPGT and KPGT+UniPert tokens.
All variants share the same deterministic split subset, loss, optimizer, seed,
epoch budget, and downstream evaluator.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import logging
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "external" / "xpert_source"
FULL_DATA = SOURCE / "processed_data" / "l1000_sdst_78453.h5ad"
UNIMOL = SOURCE / "processed_data" / "all_drugs_unimol_arr.npy"
KPGT = SOURCE / "processed_data" / "all_drugs_idx2KPGT.npy"
UNIPERT = SOURCE / "processed_data" / "all_drugs_idx2UniPert.npy"
GLOBAL_ADAPTER = SOURCE / "processed_data" / "l1000_sdst_broad_crc_global_adapter.h5ad"
SIGNATURE = ROOT / "mvp" / "core_data" / "crc_disease_signature_exact978.tsv"
BROAD_RESPONSE = ROOT / "mvp" / "foundation" / "xpert" / "BROAD_PRISM_CRC_V1.parquet"
RESULT_ROOT = ROOT / "mvp" / "foundation" / "xpert" / "EXP005_FAST"
PROFILE_ROOT = SOURCE / "experiment" / "exp005_fast"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _logger() -> logging.Logger:
    logger = logging.getLogger("drugscreenlab.exp005.fast")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    return logger


def _official_imports() -> dict[str, Any]:
    source_text = str(SOURCE)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from datasets.MyDataset import MyDataset
    from models.model_XPert import XPertNet
    from utils import mse_loss_ls_sum, pcc_loss_sum

    return {
        "MyDataset": MyDataset,
        "XPertNet": XPertNet,
        "mse_loss_ls_sum": mse_loss_ls_sum,
        "pcc_loss_sum": pcc_loss_sum,
    }


def _args(*, device: str, mode: str = "train") -> SimpleNamespace:
    # These fields are the official XPert argument contract consumed by
    # model_XPert.py and model_utils.py.
    return SimpleNamespace(
        mode=mode,
        dataset="l1000_sdst",
        drug_feat="unimol",
        device=device,
        pretrained_mode="global",
        include_cell_idx=True,
        wo_HG=False,
        wo_atom=False,
        wo_atom_HG=False,
        wo_unimol=False,
        wo_ppi=False,
        use_gene_pos_emed=False,
        output_attention=False,
        output_cls_embed=False,
    )


def _split_positions(obs: Any, split_column: str, limit: int | None) -> dict[str, np.ndarray]:
    labels = obs[split_column].astype(str).to_numpy()
    names = obs.index.astype(str).to_numpy()
    result: dict[str, np.ndarray] = {}
    for label in ("train", "valid", "test"):
        positions = np.flatnonzero(labels == label)
        order = np.argsort(names[positions], kind="mergesort")
        positions = positions[order]
        result[label] = positions if limit is None else positions[:limit]
    # Official XPert L1000 SDST cold splits expose train/test only.  The
    # upstream loader reuses test as validation when there is no valid label;
    # our fixed-epoch FAST runner does not select a checkpoint on it.
    if len(result["valid"]) == 0:
        result["valid"] = result["test"].copy()
    return result


def _sample_id_digest(obs: Any, positions: np.ndarray) -> str:
    values = obs.index.astype(str).to_numpy()[positions]
    return sha256(("\n".join(values.tolist()) + "\n").encode("utf-8")).hexdigest()


def _cold_split_contract(obs: Any, positions: dict[str, np.ndarray], split_column: str) -> dict[str, Any]:
    train = obs.iloc[positions["train"]]
    test = obs.iloc[positions["test"]]
    train_contexts = set(train["cell_iname"].astype(str))
    test_contexts = set(test["cell_iname"].astype(str))
    train_drugs = set(train["pert_id"].astype(str))
    test_drugs = set(test["pert_id"].astype(str))
    context_overlap = sorted(train_contexts & test_contexts)
    drug_overlap = sorted(train_drugs & test_drugs)
    if "cold_cell" in split_column and context_overlap:
        raise ValueError(f"{split_column} violates cold-context separation in selected rows")
    if "cold_drug" in split_column and drug_overlap:
        raise ValueError(f"{split_column} violates cold-drug separation in selected rows")
    return {
        "split_column": split_column,
        "selection": "lexicographically sorted official sample_id; first max_rows per official partition; no response labels",
        "partitions": {
            label: {
                "rows": int(len(values)),
                "sample_id_sha256": _sample_id_digest(obs, values),
            }
            for label, values in positions.items()
        },
        "train_test_context_overlap_count": int(len(context_overlap)),
        "train_test_drug_overlap_count": int(len(drug_overlap)),
        "cold_assertion": (
            "zero context overlap asserted" if "cold_cell" in split_column else
            "zero drug overlap asserted" if "cold_drug" in split_column else
            "not a cold split"
        ),
    }


def _memory_subset(backed: Any, positions: np.ndarray) -> Any:
    return backed[positions].to_memory()


def _dataset(
    *,
    data: Any,
    variant: str,
    args: Any,
    config: dict[str, Any],
    logger: logging.Logger,
    unimol: np.ndarray,
    kpgt: dict[Any, Any] | None,
    unipert: np.ndarray | None,
    official: dict[str, Any],
) -> Any:
    common = {
        "args": args,
        "config": config,
        "logger": logger,
        "max_value": config["dataset"]["max_value"],
        "min_value": config["dataset"]["min_value"],
    }
    if variant == "A":
        return official["MyDataset"](data, unimol, **common)
    from drug_screen.foundation.xpert_extension import build_xpert_extension_dataset

    extension_dataset = build_xpert_extension_dataset(official["MyDataset"])
    return extension_dataset(
        data,
        unimol,
        kpgt_features=kpgt if variant in {"B", "C"} else None,
        unipert_features=unipert if variant == "C" else None,
        use_kpgt=variant in {"B", "C"},
        use_unipert=variant == "C",
        **common,
    )


def _model(
    *,
    variant: str,
    args: Any,
    config: dict[str, Any],
    device: Any,
    logger: logging.Logger,
    official: dict[str, Any],
    checkpoint: str | Path | None = None,
) -> Any:
    from drug_screen.foundation.xpert_extension import build_xpert_additive_model

    if variant == "A":
        model_class = official["XPertNet"]
    else:
        model_class = build_xpert_additive_model(
            official["XPertNet"],
            use_kpgt=True,
            use_unipert=variant == "C",
            freeze_official=checkpoint is not None and not getattr(args, "finetune_official", False),
        )
    model = model_class(args, config, device, logger)
    model.init_weights()
    if checkpoint is not None:
        from drug_screen.foundation.xpert_extension import load_xpert_checkpoint

        load_xpert_checkpoint(model, checkpoint, map_location=device)
    return model.to(device)


def _loss(model: Any, batch: Any, config: dict[str, Any], official: dict[str, Any]) -> Any:
    output = model(batch)
    trt_output, ctl_output, deg_output = output[:3]
    trt_raw_data = batch[0].to(model.device)
    ctl_raw_data = batch[1].to(model.device)
    mse_loss = official["mse_loss_ls_sum"]
    pcc_loss = official["pcc_loss_sum"]
    loss1 = mse_loss(trt_output, trt_raw_data)
    cell_class_true = output[6]
    cell_class_predict = output[7]
    if cell_class_predict is None:
        loss2 = mse_loss(ctl_output, ctl_raw_data)
    else:
        import torch

        class_loss = torch.nn.CrossEntropyLoss(reduction="sum")
        loss2 = class_loss(cell_class_predict[0], cell_class_true) + class_loss(cell_class_predict[1], cell_class_true)
    loss3 = mse_loss(deg_output, trt_raw_data - ctl_raw_data)
    loss4 = pcc_loss(deg_output, trt_raw_data - ctl_raw_data)
    a, b, c, d = config["train"]["loss_weight"]
    return loss1 * a + loss2 * b + loss3 * c + loss4 * d, (loss1, loss2, loss3, loss4)


def _predict(model: Any, loader: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    model.eval()
    true_rows: list[np.ndarray] = []
    ctl_rows: list[np.ndarray] = []
    pred_rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch)
            deg_pred = output[2].detach().cpu().numpy()
            ctl = batch[1].detach().cpu().numpy()
            trt = batch[0].detach().cpu().numpy()
            pred_rows.append(deg_pred)
            ctl_rows.append(ctl)
            true_rows.append(trt - ctl)
    return np.concatenate(true_rows), np.concatenate(pred_rows), np.concatenate(ctl_rows)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size == 0 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _metrics(true_delta: np.ndarray, pred_delta: np.ndarray) -> dict[str, Any]:
    true_flat = true_delta.reshape(-1)
    pred_flat = pred_delta.reshape(-1)
    row_pearson = [_corr(left, right) for left, right in zip(true_delta, pred_delta, strict=True)]
    row_spearman = [_corr(_rank(left), _rank(right)) for left, right in zip(true_delta, pred_delta, strict=True)]
    return {
        "rows": int(len(true_delta)),
        "genes": int(true_delta.shape[1]),
        "mse": float(np.mean((true_delta - pred_delta) ** 2)),
        "rmse": float(np.sqrt(np.mean((true_delta - pred_delta) ** 2))),
        "pearson_flat": _corr(true_flat, pred_flat),
        "spearman_flat": _corr(_rank(true_flat), _rank(pred_flat)),
        "pearson_row_mean": float(np.nanmean(row_pearson)),
        "spearman_row_mean": float(np.nanmean(row_spearman)),
        "prediction_std": float(np.std(pred_delta)),
    }


def _train(
    model: Any,
    loader: Any,
    *,
    config: dict[str, Any],
    official: dict[str, Any],
    epochs: int,
) -> list[dict[str, Any]]:
    import torch

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["train"]["train_lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    history: list[dict[str, Any]] = []
    model.train()
    for epoch in range(epochs):
        losses: list[float] = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            total, components = _loss(model, batch, config, official)
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite EXP-005 loss at epoch {epoch}")
            total.backward()
            optimizer.step()
            losses.append(float(total.detach().cpu()))
        history.append({"epoch": int(epoch), "mean_loss": float(np.mean(losses)), "batches": len(losses)})
    return history


def run(args: argparse.Namespace) -> dict[str, Any]:
    import scanpy as sc
    import torch
    from torch.utils.data import DataLoader
    import yaml

    if args.variant not in {"A", "B", "C"}:
        raise ValueError("variant must be A, B, or C")
    _seed_everything(args.seed)
    official = _official_imports()
    logger = _logger()
    config = yaml.safe_load((SOURCE / "configs" / "config_l1000_foundation_bounded.yaml").read_text())
    # Official model constructors resolve these paths relative to their
    # checkout; make the same official assets explicit for a repo-level run.
    config["model"]["ATTN"]["ppi_gene_vector_path"] = str(SOURCE / "processed_data" / "PPI_gene_vector_128d.npy")
    config["model"]["HG"]["drug_hg_pretrained_embed_path"] = str(SOURCE / "HG_data" / "saved_embedding" / "HG_drug_embeddings.npy")
    args_model = _args(device=args.device)
    args_model.finetune_official = bool(getattr(args, "finetune_official", False))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    backed = sc.read_h5ad(FULL_DATA, backed="r")
    split_positions = _split_positions(backed.obs, args.split, args.max_rows)
    if any(len(split_positions[key]) == 0 for key in ("train", "test")):
        raise ValueError(f"requested split has an empty partition: {args.split}")
    train_data = _memory_subset(backed, split_positions["train"])
    test_data = _memory_subset(backed, split_positions["test"])
    split_contract = _cold_split_contract(backed.obs, split_positions, args.split)
    try:
        backed.file.close()
    except Exception:
        pass

    unimol = np.load(UNIMOL, mmap_mode="r", allow_pickle=False)
    kpgt = np.load(KPGT, allow_pickle=True).item() if args.variant in {"B", "C"} else None
    unipert = np.load(UNIPERT, mmap_mode="r", allow_pickle=False) if args.variant == "C" else None
    train_dataset = _dataset(data=train_data, variant=args.variant, args=args_model, config=config, logger=logger, unimol=unimol, kpgt=kpgt, unipert=unipert, official=official)
    test_dataset = _dataset(data=test_data, variant=args.variant, args=args_model, config=config, logger=logger, unimol=unimol, kpgt=kpgt, unipert=unipert, official=official)
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 0, "drop_last": False}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = _model(
        variant=args.variant,
        args=args_model,
        config=config,
        device=device,
        logger=logger,
        official=official,
        checkpoint=getattr(args, "checkpoint", None),
    )
    history = _train(model, train_loader, config=config, official=official, epochs=args.epochs)
    test_true, test_pred, _ = _predict(model, test_loader)

    output: dict[str, Any] = {
        "format": "exp005_xpert_additive_fast_v1",
        "status": "COMPLETE",
        "variant": args.variant,
        "variant_definition": {"A": "XPert official", "B": "XPert + KPGT token", "C": "XPert + KPGT token + UniPert token"}[args.variant],
        "split": args.split,
        "seed": int(args.seed),
        "budget": {"epochs": int(args.epochs), "batch_size": int(args.batch_size), "max_rows_per_partition": int(args.max_rows), "optimizer": "Adam", "loss": "official XPert loss_weight [0.2, 0.003, 0.2, 1] including official cell-index classification branch", "precision": "float32"},
        "partitions": {key: int(len(value)) for key, value in split_positions.items()},
        "data_contract": split_contract,
        "training": history,
        "test_delta978": _metrics(test_true, test_pred),
        "official_architecture_preserved": True,
        "checkpoint_inheritance": getattr(model, "checkpoint_audit", None),
        "official_parameters_frozen": getattr(model, "official_parameters_frozen", False),
        "additive_gate_init": (
            model.additive_gate.detach().cpu().tolist()
            if hasattr(model, "additive_gate")
            else None
        ),
        "token_policy": "KPGT and UniPert are separate projected tokens appended after official UniMol/HG drug sequence; no external concatenation and no transformer rewrite",
    }

    if not args.skip_broad:
        broad_data = sc.read_h5ad(GLOBAL_ADAPTER)
        broad_dataset = _dataset(data=broad_data, variant=args.variant, args=args_model, config=config, logger=logger, unimol=unimol, kpgt=kpgt, unipert=unipert, official=official)
        broad_loader = DataLoader(broad_dataset, shuffle=False, **loader_kwargs)
        _, broad_pred, broad_ctl = _predict(model, broad_loader)
        PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
        profile_path = PROFILE_ROOT / f"{args.split}_{args.variant}_broad_profile.npy"
        np.save(profile_path, {"deg_pred": broad_pred.astype(np.float32), "ctl_true": broad_ctl.astype(np.float32), "y_pred": (broad_pred + broad_ctl).astype(np.float32)})
        from drug_screen.evaluation.xpert_broad import build as evaluate_broad

        evaluation_path = RESULT_ROOT / f"{args.split}_{args.variant}_BROAD_EVALUATION.json"
        evaluation = evaluate_broad(
            profile_path=profile_path,
            adapter_path=GLOBAL_ADAPTER,
            signature_path=SIGNATURE,
            prism_path=BROAD_RESPONSE,
            observed_lincs_path=FULL_DATA,
            minimum_candidates=20,
        )
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation_path.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        output["broad"] = {
            "profile": str(profile_path),
            "evaluation": str(evaluation_path),
            "line_metrics": evaluation["broad_prism"]["line_metrics"],
            "oracle_line_metrics": evaluation["observed_lincs_oracle"].get("line_metrics"),
        }

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    result_path = RESULT_ROOT / f"{args.split}_{args.variant}.json"
    result_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "variant": args.variant, "split": args.split, "test": output["test_delta978"], "broad": output.get("broad", {}).get("line_metrics")}, sort_keys=True))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["A", "B", "C"], required=True)
    parser.add_argument("--split", default="split_cold_cell_1")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=4096)
    parser.add_argument("--skip-broad", action="store_true")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="official XPert checkpoint to inherit (extension parameters remain freshly gated)",
    )
    parser.add_argument(
        "--finetune-official",
        action="store_true",
        help="opt in to updating inherited official XPert weights (default: frozen foundation)",
    )
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
