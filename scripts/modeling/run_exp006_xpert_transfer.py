"""EXP-006 A vs B large-scale Genetic→Chemical transfer on XPert.

A: official UniMol/HG XPert, chemical-only fine-tune
B: same backbone after genetic pretraining through a one-token UniPert adapter

Unique variable: genetic pretraining. No new fusion, no 256-gene MLP.
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
CHECKPOINT = SOURCE / "saved_model" / "l1000_sdst_warm_split.pth"
CONFIG = SOURCE / "configs" / "config_l1000_foundation_bounded.yaml"
CONTRACT = ROOT / "artifacts" / "experiments" / "EXP-006" / "LARGE_SCALE_TRANSFER_CONTRACT.json"
SPLITS = ROOT / "artifacts" / "experiments" / "EXP-006" / "compound_splits.json"
COVERAGE = ROOT / "artifacts" / "experiments" / "EXP-006" / "CONTEXT_COVERAGE.json"
GENETIC_H5AD = ROOT / "artifacts" / "experiments" / "EXP-006" / "genetic_paired_adapter.h5ad"
UNIPERT_FEATURES = ROOT / "artifacts" / "experiments" / "EXP-006" / "unipert_genetic_features.npy"
SIGNATURE = ROOT / "mvp" / "core_data" / "crc_disease_signature_exact978.tsv"
PRISM = ROOT / "mvp" / "foundation" / "xpert" / "BROAD_PRISM_CRC_V1.parquet"
RUN_ROOT = ROOT / "artifacts" / "experiments" / "EXP-006" / "runs"
RESULT = ROOT / "artifacts" / "experiments" / "EXP-006" / "TRANSFER_RESULT.json"

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


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
    logger = logging.getLogger("drugscreenlab.exp006")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    return logger


def _args(*, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        mode="train",
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


class Exp006PairedDataset:
    """Lazy paired exact978 dataset compatible with official XPert fields."""

    def __init__(
        self,
        treated: np.ndarray,
        control: np.ndarray,
        obs: Any,
        *,
        unimol: np.ndarray,
        config: dict[str, Any],
        unipert: np.ndarray | None = None,
        genetic: bool = False,
    ) -> None:
        import torch
        from datasets.MyDataset import assign_pert_dose_corrected, _digitize

        self.torch = torch
        self.treated = np.asarray(treated, dtype=np.float32)
        self.control = np.asarray(control, dtype=np.float32)
        self.obs = obs.reset_index(drop=True).copy()
        self.unimol = unimol
        self.unipert = unipert
        self.genetic = bool(genetic)
        n_bins = int(config["dataset"]["n_bins"])
        bins = np.quantile(
            np.array([float(config["dataset"]["min_value"]), float(config["dataset"]["max_value"])]),
            np.linspace(0, 1, n_bins - 1),
        )
        self.treated_binned = _digitize(self.treated, bins, side="one")
        self.control_binned = _digitize(self.control, bins, side="one")
        if "pert_dose_idx" not in self.obs.columns:
            self.obs["pert_dose_idx"] = (
                self.obs["pert_dose"].astype(np.float32).apply(assign_pert_dose_corrected)
            )
        if "pert_time_idx" not in self.obs.columns:
            time_map = {3: 0, 6: 1, 24: 2, 3.0: 0, 6.0: 1, 24.0: 2, 96: 2, 96.0: 2}
            self.obs["pert_time_idx"] = self.obs["pert_time"].astype(np.float32).map(time_map).fillna(2)

    def __len__(self) -> int:
        return int(len(self.obs))

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        torch = self.torch
        row = self.obs.iloc[index]
        pert_idx = int(row["pert_idx"])
        drug_feat = np.asarray(self.unimol[pert_idx], dtype=np.float32)
        item = (
            torch.as_tensor(self.treated[index], dtype=torch.float32),
            torch.as_tensor(self.control[index], dtype=torch.float32),
            torch.as_tensor(self.treated_binned[index], dtype=torch.int64),
            torch.as_tensor(self.control_binned[index], dtype=torch.int64),
            torch.as_tensor(drug_feat, dtype=torch.float32),
            torch.as_tensor(int(row["pert_dose_idx"]), dtype=torch.int64),
            torch.as_tensor(int(row["pert_time_idx"]), dtype=torch.int64),
            torch.as_tensor(pert_idx, dtype=torch.long),
            torch.as_tensor(int(row["cell_idx"]), dtype=torch.long),
            torch.as_tensor(int(row["tissue_idx"]), dtype=torch.long),
        )
        if self.genetic:
            if self.unipert is None:
                raise ValueError("genetic dataset requires UniPert features")
            feature = np.asarray(self.unipert[int(row["unipert_row"])], dtype=np.float32)
            direction = int(row["direction_idx"])
            return (
                *item,
                torch.as_tensor(feature, dtype=torch.float32),
                torch.as_tensor(direction, dtype=torch.long),
            )
        return item


def _chemical_arrays(adata: Any, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, Any]:
    positions = np.flatnonzero(mask)
    if len(positions) == 0:
        raise ValueError("requested chemical subset is empty")
    subset = adata[positions]
    treated = np.asarray(subset.X, dtype=np.float32)
    control = np.asarray(subset.obsm["X_ctl"], dtype=np.float32)
    if hasattr(treated, "toarray"):
        treated = treated.toarray()
    if hasattr(control, "toarray"):
        control = control.toarray()
    return treated, control, subset.obs.copy()


def _official_imports() -> dict[str, Any]:
    from models.model_XPert import XPertNet
    from utils import mse_loss_ls_sum, pcc_loss_sum

    return {"XPertNet": XPertNet, "mse_loss_ls_sum": mse_loss_ls_sum, "pcc_loss_sum": pcc_loss_sum}


def _load_config() -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["model"]["ATTN"]["ppi_gene_vector_path"] = str(SOURCE / "processed_data" / "PPI_gene_vector_128d.npy")
    config["model"]["HG"]["drug_hg_pretrained_embed_path"] = str(
        SOURCE / "HG_data" / "saved_embedding" / "HG_drug_embeddings.npy"
    )
    return config


def _build_model(
    *,
    genetic: bool,
    args: Any,
    config: dict[str, Any],
    device: Any,
    logger: logging.Logger,
    init_checkpoint: Path | str = CHECKPOINT,
) -> Any:
    official = _official_imports()
    if genetic:
        from drug_screen.foundation.exp006_transfer import build_xpert_genetic_transfer_model

        model_class = build_xpert_genetic_transfer_model(official["XPertNet"])
    else:
        model_class = official["XPertNet"]
    model = model_class(args, config, device, logger)
    model.init_weights()
    from drug_screen.foundation.xpert_extension import load_xpert_checkpoint

    load_xpert_checkpoint(model, init_checkpoint, map_location="cpu")
    return model.to(device)


def _loss(model: Any, batch: Any, config: dict[str, Any], official: dict[str, Any]) -> Any:
    import torch

    output = model(batch)
    trt_output, ctl_output, deg_output = output[:3]
    trt_raw = batch[0].to(model.device)
    ctl_raw = batch[1].to(model.device)
    mse_loss = official["mse_loss_ls_sum"]
    pcc_loss = official["pcc_loss_sum"]
    loss1 = mse_loss(trt_output, trt_raw)
    cell_class_true = output[6]
    cell_class_predict = output[7]
    if cell_class_predict is None:
        loss2 = mse_loss(ctl_output, ctl_raw)
    else:
        class_loss = torch.nn.CrossEntropyLoss(reduction="sum")
        loss2 = class_loss(cell_class_predict[0], cell_class_true) + class_loss(
            cell_class_predict[1], cell_class_true
        )
    loss3 = mse_loss(deg_output, trt_raw - ctl_raw)
    loss4 = pcc_loss(deg_output, trt_raw - ctl_raw)
    a, b, c, d = config["train"]["loss_weight"]
    return loss1 * a + loss2 * b + loss3 * c + loss4 * d


def _train(
    model: Any,
    loader: Any,
    *,
    config: dict[str, Any],
    official: dict[str, Any],
    epochs: int,
    lr: float,
    logger: logging.Logger,
    tag: str,
) -> list[dict[str, Any]]:
    import torch

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(config["train"]["weight_decay"]))
    history: list[dict[str, Any]] = []
    model.train()
    for epoch in range(epochs):
        losses: list[float] = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            total = _loss(model, batch, config, official)
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite EXP-006 loss at {tag} epoch {epoch}")
            total.backward()
            optimizer.step()
            losses.append(float(total.detach().cpu()))
        record = {"epoch": int(epoch), "mean_loss": float(np.mean(losses)), "batches": len(losses), "tag": tag}
        history.append(record)
        logger.info("%s", record)
    return history


def _predict(model: Any, loader: Any) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    true_rows: list[np.ndarray] = []
    pred_rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch)
            pred_rows.append(output[2].detach().cpu().numpy())
            true_rows.append((batch[0] - batch[1]).detach().cpu().numpy())
    return np.concatenate(true_rows), np.concatenate(pred_rows)


def _downstream(true_obs: Any, pred_delta: np.ndarray, contexts: list[str]) -> dict[str, Any]:
    from drug_screen.evaluation.phase1_prism import reversal_score
    from drug_screen.evaluation.xpert_broad import rank_metrics

    if not PRISM.exists() or not SIGNATURE.exists():
        return {"status": "SKIPPED_MISSING_ASSETS"}
    signature = __import__("pandas").read_csv(SIGNATURE, sep="\t")
    indices = signature["gene_index_978"].astype(int).to_numpy()
    values = signature["signed_log2fc"].astype(float).to_numpy()
    prism = __import__("pandas").read_parquet(PRISM)
    if "sensitivity_score" not in prism.columns:
        return {"status": "SKIPPED_NO_SENSITIVITY"}
    prism = prism.copy()
    prism["base_context"] = prism["ccle_name"].astype(str).str.split("_").str[0]
    rows = []
    for index, row in true_obs.reset_index(drop=True).iterrows():
        rows.append(
            {
                "cell_iname": str(row["cell_iname"]),
                "pert_id": str(row["pert_id"]),
                "reversal_predicted": reversal_score(values, pred_delta[index][indices]),
            }
        )
    predicted = __import__("pandas").DataFrame(rows)
    predicted = (
        predicted.groupby(["cell_iname", "pert_id"], as_index=False, sort=True)
        .agg(reversal_predicted=("reversal_predicted", "mean"))
    )
    joined = predicted.merge(
        prism,
        left_on=["cell_iname", "pert_id"],
        right_on=["base_context", "pert_id"],
        how="inner",
    )
    if joined.empty:
        return {
            "status": "NO_PRISM_OVERLAP",
            "note": "frozen chemical test in selected contexts has no exact PRISM join; not used for selection",
        }
    line_rows = []
    for context_id, group in joined.groupby("cell_iname", sort=True, observed=True):
        metrics = rank_metrics(group, score_column="reversal_predicted", null_seed=20260813)
        line_rows.append({"context_id": str(context_id), **metrics})
    eligible = [row for row in line_rows if row.get("eligible")]
    lift = [
        row["top_k"]["10"]["overlap_lift"]
        for row in eligible
        if "10" in row.get("top_k", {}) and row["top_k"]["10"].get("overlap_lift") is not None
    ]
    ndcg = [
        row["top_k"]["10"]["delta_ndcg"]
        for row in eligible
        if "10" in row.get("top_k", {}) and row["top_k"]["10"].get("delta_ndcg") is not None
    ]
    compact_lines = []
    for row in line_rows:
        compact = {
            "context_id": row.get("context_id"),
            "eligible": row.get("eligible"),
            "candidate_count": row.get("candidate_count"),
            "spearman": row.get("spearman"),
        }
        top = row.get("top_k") or {}
        compact_top = {}
        for key in ("10", "20", "50"):
            if key not in top:
                continue
            item = top[key]
            compact_top[key] = {
                "effective_k": item.get("effective_k"),
                "overlap_count": item.get("overlap_count"),
                "overlap_rate": item.get("overlap_rate"),
                "hitrate_recall_at_k": item.get("overlap_rate"),
                "ndcg": item.get("ndcg"),
                "overlap_lift_vs_null": item.get("overlap_lift"),
                "ndcg_excess_vs_null": item.get("delta_ndcg"),
            }
        compact["top_k"] = compact_top
        if isinstance(row.get("null_baseline"), dict):
            compact["null_spearman_mean"] = row["null_baseline"].get("spearman_mean")
        compact_lines.append(compact)
    return {
        "status": "EVALUATED_AFTER_PREDICTION_FREEZE",
        "prism_used_for_selection": False,
        "line_rows": compact_lines,
        "macro_top10_lift_vs_null": float(np.mean(lift)) if lift else None,
        "macro_ndcg10_excess_vs_null": float(np.mean(ndcg)) if ndcg else None,
        "eligible_lines": int(len(eligible)),
        "joined_pairs": int(len(joined)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import scanpy as sc
    import torch
    from torch.utils.data import DataLoader

    from drug_screen.foundation.exp006_transfer import delta978_metrics, write_json

    logger = _logger()
    _seed_everything(args.seed)
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    contexts = [str(value) for value in contract["selected_contexts"]]
    train_compounds = set(splits["fractions"][str(args.fraction) if args.fraction != 1.0 else "1.0"])
    if args.fraction == 1.0:
        train_compounds = set(splits["split"]["train"])
    elif args.fraction == 0.2:
        train_compounds = set(splits["fractions"]["0.2"])
    elif args.fraction == 0.1:
        train_compounds = set(splits["fractions"]["0.1"])
    else:
        raise ValueError("fraction must be 1.0, 0.2, or 0.1")
    test_compounds = set(splits["split"]["test"])
    if train_compounds & test_compounds:
        raise ValueError("train/test unique-compound leakage")

    config = _load_config()
    official = _official_imports()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_args = _args(device=str(device))
    unimol = np.load(UNIMOL, mmap_mode="r", allow_pickle=False)

    chemical = sc.read_h5ad(FULL_DATA, backed="r")
    obs = chemical.obs
    context_mask = obs["cell_iname"].astype(str).isin(contexts)
    train_mask = context_mask & obs["pert_id"].astype(str).isin(train_compounds)
    test_mask = context_mask & obs["pert_id"].astype(str).isin(test_compounds)
    train_x, train_ctl, train_obs = _chemical_arrays(chemical, train_mask.to_numpy())
    test_x, test_ctl, test_obs = _chemical_arrays(chemical, test_mask.to_numpy())
    try:
        chemical.file.close()
    except Exception:
        pass

    train_ds = Exp006PairedDataset(train_x, train_ctl, train_obs, unimol=unimol, config=config)
    test_ds = Exp006PairedDataset(test_x, test_ctl, test_obs, unimol=unimol, config=config)
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 0, "drop_last": False}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    genetic_history: list[dict[str, Any]] = []
    genetic = args.model == "B"
    init_checkpoint = Path(args.init_checkpoint) if args.init_checkpoint else CHECKPOINT
    model = _build_model(
        genetic=genetic,
        args=model_args,
        config=config,
        device=device,
        logger=logger,
        init_checkpoint=init_checkpoint,
    )
    if genetic and not args.skip_genetic:
        genetic_data = sc.read_h5ad(args.genetic_h5ad)
        features = np.load(args.unipert_features, mmap_mode="r")
        genetic_ds = Exp006PairedDataset(
            np.asarray(genetic_data.X, dtype=np.float32),
            np.asarray(genetic_data.obsm["X_ctl"], dtype=np.float32),
            genetic_data.obs,
            unimol=unimol,
            config=config,
            unipert=features,
            genetic=True,
        )
        genetic_loader = DataLoader(genetic_ds, shuffle=True, **loader_kwargs)
        logger.info("genetic pretrain records=%s genes=%s", len(genetic_ds), genetic_data.obs["gene_symbol"].nunique())
        genetic_history = _train(
            model,
            genetic_loader,
            config=config,
            official=official,
            epochs=args.genetic_epochs,
            lr=args.lr,
            logger=logger,
            tag="genetic_pretrain",
        )
        del genetic_loader, genetic_ds, genetic_data

    chemical_history: list[dict[str, Any]] = []
    if args.chemical_epochs > 0:
        chemical_history = _train(
            model,
            train_loader,
            config=config,
            official=official,
            epochs=args.chemical_epochs,
            lr=args.lr,
            logger=logger,
            tag=f"chemical_ft_{args.model}_{args.fraction}",
        )
    if args.chemical_epochs > 0:
        true_delta, pred_delta = _predict(model, test_loader)
        metrics = delta978_metrics(true_delta, pred_delta)
        downstream = _downstream(test_obs, pred_delta, contexts) if args.downstream else {"status": "DEFERRED"}
    else:
        metrics = {"status": "PRETRAIN_ONLY_NO_CHEMICAL_EVAL"}
        downstream = {"status": "PRETRAIN_ONLY"}

    run_dir = RUN_ROOT / f"{args.model}_frac{args.fraction}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.save_checkpoint:
        torch.save(model.state_dict(), run_dir / "model.pt")
    payload = {
        "format": "exp006_xpert_transfer_run_v1",
        "model": args.model,
        "fraction": float(args.fraction),
        "seed": int(args.seed),
        "contexts": contexts,
        "train_unique_compounds": int(len(train_compounds)),
        "test_unique_compounds": int(len(test_compounds)),
        "train_rows": int(len(train_ds)),
        "test_rows": int(len(test_ds)),
        "genetic_pretrain_epochs": int(args.genetic_epochs if genetic else 0),
        "chemical_finetune_epochs": int(args.chemical_epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "optimizer": "Adam",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_audit": getattr(model, "checkpoint_audit", None),
        "genetic_history": genetic_history,
        "chemical_history": chemical_history,
        "test_delta978": metrics,
        "downstream": downstream,
        "unique_variable": "genetic_pretraining",
        "new_fusion": False,
    }
    write_json(run_dir / "metrics.json", payload)
    logger.info("wrote %s", run_dir / "metrics.json")
    return payload


def merge_results(paths: list[Path], output: Path) -> dict[str, Any]:
    from drug_screen.foundation.exp006_transfer import write_json

    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    table: dict[str, Any] = {}
    for run in runs:
        table.setdefault(str(run["fraction"]), {})[run["model"]] = {
            "delta978": run["test_delta978"],
            "downstream": run.get("downstream"),
            "train_unique_compounds": run["train_unique_compounds"],
            "test_unique_compounds": run["test_unique_compounds"],
        }
    comparison = {}
    for fraction, models in table.items():
        if "A" in models and "B" in models:
            a = models["A"]["delta978"]
            b = models["B"]["delta978"]
            comparison[fraction] = {
                "spearman_gain_B_minus_A": (
                    None
                    if a["spearman_row_mean"] is None or b["spearman_row_mean"] is None
                    else float(b["spearman_row_mean"] - a["spearman_row_mean"])
                ),
                "mse_B_minus_A": float(b["mse"] - a["mse"]),
                "direction_B_minus_A": float(b["direction_consistency"] - a["direction_consistency"]),
            }
    payload = {
        "format": "exp006_transfer_result_v1",
        "status": "PARTIAL" if len(runs) < 6 else "COMPLETE",
        "seed": 20260813,
        "runs": table,
        "comparison": comparison,
        "primary_regimes": [0.2, 0.1],
        "failures_kept": [],
    }
    write_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["A", "B"], required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--genetic-epochs", type=int, default=3)
    parser.add_argument("--chemical-epochs", type=int, default=3)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--splits", type=Path, default=SPLITS)
    parser.add_argument("--genetic-h5ad", type=Path, default=GENETIC_H5AD)
    parser.add_argument("--unipert-features", type=Path, default=UNIPERT_FEATURES)
    parser.add_argument("--downstream", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--skip-genetic", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if args.merge_only:
        paths = sorted(RUN_ROOT.glob("*_frac*_seed*/metrics.json"))
        print(json.dumps(merge_results(paths, RESULT), sort_keys=True))
        return 0
    payload = run(args)
    print(json.dumps({"model": payload["model"], "fraction": payload["fraction"], "test": payload["test_delta978"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
