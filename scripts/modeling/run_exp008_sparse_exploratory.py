from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
DEFAULT_H5AD = ROOT / "data/external/xpert_source/processed_data/l1000_sdst_78453.h5ad"
DEFAULT_FEATURES = ROOT / "artifacts/experiments/EXP-008/target_pathway_features.npy"
DEFAULT_OUT = ROOT / "artifacts/experiments/EXP-008/exploratory"


def _official_imports() -> dict[str, Any]:
    import sys
    source = ROOT / "data/external/xpert_source"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from datasets.MyDataset import MyDataset
    from models.model_XPert import XPertNet
    from utils import mse_loss_ls_sum, pcc_loss_sum
    return {"XPertNet": XPertNet, "MyDataset": MyDataset, "mse_loss_ls_sum": mse_loss_ls_sum, "pcc_loss_sum": pcc_loss_sum}


def _load_config() -> dict[str, Any]:
    import yaml
    source = ROOT / "data/external/xpert_source"
    config = yaml.safe_load((source / "configs/config_l1000_foundation_bounded.yaml").read_text(encoding="utf-8"))
    config["model"]["ATTN"]["ppi_gene_vector_path"] = str(source / "processed_data/PPI_gene_vector_128d.npy")
    config["model"]["HG"]["drug_hg_pretrained_embed_path"] = str(source / "HG_data/saved_embedding/HG_drug_embeddings.npy")
    return config


def _official_args(device: str) -> Any:
    from types import SimpleNamespace
    return SimpleNamespace(mode="train", dataset="l1000_sdst", drug_feat="unimol", device=device, pretrained_mode="global", include_cell_idx=True, wo_HG=False, wo_atom=False, wo_atom_HG=False, wo_unimol=False, wo_ppi=False, use_gene_pos_emed=False, output_attention=False, output_cls_embed=False)


def _build_official_model(device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    from drug_screen.foundation.xpert_extension import load_xpert_checkpoint
    source = ROOT / "data/external/xpert_source"
    official = _official_imports()
    import logging
    logger = logging.getLogger("exp008.sparse.xpert")
    model = official["XPertNet"](_official_args(str(device)), _load_config(), device, logger)
    model.init_weights()
    audit = load_xpert_checkpoint(model, source / "saved_model/l1000_sdst_warm_split.pth", strict_official=True, map_location="cpu")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device), audit


class SparseMechanismResidual(nn.Module):
    def __init__(self, backbone: nn.Module, *, feature_dim: int = 978, n_genes: int = 978) -> None:
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.raw_gamma = nn.Parameter(torch.zeros((), dtype=torch.float32))
        if feature_dim != 978:
            raise ValueError("EXP-008 mechanism contract v1 requires 978-dimensional features")
        self.encoder = nn.Sequential(nn.Linear(feature_dim, 256), nn.LayerNorm(256), nn.GELU())
        self.output = nn.Linear(256, n_genes)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @property
    def gamma(self) -> Tensor:
        return 0.05 * torch.tanh(self.raw_gamma)

    def forward(self, batch: tuple[Tensor, ...] | Tensor, features: Tensor | None = None) -> Tensor:
        if torch.is_tensor(batch):
            if features is None:
                raise ValueError("tensor baseline requires mechanism features")
            baseline = self.backbone(batch)
            return baseline + self.gamma * self.output(self.encoder(features))
        if len(batch) != 11:
            raise ValueError("expected official ten-field batch plus one mechanism feature tensor")
        output = self.backbone(batch[:10])
        baseline = output[2]
        residual = self.output(self.encoder(batch[10]))
        return baseline + self.gamma * residual


def align_contract_features(features: Tensor | np.ndarray, ordered_pert_idx: list[int] | np.ndarray, sample_pert_idx: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    ordered = np.asarray(ordered_pert_idx, dtype=np.int64)
    samples = np.asarray(sample_pert_idx, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != len(ordered) or matrix.shape[1] != 978:
        raise ValueError(f"feature matrix must be (len(ordered_pert_idx),978), got {matrix.shape}")
    if len(np.unique(ordered)) != len(ordered):
        raise ValueError("ordered_pert_idx must be unique")
    positions = {int(pert_idx): index for index, pert_idx in enumerate(ordered.tolist())}
    aligned = np.zeros((len(samples), 978), dtype=np.float32)
    for row, pert_idx in enumerate(samples.tolist()):
        source = positions.get(int(pert_idx))
        if source is not None:
            aligned[row] = matrix[source]
    return aligned


class _FeatureDataset(Dataset[tuple[Any, ...]]):
    def __init__(self, official_dataset: Dataset[Any], features: Tensor) -> None:
        if len(official_dataset) != len(features):
            raise ValueError("mechanism features and official dataset row counts differ")
        self.official_dataset = official_dataset
        self.features = features

    def __len__(self) -> int:
        return len(self.official_dataset)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        return (*self.official_dataset[index], self.features[index])


def align_feature_matrix(features: Tensor | np.ndarray, pert_idx: list[int] | np.ndarray) -> Tensor:
    matrix = torch.as_tensor(features, dtype=torch.float32)
    indices = np.asarray(pert_idx, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape != (8276, 978):
        raise ValueError(f"expected feature matrix (8276,978), got {tuple(matrix.shape)}")
    if indices.ndim != 1 or len(indices) != 8276 or len(np.unique(indices)) != 8276 or np.any(indices < 0) or np.any(indices >= 8276):
        raise ValueError("pert_idx must be a permutation of all 8276 drug rows")
    return matrix[torch.as_tensor(indices, dtype=torch.long)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EXP-008 sparse exploratory residual; never formal evidence")
    parser.add_argument("--full-sdst", action="store_true", required=True)
    parser.add_argument("--split", choices=("split_cold_drug_1", "split_cold_cell_1"), default="split_cold_drug_1")
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--smoke-batches", type=int, choices=(1,), default=None, help="run exactly one real official batch and emit smoke audit only")
    return parser


def _spearman(pred: np.ndarray, target: np.ndarray) -> float:
    from scipy.stats import spearmanr
    values = [spearmanr(pred[i], target[i]).statistic for i in range(pred.shape[0])]
    return float(np.nanmean(np.asarray(values, dtype=np.float64)))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float), encoding="utf-8")


def _load_data(args: argparse.Namespace) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import anndata as ad
    data = ad.read_h5ad(args.h5ad)
    if tuple(data.shape) != (78453, 978) or "X_ctl" not in data.obsm:
        raise ValueError(f"official SDST shape/X_ctl mismatch: shape={data.shape}")
    obs = data.obs
    if args.split not in obs:
        raise ValueError(f"missing split column {args.split}")
    x = np.asarray(data.X, dtype=np.float32)
    ctl = np.asarray(data.obsm["X_ctl"], dtype=np.float32)
    delta = x - ctl
    features = np.load(args.features, mmap_mode="r")
    contract_path = args.features.parent / "MECHANISM_CONTRACT.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    ordered_pert_idx = contract["drug_index"]["ordered_pert_idx"]
    if tuple(features.shape) != (len(ordered_pert_idx), 978):
        raise ValueError(f"feature shape/contract mismatch: {features.shape}")
    feature_rows = align_contract_features(features, ordered_pert_idx, np.asarray(obs["pert_idx"], dtype=np.int64))
    labels = obs[args.split].astype(str).to_numpy()
    if set(labels) != {"train", "test"}:
        raise ValueError(f"unexpected split labels: {sorted(set(labels))}")
    train = labels == "train"
    test = labels == "test"
    identity_col = "pert_id" if "drug" in args.split else "cell_iname"
    identities = obs[identity_col].astype(str).to_numpy()
    overlap = len(set(identities[train]) & set(identities[test]))
    if overlap != 0:
        raise ValueError(f"identity overlap for {identity_col}: {overlap}")
    feature_sha = hashlib.sha256(np.asarray(features).tobytes()).hexdigest()
    meta = {"shape": list(data.shape), "train_rows": int(train.sum()), "test_rows": int(test.sum()), "split": args.split, "label_counts": {"train": int(train.sum()), "test": int(test.sum())}, "identity_column": identity_col, "identity_overlap": overlap, "contract_format": contract.get("format"), "contract_status": contract.get("status"), "contract_feature_shape": contract["features"]["shape"], "contract_feature_sha256": contract["features"]["sha256"], "feature_sha256_observed": feature_sha, "ordered_pert_idx_count": len(ordered_pert_idx), "sample_rows_mapped": int(sum(int(value) in set(ordered_pert_idx) for value in np.asarray(obs["pert_idx"], dtype=np.int64))), "sample_rows_zero_filled": int(sum(int(value) not in set(ordered_pert_idx) for value in np.asarray(obs["pert_idx"], dtype=np.int64)))}
    return data, delta, feature_rows, train, meta


def _smoke_audit(model: SparseMechanismResidual, loader: DataLoader, device: torch.device, checkpoint_audit: dict[str, Any]) -> dict[str, Any]:
    batch = next(iter(loader))
    moved = tuple(item.to(device) if torch.is_tensor(item) else item for item in batch)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if any(name.startswith("backbone.") for name in trainable):
        raise RuntimeError(f"official backbone parameters unexpectedly trainable: {trainable}")
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3)
    model.train()
    model.backbone.eval()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(moved)
    target = moved[0] - moved[1]
    loss = torch.mean((prediction - target) ** 2)
    if not torch.isfinite(loss):
        raise RuntimeError("smoke loss is non-finite")
    loss.backward()
    optimizer.step()
    return {"result_status": "EXPLORATORY_SMOKE_ONLY", "batch_size": int(moved[0].shape[0]), "batch_shapes": [list(item.shape) for item in moved if torch.is_tensor(item)], "trainable_parameter_names": trainable, "loss_finite": bool(torch.isfinite(loss).item()), "loss": float(loss.detach().cpu()), "checkpoint_audit": checkpoint_audit}


def _evaluate(model: SparseMechanismResidual, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            moved = tuple(item.to(device) if torch.is_tensor(item) else item for item in batch)
            predictions.append(model(moved).cpu().numpy())
            targets.append((batch[0] - batch[1]).numpy())
    pred = np.concatenate(predictions)
    target = np.concatenate(targets)
    return {"rows": int(len(pred)), "delta_spearman_macro": _spearman(pred, target), "prediction_std": float(pred.std()), "target_std": float(target.std())}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    random.seed(20260815)
    np.random.seed(20260815)
    torch.manual_seed(20260815)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"result_status": "DATA_BLOCKED", "contract_status": "DATA_BLOCKED", "mode": "exploratory", "data_sparse": True, "feature_coverage": 0.019333, "string_status": "aliases_unavailable_zero_block", "GO_status": "no_GO_BP_block", "seed": 20260815, "official_xpert_used": False, "surrogate_used": False}
    try:
        data, delta, features, train_mask, meta = _load_data(args)
        manifest.update(meta)
        test_mask = ~train_mask
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        official = _official_imports()
        config = _load_config()
        args_model = _official_args(str(device))
        source = ROOT / "data/external/xpert_source"
        unimol = np.load(source / "processed_data/all_drugs_unimol_arr.npy", mmap_mode="r", allow_pickle=False)
        train_pos = np.flatnonzero(train_mask)
        test_pos = np.flatnonzero(test_mask)
        logger = __import__("logging").getLogger("exp008.sparse")
        if args.smoke_batches == 1:
            smoke_pos = train_pos[:args.batch_size]
            train_data = data[smoke_pos].copy()
            train_base = official["MyDataset"](train_data, unimol, args=args_model, config=config, logger=logger, max_value=config["dataset"]["max_value"], min_value=config["dataset"]["min_value"])
            train_dataset = _FeatureDataset(train_base, torch.from_numpy(features[smoke_pos]))
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            test_loader = None
        else:
            train_data = data[train_pos].copy()
            test_data = data[test_pos].copy()
            train_base = official["MyDataset"](train_data, unimol, args=args_model, config=config, logger=logger, max_value=config["dataset"]["max_value"], min_value=config["dataset"]["min_value"])
            test_base = official["MyDataset"](test_data, unimol, args=args_model, config=config, logger=logger, max_value=config["dataset"]["max_value"], min_value=config["dataset"]["min_value"])
            train_dataset = _FeatureDataset(train_base, torch.from_numpy(features[train_mask]))
            test_dataset = _FeatureDataset(test_base, torch.from_numpy(features[test_mask]))
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        backbone, checkpoint_audit = _build_official_model(device)
        manifest.update({"official_xpert_used": True, "checkpoint_audit": checkpoint_audit})
        model = SparseMechanismResidual(backbone).to(device)
        if args.smoke_batches == 1:
            smoke = _smoke_audit(model, train_loader, device, checkpoint_audit)
            manifest.update({"result_status": "EXPLORATORY_SMOKE_ONLY", "smoke": smoke, "official_xpert_used": True, "checkpoint_audit": checkpoint_audit, "batch_source": "first deterministic train batch of split_cold_drug_1", "elapsed_seconds": time.time() - started})
            _write_json(args.output_dir / "exploratory_manifest.json", manifest)
            (args.output_dir / "exploratory_report.md").write_text("# EXP-008 XPert smoke audit\\n\\n" + json.dumps(manifest, indent=2, default=float) + "\\n", encoding="utf-8")
            return 0
        a_drug = _evaluate(model, test_loader, device)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        best = float("inf")
        stale = 0
        history: list[float] = []
        for _epoch in range(args.epochs):
            model.train()
            model.backbone.eval()
            losses: list[float] = []
            for batch in train_loader:
                moved = tuple(item.to(device) if torch.is_tensor(item) else item for item in batch)
                optimizer.zero_grad(set_to_none=True)
                pred = model(moved)
                target = moved[0] - moved[1]
                loss = torch.mean((pred - target) ** 2)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            epoch_loss = float(np.mean(losses))
            history.append(epoch_loss)
            if epoch_loss < best - 1e-8:
                best, stale = epoch_loss, 0
                torch.save({"model_state_dict": model.state_dict(), "epoch_loss": best, "mode": "exploratory", "checkpoint_audit": checkpoint_audit}, args.output_dir / "sparse_residual_checkpoint.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    break
        b_drug = _evaluate(model, test_loader, device)
        manifest.update({"result_status": "EXPLORATORY_TRAINED", "epochs_completed": len(history), "train_loss_history": history, "a_eval_cold_drug": a_drug, "b_eval_cold_drug": b_drug, "elapsed_seconds": time.time() - started})
        _write_json(args.output_dir / "A_eval_cold_drug.json", a_drug)
        _write_json(args.output_dir / "B_eval_cold_drug.json", b_drug)
    except Exception as exc:
        manifest.update({"result_status": "EXPLORATORY_FAILED", "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.time() - started})
    _write_json(args.output_dir / "exploratory_manifest.json", manifest)
    report = "\n".join(["# EXP-008 sparse exploratory fallback", "", f"- result_status: {manifest['result_status']}", "- contract_status: DATA_BLOCKED", "- mode: exploratory; data_sparse: true", "- feature_coverage: 0.019333", "- official_xpert_used: " + str(manifest.get("official_xpert_used", False)).lower() + "; surrogate_used: false", "- formal EXP-008 PASS/VALID: prohibited", "", "## Command", "`" + " ".join(__import__("sys").argv) + "`", "", "## Details", "```json", json.dumps(manifest, indent=2, default=float), "```"])
    (args.output_dir / "exploratory_report.md").write_text(report + "\n", encoding="utf-8")
    return 0 if manifest["result_status"] != "EXPLORATORY_FAILED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
