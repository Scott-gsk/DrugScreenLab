"""Leakage-safe source and target training loops for EXP-011."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import io
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from drug_screen.foundation.pharmaformer_adapter import TrainOnlyGeneTransform, TrainOnlySmilesEncoder
from drug_screen.modeling.exp011_data import CohortDrugBalancedBatchSampler, IndexedPairData, IndexedPairDataset
from drug_screen.modeling.exp011_models import optimizer_parameter_groups, target_phase


SOURCE_POLICY = {
    "optimizer": "AdamW",
    "learning_rate": 1e-5,
    "weight_decay": 1e-5,
    "batch_size": 64,
    "gradient_accumulation": 2,
    "mixed_precision": True,
    "gradient_clip_norm": 1.0,
    "min_epochs": 10,
    "max_epochs": 50,
    "patience": 8,
    "loss": "MSE(1-AUC)",
}
TARGET_POLICY = {
    "optimizer": "AdamW",
    "weight_decay": 1e-5,
    "batch_size": 16,
    "gradient_clip_norm": 1.0,
    "min_epochs": 20,
    "max_epochs": 100,
    "patience": 15,
    "loss": "Huber(z_response)+0.5*within_cohort_drug_pairwise_ranking",
    "adapter_head_learning_rate": 1e-4,
    "last_transformer_learning_rate": 1e-6,
}


@dataclass(frozen=True)
class FitResult:
    best_epoch: int
    epochs_run: int
    best_validation: float | None
    selection_metric: str
    policy: Mapping[str, Any]


class TrainOnlyResponseStandardizer:
    """Per-cohort/drug response z-score fitted only on the training partition."""

    def __init__(self, *, min_train_per_group: int = 8) -> None:
        self.min_train_per_group = int(min_train_per_group)
        self._mean: dict[str, float] = {}
        self._scale: dict[str, float] = {}
        self.fit_pair_indices: tuple[int, ...] = ()

    @staticmethod
    def _keys(data: IndexedPairData, indices: np.ndarray) -> np.ndarray:
        return np.asarray([f"{data.pair_cohort_id[i]}\x1f{data.pair_drug_idx[i]}" for i in indices], dtype=str)

    def fit(self, data: IndexedPairData, train_indices: Sequence[int]) -> "TrainOnlyResponseStandardizer":
        indices = np.asarray(train_indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices):
            raise ValueError("response standardizer needs non-empty training pairs")
        keys = self._keys(data, indices)
        self._mean, self._scale = {}, {}
        for key in sorted(set(keys.tolist())):
            values = data.pair_response[indices[keys == key]].astype(float)
            if len(values) < self.min_train_per_group:
                raise ValueError(f"cohort--drug {key!r} has {len(values)} train pairs; requires {self.min_train_per_group}")
            scale = float(values.std(ddof=0))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"cohort--drug {key!r} has no finite response variation in train")
            self._mean[key] = float(values.mean())
            self._scale[key] = scale
        self.fit_pair_indices = tuple(int(value) for value in indices.tolist())
        return self

    def transform(self, data: IndexedPairData, indices: Sequence[int]) -> np.ndarray:
        if not self._mean:
            raise RuntimeError("response standardizer is not fit")
        values = np.asarray(indices, dtype=np.int64)
        keys = self._keys(data, values)
        missing = sorted(set(keys.tolist()) - set(self._mean))
        if missing:
            raise ValueError(f"response transform has unseen cohort--drug groups: {missing[:3]}")
        return np.asarray([(data.pair_response[i] - self._mean[key]) / self._scale[key] for i, key in zip(values, keys)], dtype=np.float32)

    def inverse(self, data: IndexedPairData, indices: Sequence[int], values: Sequence[float]) -> np.ndarray:
        indices_array = np.asarray(indices, dtype=np.int64)
        value_array = np.asarray(values, dtype=np.float32)
        if len(indices_array) != len(value_array):
            raise ValueError("prediction values and pair indices must align")
        keys = self._keys(data, indices_array)
        return np.asarray([value * self._scale[key] + self._mean[key] for value, key in zip(value_array, keys)], dtype=np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "train_only_per_cohort_drug_zscore",
            "min_train_per_group": self.min_train_per_group,
            "mean": self._mean,
            "scale": self._scale,
            "fit_pair_indices": self.fit_pair_indices,
        }


def fit_source_bpe_codes(train_smiles: Sequence[str], output_path: str | Path) -> Path:
    """Fit upstream BPE only to source-train SMILES and persist a frozen asset."""
    values = sorted(set(str(value) for value in train_smiles))
    if not values or any(not value for value in values):
        raise ValueError("source-train SMILES must be non-empty")
    try:
        from subword_nmt.learn_bpe import learn_bpe
    except ImportError as error:
        raise RuntimeError("subword-nmt is required to fit the official frozen BPE") from error
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with io.StringIO("\n".join(values) + "\n") as source, target.open("w", encoding="utf-8") as destination:
        learn_bpe(source, destination, num_symbols=10_000)
    if not target.read_text(encoding="utf-8").strip():
        raise RuntimeError("official BPE fitting created an empty code file")
    return target


def fit_gene_transform(data: IndexedPairData, train_indices: Sequence[int]) -> TrainOnlyGeneTransform:
    """Fit expression scaling once per unique training sample, never per pair."""
    pair_indices = np.asarray(train_indices, dtype=np.int64)
    samples = np.unique(data.pair_sample_idx[pair_indices])
    return TrainOnlyGeneTransform().fit(data.expression[samples], data.sample_id[samples].tolist())


def encode_data_smiles(
    data: IndexedPairData,
    encoder: TrainOnlySmilesEncoder,
    *,
    allow_external_smiles: bool,
) -> np.ndarray:
    """Encode each unique drug once with frozen source BPE; never refit it."""
    return encoder.transform(data.canonical_smiles.tolist(), allow_external_vocabulary=allow_external_smiles)


def transformed_data(data: IndexedPairData, transform: TrainOnlyGeneTransform) -> IndexedPairData:
    """Return a view-equivalent compact data object with transformed unique expression rows."""
    from dataclasses import replace

    return replace(data, expression=transform.transform(data.expression))


def build_source_transforms(
    data: IndexedPairData,
    train_indices: Sequence[int],
    bpe_codes_path: str | Path,
) -> tuple[TrainOnlyGeneTransform, TrainOnlySmilesEncoder]:
    """Fit all source transforms from source-train identities only."""
    train = np.asarray(train_indices, dtype=np.int64)
    gene = fit_gene_transform(data, train)
    drug_indices = np.unique(data.pair_drug_idx[train])
    encoder = TrainOnlySmilesEncoder().fit(
        data.canonical_smiles[drug_indices].tolist(),
        data.drug_id[drug_indices].tolist(),
        bpe_codes_path=bpe_codes_path,
    )
    return gene, encoder


def _dataloader(dataset: IndexedPairDataset, *, batch_size: int, shuffle: bool, seed: int, target_balanced: bool = False) -> Any:
    import torch
    from torch.utils.data import DataLoader

    if target_balanced:
        return DataLoader(dataset, batch_sampler=CohortDrugBalancedBatchSampler(dataset, batch_size, seed=seed), num_workers=0)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def _amp_context(device: str) -> Any:
    import torch

    enabled = str(device).startswith("cuda") and torch.cuda.is_available()
    try:
        return torch.amp.autocast(device_type="cuda" if enabled else "cpu", enabled=enabled)
    except AttributeError:  # pragma: no cover - legacy approved torch
        return torch.cuda.amp.autocast(enabled=enabled)


def _scaler(device: str) -> Any:
    import torch

    enabled = str(device).startswith("cuda") and torch.cuda.is_available()
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - legacy approved torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _clone_state(model: Any) -> dict[str, Any]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _mse(model: Any, loader: Any, *, device: str) -> float:
    import torch
    from torch.nn import functional as functional

    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for gene, drug, target, _, _ in loader:
            output = model(gene.to(device), drug.to(device)).reshape(-1)
            loss = functional.mse_loss(output, target.to(device), reduction="sum")
            total += float(loss.detach().cpu())
            count += len(target)
    return total / count if count else float("inf")


def _rank_loss(predicted: Any, target: Any, group: Any) -> Any:
    """Average logistic pair loss among rows with the same cohort--drug group."""
    import torch
    from torch.nn import functional as functional

    loss_terms: list[Any] = []
    for value in torch.unique(group):
        indices = torch.nonzero(group == value, as_tuple=False).flatten()
        if len(indices) < 2:
            continue
        y = target[indices]
        p = predicted[indices]
        differences = y[:, None] - y[None, :]
        upper = torch.triu(torch.ones_like(differences, dtype=torch.bool), diagonal=1) & (differences != 0)
        if upper.any():
            direction = torch.sign(differences[upper])
            prediction_difference = (p[:, None] - p[None, :])[upper]
            loss_terms.append(functional.softplus(-direction * prediction_difference).mean())
    return torch.stack(loss_terms).mean() if loss_terms else predicted.sum() * 0.0


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    # Average ranks are sufficient here and avoid a SciPy dependency in the
    # model loop.  Ties are handled deterministically by stable sorting.
    def rank(value: np.ndarray) -> np.ndarray:
        order = np.argsort(value, kind="mergesort")
        result = np.empty(len(value), dtype=float)
        sorted_value = value[order]
        start = 0
        while start < len(value):
            end = start + 1
            while end < len(value) and sorted_value[end] == sorted_value[start]:
                end += 1
            result[order[start:end]] = (start + end - 1) / 2.0
            start = end
        return result
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def macro_cohort_drug_spearman(prediction: np.ndarray, target: np.ndarray, groups: np.ndarray, *, min_support: int = 3) -> float:
    """Macro-average Spearman across cohort--drug groups with enough samples."""
    values: list[float] = []
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        if int(mask.sum()) >= min_support:
            score = _spearman(prediction[mask], target[mask])
            if np.isfinite(score):
                values.append(score)
    return float(np.mean(values)) if values else float("-inf")


def predict_dataset(model: Any, dataset: IndexedPairDataset, *, device: str, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Predict indexed pairs without materialising duplicated feature matrices."""
    import torch

    loader = _dataloader(dataset, batch_size=batch_size, shuffle=False, seed=0)
    prediction: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    model.to(device).eval()
    with torch.no_grad():
        for gene, drug, _, _, pair_index in loader:
            prediction.append(model(gene.to(device), drug.to(device)).detach().cpu().numpy().reshape(-1))
            indices.append(pair_index.detach().cpu().numpy().reshape(-1))
    return np.concatenate(indices), np.concatenate(prediction).astype(np.float32)


def fit_source_model(
    model: Any,
    train_dataset: IndexedPairDataset,
    validation_dataset: IndexedPairDataset,
    *,
    device: str,
    seed: int = 20260823,
    policy: Mapping[str, Any] = SOURCE_POLICY,
) -> FitResult:
    """Train M2 with the fixed AdamW/AMP/accumulation source schedule."""
    import torch
    from torch.nn import functional as functional

    if policy["min_epochs"] > policy["max_epochs"]:
        raise ValueError("source policy min_epochs exceeds max_epochs")
    torch.manual_seed(seed)
    model.to(device)
    train_loader = _dataloader(train_dataset, batch_size=int(policy["batch_size"]), shuffle=True, seed=seed)
    validation_loader = _dataloader(validation_dataset, batch_size=int(policy["batch_size"]), shuffle=False, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(policy["learning_rate"]), weight_decay=float(policy["weight_decay"]))
    scaler = _scaler(device)
    best_state: dict[str, Any] | None = None
    best, best_epoch, stale = float("inf"), 0, 0
    accumulation = int(policy["gradient_accumulation"])
    for epoch in range(1, int(policy["max_epochs"]) + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        for step, (gene, drug, target, _, _) in enumerate(train_loader, start=1):
            with _amp_context(device):
                loss = functional.mse_loss(model(gene.to(device), drug.to(device)).reshape(-1), target.to(device)) / accumulation
            scaler.scale(loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(policy["gradient_clip_norm"]))
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        validation = _mse(model, validation_loader, device=device)
        if validation < best:
            best, best_epoch, stale, best_state = validation, epoch, 0, _clone_state(model)
        else:
            stale += 1
        if epoch >= int(policy["min_epochs"]) and stale >= int(policy["patience"]):
            break
    assert best_state is not None
    model.load_state_dict(best_state, strict=True)
    return FitResult(best_epoch, epoch, best, "validation_mse", dict(policy))


def refit_source_model_full(
    model: Any,
    train_dataset: IndexedPairDataset,
    *,
    fixed_epochs: int,
    device: str,
    seed: int = 20260823,
    policy: Mapping[str, Any] = SOURCE_POLICY,
) -> FitResult:
    """Refit source M2 on every source pair after gate-approved epoch freezing.

    There is deliberately no validation loader, early stopping, PRISM input or
    PDO input in this function.  ``fixed_epochs`` must be selected from the
    completed cold-cell folds before this call.
    """
    import torch
    from torch.nn import functional as functional

    if fixed_epochs < 1:
        raise ValueError("source-final refit needs a positive frozen epoch count")
    torch.manual_seed(seed)
    model.to(device)
    loader = _dataloader(train_dataset, batch_size=int(policy["batch_size"]), shuffle=True, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(policy["learning_rate"]), weight_decay=float(policy["weight_decay"]))
    scaler = _scaler(device)
    accumulation = int(policy["gradient_accumulation"])
    for _epoch in range(1, fixed_epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        for step, (gene, drug, target, _, _) in enumerate(loader, start=1):
            with _amp_context(device):
                loss = functional.mse_loss(model(gene.to(device), drug.to(device)).reshape(-1), target.to(device)) / accumulation
            scaler.scale(loss).backward()
            if step % accumulation == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(policy["gradient_clip_norm"]))
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
    frozen_policy = dict(policy) | {"fixed_epochs_from_outer_fold_median": int(fixed_epochs), "early_stopping": "disabled"}
    return FitResult(int(fixed_epochs), int(fixed_epochs), None, "frozen_outer_fold_median_epoch", frozen_policy)


def fit_target_model(
    model: Any,
    model_id: str,
    train_dataset: IndexedPairDataset,
    validation_dataset: IndexedPairDataset,
    *,
    device: str,
    seed: int = 20260823,
    policy: Mapping[str, Any] = TARGET_POLICY,
) -> FitResult:
    """Fine-tune M3/M4 under the predeclared Huber+rank schedule."""
    import torch
    from torch.nn import functional as functional

    if model_id not in {"M3-adapter", "M3-full", "M4"}:
        raise ValueError("target fitting requires M3-adapter, M3-full or M4")
    torch.manual_seed(seed)
    model.to(device)
    train_loader = _dataloader(train_dataset, batch_size=int(policy["batch_size"]), shuffle=False, seed=seed, target_balanced=True)
    validation_loader = _dataloader(validation_dataset, batch_size=int(policy["batch_size"]), shuffle=False, seed=seed)
    scaler = _scaler(device)
    optimizer: Any | None = None
    phase_name: str | None = None
    best_state: dict[str, Any] | None = None
    best, best_epoch, stale = float("-inf"), 0, 0
    for epoch in range(1, int(policy["max_epochs"]) + 1):
        phase = target_phase(model_id, epoch)
        if optimizer is None or phase.name != phase_name:
            groups = optimizer_parameter_groups(model, model_id, epoch)
            optimizer = torch.optim.AdamW(groups, weight_decay=float(policy["weight_decay"]))
            phase_name = phase.name
        model.train(); optimizer.zero_grad(set_to_none=True)
        for gene, drug, target, group, _ in train_loader:
            gene, drug, target, group = gene.to(device), drug.to(device), target.to(device), group.to(device)
            with _amp_context(device):
                prediction = model(gene, drug).reshape(-1)
                loss = functional.huber_loss(prediction, target) + 0.5 * _rank_loss(prediction, target, group)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(policy["gradient_clip_norm"]))
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        # Selection uses the declared per-drug ranking metric, never test rows.
        val_indices, val_prediction = predict_dataset(model, validation_dataset, device=device, batch_size=int(policy["batch_size"]))
        local = {int(pair): idx for idx, pair in enumerate(validation_dataset.pair_indices.tolist())}
        order = np.asarray([local[int(pair)] for pair in val_indices], dtype=np.int64)
        groups_text = np.asarray([
            f"{validation_dataset.data.pair_cohort_id[pair]}\x1f{validation_dataset.data.pair_drug_idx[pair]}"
            for pair in val_indices
        ], dtype=str)
        validation = macro_cohort_drug_spearman(val_prediction, validation_dataset.targets.cpu().numpy()[order], groups_text)
        if validation > best:
            best, best_epoch, stale, best_state = validation, epoch, 0, _clone_state(model)
        else:
            stale += 1
        if epoch >= int(policy["min_epochs"]) and stale >= int(policy["patience"]):
            break
    if best_state is None or not np.isfinite(best):
        raise RuntimeError("target validation has no finite cohort--drug Spearman; inspect support contract")
    model.load_state_dict(best_state, strict=True)
    return FitResult(best_epoch, epoch, best, "validation_macro_cohort_drug_spearman", dict(policy))
