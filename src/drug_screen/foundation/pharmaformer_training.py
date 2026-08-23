"""Small, explicit training loop for EXP-010 PharmaFormer M2/M3/M4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FitResult:
    best_epoch: int
    best_validation_mse: float
    epochs_run: int


def training_policy(variant: str, domain: str) -> str:
    """Return the only permitted optimization action for an EXP-010 stage."""
    policies = {("M2", "source_gdsc"): "fit", ("M2", "target_mcrc"): "frozen_inference",
                ("M3", "target_mcrc"): "finetune", ("M4", "target_mcrc"): "fit_random_init"}
    try:
        return policies[(variant, domain)]
    except KeyError as error:
        raise ValueError(f"forbidden variant/domain combination: {variant}/{domain}") from error


def fit_model(model: Any, train_gene: np.ndarray, train_drug: np.ndarray, train_y: np.ndarray,
              validation_gene: np.ndarray, validation_drug: np.ndarray, validation_y: np.ndarray,
              *, device: str, epochs: int, batch_size: int, learning_rate: float = 1e-5,
              patience: int = 5, seed: int = 20260823) -> FitResult:
    """Fit only supplied train arrays and select epochs only on supplied validation."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    def dataset(gene: np.ndarray, drug: np.ndarray, y: np.ndarray) -> TensorDataset:
        if len(gene) != len(drug) or len(gene) != len(y):
            raise ValueError("gene, drug and response arrays must align")
        return TensorDataset(torch.as_tensor(gene, dtype=torch.float32),
                             torch.as_tensor(drug, dtype=torch.float32),
                             torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1))
    train_loader = DataLoader(dataset(train_gene, train_drug, train_y), batch_size=batch_size,
                              shuffle=True, generator=generator)
    validation_loader = DataLoader(dataset(validation_gene, validation_drug, validation_y),
                                   batch_size=batch_size, shuffle=False)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for gene, drug, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(gene.to(device), drug.to(device)), target.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for gene, drug, target in validation_loader:
                batch_loss = criterion(model(gene.to(device), drug.to(device)), target.to(device))
                total += float(batch_loss) * len(gene)
                count += len(gene)
        validation_loss = total / count
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return FitResult(best_epoch, best_loss, epoch)


def predict(model: Any, gene: np.ndarray, drug: np.ndarray, *, device: str, batch_size: int = 16) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    if len(gene) != len(drug):
        raise ValueError("gene and drug arrays must align")
    loader = DataLoader(TensorDataset(torch.as_tensor(gene, dtype=torch.float32),
                                     torch.as_tensor(drug, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=False)
    model.to(device).eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for gene_batch, drug_batch in loader:
            output.append(model(gene_batch.to(device), drug_batch.to(device)).cpu().numpy().reshape(-1))
    return np.concatenate(output) if output else np.empty(0, dtype=np.float32)
