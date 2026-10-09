from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/modeling"))
import run_exp058_state_conditional_residual as exp058  # noqa: E402


def test_object_block_shuffle_is_role_local_derangement() -> None:
    train_ids = [f"tr{i}" for i in range(8)]
    dev_ids = [f"dv{i}" for i in range(5)]
    train_map = exp058.derangement(train_ids, 58059)
    dev_map = exp058.derangement(dev_ids, 58060)
    assert set(train_map) == set(train_map.values()) == set(train_ids)
    assert set(dev_map) == set(dev_map.values()) == set(dev_ids)
    assert all(key != value for key, value in train_map.items())
    assert all(key != value for key, value in dev_map.items())
    assert not set(train_map) & set(dev_map.values())


def test_capacity_matched_control_has_at_least_candidate_parameters() -> None:
    candidate, _, _ = exp058.build_model(torch, "nonlinear_state_residual", 5, 16, 58058)
    control, _, _ = exp058.build_model(torch, "capacity_matched_drug_only", 5, 16, 58060)
    assert sum(p.numel() for p in control.parameters()) >= sum(p.numel() for p in candidate.parameters())


def test_state_pca_fits_train_objects_and_transforms_disjoint_dev() -> None:
    train_keys = pd.DataFrame({"source_object_id": [f"tr{i}" for i in range(6) for _ in range(2)]})
    dev_keys = pd.DataFrame({"source_object_id": [f"dv{i}" for i in range(4) for _ in range(2)]})
    rng = np.random.default_rng(58)
    train_ctl = rng.normal(size=(12, 978))
    dev_ctl = rng.normal(loc=2.0, size=(8, 978))
    train_x, dev_x, receipt = exp058.make_state_features(
        train_keys, dev_keys, train_ctl, dev_ctl, n_components=3, shuffle_seed=None)
    assert train_x.shape == (12, 3)
    assert dev_x.shape == (8, 3)
    assert receipt["fit_role"] == "TRAIN source_object_id object-mean X_ctl only"
    assert receipt["train_inner_dev_object_overlap"] == 0


def test_one_finite_training_epoch_keeps_candidate_outputs_finite() -> None:
    model, _, _ = exp058.build_model(torch, "nonlinear_state_residual", 5, 16, 58058)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    rng = np.random.default_rng(59)
    features = torch.as_tensor(rng.integers(0, 2, size=(8, 2048)).astype(np.float32))
    drugs = torch.as_tensor(np.tile(np.arange(4), 2), dtype=torch.long)
    state = torch.as_tensor(rng.normal(size=(8, 16)).astype(np.float32))
    target = torch.as_tensor(rng.normal(size=(8, 978)).astype(np.float32))
    weights = torch.full((8,), 1.0 / 8, dtype=torch.float32)
    variance = torch.ones((978,), dtype=torch.float32)
    loss, receipt = exp058.train_epoch(torch, "nonlinear_state_residual", model, optimizer,
        fp=features, drug=drugs, state=state, residual_target=target, row_weights=weights,
        gene_variance=variance, batch_size=4, seed=58061)
    assert np.isfinite(loss)
    assert receipt == {"batches": 2, "finite": True}


def test_frozen_exp053_ridge_comparator_interface_is_callable() -> None:
    train_keys = pd.DataFrame({
        "source_object_id": ["a", "b", "c", "d"] * 2,
        "full_inchikey": ["drug-a"] * 4 + ["drug-b"] * 4,
    })
    dev_keys = pd.DataFrame({
        "source_object_id": ["e", "f", "g", "h"],
        "full_inchikey": ["drug-a", "drug-a", "drug-b", "drug-b"],
    })
    rng = np.random.default_rng(58058)
    train_y = rng.normal(size=(8, 4))
    means = {drug: train_y[i * 4:(i + 1) * 4].mean(axis=0)
             for i, drug in enumerate(("drug-a", "drug-b"))}
    train_base = np.stack([means[d] for d in train_keys.full_inchikey])
    dev_base = np.stack([means[d] for d in dev_keys.full_inchikey])
    states = {obj: rng.normal(size=3) for obj in ("a", "b", "c", "d", "e", "f", "g", "h")}
    prediction, receipt = exp058.exp053.fit_state_residual(
        train_keys, train_y, dev_keys, train_base, dev_base, states, None, alpha=1.0)
    assert prediction.shape == (4, 4)
    assert np.isfinite(prediction).all()
    assert receipt["fit_objects"] == 4
