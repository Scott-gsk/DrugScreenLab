"""Run one real-data CPU update through the preregistered EXP-039 objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from drug_screen.models.conditional_perturbation import SharedEffectInteraction
from exp039_b5_fold import bidirectional_ranking_loss, load_exp037_helpers


def run(condition_dir: Path, input_dir: Path, contract_dir: Path,
        cache_path: Path, n_conditions: int, seed: int) -> dict:
    helpers = load_exp037_helpers()
    arrays = helpers.Arrays(condition_dir, input_dir, load_to_memory=False)
    if n_conditions < 4 or n_conditions > len(arrays.keys):
        raise ValueError("smoke sample size must be between 4 and the available conditions")
    rng = np.random.default_rng(seed)
    index = np.sort(rng.choice(len(arrays.keys), size=n_conditions, replace=False))
    noise = helpers.control_noise_table(contract_dir, cache_path)
    prep = helpers.fit_preprocess(arrays, index, noise)
    device = torch.device("cpu")
    state, drug, exposure, target, target_raw = arrays.tensors(index, prep, device)
    model = SharedEffectInteraction(rank=16).to(device)
    prediction = model(state, drug, exposure, use_state=True)
    regression_loss = F.huber_loss(prediction, target)
    ranking_loss, counts = bidirectional_ranking_loss(
        target_raw, prediction,
        torch.from_numpy(prep.gene_scale),
        torch.from_numpy(prep.direction_threshold),
    )
    if counts["conditions_with_any_direction"] == 0:
        raise ValueError("smoke sample has no reliable up or down gene ranking")
    loss = regression_loss + 0.1 * ranking_loss
    if not torch.isfinite(loss):
        raise FloatingPointError("combined EXP-039 training loss is non-finite")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("EXP-039 smoke update produced missing or non-finite gradients")
    return {
        "status": "REAL_DATA_TRAINING_STEP_PASS",
        "n_conditions": int(n_conditions),
        "n_cells": int(arrays.keys.iloc[index]["cell_id"].nunique()),
        "regression_loss": float(regression_loss.detach()),
        "ranking_loss": float(ranking_loss.detach()),
        "combined_loss": float(loss.detach()),
        **counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--n-conditions", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    print(json.dumps(run(args.condition_dir, args.input_dir, args.contract_dir,
                         args.cache, args.n_conditions, args.seed), ensure_ascii=False))


if __name__ == "__main__":
    main()
