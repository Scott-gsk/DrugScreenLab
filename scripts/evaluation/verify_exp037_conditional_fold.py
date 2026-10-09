"""Independently recompute an EXP-037 fold's cell scores from saved delta predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ("conditional", "state_ablated")
SEEDS = (20260922, 20260923, 20260924)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def independently_score(keys: pd.DataFrame, observed: np.ndarray,
                        predicted: np.ndarray, scale: np.ndarray) -> tuple[pd.DataFrame, float]:
    error = np.square((predicted.astype(np.float64) - observed.astype(np.float64))
                      / scale.astype(np.float64)).mean(axis=1)
    frame = keys[["cell_id", "pert_id", "condition_id"]].copy()
    frame["error"] = error
    drug = frame.groupby(["cell_id", "pert_id"], sort=True, as_index=False).agg(
        drug_error=("error", "mean"), n_conditions=("condition_id", "size"))
    cell = drug.groupby("cell_id", sort=True, as_index=False).agg(
        normalized_mse=("drug_error", "mean"), n_drugs=("pert_id", "size"),
        n_conditions=("n_conditions", "sum"))
    return cell, float(cell["normalized_mse"].mean())


def verify(condition_dir: Path, split_path: Path, fold_dir: Path) -> dict:
    summary = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    fold = int(summary["fold"])
    if (fold not in range(5) or summary["seeds"] != list(SEEDS)
            or summary["split_sha256"] != sha256(split_path)
            or summary["data_manifest_sha256"] != sha256(condition_dir / "manifest.json")):
        raise ValueError("fold receipt does not match the fixed seeds")
    keys = pd.read_parquet(condition_dir / "conditions.parquet")
    test = keys.loc[keys["cell_id"].isin(split["folds"][fold]["outer_test_ids"])].copy()
    train = keys.loc[keys["cell_id"].isin(split["folds"][fold]["outer_train_ids"])]
    test = test.loc[test["pert_id"].isin(set(train["pert_id"]))]
    saved_ids = np.load(fold_dir / "test_condition_ids.npy")
    if not np.array_equal(test["condition_id"].to_numpy(np.int64), saved_ids):
        raise ValueError("saved prediction axis differs from the frozen unseen-cell test panel")
    if len(test) != summary["n_test_conditions"]:
        raise ValueError("test condition denominator differs from receipt")
    gene_scale = np.load(fold_dir / "preprocess.npz")["gene_scale"]
    observed_all = np.load(condition_dir / "condition_delta.npy", mmap_mode="r")
    train_delta = np.asarray(observed_all[train.index.to_numpy()], dtype=np.float32)
    expected_scale = train_delta.std(axis=0, dtype=np.float64).astype(np.float32)
    expected_scale[expected_scale <= 1e-8] = 1.0
    scale_error = float(np.max(np.abs(gene_scale - expected_scale)))
    del train_delta
    if scale_error != 0.0:
        raise ValueError("saved gene scale differs from outer training cells")
    observed = np.asarray(observed_all[test.index.to_numpy()], dtype=np.float32)
    saved_scores = pd.read_parquet(fold_dir / "cell_scores.parquet")
    if saved_scores.duplicated(["fold", "arm", "seed", "cell_id"]).any():
        raise ValueError("duplicate saved cell score")
    max_error = 0.0
    for arm in ARMS:
        for seed in SEEDS:
            predicted = np.load(fold_dir / f"predicted_delta_{arm}_seed{seed}.npy", mmap_mode="r")
            if predicted.shape != observed.shape or not np.isfinite(predicted).all():
                raise ValueError("prediction matrix has invalid axis or non-finite value")
            cell, mean = independently_score(test.reset_index(drop=True), observed,
                                             predicted, gene_scale)
            saved = saved_scores.loc[saved_scores["arm"].eq(arm) & saved_scores["seed"].eq(seed)]
            comparison = saved.merge(cell, on="cell_id", suffixes=("_saved", "_new"),
                                     validate="one_to_one")
            if (len(comparison) != len(cell) or len(saved) != len(cell)
                    or not saved["fold"].eq(fold).all()
                    or not np.array_equal(comparison["n_drugs_saved"], comparison["n_drugs_new"])
                    or not np.array_equal(comparison["n_conditions_saved"], comparison["n_conditions_new"])):
                raise ValueError("saved cell-score axes or denominators disagree")
            difference = float(np.max(np.abs(comparison["normalized_mse_saved"]
                                             - comparison["normalized_mse_new"])))
            max_error = max(max_error, difference)
            if difference > 1e-10:
                raise ValueError("saved cell scores disagree with prediction-based recomputation")
            reported = next(row for row in summary["final_receipts"]
                            if row["arm"] == arm and row["seed"] == seed)
            if abs(mean - reported["test_nmse"]) > 1e-10:
                raise ValueError("fold mean score differs from independently computed cell scores")
    checked_files = ["preprocess.npz", "test_condition_ids.npy", "cell_scores.parquet"] + [
        f"predicted_delta_{arm}_seed{seed}.npy" for arm in ARMS for seed in SEEDS]
    return {"status": "PASS", "fold": fold, "n_conditions": len(test),
            "n_cells": test["cell_id"].nunique(), "n_prediction_matrices": len(ARMS) * len(SEEDS),
            "max_cell_score_error": max_error, "max_train_scale_error": scale_error,
            "files_sha256": {name: sha256(fold_dir / name) for name in checked_files}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(args.condition_dir, args.split, args.fold_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
