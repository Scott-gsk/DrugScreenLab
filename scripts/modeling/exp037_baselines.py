"""EXP-037：未见细胞外五折的零、全局均值和训练侧逐药均值基线。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from drug_screen.evaluation.perturbation import (
    condition_normalized_mse, equal_weight_cell_score, fit_gene_scale,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sign_flip_p(difference: np.ndarray, *, n: int, seed: int) -> float:
    difference = np.asarray(difference, dtype=np.float64)
    observed = abs(float(difference.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n):
        statistic = abs(float(np.mean(difference * rng.choice([-1.0, 1.0], len(difference)))))
        extreme += statistic >= observed
    return float((extreme + 1) / (n + 1))


def bootstrap_gain(difference: np.ndarray, *, n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    samples = np.empty(n)
    for index in range(n):
        samples[index] = difference[rng.integers(0, len(difference), len(difference))].mean()
    return {"estimate": float(difference.mean()), "low": float(np.quantile(samples, .025)),
            "high": float(np.quantile(samples, .975)), "n": n, "unit": "cell"}


def run(condition_dir: Path, split_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"run output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    keys = pd.read_parquet(condition_dir / "conditions.parquet")
    delta = np.load(condition_dir / "condition_delta.npy", mmap_mode="r")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if len(keys) != len(delta) or keys["condition_id"].tolist() != list(range(len(keys))):
        raise ValueError("condition keys and matrix axis disagree")
    all_condition_scores, all_cell_scores = [], []
    fold_reports = []
    for fold_index, fold in enumerate(split["folds"]):
        train_mask = keys["cell_id"].isin(fold["outer_train_ids"]).to_numpy()
        test_mask = keys["cell_id"].isin(fold["outer_test_ids"]).to_numpy()
        train_index = np.flatnonzero(train_mask)
        test_index_all = np.flatnonzero(test_mask)
        seen_drugs = set(keys.loc[train_mask, "pert_id"])
        test_index = test_index_all[keys.iloc[test_index_all]["pert_id"].isin(seen_drugs).to_numpy()]
        train_values = np.asarray(delta[train_index], dtype=np.float32)
        gene_scale = fit_gene_scale(train_values)
        global_mean = train_values.mean(axis=0, dtype=np.float64).astype(np.float32)
        train_keys = keys.iloc[train_index].reset_index(drop=True)
        grouped = train_keys.groupby("pert_id", sort=True).indices
        drug_ids = sorted(grouped)
        drug_position = {drug: i for i, drug in enumerate(drug_ids)}
        drug_mean = np.empty((len(drug_ids), delta.shape[1]), dtype=np.float32)
        for drug, positions in grouped.items():
            drug_mean[drug_position[drug]] = train_values[np.asarray(positions)].mean(axis=0, dtype=np.float64)
        fold_dir = output_dir / f"fold_{fold_index}"
        fold_dir.mkdir()
        np.save(fold_dir / "gene_mean.npy", gene_scale.mean.astype(np.float32))
        np.save(fold_dir / "gene_scale.npy", gene_scale.scale.astype(np.float32))
        np.save(fold_dir / "global_mean.npy", global_mean)
        np.save(fold_dir / "drug_mean.npy", drug_mean)
        (fold_dir / "drug_ids.json").write_text(json.dumps(drug_ids) + "\n", encoding="utf-8")
        test_keys = keys.iloc[test_index].reset_index(drop=True)
        observed = np.asarray(delta[test_index], dtype=np.float32)
        predictions = {
            "zero": np.zeros_like(observed),
            "global_mean": np.broadcast_to(global_mean, observed.shape),
            "train_drug_mean": drug_mean[np.array([drug_position[x] for x in test_keys["pert_id"]])],
        }
        for model, prediction in predictions.items():
            score = condition_normalized_mse(observed, prediction, gene_scale.scale)
            frame = test_keys[["condition_id", "cell_id", "pert_id"]].copy()
            frame.insert(0, "fold", fold_index)
            frame.insert(1, "model", model)
            frame["normalized_mse"] = score
            all_condition_scores.append(frame)
            cell, total = equal_weight_cell_score(test_keys, score)
            cell.insert(0, "fold", fold_index)
            cell.insert(1, "model", model)
            all_cell_scores.append(cell)
            fold_reports.append({"fold": fold_index, "model": model,
                                 "normalized_mse": total, "n_cells": len(cell),
                                 "n_conditions": len(test_keys),
                                 "n_drugs": int(test_keys["pert_id"].nunique())})
        del train_values, observed, drug_mean
        print(json.dumps({"fold": fold_index, "n_test_conditions": len(test_keys)}), flush=True)
    condition_scores = pd.concat(all_condition_scores, ignore_index=True)
    cell_scores = pd.concat(all_cell_scores, ignore_index=True)
    condition_path = output_dir / "condition_scores.parquet"
    cell_path = output_dir / "cell_scores.parquet"
    condition_scores.to_parquet(condition_path, index=False)
    cell_scores.to_parquet(cell_path, index=False)
    statistics = []
    for baseline in ("zero", "global_mean"):
        pivot = cell_scores.loc[cell_scores["model"].isin(["train_drug_mean", baseline])].pivot(
            index=["fold", "cell_id"], columns="model", values="normalized_mse").dropna()
        # Positive is improvement because lower nMSE is better.
        difference = (pivot[baseline] - pivot["train_drug_mean"]).to_numpy()
        statistics.append({"comparison": f"train_drug_mean_vs_{baseline}",
                           "bootstrap": bootstrap_gain(difference, n=2000, seed=20260922),
                           "permutation_p_two_sided": sign_flip_p(difference, n=10000, seed=20260922),
                           "n_cells": len(difference)})
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        git_status = subprocess.check_output(["git", "status", "--short"], text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_commit, git_status = "unavailable", ["unavailable"]
    report = {
        "experiment_id": "EXP-037", "run_id": output_dir.name,
        "status": "BASELINES_COMPLETE_PENDING_REVIEW",
        "data": {"condition_manifest_sha256": sha256(condition_dir / "manifest.json"),
                 "condition_matrix_sha256": sha256(condition_dir / "condition_delta.npy"),
                 "split_sha256": sha256(split_path)},
        "metric": "training-gene-scale normalized MSE; conditions within drug, drugs within cell, cells equal",
        "main_test_policy": "outer held-out cells and drugs present in corresponding outer training cells",
        "fold_results": fold_reports, "paired_statistics": statistics,
        "git_commit": git_commit, "git_status_at_run": git_status,
        "files_sha256": {"condition_scores.parquet": sha256(condition_path),
                         "cell_scores.parquet": sha256(cell_path)},
    }
    temporary = output_dir / "run_summary.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_dir / "run_summary.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.condition_dir, args.split, args.output_dir)
    print(json.dumps({"status": result["status"], "fold_results": result["fold_results"],
                      "paired_statistics": result["paired_statistics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
