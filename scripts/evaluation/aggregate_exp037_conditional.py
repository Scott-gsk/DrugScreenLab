"""Aggregate five frozen outer folds of EXP-037 after every fold completes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (20260922, 20260923, 20260924)
ARMS = ("conditional", "state_ablated")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_gain(values: np.ndarray, *, seed: int) -> dict:
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("paired cell gain is empty or non-finite")
    rng = np.random.default_rng(seed)
    boot = np.array([values[rng.integers(0, len(values), len(values))].mean()
                     for _ in range(2000)])
    rng = np.random.default_rng(seed)
    observed = abs(float(values.mean()))
    extreme = sum(abs(float(np.mean(values * rng.choice([-1.0, 1.0], len(values))))) >= observed
                  for _ in range(10000))
    return {"estimate": float(values.mean()),
            "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "p_two_sided_uncorrected": (extreme + 1) / 10001,
            "n_cells": len(values), "bootstrap_replicates": 2000,
            "sign_flip_replicates": 10000, "unit": "held-out cell"}


def aggregate(folds_dir: Path, baseline_dir: Path, split_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"aggregate output exists: {output_dir}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    baseline = pd.read_parquet(baseline_dir / "cell_scores.parquet")
    baseline = baseline.loc[baseline["model"].eq("train_drug_mean")].copy()
    if baseline.duplicated(["fold", "cell_id"]).any():
        raise ValueError("duplicate baseline fold/cell score")
    frames, receipts = [], []
    for fold in range(5):
        fold_dir = folds_dir / f"fold_{fold}"
        summary = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
        if (summary["status"] != "FOLD_COMPLETE_PENDING_CROSS_FOLD_REVIEW"
                or summary["fold"] != fold or summary["split_sha256"] != sha256(split_path)
                or summary["seeds"] != list(SEEDS)):
            raise ValueError(f"fold {fold} receipt disagrees with frozen protocol")
        scores = pd.read_parquet(fold_dir / "cell_scores.parquet")
        expected_cells = set(split["folds"][fold]["outer_test_ids"])
        if (scores.duplicated(["fold", "arm", "seed", "cell_id"]).any()
                or set(scores["cell_id"]) != expected_cells
                or set(scores["arm"]) != set(ARMS)
                or set(scores["seed"]) != set(SEEDS)
                or not scores["fold"].eq(fold).all()
                or len(scores) != len(expected_cells) * len(ARMS) * len(SEEDS)
                or not np.isfinite(scores["normalized_mse"]).all()):
            raise ValueError(f"fold {fold} cell-score axis or values invalid")
        denominators = scores.groupby(["cell_id", "arm"])[["n_drugs", "n_conditions"]].nunique()
        if denominators.gt(1).any().any():
            raise ValueError(f"fold {fold} changes cell denominator across seeds")
        frames.append(scores)
        receipts.append({"fold": fold, "summary_sha256": sha256(fold_dir / "fold_summary.json"),
                         "cell_scores_sha256": sha256(fold_dir / "cell_scores.parquet"),
                         "n_test_conditions": summary["n_test_conditions"]})
    all_scores = pd.concat(frames, ignore_index=True)
    if all_scores["cell_id"].nunique() != 70:
        raise ValueError("five folds do not cover all 70 cells")
    mean_seed = all_scores.groupby(["fold", "cell_id", "arm"], sort=True, as_index=False).agg(
        normalized_mse=("normalized_mse", "mean"),
        n_drugs=("n_drugs", "first"), n_conditions=("n_conditions", "first"))
    pivot = mean_seed.pivot(index=["fold", "cell_id"], columns="arm", values="normalized_mse")
    reference = baseline.set_index(["fold", "cell_id"])["normalized_mse"].rename("train_drug_mean")
    paired = pivot.join(reference, how="inner", validate="one_to_one")
    if len(paired) != 70 or paired.isna().any().any():
        raise ValueError("baseline and conditional scores do not align on all 70 cells")
    if (mean_seed.merge(baseline[["fold", "cell_id", "n_drugs", "n_conditions"]],
                        on=["fold", "cell_id"], suffixes=("_model", "_baseline"),
                        validate="many_to_one")
            .eval("n_drugs_model != n_drugs_baseline or n_conditions_model != n_conditions_baseline")
            .any()):
        raise ValueError("model and baseline use different condition/drug denominators")
    paired = paired.reset_index()
    paired["gain_vs_train_drug_mean"] = paired["train_drug_mean"] - paired["conditional"]
    paired["gain_vs_state_ablated"] = paired["state_ablated"] - paired["conditional"]
    statistics = {
        "conditional_vs_train_drug_mean": summarize_gain(paired["gain_vs_train_drug_mean"].to_numpy(), seed=20260922),
        "conditional_vs_state_ablated": summarize_gain(paired["gain_vs_state_ablated"].to_numpy(), seed=20260923),
    }
    output_dir.mkdir(parents=True)
    mean_seed.to_parquet(output_dir / "cell_scores_seed_averaged.parquet", index=False)
    paired.to_parquet(output_dir / "paired_cell_scores.parquet", index=False)
    report = {"experiment_id": "EXP-037", "status": "AGGREGATED_PENDING_INDEPENDENT_PREDICTION_RECALC",
              "metric": "training-gene-scale normalized MSE; conditions within drug, drugs within cell, cells equal",
              "seed_aggregation": "compute each seed/cell score, then average scores for same cell",
              "n_cells": len(paired), "n_conditions": sum(x["n_test_conditions"] for x in receipts),
              "mean_cell_nmse": {column: float(paired[column].mean()) for column in
                                 ("conditional", "state_ablated", "train_drug_mean")},
              "statistics": statistics, "route_B_holm_family_status": "PENDING_B2_AND_B5",
              "split_sha256": sha256(split_path),
              "baseline_cell_scores_sha256": sha256(baseline_dir / "cell_scores.parquet"),
              "fold_receipts": receipts,
              "files_sha256": {name: sha256(output_dir / name) for name in
                               ("cell_scores_seed_averaged.parquet", "paired_cell_scores.parquet")}}
    (output_dir / "run_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.folds_dir, args.baseline_dir, args.split, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
