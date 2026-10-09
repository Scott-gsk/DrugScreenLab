"""Aggregate verified EXP-039 reversal scores without selecting a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .paired_cell_statistics import paired_cell_statistics
except ImportError:
    from paired_cell_statistics import paired_cell_statistics


ARMS = ("huber", "b5_rank")
SEEDS = (20260922, 20260923, 20260924)
PRIMARY = "spearman"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _complete_seed_mean(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    def average_if_complete(values: pd.Series) -> float:
        array = values.to_numpy(dtype=np.float64)
        return float(array.mean()) if len(array) == len(SEEDS) and np.isfinite(array).all() else float("nan")

    return (frame.groupby(["fold", "cell_id", "arm"], sort=True, as_index=False)
            .agg(**{value: (value, average_if_complete)},
                 n_drugs=("n_drugs", "first"),
                 n_seeds=("seed", "nunique")))


def aggregate(scoring_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"EXP-039 aggregate output exists: {output_dir}")
    summary_path = scoring_dir / "run_summary.json"
    verification_path = scoring_dir / "independent_verification.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if (summary.get("experiment_id") != "EXP-039"
            or summary.get("status") != "B5_REVERSAL_SCORED_PENDING_INDEPENDENT_REVIEW"
            or verification.get("status") != "PASS"
            or verification.get("run_summary_sha256") != sha256(summary_path)):
        raise ValueError("EXP-039 reversal score has not passed its independent verification")
    for name, digest in summary["files_sha256"].items():
        if sha256(scoring_dir / name) != digest:
            raise ValueError(f"scored file changed after its receipt: {name}")

    cells = pd.read_parquet(scoring_dir / "cell_scores.parquet")
    drugs = pd.read_parquet(scoring_dir / "drug_scores.parquet")
    if (cells.duplicated(["fold", "arm", "seed", "cell_id"]).any()
            or drugs.duplicated(["fold", "arm", "seed", "cell_id", "pert_id"]).any()
            or set(cells["arm"]) != set(ARMS)
            or set(cells["seed"]) != set(SEEDS)
            or set(cells["signature"]) != {summary["signature"]}
            or not drugs["signature"].eq(summary["signature"]).all()):
        raise ValueError("EXP-039 scored axes contain missing, duplicate, or foreign rows")
    cell_counts = cells.groupby(["fold", "arm"])["cell_id"].nunique()
    if set(cell_counts.index.get_level_values("fold")) != set(range(5)):
        raise ValueError("EXP-039 score does not cover all five held-out folds")
    for fold in range(5):
        fold_cells = cells.loc[cells["fold"].eq(fold)]
        arm_sets = {arm: set(fold_cells.loc[fold_cells["arm"].eq(arm), "cell_id"]) for arm in ARMS}
        if arm_sets[ARMS[0]] != arm_sets[ARMS[1]]:
            raise ValueError(f"EXP-039 fold {fold} arms use different cell panels")
        for arm in ARMS:
            subset = fold_cells.loc[fold_cells["arm"].eq(arm)]
            if len(subset) != len(arm_sets[arm]) * len(SEEDS) or set(subset["seed"]) != set(SEEDS):
                raise ValueError(f"EXP-039 fold {fold} has an incomplete seed-by-cell axis")
        denominator = drugs.loc[drugs["fold"].eq(fold)].groupby(
            ["cell_id", "arm", "seed"])["pert_id"].nunique()
        if denominator.empty or denominator.min() < 5:
            raise ValueError(f"EXP-039 fold {fold} contains a cell below the fixed five-drug gate")

    averaged = _complete_seed_mean(cells, PRIMARY)
    pivot = averaged.pivot(index=["fold", "cell_id"], columns="arm", values=PRIMARY)
    paired = pivot.reset_index()
    paired["b5_rank_minus_huber"] = paired["b5_rank"] - paired["huber"]
    paired["paired_valid"] = np.isfinite(paired["b5_rank"]) & np.isfinite(paired["huber"])
    valid = paired.loc[paired["paired_valid"]].copy()
    if len(valid) < 3:
        raise ValueError("fewer than three cells have complete finite scores in both arms")
    primary = paired_cell_statistics(valid["b5_rank_minus_huber"].to_numpy(), seed=20260922)
    primary.update({
        "comparison": "B5 ranking objective minus matched Huber objective",
        "higher_is_better": True,
        "n_fixed_panel_cells": int(paired["cell_id"].nunique()),
        "n_cells_missing_finite_scores": int((~paired["paired_valid"]).sum()),
    })

    secondary = {}
    for metric in ("negative_cosine_spearman", "top_decile_overlap", "measured_selection_loss"):
        averaged_metric = _complete_seed_mean(cells, metric)
        metric_pivot = averaged_metric.pivot(index=["fold", "cell_id"], columns="arm", values=metric)
        candidate = metric_pivot[ARMS[1]].to_numpy(dtype=np.float64)
        baseline = metric_pivot[ARMS[0]].to_numpy(dtype=np.float64)
        valid_metric = np.isfinite(candidate) & np.isfinite(baseline)
        if valid_metric.sum() >= 3:
            difference = (baseline[valid_metric] - candidate[valid_metric]
                          if metric == "measured_selection_loss"
                          else candidate[valid_metric] - baseline[valid_metric])
            secondary[metric] = paired_cell_statistics(difference, seed=20260923)
            secondary[metric]["n_cells_missing_finite_scores"] = int((~valid_metric).sum())
        else:
            secondary[metric] = {"status": "NOT_ESTIMABLE",
                                 "n_cells_with_paired_finite_scores": int(valid_metric.sum())}

    output_dir.mkdir(parents=True)
    averaged_path = output_dir / "cell_scores_seed_averaged.parquet"
    paired_path = output_dir / "paired_cell_scores.parquet"
    averaged.to_parquet(averaged_path, index=False)
    paired.to_parquet(paired_path, index=False)
    report = {
        "experiment_id": "EXP-039",
        "status": "B5_AGGREGATED_PENDING_ROUTE_B_HOLM",
        "signature": summary["signature"],
        "primary_metric": "within-cell Spearman of predicted versus measured reversal drug scores",
        "seed_aggregation": "average three finite seed scores within each held-out cell; incomplete seed groups remain undefined",
        "primary_statistics": primary,
        "secondary_statistics": secondary,
        "n_cells": int(paired["cell_id"].nunique()),
        "n_paired_finite_cells": int(valid["cell_id"].nunique()),
        "fold_score_receipts": summary["folds"],
        "source_sha256": {
            "scoring_summary": sha256(summary_path),
            "independent_verification": sha256(verification_path),
            "cell_scores": sha256(scoring_dir / "cell_scores.parquet"),
            "drug_scores": sha256(scoring_dir / "drug_scores.parquet"),
        },
        "files_sha256": {
            averaged_path.name: sha256(averaged_path),
            paired_path.name: sha256(paired_path),
        },
        "route_b_holm_family_status": "PENDING_B1_AND_B2",
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.scoring_dir, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
