"""Aggregate EXP-038 held-out cell reversal-ranking scores."""

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


MODELS = ("conditional", "state_ablated")
SEEDS = (20260922, 20260923, 20260924)
BASELINES = ("train_drug_mean", "zero")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def complete_seed_mean(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    def mean_all_seeds(values: pd.Series) -> float:
        array = values.to_numpy(dtype=np.float64)
        return float(array.mean()) if len(array) == len(SEEDS) and np.isfinite(array).all() else float("nan")

    return (frame.groupby(["fold", "cell_id", "arm"], sort=True, as_index=False)
            .agg(**{metric: (metric, mean_all_seeds)},
                 n_drugs=("n_drugs", "first"),
                 n_seeds=("seed", "nunique")))


def aggregate(scoring_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"EXP-038 aggregate output already exists: {output_dir}")
    summary_path = scoring_dir / "run_summary.json"
    verification_path = scoring_dir / "independent_verification.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if (summary.get("experiment_id") != "EXP-038"
            or summary.get("status") != "B2_SCORED_PENDING_INDEPENDENT_REVIEW"
            or summary.get("signature") != "gse74602_crc_signature_v2"
            or verification.get("status") != "PASS"
            or verification.get("signature") != summary.get("signature")
            or verification.get("run_summary_sha256") != sha256(summary_path)):
        raise ValueError("EXP-038 reversal scores have not passed independent verification")
    for name, digest in summary["files_sha256"].items():
        if sha256(scoring_dir / name) != digest:
            raise ValueError(f"EXP-038 scoring output changed after its receipt: {name}")

    cells = pd.read_parquet(scoring_dir / "cell_scores.parquet")
    drugs = pd.read_parquet(scoring_dir / "drug_scores.parquet")
    required_arms = set((*MODELS, *BASELINES))
    if (cells.duplicated(["fold", "arm", "seed", "cell_id"]).any()
            or drugs.duplicated(["fold", "arm", "seed", "cell_id", "pert_id"]).any()
            or set(cells["arm"]) != required_arms
            or set(drugs["arm"]) != required_arms):
        raise ValueError("EXP-038 scoring tables have duplicate or unexpected axes")
    folds = {int(row["fold"]): row for row in summary["folds"]}
    if set(folds) != set(range(5)):
        raise ValueError("EXP-038 scoring receipt does not cover all five outer folds")
    for fold in range(5):
        part = cells.loc[cells["fold"].eq(fold)]
        model_cells = set(part.loc[part["arm"].eq(MODELS[0]), "cell_id"])
        if len(model_cells) == 0 or set(part.loc[part["arm"].eq(MODELS[1]), "cell_id"]) != model_cells:
            raise ValueError(f"EXP-038 fold {fold} model arms do not share a cell panel")
        for arm in MODELS:
            subset = part.loc[part["arm"].eq(arm)]
            if len(subset) != len(model_cells) * len(SEEDS) or set(subset["seed"]) != set(SEEDS):
                raise ValueError(f"EXP-038 fold {fold} model seed axis is incomplete")
        for arm in BASELINES:
            subset = part.loc[part["arm"].eq(arm)]
            if len(subset) != len(model_cells) or set(subset["seed"]) != {0}:
                raise ValueError(f"EXP-038 fold {fold} baseline axis is incomplete")
        if folds[fold]["n_cells"] != len(model_cells):
            raise ValueError(f"EXP-038 fold {fold} differs from its score receipt")
        drug_count = drugs.loc[drugs["fold"].eq(fold)].groupby(
            ["cell_id", "arm", "seed"])["pert_id"].nunique()
        if drug_count.empty or drug_count.min() < 5:
            raise ValueError(f"EXP-038 fold {fold} has a cell below the fixed five-drug gate")

    model_scores = complete_seed_mean(cells.loc[cells["arm"].isin(MODELS)], "spearman")
    model_pivot = model_scores.pivot(index=["fold", "cell_id"], columns="arm", values="spearman")
    baseline_scores = cells.loc[cells["arm"].isin(BASELINES), ["fold", "cell_id", "arm", "spearman"]]
    baseline_pivot = baseline_scores.pivot(index=["fold", "cell_id"], columns="arm", values="spearman")
    paired = model_pivot.join(baseline_pivot, how="outer", validate="one_to_one").reset_index()
    if paired["cell_id"].nunique() != summary["n_cells"]:
        raise ValueError("five folds do not cover the declared held-out cell set")
    paired["conditional_vs_measured_zero"] = paired["conditional"]
    paired["conditional_minus_train_drug_mean"] = (
        paired["conditional"] - paired["train_drug_mean"])
    paired["conditional_minus_state_ablated"] = (
        paired["conditional"] - paired["state_ablated"])
    paired["primary_valid"] = np.isfinite(paired["conditional"])
    primary_cells = paired.loc[paired["primary_valid"]]
    if len(primary_cells) < 3:
        raise ValueError("fewer than three held-out cells have a finite primary B2 score")
    primary = paired_cell_statistics(
        primary_cells["conditional_vs_measured_zero"].to_numpy(), seed=20260922)
    primary.update({
        "comparison": "conditional-model reversal-ranking Spearman versus zero",
        "higher_is_better": True,
        "n_fixed_panel_cells": int(paired["cell_id"].nunique()),
        "n_cells_missing_finite_scores": int((~paired["primary_valid"]).sum()),
    })
    secondary = {}
    for name, metric in (
        ("conditional_vs_train_drug_mean", "conditional_minus_train_drug_mean"),
        ("conditional_vs_state_ablated", "conditional_minus_state_ablated"),
    ):
        valid = np.isfinite(paired[metric].to_numpy(float))
        if valid.sum() >= 3:
            secondary[name] = paired_cell_statistics(
                paired.loc[valid, metric].to_numpy(), seed=20260923)
            secondary[name]["n_cells_missing_finite_scores"] = int((~valid).sum())
        else:
            secondary[name] = {"status": "NOT_ESTIMABLE",
                               "n_cells_with_paired_finite_scores": int(valid.sum())}

    output_dir.mkdir(parents=True)
    paired_path = output_dir / "paired_cell_scores.parquet"
    paired.to_parquet(paired_path, index=False)
    report = {
        "experiment_id": "EXP-038",
        "status": "B2_AGGREGATED_PENDING_ROUTE_B_HOLM",
        "signature": summary["signature"],
        "primary_metric": "within-cell Spearman between predicted and measured reversal drug scores",
        "seed_aggregation": "average three finite model seed scores within the same held-out cell",
        "primary_statistics": primary,
        "secondary_statistics": secondary,
        "n_cells": int(paired["cell_id"].nunique()),
        "fold_score_receipts": summary["folds"],
        "source_sha256": {
            "scoring_summary": sha256(summary_path),
            "independent_verification": sha256(verification_path),
            "drug_scores": sha256(scoring_dir / "drug_scores.parquet"),
            "cell_scores": sha256(scoring_dir / "cell_scores.parquet"),
        },
        "files_sha256": {paired_path.name: sha256(paired_path)},
        "route_b_holm_family_status": "PENDING_B1_AND_B5",
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
