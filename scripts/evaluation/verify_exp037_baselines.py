"""独立复算 EXP-037 正式基线的训练侧参数、逐条件误差和等权汇总。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from drug_screen.evaluation.perturbation import condition_normalized_mse, equal_weight_cell_score


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(condition_dir: Path, split_path: Path, run_dir: Path) -> dict:
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    keys = pd.read_parquet(condition_dir / "conditions.parquet")
    delta = np.load(condition_dir / "condition_delta.npy", mmap_mode="r")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    condition_scores = pd.read_parquet(run_dir / "condition_scores.parquet")
    cell_scores = pd.read_parquet(run_dir / "cell_scores.parquet")
    for name in ("condition_scores.parquet", "cell_scores.parquet"):
        if sha256(run_dir / name) != summary["files_sha256"][name]:
            raise ValueError(f"run file checksum mismatch: {name}")
    max_parameter_error = 0.0
    max_score_error = 0.0
    max_aggregate_error = 0.0
    for fold_index, fold in enumerate(split["folds"]):
        train_mask = keys["cell_id"].isin(fold["outer_train_ids"]).to_numpy()
        test_mask = keys["cell_id"].isin(fold["outer_test_ids"]).to_numpy()
        seen = set(keys.loc[train_mask, "pert_id"])
        expected_test = keys.loc[test_mask & keys["pert_id"].isin(seen).to_numpy(), "condition_id"]
        saved = condition_scores.loc[condition_scores["fold"].eq(fold_index)]
        for model in ("zero", "global_mean", "train_drug_mean"):
            ids = saved.loc[saved["model"].eq(model), "condition_id"]
            if set(ids) != set(expected_test) or len(ids) != len(expected_test):
                raise ValueError(f"test condition identity mismatch in fold {fold_index}, {model}")
        fold_dir = run_dir / f"fold_{fold_index}"
        scale = np.load(fold_dir / "gene_scale.npy")
        global_mean = np.load(fold_dir / "global_mean.npy")
        drug_ids = json.loads((fold_dir / "drug_ids.json").read_text())
        drug_mean = np.load(fold_dir / "drug_mean.npy", mmap_mode="r")
        drug_position = {drug: i for i, drug in enumerate(drug_ids)}
        train_index = np.flatnonzero(train_mask)
        sample_drugs = [drug_ids[i] for i in np.linspace(
            0, len(drug_ids) - 1, min(10, len(drug_ids)), dtype=int)]
        for drug in sample_drugs:
            positions = train_index[keys.iloc[train_index]["pert_id"].eq(drug).to_numpy()]
            expected = np.asarray(delta[positions], dtype=np.float64).mean(axis=0)
            error = float(np.max(np.abs(expected - drug_mean[drug_position[drug]])))
            max_parameter_error = max(max_parameter_error, error)
            if error > 2e-6:
                raise ValueError(f"training drug mean mismatch: fold {fold_index}, {drug}")
        sample_ids = expected_test.iloc[np.linspace(0, len(expected_test) - 1, 25, dtype=int)].to_numpy()
        observed = np.asarray(delta[sample_ids], dtype=np.float64)
        sample_keys = keys.set_index("condition_id").loc[sample_ids]
        predictions = {
            "zero": np.zeros_like(observed),
            "global_mean": np.broadcast_to(global_mean, observed.shape),
            "train_drug_mean": np.stack([drug_mean[drug_position[x]] for x in sample_keys["pert_id"]]),
        }
        for model, prediction in predictions.items():
            expected_score = condition_normalized_mse(observed, prediction, scale)
            indexed = saved.loc[saved["model"].eq(model)].set_index("condition_id")
            actual_score = indexed.loc[sample_ids, "normalized_mse"].to_numpy()
            error = float(np.max(np.abs(expected_score - actual_score)))
            max_score_error = max(max_score_error, error)
            if error > 2e-6:
                raise ValueError(f"condition score mismatch: fold {fold_index}, {model}")
            full = saved.loc[saved["model"].eq(model)]
            cells, total = equal_weight_cell_score(
                keys.set_index("condition_id").loc[full["condition_id"]].reset_index(),
                full["normalized_mse"].to_numpy())
            actual_cells = cell_scores.loc[
                cell_scores["fold"].eq(fold_index) & cell_scores["model"].eq(model)
            ].set_index("cell_id")
            error = float(np.max(np.abs(cells.set_index("cell_id")["normalized_mse"]
                                        - actual_cells["normalized_mse"])))
            max_aggregate_error = max(max_aggregate_error, error)
            if error > 1e-10:
                raise ValueError(f"cell equal-weight aggregation mismatch: {fold_index}, {model}")
            reported = next(x["normalized_mse"] for x in summary["fold_results"]
                            if x["fold"] == fold_index and x["model"] == model)
            if abs(total - reported) > 1e-10:
                raise ValueError(f"fold summary mismatch: {fold_index}, {model}")
    return {"status": "PASS", "verified_folds": 5, "verified_models": 3,
            "max_training_parameter_error": max_parameter_error,
            "max_condition_score_error": max_score_error,
            "max_aggregation_error": max_aggregate_error}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.condition_dir, args.split, args.run_dir)
    path = args.run_dir / "independent_verification.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
