"""从冻结划分与响应合同独立复算 EXP-036 均值基线。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(contract_dir: Path, split_path: Path, run_dir: Path) -> dict:
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    source = pd.read_parquet(contract_dir / "source_pairs.parquet")
    target = pd.read_parquet(contract_dir / "target_pairs.parquet")
    samples = pd.read_parquet(contract_dir / "target_samples.parquet")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    source_pred = pd.read_parquet(run_dir / "source_predictions.parquet")
    target_pred = pd.read_parquet(run_dir / "target_predictions.parquet")
    scores = pd.read_parquet(run_dir / "unit_scores.parquet")
    for name in ("source_predictions.parquet", "target_predictions.parquet", "unit_scores.parquet"):
        if sha256(run_dir / name) != summary["files_sha256"][name]:
            raise ValueError(f"checksum mismatch: {name}")
    target = target.merge(samples[["sample_id", "patient_id"]], on="sample_id", validate="many_to_one")
    target = target.groupby(["patient_id", "drug_id"], as_index=False)["response"].mean()
    max_prediction_error = 0.0
    max_score_error = 0.0
    for fold_index in range(5):
        source_fold = split["source_folds"][fold_index]
        source_mean = source.loc[source["sample_id"].isin(source_fold["outer_train_ids"])] \
            .groupby("drug_id")["response"].mean()
        saved_source = source_pred.loc[source_pred["fold"].eq(fold_index)]
        expected_ids = source.loc[source["sample_id"].isin(source_fold["outer_test_ids"]),
                                  ["sample_id", "drug_id"]]
        if set(map(tuple, expected_ids.to_numpy())) != set(map(tuple, saved_source[["sample_id", "drug_id"]].to_numpy())):
            raise ValueError("source test keys differ from frozen split")
        error = float(np.max(np.abs(saved_source["predicted"] - saved_source["drug_id"].map(source_mean))))
        max_prediction_error = max(max_prediction_error, error)
        target_fold = split["target_folds"][fold_index]
        test_patients = set(target_fold["outer_test_ids"])
        expected_test = target.loc[target["patient_id"].isin(test_patients)]
        saved_source_target = target_pred.loc[
            target_pred["fold"].eq(fold_index) & target_pred["fraction"].eq(0.0)]
        error = float(np.max(np.abs(saved_source_target["predicted"]
                                    - saved_source_target["drug_id"].map(source_mean))))
        max_prediction_error = max(max_prediction_error, error)
        for item in target_fold["label_fractions"]:
            fraction = float(item["fraction"])
            target_mean = target.loc[target["patient_id"].isin(item["train_ids"])] \
                .groupby("drug_id")["response"].mean()
            saved = target_pred.loc[target_pred["fold"].eq(fold_index)
                                    & target_pred["fraction"].eq(fraction)
                                    & target_pred["model"].eq("target_train_drug_mean")]
            expected_keys = expected_test.loc[expected_test["drug_id"].isin(target_mean.index),
                                              ["patient_id", "drug_id"]]
            if set(map(tuple, saved[["patient_id", "drug_id"]].to_numpy())) != set(map(tuple, expected_keys.to_numpy())):
                raise ValueError("target test keys differ from frozen split/availability")
            error = float(np.max(np.abs(saved["predicted"] - saved["drug_id"].map(target_mean))))
            max_prediction_error = max(max_prediction_error, error)
    for row in scores.itertuples(index=False):
        if row.model == "source_train_drug_mean" and pd.isna(row.fraction):
            frame = source_pred.loc[(source_pred["fold"] == row.fold)
                                    & (source_pred["sample_id"] == row.unit_id)]
        else:
            frame = target_pred.loc[(target_pred["fold"] == row.fold)
                                    & (target_pred["fraction"] == row.fraction)
                                    & (target_pred["model"] == row.model)
                                    & (target_pred["patient_id"] == row.unit_id)]
        expected = float(spearmanr(frame["observed"], frame["predicted"]).statistic)
        error = abs(expected - row.spearman)
        max_score_error = max(max_score_error, error)
        if error > 1e-12:
            raise ValueError(f"unit score mismatch: {row.unit_id}")
    if max_prediction_error > 1e-12:
        raise ValueError(f"prediction mismatch: {max_prediction_error}")
    return {"status": "PASS", "verified_source_predictions": len(source_pred),
            "verified_target_predictions": len(target_pred), "verified_unit_scores": len(scores),
            "max_prediction_error": max_prediction_error, "max_score_error": max_score_error}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.contract_dir, args.split, args.run_dir)
    (args.run_dir / "independent_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
