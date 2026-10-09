"""EXP-036：来源冷细胞及目标患者留出的训练侧逐药均值基线。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from drug_screen.evaluation.metrics import spearman_rho


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def patient_scores(frame: pd.DataFrame, unit: str) -> pd.DataFrame:
    rows = []
    for identity, group in frame.groupby(unit, sort=True):
        if group["drug_id"].nunique() < 3:
            continue
        score = spearman_rho(group["observed"], group["predicted"])
        rows.append({unit: identity, "spearman": score,
                     "n_drugs": int(group["drug_id"].nunique())})
    return pd.DataFrame(rows)


def bootstrap(difference: np.ndarray, seed: int = 20260922) -> dict:
    difference = np.asarray(difference, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.array([difference[rng.integers(0, len(difference), len(difference))].mean()
                      for _ in range(2000)])
    observed = abs(float(difference.mean()))
    permuted = np.array([abs(float(np.mean(difference * rng.choice([-1, 1], len(difference)))))
                         for _ in range(10000)])
    return {"estimate": float(difference.mean()), "low": float(np.quantile(draws, .025)),
            "high": float(np.quantile(draws, .975)),
            "permutation_p_two_sided": float((1 + (permuted >= observed).sum()) / 10001),
            "n_units": len(difference)}


def run(contract_dir: Path, split_path: Path, source_npz: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"run directory exists: {output_dir}")
    output_dir.mkdir(parents=True)
    source_pairs = pd.read_parquet(contract_dir / "source_pairs.parquet")
    target_pairs = pd.read_parquet(contract_dir / "target_pairs.parquet")
    target_samples = pd.read_parquet(contract_dir / "target_samples.parquet")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    with np.load(source_npz, allow_pickle=True) as source:
        source_samples = pd.DataFrame({"sample_id": source["sample_id"].astype(str),
                                       "cell_line_id": source["cell_line_id"].astype(str)})
    target_pairs = target_pairs.merge(
        target_samples[["sample_id", "patient_id"]], on="sample_id", how="left", validate="many_to_one")
    # Multiple organoids from one patient are combined before patient-level ranking.
    target_patient = target_pairs.groupby(["patient_id", "drug_id"], as_index=False).agg(
        response=("response", "mean"), n_samples=("sample_id", "nunique"))
    source_predictions, target_predictions, score_frames = [], [], []
    receipts = []
    for fold_index in range(5):
        source_fold = split["source_folds"][fold_index]
        source_train = source_pairs.loc[source_pairs["sample_id"].isin(source_fold["outer_train_ids"])]
        source_test = source_pairs.loc[source_pairs["sample_id"].isin(source_fold["outer_test_ids"])]
        source_mean = source_train.groupby("drug_id")["response"].mean()
        source_pred = source_test.loc[source_test["drug_id"].isin(source_mean.index),
                                      ["sample_id", "drug_id", "response"]].copy()
        source_pred["predicted"] = source_pred["drug_id"].map(source_mean)
        source_pred = source_pred.rename(columns={"response": "observed"})
        source_pred.insert(0, "fold", fold_index)
        source_pred.insert(1, "model", "source_train_drug_mean")
        source_predictions.append(source_pred)
        source_score = patient_scores(source_pred, "sample_id")
        source_score.insert(0, "fold", fold_index)
        source_score.insert(1, "model", "source_train_drug_mean")
        score_frames.append(source_score.rename(columns={"sample_id": "unit_id"}))

        target_fold = split["target_folds"][fold_index]
        target_test = target_patient.loc[target_patient["patient_id"].isin(target_fold["outer_test_ids"])]
        source_target = target_test.loc[target_test["drug_id"].isin(source_mean.index),
                                        ["patient_id", "drug_id", "response"]].copy()
        source_target["predicted"] = source_target["drug_id"].map(source_mean)
        source_target = source_target.rename(columns={"response": "observed"})
        source_target.insert(0, "fold", fold_index)
        source_target.insert(1, "fraction", 0.0)
        source_target.insert(2, "model", "source_train_drug_mean")
        target_predictions.append(source_target)
        scores = patient_scores(source_target, "patient_id")
        scores.insert(0, "fold", fold_index); scores.insert(1, "fraction", 0.0)
        scores.insert(2, "model", "source_train_drug_mean")
        score_frames.append(scores.rename(columns={"patient_id": "unit_id"}))
        for fraction_data in target_fold["label_fractions"]:
            fraction = float(fraction_data["fraction"])
            train_ids = fraction_data["train_ids"]
            train = target_patient.loc[target_patient["patient_id"].isin(train_ids)]
            target_mean = train.groupby("drug_id")["response"].mean()
            prediction = target_test.loc[target_test["drug_id"].isin(target_mean.index),
                                         ["patient_id", "drug_id", "response"]].copy()
            prediction["predicted"] = prediction["drug_id"].map(target_mean)
            prediction = prediction.rename(columns={"response": "observed"})
            prediction.insert(0, "fold", fold_index); prediction.insert(1, "fraction", fraction)
            prediction.insert(2, "model", "target_train_drug_mean")
            target_predictions.append(prediction)
            scores = patient_scores(prediction, "patient_id")
            scores.insert(0, "fold", fold_index); scores.insert(1, "fraction", fraction)
            scores.insert(2, "model", "target_train_drug_mean")
            score_frames.append(scores.rename(columns={"patient_id": "unit_id"}))
            receipts.append({"fold": fold_index, "fraction": fraction,
                             "n_train_patients": len(set(train_ids)),
                             "n_train_drugs": len(target_mean),
                             "n_test_patients_scored": len(scores),
                             "mean_patient_spearman": float(scores["spearman"].mean())})
    source_predictions = pd.concat(source_predictions, ignore_index=True)
    target_predictions = pd.concat(target_predictions, ignore_index=True)
    scores = pd.concat(score_frames, ignore_index=True)
    source_predictions.to_parquet(output_dir / "source_predictions.parquet", index=False)
    target_predictions.to_parquet(output_dir / "target_predictions.parquet", index=False)
    scores.to_parquet(output_dir / "unit_scores.parquet", index=False)
    comparisons = []
    source_score = scores.loc[(scores["model"] == "source_train_drug_mean") &
                              (scores["fraction"] == 0.0), ["unit_id", "spearman"]]
    for fraction in (0.25, 0.5, 1.0):
        target_score = scores.loc[(scores["model"] == "target_train_drug_mean") &
                                  (scores["fraction"] == fraction), ["unit_id", "spearman"]]
        paired = source_score.merge(target_score, on="unit_id", suffixes=("_source", "_target"))
        difference = paired["spearman_target"].to_numpy() - paired["spearman_source"].to_numpy()
        comparisons.append({"comparison": f"target_mean_minus_source_mean_fraction_{fraction}",
                            **bootstrap(difference)})
    files = ["source_predictions.parquet", "target_predictions.parquet", "unit_scores.parquet"]
    report = {"experiment_id": "EXP-036", "run_id": output_dir.name,
              "status": "BASELINES_COMPLETE_PENDING_REVIEW",
              "source_internal_mean_spearman": float(scores.loc[
                  (scores["model"] == "source_train_drug_mean") & scores["fraction"].isna(),
                  "spearman"].mean()),
              "target_receipts": receipts, "paired_statistics": comparisons,
              "data_manifest_sha256": sha256(contract_dir / "manifest.json"),
              "split_sha256": sha256(split_path),
              "files_sha256": {name: sha256(output_dir / name) for name in files}}
    temporary = output_dir / "run_summary.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_dir / "run_summary.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--source-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.contract_dir, args.split, args.source_npz, args.output_dir),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
