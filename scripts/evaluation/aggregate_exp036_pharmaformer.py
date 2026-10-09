"""Aggregate five verified patient-held-out PharmaFormer folds without model selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ARMS = ("random_full", "source_head", "source_full", "source_adapter")
SEEDS = (20260922, 20260923, 20260924)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_statistics(gain: np.ndarray, *, seed: int) -> dict:
    if len(gain) != 51 or not np.isfinite(gain).all():
        raise ValueError("paired PharmaFormer comparison must cover all 51 finite patients")
    rng = np.random.default_rng(seed)
    bootstrap = np.array([gain[rng.integers(0, len(gain), len(gain))].mean()
                          for _ in range(2000)])
    rng = np.random.default_rng(seed)
    observed = abs(float(gain.mean()))
    extreme = sum(abs(float((gain * rng.choice([-1.0, 1.0], len(gain))).mean())) >= observed
                  for _ in range(10000))
    return {"estimate": float(gain.mean()),
            "ci95": [float(np.quantile(bootstrap, .025)), float(np.quantile(bootstrap, .975))],
            "p_two_sided_uncorrected": (extreme + 1) / 10001,
            "n_patients": len(gain), "bootstrap_replicates": 2000,
            "sign_flip_replicates": 10000, "unit": "held-out patient"}


def aggregate(folds_dir: Path, baseline_dir: Path, input_dir: Path,
              split_path: Path, output_dir: Path, fraction: float = 0.5) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"aggregate output exists: {output_dir}")
    if fraction not in {0.25, 0.5, 1.0}:
        raise ValueError("label fraction must be one of the frozen 0.25, 0.5, or 1.0 levels")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    frames, fold_receipts = [], []
    for fold in range(5):
        fold_dir = folds_dir / f"fold_{fold}"
        receipt = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
        verification = json.loads((fold_dir / "verification.json").read_text(encoding="utf-8"))
        if (receipt["status"] != "FOLD_COMPLETE_PENDING_INDEPENDENT_REVIEW"
                or receipt["fold"] != fold or receipt["split_sha256"] != sha256(split_path)
                or receipt["input_manifest_sha256"] != sha256(input_dir / "manifest.json")
                or float(receipt["fraction"]) != fraction
                or receipt["seeds"] != list(SEEDS) or set(receipt["arms"]) != set(ARMS)
                or verification["status"] != "PASS" or verification["fold"] != fold
                or verification["patient_scores_sha256"] != sha256(fold_dir / "patient_scores.parquet")
                or verification["predictions_sha256"] != sha256(fold_dir / "predictions.parquet")):
            raise ValueError(f"PharmaFormer fold {fold} is incomplete or unverified")
        scores = pd.read_parquet(fold_dir / "patient_scores.parquet")
        patients = set(split["target_folds"][fold]["outer_test_ids"])
        if (scores.duplicated(["fold", "arm", "seed", "patient_id"]).any()
                or len(scores) != len(patients) * len(ARMS) * len(SEEDS)
                or set(scores["patient_id"]) != patients
                or set(scores["arm"]) != set(ARMS) or set(scores["seed"]) != set(SEEDS)
                or not scores["fold"].eq(fold).all()
                or not scores["fraction"].eq(fraction).all()
                or not np.isfinite(scores["spearman"]).all()):
            raise ValueError(f"PharmaFormer fold {fold} changes patient or seed denominator")
        frames.append(scores)
        fold_receipts.append({"fold": fold,
                              "summary_sha256": sha256(fold_dir / "fold_summary.json"),
                              "verification_sha256": sha256(fold_dir / "verification.json"),
                              "patient_scores_sha256": sha256(fold_dir / "patient_scores.parquet")})
    all_scores = pd.concat(frames, ignore_index=True)
    if all_scores["patient_id"].nunique() != 51:
        raise ValueError("five target folds do not cover 51 unique patients")
    averaged = all_scores.groupby(["fold", "patient_id", "arm"], sort=True, as_index=False).agg(
        spearman=("spearman", "mean"), n_drugs=("n_drugs", "first"))
    if all_scores.groupby(["fold", "patient_id", "arm"])["n_drugs"].nunique().gt(1).any():
        raise ValueError("PharmaFormer drug denominator changes across seeds")
    pivot = averaged.pivot(index=["fold", "patient_id"], columns="arm", values="spearman")
    baseline = pd.read_parquet(baseline_dir / "unit_scores.parquet")
    baseline = baseline.loc[baseline["model"].eq("target_train_drug_mean")
                            & baseline["fraction"].eq(fraction)]
    if baseline.duplicated(["fold", "unit_id"]).any() or len(baseline) != 51:
        raise ValueError("target-train drug mean baseline has wrong patient axis")
    reference = baseline.set_index(["fold", "unit_id"])["spearman"].rename("target_train_drug_mean")
    reference.index.names = ["fold", "patient_id"]
    paired = pivot.join(reference, how="inner", validate="one_to_one")
    if len(paired) != 51 or paired.isna().any().any():
        raise ValueError("PharmaFormer and target-drug-mean patient scores do not align")
    comparisons = {
        "A1_random_full_vs_target_drug_mean": ("random_full", "target_train_drug_mean"),
        "A2_source_full_vs_random_full": ("source_full", "random_full"),
        "A3_source_adapter_vs_source_full": ("source_adapter", "source_full"),
        "source_full_vs_target_drug_mean_descriptive_mixed_supervision": (
            "source_full", "target_train_drug_mean"),
        "source_adapter_vs_target_drug_mean": ("source_adapter", "target_train_drug_mean"),
        "source_head_vs_target_drug_mean": ("source_head", "target_train_drug_mean"),
    }
    statistics = {name: paired_statistics((paired[left] - paired[right]).to_numpy(),
                                           seed=20260923 + index)
                  for index, (name, (left, right)) in enumerate(comparisons.items())}
    output_dir.mkdir(parents=True)
    averaged_path = output_dir / "patient_scores_seed_averaged.parquet"
    paired_path = output_dir / "paired_patient_scores.parquet"
    averaged.to_parquet(averaged_path, index=False)
    paired.reset_index().to_parquet(paired_path, index=False)
    report = {"experiment_id": "EXP-036/041", "fraction": fraction,
              "status": "FIVE_FOLDS_AGGREGATED_PENDING_HOLM_AND_REVIEW",
              "metric": "within-patient Spearman over measured common drugs, higher is better",
              "seed_aggregation": "score each seed and patient before averaging three seeds per patient",
              "n_patients": len(paired), "n_folds": 5,
              "mean_patient_spearman": {column: float(paired[column].mean()) for column in paired.columns},
              "statistics": statistics,
              "confirmatory_family_status": "PENDING_A1_DRUG_ONLY_AND_EXPRESSION_PERMUTATION_AND_HOLM",
              "folds": fold_receipts,
              "source_sha256": {"split": sha256(split_path),
                                "input_manifest": sha256(input_dir / "manifest.json"),
                                "baseline_run_summary": sha256(baseline_dir / "run_summary.json"),
                                "baseline_unit_scores": sha256(baseline_dir / "unit_scores.parquet")},
              "files_sha256": {averaged_path.name: sha256(averaged_path),
                               paired_path.name: sha256(paired_path)}}
    (output_dir / "run_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fraction", type=float, choices=(0.25, 0.5, 1.0), default=0.5)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.folds_dir, args.baseline_dir, args.input_dir,
                               args.split, args.output_dir, args.fraction), ensure_ascii=False))


if __name__ == "__main__":
    main()
