"""独立核验 EXP-036 Ridge 交互预测身份、患者分数与配对差。"""

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


def verify(split_path: Path, run_dir: Path) -> dict:
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    scores = pd.read_parquet(run_dir / "patient_scores.parquet")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    for name in ("predictions.parquet", "patient_scores.parquet"):
        if sha256(run_dir / name) != summary["files_sha256"][name]:
            raise ValueError(f"checksum mismatch: {name}")
    if predictions.duplicated(["fold", "fraction", "model", "seed", "patient_id", "drug_id"]).any():
        raise ValueError("duplicate prediction identity")
    if not np.isfinite(predictions[["observed", "predicted"]].to_numpy()).all():
        raise ValueError("non-finite predictions")
    for fold_index, fold in enumerate(split["target_folds"]):
        actual = set(predictions.loc[predictions["fold"].eq(fold_index), "patient_id"])
        if actual != set(fold["outer_test_ids"]):
            raise ValueError("prediction patients differ from frozen outer test")
        if actual & set(fold["outer_train_ids"]):
            raise ValueError("outer train/test patient leakage")
    maximum = 0.0
    for row in scores.itertuples(index=False):
        frame = predictions.loc[
            predictions["fold"].eq(row.fold)
            & predictions["fraction"].eq(row.fraction)
            & predictions["model"].eq(row.model)
            & predictions["seed"].eq(row.seed)
            & predictions["patient_id"].eq(row.patient_id)]
        expected = float(spearmanr(frame["observed"], frame["predicted"]).statistic)
        error = abs(expected - row.spearman)
        maximum = max(maximum, error)
        if error > 1e-12:
            raise ValueError("patient score mismatch")
    recomputed = []
    for fraction in (0.25, 0.5, 1.0):
        subset = scores.loc[scores["fraction"].eq(fraction)]
        interaction = subset.loc[subset["model"].eq("expression_drug_interaction")].set_index("patient_id")
        drug = subset.loc[subset["model"].eq("drug_only_ridge")].set_index("patient_id")
        perm = subset.loc[subset["model"].eq("patient_expression_permuted")].groupby("patient_id")["spearman"].mean()
        common = interaction.index.intersection(drug.index).intersection(perm.index)
        recomputed.append({"fraction": fraction,
                           "interaction_minus_drug_only": float(
                               (interaction.loc[common, "spearman"] - drug.loc[common, "spearman"]).mean()),
                           "interaction_minus_permuted": float(
                               (interaction.loc[common, "spearman"] - perm.loc[common]).mean()),
                           "n_patients": len(common)})
    for actual, expected in zip(summary["comparisons"], recomputed, strict=True):
        if (abs(actual["interaction_minus_drug_only"]["estimate"]
                - expected["interaction_minus_drug_only"]) > 1e-12
                or abs(actual["interaction_minus_permuted"]["estimate"]
                       - expected["interaction_minus_permuted"]) > 1e-12):
            raise ValueError("reported paired effect differs from patient scores")
    return {"status": "PASS", "n_predictions": len(predictions),
            "n_patient_scores": len(scores), "max_score_error": maximum,
            "recomputed_effects": recomputed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.split, args.run_dir)
    (args.run_dir / "independent_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
