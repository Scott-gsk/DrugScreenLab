"""Independently verify PharmaFormer target predictions against frozen patient keys."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ARMS = {"random_full", "source_head", "source_full", "source_adapter"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(input_dir: Path, split_path: Path, fold_dir: Path) -> dict:
    summary = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
    fold = int(summary["fold"])
    fraction = float(summary.get("fraction", float("nan")))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if (fold not in range(5) or summary["split_sha256"] != sha256(split_path)
            or summary["input_manifest_sha256"] != sha256(input_dir / "manifest.json")
            or set(summary["arms"]) != ARMS
            or fraction not in {0.25, 0.5, 1.0}
            or fraction not in {float(item["fraction"])
                                for item in split["target_folds"][fold]["label_fractions"]}):
        raise ValueError("fold receipt does not match the frozen inputs")
    for name in ("predictions.parquet", "patient_scores.parquet", "inner_selection.parquet"):
        if summary["files_sha256"][name] != sha256(fold_dir / name):
            raise ValueError(f"fold artifact checksum mismatch: {name}")
    if summary["status"] != "SMOKE_ONLY":
        expected_model_keys = {(arm, seed) for arm in ARMS for seed in summary["seeds"]}
        receipts = summary["final_receipts"]
        if {(item["arm"], item["seed"]) for item in receipts} != expected_model_keys \
                or len(receipts) != len(expected_model_keys):
            raise ValueError("missing or duplicate final model receipts")
        for item in receipts:
            checkpoint = fold_dir / f"model_{item['arm']}_seed{item['seed']}.pt"
            if item["checkpoint_sha256"] != sha256(checkpoint):
                raise ValueError(f"checkpoint checksum mismatch: {checkpoint}")
    observed = pd.read_parquet(input_dir / "patient_pairs.parquet")
    predictions = pd.read_parquet(fold_dir / "predictions.parquet")
    scores = pd.read_parquet(fold_dir / "patient_scores.parquet")
    expected = observed.loc[observed["patient_id"].isin(
        split["target_folds"][fold]["outer_test_ids"]),
        ["patient_id", "drug_id", "response"]].rename(columns={"response": "observed"})
    if expected.duplicated(["patient_id", "drug_id"]).any():
        raise ValueError("frozen patient/drug keys are not unique")
    if (predictions.duplicated(["fold", "arm", "seed", "patient_id", "drug_id"]).any()
            or scores.duplicated(["fold", "arm", "seed", "patient_id"]).any()
            or not predictions["fold"].eq(fold).all()
            or not scores["fold"].eq(fold).all()
            or not predictions["fraction"].eq(fraction).all()
            or not scores["fraction"].eq(fraction).all()
            or not np.isfinite(predictions[["observed", "predicted"]].to_numpy()).all()):
        raise ValueError("saved prediction or patient-score axes are invalid")
    expected_pairs = set(map(tuple, expected[["patient_id", "drug_id"]].to_numpy()))
    expected_patients = set(expected["patient_id"])
    max_observed_error = 0.0
    max_score_error = 0.0
    for arm in ARMS:
        for seed in summary["seeds"]:
            part = predictions.loc[predictions["arm"].eq(arm) & predictions["seed"].eq(seed)]
            unit = scores.loc[scores["arm"].eq(arm) & scores["seed"].eq(seed)]
            if (set(map(tuple, part[["patient_id", "drug_id"]].to_numpy())) != expected_pairs
                    or set(unit["patient_id"]) != expected_patients
                    or len(part) != len(expected) or len(unit) != len(expected_patients)):
                raise ValueError(f"missing or extra patient/drug predictions: {arm}, {seed}")
            joined = part.merge(expected, on=["patient_id", "drug_id"],
                                suffixes=("_saved", "_source"), validate="one_to_one")
            error = float(np.max(np.abs(joined["observed_saved"] - joined["observed_source"])))
            max_observed_error = max(max_observed_error, error)
            if error > 1e-10:
                raise ValueError("saved observed responses disagree with frozen inputs")
            for row in unit.itertuples(index=False):
                patient = part.loc[part["patient_id"].eq(row.patient_id)]
                y = patient["observed"].to_numpy(dtype=float)
                yhat = patient["predicted"].to_numpy(dtype=float)
                expected_score = float(spearmanr(y, yhat).statistic) if len(y) >= 3 else float("nan")
                if np.isfinite(expected_score) != np.isfinite(row.spearman):
                    raise ValueError("saved score validity disagrees with independent recomputation")
                if np.isfinite(expected_score):
                    difference = abs(expected_score - row.spearman)
                    max_score_error = max(max_score_error, difference)
                    if difference > 1e-12:
                        raise ValueError("saved patient score disagrees with independent recomputation")
                if len(y) != row.n_drugs:
                    raise ValueError("saved patient drug denominator is wrong")
    return {"status": "PASS", "fold": fold, "scope": summary["status"],
            "n_predictions": len(predictions), "n_patient_scores": len(scores),
            "n_checkpoints_verified": 0 if summary["status"] == "SMOKE_ONLY" else len(receipts),
            "max_observed_error": max_observed_error, "max_score_error": max_score_error,
            "predictions_sha256": sha256(fold_dir / "predictions.parquet"),
            "patient_scores_sha256": sha256(fold_dir / "patient_scores.parquet")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.input_dir, args.split, args.fold_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
