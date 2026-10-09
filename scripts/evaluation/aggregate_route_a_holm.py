"""Close the preregistered A1/A2/A3 Route-A comparison family."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from drug_screen.evaluation.exp031_eval import patient_score_paired_bootstrap
from drug_screen.evaluation.permutation import permutation_p_value
from scripts.evaluation.verify_exp036_ridge_interaction import verify as verify_ridge


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    if not p_values or any(not np.isfinite(value) or value < 0 or value > 1
                           for value in p_values.values()):
        raise ValueError("Holm family requires finite p-values in [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def _comparison(*, p: float, effect: float, ci: list[float], n: int,
                meaning: str) -> dict:
    values = np.asarray([p, effect, *ci], dtype=float)
    if (not np.isfinite(values).all() or not 0 <= p <= 1 or n < 3
            or len(ci) != 2 or ci[0] > ci[1]):
        raise ValueError(f"invalid registered comparison: {meaning}")
    return {
        "p_two_sided_unadjusted": float(p),
        "effect": float(effect),
        "ci95": [float(ci[0]), float(ci[1])],
        "n_patients": int(n),
        "unit": "held-out patient",
        "higher_is_better": True,
        "meaning": meaning,
    }


def _resample_statistics(difference: np.ndarray, *, seed: int) -> dict:
    difference = np.asarray(difference, dtype=float)
    if difference.size < 3 or not np.isfinite(difference).all():
        raise ValueError("paired patient differences are incomplete or undefined")
    rng = np.random.default_rng(seed)
    bootstrap = np.array([
        difference[rng.integers(0, len(difference), len(difference))].mean()
        for _ in range(2000)
    ])
    rng = np.random.default_rng(seed)
    observed = abs(float(difference.mean()))
    null = np.array([
        abs(float(np.mean(difference * rng.choice([-1.0, 1.0], len(difference)))))
        for _ in range(10000)
    ])
    return {
        "estimate": float(difference.mean()),
        "ci95": [float(np.quantile(bootstrap, 0.025)),
                 float(np.quantile(bootstrap, 0.975))],
        "p": permutation_p_value(observed, null, alternative="greater"),
        "n": int(len(difference)),
    }


def _ridge_resample_statistics(difference: np.ndarray, *, seed: int) -> dict:
    """Replay Ridge's registered single-stream bootstrap and sign-flip draws."""
    difference = np.asarray(difference, dtype=float)
    if difference.size < 3 or not np.isfinite(difference).all():
        raise ValueError("paired Ridge patient differences are incomplete or undefined")
    rng = np.random.default_rng(seed)
    bootstrap = np.array([
        difference[rng.integers(0, len(difference), len(difference))].mean()
        for _ in range(2000)
    ])
    observed = abs(float(difference.mean()))
    null = np.array([
        abs(float(np.mean(difference * rng.choice([-1.0, 1.0], len(difference)))) )
        for _ in range(10000)
    ])
    return {
        "estimate": float(difference.mean()),
        "ci95": [float(np.quantile(bootstrap, 0.025)),
                 float(np.quantile(bootstrap, 0.975))],
        "p": float((1 + int(np.count_nonzero(null >= observed))) / 10001),
        "n": int(len(difference)),
    }


def _assert_statistics_match(stored: dict, actual: dict, *, p_key: str,
                             low_key: str = "ci95", high_key: str | None = None,
                             n_key: str) -> None:
    if high_key is None:
        ci = stored[low_key]
    else:
        ci = [stored[low_key], stored[high_key]]
    actual_ci = actual["ci95"]
    if (stored.get(n_key) != actual["n"]
            or not np.isclose(stored["estimate"], actual["estimate"], rtol=0, atol=1e-12)
            or not np.isclose(stored[p_key], actual["p"], rtol=0, atol=1e-12)
            or not np.allclose(ci, actual_ci, rtol=0, atol=1e-12)):
        raise ValueError("stored paired statistic differs from independent recomputation")


def verify_pharmaformer_aggregate(
    summary_path: Path, folds_dir: Path, baseline_dir: Path, input_dir: Path,
    split_path: Path,
) -> tuple[dict, pd.DataFrame]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("experiment_id") != "EXP-036"
            or summary.get("status") != "FIVE_FOLDS_AGGREGATED_PENDING_HOLM_AND_REVIEW"
            or summary.get("n_folds") != 5 or summary.get("n_patients") != 51):
        raise ValueError("EXP-036 PharmaFormer aggregate is incomplete or unverified")
    if (summary.get("source_sha256", {}).get("split") != sha256(split_path)
            or summary.get("source_sha256", {}).get("input_manifest")
            != sha256(input_dir / "manifest.json")
            or summary.get("source_sha256", {}).get("baseline_run_summary")
            != sha256(baseline_dir / "run_summary.json")):
        raise ValueError("EXP-036 aggregate source hashes do not match the frozen inputs")
    baseline_summary_path = baseline_dir / "run_summary.json"
    baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    baseline_scores_path = baseline_dir / "unit_scores.parquet"
    if (baseline_summary.get("status") != "BASELINES_COMPLETE_PENDING_REVIEW"
            or baseline_summary.get("files_sha256", {}).get("unit_scores.parquet")
            != sha256(baseline_scores_path)):
        raise ValueError("EXP-036 target mean baseline receipt is stale or incomplete")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    receipts = summary.get("folds", [])
    if len(receipts) != 5 or [row.get("fold") for row in receipts] != list(range(5)):
        raise ValueError("EXP-036 aggregate does not contain all five ordered fold receipts")
    fold_scores = []
    for fold, receipt in enumerate(receipts):
        fold_dir = folds_dir / f"fold_{fold}"
        summary_file = fold_dir / "fold_summary.json"
        verification_file = fold_dir / "verification.json"
        prediction_file = fold_dir / "predictions.parquet"
        scores_file = fold_dir / "patient_scores.parquet"
        run = json.loads(summary_file.read_text(encoding="utf-8"))
        verification = json.loads(verification_file.read_text(encoding="utf-8"))
        if (receipt.get("summary_sha256") != sha256(summary_file)
                or receipt.get("verification_sha256") != sha256(verification_file)
                or receipt.get("patient_scores_sha256") != sha256(scores_file)
                or run.get("status") != "FOLD_COMPLETE_PENDING_INDEPENDENT_REVIEW"
                or run.get("fold") != fold or run.get("fraction") != 0.5
                or run.get("split_sha256") != sha256(split_path)
                or run.get("input_manifest_sha256") != sha256(input_dir / "manifest.json")
                or set(run.get("arms", [])) != {
                    "random_full", "source_head", "source_full", "source_adapter",
                }
                or run.get("seeds") != [20260922, 20260923, 20260924]
                or verification.get("status") != "PASS"
                or verification.get("fold") != fold
                or verification.get("n_checkpoints_verified") != 12
                or verification.get("predictions_sha256") != sha256(prediction_file)
                or verification.get("patient_scores_sha256") != sha256(scores_file)
                or run.get("files_sha256", {}).get("predictions.parquet")
                != sha256(prediction_file)
                or run.get("files_sha256", {}).get("patient_scores.parquet")
                != sha256(scores_file)):
            raise ValueError(f"EXP-036 PharmaFormer fold {fold} receipt is stale or incomplete")
        scores = pd.read_parquet(scores_file)
        expected_patients = set(map(str, split["target_folds"][fold]["outer_test_ids"]))
        if (scores.duplicated(["fold", "arm", "seed", "patient_id"]).any()
                or set(scores["patient_id"].astype(str)) != expected_patients
                or scores["patient_id"].nunique() != len(expected_patients)
                or len(scores) != len(expected_patients) * 4 * 3
                or set(scores["arm"]) != {
                    "random_full", "source_head", "source_full", "source_adapter",
                }
                or set(scores["seed"].astype(int)) != {20260922, 20260923, 20260924}
                or not scores["fold"].eq(fold).all()
                or not scores["fraction"].eq(0.5).all()
                or not np.isfinite(scores["spearman"].to_numpy(float)).all()):
            raise ValueError(f"EXP-036 PharmaFormer fold {fold} patient scores are invalid")
        fold_scores.append(scores)

    for name, expected in summary.get("files_sha256", {}).items():
        if sha256(summary_path.parent / name) != expected:
            raise ValueError(f"EXP-036 aggregate artifact hash mismatch: {name}")
    paired = pd.read_parquet(summary_path.parent / "paired_patient_scores.parquet")
    if (paired["patient_id"].nunique() != 51
            or paired.duplicated(["fold", "patient_id"]).any()
            or not np.isfinite(paired[
                ["random_full", "source_full", "target_train_drug_mean"]
            ].to_numpy(float)).all()):
        raise ValueError("EXP-036 paired aggregate patient matrix is invalid")
    all_scores = pd.concat(fold_scores, ignore_index=True)
    averaged = all_scores.groupby(
        ["fold", "patient_id", "arm"], sort=True, as_index=False,
    ).agg(spearman=("spearman", "mean"))
    expected_paired = averaged.pivot(
        index=["fold", "patient_id"], columns="arm", values="spearman",
    )
    baseline = pd.read_parquet(baseline_scores_path)
    baseline = baseline.loc[
        baseline["model"].eq("target_train_drug_mean")
        & baseline["fraction"].eq(0.5)
    ]
    if baseline.duplicated(["fold", "unit_id"]).any() or len(baseline) != 51:
        raise ValueError("EXP-036 target training mean baseline has the wrong patient axis")
    reference = baseline.set_index(["fold", "unit_id"])["spearman"]
    reference.index.names = ["fold", "patient_id"]
    expected_paired = expected_paired.join(reference.rename("target_train_drug_mean"),
                                            how="inner", validate="one_to_one")
    actual_paired = paired.set_index(["fold", "patient_id"]).sort_index()
    if (len(expected_paired) != 51 or not actual_paired.index.equals(expected_paired.index)
            or not np.allclose(
                actual_paired[expected_paired.columns].to_numpy(float),
                expected_paired.to_numpy(float), rtol=0, atol=1e-12,
            )):
        raise ValueError("EXP-036 paired patient aggregate differs from fold scores and baseline")

    primary = {
        "A1_random_full_vs_target_drug_mean": (
            paired["random_full"] - paired["target_train_drug_mean"], 20260923,
        ),
        "A2_source_full_vs_random_full": (
            paired["source_full"] - paired["random_full"], 20260924,
        ),
    }
    for name, (delta, seed) in primary.items():
        actual = _resample_statistics(delta.to_numpy(float), seed=seed)
        _assert_statistics_match(
            summary["statistics"][name], actual,
            p_key="p_two_sided_uncorrected", n_key="n_patients",
        )
    return summary, paired


def verify_ridge_primary(ridge_dir: Path, ridge: dict) -> dict:
    scores = pd.read_parquet(ridge_dir / "patient_scores.parquet")
    subset = scores.loc[scores["fraction"].eq(0.5)]

    def patient_series(model: str) -> pd.Series:
        frame = subset.loc[subset["model"].eq(model)]
        if model == "patient_expression_permuted":
            frame = frame.groupby("patient_id", sort=True)["spearman"].mean()
            result = frame
        else:
            if frame["patient_id"].duplicated().any():
                raise ValueError(f"EXP-036 Ridge has duplicate {model} patient scores")
            result = frame.set_index("patient_id")["spearman"]
        return result

    interaction = patient_series("expression_drug_interaction")
    drug_only = patient_series("drug_only_ridge")
    permuted = patient_series("patient_expression_permuted")
    if (not interaction.index.is_unique or not drug_only.index.is_unique
            or not permuted.index.is_unique or len(interaction) != 51
            or set(interaction.index) != set(drug_only.index)
            or set(interaction.index) != set(permuted.index)
            or not np.isfinite(np.stack([
                interaction.to_numpy(float), drug_only.to_numpy(float),
                permuted.to_numpy(float),
            ])).all()):
        raise ValueError("EXP-036 Ridge primary scores do not align over 51 patients")
    stored = next(row for row in ridge["comparisons"] if row.get("fraction") == 0.5)
    results = {}
    for name, left, right in (
        ("interaction_minus_drug_only", interaction, drug_only),
        ("interaction_minus_permuted", interaction, permuted),
    ):
        common = left.index.intersection(right.index)
        actual = _ridge_resample_statistics(
            (left.loc[common] - right.loc[common]).to_numpy(float), seed=20260922,
        )
        _assert_statistics_match(
            stored[name], actual, p_key="p_two_sided",
            low_key="low", high_key="high", n_key="n_patients",
        )
        results[name] = stored[name]
    return {"experiment_id": ridge.get("experiment_id"),
            "status": ridge.get("status"), "comparisons": [
                {"fraction": 0.5, **results},
            ]}


def verify_a3(a3_dir: Path) -> dict:
    summary_path = a3_dir / "a3_summary.json"
    scores_path = a3_dir / "patient_scores.parquet"
    manifest_path = a3_dir / "manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("outputs", {}).get("a3_summary.json") != sha256(summary_path)
            or manifest.get("outputs", {}).get("patient_scores.parquet") != sha256(scores_path)
            or summary.get("patient_scores_sha256") != sha256(scores_path)):
        raise ValueError("EXP-041 A3 aggregate hash receipt is incomplete or stale")
    eligible = pd.read_parquet(scores_path)
    eligible = eligible.loc[eligible["eligible_primary"].astype(bool)].copy()
    if (eligible["patient_id"].duplicated().any()
            or not np.isfinite(eligible["gain_adapter_minus_full"].to_numpy(float)).all()
            or len(eligible) != summary.get("main", {}).get("n_valid_patients")
            or not np.isclose(eligible["gain_adapter_minus_full"].mean(),
                              summary["main"]["gain"], rtol=0, atol=1e-12)):
        raise ValueError("EXP-041 A3 paired patient effects do not reproduce the summary")
    adapter = eligible.set_index("patient_id")["source_adapter_seed_mean_spearman"]
    full = eligible.set_index("patient_id")["source_full_seed_mean_spearman"]
    ci = patient_score_paired_bootstrap(adapter, full, n=2000, seed=20260922)
    delta = adapter.to_numpy(float) - full.to_numpy(float)
    rng = np.random.default_rng(20260922)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(10000, len(delta)))
    null = np.abs((signs * delta).mean(axis=1))
    p_value = permutation_p_value(
        abs(float(delta.mean())), null, alternative="greater",
    )
    main = summary["main"]
    if (main.get("n_valid_patients") != ci.n_units
            or not np.isclose(main.get("gain"), ci.point_estimate, rtol=0, atol=1e-12)
            or not np.allclose(main.get("ci95"), [ci.low, ci.high], rtol=0, atol=1e-12)
            or not np.isclose(main.get("paired_sign_flip_p_two_sided_finite_corrected"),
                              p_value, rtol=0, atol=1e-12)):
        raise ValueError("EXP-041 A3 confidence interval or test differs on independent recomputation")
    return summary


def build_family(pharmaformer: dict, ridge: dict, a3: dict) -> dict:
    """Map every preregistered route-A contrast to its unadjusted statistic."""
    if (pharmaformer.get("experiment_id") != "EXP-036"
            or pharmaformer.get("status") != "FIVE_FOLDS_AGGREGATED_PENDING_HOLM_AND_REVIEW"
            or pharmaformer.get("n_folds") != 5 or pharmaformer.get("n_patients") != 51
            or [row.get("fold") for row in pharmaformer.get("folds", [])] != list(range(5))):
        raise ValueError("EXP-036 PharmaFormer aggregate is incomplete or unverified")
    if ridge.get("experiment_id") != "EXP-036" or ridge.get("status") != "RIDGE_INTERACTION_COMPLETE_PENDING_REVIEW":
        raise ValueError("EXP-036 Ridge interaction run is incomplete")
    if (a3.get("experiment_id") != "EXP-041"
            or a3.get("status") != "A3_PRIMARY_CONTRAST_READY_LEARNING_CURVE_AND_ROUTE_A_HOLM_PENDING"
            or a3.get("fraction") != 0.5 or a3.get("parent_experiment_id") != "EXP-036"
            or a3.get("n_test_patients_total") != 51):
        raise ValueError("EXP-041 A3 aggregate is incomplete or unverified")

    ridge_primary = next(
        (row for row in ridge.get("comparisons", []) if row.get("fraction") == 0.5), None
    )
    if ridge_primary is None:
        raise ValueError("EXP-036 Ridge summary lacks the frozen 50% primary tier")
    pharma_stats = pharmaformer.get("statistics", {})
    a1_pharma = pharma_stats.get("A1_random_full_vs_target_drug_mean")
    a2 = pharma_stats.get("A2_source_full_vs_random_full")
    if a1_pharma is None or a2 is None:
        raise ValueError("EXP-036 PharmaFormer summary lacks a registered A1 or A2 contrast")
    a3_main = a3.get("main", {})

    comparisons = {
        "A1_pharmaformer_random_vs_target_drug_mean": _comparison(
            p=a1_pharma["p_two_sided_uncorrected"], effect=a1_pharma["estimate"],
            ci=a1_pharma["ci95"], n=a1_pharma["n_patients"],
            meaning="random-initialized PharmaFormer minus target-training-patient drug mean",
        ),
        "A1_ridge_interaction_vs_drug_only": _comparison(
            p=ridge_primary["interaction_minus_drug_only"]["p_two_sided"],
            effect=ridge_primary["interaction_minus_drug_only"]["estimate"],
            ci=[ridge_primary["interaction_minus_drug_only"]["low"],
                ridge_primary["interaction_minus_drug_only"]["high"]],
            n=ridge_primary["interaction_minus_drug_only"]["n_patients"],
            meaning="low-rank expression-drug interaction minus drug-only Ridge",
        ),
        "A1_ridge_interaction_vs_patient_expression_permutation": _comparison(
            p=ridge_primary["interaction_minus_permuted"]["p_two_sided"],
            effect=ridge_primary["interaction_minus_permuted"]["estimate"],
            ci=[ridge_primary["interaction_minus_permuted"]["low"],
                ridge_primary["interaction_minus_permuted"]["high"]],
            n=ridge_primary["interaction_minus_permuted"]["n_patients"],
            meaning="low-rank interaction minus training-side patient-expression permutation",
        ),
        "A2_source_full_vs_random_full": _comparison(
            p=a2["p_two_sided_uncorrected"], effect=a2["estimate"],
            ci=a2["ci95"], n=a2["n_patients"],
            meaning="source-pretrained PharmaFormer fine-tuning minus matched random initialization",
        ),
        "A3_source_adapter_vs_source_full": _comparison(
            p=a3_main["paired_sign_flip_p_two_sided_finite_corrected"],
            effect=a3_main["gain"], ci=a3_main["ci95"],
            n=a3_main["n_valid_patients"],
            meaning="frozen-backbone residual adapter minus full fine-tuning",
        ),
    }
    adjusted = holm_adjust({name: row["p_two_sided_unadjusted"]
                            for name, row in comparisons.items()})
    for name, row in comparisons.items():
        row["p_holm_adjusted"] = adjusted[name]
    return {
        "status": "ROUTE_A_HOLM_FAMILY_AGGREGATED_PENDING_REVIEW",
        "family": list(comparisons),
        "adjustment": "Holm step-down family-wise error correction across all five preregistered A1/A2/A3 contrasts",
        "comparisons": comparisons,
        "interpretation": "Report paired patient effect, 95% interval, valid patient count, and adjusted p-value together; statistical significance alone does not establish practical benefit.",
    }


def aggregate(pharmaformer_summary_path: Path, pharmaformer_folds_dir: Path,
              baseline_dir: Path, input_dir: Path, ridge_dir: Path,
              split_path: Path, a3_dir: Path, output_path: Path) -> dict:
    if output_path.exists():
        raise FileExistsError(f"Route-A Holm output already exists: {output_path}")
    pharmaformer, paired = verify_pharmaformer_aggregate(
        pharmaformer_summary_path, pharmaformer_folds_dir, baseline_dir,
        input_dir, split_path,
    )
    ridge_summary_path = ridge_dir / "run_summary.json"
    ridge = json.loads(ridge_summary_path.read_text(encoding="utf-8"))
    ridge_receipt = json.loads((ridge_dir / "independent_verification.json").read_text(encoding="utf-8"))
    if ridge_receipt.get("status") != "PASS":
        raise ValueError("EXP-036 Ridge independent verification is not PASS")
    for name, expected in ridge.get("files_sha256", {}).items():
        if sha256(ridge_dir / name) != expected:
            raise ValueError(f"EXP-036 Ridge artifact hash mismatch: {name}")
    fresh_ridge_receipt = verify_ridge(split_path, ridge_dir)
    if fresh_ridge_receipt.get("status") != "PASS":
        raise ValueError("fresh EXP-036 Ridge verification failed")
    if ridge.get("split_sha256") != sha256(split_path):
        raise ValueError("EXP-036 Ridge run used a different frozen patient split")
    ridge = verify_ridge_primary(ridge_dir, ridge)
    a3_summary_path = a3_dir / "a3_summary.json"
    a3_scores_path = a3_dir / "patient_scores.parquet"
    a3 = verify_a3(a3_dir)
    report = build_family(pharmaformer, ridge, a3)
    report["independent_recomputation"] = {
        "EXP-036_PharmaFormer": "five fold score and source-hash checks; A1/A2 paired bootstrap and sign-flip statistics recomputed",
        "EXP-036_Ridge": "outer split and patient scores rechecked; 50% primary effects, bootstrap intervals, and sign-flip tests recomputed",
        "EXP-041_A3": "patient-level adapter/full differences, bootstrap interval, and finite corrected sign-flip test recomputed",
    }
    report["sources"] = {
        "EXP-036_PharmaFormer_aggregate_sha256": sha256(pharmaformer_summary_path),
        "EXP-036_Ridge_run_summary_sha256": sha256(ridge_summary_path),
        "EXP-036_Ridge_predictions_sha256": sha256(ridge_dir / "predictions.parquet"),
        "EXP-036_Ridge_patient_scores_sha256": sha256(ridge_dir / "patient_scores.parquet"),
        "EXP-036_Ridge_fresh_verification_status": fresh_ridge_receipt["status"],
        "EXP-041_A3_summary_sha256": sha256(a3_summary_path),
        "EXP-041_A3_patient_scores_sha256": sha256(a3_scores_path),
        "patient_split_sha256": sha256(split_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_path.parent,
        prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pharmaformer-summary", type=Path, required=True)
    parser.add_argument("--pharmaformer-folds-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--ridge-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(
        args.pharmaformer_summary, args.pharmaformer_folds_dir,
        args.baseline_dir, args.input_dir, args.ridge_dir, args.split,
        args.a3_dir, args.output,
    )
    print(json.dumps({"status": report["status"], "comparisons": report["comparisons"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
