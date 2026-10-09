"""Score and aggregate all five frozen EXP-048 outer folds after prediction lock.

This is a post-prediction scorer. It never selects arms, candidates, folds, or
seeds. Outer response values are joined only after all five prediction receipts
are complete and hash-verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FOLDS_ROOT = ROOT / "artifacts/experiments/EXP-048/20260926_fusion_training_v1"
DEFAULT_INPUT_DIR = ROOT / "data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1"
DEFAULT_SPLIT = ROOT / "data/splits/dual_route_splits_20260922_v1.json"
DEFAULT_CONFIG = ROOT / "configs/experiments/EXP-048.yaml"
DEFAULT_BINDING = ROOT / "artifacts/experiments/EXP-048/20260926_target_feature_binding_v1/receipt.json"
ARMS = ("F-A", "F-B", "F-F", "F-C", "F-S")
SEEDS = (20260922, 20260923, 20260924)
CONTRASTS = {
    "F-F_vs_F-A": ("F-F", "F-A"),
    "F-F_vs_F-B": ("F-F", "F-B"),
    "F-F_vs_F-C": ("F-F", "F-C"),
    "F-F_vs_F-S": ("F-F", "F-S"),
}
BOOTSTRAPS = 2_000
SIGN_FLIPS = 10_000
STAT_SEED = 20260922
EXPECTED = {
    "input_manifest": "c6f9c3349b0a8598280ba099381b781c68ca5599f6f3b7e749090a0e46460b72",
    "split": "9bfeeb4e0982421cc188a54e2a65f348709043c05529e641c7dd11ffc0635be8",
    "binding_receipt": "4877ba04d628258033f42770728bce52c56bcf94a2c0335a20464bb80fae6caf",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _assert_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"frozen {label} hash changed: {actual} != {expected}")


def load_predictions(folds_root: Path, split_path: Path, input_dir: Path,
                     config_path: Path, binding_path: Path) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    _assert_hash(split_path, EXPECTED["split"], "split")
    _assert_hash(input_dir / "manifest.json", EXPECTED["input_manifest"], "target input manifest")
    _assert_hash(binding_path, EXPECTED["binding_receipt"], "target identity/key binding receipt")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != "EXP-048" or len(config["drug_panel"]["target_inchikeys"]) != 9:
        raise ValueError("scoring config differs from the frozen nine-drug EXP-048 panel")
    panel = set(str(value) for value in config["drug_panel"]["target_inchikeys"])
    split = _load_json(split_path)
    binding = _load_json(binding_path)
    mask_info = binding["target_key_mask"]
    mask_path = ROOT / str(mask_info["path"])
    if not mask_path.is_file() or sha256(mask_path) != mask_info.get("sha256"):
        raise ValueError("fixed observed patient-drug key mask is missing or changed")
    full_mask = pd.read_csv(mask_path)
    full_mask["patient_id"] = full_mask["patient_id"].astype(str)
    full_mask["drug_id"] = full_mask["drug_id"].astype(str)
    if (full_mask.duplicated(["patient_id", "drug_id"]).any()
            or len(full_mask) != 459
            or full_mask["patient_id"].nunique() != 51
            or not full_mask.groupby("patient_id").size().eq(9).all()
            or not set(full_mask["drug_id"]).issubset(panel)):
        raise ValueError("fixed key mask must contain all 51x9 unique patient-panel slots")
    mask = full_mask.loc[full_mask["target_key_present"].astype(bool), ["patient_id", "drug_id"]]
    if (mask.duplicated(["patient_id", "drug_id"]).any() or len(mask) != 357
            or not mask.groupby("patient_id").size().eq(7).all()):
        raise ValueError("fixed scoring intersection must contain seven unique observed keys per patient")
    observed_keys = set(map(tuple, mask.to_numpy(dtype=str)))

    frames: list[pd.DataFrame] = []
    fold_receipts: list[dict[str, Any]] = []
    shared_input_codes: dict[str, str] | None = None
    fold_patient_sets: list[set[str]] = []
    all_outer_patients: set[str] = set()
    for fold in range(5):
        fold_dir = folds_root / f"fold_{fold}"
        summary_path = fold_dir / "fold_summary.json"
        manifest_path = fold_dir / "run_manifest.json"
        prediction_path = fold_dir / "predictions.parquet"
        if not all(path.is_file() for path in (summary_path, manifest_path, prediction_path)):
            raise FileNotFoundError(f"fold {fold} lacks summary, manifest, or predictions")
        summary, manifest = _load_json(summary_path), _load_json(manifest_path)
        if (summary.get("experiment_id") != "EXP-048" or summary.get("fold") != fold
                or summary.get("phase") != "predictions"
                or summary.get("status") != "FOLD_PREDICTIONS_COMPLETE_PENDING_INDEPENDENT_VERIFICATION"
                or manifest.get("fold") != fold
                or "predictions" not in manifest.get("phases_completed", [])):
            raise ValueError(f"fold {fold} does not have a complete frozen prediction phase")
        if summary.get("predictions_sha256") != sha256(prediction_path):
            raise ValueError(f"fold {fold} prediction file hash differs from its summary")
        hashes = summary.get("input_and_code_sha256", {})
        if shared_input_codes is None:
            shared_input_codes = hashes
        elif hashes != shared_input_codes:
            raise ValueError("fold input/code hashes differ; do not aggregate mixed protocol versions")
        if manifest.get("input_and_code_sha256") != hashes:
            raise ValueError(f"fold {fold} run manifest does not bind its summary hashes")

        verification_path = fold_dir / "independent_verification.json"
        if not verification_path.is_file():
            raise FileNotFoundError(f"fold {fold} lacks an independent label-blind verification receipt")
        verification = _load_json(verification_path)
        if (verification.get("status") != "PASS"
                or verification.get("fold") != fold
                or verification.get("summary_sha256") != sha256(summary_path)
                or verification.get("manifest_sha256") != sha256(manifest_path)
                or verification.get("predictions_sha256") != sha256(prediction_path)
                or verification.get("inner_selection_sha256") != summary.get("inner_selection_sha256")
                or verification.get("outer_response_values_read") is not False
                or verification.get("outer_metrics_calculated") is not False):
            raise ValueError(f"fold {fold} independent verification receipt is stale or invalid")

        outer_patients = set(str(value) for value in split["target_folds"][fold]["outer_test_ids"])
        if (len(outer_patients) != len(split["target_folds"][fold]["outer_test_ids"])
                or outer_patients & all_outer_patients
                or outer_patients - set(mask["patient_id"])):
            raise ValueError(f"fold {fold} outer patient identities overlap or are outside the key mask")
        all_outer_patients |= outer_patients
        fold_patient_sets.append(outer_patients)
        expected_keys = {key for key in observed_keys if key[0] in outer_patients}
        prediction = pd.read_parquet(prediction_path)
        required = {"fold", "arm", "seed", "patient_id", "drug_id", "prediction"}
        if not required.issubset(prediction.columns):
            raise ValueError(f"fold {fold} predictions lack required keyed columns")
        prediction["patient_id"] = prediction["patient_id"].astype(str)
        prediction["drug_id"] = prediction["drug_id"].astype(str)
        if (prediction.duplicated(["fold", "arm", "seed", "patient_id", "drug_id"]).any()
                or set(prediction["arm"]) != set(ARMS)
                or set(prediction["seed"].astype(int)) != set(SEEDS)
                or not prediction["fold"].eq(fold).all()
                or set(prediction["patient_id"]) != outer_patients
                or not np.isfinite(pd.to_numeric(prediction["prediction"], errors="raise")).all()):
            raise ValueError(f"fold {fold} prediction keys/arms/seeds are incomplete or invalid")
        actual_keys = set(map(tuple, prediction[["patient_id", "drug_id"]].drop_duplicates().to_numpy(dtype=str)))
        if actual_keys != expected_keys:
            raise ValueError(f"fold {fold} prediction keys differ from frozen common observed keys")
        expected_rows = len(expected_keys) * len(ARMS) * len(SEEDS)
        if len(prediction) != expected_rows:
            raise ValueError(f"fold {fold} expected {expected_rows} predictions, found {len(prediction)}")
        frames.append(prediction)
        fold_receipts.append({"fold": fold, "patient_count": len(outer_patients),
                              "summary_sha256": sha256(summary_path),
                              "manifest_sha256": sha256(manifest_path),
                              "predictions_sha256": sha256(prediction_path),
                              "independent_verification_sha256": sha256(verification_path)})
    if len(fold_patient_sets) != 5 or all_outer_patients != set(mask["patient_id"]):
        raise ValueError("five outer folds do not cover exactly the 51 frozen target patients")
    return pd.concat(frames, ignore_index=True), fold_receipts, {
        "config_sha256": sha256(config_path),
        "input_manifest_sha256": sha256(input_dir / "manifest.json"),
        "split_sha256": sha256(split_path), "binding_receipt_sha256": sha256(binding_path),
        "key_mask_sha256": sha256(mask_path), "folds": fold_receipts,
    }


def score_patients(predictions: pd.DataFrame, input_dir: Path) -> pd.DataFrame:
    # This is the sole scoring stage: all five prediction receipts are already
    # locked and checked before this function reads the response column.
    labels = pd.read_parquet(input_dir / "patient_pairs.parquet", columns=["patient_id", "drug_id", "response"])
    labels["patient_id"] = labels["patient_id"].astype(str)
    labels["drug_id"] = labels["drug_id"].astype(str)
    labels["response"] = pd.to_numeric(labels["response"], errors="raise")
    if labels.duplicated(["patient_id", "drug_id"]).any():
        raise ValueError("target response table has duplicate patient-drug keys")
    keys = predictions[["patient_id", "drug_id"]].drop_duplicates()
    labels = keys.merge(labels, on=["patient_id", "drug_id"], how="left", validate="one_to_one")
    if labels["response"].isna().any() or not np.isfinite(labels["response"].to_numpy(dtype=float)).all():
        raise ValueError("one or more frozen common prediction keys lacks a finite observed response")
    keyed = predictions.merge(labels, on=["patient_id", "drug_id"], how="left", validate="many_to_one")
    if keyed["response"].isna().any():
        raise ValueError("prediction-to-response identity join lost a key")
    rows: list[dict[str, Any]] = []
    for (fold, arm, seed, patient), group in keyed.groupby(["fold", "arm", "seed", "patient_id"], sort=True):
        observed = group["response"].to_numpy(dtype=np.float64)
        predicted = group["prediction"].to_numpy(dtype=np.float64)
        if len(group) != 7 or group["drug_id"].nunique() != 7:
            raise ValueError(f"{patient}/{arm}/{seed}: expected seven frozen common keys")
        if np.ptp(observed) == 0:
            rho, reason = float("nan"), "CONSTANT_OBSERVED"
        elif np.ptp(predicted) == 0:
            rho, reason = float("nan"), "CONSTANT_PREDICTED"
        else:
            rho = float(spearmanr(observed, predicted).statistic)
            reason = None if math.isfinite(rho) else "UNDEFINED_SPEARMAN"
        rows.append({"fold": int(fold), "patient_id": str(patient), "arm": str(arm),
                     "seed": int(seed), "n_observed_drugs": 7,
                     "spearman": rho, "undefined_reason": reason})
    result = pd.DataFrame(rows)
    expected = 51 * len(ARMS) * len(SEEDS)
    if len(result) != expected or result.duplicated(["fold", "patient_id", "arm", "seed"]).any():
        raise ValueError(f"patient score table is incomplete: expected {expected}, got {len(result)}")
    return result


def seed_average(patient_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (fold, patient, arm), group in patient_scores.groupby(["fold", "patient_id", "arm"], sort=True):
        values = group["spearman"].to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        rows.append({"fold": int(fold), "patient_id": str(patient), "arm": str(arm),
                     "spearman": float(values.mean()) if finite.all() and len(values) == len(SEEDS) else float("nan"),
                     "seed_count": int(finite.sum()),
                     "undefined_reason": None if finite.all() and len(values) == len(SEEDS)
                     else "UNDEFINED_IN_AT_LEAST_ONE_SEED"})
    averaged = pd.DataFrame(rows)
    expected = 51 * len(ARMS)
    if len(averaged) != expected or averaged.duplicated(["fold", "patient_id", "arm"]).any():
        raise ValueError("seed-averaged patient score table has invalid keys")
    return averaged


def paired_statistics(gain: np.ndarray, *, seed: int) -> dict[str, Any]:
    gain = np.asarray(gain, dtype=np.float64)
    if len(gain) < 2 or not np.isfinite(gain).all():
        raise ValueError("paired comparison requires at least two finite common patients")
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray([gain[rng.integers(0, len(gain), len(gain))].mean()
                            for _ in range(BOOTSTRAPS)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    observed = abs(float(gain.mean()))
    extreme = 0
    for _ in range(SIGN_FLIPS):
        randomized = gain * rng.choice(np.asarray([-1.0, 1.0]), len(gain))
        extreme += abs(float(randomized.mean())) >= observed
    return {"estimate_fusion_minus_control": float(gain.mean()),
            "ci95_patient_bootstrap": [float(np.quantile(bootstrap, 0.025)),
                                       float(np.quantile(bootstrap, 0.975))],
            "p_two_sided_sign_flip_uncorrected": (extreme + 1) / (SIGN_FLIPS + 1),
            "n_paired_patients": int(len(gain)), "bootstrap_replicates": BOOTSTRAPS,
            "sign_flip_replicates": SIGN_FLIPS, "independent_unit": "patient"}


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, (count - rank) * value)
        adjusted[name] = min(1.0, running)
    return adjusted


def aggregate(folds_root: Path, input_dir: Path, split_path: Path,
              config_path: Path, binding_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"scoring output already exists: {output_dir}")
    predictions, fold_receipts, sources = load_predictions(
        folds_root, split_path, input_dir, config_path, binding_path)
    per_seed = score_patients(predictions, input_dir)
    averaged = seed_average(per_seed)
    pivot = averaged.pivot(index=["fold", "patient_id"], columns="arm", values="spearman")
    if set(pivot.columns) != set(ARMS) or len(pivot) != 51:
        raise ValueError("paired score matrix does not cover all arms and patients")

    statistics: dict[str, Any] = {}
    for index, (name, (left, right)) in enumerate(CONTRASTS.items()):
        paired = pivot[[left, right]].dropna()
        result = paired_statistics((paired[left] - paired[right]).to_numpy(), seed=STAT_SEED + index)
        result["comparison"] = f"{left} minus {right}; positive favors F-F"
        result["left_arm_mean"] = float(paired[left].mean())
        result["right_arm_mean"] = float(paired[right].mean())
        statistics[name] = result
    adjusted = holm_adjust({name: item["p_two_sided_sign_flip_uncorrected"]
                            for name, item in statistics.items()})
    for name, value in adjusted.items():
        statistics[name]["p_holm_four_comparisons"] = value

    all_arm_valid = pivot[list(ARMS)].dropna()
    arm_summary = {}
    for arm in ARMS:
        valid = pivot[arm].dropna()
        arm_summary[arm] = {"mean_patient_spearman": float(valid.mean()),
                            "sd_between_patients": float(valid.std(ddof=1)) if len(valid) > 1 else None,
                            "n_valid_patients": int(len(valid))}
    output_dir.mkdir(parents=True, exist_ok=False)
    per_seed_path = output_dir / "patient_scores_by_seed.parquet"
    averaged_path = output_dir / "patient_scores_seed_averaged.parquet"
    predictions_path = output_dir / "outer_predictions_verified.parquet"
    paired_path = output_dir / "paired_patient_scores.parquet"
    per_seed.to_parquet(per_seed_path, index=False)
    averaged.to_parquet(averaged_path, index=False)
    predictions.to_parquet(predictions_path, index=False)
    paired = pivot.reset_index()
    paired.to_parquet(paired_path, index=False)
    report = {
        "schema_version": 1, "experiment_id": "EXP-048", "created_at_utc": utc_now(),
        "status": "FIVE_FOLDS_SCORED_PENDING_INDEPENDENT_REVIEW",
        "metric": "patient-macro within-patient Spearman over seven fixed observed keys; higher is better",
        "seed_aggregation": "score each arm and seed within patient, then average all three finite seed scores per patient",
        "outer_test_labels_used_only_for": "post-prediction scoring after all five fold receipts and keyed predictions passed verification; not used for fusion weights, candidate selection, arm selection, fold selection, or seed selection",
        "n_patients": int(pivot.shape[0]), "n_folds": 5,
        "n_patients_finite_for_all_five_arms": int(len(all_arm_valid)),
        "mean_patient_spearman_by_arm": arm_summary,
        "registered_paired_comparisons": statistics,
        "multiplicity": "Holm across the four registered F-F pairwise comparisons",
        "undefined_scores": {"per_seed": int(per_seed["spearman"].isna().sum()),
                              "seed_averaged": int(averaged["spearman"].isna().sum()),
                              "reason_counts_per_seed": per_seed["undefined_reason"].value_counts(dropna=False).to_dict()},
        "folds": fold_receipts, "source_sha256": sources,
        "output_sha256": {path.name: sha256(path) for path in
                          (per_seed_path, averaged_path, predictions_path, paired_path)},
    }
    atomic_json(output_dir / "run_summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds-root", type=Path, default=DEFAULT_FOLDS_ROOT)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--binding-receipt", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.folds_root, args.input_dir, args.split,
                               args.config, args.binding_receipt, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
