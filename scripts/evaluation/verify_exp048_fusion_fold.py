"""Independently verify one EXP-048 fold without reading target responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from aggregate_exp048_fusion import (
    ARMS,
    DEFAULT_BINDING,
    DEFAULT_CONFIG,
    DEFAULT_FOLDS_ROOT,
    DEFAULT_INPUT_DIR,
    DEFAULT_SPLIT,
    EXPECTED,
    ROOT,
    SEEDS,
    _assert_hash,
    _load_json,
    atomic_json,
    sha256,
    utc_now,
)


def verify_fold(fold: int, folds_root: Path = DEFAULT_FOLDS_ROOT,
                input_dir: Path = DEFAULT_INPUT_DIR, split_path: Path = DEFAULT_SPLIT,
                config_path: Path = DEFAULT_CONFIG,
                binding_path: Path = DEFAULT_BINDING) -> dict[str, Any]:
    """Check frozen identities, selected-fit provenance, checkpoints and predictions.

    This routine intentionally never opens patient_pairs.parquet and calculates
    no target response metrics. It checks only the fixed identity/key mask and
    label-blind model outputs.
    """
    if fold not in range(5):
        raise ValueError("fold must be one of the five frozen outer folds")
    _assert_hash(split_path, EXPECTED["split"], "split")
    input_manifest = input_dir / "manifest.json"
    _assert_hash(input_manifest, EXPECTED["input_manifest"], "target input manifest")
    _assert_hash(binding_path, EXPECTED["binding_receipt"], "target identity/key binding receipt")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != "EXP-048" or len(config["drug_panel"]["target_inchikeys"]) != 9:
        raise ValueError("configuration differs from the frozen nine-drug EXP-048 panel")
    split = _load_json(split_path)
    binding = _load_json(binding_path)
    key_info = binding["target_key_mask"]
    key_path = ROOT / str(key_info["path"])
    if not key_path.is_file() or sha256(key_path) != key_info.get("sha256"):
        raise ValueError("frozen patient-drug key mask is missing or changed")
    full_mask = pd.read_csv(key_path)
    full_mask["patient_id"] = full_mask["patient_id"].astype(str)
    full_mask["drug_id"] = full_mask["drug_id"].astype(str)
    panel = set(str(value) for value in config["drug_panel"]["target_inchikeys"])
    if (full_mask.duplicated(["patient_id", "drug_id"]).any()
            or len(full_mask) != 459
            or full_mask["patient_id"].nunique() != 51
            or not full_mask.groupby("patient_id").size().eq(9).all()
            or not set(full_mask["drug_id"]).issubset(panel)):
        raise ValueError("fixed key mask must contain all 51x9 unique patient-panel slots")
    key_mask = full_mask.loc[full_mask["target_key_present"].astype(bool), ["patient_id", "drug_id"]]
    if (key_mask.duplicated(["patient_id", "drug_id"]).any()
            or len(key_mask) != 357
            or not key_mask.groupby("patient_id").size().eq(7).all()):
        raise ValueError("fixed key intersection must contain seven unique observed keys per patient")
    observed_keys = set(map(tuple, key_mask.to_numpy(dtype=str)))

    fold_dir = folds_root / f"fold_{fold}"
    summary_path = fold_dir / "fold_summary.json"
    manifest_path = fold_dir / "run_manifest.json"
    inner_path = fold_dir / "inner_selection.json"
    prediction_path = fold_dir / "predictions.parquet"
    if not all(path.is_file() for path in (summary_path, manifest_path, inner_path, prediction_path)):
        raise FileNotFoundError(f"fold {fold} lacks a complete prediction-phase artifact set")
    summary, manifest = _load_json(summary_path), _load_json(manifest_path)
    if (summary.get("experiment_id") != "EXP-048"
            or summary.get("fold") != fold
            or summary.get("phase") != "predictions"
            or summary.get("status") != "FOLD_PREDICTIONS_COMPLETE_PENDING_INDEPENDENT_VERIFICATION"
            or manifest.get("fold") != fold
            or "predictions" not in manifest.get("phases_completed", [])):
        raise ValueError(f"fold {fold} is not a complete frozen prediction run")
    if summary.get("inner_selection_sha256") != sha256(inner_path):
        raise ValueError(f"fold {fold} inner selection hash differs from its summary")
    hashes = summary.get("input_and_code_sha256", {})
    if (not hashes or manifest.get("input_and_code_sha256") != hashes
            or hashes.get("config") != sha256(config_path)
            or hashes.get("split") != sha256(split_path)
            or hashes.get("input_manifest") != sha256(input_manifest)
            or hashes.get("binding_receipt") != sha256(binding_path)):
        raise ValueError(f"fold {fold} manifest/input hashes do not match the frozen scoring contract")

    outer_list = [str(value) for value in split["target_folds"][fold]["outer_test_ids"]]
    outer_patients = set(outer_list)
    all_target_patients = set(key_mask["patient_id"])
    if len(outer_patients) != len(outer_list) or not outer_patients.issubset(all_target_patients):
        raise ValueError(f"fold {fold} outer patients are duplicated or outside the frozen target cohort")
    expected_keys = {key for key in observed_keys if key[0] in outer_patients}

    if summary.get("predictions_sha256") != sha256(prediction_path):
        raise ValueError(f"fold {fold} predictions hash differs from its summary")
    prediction = pd.read_parquet(prediction_path)
    required = {"fold", "arm", "seed", "patient_id", "drug_id", "prediction"}
    if not required.issubset(prediction.columns):
        raise ValueError(f"fold {fold} predictions lack frozen keyed columns")
    prediction["patient_id"] = prediction["patient_id"].astype(str)
    prediction["drug_id"] = prediction["drug_id"].astype(str)
    prediction["seed"] = pd.to_numeric(prediction["seed"], errors="raise").astype(int)
    values = pd.to_numeric(prediction["prediction"], errors="raise").to_numpy(dtype=np.float64)
    if (prediction.duplicated(["fold", "arm", "seed", "patient_id", "drug_id"]).any()
            or set(prediction["arm"]) != set(ARMS)
            or set(prediction["seed"]) != set(SEEDS)
            or not prediction["fold"].eq(fold).all()
            or set(prediction["patient_id"]) != outer_patients
            or not np.isfinite(values).all()):
        raise ValueError(f"fold {fold} prediction arms, seeds, identities or values are incomplete/invalid")
    actual_keys = set(map(tuple, prediction[["patient_id", "drug_id"]].drop_duplicates().to_numpy(dtype=str)))
    if actual_keys != expected_keys:
        raise ValueError(f"fold {fold} prediction keys do not equal the frozen observed-key intersection")
    expected_rows = len(expected_keys) * len(ARMS) * len(SEEDS)
    if len(prediction) != expected_rows:
        raise ValueError(f"fold {fold} expected {expected_rows} prediction rows, found {len(prediction)}")

    final_receipts = summary.get("final_fit_receipts", [])
    expected_units = {(arm, seed) for arm in ARMS for seed in SEEDS}
    actual_units = {(str(item.get("arm")), int(item.get("seed"))) for item in final_receipts}
    if len(final_receipts) != len(expected_units) or actual_units != expected_units:
        raise ValueError(f"fold {fold} final-fit receipts do not cover all frozen arm/seed pairs")
    checkpoint_receipts = []
    for item in final_receipts:
        checkpoint_path = ROOT / str(item["checkpoint"])
        if not checkpoint_path.is_file() or sha256(checkpoint_path) != item.get("checkpoint_sha256"):
            raise ValueError(f"fold {fold} final checkpoint missing or changed: {item.get('checkpoint')}")
        checkpoint_receipts.append({
            "arm": item["arm"], "seed": int(item["seed"]),
            "checkpoint": str(item["checkpoint"]),
            "checkpoint_sha256": item["checkpoint_sha256"],
        })

    receipt = {
        "schema_version": 1,
        "experiment_id": "EXP-048",
        "fold": fold,
        "status": "PASS",
        "verification_scope": "frozen-input identity, prediction-key completeness/finiteness, inner-selection binding and final-checkpoint hashes",
        "n_outer_patients": len(outer_patients),
        "n_observed_patient_drug_keys": len(expected_keys),
        "n_prediction_rows": int(len(prediction)),
        "n_arms": len(ARMS),
        "seeds": list(SEEDS),
        "n_checkpoints_verified": len(checkpoint_receipts),
        "checkpoints": checkpoint_receipts,
        "summary_sha256": sha256(summary_path),
        "manifest_sha256": sha256(manifest_path),
        "inner_selection_sha256": sha256(inner_path),
        "predictions_sha256": sha256(prediction_path),
        "key_mask_sha256": sha256(key_path),
        "frozen_input_hashes": {
            "config": sha256(config_path),
            "split": sha256(split_path),
            "input_manifest": sha256(input_manifest),
            "binding_receipt": sha256(binding_path),
        },
        "outer_response_values_read": False,
        "source_patient_pairs_opened": False,
        "outer_metrics_calculated": False,
        "verified_at_utc": utc_now(),
    }
    output_path = fold_dir / "independent_verification.json"
    atomic_json(output_path, receipt)
    receipt["receipt_sha256"] = sha256(output_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--folds-root", type=Path, default=DEFAULT_FOLDS_ROOT)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--binding-receipt", type=Path, default=DEFAULT_BINDING)
    args = parser.parse_args()
    print(json.dumps(verify_fold(args.fold, args.folds_root, args.input_dir,
                                 args.split, args.config, args.binding_receipt),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
