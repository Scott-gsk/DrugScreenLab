"""Independently verify EXP-039 split, preprocessing, inner selection, and predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ("huber", "b5_rank")
SEEDS = (20260922, 20260923, 20260924)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std <= 1e-8] = 1.0
    return mean, std


def control_threshold(contract_dir: Path, cache_path: Path, train_cells: set[str]) -> np.ndarray:
    controls = pd.read_parquet(contract_dir / "controls.parquet")
    controls = controls.loc[controls["cell_id"].isin(train_cells)]
    cache = np.load(cache_path, mmap_mode="r")
    rows = []
    for _, frame in controls.groupby("control_group_id", sort=False):
        values = np.asarray(cache[frame["cache_row"].to_numpy(np.int64)], dtype=np.float32)
        rows.append(values.std(axis=0, dtype=np.float64).astype(np.float32))
    if not rows:
        raise ValueError("no outer-training control groups")
    threshold = np.median(np.stack(rows), axis=0).astype(np.float32)
    if threshold.shape != (978,) or not np.isfinite(threshold).all():
        raise ValueError("invalid independent control threshold")
    return threshold


def verify_selection(fold_dir: Path, summary: dict) -> None:
    table = pd.read_parquet(fold_dir / "inner_selection.parquet")
    if len(table) != 36 or table.duplicated(["candidate", "inner_fold"]).any():
        raise ValueError("incomplete or duplicate inner candidate results")
    expected = {f"r{rank}_wd{decay:g}_rank{weight:g}"
                for rank in (16, 32) for decay in (1e-4, 1e-3) for weight in (0.0, 0.1, 0.3)}
    if (set(table["candidate"]) != expected or set(table["inner_fold"]) != {0, 1, 2}
            or not np.isfinite(table["best_validation_reversal_spearman"].to_numpy(float)).all()):
        raise ValueError("inner results differ from the registered candidate panel")
    for arm, allowed_weights in (("huber", {0.0}), ("b5_rank", {0.1, 0.3})):
        selected = summary["selected"][arm]["candidate"]
        key = f"r{selected['rank']}_wd{selected['weight_decay']:g}_rank{selected['ranking_weight']:g}"
        allowed = table.loc[table["ranking_weight"].isin(allowed_weights)]
        means = allowed.groupby("candidate")["best_validation_reversal_spearman"].mean()
        best = sorted(means.index, key=lambda name: (-means[name], name))[0]
        if key != best:
            raise ValueError(f"{arm} model does not match frozen inner-metric selection")
    if (summary["selected"]["huber"]["candidate"]["ranking_weight"] != 0.0
            or summary["selected"]["b5_rank"]["candidate"]["ranking_weight"] not in {0.1, 0.3}):
        raise ValueError("Huber and ranking arms have inconsistent objectives")


def verify(condition_dir: Path, input_dir: Path, contract_dir: Path, cache_path: Path,
           split_path: Path, fold_dir: Path) -> dict:
    summary = json.loads((fold_dir / "fold_summary.json").read_text(encoding="utf-8"))
    provenance_path = fold_dir.parent / "run_provenance.json"
    if not provenance_path.is_file() or summary.get("run_provenance_sha256") != sha256(provenance_path):
        raise ValueError("fold result is not bound to the captured run provenance")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("experiment_id") != "EXP-039":
        raise ValueError("run provenance identifies a different experiment")
    for name, digest in {**provenance["code_sha256"], **provenance["input_sha256"]}.items():
        path = Path(name)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"run provenance file changed after capture: {name}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    fold = int(summary["fold"])
    if (summary["experiment_id"] != "EXP-039"
            or summary["status"] != "FOLD_COMPLETE_PENDING_INDEPENDENT_REVIEW"
            or fold not in range(5) or summary["seeds"] != list(SEEDS)
            or summary["split_sha256"] != sha256(split_path)
            or summary["condition_manifest_sha256"] != sha256(condition_dir / "manifest.json")
            or summary["input_manifest_sha256"] != sha256(input_dir / "manifest.json")
            or summary["contract_sha256"] != sha256(contract_dir / "verification.json")):
        raise ValueError("fold receipt does not match frozen EXP-039 inputs")
    for name, digest in summary["files_sha256"].items():
        if digest != sha256(fold_dir / name):
            raise ValueError(f"fold checksum mismatch: {name}")
    verify_selection(fold_dir, summary)

    keys = pd.read_parquet(condition_dir / "conditions.parquet")
    observed_all = np.load(condition_dir / "condition_delta.npy", mmap_mode="r")
    if observed_all.shape != (len(keys), 978) or not keys["condition_id"].is_unique:
        raise ValueError("condition and delta axes are invalid")
    outer = split["folds"][fold]
    train_cells, test_cells = set(outer["outer_train_ids"]), set(outer["outer_test_ids"])
    if train_cells & test_cells or train_cells | test_cells != set(keys["cell_id"]):
        raise ValueError("outer fold does not partition all cells")
    train_index = np.flatnonzero(keys["cell_id"].isin(train_cells).to_numpy())
    test_all = np.flatnonzero(keys["cell_id"].isin(test_cells).to_numpy())
    seen_drugs = set(keys.iloc[train_index]["pert_id"])
    test_index = test_all[keys.iloc[test_all]["pert_id"].isin(seen_drugs).to_numpy()]
    expected_ids = keys.iloc[test_index]["condition_id"].to_numpy(np.int64)
    saved_ids = np.load(fold_dir / "test_condition_ids.npy")
    if not np.array_equal(saved_ids, expected_ids) or len(saved_ids) != summary["n_test_conditions"]:
        raise ValueError("prediction IDs escape the unseen-cell, seen-drug test set")

    state_all = np.load(input_dir / "condition_control.npy", mmap_mode="r")
    exposure = pd.read_parquet(input_dir / "condition_exposure.parquet").iloc[train_index][
        ["log10_dose_um", "time_hours"]].to_numpy(np.float32)
    state_mean, state_scale = scale(np.asarray(state_all[train_index], dtype=np.float32))
    exposure_mean, exposure_scale = scale(exposure)
    train_delta = np.asarray(observed_all[train_index], dtype=np.float32)
    _, gene_scale = scale(train_delta)
    del train_delta
    threshold = control_threshold(contract_dir, cache_path, train_cells)
    prep = np.load(fold_dir / "preprocess.npz")
    scale_errors = {
        "state_mean": float(np.max(np.abs(prep["state_mean"] - state_mean))),
        "state_scale": float(np.max(np.abs(prep["state_scale"] - state_scale))),
        "exposure_mean": float(np.max(np.abs(prep["exposure_mean"] - exposure_mean))),
        "exposure_scale": float(np.max(np.abs(prep["exposure_scale"] - exposure_scale))),
        "gene_scale": float(np.max(np.abs(prep["gene_scale"] - gene_scale))),
        "direction_threshold": float(np.max(np.abs(prep["direction_threshold"] - threshold))),
    }
    if any(value != 0.0 for value in scale_errors.values()):
        raise ValueError(f"preprocessing is not an independent outer-train fit: {scale_errors}")

    receipts = {(row["arm"], row["seed"]): row for row in summary["final_receipts"]}
    expected_receipts = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if set(receipts) != expected_receipts or len(summary["final_receipts"]) != len(expected_receipts):
        raise ValueError("missing or duplicate final model receipts")
    observed = np.asarray(observed_all[test_index], dtype=np.float32)
    cell_rows = []
    max_cell_score_recompute_error = 0.0
    for arm in ARMS:
        for seed in SEEDS:
            pred_path = fold_dir / f"predicted_delta_{arm}_seed{seed}.npy"
            model_path = fold_dir / f"model_{arm}_seed{seed}.pt"
            receipt = receipts[(arm, seed)]
            if receipt["prediction_sha256"] != sha256(pred_path) or receipt["model_sha256"] != sha256(model_path):
                raise ValueError("prediction or checkpoint receipt checksum mismatch")
            predicted = np.load(pred_path, mmap_mode="r")
            if predicted.shape != observed.shape or not np.isfinite(predicted).all():
                raise ValueError("prediction matrix has invalid dimensions or values")
            error = np.square((np.asarray(predicted, dtype=np.float64) - observed.astype(np.float64))
                              / gene_scale.astype(np.float64)).mean(axis=1)
            frame = keys.iloc[test_index][["cell_id", "pert_id", "condition_id"]].copy()
            frame["error"] = error
            by_drug = frame.groupby(["cell_id", "pert_id"], sort=True, as_index=False).agg(
                drug_error=("error", "mean"), n_conditions=("condition_id", "size"))
            by_cell = by_drug.groupby("cell_id", sort=True, as_index=False).agg(
                normalized_mse=("drug_error", "mean"), n_drugs=("pert_id", "size"),
                n_conditions=("n_conditions", "sum"))
            if len(by_cell) != len(test_cells) or not np.isfinite(by_cell["normalized_mse"]).all():
                raise ValueError("independent cell error scores are incomplete")
            by_cell.insert(0, "fold", fold)
            by_cell.insert(1, "arm", arm)
            by_cell.insert(2, "seed", seed)
            cell_rows.extend(by_cell.to_dict("records"))
    cell_scores = pd.DataFrame(cell_rows)
    cell_path = fold_dir / "cell_scores_delta.parquet"
    cell_scores.to_parquet(cell_path, index=False)
    return {
        "status": "PASS", "fold": fold,
        "run_provenance_sha256": sha256(provenance_path),
        "n_test_conditions": len(test_index), "n_test_cells": len(test_cells),
        "n_verified_models": len(expected_receipts),
        "train_only_preprocess_max_abs_error": scale_errors,
        "prediction_cell_scores": "recomputed from frozen keys, measured deltas, and saved predictions",
        "cell_scores_sha256": sha256(cell_path),
        "files_sha256": {name: sha256(fold_dir / name) for name in summary["files_sha256"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.condition_dir, args.input_dir, args.contract_dir,
                    args.cache, args.split, args.fold_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
