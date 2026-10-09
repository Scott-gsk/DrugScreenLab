"""Independently verify EXP-042 identities, selection scores, predictions, and statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits


MEAN = "source_train_drug_mean"
SELECTED = "inner_selected_source_model"
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
CANDIDATES = (MEAN,) + tuple(f"ridge_alpha_{alpha:g}" for alpha in ALPHAS)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _score_cells(frame: pd.DataFrame) -> pd.DataFrame:
    output = []
    for sample_id, group in frame.groupby("sample_id", sort=True):
        count = int(group["drug_id"].nunique())
        y = group["observed"].to_numpy(dtype=np.float64)
        prediction = group["predicted"].to_numpy(dtype=np.float64)
        if count < 3:
            value, reason = np.nan, "FEWER_THAN_THREE_DRUGS"
        elif not np.isfinite(y).all() or not np.isfinite(prediction).all():
            value, reason = np.nan, "NONFINITE_VALUE"
        elif np.ptp(y) == 0:
            value, reason = np.nan, "CONSTANT_OBSERVED"
        elif np.ptp(prediction) == 0:
            value, reason = np.nan, "CONSTANT_PREDICTED"
        else:
            value = float(spearmanr(y, prediction).statistic)
            reason = None if np.isfinite(value) else "UNDEFINED_CORRELATION"
        output.append({"sample_id": str(sample_id), "spearman": value,
                       "n_drugs": count, "reason": reason})
    return pd.DataFrame(output)


def _reference_transform(expression: np.ndarray, train_rows: np.ndarray) -> np.ndarray:
    train = np.asarray(expression[train_rows], dtype=np.float64)
    if np.isinf(train).any():
        raise ValueError("infinite training expression value")
    observed = np.isfinite(train)
    count = observed.sum(axis=0)
    center = np.zeros(train.shape[1], dtype=np.float64)
    valid = count > 0
    center[valid] = np.where(observed[:, valid], train[:, valid], 0.0).sum(axis=0) / count[valid]
    filled = np.where(observed, train, center[None, :])
    scale = filled.std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    # Match the model input's recorded float32 storage boundary.
    center32, scale32 = center.astype(np.float32), scale.astype(np.float32)
    values = np.asarray(expression, dtype=np.float32)
    imputed = np.where(np.isfinite(values), values, center32[None, :])
    return ((imputed - center32[None, :]) / scale32[None, :]).astype(np.float32)


def _bootstrap(values: np.ndarray) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not len(data):
        return {"estimate": None, "ci95_low": None, "ci95_high": None,
                "sign_flip_p_two_sided": None, "n_cells": 0}
    rng = np.random.default_rng(20260922)
    draws = np.empty(2000, dtype=np.float64)
    for index in range(len(draws)):
        draws[index] = data[rng.integers(0, len(data), len(data))].mean()
    signs = rng.choice(np.array([-1.0, 1.0]), size=(10000, len(data)))
    statistic = abs(float(data.mean()))
    null = np.abs((signs * data[None, :]).mean(axis=1))
    return {"estimate": float(data.mean()),
            "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)),
            "sign_flip_p_two_sided": float((1 + np.count_nonzero(null >= statistic)) / 10001),
            "n_cells": int(len(data)), "bootstrap_replicates": 2000,
            "sign_flip_replicates": 10000, "resampling_seed": 20260922}


def _check_keys(actual: pd.DataFrame, expected: pd.DataFrame,
                key_columns: list[str], label: str) -> None:
    if actual.duplicated(key_columns).any() or expected.duplicated(key_columns).any():
        raise ValueError(f"duplicate keys in {label}")
    left = actual[key_columns].sort_values(key_columns).reset_index(drop=True)
    right = expected[key_columns].sort_values(key_columns).reset_index(drop=True)
    if not left.equals(right):
        raise ValueError(f"key set differs in {label}")


def verify(run_dir: Path, contract_dir: Path, source_npz: Path, split_path: Path) -> dict[str, Any]:
    receipt_path = run_dir / "verification.json"
    if receipt_path.exists():
        raise FileExistsError(f"verification receipt already exists: {receipt_path}")
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    if summary.get("experiment_id") != "EXP-042" or summary.get("status") != \
            "FIVE_FOLDS_COMPLETE_PENDING_INDEPENDENT_VERIFICATION":
        raise ValueError("run summary is incomplete or belongs to another experiment")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    pairs = pd.read_parquet(contract_dir / "source_pairs.parquet").copy()
    pairs["sample_id"] = pairs["sample_id"].astype(str)
    pairs["drug_id"] = pairs["drug_id"].astype(str)
    source = np.load(source_npz, allow_pickle=False)
    expression = np.asarray(source["expression"], dtype=np.float32)
    source_ids = source["sample_id"].astype(str).tolist()
    cell_ids = source["cell_line_id"].astype(str).tolist()
    genes = json.loads((contract_dir / "genes.json").read_text(encoding="utf-8"))
    if source["ordered_genes"].astype(str).tolist() != list(map(str, genes)):
        raise ValueError("source expression gene axis differs from the frozen contract")
    sample_position = {sample: position for position, sample in enumerate(source_ids)}
    cell_position = {cell: position for position, cell in enumerate(cell_ids)}
    sample_to_cell = dict(zip(source_ids, cell_ids, strict=True))
    if len(sample_position) != len(source_ids) or len(cell_position) != len(cell_ids):
        raise ValueError("duplicate source expression identity")
    pairs["cell_line_id"] = pairs["sample_id"].map(sample_to_cell)
    if pairs["cell_line_id"].isna().any() or set(cell_ids) != set(source_ids):
        raise ValueError("source pair identities do not map one-to-one to expression")
    _check_hashes = {
        "contract_manifest_sha256": _sha256(contract_dir / "manifest.json"),
        "source_pairs_sha256": _sha256(contract_dir / "source_pairs.parquet"),
        "source_npz_sha256": _sha256(source_npz),
        "split_sha256": _sha256(split_path),
        "genes_sha256": _sha256(contract_dir / "genes.json"),
    }
    for key, value in _check_hashes.items():
        if summary["inputs"].get(key) != value:
            raise ValueError(f"input checksum mismatch: {key}")
    model_script = Path(__file__).resolve().parents[2] / "scripts" / "modeling" / "exp042_source_ridge.py"
    if summary["code_sha256"].get("modeling_script") != _sha256(model_script):
        raise ValueError("training code checksum differs from the completed run")
    provenance = json.loads((run_dir / "run_provenance.json").read_text(encoding="utf-8"))
    if provenance.get("experiment_id") != "EXP-042" or provenance.get("run_id") != run_dir.name:
        raise ValueError("run provenance identifies a different experiment or run")
    if provenance.get("code_sha256") != summary.get("code_sha256") or \
            provenance.get("inputs") != summary.get("inputs"):
        raise ValueError("run provenance differs from the completed summary")
    if provenance.get("configuration", {}).get("smoke_only") is not False:
        raise ValueError("formal verifier refuses an engineering smoke run")
    for filename, digest in summary["output_files_sha256"].items():
        if _sha256(run_dir / filename) != digest:
            raise ValueError(f"output checksum mismatch: {filename}")

    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    inner = pd.read_parquet(run_dir / "inner_predictions.parquet")
    candidate_receipts = pd.read_parquet(run_dir / "inner_candidate_scores.parquet")
    if set(predictions["outer_fold"].astype(int)) != set(range(5)):
        raise ValueError("outer predictions do not cover exactly five folds")
    if set(predictions["model"]) != {MEAN, SELECTED}:
        raise ValueError("outer predictions contain unexpected model arms")
    if predictions.duplicated(["outer_fold", "model", "sample_id", "drug_id"]).any():
        raise ValueError("duplicate outer prediction key")
    if set(inner["outer_fold"].astype(int)) != set(range(5)) or \
            set(inner["inner_fold"].astype(int)) != {0, 1, 2} or \
            set(inner["candidate"]) != set(CANDIDATES):
        raise ValueError("inner predictions do not cover the registered folds and candidates")
    if inner.duplicated(["outer_fold", "inner_fold", "candidate", "sample_id", "drug_id"]).any():
        raise ValueError("duplicate inner prediction key")
    if len(candidate_receipts) != 5 * 3 * len(CANDIDATES):
        raise ValueError("inner candidate receipt count is incomplete")
    if candidate_receipts.duplicated(["outer_fold", "inner_fold", "candidate"]).any():
        raise ValueError("duplicate inner candidate receipt")
    outer_receipts = summary.get("outer_fold_receipts", [])
    if len(outer_receipts) != 5 or {int(item["outer_fold"]) for item in outer_receipts} != set(range(5)):
        raise ValueError("outer fold receipts do not cover exactly five folds")
    fold_rows = []
    all_unit_scores = []
    final_prediction_max_abs_error = 0.0
    inner_score_max_abs_error = 0.0

    with threadpool_limits(limits=1):
        for outer in split["source_folds"]:
            fold = int(outer["fold"])
            outer_train = set(map(str, outer["outer_train_ids"]))
            outer_test = set(map(str, outer["outer_test_ids"]))
            test_pairs = pairs.loc[pairs["cell_line_id"].isin(outer_test)].copy()
            expected_output = []
            for model_name in (MEAN, SELECTED):
                output = predictions.loc[
                    predictions["outer_fold"].eq(fold) & predictions["model"].eq(model_name)
                ].copy()
                _check_keys(output, test_pairs, ["sample_id", "drug_id"], f"outer {fold}/{model_name}")
                joined = output.merge(test_pairs[["sample_id", "drug_id", "response"]],
                                      on=["sample_id", "drug_id"], suffixes=("_saved", "_input"),
                                      validate="one_to_one")
                if not np.array_equal(joined["observed"].to_numpy(), joined["response"].to_numpy()):
                    raise ValueError(f"observed labels differ from frozen source input in outer fold {fold}")
                if output["sample_id"].map(sample_to_cell).isin(outer_train).any():
                    raise ValueError(f"outer training identity appears in test predictions for fold {fold}")
                expected_output.append(output)

            mean_output = expected_output[0]
            train_pairs = pairs.loc[pairs["cell_line_id"].isin(outer_train)]
            means = train_pairs.groupby("drug_id", sort=True)["response"].mean()
            expected_mean = mean_output["drug_id"].map(means).to_numpy(dtype=np.float64)
            mean_error = float(np.max(np.abs(expected_mean - mean_output["predicted"].to_numpy(dtype=np.float64))))
            final_prediction_max_abs_error = max(final_prediction_max_abs_error, mean_error)
            if mean_error > 1e-10:
                raise ValueError(f"outer train-only mean prediction mismatch in fold {fold}")

            fraction = next(item for item in outer["label_fractions"] if float(item["fraction"]) == 1.0)
            inner_fold_scores: dict[str, list[float]] = {candidate: [] for candidate in CANDIDATES}
            for inner_fold in fraction["inner_folds"]:
                inner_id = int(inner_fold["inner_fold"])
                validation_ids = set(map(str, inner_fold["validation_ids"]))
                validation_pairs = pairs.loc[pairs["cell_line_id"].isin(validation_ids)]
                for candidate in CANDIDATES:
                    group = inner.loc[
                        inner["outer_fold"].eq(fold)
                        & inner["inner_fold"].eq(inner_id)
                        & inner["candidate"].eq(candidate)
                    ].copy()
                    _check_keys(group, validation_pairs, ["sample_id", "drug_id"],
                                f"inner {fold}/{inner_id}/{candidate}")
                    if group["sample_id"].map(sample_to_cell).isin(outer_test | (set(map(str, outer["outer_train_ids"])) - validation_ids)).any():
                        raise ValueError(f"non-validation identity in inner prediction {fold}/{inner_id}/{candidate}")
                    scores = _score_cells(group)
                    metric = float(scores["spearman"].mean())
                    saved = candidate_receipts.loc[
                        candidate_receipts["outer_fold"].eq(fold)
                        & candidate_receipts["inner_fold"].eq(inner_id)
                        & candidate_receipts["candidate"].eq(candidate)
                    ]
                    if len(saved) != 1:
                        raise ValueError(f"inner candidate receipt count invalid for {fold}/{inner_id}/{candidate}")
                    saved_metric = float(saved.iloc[0]["validation_mean_cell_spearman"])
                    error = abs(metric - saved_metric)
                    inner_score_max_abs_error = max(inner_score_max_abs_error, error)
                    if error > 1e-12:
                        raise ValueError(f"inner candidate metric mismatch for {fold}/{inner_id}/{candidate}")
                    if int(saved.iloc[0]["n_validation_cells_scored"]) != int(scores["spearman"].notna().sum()):
                        raise ValueError(f"inner candidate denominator mismatch for {fold}/{inner_id}/{candidate}")
                    inner_fold_scores[candidate].append(metric)

            candidate_means = {name: float(np.mean(values)) for name, values in inner_fold_scores.items()}
            best = max(candidate_means.values())
            tied = [name for name, score in candidate_means.items()
                    if np.isclose(score, best, rtol=0, atol=1e-12)]
            selected = MEAN if MEAN in tied else max(
                tied, key=lambda name: float(name.removeprefix("ridge_alpha_")))
            receipt = next(item for item in outer_receipts if int(item["outer_fold"]) == fold)
            if receipt["selected_candidate"] != selected:
                raise ValueError(f"inner selection decision mismatch for outer fold {fold}")
            for candidate in CANDIDATES:
                if not np.isclose(receipt["inner_candidate_mean_spearman"][candidate],
                                  candidate_means[candidate], rtol=0, atol=1e-12):
                    raise ValueError(f"inner candidate fold mean mismatch for outer fold {fold}/{candidate}")

            if selected == MEAN:
                selected_output = expected_output[1]
                mean_lookup = mean_output.set_index(["sample_id", "drug_id"])["predicted"]
                selected_lookup = selected_output.set_index(["sample_id", "drug_id"])["predicted"]
                if not np.array_equal(mean_lookup.sort_index().to_numpy(), selected_lookup.sort_index().to_numpy()):
                    raise ValueError(f"selected mean predictions differ from baseline in outer fold {fold}")
            else:
                alpha = float(selected.removeprefix("ridge_alpha_"))
                selected_output = expected_output[1]
                train_pairs = pairs.loc[pairs["cell_line_id"].isin(outer_train)]
                train_rows = np.array([cell_position[cell] for cell in sorted(outer_train)], dtype=np.int64)
                transformed = _reference_transform(expression, train_rows)
                expected_values = np.empty(len(selected_output), dtype=np.float64)
                for drug_id in sorted(selected_output["drug_id"].unique()):
                    train_drug = train_pairs.loc[train_pairs["drug_id"].eq(drug_id)]
                    mask = selected_output["drug_id"].eq(drug_id).to_numpy()
                    train_x = np.array([transformed[sample_position[sample]]
                                        for sample in train_drug["sample_id"]], dtype=np.float32)
                    test_x = np.array([transformed[sample_position[sample]]
                                       for sample in selected_output.loc[mask, "sample_id"]], dtype=np.float32)
                    model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr", max_iter=1000, tol=1e-4)
                    model.fit(train_x, train_drug["response"].to_numpy(dtype=np.float64))
                    expected_values[mask] = model.predict(test_x)
                observed_values = selected_output["predicted"].to_numpy(dtype=np.float64)
                error = float(np.max(np.abs(expected_values - observed_values)))
                final_prediction_max_abs_error = max(final_prediction_max_abs_error, error)
                if not np.allclose(expected_values, observed_values, rtol=1e-4, atol=2e-4):
                    raise ValueError(f"independent final Ridge refit differs in outer fold {fold}: {error}")

            mean_scores = _score_cells(mean_output)
            selected_scores = _score_cells(selected_output)
            if int(receipt["n_test_cells_scored_mean"]) != int(mean_scores["spearman"].notna().sum()) or \
                    int(receipt["n_test_cells_scored_selected"]) != int(selected_scores["spearman"].notna().sum()):
                raise ValueError(f"outer score denominator mismatch for fold {fold}")
            score_path = run_dir / f"fold_{fold}_unit_scores.parquet"
            saved_scores = pd.read_parquet(score_path)
            recalculated = mean_scores.merge(selected_scores, on="sample_id", suffixes=("_mean", "_selected"))
            recalculated["difference_selected_minus_mean"] = (
                recalculated["spearman_selected"] - recalculated["spearman_mean"])
            _check_keys(saved_scores, recalculated, ["sample_id"], f"fold {fold} unit-score")
            saved_scores = saved_scores.sort_values("sample_id").reset_index(drop=True)
            recalculated = recalculated.sort_values("sample_id").reset_index(drop=True)
            for column in ("spearman_mean", "spearman_selected", "difference_selected_minus_mean"):
                if not np.allclose(saved_scores[column].to_numpy(dtype=float),
                                   recalculated[column].to_numpy(dtype=float),
                                   rtol=0, atol=1e-12, equal_nan=True):
                    raise ValueError(f"saved cell ranking score differs in fold {fold}: {column}")
            for column in ("n_drugs_mean", "n_drugs_selected", "reason_mean", "reason_selected"):
                if not saved_scores[column].fillna("<NA>").equals(
                    recalculated[column].fillna("<NA>")
                ):
                    raise ValueError(f"saved cell score metadata differs in fold {fold}: {column}")
            all_unit_scores.append(saved_scores)
            fold_rows.append({"outer_fold": fold, "n_test_keys": len(test_pairs),
                              "n_scored_cells": int(recalculated["spearman_mean"].notna().sum()),
                              "selected_candidate": selected,
                              "max_inner_metric_abs_error": max(
                                  abs(candidate_means[name] - receipt["inner_candidate_mean_spearman"][name])
                                  for name in CANDIDATES)})

    scores = pd.concat(all_unit_scores, ignore_index=True).sort_values(
        ["outer_fold", "sample_id"]).reset_index(drop=True)
    stats = _bootstrap(scores["difference_selected_minus_mean"].to_numpy(dtype=np.float64))
    reported = summary["paired_primary_statistics"]
    for key in ("estimate", "ci95_low", "ci95_high", "sign_flip_p_two_sided"):
        if not np.isclose(stats[key], reported[key], rtol=0, atol=1e-12):
            raise ValueError(f"primary statistic mismatch: {key}")
    if stats["n_cells"] != reported["n_cells"]:
        raise ValueError("primary statistic denominator mismatch")

    result = {
        "experiment_id": "EXP-042", "run_id": run_dir.name,
        "status": "PASS", "n_verified_outer_folds": len(fold_rows),
        "fold_receipts": fold_rows, "max_inner_metric_abs_error": inner_score_max_abs_error,
        "max_final_prediction_abs_error": final_prediction_max_abs_error,
        "primary_statistics_recomputed": stats,
        "checked_input_sha256": _check_hashes,
        "verified_output_file_count": len(summary["output_files_sha256"]),
        "verifier_sha256": _sha256(Path(__file__).resolve()),
    }
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(receipt_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--source-npz", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = verify(args.run_dir, args.contract_dir, args.source_npz, args.split)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
