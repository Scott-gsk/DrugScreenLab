"""EXP-042 source cold-cell per-drug Ridge benchmark on the shared drug panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits


ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
MODEL_MEAN = "source_train_drug_mean"
MODEL_SELECTED = "inner_selected_source_model"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def macro_cell_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_id, group in predictions.groupby("sample_id", sort=True):
        n_drugs = int(group["drug_id"].nunique())
        observed = group["observed"].to_numpy(dtype=np.float64)
        predicted = group["predicted"].to_numpy(dtype=np.float64)
        if n_drugs < 3:
            value, reason = np.nan, "FEWER_THAN_THREE_DRUGS"
        elif not np.isfinite(observed).all() or not np.isfinite(predicted).all():
            value, reason = np.nan, "NONFINITE_VALUE"
        elif np.ptp(observed) == 0:
            value, reason = np.nan, "CONSTANT_OBSERVED"
        elif np.ptp(predicted) == 0:
            value, reason = np.nan, "CONSTANT_PREDICTED"
        else:
            value = float(spearmanr(observed, predicted).statistic)
            reason = None if np.isfinite(value) else "UNDEFINED_CORRELATION"
        rows.append({"sample_id": str(sample_id), "spearman": value,
                     "n_drugs": n_drugs, "reason": reason})
    return pd.DataFrame(rows, columns=["sample_id", "spearman", "n_drugs", "reason"])


def fit_expression_transform(expression: np.ndarray, train_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(expression[train_rows], dtype=np.float64)
    if np.isinf(train).any():
        raise ValueError("infinite training expression values")
    observed = np.isfinite(train)
    counts = observed.sum(axis=0)
    means = np.divide(np.where(observed, train, 0.0).sum(axis=0), counts,
                      out=np.zeros(train.shape[1], dtype=np.float64), where=counts > 0)
    filled = np.where(observed, train, means[None, :])
    center = filled.mean(axis=0)
    scale = filled.std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return center.astype(np.float32), scale.astype(np.float32)


def transform_expression(expression: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = np.asarray(expression, dtype=np.float32)
    if np.isinf(values).any():
        raise ValueError("infinite expression values")
    imputed = np.where(np.isfinite(values), values, center[None, :])
    result = (imputed - center[None, :]) / scale[None, :]
    if not np.isfinite(result).all():
        raise ValueError("expression transform produced non-finite values")
    return result.astype(np.float32, copy=False)


def training_drug_mean(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("drug_id", sort=True)["response"].mean()


def predict_mean(frame: pd.DataFrame, means: pd.Series) -> pd.DataFrame:
    result = frame[["sample_id", "drug_id", "response"]].copy()
    result["predicted"] = result["drug_id"].map(means)
    result = result.rename(columns={"response": "observed"})
    if result["predicted"].isna().any():
        raise ValueError("training-side drug mean lacks an evaluation drug")
    return result


def fit_predict_ridge(train: pd.DataFrame, evaluation: pd.DataFrame,
                      expression_scaled: np.ndarray, sample_position: dict[str, int],
                      alpha: float) -> pd.DataFrame:
    result = evaluation[["sample_id", "drug_id", "response"]].copy()
    result["predicted"] = np.nan
    for drug_id in sorted(evaluation["drug_id"].unique()):
        train_drug = train.loc[train["drug_id"].eq(drug_id)]
        eval_mask = result["drug_id"].eq(drug_id).to_numpy()
        if train_drug.empty or not eval_mask.any():
            raise ValueError(f"drug {drug_id} is missing from train or evaluation")
        train_rows = np.array([sample_position[str(x)] for x in train_drug["sample_id"]], dtype=np.int64)
        eval_rows = np.array([sample_position[str(x)] for x in result.loc[eval_mask, "sample_id"]], dtype=np.int64)
        model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr", max_iter=1000, tol=1e-4)
        model.fit(expression_scaled[train_rows], train_drug["response"].to_numpy(dtype=np.float64))
        result.loc[eval_mask, "predicted"] = model.predict(expression_scaled[eval_rows])
    result = result.rename(columns={"response": "observed"})
    if not np.isfinite(result["predicted"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Ridge produced non-finite predictions")
    return result


def _validate_partitions(split: dict, source_ids: set[str], pairs: pd.DataFrame) -> None:
    folds = split.get("source_folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("source split must contain five outer folds")
    test_sets = []
    for fold_index, fold in enumerate(folds):
        train_ids = set(map(str, fold["outer_train_ids"]))
        test_ids = set(map(str, fold["outer_test_ids"]))
        if train_ids & test_ids or train_ids | test_ids != source_ids:
            raise ValueError(f"outer fold {fold_index} does not partition source identities")
        fraction = next(item for item in fold["label_fractions"] if float(item["fraction"]) == 1.0)
        inner = fraction["inner_folds"]
        if len(inner) != 3:
            raise ValueError(f"outer fold {fold_index} must have three inner folds")
        validation_seen: set[str] = set()
        for inner_fold in inner:
            inner_train = set(map(str, inner_fold["train_ids"]))
            validation = set(map(str, inner_fold["validation_ids"]))
            if inner_train & validation or inner_train | validation != train_ids:
                raise ValueError(f"outer fold {fold_index} inner split is invalid")
            if validation_seen & validation:
                raise ValueError(f"outer fold {fold_index} repeats inner validation identity")
            validation_seen |= validation
        if validation_seen != train_ids:
            raise ValueError(f"outer fold {fold_index} inner validation does not cover training identities")
        test_sets.append(test_ids)
    if set().union(*test_sets) != source_ids or any(
        test_sets[left] & test_sets[right]
        for left in range(5) for right in range(left + 1, 5)
    ):
        raise ValueError("outer test folds do not partition source identities")
    if pairs.duplicated(["sample_id", "drug_id"]).any():
        raise ValueError("source response contains duplicate sample/drug keys")


def bootstrap_summary(differences: np.ndarray) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"estimate": None, "ci95_low": None, "ci95_high": None,
                "sign_flip_p_two_sided": None, "n_cells": 0}
    rng = np.random.default_rng(20260922)
    draws = np.empty(2000, dtype=np.float64)
    for index in range(len(draws)):
        draws[index] = values[rng.integers(0, len(values), len(values))].mean()
    signs = rng.choice(np.array([-1.0, 1.0]), size=(10000, len(values)))
    observed = abs(float(values.mean()))
    permuted = np.abs((signs * values[None, :]).mean(axis=1))
    return {"estimate": float(values.mean()),
            "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)),
            "sign_flip_p_two_sided": float((1 + np.count_nonzero(permuted >= observed)) / 10001),
            "n_cells": int(len(values)), "bootstrap_replicates": 2000,
            "sign_flip_replicates": 10000, "resampling_seed": 20260922}


def run(contract_dir: Path, source_npz: Path, split_path: Path,
        output_dir: Path, *, smoke: bool = False) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"run directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        manifest_path = contract_dir / "manifest.json"
        contract_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pairs = pd.read_parquet(contract_dir / "source_pairs.parquet")
        drugs = pd.read_parquet(contract_dir / "drugs.parquet")
        genes = json.loads((contract_dir / "genes.json").read_text(encoding="utf-8"))
        source = np.load(source_npz, allow_pickle=False)
        expression = np.asarray(source["expression"], dtype=np.float32)
        source_ids = source["sample_id"].astype(str).tolist()
        cell_ids = source["cell_line_id"].astype(str).tolist()
        sample_position = {sample_id: index for index, sample_id in enumerate(source_ids)}
        cell_position = {cell_id: index for index, cell_id in enumerate(cell_ids)}
        split = json.loads(split_path.read_text(encoding="utf-8"))
        _validate_partitions(split, set(cell_ids), pairs)
        if len(set(source_ids)) != len(source_ids) or len(set(cell_ids)) != len(cell_ids):
            raise ValueError("source sample or cell-line axis has duplicate identities")
        if set(pairs["sample_id"].astype(str)) - set(source_ids):
            raise ValueError("source response IDs do not map to the expression axis")
        if set(cell_ids) != set(source_ids):
            raise ValueError("source sample and cell-line identities are not one-to-one")
        if expression.shape != (len(source_ids), len(genes)) or len(genes) != 19091:
            raise ValueError("source expression shape differs from the frozen gene axis")
        if source["ordered_genes"].astype(str).tolist() != list(map(str, genes)):
            raise ValueError("source NPZ gene order differs from the frozen gene contract")
        if set(pairs["drug_id"].astype(str)) != set(drugs["drug_id"].astype(str)):
            raise ValueError("source response and drug contract axes differ")
        if pairs["response"].isna().any() or not np.isfinite(pairs["response"].to_numpy(dtype=float)).all():
            raise ValueError("source response contains missing or non-finite values")
        if pairs.duplicated(["sample_id", "drug_id"]).any():
            raise ValueError("duplicate source sample/drug response key")

        sample_to_cell = dict(zip(source_ids, cell_ids, strict=True))
        pairs = pairs.copy()
        pairs["sample_id"] = pairs["sample_id"].astype(str)
        pairs["drug_id"] = pairs["drug_id"].astype(str)
        pairs["cell_line_id"] = pairs["sample_id"].map(sample_to_cell)
        candidate_rows: list[dict[str, Any]] = []
        inner_prediction_frames: list[pd.DataFrame] = []
        selected_rows: list[pd.DataFrame] = []
        mean_rows: list[pd.DataFrame] = []
        outer_receipts: list[dict[str, Any]] = []

        alphas = ALPHAS[:1] if smoke else ALPHAS
        outer_folds = split["source_folds"][:1] if smoke else split["source_folds"]
        for outer in outer_folds:
            fold_id = int(outer["fold"])
            outer_train_ids = set(map(str, outer["outer_train_ids"]))
            outer_test_ids = set(map(str, outer["outer_test_ids"]))
            fraction = next(item for item in outer["label_fractions"]
                            if float(item["fraction"]) == 1.0)
            fold_candidates: dict[str, list[float]] = {MODEL_MEAN: []}
            fold_candidates.update({f"ridge_alpha_{alpha:g}": [] for alpha in alphas})

            inner_folds = fraction["inner_folds"][:1] if smoke else fraction["inner_folds"]
            for inner in inner_folds:
                inner_id = int(inner["inner_fold"])
                train_ids = set(map(str, inner["train_ids"]))
                validation_ids = set(map(str, inner["validation_ids"]))
                train = pairs.loc[pairs["cell_line_id"].isin(train_ids)]
                validation = pairs.loc[pairs["cell_line_id"].isin(validation_ids)]
                if train.empty or validation.empty:
                    raise ValueError(f"empty training/validation rows in fold {fold_id}/{inner_id}")
                # Fit the transform on the distinct training cells, never on pair-expanded rows.
                distinct_train_cells = sorted(train_ids)
                expression_rows = np.array([cell_position[cell_id] for cell_id in distinct_train_cells],
                                           dtype=np.int64)
                center, scale = fit_expression_transform(expression, expression_rows)
                expression_scaled = transform_expression(expression, center, scale)
                baseline = predict_mean(validation, training_drug_mean(train))
                baseline_scores = macro_cell_scores(baseline)
                baseline_metric = float(baseline_scores["spearman"].mean())
                if not np.isfinite(baseline_metric):
                    raise ValueError(f"no valid inner baseline score for fold {fold_id}/{inner_id}")
                fold_candidates[MODEL_MEAN].append(baseline_metric)
                candidate_rows.append({"outer_fold": fold_id, "inner_fold": inner_id,
                                       "candidate": MODEL_MEAN, "validation_mean_cell_spearman": baseline_metric,
                                       "n_validation_cells_scored": int(baseline_scores["spearman"].notna().sum()),
                                       "n_train_cells": len(train_ids), "n_validation_cells": len(validation_ids)})
                baseline.insert(0, "outer_fold", fold_id)
                baseline.insert(1, "inner_fold", inner_id)
                baseline.insert(2, "candidate", MODEL_MEAN)
                inner_prediction_frames.append(baseline)
                for alpha in alphas:
                    predicted = fit_predict_ridge(train, validation, expression_scaled,
                                                  sample_position, alpha)
                    scores = macro_cell_scores(predicted)
                    metric = float(scores["spearman"].mean())
                    if not np.isfinite(metric):
                        raise ValueError(f"no valid validation cell score for fold {fold_id}/{inner_id}/alpha {alpha}")
                    fold_candidates[f"ridge_alpha_{alpha:g}"].append(metric)
                    candidate_rows.append({"outer_fold": fold_id, "inner_fold": inner_id,
                                           "candidate": f"ridge_alpha_{alpha:g}",
                                           "validation_mean_cell_spearman": metric,
                                           "n_validation_cells_scored": int(scores["spearman"].notna().sum()),
                                           "n_train_cells": len(train_ids), "n_validation_cells": len(validation_ids)})
                    predicted.insert(0, "outer_fold", fold_id)
                    predicted.insert(1, "inner_fold", inner_id)
                    predicted.insert(2, "candidate", f"ridge_alpha_{alpha:g}")
                    inner_prediction_frames.append(predicted)

            means = {name: float(np.mean(values)) for name, values in fold_candidates.items()}
            best_metric = max(means.values())
            tied = [name for name, value in means.items() if np.isclose(value, best_metric, rtol=0, atol=1e-12)]
            if MODEL_MEAN in tied:
                selected = MODEL_MEAN
            else:
                selected = max(tied, key=lambda name: float(name.removeprefix("ridge_alpha_")))
            outer_train = pairs.loc[pairs["cell_line_id"].isin(outer_train_ids)]
            outer_test = pairs.loc[pairs["cell_line_id"].isin(outer_test_ids)]
            train_expression_rows = np.array([cell_position[cell_id] for cell_id in sorted(outer_train_ids)],
                                             dtype=np.int64)
            center, scale = fit_expression_transform(expression, train_expression_rows)
            expression_scaled = transform_expression(expression, center, scale)
            baseline_test = predict_mean(outer_test, training_drug_mean(outer_train))
            if selected == MODEL_MEAN:
                selected_test = baseline_test.copy()
            else:
                alpha = float(selected.removeprefix("ridge_alpha_"))
                selected_test = fit_predict_ridge(outer_train, outer_test, expression_scaled,
                                                  sample_position, alpha)
            baseline_scores = macro_cell_scores(baseline_test)
            selected_scores = macro_cell_scores(selected_test)
            baseline_test.insert(0, "outer_fold", fold_id)
            baseline_test.insert(1, "model", MODEL_MEAN)
            selected_test.insert(0, "outer_fold", fold_id)
            selected_test.insert(1, "model", MODEL_SELECTED)
            mean_rows.append(baseline_test)
            selected_rows.append(selected_test)
            score_frame = baseline_scores[["sample_id", "spearman", "n_drugs", "reason"]].merge(
                selected_scores[["sample_id", "spearman", "n_drugs", "reason"]],
                on="sample_id", suffixes=("_mean", "_selected"), validate="one_to_one")
            score_frame["outer_fold"] = fold_id
            score_frame["difference_selected_minus_mean"] = (
                score_frame["spearman_selected"] - score_frame["spearman_mean"])
            score_frame.to_parquet(output_dir / f"fold_{fold_id}_unit_scores.parquet", index=False)
            outer_receipts.append({
                "outer_fold": fold_id, "n_outer_train_cells": len(outer_train_ids),
                "n_outer_test_cells": len(outer_test_ids), "n_test_cells_with_any_pair": int(outer_test["cell_line_id"].nunique()),
                "n_test_cells_scored_mean": int(baseline_scores["spearman"].notna().sum()),
                "n_test_cells_scored_selected": int(selected_scores["spearman"].notna().sum()),
                "selected_candidate": selected, "selected_inner_mean_cell_spearman": means[selected],
                "inner_candidate_mean_spearman": means,
                "n_outer_test_pairs": len(outer_test),
                "n_outer_test_drugs": int(outer_test["drug_id"].nunique()),
                "train_only_imputation_and_standardization": True,
            })

        predictions = pd.concat(mean_rows + selected_rows, ignore_index=True)
        predictions.to_parquet(output_dir / "predictions.parquet", index=False)
        pd.DataFrame(candidate_rows).to_parquet(output_dir / "inner_candidate_scores.parquet", index=False)
        pd.concat(inner_prediction_frames, ignore_index=True).to_parquet(
            output_dir / "inner_predictions.parquet", index=False)
        scores = pd.concat([
            pd.read_parquet(output_dir / f"fold_{item['outer_fold']}_unit_scores.parquet")
            for item in outer_receipts
        ], ignore_index=True)
        differences = scores["difference_selected_minus_mean"].to_numpy(dtype=np.float64)
        summary = {
            "experiment_id": "EXP-042", "run_id": output_dir.name,
            "status": ("SMOKE_ONLY_NOT_SCIENTIFIC_RESULT" if smoke else
                       "FIVE_FOLDS_COMPLETE_PENDING_INDEPENDENT_VERIFICATION"),
            "endpoint": "source_cell_macro_spearman_gain_over_train_drug_mean",
            "source_endpoint": "GDSC2 fitted AUC transformed to 1-AUC; higher means more sensitive",
            "n_source_cell_axis": len(source_ids), "n_source_cells_with_any_common_drug_pair": int(pairs["sample_id"].nunique()),
            "n_source_drugs": int(pairs["drug_id"].nunique()), "n_source_pairs": len(pairs),
            "n_source_cells_at_least_three_test_drugs": int((scores["n_drugs_mean"] >= 3).sum()),
            "model_selection": "mean of inner cell-macro Spearman over three fixed inner folds; mean baseline wins exact ties",
            "outer_fold_receipts": outer_receipts,
            "paired_primary_statistics": None if smoke else bootstrap_summary(differences),
            "inputs": {
                "contract_manifest_sha256": sha256(manifest_path),
                "source_pairs_sha256": sha256(contract_dir / "source_pairs.parquet"),
                "source_npz_sha256": sha256(source_npz),
                "split_sha256": sha256(split_path),
                "genes_sha256": sha256(contract_dir / "genes.json"),
            },
            "code_sha256": {"modeling_script": sha256(Path(__file__).resolve())},
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "scipy": scipy.__version__,
                            "scikit_learn": sklearn.__version__},
            "resources": {"device": "cpu", "blas_threads": 1,
                          "elapsed_seconds": time.perf_counter() - started},
        }
        files = ["predictions.parquet", "inner_candidate_scores.parquet", "inner_predictions.parquet"] + [
            f"fold_{item['outer_fold']}_unit_scores.parquet" for item in outer_receipts]
        summary["output_files_sha256"] = {name: sha256(output_dir / name) for name in files}
        _write_json_atomic(output_dir / "run_summary.json", summary)
        _write_json_atomic(output_dir / "run_provenance.json", {
            "experiment_id": "EXP-042", "run_id": output_dir.name,
            "started_utc": started_utc,
            "code_sha256": summary["code_sha256"], "inputs": summary["inputs"],
            "configuration": {"alphas": list(alphas), "outer_folds": len(outer_folds),
                              "inner_folds": 1 if smoke else 3, "smoke_only": smoke,
                              "bootstrap_replicates": 2000, "sign_flip_replicates": 10000,
                              "seed": 20260922},
            "argv": os.sys.argv,
        })
        return summary
    except Exception as error:
        _write_json_atomic(output_dir / "FAILURE.json", {
            "experiment_id": "EXP-042", "run_id": output_dir.name,
            "error_type": type(error).__name__, "error": str(error),
            "elapsed_seconds": time.perf_counter() - started,
        })
        raise


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--source-npz", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="run one outer/inner fold and one alpha for engineering validation only")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = run(args.contract_dir, args.source_npz, args.split, args.output_dir, smoke=args.smoke)
    print(json.dumps({"run_id": result["run_id"], "status": result["status"],
                      "primary": result["paired_primary_statistics"],
                      "output_files_sha256": result["output_files_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
