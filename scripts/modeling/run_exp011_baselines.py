"""Run leakage-safe, fixed simple baselines for one EXP-011 grouped fold.

This is intentionally a small comparator, not a second data pipeline.  It
loads the same indexed-pair contract and calls the same ``build_fold_indices``
function as the PharmaFormer runner.  Its only fitted objects (HVG selection,
gene scaling, drug vocabulary and regressors) are fit from the train partition.
No validation or test response is used by any preprocessing or model step.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from drug_screen.evaluation.exp011 import PREDICTION_COLUMNS  # noqa: E402
from drug_screen.foundation.pharmaformer_exp011 import EXP011_EXPERIMENT_ID, sha256_file  # noqa: E402
from drug_screen.modeling.exp011_data import IndexedPairData, build_fold_indices, load_indexed_pair_data  # noqa: E402


TOP_MANIFEST_FORMAT = "exp011_indexed_drug_response_v1"
DEFAULT_FEATURE_COUNT = 512
RIDGE_ALPHA = 10.0
_ROLE_FOR_STAGE = {"source": "gdsc_cold_cell", "target": "pdo_primary"}


@dataclass(frozen=True)
class BaselinePreprocessor:
    """Train-only feature contract for the fixed Ridge/RF comparators."""

    selected_gene_indices: np.ndarray
    selected_gene_ids: tuple[str, ...]
    gene_mean: np.ndarray
    gene_scale: np.ndarray
    train_drug_ids: tuple[str, ...]
    train_sample_ids: tuple[str, ...]

    def transform(self, data: IndexedPairData, pair_indices: Sequence[int]) -> np.ndarray:
        pairs = np.asarray(pair_indices, dtype=np.int64)
        if pairs.ndim != 1 or not len(pairs):
            raise ValueError("baseline transform requires non-empty pair indices")
        sample_indices = data.pair_sample_idx[pairs]
        gene = data.expression[sample_indices][:, self.selected_gene_indices].astype(np.float32, copy=False)
        gene = (gene - self.gene_mean) / self.gene_scale
        drug_index = {value: index for index, value in enumerate(self.train_drug_ids)}
        drug = np.zeros((len(pairs), len(self.train_drug_ids)), dtype=np.float32)
        for row, index in enumerate(data.pair_drug_idx[pairs].tolist()):
            column = drug_index.get(str(data.drug_id[index]))
            if column is not None:
                drug[row, column] = 1.0
        return np.concatenate([gene, drug], axis=1, dtype=np.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "exp011_baseline_train_only_preprocessor_v1",
            "requested_feature_count": DEFAULT_FEATURE_COUNT,
            "selected_feature_count": int(len(self.selected_gene_indices)),
            "selected_gene_ids": list(self.selected_gene_ids),
            "train_drug_ids": list(self.train_drug_ids),
            "train_sample_ids": list(self.train_sample_ids),
        }


def _default_data_root() -> Path:
    return Path(os.environ.get("DRUGSCREEN_DATA_ROOT", ROOT / "data"))


def _resolve_relative(parent: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (parent / path).resolve()


def resolve_cohort_manifest(manifest_path: str | Path, *, cohort_id: str | None) -> tuple[Path, str, bool]:
    """Resolve either one raw manifest or an aggregate manifest plus cohort ID.

    Returns ``(raw_manifest_or_npz, resolved_cohort_id, from_aggregate)``.
    A direct NPZ stays supported for local development, but callers can reject
    it in formal mode before any model artifacts are written.
    """

    path = Path(manifest_path).resolve()
    if path.suffix.lower() == ".npz":
        if not cohort_id:
            raise ValueError("direct development NPZ input requires --cohort-id")
        return path, str(cohort_id), False
    if not path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    if payload.get("schema") == "drugscreenlab.exp011.raw-indexed-pairs.v1":
        resolved = str(payload.get("cohort_id", "")).strip()
        if not resolved:
            if not cohort_id:
                raise ValueError("raw manifest without cohort_id requires --cohort-id")
            resolved = str(cohort_id)
        if cohort_id and str(cohort_id) != resolved:
            raise ValueError(f"--cohort-id {cohort_id} does not match raw manifest cohort_id {resolved}")
        return path, resolved, False
    if payload.get("format") != TOP_MANIFEST_FORMAT:
        raise ValueError("manifest must be an EXP-011 raw manifest or cohort_index_manifest.json")
    if not cohort_id:
        raise ValueError("aggregate cohort_index_manifest.json requires --cohort-id")
    records = payload.get("cohorts")
    if not isinstance(records, list):
        raise ValueError("aggregate manifest has no cohorts list")
    matches = [record for record in records if isinstance(record, dict) and record.get("cohort_id") == cohort_id]
    if len(matches) != 1:
        raise ValueError(f"aggregate manifest has no unique cohort_id={cohort_id}")
    bundle = matches[0].get("bundle")
    if not isinstance(bundle, dict) or not isinstance(bundle.get("raw_manifest"), str):
        raise ValueError(f"aggregate cohort {cohort_id} has no raw_manifest bundle")
    return _resolve_relative(path.parent, bundle["raw_manifest"]), str(cohort_id), True


def select_train_variance_genes(
    data: IndexedPairData,
    train_pair_indices: Sequence[int],
    *,
    feature_count: int = DEFAULT_FEATURE_COUNT,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Pick top train-sample variance genes, deterministic under tied variance."""

    if feature_count < 1:
        raise ValueError("feature_count must be positive")
    pairs = np.asarray(train_pair_indices, dtype=np.int64)
    train_samples = np.unique(data.pair_sample_idx[pairs])
    if len(train_samples) < 2:
        raise ValueError("train-only variance selection requires at least two unique samples")
    variance = np.var(data.expression[train_samples], axis=0, ddof=0)
    order = sorted(range(data.gene_count), key=lambda index: (-float(variance[index]), str(data.ordered_genes[index])))
    selected = np.asarray(order[: min(int(feature_count), data.gene_count)], dtype=np.int64)
    identifiers = tuple(str(data.ordered_genes[index]) for index in selected)
    return selected, identifiers, train_samples


def fit_preprocessor(
    data: IndexedPairData,
    train_pair_indices: Sequence[int],
    *,
    feature_count: int = DEFAULT_FEATURE_COUNT,
) -> BaselinePreprocessor:
    """Fit HVG selection, gene standardisation and drug one-hot on train only."""

    pairs = np.asarray(train_pair_indices, dtype=np.int64)
    selected, identifiers, train_samples = select_train_variance_genes(data, pairs, feature_count=feature_count)
    train_expression = data.expression[train_samples][:, selected].astype(np.float64, copy=False)
    mean = np.mean(train_expression, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.std(train_expression, axis=0, dtype=np.float64).astype(np.float32)
    scale[scale == 0.0] = 1.0
    train_drug_indices = np.unique(data.pair_drug_idx[pairs])
    train_drugs = tuple(sorted(str(data.drug_id[index]) for index in train_drug_indices))
    if not train_drugs:
        raise ValueError("baseline fit requires at least one train drug")
    return BaselinePreprocessor(
        selected_gene_indices=selected,
        selected_gene_ids=identifiers,
        gene_mean=mean,
        gene_scale=scale,
        train_drug_ids=train_drugs,
        train_sample_ids=tuple(str(data.sample_id[index]) for index in train_samples),
    )


def train_only_drug_median(data: IndexedPairData, train_pair_indices: Sequence[int], test_pair_indices: Sequence[int]) -> np.ndarray:
    """Drug-median control with global-train-median fallback for unseen drugs."""

    train = np.asarray(train_pair_indices, dtype=np.int64)
    test = np.asarray(test_pair_indices, dtype=np.int64)
    values = pd.DataFrame({"drug_index": data.pair_drug_idx[train], "response": data.pair_response[train]})
    medians = values.groupby("drug_index", observed=True)["response"].median().to_dict()
    fallback = float(np.median(data.pair_response[train]))
    return np.asarray([medians.get(int(index), fallback) for index in data.pair_drug_idx[test]], dtype=np.float32)


def fit_predict_baselines(
    data: IndexedPairData,
    *,
    train_pair_indices: Sequence[int],
    test_pair_indices: Sequence[int],
    seed: int,
    with_random_forest: bool = False,
    rf_estimators: int = 200,
) -> tuple[BaselinePreprocessor, dict[str, np.ndarray]]:
    """Fit the fixed baseline family without ever consulting held-out labels."""

    if rf_estimators < 1:
        raise ValueError("rf_estimators must be positive")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    train = np.asarray(train_pair_indices, dtype=np.int64)
    test = np.asarray(test_pair_indices, dtype=np.int64)
    preprocessor = fit_preprocessor(data, train)
    x_train = preprocessor.transform(data, train)
    x_test = preprocessor.transform(data, test)
    y_train = data.pair_response[train].astype(np.float32, copy=False)
    ridge = Ridge(alpha=RIDGE_ALPHA).fit(x_train, y_train)
    predictions = {
        "B0_drug_median": train_only_drug_median(data, train, test),
        "B1_ridge": np.asarray(ridge.predict(x_test), dtype=np.float32),
    }
    if with_random_forest:
        forest = RandomForestRegressor(
            n_estimators=int(rf_estimators),
            min_samples_leaf=2,
            random_state=int(seed),
            n_jobs=1,
        ).fit(x_train, y_train)
        predictions["B2_random_forest"] = np.asarray(forest.predict(x_test), dtype=np.float32)
    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise RuntimeError("baseline produced non-finite held-out predictions")
    return preprocessor, predictions


def prediction_frame(
    data: IndexedPairData,
    pair_indices: Sequence[int],
    prediction: Sequence[float],
    *,
    model_id: str,
    fold_id: int,
    evaluation_role: str,
) -> pd.DataFrame:
    """Emit exactly the frozen 18-column EXP-011 evaluator schema."""

    pairs = np.asarray(pair_indices, dtype=np.int64)
    values = np.asarray(prediction, dtype=np.float32)
    if len(pairs) != len(values):
        raise ValueError("prediction count must equal test pair count")
    samples = data.pair_sample_idx[pairs]
    drugs = data.pair_drug_idx[pairs]
    frame = pd.DataFrame(
        {
            "experiment_id": EXP011_EXPERIMENT_ID,
            "model_id": model_id,
            "evaluation_role": evaluation_role,
            "cohort_id": data.pair_cohort_id[pairs],
            "fold_id": int(fold_id),
            "split": "test",
            "patient_id": data.patient_id[samples],
            "sample_id": data.sample_id[samples],
            "pdo_id": data.pdo_id[samples],
            "cell_line_id": data.cell_line_id[samples],
            "drug_id": data.drug_id[drugs],
            "inchikey": data.inchikey[drugs],
            "drug_scaffold": data.drug_scaffold[drugs],
            "response_name": data.pair_response_name[pairs],
            "response_scale": data.pair_response_scale[pairs],
            "higher_is_sensitive": data.pair_higher_is_sensitive[pairs],
            "y_true": data.pair_response[pairs],
            "y_pred": values,
        }
    )
    if frame.columns.tolist() != list(PREDICTION_COLUMNS):
        raise AssertionError("baseline prediction schema diverges from EXP-011 evaluator contract")
    return frame


def _preprocessor_digest(preprocessor: BaselinePreprocessor) -> str:
    value = {
        "gene_ids": list(preprocessor.selected_gene_ids),
        "mean": preprocessor.gene_mean.tolist(),
        "scale": preprocessor.gene_scale.tolist(),
        "drug_ids": list(preprocessor.train_drug_ids),
        "sample_ids": list(preprocessor.train_sample_ids),
    }
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _require_role_for_stage(stage: str, evaluation_role: str | None) -> str:
    expected = _ROLE_FOR_STAGE[stage]
    role = expected if evaluation_role is None else str(evaluation_role)
    if role != expected:
        raise ValueError(
            f"{stage} baselines use the shared grouped fold and must emit {expected}; "
            f"{role} requires a different external/scaffold split contract"
        )
    return role


def run(args: argparse.Namespace) -> dict[str, object]:
    raw_manifest, cohort_id, from_aggregate = resolve_cohort_manifest(args.manifest, cohort_id=args.cohort_id)
    if args.formal and raw_manifest.suffix.lower() == ".npz":
        raise ValueError("formal EXP-011 baselines reject development direct NPZ input; use cohort_index_manifest.json")
    data = load_indexed_pair_data(raw_manifest)
    if args.formal and bool(data.manifest.get("development_input", False)):
        raise ValueError("formal EXP-011 baselines reject development direct NPZ input")
    observed_cohorts = set(data.pair_cohort_id.astype(str).tolist())
    if observed_cohorts != {cohort_id}:
        raise ValueError(f"resolved cohort_id {cohort_id} does not match pair cohort IDs {sorted(observed_cohorts)}")
    if not data.pair_higher_is_sensitive.all():
        raise ValueError("baseline input labels must be oriented so higher means more sensitive")
    evaluation_role = _require_role_for_stage(args.stage, args.evaluation_role)
    fold = build_fold_indices(data, stage=args.stage, fold_id=args.fold_id, seed=args.seed)
    preprocessor, predictions = fit_predict_baselines(
        data,
        train_pair_indices=fold.train,
        test_pair_indices=fold.test,
        seed=args.seed,
        with_random_forest=args.with_random_forest,
        rf_estimators=args.rf_estimators,
    )
    frames = [
        prediction_frame(
            data, fold.test, values, model_id=model_id, fold_id=fold.fold_id, evaluation_role=evaluation_role
        )
        for model_id, values in predictions.items()
    ]
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["model_id", "sample_id", "drug_id"]).any():
        raise RuntimeError("baseline output contains duplicate model/sample/drug predictions")
    args.output.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output / f"baselines_{cohort_id}_fold{fold.fold_id}_predictions.parquet"
    preprocessing_path = args.output / f"baselines_{cohort_id}_fold{fold.fold_id}_preprocessing.json"
    summary_path = args.output / f"baselines_{cohort_id}_fold{fold.fold_id}_summary.json"
    result.to_parquet(prediction_path, index=False)
    preprocessing_path.write_text(json.dumps(preprocessor.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "experiment_id": EXP011_EXPERIMENT_ID,
        "format": "exp011_fixed_baselines_run_v1",
        "cohort_id": cohort_id,
        "stage": args.stage,
        "evaluation_role": evaluation_role,
        "fold_id": int(fold.fold_id),
        "seed": int(args.seed),
        "models": list(predictions),
        "ridge_alpha": RIDGE_ALPHA,
        "random_forest": {
            "enabled": bool(args.with_random_forest),
            "estimators": int(args.rf_estimators) if args.with_random_forest else None,
            "random_state": int(args.seed) if args.with_random_forest else None,
        },
        "train_only_preprocessing": {
            "feature_selection": "top_512_variance_genes_over_unique_train_samples",
            "gene_standardisation": "mean_and_std_over_unique_train_samples",
            "drug_one_hot": "vocabulary_from_train_pairs; unseen_test_drugs_all_zero",
            "test_labels_read_during_fit": False,
            "preprocessor_sha256": _preprocessor_digest(preprocessor),
            "selected_feature_count": int(len(preprocessor.selected_gene_indices)),
        },
        "partitions": {
            "train_pair_count": int(len(fold.train)),
            "validation_pair_count_unused": int(len(fold.validation)),
            "test_pair_count": int(len(fold.test)),
            "group_field": fold.group_field,
            "stratification_field": fold.stratification_field,
        },
        "manifest": str(raw_manifest),
        "manifest_sha256": sha256_file(raw_manifest),
        "aggregate_manifest_input": bool(from_aggregate),
        "formal": bool(args.formal),
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "preprocessing_path": str(preprocessing_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=_default_data_root() / "processed/exp011/cohort_index_manifest.json")
    parser.add_argument("--cohort-id", help="required with aggregate cohort_index_manifest.json")
    parser.add_argument("--stage", choices=("source", "target"), required=True)
    parser.add_argument("--fold-id", type=int, choices=range(5), required=True)
    parser.add_argument("--evaluation-role", choices=("gdsc_cold_cell", "pdo_primary"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--with-random-forest", action="store_true")
    parser.add_argument("--rf-estimators", type=int, default=200)
    parser.add_argument("--formal", action="store_true", help="reject development direct NPZ inputs")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
