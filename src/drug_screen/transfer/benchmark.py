"""Development-only, admission-gated nested-CV transfer benchmarks.

This module deliberately consumes an indexed, registered cohort rather than
discovering files.  It is a runner primitive, not evidence that a cohort has
been admitted or that a development result is external validation.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from .chemistry import morgan_fingerprint
from .cohorts import load_registered_cohort, sha256
from .metrics_v2 import evaluate_predictions, ndcg_at_k
from .preprocessing import ExpressionPreprocessor
from .splits import assign_patient_folds


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def require_development_admission(path: str | Path, *, cohort_id: str, registry: str | Path) -> dict[str, Any]:
    """Validate the explicit, hash-bound admission receipt before any load."""
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    if receipt.get("cohort_id") != cohort_id or receipt.get("status") != "DEVELOPMENT_ADMITTED":
        raise ValueError("admission must name the cohort and DEVELOPMENT_ADMITTED status")
    expected = receipt.get("registry_sha256", receipt.get("registry_hash"))
    actual = sha256(Path(registry))
    if not isinstance(expected, str) or expected != actual:
        raise ValueError("admission registry hash does not match the supplied registry")
    return receipt


def _identity(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _table(bundle: Any) -> pd.DataFrame:
    samples = bundle.samples.copy()
    if "patient_id" not in samples:
        raise ValueError("development cohort lacks patient_id")
    samples["sample_id"] = samples.sample_id.map(lambda value: _identity(value, "sample_id"))
    samples["patient_id"] = samples.patient_id.map(lambda value: _identity(value, "patient_id"))
    if "pdo_id" in samples:
        model = samples.pdo_id.where(samples.pdo_id.notna() & samples.pdo_id.astype(str).str.strip().ne(""), samples.sample_id).astype(str)
        samples["model_id"] = model.map(lambda value: _identity(value, "model_id"))
    else:
        samples["model_id"] = samples.sample_id
    expression_index = {str(sample): index for index, sample in enumerate(bundle.sample_ids)}
    samples["expression_index"] = samples.sample_id.map(expression_index)
    if samples.expression_index.isna().any():
        raise ValueError("sample metadata does not align with expression")
    pairs = bundle.pairs.copy()
    pairs["sample_id"] = pairs.sample_id.map(lambda value: _identity(value, "sample_id"))
    pairs["drug_id"] = pairs.drug_id.map(lambda value: _identity(value, "drug_id"))
    pairs["response"] = pd.to_numeric(pairs.response, errors="raise").astype(float)
    table = pairs.merge(samples[["sample_id", "patient_id", "model_id", "expression_index"]], on="sample_id", how="left", validate="many_to_one")
    if table[["patient_id", "model_id", "expression_index"]].isna().any().any():
        raise ValueError("response pair is missing sample metadata")
    if table.duplicated(["patient_id", "model_id", "drug_id"]).any():
        raise ValueError("benchmark requires unique biological model-drug responses")
    return table


def _fit_b0(train: pd.DataFrame) -> dict[str, Any]:
    # Patient means are formed before the across-patient drug mean, preventing
    # a patient with multiple PDO replicates from receiving extra B0 weight.
    reduced = train.groupby(["patient_id", "drug_id"], as_index=False).response.mean()
    means = reduced.groupby("drug_id").response.mean().to_dict()
    return {"drug_means": {str(key): float(value) for key, value in means.items()}, "global_mean": float(reduced.response.mean())}


def _predict_b0(model: Mapping[str, Any], drugs: pd.Series) -> np.ndarray:
    return np.asarray([model["drug_means"].get(str(drug), model["global_mean"]) for drug in drugs], dtype=float)


def _fit_b1(features: np.ndarray, train: pd.DataFrame, *, alpha: float, min_train: int) -> dict[str, Any]:
    fallback = _fit_b0(train)
    models: dict[str, tuple[float, np.ndarray]] = {}
    for drug, rows in train.groupby("drug_id", sort=True):
        indices = rows.expression_index.to_numpy(dtype=int)
        if len(indices) < min_train:
            continue
        x = features[indices]
        y = rows.response.to_numpy(float)
        # Solve in sample space: gene-space normal equations can allocate GBs.
        x_mean, y_mean = x.mean(axis=0), y.mean()
        centered = x - x_mean
        weights = centered.T @ np.linalg.solve(
            centered @ centered.T + alpha * np.eye(len(x)), y - y_mean)
        models[str(drug)] = (float(y_mean - x_mean @ weights), weights)
    return {"fallback": fallback, "models": models, "alpha": float(alpha)}


def _predict_b1(model: Mapping[str, Any], features: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    fallback = _predict_b0(model["fallback"], rows.drug_id)
    answer = fallback.copy()
    for position, row in enumerate(rows.itertuples(index=False)):
        current = model["models"].get(str(row.drug_id))
        if current is not None:
            answer[position] = current[0] + features[int(row.expression_index)] @ current[1]
    return answer


def _score(rows: pd.DataFrame, prediction: np.ndarray) -> float:
    values = []
    for (patient, _), group in rows.assign(prediction=prediction).groupby(["patient_id", "model_id"], sort=True):
        if len(group) >= 10:
            value = ndcg_at_k(group.response.to_numpy(float), group.prediction.to_numpy(float), k=5)
            if value is not None and np.isfinite(value):
                values.append((patient, value))
    if not values:
        return float("-inf")
    return float(pd.DataFrame(values, columns=["patient", "score"]).groupby("patient").score.mean().mean())


def _inner_partitions(train: pd.DataFrame, *, folds: int, seed: int):
    assignment = assign_patient_folds(train.patient_id.astype(str).tolist(), n_folds=folds, seed=seed)
    for fold in range(folds):
        valid_patients = {patient for patient, number in assignment.patient_to_test_fold.items() if number == fold}
        yield train.loc[~train.patient_id.isin(valid_patients)], train.loc[train.patient_id.isin(valid_patients)]


def _select_alpha(train: pd.DataFrame, expression: np.ndarray, genes: np.ndarray, config: Mapping[str, Any], *, seed: int) -> float:
    alphas = [float(value) for value in config.get("b1_alphas", [1, 10, 100])]
    if alphas != [1.0, 10.0, 100.0]:
        raise ValueError("B1 alpha search must be exactly [1, 10, 100]")
    scores = {alpha: [] for alpha in alphas}
    for inner_train, inner_valid in _inner_partitions(train, folds=int(config.get("inner_folds", 3)), seed=seed):
        fit_rows = np.unique(inner_train.expression_index.to_numpy(dtype=int))
        prep = ExpressionPreprocessor.fit(expression[fit_rows], genes, n_features=config.get("n_features"))
        x = prep.transform(expression, genes)
        for alpha in alphas:
            scores[alpha].append(_score(inner_valid, _predict_b1(_fit_b1(x, inner_train, alpha=alpha, min_train=int(config.get("b1_min_train", 2))), x, inner_valid)))
    return min(alphas, key=lambda alpha: (-np.mean(scores[alpha]), alpha))


def _fingerprints(bundle: Any, drugs: pd.Series) -> dict[str, np.ndarray]:
    chemical = bundle.drugs.copy()
    field = next((name for name in ("smiles", "canonical_isomeric_smiles") if name in chemical), None)
    if field is None:
        raise ValueError("M1 requires registered drug SMILES")
    lookup = dict(zip(chemical.drug_id.astype(str), chemical[field], strict=True))
    return {str(drug): morgan_fingerprint(str(lookup[str(drug)]), radius=2, n_bits=2048).astype(float) for drug in drugs.unique()}


def _fit_m1(features: np.ndarray, fingerprints: Mapping[str, np.ndarray], train: pd.DataFrame, params: Mapping[str, Any]):
    from .models import DualEncoderPairModel
    model = DualEncoderPairModel(**params)
    idx = train.expression_index.to_numpy(int)
    vectors = np.vstack([fingerprints[str(drug)] for drug in train.drug_id])
    return model.fit(features[idx], vectors, train.response.to_numpy(float), pdo_ids=train.model_id.astype(str).to_numpy())


def _predict_m1(model: Any, features: np.ndarray, fingerprints: Mapping[str, np.ndarray], rows: pd.DataFrame) -> np.ndarray:
    return model.predict(features[rows.expression_index.to_numpy(int)], np.vstack([fingerprints[str(drug)] for drug in rows.drug_id]))


def run_development_benchmark(config: Mapping[str, Any], *, admission_path: str | Path, data_root: str | Path,
                              output_dir: str | Path, registry_path: str | Path | None = None,
                              loader: Callable[..., Any] = load_registered_cohort) -> dict[str, Any]:
    """Run 5x3 patient-nested development CV and persist auditable artifacts."""
    frozen = dict(config)
    cohort_id = _identity(frozen.get("cohort_id"), "cohort_id")
    if int(frozen.get("outer_folds", 5)) != 5 or int(frozen.get("inner_folds", 3)) != 3:
        raise ValueError("development protocol requires outer_folds=5 and inner_folds=3")
    models = list(frozen.get("models", ["B0", "B1"]))
    if not set(models) <= {"B0", "B1", "M1"} or "B0" not in models:
        raise ValueError("models must include B0 and may include B1 and M1")
    m1_configs = list(frozen.get("m1_configs", []))
    if "M1" in models and not 1 <= len(m1_configs) <= 3:
        raise ValueError("M1 requires one to three frozen configs")
    root, output = Path(data_root), Path(output_dir)
    registry = Path(registry_path) if registry_path is not None else root / "registry" / "transfer_cohorts.json"
    receipt = require_development_admission(admission_path, cohort_id=cohort_id, registry=registry)
    bundle = loader(cohort_id, root=root, registry=registry)
    table, genes = _table(bundle), np.asarray(bundle.gene_ids, dtype=str)
    expression = np.asarray(bundle.expression, dtype=float)
    output.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(_canonical(frozen)).hexdigest()
    (output / "frozen_config.json").write_text(json.dumps({"config": frozen, "sha256": config_hash,
                                                              "registry_sha256": sha256(registry), "admission": receipt}, indent=2), encoding="utf-8")
    assignment = assign_patient_folds(table.patient_id.astype(str).tolist(), n_folds=5, seed=int(frozen.get("seed", 0)))
    fingerprints = _fingerprints(bundle, table.drug_id) if "M1" in models else None
    all_rows: list[pd.DataFrame] = []
    explanations: list[dict[str, Any]] = []
    for fold in range(5):
        test_patients = {patient for patient, value in assignment.patient_to_test_fold.items() if value == fold}
        train, test = table.loc[~table.patient_id.isin(test_patients)].copy(), table.loc[table.patient_id.isin(test_patients)].copy()
        if set(train.patient_id) & set(test.patient_id):
            raise AssertionError("patient leakage across outer fold")
        fit_indices = np.unique(train.expression_index.to_numpy(int))
        prep = ExpressionPreprocessor.fit(expression[fit_indices], genes, n_features=frozen.get("n_features"))
        x = prep.transform(expression, genes)
        prep.save(output / f"fold_{fold}_preprocessor.npz")
        b0 = _fit_b0(train); baseline = _predict_b0(b0, test.drug_id)
        with (output / f"fold_{fold}_b0.pkl").open("wb") as handle: pickle.dump(b0, handle)
        all_rows.append(pd.DataFrame({"patient_id": test.patient_id.astype(str), "model_id": test.model_id.astype(str), "drug_id": test.drug_id.astype(str), "observed": test.response, "predicted": baseline, "baseline": baseline, "fold": fold, "algorithm": "B0"}))
        if "B1" in models:
            alpha = _select_alpha(train, expression, genes, frozen, seed=int(frozen.get("seed", 0)) + fold)
            b1 = _fit_b1(x, train, alpha=alpha, min_train=int(frozen.get("b1_min_train", 2)))
            with (output / f"fold_{fold}_b1.pkl").open("wb") as handle: pickle.dump(b1, handle)
            all_rows.append(pd.DataFrame({"patient_id": test.patient_id.astype(str), "model_id": test.model_id.astype(str), "drug_id": test.drug_id.astype(str), "observed": test.response, "predicted": _predict_b1(b1, x, test), "baseline": baseline, "fold": fold, "algorithm": "B1"}))
        if "M1" in models:
            # Config selection uses only inner validation PDO rankings.
            candidate_scores = []
            for number, params in enumerate(m1_configs):
                score = []
                for inner_train, inner_valid in _inner_partitions(train, folds=3, seed=int(frozen.get("seed", 0)) + fold):
                    inner_prep = ExpressionPreprocessor.fit(expression[np.unique(inner_train.expression_index.to_numpy(int))], genes, n_features=frozen.get("n_features"))
                    inner_x = inner_prep.transform(expression, genes)
                    candidate = _fit_m1(inner_x, fingerprints, inner_train, params)
                    score.append(_score(inner_valid, _predict_m1(candidate, inner_x, fingerprints, inner_valid)))
                candidate_scores.append((float(np.mean(score)), number, params))
            _, _, selected = min(candidate_scores, key=lambda row: (-row[0], row[1]))
            m1 = _fit_m1(x, fingerprints, train, selected); m1.save(output / f"fold_{fold}_m1.pt")
            predicted = _predict_m1(m1, x, fingerprints, test)
            all_rows.append(pd.DataFrame({"patient_id": test.patient_id.astype(str), "model_id": test.model_id.astype(str), "drug_id": test.drug_id.astype(str), "observed": test.response, "predicted": predicted, "baseline": baseline, "fold": fold, "algorithm": "M1"}))
            # Deterministic identity selection, never based on response or score.
            for row in test.sort_values(["model_id", "drug_id"]).head(3).itertuples(index=False):
                item = pd.DataFrame([row._asdict()])
                why = m1.explain(x[[int(row.expression_index)]], np.vstack([fingerprints[str(row.drug_id)]]))
                explanations.append({"fold": fold, "model_id": str(row.model_id), "drug_id": str(row.drug_id),
                                     "reference": float(why["reference"][0]), "prediction": float(why["prediction"][0]),
                                     "sample_attributions": why["sample_attributions"][0].tolist(),
                                     "drug_attributions": why["drug_attributions"][0].tolist()})
    predictions = pd.concat(all_rows, ignore_index=True)
    predictions.to_parquet(output / "outer_oof_predictions.parquet", index=False)
    (output / "explanations.json").write_text(json.dumps(explanations, indent=2), encoding="utf-8")
    resamples = int(frozen.get("bootstrap_resamples", 2000))
    results = {"cohort_id": cohort_id, "status": "DEVELOPMENT_ONLY", "config_sha256": config_hash,
               "bootstrap_resamples": resamples, "bootstrap_label": "FORMAL_2000" if resamples == 2000 else "SMOKE_ONLY",
               "metrics": {}}
    for algorithm, rows in predictions.groupby("algorithm", sort=True):
        results["metrics"][str(algorithm)] = evaluate_predictions(rows.drop(columns="algorithm"), seed=int(frozen.get("seed", 0)), resamples=resamples)
    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
