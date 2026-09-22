"""Pre-registered, source-bound PDO transfer development runner."""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import torch

from .benchmark import _fit_b0, _predict_b0, _table
from .chemistry import morgan_fingerprint
from .cohorts import load_registered_cohort, sha256
from .metrics_v2 import evaluate_predictions
from .models import DualEncoderPairModel
from .preprocessing import ExpressionPreprocessor
from .splits import assign_patient_folds


_PARAMS = {"hidden_dim": 32, "epochs": 50, "learning_rate": .001,
           "ranking_weight": .2, "max_rank_pairs": 128, "batch_groups": 8}


def _target_receipt(path: Path, cohort_id: str, registry: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("cohort_id") != cohort_id or receipt.get("status") not in {"DEVELOPMENT_ADMITTED", "EXPLORATORY_ADMITTED"}:
        raise ValueError("target admission must be DEVELOPMENT_ADMITTED or EXPLORATORY_ADMITTED")
    if receipt.get("registry_sha256", receipt.get("registry_hash")) != sha256(registry):
        raise ValueError("target admission registry hash mismatch")
    return receipt


def _source(source_dir: Path) -> tuple[ExpressionPreprocessor, DualEncoderPairModel, set[str], dict[str, Any]]:
    frozen_path, model_path, prep_path = (source_dir / name for name in ("frozen_run.json", "source_model.pt", "expression_transform.npz"))
    if not all(path.is_file() for path in (frozen_path, model_path, prep_path)):
        raise ValueError("source output requires frozen_run.json, source_model.pt, and expression_transform.npz")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    source_drugs = frozen.get("source_drug_ids")
    if not isinstance(source_drugs, list) or not source_drugs or not all(isinstance(value, str) and value for value in source_drugs):
        raise ValueError("source frozen_run.json requires source_drug_ids")
    model = DualEncoderPairModel.load(model_path)
    if model.config["hidden_dim"] != _PARAMS["hidden_dim"]:
        raise ValueError("source model hidden_dim must match the fixed transfer architecture")
    return ExpressionPreprocessor.load(prep_path), model, set(source_drugs), frozen


def _fingerprints(bundle: Any, ids: pd.Series) -> dict[str, np.ndarray]:
    drugs = bundle.drugs.copy(); field = next((key for key in ("canonical_smiles", "canonical_isomeric_smiles", "smiles") if key in drugs), None)
    if field is None: raise ValueError("target drugs require canonical_smiles, canonical_isomeric_smiles, or smiles")
    lookup = dict(zip(drugs.drug_id.astype(str), drugs[field], strict=True))
    if any(str(drug) not in lookup or not isinstance(lookup[str(drug)], str) or not lookup[str(drug)].strip() for drug in ids.unique()):
        raise ValueError("target response drug lacks a registered SMILES")
    return {str(drug): morgan_fingerprint(lookup[str(drug)], radius=2, n_bits=2048).astype(float) for drug in ids.unique()}


def _arrays(features: np.ndarray, fingerprints: Mapping[str, np.ndarray], rows: pd.DataFrame):
    return (features[rows.expression_index.to_numpy(int)], np.vstack([fingerprints[str(d)] for d in rows.drug_id]),
            rows.response.to_numpy(float))


def _oof(rows: pd.DataFrame, *, prediction: np.ndarray, baseline: np.ndarray, fold: int, algorithm: str, source_drugs: set[str]) -> pd.DataFrame:
    return pd.DataFrame({"patient_id": rows.patient_id.astype(str), "model_id": rows.model_id.astype(str),
        "drug_id": rows.drug_id.astype(str), "fold": fold, "algorithm": algorithm, "observed": rows.response.to_numpy(float),
        "predicted": prediction, "baseline": baseline, "source_drug_status": np.where(rows.drug_id.astype(str).isin(source_drugs), "known", "unknown")})


def _move_to_device(model: DualEncoderPairModel, device: str) -> DualEncoderPairModel:
    """Move a CPU-loaded source checkpoint to the declared runner device."""
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requires an available CUDA device")
    model.device = torch.device(device)
    assert model.network is not None
    model.network.to(model.device)
    return model


def _with_comparator(candidate: pd.DataFrame, comparator: pd.DataFrame) -> pd.DataFrame:
    keys = ["patient_id", "model_id", "drug_id", "fold"]
    reference = comparator[keys + ["predicted"]].rename(columns={"predicted": "baseline"})
    result = candidate.drop(columns="baseline").merge(reference, on=keys, how="inner", validate="one_to_one")
    if len(result) != len(candidate): raise AssertionError("missing paired comparator prediction")
    return result


def run_transfer_experiment(config: Mapping[str, Any], *, target_admission: str | Path, source_output: str | Path,
                            data_root: str | Path, output_dir: str | Path, registry_path: str | Path | None = None,
                            loader: Callable[..., Any] = load_registered_cohort) -> dict[str, Any]:
    """Run target patient-outer-5-fold exploratory transfer; never an external claim."""
    cfg, root, output = dict(config), Path(data_root), Path(output_dir)
    cohort_id = str(cfg.get("cohort_id", "")).strip()
    if not cohort_id: raise ValueError("cohort_id is required")
    seeds = cfg.get("seeds", [cfg.get("seed", 0)])
    if not isinstance(seeds, list) or not seeds or any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) for seed in seeds): raise ValueError("seeds must be a non-empty integer list")
    if output.exists() and any(output.iterdir()): raise ValueError("output must be new or empty; never overwrite a run")
    device = str(cfg.get("device", "cpu"))
    if device not in {"cpu", "cuda"}: raise ValueError("device must be 'cpu' or 'cuda'")
    registry = Path(registry_path) if registry_path else root / "registry" / "transfer_cohorts.json"
    receipt = _target_receipt(Path(target_admission), cohort_id, registry)
    source_path = Path(source_output)
    source_prep, _, source_drugs, frozen_source = _source(source_path)
    bundle = loader(cohort_id, root=root, registry=registry); table = _table(bundle)
    expression, genes = np.asarray(bundle.expression, float), np.asarray(bundle.gene_ids, str)
    fingerprints = _fingerprints(bundle, table.drug_id)
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: sha256(source_path / name) for name in ("frozen_run.json", "source_model.pt", "expression_transform.npz")}
    (output / "frozen_config.json").write_text(json.dumps({"config": cfg, "target_admission": receipt,
        "target_registry_sha256": sha256(registry), "source_artifact_sha256": source_hashes,
        "source_frozen_run": frozen_source}, indent=2), encoding="utf-8")
    all_predictions: list[pd.DataFrame] = []; started = time.perf_counter()
    for seed in map(int, seeds):
        assignment = assign_patient_folds(table.patient_id.astype(str).tolist(), n_folds=5, seed=seed)
        for fold in range(5):
            test_patients = {p for p, value in assignment.patient_to_test_fold.items() if value == fold}
            train, test = table.loc[~table.patient_id.isin(test_patients)].copy(), table.loc[table.patient_id.isin(test_patients)].copy()
            if set(train.patient_id) & set(test.patient_id): raise AssertionError("patient leakage across outer fold")
            prefix = output / f"seed_{seed}_fold_{fold}"; b0 = _fit_b0(train); baseline = _predict_b0(b0, test.drug_id)
            print(f'seed={seed} fold={fold} training_patients={train.patient_id.nunique()} test_patients={test.patient_id.nunique()}', flush=True)
            with prefix.with_name(prefix.name + "_b0.pkl").open("wb") as handle: pickle.dump(b0, handle)
            native = ExpressionPreprocessor.fit(expression[np.unique(train.expression_index.to_numpy(int))], genes, n_features=512)
            native_x, source_x = native.transform(expression, genes), source_prep.transform(expression, genes)
            native.save(prefix.with_name(prefix.name + "_native_preprocessor.npz")); source_prep.save(prefix.with_name(prefix.name + "_source_preprocessor.npz"))
            all_predictions.append(_oof(test, prediction=baseline, baseline=baseline, fold=fold, algorithm=f"B0_seed_{seed}", source_drugs=source_drugs))
            params = _PARAMS | {"seed": seed, "device": device}
            # M1_source_scaled isolates preprocessing transfer: it starts from
            # random weights and sees target labels only, like native M1.
            for name, features, initialize in (("M1", native_x, False), ("M1_source_scaled", source_x, False), ("M2", source_x, True), ("M3", source_x, True)):
                model = _move_to_device(DualEncoderPairModel.load(source_path / "source_model.pt"), device) if initialize else DualEncoderPairModel(**params)
                if name == "M3":
                    # Source weights initialize M3; target optimization itself
                    # is fixed by this runner's pre-registered parameters.
                    model.config.update({key: value for key, value in params.items() if key != "device"}); model.batch_groups = params["batch_groups"]
                if name != "M2": model.fit(*_arrays(features, fingerprints, train), pdo_ids=train.model_id.astype(str).to_numpy())
                if name != "M2":
                    prefix.with_name(prefix.name + f"_{name.lower()}_history.json").write_text(json.dumps(model.training_history, indent=2))
                model.save(prefix.with_name(prefix.name + f"_{name.lower()}.pt"))
                prediction = model.predict(features[test.expression_index.to_numpy(int)], np.vstack([fingerprints[str(d)] for d in test.drug_id]))
                all_predictions.append(_oof(test, prediction=prediction, baseline=baseline, fold=fold, algorithm=f"{name}_seed_{seed}", source_drugs=source_drugs))
    predictions = pd.concat(all_predictions, ignore_index=True); predictions.to_parquet(output / "outer_oof_predictions.parquet", index=False)
    resamples = int(cfg.get("bootstrap_resamples", 2000)); results: dict[str, Any] = {"status": "EXPLORATORY_TRANSFER_ONLY", "cohort_id": cohort_id, "bootstrap_resamples": resamples, "bootstrap_label": "FORMAL_2000" if resamples == 2000 else "SMOKE_ONLY", "metrics": {}, "source_drug_ids_count": len(source_drugs), "elapsed_seconds": time.perf_counter()-started}
    for algorithm, rows in predictions.groupby("algorithm", sort=True):
        seed = algorithm.rsplit("_seed_", 1)[1]
        comparator = predictions.loc[predictions.algorithm.eq(f"M1_source_scaled_seed_{seed}")]
        known = _with_comparator(rows.loc[rows.source_drug_status.eq("known")], comparator.loc[comparator.source_drug_status.eq("known")]).drop(columns=["algorithm", "source_drug_status"])
        report = evaluate_predictions(known, seed=0, resamples=resamples)
        differences = [row["ndcg_at_5_difference"] for row in report["pdos"] if row["ndcg_at_5_difference"] is not None]
        results["metrics"][algorithm] = {"comparator": f"M1_source_scaled_seed_{seed}", "known_primary": report, "n_unknown_rows": int((rows.source_drug_status == "unknown").sum()),
            "summary": {"scale": {"known_rows": len(known), "known_patients": report["n_patients"], "known_models": report["n_models"], "known_drugs": report["n_drugs"]},
                        "ndcg_at_5_difference_by_pdo": {"positive": sum(value > 0 for value in differences), "negative": sum(value < 0 for value in differences), "zero": sum(value == 0 for value in differences), "undefined": len(report["pdos"]) - len(differences)}}}
    for seed in map(int, seeds):
        m3 = predictions.loc[predictions.algorithm.eq(f"M3_seed_{seed}") & predictions.source_drug_status.eq("known")]
        comparisons = {}
        for reference in ("M1", "M1_source_scaled"):
            other = predictions.loc[predictions.algorithm.eq(f"{reference}_seed_{seed}") & predictions.source_drug_status.eq("known")]
            comparisons[reference] = evaluate_predictions(_with_comparator(m3, other).drop(columns=["algorithm", "source_drug_status"]), seed=seed, resamples=resamples)
        results.setdefault("paired_m3_comparisons", {})[f"seed_{seed}"] = comparisons
    results["runtime_device"] = device
    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
