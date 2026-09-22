"""Single frozen-model prediction package for PDO candidate ranking."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .chemistry import morgan_fingerprint
from .cohorts import sha256
from .models import DualEncoderPairModel
from .preprocessing import ExpressionPreprocessor


def _ids(values: np.ndarray, name: str) -> np.ndarray:
    answer = np.asarray(values).astype(str)
    if answer.ndim != 1 or not len(answer) or any(not value.strip() for value in answer) or len(set(answer)) != len(answer):
        raise ValueError(f"{name} must be non-empty unique identities")
    return answer


def _expression(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"sample_id", "ordered_genes", "expression"}
        if not required <= set(archive.files): raise ValueError("expression NPZ requires sample_id, ordered_genes, and expression")
        sample_ids, genes, values = _ids(archive["sample_id"], "sample_id"), _ids(archive["ordered_genes"], "ordered_genes"), np.asarray(archive["expression"], float)
    if values.ndim != 2 or values.shape != (len(sample_ids), len(genes)) or np.isinf(values).any():
        raise ValueError("expression must align identities and cannot contain infinity")
    return sample_ids, genes, values


def _candidates(path: Path) -> pd.DataFrame:
    table = pd.read_parquet(path)
    if not {"drug_id", "canonical_smiles"} <= set(table):
        raise ValueError("candidate drugs require drug_id and canonical_smiles")
    table = table[["drug_id", "canonical_smiles"]].copy()
    table["drug_id"] = table.drug_id.astype(str)
    if table.empty or table.drug_id.str.strip().eq("").any() or table.drug_id.duplicated().any():
        raise ValueError("candidate drug_id values must be non-empty and unique")
    if table.canonical_smiles.isna().any() or not table.canonical_smiles.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("candidate canonical_smiles values must be non-empty strings")
    # morgan_fingerprint validates every input: invalid chemistry aborts rather
    # than silently omitting a candidate from a clinical-style ranking.
    table["fingerprint"] = [morgan_fingerprint(value, radius=2, n_bits=2048).astype(float) for value in table.canonical_smiles]
    return table


def _trusted_artifact(model_path: Path, preprocessor_path: Path) -> None:
    if model_path.suffix != ".pt" or preprocessor_path.suffix != ".npz":
        raise ValueError("model must be a .pt project artifact and preprocessor must be a .npz project artifact")
    if model_path.parent != preprocessor_path.parent:
        raise ValueError("model and corresponding preprocessor must share a project run directory")
    if not any((model_path.parent / name).is_file() for name in ("frozen_config.json", "frozen_run.json")):
        raise ValueError("model directory lacks a frozen project-run record")


def predict_transfer(model_path: str | Path, preprocessor_path: str | Path, expression_path: str | Path,
                     candidates_path: str | Path, output_dir: str | Path, *, explain_top_k: int = 3,
                     explain_steps: int = 64) -> dict[str, Any]:
    """Rank all candidates per PDO using one frozen model, without uncertainty claims."""
    model_file, prep_file, expression_file, candidates_file, output = map(Path, (model_path, preprocessor_path, expression_path, candidates_path, output_dir))
    if isinstance(explain_top_k, bool) or not isinstance(explain_top_k, int) or explain_top_k < 0:
        raise ValueError("explain_top_k must be a non-negative integer")
    if output.exists() and any(output.iterdir()): raise ValueError("output must be new or empty; never overwrite a prediction run")
    _trusted_artifact(model_file, prep_file)
    model, preprocessor = DualEncoderPairModel.load(model_file), ExpressionPreprocessor.load(prep_file)
    sample_ids, genes, expression = _expression(expression_file)
    transformed, coverage = preprocessor.transform_with_report(expression, genes)
    candidates = _candidates(candidates_file)
    if model.sample_dim != transformed.shape[1] or model.drug_dim != 2048:
        raise ValueError("model dimensions do not match frozen preprocessing or Morgan fingerprint contract")
    repeated_features = np.repeat(transformed, len(candidates), axis=0)
    repeated_drugs = np.tile(np.vstack(candidates.fingerprint.to_numpy()), (len(sample_ids), 1))
    score = model.predict(repeated_features, repeated_drugs)
    rows = pd.DataFrame({"sample_id": np.repeat(sample_ids, len(candidates)), "drug_id": np.tile(candidates.drug_id.to_numpy(), len(sample_ids)), "score": score})
    rows = rows.sort_values(["sample_id", "score", "drug_id"], ascending=[True, False, True], kind="mergesort").reset_index(drop=True)
    rows["rank"] = rows.groupby("sample_id", sort=False).cumcount() + 1
    output.mkdir(parents=True, exist_ok=True); rows.to_parquet(output / "ranked_predictions.parquet", index=False)
    explanations = []
    if explain_top_k:
        # This is explicitly post-prediction selection by score, not a
        # prospectively chosen explanatory cohort or a confidence estimate.
        for row in rows.groupby("sample_id", sort=True).head(explain_top_k).itertuples(index=False):
            sample_index = int(np.flatnonzero(sample_ids == row.sample_id)[0]); drug_index = int(np.flatnonzero(candidates.drug_id.to_numpy() == row.drug_id)[0])
            why = model.explain(transformed[[sample_index]], candidates.fingerprint.iloc[[drug_index]].tolist(), steps=explain_steps)
            explanations.append({"sample_id": row.sample_id, "drug_id": row.drug_id, "rank": int(row.rank), "score": float(row.score),
                "selection": "postprediction_top_score", "reference": float(why["reference"][0]), "prediction": float(why["prediction"][0]),
                "sample_attributions": why["sample_attributions"][0].tolist(), "drug_attributions": why["drug_attributions"][0].tolist()})
    (output / "top_score_explanations.json").write_text(json.dumps(explanations, indent=2), encoding="utf-8")
    report = {"status": "SINGLE_FROZEN_MODEL_PREDICTION", "calibrated_uncertainty": False,
              "note": "Scores from different fold models are not calibrated confidence estimates.",
              "model_sha256": sha256(model_file), "preprocessor_sha256": sha256(prep_file), "expression_sha256": sha256(expression_file),
              "candidates_sha256": sha256(candidates_file), "n_pdos": len(sample_ids), "n_candidates": len(candidates),
              "missing_gene_report": coverage.__dict__, "explanation_selection": "postprediction_top_score", "explanations_written": len(explanations)}
    (output / "prediction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
