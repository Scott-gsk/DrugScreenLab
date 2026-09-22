import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from drug_screen.transfer.benchmark import run_development_benchmark
from drug_screen.transfer.benchmark import _score, _fit_b1, _predict_b1
from drug_screen.transfer.bundles import IndexedBundle


def _bundle():
    patients = [f"p{i}" for i in range(5)]
    samples = pd.DataFrame({"sample_id": [f"s{i}" for i in range(5)], "patient_id": patients})
    drugs = pd.DataFrame({"drug_id": [f"d{i}" for i in range(10)]})
    pairs = pd.DataFrame([
        {"sample_id": f"s{i}", "drug_id": f"d{j}", "response": float(i + j),
         "response_name": "dss", "higher_is_sensitive": True}
        for i in range(5) for j in range(10)
    ])
    return IndexedBundle("synthetic", "PDO_DEVELOPMENT", samples, drugs, pairs,
                         np.arange(20, dtype=float).reshape(5, 4), np.array(samples.sample_id),
                         np.array(["g1", "g2", "g3", "g4"]), "unit", True, "dss")


def test_selection_patient_weight_unchanged_by_duplicate_pdo():
    rows = pd.DataFrame([
        {"patient_id": patient, "model_id": model, "response": float(j)}
        for patient, model in [("a", "a1"), ("b", "b1")]
        for j in range(10)
    ])
    prediction = np.r_[np.arange(10), np.arange(10)[::-1]]
    duplicate = rows.iloc[:10].copy(); duplicate["model_id"] = "a2"
    assert _score(rows, prediction) == pytest.approx(_score(
        pd.concat([rows, duplicate], ignore_index=True), np.r_[prediction, prediction[:10]]))


def test_dual_ridge_matches_reference_in_high_dimension():
    from sklearn.linear_model import Ridge
    x = np.random.default_rng(7).normal(size=(7, 50))
    y = np.arange(7, dtype=float)
    rows = pd.DataFrame({"patient_id": [f"p{i}" for i in range(7)],
                         "drug_id": "d", "expression_index": np.arange(7), "response": y})
    model = _fit_b1(x, rows, alpha=10, min_train=2)
    assert np.allclose(_predict_b1(model, x, rows), Ridge(alpha=10).fit(x, y).predict(x))


def _receipt(tmp_path, *, status="DEVELOPMENT_ADMITTED"):
    registry = tmp_path / "registry.json"; registry.write_text('{"cohorts": []}', encoding="utf-8")
    digest = hashlib.sha256(registry.read_bytes()).hexdigest()
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps({"cohort_id": "synthetic", "status": status, "registry_sha256": digest}), encoding="utf-8")
    return registry, admission


def _loader(*args, **kwargs):
    return _bundle()


def test_nested_runner_writes_frozen_oof_and_patient_isolated_artifacts(tmp_path):
    registry, admission = _receipt(tmp_path)
    output = tmp_path / "out"
    result = run_development_benchmark(
        {"cohort_id": "synthetic", "models": ["B0", "B1"], "seed": 4,
         "bootstrap_resamples": 2, "n_features": 3, "b1_min_train": 2},
        admission_path=admission, data_root=tmp_path, output_dir=output,
        registry_path=registry, loader=_loader,
    )
    assert result["status"] == "DEVELOPMENT_ONLY"
    frozen = json.loads((output / "frozen_config.json").read_text())
    assert frozen["sha256"] == result["config_sha256"]
    oof = pd.read_parquet(output / "outer_oof_predictions.parquet")
    assert {"patient_id", "model_id", "drug_id", "fold", "algorithm"} <= set(oof)
    assert oof.groupby("patient_id").fold.nunique().eq(1).all()
    assert sorted(output.glob("fold_*_preprocessor.npz"))
    assert sorted(output.glob("fold_*_b1.pkl"))


def test_runner_rejects_cohort_without_development_admission(tmp_path):
    registry, admission = _receipt(tmp_path, status="PENDING")
    with pytest.raises(ValueError, match="DEVELOPMENT_ADMITTED"):
        run_development_benchmark({"cohort_id": "synthetic"}, admission_path=admission,
                                  data_root=tmp_path, output_dir=tmp_path / "out",
                                  registry_path=registry, loader=_loader)
