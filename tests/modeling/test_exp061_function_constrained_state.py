import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/modeling/run_function_constrained_state_residual.py"
SPEC = importlib.util.spec_from_file_location("exp061_function_residual", MODULE_PATH)
exp061 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(exp061)


def test_hallmark_basis_requires_fifty_sets_and_normalizes_covered_columns():
    lines = []
    axis = [f"G{i}" for i in range(978)]
    for i in range(50):
        members = [f"G{j}" for j in range(i, i + 5)]
        lines.append("\t".join([f"HALLMARK_{i:02d}", "description", *members]))
    basis, receipt = exp061.build_basis_from_symbols("\n".join(lines), axis)
    assert basis.shape == (978, 50)
    assert np.allclose(np.linalg.norm(basis, axis=0), 1.0)
    assert receipt["eligible_set_count"] == 50
    assert receipt["covered_gene_count"] == 54
    assert receipt["uncovered_gene_count"] == 924
    with pytest.raises(ValueError, match="contain 50 sets"):
        exp061.parse_gmt_text("\t".join(["HALLMARK_ONLY", "description", *axis[:5]]))


def test_closed_form_mapping_matches_fixed_prediction_equation():
    rng = np.random.default_rng(12)
    objects = np.repeat(["o1", "o2", "o3", "o4"], 2)
    keys = pd.DataFrame({"source_object_id": objects, "full_inchikey": ["drug"] * 8})
    basis = np.zeros((978, 2), dtype=np.float64)
    basis[:3, 0] = 1 / np.sqrt(3)
    basis[3:6, 1] = 1 / np.sqrt(3)
    q = rng.normal(size=(8, 2))
    mean = np.zeros((8, 978), dtype=np.float64)
    y = rng.normal(size=(8, 978))
    variance = np.ones(978, dtype=np.float64)
    fitted, receipt = exp061.fit_drug_mapping(keys, y, mean, q, basis, variance, "drug", alpha=1.0)
    pred = exp061.predictions_for_drug(q, basis, fitted, np.zeros(978))
    weights = exp061.make_row_weights(keys)
    residual = y
    gram = basis.T @ basis / 978.0
    state_gram = q.T @ (weights[:, None] * q)
    projected = (residual @ basis / 978.0).T @ (weights[:, None] * q)
    manual = np.linalg.solve(np.kron(state_gram.T, gram) + np.eye(4),
                             projected.reshape(-1, order="F")).reshape((2, 2), order="F")
    assert np.isfinite(fitted).all() and np.isfinite(pred).all()
    assert receipt["parameters"] == 4
    assert np.isclose(weights.sum(), 1.0)
    assert np.allclose(fitted, manual, rtol=1e-10, atol=1e-10)


def test_derangement_is_a_full_no_fixed_point_object_mapping():
    ids = [f"object_{i:03d}" for i in range(110)]
    mapping = exp061.derangement(ids, seed=61064)
    assert set(mapping) == set(ids)
    assert set(mapping.values()) == set(ids)
    assert all(key != value for key, value in mapping.items())


def test_paired_bootstrap_uses_method_to_object_score_maps():
    objects = ["o1", "o2", "o3"]
    scores = {
        "per_drug_mean": {obj: {"nmse": 1.0} for obj in objects},
        "functional_hallmark": {obj: {"nmse": 0.8} for obj in objects},
        "random_gene_map": {obj: {"nmse": 1.1} for obj in objects},
        "random_object_state": {obj: {"nmse": 1.2} for obj in objects},
        "train_state_mismatch": {obj: {"nmse": 1.3} for obj in objects},
        "zero_change": {obj: {"nmse": 2.0} for obj in objects},
    }
    result = exp061.paired_bootstrap(scores, list(scores), replicates=40, seed=7)
    candidate = result["functional_hallmark"]
    assert candidate["mean_absolute_nmse_improvement_over_per_drug_mean"] == pytest.approx(0.2)
    assert candidate["relative_nmse_improvement_over_per_drug_mean"] == pytest.approx(0.2)
    assert candidate["objects_with_lower_nmse_than_per_drug_mean"] == 3
    assert candidate["paired_95pct_percentile_interval_absolute_improvement"] == pytest.approx([0.2, 0.2])


def test_nonfinite_predictions_fail_closed():
    basis = np.zeros((978, 1), dtype=np.float64)
    basis[0, 0] = 1.0
    with pytest.raises(FloatingPointError, match="nonfinite"):
        exp061.predictions_for_drug(np.asarray([[np.inf]]), basis, np.zeros((1, 1)), np.zeros(978))


def test_registered_config_has_explicit_fixed_parameters_and_limits():
    config_path = ROOT / "configs/experiments/EXP-061-function-constrained-state.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert cfg["experiment_id"] == "EXP-061"
    assert cfg["run_id"] == "20261008_exp061_functional_state_rootrepair_v3"
    assert cfg["model"]["ridge_alpha"] == 1.0
    assert [cfg["arms"][key] for key in (
        "random_gene_map_seed", "random_object_state_seed", "train_state_mismatch_seed"
    )] == [61062, 61063, 61064]
    assert cfg["resources"]["cpu_cap_seconds"] == 1800
    assert cfg["resources"]["gpu_hours_cap"] == 0
