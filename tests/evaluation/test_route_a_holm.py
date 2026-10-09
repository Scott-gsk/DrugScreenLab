from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from drug_screen.evaluation.exp031_eval import patient_score_paired_bootstrap
from drug_screen.evaluation.permutation import permutation_p_value
from scripts.evaluation.aggregate_exp036_pharmaformer import paired_statistics
from scripts.evaluation.aggregate_route_a_holm import (
    _resample_statistics,
    build_family,
    holm_adjust,
    verify_a3,
)


def _comparison(*, p: float, effect: float = 0.1, n: int = 51) -> dict:
    return {"p_two_sided_uncorrected": p, "estimate": effect,
            "ci95": [-0.1, 0.2], "n_patients": n}


def _inputs() -> tuple[dict, dict, dict]:
    pharmaformer = {
        "experiment_id": "EXP-036",
        "status": "FIVE_FOLDS_AGGREGATED_PENDING_HOLM_AND_REVIEW",
        "n_folds": 5,
        "n_patients": 51,
        "folds": [{"fold": fold} for fold in range(5)],
        "statistics": {
            "A1_random_full_vs_target_drug_mean": _comparison(p=0.03),
            "A2_source_full_vs_random_full": _comparison(p=0.04, effect=-0.02),
        },
    }
    ridge = {
        "experiment_id": "EXP-036",
        "status": "RIDGE_INTERACTION_COMPLETE_PENDING_REVIEW",
        "comparisons": [
            {"fraction": 0.25},
            {"fraction": 0.5,
             "interaction_minus_drug_only": {
                 "p_two_sided": 0.01, "estimate": -0.2,
                 "low": -0.3, "high": -0.1, "n_patients": 51,
             },
             "interaction_minus_permuted": {
                 "p_two_sided": 0.02, "estimate": 0.01,
                 "low": -0.04, "high": 0.06, "n_patients": 51,
             }},
        ],
    }
    a3 = {
        "experiment_id": "EXP-041",
        "status": "A3_PRIMARY_CONTRAST_READY_LEARNING_CURVE_AND_ROUTE_A_HOLM_PENDING",
        "fraction": 0.5,
        "parent_experiment_id": "EXP-036",
        "n_test_patients_total": 51,
        "main": {
            "paired_sign_flip_p_two_sided_finite_corrected": 0.001,
            "gain": 0.05, "ci95": [0.01, 0.09], "n_valid_patients": 45,
        },
    }
    return pharmaformer, ridge, a3


def test_route_a_holm_keeps_all_preregistered_contrasts_and_controls_sign() -> None:
    report = build_family(*_inputs())

    assert report["status"] == "ROUTE_A_HOLM_FAMILY_AGGREGATED_PENDING_REVIEW"
    assert len(report["family"]) == 5
    assert report["comparisons"]["A2_source_full_vs_random_full"]["effect"] == pytest.approx(-0.02)
    assert report["comparisons"]["A3_source_adapter_vs_source_full"]["n_patients"] == 45
    adjusted = [row["p_holm_adjusted"] for row in report["comparisons"].values()]
    assert all(0 <= value <= 1 for value in adjusted)


def test_holm_is_monotone_in_p_value_order_and_caps_at_one() -> None:
    adjusted = holm_adjust({"c": 0.04, "a": 0.001, "b": 0.02})

    assert adjusted["a"] == pytest.approx(0.003)
    assert adjusted["b"] == pytest.approx(0.04)
    assert adjusted["c"] == pytest.approx(0.04)
    assert holm_adjust({"x": 0.9, "y": 0.95})["y"] == pytest.approx(1.0)


def test_route_a_resampling_matches_existing_pharmaformer_statistic_contract() -> None:
    difference = np.linspace(-0.25, 0.31, 51)

    actual = _resample_statistics(difference, seed=20260923)
    expected = paired_statistics(difference, seed=20260923)

    assert actual["estimate"] == pytest.approx(expected["estimate"], abs=1e-12)
    assert actual["ci95"] == pytest.approx([expected["ci95"][0], expected["ci95"][1]])
    assert actual["p"] == pytest.approx(expected["p_two_sided_uncorrected"], abs=1e-12)
    assert actual["n"] == expected["n_patients"]


def test_a3_gate_recomputes_patient_bootstrap_and_finite_sign_flip(tmp_path) -> None:
    patients = [f"p{i}" for i in range(5)]
    adapter = pd.Series([0.8, 0.6, 0.2, -0.1], index=patients[:4], dtype=float)
    full = pd.Series([0.3, 0.2, 0.1, 0.0], index=patients[:4], dtype=float)
    ci = patient_score_paired_bootstrap(adapter, full, n=2000, seed=20260922)
    difference = adapter.to_numpy() - full.to_numpy()
    rng = np.random.default_rng(20260922)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(10000, len(difference)))
    null = np.abs((signs * difference).mean(axis=1))
    p_value = permutation_p_value(
        abs(float(difference.mean())), null, alternative="greater",
    )
    scores = pd.DataFrame({
        "patient_id": patients[:4],
        "source_adapter_seed_mean_spearman": adapter.to_numpy(),
        "source_full_seed_mean_spearman": full.to_numpy(),
        "gain_adapter_minus_full": difference,
        "eligible_primary": True,
    })
    score_path = tmp_path / "patient_scores.parquet"
    scores.to_parquet(score_path, index=False)
    summary = {
        "main": {
            "gain": ci.point_estimate,
            "ci95": [ci.low, ci.high],
            "n_valid_patients": ci.n_units,
            "paired_sign_flip_p_two_sided_finite_corrected": p_value,
        },
    }
    summary_path = tmp_path / "a3_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    def digest(path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    summary["patient_scores_sha256"] = digest(score_path)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "outputs": {
            "a3_summary.json": digest(summary_path),
            "patient_scores.parquet": digest(score_path),
        },
    }), encoding="utf-8")

    verified = verify_a3(tmp_path)

    assert verified["main"]["gain"] == pytest.approx(difference.mean())


def test_route_a_gate_rejects_a_missing_registered_contrast() -> None:
    pharmaformer, ridge, a3 = _inputs()
    del pharmaformer["statistics"]["A2_source_full_vs_random_full"]

    with pytest.raises(ValueError, match="A1 or A2 contrast"):
        build_family(pharmaformer, ridge, a3)
