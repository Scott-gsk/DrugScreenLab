"""Light smoke tests for evaluation skeleton (no full experiments)."""

from __future__ import annotations

import numpy as np

from drug_screen.evaluation import (
    ExpressionPermutationContrast,
    check_split_leakage,
    evaluate_task_a,
    evaluate_task_b,
    judge_model_gain,
    mae,
    ndcg_at_k,
    paired_resample_ci,
    rmse,
    spearman_rho,
    top_k_overlap,
)


def test_metrics_perfect_ranking():
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    assert ndcg_at_k(y, y, k=5) == 1.0
    assert spearman_rho(y, y) == 1.0
    assert top_k_overlap(y, y, k=5) == 1.0
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0


def test_task_interfaces():
    a = evaluate_task_a(drug_id="d1", y_true=[1, 2, 3], y_pred=[1.1, 1.9, 3.2])
    assert a.drug_id == "d1" and a.n_patients == 3
    b = evaluate_task_b(
        patient_id="p1",
        y_true=[5, 4, 3, 2, 1, 0],
        y_score=[5, 4, 3, 2, 1, 0],
        drug_ids=list("abcdef"),
    )
    assert b.ndcg_at_5 == 1.0 and b.top5_overlap == 1.0


def test_leakage_and_ci_judgment():
    report = check_split_leakage(
        train_patients=["p1"],
        test_patients=["p2"],
        train_cells=["c1"],
        test_cells=["c2"],
    )
    assert report.ok

    leak = check_split_leakage(
        train_patients=["p1"],
        test_patients=["p1"],
    )
    assert not leak.ok

    rng = np.random.default_rng(0)
    base = rng.normal(size=30)
    model = base + 0.5
    ci = paired_resample_ci(model, base, n_resamples=200, random_state=0)
    assert ci.estimate > 0
    judgment = judge_model_gain(ci)
    assert judgment.claim_level


def test_permutation_hook_runs():
    expr = np.arange(12, dtype=float).reshape(4, 3)
    y = np.array([0.0, 1.0, 2.0, 3.0])

    def predict(x: np.ndarray) -> np.ndarray:
        return x[:, 0]

    def metric(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(np.corrcoef(yt, yp)[0, 1])

    contrast = ExpressionPermutationContrast(
        predict_fn=predict,
        metric_fn=metric,
        n_repeats=5,
        random_state=1,
    )
    result = contrast.run(expr, y)
    assert result.n_repeats == 5
