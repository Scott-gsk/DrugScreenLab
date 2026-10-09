"""Synthetic artifact readback test for the EXP-066 independent verifier."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from scripts.evaluation.verify_exp066_response_prediction import (
    ARMS,
    _aggregate,
    _metric_arrays,
    verify_run,
)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_partial_cost_gate_artifact_verifies_from_keyed_predictions(tmp_path) -> None:
    rng = np.random.default_rng(660066)
    n_conditions = 3
    y_true = rng.normal(size=(n_conditions, 978)).astype(np.float32)
    variance = np.linspace(0.1, 1.0, 978, dtype=np.float64)
    objects = np.asarray(["P1", "P1", "P2"])
    keys = np.asarray([
        "AAAAAAAAAAAAAA-BBBBBBBBBB-C",
        "AAAAAAAAAAAAAA-BBBBBBBBBB-C",
        "CCCCCCCCCCCCCC-DDDDDDDDDD-E",
    ])
    predictions = {
        arm: rng.normal(size=(n_conditions, 978)).astype(np.float32)
        for arm in ARMS
    }
    metrics = {
        arm: _metric_arrays(y_true.astype(np.float64), prediction.astype(np.float64), variance)
        for arm, prediction in predictions.items()
    }
    axis_hash = "synthetic-axis-sha256"
    axis = np.asarray([f"G{index:03d}" for index in range(978)])
    condition_ids = np.asarray(["C1", "C2", "C3"])
    artifact = tmp_path / "fold_0.npz"
    arrays = {
        "condition_id": condition_ids,
        "pert_id": np.asarray(["P1", "P2", "P3"]),
        "lincs_cell_id": np.asarray(["L1", "L1", "L2"]),
        "depmap_model_id": np.asarray(["M1", "M1", "M2"]),
        "source_object_id": objects,
        "full_inchikey": keys,
        "fold_id": np.zeros(n_conditions, dtype=np.int8),
        "y_true": y_true,
        "train_gene_variance": variance,
        "target_gene_ids": axis,
        "target_axis_hash": np.full(n_conditions, axis_hash),
    }
    for arm in ARMS:
        arrays[f"pred__{arm}"] = predictions[arm]
        arrays[f"prediction_valid__{arm}"] = np.isfinite(predictions[arm]).all(axis=1)
        for metric_name, values in metrics[arm].items():
            arrays[f"metric__{arm}__{metric_name}"] = values
    np.savez(artifact, **arrays)

    summary_path = tmp_path / "run_summary.json"
    summary = {
        "experiment_id": "EXP-066",
        "run_id": "synthetic-partial",
        "arms": list(ARMS),
        "fold_count_completed": 1,
        "scientific_comparison_valid": False,
        "folds": [{
            "heldout_metrics": _aggregate(metrics, objects, keys),
        }],
        "out_of_fold_metrics": None,
        "files": {
            "fold_0_predictions": {
                "path": str(artifact),
                "sha256": _sha256(artifact),
            },
        },
    }
    summary_path.write_text(json.dumps(summary, allow_nan=False), encoding="utf-8")
    readback = {
        "run_id": "synthetic-partial",
        "run_summary_sha256": _sha256(summary_path),
        "prediction_artifact_count": 1,
        "prediction_artifact_sha256": {
            "fold_0_predictions": _sha256(artifact),
        },
    }
    (tmp_path / "run_readback.json").write_text(json.dumps(readback), encoding="utf-8")

    result = verify_run(tmp_path)
    assert result["status"] == "PASS_EXP066_PARTIAL_COST_GATE_READBACK"
    assert result["verified_folds"] == 1
    assert result["verified_conditions"] == n_conditions
    assert result["target_gene_count"] == 978
