from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from drug_screen.evaluation.exp011 import PREDICTION_COLUMNS, validate_prediction_frame
from drug_screen.modeling.exp011_data import build_fold_indices, load_indexed_pair_data


RUNNER = Path(__file__).resolve().parents[2] / "scripts/modeling/run_exp011_baselines.py"
spec = importlib.util.spec_from_file_location("run_exp011_baselines", RUNNER)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def _bundle(tmp_path: Path, *, cohort_id: str = "mcrc_pdo", cell_lines: bool = False) -> tuple[Path, Path, Path]:
    """Create a small indexed data bundle with every drug in every sample."""

    sample_count, drug_count, gene_count = 20, 3, 8
    archive = tmp_path / f"{cohort_id}_indexed.npz"
    expression = np.stack(
        [
            np.arange(sample_count, dtype=np.float32),
            np.arange(sample_count, dtype=np.float32)[::-1],
            np.ones(sample_count, dtype=np.float32),
            np.arange(sample_count, dtype=np.float32) % 2,
            np.linspace(0, 1, sample_count, dtype=np.float32),
            np.linspace(1, 0, sample_count, dtype=np.float32),
            np.zeros(sample_count, dtype=np.float32),
            np.arange(sample_count, dtype=np.float32) * 2,
        ],
        axis=1,
    )
    pair_sample_idx = np.repeat(np.arange(sample_count), drug_count)
    pair_drug_idx = np.tile(np.arange(drug_count), sample_count)
    response = expression[pair_sample_idx, 0] * 0.25 + pair_drug_idx * 3.0
    np.savez_compressed(
        archive,
        expression=expression,
        ordered_genes=np.asarray([f"G{index}" for index in range(gene_count)]),
        sample_id=np.asarray([f"S{index}" for index in range(sample_count)]),
        patient_id=np.asarray([f"P{index}" for index in range(sample_count)]),
        pdo_id=np.asarray([f"O{index}" for index in range(sample_count)]),
        cell_line_id=np.asarray([f"CL{index}" for index in range(sample_count)]) if cell_lines else np.asarray([""] * sample_count),
        drug_id=np.asarray([f"D{index}" for index in range(drug_count)]),
        canonical_smiles=np.asarray(["CC", "CCC", "CCCC"]),
        inchikey=np.asarray([f"IK{index}" for index in range(drug_count)]),
        drug_scaffold=np.asarray([f"SC{index}" for index in range(drug_count)]),
        pair_sample_idx=pair_sample_idx,
        pair_drug_idx=pair_drug_idx,
        pair_response=response.astype(np.float32),
        pair_cohort_id=np.asarray([cohort_id] * len(pair_sample_idx)),
        pair_response_name=np.asarray(["1_minus_auc" if cell_lines else "dss"] * len(pair_sample_idx)),
        pair_response_scale=np.asarray(["1-AUC" if cell_lines else "DSS"] * len(pair_sample_idx)),
        pair_higher_is_sensitive=np.ones(len(pair_sample_idx), dtype=bool),
    )
    raw = tmp_path / f"{cohort_id}_raw_manifest.json"
    raw.write_text(
        json.dumps(
            {
                "schema": "drugscreenlab.exp011.raw-indexed-pairs.v1",
                "raw_npz": archive.name,
                "formal_gene_contract": False,
                "cohort_id": cohort_id,
            }
        ),
        encoding="utf-8",
    )
    top = tmp_path / "cohort_index_manifest.json"
    top.write_text(
        json.dumps(
            {
                "format": "exp011_indexed_drug_response_v1",
                "cohorts": [{"cohort_id": cohort_id, "bundle": {"raw_manifest": raw.name}}],
            }
        ),
        encoding="utf-8",
    )
    return archive, raw, top


def _args(manifest: Path, output: Path, *, cohort_id: str, stage: str = "target", **extra: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "manifest": manifest,
        "cohort_id": cohort_id,
        "stage": stage,
        "fold_id": 0,
        "evaluation_role": None,
        "output": output,
        "seed": 20260823,
        "with_random_forest": False,
        "rf_estimators": 5,
        "formal": False,
    }
    values.update(extra)
    return SimpleNamespace(**values)


def test_aggregate_manifest_requires_cohort_id_and_resolves_raw_bundle(tmp_path: Path) -> None:
    _, raw, top = _bundle(tmp_path)
    with pytest.raises(ValueError, match="requires --cohort-id"):
        runner.resolve_cohort_manifest(top, cohort_id=None)
    resolved, cohort_id, aggregate = runner.resolve_cohort_manifest(top, cohort_id="mcrc_pdo")
    assert resolved == raw.resolve() and cohort_id == "mcrc_pdo" and aggregate is True


def test_train_only_feature_fit_is_stable_when_test_expression_or_labels_change(tmp_path: Path) -> None:
    _, raw, _ = _bundle(tmp_path)
    data = load_indexed_pair_data(raw)
    fold = build_fold_indices(data, stage="target", fold_id=0, seed=20260823)
    original, original_predictions = runner.fit_predict_baselines(
        data, train_pair_indices=fold.train, test_pair_indices=fold.test, seed=7
    )
    # Mutating all held-out labels and expression cannot change the fitted HVG
    # set, scaling state, train drug vocabulary, or label-trained coefficients.
    changed_expression = data.expression.copy()
    held_samples = np.unique(data.pair_sample_idx[fold.test])
    changed_expression[held_samples, 6] = 1e9
    changed_response = data.pair_response.copy()
    changed_response[fold.test] = -1e9
    mutated = replace(data, expression=changed_expression, pair_response=changed_response)
    changed, changed_predictions = runner.fit_predict_baselines(
        mutated, train_pair_indices=fold.train, test_pair_indices=fold.test, seed=7
    )
    assert original.selected_gene_ids == changed.selected_gene_ids
    np.testing.assert_array_equal(original.gene_mean, changed.gene_mean)
    np.testing.assert_array_equal(original.gene_scale, changed.gene_scale)
    assert original.train_drug_ids == changed.train_drug_ids
    # Ridge/RF fit ignores held-out labels.  Test expression naturally affects
    # test predictions, so only the train-derived B0 value must also agree.
    np.testing.assert_array_equal(original_predictions["B0_drug_median"], changed_predictions["B0_drug_median"])


def test_runner_emits_evaluator_schema_for_target_and_optional_rf(tmp_path: Path) -> None:
    _, _, top = _bundle(tmp_path)
    output = tmp_path / "out"
    summary = runner.run(
        _args(top, output, cohort_id="mcrc_pdo", with_random_forest=True, rf_estimators=3)
    )
    prediction = pd.read_parquet(summary["prediction_path"])
    assert prediction.columns.tolist() == list(PREDICTION_COLUMNS)
    assert set(prediction.model_id) == {"B0_drug_median", "B1_ridge", "B2_random_forest"}
    assert set(prediction.evaluation_role) == {"pdo_primary"}
    assert summary["ridge_alpha"] == 10.0
    assert summary["train_only_preprocessing"]["test_labels_read_during_fit"] is False
    validate_prediction_frame(prediction)


def test_runner_reuses_source_cell_line_folds_and_role(tmp_path: Path) -> None:
    _, _, top = _bundle(tmp_path, cohort_id="gdsc2_cell_line", cell_lines=True)
    summary = runner.run(_args(top, tmp_path / "source_out", cohort_id="gdsc2_cell_line", stage="source"))
    prediction = pd.read_parquet(summary["prediction_path"])
    assert summary["partitions"]["group_field"] == "cell_line_id"
    assert set(prediction.evaluation_role) == {"gdsc_cold_cell"}
    assert prediction.cell_line_id.str.len().gt(0).all()
    validate_prediction_frame(prediction)


def test_formal_mode_rejects_direct_development_npz(tmp_path: Path) -> None:
    archive, _, _ = _bundle(tmp_path)
    with pytest.raises(ValueError, match="reject development direct NPZ"):
        runner.run(_args(archive, tmp_path / "formal", cohort_id="mcrc_pdo", formal=True))


def test_stage_role_cannot_claim_a_different_split_protocol(tmp_path: Path) -> None:
    _, _, top = _bundle(tmp_path)
    with pytest.raises(ValueError, match="must emit pdo_primary"):
        runner.run(
            _args(top, tmp_path / "bad", cohort_id="mcrc_pdo", evaluation_role="gdsc_cold_cell")
        )
