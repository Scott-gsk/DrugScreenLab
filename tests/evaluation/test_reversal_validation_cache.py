import numpy as np
import pandas as pd
import pytest
import torch
from types import SimpleNamespace

from drug_screen.evaluation.reversal_validation import ReversalValidationCache
from scripts.modeling.exp039_b5_fold import (
    predict_delta,
    predict_delta_inputs_only,
    reversal_validation_metric,
)
from drug_screen.models.conditional_perturbation import SharedEffectInteraction


def _validation_example():
    rng = np.random.default_rng(271828)
    genes = [f"G{i:02d}" for i in range(20)]
    up = genes[:4]
    down = genes[4:8]
    rows = []
    observed = []
    prediction = []
    for cell_index in range(4):
        for drug_index in range(6):
            base = rng.normal(size=len(genes))
            base[drug_index % 4] += 2.0 + drug_index / 10
            base[4 + ((drug_index + cell_index) % 4)] -= 1.5
            for replicate in range(2):
                rows.append({"cell_id": f"C{cell_index}", "pert_id": f"D{drug_index}"})
                delta = base + rng.normal(0, 0.05, size=len(genes))
                observed.append(delta.astype(np.float32))
                prediction.append((0.7 * delta + rng.normal(0, 0.25, size=len(genes))).astype(np.float32))
    return (pd.DataFrame(rows), np.asarray(observed), np.asarray(prediction), genes, up, down)


def test_cached_reversal_validation_matches_original_for_realistic_grouping():
    keys, observed, prediction, genes, up, down = _validation_example()
    expected_score, expected_cells = reversal_validation_metric(
        type("Data", (), {"keys": keys, "delta": observed})(),
        np.arange(len(keys)), prediction, genes, up, down,
    )
    cache = ReversalValidationCache(keys, observed, genes, up, down)
    actual_score, actual_cells = cache.score(prediction)

    assert actual_score == expected_score
    pd.testing.assert_frame_equal(actual_cells, expected_cells)


def test_cached_reversal_validation_preserves_ineligible_cell_reason():
    keys, observed, prediction, genes, up, down = _validation_example()
    keep = keys["cell_id"] != "C3"
    # Retain only four drugs for C2; all other cells retain six.
    keep &= ~((keys["cell_id"] == "C2") & (keys["pert_id"].isin(["D4", "D5"])))
    keys = keys.loc[keep].reset_index(drop=True)
    observed = observed[keep.to_numpy()]
    prediction = prediction[keep.to_numpy()]
    expected_score, expected_cells = reversal_validation_metric(
        type("Data", (), {"keys": keys, "delta": observed})(),
        np.arange(len(keys)), prediction, genes, up, down,
    )
    actual_score, actual_cells = ReversalValidationCache(
        keys, observed, genes, up, down
    ).score(prediction)

    assert np.isnan(actual_score) and np.isnan(expected_score)
    pd.testing.assert_frame_equal(actual_cells, expected_cells)
    assert "FEWER_THAN_FIVE_DRUGS" in set(actual_cells["reason"].dropna())


def test_cache_rejects_prediction_axis_mismatch():
    keys, observed, prediction, genes, up, down = _validation_example()
    cache = ReversalValidationCache(keys, observed, genes, up, down)
    with pytest.raises(ValueError, match="gene axis"):
        cache.score(prediction[:, :-1])


def test_inputs_only_inference_matches_legacy_and_never_reads_targets():
    rng = np.random.default_rng(123)
    rows = 5
    data = SimpleNamespace(
        state=rng.normal(size=(rows, 978)).astype(np.float32),
        fingerprint=rng.integers(0, 2, size=(3, 2048), dtype=np.uint8),
        drug_index=np.array([0, 1, 0, 2, 1], dtype=np.int32),
        exposure=rng.normal(size=(rows, 2)).astype(np.float32),
        delta=rng.normal(size=(rows, 978)).astype(np.float32),
    )

    def tensors(index, prep, device):
        state = (data.state[index] - prep.state_mean) / prep.state_scale
        drug = data.fingerprint[data.drug_index[index]].astype(np.float32)
        exposure = (data.exposure[index] - prep.exposure_mean) / prep.exposure_scale
        target_raw = data.delta[index]
        target = target_raw / prep.gene_scale
        return tuple(torch.from_numpy(value).to(device) for value in
                     (state, drug, exposure, target, target_raw))

    data.tensors = tensors
    prep = SimpleNamespace(
        state_mean=np.zeros(978, dtype=np.float32),
        state_scale=np.ones(978, dtype=np.float32),
        exposure_mean=np.zeros(2, dtype=np.float32),
        exposure_scale=np.ones(2, dtype=np.float32),
        gene_scale=np.linspace(0.5, 1.5, 978, dtype=np.float32),
    )
    torch.manual_seed(123)
    model = SharedEffectInteraction(rank=16).eval()
    positions = np.array([0, 2, 4], dtype=np.int64)
    reference = predict_delta(model, data, positions, prep, torch.device("cpu"), 2)

    class InputOnlyView:
        state = data.state
        fingerprint = data.fingerprint
        drug_index = data.drug_index
        exposure = data.exposure

        @property
        def delta(self):
            raise AssertionError("input-only validation inference accessed response labels")

        @staticmethod
        def tensors(*_args, **_kwargs):
            raise AssertionError("input-only validation inference called the label-bearing tensor API")

    optimized = predict_delta_inputs_only(
        model, InputOnlyView(), positions, prep, torch.device("cpu"), 2)
    np.testing.assert_array_equal(optimized, reference)
