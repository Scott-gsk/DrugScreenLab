import json
import numpy as np
import pandas as pd
import pytest
import torch

from drug_screen.transfer.inference import predict_transfer
from drug_screen.transfer.models import DualEncoderPairModel
from drug_screen.transfer.preprocessing import ExpressionPreprocessor


def _artifact(tmp_path):
    run = tmp_path / "run"; run.mkdir(); expression = np.arange(12., dtype=float).reshape(3, 4)
    genes = np.array(["g1", "g2", "g3", "g4"]); prep = ExpressionPreprocessor.fit(expression, genes, n_features=4)
    x = prep.transform(expression, genes); model = DualEncoderPairModel(hidden_dim=4, epochs=2, seed=3)
    model.fit(x, np.zeros((3, 2048)), np.array([1., 2., 3.]), pdo_ids=np.array(["a", "b", "c"]))
    with torch.no_grad():
        for parameter in model.network.parameters(): parameter.zero_()
    prep.save(run / "prep.npz"); model.save(run / "model.pt"); (run / "frozen_run.json").write_text("{}")
    np.savez(run / "expression.npz", sample_id=np.array(["pdo2", "pdo1"]), ordered_genes=genes, expression=expression[:2])
    pd.DataFrame({"drug_id": ["z", "a"], "canonical_smiles": ["CCO", "CCN"]}).to_parquet(run / "candidates.parquet", index=False)
    return run, model, prep


def test_single_model_prediction_restores_and_stably_ranks(tmp_path):
    run, model, prep = _artifact(tmp_path)
    report = predict_transfer(run / "model.pt", run / "prep.npz", run / "expression.npz", run / "candidates.parquet", run / "out", explain_steps=4)
    ranked = pd.read_parquet(run / "out" / "ranked_predictions.parquet")
    candidates = pd.read_parquet(run / "candidates.parquet")
    x = prep.transform(np.arange(12., dtype=float).reshape(3, 4)[:2], ["g1", "g2", "g3", "g4"])
    from drug_screen.transfer.chemistry import morgan_fingerprint
    expected = model.predict(np.repeat(x, 2, axis=0), np.tile(np.vstack([morgan_fingerprint(s) for s in candidates.canonical_smiles]), (2, 1)))
    assert np.allclose(np.sort(ranked.score), np.sort(expected))
    assert ranked.groupby("sample_id")["rank"].max().eq(2).all()
    assert all(value == ["a", "z"] for value in ranked.groupby("sample_id").drug_id.apply(list))
    assert report["explanations_written"] == 4
    assert json.loads((run / "out" / "prediction_report.json").read_text())["calibrated_uncertainty"] is False


def test_invalid_candidate_smiles_aborts_without_silent_drop(tmp_path):
    run, _, _ = _artifact(tmp_path)
    pd.DataFrame({"drug_id": ["bad"], "canonical_smiles": ["not a smiles"]}).to_parquet(run / "bad.parquet", index=False)
    with pytest.raises(ValueError, match="cannot fingerprint"):
        predict_transfer(run / "model.pt", run / "prep.npz", run / "expression.npz", run / "bad.parquet", run / "out")
