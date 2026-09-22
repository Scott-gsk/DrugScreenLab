import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from drug_screen.transfer.bundles import IndexedBundle
from drug_screen.transfer.models import DualEncoderPairModel
from drug_screen.transfer.preprocessing import ExpressionPreprocessor
from drug_screen.transfer.transfer_experiment import _move_to_device, run_transfer_experiment


def _bundle():
    patients = [f"p{i}" for i in range(5)]
    samples = pd.DataFrame({"sample_id": [f"s{i}" for i in range(5)], "patient_id": patients})
    drugs = pd.DataFrame({"drug_id": [f"d{i}" for i in range(10)], "canonical_smiles": ["CCO" if i % 2 else "CCN" for i in range(10)]})
    pairs = pd.DataFrame([{"sample_id": s, "drug_id": d, "response": float(i + j), "response_name": "dss", "higher_is_sensitive": True}
                          for i, s in enumerate(samples.sample_id) for j, d in enumerate(drugs.drug_id)])
    return IndexedBundle("target", "PDO_DEVELOPMENT", samples, drugs, pairs, np.arange(20., dtype=float).reshape(5, 4),
                         np.array(samples.sample_id), np.array(["g1", "g2", "g3", "g4"]), "unit", True, "dss")


def _source(path, bundle):
    path.mkdir(); prep = ExpressionPreprocessor.fit(bundle.expression, bundle.gene_ids, n_features=4)
    x = prep.transform(bundle.expression, bundle.gene_ids)
    # A compact fitted artifact stands in for source_training.py output.
    model = DualEncoderPairModel(hidden_dim=32, epochs=1, seed=1)
    drugs = np.zeros((5, 2048)); model.fit(x, drugs, np.arange(5., dtype=float), pdo_ids=np.array(["a", "b", "c", "d", "e"]))
    prep.save(path / "expression_transform.npz"); model.save(path / "source_model.pt")
    (path / "frozen_run.json").write_text(json.dumps({"source_drug_ids": [f"d{i}" for i in range(8)]}))


def test_source_bound_target_runner_writes_known_only_metrics_and_oof(tmp_path):
    registry = tmp_path / "registry.json"; registry.write_text("{}")
    admission = tmp_path / "admission.json"; admission.write_text(json.dumps({"cohort_id": "target", "status": "EXPLORATORY_ADMITTED", "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest()}))
    bundle = _bundle(); source = tmp_path / "source"; _source(source, bundle)
    result = run_transfer_experiment({"cohort_id": "target", "seeds": [4], "bootstrap_resamples": 2}, target_admission=admission,
        source_output=source, data_root=tmp_path, output_dir=tmp_path / "out", registry_path=registry, loader=lambda *args, **kwargs: bundle)
    assert result["bootstrap_label"] == "SMOKE_ONLY"
    oof = pd.read_parquet(tmp_path / "out" / "outer_oof_predictions.parquet")
    assert {"patient_id", "model_id", "drug_id", "fold", "algorithm", "observed", "predicted", "baseline"} <= set(oof)
    assert oof.groupby("patient_id").fold.nunique().eq(1).all()
    assert set(oof.source_drug_status) == {"known", "unknown"}
    assert result["metrics"]["M1_seed_4"]["known_primary"]["n_drugs"] == 8
    assert "M1_source_scaled_seed_4" in set(oof.algorithm)
    assert set(result["paired_m3_comparisons"]["seed_4"]) == {"M1", "M1_source_scaled"}
    frozen = json.loads((tmp_path / "out" / "frozen_config.json").read_text())
    assert set(frozen["source_artifact_sha256"]) == {"frozen_run.json", "source_model.pt", "expression_transform.npz"}
    assert result["runtime_device"] == "cpu"


def test_runner_rejects_source_without_drug_identity_set(tmp_path):
    registry = tmp_path / "registry.json"; registry.write_text("{}")
    admission = tmp_path / "admission.json"; admission.write_text(json.dumps({"cohort_id": "target", "status": "DEVELOPMENT_ADMITTED", "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest()}))
    source = tmp_path / "source"; _source(source, _bundle()); (source / "frozen_run.json").write_text("{}")
    with pytest.raises(ValueError, match="source_drug_ids"):
        run_transfer_experiment({"cohort_id": "target"}, target_admission=admission, source_output=source,
            data_root=tmp_path, output_dir=tmp_path / "out", registry_path=registry, loader=lambda *args, **kwargs: _bundle())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_loaded_source_checkpoint_moves_to_runner_cuda_device(tmp_path):
    source = tmp_path / "source"; _source(source, _bundle())
    loaded = DualEncoderPairModel.load(source / "source_model.pt")
    moved = _move_to_device(loaded, "cuda")
    assert moved.device.type == "cuda"
    assert next(moved.network.parameters()).device.type == "cuda"
