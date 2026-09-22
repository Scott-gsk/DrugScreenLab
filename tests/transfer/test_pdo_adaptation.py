import json
import numpy as np
import pandas as pd

from drug_screen.transfer.bundles import IndexedBundle
from drug_screen.transfer.models import DualEncoderPairModel
from drug_screen.transfer.pdo_adaptation import (
    common_gene_ids, common_source_target_gene_space, fit_target_preprocessor, representation_matrix,
    run_pdo_adaptation, validate_adaptation_config,
    strict_full_inchikey_drug_ids, train_common_space_source_checkpoint, _inner_ndcg3_strict7,
)
from drug_screen.transfer.preprocessing import ExpressionPreprocessor


def _bundle():
    patients = [f"p{i}" for i in range(5)]
    samples = pd.DataFrame({"sample_id": [f"s{i}" for i in range(5)], "patient_id": patients})
    drugs = pd.DataFrame({"drug_id": [f"d{i}" for i in range(7)], "canonical_smiles": ["CCO", "CCN", "CCC", "CCCl", "CCBr", "CCF", "CCI"],
                          "inchikey": [f"KEY{i:011d}-UHFFFAOYSA-N" for i in range(7)]})
    pairs = pd.DataFrame([{"sample_id": s, "drug_id": d, "response": float(i + j), "response_name": "dss", "higher_is_sensitive": True}
                          for i, s in enumerate(samples.sample_id) for j, d in enumerate(drugs.drug_id)])
    return IndexedBundle("target", "PDO_DEVELOPMENT", samples, drugs, pairs,
                         np.arange(20., dtype=float).reshape(5, 4), np.array(samples.sample_id),
                         np.array(["g1", "g2", "g3", "g4"]), "FPKM", True, "dss")


def _source(path, bundle):
    path.mkdir(); prep = ExpressionPreprocessor.fit(np.log2(bundle.expression + 1), bundle.gene_ids, n_features=4)
    x = prep.transform(np.log2(bundle.expression + 1), bundle.gene_ids)
    model = DualEncoderPairModel(hidden_dim=32, epochs=1, seed=3)
    model.fit(x, np.zeros((5, 2048)), np.arange(5., dtype=float), pdo_ids=np.array(["a", "b", "c", "d", "e"]))
    prep.save(path / "expression_transform.npz"); model.save(path / "source_model.pt")
    (path / "frozen_run.json").write_text(json.dumps({"source_drug_ids": [f"KEY{i:011d}-UHFFFAOYSA-N" for i in range(7)], "common_source_target_gene_ids": list(prep.gene_ids)}))


def test_representations_keep_common_gene_order_and_frozen_control():
    genes = np.array(["g1", "g2", "g3", "g4"]); expression = np.arange(20., dtype=float).reshape(5, 4)
    source = ExpressionPreprocessor.fit(np.log2(expression + 1), genes, n_features=4)
    assert common_gene_ids(source, genes) == tuple(source.gene_ids)
    ranked = representation_matrix(expression, genes, kind="samplewise_rank")
    assert np.allclose(np.sort(ranked[0]), [.25, .5, .75, 1.])
    frozen, shown = fit_target_preprocessor(expression[:3], genes, source, representation="source_log")
    assert frozen == source and np.allclose(shown, np.log2(expression[:3] + 1))
    assert common_source_target_gene_space(genes, genes, n_features=4) == tuple(genes)


def test_common_source_receipt_records_selected_checkpoint_inputs(tmp_path):
    bundle = _bundle()
    receipt = train_common_space_source_checkpoint(
        bundle, bundle.gene_ids, output_dir=tmp_path / "common", seed=7,
        epochs=1, device="cpu", n_features=4,
    )
    fitted = ExpressionPreprocessor.load(tmp_path / "common" / "expression_transform.npz")
    assert tuple(receipt["common_source_target_gene_ids"]) == fitted.gene_ids
    assert set(receipt["common_source_target_gene_ids"]) <= set(receipt["common_source_target_candidate_gene_ids"])


def test_strict_seven_drug_filter_and_inner_endpoint_reject_incomplete_panels():
    bundle = _bundle()
    source_keys = set(bundle.drugs.inchikey)
    assert strict_full_inchikey_drug_ids(bundle, source_keys) == set(bundle.drugs.drug_id)
    rows = pd.DataFrame({"patient_id": ["p"] * 7, "drug_id": bundle.drugs.drug_id, "response": range(7)})
    assert _inner_ndcg3_strict7(rows, np.arange(7.)) == 1.
    try:
        _inner_ndcg3_strict7(rows.iloc[:-1], np.arange(6.))
    except ValueError as error:
        assert "exactly seven" in str(error)
    else:
        raise AssertionError("incomplete panel must not be inner-selected")


def test_runner_writes_development_only_artifacts(tmp_path):
    bundle = _bundle(); source = tmp_path / "source"; _source(source, bundle)
    config = {"cohort_id": "target", "outer_folds": 5, "inner_folds": 3, "epochs": 1,
              "final_init_seeds": [1, 2, 3],
              "candidates": [{"representation": "source_log", "mode": "head_only", "shrinkage": 0}],
              "learning_curve_budgets": [.1, .25, .5, 1.]}
    assert validate_adaptation_config(config)["cohort_id"] == "target"
    (tmp_path / "none.json").write_text("{}")
    result = run_pdo_adaptation(config, source_output=source, data_root=tmp_path, registry_path=tmp_path / "none.json",
                                output_dir=tmp_path / "out", loader=lambda *args, **kwargs: bundle)
    assert result["status"] == "DEVELOPMENT_ONLY_NOT_EXTERNAL"
    assert (tmp_path / "out" / "outer_oof_predictions.parquet").is_file()
    assert len(result["selected_candidates"]) == 5
