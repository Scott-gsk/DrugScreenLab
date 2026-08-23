import json
from pathlib import Path
from types import SimpleNamespace
import importlib.util

import numpy as np
import pytest

from drug_screen.modeling.exp011_data import (
    CohortDrugBalancedBatchSampler,
    IndexedPairDataset,
    build_fold_indices,
    load_indexed_pair_data,
)
from drug_screen.modeling.exp011_models import target_phase
from drug_screen.modeling.exp011_training import (
    SOURCE_POLICY,
    TARGET_POLICY,
    TrainOnlyResponseStandardizer,
    fit_source_model,
    fit_target_model,
    macro_cohort_drug_spearman,
    attention_backend_for_capability,
)

RUNNER = Path(__file__).resolve().parents[2] / "scripts/modeling/run_exp011_pharmaformer.py"
spec = importlib.util.spec_from_file_location("run_exp011_pharmaformer", RUNNER)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)

ENSEMBLER = Path(__file__).resolve().parents[2] / "scripts/modeling/ensemble_exp011_predictions.py"
ensemble_spec = importlib.util.spec_from_file_location("ensemble_exp011_predictions", ENSEMBLER)
ensembler = importlib.util.module_from_spec(ensemble_spec)
assert ensemble_spec.loader is not None
ensemble_spec.loader.exec_module(ensembler)


def _data(tmp_path: Path):
    archive = tmp_path / "raw_indexed.npz"
    # Fifteen unique PDOs, one response each.  Expression remains a 15 x 3
    # unique matrix; no 19k-gene sample is duplicated in response pairs.
    np.savez_compressed(
        archive,
        expression=np.arange(45, dtype=np.float32).reshape(15, 3),
        ordered_genes=np.asarray(["A", "B", "C"]),
        sample_id=np.asarray([f"s{i}" for i in range(15)]),
        patient_id=np.asarray([f"p{i}" for i in range(15)]),
        pdo_id=np.asarray([f"o{i}" for i in range(15)]),
        cell_line_id=np.asarray([""] * 15),
        drug_id=np.asarray(["d1"]), canonical_smiles=np.asarray(["CC"]),
        inchikey=np.asarray(["IK"]), drug_scaffold=np.asarray(["SC"]),
        pair_sample_idx=np.arange(15), pair_drug_idx=np.zeros(15, dtype=np.int64),
        pair_response=np.asarray([1., 2., 4.] * 5, dtype=np.float32),
        pair_cohort_id=np.asarray(["c1"] * 15),
        pair_response_name=np.asarray(["dss"] * 15), pair_response_scale=np.asarray(["DSS"] * 15),
        pair_higher_is_sensitive=np.ones(15, dtype=bool),
    )
    manifest = tmp_path / "cohort_index_manifest.json"
    manifest.write_text(json.dumps({
        "schema": "drugscreenlab.exp011.raw-indexed-pairs.v1", "raw_npz": archive.name,
        "formal_gene_contract": False, "cohort_id": "c1",
    }), encoding="utf-8")
    return load_indexed_pair_data(manifest)


def test_indexed_dataset_gathers_unique_expression_and_model_builds_patient_folds(tmp_path: Path) -> None:
    data = _data(tmp_path)
    fold = build_fold_indices(data, stage="target", fold_id=0, seed=2)
    assert set(fold.train) | set(fold.validation) | set(fold.test) == set(range(data.pair_count))
    assert not (set(fold.train) & set(fold.test))
    encoded = np.zeros((1, 128), dtype=np.float32)
    dataset = IndexedPairDataset(data, fold.train, encoded)
    assert dataset.expression.shape == (15, 3)
    gene, drug, _, _, _ = dataset[0]
    assert gene.shape == (3,) and drug.shape == (128,)
    # Each patient is in only one model-created partition.
    assert len(fold.group_assignments) == 15


def test_response_standardization_is_train_only_and_requires_support(tmp_path: Path) -> None:
    data = _data(tmp_path)
    transform = TrainOnlyResponseStandardizer(min_train_per_group=3).fit(data, np.asarray([0, 1, 2]))
    held = transform.transform(data, np.asarray([3, 4, 5]))
    np.testing.assert_allclose(held, (np.asarray([1., 2., 4.]) - 7 / 3) / np.std([1., 2., 4.]))
    with pytest.raises(ValueError, match="requires 4"):
        TrainOnlyResponseStandardizer(min_train_per_group=4).fit(data, np.asarray([0, 1, 2]))


def test_balanced_sampler_produces_ranking_pairs_and_target_phases_are_fixed(tmp_path: Path) -> None:
    data = _data(tmp_path)
    dataset = IndexedPairDataset(data, np.asarray([0, 1, 2]), np.zeros((1, 128), dtype=np.float32))
    batch = next(iter(CohortDrugBalancedBatchSampler(dataset, batch_size=2, seed=1)))
    assert len(batch) == 2
    assert target_phase("M3-adapter", 20).name == "adapter_head"
    assert target_phase("M3-adapter", 21).name == "adapter_head_last_transformer"
    assert target_phase("M3-full", 1).name == "all_parameters"


def test_per_drug_macro_spearman_does_not_mix_drug_main_effects() -> None:
    predicted = np.asarray([1., 2., 3., 100., 101., 102.])
    target = np.asarray([3., 2., 1., 102., 101., 100.])
    groups = np.asarray(["drug_a"] * 3 + ["drug_b"] * 3)
    assert np.corrcoef(predicted, target)[0, 1] > .9
    assert macro_cohort_drug_spearman(predicted, target, groups) == pytest.approx(-1.0)


def test_pre_ampere_devices_use_math_attention_without_changing_model_geometry() -> None:
    assert attention_backend_for_capability(7) == "math_sdp_pre_ampere"
    assert attention_backend_for_capability(8) == "default_sdp_ampere_or_newer"


def test_direct_raw_npz_is_marked_development_only(tmp_path: Path) -> None:
    data = _data(tmp_path)
    direct = load_indexed_pair_data(data.manifest_path.parent / "raw_indexed.npz")
    assert direct.manifest["development_input"] is True


def test_top_level_manifest_resolves_selected_raw_cohort(tmp_path: Path) -> None:
    data = _data(tmp_path)
    top = tmp_path / "aggregate.json"
    top.write_text(json.dumps({
        "experiment_id": "EXP-011",
        "cohorts": [{"cohort_id": "c1", "bundle": {"raw_manifest": data.manifest_path.name}}],
    }), encoding="utf-8")
    selected = load_indexed_pair_data(top, cohort_id="c1")
    assert selected.manifest_path == data.manifest_path
    assert selected.manifest["cohort_index_manifest"] == str(top.resolve())
    with pytest.raises(ValueError, match="--cohort-id"):
        load_indexed_pair_data(top)


def test_formal_runner_hard_rejects_direct_raw_npz(tmp_path: Path) -> None:
    data = _data(tmp_path)
    with pytest.raises(ValueError, match="development NPZ"):
        runner.run(SimpleNamespace(manifest=data.manifest_path.parent / "raw_indexed.npz", formal=True))


def test_formal_source_requires_tissue_stratification_field(tmp_path: Path) -> None:
    data = _data(tmp_path)
    with pytest.raises(ValueError, match="sample_tissue"):
        runner.run(SimpleNamespace(manifest=data.manifest_path, formal=True, stage="source"))


def test_prediction_schema_is_evaluator_long_form(tmp_path: Path) -> None:
    data = _data(tmp_path)
    frame = runner._prediction_frame(data, np.asarray([0, 1]), np.asarray([.1, .2]), model_id="M2", fold_id=3,
                                     evaluation_role="gdsc_cold_cell")
    assert frame.columns.tolist() == [
        "experiment_id", "model_id", "evaluation_role", "cohort_id", "fold_id", "split", "patient_id",
        "sample_id", "pdo_id", "cell_line_id", "drug_id", "inchikey", "drug_scaffold", "response_name",
        "response_scale", "higher_is_sensitive", "y_true", "y_pred",
    ]
    external = runner._prediction_frame(data, np.asarray([0]), np.asarray([.1]), model_id="M2", fold_id=3,
                                        evaluation_role="external_exploratory")
    assert external.loc[0, "split"] == "external" and external.loc[0, "fold_id"] == -1


def test_source_and_target_loops_run_on_indexed_pairs(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    data = _data(tmp_path)
    encoded = np.zeros((1, 128), dtype=np.float32)
    source_model = torch.nn.Sequential()  # a two-input model is defined below

    class TwoInput(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gene = torch.nn.Linear(3, 1)
            self.drug = torch.nn.Linear(128, 1)

        def forward(self, gene, drug):
            return self.gene(gene) + self.drug(drug)

    source_fit = fit_source_model(
        TwoInput(), IndexedPairDataset(data, np.asarray([0, 1, 2]), encoded),
        IndexedPairDataset(data, np.asarray([3, 4, 5]), encoded), device="cpu", seed=1,
        policy={**SOURCE_POLICY, "min_epochs": 1, "max_epochs": 2, "patience": 1, "batch_size": 2},
    )
    assert source_fit.selection_metric == "validation_mse" and source_fit.epochs_run <= 2
    standardizer = TrainOnlyResponseStandardizer(min_train_per_group=3).fit(data, np.asarray([0, 1, 2]))
    from dataclasses import replace
    standardized = replace(data, pair_response=standardizer.transform(data, np.arange(data.pair_count)))
    target_fit = fit_target_model(
        TwoInput(), "M4", IndexedPairDataset(standardized, np.asarray([0, 1, 2]), encoded),
        IndexedPairDataset(standardized, np.asarray([3, 4, 5]), encoded), device="cpu", seed=1,
        policy={**TARGET_POLICY, "min_epochs": 1, "max_epochs": 2, "patience": 1, "batch_size": 2},
    )
    assert target_fit.selection_metric == "validation_macro_cohort_drug_spearman" and target_fit.epochs_run <= 2


def test_external_stage_only_strict_loads_checkpoint_and_never_fits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    from drug_screen.foundation.pharmaformer_adapter import TrainOnlyGeneTransform, TrainOnlySmilesEncoder

    data = _data(tmp_path)
    code_path = tmp_path / "bpe.codes"; code_path.write_text("#version: 0.2\nC C\n", encoding="utf-8")
    transform = TrainOnlyGeneTransform().fit(data.expression[:3], data.sample_id[:3].tolist())
    encoder = TrainOnlySmilesEncoder().fit(["CC"], ["d1"], bpe_codes_path=code_path)

    class TwoInput(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.gene = torch.nn.Linear(3, 1); self.drug = torch.nn.Linear(128, 1)
        def forward(self, gene, drug):
            return self.gene(gene) + self.drug(drug)

    checkpoint = tmp_path / "frozen.pt"; checkpoint.write_bytes(b"test checkpoint")
    payload = {"variant": "M2", "gene_contract_sha256": "test-only-geometry", "gene_transform_object": transform,
               "smiles_encoder_object": encoder}
    monkeypatch.setattr(runner, "upstream_checkout", lambda _: object())
    monkeypatch.setattr(runner, "load_exp011_checkpoint", lambda *_args, **_kwargs: (TwoInput(), payload))
    monkeypatch.setattr(runner, "fit_source_model", lambda *_args, **_kwargs: pytest.fail("external must not fit source"))
    monkeypatch.setattr(runner, "fit_target_model", lambda *_args, **_kwargs: pytest.fail("external must not fit target"))
    args = SimpleNamespace(
        manifest=data.manifest_path, cohort_id=None, formal=False, stage="external", evaluation_role="external_exploratory",
        model_id="M2", checkpoint=checkpoint, source_checkpoint=None, fold_id=None, output=tmp_path / "out", device="cpu", seed=1,
        min_train_per_group=3, gene_hidden_size=4, drug_hidden_size=4, feature_dim=4, nhead=2, num_layers=1,
        dim_feedforward=8, dropout=.1, adapter_bottleneck=2, adapter_dropout=.3, allow_test_geometry=True, upstream=tmp_path,
    )
    result = runner.run(args)
    assert result["fit"] is None and result["fold_id"] == -1 and result["checkpoint_strictly_loaded"] is True
    output = __import__("pandas").read_parquet(result["predictions"])
    assert set(output["split"]) == {"external"} and set(output["fold_id"]) == {-1}


def test_source_final_uses_only_five_frozen_outer_epochs_and_passed_gate(tmp_path: Path) -> None:
    summaries = []
    for fold, epoch in enumerate([4, 1, 7, 3, 5]):
        path = tmp_path / f"fold{fold}.json"
        path.write_text(json.dumps({"experiment_id": "EXP-011", "stage": "source", "model_id": "M2", "fold_id": fold,
                                    "fit": {"best_epoch": epoch}}), encoding="utf-8")
        summaries.append(path)
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"format": "exp011_cell_line_gate_v1", "passed": True,
                                "prism_frozen": {"labels_used_for_model_selection": False}}), encoding="utf-8")
    epoch, audit = runner._frozen_source_final_epoch(summaries, gate)
    assert epoch == 4 and audit["post_gate_full_source_refit"] is True
    gate.write_text(json.dumps({"format": "exp011_cell_line_gate_v1", "passed": False,
                                "prism_frozen": {"labels_used_for_model_selection": False}}), encoding="utf-8")
    with pytest.raises(ValueError, match="passed"):
        runner._frozen_source_final_epoch(summaries, gate)


def test_target_stage_refuses_an_outer_fold_source_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    data = _data(tmp_path)

    class TwoInput(torch.nn.Module):
        def forward(self, gene, drug):
            return gene[:, :1] + drug[:, :1]

    monkeypatch.setattr(runner, "upstream_checkout", lambda _: object())
    monkeypatch.setattr(runner, "load_exp011_checkpoint", lambda *_args, **_kwargs: (
        TwoInput(), {"variant": "M2", "model_variant": "CombinedModel", "provenance": {}}
    ))
    args = SimpleNamespace(
        manifest=data.manifest_path, cohort_id=None, formal=False, stage="target", evaluation_role="pdo_primary", model_id="M2",
        checkpoint=None, source_checkpoint=tmp_path / "outer_fold.pt", fold_id=0, output=tmp_path / "out", device="cpu", seed=1,
        min_train_per_group=3, gene_hidden_size=4, drug_hidden_size=4, feature_dim=4, nhead=2, num_layers=1,
        dim_feedforward=8, dropout=.1, adapter_bottleneck=2, adapter_dropout=.3, allow_test_geometry=True, upstream=tmp_path,
    )
    with pytest.raises(ValueError, match="post-gate full-source"):
        runner.run(args)


def test_five_fold_ensemble_requires_exact_metadata_and_uses_unweighted_mean(tmp_path: Path) -> None:
    rows = []
    for sample, response in [("s1", 1.0), ("s2", 2.0)]:
        rows.append({
            "experiment_id": "EXP-011", "model_id": "M2", "evaluation_role": "prism_frozen", "cohort_id": "prism_crc",
            "fold_id": -1, "split": "external", "patient_id": "", "sample_id": sample, "pdo_id": "", "cell_line_id": sample,
            "drug_id": "d1", "inchikey": "IK", "drug_scaffold": "SC", "response_name": "logfc", "response_scale": "-log2FC",
            "higher_is_sensitive": True, "y_true": response, "y_pred": 0.0,
        })
    paths = []
    for fold in range(5):
        frame = __import__("pandas").DataFrame(rows)
        frame["y_pred"] = [fold, fold + 10]
        path = tmp_path / f"fold{fold}.parquet"; frame.to_parquet(path, index=False); paths.append(path)
    result = ensembler.ensemble_predictions(paths, model_id="M2")
    assert result.columns.tolist() == list(ensembler.PREDICTION_COLUMNS)
    assert result.y_pred.tolist() == [2.0, 12.0] and set(result.model_id) == {"M2"}
    mismatched = __import__("pandas").read_parquet(paths[-1]); mismatched.loc[0, "y_true"] = 999.0; mismatched.to_parquet(paths[-1], index=False)
    with pytest.raises(ValueError, match="observed labels"):
        ensembler.ensemble_predictions(paths)
