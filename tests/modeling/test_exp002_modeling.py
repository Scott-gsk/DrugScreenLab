from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from drug_screen.modeling import exp002


def _gene_digest(genes: list[str]) -> str:
    digest = sha256()
    for gene in genes:
        digest.update(gene.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _record(
    *, scheme: str, split: str, index: int, context: str, drug: str, offset: float
) -> exp002.Delta978Record:
    values = (np.arange(exp002.GENE_COUNT, dtype=np.float32) / 100 + offset).astype(
        np.float32
    )
    return exp002.Delta978Record(
        experiment_id=exp002.EXPERIMENT_ID,
        split_scheme=scheme,
        split=split,
        treatment_inst_id=f"{scheme}-{split}-t{index}",
        drug_id=drug,
        context_id=context,
        replicate_family_id=f"{scheme}-{split}-family-{index}",
        rna_plate="plate-not-a-feature",
        dose=("1", "uM"),
        time=("24", "h"),
        control_inst_ids=(f"{scheme}-{split}-control-{index}",),
        ordered_gene_ids_sha256=exp002.ORDERED_GENE_IDS_SHA256,
        delta978=values,
    )


def _config() -> dict[str, object]:
    return {
        "frozen_contract": {
            "experiment_id": exp002.EXPERIMENT_ID,
            "dataset_registry_id": exp002.DATASET_ID,
            "derived_storage_registry_id": exp002.DERIVED_STORAGE_ID,
            "contract_artifact_sha256": exp002.CONTRACT_ARTIFACT_SHA256,
            "ordered_gene_ids_sha256": exp002.ORDERED_GENE_IDS_SHA256,
            "gene_count": exp002.GENE_COUNT,
            "split_schemes": list(exp002.SPLIT_SCHEMES),
        },
        "evaluation": {
            "metrics": list(exp002.METRICS),
            "primary_metric": "pearson",
            "direction_zero_epsilon": 0.0,
            "unseen_context_fallback": "global_train_mean",
            "replicate_aggregation": "arithmetic_mean",
            "bootstrap_resamples": 20,
            "bootstrap_seed": 20260811,
            "paired_improvement_threshold": 0.0,
            "seed_consistency_rule": "all_seeds_positive_paired_improvement",
            "minimum_eligible_group_count": 2,
        },
        "reproducibility": {"seeds": [17, 29]},
    }


@pytest.fixture
def synthetic_contract(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    genes = [f"gene-{index}" for index in range(exp002.GENE_COUNT)]
    monkeypatch.setattr(exp002, "ORDERED_GENE_IDS_SHA256", _gene_digest(genes))
    return genes


def test_baselines_use_train_only_and_unseen_context_falls_back(synthetic_contract):
    records = [
        _record(scheme="cold_drug", split="train", index=1, context="A", drug="d1", offset=1),
        _record(scheme="cold_drug", split="train", index=2, context="A", drug="d2", offset=3),
        _record(scheme="cold_drug", split="train", index=3, context="B", drug="d3", offset=8),
    ]
    test = [
        _record(scheme="cold_drug", split="test", index=4, context="A", drug="d4", offset=100),
        _record(scheme="cold_drug", split="test", index=5, context="unseen", drug="d5", offset=200),
    ]
    global_model = exp002.GlobalMeanBaseline().fit(records)
    context_model = exp002.ContextConditionedMeanBaseline().fit(records)
    global_prediction = global_model.predict(test)
    context_prediction, fallback = context_model.predict(test)
    assert fallback.tolist() == [False, True]
    np.testing.assert_array_equal(context_prediction[1], global_prediction[1])
    assert context_prediction[0, 0] == pytest.approx(2.0)


def test_dataset_rejects_gene_order_and_manifest_leakage(synthetic_contract):
    record = _record(
        scheme="cold_drug", split="train", index=1, context="A", drug="d1", offset=0
    )
    with pytest.raises(ValueError, match="frozen exact-978 order"):
        exp002.Delta978Dataset([record], list(reversed(synthetic_contract)))
    train = exp002.Delta978Dataset([record], synthetic_contract)
    leaked = exp002.Delta978Dataset(
        [
            exp002.Delta978Record(
                **{
                    **record.__dict__,
                    "split": "test",
                    "treatment_inst_id": "other-treatment",
                }
            )
        ],
        synthetic_contract,
    )
    with pytest.raises(ValueError, match="family leakage"):
        exp002.assert_manifest_isolation([train, leaked], "cold_drug")


def _write_materialization(root: Path, genes: list[str]) -> None:
    root.mkdir()
    (root / "genes.json").write_text(json.dumps(genes), encoding="utf-8")
    for scheme in exp002.SPLIT_SCHEMES:
        (root / scheme).mkdir()
        values = {
            "train": [
                _record(scheme=scheme, split="train", index=1, context="A", drug="d1", offset=1),
                _record(scheme=scheme, split="train", index=2, context="A", drug="d2", offset=2),
            ],
            "validation": [
                _record(scheme=scheme, split="validation", index=3, context="V", drug="dv", offset=3)
            ],
            "test": [
                _record(scheme=scheme, split="test", index=4, context="T" if scheme == "cold_context" else "A", drug="dt1", offset=4),
                _record(scheme=scheme, split="test", index=5, context="U" if scheme == "cold_context" else "A", drug="dt2", offset=5),
            ],
        }
        for split, records in values.items():
            payloads = []
            for record in records:
                payload = dict(record.__dict__)
                payload["delta978"] = record.delta978.tolist()
                payloads.append(json.dumps(payload, sort_keys=True))
            (root / scheme / f"{split}.jsonl").write_text(
                "\n".join(payloads) + "\n", encoding="utf-8"
            )
    files = {}
    for path in [root / "genes.json", *root.glob("*/*.jsonl")]:
        files[str(path.relative_to(root))] = sha256(path.read_bytes()).hexdigest()
    metadata = {
        "experiment_id": exp002.EXPERIMENT_ID,
        "source_contract_artifact_sha256": exp002.CONTRACT_ARTIFACT_SHA256,
        "ordered_gene_ids_sha256": exp002.ORDERED_GENE_IDS_SHA256,
        "gene_count": exp002.GENE_COUNT,
        "files": files,
    }
    (root / "materialization.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )


def test_runner_is_reproducible_and_reports_cold_context_fallback(
    tmp_path: Path, synthetic_contract
):
    root = tmp_path / "materialized"
    _write_materialization(root, synthetic_contract)
    config = _config()
    first = exp002.run_evaluation(config, root)
    second = exp002.run_evaluation(config, root)
    assert first == second
    assert first["results"]["cold_context"]["context_fallback_count"] == 2
    assert first["results"]["cold_context"]["dropped_count"] == 0
    assert {row["seed"] for row in first["results"]["cold_drug"]["rows"]} == {17, 29}
    assert any(
        summary.get("paired_bootstrap") is not None
        for summary in first["results"]["cold_drug"]["summaries"]
    )


def test_committed_config_is_frozen():
    path = Path("configs/experiments/EXP-002/baselines.json")
    config = exp002.load_exp002_config(path)
    assert config["model"]["plate_feature_allowed"] is False
    broken = _config()
    broken["evaluation"]["direction_zero_epsilon"] = 0.1
    with pytest.raises(ValueError, match="frozen at 0.0"):
        exp002._validate_config(broken)
