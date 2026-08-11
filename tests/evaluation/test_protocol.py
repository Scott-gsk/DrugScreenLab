import math

import pytest

from drug_screen.evaluation.protocol import (
    EvaluationRecord,
    StratifiedMetricRecord,
    assert_cold_split_isolation,
    assert_stratified_reporting_complete,
    assert_test_label_isolation,
    bootstrap_macro_mean,
    compute_vector_metrics,
    paired_group_bootstrap_difference,
)

METRICS = ("pearson", "spearman", "rmse", "mae", "direction_accuracy")


def _control(
    sample_id: str,
    family: str,
    context: str = "A375|24h",
    match_key: str | None = None,
) -> EvaluationRecord:
    return EvaluationRecord(
        sample_id, None, context, None, family, True, match_key or context
    )


def _treatment(
    sample_id: str,
    compound: str,
    control_id: str,
    family: str,
    context: str = "A375|24h",
    match_key: str | None = None,
) -> EvaluationRecord:
    return EvaluationRecord(
        sample_id, compound, context, control_id, family, False, match_key or context
    )


def test_cold_drug_isolates_compounds_controls_and_replicate_families():
    records = [_control("c1", "f1"), _treatment("t1", "CMP:1", "c1", "f1"), _control("c2", "f2"), _treatment("t2", "CMP:2", "c2", "f2")]
    assert_cold_split_isolation(records, {"c1": "train", "t1": "train", "c2": "test", "t2": "test"}, "cold_drug")

    leaked = records + [_control("c3", "f3"), _treatment("t3", "CMP:1", "c3", "f3")]
    with pytest.raises(ValueError, match="compound leakage"):
        assert_cold_split_isolation(
            leaked,
            {"c1": "train", "t1": "train", "c2": "test", "t2": "test", "c3": "test", "t3": "test"},
            "cold_drug",
        )


def test_cold_context_isolates_context_but_not_compound_by_definition():
    records = [_control("c1", "f1", "A375|24h"), _treatment("t1", "CMP:1", "c1", "f1", "A375|24h"), _control("c2", "f2", "MCF7|24h"), _treatment("t2", "CMP:1", "c2", "f2", "MCF7|24h")]
    assert_cold_split_isolation(records, {"c1": "train", "t1": "train", "c2": "test", "t2": "test"}, "cold_context")

    with pytest.raises(ValueError, match="replicate-family leakage"):
        assert_cold_split_isolation(records, {"c1": "train", "t1": "test", "c2": "test", "t2": "test"}, "cold_context")

    context_leak = records + [
        _control("c3", "f3", "A375|24h"),
        _treatment("t3", "CMP:2", "c3", "f3", "A375|24h"),
    ]
    with pytest.raises(ValueError, match="context leakage"):
        assert_cold_split_isolation(
            context_leak,
            {
                "c1": "train",
                "t1": "train",
                "c2": "test",
                "t2": "test",
                "c3": "test",
                "t3": "test",
            },
            "cold_context",
        )


def test_replicate_family_isolation_is_always_required():
    records = [_control("c1", "f1"), _treatment("t1", "CMP:1", "c1", "f1"), _treatment("t2", "CMP:2", "c1", "f1")]
    with pytest.raises(ValueError, match="replicate-family leakage"):
        assert_cold_split_isolation(records, {"c1": "train", "t1": "train", "t2": "test"}, "cold_drug")


def test_matched_control_requires_the_same_canonical_matching_key():
    records = [
        _control("c1", "f1", match_key="plate-1|A375|24h"),
        _treatment(
            "t1", "CMP:1", "c1", "f1", match_key="plate-2|A375|24h"
        ),
    ]
    with pytest.raises(ValueError, match="key mismatch"):
        assert_cold_split_isolation(
            records, {"c1": "test", "t1": "test"}, "cold_drug"
        )


def test_test_labels_are_unavailable_to_selection_until_configuration_lock():
    assignments = {
        "train-1": "train",
        "val-1": "validation",
        "test-1": "test",
        "test-2": "test",
    }
    assert_test_label_isolation(
        ["train-1", "val-1"],
        ["test-1"],
        assignments,
        configuration_locked=True,
    )
    with pytest.raises(ValueError, match="test-label leakage"):
        assert_test_label_isolation(
            ["test-2"], ["test-1"], assignments, configuration_locked=True
        )
    with pytest.raises(ValueError, match="locked"):
        assert_test_label_isolation(
            ["val-1"], ["test-1"], assignments, configuration_locked=False
        )
    with pytest.raises(ValueError, match="test-only"):
        assert_test_label_isolation(
            ["train-1"], ["val-1"], assignments, configuration_locked=True
        )


def test_vector_metrics_and_bootstrap_are_deterministic():
    genes = ["g1", "g2", "g3"]
    metrics = compute_vector_metrics(
        [1.0, -2.0, 3.0],
        [1.0, -1.0, -3.0],
        observed_gene_ids=genes,
        predicted_gene_ids=genes,
        expected_gene_ids=genes,
        expected_gene_count=3,
    )
    assert metrics.pearson < 1
    assert metrics.direction_accuracy == pytest.approx(2 / 3)
    assert metrics.rmse == pytest.approx(math.sqrt(37 / 3))
    interval = bootstrap_macro_mean([0.1, 0.4, 0.8], seed=7, resamples=100)
    assert interval == bootstrap_macro_mean([0.1, 0.4, 0.8], seed=7, resamples=100)


def test_vector_metrics_reject_wrong_gene_universe_or_undefined_direction():
    with pytest.raises(ValueError, match="exactly 978"):
        compute_vector_metrics(
            [1.0],
            [1.0],
            observed_gene_ids=["g1"],
            predicted_gene_ids=["g1"],
            expected_gene_ids=["g1"],
        )
    with pytest.raises(ValueError, match="eligible"):
        compute_vector_metrics(
            [0.0, 0.0],
            [0.0, 0.0],
            observed_gene_ids=["g1", "g2"],
            predicted_gene_ids=["g1", "g2"],
            expected_gene_ids=["g1", "g2"],
            expected_gene_count=2,
        )


def test_vector_metrics_reject_gene_reordering_constant_vectors_and_threshold_tuning():
    genes = ["g1", "g2"]
    kwargs = {
        "observed_gene_ids": genes,
        "expected_gene_ids": genes,
        "expected_gene_count": 2,
    }
    with pytest.raises(ValueError, match="identity/order"):
        compute_vector_metrics(
            [1.0, 2.0],
            [1.0, 2.0],
            predicted_gene_ids=list(reversed(genes)),
            **kwargs,
        )
    with pytest.raises(ValueError, match="constant vectors"):
        compute_vector_metrics(
            [1.0, 1.0], [1.0, 2.0], predicted_gene_ids=genes, **kwargs
        )
    with pytest.raises(ValueError, match="frozen"):
        compute_vector_metrics(
            [1.0, 2.0],
            [1.0, 2.0],
            predicted_gene_ids=genes,
            direction_zero_epsilon=0.1,
            **kwargs,
        )
    with pytest.raises(ValueError, match="non-empty"):
        compute_vector_metrics(
            [1.0, 2.0],
            [1.0, 2.0],
            observed_gene_ids=["g1", ""],
            predicted_gene_ids=genes,
            expected_gene_ids=genes,
            expected_gene_count=2,
        )


def test_paired_group_bootstrap_resamples_groups_and_preserves_pairing():
    interval = paired_group_bootstrap_difference(
        [0.8, 0.6, 0.5],
        [0.5, 0.4, 0.7],
        ["drug-1", "drug-1", "drug-2"],
        higher_is_better=True,
        seed=11,
        resamples=200,
    )
    assert interval.point_estimate == pytest.approx(0.025)
    assert interval == paired_group_bootstrap_difference(
        [0.8, 0.6, 0.5],
        [0.5, 0.4, 0.7],
        ["drug-1", "drug-1", "drug-2"],
        higher_is_better=True,
        seed=11,
        resamples=200,
    )
    lower_is_better = paired_group_bootstrap_difference(
        [1.0, 2.0],
        [2.0, 4.0],
        ["context-1", "context-2"],
        higher_is_better=False,
        seed=3,
        resamples=50,
    )
    assert lower_is_better.point_estimate == pytest.approx(1.5)
    with pytest.raises(ValueError, match="at least two"):
        paired_group_bootstrap_difference(
            [0.8, 0.6],
            [0.5, 0.4],
            ["drug-1", "drug-1"],
            higher_is_better=True,
            seed=11,
        )


def _complete_stratified_records() -> list[StratifiedMetricRecord]:
    return [
        StratifiedMetricRecord(
            evaluation_unit_id=f"{stratum}-{seed}-{metric}",
            group_id="group-1",
            drug_id="CMP:1",
            context_id="A375|24h",
            gene_stratum="exact-978",
            ood_stratum=stratum,
            failure_stratum="none",
            seed=seed,
            metric_name=metric,
            candidate_value=0.5,
            baseline_value=0.4,
        )
        for stratum in ("cold_drug", "cold_context")
        for seed in (1, 2)
        for metric in METRICS
    ]


def test_stratified_reporting_requires_every_ood_seed_and_metric_cell():
    records = _complete_stratified_records()
    assert_stratified_reporting_complete(records, required_seeds=[1, 2])
    with pytest.raises(ValueError, match="missing stratified metric cells"):
        assert_stratified_reporting_complete(records[:-1], required_seeds=[1, 2])
    with pytest.raises(ValueError, match="at least two unique"):
        assert_stratified_reporting_complete(records, required_seeds=[1])


def test_stratified_reporting_rejects_unregistered_seed_and_empty_failure_stratum():
    records = _complete_stratified_records()
    extra = records[0]
    records.append(
        StratifiedMetricRecord(
            **{**extra.__dict__, "evaluation_unit_id": "extra", "seed": 3}
        )
    )
    with pytest.raises(ValueError, match="unregistered result seeds"):
        assert_stratified_reporting_complete(records, required_seeds=[1, 2])
    with pytest.raises(ValueError, match="failure_stratum"):
        StratifiedMetricRecord(**{**extra.__dict__, "failure_stratum": ""})
    with pytest.raises(ValueError, match="rows must be unique"):
        assert_stratified_reporting_complete(
            _complete_stratified_records() + [_complete_stratified_records()[0]],
            required_seeds=[1, 2],
        )
