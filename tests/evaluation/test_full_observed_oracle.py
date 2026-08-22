from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from drug_screen.evaluation.full_observed_oracle import (
    CANONICAL_CONDITIONS,
    CRC_EXACT_CONTEXTS,
    NULL_REPEATS,
    NULL_SEED,
    assign_matched_controls,
    choose_primary_condition,
    compute_delta978,
    coverage_by_condition,
    evaluate_oracle_frame,
    oracle_near_null,
    predicted_oracle_gap,
    response_blind_eligible_pert_ids,
    score_reversal,
    select_canonical_instances,
    stratified_support_summary,
    weighted_ks_reversal,
)


def _instances() -> pd.DataFrame:
    rows = [
        # plate A: HT29 24h, vehicle present
        ("t1", "P_A", "BRD-AAA", "trt_cp", 10.0, "um", 24.0, "h", "HT29"),
        ("t2", "P_A", "BRD-AAA", "trt_cp", 10.0, "um", 24.0, "h", "HT29"),
        ("t3", "P_A", "BRD-BBB", "trt_cp", 10.0, "um", 24.0, "h", "HT29"),
        ("c1", "P_A", "DMSO", "ctl_vehicle", 0.1, "%", 24.0, "h", "HT29"),
        ("c2", "P_A", "DMSO", "ctl_vehicle", 0.1, "%", 24.0, "h", "HT29"),
        # plate B: HCT116 24h, vehicle missing, untreated present
        ("t4", "P_B", "BRD-AAA", "trt_cp", 10.0, "um", 24.0, "h", "HCT116"),
        ("u1", "P_B", "Untrt", "ctl_untrt", np.nan, "-666", 24.0, "h", "HCT116"),
        # plate C: HT29 6h, vehicle present
        ("t5", "P_C", "BRD-AAA", "trt_cp", 10.0, "um", 6.0, "h", "HT29"),
        ("c3", "P_C", "DMSO", "ctl_vehicle", 0.1, "%", 6.0, "h", "HT29"),
        # plate D: no matched control
        ("t6", "P_D", "BRD-CCC", "trt_cp", 10.0, "um", 24.0, "h", "SW620"),
        # non-CRC / non-canonical
        ("t7", "P_E", "BRD-AAA", "trt_cp", 10.0, "um", 24.0, "h", "MCF7"),
        ("t8", "P_F", "BRD-AAA", "trt_cp", 1.0, "um", 24.0, "h", "HT29"),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "inst_id",
            "rna_plate",
            "pert_id",
            "pert_type",
            "pert_dose",
            "pert_dose_unit",
            "pert_time",
            "pert_time_unit",
            "cell_id",
        ],
    )
    frame["_cache_row"] = np.arange(len(frame), dtype=np.int64)
    return frame


def test_canonical_selection_does_not_mix_dose_time() -> None:
    selected = select_canonical_instances(
        _instances(),
        contexts=CRC_EXACT_CONTEXTS,
        pert_ids={"BRD-AAA", "BRD-BBB", "BRD-CCC"},
        dose_um=10.0,
        time_h=24.0,
    )
    assert set(selected["pert_time"].astype(float)) == {24.0}
    assert set(selected["cell_id"]) <= set(CRC_EXACT_CONTEXTS)
    assert "MCF7" not in set(selected["cell_id"])
    assert 1.0 not in set(pd.to_numeric(selected["pert_dose"]))


def test_matched_control_prefers_vehicle_then_untrt_and_drops_unmatched() -> None:
    instances = _instances()
    selected = select_canonical_instances(
        instances,
        contexts=CRC_EXACT_CONTEXTS,
        pert_ids={"BRD-AAA", "BRD-BBB", "BRD-CCC"},
        dose_um=10.0,
        time_h=24.0,
    )
    matched, audit = assign_matched_controls(selected, instances)
    assert set(matched["pert_id"]) == {"BRD-AAA", "BRD-BBB"}
    ht29 = matched.loc[matched["cell_id"].eq("HT29")]
    assert set(ht29["control_type"]) == {"ctl_vehicle"}
    hct = matched.loc[matched["cell_id"].eq("HCT116")]
    assert set(hct["control_type"]) == {"ctl_untrt"}
    assert audit["dropped_unmatched_treatment_rows"] == 1
    assert audit["dropped_unmatched_unique_pairs"] == 1
    assert "SW620" not in set(matched["cell_id"])


def test_primary_condition_is_response_blind_max_coverage() -> None:
    instances = _instances()
    coverage = coverage_by_condition(
        instances,
        contexts={"HT29", "HCT116", "SW620"},
        pert_ids={"BRD-AAA", "BRD-BBB", "BRD-CCC"},
    )
    assert coverage["10uM_24h"]["unique_pairs_with_matched_control"] == 3
    assert coverage["10uM_6h"]["unique_pairs_with_matched_control"] == 1
    primary, sensitivity = choose_primary_condition(coverage)
    assert primary == "10uM_24h"
    assert sensitivity == "10uM_6h"
    assert set(CANONICAL_CONDITIONS) == {"10uM_6h", "10uM_24h"}


def test_delta978_is_mean_treatment_minus_mean_matched_control() -> None:
    instances = _instances()
    cache = np.zeros((len(instances), 4), dtype=np.float32)
    cache[0] = [4.0, 4.0, 4.0, 4.0]  # t1
    cache[1] = [6.0, 6.0, 6.0, 6.0]  # t2
    cache[2] = [3.0, 3.0, 3.0, 3.0]  # t3
    cache[3] = [1.0, 1.0, 1.0, 1.0]  # vehicle
    cache[4] = [3.0, 3.0, 3.0, 3.0]  # vehicle
    selected = select_canonical_instances(
        instances,
        contexts={"HT29"},
        pert_ids={"BRD-AAA", "BRD-BBB"},
        dose_um=10.0,
        time_h=24.0,
    )
    matched, _ = assign_matched_controls(selected, instances)
    delta = compute_delta978(cache, matched)
    aaa = delta.loc[delta["pert_id"].eq("BRD-AAA")].iloc[0]
    assert aaa["treatment_rows"] == 2
    assert aaa["control_rows"] == 2
    np.testing.assert_allclose(aaa["delta978"], np.full(4, 3.0))


def test_identity_eligibility_ignores_prism_values() -> None:
    bridge = pd.DataFrame(
        {
            "lincs_pert_id": ["BRD-AAA", "BRD-BBB", "BRD-CCC"],
            "match_status": ["MATCHED_IDENTITY", "MATCHED_IDENTITY", "AMBIGUOUS_OR_UNMATCHED"],
            "match_method": ["exact_pert_id", "exact_inchi_key", "ambiguous_alias"],
        }
    )
    registry = {
        "drugs": [
            {"pert_id": "BRD-AAA", "broad_inference_eligible": True},
            {"pert_id": "BRD-BBB", "broad_inference_eligible": False},
            {"pert_id": "BRD-CCC", "broad_inference_eligible": True},
        ]
    }
    eligible = response_blind_eligible_pert_ids(bridge, registry)
    assert eligible == {"BRD-AAA"}


def test_evaluate_oracle_reports_per_line_and_null_lift(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "context_id": ["HT29"] * 5,
            "depmap_id": ["ACH-000552"] * 5,
            "ccle_name": ["HT29_LARGE_INTESTINE"] * 5,
            "pert_id": list("ABCDE"),
            "reversal_observed": [5.0, 4.0, 3.0, 2.0, 1.0],
            "sensitivity_score": [50.0, 40.0, 30.0, 20.0, 10.0],
        }
    )
    result = evaluate_oracle_frame(
        frame,
        score_column="reversal_observed",
        ks=(2,),
        minimum_candidates=3,
        null_seed=NULL_SEED,
        null_repeats=32,
        bootstrap_repeats=16,
    )
    assert result["eligible_line_count"] == 1
    line = result["line_rows"][0]
    assert line["candidate_count"] == 5
    assert line["top_k"]["2"]["overlap_lift"] > 1.0
    assert line["null_baseline"]["seed"] == NULL_SEED
    assert "support_stratum" in line
    summary_path = tmp_path / "out.json"
    summary_path.write_text(json.dumps(result["support_strata"]), encoding="utf-8")
    assert summary_path.exists()


def test_predicted_gap_and_stratified_summary_are_pair_aligned() -> None:
    joined = pd.DataFrame(
        {
            "context_id": ["HT29", "HT29", "HCT116"],
            "pert_id": ["A", "B", "A"],
            "reversal_observed": [1.0, 0.0, 2.0],
            "reversal_predicted": [1.0, 0.2, 1.5],
            "candidate_count": [80, 80, 400],
        }
    )
    gap = predicted_oracle_gap(joined)
    assert gap["pair_count"] == 3
    assert gap["pearson"] is not None
    assert gap["mae"] < 0.5
    strata = stratified_support_summary(
        [
            {"candidate_count": 80, "eligible": True, "top_k": {"10": {"overlap_lift": 2.0}}},
            {"candidate_count": 400, "eligible": True, "top_k": {"10": {"overlap_lift": 1.2}}},
        ]
    )
    assert "n_20_99" in strata
    assert "n_100_499" in strata
    assert strata["forbidden_interpretation"]


def test_gene_order_digest_matches_stream_contract() -> None:
    from drug_screen.evaluation.full_observed_oracle import gene_order_digest

    assert gene_order_digest(["10", "20"]) != gene_order_digest(["20", "10"])


def test_oracle_near_null_requires_majority_top10_lift() -> None:
    weak = [
        {
            "eligible": True,
            "top_k": {"10": {"overlap_lift": 0.0}},
            "null_baseline": {"spearman_delta": -0.08},
        }
        for _ in range(9)
    ] + [
        {
            "eligible": True,
            "top_k": {"10": {"overlap_lift": 1.4}},
            "null_baseline": {"spearman_delta": 0.12},
        }
    ]
    assert oracle_near_null(weak) is True
    strong = [
        {
            "eligible": True,
            "top_k": {"10": {"overlap_lift": 2.0}},
            "null_baseline": {"spearman_delta": 0.12},
        }
        for _ in range(6)
    ] + [
        {
            "eligible": True,
            "top_k": {"10": {"overlap_lift": 0.8}},
            "null_baseline": {"spearman_delta": 0.02},
        }
        for _ in range(2)
    ]
    assert oracle_near_null(strong) is False


def test_reversal_and_weighted_ks_are_defined_for_signed_signature() -> None:
    signature_values = np.array([1.0, 0.5, -0.5, -1.0])
    delta = np.array([-1.0, -0.5, 0.5, 1.0])
    score = score_reversal(signature_values, delta)
    assert score > 0
    cmap = weighted_ks_reversal(delta, np.arange(4), signature_values)
    assert np.isfinite(cmap)
