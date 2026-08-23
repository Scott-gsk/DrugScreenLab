import json
import math

import pandas as pd
import pytest

from drug_screen.evaluation.exp010 import (
    PredictionRecord, assert_patient_isolation, cluster_bootstrap_delta, fold_roles,
    patient_outer_folds, patient_ranking_metrics, promising_fast_decision,
)
from drug_screen.evaluation.exp010_breast import (
    TORG_ORDER, gse303509_rank_validation, gse287429_contrast_manifest,
    gse276609_domain_shift, multiregional_public_gate, reactome_overrepresentation,
)


def test_patient_folds_are_deterministic_balanced_and_isolated():
    assignments = patient_outer_folds([f"P{i:02}" for i in range(12)])
    assert assignments == patient_outer_folds(reversed([f"P{i:02}" for i in range(12)]))
    assert max(pd.Series(assignments).value_counts()) - min(pd.Series(assignments).value_counts()) <= 1
    roles = fold_roles(assignments, 2)
    frame = pd.DataFrame([{"patient_id": p, "pdo_id": f"{p}_A", "split": role} for p, role in roles.items()])
    assert_patient_isolation(frame)
    leaked = pd.concat([frame, pd.DataFrame([{"patient_id": "P00", "pdo_id": "P00_B", "split": "test"}])])
    with pytest.raises(ValueError, match="patient leakage"):
        assert_patient_isolation(leaked)


def _records(model, offset=0.0):
    rows = []
    for patient in ("P1", "P2"):
        for index, observed in enumerate((1.0, 2.0, 3.0, 4.0, 5.0)):
            rows.append(PredictionRecord("EXP-010", model, "test", 0, patient, patient + "O", f"D{index}", observed, observed + offset))
    return rows


def test_patient_macro_metrics_and_bootstrap_are_patient_equal():
    metrics = patient_ranking_metrics(_records("M3"))
    assert metrics["patient_count"] == 2
    assert metrics["macro"]["spearman"] == pytest.approx(1.0)
    assert metrics["macro"]["ndcg_at_5"] == pytest.approx(1.0)
    m3, m2 = pd.DataFrame(metrics["per_patient"]), pd.DataFrame(patient_ranking_metrics(_records("M2", 1.0))["per_patient"])
    interval = cluster_bootstrap_delta(m3, m2, repeats=200, seed=7)
    assert interval["patient_count"] == 2
    assert interval["delta_mean"] == pytest.approx(0.0)


def test_fast_decision_requires_every_gate():
    m2 = {"spearman": .4, "ndcg_at_5": .6, "mae": 1.0}
    m3 = {"spearman": .46, "ndcg_at_5": .6, "mae": 1.05}
    assert promising_fast_decision(m3, m2, .001)["status"] == "PROMISING_FAST"
    assert promising_fast_decision(m3, m2, 0.0)["status"] == "NO_GO"


def test_gse303509_paper_positive_control():
    # Figure 4D paper prediction order is the frozen listed TORG order.
    result = gse303509_rank_validation({sample: index for index, sample in enumerate(TORG_ORDER)})
    assert result["spearman"] == pytest.approx(0.6071428571428571)
    assert result["pairwise_concordant"] == 15
    assert result["status"] == "SUPPORTS"
    # Regression: real CSV inputs become NumPy scalars before evaluation; the
    # frozen result bundle must remain directly JSON serializable.
    real_shape_predictions = pd.Series(range(7), index=TORG_ORDER, dtype="int64").to_dict()
    serialized = json.dumps(gse303509_rank_validation(real_shape_predictions))
    assert '"pairwise_concordant": 15' in serialized


def test_gse287429_has_four_time_matched_contrasts():
    rows = []
    for time in (23, 47):
        for treatment in ("CONTROL", "DOX", "BTZ"):
            for replicate in (1, 2):
                rows.append({"sample_id": f"{treatment}_{time}_{replicate}", "treatment": treatment, "time_hours": time, "replicate": replicate})
    result = gse287429_contrast_manifest(pd.DataFrame(rows))
    assert len(result["contrasts"]) == 4
    assert "not drug sensitivity" in result["interpretation"]


def test_domain_shift_and_multiregional_not_evaluable_gate():
    reference = pd.DataFrame({"G1": [0., 1., 2.], "G2": [0., 1., 0.]})
    target = pd.DataFrame({"G1": [10., 11., 12.], "G2": [0., 1., 0.]})
    result = gse276609_domain_shift(reference, target)
    assert math.isfinite(result["centroid_distance"])
    gate = multiregional_public_gate([pd.DataFrame({"patient_id": ["P"], "pdo_id": ["O"], "drug_id": ["cisplatin"], "ic50": [1.]})], required_drugs=["cisplatin"])
    assert gate["status"] == "NOT_EVALUABLE_PUBLIC_MATRIX"
    assert not gate["checks"]["expression_features"]


def test_reactome_enrichment_uses_declared_universe_and_bh():
    result = reactome_overrepresentation(
        ["A", "B", "C", "D"], {"R1": ["A", "B"], "R2": ["D"], "R3": ["X"]},
        top_n=2, universe=["A", "B", "C", "D"],
    )
    assert result["tested_pathway_count"] == 3
    assert result["rows"][0]["pathway"] == "R1"
    assert 0 <= result["rows"][0]["fdr_bh"] <= 1
