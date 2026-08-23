from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from drug_screen.evaluation.exp011 import (
    EvaluationContractError,
    Exp011Conclusion,
    evaluate_cell_line_gate,
    evaluate_exp011_predictions,
    pdo_primary_metrics,
    prediction_schema,
    prism_frozen_metrics,
    validate_prediction_frame,
)


def _row(
    *,
    model: str,
    role: str,
    cohort: str,
    fold: int,
    split: str,
    patient: str = "",
    sample: str,
    pdo: str = "",
    cell_line: str = "",
    drug: str,
    inchikey: str | None = None,
    scaffold: str | None = None,
    y_true: float,
    y_pred: float,
) -> dict[str, object]:
    return {
        "experiment_id": "EXP-011",
        "model_id": model,
        "evaluation_role": role,
        "cohort_id": cohort,
        "fold_id": fold,
        "split": split,
        "patient_id": patient,
        "sample_id": sample,
        "pdo_id": pdo,
        "cell_line_id": cell_line,
        "drug_id": drug,
        "inchikey": inchikey or f"INCHI-{drug}",
        "drug_scaffold": scaffold or f"SCAFFOLD-{drug}",
        "response_name": "sensitivity",
        "response_scale": "normalised_higher_sensitive",
        "higher_is_sensitive": True,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _gate_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    # Two GDSC drugs across ten held-out cell models. M2 is exact and Ridge is
    # intentionally anti-correlated, giving a deterministic paired gate pass.
    for drug_index in range(2):
        for cell_index in range(10):
            values = dict(
                role="gdsc_cold_cell", cohort="gdsc2", fold=cell_index % 5, split="test",
                sample=f"G{cell_index}", cell_line=f"G{cell_index}", drug=f"D{drug_index}",
                y_true=float(cell_index),
            )
            rows.append(_row(model="M2", y_pred=float(cell_index), **values))
            rows.append(_row(model="B1_ridge", y_pred=float(-cell_index), **values))
    # Frozen PRISM contract: 35 CRC lines, 81 exact InChIKeys, 2,784 rows.
    missing_pairs = {(0, drug) for drug in range(51)}
    for line_index in range(35):
        for drug_index in range(81):
            if (line_index, drug_index) in missing_pairs:
                continue
            response = float(drug_index + line_index / 100.0)
            rows.append(
                _row(
                    model="M2", role="prism_frozen", cohort="prism_crc", fold=-1, split="external",
                    sample=f"PRISM{line_index}", cell_line=f"PRISM{line_index}", drug=f"PD{drug_index}",
                    inchikey=f"EXACTKEY{drug_index:03d}", y_true=response, y_pred=response,
                )
            )
    assert sum(row["evaluation_role"] == "prism_frozen" for row in rows) == 2784
    return rows


def _pdo_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cohort in ("mcrc_pdo", "zhu_hepatobiliary"):
        for patient_index in range(6):
            for drug_index in range(3):
                # Each drug has patient variation; a drug-effect-only model
                # would not correlate within drug. Candidate has the signal.
                true = 10.0 * drug_index + (patient_index if drug_index % 2 == 0 else -patient_index)
                common = dict(
                    role="pdo_primary", cohort=cohort, fold=patient_index % 5, split="test",
                    patient=f"{cohort}_P{patient_index}", sample=f"{cohort}_S{patient_index}",
                    pdo=f"{cohort}_O{patient_index}", drug=f"PDO_D{drug_index}", y_true=true,
                )
                rows.append(_row(model="M3_adapter", y_pred=true, **common))
                rows.append(_row(model="M2", y_pred=10.0 * drug_index - (patient_index if drug_index % 2 == 0 else -patient_index), **common))
                rows.append(_row(model="B1_ridge", y_pred=true, **common))
    return rows


def test_schema_is_strict_about_identity_role_and_duplicates():
    assert prediction_schema()["format"] == "exp011_heldout_predictions_v1"
    frame = pd.DataFrame(
        [_row(model="M2", role="gdsc_cold_cell", cohort="gdsc2", fold=0, split="test", sample="S", cell_line="C", drug="D", y_true=1.0, y_pred=1.0)]
    )
    assert list(validate_prediction_frame(frame).columns) == list(prediction_schema()["columns"])
    invalid = frame.copy()
    invalid.loc[0, "fold_id"] = -1
    with pytest.raises(EvaluationContractError, match="external rows"):
        validate_prediction_frame(invalid)
    duplicate = pd.concat([frame, frame], ignore_index=True)
    with pytest.raises(EvaluationContractError, match="duplicate"):
        validate_prediction_frame(duplicate)


def test_prism_requires_the_frozen_exact_support():
    frame = validate_prediction_frame(pd.DataFrame(_gate_rows()))
    prism = frame[(frame.evaluation_role == "prism_frozen") & (frame.model_id == "M2")]
    result = prism_frozen_metrics(prism)
    assert result["support"]["observed"] == {"row_count": 2784, "drug_count": 81, "cell_line_count": 35}
    assert result["macro_spearman"] == pytest.approx(1.0)
    with pytest.raises(EvaluationContractError, match="support mismatch"):
        prism_frozen_metrics(prism.iloc[:-1])


def test_prism_gate_uses_per_drug_cross_cell_metric_not_linewise_drug_ranking():
    rows: list[dict[str, object]] = []
    for line_index in range(2):
        for drug_index in range(3):
            # This predictor exactly preserves the across-drug order in every
            # line, but is constant within each drug and has no cell-line
            # response signal.  A linewise gate would falsely call it perfect.
            rows.append(
                _row(
                    model="M2", role="prism_frozen", cohort="prism_crc", fold=-1, split="external",
                    sample=f"S{line_index}", cell_line=f"L{line_index}", drug=f"D{drug_index}",
                    inchikey=f"K{drug_index}", y_true=float(10 * drug_index + line_index),
                    y_pred=float(10 * drug_index),
                )
            )
    result = prism_frozen_metrics(
        validate_prediction_frame(pd.DataFrame(rows)), expected_drug_count=3,
        expected_cell_line_count=2, expected_row_count=6, minimum_cells_per_drug=2,
    )
    assert result["macro_spearman"] is None
    assert result["cell_line_cross_drug_ranking_diagnostic"]["macro_spearman"] == pytest.approx(1.0)


def test_cell_line_gate_uses_per_drug_cell_metrics_and_frozen_prism():
    frame = validate_prediction_frame(pd.DataFrame(_gate_rows()))
    gate = evaluate_cell_line_gate(frame, repeats=100, seed=7)
    assert gate["passed"] is True
    assert gate["cold_cell"]["source"]["macro_spearman"] == pytest.approx(1.0)
    assert gate["cold_cell"]["source_minus_ridge"]["ci_95"][0] >= 1.9
    assert gate["prism_frozen"]["labels_used_for_model_selection"] is False


def test_pdo_endpoint_is_equal_cohort_per_drug_and_cluster_bootstrapped():
    frame = validate_prediction_frame(pd.DataFrame(_pdo_rows()))
    result = evaluate_exp011_predictions(
        pd.concat([pd.DataFrame(_gate_rows()), frame], ignore_index=True), bootstrap_repeats=100, seed=11
    )
    assert result["conclusion"]["status"] == Exp011Conclusion.PROMISING_MULTI_PDO.value
    primary = result["pdo_primary"]["M3_adapter"]
    assert primary["macro_spearman_equal_cohort"] == pytest.approx(1.0)
    assert {row["cohort_id"] for row in primary["per_cohort"]} == {"mcrc_pdo", "zhu_hepatobiliary"}
    assert result["pdo_primary"]["M3_adapter_minus_M2"]["repeats"] == 100
    # The resulting compact report is directly JSON serialisable.
    assert "PROMISING_MULTI_PDO" in json.dumps(result)


def test_drug_main_effect_only_ranking_is_not_a_valid_primary_signal():
    rows: list[dict[str, object]] = []
    for cohort in ("mcrc_pdo", "zhu_hepatobiliary"):
        for patient_index in range(6):
            for drug_index in range(2):
                # The predictor knows only the global high/low drug ordering.
                true = 10.0 * drug_index + (patient_index if drug_index == 0 else -patient_index)
                rows.append(
                    _row(
                        model="drug_median", role="pdo_primary", cohort=cohort,
                        fold=patient_index % 5, split="test", patient=f"{cohort}_{patient_index}",
                        sample=f"{cohort}_{patient_index}", pdo=f"{cohort}_{patient_index}", drug=f"D{drug_index}",
                        y_true=true, y_pred=10.0 * drug_index,
                    )
                )
    metrics = pdo_primary_metrics(validate_prediction_frame(pd.DataFrame(rows)))
    # Old patient-wise cross-drug ranking says the model is perfect because it
    # merely gets the shared drug main effect right.  The new endpoint has no
    # evaluable per-drug patient correlation because its prediction is constant
    # within every drug.
    assert metrics["raw_patient_ranking_diagnostic"]["macro_spearman"] == pytest.approx(1.0)
    assert metrics["macro_spearman_equal_cohort"] is None
    assert metrics["residualized_patient_ranking_diagnostic"]["macro_spearman"] is None


def test_failed_cell_gate_suppresses_pdo_interpretation_and_exploratory_rows_are_out_of_verdict():
    rows = _gate_rows()
    for row in rows:
        if row["evaluation_role"] == "gdsc_cold_cell" and row["model_id"] == "M2":
            row["y_pred"] = 0.0
    rows.extend(_pdo_rows())
    rows.append(
        _row(
            model="M3_adapter", role="external_exploratory", cohort="breast_gse303509", fold=-1,
            split="external", patient="", sample="TORG1", pdo="", cell_line="", drug="cisplatin",
            y_true=0.2, y_pred=0.8,
        )
    )
    result = evaluate_exp011_predictions(pd.DataFrame(rows), bootstrap_repeats=100, seed=9)
    assert result["conclusion"]["status"] == Exp011Conclusion.CELL_LINE_GATE_FAILED.value
    assert result["pdo_primary"]["status"] == "NOT_INTERPRETED_CELL_LINE_GATE_FAILED"
    assert result["external_exploratory"]["not_used_for_primary_verdict"] is True
