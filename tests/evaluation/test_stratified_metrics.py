"""EXP-026 分层评价终点单元测试。"""

import json

import numpy as np
import pandas as pd
import pytest

from drug_screen.evaluation.stratified_metrics import (
    StructuralDrugs,
    evaluate_stratified,
    full_panel_patient_values,
    load_structural_drugs,
    nonstructural_patient_values,
    per_drug_patient_values,
)

STRUCTURAL = StructuralDrugs(
    pan_effective=frozenset({"D_TOP"}),
    pan_resistant=frozenset({"D_BOT"}),
    nonstructural=frozenset({"D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"}),
)

DRUGS = ["D_TOP", "D_BOT", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"]
PATIENTS = {"P1": 0, "P2": 0, "P3": 1, "P4": 1}

# 观测：结构性药极端固定；非结构药每个患者不同次序（患者间异质）。
# 每 PDO 10 个已知药 → 满足官方全面板资格（≥10）。
_OBS = {
    "P1": {"D_TOP": 30.0, "D_BOT": 0.1, "D3": 9.0, "D4": 7.0, "D5": 5.0, "D6": 3.0, "D7": 8.0, "D8": 6.0, "D9": 4.0, "D10": 2.0},
    "P2": {"D_TOP": 28.0, "D_BOT": 0.2, "D3": 3.0, "D4": 9.0, "D5": 5.0, "D6": 7.0, "D7": 4.0, "D8": 8.0, "D9": 6.0, "D10": 2.0},
    "P3": {"D_TOP": 25.0, "D_BOT": 0.1, "D3": 5.0, "D4": 3.0, "D5": 9.0, "D6": 7.0, "D7": 6.0, "D8": 2.0, "D9": 8.0, "D10": 4.0},
    "P4": {"D_TOP": 27.0, "D_BOT": 0.3, "D3": 7.0, "D4": 5.0, "D5": 3.0, "D6": 9.0, "D7": 2.0, "D8": 4.0, "D9": 6.0, "D10": 8.0},
}


def _frame(predicted_fn, algorithm="M") -> pd.DataFrame:
    rows = []
    for patient, fold in PATIENTS.items():
        pdo = f"{patient}_t1"
        for drug in DRUGS:
            rows.append({
                "patient_id": patient,
                "model_id": pdo,
                "drug_id": drug,
                "fold": fold,
                "algorithm": algorithm,
                "observed": float(_OBS[patient][drug]),
                "predicted": float(predicted_fn(patient, drug)),
                "source_drug_status": "known",
            })
    return pd.DataFrame(rows)


def _b0(patient, drug):
    fold = PATIENTS[patient]
    train = [p for p, f in PATIENTS.items() if f != fold]
    return float(np.mean([_OBS[p][drug] for p in train]))


def test_load_structural_drugs(tmp_path):
    obj = {
        "status": "FROZEN",
        "pan_effective": ["A"],
        "pan_resistant": ["B"],
        "nonstructural": ["C", "D"],
    }
    path = tmp_path / "structural.json"
    path.write_text(json.dumps(obj))
    loaded = load_structural_drugs(path)
    assert loaded.pan_effective == frozenset({"A"})
    assert loaded.structural == frozenset({"A", "B"})
    assert loaded.nonstructural == frozenset({"C", "D"})


def test_load_structural_drugs_rejects_unfrozen(tmp_path):
    path = tmp_path / "structural.json"
    path.write_text(json.dumps({
        "status": "DRAFT", "pan_effective": [], "pan_resistant": [],
        "nonstructural": []}))
    with pytest.raises(ValueError, match="not FROZEN"):
        load_structural_drugs(path)


def test_load_structural_drugs_rejects_overlap(tmp_path):
    path = tmp_path / "structural.json"
    path.write_text(json.dumps({
        "status": "FROZEN", "pan_effective": ["A"], "pan_resistant": ["A"],
        "nonstructural": []}))
    with pytest.raises(ValueError, match="overlapping"):
        load_structural_drugs(path)


def test_nonstructural_subset_excludes_structural():
    df = _frame(lambda p, d: _OBS[p][d])  # 完美预测
    ns = nonstructural_patient_values(df, STRUCTURAL)
    assert ns.notna().all()  # 每 PDO 有 5 个非结构药 ≥ k=5，非空断言
    assert (ns == 1.0).all()
    assert len(ns) == 4  # 4 个 PDO


def test_nonstructural_fewer_than_k_is_nan():
    df = _frame(lambda p, d: _OBS[p][d])
    df = df[df.drug_id.isin(["D3", "D4", "D5", "D_TOP"])]  # 非结构药仅 3 个 < k=5
    ns = nonstructural_patient_values(df, STRUCTURAL, k=5)
    assert ns.isna().all()


def test_full_panel_matches_ndcg_semantics():
    df = _frame(lambda p, d: _OBS[p][d])
    fp = full_panel_patient_values(df, k=5)
    assert (fp.dropna() == 1.0).all()  # 完美预测 → 全面板 NDCG@5 = 1


def test_per_drug_spearman_perfect_tracker():
    # 模型逐患者完美预测 → 每药跨患者 Spearman = 1
    df = _frame(lambda p, d: _OBS[p][d])
    values = per_drug_patient_values(df, min_patients=4)
    assert values.notna().all()
    assert (values == 1.0).all()
    assert len(values) == 10  # 10 个药


def test_per_drug_spearman_anti_tracking():
    # 逐患者反向预测 → 每药跨患者 Spearman = -1
    df = _frame(lambda p, d: -_OBS[p][d])
    values = per_drug_patient_values(df, min_patients=4)
    assert (values.dropna() < -0.999).all()


def test_per_drug_spearman_min_patients_gate():
    df = _frame(lambda p, d: _OBS[p][d])
    values = per_drug_patient_values(df, min_patients=5)  # 每药仅 4 患者
    assert values.isna().all()


def test_per_drug_spearman_b0_is_valid_null():
    # B0：预测为折常数（每折两个患者同值）；跨患者观测与折分配独立。
    # 两折构造下非结构药的 B0 Spearman 不应达到完美追踪（|rho| < 1），
    # 且对「按药宏平均」而言其构成零假设噪声——本测试验证不产生伪完美信号。
    df = _frame(_b0, algorithm="B0")
    values = per_drug_patient_values(df, min_patients=4)
    finite = values.dropna()
    assert len(finite) > 0
    assert (finite.abs() < 1.0).all()  # 关键：无伪 ±1（原残差设计会出现的伪影）


def test_evaluate_stratified_end_to_end():
    df = _frame(lambda p, d: _OBS[p][d], algorithm="PERFECT")
    summary, pdo_detail, drug_detail = evaluate_stratified(
        df, STRUCTURAL, algorithm="PERFECT", min_patients_per_drug=4)
    assert summary.nonstructural_ndcg5_macro == pytest.approx(1.0)
    assert summary.per_drug_patient_spearman_macro == pytest.approx(1.0)
    assert summary.full_panel_ndcg5_macro == pytest.approx(1.0)
    assert set(summary.nonstructural_drugs_covered) == {
        "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"}
    assert summary.n_patients_nonstructural == 4
    assert {"nonstructural_ndcg", "full_panel_ndcg"} <= set(pdo_detail.columns)
    assert {"drug_id", "per_drug_patient_spearman"} == set(drug_detail.columns)


def test_evaluate_stratified_requires_algorithm_selection():
    df = _frame(lambda p, d: _OBS[p][d], algorithm="M")
    df2 = _frame(lambda p, d: _OBS[p][d], algorithm="OTHER")
    with pytest.raises(ValueError, match="multiple algorithms"):
        evaluate_stratified(pd.concat([df, df2], ignore_index=True), STRUCTURAL)
    # 指定 algorithm 后正常
    summary, _, _ = evaluate_stratified(
        pd.concat([df, df2], ignore_index=True), STRUCTURAL, algorithm="M")
    assert summary.algorithm == "M"


def test_evaluate_stratified_unknown_algorithm():
    df = _frame(lambda p, d: _OBS[p][d])
    with pytest.raises(ValueError, match="no known rows"):
        evaluate_stratified(df, STRUCTURAL, algorithm="NOPE")


def test_b0_full_panel_reference_still_defined():
    # B0 在非残差口径（全面板参照）下可排序；逐药 Spearman 为零假设噪声
    df = _frame(_b0, algorithm="B0")
    summary, _, _ = evaluate_stratified(
        df, STRUCTURAL, algorithm="B0", min_patients_per_drug=4)
    assert summary.full_panel_ndcg5_macro is not None
    # 4 患者两折的 B0 逐药 Spearman 均应为有限噪声值
    assert summary.per_drug_patient_spearman_macro is not None
