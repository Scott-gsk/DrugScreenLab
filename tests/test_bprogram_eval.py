"""工作包 B 评估侧测试（B5 + schema + baselines + 统计）。

覆盖执行合同 B5 评估侧项目：#6 常数相关性、#8 面板/标签不一致拒绝比较、
#9 显著性变量 + 可复现、#10 人工构造合理指标变化，以及三类均值基线、
统一 schema 校验器、资格规则、配对 bootstrap 与 Holm 校正。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drug_screen.evaluation.exp031_eval import (
    assert_comparable,
    constant_prediction_guard,
    gene_dependency_mean_baseline,
    holm_correct,
    intra_patient_spearman_patient_macro,
    paired_bootstrap_ci,
    source_drug_mean_baseline,
    target_drug_mean_baseline,
    validate_prediction_frame,
)
from drug_screen.evaluation.metrics import ndcg_at_k


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _single_patient_frame(observed, predicted, patient="p0"):
    n = len(observed)
    return pd.DataFrame({
        "patient_id": [patient] * n,
        "drug_id": [f"d{i}" for i in range(n)],
        "observed": np.asarray(observed, dtype=float),
        "predicted": np.asarray(predicted, dtype=float),
    })


def _pair_frame(n_patients=2, n_drugs=3, seed=0):
    rows = []
    for i in range(n_patients):
        for j in range(n_drugs):
            rows.append({
                "experiment_id": "e1",
                "contract_id": "c1",
                "split_id": "s1",
                "fold": 0,
                "seed": seed,
                "patient_id": f"p{i}",
                "sample_id": f"p{i}_d{j}",
                "drug_id": f"d{j}",
                "y_true": float(i * 10 + j),
                "y_pred": float(i * 10 + j),
            })
    return pd.DataFrame(rows)


def _valid_schema_frame(n=3):
    rows = []
    for i in range(n):
        rows.append({
            "experiment_id": "e1",
            "run_id": f"r{i}",
            "contract_id": "c1",
            "split_id": "s1",
            "fold": 0,
            "seed": 0,
            "patient_id": f"p{i}",
            "model_id": f"m{i}",
            "sample_id": f"s{i}",
            "drug_id": f"d{i}",
            "gene_id": float("nan"),
            "y_true": float(i),
            "y_pred": float(i),
            "evaluation_role": "test",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B5 #6 —— 常数相关性守卫
# ---------------------------------------------------------------------------


def test_constant_prediction_guard_within_and_across_fold():
    rows = [
        {"fold": 0, "drug_id": "A", "predicted": 0.5, "sample": "s1"},
        {"fold": 0, "drug_id": "A", "predicted": 0.5, "sample": "s2"},
        {"fold": 1, "drug_id": "A", "predicted": 0.7, "sample": "s3"},
        {"fold": 1, "drug_id": "A", "predicted": 0.7, "sample": "s4"},
        {"fold": 0, "drug_id": "B", "predicted": 0.3, "sample": "s5"},
        {"fold": 0, "drug_id": "B", "predicted": 0.3, "sample": "s6"},
        {"fold": 1, "drug_id": "B", "predicted": 0.3, "sample": "s7"},
        {"fold": 1, "drug_id": "B", "predicted": 0.3, "sample": "s8"},
        {"fold": 0, "drug_id": "C", "predicted": 1.0, "sample": "s9"},
        {"fold": 0, "drug_id": "C", "predicted": 2.0, "sample": "s10"},
    ]
    out = constant_prediction_guard(pd.DataFrame(rows))

    def reason(fold, drug):
        sub = out[(out["fold"] == fold) & (out["drug_id"] == drug)]
        return set(sub["undefined_constant_reason"].dropna())

    # 折内常数（各折取值不同）→ within
    assert reason(0, "A") == {"constant_within_fold"}
    assert reason(1, "A") == {"constant_within_fold"}
    # 跨折全局常数（各折取值相同）→ across
    assert reason(0, "B") == {"constant_across_folds"}
    assert reason(1, "B") == {"constant_across_folds"}
    # 有变化 → 不标记
    assert reason(0, "C") == set()


def test_constant_prediction_guard_insufficient_shape():
    # 缺失必需列时抛错，而非静默产出。
    with pytest.raises(ValueError):
        constant_prediction_guard(pd.DataFrame({"fold": [0]}))


# ---------------------------------------------------------------------------
# B5 #8 —— assert_comparable 拒绝不同面板 / 标签
# ---------------------------------------------------------------------------


def test_assert_comparable_drug_panel_mismatch():
    a = _pair_frame()
    b = _pair_frame()
    extra = pd.DataFrame([{
        "experiment_id": "e1", "contract_id": "c1", "split_id": "s1",
        "fold": 0, "seed": 0, "patient_id": "p0", "sample_id": "p0_dX",
        "drug_id": "dX", "y_true": 999.0, "y_pred": 999.0,
    }])
    with pytest.raises(ValueError, match="drug_id panel differs"):
        assert_comparable(pd.concat([a, extra], ignore_index=True), b)


def test_assert_comparable_label_mismatch():
    a = _pair_frame()
    b = _pair_frame()
    b.loc[0, "y_true"] = 12345.0
    with pytest.raises(ValueError, match="y_true differs"):
        assert_comparable(a, b)


def test_assert_comparable_seed_mismatch():
    a = _pair_frame(seed=0)
    b1 = _pair_frame(seed=1)
    b = pd.concat([_pair_frame(seed=0), b1], ignore_index=True)
    # 单种子 vs 多种子默认拒绝。
    with pytest.raises(ValueError, match="seed sets differ"):
        assert_comparable(a, b)
    # 显式允许后不再因种子 / 组合报错（其它检查仍通过）。
    assert_comparable(a, b, allow_seed_mismatch=True)


def test_assert_comparable_ok():
    a = _pair_frame()
    b = _pair_frame()
    # 仅预测列不同不构成比较障碍（比较检查只看标签 / 面板 / 键 / 权重）。
    b["y_pred"] = b["y_pred"] + 1.0
    assert_comparable(a, b)


# ---------------------------------------------------------------------------
# B5 #9 —— 显著性条件使用正确变量（患者为单元 + 种子固定可复现）
# ---------------------------------------------------------------------------


def test_paired_bootstrap_ci_patient_unit_reproducible():
    n_patients, n_drugs = 8, 5
    rows_a, rows_b = [], []
    for i in range(n_patients):
        truth = np.arange(1.0, n_drugs + 1.0)
        for j in range(n_drugs):
            rows_a.append({"patient_id": f"p{i}", "y_true": truth[j],
                           "y_pred": truth[j]})
            rows_b.append({"patient_id": f"p{i}", "y_true": truth[j],
                           "y_pred": truth[j] + 1.0})
    a = pd.DataFrame(rows_a)
    b = pd.DataFrame(rows_b)

    metric = lambda t, p: float(-np.mean((t - p) ** 2))
    r1 = paired_bootstrap_ci(a, b, "patient_id", metric, n=2000, seed=0)
    r2 = paired_bootstrap_ci(a, b, "patient_id", metric, n=2000, seed=0)

    # 以患者为单元：单元数 = 患者数（非行数）。
    assert r1.n_units == n_patients
    # 每患者 delta = 0 - (-1) = 1 → 点估计 = 1（患者级均值，非行级）。
    assert r1.point_estimate == pytest.approx(1.0)
    # 固定随机种子 → 完全可复现。
    assert (r1.low, r1.high) == (r2.low, r2.high)
    assert r1.point_estimate == pytest.approx(r2.point_estimate)


def test_holm_correct():
    # m=4: 排序 0.005,0.01,0.03,0.04 → 校正 0.02,0.03,0.06,0.06（累积取大）。
    got = holm_correct([0.01, 0.04, 0.03, 0.005])
    assert got == pytest.approx([0.03, 0.06, 0.06, 0.02])
    # 校正值封顶 1。
    assert holm_correct([0.5, 0.5, 0.5]) == [1.0, 1.0, 1.0]
    # 空输入 → 空列表。
    assert holm_correct([]) == []


# ---------------------------------------------------------------------------
# B5 #10 —— 正确/随机/加噪的合理指标变化（主指标 + NDCG@5）
# ---------------------------------------------------------------------------


def test_noise_monotonic_main_metric():
    n = 20
    observed = np.arange(1.0, n + 1.0)
    preds = []
    for k in range(n):
        # 第 k 级把最小值 1 后移到位置 k：其余元素保持顺序。
        preds.append(np.concatenate(
            [observed[1:k + 1], [observed[0]], observed[k + 1:]]))
    macros = []
    for pred in preds:
        macros.append(
            intra_patient_spearman_patient_macro(
                _single_patient_frame(observed, pred))[0]
        )
    # 正确排序 → 高指标。
    assert macros[0] == pytest.approx(1.0)
    # 逐步加噪 → 单调下降。
    for i in range(1, len(macros)):
        assert macros[i - 1] >= macros[i] - 1e-12
    assert macros[-1] < macros[0]
    # 随机排序 → 近 0。
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(observed)
    rand_macro = intra_patient_spearman_patient_macro(
        _single_patient_frame(observed, shuffled))[0]
    assert abs(rand_macro) < 0.5


def _score_for_order(order, n):
    rank = {idx: k for k, idx in enumerate(order)}
    return np.array([n - 1 - rank[i] for i in range(n)], dtype=float)


def test_noise_monotonic_ndcg5():
    y_true = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    # 将最高相关（位置 0，值 5）逐步后移：每次相邻换位使 DCG@5 严格下降。
    orders = [
        [0, 1, 2, 3, 4],
        [1, 0, 2, 3, 4],
        [1, 2, 0, 3, 4],
        [1, 2, 3, 0, 4],
        [1, 2, 3, 4, 0],
    ]
    ndcgs = [ndcg_at_k(y_true, _score_for_order(o, 5), k=5) for o in orders]
    assert ndcgs[0] == pytest.approx(1.0)
    for i in range(1, len(ndcgs)):
        assert ndcgs[i - 1] > ndcgs[i]
    assert ndcgs[-1] < ndcgs[0]
    # 随机排序 → 明显低于正确排序。
    shuffled = np.array([1.0, 3.0, 5.0, 2.0, 4.0])
    rand_ndcg = ndcg_at_k(y_true, 4.0 - shuffled, k=5)
    assert rand_ndcg < ndcgs[0]


# ---------------------------------------------------------------------------
# 三类均值基线（A4）
# ---------------------------------------------------------------------------


def test_target_drug_mean_baseline_train_only():
    rows = []
    # fold 0 训练折: 2 患者 × 2 药
    for patient in ("p0", "p1"):
        for j, obs in enumerate((1.0, 3.0)):
            rows.append({"patient_id": patient, "drug_id": f"d{j}",
                         "fold": 0, "observed": obs + (1.0 if patient == "p1" else 0.0),
                         "evaluation_role": "train"})
    # fold 0 测试折: 1 患者 × 2 药（含毒标签）
    for j, obs in enumerate((100.0, 300.0)):
        rows.append({"patient_id": "p_test", "drug_id": f"d{j}",
                     "fold": 0, "observed": obs, "evaluation_role": "test"})
    frame = pd.DataFrame(rows)

    # 期望逐药常数：d0 = (1+2)/2 = 1.5, d1 = (3+4)/2 = 3.5。
    before = target_drug_mean_baseline(frame)
    test_before = before.loc[before["evaluation_role"] == "test", "predicted"].to_numpy()
    assert np.allclose(sorted(test_before), [1.5, 3.5])

    # 篡改测试折标签后，基线预测不变（只用训练折）。
    tampered = frame.copy()
    tampered.loc[tampered["evaluation_role"] == "test", "observed"] = 9999.0
    after = target_drug_mean_baseline(tampered)
    test_after = after.loc[after["evaluation_role"] == "test", "predicted"].to_numpy()
    assert np.allclose(sorted(test_before), sorted(test_after))
    assert np.allclose(sorted(test_after), [1.5, 3.5])


def test_source_drug_mean_baseline_gdsc_filter():
    rows = [
        {"cell_line_id": "c1", "drug_id": "d0", "observed": 1.0, "study_id": "GDSC"},
        {"cell_line_id": "c2", "drug_id": "d0", "observed": 3.0, "study_id": "GDSC"},
        {"cell_line_id": "c1", "drug_id": "d1", "observed": 5.0, "study_id": "GDSC"},
        {"cell_line_id": "c2", "drug_id": "d1", "observed": 7.0, "study_id": "GDSC"},
        {"cell_line_id": "p1", "drug_id": "d0", "observed": 100.0, "study_id": "PRISM"},
        {"cell_line_id": "p2", "drug_id": "d0", "observed": 100.0, "study_id": "PRISM"},
        {"cell_line_id": "p1", "drug_id": "d1", "observed": 999.0, "study_id": "PRISM"},
    ]
    src = pd.DataFrame(rows)
    out = source_drug_mean_baseline(src, source_study="GDSC")
    by_drug = dict(zip(out["drug_id"], out["predicted"]))
    assert by_drug["d0"] == pytest.approx(2.0)
    assert by_drug["d1"] == pytest.approx(6.0)
    # 均值来源注明 GDSC。
    assert set(out["source_study"]) == {"GDSC"}
    # 多来源未指定 study 时要求显式过滤。
    with pytest.raises(ValueError):
        source_drug_mean_baseline(src)


def test_gene_dependency_mean_baseline_per_gene_constant():
    rows = [
        {"sample_id": "s0", "gene_id": "g0", "y": 1.0, "fold": 0, "evaluation_role": "train"},
        {"sample_id": "s0", "gene_id": "g1", "y": 10.0, "fold": 0, "evaluation_role": "train"},
        {"sample_id": "s1", "gene_id": "g0", "y": 3.0, "fold": 0, "evaluation_role": "train"},
        {"sample_id": "s1", "gene_id": "g1", "y": 30.0, "fold": 0, "evaluation_role": "train"},
        {"sample_id": "s2", "gene_id": "g0", "y": 999.0, "fold": 0, "evaluation_role": "test"},
        {"sample_id": "s2", "gene_id": "g1", "y": 888.0, "fold": 0, "evaluation_role": "test"},
    ]
    dep = pd.DataFrame(rows)
    out = gene_dependency_mean_baseline(dep)
    test_rows = out[out["evaluation_role"] == "test"]
    pred = dict(zip(test_rows["gene_id"], test_rows["predicted"]))
    assert pred["g0"] == pytest.approx(2.0)
    assert pred["g1"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# 统一预测 schema 校验器
# ---------------------------------------------------------------------------


def test_validate_prediction_frame_missing_column():
    df = _valid_schema_frame().drop(columns=["run_id"])
    errors, missing = validate_prediction_frame(df)
    assert any("missing required column: run_id" in e for e in errors)
    assert missing.empty


def test_validate_prediction_frame_duplicate_rows():
    df = pd.concat([_valid_schema_frame(), _valid_schema_frame().iloc[[0]]],
                   ignore_index=True)
    errors, missing = validate_prediction_frame(df)
    assert any("duplicate rows" in e for e in errors)
    assert missing.empty


def test_validate_prediction_frame_nan_prediction():
    df = _valid_schema_frame()
    df.loc[1, "y_pred"] = np.nan
    errors, missing = validate_prediction_frame(df)
    assert any("NaN y_pred" in e for e in errors)
    assert len(missing) == 1
    assert missing.iloc[0]["sample_id"] == "s1"


def test_validate_prediction_frame_ok():
    df = _valid_schema_frame()
    errors, missing = validate_prediction_frame(df)
    assert errors == []
    assert missing.empty


# ---------------------------------------------------------------------------
# 主指标资格规则
# ---------------------------------------------------------------------------


def test_main_metric_insufficient_drugs_nan_and_reason():
    frame = pd.concat([
        _single_patient_frame([1.0, 2.0], [1.0, 2.0], patient="p_short"),
        _single_patient_frame([5.0, 4.0, 3.0, 2.0, 1.0], [5.0, 4.0, 3.0, 2.0, 1.0],
                              patient="p_full"),
    ], ignore_index=True)
    macro, n_valid, per_patient = intra_patient_spearman_patient_macro(frame)
    assert n_valid == 1
    assert macro == pytest.approx(1.0)
    short = per_patient.loc[per_patient["patient_id"] == "p_short"].iloc[0]
    assert np.isnan(short["intra_patient_spearman"])
    assert short["reason"] == "insufficient_drugs"
    full = per_patient.loc[per_patient["patient_id"] == "p_full"].iloc[0]
    assert full["reason"] is None


def test_main_metric_multi_pdo_aggregate():
    # 同一患者两枚 PDO，同一药物两次观测；先患者内均值再算排序。
    rows = []
    for pdo, shift in (("m1", 0.0), ("m2", 0.2)):
        for j, obs in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
            rows.append({"patient_id": "p0", "drug_id": f"d{j}",
                         "observed": obs + shift, "predicted": float(j + 1)})
    frame = pd.DataFrame(rows)
    macro, n_valid, per_patient = intra_patient_spearman_patient_macro(frame)
    assert n_valid == 1
    assert macro == pytest.approx(1.0)
    assert per_patient.iloc[0]["n_drugs"] == 5


def test_main_metric_constant_variance_nan_and_reason():
    # 观测零方差 → constant_observed；预测零方差 → constant_predicted。
    frame = pd.concat([
        _single_patient_frame([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0],
                              patient="p_const_obs"),
        _single_patient_frame([1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0],
                              patient="p_const_pred"),
    ], ignore_index=True)
    macro, n_valid, per_patient = intra_patient_spearman_patient_macro(frame)
    assert n_valid == 0
    assert macro is None
    reasons = dict(zip(per_patient["patient_id"], per_patient["reason"]))
    assert reasons["p_const_obs"] == "constant_observed"
    assert reasons["p_const_pred"] == "constant_predicted"