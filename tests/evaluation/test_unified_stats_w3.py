"""W3-07~W3-11 评估统计口径统一修复的单元测试（2026-09-22 审计）。

覆盖 docs/reports/audit_repair_wave3_20260922.md 的五个统计口径条目：

- W3-07 种子统计口径：先逐种子计算患者分数 → 患者的种子分数平均 →
  患者级配对统计；预测平均集成另列为独立模型（数值与手算一致）。
- W3-08 折内同药跨患者相关性：按折评价；常数记 NaN+原因；禁止跨折
  池化负基线作增益依据（基线缺失折不跨折借用）。
- W3-09 相关性并列秩 + 缺失掩码（pairwise complete，显式 mask）；
  置换 p 值有限次数修正（禁止 p=0.0）。
- W3-10 多重比较 family 预注册 + Holm（跨种子平均 p 拒绝）。
- W3-11 患者信息置换：身份映射后广播到药物记录（药物对应保持、
  划内置换）。
- 统一 canonical 入口 run_formal_comparison。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, spearmanr

from drug_screen.evaluation.exp031_eval import (
    ComparisonFamily,
    ComparisonPValue,
    ComparisonSpec,
    aggregate_seed_patient_scores,
    holm_correct_family,
    patient_score_paired_bootstrap,
    per_fold_drug_patient_correlation,
    per_fold_gain,
    run_formal_comparison,
)
from drug_screen.evaluation.metrics import masked_pearson, masked_spearman
from drug_screen.evaluation.permutation import (
    permutation_p_value,
    permute_patient_identity_broadcast,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _seed_frame(per_patient):
    """构造预测长表：per_patient = [(patient, observed, predicted), ...]。"""
    rows = []
    for patient, observed, predicted in per_patient:
        for j in range(len(observed)):
            rows.append({
                "patient_id": patient,
                "drug_id": f"d{j}",
                "observed": float(observed[j]),
                "predicted": float(predicted[j]),
            })
    return pd.DataFrame(rows)


def _fold_frame(fold, drug, observed, predicted, patients=None):
    """构造折内单药多患者长表。"""
    n = len(observed)
    patients = patients or [f"P{i}" for i in range(n)]
    return pd.DataFrame({
        "fold": [fold] * n,
        "patient_id": patients,
        "drug_id": [drug] * n,
        "observed": list(map(float, observed)),
        "predicted": list(map(float, predicted)),
    })


# ---------------------------------------------------------------------------
# W3-07 种子统计口径
# ---------------------------------------------------------------------------


class TestSeedAggregation:
    """2 种子 × 3 患者的手算验证。"""

    @pytest.fixture()
    def aggregation(self):
        obs = [1.0, 2.0, 3.0, 4.0, 5.0]
        # P1: seed0 rho=1.0；seed1 预测 [2,1,4,3,5] → rho=0.8（Σd²=4）
        #     ensemble 预测 [1.5,1.5,3.5,3.5,5] → 秩 Pearson = 9/sqrt(90)
        # P2: seed0 rho=0.9；seed1 rho=-1.0（观测反序）
        # P3: seed0 rho=1.0；seed1 常数预测 → NaN（资格性未定义）
        frame0 = _seed_frame([
            ("P1", obs, [1.0, 2.0, 3.0, 4.0, 5.0]),
            ("P2", obs, [1.0, 2.0, 3.0, 5.0, 4.0]),
            ("P3", obs, [1.0, 2.0, 3.0, 4.0, 5.0]),
        ])
        frame1 = _seed_frame([
            ("P1", obs, [2.0, 1.0, 4.0, 3.0, 5.0]),
            ("P2", obs, [5.0, 4.0, 3.0, 2.0, 1.0]),
            ("P3", obs, [3.0, 3.0, 3.0, 3.0, 3.0]),
        ])
        return aggregate_seed_patient_scores({0: frame0, 1: frame1})

    def test_per_seed_patient_scores_manual(self, aggregation):
        per_seed = aggregation.per_seed_patient_scores
        assert list(per_seed.columns) == ["0", "1"]
        # seed0：P1=1.0（完美）、P2=0.9（Σd²=2）、P3=1.0
        assert per_seed.loc["P1", "0"] == pytest.approx(1.0)
        assert per_seed.loc["P2", "0"] == pytest.approx(0.9)
        assert per_seed.loc["P3", "0"] == pytest.approx(1.0)
        # seed1：P1=0.8（Σd²=4）、P2=-1.0（反序）、P3 常数 → NaN
        assert per_seed.loc["P1", "1"] == pytest.approx(0.8)
        assert per_seed.loc["P2", "1"] == pytest.approx(-1.0)
        assert np.isnan(per_seed.loc["P3", "1"])

    def test_seed_mean_manual_and_nan_handling(self, aggregation):
        seed_mean = aggregation.seed_mean_patient_scores
        assert seed_mean.loc["P1"] == pytest.approx((1.0 + 0.8) / 2)
        assert seed_mean.loc["P2"] == pytest.approx((0.9 - 1.0) / 2)
        # P3：seed1 常数未定义 → 跳过该种子，只用 seed0；n_seeds_used=1
        assert seed_mean.loc["P3"] == pytest.approx(1.0)
        assert int(aggregation.n_seeds_used.loc["P3"]) == 1
        assert int(aggregation.n_seeds_used.loc["P1"]) == 2

    def test_ensemble_is_independent_model_differs_from_seed_mean(
            self, aggregation):
        # 预测平均集成：逐 (patient, drug) 平均预测后重算——独立模型口径。
        ens = aggregation.ensemble_patient_scores
        # P1：[1.5,1.5,3.5,3.5,5] 与 obs 的秩 Pearson = 9/sqrt(90)
        assert ens.loc["P1"] == pytest.approx(9.0 / np.sqrt(90.0))
        # 集成 ≠ 逐种子平均（0.9）——两种口径不得混淆
        assert ens.loc["P1"] != pytest.approx(
            aggregation.seed_mean_patient_scores.loc["P1"])
        # P3：均值 [2,2.5,3,3.5,4] 严格递增 → rho=1（与 seed_mean 恰好相同，
        # 但来自集成重算而非种子平均）
        assert ens.loc["P3"] == pytest.approx(1.0)
        # 集成预测表可直接喂主指标（结构化落点）
        assert {"patient_id", "drug_id", "observed", "predicted"} == set(
            aggregation.ensemble_predictions.columns)

    def test_ensemble_matches_scipy_on_averaged_predictions(self, aggregation):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean_pred = (np.array([1.0, 2.0, 3.0, 5.0, 4.0])
                     + np.array([5.0, 4.0, 3.0, 2.0, 1.0])) / 2.0
        expected = spearmanr(obs, mean_pred).statistic
        assert aggregation.ensemble_patient_scores.loc["P2"] == pytest.approx(
            float(expected))

    def test_observed_mismatch_across_seeds_rejected(self):
        frame0 = _seed_frame([("P1", [1, 2, 3], [1, 2, 3])])
        frame1 = _seed_frame([("P1", [1, 2, 9], [1, 2, 3])])
        with pytest.raises(ValueError, match="observed differs"):
            aggregate_seed_patient_scores({0: frame0, 1: frame1})

    def test_key_set_mismatch_across_seeds_rejected(self):
        frame0 = _seed_frame([("P1", [1, 2, 3], [1, 2, 3])])
        frame1 = _seed_frame([
            ("P1", [1, 2, 3], [1, 2, 3]),
            ("P2", [1, 2, 3], [1, 2, 3]),
        ])
        with pytest.raises(ValueError, match="key set differs"):
            aggregate_seed_patient_scores({0: frame0, 1: frame1})


def test_patient_score_paired_bootstrap_manual():
    scores_a = pd.Series({"P1": 1.0, "P2": 0.8, "P3": 0.6})
    scores_b = pd.Series({"P1": 0.5, "P2": 0.5, "P3": 0.5})
    ci = patient_score_paired_bootstrap(scores_a, scores_b, n=200, seed=0)
    # 手算：delta = [0.5, 0.3, 0.1] → 点估计 = 0.3（患者级均值）
    assert ci.point_estimate == pytest.approx(np.mean([0.5, 0.3, 0.1]))
    assert ci.n_units == 3
    assert ci.unit == "patient_id"
    # 固定种子可复现
    ci2 = patient_score_paired_bootstrap(scores_a, scores_b, n=200, seed=0)
    assert (ci.low, ci.high) == (ci2.low, ci2.high)

    # 患者集合不一致 → 拒绝
    with pytest.raises(ValueError, match="patient sets differ"):
        patient_score_paired_bootstrap(
            scores_a, scores_b.reindex(["P1", "P2", "P4"]))
    # NaN 分数 → 拒绝（先资格过滤）
    bad = pd.Series({"P1": 1.0, "P2": np.nan, "P3": 0.6})
    with pytest.raises(ValueError, match="NaN"):
        patient_score_paired_bootstrap(bad, scores_b)


def test_canonical_chain_seed_mean_to_holm():
    """W3-07 主分析链路演示：逐种子患者分数 → 平均 → 配对统计 → family Holm。"""
    obs = [1.0, 2.0, 3.0, 4.0, 5.0]
    frame0 = _seed_frame([
        ("P1", obs, [1.0, 2.0, 3.0, 4.0, 5.0]),
        ("P2", obs, [5.0, 4.0, 3.0, 2.0, 1.0]),
    ])
    frame1 = _seed_frame([
        ("P1", obs, [1.0, 2.0, 3.0, 4.0, 5.0]),
        ("P2", obs, [5.0, 4.0, 3.0, 2.0, 1.0]),
    ])
    agg = aggregate_seed_patient_scores({0: frame0, 1: frame1})
    baseline = pd.Series({"P1": 0.0, "P2": 0.0})
    ci = patient_score_paired_bootstrap(
        agg.seed_mean_patient_scores, baseline, n=200, seed=0)
    # 每患者种子平均 = (1 + (-1)) / 2 = 0 → 配对差 = 0
    assert ci.point_estimate == pytest.approx(0.0)
    # 集成（独立模型）：P1 平均预测仍完美、P2 仍反序 → rho ±1 → 差 0
    ci_ens = patient_score_paired_bootstrap(
        agg.ensemble_patient_scores, baseline, n=200, seed=0)
    assert ci_ens.point_estimate == pytest.approx(0.0)
    # family Holm 收口：各比较 p 值经 permutation_p_value（有限修正），
    # 在预注册 family 内校正
    family = ComparisonFamily(name="demo", comparisons=(
        ComparisonSpec("model_minus_baseline"),
        ComparisonSpec("ensemble_minus_baseline")))
    perm_null = [0.0, 0.1, -0.05, 0.2, -0.1]
    rows = holm_correct_family(family, {
        "model_minus_baseline": permutation_p_value(
            ci.point_estimate, perm_null),
        "ensemble_minus_baseline": permutation_p_value(
            ci_ens.point_estimate, perm_null),
    })
    assert list(rows["comparison"]) == list(family.comparison_names)
    assert (rows["p_holm"] >= rows["p_raw"] - 1e-12).all()


# ---------------------------------------------------------------------------
# W3-08 折内同药跨患者相关性
# ---------------------------------------------------------------------------


class TestPerFoldDrugPatientCorrelation:

    def test_per_fold_independent_not_pooled(self):
        # fold0 反相关、fold1 完美相关：按折分别报告，不跨折池化。
        frame = pd.concat([
            _fold_frame(0, "X", [1, 2, 3, 4], [4, 3, 2, 1]),
            _fold_frame(1, "X", [1, 2, 3, 4], [1, 2, 3, 4]),
        ], ignore_index=True)
        out = per_fold_drug_patient_correlation(frame, min_patients=4)
        assert len(out) == 2
        by_fold = out.set_index("fold")["spearman"]
        assert by_fold.loc[0] == pytest.approx(-1.0)
        assert by_fold.loc[1] == pytest.approx(1.0)
        # 跨折池化值与两折值均不同（池化含折间伪变异，不得作为报告口径）
        pooled = spearmanr(
            [1, 2, 3, 4, 1, 2, 3, 4], [4, 3, 2, 1, 1, 2, 3, 4]).statistic
        assert by_fold.loc[0] != pytest.approx(float(pooled))
        assert by_fold.loc[1] != pytest.approx(float(pooled))

    def test_constant_prediction_nan_with_reason(self):
        frame = _fold_frame(0, "X", [1, 2, 3, 4], [2.0, 2.0, 2.0, 2.0])
        out = per_fold_drug_patient_correlation(frame, min_patients=4)
        row = out.iloc[0]
        assert np.isnan(row["spearman"])
        assert row["reason"] == "constant_predicted"

    def test_constant_truth_nan_with_reason(self):
        frame = _fold_frame(0, "X", [1, 1, 1, 1], [1, 2, 3, 4])
        out = per_fold_drug_patient_correlation(frame, min_patients=4)
        row = out.iloc[0]
        assert np.isnan(row["spearman"])
        assert row["reason"] == "constant_observed"

    def test_insufficient_patients_nan_with_reason(self):
        frame = _fold_frame(0, "X", [1, 2], [1, 2])
        out = per_fold_drug_patient_correlation(frame, min_patients=3)
        row = out.iloc[0]
        assert np.isnan(row["spearman"])
        assert row["reason"] == "insufficient_patients"

    def test_multi_patient_pdo_aggregation_before_correlation(self):
        # 同患者两行（多 PDO）先等权平均，再跨患者相关。
        rows = pd.concat([
            _fold_frame(0, "X", [1.0, 3.0], [1.0, 3.0], patients=["PA", "PB"]),
            _fold_frame(0, "X", [3.0, 1.0], [3.0, 1.0], patients=["PA", "PB"]),
        ], ignore_index=True)
        # PA: obs 均值 2、pred 均值 2；PB: obs 均值 2、pred 均值 2 → 常数
        out = per_fold_drug_patient_correlation(rows, min_patients=2)
        assert out.iloc[0]["reason"] == "constant_observed"

    def test_multiseed_frame_rejected(self):
        frame = _fold_frame(0, "X", [1, 2, 3, 4], [1, 2, 3, 4])
        frame["seed"] = [0, 0, 1, 1]
        with pytest.raises(ValueError, match="aggregate_seed_patient_scores"):
            per_fold_drug_patient_correlation(frame, min_patients=4)


def test_per_fold_gain_no_cross_fold_borrowing():
    model = pd.DataFrame([
        {"fold": 0, "drug_id": "X", "spearman": 0.5, "reason": None},
        {"fold": 1, "drug_id": "X", "spearman": 0.7, "reason": None},
    ])
    # 基线只在 fold0 有定义；fold1 缺失 → 增益 NaN，不跨折借用 fold0 基线
    baseline = pd.DataFrame([
        {"fold": 0, "drug_id": "X", "spearman": 0.2, "reason": None},
    ])
    out = per_fold_gain(model, baseline)
    out = out.set_index("fold")
    assert out.loc[0, "gain"] == pytest.approx(0.3)
    assert bool(out.loc[0, "gain_defined"]) is True
    assert np.isnan(out.loc[1, "gain"])
    assert bool(out.loc[1, "gain_defined"]) is False


def test_per_fold_gain_constant_baseline_undefined_not_negative():
    # 跨折有变异但折内常数的基线（如全局扰动均值）：折内未定义 → 增益
    # NaN，不得以跨折池化负基线充当增益依据。
    model = per_fold_drug_patient_correlation(pd.concat([
        _fold_frame(0, "X", [1, 2, 3, 4], [1, 2, 3, 4]),
        _fold_frame(1, "X", [1, 2, 3, 4], [4, 3, 2, 1]),
    ], ignore_index=True), min_patients=4)
    baseline = per_fold_drug_patient_correlation(pd.concat([
        _fold_frame(0, "X", [1, 2, 3, 4], [2.0, 2.0, 2.0, 2.0]),
        _fold_frame(1, "X", [1, 2, 3, 4], [2.5, 2.5, 2.5, 2.5]),
    ], ignore_index=True), min_patients=4)
    out = per_fold_gain(model, baseline).set_index("fold")
    for fold in (0, 1):
        assert np.isnan(out.loc[fold, "gain"])
        assert out.loc[fold, "baseline_reason"] == "constant_predicted"


# ---------------------------------------------------------------------------
# W3-09 并列秩 + 缺失掩码；置换 p 有限次数修正
# ---------------------------------------------------------------------------


def test_masked_spearman_ties_and_missing_match_scipy():
    a = [1.0, 2.0, 2.0, 3.0, 4.0, np.nan, 5.0]
    b = [2.0, 2.0, 3.0, 1.0, 4.0, 0.5, np.nan]
    rho, n_used = masked_spearman(a, b)
    # 成对完整参考：先删 NaN 再 scipy（并列秩 average）
    mask = ~np.isnan(np.asarray(a)) & ~np.isnan(np.asarray(b))
    expected = spearmanr(np.asarray(a)[mask], np.asarray(b)[mask]).statistic
    assert rho == pytest.approx(float(expected))
    assert n_used == 5


def test_masked_spearman_explicit_mask_subset():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [5.0, 4.0, 3.0, 2.0, 1.0]
    # 显式 mask 只取后 3 个位置：a=[3,4,5], b=[3,2,1] → rho=-1
    rho, n_used = masked_spearman(a, b, mask=[False, False, True, True, True])
    assert rho == pytest.approx(-1.0)
    assert n_used == 3
    # mask 与 NaN 叠加：有限性仍是必要条件
    rho2, n2 = masked_spearman(a, b, mask=[True, True, True, True, False])
    assert rho2 == pytest.approx(-1.0)
    assert n2 == 4


def test_masked_spearman_shape_and_constant_guards():
    with pytest.raises(ValueError, match="same shape"):
        masked_spearman([1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="mask must have the same shape"):
        masked_spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], mask=[True, False])
    rho, n_used = masked_spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    assert np.isnan(rho)  # 常数 → 未定义
    assert n_used == 3


def test_masked_pearson_matches_scipy_with_missing():
    rng = np.random.default_rng(7)
    a = rng.normal(size=40)
    b = a * 0.7 + rng.normal(size=40) * 0.4
    a[3] = np.nan
    b[11] = np.nan
    rho, n_used = masked_pearson(a, b)
    mask = ~np.isnan(a) & ~np.isnan(b)
    expected = pearsonr(a[mask], b[mask]).statistic
    assert rho == pytest.approx(float(expected))
    assert n_used == 38


def test_permutation_p_value_finite_correction():
    # 全部置换统计量低于观测 → 最小 p = 1/(n_perm+1)，禁止 p=0.0
    p = permutation_p_value(0.9, [0.1] * 100)
    assert p == pytest.approx(1.0 / 101.0)
    assert p > 0.0
    # 3 个置换 >= 观测 → p = (1+3)/101
    p2 = permutation_p_value(0.9, [0.95] * 3 + [0.1] * 97)
    assert p2 == pytest.approx(4.0 / 101.0)
    # less 方向：obs=-0.5，两个置换 <= obs → p = (1+2)/4
    p3 = permutation_p_value(-0.5, [-0.4, -0.6, -0.7], alternative="less")
    assert p3 == pytest.approx(3.0 / 4.0)
    # 观测未定义 → NaN
    assert np.isnan(permutation_p_value(np.nan, [0.1, 0.2]))
    # 方向参数校验
    with pytest.raises(ValueError, match="alternative"):
        permutation_p_value(0.1, [0.1, 0.2], alternative="two-sided")


# ---------------------------------------------------------------------------
# W3-10 预注册 family + Holm
# ---------------------------------------------------------------------------


def _demo_family():
    return ComparisonFamily(
        name="exp_main_comparisons",
        comparisons=(
            ComparisonSpec("a_minus_base", "arm a vs baseline"),
            ComparisonSpec("b_minus_base", "arm b vs baseline"),
            ComparisonSpec("c_minus_base", "arm c vs baseline"),
            ComparisonSpec("d_minus_base", "arm d vs baseline"),
        ))


def test_holm_correct_family_matches_manual():
    family = _demo_family()
    rows = holm_correct_family(family, {
        "a_minus_base": 0.01, "b_minus_base": 0.04,
        "c_minus_base": 0.03, "d_minus_base": 0.005})
    # 手算（Holm，m=4）：排序 0.005,0.01,0.03,0.04
    # → 0.02, 0.03, 0.06, 0.06（累积取大）
    by = rows.set_index("comparison")
    assert by.loc["a_minus_base", "p_holm"] == pytest.approx(0.03)
    assert by.loc["b_minus_base", "p_holm"] == pytest.approx(0.06)
    assert by.loc["c_minus_base", "p_holm"] == pytest.approx(0.06)
    assert by.loc["d_minus_base", "p_holm"] == pytest.approx(0.02)
    assert bool(by.loc["d_minus_base", "reject_holm"]) is True
    assert list(rows["comparison"]) == list(family.comparison_names)


def test_holm_correct_family_key_mismatch_rejected():
    family = _demo_family()
    with pytest.raises(ValueError, match="missing="):
        holm_correct_family(family, {"a_minus_base": 0.01})
    with pytest.raises(ValueError, match="extra="):
        holm_correct_family(family, {
            "a_minus_base": 0.01, "b_minus_base": 0.04,
            "c_minus_base": 0.03, "d_minus_base": 0.005,
            "e_minus_base": 0.02})


def test_holm_correct_family_rejects_seed_averaged_p():
    family = _demo_family()
    p_values = {
        "a_minus_base": ComparisonPValue("a_minus_base", 0.01, n_seeds_averaged=3),
        "b_minus_base": 0.04,
        "c_minus_base": 0.03,
        "d_minus_base": 0.005,
    }
    with pytest.raises(ValueError, match="跨种子平均 p 值"):
        holm_correct_family(family, p_values)
    # 单种子来源（默认 n_seeds_averaged=1）正常通过
    ok = {
        "a_minus_base": ComparisonPValue("a_minus_base", 0.01),
        "b_minus_base": 0.04, "c_minus_base": 0.03, "d_minus_base": 0.005,
    }
    rows = holm_correct_family(family, ok)
    assert len(rows) == 4


def test_comparison_family_serialization_roundtrip():
    family = _demo_family()
    obj = family.to_dict()
    restored = ComparisonFamily.from_dict(obj)
    assert restored.name == family.name
    assert restored.comparison_names == family.comparison_names
    assert restored.comparisons[1].description == "arm b vs baseline"
    with pytest.raises(ValueError, match="requires"):
        ComparisonFamily.from_dict({"name": "x"})


def test_comparison_family_validation():
    with pytest.raises(ValueError, match="duplicate"):
        ComparisonFamily(name="f", comparisons=(
            ComparisonSpec("a"), ComparisonSpec("a")))
    with pytest.raises(ValueError, match="at least one"):
        ComparisonFamily(name="f", comparisons=())
    with pytest.raises(ValueError, match="non-empty string"):
        ComparisonSpec("  ")


# ---------------------------------------------------------------------------
# W3-11 患者身份置换（身份映射广播）
# ---------------------------------------------------------------------------


def _patient_df():
    rows = []
    expr = {"P1": 1.0, "P2": 2.0, "P3": 3.0}
    for patient, value in expr.items():
        for drug in ("dA", "dB", "dC"):
            rows.append({
                "patient_id": patient,
                "drug_id": drug,
                "observed": float(len(rows)),
                "expr_gene1": value,
                "expr_gene2": value * 10.0,
            })
    return pd.DataFrame(rows)


def test_patient_identity_permutation_structure_preserved():
    df = _patient_df()
    original_expr = df.groupby("patient_id")[["expr_gene1", "expr_gene2"]].first()
    perms = list(permute_patient_identity_broadcast(
        df, 5, "patient_id", ("drug_id",),
        info_cols=("expr_gene1", "expr_gene2"), seed=0))
    assert len(perms) == 5
    for perm in perms:
        out = perm.frame
        # 行结构不变：患者×药物组合与观测标签保持（药物-患者对应不被打乱）
        assert sorted(zip(out["patient_id"], out["drug_id"])) == sorted(
            zip(df["patient_id"], df["drug_id"]))
        assert sorted(out["observed"]) == sorted(df["observed"])
        # 每患者的药物集合不变（仅身份信息来源换人）
        for patient in ("P1", "P2", "P3"):
            assert set(out[out["patient_id"] == patient]["drug_id"]) == {
                "dA", "dB", "dC"}
        # 每患者信息列唯一（供体整体广播到全部药物行）
        assert (out.groupby("patient_id")[["expr_gene1", "expr_gene2"]]
                .nunique() == 1).all().all()
        # 映射为全患者集合上的双射
        assert set(perm.mapping) == {"P1", "P2", "P3"}
        assert set(perm.mapping.values()) == {"P1", "P2", "P3"}
        # 信息值 = 供体的原信息（身份映射 + 广播验证）
        for patient in ("P1", "P2", "P3"):
            donor = perm.mapping[patient]
            row = out[out["patient_id"] == patient].iloc[0]
            assert row["expr_gene1"] == original_expr.loc[donor, "expr_gene1"]
            assert row["expr_gene2"] == original_expr.loc[donor, "expr_gene2"]


def test_patient_identity_permutation_within_split():
    rows = []
    expr = {"T1": 1.0, "T2": 2.0, "E1": 3.0, "E2": 4.0}
    for patient, value in expr.items():
        split = "train" if patient.startswith("T") else "test"
        for drug in ("dA", "dB"):
            rows.append({
                "patient_id": patient, "drug_id": drug,
                "evaluation_role": split, "expr_gene1": value,
            })
    df = pd.DataFrame(rows)
    train_patients = {"T1", "T2"}
    test_patients = {"E1", "E2"}
    for perm in permute_patient_identity_broadcast(
            df, 4, "patient_id", ("drug_id",),
            info_cols=("expr_gene1",), split_col="evaluation_role", seed=1):
        for source, donor in perm.mapping.items():
            if source in train_patients:
                assert donor in train_patients  # 训练侧不串测试侧
            else:
                assert donor in test_patients
        # 划分列本身不被置换改写
        assert set(perm.frame["evaluation_role"]) == {"train", "test"}


def test_patient_identity_permutation_mapping_only_mode():
    df = _patient_df()
    for perm in permute_patient_identity_broadcast(
            df, 2, "patient_id", ("drug_id",), seed=0):
        assert perm.frame is None
        assert set(perm.mapping) == {"P1", "P2", "P3"}


def test_patient_identity_permutation_guards():
    df = _patient_df()
    # 患者内不一致的信息列（非患者级信息）→ 拒绝
    bad = df.copy()
    bad.loc[0, "expr_gene1"] = 99.0
    with pytest.raises(ValueError, match="patient-level"):
        list(permute_patient_identity_broadcast(
            bad, 1, "patient_id", ("drug_id",), info_cols=("expr_gene1",)))
    # 重复 (patient, drug) 行 → 拒绝（药物-患者对应必须良构）
    dup = pd.concat([df, df[df["drug_id"] == "dA"]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        list(permute_patient_identity_broadcast(
            dup, 1, "patient_id", ("drug_id",)))
    # 缺列 → 拒绝
    with pytest.raises(ValueError, match="missing columns"):
        list(permute_patient_identity_broadcast(
            pd.DataFrame({"patient_id": ["P1"]}), 1, "patient_id",
            ("drug_id",)))


# ---------------------------------------------------------------------------
# 统一 canonical 入口
# ---------------------------------------------------------------------------


def _formal_frames():
    obs = [1.0, 2.0, 3.0, 4.0, 5.0]
    model_rows, baseline_rows = [], []
    for i in range(4):
        for j in range(5):
            common = {
                "experiment_id": "e1", "contract_id": "c1", "split_id": "s1",
                "fold": 0, "seed": 0, "patient_id": f"p{i}",
                "sample_id": f"p{i}_d{j}", "drug_id": f"d{j}",
                "y_true": obs[j],
            }
            model_rows.append({**common, "observed": obs[j],
                               "predicted": obs[j]})
            baseline_rows.append({**common, "observed": obs[j],
                                 "predicted": obs[4 - j]})
    return pd.DataFrame(model_rows), pd.DataFrame(baseline_rows)


def test_run_formal_comparison_end_to_end():
    model, baseline = _formal_frames()
    result = run_formal_comparison(
        model, baseline, comparison="model_minus_baseline",
        n_bootstrap=500, seed=0)
    # 每患者 model rho=1、baseline rho=-1 → 配对差恒 2 → CI=[2,2]
    assert result.point_estimate == pytest.approx(2.0)
    assert result.low == pytest.approx(2.0)
    assert result.high == pytest.approx(2.0)
    assert result.crosses_zero is False
    assert result.n_patients == 4
    assert result.comparison == "model_minus_baseline"
    # 患者分数（患者等权口径；rho 精确 ±1 有浮点尾差，用 approx）
    assert np.allclose(result.model_patient_scores.to_numpy(), 1.0)
    assert np.allclose(result.baseline_patient_scores.to_numpy(), -1.0)


def test_run_formal_comparison_rejects_incomparable():
    model, baseline = _formal_frames()
    tampered = baseline.copy()
    tampered.loc[0, "y_true"] = 999.0  # 标签不一致 → 比较检查器拒绝
    with pytest.raises(ValueError, match="y_true differs"):
        run_formal_comparison(model, tampered)


def test_run_formal_comparison_rejects_multiseed_frame():
    model, baseline = _formal_frames()
    multi = pd.concat([model, model.assign(seed=1)], ignore_index=True)
    with pytest.raises(ValueError, match="aggregate_seed_patient_scores"):
        run_formal_comparison(multi, baseline)


def test_unified_module_exports():
    """统一接口可从包级导入（唯一 canonical 路径可发现性）。"""
    from drug_screen.evaluation import (  # noqa: F401
        ComparisonFamily,
        aggregate_seed_patient_scores,
        holm_correct_family,
        masked_spearman,
        patient_score_paired_bootstrap,
        per_fold_drug_patient_correlation,
        permutation_p_value,
        permute_patient_identity_broadcast,
        run_formal_comparison,
    )
