"""EXP-026 归一化原语单元测试（训练折拟合、测试折变换的防泄漏接口）。"""
from __future__ import annotations

import numpy as np
import pytest

from drug_screen.data.normalization import (
    QuantileNormalizer,
    RankNormalizer,
    ZScoreScaler,
)


@pytest.fixture()
def rng():
    return np.random.default_rng(20260919)


def test_zscore_train_statistics(rng):
    X = rng.normal(loc=3.0, scale=2.0, size=(200, 7))
    Z = ZScoreScaler().fit_transform(X)
    assert Z.shape == X.shape
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(Z.std(axis=0), 1.0, atol=1e-3)


def test_zscore_test_uses_train_statistics(rng):
    Xtr = rng.normal(loc=0.0, scale=1.0, size=(100, 4))
    Xte = rng.normal(loc=5.0, scale=3.0, size=(50, 4))
    sc = ZScoreScaler().fit(Xtr)
    Zte = sc.transform(Xte)
    # 用训练统计 (mu≈0, sd≈1) 变换：均值≈5、标准差≈3（抽样波动放宽）
    assert np.allclose(Zte.mean(axis=0), 5.0, atol=0.5)
    assert np.allclose(Zte.std(axis=0), 3.0, atol=1.0)


def test_quantile_train_columns_share_reference(rng):
    G, N = 6, 300
    scales = np.logspace(-1, 1, G)
    X = rng.normal(size=(N, G)) * scales + np.arange(G) * 4.0
    qn = QuantileNormalizer()
    Z = qn.fit_transform(X)
    # 变换后各列排序值应与参考分布一致（分位对齐）
    sorted_Z = np.sort(Z, axis=0)
    assert np.allclose(sorted_Z, qn.reference_[:, None], atol=1e-10)


def test_quantile_monotone_and_clipped(rng):
    Xtr = rng.normal(size=(200, 3))
    qn = QuantileNormalizer().fit(Xtr)
    Xte = np.array([
        [-1e3, 0.0, 1e3],
        [0.0, 0.0, 0.0],
        [1e3, 0.0, -1e3],
    ])
    Zte = qn.transform(Xte)
    assert np.all(np.isfinite(Zte))
    # 越界值截断到参考分布端点
    assert Zte[0, 0] == qn.reference_[0]
    assert Zte[0, 2] == qn.reference_[-1]
    # 每列单调（顺序保持）
    mono = qn.transform(np.array([[-2.0, -2.0, -2.0],
                                  [0.0, 0.0, 0.0],
                                  [2.0, 2.0, 2.0]]))
    assert np.all(np.diff(mono, axis=0) >= -1e-12)


def test_rank_in_open_unit_interval(rng):
    Xtr = rng.normal(size=(150, 5))
    Xte = rng.normal(size=(40, 5)) * 10.0
    Z = RankNormalizer().fit(Xtr).transform(Xte)
    assert np.all(Z > 0.0) and np.all(Z < 1.0)


def test_rank_monotone_within_column(rng):
    Xtr = rng.normal(size=(100, 2))
    rn = RankNormalizer().fit(Xtr)
    col = np.array([[-3.0, -3.0], [0.0, 0.0], [3.0, 3.0]])
    Z = rn.transform(col)
    assert np.all(np.diff(Z, axis=0) > 0)


def rank_reference_ties_handled():
    Xtr = np.zeros((10, 1))  # 常数列
    Z = RankNormalizer().fit(Xtr).transform(np.zeros((3, 1)))
    assert np.allclose(Z, (10 + 0.5) / 11.0)


def test_dimension_mismatch_raises(rng):
    Xtr = rng.normal(size=(50, 3))
    Xbad = rng.normal(size=(10, 4))
    with pytest.raises(ValueError):
        QuantileNormalizer().fit(Xtr).transform(Xbad)
    with pytest.raises(ValueError):
        RankNormalizer().fit(Xtr).transform(Xbad)
