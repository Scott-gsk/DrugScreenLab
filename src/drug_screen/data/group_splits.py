"""按唯一生物身份冻结外层与训练侧内层划分。"""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil

import numpy as np
from sklearn.model_selection import GroupKFold


def nested_identity_folds(
    identities: Sequence[str], *, seed: int, outer_folds: int = 5,
    inner_folds: int = 3, fractions: Sequence[float] = (0.25, 0.5, 1.0),
) -> list[dict]:
    """只用身份生成划分；每档训练患者为前一档的超集。

    每个外层测试身份恰出现一次，内层验证只来自该外层训练身份。
    不读取标签、表达或药物，因而无法按测试结果选择分割。
    """
    ids = [str(x).strip() for x in identities]
    if any(not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("identities must be non-empty and unique")
    if len(ids) < outer_folds or outer_folds < 2 or inner_folds < 2:
        raise ValueError("insufficient identities or invalid fold count")
    levels = tuple(float(x) for x in fractions)
    if not levels or sorted(set(levels)) != list(levels) or levels[-1] != 1.0:
        raise ValueError("fractions must be unique increasing levels ending at 1.0")
    if any(not 0 < x <= 1 for x in levels):
        raise ValueError("fractions must lie in (0,1]")

    ordered = sorted(ids)
    shuffled = [ordered[i] for i in np.random.RandomState(seed).permutation(len(ordered))]
    outer = GroupKFold(n_splits=outer_folds)
    result: list[dict] = []
    tested: set[str] = set()
    for fold, (train_idx, test_idx) in enumerate(
        outer.split(shuffled, groups=shuffled)
    ):
        train = [shuffled[int(i)] for i in train_idx]
        test = sorted(shuffled[int(i)] for i in test_idx)
        if tested.intersection(test):
            raise AssertionError("identity appears in multiple test folds")
        tested.update(test)
        rng = np.random.RandomState(seed + fold + 1)
        train_order = [train[int(i)] for i in rng.permutation(len(train))]
        levels_out = []
        previous: set[str] = set()
        for fraction in levels:
            chosen = train_order[:ceil(len(train_order) * fraction)]
            chosen_set = set(chosen)
            if not previous.issubset(chosen_set):
                raise AssertionError("nested label fractions are not nested")
            previous = chosen_set
            if len(chosen) < inner_folds:
                levels_out.append({"fraction": fraction, "train_ids": sorted(chosen),
                                   "inner_folds": [], "status": "NOT_EVALUABLE_TOO_FEW_IDENTITIES"})
                continue
            inner_shuffled = [chosen[int(i)] for i in
                              np.random.RandomState(seed + 100 + fold).permutation(len(chosen))]
            inner = GroupKFold(n_splits=inner_folds)
            inner_out = []
            for inner_fold, (itr, iva) in enumerate(
                inner.split(inner_shuffled, groups=inner_shuffled)
            ):
                inner_out.append({
                    "inner_fold": inner_fold,
                    "train_ids": sorted(inner_shuffled[int(i)] for i in itr),
                    "validation_ids": sorted(inner_shuffled[int(i)] for i in iva),
                })
            levels_out.append({"fraction": fraction, "train_ids": sorted(chosen),
                               "inner_folds": inner_out, "status": "SPLIT_READY_LABEL_COVERAGE_PENDING"})
        result.append({"fold": fold, "outer_train_ids": sorted(train),
                       "outer_test_ids": test, "label_fractions": levels_out})
    if tested != set(ids):
        raise AssertionError("outer test folds do not cover all identities")
    return result
