"""跨资源矩阵的显式基因清单对齐（W3-30/W3-37 工程加固）。

审计背景（``docs/reports/audit_repair_wave3_20260922.md`` W3-30、W3-37）：
跨资源矩阵读取存在基因顺序错位风险——EXP-030 ``predictions.parquet``
的标签 sample-major / 数值 gene-major 行序错位即为实例。本模块固化约定：

- 跨资源矩阵必须按**显式基因清单**重排（``align_matrix_by_genes``）；
- 重排结果的列名集合、顺序、长度与清单完全一致；缺列/多列/长度不符
  时立即报错（缺失基因列表写入错误消息），禁止静默按位置拼接；
- 已组装完成的矩阵在读取入口用 ``assert_gene_alignment`` 做事后断言，
  捕获上游漂移。

按名映射的安全读取（如 ``transfer.preprocessing`` 的冻结基因链、缺失插补
契约）不适用强断言，本工具只用于要求列轴完全一致的跨资源对齐点。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "GeneAlignmentError",
    "assert_gene_alignment",
    "align_matrix_by_genes",
]


class GeneAlignmentError(ValueError):
    """矩阵基因列轴与显式清单不一致（缺列 / 多列 / 重复 / 顺序 / 长度）。"""


def _normalized_ids(values: Sequence[str], what: str) -> list[str]:
    ids = [str(v).strip() for v in values]
    if not ids:
        raise GeneAlignmentError(f"{what} must be non-empty")
    if len(set(ids)) != len(ids):
        seen: set[str] = set()
        dupes = sorted({v for v in ids if v in seen or seen.add(v)})
        raise GeneAlignmentError(f"{what} contains duplicate genes: {dupes[:10]}")
    return ids


def assert_gene_alignment(
    actual_columns: Sequence[str], expected_genes: Sequence[str]
) -> list[str]:
    """断言 ``actual_columns`` 与 ``expected_genes`` 集合、顺序、长度完全一致。

    通过时返回规范化（去空白）的清单列表，便于链式使用；任一不一致抛
    ``GeneAlignmentError``，消息携带缺失/多余基因列表或首个顺序分歧行。

    用途：已组装矩阵的读取入口事后断言（不重排、不改数据）。
    """
    expected = _normalized_ids(expected_genes, "expected_genes")
    actual = _normalized_ids(actual_columns, "actual_columns")
    expected_set = set(expected)
    actual_set = set(actual)
    missing = [g for g in expected if g not in actual_set]
    extra = [c for c in actual if c not in expected_set]
    if missing or extra:
        raise GeneAlignmentError(
            "gene axis set mismatch: "
            f"missing={missing[:20]}{'...' if len(missing) > 20 else ''} "
            f"extra={extra[:20]}{'...' if len(extra) > 20 else ''}")
    if len(actual) != len(expected):
        # 集合相等但长度不等只在存在重复时可能；重复已在规范化时拦截，
        # 此处为防御性兜底。
        raise GeneAlignmentError(
            f"gene axis length mismatch: actual {len(actual)} != expected "
            f"{len(expected)}")
    for position, (a, e) in enumerate(zip(actual, expected)):
        if a != e:
            raise GeneAlignmentError(
                f"gene order mismatch at position {position}: "
                f"actual {a!r} != expected {e!r}")
    return expected


def align_matrix_by_genes(
    matrix: pd.DataFrame | np.ndarray,
    genes: Sequence[str],
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame | np.ndarray:
    """按显式基因清单重排矩阵列；列集合与长度必须与清单完全一致。

    Parameters
    ----------
    matrix:
        ``pd.DataFrame``（基因为列），或 ``(values, columns)`` 二元组，
        或配合 ``columns=`` 的二维 ``np.ndarray``。
    genes:
        显式基因清单（目标列顺序）。清单内不得有重复。
    columns:
        ``matrix`` 为 ndarray 时的列名（与 ``matrix`` 为二元组时互斥）。

    Returns
    -------
    与输入同形态：DataFrame 输入返回按清单重排后的 DataFrame；
    ndarray 输入返回按清单重排后的 ndarray（列序 == ``genes``）。

    Raises
    ------
    GeneAlignmentError
        缺列（消息含缺失基因列表）、多列、重复或长度不一致时抛出。
        顺序不一致**不**抛错——按清单重排正是本函数的职责。
    """
    expected = _normalized_ids(genes, "genes")

    if isinstance(matrix, pd.DataFrame):
        if columns is not None:
            raise ValueError("columns must be None when matrix is a DataFrame")
        actual = _normalized_ids(matrix.columns, "matrix.columns")
        return _reorder_frame(matrix, actual, expected)
    if isinstance(matrix, tuple):
        if columns is not None:
            raise ValueError("columns must be None when matrix is a (values, columns) pair")
        if len(matrix) != 2:
            raise ValueError("(values, columns) pair must have exactly two elements")
        values, col_names = matrix
        actual = _normalized_ids(col_names, "columns")
        return _reorder_array(values, actual, expected)
    if columns is None:
        raise ValueError("ndarray input requires columns=")
    actual = _normalized_ids(columns, "columns")
    return _reorder_array(matrix, actual, expected)


def _check_set_equality(actual: list[str], expected: list[str]) -> None:
    actual_set = set(actual)
    missing = [g for g in expected if g not in actual_set]
    if missing:
        raise GeneAlignmentError(
            "matrix is missing genes: "
            f"{missing[:20]}{'...' if len(missing) > 20 else ''}")
    expected_set = set(expected)
    extra = [c for c in actual if c not in expected_set]
    if extra:
        raise GeneAlignmentError(
            "matrix has extra genes not in the list: "
            f"{extra[:20]}{'...' if len(extra) > 20 else ''}")
    if len(actual) != len(expected):
        # 集合相等但长度不等只在存在重复时可能；重复已在规范化时拦截，
        # 此处为防御性兜底（长度不一致必须报错）。
        raise GeneAlignmentError(
            f"gene axis length mismatch: actual {len(actual)} != expected "
            f"{len(expected)}")


def _reorder_frame(frame: pd.DataFrame, actual: list[str], expected: list[str]) -> pd.DataFrame:
    _check_set_equality(actual, expected)
    return frame.loc[:, expected]


def _reorder_array(
    values: np.ndarray, actual: list[str], expected: list[str]
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise GeneAlignmentError("matrix must be two-dimensional (samples x genes)")
    if array.shape[1] != len(actual):
        raise GeneAlignmentError(
            f"matrix width {array.shape[1]} does not match number of column "
            f"names {len(actual)}")
    _check_set_equality(actual, expected)
    if actual == expected:
        return array
    positions = {gene: index for index, gene in enumerate(actual)}
    return array[:, [positions[gene] for gene in expected]]
