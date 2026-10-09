"""跨资源矩阵基因清单对齐工具单元测试（W3-30/W3-37 工程加固）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drug_screen.data.gene_alignment import (
    GeneAlignmentError,
    align_matrix_by_genes,
    assert_gene_alignment,
)


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(size=(6, 4)), columns=["A1BG", "A2M", "BRCA1", "TP53"]
    )


def _genes() -> list[str]:
    return ["A1BG", "A2M", "BRCA1", "TP53"]


# ------------------------------------------------------------------ align ---

def test_align_dataframe_shuffled_columns_restores_reference_order() -> None:
    frame = _frame()
    shuffled = frame.loc[:, ["TP53", "A1BG", "BRCA1", "A2M"]]
    aligned = align_matrix_by_genes(shuffled, _genes())
    assert isinstance(aligned, pd.DataFrame)
    # 打乱列序后对齐：列序与数值和原序完全一致。
    pd.testing.assert_frame_equal(aligned, frame)


def test_align_dataframe_already_ordered_is_identity() -> None:
    frame = _frame()
    aligned = align_matrix_by_genes(frame, _genes())
    pd.testing.assert_frame_equal(aligned, frame)


def test_align_ndarray_pair_shuffled_columns() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(size=(5, 4))
    shuffled_cols = ["TP53", "A2M", "A1BG", "BRCA1"]
    aligned = align_matrix_by_genes((values, shuffled_cols), _genes())
    assert isinstance(aligned, np.ndarray)
    expected = values[:, [2, 1, 3, 0]]
    np.testing.assert_array_equal(aligned, expected)


def test_align_ndarray_with_columns_keyword() -> None:
    rng = np.random.default_rng(2)
    values = rng.normal(size=(3, 4))
    aligned = align_matrix_by_genes(values, _genes(), columns=["BRCA1", "A2M", "TP53", "A1BG"])
    np.testing.assert_array_equal(aligned, values[:, [3, 1, 0, 2]])


def test_align_missing_gene_raises_with_missing_list() -> None:
    frame = _frame().drop(columns=["BRCA1"])
    with pytest.raises(GeneAlignmentError, match="missing genes.*BRCA1"):
        align_matrix_by_genes(frame, _genes())


def test_align_extra_gene_raises() -> None:
    frame = _frame().assign(EGFR=np.zeros(6))
    with pytest.raises(GeneAlignmentError, match="extra genes.*EGFR"):
        align_matrix_by_genes(frame, _genes())


def test_align_shorter_column_axis_raises() -> None:
    # 长度不一致（列数少于清单）必须报错，且给出缺失基因列表。
    frame = _frame().loc[:, ["A1BG", "A2M"]]
    with pytest.raises(GeneAlignmentError, match="missing genes"):
        align_matrix_by_genes(frame, _genes())


def test_align_duplicate_matrix_columns_raise() -> None:
    frame = _frame().set_axis(["A1BG", "A2M", "A2M", "BRCA1"], axis=1)
    with pytest.raises(GeneAlignmentError, match="duplicate genes"):
        align_matrix_by_genes(frame, _genes())


def test_align_duplicate_list_genes_raise() -> None:
    frame = _frame()
    with pytest.raises(GeneAlignmentError, match="duplicate genes"):
        align_matrix_by_genes(frame, ["A1BG", "A2M", "A2M", "BRCA1"])


def test_align_empty_list_raises() -> None:
    with pytest.raises(GeneAlignmentError, match="non-empty"):
        align_matrix_by_genes(_frame(), [])


def test_align_ndarray_requires_columns() -> None:
    with pytest.raises(ValueError, match="requires columns"):
        align_matrix_by_genes(np.zeros((2, 2)), _genes())


def test_align_rejects_columns_with_dataframe() -> None:
    with pytest.raises(ValueError, match="must be None"):
        align_matrix_by_genes(_frame(), _genes(), columns=_genes())


# ----------------------------------------------------------------- assert ---

def test_assert_gene_alignment_exact_match_passes() -> None:
    normalized = assert_gene_alignment(["A1BG", "A2M", "BRCA1", "TP53"], _genes())
    assert normalized == _genes()


def test_assert_gene_alignment_order_mismatch_reports_position() -> None:
    with pytest.raises(GeneAlignmentError, match="order mismatch at position 1"):
        assert_gene_alignment(["A1BG", "BRCA1", "A2M", "TP53"], _genes())


def test_assert_gene_alignment_missing_and_extra_reported() -> None:
    with pytest.raises(GeneAlignmentError, match="missing=.*A2M.*extra=.*EGFR"):
        assert_gene_alignment(["A1BG", "EGFR", "BRCA1", "TP53"], _genes())


def test_assert_gene_alignment_length_mismatch() -> None:
    # 长度不一致报错（列数少于清单 → 缺失基因列表）。
    with pytest.raises(GeneAlignmentError, match="missing"):
        assert_gene_alignment(["A1BG", "A2M"], _genes())


def test_assert_gene_alignment_extra_columns() -> None:
    # 长度不一致报错（列数多于清单 → 多余基因列表）。
    with pytest.raises(GeneAlignmentError, match="extra=.*EGFR"):
        assert_gene_alignment(["A1BG", "A2M", "BRCA1", "TP53", "EGFR"], _genes())


def test_align_output_always_satisfies_assert() -> None:
    frame = _frame().loc[:, ["TP53", "BRCA1", "A2M", "A1BG"]]
    aligned = align_matrix_by_genes(frame, _genes())
    assert isinstance(aligned, pd.DataFrame)
    # 对齐结果恒满足强断言（列名、顺序、长度完全一致）。
    assert_gene_alignment(aligned.columns, _genes())
