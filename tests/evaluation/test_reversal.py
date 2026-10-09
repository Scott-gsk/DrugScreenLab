"""方案 6.5 节固定逆转读出的手算边界。"""

import pytest

from drug_screen.evaluation.reversal import reversal_score


GENES = ("u1", "u2", "d1", "d2")
UP = ("u1", "u2")
DOWN = ("d1", "d2")


def test_perfect_reversal_and_same_direction():
    assert reversal_score((-2, -1, 2, 1), GENES, UP, DOWN) == 1.0
    assert reversal_score((2, 1, -2, -1), GENES, UP, DOWN) == -1.0


def test_all_zero_and_tied_genes_do_not_depend_on_gene_name_order():
    assert reversal_score((0, 0, 0, 0), GENES, UP, DOWN) == 0.0
    # 同值组整组评价：上调和下调各占一半，不能按名称任意拆开。
    assert reversal_score((2, 2, -1, -1), GENES, UP, DOWN) == -1.0
    assert reversal_score((2, -1, 2, -1), GENES, UP, DOWN) == 0.0
    assert reversal_score((-1, 2, -1, 2), GENES, UP, DOWN) == 0.0


@pytest.mark.parametrize(
    "delta, genes, up, down",
    [
        ((1, 2, 3), GENES, UP, DOWN),
        ((1, 2, 3, 4), ("u1", "u1", "d1", "d2"), UP, DOWN),
        ((1, 2, 3, float("nan")), GENES, UP, DOWN),
        ((1, 2, 3, 4), GENES, ("absent",), DOWN),
        ((1, 2, 3, 4), GENES, UP, ("u1", "d1")),
        ((1, 2, 3, 4), GENES, (), DOWN),
    ],
)
def test_invalid_axis_or_signature_is_rejected(delta, genes, up, down):
    with pytest.raises(ValueError):
        reversal_score(delta, genes, up, down)
