import pytest

from scripts.data.audit_exp045_gene_axes import (
    _axis_sha256,
    _assert_exact_ordered_axis_match,
    _depmap_gene_symbols,
    _nature_expression_gene_symbols,
    _ordered_intersection,
    _positions,
)


def test_depmap_headers_keep_exact_symbol_and_drop_entrez_suffix():
    assert _depmap_gene_symbols(["ModelID", "A1BG (1)", "A2M (2)"]) == ["A1BG", "A2M"]


def test_depmap_header_mapping_fails_closed_on_duplicate_symbols():
    with pytest.raises(ValueError, match="duplicate normalized"):
        _depmap_gene_symbols(["ModelID", "GENE (1)", "GENE (2)"])


def test_nature_expression_rule_removes_only_terminal_exp_suffix():
    assert _nature_expression_gene_symbols(["sample_ID", "A1BG_exp", "A2M_exp"]) == ["A1BG", "A2M"]


def test_nature_expression_rejects_unregistered_non_feature_columns():
    with pytest.raises(ValueError, match="schema differs"):
        _nature_expression_gene_symbols(["sample_ID", "batch", "A1BG_exp"])


def test_common_axis_preserves_reference_order_and_position_maps():
    project = ["A", "B", "C", "D"]
    depmap = ["C", "A", "D"]
    nature = ["D", "C", "A", "B"]

    common = _ordered_intersection(project, depmap, nature)

    assert common == ["A", "C", "D"]
    assert _positions(project, common) == [0, 2, 3]
    assert _positions(depmap, common) == [1, 0, 2]
    assert _positions(nature, common) == [2, 1, 0]


def test_ordered_axis_hash_is_order_sensitive():
    assert _axis_sha256(["A", "B"]) != _axis_sha256(["B", "A"])


def test_source_axis_exact_order_match_passes():
    assert _assert_exact_ordered_axis_match(["A", "B", "C"], ["A", "B", "C"])


def test_source_axis_wrong_order_fails_closed():
    with pytest.raises(ValueError, match="exactly match expected order"):
        _assert_exact_ordered_axis_match(["A", "B", "C"], ["B", "A", "C"])


def test_source_axis_missing_gene_fails_closed():
    with pytest.raises(ValueError, match="exactly match expected order"):
        _assert_exact_ordered_axis_match(["A", "B", "C"], ["A", "C"])


def test_source_axis_duplicate_gene_fails_closed():
    with pytest.raises(ValueError, match="duplicate normalized"):
        _assert_exact_ordered_axis_match(["A", "B", "C"], ["A", "B", "B"])
