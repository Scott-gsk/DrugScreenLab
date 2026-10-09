from scripts.data.breast_subtype_evidence import _is_exact_ctrp_status, receptor_pattern


def test_receptor_pattern_uses_explicit_receptor_calls_only():
    assert receptor_pattern("+", "-", "-") == "HR_positive_HER2_negative"
    assert receptor_pattern("-", "+", "+") == "HR_positive_HER2_positive"
    assert receptor_pattern("-", "-", "+") == "HR_negative_HER2_positive"
    assert receptor_pattern("-", "-", "-") == "triple_negative_receptor_pattern"


def test_receptor_pattern_keeps_ambiguous_or_missing_calls_unknown():
    assert receptor_pattern("+", "+/-", "-") == "unknown_incomplete_or_indeterminate"
    assert receptor_pattern("-", "-", "") == "unknown_incomplete_or_indeterminate"


def test_documented_cross_source_conflict_overrides_single_table_pattern():
    assert receptor_pattern("+", "-", "+", conflicting_source=True) == "unknown_public_annotation_conflict"


def test_ctrp_status_does_not_count_negative_exact_search_status_as_match():
    assert _is_exact_ctrp_status("exact_unique_cellosaurus_rrid")
    assert not _is_exact_ctrp_status("no_exact_CTRP_cell_mapping")

