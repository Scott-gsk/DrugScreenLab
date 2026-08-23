import json

from scripts.analysis.run_exp010_breast_inputs import _bh, multiregional_status


def test_bh_is_monotone_in_pvalue_order():
    adjusted = _bh(__import__("numpy").array([.04, .001, .02]))
    assert list(adjusted[[1, 2, 0]]) == sorted(adjusted)


def test_multiregional_access_failure_is_formal_not_evaluable(tmp_path):
    path = tmp_path / "access.json"
    path.write_text(json.dumps({"status": "ACCESS_BLOCKED", "attempts": [{"accepted_as_data": False}],
                                "data_decision": "real attachments unavailable", "expected_assets": ["S1.xlsx"]}))
    result = multiregional_status(path)
    assert result["status"] == "NOT_EVALUABLE_PUBLIC_MATRIX"
    assert result["accepted_asset_count"] == 0
