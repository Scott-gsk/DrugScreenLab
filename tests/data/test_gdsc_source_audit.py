from drug_screen.data.gdsc_source_audit import _frame_equal, _normalized_drug_name

import pandas as pd


def test_frame_equal_ignores_dtype_but_not_values():
    assert _frame_equal(pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [1.0]}))
    assert not _frame_equal(pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [2]}))


def test_legacy_drug_alias_normalization_only_removes_punctuation_and_case():
    assert _normalized_drug_name("BMS-754807") == _normalized_drug_name("BMS.754807")
    assert _normalized_drug_name("BMS-754807") != _normalized_drug_name("BMS-754808")
