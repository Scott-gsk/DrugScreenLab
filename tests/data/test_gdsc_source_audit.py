from drug_screen.data.gdsc_source_audit import _frame_equal

import pandas as pd


def test_frame_equal_ignores_dtype_but_not_values():
    assert _frame_equal(pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [1.0]}))
    assert not _frame_equal(pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [2]}))
