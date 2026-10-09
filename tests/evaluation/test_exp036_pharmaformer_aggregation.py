"""患者配对统计必须保留完整患者分母并给有限置换 p 值。"""

import numpy as np
import pytest

from scripts.evaluation.aggregate_exp036_pharmaformer import paired_statistics


def test_all_positive_patient_gain_has_finite_corrected_p_value() -> None:
    result = paired_statistics(np.ones(51), seed=7)
    assert result["estimate"] == 1.0
    assert result["ci95"] == [1.0, 1.0]
    assert result["n_patients"] == 51
    assert 0 < result["p_two_sided_uncorrected"] <= 1


def test_missing_or_undefined_patient_gain_blocks_confirmatory_aggregation() -> None:
    with pytest.raises(ValueError, match="51 finite patients"):
        paired_statistics(np.ones(50), seed=7)
    invalid = np.ones(51)
    invalid[10] = np.nan
    with pytest.raises(ValueError, match="51 finite patients"):
        paired_statistics(invalid, seed=7)
