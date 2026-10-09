"""Verify the step-down correction used for the fixed B1/B2/B5 family."""

import pytest

from scripts.evaluation.aggregate_route_b_holm import holm_adjust


def test_holm_adjustment_is_monotone_and_keeps_original_names() -> None:
    adjusted = holm_adjust({"B1": 0.01, "B2": 0.04, "B5": 0.03})
    assert adjusted == {"B1": pytest.approx(0.03),
                        "B5": pytest.approx(0.06),
                        "B2": pytest.approx(0.06)}


@pytest.mark.parametrize("values", [
    {"B1": float("nan"), "B2": 0.2},
    {"B1": -0.1},
    {"B1": 1.1},
])
def test_holm_rejects_invalid_family_p_values(values) -> None:
    with pytest.raises(ValueError):
        holm_adjust(values)
