"""Check the independent reversal verifier against fixed hand outcomes."""

import numpy as np
import pytest

from scripts.evaluation.verify_exp039_b5_reversal import reversal_reference


GENES = ["u1", "u2", "d1", "d2"]
UP = {"u1", "u2"}
DOWN = {"d1", "d2"}


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        ([-2, -1, 2, 1], 1.0),
        ([2, 1, -2, -1], -1.0),
        ([0, 0, 0, 0], 0.0),
        ([2, 2, -1, -1], -1.0),
        ([2, -1, 2, -1], 0.0),
    ],
)
def test_independent_prefix_walk_matches_frozen_hand_cases(delta, expected) -> None:
    assert reversal_reference(np.asarray(delta, dtype=float), GENES, UP, DOWN) == pytest.approx(expected)
