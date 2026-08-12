import numpy as np

from drug_screen.modeling.mechanism_fast import MECHANISM_DIM, mechanism_feature_vector


def test_mechanism_prior_is_fixed_and_sparse() -> None:
    first = mechanism_feature_vector(["MET"], ["R-HSA-6806942"])
    second = mechanism_feature_vector(["MET"], ["R-HSA-6806942"])
    assert first.shape == (MECHANISM_DIM,)
    assert np.array_equal(first, second)
    assert first[0] == 1 / 16
    assert first[1] == 1 / 64
    assert np.count_nonzero(first) >= 4
