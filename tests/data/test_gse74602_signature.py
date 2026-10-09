import numpy as np
import pytest

from scripts.data.build_gse74602_crc_signature import quantile_normalize_positive
from scripts.data.verify_gse74602_crc_signature import rank_mean_normalize


def test_quantile_normalization_gives_each_sample_the_shared_distribution():
    raw = np.array([[1.0, 10.0], [5.0, 40.0], [9.0, 30.0]])

    normalized = quantile_normalize_positive(raw)

    np.testing.assert_allclose(np.sort(normalized[:, 0]), np.sort(normalized[:, 1]))
    np.testing.assert_allclose(normalized, rank_mean_normalize(raw))


@pytest.mark.parametrize("values", [np.array([[0.0]]), np.array([[np.nan]])])
def test_quantile_normalization_rejects_nonpositive_or_nonfinite_signals(values):
    with pytest.raises(ValueError, match="finite and positive"):
        quantile_normalize_positive(values)
