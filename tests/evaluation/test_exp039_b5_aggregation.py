"""Scientific safeguards for the registered B5 paired-cell statistics."""

import numpy as np
import pandas as pd
import pytest

from scripts.evaluation.aggregate_exp039_b5 import _complete_seed_mean
from scripts.evaluation.paired_cell_statistics import paired_cell_statistics


def test_b5_gain_statistics_use_cells_and_finite_corrected_sign_flip_p() -> None:
    result = paired_cell_statistics(np.ones(6, dtype=float), seed=20260922)
    assert result["estimate"] == 1.0
    assert result["ci95"] == [1.0, 1.0]
    # With six independent cells, all signs align in about 2/2^6 of flips;
    # finite Monte Carlo p must reflect that coarse attainable resolution.
    assert 0.02 < result["p_two_sided_finite_corrected"] < 0.04
    assert result["n_cells"] == 6
    assert result["unit"] == "held-out cell"


def test_b5_seed_aggregation_requires_all_three_seeds_per_cell() -> None:
    frame = pd.DataFrame([
        {"fold": 0, "cell_id": "A", "arm": "huber", "seed": seed,
         "spearman": score, "n_drugs": 5}
        for seed, score in zip((20260922, 20260923, 20260924), (0.2, 0.4, 0.6))
    ] + [
        {"fold": 0, "cell_id": "B", "arm": "huber", "seed": 20260922,
         "spearman": 0.5, "n_drugs": 5},
        {"fold": 0, "cell_id": "B", "arm": "huber", "seed": 20260923,
         "spearman": np.nan, "n_drugs": 5},
        {"fold": 0, "cell_id": "B", "arm": "huber", "seed": 20260924,
         "spearman": 0.7, "n_drugs": 5},
    ])
    averaged = _complete_seed_mean(frame, "spearman").set_index("cell_id")
    assert averaged.loc["A", "spearman"] == pytest.approx(0.4)
    assert np.isnan(averaged.loc["B", "spearman"])
    assert averaged.loc["A", "n_seeds"] == 3
