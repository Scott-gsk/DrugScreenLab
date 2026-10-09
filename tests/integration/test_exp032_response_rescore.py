"""用独立的 scipy 逐患者计算复核正式重算入口。"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.evaluation.compare_response_predictions import compare


SOURCE = Path(__file__).resolve().parents[2] / "artifacts/exp032_l3l4/predictions.parquet"


def test_exp032_c3_minus_c2_half_labels_independent_recalculation():
    reported = compare(SOURCE, "C3", "C2", 0.5, n_bootstrap=2000,
                       n_permutation=10000)
    frame = pd.read_parquet(SOURCE)
    frame = frame.loc[
        frame["arm"].isin(["C3", "C2"]) & frame["frac"].eq(0.5)
        & frame["evaluation_role"].eq("test")
    ]
    grouped = (frame.groupby(["arm", "seed", "patient_id", "drug_id"], sort=True)
               .agg(truth=("y_true", "mean"), prediction=("y_pred", "mean"))
               .reset_index())
    individual = {}
    for (arm, seed, patient), sub in grouped.groupby(["arm", "seed", "patient_id"]):
        assert sub["drug_id"].nunique() >= 3
        individual[(arm, seed, patient)] = float(
            spearmanr(sub["truth"].to_numpy(), sub["prediction"].to_numpy()).statistic)
    patients = sorted(frame["patient_id"].unique())
    seeds = sorted(frame["seed"].unique())
    deltas = [
        np.mean([individual[("C3", seed, patient)] for seed in seeds])
        - np.mean([individual[("C2", seed, patient)] for seed in seeds])
        for patient in patients
    ]
    assert len(patients) == reported["main"]["n_valid_patients"] == 51
    assert np.isclose(np.mean(deltas), reported["main"]["gain"], atol=1e-12)
    assert np.isclose(np.mean(deltas), 0.002869726698847289, atol=1e-12)
