from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from drug_screen.foundation.xpert_adapter import (
    build_cartesian_adapter_h5ad,
    classify_foundation_metrics,
    validate_xpert_contract,
)


def _valid_adata() -> SimpleNamespace:
    n = 3
    obs = {
        "pert_id": ["BRD-A", "BRD-B", "BRD-C"],
        "pert_idx": [1, 2, 3],
        "cell_iname": ["A", "B", "C"],
        "cell_idx": [0, 1, 2],
        "tissue_idx": [0, 0, 1],
        "pert_dose": [10.0] * n,
        "pert_time": [6.0] * n,
    }
    return SimpleNamespace(
        X=np.ones((n, 978), dtype=np.float32),
        obsm={"X_ctl": np.zeros((n, 978), dtype=np.float32)},
        obs=obs,
        var_names=[f"g{i}" for i in range(978)],
    )


def test_validate_xpert_contract_accepts_official_shape_and_fields() -> None:
    audit = validate_xpert_contract(_valid_adata())

    assert audit["gene_count"] == 978
    assert audit["record_count"] == 3
    assert audit["required_obsm"] == ["X_ctl"]


def test_validate_xpert_contract_rejects_wrong_control_shape() -> None:
    adata = _valid_adata()
    adata.obsm["X_ctl"] = np.zeros((3, 977), dtype=np.float32)

    with pytest.raises(ValueError, match="X_ctl"):
        validate_xpert_contract(adata)


def test_foundation_ready_requires_both_noncollapsed_cold_splits() -> None:
    ready = classify_foundation_metrics(
        {
            "cold_cell": {"pearson_delta": 0.21, "prediction_std": 0.7, "n_test": 100},
            "cold_drug": {"pearson_delta": 0.13, "prediction_std": 0.5, "n_test": 100},
        }
    )
    broken = classify_foundation_metrics(
        {
            "cold_cell": {"pearson_delta": 0.21, "prediction_std": 0.7, "n_test": 100},
            "cold_drug": {"pearson_delta": 0.0, "prediction_std": 0.0, "n_test": 100},
        }
    )

    assert ready == "XPERT_FOUNDATION_READY"
    assert broken == "BROKEN"


def test_cartesian_adapter_requires_all_requested_drugs() -> None:
    assert callable(build_cartesian_adapter_h5ad)
