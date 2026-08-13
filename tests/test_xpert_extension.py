from __future__ import annotations

import numpy as np
import pandas as pd

from drug_screen.foundation.xpert_extension import PerturbagenEncoder, append_perturbagen_tokens
from drug_screen.foundation.xpert_registry import (
    build_global_cartesian_adapter_h5ad,
    eligible_registry_drugs,
    validate_drug_registry_payload,
)


def test_perturbagen_encoder_keeps_additive_features_as_separate_tokens() -> None:
    torch = __import__("torch")
    encoder = PerturbagenEncoder(
        hidden_size=16,
        kpgt_dim=8,
        unipert_dim=4,
        use_kpgt=True,
        use_unipert=True,
        dropout=0.0,
    )
    tokens = encoder(
        kpgt=torch.ones((3, 8)),
        unipert=torch.ones((3, 4)),
    )
    assert tokens.shape == (3, 2, 16)
    assert torch.isfinite(tokens).all()


def test_additive_tokens_are_appended_without_merging_feature_spaces() -> None:
    torch = __import__("torch")
    base = torch.zeros((2, 5, 16))
    extra = torch.ones((2, 2, 16))
    result = append_perturbagen_tokens(base, extra)
    assert result.shape == (2, 7, 16)
    assert torch.equal(result[:, :5], base)
    assert torch.equal(result[:, 5:], extra)


def test_registry_requires_baseline_features_and_keeps_broad_identity_separate() -> None:
    payload = {
        "format": "xpert_drug_registry_v1",
        "drugs": [
            {"pert_id": "BRD-A", "pert_idx": 1, "unimol": True, "hg": True, "kpgt": True, "broad_identity": True},
            {"pert_id": "BRD-B", "pert_idx": 2, "unimol": True, "hg": True, "kpgt": False, "broad_identity": True},
            {"pert_id": "BRD-C", "pert_idx": 3, "unimol": True, "hg": True, "kpgt": True, "broad_identity": False},
        ],
    }
    validate_drug_registry_payload(payload)
    selected = eligible_registry_drugs(payload)
    assert selected == ["BRD-A"]


def test_global_cartesian_uses_registry_drugs_not_source_adapter_drugs(tmp_path) -> None:
    ad = __import__("anndata")
    n_genes = 978
    source = ad.AnnData(
        X=np.zeros((2, n_genes), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "pert_id": ["SOURCE-A", "SOURCE-B"],
                "pert_idx": [1, 2],
                "cell_iname": ["CTX-A", "CTX-B"],
                "cell_idx": [0, 1],
                "tissue_idx": [0, 1],
                "pert_dose": [10.0, 10.0],
                "pert_time": [6.0, 6.0],
            }
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
    )
    source.obsm["X_ctl"] = np.ones((2, n_genes), dtype=np.float32)
    source_path = tmp_path / "source.h5ad"
    output_path = tmp_path / "global.h5ad"
    source.write_h5ad(source_path)
    registry = {
        "format": "xpert_drug_registry_v1",
        "drugs": [
            {"pert_id": "SOURCE-A", "pert_idx": 1, "unimol": True, "hg": True, "kpgt": True, "broad_identity": True},
            {"pert_id": "GLOBAL-C", "pert_idx": 3, "unimol": True, "hg": True, "kpgt": True, "broad_identity": True},
        ],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(__import__("json").dumps(registry), encoding="utf-8")

    audit = build_global_cartesian_adapter_h5ad(
        source_path=source_path,
        registry_path=registry_path,
        output_path=output_path,
    )
    result = ad.read_h5ad(output_path)
    assert audit["record_count"] == 4
    assert sorted(result.obs["pert_id"].unique().tolist()) == ["GLOBAL-C", "SOURCE-A"]
    assert result.obs["split_1"].eq("test").all()


def test_global_cartesian_broad_only_filters_registry_candidates(tmp_path) -> None:
    ad = __import__("anndata")
    source = ad.AnnData(
        X=np.zeros((1, 978), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "pert_id": ["SOURCE-A"],
                "pert_idx": [1],
                "cell_iname": ["CTX-A"],
                "cell_idx": [0],
                "tissue_idx": [0],
                "pert_dose": [10.0],
                "pert_time": [6.0],
            }
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(978)]),
    )
    source.obsm["X_ctl"] = np.ones((1, 978), dtype=np.float32)
    source_path = tmp_path / "source.h5ad"
    source.write_h5ad(source_path)
    registry = {
        "format": "xpert_drug_registry_v1",
        "drugs": [
            {"pert_id": "SOURCE-A", "pert_idx": 1, "unimol": True, "hg": True, "kpgt": True, "broad_identity": True, "broad_inference_eligible": True},
            {"pert_id": "GLOBAL-C", "pert_idx": 3, "unimol": True, "hg": True, "kpgt": True, "broad_identity": False, "broad_inference_eligible": False},
        ],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(__import__("json").dumps(registry), encoding="utf-8")
    output_path = tmp_path / "global.h5ad"
    audit = build_global_cartesian_adapter_h5ad(
        source_path=source_path,
        registry_path=registry_path,
        output_path=output_path,
        broad_only=True,
    )
    assert audit["drug_ids"] == ["SOURCE-A"]
