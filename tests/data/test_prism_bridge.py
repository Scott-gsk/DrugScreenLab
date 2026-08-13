from __future__ import annotations

import pandas as pd

from drug_screen.data.prism_bridge import build_identity_bridge


def test_identity_bridge_prefers_exact_pert_id_and_does_not_need_response() -> None:
    prism = pd.DataFrame(
        [
            {"broad_id": "BRD-K00000001-001-01-0", "name": "x", "smiles": "CCO"},
            {"broad_id": "BRD-K00000002-001-01-0", "name": "y", "smiles": "CCN"},
        ]
    )
    lincs = pd.DataFrame(
        [
            {"pert_id": "BRD-K00000001", "pert_iname": "x", "pert_type": "trt_cp", "inchi_key": "", "canonical_smiles": "CCO"},
            {"pert_id": "BRD-K00000002", "pert_iname": "y", "pert_type": "trt_cp", "inchi_key": "", "canonical_smiles": "CCN"},
        ]
    )
    bridge, audit = build_identity_bridge(prism, lincs)
    assert bridge["match_method"].tolist() == ["exact_pert_id", "exact_pert_id"]
    assert audit["response_values_read"] is False
    assert audit["identity_matches"] == 2


def test_identity_bridge_allows_unique_alias_fallback() -> None:
    prism = pd.DataFrame([{"broad_id": "BRD-Z00000001-001-01-0", "name": "Alias Drug", "smiles": "CCO"}])
    lincs = pd.DataFrame(
        [{"pert_id": "LINCS_ONLY", "pert_iname": "Alias Drug", "pert_type": "trt_cp", "inchi_key": "-666", "canonical_smiles": "CCN"}]
    )
    bridge, audit = build_identity_bridge(prism, lincs)
    assert bridge.loc[0, "match_method"] == "exact_alias"
    assert audit["exact_alias_matches"] == 1
    assert audit["formal_eligible_identity_rows"] == 0
