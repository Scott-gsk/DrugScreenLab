from __future__ import annotations

import json

import pandas as pd

from drug_screen.foundation.broad_prism import select_frozen_response_mapping


def test_mapping_joins_full_broad_id_and_filters_registry_eligibility(tmp_path) -> None:
    cohort = {
        "eligible_base_ids": ["BRD-A00000001", "BRD-A00000002"],
        "eligible_crc_depmap_ids": ["ACH-1"],
    }
    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(json.dumps(cohort), encoding="utf-8")
    bridge = pd.DataFrame(
        [
            {
                "prism_broad_id": "BRD-A00000001-001-01-0",
                "prism_broad_id_base": "BRD-A00000001",
                "lincs_pert_id": "BRD-A00000001",
                "lincs_pert_iname": "A",
                "match_status": "MATCHED_IDENTITY",
            },
            {
                "prism_broad_id": "BRD-A00000002-001-01-0",
                "prism_broad_id_base": "BRD-A00000002",
                "lincs_pert_id": "BRD-A00000002",
                "lincs_pert_iname": "B",
                "match_status": "MATCHED_IDENTITY",
            },
        ]
    )
    bridge_path = tmp_path / "bridge.csv"
    bridge.to_csv(bridge_path, index=False)
    treatment = pd.DataFrame(
        [
            {"broad_id": "BRD-A00000001-001-01-0", "name": "A", "column_name": "col-a", "dose": "2.5", "screen_id": "HTS"},
            {"broad_id": "BRD-A00000002-001-01-0", "name": "B", "column_name": "col-b", "dose": "2.5", "screen_id": "HTS"},
        ]
    )
    treatment_path = tmp_path / "treatment.csv"
    treatment.to_csv(treatment_path, index=False)
    registry = {
        "drugs": [
            {"pert_id": "BRD-A00000001", "broad_inference_eligible": True},
            {"pert_id": "BRD-A00000002", "broad_inference_eligible": False},
        ]
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    result = select_frozen_response_mapping(
        cohort_path=cohort_path,
        bridge_path=bridge_path,
        treatment_info_path=treatment_path,
        registry_path=registry_path,
    )

    assert result["column_name"].tolist() == ["col-a"]
    assert result["pert_id"].tolist() == ["BRD-A00000001"]


def test_mapping_deduplicates_identical_treatment_columns(tmp_path) -> None:
    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(json.dumps({"eligible_base_ids": ["BRD-A00000001"]}), encoding="utf-8")
    bridge_path = tmp_path / "bridge.csv"
    pd.DataFrame(
        [{
            "prism_broad_id": "BRD-A00000001-001-01-0",
            "prism_broad_id_base": "BRD-A00000001",
            "lincs_pert_id": "BRD-A00000001",
            "match_status": "MATCHED_IDENTITY",
        }]
    ).to_csv(bridge_path, index=False)
    treatment_path = tmp_path / "treatment.csv"
    pd.DataFrame(
        [
            {"broad_id": "BRD-A00000001-001-01-0", "name": "A", "column_name": "col-a"},
            {"broad_id": "BRD-A00000001-001-01-0", "name": "A", "column_name": "col-a"},
        ]
    ).to_csv(treatment_path, index=False)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"drugs": [{"pert_id": "BRD-A00000001", "broad_inference_eligible": True}]}), encoding="utf-8")

    result = select_frozen_response_mapping(
        cohort_path=cohort_path,
        bridge_path=bridge_path,
        treatment_info_path=treatment_path,
        registry_path=registry_path,
    )

    assert len(result) == 1
