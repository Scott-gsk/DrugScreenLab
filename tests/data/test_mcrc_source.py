import json

import numpy as np
import pandas as pd
import pytest

from drug_screen.data.mcrc_source import build_sample_metadata, rebuild_dss, rebuild_log2cpm, resolve_isomeric_structures


def test_log2cpm_sums_duplicate_hgnc_before_full_library_normalization():
    raw = pd.DataFrame({
        "hgnc_symbol": ["A", "a", "B", None], "Pt1_t1": [2, 3, 5, 99], "Pt2_t2": [5, 5, 10, 99],
    })
    expression, samples, genes = rebuild_log2cpm(raw)
    assert samples == ("Pt1_t1", "Pt2_t2")
    assert genes == ("A", "B")
    np.testing.assert_allclose(expression, [[np.log2(500001), np.log2(500001)], [np.log2(500001), np.log2(500001)]])


def test_dss_medians_library_replicates_and_records_combination_and_missing_mapping():
    raw = pd.DataFrame({"sample_id": ["Pt1_t1", "Pt2_t1"], "Drug_lib1": [1.0, 3.0], "Drug_lib2": [5.0, 7.0], "TAS-102_lib1": [2.0, 2.0], "run_id_lib1": [1, 2], "Volasertib.SN38_lib2": [2.0, 3.0]})
    structures = {"Drug": {"drug_id": "KEY"}}
    pairs, exclusions = rebuild_dss(raw, allowed_names={"Drug"}, structures=structures)
    assert pairs.response.tolist() == [3.0, 5.0]
    assert pairs.higher_is_sensitive.tolist() == [True, True]
    assert exclusions == [
        {"drug_name": "TAS-102", "reason": "combination_or_biologic"},
        {"drug_name": "Volasertib.SN38", "reason": "combination_or_biologic"},
        {"drug_name": "run_id", "reason": "source_metadata_column"},
    ]
    pairs, exclusions = rebuild_dss(raw[["sample_id", "Drug_lib1"]], allowed_names={"Drug"}, structures={})
    assert pairs.empty
    assert exclusions == [{"drug_name": "Drug", "reason": "quarantined_fixed_structure"}]


def test_isomeric_structure_full_identity_conflict_is_rejected(tmp_path):
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps([{"primary_name": f"D{i}", "pubchem_cid": i} for i in range(1, 50)]), encoding="utf-8")
    fixed = tmp_path / "fixed.json"
    fixed.write_text(json.dumps({"PropertyTable": {"Properties": [
        {"CID": i, "SMILES": "C[C@H](O)F", "InChIKey": "WRONG"} for i in range(1, 50)
    ]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="full InChIKey mismatch"):
        resolve_isomeric_structures(legacy, fixed)


def test_s1_metadata_retains_expression_samples_and_marks_unverified_prefixes_pending_g0():
    raw = pd.DataFrame({
        "sample_id": ["Pt1_t1", "Pt1_t1", "Pt2_t1", "Pt3_t1"],
        "patient": ["Pt1", "Pt1", "Pt2", "Pt3"],
        "sample_type": ["PDO", "CRLM", "CRLM", "PDO"],
    })
    samples, exclusions = build_sample_metadata(raw, ("Pt1_t1", "Pt2_t1", "Pt4_t1"))
    assert samples.to_dict(orient="records")[0]["patient_id"] == "Pt1"
    assert samples.to_dict(orient="records")[0]["source_sample_types"] == "CRLM;PDO"
    assert samples.to_dict(orient="records")[1]["patient_id_provenance"] == "source_sample_prefix_not_independently_verified"
    assert samples.to_dict(orient="records")[2]["identity_status"] == "pending_G0_identity_verification"
    assert exclusions == [
        {"sample_id": "Pt2_t1", "reason": "no_PDO_mapping_in_S1", "sample_types": ["CRLM"]},
        {"sample_id": "Pt4_t1", "reason": "missing_S1_mutation_record"},
    ]


def test_s1_patient_prefix_conflict_fails_instead_of_relabeling_the_sample():
    raw = pd.DataFrame({"sample_id": ["Pt1_t1"], "patient": ["Pt2"], "sample_type": ["PDO"]})
    with pytest.raises(ValueError, match="conflicts with sample prefix"):
        build_sample_metadata(raw, ("Pt1_t1",))
