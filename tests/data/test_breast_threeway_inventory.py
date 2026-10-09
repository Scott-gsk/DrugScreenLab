from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/data/breast_threeway_inventory.py"
SPEC = importlib.util.spec_from_file_location("breast_threeway_inventory", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_full_inchikey_contract_rejects_partial_and_nonchemical_keys() -> None:
    assert module.is_full_inchikey("BSYNRYMUTXBXSQ-UHFFFAOYSA-N")
    assert not module.is_full_inchikey("BSYNRYMUTXBXSQ")
    assert not module.is_full_inchikey("NA")
    assert not module.is_full_inchikey("BSYNRYMUTXBXSQ-UHFFFAOYSA")


def test_exact_structure_map_excludes_ambiguous_ids_and_bad_keys() -> None:
    source = pd.DataFrame(
        {
            "drugid": ["d1", "d1", "d2", "d3", "d3"],
            "inchikey": [
                "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                "not-a-key",
                "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                "AAAAAAAAAAAAAA-BBBBBBBBBB-C",
            ],
        }
    )
    mapped, detail = module._make_exact_structure_map(source, "drugid", "inchikey")
    assert mapped.to_dict("records") == [
        {"drugid": "d1", "full_inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"}
    ]
    status = detail.groupby("drugid").identity_status.agg(set).to_dict()
    assert status["d1"] == {"verified_full_inchikey"}
    assert status["d2"] == {"missing_or_invalid_full_inchikey"}
    assert status["d3"] == {"ambiguous_id_to_multiple_inchikeys"}


def test_stable_id_mapping_is_punctuation_insensitive_but_not_many_to_one() -> None:
    source = pd.DataFrame(
        {
            "rrid": ["CVCL_0031", "CVCL-0031", "CVCL_9999"],
            "model_id": ["M1", "M1", "M2"],
        }
    )
    mapping, ambiguous = module._unique_stable_id_map(source, "rrid", "model_id")
    assert mapping == {"CVCL0031": "M1", "CVCL9999": "M2"}
    assert ambiguous == set()

    conflicting = pd.concat(
        [source, pd.DataFrame({"rrid": ["CVCL.0031"], "model_id": ["M3"]})],
        ignore_index=True,
    )
    mapping, ambiguous = module._unique_stable_id_map(conflicting, "rrid", "model_id")
    assert "CVCL0031" not in mapping
    assert ambiguous == {"CVCL0031"}
