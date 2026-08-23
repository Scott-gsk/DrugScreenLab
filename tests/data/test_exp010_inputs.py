from pathlib import Path

import pandas as pd

from drug_screen.data.exp010_inputs import _patient_id, fetch_pubchem


def test_patient_identity_preserves_subpatient() -> None:
    assert _patient_id("Pt6.2_t2") == "Pt6.2"


def test_pubchem_empty_is_auditable(tmp_path: Path) -> None:
    frame, summary = fetch_pubchem([], tmp_path / "cache.json")
    assert frame.empty
    assert summary == {"drugs": 0, "mapped": 0, "mapping_rate": 0.0}
