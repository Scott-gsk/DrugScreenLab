from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SIGNATURE = ROOT / "mvp/core_data/crc_disease_signature_exact978.tsv"
EXPECTED_SIGNATURE_SHA256 = "61e95b6a6da1d4c8b91ed1a99e96471027917c7e7e2501b507875ce28cd310c3"


def test_gse74602_signature_is_tumor_higher_not_inverted() -> None:
    assert SIGNATURE.exists()
    digest = sha256(SIGNATURE.read_bytes()).hexdigest()
    assert digest == EXPECTED_SIGNATURE_SHA256
    frame = pd.read_csv(SIGNATURE, sep="\t")
    signed = pd.to_numeric(frame["signed_log2fc"], errors="coerce")
    assert set(frame["comparison"].astype(str)) == {"tumor_vs_normal"}
    assert int((frame["direction"].astype(str) == "up").sum()) > int(
        (frame["direction"].astype(str) == "down").sum()
    )
    assert bool(((frame["direction"].astype(str) == "up") == (signed > 0)).all())
    myc = frame.loc[frame["gene_symbol"].astype(str).eq("MYC")]
    assert not myc.empty
    assert float(myc.iloc[0]["signed_log2fc"]) > 0
    assert str(myc.iloc[0]["direction"]) == "up"
