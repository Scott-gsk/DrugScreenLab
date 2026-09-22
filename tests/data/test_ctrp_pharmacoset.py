from __future__ import annotations
import gzip
from pathlib import Path
from drug_screen.data.ctrp_pharmacoset import FIELD_MARKERS, SLOT_MARKERS, find_gzip_markers, sha256_file


def test_marker_scan_requires_complete_identifier(tmp_path: Path):
    source = tmp_path / "fixture.rds"
    source.write_bytes(gzip.compress(b"prefix PharmacoSet sensitivityInfo cellid cpd_smiles suffix"))
    assert find_gzip_markers(source, SLOT_MARKERS) == ["PharmacoSet", "sensitivityInfo"]
    assert find_gzip_markers(source, FIELD_MARKERS) == ["cellid", "cpd_smiles"]


def test_sha256_helper_is_content_addressed(tmp_path: Path):
    source = tmp_path / "fixture.rds"
    source.write_bytes(b"fixture")
    assert sha256_file(source) == "f16d05ec6b29248d2c61adb1e9263f78e4f7bace1b955014a2d17872cfe4064d"
