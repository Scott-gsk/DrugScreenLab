from pathlib import Path

from drug_screen.data.registry import load_registry, validate_registry


def test_exp013_public_assets_are_registered_and_present():
    errors = validate_registry()
    assert errors == []
    by_id = {row["id"]: row for row in load_registry()}
    for asset_id in (
        "paad_ncomms_2025_supp10_auc_raw_v1",
        "ctrp_v2_pharmacoset_zenodo_3905470_v1",
        "msigdb_hallmark_2023_2_hs_gmt_raw_v1",
    ):
        assert asset_id in by_id
        relative = by_id[asset_id]["path"]["relative"]
        path = Path("data") / relative
        assert path.exists()
        assert path.stat().st_size == by_id[asset_id]["checksum"]["bytes"]
