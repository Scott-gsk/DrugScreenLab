import json
from pathlib import Path

from drug_screen.data.registry import load_registry, validate_registry

def test_registry_is_valid():
    assert validate_registry() == []


def test_exact978_cache_is_a_versioned_provenance_complete_processed_asset():
    entries = {entry["id"]: entry for entry in load_registry()}
    asset = entries["lincs_gse92742_exact978_cache_v1"]
    assert asset["asset_type"] == "processed"
    assert asset["checksum"]["algorithm"] == "sha256"
    assert asset["schema"]["matrix_shape"] == [1319138, 978]
    assert asset["schema"]["ordered_gene_ids_sha256"]
    assert asset["source"]["source_level3_sha512"]
    assert asset["preprocessing_contract"]["sha256"]
    assert asset["provenance"]["runner_sha256"]
    manifest_path = Path("data") / asset["path"]["relative"] / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["cache_sha256"] == asset["checksum"]["value"]
    assert manifest["ordered_gene_ids_sha256"] == asset["schema"]["ordered_gene_ids_sha256"]
