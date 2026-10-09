import hashlib
import json

from scripts.data.audit_exp045_source_index_binding import build_receipt


def test_binding_hashes_opaque_source_bytes_and_compares_prior_receipt(tmp_path):
    root = tmp_path / "repo"
    source = root / "data/processed/exp011/index.npz"
    prior = root / "artifacts/EXP-036/asset_audit.json"
    script = root / "scripts/data/audit_exp045_source_index_binding.py"
    source.parent.mkdir(parents=True)
    prior.parent.mkdir(parents=True)
    script.parent.mkdir(parents=True)
    source_bytes = b"opaque archive bytes; do not parse"
    source.write_bytes(source_bytes)
    prior.write_text(json.dumps({"source_index_sha256": hashlib.sha256(source_bytes).hexdigest()}))
    script.write_text("audit code")

    receipt = build_receipt(source, prior, root=root, audit_script_path=script)

    assert receipt["status"] == "SOURCE_INDEX_BYTES_MATCH_PRIOR_SOURCE_ONLY_ASSET_RECEIPT"
    assert receipt["source_index"]["current_byte_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert receipt["scope"]["npz_archive_opened"] is False
    assert receipt["scope"]["npz_members_read"] == []
    assert receipt["historical_receipt"]["historical_source_path_or_code_hash_reconstructed"] is False


def test_binding_reports_prior_digest_mismatch(tmp_path):
    root = tmp_path / "repo"
    source = root / "index.npz"
    prior = root / "asset_audit.json"
    script = root / "audit.py"
    root.mkdir(parents=True)
    source.write_bytes(b"current"); prior.write_text('{"source_index_sha256":"wrong"}'); script.write_text("code")

    receipt = build_receipt(source, prior, root=root, audit_script_path=script)

    assert receipt["status"] == "SOURCE_INDEX_BYTES_DIFFER_FROM_PRIOR_SOURCE_ONLY_ASSET_RECEIPT"
    assert receipt["historical_receipt"]["current_source_index_sha256_matches"] is False
