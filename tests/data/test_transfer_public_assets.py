from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import pytest

from drug_screen.data.public_assets import AssetError, fetch_asset, load_manifest, run_manifest


class Response:
    def __init__(self, body: bytes, headers: dict[str, str], status: int = 200):
        self._body = BytesIO(body)
        self.headers = headers
        self.status = status

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def getcode(self) -> int:
        return self.status

    def geturl(self) -> str:
        return "https://official.example/final"

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class ReadFailureResponse(Response):
    def __init__(self, body: bytes, headers: dict[str, str], status: int = 200):
        super().__init__(body, headers, status)
        self._reads = 0

    def read(self, size: int = -1) -> bytes:
        self._reads += 1
        if self._reads > 1:
            raise OSError("connection dropped")
        return super().read(3)


def asset(**extra):
    value = {"key": "small", "url": "https://official.example/file", "relative_path": "raw/transfer/a.txt", "max_bytes": 100}
    value.update(extra)
    return value


def test_fetch_finalizes_atomically_and_is_idempotent(tmp_path: Path):
    payload = b"official public data"
    calls = []

    def opener(request, **_kwargs):
        calls.append(dict(request.header_items()))
        return Response(payload, {"ETag": '"v1"', "Content-Length": str(len(payload))})

    first = fetch_asset(asset(sha256=hashlib.sha256(payload).hexdigest()), tmp_path, opener=opener)
    second = fetch_asset(asset(sha256=hashlib.sha256(payload).hexdigest()), tmp_path, opener=opener)
    assert first["status"] == "downloaded"
    assert second["status"] == "exists"
    assert len(calls) == 1
    assert not (tmp_path / "raw/transfer/a.txt.part").exists()


def test_resume_requires_matching_range_and_validator(tmp_path: Path):
    destination = tmp_path / "raw/transfer/a.txt"
    destination.parent.mkdir(parents=True)
    part = destination.with_name("a.txt.part")
    part.write_bytes(b"abc")
    part.with_name("a.txt.part.json").write_text(json.dumps({"url": "https://official.example/file", "validators": {"etag": '"v1"'}}))
    requests = []

    def opener(request, **_kwargs):
        requests.append(dict(request.header_items()))
        return Response(b"def", {"ETag": '"v1"', "Content-Range": "bytes 3-5/6", "Content-Length": "3"}, 206)

    result = fetch_asset(asset(bytes=6), tmp_path, opener=opener)
    assert result["sha256"] == hashlib.sha256(b"abcdef").hexdigest()
    lowered = {key.lower(): value for key, value in requests[0].items()}
    assert lowered["range"] == "bytes=3-"
    assert lowered["if-range"] == '"v1"'


def test_mismatch_never_creates_final_file(tmp_path: Path):
    def opener(_request, **_kwargs):
        return Response(b"wrong", {"ETag": '"v1"', "Content-Length": "5"})

    with pytest.raises(AssetError, match="sha256 mismatch"):
        fetch_asset(asset(sha256="0" * 64), tmp_path, opener=opener)
    assert not (tmp_path / "raw/transfer/a.txt").exists()


def test_complete_initial_transfer_without_validator_is_allowed(tmp_path: Path):
    def opener(_request, **_kwargs):
        return Response(b"complete", {"Content-Length": "8"})

    result = fetch_asset(asset(bytes=8), tmp_path, opener=opener)
    assert result["status"] == "downloaded"
    assert result["validators"] == {}


def test_truncated_content_length_never_finalizes(tmp_path: Path):
    def opener(_request, **_kwargs):
        return Response(b"short", {"ETag": '"v1"', "Content-Length": "6"})

    with pytest.raises(AssetError, match="truncated response"):
        fetch_asset(asset(), tmp_path, opener=opener)
    assert not (tmp_path / "raw/transfer/a.txt").exists()


def test_changed_etag_refuses_resume_and_preserves_final_absence(tmp_path: Path):
    destination = tmp_path / "raw/transfer/a.txt"
    destination.parent.mkdir(parents=True)
    part = destination.with_name("a.txt.part")
    part.write_bytes(b"abc")
    part.with_name("a.txt.part.json").write_text(
        json.dumps({"url": "https://official.example/file", "validators": {"etag": '"v1"'}})
    )

    def opener(_request, **_kwargs):
        return Response(b"def", {"ETag": '"v2"', "Content-Range": "bytes 3-5/6", "Content-Length": "3"}, 206)

    with pytest.raises(AssetError, match="validator changed"):
        fetch_asset(asset(bytes=6), tmp_path, opener=opener)
    assert part.read_bytes() == b"abc"
    assert not destination.exists()


def test_partial_metadata_exists_after_read_failure(tmp_path: Path):
    def opener(_request, **_kwargs):
        return ReadFailureResponse(b"abcdef", {"ETag": '"v1"', "Content-Length": "6"})

    with pytest.raises(OSError, match="connection dropped"):
        fetch_asset(asset(), tmp_path, opener=opener)
    part = tmp_path / "raw/transfer/a.txt.part"
    metadata = json.loads(part.with_name("a.txt.part.json").read_text())
    assert part.read_bytes() == b"abc"
    assert metadata["url"] == "https://official.example/file"
    assert metadata["validators"] == {"etag": '"v1"'}


def test_existing_final_is_verified_not_overwritten(tmp_path: Path):
    final = tmp_path / "raw/transfer/a.txt"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"original")

    def opener(_request, **_kwargs):
        raise AssertionError("network must not be called for an existing final")

    with pytest.raises(AssetError, match="sha256 mismatch"):
        fetch_asset(asset(sha256="0" * 64), tmp_path, opener=opener)
    assert final.read_bytes() == b"original"


def test_manifest_rejects_traversal_and_pending_is_reported(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"assets": [asset(relative_path="../outside"), {"key": "pending", "status": "pending", "reason": "controlled"}]}))
    with pytest.raises(AssetError, match="unsafe relative path"):
        load_manifest(manifest_path)
    results = run_manifest({"assets": [{"key": "pending", "status": "pending", "reason": "controlled"}]}, tmp_path, "dry-run")
    assert results[0]["status"] == "pending"
