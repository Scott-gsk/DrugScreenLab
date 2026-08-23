from pathlib import Path
import gzip
import io
import tarfile
import zipfile
from email.message import Message
from unittest.mock import patch

from drug_screen.data.exp010_public import (
    PublicAsset, SampleContext, archive_gate, download, freeze_manifest,
    ordered_gene_universe, patient_split_gate, sha256,
)


def test_sha256_and_manifest(tmp_path: Path) -> None:
    root = tmp_path / "external"
    target = root / "reactome" / "tiny.zip"
    target.parent.mkdir(parents=True)
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("x.gmt", "p\td\tA\n")
    asset = PublicAsset("Reactome", "tiny.zip", "https://invalid.example/tiny.zip", "v", "license", archive="zip")
    output = tmp_path / "manifest.json"
    manifest = freeze_manifest(root, output, [asset])
    assert manifest["assets"][0]["archive_ok"] is True
    assert manifest["assets"][0]["sha256"] == sha256(target)
    assert output.is_file()


def test_download_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "external"
    target = root / "reactome" / "tiny.zip"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"immutable")
    asset = PublicAsset("Reactome", "tiny.zip", "https://invalid.example/tiny.zip", "v", "license")
    with patch("urllib.request.urlopen", side_effect=OSError("offline")):
        assert download(asset, root).read_bytes() == b"immutable"


class _Response(io.BytesIO):
    def __init__(self, content: bytes, status: int, headers: dict[str, str]):
        super().__init__(content)
        self.status = status
        self.headers = Message()
        for key, value in headers.items():
            self.headers[key] = value
    def getcode(self) -> int:
        return self.status


def test_range_ignored_replaces_partial_instead_of_appending(tmp_path: Path) -> None:
    root = tmp_path / "external"
    partial = root / "reactome" / "tiny.zip.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"stale")
    data = b"complete"
    asset = PublicAsset("Reactome", "tiny.zip", "https://example/tiny.zip", "v", "license")
    replies = [_Response(b"", 200, {"Content-Length": str(len(data))}), _Response(data, 200, {})]
    with patch("urllib.request.urlopen", side_effect=replies):
        assert download(asset, root).read_bytes() == data


def test_gzip_gate(tmp_path: Path) -> None:
    path = tmp_path / "a.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(b"header\n")
    assert archive_gate(path, "gzip")["archive_ok"] is True


def test_gene_universe_is_ordered_and_hashed() -> None:
    genes, digest = ordered_gene_universe(["b", "A"], ["a", "B"], minimum=2)
    assert genes == ["A", "B"]
    assert len(digest) == 64


def test_patient_split_gate_rejects_region_leakage() -> None:
    contexts = [
        SampleContext("s1", "p1", "m1", "colon", "r1"),
        SampleContext("s2", "p1", "m1", "colon", "r1"),
    ]
    try:
        patient_split_gate(contexts, {"s1": 0, "s2": 1})
    except ValueError as error:
        assert "identity leakage" in str(error)
    else:
        raise AssertionError("leakage was accepted")


def test_tar_gate_detects_corrupt_nested_gzip(tmp_path: Path) -> None:
    nested = io.BytesIO()
    with tarfile.open(fileobj=nested, mode="w:gz") as inner:
        payload = b"abc"
        info = tarfile.TarInfo("x.txt")
        info.size = len(payload)
        inner.addfile(info, io.BytesIO(payload))
    truncated = nested.getvalue()[:-8]
    outer_path = tmp_path / "outer.tar"
    with tarfile.open(outer_path, "w") as outer:
        info = tarfile.TarInfo("inner.tar.gz")
        info.size = len(truncated)
        outer.addfile(info, io.BytesIO(truncated))
    result = archive_gate(outer_path, "tar")
    assert result["archive_ok"] is False
    assert result["corrupt_members"]
