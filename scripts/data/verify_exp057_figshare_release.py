"""Verify the official XPert Figshare v1 archive and selectively extract approved files.

The archive-level publisher MD5 is checked over the original bytes. The ZIP
central directory is inventoried without opening member payloads. Extraction
is allowlisted and reads only the explicitly requested PANACEA/metadata files;
this avoids materializing the unrelated CDSDB response matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable


FILE_ID = "58181296"
ARTICLE_ID = "28955141"
VERSION = 1
EXPECTED_SIZE = 1_579_752_267
EXPECTED_MD5 = "4a58bc01b42c7909e7adc2ddd40b30ac"
OFFICIAL_URL = "https://ndownloader.figshare.com/files/58181296"
LICENSE = "CC BY 4.0"
REQUIRED_PANACEA_BASENAME = "PANACEA_sdst_352.h5ad"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_hashes(path: Path) -> dict[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def validate_member_name(info: zipfile.ZipInfo) -> None:
    name = info.filename
    path = PurePosixPath(name)
    mode = info.external_attr >> 16
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or ".." in path.parts
        or (path.parts and ":" in path.parts[0])
    ):
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    if stat.S_ISLNK(mode):
        raise ValueError(f"symlink ZIP member is not allowed: {name!r}")
    if info.header_offset < 0 or info.header_offset >= EXPECTED_SIZE:
        raise ValueError(f"invalid local-header offset for ZIP member: {name!r}")


def inventory_archive(archive: Path, hashes: dict[str, str]) -> list[zipfile.ZipInfo]:
    if archive.stat().st_size != EXPECTED_SIZE:
        raise ValueError(
            f"archive size mismatch: expected {EXPECTED_SIZE}, got {archive.stat().st_size}"
        )
    if hashes["md5"] != EXPECTED_MD5:
        raise ValueError(f"official Figshare MD5 mismatch: {hashes['md5']}")
    with zipfile.ZipFile(archive, "r") as zf:
        infos = zf.infolist()
        if not infos:
            raise ValueError("ZIP central directory contains no members")
        names: set[str] = set()
        for info in infos:
            validate_member_name(info)
            if info.filename in names:
                raise ValueError(f"duplicate ZIP member name: {info.filename!r}")
            names.add(info.filename)
        panacea = [i.filename for i in infos if PurePosixPath(i.filename).name == REQUIRED_PANACEA_BASENAME]
        if len(panacea) != 1:
            raise ValueError(f"expected one PANACEA H5AD member, found {panacea}")
    return infos


def extract_selected(archive: Path, out_dir: Path, member_names: Iterable[str]) -> list[dict[str, object]]:
    requested = list(dict.fromkeys(member_names))
    if not requested:
        raise ValueError("--extract-member must name at least one exact archive member")
    extracted: list[dict[str, object]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir.resolve()
    with zipfile.ZipFile(archive, "r") as zf:
        by_name = {i.filename: i for i in zf.infolist()}
        for name in requested:
            if name not in by_name:
                raise ValueError(f"requested member is not present in official archive: {name!r}")
            info = by_name[name]
            validate_member_name(info)
            if "cdsdb" in name.lower() or "cdsdb" in PurePosixPath(name).name.lower():
                raise ValueError("CDSDB members are outside the approved extraction scope")
            dest = (out_dir / Path(*PurePosixPath(name).parts)).resolve()
            if root not in dest.parents:
                raise ValueError(f"extraction path escapes output root: {name!r}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise FileExistsError(f"refusing to overwrite extracted file: {dest}")
            temp = dest.with_name(dest.name + ".part")
            if temp.exists():
                raise FileExistsError(f"refusing to overwrite partial extraction: {temp}")
            copied = 0
            try:
                with zf.open(info, "r") as src, temp.open("xb") as dst:
                    while True:
                        block = src.read(8 * 1024 * 1024)
                        if not block:
                            break
                        dst.write(block)
                        copied += len(block)
                os.replace(temp, dest)
            except Exception:
                if temp.exists():
                    temp.unlink()
                raise
            extracted.append({
                "archive_member": name,
                "path": str(dest),
                "bytes": copied,
                "crc32_verified_by_complete_member_read": True,
                "sha256": sha256_file(dest),
            })
    return extracted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--extract-member", action="append", default=[])
    args = parser.parse_args()
    started = time.monotonic()
    archive = args.archive.resolve()
    out_root = args.output_root.resolve()
    hashes = file_hashes(archive)
    infos = inventory_archive(archive, hashes)
    members = [
        {
            "name": i.filename,
            "compressed_bytes": i.compress_size,
            "uncompressed_bytes": i.file_size,
            "compression": i.compress_type,
            "crc32": f"{i.CRC:08x}",
        }
        for i in infos
    ]
    panacea_members = [i.filename for i in infos if PurePosixPath(i.filename).name == REQUIRED_PANACEA_BASENAME]
    extracted = extract_selected(archive, out_root / "extracted", args.extract_member) if args.extract_member else []
    manifest = {
        "experiment_id": "EXP-057",
        "run_id": "20261007_exp057_panacea_external_development_v6",
        "release": {
            "publisher": "Figshare",
            "article_id": ARTICLE_ID,
            "version": VERSION,
            "file_id": FILE_ID,
            "download_url": OFFICIAL_URL,
            "license": LICENSE,
            "expected_bytes": EXPECTED_SIZE,
            "observed_bytes": archive.stat().st_size,
            "expected_md5": EXPECTED_MD5,
            "observed_md5": hashes["md5"],
            "observed_sha256": hashes["sha256"],
            "file_size_and_publisher_md5_pass": archive.stat().st_size == EXPECTED_SIZE and hashes["md5"] == EXPECTED_MD5,
        },
        "archive": {
            "path": str(archive),
            "zip_central_directory_opened": True,
            "member_count": len(members),
            "duplicate_names": False,
            "all_member_paths_safe": True,
            "all_payloads_read_for_global_crc": False,
            "global_integrity_basis": "complete archive byte stream matches the Figshare v1 publisher MD5; only approved extracted members are decompressed and CRC-checked",
            "panacea_h5ad_members": panacea_members,
            "member_inventory": members,
        },
        "extracted_members": extracted,
        "cdsdb_payload_read_or_extracted": False,
        "pdo_outer_locked_values_accessed": False,
        "elapsed_wall_seconds": time.monotonic() - started,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    target = out_root / ("figshare_extraction_manifest.json" if extracted else "figshare_release_manifest.json")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite release manifest: {target}")
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "archive": str(archive),
        "bytes": archive.stat().st_size,
        "md5": hashes["md5"],
        "sha256": hashes["sha256"],
        "member_count": len(members),
        "panacea_h5ad_members": panacea_members,
        "extracted_members": extracted,
        "manifest": str(target),
        "elapsed_wall_seconds": manifest["elapsed_wall_seconds"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
