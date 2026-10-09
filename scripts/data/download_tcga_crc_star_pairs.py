"""Download only the frozen open GDC STAR-count pairs, verifying every source MD5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path


UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def hash_file(path: Path) -> tuple[str, str, int]:
    md5, sha256, size = hashlib.md5(), hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
            size += len(block)
    return md5.hexdigest(), sha256.hexdigest(), size


def download(pairs_path: Path, raw_dir: Path, *, limit: int | None = None) -> dict:
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    files = [part for pair in pairs for part in (pair["tumor"], pair["normal"])]
    if len(files) != 100 or len({part["file_id"] for part in files}) != 100:
        raise ValueError("frozen TCGA pair manifest does not contain 100 unique files")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    (raw_dir / "files").mkdir(parents=True, exist_ok=True)
    receipts = []
    for number, part in enumerate(files[:limit] if limit else files, start=1):
        file_id = part["file_id"]
        if not UUID.fullmatch(file_id):
            raise ValueError("GDC file ID is not a UUID")
        destination = raw_dir / "files" / f"{file_id}.tsv"
        partial = raw_dir / "files" / f"{file_id}.tsv.partial"
        if destination.exists():
            md5, sha256, size = hash_file(destination)
            if md5 != part["md5sum"] or size != part["file_size"]:
                raise ValueError(f"existing raw GDC file has incorrect checksum: {file_id}")
            status = "VERIFIED_EXISTING"
        else:
            if partial.exists():
                raise FileExistsError(f"incomplete partial file needs explicit review: {partial}")
            request = urllib.request.Request(
                f"https://api.gdc.cancer.gov/data/{file_id}",
                headers={"User-Agent": "DrugScreenLab/EXP-038"})
            md5_hash, sha_hash, size = hashlib.md5(), hashlib.sha256(), 0
            try:
                with urllib.request.urlopen(request, timeout=120) as response, partial.open("xb") as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                        md5_hash.update(block)
                        sha_hash.update(block)
                        size += len(block)
            except Exception:
                raise RuntimeError(f"GDC transfer interrupted; partial retained for review: {partial}")
            md5, sha256 = md5_hash.hexdigest(), sha_hash.hexdigest()
            if md5 != part["md5sum"] or size != part["file_size"]:
                raise ValueError(f"downloaded GDC file failed source MD5/size validation: {file_id}")
            os.replace(partial, destination)
            status = "DOWNLOADED_AND_VERIFIED"
        receipts.append({"case_id": part["case_id"], "sample_id": part["sample_id"],
                         "file_id": file_id, "file_name": part["file_name"],
                         "relative_path": f"files/{file_id}.tsv", "md5": md5,
                         "sha256": sha256, "size": size, "status": status})
        print(json.dumps({"number": number, "total": len(files), "file_id": file_id,
                          "status": status}), flush=True)
    result = {"experiment_id": "EXP-038",
              "status": "RAW_GDC_FILES_VERIFIED" if len(receipts) == len(files) else "ENGINEERING_PARTIAL_DOWNLOAD",
              "n_files": len(receipts), "n_expected_files": len(files),
              "selected_pairs_sha256": hash_file(pairs_path)[1], "files": receipts}
    if limit is None:
        manifest = raw_dir / "manifest.json"
        if manifest.exists():
            raise FileExistsError(f"raw GDC manifest already exists: {manifest}")
        temporary = raw_dir / "manifest.json.tmp"
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, manifest)
    return {key: value for key, value in result.items() if key != "files"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(download(args.pairs, args.raw_dir, limit=args.limit), ensure_ascii=False))


if __name__ == "__main__":
    main()
