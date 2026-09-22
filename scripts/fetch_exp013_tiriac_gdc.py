"""List and optionally download the Tiriac GDC open RNA-seq bucket."""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from drug_screen.data.root import data_root

BUCKET = "gdc-organoid-pancreatic-phs001611-2-open"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_objects(max_pages: int = 50) -> list[dict]:
    objects: list[dict] = []
    token = None
    for _ in range(max_pages):
        query = {"list-type": "2", "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        url = f"https://{BUCKET}.s3.amazonaws.com/?" + urllib.parse.urlencode(query)
        request = urllib.request.Request(url, headers={"User-Agent": "DrugScreenLab/EXP-013"})
        with urllib.request.urlopen(request, timeout=60) as response:
            xml = response.read()
        root = ET.fromstring(xml)
        for node in root.findall("s3:Contents", NS):
            key = node.findtext("s3:Key", default="", namespaces=NS)
            size = int(node.findtext("s3:Size", default="0", namespaces=NS))
            objects.append({"key": key, "bytes": size})
        if root.findtext("s3:IsTruncated", default="false", namespaces=NS) != "true":
            break
        token = root.findtext("s3:NextContinuationToken", default="", namespaces=NS)
        if not token:
            break
    return objects


def main() -> int:
    root = data_root()
    dest_dir = root / "raw/exp013/tiriac_2018/gdc_open"
    dest_dir.mkdir(parents=True, exist_ok=True)
    objects = list_objects()
    fpkm = [row for row in objects if "FPKM" in row["key"].upper() or row["key"].endswith(".gz")]
    manifest = {
        "bucket": BUCKET,
        "n_objects": len(objects),
        "n_fpkm_like": len(fpkm),
        "total_bytes": sum(row["bytes"] for row in objects),
        "objects": objects[:200],
        "fpkm_like": fpkm[:80],
    }
    (root / "artifacts/EXP-013/tiriac_gdc_open_listing.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    # Download at most 8 small FPKM files to prove access without pulling the whole cohort blindly.
    downloaded = []
    for row in fpkm:
        if row["bytes"] == 0 or row["bytes"] > 8_000_000:
            continue
        name = Path(row["key"]).name
        dest = dest_dir / name
        if dest.exists():
            downloaded.append({"name": name, "status": "exists", "bytes": dest.stat().st_size})
            if len(downloaded) >= 8:
                break
            continue
        url = f"https://{BUCKET}.s3.amazonaws.com/{urllib.parse.quote(row['key'])}"
        request = urllib.request.Request(url, headers={"User-Agent": "DrugScreenLab/EXP-013"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        dest.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        downloaded.append({"name": name, "status": "downloaded", "bytes": len(payload), "sha256": digest, "key": row["key"]})
        if len(downloaded) >= 8:
            break
    manifest["downloaded_sample"] = downloaded
    (root / "artifacts/EXP-013/tiriac_gdc_open_listing.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"n_objects": len(objects), "n_fpkm_like": len(fpkm), "downloaded": len(downloaded)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
