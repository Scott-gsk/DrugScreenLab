"""Fetch the four organoid GEO accessions from NCBI FTP.

This is a live intake, not a metadata-only rewrite.  Failures stay
DATA_PARTIAL and are recorded; HTML/WAF bodies are never saved as matrices.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DEST = ROOT / "data/raw/geo"
AUDIT = ROOT / "artifacts/experiments/EXP-007/ORGANOID_GEO_FETCH.json"
UA = "DrugScreenLab-data-audit/1.0 (response-blind organoid intake)"
TIMEOUT = 180

TARGETS = {
    "GSE280506": {
        "bucket": "GSE280nnn",
        "files": [
            ("suppl", "GSE280506_filtered_feature_bc_matrix.h5"),
            ("suppl", "GSE280506_barcodes.tsv.gz"),
            ("suppl", "GSE280506_features.tsv.gz"),
            ("suppl", "GSE280506_cell_identities.csv.gz"),
            ("suppl", "GSE280506_CRISPRi_reference_guide_barcode_library.xlsx"),
            ("suppl", "GSE280506_CRISPRa_reference_guide_barcode_library.xlsx"),
            ("matrix", "GSE280506_series_matrix.txt.gz"),
            ("soft", "GSE280506_family.soft.gz"),
        ],
    },
    "GSE145308": {
        "bucket": "GSE145nnn",
        "files": [
            ("suppl", "GSE145308_RAW.tar"),
            ("suppl", "filelist.txt"),
            ("matrix", "GSE145308-GPL20301_series_matrix.txt.gz"),
            ("matrix", "GSE145308-GPL24676_series_matrix.txt.gz"),
            ("soft", "GSE145308_family.soft.gz"),
        ],
    },
    "GSE167285": {
        "bucket": "GSE167nnn",
        "files": [
            ("suppl", "GSE167285_Human_CRISPR_organoids_raw_counts.txt.gz"),
            ("suppl", "GSE167285_Human_CRISPR_organoids_normalized_counts.txt.gz"),
            ("matrix", "GSE167285_series_matrix.txt.gz"),
            ("soft", "GSE167285_family.soft.gz"),
        ],
    },
    "GSE241659": {
        "bucket": "GSE241nnn",
        "files": [
            ("suppl", "GSE241659_counts_normalised.tsv.gz"),
            ("matrix", "GSE241659_series_matrix.txt.gz"),
            ("soft", "GSE241659_family.soft.gz"),
        ],
    },
}


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like_html(path: Path) -> bool:
    with path.open("rb") as handle:
        head = handle.read(200).lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"aws waf" in head


def _url(accession: str, bucket: str, kind: str, filename: str) -> str:
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{bucket}/{accession}/{kind}/{filename}"


def fetch_one(url: str, dest: Path) -> dict[str, object]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            status = int(getattr(response, "status", 200) or 200)
            content_type = response.headers.get("Content-Type", "")
            tmp = dest.with_suffix(dest.suffix + ".part")
            with tmp.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        tmp.replace(dest)
    except urllib.error.HTTPError as error:
        return {
            "url": url,
            "path": str(dest.relative_to(ROOT).as_posix()),
            "ok": False,
            "status": int(error.code),
            "error": str(error),
        }
    except Exception as error:  # noqa: BLE001 - live fetch must record the real failure
        return {
            "url": url,
            "path": str(dest.relative_to(ROOT).as_posix()),
            "ok": False,
            "status": None,
            "error": f"{type(error).__name__}: {error}",
        }
    if _looks_like_html(dest):
        dest.unlink(missing_ok=True)
        return {
            "url": url,
            "path": str(dest.relative_to(ROOT).as_posix()),
            "ok": False,
            "status": status,
            "error": "HTML_OR_WAF_BODY_NOT_SAVED",
            "content_type": content_type,
        }
    return {
        "url": url,
        "path": str(dest.relative_to(ROOT).as_posix()),
        "ok": True,
        "status": status,
        "content_type": content_type,
        "bytes": dest.stat().st_size,
        "sha256": _sha256(dest),
    }


def build() -> dict[str, object]:
    datasets = []
    for accession, spec in TARGETS.items():
        files = []
        for kind, filename in spec["files"]:
            dest = DEST / accession / filename
            files.append(fetch_one(_url(accession, spec["bucket"], kind, filename), dest))
        ok_files = [row for row in files if row.get("ok")]
        datasets.append(
            {
                "accession": accession,
                "n_requested": len(files),
                "n_ok": len(ok_files),
                "local_matrix_present": any(
                    name in Path(str(row["path"])).name.lower()
                    for row in ok_files
                    for name in ("matrix", "counts", "raw.tar", ".h5")
                ),
                "files": files,
            }
        )
    payload = {
        "format": "organoid_geo_fetch_v1",
        "status": "FETCHED" if all(row["n_ok"] == row["n_requested"] for row in datasets) else "PARTIAL",
        "datasets": datasets,
        "user_agent": UA,
        "source": "NCBI GEO FTP",
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    payload = build()
    print(json.dumps({"status": payload["status"], "audit": str(AUDIT)}, sort_keys=True))
    return 0 if payload["status"] == "FETCHED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
