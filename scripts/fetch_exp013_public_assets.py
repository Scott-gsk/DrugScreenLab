"""Download public EXP-013 assets into new paths under data/raw/exp013."""

from __future__ import annotations

import hashlib
import json
import ssl
import urllib.request
from pathlib import Path

from drug_screen.data.root import data_root

USER_AGENT = "DrugScreenLab/EXP-013 (reproducible research fetch; Python urllib)"
TIMEOUT = 600

ASSETS = [
    {
        "key": "ctrp_v2_curves",
        "url": "https://ctd2-data.nci.nih.gov/Public/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset/v20.data.curves_post_qc.txt",
        "relative": "raw/exp013/ctrp_v2/v20.data.curves_post_qc.txt",
    },
    {
        "key": "ctrp_v2_compounds",
        "url": "https://ctd2-data.nci.nih.gov/Public/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset/v20.meta.per_compound.txt",
        "relative": "raw/exp013/ctrp_v2/v20.meta.per_compound.txt",
    },
    {
        "key": "ctrp_v2_cells",
        "url": "https://ctd2-data.nci.nih.gov/Public/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset/v20.meta.per_cell_line.txt",
        "relative": "raw/exp013/ctrp_v2/v20.meta.per_cell_line.txt",
    },
    {
        "key": "ctrp_v2_experiment",
        "url": "https://ctd2-data.nci.nih.gov/Public/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset/v20.meta.per_experiment.txt",
        "relative": "raw/exp013/ctrp_v2/v20.meta.per_experiment.txt",
    },
    {
        "key": "ctrp_v2_zip",
        "url": "https://ctd2-data.nci.nih.gov/Public/Broad/CTRPv2.0_2015_ctd2_ExpandedDataset/CTRPv2.0_2015_ctd2_ExpandedDataset.zip",
        "relative": "raw/exp013/ctrp_v2/CTRPv2.0_2015_ctd2_ExpandedDataset.zip",
    },
    {
        "key": "ctrp_v2_pharmacoset_rds",
        "url": "https://zenodo.org/records/3905470/files/CTRPv2.rds?download=1",
        "relative": "raw/exp013/ctrp_v2/CTRPv2.rds",
    },
    {
        "key": "paad_ncomms_supp10",
        "url": "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-57520-3/MediaObjects/41467_2025_57520_MOESM13_ESM.xlsx",
        "relative": "raw/exp013/paad_ncomms_2025/41467_2025_57520_MOESM13_ESM.xlsx",
    },
    {
        "key": "tiriac_gse114845_soft",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE114nnn/GSE114845/soft/GSE114845_family.soft.gz",
        "relative": "raw/exp013/tiriac_2018/GSE114845_family.soft.gz",
    },
    {
        "key": "tiriac_gse114845_miniml",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE114nnn/GSE114845/miniml/GSE114845_family.xml.tgz",
        "relative": "raw/exp013/tiriac_2018/GSE114845_family.xml.tgz",
    },
    {
        "key": "msigdb_hallmark_symbols",
        "url": "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.2.Hs/h.all.v2023.2.Hs.symbols.gmt",
        "relative": "raw/exp013/msigdb/h.all.v2023.2.Hs.symbols.gmt",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return {
            "status": "exists",
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT, context=context) as response:
            payload = response.read()
            info = dict(response.headers)
            final_url = response.geturl()
    except Exception as exc:  # noqa: BLE001 - record fetch failures for the audit
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    if len(payload) < 64:
        return {"status": "error", "error": "payload too small", "bytes": len(payload)}
    content_type = str(info.get("Content-Type", ""))
    if "text/html" in content_type and not destination.name.endswith(".html"):
        return {
            "status": "error",
            "error": f"HTML interstitial instead of file ({content_type})",
            "final_url": final_url,
            "bytes": len(payload),
        }
    destination.write_bytes(payload)
    return {
        "status": "downloaded",
        "bytes": len(payload),
        "sha256": sha256_file(destination),
        "final_url": final_url,
        "content_type": content_type,
    }


def main() -> int:
    root = data_root()
    log = []
    for asset in ASSETS:
        destination = root / asset["relative"]
        result = {"key": asset["key"], "url": asset["url"], "path": asset["relative"]}
        result.update(fetch(asset["url"], destination))
        log.append(result)
        print(json.dumps(result, ensure_ascii=False))
    out = root / "artifacts" / "EXP-013" / "public_fetch_log.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(out), "n": len(log)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
