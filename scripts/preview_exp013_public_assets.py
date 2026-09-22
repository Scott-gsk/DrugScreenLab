"""Summarize downloaded EXP-013 public files without training."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from drug_screen.data.root import data_root


def main() -> int:
    root = data_root()
    report: dict[str, object] = {}
    xlsx = root / "raw/exp013/paad_ncomms_2025/41467_2025_57520_MOESM13_ESM.xlsx"
    if xlsx.exists():
        book = pd.ExcelFile(xlsx)
        sheets = {}
        for name in book.sheet_names:
            frame = pd.read_excel(xlsx, sheet_name=name, nrows=5)
            sheets[name] = {
                "columns": [str(col) for col in frame.columns],
                "preview_rows": int(len(frame)),
            }
        full_counts = {
            name: int(len(pd.read_excel(xlsx, sheet_name=name)))
            for name in book.sheet_names
        }
        report["paad_supp10"] = {"sheets": list(book.sheet_names), "preview": sheets, "n_rows": full_counts}
    soft = root / "raw/exp013/tiriac_2018/GSE114845_family.soft.gz"
    if soft.exists():
        text = gzip.open(soft, "rt", errors="replace").read(4000)
        report["tiriac_soft_head"] = text[:1500]
        report["tiriac_soft_bytes"] = soft.stat().st_size
    rds = root / "raw/exp013/ctrp_v2/CTRPv2.rds"
    if rds.exists():
        report["ctrp_rds_bytes"] = rds.stat().st_size
    out = root / "artifacts/EXP-013/public_asset_preview.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "tiriac_soft_head"}, indent=2)[:4000])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
