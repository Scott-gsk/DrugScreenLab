#!/usr/bin/env python
"""Build the EXP-011 indexed cohort bundles without writing to raw assets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from drug_screen.data.exp011_materialize import build_exp011


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get("DRUGSCREEN_DATA_ROOT", "data")),
        help="immutable/public source data root; never hard-code a Windows path",
    )
    parser.add_argument(
        "--output", type=Path,
        default=None,
        help="derived output directory (default: <data-root>/processed/exp011)",
    )
    parser.add_argument(
        "--project-root", type=Path, default=None,
        help="repository holding the frozen PRISM identity metadata (default: parent of data-root)",
    )
    parser.add_argument("--allow-pubchem", action="store_true", help="resolve missing PDO small-molecule structures through PubChem and cache the results under --output")
    parser.add_argument("--with-prism", action="store_true", help="require materialization of the frozen PRISM 24Q2 CRC external cohort")
    parser.add_argument("--download-prism", action="store_true", help="download the official public PRISM 24Q2 files to an OS temporary directory; they are never copied into data/raw")
    parser.add_argument("--prism-matrix", type=Path)
    parser.add_argument("--prism-treatment", type=Path)
    parser.add_argument("--prism-cells", type=Path)
    args = parser.parse_args()
    output = args.output or args.data_root / "processed" / "exp011"
    project_root = args.project_root or args.data_root.parent
    payload = build_exp011(
        data_root=args.data_root,
        output=output,
        project_root=project_root,
        allow_pubchem=args.allow_pubchem,
        with_prism=args.with_prism,
        prism_matrix=args.prism_matrix,
        prism_treatment=args.prism_treatment,
        prism_cells=args.prism_cells,
        download_prism=args.download_prism,
    )
    print(json.dumps({"manifest": str(output / "cohort_index_manifest.json"), "cohorts": [row["cohort_id"] for row in payload["cohorts"]], "prism_status": payload["prism_status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
