"""Build a bounded XPert h5ad adapter from the registered Phase-1 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drug_screen.foundation.xpert_adapter import build_phase1_adapter_h5ad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--drug-info", type=Path, required=True)
    parser.add_argument("--gene-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drug-id", action="append", dest="drug_ids")
    parser.add_argument("--split", action="append", dest="splits")
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()
    audit = build_phase1_adapter_h5ad(
        manifest_path=args.manifest,
        cache_path=args.cache,
        drug_info_path=args.drug_info,
        gene_info_path=args.gene_info,
        output_path=args.output,
        drug_ids=args.drug_ids,
        splits=args.splits,
        max_records=args.max_records,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
