"""Build an inference-only exact-context × drug XPert adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drug_screen.foundation.xpert_adapter import build_cartesian_adapter_h5ad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drug-id", action="append", dest="drug_ids", required=True)
    args = parser.parse_args()
    result = build_cartesian_adapter_h5ad(
        source_path=args.source,
        output_path=args.output,
        drug_ids=args.drug_ids,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
