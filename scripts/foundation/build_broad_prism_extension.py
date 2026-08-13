"""Build the frozen-candidate Broad CRC response compact after identity freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drug_screen.foundation.broad_prism import build_broad_prism_compact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--treatment-info", type=Path, required=True)
    parser.add_argument("--cell-info", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    result = build_broad_prism_compact(
        cohort_path=args.cohort,
        bridge_path=args.bridge,
        treatment_info_path=args.treatment_info,
        cell_info_path=args.cell_info,
        registry_path=args.registry,
        matrix_path=args.matrix,
        output_path=args.output,
        audit_path=args.audit,
    )
    print(json.dumps(result["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
