"""Build the independent XPert Context and global Drug registries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drug_screen.foundation.xpert_registry import build_context_registry, build_drug_registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-source", type=Path, required=True)
    parser.add_argument("--prism-response", type=Path, required=True)
    parser.add_argument("--context-output", type=Path, required=True)
    parser.add_argument("--drug-info", type=Path, required=True)
    parser.add_argument("--unimol", type=Path, required=True)
    parser.add_argument("--kpgt", type=Path, required=True)
    parser.add_argument("--hg", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--unipert", type=Path)
    parser.add_argument("--unipert-available", type=Path)
    parser.add_argument("--drug-output", type=Path, required=True)
    args = parser.parse_args()
    context = build_context_registry(
        adapter_path=args.context_source,
        prism_response_path=args.prism_response,
        output_path=args.context_output,
    )
    drug = build_drug_registry(
        drug_info_path=args.drug_info,
        unimol_path=args.unimol,
        kpgt_path=args.kpgt,
        hg_path=args.hg,
        bridge_path=args.bridge,
        cohort_path=args.cohort,
        unipert_path=args.unipert,
        unipert_available_path=args.unipert_available,
        output_path=args.drug_output,
    )
    print(json.dumps({"context_counts": context["counts"], "drug_counts": drug["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
