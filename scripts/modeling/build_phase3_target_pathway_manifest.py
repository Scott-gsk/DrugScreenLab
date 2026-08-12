"""Build the one-seed Target/Pathway FAST feature variant."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from drug_screen.modeling.mechanism_fast import build_target_pathway_manifest


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--perturbagens", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--max-records", type=int, default=2048)
    args = parser.parse_args()
    audit = build_target_pathway_manifest(
        base_manifest_path=args.base_manifest,
        perturbagen_path=args.perturbagens,
        output_dir=args.output_dir,
        root=args.root,
        max_records=args.max_records,
    )
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
