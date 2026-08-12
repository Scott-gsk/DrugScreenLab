"""Build a frozen UniPert chemical representation variant of the Phase-1 manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from drug_screen.modeling.phase2_fast import build_unipert_manifest


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--perturbagens", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    audit = build_unipert_manifest(
        base_manifest_path=args.base_manifest,
        perturbagen_path=args.perturbagens,
        model_path=args.model,
        output_dir=args.output_dir,
        root=args.root,
    )
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
