"""Build an additive KPGT+UniPert Phase-1 manifest from frozen feature tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drug_screen.modeling.phase2_fusion import build_fused_manifest


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--kpgt-features", type=Path, required=True)
    parser.add_argument("--kpgt-mapping", type=Path, required=True)
    parser.add_argument("--unipert-features", type=Path, required=True)
    parser.add_argument("--unipert-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    audit = build_fused_manifest(
        base_manifest_path=args.base_manifest,
        kpgt_features_path=args.kpgt_features,
        kpgt_mapping_path=args.kpgt_mapping,
        unipert_features_path=args.unipert_features,
        unipert_mapping_path=args.unipert_mapping,
        output_dir=args.output_dir,
        root=args.root,
    )
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
