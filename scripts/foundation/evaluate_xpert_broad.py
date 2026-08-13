"""Evaluate XPert global predictions against Broad PRISM and LINCS Oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drug_screen.evaluation.xpert_broad import build


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--signature", type=Path, required=True)
    parser.add_argument("--prism", type=Path, required=True)
    parser.add_argument("--observed-lincs", type=Path)
    parser.add_argument("--minimum-candidates", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        profile_path=args.profile,
        adapter_path=args.adapter,
        signature_path=args.signature,
        prism_path=args.prism,
        observed_lincs_path=args.observed_lincs,
        minimum_candidates=args.minimum_candidates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "broad_prism": result["broad_prism"]["line_metrics"], "oracle": result["observed_lincs_oracle"].get("line_metrics")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
