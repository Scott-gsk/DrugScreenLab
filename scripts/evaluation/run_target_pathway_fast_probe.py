"""Run the bounded, label-free Target/Pathway forward-preparation probe."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from drug_screen.modeling.mechanism_fast import build_frozen_candidate_mechanism_probe


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perturbagens", type=Path, required=True)
    parser.add_argument("--prism-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_frozen_candidate_mechanism_probe(
        perturbagen_path=args.perturbagens,
        prism_audit_path=args.prism_audit,
        output_dir=args.output_dir,
    )
    print(json.dumps({"status": result["status"], "decision": result["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
