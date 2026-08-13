"""Build a response-blind Broad PRISM↔LINCS compound identity bridge."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from drug_screen.data.prism_bridge import build_identity_bridge


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prism-treatment-info", type=Path, required=True)
    parser.add_argument("--lincs-pert-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    prism = pd.read_csv(args.prism_treatment_info, low_memory=False)
    lincs = pd.read_csv(args.lincs_pert_info, sep="\t", low_memory=False)
    bridge, audit = build_identity_bridge(prism, lincs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bridge.to_csv(args.output, index=False)
    audit.update(
        {
            "prism_treatment_info": str(args.prism_treatment_info),
            "prism_treatment_info_sha256": _sha256(args.prism_treatment_info),
            "lincs_pert_info": str(args.lincs_pert_info),
            "lincs_pert_info_sha256": _sha256(args.lincs_pert_info),
            "output": str(args.output),
            "output_sha256": _sha256(args.output),
        }
    )
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
