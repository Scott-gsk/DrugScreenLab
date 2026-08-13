"""Build the global-registry-driven XPert Cartesian inference adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drug_screen.foundation.xpert_registry import build_global_cartesian_adapter_h5ad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", action="append", dest="contexts")
    parser.add_argument("--broad-only", action="store_true")
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    audit = build_global_cartesian_adapter_h5ad(
        source_path=args.source,
        registry_path=args.registry,
        output_path=args.output,
        context_ids=args.contexts,
        broad_only=args.broad_only,
    )
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
