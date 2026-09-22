"""Run the read-only GDSC2 original-source correspondence audit."""

import argparse
import json
from pathlib import Path

from drug_screen.data.gdsc_source_audit import audit_gdsc_source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/artifacts/EXP-014/gdsc_source_audit.json"))
    args = parser.parse_args()
    report = audit_gdsc_source(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
