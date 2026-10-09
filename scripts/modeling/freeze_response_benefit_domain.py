"""Freeze a response-blind shared exact-key development domain for EXP-066/067."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from drug_screen.data.response_benefit_contract import freeze_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output = freeze_metadata(ROOT, config_path, args.run_id)
    manifest = json.loads((output / "metadata_freeze.json").read_text(encoding="utf-8"))
    print(json.dumps({"output_dir": str(output), "manifest": manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
