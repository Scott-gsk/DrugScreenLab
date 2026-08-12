"""Run the bounded MVP-001 learned perturbation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drug_screen.modeling.mvp001 import IntegrationError, load_config, run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    try:
        result = run(load_config(args.config), args.manifest, args.output_dir, data_root=args.data_root)
    except IntegrationError as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(json.dumps({"experiment_id": "MVP-001", "status": "BROKEN", "integration_error": str(error)}, indent=2) + "\n", encoding="utf-8")
        return 2
    print(json.dumps({"status": result["status"], "summary": str(args.output_dir / "summary.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
