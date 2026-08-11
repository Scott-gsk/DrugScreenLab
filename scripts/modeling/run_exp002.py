"""Run the frozen EXP-002 baselines against a Data Steward materialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drug_screen.modeling.exp002 import load_exp002_config, run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_exp002_config(args.config)
    result = run_evaluation(config, args.materialized_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
