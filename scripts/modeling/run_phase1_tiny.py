"""Run the bounded Phase-1 Tiny/Small diagnostic on a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drug_screen.modeling.phase1 import run_tiny


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    result = run_tiny(
        args.manifest,
        root=args.root,
        max_records=args.max_records,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
