from __future__ import annotations
import argparse
from pathlib import Path
from drug_screen.transfer.functional_feasibility import run_functional_feasibility
parser = argparse.ArgumentParser(); parser.add_argument("--data-dir", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path); parser.add_argument("--epochs", type=int, default=100)
def main():
    args = parser.parse_args(); run_functional_feasibility(args.data_dir, args.output_dir, epochs=args.epochs); return 0
if __name__ == "__main__": raise SystemExit(main())
