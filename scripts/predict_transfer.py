"""Rank candidate drugs with one frozen project transfer model."""
from __future__ import annotations
import argparse
from pathlib import Path
from drug_screen.transfer.inference import predict_transfer

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, type=Path); parser.add_argument("--preprocessor", required=True, type=Path)
parser.add_argument("--expression", required=True, type=Path); parser.add_argument("--candidates", required=True, type=Path)
parser.add_argument("--output-dir", required=True, type=Path); parser.add_argument("--explain-top-k", type=int, default=3)
parser.add_argument("--explain-steps", type=int, default=64)
def main() -> int:
    args = parser.parse_args()
    predict_transfer(args.model, args.preprocessor, args.expression, args.candidates, args.output_dir,
                     explain_top_k=args.explain_top_k, explain_steps=args.explain_steps)
    return 0
if __name__ == "__main__": raise SystemExit(main())
