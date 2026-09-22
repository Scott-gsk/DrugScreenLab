"""Run a new EXP-016 cold-cell diagnostic without overwriting cold_cell_v1."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drug_screen.transfer.bundles import IndexedBundle
from drug_screen.transfer.cold_cell import run_cold_cell


def _bundle():
    root, directory = ROOT / "data", ROOT / "data/processed/exp014/gdsc_v3"
    samples, drugs, pairs = (pd.read_parquet(directory / name) for name in ("samples.parquet", "drugs.parquet", "pairs.parquet"))
    with np.load(directory / "indexed.npz", allow_pickle=True) as archive:
        expression, sample_ids, genes = np.asarray(archive["expression"]), archive["sample_id"].astype(str), archive["ordered_genes"].astype(str)
    return root, IndexedBundle("gdsc2_cell_line", "SOURCE_TRAIN_AND_COLD_CELL_EVALUATION", samples, drugs, pairs, expression, sample_ids, genes, "log2(CMP TPM+1)", True, "1_minus_auc")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="cold_cell_v2_smoke")
    parser.add_argument("--split-seeds", default="20260913")
    parser.add_argument("--init-seeds", default="20260913")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutation-repeats", type=int, default=100)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    root, bundle = _bundle()
    result = run_cold_cell(bundle=bundle, output_dir=root / "artifacts/EXP-016" / args.output,
                           split_seeds=tuple(int(x) for x in args.split_seeds.split(",")),
                           init_seeds=tuple(int(x) for x in args.init_seeds.split(",")),
                           epochs=args.epochs, device=args.device, resamples=args.bootstrap,
                           permutation_repeats=args.permutation_repeats)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__": main()
