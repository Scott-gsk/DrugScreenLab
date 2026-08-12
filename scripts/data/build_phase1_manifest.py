"""Build the Phase-1 canonical exact-978 context/chemical manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from drug_screen.data.phase1 import build_phase1_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--dose-um", type=float, default=10.0)
    parser.add_argument("--time-h", type=float, default=6.0)
    parser.add_argument("--split-mode", choices=["random_group", "cold_drug", "cold_context"], default="random_group")
    parser.add_argument("--split-seed", type=int, default=20260812)
    parser.add_argument("--n-bits", type=int, default=128)
    args = parser.parse_args()
    result = build_phase1_manifest(
        data_root=args.data_root,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        dose_um=args.dose_um,
        time_h=args.time_h,
        split_mode=args.split_mode,
        split_seed=args.split_seed,
        n_bits=args.n_bits,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
