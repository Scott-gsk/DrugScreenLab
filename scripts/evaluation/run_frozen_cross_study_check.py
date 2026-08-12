"""Run one frozen external ranking check when the official label asset is present."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from drug_screen.evaluation.cross_study import evaluate_frozen_cross_study


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    args = parser.parse_args()
    result = evaluate_frozen_cross_study(
        labels_path=args.labels,
        predictions_path=args.predictions,
        output_path=args.output,
        frozen_candidate_ids=args.candidate,
    )
    print(json.dumps({"status": result["status"], "macro_mean_spearman": result["macro_mean_spearman"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
