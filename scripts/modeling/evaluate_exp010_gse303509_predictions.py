"""Serialize the frozen breast evaluation module without model retraining."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from drug_screen.evaluation.exp010_breast import gse303509_rank_validation  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--predictions", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
frame = pd.read_csv(args.predictions)
if set(frame) != {"torg_id", "predicted_score"}:
    raise ValueError("prediction schema mismatch")
result = gse303509_rank_validation(dict(zip(frame.torg_id.astype(str), frame.predicted_score.astype(float), strict=True)))
args.output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                  default=lambda value: value.item() if isinstance(value, np.generic) else str(value)),
                       encoding="utf-8")
print(json.dumps({k: result[k] for k in ("status", "spearman", "pairwise_concordant", "pairwise_total")},
                 default=lambda value: value.item() if isinstance(value, np.generic) else str(value)))
