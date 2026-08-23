"""Validate an EXP-010 checkpoint and compact prediction artifact."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--predictions", type=Path, required=True)
args = parser.parse_args()
prediction = np.load(args.predictions, allow_pickle=False)
for key in ("observed", "predicted"):
    if not np.isfinite(prediction[key]).all():
        raise ValueError(f"non-finite {key}")
checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
if checkpoint.get("format") != "drugscreenlab_exp010_pharmaformer_v1":
    raise ValueError("checkpoint format mismatch")
print(json.dumps({"variant": checkpoint["variant"], "rows": len(prediction["predicted"]),
                  "prediction_range": [float(prediction["predicted"].min()), float(prediction["predicted"].max())],
                  "state_keys": len(checkpoint["model_state_dict"])}))
