"""Create a bounded response-blind prefix used only for hardware probes."""
import argparse
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--rows", type=int, default=128)
args = parser.parse_args()
source = np.load(args.source, allow_pickle=False)
payload = {key: source[key][: args.rows] for key in source.files}
args.output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.output, **payload)
print(args.output)
