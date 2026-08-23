"""Print compact schema diagnostics for EXP-010 model materialization."""
from pathlib import Path
import os
import pandas as pd

root = Path(os.environ["DRUGSCREEN_DATA_ROOT"]) / "processed" / "exp010"
for path in sorted(root.glob("*.parquet")):
    frame = pd.read_parquet(path)
    print(path.name, frame.shape, frame.index.name, list(frame.columns)[:20], list(frame.index[:3]))
