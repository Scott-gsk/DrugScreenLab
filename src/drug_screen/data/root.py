from __future__ import annotations
import os
from pathlib import Path

def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]

def data_root() -> Path:
    configured = os.environ.get("DRUGSCREEN_DATA_ROOT")
    return Path(configured).expanduser().resolve() if configured else repository_root() / "data"
