from __future__ import annotations
import argparse
import json
from pathlib import Path
from .root import data_root
from .p0 import DatasetReadiness

def load_registry(root: Path | None = None) -> list[dict]:
    registry_dir = (root or data_root()) / "registry"
    entries = []
    for path in sorted(registry_dir.glob("datasets*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        entries.extend(value if isinstance(value, list) else [value])
    return entries

def validate_registry(root: Path | None = None) -> list[str]:
    base = root or data_root()
    errors = []
    for entry in load_registry(base):
        for field in ("id", "version", "asset_type", "path"):
            if field not in entry:
                errors.append(f"missing {field}")
        relative = entry.get("path", {}).get("relative")
        if relative and not (base / relative).exists():
            errors.append(f"missing asset: {relative}")
    return errors


def validate_readiness_matrix(path: Path) -> list[str]:
    """Validate the Phase-0 matrix without treating unavailable datasets as present."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        return ["readiness matrix must be a list"]
    errors = []
    for index, row in enumerate(value):
        try:
            DatasetReadiness.from_mapping(row)
        except (TypeError, ValueError) as exc:
            errors.append(f"row {index}: {exc}")
    return errors

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    errors = validate_registry(args.root)
    if errors:
        print("FAIL")
        print("\n".join(errors))
        return 1
    print("PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
