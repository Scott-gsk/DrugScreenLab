from __future__ import annotations

from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPR = ROOT / "data/raw/depmap/24q2/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
MODEL = ROOT / "data/raw/depmap/24q2/Model.csv"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    print("expr_bytes", EXPR.stat().st_size)
    print("model_bytes", MODEL.stat().st_size)
    with EXPR.open("r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\n")
        n_rows = sum(1 for _ in handle)
    cols = header.split(",")
    print("n_cols", len(cols))
    print("first10", cols[:10])
    print("last3", cols[-3:])
    print("n_rows", n_rows)
    print("expr_sha256", digest(EXPR))
    print("model_sha256", digest(MODEL))


if __name__ == "__main__":
    main()
