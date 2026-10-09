"""逐 row group 独立核对旧预测与身份修复版的每个原字段。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def verify(original: Path, repaired: Path, kind: str) -> dict:
    old = pq.ParquetFile(original)
    new = pq.ParquetFile(repaired)
    if old.metadata.num_rows != new.metadata.num_rows:
        raise ValueError("row count changed")
    if old.num_row_groups != new.num_row_groups:
        raise ValueError("row group count changed")
    source_columns = old.schema.names
    for group in range(old.num_row_groups):
        before = old.read_row_group(group)
        after = new.read_row_group(group)
        if before.num_rows != after.num_rows:
            raise ValueError(f"row group {group} size changed")
        preserved = [column for column in source_columns if column != "model_id"]
        if not before.select(preserved).equals(after.select(preserved)):
            raise ValueError(f"row group {group} original fields changed")
        if kind == "l1":
            if not before.column("model_id").equals(after.column("algorithm")):
                raise ValueError(f"row group {group} algorithm value mismatch")
            if not before.column("sample_id").equals(after.column("model_id")):
                raise ValueError(f"row group {group} biological model value mismatch")
        elif kind == "c4":
            if not before.column("arm").equals(after.column("algorithm")):
                raise ValueError(f"row group {group} algorithm value mismatch")
            if not before.column("sample_id").equals(after.column("model_id")):
                raise ValueError(f"row group {group} biological model value mismatch")
        else:
            raise ValueError(f"unknown kind: {kind}")
    return {"status": "PASS_ALL_ROWS_ALL_ORIGINAL_FIELDS",
            "kind": kind, "rows_checked": old.metadata.num_rows,
            "row_groups_checked": old.num_row_groups}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--repaired", type=Path, required=True)
    parser.add_argument("--kind", choices=["l1", "c4"], required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.original, args.repaired, args.kind), ensure_ascii=False))


if __name__ == "__main__":
    main()
