"""不改写旧预测，生成生物 model_id 与算法身份分列的新版本。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from drug_screen.data.identity_contracts import assert_model_id_is_biological


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _repair_l1(source: Path, destination: Path) -> int:
    reader = pq.ParquetFile(source)
    writer: pq.ParquetWriter | None = None
    total = 0
    try:
        for group in range(reader.num_row_groups):
            frame = reader.read_row_group(group).to_pandas()
            if "algorithm" in frame.columns:
                raise ValueError("source already has algorithm column")
            if not frame["sample_id"].astype(str).eq(frame["patient_id"].astype(str)).all():
                raise ValueError("L1 sample and cell identity differ; cannot assign model_id")
            frame["algorithm"] = frame["model_id"].astype(str)
            frame["model_id"] = frame["sample_id"].astype(str)
            assert_model_id_is_biological(frame)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
            total += len(frame)
    finally:
        if writer is not None:
            writer.close()
    if total != reader.metadata.num_rows:
        raise ValueError("output row count differs from source")
    return total


def _repair_c4(source: Path, destination: Path) -> int:
    frame = pd.read_parquet(source)
    if "model_id" in frame.columns or "algorithm" in frame.columns:
        raise ValueError("C4 source is already normalized")
    frame["model_id"] = frame["sample_id"].astype(str)
    frame["algorithm"] = frame["arm"].astype(str)
    assert_model_id_is_biological(frame)
    frame.to_parquet(destination, index=False)
    return len(frame)


def repair(source: Path, destination: Path, kind: str) -> dict:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"partial output already exists: {temporary}")
    try:
        count = (_repair_l1(source, temporary) if kind == "l1"
                 else _repair_c4(source, temporary))
        if pq.ParquetFile(temporary).metadata.num_rows != count:
            raise ValueError("written parquet row count mismatch")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result = {
        "status": "IDENTITY_SEMANTICS_REPAIRED_SOURCE_PRESERVED",
        "kind": kind, "source": str(source.resolve()),
        "source_sha256": _hash(source),
        "output": str(destination.resolve()),
        "output_sha256": _hash(destination),
        "rows": count,
        "repair_script_sha256": _hash(Path(__file__)),
        "git_head": head,
        "working_tree_status": subprocess.check_output(
            ["git", "status", "--short"], text=True),
        "limitations": (
            "only identity semantics repaired; source model/checkpoint and original protocol unchanged"
            if kind == "l1" else
            "only model_id and algorithm added; legacy C4 seed/run/contract metadata absent and not invented"
        ),
    }
    manifest = destination.with_suffix(".manifest.json")
    if manifest.exists():
        raise FileExistsError(f"refusing to overwrite manifest {manifest}")
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=["l1", "c4"], required=True)
    args = parser.parse_args()
    result = repair(args.input, args.output, args.kind)
    print(json.dumps({"output": result["output"], "rows": result["rows"],
                      "status": result["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
