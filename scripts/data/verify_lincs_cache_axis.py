"""核验 exact978 缓存的实例键、基因轴和来源矩阵数值对应。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def verify(gctx: Path, cache_path: Path, gene_info: Path, inst_info: Path) -> dict:
    genes = pd.read_csv(gene_info, sep="\t", dtype={"pr_gene_id": str})
    landmark = set(genes.loc[genes["pr_is_lm"].eq(1), "pr_gene_id"])
    with h5py.File(gctx, "r") as source:
        instance_ids = [x.decode() for x in source["0/META/COL/id"][:]]
        source_genes = [x.decode() for x in source["0/META/ROW/id"][:]]
        positions = [i for i, gene in enumerate(source_genes) if gene in landmark]
        ordered_genes = [source_genes[i] for i in positions]
        matrix = source["0/DATA/0/matrix"]
        cache = np.load(cache_path, mmap_mode="r")
        if matrix.shape != (len(instance_ids), len(source_genes)):
            raise ValueError("GCTX axis lengths disagree with matrix shape")
        if cache.shape != (len(instance_ids), len(ordered_genes)):
            raise ValueError("cache shape disagrees with GCTX instance/landmark axes")
        if len(ordered_genes) != 978 or len(set(ordered_genes)) != 978:
            raise ValueError("expected 978 unique directly measured landmark genes")
        if len(set(instance_ids)) != len(instance_ids):
            raise ValueError("duplicate GCTX instance IDs")
        metadata_ids = pd.read_csv(inst_info, sep="\t", usecols=["inst_id"], dtype=str)["inst_id"]
        if len(metadata_ids) != len(instance_ids) or set(metadata_ids) != set(instance_ids):
            raise ValueError("inst_info and GCTX instance ID sets differ")
        sampled_rows = sorted(set([0, 1, 31, len(instance_ids) // 2, len(instance_ids) - 1]))
        maximum_difference = 0.0
        for row in sampled_rows:
            expected = np.asarray(matrix[row, positions], dtype=np.float32)
            observed = np.asarray(cache[row], dtype=np.float32)
            difference = float(np.max(np.abs(expected - observed)))
            maximum_difference = max(maximum_difference, difference)
            if not np.array_equal(expected, observed, equal_nan=True):
                raise ValueError(f"cache numerical mismatch at instance row {row}")
    return {
        "status": "PASS_SAMPLED_VALUES_AND_COMPLETE_ID_SET",
        "source_gctx": str(gctx.resolve()),
        "cache": str(cache_path.resolve()),
        "n_instances": len(instance_ids),
        "n_source_genes": len(source_genes),
        "n_landmark_genes": len(ordered_genes),
        "sampled_instance_rows": sampled_rows,
        "max_abs_difference": maximum_difference,
        "axis_rule": "GCTX COL/id gives cache row order; GCTX ROW/id filtered in source order by gene_info pr_is_lm=1 gives cache columns",
        "metadata_identity_rule": "inst_info inst_id set exactly equals GCTX COL/id set; metadata row order is not assumed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gctx", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--gene-info", type=Path, required=True)
    parser.add_argument("--inst-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.gctx, args.cache, args.gene_info, args.inst_info)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"status": result["status"],
                      "n_instances": result["n_instances"],
                      "max_abs_difference": result["max_abs_difference"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
