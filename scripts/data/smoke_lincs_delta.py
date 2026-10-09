"""在已核验的 LINCS 缓存上运行一个真实处理-对照连续标签小样例。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from drug_screen.data.lincs_delta import LincsMatchedDelta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gctx", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--inst-info", type=Path, required=True)
    parser.add_argument("--axis-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.axis_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_SAMPLED_VALUES_AND_COMPLETE_ID_SET":
        raise ValueError("cache axis audit has not passed")
    with h5py.File(args.gctx, "r") as source:
        instance_ids = [value.decode() for value in source["0/META/COL/id"][:]]
    metadata = pd.read_csv(args.inst_info, sep="\t", dtype=str, keep_default_na=False)
    candidate = metadata.loc[
        metadata["pert_type"].eq("trt_cp")
        & metadata["pert_time"].eq("24")
        & metadata["pert_time_unit"].eq("h")
        & metadata["pert_dose_unit"].str.lower().eq("um")
        & pd.to_numeric(metadata["pert_dose"], errors="coerce").eq(10)
    ]
    source = LincsMatchedDelta(args.cache, instance_ids, metadata)
    chosen = next(
        (row.inst_id for row in candidate.itertuples(index=False)
         if source.controls.get((row.rna_plate, row.cell_id,
                                 row.pert_time, row.pert_time_unit))),
        None,
    )
    if chosen is None:
        raise ValueError("no eligible treatment has matched vehicle")
    delta, controls = source.delta(chosen)
    if delta.shape != (978,) or not np.isfinite(delta).all():
        raise ValueError("invalid continuous delta vector")
    digest = hashlib.sha256(delta.tobytes()).hexdigest()
    result = {
        "status": "REAL_DATA_SMOKE_PASS_NOT_TRAINING_ADMISSION",
        "treatment_inst_id": chosen,
        "matched_control_inst_ids": list(controls),
        "n_genes": len(delta),
        "delta_float32_sha256": digest,
        "delta_min": float(delta.min()),
        "delta_max": float(delta.max()),
        "delta_mean": float(delta.mean()),
        "axis_audit": str(args.axis_audit.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
