"""Lossless serialization-only recovery for the runner's first prediction archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    run = ROOT / "artifacts/experiments/EXP-056" / args.run_id
    original = run / "full_keyed_predictions.npz"
    result_path = run / "result.json"
    if not original.is_file() or not result_path.is_file():
        raise FileNotFoundError("original prediction archive and result.json are required")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    original_sha = sha256(original)
    if result.get("predictions", {}).get("sha256") != original_sha:
        raise ValueError("original archive bytes differ from the just-completed run result")
    recovery_dir = run / "serialization_recovery_001"
    if recovery_dir.exists():
        raise FileExistsError("serialization recovery is immutable and this recovery id already exists")
    recovery_dir.mkdir(parents=False)
    # This is the run's own immediately-created archive, bound by result.json and SHA256.
    # The only change is string sidecar dtype: object strings become fixed-width Unicode.
    with np.load(original, allow_pickle=True) as src:
        arrays = {key: src[key].copy() for key in src.files}
    for key, value in list(arrays.items()):
        if value.dtype.kind == "O":
            if not all(isinstance(v, (str, np.str_)) for v in value.reshape(-1).tolist()):
                raise TypeError(f"object array {key} contains a non-string object")
            arrays[key] = value.astype(str)
    if any(value.dtype.kind == "O" for value in arrays.values()):
        raise TypeError("recovery archive still contains object arrays")
    recovered = recovery_dir / "full_keyed_predictions.npz"
    np.savez_compressed(recovered, **arrays)
    with np.load(recovered, allow_pickle=False) as check:
        roundtrip = {key: check[key].copy() for key in check.files}
    if set(roundtrip) != set(arrays):
        raise ValueError("recovery archive changed array names")
    max_numeric_abs_error = 0.0
    checks: dict[str, bool] = {}
    for key, before in arrays.items():
        after = roundtrip[key]
        if before.shape != after.shape:
            raise ValueError(f"recovery archive changed shape for {key}")
        if before.dtype.kind in "biufc":
            if not np.array_equal(before, after):
                max_numeric_abs_error = max(max_numeric_abs_error,
                    float(np.max(np.abs(before.astype(np.float64) - after.astype(np.float64)))))
                raise ValueError(f"recovery archive changed numerical values for {key}")
        elif not np.array_equal(before.astype(str), after.astype(str)):
            raise ValueError(f"recovery archive changed identity strings for {key}")
        checks[key] = True
    receipt: dict[str, Any] = {
        "status": "PASS_LOSSLESS_STRING_DTYPE_RECOVERY",
        "run_id": args.run_id,
        "original_archive": str(original.relative_to(ROOT)),
        "original_archive_sha256": original_sha,
        "recovered_archive": "serialization_recovery_001/full_keyed_predictions.npz",
        "recovered_archive_sha256": sha256(recovered),
        "cause": "first archive encoded pandas string identity axes as object dtype; allow_pickle=False readback failed",
        "repair": "convert only object-dtype string arrays to fixed-width Unicode; preserve all numeric arrays exactly",
        "exact_array_roundtrip_checks": checks,
        "max_numeric_absolute_error": max_numeric_abs_error,
        "model_refit": False,
        "hyperparameter_or_result_change": False,
        "gpu_hours": 0.0,
        "cpu_seconds": time.monotonic() - started,
    }
    (recovery_dir / "recovery_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    recovered_result = json.loads(json.dumps(result))
    recovered_result["predictions"]["file"] = "serialization_recovery_001/full_keyed_predictions.npz"
    recovered_result["predictions"]["sha256"] = sha256(recovered)
    recovered_result["predictions"]["numpy_allow_pickle_required"] = False
    recovered_result["serialization_recovery"] = {
        "receipt": "serialization_recovery_001/recovery_receipt.json",
        "receipt_sha256": sha256(recovery_dir / "recovery_receipt.json"),
        "lossless": True,
        "fit_or_metrics_recomputed": False,
        "only_numeric_archive_serialization_changed": True,
    }
    (recovery_dir / "recovery_result.json").write_text(
        json.dumps(recovered_result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    recovery_manifest = pd.DataFrame([
        {"file": p.name, "sha256": sha256(p), "bytes": p.stat().st_size}
        for p in sorted(recovery_dir.iterdir()) if p.is_file()
    ])
    recovery_manifest.to_csv(recovery_dir / "sha256_manifest.tsv", sep="\t", index=False)
    print(json.dumps({"status": receipt["status"], "run_id": args.run_id,
                      "original_archive_sha256": original_sha,
                      "recovered_archive_sha256": receipt["recovered_archive_sha256"],
                      "arrays": len(checks), "max_numeric_absolute_error": max_numeric_abs_error,
                      "model_refit": False, "gpu_hours": 0.0,
                      "recovery_dir": str(recovery_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
