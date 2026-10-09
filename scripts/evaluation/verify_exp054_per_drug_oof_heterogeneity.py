"""Independent readback of the EXP-054 per-drug TRAIN OOF decomposition."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("zero_change", "per_drug_mean", "state_residual", "shuffled_state_control")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    out = ROOT / args.run_dir
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    if result.get("status") != "COMPLETED_POSTHOC_TRAIN_OOF_PER_DRUG_DECOMPOSITION":
        raise ValueError("EXP-054 analysis has not completed")
    source = ROOT / "artifacts/experiments/EXP-053/20261007_exp053_drug_shared_state_residual_v1"
    archive_path = source / "train_object_oof_predictions.npz"
    with np.load(archive_path, allow_pickle=False) as z:
        arr = {k: z[k].copy() for k in z.files}
    y = arr["truth_delta"].astype(np.float64)
    pred = {m: arr[m].astype(np.float64) for m in METHODS}
    obj = arr["source_object_id"].astype(str)
    key = arr["full_inchikey"].astype(str)
    fold = arr["fold_id"].astype(int)
    if y.shape != (419, 978) or set(key).__len__() != 5 or len(set(obj)) != 110:
        raise ValueError("fixed TRAIN OOF axes differ from 419x978, 5 drugs, 110 objects")
    if not np.isfinite(y).all() or any(not np.isfinite(v).all() for v in pred.values()):
        raise FloatingPointError("non-finite frozen TRAIN OOF values")

    # Recompute object-by-drug NMSE independently from the saved predictions.
    recomputed: dict[tuple[str, str, str], float] = {}
    for f in range(5):
        train = np.flatnonzero(fold != f)
        test = np.flatnonzero(fold == f)
        variance = np.var(y[train], axis=0, ddof=0)
        use = np.isfinite(variance) & (variance > 1e-12)
        for d in sorted(set(key[test])):
            di = test[key[test] == d]
            for o in sorted(set(obj[di])):
                ix = di[obj[di] == o]
                for m in METHODS:
                    e = pred[m][ix] - y[ix]
                    value = float(np.mean(np.mean(np.square(e[:, use]) / variance[use][None, :], axis=1)))
                    recomputed[(d, o, m)] = value
    per_object_path = out / "per_object_drug_method_metrics.tsv"
    saved_obj = pd.read_csv(per_object_path, sep="\t", dtype={"full_inchikey": str, "source_object_id": str})
    if len(saved_obj) != len(recomputed):
        raise ValueError("per-object/drug denominator differs from independent OOF recomputation")
    max_error = 0.0
    for r in saved_obj.itertuples(index=False):
        value = recomputed[(r.full_inchikey, r.source_object_id, r.method)]
        max_error = max(max_error, abs(value - float(r.nmse)))
    if max_error > 1e-10:
        raise ValueError(f"per-object/drug NMSE readback mismatch: {max_error}")

    saved_summary = pd.read_csv(out / "per_drug_method_summary.tsv", sep="\t")
    key_results = {x["full_inchikey"]: x for x in result["drug_strata"]}
    paired = pd.read_csv(out / "per_drug_paired_nmse_differences.tsv", sep="\t")
    drug_improved = 0
    bootstrap_error = 0.0
    for key_index, d in enumerate(sorted(set(key))):
        table = saved_summary.loc[saved_summary.full_inchikey.eq(d)].set_index("method")
        n = len({o for (k, o, m) in recomputed if k == d and m == "per_drug_mean"})
        condition_n = int(np.sum(key == d))
        if (int(table.loc["per_drug_mean", "source_objects"]) != n
                or int(table.loc["per_drug_mean", "conditions"]) != condition_n):
            raise ValueError(f"drug denominator mismatch for {d}")
        mean_base = float(np.mean([v for (k, _, m), v in recomputed.items() if k == d and m == "per_drug_mean"]))
        mean_state = float(np.mean([v for (k, _, m), v in recomputed.items() if k == d and m == "state_residual"]))
        mean_shuffle = float(np.mean([v for (k, _, m), v in recomputed.items() if k == d and m == "shuffled_state_control"]))
        if not np.isclose(mean_base, float(table.loc["per_drug_mean", "object_macro_nmse"]), rtol=0, atol=1e-10):
            raise ValueError(f"per-drug mean summary mismatch for {d}")
        if not np.isclose(mean_state, float(table.loc["state_residual", "object_macro_nmse"]), rtol=0, atol=1e-10):
            raise ValueError(f"state-residual summary mismatch for {d}")
        if not np.isclose(mean_shuffle, float(table.loc["shuffled_state_control", "object_macro_nmse"]), rtol=0, atol=1e-10):
            raise ValueError(f"shuffled-state summary mismatch for {d}")
        if mean_state < mean_base:
            drug_improved += 1
        delta = mean_base - mean_state
        record = key_results[d]
        if not np.isclose(delta, -float(record["state_minus_per_drug_mean_nmse"]), rtol=0, atol=1e-10):
            raise ValueError(f"paired mean difference mismatch for {d}")
        paired_rows = paired.loc[paired.full_inchikey.eq(d)]
        if len(paired_rows) != 2:
            raise ValueError(f"paired-comparison count mismatch for {d}")
        objects_for_drug = sorted({o for (k, o, m) in recomputed if k == d and m == "per_drug_mean"})
        for pair_index, reference in enumerate(("per_drug_mean", "shuffled_state_control")):
            row = paired_rows.loc[paired_rows.reference.eq(reference)].iloc[0]
            differences = np.asarray([
                recomputed[(d, o, reference)] - recomputed[(d, o, "state_residual")]
                for o in objects_for_drug
            ], dtype=np.float64)
            seed = 54053 + key_index * 10 + pair_index
            rng = np.random.default_rng(seed)
            indices = rng.integers(0, len(differences), size=(20000, len(differences)))
            samples = differences[indices].mean(axis=1)
            expected = np.asarray([differences.mean(), *np.quantile(samples, [0.025, 0.975])])
            actual = np.asarray([float(row.reference_minus_candidate_nmse),
                                 float(row.bootstrap_95pct_ci_low), float(row.bootstrap_95pct_ci_high)])
            bootstrap_error = max(bootstrap_error, float(np.max(np.abs(expected - actual))))
            if np.max(np.abs(expected - actual)) > 1e-10:
                raise ValueError(f"paired estimate/bootstrap differs for {d}/{reference}")
    if drug_improved != int(result["state_residual_lower_nmse_than_per_drug_mean_drug_strata"]):
        raise ValueError("result-level count of improved drug strata is incorrect")

    files = [out / "per_object_drug_method_metrics.tsv", out / "per_drug_method_summary.tsv",
             out / "per_drug_paired_nmse_differences.tsv", out / "result.json", out / "sha256_manifest.tsv"]
    manifest = pd.read_csv(out / "sha256_manifest.tsv", sep="\t", dtype=str)
    for row in manifest.itertuples(index=False):
        p = out / row.file
        if not p.is_file() or digest(p) != row.sha256 or p.stat().st_size != int(row.bytes):
            raise ValueError(f"output hash/size mismatch: {row.file}")
    receipt = {
        "status": "PASS_ROOT_INDEPENDENT_OOF_READBACK",
        "run_id": result["run_id"],
        "source_run_id": "20261007_exp053_drug_shared_state_residual_v1",
        "claim_boundary": "fixed EXP-053 TRAIN-only source-object OOF per-drug description; no new fit or PDO/generalization inference",
        "checks": {"allow_pickle_false": True, "train_rows": int(len(y)), "genes": int(y.shape[1]),
                   "source_objects": int(len(set(obj))), "full_inchikeys": int(len(set(key))),
                   "folds": int(len(set(fold))), "per_object_drug_rows": int(len(saved_obj)),
                   "all five drug strata recomputed": True,
                   "state residual better than per-drug mean strata": drug_improved,
                   "inner_dev_pdo_outer_locked_values_read": False,
                   "models_refit": False, "gpu_hours": 0.0,
                   "maximum_per_object_nmse_absolute_error": max_error,
                   "maximum_paired_bootstrap_absolute_error": bootstrap_error,
                   "listed_output_sha256_and_sizes_pass": True},
        "input_sha256": {"source_oof_npz": digest(archive_path), "analysis_result": digest(out / "result.json")},
        "verified_cpu_wall_seconds": time.monotonic() - started,
    }
    receipt_path = out / "root_independent_readback_v2.json"
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite {receipt_path}")
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
