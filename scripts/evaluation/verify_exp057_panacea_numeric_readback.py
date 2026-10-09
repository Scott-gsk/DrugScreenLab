"""Independent root recomputation for the frozen EXP-057 PANACEA baseline run.

This utility intentionally reads only the 11 frozen PANACEA treatment rows and
their paired controls, plus the separately materialized source TRAIN target
array. It never indexes H5AD layers or any restricted evaluation data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/experiments/EXP-057-panacea-development.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _close(actual: float | None, expected: Any, *, atol: float = 2e-12) -> bool:
    if expected is None or (isinstance(expected, float) and np.isnan(expected)):
        return actual is None
    return actual is not None and bool(np.isclose(actual, float(expected), rtol=2e-12, atol=atol))


def _selected_rows(node: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    if not isinstance(node, h5py.Dataset) or len(node.shape) != 2 or node.shape[1] != 978:
        raise ValueError(f"unexpected selected matrix node/shape: {getattr(node, 'name', None)} {getattr(node, 'shape', None)}")
    if len(rows) != 11 or len(set(rows.tolist())) != 11 or rows.min() < 0 or rows.max() >= node.shape[0]:
        raise ValueError("root readback requires the exact 11 unique frozen row indices")
    order = np.argsort(rows, kind="mergesort")
    selected = np.asarray(node[rows[order], :], dtype=np.float64)
    restored = np.empty_like(selected)
    restored[order] = selected
    if not np.isfinite(restored).all():
        raise FloatingPointError(f"non-finite value in explicitly selected rows of {node.name}")
    return restored


def _axis_reindex(source_labels: list[str], target_genes: list[str], mapping_path: Path, gene_info_path: Path) -> np.ndarray:
    mapping = pd.read_csv(mapping_path, sep="\t", dtype=str, keep_default_na=False).sort_values("axis_index", key=lambda s: pd.to_numeric(s), kind="stable")
    info = pd.read_csv(gene_info_path, dtype=str, keep_default_na=False)
    if len(mapping) != 978 or mapping["axis_index"].astype(int).tolist() != list(range(978)):
        raise ValueError("source axis mapping is not the frozen ordered 978-axis")
    source_var = mapping["h5ad_var_name"].astype(str).tolist()
    gene_ids = mapping["l1000_gene_info_value"].astype(str).tolist()
    if source_labels != source_var or len(set(gene_ids)) != 978:
        raise ValueError("TRAIN archive labels do not match the frozen source axis mapping")
    if not {"gene_id", "gene_symbol"}.issubset(info.columns):
        raise ValueError("official gene-info file lacks gene_id/gene_symbol")
    info = info[["gene_id", "gene_symbol"]].copy()
    info["gene_id"] = info["gene_id"].astype(str)
    info["gene_symbol"] = info["gene_symbol"].astype(str)
    if info["gene_id"].duplicated().any() or info["gene_symbol"].duplicated().any():
        raise ValueError("official gene crosswalk is not one-to-one")
    crosswalk = info.set_index("gene_id")["gene_symbol"]
    symbols = [crosswalk.loc[gene_id] for gene_id in gene_ids]
    if len(set(symbols)) != 978 or set(symbols) != set(target_genes):
        raise ValueError("mapped source TRAIN gene symbols differ from the target frozen axis")
    source_position = {symbol: i for i, symbol in enumerate(symbols)}
    return np.asarray([source_position[symbol] for symbol in target_genes], dtype=np.int64)


def _ranks(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average")


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) != len(b) or len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    ra, rb = _ranks(a), _ranks(b)
    if np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _condition_metrics(target: np.ndarray, prediction: np.ndarray, hallmark_sets: dict[str, np.ndarray]) -> dict[str, Any]:
    active = target != 0.0
    n_nonzero = int(active.sum())
    direction_matches = int((np.sign(target[active]) == np.sign(prediction[active])).sum()) if n_nonzero else 0
    set_target_active = 0
    set_matches = 0
    for indices in hallmark_sets.values():
        target_mean = float(np.mean(target[indices], dtype=np.float64))
        pred_mean = float(np.mean(prediction[indices], dtype=np.float64))
        if target_mean != 0:
            set_target_active += 1
            if np.sign(target_mean) == np.sign(pred_mean):
                set_matches += 1
    return {
        "gene_direction_agreement": direction_matches / n_nonzero if n_nonzero else None,
        "gene_direction_numerator": direction_matches,
        "gene_direction_denominator_target_nonzero": n_nonzero,
        "target_zero_gene_count": int((~active).sum()),
        "prediction_zero_gene_count": int((prediction == 0.0).sum()),
        "gene_spearman": _spearman(target, prediction),
        "gene_spearman_finite_gene_count": int(len(target)),
        "hallmark_direction_agreement": set_matches / set_target_active if set_target_active else None,
        "hallmark_direction_numerator": set_matches,
        "hallmark_direction_denominator_target_nonzero": set_target_active,
        "hallmark_target_zero_set_count": len(hallmark_sets) - set_target_active,
        "hallmark_prediction_nonzero_set_count": sum(float(np.mean(prediction[idx])) != 0.0 for idx in hallmark_sets.values()),
        "hallmark_both_nonzero_set_count": sum(float(np.mean(target[idx])) != 0.0 and float(np.mean(prediction[idx])) != 0.0 for idx in hallmark_sets.values()),
        "hallmark_set_count": len(hallmark_sets),
    }


def _bootstrap(values: np.ndarray, seed: int, reps: int) -> tuple[float | None, float | None, float | None, int]:
    finite = values[np.isfinite(values)]
    n = len(finite)
    if n == 0:
        return None, None, None, 0
    point = float(np.mean(finite, dtype=np.float64))
    if n < 2:
        return point, None, None, n
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(reps, n))
    low, high = np.quantile(finite[draws].mean(axis=1, dtype=np.float64), [0.025, 0.975])
    return point, float(low), float(high), n


def _paired_bootstrap(left: np.ndarray, right: np.ndarray, seed: int, reps: int) -> tuple[float | None, float | None, float | None, int]:
    valid = np.isfinite(left) & np.isfinite(right)
    diff = left[valid] - right[valid]
    n = len(diff)
    if n == 0:
        return None, None, None, 0
    point = float(np.mean(diff, dtype=np.float64))
    if n < 2:
        return point, None, None, n
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(reps, n))
    low, high = np.quantile(diff[draws].mean(axis=1, dtype=np.float64), [0.025, 0.975])
    return point, float(low), float(high), n


def _verify(config_path: Path, receipt_path: Path | None) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_id = str(config["execution"]["run_id"])
    out = ROOT / "artifacts/experiments/EXP-057" / run_id
    manifest_path = out / "run_manifest.json"
    result_path = out / "summary_metrics.json"
    if not manifest_path.is_file() or not result_path.is_file():
        raise FileNotFoundError("completed run manifest and summary JSON are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if manifest.get("run_id") != run_id or result.get("run_id") != run_id or result.get("status") != "COMPLETED_EXTERNAL_DEVELOPMENT_FIXED_BASELINES":
        raise ValueError("run identity/status does not match the frozen external-development run")
    if manifest.get("gpu_hours") != 0.0 or manifest.get("pdo_outer_locked_values_accessed") is not False:
        raise ValueError("manifest violates the registered resource/data boundary")
    scope = manifest.get("target_numeric_matrix_scope", {})
    if scope.get("H5AD_layers_values_read") is not False or scope.get("other_H5AD_X_or_control_rows_read") != 0:
        raise ValueError("manifest reports an access beyond frozen target/control rows")
    for name, expected in manifest.get("output_sha256", {}).items():
        path = out / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"manifest output checksum mismatch: {name}")

    included = pd.read_csv(out / "included_rows.tsv", sep="\t", dtype=str, keep_default_na=False)
    genes = (out / "metadata_candidate_genes.txt").read_text(encoding="utf-8").splitlines()
    required_key = str(config["data"]["required_key_value"])
    if len(included) != 11 or included["cell_iname"].nunique() != 11 or included["full_inchikey"].tolist() != [required_key] * 11 or len(genes) != 978:
        raise ValueError("the frozen 11-object/full-key/978-gene domain changed")
    row_ids = included["row_index"].astype(int).to_numpy()
    candidate = ROOT / config["data"]["candidate_path"]
    if _sha256(candidate) != manifest.get("target_h5ad_sha256"):
        raise ValueError("numeric H5AD bytes differ from the run manifest")
    with h5py.File(candidate, "r") as h5:
        # Deliberately index only the 11 frozen X/control rows; layers are never opened.
        x = _selected_rows(h5["X"], row_ids)
        ctl = _selected_rows(h5["obsm"]["X_ctl"], row_ids)
    target = x - ctl

    train_npz = ROOT / config["data"]["source_train_values_npz"]
    identity_path = ROOT / config["data"]["source_train_identity_table"]
    identity = pd.read_csv(identity_path, sep="\t", dtype=str, keep_default_na=False)
    train_meta = identity.loc[identity["split_role"].str.lower().eq("train")].copy()
    if len(train_meta) != 419 or train_meta["source_object_id"].nunique() != 110 or train_meta["full_inchikey"].nunique() != 5:
        raise ValueError("source TRAIN identity denominator changed")
    with np.load(train_npz, allow_pickle=False) as archive:
        keys = archive["train_condition_key"].astype(str)
        source_labels = archive["genes"].astype(str).tolist()
        source_delta_original = archive["train_target_delta"].astype(np.float64)
        if len(keys) != 419 or len(set(keys.tolist())) != 419 or source_delta_original.shape != (419, 978):
            raise ValueError("source archive TRAIN-only arrays have unexpected dimensions")
    if set(keys.tolist()) != set(train_meta["full_condition_key"].tolist()):
        raise ValueError("TRAIN target keys do not equal response-blind TRAIN metadata keys")
    reindex = _axis_reindex(source_labels, genes, ROOT / config["data"]["source_gene_axis_mapping"], ROOT / config["data"]["official_gene_info"])
    train_delta = source_delta_original[:, reindex]
    ordered_meta = train_meta.set_index("full_condition_key", verify_integrity=True).loc[keys.tolist()].reset_index()
    if not np.isfinite(train_delta).all():
        raise FloatingPointError("source TRAIN target array contains non-finite values")
    global_pred = train_delta.mean(axis=0, dtype=np.float64).astype(np.float32).astype(np.float64)
    key_mask = ordered_meta["full_inchikey"].astype(str).to_numpy() == required_key
    if int(key_mask.sum()) == 0:
        raise ValueError("exact full InChIKey is absent from source TRAIN")
    exact_key_pred = train_delta[key_mask].mean(axis=0, dtype=np.float64).astype(np.float32).astype(np.float64)
    zero_pred = np.zeros(978, dtype=np.float64)

    npz_path = out / "keyed_baseline_predictions.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        if any(archive[name].dtype.kind == "O" for name in archive.files):
            raise TypeError("keyed baseline NPZ contains a pickle/object array")
        saved_target = archive["target_log2_tpm_plus1_delta"].astype(np.float64)
        saved_global = archive["source_train_global_mean_delta"].astype(np.float64)
        saved_key = archive["source_train_exact_full_inchikey_mean_delta"].astype(np.float64)
        saved_zero = archive["zero_change"].astype(np.float64)
        keys_match = np.array_equal(archive["observation_key"].astype(str), included["obs_name"].astype(str).to_numpy())
        genes_match = np.array_equal(archive["genes"].astype(str), np.asarray(genes, dtype=str))
    if saved_target.shape != (11, 978) or not np.array_equal(saved_target, target):
        raise ValueError("saved target deltas do not equal root-selected treatment minus paired-control values")
    expected_global = np.broadcast_to(global_pred, (11, 978))
    expected_key = np.broadcast_to(exact_key_pred, (11, 978))
    if not (keys_match and genes_match and np.array_equal(saved_global, expected_global) and np.array_equal(saved_key, expected_key) and np.array_equal(saved_zero, np.zeros((11, 978)))):
        raise ValueError("keyed baseline predictions/identity/gene axis fail independent TRAIN recomputation")

    gmt = ROOT / config["functional_sets"]["path"]
    if _sha256(gmt) != config["functional_sets"]["sha256"]:
        raise ValueError("registered Hallmark GMT checksum changed")
    gene_idx = {gene: i for i, gene in enumerate(genes)}
    hallmark_sets: dict[str, np.ndarray] = {}
    for line in gmt.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        idx = sorted({gene_idx[s] for s in fields[2:] if s in gene_idx})
        if fields[0] in hallmark_sets or not idx:
            raise ValueError(f"invalid frozen Hallmark set {fields[0]}")
        hallmark_sets[fields[0]] = np.asarray(idx, dtype=np.int64)
    if len(hallmark_sets) != 50:
        raise ValueError("fixed functional denominator is not 50 Hallmark sets")

    methods = {
        "source_train_global_mean_delta": global_pred,
        "source_train_exact_full_inchikey_mean_delta": exact_key_pred,
        "zero_change": zero_pred,
    }
    condition_table = pd.read_csv(out / "per_condition_metrics.tsv", sep="\t")
    object_table = pd.read_csv(out / "per_object_metrics.tsv", sep="\t")
    drug_table = pd.read_csv(out / "per_drug_metrics.tsv", sep="\t")
    summary_table = pd.read_csv(out / "summary_metrics.tsv", sep="\t")
    paired_table = pd.read_csv(out / "paired_object_bootstrap.tsv", sep="\t")
    condition_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    object_values: dict[str, dict[str, list[float]]] = {m: {metric: [] for metric in ("gene_direction_agreement", "gene_spearman", "hallmark_direction_agreement")} for m in methods}
    # Match the registered table's deterministic object order for reproducible
    # seeded bootstrap draws; condition scoring itself remains keyed to row IDs.
    object_order = sorted(range(len(included)), key=lambda i: (str(included.iloc[i]["cell_iname"]), str(included.iloc[i]["full_inchikey"])))
    for i in object_order:
        row = included.iloc[i]
        for method, prediction in methods.items():
            metrics = _condition_metrics(target[i], prediction, hallmark_sets)
            condition_rows.append({"condition_row_index": int(row["row_index"]), "cell_line": row["cell_iname"], "full_inchikey": row["full_inchikey"], "baseline": method, **metrics})
            object_rows.append({"cell_line": row["cell_iname"], "full_inchikey": row["full_inchikey"], "baseline": method, **metrics})
            for metric in object_values[method]:
                value = metrics[metric]
                object_values[method][metric].append(float(value) if value is not None else np.nan)
    condition_check = pd.DataFrame(condition_rows)
    for _, expected in condition_check.iterrows():
        row = condition_table.loc[(condition_table["condition_row_index"] == expected["condition_row_index"]) & condition_table["baseline"].eq(expected["baseline"])].iloc[0]
        for column in ("gene_direction_agreement", "gene_spearman", "hallmark_direction_agreement"):
            actual = None if pd.isna(expected[column]) else float(expected[column])
            recorded = None if pd.isna(row[column]) else float(row[column])
            if not _close(actual, recorded):
                raise ValueError(f"condition-level numeric mismatch for {column} at row {expected['condition_row_index']}, {expected['baseline']}")
    object_check = pd.DataFrame(object_rows)
    if len(object_check) != 33 or len(object_table) != 33:
        raise ValueError("per-object metric table does not contain the frozen 11 objects by 3 baselines")
    object_columns = ("gene_direction_agreement", "gene_spearman", "hallmark_direction_agreement")
    for _, expected in object_check.iterrows():
        mask = (
            object_table["cell_line"].astype(str).eq(str(expected["cell_line"]))
            & object_table["full_inchikey"].astype(str).eq(str(expected["full_inchikey"]))
            & object_table["baseline"].astype(str).eq(str(expected["baseline"]))
        )
        row = object_table.loc[mask]
        if len(row) != 1:
            raise ValueError("per-object key set is not a one-to-one match with independent recomputation")
        row = row.iloc[0]
        for column in object_columns:
            actual = None if pd.isna(expected[column]) else float(expected[column])
            recorded = None if pd.isna(row[column]) else float(row[column])
            if not _close(actual, recorded):
                raise ValueError(f"per-object metric mismatch for {expected['cell_line']}/{expected['baseline']}/{column}")
    if len(drug_table) != 3 or drug_table["full_inchikey"].astype(str).nunique() != 1:
        raise ValueError("per-drug table does not match the one exact shared key and three baselines")
    for method in methods:
        row = drug_table.loc[drug_table["baseline"].astype(str).eq(method)]
        if len(row) != 1 or int(row.iloc[0]["n_objects"]) != 11:
            raise ValueError(f"per-drug object denominator mismatch for {method}")
        for column in object_columns:
            vals = np.asarray(object_values[method][column], dtype=np.float64)
            point = float(np.nanmean(vals)) if np.isfinite(vals).any() else None
            recorded = None if pd.isna(row.iloc[0][column]) else float(row.iloc[0][column])
            if not _close(point, recorded):
                raise ValueError(f"per-drug mean mismatch for {method}/{column}")

    reps = int(config["metrics"]["bootstrap_replicates"])
    seed_base = int(config["metrics"]["bootstrap_seed"])
    metric_order = ["gene_direction_agreement", "gene_spearman", "hallmark_direction_agreement"]
    summary_check: dict[tuple[str, str], tuple[float | None, float | None, float | None, int]] = {}
    for method_i, method in enumerate(methods):
        for metric_i, metric in enumerate(metric_order):
            vals = np.asarray(object_values[method][metric], dtype=np.float64)
            summary_check[(method, metric)] = _bootstrap(vals, seed_base + method_i * 100 + metric_i, reps)
            row = summary_table.loc[summary_table["baseline"].eq(method) & summary_table["metric"].eq(metric)].iloc[0]
            for col, value in zip(("object_macro_mean", "object_bootstrap_95ci_low", "object_bootstrap_95ci_high"), summary_check[(method, metric)][:3]):
                recorded = None if pd.isna(row[col]) else float(row[col])
                if not _close(value, recorded):
                    raise ValueError(f"summary metric mismatch for {method}/{metric}/{col}: recomputed={value!r}, recorded={recorded!r}")
            if int(row["valid_objects"]) != summary_check[(method, metric)][3] or int(row["total_objects"]) != 11:
                raise ValueError(f"summary denominator mismatch for {method}/{metric}")

    pairs = [
        ("source_train_global_mean_delta", "source_train_exact_full_inchikey_mean_delta"),
        ("source_train_global_mean_delta", "zero_change"),
        ("source_train_exact_full_inchikey_mean_delta", "zero_change"),
    ]
    paired_check: list[dict[str, Any]] = []
    for metric_i, metric in enumerate(metric_order):
        for pair_i, (left, right) in enumerate(pairs):
            pair_result = _paired_bootstrap(np.asarray(object_values[left][metric]), np.asarray(object_values[right][metric]), seed_base + 1000 + metric_i * 100 + pair_i, reps)
            paired_check.append({
                "metric": metric,
                "left_baseline": left,
                "right_baseline": right,
                "difference_left_minus_right": pair_result[0],
                "paired_object_bootstrap_95ci_low": pair_result[1],
                "paired_object_bootstrap_95ci_high": pair_result[2],
                "valid_paired_objects": pair_result[3],
                "total_paired_objects": 11,
            })
            row = paired_table.loc[paired_table["metric"].eq(metric) & paired_table["left_baseline"].eq(left) & paired_table["right_baseline"].eq(right)].iloc[0]
            for col, value in zip(("difference_left_minus_right", "paired_object_bootstrap_95ci_low", "paired_object_bootstrap_95ci_high"), pair_result[:3]):
                recorded = None if pd.isna(row[col]) else float(row[col])
                if not _close(value, recorded):
                    raise ValueError(f"paired bootstrap mismatch for {metric}/{left}-{right}/{col}")
            if int(row["valid_paired_objects"]) != pair_result[3] or int(row["total_paired_objects"]) != 11 or bool(row["p_value_computed"]):
                raise ValueError(f"paired bootstrap denominator/inference mismatch for {metric}/{left}-{right}")

    return {
        "experiment_id": "EXP-057",
        "run_id": run_id,
        "status": "ROOT_INDEPENDENT_NUMERICAL_READBACK_PASS",
        "verification_method": "independent NumPy/pandas/h5py/SciPy recomputation; evaluator implementation not imported",
        "source_train_numeric_scope": {
            "npz_sha256": _sha256(train_npz),
            "identity_table_sha256": _sha256(identity_path),
            "arrays_loaded": ["train_condition_key", "genes", "train_target_delta"],
            "conditions": 419,
            "objects": 110,
            "full_inchikeys": 5,
            "inner_dev_or_other_roles_loaded": False,
            "source_gene_axis_ordered_match": True,
            "exact_full_key_train_rows": int(key_mask.sum()),
        },
        "target_numeric_scope": {
            "h5ad_sha256": _sha256(candidate),
            "treatment_rows_read": sorted(int(x) for x in row_ids),
            "matched_control_rows_read": sorted(int(x) for x in row_ids),
            "other_h5ad_rows_read": 0,
            "layers_read": False,
            "pdo_outer_locked_values_read": False,
            "target_shape": list(target.shape),
            "target_is_rowwise_treatment_minus_control": True,
        },
        "output_integrity": {name: _sha256(out / name) for name in manifest["output_sha256"]},
        "recomputed_summary_metrics": [
            {
                "baseline": method,
                "metric": metric,
                "object_macro_mean": summary_check[(method, metric)][0],
                "object_bootstrap_95ci_low": summary_check[(method, metric)][1],
                "object_bootstrap_95ci_high": summary_check[(method, metric)][2],
                "valid_objects": summary_check[(method, metric)][3],
                "total_objects": 11,
            }
            for method in methods for metric in metric_order
        ],
        "recomputed_paired_differences": paired_check,
        "checks": {
            "manifest_outputs_match": True,
            "keyed_predictions_match_independent_train_means": True,
            "keyed_targets_match_selected_h5ad_rows": True,
            "condition_metrics_recomputed": int(len(condition_check)),
            "condition_metrics_match": True,
            "per_object_metrics_recomputed": int(len(object_check)),
            "per_object_and_per_drug_metrics_match": True,
            "summary_and_bootstrap_recomputed": int(len(summary_check)),
            "paired_differences_and_bootstrap_recomputed": int(len(metric_order) * len(pairs)),
            "summary_tables_match": True,
            "zero_change_spearman_undefined_retained": True,
            "gpu_hours": 0.0,
        },
        "interpretation": "External development only: 11 target objects and one exact drug key. Direction/rank/function agreement is weak; global mean is only marginally higher on direction and Hallmark direction than the exact-key mean, while rank is marginally lower. No independent test or PDO efficacy claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    receipt = _verify(args.config.resolve(), None)
    out = ROOT / "artifacts/experiments/EXP-057" / receipt["run_id"]
    receipt_path = out / "root_independent_numeric_readback_v2.json"
    temporary = receipt_path.with_name(receipt_path.name + ".part")
    if receipt_path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite root receipt: {receipt_path}")
    receipt["verifier_script_sha256"] = _sha256(Path(__file__).resolve())
    receipt["created_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, receipt_path)
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path), "sha256": _sha256(receipt_path), "checks": receipt["checks"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
