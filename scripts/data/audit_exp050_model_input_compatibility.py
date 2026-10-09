"""Audit target control-expression compatibility without reading treated responses.

This is an input contract diagnostic only. It compares the frozen EXP-050 common
axis in target GSE287429 time-matched control RNA-seq against LINCS Level-3
control profiles from the existing EXP-037 inner-training partition. It does
not predict drug responses or estimate transfer performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP050_RUN = ROOT / "artifacts/experiments/EXP-050/20260930_keyed_response_v7"
EXP037_RUN = ROOT / "artifacts/experiments/EXP-037/20260928_lincs_scale_16_121"
TARGET_MATRIX = ROOT / "data/external/exp010/geo/gse287429/GSE287429_normalized_counts_allsample.txt.gz"
SOURCE_GENES = ROOT / "data/processed/lincs/GSE92742/continuous_contract_v2/genes.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corr(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return None
    left, right = a[mask], b[mask]
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def quantiles(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(values, [0, 0.25, 0.5, 0.75, 0.95, 1])
    return {name: float(value) for name, value in zip(
        ["min", "q25", "median", "q75", "q95", "max"], q)}


def audit(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing model-input audit: {output}")

    manifest = json.loads((EXP050_RUN / "run_manifest.json").read_text(encoding="utf-8"))
    axis_path = EXP050_RUN / "EXP050_common_gene_axis_914.tsv"
    expected_axis = manifest["output_hashes"][axis_path.name]["sha256"]
    if sha256(axis_path) != expected_axis:
        raise ValueError("frozen EXP-050 gene-axis hash does not match the v7 manifest")
    axis = pd.read_csv(axis_path, sep="\t")
    if len(axis) != 914 or axis.axis_index.tolist() != list(range(914)):
        raise ValueError("expected the ordered 914-gene EXP-050 axis")
    if any(axis[column].duplicated().any() for column in
           ["LINCS_Entrez_id", "HGNC_symbol", "GSE287429_Ensembl_gene_id"]):
        raise ValueError("frozen EXP-050 gene-axis identifiers must each be unique")

    source_gene_ids = list(map(str, json.loads(SOURCE_GENES.read_text(encoding="utf-8"))))
    source_position = {gene_id: index for index, gene_id in enumerate(source_gene_ids)}
    if len(source_position) != 978 or not set(axis.LINCS_Entrez_id.astype(str)).issubset(source_position):
        raise ValueError("EXP-050 axis does not bind uniquely to the 978-gene LINCS input axis")
    axis_positions = np.asarray(
        [source_position[gene_id] for gene_id in axis.LINCS_Entrez_id.astype(str)], dtype=np.int64)

    prepared = EXP037_RUN / "prepared"
    prepared_manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    if prepared_manifest.get("status") != "PREPARED_TRAIN_VALIDATION_ONLY":
        raise ValueError("EXP-037 prepared input is not explicitly scoped to train/validation")
    train_rows = np.load(prepared / "train121.npy", allow_pickle=False)
    control = np.load(prepared / "control.npy", mmap_mode="r", allow_pickle=False)
    if control.ndim != 2 or control.shape[1] != 978 or int(len(train_rows)) != int(
            prepared_manifest["n_train121"]):
        raise ValueError("EXP-037 training-control array shape or train-row count mismatch")
    if len(train_rows) == 0 or train_rows.min() < 0 or train_rows.max() >= len(control):
        raise ValueError("EXP-037 training row index is out of bounds")
    source_control_means = np.asarray(control[train_rows].mean(axis=0), dtype=np.float64)[axis_positions]
    if not np.isfinite(source_control_means).all():
        raise ValueError("nonfinite EXP-037 training-side control profile")

    target = pd.read_csv(TARGET_MATRIX, sep=",", index_col=0)
    if target.index.has_duplicates or target.columns.has_duplicates:
        raise ValueError("GSE287429 expression identifiers must be unique")
    target_ids = axis.GSE287429_Ensembl_gene_id.astype(str).tolist()
    if not set(target_ids).issubset(set(target.index.astype(str))):
        raise ValueError("frozen axis target Ensembl IDs are absent from GSE287429 matrix")
    contrasts = {
        "23h_time_matched_control": ["CTR23h_1", "CTR23h_2"],
        "47h_time_matched_control": ["CTR47h_1", "CTR47h_2"],
    }
    rows = []
    for contrast, sample_keys in contrasts.items():
        if not set(sample_keys).issubset(target.columns):
            raise ValueError(f"required target control columns are absent: {contrast}")
        target_control_means = target.loc[target_ids, sample_keys].to_numpy(dtype=np.float64).mean(axis=1)
        if not np.isfinite(target_control_means).all():
            raise ValueError(f"nonfinite GSE287429 control expression: {contrast}")
        source_ranks = pd.Series(source_control_means).rank(method="average").to_numpy(dtype=np.float64)
        target_ranks = pd.Series(target_control_means).rank(method="average").to_numpy(dtype=np.float64)
        rows.append({
            "target_control_contrast": contrast,
            "target_sample_keys": sample_keys,
            "target_control_sample_count": len(sample_keys),
            "gene_count": len(axis),
            "LINCS_training_control_expression_quantiles": quantiles(source_control_means),
            "GSE287429_author_scale_control_expression_quantiles": quantiles(target_control_means),
            "pearson_gene_profile_correlation": corr(source_control_means, target_control_means),
            "spearman_gene_rank_correlation": corr(source_ranks, target_ranks),
            "interpretation": "input-profile diagnostic only; expression units are not established comparable",
        })

    input_paths = [
        EXP050_RUN / "run_manifest.json", axis_path, TARGET_MATRIX, SOURCE_GENES,
        prepared / "manifest.json", prepared / "train121.npy", prepared / "control.npy",
    ]
    input_hashes = {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size,
                    "sha256": sha256(path)} for path in input_paths}
    output.mkdir(parents=True)
    result = {
        "experiment_id": "EXP-050",
        "audit_id": output.name,
        "status": "completed_descriptive_model_input_compatibility_audit",
        "environment": "WSL2 Conda drugscreening-gpu",
        "target_treated_responses_read": False,
        "outer_test_responses_read": False,
        "target_axis": {"n": len(axis), "sha256": expected_axis,
                        "mapping": "frozen EXP-050 914 genes; exact Ensembl and LINCS Entrez IDs"},
        "source_scope": {"run": "EXP-037 20260928_lincs_scale_16_121",
                         "role": "inner training rows only", "control_profile_rows": int(len(train_rows)),
                         "control_input": "prepared raw LINCS Level-3 control expression"},
        "target_scope": {"dataset": "GSE287429", "role": "time-matched CTR samples only",
                         "expression_scale": "author-normalized; transformation not established"},
        "contrasts": rows,
        "inference_boundary": "No model prediction, model score, platform calibration, drug effect, or patient-generalization claim.",
        "input_hashes": input_hashes,
    }
    path = output / "summary.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/experiments/EXP-050/20260930_model_input_audit_v1"))
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    print(json.dumps(audit(output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
