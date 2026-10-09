#!/usr/bin/env python3
"""Descriptive GEX-only summary for the registered GSE288798 Seurat RDS.

This CPU-only intake uses the author's existing RNA.data layer and existing
cell labels. Cells are coverage units, never biological replicates. No ATAC,
gene-activity matrix, inferential statistics, efficacy labels, ranking, or model
selection is performed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import resource
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import rdata
import scipy
from scipy.sparse import csc_matrix


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20261001_gse288798_expression_only_v1"
OUT = ROOT / "artifacts/experiments/EXP-050" / RUN_ID
INPUT = ROOT / (
    "data/raw/transfer_public_assets/gse_breast_drug_response/GSE288798/"
    "GSE288798_Organoid_TNBC_JQ1_MS177_24hr_3_conditions_with_gene_activity_"
    "with_cluster_identities_with_doublets_classification_01082025.rds.gz"
)
GMT = ROOT / "data/raw/exp013/msigdb/h.all.v2023.2.Hs.symbols.gmt"
INTAKE = OUT / "intake_summary.json"
MANIFEST = OUT / "run_manifest.json"
GENE_OUT = OUT / "cluster_expression_summary.parquet"
PATHWAY_OUT = OUT / "descriptive_pathway_scores.tsv"
LOG = OUT / "analysis.log"
EXPECTED_INPUT_SHA256 = "de3aa029bac204fe9b78c911f2395f569e4ffe75b51fc22fceb3f62384d132ae"
EXPECTED_GMT_SHA256 = "8ff1f036f0988c99d42cf9e5ea24d355f10b8ece5e0d4f6edf26a0458f480780"
MAX_ADDRESS_SPACE = 96 * 1024**3
OFFICIAL_GEX_GSM = {
    "control": "GSM8775686",
    "jq1": "GSM8775688",
    "ms177": "GSM8775690",
}
OFFICIAL_METADATA = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE288798"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"{now_iso()} {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as stream:
        temp = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


def treatment_from_label(value: Any) -> str:
    label = str(value).casefold()
    matches = [
        treatment
        for token, treatment in (("ms177", "ms177"), ("jq1", "jq1"), ("dmso", "control"), ("control", "control"))
        if token in label
    ]
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise ValueError(f"condition label does not identify exactly one treatment: {value!r}")
    return matches[0]


def load_hallmark(gene_names: np.ndarray) -> tuple[dict[str, set[str]], dict[str, Any]]:
    observed_sha = sha256_file(GMT)
    if observed_sha != EXPECTED_GMT_SHA256:
        raise ValueError(f"registered Hallmark checksum mismatch: {observed_sha}")
    pathways: dict[str, set[str]] = {}
    for line_no, line in enumerate(GMT.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split("\t")
        if len(fields) < 3 or fields[0] in pathways:
            raise ValueError(f"invalid/duplicate Hallmark GMT row {line_no}")
        pathways[fields[0]] = {gene.strip().upper() for gene in fields[2:] if gene.strip()}
    if len(pathways) != 50:
        raise ValueError(f"expected 50 Hallmark sets, found {len(pathways)}")
    symbols = [str(gene).upper() for gene in gene_names]
    if len(set(symbols)) != len(symbols):
        raise ValueError("RNA gene axis has duplicate case-normalized names; exact mapping is ambiguous")
    index = {symbol: i for i, symbol in enumerate(symbols)}
    coverage = {
        name: {"set_size": len(members), "n_matched": len(members.intersection(index))}
        for name, members in pathways.items()
    }
    return pathways, {
        "resource": "MSigDB Hallmark v2023.2.Hs",
        "path": str(GMT.relative_to(ROOT)).replace("\\", "/"),
        "sha256": observed_sha,
        "set_count": len(pathways),
        "method": "mean per-gene RNA.data difference over exact case-normalized symbol members; descriptive only",
        "gene_coverage": coverage,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    resource.setrlimit(resource.RLIMIT_AS, (MAX_ADDRESS_SPACE, MAX_ADDRESS_SPACE))
    start_wall = time.monotonic()
    start_at = now_iso()
    run: dict[str, Any] = {
        "run_id": RUN_ID,
        "status": "RUNNING",
        "started_at": start_at,
        "inputs": {
            "rds": str(INPUT.relative_to(ROOT)).replace("\\", "/"),
            "rds_bytes": INPUT.stat().st_size,
            "rds_sha256_registered": EXPECTED_INPUT_SHA256,
            "hallmark_gmt": str(GMT.relative_to(ROOT)).replace("\\", "/"),
            "hallmark_sha256_registered": EXPECTED_GMT_SHA256,
            "official_sample_metadata": OFFICIAL_METADATA,
        },
        "protocol": {
            "biological_unit": "one TNBC organoid model; cells are coverage only",
            "modality": "RNA.data only; ATAC and gene_activity matrices are not analyzed",
            "condition_assignment": "RDS dataset/long_condition must map one-to-one to the three official GEX GSMs",
            "labels": "existing epithelial_or_stem and wnn_cluster_identity only; no reclustering",
            "statistics": "descriptive means and treatment-minus-control gene-set means; no p/FDR/CI/ranking/model selection",
            "gpu_hours": 0.0,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "rdata": getattr(rdata, "__version__", "unknown"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "r_runtime_used": False,
            "max_address_space_gib": 96,
        },
    }
    atomic_json(MANIFEST, run)
    log("START: verify frozen source checksums before RDS parsing")
    input_sha = sha256_file(INPUT)
    if INPUT.stat().st_size != 2_936_166_213 or input_sha != EXPECTED_INPUT_SHA256:
        raise ValueError("official RDS size/checksum differs from frozen intake")
    log(f"PASS: RDS checksum verified; starting rdata conversion; pid={os.getpid()}")

    conversion_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with gzip.open(INPUT, "rb") as stream:
            parsed = rdata.parser.parse_file(stream)
        converted = rdata.conversion.convert(parsed)
        conversion_warnings = [str(item.message) for item in caught[:20]]
        warning_count = len(caught)
    del parsed
    log("PARSE_DONE: converted registered Seurat RDS; validating RNA keys and sample labels")

    if not isinstance(converted, SimpleNamespace):
        raise TypeError(f"expected Seurat-like namespace, got {type(converted).__name__}")
    assays = getattr(converted, "assays")
    if not isinstance(assays, dict) or "RNA" not in assays:
        raise ValueError("Seurat RDS does not expose the registered RNA assay")
    rna = assays["RNA"]
    data_slot = getattr(rna, "data", None)
    if data_slot is None or not all(hasattr(data_slot, field) for field in ("Dim", "Dimnames", "x", "i", "p")):
        raise ValueError("RNA.data is not a recoverable sparse matrix")

    dims = tuple(int(x) for x in np.asarray(data_slot.Dim).reshape(-1))
    if len(dims) != 2:
        raise ValueError(f"RNA.data has invalid dimensions: {dims}")
    gene_names = np.asarray(data_slot.Dimnames[0]).astype(str)
    cell_names = np.asarray(data_slot.Dimnames[1]).astype(str)
    values = np.asarray(data_slot.x)
    row_indices = np.asarray(data_slot.i)
    col_ptr = np.asarray(data_slot.p)
    if dims != (len(gene_names), len(cell_names)):
        raise ValueError("RNA.data dimensions do not match its named gene/cell axes")
    if len(set(gene_names.tolist())) != len(gene_names) or len(set(cell_names.tolist())) != len(cell_names):
        raise ValueError("RNA.data contains duplicated gene or cell keys")
    if len(values) != len(row_indices) or len(col_ptr) != dims[1] + 1:
        raise ValueError("RNA.data sparse vectors do not match its dimensions")
    if int(col_ptr[0]) != 0 or int(col_ptr[-1]) != len(values) or np.any(np.diff(col_ptr) < 0):
        raise ValueError("RNA.data sparse column pointers are invalid")
    if np.any(row_indices < 0) or np.any(row_indices >= dims[0]):
        raise ValueError("RNA.data sparse row indices are out of range")
    finite_nnz = int(np.isfinite(values).sum())
    if finite_nnz != len(values):
        raise ValueError(f"RNA.data contains {len(values) - finite_nnz} nonfinite stored values")
    matrix = csc_matrix((values, row_indices, col_ptr), shape=dims, copy=False)

    meta = getattr(converted, "meta.data", None)
    if not isinstance(meta, pd.DataFrame):
        raise ValueError("Seurat meta.data did not convert to a DataFrame")
    if not meta.index.is_unique:
        raise ValueError("Seurat meta.data cell index is not unique")
    meta.index = meta.index.astype(str)
    if set(meta.index) != set(cell_names.tolist()):
        raise ValueError(
            f"cell-key mismatch: RNA cells={len(cell_names)}, metadata={len(meta.index)}, "
            f"RNA-only={len(set(cell_names)-set(meta.index))}, metadata-only={len(set(meta.index)-set(cell_names))}"
        )
    meta = meta.loc[cell_names].copy()
    required = {"dataset", "long_condition", "epithelial_or_stem", "wnn_cluster_identity"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"registered source labels missing: {sorted(missing)}")

    dataset_values = meta["dataset"].astype("string")
    long_condition = meta["long_condition"].astype("string")
    if dataset_values.isna().any() or long_condition.isna().any():
        raise ValueError("dataset or long_condition contains missing cell labels")
    condition = long_condition.map(treatment_from_label)
    dataset_map: dict[str, str] = {}
    for dataset in sorted(dataset_values.unique().tolist()):
        conditions = sorted(condition[dataset_values == dataset].unique().tolist())
        if len(conditions) != 1:
            raise ValueError(f"RDS dataset label {dataset!r} spans treatments: {conditions}")
        dataset_map[str(dataset)] = str(conditions[0])
    if set(dataset_map.values()) != set(OFFICIAL_GEX_GSM) or len(dataset_map) != 3:
        raise ValueError(
            "RDS dataset labels do not map one-to-one to control/JQ1/MS177 and the three official GEX GSMs: "
            f"{dataset_map}"
        )
    label_treatment = dataset_values.map(dataset_map)
    if not np.array_equal(label_treatment.to_numpy(dtype=str), condition.to_numpy(dtype=str)):
        raise ValueError("RDS dataset and long_condition labels disagree at the cell level")
    gex_map = {dataset: OFFICIAL_GEX_GSM[treatment] for dataset, treatment in dataset_map.items()}

    epithelial = meta["epithelial_or_stem"].astype("string")
    wnn_cluster = meta["wnn_cluster_identity"].astype("string")
    if epithelial.isna().any() or wnn_cluster.isna().any():
        raise ValueError("author-provided cell group labels contain missing values")
    epi_expected = long_condition.str.extract(r"_(epithelial_like|stem-cell_like)$", expand=False)
    if epi_expected.isna().any():
        raise ValueError("long_condition includes an unrecognized author cell class")
    # Verify the two metadata columns describe the same existing broad cell class.
    epi_normalized = epithelial.str.replace("_", "-", regex=False).str.casefold()
    expected_normalized = epi_expected.str.replace("_", "-", regex=False).str.casefold()
    if not np.array_equal(epi_normalized.to_numpy(dtype=str), expected_normalized.to_numpy(dtype=str)):
        raise ValueError("epithelial_or_stem and long_condition cell-class labels disagree")

    pathways, pathway_resource = load_hallmark(gene_names)
    log(f"KEYS_PASS: {dims[0]} RNA genes x {dims[1]} cells; 3 exact treatment labels; matrix nnz={len(values)}")
    log(f"GSM_MAP: {json.dumps(gex_map, ensure_ascii=False, sort_keys=True)}")

    commands = getattr(converted, "commands", {})
    normalize_cmd = commands.get("NormalizeData.RNA") if isinstance(commands, dict) else None
    normalization = "RNA.data layer from deposited Seurat object; normalization provenance not found in NormalizeData.RNA command"
    if normalize_cmd is not None:
        params = getattr(normalize_cmd, "params", {})
        if isinstance(params, dict):
            normalization = {
                str(key): scalar_text(value)
                for key, value in params.items()
                if key in {"normalization.method", "scale.factor", "margin"}
            }
    cluster_fields = [
        ("epithelial_or_stem", epithelial.astype(str).to_numpy()),
        ("wnn_cluster_identity", wnn_cluster.astype(str).to_numpy()),
    ]
    treatments = ["control", "jq1", "ms177"]
    pathway_rows: list[dict[str, Any]] = []
    gene_axis_lookup = {str(name).upper(): idx for idx, name in enumerate(gene_names)}
    parquet_writer = None
    rows_written = 0
    contrast_group_count = 0

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        for cluster_field, cluster_labels in cluster_fields:
            unique_clusters = sorted(set(cluster_labels.tolist()))
            for cluster in unique_clusters:
                cluster_mask = cluster_labels == cluster
                control_mask = cluster_mask & (condition.to_numpy(dtype=str) == "control")
                control_count = int(control_mask.sum())
                control_mean = (
                    np.asarray(matrix[:, control_mask].mean(axis=1)).reshape(-1)
                    if control_count
                    else None
                )
                for treatment in treatments:
                    group_mask = cluster_mask & (condition.to_numpy(dtype=str) == treatment)
                    n_cells = int(group_mask.sum())
                    if not n_cells:
                        continue
                    mean_values = np.asarray(matrix[:, group_mask].mean(axis=1)).reshape(-1)
                    if not np.isfinite(mean_values).all():
                        raise ValueError(f"nonfinite expression mean for {cluster_field}={cluster!r}/{treatment}")
                    matched_control = control_mean if control_count else np.full(len(gene_names), np.nan)
                    delta = (
                        np.zeros(len(gene_names), dtype=np.float64)
                        if treatment == "control" and control_count
                        else mean_values - control_mean
                        if control_count
                        else np.full(len(gene_names), np.nan)
                    )
                    datasets_in_group = sorted(dataset_values[group_mask].unique().tolist())
                    if len(datasets_in_group) != 1:
                        raise ValueError(f"treatment group has ambiguous source datasets: {datasets_in_group}")
                    source_dataset = str(datasets_in_group[0])
                    frame = pd.DataFrame({
                        "gene": gene_names,
                        "cluster_field": cluster_field,
                        "cluster": cluster,
                        "treatment": treatment,
                        "source_dataset": source_dataset,
                        "gsm_id": gex_map[source_dataset],
                        "n_cells_coverage": np.full(len(gene_names), n_cells, dtype=np.int32),
                        "n_control_cells_coverage": np.full(len(gene_names), control_count, dtype=np.int32),
                        "mean_rna_data": mean_values,
                        "control_mean_rna_data": matched_control,
                        "delta_from_control": delta,
                        "contrast_status": "DESCRIPTIVE_MATCHED_CONTROL" if control_count else "NO_MATCHED_CONTROL_CELLS",
                    })
                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    if parquet_writer is None:
                        parquet_writer = pq.ParquetWriter(GENE_OUT, table.schema, compression="zstd")
                    parquet_writer.write_table(table, row_group_size=len(frame))
                    rows_written += len(frame)

                    if treatment != "control":
                        contrast_group_count += 1
                        normalized_delta = delta
                        for name, members in pathways.items():
                            present = sorted(members.intersection(gene_axis_lookup))
                            coverage = len(present) / len(members) if members else 0.0
                            score = (
                                float(np.mean([normalized_delta[gene_axis_lookup[g]] for g in present]))
                                if present and control_count
                                else None
                            )
                            pathway_rows.append({
                                "run_id": RUN_ID,
                                "cluster_field": cluster_field,
                                "cluster": cluster,
                                "treatment": treatment,
                                "contrast": f"{treatment}-control_within_same_{cluster_field}",
                                "gsm_id": gex_map[source_dataset],
                                "n_treatment_cells_coverage": n_cells,
                                "n_control_cells_coverage": control_count,
                                "hallmark": name,
                                "mean_gene_expression_delta": score,
                                "n_set_genes": len(members),
                                "n_genes_on_rna_axis": len(present),
                                "gene_coverage": coverage,
                                "score_status": "DESCRIPTIVE_ONLY" if present and control_count else "NOT_COMPUTED_NO_MATCHED_CONTROL_OR_GENES",
                            })
                log(f"SUMMARIZED: {cluster_field}; {len(unique_clusters)} existing groups; rows={rows_written}")
    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    if parquet_writer is None:
        raise ValueError("no expression summary groups were written")
    pd.DataFrame(pathway_rows).to_csv(PATHWAY_OUT, sep="\t", index=False, na_rep="")

    dataset_counts = dataset_values.value_counts().sort_index().to_dict()
    condition_counts = long_condition.value_counts().sort_index().to_dict()
    cluster_counts = wnn_cluster.value_counts().sort_index().to_dict()
    epi_counts = epithelial.value_counts().sort_index().to_dict()
    normalization_note = normalization
    coverage_values = [item["n_matched"] / item["set_size"] for item in pathway_resource["gene_coverage"].values()]
    compact = {
        "run_id": RUN_ID,
        "status": "DESCRIPTIVE_GEX_SUMMARIES_COMPLETE",
        "asset": str(INPUT.relative_to(ROOT)).replace("\\", "/"),
        "asset_bytes": INPUT.stat().st_size,
        "asset_sha256": input_sha,
        "parser": {"package": "rdata", "version": getattr(rdata, "__version__", "unknown"), "r_runtime_used": False,
                   "conversion_warning_count": warning_count, "conversion_warning_examples": conversion_warnings},
        "rds_class": "Seurat-like S4 object converted through rdata",
        "assays_present_but_not_analyzed": sorted([name for name in assays if name != "RNA"]),
        "selected_assay": {"name": "RNA", "slot": "data", "dimensions_genes_by_cells": list(dims),
                           "stored_values": len(values), "finite_stored_values": finite_nnz,
                           "normalization_provenance": normalization_note},
        "key_validation": {"unique_gene_names": len(gene_names), "unique_cell_names": len(cell_names),
                           "metadata_rows": len(meta), "metadata_cell_keys_exactly_match_rna_axis": True,
                           "gene_axis_casefold_unique": len(set(x.upper() for x in gene_names)) == len(gene_names)},
        "biological_unit": "one TNBC organoid model; cell counts are coverage only and not independent n",
        "official_sample_mapping": {
            "source": OFFICIAL_METADATA,
            "official_gex_gsm_by_treatment": {"control": OFFICIAL_GEX_GSM["control"], "JQ1": OFFICIAL_GEX_GSM["jq1"], "MS177": OFFICIAL_GEX_GSM["ms177"]},
            "rds_dataset_to_treatment": dataset_map,
            "rds_dataset_to_gsm": gex_map,
            "dataset_cell_coverage": {str(k): int(v) for k, v in dataset_counts.items()},
            "long_condition_cell_coverage": {str(k): int(v) for k, v in condition_counts.items()},
            "epithelial_or_stem_cell_coverage": {str(k): int(v) for k, v in epi_counts.items()},
            "wnn_cluster_identity_cell_coverage": {str(k): int(v) for k, v in cluster_counts.items()},
            "cell_class_consistency": "epithelial_or_stem matches long_condition suffix for all cells",
        },
        "summaries": {
            "expression_file": str(GENE_OUT.relative_to(ROOT)).replace("\\", "/"),
            "expression_rows": rows_written,
            "group_dimensions": ["epithelial_or_stem", "wnn_cluster_identity"],
            "measure": "mean deposited RNA.data value per group; treatment-control delta within the same existing group",
            "cells_are_independent_n": False,
            "pathway_file": str(PATHWAY_OUT.relative_to(ROOT)).replace("\\", "/"),
            "pathway_rows": len(pathway_rows),
            "pathway_method": pathway_resource["method"],
            "pathway_sets": pathway_resource["set_count"],
            "median_hallmark_gene_coverage": float(np.median(coverage_values)) if coverage_values else None,
            "min_hallmark_gene_coverage": float(np.min(coverage_values)) if coverage_values else None,
            "max_hallmark_gene_coverage": float(np.max(coverage_values)) if coverage_values else None,
            "matched_treatment_group_count": contrast_group_count,
        },
        "scope": "Descriptive within one organoid model only; no cell-level inference, no ATAC analysis, no efficacy prediction or model selection.",
        "gpu_hours": 0.0,
        "wall_seconds": round(time.monotonic() - start_wall, 3),
        "peak_rss_mib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }

    # Preserve the earlier metadata-only structural receipt as an auditable
    # subrecord while replacing its stale RUNNING/PENDING top-level status.
    prior = json.loads(INTAKE.read_text(encoding="utf-8")) if INTAKE.exists() else {}
    if "top_level" in prior:
        compact["structural_probe"] = {
            "status_before_expression_summary": prior.get("status"),
            "parser_wall_seconds": prior.get("parse_wall_seconds"),
            "parser_peak_rss_mib": prior.get("peak_rss_mib"),
            "top_level_type": prior.get("top_level", {}).get("type"),
            "metadata_shape": prior.get("top_level", {}).get("fields", {}).get("meta.data", {}).get("shape"),
        }
    atomic_json(INTAKE, compact)

    run.update({
        "status": "DESCRIPTIVE_GEX_SUMMARIES_COMPLETE",
        "completed_at": now_iso(),
        "wall_seconds": compact["wall_seconds"],
        "peak_rss_mib": compact["peak_rss_mib"],
        "gpu_hours": 0.0,
        "conversion_warning_count": warning_count,
        "official_sample_mapping": compact["official_sample_mapping"],
        "results": compact["summaries"],
        "output_sha256": {
            path.name: sha256_file(path)
            for path in (INTAKE, GENE_OUT, PATHWAY_OUT)
        },
    })
    atomic_json(MANIFEST, run)
    log(f"COMPLETE: {json.dumps(run['results'], ensure_ascii=False, sort_keys=True)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        if MANIFEST.exists():
            try:
                run = json.loads(MANIFEST.read_text(encoding="utf-8"))
            except Exception:
                run = {"run_id": RUN_ID}
        else:
            run = {"run_id": RUN_ID}
        run.update({"status": "FAILED", "failed_at": now_iso(), "error_type": type(exc).__name__, "error": str(exc),
                    "wall_seconds": None, "gpu_hours": 0.0,
                    "peak_rss_mib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)})
        atomic_json(MANIFEST, run)
        log(f"FAILED: {type(exc).__name__}: {exc}")
        raise
