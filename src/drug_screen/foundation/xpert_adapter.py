"""Thin DrugScreenLab-to-XPert data adapter and foundation readiness checks.

The official XPert source remains external and unmodified.  This module only
translates the already-registered DrugScreenLab exact978 manifest into XPert's
published paired-h5ad contract and records a bounded readiness decision.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


XPERT_REQUIRED_OBS = (
    "pert_id",
    "pert_idx",
    "cell_iname",
    "cell_idx",
    "tissue_idx",
    "pert_dose",
    "pert_time",
)


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return tuple(np.asarray(value).shape)
    return tuple(int(item) for item in shape)


def _obs_columns(obs: Any) -> set[str]:
    columns = getattr(obs, "columns", None)
    if columns is not None:
        return {str(value) for value in columns}
    if isinstance(obs, Mapping):
        return {str(value) for value in obs}
    raise TypeError("adata.obs must expose columns or mapping keys")


def validate_xpert_contract(adata: Any) -> dict[str, Any]:
    """Validate the official XPert paired h5ad interface without changing it."""

    x_shape = _shape(adata.X)
    if len(x_shape) != 2 or x_shape[1] != 978:
        raise ValueError(f"XPert requires adata.X with shape [n, 978], got {x_shape}")
    if "X_ctl" not in adata.obsm:
        raise ValueError('XPert requires matched controls in adata.obsm["X_ctl"]')
    ctl_shape = _shape(adata.obsm["X_ctl"])
    if ctl_shape != x_shape:
        raise ValueError(f'adata.obsm["X_ctl"] must match adata.X shape {x_shape}, got {ctl_shape}')
    missing = sorted(set(XPERT_REQUIRED_OBS).difference(_obs_columns(adata.obs)))
    if missing:
        raise ValueError(f"XPert metadata missing required obs columns: {missing}")
    var_names = getattr(adata, "var_names", None)
    if var_names is not None and len(var_names) != 978:
        raise ValueError(f"XPert requires 978 ordered var_names, got {len(var_names)}")
    return {
        "format": "xpert_paired_h5ad_contract_v1",
        "record_count": int(x_shape[0]),
        "gene_count": int(x_shape[1]),
        "required_obsm": ["X_ctl"],
        "required_obs": list(XPERT_REQUIRED_OBS),
        "ordered_gene_count": int(len(var_names)) if var_names is not None else 978,
    }


def classify_foundation_metrics(
    metrics: Mapping[str, Mapping[str, Any]],
    *,
    minimum_positive_pearson: float = 0.05,
    minimum_prediction_std: float = 1e-6,
) -> str:
    """Return the closed foundation status from one cold-cell and one cold-drug run."""

    for split in ("cold_cell", "cold_drug"):
        row = metrics.get(split)
        if row is None:
            return "BROKEN"
        try:
            pearson = float(row["pearson_delta"])
            prediction_std = float(row["prediction_std"])
            n_test = int(row["n_test"])
        except (KeyError, TypeError, ValueError):
            return "BROKEN"
        if not np.isfinite(pearson) or not np.isfinite(prediction_std):
            return "BROKEN"
        if n_test <= 0 or pearson <= minimum_positive_pearson or prediction_std <= minimum_prediction_std:
            return "BROKEN"
    return "XPERT_FOUNDATION_READY"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_drug_map(drug_info: pd.DataFrame) -> dict[str, int]:
    required = {"pert_id", "drug_node_idx"}
    missing = required.difference(drug_info.columns)
    if missing:
        raise ValueError(f"XPert drug info missing columns: {sorted(missing)}")
    mapping: dict[str, int] = {}
    for pert_id, group in drug_info.groupby("pert_id", sort=False):
        indices = {int(value) for value in group["drug_node_idx"].dropna().tolist()}
        if len(indices) != 1:
            raise ValueError(f"XPert drug index is ambiguous for {pert_id}: {sorted(indices)}")
        mapping[str(pert_id)] = indices.pop()
    return mapping


def build_phase1_adapter_h5ad(
    *,
    manifest_path: Path | str,
    cache_path: Path | str,
    drug_info_path: Path | str,
    gene_info_path: Path | str,
    output_path: Path | str,
    drug_ids: Iterable[str] | None = None,
    splits: Sequence[str] | None = None,
    max_records: int | None = None,
) -> dict[str, Any]:
    """Materialize a bounded paired h5ad adapter from immutable exact978 rows."""

    try:
        import anndata as ad
    except ImportError as error:  # pragma: no cover - runtime environment contract
        raise RuntimeError("anndata is required to build the XPert adapter") from error

    manifest_file = Path(manifest_path)
    cache_file = Path(cache_path)
    drug_info_file = Path(drug_info_path)
    gene_info_file = Path(gene_info_path)
    output_file = Path(output_path)
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    records = list(payload.get("records", []))
    allowed_drugs = None if drug_ids is None else {str(value) for value in drug_ids}
    allowed_splits = None if splits is None else {str(value) for value in splits}
    selected = [
        row
        for row in records
        if (allowed_drugs is None or str(row["drug_id"]) in allowed_drugs)
        and (allowed_splits is None or str(row["split"]) in allowed_splits)
    ]
    selected.sort(key=lambda row: (str(row.get("sample_id", "")), int(row["treatment_cache_row"])))
    if max_records is not None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        selected = selected[:max_records]
    if not selected:
        raise ValueError("XPert adapter selection is empty")

    cache = np.load(cache_file, mmap_mode="r")
    if cache.ndim != 2 or cache.shape[1] != 978:
        raise ValueError(f"exact978 cache must have shape [n, 978], got {cache.shape}")
    treatment_rows = np.asarray([int(row["treatment_cache_row"]) for row in selected], dtype=np.int64)
    control_rows = np.asarray([int(row["control_cache_row"]) for row in selected], dtype=np.int64)
    if treatment_rows.max(initial=-1) >= cache.shape[0] or control_rows.max(initial=-1) >= cache.shape[0]:
        raise ValueError("XPert adapter row index exceeds exact978 cache")
    treated = np.asarray(cache[treatment_rows], dtype=np.float32)
    controls = np.asarray(cache[control_rows], dtype=np.float32)
    if not np.isfinite(treated).all() or not np.isfinite(controls).all():
        raise ValueError("XPert adapter contains non-finite exact978 values")

    drug_info = pd.read_csv(drug_info_file)
    drug_map = _unique_drug_map(drug_info)
    missing_drugs = sorted({str(row["drug_id"]) for row in selected}.difference(drug_map))
    if missing_drugs:
        raise ValueError(f"XPert official drug map lacks selected DrugScreenLab IDs: {missing_drugs}")

    gene_info = pd.read_csv(gene_info_file)
    if len(gene_info) != 978:
        raise ValueError(f"XPert gene info must contain 978 ordered genes, got {len(gene_info)}")
    gene_column = "gene_id" if "gene_id" in gene_info.columns else "gene_name"
    gene_ids = gene_info[gene_column].astype(str).tolist()
    contexts = sorted({str(row["context_id"]) for row in selected})
    context_map = {context: index for index, context in enumerate(contexts)}
    if len(context_map) > 240:
        raise ValueError("XPert l1000 cell index contract supports at most 240 contexts")

    obs = pd.DataFrame(
        {
            "sample_id": [str(row.get("sample_id", "")) for row in selected],
            "pert_id": [str(row["drug_id"]) for row in selected],
            "pert_idx": [drug_map[str(row["drug_id"])] for row in selected],
            "cell_iname": [str(row["context_id"]) for row in selected],
            "cell_idx": [context_map[str(row["context_id"])] for row in selected],
            "tissue_idx": [0] * len(selected),
            "pert_dose": [float(row["dose_um"]) for row in selected],
            "pert_time": [float(row["time_h"]) for row in selected],
            "split": [str(row.get("split", "infer")) for row in selected],
            "pert_type": ["trt_cp"] * len(selected),
            "control_cache_row": control_rows,
            "treatment_cache_row": treatment_rows,
        },
        index=[str(row.get("sample_id", index)) for index, row in enumerate(selected)],
    )
    var = pd.DataFrame(index=pd.Index(gene_ids, name="gene_id"))
    adata = ad.AnnData(X=treated, obs=obs, var=var)
    adata.obsm["X_ctl"] = controls
    contract = validate_xpert_contract(adata)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_file)
    return {
        **contract,
        "output_path": str(output_file),
        "output_sha256": _file_sha256(output_file),
        "source_manifest": str(manifest_file),
        "source_manifest_sha256": _file_sha256(manifest_file),
        "source_cache": str(cache_file),
        "source_cache_sha256": payload.get("cache", {}).get("sha256"),
        "drug_count": int(obs["pert_id"].nunique()),
        "context_count": int(obs["cell_iname"].nunique()),
        "split_counts": {str(key): int(value) for key, value in obs["split"].value_counts().sort_index().items()},
        "gene_id_column": gene_column,
        "tissue_idx_policy": "0=unknown adapter placeholder; XPert prediction path does not use tissue_idx as a learned input",
    }


def build_cartesian_adapter_h5ad(
    *,
    source_path: Path | str,
    output_path: Path | str,
    drug_ids: Sequence[str],
) -> dict[str, Any]:
    """Expand one matched control context to every requested drug for inference.

    This is deliberately an inference-only adapter.  ``X`` is copied from the
    matched control as a placeholder because XPert's forward pass consumes the
    control/context tensor and drug features; no placeholder row is used as a
    training target or as an evaluation label.
    """

    try:
        import anndata as ad
    except ImportError as error:  # pragma: no cover - runtime environment contract
        raise RuntimeError("anndata is required to build the XPert adapter") from error

    requested = [str(value) for value in drug_ids]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("drug_ids must be a non-empty unique sequence")
    source_file = Path(source_path)
    output_file = Path(output_path)
    source = ad.read_h5ad(source_file)
    source_audit = validate_xpert_contract(source)
    frame = source.obs.reset_index(drop=True).copy()
    mapping_frame = frame[["pert_id", "pert_idx"]].drop_duplicates("pert_id")
    drug_map = {str(row.pert_id): int(row.pert_idx) for row in mapping_frame.itertuples(index=False)}
    missing = sorted(set(requested).difference(drug_map))
    if missing:
        raise ValueError(f"source adapter lacks requested drugs: {missing}")

    # One real matched-control row per context is enough to provide an
    # individualized context.  Keep the first row in deterministic sample
    # order so repeated runs produce identical adapter bytes.
    frame["source_row"] = np.arange(len(frame), dtype=np.int64)
    representatives = frame.sort_values(["cell_iname", "source_row"], kind="mergesort").drop_duplicates("cell_iname")
    control = np.asarray(source.obsm["X_ctl"], dtype=np.float32)
    rows: list[dict[str, Any]] = []
    x_rows: list[np.ndarray] = []
    ctl_rows: list[np.ndarray] = []
    for representative in representatives.itertuples(index=False):
        source_row = int(representative.source_row)
        for drug_id in requested:
            rows.append(
                {
                    "sample_id": f"xpert_prism_context:{representative.cell_iname}:{drug_id}",
                    "pert_id": drug_id,
                    "pert_idx": drug_map[drug_id],
                    "cell_iname": str(representative.cell_iname),
                    "cell_idx": int(representative.cell_idx),
                    "tissue_idx": int(representative.tissue_idx),
                    "pert_dose": float(representative.pert_dose),
                    "pert_time": float(representative.pert_time),
                    "split": "infer",
                    # The official XPert loader selects the inference rows by
                    # the requested nfold column.  Mark every Cartesian row
                    # as test so the adapter remains usable with the
                    # unmodified official test path.
                    "split_1": "test",
                    "pert_type": "trt_cp",
                    "context_source": "exact_lincs_context",
                    "inference_target_policy": "control_placeholder_not_label",
                }
            )
            x_rows.append(control[source_row])
            ctl_rows.append(control[source_row])

    obs = pd.DataFrame(rows)
    obs.index = obs["sample_id"].astype(str)
    var = source.var.copy()
    result = ad.AnnData(X=np.stack(x_rows).astype(np.float32), obs=obs, var=var)
    result.obsm["X_ctl"] = np.stack(ctl_rows).astype(np.float32)
    result.uns["xpert_adapter"] = {
        "format": "xpert_inference_cartesian_adapter_v1",
        "source_path": str(source_file),
        "drug_ids": requested,
        "control_policy": "exact_lincs_context",
        "target_policy": "X copied from matched control only as an inference placeholder; never a training/evaluation label",
    }
    contract = validate_xpert_contract(result)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(output_file)
    return {
        **contract,
        "output_path": str(output_file),
        "source_path": str(source_file),
        "source_record_count": source_audit["record_count"],
        "drug_ids": requested,
        "drug_count": len(requested),
        "context_count": int(obs["cell_iname"].nunique()),
        "placeholder_target_policy": "control_placeholder_not_label",
    }
