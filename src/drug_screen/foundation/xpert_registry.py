"""Context and global XPert perturbagen registries.

The registries separate identity/feature availability from any source adapter
and make Cartesian inference independent of the drugs present in one h5ad.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


DRUG_REGISTRY_FORMAT = "xpert_drug_registry_v1"
CONTEXT_REGISTRY_FORMAT = "xpert_context_registry_v1"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"registry payload must be an object: {path}")
    return value


def validate_drug_registry_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("format") != DRUG_REGISTRY_FORMAT:
        raise ValueError(f"unexpected drug registry format: {payload.get('format')}")
    drugs = payload.get("drugs")
    if not isinstance(drugs, list) or not drugs:
        raise ValueError("drug registry must contain non-empty drugs")
    ids = [str(row.get("pert_id", "")) for row in drugs if isinstance(row, Mapping)]
    if len(ids) != len(drugs) or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("drug registry pert_id values must be unique and non-empty")


def eligible_registry_drugs(payload: Mapping[str, Any]) -> list[str]:
    validate_drug_registry_payload(payload)
    result = []
    for row in payload["drugs"]:
        if all(bool(row.get(key)) for key in ("unimol", "hg", "kpgt")) and bool(row.get("broad_identity")):
            result.append(str(row["pert_id"]))
    return sorted(result)


def build_context_registry(
    *,
    adapter_path: Path | str,
    prism_response_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    import anndata as ad

    adapter_file = Path(adapter_path)
    prism_file = Path(prism_response_path)
    adapter = ad.read_h5ad(adapter_file, backed="r")
    obs = adapter.obs.reset_index(drop=True).copy()
    prism = pd.read_parquet(prism_file)
    prism["base_context"] = prism["ccle_name"].astype(str).str.split("_").str[0]
    broad_by_context = prism.groupby("base_context", sort=True, observed=True)["depmap_id"].agg(
        lambda values: sorted(set(values.astype(str)))
    )
    rows = []
    for context_id, group in obs.groupby("cell_iname", sort=True, observed=True):
        representative = group.iloc[0]
        rows.append(
            {
                "context_id": str(context_id),
                "context_kind": "cell_line",
                "platform": "LINCS_exact978_control",
                "xpert_cell_idx": int(representative["cell_idx"]),
                "xpert_tissue_idx": int(representative["tissue_idx"]),
                "matched_control_record_count": int(len(group)),
                "representative_dose_um": float(representative["pert_dose"]),
                "representative_time_h": float(representative["pert_time"]),
                "broad_depmap_ids": broad_by_context.get(str(context_id), []),
                "broad_exact_context": str(context_id) in broad_by_context,
                "future_context_adapters": ["CCLE", "DepMap", "organoid", "PDO"],
            }
        )
    payload = {
        "format": CONTEXT_REGISTRY_FORMAT,
        "status": "FROZEN_EXACT_LINCS_CONTEXTS",
        "identity_key": "context_id",
        "control_policy": "one matched exact978 LINCS control per context; no reference-context fallback",
        "source": {
            "adapter": str(adapter_file),
            "adapter_sha256": file_sha256(adapter_file),
            "prism_response": str(prism_file),
            "prism_response_sha256": file_sha256(prism_file),
            "response_values_used_for_context_identity": False,
        },
        "counts": {
            "contexts": len(rows),
            "broad_exact_contexts": sum(bool(row["broad_exact_context"]) for row in rows),
        },
        "contexts": rows,
    }
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_drug_registry(
    *,
    drug_info_path: Path | str,
    unimol_path: Path | str,
    kpgt_path: Path | str,
    hg_path: Path | str,
    bridge_path: Path | str,
    cohort_path: Path | str,
    unipert_path: Path | str | None,
    unipert_available_path: Path | str | None,
    output_path: Path | str,
) -> dict[str, Any]:
    info_file = Path(drug_info_path)
    info = pd.read_csv(info_file, dtype=str, keep_default_na=False)
    required = {"pert_id", "drug_node_idx", "canonical_smiles", "inchi_key"}
    missing = sorted(required.difference(info.columns))
    if missing:
        raise ValueError(f"official XPert drug info missing columns: {missing}")
    info["drug_node_idx"] = pd.to_numeric(info["drug_node_idx"], errors="raise").astype(int)
    unimol_file = Path(unimol_path)
    unimol = np.load(unimol_file, mmap_mode="r", allow_pickle=False)
    kpgt_file = Path(kpgt_path)
    kpgt = np.load(kpgt_file, allow_pickle=True).item()
    hg_file = Path(hg_path)
    hg = np.load(hg_file, mmap_mode="r", allow_pickle=False)
    if unimol.ndim != 3 or unimol.shape[0] <= int(info["drug_node_idx"].max()):
        raise ValueError("official UniMol asset cannot cover XPert drug indices")
    if hg.ndim != 2 or hg.shape[0] <= int(info["drug_node_idx"].max()):
        raise ValueError("official HG asset cannot cover XPert drug indices")
    if not isinstance(kpgt, dict):
        raise ValueError("official KPGT asset must be an index dictionary")

    bridge = pd.read_csv(bridge_path, dtype=str, keep_default_na=False)
    bridge = bridge.loc[bridge["match_status"].eq("MATCHED_IDENTITY")].drop_duplicates("prism_broad_id_base")
    cohort = _load_json(Path(cohort_path))
    broad_ids = set(str(value) for value in cohort.get("eligible_base_ids", []))
    broad_map = dict(zip(bridge["lincs_pert_id"].astype(str), bridge["prism_broad_id_base"].astype(str)))
    unipert = None
    unipert_available = None
    if unipert_path:
        unipert = np.load(Path(unipert_path), mmap_mode="r", allow_pickle=False)
        if unipert.ndim != 2 or unipert.shape[0] <= int(info["drug_node_idx"].max()):
            raise ValueError("UniPert registry feature asset cannot cover XPert drug indices")
        if unipert_available_path:
            unipert_available = np.load(Path(unipert_available_path), mmap_mode="r", allow_pickle=False)
            if unipert_available.ndim != 1 or unipert_available.shape[0] != unipert.shape[0]:
                raise ValueError("UniPert availability mask must align with the UniPert feature asset")

    rows = []
    for pert_id, group in info.groupby("pert_id", sort=True):
        idx = int(group["drug_node_idx"].iloc[0])
        smiles = str(group["canonical_smiles"].iloc[0])
        broad_base = broad_map.get(str(pert_id))
        row = {
            "pert_id": str(pert_id),
            "pert_idx": idx,
            "canonical_smiles": smiles,
            "inchi_key": str(group["inchi_key"].iloc[0]),
            "cmap_name": str(group["cmap_name"].iloc[0]),
            "broad_base_id": broad_base,
            "broad_identity": broad_base in broad_ids if broad_base else False,
            "unimol": bool(np.isfinite(np.asarray(unimol[idx])).all()),
            "hg": bool(np.isfinite(np.asarray(hg[idx])).all()),
            "kpgt": idx in kpgt and bool(np.isfinite(np.asarray(kpgt[idx])).all()),
            "unipert": bool(
                unipert is not None
                and (unipert_available is None or bool(unipert_available[idx]))
                and np.isfinite(np.asarray(unipert[idx])).all()
            ),
        }
        row["global_inference_eligible"] = all(row[key] for key in ("unimol", "hg", "kpgt"))
        row["broad_inference_eligible"] = row["global_inference_eligible"] and row["broad_identity"]
        rows.append(row)
    payload = {
        "format": DRUG_REGISTRY_FORMAT,
        "status": "FROZEN_GLOBAL_XPERT_DRUG_REGISTRY",
        "identity_key": "pert_id",
        "feature_index_key": "pert_idx == official drug_node_idx",
        "source": {
            "drug_info": str(info_file),
            "drug_info_sha256": file_sha256(info_file),
            "unimol": str(unimol_file),
            "unimol_sha256": file_sha256(unimol_file),
            "kpgt": str(kpgt_file),
            "kpgt_sha256": file_sha256(kpgt_file),
            "hg": str(hg_file),
            "hg_sha256": file_sha256(hg_file),
            "bridge": str(bridge_path),
            "cohort": str(cohort_path),
            "unipert": str(unipert_path) if unipert_path else None,
            "unipert_available": str(unipert_available_path) if unipert_available_path else None,
            "response_values_used_for_identity": False,
        },
        "dimensions": {
            "unimol_atoms": int(unimol.shape[1]),
            "unimol_feature_dim": int(unimol.shape[2] - 2),
            "kpgt_dim": 2304,
            "hg_dim": int(hg.shape[1]),
            "unipert_dim": int(unipert.shape[1]) if unipert is not None else 256,
        },
        "counts": {
            "official_pert_ids": len(rows),
            "global_inference_eligible": sum(row["global_inference_eligible"] for row in rows),
            "broad_identity_candidates": sum(row["broad_identity"] for row in rows),
            "broad_inference_eligible": sum(row["broad_inference_eligible"] for row in rows),
            "unipert_feature_complete": sum(row["unipert"] for row in rows),
        },
        "drugs": rows,
    }
    validate_drug_registry_payload(payload)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_global_cartesian_adapter_h5ad(
    *,
    source_path: Path | str,
    registry_path: Path | str,
    output_path: Path | str,
    context_ids: list[str] | None = None,
    broad_only: bool = False,
) -> dict[str, Any]:
    """Expand exact contexts against registry-eligible drugs, not source drugs."""
    import anndata as ad

    source_file = Path(source_path)
    registry_file = Path(registry_path)
    source = ad.read_h5ad(source_file)
    payload = _load_json(registry_file)
    if broad_only:
        selected_ids = {
            str(row["pert_id"])
            for row in payload["drugs"]
            if bool(row.get("broad_inference_eligible"))
        }
    else:
        selected_ids = set(eligible_registry_drugs(payload))
    if context_ids is not None:
        selected_contexts = set(str(value) for value in context_ids)
    else:
        selected_contexts = set(str(value) for value in source.obs["cell_iname"].unique())
    frame = source.obs.reset_index(drop=True).copy()
    frame["source_row"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.loc[frame["cell_iname"].astype(str).isin(selected_contexts)]
    representatives = frame.sort_values(["cell_iname", "source_row"], kind="mergesort").drop_duplicates("cell_iname")
    if representatives.empty or not selected_ids:
        raise ValueError("global Cartesian selection is empty")
    registry_rows = {str(row["pert_id"]): row for row in payload["drugs"] if str(row["pert_id"]) in selected_ids}
    ordered_drugs = sorted(registry_rows)
    control = np.asarray(source.obsm["X_ctl"], dtype=np.float32)
    rows: list[dict[str, Any]] = []
    x_rows: list[np.ndarray] = []
    ctl_rows: list[np.ndarray] = []
    for representative in representatives.itertuples(index=False):
        source_row = int(representative.source_row)
        for pert_id in ordered_drugs:
            registry_row = registry_rows[pert_id]
            rows.append(
                {
                    "sample_id": f"xpert_global_context:{representative.cell_iname}:{pert_id}",
                    "pert_id": pert_id,
                    "pert_idx": int(registry_row["pert_idx"]),
                    "cell_iname": str(representative.cell_iname),
                    "cell_idx": int(representative.cell_idx),
                    "tissue_idx": int(representative.tissue_idx),
                    "pert_dose": float(representative.pert_dose),
                    "pert_time": float(representative.pert_time),
                    "split": "infer",
                    "split_1": "test",
                    "pert_type": "trt_cp",
                    "context_source": "exact_lincs_context_registry",
                    "drug_source": "global_xpert_drug_registry",
                    "inference_target_policy": "control_placeholder_not_label",
                    "broad_base_id": str(registry_row.get("broad_base_id") or ""),
                }
            )
            x_rows.append(control[source_row])
            ctl_rows.append(control[source_row])
    obs = pd.DataFrame(rows)
    obs.index = obs["sample_id"].astype(str)
    result = ad.AnnData(X=np.stack(x_rows).astype(np.float32), obs=obs, var=source.var.copy())
    result.obsm["X_ctl"] = np.stack(ctl_rows).astype(np.float32)
    result.uns["xpert_adapter"] = {
        "format": "xpert_global_cartesian_adapter_v1",
        "source_path": str(source_file),
        "registry_path": str(registry_file),
        "control_policy": "exact_lincs_context_registry",
        "drug_policy": "global_xpert_drug_registry_feature_complete",
        "broad_only": bool(broad_only),
        "target_policy": "X copied from matched control only as an inference placeholder; never a training/evaluation label",
    }
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(output_file)
    return {
        "format": "xpert_global_cartesian_adapter_v1",
        "output_path": str(output_file),
        "source_path": str(source_file),
        "registry_path": str(registry_file),
        "record_count": int(len(obs)),
        "context_count": int(obs["cell_iname"].nunique()),
        "drug_count": int(obs["pert_id"].nunique()),
        "drug_ids": ordered_drugs,
        "broad_only": bool(broad_only),
        "context_ids": sorted(obs["cell_iname"].unique().tolist()),
        "placeholder_target_policy": "control_placeholder_not_label",
    }
