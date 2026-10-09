"""Freeze label-free strict-unseen PRISM inputs for EXP-044."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


EXPECTED_ORTHOGONAL_CELLOSAURUS_CROSSWALK = {
    "SIDM00427": {
        "accession": "CVCL_1658",
        "cell_line_name": "EW-8",
        "depmap_model_ids": {"ACH-000499", "ACH-002194"},
    },
    "SIDM01117": {
        "accession": "CVCL_1588",
        "cell_line_name": "NCI-H748",
        "depmap_model_ids": {"ACH-002176", "ACH-003071"},
    },
}
EXPECTED_ONE_TO_MANY_BRIDGE = {
    "SIDM00117": {"ACH-000837", "ACH-002172"},
}
IDENTITY_RESOLUTION_STATUS = "RESOLVED_ORTHOGONAL_AUTHORITATIVE_CROSSWALK"
STRICT_UNSEEN_INTERPRETATION = "REVISED_STRICT_UNSEEN_AFTER_RESPONSE_BLIND_IDENTITY_CROSSWALK"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def write_failure_receipt(path: Path, value: dict[str, Any]) -> None:
    """Publish a failure receipt atomically without replacing an earlier receipt."""
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".FAILURE.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A hard-link creation is atomic and fails if FAILURE.json already exists.
            os.link(temporary, path)
        except FileExistsError:
            return
        except OSError:
            # Some mounted filesystems do not support hard links. Exclusive creation
            # still guarantees that a prior receipt is never overwritten.
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                return
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
    finally:
        temporary.unlink(missing_ok=True)


def _validated_identity_column(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"{label} identity table lacks {column}")
    values = frame[column]
    if values.isna().any():
        raise ValueError(f"{label} identity column {column} contains missing values")
    values = values.astype(str).str.strip()
    if values.eq("").any():
        raise ValueError(f"{label} identity column {column} contains blank values")
    return values


def audit_identity_bridge(prism_ids: set[str], gdsc_sanger_ids: set[str],
                          model_bridge: pd.DataFrame,
                          external_crosswalk: dict[str, dict[str, Any]] | None = None
                          ) -> tuple[set[str], set[str], dict[str, Any]]:
    """Apply EXP-019's conservative set-union bridge to source Sanger IDs."""
    required = {"ModelID", "SangerModelID"}
    if required - set(model_bridge.columns):
        raise ValueError("DepMap Model.csv is missing its Sanger/Model identity bridge")
    bridge = model_bridge[["ModelID", "SangerModelID"]].copy()
    bridge["ModelID"] = bridge["ModelID"].astype("string").str.strip()
    bridge["SangerModelID"] = bridge["SangerModelID"].astype("string").str.strip()
    relevant = bridge["SangerModelID"].isin(gdsc_sanger_ids)
    matched = bridge.loc[relevant]
    if matched.empty:
        raise ValueError("DepMap bridge maps no GDSC Sanger identities")
    if matched["ModelID"].isna().any() or matched["ModelID"].eq("").any():
        raise ValueError("a GDSC Sanger identity has no DepMap ModelID")
    source_to_models: dict[str, set[str]] = {}
    for source_id, model_id in zip(matched["SangerModelID"], matched["ModelID"]):
        source_to_models.setdefault(str(source_id), set()).add(str(model_id))
    direct_unmapped = sorted(gdsc_sanger_ids - set(source_to_models))
    external_crosswalk = external_crosswalk or {}
    if external_crosswalk and set(external_crosswalk) != set(direct_unmapped):
        raise ValueError("orthogonal crosswalk must resolve exactly every directly unmapped source identity")
    if set(external_crosswalk) != set(EXPECTED_ORTHOGONAL_CELLOSAURUS_CROSSWALK) and external_crosswalk:
        raise ValueError("orthogonal crosswalk source identities differ from the frozen audit findings")
    for source_id, evidence in external_crosswalk.items():
        model_ids = set(map(str, evidence.get("depmap_model_ids", [])))
        expected = EXPECTED_ORTHOGONAL_CELLOSAURUS_CROSSWALK[source_id]
        if (evidence.get("cellosaurus_accession") != expected["accession"]
                or evidence.get("cell_line_name") != expected["cell_line_name"]
                or model_ids != expected["depmap_model_ids"]):
            raise ValueError(f"orthogonal Cellosaurus crosswalk differs from frozen evidence for {source_id}")
        source_to_models[source_id] = model_ids
    mapped = set().union(*source_to_models.values())
    overlap = prism_ids & mapped
    strict = prism_ids - mapped
    if (overlap & strict) or (overlap | strict) != prism_ids:
        raise ValueError("strict-unseen and overlap identity sets do not partition PRISM")
    unmapped = sorted(gdsc_sanger_ids - set(source_to_models))
    one_to_many = {
        source_id: sorted(model_ids)
        for source_id, model_ids in sorted(source_to_models.items())
        if len(model_ids) > 1
    }
    audit = {
        "mapping_policy": "conservative_union_of_DepMap_24Q2_Sanger_bridge_and_authoritative_Cellosaurus_crossrefs",
        "source_sanger_identity_count": len(gdsc_sanger_ids),
        "directly_matched_source_sanger_identity_count": len(source_to_models) - len(external_crosswalk),
        "orthogonally_resolved_source_sanger_ids": sorted(external_crosswalk),
        "unmapped_source_sanger_ids": unmapped,
        "one_to_many_sanger_model_candidates": one_to_many,
        "candidate_depmap_model_id_count": len(mapped),
        "matched_bridge_row_count": len(matched),
        "orthogonal_identity_evidence": external_crosswalk,
        "all_candidate_model_ids_tested_against_prism_axis": True,
    }
    return strict, overlap, audit


def load_cellosaurus_crosswalk(ew8_path: Path, h748_path: Path) -> dict[str, dict[str, Any]]:
    """Read fixed Cellosaurus JSON snapshots and validate exact cross-references."""
    snapshots = {
        "SIDM00427": (ew8_path, "CVCL_1658"),
        "SIDM01117": (h748_path, "CVCL_1588"),
    }
    result: dict[str, dict[str, Any]] = {}
    for source_id, (path, expected_accession) in snapshots.items():
        root = json.loads(path.read_text(encoding="utf-8"))
        entries = root.get("Cellosaurus", {}).get("cell-line-list", [])
        matches = [
            entry for entry in entries
            if any(item.get("type") == "primary" and item.get("value") == expected_accession
                   for item in entry.get("accession-list", []))
        ]
        if len(matches) != 1:
            raise ValueError(f"Cellosaurus snapshot does not contain exactly one {expected_accession} entry")
        entry = matches[0]
        name_items = [item.get("value") for item in entry.get("name-list", [])
                      if item.get("type") == "identifier"]
        sanger_ids = {
            str(item.get("accession")) for item in entry.get("xref-list", [])
            if item.get("database") == "Cell_Model_Passport"
        }
        depmap_ids = {
            str(item.get("accession")) for item in entry.get("xref-list", [])
            if item.get("database") == "DepMap"
        }
        expected = EXPECTED_ORTHOGONAL_CELLOSAURUS_CROSSWALK[source_id]
        if (len(name_items) != 1 or name_items[0] != expected["cell_line_name"]
                or source_id not in sanger_ids
                or depmap_ids != expected["depmap_model_ids"]):
            raise ValueError(f"Cellosaurus source/DepMap cross-references differ from the audited mapping for {source_id}")
        result[source_id] = {
            "cellosaurus_accession": expected_accession,
            "cell_line_name": name_items[0],
            "cellosaurus_entry_version": str(entry.get("entry-version", "")),
            "cellosaurus_last_updated": str(entry.get("last-updated", "")),
            "cellosaurus_snapshot_sha256": sha256(path),
            "cellosaurus_snapshot_file": path.as_posix(),
            "cellosaurus_source_url": f"https://www.cellosaurus.org/{expected_accession}.json",
            "cell_model_passport_ids": sorted(sanger_ids),
            "depmap_model_ids": sorted(depmap_ids),
        }
    return result


def resolve_strict_unseen_ids(prism_ids: set[str], gdsc_sanger_ids: set[str],
                              model_bridge: pd.DataFrame) -> tuple[set[str], set[str]]:
    """Compatibility wrapper returning the conservative set-union partition."""
    strict, overlap, _ = audit_identity_bridge(prism_ids, gdsc_sanger_ids, model_bridge)
    return strict, overlap


def transform_expression(raw: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    """Apply the frozen source-fold transform; missing genes use source-train means."""
    values = np.asarray(raw, dtype=np.float64)
    mean = np.asarray(transform["mean"], dtype=np.float64)
    scale = np.asarray(transform["scale"], dtype=np.float64)
    minimum = np.asarray(transform["min"], dtype=np.float64)
    span = np.asarray(transform["range"], dtype=np.float64)
    if values.ndim != 2 or any(array.shape != (values.shape[1],)
                               for array in (mean, scale, minimum, span)):
        raise ValueError("source transform dimensions do not match the PRISM gene axis")
    if not all(np.isfinite(array).all() for array in (mean, scale, minimum, span)):
        raise ValueError("source transform contains non-finite values")
    if (scale <= 0).any() or (span <= 0).any() or np.isinf(values).any():
        raise ValueError("expression or source scale has an invalid value")
    filled = np.where(np.isfinite(values), values, mean[None, :])
    output = ((filled - mean[None, :]) / scale[None, :] - minimum[None, :]) / span[None, :]
    if not np.isfinite(output).all():
        raise ValueError("source-transformed PRISM expression is non-finite")
    return output.astype(np.float32)


def materialize(indexed_npz: Path, samples_path: Path, drugs_path: Path,
                strict_ids_path: Path, overlap_ids_path: Path, identity_run_hashes_path: Path,
                gdsc_samples_path: Path, model_csv_path: Path, gene_axis_path: Path,
                target_drug_axis_path: Path, target_drug_vectors_path: Path,
                source_indices_path: Path, checkpoint_path: Path,
                cellosaurus_ew8_path: Path, cellosaurus_h748_path: Path,
                output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"materialized input directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    try:
        input_paths = {
            "prism_indexed_npz": indexed_npz,
            "prism_samples": samples_path,
            "prism_drugs": drugs_path,
            "prism_strict_ids": strict_ids_path,
            "prism_overlap_ids": overlap_ids_path,
            "prism_identity_run_hashes": identity_run_hashes_path,
            "gdsc_source_samples": gdsc_samples_path,
            "depmap_model_bridge": model_csv_path,
            "source_gene_axis": gene_axis_path,
            "pdo_panel_drug_axis": target_drug_axis_path,
            "pdo_panel_drug_vectors": target_drug_vectors_path,
            "source_fold0_indices": source_indices_path,
            "source_fold0_checkpoint": checkpoint_path,
            "cellosaurus_ew8_identity_snapshot": cellosaurus_ew8_path,
            "cellosaurus_nci_h748_identity_snapshot": cellosaurus_h748_path,
        }
        inputs_sha256 = {name: sha256(path) for name, path in input_paths.items()}
        legacy_hashes = json.loads(identity_run_hashes_path.read_text(encoding="utf-8"))
        if legacy_hashes.get("data/artifacts/EXP-019/prism_frozen_v2/strict_unseen_cell_lines.csv") != inputs_sha256["prism_strict_ids"]:
            raise ValueError("strict-unseen identity file differs from the locked EXP-019 output receipt")
        if legacy_hashes.get("data/artifacts/EXP-019/prism_frozen_v2/excluded_gdsc_overlap_cell_lines.csv") != inputs_sha256["prism_overlap_ids"]:
            raise ValueError("excluded-overlap identity file differs from the locked EXP-019 output receipt")

        prism_samples = pd.read_parquet(samples_path, columns=["sample_id", "cell_line_id"]).copy()
        prism_drugs = pd.read_parquet(drugs_path, columns=["drug_id"]).copy()
        gdsc_samples = pd.read_parquet(gdsc_samples_path, columns=["cell_line_id"]).copy()
        strict_file = pd.read_csv(strict_ids_path, usecols=["cell_line_id"])
        overlap_file = pd.read_csv(overlap_ids_path, usecols=["cell_line_id"])
        model_bridge = pd.read_csv(model_csv_path, low_memory=False,
                                   usecols=["ModelID", "SangerModelID", "CellLineName"])
        prism_samples["sample_id"] = _validated_identity_column(prism_samples, "sample_id", "PRISM sample")
        prism_samples["cell_line_id"] = _validated_identity_column(prism_samples, "cell_line_id", "PRISM sample")
        prism_drugs["drug_id"] = _validated_identity_column(prism_drugs, "drug_id", "PRISM drug")
        gdsc_samples["cell_line_id"] = _validated_identity_column(gdsc_samples, "cell_line_id", "GDSC sample")
        strict_file["cell_line_id"] = _validated_identity_column(strict_file, "cell_line_id", "strict-unseen")
        overlap_file["cell_line_id"] = _validated_identity_column(overlap_file, "cell_line_id", "overlap")
        prism_ids = set(prism_samples["sample_id"])
        gdsc_ids = set(gdsc_samples["cell_line_id"])
        if (prism_samples["sample_id"].duplicated().any()
                or gdsc_samples["cell_line_id"].duplicated().any()
                or prism_drugs["drug_id"].duplicated().any()):
            raise ValueError("duplicate PRISM/GDSC biological identity or PRISM drug identity")
        if (strict_file["cell_line_id"].duplicated().any()
                or overlap_file["cell_line_id"].duplicated().any()):
            raise ValueError("frozen PRISM identity files contain duplicate cell identities")
        if not prism_samples["sample_id"].eq(prism_samples["cell_line_id"]).all():
            raise ValueError("PRISM sample and cell-line identity axes disagree")
        frozen_recomputed_strict, frozen_recomputed_overlap, _ = audit_identity_bridge(
            prism_ids, gdsc_ids, model_bridge,
        )
        frozen_strict = set(strict_file["cell_line_id"])
        frozen_overlap = set(overlap_file["cell_line_id"])
        if (frozen_recomputed_strict != frozen_strict or frozen_recomputed_overlap != frozen_overlap
                or len(frozen_strict) != 353 or len(frozen_overlap) != 524):
            raise ValueError("recomputed DepMap bridge does not reproduce the frozen 353/524 cell partition")
        external_crosswalk = load_cellosaurus_crosswalk(cellosaurus_ew8_path, cellosaurus_h748_path)
        model_bridge["ModelID"] = model_bridge["ModelID"].astype("string").str.strip()
        model_names = model_bridge.set_index("ModelID")["CellLineName"].astype("string").to_dict()
        for model_id, expected_name in {"ACH-000499": "EW8", "ACH-003071": "NCIH748"}.items():
            actual_name = str(model_names.get(model_id, ""))
            if re.sub(r"[^A-Z0-9]", "", actual_name.upper()) != expected_name:
                raise ValueError(f"DepMap 24Q2 metadata does not confirm the orthogonal identity for {model_id}")
        for retired_id in ("ACH-002194", "ACH-002176"):
            if retired_id in set(model_bridge["ModelID"].astype(str)):
                raise ValueError(f"retired Cellosaurus candidate {retired_id} unexpectedly appears in DepMap 24Q2")
        strict_ids, overlap_ids, bridge_audit = audit_identity_bridge(
            prism_ids, gdsc_ids, model_bridge, external_crosswalk,
        )
        if bridge_audit["unmapped_source_sanger_ids"]:
            raise ValueError("response-blind identity crosswalk leaves GDSC source identities unresolved")
        if {key: set(value) for key, value in bridge_audit[
                "one_to_many_sanger_model_candidates"].items()} != {
                    **EXPECTED_ONE_TO_MANY_BRIDGE,
                    "SIDM00427": EXPECTED_ORTHOGONAL_CELLOSAURUS_CROSSWALK["SIDM00427"]["depmap_model_ids"],
                    "SIDM01117": EXPECTED_ORTHOGONAL_CELLOSAURUS_CROSSWALK["SIDM01117"]["depmap_model_ids"],
                }:
            raise ValueError("resolved source mapping differs from the frozen one-to-many identity audit")
        moved_to_overlap = sorted(frozen_strict - strict_ids)
        added_to_strict = sorted(strict_ids - frozen_strict)
        if (len(strict_ids) != 352 or len(overlap_ids) != 525
                or moved_to_overlap != ["ACH-000499"] or added_to_strict
                or frozen_overlap - overlap_ids or sorted(overlap_ids - frozen_overlap) != ["ACH-000499"]):
            raise ValueError("resolved identity partition differs from the response-blind one-cell correction")
        bridge_audit["reproduces_frozen_exp019_partition"] = True
        bridge_audit["revised_partition_from_frozen_exp019"] = {
            "strict_before": len(frozen_strict), "overlap_before": len(frozen_overlap),
            "strict_after": len(strict_ids), "overlap_after": len(overlap_ids),
            "moved_from_strict_to_overlap": moved_to_overlap,
        }
        bridge_audit["strict_unseen_interpretation"] = STRICT_UNSEEN_INTERPRETATION

        source_indices = json.loads(source_indices_path.read_text(encoding="utf-8"))
        if source_indices.get("stage") != "source" or int(source_indices.get("fold_id", -1)) != 0:
            raise ValueError("source fold identity file is not fold0")
        source_train_ids = {
            str(identity) for identity, role in source_indices["group_assignments"].items()
            if role == "train"
        }
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        fitted_ids = set(map(str, checkpoint["gene_transform"]["fit_sample_ids"]))
        if len(source_train_ids) != 577 or fitted_ids != source_train_ids:
            raise ValueError("checkpoint preprocessing identities differ from the 577 fold0 training identities")
        transform = checkpoint["gene_transform"]
        genes = list(map(str, json.loads(gene_axis_path.read_text(encoding="utf-8"))))

        with np.load(indexed_npz, allow_pickle=False) as source:
            npz_samples = list(map(str, source["sample_id"].tolist()))
            npz_genes = list(map(str, source["ordered_genes"].tolist()))
            if len(npz_samples) != len(set(npz_samples)):
                raise ValueError("PRISM expression sample axis contains duplicate identities")
            if set(npz_samples) != prism_ids:
                raise ValueError("PRISM expression sample IDs do not exactly match the sample table")
            if (npz_genes != genes or len(genes) != len(set(genes))
                    or len(genes) != int(checkpoint["model_config"]["gene_input_size"])):
                raise ValueError("PRISM expression gene axis differs from the frozen source model axis")
            if not strict_ids <= set(npz_samples):
                raise ValueError("a frozen strict-unseen identity lacks a PRISM expression row")
            expression = source["expression"]
            if expression.shape != (len(npz_samples), len(npz_genes)):
                raise ValueError("PRISM expression matrix shape differs from its explicit axes")
            sample_position = {sample_id: index for index, sample_id in enumerate(npz_samples)}
            ordered_cells = sorted(strict_ids)
            raw_selected = np.asarray(
                expression[[sample_position[cell] for cell in ordered_cells]], dtype=np.float64,
            )
        transformed = transform_expression(raw_selected, transform)

        panel_axis = pd.read_parquet(target_drug_axis_path).copy()
        panel_vectors = np.load(target_drug_vectors_path, allow_pickle=False)
        panel_axis["drug_id"] = _validated_identity_column(panel_axis, "drug_id", "PDO panel drug")
        if panel_axis["drug_id"].duplicated().any() or panel_vectors.shape != (len(panel_axis), 128):
            raise ValueError("frozen PDO panel drug axis/vectors are invalid")
        prism_drugs["drug_id"] = prism_drugs["drug_id"].astype(str)
        panel_axis["drug_id"] = panel_axis["drug_id"].astype(str)
        prism_drug_ids = set(prism_drugs["drug_id"])
        keep = np.asarray([drug_id in prism_drug_ids for drug_id in panel_axis["drug_id"]], dtype=bool)
        selected_drugs = panel_axis.loc[keep].reset_index(drop=True)
        selected_vectors = np.asarray(panel_vectors[keep], dtype=np.float32)
        if len(selected_drugs) != 13 or "PKCDDUHJAFVJJB-UHFFFAOYSA-N" in set(selected_drugs["drug_id"]):
            raise ValueError("the PRISM exact-structure panel does not match the preregistered 13-drug intersection")
        selected_smiles = set(selected_drugs["canonical_isomeric_smiles"].astype(str))
        allowed_smiles = set(map(str, checkpoint["smiles_encoder"]["allowed_smiles"]))
        if not selected_smiles <= allowed_smiles:
            raise ValueError("a matched PRISM drug is not allowed by the audited source encoder")

        cell_axis_path = output_dir / "cell_axis.json"
        expression_path = output_dir / "cell_expression.npy"
        drug_axis_out = output_dir / "drug_axis.parquet"
        drug_vectors_out = output_dir / "drug_ordinal128.npy"
        strict_ids_out = output_dir / "strict_unseen_cell_lines_v2.csv"
        overlap_ids_out = output_dir / "excluded_source_overlap_cell_lines_v2.csv"
        identity_resolution_out = output_dir / "identity_resolution.json"
        cell_axis_path.write_text(json.dumps(ordered_cells, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
        np.save(expression_path, transformed, allow_pickle=False)
        selected_drugs.to_parquet(drug_axis_out, index=False)
        np.save(drug_vectors_out, selected_vectors, allow_pickle=False)
        pd.DataFrame({"cell_line_id": sorted(strict_ids)}).to_csv(strict_ids_out, index=False)
        pd.DataFrame({"cell_line_id": sorted(overlap_ids)}).to_csv(overlap_ids_out, index=False)
        identity_resolution = {
            "experiment_id": "EXP-044",
            "identity_revision": "cellosaurus_crosswalk_v1_applied_to_exp019_frozen_v2_partition",
            "status": IDENTITY_RESOLUTION_STATUS,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "strict_unseen_accepted": True,
            "target_response_values_read": False,
            "response_label_files_opened": [],
            "identity_bridge_audit": bridge_audit,
            "strict_unseen_ids_sha256": sha256(strict_ids_out),
            "source_overlap_ids_sha256": sha256(overlap_ids_out),
            "files_sha256": {
                strict_ids_out.name: sha256(strict_ids_out),
                overlap_ids_out.name: sha256(overlap_ids_out),
            },
        }
        write_json(identity_resolution_out, identity_resolution)
        manifest = {
            "experiment_id": "EXP-044",
            "status": "PRISM_SOURCE_ONLY_INPUTS_MATERIALIZED_LABEL_FREE_IDENTITY_RESOLVED_PENDING_VERIFY",
            "feature_materialization_status": "MATERIALIZED_PENDING_INDEPENDENT_VERIFY",
            "source_fold": 0,
            "n_strict_unseen_prism_cells": len(ordered_cells),
            "n_prism_drugs": len(selected_drugs),
            "n_expression_genes": len(genes),
            "n_expression_missing_values_before_source_imputation": int(np.isnan(raw_selected).sum()),
            "expression_transform": "fold0-source-mean-imputation, StandardScaler, MinMaxScaler; no target fitting",
            "legacy_strict_unseen_ids_sha256": inputs_sha256["prism_strict_ids"],
            "legacy_source_overlap_ids_sha256": inputs_sha256["prism_overlap_ids"],
            "strict_unseen_ids_sha256": sha256(strict_ids_out),
            "source_checkpoint_sha256": inputs_sha256["source_fold0_checkpoint"],
            "source_training_identity_count": len(source_train_ids),
            "identity_resolution_status": IDENTITY_RESOLUTION_STATUS,
            "scientific_strict_unseen_accepted": True,
            "strict_unseen_interpretation": STRICT_UNSEEN_INTERPRETATION,
            "identity_bridge_audit": bridge_audit,
            "target_response_loaded": False,
            "target_response_values_read": False,
            "response_label_files_opened": [],
            "inputs_sha256": inputs_sha256,
            "files_sha256": {
                "cell_axis.json": sha256(cell_axis_path),
                "cell_expression.npy": sha256(expression_path),
                "drug_axis.parquet": sha256(drug_axis_out),
                "drug_ordinal128.npy": sha256(drug_vectors_out),
                "strict_unseen_cell_lines_v2.csv": sha256(strict_ids_out),
                "excluded_source_overlap_cell_lines_v2.csv": sha256(overlap_ids_out),
                "identity_resolution.json": sha256(identity_resolution_out),
            },
        }
        write_json(output_dir / "manifest.json", manifest)
        del checkpoint
        return manifest
    except Exception as error:
        try:
            write_failure_receipt(output_dir / "FAILURE.json", {
                "experiment_id": "EXP-044", "status": "FAIL",
                "error_type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc(), "target_response_loaded": False,
                "target_response_values_read": False, "response_label_files_opened": [],
            })
        except Exception as receipt_error:
            error.add_note(f"Could not write failure receipt: {type(receipt_error).__name__}: {receipt_error}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indexed-npz", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--drugs", type=Path, required=True)
    parser.add_argument("--strict-ids", type=Path, required=True)
    parser.add_argument("--overlap-ids", type=Path, required=True)
    parser.add_argument("--identity-run-hashes", type=Path, required=True)
    parser.add_argument("--gdsc-samples", type=Path, required=True)
    parser.add_argument("--model-csv", type=Path, required=True)
    parser.add_argument("--genes", type=Path, required=True)
    parser.add_argument("--panel-drugs", type=Path, required=True)
    parser.add_argument("--panel-vectors", type=Path, required=True)
    parser.add_argument("--source-indices", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cellosaurus-ew8", type=Path, required=True)
    parser.add_argument("--cellosaurus-h748", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(
        args.indexed_npz, args.samples, args.drugs, args.strict_ids, args.overlap_ids,
        args.identity_run_hashes, args.gdsc_samples, args.model_csv, args.genes,
        args.panel_drugs, args.panel_vectors, args.source_indices, args.checkpoint,
        args.cellosaurus_ew8, args.cellosaurus_h748, args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
