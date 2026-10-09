"""Independently verify label-free EXP-044 PRISM model inputs.

This verifier intentionally does not import the materializer. It opens only
identity/sample metadata, the response-free drug axis, source-fold metadata,
and the three approved NPZ arrays (sample_id, ordered_genes, expression).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


EXPECTED_STRICT_COUNT = 352
EXPECTED_OVERLAP_COUNT = 525
EXPECTED_SOURCE_TRAIN_COUNT = 577
EXCLUDED_DRUG_ID = "PKCDDUHJAFVJJB-UHFFFAOYSA-N"
EXPECTED_DIRECTLY_UNMAPPED_SOURCE_SANGER_IDS = {"SIDM00427", "SIDM01117"}
EXPECTED_ORTHOGONAL_CELLOSAURUS_CROSSWALK = {
    "SIDM00427": {
        "accession": "CVCL_1658", "cell_line_name": "EW-8",
        "depmap_model_ids": {"ACH-000499", "ACH-002194"},
    },
    "SIDM01117": {
        "accession": "CVCL_1588", "cell_line_name": "NCI-H748",
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


def _unique_ids(values: pd.Series, description: str) -> list[str]:
    if values.isna().any():
        raise ValueError(f"{description} contains missing identities")
    result = values.astype(str).str.strip().tolist()
    if any(not value for value in result):
        raise ValueError(f"{description} contains blank identities")
    if len(set(result)) != len(result):
        raise ValueError(f"{description} contains duplicate identities")
    return result


def _independent_partition(prism_ids: set[str], source_sanger_ids: set[str],
                           bridge_frame: pd.DataFrame,
                           external_crosswalk: dict[str, dict[str, Any]] | None = None
                           ) -> tuple[set[str], set[str], dict[str, Any]]:
    """Rebuild the GDSC-to-DepMap overlap, rejecting non-bijective bridge rows."""
    bridge = bridge_frame.loc[:, ["ModelID", "SangerModelID"]].copy()
    bridge["ModelID"] = bridge["ModelID"].astype("string").str.strip()
    bridge["SangerModelID"] = bridge["SangerModelID"].astype("string").str.strip()
    joined = bridge.loc[bridge["SangerModelID"].isin(source_sanger_ids)]
    if joined.empty:
        raise ValueError("DepMap bridge maps no GDSC Sanger identities")
    if joined["ModelID"].isna().any() or joined["ModelID"].eq("").any():
        raise ValueError("a matched GDSC identity has no DepMap ModelID")
    candidate_ids: dict[str, set[str]] = {}
    for source_id, model_id in zip(joined["SangerModelID"], joined["ModelID"]):
        candidate_ids.setdefault(str(source_id), set()).add(str(model_id))
    directly_unmapped = sorted(source_sanger_ids - set(candidate_ids))
    external_crosswalk = external_crosswalk or {}
    if external_crosswalk and set(external_crosswalk) != set(directly_unmapped):
        raise ValueError("orthogonal source crosswalk does not resolve exactly the direct bridge gaps")
    if external_crosswalk and set(external_crosswalk) != set(EXPECTED_ORTHOGONAL_CELLOSAURUS_CROSSWALK):
        raise ValueError("orthogonal identity source IDs differ from the audited source identities")
    for source_id, evidence in external_crosswalk.items():
        models = set(map(str, evidence.get("depmap_model_ids", [])))
        expected = EXPECTED_ORTHOGONAL_CELLOSAURUS_CROSSWALK[source_id]
        if (evidence.get("cellosaurus_accession") != expected["accession"]
                or evidence.get("cell_line_name") != expected["cell_line_name"]
                or models != expected["depmap_model_ids"]):
            raise ValueError(f"Cellosaurus identity evidence differs for {source_id}")
        candidate_ids[source_id] = models
    mapped_ids = set().union(*candidate_ids.values())
    overlap_ids = prism_ids.intersection(mapped_ids)
    strict_ids = prism_ids.difference(mapped_ids)
    if (strict_ids.intersection(overlap_ids)
            or strict_ids.union(overlap_ids) != prism_ids):
        raise ValueError("recomputed strict/overlap IDs fail to partition PRISM")
    unmapped = sorted(source_sanger_ids - set(candidate_ids))
    one_to_many = {
        identity: sorted(models)
        for identity, models in sorted(candidate_ids.items())
        if len(models) > 1
    }
    audit = {
        "mapping_policy": "conservative_union_of_DepMap_24Q2_Sanger_bridge_and_authoritative_Cellosaurus_crossrefs",
        "source_sanger_identity_count": len(source_sanger_ids),
        "directly_matched_source_sanger_identity_count": len(candidate_ids) - len(external_crosswalk),
        "orthogonally_resolved_source_sanger_ids": sorted(external_crosswalk),
        "unmapped_source_sanger_ids": unmapped,
        "one_to_many_sanger_model_candidates": one_to_many,
        "candidate_depmap_model_id_count": len(mapped_ids),
        "matched_bridge_row_count": len(joined),
        "orthogonal_identity_evidence": external_crosswalk,
        "all_candidate_model_ids_tested_against_prism_axis": True,
    }
    return strict_ids, overlap_ids, audit


def _independent_cellosaurus_crosswalk(ew8_path: Path, h748_path: Path) -> dict[str, dict[str, Any]]:
    """Independently parse saved Cellosaurus snapshots, without materializer imports."""
    snapshots = {
        "SIDM00427": (ew8_path, "CVCL_1658"),
        "SIDM01117": (h748_path, "CVCL_1588"),
    }
    resolved: dict[str, dict[str, Any]] = {}
    for source_id, (path, accession) in snapshots.items():
        root = json.loads(path.read_text(encoding="utf-8"))
        entries = root.get("Cellosaurus", {}).get("cell-line-list", [])
        matched = [entry for entry in entries if any(
            item.get("type") == "primary" and item.get("value") == accession
            for item in entry.get("accession-list", [])
        )]
        if len(matched) != 1:
            raise ValueError(f"authoritative snapshot lacks one primary {accession} record")
        entry = matched[0]
        names = [item.get("value") for item in entry.get("name-list", [])
                 if item.get("type") == "identifier"]
        passport_ids = {
            str(item.get("accession")) for item in entry.get("xref-list", [])
            if item.get("database") == "Cell_Model_Passport"
        }
        depmap_ids = {
            str(item.get("accession")) for item in entry.get("xref-list", [])
            if item.get("database") == "DepMap"
        }
        expected = EXPECTED_ORTHOGONAL_CELLOSAURUS_CROSSWALK[source_id]
        if (names != [expected["cell_line_name"]] or source_id not in passport_ids
                or depmap_ids != expected["depmap_model_ids"]):
            raise ValueError(f"saved Cellosaurus cross-references differ for {source_id}")
        resolved[source_id] = {
            "cellosaurus_accession": accession,
            "cell_line_name": names[0],
            "cellosaurus_entry_version": str(entry.get("entry-version", "")),
            "cellosaurus_last_updated": str(entry.get("last-updated", "")),
            "cellosaurus_snapshot_sha256": sha256(path),
            "cellosaurus_snapshot_file": path.as_posix(),
            "cellosaurus_source_url": f"https://www.cellosaurus.org/{accession}.json",
            "cell_model_passport_ids": sorted(passport_ids),
            "depmap_model_ids": sorted(depmap_ids),
        }
    return resolved


def _source_transform(values: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    """Independent implementation of source-fold mean imputation and scaling."""
    array = np.asarray(values, dtype=np.float64)
    location = np.asarray(parameters["mean"], dtype=np.float64)
    deviation = np.asarray(parameters["scale"], dtype=np.float64)
    lower = np.asarray(parameters["min"], dtype=np.float64)
    width = np.asarray(parameters["range"], dtype=np.float64)
    gene_count = array.shape[1] if array.ndim == 2 else -1
    if gene_count < 1 or any(v.shape != (gene_count,) for v in (location, deviation, lower, width)):
        raise ValueError("source-fold parameters do not match the expression gene axis")
    if not all(np.isfinite(v).all() for v in (location, deviation, lower, width)):
        raise ValueError("source-fold parameters contain non-finite values")
    if (deviation <= 0).any() or (width <= 0).any() or np.isinf(array).any():
        raise ValueError("expression or source-fold scale is invalid")
    imputed = np.where(np.isfinite(array), array, location[None, :])
    scaled = ((imputed - location[None, :]) / deviation[None, :] - lower[None, :]) / width[None, :]
    if not np.isfinite(scaled).all():
        raise ValueError("independently transformed expression contains non-finite values")
    return scaled.astype(np.float32)


def _input_paths(indexed_npz: Path, samples_path: Path, drugs_path: Path,
                 strict_ids_path: Path, overlap_ids_path: Path,
                 identity_run_hashes_path: Path, gdsc_samples_path: Path,
                 model_csv_path: Path, gene_axis_path: Path,
                 target_drug_axis_path: Path, target_drug_vectors_path: Path,
                 source_indices_path: Path, checkpoint_path: Path,
                 cellosaurus_ew8_path: Path, cellosaurus_h748_path: Path) -> dict[str, Path]:
    return {
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


def verify(indexed_npz: Path, samples_path: Path, drugs_path: Path,
           strict_ids_path: Path, overlap_ids_path: Path, identity_run_hashes_path: Path,
           gdsc_samples_path: Path, model_csv_path: Path, gene_axis_path: Path,
           target_drug_axis_path: Path, target_drug_vectors_path: Path,
           source_indices_path: Path, checkpoint_path: Path,
           cellosaurus_ew8_path: Path, cellosaurus_h748_path: Path,
           input_dir: Path) -> dict[str, Any]:
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "EXP-044":
        raise ValueError("materialized input manifest is not for EXP-044")
    if (manifest.get("identity_resolution_status") != IDENTITY_RESOLUTION_STATUS
            or manifest.get("scientific_strict_unseen_accepted") is not True
            or manifest.get("strict_unseen_interpretation") != STRICT_UNSEEN_INTERPRETATION):
        raise ValueError("manifest does not bind the resolved response-blind strict-unseen identity revision")
    if manifest.get("target_response_loaded") is not False or manifest.get("target_response_values_read") is not False:
        raise ValueError("manifest does not certify response-free input materialization")
    if manifest.get("response_label_files_opened") != []:
        raise ValueError("manifest reports opening a response-label file")

    inputs = _input_paths(
        indexed_npz, samples_path, drugs_path, strict_ids_path, overlap_ids_path,
        identity_run_hashes_path, gdsc_samples_path, model_csv_path, gene_axis_path,
        target_drug_axis_path, target_drug_vectors_path, source_indices_path, checkpoint_path,
        cellosaurus_ew8_path, cellosaurus_h748_path,
    )
    actual_input_hashes = {name: sha256(path) for name, path in inputs.items()}
    if manifest.get("inputs_sha256") != actual_input_hashes:
        raise ValueError("input checksum set differs from the materialization manifest")
    if manifest.get("source_checkpoint_sha256") != actual_input_hashes["source_fold0_checkpoint"]:
        raise ValueError("source checkpoint checksum differs from the frozen input")

    locked_hashes = json.loads(identity_run_hashes_path.read_text(encoding="utf-8"))
    if (locked_hashes.get("data/artifacts/EXP-019/prism_frozen_v2/strict_unseen_cell_lines.csv")
            != actual_input_hashes["prism_strict_ids"]):
        raise ValueError("strict-unseen IDs differ from the EXP-019 frozen receipt")
    if (locked_hashes.get("data/artifacts/EXP-019/prism_frozen_v2/excluded_gdsc_overlap_cell_lines.csv")
            != actual_input_hashes["prism_overlap_ids"]):
        raise ValueError("overlap IDs differ from the EXP-019 frozen receipt")
    if (manifest.get("legacy_strict_unseen_ids_sha256") != actual_input_hashes["prism_strict_ids"]
            or manifest.get("legacy_source_overlap_ids_sha256") != actual_input_hashes["prism_overlap_ids"]):
        raise ValueError("materialization manifest does not bind both immutable EXP-019 identity inputs")

    prism_samples = pd.read_parquet(samples_path, columns=["sample_id", "cell_line_id"])
    prism_ids_list = _unique_ids(prism_samples["sample_id"], "PRISM sample table")
    prism_cell_ids = _unique_ids(prism_samples["cell_line_id"], "PRISM cell-line column")
    prism_ids = set(prism_ids_list)
    if prism_ids_list != prism_cell_ids:
        raise ValueError("PRISM sample_id and cell_line_id row keys disagree")
    source_samples = pd.read_parquet(gdsc_samples_path, columns=["cell_line_id"])
    source_ids = set(_unique_ids(source_samples["cell_line_id"], "GDSC source sample table"))
    strict_frame = pd.read_csv(strict_ids_path, usecols=["cell_line_id"])
    overlap_frame = pd.read_csv(overlap_ids_path, usecols=["cell_line_id"])
    frozen_strict = set(_unique_ids(strict_frame["cell_line_id"], "frozen strict-unseen IDs"))
    frozen_overlap = set(_unique_ids(overlap_frame["cell_line_id"], "frozen overlap IDs"))
    bridge = pd.read_csv(model_csv_path, low_memory=False,
                         usecols=["ModelID", "SangerModelID", "CellLineName"])
    legacy_strict, legacy_overlap, direct_audit = _independent_partition(
        prism_ids, source_ids, bridge,
    )
    if (legacy_strict != frozen_strict or legacy_overlap != frozen_overlap
            or len(frozen_strict) != 353 or len(frozen_overlap) != 524
            or set(direct_audit["unmapped_source_sanger_ids"]) != EXPECTED_DIRECTLY_UNMAPPED_SOURCE_SANGER_IDS
            or direct_audit["one_to_many_sanger_model_candidates"].get("SIDM00117")
            != ["ACH-000837", "ACH-002172"]):
        raise ValueError("frozen 353/524 PRISM identity partition does not reproduce")
    crosswalk = _independent_cellosaurus_crosswalk(cellosaurus_ew8_path, cellosaurus_h748_path)
    names = bridge.set_index(bridge["ModelID"].astype("string").str.strip())["CellLineName"].astype("string").to_dict()
    for model_id, expected_name in {"ACH-000499": "EW8", "ACH-003071": "NCIH748"}.items():
        if re.sub(r"[^A-Z0-9]", "", str(names.get(model_id, "")).upper()) != expected_name:
            raise ValueError(f"DepMap 24Q2 model metadata fails to confirm {model_id}")
    if any(retired_id in set(bridge["ModelID"].astype(str))
           for retired_id in ("ACH-002194", "ACH-002176")):
        raise ValueError("a retired DepMap candidate unexpectedly occurs in the exact 24Q2 model table")
    recomputed_strict, recomputed_overlap, bridge_audit = _independent_partition(
        prism_ids, source_ids, bridge, crosswalk,
    )
    moved_to_overlap = sorted(frozen_strict - recomputed_strict)
    if (bridge_audit["unmapped_source_sanger_ids"]
            or len(recomputed_strict) != EXPECTED_STRICT_COUNT
            or len(recomputed_overlap) != EXPECTED_OVERLAP_COUNT
            or moved_to_overlap != ["ACH-000499"]
            or sorted(recomputed_overlap - frozen_overlap) != ["ACH-000499"]
            or frozen_overlap - recomputed_overlap):
        raise ValueError("orthogonal bridge does not reproduce the frozen one-cell partition correction")
    revised_strict_frame = pd.read_csv(input_dir / "strict_unseen_cell_lines_v2.csv")
    revised_overlap_frame = pd.read_csv(input_dir / "excluded_source_overlap_cell_lines_v2.csv")
    revised_strict = set(_unique_ids(revised_strict_frame["cell_line_id"], "revised strict-unseen IDs"))
    revised_overlap = set(_unique_ids(revised_overlap_frame["cell_line_id"], "revised source-overlap IDs"))
    if revised_strict != recomputed_strict or revised_overlap != recomputed_overlap:
        raise ValueError("saved revised identity lists differ from independent crosswalk reconstruction")
    resolution_receipt = json.loads((input_dir / "identity_resolution.json").read_text(encoding="utf-8"))
    bridge_audit["reproduces_frozen_exp019_partition"] = True
    bridge_audit["revised_partition_from_frozen_exp019"] = {
        "strict_before": len(frozen_strict), "overlap_before": len(frozen_overlap),
        "strict_after": len(revised_strict), "overlap_after": len(revised_overlap),
        "moved_from_strict_to_overlap": moved_to_overlap,
    }
    bridge_audit["strict_unseen_interpretation"] = STRICT_UNSEEN_INTERPRETATION
    if manifest.get("identity_bridge_audit") != bridge_audit:
        raise ValueError("manifest bridge audit differs from the independent set-union reconstruction")
    if (resolution_receipt.get("identity_bridge_audit") != bridge_audit
            or resolution_receipt.get("strict_unseen_accepted") is not True
            or resolution_receipt.get("target_response_values_read") is not False
            or resolution_receipt.get("response_label_files_opened") != []):
        raise ValueError("identity-resolution receipt differs from independent response-blind reconstruction")

    split_receipt = json.loads(source_indices_path.read_text(encoding="utf-8"))
    if split_receipt.get("stage") != "source" or int(split_receipt.get("fold_id", -1)) != 0:
        raise ValueError("source identity receipt is not fold 0")
    assignments = split_receipt.get("group_assignments")
    if not isinstance(assignments, dict):
        raise ValueError("source fold receipt has no group assignments")
    source_train_ids = {str(identity) for identity, role in assignments.items() if role == "train"}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    fitted_ids = set(map(str, checkpoint["gene_transform"]["fit_sample_ids"]))
    if (len(source_train_ids) != EXPECTED_SOURCE_TRAIN_COUNT
            or fitted_ids != source_train_ids):
        raise ValueError("checkpoint fit identities differ from the 577 fold-0 source train IDs")
    if source_train_ids.intersection(prism_ids):
        raise ValueError("source-fold training identities overlap the PRISM sample namespace")

    genes = list(map(str, json.loads(gene_axis_path.read_text(encoding="utf-8"))))
    if not genes or len(genes) != len(set(genes)):
        raise ValueError("frozen source gene axis is empty or duplicated")
    if len(genes) != int(checkpoint["model_config"]["gene_input_size"]):
        raise ValueError("checkpoint gene input size differs from the frozen gene axis")
    # Access only these three members. In particular, pair_response is never indexed.
    with np.load(indexed_npz, allow_pickle=False) as archive:
        npz_sample_ids = list(map(str, archive["sample_id"].tolist()))
        npz_genes = list(map(str, archive["ordered_genes"].tolist()))
        if len(npz_sample_ids) != len(set(npz_sample_ids)):
            raise ValueError("raw PRISM expression sample axis contains duplicate IDs")
        if set(npz_sample_ids) != prism_ids:
            raise ValueError("raw NPZ sample IDs do not exactly match the sample metadata IDs")
        if npz_genes != genes:
            raise ValueError("raw NPZ ordered genes differ from the frozen source gene axis")
        expression = archive["expression"]
        if expression.shape != (len(npz_sample_ids), len(npz_genes)):
            raise ValueError("raw PRISM expression shape differs from its explicit axes")
        row_for_sample = {identity: position for position, identity in enumerate(npz_sample_ids)}
        ordered_cells = sorted(revised_strict)
        raw_selected = np.asarray(
            expression[[row_for_sample[identity] for identity in ordered_cells]], dtype=np.float64,
        )
    independently_scaled = _source_transform(raw_selected, checkpoint["gene_transform"])
    output_files = {
        "cell_axis.json", "cell_expression.npy", "drug_axis.parquet",
        "drug_ordinal128.npy", "strict_unseen_cell_lines_v2.csv",
        "excluded_source_overlap_cell_lines_v2.csv", "identity_resolution.json", "manifest.json",
    }
    found_files = {path.name for path in input_dir.iterdir() if path.is_file()}
    if found_files != output_files:
        raise ValueError("materialized directory contains missing or unexpected files")
    expected_cell_axis = json.loads((input_dir / "cell_axis.json").read_text(encoding="utf-8"))
    if expected_cell_axis != ordered_cells:
        raise ValueError("materialized cell axis differs from independently sorted strict IDs")
    actual_expression = np.load(input_dir / "cell_expression.npy", allow_pickle=False)
    if (actual_expression.shape != independently_scaled.shape
            or not np.isfinite(actual_expression).all()
            or not np.allclose(actual_expression, independently_scaled, rtol=0, atol=1e-6)):
        raise ValueError("materialized cell expression differs from independent source-fold transform")

    prism_drugs = pd.read_parquet(drugs_path, columns=["drug_id"])
    prism_drug_ids = set(_unique_ids(prism_drugs["drug_id"], "PRISM drug table"))
    panel_axis = pd.read_parquet(target_drug_axis_path)
    if "drug_id" not in panel_axis.columns or "canonical_isomeric_smiles" not in panel_axis.columns:
        raise ValueError("frozen PDO panel lacks drug identity or structure")
    panel_drug_ids = _unique_ids(panel_axis["drug_id"], "PDO panel drug axis")
    keep = np.asarray([drug_id in prism_drug_ids for drug_id in panel_drug_ids], dtype=bool)
    expected_drugs = panel_axis.loc[keep].reset_index(drop=True)
    actual_drugs = pd.read_parquet(input_dir / "drug_axis.parquet")
    pd.testing.assert_frame_equal(actual_drugs, expected_drugs, check_dtype=True)
    vectors = np.load(target_drug_vectors_path, allow_pickle=False)
    if vectors.shape != (len(panel_axis), 128) or not np.isfinite(vectors).all():
        raise ValueError("frozen panel drug vectors have invalid shape or values")
    actual_vectors = np.load(input_dir / "drug_ordinal128.npy", allow_pickle=False)
    expected_vectors = np.asarray(vectors[keep], dtype=np.float32)
    if (len(expected_drugs) != 13 or EXCLUDED_DRUG_ID in set(expected_drugs["drug_id"].astype(str))
            or not np.array_equal(actual_vectors, expected_vectors)):
        raise ValueError("materialized drug axis or vectors differ from the 13-drug frozen intersection")
    allowed_smiles = set(map(str, checkpoint["smiles_encoder"]["allowed_smiles"]))
    if not set(expected_drugs["canonical_isomeric_smiles"].astype(str)) <= allowed_smiles:
        raise ValueError("a materialized drug is absent from the frozen source encoder")

    expected_counts = {
        "n_strict_unseen_prism_cells": len(ordered_cells),
        "n_prism_drugs": len(expected_drugs),
        "n_expression_genes": len(genes),
        "n_expression_missing_values_before_source_imputation": int(np.isnan(raw_selected).sum()),
        "source_training_identity_count": len(source_train_ids),
    }
    for key, expected in expected_counts.items():
        if manifest.get(key) != expected:
            raise ValueError(f"manifest count differs from independent recomputation: {key}")
    checksums = {
        name: sha256(input_dir / name)
        for name in ("cell_axis.json", "cell_expression.npy", "drug_axis.parquet", "drug_ordinal128.npy",
                     "strict_unseen_cell_lines_v2.csv", "excluded_source_overlap_cell_lines_v2.csv",
                     "identity_resolution.json")
    }
    if manifest.get("files_sha256") != checksums:
        raise ValueError("materialized feature file checksums do not match the manifest")
    strict_list_sha = checksums["strict_unseen_cell_lines_v2.csv"]
    overlap_list_sha = checksums["excluded_source_overlap_cell_lines_v2.csv"]
    if (manifest.get("strict_unseen_ids_sha256") != strict_list_sha
            or resolution_receipt.get("strict_unseen_ids_sha256") != strict_list_sha
            or resolution_receipt.get("source_overlap_ids_sha256") != overlap_list_sha
            or resolution_receipt.get("files_sha256") != {
                "strict_unseen_cell_lines_v2.csv": strict_list_sha,
                "excluded_source_overlap_cell_lines_v2.csv": overlap_list_sha,
            }):
        raise ValueError("revised identity-list hashes are not bound consistently across the receipts")

    return {
        "experiment_id": "EXP-044",
        "status": "PASS_RESPONSE_BLIND_IDENTITY_RESOLVED",
        "feature_verification_status": "PASS",
        "identity_resolution_status": IDENTITY_RESOLUTION_STATUS,
        "scientific_strict_unseen_accepted": True,
        "strict_unseen_interpretation": STRICT_UNSEEN_INTERPRETATION,
        "identity_bridge_audit": bridge_audit,
        "n_strict_unseen_prism_cells": len(ordered_cells),
        "n_prism_drugs": len(expected_drugs),
        "n_expression_genes": len(genes),
        "n_expression_missing_values_before_source_imputation": int(np.isnan(raw_selected).sum()),
        "source_training_identity_count": len(source_train_ids),
        "max_expression_abs_error": float(np.max(np.abs(actual_expression - independently_scaled))),
        "input_manifest_sha256": sha256(manifest_path),
        "source_checkpoint_sha256": actual_input_hashes["source_fold0_checkpoint"],
        "input_sha256": actual_input_hashes,
        "feature_file_sha256": checksums,
        "target_response_values_read": False,
        "response_label_files_opened": [],
        "npz_members_read": ["sample_id", "ordered_genes", "expression"],
    }


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
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.indexed_npz, args.samples, args.drugs, args.strict_ids, args.overlap_ids,
        args.identity_run_hashes, args.gdsc_samples, args.model_csv, args.genes,
        args.panel_drugs, args.panel_vectors, args.source_indices, args.checkpoint,
        args.cellosaurus_ew8, args.cellosaurus_h748, args.input_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
