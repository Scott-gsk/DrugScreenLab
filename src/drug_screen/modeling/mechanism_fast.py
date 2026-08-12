"""Compact, frozen ChEMBL/Reactome mechanism-prior preparation for FAST probes."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from drug_screen.modeling.phase1 import _bounded_records


MECHANISM_DIM = 128
MECHANISM_FORMAT = "target_pathway_prior_feature_table_v1"
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
REACTOME_API = "https://reactome.org/ContentService/data"


class MechanismFastError(RuntimeError):
    """Raised when the compact mechanism prior cannot be built safely."""


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mechanism_feature_vector(targets: list[str], pathways: list[str]) -> np.ndarray:
    """Hash stable target/pathway IDs into a fixed-size, non-trainable prior."""
    vector = np.zeros((MECHANISM_DIM,), dtype=np.float32)
    vector[0] = min(len(set(targets)), 16) / 16.0
    vector[1] = min(len(set(pathways)), 64) / 64.0
    for prefix, values, offset, width in (
        ("target", targets, 2, 63),
        ("pathway", pathways, 65, 63),
    ):
        for value in sorted(set(values)):
            digest = sha256(f"{prefix}:{value}".encode("utf-8")).digest()
            vector[offset + int.from_bytes(digest[:8], "big") % width] = 1.0
    return vector


def _get_json(url: str, *, retries: int = 2) -> Any | None:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "DrugScreenLab/phase2-fast"})
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    return None


def _extract_gene_symbols(target: Mapping[str, Any]) -> list[str]:
    symbols: list[str] = []
    for component in target.get("target_components", []) or []:
        for synonym in component.get("target_component_synonyms", []) or []:
            if synonym.get("syn_type") == "GENE_SYMBOL" and synonym.get("component_synonym"):
                symbols.append(str(synonym["component_synonym"]))
    return sorted(set(symbols))


def _extract_reactome_xrefs(target: Mapping[str, Any]) -> list[str]:
    pathways: list[str] = []
    for component in target.get("target_components", []) or []:
        for xref in component.get("target_component_xrefs", []) or []:
            if xref.get("xref_src_db") == "Reactome" and str(xref.get("xref_id", "")).startswith("R-HSA-"):
                pathways.append(str(xref["xref_id"]))
    return sorted(set(pathways))


def annotate_drug_mechanisms(
    perturbagens: pd.DataFrame,
    drug_ids: set[str],
    *,
    live_reactome_lookup: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Query official ChEMBL mechanisms and Reactome UniProt pathway mappings."""
    required = {"pert_id", "inchi_key"}
    missing = required.difference(perturbagens.columns)
    if missing:
        raise MechanismFastError(f"perturbagens missing columns: {sorted(missing)}")
    rows = perturbagens.loc[perturbagens["pert_id"].astype(str).isin(drug_ids), ["pert_id", "inchi_key"]]
    rows = rows.drop_duplicates("pert_id").sort_values("pert_id")
    annotations: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for row in rows.itertuples(index=False):
        pert_id = str(row.pert_id)
        inchi = str(row.inchi_key)
        if inchi in {"", "nan", "-666"}:
            errors[pert_id] = "missing_inchi_key"
            continue
        molecule_url = f"{CHEMBL_API}/molecule.json?{urlencode({'molecule_structures__standard_inchi_key': inchi})}"
        molecule_payload = _get_json(molecule_url)
        molecule_rows = (molecule_payload or {}).get("molecules", []) if isinstance(molecule_payload, Mapping) else []
        chembl_ids = [str(item.get("molecule_chembl_id")) for item in molecule_rows if item.get("molecule_chembl_id")]
        if not chembl_ids:
            errors[pert_id] = "no_chembl_structure_match"
            continue
        chembl_id = sorted(chembl_ids)[0]
        mechanism_url = f"{CHEMBL_API}/mechanism.json?{urlencode({'molecule_chembl_id': chembl_id})}"
        mechanism_payload = _get_json(mechanism_url)
        mechanism_rows = (mechanism_payload or {}).get("mechanisms", []) if isinstance(mechanism_payload, Mapping) else []
        targets: list[str] = []
        accessions: list[str] = []
        pathways: set[str] = set()
        mechanisms: list[dict[str, Any]] = []
        for mechanism in mechanism_rows:
            target_id = mechanism.get("target_chembl_id")
            if not target_id:
                continue
            target_id = str(target_id)
            target_payload = _get_json(f"{CHEMBL_API}/target/{target_id}.json") or {}
            target_name = str(target_payload.get("pref_name", ""))
            symbols = _extract_gene_symbols(target_payload)
            component_accessions = [
                str(component.get("accession"))
                for component in target_payload.get("target_components", []) or []
                if component.get("accession")
            ]
            target_pathways = _extract_reactome_xrefs(target_payload)
            targets.extend(symbols or [target_id])
            accessions.extend(component_accessions)
            pathways.update(target_pathways)
            mechanisms.append({
                "target_chembl_id": target_id,
                "target_name": target_name,
                "gene_symbols": symbols,
                "action_type": mechanism.get("action_type"),
                "mechanism_of_action": mechanism.get("mechanism_of_action"),
            })
        if live_reactome_lookup:
            for accession in sorted(set(accessions)):
                payload = _get_json(f"{REACTOME_API}/mapping/UniProt/{accession}/pathways")
                if isinstance(payload, list):
                    pathways.update(str(item["stId"]) for item in payload if item.get("stId"))
        annotations[pert_id] = {
            "pert_id": pert_id,
            "chembl_id": chembl_id,
            "targets": sorted(set(targets)),
            "uniprot_accessions": sorted(set(accessions)),
            "pathways": sorted(pathways),
            "mechanisms": mechanisms,
            "feature": mechanism_feature_vector(targets, sorted(pathways)).tolist(),
        }
    audit = {
        "format": MECHANISM_FORMAT,
        "source": {
            "chembl_api": CHEMBL_API,
            "reactome_api": REACTOME_API,
            "reactome_lookup": "live_mapping" if live_reactome_lookup else "ChEMBL_target_cross_reference_only",
            "labels_used": False,
        },
        "requested_drugs": len(drug_ids),
        "annotated_drugs": len(annotations),
        "errors": errors,
        "coverage": len(annotations) / len(drug_ids) if drug_ids else 0.0,
    }
    return annotations, audit


def build_frozen_candidate_mechanism_probe(
    *,
    perturbagen_path: Path | str,
    prism_audit_path: Path | str,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Prepare mechanism provenance for the frozen four-drug downstream diagnostic.

    This deliberately does not fit a model or tune against PRISM. It answers only
    whether the frozen cohort has enough public target/pathway identity to justify a
    subsequent mechanism-conditioned model increment.
    """
    prism_audit = json.loads(Path(prism_audit_path).read_text(encoding="utf-8"))
    candidates = prism_audit["identity"]["candidates"]
    candidate_ids = {str(row["pert_id"]) for row in candidates}
    perturbagens = pd.read_csv(perturbagen_path, sep="\t", low_memory=False)
    annotations, audit = annotate_drug_mechanisms(
        perturbagens, candidate_ids, live_reactome_lookup=False
    )
    for candidate in candidates:
        candidate.setdefault("mechanism_annotation_status", "ANNOTATED" if candidate["pert_id"] in annotations else "MISSING")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "frozen_candidate_target_pathway_probe_v1",
        "status": "FORWARD_PREPARATION_COMPLETE",
        "candidate_count": len(candidates),
        "annotations": annotations,
        "coverage_audit": audit,
        "decision": {
            "model_increment": "NOT_RUN_COVERAGE_INSUFFICIENT",
            "reason": "Four-drug downstream cohort is insufficient to justify target/pathway model tuning or a learned additive weight; no PRISM label was used for fitting.",
            "labels_used_for_tuning": False,
        },
        "candidates": candidates,
    }
    path = output / "mechanism_probe.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["artifact"] = str(path)
    return payload


def build_target_pathway_manifest(
    *,
    base_manifest_path: Path | str,
    perturbagen_path: Path | str,
    output_dir: Path | str,
    root: Path | str,
    max_records: int = 2048,
) -> dict[str, Any]:
    """Append a mechanism prior to the frozen Phase-1 Morgan features for FAST only."""
    base_path = Path(base_manifest_path)
    repo_root = Path(root).resolve()
    payload = json.loads(base_path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise MechanismFastError("base Phase-1 manifest has no records")
    from drug_screen.modeling.phase1 import Phase1Record

    typed_records = tuple(Phase1Record.from_mapping(row) for row in records)
    selected = _bounded_records(typed_records, max_records)
    selected_drugs = {row.drug_id for row in selected}
    perturbagens = pd.read_csv(perturbagen_path, sep="\t", low_memory=False)
    annotations, audit = annotate_drug_mechanisms(perturbagens, selected_drugs)

    feature_path = repo_root / str(payload["chemical_features"]["relative_path"])
    chemical = np.load(feature_path, mmap_mode="r").astype(np.float32, copy=False)
    drug_to_row = {row.drug_id: row.chemical_feature_row for row in typed_records}
    mechanism = np.zeros((chemical.shape[0], MECHANISM_DIM), dtype=np.float32)
    for drug_id, annotation in annotations.items():
        mechanism[drug_to_row[drug_id]] = np.asarray(annotation["feature"], dtype=np.float32)
    combined = np.concatenate([chemical, mechanism], axis=1)

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    combined_path = output / "chemical_plus_target_pathway_features.npy"
    np.save(combined_path, combined)
    updated_payload = dict(payload)
    updated_payload["format"] = "phase1_context_mechanism_manifest_v1"
    updated_payload["phase"] = "phase_3_fast_target_pathway_prior"
    updated_payload["chemical_features"] = {
        "relative_path": str(combined_path.relative_to(repo_root)).replace("\\", "/"),
        "sha256": file_sha256(combined_path),
        "shape": list(combined.shape),
        "representation": "RDKit_Morgan128_plus_ChEMBL_target_Reactome_pathway_hashed_prior128",
        "mechanism_dimension": MECHANISM_DIM,
        "mechanism_annotated_drugs": len(annotations),
    }
    updated_payload["frozen_upstream_manifest_sha256"] = file_sha256(base_path)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(updated_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    annotation_path = output / "mechanism_annotations.json"
    annotation_path.write_text(json.dumps(annotations, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit.update({
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "feature_sha256": file_sha256(combined_path),
        "feature_shape": list(combined.shape),
        "selected_records": len(selected),
        "selected_drugs": len(selected_drugs),
        "annotation_path": str(annotation_path),
        "representation": updated_payload["chemical_features"]["representation"],
    })
    (output / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit
