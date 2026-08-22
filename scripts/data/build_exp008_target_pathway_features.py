"""Freeze response-blind EXP-008 drug-to-target-to-pathway features.

The builder uses only approved identities and frozen SDST metadata.  It never
reads responses, efficacy labels, PRISM, GDSC, or split labels for selection.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_H5AD = ROOT / "data/external/xpert_source/processed_data/l1000_sdst_78453.h5ad"
DEFAULT_CHEMBL = Path("/mnt/d/Code/TranSiGen/data/screening/libraries/chembl_36/chembl_36_sqlite/chembl_36.db")
DEFAULT_STRING = ROOT / "data/raw/STRING/9606.protein.links.v12.0.txt.gz"
DEFAULT_GO_BP = ROOT / "data/raw/pathways/go/go_bp_human.gmt"
DEFAULT_DRUGS_INFO = ROOT / "data/external/xpert_source/processed_data/l1000_sdst_drugs_info_8432.csv"
DEFAULT_CROSSWALK = ROOT / "artifacts/experiments/EXP-008/uniprot_entrez_crosswalk.tsv"
DEFAULT_OUTPUT = ROOT / "artifacts/experiments/EXP-008"
STRING_PHYSICAL_SCORE_MIN = 700


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _axis_sha256(values: Iterable[str]) -> str:
    digest = sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_if_readable(path: Path) -> str | None:
    try:
        return _sha256(path)
    except OSError:
        return None


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def _table_columns(connection: sqlite3.Connection) -> dict[str, list[str]]:
    tables = connection.execute("SELECT name FROM sqlite_master WHERE type = ? ORDER BY name", ("table",)).fetchall()
    return {str(name): [str(row[1]) for row in connection.execute(f'PRAGMA table_info("{name}")')] for (name,) in tables}


def _resolve_schema(connection: sqlite3.Connection) -> dict[str, str] | None:
    tables = _table_columns(connection)
    fixture = {
        "drug_identity": {"pert_id", "chembl_id"},
        "direct_human_single_protein": {"chembl_id", "target_accession", "direct_interaction", "organism", "target_type"},
    }
    if all(required.issubset(tables.get(table, [])) for table, required in fixture.items()):
        return {"kind": "fixture", "identity": "drug_identity", "interaction": "direct_human_single_protein"}
    expected = {"molecule_dictionary", "drug_mechanism", "target_dictionary", "target_components", "component_sequences", "compound_structures"}
    return {"kind": "chembl36"} if expected.issubset(tables) else None


def _load_accession_crosswalk(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["accession"], []).append(row)
    mapped: dict[str, str] = {}
    excluded: list[str] = []
    for accession, entries in sorted(grouped.items()):
        eligible = [row for row in entries if row["mapping_status"] == "mapped" and row["ambiguous"].lower() == "false"]
        ids = {row["entrez_id"] for row in eligible}
        if len(entries) == 1 and len(eligible) == 1 and len(ids) == 1:
            mapped[accession] = next(iter(ids))
        else:
            excluded.append(accession)
    return mapped, {
        "source": str(path), "sha256": _sha256(path), "rows": len(rows), "mapped_unique": len(mapped),
        "ambiguous_rows": sum(sum(row["ambiguous"].lower() == "true" for row in grouped[a]) for a in excluded), "excluded_accessions": excluded,
        "rule": "only mapping_status=mapped and ambiguous=false singleton accession rows contribute",
    }


def _load_mappings(chembl_path: Path, pert_ids: list[str], crosswalk_path: Path | None, drugs_info_path: Path) -> tuple[dict[str, list[str]], dict[str, Any]]:
    if crosswalk_path is None or not crosswalk_path.exists():
        return {}, {"status": "DATA_BLOCKED", "reason": "required accession-to-Entrez crosswalk is unavailable"}
    crosswalk, audit = _load_accession_crosswalk(crosswalk_path)
    connection = sqlite3.connect(f"file:{chembl_path}?mode=ro", uri=True)
    try:
        schema = _resolve_schema(connection)
        if schema is None:
            return {}, {"status": "DATA_BLOCKED", "reason": "ChEMBL36 SQLite lacks required audited tables", "available_tables": _table_columns(connection)}
        placeholders = ",".join("?" for _ in pert_ids)
        if schema["kind"] == "fixture":
            identities = connection.execute(f"SELECT pert_id, chembl_id FROM {schema['identity']} WHERE pert_id IN ({placeholders})", pert_ids).fetchall()
            target_rows = connection.execute(
                f"SELECT i.pert_id,t.target_accession FROM {schema['identity']} i JOIN {schema['interaction']} t ON i.chembl_id=t.chembl_id WHERE i.pert_id IN ({placeholders}) AND t.direct_interaction=1 AND t.organism='Homo sapiens' AND t.target_type='SINGLE PROTEIN' ORDER BY i.pert_id,t.target_accession", pert_ids
            ).fetchall()
        else:
            with drugs_info_path.open(encoding="utf-8", newline="") as handle:
                inchi = {row["pert_id"]: row["inchi_key"] for row in csv.DictReader(handle) if row["pert_id"] in set(pert_ids) and row["inchi_key"]}
            connection.execute("CREATE TEMP TABLE sdst_drug (pert_id TEXT, inchi_key TEXT)")
            connection.executemany("INSERT INTO sdst_drug VALUES (?,?)", sorted(inchi.items()))
            identities = connection.execute("SELECT DISTINCT s.pert_id,m.chembl_id,m.molregno FROM sdst_drug s JOIN compound_structures c ON c.standard_inchi_key=s.inchi_key JOIN molecule_dictionary m ON m.molregno=c.molregno ORDER BY s.pert_id,m.chembl_id").fetchall()
            target_rows = connection.execute("SELECT DISTINCT i.pert_id,cs.accession FROM (SELECT DISTINCT s.pert_id,m.chembl_id,m.molregno FROM sdst_drug s JOIN compound_structures c ON c.standard_inchi_key=s.inchi_key JOIN molecule_dictionary m ON m.molregno=c.molregno) i JOIN drug_mechanism dm ON dm.molregno=i.molregno JOIN target_dictionary td ON td.tid=dm.tid JOIN target_components tc ON tc.tid=td.tid JOIN component_sequences cs ON cs.component_id=tc.component_id WHERE td.organism='Homo sapiens' AND td.target_type='SINGLE PROTEIN' AND dm.direct_interaction=1 AND cs.accession IS NOT NULL ORDER BY i.pert_id,cs.accession").fetchall()
    finally:
        connection.close()
    mapped = {str(row[0]): [] for row in identities}
    for pert_id, accession in target_rows:
        if (entrez := crosswalk.get(str(accession))) is not None:
            mapped[str(pert_id)].append(entrez)
    for pert_id in mapped:
        mapped[pert_id] = sorted(set(mapped[pert_id]), key=lambda value: (len(value), value))
    return mapped, {"status": "OK", "identity_rows": len(identities), "direct_target_rows": len(target_rows), "mapped_target_rows": sum(bool(v) for v in mapped.values()), "schema": schema, "crosswalk": audit}


def _resolve_physical_links(path: Path) -> Path | None:
    if "physical.links" in path.name:
        return path
    candidate = path.with_name(path.name.replace("protein.links", "protein.physical.links"))
    return candidate if candidate.exists() else None


def _string_neighbors(string_path: Path, direct_targets: set[str], frozen_axis: set[str]) -> tuple[dict[str, set[str]], dict[str, Any]]:
    physical_path = _resolve_physical_links(string_path)
    aliases_path = string_path.with_name(string_path.name.replace("protein.links", "protein.aliases").replace("protein.physical.links", "protein.aliases"))
    if physical_path is None or not aliases_path.exists():
        return {}, {"status": "NOT_USED_DATA_BLOCKED", "reason": "STRING physical links and audited aliases files are both required", "input_path": str(string_path)}
    wanted = direct_targets | frozen_axis
    protein_to_entrez: dict[str, set[str]] = {}
    with _open_text(aliases_path) as handle:
        header = next(handle, "").rstrip("\n").split("\t")
        if header != ["#string_protein_id", "alias", "source"]:
            return {}, {"status": "NOT_USED_DATA_BLOCKED", "reason": "unexpected STRING aliases schema", "header": header}
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 3 and fields[2] == "Ensembl_EntrezGene" and fields[1] in wanted:
                protein_to_entrez.setdefault(fields[0], set()).add(fields[1])
    if not protein_to_entrez:
        return {}, {"status": "NOT_USED_DATA_BLOCKED", "reason": "no official STRING Ensembl_EntrezGene aliases map to targets or frozen axis"}
    neighbors: dict[str, set[str]] = {target: set() for target in direct_targets}
    with _open_text(physical_path) as handle:
        header = next(handle, "").strip().split()
        if header != ["protein1", "protein2", "combined_score"]:
            return {}, {"status": "NOT_USED_DATA_BLOCKED", "reason": "unexpected STRING physical links schema", "header": header}
        for line in handle:
            fields = line.split()
            if len(fields) != 3 or int(fields[2]) < STRING_PHYSICAL_SCORE_MIN:
                continue
            left, right = protein_to_entrez.get(fields[0], set()), protein_to_entrez.get(fields[1], set())
            for seed in left & direct_targets:
                neighbors[seed].update(right & frozen_axis)
            for seed in right & direct_targets:
                neighbors[seed].update(left & frozen_axis)
    audit = {"status": "OK", "links_path": str(physical_path), "links_sha256": _sha256(physical_path), "aliases_path": str(aliases_path), "aliases_sha256": _sha256(aliases_path), "id_resolution": "STRING protein ID -> STRING aliases source Ensembl_EntrezGene -> Entrez; no symbols used", "physical_score_min": STRING_PHYSICAL_SCORE_MIN, "mapped_string_proteins": len(protein_to_entrez), "seed_targets": len(direct_targets), "qualified_neighbor_edges": sum(len(values) for values in neighbors.values())}
    if not any(neighbors.values()):
        audit.update({"status": "NOT_USED_DATA_BLOCKED", "reason": "STRING aliases mapped, but no qualifying high-confidence physical neighbors reached the frozen axis"})
    return neighbors, audit


def _go_bp_summary(gmt_path: Path, drug_genes: dict[str, set[str]], frozen_axis: set[str]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    terms: dict[str, set[str]] = {}
    try:
        with gmt_path.open(encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 3:
                    continue
                members = set(fields[2:]) & frozen_axis
                if members and all(member.isdigit() for member in members):
                    terms[fields[1]] = members
    except (OSError, UnicodeError) as exc:
        return np.zeros((len(drug_genes), 0), dtype=np.float32), [], {"status": "DATA_BLOCKED", "reason": f"GO BP GMT is unavailable or unreadable: {exc}", "source": str(gmt_path), "summary_dimension": 0, "summary_nonzero_drugs": 0}
    term_ids = sorted(terms)
    matrix = np.zeros((len(drug_genes), len(term_ids)), dtype=np.float32)
    for row, pert_id in enumerate(drug_genes):
        genes = drug_genes[pert_id]
        for column, term_id in enumerate(term_ids):
            if genes & terms[term_id]:
                matrix[row, column] = 1.0
    status = "OK" if term_ids else "NOT_USED_DATA_BLOCKED"
    return matrix, term_ids, {"status": status, "reason": None if term_ids else "GO BP GMT has no Entrez members on the frozen axis; gene-symbol conversion is intentionally forbidden without an official crosswalk", "source": str(gmt_path), "sha256": _sha256(gmt_path), "term_universe_rule": "sorted GO term identifiers whose GMT members are explicit Entrez IDs on frozen 978 axis", "summary_dimension": len(term_ids), "summary_nonzero_drugs": int(np.count_nonzero(matrix.sum(axis=1)))}


def build_feature_contract(*, h5ad_path: Path, chembl_path: Path, string_path: Path, go_bp_gmt_path: Path, output_dir: Path, crosswalk_path: Path | None = DEFAULT_CROSSWALK) -> dict[str, Any]:
    h5ad_path, chembl_path, string_path, go_bp_gmt_path, output_dir = map(Path, (h5ad_path, chembl_path, string_path, go_bp_gmt_path, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    sdst = ad.read_h5ad(h5ad_path, backed="r")
    source_shape = list(sdst.shape)
    try:
        required = {"pert_id", "pert_idx", "split_cold_drug_1", "split_cold_cell_1"}
        if missing := required.difference(sdst.obs.columns):
            raise ValueError(f"official SDST missing required obs fields: {sorted(missing)}")
        if "gene_id" not in sdst.var:
            raise ValueError("official SDST missing var.gene_id Entrez axis")
        index = sdst.obs[["pert_id", "pert_idx"]].copy()
        index["pert_id"], index["pert_idx"] = index["pert_id"].astype(str), index["pert_idx"].astype(int)
        conflicting_idx = index.groupby("pert_idx")["pert_id"].nunique()
        if (conflicting_idx > 1).any():
            raise ValueError("multiple pert_id values share an official pert_idx")
        index = index.drop_duplicates("pert_idx").sort_values("pert_idx", kind="stable")
        ordered_ids, ordered_idx = index["pert_id"].tolist(), index["pert_idx"].tolist()
        entrez_ids = sdst.var["gene_id"].astype(str).tolist()
        if len(entrez_ids) != 978 or len(set(entrez_ids)) != 978:
            raise ValueError("frozen 978 Entrez axis is not unique and exact")
        split_summary = {name: {str(k): int(v) for k, v in sdst.obs[name].value_counts().items()} for name in ("split_cold_drug_1", "split_cold_cell_1")}
    finally:
        sdst.file.close()
    targets, mapping = _load_mappings(chembl_path, ordered_ids, crosswalk_path, DEFAULT_DRUGS_INFO)
    frozen = set(entrez_ids)
    all_direct = set().union(*targets.values()) if targets else set()
    target_block = np.zeros((len(ordered_ids), len(entrez_ids)), dtype=np.float32)
    axis = {entrez: column for column, entrez in enumerate(entrez_ids)}
    for row, pert_id in enumerate(ordered_ids):
        for entrez in targets.get(pert_id, []):
            if entrez in axis:
                target_block[row, axis[entrez]] = 1.0
    neighbors_by_seed, string_audit = _string_neighbors(string_path, all_direct, frozen)
    string_block = np.zeros_like(target_block)
    pathway_genes: dict[str, set[str]] = {}
    for row, pert_id in enumerate(ordered_ids):
        genes = set(targets.get(pert_id, []))
        neighbor_genes = set().union(*(neighbors_by_seed.get(seed, set()) for seed in genes)) if genes else set()
        pathway_genes[pert_id] = genes | neighbor_genes
        for entrez in neighbor_genes:
            string_block[row, axis[entrez]] = 1.0
    go_block, go_term_ids, go_audit = _go_bp_summary(go_bp_gmt_path, pathway_genes, frozen)
    matrix = np.concatenate((target_block, string_block, go_block), axis=1)
    nonzero = int(np.count_nonzero(matrix.sum(axis=1)))
    coverage = nonzero / len(ordered_ids)
    blockers = [audit["reason"] for audit in (mapping, string_audit, go_audit) if audit["status"] not in {"OK"} and audit.get("reason")]
    if coverage < 0.30:
        blockers.append("complete target-plus-pathway feature coverage is below the required 30% threshold")
    status = "READY_FOR_TRAINING" if not blockers else "DATA_BLOCKED"
    feature_path = output_dir / "target_pathway_features.npy"
    np.save(feature_path, matrix)
    contract: dict[str, Any] = {
        "format": "exp008_target_pathway_feature_contract_v2", "exp_id": "EXP-008", "status": status,
        "reason": "; ".join(blockers) if blockers else None,
        "source": {"sdst_h5ad": str(h5ad_path), "sdst_h5ad_sha256": _sha256(h5ad_path), "chembl36_sqlite": str(chembl_path), "chembl36_sqlite_sha256": _sha256_if_readable(chembl_path), "uniprot_entrez_crosswalk": str(crosswalk_path) if crosswalk_path else None, "uniprot_entrez_crosswalk_sha256": _sha256(crosswalk_path) if crosswalk_path and crosswalk_path.exists() else None, "string_v12_requested": str(string_path), "go_bp_human_gmt": str(go_bp_gmt_path), "go_bp_human_gmt_sha256": _sha256(go_bp_gmt_path) if go_bp_gmt_path.exists() else None},
        "official_sdst": {"shape": source_shape, "split_summary": split_summary},
        "drug_index": {"identity_field": "obs.pert_id", "index_field": "obs.pert_idx", "ordered_pert_idx": ordered_idx, "ordered_pert_ids": ordered_ids, "feature_row_rule": "row position is sorted unique official obs.pert_idx; retained-index gaps are not matrix offsets"},
        "gene_axis": {"field": "var.gene_id", "entrez_ids": entrez_ids, "sha256": _axis_sha256(entrez_ids)},
        "mapping": {"identity_join": "exact approved ChEMBL identity; no name matching", "target_filter": "Homo sapiens; SINGLE PROTEIN; direct_interaction = 1", "accession_join": "only singleton mapping_status=mapped, ambiguous=false UniProt crosswalk rows", "annotated_drugs": sum(bool(targets.get(p)) for p in ordered_ids), "direct_target_978_nonzero_drugs": int(np.count_nonzero(target_block.sum(axis=1))), "audit": mapping},
        "string": string_audit, "go_bp": {**go_audit, "term_ids": go_term_ids},
        "features": {"relative_path": feature_path.name, "shape": list(matrix.shape), "dtype": str(matrix.dtype), "sha256": _sha256(feature_path), "blocks": [{"name": "direct_target_978", "columns": len(entrez_ids)}, {"name": "string_physical_neighbor_978", "columns": len(entrez_ids)}, {"name": "go_bp_summary", "columns": len(go_term_ids)}], "nonzero_drugs": nonzero, "coverage": coverage, "zero_vector_drugs": int((matrix.sum(axis=1) == 0).sum())},
        "forbidden_data": {"efficacy_data_read": False, "response_values_read": False, "test_labels_used_for_feature_selection": False, "raw_sdst_modified": False},
    }
    (output_dir / "MECHANISM_CONTRACT.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": status, "output_sha256": contract["features"]["sha256"], "forbidden_data": contract["forbidden_data"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD); parser.add_argument("--chembl", type=Path, default=DEFAULT_CHEMBL)
    parser.add_argument("--string", type=Path, default=DEFAULT_STRING); parser.add_argument("--go-bp-gmt", type=Path, default=DEFAULT_GO_BP)
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build_feature_contract(h5ad_path=args.h5ad, chembl_path=args.chembl, string_path=args.string, go_bp_gmt_path=args.go_bp_gmt, output_dir=args.output_dir, crosswalk_path=args.crosswalk), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
