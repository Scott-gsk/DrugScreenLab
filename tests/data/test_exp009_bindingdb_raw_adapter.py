from __future__ import annotations

import csv
from hashlib import sha256
import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "data" / "normalize_exp009_bindingdb_raw.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("exp009_bindingdb_raw_adapter", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_raw(path: Path) -> None:
    columns = [
        "BindingDB Reactant_set_id",
        "Ligand SMILES",
        "Ligand InChI Key",
        "Number of Protein Chains in Target (>1 implies a multichain complex)",
        "UniProt (SwissProt) Primary ID of Target Chain 1",
        "UniProt (TrEMBL) Primary ID of Target Chain 1",
        "Target Source Organism According to Curator or DataSource",
        "Ki (nM)",
        "IC50 (nM)",
        "Kd (nM)",
        "EC50 (nM)",
    ]
    rows = [
        ["R1", "CCO", "KEY-1", "1", "P11111", "", "Homo sapiens", "10", "100", "", ""],
        ["R2", "CCN", "KEY-2", "1", "", "A0A000", "Homo sapiens", "", "", "50", ""],
        ["R3", "CCC", "KEY-3", "1", "P33333", "Q33333", "Homo sapiens", "20", "", "", ""],
        ["R4", "COC", "KEY-4", "1", "P44444", "", "Homo sapiens", "", "", "", "5"],
        ["R5", "CNC", "KEY-5", "1", "P55555", "", "Homo sapiens", ">100", "", "", ""],
        ["R6", "CNO", "", "1", "P66666", "", "Homo sapiens", "10", "", "", ""],
        ["R7", "", "KEY-7", "1", "P77777", "", "Homo sapiens", "10", "", "", ""],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        writer.writerows(rows)


def test_normalizes_official_columns_and_expands_affinity_cells(tmp_path: Path):
    module = _load_module()
    source = tmp_path / "BindingDB_All_202608.tsv"
    output = tmp_path / "normalized.tsv"
    audit = tmp_path / "audit.json"
    _write_raw(source)

    summary = module.normalize_bindingdb_tsv(source, output, audit)

    with output.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    assert len(records) == 4
    assert all(row["bindingdb_id"] not in {"R6", "R7"} for row in records)
    assert [(row["bindingdb_id"], row["measurement_type"], row["value"], row["uniprot_id"]) for row in records] == [
        ("R1", "Ki", "10", "P11111"),
        ("R1", "IC50", "100", "P11111"),
        ("R2", "Kd", "50", "A0A000"),
        ("R4", "EC50", "5", "P44444"),
    ]
    assert all(row["unit"] == "nM" for row in records)
    assert records[0]["source_affinity_column"] == "Ki (nM)"
    assert records[0]["source_affinity_raw"] == "10"
    assert records[-1]["source_literature_organism"] == ""
    assert summary["emitted_records"] == 4
    assert summary["rejected_rows"] == {
        "uniprot_conflict": 1,
        "censored_affinity": 1,
        "missing_structure_identity": 2,
    }
    assert summary["organism_literature_conflicts"] == 0


def test_rejects_missing_required_official_header(tmp_path: Path):
    module = _load_module()
    source = tmp_path / "bad.tsv"
    source.write_text("BindingDB Reactant_set_id\tLigand SMILES\nR1\tCCO\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required official columns"):
        module.normalize_bindingdb_tsv(source, tmp_path / "out.tsv", tmp_path / "audit.json")


def test_normalizes_explicit_single_chain1_real_schema_and_rejects_multichain(tmp_path: Path):
    module = _load_module()
    source = tmp_path / "BindingDB_All_202608.tsv"
    output = tmp_path / "normalized.tsv"
    audit = tmp_path / "audit.json"
    columns = [
        "BindingDB Reactant_set_id",
        "Ligand SMILES",
        "Ligand InChI Key",
        "Number of Protein Chains in Target (>1 implies a multichain complex)",
        "UniProt (SwissProt) Primary ID of Target Chain 1",
        "UniProt (TrEMBL) Primary ID of Target Chain 1",
        "Target Source Organism According to Curator or DataSource",
        "Ki (nM)",
        "Kd (nM)",
        "IC50 (nM)",
        "EC50 (nM)",
    ]
    rows = [
        ["R-single", "CCO", "KEY-SINGLE", "1", "P11111", "", "Homo sapiens", "10", "", "", ""],
        ["R-multi", "CCN", "KEY-MULTI", "2", "P22222", "", "Homo sapiens", "20", "", "", ""],
    ]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        writer.writerows(rows)

    summary = module.normalize_bindingdb_tsv(source, output, audit)

    with output.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    assert records == [
        {
            "bindingdb_id": "R-single",
            "inchi_key": "KEY-SINGLE",
            "canonical_smiles": "CCO",
            "uniprot_id": "P11111",
            "organism": "Homo sapiens",
            "measurement_type": "Ki",
            "value": "10",
            "unit": "nM",
            "source_reactant_set_id": "R-single",
            "source_affinity_column": "Ki (nM)",
            "source_affinity_raw": "10",
            "source_uniprot_column": "UniProt (SwissProt) Primary ID of Target Chain 1",
            "source_organism_column": "Target Source Organism According to Curator or DataSource",
            "source_literature_organism": "",
        }
    ]
    assert summary["emitted_records"] == 1
    assert summary["rejected_rows"] == {"multichain_target": 1}
    assert summary["organism_literature_conflicts"] == 0


def test_normalizes_unordered_source_in_row_and_fixed_affinity_order(tmp_path: Path):
    module = _load_module()
    source = tmp_path / "unordered.tsv"
    output = tmp_path / "normalized.tsv"
    audit = tmp_path / "audit.json"
    columns = [
        "BindingDB Reactant_set_id",
        "Ligand SMILES",
        "Ligand InChI Key",
        "Number of Protein Chains in Target (>1 implies a multichain complex)",
        "UniProt (SwissProt) Primary ID of Target Chain 1",
        "UniProt (TrEMBL) Primary ID of Target Chain 1",
        "Target Source Organism According to Curator or DataSource",
        "Ki (nM)",
        "Kd (nM)",
        "IC50 (nM)",
        "EC50 (nM)",
    ]
    rows = [
        ["R9", "CCO", "KEY-9", "1", "P99999", "", "Homo sapiens", "9", "", "90", ""],
        ["R1", "CCN", "KEY-1", "1", "P11111", "", "Homo sapiens", "1", "10", "", ""],
    ]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        writer.writerows(rows)

    first = module.normalize_bindingdb_tsv(source, output, audit)
    first_bytes = output.read_bytes()
    first_hash = sha256(first_bytes).hexdigest()
    second = module.normalize_bindingdb_tsv(source, output, audit)

    with output.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    assert [(row["bindingdb_id"], row["measurement_type"]) for row in records] == [
        ("R9", "Ki"),
        ("R9", "IC50"),
        ("R1", "Ki"),
        ("R1", "Kd"),
    ]
    assert output.read_bytes() == first_bytes
    assert sha256(output.read_bytes()).hexdigest() == first_hash
    assert first["ordering_policy"] == "input_row_order_then_fixed_affinity_order"
    assert first["streaming"] is True
    assert first["max_in_memory_records"] == 0
    assert second["emitted_records"] == first["emitted_records"]


def test_normalization_failure_preserves_existing_final_outputs_and_cleans_partials(tmp_path: Path):
    module = _load_module()
    source = tmp_path / "invalid.tsv"
    output = tmp_path / "normalized.tsv"
    audit = tmp_path / "audit.json"
    source.write_text("BindingDB Reactant_set_id\n", encoding="utf-8")
    output.write_text("stale output", encoding="utf-8")
    audit.write_text("stale audit", encoding="utf-8")
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("preserve", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required official columns"):
        module.normalize_bindingdb_tsv(source, output, audit)

    assert output.read_text(encoding="utf-8") == "stale output"
    assert audit.read_text(encoding="utf-8") == "stale audit"
    assert unrelated.read_text(encoding="utf-8") == "preserve"
    assert not list(tmp_path.glob("*.partial"))


def test_rejects_invalid_chain_counts_without_emitting_rows(tmp_path: Path):
    module = _load_module()
    source = tmp_path / "invalid_chain_counts.tsv"
    output = tmp_path / "normalized.tsv"
    audit = tmp_path / "audit.json"
    columns = [
        "BindingDB Reactant_set_id",
        "Ligand SMILES",
        "Ligand InChI Key",
        "Number of Protein Chains in Target (>1 implies a multichain complex)",
        "UniProt (SwissProt) Primary ID of Target Chain 1",
        "UniProt (TrEMBL) Primary ID of Target Chain 1",
        "Target Source Organism According to Curator or DataSource",
        "Ki (nM)",
        "Kd (nM)",
        "IC50 (nM)",
        "EC50 (nM)",
    ]
    rows = [
        ["R-empty", "CCO", "KEY-EMPTY", "", "P11111", "", "Homo sapiens", "10", "", "", ""],
        ["R-text", "CCN", "KEY-TEXT", "one", "P22222", "", "Homo sapiens", "20", "", "", ""],
        ["R-zero", "CCC", "KEY-ZERO", "0", "P33333", "", "Homo sapiens", "30", "", "", ""],
        ["R-negative", "CCCl", "KEY-NEGATIVE", "-1", "P44444", "", "Homo sapiens", "40", "", "", ""],
    ]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        writer.writerows(rows)

    summary = module.normalize_bindingdb_tsv(source, output, audit)

    with output.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle, delimiter="\t")) == []
    assert summary["emitted_records"] == 0
    assert summary["rejected_rows"] == {"invalid_chain_count": 4}


def test_normalize_failure_preserves_previously_published_outputs(tmp_path: Path):
    module = _load_module()
    source = tmp_path / "invalid.tsv"
    output = tmp_path / "normalized.tsv"
    audit = tmp_path / "audit.json"
    source.write_text("BindingDB Reactant_set_id\nR1\n", encoding="utf-8")
    output.write_text("previous-normalized-output\n", encoding="utf-8")
    audit.write_text("{\"previous\": true}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required official columns"):
        module.normalize_bindingdb_tsv(source, output, audit)

    assert output.read_text(encoding="utf-8") == "previous-normalized-output\n"
    assert audit.read_text(encoding="utf-8") == "{\"previous\": true}\n"
    assert not list(tmp_path.glob("*.partial"))
