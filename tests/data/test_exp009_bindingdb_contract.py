from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "data"
    / "build_exp009_bindingdb_contract.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("exp009_bindingdb_contract", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _source_metadata() -> dict[str, str]:
    return {
        "release_id": "BindingDB_2026m08",
        "source_url": "https://www.bindingdb.org/bind/chemsearch/marvin/BindingDB_All.tsv.zip",
        "source_sha256": "a" * 64,
    }


def test_build_contract_filters_to_auditable_human_quantitative_structural_records(tmp_path: Path):
    module = _load_module()
    source = tmp_path / "bindingdb_local.tsv"
    registry = tmp_path / "xpert_drugs.json"
    metadata = tmp_path / "bindingdb_release.json"
    output_dir = tmp_path / "contract"
    _write_tsv(
        source,
        [
            {
                "bindingdb_id": "B1",
                "inchi_key": "GOOD-INCHI-KEY",
                "canonical_smiles": "CCO",
                "uniprot_id": "P00533",
                "organism": "Homo sapiens",
                "measurement_type": "Ki",
                "value": "10",
                "unit": "nM",
            },
            {
                "bindingdb_id": "B2",
                "inchi_key": "GOOD-INCHI-KEY",
                "canonical_smiles": "CCO",
                "uniprot_id": "P00533",
                "organism": "Mus musculus",
                "measurement_type": "Ki",
                "value": "10",
                "unit": "nM",
            },
            {
                "bindingdb_id": "B3",
                "inchi_key": "GOOD-INCHI-KEY",
                "canonical_smiles": "CCO",
                "uniprot_id": "P00533",
                "organism": "Homo sapiens",
                "measurement_type": "IC50",
                "value": "0",
                "unit": "nM",
            },
            {
                "bindingdb_id": "B4",
                "inchi_key": "NO-XPERT-MATCH",
                "canonical_smiles": "CCC",
                "uniprot_id": "P00533",
                "organism": "Homo sapiens",
                "measurement_type": "Kd",
                "value": "1",
                "unit": "uM",
            },
        ],
    )
    registry.write_text(
        json.dumps(
            {
                "format": "xpert_drug_registry_v1",
                "drugs": [
                    {
                        "pert_id": "BRD-A",
                        "inchi_key": "GOOD-INCHI-KEY",
                        "canonical_smiles": "CCO",
                        "global_inference_eligible": True,
                    },
                    {
                        "pert_id": "BRD-B",
                        "inchi_key": "OTHER-INCHI-KEY",
                        "canonical_smiles": "CCN",
                        "global_inference_eligible": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    metadata.write_text(json.dumps(_source_metadata()), encoding="utf-8")

    audit = module.build_contract(source, registry, metadata, output_dir)

    with (output_dir / "bindingdb_teacher.tsv").open(encoding="utf-8", newline="") as handle:
        teacher = list(csv.DictReader(handle, delimiter="\t"))
    with (output_dir / "xpert_bridge.tsv").open(encoding="utf-8", newline="") as handle:
        bridge = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["bindingdb_id"] for row in teacher] == ["B1", "B4"]
    assert "xpert_pert_id" not in teacher[0]
    assert float(teacher[0]["paffinity"]) == pytest.approx(8.0)
    assert bridge == [{"bindingdb_id": "B1", "inchi_key": "GOOD-INCHI-KEY", "xpert_pert_id": "BRD-A"}]
    assert audit["teacher"] == {
        "records": 2,
        "unique_inchi_keys": 2,
        "unique_uniprot_ids": 1,
        "sha256": audit["outputs"]["teacher_table_sha256"],
    }
    assert audit["bridge"] == {
        "records": 1,
        "covered_xpert_drugs": 1,
        "eligible_xpert_drugs": 2,
        "coverage_fraction": 0.5,
        "sha256": audit["outputs"]["xpert_bridge_sha256"],
    }
    assert audit["rejected_records"] == {
        "not_homo_sapiens": 1,
        "nonpositive_or_invalid_value": 1,
    }
    assert audit["forbidden_inputs"] == ["SDST response", "drug efficacy", "disease signature", "external validation labels"]


def test_build_contract_preserves_unordered_input_and_repeated_matching_bridge_coverage(tmp_path: Path):
    module = _load_module()
    source = tmp_path / "unordered.tsv"
    registry = tmp_path / "xpert_drugs.json"
    metadata = tmp_path / "bindingdb_release.json"
    output_dir = tmp_path / "contract"
    _write_tsv(
        source,
        [
            {"bindingdb_id": "B9", "inchi_key": "MATCH-KEY", "canonical_smiles": "CCO", "uniprot_id": "P99999", "organism": "Homo sapiens", "measurement_type": "IC50", "value": "9", "unit": "nM"},
            {"bindingdb_id": "B1", "inchi_key": "OTHER-KEY", "canonical_smiles": "CCN", "uniprot_id": "P11111", "organism": "Homo sapiens", "measurement_type": "Ki", "value": "1", "unit": "nM"},
            {"bindingdb_id": "B8", "inchi_key": "MATCH-KEY", "canonical_smiles": "CCO", "uniprot_id": "P88888", "organism": "Homo sapiens", "measurement_type": "Kd", "value": "8", "unit": "nM"},
        ],
    )
    registry.write_text(json.dumps({"format": "xpert_drug_registry_v1", "drugs": [
        {"pert_id": "BRD-MATCH-A", "inchi_key": "MATCH-KEY", "global_inference_eligible": True},
        {"pert_id": "BRD-MATCH-B", "inchi_key": "MATCH-KEY", "global_inference_eligible": True},
        {"pert_id": "BRD-UNMATCHED", "inchi_key": "UNMATCHED-KEY", "global_inference_eligible": True},
    ]}), encoding="utf-8")
    metadata.write_text(json.dumps(_source_metadata()), encoding="utf-8")

    first = module.build_contract(source, registry, metadata, output_dir)
    teacher_path = output_dir / "bindingdb_teacher.tsv"
    bridge_path = output_dir / "xpert_bridge.tsv"
    teacher_bytes = teacher_path.read_bytes()
    bridge_bytes = bridge_path.read_bytes()
    second = module.build_contract(source, registry, metadata, output_dir)

    with teacher_path.open(encoding="utf-8", newline="") as handle:
        teacher = list(csv.DictReader(handle, delimiter="\t"))
    with bridge_path.open(encoding="utf-8", newline="") as handle:
        bridge = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["bindingdb_id"] for row in teacher] == ["B9", "B1", "B8"]
    assert [(row["bindingdb_id"], row["xpert_pert_id"]) for row in bridge] == [
        ("B9", "BRD-MATCH-A"),
        ("B9", "BRD-MATCH-B"),
        ("B8", "BRD-MATCH-A"),
        ("B8", "BRD-MATCH-B"),
    ]
    assert first["bridge"]["records"] == 4
    assert first["bridge"]["covered_xpert_drugs"] == 2
    assert first["bridge"]["coverage_fraction"] == 2 / 3
    assert teacher_path.read_bytes() == teacher_bytes
    assert bridge_path.read_bytes() == bridge_bytes
    assert second["outputs"]["teacher_table_sha256"] == first["outputs"]["teacher_table_sha256"]
    assert second["outputs"]["xpert_bridge_sha256"] == first["outputs"]["xpert_bridge_sha256"]
    assert first["ordering_policy"] == "input_row_order_then_fixed_affinity_order"
    assert first["streaming"] is True
    assert first["max_in_memory_records"] == 0


def test_build_contract_failure_does_not_publish_partial_outputs(tmp_path: Path):
    module = _load_module()
    source = tmp_path / "invalid.tsv"
    registry = tmp_path / "xpert_drugs.json"
    metadata = tmp_path / "bindingdb_release.json"
    output_dir = tmp_path / "contract"
    source.write_text("bindingdb_id\n", encoding="utf-8")
    registry.write_text(json.dumps({"format": "xpert_drug_registry_v1", "drugs": [
        {"pert_id": "BRD-A", "inchi_key": "MATCH-KEY", "global_inference_eligible": True},
    ]}), encoding="utf-8")
    metadata.write_text(json.dumps(_source_metadata()), encoding="utf-8")

    with pytest.raises(ValueError, match="BindingDB TSV missing required columns"):
        module.build_contract(source, registry, metadata, output_dir)

    assert not (output_dir / "bindingdb_teacher.tsv").exists()
    assert not (output_dir / "xpert_bridge.tsv").exists()
    assert not (output_dir / "audit.json").exists()
    assert list(output_dir.glob("*.partial")) == []


def test_build_contract_failure_preserves_previously_published_outputs(tmp_path: Path):
    module = _load_module()
    source = tmp_path / "invalid.tsv"
    registry = tmp_path / "xpert_drugs.json"
    metadata = tmp_path / "bindingdb_release.json"
    output_dir = tmp_path / "contract"
    output_dir.mkdir()
    source.write_text("bindingdb_id\n", encoding="utf-8")
    registry.write_text(json.dumps({"format": "xpert_drug_registry_v1", "drugs": [
        {"pert_id": "BRD-A", "inchi_key": "MATCH-KEY", "global_inference_eligible": True},
    ]}), encoding="utf-8")
    metadata.write_text(json.dumps(_source_metadata()), encoding="utf-8")
    existing = {
        "bindingdb_teacher.tsv": "previous-teacher\n",
        "xpert_bridge.tsv": "previous-bridge\n",
        "audit.json": "{\"previous\": true}\n",
    }
    for filename, content in existing.items():
        (output_dir / filename).write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="BindingDB TSV missing required columns"):
        module.build_contract(source, registry, metadata, output_dir)

    for filename, content in existing.items():
        assert (output_dir / filename).read_text(encoding="utf-8") == content
    assert list(output_dir.glob("*.partial")) == []


def test_load_xpert_structures_preserves_exact_one_to_many_inchikey_mapping(tmp_path: Path):
    module = _load_module()
    registry = tmp_path / "xpert_drugs.json"
    registry.write_text(json.dumps({"format": "xpert_drug_registry_v1", "drugs": [
        {"pert_id": "BRD-A", "inchi_key": "DUPLICATE-KEY", "global_inference_eligible": True},
        {"pert_id": "BRD-B", "inchi_key": "DUPLICATE-KEY", "global_inference_eligible": True},
        {"pert_id": "BRD-C", "inchi_key": "OTHER-KEY", "global_inference_eligible": True},
    ]}), encoding="utf-8")

    mapping = module.load_xpert_structures(registry)

    assert mapping["DUPLICATE-KEY"] == ("BRD-A", "BRD-B")
    assert mapping["OTHER-KEY"] == ("BRD-C",)


def test_cli_maps_xpert_drug_registry_argument_to_contract_parameter(tmp_path: Path):
    source = tmp_path / "bindingdb.tsv"
    registry = tmp_path / "xpert_drugs.json"
    metadata = tmp_path / "bindingdb_release.json"
    output_dir = tmp_path / "contract"
    _write_tsv(
        source,
        [
            {
                "bindingdb_id": "B1",
                "inchi_key": "MATCH-KEY",
                "canonical_smiles": "CCO",
                "uniprot_id": "P11111",
                "organism": "Homo sapiens",
                "measurement_type": "Ki",
                "value": "1",
                "unit": "nM",
            }
        ],
    )
    registry.write_text(json.dumps({"format": "xpert_drug_registry_v1", "drugs": [
        {"pert_id": "BRD-A", "inchi_key": "MATCH-KEY", "global_inference_eligible": True},
    ]}), encoding="utf-8")
    metadata.write_text(json.dumps(_source_metadata()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--bindingdb-tsv", str(source),
            "--xpert-drug-registry", str(registry),
            "--release-metadata", str(metadata),
            "--output-dir", str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["bridge"]["records"] == 1


def test_build_contract_requires_fixed_official_archive_identity(tmp_path: Path):
    module = _load_module()
    metadata = tmp_path / "incomplete_metadata.json"
    metadata.write_text(
        json.dumps({"release_id": "BindingDB_2026m08", "source_url": "https://www.bindingdb.org/archive.tsv"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_sha256"):
        module.load_release_metadata(metadata)
