"""Build response-blind BindingDB teacher and frozen XPert bridge contracts for EXP-009.

The teacher table intentionally preserves every valid BindingDB structural affinity record.
The separate bridge is the only SDST-facing identity step and uses exact InChIKey matching
against the frozen XPert drug registry. No SDST response, efficacy, disease, or external
validation labels are read.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any


REQUIRED_RELEASE_FIELDS = ("release_id", "source_url", "source_sha256")
REQUIRED_TSV_COLUMNS = (
    "bindingdb_id",
    "inchi_key",
    "canonical_smiles",
    "uniprot_id",
    "organism",
    "measurement_type",
    "value",
    "unit",
)
TEACHER_COLUMNS = (
    "bindingdb_id",
    "inchi_key",
    "canonical_smiles",
    "uniprot_id",
    "organism",
    "measurement_type",
    "value",
    "unit",
    "paffinity",
)
BRIDGE_COLUMNS = ("bindingdb_id", "inchi_key", "xpert_pert_id")
ACCEPTED_MEASUREMENTS = {"Ki", "Kd", "IC50", "EC50"}
UNIT_TO_MOLAR = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9, "pM": 1e-12}
FORBIDDEN_INPUTS = ["SDST response", "drug efficacy", "disease signature", "external validation labels"]


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_release_metadata(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("release metadata must be a JSON object")
    missing = [field for field in REQUIRED_RELEASE_FIELDS if not str(payload.get(field, "")).strip()]
    if missing:
        raise ValueError(f"release metadata missing required fields: {', '.join(missing)}")
    source_url = str(payload["source_url"])
    source_sha256 = str(payload["source_sha256"]).lower()
    if not source_url.startswith("https://") or "bindingdb.org" not in source_url.lower():
        raise ValueError("source_url must be an official https://bindingdb.org archive URL")
    if len(source_sha256) != 64 or any(char not in "0123456789abcdef" for char in source_sha256):
        raise ValueError("source_sha256 must be a 64-character lowercase SHA-256 digest")
    return {
        "release_id": str(payload["release_id"]),
        "source_url": source_url,
        "source_sha256": source_sha256,
    }


def load_xpert_structures(path: Path) -> dict[str, tuple[str, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "xpert_drug_registry_v1":
        raise ValueError("XPert registry must use format xpert_drug_registry_v1")
    structure_to_perturbagens: dict[str, list[str]] = {}
    for drug in payload.get("drugs", []):
        if not drug.get("global_inference_eligible"):
            continue
        inchi_key = str(drug.get("inchi_key", "")).strip().upper()
        pert_id = str(drug.get("pert_id", "")).strip()
        if not inchi_key or not pert_id:
            continue
        perturbagens = structure_to_perturbagens.setdefault(inchi_key, [])
        if pert_id not in perturbagens:
            perturbagens.append(pert_id)
    if not structure_to_perturbagens:
        raise ValueError("XPert registry contains no globally eligible structural identities")
    return {inchi_key: tuple(perturbagens) for inchi_key, perturbagens in structure_to_perturbagens.items()}


def paffinity(value: str, unit: str) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("nonpositive_or_invalid_value") from error
    normalized_unit = str(unit).strip()
    if not math.isfinite(numeric_value) or numeric_value <= 0 or normalized_unit not in UNIT_TO_MOLAR:
        raise ValueError("nonpositive_or_invalid_value")
    return -math.log10(numeric_value * UNIT_TO_MOLAR[normalized_unit])


def _required_columns(reader: csv.DictReader[str]) -> None:
    present = set(reader.fieldnames or [])
    missing = [column for column in REQUIRED_TSV_COLUMNS if column not in present]
    if missing:
        raise ValueError(f"BindingDB TSV missing required columns: {', '.join(missing)}")


class _HashingWriter:
    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self.digest = sha256()

    def write(self, value: str) -> int:
        self.digest.update(value.encode("utf-8"))
        return self._handle.write(value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


def _partial_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.partial")


def build_contract(
    bindingdb_tsv: Path,
    xpert_registry: Path,
    release_metadata: Path,
    output_dir: Path,
) -> dict[str, Any]:
    release = load_release_metadata(release_metadata)
    xpert_by_inchi_key = load_xpert_structures(xpert_registry)
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_path = output_dir / "bindingdb_teacher.tsv"
    bridge_path = output_dir / "xpert_bridge.tsv"
    audit_path = output_dir / "audit.json"
    teacher_partial = _partial_path(teacher_path)
    bridge_partial = _partial_path(bridge_path)
    audit_partial = _partial_path(audit_path)
    rejected: Counter[str] = Counter()
    teacher_records = 0
    bridge_records = 0
    batch_size = 10_000
    try:
        with tempfile.TemporaryDirectory(prefix="exp009_bindingdb_") as temporary_directory:
            database_path = Path(temporary_directory) / "streaming_index.sqlite3"
            connection = sqlite3.connect(database_path)
            try:
                with bindingdb_tsv.open(encoding="utf-8", newline="") as source_handle, teacher_partial.open(
                    "w", encoding="utf-8", newline=""
                ) as teacher_handle, bridge_partial.open("w", encoding="utf-8", newline="") as bridge_handle:
                    connection.execute("CREATE TABLE unique_teacher_inchi_key (value TEXT PRIMARY KEY)")
                    connection.execute("CREATE TABLE unique_uniprot (value TEXT PRIMARY KEY)")
                    connection.execute("CREATE TABLE covered_xpert_pert_id (value TEXT PRIMARY KEY)")
                    teacher_output = _HashingWriter(teacher_handle)
                    bridge_output = _HashingWriter(bridge_handle)
                    teacher_writer = csv.DictWriter(teacher_output, fieldnames=TEACHER_COLUMNS, delimiter="\t")
                    bridge_writer = csv.DictWriter(bridge_output, fieldnames=BRIDGE_COLUMNS, delimiter="\t")
                    teacher_writer.writeheader()
                    bridge_writer.writeheader()
                    reader = csv.DictReader(source_handle, delimiter="\t")
                    _required_columns(reader)
                    for row in reader:
                        inchi_key = str(row["inchi_key"] or "").strip().upper()
                        smiles = str(row["canonical_smiles"] or "").strip()
                        if not inchi_key or not smiles:
                            rejected["missing_structure_identity"] += 1
                            continue
                        if str(row["organism"] or "").strip() != "Homo sapiens":
                            rejected["not_homo_sapiens"] += 1
                            continue
                        uniprot_id = str(row["uniprot_id"] or "").strip().upper()
                        if not uniprot_id:
                            rejected["missing_uniprot_id"] += 1
                            continue
                        measurement_type = str(row["measurement_type"] or "").strip()
                        if measurement_type not in ACCEPTED_MEASUREMENTS:
                            rejected["unsupported_measurement_type"] += 1
                            continue
                        try:
                            affinity = paffinity(str(row["value"] or ""), str(row["unit"] or ""))
                        except ValueError as error:
                            rejected[str(error)] += 1
                            continue
                        teacher_record = {
                            "bindingdb_id": str(row["bindingdb_id"] or "").strip(),
                            "inchi_key": inchi_key,
                            "canonical_smiles": smiles,
                            "uniprot_id": uniprot_id,
                            "organism": "Homo sapiens",
                            "measurement_type": measurement_type,
                            "value": str(row["value"]).strip(),
                            "unit": str(row["unit"]).strip(),
                            "paffinity": format(affinity, ".8g"),
                        }
                        teacher_writer.writerow(teacher_record)
                        teacher_records += 1
                        connection.execute("INSERT OR IGNORE INTO unique_teacher_inchi_key VALUES (?)", (inchi_key,))
                        connection.execute("INSERT OR IGNORE INTO unique_uniprot VALUES (?)", (uniprot_id,))
                        pert_ids = xpert_by_inchi_key.get(inchi_key, ())
                        for pert_id in pert_ids:
                            bridge_writer.writerow({
                                "bindingdb_id": teacher_record["bindingdb_id"],
                                "inchi_key": inchi_key,
                                "xpert_pert_id": pert_id,
                            })
                            bridge_records += 1
                            connection.execute("INSERT OR IGNORE INTO covered_xpert_pert_id VALUES (?)", (pert_id,))
                        if teacher_records % batch_size == 0:
                            connection.commit()
                    connection.commit()
                    unique_inchi_keys = connection.execute("SELECT COUNT(*) FROM unique_teacher_inchi_key").fetchone()[0]
                    unique_uniprot_ids = connection.execute("SELECT COUNT(*) FROM unique_uniprot").fetchone()[0]
                    covered_xpert_drugs = connection.execute("SELECT COUNT(*) FROM covered_xpert_pert_id").fetchone()[0]
                    teacher_checksum = teacher_output.digest.hexdigest()
                    bridge_checksum = bridge_output.digest.hexdigest()
            finally:
                connection.close()

        audit: dict[str, Any] = {
            "format": "exp009_bindingdb_teacher_bridge_contract_audit_v3",
            "experiment_id": "EXP-009",
            "chain": "BindingDB_teacher_to_XPert_SDST_Delta978_bridge",
            "release": release,
            "input_assets": {
                "bindingdb_tsv": str(bindingdb_tsv),
                "bindingdb_tsv_sha256": file_sha256(bindingdb_tsv),
                "xpert_drug_registry": str(xpert_registry),
                "xpert_drug_registry_sha256": file_sha256(xpert_registry),
            },
            "teacher_inclusion_rules": {
                "structure_identity": "nonempty InChIKey plus canonical SMILES",
                "organism": "Homo sapiens",
                "uniprot": "nonempty parsed UniProt accession field",
                "measurements": sorted(ACCEPTED_MEASUREMENTS),
                "quantity": "strictly positive finite value with M/mM/uM/nM/pM unit",
                "paffinity": "-log10(value_in_molar)",
            },
            "bridge_rule": "exact InChIKey join from BindingDB teacher rows to every frozen global_inference_eligible XPert pert_id sharing that exact key; no name or alias matching",
            "ordering_policy": "input_row_order_then_fixed_affinity_order",
            "streaming": True,
            "max_in_memory_records": 0,
            "checksum_compatibility": "legacy sorted-output checksums are not directly comparable to this input-order streaming output.",
            "temporary_index": {
                "backend": "sqlite3",
                "version": sqlite3.sqlite_version,
                "retained": False,
                "transaction_batch_policy": f"commit every {batch_size} accepted teacher records and at end of stream",
            },
            "forbidden_inputs": FORBIDDEN_INPUTS,
            "response_values_read": False,
            "efficacy_values_read": False,
            "rejected_records": dict(sorted(rejected.items())),
            "teacher": {
                "records": teacher_records,
                "unique_inchi_keys": unique_inchi_keys,
                "unique_uniprot_ids": unique_uniprot_ids,
                "sha256": teacher_checksum,
            },
            "bridge": {
                "records": bridge_records,
                "covered_xpert_drugs": covered_xpert_drugs,
                "eligible_xpert_drugs": sum(len(pert_ids) for pert_ids in xpert_by_inchi_key.values()),
                "coverage_fraction": covered_xpert_drugs / sum(
                    len(pert_ids) for pert_ids in xpert_by_inchi_key.values()
                ),
                "sha256": bridge_checksum,
            },
            "outputs": {
                "teacher_table": str(teacher_path),
                "teacher_table_sha256": teacher_checksum,
                "xpert_bridge": str(bridge_path),
                "xpert_bridge_sha256": bridge_checksum,
            },
        }
        audit_partial.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(teacher_partial, teacher_path)
        os.replace(bridge_partial, bridge_path)
        os.replace(audit_partial, audit_path)
        return audit
    except Exception:
        teacher_partial.unlink(missing_ok=True)
        bridge_partial.unlink(missing_ok=True)
        audit_partial.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindingdb-tsv", type=Path, required=True)
    parser.add_argument("--xpert-drug-registry", dest="xpert_registry", type=Path, required=True)
    parser.add_argument("--release-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_contract(**vars(args)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
