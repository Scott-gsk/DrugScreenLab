"""Normalize official BindingDB TSV columns for the EXP-009 teacher contract.

This adapter is intentionally response-blind and does not download archives. It converts
BindingDB's wide affinity columns into one normalized record per valid affinity cell.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import math
import os
from pathlib import Path
from typing import Any


COLUMN_MAP = {
    "bindingdb_id": "BindingDB Reactant_set_id",
    "canonical_smiles": "Ligand SMILES",
    "inchi_key": "Ligand InChI Key",
    "protein_chain_count": "Number of Protein Chains in Target (>1 implies a multichain complex)",
    "swissprot": "UniProt (SwissProt) Primary ID of Target Chain 1",
    "trembl": "UniProt (TrEMBL) Primary ID of Target Chain 1",
    "organism": "Target Source Organism According to Curator or DataSource",
}
AFFINITY_COLUMNS = {
    "Ki (nM)": "Ki",
    "Kd (nM)": "Kd",
    "IC50 (nM)": "IC50",
    "EC50 (nM)": "EC50",
}
REQUIRED_OFFICIAL_COLUMNS = tuple(COLUMN_MAP.values()) + tuple(AFFINITY_COLUMNS)
OUTPUT_COLUMNS = (
    "bindingdb_id",
    "inchi_key",
    "canonical_smiles",
    "uniprot_id",
    "organism",
    "measurement_type",
    "value",
    "unit",
    "source_reactant_set_id",
    "source_affinity_column",
    "source_affinity_raw",
    "source_uniprot_column",
    "source_organism_column",
    "source_literature_organism",
)


def _text(row: dict[str, str], column: str) -> str:
    return str(row.get(column, "") or "").strip()


def _parse_numeric(raw: str) -> float:
    value = raw.strip()
    if not value:
        raise ValueError("empty_affinity")
    if value.startswith((">", "<")):
        raise ValueError("censored_affinity")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError("invalid_affinity") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError("nonpositive_or_invalid_affinity")
    return parsed


def _parse_chain_count(raw: str) -> int:
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError("invalid_chain_count") from error


def _validate_headers(reader: csv.DictReader[str]) -> None:
    present = set(reader.fieldnames or [])
    missing = [column for column in REQUIRED_OFFICIAL_COLUMNS if column not in present]
    if missing:
        raise ValueError(f"missing required official columns: {', '.join(missing)}")


def _partial_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.partial")


def normalize_bindingdb_tsv(source: Path, output: Path, audit_path: Path) -> dict[str, Any]:
    rejected: Counter[str] = Counter()
    literature_conflicts = 0
    total_rows = 0
    emitted_records = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    output_partial = _partial_path(output)
    audit_partial = _partial_path(audit_path)
    try:
        with source.open(encoding="utf-8", newline="") as source_handle, output_partial.open(
            "w", encoding="utf-8", newline=""
        ) as output_handle:
            reader = csv.DictReader(source_handle, delimiter="\t")
            _validate_headers(reader)
            writer = csv.DictWriter(output_handle, fieldnames=OUTPUT_COLUMNS, delimiter="\t")
            writer.writeheader()
            for row in reader:
                total_rows += 1
                bindingdb_id = _text(row, COLUMN_MAP["bindingdb_id"])
                inchi_key = _text(row, COLUMN_MAP["inchi_key"])
                canonical_smiles = _text(row, COLUMN_MAP["canonical_smiles"])
                try:
                    chain_count = _parse_chain_count(_text(row, COLUMN_MAP["protein_chain_count"]))
                except ValueError as error:
                    rejected[str(error)] += 1
                    continue
                if chain_count <= 0:
                    rejected["invalid_chain_count"] += 1
                    continue
                if chain_count > 1:
                    rejected["multichain_target"] += 1
                    continue
                if not bindingdb_id:
                    rejected["missing_bindingdb_id"] += 1
                    continue
                if not inchi_key or not canonical_smiles:
                    rejected["missing_structure_identity"] += 1
                    continue
                swissprot = _text(row, COLUMN_MAP["swissprot"])
                trembl = _text(row, COLUMN_MAP["trembl"])
                if swissprot and trembl and swissprot != trembl:
                    rejected["uniprot_conflict"] += 1
                    continue
                uniprot_id = swissprot or trembl
                if not uniprot_id:
                    rejected["missing_uniprot"] += 1
                    continue
                common = {
                    "bindingdb_id": bindingdb_id,
                    "inchi_key": inchi_key,
                    "canonical_smiles": canonical_smiles,
                    "uniprot_id": uniprot_id,
                    "organism": _text(row, COLUMN_MAP["organism"]),
                    "unit": "nM",
                    "source_reactant_set_id": bindingdb_id,
                    "source_uniprot_column": COLUMN_MAP["swissprot"] if swissprot else COLUMN_MAP["trembl"],
                    "source_organism_column": COLUMN_MAP["organism"],
                    "source_literature_organism": "",
                }
                for source_column, measurement_type in AFFINITY_COLUMNS.items():
                    raw = _text(row, source_column)
                    if not raw:
                        continue
                    try:
                        _parse_numeric(raw)
                    except ValueError as error:
                        rejected[str(error)] += 1
                        continue
                    writer.writerow({
                        **common,
                        "measurement_type": measurement_type,
                        "value": raw,
                        "source_affinity_column": source_column,
                        "source_affinity_raw": raw,
                    })
                    emitted_records += 1
        audit = {
            "format": "exp009_bindingdb_raw_normalized_audit_v2",
            "source": str(source),
            "output": str(output),
            "total_source_rows": total_rows,
            "emitted_records": emitted_records,
            "rejected_rows": dict(sorted(rejected.items())),
            "organism_literature_conflicts": literature_conflicts,
            "organism_policy": "Target Source Organism According to Curator or DataSource is authoritative; no literature organism is required or emitted.",
            "chain_policy": "only strict positive integer chain count 1 is accepted; counts greater than 1 are rejected as multichain_target and empty, noninteger, zero, or negative counts as invalid_chain_count.",
            "uniprot_policy": "unique Chain 1 SwissProt preferred; otherwise unique Chain 1 TrEMBL; discordant nonempty pair rejected.",
            "affinity_policy": "each valid uncensored positive numeric Ki/Kd/IC50/EC50 nM cell emits one row; censored values are not expanded.",
            "ordering_policy": "input_row_order_then_fixed_affinity_order",
            "streaming": True,
            "max_in_memory_records": 0,
            "checksum_compatibility": "legacy sorted-output checksums are not directly comparable to this input-order streaming output.",
            "response_values_read": False,
            "efficacy_values_read": False,
        }
        audit_partial.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(output_partial, output)
        os.replace(audit_partial, audit_path)
        return audit
    except Exception:
        output_partial.unlink(missing_ok=True)
        audit_partial.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(normalize_bindingdb_tsv(args.source, args.output, args.audit), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
