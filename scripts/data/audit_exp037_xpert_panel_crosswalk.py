"""Build the Sol-authorized EXP-036/XPert 14-drug panel crosswalk only.

This deliberately narrow audit does not inspect XPert expression, labels,
checkpoints, embeddings, predictions, or any non-identity data columns.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "data/registry/datasets.json"
CONTRACT = ROOT / "data/processed/dual_route/EXP-036/direct_response_contract_v1"
CONTRACT_MANIFEST = CONTRACT / "manifest.json"
TARGET_DRUGS = CONTRACT / "drugs.parquet"
XPERT_CSV = ROOT / "data/external/xpert_source/processed_data/l1000_sdst_drugs_info_8432.csv"
OUTPUT = ROOT / "artifacts/experiments/EXP-037/20260924_xpert_panel_feature_row_crosswalk_v1"
DATASET_ID = "dual_route_exp036_direct_response_contract_v1"
XPERT_DRUG_REGISTRY_ID = "xpert_drug_registry_v1"
XPERT_COLUMNS = ("pert_id", "inchi_key", "drug_node_idx")
TARGET_COLUMNS = ("drug_index", "drug_id", "canonical_isomeric_smiles")
INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
REQUIRED_FEATURE_FLAGS = ("unimol", "hg")
EXPECTED_FEATURE_INDEX_KEY = "pert_idx == official drug_node_idx"
XPERT_SOURCE_HASHES = {
    "data/external/xpert_source/configs/config_l1000.yaml":
        "94a2206ced6e679ee9556595459cec4a871219cf94487cd74f9f83f9d1a96b41",
    "data/external/xpert_source/datasets/MyDataset.py":
        "d4cb1ed1d293d66ddff0e4f71c4931b2719c4cbbea7ece32da7d62651c6c2eda",
    "data/external/xpert_source/models/model_XPert.py":
        "87b08745515f72813a622170d2df774d3b2aac0975e0b93e07008ad67772e5ba",
    "data/external/xpert_source/utils.py":
        "da15f152943a602b3c38d8e1ba2365e4e47c74c16b16e37423d28f8132faa95a",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stat_signature(stream: BinaryIO) -> tuple[int, int, int, int, int]:
    stat = os.fstat(stream.fileno())
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _hash_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    stream.seek(0)
    return digest.hexdigest()


def _read_stable_json(path: Path) -> tuple[object, str]:
    with path.open("rb") as stream:
        before = _stat_signature(stream)
        content = stream.read()
        after = _stat_signature(stream)
    if before != after:
        raise ValueError(f"file changed while reading: {path.name}")
    digest = hashlib.sha256(content).hexdigest()
    return json.loads(content.decode("utf-8")), digest


def _record(stream: BinaryIO, selected: set[int] | None) -> tuple[dict[int, bytes], int] | None:
    """Read one RFC-4180-style record, retaining only selected data fields.

    With selected=None this is used only for the header. For data rows, bytes
    in all other fields are scanned for CSV structure but never decoded or
    retained.
    """
    capture_all = selected is None
    values: dict[int, bytes] = {}
    field_index = 0
    field_started = False
    in_quotes = False
    after_quote = False
    row_started = False
    buffer: bytearray | None = bytearray() if capture_all or 0 in (selected or set()) else None
    pending: bytes | None = None

    def append(value: bytes) -> None:
        if buffer is not None:
            buffer.extend(value)

    def finish_field() -> None:
        nonlocal buffer
        if capture_all or field_index in (selected or set()):
            values[field_index] = bytes(buffer or b"")

    def next_buffer(index: int) -> bytearray | None:
        return bytearray() if capture_all or index in (selected or set()) else None

    while True:
        if pending is not None:
            byte, pending = pending, None
        else:
            byte = stream.read(1)
        if not byte:
            if in_quotes:
                raise ValueError("unterminated quoted CSV field")
            if not row_started:
                return None
            finish_field()
            return values, field_index + 1

        row_started = True
        if in_quotes:
            if byte == b'"':
                following = stream.read(1)
                if following == b'"':
                    append(b'"')
                else:
                    in_quotes = False
                    after_quote = True
                    pending = following or None
            else:
                append(byte)
            continue

        if after_quote and byte not in (b",", b"\r", b"\n"):
            raise ValueError("unexpected byte after a quoted CSV field")
        if byte == b'"':
            if field_started:
                raise ValueError("quote inside an unquoted CSV field")
            in_quotes = True
            field_started = True
        elif byte == b",":
            finish_field()
            field_index += 1
            buffer = next_buffer(field_index)
            field_started = False
            after_quote = False
        elif byte in (b"\r", b"\n"):
            if byte == b"\r":
                following = stream.read(1)
                if following not in (b"", b"\n"):
                    pending = following
            finish_field()
            return values, field_index + 1
        else:
            append(byte)
            field_started = True
            after_quote = False


def _selected_xpert_rows_stream(stream: BinaryIO, panel_keys: set[str]) -> tuple[list[dict[str, str]], int]:
    """Stream the approved three columns; discard all other field bytes."""
    header_record = _record(stream, selected=None)
    if header_record is None:
        raise ValueError("XPert CSV is empty")
    header_bytes, _ = header_record
    header = [header_bytes[i].decode("utf-8-sig" if i == 0 else "utf-8", errors="strict")
              for i in range(len(header_bytes))]
    if any(header.count(name) != 1 for name in XPERT_COLUMNS):
        raise ValueError("approved XPert identity columns are missing or duplicated")
    indexes = {name: header.index(name) for name in XPERT_COLUMNS}
    selected_indexes = set(indexes.values())
    chosen: list[dict[str, str]] = []
    data_rows = 0
    while (item := _record(stream, selected=selected_indexes)) is not None:
        cells, width = item
        data_rows += 1
        if width <= max(selected_indexes):
            raise ValueError(f"CSV record {data_rows} is missing an approved identity field")
        row = {name: cells[index].decode("utf-8", errors="strict")
               for name, index in indexes.items()}
        if row["inchi_key"] in panel_keys:
            chosen.append(row)
    return chosen, data_rows


def selected_xpert_rows(path: Path, panel_keys: set[str]) -> tuple[list[dict[str, str]], int]:
    with path.open("rb") as stream:
        return _selected_xpert_rows_stream(stream, panel_keys)


def _rdkit_smiles_identity(smiles: str) -> tuple[str, str]:
    from rdkit import Chem
    from rdkit.Chem import inchi

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("invalid SMILES in an approved identity asset")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    return canonical, inchi.MolToInchiKey(molecule)


def _validate_official_source_snapshot() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in XPERT_SOURCE_HASHES.items():
        path = ROOT / relative
        actual = sha256(path)
        if actual.lower() != expected.lower():
            raise ValueError(f"approved XPert source snapshot changed: {Path(relative).name}")
        observed[relative] = actual
    return observed


def _load_registered_drug_registry(registry: list[dict[str, object]]) -> tuple[dict[str, object], str, str]:
    entries = [item for item in registry if item.get("id") == XPERT_DRUG_REGISTRY_ID]
    if len(entries) != 1:
        raise ValueError("registered XPert drug identity sidecar is not unique")
    entry = entries[0]
    if (entry.get("version") != 1 or entry.get("immutable") is not True
            or entry.get("asset_type") != "derived_metadata"
            or entry.get("intended_role") != "XPERT_FOUNDATION_PERTURBAGEN_REGISTRY"):
        raise ValueError("registered XPert sidecar is not the expected immutable identity asset")
    path_info = entry.get("path")
    checksum = entry.get("checksum")
    relative = path_info.get("relative") if isinstance(path_info, dict) else None
    expected_sha = checksum.get("value") if isinstance(checksum, dict) else None
    if not isinstance(relative, str) or not isinstance(expected_sha, str):
        raise ValueError("registered XPert sidecar path/checksum is incomplete")
    path = (ROOT / "data" / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("registered XPert sidecar resolves outside the project root")
    sidecar, actual_sha = _read_stable_json(path)
    if actual_sha.lower() != expected_sha.lower():
        raise ValueError("XPert sidecar bytes differ from the registered checksum")
    schema = entry.get("schema")
    if (not isinstance(sidecar, dict) or sidecar.get("format") != XPERT_DRUG_REGISTRY_ID
            or sidecar.get("identity_key") != "pert_id"
            or sidecar.get("feature_index_key") != EXPECTED_FEATURE_INDEX_KEY
            or sidecar.get("status") != "FROZEN_GLOBAL_XPERT_DRUG_REGISTRY"
            or not isinstance(schema, dict)
            or not isinstance(sidecar.get("drugs"), list)
            or len(sidecar["drugs"]) != schema.get("official_pert_ids")):
        raise ValueError("XPert sidecar format/index contract or registered record count is invalid")
    return sidecar, actual_sha, relative


def _preflight() -> tuple[list[dict[str, object]], dict[str, object], dict[str, str], str]:
    registry, registry_sha = _read_stable_json(REGISTRY)
    entries = [item for item in registry if item.get("id") == DATASET_ID]
    if len(entries) != 1:
        raise ValueError("registered EXP-036 drug contract is not unique")
    entry = entries[0]
    if (entry.get("version") != 1 or entry.get("status") != "QC_PASS"
            or entry.get("immutable") is not True
            or entry.get("schema", {}).get("n_common_drugs") != 14):
        raise ValueError("registered EXP-036 contract does not match the approved 14-drug scope")
    manifest, manifest_sha = _read_stable_json(CONTRACT_MANIFEST)
    if manifest_sha != entry.get("checksum", {}).get("value"):
        raise ValueError("EXP-036 manifest hash differs from the registered checksum")
    expected_drug_sha = manifest.get("files_sha256", {}).get("drugs.parquet")
    if (manifest.get("protocol") != "EXP-036 direct-response contract v1"
            or manifest.get("status") != "PASS"
            or manifest.get("n_common_drugs") != 14
            or not expected_drug_sha):
        raise ValueError("EXP-036 manifest does not declare the expected 14-drug contract")
    with TARGET_DRUGS.open("rb") as stream:
        before = _stat_signature(stream)
        target_sha = _hash_stream(stream)
        if target_sha != expected_drug_sha:
            raise ValueError("EXP-036 drugs.parquet hash differs from its manifest")
        target = pd.read_parquet(stream, columns=list(TARGET_COLUMNS))
        parsed_signature = _stat_signature(stream)
        sha_after = _hash_stream(stream)
        after = _stat_signature(stream)
    if before != parsed_signature or before != after or target_sha != sha_after:
        raise ValueError("EXP-036 drugs.parquet changed during hash/parse")
    if (len(target) != 14 or target["drug_index"].isna().any()
            or target["drug_id"].isna().any()
            or target["canonical_isomeric_smiles"].isna().any()):
        raise ValueError("EXP-036 panel must contain exactly 14 complete rows")
    indexes = target["drug_index"].tolist()
    if any(isinstance(value, bool) or int(value) != value for value in indexes):
        raise ValueError("EXP-036 drug_index values must be integers")
    if sorted(int(value) for value in indexes) != list(range(14)):
        raise ValueError("EXP-036 drug_index values must be unique and contiguous 0..13")
    target["drug_id"] = target["drug_id"].astype(str)
    target["canonical_isomeric_smiles"] = target["canonical_isomeric_smiles"].astype(str)
    if target["drug_id"].duplicated().any() or not target["drug_id"].map(
            lambda key: INCHIKEY_RE.fullmatch(key) is not None).all():
        raise ValueError("EXP-036 drug_id must be 14 unique full InChIKeys")
    if target["canonical_isomeric_smiles"].duplicated().any():
        raise ValueError("EXP-036 panel canonical isomeric SMILES must be unique")
    target = target.sort_values("drug_index", kind="stable")
    panel = [{"drug_index": int(row.drug_index), "drug_id": str(row.drug_id),
              "canonical_isomeric_smiles": str(row.canonical_isomeric_smiles)}
             for row in target.itertuples(index=False)]
    for item in panel:
        _, derived_key = _rdkit_smiles_identity(str(item["canonical_isomeric_smiles"]))
        if derived_key != item["drug_id"]:
            raise ValueError("EXP-036 canonical isomeric SMILES does not reproduce its full InChIKey")
    sidecar, sidecar_sha, sidecar_relative = _load_registered_drug_registry(registry)
    source_hashes = _validate_official_source_snapshot()
    hashes = {"registry_sha256": registry_sha, "manifest_sha256": manifest_sha,
              "drugs_parquet_sha256": target_sha, "xpert_drug_registry_sha256": sidecar_sha}
    hashes.update({f"source_{Path(path).stem}_sha256": digest
                   for path, digest in source_hashes.items()})
    return panel, sidecar, hashes, sidecar_relative


def build_crosswalk(panel: list[dict[str, object]], rows: list[dict[str, str]]) -> dict[str, object]:
    keys = [str(row["drug_id"]) for row in panel]
    by_key: dict[str, list[tuple[str, int]]] = {key: [] for key in keys}
    physical_rows_by_key = {key: 0 for key in keys}
    invalid_node_rows = 0
    invalid_pert_id_rows = 0
    for row in rows:
        if row["inchi_key"] in physical_rows_by_key:
            physical_rows_by_key[row["inchi_key"]] += 1
        raw_node = row["drug_node_idx"]
        node_index = int(raw_node) if re.fullmatch(r"[0-9]+", raw_node) else None
        invalid_node_rows += node_index is None
        pert_id = row["pert_id"]
        pert_id_valid = bool(pert_id) and pert_id == pert_id.strip()
        invalid_pert_id_rows += not pert_id_valid
        if row["inchi_key"] in by_key and node_index is not None and pert_id_valid:
            by_key[row["inchi_key"]].append((pert_id, node_index))
    distinct_by_key = {key: set(matches) for key, matches in by_key.items()}
    missing = sum(not matches for matches in distinct_by_key.values())
    one_tuple = sum(len(matches) == 1 for matches in distinct_by_key.values())
    multi_tuple = sum(len(matches) > 1 for matches in distinct_by_key.values())
    repeated_rows = sum(max(0, physical_rows_by_key[key] - len(distinct_by_key[key]))
                        for key in physical_rows_by_key)
    blocked = missing > 0 or multi_tuple > 0 or invalid_node_rows > 0 or invalid_pert_id_rows > 0
    summary = {
        "panel_size": len(panel),
        "panel_keys_with_candidate_rows": sum(count > 0 for count in physical_rows_by_key.values()),
        "panel_keys_with_one_distinct_identity_node_tuple": one_tuple,
        "panel_keys_with_repeated_physical_rows": sum(count > 1 for count in physical_rows_by_key.values()),
        "missing_panel_keys": missing,
        "panel_keys_with_multiple_distinct_identity_node_tuples": multi_tuple,
        "repeated_identical_candidate_rows": repeated_rows,
        "invalid_drug_node_idx_candidate_rows": invalid_node_rows,
        "invalid_pert_id_candidate_rows": invalid_pert_id_rows,
        "candidate_rows_for_frozen_panel": len(rows),
        "status": "BLOCKED_PARTIAL_CANDIDATE" if blocked else "PANEL_PERT_ID_NODE_CROSSWALK_VERIFIED",
    }
    return summary


def build_input_chain(panel: list[dict[str, object]], rows: list[dict[str, str]],
                      sidecar: dict[str, object]) -> dict[str, object]:
    sidecar_rows = sidecar["drugs"]
    pert_ids = [entry.get("pert_id") for entry in sidecar_rows if isinstance(entry, dict)]
    if (len(pert_ids) != len(sidecar_rows)
            or any(not isinstance(value, str) or not value or value != value.strip() for value in pert_ids)
            or len(set(pert_ids)) != len(pert_ids)):
        raise ValueError("XPert sidecar pert_id axis is incomplete or duplicated")
    panel_keys = {str(item["drug_id"]) for item in panel}
    by_key: dict[str, list[dict[str, object]]] = {key: [] for key in panel_keys}
    by_index: dict[int, list[dict[str, object]]] = {}
    for entry in sidecar_rows:
        if not isinstance(entry, dict):
            raise ValueError("XPert sidecar contains a non-object drug record")
        key = entry.get("inchi_key")
        if key in by_key:
            by_key[str(key)].append(entry)
        index = entry.get("pert_idx")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("XPert sidecar contains an invalid feature index")
        for feature_flag in REQUIRED_FEATURE_FLAGS:
            if not isinstance(entry.get(feature_flag), bool):
                raise ValueError("XPert sidecar contains an invalid required-feature flag")
        by_index.setdefault(index, []).append(entry)

    csv_summary = build_crosswalk(panel, rows)
    csv_by_key: dict[str, list[dict[str, str]]] = {key: [] for key in panel_keys}
    for row in rows:
        if row["inchi_key"] in csv_by_key:
            csv_by_key[row["inchi_key"]].append(row)

    target_by_key = {str(item["drug_id"]): item for item in panel}
    registry_found = registry_single = smiles_key_match = smiles_exact_match = 0
    csv_registry_agreement = panel_row_index_unique = flags_complete = full_chain = 0
    registry_multi = 0
    target_smiles_key_valid = 0
    for key, target in target_by_key.items():
        target_smiles, target_derived_key = _rdkit_smiles_identity(
            str(target["canonical_isomeric_smiles"]))
        target_smiles_ok = target_derived_key == key
        target_smiles_key_valid += target_smiles_ok
        matches = by_key[key]
        if matches:
            registry_found += 1
        if len(matches) != 1:
            registry_multi += len(matches) > 1
            continue
        entry = matches[0]
        registry_single += 1
        side_smiles, side_derived_key = _rdkit_smiles_identity(str(entry.get("canonical_smiles", "")))
        side_smiles_ok = side_derived_key == key
        smiles_key_match += side_smiles_ok
        smiles_exact_match += side_smiles_ok and side_smiles == target_smiles

        index = entry["pert_idx"]
        index_unique = len(by_index.get(index, [])) == 1
        panel_row_index_unique += index_unique
        required_flags = all(entry.get(flag) is True for flag in REQUIRED_FEATURE_FLAGS)
        flags_complete += required_flags

        expected_tuple = {(str(entry["pert_id"]), int(index))}
        csv_tuples: set[tuple[str, int]] = set()
        csv_invalid = False
        for candidate in csv_by_key[key]:
            raw_node = candidate["drug_node_idx"]
            pert_id = candidate["pert_id"]
            if (not re.fullmatch(r"[0-9]+", raw_node) or not pert_id
                    or pert_id != pert_id.strip()):
                csv_invalid = True
                continue
            csv_tuples.add((pert_id, int(raw_node)))
        agreement = bool(csv_tuples) and csv_tuples == expected_tuple and not csv_invalid
        csv_registry_agreement += agreement
        full_chain += (target_smiles_ok and side_smiles_ok and side_smiles == target_smiles
                       and index_unique and required_flags and agreement)

    global_unique_indexes = len({entry["pert_idx"] for entry in sidecar_rows})
    summary: dict[str, object] = {
        "panel_size": len(panel),
        "target_smiles_to_full_inchikey_valid": target_smiles_key_valid,
        "registered_sidecar_records": len(sidecar_rows),
        "registered_sidecar_unique_pert_ids": len({entry.get("pert_id") for entry in sidecar_rows}),
        "registered_sidecar_unique_feature_indexes": global_unique_indexes,
        "registered_sidecar_reused_feature_index_count": len(sidecar_rows) - global_unique_indexes,
        "registered_panel_inchikeys_found": registry_found,
        "registered_panel_inchikeys_with_one_record": registry_single,
        "registered_panel_inchikeys_with_multiple_records": registry_multi,
        "panel_smiles_to_sidecar_inchikey_valid": smiles_key_match,
        "panel_smiles_canonical_match_to_sidecar": smiles_exact_match,
        "panel_feature_indexes_unique_across_sidecar": panel_row_index_unique,
        "panel_rows_with_required_unimol_and_hg_features": flags_complete,
        "panel_rows_with_csv_node_and_sidecar_index_agreement": csv_registry_agreement,
        "panel_rows_with_complete_identity_to_model_index_chain": full_chain,
        "panel_rows_missing_from_registered_xpert_sidecar": len(panel) - registry_found,
        "panel_rows_missing_from_xpert_csv": csv_summary["missing_panel_keys"],
        "csv_panel_candidate_summary": csv_summary,
        "feature_index_contract": sidecar["feature_index_key"],
        "model_input_contract": "official config binds UniMol drug features and HG drug embeddings; static MyDataset/model sources consume the registered pert_idx axis",
        "required_feature_flags": list(REQUIRED_FEATURE_FLAGS),
        "status": "IDENTITY_CHAIN_VERIFIED_FOR_PANEL" if full_chain == len(panel)
        else "DATA_BLOCKED_PARTIAL_IDENTITY_CHAIN",
        "limitation": "The registered sidecar and source contract were checked without opening NPY feature values or model weights; this receipt does not verify current feature-array bytes, H5AD batch rows, or model outputs.",
    }
    return summary


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing XPert audit: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    started_monotonic = time.perf_counter()
    receipt: dict[str, object] = {
        "experiment_id": "EXP-037",
        "audit_id": "20260924_xpert_panel_feature_row_crosswalk_v1",
        "started_at_utc": started,
        "status": "PREFLIGHT_BLOCKED",
        "supervision": "existing EXP-037 read-only asset authorization; identity-only audit",
        "authorization_scope": {
            "target_contract": "registered EXP-036 dual_route_exp036_direct_response_contract_v1 v1 QC_PASS, 14 drugs",
            "target_columns": list(TARGET_COLUMNS),
            "xpert_columns": list(XPERT_COLUMNS),
            "sidecar": "registered xpert_drug_registry_v1 JSON identity metadata only",
            "forbidden": ["response/labels", "predictions/results", "checkpoints/pickles", "NPY feature values", "expression/H5AD", "CUDA smoke", "training"],
        },
        "code_sha256": sha256(Path(__file__).resolve()),
    }
    try:
        panel, sidecar, hashes, sidecar_relative = _preflight()
        receipt.update(hashes)
        receipt["xpert_drug_registry_path"] = sidecar_relative
        receipt["preflight"] = "PASS: registered EXP-036 and XPert sidecar checksums match; 14 full InChIKeys reproduce from canonical isomeric SMILES"
        with XPERT_CSV.open("rb") as stream:
            before = _stat_signature(stream)
            csv_sha = _hash_stream(stream)
            after_hash = _stat_signature(stream)
            if before != after_hash:
                raise ValueError("XPert CSV changed while hashing")
            selected_rows, source_rows = _selected_xpert_rows_stream(
                stream, {str(row["drug_id"]) for row in panel})
            parsed_signature = _stat_signature(stream)
            sha_after = _hash_stream(stream)
            after = _stat_signature(stream)
        if before != parsed_signature or before != after or csv_sha != sha_after:
            raise ValueError("XPert CSV changed during hash/parse")
        summary = build_input_chain(panel, selected_rows, sidecar)
        receipt.update({
            "status": summary["status"],
            "xpert_csv_sha256": csv_sha,
            "xpert_input_path": "data/external/xpert_source/processed_data/l1000_sdst_drugs_info_8432.csv",
            "xpert_upstream_release": "not established by this audit; local file path and SHA-256 are recorded",
            "xpert_csv_data_rows_scanned": source_rows,
            "xpert_selected_columns": list(XPERT_COLUMNS),
            "other_csv_columns": "Non-selected data cells are scanned only as raw bytes for CSV structure; they are not decoded, retained, output, or cached. Header field names are decoded to locate the approved columns.",
            "static_source_snapshot_sha256": {
                Path(path).name: digest for path, digest in XPERT_SOURCE_HASHES.items()
            },
            "aggregate": summary,
            "runtime": {
                "device": "CPU only",
                "gpu_used": False,
                "elapsed_seconds": round(time.perf_counter() - started_monotonic, 3),
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        receipt["failure"] = f"{type(exc).__name__}: {exc}"
        receipt["runtime"] = {
            "device": "CPU only",
            "gpu_used": False,
            "elapsed_seconds": round(time.perf_counter() - started_monotonic, 3),
        }
        receipt["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(OUTPUT / "receipt.json", receipt)
        raise
    _atomic_json(OUTPUT / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "aggregate": receipt["aggregate"],
                      "receipt": str((OUTPUT / "receipt.json").relative_to(ROOT))},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
