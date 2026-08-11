"""Generate the small, identity-first DATA CONTRACT audit for EXP-002."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256, sha512
import gzip
import json
import math
from pathlib import Path

import h5py
import pandas as pd

from drug_screen.data import exp002


EXPERIMENT_ID = "EXP-002"
CONTRACT_VERSION = "2.0.0"
RAW_REGISTRY_ID = "lincs_gse92742_raw_level3_v1"
INTERIM_REGISTRY_ID = "lincs_gse92742_level3_level4_level5"
MATCH_KEY = ("rna_plate", "cell_id", "pert_time", "pert_time_unit")
METADATA_FILES = (
    "GSE92742_Broad_LINCS_gene_info.txt.gz",
    "GSE92742_Broad_LINCS_pert_info.txt.gz",
    "GSE92742_Broad_LINCS_cell_info.txt.gz",
    "GSE92742_Broad_LINCS_inst_info.txt.gz",
)
LEVEL3_GZIP = "GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx.gz"
LEVEL3_HDF5 = "GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx"


def _file_digest(path: Path, algorithm: str) -> str:
    digest = sha256() if algorithm == "sha256" else sha512()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_digest(values: list[str] | pd.Series) -> str:
    digest = sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _mapping_digest(values: dict[str, str]) -> str:
    return exp002.canonical_digest(sorted(values.items()))


def _decode(values: object) -> list[str]:
    return [value.decode("ascii") if isinstance(value, bytes) else str(value) for value in values]


def _canonical_scalar(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if isinstance(value, float):
        return format(value, ".15g")
    return str(value)


def _matching_key(row: object) -> tuple[str, ...]:
    return tuple(_canonical_scalar(getattr(row, column)) for column in MATCH_KEY)


def _official_checksums(path: Path) -> dict[str, str]:
    with gzip.open(path, "rt", encoding="ascii") as handle:
        return {
            filename: checksum
            for line in handle
            for checksum, filename in [line.strip().split(maxsplit=1)]
        }


def _registry_entry(registry_path: Path, entry_id: str) -> dict[str, object]:
    entries = json.loads(registry_path.read_text(encoding="utf-8"))
    matches = [entry for entry in entries if entry.get("id") == entry_id]
    if len(matches) != 1:
        raise ValueError(f"registry must contain exactly one {entry_id} entry")
    return matches[0]


def _prepare_treatments(
    instances: pd.DataFrame, perturbagens: pd.DataFrame, cells: pd.DataFrame
) -> pd.DataFrame:
    treatments = instances.loc[instances["pert_type"].eq("trt_cp")].copy()
    treatments = treatments.merge(perturbagens, on="pert_id", how="left", validate="many_to_one")
    treatments = treatments.merge(cells, on="cell_id", how="left", validate="many_to_one")
    if treatments["inst_id"].duplicated().any():
        raise ValueError("treatment inst_id values must be unique")
    treatments["drug_id"] = treatments["inchi_key"].map(
        lambda value: None if pd.isna(value) else f"INCHIKEY:{value}"
    )
    fallback_drug = "LINCS_PERT_ID:" + treatments["pert_id"].astype(str)
    treatments["drug_id"] = treatments["drug_id"].fillna(fallback_drug)
    treatments["context_id"] = treatments["base_cell_id"].map(
        lambda value: None if pd.isna(value) else f"BASE_CELL:{value}"
    )
    fallback_context = "CELL_ID:" + treatments["cell_id"].astype(str)
    treatments["context_id"] = treatments["context_id"].fillna(fallback_context)
    family_fields = (
        "drug_id",
        "context_id",
        "pert_dose",
        "pert_dose_unit",
        "pert_time",
        "pert_time_unit",
    )
    treatments["replicate_family_id"] = [
        "TRT_FAMILY:" + exp002.canonical_digest(
            {field: _canonical_scalar(value) for field, value in zip(family_fields, values)}
        )
        for values in treatments.loc[:, family_fields].itertuples(index=False, name=None)
    ]
    return treatments


def _audit_mode(
    matched: pd.DataFrame,
    vehicles: pd.DataFrame,
    entity_column: str,
    namespace: str,
) -> dict[str, object]:
    frame = matched.copy()
    entity_ids = sorted(frame[entity_column].unique())
    entity_splits = {
        entity_id: exp002.deterministic_split(f"{namespace}:{entity_id}")
        for entity_id in entity_ids
    }
    frame["split"] = frame[entity_column].map(entity_splits)

    family_split_counts = frame.groupby("replicate_family_id")["split"].nunique()
    entity_split_counts = frame.groupby(entity_column)["split"].nunique()
    treatment_split_counts = frame.groupby("inst_id")["split"].nunique()

    controls_by_key = {
        tuple(_canonical_scalar(value) for value in key): sorted(group["inst_id"].astype(str))
        for key, group in vehicles.groupby(list(MATCH_KEY), dropna=False, sort=False)
    }
    active_by_key = {
        tuple(_canonical_scalar(value) for value in key): set(group["split"])
        for key, group in frame.groupby(list(MATCH_KEY), dropna=False, sort=False)
    }
    control_splits: dict[str, str] = {}
    controls_for_key_split: dict[tuple[tuple[str, ...], str], tuple[str, ...]] = {}
    insufficient_keys = 0
    for key in sorted(active_by_key):
        controls = controls_by_key.get(key, [])
        try:
            allocation = exp002.deterministic_vehicle_partition(
                key, controls, active_by_key[key], seed=exp002.DEFAULT_SEED
            )
        except ValueError:
            insufficient_keys += 1
            continue
        overlap = set(control_splits).intersection(allocation)
        if overlap:
            raise ValueError(f"raw controls occur under multiple matching keys: {sorted(overlap)[:3]}")
        control_splits.update(allocation)
        for split_name in active_by_key[key]:
            controls_for_key_split[(key, split_name)] = tuple(
                sorted(control for control, owner in allocation.items() if owner == split_name)
            )

    if insufficient_keys:
        raise ValueError(f"{namespace} has {insufficient_keys} matching keys without enough controls")

    family_members = {
        family_id: set(group[entity_column])
        for family_id, group in frame.groupby("replicate_family_id", sort=False)
    }
    exp002.assert_exclusive_assignments(entity_splits, family_members, control_splits)

    pair_digest = sha256()
    for row in frame.sort_values("inst_id").itertuples(index=False):
        key = _matching_key(row)
        record = (
            str(row.inst_id),
            str(row.split),
            controls_for_key_split[(key, row.split)],
        )
        pair_digest.update(json.dumps(record, ensure_ascii=True, separators=(",", ":")).encode())
        pair_digest.update(b"\n")

    plate_cross_split = int((frame.groupby("rna_plate")["split"].nunique() > 1).sum())
    controls_per_target = [len(controls_for_key_split[(_matching_key(row), row.split)]) for row in frame.itertuples(index=False)]
    return {
        "cold_entity": namespace,
        "cold_entity_count": len(entity_splits),
        "cold_entity_cross_split_count": int((entity_split_counts > 1).sum()),
        "cold_entity_split_counts": dict(sorted(Counter(entity_splits.values()).items())),
        "treatment_instance_count": int(len(frame)),
        "treatment_instance_cross_split_count": int((treatment_split_counts > 1).sum()),
        "treatment_split_counts": dict(sorted(Counter(frame["split"]).items())),
        "replicate_family_count": int(frame["replicate_family_id"].nunique()),
        "replicate_family_cross_split_count": int((family_split_counts > 1).sum()),
        "matching_key_count": len(active_by_key),
        "matching_keys_without_enough_disjoint_controls": insufficient_keys,
        "maximum_active_splits_per_matching_key": max(map(len, active_by_key.values())),
        "assigned_raw_vehicle_count": len(control_splits),
        "raw_vehicle_cross_split_count": 0,
        "assigned_raw_vehicle_split_counts": dict(sorted(Counter(control_splits.values()).items())),
        "minimum_controls_per_treatment_target": min(controls_per_target),
        "maximum_controls_per_treatment_target": max(controls_per_target),
        "plates_shared_across_splits_allowed_count": plate_cross_split,
        "non_degenerate": exp002.has_all_splits(entity_splits.values()),
        "manifest_digests": {
            "cold_entity_assignments_sha256": _mapping_digest(entity_splits),
            "raw_vehicle_assignments_sha256": _mapping_digest(control_splits),
            "treatment_control_manifest_sha256": pair_digest.hexdigest(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.root / "raw" / "lincs" / "GSE92742"
    interim = args.root / "interim" / "lincs" / "GSE92742"
    registry_path = args.root / "registry" / "datasets.json"

    official_manifest = raw / "GSE92742_SHA512SUMS.txt.gz"
    expected = _official_checksums(official_manifest)
    actual_metadata = {name: _file_digest(raw / name, "sha512") for name in METADATA_FILES}
    metadata_verified = all(actual_metadata[name] == expected[name] for name in METADATA_FILES)
    if not metadata_verified:
        raise ValueError("GSE92742 metadata checksum mismatch")

    registry_entry = _registry_entry(registry_path, RAW_REGISTRY_ID)
    registered_level3 = registry_entry["files"][LEVEL3_GZIP]["sha512"]
    if registered_level3 != expected[LEVEL3_GZIP]:
        raise ValueError("registered Level-3 checksum differs from official manifest")

    instances = pd.read_csv(raw / METADATA_FILES[3], sep="\t", low_memory=False)
    perturbagens = pd.read_csv(
        raw / METADATA_FILES[1], sep="\t", usecols=["pert_id", "inchi_key"]
    )
    cells = pd.read_csv(
        raw / METADATA_FILES[2], sep="\t", usecols=["cell_id", "base_cell_id"]
    )
    genes = pd.read_csv(raw / METADATA_FILES[0], sep="\t", low_memory=False)
    treatments = _prepare_treatments(instances, perturbagens, cells)

    vehicles = instances.loc[
        instances["pert_type"].eq("ctl_vehicle"), list(MATCH_KEY) + ["inst_id"]
    ].copy()
    vehicle_keys = vehicles.loc[:, MATCH_KEY].drop_duplicates()
    matched = treatments.merge(vehicle_keys, on=list(MATCH_KEY), how="inner", validate="many_to_one")

    with h5py.File(interim / LEVEL3_HDF5, "r") as handle:
        matrix_shape = tuple(int(value) for value in handle["0/DATA/0/matrix"].shape)
        row_ids = _decode(handle["0/META/ROW/id"][:])
        column_ids = _decode(handle["0/META/COL/id"][:])
    gene_flags = dict(zip(genes["pr_gene_id"].astype(str), genes["pr_is_lm"].astype(int)))
    if len(row_ids) != len(set(row_ids)) or set(row_ids) != set(gene_flags):
        raise ValueError("GCTX row gene identity differs from gene_info")
    landmark_order = [gene_id for gene_id in row_ids if gene_flags[gene_id] == 1]
    if len(landmark_order) != 978 or len(set(landmark_order)) != 978:
        raise ValueError("Level-3 landmark gene universe is not exact-978")
    metadata_instances = instances["inst_id"].astype(str)
    if matrix_shape != (len(column_ids), len(row_ids)):
        raise ValueError("Level-3 matrix shape differs from GCTX metadata")
    if set(column_ids) != set(metadata_instances):
        raise ValueError("GCTX column identities differ from inst_info")

    cold_drug = _audit_mode(matched, vehicles, "drug_id", "cold_drug")
    cold_context = _audit_mode(matched, vehicles, "context_id", "cold_context")
    contract_stable = all(
        mode["non_degenerate"]
        and mode["cold_entity_cross_split_count"] == 0
        and mode["treatment_instance_cross_split_count"] == 0
        and mode["replicate_family_cross_split_count"] == 0
        and mode["raw_vehicle_cross_split_count"] == 0
        and mode["matching_keys_without_enough_disjoint_controls"] == 0
        for mode in (cold_drug, cold_context)
    )

    script_path = Path(__file__).resolve()
    module_path = Path(exp002.__file__).resolve()
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "artifact_type": "identity_first_data_contract_audit",
        "contract_version": CONTRACT_VERSION,
        "data_contract_stable": contract_stable,
        "data_status": "DATA_READY" if contract_stable else "DATA_BLOCKED",
        "source_revision": {
            "accession": "GSE92742",
            "registry_id": RAW_REGISTRY_ID,
            "registry_version": registry_entry["version"],
            "official_checksum_manifest": official_manifest.name,
            "official_checksum_manifest_sha256": _file_digest(official_manifest, "sha256"),
        },
        "provenance": {
            "raw_path": str(raw.relative_to(args.root)),
            "level3_compressed_file": LEVEL3_GZIP,
            "level3_compressed_sha512": expected[LEVEL3_GZIP],
            "level3_checksum_verification": registry_entry["checksum"]["verification"],
            "level3_local_size_bytes": (raw / LEVEL3_GZIP).stat().st_size,
            "derived_hdf5_path": str((interim / LEVEL3_HDF5).relative_to(args.root)),
            "derived_storage_registry_id": INTERIM_REGISTRY_ID,
            "metadata_sha512_verified_against_official_manifest": metadata_verified,
            "metadata_sha512": actual_metadata,
            "audit_code_sha256": {
                str(script_path.relative_to(Path.cwd())): _file_digest(script_path, "sha256"),
                str(module_path.relative_to(Path.cwd())): _file_digest(module_path, "sha256"),
            },
        },
        "schema": {
            "sample_id": "inst_info.inst_id; unique raw treatment identity",
            "drug_id": "INCHIKEY:<inchi_key>; fallback LINCS_PERT_ID:<pert_id>",
            "context_id": "BASE_CELL:<base_cell_id>; fallback CELL_ID:<cell_id>",
            "dose": ["pert_dose", "pert_dose_unit"],
            "time": ["pert_time", "pert_time_unit"],
            "matching_key": list(MATCH_KEY),
            "replicate_family_id": "sha256(drug_id,context_id,dose,time)",
            "raw_vehicle_id": "ctl_vehicle inst_info.inst_id",
        },
        "gene_universe": {
            "source": "GCTX ROW/id filtered in source order by gene_info.pr_is_lm=1",
            "gene_count": len(landmark_order),
            "ordered_gene_ids_sha256": _stream_digest(landmark_order),
            "all_gctx_gene_ids_sha256": _stream_digest(row_ids),
            "gctx_instance_order_sha256": _stream_digest(column_ids),
            "matrix_shape_instances_by_genes": list(matrix_shape),
        },
        "identity": {
            "missing_inchi_fallback_treatment_rows": int(treatments["inchi_key"].isna().sum()),
            "missing_inchi_fallback_pert_id_count": int(
                treatments.loc[treatments["inchi_key"].isna(), "pert_id"].nunique()
            ),
            "missing_base_cell_fallback_treatment_rows": int(treatments["base_cell_id"].isna().sum()),
        },
        "matched_control": {
            "selection": "partition all candidate raw vehicle instances by active split per matching key",
            "partition_seed": exp002.DEFAULT_SEED,
            "partition_order": list(exp002.SPLITS),
            "aggregation": "arithmetic mean of all controls assigned to the treatment split and matching key",
            "treatment_instances": int(len(treatments)),
            "matched_treatment_instances": int(len(matched)),
            "excluded_without_vehicle": int(len(treatments) - len(matched)),
        },
        "normalization": {
            "input": "official GSE92742 Level-3 INF values; no additional cross-sample normalization",
            "target": "Delta978 = treatment exact-978 vector minus assigned matched-control arithmetic mean",
            "training_statistics": "none; any later scaling must be fit on train only and recorded separately",
        },
        "split": {
            "algorithm": "sha256(seed|entity|<cold scheme>:<canonical identity>) modulo 10",
            "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
            "cold_drug": cold_drug,
            "cold_context": cold_context,
            "plate_policy": "plate may cross splits; plate identity is forbidden as a model feature",
        },
        "forbidden_data": [
            "GSE92742 inferred non-landmark genes as target or ground truth",
            "GSE92742 Level-4 or Level-5 values",
            "raw treatment inst_id, raw vehicle inst_id, or treatment replicate_family_id across splits",
            "cold entity identity across splits",
            "plate identity as a model feature",
            "validation/test targets or statistics in training-time fitting or normalization",
            "external-test data, MCPIRE_PDO model artifacts, or TriPerturb dependencies",
        ],
        "output_contract": {
            "one_record_per": "matched chemical treatment instance per split scheme",
            "required_fields": [
                "experiment_id", "split_scheme", "split", "treatment_inst_id", "drug_id",
                "context_id", "replicate_family_id", "rna_plate", "dose", "time",
                "control_inst_ids", "ordered_gene_ids_sha256", "delta978",
            ],
            "delta978_dtype_shape": "float32[978]",
            "materialized_in_this_audit": False,
            "manifest_requirement": "full manifests must reproduce the audit manifest digests",
        },
        "legacy_blocker_correction": {
            "invalid_rule": "collapse each matching key to one vehicle and make plate a split identity",
            "plate_is_split_identity": False,
            "regression_test": "test_legacy_single_control_plate_rule_reproduces_false_single_component",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "data_contract_stable": contract_stable,
        "data_status": payload["data_status"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0 if contract_stable else 1


if __name__ == "__main__":
    raise SystemExit(main())
