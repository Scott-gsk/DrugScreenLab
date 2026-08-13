"""Build a response-blind exact-978 genetic manifest for E2."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def assign_split(values: pd.Series, seed: int) -> pd.Series:
    def split(value: object) -> str:
        raw = sha256(f"{seed}:{value}".encode("utf-8")).digest()
        unit = int.from_bytes(raw[:8], "big") / float(2**64)
        return "train" if unit < 0.8 else "validation" if unit < 0.9 else "test"

    return values.map(split)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inst-info", type=Path, required=True)
    parser.add_argument("--chemical-manifest", type=Path, required=True)
    parser.add_argument("--genetic-features", type=Path, required=True)
    parser.add_argument("--genetic-mapping", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--split-seed", type=int, default=20260813)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inst = pd.read_csv(args.inst_info, sep="\t", low_memory=False)
    inst["_cache_row"] = np.arange(len(inst), dtype=np.int64)
    genes = {
        line.strip().upper()
        for line in (output_dir / "selected_genes.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    feature_mapping = json.loads(args.genetic_mapping.read_text(encoding="utf-8"))
    gene_to_row = {str(k): int(v) for k, v in feature_mapping["gene_to_row"].items()}
    pert_to_row = {str(k): int(v) for k, v in feature_mapping["pert_id_to_row"].items()}
    genes &= set(gene_to_row)
    if not genes:
        raise ValueError("selected gene list has no UniPert feature rows")

    treatments = inst.loc[
        inst["pert_type"].eq("trt_sh")
        & inst["pert_time"].eq(96)
        & inst["pert_time_unit"].eq("h")
    ].copy()
    treatments["gene_symbol"] = treatments["pert_iname"].astype(str).str.strip().str.upper()
    treatments = treatments.loc[
        treatments["gene_symbol"].isin(genes) & treatments["pert_id"].astype(str).isin(pert_to_row)
    ].copy()
    if treatments.empty:
        raise ValueError("selected genes have no matched trt_sh records")

    key_columns = ["rna_plate", "cell_id", "pert_time", "pert_time_unit"]
    treatments["match_key"] = treatments[key_columns].astype(str).agg("||".join, axis=1)
    control_frames = []
    for priority, control_type in enumerate(("ctl_vector", "ctl_untrt")):
        controls = inst.loc[inst["pert_type"].eq(control_type)].copy()
        controls["match_key"] = controls[key_columns].astype(str).agg("||".join, axis=1)
        controls["control_priority"] = priority
        control_frames.append(controls)
    controls = pd.concat(control_frames, ignore_index=True)
    controls = controls.sort_values(["match_key", "control_priority", "_cache_row"])
    controls = controls.drop_duplicates("match_key", keep="first")
    treatments = treatments.merge(
        controls[["match_key", "_cache_row", "pert_type"]].rename(
            columns={"_cache_row": "control_cache_row", "pert_type": "control_type"}
        ),
        on="match_key",
        how="inner",
        validate="many_to_one",
    )
    treatments["group_id"] = treatments.apply(
        lambda row: f"{row.gene_symbol}|{row.cell_id}|96h", axis=1
    )
    treatments["split"] = assign_split(treatments["group_id"], args.split_seed)
    treatments["treatment_cache_row"] = treatments["_cache_row"].astype(np.int64)
    treatments = treatments.sort_values(["split", "group_id", "inst_id"])

    genetic_features = np.load(args.genetic_features, mmap_mode="r")
    chemical_payload = json.loads(args.chemical_manifest.read_text(encoding="utf-8"))
    chemical_features_path = (
        repo_root / chemical_payload["chemical_features"]["relative_path"]
    ).resolve()
    chemical_features = np.load(chemical_features_path, mmap_mode="r")
    if chemical_features.dtype != np.float32 or genetic_features.dtype != np.float32:
        raise ValueError("chemical and genetic feature tables must be float32")
    if chemical_features.shape[1] != genetic_features.shape[1]:
        raise ValueError("chemical and genetic feature dimensions must match")
    unified_features = np.concatenate(
        [np.asarray(chemical_features), np.asarray(genetic_features)], axis=0
    ).astype(np.float32, copy=False)
    unified_features_path = output_dir / "unified_perturbagen_features.npy"
    np.save(unified_features_path, unified_features)

    records = []
    for row in treatments.itertuples(index=False):
        records.append(
            {
                "sample_id": str(row.inst_id),
                "treatment_group_id": str(row.group_id),
                "perturbagen_id": str(row.pert_id),
                "modality": "genetic",
                "perturbation_direction": "knockdown",
                "context_id": str(row.cell_id),
                "dose_um": 1.0,
                "time_h": 96.0,
                "split": str(row.split),
                "treatment_cache_row": int(row.treatment_cache_row),
                "control_cache_row": int(row.control_cache_row),
                "perturbagen_feature_row": int(chemical_features.shape[0] + pert_to_row[str(row.pert_id)]),
                "control_type": str(row.control_type),
                "gene_symbol": str(row.gene_symbol),
            }
        )

    cache_manifest = args.cache.with_name("asset_manifest.json")
    cache_payload = json.loads(cache_manifest.read_text(encoding="utf-8"))
    manifest = {
        "format": "e2_genetic_response_manifest_v1",
        "phase": "e2_genetic_supervision_for_chemical_low_data_transfer",
        "condition": {"pert_type": "trt_sh", "time_h": 96.0},
        "control_policy": "same_rna_plate_same_cell_same_time_ctl_vector_preferred_ctl_untrt_fallback",
        "gene_count": 978,
        "split_seed": args.split_seed,
        "split_entity": "gene_symbol|cell_id|96h_group_atomic",
        "response_values_read": False,
        "cache": {
            "relative_path": os.path.relpath(args.cache.resolve(), repo_root).replace("\\", "/"),
            "sha256": str(cache_payload["cache_sha256"]),
            "shape": list(cache_payload["cache_shape"]),
            "asset_id": "lincs_gse92742_exact978_cache_v1",
        },
        "perturbagen_features": {
            "relative_path": os.path.relpath(unified_features_path, repo_root).replace("\\", "/"),
            "sha256": digest(unified_features_path),
            "shape": list(unified_features.shape),
            "representation": "chemical_UniPert_256d_plus_genetic_UniPert_256d_row_aligned",
            "chemical_source_manifest": str(args.chemical_manifest),
            "genetic_source_features": str(args.genetic_features),
        },
        "records": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "format": "e2_genetic_response_manifest_audit_v1",
        "manifest": str(manifest_path),
        "manifest_sha256": digest(manifest_path),
        "records": len(records),
        "genes": len({row["gene_symbol"] for row in records}),
        "contexts": len({row["context_id"] for row in records}),
        "groups": len({row["treatment_group_id"] for row in records}),
        "split_counts": treatments["split"].value_counts().sort_index().to_dict(),
        "control_type_counts": treatments["control_type"].value_counts().sort_index().to_dict(),
        "response_values_read": False,
        "test_role": "genetic_only_diagnostic; chemical_test_is_frozen_upstream_manifest",
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
