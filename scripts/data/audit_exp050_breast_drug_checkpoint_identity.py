"""Bind EXP-050 doxorubicin/bortezomib identity to frozen model feature rows.

This is an identity and source-development row-coverage audit only. It does not
read expression responses or outer-test values and does not score either model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
V7 = ROOT / "artifacts/experiments/EXP-050/20260930_keyed_response_v7"
PREPARED = ROOT / "artifacts/experiments/EXP-037/20260928_lincs_scale_16_121/prepared"
XP_RUN = ROOT / "artifacts/experiments/EXP-037/20260928_lincs_scale_16_121/xpert_121"
RUN_ID = "20260930_breast_drug_checkpoint_identity_v1"
DRUGS = {
    "doxorubicin": {"pert_id": "BRD-K92093830", "inchi_key": "AOJJSUZBOXZQNB-TZSSRYMLSA-N"},
    "bortezomib": {"pert_id": "BRD-K88510285", "inchi_key": "GXJABQQUPOEUTA-RDJZCZTQSA-N"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def load_snapshot_adapter():
    snapshot = ROOT / "artifacts/experiments/EXP-037/20260928_lincs_scale_16_121/wrapper_xpert121/code_snapshot"
    module_path = snapshot / "exp037_xpert_fold.py"
    expected = "5e6e47f7b0ab82269300c688ce69bfbfb9435c6ad75319b0c487e3d296de39b0"
    if sha256(module_path) != expected:
        raise ValueError("frozen EXP-037 XPert adapter source hash mismatch")
    spec = importlib.util.spec_from_file_location("exp050_identity_xpert_adapter", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot import the frozen XPert identity adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def audit(output: Path) -> dict[str, Any]:
    if sys.platform != "linux" or os.environ.get("CONDA_DEFAULT_ENV") != "drugscreening-gpu":
        raise RuntimeError("EXP-050 requires WSL2 Conda environment drugscreening-gpu")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing identity audit: {output}")

    v7_manifest_path = V7 / "run_manifest.json"
    v7_manifest = json.loads(v7_manifest_path.read_text(encoding="utf-8"))
    crosswalk_path = V7 / "EXP050_drug_identity_crosswalk.tsv"
    crosswalk = pd.read_csv(crosswalk_path, sep="\t", dtype=str)
    for name, identity in DRUGS.items():
        rows = crosswalk[crosswalk.drug == name]
        if (len(rows) != 1 or rows.iloc[0].LINCS_pert_id != identity["pert_id"]
                or rows.iloc[0].PubChem_full_InChIKey != identity["inchi_key"]
                or rows.iloc[0].LINCS_full_InChIKey != identity["inchi_key"]
                or rows.iloc[0].exact_full_key_match.lower() != "true"):
            raise ValueError(f"EXP-050 drug identity crosswalk failed for {name}")
    crosswalk_manifest = v7_manifest.get("output_hashes", {}).get(crosswalk_path.name, {})
    if crosswalk_manifest and sha256(crosswalk_path) != crosswalk_manifest.get("sha256"):
        raise ValueError("drug identity crosswalk differs from the v7 run manifest")

    prepared_manifest_path = PREPARED / "manifest.json"
    prepared_manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    if (prepared_manifest.get("status") != "PREPARED_TRAIN_VALIDATION_ONLY"
            or prepared_manifest.get("n_train121") != 32094
            or prepared_manifest.get("n_validation") != 4954):
        raise ValueError("EXP-037 prepared train/validation tier does not match its manifest")
    for filename in ("conditions.parquet", "train121.npy", "validation.npy"):
        if sha256(PREPARED / filename) != prepared_manifest["files_sha256"][filename]:
            raise ValueError(f"prepared source-side row input changed: {filename}")
    keys = pd.read_parquet(PREPARED / "conditions.parquet")
    train_positions = np.load(PREPARED / "train121.npy", allow_pickle=False)
    validation_positions = np.load(PREPARED / "validation.npy", allow_pickle=False)
    train_keys = keys.iloc[train_positions]
    validation_keys = keys.iloc[validation_positions]
    if (not np.allclose(pd.to_numeric(keys.pert_dose).to_numpy(np.float64), 10.0)
            or not np.allclose(pd.to_numeric(keys.pert_time).to_numpy(np.float64), 24.0)):
        raise ValueError("prepared source conditions are outside the frozen 10 uM / 24 h scope")

    input_dir = ROOT / "data/processed/lincs/GSE92742/continuous_inputs_v2"
    target_drugs = pd.read_parquet(input_dir / "drugs.parquet")
    target_matches = target_drugs[target_drugs.pert_id.astype(str).isin(
        [item["pert_id"] for item in DRUGS.values()])]
    if len(target_matches) != 2 or target_matches.pert_id.astype(str).duplicated().any():
        raise ValueError("the project LINCS drug feature index does not uniquely contain both target drugs")
    if "inchi_key" not in target_matches:
        raise ValueError("the LINCS drug feature table lacks the required full InChIKey field")

    source_root = ROOT / "data/external/xpert_source"
    source_drugs_path = source_root / "processed_data/l1000_sdst_drugs_info_8432.csv"
    source_drugs = pd.read_csv(source_drugs_path, dtype={"pert_id": str, "inchi_key": str})
    xpert_unimol = np.load(source_root / "processed_data/all_drugs_unimol_arr.npy",
                           mmap_mode="r", allow_pickle=False)
    xpert_hg = np.load(source_root / "HG_data/saved_embedding/HG_drug_embeddings.npy",
                       mmap_mode="r", allow_pickle=False)
    input_manifest_path = input_dir / "manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if (input_manifest.get("status") != "PASS"
            or input_manifest.get("drug_feature_shape") != [20336, 2048]
            or input_manifest.get("files_sha256", {}).get("drugs.parquet") != sha256(input_dir / "drugs.parquet")
            or input_manifest.get("files_sha256", {}).get("drug_morgan2048.npy") != sha256(input_dir / "drug_morgan2048.npy")):
        raise ValueError("registered TranSiGen drug identity/feature inputs do not match their manifest")

    adapter = load_snapshot_adapter()
    paths = adapter.ProjectPaths.from_root(ROOT)
    model_config = adapter.load_model_config(paths)
    drug_index = adapter.validate_drug_feature_index(
        target_drugs, source_drugs, xpert_unimol.shape, xpert_hg.shape,
        hidden_size=int(model_config["model"]["ATTN"]["hidden_size"]),
        expected_shared_count=6836, exclude_conflicting_nodes=True,
    )

    morgan = np.load(input_dir / "drug_morgan2048.npy", mmap_mode="r", allow_pickle=False)
    source_ecfp = np.load(PREPARED / "ecfp4.npy", mmap_mode="r", allow_pickle=False)
    if (morgan.ndim != 2 or morgan.shape != (20336, 2048)
            or source_ecfp.shape != (len(keys), 2048)
            or xpert_unimol.ndim != 3 or xpert_unimol.shape[1:] != (122, 514)
            or xpert_hg.ndim != 2):
        raise ValueError("drug feature matrix axes do not match their frozen model input contracts")

    results = []
    for name, identity in DRUGS.items():
        pert_id, inchi_key = identity["pert_id"], identity["inchi_key"]
        project_rows = target_drugs[target_drugs.pert_id.astype(str) == pert_id]
        source_rows = source_drugs[source_drugs.pert_id.astype(str) == pert_id]
        if (len(project_rows) != 1 or source_rows.empty
                or source_rows.inchi_key.nunique(dropna=True) != 1
                or source_rows.drug_node_idx.astype(int).nunique() != 1):
            raise ValueError(f"drug-to-feature row is not unique for {pert_id}")
        project_row = project_rows.iloc[0]
        source_row = source_rows.iloc[0]
        if (str(project_row.inchi_key) != inchi_key or str(source_row.inchi_key) != inchi_key
                or pert_id not in drug_index.exact_shared_pert_ids):
            raise ValueError(f"full chemical identity does not bind to both model inputs for {pert_id}")

        target_index = int(drug_index.target_index_by_pert_id[pert_id])
        node_index = int(drug_index.node_index_by_pert_id[pert_id])
        project_index = int(project_row.drug_index)
        if target_index != project_index or not (0 <= project_index < len(morgan)):
            raise ValueError(f"TranSiGen drug feature index mismatch for {pert_id}")
        feature = np.asarray(morgan[project_index])
        xpert_mol = np.asarray(xpert_unimol[node_index])
        xpert_graph = np.asarray(xpert_hg[node_index])
        if not all(np.isfinite(values).all() for values in (feature, xpert_mol, xpert_graph)):
            raise ValueError(f"nonfinite registered drug feature row for {pert_id}")

        train_mask = train_keys.pert_id.astype(str).to_numpy() == pert_id
        validation_mask = validation_keys.pert_id.astype(str).to_numpy() == pert_id
        prepared_mask = keys.pert_id.astype(str).to_numpy() == pert_id
        train_global_positions = train_positions[train_mask]
        validation_global_positions = validation_positions[validation_mask]
        observed_feature_rows = np.asarray(source_ecfp[train_global_positions], dtype=np.float32)
        ecfp_rows_match = bool(len(observed_feature_rows) > 0 and np.array_equal(
            observed_feature_rows, np.broadcast_to(feature.astype(np.float32), observed_feature_rows.shape)
        ))
        node_group = source_drugs[source_drugs.drug_node_idx.astype(int) == node_index]
        node_inchi_count = int(node_group.inchi_key.nunique(dropna=True))
        result = {
            "drug": name,
            "pert_id": pert_id,
            "inchi_key": inchi_key,
            "project_drug_index": project_index,
            "xpert_drug_node_index": node_index,
            "xpert_source_identity_rows": int(len(source_rows)),
            "project_morgan2048_row_sha256": vector_sha(feature),
            "prepared_exp037_ecfp_rows_match_project_feature": ecfp_rows_match,
            "prepared_exp037_condition_count": int(prepared_mask.sum()),
            "inner_train_condition_count": int(train_mask.sum()),
            "inner_train_cell_count": int(train_keys.iloc[np.flatnonzero(train_mask)].cell_id.nunique()),
            "inner_validation_condition_count": int(validation_mask.sum()),
            "inner_validation_cell_count": int(validation_keys.iloc[np.flatnonzero(validation_mask)].cell_id.nunique()),
            "xpert_unique_node_key_count": node_inchi_count,
            "xpert_node_key_matches": node_inchi_count == 1 and str(node_group.inchi_key.iloc[0]) == inchi_key,
            "xpert_unimol_row_sha256": vector_sha(xpert_mol),
            "xpert_graph_row_sha256": vector_sha(xpert_graph),
        "xpert_unimol_shape": [int(dimension) for dimension in xpert_mol.shape],
            "xpert_graph_width": int(xpert_graph.shape[0]),
        }
        if not result["prepared_exp037_ecfp_rows_match_project_feature"] or not result["xpert_node_key_matches"]:
            raise ValueError(f"checkpoint feature row identity does not pass for {pert_id}")
        results.append(result)

    output.mkdir(parents=True)
    input_paths = {
        "v7_manifest": V7 / "run_manifest.json",
        "v7_drug_crosswalk": crosswalk_path,
        "prepared_manifest": prepared_manifest_path,
        "prepared_conditions": PREPARED / "conditions.parquet",
        "prepared_train_rows": PREPARED / "train121.npy",
        "prepared_validation_rows": PREPARED / "validation.npy",
        "xpert_source_drug_table": source_drugs_path,
        "xpert_adapter_snapshot": ROOT / "artifacts/experiments/EXP-037/20260928_lincs_scale_16_121/wrapper_xpert121/code_snapshot/exp037_xpert_fold.py",
        "continuous_input_manifest": input_manifest_path,
        "project_drug_table": input_dir / "drugs.parquet",
        "project_morgan_feature_matrix": input_dir / "drug_morgan2048.npy",
        "xpert_graph_feature_matrix": source_root / "HG_data/saved_embedding/HG_drug_embeddings.npy",
    }
    xpert_unimol_path = source_root / "processed_data/all_drugs_unimol_arr.npy"
    replay_receipt_path = ROOT / "artifacts/experiments/EXP-050/20260930_xpert_validation_replay_v4/summary.json"
    replay_receipt = json.loads(replay_receipt_path.read_text(encoding="utf-8"))
    replay_unimol_hash = replay_receipt.get("input_hashes", {}).get("official_unimol_features", {})
    if (replay_receipt.get("status") != "PASS"
            or replay_unimol_hash.get("sha256") != "d5b6aab93e9c5ee8b9b62171379ca6a4d9e9667aaf26691d3cf8e5483a617cc9"
            or xpert_unimol_path.stat().st_size != replay_unimol_hash.get("bytes")):
        raise ValueError("XPert UniMol whole-file hash/size differs from the frozen replay receipt")
    result = {
        "experiment_id": "EXP-050",
        "run_id": RUN_ID,
        "status": "PASS",
        "scope": "DOX/BTZ full InChIKey to project Morgan and XPert feature row identity; source inner-train/validation condition coverage only",
        "target_treated_responses_read": False,
        "outer_test_responses_read": False,
        "expression_response_values_read": False,
        "drug_results": results,
        "input_hashes": {name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                         for name, path in input_paths.items()},
        "xpert_unimol_matrix_provenance": {
            "bytes": int(xpert_unimol_path.stat().st_size),
            "sha256_verified_by_prior_replay": replay_unimol_hash["sha256"],
            "prior_replay_receipt": str(replay_receipt_path.relative_to(ROOT)),
            "prior_replay_receipt_sha256": sha256(replay_receipt_path),
            "row_sha256_values_recomputed_for_this_audit": True,
        },
        "interpretation_boundary": "identity, feature-row binding, and source condition coverage only; no model score, PDO response prediction, or drug-sensitivity claim",
    }
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(
        f"artifacts/experiments/EXP-050/{RUN_ID}"))
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    print(json.dumps(audit(output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
