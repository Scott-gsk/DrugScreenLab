"""Build a global UniPert chemical feature array indexed by XPert drug_node_idx."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from drug_screen.modeling.phase2_fast import build_unipert_chemical_features, file_sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug-info", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    info = pd.read_csv(args.drug_info, dtype=str, keep_default_na=False)
    required = {"pert_id", "drug_node_idx", "canonical_smiles"}
    missing = sorted(required.difference(info.columns))
    if missing:
        raise ValueError(f"drug info missing columns: {missing}")
    info["drug_node_idx"] = pd.to_numeric(info["drug_node_idx"], errors="raise").astype(int)
    features, mapping, feature_audit = build_unipert_chemical_features(info, model_path=args.model)
    index_by_id = (
        info[["pert_id", "drug_node_idx"]]
        .drop_duplicates("pert_id")
        .set_index("pert_id")["drug_node_idx"]
        .to_dict()
    )
    max_idx = int(info["drug_node_idx"].max())
    indexed = np.zeros((max_idx + 1, features.shape[1]), dtype=np.float32)
    available = np.zeros((max_idx + 1,), dtype=np.uint8)
    for pert_id, feature_row in mapping.items():
        idx = int(index_by_id[str(pert_id)])
        indexed[idx] = features[int(feature_row)]
        available[idx] = 1
    if not np.isfinite(indexed[available.astype(bool)]).all():
        raise ValueError("global UniPert feature array contains non-finite values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, indexed)
    mask_path = args.output.with_name(args.output.stem + "_available.npy")
    np.save(mask_path, available)
    audit = {
        "format": "xpert_global_unipert_chemical_features_v1",
        "representation": feature_audit["representation"],
        "source_model": str(args.model),
        "source_model_sha256": file_sha256(args.model),
        "source_drug_info": str(args.drug_info),
        "source_drug_info_sha256": file_sha256(args.drug_info),
        "feature_path": str(args.output),
        "feature_shape": list(indexed.shape),
        "availability_mask": str(mask_path),
        "available_indices": int(available.sum()),
        "official_pert_ids": int(info["pert_id"].nunique()),
        "labels_used": False,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
