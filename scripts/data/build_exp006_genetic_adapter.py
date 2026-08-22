"""Materialize the EXP-006 genetic paired-h5ad adapter from exact978 cache.

Response-blind: copies treatment/control vectors only after coverage freeze.
Does not read PRISM or use test performance.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from drug_screen.foundation.exp006_transfer import (  # noqa: E402
    DIRECTIONS,
    GENETIC_TYPES,
    direction_from_pert_type,
)
from drug_screen.foundation.xpert_adapter import validate_xpert_contract  # noqa: E402


INST_INFO = ROOT / "data" / "raw" / "lincs" / "GSE92742" / "GSE92742_Broad_LINCS_inst_info.txt.gz"
CACHE = ROOT / "data" / "processed" / "lincs" / "GSE92742" / "exact978_cache_v1" / "exact978_cache.npy"
COVERAGE = ROOT / "artifacts" / "experiments" / "EXP-006" / "CONTEXT_COVERAGE.json"
FEATURES = ROOT / "artifacts" / "experiments" / "EXP-006" / "unipert_genetic_features.npy"
MAPPING = ROOT / "artifacts" / "experiments" / "EXP-006" / "unipert_genetic_mapping.json"
GENE_INFO = ROOT / "data" / "external" / "xpert_source" / "processed_data" / "l1000_gene_info_978.csv"
OUTPUT = ROOT / "artifacts" / "experiments" / "EXP-006" / "genetic_paired_adapter.h5ad"


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    try:
        import anndata as ad
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("anndata is required") from error

    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", type=Path, default=COVERAGE)
    parser.add_argument("--mapping", type=Path, default=MAPPING)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--audit", type=Path, default=OUTPUT.with_suffix(".audit.json"))
    args = parser.parse_args()

    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    gene_to_row = {str(key).upper(): int(value) for key, value in mapping["gene_to_row"].items()}
    contexts = [str(value) for value in coverage["selected_contexts"]]
    cell_idx_map = {str(key): int(value) for key, value in coverage["cell_idx_map"].items()}

    inst = pd.read_csv(INST_INFO, sep="\t", low_memory=False)
    inst["_cache_row"] = np.arange(len(inst), dtype=np.int64)
    treatments = inst.loc[inst["pert_type"].isin(GENETIC_TYPES)].copy()
    treatments["gene_symbol"] = treatments["pert_iname"].astype(str).str.strip().str.upper()
    treatments = treatments.loc[
        treatments["cell_id"].astype(str).isin(contexts)
        & treatments["gene_symbol"].isin(gene_to_row)
    ].copy()
    key_cols = ["rna_plate", "cell_id", "pert_time", "pert_time_unit"]
    treatments["match_key"] = treatments[key_cols].astype(str).agg("||".join, axis=1)
    control_frames = []
    for priority, control_type in enumerate(("ctl_vector", "ctl_untrt")):
        controls = inst.loc[inst["pert_type"].eq(control_type)].copy()
        controls["match_key"] = controls[key_cols].astype(str).agg("||".join, axis=1)
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
    treatments = treatments.sort_values(["cell_id", "gene_symbol", "inst_id"]).reset_index(drop=True)
    if treatments.empty:
        raise ValueError("no matched genetic records remain after coverage freeze")

    cache = np.load(args.cache, mmap_mode="r")
    if cache.ndim != 2 or cache.shape[1] != 978:
        raise ValueError(f"exact978 cache must be [n, 978], got {cache.shape}")
    treat_rows = treatments["_cache_row"].to_numpy(np.int64)
    ctl_rows = treatments["control_cache_row"].to_numpy(np.int64)
    treated = np.asarray(cache[treat_rows], dtype=np.float32)
    controls_x = np.asarray(cache[ctl_rows], dtype=np.float32)
    if not np.isfinite(treated).all() or not np.isfinite(controls_x).all():
        raise ValueError("non-finite exact978 values in genetic adapter")

    gene_info = pd.read_csv(GENE_INFO)
    gene_column = "gene_id" if "gene_id" in gene_info.columns else "gene_name"
    gene_ids = gene_info[gene_column].astype(str).tolist()
    if len(gene_ids) != 978:
        raise ValueError("XPert gene info must have 978 ordered genes")

    direction = treatments["pert_type"].map(direction_from_pert_type)
    unipert_row = treatments["gene_symbol"].map(gene_to_row).astype(np.int64)
    obs = pd.DataFrame(
        {
            "sample_id": treatments["inst_id"].astype(str),
            "pert_id": treatments["pert_id"].astype(str),
            "pert_idx": np.zeros(len(treatments), dtype=np.int64),
            "cell_iname": treatments["cell_id"].astype(str),
            "cell_idx": treatments["cell_id"].astype(str).map(cell_idx_map).astype(np.int64),
            "tissue_idx": np.zeros(len(treatments), dtype=np.int64),
            "pert_dose": np.ones(len(treatments), dtype=np.float32),
            "pert_time": pd.to_numeric(treatments["pert_time"], errors="coerce").fillna(96.0),
            "pert_type": treatments["pert_type"].astype(str),
            "gene_symbol": treatments["gene_symbol"].astype(str),
            "direction": direction.astype(str),
            "direction_idx": direction.map({name: index for index, name in enumerate(DIRECTIONS)}).astype(np.int64),
            "unipert_row": unipert_row,
            "control_type": treatments["control_type"].astype(str),
            "treatment_cache_row": treat_rows,
            "control_cache_row": ctl_rows,
            "split": "train",
        }
    )
    obs.index = obs["sample_id"]
    var = pd.DataFrame(index=pd.Index(gene_ids, name="gene_id"))
    adata = ad.AnnData(X=treated, obs=obs, var=var)
    adata.obsm["X_ctl"] = controls_x
    contract = validate_xpert_contract(adata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output)
    audit = {
        **contract,
        "output_path": str(args.output),
        "output_sha256": _digest(args.output),
        "contexts": contexts,
        "unique_genes": int(obs["gene_symbol"].nunique()),
        "unique_genetic_perturbagens": int(obs["pert_id"].nunique()),
        "records": int(len(obs)),
        "per_context": (
            obs.groupby("cell_iname", sort=True)
            .agg(unique_genes=("gene_symbol", "nunique"), records=("sample_id", "size"))
            .reset_index()
            .to_dict(orient="records")
        ),
        "response_values_used_for_selection": False,
        "dummy_chemical_pert_idx": 0,
        "features_mapping": str(args.mapping),
    }
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"records": audit["records"], "unique_genes": audit["unique_genes"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
