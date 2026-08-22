"""Response-blind EXP-006 dual-coverage audit and contract freeze.

Reads only identity/coverage metadata.  Never opens PRISM response values
or Δ978 labels for context selection.
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
    GENETIC_TYPES,
    SEED,
    assign_compound_level_splits,
    sample_unique_compounds,
    select_dual_coverage_contexts,
    split_digest,
    write_json,
)


INST_INFO = ROOT / "data" / "raw" / "lincs" / "GSE92742" / "GSE92742_Broad_LINCS_inst_info.txt.gz"
UNIPERT_REF = ROOT / "data" / "external" / "unipert_source" / "data" / "ref_targets.csv"
XPERT_H5AD = ROOT / "data" / "external" / "xpert_source" / "processed_data" / "l1000_sdst_78453.h5ad"
CACHE_MANIFEST = ROOT / "data" / "processed" / "lincs" / "GSE92742" / "exact978_cache_v1" / "asset_manifest.json"
OUTPUT_COVERAGE = ROOT / "artifacts" / "experiments" / "EXP-006" / "CONTEXT_COVERAGE.json"
OUTPUT_CONTRACT = ROOT / "artifacts" / "experiments" / "EXP-006" / "LARGE_SCALE_TRANSFER_CONTRACT.json"


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _match_controls(inst: pd.DataFrame) -> pd.DataFrame:
    genetic = inst.loc[inst["pert_type"].isin(GENETIC_TYPES)].copy()
    genetic["gene_symbol"] = genetic["pert_iname"].astype(str).str.strip().str.upper()
    genetic = genetic.loc[~genetic["gene_symbol"].isin({"", "NAN", "-666"})]
    key_cols = ["rna_plate", "cell_id", "pert_time", "pert_time_unit"]
    genetic["match_key"] = genetic[key_cols].astype(str).agg("||".join, axis=1)
    control_frames = []
    for priority, control_type in enumerate(("ctl_vector", "ctl_untrt")):
        controls = inst.loc[inst["pert_type"].eq(control_type)].copy()
        if controls.empty:
            continue
        controls["match_key"] = controls[key_cols].astype(str).agg("||".join, axis=1)
        controls["control_priority"] = priority
        control_frames.append(controls[["match_key", "control_priority", "_cache_row", "pert_type"]])
    if not control_frames:
        genetic["has_matched_control"] = False
        genetic["control_cache_row"] = np.int64(-1)
        genetic["control_type"] = ""
        return genetic
    controls = pd.concat(control_frames, ignore_index=True)
    controls = controls.sort_values(["match_key", "control_priority", "_cache_row"])
    controls = controls.drop_duplicates("match_key", keep="first")
    merged = genetic.merge(
        controls.rename(columns={"_cache_row": "control_cache_row", "pert_type": "control_type"}),
        on="match_key",
        how="left",
        validate="many_to_one",
    )
    merged["has_matched_control"] = merged["control_cache_row"].notna()
    return merged


def build_coverage(*, min_unique_genes: int, min_unique_compounds: int, max_contexts: int) -> dict:
    inst = pd.read_csv(INST_INFO, sep="\t", low_memory=False)
    inst["_cache_row"] = np.arange(len(inst), dtype=np.int64)
    cache_manifest = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
    if int(cache_manifest["cache_shape"][0]) != len(inst):
        raise ValueError("exact978 cache rows do not align with inst_info")

    local_genes = {
        str(value).strip().upper()
        for value in pd.read_csv(UNIPERT_REF, low_memory=False)["Approved symbol"]
    }
    genetic = _match_controls(inst)
    genetic["unipert_mappable"] = genetic["gene_symbol"].isin(local_genes)

    import anndata as ad

    xpert = ad.read_h5ad(XPERT_H5AD, backed="r")
    chemical = xpert.obs[["cell_iname", "pert_id"]].copy()
    chemical["cell_iname"] = chemical["cell_iname"].astype(str)
    chemical["pert_id"] = chemical["pert_id"].astype(str)
    official_cell_idx = (
        xpert.obs[["cell_iname", "cell_idx"]]
        .drop_duplicates("cell_iname")
        .assign(cell_iname=lambda frame: frame["cell_iname"].astype(str))
    )
    official_contexts = set(official_cell_idx["cell_iname"])
    xpert.file.close()
    genetic = genetic.loc[genetic["cell_id"].astype(str).isin(official_contexts)].copy()
    chemical = chemical.loc[chemical["cell_iname"].isin(official_contexts)].copy()

    decision = select_dual_coverage_contexts(
        genetic,
        chemical,
        min_unique_genes=min_unique_genes,
        min_unique_compounds=min_unique_compounds,
        max_contexts=max_contexts,
        target_unique_genes=2000,
    )
    decision["official_xpert_context_count"] = int(len(official_contexts))

    selected_contexts = decision["selected_contexts"]
    chemical_selected = chemical.loc[chemical["cell_iname"].isin(selected_contexts)]
    compounds = sorted(chemical_selected["pert_id"].unique())
    split = assign_compound_level_splits(compounds, seed=SEED)
    fractions = {
        "1.0": split["train"],
        "0.2": sample_unique_compounds(split["train"], fraction=0.2, seed=SEED),
        "0.1": sample_unique_compounds(split["train"], fraction=0.1, seed=SEED),
    }
    genetic_selected = genetic.loc[
        genetic["cell_id"].astype(str).isin(selected_contexts)
        & genetic["has_matched_control"]
        & genetic["unipert_mappable"]
    ]
    decision["official_xpert_overlap_only"] = True
    decision["response_values_read"] = False
    decision["prism_values_read"] = False
    decision["source"] = {
        "inst_info": str(INST_INFO),
        "inst_info_sha256": _digest(INST_INFO),
        "unipert_ref": str(UNIPERT_REF),
        "unipert_ref_sha256": _digest(UNIPERT_REF),
        "xpert_h5ad": str(XPERT_H5AD),
        "exact978_cache_manifest": cache_manifest,
    }
    decision["selected_pool"] = {
        "unique_genes": int(genetic_selected["gene_symbol"].nunique()),
        "genetic_records": int(len(genetic_selected)),
        "unique_compounds": int(len(compounds)),
        "chemical_records": int(len(chemical_selected)),
        "compound_split": {
            role: {"count": len(values), **split_digest({role: values})[role]}
            for role, values in split.items()
        },
        "train_fraction_unique_compounds": {key: int(len(value)) for key, value in fractions.items()},
    }
    cell_idx_map = {
        str(row.cell_iname): int(row.cell_idx)
        for row in official_cell_idx.itertuples(index=False)
    }
    decision["cell_idx_map"] = {context: cell_idx_map[context] for context in selected_contexts}
    return {
        "coverage": decision,
        "split": split,
        "fractions": fractions,
        "genetic_selected_count": int(len(genetic_selected)),
        "cache_manifest": cache_manifest,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-unique-genes", type=int, default=200)
    parser.add_argument("--min-unique-compounds", type=int, default=200)
    parser.add_argument("--max-contexts", type=int, default=5)
    parser.add_argument("--coverage-output", type=Path, default=OUTPUT_COVERAGE)
    parser.add_argument("--contract-output", type=Path, default=OUTPUT_CONTRACT)
    args = parser.parse_args()

    built = build_coverage(
        min_unique_genes=args.min_unique_genes,
        min_unique_compounds=args.min_unique_compounds,
        max_contexts=args.max_contexts,
    )
    coverage = built["coverage"]
    write_json(args.coverage_output, coverage)

    contract = {
        "format": "large_scale_xpert_genetic_chemical_transfer_contract_v1",
        "exp_id": "EXP-006",
        "status": "COVERAGE_FROZEN",
        "approval": "user_program_execution_mandate_2026-08-13",
        "legacy_fast": {
            "status": "LEGACY_POSITIVE_PRIOR",
            "max_genes": 256,
            "max_chemical_train_groups": 4000,
            "model": "small_mlp_not_xpert",
            "usable_as_formal_claim": False,
        },
        "comparison": {
            "A": "XPert chemical-only; official UniMol/HG → shared XPert response backbone",
            "B": "Genetic-pretrained XPert → chemical fine-tune; UniPert genetic → minimal adapter → same backbone",
            "unique_variable": "genetic_pretraining",
            "forbidden": [
                "new fusion architecture",
                "tiny FAST dataset",
                "test-performance context selection",
                "PRISM-supervised selection",
            ],
        },
        "selected_contexts": coverage["selected_contexts"],
        "per_context": coverage["per_context"],
        "scale": {
            "genetic_unique_genes": coverage["selected_pool"]["unique_genes"],
            "genetic_records": coverage["selected_pool"]["genetic_records"],
            "chemical_unique_compounds": coverage["selected_pool"]["unique_compounds"],
            "chemical_records": coverage["selected_pool"]["chemical_records"],
            "max_genes_cap": None,
            "max_chemical_train_groups_cap": None,
            "downsample": coverage["downsample"],
        },
        "split": {
            "entity": "unique_compound_pert_id",
            "seed": SEED,
            "roles": coverage["selected_pool"]["compound_split"],
            "train_fraction_unique_compounds": coverage["selected_pool"]["train_fraction_unique_compounds"],
        },
        "training": {
            "backbone": "official XPert Foundation champion",
            "checkpoint": "data/external/xpert_source/saved_model/l1000_sdst_warm_split.pth",
            "seed": SEED,
            "hyperparameter_search": "none; single seed; bounded budget shared by A and B",
            "optimizer": "Adam",
            "batch_size": 32,
            "genetic_pretrain_epochs": 3,
            "chemical_finetune_epochs": 3,
            "initialization_note": (
                "A and B inherit the same champion weights. Unique variable is "
                "whether B first updates the shared backbone on genetic supervision."
            ),
        },
        "metrics": {
            "perturbation": ["delta978_pearson", "delta978_spearman", "mse", "direction_consistency"],
            "downstream_required": ["topk_lift", "ndcg_excess", "hitrate_recall_at_k", "per_line_consistency"],
            "primary_regimes": [0.2, 0.1],
        },
        "response_values_read": False,
        "prism_values_read": False,
        "split_compound_lists": {
            "relative_path": "artifacts/experiments/EXP-006/compound_splits.json",
        },
    }
    write_json(args.contract_output, contract)
    write_json(
        args.contract_output.with_name("compound_splits.json"),
        {
            "format": "exp006_compound_level_split_v1",
            "seed": SEED,
            "split": built["split"],
            "fractions": built["fractions"],
            "digest": split_digest(built["split"]),
        },
    )
    print(
        json.dumps(
            {
                "selected_contexts": coverage["selected_contexts"],
                "per_context": [
                    {
                        "context_id": row["context_id"],
                        "unique_genes": row["unique_genes"],
                        "unique_compounds": row["unique_compounds"],
                    }
                    for row in coverage["per_context"]
                ],
                "pool": coverage["selected_pool"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
