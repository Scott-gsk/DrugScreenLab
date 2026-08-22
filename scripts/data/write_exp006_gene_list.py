"""Write the response-blind EXP-006 UniPert gene list from frozen coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INST = ROOT / "data" / "raw" / "lincs" / "GSE92742" / "GSE92742_Broad_LINCS_inst_info.txt.gz"
REF = ROOT / "data" / "external" / "unipert_source" / "data" / "ref_targets.csv"
COVERAGE = ROOT / "artifacts" / "experiments" / "EXP-006" / "CONTEXT_COVERAGE.json"
OUTPUT = ROOT / "artifacts" / "experiments" / "EXP-006" / "selected_genes.txt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", type=Path, default=COVERAGE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    contexts = set(coverage["selected_contexts"])
    inst = pd.read_csv(INST, sep="\t", low_memory=False)
    local_genes = set(pd.read_csv(REF, low_memory=False)["Approved symbol"].astype(str).str.strip().str.upper())
    genetic = inst.loc[inst["pert_type"].isin(["trt_sh", "trt_sh.cgs", "trt_sh.css", "trt_oe", "trt_oe.mut"])].copy()
    genetic["gene_symbol"] = genetic["pert_iname"].astype(str).str.strip().str.upper()
    genetic = genetic.loc[genetic["cell_id"].astype(str).isin(contexts) & genetic["gene_symbol"].isin(local_genes)]
    genes = sorted(set(genetic["gene_symbol"]))
    if not genes:
        raise ValueError("no UniPert-mappable genes remain in selected contexts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(genes) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "unique_genes": len(genes)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
