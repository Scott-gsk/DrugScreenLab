"""Build the non-registered Zhu v2 transfer bundle from author supplements."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from drug_screen.data.zhu_source import rebuild_zhu, sha256

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("data/processed/transfer/zhu_v2"))
    p.add_argument("--report", type=Path, default=Path("data/artifacts/EXP-015/zhu_source_audit.json"))
    p.add_argument("--catalogue-expansion", action="store_true")
    args = p.parse_args(); raw = args.root / "external/pdo_public_candidates/zhu2024_hepatobiliary_64"
    if args.output.exists() or args.report.exists(): raise FileExistsError("zhu_v2 output/report must be new; refusing overwrite")
    rebuilt = rebuild_zhu(raw / "mmc4.xlsx", raw / "mmc5.xlsx", catalogue_expansion=args.catalogue_expansion)
    args.output.mkdir(parents=True)
    rebuilt.samples.to_parquet(args.output / "samples.parquet", index=False)
    rebuilt.drugs.to_parquet(args.output / "drugs.parquet", index=False)
    rebuilt.pairs.to_parquet(args.output / "pairs.parquet", index=False)
    np.savez_compressed(args.output / "indexed.npz", expression=rebuilt.expression, sample_id=np.array(rebuilt.sample_ids), ordered_genes=np.array(rebuilt.gene_ids))
    files = {x.name: sha256(x) for x in args.output.iterdir()}
    report = {**rebuilt.audit, "cohort_id": "zhu_hepatobiliary_v3" if args.catalogue_expansion else "zhu_hepatobiliary_v2", "scientific_admission": "EXPLORATORY_PDO_DEVELOPMENT_ONLY", "source_sha256": {x.name: sha256(x) for x in (raw / "mmc4.xlsx", raw / "mmc5.xlsx")}, "output_sha256": files, "structure_exclusions": rebuilt.exclusions}
    args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
if __name__ == "__main__": main()
