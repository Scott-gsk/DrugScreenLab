"""Build a new, immutable MCRC transfer bundle from the original public tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from drug_screen.data.mcrc_source import rebuild_mcrc, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/transfer/mcrc_v3"))
    parser.add_argument("--report", type=Path, default=Path("data/artifacts/EXP-014/mcrc_rebuild_v3.json"))
    args = parser.parse_args()
    root = args.root
    raw = root / "raw/mcrc_pdo_2026/v3"
    legacy = root / "external/exp010/pubchem/mcrc_drug_structures.json"
    fixed = root / "raw/transfer_public_assets/pubchem/mcrc_isomeric_20260912.json"
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {args.report}")
    rebuilt = rebuild_mcrc(
        pd.read_excel(raw / "Data S1.xlsx"),
        pd.read_excel(raw / "Data S2.xlsx"),
        pd.read_excel(raw / "Data S5.xlsx"),
        legacy,
        fixed,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    rebuilt.samples.to_parquet(args.output / "samples.parquet", index=False)
    rebuilt.drugs.to_parquet(args.output / "drugs.parquet", index=False)
    rebuilt.pairs.to_parquet(args.output / "pairs.parquet", index=False)
    np.savez_compressed(args.output / "indexed.npz", expression=rebuilt.expression, sample_id=np.array(rebuilt.sample_ids), ordered_genes=np.array(rebuilt.gene_ids))
    files = {path.name: sha256(path) for path in (args.output / "samples.parquet", args.output / "drugs.parquet", args.output / "pairs.parquet", args.output / "indexed.npz")}
    manifest = {
        "cohort_id": "mcrc_pdo_v3", "source": "Mendeley hr94h42xdc.3", "response_name": "DSS", "higher_is_sensitive": True,
        "patient_identity": "S1 PDO mappings verified where present; remaining group IDs are source sample-prefix derived and pending G0 review", "scientific_admission": "PENDING_ASSAY_AND_CLINICAL_REVIEW",
        "source_sha256": {name: sha256(raw / name) for name in ("Data S1.xlsx", "Data S2.xlsx", "Data S5.xlsx")},
        "fixed_pubchem_sha256": sha256(fixed), "legacy_name_cid_mapping_sha256": sha256(legacy),
        "n_samples": len(rebuilt.samples), "n_genes": len(rebuilt.gene_ids), "n_drugs": len(rebuilt.drugs), "n_pairs": len(rebuilt.pairs),
        "exclusions": rebuilt.exclusions, "output_sha256": files,
        "source_audit_status": "REBUILD_COMPLETE_G0_REVIEW_REQUIRED",
        "metadata_interpretation_correction": {
            "supersedes": "mcrc_v2 patient metadata interpretation only; mcrc_v2 files remain unchanged",
            "reason": "Data S1 is a mutation table, not a complete sample register; absent S1 rows cannot exclude S2 expression samples.",
            "v3_policy": "All S2 expression samples are retained. S1 PDO mappings verify matching patient prefixes; remaining patient group IDs are prefix-derived and pending G0 identity verification.",
        },
        "g0_gaps": [
            "S1 is an incomplete mutation table. Prefix-derived group IDs are not independently verified and must not enter automated training before G0 identity review.",
            "Author-provided DSS is retained; dose-response refitting and repeat-noise analysis remain unverified.",
            "Oxaliplatin is a legitimate monotherapy but is excluded pending a supported coordination-complex representation.",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({**manifest, "rebuild_status": "BUILT_NOT_SCIENTIFICALLY_ADMITTED"}, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
