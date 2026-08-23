#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from drug_screen.data.exp010_inputs import align_expression_contract, audit_gdsc_raw, fetch_pubchem, parse_cmp_tpm, parse_gdsc_fitted, parse_gse287429, parse_gse303509, parse_mcrc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DRUGSCREEN_DATA_ROOT", "data")))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--with-gdsc", action="store_true")
    parser.add_argument("--with-pubchem", action="store_true")
    parser.add_argument("--with-geo", action="store_true")
    parser.add_argument("--with-cmp", action="store_true")
    args = parser.parse_args()
    output = args.output or args.data_root / "processed" / "exp010"
    output.mkdir(parents=True, exist_ok=True)
    expression, responses, mcrc_summary = parse_mcrc(args.data_root / "raw" / "mcrc_pdo_2026" / "v3")
    expression.to_parquet(output / "mcrc_log2cpm.parquet")
    responses.to_parquet(output / "mcrc_monotherapy_dss.parquet", index=False)
    summary_path = output / "input_summary.json"
    summary: dict[str, object] = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.pop("pubchem", None)  # migrate the pre-split combined mapping summary
    summary["mcrc"] = mcrc_summary
    mcrc_names = responses["drug_name"].tolist()
    gdsc_names: list[str] = []
    if args.with_gdsc:
        gdsc, gdsc_summary = parse_gdsc_fitted(args.data_root / "external" / "exp010" / "gdsc2_release8.5" / "GDSC2_fitted_dose_response_27Oct23.xlsx")
        gdsc.to_parquet(output / "gdsc2_fitted_release8.5.parquet", index=False)
        summary["gdsc2"] = gdsc_summary
        raw_path = args.data_root / "external" / "exp010" / "gdsc2_release8.5" / "GDSC2_public_raw_data_27Oct23.csv"
        if raw_path.exists():
            summary["gdsc2"]["raw"] = audit_gdsc_raw(raw_path)
        gdsc_names = gdsc["DRUG_NAME"].dropna().astype(str).tolist()
    if args.with_pubchem:
        pubchem_root = args.data_root / "external" / "exp010" / "pubchem"
        _, summary["pubchem_mcrc"] = fetch_pubchem(mcrc_names, pubchem_root / "mcrc_drug_structures.json")
        if gdsc_names:
            _, summary["pubchem_gdsc2"] = fetch_pubchem(gdsc_names, pubchem_root / "gdsc2_drug_structures.json")
    if args.with_geo:
        geo_root = args.data_root / "external" / "exp010" / "geo"
        gse303_matrix, context, summary["gse303509"] = parse_gse303509(geo_root / "gse303509" / "GSE303509_RAW.tar")
        gse303_matrix.to_parquet(output / "gse303509_tpm.parquet")
        context.to_parquet(output / "gse303509_context.parquet", index=False)
        matrix, context, summary["gse287429"] = parse_gse287429(geo_root / "gse287429" / "GSE287429_normalized_counts_allsample.txt.gz")
        matrix.to_parquet(output / "gse287429_author_normalized.parquet")
        context.to_parquet(output / "gse287429_context.parquet", index=False)
    if args.with_cmp:
        if not args.with_gdsc or not args.with_geo:
            raise ValueError("--with-cmp requires --with-gdsc and --with-geo for the frozen gene contract")
        cmp, summary["cmp"] = parse_cmp_tpm(args.data_root / "external" / "exp010" / "cmp_20220624" / "rnaseq_all_20220624.zip", set(gdsc["SANGER_MODEL_ID"].dropna().astype(str)))
        expression, cmp, gse303_matrix, summary["gene_contract"] = align_expression_contract(expression, cmp, gse303_matrix, args.data_root / "raw" / "mcrc_pdo_2026" / "v3" / "Data S2.xlsx")
        expression.to_parquet(output / "mcrc_gene_contract.parquet")
        cmp.to_parquet(output / "cmp_gdsc2_gene_contract.parquet")
        gse303_matrix.to_parquet(output / "gse303509_gene_contract.parquet")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
