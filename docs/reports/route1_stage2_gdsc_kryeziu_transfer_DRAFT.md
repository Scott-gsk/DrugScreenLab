# GDSC→Kryeziu transfer DRAFT results

Status: **DRAFT / not frozen**. Used steward admission + gene-space contract.

- Admission: `data/processed/identity/transfer/kryeziu_mcrc_target_admission_DRAFT.json` (EXPLORATORY_ADMITTED, 51 patients / 60 samples)
- Gene contract: `.../gdsc_kryeziu_gene_space_contract_DRAFT.json` (n=19091, sha256=0fd5fdd7…)

| tag | NDCG@5 | Spearman | path |
|---|---|---|---|
| gdsc_kryeziu_xfer_contract | 0.9044 ± 0.0110 | 0.792977 | `artifacts/route1_stage2_gdsc_kryeziu_transfer_DRAFT/` |
| gdsc_kryeziu_xfer_contract_tiny | 0.9059 ± 0.0099 | 0.796136 | `artifacts/route1_stage2_gdsc_kryeziu_transfer_DRAFT/` |

Note: official `scripts/run_transfer_experiment.py` still needs registry alias `kryeziu_mcrc_v3`→current mcrc_v3 assets; this run used contract-pinned DRAFT runner.
