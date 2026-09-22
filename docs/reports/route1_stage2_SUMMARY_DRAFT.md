# Route1 Stage2 SUMMARY (DRAFT)

Local WSL. Contracts DRAFT. Transfer under admission+gene-space contracts in flight.

| model | n | NDCG@5 | Spearman | note |
|---|---:|---|---|---|
| dual_mlp_genes_h64_wd02 | 5 | 0.9098 ± 0.0083 | 0.7928 | genes |
| dual_mlp_pathway_pca_reg | 5 | 0.9096 ± 0.0117 | 0.7987 | pathway_pca |
| dual_mlp_tiny_wd05 | 5 | 0.9088 ± 0.0141 | 0.7853 | genes |
| dual_mlp_hallmark_ssgsea | 5 | 0.9082 ± 0.0105 | 0.7803 | pathway_hallmark_ssgsea |
| dual_mlp_pathway_pca_h64_wd02 | 5 | 0.9073 ± 0.0128 | 0.7904 | pathway_pca |
| dual_mlp_pathway_pca | 5 | 0.9073 ± 0.0104 | 0.7878 | pathway_pca |
| dual_mlp_small_reg_strong | 5 | 0.9068 ± 0.0116 | 0.7919 | genes |
| dual_mlp_drop05_es | 5 | 0.9067 ± 0.0060 | 0.7904 |  |
| dual_mlp_gene_intersect_xfer_tiny | 5 | 0.9063 ± 0.0058 | 0.7853 | gene_intersect_contract_space |
| dual_mlp_hallmark_h128_wd001 | 5 | 0.9062 ± 0.0109 | 0.7852 | pathway_hallmark_ssgsea |
| dual_mlp_hallmark_h64_wd02 | 5 | 0.9056 ± 0.0090 | 0.7796 | pathway_hallmark_ssgsea |
| dual_mlp_hallmark_ssgsea_reg | 5 | 0.9053 ± 0.0112 | 0.7864 | pathway_hallmark_ssgsea |
| dual_mlp_genes_h32_wd05 | 5 | 0.9050 ± 0.0067 | 0.7900 | genes |
| dual_mlp_gene_intersect_xfer | 5 | 0.9050 ± 0.0059 | 0.7849 | gene_intersect_contract_space |
| dual_mlp_small_reg | 5 | 0.9049 ± 0.0089 | 0.7906 | genes |
| random_forest | 5 | 0.9047 ± 0.0085 | 0.7085 |  |
| dual_mlp_deep | 5 | 0.9046 ± 0.0083 | 0.7878 | genes |
| dual_mlp_pathway_pca_tiny | 5 | 0.9036 ± 0.0116 | 0.7933 | pathway_pca |
| hist_gradient_boosting | 5 | 0.9036 ± 0.0103 | 0.7837 |  |
| dual_mlp_genes_h128_wd01 | 5 | 0.9004 ± 0.0098 | 0.7939 | genes |
| pathway_hallmark_ridge | 5 | 0.9002 ± 0.0135 | 0.7696 |  |
| gdsc_pdo_transfer_smoke | 5 | 0.8992 ± 0.0103 | 0.7824 | pre-contract smoke |

In flight: `gdsc_kryeziu_xfer_contract` / `_tiny` on GPU0/1 with gene_order_sha256=0fd5fdd7… and EXPLORATORY_ADMITTED 51/60.
