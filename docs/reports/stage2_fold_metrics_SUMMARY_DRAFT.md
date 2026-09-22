# Route-1 Stage2 折结果评价汇总（DRAFT）

- 口径：`stage1_metrics_aggregation_DRAFT.md`；对照 **Stage1-B0 NDCG@5=0.9089**
- 本轮主更新：`dual_mlp_genes_h64_wd02`；顺带并入已齐的 tiny_wd05 / drop05_es / hallmark_ssgsea_reg / hallmark_h128_wd001
- 时间：2026-09-18 17:35 Asia/Shanghai · DSL-评价审计
- CI：fold 单位；**未过 CI 不得宣称增益**

## 宏平均对照（NDCG@5 降序）

| model | NDCG@5 | Spearman_B | Δ vs B0 | fold-CI 判断 |
|---|---:|---:|---:|---|
| dual_mlp_genes_h64_wd02 | 0.9098±0.0083 | 0.7928±0.0195 | +0.0010 | 证据不足 (Δ=+0.0010, cross=True) |
| dual_mlp_pathway_pca_reg | 0.9096±0.0117 | 0.7987±0.0138 | +0.0007 | 证据不足 (Δ=+0.0007, cross=True) |
| B0_drug_mean | 0.9089±0.0051 | 0.8078 | +0.0000 | — |
| dual_mlp_tiny_wd05 | 0.9088±0.0141 | 0.7853±0.0139 | -0.0001 | 证据不足 (Δ=-0.0001, cross=True) |
| dual_mlp_hallmark_ssgsea | 0.9082±0.0105 | 0.7803±0.0041 | -0.0007 | 证据不足 (Δ=-0.0007, cross=True) |
| dual_mlp_pathway_pca | 0.9073±0.0104 | 0.7878±0.0111 | -0.0016 | 证据不足 (Δ=-0.0016, cross=True) |
| dual_mlp_small_reg_strong | 0.9068±0.0116 | 0.7919±0.0127 | -0.0021 | 证据不足 (Δ=-0.0021, cross=True) |
| dual_mlp_drop05_es | 0.9067±0.0060 | 0.7904±0.0156 | -0.0021 | 弱于对照 (Δ=-0.0021, cross=False) |
| dual_mlp_hallmark_h128_wd001 | 0.9062±0.0109 | 0.7852±0.0086 | -0.0026 | 证据不足 (Δ=-0.0026, cross=True) |
| B1_per_drug_elasticnet | 0.9058 | 0.8097 | -0.0031 | n/a |
| dual_mlp_hallmark_ssgsea_reg | 0.9053±0.0112 | 0.7864±0.0100 | -0.0036 | 证据不足 (Δ=-0.0036, cross=True) |
| dual_mlp_small_reg | 0.9049±0.0089 | 0.7906±0.0147 | -0.0040 | 证据不足 (Δ=-0.0040, cross=True) |
| random_forest | 0.9047±0.0085 | 0.7085±0.0272 | -0.0042 | 弱于对照 (Δ=-0.0042, cross=False) |
| dual_mlp | 0.9042±0.0106 | 0.7871±0.0112 | -0.0047 | 证据不足 (Δ=-0.0047, cross=True) |
| hist_gradient_boosting | 0.9036±0.0103 | 0.7837±0.0095 | -0.0053 | 证据不足 (Δ=-0.0053, cross=True) |
| B1_per_drug_ridge | 0.9009 | 0.8135 | -0.0080 | n/a |
| pathway_hallmark_ridge | 0.9002±0.0135 | 0.7696±0.0070 | -0.0087 | 弱于对照 (Δ=-0.0087, cross=False) |
| dual_mlp_deeper | 0.8807±0.0140 | 0.7569±0.0122 | -0.0282 | 弱于对照 (Δ=-0.0282, cross=False) |

## 本轮新并入

- **dual_mlp_genes_h64_wd02**：NDCG@5=0.9098±0.0083，Spearman=0.7928±0.0195
- vs Stage1-B0：Δ=+0.0010，95% fold-CI [-0.0021, 0.0038]，**证据不足**
- 亦并：`dual_mlp_tiny_wd05`、`dual_mlp_drop05_es`、`dual_mlp_hallmark_ssgsea_reg`、`dual_mlp_hallmark_h128_wd001`
- 源：`artifacts/route1_stage2_dual_mlp_variants_DRAFT/*_results.json`

## 评价判断

1. `genes_h64_wd02` 点估计 **0.9098** 略高于 Stage1-B0 **0.9089**（Δ=+0.0010），但 fold-CI **跨零** → **证据不足**，不得宣称增益。
2. 其余新并模型同样按 fold-CI：跨零=证据不足，显著为负=弱于对照。
3. 正式患者单位 CI 仍待预测分落盘。

## 产物

| 文件 | 说明 |
|---|---|
| `docs/reports/stage2_fold_metrics_SUMMARY_DRAFT.md` | 本报告 |
| `docs/reports/stage2_fold_metrics_SUMMARY_DRAFT.csv` | 折表 |
| `docs/reports/stage2_fold_ci_vs_B0_DRAFT.csv` | fold-CI |

