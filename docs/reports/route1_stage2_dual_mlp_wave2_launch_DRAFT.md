# 路线一 · 阶段2 DualMLP 第二波并行启动 DRAFT

## 1) 基线对照表（已合并）

- 折别：`docs/reports/route1_stage2_model_comparison_by_fold_DRAFT.csv`
- 宏平均：`docs/reports/route1_stage2_model_comparison_macro_DRAFT.csv`
- 含：`dual_mlp_base` + CPU 树模型（HistGBM / RF）

宏平均（NDCG@5）：
- random_forest 0.9047
- dual_mlp_base 0.9042
- hist_gradient_boosting 0.9036

## 2) 第二波已在双 A5000 启动（不要闲置）

脚本：`scripts/modeling/train_stage2_dual_mlp_variants_DRAFT.py`

| GPU | PID | tag | 配置 | folds | log |
|---|---|---|---|---|---|
| 0 | 1089 | `dual_mlp_deep` | feat=genes, hidden=512, depth=3, epochs=50 | 0-4 | `logs/dual_mlp_deep_gpu0.log` |
| 1 | 1090 | `dual_mlp_pathway_pca` | feat=pathway_pca (train-fit PCA64 DRAFT proxy), hidden=256, depth=2, epochs=50 | 0-4 | `logs/dual_mlp_pathway_pca_gpu1.log` |

产物目录：`artifacts/route1_stage2_dual_mlp_variants_DRAFT/`

说明：`pathway_pca` 是通路合同冻结前的 **DRAFT 代理特征**（训练集 PCA），不是 Hallmark 分数。

## 3) 树模型 CPU

已完成，见 `docs/reports/route1_stage1_tree_baselines_DRAFT.md`。
