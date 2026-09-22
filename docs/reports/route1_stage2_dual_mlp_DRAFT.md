# 路线一 · 阶段2 DualMLP（本机双 A5000，DRAFT）

- 状态：**已跑完**（进程已退出；checkpoint 与结果 JSON 齐全，无需重拉）
- 环境：WSL `drugscreening-gpu`；物理 GPU0/GPU1 = NVIDIA RTX A5000
- 脚本：`scripts/modeling/train_stage2_dual_mlp_DRAFT.py`
- 划分：核实患者外 5 折 DRAFT（`data/splits/kryeziu_patient_outer5fold_DRAFT.json`）
- 数据：`data/processed/transfer/mcrc_v3`（top-2000 方差基因 + 48 药 embedding）
- 并行约定：GPU0 → folds 0,1,2；GPU1 → folds 3,4
- 产物目录：`artifacts/route1_stage2_dual_mlp_DRAFT/`

## 进程核查（本次）

- 检查时双 A5000 显存接近空闲，无存活 DualMLP / stage2 python 训练进程。
- 已存在完整 ckpt：`dual_mlp_gpu0_fold{0,1,2}.pt`、`dual_mlp_gpu1_fold{3,4}.pt`。
- 已存在结果：`dual_mlp_gpu0_results.json`、`dual_mlp_gpu1_results.json`（时间戳约 2026-09-18 16:32）。
- 结论：**不重拉**；本报告汇总既有结果。

## 树模型 CPU 任务确认

- `scripts/modeling/train_stage1_trees_DRAFT.py` 已完成（无存活进程）。
- 报告：`docs/reports/route1_stage1_tree_baselines_DRAFT.md`
- 指标：`docs/reports/route1_stage1_tree_baselines_DRAFT.metrics.json`
- 摘要：HistGBM NDCG@5 0.9036±0.0103 / Spearman 0.7837±0.0095；RF NDCG@5 0.9047±0.0085 / Spearman 0.7085±0.0272。

## DualMLP 折别测试集指标

| fold | GPU | n_patients | NDCG@5 | Spearman | MAE | n_train | n_test |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 11 | 0.9002 | 0.7880 | 4.2179 | 2498 | 646 |
| 1 | 0 | 10 | 0.9022 | 0.7969 | 4.1491 | 2480 | 664 |
| 2 | 0 | 10 | 0.9153 | 0.7734 | 4.6437 | 2560 | 584 |
| 3 | 1 | 10 | 0.9157 | 0.8015 | 4.3491 | 2558 | 586 |
| 4 | 1 | 10 | 0.8874 | 0.7756 | 4.2445 | 2480 | 664 |

## 宏平均（折等权）

- NDCG@5：**0.9042 ± 0.0106**
- Spearman：**0.7871 ± 0.0112**
- MAE：4.3209 ± 0.1737

## 训练设定（据结果 JSON）

- 模型：DualMLP（表达 top-2000 方差基因 + 药物 embedding）
- 标签：药敏响应（越高越敏感）
- 训练仅用 fold train 患者；test 患者不参与训练
- GPU0 墙钟约 31.42807102203369 s；GPU1 墙钟约 21.226938724517822 s

## 验收点

1. 五折 ckpt 与两份 results JSON 存在。
2. 本报告路径：`docs/reports/route1_stage2_dual_mlp_DRAFT.md`
3. 树模型 CPU 基线报告已在盘，无需再开训。
4. DRAFT：指标口径需与评价侧 `stage1_metrics_aggregation_DRAFT` / Task-B 患者等权定义再对齐核对。

## 正式对照表（已合并）

- 折别：`docs/reports/route1_stage2_model_comparison_by_fold_DRAFT.csv`
- 宏平均：`docs/reports/route1_stage2_model_comparison_macro_DRAFT.csv`

### 宏平均摘要

| model                  | variant                 | fold   |   n_folds |   ndcg@5_mean |   ndcg@5_std |   spearman_B_mean |   spearman_B_std |   mae_mean |
|:-----------------------|:------------------------|:-------|----------:|--------------:|-------------:|------------------:|-----------------:|-----------:|
| random_forest          | tree_cpu_top2000_onehot | macro  |         5 |      0.904703 |   0.00846367 |          0.708521 |       0.0271985  |  nan       |
| dual_mlp_base          | expr_top2000_hidden256  | macro  |         5 |      0.904177 |   0.0105712  |          0.787055 |       0.0111825  |    4.32088 |
| hist_gradient_boosting | tree_cpu_top2000_onehot | macro  |         5 |      0.903613 |   0.0102586  |          0.78373  |       0.00946371 |  nan       |

