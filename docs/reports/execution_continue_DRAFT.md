# 执行不停报（DRAFT）2026-09-18 17:20

## 本机状态
- GPU0/GPU1 A5000：**正在跑** `dual_mlp_small_reg_strong`（h64 d1 wd=0.01）与 `dual_mlp_pathway_pca_reg`（h128 d1 wd=0.01）
- 正式结果只认本机 WSL `drugscreening-gpu`

## 已完成（路线一）
- B0 / 树 / DualMLP base / deeper / Hallmark Ridge / small_reg / pathway_pca
- live macros：`docs/reports/route1_stage2_live_macros_DRAFT.md`

## 进行中
- 数据管家：划分冻结清单 + 三套输入合同缺口
- 评价：CI / judge_model_gain 骨架
- 扰动：药物条件头（CPU/空闲 GPU）

## 原则
实施方案未完成不停止；双卡不闲置；不盲目加宽加深。
