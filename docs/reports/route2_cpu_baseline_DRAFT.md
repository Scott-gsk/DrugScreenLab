# 路线二轻量 CPU 基线 DRAFT

- 时间：2026-09-18T16:39:13+08:00
- 设备：CPU（`CUDA_VISIBLE_DEVICES=''`，不占 A5000）
- 脚本：`scripts/run_route2_cpu_baseline_DRAFT.py`
- 指标 JSON：`artifacts/experiments/route2_cpu_baseline_DRAFT/metrics.json`

## 设置

- 抽样匹配对照对：可用 3000（请求 3000，scan≤250000）
- 划分：`compound_holdout`，test_frac=0.2，seed=42
- train/test pairs：2414 / 586
- train/test compounds：282 / 70
- 特征：exact978 Level3 差值（treat−plate-matched control）

## 基线结果（test 上 pred vs true Δ 的向量相似）

| baseline | n | mean cosine | mean pearson | median cosine | median pearson |
|---|---:|---:|---:|---:|---:|
| null_zero | 586 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| global_mean_delta | 586 | 0.1685 | 0.1691 | 0.1762 | 0.1769 |
| drug_mean_delta | 586 | 0.1685 | 0.1691 | 0.1762 | 0.1769 |

## 验收点

- [x] 未使用 GPU / 未抢 A5000
- [x] 只用实测 LINCS L3 landmark 差值 + 板内对照
- [x] 无可靠对照对不进入评价集
- [x] 化学扰动（genetic 未混入）
- [ ] ChemPert 实测矩阵（待合同/物化）
- [ ] Ridge / XPert 等预测模型（明确后续，本轮不做）

耗时约 8.0 s。

## 次协议：pair_random（可见药均值）

| baseline | n | mean cosine | mean pearson |
|---|---:|---:|---:|
| null_zero | 600 | 0.0000 | 0.0000 |
| global_mean_delta | 600 | 0.1794 | 0.1803 |
| drug_mean_delta | 600 | 0.0293 | 0.0298 |

- test 对中药已在 train 出现：600/600
- 主表 compound_holdout 下 drug_mean≡global_mean 是冷启动药的预期行为。
