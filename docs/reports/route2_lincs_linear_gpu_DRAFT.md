# 路线二 LINCS 线性扰动基线（GPU）DRAFT

- 时间：2026-09-18T16:42:27+08:00
- GPU：physical 0 / NVIDIA RTX A5000
- 产物：`artifacts/experiments/route2_lincs_linear_gpu_DRAFT/`
- 与路线一隔离：不读 Kryeziu splits，不写 stage1 表

## 设置
- pairs usable 12000; train/test 9576/2424
- split `compound_holdout`; seed 7
- model Linear(978→978) on z-scored control → Δ; epochs 40

## 结果（test，pred vs true Δ）

| name | n | mean cosine | mean pearson |
|---|---:|---:|---:|
| null_zero | 2424 | 0.0000 | 0.0000 |
| global_mean_delta | 2424 | 0.0867 | 0.0874 |
| linear_ctrl_to_delta_gpu | 2424 | 0.6621 | 0.6628 |

耗时 20.05 s。
