# 路线二：药物条件扰动头（drug-emb + ctrl→Δ）DRAFT

- 时间：2026-09-18T17:21:06+08:00
- GPU：physical 1 / NVIDIA RTX A5000
- 产物：`artifacts/experiments/route2_lincs_drugcond_gpu_DRAFT/`
- 脚本：`scripts/run_route2_lincs_drugcond_gpu_DRAFT.py`

## 设计意图

- 纠正「仅 ctrl→Δ」被误读为药特异：同设置下对比 **ctrl_only** vs **drug_cond**。
- `compound_holdout`：测试药全 UNK，检验冷启动药。
- `pair_random`：药可见时，drug_cond 是否高于 ctrl_only。

## 协议 `compound_holdout`

- train/test pairs：11990/3010
- compounds train/test：569/142
- test UNK drug rows：3010 (100.00%)

| name | n | mean cosine | mean pearson |
|---|---:|---:|---:|
| null_zero | 3010 | 0.0000 | 0.0000 |
| global_mean_delta | 3010 | 0.0758 | 0.0767 |
| ctrl_only_linear | 3010 | 0.6702 | 0.6705 |
| drug_cond_mlp | 3010 | 0.6386 | 0.6397 |

## 协议 `pair_random`

- train/test pairs：12000/3000
- compounds train/test：711/699
- test UNK drug rows：0 (0.00%)

| name | n | mean cosine | mean pearson |
|---|---:|---:|---:|
| null_zero | 3000 | 0.0000 | 0.0000 |
| global_mean_delta | 3000 | 0.0757 | 0.0764 |
| ctrl_only_linear | 3000 | 0.6723 | 0.6727 |
| drug_cond_mlp | 3000 | 0.6552 | 0.6565 |

耗时 72.98 s。
