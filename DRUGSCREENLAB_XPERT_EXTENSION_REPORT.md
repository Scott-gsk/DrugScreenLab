# DrugScreenLab XPert Extension Report

## 当前状态

`EXP-005 FAST_COMPLETE · NO_MATERIAL_FAST_INCREMENT`

本报告记录 XPert Foundation Ready 之后的 DrugScreenLab extension。`EXP-004` 遗传→化学 FAST
结果未被修改、覆盖或重新解释。当前 historical/simple `Context + Chemical -> Delta978` 只保留为历史
比较对象，不再作为主研究 backbone。

## Architecture

```text
Official XPert baseline (Model A)
  control-context -> official XPert cell embedding
  HG + UniMol -> official XPert drug embedding -> official attention/heads

Model B
  Model A drug sequence + separate projected KPGT token

Model C
  Model A drug sequence + separate projected KPGT token
                           + separate projected UniPert token
```

- 官方 HG + UniMol 路径保留。
- KPGT `2304d` 和 UniPert `256d` 各自独立投影至 XPert hidden size；不在 XPert 外直接拼接。
- token 在官方 `drug_emb` 输出后追加；官方 XPert cross-/self-attention、loss 与 prediction head 未重写。
- GPU smoke 已验证 A/B/C 的完整 forward/backward 数据契约。

实现：`src/drug_screen/foundation/xpert_extension.py`；fixed runner：
`scripts/foundation/run_xpert_extension_fast.py`。

## Broad PRISM coverage

| Stage | Coverage |
| --- | ---: |
| 官方 XPert global perturbagen registry | 8,418 `pert_id` |
| feature-complete global inference eligible | 8,418 |
| response-blind identity-matched Broad candidates | 1,836 |
| frozen PRISM response columns | 1,916 |
| CRC PRISM lines | 35 |
| exact LINCS context overlap for XPert global adapter | 10 |
| global Cartesian inference records | 18,360 |
| finite Broad compact response rows | 64,823 |

Candidate construction is response-blind: `Broad PRISM candidate → canonical identity bridge → official XPert
global registry → valid UniMol/HG/KPGT feature`. The evaluation then uses exact base `ccle_name` to `cell_iname`
matching with no reference-context fallback.

Evidence: `mvp/foundation/xpert/CONTEXT_REGISTRY.json`,
`mvp/foundation/xpert/DRUG_REGISTRY.json`, `mvp/foundation/xpert/BROAD_GLOBAL_ADAPTER_AUDIT.json`,
and `mvp/foundation/xpert/BROAD_PRISM_CRC_V1_AUDIT.json`.

## XPert foundation Broad integration baseline

Official warm XPert checkpoint inference completed on the 18,360-record global Cartesian adapter. This is a
foundation integration baseline, **not** the EXP-005 A/B/C comparison.

| Metric | Value |
| --- | ---: |
| exact-context Broad lines evaluated | 10 |
| macro PRISM Spearman | -0.02736 |
| macro NDCG@10 | 0.72320 |
| macro top-10 overlap | 0.080 |
| Observed LINCS Oracle eligible lines | 2 |
| Oracle macro PRISM Spearman | 0.15389 |

The Oracle has only two lines with enough observed LINCS candidate overlap; it is an operational ceiling/coverage
diagnostic, not a trained model and not efficacy ground truth.

## EXP-005 fixed FAST comparison

Models A/B/C were evaluated for both `split_cold_cell_1` and `split_cold_drug_1` with the same seed, fixed
sorted official sample prefix, optimizer, official loss weights, batch size, epoch budget and no test/PRISM-driven
checkpoint selection. Every completed result has 4,096 train and 4,096 held-out rows, finite predictions and
non-collapsed prediction standard deviation. One initial A cold-drug run exceeded a 60-minute runtime bound on
GPU1; it was recorded as `BROKEN_TIMEOUT` and rerun with the identical protocol on the identical-model GPU0,
where it completed in about six minutes. This runtime incident does not enter the scientific metrics.

| Model | Cold-cell Δ978 | Cold-drug Δ978 | Disease reversal | Broad PRISM |
| --- | --- | --- | --- | --- |
| A: XPert | row Spearman `0.08601` | `0.05294` | Broad Spearman `0.01361` / `-0.01554` | NDCG@10 `0.57089` / `0.50838` |
| B: XPert + KPGT | `0.08102` (gain `-0.00498`) | `0.08500` (gain `+0.03207`) | `0.00764` / `-0.02389` | `0.60155` / `0.47709` |
| C: XPert + KPGT + UniPert | `0.05608` (gain `-0.02993`) | `0.05584` (gain `+0.00290`) | `-0.00918` / `-0.03085` | `0.57410` / `0.44198` |

Values in each Broad column are cold-cell / cold-drug. The pre-registered `+0.02` requirement must hold on
both splits. B passes only cold-drug; C passes neither, so the decision is **`NO_MATERIAL_FAST_INCREMENT`**.
No MEDIUM architecture escalation is triggered; the official XPert baseline remains the foundation candidate.
The compact comparison is `mvp/foundation/xpert/EXP005_FAST_COMPARISON.json`.

## Decision / next steps

- A material FAST increment requires a `+0.02` per-row held-out Delta978 Spearman increase over A on both
  cold-cell and cold-drug, finite/non-collapsed prediction, and no PRISM response use in fitting or selection.
- Only that pre-registered rule triggers a larger MEDIUM loop; otherwise the result is
  `NO_MATERIAL_FAST_INCREMENT`.
- In parallel, `EXP-006` genetic→chemical data preparation and `CONTEXT_ADAPTER_TRACK` only define normalized
  interfaces for CCLE/DepMap/organoid/PDO. They do not change XPert foundation or initiate training.
