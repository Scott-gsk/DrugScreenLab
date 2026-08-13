# Progress

## 2026-08-13

- Foundation `XPERT_FOUNDATION_READY` 维持不变；没有重跑或修改 EXP-004。
- 已建立 55 个 exact LINCS context、8,418 个官方 XPert perturbagen registry；Broad identity + valid official feature 交集为 1,836 个 perturbagens。
- 已构建 10 exact contexts × 1,836 drugs 的 global Cartesian inference adapter，明确不再从 source adapter 推导 drug map。
- Broad CRC compact 在 identity/context freeze 后读取：35 lines、1,916 response columns、64,823 finite rows。
- 官方 XPert warm checkpoint 完成 global inference；integration baseline 已记录。此结果不是 EXP-005 A/B/C 科学对照。
- A/B/C token overlay GPU smoke 均 PASS；正式 fixed FAST 已完成：六组结果均为 4,096-row held-out，
  prediction 未 collapse，且 contract/seed/budget digest 一致。
- EXP-005 判定为 `NO_MATERIAL_FAST_INCREMENT`：B 仅在 cold-drug 获得 `+0.03207`，cold-cell 为
  `-0.00498`；C 在 cold-cell 为 `-0.02993`、cold-drug 为 `+0.00290`。因此不启动 MEDIUM loop，
  保留官方 XPert baseline。
- A cold-drug 初次 GPU1 运行超过 60 分钟，记录 `BROKEN_TIMEOUT`；同协议在 GPU0 完成，运行态偏差
  不进入科学指标。
- EXP-006 preparation 与 `CONTEXT_ADAPTER_TRACK` 已整合，状态仍为 `PREPARATION_ONLY / DATA_PARTIAL`，
  未创建或执行 EXP-006。
