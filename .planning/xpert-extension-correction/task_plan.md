# XPert Extension Correction Track

## Goal

在不改变官方 XPert backbone 的前提下，纠正 source-adapter 的四药限制，建立 global
context/drug registry，完成 response-blind Broad screening 接入，并按 EXP-005 比较 A/B/C token overlay。

## Steps

- [x] Context Registry、Drug Registry 与 global Cartesian adapter。
- [x] response-blind Broad identity/context freeze 后构建 CRC response compact。
- [x] 官方 XPert global inference 和 baseline Broad evaluation。
- [x] KPGT/UniPert 独立 token overlay GPU smoke。
- [x] EXP-005 fixed FAST：cold-cell / cold-drug A/B/C。
- [x] EXP-006 genetic→chemical 与 CONTEXT_ADAPTER_TRACK preparation。
- [x] 报告、审计、GitHub checkpoint（commit `2dbf038` 已推送至 `origin/master`）。
