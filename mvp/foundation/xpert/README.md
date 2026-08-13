# XPert Foundation Track

状态：`XPERT_FOUNDATION_READY`

这是 Foundation / engineering integration track，不创建新的 `EXP-ID`。官方 XPert 模型、loss、数据契约、split 和 evaluation path 保持不变；本地修改仅为运行时兼容修复。

核心记录：

- [ASSET_MANIFEST.json](ASSET_MANIFEST.json)：官方代码、Figshare 资产、checksum 和兼容性修复。
- [XPERT_FOUNDATION_RESULT.json](XPERT_FOUNDATION_RESULT.json)：bounded demo、warm sanity、cold-cell/cold-drug 独立测试与最终 gate。
- [ADAPTER_DOWNSTREAM_RESULT.json](ADAPTER_DOWNSTREAM_RESULT.json)：`exact978 -> disease reversal -> Broad PRISM` integration diagnostic。

官方 source 和大型数据/模型资产位于被忽略的 `data/external/xpert_source/`；不要提交 raw data、h5ad、prediction profile 或 checkpoint。

最终 gate 使用：

- `split_cold_cell_1`：25,578 test rows，Delta978 Pearson `0.282`。
- `split_cold_drug_1`：15,829 test rows，Delta978 Pearson `0.356`。

两者 prediction profile 均 finite 且非 collapse，因此不启用 MultiDCP fallback。`EXP-004` 保持不变；`EXP-005` 保留给第一个 Novel Extension。
