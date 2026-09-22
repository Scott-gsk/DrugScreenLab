# 数据登记目录

本目录只保存小型、机器可读的数据身份和来源记录。数据集的人类可读总览统一位于 [`../DATA_CATALOG.md`](../DATA_CATALOG.md)。

## 文件类别

| 文件 | 作用 |
|---|---|
| `datasets.json` | 主要数据资产登记 |
| `datasets_acquisition_*.json` | 一次公开资源获取批次的清单登记 |
| `datasets_transfer.json` | 跨来源数据资产登记 |
| `transfer_cohorts.json` | 跨来源样本组及其配对状态 |
| `exp*_source.json`、`exp*_target*.json` | 特定研究阶段冻结的数据输入版本 |
| `limited_use_*.json`、`pdo_resource_triage.json` | 尚未形成完整配对或用途受限的资源事实登记 |
| `legacy_runtime_registry.jsonl`、`migration_manifest.json` | 历史数据迁移记录；新代码不以其旧名称作为数据身份 |

## 自动验证边界

`datasets*.json` 会由 `drug_screen.data.registry` 自动读取。每个顶层条目必须包含：

- `id`
- `version`
- `asset_type`
- `path.relative`

因此，普通说明性 JSON、候选资源笔记或审计结果不得使用 `datasets` 文件名前缀。审计结果应放入 `data/artifacts/`，人类可读说明应写入 `DATA_CATALOG.md`。

登记文件只描述数据身份，不存放大型矩阵、模型、预测或日志。路径统一相对于 `data/`，原始资产登记为不可变，处理后资产使用新版本而不是覆盖旧版本。
