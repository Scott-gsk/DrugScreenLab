# 数据集迁移报告

迁移来源：`D:/Code/Drug_model-MCPIRE_PDO/MCPIRE_PDO/runtime/`。

已迁移资产登记于 `data/registry/datasets.json`，并由
`data/registry/migration_manifest.json` 审计。迁移后已核对文件数量和总字节数；
原始数据集合保留来源 `SHA256SUMS.local` 文件，外部数据集合的 SHA-512 内容摘要
已记录在注册表中。

按政策排除：`runtime/models/`、`runtime/predictions/`、`runtime/reports/`、
`runtime/logs/`、`runtime/splits/triperturb_v2/` 和
`runtime/processed/triperturb_v2/`。这些内容属于模型、实验、缓存或已废弃实现
产物，而非可复用数据资产。

旧运行时注册表 `data/registry/legacy_runtime_registry.jsonl` 仅作为溯源参考保留，
不定义新项目的数据集身份。
