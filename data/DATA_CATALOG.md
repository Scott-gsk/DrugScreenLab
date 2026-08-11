# 数据集目录

本文件说明当前 DrugScreenLab 本地工作区中已经登记的数据资产，以及克隆 GitHub
仓库后可以获得和不能获得的内容。它基于 `data/registry/datasets.json`、
`data/registry/legacy_runtime_registry.jsonl` 与
`data/registry/migration_manifest.json` 编写；这些注册表才是数据身份、版本、来源、
用途边界和校验信息的机器可读事实来源。

## GitHub 中包含什么

GitHub 仓库包含数据说明和注册表：

- `data/README.md` 与本文件；
- `data/registry/datasets.json`；
- `data/registry/legacy_runtime_registry.jsonl`；
- `data/registry/migration_manifest.json`。

它不包含 `data/raw/`、`data/external/`、`data/interim/`、`data/processed/` 或
`data/splits/` 中的实际数据文件。这是有意的版本控制边界，而不是数据未登记。
这些目录由 `.gitignore` 排除，避免将大型、受来源许可约束或可重新生成的资产提交到
GitHub。

当前本地容量快照为：`raw/` 约 122 GiB、`interim/` 约 144 GiB、`processed/`
约 41 MiB、`splits/` 约 5.4 MiB、`external/` 约 208 KiB。容量会随本地数据版本
变化，不能替代注册表校验。

## 资产集合

`datasets.json` 登记了 8 个资产集合，均为 version `1`：

| 数据集 ID | 类型 | 本地相对路径 | 来源/版本 | Git 状态 |
| --- | --- | --- | --- | --- |
| `legacy_raw_collection` | raw | `data/raw/` | `MCPIRE_PDO/runtime/raw` 迁移溯源 | 仅元数据 |
| `legacy_external_collection` | external | `data/external/` | 第三方外部资产迁移溯源 | 仅元数据 |
| `lincs_gse92742_level3_level4_level5` | interim | `data/interim/lincs/GSE92742/` | Broad LINCS `GSE92742` | 仅元数据 |
| `betge_2022` | processed | `data/processed/betge_2022/` | Betge et al. `2022` | 仅元数据 |
| `inflammation_geo_collection` | processed | `data/processed/inflammation/` | GEO 多研究集合 | 仅元数据 |
| `jci_158060` | processed | `data/processed/jci_158060/` | JCI `158060` | 仅元数据 |
| `mcrc_pdo_2026` | processed | `data/processed/mcrc_pdo_2026/` | internal `mcrc-pdo-2026` | 仅元数据 |
| `tornado_gse157167` | processed | `data/processed/tornado_gse157167/` | GEO `GSE157167` | 仅元数据 |

`legacy_*` 名称只描述迁移来源和溯源，不得作为新代码的依赖或数据集身份替代。

## 来源研究与允许用途

下表来自详细来源注册表。`allowed_roles` 和 `forbidden_claims` 是研究设计必须遵守
的边界；它们不自动授予超出原始数据许可的使用权。

| 研究 ID | 来源 | 模型体系 | 允许用途 | 禁止主张 |
| --- | --- | --- | --- | --- |
| `lincs_gse92742` | GEO `GSE92742`，Broad LINCS | 未在注册表确认 | `pretraining`，`mechanism_reference` | 无登记项 |
| `betge_2022` | GEO `GSE117548` | 未在注册表确认 | `external_validation`，`bridge_training` | 无登记项 |
| `mcrc_pdo_2026` | Mendeley `10.17632/hr94h42xdc.3` | 未在注册表确认 | `supervised_training`，`calibration`，`external_validation` | 无登记项 |
| `cf_gse263022` | GEO `GSE263022` | 未在注册表确认 | `mechanism_reference` | 无登记项 |
| `tornado_gse157167` | GEO `GSE157167` | `organoid_3D` | `mechanism_reference`，`external_validation` | `human_pdo_efficacy`，`anti_inflammatory_efficacy`，`full_transcriptome_ground_truth` |
| `ipf_gse281994` | GEO `GSE281994` | 未在注册表确认 | `mechanism_reference` | 无登记项 |
| `ipf_gse282322` | GEO `GSE282322` | `organoid_3D` | `mechanism_reference` | `anti_fibrotic_efficacy`，`clinical_efficacy`，`general_drug_response` |
| `jci_158060_gse212014` | GEO `GSE212014` 与 JCI `158060` | `organoid_3D` | `external_validation`，`bridge_training` | `DSS_equivalence`，`clinical_efficacy_probability`，`anti_inflammatory_efficacy` |

全部 8 项在详细注册表中的 `qc_status` 为 `downloaded`。这只表示迁移时已登记和
核验下载状态，不等价于当前实验已经批准使用，也不等价于任意机器都已拥有本地文件。

## 数据划分与验证

本地 `data/splits/` 目前存在 cold drug、cold scaffold、patient 及 benchmark 相关
Parquet 文件，但它们同样不上传。任何实验必须在计划中指定所用 split 的确切文件、
版本和泄漏防护规则，不能仅凭本目录存在就默认采用。

在拥有数据文件的正式 WSL 环境中运行：

```bash
make data-check
```

该命令验证 `datasets.json` 中登记的相对路径存在。它不会下载、修改或重新生成数据。

## 获取与重建边界

公开来源链接记录在详细注册表的 `notes` 字段，以及部分本地来源 manifest 中；例如
LINCS 和 GEO 使用相应 accession，`mcrc_pdo_2026` 指向 Mendeley 数据集。

当前仓库没有受版本控制的一键下载或 preprocessing 脚本，因而仅凭 GitHub 克隆无法
保证重建完整本地数据树。需要新增下载、数据转换或 split 生成流程时，必须先创建
获批的实验或数据计划，记录来源许可、精确版本、校验和、输出目录和再现命令；不得
把下载结果或大体量产物直接提交到 Git。
