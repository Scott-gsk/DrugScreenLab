# EXP-001 P0 审计报告

审计日期：2026-08-11。范围仅限已批准的 Phase 0；未下载、训练、预处理或修改 `data/raw/`。

## 结论

`phase_0_readiness_verdict: PARTIAL`。本地 Level-2 文件确实不存在，但这不是
Level-3 landmark-core `Delta978` 的自动 blocker。官方 GEO 定义 Level-3 为经
invariant-set scaling 和 quantile normalization 后的“directly measured landmark
transcripts plus inferred genes”；本地 `gene_info.pr_is_lm` 精确标记 978 个 landmark。
因此可为这 978 个基因建立 matched-control `Delta978` 派生表，但本轮**没有**读取矩阵
数据或 materialize 任何 Delta table。inferred non-landmark genes 仍不得作为真实
whole-transcriptome ground truth。

## GSE92742 直接证据

`data/raw/lincs/GSE92742/` 文件名只包含：

- `GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx.gz`
- `GSE92742_Broad_LINCS_Level4_ZSPCINF_mlr12k_n1319138x12328.gctx.gz`
- `GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz`

`data/interim/lincs/GSE92742/` 是上述三个文件的解压副本。目录中没有名称含
`Level2` 的文件；`data/registry/datasets.json` 亦仅登记
`lincs_gse92742_level3_level4_level5`。这只表示 Level-2 可选资源 `UNAVAILABLE`，
不表示 Level-3 的 978 direct-landmark subset 不可用。

## Level-3 Landmark-Core Evidence

- HDF5 header：`matrix_shape=(1,319,138, 12,328)`；GCTX column IDs 与
  `inst_info.inst_id` 的集合一致（顺序不同）。
- `gene_info`：12,328 unique `pr_gene_id`，其中 `pr_is_lm=1` 恰为 978。
- 化学实例：`trt_cp=672,128`。以
  `(rna_plate, cell_id, pert_time, pert_time_unit)` 匹配 `ctl_vehicle`，671,881
  (99.963251%) 有候选同 plate vehicle control，247 必须排除，不能使用 global control
  或伪造配对。
- 可复现命令：
  `PYTHONPATH=src /home/dell/miniconda3/bin/conda run --no-capture-output -n drugscreening-gpu python scripts/audit_exp001_level3.py --root data`。

官方依据：[GEO GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742)
说明 Level-3 包含 direct landmark 与 inferred genes；
[XPert Methods](https://www.nature.com/articles/s42256-025-01165-w) 从 Level-3 取
treatment/control，并从同 plate 随机选择 DMSO control。XPert 是 preprocessing
先例，不等于本轮已经生成其 z-score 或 replicate-collapsed vectors。

## Readiness Matrix

机器可读矩阵：`readiness_matrix.json`。所有冻结资源均保持批准记录中的角色；特别是
`GSE117548` 为 `EXTERNAL_TEST`，没有沿用旧迁移登记的 `bridge_training` 角色。

## 运行时与完整性

- WSL 数据盘 `/mnt/d`：3.7 TiB，总可用约 215 GiB（95% 已用）。
- 内存：125 GiB；Swap：32 GiB。
- `data/raw/` 审计时文件总量：130,813,873,359 bytes。
- LINCS 的 `cell_info`、`gene_info`、`pert_info`、`sig_info`、`sig_metrics` 与
  `GSE92742_SHA512SUMS` 六个小型 metadata 文件通过 `SHA256SUMS.local` 逐项校验。
  该清单包含自身的旧 checksum，故未将“清单校验自身”的不匹配解释为数据矩阵损坏；
  三个大矩阵未在本轮重新散列，完整性仅由既有登记与上述 metadata 证据支持。
- 本轮 Git 状态未显示上述被忽略数据目录的变化，但未建立审计前后的完整快照；因此
  无法独立证明 `data/raw/` 在本轮未被写入。大矩阵也未重新散列。
- Conda 不在 shell `PATH`，但发现 `/home/dell/miniconda3/bin/conda`；后续正式命令应显式使用该路径。

## Blockers

1. 未 materialize matched-control `Delta978` 派生表；247 个 chemical instances 没有同 plate vehicle candidate，后续必须排除。
2. 大型 Level-3 matrix 没有在本轮重新散列。
3. CCLE/DepMap、UniPert、GSE139944、GSE186341、PRISM、CTRPv2、GDSC1/2、GSE280506、GSE145308 未本地登记。
4. 可用的 STRING 与 Reactome/GO 文件缺少独立 registry version/license/checksum 证据。

## SCIENTIFIC_CLARIFICATION

先前把“缺少本地 Level-2”写成“Level-3 不能提供真实 `Delta978`”是误判，现已依据
GEO 层级定义、`gene_info.pr_is_lm` 和 XPert Level-3 同 plate DMSO 配对流程修正。
修正仅限 landmark core：不提升 inferred non-landmark genes 的证据等级，也不改变
EXP-001 hypothesis、数据角色或禁止事项。

## DEVIATION_FROM_PLAN

无。计划要求区分直接 978 core 与 inferred gene space；本澄清恢复该边界，并未下载
Level-2 或创建/训练派生数据。

## Independent Reviewer Input

`VALID`。Independent Reviewer 在不修改实现的前提下重新核对 GEO 的 Level-3 定义、
XPert 的 Level-3 同 plate DMSO 配对先例、本地 landmark mapping、instance metadata
连接、配对候选计数和测试结果。Reviewer 确认 inferred non-landmark genes 未被放宽为
experimental ground truth，Level-2 不再构成 Phase-1 hard dependency；但 `PARTIAL`
仍然正确，因为 Delta 表尚未 materialize、247 个实例必须排除、大矩阵未重散列，且其他
P0 资源仍有登记与 provenance blocker。完整独立意见见 `REVIEWER_INPUT.md`。
