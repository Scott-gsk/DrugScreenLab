# 数据阶段摸底清单 DRAFT

> **状态：DRAFT** — 摸底稿；未冻结 checksum；未启动大训练；不得当作验收完成。  
> **日期：** 2026-09-18  
> **生成：** 2026-09-18 14:10 CST（Asia/Shanghai）  
> **角色：** DSL-数据管家 / data-phase P0  
> **依据：** `docs/研究实施方案.md` §二–§三；`docs/exec-plans/active/pathway_transfer_pdo_20260918.md`  
> **机器可读来源（已读）：** `data/registry/datasets.json`、`legacy_runtime_registry.jsonl`、`migration_manifest.json`、`data/DATA_CATALOG.md`、`data/README.md`、`experiments/records/EXP-001/readiness_matrix.json`（仅作登记缺口参考，**不作选模证据**）、git 树内 `mvp/phase1/CTRP_EXTERNAL_ASSET_AUDIT.json`  
> **本机 raw / kryeziu_audit：** 本 executor **未** 通过 `machineId=96b95440-…` 到达用户 WSL；实地 `ls`/`du` **未执行**。

---

## 0. 可达性与证据边界

| 项 | 结果 |
|---|---|
| 目标机 | `machineId=96b95440-6d33-4ea8-a3f3-6284d6770786`（`D:\\Code\\DrugScreenLab` / `/mnt/d/Code/DrugScreenLab`） |
| 本 executor Shell | 落在 Grok Bot box；`machineId` 参数**未注入**工具面 |
| conda `drugscreening-gpu` | box 上不可用；用户机未实测 |
| `data/raw/` | **禁止修改**；本轮未触碰 |
| Kryeziu audit 路径 | 期望 `data/artifacts/resource_acquisition_20260916/kryeziu_audit/` — **未观测** |
| 证据层级 | **A** 方案宣称规模（§二.1）· **B** git 注册表已读事实 · **C** 用户机实地统计（缺失） |

正式冻结前须在用户机 WSL 用 §8 命令把 **C** 填齐。

---

## 1. 关键数据集登记（方案 × 注册表）

方案规模摘自 `docs/研究实施方案.md` §二.1（设计口径）。注册表 id 摘自 `data/registry/datasets.json`（B 级）。

| 数据集 | 方案宣称规模（A） | 角色（方案） | registry id(s)（B） | 相对路径（B） | asset_type（B） | 可观测规模（B，无重算） | 相对方案表缺口 |
|---|---|---|---|---|---|---|---|
| **Kryeziu**（主 PDO） | 91 表达+药敏样本 / 48 结构确认单药 / 3,183 评分 | 主 PDO 建模与验证 | **无** `kryeziu*` id；近邻 **`mcrc_pdo_2026`**（Mendeley `10.17632/hr94h42xdc.3`） | `processed/mcrc_pdo_2026` | processed | 注册表无 nrows；legacy 列 Data S1–S5 文件 hash；**patient 表未读** | 缺统一 Kryeziu 身份/药物登记 id；缺 audit→registry 映射；缺患者 5 折冻结 |
| **GDSC(2) / Cell Model Passports** | 963 细胞系 / 219 药 / 179,410 响应 | 细胞系药敏建模 | **无** `gdsc*` 于 `datasets.json` | （未登记） | — | EXP-001 readiness 曾标 `GDSC1_2` absent（历史快照） | **未登记**完整 GDSC2+Passports；缺 identity 对齐 |
| **CTRPv2** | 820 细胞系 / 544 药 / 364,761 响应 | 扩大训练；跨研究评价 | **无**完整 CTRPv2 sensitivity 于 `datasets.json` | 外部审计指向 NCI/CTD2 zip | — | `CTRP_EXTERNAL_ASSET_AUDIT`：官方文件服务不可用 | 官方包未落地；缺药物映射表 |
| **PRISM** | 877 细胞系 / 104 药 / 59,037 响应 | **仅跨研究评价；不参与超参** | `prism_repurposing_primary_19q4_compact_crc_v1`；`lincs_prism_identity_bridge_v1`；`xpert_broad_prism_crc_response_v1` | `../mvp/core_data/compact_prism_response.parquet` 等 | processed / derived_metadata | compact：**135** rows，**35** CRC lines，**4** candidates（≠ 方案全量 59k） | 全量 PRISM 未按方案规模登记 |
| **LINCS GSE92742** | 药物处理表达 / 对照 / 签名 | 转录扰动 | `lincs_gse92742_raw_level3_v1`；`…_level3_level4_level5`；`…_exact978_cache_v1` | `raw/lincs/GSE92742`；`interim/…`；`processed/…/exact978_cache_v1` | raw / interim / processed | Level3 gctx.gz **52407725132** bytes；exact978 **[1319138 × 978]** | Level3 vs Level5 分合同待冻结；ChemPert 交叉未登记 |
| **ChemPert** | 多来源药物转录响应 | 扰动补充 / 跨来源 | **无** `chempert*` | （未登记） | — | 未观测 | **整源未登记** |
| Zhu | 20 配对 / 54 药 / 1,080 响应 | 第二 PDO | **无** | — | — | 未观测 | 未登记 |
| Betge（注册表有） | — | external / bridge | `betge_2022` | `processed/betge_2022` | processed | 本轮未读 nrows | 非七源主表但已登记 |

`DATA_CATALOG.md` 历史容量快照：raw≈122 GiB，interim≈144 GiB，processed≈41 MiB，splits≈5.4 MiB — **未复测**。

---

## 2. 已有身份 / 药物表或审计（路径）

| 路径 | 性质 | 备注 |
|---|---|---|
| `data/registry/datasets.json` | 资产登记 | LINCS / PRISM compact / mcrc 等 |
| `data/registry/legacy_runtime_registry.jsonl` | 来源研究级登记 | 含 mcrc / LINCS 文件 hash |
| `data/registry/migration_manifest.json` | 迁移溯源 | processed 集合 |
| `data/DATA_CATALOG.md` / `data/README.md` | 人类可读目录 | raw 不进 Git |
| `mvp/extension/LINCS_PRISM_IDENTITY_BRIDGE_AUDIT.json` | 身份桥 | registry: `lincs_prism_identity_bridge_v1` |
| `mvp/phase1/CTRP_EXTERNAL_ASSET_AUDIT.json` | CTRP 获取阻塞 | 官方 zip 不可用 |
| `mvp/extension/GDSC_FROZEN_BENCHMARK_CONTRACT.md` | GDSC 评测合同 | 非全量登记 |
| `mvp/data/MVP-001_betge_2022_identity_audit.json` | Betge 身份 | 禁止 mcrc 字段污染 |
| `experiments/records/EXP-001/readiness_matrix.json` | 历史 P0 | **勿作新协议选模证据** |
| `data/artifacts/resource_acquisition_20260916/kryeziu_audit/` | **期望** Kryeziu 审计 | **本轮未读到** |

---

## 3. 明确缺口（对照数据阶段交付）

| 交付 | 状态 | 说明 |
|---|---|---|
| **统一 identity master table** | **缺失（仅合同草稿）** | `identity_contracts.py`；无 parquet/json 主表 |
| **三套表达输入合同** | **接口 DRAFT 已写** | `expression_contracts.py`；无重预处理 |
| **覆盖报告（coverage）** | **缺失** | 需用户机审计 |
| **Kryeziu 患者外 5 折 DRAFT** | **schema DRAFT；folds 空** | 见 `data/splits/*DRAFT*`；**无真实 patient_id** |
| 数据版本冻结清单 | 部分（LINCS/PRISM compact） | GDSC2/CTRPv2/ChemPert/Kryeziu 主表未齐 |
| 描述性分析 | **缺失** | 未跑 |

---

## 4. 代码骨架（本轮）

| 模块 | 路径 |
|---|---|
| 身份/药物合同 | `src/drug_screen/data/identity_contracts.py` |
| 三套表达合同 | `src/drug_screen/data/expression_contracts.py` |
| 兼容再导出 | `src/drug_screen/data/contracts.py` |
| 包导出 | `src/drug_screen/data/__init__.py` |
| 既有模块（自 git） | `registry.py` / `p0.py` / `root.py` / `exp002.py` / `phase1.py` / `prism_bridge.py` |
| 单测 | `tests/test_data_contracts_draft.py` |

规则：raw 只读；预处理 train-only fit；未测响应 ≠ 0。

---

## 5. 划分草案

`data/splits/kryeziu_patient_outer5fold_DRAFT.json` + `README_kryeziu_patient_outer5fold_DRAFT.md`  
`status=DRAFT`，`folds=[]`，`used_real_patient_ids=false`。

---

## 6. Import / 测试

用户机：

```bash
cd /mnt/d/Code/DrugScreenLab
PYTHONPATH=src conda run --no-capture-output -n drugscreening-gpu \
  python -c "from drug_screen.data import GeneExpressionContract, StubIdentityTableLoader; print('ok')"
PYTHONPATH=src conda run --no-capture-output -n drugscreening-gpu \
  pytest -q tests/test_data_contracts_draft.py
```

box 系统 Python 冒烟见执行记录；conda **未跑**。

---

## 7. 阻塞（P0）

1. 用户机 Shell（machineId）对本 executor 不可用。  
2. 无真实 Kryeziu patient_id 表 → folds 无法填充。  
3. GDSC2 / CTRPv2 / ChemPert 未在 `datasets.json` 完整登记。  
4. identity master + coverage report 文件尚未产出。

---

## 8. 用户机复跑（只读）

```bash
cd /mnt/d/Code/DrugScreenLab
find data/registry data/raw data/processed data/splits data/artifacts \
     src/drug_screen/data -maxdepth 3 \( -type d -o -type f \) 2>/dev/null | sort | head -400
rg -n -i "kryeziu|GDSC|CTRP|PRISM|ChemPert|GSE92742|patient" data/registry data/artifacts 2>/dev/null | head -200
ls -la data/artifacts/resource_acquisition_20260916/kryeziu_audit 2>/dev/null
PYTHONPATH=src conda run --no-capture-output -n drugscreening-gpu \
  python -c "from drug_screen.data import identity_contracts, expression_contracts; print('ok')"
```

---

## 附录 — 引用

- `docs/研究实施方案.md` §二、§三  
- `docs/exec-plans/active/pathway_transfer_pdo_20260918.md`  
- `data/registry/datasets.json`（20 entries）  
- `data/DATA_CATALOG.md`  
- Mendeley：`10.17632/hr94h42xdc.3`（本轮未下载数据文件）

## Changelog
- 2026-09-18 14:17 CST: Added `data/processed/identity/kryeziu_drug_identity_48_DRAFT.{json,parquet}` from `processed/transfer/mcrc_v3/drugs.parquet` (n=48, Oxaliplatin quarantined/absent). Extended `mcrc_source.load_processed_mcrc_v3` to read processed-only tables.

