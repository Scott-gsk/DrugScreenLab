# ChemPert 合同并行摸底 DRAFT（路线二）

- 状态：DRAFT / 摸底；**未训练**；未物化全量表达矩阵
- 时间：2026-09-18 17:45 CST
- 执行：DSL-扰动建模（并行于 residual drug-cond；合同正式冻结仍归 DSL-数据管家）

## 已在盘资产

| 资产 | 路径 | 状态 |
|---|---|---|
| 响应元数据 CSV | `data/raw/.../improve_chempert/chempert_information_for_transcriptional_responses.csv` | 82,270 行（与 audit 一致） |
| 扰动元数据 CSV | `chempert_information_for_perturbagens.csv` | 在盘 |
| 表达信息 CSV | `chempert_information_for_gene_expression.csv` | 在盘 |
| responses RDS | `chempert_transcriptional_responses.rds` | 在盘；schema audit 已解码 82,270×5 |
| expression RDS | `chempert_gene_expression_profiles.rds` | 在盘；**全表转换未做** |
| targets RDS | `chempert_targets_of_perturbagens.rds` | 在盘；全表转换未做 |
| schema audit | `data/artifacts/.../chempert_schema_audit.json` | 有 |
| acquisition audit | `data/artifacts/.../improve_chempert_audit.json` | 有（provenance only） |
| 物化脚本 | `scripts/data/materialize_chempert_metadata.py` | 有 |
| 获取审计脚本 | `scripts/data/audit_improve_chempert_acquisition.py` | 有 |

## 合同缺口（阻塞正式扰动评测）

1. **无冻结 ChemPert 扰动基因合同**：Level/签名形态、对照可用性、化/遗分离未钉死。
2. **expression/targets RDS 未转为常用矩阵**（audit 明确 full dataframe conversion not attempted）。
3. **ChemPert∩LINCS 防泄漏键表未冻结**（目录仅有数量级描述；评价侧可挂 `check_split_leakage` 干跑，但缺正式划分产物）。
4. **无 processed/chempert 物化树**（此前 inventory 已记）。
5. 与 IMPROVE CSA zip 同目录存放，勿把 CSA 药敏基准误当作 ChemPert 扰动标签。

## 建议下一合同切片（给数据管家，非本角色越权冻结）

1. 冻结 Response_ID ↔ Chemical_ID ↔ Data_Accession ↔ Expression_ID 主键表 + checksum。
2. 标注 GSE92742 / GSE70138 / 其他 accession 分组，输出 ∩ 泄漏原子键。
3. 对照可用性字段：无可靠对照 → `delta_supervision=false`。
4. 可选：先物化「仅元数据 + landmark 交集」小合同，再谈全基因组 RDS。

## 路线二当前正式口径（不受本摸底改变）

- LINCS ctrl→Δ 高分主因对照结构；删除对照 + Morgan 不支持药特异。
- ChemPert 未齐前，不宣称跨库扰动泛化。