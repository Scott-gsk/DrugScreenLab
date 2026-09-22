# EXP-014 CTRPv2 PharmacoSet 来源审计

原始文件 `data/raw/exp013/ctrp_v2/CTRPv2.rds` 与已登记来源
`10.5281/zenodo.3905470` 的大小（43,118,179 bytes）和 SHA-256
`8587d711265261ee23498c3d18efe74f149134e38fd27dc2047b4259a54ccb28` 一致。

该文件是 gzip 压缩的 R 序列化对象。只读嵌入字符串扫描发现 `PharmacoSet` 类，以及
`sensitivityInfo`、`sensitivityRaw`、`sensitivityProfiles`、`sensitivityN`、
`sensitivity`、`ctrp.drugs`、`ctrp.cells`、`molecularProfiles` 槽位标识；发现
`cellid`、`smiles`、`cpd_smiles` 字段标识。这些仅证明相应名称被嵌入，不能证明列的
类型、行数、响应方向或关联关系。

当前环境的 `pyreadr` 返回 `LibrdataError: The file contains an unrecognized object`，不能
解析该 S4 `PharmacoSet`。因此没有创建 `data/processed/transfer/ctrp_v2/` 中的表：不能把
RDS 字节流或字符串扫描结果当作药敏矩阵，也不能把 `sensitivity*` 槽位名称断言为具体
AUC/IC50 终点。后续只有在 S4-capable R/Bioconductor 读取器可逐槽 materialize 且审计
具体 schema 后，才可导出 response、药物结构和细胞身份表，并由独立流程决定是否登记。
