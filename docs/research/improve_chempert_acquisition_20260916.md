# IMPROVE 与 ChemPert 候选资源审计（2026-09-16）

本记录完成公开来源、固定版本和本地资产的审计；没有训练模型，也没有改变任何既有实验结论。

## 结论

| 资源 | 本次新增的实际生物数据 | 当前可用价值 | 当前不能用于 |
|---|---|---|---|
| IMPROVE CSA | 有：下载了单一、固定版本的标准化基准包；其中含统一封装的既有研究和 gCSI/GDSCv1。 | 作为五个细胞系研究的固定跨研究划分与统一格式的候选评测基础。 | PDO/临床验证；未经端点、身份和划分审计的训练或阈值选择。 |
| ChemPert | 有：公开下载了三个 RDS（初始表达、转录反应、药物靶点）和三个元数据表。 | 非癌细胞的药物扰动机制与疾病表达逆转候选参考。 | AUC/IC50 药敏监督、细胞系/PDO 药效外部验证、临床用途。 |

## IMPROVE CSA

- 固定数据记录：[Zenodo 15258883](https://zenodo.org/records/15258883)，DOI
  `10.5281/zenodo.15258883`。该公开记录只包含一个 `csa_data.zip`
  （288,561,534 字节，发布方 MD5 `984eabec912914e62d56d8e0073475f8`）。
- 固定参考实现：[IMPROVE v0.1.0](https://github.com/JDACS4C-IMPROVE/IMPROVE/tree/v0.1.0)。
  本地校验过该源码归档包含 CSA workflow、下载脚本及 MIT 软件许可证。
- 文档中定义的研究为 `CCLE`、`gCSI`、`GDSCv1`、`GDSCv2`、`CTRPv2`；每个源研究有
  `split_0` 至 `split_9` 的训练、验证和测试文件。它可提供可复用的跨研究评测合同，
  但不自动兼容本项目现有表达特征、药物映射或响应方向。
- 完整压缩包已作为独立、不可拆分的固定 benchmark 包保存；它会重复封装本项目已存在或
  已取得的 CCLE/GDSCv2/CTRPv2 资产，并混入新的 gCSI/GDSCv1。后续适配必须保留这一
  独立版本，不能静默替代现有源文件。Zenodo 的当前 API 元数据未声明数据许可证，所以公开
  访问本身不构成训练/再分发许可结论。

## ChemPert

- ChemPert 是“非癌细胞药物扰动转录组”汇编，公开网站显示 82,270 条转录反应、
  2,566 个扰动剂、167 个非癌细胞类型/细胞系/组织；本轮下载的实际元数据为
  82,270 条反应、2,508 个有反应记录的 `Chemical_ID`、343 个原始研究 accession、
  249 个 `Cell_Source`。论文为 Zheng 等，NAR，DOI
  [`10.1093/nar/gkac862`](https://doi.org/10.1093/nar/gkac862)。
- 数据的模态是处理前表达索引、处理后的“转录调控因子响应”及药物—靶点注释；它没有
  细胞存活率、AUC、IC50 或患者/PDO 药物响应配对。因此它能补充机制与表达逆转分析，
  不能补充当前药敏监督矩阵。
- 82270 条记录中，34,702 条（42.18%）来自本项目已有的 LINCS Phase I `GSE92742`；
  42,978 条（52.24%）来自当前未登记本地资产的 LINCS Phase II `GSE70138`；元数据中
  没有 `GSE139944`，故没有直接的 sci-Plex accession 重叠。前一部分是对已有 LINCS
  原始来源的非癌子集再整理，并非独立药物扰动实验。
- 公开网页允许直接下载，但下载页与网站条款没有给出数据集再利用许可证；代码仓库标为
  AGPLv3，不能外推为数据许可证。候选资源在许可和模态合同复核完成前不进入训练数据。
  ChemPert 网站条款也明确其工具不用于临床用途。

## 本地证据与复核

- 完整的机器可读审计（版本、URL、哈希、文件大小、计数、限制）：
  `data/artifacts/resource_acquisition_20260916/improve_chempert/improve_chempert_audit.json`。
- 独立候选登记：
  `data/registry/datasets_improve_chempert_candidates_20260916.json`；它没有修改共享
  `datasets.json`，也不表示资源已经准入。
- 可重复生成命令（必须在 WSL2 的 `drugscreening-gpu` 环境中运行）：

```bash
cd /mnt/d/Code/DrugScreenLab
/home/dell/miniconda3/bin/conda run --no-capture-output -n drugscreening-gpu \
  python scripts/data/audit_improve_chempert_acquisition.py --root data
```
