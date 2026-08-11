# 数据资产

`raw/` 保存不可修改的源数据；`external/` 保存第三方已处理资产；`interim/`
保存可重新生成的中间数据；`processed/` 保存可供实验使用的数据；`splits/`
保存带元数据和版本信息的数据划分。

大型数据资产有意不纳入 Git。每个被实验使用的数据集都必须在 `registry/` 中有
JSON 登记项，记录其 identity、version、source、relative path、checksum 和
provenance。上述 schema key 保持英文，以维持程序接口稳定。

原始数据目录不得被 preprocessing、训练或评估代码修改。需要生成的新数据应写入
适当的中间或处理后目录，并与相应的实验和数据集版本建立可审计关联。

当前本地数据集、来源、允许用途、禁止主张、Git 状态与重建边界见
`data/DATA_CATALOG.md`。该目录由受版本控制的 `registry/` 元数据说明，不代表
大型数据文件已上传到 GitHub。
