# EXP-015 Zhu 结构缺口审计

本审计不修改 `data/processed/transfer/zhu_v2`，也不阻塞现有 33 药探索性训练。它只使用作者 Table S4 中的药物编号、Selleck catalogue 和同一工作簿的 301 化合物表进行严格连接；没有按药名模糊匹配。

## 已核对的范围

- 目标表有 64 个进一步筛选药；`zhu_v2` 现有 33 个具当前保守结构规则支持的单一小分子，故待审计缺口为 31 个。
- 31 个中有 20 个可由 `Further drug screen_Drug list` 的 catalogue 与 `Chemical Data-301cpds.cat` 一对一精确连接，并得到一个可解析的完整 InChIKey。
- 现有 GDSC v2（227 药）比较采用完整 InChIKey 为正式交集；前 14 位 connectivity 仅作诊断，绝不静默合并立体异构体。

| 范围 | 完整 InChIKey 交集 | connectivity 交集 |
| --- | ---: | ---: |
| 已纳入的 33 Zhu 药 | 3 | 4 |
| 20 个 catalogue 精确且结构可解析的缺药 | 4 | 4 |
| 盐/水合物的单一有机母体候选 | 7 | 7 |

已纳入 33 药中，Piperlongumine 是唯一 connectivity 相同而完整 InChIKey 不同的例子：Zhu 为 `VABYUUZNAVQNPG-BQYQJAHWSA-N`，GDSC 为 `VABYUUZNAVQNPG-UHFFFAOYSA-N`。它不进入完整身份交集。

## 直接可复核的 catalogue 结构候选

下列项目可由同一作者工作簿的 catalogue 一对一定位，不是药名推断。它们仍没有被加入 `zhu_v2`；若要扩展，需单独冻结这条 catalogue→结构规则并重新生成队列。

| Zhu 编号 | 药物 | catalogue | InChIKey | GDSC 完整交集 |
| --- | --- | --- | --- | --- |
| 03 | SNX-2112 | S2639 | ZFVRYNYOPQZKDG-UHFFFAOYSA-N | 否 |
| 04 | Rigosertib | S1362 | OWBFCJROIKNMGD-BQYQJAHWSA-M | 否 |
| 05 | Adavosertib | S1525 | BKWJAKQVGHWELA-UHFFFAOYSA-N | 是 |
| 06 | Sonidegib | S2151 | VZZJRYRQSPEMTK-UHFFFAOYSA-N | 否 |
| 11 | 2-Methoxyestradiol | S1233 | CQOQDQWUFQDJMK-UHFFFAOYSA-N | 否 |
| 13 | Tivantinib | S2753 | UCEQXRCJXIVODC-UHFFFAOYSA-N | 否 |
| 22 | Erdafitinib | S8401 | OLAHOMJCDNXHFI-UHFFFAOYSA-N | 否 |
| 23 | Vorinostat | S1047 | WAEXFXRVDQXREF-UHFFFAOYSA-N | 是 |
| 24 | Tozasertib | S1048 | GCIKSSRWRFVXBI-UHFFFAOYSA-N | 是 |
| 26 | Belinostat | S1085 | NCNRHFGMJRPRSK-MDZDMXLPSA-N | 否 |
| 29 | Azacitidine | S1782 | NMUSYJAQQFHJEW-UHFFFAOYSA-N | 是 |
| 32 | GMX1778 | S8117 | BOIPLTNGIAPDBY-UHFFFAOYSA-N | 否 |
| 41 | RAF265 | S2161 | YABJJWZLRMPFSI-UHFFFAOYSA-N | 否 |
| 47 | Seocalcitol | HY-32341 | LVLLALCJVJNGQQ-SEODYNFXSA-N | 否 |
| 49 | Everolimus | S1120 | HKVAMNSJSFKALM-PYJGUTGCSA-N | 否 |
| 50 | Regorafenib | S1178 | FNHKPVJBJVTLMP-UHFFFAOYSA-N | 否 |
| 52 | Flavopiridol | S1230 | BIIVYFLTOXDAOV-UHFFFAOYSA-N | 否 |
| 55 | Ispinesib | S1452 | QJZRFPJCWMNVAV-UHFFFAOYSA-N | 否 |
| 57 | Panobinostat | S1030 | FPOHNWQLNRZRFC-ZHACJKMWSA-N | 否 |
| 59 | Pacritinib | S8057 | HWXVIOGONBBTBY-ONEGZZNKSA-N | 否 |

## 有直接 catalogue 证据、但需独立盐处理规则的项目

作者 301 表以多组分 SMILES 记录以下制剂。当前结构标准化器只移除显式安全反离子，不选择中性 `Cl`、水合物或有机对离子，因此正确地排除了它们。每项都有唯一的碳骨架候选，下面仅报告候选；不代表已准入。

| Zhu 编号 | 药物 | catalogue | 有机母体 InChIKey | 与 GDSC 完整交集 |
| --- | --- | --- | --- | --- |
| 07 | Irinotecan HCl Trihydrate | S2217 | UWKQSNNFCGGAFS-UHFFFAOYSA-N | 是 |
| 27 | Ruboxistaurin HCl | S7663 | ZCBUQCWBWNUWSU-UHFFFAOYSA-N | 否 |
| 30 | Mitoxantrone 2HCl | S2485 | KKZJGLLVHKMTCM-UHFFFAOYSA-N | 是 |
| 46 | Palbociclib Isethionate | S1579 | no accepted parent under current rule | — |
| 63 | Fingolimod HCl | S5002 | KKGQTZUTZRNORY-UHFFFAOYSA-N | 否 |
| 64 | Doxorubicin HCl | S1208 | AOJJSUZBOXZQNB-UHFFFAOYSA-N | 是 |

Sorafenib tosylate、Afatinib dimaleate 与 Palbociclib isethionate 含有机对离子，不能由“唯一碳骨架”规则安全地自动选择，继续排除。JNJ-64619178、Romidepsin、Trichostatin A 在作者 301 表没有对应 catalogue 记录，不能通过此证据链补齐。

## 可重复检查

`scripts/audit_zhu_structure_gaps.py` 在 WSL2 Conda `drugscreening-gpu` 中读取作者 `mmc5.xlsx` 与 `data/processed/exp014/gdsc_v2/drugs.parquet`，输出每个缺口的 catalogue 匹配数、解析状态、完整 InChIKey、connectivity 诊断和母体候选。这是只读审计脚本，不写入任何队列或注册表。
