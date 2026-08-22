"""Bounded, response-blind EXP-008 identity rescue audit."""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path

SDST = Path("/mnt/d/Code/DrugScreenLab/data/external/xpert_source/processed_data/l1000_sdst_drugs_info_8432.csv")
CHEMBL = Path("/mnt/d/Code/TranSiGen/data/screening/libraries/chembl_36/chembl_36_sqlite/chembl_36.db")
UNIPROT = Path("/mnt/d/Code/DrugScreenLab/data/external/unipert_source/data/ref_targets.csv")
GENE978 = Path("/mnt/d/Code/DrugScreenLab/data/external/xpert_source/processed_data/l1000_gene_info_978.csv")
OUT = Path("/mnt/d/Code/DrugScreenLab/artifacts/experiments/EXP-008")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def columns(con: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")')]


def main() -> None:
    rows = []
    with SDST.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["pert_id"], r["inchi_key"]))
    unique = sorted(set(rows))
    # The CSV is an 8,432-row drug table; the official SDST h5ad retains 8,276
    # unique pert_idx across 8,382 pert_id values. Restrict rescue to that approved universe.
    sdst_h5ad = Path("/mnt/d/Code/DrugScreenLab/data/external/xpert_source/processed_data/l1000_sdst_78453.h5ad")
    import anndata as ad
    backed = ad.read_h5ad(sdst_h5ad, backed="r")
    try:
        official_pairs = backed.obs[["pert_id", "pert_idx"]].copy()
        official_pairs["pert_id"] = official_pairs["pert_id"].astype(str)
        official_pairs["pert_idx"] = official_pairs["pert_idx"].astype(int)
        official_pairs = official_pairs.drop_duplicates("pert_idx").sort_values("pert_idx", kind="stable")
        official_pert = set(official_pairs["pert_id"])
    finally:
        backed.file.close()
    unique_pert = sorted(official_pert)
    unique = sorted((p, k) for p, k in set(rows) if p in official_pert)
    inchi_unique = sorted({k for _, k in unique if k})

    con = sqlite3.connect(f"file:{CHEMBL}?mode=ro", uri=True)
    try:
        con.execute("CREATE TEMP TABLE sdst_drug (pert_id TEXT NOT NULL, inchi_key TEXT NOT NULL)")
        con.executemany("INSERT INTO sdst_drug VALUES (?, ?)", [(p, k) for p, k in unique if k])
        con.execute("CREATE INDEX idx_tmp_sdst_inchi ON sdst_drug(inchi_key)")
        # Exact InChIKey equality only; all joins are indexed by SQLite on std key and temp key.
        exact = con.execute(
            """SELECT s.pert_id, s.inchi_key, m.chembl_id, m.molregno
               FROM sdst_drug s
               JOIN compound_structures c ON c.standard_inchi_key = s.inchi_key
               JOIN molecule_dictionary m ON m.molregno = c.molregno
               ORDER BY s.pert_id, m.chembl_id"""
        ).fetchall()
        drug_ids = sorted({r[0] for r in exact})
        # Explicit ChEMBL36 schema: drug_mechanism.direct_interaction, target_dictionary filters,
        # target_components/component_sequences provide the UniProt accession.
        targets = con.execute(
            """SELECT DISTINCT e.pert_id, e.chembl_id, td.tid, cs.accession,
                              td.organism, td.target_type, dm.direct_interaction
               FROM (
                 SELECT s.pert_id, m.chembl_id, m.molregno
                 FROM sdst_drug s
                 JOIN compound_structures c ON c.standard_inchi_key = s.inchi_key
                 JOIN molecule_dictionary m ON m.molregno = c.molregno
               ) e
               JOIN drug_mechanism dm ON dm.molregno = e.molregno
               JOIN target_dictionary td ON td.tid = dm.tid
               JOIN target_components tc ON tc.tid = td.tid
               JOIN component_sequences cs ON cs.component_id = tc.component_id
               WHERE td.organism = 'Homo sapiens'
                 AND td.target_type = 'SINGLE PROTEIN'
                 AND dm.direct_interaction = 1
                 AND cs.accession IS NOT NULL
               ORDER BY e.pert_id, e.chembl_id, cs.accession"""
        ).fetchall()
        schema = {t: columns(con, t) for t in (
            "molecule_dictionary", "compound_structures", "drug_mechanism",
            "target_dictionary", "target_components", "component_sequences"
        )}
        index_info = {
            "compound_structures": [r[1] for r in con.execute("PRAGMA index_list(compound_structures)")],
            "molecule_dictionary": [r[1] for r in con.execute("PRAGMA index_list(molecule_dictionary)")],
            "temporary_sdst_drug": "idx_tmp_sdst_inchi",
        }
    finally:
        con.close()

    # Local files are inventory evidence only. ref_targets is symbol->accession and gene_info is
    # Entrez->symbol; no accession->Entrez crosswalk exists, so symbol chaining is prohibited.
    local_crosswalk = False
    report = {
        "status": "BLOCKED",
        "exp_id": "EXP-008",
        "scope": {"response_blind": True, "efficacy_labels_read": False, "downloads": False},
        "sources": {
            "sdst_drug_csv": {"path": str(SDST), "sha256": sha256(SDST)},
            "chembl36_sqlite": {"path": str(CHEMBL), "sha256": sha256(CHEMBL)},
            "existing_uniprot_reference": {"path": str(UNIPROT), "sha256": sha256(UNIPROT), "role": "symbol_to_uniprot_only"},
            "official_978_gene_info": {"path": str(GENE978), "sha256": sha256(GENE978), "role": "entrez_to_symbol_only"},
        },
        "sdst_counts": {
            "raw_rows": len(rows), "csv_unique_pert_ids": len({p for p, _ in rows}),
            "official_sdst_unique_pert_ids": len(unique_pert), "official_sdst_unique_pert_idx": 8276,
            "unique_pert_inchi_pairs_in_official_universe": len(unique), "unique_nonempty_inchi_keys": len(inchi_unique),
        },
        "exact_drug_join": {
            "rule": "SDST inchi_key = ChEMBL36 compound_structures.standard_inchi_key",
            "matched_rows": len(exact), "matched_unique_pert_ids": len(drug_ids),
            "coverage_over_unique_pert_ids": len(drug_ids) / len(unique_pert),
            "duplicate_chembl_rows": len(exact) - len(drug_ids),
            "sample_top20": [
                {"pert_id": p, "inchi_key": k, "chembl_id": c, "molregno": mr}
                for p, k, c, mr in exact[:20]
            ],
        },
        "target_filter": {
            "rule": "target_dictionary.organism='Homo sapiens' AND target_type='SINGLE PROTEIN' AND drug_mechanism.direct_interaction=1",
            "schema": schema, "indexes": index_info,
            "target_accession_rows": len(targets),
            "target_accession_unique_pert_ids": len({r[0] for r in targets}),
            "sample_top20": [
                {"pert_id": p, "chembl_id": c, "tid": tid, "accession": a, "organism": o, "target_type": tt, "direct_interaction": di}
                for p, c, tid, a, o, tt, di in targets[:20]
            ],
        },
        "accession_to_entrez": {
            "status": "MISSING",
            "crosswalk_found": local_crosswalk,
            "checked_local_sources": [str(UNIPROT), str(GENE978)],
            "reason": "仅有 symbol→UniProt accession 与 Entrez→symbol 两个方向表；禁止通过 gene symbol 反推 accession→Entrez，故无可审计精确交叉表。",
            "mapped_target_rows": 0,
            "coverage_over_exact_drug_ids": 0.0,
        },
        "minimum_gap": "提供经批准、版本化的 accession→Entrez 精确交叉表后，按 accession 精确连接并重建；在此之前合同保持 DATA_BLOCKED。",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "identity_mapping_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = f"""# EXP-008 身份映射数据救援报告

## 结论

**BLOCKED**。药物身份段已完成 ChEMBL36 InChIKey 精确连接，但 accession→Entrez 缺少批准的本地精确交叉表，因此不得更新机制合同为可训练状态，也不得伪造 coverage。

## 精确统计

- SDST CSV 原始行：`{len(rows)}`；官方 SDST h5ad unique `pert_id`：`{len(unique_pert)}`，unique `pert_idx`：`8276`（按合同的药物 universe）。
- InChIKey 精确等值连接：`{len(drug_ids)}` / `{len(unique_pert)}` unique `pert_id`，coverage=`{len(drug_ids)/len(unique_pert):.6f}`。
- ChEMBL36 人类 `SINGLE PROTEIN` 且 `direct_interaction=1`：`{len(targets)}` accession 行，涉及 `{len({r[0] for r in targets})}` 个药物。
- accession→Entrez：`MISSING`，可审计映射行 0，覆盖率 `0.000000`。

## Schema 与审计

使用 ChEMBL36 SQLite 真实字段：`compound_structures.standard_inchi_key`、`molecule_dictionary.chembl_id/molregno`、`drug_mechanism.molregno/tid/direct_interaction`、`target_dictionary.tid/organism/target_type`、`target_components.tid/component_id`、`component_sequences.component_id/accession`。SQL 建立临时 `sdst_drug` 表及 `idx_tmp_sdst_inchi`，并利用 ChEMBL `idx_cmpdstr_stdkey` 索引；未进行全库逐行 Python 扫描。

## 下一步

仅补充获批、版本化的 `accession→Entrez`（字段 `accession,entrez_id`）交叉表，再重跑同一审计与特征构建；不得使用药名近似、相似结构、gene symbol 猜测、药效标签或模型结果。

完整源文件 SHA256、SQL 统计、schema 和前 20 条样例见 `identity_mapping_audit.json`。
"""
    (OUT / "data_rescue_report.md").write_text(md, encoding="utf-8")
    print(json.dumps({"status": report["status"], "unique_pert_ids": len(unique_pert), "drug_matches": len(drug_ids), "target_rows": len(targets)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
