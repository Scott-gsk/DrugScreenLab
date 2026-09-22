"""EXP-026 任务 6：肿瘤类器官依赖图获取准入审计（只读，不训练不评测）。

资源：Herranz-Ors 等 Nature 2026（10.1038/s41586-026-10830-y），
Figshare 28339340 处理后数据 + Nature 补充材料 MOESM3。
原始文件已于 2026-09-12/13 下载至 raw/transfer_public_assets/organoid_biobank/，
配对表由 EXP-015 物化至 processed/transfer/biobank_functional_v1/。

本脚本执行准入审计：
1. 原始与配对表 sha256 复验（对照 EXP-014/015 审计记录）；
2. individual_ID 双文库规则审计（16 个双文库模型，预登记 minLib 优先）；
3. 癌种构成表与 individual_ID 隔离单元审计；
4. RNA-seq/依赖基因空间与项目主基因空间（GDSC∩Kryzei 19091）交集
   及 <5000 降级条款判定；
5. 写出 admission 审计（ACQUIRED_EXPLORATORY；CRISPR 依赖 ≠ 药敏标签）。
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

from drug_screen.data.organoid_depmap import (
    gene_space_report,
    load_kryzei_genes,
    load_project_genes,
    resolve_dual_library,
    tissue_composition,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/transfer_public_assets/organoid_biobank"
PROCESSED = ROOT / "data/processed/transfer/biobank_functional_v1"
OUT = ROOT / "data/artifacts/EXP-026/organoid_depmap_admission.json"

EXPECTED_RAW_SHA256 = {
    "41586_2026_10830_MOESM3_ESM.zip": "7e83b7551c9c272de0bc0455ea9a0e7803a8043d15d9cfedf5a1c3e609a2a6e2",
    "data_RNAseq_ALL-organoids.csv": "92859e18417c3737956236e6990f8001a0d2cc352890b9355b7ed9603b6a6b1e",
    "data_covariates_ALL-organoids.csv": "e57603e0ae26780ef112cbdbf063818dd17837858353894ba29e75f6936e9f64",
}
EXPECTED_PARQUET_SHA256 = {
    "dependency_by_library.parquet": "bf40474b804ab3bb9029c551cd0f7a8604dbdcbead7011ba2e96894f17f20fb7",
    "expression_author_log2tpm.parquet": "9f4df2c082fbb0fd1d678661f09ba83e15468fe9460aa5c2ab7afb317b26c184",
    "samples.parquet": "2864dc5340d6bca73aea88ece6dfbad533cd100f245152915df1e200773892ac",
}
EXPECTED_PAIRING = {
    "paired_models": 162,
    "paired_individuals": 157,
    "dependency_rows": 2_948_036,
    "dependency_genes": 16_562,
    "multi_library_models": 16,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    checks: list[dict] = []
    ok = True

    def check(name: str, passed: bool, detail: dict) -> None:
        nonlocal ok
        ok = ok and passed
        checks.append({"check": name, "passed": bool(passed), **detail})

    # 1. sha256 复验
    for filename, expected in EXPECTED_RAW_SHA256.items():
        actual = sha256_file(RAW / filename)
        check(f"raw:{filename}", actual == expected, {"expected": expected, "actual": actual})
    for filename, expected in EXPECTED_PARQUET_SHA256.items():
        actual = sha256_file(PROCESSED / filename)
        check(f"parquet:{filename}", actual == expected, {"expected": expected, "actual": actual})

    # 2. 加载配对表
    samples = pd.read_parquet(PROCESSED / "samples.parquet")
    dep = pd.read_parquet(PROCESSED / "dependency_by_library.parquet")
    expr = pd.read_parquet(PROCESSED / "expression_author_log2tpm.parquet")

    check(
        "pairing_counts",
        len(samples) == EXPECTED_PAIRING["paired_models"]
        and samples["individual_ID"].nunique() == EXPECTED_PAIRING["paired_individuals"]
        and len(dep) == EXPECTED_PAIRING["dependency_rows"]
        and dep["gene"].nunique() == EXPECTED_PAIRING["dependency_genes"]
        and expr.shape == (EXPECTED_PAIRING["paired_models"], 15_764),
        {
            "n_models": len(samples),
            "n_individuals": int(samples["individual_ID"].nunique()),
            "n_dependency_rows": len(dep),
            "n_dependency_genes": int(dep["gene"].nunique()),
            "expression_shape": list(expr.shape),
        },
    )

    # 3. 双文库规则审计
    resolved, dual_audit = resolve_dual_library(dep)
    check(
        "dual_library_rule",
        dual_audit["n_models_multi_library"] == EXPECTED_PAIRING["multi_library_models"]
        and resolved["sample_ID"].nunique() == dual_audit["n_models_total"],
        {"audit": dual_audit},
    )

    # 4. 癌种构成与隔离单元
    tissues = tissue_composition(samples)
    check(
        "tissue_composition",
        tissues["n_models"] == 162 and tissues["n_individuals"] == 157,
        {"composition": tissues},
    )

    # 5. 基因空间交集与降级条款
    kryzei_genes = load_kryzei_genes(ROOT / "data/processed/transfer/mcrc_v3/indexed.npz")
    project_genes = load_project_genes(
        ROOT / "data/processed/identity/transfer/gdsc_kryeziu_gene_intersection_ordered_DRAFT.txt"
    )
    genes = gene_space_report(
        expr_columns=list(expr.columns),
        dep_genes=dep["gene"].drop_duplicates().tolist(),
        project_genes=project_genes,
    )
    genes["n_kryzei_genes"] = len(kryzei_genes)
    check(
        "gene_space_above_downgrade_threshold",
        not genes["domain_adaptation_downgraded"],
        {"gene_space": genes},
    )

    # 6. 写出 admission
    admission = {
        "id": "organoid_depmap_admission_exp026",
        "admission_version": 1,
        "date": str(date.today()),
        "resource": {
            "paper": "Herranz-Ors et al., Nature 2026, 10.1038/s41586-026-10830-y",
            "figshare": "28339340.v1",
            "raw_path": "raw/transfer_public_assets/organoid_biobank",
            "processed_path": "processed/transfer/biobank_functional_v1",
            "prior_audits": [
                "artifacts/EXP-014/biobank_audit.json",
                "artifacts/EXP-015/nature_dependency_pairing.json",
            ],
        },
        "checks": checks,
        "dual_library_audit": dual_audit,
        "tissue_composition": tissues,
        "gene_space": genes,
        "status": "PASS" if ok else "FAIL",
        "admission": {
            "status": "ACQUIRED_EXPLORATORY",
            "decided_in": "EXP-026",
            "authorized_use": [
                "migration testing (cell-line CRISPR -> organoid dependency transfer)",
                "domain adaptation corpora (EXP-029/030)",
                "auxiliary pretraining tasks (EXP-031)",
            ],
            "not_authorized": [
                "drug-response labels or drug-sensitivity endpoints",
                "any claim that CRISPR dependency equals drug response",
                "target-domain (Kryzei PDO) evaluation",
            ],
            "split_rule": "individual_ID whole-group isolation in all downstream splits",
            "downgrade_clause": (
                f"if expression-project gene intersection < {genes['downgrade_threshold']}, "
                "domain-adaptation use is downgraded to dependency-label-only use"
            ),
            "domain_adaptation_downgraded": genes["domain_adaptation_downgraded"],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(admission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"status={admission['status']} checks={len(checks)} failed={sum(1 for c in checks if not c['passed'])}")
    print(f"gene intersections: expr∩project={genes['n_expression_project_intersection']} dep∩project={genes['n_dependency_project_intersection']} downgrade={genes['domain_adaptation_downgraded']}")
    print(f"dual-library: {dual_audit['n_models_multi_library']} models, rows {dual_audit['n_rows_before']}->{dual_audit['n_rows_after']}")
    for item in checks:
        if not item["passed"]:
            print("FAILED:", json.dumps(item, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
