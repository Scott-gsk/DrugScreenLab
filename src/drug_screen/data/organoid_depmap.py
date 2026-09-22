"""EXP-026 依赖图准入审计逻辑（纯函数，供 scripts/data/acquire_organoid_depmap.py 调用）。

预登记规则（先于任何依赖图实验写入，EXP-026 任务 6）：
1. 双文库规则：依赖数据行单位 = 模型 × 文库对；同一模型同时存在 minLib
   与 v1_1 筛选时优先保留 minLib（避免文库选择自由度）；任何下游划分按
   individual_ID 整组隔离，双文库不构成患者级泄漏。
2. 基因空间降级条款：RNA-seq 基因空间与项目主基因空间（GDSC∩Kryzei 交集）
   交集 < 5000 时，域适应用途降级为「仅依赖标签用途」并显式声明。
3. 声明边界：CRISPR 依赖 ≠ 药物响应；本资源不作药敏标签，只用于迁移检验、
   域适应与辅助任务。
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np
import pandas as pd

PREFERRED_LIBRARY = "minLib"
DOWNGRADE_THRESHOLD = 5000
_EXPR_SUFFIX = "_exp"


def resolve_dual_library(
    dep: pd.DataFrame,
    prefer: str = PREFERRED_LIBRARY,
) -> tuple[pd.DataFrame, dict]:
    """按预登记双文库规则去重：同模型多文库时只保留 ``prefer`` 文库行。

    单文库模型原样保留；不改动任何数值。
    """
    libraries_per_model = dep.groupby("sample_ID")["library"].nunique()
    multi_library_models = sorted(libraries_per_model[libraries_per_model > 1].index.tolist())
    keep = ~dep["sample_ID"].isin(multi_library_models) | (dep["library"] == prefer)
    resolved = dep.loc[keep].reset_index(drop=True)
    audit = {
        "rule": f"one library per model; models with multiple libraries keep '{prefer}'",
        "n_models_total": int(dep["sample_ID"].nunique()),
        "n_models_multi_library": len(multi_library_models),
        "multi_library_models": multi_library_models,
        "n_rows_before": int(len(dep)),
        "n_rows_after": int(len(resolved)),
        "n_rows_dropped": int(len(dep) - len(resolved)),
        "library_model_counts": {
            str(lib): int(count)
            for lib, count in dep.groupby("library")["sample_ID"].nunique().items()
        },
    }
    return resolved, audit


def gene_space_report(
    expr_columns: Sequence[str],
    dep_genes: Sequence[str],
    project_genes: Collection[str],
    downgrade_threshold: int = DOWNGRADE_THRESHOLD,
) -> dict:
    """RNA-seq / 依赖基因空间与项目主基因空间的交集与降级判定。"""
    expr_genes = {str(column)[: -len(_EXPR_SUFFIX)] for column in expr_columns}
    project = {str(gene) for gene in project_genes}
    dep_set = {str(gene) for gene in dep_genes}
    expr_project = expr_genes & project
    dep_project = dep_set & project
    return {
        "n_expression_genes": len(expr_genes),
        "n_dependency_genes": len(dep_set),
        "n_project_genes": len(project),
        "n_expression_project_intersection": len(expr_project),
        "n_dependency_project_intersection": len(dep_project),
        "n_expression_dependency_intersection": len(expr_genes & dep_set),
        "downgrade_threshold": downgrade_threshold,
        "domain_adaptation_downgraded": len(expr_project) < downgrade_threshold,
    }


def tissue_composition(samples: pd.DataFrame) -> dict:
    """癌种构成 + individual_ID 模型数分布（划分隔离单元审计）。"""
    per_individual = samples.groupby("individual_ID")["sample_ID"].nunique()
    return {
        "n_models": int(len(samples)),
        "n_individuals": int(samples["individual_ID"].nunique()),
        "tissue_counts": {
            str(tissue): int(count)
            for tissue, count in samples["primary_tumour_type"].value_counts().items()
        },
        "individuals_with_multiple_models": int((per_individual > 1).sum()),
        "max_models_per_individual": int(per_individual.max()),
    }


def load_kryzei_genes(npz_path) -> list[str]:
    """从 mcrc_v3 indexed.npz 读取有序基因清单。"""
    with np.load(npz_path) as data:
        return [str(gene) for gene in data["ordered_genes"]]


def load_project_genes(txt_path) -> set[str]:
    """从 GDSC∩Kryzei 交集清单读取项目主基因空间。"""
    with open(txt_path, encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}
