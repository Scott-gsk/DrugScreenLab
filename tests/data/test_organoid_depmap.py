"""EXP-026 依赖图准入审计纯函数单元测试。"""

from __future__ import annotations

import pandas as pd

from drug_screen.data.organoid_depmap import gene_space_report, resolve_dual_library, tissue_composition


def _dep_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_ID": ["A", "A", "A", "A", "B", "B", "C"],
            "gene": ["G1", "G1", "G2", "G2", "G1", "G2", "G1"],
            "library": ["minLib", "v1_1", "minLib", "v1_1", "v1_1", "v1_1", "minLib"],
            "LFC": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        }
    )


def test_resolve_dual_library_prefers_declared_library() -> None:
    resolved, audit = resolve_dual_library(_dep_frame(), prefer="minLib")
    # 模型 A 双文库 -> 只留 minLib；B 仅 v1_1 原样；C 仅 minLib 原样。
    assert resolved["sample_ID"].tolist() == ["A", "A", "B", "B", "C"]
    assert resolved.loc[resolved["sample_ID"] == "A", "LFC"].tolist() == [0.1, 0.3]
    assert audit["n_models_multi_library"] == 1
    assert audit["multi_library_models"] == ["A"]
    assert audit["n_rows_before"] == 7 and audit["n_rows_after"] == 5
    assert audit["library_model_counts"] == {"minLib": 2, "v1_1": 2}


def test_resolve_dual_library_keeps_single_library_models_unchanged() -> None:
    frame = _dep_frame()
    resolved, audit = resolve_dual_library(frame, prefer="minLib")
    # B/C 模型行原样保留（数值未改）。
    b_rows = resolved[resolved["sample_ID"] == "B"]
    assert b_rows["LFC"].tolist() == [0.5, 0.6]
    assert audit["n_models_total"] == 3


def test_gene_space_report_intersections_and_no_downgrade() -> None:
    report = gene_space_report(
        expr_columns=["G1_exp", "G2_exp", "G3_exp"],
        dep_genes=["G1", "G4"],
        project_genes={"G1", "G2", "G5"},
        downgrade_threshold=2,
    )
    assert report["n_expression_genes"] == 3
    assert report["n_dependency_genes"] == 2
    assert report["n_expression_project_intersection"] == 2  # G1, G2
    assert report["n_dependency_project_intersection"] == 1  # G1
    assert report["n_expression_dependency_intersection"] == 1  # G1
    assert report["domain_adaptation_downgraded"] is False


def test_gene_space_report_downgrade_triggers_below_threshold() -> None:
    report = gene_space_report(
        expr_columns=["G1_exp", "G9_exp"],
        dep_genes=["G1"],
        project_genes={"G1", "G2", "G3"},
        downgrade_threshold=2,
    )
    assert report["n_expression_project_intersection"] == 1
    assert report["domain_adaptation_downgraded"] is True


def test_tissue_composition_counts_individuals() -> None:
    samples = pd.DataFrame(
        {
            "sample_ID": ["S1", "S2", "S3", "S4"],
            "individual_ID": [1, 1, 2, 3],
            "primary_tumour_type": ["Colorectal", "Colorectal", "Ovarian", "Ovarian"],
        }
    )
    composition = tissue_composition(samples)
    assert composition["n_models"] == 4
    assert composition["n_individuals"] == 3
    assert composition["tissue_counts"] == {"Colorectal": 2, "Ovarian": 2}
    assert composition["individuals_with_multiple_models"] == 1
    assert composition["max_models_per_individual"] == 2
