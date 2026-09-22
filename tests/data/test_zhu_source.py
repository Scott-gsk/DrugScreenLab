from pathlib import Path
import numpy as np
from drug_screen.data.zhu_source import rebuild_zhu

def test_zhu_author_tables_rebuild_exact_paired_pdo_bundle():
    raw = Path("data/external/pdo_public_candidates/zhu2024_hepatobiliary_64")
    rebuilt = rebuild_zhu(raw / "mmc4.xlsx", raw / "mmc5.xlsx")
    assert len(rebuilt.sample_ids) == len(rebuilt.samples) == 20
    assert rebuilt.pairs.groupby("sample_id").drug_id.nunique().eq(len(rebuilt.drugs)).all()
    assert rebuilt.pairs.higher_is_sensitive.all()
    assert set(rebuilt.pairs.response_name) == {"1_minus_auc"}
    assert np.allclose(rebuilt.pairs.response + rebuilt.pairs.source_auc, 1.0)
    assert np.isfinite(rebuilt.expression).all()
    assert rebuilt.audit["expression_transform"] == "none; author FPKM numeric values preserved after gene-label aggregation only"
    assert rebuilt.drugs.drug_id.str.len().eq(27).all()
    assert rebuilt.drugs.drug_id.str.contains("-").all()


def test_zhu_catalogue_expansion_is_prelabel_structure_coverage_only():
    raw = Path("data/external/pdo_public_candidates/zhu2024_hepatobiliary_64")
    rebuilt = rebuild_zhu(raw / "mmc4.xlsx", raw / "mmc5.xlsx", catalogue_expansion=True)
    assert len(rebuilt.drugs) > 33
    assert rebuilt.drugs.structure_linkage.isin({"exact_normalized_author_name", "exact_author_catalogue"}).all()
    assert rebuilt.pairs.source_drug_id.isin(rebuilt.drugs.source_drug_id).all()
