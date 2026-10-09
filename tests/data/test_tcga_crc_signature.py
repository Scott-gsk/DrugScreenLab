"""GDC 文件身份连接与独立多重检验复算的基本约束。"""

import numpy as np

from scripts.data.audit_tcga_crc_star_availability import resolve_sample
from scripts.data.verify_tcga_crc_signature import bh


def test_file_aliquot_links_to_one_sample_not_every_sample_in_case() -> None:
    file = {
        "associated_entities": [{"entity_type": "aliquot", "entity_submitter_id": "T-aliquot"}],
        "cases": [{"submitter_id": "case-1", "project": {"project_id": "TCGA-COAD"},
                   "samples": [
                       {"submitter_id": "T", "sample_type": "Primary Tumor",
                        "portions": [{"analytes": [{"aliquots": [{"submitter_id": "T-aliquot"}]}]}]},
                       {"submitter_id": "N", "sample_type": "Solid Tissue Normal",
                        "portions": [{"analytes": [{"aliquots": [{"submitter_id": "N-aliquot"}]}]}]},
                   ]}],
    }
    result = resolve_sample(file)
    assert result["identity_status"] == "LINKED"
    assert result["sample_id"] == "T"
    assert result["sample_type"] == "Primary Tumor"


def test_manual_bh_corrects_over_complete_gene_family() -> None:
    np.testing.assert_allclose(bh(np.array([0.01, 0.04, 0.03])), [0.03, 0.04, 0.04])
