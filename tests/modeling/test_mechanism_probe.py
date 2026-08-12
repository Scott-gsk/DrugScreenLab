from pathlib import Path

import pandas as pd

from drug_screen.modeling.mechanism_fast import build_frozen_candidate_mechanism_probe


def test_candidate_probe_is_explicitly_non_tuning(tmp_path: Path) -> None:
    perturbagens = tmp_path / "pert.tsv"
    pd.DataFrame({"pert_id": ["A"], "inchi_key": ["-666"]}).to_csv(perturbagens, sep="\t", index=False)
    audit = tmp_path / "prism.json"
    audit.write_text(
        '{"identity":{"candidates":[{"pert_id":"A","drug_name":"A"}]}}\n',
        encoding="utf-8",
    )
    result = build_frozen_candidate_mechanism_probe(
        perturbagen_path=perturbagens,
        prism_audit_path=audit,
        output_dir=tmp_path / "out",
    )
    assert result["decision"]["labels_used_for_tuning"] is False
    assert result["decision"]["model_increment"] == "NOT_RUN_COVERAGE_INSUFFICIENT"
