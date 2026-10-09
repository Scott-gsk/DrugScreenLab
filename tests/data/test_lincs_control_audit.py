"""同板真实对照匹配的最小合同测试。"""

from pathlib import Path

import pandas as pd

from scripts.data.audit_lincs_continuous_controls import audit


def test_control_must_share_plate_cell_and_time(tmp_path: Path):
    rows = [
        ("c1", "p1", "ctl_vehicle", "DMSO", "A", "0.1", "%", "24", "h"),
        ("c2", "p1", "ctl_vehicle", "DMSO", "B", "0.1", "%", "24", "h"),
        ("t1", "p1", "trt_cp", "drug1", "A", "10", "um", "24", "h"),
        ("t2", "p1", "trt_cp", "drug2", "A", "10", "um", "6", "h"),
        ("t3", "p2", "trt_cp", "drug3", "B", "10", "um", "24", "h"),
    ]
    path = tmp_path / "inst_info.tsv"
    pd.DataFrame(rows, columns=[
        "inst_id", "rna_plate", "pert_type", "pert_id", "cell_id",
        "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit",
    ]).to_csv(path, sep="\t", index=False)
    report = audit(path)
    assert report["n_treatment_instances"] == 3
    assert report["n_treatment_with_candidate_control"] == 1
    assert report["n_treatment_without_candidate_control"] == 2
    assert report["fixed_24h_10um"] == {
        "n_treatment_instances": 2,
        "n_with_candidate_control": 1,
        "n_cells_with_candidate_control": 1,
        "n_compounds_with_candidate_control": 1,
    }
