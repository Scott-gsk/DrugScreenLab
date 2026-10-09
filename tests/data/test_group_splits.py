"""新方案患者和细胞身份隔离、标签量档位嵌套。"""

from pathlib import Path

from drug_screen.data.group_splits import nested_identity_folds
from scripts.data.freeze_dual_route_splits import build


ROOT = Path(__file__).resolve().parents[2]


def test_identity_folds_are_disjoint_nested_and_deterministic():
    ids = [f"p{i:02d}" for i in range(51)]
    folds = nested_identity_folds(ids, seed=20260922)
    assert folds == nested_identity_folds(list(reversed(ids)), seed=20260922)
    assert sorted(p for fold in folds for p in fold["outer_test_ids"]) == ids
    for fold in folds:
        train, test = set(fold["outer_train_ids"]), set(fold["outer_test_ids"])
        assert not train & test
        levels = [set(level["train_ids"]) for level in fold["label_fractions"]]
        assert levels[0] < levels[1] < levels[2] == train
        for level in fold["label_fractions"]:
            chosen = set(level["train_ids"])
            for inner in level["inner_folds"]:
                itr = set(inner["train_ids"])
                val = set(inner["validation_ids"])
                assert itr | val == chosen
                assert not itr & val
                assert not (itr | val) & test


def test_actual_admission_folds_keep_all_pdo_samples_with_patient():
    report = build(
        ROOT / "data/processed/identity/kryzei_admission_audit_20260921.json",
        ROOT / "data/processed/transfer/mcrc_v3/samples.parquet",
        ROOT / "data/processed/exp012/gdsc2_cell_line_samples.parquet",
    )
    assert report["n_target_patients"] == 51
    assert report["n_target_samples"] == 60
    assert report["n_source_cell_lines"] == 963
    for fold in report["target_folds"]:
        assert not set(fold["outer_train_samples"]) & set(fold["outer_test_samples"])
    assert len({x for fold in report["source_folds"]
                for x in fold["outer_test_ids"]}) == 963
