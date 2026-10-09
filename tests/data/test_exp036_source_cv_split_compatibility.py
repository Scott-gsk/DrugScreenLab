import json

from scripts.data.audit_exp036_source_cv_split_compatibility import audit_compatibility


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compatibility_requires_every_current_test_identity_to_be_untouched(tmp_path):
    identities = [f"cell_{index}" for index in range(10)]
    current_folds = []
    for fold in range(5):
        test = identities[fold * 2 : fold * 2 + 2]
        train = [identity for identity in identities if identity not in test]
        current_folds.append({"fold": fold, "outer_train_ids": train, "outer_test_ids": test})
    split_path = tmp_path / "split.json"
    _write_json(split_path, {"source_folds": current_folds})

    legacy_root = tmp_path / "legacy"
    for fold in range(5):
        # Each old checkpoint is clean for exactly the matching current fold.
        test = identities[fold * 2 : fold * 2 + 2]
        train = [identity for identity in identities if identity not in test]
        assignments = {identity: "train" for identity in train}
        assignments.update({identity: "test" for identity in test})
        _write_json(
            legacy_root / f"fold{fold}" / f"fold{fold}_source_indices.json",
            {"group_assignments": assignments},
        )

    result = audit_compatibility(split_path, legacy_root)
    assert result["all_folds_have_strict_legacy_checkpoint"] is True
    assert result["strictly_compatible_legacy_checkpoints_by_current_outer_fold"] == {
        str(fold): [fold] for fold in range(5)
    }


def test_validation_overlap_also_disqualifies_a_legacy_checkpoint(tmp_path):
    identities = [f"cell_{index}" for index in range(10)]
    current_folds = []
    for fold in range(5):
        test = identities[fold * 2 : fold * 2 + 2]
        train = [identity for identity in identities if identity not in test]
        current_folds.append({"fold": fold, "outer_train_ids": train, "outer_test_ids": test})
    split_path = tmp_path / "split.json"
    _write_json(split_path, {"source_folds": current_folds})

    legacy_root = tmp_path / "legacy"
    for fold in range(5):
        assignments = {identity: "train" for identity in identities}
        assignments[identities[fold * 2]] = "test"
        assignments[identities[fold * 2 + 1]] = "validation"
        _write_json(
            legacy_root / f"fold{fold}" / f"fold{fold}_source_indices.json",
            {"group_assignments": assignments},
        )

    result = audit_compatibility(split_path, legacy_root)
    assert result["all_folds_have_strict_legacy_checkpoint"] is False
    assert result["strictly_compatible_legacy_checkpoints_by_current_outer_fold"] == {
        str(fold): [] for fold in range(5)
    }
    matching = next(
        row for row in result["fold_comparisons"]
        if row["current_outer_fold"] == 0 and row["legacy_checkpoint_fold"] == 0
    )
    assert matching["overlap_with_legacy_validation"] == 1
    assert matching["strictly_untouched_for_legacy_checkpoint"] is False


def test_audit_rejects_legacy_identity_universe_mismatch(tmp_path):
    identities = [f"cell_{index}" for index in range(10)]
    current_folds = []
    for fold in range(5):
        test = identities[fold * 2 : fold * 2 + 2]
        train = [identity for identity in identities if identity not in test]
        current_folds.append({"fold": fold, "outer_train_ids": train, "outer_test_ids": test})
    split_path = tmp_path / "split.json"
    _write_json(split_path, {"source_folds": current_folds})

    legacy_root = tmp_path / "legacy"
    for fold in range(5):
        assignments = {identity: "train" for identity in identities[:-1]}
        assignments["foreign_cell"] = "test"
        _write_json(
            legacy_root / f"fold{fold}" / f"fold{fold}_source_indices.json",
            {"group_assignments": assignments},
        )

    try:
        audit_compatibility(split_path, legacy_root)
    except ValueError as error:
        assert "identity universe differs" in str(error)
    else:
        raise AssertionError("identity universe mismatch must fail closed")
