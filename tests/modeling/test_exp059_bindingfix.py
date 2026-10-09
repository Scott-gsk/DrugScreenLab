from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest

from scripts.modeling.exp051_contract_fit_smoke import (
    EXP059_CPU_RUN_ROOT_ALLOWED_ENTRIES,
    EXP059_TRANSIGEN_LR_STABILITY_RUN_ID,
    _assert_exp055_train_rows_match_frozen_archive,
    new_attempt_path,
    resolve_exp059_parent_evidence,
)


PARENT_RELATIVE = (
    "artifacts/experiments/EXP-055/20261007_exp055_transigen_cpu_norm_recovery_v1/"
    "fit_attempt_001/TranSiGen_last_finite.pt"
)
PARENT_SHA = "5555e4f0aa3a85c34d075c9211a494862ccb35b65c766b559919ab7f60bc2dfb"


def test_exp059_bindingfix_is_the_single_registered_run() -> None:
    assert EXP059_TRANSIGEN_LR_STABILITY_RUN_ID == "20261008_exp059_transigen_lr1e4_bindingfix_v2"
    assert {"exp059_fresh_preflight.json", "root_entry_preflight_readback.json",
            "repair_provenance"}.issubset(EXP059_CPU_RUN_ROOT_ALLOWED_ENTRIES)


def test_config_bound_parent_evidence_accepts_correct_and_rejects_wrong_sha() -> None:
    valid = {"authorization": {
        "mechanism_replay_checkpoint": PARENT_RELATIVE,
        "mechanism_replay_checkpoint_sha256": PARENT_SHA,
    }}
    path, digest = resolve_exp059_parent_evidence(valid)
    assert path.is_file()
    assert digest == PARENT_SHA

    invalid = {"authorization": {
        "mechanism_replay_checkpoint": PARENT_RELATIVE,
        "mechanism_replay_checkpoint_sha256": "0" * 64,
    }}
    with pytest.raises(RuntimeError, match="config-bound parent evidence SHA mismatch"):
        resolve_exp059_parent_evidence(invalid)


def test_new_attempt_directory_cannot_be_reopened_or_overwritten() -> None:
    with tempfile.TemporaryDirectory() as temp:
        run_dir = Path(temp) / EXP059_TRANSIGEN_LR_STABILITY_RUN_ID
        attempt = new_attempt_path(run_dir, "fit_attempt_001")
        assert not attempt.exists()
        attempt.mkdir(parents=True)
        sentinel = attempt / "immutable.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with pytest.raises((FileExistsError, ValueError)):
            new_attempt_path(run_dir, "fit_attempt_001")
        assert sentinel.read_text(encoding="utf-8") == "keep"


def test_postcheckpoint_train_comparison_keeps_gene_axis_outside_row_mask() -> None:
    mask = [True, False, True, False]
    frozen = {
        "source_row": [1, 2, 3, 4],
        "source_object_id": ["A", "B", "C", "D"],
        "full_inchikey": ["K1", "K1", "K2", "K2"],
        "pert_id": ["P1", "P1", "P2", "P2"],
        "genes": [f"G{i}" for i in range(978)],
        "X": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]],
        "X_ctl": [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
        "delta": [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    }
    train = {"source_row": [1, 3], "source_object_id": ["A", "C"],
        "full_inchikey": ["K1", "K2"], "pert_id": ["P1", "P2"],
        "genes": frozen["genes"], "X": [frozen["X"][0], frozen["X"][2]],
        "X_ctl": [frozen["X_ctl"][0], frozen["X_ctl"][2]],
        "delta": [frozen["delta"][0], frozen["delta"][2]]}
    frozen_arrays = {name: np.asarray(value) for name, value in frozen.items()}
    train_arrays = {name: np.asarray(value) for name, value in train.items()}
    _assert_exp055_train_rows_match_frozen_archive(frozen_arrays, train_arrays, np.asarray(mask))
