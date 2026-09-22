"""Work-package B data-side tests (B5 items: whitelist, split isolation, patient-unit permutation, train-fitted preprocessing)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from drug_screen.data.admission import AdmissionWhitelist, patient_unit_permutation
from drug_screen.data.trainfitted import FitOnTrainTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_admission_rejects_non_admitted_sample_of_admitted_patient():
    """B5 #1: a non-admitted sample is rejected even when its patient is admitted."""
    engine = AdmissionWhitelist(
        patient_whitelist=["Pt1", "Pt2"],
        sample_whitelist=["Pt1_t1", "Pt2_t1"],
        patient_of_sample={"Pt1_t1": "Pt1", "Pt1_t2": "Pt1", "Pt2_t1": "Pt2", "Pt3_t1": "Pt3"},
    )
    # Pt1 is admitted, but Pt1_t2 is not in the sample whitelist -> reject.
    verdict = engine.adjudicate("Pt1_t2", patient_id="Pt1")
    assert verdict.admitted is False
    assert verdict.reason == "non_admitted_sample_of_admitted_patient"

    admitted, rejected = engine.reject_non_admitted(
        ["Pt1_t1", "Pt1_t2", "Pt3_t1"],
        patient_of_sample={"Pt1_t1": "Pt1", "Pt1_t2": "Pt1", "Pt3_t1": "Pt3"},
    )
    assert admitted == ["Pt1_t1"]
    assert [v.reason for v in rejected] == ["non_admitted_sample_of_admitted_patient", "non_admitted_patient"]


def test_admission_admits_whitelisted_sample_and_flags_non_patient():
    engine = AdmissionWhitelist(["P"], ["S1"], patient_of_sample={"S1": "P", "S2": "P", "SX": "?"})
    assert engine.adjudicate("S1").admitted is True
    assert engine.adjudicate("SX").reason == "unknown_patient_mapping"


def test_shared_splits_no_patient_cross_fold():
    """B5 #2: every whitelist patient appears in exactly one outer test fold."""
    path = PROJECT_ROOT / "data" / "splits" / "kryzei_shared_splits_v1.json"
    assert path.is_file(), f"missing {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["split_unit"] == "patient_id"
    outer = data["outer_folds"]

    test_fold_of: dict[str, int] = {}
    for fold in outer:
        train = set(fold["train_patient_ids"])
        test = set(fold["test_patient_ids"])
        assert not (train & test), f"fold {fold['fold']} train/test patient overlap"
        for pid in test:
            assert pid not in test_fold_of, f"patient {pid} in two test folds"
            test_fold_of[pid] = fold["fold"]
        for inner in fold["inner_folds"]:
            itr = set(inner["inner_train_patient_ids"])
            iva = set(inner["inner_val_patient_ids"])
            assert iva <= train, f"fold {fold['fold']} inner val escapes outer train"
            assert not (itr & iva), f"fold {fold['fold']} inner train/val overlap"

    # every whitelist patient is held out in exactly one outer test fold
    all_patients = sorted({p for fold in outer for p in fold["train_patient_ids"] + fold["test_patient_ids"]})
    assert len(all_patients) == data["n_patients"]
    assert sorted(test_fold_of) == all_patients

    # inner-val union within a fold == that fold's outer train patients
    for fold in outer:
        inner_vals = [p for inner in fold["inner_folds"] for p in inner["inner_val_patient_ids"]]
        assert sorted(set(inner_vals)) == sorted(fold["train_patient_ids"]), \
            f"fold {fold['fold']} inner-val union != outer train patients"


def test_patient_unit_permutation_consistent_within_patient():
    """B5 #4: whole-row expression permutation is patient-unit, so all drug records
    of a patient stay consistent under the permutation."""
    patient_ids = ["P1", "P1", "P1", "P2", "P2", "P3", "P3", "P3", "P3"]
    rng = np.random.default_rng(20260921)
    perm = patient_unit_permutation(patient_ids, rng=rng)
    assert len(perm) == len(patient_ids)
    assert set(perm.tolist()) == set(range(len(patient_ids)))  # valid permutation

    permuted = np.asarray(patient_ids)[perm]
    assert Counter(permuted.tolist()) == Counter(patient_ids)  # counts preserved

    # contiguity: each patient's rows form one contiguous block
    positions: dict[str, list[int]] = {}
    for i, p in enumerate(permuted.tolist()):
        positions.setdefault(p, []).append(i)
    for p, idx in positions.items():
        assert idx == list(range(idx[0], idx[0] + len(idx))), f"patient {p} rows not contiguous"

    # a patient's permuted rows keep their original relative order (whole-block move)
    original_order = {p: [i for i, x in enumerate(patient_ids) if x == p] for p in set(patient_ids)}
    for p, idx in positions.items():
        assert perm[idx].tolist() == original_order[p]


def test_fitontrain_missing_gene_maps_to_train_mean():
    """B5 #7a: a missing gene standardizes to z=0 (training mean), not an arbitrary value."""
    Xtr = np.array([
        [1.0, 2.0, 3.0],
        [3.0, np.nan, 5.0],
        [5.0, 4.0, 1.0],
    ])
    tr = FitOnTrainTransformer().fit(Xtr)
    Zte = tr.transform(np.array([[np.nan, np.nan, np.nan]], dtype=float))
    assert np.allclose(Zte[0], [0.0, 0.0, 0.0], atol=1e-12)

    # the fitted gene means are derived only from finite training values
    assert np.allclose(tr.mean_, [3.0, 3.0, 3.0], atol=1e-12)


def test_fitontrain_transform_does_not_read_test_rows():
    """B5 #7b: transform must reuse fitted train statistics and never mutate state."""
    rng = np.random.default_rng(20260921)
    Xtr = rng.normal(loc=0.0, scale=1.0, size=(200, 5))
    tr = FitOnTrainTransformer().fit(Xtr, folds=[0, 1, 2, 3])
    mean_before = tr.mean_.copy()
    std_before = tr.std_.copy()
    assert tr.fit_folds == frozenset({"0", "1", "2", "3"})

    Xte = rng.normal(loc=1000.0, scale=50.0, size=(20, 5))
    Z1 = tr.transform(Xte)
    Z2 = tr.transform(Xte)
    assert np.allclose(Z1, Z2)                       # idempotent
    assert np.array_equal(tr.mean_, mean_before)     # state unchanged by transform
    assert np.array_equal(tr.std_, std_before)

    # test rows use TRAIN statistics (mean ~0, std ~1); a test block with mean
    # 1000 therefore maps to ~1000, NOT to 0 (which would imply the scaler
    # re-fitted itself on the test rows).
    assert np.all(np.abs(Z1.mean(axis=0) - 1000.0) < 250.0)


