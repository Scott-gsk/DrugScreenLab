import numpy as np
import pytest

from drug_screen.transfer.chemistry import morgan_fingerprint, standardize_smiles


def test_standardization_keeps_isomeric_identity_and_full_inchikey():
    identity = standardize_smiles("C[C@H](O)F")
    assert identity.accepted
    assert "@" in identity.canonical_isomeric_smiles
    assert identity.inchikey is not None and len(identity.inchikey) == 27


def test_safe_single_counterion_is_removed_but_mixtures_are_quarantined():
    salt = standardize_smiles("CCO.[Cl-]")
    assert salt.status == "counterion_removed"
    assert salt.canonical_isomeric_smiles == "CCO"
    assert salt.removed_components == ("[Cl-]",)
    mixture = standardize_smiles("CCO.CCN")
    assert mixture.status == "mixture"
    assert mixture.canonical_isomeric_smiles is None


def test_isolated_counterion_is_quarantined_instead_of_accepted():
    ion = standardize_smiles("[Na+]")
    assert ion.status == "counterion_only"
    assert not ion.accepted
    assert ion.removed_components == ("[Na+]",)


def test_invalid_and_quarantined_structures_cannot_be_fingerprinted():
    assert standardize_smiles("not a smiles").status == "invalid"
    with pytest.raises(ValueError, match="quarantined"):
        morgan_fingerprint("CCO.CCN")


def test_morgan_features_are_2048_bit_and_chirality_aware():
    left = morgan_fingerprint("C[C@H](O)F")
    right = morgan_fingerprint("C[C@@H](O)F")
    assert left.shape == (2048,)
    assert left.dtype == np.uint8
    assert not np.array_equal(left, right)
