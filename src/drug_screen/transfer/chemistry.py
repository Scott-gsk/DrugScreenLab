"""Conservative chemical identity normalization for transfer features.

The normalizer deliberately accepts salts only when every extra component is a
known, simple counterion and exactly one structural component remains.  It is
not a general mixture resolver: ambiguous inputs are quarantined for review.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


# These are deliberately small inorganic ions, rather than an expansive list of
# solvents or fragments that could silently change the compound identity.
_SAFE_COUNTERION_SMILES = frozenset(
    Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True, isomericSmiles=True)
    for smiles in (
        "[Cl-]", "[Br-]", "[I-]", "[F-]", "[Na+]", "[K+]", "[Li+]",
        "[Ca+2]", "[Mg+2]", "[Zn+2]", "[H+]", "[OH-]", "[O-]S(=O)(=O)[O-]",
    )
)


@dataclass(frozen=True)
class ChemicalIdentity:
    """Result of a non-destructive SMILES normalization attempt."""

    original_smiles: str | None
    canonical_isomeric_smiles: str | None
    inchikey: str | None
    status: str
    message: str | None = None
    removed_components: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.status in {"accepted", "counterion_removed"}


def _quarantine(
    smiles: object, status: str, message: str, *, removed_components: tuple[str, ...] = ()
) -> ChemicalIdentity:
    return ChemicalIdentity(
        original_smiles=smiles if isinstance(smiles, str) else None,
        canonical_isomeric_smiles=None,
        inchikey=None,
        status=status,
        message=message,
        removed_components=removed_components,
    )


def standardize_smiles(smiles: str | None) -> ChemicalIdentity:
    """Return an identity record, never silently selecting from a mixture.

    The original input is retained verbatim.  Canonical output is isomeric, so
    stereochemical labels participate in both the SMILES and Morgan features.
    """

    if not isinstance(smiles, str) or not smiles.strip():
        return _quarantine(smiles, "invalid", "SMILES must be a non-empty string")
    original = smiles
    parts = smiles.strip().split(".")
    if any(not part for part in parts):
        return _quarantine(original, "invalid", "SMILES contains an empty component")
    molecules = []
    for part in parts:
        molecule = Chem.MolFromSmiles(part)
        if molecule is None:
            return _quarantine(original, "invalid", "SMILES could not be parsed")
        molecules.append(molecule)

    retained = []
    removed_components: list[str] = []
    for part, molecule in zip(parts, molecules, strict=True):
        canonical_part = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        if canonical_part in _SAFE_COUNTERION_SMILES:
            removed_components.append(canonical_part)
        else:
            retained.append(molecule)
    if len(parts) > 1 and (len(retained) != 1 or len(removed_components) + len(retained) != len(parts)):
        return _quarantine(original, "mixture", "multiple non-counterion components")

    if not retained:
        return _quarantine(
            original, "counterion_only", "counterions without a structural component",
            removed_components=tuple(removed_components),
        )
    molecule = retained[0]
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    return ChemicalIdentity(
        original_smiles=original,
        canonical_isomeric_smiles=canonical,
        inchikey=Chem.MolToInchiKey(molecule),
        status="counterion_removed" if removed_components else "accepted",
        removed_components=tuple(removed_components),
    )


def morgan_fingerprint(
    identity: ChemicalIdentity | str,
    *,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Return a chirality-aware Morgan bit vector for an accepted identity."""

    if isinstance(identity, str):
        identity = standardize_smiles(identity)
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a non-negative integer")
    if isinstance(n_bits, bool) or not isinstance(n_bits, int) or n_bits < 1:
        raise ValueError("n_bits must be a positive integer")
    if not identity.accepted or identity.canonical_isomeric_smiles is None:
        raise ValueError(f"cannot fingerprint quarantined chemical: {identity.status}")
    molecule = Chem.MolFromSmiles(identity.canonical_isomeric_smiles)
    if molecule is None:  # defensive: output of standardize_smiles should parse
        raise ValueError("canonical chemical identity could not be parsed")
    bit_vector = AllChem.GetMorganFingerprintAsBitVect(
        molecule, radius, nBits=n_bits, useChirality=True
    )
    values = np.zeros((n_bits,), dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(bit_vector, values)
    return values
