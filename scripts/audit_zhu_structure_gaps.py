"""Read-only catalogue-number audit for Zhu drugs excluded from zhu_v2."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rdkit import Chem

from drug_screen.transfer.chemistry import standardize_smiles


def parent_candidate(smiles: object) -> tuple[str | None, str | None]:
    """Report, but never admit, the sole carbon-containing salt parent."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None
    molecules = [Chem.MolFromSmiles(part) for part in smiles.split(".")]
    if any(molecule is None for molecule in molecules):
        return None, None
    organic = [molecule for molecule in molecules if sum(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms())]
    if len(organic) != 1:
        return None, None
    identity = standardize_smiles(Chem.MolToSmiles(organic[0], canonical=True, isomericSmiles=True))
    return identity.inchikey, identity.canonical_isomeric_smiles


def main() -> None:
    root = Path("data")
    raw = root / "external/pdo_public_candidates/zhu2024_hepatobiliary_64/mmc5.xlsx"
    target = pd.read_excel(raw, sheet_name="Further drug screen_Drug list", header=1)
    chemicals = pd.read_excel(raw, sheet_name="Chemical Data-301cpds")
    zhu = pd.read_parquet(root / "processed/transfer/zhu_v2/drugs.parquet")
    gdsc = pd.read_parquet(root / "processed/exp014/gdsc_v2/drugs.parquet")
    existing = set(zhu.source_drug_id)
    gdsc_keys = set(gdsc.inchikey.dropna().astype(str))
    gdsc_connectivity = {key[:14] for key in gdsc_keys}
    result = []
    for row in target.itertuples(index=False):
        source_id = f"ZHU:{int(row.Number):02d}"
        if source_id in existing:
            continue
        catalogue = str(row.Catalogue).strip()
        candidates = chemicals.loc[chemicals.cat.astype(str).str.strip().eq(catalogue)]
        structures = []
        for candidate in candidates.itertuples(index=False):
            identity = standardize_smiles(getattr(candidate, "SMILES"))
            parent_key, parent_smiles = parent_candidate(getattr(candidate, "SMILES"))
            structures.append({"author_chemical_name": getattr(candidate, "name"), "status": identity.status,
                               "inchikey": identity.inchikey, "smiles": identity.canonical_isomeric_smiles,
                               "message": identity.message, "parent_candidate_inchikey": parent_key,
                               "parent_candidate_smiles": parent_smiles})
        keys = {x["inchikey"] for x in structures if x["inchikey"]}
        result.append({"source_drug_id": source_id, "author_drug": row.Drug, "catalogue": catalogue,
                       "catalogue_matches": len(candidates), "structures": structures,
                       "unambiguous_accepted_key": next(iter(keys)) if len(keys) == 1 else None,
                       "gdsc_exact": bool(keys & gdsc_keys),
                       "gdsc_connectivity_only": bool({key[:14] for key in keys} & gdsc_connectivity),
                       "parent_candidate_gdsc_exact": bool({x["parent_candidate_inchikey"] for x in structures if x["parent_candidate_inchikey"]} & gdsc_keys),
                       "parent_candidate_gdsc_connectivity_only": bool({x["parent_candidate_inchikey"][:14] for x in structures if x["parent_candidate_inchikey"]} & gdsc_connectivity),
                       "admission": "candidate_catalogue_evidence_only" if len(keys) == 1 else "exclude_pending_structure_resolution"})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
