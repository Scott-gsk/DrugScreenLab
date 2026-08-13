"""Response-blind Broad PRISM ↔ LINCS chemical identity bridge."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd


_BROAD_ID = re.compile(r"^(BRD-[A-Z]\d{8})")


class PrismBridgeError(RuntimeError):
    """Raised when compound identity cannot be made unambiguous."""


def _valid(value: object) -> bool:
    return value is not None and str(value).strip() not in {"", "nan", "-666", "-666.0"}


def _base_broad_id(value: object) -> str | None:
    match = _BROAD_ID.match(str(value).strip())
    return match.group(1) if match else None


def _inchi_key(smiles: object) -> str | None:
    if not _valid(smiles):
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import inchi
    except ImportError as error:  # pragma: no cover - environment contract
        raise PrismBridgeError("RDKit is required for structure identity") from error
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    value = inchi.MolToInchiKey(molecule)
    return value if value else None


def _alias(value: object) -> str | None:
    if not _valid(value):
        return None
    normalized = unicodedata.normalize("NFKC", str(value)).lower()
    normalized = re.sub(r"[^a-z0-9]+", "", normalized)
    return normalized or None


def build_identity_bridge(
    prism_treatments: pd.DataFrame,
    lincs_pert_info: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Match unique Broad treatment identities to chemical LINCS identities.

    The function intentionally accepts only treatment and perturbagen identity
    metadata.  A response matrix or phenotype column is neither required nor
    inspected.
    """
    prism_required = {"broad_id", "name", "smiles"}
    lincs_required = {"pert_id", "pert_iname", "pert_type", "inchi_key", "canonical_smiles"}
    if missing := sorted(prism_required.difference(prism_treatments.columns)):
        raise PrismBridgeError(f"PRISM treatment metadata missing columns: {missing}")
    if missing := sorted(lincs_required.difference(lincs_pert_info.columns)):
        raise PrismBridgeError(f"LINCS perturbagen metadata missing columns: {missing}")

    prism = prism_treatments[["broad_id", "name", "smiles"]].drop_duplicates("broad_id").copy()
    prism["prism_broad_id_base"] = prism["broad_id"].map(_base_broad_id)
    prism["prism_inchi_key"] = prism["smiles"].map(_inchi_key)
    prism = prism.loc[prism["prism_broad_id_base"].notna()].copy()

    lincs = lincs_pert_info.loc[lincs_pert_info["pert_type"].eq("trt_cp")].copy()
    lincs["lincs_pert_id"] = lincs["pert_id"].astype(str)
    lincs["lincs_inchi_key"] = lincs["inchi_key"].where(
        lincs["inchi_key"].map(_valid), lincs["canonical_smiles"].map(_inchi_key)
    )
    lincs["lincs_alias"] = lincs["pert_iname"].map(_alias)
    lincs = lincs.loc[lincs["lincs_pert_id"].map(_valid)].copy()
    by_id = {
        str(key): group
        for key, group in lincs.groupby("lincs_pert_id")
    }
    by_inchi = {
        str(key): group
        for key, group in lincs.loc[lincs["lincs_inchi_key"].notna()].groupby("lincs_inchi_key")
    }
    by_alias = {
        str(key): group
        for key, group in lincs.loc[lincs["lincs_alias"].notna()].groupby("lincs_alias")
    }

    rows: list[dict[str, object]] = []
    for row in prism.itertuples(index=False):
        base_id = str(row.prism_broad_id_base)
        match_method = "unmatched"
        match = None
        if base_id in by_id:
            candidates = by_id[base_id]
            if len(candidates) == 1:
                match = candidates.iloc[0]
                match_method = "exact_pert_id"
            else:
                match_method = "ambiguous_pert_id"
        elif _valid(row.prism_inchi_key) and str(row.prism_inchi_key) in by_inchi:
            candidates = by_inchi[str(row.prism_inchi_key)]
            if len(candidates) == 1:
                match = candidates.iloc[0]
                match_method = "exact_inchi_key"
            else:
                match_method = "ambiguous_inchi_key"
        elif _alias(row.name) and _alias(row.name) in by_alias:
            candidates = by_alias[_alias(row.name)]
            if len(candidates) == 1:
                match = candidates.iloc[0]
                match_method = "exact_alias"
            else:
                match_method = "ambiguous_alias"
        rows.append(
            {
                "prism_broad_id": str(row.broad_id),
                "prism_broad_id_base": base_id,
                "prism_name": str(row.name),
                "prism_smiles": str(row.smiles),
                "prism_inchi_key": row.prism_inchi_key,
                "lincs_pert_id": str(match["lincs_pert_id"]) if match is not None else None,
                "lincs_pert_iname": str(match["pert_iname"]) if match is not None else None,
                "lincs_smiles": str(match["canonical_smiles"]) if match is not None else None,
                "lincs_inchi_key": str(match["lincs_inchi_key"]) if match is not None else None,
                "match_method": match_method,
                "match_status": (
                    "MATCHED_IDENTITY" if match_method in {"exact_pert_id", "exact_inchi_key"}
                    else "ALIAS_ONLY_CANDIDATE" if match_method == "exact_alias"
                    else "AMBIGUOUS_OR_UNMATCHED"
                ),
            }
        )
    bridge = pd.DataFrame(rows).sort_values("prism_broad_id").reset_index(drop=True)
    counts = bridge["match_method"].value_counts().to_dict()
    unique_compounds = bridge.drop_duplicates("prism_broad_id_base")
    audit = {
        "format": "lincs_prism_identity_bridge_v1",
        "response_values_read": False,
        "prism_unique_broad_ids": int(len(prism)),
        "prism_unique_compound_base_ids": int(unique_compounds["prism_broad_id_base"].nunique()),
        "lincs_trt_cp_compounds": int(len(lincs)),
        "bridge_rows": int(len(bridge)),
        "match_method_counts": {str(key): int(value) for key, value in counts.items()},
        "identity_matches": int(bridge["match_status"].eq("MATCHED_IDENTITY").sum()),
        "alias_only_candidates": int(bridge["match_status"].eq("ALIAS_ONLY_CANDIDATE").sum()),
        "formal_eligible_identity_rows": int(bridge["match_status"].eq("MATCHED_IDENTITY").sum()),
        "formal_eligible_unique_compound_base_ids": int(
            unique_compounds["match_status"].eq("MATCHED_IDENTITY").sum()
        ),
        "exact_pert_id_matches": int((bridge["match_method"] == "exact_pert_id").sum()),
        "exact_inchi_key_matches": int((bridge["match_method"] == "exact_inchi_key").sum()),
        "exact_alias_matches": int((bridge["match_method"] == "exact_alias").sum()),
        "ambiguous_structure_matches": int((bridge["match_method"] == "ambiguous_inchi_key").sum()),
        "ambiguous_pert_id_matches": int((bridge["match_method"] == "ambiguous_pert_id").sum()),
        "ambiguous_alias_matches": int((bridge["match_method"] == "ambiguous_alias").sum()),
        "unmatched_compounds": int(bridge["lincs_pert_id"].isna().sum()),
        "identity_keys": ["pert_id", "canonical_smiles", "InChIKey", "aliases"],
    }
    return bridge, audit
