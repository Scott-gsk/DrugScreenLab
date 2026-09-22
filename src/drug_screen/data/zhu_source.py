"""Exact, source-labelled reconstruction of the Zhu hepatobiliary PDHO tables."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from drug_screen.transfer.chemistry import standardize_smiles


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _name(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _expression(raw: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    if raw.columns[0] != "genes":
        raise ValueError("Zhu RNA-seq sheet must expose a 'genes' header at row 2")
    frame = raw.rename(columns={raw.columns[0]: "gene"}).dropna(subset=["gene"]).copy()
    frame["gene"] = frame.gene.astype(str).str.strip().str.upper()
    frame = frame[frame.gene.ne("")]
    sample_ids = tuple(str(value).strip() for value in frame.columns[1:])
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Zhu RNA-seq sample identities are missing or duplicated")
    values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    if np.isinf(values.to_numpy(float)).any():
        raise ValueError("Zhu RNA-seq contains infinite values")
    # The source sheet already reports author numeric expression.  Only duplicate
    # gene labels are aggregated; no log or other abundance transform is applied.
    values.insert(0, "gene", frame.gene.to_numpy())
    grouped = values.groupby("gene", sort=True).mean(numeric_only=True)
    return grouped.T.to_numpy(float), tuple(grouped.index.astype(str)), sample_ids


def _drug_table(drug_list: pd.DataFrame, chemicals: pd.DataFrame, *, catalogue_expansion: bool = False) -> tuple[pd.DataFrame, dict[str, str], list[dict]]:
    required = {"Drug"}
    if not required <= set(drug_list.columns) or not {"name", "SMILES"} <= set(chemicals.columns):
        raise ValueError("Zhu drug sheets have unexpected headers")
    lookup: dict[str, list[pd.Series]] = {}
    for _, row in chemicals.dropna(subset=["name"]).iterrows():
        key = _name(row["name"])
        if key:
            lookup.setdefault(key, []).append(row)
    rows: list[dict] = []
    excluded: list[dict] = []
    names: dict[str, str] = {}
    for _, row in drug_list.dropna(subset=["Drug"]).iterrows():
        drug = str(row["Drug"]).strip()
        key = _name(drug)
        source_drug_id = f"ZHU:{int(row['Number']):02d}" if pd.notna(row.get("Number")) else f"ZHU:{len(rows)+1:02d}"
        candidates = lookup.get(key, [])
        linkage = "exact_normalized_author_name"
        if not candidates and catalogue_expansion:
            catalogue = str(row.get("Catalogue", "")).strip()
            candidates = [candidate for _, candidate in chemicals.iterrows()
                          if str(candidate.get("cat", "")).strip() == catalogue]
            linkage = "exact_author_catalogue"
        if not candidates:
            excluded.append({"drug": drug, "reason": "no_exact_fixed_name_match_in_author_chemical_table"})
            continue
        identities = [standardize_smiles(candidate.get("SMILES")) for candidate in candidates]
        accepted = [identity for identity in identities if identity.accepted]
        if len(candidates) > 1 and len({identity.inchikey for identity in accepted}) != 1:
            excluded.append({"drug": drug, "reason": "duplicate_or_conflicting_author_chemical_structures",
                             "candidate_count": len(candidates)})
            continue
        if not accepted:
            identity = identities[0]
            excluded.append({"drug": drug, "reason": f"structure_{identity.status}", "message": identity.message})
            continue
        identity = accepted[0]
        rows.append({
            "drug_id": identity.inchikey, "source_drug_id": source_drug_id, "source_drug_name": drug, "author_catalogue": row.get("Catalogue"),
            "canonical_smiles": identity.canonical_isomeric_smiles, "inchikey": identity.inchikey,
            "structure_status": identity.status, "structure_linkage": linkage, "is_small_molecule": True,
        })
        names[source_drug_id] = identity.inchikey
    result = pd.DataFrame(rows)
    if result.empty or result.drug_id.duplicated().any() or result.inchikey.duplicated().any():
        raise ValueError("Zhu eligible drug identities are missing or unresolved")
    return result, names, excluded


@dataclass(frozen=True)
class ZhuRebuild:
    expression: np.ndarray
    sample_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    samples: pd.DataFrame
    drugs: pd.DataFrame
    pairs: pd.DataFrame
    exclusions: list[dict]
    audit: dict


def rebuild_zhu(mmc4: Path, mmc5: Path, *, catalogue_expansion: bool = False) -> ZhuRebuild:
    expression_raw = pd.read_excel(mmc4, sheet_name="RNA-seq", header=1)
    # Rows 0–1 are the table title and drug-number aliases; model/drug labels
    # are the third row in the published worksheet.
    auc_raw = pd.read_excel(mmc5, sheet_name="AUC，1-AA", header=2)
    drug_list = pd.read_excel(mmc5, sheet_name="Further drug screen_Drug list", header=1)
    chemicals = pd.read_excel(mmc5, sheet_name="Chemical Data-301cpds")
    expression_all, genes, expression_samples = _expression(expression_raw)
    if auc_raw.columns[0] != "Cell model":
        raise ValueError("Zhu AUC sheet must expose a 'Cell model' header at row 2")
    auc = auc_raw.rename(columns={auc_raw.columns[0]: "sample_id"}).dropna(subset=["sample_id"]).copy()
    auc.sample_id = auc.sample_id.astype(str).str.strip()
    if auc.sample_id.duplicated().any():
        raise ValueError("Zhu AUC models are duplicated")
    shared = tuple(sample for sample in auc.sample_id if sample in set(expression_samples))
    if not shared:
        raise ValueError("Zhu RNA-seq and AUC tables have no shared PDHO model IDs")
    expression_index = {sample: index for index, sample in enumerate(expression_samples)}
    expression = expression_all[[expression_index[sample] for sample in shared]]
    samples = pd.DataFrame({
        "sample_id": shared, "pdo_id": shared,
        "patient_id": [re.sub(r"^[A-Z]+-", "", sample) for sample in shared],
        "source_model_id": shared, "sample_type": "patient_derived_hepatobiliary_tumor_organoid",
        "patient_identity_evidence": "author PDHO tumour/model number parsed from model ID (for example HCC-350-T/O in source figure); shared cohort has one model per parsed tumour number",
    })
    drugs, drug_ids, exclusions = _drug_table(drug_list, chemicals, catalogue_expansion=catalogue_expansion)
    # The source matrix is AUC (the worksheet is titled "AUC，1-AA"); the
    # publication uses lower viability AUC as greater drug effect.  Store the
    # explicit 1-AUC transform so the shared transfer contract is higher=more
    # sensitive, retaining source AUC alongside it for audit.
    records: list[dict] = []
    number_by_position = {index: f"ZHU:{int(number):02d}" for index, number in enumerate(drug_list["Number"], start=1)}
    for _, row in auc.set_index("sample_id").loc[list(shared)].iterrows():
        for position, auc_column in enumerate(auc.columns[1:], start=1):
            source_drug_id = number_by_position.get(position)
            drug_id = drug_ids.get(source_drug_id)
            if drug_id is None:
                continue
            value = pd.to_numeric(pd.Series([row[auc_column]]), errors="coerce").iloc[0]
            if pd.notna(value):
                records.append({"sample_id": row.name, "drug_id": drug_id, "response": float(1.0 - value),
                                "source_auc": float(value),
                                "response_name": "1_minus_auc", "higher_is_sensitive": True,
                                "source_endpoint": "author worksheet AUC，1-AA", "source_auc_column": str(auc_column),
                                "source_drug_id": source_drug_id})
    pairs = pd.DataFrame(records)
    if pairs.empty or pairs.duplicated(["sample_id", "drug_id"]).any() or not np.isfinite(pairs.response).all():
        raise ValueError("Zhu AUC response pairing is incomplete or nonfinite")
    covered = set(pairs.drug_id)
    absent = drugs.loc[~drugs.drug_id.isin(covered), ["drug_id", "source_drug_name"]]
    exclusions.extend({"drug": row.source_drug_name, "reason": "no_finite_author_AUC_for_any_paired_PDHO"}
                      for row in absent.itertuples(index=False))
    drugs = drugs[drugs.drug_id.isin(covered)].reset_index(drop=True)
    audit = {
        "source_expression_sheet": "mmc4.xlsx/RNA-seq (Table S3)",
        "source_response_sheet": "mmc5.xlsx/AUC，1-AA (Table S4)",
        "expression_transform": "none; author FPKM numeric values preserved after gene-label aggregation only",
        "response_direction_evidence": "source Methods: five-dose treatment, four-day CellTiter-Glo 3D cell-viability readout, then GRmetrics AUC; lower viability-curve AUC means greater inhibition. Published AUC is retained as source_auc and response=1-AUC sets higher_is_sensitive=True",
        "rna_models_source": len(expression_samples), "auc_models_source": len(auc), "paired_models": len(shared),
        "source_model_ids": list(shared), "n_genes": len(genes), "n_eligible_drugs": len(drugs),
        "n_pairs": len(pairs), "expression_quantiles": np.nanquantile(expression, [0, .01, .5, .99, 1]).tolist(),
        "response_quantiles": np.quantile(pairs.response, [0, .01, .5, .99, 1]).tolist(),
        "identity_limit": "model IDs establish expression-response pairing; grouping uses parsed author tumour/model number, pending a formal clinical patient-map table",
        "structure_policy": "author exact normalized name only" if not catalogue_expansion else "author exact normalized name, then author catalogue exact match",
    }
    return ZhuRebuild(expression, shared, genes, samples, drugs, pairs, exclusions, audit)
