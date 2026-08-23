"""Source readers and materializer for the EXP-011 indexed data contract."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import re
import tempfile
import urllib.parse
import urllib.request
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from drug_screen.data.exp011_contract import PAIR_COLUMNS, REQUIRED_MANIFEST_FIELDS, IndexedDrugResponseDataset, gene_digest, sha256_file


COMBINATION_MARKERS = ("+", "&", "MVAC", "FLIRI", "FLOX", "FLV", "MAPK.3X", "VOLASERTIB.SN38", "TAS-102")
ANTIBODY_MARKERS = ("TRASTUZUMAB", "PERTUZUMAB", "CETUXIMAB")
STRUCTURE_ALIASES = {
    "5FU": "5-FU", "CABOZANITIB": "CABOZANTINIB", "TEMEZOLOMIDE": "TEMOZOLOMIDE",
    "ADAVOSERTIBMK1775": "ADAVOSERTIB", "FSEOCALCITOL": "SEOCALCITOL",
}
PRISM_24Q2_URLS = {
    "matrix": "https://ndownloader.figshare.com/files/46630987",
    "treatment": "https://ndownloader.figshare.com/files/46631146",
    "cells": "https://ndownloader.figshare.com/files/46630978",
}


def normalize_name(value: object) -> str:
    return "".join(char for char in str(value).casefold() if char.isalnum())


def source_path_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.name): sha256_file(path) for path in paths if path.is_file()}


def _structure_key(name: object) -> str:
    key = normalize_name(name)
    return normalize_name(STRUCTURE_ALIASES.get(key.upper(), str(name)))


def _is_combination(name: object) -> bool:
    text = str(name).upper()
    return any(marker in text for marker in COMBINATION_MARKERS)


def _is_antibody(name: object) -> bool:
    text = str(name).upper()
    return any(marker in text for marker in ANTIBODY_MARKERS)


def _canonicalize_smiles(smiles: str) -> tuple[str, str, str]:
    """Return canonical SMILES, InChIKey and Bemis--Murcko scaffold."""
    from rdkit import Chem
    from rdkit.Chem import inchi
    from rdkit.Chem.Scaffolds import MurckoScaffold

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"invalid structure: {smiles!r}")
    canonical = Chem.MolToSmiles(molecule, canonical=True)
    inchikey = inchi.MolToInchiKey(molecule)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule) or "ACYCLIC"
    if not inchikey:
        raise ValueError(f"no InChIKey for {smiles!r}")
    return canonical, inchikey, scaffold


@dataclass(frozen=True)
class Structure:
    source_name: str
    canonical_smiles: str
    inchikey: str
    drug_scaffold: str

    @property
    def drug_id(self) -> str:
        return self.inchikey


class StructureResolver:
    """Exact structure resolver with an explicit, auditable PubChem fallback."""

    def __init__(self, structures: Iterable[Structure], cache_path: Path | None = None) -> None:
        self.by_name: dict[str, Structure] = {}
        self.cache_path = cache_path
        for structure in structures:
            self.by_name[_structure_key(structure.source_name)] = structure
        if cache_path and cache_path.exists():
            for item in json.loads(cache_path.read_text(encoding="utf-8")):
                if item.get("canonical_smiles") and item.get("inchikey"):
                    canonical, inchikey, scaffold = _canonicalize_smiles(item["canonical_smiles"])
                    self.by_name[_structure_key(item["source_name"])] = Structure(item["source_name"], canonical, inchikey, scaffold)

    @classmethod
    def from_exp010(cls, data_root: Path, cache_path: Path | None = None) -> "StructureResolver":
        records: list[Structure] = []
        for relative in (
            "external/exp010/pubchem/drug_structures.json",
            "external/exp010/pubchem/mcrc_drug_structures.json",
        ):
            path = data_root / relative
            if not path.is_file():
                continue
            for item in json.loads(path.read_text(encoding="utf-8")):
                if not (item.get("mapped") and item.get("canonical_smiles")):
                    continue
                try:
                    canonical, inchikey, scaffold = _canonicalize_smiles(str(item["canonical_smiles"]))
                except ValueError:
                    continue
                records.append(Structure(str(item["primary_name"]), canonical, inchikey, scaffold))
        return cls(records, cache_path)

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"source_name": value.source_name, "canonical_smiles": value.canonical_smiles, "inchikey": value.inchikey}
            for _, value in sorted(self.by_name.items())
        ]
        self.cache_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _candidate_names(name: str) -> list[str]:
        candidates = [name]
        normalized = str(name).replace("（", "(").replace("）", ")").strip()
        candidates.append(normalized)
        candidates.append(re.sub(r"\s*\([^)]*\)", "", normalized).strip())
        candidates.append(re.sub(r"\b(?:HCl|HCl Trihydrate|Tosylate|Dimaleate|Isethionate|Sodium)\b", "", normalized, flags=re.I).strip())
        candidates.append(re.sub(r"^[a-zA-Z]:", "", normalized).strip())
        alias = STRUCTURE_ALIASES.get(normalize_name(normalized).upper())
        if alias:
            candidates.append(alias)
        return [candidate for candidate in dict.fromkeys(candidates) if candidate]

    @staticmethod
    def _pubchem(name: str) -> Structure | None:
        for candidate in StructureResolver._candidate_names(name):
            url = (
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
                f"{urllib.parse.quote(candidate, safe='')}/property/CanonicalSMILES,InChIKey/JSON"
            )
            try:
                with urllib.request.urlopen(url, timeout=20) as response:
                    value = json.load(response)["PropertyTable"]["Properties"][0]
                smiles = value.get("ConnectivitySMILES") or value.get("CanonicalSMILES")
                if smiles:
                    canonical, inchikey, scaffold = _canonicalize_smiles(smiles)
                    return Structure(name, canonical, inchikey, scaffold)
            except Exception:
                continue
        return None

    def resolve(self, name: object, *, allow_pubchem: bool) -> Structure | None:
        source_name = str(name).strip()
        resolved = self.by_name.get(_structure_key(source_name))
        if resolved is not None:
            return resolved
        if not allow_pubchem:
            return None
        resolved = self._pubchem(source_name)
        if resolved is not None:
            self.by_name[_structure_key(source_name)] = resolved
            self._save_cache()
        return resolved

    def resolve_many(self, names: Iterable[object], *, allow_pubchem: bool, workers: int = 6) -> dict[str, Structure]:
        unique = sorted({str(name).strip() for name in names})
        resolved = {name: self.resolve(name, allow_pubchem=False) for name in unique}
        pending = [name for name, value in resolved.items() if value is None]
        if allow_pubchem and pending:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(self._pubchem, name): name for name in pending}
                for future in as_completed(futures):
                    name = futures[future]
                    result = future.result()
                    if result is not None:
                        self.by_name[_structure_key(name)] = result
                        resolved[name] = result
            self._save_cache()
        return {name: value for name, value in resolved.items() if value is not None}


def align_expression(matrix: pd.DataFrame, ordered_genes: list[str], *, minimum_mapped: int = 15_000) -> tuple[np.ndarray, int]:
    """Align by upper-case HGNC symbols, summing duplicated source columns.

    Missing genes are zero-filled only after the number of directly measured
    contract genes passes a hard gate.  The count is persisted in the manifest
    so no downstream consumer can mistake imputed zeros for measured genes.
    """
    frame = matrix.copy()
    frame.columns = pd.Index([str(column).strip().upper() for column in frame.columns])
    frame = frame.loc[:, frame.columns != ""]
    frame = frame.T.groupby(level=0, sort=True).mean().T
    mapped = int(len(set(frame.columns).intersection(ordered_genes)))
    if mapped < minimum_mapped:
        raise ValueError(f"only {mapped} expression genes map to EXP-011 contract (<{minimum_mapped})")
    aligned = frame.reindex(columns=ordered_genes, fill_value=0.0).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    values = aligned.to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("non-finite expression after alignment")
    return values, mapped


def _samples(
    sample_ids: Iterable[object], *, patient: Callable[[str], str] | None = None,
    pdo: bool = False, cell_line: bool = False, tissue: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    ids = [str(item) for item in sample_ids]
    return pd.DataFrame(
        {
            "sample_id": ids,
            "patient_id": [patient(item) if patient else "" for item in ids],
            "pdo_id": ids if pdo else [""] * len(ids),
            "cell_line_id": ids if cell_line else [""] * len(ids),
            "sample_tissue": [tissue(item) if tissue else "" for item in ids],
        }
    )


def _finite_expression(frame: pd.DataFrame) -> np.ndarray:
    """Use zero only for source-missing expression values, never response labels."""
    numeric = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return numeric.to_numpy(np.float32)


def _pairs_from_rows(
    rows: pd.DataFrame,
    *,
    cohort_id: str,
    name_to_structure: dict[str, Structure],
    response_name: str | pd.Series,
    response_scale: str | pd.Series,
    response: pd.Series,
    higher_is_sensitive: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = rows.copy()
    frame["source_drug_name"] = frame["source_drug_name"].astype(str).str.strip()
    frame["_structure"] = frame["source_drug_name"].map(name_to_structure)
    if frame["_structure"].isna().any():
        missing = sorted(frame.loc[frame["_structure"].isna(), "source_drug_name"].unique())
        raise ValueError(f"unresolved structures in {cohort_id}: {missing}")
    frame["drug_id"] = frame["_structure"].map(lambda item: item.drug_id)
    frame["response"] = pd.to_numeric(response, errors="coerce")
    frame["response_name"] = response_name
    frame["response_scale"] = response_scale
    frame["higher_is_sensitive"] = bool(higher_is_sensitive)
    frame["cohort_id"] = cohort_id
    frame = frame.dropna(subset=["response"])
    drugs = (
        pd.DataFrame(
            {
                "drug_id": [item.drug_id for item in frame["_structure"]],
                "canonical_smiles": [item.canonical_smiles for item in frame["_structure"]],
                "inchikey": [item.inchikey for item in frame["_structure"]],
                "drug_scaffold": [item.drug_scaffold for item in frame["_structure"]],
            }
        )
        .drop_duplicates("drug_id")
        .sort_values("drug_id", kind="mergesort")
        .reset_index(drop=True)
    )
    return frame.loc[:, [*PAIR_COLUMNS]].sort_values(["sample_id", "drug_id", "response_name"], kind="mergesort").reset_index(drop=True), drugs


def _gdsc(data_root: Path, genes: list[str], resolver: StructureResolver) -> tuple[IndexedDrugResponseDataset, dict[str, int]]:
    expression_frame = pd.read_parquet(data_root / "processed/exp010/cmp_gdsc2_gene_contract.parquet")
    if expression_frame.columns.tolist() != genes:
        raise ValueError("CMP gene order differs from frozen EXP-011 gene contract")
    raw = pd.read_parquet(data_root / "processed/exp010/gdsc2_fitted_release8.5.parquet")
    raw = raw.dropna(subset=["SANGER_MODEL_ID", "DRUG_NAME", "AUC"]).copy()
    raw["sample_id"] = raw["SANGER_MODEL_ID"].astype(str)
    raw = raw.loc[raw["sample_id"].isin(expression_frame.index.astype(str))].copy()
    raw["source_drug_name"] = raw["DRUG_NAME"].astype(str)
    structures = resolver.resolve_many(raw["source_drug_name"].unique(), allow_pubchem=False)
    raw = raw.loc[raw["source_drug_name"].isin(structures)].copy()
    raw["response"] = 1.0 - pd.to_numeric(raw["AUC"], errors="coerce")
    if not raw["response"].between(0.0, 1.0).all():
        raise ValueError("GDSC 1-AUC response is outside [0, 1]")
    details_path = data_root / "external/exp010/gdsc2_release8.5/Cell_Lines_Details.xlsx"
    details = pd.read_excel(details_path)
    details["_cosmic_id"] = pd.to_numeric(details["COSMIC identifier"], errors="coerce").astype("Int64")
    detail_tissue = (
        details.loc[:, ["_cosmic_id", "GDSC\nTissue descriptor 1"]]
        .dropna(subset=["_cosmic_id"])
        .rename(columns={"GDSC\nTissue descriptor 1": "_fallback_tissue"})
        .drop_duplicates("_cosmic_id")
    )
    raw["_cosmic_id"] = pd.to_numeric(raw["COSMIC_ID"], errors="coerce").astype("Int64")
    raw = raw.merge(detail_tissue, on="_cosmic_id", how="left", validate="many_to_one")
    raw["_tissue"] = raw["TCGA_DESC"].astype("string").str.strip().str.upper()
    raw["_fallback_tissue"] = raw["_fallback_tissue"].astype("string").str.strip().str.upper()
    raw["_tissue"] = raw["_tissue"].where(raw["_tissue"].notna() & raw["_tissue"].ne(""), raw["_fallback_tissue"])
    # GDSC can have different numeric IDs for a display-name duplicate.  The
    # author-facing unique assay unit is model x drug display name; retain one
    # deterministic fitted row to avoid duplicate labels under one structure.
    raw = raw.sort_values(["sample_id", "source_drug_name", "DRUG_ID"], kind="mergesort").drop_duplicates(["sample_id", "source_drug_name"])
    tissue_by_model = (
        raw.loc[:, ["sample_id", "_tissue"]]
        .replace({"": pd.NA, "NAN": pd.NA})
        .groupby("sample_id", sort=True)["_tissue"].agg(lambda values: sorted(set(values.dropna())))
    )
    ambiguous_tissue = {sample: values for sample, values in tissue_by_model.items() if len(values) != 1}
    if ambiguous_tissue:
        raise ValueError(f"GDSC tissue contract needs one nonempty TCGA_DESC per model: {dict(list(ambiguous_tissue.items())[:5])}")
    tissue_lookup = {str(sample): values[0] for sample, values in tissue_by_model.items()}
    pairs, drugs = _pairs_from_rows(
        raw, cohort_id="gdsc2_cell_line", name_to_structure=structures,
        response_name="1_minus_auc", response_scale="GDSC2 fitted AUC transformed to 1-AUC", response=raw["response"],
    )
    samples = _samples(sorted(expression_frame.index.astype(str)), cell_line=True, tissue=lambda item: tissue_lookup[item])
    expression = _finite_expression(expression_frame.loc[samples["sample_id"], genes])
    dataset = IndexedDrugResponseDataset(
        cohort_id="gdsc2_cell_line", expression=expression, ordered_genes=genes, samples=samples, drugs=drugs, pairs=pairs,
        expression_unit="CMP TPM (author-provided)",
        source_hashes=source_path_hashes([
            data_root / "processed/exp010/cmp_gdsc2_gene_contract.parquet",
            data_root / "processed/exp010/gdsc2_fitted_release8.5.parquet",
            data_root / "external/exp010/pubchem/drug_structures.json",
            details_path,
        ]),
    )
    counts = dataset.counts
    counts |= {
        "sample_tissue_categories": int(samples["sample_tissue"].nunique()),
        "sample_tissue_missing": int(samples["sample_tissue"].astype(str).eq("").sum()),
    }
    expected = {"samples": 963, "drugs": 229, "response_rows": 192_845}
    if {key: counts[key] for key in expected} != expected:
        raise ValueError(f"GDSC acceptance gate failed: {counts}; expected {expected}")
    return dataset, counts


def _mcrc(data_root: Path, genes: list[str], resolver: StructureResolver) -> tuple[IndexedDrugResponseDataset, dict[str, int]]:
    expression_frame = pd.read_parquet(data_root / "processed/exp010/mcrc_gene_contract.parquet")
    if expression_frame.columns.tolist() != genes:
        raise ValueError("mCRC gene order differs from frozen EXP-011 gene contract")
    raw = pd.read_parquet(data_root / "processed/exp010/mcrc_monotherapy_dss.parquet")
    raw = raw.rename(columns={"drug_name": "source_drug_name", "dss": "response"})
    raw["sample_id"] = raw["sample_id"].astype(str)
    raw = raw.loc[raw["sample_id"].isin(expression_frame.index.astype(str))].copy()
    structures = resolver.resolve_many(raw["source_drug_name"].unique(), allow_pubchem=False)
    pairs, drugs = _pairs_from_rows(
        raw, cohort_id="mcrc_pdo", name_to_structure=structures, response_name="dss", response_scale="DSS", response=raw["response"],
    )
    samples = _samples(sorted(expression_frame.index.astype(str)), patient=lambda item: str(raw.loc[raw["sample_id"].eq(item), "patient_id"].iloc[0]), pdo=True)
    expression = _finite_expression(expression_frame.loc[samples["sample_id"], genes])
    dataset = IndexedDrugResponseDataset(
        cohort_id="mcrc_pdo", expression=expression, ordered_genes=genes, samples=samples, drugs=drugs, pairs=pairs,
        expression_unit="log2(CPM+1) from mCRC source count table",
        source_hashes=source_path_hashes([
            data_root / "processed/exp010/mcrc_gene_contract.parquet",
            data_root / "processed/exp010/mcrc_monotherapy_dss.parquet",
            data_root / "external/exp010/pubchem/mcrc_drug_structures.json",
        ]),
    )
    expected = {"patients": 52, "pdo_samples": 91, "response_rows": 3268}
    if {key: dataset.counts[key] for key in expected} != expected:
        raise ValueError(f"mCRC acceptance gate failed: {dataset.counts}; expected {expected}")
    return dataset, dataset.counts


def _zhu(data_root: Path, genes: list[str], resolver: StructureResolver, *, allow_pubchem: bool) -> tuple[IndexedDrugResponseDataset, dict[str, int]]:
    root = data_root / "external/pdo_public_candidates/zhu2024_hepatobiliary_64"
    expr_path, response_path = root / "mmc4.xlsx", root / "mmc5.xlsx"
    expr_raw = pd.read_excel(expr_path, sheet_name="RNA-seq", header=1)
    gene_column = expr_raw.columns[0]
    expression_frame = expr_raw.set_index(gene_column).T
    expression_frame.index = expression_frame.index.astype(str)
    response_wide = pd.read_excel(response_path, sheet_name="AUC，1-AA", header=2)
    response_wide = response_wide.rename(columns={response_wide.columns[0]: "sample_id"})
    response_wide["sample_id"] = response_wide["sample_id"].astype(str)
    common = sorted(set(expression_frame.index).intersection(response_wide["sample_id"]))
    raw = response_wide.loc[response_wide["sample_id"].isin(common)].melt(id_vars="sample_id", var_name="source_drug_name", value_name="auc")
    raw["auc"] = pd.to_numeric(raw["auc"], errors="coerce")
    raw = raw.dropna(subset=["auc"])
    raw["response"] = 1.0 - raw["auc"]
    if not raw["response"].between(0.0, 1.0).all():
        raise ValueError("Zhu 1-AUC response is outside [0, 1]")
    structures = resolver.resolve_many(raw["source_drug_name"].unique(), allow_pubchem=allow_pubchem)
    pairs, drugs = _pairs_from_rows(
        raw, cohort_id="zhu_hepatobiliary", name_to_structure=structures,
        response_name="1_minus_auc", response_scale="author AUC transformed to 1-AUC", response=raw["response"],
    )
    samples = _samples(common, patient=lambda item: item, pdo=True)
    # This public supplement contains 16,410 reported symbols and 14,742
    # exact HGNC matches to the fixed 19,091-gene source contract.  Preserve
    # the contract dimensionality with explicit zero-fill for absent genes,
    # but fail if its documented coverage materially regresses.
    expression, mapped_genes = align_expression(expression_frame.loc[common], genes, minimum_mapped=14_000)
    dataset = IndexedDrugResponseDataset(
        cohort_id="zhu_hepatobiliary", expression=expression, ordered_genes=genes, samples=samples, drugs=drugs, pairs=pairs,
        expression_unit="author-provided RNA-seq expression; publication unit not explicitly labelled",
        source_hashes=source_path_hashes([expr_path, response_path]),
    )
    counts = dataset.counts | {"directly_mapped_genes": mapped_genes}
    expected = {"pdo_samples": 20, "drugs": 64, "response_rows": 1280}
    if {key: counts[key] for key in expected} != expected:
        raise ValueError(f"Zhu acceptance gate failed: {counts}; expected {expected}")
    return dataset, counts


def _merrill(data_root: Path, genes: list[str], resolver: StructureResolver, *, allow_pubchem: bool) -> tuple[IndexedDrugResponseDataset, dict[str, int]]:
    root = data_root / "external/pdo_public_candidates/merrill2024_bladder_65"
    workbook_path = root / "NIHMS2149379-supplement-Supplementary_Tables.xlsx"
    expression_raw = pd.read_excel(workbook_path, sheet_name="Table S5", header=1)
    expression_frame = expression_raw.set_index(expression_raw.columns[0]).T
    expression_frame.index = expression_frame.index.astype(str)
    response_wide = pd.read_excel(workbook_path, sheet_name="Table S11", header=1)
    response_wide = response_wide.rename(columns={response_wide.columns[0]: "sample_id"})
    response_wide["sample_id"] = response_wide["sample_id"].astype(str)
    common = sorted(set(expression_frame.index).intersection(response_wide["sample_id"]))
    response_wide = response_wide.loc[response_wide["sample_id"].isin(common)]
    rows: list[dict[str, object]] = []
    for column in response_wide.columns[1:]:
        label = str(column)
        is_dss = label.upper().endswith("DSS")
        drug_name = label[:-3] if is_dss else label
        if _is_combination(drug_name) or _is_antibody(drug_name):
            continue
        values = pd.to_numeric(response_wide[column], errors="coerce")
        for sample_id, value in zip(response_wide["sample_id"], values, strict=True):
            if pd.isna(value):
                continue
            rows.append(
                {
                    "sample_id": str(sample_id), "source_drug_name": drug_name,
                    "response": float(value) if is_dss else float(100.0 - value),
                    "response_name": "dss3" if is_dss else "cmax_sensitivity_100_minus_viability",
                    "response_scale": "DSS3" if is_dss else "Cmax viability transformed to 100-viability",
                }
            )
    raw = pd.DataFrame(rows)
    structures = resolver.resolve_many(raw["source_drug_name"].unique(), allow_pubchem=allow_pubchem)
    pairs, drugs = _pairs_from_rows(
        raw, cohort_id="merrill_bladder_external", name_to_structure=structures,
        response_name=raw["response_name"], response_scale=raw["response_scale"], response=raw["response"],
    )
    samples = _samples(common, patient=lambda item: item, pdo=True)
    expression, mapped_genes = align_expression(expression_frame.loc[common], genes)
    dataset = IndexedDrugResponseDataset(
        cohort_id="merrill_bladder_external", expression=expression, ordered_genes=genes, samples=samples, drugs=drugs, pairs=pairs,
        expression_unit="primary-tumor whole-transcriptome log2-normalized RNA-seq; linked to matched PDO response",
        source_hashes=source_path_hashes([workbook_path]),
    )
    if dataset.counts["pdo_samples"] != 65:
        raise ValueError(f"Merrill exact expression-response pairing gate failed: {dataset.counts}")
    return dataset, dataset.counts | {"directly_mapped_genes": mapped_genes}


def _download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size:
        return target
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
        while block := response.read(8 << 20):
            output.write(block)
    partial.replace(target)
    return target


def _prism(
    data_root: Path,
    project_root: Path,
    genes: list[str],
    resolver: StructureResolver,
    *,
    prism_matrix: Path | None,
    prism_treatment: Path | None,
    prism_cells: Path | None,
    download_prism: bool,
) -> tuple[IndexedDrugResponseDataset, dict[str, int]]:
    if download_prism:
        cache = Path(tempfile.gettempdir()) / "drugscreenlab_exp011_prism24q2"
        prism_matrix = prism_matrix or _download(PRISM_24Q2_URLS["matrix"], cache / "PRISM_Repurposing_24Q2_LFC.csv")
        prism_treatment = prism_treatment or _download(PRISM_24Q2_URLS["treatment"], cache / "PRISM_Repurposing_24Q2_Treatment_Info.csv")
        prism_cells = prism_cells or _download(PRISM_24Q2_URLS["cells"], cache / "PRISM_Repurposing_24Q2_Cell_Line_Info.csv")
    if not all(path is not None and path.is_file() for path in (prism_matrix, prism_treatment, prism_cells)):
        raise FileNotFoundError("PRISM 24Q2 requires --prism-matrix, --prism-treatment and --prism-cells, or --download-prism")
    assert prism_matrix is not None and prism_treatment is not None and prism_cells is not None
    legacy_treatment = project_root / "mvp/core_data/_source_cache/primary-screen-replicate-collapsed-treatment-info.csv"
    legacy_cells = project_root / "mvp/core_data/_source_cache/primary-screen-cell-line-info.csv"
    legacy = pd.read_csv(legacy_treatment, dtype=str).dropna(subset=["broad_id", "smiles"]).copy()
    legacy["_structure"] = legacy["smiles"].map(lambda value: _canonicalize_smiles(value) if pd.notna(value) else None)
    legacy = legacy.loc[legacy["_structure"].notna()].copy()
    legacy["canonical_smiles"] = legacy["_structure"].map(lambda value: value[0])
    legacy["inchikey"] = legacy["_structure"].map(lambda value: value[1])
    legacy["drug_scaffold"] = legacy["_structure"].map(lambda value: value[2])
    legacy = legacy.drop_duplicates("broad_id")
    treatment = pd.read_csv(prism_treatment, dtype=str)
    required = {"profile_id", "broad_id", "name"}
    if missing := required.difference(treatment.columns):
        raise ValueError(f"PRISM 24Q2 treatment metadata missing {sorted(missing)}")
    # PRISM 24Q2 response labels are not opened until its profile identities
    # have been intersected with source GDSC by exact InChIKey.  The 24Q2
    # metadata does not publish structures, so first use its response-blind
    # 19Q4 broad-ID structure record and then PubChem only for unmatched names.
    selected = treatment.merge(
        legacy[["broad_id", "canonical_smiles", "inchikey", "drug_scaffold"]], on="broad_id", how="left", validate="many_to_one"
    ).drop_duplicates("profile_id")
    unresolved_names = selected.loc[selected["canonical_smiles"].isna(), "name"].astype(str).unique()
    by_name = resolver.resolve_many(unresolved_names, allow_pubchem=True)
    selected["_structure"] = selected["name"].astype(str).map(by_name)
    selected["canonical_smiles"] = selected["canonical_smiles"].where(
        selected["canonical_smiles"].notna(), selected["_structure"].map(lambda value: value.canonical_smiles if isinstance(value, Structure) else None)
    )
    selected["inchikey"] = selected["inchikey"].where(
        selected["inchikey"].notna(), selected["_structure"].map(lambda value: value.inchikey if isinstance(value, Structure) else None)
    )
    selected["drug_scaffold"] = selected["drug_scaffold"].where(
        selected["drug_scaffold"].notna(), selected["_structure"].map(lambda value: value.drug_scaffold if isinstance(value, Structure) else None)
    )
    gdsc_inchikeys = {value.inchikey for value in resolver.by_name.values()}
    selected = selected.loc[selected["inchikey"].isin(gdsc_inchikeys)].drop_duplicates("inchikey")
    # The frozen CRC panel is inherited from the response-blind 19Q4 cell
    # metadata.  The current 24Q2 matrix is only opened after this identity set
    # and the exact InChIKey drug set are fixed.
    legacy_crc = pd.read_csv(legacy_cells, dtype=str)
    crc_ids = sorted(legacy_crc.loc[legacy_crc["primary_tissue"].str.lower().eq("colorectal"), "depmap_id"].astype(str).unique())
    if len(crc_ids) != 35:
        raise ValueError(f"frozen PRISM CRC panel expected 35 lines, got {len(crc_ids)}")
    if selected["inchikey"].nunique() != 81:
        raise ValueError(f"PRISM exact InChIKey gate expected 81 drugs, got {selected['inchikey'].nunique()}")
    header = pd.read_csv(prism_matrix, nrows=0).columns.tolist()
    lfc_required = {"row_id", "profile_id", "LFC"}
    if missing := lfc_required.difference(header):
        raise ValueError(f"PRISM 24Q2 LFC schema missing {sorted(missing)}")
    usecols = ["row_id", "profile_id", "LFC"] + (["PASS"] if "PASS" in header else [])
    lfc = pd.read_csv(prism_matrix, usecols=usecols, dtype={"row_id": str, "profile_id": str})
    lfc["sample_id"] = lfc["row_id"].astype(str).str.split("::", n=1).str[0]
    lfc = lfc.loc[lfc["sample_id"].isin(crc_ids) & lfc["profile_id"].isin(selected["profile_id"].astype(str))].copy()
    if "PASS" in lfc.columns:
        lfc = lfc.loc[lfc["PASS"].astype(str).str.upper().isin(["TRUE", "1"])]
    values = lfc.assign(LFC=pd.to_numeric(lfc["LFC"], errors="coerce")).dropna(subset=["LFC"]).groupby(["sample_id", "profile_id"], as_index=False, sort=True)["LFC"].mean()
    missing_cells = set(crc_ids).difference(values["sample_id"].astype(str))
    if missing_cells:
        raise ValueError(f"frozen PRISM CRC lines missing from 24Q2 response: {sorted(missing_cells)}")
    rows: list[dict[str, object]] = []
    values = values.merge(selected[["profile_id", "name", "inchikey"]], on="profile_id", how="inner", validate="many_to_one")
    for value in values.itertuples(index=False):
        rows.append({"sample_id": str(value.sample_id), "source_drug_name": str(value.name), "drug_id": str(value.inchikey), "response": float(-value.LFC)})
    raw = pd.DataFrame(rows)
    if len(raw) != 2784:
        raise ValueError(f"PRISM finite response gate expected 2784 rows, got {len(raw)}")
    model = pd.read_csv(data_root / "raw/depmap/24q2/Model.csv", dtype=str)
    if "ModelID" not in model.columns:
        raise ValueError("DepMap 24Q2 Model.csv missing ModelID")
    expression_path = data_root / "raw/depmap/24q2/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    expression_header = pd.read_csv(expression_path, nrows=0).columns.tolist()
    symbol_to_column = {re.sub(r"\s*\([^)]*\)$", "", column).strip().upper(): column for column in expression_header[1:]}
    selected_columns = [symbol_to_column[gene] for gene in genes if gene in symbol_to_column]
    expression_raw = pd.read_csv(expression_path, usecols=[expression_header[0], *selected_columns])
    expression_raw = expression_raw.rename(columns={expression_header[0]: "sample_id"}).set_index("sample_id")
    expression_raw.index = expression_raw.index.astype(str)
    expression_values, mapped_genes = align_expression(expression_raw.loc[crc_ids], genes)
    samples = _samples(crc_ids, cell_line=True)
    drugs = selected[["inchikey", "canonical_smiles", "drug_scaffold"]].drop_duplicates("inchikey").rename(columns={"inchikey": "drug_id"})
    pairs = raw.assign(
        cohort_id="prism_crc_external", response_name="neg_log2_fold_change", response_scale="PRISM log2 fold-change multiplied by -1", higher_is_sensitive=True,
    ).loc[:, PAIR_COLUMNS]
    dataset = IndexedDrugResponseDataset(
        cohort_id="prism_crc_external", expression=expression_values, ordered_genes=genes, samples=samples, drugs=drugs, pairs=pairs,
        expression_unit="DepMap 24Q2 log2(TPM+1)",
        source_hashes=source_path_hashes([prism_matrix, prism_treatment, prism_cells, legacy_treatment, legacy_cells, expression_path]),
    )
    return dataset, dataset.counts | {"directly_mapped_genes": mapped_genes}


def build_exp011(
    *,
    data_root: Path,
    output: Path,
    project_root: Path,
    allow_pubchem: bool = False,
    with_prism: bool = False,
    prism_matrix: Path | None = None,
    prism_treatment: Path | None = None,
    prism_cells: Path | None = None,
    download_prism: bool = False,
) -> dict[str, object]:
    """Materialize all available EXP-011 cohorts and one top-level manifest."""
    output.mkdir(parents=True, exist_ok=True)
    gene_source = data_root / "processed/exp010/cmp_gdsc2_gene_contract.parquet"
    genes = pd.read_parquet(gene_source).columns.astype(str).tolist()
    if len(genes) != 19_091 or genes != sorted(genes):
        raise ValueError("EXP-011 must inherit exactly 19,091 sorted genes from the frozen EXP-010 contract")
    resolver = StructureResolver.from_exp010(data_root, output / "structure_resolution_cache.json")
    materializers: list[tuple[str, Callable[[], tuple[IndexedDrugResponseDataset, dict[str, int]]], str, str, str]] = [
        ("gdsc2_cell_line", lambda: _gdsc(data_root, genes, resolver), "SOURCE_TRAIN_AND_COLD_CELL_EVALUATION", "cell_line", "GDSC2/CMP source; one response per cell-line × display drug name"),
        ("mcrc_pdo", lambda: _mcrc(data_root, genes, resolver), "PDO_DEVELOPMENT", "patient_derived_organoid", "DSS monotherapy; patient is grouping unit"),
        ("zhu_hepatobiliary", lambda: _zhu(data_root, genes, resolver, allow_pubchem=allow_pubchem), "PDO_DEVELOPMENT", "patient_derived_organoid", "paired RNA-seq/PDO response; source AUC transformed to 1-AUC"),
        ("merrill_bladder_external", lambda: _merrill(data_root, genes, resolver, allow_pubchem=allow_pubchem), "FROZEN_EXTERNAL_BRIDGE", "primary_tumor_expression_matched_to_pdo_response", "expression is primary tumor, not PDO expression; combination/antibody excluded"),
    ]
    if with_prism:
        materializers.append(("prism_crc_external", lambda: _prism(data_root, project_root, genes, resolver, prism_matrix=prism_matrix, prism_treatment=prism_treatment, prism_cells=prism_cells, download_prism=download_prism), "FROZEN_EXTERNAL_CELL_LINE", "cell_line", "PRISM 24Q2 CRC exact-InChIKey overlap; labels never used for selection"))
    records: list[dict[str, object]] = []
    evaluation_roles = {
        "gdsc2_cell_line": ["gdsc_cold_cell", "gdsc_cold_drug"],
        "mcrc_pdo": ["pdo_primary"],
        "zhu_hepatobiliary": ["pdo_primary"],
        "merrill_bladder_external": ["external_exploratory"],
        "prism_crc_external": ["prism_frozen"],
    }
    for cohort_id, loader, role, sample_type, notes in materializers:
        print(f"[EXP-011] materializing {cohort_id}", flush=True)
        dataset, supplemental_counts = loader()
        output_hashes = dataset.write(output)
        record = dataset.manifest_record(role=role, sample_type=sample_type, notes=notes).to_dict()
        record["counts"] = dict(record["counts"]) | supplemental_counts
        record["hashes"] = dict(record["hashes"]) | {f"output_{key}": value for key, value in output_hashes.items()}
        raw_manifest_name = f"{cohort_id}_raw_manifest.json"
        raw_manifest_path = output / raw_manifest_name
        raw_manifest = {
            "schema": "drugscreenlab.exp011.raw-indexed-pairs.v1",
            "raw_npz": f"{cohort_id}_indexed.npz",
            "formal_gene_contract": True,
            "cohort_id": cohort_id,
            "role": role,
            "counts": record["counts"],
            "source_hashes": dataset.source_hashes,
            "bundle_hashes": output_hashes,
        }
        raw_manifest_path.write_text(json.dumps(raw_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        record["bundle"] = {
            "raw_manifest": raw_manifest_name,
            "raw_npz": f"{cohort_id}_indexed.npz",
            "samples": f"{cohort_id}_samples.parquet",
            "drugs": f"{cohort_id}_drugs.parquet",
            "pairs": f"{cohort_id}_pairs.parquet",
        }
        record["hashes"]["output_raw_manifest"] = sha256_file(raw_manifest_path)
        record["evaluation_roles"] = evaluation_roles[cohort_id]
        record["prediction_metadata_required"] = [
            "cohort_id", "fold_id", "split", "patient_id", "pdo_id", "cell_line_id", "drug_id", "inchikey",
            "drug_scaffold", "response_name", "response_scale", "higher_is_sensitive", "y_true",
        ]
        if cohort_id == "gdsc2_cell_line":
            record["sample_tissue_contract"] = {
                "field": "samples.sample_tissue / indexed.npz:sample_tissue",
                "source": "GDSC2 fitted dose-response TCGA_DESC; missing TCGA_DESC falls back by exact COSMIC_ID to Cell_Lines_Details GDSC Tissue descriptor 1",
                "cleaning": "strip, uppercase; empty/NAN rejected after fallback; one distinct label required per retained model",
                "missing_after_cleaning": int(record["counts"]["sample_tissue_missing"]),
            }
        records.append(record)
        print(f"[EXP-011] accepted {cohort_id}: {record['counts']}", flush=True)
    exclusions = [
        {"cohort_id": "farin_crc_29x4", "reason": "response table is public but gene-expression matrix is controlled-access; no paired public expression-response input"},
        {"cohort_id": "shi_pancreatic_84", "reason": "current supplementary assets lack an immediately pairable whole-transcriptome expression matrix"},
        {"cohort_id": "multiregional_breast_pdo", "reason": "six supplied supplements contain metadata/figures, not a sample-level numeric RNA-seq × drug-response matrix"},
        {"cohort_id": "gse303509_breast", "reason": "14 RNA-seq profiles but no public numeric PDO drug-response matrix; frozen directional control only"},
        {"cohort_id": "gse276609_and_gse287429", "reason": "expression/perturbation references without supervised drug-sensitivity labels"},
    ]
    payload: dict[str, object] = {
        "experiment_id": "EXP-011",
        "format": "exp011_indexed_drug_response_v1",
        "gene_contract": {"count": len(genes), "sha256": gene_digest(genes), "order": "lexicographic uppercase HGNC"},
        "required_fields": list(REQUIRED_MANIFEST_FIELDS),
        "cohorts": records,
        "bundle_schema": "drugscreenlab.exp011.raw-indexed-pairs.v1",
        "excluded_or_not_materialized": exclusions,
        # The planned PRISM external evaluation is deliberately not silently
        # dropped.  The official 24Q2 LFC downloader was probed on this
        # execution track and returned HTTP 403 for the first artifact below.
        # Keep that fact in the aggregate manifest so a training run cannot
        # mistake the four materialized cohorts for a completed PRISM gate.
        "prism_status": "MATERIALIZED" if with_prism else "PENDING",
        "prism_acquisition": (
            {"status": "MATERIALIZED"}
            if with_prism
            else {
                "status": "PENDING_OFFICIAL_24Q2_INPUT",
                "last_observed_official_download_failure": {
                    "error_type": "HTTPError",
                    "http_status": 403,
                    "failed_artifact": "PRISM_Repurposing_24Q2_LFC.csv",
                    "failed_url": PRISM_24Q2_URLS["matrix"],
                    "official_urls": PRISM_24Q2_URLS,
                },
                "recommended_local_directory": "data/external/exp011/prism24q2",
                "required_local_files": [
                    "PRISM_Repurposing_24Q2_LFC.csv",
                    "PRISM_Repurposing_24Q2_Treatment_Info.csv",
                    "PRISM_Repurposing_24Q2_Cell_Line_Info.csv",
                ],
                "materialization_cli": (
                    "build_exp011_inputs.py --with-prism "
                    "--prism-matrix <LFC.csv> --prism-treatment <Treatment_Info.csv> "
                    "--prism-cells <Cell_Line_Info.csv>"
                ),
            }
        ),
        "pair_storage": "indexed; expression matrix is unique per sample and never duplicated per response pair",
    }
    manifest_path = output / "cohort_index_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return payload
