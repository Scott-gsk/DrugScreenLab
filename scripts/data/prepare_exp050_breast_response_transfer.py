"""Prepare and score the frozen descriptive breast organoid response transfer slice.

All inputs are read-only.  The target/source similarity is gene-axis descriptive;
the 914 genes are not biological replicates and no inferential p-values are made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/raw/transfer_public_assets/gse_breast_drug_response"
LINCS = ROOT / "data/processed/lincs/GSE92742"
DOSE = {"DOX": "BRD-K92093830", "BTZ": "BRD-K88510285"}
DRUG_NAME = {"DOX": "doxorubicin", "BTZ": "bortezomib"}
PUBCHEM_INCHIKEY = {
    "DOX": "AOJJSUZBOXZQNB-TZSSRYMLSA-N",
    "BTZ": "GXJABQQUPOEUTA-RDJZCZTQSA-N",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def checked_inputs() -> dict[str, dict[str, Any]]:
    manifest = pd.read_csv(BASE / "intake_manifest.tsv", sep="\t", dtype=str).fillna("")
    result: dict[str, dict[str, Any]] = {}
    for row in manifest.to_dict("records"):
        rel = row.get("local_path_repo_relative", "")
        expected = row.get("sha256", "")
        if not rel or not expected or row.get("acquisition_status") == "not downloaded":
            continue
        path = ROOT / rel
        if not path.is_file():
            continue
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f"intake checksum mismatch: {rel}")
        result[rel] = {"bytes": path.stat().st_size, "sha256": observed}

    audits = {
        "data/processed/exp010/gse287429_author_normalized.parquet":
            "a7c6c918ade78a16edcc7c518622f3e5f72d4fd820e49415f303607acf4befa7",
        "data/processed/exp010/gse287429_context.parquet":
            "4dff69c2457600ff44855505eda7300f31834f8fb1ac8529937d526246af6504",
    }
    for rel, expected in audits.items():
        path = ROOT / rel
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f"pre-existing GSE287429 asset checksum mismatch: {rel}")
        result[rel] = {"bytes": path.stat().st_size, "sha256": observed}
    direct_inputs = [
        "data/raw/transfer_public_assets/gse_breast_drug_response/intake_manifest.tsv",
        "data/raw/transfer_public_assets/gse_breast_drug_response/GSE216540/GSE216540_sample_metadata.tsv",
        "data/raw/transfer_public_assets/gse_breast_drug_response/GSE287429/GSE287429_sample_metadata.tsv",
        "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_pert_info.txt.gz",
        "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_cell_info.txt.gz",
    ]
    for rel in direct_inputs:
        path = ROOT / rel
        result[rel] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    lincs_used = [
        "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_gene_info.txt.gz",
        "data/processed/lincs/GSE92742/continuous_contract_v2/genes.json",
        "data/processed/lincs/GSE92742/continuous_contract_v2/quality_report.json",
        "data/processed/lincs/GSE92742/continuous_contract_v2/treatments.parquet",
        "data/processed/lincs/GSE92742/continuous_conditions_v2/conditions.parquet",
        "data/processed/lincs/GSE92742/continuous_conditions_v2/condition_delta.npy",
        "data/processed/lincs/GSE92742/continuous_conditions_v2/manifest.json",
        "data/processed/lincs/GSE92742/continuous_conditions_v2/verification.json",
        "data/raw/exp013/msigdb/h.all.v2023.2.Hs.symbols.gmt",
    ]
    for rel in lincs_used:
        path = ROOT / rel
        result[rel] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return result


def finite_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return None
    a, b = x[mask], y[mask]
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return None
    a = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=np.float64)
    b = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=np.float64)
    return finite_correlation(a, b)


def summarize_pair(target: np.ndarray, source: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(target) & np.isfinite(source)
    nz = finite & (target != 0) & (source != 0)
    both_zero = finite & (target == 0) & (source == 0)
    target_zero = finite & (target == 0) & (source != 0)
    source_zero = finite & (target != 0) & (source == 0)
    same = nz & (np.sign(target) == np.sign(source))
    return {
        "n_finite_gene_pairs": int(finite.sum()),
        "pearson_r": finite_correlation(target, source),
        "spearman_rho": rank_correlation(target, source),
        "n_both_nonzero": int(nz.sum()),
        "n_same_sign": int(same.sum()),
        "sign_agreement": float(same.sum() / nz.sum()) if nz.any() else None,
        "n_both_zero": int(both_zero.sum()),
        "n_target_zero_source_nonzero": int(target_zero.sum()),
        "n_target_nonzero_source_zero": int(source_zero.sum()),
    }


def hallmark_gene_set_summary(vectors: dict[str, np.ndarray], symbols: list[str | None],
                              unit: str, gmt_path: Path) -> pd.DataFrame:
    pathways: dict[str, set[str]] = {}
    for line_number, line in enumerate(gmt_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("\t")
        if len(fields) < 3 or fields[0] in pathways:
            raise ValueError(f"invalid or duplicate Hallmark GMT row {line_number}")
        pathways[fields[0]] = {gene.strip().upper() for gene in fields[2:] if gene.strip()}
    if len(pathways) != 50:
        raise ValueError(f"expected 50 frozen Hallmark sets, got {len(pathways)}")
    index = {symbol: i for i, symbol in enumerate(symbols) if symbol}
    if len(index) != sum(symbol is not None for symbol in symbols):
        raise ValueError("duplicate symbol on Hallmark response axis")
    rows = []
    for contrast, values in vectors.items():
        for name, members in pathways.items():
            present = sorted(members.intersection(index))
            selected = np.asarray([values[index[gene]] for gene in present], dtype=np.float64)
            finite = selected[np.isfinite(selected)]
            rows.append({"contrast": contrast, "hallmark_set": name,
                         "n_genes_in_set": len(members), "n_genes_present": len(present),
                         "coverage_fraction": len(present) / len(members),
                         "n_finite_response_genes": len(finite),
                         "mean_response_delta": float(finite.mean()) if len(finite) else None,
                         "median_response_delta": float(np.median(finite)) if len(finite) else None,
                         "mean_absolute_response_delta": float(np.abs(finite).mean()) if len(finite) else None,
                         "n_positive": int((finite > 0).sum()), "n_negative": int((finite < 0).sum()),
                         "n_zero": int((finite == 0).sum()), "response_unit": unit,
                         "inference": "descriptive gene-set average; no pathway p-value or activity claim"})
    return pd.DataFrame(rows)


def prepare(config_path: Path, run_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}", run_id):
        raise ValueError("run_id must be a short path-safe identifier")
    out = ROOT / "artifacts/experiments/EXP-050" / run_id
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {out}")
    out.mkdir(parents=True)
    inputs = checked_inputs()

    # GSE216540: preserve every documented GEO group/batch and calculate
    # treatment-minus-DMSO after log2(TPM+1), without replicate-level pairing.
    g216 = BASE / "GSE216540"
    meta216 = pd.read_csv(g216 / "GSE216540_sample_metadata.tsv", sep="\t", dtype=str).fillna("")
    expr216 = pd.read_csv(g216 / "GSE216540_allData_tpm_human.txt.gz", sep="\t")
    sample_keys = expr216.columns[8:].tolist()
    if sample_keys != meta216["expression_column_key"].tolist():
        raise ValueError("GSE216540 TPM columns do not match audited metadata order")
    gene_id = expr216["Geneid"].astype(str).str.replace(r"\.\d+$", "", regex=True)
    symbol = expr216["GeneSymbol"].astype(str).str.strip()
    if gene_id.duplicated().any() or symbol.duplicated().any() or symbol.eq("").any():
        raise ValueError("GSE216540 gene-id/symbol mapping is not one-to-one")
    values = expr216[sample_keys].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("GSE216540 TPM contains negative or non-finite values")
    log_values = np.log2(values + 1.0)
    col_idx = {key: i for i, key in enumerate(sample_keys)}
    meta216["_pdo"] = meta216["pdo_label_from_characteristics"].str.extract(r"#(\d+)")[0]
    meta216["_media"] = meta216["media_code_from_expression_key"]
    meta216["_drug"] = meta216["drug_code_from_expression_key"]
    meta216["_batch"] = meta216["batch_from_title"]
    if meta216["_pdo"].isna().any() or meta216["_media"].isin([""]).any():
        raise ValueError("GSE216540 has missing group/media keys")

    batch_vectors: dict[str, np.ndarray] = {}
    group_vectors: dict[str, np.ndarray] = {}
    group_counts: list[dict[str, Any]] = []
    for (pdo, media, batch), frame in meta216.groupby(["_pdo", "_media", "_batch"], sort=True):
        drug = frame.loc[frame["_drug"].eq("FULV"), "expression_column_key"].tolist()
        ctrl = frame.loc[frame["_drug"].eq("DMSO"), "expression_column_key"].tolist()
        if not drug or not ctrl:
            raise ValueError(f"missing GSE216540 drug/control arm: {pdo}/{media}/{batch}")
        d = log_values[:, [col_idx[k] for k in drug]].mean(axis=1)
        c = log_values[:, [col_idx[k] for k in ctrl]].mean(axis=1)
        key = f"PDO{pdo}_{media}_batch{batch}"
        batch_vectors[key] = d - c
        group_counts.append({"pdo_label": f"PDO #{pdo}", "media_code": media,
                             "batch_from_title": batch, "fulvestrant_n": len(drug),
                             "dmso_n": len(ctrl), "fulvestrant_keys": drug,
                             "dmso_keys": ctrl, "batch_pairing_claim": "same label stratum; not paired culture proof"})
    for pdo in sorted(meta216["_pdo"].unique()):
        for media in ["CM", "NC"]:
            prefix = f"PDO{pdo}_{media}_batch"
            members = [v for k, v in batch_vectors.items() if k.startswith(prefix)]
            if not members:
                raise ValueError(f"missing GSE216540 group response: {prefix}")
            group_vectors[f"PDO{pdo}_{media}"] = np.mean(np.stack(members), axis=0)
    interaction_vectors: dict[str, np.ndarray] = {}
    for pdo in sorted(meta216["_pdo"].unique()):
        batch_ids = sorted({r["batch_from_title"] for r in group_counts if r["pdo_label"] == f"PDO #{pdo}"})
        per_batch = []
        for batch in batch_ids:
            cm = batch_vectors.get(f"PDO{pdo}_CM_batch{batch}")
            nc = batch_vectors.get(f"PDO{pdo}_NC_batch{batch}")
            if cm is None or nc is None:
                continue
            value = cm - nc
            interaction_vectors[f"PDO{pdo}_CM_minus_NC_batch{batch}"] = value
            per_batch.append(value)
        interaction_vectors[f"PDO{pdo}_CM_minus_NC_group_mean"] = np.mean(np.stack(per_batch), axis=0)

    def response_frame(vectors: dict[str, np.ndarray]) -> pd.DataFrame:
        frame = pd.DataFrame(vectors)
        frame.insert(0, "GeneSymbol", symbol.to_numpy())
        frame.insert(0, "Ensembl_gene_id", gene_id.to_numpy())
        return frame

    batch216 = response_frame(batch_vectors)
    group216 = response_frame(group_vectors)
    interaction216 = response_frame(interaction_vectors)
    write_parquet(batch216, out / "GSE216540_FULV_vs_DMSO_batch_log2TPM1_mean_delta.parquet")
    write_parquet(group216, out / "GSE216540_FULV_vs_DMSO_group_log2TPM1_mean_delta.parquet")
    write_parquet(interaction216, out / "GSE216540_CAF_minus_control_media_response_delta.parquet")

    top216 = []
    for contrast in group_vectors:
        v = group_vectors[contrast]
        ix = np.argsort(np.abs(v))[::-1][:50]
        for rank, j in enumerate(ix, 1):
            top216.append({"contrast": contrast, "rank_by_absolute_delta": rank,
                           "Ensembl_gene_id": gene_id.iloc[j], "GeneSymbol": symbol.iloc[j],
                           "log2_TPM_plus1_mean_delta": float(v[j])})
    pd.DataFrame(top216).to_csv(out / "GSE216540_top50_descriptive_response_genes.tsv", sep="\t", index=False)
    hallmark_path = ROOT / "data/raw/exp013/msigdb/h.all.v2023.2.Hs.symbols.gmt"
    hallmark216 = hallmark_gene_set_summary(
        group_vectors, list(symbol), "log2(TPM+1) treatment-minus-DMSO mean difference", hallmark_path)
    hallmark216.to_csv(out / "EXP050_Hallmark_GSE216540_descriptive_gene_set_means.tsv", sep="\t", index=False)

    # GSE287429: use the verified keyed author-normalized matrix as-is.
    norm = pd.read_parquet(ROOT / "data/processed/exp010/gse287429_author_normalized.parquet")
    ctx = pd.read_parquet(ROOT / "data/processed/exp010/gse287429_context.parquet")
    if norm.index.has_duplicates or norm.columns.has_duplicates or set(norm.index) != set(ctx["sample_id"]):
        raise ValueError("GSE287429 matrix/context sample identity mismatch")
    meta287 = pd.read_csv(BASE / "GSE287429/GSE287429_sample_metadata.tsv", sep="\t", dtype=str)
    if (meta287.geo_sample_accession.duplicated().any()
            or set(meta287.expression_column_key_matched_from_title) != set(norm.index)):
        raise ValueError("GSE287429 GSM/title crosswalk does not bind all normalized-matrix samples")
    ctx = ctx.set_index("sample_id").loc[norm.index]
    if not np.isfinite(norm.to_numpy(dtype=np.float64)).all():
        raise ValueError("GSE287429 author-normalized matrix has non-finite values")
    ensg_to_symbol = dict(zip(gene_id, symbol))
    target_full: dict[str, np.ndarray] = {}
    target_labels: dict[str, dict[str, Any]] = {}
    for drug in ["DOX", "BTZ"]:
        for time in [23, 47]:
            treat = ctx.index[(ctx.treatment.eq(drug)) & (ctx.time_hours.astype(int).eq(time))].tolist()
            control = ctx.index[(ctx.treatment.eq("CTR")) & (ctx.time_hours.astype(int).eq(time))].tolist()
            if len(treat) != 2 or len(control) != 2:
                raise ValueError(f"unexpected GSE287429 group size: {drug}/{time}h")
            name = f"{drug}_{time}h"
            delta = norm.loc[treat].mean(axis=0).to_numpy(float) - norm.loc[control].mean(axis=0).to_numpy(float)
            target_full[name] = delta
            target_labels[name] = {"drug": drug, "time_hours": time, "treatment_keys": treat,
                                   "control_keys": control, "treatment_n": len(treat), "control_n": len(control),
                                   "unit": "author-normalized expression mean difference; scale not established"}
    target_ens = list(map(str, norm.columns))
    target_symbols = [ensg_to_symbol.get(re.sub(r"\.\d+$", "", g)) for g in target_ens]
    full_target = pd.DataFrame({"Ensembl_gene_id": target_ens, "GeneSymbol_from_GSE216540_static_annotation": target_symbols, **target_full})
    write_parquet(full_target, out / "GSE287429_drug_vs_time_matched_control_author_scale_delta.parquet")
    top287 = []
    for contrast, v in target_full.items():
        ix = np.argsort(np.abs(v))[::-1][:50]
        for rank, j in enumerate(ix, 1):
            top287.append({"contrast": contrast, "rank_by_absolute_delta": rank,
                           "Ensembl_gene_id": target_ens[j], "GeneSymbol": target_symbols[j],
                           "author_normalized_mean_delta": float(v[j])})
    pd.DataFrame(top287).to_csv(out / "GSE287429_top50_descriptive_response_genes.tsv", sep="\t", index=False)
    hallmark287 = hallmark_gene_set_summary(
        target_full, target_symbols,
        "author-normalized treatment-minus-time-matched-control mean difference (scale unresolved)", hallmark_path)
    hallmark287.to_csv(out / "EXP050_Hallmark_GSE287429_descriptive_gene_set_means.tsv", sep="\t", index=False)

    # Freeze one unique evaluation axis: exact GSE287429 ENSEMBL -> static
    # GSE216540 symbol -> LINCS exact978 Entrez-symbol map, in LINCS order.
    exact978 = json.loads((LINCS / "continuous_contract_v2/genes.json").read_text())
    gene_info = pd.read_csv(ROOT / "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_gene_info.txt.gz",
                            sep="\t", dtype=str)
    gene_map = gene_info.loc[gene_info.pr_gene_id.isin(set(exact978)), ["pr_gene_id", "pr_gene_symbol"]]
    if gene_map.pr_gene_id.duplicated().any() or gene_map.pr_gene_symbol.duplicated().any():
        raise ValueError("LINCS exact978 Entrez-symbol map is not one-to-one")
    id_to_symbol = dict(zip(gene_map.pr_gene_id, gene_map.pr_gene_symbol))
    ordered = [(str(entrez), id_to_symbol.get(str(entrez))) for entrez in exact978]
    if len(ordered) != 978 or any(s is None for _, s in ordered) or len({s for _, s in ordered}) != 978:
        raise ValueError("LINCS exact978 ordered gene map is incomplete or duplicated")
    pert_info = pd.read_csv(ROOT / "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_pert_info.txt.gz",
                            sep="\t", dtype=str)
    identity_rows = []
    for drug, pert_id in DOSE.items():
        hit = pert_info.loc[pert_info.pert_id.eq(pert_id)]
        if len(hit) != 1 or hit.inchi_key.iloc[0] != PUBCHEM_INCHIKEY[drug]:
            raise ValueError(f"LINCS exact structure-key verification failed: {drug}/{pert_id}")
        identity_rows.append({"drug": DRUG_NAME[drug], "LINCS_pert_id": pert_id,
                              "PubChem_full_InChIKey": PUBCHEM_INCHIKEY[drug],
                              "LINCS_full_InChIKey": hit.inchi_key.iloc[0],
                              "exact_full_key_match": True,
                              "PubChem_source": "https://pubchem.ncbi.nlm.nih.gov/compound/" + DRUG_NAME[drug]})
    target_symbol_to_ens: dict[str, str] = {}
    for ens, sym in zip(target_ens, target_symbols):
        if sym:
            if sym in target_symbol_to_ens:
                raise ValueError(f"duplicate target gene symbol on GSE287429 mapping: {sym}")
            target_symbol_to_ens[sym] = ens
    g216_symbols = set(symbol)
    common = [(entrez, sym) for entrez, sym in ordered if sym in target_symbol_to_ens and sym in g216_symbols]
    if len(common) != 914:
        raise ValueError(f"frozen common-gene axis expected 914 unique genes, got {len(common)}")
    axis_symbols = [sym for _, sym in common]
    axis_payload = "\n".join(f"{entrez}\t{sym}" for entrez, sym in common) + "\n"
    axis_hash = hashlib.sha256(axis_payload.encode()).hexdigest()
    pd.DataFrame([{"axis_index": i, "LINCS_Entrez_id": entrez, "HGNC_symbol": sym,
                   "GSE287429_Ensembl_gene_id": target_symbol_to_ens[sym],
                   "mapping": "exact symbol via GSE216540 static annotation; ordered by LINCS exact978 contract"}
                  for i, (entrez, sym) in enumerate(common)]).to_csv(
                  out / "EXP050_common_gene_axis_914.tsv", sep="\t", index=False)
    pd.DataFrame(identity_rows).to_csv(out / "EXP050_drug_identity_crosswalk.tsv", sep="\t", index=False)

    # Replicate/batch consistency is summarized only as cross-gene description.
    tpm_symbol_idx = {s: i for i, s in enumerate(symbol)}
    stability_rows: list[dict[str, Any]] = []
    for pdo in ["28", "30"]:
        for media in ["CM", "NC"]:
            left = batch_vectors[f"PDO{pdo}_{media}_batch1"]
            right = batch_vectors[f"PDO{pdo}_{media}_batch2"]
            for scope, indices in [
                ("all_59428_genes", np.arange(len(symbol), dtype=int)),
                ("frozen_common_914_genes", np.array([tpm_symbol_idx[s] for s in axis_symbols], dtype=int)),
            ]:
                stats = summarize_pair(left[indices], right[indices])
                stability_rows.append({"GEO_pdo_label": f"PDO #{pdo}", "media_code": media,
                                       "batch_a": "1", "batch_b": "2", "gene_scope": scope,
                                       "interpretation": "cross-gene batch-response consistency; not biological replicate inference",
                                       **stats})
    pd.DataFrame(stability_rows).to_csv(out / "GSE216540_batch_response_consistency.tsv", sep="\t", index=False)

    # The matrix row identity follows the verified generator contract: row i is condition_id i.
    condition_dir = LINCS / "continuous_conditions_v2"
    conditions = pd.read_parquet(condition_dir / "conditions.parquet")
    delta_path = condition_dir / "condition_delta.npy"
    lincs_delta = np.load(delta_path, mmap_mode="r")
    if lincs_delta.shape != (len(conditions), 978):
        raise ValueError("LINCS condition-delta shape does not match 978-gene keyed conditions")
    if "condition_id" not in conditions or not np.array_equal(conditions.condition_id.to_numpy(), np.arange(len(conditions))):
        raise ValueError("LINCS condition rows do not preserve condition_id row order")
    treatment_contract = pd.read_parquet(LINCS / "continuous_contract_v2/treatments.parquet")
    cell_info = pd.read_csv(ROOT / "data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_cell_info.txt.gz",
                            sep="\t", dtype=str).fillna("")
    required_cell_fields = {"cell_id", "cell_type", "sample_type", "primary_site", "subtype"}
    if not required_cell_fields.issubset(cell_info.columns) or cell_info.cell_id.duplicated().any():
        raise ValueError("LINCS official cell_info fields are incomplete or cell_id is duplicated")
    breast_mask = (
        cell_info.sample_type.str.casefold().eq("tumor")
        & cell_info.primary_site.str.casefold().eq("breast")
        & cell_info.cell_type.str.casefold().eq("cell line")
    )
    breast_lines = cell_info.loc[breast_mask].sort_values("cell_id")
    if breast_lines.empty:
        raise ValueError("official LINCS cell_info has no breast tumor cell lines")
    breast_coverage_rows: list[dict[str, Any]] = []
    eligible_source_lines_by_drug: dict[str, set[str]] = {drug: set() for drug in DOSE}
    source_condition_keys = ["pert_id", "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit"]
    for cell in breast_lines.itertuples(index=False):
        for drug, pert_id in DOSE.items():
            eligible_instances = treatment_contract.loc[
                treatment_contract.cell_id.eq(cell.cell_id)
                & treatment_contract.pert_id.eq(pert_id)
                & treatment_contract.quality_status.eq("PASS")
                & treatment_contract.treatment_finite.astype(bool)
                & treatment_contract.all_controls_finite.astype(bool)
            ]
            expected_exposures = set(map(
                tuple, eligible_instances[source_condition_keys].astype("string").to_numpy()))
            line_conditions = conditions.loc[
                conditions.cell_id.eq(cell.cell_id) & conditions.pert_id.eq(pert_id)
            ]
            observed_exposures = set(map(
                tuple, line_conditions[source_condition_keys].astype("string").to_numpy()))
            if expected_exposures != observed_exposures:
                raise ValueError(f"LINCS breast-line exposure universe mismatch: {cell.cell_id}/{pert_id}")
            if expected_exposures:
                eligible_source_lines_by_drug[drug].add(str(cell.cell_id))
            breast_coverage_rows.append({
                "cell_id": str(cell.cell_id), "cell_type": cell.cell_type,
                "sample_type": cell.sample_type, "primary_site": cell.primary_site,
                "subtype": cell.subtype, "drug": DRUG_NAME[drug], "pert_id": pert_id,
                "eligible_treatment_instance_rows": int(len(eligible_instances)),
                "eligible_exposure_condition_count": int(len(expected_exposures)),
                "condition_matrix_rows": int(len(line_conditions)),
                "source_line_eligible": bool(expected_exposures),
                "eligibility_rule": "official cell_info tumor/breast/cell line; QC PASS; finite treatment and all matched controls",
            })
    breast_coverage = pd.DataFrame(breast_coverage_rows)
    breast_coverage.to_csv(out / "LINCS_breast_cancer_source_line_coverage.tsv", sep="\t", index=False)
    if eligible_source_lines_by_drug != {"DOX": {"MCF7"}, "BTZ": {"MCF7"}}:
        raise ValueError(f"breast-cancer source-line universe changed: {eligible_source_lines_by_drug}")
    qualifying = treatment_contract.loc[
        treatment_contract.quality_status.eq("PASS") & treatment_contract.treatment_finite.astype(bool)
        & treatment_contract.all_controls_finite.astype(bool)
        & treatment_contract.cell_id.eq("MCF7")
        & treatment_contract.pert_id.isin(list(DOSE.values()))]
    expected_conditions = set(zip(qualifying.pert_id, qualifying.pert_dose.astype(float),
                                  qualifying.pert_dose_unit, qualifying.pert_time.astype(str), qualifying.pert_time_unit))
    source = conditions.loc[conditions.cell_id.eq("MCF7") & conditions.pert_id.isin(list(DOSE.values()))].copy()
    observed_conditions = set(zip(source.pert_id, source.pert_dose.astype(float), source.pert_dose_unit,
                                  source.pert_time.astype(str), source.pert_time_unit))
    if expected_conditions != observed_conditions:
        raise ValueError("LINCS MCF7 source condition universe differs from QC-PASS treatment contract")
    expected_n = {"BRD-K92093830": 18, "BRD-K88510285": 10}
    if source.groupby("pert_id").size().to_dict() != expected_n:
        raise ValueError("unexpected LINCS MCF7 exact-drug condition counts")
    axis_positions = {str(entrez): i for i, entrez in enumerate(exact978)}
    axis_cols = np.array([axis_positions[entrez] for entrez, _ in common], dtype=np.int64)
    target_sym_idx = {s: i for i, s in enumerate(symbol)}
    target_gene_ens_idx = {name: np.array([target_ens.index(target_symbol_to_ens[s]) for s in axis_symbols], dtype=int)
                           for name in target_full}

    comparisons: list[dict[str, Any]] = []
    vector_rows: list[pd.DataFrame] = []
    exposure_records: list[dict[str, Any]] = []
    for row in source.sort_values(["pert_id", "pert_time", "pert_dose"]).itertuples(index=False):
        drug = "DOX" if row.pert_id == DOSE["DOX"] else "BTZ"
        src_vec = np.asarray(lincs_delta[int(row.condition_id), axis_cols], dtype=np.float64)
        exposure_records.append({"cell_id": row.cell_id, "pert_id": row.pert_id, "drug": DRUG_NAME[drug],
                                 "pert_dose": float(row.pert_dose), "pert_dose_unit": row.pert_dose_unit,
                                 "pert_time": str(row.pert_time), "pert_time_unit": row.pert_time_unit,
                                 "n_treatment_instances": int(row.n_treatment_instances),
                                 "n_control_groups": int(row.n_control_groups), "condition_id": int(row.condition_id),
                                 "QC_rule": "existing EXP-037 continuous_contract_v2 QC_PASS, finite treatment/control, MCF7"})
        for target_name, target_info in target_labels.items():
            if target_info["drug"] != drug:
                continue
            target_vec = target_full[target_name][target_gene_ens_idx[target_name]]
            stats = summarize_pair(target_vec, src_vec)
            record = {"target_dataset": "GSE287429", "target_contrast": target_name,
                      "target_drug": DRUG_NAME[drug], "target_time_hours": target_info["time_hours"],
                      "target_dose": "unknown in GEO", "target_n_treatment_samples": target_info["treatment_n"],
                      "target_n_control_samples": target_info["control_n"], "source_dataset": "LINCS GSE92742",
                      "source_cell_line": "MCF7", "source_pert_id": row.pert_id,
                      "source_dose": float(row.pert_dose), "source_dose_unit": row.pert_dose_unit,
                      "source_time": str(row.pert_time), "source_time_unit": row.pert_time_unit,
                      "source_n_treatment_instances": int(row.n_treatment_instances),
                      "source_n_matched_control_groups": int(row.n_control_groups),
                      "common_gene_axis_n": len(common), "common_gene_axis_sha256": axis_hash,
                      "exposure_relation": "same exact drug; target dose unknown; source 6/24 h vs target 23/47 h",
                      "interpretation": "descriptive gene-response similarity only; no biological-replicate inference or cross-platform amplitude comparison",
                      **stats}
            comparisons.append(record)
            vector_rows.append(pd.DataFrame({"target_contrast": target_name, "source_pert_id": row.pert_id,
                                             "source_dose": float(row.pert_dose), "source_dose_unit": row.pert_dose_unit,
                                             "source_time": str(row.pert_time), "source_time_unit": row.pert_time_unit,
                                             "gene_index": np.arange(len(common)), "LINCS_Entrez_id": [e for e, _ in common],
                                             "GeneSymbol": axis_symbols, "target_author_scale_delta": target_vec,
                                             "LINCS_condition_delta": src_vec}))
    if len(comparisons) != 56:
        raise ValueError(f"frozen comparison universe expected 56 rows, got {len(comparisons)}")
    compare_frame = pd.DataFrame(comparisons)
    exposure_frame = pd.DataFrame(exposure_records).sort_values(["drug", "pert_time", "pert_dose"])
    compare_frame.to_csv(out / "GSE287429_MCF7_all_exposure_similarity_56.tsv", sep="\t", index=False)
    exposure_frame.to_csv(out / "LINCS_MCF7_DOX_BTZ_QC_eligible_exposures.tsv", sep="\t", index=False)
    vectors = pd.concat(vector_rows, ignore_index=True)
    write_parquet(vectors, out / "GSE287429_MCF7_all_exposure_response_vectors.parquet")
    by_target = []
    for target_name, frame in compare_frame.groupby("target_contrast", sort=True):
        record = {"target_contrast": target_name, "source_condition_count": int(len(frame)),
                  "source_cells": ",".join(sorted(frame.source_cell_line.unique())),
                  "summary_scope": "fixed complete QC-eligible MCF7 exposure set; descriptive, not independent replicates"}
        for metric in ["pearson_r", "spearman_rho", "sign_agreement"]:
            vals = pd.to_numeric(frame[metric], errors="coerce").dropna()
            record[f"{metric}_min"] = float(vals.min()) if len(vals) else None
            record[f"{metric}_median"] = float(vals.median()) if len(vals) else None
            record[f"{metric}_max"] = float(vals.max()) if len(vals) else None
        by_target.append(record)
    pd.DataFrame(by_target).to_csv(out / "GSE287429_MCF7_similarity_summary_by_target.tsv", sep="\t", index=False)

    summary = {
        "GSE216540": {
            "samples": int(len(meta216)), "genes": int(len(symbol)), "groups_are_GEO_labels_not_verified_patients": True,
            "series_patient_number_crosswalk": "unresolved; #8/#10/#26 not mapped to GEO labels #28/#30/#46",
            "response_scale": "Fulvestrant mean minus DMSO mean of log2(TPM+1), within GEO-labeled group x medium x batch",
            "explicit_dose_time": "Fulvestrant 500 nM, 4 days; DMSO sample duration not individually specified",
            "#46_batch": "batch1 explicitly stated in all eight GSM titles",
            "replicate_pairing": "same group/medium/batch arm means only; not paired culture proof",
            "per_group_batch_arm_counts": group_counts,
            "group_response_vector_count": len(group_vectors), "batch_response_vector_count": len(batch_vectors),
            "CAF_minus_control_response_vector_count": len(interaction_vectors),
        },
        "GSE287429": {
            "samples": int(norm.shape[0]), "genes": int(norm.shape[1]), "mapped_to_static_HGNC_symbol": int(pd.Series(target_symbols).notna().sum()),
            "response_scale": "author-normalized expression mean difference; transform scale unknown",
            "contrasts": target_labels, "patient_or_PDO_count": "unresolved; not inferred from 12 RNA samples",
        },
        "hallmark_functional_summary": {
            "resource": "MSigDB Hallmark v2023.2.Hs", "set_count": 50,
            "method": "per-contrast mean/median response delta over exact HGNC-symbol set members; report coverage",
            "inference": "descriptive only; no pathway p-values or pathway-activity claim",
            "outputs": ["EXP050_Hallmark_GSE216540_descriptive_gene_set_means.tsv",
                        "EXP050_Hallmark_GSE287429_descriptive_gene_set_means.tsv"],
        },
        "gene_axis": {"name": "intersection of GSE287429, GSE216540 static ENSG-symbol map, and LINCS exact978",
                      "n": len(common), "LINCS_exact978_coverage_fraction": len(common) / 978,
                      "sha256_ordered_entrez_tab_symbol": axis_hash,
                      "gene_mapping": "exact identifiers/symbols only; axis ordered by LINCS genes.json; no aliases or inferred matches",
                      "GSE287429_ENSG_unique": len(target_ens), "GSE287429_ENSG_mapped_to_GSE216540": int(pd.Series(target_symbols).notna().sum()),
                      "GSE216540_symbols_intersect_exact978": int(sum(s in {z for _, z in ordered} for s in symbol))},
        "LINCS_source": {"source_lines": sorted(source.cell_id.unique().tolist()),
                         "official_breast_tumor_cell_lines": sorted(breast_lines.cell_id.tolist()),
                         "breast_cancer_line_coverage_output": "LINCS_breast_cancer_source_line_coverage.tsv",
                         "eligible_breast_source_lines_by_drug": {
                             DRUG_NAME[drug]: sorted(lines)
                             for drug, lines in eligible_source_lines_by_drug.items()},
                         "condition_count_by_drug": {
                             DRUG_NAME["DOX"] if k == DOSE["DOX"] else DRUG_NAME["BTZ"]: int(v)
                             for k, v in source.groupby("pert_id").size().items()},
                         "exposure_count": len(source), "comparison_count": len(compare_frame),
                         "interpretation": "one source line; each source condition is not an independent biological replicate"},
        "drug_identity_crosswalk": identity_rows,
        "metrics": ["Pearson and Spearman gene-axis correlations",
                    "sign agreement among finite nonzero pairs"],
        "inference_boundary": "No p-values or patient-level/generalization claims; gene axis is not independent n; no amplitude comparison across platforms; all eligible MCF7 conditions are included without target-based selection.",
        "comparisons_by_drug": {k: int(v) for k, v in compare_frame.groupby("target_drug").size().items()},
        "similarity_range": {metric: {"min": float(compare_frame[metric].min()), "median": float(compare_frame[metric].median()), "max": float(compare_frame[metric].max())}
                             for metric in ["pearson_r", "spearman_rho", "sign_agreement"]},
    }
    (out / "descriptive_response_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output_hashes = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                     for p in sorted(out.iterdir()) if p.is_file()}
    config = config_path if config_path.is_absolute() else ROOT / config_path
    result = {"experiment_id": "EXP-050", "run_id": run_id, "status": "completed_descriptive_response_and_transfer_comparison",
              "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "environment": "WSL2 Conda drugscreening-gpu",
              "config_path": str(config.relative_to(ROOT)), "config_sha256": sha256(config),
              "runner_path": "scripts/data/prepare_exp050_breast_response_transfer.py",
              "runner_sha256": sha256(Path(__file__).resolve()),
              "input_hashes": inputs, "output_hashes": output_hashes, "summary": summary}
    (out / "run_manifest.json.tmp").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "run_manifest.json.tmp").replace(out / "run_manifest.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.config, args.run_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
