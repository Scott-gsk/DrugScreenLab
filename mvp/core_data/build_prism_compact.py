"""Build the bounded MVP-001 PRISM colorectal response asset.

The official PRISM matrix is downloaded to a local temporary path and streamed
into a tiny response table containing only the frozen candidate cohort and
primary-tissue colorectal cell lines.  The full matrix is never tracked.  No
response value is used to choose the disease signature or resolve identities.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import pandas as pd


EXPERIMENT_ID = "MVP-001"
SOURCE_ID = "PRISM_REPURPOSING_PRIMARY_19Q4"
SOURCE_URL = "https://ndownloader.figshare.com/files/20237709"
SOURCE_RELEASE_URL = "https://depmap.org/repurposing/"
README_URL = "https://ndownloader.figshare.com/files/20237700"
CANDIDATE_NAMES = ("BMS-299897", "BMS-777607", "PD-0325901", "trametinib")


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def structure_mapping(prism: pd.DataFrame, lincs: pd.DataFrame) -> pd.DataFrame:
    """Resolve the frozen four-drug cohort by exact canonical structure.

    PRISM trametinib has a comma-separated duplicate SMILES string; splitting
    that field is a source-format normalization, not an alias lookup.
    """
    smiles_to_rows: dict[str, list[pd.Series]] = {}
    for _, row in lincs[lincs["pert_type"].eq("trt_cp")].iterrows():
        smiles = str(row["canonical_smiles"])
        if smiles and smiles != "-666":
            smiles_to_rows.setdefault(smiles, []).append(row)

    selected = prism[prism["name"].isin(CANDIDATE_NAMES)].copy()
    if set(selected["name"]) != set(CANDIDATE_NAMES):
        raise RuntimeError("frozen MVP candidate names are missing from PRISM metadata")
    rows: list[dict[str, str]] = []
    for _, row in selected.sort_values("name").iterrows():
        smiles = str(row["smiles"])
        components = [part.strip() for part in smiles.split(",") if part.strip()]
        matches: list[pd.Series] = []
        for component in components:
            matches.extend(smiles_to_rows.get(component, []))
        pert_ids = sorted({str(match["pert_id"]) for match in matches})
        if len(pert_ids) != 1:
            raise RuntimeError(f"candidate {row['name']} lacks a unique exact structure mapping: {pert_ids}")
        match = next(match for match in matches if str(match["pert_id"]) == pert_ids[0])
        rows.append(
            {
                "broad_id": str(row["broad_id"]),
                "drug_name": str(row["name"]),
                "column_name": str(row["column_name"]),
                "dose": str(row["dose"]),
                "screen_id": str(row["screen_id"]),
                "smiles": str(row["smiles"]),
                "pert_id": str(match["pert_id"]),
                "pert_iname": str(match["pert_iname"]),
                "inchi_key": str(match["inchi_key"]),
            }
        )
    return pd.DataFrame(rows).sort_values("drug_name").reset_index(drop=True)


def build(
    *,
    cell_info_path: Path,
    treatment_info_path: Path,
    lincs_pert_info_path: Path,
    matrix_path: Path,
    output_path: Path,
    metadata_path: Path,
) -> dict[str, object]:
    cell_info = pd.read_csv(cell_info_path, dtype=str, keep_default_na=False)
    treatment = pd.read_csv(treatment_info_path, dtype=str, keep_default_na=False)
    lincs = pd.read_csv(lincs_pert_info_path, sep="\t", dtype=str, compression="gzip", keep_default_na=False)
    mapping = structure_mapping(treatment, lincs)
    colorectal = cell_info[cell_info["primary_tissue"].str.lower().eq("colorectal")].copy()
    if len(colorectal) < 30:
        raise RuntimeError("PRISM primary-tissue colorectal cohort is unexpectedly small")

    # The matrix is wide: usecols avoids retaining unrelated 4,500-drug data.
    # The official wide matrix has an empty first header cell; pandas exposes
    # it as ``Unnamed: 0``.  Select that column by position and drug columns by
    # their immutable full `column_name` values.
    header = pd.read_csv(matrix_path, nrows=0).columns.tolist()
    missing_columns = sorted(set(mapping["column_name"]) - set(header))
    if missing_columns:
        raise RuntimeError(f"frozen PRISM candidate columns absent from matrix: {missing_columns}")
    usecols = [header.index("")] if "" in header else [0]
    usecols += [header.index(column) for column in mapping["column_name"]]
    values = pd.read_csv(matrix_path, usecols=usecols, index_col=0)
    values.index = values.index.astype(str)
    missing_lines = sorted(set(colorectal["depmap_id"]) - set(values.index))
    if missing_lines:
        raise RuntimeError(f"colorectal PRISM lines absent from response matrix: {missing_lines[:5]}")
    long_rows: list[dict[str, object]] = []
    cell_lookup = colorectal.set_index("depmap_id").to_dict("index")
    for _, drug in mapping.iterrows():
        column = drug["column_name"]
        for depmap_id in sorted(colorectal["depmap_id"]):
            raw = pd.to_numeric(values.loc[depmap_id, column], errors="coerce")
            if pd.isna(raw):
                continue
            info = cell_lookup[depmap_id]
            # Official PRISM primary log fold-change is lower-is-more-sensitive;
            # freeze the oriented score so larger means more sensitivity.
            long_rows.append(
                {
                    "study_id": SOURCE_ID,
                    "source_revision": "PRISM Repurposing 19Q4 primary replicate-collapsed",
                    "depmap_id": depmap_id,
                    "ccle_name": info["ccle_name"],
                    "primary_tissue": info["primary_tissue"],
                    "broad_id": drug["broad_id"],
                    "drug_name": drug["drug_name"],
                    "pert_id": drug["pert_id"],
                    "inchi_key": drug["inchi_key"],
                    "smiles": drug["smiles"],
                    "screen_id": drug["screen_id"],
                    "dose": float(drug["dose"]),
                    "response_raw": float(raw),
                    "response_unit": "PRISM log2 fold-change",
                    "response_direction": "lower_log2fc_more_sensitive",
                    "sensitivity_score": float(-raw),
                    "source_row_id": f"{depmap_id}|{column}",
                }
            )
    compact = pd.DataFrame(long_rows)
    if compact.empty:
        raise RuntimeError("PRISM compact response has no finite colorectal rows")
    compact = compact.sort_values(["depmap_id", "drug_name"], kind="mergesort").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compact.to_parquet(output_path, index=False)

    mapping_payload = "\n".join(
        "|".join(str(row[col]) for col in ["broad_id", "drug_name", "column_name", "pert_id", "inchi_key"])
        for _, row in mapping.iterrows()
    ) + "\n"
    metadata = {
        "format": "mvp001_prism_compact_response_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "DATA_READY",
        "source": {
            "dataset_id": SOURCE_ID,
            "release": "PRISM Repurposing 19Q4",
            "official_release_url": SOURCE_RELEASE_URL,
            "matrix_url": SOURCE_URL,
            "matrix_local_path": str(matrix_path),
            "matrix_sha256": digest(matrix_path),
            "matrix_bytes": matrix_path.stat().st_size,
            "readme_url": README_URL,
            "readme_sha256": "309310719fd96580d8f211ca12d501f6624c245b268aa29d8a5286b411f0c63f",
            "identity_metadata": [
                {
                    "dataset_id": "PRISM_primary_screen_cell_line_info",
                    "local_path": str(cell_info_path),
                    "source_file_id": "20237718",
                    "sha256": digest(cell_info_path),
                    "bytes": cell_info_path.stat().st_size,
                },
                {
                    "dataset_id": "PRISM_primary_screen_replicate_collapsed_treatment_info",
                    "local_path": str(treatment_info_path),
                    "source_file_id": "20237715",
                    "sha256": digest(treatment_info_path),
                    "bytes": treatment_info_path.stat().st_size,
                },
            ],
            "processing_rule": "stream wide primary replicate-collapsed matrix; retain frozen four candidates and primary_tissue=colorectal lines only",
        },
        "identity": {
            "candidate_mapping_rule": "exact canonical structure; PRISM trametinib comma-separated duplicate components split and deduplicated",
            "candidate_mapping_sha256": sha256(mapping_payload.encode()).hexdigest(),
            "candidate_count": int(len(mapping)),
            "candidates": mapping.to_dict("records"),
            "cell_line_unit": "depmap_id",
            "colorectal_line_count": int(colorectal["depmap_id"].nunique()),
        },
        "response": {
            "rows": int(len(compact)),
            "finite_rows": int(compact["sensitivity_score"].notna().sum()),
            "response_raw": "official PRISM replicate-collapsed log2 fold-change",
            "sensitivity_score": "-response_raw; larger means more sensitive",
            "screen_dose_rule": "one official collapsed row per frozen candidate (HTS, dose 2.5)",
            "binary_label": None,
        },
        "output": {
            "local_path": str(output_path),
            "sha256": digest(output_path),
            "schema": list(compact.columns),
        },
        "provenance": {
            "generator": "mvp/core_data/build_prism_compact.py",
            "generator_sha256": digest(Path(__file__)),
            "creation_command": "curl -L --fail -o /tmp/primaryscreenreplicatecollapsedlogfoldchange.csv https://ndownloader.figshare.com/files/20237709 && python mvp/core_data/build_prism_compact.py",
            "environment": "WSL2; conda drugscreening-gpu",
            "large_source_not_tracked": True,
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-info", type=Path, default=Path("mvp/core_data/_source_cache/primary-screen-cell-line-info.csv"))
    parser.add_argument("--treatment-info", type=Path, default=Path("mvp/core_data/_source_cache/primary-screen-replicate-collapsed-treatment-info.csv"))
    parser.add_argument("--lincs-pert-info", type=Path, default=Path("data/raw/lincs/GSE92742/GSE92742_Broad_LINCS_pert_info.txt.gz"))
    parser.add_argument("--matrix", type=Path, default=Path("/tmp/primaryscreenreplicatecollapsedlogfoldchange.csv"))
    parser.add_argument("--output", type=Path, default=Path("mvp/core_data/compact_prism_response.parquet"))
    parser.add_argument("--metadata-output", type=Path, default=Path("mvp/core_data/prism_compact_audit.json"))
    args = parser.parse_args()
    metadata = build(
        cell_info_path=args.cell_info,
        treatment_info_path=args.treatment_info,
        lincs_pert_info_path=args.lincs_pert_info,
        matrix_path=args.matrix,
        output_path=args.output,
        metadata_path=args.metadata_output,
    )
    print(json.dumps({"rows": metadata["response"]["rows"], "output": metadata["output"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
