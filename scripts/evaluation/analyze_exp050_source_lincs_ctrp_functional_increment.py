#!/usr/bin/env python3
"""Exploratory source-only grouped analysis of Hallmark response increments.

This is EXP-050 section 3.4, preregistered in configs/experiments/EXP-050.yaml.
It reads only the frozen train/validation LINCS--CTRP pairs and response-blind
LINCS condition contract. It does not read target organoid treated expression,
locked-evaluation labels, or select compounds/features using prior correlations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

try:  # Support both direct script execution and module-based inspection.
    from .analyze_exp050_source_lincs_ctrp_association import read_exact_drugs, read_frozen_cells
except ImportError:
    from analyze_exp050_source_lincs_ctrp_association import read_exact_drugs, read_frozen_cells


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_RUN_ID = "20261001_lincs_ctrp_functional_increment_v1"
EXPECTED_CONTROLS = {
    "HALLMARK_E2F_TARGETS": 53,
    "HALLMARK_G2M_CHECKPOINT": 45,
    "HALLMARK_MYC_TARGETS_V1": 20,
    "HALLMARK_UNFOLDED_PROTEIN_RESPONSE": 24,
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB": 38,
    "HALLMARK_REACTIVE_OXYGEN_SPECIES_PATHWAY": 7,
}
N_RANDOM_PANELS = 50
N_SOURCE_PERMUTATIONS = 99
N_FOLDS = 5
SEED = 20261001
RIDGE_ALPHA = 1.0
MIN_ORIGINS = 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as stream:
        temp = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temp, path)


def atomic_frame(path: Path, frame: pd.DataFrame, *, parquet: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".parquet" if parquet else ".tsv"
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=suffix) as stream:
        temp = Path(stream.name)
    try:
        if parquet:
            frame.to_parquet(temp, index=False)
        else:
            frame.to_csv(temp, sep="\t", index=False, na_rep="")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def finite_float(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def load_sets(root: Path, genes: list[str], gene_info_path: Path, gmt_path: Path):
    gene_ids = [str(value) for value in genes]
    if len(gene_ids) != 978 or len(set(gene_ids)) != 978:
        raise ValueError("Frozen LINCS gene axis must contain 978 unique Entrez IDs")
    axis = {gene_id: index for index, gene_id in enumerate(gene_ids)}
    info = pd.read_csv(
        gene_info_path,
        sep="\t",
        compression="gzip",
        usecols=["pr_gene_id", "pr_gene_symbol"],
        dtype="string",
    ).dropna(subset=["pr_gene_id", "pr_gene_symbol"])
    info["pr_gene_id"] = info["pr_gene_id"].str.strip()
    info["pr_gene_symbol"] = info["pr_gene_symbol"].str.strip()
    info = info.loc[info["pr_gene_id"].isin(axis)].drop_duplicates()
    if info["pr_gene_id"].duplicated().any() or info["pr_gene_symbol"].duplicated().any():
        raise ValueError("LINCS exact-axis Entrez-to-symbol mapping is not one-to-one")
    symbol_to_index = {
        str(row.pr_gene_symbol): axis[str(row.pr_gene_id)] for row in info.itertuples(index=False)
    }
    if len(symbol_to_index) != 978:
        raise ValueError(f"Expected 978 exact LINCS gene symbols; found {len(symbol_to_index)}")

    sets: dict[str, np.ndarray] = {}
    for line in gmt_path.read_text(encoding="utf-8").splitlines():
        fields = line.rstrip("\r\n").split("\t")
        if len(fields) < 3:
            raise ValueError("Malformed Hallmark GMT row")
        name = fields[0]
        if name in sets:
            raise ValueError(f"Duplicate Hallmark name: {name}")
        indices = sorted({symbol_to_index[symbol] for symbol in fields[2:] if symbol in symbol_to_index})
        sets[name] = np.asarray(indices, dtype=np.int32)
    if len(sets) != 50:
        raise ValueError(f"Expected 50 Hallmark sets; found {len(sets)}")
    actual_controls = {name: int(len(sets.get(name, []))) for name in EXPECTED_CONTROLS}
    if actual_controls != EXPECTED_CONTROLS:
        raise ValueError(f"Frozen control gene-set sizes differ: {actual_controls}")
    if any(len(indices) == 0 for indices in sets.values()):
        raise ValueError("A Hallmark set has no exact symbols on the frozen LINCS axis")
    return sets, actual_controls


def select_condition_rows(root: Path, pairs: pd.DataFrame, cells: pd.DataFrame, lincs_drugs: pd.DataFrame):
    base = root / "data/processed/lincs/GSE92742/continuous_conditions_v2"
    columns = [
        "condition_id", "cell_id", "pert_id", "pert_dose", "pert_dose_unit",
        "pert_time", "pert_time_unit",
    ]
    conditions = pd.read_parquet(base / "conditions.parquet", columns=columns)
    conditions["condition_id"] = pd.to_numeric(conditions["condition_id"], errors="raise").astype(np.int64)
    if conditions["condition_id"].duplicated().any():
        raise ValueError("LINCS condition_id is not unique")
    lincs_drugs = lincs_drugs[["pert_id", "inchikey"]].drop_duplicates()
    if lincs_drugs["pert_id"].duplicated().any():
        raise ValueError("One LINCS pert_id maps to more than one exact InChIKey")
    source_cells = cells[["lincs_cell_id", "PatientID"]].drop_duplicates().rename(
        columns={"lincs_cell_id": "cell_id", "PatientID": "source_origin_id"}
    )
    if source_cells["cell_id"].duplicated().any():
        raise ValueError("A frozen LINCS cell maps to more than one source origin")
    exposures = pairs[["source_origin_id", "inchikey", "dose_uM", "time_h"]].drop_duplicates()
    if exposures.duplicated(["source_origin_id", "inchikey"]).any():
        raise ValueError("Frozen source pairs contain multiple exposures for a source origin and drug")
    drug_exposures = pairs[["inchikey", "dose_uM", "time_h"]].drop_duplicates()
    if drug_exposures.duplicated(["inchikey"]).any():
        raise ValueError("Frozen v3 table does not have one metadata-selected exposure per exact drug")

    selected = conditions.merge(source_cells, on="cell_id", how="inner", validate="many_to_one")
    selected = selected.merge(lincs_drugs, on="pert_id", how="inner", validate="many_to_one")
    selected["dose_uM"] = pd.to_numeric(selected["pert_dose"], errors="coerce")
    selected["time_h"] = pd.to_numeric(selected["pert_time"], errors="coerce")
    dose_unit = selected["pert_dose_unit"].astype("string").str.strip().str.lower().str.replace("μ", "u", regex=False)
    time_unit = selected["pert_time_unit"].astype("string").str.strip().str.lower()
    selected = selected.loc[dose_unit.eq("um") & time_unit.eq("h")].copy()
    selected = selected.merge(
        exposures,
        on=["source_origin_id", "inchikey"],
        how="inner",
        suffixes=("", "_frozen"),
        validate="many_to_many",
    )
    dose_match = np.isclose(selected["dose_uM"], selected["dose_uM_frozen"], rtol=0.0, atol=1e-9)
    time_match = np.isclose(selected["time_h"], selected["time_h_frozen"], rtol=0.0, atol=1e-9)
    selected = selected.loc[dose_match & time_match].copy()
    if selected.empty:
        raise ValueError("No exact LINCS conditions join to frozen source pairs")
    selected = selected[
        ["condition_id", "cell_id", "source_origin_id", "inchikey", "dose_uM", "time_h"]
    ].drop_duplicates()
    if selected["condition_id"].duplicated().any():
        raise ValueError("A condition maps to multiple frozen source-origin/drug keys")
    return selected, conditions


def make_condition_features(
    selected: pd.DataFrame,
    conditions: pd.DataFrame,
    delta_path: Path,
    sets: dict[str, np.ndarray],
    random_groups: list[list[np.ndarray]],
):
    delta = np.load(delta_path, mmap_mode="r", allow_pickle=False)
    if delta.ndim != 2 or delta.shape[1] != 978:
        raise ValueError(f"Expected condition_delta with 978 columns; found {delta.shape}")
    ids = selected["condition_id"].to_numpy(dtype=np.int64)
    if ids.min() < 0 or ids.max() >= delta.shape[0]:
        raise ValueError("Selected condition_id falls outside condition_delta")
    matrix = np.asarray(delta[ids, :], dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("A selected 978-gene response vector contains non-finite values")

    set_names = sorted(sets)
    feature_names = [f"hallmark::{name}" for name in set_names]
    feature_matrix = np.empty((len(ids), 1 + len(set_names)), dtype=np.float32)
    feature_matrix[:, 0] = np.abs(matrix).mean(axis=1)
    for column, name in enumerate(set_names, start=1):
        feature_matrix[:, column] = matrix[:, sets[name]].mean(axis=1)

    random_names: list[str] = []
    random_matrix = np.empty((len(ids), N_RANDOM_PANELS * 44), dtype=np.float32)
    cursor = 0
    for panel_number, groups in enumerate(random_groups, start=1):
        if len(groups) != 44:
            raise ValueError("Each random panel must contain 44 matched-size groups")
        for group_number, indices in enumerate(groups, start=1):
            random_matrix[:, cursor] = matrix[:, indices].mean(axis=1)
            random_names.append(f"random_panel_{panel_number:02d}_group_{group_number:02d}")
            cursor += 1

    combined = np.column_stack([feature_matrix, random_matrix])
    combined_names = ["magnitude_978"] + feature_names + random_names
    if not np.isfinite(combined).all():
        raise ValueError("A computed Hallmark or random-group feature is non-finite")
    out = pd.DataFrame(combined, columns=combined_names)
    out.insert(0, "condition_id", ids)
    out = selected.merge(out, on="condition_id", how="inner", validate="one_to_one")
    if len(out) != len(selected):
        raise ValueError("Condition features did not round-trip to selected condition keys")

    feature_cols = combined_names
    cell = (
        out.groupby(["source_origin_id", "cell_id", "inchikey"], as_index=False, sort=True)[feature_cols]
        .median()
    )
    source = (
        cell.groupby(["source_origin_id", "inchikey"], as_index=False, sort=True)[feature_cols]
        .median()
    )
    counts = (
        selected.groupby(["source_origin_id", "inchikey"], as_index=False, sort=True)
        .agg(recomputed_n_cell_lines_lincs=("cell_id", "nunique"), recomputed_n_condition_rows=("condition_id", "nunique"))
    )
    source = source.merge(counts, on=["source_origin_id", "inchikey"], how="inner", validate="one_to_one")
    return source, feature_cols, matrix, set_names, ids


def standardize_for_fold(
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    feature_matrix: np.ndarray,
    outcomes: np.ndarray,
    drugs: np.ndarray,
):
    """Fit per-drug outcome/feature means and scales using training rows only."""
    train_drugs = sorted(set(drugs[train_rows].tolist()))
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    x_train = np.zeros((len(train_rows), feature_matrix.shape[1]), dtype=np.float64)
    x_test = np.zeros((len(test_rows), feature_matrix.shape[1]), dtype=np.float64)
    y_train = np.zeros(len(train_rows), dtype=np.float64)
    valid_test = np.zeros(len(test_rows), dtype=bool)
    y_base = np.full(len(test_rows), np.nan, dtype=np.float64)
    y_scale = np.full(len(test_rows), np.nan, dtype=np.float64)
    train_pos = {int(row): pos for pos, row in enumerate(train_rows)}
    test_pos = {int(row): pos for pos, row in enumerate(test_rows)}
    for drug in train_drugs:
        tr = train_rows[drugs[train_rows] == drug]
        te = test_rows[drugs[test_rows] == drug]
        if len(tr) < 2:
            continue
        y_mean = float(np.mean(outcomes[tr]))
        y_sd = float(np.std(outcomes[tr], ddof=1))
        if not np.isfinite(y_sd) or y_sd <= 0:
            continue
        means[drug] = y_mean
        scales[drug] = y_sd
        tr_positions = np.asarray([train_pos[int(row)] for row in tr], dtype=np.int64)
        y_train[tr_positions] = (outcomes[tr] - y_mean) / y_sd
        x_mu = feature_matrix[tr].mean(axis=0)
        x_sd = feature_matrix[tr].std(axis=0, ddof=1)
        x_sd[~np.isfinite(x_sd) | (x_sd <= 0)] = 1.0
        x_train[tr_positions] = (feature_matrix[tr] - x_mu) / x_sd
        if len(te):
            te_positions = np.asarray([test_pos[int(row)] for row in te], dtype=np.int64)
            x_test[te_positions] = (feature_matrix[te] - x_mu) / x_sd
            valid_test[te_positions] = True
            y_base[te_positions] = y_mean
            y_scale[te_positions] = y_sd
    return x_train, x_test, y_train, valid_test, y_base, y_scale, means, scales


def fit_predict(
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    feature_matrix: np.ndarray,
    outcomes: np.ndarray,
    drugs: np.ndarray,
):
    x_train, x_test, y_train, valid_test, y_base, y_scale, means, scales = standardize_for_fold(
        train_rows, test_rows, feature_matrix, outcomes, drugs
    )
    if feature_matrix.shape[1] == 0:
        pred = y_base[valid_test].copy()
    else:
        active_train_drugs = sorted(set(drugs[train_rows].tolist()).intersection(means))
        if not active_train_drugs:
            raise ValueError("No drug has a finite training-fold AAC scale")
        active_train_mask = np.isin(drugs[train_rows], active_train_drugs)
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        model.fit(x_train[active_train_mask], y_train[active_train_mask])
        residual = model.predict(x_test[valid_test])
        pred = y_base[valid_test] + residual * y_scale[valid_test]
    valid_rows = test_rows[valid_test]
    result = pd.DataFrame(
        {
            "row_id": valid_rows,
            "aac_pred": pred,
            "train_aac_mean": y_base[valid_test],
            "train_aac_sd": y_scale[valid_test],
        }
    )
    return result, {"n_train_rows": int(len(train_rows)), "n_test_rows": int(len(valid_rows)), "n_test_rows_without_train_scale": int((~valid_test).sum())}


def object_macro_mse(frame: pd.DataFrame, prediction_column: str) -> float:
    if frame.empty:
        return float("nan")
    normalized = (frame["aac"].to_numpy() - frame[prediction_column].to_numpy()) ** 2 / frame["train_aac_sd"].to_numpy() ** 2
    temp = frame[["source_origin_id"]].copy()
    temp["normalized_squared_error"] = normalized
    return float(temp.groupby("source_origin_id", sort=True)["normalized_squared_error"].mean().mean())


def scored_rows(source: pd.DataFrame, prediction_parts: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    key_columns = ["source_origin_id", "inchikey"]
    out = source[["row_id", *key_columns, "dose_uM", "time_h", "aac"]].copy()
    for name, parts in prediction_parts.items():
        merged = pd.concat(parts, ignore_index=True)
        merged = merged.rename(columns={"aac_pred": f"pred_{name}"})
        cols = ["row_id", f"pred_{name}"]
        if name == "drug_mean":
            cols.extend(["train_aac_mean", "train_aac_sd"])
        out = out.merge(merged[cols], on="row_id", how="left", validate="one_to_one")
    if "train_aac_sd" not in out:
        raise ValueError("Drug-mean arm did not provide training-fold AAC scales")
    return out


def create_random_panels(sets: dict[str, np.ndarray], seed: int) -> list[list[np.ndarray]]:
    controls = set(EXPECTED_CONTROLS)
    sizes = [len(sets[name]) for name in sorted(set(sets).difference(controls))]
    if len(sizes) != 44:
        raise ValueError(f"Expected 44 non-control Hallmark sizes; found {len(sizes)}")
    rng = np.random.default_rng(seed)
    panels: list[list[np.ndarray]] = []
    for _ in range(N_RANDOM_PANELS):
        panels.append([rng.choice(978, size=size, replace=False).astype(np.int32) for size in sizes])
    return panels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = (root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    contract = config.get("source_lincs_ctrp_functional_increment_exploratory", {})
    if args.run_id != EXPECTED_RUN_ID or contract.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("Run ID does not match the frozen EXP-050 functional-increment contract")
    if contract.get("status") != "preregistered_exploratory_pending_execution":
        raise ValueError("Functional-increment protocol is not pending execution")
    if int(contract.get("gpu_hours", -1)) != 0:
        raise ValueError("Frozen functional-increment work package must remain CPU-only")

    out_dir = root / contract["outputs"]["directory"]
    if out_dir.exists():
        raise FileExistsError(f"Run output already exists; preserve it and use an approved recovery: {out_dir}")
    input_paths = {
        "source_pairs": root / contract["inputs"]["source_pairs"],
        "conditions": root / contract["inputs"]["lincs_conditions"],
        "condition_delta": root / contract["inputs"]["lincs_deltas"],
        "gene_axis": root / contract["inputs"]["lincs_gene_axis"],
        "gene_symbols": root / contract["inputs"]["lincs_gene_symbols"],
        "hallmark_sets": root / contract["inputs"]["hallmark_sets"],
        "identity_split": root / contract["inputs"]["identity_split"],
        "config": config_path,
        "association_implementation": Path(__file__).resolve().with_name("analyze_exp050_source_lincs_ctrp_association.py"),
        "analysis_script": Path(__file__).resolve(),
    }
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required frozen input(s) missing: {missing}")

    cells = read_frozen_cells(root)
    if cells["split_role"].eq("locked_eval").any():
        raise ValueError("Frozen identity helper exposed locked_eval rows")
    _, lincs_drugs = read_exact_drugs(root)
    pairs = pd.read_parquet(
        input_paths["source_pairs"],
        columns=[
            "source_origin_id", "inchikey", "dose_uM", "time_h", "response_magnitude", "aac",
            "n_cell_lines_lincs", "n_condition_rows", "n_cell_lines_ctrp", "n_aac_rows", "n_origins",
        ],
    )
    pairs["source_origin_id"] = pairs["source_origin_id"].astype("string")
    pairs["inchikey"] = pairs["inchikey"].astype("string").str.strip().str.upper()
    if pairs.duplicated(["source_origin_id", "inchikey"]).any():
        raise ValueError("Frozen source-pair table has duplicate origin x exact-drug keys")
    if not np.isfinite(pairs[["dose_uM", "time_h", "response_magnitude", "aac"]].to_numpy(dtype=np.float64)).all():
        raise ValueError("Frozen source pairs contain non-finite dose/time/features/outcomes")
    allowed_origins = set(cells.loc[cells["split_role"].isin(["train", "val"]), "PatientID"].astype(str))
    if not set(pairs["source_origin_id"].astype(str)).issubset(allowed_origins):
        raise ValueError("Source-pair table contains an origin outside frozen train/validation identities")
    drug_coverage = pairs.groupby("inchikey")["source_origin_id"].nunique()
    eligible_keys = sorted(drug_coverage.loc[drug_coverage.ge(MIN_ORIGINS)].index.astype(str))
    pairs = pairs.loc[pairs["inchikey"].isin(eligible_keys)].copy().sort_values(
        ["source_origin_id", "inchikey"], kind="mergesort"
    ).reset_index(drop=True)
    pairs.insert(0, "row_id", np.arange(len(pairs), dtype=np.int64))
    if pairs.empty or pairs["source_origin_id"].nunique() < N_FOLDS:
        raise ValueError("At least five eligible source origins are required for GroupKFold")
    if not set(pairs["inchikey"].astype(str)).issubset(set(lincs_drugs["inchikey"].astype(str))):
        raise ValueError("A frozen exact InChIKey is absent from the LINCS perturbagen identity map")

    axis_genes = json.loads(input_paths["gene_axis"].read_text(encoding="utf-8"))
    sets, control_sizes = load_sets(root, axis_genes, input_paths["gene_symbols"], input_paths["hallmark_sets"])
    random_groups = create_random_panels(sets, SEED)
    selected, conditions = select_condition_rows(root, pairs, cells, lincs_drugs)
    if int(conditions["condition_id"].max()) >= np.load(input_paths["condition_delta"], mmap_mode="r", allow_pickle=False).shape[0]:
        raise ValueError("Condition metadata exceeds the frozen delta matrix row count")
    source_features, feature_cols, condition_matrix, set_names, condition_ids = make_condition_features(
        selected, conditions, input_paths["condition_delta"], sets, random_groups
    )
    source = pairs.merge(source_features, on=["source_origin_id", "inchikey"], how="left", validate="one_to_one")
    if source[feature_cols].isna().any().any():
        missing_pairs = int(source[feature_cols].isna().any(axis=1).sum())
        raise ValueError(f"Frozen source-pair rows missing selected LINCS features: {missing_pairs}")
    if not source["n_cell_lines_lincs"].eq(source["recomputed_n_cell_lines_lincs"]).all():
        raise ValueError("LINCS contributing cell-line counts differ from the frozen v3 table")
    if not source["n_condition_rows"].eq(source["recomputed_n_condition_rows"]).all():
        raise ValueError("LINCS condition-row counts differ from the frozen v3 table")
    magnitude_diff = np.abs(source["response_magnitude"].to_numpy(dtype=np.float64) - source["magnitude_978"].to_numpy(dtype=np.float64))
    magnitude_max_abs_diff = float(magnitude_diff.max())
    if magnitude_max_abs_diff > 1e-6:
        raise ValueError(f"Recomputed magnitude does not reproduce v3; max abs difference={magnitude_max_abs_diff}")
    if len(source) != len(pairs):
        raise ValueError("Feature join changed the frozen source-pair denominator")

    # The six control Hallmarks stay aligned in the nuisance arm. The 44 remaining
    # Hallmark features are the registered functional increment and permutation block.
    control_names = sorted(EXPECTED_CONTROLS)
    all_hallmark_names = sorted(sets)
    incremental_names = [name for name in all_hallmark_names if name not in EXPECTED_CONTROLS]
    base_columns = ["magnitude_978", *[f"hallmark::{name}" for name in control_names]]
    full_columns = ["magnitude_978", *[f"hallmark::{name}" for name in all_hallmark_names]]
    incremental_columns = [f"hallmark::{name}" for name in incremental_names]
    if len(incremental_names) != 44 or len(full_columns) != 51 or len(base_columns) != 7:
        raise ValueError("Frozen Hallmark and nuisance feature dimensions do not match the contract")

    y = source["aac"].to_numpy(dtype=np.float64)
    drugs = source["inchikey"].astype(str).to_numpy()
    origins = source["source_origin_id"].astype(str).to_numpy()
    base_matrix = source[base_columns].to_numpy(dtype=np.float64)
    full_matrix = source[full_columns].to_numpy(dtype=np.float64)
    magnitude_matrix = source[["magnitude_978"]].to_numpy(dtype=np.float64)
    feature_map = {
        (str(row["source_origin_id"]), str(row["inchikey"])): source.loc[index, incremental_columns].to_numpy(dtype=np.float64)
        for index, row in source[["source_origin_id", "inchikey"]].iterrows()
    }

    splitter = GroupKFold(n_splits=N_FOLDS)
    splits = list(splitter.split(source, y, groups=origins))
    if len(splits) != N_FOLDS:
        raise ValueError("GroupKFold did not produce five folds")
    seen_test_rows: list[int] = []
    fold_origins: list[dict[str, Any]] = []
    for fold, (train_rows, test_rows) in enumerate(splits, start=1):
        train_origins = set(origins[train_rows])
        test_origins = set(origins[test_rows])
        if train_origins.intersection(test_origins):
            raise ValueError("Source origins leak across a grouped fold")
        seen_test_rows.extend(test_rows.tolist())
        fold_origins.append({"fold": fold, "train_origins": sorted(train_origins), "test_origins": sorted(test_origins)})
    if sorted(seen_test_rows) != list(range(len(source))):
        raise ValueError("Five-fold test partitions do not cover every eligible source pair exactly once")

    # Reserve the output directory only after identities, joins, and magnitude parity pass.
    out_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "run_id": args.run_id,
        "status": "RUNNING_CPU_ONLY",
        "started_at_utc": started_at,
        "execution_environment": "WSL2 Conda drugscreening-gpu",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "supervision": "gpt-6.1-sol/medium directed execution of the frozen source functional-increment analysis",
        "protocol_source": "configs/experiments/EXP-050.yaml:source_lincs_ctrp_functional_increment_exploratory",
        "exploratory_only_reason": "Prior CTRPv2 response output was inadvertently exposed before the analysis protocol was frozen; this result is not independent confirmation or an unbiased generalization estimate.",
        "forbidden_inputs_read": {
            "target_pdo_expression": False,
            "target_treated_expression": False,
            "locked_eval_labels": False,
            "outer_test_labels": False,
            "prior_v3_rho_used_for_selection": False,
            "dose_or_time_reselected_from_aac": False,
        },
        "resource_use": {"gpu_hours": 0.0, "cpu_only": True},
        "execution_deviation": "A prior Windows PowerShell Import-Csv attempt used incorrect column names on the already-aggregated exploratory per-drug TSV and yielded no valid counts. It did not process raw labels or locked_eval; all reported calculations here run in WSL2 Conda.",
        "inputs": {},
        "outputs": {},
    }
    for label, path in input_paths.items():
        manifest["inputs"][label] = {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    atomic_json(out_dir / "run_manifest.json", manifest)

    try:
        prediction_parts: dict[str, list[pd.DataFrame]] = {
            "drug_mean": [], "magnitude": [], "nuisance": [], "full": []
        }
        fold_coverage_rows: list[dict[str, Any]] = []
        random_panel_rows: list[dict[str, Any]] = []
        oof_core = source[["row_id", "source_origin_id", "inchikey", "dose_uM", "time_h", "aac"]].copy()
        oof_core["fold"] = -1
        for panel_number in range(1, N_RANDOM_PANELS + 1):
            prediction_parts[f"random_{panel_number:02d}"] = []

        for fold, (train_rows, test_rows) in enumerate(splits, start=1):
            arms = {
                "drug_mean": np.empty((len(source), 0), dtype=np.float64),
                "magnitude": magnitude_matrix,
                "nuisance": base_matrix,
                "full": full_matrix,
            }
            fold_results: dict[str, pd.DataFrame] = {}
            for arm_name, matrix_for_arm in arms.items():
                result, counts = fit_predict(train_rows, test_rows, matrix_for_arm, y, drugs)
                result["fold"] = fold
                prediction_parts[arm_name].append(result)
                fold_results[arm_name] = result
                if arm_name == "drug_mean":
                    oof_core.loc[test_rows, "fold"] = fold
            nuisance_lookup = fold_results["nuisance"].set_index("row_id")
            full_lookup = fold_results["full"].set_index("row_id")
            mean_lookup = fold_results["drug_mean"].set_index("row_id")
            magnitude_lookup = fold_results["magnitude"].set_index("row_id")
            for key in sorted(set(drugs[test_rows])):
                train_for_drug = train_rows[drugs[train_rows] == key]
                test_for_drug = test_rows[drugs[test_rows] == key]
                train_sd = float(np.std(y[train_for_drug], ddof=1)) if len(train_for_drug) >= 2 else float("nan")
                fold_coverage_rows.append(
                    {
                        "fold": fold,
                        "inchikey": key,
                        "n_train_source_origins": int(pd.Series(origins[train_rows][drugs[train_rows] == key]).nunique()),
                        "n_test_source_origins": int(pd.Series(origins[test_for_drug]).nunique()),
                        "n_train_rows": int(len(train_for_drug)),
                        "n_test_rows": int(len(test_for_drug)),
                        "training_fold_aac_sample_sd": train_sd if np.isfinite(train_sd) else None,
                        "scored": bool(len(train_for_drug) >= 2 and np.isfinite(train_sd) and train_sd > 0),
                    }
                )
            for panel_number in range(1, N_RANDOM_PANELS + 1):
                random_cols = [f"random_panel_{panel_number:02d}_group_{group:02d}" for group in range(1, 45)]
                random_matrix_panel = source[[*base_columns, *random_cols]].to_numpy(dtype=np.float64)
                result, _ = fit_predict(train_rows, test_rows, random_matrix_panel, y, drugs)
                result["fold"] = fold
                prediction_parts[f"random_{panel_number:02d}"].append(result)

        oof = scored_rows(source, prediction_parts)
        oof = oof.merge(oof_core[["row_id", "fold"]], on="row_id", how="left", validate="one_to_one")
        score_names = ["drug_mean", "magnitude", "nuisance", "full"]
        for arm in score_names:
            pred_col = f"pred_{arm}"
            scored = oof.dropna(subset=[pred_col, "train_aac_sd"]).copy()
            arm_mse = object_macro_mse(scored, pred_col)
            if not np.isfinite(arm_mse):
                raise ValueError(f"No finite grouped out-of-fold score for arm {arm}")
            manifest.setdefault("arm_scores", {})[arm] = {
                "object_macro_normalized_mse": arm_mse,
                "n_rows": int(len(scored)),
                "n_source_origins": int(scored["source_origin_id"].nunique()),
                "interpretation": "Lower is better; rows are normalized by each drug's training-fold AAC SD squared, averaged within source origin, then equally across source origins.",
            }
        if not oof["pred_nuisance"].notna().eq(oof["pred_full"].notna()).all():
            raise ValueError("Nuisance and full Hallmark arms do not share the same OOF key denominator")
        primary_increment = manifest["arm_scores"]["nuisance"]["object_macro_normalized_mse"] - manifest["arm_scores"]["full"]["object_macro_normalized_mse"]

        # Same nuisance baseline plus 44 independent, exact-size random gene groups.
        for panel_number in range(1, N_RANDOM_PANELS + 1):
            pred = pd.concat(prediction_parts[f"random_{panel_number:02d}"], ignore_index=True)
            panel_oof = source[["row_id", "source_origin_id", "inchikey", "aac"]].merge(
                pred[["row_id", "aac_pred", "train_aac_sd"]].rename(columns={"aac_pred": "pred_random"}),
                on="row_id", how="inner", validate="one_to_one",
            )
            random_panel_rows.append(
                {
                    "panel": panel_number,
                    "object_macro_normalized_mse": object_macro_mse(panel_oof, "pred_random"),
                    "paired_nuisance_increment": manifest["arm_scores"]["nuisance"]["object_macro_normalized_mse"] - object_macro_mse(panel_oof, "pred_random"),
                    "n_rows": int(len(panel_oof)),
                    "n_source_origins": int(panel_oof["source_origin_id"].nunique()),
                    "gene_group_sizes": ",".join(str(len(group)) for group in random_groups[panel_number - 1]),
                }
            )
        random_scores = pd.DataFrame(random_panel_rows)

        # Preserve the 44 added Hallmarks as one source-origin block. The six
        # preregistered nuisance Hallmarks and response magnitude stay paired.
        perm_rng = np.random.default_rng(SEED + 1)
        permutation_rows: list[dict[str, Any]] = []
        for permutation in range(1, N_SOURCE_PERMUTATIONS + 1):
            paired_test_parts: list[pd.DataFrame] = []
            permutation_nuisance_parts: list[pd.DataFrame] = []
            permutation_full_parts: list[pd.DataFrame] = []
            n_unmatched_train = 0
            n_unmatched_test = 0
            for fold, (train_rows, test_rows) in enumerate(splits, start=1):
                train_origins = np.asarray(sorted(set(origins[train_rows])), dtype=object)
                test_origins = np.asarray(sorted(set(origins[test_rows])), dtype=object)
                train_map = dict(zip(train_origins.tolist(), perm_rng.permutation(train_origins).tolist()))
                test_map = dict(zip(test_origins.tolist(), perm_rng.permutation(test_origins).tolist()))

                def mapped_rows(rows: np.ndarray, mapping: dict[str, str]):
                    kept: list[int] = []
                    mapped_features: list[np.ndarray] = []
                    for row_id in rows.tolist():
                        recipient = str(origins[row_id])
                        donor = mapping[recipient]
                        block = feature_map.get((donor, str(drugs[row_id])))
                        if block is None:
                            continue
                        kept.append(int(row_id))
                        mapped_features.append(block)
                    return np.asarray(kept, dtype=np.int64), (
                        np.vstack(mapped_features) if mapped_features else np.empty((0, 44), dtype=np.float64)
                    )

                train_kept, train_added = mapped_rows(train_rows, train_map)
                test_kept, test_added = mapped_rows(test_rows, test_map)
                n_unmatched_train += int(len(train_rows) - len(train_kept))
                n_unmatched_test += int(len(test_rows) - len(test_kept))
                if len(train_kept) == 0 or len(test_kept) == 0:
                    continue
                nuisance_pair, _ = fit_predict(train_kept, test_kept, base_matrix, y, drugs)
                permuted_matrix = np.full((len(source), base_matrix.shape[1] + 44), np.nan, dtype=np.float64)
                permuted_matrix[train_kept] = np.column_stack([base_matrix[train_kept], train_added])
                permuted_matrix[test_kept] = np.column_stack([base_matrix[test_kept], test_added])
                perm_full, _ = fit_predict(train_kept, test_kept, permuted_matrix, y, drugs)
                common = nuisance_pair[["row_id", "aac_pred", "train_aac_sd"]].merge(
                    perm_full[["row_id", "aac_pred"]], on="row_id", suffixes=("_nuisance", "_full"),
                    how="inner", validate="one_to_one",
                )
                common = common.merge(source[["row_id", "source_origin_id", "inchikey", "aac"]], on="row_id", how="inner", validate="one_to_one")
                if common.empty:
                    continue
                common["fold"] = fold
                paired_test_parts.append(common)
                permutation_nuisance_parts.append(common.rename(columns={"aac_pred_nuisance": "aac_pred"})[["row_id", "source_origin_id", "inchikey", "aac", "train_aac_sd", "aac_pred"]])
                permutation_full_parts.append(common.rename(columns={"aac_pred_full": "aac_pred"})[["row_id", "source_origin_id", "inchikey", "aac", "train_aac_sd", "aac_pred"]])
            if paired_test_parts:
                paired_all = pd.concat(paired_test_parts, ignore_index=True)
                n_mse = object_macro_mse(pd.concat(permutation_nuisance_parts, ignore_index=True), "aac_pred")
                f_mse = object_macro_mse(pd.concat(permutation_full_parts, ignore_index=True), "aac_pred")
                permutation_rows.append(
                    {
                        "permutation": permutation,
                        "nuisance_object_macro_normalized_mse": n_mse,
                        "permuted_full_object_macro_normalized_mse": f_mse,
                        "paired_increment_nuisance_minus_full": n_mse - f_mse,
                        "n_rows": int(paired_all["row_id"].nunique()),
                        "n_source_origins": int(paired_all["source_origin_id"].nunique()),
                        "n_drugs": int(paired_all["inchikey"].nunique()),
                        "unmatched_train_rows_dropped": n_unmatched_train,
                        "unmatched_test_rows_dropped": n_unmatched_test,
                    }
                )
        permutation_scores = pd.DataFrame(permutation_rows)
        if len(permutation_scores) != N_SOURCE_PERMUTATIONS:
            raise ValueError(f"Expected 99 complete source-object permutations; got {len(permutation_scores)}")

        fold_cov = pd.DataFrame(fold_coverage_rows).drop_duplicates().sort_values(["fold", "inchikey"]).reset_index(drop=True)
        oof_output_columns = [
            "row_id", "fold", "source_origin_id", "inchikey", "dose_uM", "time_h", "aac",
            "train_aac_mean", "train_aac_sd", "pred_drug_mean", "pred_magnitude", "pred_nuisance", "pred_full",
        ]
        oof_output = oof[oof_output_columns].sort_values(["fold", "source_origin_id", "inchikey"]).reset_index(drop=True)
        random_scores = random_scores.sort_values("panel").reset_index(drop=True)
        permutation_scores = permutation_scores.sort_values("permutation").reset_index(drop=True)

        summary = {
            "run_id": args.run_id,
            "status": "COMPLETED_EXPLORATORY",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "research_scope": "Source-domain CTRPv2 AAC association only; no PDO efficacy, prediction, drug ranking, independent confirmation, or generalization claim.",
            "coverage": {
                "source_pair_rows_after_n_ge_10_coverage_filter": int(len(source)),
                "unique_source_origins": int(source["source_origin_id"].nunique()),
                "unique_exact_drugs": int(source["inchikey"].nunique()),
                "selected_condition_rows": int(len(selected)),
                "condition_delta_dtype": str(np.load(input_paths["condition_delta"], mmap_mode="r", allow_pickle=False).dtype),
                "hallmark_sets": int(len(sets)),
                "exact_axis_gene_count": 978,
                "random_panels": N_RANDOM_PANELS,
                "source_object_permutations": N_SOURCE_PERMUTATIONS,
                "grouped_folds": N_FOLDS,
                "locked_eval_rows_read": 0,
                "target_treated_expression_rows_read": 0,
                "magnitude_max_absolute_difference_vs_v3": magnitude_max_abs_diff,
                "nuisance_hallmark_sizes": control_sizes,
                "incremental_hallmark_count": len(incremental_names),
                "fold_source_origin_counts": [
                    {"fold": item["fold"], "n_train": len(item["train_origins"]), "n_test": len(item["test_origins"])}
                    for item in fold_origins
                ],
            },
            "protocol": {
                "unit": "DepMap PatientID source-origin x exact full InChIKey selected LINCS dose/time; source-origin does not mean PDO patient.",
                "cv": "Deterministic 5-fold GroupKFold by source origin; no origin split across train/test.",
                "model": f"Ridge(alpha={RIDGE_ALPHA}), fit on training-fold AAC residuals standardized by training-fold per-drug sample SD; response features centered/scaled within exact drug using training rows only.",
                "endpoint": "OOF normalized squared AAC error; row errors divide by training-fold per-drug AAC sample SD squared, mean across drugs within each source origin, then equal mean across source origins. Lower is better.",
                "primary_increment": "Nuisance-control error minus full 50-Hallmark error; positive favors Hallmark features.",
                "random_control": "Each of 50 panels adds 44 random gene-group means to the same magnitude plus six predeclared Hallmark nuisance features; each group matches one non-control Hallmark size and samples genes without replacement within group.",
                "source_object_permutation": "Within each fold, permute the 44 added Hallmark features as a single block among training origins and separately among held-out origins; preserve exact drug keys and nuisance features; drop unmatched donor-origin/drug pairs and report denominators.",
                "label_status": "Previously exposed CTRPv2 outcomes make the result exploratory; no p value, confidence interval, independent confirmation, target efficacy, or ranking.",
                "selection": "Drug inclusion uses source-origin coverage n>=10 only; no prior per-drug rho was read for inclusion or feature selection; dose/time is reused from v3 metadata-only selection.",
            },
            "arm_scores": manifest["arm_scores"],
            "primary_comparison": {
                "nuisance_object_macro_normalized_mse": manifest["arm_scores"]["nuisance"]["object_macro_normalized_mse"],
                "full_hallmark_object_macro_normalized_mse": manifest["arm_scores"]["full"]["object_macro_normalized_mse"],
                "increment_nuisance_minus_full": primary_increment,
                "direction": "positive favors the full Hallmark arm; negative favors nuisance controls",
                "importance": "Descriptive point estimate on source-domain grouped folds only; no uncertainty or confirmatory inference is claimed.",
            },
            "random_control_distribution": {
                "minimum_mse": float(random_scores["object_macro_normalized_mse"].min()),
                "median_mse": float(random_scores["object_macro_normalized_mse"].median()),
                "maximum_mse": float(random_scores["object_macro_normalized_mse"].max()),
                "full_hallmark_mse_below_random_panel_count": int((random_scores["object_macro_normalized_mse"] > manifest["arm_scores"]["full"]["object_macro_normalized_mse"]).sum()),
                "panel_count": int(len(random_scores)),
                "interpretation": "Descriptive matched-size reference distribution, not a p-value or confidence interval.",
            },
            "source_object_permutation_distribution": {
                "increment_minimum": float(permutation_scores["paired_increment_nuisance_minus_full"].min()),
                "increment_median": float(permutation_scores["paired_increment_nuisance_minus_full"].median()),
                "increment_maximum": float(permutation_scores["paired_increment_nuisance_minus_full"].max()),
                "permutation_count": int(len(permutation_scores)),
                "median_retained_rows": float(permutation_scores["n_rows"].median()),
                "median_retained_source_origins": float(permutation_scores["n_source_origins"].median()),
                "interpretation": "Descriptive block-permutation reference; no significance claim.",
            },
            "interpretation_boundary": [
                "CTRPv2 AAC is an existing source-cell-line drug sensitivity label; it is not a PDO endpoint.",
                "Selected dose/time varies across exact drugs and is fixed from the earlier coverage-only exposure selection.",
                "The result cannot establish drug efficacy, clinical benefit, or PDO patient response.",
                "Hallmark signed means are set summaries of source LINCS perturbation vectors, not pathway activity tests.",
            ],
        }
        outputs = {
            "summary.json": out_dir / "summary.json",
            "oof_predictions.parquet": out_dir / "oof_predictions.parquet",
            "random_panel_scores.tsv": out_dir / "random_panel_scores.tsv",
            "source_object_permutation_scores.tsv": out_dir / "source_object_permutation_scores.tsv",
            "per_drug_fold_coverage.tsv": out_dir / "per_drug_fold_coverage.tsv",
        }
        atomic_frame(outputs["oof_predictions.parquet"], oof_output, parquet=True)
        atomic_frame(outputs["random_panel_scores.tsv"], random_scores)
        atomic_frame(outputs["source_object_permutation_scores.tsv"], permutation_scores)
        atomic_frame(outputs["per_drug_fold_coverage.tsv"], fold_cov)
        atomic_json(outputs["summary.json"], summary)
        manifest["status"] = "COMPLETED_EXPLORATORY"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["arm_scores"] = summary["arm_scores"]
        manifest["primary_increment_nuisance_minus_full"] = primary_increment
        for name, path in outputs.items():
            manifest["outputs"][name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        atomic_json(out_dir / "run_manifest.json", manifest)
    except Exception as exc:
        manifest["status"] = "FAILED_NO_ACCEPTED_RESULT"
        manifest["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        atomic_json(out_dir / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
