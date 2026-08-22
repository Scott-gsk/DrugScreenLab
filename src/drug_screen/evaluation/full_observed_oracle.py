"""Response-blind Full Observed Oracle from the GSE92742 exact978 cache.

This module never uses the XPert 78k processed h5ad as an Oracle source.
PRISM response values are not read while identities, dose/time, or controls
are being frozen.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from drug_screen.evaluation.phase1_prism import reversal_score, spearman
from drug_screen.evaluation.xpert_broad import rank_metrics

NULL_SEED = 20260813
NULL_REPEATS = 256
CRC_EXACT_CONTEXTS = (
    "CL34",
    "HCT116",
    "HT29",
    "LOVO",
    "RKO",
    "SNUC4",
    "SNUC5",
    "SW480",
    "SW620",
    "SW948",
)
CANONICAL_CONDITIONS = {
    "10uM_6h": {"dose_um": 10.0, "time_h": 6.0},
    "10uM_24h": {"dose_um": 10.0, "time_h": 24.0},
}
EXPECTED_CACHE_SHAPE = (1_319_138, 978)
EXPECTED_CACHE_SHA256 = "04b8bb746a61ba4992e49566315327023783ec1c0448da2a9e263e0881281733"
ORDERED_GENE_IDS_SHA256 = "b4e2fca877c5cfdcc1c712ad0fd67e97a88b6f7566b013e4bab065f699ebb623"
SUPPORT_BINS = (
    ("n_lt_20", 0, 20),
    ("n_20_99", 20, 100),
    ("n_100_499", 100, 500),
    ("n_ge_500", 500, 10**9),
)


def file_sha256(path) -> str:
    digest = sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_landmark_gene_ids(
    gene_info: pd.DataFrame,
    *,
    gctx_row_ids: Iterable[str] | None = None,
) -> list[str]:
    """Return landmark gene IDs in GCTX source order when available.

    The registered gene-order digest is computed from ``GCTX ROW/id`` filtered
    by ``gene_info.pr_is_lm=1``.  Table order in ``gene_info`` is not that
    universe and must not be hashed as a substitute.
    """
    flags = dict(zip(gene_info["pr_gene_id"].astype(str), gene_info["pr_is_lm"].astype(str)))
    if gctx_row_ids is None:
        ordered = [gene_id for gene_id, flag in flags.items() if flag == "1"]
    else:
        ordered = [str(gene_id) for gene_id in gctx_row_ids if flags.get(str(gene_id)) == "1"]
    if len(ordered) != 978:
        raise ValueError(f"expected 978 landmark genes, found {len(ordered)}")
    return ordered


def gene_order_digest(gene_ids: Iterable[str]) -> str:
    digest = sha256()
    for gene_id in gene_ids:
        digest.update(str(gene_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def response_blind_eligible_pert_ids(
    bridge: pd.DataFrame,
    registry: Mapping[str, Any],
) -> set[str]:
    """Formal LINCS-PRISM identity ∩ registry Broad-inference eligibility.

    The function inspects identity columns only.  Response values are neither
    required nor used.
    """
    required = {"lincs_pert_id", "match_status"}
    missing = sorted(required.difference(bridge.columns))
    if missing:
        raise ValueError(f"identity bridge missing columns: {missing}")
    matched = (
        bridge.loc[bridge["match_status"].eq("MATCHED_IDENTITY"), "lincs_pert_id"]
        .dropna()
        .astype(str)
    )
    registry_eligible = {
        str(row["pert_id"])
        for row in registry.get("drugs", [])
        if isinstance(row, Mapping) and bool(row.get("broad_inference_eligible"))
    }
    return set(matched) & registry_eligible


def select_canonical_instances(
    instances: pd.DataFrame,
    *,
    contexts: Iterable[str],
    pert_ids: Iterable[str],
    dose_um: float,
    time_h: float,
    tolerance: float = 1e-5,
) -> pd.DataFrame:
    required = {
        "pert_type",
        "pert_id",
        "pert_dose",
        "pert_dose_unit",
        "pert_time",
        "pert_time_unit",
        "cell_id",
    }
    missing = sorted(required.difference(instances.columns))
    if missing:
        raise ValueError(f"instances missing columns: {missing}")
    dose = pd.to_numeric(instances["pert_dose"], errors="coerce")
    time = pd.to_numeric(instances["pert_time"], errors="coerce")
    mask = (
        instances["pert_type"].astype(str).eq("trt_cp")
        & instances["cell_id"].astype(str).isin(set(map(str, contexts)))
        & instances["pert_id"].astype(str).isin(set(map(str, pert_ids)))
        & instances["pert_dose_unit"].astype(str).str.lower().eq("um")
        & instances["pert_time_unit"].astype(str).str.lower().eq("h")
        & np.isclose(dose, dose_um, rtol=tolerance, atol=tolerance)
        & np.isclose(time, time_h, rtol=tolerance, atol=tolerance)
    )
    selected = instances.loc[mask].copy()
    if "_cache_row" not in selected.columns:
        selected["_cache_row"] = instances.index.to_numpy()[mask.to_numpy()]
    return selected.reset_index(drop=True)


def _match_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame[["rna_plate", "cell_id", "pert_time", "pert_time_unit"]]
        .astype(str)
        .agg("||".join, axis=1)
    )


def assign_matched_controls(
    treatments: pd.DataFrame,
    instances: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach same-plate, same-cell, same-time controls.

    Preference: ``ctl_vehicle`` if any exist on the match key, else
    ``ctl_untrt``.  Pairs without a matched control are dropped and counted.
    """
    if treatments.empty:
        return treatments.copy(), {
            "dropped_unmatched_treatment_rows": 0,
            "dropped_unmatched_unique_pairs": 0,
            "vehicle_matched_rows": 0,
            "untrt_matched_rows": 0,
        }
    working = treatments.copy()
    working["match_key"] = _match_key(working)
    controls = instances.loc[instances["pert_type"].astype(str).isin({"ctl_vehicle", "ctl_untrt"})].copy()
    if controls.empty:
        dropped_pairs = int(working[["cell_id", "pert_id"]].drop_duplicates().shape[0])
        return working.iloc[0:0].copy(), {
            "dropped_unmatched_treatment_rows": int(len(working)),
            "dropped_unmatched_unique_pairs": dropped_pairs,
            "vehicle_matched_rows": 0,
            "untrt_matched_rows": 0,
        }
    if "_cache_row" not in controls.columns:
        controls = controls.copy()
        controls["_cache_row"] = controls.index.to_numpy()
    controls["match_key"] = _match_key(controls)
    by_key: dict[str, dict[str, list[int]]] = {}
    for key, group in controls.groupby("match_key", sort=False):
        types = {
            str(pert_type): group.loc[group["pert_type"].astype(str).eq(pert_type), "_cache_row"]
            .astype(np.int64)
            .tolist()
            for pert_type in ("ctl_vehicle", "ctl_untrt")
        }
        by_key[str(key)] = types

    control_types: list[str] = []
    control_rows: list[list[int]] = []
    keep: list[bool] = []
    for key in working["match_key"].astype(str):
        available = by_key.get(key, {"ctl_vehicle": [], "ctl_untrt": []})
        if available["ctl_vehicle"]:
            control_types.append("ctl_vehicle")
            control_rows.append(available["ctl_vehicle"])
            keep.append(True)
        elif available["ctl_untrt"]:
            control_types.append("ctl_untrt")
            control_rows.append(available["ctl_untrt"])
            keep.append(True)
        else:
            control_types.append("")
            control_rows.append([])
            keep.append(False)
    working["control_type"] = control_types
    working["control_cache_rows"] = control_rows
    keep_mask = np.asarray(keep, dtype=bool)
    dropped = working.loc[~keep_mask]
    matched = working.loc[keep_mask].copy().reset_index(drop=True)
    audit = {
        "dropped_unmatched_treatment_rows": int((~keep_mask).sum()),
        "dropped_unmatched_unique_pairs": int(dropped[["cell_id", "pert_id"]].drop_duplicates().shape[0])
        if not dropped.empty
        else 0,
        "vehicle_matched_rows": int((matched["control_type"] == "ctl_vehicle").sum()) if not matched.empty else 0,
        "untrt_matched_rows": int((matched["control_type"] == "ctl_untrt").sum()) if not matched.empty else 0,
    }
    return matched, audit


def coverage_by_condition(
    instances: pd.DataFrame,
    *,
    contexts: Iterable[str],
    pert_ids: Iterable[str],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, spec in CANONICAL_CONDITIONS.items():
        selected = select_canonical_instances(
            instances,
            contexts=contexts,
            pert_ids=pert_ids,
            dose_um=float(spec["dose_um"]),
            time_h=float(spec["time_h"]),
        )
        matched, audit = assign_matched_controls(selected, instances)
        pairs = (
            matched[["cell_id", "pert_id"]].drop_duplicates()
            if not matched.empty
            else pd.DataFrame(columns=["cell_id", "pert_id"])
        )
        per_line = (
            pairs.groupby("cell_id", sort=True)["pert_id"].nunique().astype(int).to_dict()
            if not pairs.empty
            else {}
        )
        report[name] = {
            "dose_um": float(spec["dose_um"]),
            "time_h": float(spec["time_h"]),
            "treatment_rows_before_control": int(len(selected)),
            "unique_pairs_before_control": int(selected[["cell_id", "pert_id"]].drop_duplicates().shape[0])
            if not selected.empty
            else 0,
            "unique_pairs_with_matched_control": int(len(pairs)),
            "unique_contexts_with_matched_control": int(pairs["cell_id"].nunique()) if not pairs.empty else 0,
            "unique_compounds_with_matched_control": int(pairs["pert_id"].nunique()) if not pairs.empty else 0,
            "per_context_unique_compounds": {str(k): int(v) for k, v in per_line.items()},
            "control_audit": audit,
        }
    return report


def choose_primary_condition(coverage: Mapping[str, Any]) -> tuple[str, str]:
    ranked = sorted(
        coverage.items(),
        key=lambda item: (
            -int(item[1]["unique_pairs_with_matched_control"]),
            -int(item[1]["unique_contexts_with_matched_control"]),
            item[0],
        ),
    )
    if not ranked:
        raise ValueError("coverage is empty")
    primary = ranked[0][0]
    others = [name for name, _ in ranked if name != primary]
    if not others:
        raise ValueError("need both 6h and 24h conditions")
    return primary, others[0]


def _extract_rows(cache: np.ndarray, rows: Iterable[int]) -> np.ndarray:
    index = np.asarray(list(rows), dtype=np.int64)
    if len(index) == 0:
        return np.empty((0, cache.shape[1]), dtype=np.float32)
    out = np.empty((len(index), cache.shape[1]), dtype=np.float32)
    for start in range(0, len(index), 4096):
        stop = min(start + 4096, len(index))
        out[start:stop] = np.asarray(cache[index[start:stop]])
    return out


def compute_delta978(cache: np.ndarray, matched: pd.DataFrame) -> pd.DataFrame:
    """Pair-level mean(treatment) − mean(matched control) on exact978."""
    if matched.empty:
        return pd.DataFrame(
            columns=[
                "context_id",
                "pert_id",
                "treatment_rows",
                "control_rows",
                "control_type",
                "delta978",
            ]
        )
    treatment_rows = matched["_cache_row"].astype(np.int64).to_numpy()
    treatment_values = _extract_rows(cache, treatment_rows)
    unique_controls = sorted({int(row) for rows in matched["control_cache_rows"] for row in rows})
    control_lookup = {row: index for index, row in enumerate(unique_controls)}
    control_values = _extract_rows(cache, unique_controls)
    instance_deltas = np.empty_like(treatment_values)
    for offset, control_ids in enumerate(matched["control_cache_rows"]):
        control_index = [control_lookup[int(row)] for row in control_ids]
        instance_deltas[offset] = treatment_values[offset] - control_values[control_index].mean(axis=0)
    if not np.isfinite(instance_deltas).all():
        raise ValueError("non-finite exact978 values in Full Observed Oracle")
    rows: list[dict[str, Any]] = []
    grouped = matched.reset_index(drop=True).groupby(["cell_id", "pert_id"], sort=True, observed=True)
    for (context_id, pert_id), group in grouped:
        idx = group.index.to_numpy()
        control_ids = sorted({int(row) for rows in group["control_cache_rows"] for row in rows})
        control_types = sorted(set(group["control_type"].astype(str)))
        rows.append(
            {
                "context_id": str(context_id),
                "pert_id": str(pert_id),
                "treatment_rows": int(len(idx)),
                "control_rows": int(len(control_ids)),
                "control_type": ",".join(control_types),
                "delta978": instance_deltas[idx].mean(axis=0).astype(np.float32),
            }
        )
    return pd.DataFrame(rows)


def score_reversal(signature_values: np.ndarray, delta: np.ndarray) -> float:
    return float(reversal_score(np.asarray(signature_values, dtype=float), np.asarray(delta, dtype=float)))


def _enrichment(ranks: np.ndarray, membership: np.ndarray) -> float:
    n = int(len(ranks))
    hit = membership.astype(bool)
    t = int(hit.sum())
    if n == 0 or t == 0 or t == n:
        return 0.0
    order = np.argsort(ranks, kind="mergesort")
    ordered_hit = hit[order]
    increment = np.sqrt((n - t) / t)
    decrement = np.sqrt(t / (n - t))
    running = np.where(ordered_hit, increment, -decrement).cumsum()
    max_pos = float(running.max()) if running.size else 0.0
    max_neg = float(running.min()) if running.size else 0.0
    return max_pos if abs(max_pos) >= abs(max_neg) else max_neg


def weighted_ks_reversal(
    delta: np.ndarray,
    gene_indices: np.ndarray,
    signature_values: np.ndarray,
) -> float:
    """CMap-style weighted-KS reversal.  Larger means stronger anti-disease."""
    values = np.asarray(delta, dtype=float)[np.asarray(gene_indices, dtype=int)]
    signed = np.asarray(signature_values, dtype=float)
    ranks = (-values).argsort(kind="mergesort").argsort().astype(float)
    up = signed > 0
    down = signed < 0
    ks_up = _enrichment(ranks, up)
    ks_down = _enrichment(ranks, down)
    tau = (ks_up - ks_down) / 2.0 if np.sign(ks_up) != np.sign(ks_down) else 0.0
    return float(-tau)


def score_oracle_frame(
    delta_frame: pd.DataFrame,
    *,
    signature_indices: np.ndarray,
    signature_values: np.ndarray,
    include_cmap: bool = False,
) -> pd.DataFrame:
    rows = []
    for row in delta_frame.itertuples(index=False):
        vector = np.asarray(row.delta978, dtype=float)
        scored = {
            "context_id": str(row.context_id),
            "pert_id": str(row.pert_id),
            "treatment_rows": int(row.treatment_rows),
            "control_rows": int(row.control_rows),
            "control_type": str(row.control_type),
            "reversal_observed": score_reversal(signature_values, vector[signature_indices]),
        }
        if include_cmap:
            scored["reversal_cmap_weighted_ks"] = weighted_ks_reversal(
                vector, signature_indices, signature_values
            )
        rows.append(scored)
    return pd.DataFrame(rows)


def kendall(left: np.ndarray, right: np.ndarray) -> float | None:
    from scipy.stats import kendalltau

    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if len(left_arr) < 3 or np.std(left_arr) == 0.0 or np.std(right_arr) == 0.0:
        return None
    result = kendalltau(left_arr, right_arr, nan_policy="omit")
    if result.correlation is None or not np.isfinite(result.correlation):
        return None
    return float(result.correlation)


def _support_stratum(candidate_count: int) -> str:
    for name, low, high in SUPPORT_BINS:
        if low <= int(candidate_count) < high:
            return name
    return "n_ge_500"


def _compact_null(null: Mapping[str, Any]) -> dict[str, Any]:
    compact = {
        "seed": int(null["seed"]),
        "repeats": int(null["repeats"]),
        "spearman_mean": float(null["spearman_mean"]),
        "spearman_std": float(np.std(null["spearman_distribution"])) if null.get("spearman_distribution") else None,
        "top_k": {},
    }
    for key, payload in null.get("top_k", {}).items():
        compact["top_k"][str(key)] = {
            "expected_overlap_rate": float(payload["expected_overlap_rate"]),
            "ndcg_mean": float(payload["ndcg_mean"]),
            "overlap_std": float(np.std(payload["overlap_distribution"])) if payload.get("overlap_distribution") else None,
            "ndcg_std": float(np.std(payload["ndcg_distribution"])) if payload.get("ndcg_distribution") else None,
        }
    return compact


def _bootstrap_uncertainty(
    frame: pd.DataFrame,
    *,
    score_column: str,
    ks: Iterable[int],
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    if repeats < 1 or len(frame) < 3:
        return {"repeats": int(repeats), "status": "skipped"}
    rng = np.random.default_rng(int(seed))
    values = frame[[score_column, "sensitivity_score"]].to_numpy(float)
    spearmans: list[float] = []
    for _ in range(int(repeats)):
        index = rng.integers(0, len(values), size=len(values))
        sample = values[index]
        corr = spearman(sample[:, 0], sample[:, 1])
        if corr is not None:
            spearmans.append(float(corr))
    if not spearmans:
        return {"repeats": int(repeats), "status": "undefined"}
    return {
        "repeats": int(repeats),
        "status": "ok",
        "spearman_mean": float(np.mean(spearmans)),
        "spearman_std": float(np.std(spearmans)),
        "spearman_q05": float(np.quantile(spearmans, 0.05)),
        "spearman_q95": float(np.quantile(spearmans, 0.95)),
        "ks": [int(k) for k in ks],
    }


def evaluate_oracle_frame(
    frame: pd.DataFrame,
    *,
    score_column: str,
    ks: Iterable[int] = (10, 20, 50),
    minimum_candidates: int = 20,
    null_seed: int = NULL_SEED,
    null_repeats: int = NULL_REPEATS,
    bootstrap_repeats: int = 256,
) -> dict[str, Any]:
    line_rows: list[dict[str, Any]] = []
    context_column = "context_id" if "context_id" in frame.columns else "cell_id"
    group_cols = [col for col in (context_column, "depmap_id", "ccle_name") if col in frame.columns]
    if not group_cols:
        group_cols = [context_column]
    for keys, group in frame.groupby(group_cols, sort=True, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = {col: str(value) for col, value in zip(group_cols, keys)}
        metrics = rank_metrics(
            group,
            score_column=score_column,
            ks=ks,
            minimum_candidates=minimum_candidates,
            null_seed=null_seed,
            null_repeats=null_repeats,
        )
        if metrics.get("eligible"):
            finite = group.dropna(subset=[score_column, "sensitivity_score"])
            metrics["kendall"] = kendall(
                finite[score_column].to_numpy(float),
                finite["sensitivity_score"].to_numpy(float),
            )
            metrics["support_stratum"] = _support_stratum(int(metrics["candidate_count"]))
            metrics["bootstrap"] = _bootstrap_uncertainty(
                finite,
                score_column=score_column,
                ks=ks,
                seed=null_seed,
                repeats=bootstrap_repeats,
            )
            metrics["null_baseline"] = {
                **metrics["null_baseline"],
                **_compact_null(metrics["null_baseline"]),
            }
            metrics["null_baseline"].pop("spearman_distribution", None)
            for payload in metrics["null_baseline"].get("top_k", {}).values():
                payload.pop("overlap_distribution", None)
                payload.pop("ndcg_distribution", None)
        line_rows.append({**identity, **metrics})
    eligible = [row for row in line_rows if row.get("eligible")]
    return {
        "line_count": int(len(line_rows)),
        "eligible_line_count": int(len(eligible)),
        "line_rows": line_rows,
        "support_strata": stratified_support_summary(line_rows),
        "null": {"seed": int(null_seed), "repeats": int(null_repeats)},
    }


def stratified_support_summary(line_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in line_rows if row.get("eligible")]
    strata: dict[str, Any] = {
        name: {"line_count": 0, "candidate_counts": [], "top10_overlap_lifts": []}
        for name, _, _ in SUPPORT_BINS
    }
    for row in eligible:
        name = str(row.get("support_stratum") or _support_stratum(int(row.get("candidate_count", 0))))
        bucket = strata.setdefault(
            name, {"line_count": 0, "candidate_counts": [], "top10_overlap_lifts": []}
        )
        bucket["line_count"] += 1
        bucket["candidate_counts"].append(int(row.get("candidate_count", 0)))
        top = row.get("top_k", {})
        if "10" in top and top["10"].get("overlap_lift") is not None:
            bucket["top10_overlap_lifts"].append(float(top["10"]["overlap_lift"]))
    for bucket in strata.values():
        lifts = bucket["top10_overlap_lifts"]
        bucket["mean_top10_overlap_lift"] = float(np.mean(lifts)) if lifts else None
        bucket["note"] = "do_not_macro_average_across_strata"
    return {
        **strata,
        "eligible_line_count": int(len(eligible)),
        "forbidden_interpretation": (
            "do not macro-average lift across lines with different candidate sizes"
        ),
    }


def predicted_oracle_gap(joined: pd.DataFrame) -> dict[str, Any]:
    required = {"reversal_observed", "reversal_predicted"}
    missing = sorted(required.difference(joined.columns))
    if missing:
        raise ValueError(f"gap table missing columns: {missing}")
    finite = joined.dropna(subset=list(required)).copy()
    if finite.empty:
        return {"pair_count": 0, "pearson": None, "spearman": None, "mae": None, "kendall": None}
    observed = finite["reversal_observed"].to_numpy(float)
    predicted = finite["reversal_predicted"].to_numpy(float)
    pearson = None
    if np.std(observed) > 0 and np.std(predicted) > 0:
        pearson = float(np.corrcoef(predicted, observed)[0, 1])
    return {
        "pair_count": int(len(finite)),
        "context_count": int(finite["context_id"].nunique()) if "context_id" in finite.columns else None,
        "compound_count": int(finite["pert_id"].nunique()) if "pert_id" in finite.columns else None,
        "pearson": pearson,
        "spearman": spearman(predicted, observed),
        "kendall": kendall(predicted, observed),
        "mae": float(np.mean(np.abs(predicted - observed))),
    }


def join_prism_after_identity_freeze(
    oracle: pd.DataFrame,
    prism: pd.DataFrame,
) -> pd.DataFrame:
    working = prism.copy()
    if "context_id" not in working.columns:
        working["context_id"] = working["ccle_name"].astype(str).str.split("_", n=1).str[0]
    working = (
        working.groupby(["depmap_id", "ccle_name", "context_id", "pert_id"], as_index=False, sort=True, observed=True)
        .agg(sensitivity_score=("sensitivity_score", "mean"), prism_response_rows=("sensitivity_score", "size"))
    )
    return working.merge(
        oracle,
        on=["context_id", "pert_id"],
        how="inner",
        validate="one_to_one",
    )


def oracle_near_null(line_rows: Iterable[Mapping[str, Any]], *, min_lines: int = 3) -> bool:
    """True when Top-K/Spearman evidence does not beat the line-wise null.

    A line is treated as beating the Top-10 null only when overlap lift > 1.
    The Full Oracle is *not* called above-null unless a majority of eligible
    lines beat that null.  A negative median Spearman delta is near-null, not
    evidence of a theory chain.
    """
    eligible = [row for row in line_rows if row.get("eligible")]
    if len(eligible) < min_lines:
        return True
    lifts = []
    deltas = []
    for row in eligible:
        top = row.get("top_k", {})
        if "10" in top and top["10"].get("overlap_lift") is not None:
            lifts.append(float(top["10"]["overlap_lift"]))
        if isinstance(row.get("null_baseline"), Mapping) and row["null_baseline"].get("spearman_delta") is not None:
            deltas.append(float(row["null_baseline"]["spearman_delta"]))
    if not lifts:
        return True
    median_lift = float(np.median(lifts))
    median_delta = float(np.median(deltas)) if deltas else 0.0
    fraction_above = float(np.mean(np.asarray(lifts) > 1.0))
    return median_lift <= 1.15 or fraction_above < 0.5 or median_delta <= 0.03
