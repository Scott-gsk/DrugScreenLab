"""Frozen, prediction-only evaluation contract for EXP-011.

The evaluator deliberately has no fitting, calibration, threshold selection, or
data-discovery path.  It consumes a long-form, held-out prediction table and
implements the decision order agreed for EXP-011:

``GDSC cold-cell + frozen PRISM gate -> direct PDO endpoint -> exploratory``.

Bladder, clinical, and breast rows are accepted only as labelled exploratory
artifacts.  They are never read by the primary verdict code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from drug_screen.evaluation.phase1_prism import spearman


EXPERIMENT_ID = "EXP-011"
DEFAULT_SEED = 20260823
PRIMARY_PDO_COHORTS = ("mcrc_pdo", "zhu_hepatobiliary")
EXPLORATORY_ROLE = "external_exploratory"

# A full schema is intentional: every emitted prediction can be traced to its
# response definition, chemical identity and test partition without consulting
# a training artifact.  Identity fields unused by one biological system must be
# present as empty values rather than silently omitted.
PREDICTION_COLUMNS = (
    "experiment_id",
    "model_id",
    "evaluation_role",
    "cohort_id",
    "fold_id",
    "split",
    "patient_id",
    "sample_id",
    "pdo_id",
    "cell_line_id",
    "drug_id",
    "inchikey",
    "drug_scaffold",
    "response_name",
    "response_scale",
    "higher_is_sensitive",
    "y_true",
    "y_pred",
)

_ROLES = frozenset(
    {
        "gdsc_cold_cell",
        "gdsc_cold_drug",
        "prism_frozen",
        "pdo_primary",
        EXPLORATORY_ROLE,
    }
)
_ROLE_SPLITS = {
    "gdsc_cold_cell": "test",
    "gdsc_cold_drug": "test",
    "pdo_primary": "test",
    "prism_frozen": "external",
    EXPLORATORY_ROLE: "external",
}


class EvaluationContractError(ValueError):
    """Raised when a prediction artifact cannot support the frozen protocol."""


class Exp011Conclusion(str, Enum):
    """The only final conclusions permitted by the EXP-011 decision tree."""

    CELL_LINE_GATE_FAILED = "CELL_LINE_GATE_FAILED"
    PROMISING_MULTI_PDO = "PROMISING_MULTI_PDO"
    TRANSFER_ONLY = "TRANSFER_ONLY"
    NO_GO = "NO_GO"


@dataclass(frozen=True)
class PredictionSchema:
    """Machine-readable description of the required held-out prediction table."""

    format: str = "exp011_heldout_predictions_v1"
    columns: tuple[str, ...] = PREDICTION_COLUMNS
    model_selection_roles: tuple[str, ...] = (
        "gdsc_cold_cell",
        "gdsc_cold_drug",
        "pdo_primary",
    )
    frozen_external_role: str = "prism_frozen"
    exploratory_role: str = EXPLORATORY_ROLE
    external_not_in_primary_verdict: tuple[str, ...] = ("bladder", "clinical", "breast")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def prediction_schema() -> dict[str, object]:
    """Return the public, serialisable input contract used by the CLI."""

    return PredictionSchema().to_dict()


def _text(value: object, column: str, *, allow_empty: bool = False) -> str:
    if pd.isna(value):
        if allow_empty:
            return ""
        raise EvaluationContractError(f"{column} cannot be null")
    result = str(value).strip()
    if not result and not allow_empty:
        raise EvaluationContractError(f"{column} cannot be empty")
    return result


def _normalized_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise EvaluationContractError("higher_is_sensitive must contain boolean values")


def validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    expected_experiment_id: str = EXPERIMENT_ID,
) -> pd.DataFrame:
    """Validate and canonically type the only accepted EXP-011 prediction input.

    The function intentionally requires all columns even for fields which are
    blank in a given cohort.  This turns accidental loss of a join identity into
    an early error instead of a silently smaller evaluation set.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame")
    missing = set(PREDICTION_COLUMNS).difference(frame.columns)
    if missing:
        raise EvaluationContractError(f"prediction table misses columns: {sorted(missing)}")
    if frame.empty:
        raise EvaluationContractError("prediction table cannot be empty")
    result = frame.loc[:, PREDICTION_COLUMNS].copy()
    for column in (
        "experiment_id",
        "model_id",
        "evaluation_role",
        "cohort_id",
        "split",
        "sample_id",
        "drug_id",
        "inchikey",
        "drug_scaffold",
        "response_name",
        "response_scale",
    ):
        result[column] = result[column].map(lambda value: _text(value, column))
    for column in ("patient_id", "pdo_id", "cell_line_id"):
        result[column] = result[column].map(lambda value: _text(value, column, allow_empty=True))
    if set(result["experiment_id"]) != {expected_experiment_id}:
        raise EvaluationContractError(
            f"prediction experiment_id must be exactly {expected_experiment_id}"
        )
    unknown_roles = sorted(set(result["evaluation_role"]).difference(_ROLES))
    if unknown_roles:
        raise EvaluationContractError(f"unsupported evaluation_role values: {unknown_roles}")
    expected_splits = result["evaluation_role"].map(_ROLE_SPLITS)
    if not result["split"].eq(expected_splits).all():
        raise EvaluationContractError("each evaluation_role requires its frozen split value")
    try:
        folds = pd.to_numeric(result["fold_id"], errors="raise")
    except (TypeError, ValueError) as error:
        raise EvaluationContractError("fold_id must be an integer") from error
    if not np.isfinite(folds.to_numpy(float)).all() or not np.equal(folds, np.floor(folds)).all():
        raise EvaluationContractError("fold_id must be a finite integer")
    result["fold_id"] = folds.astype(int)
    is_external = result["evaluation_role"].isin({"prism_frozen", EXPLORATORY_ROLE})
    if not (result.loc[is_external, "fold_id"] == -1).all() or not (result.loc[~is_external, "fold_id"] >= 0).all():
        raise EvaluationContractError("external rows require fold_id=-1; held-out rows require fold_id>=0")
    result["higher_is_sensitive"] = result["higher_is_sensitive"].map(_normalized_bool)
    for column in ("y_true", "y_pred"):
        try:
            result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
        except (TypeError, ValueError) as error:
            raise EvaluationContractError(f"{column} must be numeric") from error
        if not np.isfinite(result[column].to_numpy(float)).all():
            raise EvaluationContractError(f"{column} must be finite")
    # The response direction must already have been normalised before model
    # training/prediction.  A mixed-direction group is not interpretable.
    direction_counts = result.groupby(["cohort_id", "response_name", "response_scale"], observed=True)[
        "higher_is_sensitive"
    ].nunique()
    if (direction_counts != 1).any():
        raise EvaluationContractError("response direction differs within a cohort response definition")
    cell_roles = result["evaluation_role"].isin({"gdsc_cold_cell", "gdsc_cold_drug", "prism_frozen"})
    if (result.loc[cell_roles, "cell_line_id"] == "").any():
        raise EvaluationContractError("cell-line evaluations require cell_line_id")
    if (result.loc[result["evaluation_role"] == "pdo_primary", "patient_id"] == "").any() or (
        result.loc[result["evaluation_role"] == "pdo_primary", "pdo_id"] == ""
    ).any():
        raise EvaluationContractError("primary PDO evaluations require patient_id and pdo_id")
    key = [
        "experiment_id",
        "model_id",
        "evaluation_role",
        "cohort_id",
        "fold_id",
        "sample_id",
        "drug_id",
    ]
    if result.duplicated(key).any():
        raise EvaluationContractError("duplicate held-out sample-drug prediction rows must be resolved")
    return result


def _correlation(observed: np.ndarray, predicted: np.ndarray, *, kind: str) -> float | None:
    if len(observed) < 2 or np.std(observed) == 0.0 or np.std(predicted) == 0.0:
        return None
    if kind == "spearman":
        return spearman(predicted, observed)
    if kind == "pearson":
        return float(np.corrcoef(predicted, observed)[0, 1])
    raise ValueError(f"unsupported correlation: {kind}")


def _ndcg_at_k(observed: np.ndarray, predicted: np.ndarray, *, k: int = 5) -> float | None:
    if len(observed) < 2:
        return None
    width = min(int(k), len(observed))
    if width < 1:
        return None
    # Shifting preserves order while making residual response values valid
    # non-negative ranking relevance scores.
    relevance = observed - float(np.min(observed))
    pred_order = np.argsort(-predicted, kind="mergesort")[:width]
    ideal_order = np.argsort(-observed, kind="mergesort")[:width]
    discounts = np.log2(np.arange(2, width + 2, dtype=float))
    actual = float(np.sum(relevance[pred_order] / discounts))
    ideal = float(np.sum(relevance[ideal_order] / discounts))
    return actual / ideal if ideal > 0.0 else None


def _macro(values: Iterable[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(usable)) if usable else None


def _select_role_model(frame: pd.DataFrame, role: str, model_id: str) -> pd.DataFrame:
    selected = frame[(frame["evaluation_role"] == role) & (frame["model_id"] == model_id)].copy()
    if selected.empty:
        raise EvaluationContractError(f"missing {role} predictions for model {model_id}")
    return selected


def assert_heldout_group_isolation(frame: pd.DataFrame, *, group_column: str) -> None:
    """Require each held-out biological group to occur in exactly one test fold."""

    if group_column not in frame.columns:
        raise EvaluationContractError(f"missing isolation group column: {group_column}")
    values = frame[frame["split"] == "test"]
    if values.empty:
        raise EvaluationContractError("held-out isolation requires test rows")
    folds = values.groupby(group_column, observed=True)["fold_id"].nunique()
    if (folds > 1).any():
        leaked = folds[folds > 1].index.astype(str).tolist()[:5]
        raise EvaluationContractError(f"{group_column} appears in multiple held-out folds: {leaked}")


def cell_line_macro_metrics(
    frame: pd.DataFrame,
    *,
    minimum_cells_per_drug: int = 10,
    isolation_group_column: str | None = "cell_line_id",
) -> dict[str, object]:
    """Macro drug metrics over cell lines for a frozen GDSC held-out set."""

    if minimum_cells_per_drug < 2:
        raise ValueError("minimum_cells_per_drug must be at least two")
    if frame.empty:
        raise EvaluationContractError("cell-line metric input cannot be empty")
    if isolation_group_column is not None:
        assert_heldout_group_isolation(frame, group_column=isolation_group_column)
    rows: list[dict[str, object]] = []
    for drug_id, group in frame.groupby("drug_id", sort=True, observed=True):
        observed = group["y_true"].to_numpy(float)
        predicted = group["y_pred"].to_numpy(float)
        eligible = int(group["cell_line_id"].nunique()) >= minimum_cells_per_drug
        rows.append(
            {
                "drug_id": str(drug_id),
                "cell_line_count": int(group["cell_line_id"].nunique()),
                "pair_count": int(len(group)),
                "eligible": eligible,
                "spearman": _correlation(observed, predicted, kind="spearman") if eligible else None,
                "pearson": _correlation(observed, predicted, kind="pearson") if eligible else None,
                "mae": float(np.mean(np.abs(observed - predicted))) if eligible else None,
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    return {
        "format": "exp011_cell_line_per_drug_macro_v1",
        "drug_count": len(rows),
        "eligible_drug_count": len(eligible),
        "minimum_cells_per_drug": minimum_cells_per_drug,
        "isolation_group_column": isolation_group_column,
        "macro_spearman": _macro(row["spearman"] for row in eligible),
        "macro_pearson": _macro(row["pearson"] for row in eligible),
        "macro_mae": _macro(row["mae"] for row in eligible),
        "prediction_std": float(frame["y_pred"].std(ddof=0)),
        "observed_std": float(frame["y_true"].std(ddof=0)),
        "per_drug": rows,
    }


def _paired_metric_bootstrap(
    candidate_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
    *,
    metric: str,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    if repeats < 100:
        raise ValueError("at least 100 bootstrap repeats are required")
    candidate = pd.DataFrame(candidate_rows).set_index("drug_id")[metric].rename("candidate")
    baseline = pd.DataFrame(baseline_rows).set_index("drug_id")[metric].rename("baseline")
    joined = pd.concat([candidate, baseline], axis=1, join="inner").dropna()
    if len(joined) < 2 or set(candidate.dropna().index) != set(baseline.dropna().index):
        raise EvaluationContractError("paired drug bootstrap requires the same evaluable drug support")
    delta = (joined["candidate"] - joined["baseline"]).to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.mean(delta[rng.integers(0, len(delta), size=(repeats, len(delta)))], axis=1)
    return {
        "metric": metric,
        "unit": "drug",
        "unit_count": int(len(delta)),
        "repeats": int(repeats),
        "seed": int(seed),
        "delta_mean": float(np.mean(delta)),
        "ci_95": [float(value) for value in np.quantile(values, [0.025, 0.975])],
    }


def prism_frozen_metrics(
    frame: pd.DataFrame,
    *,
    expected_drug_count: int = 81,
    expected_cell_line_count: int = 35,
    expected_row_count: int = 2784,
    minimum_cells_per_drug: int = 10,
) -> dict[str, object]:
    """Evaluate the pre-frozen PRISM bridge without changing its support set."""

    if frame.empty:
        raise EvaluationContractError("PRISM prediction input cannot be empty")
    counts = {
        "row_count": int(len(frame)),
        "drug_count": int(frame["inchikey"].nunique()),
        "cell_line_count": int(frame["cell_line_id"].nunique()),
    }
    expected = {
        "row_count": int(expected_row_count),
        "drug_count": int(expected_drug_count),
        "cell_line_count": int(expected_cell_line_count),
    }
    if counts != expected:
        raise EvaluationContractError(f"frozen PRISM support mismatch: observed={counts}, expected={expected}")
    if frame[["cell_line_id", "inchikey"]].duplicated().any():
        raise EvaluationContractError("PRISM contains duplicate cell-line/InChIKey pairs")
    # The gate uses the same endpoint as GDSC: for each drug, correlate across
    # independent cell lines and macro-average drugs.  A line-wise cross-drug
    # rank could be driven by a drug main effect and is retained only below as
    # a diagnostic.
    primary = cell_line_macro_metrics(
        frame,
        minimum_cells_per_drug=minimum_cells_per_drug,
        isolation_group_column=None,
    )
    rows: list[dict[str, object]] = []
    for cell_line, group in frame.groupby("cell_line_id", sort=True, observed=True):
        observed = group["y_true"].to_numpy(float)
        predicted = group["y_pred"].to_numpy(float)
        eligible = len(group) >= 2
        rows.append(
            {
                "cell_line_id": str(cell_line),
                "drug_count": int(len(group)),
                "eligible": eligible,
                "spearman": _correlation(observed, predicted, kind="spearman") if eligible else None,
                "pearson": _correlation(observed, predicted, kind="pearson") if eligible else None,
                "mae": float(np.mean(np.abs(observed - predicted))) if eligible else None,
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    return {
        "format": "exp011_prism_frozen_per_drug_macro_v1",
        "support": {"observed": counts, "expected": expected, "exact_inchikey_required": True},
        "main_endpoint": "per_drug_macro_spearman_across_frozen_PRISM_cell_lines",
        "minimum_cells_per_drug": minimum_cells_per_drug,
        "eligible_drug_count": primary["eligible_drug_count"],
        "macro_spearman": primary["macro_spearman"],
        "macro_pearson": primary["macro_pearson"],
        "macro_mae": primary["macro_mae"],
        "per_drug": primary["per_drug"],
        "cell_line_cross_drug_ranking_diagnostic": {
            "not_a_gate_metric": True,
            "minimum_drugs_per_line": 2,
            "eligible_cell_line_count": len(eligible),
            "macro_spearman": _macro(row["spearman"] for row in eligible),
            "macro_pearson": _macro(row["pearson"] for row in eligible),
            "macro_mae": _macro(row["mae"] for row in eligible),
            "per_cell_line": rows,
        },
        "labels_used_for_model_selection": False,
    }


def evaluate_cell_line_gate(
    frame: pd.DataFrame,
    *,
    source_model_id: str = "M2",
    ridge_model_id: str = "B1_ridge",
    repeats: int = 2000,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Apply the pre-PDO hard gate using GDSC and frozen PRISM only."""

    source = _select_role_model(frame, "gdsc_cold_cell", source_model_id)
    ridge = _select_role_model(frame, "gdsc_cold_cell", ridge_model_id)
    support_keys = ["cohort_id", "fold_id", "cell_line_id", "sample_id", "drug_id"]
    source_labels = source.set_index(support_keys)["y_true"].sort_index()
    ridge_labels = ridge.set_index(support_keys)["y_true"].sort_index()
    if not source_labels.index.equals(ridge_labels.index) or not np.allclose(
        source_labels.to_numpy(float), ridge_labels.to_numpy(float)
    ):
        raise EvaluationContractError("source and Ridge cold-cell evaluations must have identical labelled support")
    source_metrics = cell_line_macro_metrics(source)
    ridge_metrics = cell_line_macro_metrics(ridge)
    source_support = {
        row["drug_id"] for row in source_metrics["per_drug"] if row["eligible"]
    }
    ridge_support = {
        row["drug_id"] for row in ridge_metrics["per_drug"] if row["eligible"]
    }
    if source_support != ridge_support:
        raise EvaluationContractError("source and Ridge cold-cell evaluations must have identical drug support")
    if source_metrics["macro_spearman"] is None or ridge_metrics["macro_spearman"] is None:
        delta: dict[str, object] = {
            "metric": "spearman",
            "unit": "drug",
            "status": "UNDEFINED_CONSTANT_PREDICTION_OR_RESPONSE",
            "delta_mean": None,
            "ci_95": [None, None],
        }
    else:
        delta = _paired_metric_bootstrap(
            source_metrics["per_drug"], ridge_metrics["per_drug"], metric="spearman", repeats=repeats, seed=seed
        )
    prism = prism_frozen_metrics(_select_role_model(frame, "prism_frozen", source_model_id))
    observed_std = float(source_metrics["observed_std"])
    prediction_std = float(source_metrics["prediction_std"])
    spread_ratio = prediction_std / observed_std if observed_std > 0.0 else 0.0
    conditions = {
        "cold_cell_macro_spearman_at_least_0_30": bool(
            source_metrics["macro_spearman"] is not None and source_metrics["macro_spearman"] >= 0.30
        ),
        "cold_cell_vs_ridge_ci_lower_at_least_minus_0_02": bool(
            delta["ci_95"][0] is not None and float(delta["ci_95"][0]) >= -0.02
        ),
        "prism_macro_spearman_gt_0_10": bool(
            prism["macro_spearman"] is not None and prism["macro_spearman"] > 0.10
        ),
        "prediction_spread_at_least_5_percent_of_observed": bool(spread_ratio >= 0.05),
    }
    return {
        "format": "exp011_cell_line_gate_v1",
        "source_model_id": source_model_id,
        "ridge_model_id": ridge_model_id,
        "cold_cell": {"source": source_metrics, "ridge": ridge_metrics, "source_minus_ridge": delta},
        "prism_frozen": prism,
        "prediction_spread_ratio": float(spread_ratio),
        "conditions": conditions,
        "passed": bool(all(conditions.values())),
    }


def _patient_drug_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Average replicate/region PDO rows before patient-level PDO scoring."""

    columns = ["cohort_id", "fold_id", "patient_id", "drug_id"]
    grouped = frame.groupby(columns, sort=True, observed=True)
    result = grouped[["y_true", "y_pred"]].mean().reset_index()
    # A patient cannot be a held-out patient in more than one fold; this check
    # catches concatenated or mislabelled folds before bootstrap resampling.
    assignments = result.groupby(["cohort_id", "patient_id"], observed=True)["fold_id"].nunique()
    if (assignments > 1).any():
        leaked = ["::".join(map(str, value)) for value in assignments[assignments > 1].index.tolist()[:5]]
        raise EvaluationContractError(f"PDO patient appears in multiple test folds: {leaked}")
    return result


def _pdo_per_drug_metrics(patient_drug: pd.DataFrame, *, minimum_patients_per_drug: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    # All five out-of-fold predictions are pooled here.  Every patient is
    # unseen for its own prediction; splitting this metric by fold would leave
    # only ~4 hepatobiliary patients per fold and make the predeclared minimum
    # support impossible.
    for (cohort, drug), group in patient_drug.groupby(["cohort_id", "drug_id"], sort=True, observed=True):
        observed = group["y_true"].to_numpy(float)
        predicted = group["y_pred"].to_numpy(float)
        eligible = int(group["patient_id"].nunique()) >= minimum_patients_per_drug
        rows.append(
            {
                "cohort_id": str(cohort),
                "drug_id": str(drug),
                "patient_count": int(group["patient_id"].nunique()),
                "eligible": eligible,
                "spearman": _correlation(observed, predicted, kind="spearman") if eligible else None,
                "pearson": _correlation(observed, predicted, kind="pearson") if eligible else None,
                "mae": float(np.mean(np.abs(observed - predicted))) if eligible else None,
            }
        )
    return pd.DataFrame(rows)


def raw_patient_ranking_metrics(patient_drug: pd.DataFrame, *, k: int = 5) -> dict[str, object]:
    """Legacy raw cross-drug patient ranking diagnostic, never a primary metric."""

    rows: list[dict[str, object]] = []
    for (cohort, patient), group in patient_drug.groupby(["cohort_id", "patient_id"], sort=True, observed=True):
        observed = group["y_true"].to_numpy(float)
        predicted = group["y_pred"].to_numpy(float)
        if len(group) < 2:
            continue
        rows.append(
            {
                "cohort_id": str(cohort),
                "patient_id": str(patient),
                "drug_count": int(len(group)),
                "spearman": _correlation(observed, predicted, kind="spearman"),
                "ndcg_at_5": _ndcg_at_k(observed, predicted, k=k),
            }
        )
    per_patient = pd.DataFrame(rows)
    return {
        "format": "exp011_raw_patient_ranking_diagnostic_v1",
        "not_a_primary_endpoint": True,
        "patient_count": int(len(per_patient)),
        "macro_spearman": _macro(per_patient.get("spearman", pd.Series(dtype=float)).tolist()),
        "macro_ndcg_at_5": _macro(per_patient.get("ndcg_at_5", pd.Series(dtype=float)).tolist()),
        "per_patient": rows,
    }


def residualized_patient_ranking_metrics(patient_drug: pd.DataFrame, *, k: int = 5) -> dict[str, object]:
    """Test-set centred patient ranking diagnostic that removes drug main effects.

    This uses held-out labels only after predictions are frozen.  It is a
    descriptive diagnostic, never a training transform, checkpoint selector or
    primary endpoint.  The main endpoint remains per-drug patient Spearman.
    """

    values = patient_drug.copy()
    centers = values.groupby(["cohort_id", "drug_id"], observed=True)[["y_true", "y_pred"]].transform("mean")
    values["residual_true"] = values["y_true"] - centers["y_true"]
    values["residual_pred"] = values["y_pred"] - centers["y_pred"]
    rows: list[dict[str, object]] = []
    for (cohort, patient), group in values.groupby(["cohort_id", "patient_id"], sort=True, observed=True):
        observed = group["residual_true"].to_numpy(float)
        predicted = group["residual_pred"].to_numpy(float)
        if len(group) < 2:
            continue
        rows.append(
            {
                "cohort_id": str(cohort),
                "patient_id": str(patient),
                "drug_count": int(len(group)),
                "spearman": _correlation(observed, predicted, kind="spearman"),
                "ndcg_at_5": _ndcg_at_k(observed, predicted, k=k),
            }
        )
    per_patient = pd.DataFrame(rows)
    return {
        "format": "exp011_residualized_patient_ranking_diagnostic_v1",
        "test_label_centering": True,
        "not_for_model_selection": True,
        "patient_count": int(len(per_patient)),
        "macro_spearman": _macro(per_patient.get("spearman", pd.Series(dtype=float)).tolist()),
        "macro_ndcg_at_5": _macro(per_patient.get("ndcg_at_5", pd.Series(dtype=float)).tolist()),
        "per_patient": rows,
    }


def pdo_primary_metrics(
    frame: pd.DataFrame,
    *,
    primary_cohorts: Sequence[str] = PRIMARY_PDO_COHORTS,
    minimum_patients_per_drug: int = 5,
) -> dict[str, object]:
    """Compute the pre-registered direct PDO endpoint and diagnostics.

    Per-drug correlations are calculated across held-out patients.  We first
    average multiple PDO regions/replicates for the same patient/drug, macro
    average drugs within each cohort, then give mCRC and hepatobiliary equal
    weight regardless of their response-table sizes.
    """

    expected = tuple(str(value) for value in primary_cohorts)
    values = frame[frame["cohort_id"].isin(expected)].copy()
    if set(values["cohort_id"]) != set(expected):
        missing = sorted(set(expected).difference(values["cohort_id"]))
        raise EvaluationContractError(f"missing primary PDO cohorts: {missing}")
    values = values.assign(_cohort_patient_id=values["cohort_id"] + "::" + values["patient_id"])
    assert_heldout_group_isolation(values, group_column="_cohort_patient_id")
    patient_drug = _patient_drug_table(values)
    per_drug = _pdo_per_drug_metrics(patient_drug, minimum_patients_per_drug=minimum_patients_per_drug)
    cohort_rows: list[dict[str, object]] = []
    for cohort in expected:
        group = per_drug[per_drug["cohort_id"] == cohort]
        eligible = group[group["eligible"]]
        cohort_rows.append(
            {
                "cohort_id": cohort,
                "patient_count": int(patient_drug.loc[patient_drug["cohort_id"] == cohort, "patient_id"].nunique()),
                "drug_count": int(len(group)),
                "eligible_drug_count": int(len(eligible)),
                "macro_spearman": _macro(eligible["spearman"].tolist()),
                "macro_pearson": _macro(eligible["pearson"].tolist()),
                "same_scale_mae": _macro(eligible["mae"].tolist()),
            }
        )
    per_cohort = pd.DataFrame(cohort_rows)
    return {
        "format": "exp011_primary_pdo_metrics_v1",
        "primary_cohorts": list(expected),
        "minimum_patients_per_drug": int(minimum_patients_per_drug),
        "main_endpoint": "equal_cohort_mean_of_per_drug_heldout_patient_spearman",
        "macro_spearman_equal_cohort": _macro(per_cohort["macro_spearman"].tolist()),
        "macro_pearson_equal_cohort": _macro(per_cohort["macro_pearson"].tolist()),
        "same_scale_mae_by_cohort": {
            str(row["cohort_id"]): row["same_scale_mae"] for row in cohort_rows
        },
        "per_cohort": cohort_rows,
        "per_drug": per_drug.to_dict("records"),
        "raw_patient_ranking_diagnostic": raw_patient_ranking_metrics(patient_drug),
        "residualized_patient_ranking_diagnostic": residualized_patient_ranking_metrics(patient_drug),
    }


def _aligned_models(frame: pd.DataFrame, candidate_model_id: str, baseline_model_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = frame[frame["model_id"] == candidate_model_id].copy()
    baseline = frame[frame["model_id"] == baseline_model_id].copy()
    keys = ["cohort_id", "fold_id", "patient_id", "sample_id", "pdo_id", "drug_id"]
    left = candidate.set_index(keys)["y_true"].sort_index()
    right = baseline.set_index(keys)["y_true"].sort_index()
    if not left.index.equals(right.index) or not np.allclose(left.to_numpy(float), right.to_numpy(float)):
        raise EvaluationContractError("candidate and baseline PDO predictions must have identical labelled support")
    return candidate, baseline


def _patient_resample_draws(frame: pd.DataFrame, *, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw patient indices once so paired models use identical clusters."""

    draws: dict[str, np.ndarray] = {}
    for cohort, group in frame.groupby("cohort_id", sort=True, observed=True):
        patients = np.asarray(sorted(group["patient_id"].unique()), dtype=object)
        draws[str(cohort)] = rng.integers(0, len(patients), size=len(patients))
    return draws


def _resample_patients_within_cohort(frame: pd.DataFrame, *, draws: Mapping[str, np.ndarray]) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for cohort, group in frame.groupby("cohort_id", sort=True, observed=True):
        patients = np.asarray(sorted(group["patient_id"].unique()), dtype=object)
        cohort_draws = draws.get(str(cohort))
        if cohort_draws is None or len(cohort_draws) != len(patients):
            raise EvaluationContractError("paired bootstrap draws do not match cohort patient support")
        for draw_index, patient_index in enumerate(cohort_draws):
            patient = patients[patient_index]
            copied = group[group["patient_id"] == patient].copy()
            # Resampled repeats must remain distinct patients to preserve the
            # patient-equal macro endpoint during reaggregation.
            suffix = f"__bootstrap_{draw_index}"
            copied["patient_id"] = copied["patient_id"].astype(str) + suffix
            copied["pdo_id"] = copied["pdo_id"].astype(str) + suffix
            copied["sample_id"] = copied["sample_id"].astype(str) + suffix
            chunks.append(copied)
    return pd.concat(chunks, ignore_index=True)


def patient_within_cohort_cluster_bootstrap_delta(
    frame: pd.DataFrame,
    *,
    candidate_model_id: str,
    baseline_model_id: str,
    primary_cohorts: Sequence[str] = PRIMARY_PDO_COHORTS,
    minimum_patients_per_drug: int = 5,
    repeats: int = 2000,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Paired 2,000-repeat patient-within-cohort cluster bootstrap for PDOs."""

    if repeats < 100:
        raise ValueError("at least 100 bootstrap repeats are required")
    values = frame[frame["cohort_id"].isin(tuple(primary_cohorts))].copy()
    candidate, baseline = _aligned_models(values, candidate_model_id, baseline_model_id)
    candidate_point = pdo_primary_metrics(
        candidate, primary_cohorts=primary_cohorts, minimum_patients_per_drug=minimum_patients_per_drug
    )["macro_spearman_equal_cohort"]
    baseline_point = pdo_primary_metrics(
        baseline, primary_cohorts=primary_cohorts, minimum_patients_per_drug=minimum_patients_per_drug
    )["macro_spearman_equal_cohort"]
    if candidate_point is None or baseline_point is None:
        raise EvaluationContractError("primary PDO endpoint is undefined; bootstrap cannot proceed")
    rng = np.random.default_rng(seed)
    bootstrap: list[float] = []
    for _ in range(repeats):
        draws = _patient_resample_draws(candidate, rng=rng)
        candidate_sample = _resample_patients_within_cohort(candidate, draws=draws)
        baseline_sample = _resample_patients_within_cohort(baseline, draws=draws)
        candidate_value = pdo_primary_metrics(
            candidate_sample, primary_cohorts=primary_cohorts, minimum_patients_per_drug=minimum_patients_per_drug
        )["macro_spearman_equal_cohort"]
        baseline_value = pdo_primary_metrics(
            baseline_sample, primary_cohorts=primary_cohorts, minimum_patients_per_drug=minimum_patients_per_drug
        )["macro_spearman_equal_cohort"]
        if candidate_value is not None and baseline_value is not None:
            bootstrap.append(float(candidate_value - baseline_value))
    if len(bootstrap) != repeats:
        raise EvaluationContractError("bootstrap produced undefined endpoint draws; increase PDO support")
    return {
        "format": "exp011_patient_within_cohort_cluster_bootstrap_v1",
        "metric": "equal_cohort_mean_of_per_drug_heldout_patient_spearman",
        "candidate_model_id": candidate_model_id,
        "baseline_model_id": baseline_model_id,
        "repeats": int(repeats),
        "seed": int(seed),
        "point_delta": float(candidate_point - baseline_point),
        "ci_95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
    }


def cold_drug_report(frame: pd.DataFrame, *, source_model_id: str = "M2") -> dict[str, object] | None:
    """Report, but do not gate on, the separate GDSC cold-drug stress test."""

    selected = frame[(frame["evaluation_role"] == "gdsc_cold_drug") & (frame["model_id"] == source_model_id)]
    return None if selected.empty else cell_line_macro_metrics(selected, isolation_group_column="drug_id")


def external_exploratory_manifest(frame: pd.DataFrame) -> dict[str, object]:
    """Expose external assets without allowing them to affect the primary verdict."""

    values = frame[frame["evaluation_role"] == EXPLORATORY_ROLE]
    return {
        "format": "exp011_external_exploratory_manifest_v1",
        "not_used_for_model_selection": True,
        "not_used_for_primary_verdict": True,
        "row_count": int(len(values)),
        "cohorts": [
            {
                "cohort_id": str(cohort),
                "row_count": int(len(group)),
                "models": sorted(group["model_id"].astype(str).unique().tolist()),
                "response_definitions": sorted(
                    set(zip(group["response_name"].astype(str), group["response_scale"].astype(str)))
                ),
            }
            for cohort, group in values.groupby("cohort_id", sort=True, observed=True)
        ],
    }


def determine_exp011_conclusion(
    *,
    cell_line_gate: Mapping[str, object],
    candidate_pdo: Mapping[str, object] | None = None,
    source_pdo: Mapping[str, object] | None = None,
    ridge_pdo: Mapping[str, object] | None = None,
    candidate_minus_source: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Apply the one-way EXP-011 verdict tree without reading exploratory data."""

    if not bool(cell_line_gate["passed"]):
        return {
            "status": Exp011Conclusion.CELL_LINE_GATE_FAILED.value,
            "reason": "cell-line generalization gate failed; PDO results are intentionally not interpreted",
        }
    if any(value is None for value in (candidate_pdo, source_pdo, ridge_pdo, candidate_minus_source)):
        raise EvaluationContractError("PDO verdict requires all three primary models and paired bootstrap")
    candidate_score = candidate_pdo["macro_spearman_equal_cohort"]
    source_score = source_pdo["macro_spearman_equal_cohort"]
    ridge_score = ridge_pdo["macro_spearman_equal_cohort"]
    if candidate_score is None or source_score is None or ridge_score is None:
        return {"status": Exp011Conclusion.NO_GO.value, "reason": "undefined primary PDO endpoint"}
    per_cohort = {str(row["cohort_id"]): row["macro_spearman"] for row in candidate_pdo["per_cohort"]}
    direct_cohorts_positive = bool(per_cohort) and all(
        value is not None and float(value) > 0.0 for value in per_cohort.values()
    )
    delta = float(candidate_minus_source["point_delta"])
    ci_lower = float(candidate_minus_source["ci_95"][0])
    materially_improves_source = delta >= 0.05 and ci_lower > 0.0
    no_worse_than_ridge = float(candidate_score) - float(ridge_score) >= -0.02
    if materially_improves_source and no_worse_than_ridge and direct_cohorts_positive:
        status = Exp011Conclusion.PROMISING_MULTI_PDO
        reason = "M3 adapter improves M2, is not materially below Ridge, and both direct PDO cohorts are positive"
    elif ci_lower > 0.0:
        status = Exp011Conclusion.TRANSFER_ONLY
        reason = "M3 improves M2 but does not satisfy all multi-PDO/Ridge requirements"
    else:
        status = Exp011Conclusion.NO_GO
        reason = "M3 does not show a positive paired PDO improvement over M2"
    return {
        "status": status.value,
        "reason": reason,
        "candidate_minus_source": delta,
        "candidate_minus_source_ci_lower": ci_lower,
        "candidate_minus_ridge": float(candidate_score) - float(ridge_score),
        "direct_cohorts_positive": direct_cohorts_positive,
    }


def evaluate_exp011_predictions(
    frame: pd.DataFrame,
    *,
    source_model_id: str = "M2",
    candidate_model_id: str = "M3_adapter",
    ridge_model_id: str = "B1_ridge",
    primary_cohorts: Sequence[str] = PRIMARY_PDO_COHORTS,
    bootstrap_repeats: int = 2000,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Build the complete, read-only EXP-011 summary from a prediction table."""

    values = validate_prediction_frame(frame)
    cell_gate = evaluate_cell_line_gate(
        values, source_model_id=source_model_id, ridge_model_id=ridge_model_id, repeats=bootstrap_repeats, seed=seed
    )
    report: dict[str, object] = {
        "format": "exp011_evaluation_bundle_v1",
        "experiment_id": EXPERIMENT_ID,
        "prediction_schema": prediction_schema(),
        "labels_used_for_model_selection": {
            "gdsc_cold_cell": True,
            "gdsc_cold_drug": True,
            "pdo_primary": True,
            "prism_frozen": False,
            "external_exploratory": False,
        },
        "cell_line_gate": cell_gate,
        "cold_drug_stress_test": cold_drug_report(values, source_model_id=source_model_id),
        "external_exploratory": external_exploratory_manifest(values),
    }
    if not cell_gate["passed"]:
        report["conclusion"] = determine_exp011_conclusion(cell_line_gate=cell_gate)
        report["pdo_primary"] = {"status": "NOT_INTERPRETED_CELL_LINE_GATE_FAILED"}
        return report
    pdo_values = values[values["evaluation_role"] == "pdo_primary"]
    candidate = pdo_primary_metrics(
        _select_role_model(pdo_values, "pdo_primary", candidate_model_id), primary_cohorts=primary_cohorts
    )
    source = pdo_primary_metrics(
        _select_role_model(pdo_values, "pdo_primary", source_model_id), primary_cohorts=primary_cohorts
    )
    ridge = pdo_primary_metrics(
        _select_role_model(pdo_values, "pdo_primary", ridge_model_id), primary_cohorts=primary_cohorts
    )
    bootstrap = patient_within_cohort_cluster_bootstrap_delta(
        pdo_values,
        candidate_model_id=candidate_model_id,
        baseline_model_id=source_model_id,
        primary_cohorts=primary_cohorts,
        repeats=bootstrap_repeats,
        seed=seed,
    )
    report["pdo_primary"] = {
        candidate_model_id: candidate,
        source_model_id: source,
        ridge_model_id: ridge,
        f"{candidate_model_id}_minus_{source_model_id}": bootstrap,
    }
    report["conclusion"] = determine_exp011_conclusion(
        cell_line_gate=cell_gate,
        candidate_pdo=candidate,
        source_pdo=source,
        ridge_pdo=ridge,
        candidate_minus_source=bootstrap,
    )
    return report
