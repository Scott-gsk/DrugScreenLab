"""Patient-level evaluation for models frozen without target drug-response labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _rho(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, str | None]:
    if len(observed) < 3:
        return float("nan"), "FEWER_THAN_THREE_DRUGS"
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        return float("nan"), "NONFINITE_VALUE"
    if np.ptp(observed) == 0:
        return float("nan"), "CONSTANT_OBSERVED"
    if np.ptp(predicted) == 0:
        return float("nan"), "CONSTANT_PREDICTED"
    value = float(spearmanr(observed, predicted).statistic)
    return (value, None) if np.isfinite(value) else (float("nan"), "UNDEFINED_CORRELATION")


def score_zero_target_predictions(predictions: pd.DataFrame,
                                  outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join frozen predictions to outcomes by identity and score each patient."""
    prediction_columns = (
        "patient_id", "drug_id", "predicted_source_pharmaformer", "predicted_source_train_mean"
    )
    outcome_columns = ("patient_id", "drug_id", "response")
    if set(prediction_columns) - set(predictions.columns) or set(outcome_columns) - set(outcomes.columns):
        raise ValueError("prediction or outcome table is missing required columns")
    pred = predictions[list(prediction_columns)].copy()
    target = outcomes[["patient_id", "drug_id", "response"]].copy()
    for frame in (pred, target):
        frame["patient_id"] = frame["patient_id"].astype(str)
        frame["drug_id"] = frame["drug_id"].astype(str)
    if pred.duplicated(["patient_id", "drug_id"]).any() or target.duplicated(["patient_id", "drug_id"]).any():
        raise ValueError("duplicate patient/drug keys")
    if not np.isfinite(pred[["predicted_source_pharmaformer", "predicted_source_train_mean"]]
                       .to_numpy(dtype=float)).all():
        raise ValueError("frozen predictions contain missing or non-finite values")
    if target["response"].isna().any() or not np.isfinite(target["response"].to_numpy(dtype=float)).all():
        raise ValueError("target response contains missing or non-finite values")
    scored_pairs = target.merge(pred, on=["patient_id", "drug_id"], how="left",
                                validate="one_to_one", indicator=True)
    if not scored_pairs["_merge"].eq("both").all():
        raise ValueError("some observed target patient/drug keys lack frozen predictions")
    scored_pairs = scored_pairs.drop(columns="_merge")
    rows = []
    for patient_id, group in scored_pairs.groupby("patient_id", sort=True):
        observed = group["response"].to_numpy(dtype=np.float64)
        model_rho, model_reason = _rho(
            observed, group["predicted_source_pharmaformer"].to_numpy(dtype=np.float64))
        mean_rho, mean_reason = _rho(
            observed, group["predicted_source_train_mean"].to_numpy(dtype=np.float64))
        rows.append({"patient_id": patient_id, "n_drugs": int(group["drug_id"].nunique()),
                     "spearman_source_pharmaformer": model_rho,
                     "reason_source_pharmaformer": model_reason,
                     "spearman_source_train_mean": mean_rho,
                     "reason_source_train_mean": mean_reason,
                     "difference_model_minus_mean": model_rho - mean_rho})
    return scored_pairs, pd.DataFrame(rows)


def paired_resampling_summary(unit_scores: pd.DataFrame, *, seed: int = 20260922,
                              bootstrap_replicates: int = 2000,
                              sign_flip_replicates: int = 10000) -> dict:
    values = unit_scores["difference_model_minus_mean"].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("no patients have valid paired ranking scores")
    rng = np.random.default_rng(seed)
    draws = np.empty(bootstrap_replicates, dtype=np.float64)
    for index in range(bootstrap_replicates):
        draws[index] = values[rng.integers(0, len(values), len(values))].mean()
    signs = rng.choice(np.array([-1.0, 1.0]), size=(sign_flip_replicates, len(values)))
    null = np.abs((signs * values[None, :]).mean(axis=1))
    observed = abs(float(values.mean()))
    return {
        "estimate_model_minus_source_mean": float(values.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "sign_flip_p_two_sided": float((1 + np.count_nonzero(null >= observed))
                                        / (sign_flip_replicates + 1)),
        "n_patients": int(len(values)), "bootstrap_replicates": bootstrap_replicates,
        "sign_flip_replicates": sign_flip_replicates, "seed": seed,
    }


def expression_identity_permutation_test(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    permutations: int = 10000,
    seed: int = 20260922,
) -> tuple[dict, pd.DataFrame]:
    """Test patient/expression alignment by permuting unique patient expression donors."""
    needed_pred = ("expression_donor_patient_id", "drug_id", "predicted_source_pharmaformer")
    needed_outcome = ("patient_id", "drug_id", "response")
    if set(needed_pred) - set(predictions.columns) or set(needed_outcome) - set(outcomes.columns):
        raise ValueError("permutation inputs are missing required identity or value columns")
    pred = predictions[list(needed_pred)].copy()
    target = outcomes[["patient_id", "drug_id", "response"]].copy()
    pred["expression_donor_patient_id"] = pred["expression_donor_patient_id"].astype(str)
    pred["drug_id"] = pred["drug_id"].astype(str)
    target["patient_id"] = target["patient_id"].astype(str)
    target["drug_id"] = target["drug_id"].astype(str)
    if pred.duplicated(["expression_donor_patient_id", "drug_id"]).any():
        raise ValueError("duplicate expression-donor/drug predictions")
    if not np.isfinite(pred["predicted_source_pharmaformer"].to_numpy(dtype=float)).all():
        raise ValueError("expression-donor predictions contain missing or non-finite values")
    if target.duplicated(["patient_id", "drug_id"]).any():
        raise ValueError("duplicate target patient/drug keys")
    if target["response"].isna().any() or not np.isfinite(target["response"].to_numpy(dtype=float)).all():
        raise ValueError("target response contains missing or non-finite values")
    recipients = sorted(target["patient_id"].unique())
    donors = sorted(pred["expression_donor_patient_id"].unique())
    if recipients != donors:
        raise ValueError("expression permutation requires exactly the same unique patient identities")
    if permutations < 1:
        raise ValueError("permutation count must be positive")

    donor_predictions = {
        donor: group.set_index("drug_id")["predicted_source_pharmaformer"]
        for donor, group in pred.groupby("expression_donor_patient_id", sort=True)
    }
    outcome_groups = {patient: group for patient, group in target.groupby("patient_id", sort=True)}
    rho_matrix = np.full((len(recipients), len(donors)), np.nan, dtype=np.float64)
    for recipient_index, patient in enumerate(recipients):
        group = outcome_groups[patient]
        observed = group["response"].to_numpy(dtype=np.float64)
        drug_ids = group["drug_id"].tolist()
        for donor_index, donor in enumerate(donors):
            predicted_series = donor_predictions[donor]
            if not set(drug_ids) <= set(predicted_series.index.astype(str)):
                raise ValueError(f"donor {donor} lacks a drug measured for patient {patient}")
            predicted = predicted_series.reindex(drug_ids).to_numpy(dtype=np.float64)
            rho_matrix[recipient_index, donor_index] = _rho(observed, predicted)[0]

    diagonal = rho_matrix[np.arange(len(recipients)), np.arange(len(recipients))]
    fixed_recipients = np.isfinite(diagonal)
    if not fixed_recipients.any():
        raise ValueError("no valid patient score for the true expression alignment")
    observed_macro = float(diagonal[fixed_recipients].mean())
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(permutations):
        assignment = rng.permutation(len(donors))
        draw = rho_matrix[np.arange(len(recipients)), assignment]
        valid = np.isfinite(draw)
        comparable = bool(valid[fixed_recipients].all())
        rows.append({
            "permutation": index,
            "macro_spearman": float(draw[fixed_recipients].mean()) if comparable else np.nan,
            "n_fixed_patients_scored": int(fixed_recipients.sum()),
            "n_finite_donor_patient_scores": int(valid.sum()),
            "comparable_to_observed_denominator": comparable,
        })
    null_scores = pd.DataFrame(rows)
    valid_null = null_scores.loc[null_scores["comparable_to_observed_denominator"], "macro_spearman"]
    if not len(valid_null):
        raise ValueError("no expression permutations preserve the observed valid-patient denominator")
    summary = {
        "observed_true_alignment_macro_spearman": observed_macro,
        "permutation_null_mean": float(valid_null.mean()),
        "permutation_null_median": float(valid_null.median()),
        "permutation_null_95th_percentile": float(valid_null.quantile(0.95)),
        "upper_tail_empirical_p": float((1 + np.count_nonzero(valid_null.to_numpy() >= observed_macro))
                                        / (len(valid_null) + 1)),
        "n_fixed_patients_scored": int(fixed_recipients.sum()),
        "n_permutations_requested": permutations,
        "n_permutations_with_comparable_denominator": int(len(valid_null)),
        "seed": seed,
    }
    return summary, null_scores
