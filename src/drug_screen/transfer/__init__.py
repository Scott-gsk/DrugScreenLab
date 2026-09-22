"""Cell-line to organoid drug-response transfer primitives for EXP-013."""

from .baselines import fit_b0_drug_means, predict_b0, fit_b1_per_drug_ridge, predict_b1
from .contract import TransferObservation, assert_higher_is_sensitive, eligibility_report
from .metrics import TransferVerdict, evaluate_predictions, ndcg_at_k, spearman
from .splits import PatientFoldAssignment, assign_patient_folds, assert_patient_isolation

__all__ = [
    "PatientFoldAssignment",
    "TransferObservation",
    "TransferVerdict",
    "assert_higher_is_sensitive",
    "assert_patient_isolation",
    "assign_patient_folds",
    "eligibility_report",
    "evaluate_predictions",
    "fit_b0_drug_means",
    "fit_b1_per_drug_ridge",
    "ndcg_at_k",
    "predict_b0",
    "predict_b1",
    "spearman",
]
