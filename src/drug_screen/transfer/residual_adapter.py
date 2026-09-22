"""Low-capacity, cross-fitted residual adaptation control for transfer scores."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


class ResidualAdapter:
    """Calibrate frozen scores then add an externally fixed additive Ridge residual.

    This is an ablation for over-adaptation, not an established novel method.
    """
    def __init__(self, *, alpha: float = 10.0, lambda_: float = 1.0, seed: int = 0):
        if not np.isfinite(alpha) or alpha < 0 or lambda_ not in {0.0, 0.1, 1.0}:
            raise ValueError("alpha must be non-negative and lambda_ must be externally fixed to 0, .1, or 1")
        self.alpha, self.lambda_, self.seed = float(alpha), float(lambda_), int(seed)
        self.a = self.b = None; self.coefficients = self.intercept = None
        self.sample_dim = self.drug_dim = None; self.crossfit_audit: dict[str, Any] | None = None

    @staticmethod
    def _inputs(source_score, x, drugfeat, y=None, groups=None):
        score = np.asarray(source_score, float).reshape(-1); sample, drug = np.asarray(x, float), np.asarray(drugfeat, float)
        if sample.ndim != 2 or drug.ndim != 2 or len(score) == 0 or len(sample) != len(score) or len(drug) != len(score) or not np.isfinite(score).all() or not np.isfinite(sample).all() or not np.isfinite(drug).all(): raise ValueError("finite aligned source_score, x, and drugfeat arrays required")
        if y is None: return score, sample, drug
        target, ids = np.asarray(y, float).reshape(-1), np.asarray(groups).reshape(-1)
        if len(target) != len(score) or len(ids) != len(score) or not np.isfinite(target).all() or any(not str(group).strip() for group in ids): raise ValueError("finite aligned y and non-empty groups required")
        return score, sample, drug, target, ids.astype(str)

    @staticmethod
    def _calibration(score, y):
        centered = score-score.mean(); denominator = centered @ centered
        a = max(0.0, float(centered @ (y-y.mean()) / denominator)) if denominator else 0.0
        return a, float(y.mean()-a*score.mean())

    def fit(self, source_score, x, drugfeat, y, groups):
        score, sample, drug, target, ids = self._inputs(source_score, x, drugfeat, y, groups)
        unique = np.array(sorted(set(ids)))
        if len(unique) < 6: raise ValueError("cross-fitted residuals require at least six patient groups")
        rng = np.random.default_rng(self.seed); rng.shuffle(unique); folds = {group: i % 3 for i, group in enumerate(unique)}
        oof = np.empty(len(target)); audit = {}
        for fold in range(3):
            held = np.array([folds[group] == fold for group in ids]); a, b = self._calibration(score[~held], target[~held]); oof[held] = target[held] - (a*score[held]+b)
            audit[str(fold)] = {"train_groups": sorted(set(ids[~held])), "heldout_groups": sorted(set(ids[held])), "a": a, "b": b}
        design = np.concatenate((sample, drug), axis=1); mean = design.mean(0); centered = design-mean
        self.coefficients = np.linalg.solve(centered.T@centered + self.alpha*np.eye(design.shape[1]), centered.T@(oof-oof.mean()))
        self.intercept = float(oof.mean()-mean@self.coefficients); self.a, self.b = self._calibration(score, target)
        self.sample_dim, self.drug_dim = sample.shape[1], drug.shape[1]; self.crossfit_audit = {"n_folds": 3, "folds": audit, "oof_residuals": oof.tolist()}
        return self

    def predict(self, source_score, x, drugfeat):
        if self.a is None: raise RuntimeError("adapter is not fitted")
        score, sample, drug = self._inputs(source_score, x, drugfeat)
        if (sample.shape[1], drug.shape[1]) != (self.sample_dim, self.drug_dim): raise ValueError("feature dimensions differ from fitted adapter")
        residual = np.concatenate((sample, drug), axis=1)@self.coefficients+self.intercept
        return self.a*score+self.b+self.lambda_*residual

    def explain(self):
        if self.a is None: raise RuntimeError("adapter is not fitted")
        return {"calibration_a": self.a, "calibration_b": self.b, "lambda": self.lambda_, "ridge_alpha": self.alpha, "coefficients": self.coefficients.copy(), "intercept": self.intercept, "crossfit_audit": self.crossfit_audit}

    def save(self, path: str | Path):
        with Path(path).open("wb") as handle: pickle.dump(self.__dict__, handle)

    @classmethod
    def load(cls, path: str | Path):
        with Path(path).open("rb") as handle: state = pickle.load(handle)
        model = cls(alpha=state["alpha"], lambda_=state["lambda_"], seed=state["seed"]); model.__dict__.update(state); return model
