"""Controlled PDO adaptation primitives for the EXP-017 development protocol.

This module deliberately separates expression representation from weight
transfer.  It is a development runner: Zhu has already informed development,
so its output must never be labelled an external validation result.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .benchmark import _fit_b0, _fit_b1, _predict_b0, _predict_b1, _table
from .metrics_v2 import ndcg_at_k
from .chemistry import morgan_fingerprint
from .cohorts import load_registered_cohort
from .models import DualEncoderPairModel
from .metrics_v2 import evaluate_predictions
from .preprocessing import ExpressionPreprocessor
from .splits import assign_patient_folds
from .transfer_experiment import _move_to_device, _source

_ALLOWED_SCALES = frozenset({"source_log", "target_log", "samplewise_rank"})
_ALLOWED_MODES = frozenset({"head_only", "full", "shrink_to_source"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def validate_adaptation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the bounded protocol before it can consume a cohort."""
    cfg = dict(config)
    if int(cfg.get("outer_folds", 5)) != 5 or int(cfg.get("inner_folds", 3)) != 3:
        raise ValueError("EXP-017 requires outer_folds=5 and inner_folds=3")
    if cfg.get("inner_selection_endpoint", "strict_full_inchikey_7_drug_patient_ndcg_at_3") != "strict_full_inchikey_7_drug_patient_ndcg_at_3":
        raise ValueError("EXP-017 inner selection must use strict seven-drug NDCG@3")
    candidates = list(cfg.get("candidates", []))
    if not candidates or len(candidates) > 12:
        raise ValueError("EXP-017 requires one to twelve candidate configurations")
    final_seeds = cfg.get("final_init_seeds", [cfg.get("seed", 0)])
    if not isinstance(final_seeds, list) or len(final_seeds) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in final_seeds):
        raise ValueError("EXP-017 requires exactly three integer final_init_seeds")
    for item in candidates:
        if item.get("representation") not in _ALLOWED_SCALES:
            raise ValueError("unsupported representation")
        if item.get("mode") not in _ALLOWED_MODES:
            raise ValueError("unsupported adaptation mode")
        shrinkage = float(item.get("shrinkage", 0.0))
        if item["mode"] == "shrink_to_source" and shrinkage not in {0.0, .01, .1}:
            raise ValueError("shrinkage must be one of 0, 0.01, 0.1")
        if item["mode"] != "shrink_to_source" and shrinkage != 0.0:
            raise ValueError("only shrink_to_source may use non-zero shrinkage")
    budgets = [float(value) for value in cfg.get("learning_curve_budgets", [.1, .25, .5, 1.0])]
    if budgets != [.1, .25, .5, 1.0]:
        raise ValueError("learning curve budgets must be [0.1, 0.25, 0.5, 1.0]")
    return cfg


def common_gene_ids(source_preprocessor: ExpressionPreprocessor, target_gene_ids: Any) -> tuple[str, ...]:
    """Return the ordered source feature space only when it is target-observed.

    Source checkpoints have a fixed input width.  Subsetting their genes would
    silently change feature semantics, therefore a missing source feature is a
    protocol failure rather than median-imputation masquerading as a common
    feature space.
    """
    target = {str(g).strip() for g in target_gene_ids}
    missing = [g for g in source_preprocessor.gene_ids if g not in target]
    if missing:
        raise ValueError(f"target lacks {len(missing)} frozen source genes; no common fixed feature space")
    return tuple(source_preprocessor.gene_ids)


def common_source_target_gene_space(source_gene_ids: Any, target_gene_ids: Any, *, n_features: int = 512) -> tuple[str, ...]:
    """Ordered, labelled source∩target space for a checkpoint that must be retrained.

    This deliberately does not retrofit an old checkpoint: its input weights
    belong to the old gene identities.  The returned set is the only permitted
    feature universe for a new common-space source fit and every target fold.
    """
    source = tuple(str(g).strip() for g in source_gene_ids)
    target = {str(g).strip() for g in target_gene_ids}
    common = tuple(g for g in source if g in target)
    if len(common) < n_features:
        raise ValueError(f"source∩target labelled gene space has {len(common)} genes, below required {n_features}")
    return common


def train_common_space_source_checkpoint(source_bundle: Any, target_gene_ids: Any, *, output_dir: str | Path,
                                         seed: int, epochs: int = 50, device: str = "cpu",
                                         n_features: int = 512) -> dict[str, Any]:
    """Train a new source checkpoint in a frozen source∩target *unlabelled* space.

    Target response values and target sample rows are intentionally absent from
    this function.  The caller may supply target gene identifiers only.
    """
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()): raise ValueError("common-space source output must be new or empty")
    common = common_source_target_gene_space(source_bundle.gene_ids, target_gene_ids, n_features=n_features)
    source_positions = {str(g): i for i, g in enumerate(source_bundle.gene_ids)}
    raw = np.asarray(source_bundle.expression, float)[:, [source_positions[g] for g in common]]
    table = _table(source_bundle)
    patients = np.asarray(sorted(table.patient_id.astype(str).unique())); rng = np.random.default_rng(seed); rng.shuffle(patients)
    validation_patients = set(patients[:max(2, int(np.ceil(len(patients) * .2)))])
    train, validation = table[~table.patient_id.isin(validation_patients)], table[table.patient_id.isin(validation_patients)]
    prep = ExpressionPreprocessor.fit(raw[np.unique(train.expression_index.to_numpy(int))], common, n_features=n_features)
    features = prep.transform(raw, common); fps = _fingerprints(source_bundle, table.drug_id)
    model = DualEncoderPairModel(hidden_dim=32, epochs=epochs, patience=10, ranking_weight=.2, max_rank_pairs=128,
                                 batch_groups=8, seed=seed, device=device)
    model.fit(*_arrays(features, fps, train)[:3], pdo_ids=train.model_id.astype(str).to_numpy(),
              validation_data=(*_arrays(features, fps, validation)[:3], validation.model_id.astype(str).to_numpy()))
    output.mkdir(parents=True); prep.save(output / "expression_transform.npz"); model.save(output / "source_model.pt")
    source_drugs = source_bundle.drugs.drug_id.astype(str).tolist()
    receipt = {"kind": "EXP017_COMMON_SOURCE_TARGET_UNLABELLED_GENE_SPACE", "seed": int(seed),
               "common_source_target_candidate_gene_ids": list(common),
               "common_source_target_gene_ids": list(prep.gene_ids), "source_drug_ids": source_drugs,
               "source_train_patients": sorted(train.patient_id.astype(str).unique()),
               "source_validation_patients": sorted(validation_patients),
               "target_labels_consumed": False,
               "input_hashes": {"source_expression_sha256": hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
                                "target_gene_ids_sha256": hashlib.sha256("\n".join(map(str, target_gene_ids)).encode()).hexdigest()}}
    (output / "frozen_run.json").write_text(json.dumps(receipt, indent=2)); (output / "training_history.json").write_text(json.dumps(model.training_history, indent=2))
    return receipt


def strict_full_inchikey_drug_ids(bundle: Any, source_drug_ids: set[str]) -> set[str]:
    """Accept only target drug rows whose complete InChIKey exactly occurs in source."""
    drugs = bundle.drugs.copy()
    if "inchikey" not in drugs or "drug_id" not in drugs:
        raise ValueError("strict shared-drug protocol requires target drug_id and full inchikey")
    key = drugs.inchikey.astype(str).str.strip()
    if key.str.len().ne(27).any() or key.str.count("-").ne(2).any():
        raise ValueError("target drug contains missing or non-full InChIKey")
    accepted = drugs.loc[key.isin(source_drug_ids), "drug_id"].astype(str)
    if len(accepted) != 7:
        raise ValueError(f"strict source∩target full-InChIKey panel must contain exactly 7 drugs, found {len(accepted)}")
    return set(accepted)


def _inner_ndcg3_strict7(rows: pd.DataFrame, prediction: np.ndarray) -> float:
    """Inner selection endpoint: NDCG@3, always on the exact seven-drug panel."""
    observed_column = "response" if "response" in rows.columns else "observed" if "observed" in rows.columns else None
    if observed_column is None:
        raise ValueError("strict NDCG@3 rows require response or observed labels")
    scores = []
    for patient, group in rows.assign(prediction=prediction).groupby("patient_id", sort=True):
        if group.drug_id.nunique() != 7 or len(group) != 7:
            raise ValueError("inner selection requires exactly seven strict full-InChIKey drugs per PDO")
        value = ndcg_at_k(group[observed_column].to_numpy(float), group.prediction.to_numpy(float), k=3)
        if np.isfinite(value): scores.append((str(patient), float(value)))
    if not scores: raise ValueError("inner NDCG@3 has no evaluable patients")
    return float(pd.DataFrame(scores, columns=["patient", "score"]).groupby("patient").score.mean().mean())


def representation_matrix(expression: Any, gene_ids: Any, *, kind: str) -> np.ndarray:
    """Apply the registered scale representation without claiming TPM/FPKM equivalence."""
    values = np.asarray(expression, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("expression must be a finite 2D matrix")
    if kind not in _ALLOWED_SCALES:
        raise ValueError("unsupported representation")
    if kind in {"source_log", "target_log"}:
        if (values < 0).any():
            raise ValueError("log representation requires non-negative expression")
        return np.log2(values + 1.0)
    # Tie-aware average ranks within each sample.  This removes absolute scale
    # while preserving the complete per-sample ordering on the common genes.
    return pd.DataFrame(values, columns=np.asarray(gene_ids, str)).rank(axis=1, method="average", pct=True).to_numpy(float)


def fit_target_preprocessor(expression: Any, gene_ids: Any, source_preprocessor: ExpressionPreprocessor,
                            *, representation: str) -> tuple[ExpressionPreprocessor, np.ndarray]:
    """Fit a target-only transform in the checkpoint-compatible fixed gene order."""
    common = common_gene_ids(source_preprocessor, gene_ids)
    positions = {str(g): i for i, g in enumerate(np.asarray(gene_ids, str))}
    raw = np.asarray(expression, float)[:, [positions[g] for g in common]]
    shown = representation_matrix(raw, common, kind=representation)
    if representation == "source_log":
        # This is the frozen source transform control.  It deliberately does
        # not fit target centering or feature selection, so a difference from
        # target_log can be attributed to target preprocessing rather than an
        # accidental change of source weights.
        return source_preprocessor, shown
    # n_features equals the fixed source width: no fold can select a different
    # gene identity set, but center/scale remain fitted on its training rows.
    prep = ExpressionPreprocessor.fit(shown, common, n_features=len(common))
    return prep, shown


def _fingerprints(bundle: Any, ids: pd.Series) -> dict[str, np.ndarray]:
    drugs = bundle.drugs.set_index("drug_id")
    field = next((key for key in ("canonical_smiles", "canonical_isomeric_smiles", "smiles") if key in drugs), None)
    if field is None: raise ValueError("registered target SMILES required")
    return {str(d): morgan_fingerprint(str(drugs.loc[str(d), field]), radius=2, n_bits=2048).astype(float) for d in ids.unique()}


def _arrays(features: np.ndarray, fingerprints: Mapping[str, np.ndarray], rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (features[rows.expression_index.to_numpy(int)], np.vstack([fingerprints[str(x)] for x in rows.drug_id]),
            rows.response.to_numpy(float), rows.model_id.astype(str).to_numpy())


def fit_adapted_model(source_path: str | Path, x: np.ndarray, d: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      *, mode: str, shrinkage: float, seed: int, epochs: int = 50, device: str = "cpu",
                      validation_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None) -> DualEncoderPairModel:
    """Fit one source-initialized adapter while preserving the source checkpoint.

    ``head_only`` freezes both encoders. ``shrink_to_source`` penalizes each
    parameter's squared departure from the loaded source state during target
    optimisation.  Label scaling is reset to target labels before optimisation.
    """
    if mode not in _ALLOWED_MODES: raise ValueError("unsupported adaptation mode")
    if shrinkage not in {0.0, .01, .1}: raise ValueError("unsupported shrinkage")
    model = _move_to_device(DualEncoderPairModel.load(Path(source_path) / "source_model.pt"), device)
    model.config.update({"seed": int(seed), "epochs": int(epochs), "patience": 10})
    # The first fit call's target-scale conversion is shared by all modes.
    if mode == "full" and shrinkage == 0:
        return model.fit(x, d, y, pdo_ids=groups, validation_data=validation_data)
    assert model.network is not None
    old_mean, old_std = model.label_mean, model.label_std
    mean, std = float(np.mean(y)), float(np.std(y)) or 1.0
    final = model.network.head[-1]
    with torch.no_grad():
        factor = old_std / std
        final.weight.mul_(factor); final.bias.mul_(factor).add_((old_mean - mean) / std)
    model.label_mean, model.label_std = mean, std
    source_state = {name: value.detach().clone() for name, value in model.network.named_parameters()}
    for name, parameter in model.network.named_parameters():
        parameter.requires_grad = not (mode == "head_only" and not name.startswith("head."))
    optimizer = torch.optim.Adam([p for p in model.network.parameters() if p.requires_grad], lr=model.config["learning_rate"])
    tx, td, ty = (torch.as_tensor(v, dtype=torch.float32, device=model.device) for v in (x, d, (y - mean) / std))
    group_pairs = model._group_rank_pairs(y, groups)
    model.training_history = []
    best_state, best_loss, stale = None, float("inf"), 0
    for epoch in range(epochs):
        losses=[]
        order=np.random.default_rng(int(seed)+epoch).permutation(len(group_pairs))
        for start in range(0,len(order),8):
            selected=[group_pairs[i] for i in order[start:start+8]]; idx=np.concatenate([entry[0] for entry in selected]); offset=0; pairs=[]
            for indices,left,right in selected:
                if len(left): pairs.append((left+offset,right+offset))
                offset += len(indices)
            index=torch.as_tensor(idx,dtype=torch.long,device=model.device); optimizer.zero_grad(); output=model.network(tx[index],td[index])
            loss=F.huber_loss(output,ty[index],delta=model.config["huber_delta"])+model.config["ranking_weight"]*model._ranking_loss(output,pair_groups=pairs)
            if mode == "shrink_to_source" and shrinkage: loss=loss+shrinkage*sum((p-source_state[n]).square().mean() for n,p in model.network.named_parameters())
            loss.backward();optimizer.step();losses.append(float(loss.detach().cpu()))
        validation_loss = model._validation_loss(validation_data)
        model.training_history.append({"epoch": epoch + 1, "mean_batch_training_loss": float(np.mean(losses)), "validation_loss": validation_loss})
        if validation_loss is not None:
            if validation_loss < best_loss:
                best_loss, stale = validation_loss, 0
                best_state = {key: value.detach().clone() for key, value in model.network.state_dict().items()}
            else:
                stale += 1
                if stale >= 10: break
    if best_state is not None: model.network.load_state_dict(best_state)
    return model


def _inner_select(train: pd.DataFrame, raw: np.ndarray, genes: np.ndarray, source_prep: ExpressionPreprocessor,
                  fps: Mapping[str, np.ndarray], source_output: Path, candidates: list[dict[str, Any]], *, seed: int, epochs: int, device: str) -> dict[str, Any]:
    scores: list[tuple[float, int, dict[str, Any]]] = []
    assignment = assign_patient_folds(train.patient_id.astype(str).tolist(), n_folds=3, seed=seed)
    for number, item in enumerate(candidates):
        fold_scores = []
        for fold in range(3):
            valid_ids = {p for p, f in assignment.patient_to_test_fold.items() if f == fold}
            inside, valid = train.loc[~train.patient_id.isin(valid_ids)], train.loc[train.patient_id.isin(valid_ids)]
            prep, shown = fit_target_preprocessor(raw[np.unique(inside.expression_index.to_numpy(int))], genes, source_prep, representation=item["representation"])
            # Refitting must use only inside patients; transform all raw rows with the fitted state.
            _, all_shown = fit_target_preprocessor(raw, genes, source_prep, representation=item["representation"])
            features = prep.transform(all_shown, source_prep.gene_ids)
            model = fit_adapted_model(source_output, *_arrays(features, fps, inside), mode=item["mode"], shrinkage=float(item.get("shrinkage", 0)), seed=seed + fold, epochs=epochs, device=device,
                                      validation_data=_arrays(features, fps, valid))
            fold_scores.append(_inner_ndcg3_strict7(valid, model.predict(*_arrays(features, fps, valid)[:2])))
        scores.append((float(np.mean(fold_scores)), number, item))
    return min(scores, key=lambda x: (-x[0], x[1]))[2]


def _early_stop_partition(train: pd.DataFrame, *, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Patient-disjoint early stopping partition contained wholly in an outer train fold."""
    patients = np.asarray(sorted(train.patient_id.astype(str).unique()))
    if len(patients) < 3: raise ValueError("outer training fold lacks patients for validation early stopping")
    rng = np.random.default_rng(seed); rng.shuffle(patients)
    validation = set(patients[:max(1, int(np.ceil(len(patients) * .2)))])
    return train[~train.patient_id.isin(validation)].copy(), train[train.patient_id.isin(validation)].copy()


def run_pdo_adaptation(config: Mapping[str, Any], *, source_output: str | Path, data_root: str | Path, output_dir: str | Path,
                       registry_path: str | Path, loader: Callable[..., Any] = load_registered_cohort) -> dict[str, Any]:
    """Nested 5x3 PDO development run with bounded adapter configurations."""
    cfg = validate_adaptation_config(config); output = Path(output_dir)
    if output.exists() and any(output.iterdir()): raise ValueError("output must be new or empty")
    source_prep, _, source_drugs, frozen = _source(Path(source_output))
    bundle = loader(str(cfg["cohort_id"]), root=Path(data_root), registry=Path(registry_path)); table = _table(bundle)
    raw, genes, fps = np.asarray(bundle.expression, float), np.asarray(bundle.gene_ids, str), _fingerprints(bundle, table.drug_id)
    # The supplied source checkpoint must already have been retrained in the
    # labelled source∩target universe.  An old source feature selection cannot
    # be silently median-imputed into this protocol.
    source_common = frozen.get("common_source_target_gene_ids")
    if not isinstance(source_common, list) or tuple(source_common) != tuple(source_prep.gene_ids):
        raise ValueError("source checkpoint lacks a matching common_source_target_gene_ids retraining receipt")
    common_gene_ids(source_prep, genes)
    strict_drugs = strict_full_inchikey_drug_ids(bundle, source_drugs)
    table = table.loc[table.drug_id.astype(str).isin(strict_drugs)].copy()
    if table.empty or table.drug_id.nunique() != 7:
        raise ValueError("strict seven-drug table construction failed")
    output.mkdir(parents=True)
    source_dir = Path(source_output)
    provenance = {"config": cfg, "source_frozen_run": frozen,
                  "source_artifact_sha256": {name: _sha256(source_dir / name) for name in ("frozen_run.json", "source_model.pt", "expression_transform.npz")},
                  "registry_sha256": _sha256(Path(registry_path)),
                  "input_hashes": {"target_expression_sha256": hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
                                   "target_gene_ids_sha256": hashlib.sha256("\n".join(genes).encode()).hexdigest(),
                                   "target_pairs_sha256": hashlib.sha256(bundle.pairs.to_csv(index=False).encode()).hexdigest()},
                  "code_sha256": _sha256(Path(__file__))}
    (output / "frozen_config.json").write_text(json.dumps(provenance, indent=2))
    all_rows, selections, curves = [], [], []
    seed = int(cfg.get("seed", 0)); epochs, device = int(cfg.get("epochs", 50)), str(cfg.get("device", "cpu"))
    assignment = assign_patient_folds(table.patient_id.astype(str).tolist(), n_folds=5, seed=seed)
    for fold in range(5):
        test_ids = {p for p, f in assignment.patient_to_test_fold.items() if f == fold}
        train, test = table.loc[~table.patient_id.isin(test_ids)], table.loc[table.patient_id.isin(test_ids)]
        selected = _inner_select(train, raw, genes, source_prep, fps, Path(source_output), list(cfg["candidates"]), seed=seed + fold, epochs=epochs, device=device)
        early_train, early_valid = _early_stop_partition(train, seed=seed + fold)
        # B1 uses target-log features fitted only on the early-training patients.
        b1_prep, _ = fit_target_preprocessor(raw[np.unique(early_train.expression_index.to_numpy(int))], genes, source_prep, representation="target_log")
        _, b1_shown = fit_target_preprocessor(raw, genes, source_prep, representation="target_log")
        b1_features = b1_prep.transform(b1_shown, source_prep.gene_ids)
        b0 = _predict_b0(_fit_b0(train), test.drug_id)
        b1_model = _fit_b1(b1_features, train, alpha=10., min_train=2); b1 = _predict_b1(b1_model, b1_features, test)
        common = {"patient_id": test.patient_id, "model_id": test.model_id, "drug_id": test.drug_id, "observed": test.response, "fold": fold}
        # B0/B1 are fold-identical; learned methods carry their actual init seed.
        all_rows.append(pd.DataFrame(common | {"predicted": b0, "baseline": b0, "algorithm": "B0"}))
        all_rows.append(pd.DataFrame(common | {"predicted": b1, "baseline": b0, "algorithm": "B1"}))
        native_shown = representation_matrix(raw, genes, kind="target_log")
        native = ExpressionPreprocessor.fit(native_shown[np.unique(early_train.expression_index.to_numpy(int))], genes, n_features=512)
        native_features = native.transform(native_shown, genes)
        # Candidate matrix is deliberately emitted for every declared adapter;
        # selection is represented separately as M3_selected and never hides
        # a negative candidate behind a test-driven choice.
        for init_seed in cfg["final_init_seeds"]:
            random_native = DualEncoderPairModel(hidden_dim=32, epochs=epochs, patience=10, ranking_weight=.2, max_rank_pairs=128, batch_groups=8, seed=int(init_seed), device=device)
            random_native.fit(*_arrays(native_features, fps, early_train)[:3], pdo_ids=early_train.model_id.astype(str).to_numpy(), validation_data=_arrays(native_features, fps, early_valid))
            source_prep_features = source_prep.transform(representation_matrix(raw, genes, kind="source_log"), genes)
            random_source = DualEncoderPairModel(hidden_dim=32, epochs=epochs, patience=10, ranking_weight=.2, max_rank_pairs=128, batch_groups=8, seed=int(init_seed), device=device)
            random_source.fit(*_arrays(source_prep_features, fps, early_train)[:3], pdo_ids=early_train.model_id.astype(str).to_numpy(), validation_data=_arrays(source_prep_features, fps, early_valid))
            source_model = _move_to_device(DualEncoderPairModel.load(Path(source_output) / "source_model.pt"), device)
            candidate_models = {"M1": random_native, "M1_source_scaled": random_source, "M2": source_model}
            candidate_features = {}
            for number, item in enumerate(cfg["candidates"]):
                candidate_prep, _ = fit_target_preprocessor(raw[np.unique(early_train.expression_index.to_numpy(int))], genes, source_prep, representation=item["representation"])
                _, candidate_shown = fit_target_preprocessor(raw, genes, source_prep, representation=item["representation"])
                candidate_features[number] = candidate_prep.transform(candidate_shown, source_prep.gene_ids)
                key=f"adapter_{number}_{item['representation']}_{item['mode']}_{item.get('shrinkage', 0)}"
                candidate_models[key] = fit_adapted_model(source_output, *_arrays(candidate_features[number], fps, early_train), mode=item["mode"], shrinkage=float(item.get("shrinkage", 0)), seed=int(init_seed), epochs=epochs, device=device, validation_data=_arrays(candidate_features[number], fps, early_valid))
            selected_key = f"adapter_{cfg['candidates'].index(selected)}_{selected['representation']}_{selected['mode']}_{selected.get('shrinkage', 0)}"
            for name, model in candidate_models.items():
                if name == "M2" and init_seed != cfg["final_init_seeds"][0]:
                    continue
                feature_set = native_features if name == "M1" else source_prep_features if name in {"M1_source_scaled", "M2"} else candidate_features[int(name.split("_")[1])]
                prediction = model.predict(*_arrays(feature_set, fps, test)[:2])
                all_rows.append(pd.DataFrame(common | {"predicted": prediction, "baseline": b0, "algorithm": name, "init_seed": int(init_seed) if name != "M2" else -1}))
                if name == selected_key:
                    all_rows.append(pd.DataFrame(common | {"predicted": prediction, "baseline": b0, "algorithm": "M3_selected", "init_seed": int(init_seed)}))
                    model.save(output / f"fold_{fold}_seed_{init_seed}_selected_adapter.pt")
        native.save(output / f"fold_{fold}_native_preprocessor.npz"); b1_prep.save(output / f"fold_{fold}_preprocessor.npz"); selections.append({"fold": fold, "selected": selected, "early_stopping_patients": sorted(early_valid.patient_id.astype(str).unique())})
        patients = sorted(early_train.patient_id.astype(str).unique())
        for budget in cfg["learning_curve_budgets"]:
            count = int(np.floor(len(patients) * float(budget)))
            if count < 6:
                curves.append({"fold": fold, "budget": budget, "status": "SKIPPED_LT_6_TRAIN_PATIENTS"})
                continue
            chosen_patients = np.random.default_rng(seed + fold).choice(patients, size=count, replace=False)
            budget_train = early_train.loc[early_train.patient_id.astype(str).isin(chosen_patients)]
            selected_number=cfg["candidates"].index(selected)
            budget_prep,_=fit_target_preprocessor(raw[np.unique(budget_train.expression_index.to_numpy(int))],genes,source_prep,representation=selected["representation"])
            _,budget_shown=fit_target_preprocessor(raw,genes,source_prep,representation=selected["representation"]); budget_features=budget_prep.transform(budget_shown,source_prep.gene_ids)
            budget_model = fit_adapted_model(source_output, *_arrays(budget_features, fps, budget_train), mode=selected["mode"], shrinkage=float(selected.get("shrinkage", 0)), seed=seed, epochs=epochs, device=device, validation_data=_arrays(budget_features, fps, early_valid))
            budget_pred = budget_model.predict(*_arrays(budget_features, fps, test)[:2])
            curves.append({"fold": fold, "budget": budget, "status": "COMPLETE", "training_patients": count,
                           "outer_test_ndcg3_strict7": _inner_ndcg3_strict7(test, budget_pred)})
    oof = pd.concat(all_rows, ignore_index=True); oof.to_parquet(output / "outer_oof_predictions.parquet", index=False)
    metric_rows = oof.copy(); metric_rows["init_seed"] = metric_rows.get("init_seed", pd.Series(index=metric_rows.index, dtype=float)).fillna(-1).astype(int)
    metrics = {f"{name}|seed_{init}": evaluate_predictions(rows.drop(columns=["algorithm", "init_seed"]), seed=seed + max(int(init), 0), resamples=int(cfg.get("bootstrap_resamples", 2000)))
               for (name, init), rows in metric_rows.groupby(["algorithm", "init_seed"], sort=True)}
    contrasts = {}
    selected_rows = oof[oof.algorithm.eq("M3_selected")]
    for comparator in ["B0", "M1"]:
        reference = oof[oof.algorithm.eq(comparator)]
        if comparator == "B0":
            # B0 has no initialization; pair it to each M3 seed by identity.
            merged = selected_rows.drop(columns="baseline").merge(reference[["patient_id", "model_id", "drug_id", "fold", "predicted"]].rename(columns={"predicted": "baseline"}), on=["patient_id", "model_id", "drug_id", "fold"], validate="many_to_one")
        else:
            merged = selected_rows.drop(columns="baseline").merge(reference[["patient_id", "model_id", "drug_id", "fold", "init_seed", "predicted"]].rename(columns={"predicted": "baseline"}), on=["patient_id", "model_id", "drug_id", "fold", "init_seed"], validate="one_to_one")
        contrasts[f"M3_selected_minus_{comparator}"] = {str(init): evaluate_predictions(part.drop(columns=["algorithm", "init_seed"]), seed=seed + int(init), resamples=int(cfg.get("bootstrap_resamples", 2000))) for init, part in merged.groupby("init_seed")}
    dominance = {}
    for init, selected_part in selected_rows.groupby("init_seed"):
        base_b0 = oof[oof.algorithm.eq("B0")]
        base_m1 = oof[(oof.algorithm.eq("M1")) & (oof.init_seed.eq(init))]
        joined = selected_part.merge(base_b0[["patient_id","model_id","drug_id","predicted"]].rename(columns={"predicted":"b0"}),on=["patient_id","model_id","drug_id"],validate="many_to_one").merge(base_m1[["patient_id","model_id","drug_id","predicted"]].rename(columns={"predicted":"m1"}),on=["patient_id","model_id","drug_id"],validate="one_to_one")
        rows=[]
        for patient, group in joined.groupby("patient_id", sort=True):
            m3=_inner_ndcg3_strict7(group, group.predicted.to_numpy(float)); b0score=_inner_ndcg3_strict7(group, group.b0.to_numpy(float)); m1score=_inner_ndcg3_strict7(group, group.m1.to_numpy(float))
            rows.append({"patient_id":patient,"m3_minus_max_b0_m1":m3-max(b0score,m1score)})
        values=pd.DataFrame(rows)
        dominance[str(init)]={"patientwise_m3_minus_max_b0_m1_mean":float(values.m3_minus_max_b0_m1.mean()),"negative_transfer_patient_fraction":float((values.m3_minus_max_b0_m1<0).mean()),"patients":rows}
    report = {"status": "DEVELOPMENT_ONLY_NOT_EXTERNAL", "selected_candidates": selections, "learning_curve": curves,
              "source_checkpoint_shared": True, "metrics": metrics, "paired_contrasts": contrasts,
              "m3_vs_patientwise_max_b0_m1": dominance,
              "negative_transfer_definition": "patient-level M3_selected minus M1 NDCG@3 or NDCG@5 difference below zero in paired OOF"}
    (output / "run_summary.json").write_text(json.dumps(report, indent=2)); return report
