"""Run one preregistered EXP-039 fold for bidirectional gene-ranking supervision."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from drug_screen.evaluation.reversal import reversal_score
from drug_screen.evaluation.reversal_validation import ReversalValidationCache
from drug_screen.models.conditional_perturbation import SharedEffectInteraction


SEEDS = (20260922, 20260923, 20260924)


@dataclass(frozen=True)
class Candidate:
    rank: int
    weight_decay: float
    ranking_weight: float

    @property
    def key(self) -> str:
        return f"r{self.rank}_wd{self.weight_decay:g}_rank{self.ranking_weight:g}"


CANDIDATES = tuple(
    Candidate(rank, decay, weight)
    for rank in (16, 32)
    for decay in (1e-4, 1e-3)
    for weight in (0.0, 0.1, 0.3)
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_validation_cache_gate(receipt_path: Path, *, code_files: list[Path],
                                 input_files: list[Path]) -> str:
    """Require a successful, still-current 15-fold cache-equivalence receipt."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("experiment_id") != "EXP-039"
            or receipt.get("status") != "B5_VALIDATION_CACHE_EQUIVALENCE_GATE_PASS"
            or receipt.get("n_inner_folds") != 15
            or receipt.get("all_equivalence_checks_passed") is not True):
        raise ValueError("validation cache gate receipt is absent, incomplete, or failed")

    root = Path(__file__).resolve().parents[2]
    code_hashes = receipt.get("code_sha256", {})
    input_hashes = receipt.get("input_sha256", {})
    required_code = {path.resolve().relative_to(root).as_posix() for path in code_files}
    required_inputs = {path.resolve().relative_to(root).as_posix() for path in input_files}
    if not required_code.issubset(code_hashes) or not required_inputs.issubset(input_hashes):
        raise ValueError("cache gate receipt omits required code or input bindings")
    for relative, expected in code_hashes.items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"cache gate code binding differs: {relative}")
    for relative, expected in input_hashes.items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"cache gate input binding differs: {relative}")
    config = receipt.get("config")
    expected_config_sha = hashlib.sha256(json.dumps(
        config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if receipt.get("config_sha256") != expected_config_sha:
        raise ValueError("cache gate configuration hash is invalid")
    return sha256(receipt_path)


def verify_run_provenance(run_root: Path) -> str:
    provenance_path = run_root / "run_provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError("formal EXP-039 run provenance must be captured before training")
    receipt = json.loads(provenance_path.read_text(encoding="utf-8"))
    if receipt.get("experiment_id") != "EXP-039" or receipt.get("status") != "RUN_PROVENANCE_CAPTURED":
        raise ValueError("run provenance does not identify this approved experiment")
    for name, digest in receipt["code_sha256"].items():
        path = Path(name)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"code changed after run provenance capture: {name}")
    for name, digest in receipt["input_sha256"].items():
        path = Path(name)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"frozen input changed after run provenance capture: {name}")
    return sha256(provenance_path)


def load_exp037_helpers():
    path = Path(__file__).with_name("exp037_conditional_fold.py")
    spec = importlib.util.spec_from_file_location("exp037_protocol_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen EXP-037 data and preprocessing helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def masked_softmax_cross_entropy(target: torch.Tensor, prediction: torch.Tensor,
                                 reliable: torch.Tensor, direction: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-row target-to-prediction cross entropy and rows with >=2 genes."""
    if target.ndim != 2 or target.shape != prediction.shape or target.shape != reliable.shape:
        raise ValueError("ranking loss expects aligned batch-by-gene matrices")
    valid = reliable.sum(dim=1) >= 2
    target_logits = direction * target
    prediction_logits = direction * prediction
    floor = torch.finfo(target.dtype).min
    target_logits = target_logits.masked_fill(~reliable, floor)
    prediction_logits = prediction_logits.masked_fill(~reliable, floor)
    target_probability = torch.softmax(target_logits, dim=1)
    prediction_log_probability = torch.log_softmax(prediction_logits, dim=1)
    per_row = -(target_probability * prediction_log_probability).sum(dim=1)
    return per_row, valid


def bidirectional_ranking_loss(target_raw: torch.Tensor, prediction: torch.Tensor,
                               gene_scale: torch.Tensor,
                               control_noise_threshold: torch.Tensor) -> tuple[torch.Tensor, dict[str, int]]:
    """B5 loss: cross entropy of reliable up and down gene-rank distributions."""
    if target_raw.ndim != 2 or target_raw.shape != prediction.shape:
        raise ValueError("raw target and model output shapes differ")
    if gene_scale.shape != (target_raw.shape[1],) or control_noise_threshold.shape != gene_scale.shape:
        raise ValueError("training-side gene scale or control-noise axis differs")
    target_z = target_raw / gene_scale
    prediction_z = prediction
    reliable_up = target_raw > control_noise_threshold
    reliable_down = target_raw < -control_noise_threshold
    up_loss, up_valid = masked_softmax_cross_entropy(target_z, prediction_z, reliable_up, 1.0)
    down_loss, down_valid = masked_softmax_cross_entropy(target_z, prediction_z, reliable_down, -1.0)
    available = up_valid.to(target_raw.dtype) + down_valid.to(target_raw.dtype)
    per_condition = (up_loss * up_valid + down_loss * down_valid) / available.clamp_min(1.0)
    valid = available > 0
    if bool(valid.any()):
        loss = per_condition[valid].mean()
    else:
        loss = prediction.new_zeros(())
    return loss, {
        "conditions_with_up_direction": int(up_valid.sum().item()),
        "conditions_with_down_direction": int(down_valid.sum().item()),
        "conditions_with_any_direction": int(valid.sum().item()),
    }


def predict_delta(model: SharedEffectInteraction, data, index: np.ndarray, prep,
                  device: torch.device, batch_size: int) -> np.ndarray:
    """Reference inference path, including unused target tensor construction."""
    model.eval()
    output = np.empty((len(index), 978), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(index), batch_size):
            positions = index[start:start + batch_size]
            state, drug, exposure, _, _ = data.tensors(positions, prep, device)
            predicted = model(state, drug, exposure, use_state=True)
            output[start:start + len(positions)] = predicted.cpu().numpy() * prep.gene_scale
    return output


def predict_delta_inputs_only(model: SharedEffectInteraction, data, index: np.ndarray,
                              prep, device: torch.device,
                              batch_size: int) -> np.ndarray:
    """Predict from covariates only; never read or transfer validation targets."""
    model.eval()
    output = np.empty((len(index), 978), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(index), batch_size):
            positions = index[start:start + batch_size]
            state = ((np.asarray(data.state[positions], dtype=np.float32) - prep.state_mean)
                     / prep.state_scale)
            drug = np.asarray(data.fingerprint[data.drug_index[positions]], dtype=np.float32)
            exposure = ((data.exposure[positions] - prep.exposure_mean)
                        / prep.exposure_scale)
            state_tensor, drug_tensor, exposure_tensor = (
                torch.from_numpy(values).to(device, non_blocking=True)
                for values in (state, drug, exposure)
            )
            predicted = model(state_tensor, drug_tensor, exposure_tensor, use_state=True)
            output[start:start + len(positions)] = predicted.cpu().numpy() * prep.gene_scale
    return output


def reversal_validation_metric(data, index: np.ndarray, prediction: np.ndarray,
                               genes: list[str], up: list[str], down: list[str]) -> tuple[float, pd.DataFrame]:
    keys = data.keys.iloc[index][["cell_id", "pert_id"]].reset_index(drop=True)
    observed = np.asarray(data.delta[index], dtype=np.float32)
    groups = keys.groupby(["cell_id", "pert_id"], sort=True).indices
    rows = []
    for (cell, drug), positions in groups.items():
        positions = np.asarray(positions, dtype=np.int64)
        y = observed[positions].mean(axis=0, dtype=np.float64)
        yhat = prediction[positions].mean(axis=0, dtype=np.float64)
        rows.append({"cell_id": cell, "pert_id": drug,
                     "measured_reversal": reversal_score(y, genes, up, down),
                     "predicted_reversal": reversal_score(yhat, genes, up, down),
                     "n_conditions": len(positions)})
    drug_scores = pd.DataFrame(rows)
    cell_rows = []
    for cell, frame in drug_scores.groupby("cell_id", sort=True):
        y = frame["measured_reversal"].to_numpy(dtype=np.float64)
        yhat = frame["predicted_reversal"].to_numpy(dtype=np.float64)
        reason = None
        if len(frame) < 5:
            reason = "FEWER_THAN_FIVE_DRUGS"
        elif np.ptp(y) == 0:
            reason = "CONSTANT_MEASURED_REVERSAL"
        elif np.ptp(yhat) == 0:
            reason = "CONSTANT_PREDICTED_REVERSAL"
        score = float(spearmanr(y, yhat).statistic) if reason is None else float("nan")
        if reason is None and not np.isfinite(score):
            reason = "UNDEFINED_CORRELATION"
        cell_rows.append({"cell_id": cell, "spearman": score, "n_drugs": len(frame),
                          "reason": reason, "n_conditions": int(frame["n_conditions"].sum())})
    cell_scores = pd.DataFrame(cell_rows)
    valid = cell_scores["spearman"].dropna()
    if len(valid) != len(cell_scores) or len(valid) < 3:
        return float("nan"), cell_scores
    return float(valid.mean()), cell_scores


def train_candidate(data, train_index: np.ndarray, validation_index: np.ndarray,
                    prep, candidate: Candidate, *, seed: int, device: torch.device,
                    batch_size: int, max_epochs: int, patience: int,
                    output_gene_scale: torch.Tensor,
                    noise_threshold: torch.Tensor,
                    validation_cache: ReversalValidationCache | None = None,
                    use_cached_validation: bool = False) -> tuple[SharedEffectInteraction, dict]:
    if use_cached_validation and validation_cache is None:
        raise ValueError("cached validation was requested without a fixed-panel cache")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SharedEffectInteraction(rank=candidate.rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=candidate.weight_decay)
    best_state = None
    best_score = float("-inf")
    best_epoch = 0
    stale = 0
    start = time.perf_counter()
    direction_totals = {"conditions_with_up_direction": 0,
                        "conditions_with_down_direction": 0,
                        "conditions_with_any_direction": 0}
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = train_index.copy()
        np.random.default_rng(seed + epoch).shuffle(order)
        for offset in range(0, len(order), batch_size):
            positions = order[offset:offset + batch_size]
            state, drug, exposure, target, target_raw = data.tensors(positions, prep, device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(state, drug, exposure, use_state=True)
            loss = torch.nn.functional.huber_loss(predicted, target)
            if candidate.ranking_weight > 0:
                rank_loss, counts = bidirectional_ranking_loss(
                    target_raw, predicted, output_gene_scale, noise_threshold)
                loss = loss + candidate.ranking_weight * rank_loss
                for key, value in counts.items():
                    direction_totals[key] += value
            if not torch.isfinite(loss):
                raise FloatingPointError("EXP-039 training loss became non-finite")
            loss.backward()
            optimizer.step()
        if use_cached_validation:
            validation_prediction = predict_delta_inputs_only(
                model, data, validation_index, prep, device, batch_size)
            score, _ = validation_cache.score(validation_prediction)
        else:
            validation_prediction = predict_delta(
                model, data, validation_index, prep, device, batch_size)
            score, _ = reversal_validation_metric(data, validation_index,
                                                   validation_prediction,
                                                   data._b5_genes, data._b5_up, data._b5_down)
        if np.isfinite(score) and score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            stale = 0
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise ValueError("candidate has no finite inner validation reversal score")
    model.load_state_dict(best_state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return model, {"best_validation_reversal_spearman": best_score,
                   "best_epoch": best_epoch, "epochs_run": epoch,
                   "elapsed_seconds": time.perf_counter() - start,
                   "ranking_direction_counts_training_per_epoch": {
                       key: int(round(value / epoch)) for key, value in direction_totals.items()},
                   "n_train_conditions_per_epoch": len(train_index)}


def train_final(data, train_index: np.ndarray, prep, candidate: Candidate, *,
                seed: int, epochs: int, device: torch.device, batch_size: int,
                output_gene_scale: torch.Tensor,
                noise_threshold: torch.Tensor) -> tuple[SharedEffectInteraction, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SharedEffectInteraction(rank=candidate.rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=candidate.weight_decay)
    direction_totals = {"conditions_with_up_direction": 0,
                        "conditions_with_down_direction": 0,
                        "conditions_with_any_direction": 0}
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = train_index.copy()
        np.random.default_rng(seed + epoch).shuffle(order)
        for offset in range(0, len(order), batch_size):
            positions = order[offset:offset + batch_size]
            state, drug, exposure, target, target_raw = data.tensors(positions, prep, device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(state, drug, exposure, use_state=True)
            loss = torch.nn.functional.huber_loss(predicted, target)
            if candidate.ranking_weight > 0:
                rank_loss, counts = bidirectional_ranking_loss(
                    target_raw, predicted, output_gene_scale, noise_threshold)
                loss = loss + candidate.ranking_weight * rank_loss
                for key, value in counts.items():
                    direction_totals[key] += value
            if not torch.isfinite(loss):
                raise FloatingPointError("EXP-039 final training loss became non-finite")
            loss.backward()
            optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return model, {"epochs": epochs, "elapsed_seconds": time.perf_counter() - start,
                   "ranking_direction_counts_training_per_epoch": {
                       key: int(round(value / epochs)) for key, value in direction_totals.items()},
                   "n_train_conditions_per_epoch": len(train_index),
                   "n_trainable_parameters": sum(p.numel() for p in model.parameters())}


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"fold output already exists: {args.output_dir}")
    provenance_sha256 = verify_run_provenance(args.output_dir.parent)
    cache_gate_path = getattr(args, "validation_cache_gate_receipt", None)
    use_cached_validation = cache_gate_path is not None
    cache_gate_sha256 = None
    if use_cached_validation:
        code_files = [
            Path(__file__),
            Path(__file__).with_name("exp037_conditional_fold.py"),
            Path("src/drug_screen/evaluation/reversal_validation.py"),
            Path("src/drug_screen/models/conditional_perturbation.py"),
            Path("src/drug_screen/evaluation/reversal.py"),
            Path("tests/evaluation/test_reversal_validation_cache.py"),
            Path("tests/modeling/test_exp039_b5_validation_cache_equivalence.py"),
        ]
        input_files = [
            args.condition_dir / "manifest.json",
            args.condition_dir / "conditions.parquet",
            args.input_dir / "manifest.json",
            args.input_dir / "drugs.parquet",
            args.contract_dir / "verification.json",
            args.contract_dir / "genes.json",
            args.contract_dir / "controls.parquet",
            args.cache.parent / "asset_manifest.json",
            args.split,
            args.signature_dir / "manifest.json",
            args.signature_dir / "up_genes.json",
            args.signature_dir / "down_genes.json",
            args.validation_audit,
        ]
        cache_gate_sha256 = verify_validation_cache_gate(
            cache_gate_path, code_files=code_files, input_files=input_files)
    args.output_dir.mkdir(parents=True)
    started_at = datetime.now(timezone.utc).isoformat()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    torch.set_float32_matmul_precision("high")
    helpers = load_exp037_helpers()
    data = helpers.Arrays(args.condition_dir, args.input_dir, load_to_memory=False)
    data._b5_genes = json.loads((args.contract_dir / "genes.json").read_text(encoding="utf-8"))
    data._b5_up = json.loads((args.signature_dir / "up_genes.json").read_text(encoding="utf-8"))
    data._b5_down = json.loads((args.signature_dir / "down_genes.json").read_text(encoding="utf-8"))
    if len(data._b5_genes) != 978 or min(len(data._b5_up), len(data._b5_down)) < 20:
        raise ValueError("frozen EXP-038 signature does not satisfy the gene-axis gate")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    audit = json.loads(args.validation_audit.read_text(encoding="utf-8"))
    signature_manifest = json.loads((args.signature_dir / "manifest.json").read_text(encoding="utf-8"))
    if (audit["status"] != "B5_VALIDATION_PANEL_GATE_PASS"
            or audit["source_sha256"]["split"] != sha256(args.split)
            or audit["source_sha256"]["condition_manifest"] != sha256(args.condition_dir / "manifest.json")
            or audit["source_sha256"]["exposure"] != sha256(args.input_dir / "condition_exposure.parquet")
            or audit["source_sha256"]["signature_manifest"] != sha256(args.signature_dir / "manifest.json")
            or signature_manifest.get("status") not in {
                "SIGNATURE_READY_FOR_DEVELOPMENT", "SIGNATURE_READY_FOR_CROSS_SIGNATURE_VALIDATION"}):
        raise ValueError("B5 inner panel audit or signature does not match frozen inputs")

    fold = split["folds"][args.fold]
    train_cells = set(fold["outer_train_ids"])
    test_cells = set(fold["outer_test_ids"])
    if train_cells & test_cells:
        raise ValueError("outer train and test cells overlap")
    noise = helpers.control_noise_table(args.contract_dir, args.cache)
    inner_audit = next(row for row in audit["folds"] if int(row["fold"]) == args.fold)
    fixed_panel = np.isclose(data.exposure[:, 0], 1.0, rtol=0, atol=1e-8)
    fixed_panel &= np.isclose(data.exposure[:, 1], 24.0, rtol=0, atol=1e-8)
    inner_records = []
    candidate_rows = []
    scores_by_candidate: dict[str, list[float]] = {candidate.key: [] for candidate in CANDIDATES}
    epochs_by_candidate: dict[str, list[int]] = {candidate.key: [] for candidate in CANDIDATES}
    inner_folds = fold["label_fractions"][0]["inner_folds"]
    for inner in inner_folds:
        train_index = np.flatnonzero(data.keys["cell_id"].isin(inner["train_ids"]).to_numpy())
        validation_all = np.flatnonzero(data.keys["cell_id"].isin(inner["validation_ids"]).to_numpy())
        seen_drugs = set(data.keys.iloc[train_index]["pert_id"])
        validation_seen = validation_all[
            data.keys.iloc[validation_all]["pert_id"].isin(seen_drugs).to_numpy()]
        validation_panel = validation_seen[fixed_panel[validation_seen]]
        validation_cells = set(inner["validation_ids"])
        counts = (data.keys.iloc[validation_panel].groupby("cell_id")["pert_id"].nunique()
                  .reindex(sorted(validation_cells), fill_value=0))
        eligible_cells = set(counts[counts >= 5].index)
        validation_panel = validation_panel[
            data.keys.iloc[validation_panel]["cell_id"].isin(eligible_cells).to_numpy()]
        audit_fold = next(row for row in inner_audit["inner_folds"]
                          if int(row["inner_fold"]) == int(inner["inner_fold"]))
        if len(eligible_cells) != audit_fold["n_validation_cells_with_at_least_five_seen_drugs"]:
            raise ValueError("B5 validation cell coverage differs from the frozen audit")
        if len(eligible_cells) < 3:
            raise ValueError("fewer than three eligible validation cells")
        prep = helpers.fit_preprocess(data, train_index, noise)
        validation_cache = None
        if use_cached_validation:
            validation_cache = ReversalValidationCache(
                data.keys.iloc[validation_panel][["cell_id", "pert_id"]].reset_index(drop=True),
                np.asarray(data.delta[validation_panel], dtype=np.float32),
                data._b5_genes, data._b5_up, data._b5_down)
        gene_scale = torch.from_numpy(prep.gene_scale).to(device)
        threshold = torch.from_numpy(prep.direction_threshold).to(device)
        receipt = {"inner_fold": inner["inner_fold"],
                   "n_train_conditions": len(train_index),
                   "n_validation_conditions": len(validation_panel),
                   "n_validation_cells": len(eligible_cells),
                   "n_validation_drugs": int(data.keys.iloc[validation_panel]["pert_id"].nunique()),
                   "eligible_validation_cells": sorted(eligible_cells)}
        inner_records.append(receipt)
        for candidate in CANDIDATES:
            model, fit = train_candidate(
                data, train_index, validation_panel, prep, candidate,
                seed=SEEDS[0], device=device, batch_size=args.batch_size,
                max_epochs=args.max_epochs, patience=args.patience,
                output_gene_scale=gene_scale, noise_threshold=threshold,
                validation_cache=validation_cache,
                use_cached_validation=use_cached_validation)
            scores_by_candidate[candidate.key].append(fit["best_validation_reversal_spearman"])
            epochs_by_candidate[candidate.key].append(fit["best_epoch"])
            candidate_rows.append({"fold": args.fold, **receipt, **asdict(candidate),
                                   "candidate": candidate.key, **fit})
            print(json.dumps({"fold": args.fold, "inner_fold": inner["inner_fold"],
                              "candidate": candidate.key,
                              "validation_reversal_spearman": fit["best_validation_reversal_spearman"],
                              "best_epoch": fit["best_epoch"]}), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    baseline_candidates = [candidate for candidate in CANDIDATES if candidate.ranking_weight == 0.0]
    rank_candidates = [candidate for candidate in CANDIDATES if candidate.ranking_weight > 0.0]

    def choose(candidates: list[Candidate]) -> tuple[Candidate, float, int]:
        ranked = []
        for candidate in candidates:
            values = scores_by_candidate[candidate.key]
            if len(values) == len(inner_folds) and np.isfinite(values).all():
                ranked.append((float(np.mean(values)), candidate.key, candidate))
        if not ranked:
            raise ValueError("candidate group has no finite score in every inner fold")
        ranked.sort(key=lambda row: (-row[0], row[1]))
        score, _, selected = ranked[0]
        epochs = max(1, int(round(np.median(epochs_by_candidate[selected.key]))))
        return selected, score, epochs

    baseline_candidate, baseline_validation_score, baseline_epochs = choose(baseline_candidates)
    rank_candidate, rank_validation_score, rank_epochs = choose(rank_candidates)
    selections = {"huber": {"candidate": asdict(baseline_candidate),
                            "mean_inner_validation_reversal_spearman": baseline_validation_score,
                            "refit_epochs": baseline_epochs},
                  "b5_rank": {"candidate": asdict(rank_candidate),
                              "mean_inner_validation_reversal_spearman": rank_validation_score,
                              "refit_epochs": rank_epochs}}
    pd.DataFrame(candidate_rows).to_parquet(args.output_dir / "inner_selection.parquet", index=False)

    outer_train_index = np.flatnonzero(data.keys["cell_id"].isin(sorted(train_cells)).to_numpy())
    test_all = np.flatnonzero(data.keys["cell_id"].isin(sorted(test_cells)).to_numpy())
    outer_seen_drugs = set(data.keys.iloc[outer_train_index]["pert_id"])
    test_index = test_all[
        data.keys.iloc[test_all]["pert_id"].isin(outer_seen_drugs).to_numpy()]
    prep = helpers.fit_preprocess(data, outer_train_index, noise)
    np.savez(args.output_dir / "preprocess.npz", state_mean=prep.state_mean,
             state_scale=prep.state_scale, exposure_mean=prep.exposure_mean,
             exposure_scale=prep.exposure_scale, gene_scale=prep.gene_scale,
             direction_threshold=prep.direction_threshold)
    np.save(args.output_dir / "test_condition_ids.npy",
            data.keys.iloc[test_index]["condition_id"].to_numpy(np.int64))
    final_receipts = []
    for arm, candidate, epochs in (
        ("huber", baseline_candidate, baseline_epochs),
        ("b5_rank", rank_candidate, rank_epochs),
    ):
        for seed in SEEDS:
            model, train_receipt = train_final(
                data, outer_train_index, prep, candidate, seed=seed, epochs=epochs,
                device=device, batch_size=args.batch_size,
                output_gene_scale=torch.from_numpy(prep.gene_scale).to(device),
                noise_threshold=torch.from_numpy(prep.direction_threshold).to(device))
            predictor = predict_delta_inputs_only if use_cached_validation else predict_delta
            prediction = predictor(model, data, test_index, prep, device, args.batch_size)
            prediction_name = f"predicted_delta_{arm}_seed{seed}.npy"
            np.save(args.output_dir / prediction_name, prediction)
            model_name = f"model_{arm}_seed{seed}.pt"
            torch.save(model.state_dict(), args.output_dir / model_name)
            final_receipts.append({"arm": arm, "seed": seed,
                                   "candidate": asdict(candidate),
                                   "epochs": epochs,
                                   "model_sha256": sha256(args.output_dir / model_name),
                                   "prediction_sha256": sha256(args.output_dir / prediction_name),
                                   **train_receipt})
            del model, prediction
            if device.type == "cuda":
                torch.cuda.empty_cache()
    files = ["inner_selection.parquet", "preprocess.npz", "test_condition_ids.npy"]
    files += [f"predicted_delta_{arm}_seed{seed}.npy" for arm in ("huber", "b5_rank") for seed in SEEDS]
    files += [f"model_{arm}_seed{seed}.pt" for arm in ("huber", "b5_rank") for seed in SEEDS]
    summary = {
        "experiment_id": "EXP-039", "fold": args.fold,
        "status": "FOLD_COMPLETE_PENDING_INDEPENDENT_REVIEW",
        "training_objective": "Huber versus Huber plus noise-thresholded bidirectional gene-ranking cross entropy",
        "selection_metric": "equal-cell mean validation Spearman of predicted versus measured GSE74602 reversal rankings",
        "seeds": list(SEEDS), "candidate_count": len(CANDIDATES),
        "candidate_grid": [asdict(item) for item in CANDIDATES],
        "selected": selections, "inner_validation_panels": inner_records,
        "n_train_cells": len(train_cells), "n_test_cells": len(test_cells),
        "n_train_conditions": len(outer_train_index), "n_test_conditions": len(test_index),
        "fixed_test_policy": "outer unseen cells; outer-training-seen drugs",
        "condition_manifest_sha256": sha256(args.condition_dir / "manifest.json"),
        "input_manifest_sha256": sha256(args.input_dir / "manifest.json"),
        "contract_sha256": sha256(args.contract_dir / "verification.json"),
        "signature_manifest_sha256": sha256(args.signature_dir / "manifest.json"),
        "validation_audit_sha256": sha256(args.validation_audit),
        "validation_implementation": "fixed_cache_inputs_only" if use_cached_validation else "legacy_rebuild_and_full_tensors",
        "validation_cache_gate_sha256": cache_gate_sha256,
        "split_sha256": sha256(args.split),
        "run_provenance_sha256": provenance_sha256,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "final_receipts": final_receipts,
        "files_sha256": {name: sha256(args.output_dir / name) for name in files},
    }
    temporary = args.output_dir / "fold_summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output_dir / "fold_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--signature-dir", type=Path, required=True)
    parser.add_argument("--validation-audit", type=Path, required=True)
    parser.add_argument("--validation-cache-gate-receipt", type=Path, default=None,
                        help="enable cached validation only with a passing, current 15-fold gate receipt")
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
