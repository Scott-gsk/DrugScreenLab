"""EXP-036/041 outer target fold: compare four PharmaFormer arms at a frozen label fraction."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from drug_screen.models.pharmaformer_target import configure_target_variant


SEEDS = (20260922, 20260923, 20260924)
ARMS = ("random_full", "source_head", "source_full", "source_adapter")


@dataclass(frozen=True)
class Candidate:
    learning_rate: float
    weight_decay: float
    adapter_rank: int = 0
    ranking_weight: float = 0.0

    @property
    def key(self) -> str:
        return (f"lr{self.learning_rate:g}_wd{self.weight_decay:g}"
                f"_rank{self.adapter_rank}_pair{self.ranking_weight:g}")


STANDARD_CANDIDATES = tuple(Candidate(lr, wd, ranking_weight=pair)
                            for lr in (1e-5, 3e-5, 1e-4)
                            for wd in (1e-5, 1e-4) for pair in (0.0, 0.1, 0.3))
ADAPTER_CANDIDATES = tuple(Candidate(lr, 1e-4, rank, pair)
                           for rank in (8, 16, 32) for lr in (1e-4, 3e-4)
                           for pair in (0.0, 0.1, 0.3))


def pairwise_ranking_loss(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Logistic loss on ordered drug pairs belonging to one training patient."""
    if predicted.ndim != 1 or observed.shape != predicted.shape:
        raise ValueError("patient drug predictions and labels must be aligned vectors")
    target_difference = observed[:, None] - observed[None, :]
    mask = torch.triu(torch.ones_like(target_difference, dtype=torch.bool), diagonal=1)
    mask &= target_difference != 0
    if not bool(mask.any()):
        return predicted.new_zeros(())
    prediction_difference = predicted[:, None] - predicted[None, :]
    return torch.nn.functional.softplus(
        -target_difference[mask].sign() * prediction_difference[mask]).mean()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official(path: Path):
    spec = importlib.util.spec_from_file_location("pharmaformer_official_pretrain", path)
    if spec is None or spec.loader is None:
        raise ValueError("official PharmaFormer module is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Data:
    def __init__(self, input_dir: Path, device: torch.device):
        self.patients = json.loads((input_dir / "patient_axis.json").read_text(encoding="utf-8"))
        self.drugs = pd.read_parquet(input_dir / "drug_axis.parquet")
        self.pairs = pd.read_parquet(input_dir / "patient_pairs.parquet")
        patient_position = {patient: index for index, patient in enumerate(self.patients)}
        drug_position = {drug: index for index, drug in enumerate(self.drugs["drug_id"])}
        self.pair_patient = np.array([patient_position[x] for x in self.pairs["patient_id"]], dtype=np.int64)
        self.pair_drug = np.array([drug_position[x] for x in self.pairs["drug_id"]], dtype=np.int64)
        self.labels = self.pairs["response"].to_numpy(dtype=np.float32)
        if not np.isfinite(self.labels).all() or self.pairs.duplicated(["patient_id", "drug_id"]).any():
            raise ValueError("patient/drug labels invalid")
        self.expression = torch.from_numpy(np.load(input_dir / "patient_expression.npy")).to(device)
        self.ordinal = torch.from_numpy(np.load(input_dir / "drug_ordinal128.npy")).to(device)
        self.patient_index = torch.from_numpy(self.pair_patient).to(device)
        self.drug_index = torch.from_numpy(self.pair_drug).to(device)
        if self.expression.shape != (len(self.patients), 19091) or self.ordinal.shape != (len(self.drugs), 128):
            raise ValueError("PharmaFormer target input shape mismatch")

    def indices(self, patients: list[str], *, seen_drugs: set[str] | None = None) -> np.ndarray:
        mask = self.pairs["patient_id"].isin(patients)
        if seen_drugs is not None:
            mask &= self.pairs["drug_id"].isin(seen_drugs)
        return np.flatnonzero(mask.to_numpy())

    def tensor_pair(self, index: np.ndarray):
        selector = torch.from_numpy(index).to(self.patient_index.device)
        return (self.expression[self.patient_index[selector]],
                self.ordinal[self.drug_index[selector]])

    def patient_scores(self, index: np.ndarray, predicted: np.ndarray) -> tuple[pd.DataFrame, float]:
        frame = self.pairs.iloc[index][["patient_id", "drug_id", "response"]].copy()
        frame["predicted"] = np.asarray(predicted, dtype=np.float64)
        rows = []
        for patient, group in frame.groupby("patient_id", sort=True):
            observed = group["response"].to_numpy(dtype=np.float64)
            values = group["predicted"].to_numpy(dtype=np.float64)
            if len(group) < 3:
                rho, reason = float("nan"), "FEWER_THAN_THREE_DRUGS"
            elif np.ptp(observed) == 0:
                rho, reason = float("nan"), "CONSTANT_OBSERVED"
            elif np.ptp(values) == 0:
                rho, reason = float("nan"), "CONSTANT_PREDICTED"
            else:
                rho = float(spearmanr(observed, values).statistic)
                reason = None if np.isfinite(rho) else "UNDEFINED_CORRELATION"
            rows.append({"patient_id": patient, "spearman": rho,
                         "n_drugs": len(group), "reason": reason})
        score = pd.DataFrame(rows)
        valid = score["spearman"].dropna()
        return score, float(valid.mean()) if len(valid) else float("nan")


def fit_label_standardizer(data: Data, train_index: np.ndarray) -> tuple[float, float, np.ndarray]:
    if len(train_index) < 3:
        raise ValueError("too few target labels")
    counts = np.bincount(data.pair_patient[train_index], minlength=len(data.patients))
    weight = 1.0 / counts[data.pair_patient[train_index]]
    y = data.labels[train_index].astype(np.float64)
    mean = float(np.average(y, weights=weight))
    scale = float(np.sqrt(np.average(np.square(y - mean), weights=weight)))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("target training labels have zero/invalid variation")
    return mean, scale, weight.astype(np.float32)


def build_model(official, checkpoint: dict, arm: str, candidate: Candidate,
                seed: int, device: torch.device) -> torch.nn.Module:
    torch.manual_seed(seed)
    config = checkpoint["model_config"]
    backbone = official.CombinedModel(
        gene_input_size=config["gene_input_size"], gene_hidden_size=config["gene_hidden_size"],
        drug_hidden_size=config["drug_hidden_size"], feature_dim=config["feature_dim"],
        nhead=config["nhead"], num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"], dropout=config["dropout"])
    if arm != "random_full":
        backbone.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = configure_target_variant(backbone, arm, adapter_rank=max(1, candidate.adapter_rank))
    return model.to(device)


@torch.no_grad()
def predict(model: torch.nn.Module, data: Data, index: np.ndarray,
            device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    result = np.empty(len(index), dtype=np.float32)
    for begin in range(0, len(index), batch_size):
        take = index[begin:begin + batch_size]
        expression, drug = data.tensor_pair(take)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            values = model(expression, drug).flatten()
        result[begin:begin + len(take)] = values.float().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("model produced non-finite target predictions")
    return result


def train(data: Data, official, checkpoint: dict, arm: str, candidate: Candidate,
          train_index: np.ndarray, validation_index: np.ndarray | None, *, seed: int,
          device: torch.device, batch_size: int, max_epochs: int, patience: int,
          fixed_epochs: int | None = None) -> tuple[torch.nn.Module, dict]:
    mean, scale, _ = fit_label_standardizer(data, train_index)
    targets = torch.from_numpy((data.labels[train_index] - mean) / scale).to(device)
    patient_groups = [np.flatnonzero(data.pair_patient[train_index] == patient)
                      for patient in np.unique(data.pair_patient[train_index])]
    if any(len(group) > batch_size for group in patient_groups):
        raise ValueError("batch size must accommodate every training patient's full drug panel")
    model = build_model(official, checkpoint, arm, candidate, seed, device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=candidate.learning_rate,
                                  weight_decay=candidate.weight_decay)
    best_score, best_epoch, stale = -float("inf"), 0, 0
    start = time.perf_counter()
    epochs = fixed_epochs if fixed_epochs is not None else max_epochs
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(patient_groups))
        for patient_position in order:
            subset = patient_groups[patient_position]
            indices = train_index[subset]
            expression, drug = data.tensor_pair(indices)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(expression, drug).float().flatten()
                loss = torch.nn.functional.smooth_l1_loss(output, targets[subset])
                if candidate.ranking_weight > 0:
                    loss = loss + candidate.ranking_weight * pairwise_ranking_loss(
                        output, targets[subset])
            if not torch.isfinite(loss):
                raise ValueError("training loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        if validation_index is None:
            best_epoch = epoch
            continue
        validation_prediction = predict(model, data, validation_index, device, batch_size)
        _, metric = data.patient_scores(validation_index, validation_prediction)
        if np.isfinite(metric) and metric > best_score + 1e-6:
            best_score, best_epoch, stale = metric, epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    if validation_index is not None and best_epoch == 0:
        raise ValueError("no valid inner patient-ranking validation score")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return model, {"best_validation_spearman": best_score if validation_index is not None else None,
                   "best_epoch": best_epoch, "epochs_run": epoch,
                   "target_response_mean_train_only": mean,
                   "target_response_scale_train_only": scale,
                   "n_trainable_parameters": sum(parameter.numel() for parameter in parameters),
                   "patient_equal_training_steps_per_epoch": len(patient_groups),
                   "elapsed_seconds": time.perf_counter() - start}


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"fold output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    torch.set_float32_matmul_precision("high")
    data = Data(args.input_dir, device)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    fold = split["target_folds"][args.fold]
    fraction = next(entry for entry in fold["label_fractions"]
                    if float(entry["fraction"]) == args.fraction)
    checkpoint = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["variant"] != "M2" or checkpoint["model_variant"] != "CombinedModel":
        raise ValueError("source checkpoint is incompatible with target training")
    official = load_official(args.official_source)
    inner_folds = fraction["inner_folds"][:1] if args.smoke else fraction["inner_folds"]
    arm_names = ARMS
    selection_rows, chosen, epochs = [], {}, {}
    for arm in arm_names:
        candidates = ADAPTER_CANDIDATES if arm == "source_adapter" else STANDARD_CANDIDATES
        if args.smoke:
            candidates = tuple(candidate for candidate in candidates
                               if candidate.ranking_weight == 0.1)[:1]
        by_candidate = {}
        for candidate in candidates:
            fold_scores, fold_epochs = [], []
            for inner in inner_folds:
                train_index = data.indices(inner["train_ids"])
                seen = set(data.pairs.iloc[train_index]["drug_id"])
                validation_index = data.indices(inner["validation_ids"], seen_drugs=seen)
                model, receipt = train(
                    data, official, checkpoint, arm, candidate, train_index, validation_index,
                    seed=SEEDS[0], device=device, batch_size=args.batch_size,
                    max_epochs=args.max_epochs, patience=args.patience)
                fold_scores.append(receipt["best_validation_spearman"])
                fold_epochs.append(receipt["best_epoch"])
                selection_rows.append({"arm": arm, "candidate": candidate.key,
                                       "inner_fold": inner["inner_fold"], "n_train": len(train_index),
                                       "n_validation": len(validation_index), "fraction": args.fraction,
                                       **asdict(candidate), **receipt})
                print(json.dumps({"arm": arm, "candidate": candidate.key,
                                  "inner_fold": inner["inner_fold"], "validation_spearman": fold_scores[-1],
                                  "best_epoch": fold_epochs[-1]}), flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            by_candidate[candidate.key] = (float(np.mean(fold_scores)),
                                           max(1, int(round(np.median(fold_epochs)))), candidate)
        best_key = min(by_candidate, key=lambda key: (-by_candidate[key][0], key))
        chosen[arm] = by_candidate[best_key][2]
        epochs[arm] = by_candidate[best_key][1]
    pd.DataFrame(selection_rows).to_parquet(args.output_dir / "inner_selection.parquet", index=False)
    train_index = data.indices(fraction["train_ids"])
    seen = set(data.pairs.iloc[train_index]["drug_id"])
    test_index = data.indices(fold["outer_test_ids"], seen_drugs=seen)
    if len(test_index) == 0:
        raise ValueError("outer fold has no seen-drug test pairs")
    predictions, scores, final_receipts = [], [], []
    for arm in arm_names:
        for seed in (SEEDS[:1] if args.smoke else SEEDS):
            model, receipt = train(data, official, checkpoint, arm, chosen[arm], train_index, None,
                                   seed=seed, device=device, batch_size=args.batch_size,
                                   max_epochs=args.max_epochs, patience=args.patience,
                                   fixed_epochs=epochs[arm])
            standardized = predict(model, data, test_index, device, args.batch_size)
            predicted = standardized * receipt["target_response_scale_train_only"] \
                        + receipt["target_response_mean_train_only"]
            frame = data.pairs.iloc[test_index][["patient_id", "drug_id", "response"]].copy()
            frame = frame.rename(columns={"response": "observed"})
            frame["predicted"] = predicted
            frame.insert(0, "fold", args.fold)
            frame.insert(1, "fraction", args.fraction)
            frame.insert(2, "arm", arm)
            frame.insert(3, "seed", seed)
            predictions.append(frame)
            unit, score = data.patient_scores(test_index, predicted)
            unit.insert(0, "fold", args.fold)
            unit.insert(1, "fraction", args.fraction)
            unit.insert(2, "arm", arm)
            unit.insert(3, "seed", seed)
            scores.append(unit)
            model_path = args.output_dir / f"model_{arm}_seed{seed}.pt"
            if not args.smoke:
                torch.save(model.state_dict(), model_path)
            final_receipts.append({"arm": arm, "seed": seed, "test_mean_spearman": score,
                                   "selected_candidate": asdict(chosen[arm]),
                                   "refit_epochs": epochs[arm],
                                   "checkpoint_sha256": sha256(model_path) if not args.smoke else None,
                                   **receipt})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    prediction_file = args.output_dir / "predictions.parquet"
    scores_file = args.output_dir / "patient_scores.parquet"
    pd.concat(predictions, ignore_index=True).to_parquet(prediction_file, index=False)
    pd.concat(scores, ignore_index=True).to_parquet(scores_file, index=False)
    summary = {"experiment_id": "EXP-036", "fold": args.fold,
               "status": "SMOKE_ONLY" if args.smoke else "FOLD_COMPLETE_PENDING_INDEPENDENT_REVIEW",
               "fraction": args.fraction, "arms": list(arm_names), "seeds": list(SEEDS[:1] if args.smoke else SEEDS),
               "training_objective": "patient-equal Huber plus within-patient pairwise logistic ranking",
               "ranking_weight_candidates": [0.0, 0.1, 0.3],
               "candidate_count_per_arm": {arm: len(ADAPTER_CANDIDATES if arm == "source_adapter"
                                                else STANDARD_CANDIDATES) for arm in arm_names},
               "chosen_candidates": {arm: asdict(value) for arm, value in chosen.items()},
               "refit_epochs": epochs, "final_receipts": final_receipts,
               "n_train_patients": len(fraction["train_ids"]),
               "n_test_patients": len(fold["outer_test_ids"]),
               "n_test_patient_drug_pairs": len(test_index),
               "source_checkpoint_sha256": sha256(args.source_checkpoint),
               "official_source_sha256": sha256(args.official_source),
               "input_manifest_sha256": sha256(args.input_dir / "manifest.json"),
               "split_sha256": sha256(args.split),
               "files_sha256": {"predictions.parquet": sha256(prediction_file),
                                "patient_scores.parquet": sha256(scores_file),
                                "inner_selection.parquet": sha256(args.output_dir / "inner_selection.parquet")}}
    temporary = args.output_dir / "fold_summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output_dir / "fold_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--fraction", type=float, choices=(0.25, 0.5, 1.0), default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
