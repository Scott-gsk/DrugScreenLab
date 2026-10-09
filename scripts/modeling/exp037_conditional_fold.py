"""EXP-037 单个外折：内三折选择后训练条件扰动网络与等架构去状态消融。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from drug_screen.evaluation.perturbation import condition_normalized_mse, equal_weight_cell_score
from drug_screen.models.conditional_perturbation import SharedEffectInteraction


SEEDS = (20260922, 20260923, 20260924)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class Candidate:
    rank: int
    weight_decay: float
    direction_weight: float

    @property
    def key(self) -> str:
        return f"r{self.rank}_wd{self.weight_decay:g}_dir{self.direction_weight:g}"


CANDIDATES = tuple(Candidate(rank, decay, direction)
                   for rank in (16, 32) for decay in (1e-4, 1e-3)
                   for direction in (0.0, 0.1, 0.3))


@dataclass
class Preprocess:
    state_mean: np.ndarray
    state_scale: np.ndarray
    exposure_mean: np.ndarray
    exposure_scale: np.ndarray
    gene_scale: np.ndarray
    direction_threshold: np.ndarray


@dataclass
class PreparedArrays:
    state: np.ndarray
    exposure: np.ndarray
    target: np.ndarray


def safe_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    if (~np.isfinite(mean)).any() or (~np.isfinite(scale)).any():
        raise ValueError("non-finite train-side preprocessing statistics")
    scale[scale <= 1e-8] = 1.0
    return mean, scale


def batch_indices(index: np.ndarray, batch_size: int, rng: np.random.Generator | None = None):
    order = index.copy()
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


class Arrays:
    def __init__(self, condition_dir: Path, input_dir: Path, *, load_to_memory: bool = False):
        self.keys = pd.read_parquet(condition_dir / "conditions.parquet")
        self.delta = np.load(condition_dir / "condition_delta.npy", mmap_mode="r")
        self.state = np.load(input_dir / "condition_control.npy", mmap_mode="r")
        drugs = pd.read_parquet(input_dir / "drugs.parquet")
        self.fingerprint = np.load(input_dir / "drug_morgan2048.npy", mmap_mode="r")
        if load_to_memory:
            self.delta = np.array(self.delta, dtype=np.float32, copy=True)
            self.state = np.array(self.state, dtype=np.float32, copy=True)
            self.fingerprint = np.array(self.fingerprint, dtype=np.uint8, copy=True)
        position = dict(zip(drugs["pert_id"], drugs["drug_index"]))
        self.drug_index = np.array([position[x] for x in self.keys["pert_id"]], dtype=np.int32)
        exposure = pd.read_parquet(input_dir / "condition_exposure.parquet")
        if not exposure["condition_id"].equals(self.keys["condition_id"]):
            raise ValueError("condition exposure axis mismatch")
        self.exposure = exposure[["log10_dose_um", "time_hours"]].to_numpy(np.float32)
        if self.delta.shape != self.state.shape or self.delta.shape[0] != len(self.keys):
            raise ValueError("condition matrix axes disagree")

    def prepare(self, prep: Preprocess) -> PreparedArrays:
        return PreparedArrays(
            state=((self.state - prep.state_mean) / prep.state_scale).astype(np.float32),
            exposure=((self.exposure - prep.exposure_mean) / prep.exposure_scale).astype(np.float32),
            target=(self.delta / prep.gene_scale).astype(np.float32),
        )

    def tensors(self, index: np.ndarray, prep: Preprocess, device: torch.device,
                prepared: PreparedArrays | None = None):
        state = (np.asarray(self.state[index], dtype=np.float32) - prep.state_mean) / prep.state_scale if prepared is None else prepared.state[index]
        drug = np.asarray(self.fingerprint[self.drug_index[index]], dtype=np.float32)
        exposure = ((self.exposure[index] - prep.exposure_mean) / prep.exposure_scale
                    if prepared is None else prepared.exposure[index])
        target_raw = np.asarray(self.delta[index], dtype=np.float32)
        target = target_raw / prep.gene_scale if prepared is None else prepared.target[index]
        return tuple(torch.from_numpy(x).to(device, non_blocking=True)
                     for x in (state, drug, exposure, target, target_raw))


def fit_preprocess(arrays: Arrays, train_index: np.ndarray,
                   control_noise: pd.DataFrame) -> Preprocess:
    train_state = np.asarray(arrays.state[train_index], dtype=np.float32)
    state_mean, state_scale = safe_scale(train_state)
    exposure_mean, exposure_scale = safe_scale(arrays.exposure[train_index])
    train_delta = np.asarray(arrays.delta[train_index], dtype=np.float32)
    _, gene_scale = safe_scale(train_delta)
    train_cells = set(arrays.keys.iloc[train_index]["cell_id"])
    noise = np.stack(control_noise.loc[control_noise["cell_id"].isin(train_cells), "std"])
    threshold = np.median(noise, axis=0).astype(np.float32)
    if threshold.shape != (978,) or not np.isfinite(threshold).all():
        raise ValueError("invalid train-side control-noise threshold")
    return Preprocess(state_mean, state_scale, exposure_mean, exposure_scale,
                      gene_scale, threshold)


def control_noise_table(contract_dir: Path, cache_path: Path) -> pd.DataFrame:
    controls = pd.read_parquet(contract_dir / "controls.parquet")
    cache = np.load(cache_path, mmap_mode="r")
    rows = []
    for group, frame in controls.groupby("control_group_id", sort=False):
        values = np.asarray(cache[frame["cache_row"].to_numpy(dtype=np.int64)], dtype=np.float32)
        rows.append({"control_group_id": int(group), "cell_id": frame["cell_id"].iloc[0],
                     "std": values.std(axis=0, dtype=np.float64).astype(np.float32)})
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate(model: SharedEffectInteraction, arrays: Arrays, index: np.ndarray,
             prep: Preprocess, device: torch.device, use_state: bool,
             batch_size: int, prepared: PreparedArrays | None = None) -> tuple[float, pd.DataFrame, np.ndarray]:
    model.eval()
    prediction = np.empty((len(index), 978), dtype=np.float32)
    for start in range(0, len(index), batch_size):
        positions = index[start:start + batch_size]
        state, drug, exposure, _, _ = arrays.tensors(positions, prep, device, prepared)
        standardized = model(state, drug, exposure, use_state=use_state)
        prediction[start:start + len(positions)] = (
            standardized.cpu().numpy() * prep.gene_scale)
    observed = np.asarray(arrays.delta[index], dtype=np.float32)
    scores = condition_normalized_mse(observed, prediction, prep.gene_scale)
    keys = arrays.keys.iloc[index].reset_index(drop=True)
    cell, total = equal_weight_cell_score(keys, scores)
    return total, cell, prediction


def train_model(arrays: Arrays, train_index: np.ndarray, validation_index: np.ndarray | None,
                prep: Preprocess, candidate: Candidate, *, use_state: bool, seed: int,
                device: torch.device, batch_size: int, max_epochs: int,
                patience: int, fixed_epochs: int | None = None,
                prepared: PreparedArrays | None = None) -> tuple[SharedEffectInteraction, dict]:
    torch.manual_seed(seed); np.random.seed(seed)
    model = SharedEffectInteraction(rank=candidate.rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=candidate.weight_decay)
    threshold = torch.from_numpy(prep.direction_threshold).to(device)
    best_state, best_metric, best_epoch = None, float("inf"), 0
    stale = 0; start = time.perf_counter()
    epochs = fixed_epochs if fixed_epochs is not None else max_epochs
    for epoch in range(1, epochs + 1):
        model.train(); rng = np.random.default_rng(seed + epoch)
        for positions in batch_indices(train_index, batch_size, rng):
            state, drug, exposure, target, target_raw = arrays.tensors(
                positions, prep, device, prepared)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(state, drug, exposure, use_state=use_state)
            loss = torch.nn.functional.huber_loss(predicted, target)
            if candidate.direction_weight > 0:
                mask = target_raw.abs() > threshold
                if mask.any():
                    direction = torch.sign(target_raw[mask])
                    auxiliary = torch.nn.functional.softplus(-direction * predicted[mask]).mean()
                    loss = loss + candidate.direction_weight * auxiliary
            loss.backward(); optimizer.step()
        if validation_index is None:
            best_epoch = epoch
            continue
        metric, _, _ = evaluate(model, arrays, validation_index, prep, device,
                                use_state, batch_size, prepared)
        if metric < best_metric - 1e-6:
            best_metric, best_epoch, stale = metric, epoch, 0
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.cuda.synchronize(device)
    return model, {"best_validation_nmse": best_metric if validation_index is not None else None,
                   "best_epoch": best_epoch, "elapsed_seconds": time.perf_counter() - start}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--load-to-memory", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"fold output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    arrays = Arrays(args.condition_dir, args.input_dir, load_to_memory=args.load_to_memory)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    fold = split["folds"][args.fold]
    inner_folds = fold["label_fractions"][0]["inner_folds"]
    noise = control_noise_table(args.contract_dir, args.cache)
    selection_rows = []
    chosen = {}
    epoch_choice = {}
    for arm, use_state in (("state_ablated", False), ("conditional", True)):
        candidate_metrics = {candidate.key: [] for candidate in CANDIDATES}
        candidate_epochs = {candidate.key: [] for candidate in CANDIDATES}
        for inner in inner_folds:
            train_index = np.flatnonzero(arrays.keys["cell_id"].isin(inner["train_ids"]).to_numpy())
            validation_all = np.flatnonzero(arrays.keys["cell_id"].isin(inner["validation_ids"]).to_numpy())
            seen = set(arrays.keys.iloc[train_index]["pert_id"])
            validation_index = validation_all[
                arrays.keys.iloc[validation_all]["pert_id"].isin(seen).to_numpy()]
            prep = fit_preprocess(arrays, train_index, noise)
            prepared = arrays.prepare(prep)
            for candidate in CANDIDATES:
                model, receipt = train_model(
                    arrays, train_index, validation_index, prep, candidate,
                    use_state=use_state, seed=20260922, device=device,
                    batch_size=args.batch_size, max_epochs=args.max_epochs,
                    patience=args.patience, prepared=prepared)
                candidate_metrics[candidate.key].append(receipt["best_validation_nmse"])
                candidate_epochs[candidate.key].append(receipt["best_epoch"])
                selection_rows.append({"arm": arm, "inner_fold": inner["inner_fold"],
                                       **asdict(candidate), **receipt,
                                       "n_train": len(train_index), "n_validation": len(validation_index)})
                print(json.dumps({"arm": arm, "inner_fold": inner["inner_fold"],
                                  "candidate": candidate.key,
                                  "validation_nmse": receipt["best_validation_nmse"],
                                  "best_epoch": receipt["best_epoch"]}), flush=True)
                del model; torch.cuda.empty_cache()
            del prepared
        mean_metric = {key: float(np.mean(value)) for key, value in candidate_metrics.items()}
        best_key = min(mean_metric, key=lambda key: (mean_metric[key], key))
        chosen[arm] = next(candidate for candidate in CANDIDATES if candidate.key == best_key)
        epoch_choice[arm] = max(1, int(round(np.median(candidate_epochs[best_key]))))
    pd.DataFrame(selection_rows).to_parquet(args.output_dir / "inner_selection.parquet", index=False)
    train_index = np.flatnonzero(arrays.keys["cell_id"].isin(fold["outer_train_ids"]).to_numpy())
    test_all = np.flatnonzero(arrays.keys["cell_id"].isin(fold["outer_test_ids"]).to_numpy())
    seen = set(arrays.keys.iloc[train_index]["pert_id"])
    test_index = test_all[arrays.keys.iloc[test_all]["pert_id"].isin(seen).to_numpy()]
    prep = fit_preprocess(arrays, train_index, noise)
    prepared = arrays.prepare(prep)
    np.savez(args.output_dir / "preprocess.npz", state_mean=prep.state_mean,
             state_scale=prep.state_scale, exposure_mean=prep.exposure_mean,
             exposure_scale=prep.exposure_scale, gene_scale=prep.gene_scale,
             direction_threshold=prep.direction_threshold)
    np.save(args.output_dir / "test_condition_ids.npy",
            arrays.keys.iloc[test_index]["condition_id"].to_numpy(np.int64))
    cell_rows, final_receipts = [], []
    for arm, use_state in (("state_ablated", False), ("conditional", True)):
        for seed in SEEDS:
            model, receipt = train_model(
                arrays, train_index, None, prep, chosen[arm], use_state=use_state,
                seed=seed, device=device, batch_size=args.batch_size,
                max_epochs=args.max_epochs, patience=args.patience,
                fixed_epochs=epoch_choice[arm], prepared=prepared)
            metric, cell, prediction = evaluate(model, arrays, test_index, prep, device,
                                                use_state, args.batch_size, prepared)
            np.save(args.output_dir / f"predicted_delta_{arm}_seed{seed}.npy", prediction)
            torch.save(model.state_dict(), args.output_dir / f"model_{arm}_seed{seed}.pt")
            cell.insert(0, "fold", args.fold); cell.insert(1, "arm", arm); cell.insert(2, "seed", seed)
            cell_rows.append(cell)
            final_receipts.append({"arm": arm, "seed": seed, "test_nmse": metric,
                                   "epochs": epoch_choice[arm], **receipt})
            del model, prediction; torch.cuda.empty_cache()
    cell_scores = pd.concat(cell_rows, ignore_index=True)
    cell_scores.to_parquet(args.output_dir / "cell_scores.parquet", index=False)
    report = {"experiment_id": "EXP-037", "fold": args.fold,
              "status": "FOLD_COMPLETE_PENDING_CROSS_FOLD_REVIEW",
              "chosen_candidates": {arm: asdict(candidate) for arm, candidate in chosen.items()},
              "refit_epochs": epoch_choice, "final_receipts": final_receipts,
              "n_train_conditions": len(train_index), "n_test_conditions": len(test_index),
              "test_policy": "outer unseen cells; drugs seen in outer training cells",
              "seeds": list(SEEDS), "candidate_count_per_arm": len(CANDIDATES),
              "split_sha256": sha256(args.split),
              "data_manifest_sha256": sha256(args.condition_dir / "manifest.json")}
    temporary = args.output_dir / "fold_summary.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output_dir / "fold_summary.json")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
