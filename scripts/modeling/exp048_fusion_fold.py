"""Train one frozen EXP-048 outer fold and emit label-free outer predictions.

The queue's fold-0 cost profile is the complete contract fold: it performs the
registered inner selection and seed refits, writes the keyed outer predictions,
and can be reused in the five-fold analysis after identity/hash verification.
Completed inner and final-fit units support exact-hash resume after interruption.
Outer response values are never loaded or scored here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
import yaml

from drug_screen.models.exp048_fusion import ARMS, EXP048FusionArm


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/EXP-048.yaml"
SPLIT = ROOT / "data/splits/dual_route_splits_20260922_v1.json"
INPUT_DIR = ROOT / "data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1"
SOURCE_VECTOR = ROOT / "artifacts/experiments/EXP-048/20260926_nine_drug_mcf7_fusion_inputs_v1/source_reference_perturbation.npy"
SOURCE_RECEIPT = ROOT / "artifacts/experiments/EXP-048/20260926_nine_drug_mcf7_fusion_inputs_v1/receipt.json"
BINDING_RECEIPT = ROOT / "artifacts/experiments/EXP-048/20260926_target_feature_binding_v1/receipt.json"
CPU_PREFLIGHT = ROOT / "artifacts/experiments/EXP-048/20260926_five_arm_cpu_preflight_v1/receipt.json"
CHECKPOINT = ROOT / "data/artifacts/EXP-011/source_cv/fold0/M2_fold0.pt"
OFFICIAL_SOURCE = ROOT / "data/external/exp010/pharmaformer/repo/pretrain.py"
MODEL_SOURCE = ROOT / "src/drug_screen/models/exp048_fusion.py"
EXPECTED = {
    "source_receipt": "7f70be0fdf1106038463455dd4d2bf25a01b61b140f2b78e9046c60eb24283e2",
    "binding_receipt": "4877ba04d628258033f42770728bce52c56bcf94a2c0335a20464bb80fae6caf",
    "cpu_preflight": "91d2ea62b9935b72b971404e4de303b65d8c5d9af139235cf211c59a740841b9",
    "input_manifest": "c6f9c3349b0a8598280ba099381b781c68ca5599f6f3b7e749090a0e46460b72",
    "split": "9bfeeb4e0982421cc188a54e2a65f348709043c05529e641c7dd11ffc0635be8",
    "checkpoint": "1dbe98a2f52c4affdcee091cf011f3e52e29b51992c9bb3d57a60b08fa464de6",
    "official_source": "c7c1dc2389e4050df1f007093a6042620be1c0abdeb4792f04d583fe2820f567",
}
SEEDS = (20260922, 20260923, 20260924)
INNER_SELECTION_SEED = 20260922
MAX_EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 16


@dataclass(frozen=True)
class Candidate:
    learning_rate: float
    weight_decay: float
    ranking_weight: float

    @property
    def key(self) -> str:
        return f"lr{self.learning_rate:g}_wd{self.weight_decay:g}_pair{self.ranking_weight:g}"


CANDIDATES = tuple(
    Candidate(lr, wd, ranking)
    for lr in (1e-5, 3e-5, 1e-4)
    for wd in (1e-5, 1e-4)
    for ranking in (0.0, 0.1, 0.3)
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_official(path: Path):
    spec = importlib.util.spec_from_file_location("pharmaformer_official_exp048", path)
    if spec is None or spec.loader is None:
        raise ValueError("official PharmaFormer source could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_sha(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} changed: {actual} != {expected}")


def pairwise_ranking_loss(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    if predicted.ndim != 1 or observed.shape != predicted.shape:
        raise ValueError("patient predictions and labels must be aligned vectors")
    target_difference = observed[:, None] - observed[None, :]
    mask = torch.triu(torch.ones_like(target_difference, dtype=torch.bool), diagonal=1)
    mask &= target_difference != 0
    if not bool(mask.any()):
        return predicted.new_zeros(())
    prediction_difference = predicted[:, None] - predicted[None, :]
    return torch.nn.functional.softplus(
        -target_difference[mask].sign() * prediction_difference[mask]
    ).mean()


class TargetData:
    """Frozen target features with response values exposed for outer-train patients only."""

    def __init__(self, device: torch.device, config: dict[str, Any],
                 outer_train_patients: list[str]):
        self.device = device
        self.patients = [str(value) for value in json.loads((INPUT_DIR / "patient_axis.json").read_text(encoding="utf-8"))]
        if len(self.patients) != 51 or len(set(self.patients)) != 51:
            raise ValueError("frozen target patient axis must contain 51 unique patients")
        self.drug_axis = pd.read_parquet(INPUT_DIR / "drug_axis.parquet", columns=["drug_index", "drug_id"])
        outer_train_patients = sorted(set(str(value) for value in outer_train_patients))
        if not outer_train_patients:
            raise ValueError("frozen outer-train patient list is empty")

        binding = json.loads(BINDING_RECEIPT.read_text(encoding="utf-8"))
        source = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
        target_manifest_path = INPUT_DIR / "manifest.json"
        target_manifest = json.loads(target_manifest_path.read_text(encoding="utf-8"))
        if sha256(target_manifest_path) != EXPECTED["input_manifest"]:
            raise ValueError("frozen EXP-036 target feature manifest changed")
        for filename, expected_sha in target_manifest["files_sha256"].items():
            if sha256(INPUT_DIR / filename) != expected_sha:
                raise ValueError(f"frozen target feature file changed: {filename}")
        mask_path = ROOT / binding["target_key_mask"]["path"]
        if sha256(mask_path) != binding["target_key_mask"]["sha256"]:
            raise ValueError("EXP-048 fixed observed-key mask hash changed")
        key_mask = pd.read_csv(mask_path)
        key_mask["patient_id"] = key_mask["patient_id"].astype(str)
        key_mask["drug_id"] = key_mask["drug_id"].astype(str)
        expected_panel_slots = len(self.patients) * 9
        if (key_mask.duplicated(["patient_id", "drug_id"]).any()
                or len(key_mask) != expected_panel_slots):
            raise ValueError("fixed EXP-048 target mask must contain one unique slot for every 51x9 patient-panel pair")
        fixed_pairs = key_mask.loc[key_mask["target_key_present"].astype(bool), ["patient_id", "drug_id"]]
        expected_observed_keys = len(self.patients) * 7
        if (fixed_pairs.duplicated(["patient_id", "drug_id"]).any()
                or len(fixed_pairs) != expected_observed_keys):
            raise ValueError("fixed EXP-048 target mask must contain seven unique observed keys for each of 51 patients")
        observed_per_patient = fixed_pairs.groupby("patient_id").size()
        if set(observed_per_patient.index) != set(self.patients) or not observed_per_patient.eq(7).all():
            raise ValueError("fixed EXP-048 target mask must retain seven observed drugs per patient")
        config_keys = [str(value) for value in config["drug_panel"]["target_inchikeys"]]
        config_indices = [int(value) for value in config["drug_panel"]["target_indices"]]
        if len(config_keys) != 9 or len(set(config_keys)) != 9 or len(config_indices) != 9:
            raise ValueError("EXP-048 panel must remain the frozen nine-drug panel")
        source_rows = binding["drug_identity_binding"]
        if ([int(row["drug_index"]) for row in source_rows] != config_indices
                or [str(row["target_inchikey"]) for row in source_rows] != config_keys):
            raise ValueError("source OOF drug order differs from the frozen panel binding")
        if not set(fixed_pairs["drug_id"]).issubset(set(config_keys)):
            raise ValueError("observed target keys fall outside the fixed global nine-drug panel")

        # Start from the identity-only shared key mask. Read response values
        # only for outer-train patients; outer-test labels are never loaded in
        # the training/selection process and remain unavailable (NaN) here.
        fixed_pairs = fixed_pairs.sort_values(["patient_id", "drug_id"], kind="stable").reset_index(drop=True)
        training_labels = pd.read_parquet(
            INPUT_DIR / "patient_pairs.parquet", columns=["patient_id", "drug_id", "response"],
            filters=[("patient_id", "in", outer_train_patients)],
        )
        training_labels["patient_id"] = training_labels["patient_id"].astype(str)
        training_labels["drug_id"] = training_labels["drug_id"].astype(str)
        training_labels["response"] = pd.to_numeric(training_labels["response"], errors="raise")
        if (training_labels.duplicated(["patient_id", "drug_id"]).any()
                or set(training_labels["patient_id"]) != set(outer_train_patients)):
            raise ValueError("outer-train response keys do not cover the frozen training patient axis")
        self.pairs = fixed_pairs.merge(training_labels, on=["patient_id", "drug_id"],
                                       how="left", validate="one_to_one")
        visible_train = self.pairs["patient_id"].isin(outer_train_patients).to_numpy()
        if (len(self.pairs) != len(self.patients) * 7
                or not np.isfinite(self.pairs.loc[visible_train, "response"].to_numpy(float)).all()
                or self.pairs.loc[~visible_train, "response"].notna().any()):
            raise ValueError("response loading must expose finite outer-train labels and no outer-test labels")
        self.panel_keys = config_keys
        self.panel_position = {key: position for position, key in enumerate(config_keys)}
        self.shuffle_position = [(position + 1) % len(config_keys) for position in range(len(config_keys))]
        if any(position == destination for position, destination in enumerate(self.shuffle_position)):
            raise ValueError("fixed cyclic shift unexpectedly has a fixed point")
        self.panel_drug_indices = config_indices

        drug_row_by_key = {str(row.drug_id): int(row.drug_index) for row in self.drug_axis.itertuples(index=False)}
        if [drug_row_by_key.get(key) for key in config_keys] != config_indices:
            raise ValueError("target drug rows do not match the frozen index/identity mapping")
        self.drug_feature_row = drug_row_by_key
        patient_position = {patient: index for index, patient in enumerate(self.patients)}
        self.pair_patient = np.asarray([patient_position[x] for x in self.pairs["patient_id"]], dtype=np.int64)
        self.pair_drug = np.asarray([drug_row_by_key[x] for x in self.pairs["drug_id"]], dtype=np.int64)
        self.labels = self.pairs["response"].to_numpy(dtype=np.float32)
        expression_np = np.load(INPUT_DIR / "patient_expression.npy", mmap_mode="r", allow_pickle=False)
        ordinal_np = np.load(INPUT_DIR / "drug_ordinal128.npy", mmap_mode="r", allow_pickle=False)
        source_np = np.load(SOURCE_VECTOR, mmap_mode="r", allow_pickle=False)
        if expression_np.shape != (51, 19091) or not np.isfinite(expression_np).all():
            raise ValueError("frozen PDO expression matrix must be finite 51x19091")
        if ordinal_np.shape != (14, 128) or not np.isfinite(ordinal_np).all():
            raise ValueError("frozen target drug structure matrix must be finite 14x128")
        if source_np.shape != (9, 978) or not np.isfinite(source_np).all():
            raise ValueError("frozen MCF7 source OOF matrix must be finite 9x978")
        if sha256(SOURCE_VECTOR) != source["source_oof"]["vector_sha256"]:
            raise ValueError("source OOF feature matrix hash differs from its receipt")
        self.expression = torch.as_tensor(np.asarray(expression_np), dtype=torch.float32, device=device)
        self.ordinal = torch.as_tensor(np.asarray(ordinal_np), dtype=torch.float32, device=device)
        self.source = torch.as_tensor(np.asarray(source_np), dtype=torch.float32, device=device)
        self.patient_position = patient_position

    def indices(self, patients: list[str]) -> np.ndarray:
        wanted = set(str(value) for value in patients)
        unknown = wanted - set(self.patient_position)
        if unknown:
            raise ValueError(f"split contains unknown patients: {sorted(unknown)[:3]}")
        return np.flatnonzero(self.pairs["patient_id"].isin(wanted).to_numpy())

    def tensors(self, index: np.ndarray, arm: str):
        if not len(index):
            raise ValueError("cannot tensorize an empty patient-drug subset")
        patient_index = torch.as_tensor(self.pair_patient[index], dtype=torch.long, device=self.device)
        drug_index = torch.as_tensor(self.pair_drug[index], dtype=torch.long, device=self.device)
        expression = None if arm == "F-B" else self.expression[patient_index]
        smiles = self.ordinal[drug_index]
        perturbation = None
        if arm in {"F-B", "F-F", "F-S"}:
            target_keys = self.pairs.iloc[index]["drug_id"].astype(str).to_list()
            positions = np.asarray([self.panel_position[key] for key in target_keys], dtype=np.int64)
            if arm == "F-S":
                positions = (positions + 1) % len(self.panel_keys)
            p_index = torch.as_tensor(positions, dtype=torch.long, device=self.device)
            perturbation = self.source[p_index]
        return expression, smiles, perturbation

    def scores(self, index: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
        frame = self.pairs.iloc[index][["patient_id", "drug_id", "response"]].copy()
        if not np.isfinite(frame["response"].to_numpy(dtype=np.float64)).all():
            raise RuntimeError("attempted to score patients whose response labels were not loaded for training-side evaluation")
        frame["predicted"] = np.asarray(prediction, dtype=np.float64)
        rows: list[dict[str, Any]] = []
        for patient, group in frame.groupby("patient_id", sort=True):
            observed = group["response"].to_numpy(np.float64)
            predicted = group["predicted"].to_numpy(np.float64)
            if len(group) < 3:
                rho, reason = float("nan"), "FEWER_THAN_THREE_OBSERVED_DRUGS"
            elif not np.isfinite(observed).all() or not np.isfinite(predicted).all():
                rho, reason = float("nan"), "NONFINITE_SCORE_INPUT"
            elif np.ptp(observed) == 0:
                rho, reason = float("nan"), "CONSTANT_OBSERVED"
            elif np.ptp(predicted) == 0:
                rho, reason = float("nan"), "CONSTANT_PREDICTED"
            else:
                rho = float(spearmanr(observed, predicted).statistic)
                reason = None if np.isfinite(rho) else "UNDEFINED_CORRELATION"
            rows.append({"patient_id": str(patient), "spearman": rho,
                         "n_observed_drugs": int(group["drug_id"].nunique()), "reason": reason})
        return pd.DataFrame(rows)


def standardize_labels(data: TargetData, train_index: np.ndarray) -> tuple[float, float]:
    if len(train_index) < 3:
        raise ValueError("too few visible training labels")
    counts = np.bincount(data.pair_patient[train_index], minlength=len(data.patients))
    weights = 1.0 / counts[data.pair_patient[train_index]]
    values = data.labels[train_index].astype(np.float64)
    mean = float(np.average(values, weights=weights))
    scale = float(np.sqrt(np.average(np.square(values - mean), weights=weights)))
    if not math.isfinite(scale) or scale <= 1e-8:
        raise ValueError("training-side target labels have zero or invalid variation")
    return mean, scale


def build_model(official, checkpoint: dict[str, Any], arm: str, seed: int, device: torch.device) -> EXP048FusionArm:
    config = checkpoint["model_config"]
    backbone = official.CombinedModel(
        gene_input_size=config["gene_input_size"], gene_hidden_size=config["gene_hidden_size"],
        drug_hidden_size=config["drug_hidden_size"], feature_dim=config["feature_dim"],
        nhead=config["nhead"], num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"], dropout=config["dropout"])
    backbone.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return EXP048FusionArm(backbone, arm, seed=seed).to(device)


def autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.no_grad()
def predict(data: TargetData, model: torch.nn.Module, index: np.ndarray,
            arm: str, device: torch.device) -> np.ndarray:
    model.eval()
    expression, smiles, perturbation = data.tensors(index, arm)
    with autocast(device):
        values = model(expression, smiles, perturbation).float().flatten()
    result = values.detach().cpu().numpy().astype(np.float32)
    if result.shape != (len(index),) or not np.isfinite(result).all():
        raise ValueError("model produced invalid prediction vector")
    return result


def train(data: TargetData, official, checkpoint: dict[str, Any], arm: str,
          candidate: Candidate, train_index: np.ndarray, validation_index: np.ndarray | None,
          *, seed: int, device: torch.device, fixed_epochs: int | None = None) -> tuple[torch.nn.Module, dict[str, Any]]:
    start = time.perf_counter()
    mean, scale = standardize_labels(data, train_index)
    target = torch.as_tensor((data.labels[train_index] - mean) / scale,
                             dtype=torch.float32, device=device)
    train_patients = data.pairs.iloc[train_index]["patient_id"].astype(str).to_numpy()
    patient_groups = [np.flatnonzero(train_patients == patient)
                      for patient in sorted(set(train_patients))]
    if any(len(group) > BATCH_SIZE for group in patient_groups):
        raise ValueError("batch size must contain one complete observed patient drug set")
    model = build_model(official, checkpoint, arm, seed, device)
    # The checkpoint is frozen and new modules use their own deterministic
    # per-seed initialization. Seed the training stream as well so Transformer
    # dropout is reproducible for each registered seed.
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError(f"arm {arm} has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=candidate.learning_rate,
                                  weight_decay=candidate.weight_decay)
    best_score, best_epoch, stale = -float("inf"), 0, 0
    epochs = fixed_epochs or MAX_EPOCHS
    epochs_run = 0
    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(patient_groups))
        for group_position in order:
            local = patient_groups[int(group_position)]
            take = train_index[local]
            expression, smiles, perturbation = data.tensors(take, arm)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                output = model(expression, smiles, perturbation).float().flatten()
                loss = torch.nn.functional.smooth_l1_loss(output, target[local])
                if candidate.ranking_weight > 0:
                    loss = loss + candidate.ranking_weight * pairwise_ranking_loss(output, target[local])
            if not bool(torch.isfinite(loss)):
                raise ValueError("training loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        if validation_index is None:
            best_epoch = epoch
            continue
        validation_prediction = predict(data, model, validation_index, arm, device)
        validation_scores = data.scores(validation_index, validation_prediction)
        valid = validation_scores["spearman"].to_numpy(dtype=np.float64)
        valid = valid[np.isfinite(valid)]
        score = float(valid.mean()) if len(valid) else float("nan")
        if math.isfinite(score) and score > best_score + 1e-6:
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if validation_index is not None and best_epoch == 0:
        raise ValueError("inner validation has no finite patient-macro Spearman score")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    receipt = {
        "arm": arm, "candidate": asdict(candidate), "seed": int(seed),
        "best_validation_patient_macro_spearman": best_score if validation_index is not None else None,
        "best_epoch": int(best_epoch), "epochs_run": int(epochs_run),
        "train_patient_count": int(len(patient_groups)), "train_key_count": int(len(train_index)),
        "validation_key_count": int(len(validation_index)) if validation_index is not None else 0,
        "train_only_response_mean": mean, "train_only_response_scale": scale,
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "elapsed_model_seconds": float(time.perf_counter() - start),
    }
    return model, receipt


def input_hashes() -> dict[str, str]:
    paths = {
        "config": CONFIG, "split": SPLIT, "input_manifest": INPUT_DIR / "manifest.json",
        "source_receipt": SOURCE_RECEIPT, "binding_receipt": BINDING_RECEIPT,
        "cpu_preflight": CPU_PREFLIGHT, "source_vector": SOURCE_VECTOR,
        "source_checkpoint": CHECKPOINT, "official_source": OFFICIAL_SOURCE,
        "fusion_model": MODEL_SOURCE, "fold_runner": Path(__file__).resolve(),
    }
    for name, expected in EXPECTED.items():
        path_key = {
            "source_receipt": "source_receipt", "binding_receipt": "binding_receipt",
            "cpu_preflight": "cpu_preflight", "input_manifest": "input_manifest",
            "split": "split", "checkpoint": "source_checkpoint",
            "official_source": "official_source",
        }[name]
        require_sha(paths[path_key], expected, path_key)
    return {name: sha256(path) for name, path in paths.items()}


def fold_contract(fold: int) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config.get("experiment_id") != "EXP-048":
        raise ValueError("wrong experiment config")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    split_fold = split["target_folds"][fold]
    fraction = next(item for item in split_fold["label_fractions"] if float(item["fraction"]) == 0.5)
    if len(fraction["inner_folds"]) != 3:
        raise ValueError("EXP-048 requires exactly three frozen inner patient folds")
    outer_train = set(str(value) for value in fraction["train_ids"])
    outer_test = set(str(value) for value in split_fold["outer_test_ids"])
    if not outer_train or not outer_test or outer_train & outer_test:
        raise ValueError("frozen outer train/test patient sets are empty or overlap")
    for inner in fraction["inner_folds"]:
        inner_train = set(str(value) for value in inner["train_ids"])
        inner_validation = set(str(value) for value in inner["validation_ids"])
        if (not inner_train or not inner_validation or inner_train & inner_validation
                or not inner_train.issubset(outer_train)
                or not inner_validation.issubset(outer_train)
                or inner_train & outer_test or inner_validation & outer_test):
            raise ValueError("inner split is empty, overlapping, or exposes outer-test patients")
    return {"config": config, "split": split, "outer": split_fold, "fraction": fraction}


def _existing_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cpu" or not args.device.startswith("cuda"):
        raise ValueError("EXP-048 formal/cost-profile training must use CUDA; CPU fallback is forbidden")
    if os.environ.get("CONDA_DEFAULT_ENV") != "drugscreening-gpu":
        raise RuntimeError("run this entrypoint only in WSL2 Conda drugscreening-gpu")
    if "microsoft" not in Path("/proc/version").read_text(encoding="utf-8").lower():
        raise RuntimeError("formal EXP-048 runs must execute under WSL2")
    if not os.environ.get("EXP048_APPROVED_QUEUE_ATTEMPT_ID"):
        raise RuntimeError("direct training is disabled; use the budget-gated EXP-048 queue wrapper")
    if not os.environ.get("EXP048_BUDGET_AUTHORIZATION_SHA256"):
        raise RuntimeError("no verified Sol/user budget authorization was bound to this run")
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_gpu.startswith("GPU-"):
        raise RuntimeError("queue must expose one verified A5000 by UUID; refusing an unbound CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
    device = torch.device(args.device)
    if str(device) != "cuda:0":
        raise RuntimeError("the queue must expose exactly one authorized GPU as cuda:0")
    if "A5000" not in torch.cuda.get_device_name(device):
        raise RuntimeError(f"unexpected GPU for EXP-048: {torch.cuda.get_device_name(device)}")
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    hashes = input_hashes()
    contract = fold_contract(args.fold)
    data = TargetData(device, config, list(contract["fraction"]["train_ids"]))
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != "M2" or checkpoint.get("model_variant") != "CombinedModel":
        raise ValueError("the frozen source checkpoint is not the registered M2 CombinedModel")
    official = load_official(OFFICIAL_SOURCE)

    run_dir: Path = args.output_dir.resolve()
    if not run_dir.is_relative_to(ROOT):
        raise ValueError("run output must remain under the DrugScreenLab workspace")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    if run_dir.exists() and any(run_dir.iterdir()) and not manifest_path.is_file():
        raise FileExistsError("nonempty run directory has no provenance manifest; refusing to reuse it")
    expected_manifest = {
        "experiment_id": "EXP-048", "fold": args.fold,
        "target_label_fraction": 0.5, "candidate_grid": [asdict(item) for item in CANDIDATES],
        "arms": list(ARMS), "seeds": list(SEEDS), "input_and_code_sha256": hashes,
        "created_at_utc": utc_now(), "status": "IN_PROGRESS",
    }
    prior_manifest = _existing_json(manifest_path)
    if prior_manifest is None:
        atomic_json(manifest_path, expected_manifest)
    else:
        if (prior_manifest.get("fold") != args.fold
                or prior_manifest.get("input_and_code_sha256") != hashes
                or prior_manifest.get("candidate_grid") != expected_manifest["candidate_grid"]
                or prior_manifest.get("arms") != list(ARMS)
                or prior_manifest.get("seeds") != list(SEEDS)):
            raise ValueError("resume refused: fold, code, protocol, or input hashes differ")
        if not args.resume:
            raise FileExistsError("output exists; pass --resume to reuse verified completed units")
    unit_dir = run_dir / "units"
    unit_dir.mkdir(parents=True, exist_ok=True)
    inner_results: list[dict[str, Any]] = []
    chosen: dict[str, Candidate] = {}
    chosen_epochs: dict[str, int] = {}
    selection_path = run_dir / "inner_selection.json"
    selection = _existing_json(selection_path)
    if selection is None:
        for arm in ARMS:
            by_candidate: list[dict[str, Any]] = []
            for candidate in CANDIDATES:
                fold_rows: list[dict[str, Any]] = []
                for inner in contract["fraction"]["inner_folds"]:
                    inner_id = int(inner["inner_fold"])
                    unit_path = unit_dir / "inner" / arm / candidate.key / f"inner_{inner_id}.json"
                    cached = _existing_json(unit_path)
                    unit_binding = {
                        "arm": arm, "candidate": asdict(candidate), "inner_fold": inner_id,
                        "train_patient_ids": sorted(str(x) for x in inner["train_ids"]),
                        "validation_patient_ids": sorted(str(x) for x in inner["validation_ids"]),
                        "seed": INNER_SELECTION_SEED, "input_and_code_sha256": hashes,
                    }
                    if cached is not None:
                        if cached.get("unit_binding") != unit_binding or cached.get("status") != "UNIT_COMPLETE":
                            raise ValueError(f"cached inner unit mismatch: {unit_path}")
                        result = cached["result"]
                    else:
                        train_index = data.indices(inner["train_ids"])
                        validation_index = data.indices(inner["validation_ids"])
                        model, result = train(
                            data, official, checkpoint, arm, candidate, train_index, validation_index,
                            seed=INNER_SELECTION_SEED, device=device)
                        del model
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        atomic_json(unit_path, {"status": "UNIT_COMPLETE", "unit_binding": unit_binding,
                                                "result": result, "completed_at_utc": utc_now()})
                    fold_rows.append(result)
                    print(json.dumps({"phase": "inner_unit_complete", "arm": arm,
                                      "candidate": candidate.key, "inner_fold": inner_id,
                                      "cached_or_new": "unit_complete"}), flush=True)
                scores = [float(row["best_validation_patient_macro_spearman"]) for row in fold_rows]
                finite_scores = [value for value in scores if math.isfinite(value)]
                if len(finite_scores) != 3:
                    raise ValueError("candidate must have finite scores on all three inner folds")
                mean_score = float(np.mean(finite_scores))
                median_epoch = max(1, int(round(float(np.median([row["best_epoch"] for row in fold_rows])))))
                by_candidate.append({"candidate": asdict(candidate), "mean_inner_score": mean_score,
                                     "median_best_epoch": median_epoch, "inner_results": fold_rows})
            best = sorted(by_candidate, key=lambda row: (-row["mean_inner_score"],
                                                          Candidate(**row["candidate"]).key))[0]
            chosen[arm] = Candidate(**best["candidate"])
            chosen_epochs[arm] = int(best["median_best_epoch"])
            inner_results.append({"arm": arm, "candidates": by_candidate,
                                  "selected_candidate": asdict(chosen[arm]),
                                  "refit_epochs": chosen_epochs[arm]})
        selection = {"experiment_id": "EXP-048", "fold": args.fold,
                     "status": "INNER_SELECTION_COMPLETE", "selection_seed": INNER_SELECTION_SEED,
                     "inner_folds": 3, "candidate_count_per_arm": len(CANDIDATES),
                     "arms": inner_results, "input_and_code_sha256": hashes,
                     "selected_at_utc": utc_now()}
        atomic_json(selection_path, selection)
    else:
        if (selection.get("status") != "INNER_SELECTION_COMPLETE"
                or selection.get("fold") != args.fold
                or selection.get("input_and_code_sha256") != hashes):
            raise ValueError("cached selection does not match current frozen input/code hashes")
        for row in selection["arms"]:
            arm = str(row["arm"])
            chosen[arm] = Candidate(**row["selected_candidate"])
            chosen_epochs[arm] = int(row["refit_epochs"])
        inner_results = selection["arms"]

    outer_train = data.indices(contract["fraction"]["train_ids"])
    outer_test_patients = sorted(str(value) for value in contract["outer"]["outer_test_ids"])
    outer_test = data.indices(outer_test_patients)
    if len(outer_test) == 0 or set(data.pairs.iloc[outer_test]["patient_id"].astype(str)) != set(outer_test_patients):
        raise ValueError("frozen outer test key coverage is incomplete")
    final_rows: list[dict[str, Any]] = []
    pred_frames: list[pd.DataFrame] = []
    for arm in ARMS:
        for seed in SEEDS:
            unit_path = unit_dir / "final" / arm / f"seed_{seed}.json"
            cached = _existing_json(unit_path)
            unit_binding = {"arm": arm, "seed": int(seed), "fold": args.fold,
                            "candidate": asdict(chosen[arm]), "fixed_epochs": chosen_epochs[arm],
                            "train_patient_ids": sorted(str(x) for x in contract["fraction"]["train_ids"]),
                            "input_and_code_sha256": hashes}
            model_path = unit_path.with_suffix(".pt")
            if cached is not None:
                if cached.get("unit_binding") != unit_binding or cached.get("status") != "UNIT_COMPLETE":
                    raise ValueError(f"cached final unit mismatch: {unit_path}")
                if not model_path.is_file() or sha256(model_path) != cached["model_sha256"]:
                    raise ValueError(f"cached final checkpoint missing/changed: {model_path}")
                model = build_model(official, checkpoint, arm, seed, device)
                model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=True)
                result = cached["result"]
            else:
                model, result = train(data, official, checkpoint, arm, chosen[arm], outer_train, None,
                                      seed=seed, device=device, fixed_epochs=chosen_epochs[arm])
                torch.save(model.state_dict(), model_path)
                atomic_json(unit_path, {"status": "UNIT_COMPLETE", "unit_binding": unit_binding,
                                        "model_sha256": sha256(model_path), "result": result,
                                        "completed_at_utc": utc_now()})
            final_rows.append({"arm": arm, "seed": int(seed), "checkpoint": str(model_path.relative_to(ROOT)),
                               "checkpoint_sha256": sha256(model_path), **result})
            if args.phase == "predictions":
                predicted_standardized = predict(data, model, outer_test, arm, device)
                mean = float(result["train_only_response_mean"])
                scale = float(result["train_only_response_scale"])
                predicted = predicted_standardized * scale + mean
                frame = data.pairs.iloc[outer_test][["patient_id", "drug_id"]].copy()
                frame["patient_id"] = frame["patient_id"].astype(str)
                frame["drug_id"] = frame["drug_id"].astype(str)
                frame["prediction"] = predicted.astype(np.float64)
                frame["fold"] = int(args.fold)
                frame["arm"] = arm
                frame["seed"] = int(seed)
                pred_frames.append(frame)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(json.dumps({"phase": "final_fit", "arm": arm, "seed": seed,
                              "checkpoint": model_path.name, "prediction_written": args.phase == "predictions"}),
                  flush=True)

    prediction_path = run_dir / "predictions.parquet"
    prediction_hash: str | None = None
    if args.phase == "predictions":
        prediction_frame = pd.concat(pred_frames, ignore_index=True)
        if prediction_frame.duplicated(["fold", "arm", "seed", "patient_id", "drug_id"]).any():
            raise ValueError("outer prediction keys are not unique")
        expected_rows = len(outer_test) * len(ARMS) * len(SEEDS)
        if len(prediction_frame) != expected_rows or not np.isfinite(prediction_frame["prediction"]).all():
            raise ValueError("outer prediction matrix is incomplete or non-finite")
        tmp = prediction_path.with_name(prediction_path.name + ".tmp")
        prediction_frame.to_parquet(tmp, index=False)
        os.replace(tmp, prediction_path)
        prediction_hash = sha256(prediction_path)
    summary = {
        "schema_version": 1, "experiment_id": "EXP-048", "fold": args.fold,
        "phase": args.phase,
        "status": "COST_PROFILE_COMPLETE_NO_OUTER_PREDICTION" if args.phase == "cost-profile"
                  else "FOLD_PREDICTIONS_COMPLETE_PENDING_INDEPENDENT_VERIFICATION",
        "created_at_utc": utc_now(), "runtime": {
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "device": str(device), "device_name": torch.cuda.get_device_name(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "protocol": {"arms": list(ARMS), "candidate_grid": [asdict(value) for value in CANDIDATES],
                     "inner_folds": 3, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                     "seeds": list(SEEDS), "target_label_fraction": 0.5,
                     "target_outer_fold": args.fold,
                     "source_vector_sha256": hashes["source_vector"],
                     "patient_macro_within_patient_spearman": "inner-only selection; outer labels are never emitted or scored here"},
        "n_inner_candidate_units": len(ARMS) * len(CANDIDATES) * 3,
        "n_final_fit_units": len(ARMS) * len(SEEDS),
        "outer_test_patient_count": len(outer_test_patients),
        "outer_test_key_count": len(outer_test),
        "inner_selection_sha256": sha256(selection_path),
        "final_fit_receipts": final_rows,
        "input_and_code_sha256": hashes,
        "predictions_path": str(prediction_path.relative_to(ROOT)) if prediction_hash else None,
        "predictions_sha256": prediction_hash,
        "outer_observed_labels_or_scores_in_output": False,
        "outer_test_labels_loaded_during_training": False,
        "outer_test_labels_scored": False,
    }
    atomic_json(run_dir / "fold_summary.json", summary)
    manifest = _existing_json(manifest_path) or expected_manifest
    manifest["status"] = summary["status"]
    manifest["last_updated_at_utc"] = utc_now()
    manifest.setdefault("phases_completed", [])
    if args.phase not in manifest["phases_completed"]:
        manifest["phases_completed"].append(args.phase)
    atomic_json(manifest_path, manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("predictions",), required=True,
                        help="the authorized fold run always emits label-free outer predictions; fold 0 also supplies the cost profile")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
