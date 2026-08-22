#!/usr/bin/env python3
"""FAST_PROBE: train an EXP-009 BindingDB soft-target teacher without SDST labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from drug_screen.foundation.soft_target import SoftTargetHead

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEACHER = ROOT / "artifacts/experiments/EXP-009/bindingdb_202608/contract/bindingdb_teacher.tsv"
DEFAULT_REGISTRY = ROOT / "mvp/foundation/xpert/DRUG_REGISTRY.json"
DEFAULT_UNIMOL = Path("/mnt/d/Code/DrugScreenLab/data/external/xpert_source/processed_data/all_drugs_unimol_arr.npy")
DEFAULT_OUTPUT = ROOT / "artifacts/experiments/EXP-009/teacher_probe"
SPLIT_NAMES = ("train", "validation", "test")


def normalize_inchikey(value: str | None) -> str:
    return (value or "").strip().upper()


def deterministic_split(ligands: Iterable[str]) -> dict[str, str]:
    """Assign each normalized ligand to a SHA256 80/10/10 split."""
    splits: dict[str, str] = {}
    for ligand in sorted(set(ligands)):
        bucket = int(hashlib.sha256(ligand.encode("utf-8")).hexdigest()[:16], 16) % 100
        splits[ligand] = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
    return splits


def _parse_paffinity(value: str | None) -> float | None:
    try:
        result = float(value or "")
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def read_teacher_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def select_targets(rows: Iterable[dict[str, str]], minimum_support: int = 50, max_targets: int = 64) -> list[str]:
    support: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        ligand, target = normalize_inchikey(row.get("inchi_key")), (row.get("uniprot_id") or "").strip()
        if ligand and target:
            support[target].add(ligand)
    ranked = sorted(((target, len(ligands)) for target, ligands in support.items() if len(ligands) >= minimum_support), key=lambda item: (-item[1], item[0]))
    return [target for target, _ in ranked[:max_targets]]


def aggregate_labels(rows: Iterable[dict[str, str]], targets: list[str], positive_threshold: float = 6.0) -> dict[str, list[float]]:
    target_index = {target: index for index, target in enumerate(targets)}
    labels: dict[str, list[float]] = {}
    for row in rows:
        ligand = normalize_inchikey(row.get("inchi_key"))
        target = (row.get("uniprot_id") or "").strip()
        affinity = _parse_paffinity(row.get("paffinity"))
        if not ligand or target not in target_index or affinity is None:
            continue
        vector = labels.setdefault(ligand, [0.0] * len(targets))
        if affinity >= positive_threshold:
            vector[target_index[target]] = 1.0
    return labels


def load_mappable_ligands(registry_path: Path, unimol_path: Path) -> tuple[dict[str, int], np.ndarray]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    records = registry["drugs"] if isinstance(registry, dict) else registry
    index: dict[str, int] = {}
    for record in records:
        ligand = normalize_inchikey(record.get("inchi_key"))
        pert_idx = record.get("pert_idx")
        if not record.get("global_inference_eligible") or not ligand or pert_idx is None:
            continue
        index.setdefault(ligand, int(pert_idx))
    return index, np.load(unimol_path, mmap_mode="r")


def select_probe_ligands(
    labels_by_ligand: dict[str, list[float]],
    mappable_ligands: set[str],
    max_ligands: int,
) -> list[str]:
    return [ligand for ligand in sorted(labels_by_ligand) if ligand in mappable_ligands][:max_ligands]


def map_structure_features(
    ligands: list[str],
    structure_index: dict[str, int],
    array: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    features: list[np.ndarray] = []
    kept: list[str] = []
    for ligand in ligands:
        pert_idx = structure_index.get(ligand)
        if pert_idx is None or pert_idx < 0 or pert_idx >= array.shape[0]:
            continue
        pooled = np.asarray(array[pert_idx], dtype=np.float32).mean(axis=0)
        if pooled.shape != (514,):
            raise ValueError(f"UniMol pooled shape must be (514,), got {pooled.shape}")
        features.append(np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0))
        kept.append(ligand)
    return np.asarray(features, dtype=np.float32).reshape((-1, 514)), kept


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    truth = y_true[order].astype(np.float64)
    positives = truth.sum()
    if positives == 0:
        return 0.0
    precision = np.cumsum(truth) / np.arange(1, len(truth) + 1)
    return float((precision * truth).sum() / positives)


def micro_auprc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(labels.ravel(), probabilities.ravel()))
    except ImportError:
        return average_precision(labels.ravel(), probabilities.ravel())


def train_probe(features: np.ndarray, labels: np.ndarray, splits: list[str], target_dim: int, epochs: int, batch_size: int, device: torch.device) -> tuple[SoftTargetHead, dict[str, float]]:
    tensors = {name: torch.as_tensor([index for index, split in enumerate(splits) if split == name], dtype=torch.long) for name in SPLIT_NAMES}
    if not len(tensors["train"]) or not len(tensors["test"]):
        raise ValueError("deterministic split produced no train or test ligand; increase --max-ligands")
    x, y = torch.from_numpy(features), torch.from_numpy(labels)
    model = SoftTargetHead(514, target_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    train_set = TensorDataset(x[tensors["train"]], y[tensors["train"]])
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            logits = model.target_logits(xb.to(device))
            loss = criterion(logits, yb.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        test_index = tensors["test"]
        probabilities = torch.sigmoid(model.target_logits(x[test_index].to(device))).cpu().numpy()
    heldout_labels = y[test_index].numpy()
    baseline = np.full_like(heldout_labels, y[tensors["train"]].numpy().mean(), dtype=np.float32)
    return model, {
        "heldout_micro_auprc": micro_auprc(heldout_labels, probabilities),
        "prevalence_baseline_micro_auprc": micro_auprc(heldout_labels, baseline),
        "test_ligands": float(len(test_index)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-tsv", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--unimol", type=Path, default=DEFAULT_UNIMOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-ligands", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-target-support", type=int, default=50)
    args = parser.parse_args()
    if args.max_ligands <= 0 or args.epochs <= 0:
        raise ValueError("--max-ligands and --epochs must be positive")
    rows = read_teacher_rows(args.teacher_tsv)  # single TSV read
    targets = select_targets(rows, args.min_target_support)
    if not targets:
        raise ValueError("no targets meet the minimum support")
    labels_by_ligand = aggregate_labels(rows, targets)
    structure_index, unimol_array = load_mappable_ligands(args.registry, args.unimol)
    candidate_ligands = select_probe_ligands(labels_by_ligand, set(structure_index), args.max_ligands)
    features, ligands = map_structure_features(candidate_ligands, structure_index, unimol_array)
    labels = np.asarray([labels_by_ligand[ligand] for ligand in ligands], dtype=np.float32)
    split_map = deterministic_split(ligands)
    split_values = [split_map[ligand] for ligand in ligands]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metrics = train_probe(features, labels, split_values, len(targets), args.epochs, args.batch_size, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "bindingdb_teacher_probe.pt"
    torch.save({"state_dict": model.state_dict(), "structure_dim": 514, "target_dim": len(targets), "targets": targets}, checkpoint)
    split_counts = Counter(split_values)
    manifest = {
        "experiment": "EXP-009", "mode": "FAST_PROBE", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_boundary": ["BindingDB teacher TSV", "XPert registry", "official UniMol array"],
        "teacher_tsv": str(args.teacher_tsv), "teacher_sha256": sha256_file(args.teacher_tsv),
        "registry": str(args.registry), "unimol": str(args.unimol), "max_ligands": args.max_ligands,
        "target_dim": len(targets), "targets": targets, "split_rule": "SHA256 uppercase InChIKey bucket 80/10/10",
        "split_ligands": dict(split_counts), "device": str(device), "checkpoint": str(checkpoint),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    metrics.update({"fast_probe": True, "eligible_ligands": len(ligands), "target_dim": len(targets), "epochs": args.epochs, "batch_size": args.batch_size})
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
