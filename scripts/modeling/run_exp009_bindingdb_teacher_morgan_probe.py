#!/usr/bin/env python3
"""FAST_PROBE: train an external BindingDB teacher using RDKit Morgan fingerprints."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEACHER = ROOT / "artifacts/experiments/EXP-009/bindingdb_202608/contract/bindingdb_teacher.tsv"
DEFAULT_OUTPUT = ROOT / "artifacts/experiments/EXP-009/teacher_morgan_probe_100k"
SPLIT_NAMES = ("train", "validation", "test")


def normalize_inchikey(value: str | None) -> str:
    return (value or "").strip().upper()


def deterministic_ligand_split(ligand: str) -> str:
    normalized = normalize_inchikey(ligand)
    bucket = int(hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def deterministic_split(ligands: Iterable[str]) -> dict[str, str]:
    return {ligand: deterministic_ligand_split(ligand) for ligand in set(ligands)}


def _parse_paffinity(value: str | None) -> float | None:
    try:
        result = float(value or "")
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _smiles_text(value: str | None) -> str:
    # BindingDB rows may carry a trailing atom-mapping annotation such as " |r|".
    return (value or "").split(" |", 1)[0].strip()


def canonical_morgan_fingerprint(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(_smiles_text(smiles))
    if mol is None:
        raise ValueError("SMILES cannot be parsed")
    bits = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return np.asarray(bits, dtype=np.float32)


def select_targets(rows: Iterable[dict[str, str]], minimum_support: int = 50, max_targets: int = 64) -> list[str]:
    support: dict[str, set[str]] = {}
    for row in rows:
        ligand = normalize_inchikey(row.get("inchi_key"))
        target = (row.get("uniprot_id") or "").strip()
        if ligand and target:
            support.setdefault(target, set()).add(ligand)
    ranked = sorted(
        ((target, len(ligands)) for target, ligands in support.items() if len(ligands) >= minimum_support),
        key=lambda item: (-item[1], item[0]),
    )
    return [target for target, _ in ranked[:max_targets]]


def aggregate_teacher_records(rows: Iterable[dict[str, str]], targets: list[str]) -> tuple[dict[str, dict[str, object]], int]:
    target_index = {target: index for index, target in enumerate(targets)}
    records: dict[str, dict[str, object]] = {}
    failures = 0
    for row in rows:
        ligand = normalize_inchikey(row.get("inchi_key"))
        target = (row.get("uniprot_id") or "").strip()
        if not ligand or target not in target_index:
            continue
        smiles = _smiles_text(row.get("canonical_smiles"))
        if ligand not in records:
            try:
                canonical_morgan_fingerprint(smiles)
            except ValueError:
                failures += 1
                continue
            records[ligand] = {"smiles": smiles, "labels": [0.0] * len(targets)}
        affinity = _parse_paffinity(row.get("paffinity"))
        if affinity is not None and affinity >= 6.0:
            records[ligand]["labels"][target_index[target]] = 1.0  # type: ignore[index]
    return records, failures


def _iter_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def select_targets_stream(path: Path, minimum_support: int, max_targets: int) -> list[str]:
    with tempfile.NamedTemporaryFile(suffix=".sqlite3") as temporary:
        connection = sqlite3.connect(temporary.name)
        connection.execute("CREATE TABLE support (target TEXT NOT NULL, ligand TEXT NOT NULL, PRIMARY KEY(target, ligand))")
        for row in _iter_rows(path):
            ligand = normalize_inchikey(row.get("inchi_key"))
            target = (row.get("uniprot_id") or "").strip()
            if ligand and target:
                connection.execute("INSERT OR IGNORE INTO support VALUES (?, ?)", (target, ligand))
        connection.commit()
        ranked = connection.execute(
            "SELECT target, COUNT(*) AS support FROM support GROUP BY target HAVING support >= ? ORDER BY support DESC, target LIMIT ?",
            (minimum_support, max_targets),
        ).fetchall()
        connection.close()
    return [target for target, _ in ranked]


def load_teacher_records(path: Path, targets: list[str], max_ligands: int) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
    target_index = {target: index for index, target in enumerate(targets)}
    failures = 0
    candidates = 0
    with tempfile.NamedTemporaryFile(suffix=".sqlite3") as temporary:
        connection = sqlite3.connect(temporary.name)
        connection.execute("CREATE TABLE records (ligand TEXT PRIMARY KEY, smiles TEXT NOT NULL, labels BLOB NOT NULL, hash TEXT NOT NULL, split TEXT NOT NULL)")
        for row in _iter_rows(path):
            ligand = normalize_inchikey(row.get("inchi_key"))
            target = (row.get("uniprot_id") or "").strip()
            if not ligand or target not in target_index:
                continue
            smiles = _smiles_text(row.get("canonical_smiles"))
            affinity = _parse_paffinity(row.get("paffinity"))
            existing = connection.execute("SELECT labels FROM records WHERE ligand = ?", (ligand,)).fetchone()
            if existing is None:
                try:
                    canonical_morgan_fingerprint(smiles)
                except ValueError:
                    failures += 1
                    continue
                labels = [0.0] * len(targets)
                connection.execute(
                    "INSERT INTO records VALUES (?, ?, ?, ?, ?)",
                    (ligand, smiles, json.dumps(labels), hashlib.sha256(ligand.encode()).hexdigest(), deterministic_ligand_split(ligand)),
                )
                candidates += 1
            else:
                labels = json.loads(existing[0])
            if affinity is not None and affinity >= 6.0:
                labels[target_index[target]] = 1.0
                connection.execute("UPDATE records SET labels = ? WHERE ligand = ?", (json.dumps(labels), ligand))
        connection.commit()
        selected = connection.execute("SELECT ligand, smiles, labels FROM records ORDER BY hash LIMIT ?", (max_ligands,)).fetchall()
        candidate_per_split = dict(connection.execute("SELECT split, COUNT(*) FROM records GROUP BY split").fetchall())
        valid_per_split = dict(Counter(deterministic_ligand_split(ligand) for ligand, _, _ in selected))
        connection.close()
    records = {ligand: {"smiles": smiles, "labels": json.loads(labels)} for ligand, smiles, labels in selected}
    return records, {
        "candidate_ligands": candidates,
        "smiles_failures": failures,
        "candidate_per_split": candidate_per_split,
        "valid_per_split": valid_per_split,
    }


def micro_auprc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(labels.ravel(), probabilities.ravel()))
    except ImportError:
        order = np.argsort(-probabilities.ravel(), kind="mergesort")
        truth = labels.ravel()[order].astype(np.float64)
        positives = truth.sum()
        if positives == 0:
            return 0.0
        precision = np.cumsum(truth) / np.arange(1, len(truth) + 1)
        return float((precision * truth).sum() / positives)


def train_probe(features: np.ndarray, labels: np.ndarray, splits: list[str], target_dim: int, epochs: int, batch_size: int, device: torch.device):
    x, y = torch.from_numpy(features), torch.from_numpy(labels)
    indices = {name: torch.tensor([i for i, split in enumerate(splits) if split == name], dtype=torch.long) for name in SPLIT_NAMES}
    if not len(indices["train"]) or not len(indices["test"]):
        raise ValueError("deterministic split produced no train or test ligand")
    model = nn.Linear(2048, target_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(x[indices["train"]], y[indices["train"]]), batch_size=batch_size, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = criterion(model(xb.to(device)), yb.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        test_probabilities = torch.sigmoid(model(x[indices["test"]].to(device))).cpu().numpy()
    test_labels = y[indices["test"]].numpy()
    train_prevalence = float(y[indices["train"]].numpy().mean())
    baseline = np.full_like(test_labels, train_prevalence)
    return model, {
        "heldout_micro_auprc": micro_auprc(test_labels, test_probabilities),
        "prevalence_baseline_micro_auprc": micro_auprc(test_labels, baseline),
        "heldout_prevalence": float(test_labels.mean()),
        "test_ligands": len(indices["test"]),
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-ligands", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--min-target-support", type=int, default=50)
    args = parser.parse_args()
    if args.max_ligands <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--max-ligands, --epochs and --batch-size must be positive")
    targets = select_targets_stream(args.teacher_tsv, args.min_target_support, 64)
    if not targets:
        raise ValueError("no targets meet the minimum support")
    records, scan_stats = load_teacher_records(args.teacher_tsv, targets, args.max_ligands)
    if not records:
        raise ValueError("no valid SMILES ligands found")
    ligands = list(records)
    features = np.stack([canonical_morgan_fingerprint(records[ligand]["smiles"]) for ligand in ligands]).astype(np.float32)
    labels = np.asarray([records[ligand]["labels"] for ligand in ligands], dtype=np.float32)
    split_values = [deterministic_ligand_split(ligand) for ligand in ligands]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metrics = train_probe(features, labels, split_values, len(targets), args.epochs, args.batch_size, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "bindingdb_teacher_morgan_probe.pt"
    torch.save({"state_dict": model.state_dict(), "input_dim": 2048, "target_dim": len(targets), "targets": targets}, checkpoint)
    split_counts = Counter(split_values)
    manifest = {
        "experiment": "EXP-009", "mode": "FAST_PROBE", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_boundary": ["BindingDB teacher TSV", "RDKit Morgan fingerprints"],
        "teacher_tsv": str(args.teacher_tsv), "input_sha256": sha256_file(args.teacher_tsv),
        "max_ligands": args.max_ligands, "target_dim": len(targets), "targets": targets,
        "split_rule": "SHA256 uppercase InChIKey bucket 80/10/10", "ligands_selected": len(ligands),
        "candidate_ligands": scan_stats["candidate_ligands"], "valid_ligands": len(ligands),
        "candidate_per_split": scan_stats["candidate_per_split"], "valid_per_split": scan_stats["valid_per_split"],
        "smiles_failures": scan_stats["smiles_failures"], "device": str(device), "gpu_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "checkpoint": str(checkpoint),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    metrics.update({"fast_probe": True, "eligible_ligands": len(ligands), "target_dim": len(targets), "epochs": args.epochs, "batch_size": args.batch_size})
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
