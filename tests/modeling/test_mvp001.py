from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from drug_screen.modeling import mvp001


def _write_manifest(tmp_path: Path, records: list[dict[str, object]]) -> tuple[Path, Path]:
    cache = np.zeros((8, mvp001.GENE_COUNT), dtype=np.float32)
    base = np.arange(mvp001.GENE_COUNT, dtype=np.float32) / 100
    cache[0] = base; cache[1] = 0
    cache[2] = base * 2; cache[3] = 0
    cache[4] = base * 1.5; cache[5] = 0
    cache[6] = base * 2.5; cache[7] = 0
    cache_path = tmp_path / "cache.npy"
    np.save(cache_path, cache)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "format": mvp001.MANIFEST_FORMAT,
        "cache": {"relative_path": "cache.npy", "sha256": sha256(cache_path.read_bytes()).hexdigest(), "shape": [8, mvp001.GENE_COUNT]},
        "records": records,
    }), encoding="utf-8")
    return manifest, cache_path


def _records() -> list[dict[str, object]]:
    return [
        {"sample_id": "train-d1", "treatment_group_id": "group-train-d1", "drug_id": "d1", "dose_id": "1uM", "time_id": "24h", "split": "train", "treatment_cache_row": 0, "control_cache_row": 1},
        {"sample_id": "test-d1", "treatment_group_id": "group-test-d1", "drug_id": "d1", "dose_id": "1uM", "time_id": "24h", "split": "test", "treatment_cache_row": 4, "control_cache_row": 5},
        {"sample_id": "train-d2", "treatment_group_id": "group-train-d2", "drug_id": "d2", "dose_id": "1uM", "time_id": "24h", "split": "train", "treatment_cache_row": 2, "control_cache_row": 3},
        {"sample_id": "test-d2", "treatment_group_id": "group-test-d2", "drug_id": "d2", "dose_id": "1uM", "time_id": "24h", "split": "test", "treatment_cache_row": 6, "control_cache_row": 7},
    ]


def _config() -> dict[str, object]:
    return {
        "experiment_id": "MVP-001", "phase": "tiny",
        "data": {"manifest_format": mvp001.MANIFEST_FORMAT, "gene_count": 978, "max_records": None},
        "model": {"embedding_dim": 4},
        "training": {"seed": 7, "batch_size": 2, "epochs": 2, "learning_rate": 0.01, "weight_decay": 0.0},
    }


def test_compact_manifest_requires_explicit_row_mapping(tmp_path: Path):
    broken = _records()
    del broken[0]["control_cache_row"]
    manifest, _ = _write_manifest(tmp_path, broken)
    with pytest.raises(mvp001.IntegrationError, match="missing fields"):
        mvp001.CompactManifest.load(manifest, data_root=tmp_path)


def test_run_trains_single_seed_and_writes_compact_artifacts(tmp_path: Path):
    manifest, _ = _write_manifest(tmp_path, _records())
    result = mvp001.run(_config(), manifest, tmp_path / "output", data_root=tmp_path)
    assert result["status"] == "MODEL_STAGE_COMPLETE"
    assert result["dataset"] == {"train_rows": 2, "test_rows": 2, "gene_count": 978, "split": "predeclared_within_drug_perturbation_holdout"}
    assert result["held_out_metrics"]["constant_train_mean_baseline"]["macro_across_drugs"]["rmse"] is not None
    assert (tmp_path / "output" / "summary.json").is_file()
    assert (tmp_path / "output" / "model.pt").is_file()


def test_run_rejects_unseen_drug_in_held_out_rows(tmp_path: Path):
    rows = _records()
    rows[1]["drug_id"] = "new-drug"
    manifest, _ = _write_manifest(tmp_path, rows)
    with pytest.raises(mvp001.IntegrationError, match="train occurrence"):
        mvp001.run(_config(), manifest, tmp_path / "output", data_root=tmp_path)


def test_bounded_subset_keeps_treatment_replicates_atomic():
    rows = [
        mvp001.PerturbationRow("train-1a", "g-train-1", "d1", "1uM", "24h", "train", 0, 1),
        mvp001.PerturbationRow("train-1b", "g-train-1", "d1", "1uM", "24h", "train", 2, 3),
        mvp001.PerturbationRow("test-1a", "g-test-1", "d1", "2uM", "24h", "test", 4, 5),
        mvp001.PerturbationRow("test-1b", "g-test-1", "d1", "2uM", "24h", "test", 6, 7),
        mvp001.PerturbationRow("train-2", "g-train-2", "d1", "2uM", "24h", "train", 0, 1),
    ]
    selected = mvp001._select_subset(rows, 4)
    assert [row.sample_id for row in selected] == ["train-1a", "train-1b", "test-1a", "test-1b", "train-2"]
    assert {row.treatment_group_id for row in selected} == {"g-train-1", "g-test-1", "g-train-2"}


def test_train_baseline_targets_are_equal_weighted_by_group():
    rows = [
        mvp001.PerturbationRow("train-1a", "g-train-1", "d1", "1uM", "24h", "train", 0, 1),
        mvp001.PerturbationRow("train-1b", "g-train-1", "d1", "1uM", "24h", "train", 2, 3),
        mvp001.PerturbationRow("train-2", "g-train-2", "d1", "1uM", "6h", "train", 4, 5),
    ]
    targets = np.asarray([[0.0] * mvp001.GENE_COUNT, [0.0] * mvp001.GENE_COUNT, [10.0] * mvp001.GENE_COUNT], dtype=np.float32)
    means, drugs = mvp001._grouped_targets(rows, targets)
    assert drugs == {"g-train-1": "d1", "g-train-2": "d1"}
    assert float(means["g-train-1"][0]) == 0.0
    assert float(means["g-train-2"][0]) == 10.0
    assert float(np.mean(np.stack(list(means.values())), axis=0)[0]) == 5.0
