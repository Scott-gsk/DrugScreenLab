"""Replay the frozen EXP-037 TranSiGen checkpoint on its validation inputs.

This checks checkpoint loading, preprocessing, the ordered 978-gene input axis,
and the saved validation-prediction artifact. It never reads validation labels,
target treatment responses, or outer-test response values.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/experiments/EXP-037/20260928_lincs_scale_16_121"
PREPARED = RUN / "prepared"
MODEL_OUTPUT = RUN / "transigen_121"
SOURCE_MODEL = Path(
    "/mnt/d/Code/TranSiGen/参考代码/TranSiGen/src/model.py"
)
SOURCE_MODEL_SHA256 = "a96706b11b29c8be3122640e9a05a54a1e2210039f3fd8a30b0f664d62cf7c0f"
RESERVED_GPU_HOURS = 0.25
ATOL = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_model(source_model: Path, device: torch.device) -> torch.nn.Module:
    if sha256(source_model) != SOURCE_MODEL_SHA256:
        raise ValueError("official TranSiGen source hash differs from the frozen EXP-037 receipt")
    sys.path.insert(0, str(source_model.parent))
    spec = importlib.util.spec_from_file_location("exp050_official_transigen", source_model)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official TranSiGen model from {source_model}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(20260922)
    torch.cuda.manual_seed_all(20260922)
    model = module.TranSiGen(
        978, 100, [1200], [800], 2048, [400], device=device,
        init_w=True, beta=0.1, dropout=0.1, random_seed=20260922,
    ).to(device)
    checkpoint = torch.load(
        MODEL_OUTPUT / "transigen_best.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def predict_validation(
    model: torch.nn.Module,
    control: np.ndarray,
    features: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
        torch.manual_seed(20260922)
        torch.cuda.manual_seed_all(20260922)
        for rows in np.array_split(indices, max(1, int(np.ceil(len(indices) / 256)))):
            if not len(rows):
                continue
            baseline = torch.as_tensor(control[rows], device=device)
            drug = torch.as_tensor(features[rows], device=device)
            _, _, _, treatment, *_ = model(baseline, drug)
            chunks.append((treatment - baseline).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def run(output: Path, device_index: int, run_id: str) -> dict[str, Any]:
    if sys.platform != "linux" or os.environ.get("CONDA_DEFAULT_ENV") != "drugscreening-gpu":
        raise RuntimeError("EXP-050 requires WSL2 Conda environment drugscreening-gpu")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing replay output: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; this stochastic replay is reserved for the source GPU")

    output.mkdir(parents=True)
    ledger = ROOT / "artifacts/experiments/EXP-050/gpu_ledger.jsonl"
    torch.set_num_threads(4)
    device = torch.device(f"cuda:{device_index}")
    device_name = torch.cuda.get_device_name(device)
    attempt_id = f"EXP050-{run_id.upper()}"
    started = time.monotonic()
    start_event = {
        "experiment_id": "EXP-050",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "event": "attempt_started",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reserved_gpu_hours": RESERVED_GPU_HOURS,
        "device": str(device),
        "device_name": device_name,
        "operation": "frozen checkpoint validation replay; no training",
    }
    append_jsonl(ledger, start_event)

    try:
        manifest = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "PREPARED_TRAIN_VALIDATION_ONLY":
            raise ValueError("EXP-037 prepared manifest is not train/validation-only")
        expected_hashes = manifest["files_sha256"]
        input_paths = {
            "prepared_manifest": PREPARED / "manifest.json",
            "control": PREPARED / "control.npy",
            "ecfp4": PREPARED / "ecfp4.npy",
            "validation_indices": PREPARED / "validation.npy",
            "checkpoint": MODEL_OUTPUT / "transigen_best.pt",
            "preprocess": MODEL_OUTPUT / "transigen_preprocess.npz",
            "saved_validation_prediction": MODEL_OUTPUT / "transigen_predictions.npy",
            "official_model_source": SOURCE_MODEL,
        }
        expected_file_names = {
            "control": "control.npy",
            "ecfp4": "ecfp4.npy",
            "validation_indices": "validation.npy",
        }
        hashes = {name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                  for name, path in input_paths.items()}
        for name, filename in expected_file_names.items():
            if hashes[name]["sha256"] != expected_hashes[filename]:
                raise ValueError(f"prepared {filename} does not match the EXP-037 manifest")
        if hashes["official_model_source"]["sha256"] != SOURCE_MODEL_SHA256:
            raise ValueError("official source model differs from the frozen EXP-037 receipt")

        indices = np.load(PREPARED / "validation.npy", allow_pickle=False)
        control = np.load(PREPARED / "control.npy", mmap_mode="r", allow_pickle=False)
        features = np.load(PREPARED / "ecfp4.npy", mmap_mode="r", allow_pickle=False)
        if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                or len(indices) != int(manifest["n_validation"])
                or indices.min() < 0 or indices.max() >= len(control)
                or control.ndim != 2 or control.shape[1] != 978
                or features.shape != (len(control), 2048)):
            raise ValueError("validation indices or source input shapes violate the frozen contract")
        with np.load(MODEL_OUTPUT / "transigen_preprocess.npz", allow_pickle=False) as prep:
            preprocess_shapes = {key: list(prep[key].shape)
                                 for key in ("state_mean", "state_scale", "gene_scale")}
            if any(prep[key].shape != (978,) or not np.isfinite(prep[key]).all()
                   for key in ("state_mean", "state_scale", "gene_scale")):
                raise ValueError("frozen TranSiGen preprocessing values are invalid")

        saved = np.load(MODEL_OUTPUT / "transigen_predictions.npy", mmap_mode="r", allow_pickle=False)
        if saved.shape != (len(indices), 978) or not np.isfinite(saved).all():
            raise ValueError("saved validation prediction shape/value contract is invalid")

        model = load_model(SOURCE_MODEL, device)
        replay = predict_validation(model, control, features, indices, device)
        if replay.shape != saved.shape or not np.isfinite(replay).all():
            raise ValueError("replayed validation predictions have invalid shape or values")
        diff = np.abs(replay.astype(np.float64) - np.asarray(saved, dtype=np.float64))
        result = {
            "experiment_id": "EXP-050",
            "run_id": run_id,
            "status": "PASS" if np.allclose(replay, saved, rtol=0.0, atol=ATOL) else "MISMATCH",
            "environment": "WSL2 Conda drugscreening-gpu",
            "device": str(device),
            "device_name": device_name,
            "n_validation_conditions": int(len(indices)),
            "gene_count": 978,
            "checkpoint_loaded": True,
            "preprocess_shapes": preprocess_shapes,
            "replay_match": bool(np.allclose(replay, saved, rtol=0.0, atol=ATOL)),
            "absolute_tolerance": ATOL,
            "max_absolute_difference": float(diff.max()),
            "mean_absolute_difference": float(diff.mean()),
            "p99_absolute_difference": float(np.quantile(diff, 0.99)),
            "validation_labels_read": False,
            "outer_test_labels_read": False,
            "target_treated_responses_read": False,
            "inference_boundary": "source validation prediction replay only; no performance score or target-domain claim",
            "input_hashes": hashes,
        }
        output.mkdir(parents=True, exist_ok=True)
        replay_path = output / "replayed_validation_prediction.npy"
        np.save(replay_path, replay, allow_pickle=False)
        result["replayed_prediction"] = {
            "path": replay_path.relative_to(ROOT).as_posix(),
            "bytes": replay_path.stat().st_size,
            "sha256": sha256(replay_path),
        }
        elapsed = time.monotonic() - started
        result["elapsed_seconds"] = elapsed
        result["gpu_hours_charged"] = elapsed / 3600.0
        (output / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        append_jsonl(ledger, {
            "experiment_id": "EXP-050",
            "run_id": run_id,
            "attempt_id": attempt_id,
            "event": "attempt_finished",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gpu_hours_charged": elapsed / 3600.0,
            "status": result["status"],
            "summary_sha256": sha256(output / "summary.json"),
            "replayed_prediction_sha256": result["replayed_prediction"]["sha256"],
        })
        return result
    except Exception as exc:
        elapsed = time.monotonic() - started
        append_jsonl(ledger, {
            "experiment_id": "EXP-050",
            "run_id": run_id,
            "attempt_id": attempt_id,
            "event": "attempt_finished",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gpu_hours_charged": elapsed / 3600.0,
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/experiments/EXP-050/20260930_transigen_validation_replay_v2"))
    parser.add_argument("--run-id", default="20260930_transigen_validation_replay_v2")
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    print(json.dumps(run(output, args.device_index, args.run_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
