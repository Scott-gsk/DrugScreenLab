"""CPU-only, train-side trajectory diagnosis for the frozen TranSiGen model.

This diagnostic reuses the already audited 512 TRAIN keys, their 978-gene
targets, the registered ECFP4 rows, and the upstream model implementation.  It
compares only raw-sum/no-clip with loss/(batch*978)+global-clip-5.  No response
array outside the frozen TRAIN source archive is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20261006_transigen_source_fit_trajectory_v4"
OUT = ROOT / "artifacts/experiments/EXP-050" / RUN_ID
CORE_PATH = ROOT / "scripts/modeling/compare_exp050_transigen_update_rules.py"
V3_FEATURES = ROOT / "artifacts/experiments/EXP-050/20261006_transigen_source_fit_trajectory_v3/selected_train512_ecfp4.npy"
V2_PROTOCOL = ROOT / "artifacts/experiments/EXP-050/20261001_transigen_update_rule_compare_v2/protocol.json"
AUTHORIZED_GPU_UUID = "GPU-9c366c2e-bfa5-c7b3-b16a-1117c40e6871"
MODEL_SEED = 20261003
STEP_SEED_BASE = 20262003
EVAL_SEED_BASE = 20263003
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 64
N_KEYS = 512
N_GENES = 978
MAX_STEPS = 100
CHECKPOINTS = (0, 8, 16, 32, 64, 100)
CPU_WALL_LIMIT_SECONDS = 1800.0
GPU_CONTINUATION_INTERNAL_CAP_HOURS = 1.0
FORBIDDEN_LABELS = (
    "DEV", "heldout", "outer", "locked", "PDO", "patient response",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2,
                                    sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True,
                                allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def json_safe(value: Any) -> Any:
    """Represent non-finite diagnostics as explicit strings, never silent nulls."""
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        if math.isnan(float(value)):
            return "NaN"
        return "Infinity" if float(value) > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    return value


def load_core() -> Any:
    spec = importlib.util.spec_from_file_location("exp050_audited_transigen_core_v4", CORE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import the audited update-rule runner: {CORE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bind_core_to_local_train_copies(core: Any) -> None:
    core.RUN_ID = RUN_ID
    core.KEYS = OUT / "selected_train512_condition_keys.tsv"
    core.VALUE_NPZ = OUT / "selected_train512_control_delta_treated.npz"
    core.FEATURE_CACHE = OUT / "selected_train512_ecfp4.npy"
    core.PROTOCOL = V2_PROTOCOL


def _input_manifest(core: Any) -> dict[str, Any]:
    required = {
        "keys": core.KEYS,
        "train_control_delta_treated": core.VALUE_NPZ,
        "ecfp4_features": core.FEATURE_CACHE,
    }
    return {name: {"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                   "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in required.items()}


def _code_manifest() -> dict[str, Any]:
    official = Path("/mnt/d/Code/TranSiGen/参考代码/TranSiGen/src/model.py")
    paths = {
        "diagnostic_driver": Path(__file__).resolve(),
        "audited_update_rule_runner": CORE_PATH,
        "official_model": official,
        "official_contract_audit": ROOT / "scripts/data/audit_exp050_transigen_official_contract.py",
        "feature_builder": ROOT / "artifacts/experiments/EXP-037/20260928_lincs_scale_16_121/wrapper_transigen121/code_snapshot/exp037_lincs_scale_probe.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required code sources are missing: {missing}")
    return {name: {"path": str(path), "sha256": sha256_file(path),
                   "bytes": path.stat().st_size}
            for name, path in paths.items()}


def freeze() -> dict[str, Any]:
    if OUT.exists():
        raise FileExistsError(f"immutable run directory already exists; refusing overwrite: {OUT}")
    if not V3_FEATURES.is_file():
        raise FileNotFoundError(f"the previously frozen TRAIN ECFP input is missing: {V3_FEATURES}")
    core = load_core()
    source_keys = core.KEYS
    source_values = core.VALUE_NPZ
    OUT.mkdir(parents=True)
    try:
        shutil.copy2(source_keys, OUT / "selected_train512_condition_keys.tsv")
        shutil.copy2(source_values, OUT / "selected_train512_control_delta_treated.npz")
        shutil.copy2(V3_FEATURES, OUT / "selected_train512_ecfp4.npy")
        bind_core_to_local_train_copies(core)
        keys, control, delta, features, extra = core._load_bound_inputs()
        if (len(keys) != N_KEYS or control.shape != (N_KEYS, N_GENES)
                or delta.shape != (N_KEYS, N_GENES) or features.shape != (N_KEYS, 2048)):
            raise RuntimeError("TRAIN key/value/feature inputs do not match the approved 512 x 978 contract")
        identities = keys[["selection_rank", "condition_id", "cell_id", "pert_id", "source_row"]]
        if identities["condition_id"].duplicated().any():
            raise RuntimeError("frozen TRAIN condition identities are not unique")
        if not np.array_equal(extra["treated"], control + delta):
            raise RuntimeError("TRAIN treated expression does not equal control plus delta")

        code_manifest = _code_manifest()
        input_manifest = _input_manifest(core)
        runtime = {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": sys.platform,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        protocol: dict[str, Any] = {
            "experiment_id": "EXP-050",
            "run_id": RUN_ID,
            "status": "FROZEN_CPU_TRAIN_TRAJECTORY_DIAGNOSTIC",
            "created_at_utc": utc_now(),
            "authorization": {
                "source": "user-authorized bounded continuation relayed by the research coordinator on 2026-10-06",
                "requested_lineage": [
                    "20261006_transigen_source_fit_trajectory_v1",
                    "20261006_transigen_source_fit_trajectory_v2",
                    "20261006_transigen_source_fit_trajectory_v3",
                ],
                "new_run_id_reason": "v1 and v2 are immutable manager-rejected freezes; v3 retains its failed smoke and CPU evidence",
                "resource_limits": {
                    "cpu_wall_seconds_max": CPU_WALL_LIMIT_SECONDS,
                    "gpu_hours_for_this_cpu_diagnostic": 0.0,
                    "conditional_followup_gpu_hours_internal_cap": GPU_CONTINUATION_INTERNAL_CAP_HOURS,
                    "all_gpu_costs_charge_EXP050_48_hour_cap": True,
                },
            },
            "question": "After normal TRAIN updates, does either approved TranSiGen update rule make TRAIN-key eval finite, and do training-side numeric states remain finite through at most 100 steps?",
            "data_scope": {
                "conditions": N_KEYS,
                "genes": N_GENES,
                "feature_count": 2048,
                "key_order": "the existing frozen selection_rank 1..512; minibatches are contiguous groups of 64",
                "allowed_inputs": list(input_manifest),
                "forbidden_inputs_opened": [],
                "forbidden_roles": list(FORBIDDEN_LABELS),
            },
            "model": {
                "implementation": "official upstream TranSiGen model.py; same architecture and initialization contract as the audited update-rule comparison",
                "initialization_seed": MODEL_SEED,
                "initialization": "scratch; one shared initial state hash, reconstructed identically for both arms; no checkpoint or BN calibration",
                "precision": "float32",
                "architecture": {
                    "latent_dim": 100,
                    "encoder_hidden": [1200],
                    "decoder_hidden": [800],
                    "feature_dim": 2048,
                    "feature_embed_dim": [400],
                    "beta": 0.1,
                    "dropout": 0.1,
                },
            },
            "optimizer": {
                "name": "torch.optim.Adam",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "betas": [0.9, 0.999],
                "epsilon": 1e-8,
                "step_seed": "same deterministic seed for the matched batch index in both arms",
            },
            "arms": {
                "official_raw_sum_no_clipping": {
                    "objective": "raw sum of the official six loss components",
                    "gradient_clip": None,
                },
                "normalized_global_clip_5": {
                    "objective": "official six-component sum divided by actual_batch_size * 978",
                    "gradient_clip": "global norm 5",
                },
                "matched_conditions": "same ordered rows, shared initialization, architecture, optimizer, and matched dropout/random seed; only objective scaling and clipping differ",
            },
            "trajectory": {
                "normal_train_steps_per_arm_max": MAX_STEPS,
                "reported_checkpoint_steps": list(CHECKPOINTS),
                "training_mode": "normal model.train() updates; BatchNorm updates only as a consequence of these ordinary TRAIN batches",
                "eval": "at steps 0, 8, 16, 32, 64, 100; eval over all 512 TRAIN conditions in chunks of 64; fixed independent eval seed inside a saved/restored RNG context; eval leaves the model in its prior mode and must not modify BatchNorm state",
                "step_zero_train_loss": "training-mode loss probe using a disposable copy of the shared initialization; no optimizer step and no retained model or BatchNorm state change",
                "stop_rule": "stop only the affected arm immediately after non-finite loss, gradient, post-step parameter, or optimizer state; continue the other arm if the 30-minute wall limit permits",
            },
            "outputs": [
                "protocol.json", "freeze_receipt.json", "start_receipt.json",
                "input_identity_receipt.json", "code_snapshot/", "step_trajectory.jsonl",
                "milestone_trajectory.jsonl", "cpu_run_receipt.json", "sha256_manifest.json",
            ],
            "inputs": input_manifest,
            "code": code_manifest,
            "runtime": runtime,
        }
        protocol["protocol_sha256"] = canonical_hash(protocol)
        atomic_json(OUT / "protocol.json", protocol)

        snapshot = OUT / "code_snapshot"
        snapshot.mkdir()
        for name, meta in code_manifest.items():
            source = Path(meta["path"])
            dest = snapshot / (name + source.suffix)
            shutil.copy2(source, dest)
        identity_receipt = {
            "run_id": RUN_ID,
            "key_count": len(keys),
            "unique_condition_count": int(keys["condition_id"].nunique()),
            "gene_count": int(control.shape[1]),
            "feature_count": int(features.shape[1]),
            "key_sha256": input_manifest["keys"]["sha256"],
            "control_delta_treated_sha256": input_manifest["train_control_delta_treated"]["sha256"],
            "ecfp4_sha256": input_manifest["ecfp4_features"]["sha256"],
            "ordered_key_readback_matches_frozen_contract": True,
            "treated_equals_control_plus_delta": True,
            "non_train_response_arrays_opened": False,
        }
        atomic_json(OUT / "input_identity_receipt.json", identity_receipt)
        freeze_receipt = {
            "run_id": RUN_ID,
            "status": protocol["status"],
            "protocol_sha256": protocol["protocol_sha256"],
            "protocol_file_sha256": sha256_file(OUT / "protocol.json"),
            "input_hashes": {name: value["sha256"] for name, value in input_manifest.items()},
            "code_hashes": {name: value["sha256"] for name, value in code_manifest.items()},
            "train_conditions": len(keys),
            "gene_axis": control.shape[1],
            "forbidden_response_arrays_opened": False,
        }
        atomic_json(OUT / "freeze_receipt.json", freeze_receipt)
        ledger = _ledger_snapshot(core)
        start_receipt = {
            "run_id": RUN_ID,
            "status": "CPU_TRAJECTORY_DIAGNOSTIC_READY_TO_START",
            "created_at_utc": utc_now(),
            "next_command": "cd /mnt/d/Code/DrugScreenLab && CUDA_VISIBLE_DEVICES= /home/dell/miniconda3/bin/conda run --no-capture-output -n drugscreening-gpu python scripts/modeling/diagnose_exp050_transigen_source_fit_trajectory.py --mode run",
            "cpu_wall_limit_seconds": CPU_WALL_LIMIT_SECONDS,
            "gpu_hours_for_this_diagnostic": 0.0,
            "EXP050_ledger_before": ledger,
            "same_rejected_run_directories_modified": False,
        }
        atomic_json(OUT / "start_receipt.json", start_receipt)
        return start_receipt
    except BaseException:
        # An incomplete immutable directory is evidence of a freeze failure;
        # retain it and do not silently overwrite or auto-retry under v4.
        raise


def _ledger_snapshot(core: Any) -> dict[str, Any]:
    rows = core._ledger_rows()
    state = core._ledger_state(rows)
    return {**state, "ledger_sha256": sha256_file(core.LEDGER)}


def verify_protocol(core: Any) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray,
                                        np.ndarray, dict[str, Any]]:
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    digest = protocol.get("protocol_sha256")
    payload = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    if canonical_hash(payload) != digest:
        raise RuntimeError("frozen v4 protocol hash mismatch")
    if protocol["run_id"] != RUN_ID:
        raise RuntimeError("unexpected run_id in protocol")
    if protocol["code"]["diagnostic_driver"]["sha256"] != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("diagnostic runner differs from its frozen code hash")
    for name, meta in protocol["code"].items():
        if sha256_file(Path(meta["path"])) != meta["sha256"]:
            raise RuntimeError(f"frozen code source changed: {name}")
    for name, meta in protocol["inputs"].items():
        local = OUT / Path(meta["path"]).name
        if sha256_file(local) != meta["sha256"]:
            raise RuntimeError(f"frozen input changed: {name}")
    bind_core_to_local_train_copies(core)
    keys, control, delta, features, extra = core._load_bound_inputs()
    if len(keys) != 512 or control.shape != (512, 978) or delta.shape != (512, 978):
        raise RuntimeError("loaded data violates the frozen 512-key / 978-gene TRAIN contract")
    if features.shape != (512, 2048):
        raise RuntimeError("loaded ECFP4 feature axis differs from the frozen 512 x 2048 TRAIN contract")
    return protocol, keys, control, delta, features, extra


def finite_tensor(torch: Any, value: Any) -> tuple[bool, int, int]:
    if not torch.is_tensor(value):
        return True, 0, 0
    if not (value.is_floating_point() or value.is_complex()):
        return True, 0, int(value.numel())
    mask = torch.isfinite(value)
    return bool(mask.all().item()), int((~mask).sum().item()), int(value.numel())


def finite_nested(torch: Any, value: Any) -> tuple[bool, int, int]:
    if torch.is_tensor(value):
        return finite_tensor(torch, value)
    if isinstance(value, (tuple, list)):
        parts = [finite_nested(torch, item) for item in value]
    elif isinstance(value, dict):
        parts = [finite_nested(torch, item) for item in value.values()]
    else:
        return True, 0, 0
    return (all(item[0] for item in parts), sum(item[1] for item in parts),
            sum(item[2] for item in parts))


def _state_finite(torch: Any, model: Any) -> dict[str, Any]:
    parameter_values = list(model.parameters())
    finite_flags = [bool(torch.isfinite(parameter).all().item()) for parameter in parameter_values]
    return {
        "all_parameter_tensors_finite": all(finite_flags),
        "parameter_tensor_count": len(parameter_values),
        "finite_parameter_tensor_count": sum(finite_flags),
        "parameter_value_count": int(sum(parameter.numel() for parameter in parameter_values)),
        "nonfinite_parameter_value_count": int(sum((~torch.isfinite(parameter)).sum().item()
                                                     for parameter in parameter_values)),
    }


def _optimizer_finite(torch: Any, optimizer: Any) -> dict[str, Any]:
    nonfinite = []
    for parameter_index, state in enumerate(optimizer.state.values()):
        for key, value in state.items():
            if torch.is_tensor(value) and value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
                nonfinite.append({"parameter_index": parameter_index, "state": key,
                                  "nonfinite_values": int((~torch.isfinite(value)).sum().item())})
    return {"finite": not nonfinite, "nonfinite_entries": nonfinite,
            "state_parameter_count": len(optimizer.state)}


def _bn_stats(torch: Any, model: Any, step: int) -> dict[str, Any]:
    rows = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            mean = module.running_mean.detach().cpu().numpy()
            var = module.running_var.detach().cpu().numpy()
            rows[name] = {
                "running_mean_min": float(mean.min()), "running_mean_max": float(mean.max()),
                "running_var_min": float(var.min()), "running_var_max": float(var.max()),
                "running_mean_sha256": hashlib.sha256(np.ascontiguousarray(mean).tobytes()).hexdigest(),
                "running_var_sha256": hashlib.sha256(np.ascontiguousarray(var).tobytes()).hexdigest(),
                "num_batches_tracked": int(module.num_batches_tracked.detach().cpu()),
                "finite": bool(np.isfinite(mean).all() and np.isfinite(var).all()),
            }
    if len(rows) != 5:
        raise RuntimeError(f"expected the official 5 BatchNorm layers, found {len(rows)}")
    return {"step": step, "layers": rows,
            "sha256": canonical_hash(rows),
            "source": "ordinary model.train() TRAIN minibatch updates only"}


def _model_digest(core: Any, model: Any) -> str:
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return core._state_digest(state)


def _loss_forward(torch: Any, model: Any, x1: Any, delta: Any, feature: Any) -> tuple[Any, dict[str, float]]:
    x2 = x1 + delta
    x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, _ = model(x1, feature)
    z2, mu2, logvar2 = model.encode_x2(x2)
    x2_rec = model.decode_x2(z2)
    total, mse_x1, mse_x2, mse_pert, kld_x1, kld_x2, kld_pert = model.loss(
        x1, x1_rec, mu1, logvar1, x2, x2_rec, mu2, logvar2,
        x2_pred, mu_pred, logvar_pred,
    )
    components = {
        "mse_x1": float(mse_x1.detach().cpu()),
        "mse_x2": float(mse_x2.detach().cpu()),
        "mse_pert": float(mse_pert.detach().cpu()),
        "kld_x1": float(kld_x1.detach().cpu()),
        "kld_x2": float(kld_x2.detach().cpu()),
        "kld_pert": float(kld_pert.detach().cpu()),
        "loss_total_raw_sum": float(total.detach().cpu()),
    }
    return total, components


def _batch_arrays(control: np.ndarray, delta: np.ndarray, features: np.ndarray,
                  index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = index * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(control))
    return control[start:end], delta[start:end], features[start:end]


def _train_loss_probe(torch: Any, core: Any, model_class: Any, initial_state: dict[str, Any],
                      control: np.ndarray, delta: np.ndarray, features: np.ndarray) -> dict[str, Any]:
    probe = core._new_model(torch, model_class, initial_state, torch.device("cpu"))
    probe.train()
    x1_np, delta_np, feature_np = _batch_arrays(control, delta, features, 0)
    x1 = torch.as_tensor(x1_np, dtype=torch.float32)
    d = torch.as_tensor(delta_np, dtype=torch.float32)
    f = torch.as_tensor(feature_np, dtype=torch.float32)
    with torch.random.fork_rng(devices=[], enabled=True), torch.no_grad():
        torch.manual_seed(STEP_SEED_BASE)
        _total, parts = _loss_forward(torch, probe, x1, d, f)
    del probe
    return {"loss_components": parts,
            "loss_mode": "TRAIN-mode no-grad probe on disposable initial-state copy; BN changes discarded",
            "optimizer_step_performed": False}


def _parameter_grad_finite(torch: Any, model: Any) -> dict[str, Any]:
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    flags = [bool(torch.isfinite(grad).all().item()) for grad in grads]
    return {
        "all_gradient_tensors_finite": all(flags),
        "gradient_tensor_count": len(grads),
        "finite_gradient_tensor_count": sum(flags),
        "gradient_value_count": int(sum(grad.numel() for grad in grads)),
        "nonfinite_gradient_value_count": int(sum((~torch.isfinite(grad)).sum().item() for grad in grads)),
    }


def _one_train_step(torch: Any, core: Any, model: Any, optimizer: Any,
                    control: np.ndarray, delta: np.ndarray, features: np.ndarray,
                    step: int, arm: str) -> dict[str, Any]:
    batch_index = (step - 1) % (N_KEYS // BATCH_SIZE)
    seed = STEP_SEED_BASE + batch_index
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    x1_np, d_np, f_np = _batch_arrays(control, delta, features, batch_index)
    x1 = torch.as_tensor(x1_np, dtype=torch.float32)
    d = torch.as_tensor(d_np, dtype=torch.float32)
    f = torch.as_tensor(f_np, dtype=torch.float32)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    total, parts = _loss_forward(torch, model, x1, d, f)
    if not all(math.isfinite(value) for value in parts.values()):
        return {"step": step, "arm": arm, "batch_index": batch_index,
                "batch_key_range": [batch_index * BATCH_SIZE + 1, (batch_index + 1) * BATCH_SIZE],
                "loss_components": parts, "loss_finite": False,
                "stopped": True, "stop_reason": "nonfinite_train_loss",
                "wall_seconds": time.perf_counter() - started}
    divisor = float(x1.shape[0] * N_GENES) if arm == "normalized_global_clip_5" else 1.0
    objective = total / divisor
    objective.backward()
    gradient = _parameter_grad_finite(torch, model)
    raw_gradient_norm = core._global_gradient_norm(torch, model)
    gradient["global_gradient_norm_before_clip"] = raw_gradient_norm
    gradient["gradient_finite"] = gradient["all_gradient_tensors_finite"] and math.isfinite(raw_gradient_norm)
    gradient["gradient_clip"] = None
    if not gradient["gradient_finite"]:
        return {"step": step, "arm": arm, "batch_index": batch_index,
                "batch_key_range": [batch_index * BATCH_SIZE + 1, (batch_index + 1) * BATCH_SIZE],
                "loss_components": parts, "optimizer_objective": float(objective.detach().cpu()),
                "objective_divisor": divisor, "gradient": gradient,
                "parameters_before_step": _state_finite(torch, model),
                "optimizer_before_step": _optimizer_finite(torch, optimizer),
                "stopped": True, "stop_reason": "nonfinite_gradient_before_optimizer_step",
                "wall_seconds": time.perf_counter() - started}
    if arm == "normalized_global_clip_5":
        clipped_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0,
                                                             error_if_nonfinite=True).detach().cpu())
        gradient["gradient_clip"] = {"max_norm": 5.0, "returned_preclip_norm": clipped_norm,
                                     "postclip_norm": core._global_gradient_norm(torch, model)}
    parameters = list(model.parameters())
    before = [parameter.detach().clone() for parameter in parameters]
    optimizer.step()
    parameter_check = _state_finite(torch, model)
    optimizer_check = _optimizer_finite(torch, optimizer)
    update_norm = core._parameter_update_norm(torch, before, model)
    elapsed = time.perf_counter() - started
    finite_update = math.isfinite(update_norm)
    stop_reason = None
    if not parameter_check["all_parameter_tensors_finite"] or not finite_update:
        stop_reason = "nonfinite_parameter_after_optimizer_step"
    elif not optimizer_check["finite"]:
        stop_reason = "nonfinite_optimizer_state_after_step"
    return {
        "step": step, "arm": arm, "batch_index": batch_index,
        "batch_key_range": [batch_index * BATCH_SIZE + 1, (batch_index + 1) * BATCH_SIZE],
        "seed": seed, "loss_components": parts,
        "optimizer_objective": float(objective.detach().cpu()),
        "objective_divisor": divisor,
        "loss_finite": True,
        "gradient": gradient,
        "parameter_update_l2_norm": update_norm,
        "parameter_check": parameter_check,
        "optimizer_check": optimizer_check,
        "optimizer_step_finite": finite_update,
        "stopped": stop_reason is not None,
        "stop_reason": stop_reason,
        "wall_seconds": elapsed,
    }


def _rng_snapshot(torch: Any) -> dict[str, Any]:
    return {"torch_cpu": torch.random.get_rng_state().clone(),
            "python": random.getstate(), "numpy": np.random.get_state()}


def _rng_equal(torch: Any, left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (torch.equal(left["torch_cpu"], right["torch_cpu"])
            and left["python"] == right["python"]
            and left["numpy"][0] == right["numpy"][0]
            and np.array_equal(left["numpy"][1], right["numpy"][1])
            and left["numpy"][2:] == right["numpy"][2:])


def _eval_train_keys(torch: Any, core: Any, model: Any,
                     control: np.ndarray, features: np.ndarray,
                     step: int, arm: str) -> dict[str, Any]:
    previous_mode = model.training
    bn_before = _bn_stats(torch, model, step)
    rng_before = _rng_snapshot(torch)
    first_bad: list[str] = []
    handles = []

    def hook_factory(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            finite, _bad_count, _count = finite_nested(torch, output)
            if not finite and name not in first_bad:
                first_bad.append(name or "<model-root>")
        return hook

    for name, module in model.named_modules():
        handles.append(module.register_forward_hook(hook_factory(name)))

    valid_keys = 0
    finite_values = 0
    value_count = 0
    logvar_min = math.inf
    logvar_max = -math.inf
    logvar_nonfinite = 0
    logvar_value_count = 0
    error_rows = []
    key_checks = []
    logvar_checks = []
    eval_seed = EVAL_SEED_BASE + step
    try:
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(eval_seed)
            model.eval()
            with torch.no_grad():
                for start in range(0, N_KEYS, BATCH_SIZE):
                    end = min(start + BATCH_SIZE, N_KEYS)
                    x1 = torch.as_tensor(control[start:end], dtype=torch.float32)
                    f = torch.as_tensor(features[start:end], dtype=torch.float32)
                    try:
                        output = model(x1, f)
                        prediction = output[3] - x1
                        key_finite = torch.isfinite(prediction).all(dim=1)
                        valid_keys += int(key_finite.sum().item())
                        finite_values += int(torch.isfinite(prediction).sum().item())
                        value_count += int(prediction.numel())
                        if len(output) > 5 and torch.is_tensor(output[5]):
                            logvar = output[5]
                            logvar_mask = torch.isfinite(logvar)
                            count = int(logvar_mask.sum().item())
                            logvar_nonfinite += int((~logvar_mask).sum().item())
                            logvar_value_count += int(logvar.numel())
                            if count:
                                finite_logvar = logvar[logvar_mask]
                                batch_min = float(finite_logvar.min().item())
                                batch_max = float(finite_logvar.max().item())
                                logvar_min = min(logvar_min, batch_min)
                                logvar_max = max(logvar_max, batch_max)
                            logvar_checks.append({"rows": [start, end], "finite_values": count,
                                                  "total_values": int(logvar.numel())})
                        key_checks.append({"rows": [start, end],
                                           "finite_keys": int(key_finite.sum().item()),
                                           "total_keys": end - start,
                                           "finite_prediction_values": int(torch.isfinite(prediction).sum().item()),
                                           "total_prediction_values": int(prediction.numel())})
                    except Exception as exc:  # retain a usable per-key denominator on module failures
                        error_rows.append({"rows": [start, end],
                                           "error": f"{type(exc).__name__}: {exc}"})
                        value_count += (end - start) * N_GENES
                        logvar_value_count += (end - start) * 100
            model.train(previous_mode)
    finally:
        for handle in handles:
            handle.remove()
        model.train(previous_mode)
    rng_after = _rng_snapshot(torch)
    bn_after = _bn_stats(torch, model, step)
    state = _state_finite(torch, model)
    optimizer_unchanged = bn_before["sha256"] == bn_after["sha256"]
    rng_restored = _rng_equal(torch, rng_before, rng_after)
    if not optimizer_unchanged:
        raise RuntimeError("eval-mode TRAIN pass changed BatchNorm running statistics")
    if not rng_restored:
        raise RuntimeError("independent eval did not restore the train-side random state")
    return {
        "step": step, "arm": arm, "eval_seed": eval_seed,
        "finite_train_key_fraction": valid_keys / N_KEYS,
        "finite_train_keys": valid_keys,
        "train_key_denominator": N_KEYS,
        "finite_prediction_value_fraction": finite_values / value_count if value_count else 0.0,
        "finite_prediction_values": finite_values,
        "prediction_value_denominator": value_count,
        "first_nonfinite_forward_module": first_bad[0] if first_bad else None,
        "forward_modules_with_nonfinite_output": first_bad,
        "logvar_prediction_finite_value_count": logvar_value_count - logvar_nonfinite,
        "logvar_prediction_nonfinite_value_count": logvar_nonfinite,
        "logvar_prediction_value_count": logvar_value_count,
        "logvar_prediction_finite_min": None if logvar_min == math.inf else logvar_min,
        "logvar_prediction_finite_max": None if logvar_max == -math.inf else logvar_max,
        "logvar_prediction_finite_abs_max": (None if logvar_min == math.inf
                                              else max(abs(logvar_min), abs(logvar_max))),
        "key_chunk_checks": key_checks,
        "logvar_chunk_checks": logvar_checks,
        "evaluation_errors": error_rows,
        "parameter_state": state,
        "bn_stats_before": bn_before,
        "bn_stats_after": bn_after,
        "bn_unchanged_by_eval": optimizer_unchanged,
        "train_rng_restored_after_eval": rng_restored,
    }


def run() -> dict[str, Any]:
    if not OUT.is_dir() or not (OUT / "start_receipt.json").is_file():
        raise FileNotFoundError("v4 must be frozen before execution")
    if (OUT / "cpu_run_receipt.json").exists():
        raise FileExistsError("v4 CPU run already has a terminal receipt; it cannot be retried")
    # Make CUDA unavailable before importing torch; this is strictly a CPU task.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    protocol_started = utc_now()
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    deadline = wall_started + CPU_WALL_LIMIT_SECONDS
    old_alarm = signal.getsignal(signal.SIGALRM)

    def alarm_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError("CPU trajectory diagnostic reached the 30-minute hard wall limit")

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, CPU_WALL_LIMIT_SECONDS)
    core = load_core()
    before_ledger = _ledger_snapshot(core)
    protocol: dict[str, Any] | None = None
    arms_status: dict[str, dict[str, Any]] = {}
    error_text: str | None = None
    run_status = "FAILED_ENGINEERING_OR_NUMERIC"
    step_path = OUT / "step_trajectory.jsonl"
    milestone_path = OUT / "milestone_trajectory.jsonl"
    try:
        if (sys.platform != "linux" or os.environ.get("CONDA_DEFAULT_ENV") != "drugscreening-gpu"
                or "microsoft" not in Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").lower()):
            raise RuntimeError("Python must run inside WSL2 Conda environment drugscreening-gpu")
        protocol, keys, control, delta, features, extra = verify_protocol(core)
        import torch
        if torch.cuda.is_available():
            raise RuntimeError("CPU diagnostic unexpectedly has a visible CUDA device")

        torch.set_num_threads(2)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        model_class, _source_module = core._load_model_class()
        core._set_matched_rng(torch, MODEL_SEED)
        template = model_class(core.N_GENES, core.LATENT_DIM, core.ENCODER_HIDDEN,
                               core.DECODER_HIDDEN, core.N_FEATURES,
                               core.FEATURE_EMBED_DIM, device=torch.device("cpu"),
                               init_w=True, beta=core.BETA, dropout=core.DROPOUT,
                               random_seed=MODEL_SEED)
        initial_state = {name: tensor.detach().cpu().clone()
                         for name, tensor in template.state_dict().items()}
        initial_state_sha = core._state_digest(initial_state)
        del template
        with (OUT / "initial_state_receipt.json").open("w", encoding="utf-8") as stream:
            json.dump({"initial_state_sha256": initial_state_sha,
                       "model_seed": MODEL_SEED,
                       "shared_across_arms": True,
                       "checkpoint_loaded": False}, stream, indent=2)
            stream.write("\n")

        for arm in ("official_raw_sum_no_clipping", "normalized_global_clip_5"):
            if time.perf_counter() >= deadline:
                arms_status[arm] = {"status": "NOT_STARTED_CPU_TIME_LIMIT"}
                continue
            model = core._new_model(torch, model_class, initial_state, torch.device("cpu"))
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                         weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-8)
            probe = _train_loss_probe(torch, core, model_class, initial_state,
                                      control, delta, features)
            arms_status[arm] = {
                "status": "RUNNING",
                "initial_model_state_sha256": initial_state_sha,
                "step_zero_train_loss": probe,
                "last_completed_step": 0,
                "stop_reason": None,
            }
            eval0 = _eval_train_keys(torch, core, model, control, features, 0, arm)
            zero_checkpoint = {
                "run_id": RUN_ID, "arm": arm, "step": 0,
                "train_loss": probe["loss_components"],
                "gradient_state": {"finite": None, "meaning": "no optimizer step yet"},
                "parameter_state": _state_finite(torch, model),
                "optimizer_state": _optimizer_finite(torch, optimizer),
                "eval": eval0,
                "batch_norm": eval0["bn_stats_after"],
                "model_state_sha256": _model_digest(core, model),
                "elapsed_wall_seconds": time.perf_counter() - wall_started,
            }
            append_jsonl(milestone_path, zero_checkpoint)
            last_logged_step = 0
            loss_since_checkpoint: list[float] = []
            for step in range(1, MAX_STEPS + 1):
                if time.perf_counter() >= deadline:
                    arms_status[arm].update({"status": "TIME_LIMIT", "stop_reason": "30-minute CPU wall limit"})
                    break
                row = _one_train_step(torch, core, model, optimizer,
                                     control, delta, features, step, arm)
                row["run_id"] = RUN_ID
                row["initial_model_state_sha256"] = initial_state_sha
                row["elapsed_wall_seconds"] = time.perf_counter() - wall_started
                append_jsonl(step_path, row)
                if row.get("loss_components", {}).get("loss_total_raw_sum") is not None:
                    value = row["loss_components"]["loss_total_raw_sum"]
                    if math.isfinite(value):
                        loss_since_checkpoint.append(value)
                arms_status[arm]["last_completed_step"] = step if not row["stopped"] or row.get("parameter_check", {}).get("all_parameter_tensors_finite", False) else step - 1
                if row["stopped"]:
                    arms_status[arm].update({"status": "STOPPED_NUMERIC", "stop_reason": row["stop_reason"],
                                             "failing_step": step})
                    failure_checkpoint = {
                        "run_id": RUN_ID, "arm": arm, "step": step,
                        "train_loss": row.get("loss_components"),
                        "gradient_state": row.get("gradient"),
                        "parameter_state": row.get("parameter_check", _state_finite(torch, model)),
                        "optimizer_state": row.get("optimizer_check", _optimizer_finite(torch, optimizer)),
                        "eval": None,
                        "batch_norm": _bn_stats(torch, model, step),
                        "model_state_sha256": _model_digest(core, model),
                        "stop_reason": row["stop_reason"],
                        "elapsed_wall_seconds": time.perf_counter() - wall_started,
                    }
                    append_jsonl(milestone_path, failure_checkpoint)
                    break
                if step in CHECKPOINTS:
                    eval_result = _eval_train_keys(torch, core, model, control, features, step, arm)
                    checkpoint = {
                        "run_id": RUN_ID, "arm": arm, "step": step,
                        "train_loss_last_update_raw_sum": row["loss_components"]["loss_total_raw_sum"],
                        "train_loss_last_update_objective": row["optimizer_objective"],
                        "train_loss_mean_raw_sum_since_previous_checkpoint": (
                            float(np.mean(loss_since_checkpoint)) if loss_since_checkpoint else None),
                        "train_loss_updates_in_checkpoint_interval": len(loss_since_checkpoint),
                        "gradient_state": row["gradient"],
                        "parameter_state": _state_finite(torch, model),
                        "optimizer_state": _optimizer_finite(torch, optimizer),
                        "eval": eval_result,
                        "batch_norm": eval_result["bn_stats_after"],
                        "model_state_sha256": _model_digest(core, model),
                        "elapsed_wall_seconds": time.perf_counter() - wall_started,
                    }
                    append_jsonl(milestone_path, checkpoint)
                    last_logged_step = step
                    loss_since_checkpoint = []
                    arms_status[arm]["last_eval_finite_key_fraction"] = eval_result["finite_train_key_fraction"]
                    arms_status[arm]["last_eval_first_nonfinite_module"] = eval_result["first_nonfinite_forward_module"]
                    if step == MAX_STEPS:
                        arms_status[arm].update({"status": "COMPLETED_100_STEPS",
                                                 "stop_reason": None})
            if arms_status[arm]["status"] == "RUNNING":
                arms_status[arm]["status"] = "INCOMPLETE_BEFORE_100"
                arms_status[arm]["last_logged_checkpoint_step"] = last_logged_step
            del optimizer
            del model
        completed = all(arms_status.get(arm, {}).get("status") == "COMPLETED_100_STEPS"
                        for arm in ("official_raw_sum_no_clipping", "normalized_global_clip_5"))
        failed = any(arms_status.get(arm, {}).get("status") == "STOPPED_NUMERIC"
                     for arm in ("official_raw_sum_no_clipping", "normalized_global_clip_5"))
        timed_out = any(value.get("status") in {"TIME_LIMIT", "NOT_STARTED_CPU_TIME_LIMIT"}
                        for value in arms_status.values())
        if completed:
            run_status = "COMPLETE_BOTH_ARMS_STEP100_EVAL_RECORDED"
        elif failed:
            run_status = "STOPPED_AFFECTED_ARM_ON_NONFINITE_TRAIN_STATE"
        elif timed_out:
            run_status = "INCOMPLETE_CPU_TIME_LIMIT"
        else:
            run_status = "INCOMPLETE_DIAGNOSTIC"
    except TimeoutError as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        run_status = "INCOMPLETE_CPU_TIME_LIMIT"
    except BaseException as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        run_status = "FAILED_ENGINEERING_OR_NUMERIC"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_alarm)
        after_ledger = _ledger_snapshot(core)
        if before_ledger["ledger_sha256"] != after_ledger["ledger_sha256"]:
            error_text = (error_text + "; " if error_text else "") + "GPU ledger changed during CPU-only diagnostic"
            run_status = "FAILED_LEDGER_INVARIANT"
        wall_elapsed = time.perf_counter() - wall_started
        cpu_elapsed = time.process_time() - cpu_started
        receipt = {
            "experiment_id": "EXP-050", "run_id": RUN_ID,
            "status": run_status,
            "started_at_utc": protocol_started,
            "completed_at_utc": utc_now(),
            "wall_elapsed_seconds": wall_elapsed,
            "process_cpu_seconds": cpu_elapsed,
            "wall_limit_seconds": CPU_WALL_LIMIT_SECONDS,
            "within_30_minute_wall_limit": wall_elapsed <= CPU_WALL_LIMIT_SECONDS + 1,
            "gpu_hours_charged_by_this_diagnostic": 0.0,
            "exp050_gpu_ledger_before": before_ledger,
            "exp050_gpu_ledger_after": after_ledger,
            "arms": arms_status,
            "error": error_text,
            "forbidden_response_arrays_opened": False,
            "interpretation_boundary": "TRAIN-only optimization/evaluation diagnostics. No validation, held-out, outer, locked, PDO, drug-response, efficacy, or generalization conclusion is produced.",
        }
        atomic_json(OUT / "cpu_run_receipt.json", receipt)
        file_hashes = {}
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and path.name != "sha256_manifest.json":
                file_hashes[str(path.relative_to(OUT)).replace("\\", "/")] = sha256_file(path)
        atomic_json(OUT / "sha256_manifest.json", {
            "run_id": RUN_ID, "created_at_utc": utc_now(),
            "file_count": len(file_hashes), "files": file_hashes,
        })
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "run"), required=True)
    args = parser.parse_args()
    result = freeze() if args.mode == "freeze" else run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
