"""Profile registered pathway aggregation on real, label-free EXP-036 inputs.

This isolates CPU mapping and 51-by-50 pathway-activity construction. It does
not load drug responses, checkpoints, model predictions, or create a CUDA
context. The existing unlabeled-input profiler is reused for its pinned-root
and no-symlink file reader; a separate entrypoint is needed because that
profiler validates inputs and discards decoded arrays without applying maps.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as functional


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "modeling"))

from drug_screen.models.exp046_five_arm import (  # noqa: E402
    EXP046_PATHWAY_AXIS,
    expand_mapped_pathway_map,
)
from drug_screen.transfer.pathways import parse_gmt  # noqa: E402
import profile_exp046_unlabeled_inputs as input_profile  # noqa: E402


RUN_ID = "20260925_real_pathway_activity_profile_v1"
EXPECTED = {
    "gene_axis": "bf023d7b110bd820b3b41c67ab9ce5ea6b2381ab5753c3ec6440ee2a61340610",
    "patient_axis": "824a895473d39f893317d273551a8d66b42e2f1570d644a2101db855e3fae603",
    "patient_expression": "f038d9bac0e9c287477a9f76305cdc01032fd567d47e918632c9f0235dd3c05c",
    "hallmark_gmt": "8ff1f036f0988c99d42cf9e5ea24d355f10b8ece5e0d4f6edf26a0458f480780",
    "control_maps": "96cbba6625e1a0f85b961644225ec82f21259891e1d3b4f8b9d235ea38a47f53",
}
PATHS = {
    "gene_axis": Path("data/processed/dual_route/EXP-036/direct_response_contract_v1/genes.json"),
    "patient_axis": Path("data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1/patient_axis.json"),
    "patient_expression": Path("data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1/patient_expression.npy"),
    "hallmark_gmt": Path("data/raw/exp013/msigdb/h.all.v2023.2.Hs.symbols.gmt"),
    "control_maps": Path("artifacts/experiments/EXP-046/20260924_pathway_control_maps_v1/maps.npz"),
}
MAP_KEYS = (
    ("P-C ordinary sparse", "pc"),
    ("P-R random map seed 20260922", "pr_20260922"),
    ("P-R random map seed 20260923", "pr_20260923"),
    ("P-R random map seed 20260924", "pr_20260924"),
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _environment() -> dict[str, Any]:
    release = platform.release().lower()
    if (platform.system() != "Linux"
            or not ("microsoft" in release or os.environ.get("WSL_DISTRO_NAME"))):
        raise RuntimeError("profile must run inside WSL2 Linux")
    if os.environ.get("CONDA_DEFAULT_ENV") != "drugscreening-gpu":
        raise RuntimeError("profile requires Conda environment drugscreening-gpu")
    prefix = Path(os.environ.get("CONDA_PREFIX", "")).resolve()
    python = Path(sys.executable).resolve()
    if not prefix.is_dir() or prefix not in python.parents:
        raise RuntimeError("Python executable is outside the required Conda prefix")
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "conda_env": os.environ["CONDA_DEFAULT_ENV"],
        "python_executable": python.as_posix(),
        "device_used": "CPU only; no CUDA context created",
    }


def _read_inputs(root_fd: int) -> tuple[dict[str, bytes], dict[str, str], dict[str, Any]]:
    read_start = time.perf_counter()
    blobs = {
        name: input_profile._read_allowlisted_blob(root_fd, path)
        for name, path in PATHS.items()
    }
    read_seconds = time.perf_counter() - read_start

    hash_start = time.perf_counter()
    hashes = {name: _sha256(blob) for name, blob in blobs.items()}
    hash_seconds = time.perf_counter() - hash_start
    for name, expected in EXPECTED.items():
        if hashes[name] != expected:
            raise ValueError(f"registered EXP-046 input hash mismatch: {name}")

    decode_start = time.perf_counter()
    genes_raw = json.loads(blobs["gene_axis"].decode("utf-8"))
    patients = json.loads(blobs["patient_axis"].decode("utf-8"))
    expression = np.load(io.BytesIO(blobs["patient_expression"]), allow_pickle=False)
    resource = parse_gmt(
        blobs["hallmark_gmt"].decode("utf-8"),
        resource_id="MSigDB_Hallmark",
        version="h.all.v2023.2.Hs.symbols",
    )
    with np.load(io.BytesIO(blobs["control_maps"]), allow_pickle=False) as archive:
        map_keys = tuple(archive.files)
        map_arrays = {key: np.array(archive[key], copy=True) for _, key in MAP_KEYS}
    decode_seconds = time.perf_counter() - decode_start

    genes = tuple(str(value).strip().upper() for value in genes_raw)
    if (len(genes) != EXP046_PATHWAY_AXIS.full_gene_count
            or len(set(genes)) != len(genes)
            or any(not gene for gene in genes)):
        raise ValueError("ordered EXP-036 gene axis does not match the frozen pathway contract")
    if (not isinstance(patients, list) or len(patients) != 51
            or len(set(patients)) != len(patients)
            or any(not isinstance(value, str) or not value for value in patients)):
        raise ValueError("EXP-036 patient axis is not the registered unique 51-row axis")
    expression = np.asarray(expression, dtype=np.float32, order="C")
    if (expression.shape != (len(patients), len(genes))
            or not np.isfinite(expression).all()):
        raise ValueError("patient expression is non-finite or misaligned to registered axes")
    if len(resource.pathways) != EXP046_PATHWAY_AXIS.pathway_count:
        raise ValueError("Hallmark resource differs from the frozen 50-pathway contract")
    expected_keys = {key for _, key in MAP_KEYS}
    if not expected_keys.issubset(map_keys):
        raise ValueError("frozen control-map archive is missing a registered map key")

    resource_genes = set().union(*(set(members) for members in resource.pathways.values()))
    mapped_genes = tuple(gene for gene in genes if gene in resource_genes)
    if len(mapped_genes) != EXP046_PATHWAY_AXIS.mapped_gene_count:
        raise ValueError("Hallmark mapping does not match the frozen mapped-gene count")
    mapped_positions = {gene: position for position, gene in enumerate(mapped_genes)}
    pathway_names = tuple(resource.pathways)
    true_map = np.zeros((len(pathway_names), len(mapped_genes)), dtype=np.uint8)
    for row, pathway in enumerate(pathway_names):
        for gene in resource.pathways[pathway]:
            position = mapped_positions.get(gene)
            if position is not None:
                true_map[row, position] = 1

    maps = [("P-T/P-N true Hallmark", true_map)]
    for label, key in MAP_KEYS:
        value = map_arrays[key]
        if (value.shape != (EXP046_PATHWAY_AXIS.pathway_count,
                            EXP046_PATHWAY_AXIS.mapped_gene_count)
                or not np.isin(value, (0, 1)).all()
                or not np.all(value.sum(axis=1) > 0)
                or not np.all(value.sum(axis=0) > 0)):
            raise ValueError(f"frozen map violates the registered binary axis contract: {key}")
        maps.append((label, value.astype(np.uint8, copy=False)))

    axes = {
        "n_patients": len(patients),
        "n_genes": len(genes),
        "n_mapped_genes": len(mapped_genes),
        "n_pathways": len(pathway_names),
        "expression_shape": list(expression.shape),
        "gene_axis_sha256": hashes["gene_axis"],
        "patient_axis_sha256": hashes["patient_axis"],
        "input_sha256": hashes,
        "map_keys": list(map_keys),
    }
    stages = {
        "file_read_seconds": read_seconds,
        "sha256_seconds": hash_seconds,
        "decode_seconds": decode_seconds,
    }
    data = {"genes": genes, "mapped_genes": mapped_genes,
            "pathway_names": pathway_names, "expression": expression,
            "maps": maps}
    return data, axes, stages


def run() -> dict[str, Any]:
    output_root = ROOT / "artifacts" / "experiments" / "EXP-046"
    output_dir = output_root / RUN_ID
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite profile output: {output_dir}")
    environment = _environment()
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only profile unexpectedly initialized CUDA")
    started = datetime.now(timezone.utc)
    wall_start = time.perf_counter()
    root_fd = input_profile._open_pinned_directory(ROOT)
    try:
        data, axes, stages = _read_inputs(root_fd)
    finally:
        os.close(root_fd)

    map_alignment_seconds = 0.0
    aggregation_seconds = 0.0
    map_results = []
    expression = torch.from_numpy(data["expression"])
    for label, incidence in data["maps"]:
        map_start = time.perf_counter()
        aligned = expand_mapped_pathway_map(
            incidence,
            gene_axis=data["genes"],
            mapped_gene_ids=data["mapped_genes"],
            pathway_names=data["pathway_names"],
        )
        weights = aligned.incidence_matrix
        row_sizes = weights.sum(dim=1, keepdim=True)
        if bool((row_sizes == 0).any()):
            raise ValueError(f"empty pathway after alignment: {label}")
        weights = weights / row_sizes
        map_alignment_seconds += time.perf_counter() - map_start

        aggregate_start = time.perf_counter()
        activity = functional.linear(expression, weights)
        if (tuple(activity.shape) != (axes["n_patients"], axes["n_pathways"])
                or not bool(torch.isfinite(activity).all())):
            raise ValueError(f"pathway activity output is invalid for map: {label}")
        aggregation_seconds += time.perf_counter() - aggregate_start
        map_results.append({
            "map": label,
            "edge_count": int(incidence.sum()),
            "activity_shape": list(activity.shape),
            "finite": True,
            "values_written": False,
        })
        del activity, weights, aligned

    stages["full_axis_map_alignment_seconds"] = map_alignment_seconds
    stages["real_expression_pathway_aggregation_seconds"] = aggregation_seconds
    finished = datetime.now(timezone.utc)
    receipt = {
        "experiment_id": "EXP-046",
        "run_id": RUN_ID,
        "status": "UNLABELED_REAL_PATHWAY_ACTIVITY_PROFILE_ONLY",
        "runtime": environment,
        "script_sha256": _sha256(Path(__file__).read_bytes()),
        "execution_command": (
            "wsl.exe --distribution Ubuntu --cd /mnt/d/Code/DrugScreenLab -- "
            "/home/dell/miniconda3/bin/conda run --no-capture-output "
            "-n drugscreening-gpu python scripts/modeling/profile_exp046_real_pathway_activity.py"
        ),
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "wall_seconds_excluding_receipt_write": time.perf_counter() - wall_start,
        "profiled_stages_seconds": stages,
        "input_summary": axes,
        "maps_profiled": map_results,
        "training_performed": False,
        "labels_read": False,
        "response_label_files_opened": [],
        "checkpoint_loaded": False,
        "model_predictions_computed": False,
        "gpu_used": False,
        "gpu_hours": 0.0,
        "interpretation_limit": (
            "One CPU pass of registered real expression through the frozen P-C, P-R, and true Hallmark "
            "aggregation maps. This measures input feature construction only; it is not training, "
            "prediction quality, model selection, or a complete five-arm cost estimate."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "receipt.json"
    payload = json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    write_start = time.perf_counter()
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    receipt_write_seconds = time.perf_counter() - write_start
    receipt["profiled_stages_seconds"]["initial_receipt_write_seconds"] = receipt_write_seconds
    temp_path = output_dir / "receipt.json.tmp"
    final_payload = json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with temp_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(final_payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, output_path)
    receipt_sha256 = _sha256(output_path.read_bytes())
    return {"receipt_path": output_path.relative_to(ROOT).as_posix(),
            "receipt_sha256": receipt_sha256,
            "profiled_stages_seconds": receipt["profiled_stages_seconds"],
            "input_summary": axes,
            "maps_profiled": map_results,
            "gpu_hours": 0.0}


def main() -> int:
    result = run()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
