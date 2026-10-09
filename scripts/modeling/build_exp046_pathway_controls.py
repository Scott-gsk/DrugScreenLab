"""Build and hash EXP-046 P-C/P-R pathway maps without reading response labels."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from drug_screen.transfer.pathway_controls import (  # noqa: E402
    PC_SALT,
    PR_MAX_PROPOSALS_PER_EDGE,
    PR_MIN_ACCEPTED_PER_EDGE,
    PR_MIN_EDGE_HAMMING_FRACTION,
    build_ordinary_sparse_control,
    randomize_bipartite_pathways,
)
from drug_screen.transfer.pathways import parse_gmt  # noqa: E402


RUN_ID = "20260924_pathway_control_maps_v1"
EXPECTED_GMT_SHA256 = "8ff1f036f0988c99d42cf9e5ea24d355f10b8ece5e0d4f6edf26a0458f480780"
EXPECTED_GENE_AXIS_SHA256 = "bf023d7b110bd820b3b41c67ab9ce5ea6b2381ab5753c3ec6440ee2a61340610"
EXPECTED_GENE_COUNT = 19_091
EXPECTED_PATHWAYS = 50
EXPECTED_MAPPED_GENES = 4_366
EXPECTED_EDGE_COUNT = 7_299


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _require_wsl_conda() -> dict[str, str]:
    release = platform.release().lower()
    if platform.system() != "Linux" or not ("microsoft" in release or os.environ.get("WSL_DISTRO_NAME")):
        raise RuntimeError("EXP-046 map construction must run inside WSL2")
    if os.environ.get("CONDA_DEFAULT_ENV") != "drugscreening-gpu":
        raise RuntimeError("EXP-046 map construction requires Conda environment drugscreening-gpu")
    prefix = Path(os.environ.get("CONDA_PREFIX", "")).resolve()
    python = Path(sys.executable).resolve()
    if not prefix.is_dir() or prefix not in python.parents:
        raise RuntimeError("Python executable is not inside the active drugscreening-gpu prefix")
    return {
        "platform": platform.platform(),
        "conda_env": os.environ["CONDA_DEFAULT_ENV"],
        "conda_prefix": prefix.as_posix(),
        "python_executable": python.as_posix(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "device": "cpu",
    }


def _load_inputs() -> tuple[list[str], tuple[str, ...], np.ndarray, dict[str, str], dict[str, Any]]:
    config_path = ROOT / "configs/experiments/EXP-046.yaml"
    gmt_path = ROOT / "data/raw/exp013/msigdb/h.all.v2023.2.Hs.symbols.gmt"
    axis_path = ROOT / "data/processed/dual_route/EXP-036/direct_response_contract_v1/genes.json"
    for path in (config_path, gmt_path, axis_path):
        if not path.is_file():
            raise FileNotFoundError(path.as_posix())
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gmt_bytes = gmt_path.read_bytes()
    axis_bytes = axis_path.read_bytes()
    config_hash = _sha256(config_path)
    gmt_hash = hashlib.sha256(gmt_bytes).hexdigest()
    axis_hash = hashlib.sha256(axis_bytes).hexdigest()
    if gmt_hash != EXPECTED_GMT_SHA256 or axis_hash != EXPECTED_GENE_AXIS_SHA256:
        raise ValueError("EXP-046 registered pathway resource or EXP-036 gene-axis hash changed")
    if config.get("experiment_id") != "EXP-046":
        raise ValueError("configuration is not EXP-046")
    genes_raw = json.loads(axis_bytes.decode("utf-8"))
    if not isinstance(genes_raw, list) or len(genes_raw) != EXPECTED_GENE_COUNT:
        raise ValueError("EXP-036 gene axis must be the registered 19,091-element ordered list")
    genes = [str(value).strip().upper() for value in genes_raw]
    if not all(genes) or len(set(genes)) != len(genes):
        raise ValueError("EXP-036 gene axis contains blank or duplicate identifiers")

    pathway_config = config["pathway_resource"]
    if pathway_config.get("sha256") != gmt_hash or pathway_config.get("ordered_gene_axis_sha256") != axis_hash:
        raise ValueError("EXP-046 config hashes do not match pathway inputs")
    resource = parse_gmt(
        gmt_bytes.decode("utf-8"),
        resource_id=str(pathway_config["id"]),
        version=str(pathway_config["version"]),
    )
    if len(resource.pathways) != EXPECTED_PATHWAYS:
        raise ValueError("registered Hallmark resource must contain exactly 50 pathways")
    mapped_member_set = set().union(*map(set, resource.pathways.values()))
    mapped_genes = tuple(gene for gene in genes if gene in mapped_member_set)
    if len(mapped_genes) != EXPECTED_MAPPED_GENES:
        raise ValueError("Hallmark-to-EXP-036 mapping differs from the registered 4,366 genes")
    gene_index = {gene: index for index, gene in enumerate(mapped_genes)}
    pathways = tuple(resource.pathways)
    incidence = np.zeros((len(pathways), len(mapped_genes)), dtype=np.uint8)
    for row, name in enumerate(pathways):
        for gene in resource.pathways[name]:
            index = gene_index.get(gene)
            if index is not None:
                incidence[row, index] = 1
    if int(incidence.sum()) != EXPECTED_EDGE_COUNT:
        raise ValueError("Hallmark incidence edge count differs from the registered 7,299")
    if not np.all(incidence.sum(axis=0) > 0) or not np.all(incidence.sum(axis=1) > 0):
        raise ValueError("Hallmark mapping has an uncovered mapped gene or empty pathway")

    seeds = config["random_pathway_maps"].get("seeds")
    if seeds != [20260922, 20260923, 20260924]:
        raise ValueError("P-R seeds must remain the three preregistered EXP-046 seeds")
    source_hash = hashlib.sha256(gmt_bytes + b"\0" + axis_bytes).hexdigest()
    return list(mapped_genes), pathways, incidence, {
        "config": config_hash,
        "hallmark_gmt": gmt_hash,
        "gene_axis": axis_hash,
        "combined_pathway_gene_input": source_hash,
    }, {
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "pathway_path": gmt_path.relative_to(ROOT).as_posix(),
        "gene_axis_path": axis_path.relative_to(ROOT).as_posix(),
        "pathway_resource_id": str(pathway_config["id"]),
        "pathway_resource_version": str(pathway_config["version"]),
        "ordered_gene_axis_count": len(genes),
        "mapped_gene_count": len(mapped_genes),
        "pathway_count": len(pathways),
        "edge_count": int(incidence.sum()),
        "random_seeds": seeds,
    }


def _networkx_provenance() -> dict[str, str]:
    networkx = importlib.import_module("networkx")
    simplex_module = importlib.import_module("networkx.algorithms.flow.networksimplex")
    source_path = Path(simplex_module.__file__).resolve()
    return {
        "version": str(networkx.__version__),
        "network_simplex_source": source_path.as_posix(),
        "network_simplex_source_sha256": _sha256(source_path),
    }


def run() -> dict[str, Any]:
    output_dir = ROOT / "artifacts/experiments/EXP-046" / RUN_ID
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite pathway-control artifacts: {output_dir.as_posix()}")
    started = datetime.now(timezone.utc)
    started_perf = time.perf_counter()
    runtime = _require_wsl_conda()
    networkx_receipt = _networkx_provenance()
    genes, pathways, true_map, input_hashes, input_identity = _load_inputs()

    ordinary = build_ordinary_sparse_control(genes, n_slots=50, n_edges=EXPECTED_EDGE_COUNT)
    ordinary_repeat = build_ordinary_sparse_control(genes, n_slots=50, n_edges=EXPECTED_EDGE_COUNT)
    if ordinary.edge_sha256 != ordinary_repeat.edge_sha256:
        raise RuntimeError("repeat P-C construction produced a different edge hash")
    random_maps = []
    for seed in input_identity["random_seeds"]:
        result = randomize_bipartite_pathways(
            genes,
            pathways,
            true_map,
            seed=int(seed),
            min_accepted_swaps_per_edge=PR_MIN_ACCEPTED_PER_EDGE,
            min_edge_hamming_fraction=PR_MIN_EDGE_HAMMING_FRACTION,
            max_proposals_per_edge=PR_MAX_PROPOSALS_PER_EDGE,
        )
        random_maps.append(result)

    arrays = {"pc": ordinary.incidence}
    for result in random_maps:
        arrays[f"pr_{int(result.metadata['seed'])}"] = result.incidence
    output_dir.mkdir(parents=True, exist_ok=False)
    maps_path = output_dir / "maps.npz"
    np.savez_compressed(maps_path, **arrays)
    maps_file_hash = _sha256(maps_path)

    finished = datetime.now(timezone.utc)
    random_receipts = [
        {
            "seed": int(result.metadata["seed"]),
            "map_sha256": result.edge_sha256,
            "proposal_count": int(result.metadata["proposal_count"]),
            "accepted_swaps": int(result.metadata["accepted_swaps"]),
            "minimum_accepted_swaps": int(result.metadata["minimum_accepted_swaps"]),
            "maximum_proposals": int(result.metadata["maximum_proposals"]),
            "edge_hamming_fraction": float(result.metadata["edge_hamming_fraction"]),
        }
        for result in random_maps
    ]
    receipt = {
        "experiment_id": "EXP-046",
        "run_id": RUN_ID,
        "status": "LABEL_BLIND_PATHWAY_CONTROL_MAPS_CONSTRUCTED",
        "purpose": "construct frozen P-C and three P-R maps from Hallmark membership only",
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds": float(time.perf_counter() - started_perf),
        "runtime": runtime,
        "training_performed": False,
        "labels_read": False,
        "response_label_files_opened": [],
        "gpu_hours_charged": 0.0,
        "input_identity": input_identity,
        "input_sha256": input_hashes,
        "ordinary_control": {
            "map_sha256": ordinary.edge_sha256,
            "repeat_map_sha256": ordinary_repeat.edge_sha256,
            "repeat_construction_matches": True,
            "salt": PC_SALT,
            "slot_count": len(ordinary.pathway_names),
            "edge_count": len(ordinary.edges),
            "slot_targets": ordinary.metadata["slot_targets"],
            "minimum_cost_sha256_integer": ordinary.metadata["minimum_cost_sha256_integer"],
            "method": ordinary.method,
            "networkx": networkx_receipt,
        },
        "random_pathway_maps": random_receipts,
        "map_arrays_path": maps_path.relative_to(ROOT).as_posix(),
        "map_arrays_sha256": maps_file_hash,
        "map_array_shapes": {name: list(array.shape) for name, array in arrays.items()},
        "receipt_payload_sha256": _json_hash({
            "ordinary_map_sha256": ordinary.edge_sha256,
            "random_maps": random_receipts,
            "input_sha256": input_hashes,
            "map_arrays_sha256": maps_file_hash,
        }),
        "provenance": {
            "builder_script": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "builder_script_sha256": _sha256(Path(__file__).resolve()),
            "control_builder_source_sha256": _sha256(ROOT / "src/drug_screen/transfer/pathway_controls.py"),
        },
        "interpretation_limit": (
            "仅证明盲标签构图和边/轴合同；不含模型训练、患者药敏评估或通路机制证据。"
        ),
    }
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
