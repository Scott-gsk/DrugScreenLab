"""Run a real-input, CPU-only EXP-046 pathway interface smoke without labels."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import platform
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from drug_screen.models.pharmaformer_pathway import PharmaFormerWithPathwayBranch
from drug_screen.transfer.pathways import parse_gmt


RUN_ID = "20260923_pathway_real_inputs_smoke_v1"
EXPECTED_GMT_SHA256 = "8ff1f036f0988c99d42cf9e5ea24d355f10b8ece5e0d4f6edf26a0458f480780"
EXPECTED_GENE_AXIS_SHA256 = "bf023d7b110bd820b3b41c67ab9ce5ea6b2381ab5753c3ec6440ee2a61340610"
EXPECTED_CHECKPOINT_SHA256 = "1dbe98a2f52c4affdcee091cf011f3e52e29b51992c9bb3d57a60b08fa464de6"
EXPECTED_INPUT_MANIFEST_SHA256 = "c6f9c3349b0a8598280ba099381b781c68ca5599f6f3b7e749090a0e46460b72"
EXPECTED_INPUT_VERIFICATION_SHA256 = "a3100f14733381c727a5f11c32367d52675456549ca51e7607065bef560b7e1c"
EXPECTED_PATHWAYS = 50
EXPECTED_GENES = 19091
EXPECTED_UNIQUE_MEMBERS = 4384
EXPECTED_MATCHED_GENES = 4366


class Snapshot:
    def __init__(self, path: Path, content: bytes):
        self.path = path
        self.content = content
        self.sha256 = hashlib.sha256(content).hexdigest()


def snapshot(path: Path) -> Snapshot:
    return Snapshot(path, path.read_bytes())


def _json(snapshot_value: Snapshot) -> Any:
    return json.loads(snapshot_value.content.decode("utf-8"))


def _parquet(snapshot_value: Snapshot) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(snapshot_value.content))


def _hash_values(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_environment() -> dict[str, Any]:
    release = platform.release().lower()
    if (platform.system() != "Linux"
            or not ("microsoft" in release or os.environ.get("WSL_DISTRO_NAME"))):
        raise RuntimeError("this smoke must run inside WSL2 Linux")
    if os.environ.get("CONDA_DEFAULT_ENV") != "drugscreening-gpu":
        raise RuntimeError("this smoke requires Conda environment drugscreening-gpu")
    conda_prefix = Path(os.environ.get("CONDA_PREFIX", "")).resolve()
    python_path = Path(sys.executable).resolve()
    if not conda_prefix.is_dir() or conda_prefix not in python_path.parents:
        raise RuntimeError("Python executable is not inside the active drugscreening-gpu prefix")
    return {
        "platform": platform.platform(),
        "platform_release": platform.release(),
        "conda_env": os.environ["CONDA_DEFAULT_ENV"],
        "conda_prefix": conda_prefix.as_posix(),
        "python_executable": python_path.as_posix(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "device": "cpu",
    }


def _load_official_model_source(source: Snapshot) -> types.ModuleType:
    module = types.ModuleType("pharmaformer_official_pretrain_exp046_smoke")
    module.__file__ = source.path.as_posix()
    code = compile(source.content.decode("utf-8"), source.path.as_posix(), "exec")
    exec(code, module.__dict__)
    if not hasattr(module, "CombinedModel"):
        raise ValueError("official PharmaFormer source does not define CombinedModel")
    return module


def _pathway_incidence(resource: Any, genes: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    normalized_genes = [str(gene).strip().upper() for gene in genes]
    if (len(normalized_genes) != EXPECTED_GENES
            or len(set(normalized_genes)) != len(normalized_genes)):
        raise ValueError("EXP-036 gene axis must contain 19,091 unique ordered genes")
    positions = {gene: index for index, gene in enumerate(normalized_genes)}
    pathway_names = tuple(resource.pathways)
    if len(pathway_names) != EXPECTED_PATHWAYS:
        raise ValueError("Hallmark GMT does not contain the registered 50 pathways")

    matrix = np.zeros((len(pathway_names), len(normalized_genes)), dtype=np.float32)
    resource_members: set[str] = set()
    matched_members: set[str] = set()
    pathway_coverage = []
    for row, pathway_name in enumerate(pathway_names):
        members = tuple(resource.pathways[pathway_name])
        resource_members.update(members)
        matched = [gene for gene in members if gene in positions]
        matched_members.update(matched)
        if not matched:
            raise ValueError(f"Hallmark pathway {pathway_name!r} has no genes on EXP-036 axis")
        matrix[row, [positions[gene] for gene in matched]] = 1.0
        pathway_coverage.append({
            "pathway": pathway_name,
            "resource_members": len(members),
            "matched_axis_genes": len(matched),
            "coverage_fraction": len(matched) / len(members),
        })

    if len(resource_members) != EXPECTED_UNIQUE_MEMBERS:
        raise ValueError("Hallmark GMT unique member count differs from the registered resource")
    if len(matched_members) != EXPECTED_MATCHED_GENES:
        raise ValueError("Hallmark-to-EXP-036 unique gene coverage differs from the registered map")
    edge_count = int(matrix.sum())
    coverage_values = [item["coverage_fraction"] for item in pathway_coverage]
    return matrix, {
        "n_pathways": int(matrix.shape[0]),
        "n_genes": int(matrix.shape[1]),
        "n_unique_resource_members": int(len(resource_members)),
        "n_unique_matched_axis_genes": int(len(matched_members)),
        "n_membership_edges": edge_count,
        "nonempty_pathways": int(np.count_nonzero(matrix.sum(axis=1))),
        "min_pathway_gene_coverage": float(min(coverage_values)),
        "max_pathway_gene_coverage": float(max(coverage_values)),
        "pathway_coverage": pathway_coverage,
        "pathway_names": list(pathway_names),
    }


def run_smoke() -> dict[str, Any]:
    output_dir = ROOT / "artifacts" / "experiments" / "EXP-046" / RUN_ID
    output_path = output_dir / "receipt.json"
    if output_dir.exists() or output_path.exists():
        raise FileExistsError(f"refusing to overwrite the EXP-046 smoke output: {output_dir}")

    started = datetime.now(timezone.utc)
    started_perf = time.perf_counter()
    environment = _runtime_environment()
    torch.set_num_threads(1)

    pathway_path = ROOT / "data/raw/exp013/msigdb/h.all.v2023.2.Hs.symbols.gmt"
    contract_dir = ROOT / "data/processed/dual_route/EXP-036/direct_response_contract_v1"
    input_dir = ROOT / "data/processed/dual_route/EXP-036/pharmaformer_inputs_source_fold0_v1"
    checkpoint_path = ROOT / "data/artifacts/EXP-011/source_cv/fold0/M2_fold0.pt"
    official_path = ROOT / "data/external/exp010/pharmaformer/repo/pretrain.py"
    wrapper_path = ROOT / "src/drug_screen/models/pharmaformer_pathway.py"

    input_paths = {
        "hallmark_gmt": pathway_path,
        "gene_axis": contract_dir / "genes.json",
        "contract_manifest": contract_dir / "manifest.json",
        "patient_expression": input_dir / "patient_expression.npy",
        "patient_axis": input_dir / "patient_axis.json",
        "drug_ordinal128": input_dir / "drug_ordinal128.npy",
        "drug_axis": input_dir / "drug_axis.parquet",
        "input_manifest": input_dir / "manifest.json",
        "input_verification": input_dir / "verification.json",
        "source_checkpoint": checkpoint_path,
        "official_model_source": official_path,
        "pathway_wrapper_source": wrapper_path,
        "smoke_script": Path(__file__).resolve(),
    }
    missing = [path.as_posix() for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"EXP-046 real-input smoke assets are missing: {missing}")
    snapshots = {name: snapshot(path) for name, path in input_paths.items()}
    if snapshots["hallmark_gmt"].sha256 != EXPECTED_GMT_SHA256:
        raise ValueError("Hallmark GMT checksum differs from the EXP-046 contract")
    if snapshots["gene_axis"].sha256 != EXPECTED_GENE_AXIS_SHA256:
        raise ValueError("EXP-036 ordered gene-axis checksum differs from the EXP-046 contract")
    if snapshots["source_checkpoint"].sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("M2 fold0 checkpoint checksum differs from the approved source asset")
    if snapshots["input_manifest"].sha256 != EXPECTED_INPUT_MANIFEST_SHA256:
        raise ValueError("EXP-036 materialized input manifest differs from its registered checksum")
    if snapshots["input_verification"].sha256 != EXPECTED_INPUT_VERIFICATION_SHA256:
        raise ValueError("EXP-036 materialized input verification receipt differs from its registered checksum")

    input_manifest = _json(snapshots["input_manifest"])
    input_verification = _json(snapshots["input_verification"])
    if (input_manifest.get("source_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256
            or input_verification.get("status") != "PASS"
            or input_verification.get("input_manifest_sha256") != snapshots["input_manifest"].sha256
            or input_verification.get("source_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256):
        raise ValueError("EXP-036 patient input assets lack a passing source-checkpoint verification")
    file_receipt = input_manifest.get("files_sha256", {})
    for name in ("patient_expression", "patient_axis", "drug_ordinal128", "drug_axis"):
        if file_receipt.get(input_paths[name].name) != snapshots[name].sha256:
            raise ValueError(f"EXP-036 input manifest does not bind {input_paths[name].name}")

    genes = _json(snapshots["gene_axis"])
    resource = parse_gmt(
        snapshots["hallmark_gmt"].content.decode("utf-8"),
        resource_id="MSigDB_Hallmark", version="h.all.v2023.2.Hs.symbols",
    )
    incidence, map_receipt = _pathway_incidence(resource, genes)
    expression = np.load(io.BytesIO(snapshots["patient_expression"].content), allow_pickle=False)
    patient_axis = _json(snapshots["patient_axis"])
    drug_codes = np.load(io.BytesIO(snapshots["drug_ordinal128"].content), allow_pickle=False)
    drug_axis = _parquet(snapshots["drug_axis"])
    if (expression.ndim != 2 or expression.shape != (len(patient_axis), EXPECTED_GENES)
            or not np.isfinite(expression).all()):
        raise ValueError("materialized patient expression is not finite or does not match the ordered gene axis")
    if len(patient_axis) != len(set(map(str, patient_axis))):
        raise ValueError("materialized patient axis contains duplicate identifiers")
    if (drug_codes.ndim != 2 or drug_codes.shape != (len(drug_axis), 128)
            or "drug_id" not in drug_axis
            or drug_axis["drug_id"].duplicated().any()
            or not np.isfinite(drug_codes).all()):
        raise ValueError("materialized drug codes and drug axis are malformed or misaligned")
    if list(drug_axis["drug_id"].astype(str)) != input_manifest.get("drug_axis_order"):
        manifest_order = input_manifest.get("drug_axis_order")
        if manifest_order is not None:
            raise ValueError("materialized drug axis differs from its manifest order")

    distinct_drug_rows: list[int] = []
    for index, code in enumerate(drug_codes):
        if not any(np.array_equal(code, drug_codes[prior]) for prior in distinct_drug_rows):
            distinct_drug_rows.append(index)
        if len(distinct_drug_rows) == 2:
            break
    if len(distinct_drug_rows) < 2:
        raise ValueError("at least two distinct real EXP-036 drug encodings are required")
    selected_drug_ids = [str(drug_axis.iloc[index]["drug_id"]) for index in distinct_drug_rows]

    checkpoint = torch.load(io.BytesIO(snapshots["source_checkpoint"].content),
                            map_location="cpu", weights_only=False)
    if (checkpoint.get("variant") != "M2"
            or checkpoint.get("model_variant") != "CombinedModel"
            or checkpoint.get("model_config", {}).get("gene_input_size") != EXPECTED_GENES):
        raise ValueError("M2 fold0 checkpoint architecture does not match EXP-036")
    official = _load_official_model_source(snapshots["official_model_source"])
    config = checkpoint["model_config"]
    backbone = official.CombinedModel(
        gene_input_size=config["gene_input_size"],
        gene_hidden_size=config["gene_hidden_size"],
        drug_hidden_size=config["drug_hidden_size"],
        feature_dim=config["feature_dim"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
    )
    backbone.load_state_dict(checkpoint["model_state_dict"], strict=True)
    backbone.eval()
    if backbone.feature_extractor.smiles_fc.in_features != 128:
        raise ValueError("official M2 drug encoder does not accept 128-position codes")

    torch.manual_seed(20260923)
    model = PharmaFormerWithPathwayBranch(
        backbone, torch.from_numpy(incidence), rank=8,
        zero_initialize=True, pathway_names=map_receipt["pathway_names"],
    )
    model.eval()
    patient_id = str(patient_axis[0])
    expression_row = torch.from_numpy(np.asarray(expression[0:1], dtype=np.float32).copy())
    selected_codes = torch.from_numpy(
        np.asarray(drug_codes[distinct_drug_rows], dtype=np.float32).copy()
    )
    gene_batch = expression_row.expand(2, -1).contiguous()
    with torch.inference_mode():
        baseline_prediction = backbone(gene_batch, selected_codes)
        zero_components = model.forward_components(gene_batch, selected_codes)

    if (baseline_prediction.shape != (2, 1)
            or not bool(torch.isfinite(baseline_prediction).all())
            or not bool(torch.isfinite(zero_components.prediction).all())
            or not torch.equal(zero_components.prediction, baseline_prediction)):
        raise ValueError("zero-initialized pathway branch does not exactly preserve official predictions")
    zero_reconstructed = (
        zero_components.base_prediction
        + zero_components.pathway_contributions.sum(dim=1, keepdim=True)
    )
    if not torch.equal(zero_reconstructed, zero_components.prediction):
        raise ValueError("zero-initialized pathway contributions do not reconstruct the final prediction")
    if int(torch.count_nonzero(zero_components.pathway_contributions)) != 0:
        raise ValueError("zero-initialized pathway branch must start with zero contributions")
    normalized_rows = model.pathway_gene_weights.sum(dim=1)
    normalized_row_error = float((normalized_rows - 1.0).abs().max())
    map_receipt["normalized_row_sum_max_abs_error"] = normalized_row_error
    if normalized_row_error > 1e-6:
        raise ValueError("registered pathway incidence rows did not normalize to one")

    # A deterministic, non-trained nonzero branch checks that drug encoding can
    # change per-pathway contributions; the initialized branch is restored after.
    initial_factors = model.branch.pathway_factors.detach().clone()
    generator = torch.Generator(device="cpu").manual_seed(20260923)
    with torch.no_grad():
        probe_factors = torch.randn(model.branch.pathway_factors.shape, generator=generator) * 0.02
        model.branch.pathway_factors.copy_(probe_factors)
    with torch.inference_mode():
        probe_components = model.forward_components(gene_batch, selected_codes)
    contribution_change = (
        probe_components.pathway_contributions[0]
        - probe_components.pathway_contributions[1]
    ).abs()
    if (not bool(torch.isfinite(probe_components.pathway_contributions).all())
            or float(contribution_change.max()) <= 0.0):
        raise ValueError("drug-conditioned pathway contribution probe did not vary across real drugs")
    with torch.no_grad():
        model.branch.pathway_factors.copy_(initial_factors)

    finished = datetime.now(timezone.utc)
    output_payload = {
        "patient_id": patient_id,
        "drug_ids": selected_drug_ids,
        "baseline_prediction": baseline_prediction.reshape(-1).tolist(),
        "zero_branch_prediction": zero_components.prediction.reshape(-1).tolist(),
        "zero_branch_contributions": zero_components.pathway_contributions.tolist(),
        "drug_condition_probe_contributions": probe_components.pathway_contributions.tolist(),
    }
    receipt = {
        "experiment_id": "EXP-046",
        "run_id": RUN_ID,
        "status": "REAL_INPUT_PATHWAY_INTERFACE_SMOKE_PASS_ENGINEERING_ONLY",
        "purpose": "one-row real-input CPU interface smoke; no training or scientific evaluation",
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds": float(time.perf_counter() - started_perf),
        "runtime": environment,
        "training_performed": False,
        "labels_read": False,
        "response_label_files_opened": [],
        "input_identity": {
            "patient_id": patient_id,
            "patient_row_index": 0,
            "n_patient_expression_genes": int(expression.shape[1]),
            "expression_finite": True,
            "n_drugs_in_frozen_axis": int(len(drug_axis)),
            "drug_feature_width": int(drug_codes.shape[1]),
            "selected_drugs": [
                {
                    "drug_id": selected_drug_ids[position],
                    "drug_axis_row": int(row_index),
                    "encoding_sha256": hashlib.sha256(
                        np.asarray(drug_codes[row_index]).tobytes(order="C")
                    ).hexdigest(),
                }
                for position, row_index in enumerate(distinct_drug_rows)
            ],
        },
        "pathway_mapping": map_receipt,
        "model_checks": {
            "backbone_variant": checkpoint["variant"],
            "backbone_model_variant": checkpoint["model_variant"],
            "strict_state_dict_load": True,
            "n_backbone_parameters": int(sum(parameter.numel() for parameter in backbone.parameters())),
            "branch_rank": 8,
            "branch_parameters_trainable": bool(
                model.branch.pathway_factors.requires_grad
                and model.branch.drug_projection.weight.requires_grad
            ),
            "zero_initialized": bool(torch.equal(initial_factors, torch.zeros_like(initial_factors))),
            "baseline_output_shape": list(baseline_prediction.shape),
            "zero_branch_prediction_matches_baseline_exactly": True,
            "zero_branch_max_abs_prediction_error": float(
                (zero_components.prediction - baseline_prediction).abs().max()
            ),
            "zero_branch_nonzero_contribution_count": int(
                torch.count_nonzero(zero_components.pathway_contributions)
            ),
            "additive_reconstruction_exact": True,
            "additive_reconstruction_max_abs_error": float(
                (zero_reconstructed - zero_components.prediction).abs().max()
            ),
            "drug_condition_probe_seed": 20260923,
            "drug_condition_probe_is_not_training": True,
            "drug_condition_probe_max_abs_contribution_change": float(contribution_change.max()),
            "drug_condition_probe_changes_pathway_contributions": True,
        },
        "input_sha256": {name: value.sha256 for name, value in snapshots.items()},
        "smoke_output_sha256": _hash_values(output_payload),
        "provenance": {
            "script_path": Path(__file__).resolve().as_posix(),
            "script_sha256": snapshots["smoke_script"].sha256,
            "official_model_source_sha256": snapshots["official_model_source"].sha256,
            "pathway_wrapper_source_sha256": snapshots["pathway_wrapper_source"].sha256,
        },
        "interpretation_limit": (
            "工程接口验证；零初始化保持骨干预测，非零贡献诊断仅验证药物条件接口。"
            "不包含训练、响应标签、预测性能或通路机制结论。"
        ),
    }

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite the EXP-046 smoke output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return receipt


def main() -> int:
    result = run_smoke()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
