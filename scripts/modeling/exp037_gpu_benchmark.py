"""EXP-037 训练前固定规模 GPU 吞吐与完整前反向链路检查；不产出科学结果。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from drug_screen.models.conditional_perturbation import SharedEffectInteraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-conditions", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EXP-037 benchmark")
    keys = pd.read_parquet(args.condition_dir / "conditions.parquet")
    delta = np.load(args.condition_dir / "condition_delta.npy", mmap_mode="r")
    control = np.load(args.input_dir / "condition_control.npy", mmap_mode="r")
    drug_table = pd.read_parquet(args.input_dir / "drugs.parquet")
    fingerprint = np.load(args.input_dir / "drug_morgan2048.npy", mmap_mode="r")
    exposure = pd.read_parquet(args.input_dir / "condition_exposure.parquet")
    n = min(args.n_conditions, len(keys))
    index = np.linspace(0, len(keys) - 1, n, dtype=int)
    drug_position = dict(zip(drug_table["pert_id"], drug_table["drug_index"]))
    x_control = np.asarray(control[index], dtype=np.float32)
    y = np.asarray(delta[index], dtype=np.float32)
    x_drug = np.asarray(fingerprint[[drug_position[x] for x in keys.iloc[index]["pert_id"]]],
                        dtype=np.float32)
    x_exposure = exposure.iloc[index][["log10_dose_um", "time_hours"]].to_numpy(np.float32)
    control_mean, control_std = x_control.mean(0), x_control.std(0)
    control_std[control_std <= 1e-8] = 1
    gene_scale = y.std(0); gene_scale[gene_scale <= 1e-8] = 1
    exposure_mean, exposure_std = x_exposure.mean(0), x_exposure.std(0)
    exposure_std[exposure_std <= 1e-8] = 1
    dataset = TensorDataset(
        torch.from_numpy((x_control - control_mean) / control_std), torch.from_numpy(x_drug),
        torch.from_numpy((x_exposure - exposure_mean) / exposure_std),
        torch.from_numpy(y / gene_scale))
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
    device = torch.device("cuda:0")
    receipts = []
    for use_state in (False, True):
        torch.manual_seed(20260922)
        model = SharedEffectInteraction(rank=32).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        start = time.perf_counter()
        final_loss = None
        for _ in range(args.epochs):
            for state, drug, exposed, target in loader:
                state, drug, exposed, target = [x.to(device, non_blocking=True)
                                                 for x in (state, drug, exposed, target)]
                optimizer.zero_grad(set_to_none=True)
                prediction = model(state, drug, exposed, use_state=use_state)
                loss = torch.nn.functional.huber_loss(prediction, target)
                loss.backward(); optimizer.step()
                final_loss = float(loss.detach().cpu())
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        receipts.append({"arm": "conditional" if use_state else "state_ablated",
                         "n_conditions": n, "epochs": args.epochs,
                         "elapsed_seconds": elapsed, "seconds_per_epoch": elapsed / args.epochs,
                         "final_training_huber": final_loss,
                         "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device))})
        del model, optimizer
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
    report = {"status": "PASS_ENGINEERING_BENCHMARK_NOT_SCIENTIFIC_RESULT",
              "device": torch.cuda.get_device_name(device), "torch": torch.__version__,
              "receipts": receipts}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
