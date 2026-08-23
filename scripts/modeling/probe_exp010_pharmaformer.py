"""One-batch source timing/memory probe; never writes a model checkpoint."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from drug_screen.foundation.pharmaformer_adapter import (  # noqa: E402
    PharmaFormerConfig, TrainOnlyGeneTransform, TrainOnlySmilesEncoder, build_model, load_upstream_module)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--bpe-codes", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    import torch
    arrays = np.load(args.arrays, allow_pickle=False)
    batch = min(args.batch_size, len(arrays["train_y"]))
    transform = TrainOnlyGeneTransform().fit(arrays["train_gene"], arrays["train_sample_id"].astype(str))
    encoder = TrainOnlySmilesEncoder().fit(arrays["train_smiles"].astype(str), arrays["train_drug_id"].astype(str),
                                           bpe_codes_path=args.bpe_codes)
    gene = torch.from_numpy(transform.transform(arrays["train_gene"][:batch])).to(args.device)
    drug = torch.from_numpy(encoder.transform(arrays["train_smiles"][:batch].astype(str))).to(args.device)
    target = torch.from_numpy(arrays["train_y"][:batch].astype(np.float32)).reshape(-1, 1).to(args.device)
    upstream = load_upstream_module(args.upstream)
    config = PharmaFormerConfig(gene_input_size=8192, gene_hidden_size=8192, drug_hidden_size=256,
                                feature_dim=128, nhead=8, num_layers=3, dim_feedforward=2048, dropout=.1)
    torch.cuda.reset_peak_memory_stats(args.device)
    start = time.perf_counter()
    model = build_model(upstream, config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    torch.cuda.synchronize(args.device)
    setup_seconds = time.perf_counter() - start
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(gene, drug), target)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(args.device)
    step_seconds = time.perf_counter() - start
    result = {"experiment_id": "EXP-010", "probe_only": True, "batch_size": batch,
              "device": args.device, "setup_seconds": setup_seconds, "one_train_step_seconds": step_seconds,
              "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(args.device)), "loss": float(loss.detach()),
              "checkpoint_written": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
