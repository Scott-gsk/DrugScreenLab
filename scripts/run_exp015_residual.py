from pathlib import Path
import argparse
from drug_screen.transfer.residual_experiment import run_residual_diagnostic


parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, default=Path("data/artifacts/EXP-015/zhu_residual_v1"))
args = parser.parse_args()
run_residual_diagnostic(source_dir=Path("data/artifacts/EXP-015/source_seed20260912"),data_root=Path("data"),registry_path=Path("data/registry/exp015_target_v3.json"),target_admission=Path("data/artifacts/EXP-015/target_admission_v3.json"),existing_oof=Path("data/artifacts/EXP-015/zhu_transfer_v3/outer_oof_predictions.parquet"),output_dir=args.output_dir)
