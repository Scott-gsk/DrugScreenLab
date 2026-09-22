from __future__ import annotations
import argparse, json
from pathlib import Path
from drug_screen.transfer.auxiliary_training import run_nature_auxiliary
from drug_screen.transfer.transfer_experiment import run_transfer_experiment

p = argparse.ArgumentParser(); p.add_argument("--config", type=Path, required=True); p.add_argument("--source-output", type=Path, required=True)
p.add_argument("--data-root", type=Path, required=True); p.add_argument("--source-registry", type=Path, required=True); p.add_argument("--nature-data", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
p.add_argument("--variant", choices=("N0", "N1", "N3"))
p.add_argument("--steps", type=int); p.add_argument("--auxiliary-weight", type=float, choices=(0.0, .01, .1)); p.add_argument("--permutation-seed", type=int)
p.add_argument("--disable-dependency", action="store_true")
p.add_argument("--target-admission", type=Path); p.add_argument("--target-registry", type=Path); p.add_argument("--target-cohort")
p.add_argument("--target-output-dir", type=Path); p.add_argument("--target-seed", type=int, default=20260913)
p.add_argument("--aggregate-dir", type=Path, help="write compact bounded-development summary from named target outputs")

def _aggregate(directory: Path, target_seed: int) -> dict:
    names = {
        "N0": "zhu_n0_200step_seed20260913", "N1_w001": "zhu_n1_w001_200step_seed20260913",
        "N1_w01": "zhu_n1_w01_200step_seed20260913", "N3_w001": "zhu_n3_w001_200step_seed20260913",
        "N3_w01": "zhu_n3_w01_200step_seed20260913", "N3_w01_dependency_off": "zhu_n3_w01_dependency_off_200step_seed20260913",
        "N3_w01_dependency_permuted": "zhu_n3_w01_permuted_200step_seed20260913",
    }
    result = {"status": "BOUNDED_DEVELOPMENT_SINGLE_SEED", "seed": target_seed, "metric": "strict_7_known_drug_patient_macro_NDCG_at_3", "rows": []}
    for label, child in names.items():
        metrics_path = directory / child / "metrics.json"
        if not metrics_path.is_file(): continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]
        m3 = metrics[f"M3_seed_{target_seed}"]["known_primary"]["patient_average"]
        row = {"variant": label, "M3": m3["ndcg_at_3"], "M1_source_scaled_comparator": m3["baseline_ndcg_at_3"]}
        # Every candidate metric is paired to M1_source_scaled in EXP-015.
        # Extract standalone B0/M1/M2/M1SS values explicitly rather than
        # treating that comparison field as the drug-mean B0 result.
        for key, out_name in (("B0", "B0_drug_mean"), ("M1", "M1"), ("M1_source_scaled", "M1_source_scaled"), ("M2", "M2")):
            row[out_name] = metrics[f"{key}_seed_{target_seed}"]["known_primary"]["patient_average"]["ndcg_at_3"]
        result["rows"].append(row)
    return result
def main() -> int:
    a = p.parse_args(); config = json.loads(a.config.read_text())
    if a.aggregate_dir is not None:
        aggregate = _aggregate(a.aggregate_dir, a.target_seed)
        (a.aggregate_dir / "bounded_development_aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        lines = ["# EXP-018 bounded-development aggregate", "", "Single seed, previously viewed Zhu development cohort; not external evidence.", "", "| Variant | B0 drug mean | M1 | M1 source-scaled | M2 | M3 |", "|---|---:|---:|---:|---:|---:|"]
        for row in aggregate["rows"]:
            lines.append("| {variant} | {B0_drug_mean:.4f} | {M1:.4f} | {M1_source_scaled:.4f} | {M2:.4f} | {M3:.4f} |".format(**row))
        lines += ["", "`M1_source_scaled_comparator` is the paired baseline encoded in existing candidate reports; it is distinct from B0 drug mean."]
        (a.aggregate_dir / "bounded_development_aggregate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps(aggregate, indent=2)); return 0
    if a.variant is not None: config["variant"] = a.variant
    if a.steps is not None: config["steps"] = a.steps
    if a.auxiliary_weight is not None: config["auxiliary_weight"] = a.auxiliary_weight
    if a.permutation_seed is not None: config["permutation_seed"] = a.permutation_seed
    if a.disable_dependency: config["dependency_enabled"] = False
    result = run_nature_auxiliary(config, source_output=a.source_output, data_root=a.data_root, source_registry=a.source_registry, nature_data=a.nature_data, output_dir=a.output_dir)
    target_args = (a.target_admission, a.target_registry, a.target_cohort, a.target_output_dir)
    if any(value is not None for value in target_args):
        if not all(value is not None for value in target_args): p.error("all --target-* arguments are required together")
        target_config = {"cohort_id": a.target_cohort, "seeds": [a.target_seed], "device": config.get("device", "cpu"), "bootstrap_resamples": 2000,
                         "scope": "EXP018_bounded_development_single_seed_not_model_selection"}
        result["pdo_development"] = run_transfer_experiment(target_config, target_admission=a.target_admission, source_output=a.output_dir, data_root=a.data_root, output_dir=a.target_output_dir, registry_path=a.target_registry)
    print(json.dumps(result, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
