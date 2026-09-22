"""CLI smoke coverage for EXP-017 common-space source preparation."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def _module():
    path = Path(__file__).parents[2] / "scripts" / "run_pdo_adaptation.py"
    spec = importlib.util.spec_from_file_location("run_pdo_adaptation_cli", path)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_common_source_cli_binds_registries_and_reads_target_gene_ids_only(tmp_path, monkeypatch):
    module = _module()
    source_registry, target_registry, admission = (tmp_path / name for name in ("source.json", "target.json", "admission.json"))
    for path in (source_registry, target_registry, admission): path.write_text("{}")
    observed = {}
    monkeypatch.setattr(module, "_target_receipt", lambda path, cohort, registry: {"cohort_id": cohort, "status": "DEVELOPMENT_ADMITTED"})
    def fake_train(source, target_genes, **kwargs):
        observed.update({"source": source, "genes": tuple(target_genes), **kwargs})
        kwargs["output_dir"].mkdir(); return {"target_labels_consumed": False}
    monkeypatch.setattr(module, "train_common_space_source_checkpoint", fake_train)
    def loader(cohort, **_):
        return SimpleNamespace(gene_ids=["g1", "g2"]) if cohort == "target" else SimpleNamespace(name="source")
    output = tmp_path / "out"
    assert module.main(["prepare-common-source", "--source-cohort", "source", "--source-registry", str(source_registry),
                        "--target-cohort", "target", "--target-registry", str(target_registry), "--target-admission", str(admission),
                        "--data-root", str(tmp_path), "--output-dir", str(output), "--seed", "7", "--device", "cpu"], loader=loader) == 0
    assert observed["genes"] == ("g1", "g2") and observed["seed"] == 7
    saved = json.loads((output / "cli_provenance.json").read_text())
    assert saved["target_response_labels_consumed"] is False
