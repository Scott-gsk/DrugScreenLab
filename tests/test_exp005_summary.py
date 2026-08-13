from __future__ import annotations

import json

import pytest

from drug_screen.evaluation.exp005_summary import summarize_results


def _result(*, split: str, variant: str, spearman: float, digest: str = "same") -> dict[str, object]:
    return {
        "status": "COMPLETE",
        "variant": variant,
        "split": split,
        "seed": 2026,
        "budget": {"epochs": 3, "batch_size": 128, "max_rows_per_partition": 4096},
        "data_contract": {
            "cold_assertion": "zero context overlap asserted" if "cold_cell" in split else "zero drug overlap asserted",
            "partitions": {"train": {"sample_id_sha256": digest}, "test": {"sample_id_sha256": digest}},
        },
        "partitions": {"train": 4096, "test": 4096},
        "test_delta978": {"rows": 4096, "spearman_row_mean": spearman, "pearson_row_mean": spearman + 0.1, "mse": 1.0, "prediction_std": 0.2},
        "broad": {"line_metrics": {"eligible_line_count": 10, "macro_mean_spearman": 0.1, "macro_mean_ndcg10": 0.5, "macro_mean_top10_overlap_rate": 0.1}},
    }


def test_summary_flags_variant_with_pre_registered_gain_on_both_splits(tmp_path) -> None:
    for split in ("split_cold_cell_1", "split_cold_drug_1"):
        for variant, score in {"A": 0.10, "B": 0.13, "C": 0.11}.items():
            (tmp_path / f"{split}_{variant}.json").write_text(
                json.dumps(_result(split=split, variant=variant, spearman=score)), encoding="utf-8"
            )
    result = summarize_results(result_dir=tmp_path)
    assert result["decision"]["status"] == "PROMISING_FAST"
    assert result["decision"]["variants_triggering_medium"] == ["B"]


def test_summary_rejects_data_contract_mismatch(tmp_path) -> None:
    for split in ("split_cold_cell_1", "split_cold_drug_1"):
        for variant, score in {"A": 0.10, "B": 0.13, "C": 0.11}.items():
            digest = "different" if (split, variant) == ("split_cold_drug_1", "B") else "same"
            (tmp_path / f"{split}_{variant}.json").write_text(
                json.dumps(_result(split=split, variant=variant, spearman=score, digest=digest)), encoding="utf-8"
            )
    with pytest.raises(ValueError, match="data contract mismatch"):
        summarize_results(result_dir=tmp_path)
