import numpy as np
import pytest

from drug_screen.modeling.genetic_transfer import (
    UnifiedResponseModel,
    UnifiedResponseRecord,
    group_atomic_subset,
    fit_transfer_probe,
)


def test_unified_response_model_keeps_modality_and_direction_explicit():
    model = UnifiedResponseModel(chemical_dim=4, hidden_dim=8)
    context = np.zeros((2, 978), dtype=np.float32)
    perturbagen = np.ones((2, 4), dtype=np.float32)
    dose_time = np.ones((2, 2), dtype=np.float32)
    modality = np.asarray([0, 1], dtype=np.int64)
    direction = np.asarray([0, 1], dtype=np.int64)

    output = model(
        context,
        perturbagen,
        dose_time,
        modality,
        direction,
    )

    assert output.shape == (2, 978)
    assert not np.allclose(output.detach().numpy()[0], output.detach().numpy()[1])


def test_unified_record_requires_explicit_genetic_direction():
    with pytest.raises(ValueError, match="perturbation_direction"):
        UnifiedResponseRecord.from_mapping(
            {
                "sample_id": "s1",
                "treatment_group_id": "g1",
                "perturbagen_id": "gene-1",
                "modality": "genetic",
                "context_id": "A375",
                "dose_um": 1.0,
                "time_h": 96.0,
                "split": "train",
                "treatment_cache_row": 1,
                "control_cache_row": 0,
                "perturbagen_feature_row": 0,
            }
        )


def test_group_atomic_subset_is_deterministic_and_does_not_split_groups():
    rows = tuple(
        UnifiedResponseRecord.from_mapping(
            {
                "sample_id": f"s{index}",
                "treatment_group_id": group,
                "perturbagen_id": "drug-1",
                "modality": "chemical",
                "perturbation_direction": "small_molecule",
                "context_id": "A375",
                "dose_um": 10.0,
                "time_h": 6.0,
                "split": "train",
                "treatment_cache_row": index + 1,
                "control_cache_row": 0,
                "perturbagen_feature_row": 0,
            }
        )
        for index, group in enumerate(("g1", "g1", "g2", "g2", "g3"))
    )
    selected_a = group_atomic_subset(rows, fraction=0.5, seed=7)
    selected_b = group_atomic_subset(rows, fraction=0.5, seed=7)

    assert tuple(row.sample_id for row in selected_a) == tuple(row.sample_id for row in selected_b)
    selected_groups = {row.treatment_group_id for row in selected_a}
    assert all(
        all(row.treatment_group_id in selected_groups for row in rows if row.treatment_group_id == group)
        for group in selected_groups
    )
    assert len(selected_groups) == 2


def test_transfer_probe_reports_chemical_only_and_genetic_pretraining():
    rng = np.random.default_rng(4)
    cache = rng.normal(size=(12, 978)).astype(np.float32)
    features = rng.normal(size=(3, 4)).astype(np.float32)

    def row(index, group, modality, direction, feature_row, split):
        return UnifiedResponseRecord.from_mapping(
            {
                "sample_id": f"{modality}-{index}-{split}",
                "treatment_group_id": group,
                "perturbagen_id": f"{modality}-{feature_row}",
                "modality": modality,
                "perturbation_direction": direction,
                "context_id": "A375",
                "dose_um": 10.0 if modality == "chemical" else 1.0,
                "time_h": 6.0 if modality == "chemical" else 96.0,
                "split": split,
                "treatment_cache_row": index,
                "control_cache_row": 0,
                "perturbagen_feature_row": feature_row,
            }
        )

    genetic = tuple(row(index, f"gg{index}", "genetic", "knockdown", index % 3, "train") for index in range(1, 5))
    chemical_train = tuple(row(index + 4, f"cg{index}", "chemical", "small_molecule", index % 3, "train") for index in range(4))
    chemical_test = tuple(row(index + 8, f"ct{index}", "chemical", "small_molecule", index % 3, "test") for index in range(4))

    result = fit_transfer_probe(
        genetic_records=genetic,
        chemical_train_records=chemical_train,
        chemical_test_records=chemical_test,
        cache=cache,
        perturbagen_features=features,
        chemical_fraction=0.5,
        genetic_epochs=1,
        chemical_epochs=1,
        hidden_dim=8,
        seed=3,
    )

    assert result["chemical_fraction"] == 0.5
    assert set(result["models"]) == {"chemical_only", "genetic_pretrain_then_chemical"}
    assert result["chemical_train_group_count"] == 2
    assert result["chemical_test_group_count"] == 4
    assert set(result["models"]["chemical_only"]["test_metrics"]) >= {"spearman", "pearson", "direction_accuracy"}
