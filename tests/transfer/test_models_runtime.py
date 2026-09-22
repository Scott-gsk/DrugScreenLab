import numpy as np
import pytest
import torch

from drug_screen.transfer.models import DualEncoderPairModel


def _training_data():
    rng = np.random.default_rng(31)
    sample = rng.normal(size=(9, 3))
    drug = rng.normal(size=(9, 4))
    labels = np.array([.9, .1, .5, .7, .2, .4, .8, .3, .6])
    pdo_ids = np.array(["b", "a", "b", "c", "a", "c", "b", "c", "c"])
    return sample, drug, labels, pdo_ids


def test_group_minibatches_keep_each_pdo_together_and_are_seeded():
    sample, drug, labels, pdo_ids = _training_data()
    first = DualEncoderPairModel(epochs=2, seed=9, batch_groups=1)
    first._ensure_network(sample.shape[1], drug.shape[1])
    seen_batches = []
    handle = first.network.register_forward_pre_hook(lambda _module, args: seen_batches.append(args[0].detach().cpu().numpy()))
    first.fit(sample, drug, labels, pdo_ids=pdo_ids)
    handle.remove()
    second = DualEncoderPairModel(epochs=2, seed=9, batch_groups=1).fit(sample, drug, labels, pdo_ids=pdo_ids)
    assert np.allclose(first.predict(sample, drug), second.predict(sample, drug))

    for batch in seen_batches:
        row_indices = [np.flatnonzero(np.all(np.isclose(sample, row), axis=1))[0] for row in batch]
        assert len(set(pdo_ids[row_indices])) == 1

    # Cached rank-pair indices are relative to each mini-batch.  Every pair
    # must therefore refer to the same original PDO even with interleaved rows.
    entries = first._group_rank_pairs(labels, pdo_ids)
    for indices, left, right in entries:
        assert np.all(pdo_ids[indices[left]] == pdo_ids[indices[right]])


def test_capped_ranking_pairs_sample_all_drug_pairs_deterministically():
    labels = np.arange(10, 0, -1, dtype=float)
    pdo_ids = np.repeat("pdo-1", len(labels))
    first = DualEncoderPairModel.ranking_pairs(labels, pdo_ids, max_pairs_per_pdo=5, seed=37)
    second = DualEncoderPairModel.ranking_pairs(labels, pdo_ids, max_pairs_per_pdo=5, seed=37)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert len(first[0]) == 5
    assert np.all(labels[first[0]] > labels[first[1]])
    # The previous row-order cap selected only pairs whose left drug was row 0.
    assert not np.all(first[0] == 0)


def test_cpu_runtime_covers_prediction_and_explanation():
    sample, drug, labels, pdo_ids = _training_data()
    model = DualEncoderPairModel(epochs=2, seed=3, device="cpu", batch_groups=2).fit(sample, drug, labels, pdo_ids=pdo_ids)
    result = model.explain(sample[:2], drug[:2], steps=4)
    assert model.device.type == "cpu"
    assert model.predict(sample[:2], drug[:2]).shape == (2,)
    assert result["sample_attributions"].shape == (2, 3)
    assert result["drug_attributions"].shape == (2, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_runtime_covers_prediction_and_explanation():
    sample, drug, labels, pdo_ids = _training_data()
    model = DualEncoderPairModel(epochs=2, seed=3, device="cuda", batch_groups=2).fit(sample, drug, labels, pdo_ids=pdo_ids)
    assert model.network is not None and next(model.network.parameters()).device.type == "cuda"
    assert model.predict(sample[:2], drug[:2]).shape == (2,)
    assert model.explain(sample[:1], drug[:1], steps=4)["prediction"].shape == (1,)
