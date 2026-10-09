import hashlib

import pandas as pd
import torch

from scripts.modeling.diagnose_xpert_training_reliability import (
    install_cpu_attention_layout_adapter,
    select_train32,
)


def test_select_train32_matches_frozen_hash_sampling_contract():
    rows = []
    drugs = [f"drug-{index}" for index in range(5)]
    for drug in drugs:
        for condition in range(12):
            rows.append({"full_condition_key": f"{drug}|condition-{condition:02d}",
                         "full_inchikey": drug})
    frame = pd.DataFrame(rows)

    selected = select_train32(frame)

    expected = []
    for rank, drug in enumerate(sorted(drugs)):
        keys = [row["full_condition_key"] for row in rows if row["full_inchikey"] == drug]
        ordered = sorted(keys, key=lambda key: (hashlib.sha256(f"20261006|{key}".encode()).hexdigest(), key))
        expected.extend(ordered[:7 if rank < 2 else 6])
    expected.sort()

    assert len(selected) == 32
    assert selected.full_condition_key.tolist() == expected
    assert selected.full_inchikey.value_counts().sort_index().to_dict() == {
        "drug-0": 7, "drug-1": 7, "drug-2": 6, "drug-3": 6, "drug-4": 6
    }


def test_cpu_attention_adapter_preserves_values_and_unblocks_upstream_view():
    base = torch.arange(2 * 4 * 3 * 5, dtype=torch.float32).reshape(2, 4, 3, 5)

    class FakeModelUtils:
        __file__ = __file__

        @staticmethod
        def _attention(query, key, value, dropout_p):
            return base.transpose(1, 2)

    adapter = install_cpu_attention_layout_adapter(FakeModelUtils)
    output = FakeModelUtils._attention(None, None, None, 0.1)

    assert output.is_contiguous()
    assert torch.equal(output, base.transpose(1, 2))
    assert torch.equal(output.view(2, 3, 20), base.transpose(1, 2).reshape(2, 3, 20))
    assert adapter["runtime_counts"] == {"calls": 1, "cpu_outputs_materialized": 1}
