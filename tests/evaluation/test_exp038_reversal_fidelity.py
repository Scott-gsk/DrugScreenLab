"""疾病逆转打分前必须先在细胞与药物内汇总技术重复。"""

import numpy as np
import pandas as pd

from scripts.evaluation.exp038_reversal_fidelity import (
    aggregate_drugs, negative_cosine, top_selection,
)


def test_aggregate_drugs_uses_row_positions_after_panel_filtering() -> None:
    frame = pd.DataFrame({"cell_id": ["B", "A", "A", "B"],
                          "pert_id": ["x", "y", "y", "x"]}, index=[9, 20, 4, 11])
    values = np.array([[2.0, 4.0], [1.0, 3.0], [3.0, 5.0], [4.0, 6.0]], dtype=np.float32)
    groups, averaged = aggregate_drugs(frame, values)
    assert list(zip(groups["cell_id"], groups["pert_id"], groups["n_conditions"])) == [
        ("A", "y", 2), ("B", "x", 2)]
    np.testing.assert_array_equal(averaged, [[2.0, 4.0], [3.0, 5.0]])


def test_negative_cosine_uses_only_mapped_genes_and_leaves_zero_undefined() -> None:
    vectors = np.array([[-1.0, 10.0, 0.0], [1.0, -10.0, 0.0], [0.0, 0.0, 0.0]])
    effect = np.array([1.0, np.nan, 0.0])
    # The helper requires a real signature, so use 20 mapped dimensions.
    padded = np.pad(vectors, ((0, 0), (0, 19)))
    padded_effect = np.r_[effect, np.zeros(19)]
    score = negative_cosine(padded, padded_effect)
    np.testing.assert_allclose(score[:2], [1.0, -1.0])
    assert np.isnan(score[2])


def test_top_decile_ties_use_drug_id_and_loss_uses_measured_scores() -> None:
    drugs = np.array(["b", "a", "c", "d", "e", "f", "g", "h", "i", "j"])
    measured = np.array([9.0, 9.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0])
    predicted = np.array([1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    overlap, loss, count = top_selection(drugs, measured, predicted)
    assert count == 1 and overlap == 0.0 and loss == 4.0
