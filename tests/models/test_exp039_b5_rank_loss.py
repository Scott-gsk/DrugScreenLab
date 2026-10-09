"""Hand-calculated checks for the preregistered bidirectional gene-rank loss."""

import torch
import torch.nn.functional as F

from scripts.modeling.exp039_b5_fold import bidirectional_ranking_loss


def test_bidirectional_ranking_loss_matches_two_hand_calculated_directions() -> None:
    target = torch.tensor([[2.0, 1.0, -1.0, -3.0]])
    prediction = torch.tensor([[1.5, 0.5, -0.5, -2.0]], requires_grad=True)
    scale = torch.ones(4)
    threshold = torch.full((4,), 0.5)
    loss, counts = bidirectional_ranking_loss(target, prediction, scale, threshold)
    up = -(F.softmax(torch.tensor([2.0, 1.0]), dim=0)
           * F.log_softmax(torch.tensor([1.5, 0.5]), dim=0)).sum()
    down = -(F.softmax(torch.tensor([1.0, 3.0]), dim=0)
             * F.log_softmax(torch.tensor([0.5, 2.0]), dim=0)).sum()
    assert torch.allclose(loss, (up + down) / 2, atol=1e-7)
    assert counts == {"conditions_with_up_direction": 1,
                      "conditions_with_down_direction": 1,
                      "conditions_with_any_direction": 1}
    loss.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_bidirectional_ranking_loss_uses_one_direction_or_skips_empty_conditions() -> None:
    scale = torch.ones(4)
    threshold = torch.full((4,), 0.5)
    target = torch.tensor([[2.0, 1.0, 0.1, 0.0]])
    prediction = torch.tensor([[1.5, 0.5, -0.5, -2.0]])
    loss, counts = bidirectional_ranking_loss(target, prediction, scale, threshold)
    expected = -(F.softmax(torch.tensor([2.0, 1.0]), dim=0)
                 * F.log_softmax(torch.tensor([1.5, 0.5]), dim=0)).sum()
    assert torch.allclose(loss, expected, atol=1e-7)
    assert counts["conditions_with_any_direction"] == 1

    empty, empty_counts = bidirectional_ranking_loss(
        torch.zeros((1, 4)), torch.ones((1, 4)), scale, threshold)
    assert empty.item() == 0.0
    assert empty_counts["conditions_with_any_direction"] == 0
