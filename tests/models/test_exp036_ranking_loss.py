"""手算检验患者内排序损失的方向和并列响应处理。"""

import pytest
import torch

from scripts.modeling.exp036_pharmaformer_fold import pairwise_ranking_loss


def test_pairwise_ranking_loss_rewards_correct_drug_order() -> None:
    observed = torch.tensor([2.0, 1.0, 0.0])
    correct = torch.tensor([2.0, 1.0, 0.0], requires_grad=True)
    reversed_order = torch.tensor([0.0, 1.0, 2.0])
    assert pairwise_ranking_loss(correct, observed) < pairwise_ranking_loss(reversed_order, observed)
    pairwise_ranking_loss(correct, observed).backward()
    assert correct.grad is not None and torch.isfinite(correct.grad).all()


def test_pairwise_ranking_loss_skips_tied_labels() -> None:
    observed = torch.tensor([1.0, 1.0])
    assert pairwise_ranking_loss(torch.tensor([8.0, -3.0]), observed).item() == 0.0
    with pytest.raises(ValueError, match="aligned vectors"):
        pairwise_ranking_loss(torch.zeros(2, 1), observed)
