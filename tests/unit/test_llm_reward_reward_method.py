"""`llm_reward` modeling/llm/methods/reward_method.py and the scorer it trains: last-token scores, the pairwise loss."""

from pathlib import Path

import pytest


def reward_backbone(tiny_model: Path) -> object:
    from modeling.llm.models.backbone_config import LLMBackboneConfig
    from modeling.llm.models.transformer_backbone import TransformerBackbone

    backbone = TransformerBackbone(LLMBackboneConfig(architecture="transformer", name=str(tiny_model)))
    backbone.scorer = True
    backbone.load("cpu")
    return backbone


def test_a_sequence_scores_the_same_alone_and_padded_in_a_batch(tiny_model: Path) -> None:
    backbone = reward_backbone(tiny_model)
    short, long = [5, 6, 7], [5, 6, 7, 8, 9, 10]
    alone = backbone.sequence_scores([short])
    batch = backbone.sequence_scores([short, long])
    assert type(backbone.model).__name__ == "GPT2ForSequenceClassification" and batch.shape == (2,)
    assert float(alone[0]) == pytest.approx(float(batch[0]), abs=1e-5)


def test_loss_falls_as_the_chosen_answer_outscores_the_rejected(tiny_model: Path) -> None:
    import torch
    from modeling.llm.methods.reward_method import RewardModel
    from modeling.tuning.tuning_config import TrainingConfig

    backbone = reward_backbone(tiny_model)
    method = RewardModel(RewardModel.Config(), TrainingConfig())
    rows = [{"prompt": "What is 1 plus 1?", "chosen": "2", "rejected": "3"}]
    optimizer = torch.optim.SGD(backbone.trainable(), lr=0.5)
    losses = []
    for _ in range(5):
        loss = method.loss(backbone, rows)
        losses.append(float(loss))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    assert losses[-1] < losses[0] and method.metrics["accuracy"] == 1.0
    with pytest.raises(ValueError, match="chosen must differ"):
        method.check({"prompt": "p", "chosen": "a", "rejected": "a"})
