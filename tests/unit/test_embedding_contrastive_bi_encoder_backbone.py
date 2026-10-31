"""`embedding_contrastive` modeling/embedding/models/bi_encoder_backbone.py: last-token pooling handles both padding sides."""

import torch

from modeling.embedding.models.bi_encoder_backbone import BiEncoder
from modeling.tuning.tuning_config import BackboneConfig


def test_last_token_pooling_handles_both_padding_sides() -> None:
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    pooler = BiEncoder(BackboneConfig(architecture="bi_encoder", name=".", revision=None))
    right = pooler.pool(hidden, torch.tensor([[1, 1, 0], [1, 1, 1]]))
    assert right.tolist() == [hidden[0, 1].tolist(), hidden[1, 2].tolist()]
    left = pooler.pool(hidden, torch.tensor([[0, 1, 1], [1, 1, 1]]))
    assert left.tolist() == [hidden[0, 2].tolist(), hidden[1, 2].tolist()]
