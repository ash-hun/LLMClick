"""Bi-encoder on a decoder LM: query and document go through the same tower, the last token is the vector
(Qwen3-Embedding-0.6B)."""

import torch

from modeling.embedding.models.base import EmbeddingBackbone
from modeling.embedding.models import EMBEDDING_BACKBONES


@EMBEDDING_BACKBONES.register("bi_encoder")
class BiEncoder(EmbeddingBackbone):
    def pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if bool(mask[:, -1].all()):  # left-padded: the last position is always a real token
            return hidden[:, -1]
        last = mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
