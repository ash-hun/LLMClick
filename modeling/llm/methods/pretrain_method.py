"""Continued pretraining: next-token loss on every token of plain text, documents packed into full-length blocks."""

import math
from pathlib import Path

import torch

from modeling.llm.methods.llm_method import Encoded, LLMMethod
from modeling.llm.models.llm_backbone import LLMBackbone
from modeling.tuning.training_method import Row, chunks
from core.progress_reporter import Progress


def packed(backbone: LLMBackbone, texts: list[str], length: int) -> list[list[int]]:
    """The texts' tokens one after another, each document closed by the end token, cut into blocks of `length`; the
    last block may be shorter. No token is lost to padding and no document is cut short of its block."""
    stream = [token for text in texts for token in [*backbone.token_ids(text), int(backbone.tokenizer.eos_token_id)]]
    return [stream[start:start + length] for start in range(0, len(stream), length)]


class Pretrain(LLMMethod):
    """Rows: {"text": str}, documents of any length: `prepare` packs the training rows into blocks of
    `training.max_length` tokens, so no row is too long and none is cut off at the limit."""

    def check(self, row: Row) -> None:
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError("text must be a non-empty string")

    def prepare(self, backbone: LLMBackbone, rows: list[Row], workdir: Path, progress: Progress) -> list[Row]:
        # ponytail: every token of the corpus is held in memory as Python ints (tens of bytes each); stream the
        # source and pack on the fly once corpora outgrow the host's memory
        progress.update(0, None, "packing documents")
        return [{"ids": block} for block in packed(backbone, [str(row["text"]) for row in rows], self.training.max_length)]

    def encoded(self, block: list[int]) -> Encoded:
        return block, [True] * len(block)  # every token is a target; the first has no context and is never scored

    def loss(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        scores, counts = self.log_probabilities(backbone, [self.encoded(row["ids"]) for row in rows])
        return -scores.sum() / counts.sum().clamp(min=1)

    @torch.no_grad()
    def evaluate(self, backbone: LLMBackbone, rows: list[Row], batch_size: int, progress: Progress) -> dict[str, float]:
        """Loss and perplexity per token on the held-out documents, packed like the training ones, and the loss per
        character (`nll_per_char`), which compares models whose tokenizers cut the same text differently."""
        texts = [str(row["text"]) for row in rows]
        blocks = packed(backbone, texts, self.training.max_length)
        negative, count, batches = 0.0, 0.0, chunks([{"ids": block} for block in blocks], batch_size)
        for index, batch in enumerate(batches):
            progress.update(index, len(batches), "loss")
            scores, counts = self.log_probabilities(backbone, [self.encoded(row["ids"]) for row in batch])
            negative, count = negative - float(scores.sum()), count + float(counts.sum())
        progress.update(len(batches), len(batches))
        loss = negative / max(count, 1.0)
        return {"loss": loss, "perplexity": math.exp(min(loss, 50.0)), "tokens": count,
                "nll_per_char": negative / max(sum(len(text) for text in texts), 1)}
