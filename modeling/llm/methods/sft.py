"""Supervised fine-tuning on conversations: cross-entropy on every assistant turn."""

import math

import torch

from modeling.tuning.method import Row, chunks
from modeling.llm.models.base import LLMBackbone, Message
from modeling.llm.methods.base import LLMMethod
from core.progress import Progress

ROLES = {"system", "user", "assistant"}


class SFT(LLMMethod):
    """Rows: {"messages": [{"role": "system"|"user"|"assistant", "content": str}, ...]} ending with an assistant turn."""

    def messages(self, row: Row) -> list[Message]:
        messages: list[Message] = row["messages"]
        return messages

    def check(self, row: Row) -> None:
        messages = self.messages(row)
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("the conversation must end with an assistant turn")
        if any(message["role"] not in ROLES or not isinstance(message["content"], str) for message in messages):
            raise ValueError(f"every message needs a role in {sorted(ROLES)} and a string content")

    def totals(self, backbone: LLMBackbone, rows: list[Row]) -> tuple[torch.Tensor, torch.Tensor]:
        scores, counts = self.log_probabilities(backbone, [self.encode(backbone, self.messages(row)) for row in rows])
        return -scores.sum(), counts.sum()

    def loss(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        negative, count = self.totals(backbone, rows)
        return negative / count.clamp(min=1)

    @torch.no_grad()
    def evaluate(self, backbone: LLMBackbone, rows: list[Row], batch_size: int, progress: Progress) -> dict[str, float]:
        negative, count, batches = 0.0, 0.0, chunks(rows, batch_size)
        for index, batch in enumerate(batches):
            progress.update(index, len(batches), "loss")
            batch_negative, batch_count = self.totals(backbone, batch)
            negative, count = negative + float(batch_negative), count + float(batch_count)
        progress.update(len(batches), len(batches))
        loss = negative / max(count, 1.0)
        return {"loss": loss, "perplexity": math.exp(min(loss, 50.0)), "tokens": count}
