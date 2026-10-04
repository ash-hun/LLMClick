"""Shared by the LLM methods: turning a conversation into token ids with the positions the model is graded on."""

import torch

from modeling.tuning.method import TrainingMethod
from modeling.llm.models.base import LLMBackbone, Message

Encoded = tuple[list[int], list[bool]]  # token ids, and which of them are targets


def common_prefix(a: list[int], b: list[int]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def conversation(prompt: str | list[Message]) -> list[Message]:
    """A bare string is one user turn."""
    return [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)


class LLMMethod(TrainingMethod[LLMBackbone]):
    def encode(self, backbone: LLMBackbone, messages: list[Message], last_only: bool = False) -> Encoded:
        """Targets are the assistant turns (or only the last): the tokens between a turn's generation prompt and
        its end, found by rendering the conversation up to each turn."""
        full = backbone.render(messages, False)
        targets = [False] * len(full)
        turns = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
        for turn in turns[-1:] if last_only else turns:
            start = common_prefix(backbone.render(messages[:turn], True), full)
            end = common_prefix(backbone.render(messages[:turn + 1], False), full)
            targets[start:end] = [True] * max(end - start, 0)
        limit = self.training.max_length  # rows that do not fit never get here: `fitting` stops or skips them first
        return full[:limit], targets[:limit]

    def log_probabilities(self, backbone: LLMBackbone, encoded: list[Encoded]) -> tuple[torch.Tensor, torch.Tensor]:
        """Per sequence: the summed log-probability of its target tokens, and how many there are."""
        hidden, ids, mask = backbone.hidden([ids for ids, _ in encoded])
        targets, _ = backbone.padded([[int(flag) for flag in flags] for _, flags in encoded])
        graded = targets * mask
        scores = backbone.next_token_log_probabilities(hidden, ids, graded)
        return scores.sum(dim=1), graded[:, 1:].sum(dim=1).float()
