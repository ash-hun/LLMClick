"""Shared by the LLM methods: turning a conversation into token ids with the positions the model is graded on."""

from collections.abc import Callable
from typing import Any

import torch

from modeling.tuning.training_method import Row, TrainingMethod
from modeling.llm.models.llm_backbone import LLMBackbone, Message

Encoded = tuple[list[int], list[bool]]  # token ids, and which of them are targets


def common_prefix(a: list[int], b: list[int]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))


def conversation(prompt: str | list[Message]) -> list[Message]:
    """A bare string is one user turn."""
    return [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)


def completion(backbone: LLMBackbone, prompt: list[int], tokens: list[int], cap: int) -> Encoded:
    """A sampled completion as a graded sequence. The end token is appended and graded only when the model
    produced it: a completion cut off at `cap` new tokens did not choose to stop, and grading a stop there
    would teach it to stop early whenever a truncated answer happened to score well."""
    ended = [int(backbone.tokenizer.eos_token_id)] if len(tokens) < cap else []
    return prompt + tokens + ended, [False] * len(prompt) + [True] * (len(tokens) + len(ended))


def token_log_probabilities(backbone: LLMBackbone, encoded: list[Encoded], temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probabilities of the graded tokens, and their mask; shape (sequences, tokens - 1)."""
    hidden, ids, mask = backbone.hidden([ids for ids, _ in encoded])
    targets, _ = backbone.padded([[int(flag) for flag in flags] for _, flags in encoded])
    graded = targets * mask
    return backbone.next_token_log_probabilities(hidden, ids, graded, temperature), graded[:, 1:].float()


class LLMMethod(TrainingMethod[LLMBackbone]):
    def encode(self, backbone: LLMBackbone, messages: list[Message], last_only: bool = False,
               tools: list[dict[str, Any]] | None = None) -> Encoded:
        """Targets are the assistant turns (or only the last): the tokens between a turn's generation prompt and
        its end, found by rendering the conversation up to each turn. A turn the template renders differently once
        it is no longer the last one gets no targets here; `encode_turns` grades it on its own."""
        return self.encoded_turns(backbone, messages, last_only, tools)[0]

    def encoded_turns(self, backbone: LLMBackbone, messages: list[Message], last_only: bool,
                      tools: list[dict[str, Any]] | None) -> tuple[Encoded, list[int]]:
        """The conversation with its assistant turns marked, and the turns that could not be marked in it."""
        full = backbone.render(messages, False, tools)
        targets, apart = [False] * len(full), []
        turns = [i for i, message in enumerate(messages) if message["role"] == "assistant"]
        for turn in turns[-1:] if last_only else turns:
            start = common_prefix(backbone.render(messages[:turn], True, tools), full)
            end = common_prefix(backbone.render(messages[:turn + 1], False, tools), full)
            if end <= start:
                apart.append(turn)
            targets[start:end] = [True] * max(end - start, 0)
        limit = self.training.max_length  # rows that do not fit never get here: `fitting` stops or skips them first
        return (full[:limit], targets[:limit]), apart

    def encode_turns(self, backbone: LLMBackbone, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> list[Encoded]:
        """Every assistant turn graded once, as the model sees the conversation when it writes that turn. Most
        templates render a turn the same in the middle of a conversation and at its end, and all turns share one
        sequence. Some do not: Qwen3 and Qwen3.5 open an empty thinking block on the final turn only, which is also
        what their generation prompt opens. Such a turn becomes a sequence of its own that ends with it, or the model
        would learn it after one prompt and be asked for it after another."""
        whole, apart = self.encoded_turns(backbone, messages, False, tools)
        return [whole, *(self.encode(backbone, messages[:turn + 1], last_only=True, tools=tools) for turn in apart)]

    def log_probabilities(self, backbone: LLMBackbone, encoded: list[Encoded],
                          temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Per sequence: the summed log-probability of its target tokens, and how many there are."""
        scores, graded = token_log_probabilities(backbone, encoded, temperature)
        return scores.sum(dim=1), graded.sum(dim=1)


class Reuse:
    """What a method sampled for a batch, kept until that batch has been trained on `times` times. The loop presents
    a batch `times` times in a row (`TrainingMethod.repeats`); after a resume the samples are simply drawn again."""

    def __init__(self, times: int) -> None:
        self.times = times
        self.kept: dict[tuple[int, ...], list[Any]] = {}

    def get(self, rows: list[Row], sample: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """The batch's samples: drawn by `sample` on the first visit, the same object on the following ones."""
        key = tuple(id(row) for row in rows)
        if key not in self.kept:
            self.kept[key] = [0, sample()]
        entry = self.kept[key]
        entry[0] += 1
        if entry[0] >= self.times:
            del self.kept[key]
        samples: dict[str, Any] = entry[1]
        return samples
