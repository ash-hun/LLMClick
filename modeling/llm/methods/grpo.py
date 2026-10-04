"""Group Relative Policy Optimization: sample a group of completions per prompt, score them with a reward function,
and raise the likelihood of the ones that beat their own group's average."""

import re
from typing import Any

import torch
from pydantic import Field

from modeling.llm.methods.base import Encoded, LLMMethod, conversation
from modeling.tuning.method import Row, chunks
from modeling.llm.models.base import LLMBackbone
from modeling.config import Keyed
from core.registry import Registry
from core.config.schema import Section
from core.progress import Progress

REWARDS = Registry("reward")  # (completion text, row, params) -> float
EPSILON = 1e-4
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


@REWARDS.register("exact_match")
def exact_match(completion: str, row: Row, params: dict[str, Any]) -> float:
    """1 when the completion is the row's `answer` (case and spacing ignored)."""
    return float(normalized(completion) == normalized(str(row["answer"])))


@REWARDS.register("contains")
def contains(completion: str, row: Row, params: dict[str, Any]) -> float:
    """1 when the row's `answer` appears in the completion."""
    return float(normalized(str(row["answer"])) in normalized(completion))


def last_number(text: str) -> float | None:
    found = NUMBER.findall(text)
    return float(found[-1].replace(",", "")) if found else None


@REWARDS.register("last_number")
def last_number_match(completion: str, row: Row, params: dict[str, Any]) -> float:
    """1 when the last number in the completion equals the last number in the row's `answer`; this reads worked
    solutions that end with their result, such as GSM8K's "... #### 18"."""
    expected = last_number(str(row["answer"]))
    if expected is None:
        raise ValueError("answer holds no number")
    return float(last_number(completion) == expected)


class GRPO(LLMMethod):
    """Rows: {"prompt": str | messages, ...whatever the reward reads, e.g. "answer"}."""

    class Config(Section):
        group_size: int = Field(default=4, ge=2, description="Completions sampled per prompt")
        max_new_tokens: int = Field(default=32, ge=1)
        temperature: float = Field(default=1.0, gt=0)
        reward: Keyed = Field(default_factory=lambda: Keyed(name="exact_match"))

    def reward(self, completion: str, row: Row) -> float:
        return float(REWARDS.get(self.config.reward.name)(completion, row, self.config.reward.params))

    def check_identity(self) -> Any:
        return self.config.reward.model_dump(mode="json")  # `check` scores a row with the reward

    def check(self, row: Row) -> None:
        if not isinstance(row["prompt"], (str, list)):
            raise ValueError("prompt must be a string or messages")
        self.reward("", row)  # surfaces a row the reward cannot score before any training time is spent

    def lengths(self, backbone: LLMBackbone, row: Row) -> list[int]:
        return [len(backbone.render(conversation(row["prompt"]), True)) + self.config.max_new_tokens + 1]

    def graded(self, backbone: LLMBackbone, prompt: list[int], tokens: list[int]) -> Encoded:
        """A sampled completion as a graded sequence. The end token is appended and graded only when the model
        produced it: a completion cut off at `max_new_tokens` did not choose to stop, and grading a stop there
        would teach it to stop early whenever a truncated answer happened to be rewarded."""
        ended = [int(backbone.tokenizer.eos_token_id)] if len(tokens) < self.config.max_new_tokens else []
        return prompt + tokens + ended, [False] * len(prompt) + [True] * (len(tokens) + len(ended))

    # ponytail: one policy update per sampled group, so the PPO ratio is 1 and clipping and the KL term do nothing;
    # keep the sampling-time log-probabilities and add both when a group is reused for several updates.
    def loss(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        prompts = [backbone.render(conversation(row["prompt"]), True) for row in rows]
        groups = backbone.generate(prompts, self.config.max_new_tokens, self.config.group_size, self.config.temperature)
        rewards = torch.tensor([[self.reward(backbone.text(tokens), row) for tokens in group]
                                for row, group in zip(rows, groups, strict=True)], device=backbone.device)
        advantages = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + EPSILON)
        encoded = [self.graded(backbone, prompt, tokens) for prompt, group in zip(prompts, groups, strict=True) for tokens in group]
        scores, counts = self.log_probabilities(backbone, encoded, self.config.temperature)
        self.metrics = {"reward": float(rewards.mean())}
        return -(advantages.flatten() * scores / counts.clamp(min=1)).mean()

    @torch.no_grad()
    def evaluate(self, backbone: LLMBackbone, rows: list[Row], batch_size: int, progress: Progress) -> dict[str, float]:
        """Mean reward of the greedy completion."""
        total, batches = 0.0, chunks(rows, batch_size)
        for index, batch in enumerate(batches):
            progress.update(index, len(batches), "reward")
            prompts = [backbone.render(conversation(row["prompt"]), True) for row in batch]
            groups = backbone.generate(prompts, self.config.max_new_tokens)
            total += sum(self.reward(backbone.text(group[0]), row) for row, group in zip(batch, groups, strict=True))
        progress.update(len(batches), len(batches))
        return {"reward": total / len(rows)}
