"""Group Relative Policy Optimization: sample a group of completions per prompt, score them with a reward function,
and raise the likelihood of the ones that beat their own group's average."""

import re
from typing import Any

import torch
from pydantic import Field

from modeling.llm.methods.base import Encoded, LLMMethod, Reuse, conversation
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
        iterations: int = Field(default=1, ge=1, description="Optimizer steps per sampled group")
        clip: float = Field(default=0.2, gt=0, description="The policy ratio is held within 1 +- clip when a group is reused")
        beta: float = Field(default=0.0, ge=0, description="Weight of the KL penalty towards the base model; needs training.adapter")

    def __init__(self, config: Any, training: Any) -> None:
        super().__init__(config, training)
        self.reuse = Reuse(config.iterations)

    @property
    def repeats(self) -> int:
        return int(self.config.iterations)

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

    def sample(self, backbone: LLMBackbone, rows: list[Row]) -> dict[str, Any]:
        """Completions, their rewards and advantages; fixed for as long as the batch is reused."""
        prompts = [backbone.render(conversation(row["prompt"]), True) for row in rows]
        groups = backbone.generate(prompts, self.config.max_new_tokens, self.config.group_size, self.config.temperature)
        rewards = torch.tensor([[self.reward(backbone.text(tokens), row) for tokens in group]
                                for row, group in zip(rows, groups, strict=True)], device=backbone.device)
        advantages = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + EPSILON)
        encoded = [self.graded(backbone, prompt, tokens) for prompt, group in zip(prompts, groups, strict=True) for tokens in group]
        return {"encoded": encoded, "advantages": advantages.flatten(), "reward": float(rewards.mean())}

    def token_scores(self, backbone: LLMBackbone, encoded: list[Encoded]) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token log-probabilities of the graded tokens under the sampling temperature, and their mask."""
        hidden, ids, mask = backbone.hidden([ids for ids, _ in encoded])
        targets, _ = backbone.padded([[int(flag) for flag in flags] for _, flags in encoded])
        graded = targets * mask
        return backbone.next_token_log_probabilities(hidden, ids, graded, self.config.temperature), graded[:, 1:].float()

    @torch.no_grad()
    def reference(self, backbone: LLMBackbone, encoded: list[Encoded]) -> torch.Tensor:
        """The base model's log-probabilities of the same tokens: the adapter is switched off for one forward pass,
        so no second model is held in memory."""
        if not hasattr(backbone.model, "disable_adapter"):
            raise ValueError("method.beta needs training.adapter: the KL reference is the base model under the adapter")
        training = backbone.model.training
        backbone.model.eval()  # the reference is the base model as it predicts, without dropout noise
        try:
            with backbone.model.disable_adapter():
                return self.token_scores(backbone, encoded)[0]
        finally:
            backbone.model.train(training)

    def loss(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        """Clipped surrogate on the tokens of each completion, plus a KL penalty towards the base model. On a
        group's first use the ratio is 1 and the gradient is the plain policy gradient; from the second use on
        (`iterations` > 1) the clip keeps the policy near the one that produced the samples."""
        config = self.config
        samples = self.reuse.get(rows, lambda: self.sample(backbone, rows))
        new, graded = self.token_scores(backbone, samples["encoded"])
        if "old" not in samples:
            samples["old"] = new.detach()
            samples["reference"] = self.reference(backbone, samples["encoded"]) if config.beta > 0 else None
        ratio = torch.exp(new - samples["old"])
        advantage = samples["advantages"][:, None]
        surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - config.clip, 1 + config.clip) * advantage)
        counts = graded.sum(dim=1).clamp(min=1)
        loss: torch.Tensor = -((surrogate * graded).sum(dim=1) / counts).mean()
        self.metrics = {"reward": samples["reward"], "ratio": float((ratio.detach() * graded).sum() / graded.sum().clamp(min=1))}
        if samples["reference"] is not None:
            gap = samples["reference"] - new  # log(reference / policy) on the sampled tokens
            kl = (((torch.exp(gap) - gap - 1) * graded).sum(dim=1) / counts).mean()
            self.metrics["kl"] = float(kl.detach())
            loss = loss + config.beta * kl
        return loss

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
