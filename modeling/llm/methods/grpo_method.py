"""Group Relative Policy Optimization: sample a group of completions per prompt, score them with a reward function,
and raise the likelihood of the ones that beat their own group's average."""

from typing import Any, ClassVar, Literal
from concurrent.futures import ThreadPoolExecutor

import torch
from pydantic import Field, model_validator

from modeling.llm.methods.llm_method import Encoded, LLMMethod, Reuse, completion, conversation, token_log_probabilities
from modeling.llm.models.llm_backbone import LLMBackbone
from modeling.tuning.training_method import Row, chunks
from modeling.llm.methods.grpo_reward import REWARDS
from core.config.config_schema import Section
from core.progress_reporter import Progress
from modeling.modeling_config import Keyed

EPSILON = 1e-4


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
        loss_norm: Literal["sequence", "token"] = Field(default="sequence", description="Average each completion's tokens "
                                                        "first (sequence), or every graded token of the batch at once (token)")
        scale_rewards: bool = Field(default=True, description="Divide each group's advantages by its reward standard deviation")
        reward_workers: int = Field(default=1, ge=1, le=64, description="Threads scoring a batch's completions; for slow rewards such as judge")
        policy_loss: Literal["ppo", "gspo", "cispo"] = Field(default="ppo", description="ppo: clipped per-token ratio; gspo: "
                                                            "one clipped ratio per completion; cispo: capped, gradient-free token weights")
        clip_high: float | None = Field(default=None, gt=0, description="Upper clip (ratio below 1 + clip_high); null: the same as clip")
        added_keys: ClassVar[frozenset[str]] = frozenset({"loss_norm", "scale_rewards", "reward_workers", "policy_loss", "clip_high"})

        @model_validator(mode="after")
        def _sequence_level(self) -> "GRPO.Config":
            if self.policy_loss == "gspo" and self.loss_norm == "token":
                raise ValueError("policy_loss: gspo weighs whole completions; it takes loss_norm: sequence")
            return self

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
        return completion(backbone, prompt, tokens, self.config.max_new_tokens)

    def sample(self, backbone: LLMBackbone, rows: list[Row]) -> dict[str, Any]:
        """Completions, their rewards and advantages; fixed for as long as the batch is reused."""
        prompts = [backbone.render(conversation(row["prompt"]), True) for row in rows]
        groups = backbone.generate(prompts, self.config.max_new_tokens, self.config.group_size, self.config.temperature)
        rewards = torch.tensor(self.scores(backbone, rows, groups), device=backbone.device)
        advantages = rewards - rewards.mean(dim=1, keepdim=True)
        if self.config.scale_rewards:
            advantages = advantages / (rewards.std(dim=1, keepdim=True) + EPSILON)
        encoded = [self.graded(backbone, prompt, tokens) for prompt, group in zip(prompts, groups, strict=True) for tokens in group]
        return {"encoded": encoded, "advantages": advantages.flatten(), "reward": float(rewards.mean())}

    def scores(self, backbone: LLMBackbone, rows: list[Row], groups: list[list[list[int]]]) -> list[list[float]]:
        """Each completion's reward, in order. With `reward_workers` above 1 the calls overlap, which helps a reward
        that waits on a model or an API; the result is the same as scoring them one by one."""
        pairs = [(backbone.text(tokens), row) for row, group in zip(rows, groups, strict=True) for tokens in group]
        if self.config.reward_workers > 1:
            with ThreadPoolExecutor(max_workers=self.config.reward_workers) as pool:
                flat = list(pool.map(lambda pair: self.reward(*pair), pairs))
        else:
            flat = [self.reward(text, row) for text, row in pairs]
        size = self.config.group_size
        return [flat[start:start + size] for start in range(0, len(flat), size)]

    def averaged(self, values: torch.Tensor, graded: torch.Tensor) -> torch.Tensor:
        """Mean of per-token values over the graded tokens: per completion and then over completions (`sequence`),
        or over all graded tokens of the batch (`token`), where a long completion's tokens count as much as a short one's."""
        if self.config.loss_norm == "token":
            return (values * graded).sum() / graded.sum().clamp(min=1)
        return ((values * graded).sum(dim=1) / graded.sum(dim=1).clamp(min=1)).mean()

    def token_scores(self, backbone: LLMBackbone, encoded: list[Encoded]) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token log-probabilities of the graded tokens under the sampling temperature, and their mask."""
        return token_log_probabilities(backbone, encoded, self.config.temperature)

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

    def surrogate(self, new: torch.Tensor, old: torch.Tensor, advantage: torch.Tensor, graded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The policy objective to maximize and the share of clipped tokens (or completions, for gspo). On a group's first
        use the ratio is 1 and all three give the plain policy gradient; they part only when `iterations` reuses it.
        ppo clips each token's ratio (PPO, DAPO's clip-higher with `clip_high`). gspo takes one ratio per completion,
        the exponent of its mean token log-ratio, and clips that (Qwen3's GSPO; it wants a far smaller clip, around
        3e-4). cispo caps each token's ratio from above and uses it as a weight without gradient, so a clipped token
        still learns through its own log-probability (MiniMax-M1)."""
        config = self.config
        low, high = 1 - config.clip, 1 + (config.clip_high if config.clip_high is not None else config.clip)
        if config.policy_loss == "gspo":
            ratio = torch.exp(((new - old) * graded).sum(dim=1) / graded.sum(dim=1).clamp(min=1))
            flat = advantage[:, 0]
            objective = torch.minimum(ratio * flat, ratio.clamp(low, high) * flat).mean()
            return objective, ((ratio < low) | (ratio > high)).float().mean()
        ratio = torch.exp(new - old)
        clipped = (((ratio < low) | (ratio > high)).float() * graded).sum() / graded.sum().clamp(min=1)
        if config.policy_loss == "cispo":
            weight = ratio.detach().clamp(max=high)
            clipped = ((ratio > high).float() * graded).sum() / graded.sum().clamp(min=1)
            return self.averaged(weight * advantage * new, graded), clipped
        return self.averaged(torch.minimum(ratio * advantage, ratio.clamp(low, high) * advantage), graded), clipped

    def loss(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        """The policy objective on the tokens of each completion, plus a KL penalty towards the base model."""
        config = self.config
        samples = self.reuse.get(rows, lambda: self.sample(backbone, rows))
        new, graded = self.token_scores(backbone, samples["encoded"])
        if "old" not in samples:
            samples["old"] = new.detach()
            samples["reference"] = self.reference(backbone, samples["encoded"]) if config.beta > 0 else None
        objective, clipped = self.surrogate(new, samples["old"], samples["advantages"][:, None], graded)
        loss: torch.Tensor = -objective
        ratio = torch.exp(new.detach() - samples["old"])
        self.metrics = {"reward": samples["reward"], "ratio": float((ratio * graded).sum() / graded.sum().clamp(min=1))}
        if config.iterations > 1:
            self.metrics["clipped"] = float(clipped)
        if samples["reference"] is not None:
            gap = samples["reference"] - new  # log(reference / policy) on the sampled tokens
            kl = self.averaged(torch.exp(gap) - gap - 1, graded)
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
