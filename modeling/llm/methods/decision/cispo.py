"""Decision CISPO (Jeeves, stage 2): the model reasons, then decides; reasoning that leads the head to the right
option more surely than the rest of its group is reinforced."""

from typing import Any

import torch
from pydantic import Field

from modeling.llm.methods.decision.base import DecisionMethod, Item, items
from modeling.llm.methods.base import Reuse
from modeling.llm.models.base import LLMBackbone
from modeling.tuning.method import Row


class DecisionCISPO(DecisionMethod):
    """Per question (one step forwards `batch_size` x `group_size` sequences): sample `group_size` reasoning chains, read the head after each, and reward a chain by the
    probability it gave the right option (minus a penalty for running long). Three losses are added: the policy
    gradient on the chain tokens, the head's cross-entropy after reasoning, and the head's cross-entropy without
    reasoning (the anchor that keeps the one-pass answer working). Start it from a decision-SFT checkpoint with
    `model.init`."""

    class Config(DecisionMethod.Config):
        group_size: int = Field(default=4, ge=2, description="Reasoning chains sampled per question")
        temperature: float = Field(default=1.0, gt=0)
        rollout_weight: float = Field(default=0.5, ge=0, description="Weight of the head loss after reasoning")
        anchor_weight: float = Field(default=0.5, ge=0, description="Weight of the head loss without reasoning")
        length_hinge: int | None = Field(default=None, ge=1, description="Chains longer than this many tokens lose reward")
        length_penalty_cap: float = Field(default=0.1, ge=0, le=1)
        iterations: int = Field(default=1, ge=1, description="Optimizer steps per sampled group of chains")
        clip: float = Field(default=0.2, gt=0, description="The importance weight of a reused chain token is capped at 1 + clip")
        eval_think: bool = True

    def __init__(self, config: Any, training: Any) -> None:
        super().__init__(config, training)
        self.reuse = Reuse(config.iterations)

    @property
    def repeats(self) -> int:
        return int(self.config.iterations)

    def thinks(self) -> bool:
        return True

    def sample(self, backbone: LLMBackbone, entries: list[Item]) -> dict[str, Any]:
        """Reasoning chains per question; fixed for as long as the batch is reused."""
        head = self.head(backbone)
        prompts = [head.prompt(backbone, e.state, e.instructions, e.options) for e in entries]
        return {"groups": self.reasoning(backbone, prompts, self.config.group_size, self.config.temperature)}

    def loss(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        config = self.config
        entries = [entry for row in rows for entry in items(row)]
        samples = self.reuse.get(rows, lambda: self.sample(backbone, entries))
        groups = samples["groups"]
        expanded = [entry for entry in entries for _ in range(config.group_size)]
        layouts = [self.layout(backbone, entry, chain, closed)
                   for entry, group in zip(entries, groups, strict=True) for chain, closed in group]
        scores, hidden, ids = self.scores(backbone, layouts)
        rollout = self.cross_entropy(scores, expanded)

        labels = torch.tensor([entry.label for entry in expanded], device=scores.device)
        with torch.no_grad():
            correct = torch.softmax(scores, dim=-1)[torch.arange(len(expanded)), labels]
            lengths = torch.tensor([layout.reasoning[1] - layout.reasoning[0] for layout in layouts], device=scores.device)
            penalty = torch.zeros_like(correct)
            if config.length_hinge:
                penalty = (torch.relu(lengths - config.length_hinge) / config.length_hinge).clamp(max=config.length_penalty_cap)
            reward = (correct * (1 - penalty)).view(len(entries), config.group_size)
            advantage = (reward - reward.mean(dim=1, keepdim=True)).flatten()

        # log-probability of each chain token under the distribution it was sampled from
        chain = torch.zeros_like(ids)
        for row, layout in enumerate(layouts):
            chain[row, layout.reasoning[0]:layout.reasoning[1]] = 1
        token = backbone.next_token_log_probabilities(hidden, ids, chain, config.temperature)
        if "old" not in samples:
            samples["old"] = token.detach()
        # CISPO: the importance weight is capped from above and carries no gradient, so every token keeps a
        # gradient through its own log-probability. It is 1 on a group's first use.
        weight = torch.exp(token.detach() - samples["old"]).clamp(max=1 + config.clip)
        policy = -(weight * advantage[:, None] * token).sum() / chain[:, 1:].sum().clamp(min=1)

        anchor = self.cross_entropy(self.scores(backbone, [self.layout(backbone, entry) for entry in entries])[0], entries)
        self.metrics = {"reward": float(reward.mean()), "think_accuracy": float((scores.argmax(dim=-1) == labels).float().mean()),
                        "think_tokens": float(lengths.float().mean())}
        total: torch.Tensor = policy + config.rollout_weight * rollout.mean() + config.anchor_weight * anchor.mean()
        return total
