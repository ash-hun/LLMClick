"""Decision CISPO (Jeeves, stage 2): the model reasons, then decides; reasoning that leads the head to the right
option more surely than the rest of its group is reinforced."""

from pathlib import Path

import torch
from pydantic import Field

from modeling.llm.methods.decision.base import DecisionMethod, items
from modeling.llm.models.base import LLMBackbone
from modeling.tuning.method import Row


class DecisionCISPO(DecisionMethod):
    """Per question: sample `group_size` reasoning chains, read the head after each, and reward a chain by the
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
        eval_think: bool = True

    def prepare(self, backbone: LLMBackbone, rows: list[Row], workdir: Path) -> list[Row]:
        return self.prepare_calibration(rows)

    # ponytail: chains are sampled from the weights that are then updated once, so CISPO's importance weight
    # exp(logp - old_logp) is exactly 1 and its clip does nothing; keep the sampling-time log-probabilities and
    # clip at 1 + epsilon when rollouts are produced ahead of the update (Jeeves does that with a worker thread).
    def loss(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        config, head = self.config, self.head(backbone)
        entries = [entry for row in rows for entry in items(row)]
        prompts = [head.prompt(backbone, e.state, e.instructions, e.options) for e in entries]
        groups = self.reasoning(backbone, prompts, config.group_size, config.temperature)
        expanded = [entry for entry in entries for _ in range(config.group_size)]
        layouts = [self.layout(backbone, entry, chain, closed)
                   for entry, group in zip(entries, groups, strict=True) for chain, closed in group]
        scores, logits, mask = self.scores(backbone, layouts)
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
        ids, _ = backbone.padded([layout.ids for layout in layouts])
        chain = torch.zeros_like(mask)
        for row, layout in enumerate(layouts):
            chain[row, layout.reasoning[0]:layout.reasoning[1]] = 1
        chain = chain[:, 1:] * mask[:, 1:]
        token = torch.log_softmax(logits[:, :-1].float() / config.temperature, dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        policy = -(advantage[:, None] * token * chain).sum() / chain.sum().clamp(min=1)

        anchor = self.cross_entropy(self.scores(backbone, [self.layout(backbone, entry) for entry in entries])[0], entries)
        self.metrics = {"reward": float(reward.mean()), "think_accuracy": float((scores.argmax(dim=-1) == labels).float().mean()),
                        "think_tokens": float(lengths.float().mean())}
        total: torch.Tensor = policy + config.rollout_weight * rollout.mean() + config.anchor_weight * anchor.mean()
        return total
