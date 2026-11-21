"""Multi-teacher on-policy distillation: each row's domain picks the teacher that grades the student's own answers."""

from typing import Any

import torch
from pydantic import Field, model_validator

from modeling.llm.methods.on_policy_distillation_method import OnPolicyDistillation, TeacherModel
from modeling.llm.models.llm_backbone import LLMBackbone
from modeling.tuning.training_method import Row, chunks
from modeling.llm.methods.grpo_reward import REWARDS
from modeling.llm.methods.grpo_method import EPSILON
from modeling.llm.methods.llm_method import Encoded
from core.config.config_schema import Section
from core.progress_reporter import Progress
from modeling.modeling_config import Keyed


class MultiTeacherOnPolicyDistillation(OnPolicyDistillation):
    """Rows: {"prompt": str | messages, "domain": str, ...whatever the reward reads}. Every token of a sampled answer is
    graded by the teacher of its row's domain, as in on-policy distillation with one teacher (MiMo-V2-Flash's MOPD).
    With a reward, each answer also gets its group-normalized outcome reward on every token, so the student can
    beat its teachers where the reward knows better."""

    class Config(Section):
        teachers: dict[str, TeacherModel] = Field(min_length=1, description="The teacher of each domain, by the rows' `domain`")
        samples: int = Field(default=1, ge=1, description="Answers sampled per prompt")
        max_new_tokens: int = Field(default=64, ge=1)
        temperature: float = Field(default=1.0, gt=0, description="Sampling temperature of the student's answers")
        reward: Keyed | None = Field(default=None, description="Outcome reward of each answer (a llm_grpo reward); null: teachers only")
        reward_weight: float = Field(default=1.0, ge=0, description="Weight of the normalized reward against the teachers' token gap")

        @model_validator(mode="after")
        def _group(self) -> "MultiTeacherOnPolicyDistillation.Config":
            if self.reward is not None and self.samples < 2:
                raise ValueError("method.reward compares answers to the same prompt; it needs method.samples of 2 or more")
            return self

    def teacher_specs(self) -> dict[str, TeacherModel]:
        return dict(self.config.teachers)

    def domain(self, row: Row) -> str:
        return str(row["domain"])

    def reward(self, completion: str, row: Row) -> float:
        reward = self.config.reward
        assert reward is not None
        return float(REWARDS.get(reward.name)(completion, row, reward.params))

    def check_identity(self) -> Any:
        reward = self.config.reward.model_dump(mode="json") if self.config.reward is not None else None
        return {"domains": sorted(self.config.teachers), "reward": reward}  # `check` reads both

    def check(self, row: Row) -> None:
        super().check(row)
        if row.get("domain") not in self.config.teachers:
            raise ValueError(f"domain {row.get('domain')!r} has no teacher; method.teachers has {sorted(self.config.teachers)}")
        if self.config.reward is not None:
            self.reward("", row)  # surfaces a row the reward cannot score before any training time is spent

    def texts(self, backbone: LLMBackbone, encoded: list[Encoded]) -> list[str]:
        return [backbone.text([token for token, flag in zip(ids, flags, strict=True) if flag]) for ids, flags in encoded]

    def advantages(self, backbone: LLMBackbone, rows: list[Row], encoded: list[Encoded]) -> torch.Tensor:
        """`reward_weight` times each answer's reward against its group: minus the group mean, over the group's
        standard deviation. A group whose answers all score alike gets none."""
        if self.config.reward is None:
            return torch.zeros(len(encoded))
        repeats = len(encoded) // len(rows)
        texts = self.texts(backbone, encoded)
        rewards = torch.tensor([self.reward(text, row) for text, row in zip(texts, [r for r in rows for _ in range(repeats)],
                                                                            strict=True)]).view(len(rows), repeats)
        self.metrics["reward"] = float(rewards.mean())
        normalized = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + EPSILON)
        weighted: torch.Tensor = self.config.reward_weight * normalized.flatten()
        return weighted

    @torch.no_grad()
    def evaluate(self, backbone: LLMBackbone, rows: list[Row], batch_size: int, progress: Progress) -> dict[str, float]:
        """The teachers' measures, per domain too, and with a reward the mean reward of the greedy answers (`reward`)."""
        metrics = super().evaluate(backbone, rows, batch_size, progress)
        if self.config.reward is None or not rows:
            return metrics
        scores = []
        for batch in chunks(rows, batch_size):
            # ponytail: answers again what `super().evaluate` already answered, one more greedy pass over the held-out
            # rows; share the answers when validation sets grow large
            scores += [self.reward(text, row) for text, row in zip(self.texts(backbone, self.answers(backbone, batch, 1, 0.0)),
                                                                   batch, strict=True)]
        return {**metrics, "reward": sum(scores) / len(scores)}
