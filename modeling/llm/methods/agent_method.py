"""Agentic RL: the policy answers over several turns, calling an environment's tools between them, and the whole
trajectory is rewarded by its final answer; GRPO compares the trajectories of one prompt."""

from typing import Any
from dataclasses import dataclass, field

import torch
from pydantic import Field

from modeling.llm.agent.agent_environment import ENVIRONMENTS, Environment
from modeling.llm.methods.llm_method import Encoded, completion, conversation
from modeling.llm.agent.tool_call_parser import parsed
from modeling.llm.models.llm_backbone import LLMBackbone
from modeling.tuning.training_method import Row, chunks
from modeling.llm.methods.grpo_method import EPSILON, GRPO
from core.progress_reporter import Progress
from modeling.modeling_config import Keyed


@dataclass
class Trajectory:
    """One rollout: the conversation so far, every turn as the model saw and wrote it, and how it ended."""
    messages: list[dict[str, Any]]
    turns: list[Encoded] = field(default_factory=list)
    answer: str = ""
    calls: int = 0
    malformed: bool = False
    done: bool = False


class AgentRL(GRPO):
    """Rows: {"prompt": str | messages, ...whatever the reward reads, e.g. "answer"}; the environment brings the
    tools. A turn that calls tools gets their results as tool turns and the policy goes on; a turn without calls is
    the final answer. Every turn trains as its own sequence, the prompt exactly as the model saw it followed by what
    it wrote, with the trajectory's advantage: a template that renders earlier turns differently later on (Qwen3's
    thinking block) cannot shift what was trained."""

    class Config(GRPO.Config):
        environment: Keyed = Field(default_factory=lambda: Keyed(name="calculator"))
        max_turns: int = Field(default=4, ge=1, le=16, description="Turns per trajectory; one without a final answer by then scores as an empty answer")
        max_new_tokens: int = Field(default=128, ge=1, description="Tokens per turn")

    def __init__(self, config: Any, training: Any) -> None:
        super().__init__(config, training)
        self.rollout_metrics: dict[str, float] = {}

    def environment(self) -> Environment:
        environment: Environment = ENVIRONMENTS.get(self.config.environment.name)(self.config.environment.params)
        return environment

    def check_identity(self) -> Any:
        return [super().check_identity(), self.config.environment.model_dump(mode="json")]

    def check(self, row: Row) -> None:
        super().check(row)
        self.environment()  # surfaces a bad environment setting before any training time is spent

    def lengths(self, backbone: LLMBackbone, row: Row) -> list[int]:
        # later turns are as long as the conversation grows; a rollout stops before a turn would pass max_length
        return [len(backbone.render(conversation(row["prompt"]), True, self.environment().tools)) + self.config.max_new_tokens + 1]

    def rollout(self, backbone: LLMBackbone, rows: list[Row], temperature: float) -> list[Trajectory]:
        """One trajectory per row, all advanced a turn at a time in one batch."""
        environment = self.environment()
        tools, cap = environment.tools, self.config.max_new_tokens
        trajectories = [Trajectory(conversation(row["prompt"])) for row in rows]
        for _ in range(self.config.max_turns):
            active = [trajectory for trajectory in trajectories if not trajectory.done]
            prompts = [backbone.render(trajectory.messages, True, tools) for trajectory in active]
            fitting = [(trajectory, prompt) for trajectory, prompt in zip(active, prompts, strict=True)
                       if len(prompt) + cap + 1 <= self.training.max_length]
            for trajectory in active:
                trajectory.done = trajectory.done or all(trajectory is not kept for kept, _ in fitting)
            if not fitting:
                break
            outputs = backbone.generate([prompt for _, prompt in fitting], cap, 1, temperature)
            for (trajectory, prompt), output in zip(fitting, outputs, strict=True):
                tokens = output[0]
                trajectory.turns.append(completion(backbone, prompt, tokens, cap))
                reply = parsed(backbone.text(tokens))
                if reply.malformed or not reply.calls:
                    trajectory.malformed, trajectory.done = reply.malformed, True
                    trajectory.answer = "" if reply.malformed else reply.content
                    continue
                trajectory.calls += len(reply.calls)
                trajectory.messages.append({"role": "assistant", "content": reply.content, "tool_calls": [
                    {"type": "function", "function": {"name": call.name, "arguments": call.arguments}} for call in reply.calls]})
                trajectory.messages += [{"role": "tool", "content": environment.call(call.name, call.arguments)} for call in reply.calls]
        return trajectories

    def sample(self, backbone: LLMBackbone, rows: list[Row]) -> dict[str, Any]:
        """`group_size` trajectories per prompt, their rewards, and each trajectory's advantage on all its turns."""
        size = self.config.group_size
        trajectories = self.rollout(backbone, [row for row in rows for _ in range(size)], self.config.temperature)
        flat = [self.reward(trajectory.answer, row) for row, trajectory in zip([r for r in rows for _ in range(size)], trajectories, strict=True)]
        rewards = torch.tensor(flat, device=backbone.device).view(len(rows), size)
        advantages = rewards - rewards.mean(dim=1, keepdim=True)
        if self.config.scale_rewards:
            advantages = advantages / (rewards.std(dim=1, keepdim=True) + EPSILON)
        per_trajectory = advantages.flatten().tolist()
        encoded = [turn for trajectory in trajectories for turn in trajectory.turns]
        expanded = [value for trajectory, value in zip(trajectories, per_trajectory, strict=True) for _ in trajectory.turns]
        self.rollout_metrics = summary(trajectories)
        return {"encoded": encoded, "advantages": torch.tensor(expanded, device=backbone.device), "reward": float(rewards.mean())}

    def loss(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        loss = super().loss(backbone, rows)
        self.metrics.update(self.rollout_metrics)
        return loss

    @torch.no_grad()
    def evaluate(self, backbone: LLMBackbone, rows: list[Row], batch_size: int, progress: Progress) -> dict[str, float]:
        """Greedy trajectories: mean reward, and how they went (turns, tool calls, malformed calls, finished answers)."""
        rewards: list[float] = []
        trajectories: list[Trajectory] = []
        batches = chunks(rows, batch_size)
        for index, batch in enumerate(batches):
            progress.update(index, len(batches), "rollouts")
            done = self.rollout(backbone, batch, 0.0)
            rewards += [self.reward(trajectory.answer, row) for row, trajectory in zip(batch, done, strict=True)]
            trajectories += done
        progress.update(len(batches), len(batches))
        return {"reward": sum(rewards) / len(rewards), **summary(trajectories)}


def summary(trajectories: list[Trajectory]) -> dict[str, float]:
    count = max(len(trajectories), 1)
    return {"turns": sum(len(t.turns) for t in trajectories) / count, "tool_calls": sum(t.calls for t in trajectories) / count,
            "malformed": sum(t.malformed for t in trajectories) / count,
            "answered": sum(bool(t.answer) for t in trajectories) / count}
