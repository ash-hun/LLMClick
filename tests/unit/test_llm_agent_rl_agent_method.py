"""`llm_agent_rl` modeling/llm/methods/agent_method.py: rollouts call tools between turns, every turn trains with its trajectory's advantage."""

from typing import Any
from pathlib import Path

import pytest

from modeling.llm.models.transformer_backbone import TransformerBackbone
from modeling.llm.models.backbone_config import LLMBackboneConfig
from modeling.llm.methods.agent_method import AgentRL
from modeling.tuning.tuning_config import TrainingConfig

CALL = '<tool_call>{"name": "calculate", "arguments": {"expression": "6*7"}}</tool_call>'


def scripted(monkeypatch: pytest.MonkeyPatch, backbone: TransformerBackbone, replies: dict[int, list[str]]) -> list[list[int]]:
    """Each call to generate answers trajectory i with replies[i][turn]; returns the prompts it was given."""
    seen: list[list[int]] = []
    turn = {"n": 0}

    def generate(prompts: list[list[int]], *args: Any, **keys: Any) -> list[list[list[int]]]:
        seen.extend(prompts)
        texts = [replies[i][min(turn["n"], len(replies[i]) - 1)] for i in sorted(replies)][: len(prompts)]
        turn["n"] += 1
        return [[backbone.token_ids(text)] for text in texts]

    monkeypatch.setattr(backbone, "generate", generate)
    return seen


@pytest.fixture
def agent(bpe_model: Path) -> tuple[TransformerBackbone, AgentRL]:
    backbone = TransformerBackbone(LLMBackboneConfig(architecture="transformer", name=str(bpe_model)))
    backbone.load("cpu")
    method = AgentRL(AgentRL.Config(group_size=2, max_turns=3, reward={"name": "exact_match"}), TrainingConfig(max_length=512))
    return backbone, method


def test_a_rollout_calls_the_tool_and_answers_with_its_result(agent: tuple[TransformerBackbone, AgentRL], monkeypatch: pytest.MonkeyPatch) -> None:
    backbone, method = agent
    scripted(monkeypatch, backbone, {0: [CALL, "42"]})
    trajectory = method.rollout(backbone, [{"prompt": "What is 6 times 7?", "answer": "42"}], 1.0)[0]
    assert (trajectory.answer, trajectory.calls, len(trajectory.turns), trajectory.malformed) == ("42", 1, 2, False)
    assert trajectory.messages[-1] == {"role": "tool", "content": "42"}
    first, second = trajectory.turns
    assert backbone.text([t for t, f in zip(*first, strict=True) if f]).startswith("<tool_call>")
    prompt = backbone.tokenizer.decode([t for t, f in zip(*second, strict=True) if not f])
    assert "<|im_start|>tool\n42" in prompt and backbone.text([t for t, f in zip(*second, strict=True) if f]).startswith("42")


def test_every_turn_carries_its_trajectorys_advantage(agent: tuple[TransformerBackbone, AgentRL], monkeypatch: pytest.MonkeyPatch) -> None:
    backbone, method = agent
    scripted(monkeypatch, backbone, {0: [CALL, "42"], 1: ["41"]})
    samples = method.sample(backbone, [{"prompt": "What is 6 times 7?", "answer": "42"}])
    assert len(samples["encoded"]) == 3 and samples["reward"] == 0.5
    advantages = samples["advantages"].tolist()
    assert advantages[0] == advantages[1] > 0 > advantages[2]
    assert method.rollout_metrics["tool_calls"] == 0.5 and method.rollout_metrics["answered"] == 1.0


def test_a_malformed_call_ends_the_trajectory_with_no_answer(agent: tuple[TransformerBackbone, AgentRL], monkeypatch: pytest.MonkeyPatch) -> None:
    backbone, method = agent
    scripted(monkeypatch, backbone, {0: ["<tool_call>not json</tool_call>"]})
    trajectory = method.rollout(backbone, [{"prompt": "What is 6 times 7?", "answer": "42"}], 1.0)[0]
    assert trajectory.malformed and trajectory.answer == "" and len(trajectory.turns) == 1


def test_a_rollout_stops_before_a_turn_would_pass_max_length(bpe_model: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backbone = TransformerBackbone(LLMBackboneConfig(architecture="transformer", name=str(bpe_model)))
    backbone.load("cpu")
    method = AgentRL(AgentRL.Config(max_turns=5, max_new_tokens=8), TrainingConfig(max_length=len(
        backbone.render([{"role": "user", "content": "What is 6 times 7?"}], True, AgentRL(AgentRL.Config(), TrainingConfig()).environment().tools)) + 9))
    scripted(monkeypatch, backbone, {0: [CALL]})
    trajectory = method.rollout(backbone, [{"prompt": "What is 6 times 7?"}], 1.0)[0]
    assert len(trajectory.turns) == 1 and trajectory.done and trajectory.answer == ""
