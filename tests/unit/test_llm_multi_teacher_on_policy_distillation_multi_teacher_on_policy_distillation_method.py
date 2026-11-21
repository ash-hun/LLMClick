"""`llm_multi_teacher_on_policy_distillation` modeling/llm/methods/multi_teacher_on_policy_distillation_method.py."""

from typing import Any
from pathlib import Path

import torch
import pytest


def method(tiny_model: Path, **keys: Any) -> Any:
    from modeling.llm.methods.multi_teacher_on_policy_distillation_method import MultiTeacherOnPolicyDistillation
    from modeling.tuning.tuning_config import TrainingConfig

    teacher = {"architecture": "transformer", "name": str(tiny_model)}
    config = MultiTeacherOnPolicyDistillation.Config(teachers={"math": teacher, "chat": teacher}, max_new_tokens=4, **keys)
    return MultiTeacherOnPolicyDistillation(config, TrainingConfig())


def student(tiny_model: Path) -> Any:
    from modeling.llm.models.backbone_config import LLMBackboneConfig
    from modeling.llm.models.transformer_backbone import TransformerBackbone

    backbone = TransformerBackbone(LLMBackboneConfig(architecture="transformer", name=str(tiny_model)))
    backbone.load("cpu")
    return backbone


def test_config_needs_a_teacher_and_a_group_for_the_reward(tiny_model: Path) -> None:
    from modeling.llm.methods.multi_teacher_on_policy_distillation_method import MultiTeacherOnPolicyDistillation

    with pytest.raises(ValueError, match="at least 1"):
        MultiTeacherOnPolicyDistillation.Config(teachers={})
    with pytest.raises(ValueError, match="samples of 2"):
        method(tiny_model, reward={"name": "contains"})
    assert method(tiny_model, reward={"name": "contains"}, samples=2).config.reward_weight == 1.0


def test_check_refuses_a_domain_without_teacher(tiny_model: Path) -> None:
    mopd = method(tiny_model, reward={"name": "contains"}, samples=2)
    mopd.check({"prompt": "What is 1 plus 1?", "domain": "math", "answer": "2"})
    with pytest.raises(ValueError, match="has no teacher"):
        mopd.check({"prompt": "What is 1 plus 1?", "domain": "code", "answer": "2"})
    with pytest.raises(KeyError):
        mopd.check({"prompt": "What is 1 plus 1?", "domain": "math"})


def test_each_answer_is_graded_by_its_domain_teacher(tiny_model: Path) -> None:
    from modeling.llm.methods.llm_method import token_log_probabilities

    backbone, other = student(tiny_model), student(tiny_model)
    with torch.no_grad():
        for parameter in other.model.parameters():
            parameter.add_(torch.randn_like(parameter))
    mopd = method(tiny_model, samples=2)
    rows = [{"prompt": "What is 1 plus 1?", "domain": "math"}, {"prompt": "What is 2 plus 2?", "domain": "chat"}]
    encoded = mopd.answers(backbone, rows, 2, 1.0)
    mine, graded = token_log_probabilities(backbone, encoded)
    scored = mopd.teacher_scores({"math": backbone, "chat": other}, rows, encoded, mine.shape[1])
    gap = ((mine.detach() - scored) * graded).abs().sum(dim=1)
    assert gap[:2].max() < 1e-5 and gap[2:].min() > 1e-2


def test_advantages_compare_answers_to_the_same_prompt(tiny_model: Path) -> None:
    backbone = student(tiny_model)
    mopd = method(tiny_model, reward={"name": "contains"}, samples=2, reward_weight=0.5)
    rows = [{"prompt": "a", "domain": "math", "answer": "2"}, {"prompt": "b", "domain": "chat", "answer": "4"}]
    ids = [backbone.cached((text,), text) for text in ["1 plus 1 is 2", "1 plus 1 is 3", "4", "4"]]
    encoded = [(tokens, [True] * len(tokens)) for tokens in ids]
    advantage = mopd.advantages(backbone, rows, encoded)
    assert advantage[0] > 0 > advantage[1] and advantage[0] == pytest.approx(-advantage[1])
    assert float(advantage[0]) == pytest.approx(0.5 / (2 ** 0.5), rel=1e-3)
    assert advantage[2:].tolist() == [0.0, 0.0]
    assert mopd.metrics["reward"] == 0.75
    assert method(tiny_model, samples=2).advantages(backbone, rows, encoded).tolist() == [0.0] * 4
