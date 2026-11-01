"""`llm_grpo` modeling/llm/methods/grpo_method.py: the end token rule, the two loss averages, parallel rewards, new keys."""

from typing import Any
from pathlib import Path

import pytest


def test_grpo_grades_the_end_token_only_when_the_model_stopped(tiny_model: Path) -> None:
    from modeling.llm.methods.grpo_method import GRPO
    from modeling.llm.models.backbone_config import LLMBackboneConfig
    from modeling.llm.models.transformer_backbone import TransformerBackbone
    from modeling.tuning.tuning_config import TrainingConfig

    backbone = TransformerBackbone(LLMBackboneConfig(architecture="transformer", name=str(tiny_model)))
    backbone.load("cpu")
    method = GRPO(GRPO.Config(max_new_tokens=4), TrainingConfig())
    end = backbone.tokenizer.eos_token_id
    stopped = method.graded(backbone, [1, 2], [7, 8])            # shorter than the cap: the model ended it
    cut_off = method.graded(backbone, [1, 2], [7, 8, 9, 10])     # at the cap: it was cut, it did not stop
    assert stopped == ([1, 2, 7, 8, end], [False, False, True, True, True])
    assert cut_off == ([1, 2, 7, 8, 9, 10], [False, False, True, True, True, True])


def test_token_average_weighs_every_graded_token_alike() -> None:
    import torch
    from modeling.llm.methods.grpo_method import GRPO
    from modeling.tuning.tuning_config import TrainingConfig

    values = torch.tensor([[1.0, 1.0, 1.0, 1.0], [4.0, 0.0, 0.0, 0.0]])
    graded = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
    sequence = GRPO(GRPO.Config(), TrainingConfig())
    token = GRPO(GRPO.Config(loss_norm="token"), TrainingConfig())
    assert float(sequence.averaged(values, graded)) == 2.5
    assert float(token.averaged(values, graded)) == pytest.approx(1.6)
    assert float(token.averaged(values, torch.zeros_like(graded))) == 0.0


def test_parallel_rewards_keep_the_order_and_values(tiny_model: Path, monkeypatch: Any) -> None:
    import time
    from modeling.llm.methods.grpo_method import GRPO
    from modeling.llm.methods.grpo_reward import REWARDS
    from modeling.llm.models.backbone_config import LLMBackboneConfig
    from modeling.llm.models.transformer_backbone import TransformerBackbone
    from modeling.tuning.tuning_config import TrainingConfig

    def slow(completion: str, row: dict[str, Any], params: dict[str, Any]) -> float:
        time.sleep(0.01 * (len(completion) % 3))
        return float(len(completion)) + row["k"]

    monkeypatch.setitem(REWARDS._items, "slow", slow)
    backbone = TransformerBackbone(LLMBackboneConfig(architecture="transformer", name=str(tiny_model)))
    backbone.load("cpu")
    groups = [[[5], [5, 6], [5, 6, 7]], [[7], [7, 7], [6]]]
    rows = [{"k": 0}, {"k": 100}]
    config = {"group_size": 3, "reward": {"name": "slow"}}
    serial = GRPO(GRPO.Config(**config), TrainingConfig()).scores(backbone, rows, groups)
    parallel = GRPO(GRPO.Config(**config, reward_workers=4), TrainingConfig()).scores(backbone, rows, groups)
    assert parallel == serial and len(serial) == 2 and len(serial[0]) == 3 and serial[1][0] >= 100


def test_new_grpo_keys_stay_out_of_the_fingerprint_until_set() -> None:
    from modeling.llm.methods.grpo_method import GRPO

    assert set(GRPO.Config().model_dump()) == {"group_size", "max_new_tokens", "temperature", "reward", "iterations", "clip", "beta"}
    assert GRPO.Config(scale_rewards=False).model_dump()["scale_rewards"] is False
