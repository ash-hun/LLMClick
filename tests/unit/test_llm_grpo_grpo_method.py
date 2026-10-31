"""`llm_grpo` modeling/llm/methods/grpo_method.py: the rewards, and the end token graded only when the model stopped."""

from pathlib import Path

import pytest

from modeling.llm.methods.grpo_method import contains, exact_match, last_number_match


def test_rewards() -> None:
    row = {"answer": "12"}
    assert exact_match(" 12 ", row, {}) == 1.0 and exact_match("12 .", row, {}) == 0.0
    assert contains("It is 12 .", row, {}) == 1.0 and contains("13", row, {}) == 0.0
    worked = {"answer": "She sells 16 - 7 = 9 eggs for 9 * 2 = $18.\n#### 1,018"}
    assert last_number_match("So 9 * 2 gives 1018.", worked, {}) == 1.0
    assert last_number_match("The answer is 1018 eggs, not 9", worked, {}) == 0.0
    assert last_number_match("no digits here", worked, {}) == 0.0
    with pytest.raises(ValueError, match="no number"):
        last_number_match("3", {"answer": "unknown"}, {})


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
