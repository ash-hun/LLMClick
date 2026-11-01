"""`llm_grpo` modeling/llm/methods/grpo_method.py: the end token graded only when the model stopped."""

from pathlib import Path


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
