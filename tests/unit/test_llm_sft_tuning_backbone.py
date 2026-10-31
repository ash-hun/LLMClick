"""`llm_sft` modeling/tuning/tuning_backbone.py: loaded weights are released after the block, even when it fails."""

from pathlib import Path

import pytest

from modeling.llm.models.transformer_backbone import TransformerBackbone
from modeling.llm.models.backbone_config import LLMBackboneConfig


def test_loaded_weights_are_released_after_the_block_even_when_it_fails(tiny_model: Path) -> None:
    model = TransformerBackbone(LLMBackboneConfig(architecture="transformer", name=str(tiny_model)))
    with model.loaded("cpu"):
        assert model.model is not None and model.tokenizer is not None
    assert model.model is None
    with pytest.raises(RuntimeError, match="measurement failed"):
        with model.loaded("cpu"):
            raise RuntimeError("measurement failed")
    assert model.model is None
