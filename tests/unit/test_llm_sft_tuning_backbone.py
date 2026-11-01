"""`llm_sft` modeling/tuning/tuning_backbone.py: weights released after the block, QLoRA merge, activation checkpointing."""

from pathlib import Path

import torch
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


def test_qlora_export_merges_the_adapter_into_the_full_precision_base(tiny_model: Path) -> None:
    from modeling.tuning.tuning_config import AdapterConfig
    model = TransformerBackbone(LLMBackboneConfig(architecture="transformer", name=str(tiny_model)))
    model.load("cpu")
    model.adapt(AdapterConfig(r=4, alpha=8, targets=["c_attn"]))
    for name, parameter in model.model.named_parameters():
        if "lora_B" in name:
            parameter.data.fill_(0.01)  # a fresh adapter is zero, which would merge into no change at all
    model.quantize = "nf4"  # the export path a 4-bit base takes, exercised on a full-precision model
    merged = model.exported().state_dict()
    model.quantize = None
    expected = model.exported().state_dict()  # merges in place, so it runs second
    assert set(merged) == set(expected) and not any("lora" in name for name in merged)
    assert all(torch.allclose(merged[name], expected[name], atol=1e-6) for name in expected)


def test_activation_checkpointing_is_switched_on_under_an_adapter(tiny_model: Path) -> None:
    from modeling.tuning.tuning_config import AdapterConfig
    model = TransformerBackbone(LLMBackboneConfig(architecture="transformer", name=str(tiny_model)))
    model.load("cpu")
    assert not model.model.is_gradient_checkpointing
    model.adapt(AdapterConfig(r=4, targets=["c_attn"]))
    model.checkpoint_activations()
    assert model.model.base_model.model.is_gradient_checkpointing
