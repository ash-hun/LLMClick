"""Hybrid decoders that interleave linear-attention and full-attention layers and ship with a vision tower
(Qwen3.5-0.8B, Qwen3.5-2B)."""

from typing import ClassVar

from transformers import AutoModelForImageTextToText

from modeling.llm.models.base import LLMBackbone
from modeling.llm.models import LLM_BACKBONES


@LLM_BACKBONES.register("hybrid")
class HybridBackbone(LLMBackbone):
    loader: ClassVar = AutoModelForImageTextToText
    frozen: ClassVar[tuple[str, ...]] = ("visual", "vision")  # text rows give the vision tower no gradient
    # 18 of 24 Qwen3.5-0.8B layers are linear attention, whose projections are named differently
    adapter_targets: ClassVar[tuple[str, ...]] = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj")
