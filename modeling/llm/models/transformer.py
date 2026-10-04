"""Plain decoder-only transformers loaded as causal LMs (Qwen3-0.6B, Qwen3-1.7B)."""

from typing import ClassVar

from transformers import AutoModelForCausalLM

from modeling.llm.models.base import QWEN_MARKERS, LLMBackbone, Markers
from modeling.llm.models import LLM_BACKBONES


@LLM_BACKBONES.register("transformer")
class TransformerBackbone(LLMBackbone):
    loader: ClassVar = AutoModelForCausalLM
    markers: ClassVar[Markers | None] = QWEN_MARKERS
