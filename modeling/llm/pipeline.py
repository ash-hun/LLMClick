"""The LLM recipes: one pipeline, one registered recipe per training method."""

from typing import ClassVar

from modeling.llm.config import (DecisionCISPOConfig, DecisionSFTConfig, DPOConfig, GRPOConfig, InstructionConfig,
                                 SFTConfig)
from modeling.tuning.pipeline import TuningPipeline
from modeling.llm.methods.grpo import REWARDS
from core.pipeline import recipe


class LLMPipeline(TuningPipeline):
    """Base of every LLM recipe; a new method is a new subclass with its `kind` and config class."""


@recipe
class SFTPipeline(LLMPipeline):
    kind: ClassVar[str] = "llm_sft"
    config_class = SFTConfig


@recipe
class InstructionPipeline(LLMPipeline):
    kind: ClassVar[str] = "llm_instruction"
    config_class = InstructionConfig


@recipe
class DPOPipeline(LLMPipeline):
    kind: ClassVar[str] = "llm_dpo"
    config_class = DPOConfig


@recipe
class GRPOPipeline(LLMPipeline):
    kind: ClassVar[str] = "llm_grpo"
    config_class = GRPOConfig

    @classmethod
    def catalogue(cls) -> dict[str, list[str]]:
        return {**super().catalogue(), "reward": REWARDS.names()}


@recipe
class DecisionSFTPipeline(LLMPipeline):
    kind: ClassVar[str] = "llm_decision_sft"
    config_class = DecisionSFTConfig


@recipe
class DecisionCISPOPipeline(LLMPipeline):
    kind: ClassVar[str] = "llm_decision_cispo"
    config_class = DecisionCISPOConfig
