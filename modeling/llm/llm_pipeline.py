"""The LLM recipes: one pipeline, one registered recipe per training method."""

from typing import ClassVar

from modeling.llm.llm_config import (DecisionCISPOConfig, DecisionSFTConfig, DPOConfig, GRPOConfig, InstructionConfig,
                                     KTOConfig, ORPOConfig, SFTConfig, SimPOConfig)
from modeling.tuning.tuning_pipeline import TuningPipeline
from modeling.llm.methods.grpo_reward import REWARDS
from core.pipeline_runner import recipe


class LLMPipeline(TuningPipeline):
    """Base of every LLM recipe; a new method is a new subclass with its `kind` and config class."""

    @classmethod
    def catalogue(cls) -> dict[str, list[str]]:
        heads = getattr(cls.config_class, "heads", None)
        return {**super().catalogue(), **({"head": heads.names()} if heads else {})}


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
class ORPOPipeline(LLMPipeline):
    kind: ClassVar[str] = "llm_orpo"
    config_class = ORPOConfig


@recipe
class SimPOPipeline(LLMPipeline):
    kind: ClassVar[str] = "llm_simpo"
    config_class = SimPOConfig


@recipe
class KTOPipeline(LLMPipeline):
    kind: ClassVar[str] = "llm_kto"
    config_class = KTOConfig


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
