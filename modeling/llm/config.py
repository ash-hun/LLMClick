"""One config class per LLM recipe: the family's catalog plus the method and its `method:` section."""

from typing import ClassVar

from pydantic import Field, model_validator

from modeling.llm.methods.decision.cispo import DecisionCISPO
from modeling.llm.methods.instruction import InstructionTuning
from modeling.llm.methods.decision.sft import DecisionSFT
from modeling.llm.methods.grpo import GRPO, REWARDS
from modeling.tuning.config import TuningConfig
from modeling.llm.models.heads import HEADS
from modeling.llm.models import LLM_BACKBONES
from modeling.llm.methods.dpo import DPO
from modeling.llm.methods.sft import SFT


class LLMConfig(TuningConfig):
    backbones: ClassVar = LLM_BACKBONES


class SFTConfig(LLMConfig):
    method_class: ClassVar = SFT
    method: SFT.Config = Field(default_factory=SFT.Config)


class InstructionConfig(LLMConfig):
    method_class: ClassVar = InstructionTuning
    method: InstructionTuning.Config = Field(default_factory=InstructionTuning.Config)


class DPOConfig(LLMConfig):
    method_class: ClassVar = DPO
    method: DPO.Config = Field(default_factory=DPO.Config)


class GRPOConfig(LLMConfig):
    method_class: ClassVar = GRPO
    method: GRPO.Config = Field(default_factory=GRPO.Config)

    @model_validator(mode="after")
    def _reward_registered(self) -> "GRPOConfig":
        if self.method.reward.name not in REWARDS:
            raise ValueError(f"Unknown reward {self.method.reward.name!r}; registered: {REWARDS.names()}")
        return self


class DecisionConfig(LLMConfig):
    """Decision recipes read the options through a head, so `model.head` is required."""
    heads: ClassVar = HEADS


class DecisionSFTConfig(DecisionConfig):
    method_class: ClassVar = DecisionSFT
    method: DecisionSFT.Config = Field(default_factory=DecisionSFT.Config)


class DecisionCISPOConfig(DecisionConfig):
    method_class: ClassVar = DecisionCISPO
    method: DecisionCISPO.Config = Field(default_factory=DecisionCISPO.Config)
