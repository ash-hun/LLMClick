"""One config class per LLM recipe: the family's catalog plus the method and its `method:` section."""

from pathlib import Path
from typing import ClassVar

from pydantic import Field, model_validator

from modeling.llm.methods.decision.cispo_method import DecisionCISPO
from modeling.llm.methods.instruction_method import InstructionTuning
from modeling.llm.methods.decision.decision_sft_method import DecisionSFT
from modeling.llm.methods.grpo_reward import REWARDS, reward_names
from modeling.llm.methods.distill_method import Distill
from modeling.llm.methods.reward_method import RewardModel
from modeling.llm.methods.pretrain_method import Pretrain
from modeling.llm.methods.grpo_method import GRPO
from modeling.tuning.tuning_config import TuningConfig
from modeling.llm.models.llm_backbone import LLMBackbone
from modeling.tuning.tuning_backbone import Backbone
from modeling.llm.models.backbone_config import LLMBackboneConfig
from modeling.llm.models.heads import HEADS
from core.recipe_registry import Registry
from modeling.llm.models import LLM_BACKBONES
from modeling.llm.methods.simpo_method import SimPO
from modeling.llm.methods.orpo_method import ORPO
from modeling.llm.methods.dpo_method import DPO
from modeling.llm.methods.kto_method import KTO
from modeling.llm.methods.sft_method import SFT


class LLMConfig(TuningConfig):
    backbones: ClassVar = LLM_BACKBONES
    heads: ClassVar[Registry | None] = None  # set by recipes whose method reads a decision head

    model: LLMBackboneConfig

    @model_validator(mode="after")
    def _head(self) -> "LLMConfig":
        if self.heads is None and self.model.head is not None:
            raise ValueError(f"recipe {self.recipe!r} has no decision head; remove model.head")
        if self.heads is not None and self.model.head not in self.heads:
            raise ValueError(f"model.head is {self.model.head!r}; recipe {self.recipe!r} needs one of {self.heads.names()}")
        return self

    def paths(self) -> list[str]:
        return [*super().paths(), *(self.model.tokenizer.files() if self.model.tokenizer else [])]

    def local_files(self) -> list[str]:
        return [*super().local_files(), *(self.model.tokenizer.files() if self.model.tokenizer else [])]

    def versions(self) -> dict[str, int]:
        head = {"head": self.heads.get(str(self.model.head)).version} if self.heads is not None else {}
        return {**super().versions(), **head}


class SFTConfig(LLMConfig):
    method_class: ClassVar = SFT
    method: SFT.Config = Field(default_factory=SFT.Config)


class PretrainConfig(LLMConfig):
    method_class: ClassVar = Pretrain
    method: Pretrain.Config = Field(default_factory=Pretrain.Config)


class InstructionConfig(LLMConfig):
    method_class: ClassVar = InstructionTuning
    method: InstructionTuning.Config = Field(default_factory=InstructionTuning.Config)


class DPOConfig(LLMConfig):
    method_class: ClassVar = DPO
    method: DPO.Config = Field(default_factory=DPO.Config)


class ORPOConfig(LLMConfig):
    method_class: ClassVar = ORPO
    method: ORPO.Config = Field(default_factory=ORPO.Config)


class SimPOConfig(LLMConfig):
    method_class: ClassVar = SimPO
    method: SimPO.Config = Field(default_factory=SimPO.Config)


class KTOConfig(LLMConfig):
    method_class: ClassVar = KTO
    method: KTO.Config = Field(default_factory=KTO.Config)


class RewardConfig(LLMConfig):
    method_class: ClassVar = RewardModel
    method: RewardModel.Config = Field(default_factory=RewardModel.Config)

    def build_backbone(self) -> Backbone:
        backbone = super().build_backbone()
        assert isinstance(backbone, LLMBackbone)
        backbone.scorer = True
        return backbone


class GRPOConfig(LLMConfig):
    method_class: ClassVar = GRPO
    method: GRPO.Config = Field(default_factory=GRPO.Config)

    def reward_models(self) -> list[str]:
        """The `reward_model` checkpoints the reward reads, also inside a weighted reward."""
        reward = self.method.reward
        entries = [{"name": reward.name, **reward.params}, *(reward.params.get("parts") or [] if reward.name == "weighted" else [])]
        return [str(entry["path"]) for entry in entries if isinstance(entry, dict) and entry.get("name") == "reward_model"
                and isinstance(entry.get("path"), str)]

    def paths(self) -> list[str]:
        return [*super().paths(), *self.reward_models()]

    def local_models(self) -> list[str]:
        return [*super().local_models(), *(path for path in self.reward_models() if Path(path).is_dir())]

    @model_validator(mode="after")
    def _reward_registered(self) -> "GRPOConfig":
        REWARDS.check(*reward_names(self.method.reward.name, self.method.reward.params))
        if self.method.beta > 0 and self.training.adapter is None:
            raise ValueError("method.beta needs training.adapter: the KL reference is the base model under the adapter")
        return self


class DistillConfig(LLMConfig):
    method_class: ClassVar = Distill
    method: Distill.Config

    def paths(self) -> list[str]:
        return [*super().paths(), self.method.teacher.name]

    def local_models(self) -> list[str]:
        return [*super().local_models(), *([self.method.teacher.name] if Path(self.method.teacher.name).is_dir() else [])]


class DecisionConfig(LLMConfig):
    """Decision recipes read the options through a head, so `model.head` is required."""
    heads: ClassVar = HEADS


class DecisionSFTConfig(DecisionConfig):
    method_class: ClassVar = DecisionSFT
    method: DecisionSFT.Config = Field(default_factory=DecisionSFT.Config)


class DecisionCISPOConfig(DecisionConfig):
    method_class: ClassVar = DecisionCISPO
    method: DecisionCISPO.Config = Field(default_factory=DecisionCISPO.Config)
