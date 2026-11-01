"""Simple Preference Optimization: the chosen answer's mean token log-probability beats the rejected one's by a margin."""

import torch
from pydantic import Field

from modeling.llm.methods.preference_method import PreferenceMethod
from modeling.llm.models.llm_backbone import LLMBackbone
from modeling.tuning.training_method import Row
from core.config.config_schema import Section


class SimPO(PreferenceMethod):
    """Rows: {"prompt": str | messages, "chosen": str, "rejected": str}. The mean per token, not the sum, so a longer
    answer earns nothing by length, and no reference model is needed."""

    class Config(Section):
        beta: float = Field(default=2.0, gt=0, description="Scale of the mean log-probability difference")
        gamma: float = Field(default=1.0, ge=0, description="Target margin the scaled difference must exceed")

    def margins(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        """Mean token log-probability of chosen minus that of rejected, one value per row."""
        chosen, chosen_count, rejected, rejected_count = self.answers(backbone, rows)
        return chosen / chosen_count.clamp(min=1) - rejected / rejected_count.clamp(min=1)

    def loss(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        margins = self.margins(backbone, rows)
        self.metrics = {"accuracy": float((margins > 0).float().mean())}
        return -torch.nn.functional.logsigmoid(self.config.beta * margins - self.config.gamma).mean()
