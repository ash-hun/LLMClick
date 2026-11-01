"""Odds Ratio Preference Optimization: SFT on `chosen` plus a term that raises its odds over `rejected`, no reference model."""

import torch
from pydantic import Field

from modeling.llm.methods.preference_method import PreferenceMethod
from modeling.llm.models.llm_backbone import LLMBackbone
from modeling.tuning.training_method import Row
from core.config.config_schema import Section


def log_odds(mean_log_probability: torch.Tensor) -> torch.Tensor:
    """log(p / (1 - p)) of a per-token mean log-probability; p < 1 always, the clamp keeps log1p finite at p near 1."""
    return mean_log_probability - torch.log1p(-torch.exp(mean_log_probability.clamp(max=-1e-6)))


class ORPO(PreferenceMethod):
    """Rows: {"prompt": str | messages, "chosen": str, "rejected": str}."""

    class Config(Section):
        lam: float = Field(default=0.1, gt=0, description="Weight of the odds-ratio term next to the chosen answer's SFT loss")

    def odds(self, backbone: LLMBackbone, rows: list[Row]) -> tuple[torch.Tensor, torch.Tensor]:
        """Log-odds margin of chosen over rejected, and the chosen answers' mean negative log-likelihood per token."""
        chosen, chosen_count, rejected, rejected_count = self.answers(backbone, rows)
        chosen_mean, rejected_mean = chosen / chosen_count.clamp(min=1), rejected / rejected_count.clamp(min=1)
        return log_odds(chosen_mean) - log_odds(rejected_mean), -chosen.sum() / chosen_count.sum().clamp(min=1)

    def margins(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        return self.odds(backbone, rows)[0]

    def loss(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        margins, nll = self.odds(backbone, rows)
        ratio = -torch.nn.functional.logsigmoid(margins).mean()
        self.metrics = {"accuracy": float((margins > 0).float().mean()), "nll": float(nll.detach()),
                        "odds_ratio": float(ratio.detach())}
        loss: torch.Tensor = nll + self.config.lam * ratio
        return loss
