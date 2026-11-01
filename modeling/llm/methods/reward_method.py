"""Bradley-Terry reward model: one score per answer, trained so that the chosen answer outscores the rejected one."""

import torch

from modeling.llm.methods.preference_method import PreferenceMethod
from modeling.llm.methods.llm_method import conversation
from modeling.llm.models.llm_backbone import LLMBackbone
from modeling.tuning.training_method import Row


class RewardModel(PreferenceMethod):
    """Rows: {"prompt": str | messages, "chosen": str, "rejected": str}, the DPO pairs. The checkpoint is a sequence
    classifier with one output, the format reward-model servers and other trainers read."""

    def rewards(self, backbone: LLMBackbone, rows: list[Row]) -> tuple[torch.Tensor, torch.Tensor]:
        """The score of each row's chosen answer, then of its rejected one."""
        sequences = [backbone.render([*conversation(row["prompt"]), {"role": "assistant", "content": row[key]}], False)
                     for row in rows for key in ("chosen", "rejected")]
        scores = backbone.sequence_scores(sequences)
        return scores[0::2], scores[1::2]

    def margins(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        chosen, rejected = self.rewards(backbone, rows)
        return chosen - rejected

    def loss(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        margins = self.margins(backbone, rows)
        self.metrics = {"accuracy": float((margins > 0).float().mean()), "margin": float(margins.detach().mean())}
        return -torch.nn.functional.logsigmoid(margins).mean()
