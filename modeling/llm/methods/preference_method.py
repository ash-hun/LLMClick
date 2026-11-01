"""Shared by the preference recipes (DPO, ORPO, SimPO): the chosen-over-rejected row, its lengths and its measurement."""

import torch

from modeling.llm.methods.llm_method import LLMMethod, conversation
from modeling.llm.models.llm_backbone import LLMBackbone
from modeling.tuning.training_method import Row, chunks
from core.progress_reporter import Progress


class PreferenceMethod(LLMMethod):
    """Rows: {"prompt": str | messages, "chosen": str, "rejected": str}. A subclass defines `margins` and `loss`."""

    def check(self, row: Row) -> None:
        if not all(isinstance(row[key], str) and row[key] for key in ("chosen", "rejected")):
            raise ValueError("chosen and rejected must be non-empty strings")
        if row["chosen"] == row["rejected"] or not isinstance(row["prompt"], (str, list)):
            raise ValueError("prompt must be a string or messages, and chosen must differ from rejected")

    def lengths(self, backbone: LLMBackbone, row: Row) -> list[int]:
        return [len(backbone.render([*conversation(row["prompt"]), {"role": "assistant", "content": row[key]}], False))
                for key in ("chosen", "rejected")]

    def answers(self, backbone: LLMBackbone, rows: list[Row]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Summed log-probability and token count of the chosen answers, then of the rejected ones, one value per row."""
        encoded = [self.encode(backbone, [*conversation(row["prompt"]), {"role": "assistant", "content": row[key]}],
                               last_only=True) for row in rows for key in ("chosen", "rejected")]
        scores, counts = self.log_probabilities(backbone, encoded)
        return scores[0::2], counts[0::2], scores[1::2], counts[1::2]

    def margins(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        """How much more the model prefers chosen than rejected, one value per row; above 0 means it ranks them right."""
        raise NotImplementedError(f"{type(self).__name__} must define margins")

    @torch.no_grad()
    def evaluate(self, backbone: LLMBackbone, rows: list[Row], batch_size: int, progress: Progress) -> dict[str, float]:
        values: list[float] = []
        batches = chunks(rows, batch_size)
        for index, batch in enumerate(batches):
            progress.update(index, len(batches), "preference")
            values += [float(v) for v in self.margins(backbone, batch)]
        progress.update(len(batches), len(batches))
        return {"accuracy": sum(v > 0 for v in values) / len(values), "margin": sum(values) / len(values)}
