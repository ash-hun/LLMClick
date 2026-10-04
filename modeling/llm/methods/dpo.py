"""Direct Preference Optimization: prefer `chosen` over `rejected` more than the untouched base model does."""

import json
from pathlib import Path

import torch
from pydantic import Field

from modeling.llm.methods.base import LLMMethod, conversation
from modeling.tuning.method import Row, chunks
from modeling.llm.models.base import LLMBackbone
from core.utils.files import write_json
from core.config.schema import Section
from core.progress import Progress

REFERENCE = "reference_margins.json"


class DPO(LLMMethod):
    """Rows: {"prompt": str | messages, "chosen": str, "rejected": str}."""

    class Config(Section):
        beta: float = Field(default=0.1, gt=0, description="How strongly the policy may move away from the base model")

    def check(self, row: Row) -> None:
        if not all(isinstance(row[key], str) and row[key] for key in ("chosen", "rejected")):
            raise ValueError("chosen and rejected must be non-empty strings")
        if row["chosen"] == row["rejected"] or not isinstance(row["prompt"], (str, list)):
            raise ValueError("prompt must be a string or messages, and chosen must differ from rejected")

    def margins(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        """log p(chosen | prompt) - log p(rejected | prompt), one value per row."""
        encoded = [self.encode(backbone, [*conversation(row["prompt"]), {"role": "assistant", "content": row[key]}],
                               last_only=True) for row in rows for key in ("chosen", "rejected")]
        scores, _ = self.log_probabilities(backbone, encoded)
        return scores[0::2] - scores[1::2]

    @torch.no_grad()
    def prepare(self, backbone: LLMBackbone, rows: list[Row], workdir: Path) -> list[Row]:
        """The reference model is the base model, so its margins are computed once here and no second model is kept
        in memory while training. Cached: a resumed run must not recompute them with partly trained weights."""
        cache = workdir / REFERENCE
        if not cache.exists():
            backbone.model.eval()
            values = [float(v) for batch in chunks(rows, self.training.batch_size) for v in self.margins(backbone, batch)]
            write_json(cache, values)
        return [{**row, "_reference": value} for row, value in zip(rows, json.loads(cache.read_text()), strict=True)]

    def loss(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        margins = self.margins(backbone, rows)
        reference = torch.tensor([row["_reference"] for row in rows], device=margins.device, dtype=margins.dtype)
        self.metrics = {"accuracy": float((margins > 0).float().mean())}
        return -torch.nn.functional.logsigmoid(self.config.beta * (margins - reference)).mean()

    @torch.no_grad()
    def evaluate(self, backbone: LLMBackbone, rows: list[Row], batch_size: int, progress: Progress) -> dict[str, float]:
        values: list[float] = []
        batches = chunks(rows, batch_size)
        for index, batch in enumerate(batches):
            progress.update(index, len(batches), "preference")
            values += [float(v) for v in self.margins(backbone, batch)]
        progress.update(len(batches), len(batches))
        return {"accuracy": sum(v > 0 for v in values) / len(values), "margin": sum(values) / len(values)}
