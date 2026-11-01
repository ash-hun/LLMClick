"""Direct Preference Optimization: prefer `chosen` over `rejected` more than the untouched base model does."""

import json
from pathlib import Path

import torch
from pydantic import Field

from modeling.llm.methods.preference_method import PreferenceMethod
from modeling.llm.models.llm_backbone import LLMBackbone
from modeling.tuning.training_method import Row, chunks
from core.config.config_schema import Section
from core.progress_reporter import Progress
from core.file_store import write_json

REFERENCE = "reference_margins.json"


class DPO(PreferenceMethod):
    """Rows: {"prompt": str | messages, "chosen": str, "rejected": str}."""

    class Config(Section):
        beta: float = Field(default=0.1, gt=0, description="How strongly the policy may move away from the base model")

    def margins(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        """log p(chosen | prompt) - log p(rejected | prompt), one value per row."""
        chosen, _, rejected, _ = self.answers(backbone, rows)
        return chosen - rejected

    @torch.no_grad()
    def prepare(self, backbone: LLMBackbone, rows: list[Row], workdir: Path, progress: Progress) -> list[Row]:
        """The reference model is the base model, so its margins are computed once here and no second model is kept
        in memory while training. Cached: a resumed run must not recompute them with partly trained weights."""
        cache = workdir / REFERENCE
        if not cache.exists():
            backbone.model.eval()
            values: list[float] = []
            batches = chunks(rows, self.training.batch_size)
            for index, batch in enumerate(batches):
                progress.update(index, len(batches), "reference margins")
                values += [float(v) for v in self.margins(backbone, batch)]
            write_json(cache, values)
        return [{**row, "_reference": value} for row, value in zip(rows, json.loads(cache.read_text()), strict=True)]

    def loss(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        margins = self.margins(backbone, rows)
        reference = torch.tensor([row["_reference"] for row in rows], device=margins.device, dtype=margins.dtype)
        self.metrics = {"accuracy": float((margins > 0).float().mean())}
        return -torch.nn.functional.logsigmoid(self.config.beta * (margins - reference)).mean()
