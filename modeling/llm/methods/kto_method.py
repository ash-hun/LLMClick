"""Kahneman-Tversky Optimization: single answers labelled desirable or not, no chosen-rejected pairs needed."""

import json
from pathlib import Path
from typing import Any

import torch
from pydantic import Field

from modeling.llm.methods.llm_method import Encoded, LLMMethod, conversation
from modeling.tuning.training_method import Row, chunks
from modeling.llm.models.llm_backbone import LLMBackbone
from core.config.config_schema import Section
from core.progress_reporter import Progress
from core.file_store import write_json

REFERENCE = "reference_logps.json"


class KTO(LLMMethod):
    """Rows: {"prompt": str | messages, "completion": str, "label": bool}."""

    class Config(Section):
        beta: float = Field(default=0.1, gt=0, description="How strongly the policy may move away from the base model")
        desirable_weight: float = Field(default=1.0, gt=0, description="Loss weight of desirable rows; raise it when they are the fewer")
        undesirable_weight: float = Field(default=1.0, gt=0, description="Loss weight of undesirable rows")

    def check(self, row: Row) -> None:
        if not isinstance(row["completion"], str) or not row["completion"]:
            raise ValueError("completion must be a non-empty string")
        if not isinstance(row["label"], bool):
            raise ValueError("label must be true (desirable) or false (undesirable)")
        if not isinstance(row["prompt"], (str, list)):
            raise ValueError("prompt must be a string or messages")

    def lengths(self, backbone: LLMBackbone, row: Row) -> list[int]:
        # the mismatched pair that estimates the KL term is cut at max_length by `encode`; only the row's own answer must fit
        return [len(backbone.render([*conversation(row["prompt"]), {"role": "assistant", "content": row["completion"]}], False))]

    def encoded(self, backbone: LLMBackbone, prompt: Any, completion: str) -> Encoded:
        return self.encode(backbone, [*conversation(prompt), {"role": "assistant", "content": completion}], last_only=True)

    def scores(self, backbone: LLMBackbone, rows: list[Row]) -> tuple[torch.Tensor, torch.Tensor]:
        """Summed log-probability of each row's answer, and of the mismatched answer under the row's prompt."""
        encoded = [pair for row in rows for pair in (self.encoded(backbone, row["prompt"], row["completion"]),
                                                     self.encoded(backbone, row["prompt"], row["_mismatch"]))]
        values, _ = self.log_probabilities(backbone, encoded)
        return values[0::2], values[1::2]

    @torch.no_grad()
    def prepare(self, backbone: LLMBackbone, rows: list[Row], workdir: Path, progress: Progress) -> list[Row]:
        """Each row is paired with the next row's answer (the last with the first): the base model's log-probability
        of both is computed once and cached, as in DPO, so no reference model stays in memory. The pairing follows
        the training rows' order, which the data stage fixes, so a resumed run reads the same pairs back."""
        mismatched = [{**row, "_mismatch": rows[(index + 1) % len(rows)]["completion"]} for index, row in enumerate(rows)]
        cache = workdir / REFERENCE
        if not cache.exists():
            backbone.model.eval()
            values: list[list[float]] = []
            batches = chunks(mismatched, self.training.batch_size)
            for index, batch in enumerate(batches):
                progress.update(index, len(batches), "reference log-probabilities")
                own, other = self.scores(backbone, batch)
                values += [[float(a), float(b)] for a, b in zip(own, other, strict=True)]
            write_json(cache, values)
        reference = json.loads(cache.read_text())
        return [{**row, "_reference": own, "_reference_mismatch": other}
                for row, (own, other) in zip(mismatched, reference, strict=True)]

    def loss(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        own, other = self.scores(backbone, rows)
        reference = torch.tensor([row["_reference"] for row in rows], device=own.device, dtype=own.dtype)
        reference_mismatch = torch.tensor([row["_reference_mismatch"] for row in rows], device=own.device, dtype=own.dtype)
        # the reference point: how far the policy has drifted from the base model on answers it was not trained on,
        # never below zero and without a gradient, as the KTO paper's batch estimate
        drift = (other - reference_mismatch).mean().clamp(min=0).detach()
        ratio = own - reference
        desirable = torch.tensor([bool(row["label"]) for row in rows], device=own.device)
        config = self.config
        losses = torch.where(desirable, config.desirable_weight * (1 - torch.sigmoid(config.beta * (ratio - drift))),
                             config.undesirable_weight * (1 - torch.sigmoid(config.beta * (drift - ratio))))
        self.metrics = {"kl": float(drift), "reward": float((config.beta * ratio.detach()).mean())}
        loss: torch.Tensor = losses.mean()
        return loss

    @torch.no_grad()
    def evaluate(self, backbone: LLMBackbone, rows: list[Row], batch_size: int, progress: Progress) -> dict[str, float]:
        """Mean token log-probability of the desirable answers, of the undesirable ones, and their `gap`; the
        validate stage has no reference model, so these are absolute. A side the held-out rows lack is left out."""
        sums = {True: [0.0, 0.0], False: [0.0, 0.0]}  # label -> [log-probability, tokens]
        batches = chunks(rows, batch_size)
        for index, batch in enumerate(batches):
            progress.update(index, len(batches), "label log-probabilities")
            values, counts = self.log_probabilities(backbone, [self.encoded(backbone, row["prompt"], row["completion"]) for row in batch])
            for row, value, count in zip(batch, values, counts, strict=True):
                sums[bool(row["label"])][0] += float(value)
                sums[bool(row["label"])][1] += float(count)
        progress.update(len(batches), len(batches))
        means = {label: total / tokens for label, (total, tokens) in sums.items() if tokens}
        metrics = {f"{'desirable' if label else 'undesirable'}_logp": mean for label, mean in means.items()}
        if len(means) == 2:
            metrics["gap"] = means[True] - means[False]
        return metrics
