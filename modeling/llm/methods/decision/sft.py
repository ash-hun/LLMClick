"""Decision SFT: cross-entropy of the head over the options, optionally with base-model reasoning as context."""

import json
import hashlib
from pathlib import Path

import torch
from pydantic import Field

from modeling.llm.methods.decision.base import DecisionMethod, Item, items
from modeling.llm.models.base import LLMBackbone
from modeling.tuning.method import Row, chunks
from core.utils.files import write_json
from core.progress import Progress

CHAINS = "reasoning_chains.json"


def plain_question(entry: Item) -> str:
    """The question in ordinary words, for sampling reasoning from a model that has not learnt the markers yet."""
    options = "\n".join(entry.options)
    return f"Context:\n{entry.state}\n\nQuestion: {entry.instructions}\n\nOptions:\n{options}"


class DecisionSFT(DecisionMethod):
    """`think_fraction: 0` trains a one-pass decision model (Jev, jeff). Above 0, that share of the questions gets
    a reasoning chain sampled from the base model before training, so the head learns to decide after reasoning
    as well as without it (Jeeves, stage 1). The chain is context only: its tokens carry no loss here."""

    class Config(DecisionMethod.Config):
        think_fraction: float = Field(default=0.0, ge=0, le=1, description="Share of questions given a reasoning chain")

    def thinks(self) -> bool:
        return super().thinks() or self.config.think_fraction > 0

    def with_chain(self, key: str) -> bool:
        share: float = self.config.think_fraction
        return int(hashlib.sha256(f"think:{key}".encode()).hexdigest(), 16) % 10_000 < share * 10_000

    @torch.no_grad()
    def prepare(self, backbone: LLMBackbone, rows: list[Row], workdir: Path, progress: Progress) -> list[Row]:
        """Cached like DPO's reference: a resumed run must reuse the chains the first attempt trained on."""
        rows = self.prepare_calibration(rows)
        cache = workdir / CHAINS
        if self.config.think_fraction > 0 and not cache.exists():
            wanted = [(f"{index}/{entry.key}", entry) for index, row in enumerate(rows) for entry in items(row)
                      if self.with_chain(f"{index}/{entry.key}")]
            chains: dict[str, list[int]] = {}
            batches = chunks([{"key": key, "entry": entry} for key, entry in wanted], self.training.batch_size)
            for index, batch in enumerate(batches):
                progress.update(index, len(batches), "sampling reasoning chains")
                prompts = [backbone.think_prompt(plain_question(wrapped["entry"])) for wrapped in batch]
                for wrapped, group in zip(batch, self.reasoning(backbone, prompts), strict=True):
                    chains[wrapped["key"]] = group[0][0]
            write_json(cache, chains)
        stored: dict[str, list[int]] = json.loads(cache.read_text()) if cache.exists() else {}
        return [{**row, "_chains": {entry.key: stored[f"{index}/{entry.key}"] for entry in items(row)
                                    if f"{index}/{entry.key}" in stored}} for index, row in enumerate(rows)]

    def loss(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        entries = [(entry, row.get("_chains", {}).get(entry.key)) for row in rows for entry in items(row)]
        layouts = [self.layout(backbone, entry, chain) for entry, chain in entries]
        scores = self.scores(backbone, layouts)[0]
        labels = torch.tensor([entry.label for entry, _ in entries], device=scores.device)
        self.metrics = {"accuracy": float((scores.argmax(dim=-1) == labels).float().mean())}
        return self.cross_entropy(scores, [entry for entry, _ in entries]).mean()
