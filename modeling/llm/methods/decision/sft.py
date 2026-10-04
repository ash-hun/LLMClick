"""Decision SFT: cross-entropy of the head over the options, optionally with base-model reasoning as context."""

import json
import hashlib
from pathlib import Path

import torch
from pydantic import Field

from modeling.llm.methods.decision.base import DecisionMethod, Item, items
from modeling.llm.models.base import LLMBackbone
from modeling.tuning.method import Row
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
    def annotate(self, backbone: LLMBackbone, rows: list[Row], workdir: Path, progress: Progress) -> list[Row]:
        """Cached like DPO's reference: a resumed run must reuse the chains the first attempt trained on."""
        cache = workdir / CHAINS
        if self.config.think_fraction > 0 and not cache.exists():
            wanted = [index for index in range(len(rows)) if self.with_chain(str(index))]
            chains: dict[str, list[int]] = {}
            batches = [wanted[start:start + self.training.batch_size] for start in range(0, len(wanted), self.training.batch_size)]
            for done, batch in enumerate(batches):
                progress.update(done, len(batches), "sampling reasoning chains")
                prompts = [backbone.think_prompt(plain_question(items(rows[index])[0])) for index in batch]
                for index, group in zip(batch, self.reasoning(backbone, prompts), strict=True):
                    chains[str(index)] = group[0][0]
            write_json(cache, chains)
        stored: dict[str, list[int]] = json.loads(cache.read_text()) if cache.exists() else {}
        return [{**row, "_chain": stored.get(str(index))} for index, row in enumerate(rows)]

    def loss(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        entries = [(items(row)[0], row.get("_chain")) for row in rows]
        layouts = [self.layout(backbone, entry, chain) for entry, chain in entries]
        scores = self.scores(backbone, layouts)[0]
        labels = torch.tensor([entry.label for entry, _ in entries], device=scores.device)
        self.metrics = {"accuracy": float((scores.argmax(dim=-1) == labels).float().mean())}
        return self.cross_entropy(scores, [entry for entry, _ in entries]).mean()
