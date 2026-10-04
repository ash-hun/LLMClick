"""Shared by the decision methods: the Jev row format, the head's loss, calibrated metrics and temperature fitting."""

import json
import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from pydantic import Field

from modeling.llm.models.heads.base import DecisionHead, Layout
from modeling.tuning.method import Row, chunks
from modeling.llm.models.base import LLMBackbone
from modeling.llm.methods.base import LLMMethod
from core.config.schema import Section
from core.progress import Progress

ECE_BINS = 15
TRUE = {"true", "yes", "1"}


@dataclass
class Item:
    """One question of a row, ready for a head: texts, and the probability mass each option should get."""
    key: str
    state: str
    instructions: str
    options: list[str]
    target: list[float]
    label: int


def text(value: Any) -> str:
    return "" if value is None else value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def described(name: str, description: Any) -> str:
    return name if description in (None, "") else f"{name}: {text(description)}"


def item(key: str, state: Any, question: dict[str, Any], label: Any, target: Any) -> Item:
    kind = question["type"]
    criteria: Any = question.get("criteria")
    if kind == "choice":
        keys = list(criteria)
        options, index = [described(k, v) for k, v in criteria.items()], keys.index(str(label))
    elif kind == "noul":
        keys, criteria = ["false", "true"], criteria or {}
        options = [described("no", criteria.get("false")), described("yes", criteria.get("true"))]
        index = int(label) if isinstance(label, (bool, int)) else int(str(label).lower() in TRUE)
    elif kind == "score":
        keys, options, index = [str(i) for i in range(len(criteria))], [text(c) for c in criteria], int(label)
    else:
        raise ValueError(f"unknown question type {kind!r}; use choice, noul or score")
    if not 0 <= index < len(options) or len(options) < 2:
        raise ValueError(f"label {label!r} does not pick one of {len(options)} options")
    if isinstance(target, dict):  # soft target: probability mass per option key
        mass = [float(target.get(k, 0.0)) for k in keys]
        if sum(mass) <= 0:
            raise ValueError("target puts no mass on any option")
        distribution = [m / sum(mass) for m in mass]
    else:
        distribution = [float(i == index) for i in range(len(options))]
    return Item(key, text(state), text(question.get("instructions")), options, distribution, index)


def items(row: Row) -> list[Item]:
    """Jev records hold several questions about one state; jeff Example rows hold one."""
    if "questions" in row:
        return [item(key, row["state"], question, question["label"], question.get("target"))
                for key, question in row["questions"].items()]
    return [item("question", row["state"], row["question"], row["label"], row.get("target"))]


def held_out(index: int, share: float) -> bool:
    """A stable pseudo-random share of row positions, used to keep calibration rows out of training."""
    return int(hashlib.sha256(f"calibration:{index}".encode()).hexdigest(), 16) % 10_000 < share * 10_000


def expected_calibration_error(confidence: list[float], correct: list[float]) -> float:
    bins: list[list[int]] = [[] for _ in range(ECE_BINS)]
    for i, value in enumerate(confidence):
        bins[min(ECE_BINS - 1, int(value * ECE_BINS))].append(i)
    return sum(len(group) / len(confidence) * abs(sum(confidence[i] for i in group) / len(group)
                                                  - sum(correct[i] for i in group) / len(group)) for group in bins if group)


class DecisionMethod(LLMMethod):
    """Rows: {"state", "questions": {id: {"type", "instructions", "criteria", "label", "target"?}}} (Jev), or
    {"state", "question": {...}, "label", "target"?} (jeff Example)."""

    class Config(Section):
        head_lr: float = Field(default=1e-4, gt=0, description="Learning rate of the head, which starts untrained")
        calibration: float = Field(default=0.1, gt=0, lt=1, description="Share of training rows kept for fitting the temperature")
        max_think: int = Field(default=256, ge=1, description="Reasoning tokens per question before the model must decide")
        eval_think: bool = Field(default=False, description="Also measure accuracy when the model reasons first")

    calibration_rows: list[Row]

    def check(self, row: Row) -> None:
        if not items(row):
            raise ValueError("a row needs at least one question")

    def head(self, backbone: LLMBackbone) -> DecisionHead:
        if backbone.head is None:
            raise ValueError("a decision recipe needs model.head")
        return backbone.head

    def parameter_groups(self, backbone: LLMBackbone) -> list[dict[str, Any]]:
        head = list(self.head(backbone).parameters())
        own = {id(parameter) for parameter in head}
        return [{"params": [p for p in backbone.trainable() if id(p) not in own]},
                {"params": head, "lr": self.config.head_lr}]

    def layout(self, backbone: LLMBackbone, entry: Item, chain: list[int] | None = None, closed: bool = True) -> Layout:
        layout = self.head(backbone).layout(backbone, entry.state, entry.instructions, entry.options, chain, closed)
        if len(layout.ids) > self.training.max_length:
            raise ValueError(f"question {entry.key!r} needs {len(layout.ids)} tokens, over training.max_length="
                             f"{self.training.max_length}; nothing is cut, because a cut prompt loses its options")
        return layout

    def scores(self, backbone: LLMBackbone, layouts: list[Layout]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Head scores per option, plus the hidden states and token ids of the same forward pass."""
        hidden, ids, _ = backbone.hidden([layout.ids for layout in layouts])
        return self.head(backbone).scores(hidden, layouts), hidden, ids

    def cross_entropy(self, scores: torch.Tensor, entries: list[Item]) -> torch.Tensor:
        """Per question; soft targets are supported, so a row may spread mass over several options."""
        targets = scores.new_zeros(scores.shape)
        for row, entry in enumerate(entries):
            targets[row, : len(entry.target)] = torch.tensor(entry.target, device=scores.device)
        return -(targets * torch.log_softmax(scores, dim=-1)).sum(dim=-1)

    def reasoning(self, backbone: LLMBackbone, prompts: list[list[int]], samples: int = 1,
                  temperature: float = 0.0) -> list[list[tuple[list[int], bool]]]:
        """Per prompt, `samples` reasoning chains and whether each closed itself before the token limit."""
        head = self.head(backbone)
        banned = sorted({*backbone.tokenizer.all_special_ids, *backbone.tokenizer.get_added_vocab().values()} - {head.think_end})
        groups = backbone.generate(prompts, self.config.max_think + 1, samples, temperature, stop=head.think_end, suppress=banned)
        return [[(chain[: self.config.max_think], len(chain) <= self.config.max_think) for chain in group] for group in groups]

    @torch.no_grad()
    def collect(self, backbone: LLMBackbone, rows: list[Row], batch_size: int, progress: Progress, think: bool,
                note: str) -> tuple[torch.Tensor, list[Item]]:
        """Raw head scores for every question of `rows`, without reasoning or after greedy reasoning."""
        entries = [entry for row in rows for entry in items(row)]
        batches, out = chunks([{"entry": entry} for entry in entries], batch_size), []
        for index, batch in enumerate(batches):
            progress.update(index, len(batches), note)
            group = [wrapped["entry"] for wrapped in batch]
            if think:
                head = self.head(backbone)
                prompts = [head.prompt(backbone, e.state, e.instructions, e.options) for e in group]
                chains = [chain[0] for chain in self.reasoning(backbone, prompts)]
                layouts = [self.layout(backbone, e, chain, closed) for e, (chain, closed) in zip(group, chains, strict=True)]
            else:
                layouts = [self.layout(backbone, e) for e in group]
            scores = self.scores(backbone, layouts)[0]
            out.append(scores)
        progress.update(len(batches), len(batches))
        widest = max(scores.shape[1] for scores in out)
        padded = [torch.nn.functional.pad(scores, (0, widest - scores.shape[1]), value=-1e9) for scores in out]
        return torch.cat(padded), entries

    def measured(self, backbone: LLMBackbone, scores: torch.Tensor, entries: list[Item]) -> dict[str, float]:
        probabilities = self.head(backbone).probabilities(scores)
        labels = torch.tensor([entry.label for entry in entries], device=scores.device)
        confidence, predicted = probabilities.max(dim=-1)
        correct = (predicted == labels).float()
        nll = -torch.log(probabilities[torch.arange(len(entries)), labels].clamp(min=1e-12))
        return {"accuracy": float(correct.mean()), "nll": float(nll.mean()),
                "ece": expected_calibration_error(confidence.tolist(), correct.tolist())}

    def prepare_calibration(self, rows: list[Row]) -> list[Row]:
        """Keep a share of the training rows aside; the temperature is fitted on rows the weights never saw."""
        self.calibration_rows = [row for index, row in enumerate(rows) if held_out(index, self.config.calibration)]
        kept = [row for index, row in enumerate(rows) if not held_out(index, self.config.calibration)]
        if not self.calibration_rows or not kept:
            raise ValueError(f"{len(rows)} training rows cannot be split into training and calibration rows")
        return kept

    def finish(self, backbone: LLMBackbone) -> dict[str, float]:
        """Fit one temperature on the calibration rows (answers without reasoning) and store it in the head."""
        scores, entries = self.collect(backbone, self.calibration_rows, self.training.batch_size, Progress(), False, "")
        labels = torch.tensor([entry.label for entry in entries], device=scores.device)
        grid = torch.linspace(-1.5, 2.0, 351, device=scores.device).exp()
        losses = torch.stack([torch.nn.functional.cross_entropy(scores / value, labels) for value in grid])
        best = int(losses.argmin())
        self.head(backbone).temperature = float(grid[best])
        return {"temperature": float(grid[best]), "calibration_rows": float(len(self.calibration_rows)),
                "calibration_nll": float(losses[best])}

    def evaluate(self, backbone: LLMBackbone, rows: list[Row], batch_size: int, progress: Progress) -> dict[str, float]:
        scores, entries = self.collect(backbone, rows, batch_size, progress, False, "deciding")
        metrics = {**self.measured(backbone, scores, entries), "questions": float(len(entries)),
                   "temperature": self.head(backbone).temperature}
        if self.config.eval_think:
            scores, entries = self.collect(backbone, rows, batch_size, progress, True, "reasoning")
            metrics["think_accuracy"] = self.measured(backbone, scores, entries)["accuracy"]
        return metrics
