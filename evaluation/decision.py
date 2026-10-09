"""`evaluation_decision`: a decision model on Jev rows or the JevBench public tiers, without and with reasoning:
accuracy, NLL and calibration (ECE), per tier with its chance level, reasoning length, and latency per question."""

import json
import math
import os
import time
from pathlib import Path
from urllib.request import urlopen
from typing import Any, ClassVar, Literal

import torch
from pydantic import Field

from core.config.schema import Section
from core.progress import Progress
from core.stage import Outputs, Stage
from core.utils.device import resolve_device
from core.utils.files import write_json
from evaluation.config import EvaluationData, MeasureSettings, SourceModel
from evaluation.source import SourceExperiment
from evaluation.stages import contamination, named
from modeling.llm.methods.decision.base import DecisionMethod, Item, expected_calibration_error, items
from modeling.llm.models.base import LLMBackbone
from modeling.tuning.method import Row, chunks, fitting
from modeling.tuning.sources import SOURCES
from modeling.tuning.stages import read_rows

ITEMS = "items.jsonl"
SCORES = "scores.json"
REPORT = "report.json"
TIER_FILES = {"easy": "easy.jsonl", "standard": "original.jsonl", "hard": "hard.jsonl"}  # the public tiers
JEVBENCH_REPOSITORY = "fstandhartinger/jevbench"  # JevBench v1: the public tiers under datasets/public, MIT
JEVBENCH_REVISION = "bb05a335bc809e61b20c0f745d25499a82b326fc"  # 2026-09-29: easy 48, standard 72, hard 111 items
JEVBENCH_URL = "https://raw.githubusercontent.com/{repository}/{revision}/datasets/public/{file}"
CACHE = Path(os.environ.get("LLMCLICK_CACHE", "~/.cache/llmclick")).expanduser()
Think = Literal["off", "on", "both"]


def jevbench_row(task: dict[str, Any], tier: str) -> Row:
    """One JevBench item (`id`, `state`, `question`, `labels`, `expected`, `family`) as a Jev record with one question."""
    question, kind = task["question"], task["question"]["type"]
    if kind == "choice":
        criteria: Any = {key: question["criteria"].get(key) for key in task["labels"]}
        label: Any = task["expected"]
    elif kind == "noul":
        criteria, label = question.get("criteria"), task["expected"] == "yes"
    else:
        criteria, label = list(question["criteria"]), int(task["expected"])
    return {"id": task["id"], "tier": tier, "family": task.get("family"), "state": task["state"],
            "questions": {"decision": {"type": kind, "instructions": question["instructions"], "criteria": criteria, "label": label}}}


def fetch_jevbench(revision: str, repository: str = JEVBENCH_REPOSITORY) -> Path:
    """The public tier files of JevBench at `revision`, downloaded once into the cache (`LLMCLICK_CACHE`)."""
    root = CACHE / "jevbench" / revision
    root.mkdir(parents=True, exist_ok=True)
    for file in TIER_FILES.values():
        target = root / file
        if target.is_file():
            continue
        with urlopen(JEVBENCH_URL.format(repository=repository, revision=revision, file=file)) as response:  # noqa: S310 - fixed host
            content = response.read()
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(target)
    return root


@SOURCES.register("jevbench")
def jevbench(params: dict[str, Any]) -> list[Row]:
    """params: path? (a directory with easy.jsonl, original.jsonl, hard.jsonl; without it the public tiers are
    downloaded from the JevBench repository at revision?, default the pinned one), tiers? (subset of easy, standard,
    hard), limit? per tier."""
    root = Path(params["path"]) if params.get("path") else fetch_jevbench(str(params.get("revision", JEVBENCH_REVISION)))
    wanted = params.get("tiers") or list(TIER_FILES)
    rows: list[Row] = []
    for tier in wanted:
        if tier not in TIER_FILES:
            raise ValueError(f"unknown JevBench tier {tier!r}; tiers are {list(TIER_FILES)}")
        lines = [line for line in (root / TIER_FILES[tier]).read_text().split("\n") if line.strip()]
        rows += [jevbench_row(json.loads(line), tier) for line in lines[: params.get("limit")]]
    return rows


class DecisionSettings(Section):
    think: Think = Field(default="both", description="Measure without reasoning (off), with it (on), or both")
    max_think: int | None = Field(default=None, ge=1, description="Reasoning tokens per question; null: the experiment's method.max_think")


class DecisionConfig(MeasureSettings, DecisionSettings):
    model: SourceModel
    data: EvaluationData

    def paths(self) -> list[str]:
        return [*super().paths(), *self.model.paths(), *self.data.paths()]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy, NLL and ECE over every question; per tier the accuracy next to the chance level of its option counts
    and the chance-corrected share; reasoning length; latency per question (p50, p95) at the batch size used."""
    if not records:
        return {}
    correct = [float(r["correct"]) for r in records]
    summary: dict[str, Any] = {
        "questions": float(len(records)), "accuracy": sum(correct) / len(records),
        "nll": sum(-math.log(max(r["p_label"], 1e-12)) for r in records) / len(records),
        "ece": expected_calibration_error([r["confidence"] for r in records], correct),
        "think_tokens": sum(r["think_tokens"] for r in records) / len(records),
        "closed": sum(float(r["closed"]) for r in records) / len(records)}
    seconds = sorted(r["seconds"] for r in records)
    summary["seconds_p50"] = seconds[len(seconds) // 2]
    summary["seconds_p95"] = seconds[min(len(seconds) - 1, int(0.95 * len(seconds)))]
    tiers = sorted({r["tier"] for r in records if r.get("tier")})
    if tiers:
        by_tier: dict[str, dict[str, float]] = {}
        for tier in tiers:
            own = [r for r in records if r.get("tier") == tier]
            accuracy = sum(float(r["correct"]) for r in own) / len(own)
            chance = sum(1 / r["options"] for r in own) / len(own)
            by_tier[tier] = {"questions": float(len(own)), "accuracy": accuracy, "chance": chance,
                             "corrected": max(0.0, (accuracy - chance) / (1 - chance)) if chance < 1 else 0.0}
        summary["by_tier"] = by_tier
        summary["corrected_mean"] = sum(entry["corrected"] for entry in by_tier.values()) / len(by_tier)
    return summary


class DecisionScoreStage(Stage[MeasureSettings]):
    """Every question of the rows through the head, without reasoning and, if asked, after greedy reasoning. Built
    per model by `for_`, so a comparison holds one per run; its scope makes the work shared across recipes."""
    name: ClassVar[str] = "decision"  # instances are named score (and decision:<label> in a comparison)
    scope: ClassVar[str] = "decision"
    sections: ClassVar[tuple[str, ...]] = ("evaluation", "device")
    model: SourceModel
    settings: DecisionSettings
    rows_stage: str = "rows"

    @classmethod
    def for_(cls, config: MeasureSettings, progress: Progress, model: SourceModel, settings: DecisionSettings,
             name: str = "score", rows_stage: str = "rows") -> "DecisionScoreStage":
        stage: DecisionScoreStage = named(cls, name)(config, progress)
        stage.model, stage.settings, stage.rows_stage = model, settings, rows_stage
        return stage

    def identity(self) -> Any:
        return [super().identity(), self.settings.model_dump(mode="json"), SourceExperiment(self.model).identity()]

    def dependencies(self) -> tuple[str, ...]:
        return (self.rows_stage,)

    @torch.no_grad()
    def mode(self, backbone: LLMBackbone, method: DecisionMethod, rows: list[Row], think: bool) -> list[dict[str, Any]]:
        head, batch_size = method.head(backbone), self.config.evaluation.batch_size
        pairs: list[tuple[Row, Item]] = [(row, entry) for row in rows for entry in items(row)]
        batches, records = chunks([{"row": row, "entry": entry} for row, entry in pairs], batch_size), []
        for index, batch in enumerate(batches):
            self.progress.update(index, len(batches), "reasoning then deciding" if think else "deciding")
            group: list[Item] = [wrapped["entry"] for wrapped in batch]
            started = time.perf_counter()
            if think:
                prompts = [head.prompt(backbone, e.state, e.instructions, e.options) for e in group]
                chains = [chain[0] for chain in method.reasoning(backbone, prompts)]
                layouts = [method.layout(backbone, e, chain, closed) for e, (chain, closed) in zip(group, chains, strict=True)]
            else:
                chains = [([], True)] * len(group)
                layouts = [method.layout(backbone, e) for e in group]
            probabilities = head.probabilities(method.scores(backbone, layouts)[0])
            seconds = (time.perf_counter() - started) / len(group)
            for wrapped, (chain, closed), probs in zip(batch, chains, probabilities, strict=True):
                row, entry = wrapped["row"], wrapped["entry"]
                own = probs[: len(entry.options)]
                predicted = int(own.argmax())
                records.append({"id": row.get("id"), "tier": row.get("tier"), "family": row.get("family"), "question": entry.key,
                                "options": len(entry.options), "label": entry.label, "predicted": predicted,
                                "correct": predicted == entry.label, "confidence": float(own.max()),
                                "p_label": float(own[entry.label]), "think_tokens": len(chain), "closed": bool(closed),
                                "seconds": seconds})
        self.progress.update(len(batches), len(batches))
        return records

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        source = SourceExperiment(self.model)
        backbone, method, trained = source.build()
        if not isinstance(method, DecisionMethod) or not isinstance(backbone, LLMBackbone):
            raise ValueError(f"{self.model.experiment!r} is not a decision experiment (its recipe is {trained.recipe!r})")
        think, max_think = self.settings.think, self.settings.max_think
        method.config = method.config.model_copy(update={"eval_think": think != "off", **({"max_think": max_think} if max_think else {})})
        checkpoint = source.checkpoint()
        self.progress.update(0, None, "loading checkpoint" if checkpoint else "loading base model")
        backbone.load(resolve_device(self.config.device), checkpoint)
        modes: dict[str, list[dict[str, Any]]] = {}
        try:
            rows = read_rows(Path(inputs[self.rows_stage]["rows"]))
            kept = fitting(backbone, method, rows, trained.training, self.progress, "evaluation")
            if think != "on":
                modes["nothink"] = self.mode(backbone, method, kept, think=False)
            if think != "off":
                modes["think"] = self.mode(backbone, method, kept, think=True)
            temperature = method.head(backbone).temperature
        finally:
            backbone.release()
        with (workdir / ITEMS).open("w") as stream:
            for mode, records in modes.items():
                for record in records:
                    stream.write(json.dumps({"mode": mode, **record}, ensure_ascii=False) + "\n")
        scores = {"metrics": {mode: summarize(records) for mode, records in modes.items()},
                  "rows": {"scored": len(kept), "skipped": len(rows) - len(kept)},
                  "checkpoint": str(checkpoint) if checkpoint else "base", "recipe": trained.recipe,
                  "settings": {"batch_size": self.config.evaluation.batch_size, "max_length": trained.training.max_length,
                               "max_think": method.config.max_think, "think": think, "temperature": temperature,
                               "device": backbone.device},
                  "items": str(workdir / ITEMS)}
        write_json(workdir / SCORES, scores)
        return scores


class DecisionReport(Stage[DecisionConfig]):
    name: ClassVar[str] = "report"

    def dependencies(self) -> tuple[str, ...]:
        return ("rows", "score")  # the score stage is a `decision` instance named score

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        scores = inputs["score"]
        self.progress.update(0, None, "measuring overlap with the training rows")
        report = {"model": {**SourceExperiment(self.config.model).described(), "checkpoint": scores["checkpoint"], "recipe": scores["recipe"]},
                  "settings": scores["settings"],
                  "rows": {"sources": [source.model_dump(mode="json") for source in self.config.data.sources], **scores["rows"]},
                  "scores": scores["metrics"], "items": scores["items"],
                  "contamination": contamination(self.config.model, read_rows(Path(inputs["rows"]["rows"])))}
        write_json(workdir / REPORT, report)
        return {"report": str(workdir / REPORT), "scores": scores["metrics"]}

