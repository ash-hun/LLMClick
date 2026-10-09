"""`evaluation_benchmark`: public benchmarks through lm-evaluation-harness, one fingerprinted stage per benchmark."""

import importlib.metadata
import json
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field, model_validator

from core.config.schema import Section
from core.progress import Progress
from core.stage import Outputs, Stage
from core.utils.device import resolve_device
from core.utils.files import write_json
from evaluation.config import MeasureSettings, SourceModel
from evaluation.source import SourceExperiment
from evaluation.stages import contamination, named
from modeling.llm.models.base import LLMBackbone

HARNESS = "lm_eval"
RESULTS = "results.json"
SAMPLES = "samples.jsonl"
REPORT = "report.json"

# Task names of lm-evaluation-harness. A preset is a starting point; `benchmarks.tasks` adds to it or replaces it.
PRESETS: dict[str, list[dict[str, Any]]] = {
    "small_general": [  # what the small-model cards report, plus the cheap regression panel
        {"name": "mmlu", "num_fewshot": 5}, {"name": "mmlu_pro", "num_fewshot": 5}, {"name": "gsm8k", "num_fewshot": 8},
        {"name": "ifeval"}, {"name": "hellaswag", "num_fewshot": 10}, {"name": "arc_challenge", "num_fewshot": 25},
        {"name": "truthfulqa_mc2"}, {"name": "winogrande", "num_fewshot": 5}],
    "korean": [  # original Korean exams and culture, not translations
        {"name": "kmmlu", "num_fewshot": 5}, {"name": "haerae"}, {"name": "kobest", "num_fewshot": 5},
        {"name": "click"}, {"name": "hrm8k", "num_fewshot": 5}],
}


class Task(Section):
    name: str = Field(description="An lm-evaluation-harness task or group name")
    num_fewshot: int | None = Field(default=None, ge=0, description="null: the task's own default")
    limit: int | None = Field(default=None, ge=1, description="Items per task; null: `benchmarks.limit`, then all")


class Benchmarks(Section):
    preset: str | None = Field(default=None, description=f"One of {sorted(PRESETS)}; `tasks` are added to it")
    tasks: list[Task] = Field(default_factory=list)
    limit: int | None = Field(default=None, ge=1, description="Items per task unless the task says otherwise; null: all (full runs take hours)")

    @model_validator(mode="after")
    def _some(self) -> "Benchmarks":
        if self.preset is not None and self.preset not in PRESETS:
            raise ValueError(f"benchmarks.preset is {self.preset!r}; choose from {sorted(PRESETS)}")
        if not self.resolved():
            raise ValueError("benchmarks needs a preset or at least one task")
        return self

    def resolved(self) -> list[Task]:
        """The preset's tasks and the listed ones, each with its limit filled in; a listed task replaces a preset one."""
        own = {task.name: task for task in self.tasks}
        tasks = [Task(**item) for item in PRESETS.get(self.preset or "", []) if item["name"] not in own] + self.tasks
        return [Task(name=task.name, num_fewshot=task.num_fewshot, limit=task.limit or self.limit) for task in tasks]


class Decoding(Section):
    temperature: float = Field(default=0.0, ge=0, description="0: greedy")
    max_new_tokens: int = Field(default=1024, ge=1)


class HarnessSettings(MeasureSettings):
    """What a benchmark run depends on besides the model and the task."""
    decoding: Decoding = Field(default_factory=Decoding)
    chat_template: bool = Field(default=False, description="Wrap prompts in the tokenizer's chat template (instruction-tuned models)")


class BenchmarkConfig(HarnessSettings):
    model: SourceModel
    benchmarks: Benchmarks

    def paths(self) -> list[str]:
        return [*super().paths(), *self.model.paths()]


def harness_version() -> str | None:
    try:
        return importlib.metadata.version(HARNESS)
    except importlib.metadata.PackageNotFoundError:
        return None


def harness() -> Any:
    try:
        import lm_eval
    except ImportError:
        raise ImportError("evaluation_benchmark needs lm-evaluation-harness: `uv sync --extra eval`") from None
    return lm_eval


def flat(scores: dict[str, Any]) -> dict[str, float]:
    """`{"acc,none": 0.5, "acc_stderr,none": 0.01, "alias": ...}` -> `{"acc": 0.5, "acc_stderr": 0.01}`: the numbers
    only, without the filter suffix; names and "N/A" (a stderr the task cannot give) are left out."""
    return {key.split(",")[0]: float(value) for key, value in scores.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)}


class BenchmarkStage(Stage[HarnessSettings]):
    """One benchmark on one checkpoint. Built per task (and per model, in a comparison) by `for_`; its scope makes the
    run shared by every recipe that asks for the same checkpoint, task and settings."""
    name: ClassVar[str] = "benchmark"  # instances are named score:<task> (and score:<label>:<task> in a comparison)
    scope: ClassVar[str] = "benchmark"
    sections: ClassVar[tuple[str, ...]] = ("decoding", "chat_template", "evaluation", "device", "seed")
    task: Task
    model: SourceModel

    @classmethod
    def for_(cls, config: HarnessSettings, progress: Progress, task: Task, model: SourceModel,
             name: str | None = None) -> "BenchmarkStage":
        stage: BenchmarkStage = named(cls, name or f"score:{task.name}")(config, progress)
        stage.task, stage.model = task, model
        return stage

    def identity(self) -> Any:
        return [super().identity(), self.task.model_dump(mode="json"), harness_version(),
                SourceExperiment(self.model).identity()]

    def model_args(self, source: SourceExperiment) -> dict[str, Any]:
        backbone, _, _ = source.build()
        if not isinstance(backbone, LLMBackbone):
            raise ValueError(f"{source.model.experiment!r} is not a language model; benchmarks need one")
        origin, revision = backbone.origin(source.checkpoint())
        return {"pretrained": origin, **({"revision": revision} if revision else {}), "dtype": "float32"}

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        lm_eval, task, config = harness(), self.task, self.config
        from lm_eval.tasks import TaskManager
        manager = TaskManager()
        if task.name not in manager.all_tasks:
            raise ValueError(f"unknown benchmark {task.name!r}; `lm_eval --tasks list` names the available ones")
        source = SourceExperiment(self.model)
        self.progress.update(0, None, f"{task.name}: loading model")
        decoding = {"max_gen_toks": config.decoding.max_new_tokens, "temperature": config.decoding.temperature,
                    "do_sample": config.decoding.temperature > 0}
        self.progress.update(0, None, f"{task.name}: {task.limit or 'all'} items, {task.num_fewshot if task.num_fewshot is not None else 'default'}-shot")
        output = lm_eval.simple_evaluate(
            model="hf", model_args=self.model_args(source), tasks=[task.name], num_fewshot=task.num_fewshot,
            limit=task.limit, batch_size=config.evaluation.batch_size, device=resolve_device(config.device),
            gen_kwargs=decoding, apply_chat_template=config.chat_template, log_samples=True, task_manager=manager,
            random_seed=config.seed, numpy_random_seed=config.seed, torch_random_seed=config.seed,
            fewshot_random_seed=config.seed)
        kept = {key: output.get(key) for key in ("results", "n-samples", "n-shot", "versions", "higher_is_better")}
        write_json(workdir / RESULTS, json.loads(json.dumps(kept, default=str)))
        with (workdir / SAMPLES).open("w") as stream:
            for name, samples in (output.get("samples") or {}).items():
                for sample in samples:
                    stream.write(json.dumps({"task": name, **sample}, ensure_ascii=False, default=str) + "\n")
        scores = {name: flat(values) for name, values in output["results"].items()}
        return {"task": task.name, "results": str(workdir / RESULTS), "samples": str(workdir / SAMPLES),
                "scores": scores[task.name] if task.name in scores else {}, "subtasks": {k: v for k, v in scores.items() if k != task.name},
                "n": {name: counts.get("effective") for name, counts in (output.get("n-samples") or {}).items()},
                "num_fewshot": (output.get("n-shot") or {}).get(task.name, task.num_fewshot), "limit": task.limit}


def documents(samples: Path) -> list[Any]:
    """The benchmark items a score stage saw, as the harness logged them (`doc` of every sample)."""
    return [json.loads(line).get("doc") for line in samples.read_text().split("\n") if line.strip()]


class BenchmarkReport(Stage[BenchmarkConfig]):
    """One report over every benchmark stage, in the shape every evaluation recipe writes."""
    name: ClassVar[str] = "report"
    scores: tuple[str, ...] = ()

    def dependencies(self) -> tuple[str, ...]:
        return self.scores

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        config = self.config
        source = SourceExperiment(config.model)
        benchmarks = {inputs[name]["task"]: {key: inputs[name][key] for key in ("scores", "subtasks", "n", "num_fewshot", "limit", "results")}
                      for name in self.scores}
        self.progress.update(0, None, "measuring overlap with the training rows")
        for name in self.scores:
            benchmarks[inputs[name]["task"]]["contamination"] = contamination(config.model, documents(Path(inputs[name]["samples"])))
        report = {"model": {"experiment": config.model.experiment, "checkpoint": str(source.checkpoint() or "base"),
                            "recipe": source.config().recipe},
                  "settings": {"batch_size": config.evaluation.batch_size, "decoding": config.decoding.model_dump(mode="json"),
                               "chat_template": config.chat_template, "device": resolve_device(config.device),
                               "harness": {"name": HARNESS, "version": harness_version()}},
                  "benchmarks": benchmarks, "contamination": None}
        write_json(workdir / REPORT, report)
        return {"report": str(workdir / REPORT), "scores": {task: entry["scores"] for task, entry in benchmarks.items()}}
