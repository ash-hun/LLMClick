"""`evaluation_benchmark`: public benchmarks through lm-evaluation-harness, one fingerprinted stage per benchmark."""

import os
import json
import importlib.metadata
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
# Logit elements per likelihood batch on MPS. A Metal matrix product with 2^32 outputs aborts the process (SIGABRT,
# no Python error), and the harness keeps the logits and their log-softmax at once, so 2^32 float32 already runs out
# of a 48 GB Mac's memory; 2^30 is 4 GiB per copy.
MPS_LOGITS = 2**30

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
    chat_template: bool | None = Field(default=None, description="Overrides `chat_template` for this task (multiple choice scored by "
                                                                 "likelihood often reads better without it); null: the config's")

    def fingerprint(self) -> dict[str, Any]:
        """The task as its stage's identity; an unset override is left out, so stages built before it existed keep their key."""
        return {key: value for key, value in self.model_dump(mode="json").items() if key != "chat_template" or value is not None}


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
        return [task.model_copy(update={"limit": task.limit or self.limit}) for task in tasks]


class Decoding(Section):
    temperature: float = Field(default=0.0, ge=0, description="0: greedy")
    max_new_tokens: int = Field(default=1024, ge=1)


THINK_END = "</think>"


class HarnessSettings(MeasureSettings):
    """What a benchmark run depends on besides the model and the task."""
    decoding: Decoding = Field(default_factory=Decoding)
    chat_template: bool = Field(default=False, description="Wrap prompts in the tokenizer's chat template (instruction-tuned models)")
    thinking: bool | None = Field(default=None, description="With chat_template: pass enable_thinking to the template (Qwen3, Qwen3.5); "
                                                            "true also splits answers at </think>. null: the template's default")

    @model_validator(mode="after")
    def _thinking_needs_the_template(self) -> "HarnessSettings":
        if self.thinking is not None and not self.chat_template:
            raise ValueError("thinking is a chat-template argument: set chat_template: true")
        return self


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


def length_groups(lengths: list[int], batch_size: int, rows: int) -> list[tuple[int, list[int]]]:
    """Request positions grouped by the largest batch, up to `batch_size`, whose padded length times the batch stays
    within `rows` logit rows; a request longer than `rows` still runs alone."""
    groups: dict[int, list[int]] = {}
    for position in sorted(range(len(lengths)), key=lambda position: -lengths[position]):
        groups.setdefault(max(1, min(batch_size, rows // max(lengths[position], 1))), []).append(position)
    return list(groups.items())


def mps_model(model_args: dict[str, Any], batch_size: int) -> Any:
    """lm-evaluation-harness's `hf` model whose likelihood batches stay within `MPS_LOGITS`: a long few-shot prompt in
    a full batch asks for more logits than one Metal matrix product may write or the memory holds."""
    import torch
    from lm_eval.models.huggingface import HFLM

    class CappedHFLM(HFLM):  # type: ignore[misc]
        def _model_call(self, *args: Any, **keys: Any) -> Any:
            # MPS keeps freed blocks cached, and batches of ever-changing shape grow that cache until the Mac swaps
            # (50 GB on 5-shot MMLU); emptying it before each batch holds memory near one batch's logits
            torch.mps.empty_cache()
            return super()._model_call(*args, **keys)

        def _loglikelihood_tokens(self, requests: list[Any], disable_tqdm: bool = False, override_bs: int | None = None) -> list[Any]:
            vocab = self.model.get_output_embeddings().weight.shape[0]
            lengths = [len(context) + len(continuation) for _, context, continuation in requests]
            results: list[Any] = [None] * len(requests)
            configured = int(self.batch_size)
            try:
                for size, positions in length_groups(lengths, configured, MPS_LOGITS // vocab):
                    self.batch_size_per_gpu = size
                    scored = super()._loglikelihood_tokens([requests[position] for position in positions], disable_tqdm, override_bs)
                    for position, result in zip(positions, scored, strict=True):
                        results[position] = result
            finally:
                self.batch_size_per_gpu = configured
            return results

    return CappedHFLM.create_from_arg_obj(model_args, {"batch_size": batch_size, "device": "mps"})


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
    sections: ClassVar[tuple[str, ...]] = ("decoding", "chat_template", "thinking", "evaluation", "device", "seed")
    task: Task
    model: SourceModel

    @classmethod
    def for_(cls, config: HarnessSettings, progress: Progress, task: Task, model: SourceModel,
             name: str | None = None) -> "BenchmarkStage":
        stage: BenchmarkStage = named(cls, name or f"score:{task.name}")(config, progress)
        stage.task, stage.model = task, model
        return stage

    def identity(self) -> Any:
        return [super().identity(), self.task.fingerprint(), harness_version(),
                SourceExperiment(self.model).identity()]

    @property
    def chat_template(self) -> bool:
        return self.config.chat_template if self.task.chat_template is None else self.task.chat_template

    def model_args(self, source: SourceExperiment) -> dict[str, Any]:
        if not source.hub and not isinstance(source.build()[0], LLMBackbone):
            raise ValueError(f"{source.model.experiment!r} is not a language model; benchmarks need one")
        origin, revision = source.origin()
        thinking = self.config.thinking if self.chat_template else None
        return {"pretrained": origin, **({"revision": revision} if revision else {}), "dtype": "float32",
                **({"enable_thinking": thinking} if thinking is not None else {}), **({"think_end_token": THINK_END} if thinking else {})}

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        lm_eval, task, config = harness(), self.task, self.config
        from lm_eval.tasks import TaskManager
        manager = TaskManager()
        if task.name not in manager.all_tasks:
            raise ValueError(f"unknown benchmark {task.name!r}; `lm_eval --tasks list` names the available ones")
        source = SourceExperiment(self.model)
        device = resolve_device(config.device)
        if device == "mps":  # transformers 5 copies weights onto the device from worker threads, which segfaults on MPS
            os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
        self.progress.update(0, None, f"{task.name}: loading model")
        decoding = {"max_gen_toks": config.decoding.max_new_tokens, "temperature": config.decoding.temperature,
                    "do_sample": config.decoding.temperature > 0}
        self.progress.update(0, None, f"{task.name}: {task.limit or 'all'} items, {task.num_fewshot if task.num_fewshot is not None else 'default'}-shot")
        model_args = self.model_args(source)
        model = mps_model(model_args, config.evaluation.batch_size) if device == "mps" else "hf"
        output = lm_eval.simple_evaluate(
            model=model, model_args=model_args, tasks=[task.name], num_fewshot=task.num_fewshot,
            limit=task.limit, batch_size=config.evaluation.batch_size, device=device,
            gen_kwargs=decoding, apply_chat_template=self.chat_template, log_samples=True, task_manager=manager,
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
                "num_fewshot": (output.get("n-shot") or {}).get(task.name, task.num_fewshot), "limit": task.limit,
                "chat_template": self.chat_template}


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
        report = {"model": source.described(),
                  "settings": {"batch_size": config.evaluation.batch_size, "decoding": config.decoding.model_dump(mode="json"),
                               "chat_template": config.chat_template, "thinking": config.thinking, "device": resolve_device(config.device),
                               "harness": {"name": HARNESS, "version": harness_version()}},
                  "benchmarks": benchmarks, "contamination": None}
        write_json(workdir / REPORT, report)
        return {"report": str(workdir / REPORT), "scores": {task: entry["scores"] for task, entry in benchmarks.items()}}
