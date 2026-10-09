"""`evaluation_embedding`: an embedding model on MTEB tasks, one fingerprinted stage per task, through the `mteb` package."""

import importlib.metadata
import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from pydantic import Field

from core.progress import Progress
from core.stage import Outputs, Stage
from core.utils.device import resolve_device
from core.utils.files import write_json
from evaluation.config import MeasureSettings, SourceModel
from evaluation.source import SourceExperiment
from evaluation.stages import named
from modeling.embedding.models.base import EmbeddingBackbone

PACKAGE = "mteb"
RESULTS = "results.json"
REPORT = "report.json"


class EmbeddingConfig(MeasureSettings):
    model: SourceModel
    tasks: list[str] = Field(min_length=1, description="MTEB task names, e.g. STSBenchmark, NFCorpus, Banking77Classification")
    max_length: int | None = Field(default=None, ge=8, description="Tokens per text; null: the experiment's training.max_length")

    def paths(self) -> list[str]:
        return [*super().paths(), *self.model.paths()]


def package_version() -> str | None:
    try:
        return importlib.metadata.version(PACKAGE)
    except importlib.metadata.PackageNotFoundError:
        return None


def mteb_module() -> Any:
    try:
        import mteb
    except ImportError:
        raise ImportError("evaluation_embedding needs mteb: `uv sync --extra eval`") from None
    return mteb


def resolve_task(mteb: Any, name: str) -> Any:
    """An MTEB task by name; the package's mock tasks (no download) are accepted too, for smoke tests."""
    try:
        return mteb.get_task(name)
    except (KeyError, ValueError):
        pass
    from mteb.mocks import MOCK_TASK_REGISTRY
    if name in MOCK_TASK_REGISTRY:
        task = MOCK_TASK_REGISTRY[name]
        return task() if isinstance(task, type) else task
    raise ValueError(f"unknown MTEB task {name!r}; `mteb.get_tasks()` lists them")


class Encoder:
    """What MTEB calls: batches of texts in, unit vectors out, through the experiment's backbone."""

    def __init__(self, backbone: EmbeddingBackbone, max_length: int, meta: Any) -> None:
        self.backbone, self.max_length, self.mteb_model_meta = backbone, max_length, meta

    def encode(self, inputs: Any, *, task_metadata: Any = None, hf_split: str | None = None, hf_subset: str | None = None,
               prompt_type: Any = None, **keys: Any) -> np.ndarray:
        vectors = []
        for batch in inputs:
            texts = [str(text) for text in batch["text"]]
            vectors.append(self.backbone.embed(texts, self.max_length).detach().cpu().numpy())
        return np.concatenate(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)

    def similarity(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        a, b = np.asarray(first, dtype=np.float32), np.asarray(second, dtype=np.float32)
        a = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12, None)
        b = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12, None)
        matrix: np.ndarray = a @ b.T
        return matrix

    def similarity_pairwise(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        a, b = np.asarray(first, dtype=np.float32), np.asarray(second, dtype=np.float32)
        a = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12, None)
        b = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12, None)
        pairs: np.ndarray = (a * b).sum(axis=-1)
        return pairs


def model_meta(mteb: Any, source: SourceExperiment, max_length: int) -> Any:
    from mteb.models.model_meta import ModelMeta, ScoringFunction
    return ModelMeta(loader=None, name=f"llmclick/{Path(source.model.experiment).name}", revision=source.model.checkpoint,
                     release_date=None, languages=None, n_parameters=None, memory_usage_mb=None, max_tokens=max_length,
                     embed_dim=None, license=None, open_weights=True, public_training_code=None, public_training_data=None,
                     framework=["PyTorch"], similarity_fn_name=ScoringFunction.COSINE, use_instructions=False, training_datasets=None)


class EmbeddingStage(Stage[EmbeddingConfig]):
    """One MTEB task on one checkpoint; its scope makes the run shared by every recipe asking for the same."""
    name: ClassVar[str] = "mteb"
    scope: ClassVar[str] = "mteb"
    sections: ClassVar[tuple[str, ...]] = ("evaluation", "device", "max_length")
    task: str

    @classmethod
    def for_(cls, config: EmbeddingConfig, progress: Progress, task: str, name: str | None = None) -> "EmbeddingStage":
        stage: EmbeddingStage = named(cls, name or f"score:{task}")(config, progress)
        stage.task = task
        return stage

    def identity(self) -> Any:
        return [super().identity(), self.task, package_version(), SourceExperiment(self.config.model).identity()]

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        mteb, config = mteb_module(), self.config
        task = resolve_task(mteb, self.task)
        source = SourceExperiment(config.model)
        backbone, _, trained = source.build()
        if not isinstance(backbone, EmbeddingBackbone):
            raise ValueError(f"{config.model.experiment!r} is not an embedding experiment (its recipe is {trained.recipe!r})")
        max_length = config.max_length or trained.training.max_length
        checkpoint = source.checkpoint()
        self.progress.update(0, None, f"{self.task}: loading " + ("checkpoint" if checkpoint else "base model"))
        backbone.load(resolve_device(config.device), checkpoint)
        try:
            self.progress.update(0, None, f"{self.task}: encoding")
            encoder = Encoder(backbone, max_length, model_meta(mteb, source, max_length))
            outcome = mteb.evaluate(encoder, [task], cache=None, show_progress_bar=False,
                                    encode_kwargs={"batch_size": config.evaluation.batch_size})
        finally:
            backbone.release()
        (result,) = outcome.task_results
        dumped = json.loads(json.dumps(result.model_dump(), default=str))
        write_json(workdir / RESULTS, dumped)
        return {"task": self.task, "metric": task.metadata.main_score, "main_score": float(result.get_score()),
                "type": task.metadata.type, "scores": dumped.get("scores"), "results": str(workdir / RESULTS)}


class EmbeddingReport(Stage[EmbeddingConfig]):
    name: ClassVar[str] = "report"
    scores: tuple[str, ...] = ()

    def dependencies(self) -> tuple[str, ...]:
        return self.scores

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        config, source = self.config, SourceExperiment(self.config.model)
        tasks = {inputs[name]["task"]: {key: inputs[name][key] for key in ("metric", "main_score", "type", "scores", "results")}
                 for name in self.scores}
        report = {"model": {"experiment": config.model.experiment, "checkpoint": str(source.checkpoint() or "base"),
                            "recipe": source.config().recipe},
                  "settings": {"batch_size": config.evaluation.batch_size, "max_length": config.max_length,
                               "device": resolve_device(config.device), "package": {"name": PACKAGE, "version": package_version()}},
                  "tasks": tasks, "contamination": None}
        write_json(workdir / REPORT, report)
        return {"report": str(workdir / REPORT), "scores": {task: entry["main_score"] for task, entry in tasks.items()}}
